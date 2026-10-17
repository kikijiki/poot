//! Selected-attention fixtures (POOT-1143 stage 3): attention over the blocks a selection names, on every
//! executor, against the independent f64 attention of `poot_graph_ir::test_support::selected_attention` (tier 2)
//! and, where it is cheap, the oracle's dense definition.

use poot_eval::test_support::StepFixture;
use poot_graph_ir::test_support::block_top_k::{grid_scores, reference_row};
use poot_graph_ir::test_support::selected_attention::{Case, cases};
use poot_graph_ir::{Builder, Outputs, SelectedAttentionAttrs, Slot, TensorType};
use poot_graph_plan::{KernelChoice, Target, production_plan::Submission};
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, store_from_tensors};

/// One case: the op's program over seeded operands, and the f64 reference output it must match.
pub struct SelectedAttentionCase {
    /// The selected-attention program, its constants (`q`, `k`, `v`) and the selection slots.
    pub fixture: Fixture,
    /// The reference output `[batch, hq, lq, d]`, row-major.
    pub expected: Vec<f64>,
    /// Whether the oracle's dense definition is cheap enough to run on `fixture` too.
    pub oracle: bool,
}

/// The cases: the shared selection cases (decode over a GQA pair, a prefill with empty and short selections,
/// every block selected), and `wide`, the design's deep decode: 24 query heads over 2 key/value heads of 256,
/// 32768 stored positions (8192 blocks of four), 512 blocks selected by the plain selection of random scores.
/// The device sums 2052 keys per head where the dense definition would sum 32768.
pub fn all() -> Vec<SelectedAttentionCase> {
    let mut all: Vec<SelectedAttentionCase> = cases()
        .into_iter()
        .zip([
            "selected_attention_decode",
            "selected_attention_prefill_short_selections",
            "selected_attention_every_block",
        ])
        .map(|(case, name)| build(name, &case, true))
        .collect();
    let (ratio, blocks, k) = (4, 8192, 512);
    let pos = vec![(ratio * blocks - 1) as i32];
    let ids = reference_row(&grid_scores(blocks, 1 << 20, 99), pos[0], ratio, k)[..k].to_vec();
    all.push(build(
        "selected_attention_wide",
        &Case {
            label: "wide".into(),
            batch: 1,
            hq: 24,
            n_rep: 12,
            lq: 1,
            lk: ratio * blocks,
            d: 256,
            ratio,
            k,
            pos,
            ids,
        },
        false,
    ));
    all
}

fn build(name: &'static str, case: &Case, oracle: bool) -> SelectedAttentionCase {
    let [q, k, v] = case.operands(21);
    let expected = case.reference(&q, &k, &v);
    build_with(name, case, [q, k, v], expected, oracle)
}

fn build_with(
    name: &'static str,
    case: &Case,
    [q, k, v]: [Vec<f32>; 3],
    expected: Vec<f64>,
    oracle: bool,
) -> SelectedAttentionCase {
    let (batch, hq, lq, lk, d, hkv) = (case.batch, case.hq, case.lq, case.lk, case.d, case.hkv());
    let b = Builder::for_tests();
    let q_in = b
        .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
        .unwrap();
    let k_in = b
        .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let v_in = b
        .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let ids_in = b
        .slot_named(
            Slot::Token,
            "ids",
            TensorType::new(vec![batch, 1, lq, case.k], DType::I32),
        )
        .unwrap();
    let pos_in = b
        .slot(Slot::Pos, TensorType::new(vec![batch, lq], DType::I32))
        .unwrap();
    let out = b
        .selected_attention(
            SelectedAttentionAttrs {
                n_rep: case.n_rep,
                scale: case.scale(),
                softcap: None,
                ratio: case.ratio,
                k: case.k,
            },
            q_in,
            k_in,
            v_in,
            ids_in,
            pos_in,
        )
        .unwrap();
    let graph = b.finish(Outputs::new(out)).unwrap();
    let step = || {
        [
            (
                ids_in,
                HostTensor::i32(vec![batch, 1, lq, case.k], case.ids.clone()),
            ),
            (pos_in, HostTensor::i32(vec![batch, lq], case.pos.clone())),
        ]
        .into_iter()
        .map(|(slot, tensor)| StepFixture {
            key: graph
                .meta(slot.id())
                .slot_key()
                .expect("a slot input has a key")
                .clone(),
            tensor,
        })
        .collect::<Vec<_>>()
    };
    let steps = vec![step(), step()];
    SelectedAttentionCase {
        fixture: Fixture {
            name,
            graph,
            store: store_from_tensors(vec![
                ("q", HostTensor::f32(vec![batch, hq, lq, d], q)),
                ("k", HostTensor::f32(vec![batch, hkv, lk, d], k)),
                ("v", HostTensor::f32(vec![batch, hkv, lk, d], v)),
            ]),
            steps,
        },
        expected,
        oracle,
    }
}

/// `fixture`'s program, compiled for `target`, runs its one attention through the generated kernel, not a
/// decomposition into dense primitives.
fn assert_runs_the_kernel(fixture: &Fixture, target: Target) {
    let staged = crate::staged(fixture, target, Submission::Replay);
    let kernels = staged
        .stages()
        .flat_map(|(_, _, program)| {
            program.planned().filter(move |(eqn, _)| {
                eqn.op.name().starts_with("selected_attention")
                    && matches!(
                        program.kernel_choice(eqn),
                        KernelChoice::Generated(request) if request.family() == "attention"
                    )
            })
        })
        .count();
    assert_eq!(
        kernels, 1,
        "{}: the attention runs the kernel",
        fixture.name
    );
}

/// Run `case` on `exec`: the program ran the kernel, and the device's output equals the f64 reference within
/// the tier-2 tolerance at every step. The caller then compares the device with the oracle's dense definition
/// through `run_parity` where [`SelectedAttentionCase::oracle`].
pub fn assert_device_attends(
    exec: &mut dyn poot_executor::Executor,
    target: Target,
    case: &SelectedAttentionCase,
) {
    let name = case.fixture.name;
    assert_runs_the_kernel(&case.fixture, target);
    let outputs = crate::run_outputs(exec, target, &case.fixture).unwrap_or_else(|e| panic!("{e}"));
    for (step, got) in outputs.iter().enumerate() {
        let got = got.as_f32().expect("an F32 attention output");
        assert_eq!(
            got.len(),
            case.expected.len(),
            "{name} step {step}: output length"
        );
        for (i, (g, w)) in got.iter().zip(&case.expected).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 2e-4 * (1.0 + w.abs()),
                "{name} step {step}: output element {i}: the device has {g}, the reference {w}"
            );
        }
    }
}

/// The deep decode of the perf receipt at `lk` stored positions (24 query heads over 2 key/value heads of 256,
/// blocks of four, 512 selected by the plain selection of random scores): the selected attention and, over the
/// same operands, the dense attention every key position of the cache goes through. No reference output is
/// computed (the receipt times the kernels; the cases of [`all`] check them).
pub fn deep_decode_pair(lk: usize) -> (Fixture, Fixture) {
    let (ratio, k) = (4, 512);
    let blocks = lk / ratio;
    let pos = vec![(lk - 1) as i32];
    let ids = reference_row(&grid_scores(blocks, 1 << 20, 99), pos[0], ratio, k)[..k].to_vec();
    let case = Case {
        label: "deep decode".into(),
        batch: 1,
        hq: 24,
        n_rep: 12,
        lq: 1,
        lk,
        d: 256,
        ratio,
        k,
        pos,
        ids,
    };
    let operands = case.operands(33);
    let selected = build_with(
        "selected_attention_deep_decode",
        &case,
        operands.clone(),
        Vec::new(),
        false,
    )
    .fixture;

    // Dense: the same q, k, v under the all-visible causal mask, the plain attention of a dense layer.
    let [q, kk, v] = operands;
    let b = Builder::for_tests();
    let (hq, hkv, d) = (case.hq, case.hkv(), case.d);
    let q_in = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k_in = b
        .constant("k", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let v_in = b
        .constant("v", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let mask_in = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, lk]))
        .unwrap();
    let out = poot_graph_ir::ops::attention_masked_softcap(
        &b,
        q_in,
        k_in,
        v_in,
        case.n_rep,
        case.scale(),
        mask_in,
        None,
    )
    .unwrap();
    let dense = Fixture {
        name: "dense_attention_deep_decode",
        graph: b.finish(Outputs::new(out)).unwrap(),
        store: store_from_tensors(vec![
            ("q", HostTensor::f32(vec![1, hq, 1, d], q)),
            ("k", HostTensor::f32(vec![1, hkv, lk, d], kk)),
            ("v", HostTensor::f32(vec![1, hkv, lk, d], v)),
            ("mask", HostTensor::f32(vec![1, 1, 1, lk], vec![0.0; lk])),
        ]),
        steps: vec![Vec::new(), Vec::new()],
    };
    (selected, dense)
}

/// One TSV line of a perf receipt: `receipt backend fixture median_us min_us max_us samples`.
fn print_receipt_line(backend: &str, name: &str, mut samples: Vec<std::time::Duration>) {
    samples.sort();
    let us = |d: std::time::Duration| d.as_secs_f64() * 1e6;
    println!(
        "receipt\t{backend}\t{name}\t{:.1}\t{:.1}\t{:.1}\t{}",
        us(samples[samples.len() / 2]),
        us(samples[0]),
        us(*samples.last().expect("samples")),
        samples.len()
    );
}

/// The perf receipt of the block top-k kernel on `exec` (device timestamps, not host wall) at 2048, 8192 and
/// 32768 blocks, one query of 512 kept: prints a line per size, returns how many.
pub fn print_top_k_receipt(
    exec: &mut dyn poot_executor::Executor,
    target: Target,
    backend: &str,
) -> usize {
    let top_k: Vec<Fixture> = [2048, 8192, 32768]
        .into_iter()
        .map(crate::block_top_k::timing_fixture)
        .collect();
    let refs: Vec<&Fixture> = top_k.iter().collect();
    let samples = crate::receipt(exec, target, &refs, 4, 8).unwrap_or_else(|e| panic!("{e}"));
    for (fixture, samples) in top_k.iter().zip(samples) {
        print_receipt_line(backend, fixture.name, samples);
    }
    top_k.len()
}

/// The perf receipt of the selected attention over the deep decode at `lk` stored positions, against the dense
/// attention over the same operands when `dense` (the dense row at 128K is the longest dispatch of the family: a
/// device that cannot run it, as wgpu on this box cannot, passes `false`). Prints a line per fixture, returns
/// how many.
pub fn print_attention_receipt(
    exec: &mut dyn poot_executor::Executor,
    target: Target,
    backend: &str,
    lk: usize,
    dense: bool,
) -> usize {
    let (selected, dense_fixture) = deep_decode_pair(lk);
    let fixtures: Vec<&Fixture> = if dense {
        vec![&selected, &dense_fixture]
    } else {
        vec![&selected]
    };
    let samples = crate::receipt(exec, target, &fixtures, 4, 8).unwrap_or_else(|e| panic!("{e}"));
    for (fixture, samples) in fixtures.iter().zip(samples) {
        print_receipt_line(backend, &format!("{}_{lk}", fixture.name), samples);
    }
    fixtures.len()
}
