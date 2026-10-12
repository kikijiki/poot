//! POOT-1130 (moved from `poot-graph-plan/src/passes/lora.rs`): SC-003 of the LoRA pass on the real qwen2
//! trace: four decode rows under two adapters on alternate rows each equal the dense graph with that
//! adapter merged into the weights, and adapter index 0 is the base program bit for bit. It traces a model,
//! so it sits above poot-graph-plan; the pass's typed-refusal rows stay in the crate on hand-built graphs.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, LoraPlan, Target, compile, production_plan::Submission,
};
use poot_models::model::{LogitRows, Phase};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_quant::weights::{LoraRole, LoraTarget, WeightId, WeightRole};
use poot_target::Backend;
use poot_tensor::HostTensor;
use poot_test_util::device_caps::default_caps_for;

const ROWS: usize = 4;
const RANK: usize = 2;

fn target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

/// Deterministic fill in [-1, 1) keyed by `seed`.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn name_seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |acc, byte| {
        (acc ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The base value of weight const `name`: the same for every graph that declares it.
fn base_weight(name: &str, numel: usize) -> Vec<f32> {
    fill(numel, name_seed(name))
        .into_iter()
        .map(|v| v * 0.3)
        .collect()
}

/// The tiny qwen2 the LoRA rows trace through the registry.
fn qwen2() -> poot_models::test_support::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model()
}

fn decode(rows: usize) -> Graph {
    qwen2()
        .model
        .trace(
            Phase::Decode,
            step(rows, 1, 8, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// `(in, out)` of every targeted projection of the tiny qwen2.
fn dims(target: LoraTarget) -> (usize, usize) {
    match target {
        LoraTarget::AttnQ => (16, 16),
        LoraTarget::AttnK | LoraTarget::AttnV => (16, 8),
        LoraTarget::AttnO => (16, 16),
        LoraTarget::FfnGate | LoraTarget::FfnUp => (16, 32),
        LoraTarget::FfnDown => (32, 16),
    }
}

/// One adapter: `A [in, r]`, `B [r, out]` per (layer, projection) and its scale.
struct Adapter {
    scale: f32,
    a: HashMap<(usize, LoraTarget), Vec<f32>>,
    b: HashMap<(usize, LoraTarget), Vec<f32>>,
}

fn adapter(seed: u64, scale: f32) -> Adapter {
    let (mut a, mut b) = (HashMap::new(), HashMap::new());
    for layer in 0..2 {
        for (i, target) in LoraTarget::ALL.into_iter().enumerate() {
            let (in_dim, out_dim) = dims(target);
            let s = seed + (layer * 7 + i) as u64 * 11;
            a.insert((layer, target), fill(in_dim * RANK, s));
            b.insert((layer, target), fill(RANK * out_dim, s + 1));
        }
    }
    Adapter { scale, a, b }
}

fn plan() -> LoraPlan {
    LoraPlan {
        rows: NonZeroUsize::new(ROWS).unwrap(),
        adapters: NonZeroUsize::new(2).unwrap(),
        rank: NonZeroUsize::new(RANK).unwrap(),
        projections: LoraTarget::ALL.to_vec(),
    }
}

/// The stacked consts of `adapters`, entry 0 all zeros, by const name.
fn stacked(adapters: &[&Adapter]) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    let mut scale = vec![0.0];
    scale.extend(adapters.iter().map(|adapter| adapter.scale));
    out.insert(
        WeightId::model(WeightRole::Lora(LoraRole::Scale)).const_name(),
        scale,
    );
    for layer in 0..2 {
        for target in LoraTarget::ALL {
            let (in_dim, out_dim) = dims(target);
            let mut a = vec![0.0; in_dim * RANK];
            let mut b = vec![0.0; RANK * out_dim];
            for adapter in adapters {
                a.extend(&adapter.a[&(layer, target)]);
                b.extend(&adapter.b[&(layer, target)]);
            }
            out.insert(
                WeightId::layer(layer, WeightRole::Lora(LoraRole::A(target))).const_name(),
                a,
            );
            out.insert(
                WeightId::layer(layer, WeightRole::Lora(LoraRole::B(target))).const_name(),
                b,
            );
        }
    }
    out
}

/// The base weights with `adapter` merged in: `W + scale * (A B)^T` (the stored `[out, in]` layout).
fn merged(adapter: &Adapter) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    for layer in 0..2 {
        for target in LoraTarget::ALL {
            let (in_dim, out_dim) = dims(target);
            let name = WeightId::layer(layer, target.weight_role()).const_name();
            let mut w = base_weight(&name, in_dim * out_dim);
            let (a, b) = (&adapter.a[&(layer, target)], &adapter.b[&(layer, target)]);
            for i in 0..in_dim {
                for o in 0..out_dim {
                    let delta: f32 = (0..RANK)
                        .map(|k| a[i * RANK + k] * b[k * out_dim + o])
                        .sum();
                    w[o * in_dim + i] += adapter.scale * delta;
                }
            }
            out.insert(name, w);
        }
    }
    out
}

/// Evaluate `g` on the oracle for one decode step: `tokens` per row, `lora` rows when it declares
/// the slot, consts from `overrides` or the base fill.
fn run(g: &Graph, overrides: &HashMap<String, Vec<f32>>, tokens: &[i32], lora: &[i32]) -> Vec<f32> {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in g.inputs() {
        let aval = g.aval(id).clone();
        let meta = g.meta(id);
        let tensor = match meta.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(aval.shape, tokens.to_vec()),
            Storage::Slot(Slot::Pos) => HostTensor::i32(aval.shape, vec![3; tokens.len()]),
            Storage::Slot(Slot::LoraIdx) => HostTensor::i32(aval.shape, lora.to_vec()),
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Computed(c) => c.tensor(),
            Storage::State => HostTensor::f32(aval.shape.clone(), vec![0.0; aval.numel()]),
            Storage::Const => {
                let name = meta.name.as_deref().expect("a const has a name");
                let data = overrides
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| base_weight(name, aval.numel()));
                HostTensor::f32(aval.shape, data)
            }
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, Value::from(tensor));
    }
    eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec()
}

fn compiled(g: &Graph, lora: Option<LoraPlan>) -> Graph {
    let mut options = options();
    options.lora = lora;
    compile(g, &target(), &options)
        .unwrap_or_else(|e| panic!("compile: {e}"))
        .graph()
        .clone()
}

const TOKENS: [i32; ROWS] = [5, 9, 3, 11];

/// Row `r`'s slice of `[ROWS, 1, vocab]` logits.
fn row(logits: &[f32], r: usize) -> &[f32] {
    let vocab = logits.len() / ROWS;
    &logits[r * vocab..(r + 1) * vocab]
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{what}: logit {i}: got {g}, merged-weight reference {w}"
        );
    }
}

/// SC-003: qwen2 decodes four rows under two adapters on alternate rows (and one no-adapter row);
/// each row's logits equal the dense graph with that adapter merged into the weights, and the
/// no-adapter row equals the unadapted program bit for bit.
#[test]
fn lora_rows_equal_the_per_row_merged_weight_graph_and_index_zero_is_the_base() {
    let g = decode(ROWS);
    let (first, second) = (adapter(100, 2.0), adapter(200, 3.0));
    let program = compiled(&g, Some(plan()));
    let base = compiled(&g, None);
    let consts = stacked(&[&first, &second]);
    let idx = [1, 2, 0, 2];
    let adapted = run(&program, &consts, &TOKENS, &idx);

    let reference = |adapter: &Adapter| run(&base, &merged(adapter), &TOKENS, &[]);
    let (with_first, with_second) = (reference(&first), reference(&second));
    assert_close(
        row(&adapted, 0),
        row(&with_first, 0),
        "row 0 under adapter 1",
    );
    assert_close(
        row(&adapted, 1),
        row(&with_second, 1),
        "row 1 under adapter 2",
    );
    assert_close(
        row(&adapted, 3),
        row(&with_second, 3),
        "row 3 under adapter 2",
    );

    let plain_run = run(&base, &HashMap::new(), &TOKENS, &[]);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(row(&adapted, 2)),
        bits(row(&plain_run, 2)),
        "LoraIdx 0 must equal the no-adapter program bit for bit"
    );
    // The adapters visibly change the output: a correction that were all zero would pass the rows above.
    assert!(
        row(&adapted, 0)
            .iter()
            .zip(row(&plain_run, 0))
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "adapter 1 changes row 0's logits"
    );
}
