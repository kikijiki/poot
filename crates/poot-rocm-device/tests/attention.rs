//! The generated attention kernel on ROCm, one fixture per phase, each through the one contract path
//! (`compile`, `load_weights`, `add_entry`, `step`) against the CPU oracle's evaluation of the same graph.
//!
//! Every fixture has more keys than the kernel has lanes, so each lane walks several keys and the running
//! maximum, denominator and output are rescaled between them: a kernel that skipped the running-sum rescale
//! passes only a fixture whose lanes see one key each. Decode is D = 512 (the head width that fixes the
//! lane count lowest, 16 lanes on this device) over 96 keys with a per-batch-row visible length; prefill is a
//! softcapped D = 128 over 80 rows (64 lanes), each row attending its causal prefix.
//!
//! Mutation (guards `poot-kernelgen`'s `attention.rs`, applied once, observed red, restored): drop the
//! `l *= exp(m_old - m_new)` rescale; both rows go red with the worst absolute error against the oracle.

use poot_eval::test_support::StepFixture;
use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{attention_masked, attention_prefill_softcap};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, OpKind};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, compile};
use poot_kernelgen as kg;
use poot_kernelgen::KernelRequest;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::{DType, HostTensor};
use std::collections::HashMap;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// One attention fixture: the traced chain, its four operands, and the lane count the planner must pick.
struct Case {
    graph: Graph,
    ids: [poot_graph_ir::ValueId; 4],
    consts: [ConstFixture; 4],
    lk: usize,
}

/// `attention_masked` (decode) or `attention_prefill_softcap` over `[batch, hq, lq, d]` queries and
/// `[batch, hkv, lk, d]` keys and values, with `mask` (`[batch, 1, lq, lk]`) bound as given.
fn case(
    (batch, hq, hkv, lq, lk, d): (usize, usize, usize, usize, usize, usize),
    softcap: Option<f32>,
    mask: Vec<f32>,
) -> Case {
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let m = b
        .constant("m", TensorType::f32(vec![batch, 1, lq, lk]))
        .unwrap();
    let out = if lq == 1 {
        attention_masked(&b, q, k, v, hq / hkv, scale, m).unwrap()
    } else {
        attention_prefill_softcap(&b, q, k, v, hq / hkv, scale, m, softcap).unwrap()
    };
    let tensors = [
        ("q", vec![batch, hq, lq, d], fill(1, batch * hq * lq * d)),
        ("k", vec![batch, hkv, lk, d], fill(2, batch * hkv * lk * d)),
        ("v", vec![batch, hkv, lk, d], fill(3, batch * hkv * lk * d)),
        ("m", vec![batch, 1, lq, lk], mask),
    ];
    let ids = [q.id(), k.id(), v.id(), m.id()];
    Case {
        graph: b.finish(out).unwrap(),
        ids,
        consts: tensors.map(|(name, shape, data)| ConstFixture {
            name,
            tensor: HostTensor::f32(shape, data),
        }),
        lk,
    }
}

fn check(name: &str, case: &Case) {
    check_operands(name, &case.graph, &case.ids, &case.consts, &[], case.lk);
}

/// `graph` through the contract on ROCm against the oracle, its operands `ids` bound to `consts` and its slot
/// inputs `slots` (a value and the tensor its step binds), with `lk` keys that must exceed the lanes of the
/// planned attention kernel.
fn check_operands(
    name: &str,
    graph: &Graph,
    ids: &[poot_graph_ir::ValueId],
    consts: &[ConstFixture],
    slots: &[(poot_graph_ir::ValueId, HostTensor)],
    lk: usize,
) -> Option<kg::AttentionSpec> {
    let device = open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
    let target = device.target();
    let program = compile(
        graph,
        &target,
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .unwrap_or_else(|e| panic!("{name}: the attention graph compiles for ROCm: {e}"));
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .unwrap_or_else(|| panic!("{name}: the chain fuses to one attention equation"));
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!("{name}: expected the generated attention kernel");
    };
    let spec = spec.clone();
    let lanes = spec.schedule.lanes();
    assert!(
        lk > lanes,
        "{name}: {lk} keys must exceed the {lanes} lanes so lanes walk several keys"
    );

    let inputs: HashMap<_, Value> = ids
        .iter()
        .zip(consts)
        .map(|(&id, c)| (id, Value::from(c.tensor.clone())))
        .chain(slots.iter().map(|(id, t)| (*id, Value::from(t.clone()))))
        .collect();
    let want = eval(graph, &inputs, EvalOptions::default())
        .unwrap_or_else(|e| panic!("{name}: oracle: {e}"))
        .output
        .into_host()
        .unwrap();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let step: Vec<StepFixture> = slots
        .iter()
        .map(|(id, tensor)| StepFixture {
            key: graph
                .meta(*id)
                .slot_key()
                .expect("a slot has a key")
                .clone(),
            tensor: tensor.clone(),
        })
        .collect();
    let got = run_once(&mut *exec, target, graph, consts, &step)
        .unwrap_or_else(|e| panic!("{name}: the attention runs through the contract: {e}"));
    assert_eq!(got.shape(), want.shape(), "{name}");
    let worst = poot_test_util::max_abs_error(got.as_f32().unwrap(), want.as_f32().unwrap());
    assert!(
        worst <= 1e-3,
        "{name} ({lanes} lanes): differs from the oracle by {worst:.3e}"
    );
    Some(spec)
}

#[test]
fn decode_at_head_dim_512_over_many_key_tiles_matches_the_oracle_on_rocm() {
    let (batch, hq, hkv, lk, d) = (2usize, 2usize, 1usize, 96usize, 512usize);
    // Batch row `b` sees its first `lk - 13 * b` keys.
    let mask = (0..batch * lk)
        .map(|i| {
            if i % lk < lk - 13 * (i / lk) {
                0.0
            } else {
                -1.0e9
            }
        })
        .collect();
    check(
        "decode d=512",
        &case((batch, hq, hkv, 1, lk, d), None, mask),
    );
}

#[test]
fn softcapped_prefill_over_many_key_tiles_matches_the_oracle_on_rocm() {
    let (hq, hkv, l, d) = (2usize, 1usize, 80usize, 128usize);
    let mask = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    check(
        "softcapped prefill d=128",
        &case((1, hq, hkv, l, l, d), Some(50.0), mask),
    );
}

/// Prefill into a cache (`1 < Lq < Lk`): three new rows against an 80-slot cache whose first 70 slots hold
/// the prefix (80 keys over 64 lanes), so each lane may walk two keys and the mask row stride is `Lk`, not
/// `Lq`.
#[test]
fn prefill_into_a_cache_matches_the_oracle_on_rocm() {
    let (hq, hkv, lq, lk, d) = (2usize, 1usize, 3usize, 80usize, 128usize);
    let mask = (0..lq * lk)
        .map(|i| if i % lk <= 70 + i / lk { 0.0 } else { -1.0e9 })
        .collect();
    check(
        "prefill into a cache",
        &case((1, hq, hkv, lq, lk, d), None, mask),
    );
}

// ---- softcap decode, MLA and the live bound (POOT-730) ---------------------------------------------------

use poot_eval::test_support::live_attention::LiveAttentionFixture;
use poot_executor::ExecError;
use poot_executor_parity::live_attention::live_step;

/// SC-001 on ROCm, Gemma2: a GQA decode with attention-logit softcapping, scores far past the cap, over more
/// keys than lanes. Mutation: restore the decode softcap decline in `match_attention`; the chain stays
/// materialized and `check` finds no attention equation.
#[test]
fn gemma2_softcap_decode_matches_the_oracle_on_rocm() {
    let (hq, hkv, lk, d) = (4usize, 2usize, 96usize, 32usize);
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let m = b.constant("m", TensorType::f32(vec![1, 1, 1, lk])).unwrap();
    let out =
        poot_graph_ir::ops::attention_masked_softcap(&b, q, k, v, hq / hkv, 4.0, m, Some(3.0))
            .unwrap();
    let mask: Vec<f32> = (0..lk)
        .map(|i| if i < lk - 11 { 0.0 } else { -1.0e9 })
        .collect();
    let tensors = [
        ("q", vec![1, hq, 1, d], fill(1, hq * d)),
        ("k", vec![1, hkv, lk, d], fill(2, hkv * lk * d)),
        ("v", vec![1, hkv, lk, d], fill(3, hkv * lk * d)),
        ("m", vec![1, 1, 1, lk], mask),
    ];
    let ids = [q.id(), k.id(), v.id(), m.id()];
    check(
        "gemma2 softcap decode",
        &Case {
            graph: b.finish(out).unwrap(),
            ids,
            consts: tensors.map(|(name, shape, data)| ConstFixture {
                name,
                tensor: HostTensor::f32(shape, data),
            }),
            lk,
        },
    );
}

/// A GQA attention with a per-head sink: `rows` queries into `lk` keys, over more keys than lanes. The chain's
/// `Attention` equation carries the sink and matches the oracle. One device per test: a second device opened in
/// the same process can fail with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.
fn check_per_head_sink(label: &str, rows: usize, lk: usize) {
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, rows, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let m = b
        .constant("m", TensorType::f32(vec![1, 1, rows, lk]))
        .unwrap();
    let sinks = b.constant("sinks", TensorType::f32(vec![hq])).unwrap();
    let out = poot_graph_ir::ops::attention_masked_sinks(
        &b,
        q,
        k,
        v,
        hq / hkv,
        1.0 / (d as f32).sqrt(),
        m,
        sinks,
    )
    .unwrap();
    // Row `i` sees all but the last `rows - 1 - i` keys.
    let mask = (0..rows * lk)
        .map(|at| {
            if at % lk < lk - (rows - 1 - at / lk) {
                0.0
            } else {
                -1.0e9
            }
        })
        .collect();
    let tensors = [
        ("q", vec![1, hq, rows, d], fill(1, hq * rows * d)),
        ("k", vec![1, hkv, lk, d], fill(2, hkv * lk * d)),
        ("v", vec![1, hkv, lk, d], fill(3, hkv * lk * d)),
        ("m", vec![1, 1, rows, lk], mask),
        ("sinks", vec![hq], vec![-1.0, 0.0, 0.25, 2.0]),
    ];
    let ids = [q.id(), k.id(), v.id(), m.id(), sinks.id()];
    let graph = b.finish(out).unwrap();
    check_operands(
        label,
        &graph,
        &ids,
        &tensors.map(|(name, shape, data)| ConstFixture {
            name,
            tensor: HostTensor::f32(shape, data),
        }),
        &[],
        lk,
    );
}

/// SC-001 on ROCm, gpt-oss: a GQA decode with a per-head attention sink, over more keys than lanes, fuses to the
/// generated attention (the chain's `Attention` equation carries the sink) and matches the oracle. Mutation:
/// drop the sink from the kernel's initial `(m, l)` state in `attention_body`; the sink's mass vanishes from the
/// normalizer and the row differs from the oracle.
#[test]
fn per_head_sink_decode_matches_the_oracle_on_rocm() {
    check_per_head_sink("sink decode", 1, 96);
}

/// SC-001 on ROCm, gpt-oss prefill: three rows into an 80-slot cache with the per-head sink. Mutation: as
/// `per_head_sink_decode_matches_the_oracle_on_rocm`.
#[test]
fn per_head_sink_prefill_matches_the_oracle_on_rocm() {
    check_per_head_sink("sink prefill", 3, 80);
}

/// A causal attention of `rows` queries at positions `pos0..pos0 + rows` against `lk` keys under `window`: the
/// mask `causal_mask_from_pos` traces is stated by the positions (a ranged `Slot::Pos`, as the tracers declare
/// it), the planned kernel is `MaskKind::Causal` and matches the oracle, which evaluates the mask tensor. One
/// device per test, as `check_per_head_sink`.
fn check_causal(label: &str, rows: usize, lk: usize, pos0: i32, window: Option<usize>) {
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, rows, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, lk, d]))
        .unwrap();
    let pos = b
        .slot_ranged(
            poot_graph_ir::Slot::Pos,
            TensorType::new(vec![1, rows], DType::I32),
            poot_graph_ir::IndexRange::below(lk).unwrap(),
        )
        .unwrap();
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, lk, window).unwrap();
    let out = attention_masked(&b, q, k, v, hq / hkv, 1.0 / (d as f32).sqrt(), mask).unwrap();
    let ids = [q.id(), k.id(), v.id()];
    let positions = (
        pos.id(),
        HostTensor::i32(vec![1, rows], (pos0..pos0 + rows as i32).collect()),
    );
    let graph = b.finish(out).unwrap();
    let consts = [
        ConstFixture {
            name: "q",
            tensor: HostTensor::f32(vec![1, hq, rows, d], fill(1, hq * rows * d)),
        },
        ConstFixture {
            name: "k",
            tensor: HostTensor::f32(vec![1, hkv, lk, d], fill(2, hkv * lk * d)),
        },
        ConstFixture {
            name: "v",
            tensor: HostTensor::f32(vec![1, hkv, lk, d], fill(3, hkv * lk * d)),
        },
    ];
    if let Some(spec) = check_operands(label, &graph, &ids, &consts, &[positions], lk) {
        assert_eq!(spec.mask, kg::MaskKind::Causal { window }, "{label}");
    }
}

/// SC-001 (R2-C7-006) on ROCm: a decode at position 77 of a 96-slot cache, every earlier key visible. Mutation:
/// leave the causal row's `key_end` at the capacity in `attention_body`; the windowed rows below read keys
/// their window hides and differ from the oracle.
#[test]
fn causal_decode_matches_the_oracle_on_rocm() {
    check_causal("causal decode", 1, 96, 77, None);
}

/// The same decode under a 20-key window. Mutation: as `causal_decode_matches_the_oracle_on_rocm`.
#[test]
fn windowed_decode_matches_the_oracle_on_rocm() {
    check_causal("windowed decode", 1, 96, 77, Some(20));
}

/// A prefill of three rows at positions 60..63 of an 80-slot cache under a 12-key window, each row its own
/// window. Mutation: as `causal_decode_matches_the_oracle_on_rocm`.
#[test]
fn windowed_prefill_matches_the_oracle_on_rocm() {
    check_causal("windowed prefill", 3, 80, 60, Some(12));
}

/// SC-001 on ROCm, DeepSeek-V2 MLA:/// SC-001 on ROCm, DeepSeek-V2 MLA: `dk` 192 (`k_nope` 128 per head plus the shared 64-wide `k_pe`), `dv` 128,
/// a split key over more keys than lanes. Mutation: drop the split-key recognition in `peel_key`; no
/// attention equation is planned and the fixture fails.
#[test]
fn deepseek_v2_mla_split_key_decode_matches_the_oracle_on_rocm() {
    let (hq, lk, nope, rope, dv) = (2usize, 100usize, 128usize, 64usize, 128usize);
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![1, hq, 1, nope + rope]))
        .unwrap();
    let k_nope = b
        .constant("k_nope", TensorType::f32(vec![1, hq, lk, nope]))
        .unwrap();
    let k_pe = b
        .constant("k_pe", TensorType::f32(vec![1, 1, lk, rope]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hq, lk, dv]))
        .unwrap();
    let m = b.constant("m", TensorType::f32(vec![1, 1, 1, lk])).unwrap();
    let key = b
        .concat(
            3,
            &[k_nope, poot_graph_ir::ops::repeat_kv(&b, k_pe, hq).unwrap()],
        )
        .unwrap();
    let out = attention_masked(&b, q, key, v, 1, 0.07, m).unwrap();
    let mask: Vec<f32> = (0..lk)
        .map(|i| if i < lk - 9 { 0.0 } else { -1.0e9 })
        .collect();
    let tensors = [
        (
            "q",
            vec![1, hq, 1, nope + rope],
            fill(1, hq * (nope + rope)),
        ),
        ("k_nope", vec![1, hq, lk, nope], fill(2, hq * lk * nope)),
        ("k_pe", vec![1, 1, lk, rope], fill(3, lk * rope)),
        ("v", vec![1, hq, lk, dv], fill(4, hq * lk * dv)),
        ("m", vec![1, 1, 1, lk], mask),
    ];
    let ids = [q.id(), k_nope.id(), k_pe.id(), v.id(), m.id()];
    let graph = b.finish(out).unwrap();
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let program = compile(
        &graph,
        &target,
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect("the MLA graph compiles for ROCm");
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .expect("the MLA chain fuses to one attention equation");
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!("expected the generated attention kernel");
    };
    assert_eq!(
        spec.key,
        poot_kernelgen::KeyLayout::Split { shared_dim: rope }
    );
    assert_eq!((spec.dk, spec.dv), (nope + rope, dv));
    let consts = tensors.map(|(name, shape, data)| ConstFixture {
        name,
        tensor: HostTensor::f32(shape, data),
    });
    let inputs: HashMap<_, Value> = ids
        .iter()
        .zip(&consts)
        .map(|(&id, c)| (id, Value::from(c.tensor.clone())))
        .collect();
    let want = eval(&graph, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let got = run_once(&mut *exec, target, &graph, &consts, &[]).expect("MLA runs on ROCm");
    let worst = poot_test_util::max_abs_error(got.as_f32().unwrap(), want.as_f32().unwrap());
    assert!(worst <= 1e-3, "MLA differs from the oracle by {worst:.3e}");
}

/// SC-002 on ROCm: capacity 4096, the length derived in the graph as 64 from `Pos`, NaN in every key and value
/// from 64 on. The output is finite and within the attention tier of the guarded graph's CPU evaluation and
/// the first-64-key oracle. Mutation: bound the generated loop by the capacity; the tail is consumed and the
/// output is NaN.
#[test]
fn a_live_length_never_reads_the_poisoned_tail_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 4096,
        dk: 64,
        dv: 64,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(true);
    let inputs = fixture.inputs(&built, &[64], true);
    let got = live_step(&mut *exec, target, &built, &inputs).expect("a valid length publishes");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(&built.graph, &inputs.values(), EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert!(
        poot_test_util::max_abs_error(&got, cpu.as_f32().unwrap()) <= 1e-3,
        "vs the guarded graph"
    );
    assert!(
        poot_test_util::max_abs_error(&got, &fixture.oracle(&inputs, &built, &[64])) <= 1e-3,
        "vs the first 64 keys"
    );
}

/// SC-004 on ROCm: lengths 0, -1 and capacity + 1 fail through the validation packet with nothing published;
/// the endpoints 1 and capacity run and match the oracle. Mutation: widen the guard's bound in
/// `ops::attention_live`; capacity + 1 publishes.
#[test]
fn an_out_of_range_live_length_publishes_nothing_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: Some(30.0),
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(true);
    for len in [0, -1, 10] {
        let inputs = fixture.inputs(&built, &[len], false);
        let outcome = live_step(&mut *exec, target, &built, &inputs);
        assert!(
            matches!(outcome, Err(ExecError::Validation(_))),
            "len {len}: expected a validation failure, got {outcome:?}"
        );
    }
    for len in [1, 9] {
        let inputs = fixture.inputs(&built, &[len], true);
        let got = live_step(&mut *exec, target, &built, &inputs)
            .unwrap_or_else(|e| panic!("len {len}: {e:?}"));
        assert!(
            poot_test_util::max_abs_error(&got, &fixture.oracle(&inputs, &built, &[len])) <= 1e-3,
            "len {len}"
        );
    }
}

/// POOT-1040 on ROCm: the live-attention graph with no declared length witness is never proved, so the matcher
/// declines it and it stays the generic primitives, key ends in exact I32 included. The device runs it, and with
/// NaN in every key and value past the length the output is finite and within the attention tier of the
/// first-5-key oracle. Mutation: build `LiveKeys::new`'s key ends from an F32 iota plus a `Cast` to I32
/// (`ops::attention_over_keys`); the plan refuses the F32 -> I32 cast.
#[test]
fn a_declined_live_attention_graph_runs_as_primitives_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 9,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(false);
    let inputs = fixture.inputs(&built, &[5], true);
    let got = live_step(&mut *exec, target, &built, &inputs)
        .expect("a declined graph runs on the device");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(&built.graph, &inputs.values(), EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert!(
        poot_test_util::max_abs_error(&got, cpu.as_f32().unwrap()) <= 1e-3,
        "vs the guarded graph"
    );
    assert!(
        poot_test_util::max_abs_error(&got, &fixture.oracle(&inputs, &built, &[5])) <= 1e-3,
        "vs the first 5 keys"
    );
}

/// POOT-1041 on ROCm: the guarded live composite over BF16 and F16 operands (dense and split key, batch rows
/// with different lengths, NaN in every key and value past each length) runs as generic primitives at its own
/// dtype, with the bit-moving two-byte `Select` guarding the keys, values and scores. The output is finite and
/// within the two-byte tier of the guarded graph's CPU evaluation and of the first-`len` keys' oracle.
/// Mutation: make the `Select` always take `if_true` in `emit_select` (kernelgen `fused.rs`); the poisoned tail
/// reaches the matmuls and the finiteness assertion fails.
#[test]
fn a_narrow_live_attention_graph_runs_on_rocm_and_never_reads_the_poisoned_tail() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for (dtype, tier) in [(DType::BF16, 4e-3), (DType::F16, 1e-3)] {
        for shared_dim in [None, Some(4)] {
            let fixture = LiveAttentionFixture {
                batch: 2,
                hq: 4,
                hkv: 2,
                cap: 37,
                dk: 12,
                dv: 8,
                shared_dim,
                softcap: Some(30.0),
                dtype,
            };
            let label = format!("{dtype:?} shared {shared_dim:?}");
            let built = fixture.build(true);
            let len = [5, 37];
            let inputs = fixture.inputs(&built, &len, true);
            let got = live_step(&mut *exec, target, &built, &inputs).unwrap_or_else(|e| {
                panic!("{label}: a narrow live graph runs on the device: {e:?}")
            });
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{label}: poison reached the output: {got:?}"
            );
            let cpu = eval(&built.graph, &inputs.values(), EvalOptions::default())
                .unwrap()
                .output
                .into_host()
                .unwrap();
            let vs_graph = poot_test_util::max_abs_error(&got, cpu.as_f32().unwrap());
            let vs_oracle =
                poot_test_util::max_abs_error(&got, &fixture.oracle(&inputs, &built, &len));
            eprintln!("{label}: vs guarded graph {vs_graph:.3e}, vs oracle {vs_oracle:.3e}");
            assert!(
                vs_graph <= tier,
                "{label}: vs the guarded graph: {vs_graph:.3e}"
            );
            assert!(
                vs_oracle <= tier,
                "{label}: vs the first keys: {vs_oracle:.3e}"
            );
        }
    }
}
