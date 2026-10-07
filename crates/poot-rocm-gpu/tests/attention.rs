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

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{attention_masked, attention_prefill_softcap};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, OpKind};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Submission, compile};
use poot_kernelgen::{AttentionSchedule, KernelRequest};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;
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
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![batch, hq, lq, d]));
    let k = b.constant("k", TensorType::f32(vec![batch, hkv, lk, d]));
    let v = b.constant("v", TensorType::f32(vec![batch, hkv, lk, d]));
    let m = b.constant("m", TensorType::f32(vec![batch, 1, lq, lk]));
    let out = if lq == 1 {
        attention_masked(&b, q, k, v, hq / hkv, scale, m)
    } else {
        attention_prefill_softcap(&b, q, k, v, hq / hkv, scale, m, softcap)
    };
    let tensors = [
        ("q", vec![batch, hq, lq, d], fill(1, batch * hq * lq * d)),
        ("k", vec![batch, hkv, lk, d], fill(2, batch * hkv * lk * d)),
        ("v", vec![batch, hkv, lk, d], fill(3, batch * hkv * lk * d)),
        ("m", vec![batch, 1, lq, lk], mask),
    ];
    let ids = [q.id, k.id, v.id, m.id];
    Case {
        graph: b.finish(out),
        ids,
        consts: tensors.map(|(name, shape, data)| ConstFixture {
            name,
            tensor: HostTensor::f32(shape, data),
        }),
        lk,
    }
}

fn check(name: &str, case: &Case) {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let program = compile(
        &case.graph,
        &target,
        &CompileOptions {
            execution: Submission::Replay,
            limits: CompileLimits::STANDARD,
        },
    )
    .unwrap_or_else(|e| panic!("{name}: the attention graph compiles for ROCm: {e}"));
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .unwrap_or_else(|| panic!("{name}: the chain fuses to one attention equation"));
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!("{name}: expected the generated attention kernel");
    };
    let AttentionSchedule::Streaming { lanes } = spec.schedule;
    assert!(
        case.lk > lanes,
        "{name}: {} keys must exceed the {lanes} lanes so lanes walk several keys",
        case.lk
    );

    let inputs: HashMap<_, Value> = case
        .ids
        .iter()
        .zip(&case.consts)
        .map(|(&id, c)| (id, Value::from(c.tensor.clone())))
        .collect();
    let want = eval(
        &case.graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_or_else(|e| panic!("{name}: oracle: {e}"))
    .output
    .into_host()
    .unwrap();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let got = run_once(&mut *exec, target, &case.graph, &case.consts, &[])
        .unwrap_or_else(|e| panic!("{name}: the attention runs through the contract: {e}"));
    assert_eq!(got.shape(), want.shape(), "{name}");
    let worst = got
        .as_f32()
        .unwrap()
        .iter()
        .zip(want.as_f32().unwrap())
        .map(|(a, b)| {
            if (a - b).is_nan() {
                f32::INFINITY
            } else {
                (a - b).abs()
            }
        })
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "{name} ({lanes} lanes): differs from the oracle by {worst:.3e}"
    );
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

use poot_executor::{ExecError, HostView, NoSync, StepInputs, WeightSource};
use poot_graph_ir::{Slot, Storage, ValidationOutputs};
use poot_graph_plan::{
    DeviceId, DevicePlacement, ExpertPlacement, Partition, Target, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_test_util::live_attention::{Inputs, LiveAttentionFixture, LiveAttentionGraph};
use std::sync::Arc;

/// Worst absolute difference between two result vectors; a NaN on either side is infinite.
fn worst_difference(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| {
            if (a - b).is_nan() {
                f32::INFINITY
            } else {
                (a - b).abs()
            }
        })
        .fold(0.0, f32::max)
}

/// SC-001 on ROCm, Gemma2: a GQA decode with attention-logit softcapping, scores far past the cap, over more
/// keys than lanes. Mutation: restore the decode softcap decline in `match_attention`; the chain stays
/// materialized and `check` finds no attention equation.
#[test]
fn gemma2_softcap_decode_matches_the_oracle_on_rocm() {
    let (hq, hkv, lk, d) = (4usize, 2usize, 96usize, 32usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, lk, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, lk, d]));
    let m = b.constant("m", TensorType::f32(vec![1, 1, 1, lk]));
    let out =
        poot_graph_ir::ops::attention_masked_softcap(&b, q, k, v, hq / hkv, 4.0, m, Some(3.0));
    let mask: Vec<f32> = (0..lk)
        .map(|i| if i < lk - 11 { 0.0 } else { -1.0e9 })
        .collect();
    let tensors = [
        ("q", vec![1, hq, 1, d], fill(1, hq * d)),
        ("k", vec![1, hkv, lk, d], fill(2, hkv * lk * d)),
        ("v", vec![1, hkv, lk, d], fill(3, hkv * lk * d)),
        ("m", vec![1, 1, 1, lk], mask),
    ];
    let ids = [q.id, k.id, v.id, m.id];
    check(
        "gemma2 softcap decode",
        &Case {
            graph: b.finish(out),
            ids,
            consts: tensors.map(|(name, shape, data)| ConstFixture {
                name,
                tensor: HostTensor::f32(shape, data),
            }),
            lk,
        },
    );
}

/// SC-001 on ROCm, DeepSeek-V2 MLA: `dk` 192 (`k_nope` 128 per head plus the shared 64-wide `k_pe`), `dv` 128,
/// a split key over more keys than lanes. Mutation: drop the split-key recognition in `peel_key`; no
/// attention equation is planned and the fixture fails.
#[test]
fn deepseek_v2_mla_split_key_decode_matches_the_oracle_on_rocm() {
    let (hq, lk, nope, rope, dv) = (2usize, 100usize, 128usize, 64usize, 128usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, nope + rope]));
    let k_nope = b.constant("k_nope", TensorType::f32(vec![1, hq, lk, nope]));
    let k_pe = b.constant("k_pe", TensorType::f32(vec![1, 1, lk, rope]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, lk, dv]));
    let m = b.constant("m", TensorType::f32(vec![1, 1, 1, lk]));
    let key = b.concat(3, &[k_nope, poot_graph_ir::ops::repeat_kv(&b, k_pe, hq)]);
    let out = attention_masked(&b, q, key, v, 1, 0.07, m);
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
    let ids = [q.id, k_nope.id, k_pe.id, v.id, m.id];
    let graph = b.finish(out);
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let program = compile(
        &graph,
        &target,
        &CompileOptions {
            execution: Submission::Replay,
            limits: CompileLimits::STANDARD,
        },
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
    let want = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let got = run_once(&mut *exec, target, &graph, &consts, &[]).expect("MLA runs on ROCm");
    let worst = worst_difference(got.as_f32().unwrap(), want.as_f32().unwrap());
    assert!(worst <= 1e-3, "MLA differs from the oracle by {worst:.3e}");
}

/// One step of the live-attention decode through the executor contract, validation channel included.
fn live_step(
    exec: &mut dyn Executor,
    target: Target,
    built: &LiveAttentionGraph,
    inputs: &Inputs,
) -> Result<Vec<f32>, ExecError> {
    let graph: &poot_graph_ir::Graph<ValidationOutputs> = &built.graph;
    let program = compile_staged(
        graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            limits: CompileLimits::STANDARD,
        },
    )
    .expect("the live attention compiles for ROCm");
    let mut store = WeightStore::builder();
    let mut pos_bytes = Vec::new();
    let mut pos_key = None;
    for &id in &graph.inputs {
        let meta = graph.meta(id);
        match meta.storage {
            Storage::Const => {
                let tensor = &inputs.0[&id];
                let dense = DenseWeight::try_new(
                    DType::F32,
                    tensor.shape().to_vec(),
                    Arc::from(tensor.view().bytes()),
                )
                .unwrap();
                store
                    .insert(
                        meta.name.clone().expect("a named const"),
                        WeightEntry::Dense(dense),
                    )
                    .unwrap();
            }
            Storage::Slot(Slot::Pos) => {
                pos_bytes = inputs.0[&id]
                    .as_i32()
                    .unwrap()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                pos_key = Some((meta.slot_key().unwrap().clone(), meta.aval.shape.clone()));
            }
            other => panic!("unexpected live-attention input storage {other:?}"),
        }
    }
    let exe = exec
        .load_weights(Arc::new(store.build()), WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let (key, shape) = pos_key.expect("the decode reads Slot::Pos");
    let mut step_in = StepInputs::new();
    step_in.push(
        key,
        &shape,
        HostView::new(DType::I32, shape.iter().product(), &pos_bytes).unwrap(),
    );
    let result = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .map(|mut output| {
            output
                .read()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        });
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    result
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
    };
    let built = fixture.build(true);
    let inputs = fixture.inputs(&built, &[64], true);
    let got = live_step(&mut *exec, target, &built, &inputs).expect("a valid length publishes");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert!(
        worst_difference(&got, cpu.as_f32().unwrap()) <= 1e-3,
        "vs the guarded graph"
    );
    assert!(
        worst_difference(&got, &fixture.oracle(&inputs, &built, &[64])) <= 1e-3,
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
            worst_difference(&got, &fixture.oracle(&inputs, &built, &[len])) <= 1e-3,
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
    };
    let built = fixture.build(false);
    let inputs = fixture.inputs(&built, &[5], true);
    let got = live_step(&mut *exec, target, &built, &inputs)
        .expect("a declined graph runs on the device");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    let cpu = eval(
        &built.graph,
        &inputs.values(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert!(
        worst_difference(&got, cpu.as_f32().unwrap()) <= 1e-3,
        "vs the guarded graph"
    );
    assert!(
        worst_difference(&got, &fixture.oracle(&inputs, &built, &[5])) <= 1e-3,
        "vs the first 5 keys"
    );
}
