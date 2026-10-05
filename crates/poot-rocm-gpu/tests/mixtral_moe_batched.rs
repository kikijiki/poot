//! ROCm/HSA twin of `crates/poot-gpu/tests/mixtral_moe_batched.rs`: the Mixtral shared-pool decode/prefill
//! tracers (`poot_models::mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool` /
//! `trace_mixtral_prefill_kv_shared_pool`) with the same graphs, schedules and CPU-oracle comparison,
//! dispatched through `Engine<RocmDevice>` (Card 548): the executable's `WeightStore` carries every
//! named const, the shared-pool KV state carries itself across steps by (name, aval, storage) -,
//! no more explicit `caches: Vec<RocmBuffer>` threading - and every `Storage::Computed`
//! value materializes inside `Engine::load_entry` with no caller input. The CPU oracle side keeps
//! its own full `Value` map (including consts/computed/state) since `poot_eval::eval` is unchanged.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::{Graph, Slot, SlotKey, Storage, ValueId};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_decode_kv_masked_batched_shared_pool,
    trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_gpu::device::RocmDevice;
use poot_tensor::DType;
use poot_tensor::HostTensor;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// Deterministic per-name weight fill, same scheme as `poot-gpu`'s sibling test and
/// `poot-eval`'s `tests::moe_paged_decode::bind_const`.
fn bind_const(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
        (acc ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// One staggered-admission request, same shape as `poot-gpu`'s sibling test.
struct MixtralReq {
    slot: usize,
    admit_step: usize,
    tokens: Vec<u32>,
}

fn mixtral_staggered_schedule() -> Vec<MixtralReq> {
    vec![
        MixtralReq {
            slot: 0,
            admit_step: 0,
            tokens: vec![0, 3, 1, 5, 2, 4],
        },
        MixtralReq {
            slot: 1,
            admit_step: 2,
            tokens: vec![4, 1, 0, 2],
        },
        MixtralReq {
            slot: 2,
            admit_step: 3,
            tokens: vec![5, 2, 4, 1, 3],
        },
        MixtralReq {
            slot: 3,
            admit_step: 5,
            tokens: vec![1, 0, 5],
        },
        // Reuses slot 0 after the first request there finishes at engine step 6.
        MixtralReq {
            slot: 0,
            admit_step: 7,
            tokens: vec![2, 4, 0, 3],
        },
    ]
}

/// Every named `Storage::Const` input of `g`, bound through `bind_const` (same deterministic fill
/// the CPU oracle side uses) into a `WeightStore` the executable loads once.
fn weight_store_from(g: &Graph) -> Arc<WeightStore> {
    let mut builder = WeightStore::builder();
    for &id in &g.inputs {
        let m = g.meta(id);
        if m.storage != Storage::Const {
            continue;
        }
        let name = m.name.as_deref().unwrap();
        let tensor = bind_const(name, &m.aval.shape);
        let bytes: Arc<[u8]> = Arc::from(
            tensor
                .as_f32()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let dense = DenseWeight::try_new(DType::F32, tensor.shape().to_vec(), bytes).unwrap();
        builder
            .insert(name.to_string(), WeightEntry::Dense(dense))
            .unwrap();
    }
    Arc::new(builder.build())
}

fn staged(
    target: poot_graph_plan::Target,
    g: &Graph,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile_staged")
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One slot value, byte-encoded: `(key, shape, dtype, bytes)`. Owned, so the caller can hold it in
/// a local binding for the `StepInputs` built from it to borrow.
type SlotValue = (SlotKey, Vec<usize>, DType, Vec<u8>);

fn step_inputs_from(values: &[SlotValue]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for (key, shape, dtype, bytes) in values {
        let elems = shape.iter().product();
        inputs.push(
            key.clone(),
            shape,
            poot_executor::HostView::new(*dtype, elems, bytes).unwrap(),
        );
    }
    inputs
}

/// `bind_batched_step_staggered`'s Slot entries only, byte-encoded for `StepInputs` (Const/
/// Computed/State are the executable's/engine's own job under the contract).
fn device_step_values(
    g: &Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    global_slotmap: &[i32],
) -> Vec<SlotValue> {
    let mut values = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let Storage::Slot(slot) = m.storage else {
            continue;
        };
        let key: SlotKey = m.slot_key().unwrap().clone();
        let shape = m.aval.shape.clone();
        match slot {
            Slot::Token => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>()),
            )),
            Slot::Pos => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&positions.iter().map(|&p| p as i32).collect::<Vec<_>>()),
            )),
            Slot::SeqLen => {
                let max_pos = positions.iter().max().copied().unwrap_or(0) as i32;
                values.push((key, shape, DType::I32, i32_bytes(&[max_pos + 1])));
            }
            Slot::Mask => {
                let mask: Vec<f32> = positions
                    .iter()
                    .flat_map(|&pos| (0..cap).map(move |t| if t <= pos { 0.0 } else { -1.0e9 }))
                    .collect();
                values.push((key, shape, DType::F32, f32_bytes(&mask)));
            }
            Slot::SlotMap => values.push((key, shape, DType::I32, i32_bytes(global_slotmap))),
            other => unreachable!("unexpected slot {other:?} in batched shared-pool MoE decode"),
        }
    }
    values
}

/// Bind one batched shared-pool decode step where physical rows sit at independent positions, as in
/// `poot-gpu`'s sibling test / `poot-eval`'s `bind_batched_step_staggered`.
fn bind_batched_step_staggered(
    g: &Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    batch: usize,
    global_slotmap: &[i32],
) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch], tokens.iter().map(|&t| t as i32).collect()).into(),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch], positions.iter().map(|&p| p as i32).collect())
                        .into(),
                );
            }
            Storage::Slot(Slot::SeqLen) => {
                inputs.insert(
                    id,
                    HostTensor::i32(
                        vec![],
                        vec![(positions.iter().max().copied().unwrap_or(0) + 1) as i32],
                    )
                    .into(),
                );
            }
            Storage::Slot(Slot::Mask) => {
                let mask: Vec<f32> = positions
                    .iter()
                    .flat_map(|&pos| (0..cap).map(move |t| if t <= pos { 0.0 } else { -1.0e9 }))
                    .collect();
                inputs.insert(id, HostTensor::f32(vec![batch, cap], mask).into());
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch, cap], global_slotmap.to_vec()).into(),
                );
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in batched shared-pool MoE decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape).into());
            }
            Storage::Computed(computed) => {
                inputs.insert(
                    id,
                    HostTensor::f32(computed.shape(), computed.values_f32()).into(),
                );
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// Mixtral decode tracer on ROCm (spec 266-batched phase 1): the same staggered admit/evict
/// schedule as the CPU-oracle and wgpu tests through a [`RocmGraphExecutor`], comparing each engine
/// step's ROCm logits against `eval_with_state` on the same graph and inputs. Skips without ROCm/HSA.
#[test]
fn mixtral_batched_shared_pool_decode_rocm_matches_cpu_staggered() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let mp = MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    };
    let n_slots = 4usize;
    let cap = 8usize;
    // Private per-physical-row pool region, same scope as the CPU-oracle/wgpu tests.
    let pool_slots = n_slots * cap;
    let reqs = mixtral_staggered_schedule();
    let max_step = reqs
        .iter()
        .map(|r| r.admit_step + r.tokens.len())
        .max()
        .unwrap();

    let g = trace_mixtral_decode_kv_masked_batched_shared_pool(cfg, mp, cap, n_slots, pool_slots);
    g.validate()
        .expect("mixtral batched shared-pool decode graph should validate");

    let mut cpu_caches: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    let store = weight_store_from(&g);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged(target, &g)).unwrap();

    let global_slotmap: Vec<i32> = (0..n_slots)
        .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
        .collect();

    let mut max_rel_seen = 0.0f32;
    for t in 0..max_step {
        let mut tokens = vec![0u32; n_slots];
        let mut positions = vec![0usize; n_slots];
        for r in &reqs {
            if t < r.admit_step || t >= r.admit_step + r.tokens.len() {
                continue;
            }
            let local_pos = t - r.admit_step;
            tokens[r.slot] = r.tokens[local_pos];
            positions[r.slot] = local_pos;
        }
        let cpu_inputs_slots =
            bind_batched_step_staggered(&g, &tokens, &positions, cap, n_slots, &global_slotmap);
        let device_values = device_step_values(&g, &tokens, &positions, cap, &global_slotmap);
        let inputs = step_inputs_from(&device_values);

        let mut cpu_inputs = cpu_inputs_slots;
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            cpu_inputs.insert(si, cpu_caches[ci].clone().into());
        }
        let cpu_eval = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("mixtral batched decode CPU eval");
        let cpu_logits = cpu_eval.output.into_host().expect("dense output");
        let cpu_new: Vec<HostTensor> = cpu_eval
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .expect("dense state");
        cpu_caches = cpu_new;

        let rocm_bytes = engine
            .step(exe, entry, &inputs, &mut NoSync)
            .expect("mixtral batched decode ROCm step")
            .read()
            .expect("mixtral batched decode ROCm readback");
        let rocm_logits = HostTensor::f32(cpu_logits.shape().to_vec(), bytemuck_f32(&rocm_bytes));
        step_check(
            t,
            n_slots,
            cfg.vocab,
            &rocm_logits,
            &cpu_logits,
            &mut max_rel_seen,
        );
    }
    engine.unload(exe).unwrap();
    eprintln!(
        "mixtral_batched_shared_pool_decode_rocm_matches_cpu_staggered: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {max_step} staggered engine steps, {n_slots} slots"
    );
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn step_check(
    t: usize,
    n_slots: usize,
    vocab: usize,
    rocm_logits: &HostTensor,
    cpu_logits: &HostTensor,
    max_rel_seen: &mut f32,
) {
    assert_eq!(
        rocm_logits.shape(),
        cpu_logits.shape(),
        "logits shape at step {t}"
    );
    for row in 0..n_slots {
        let got = &rocm_logits.as_f32().unwrap()[row * vocab..(row + 1) * vocab];
        let want = &cpu_logits.as_f32().unwrap()[row * vocab..(row + 1) * vocab];
        for (a, b) in got.iter().zip(want.iter()) {
            assert!(
                a.is_finite() && b.is_finite(),
                "non-finite logit at step {t} row {row}: rocm={a} cpu={b}"
            );
            let rel = (a - b).abs() / b.abs().max(1e-3);
            *max_rel_seen = max_rel_seen.max(rel);
        }
    }
    assert!(
        *max_rel_seen <= 5e-3,
        "mixtral batched shared-pool ROCm-vs-CPU diff {max_rel_seen} too large at step {t}"
    );
}

/// Bind one shared-pool prefill call over a deliberately non-contiguous physical slot layout, as in
/// `poot-gpu`'s sibling test / `poot-eval`'s `bind_shared_pool`.
fn bind_shared_pool_prefill(g: &Graph, tokens: &[u32], inv: &[i32]) -> HashMap<ValueId, Value> {
    let n = tokens.len();
    let mut inputs = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![n], tokens.iter().map(|&t| t as i32).collect()).into(),
                );
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(m.aval.shape.clone(), inv.to_vec()).into(),
                );
            }
            Storage::Slot(Slot::Mask) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                let l = m.aval.shape[2];
                let mut mask = vec![0.0f32; l * l];
                for i in 0..l {
                    for j in 0..l {
                        mask[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                    }
                }
                inputs.insert(id, HostTensor::f32(m.aval.shape.clone(), mask).into());
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in shared-pool MoE prefill")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape).into());
            }
            Storage::Computed(computed) => {
                inputs.insert(
                    id,
                    HostTensor::f32(computed.shape(), computed.values_f32()).into(),
                );
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// `bind_shared_pool_prefill`'s Slot entries only, byte-encoded for `StepInputs`.
fn device_prefill_values(g: &Graph, tokens: &[u32], inv: &[i32]) -> Vec<SlotValue> {
    let mut values = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let Storage::Slot(slot) = m.storage else {
            continue;
        };
        let key: SlotKey = m.slot_key().unwrap().clone();
        let shape = m.aval.shape.clone();
        match slot {
            Slot::Token => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>()),
            )),
            Slot::SlotMap => values.push((key, shape, DType::I32, i32_bytes(inv))),
            Slot::Mask => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                let l = shape[2];
                let mut mask = vec![0.0f32; l * l];
                for i in 0..l {
                    for j in 0..l {
                        mask[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                    }
                }
                values.push((key, shape, DType::F32, f32_bytes(&mask)));
            }
            other => unreachable!("unexpected slot {other:?} in shared-pool MoE prefill"),
        }
    }
    values
}

/// Mixtral prefill tracer on ROCm (spec 266-batched phase 1): dispatches
/// `poot_models::mixtral::trace_mixtral_prefill_kv_shared_pool` through a [`RocmGraphExecutor`] over
/// the same non-contiguous physical slot layout as the CPU-oracle/wgpu tests and compares logits
/// against `eval_with_state`. Skips without ROCm/HSA.
#[test]
fn mixtral_shared_pool_prefill_rocm_matches_cpu_noncontiguous() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let mp = MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    };
    // Copy of `poot_eval::tests::moe_paged_prefill`'s `TOKENS`/`GLOBAL_SLOTS`/`POOL_SLOTS` (same as the
    // wgpu sibling test).
    const TOKENS: [u32; 7] = [5, 11, 3, 20, 8, 1, 27];
    const GLOBAL_SLOTS: [usize; 7] = [3, 17, 1, 9, 14, 2, 11];
    const POOL_SLOTS: usize = 20;

    let n = TOKENS.len();
    let g = trace_mixtral_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    g.validate()
        .expect("mixtral shared-pool prefill graph should validate");

    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let inputs = bind_shared_pool_prefill(&g, &TOKENS, &inv);

    let mut cpu_inputs = inputs.clone();
    for &(si, _) in &g.state {
        cpu_inputs.insert(si, HostTensor::zeros(g.aval(si).shape.clone()).into());
    }
    let cpu_logits = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("mixtral shared-pool prefill CPU eval")
        .output
        .into_host()
        .expect("dense output");

    let store = weight_store_from(&g);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged(target, &g)).unwrap();
    let device_values = device_prefill_values(&g, &TOKENS, &inv);
    let device_inputs = step_inputs_from(&device_values);
    let rocm_bytes = engine
        .step(exe, entry, &device_inputs, &mut NoSync)
        .expect("mixtral shared-pool prefill ROCm step")
        .read()
        .expect("mixtral shared-pool prefill ROCm readback");
    engine.unload(exe).unwrap();
    let rocm_data = bytemuck_f32(&rocm_bytes);

    assert_eq!(
        rocm_data.len(),
        cpu_logits.as_f32().unwrap().len(),
        "logits shape"
    );
    let mut max_rel = 0.0f32;
    for (a, b) in rocm_data.iter().zip(cpu_logits.as_f32().unwrap().iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite logit: rocm={a} cpu={b}"
        );
        let rel = (a - b).abs() / b.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel <= 5e-3,
        "mixtral shared-pool prefill ROCm-vs-CPU diff {max_rel} too large"
    );
    eprintln!(
        "mixtral_shared_pool_prefill_rocm_matches_cpu_noncontiguous: max rel diff vs CPU = {max_rel:.2e}"
    );
}
