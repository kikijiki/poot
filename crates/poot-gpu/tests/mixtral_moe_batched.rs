//! GPU (wgpu) tests for the Mixtral shared-pool decode and prefill tracers
//! (`poot_models::mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool`,
//! `trace_mixtral_prefill_kv_shared_pool`). Each dispatches the graph through a real [`GpuExecutor`]
//! and checks it lowers, plans and dispatches without error and matches the CPU oracle
//! (`eval_with_state` on the identical graph and inputs) within f32 tolerance.
//!
//! Decode runs a staggered admit/evict multi-slot schedule. The schedule and binding helpers are
//! copied from `poot-eval`'s `tests::moe_paged_decode` (private test helpers of another crate).

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Executor, HostView, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::{Graph, Slot, SlotKey, Storage, ValueId};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_decode_kv_masked_batched_shared_pool,
    trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_tensor::HostTensor;

/// Compile `g` for `target` with `Submission::Replay` (Card 546b: the contract admits no other
/// submission).
fn staged(g: &Graph, target: Target) -> StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile the mixtral graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// One slot's bytes, ready for [`StepInputs::push`].
struct SlotBind {
    key: SlotKey,
    shape: Vec<usize>,
    dtype: DType,
    bytes: Vec<u8>,
}

fn step_inputs(binds: &[SlotBind]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for b in binds {
        let elems = b.shape.iter().product();
        inputs.push(
            b.key.clone(),
            &b.shape,
            HostView::new(b.dtype, elems, &b.bytes).unwrap(),
        );
    }
    inputs
}

/// Split a `HashMap<ValueId, Value>` bind (built the same way the CPU oracle's is) into the contract's
/// two bind mechanisms: every `Storage::Const` becomes a named [`WeightStore`] entry, every
/// `Storage::Slot` becomes a [`StepInputs`] row (bound in the graph's own declared dtype), and
/// `Storage::State`/`Storage::Computed` are left out - the contract zero-seeds state on first use and
/// self-materializes a computed (folded-iota) input the same way `GpuExecutor` did, neither needing a
/// caller-supplied bind.
fn split_consts_and_slots(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Const => {
                let name = m.name.clone().expect("named const");
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing const input {id:?}"));
                let bytes: Vec<u8> = t
                    .as_f32()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let dense =
                    DenseWeight::try_new(DType::F32, t.shape().to_vec(), Arc::from(bytes)).unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
            Storage::Slot(_) => {
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing slot input {id:?}"));
                let dtype = m.aval.dtype;
                let bytes: Vec<u8> = if dtype == DType::I32 {
                    t.as_i32()
                        .expect("I32-declared slot needs an I32 HostTensor")
                        .iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect()
                } else {
                    t.as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                };
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: t.shape().to_vec(),
                    dtype,
                    bytes,
                });
            }
            Storage::State | Storage::Computed(_) => {}
            Storage::Device => unreachable!(),
        }
    }
    (builder.build(), slots)
}

/// Serializes GPU access within this test binary: the Intel Arc Vulkan driver segfaults when several
/// tests use wgpu devices concurrently. Poison-tolerant.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// Deterministic per-name weight fill, same as `poot-eval`'s `tests::moe_paged_decode::bind_const`.
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

/// One staggered-admission request; mirrors `poot-eval`'s `tests::moe_paged_decode::MixtralReq`.
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

/// Binds one batched shared-pool decode step where physical rows sit at independent positions; same as
/// `poot-eval`'s `tests::moe_paged_decode::bind_batched_step_staggered`.
fn bind_batched_step_staggered(
    g: &Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    batch: usize,
    global_slotmap: &[i32],
) -> HashMap<ValueId, Value> {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
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

/// Runs the CPU-oracle test's staggered admit/evict schedule on the GPU and compares each step's logits
/// against `eval_with_state`. Skips if no Vulkan/wgpu adapter is present.
#[test]
fn mixtral_batched_shared_pool_decode_gpu_matches_cpu_staggered() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

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
    // Private per-physical-row pool region, same scope as the CPU-oracle test.
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

    let global_slotmap: Vec<i32> = (0..n_slots)
        .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
        .collect();

    // One entry for the whole staggered schedule: the contract zero-seeds its state on the first step
    // and carries it, same as the old state_in/returned-buffers hand-off. Consts never depend on
    // token/position, so step 0's bind (all slots zeroed, nothing admitted yet) is as good as any other
    // for extracting them once up front.
    let program = staged(&g, target);
    let init_inputs = bind_batched_step_staggered(
        &g,
        &vec![0u32; n_slots],
        &vec![0usize; n_slots],
        cap,
        n_slots,
        &global_slotmap,
    );
    let (store, _) = split_consts_and_slots(&g, &init_inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();

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
        let inputs =
            bind_batched_step_staggered(&g, &tokens, &positions, cap, n_slots, &global_slotmap);

        let mut cpu_inputs = inputs.clone();
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

        let (_, slot_binds) = split_consts_and_slots(&g, &inputs);
        let step_in = step_inputs(&slot_binds);
        let gpu_bytes = exec
            .step(exe, entry, &step_in, &mut NoSync)
            .expect("mixtral batched decode GPU dispatch")
            .read()
            .expect("read mixtral batched decode output");
        let gpu_logits = f32_bytes(&gpu_bytes);

        assert_eq!(
            gpu_logits.len(),
            cpu_logits.as_f32().unwrap().len(),
            "logits length at step {t}"
        );
        for row in 0..n_slots {
            let got = &gpu_logits[row * cfg.vocab..(row + 1) * cfg.vocab];
            let want = &cpu_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
            for (a, b) in got.iter().zip(want.iter()) {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "non-finite logit at step {t} row {row}: GPU {a} vs CPU {b}"
                );
                let rel = (a - b).abs() / b.abs().max(1e-3);
                max_rel_seen = max_rel_seen.max(rel);
            }
        }
        assert!(
            max_rel_seen <= 5e-3,
            "mixtral batched shared-pool GPU-vs-CPU diff {max_rel_seen} too large at step {t}"
        );
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    eprintln!(
        "mixtral_batched_shared_pool_decode_gpu_matches_cpu_staggered: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {max_step} staggered engine steps, {n_slots} slots"
    );
}

/// Binds one shared-pool prefill call over a deliberately non-contiguous physical slot layout; same as
/// `poot-eval`'s `tests::moe_paged_prefill::bind_shared_pool`.
fn bind_shared_pool_prefill(g: &Graph, tokens: &[u32], inv: &[i32]) -> HashMap<ValueId, Value> {
    let n = tokens.len();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
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

/// Dispatches `trace_mixtral_prefill_kv_shared_pool` over the non-contiguous slot layout used by
/// `poot-eval`'s `tests::moe_paged_prefill::mixtral_shared_pool_prefill_matches_contiguous` and compares
/// GPU logits against `eval_with_state`. Skips if no Vulkan/wgpu adapter is present.
#[test]
fn mixtral_shared_pool_prefill_gpu_matches_cpu_noncontiguous() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

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
    // Copy of `poot_eval::tests::moe_paged_prefill`'s `TOKENS`/`GLOBAL_SLOTS`/`POOL_SLOTS`.
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

    // Card 546b: the contract zero-seeds a newly allocated state buffer on first use (`Engine`'s own
    // state arena), the same zero seed this test used to upload to `caches` by hand.
    let program = staged(&g, target);
    let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let gpu_bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .expect("mixtral shared-pool prefill GPU dispatch")
        .read()
        .expect("read mixtral shared-pool prefill output");
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let gpu_logits = f32_bytes(&gpu_bytes);

    assert_eq!(g.aval(g.output).shape, cpu_logits.shape(), "logits shape");
    assert_eq!(
        gpu_logits.len(),
        cpu_logits.as_f32().unwrap().len(),
        "logits length"
    );
    let mut max_rel = 0.0f32;
    for (a, b) in gpu_logits.iter().zip(cpu_logits.as_f32().unwrap().iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite logit: GPU {a} vs CPU {b}"
        );
        let rel = (a - b).abs() / b.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel <= 5e-3,
        "mixtral shared-pool prefill GPU-vs-CPU diff {max_rel} too large"
    );
    eprintln!(
        "mixtral_shared_pool_prefill_gpu_matches_cpu_noncontiguous: max rel diff vs CPU = {max_rel:.2e}"
    );
}
