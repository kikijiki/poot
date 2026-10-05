//! wgpu/RADV test for `poot_models::deepseek3::trace_deepseek3_decode_kv_masked_batched_shared_pool`
//! (spec 269 stage 2, card 274, SC-003).
//!
//! Dispatches the graph through a real [`GpuExecutor`] and checks that it (1) lowers, plans and dispatches
//! without a `plan_eqn` rejection, SPIR-V validation failure or dispatch error, and (2) matches the CPU
//! oracle (`eval_with_state` on the identical graph and inputs) within f32 tolerance. Mirrors
//! `mixtral_moe_batched.rs`.
//!
//! Uses the staggered admit/evict multi-slot schedule of `poot_llm::deepseek3_mla_batched_kv`'s CPU-oracle
//! test (5 requests over 4 physical slots, one slot reused mid-run) and its `bind_deepseek3_batched_kv_step`
//! and fixture shapes, duplicated because they are private to another crate's tests. wgpu, decode, dense,
//! F32 only: no `poot-serve` wiring, MoE-pool composition or real checkpoint.

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
use poot_models::deepseek2::DeepseekV2Config;
use poot_models::deepseek3::{
    DeepseekV3MoeParams, trace_deepseek3_decode_kv_masked_batched_shared_pool,
    trace_deepseek3_prefill_kv_shared_pool,
};
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
    .expect("compile the deepseek3 mla graph")
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

/// Serialize GPU access: the Intel Arc Vulkan driver segfaults when several tests use wgpu devices
/// concurrently. Poison-tolerant.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Small non-degenerate DeepSeek-V3-MLA fixture, identical to `poot_llm::deepseek3_mla_batched_kv`'s test `cfg()`.
fn cfg() -> DeepseekV2Config {
    DeepseekV2Config {
        vocab: 20,
        hidden: 16,
        layers: 2,
        n_heads: 4,
        q_lora_rank: Some(8),
        kv_lora_rank: 8,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        rope_theta: 10_000.0,
        yarn: None,
    }
}

/// `first_k_dense_replace: 1`: layer 0 dense, layer 1 routed, so both FFN branches run. Identical to that test's `moe_params()`.
fn moe_params() -> DeepseekV3MoeParams {
    DeepseekV3MoeParams {
        n_routed_experts: 4,
        top_k: 2,
        moe_inter: 8,
        n_shared_experts: 1,
        dense_inter: 12,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

use poot_test_util::seed_of;

/// Deterministic non-degenerate synthetic data, identical to that test's `fill()`.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let seed = seed as f32 * 1e-9;
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.0137 + seed * 0.911).sin() * 0.6)
        .collect()
}

/// Every `Const` a graph declares, filled by `(name, shape)`, identical to that test's `synth_weights()`.
fn synth_weights(g: &Graph) -> HashMap<String, HostTensor> {
    let mut w = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        if let Storage::Const = m.storage {
            let name = m.name.clone().expect("named const");
            w.entry(name.clone()).or_insert_with(|| {
                HostTensor::f32(m.aval.shape.clone(), fill(m.aval.numel(), seed_of(&name)))
            });
        }
    }
    w
}

/// One staggered-admission request, identical to that test's `Req`/`staggered_schedule()`.
struct Req {
    slot: usize,
    admit_step: usize,
    tokens: Vec<u32>,
}

fn staggered_schedule() -> Vec<Req> {
    vec![
        Req {
            slot: 0,
            admit_step: 0,
            tokens: vec![0, 3, 1, 5, 2, 4],
        },
        Req {
            slot: 1,
            admit_step: 2,
            tokens: vec![4, 1, 0, 2],
        },
        Req {
            slot: 2,
            admit_step: 3,
            tokens: vec![5, 2, 4, 1, 3],
        },
        Req {
            slot: 3,
            admit_step: 5,
            tokens: vec![1, 0, 5],
        },
        // Reuses slot 0 after its first request finishes at engine step 6.
        Req {
            slot: 0,
            admit_step: 7,
            tokens: vec![2, 4, 0, 3],
        },
    ]
}

/// Bind one batched shared-KV-pool MLA decode step with physical rows at independent positions; identical to
/// the private `bind_deepseek3_batched_kv_step` in `poot_llm::deepseek3_mla_batched_kv`.
fn bind_batched_step_staggered(
    g: &Graph,
    weights: &HashMap<String, HostTensor>,
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
                unreachable!("unexpected slot {other:?} in batched shared-pool MLA decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(
                    id,
                    weights
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("no weight for {name}"))
                        .into(),
                );
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

/// Drives the staggered admit/evict schedule of the CPU-oracle test through a real [`GpuExecutor`],
/// comparing each engine step's GPU logits against `eval_with_state` on the identical graph and inputs.
/// Skips without a Vulkan/wgpu adapter.
#[test]
fn deepseek3_mla_batched_kv_decode_gpu_matches_cpu_staggered() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = cfg();
    let mp = moe_params();
    let cap = 8usize;
    let batch = 4usize;
    // Private per-physical-row pool region, as in `Deepseek3BatchedKvDecoder`.
    let kv_pool_slots = batch * cap;
    let reqs = staggered_schedule();
    let max_step = reqs
        .iter()
        .map(|r| r.admit_step + r.tokens.len())
        .max()
        .unwrap();

    let g =
        trace_deepseek3_decode_kv_masked_batched_shared_pool(cfg, mp, cap, batch, kv_pool_slots);
    g.validate()
        .expect("deepseek3 batched MLA decode graph should validate");
    let weights = synth_weights(&g);

    let mut cpu_caches: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    let global_slotmap: Vec<i32> = (0..batch)
        .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
        .collect();

    // One entry for the whole staggered schedule: the contract zero-seeds its state on the first step
    // and carries it, same as the old state_in/returned-buffers hand-off. Consts never depend on
    // token/position, so step 0's bind (all slots zeroed, nothing admitted yet) is as good as any other
    // for extracting them once up front.
    let program = staged(&g, target);
    let init_inputs = bind_batched_step_staggered(
        &g,
        &weights,
        &vec![0u32; batch],
        &vec![0usize; batch],
        cap,
        batch,
        &global_slotmap,
    );
    let (store, _) = split_consts_and_slots(&g, &init_inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();

    let mut max_rel_seen = 0.0f32;
    for t in 0..max_step {
        let mut tokens = vec![0u32; batch];
        let mut positions = vec![0usize; batch];
        for r in &reqs {
            if t < r.admit_step || t >= r.admit_step + r.tokens.len() {
                continue;
            }
            let local_pos = t - r.admit_step;
            tokens[r.slot] = r.tokens[local_pos];
            positions[r.slot] = local_pos;
        }
        let inputs = bind_batched_step_staggered(
            &g,
            &weights,
            &tokens,
            &positions,
            cap,
            batch,
            &global_slotmap,
        );

        let mut cpu_inputs = inputs.clone();
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            cpu_inputs.insert(si, cpu_caches[ci].clone().into());
        }
        let cpu_eval = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("deepseek3 batched MLA decode CPU eval");
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
            .expect("deepseek3 batched MLA decode GPU dispatch")
            .read()
            .expect("read deepseek3 batched MLA decode output");
        let gpu_logits = f32_bytes(&gpu_bytes);

        assert_eq!(
            gpu_logits.len(),
            cpu_logits.as_f32().unwrap().len(),
            "logits length at step {t}"
        );
        for row in 0..batch {
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
            "deepseek3 batched MLA shared-pool GPU-vs-CPU diff {max_rel_seen} too large at step {t}"
        );
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    eprintln!(
        "deepseek3_mla_batched_kv_decode_gpu_matches_cpu_staggered: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {max_step} staggered engine steps, {batch} slots"
    );
}

/// Bind one shared-pool MLA prefill call over a deliberately non-contiguous physical slot layout. Inputs
/// are `Slot::Token`, the `[pool]` inverse `Slot::SlotMap` and the `mask.prefill` step input (card 550a),
/// as in `mixtral_moe_batched.rs`'s `bind_shared_pool_prefill`.
fn bind_shared_pool_prefill(
    g: &Graph,
    weights: &HashMap<String, HostTensor>,
    tokens: &[u32],
    inv: &[i32],
) -> HashMap<ValueId, Value> {
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
                unreachable!("unexpected slot {other:?} in shared-pool MLA prefill")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(
                    id,
                    weights
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("no weight for {name}"))
                        .into(),
                );
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

/// `poot_models::deepseek3::trace_deepseek3_prefill_kv_shared_pool` (spec 269 stage 5, card 274) on a real
/// GPU. Mirrors `mixtral_shared_pool_prefill_gpu_matches_cpu_noncontiguous`: non-contiguous physical slots
/// and `GpuExecutor::run_resident_kv` (the inverse-map `scatter_update` still goes through the resident
/// KV path), with this file's `cfg()`/`moe_params()`/`synth_weights()` fixture. Only GPU-vs-CPU agreement
/// on the same graph is checked. Skips without a Vulkan/wgpu adapter.
#[test]
fn deepseek3_shared_pool_prefill_gpu_matches_cpu_noncontiguous() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = cfg();
    let mp = moe_params();
    // Same non-contiguous layout as `poot_eval::tests::moe_paged_prefill`, but token ids stay below this
    // file's `cfg().vocab` (20); that module's `TOKENS` includes id 27, out of range for `embed_tokens` here.
    const TOKENS: [u32; 7] = [5, 11, 3, 16, 8, 1, 19];
    const GLOBAL_SLOTS: [usize; 7] = [3, 17, 1, 9, 14, 2, 11];
    const POOL_SLOTS: usize = 20;

    let n = TOKENS.len();
    let g = trace_deepseek3_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    g.validate()
        .expect("deepseek3 shared-pool MLA prefill graph should validate");
    let weights = synth_weights(&g);

    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let inputs = bind_shared_pool_prefill(&g, &weights, &TOKENS, &inv);

    let mut cpu_inputs = inputs.clone();
    for &(si, _) in &g.state {
        cpu_inputs.insert(si, HostTensor::zeros(g.aval(si).shape.clone()).into());
    }
    let cpu_logits = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("deepseek3 shared-pool MLA prefill CPU eval")
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
        .expect("deepseek3 shared-pool MLA prefill GPU dispatch")
        .read()
        .expect("read deepseek3 shared-pool MLA prefill output");
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
        "deepseek3 shared-pool MLA prefill GPU-vs-CPU diff {max_rel} too large"
    );
    eprintln!(
        "deepseek3_shared_pool_prefill_gpu_matches_cpu_noncontiguous: max rel diff vs CPU = {max_rel:.2e}"
    );
}
