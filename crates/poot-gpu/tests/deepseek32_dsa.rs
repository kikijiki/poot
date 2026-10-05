//! wgpu/RADV test for `poot_models::deepseek32::{trace_deepseek32_dsa_decode, trace_deepseek32_dsa_prefill}`.
//!
//! The DSA graphs are single-sequence and non-pooled (`Slot::{Token,Pos,SeqLen,Mask}` only), like
//! `crate::deepseek3`'s dense MLA tracers, so they dispatch through [`GpuExecutor::run`] (prefill, stateless)
//! and [`GpuExecutor::run_resident_kv`] (decode, one ping-pong KV pair per `g.state` entry), as
//! `deepseek3_mla_batched.rs` does. No hand-written binder is needed.
//!
//! Fixtures (`tiny_cfg`/`tiny_dcfg`/`tiny_mp`) and the `1e-4` relative tolerance follow
//! `poot_models::deepseek32`'s `tests::cpu_oracle`. The reference is `poot_eval::eval`/`eval_with_state` on
//! the identical graph and inputs, since the independent oracle already ran on CPU.

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
use poot_models::deepseek2::{DeepseekV2Config, deepseek2_rope_tables_interleaved};
use poot_models::deepseek3::DeepseekV3MoeParams;
use poot_models::deepseek32::{
    DsaConfig, trace_deepseek32_dsa_decode, trace_deepseek32_dsa_prefill,
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
    .expect("compile the dsa graph")
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

/// Split a `HashMap<ValueId, Tensor>` bind (built the same way the CPU oracle's is) into the
/// contract's two bind mechanisms: every `Storage::Const` becomes a named [`WeightStore`] entry,
/// every `Storage::Slot` becomes a [`StepInputs`] row (bound in the graph's own declared dtype -
/// exact i32 bytes for an I32-declared slot, matching `Tensor::int`'s `.ints` payload), and
/// `Storage::State` is left for the contract to zero-seed on first use.
fn split_consts_and_slots(
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Const => {
                let t = inputs
                    .get(&id)
                    .unwrap_or_else(|| panic!("missing const input {id:?}"));
                let name = m.name.clone().expect("named const");
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
            Storage::State => {}
            other => panic!("unexpected storage {other:?}"),
        }
    }
    (builder.build(), slots)
}

/// The per-step [`Storage::Slot`] rows of a `HashMap<ValueId, Value>` bind (built the same way the CPU
/// oracle's `inputs` is for one decode step): `Storage::Const`/`Storage::State` entries are skipped
/// (the consts are already loaded once outside the loop, and state is contract-carried).
fn decode_step_slot_binds(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<SlotBind> {
    g.inputs
        .iter()
        .filter_map(|&id| {
            let m = g.meta(id);
            if !matches!(m.storage, Storage::Slot(_)) {
                return None;
            }
            let v = inputs
                .get(&id)
                .unwrap_or_else(|| panic!("missing slot input {id:?}"));
            let t = v.as_host().expect("slot value is dense");
            let dtype = m.aval.dtype;
            let bytes: Vec<u8> = if dtype == DType::I32 {
                t.as_i32()
                    .expect("I32-declared slot needs an I32 HostTensor")
                    .iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect()
            } else {
                t.as_f32()
                    .unwrap()
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect()
            };
            Some(SlotBind {
                key: m.slot_key().unwrap().clone(),
                shape: t.shape().to_vec(),
                dtype,
                bytes,
            })
        })
        .collect()
}

/// Serialize GPU access within this test binary.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Identical to `poot_models::deepseek32`'s own `tests::tiny_cfg`.
fn tiny_cfg() -> DeepseekV2Config {
    DeepseekV2Config {
        vocab: 10,
        hidden: 8,
        layers: 2,
        n_heads: 2,
        // Real DeepSeek-V3.2 always sets q_lora_rank; the DSA indexer's wq_b consumes MLA's qr, which only
        // exists on this branch. Distinct from the other dims so a shape mixup cannot pass. Matches
        // `poot_models::deepseek32`'s `tests::tiny_cfg`.
        q_lora_rank: Some(5),
        kv_lora_rank: 4,
        qk_nope_head_dim: 3,
        qk_rope_head_dim: 2,
        v_head_dim: 3,
        eps: 1e-5,
        max_pos: 16,
        rope_theta: 10_000.0,
        yarn: None,
    }
}

/// Identical to `poot_models::deepseek32`'s own `tests::tiny_dcfg`.
fn tiny_dcfg(index_topk: usize) -> DsaConfig {
    DsaConfig {
        index_n_heads: 2,
        index_head_dim: 4,
        index_topk,
    }
}

/// Identical to `poot_models::deepseek32`'s own `tests::tiny_mp` (every layer dense; MoE FFN is covered by `crate::deepseek3`).
fn tiny_mp() -> DeepseekV3MoeParams {
    DeepseekV3MoeParams {
        n_routed_experts: 1,
        top_k: 1,
        moe_inter: 1,
        n_shared_experts: 0,
        dense_inter: 6,
        first_k_dense_replace: 2,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

use poot_test_util::seed_of;

/// Deterministic non-degenerate synthetic data - identical scheme to `deepseek3_mla_batched.rs`'s own
/// `fill()`.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let seed = seed as f32 * 1e-9;
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.0137 + seed * 0.911).sin() * 0.6)
        .collect()
}

/// Every `Const` a graph declares, filled by `(name, shape)` as in `deepseek3_mla_batched.rs`, except the
/// RoPE tables (`rope.cos`/`rope.sin`/`index.rope.cos`/`index.rope.sin`), which are real sinusoidal tables
/// from `deepseek2_rope_tables_interleaved` so the rotation stays orthonormal per position.
fn synth_weights(
    g: &Graph,
    cfg: &DeepseekV2Config,
    dcfg: &DsaConfig,
) -> HashMap<String, HostTensor> {
    let mut w = HashMap::new();
    let (rope_cos, rope_sin) = deepseek2_rope_tables_interleaved(
        cfg.max_pos,
        cfg.qk_rope_head_dim,
        cfg.rope_theta,
        cfg.yarn.as_ref(),
    );
    w.insert(
        "rope.cos".to_string(),
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim / 2], rope_cos),
    );
    w.insert(
        "rope.sin".to_string(),
        HostTensor::f32(vec![cfg.max_pos, cfg.qk_rope_head_dim / 2], rope_sin),
    );
    let (idx_cos, idx_sin) =
        deepseek2_rope_tables_interleaved(cfg.max_pos, dcfg.index_head_dim, cfg.rope_theta, None);
    w.insert(
        "index.rope.cos".to_string(),
        HostTensor::f32(vec![cfg.max_pos, dcfg.index_head_dim / 2], idx_cos),
    );
    w.insert(
        "index.rope.sin".to_string(),
        HostTensor::f32(vec![cfg.max_pos, dcfg.index_head_dim / 2], idx_sin),
    );
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

fn assert_matches(got: &[f32], want: &[f32], tol: f32, ctx: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{ctx}: shape mismatch");
    let mut max_rel = 0.0f32;
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{ctx}: non-finite logit at index {i}: GPU {a} vs CPU {b}"
        );
        let denom = a.abs().max(b.abs()).max(1e-6);
        let rel = (a - b).abs() / denom;
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tol,
            "{ctx}: index {i}: {a} vs {b} (rel {rel} > {tol})"
        );
    }
    max_rel
}

/// `trace_deepseek32_dsa_prefill` through [`GpuExecutor::run`] (stateless), compared against
/// `poot_eval::eval` on the identical graph and inputs. Skips without a Vulkan/wgpu adapter.
#[test]
fn deepseek32_dsa_prefill_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = tiny_cfg();
    let dcfg = tiny_dcfg(3); // sparse selection: index_topk (3) < seq_len (6)
    let mp = tiny_mp();
    let tokens: [usize; 6] = [3, 7, 1, 9, 5, 2];
    let l = tokens.len();

    let g = trace_deepseek32_dsa_prefill(cfg, dcfg, mp, l);
    g.validate()
        .expect("deepseek32 dsa prefill graph should validate");
    let weights = synth_weights(&g, &cfg, &dcfg);

    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let t = match &m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                vec![l],
                tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            ),
            Storage::Slot(Slot::Mask) => {
                let name = m.name.as_deref().expect("mask slot named");
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                let mut mask = vec![0.0f32; l * l];
                for i in 0..l {
                    for j in 0..l {
                        mask[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                    }
                }
                HostTensor::f32(m.aval.shape.clone(), mask)
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?} in dsa prefill"),
            Storage::Const => {
                let name = m.name.as_deref().expect("named const");
                weights
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("no weight for {name}"))
            }
            other => panic!("unexpected storage {other:?} in a stateless dsa prefill graph"),
        };
        inputs.insert(id, t);
    }

    let eval_inputs: HashMap<ValueId, Value> = inputs
        .iter()
        .map(|(id, t)| (*id, Value::from(t.clone())))
        .collect();
    let cpu_logits = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("dsa prefill CPU eval")
        .output
        .into_host()
        .expect("dsa prefill CPU eval output is dense");
    // Card 535b: the wgpu dispatch below runs a `compile`d graph, not the raw trace - `lower_nonlast_reduces`
    // (a `compile`-pipeline-only pass since card 534a) is what plans this DSA indexer's non-last-axis
    // `Reduce{axis:1}`; the CPU oracle stays on the raw graph (compile's passes are semantics-preserving).
    // `compile_staged` (Card 546b's contract entry) runs the identical pipeline `compile` did; only the
    // submission mode (now fixed at `Replay`) differs, and that never changes which passes run.
    let program = staged(&g, target);
    let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let gpu_bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .expect("dsa prefill GPU dispatch")
        .read()
        .expect("read dsa prefill output");
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let gpu_logits = f32_bytes(&gpu_bytes);

    let max_rel = assert_matches(
        &gpu_logits,
        cpu_logits.as_f32().unwrap(),
        1e-4,
        "dsa prefill",
    );
    eprintln!("deepseek32_dsa_prefill_gpu_matches_cpu: max rel diff vs CPU = {max_rel:.2e}");
}

/// `trace_deepseek32_dsa_decode`: the 6-step decode sequence of the CPU-oracle test through
/// [`GpuExecutor::run_resident_kv`] (one ping-pong KV pair per `g.state` entry, `3 * cfg.layers` here: the
/// third is the Lightning Indexer's `k_cache`), comparing each step's logits against `eval_with_state`.
/// Covers `pos + 1 < index_topk` (steps 0-1) and real sparse selection (steps 3-5). Skips without a
/// Vulkan/wgpu adapter.
#[test]
fn deepseek32_dsa_decode_gpu_matches_cpu_at_every_position() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = tiny_cfg();
    let dcfg = tiny_dcfg(3);
    let mp = tiny_mp();
    let tokens: [usize; 6] = [3, 7, 1, 9, 5, 2];
    let cap = tokens.len();

    let g = trace_deepseek32_dsa_decode(cfg, dcfg, mp, cap);
    g.validate()
        .expect("deepseek32 dsa decode graph should validate");
    assert_eq!(
        g.state.len(),
        3 * cfg.layers,
        "SC-002: 3 state tensors per layer"
    );
    let weights = synth_weights(&g, &cfg, &dcfg);

    let mut cpu_caches: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    // One entry for the whole decode loop: the contract zero-seeds its state on the first step, and
    // carries it across steps the same way `GpuExecutor::run_resident_kv`'s own `state_in`/returned
    // buffers did.
    let program = staged(&g, target);
    let store = {
        let mut builder = WeightStore::builder();
        for &id in &g.inputs {
            let m = g.meta(id);
            if let Storage::Const = m.storage {
                let name = m.name.as_deref().expect("named const");
                let t = weights
                    .get(name)
                    .unwrap_or_else(|| panic!("no weight for {name}"));
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
        }
        builder.build()
    };
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();

    let mut max_rel_seen = 0.0f32;
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            let t = match m.storage {
                Storage::Slot(Slot::Token) => HostTensor::i32(vec![], vec![tok as i32]),
                Storage::Slot(Slot::Pos) => HostTensor::i32(vec![], vec![pos as i32]),
                Storage::Slot(Slot::SeqLen) => HostTensor::i32(vec![], vec![(pos + 1) as i32]),
                Storage::Slot(Slot::Mask) => {
                    let cap_n = m.aval.shape.iter().product::<usize>();
                    let mask: Vec<f32> = (0..cap_n)
                        .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                        .collect();
                    HostTensor::f32(m.aval.shape.clone(), mask)
                }
                Storage::Slot(other) => unreachable!("unexpected slot {other:?} in dsa decode"),
                Storage::State => continue,
                Storage::Const => {
                    let name = m.name.as_deref().expect("named const");
                    weights
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("no weight for {name}"))
                }
                other => panic!("unexpected storage {other:?}"),
            };
            inputs.insert(id, t.into());
        }

        let mut cpu_inputs = inputs.clone();
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            cpu_inputs.insert(si, cpu_caches[ci].clone().into());
        }
        let r = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("dsa decode CPU eval");
        let cpu_logits = r
            .output
            .into_host()
            .expect("dsa decode CPU eval output is dense");
        cpu_caches = r
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dsa decode CPU eval state is dense"))
            .collect();

        let slot_binds = decode_step_slot_binds(&g, &inputs);
        let step_in = step_inputs(&slot_binds);
        let gpu_bytes = exec
            .step(exe, entry, &step_in, &mut NoSync)
            .expect("dsa decode GPU dispatch")
            .read()
            .expect("read dsa decode output");
        let gpu_logits = f32_bytes(&gpu_bytes);

        let rel = assert_matches(
            &gpu_logits,
            cpu_logits.as_f32().unwrap(),
            1e-4,
            &format!("dsa decode step {pos}"),
        );
        max_rel_seen = max_rel_seen.max(rel);
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    eprintln!(
        "deepseek32_dsa_decode_gpu_matches_cpu_at_every_position: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {} steps",
        tokens.len()
    );
}
