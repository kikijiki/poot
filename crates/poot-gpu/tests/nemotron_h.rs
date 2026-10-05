//! Real-hardware (wgpu/RADV) check for `poot_models::nemotron_h::{trace_nemotron_h_prefill,
//! trace_nemotron_h_decode}` against the CPU eager executor. Same approach as
//! `crates/poot-gpu/tests/deepseek32_dsa.rs`.
//!
//! Both Nemotron-H graphs are single-sequence and non-pooled (`Slot::{Token,Pos,SeqLen,Mask}` only), so
//! they dispatch through [`GpuExecutor::run`] (prefill, stateless) and
//! [`GpuExecutor::run_resident_kv`] (decode, one ping-pong state pair per `Graph::state` entry) with no
//! hand-written binder, like `deepseek32_dsa.rs`.
//!
//! The decode graph mixes three state shapes in one `run_resident_kv` call: the Mamba2 conv ring cache
//! `[1,K-1,ConvC]`, the SSM state `[1,Hq,N,P]`, and the attention `[1,Akv,cap,D]` K/V pair. The Mamba
//! ops (`poot_graph_ir::ops::{mamba2_ssd_decode, mamba2_ssd_prefill, causal_conv1d_decode,
//! causal_conv1d_prefill}`) are plain primitive compositions.
//!
//! Fixtures reuse the toy config and `"M-M-M-M*-"` pattern (the first nine characters of the real 8B
//! `hybrid_override_pattern`, so all three layer kinds appear) from `poot_models::nemotron_h`'s CPU-oracle
//! test. The reference is `poot_eval::eval`/`eval_with_state` on the identical graph and inputs, at `1e-4`
//! relative tolerance.

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
use poot_models::nemotron_h::{
    NemotronHAttnConfig, NemotronHConfig, NemotronHMambaConfig, parse_hybrid_pattern,
    trace_nemotron_h_decode, trace_nemotron_h_prefill,
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
    .expect("compile the nemotron_h graph")
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

/// Serializes GPU access within this test binary.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// First nine characters of the real `nvidia/Nemotron-H-8B-Base-8K` `hybrid_override_pattern`; same as
/// `poot_models::nemotron_h`'s `tests::TOY_PATTERN`. Covers all three layer kinds (`M` Mamba2, `-`
/// squared-ReLU MLP, `*` NoPE GQA attention).
const TOY_PATTERN: &str = "M-M-M-M*-";

/// Same as `poot_models::nemotron_h`'s whole-model CPU-oracle fixture. Every dim is distinct so a shape
/// mixup cannot silently pass, and `n_groups = 2 < mamba_num_heads = 4` keeps the `repeat_kv` B/C group
/// broadcast non-trivial.
fn tiny_cfg() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 12,
        hidden: 6,
        mlp_inter: 10,
        eps: 1e-6,
        pattern: parse_hybrid_pattern(TOY_PATTERN),
        mamba: NemotronHMambaConfig {
            hidden: 6,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            ssm_state: 3,
            conv_kernel: 3,
        },
        attn: NemotronHAttnConfig {
            hidden: 6,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
        },
    }
}

use poot_test_util::seed_of;

/// Deterministic non-degenerate synthetic data.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let seed = seed as f32 * 1e-9;
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.0137 + seed * 0.911).sin() * 0.6)
        .collect()
}

/// Every `Const` a graph declares, deterministically filled by `(name, shape)`, except:
/// - the Mamba per-head `a`, which is always strictly negative in a real checkpoint (`A = -exp(A_log)`);
///   a positive `a` makes `exp(delta * a)` blow up over `L` positions, so the magnitude is negated.
///   (The prefill causal mask and the SSD tril are step inputs / in-graph computations since card
///   550a, so they are no longer consts this fn fills.)
fn synth_weights(g: &Graph) -> HashMap<String, HostTensor> {
    let mut w = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        if let Storage::Const = m.storage {
            let name = m.name.clone().expect("named const");
            w.entry(name.clone()).or_insert_with(|| {
                if name.ends_with(".mixer.a") {
                    let data = fill(m.aval.numel(), seed_of(&name))
                        .into_iter()
                        .map(|v| -v.abs() - 0.25)
                        .collect();
                    HostTensor::f32(m.aval.shape.clone(), data)
                } else {
                    HostTensor::f32(m.aval.shape.clone(), fill(m.aval.numel(), seed_of(&name)))
                }
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

/// Every eqn of both Nemotron-H graphs plans to a real SPIR-V kernel (or a zero-dispatch view), with no
/// planning error and no host fallback (the planner has no host plan: an unplannable eqn is a typed
/// refusal). Runs without a GPU.
///
/// The `>= 100` Compute floor (the toy config plans 366 prefill / 275 decode dispatches) pins that most
/// of the graph is real dispatched kernels rather than views.
#[test]
fn nemotron_h_graphs_plan_to_spirv_kernels_with_no_host_fallback() {
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    use poot_test_util::graph_fixtures::plan_eqn;

    let cfg = tiny_cfg();
    for (label, g) in [
        ("prefill", trace_nemotron_h_prefill(&cfg, 5)),
        ("decode", trace_nemotron_h_decode(&cfg, 5)),
    ] {
        g.validate()
            .unwrap_or_else(|e| panic!("nemotron_h {label} graph should validate: {e:?}"));
        // Every backend folds `Iota` equations into computed inputs before planning (card 550a's
        // in-graph tril range); plan what the pipeline actually plans.
        let g = poot_graph_plan::fold_iota(&g);
        let mut compute = 0usize;
        for eqn in &g.eqns {
            let out_shape = g.aval(eqn.out).shape.clone();
            let plan = plan_eqn(
                &g,
                eqn,
                Backend::SpirvVulkan,
                &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "nemotron_h {label}: {:?} eqn -> {out_shape:?} has no SPIR-V plan: {e:?}",
                    eqn.op
                )
            });
            match plan {
                Plan::Compute { .. } | Plan::ComputeMeta { .. } | Plan::ComputeChunks(_) => {
                    compute += 1
                }
                Plan::Alias(_) | Plan::View { .. } => {}
                Plan::Collective { .. } => panic!(
                    "nemotron_h {label}: unexpected multi-rank collective {:?} in a single-device graph",
                    eqn.op
                ),
            }
        }
        assert!(
            compute >= 100,
            "nemotron_h {label}: only {compute} real SPIR-V compute dispatches planned - the graph \
             collapsed to views/aliases, so the GPU-vs-CPU tests above would be comparing nothing"
        );
        eprintln!("nemotron_h {label}: {compute} SPIR-V compute dispatches planned");
    }
}

/// Dispatches `trace_nemotron_h_prefill` through [`GpuExecutor::run`] (stateless; the final SSM/conv
/// states are discarded by the top-level trace) and compares against `poot_eval::eval` on the identical
/// graph and inputs. Skips if no Vulkan/wgpu adapter is present.
#[test]
fn nemotron_h_prefill_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = tiny_cfg();
    let tokens: [usize; 5] = [1, 4, 7, 10, 2];
    let l = tokens.len();

    let g = trace_nemotron_h_prefill(&cfg, l);
    g.validate()
        .expect("nemotron_h prefill graph should validate");
    let weights = synth_weights(&g);

    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let t = match &m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                vec![l],
                tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            ),
            Storage::Slot(Slot::Mask) => {
                // The `mask.prefill` step input (card 550a): the same additive causal content the
                // CPU and GPU sides both bind (this test feeds one input map to both).
                let name = m.name.as_deref().expect("mask slot named");
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                let mut mask = vec![0.0f32; l * l];
                for t in 0..l {
                    for j in 0..l {
                        mask[t * l + j] = if j <= t { 0.0 } else { -1.0e9 };
                    }
                }
                HostTensor::f32(m.aval.shape.clone(), mask)
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?} in nemotron_h prefill"),
            Storage::Const => {
                let name = m.name.as_deref().expect("named const");
                weights
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("no weight for {name}"))
            }
            other => panic!("unexpected storage {other:?} in a stateless nemotron_h prefill graph"),
        };
        inputs.insert(id, t);
    }

    let eval_inputs: HashMap<ValueId, Value> = inputs
        .iter()
        .map(|(id, t)| (*id, Value::from(t.clone())))
        .collect();
    let cpu_logits = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("nemotron_h prefill CPU eval")
        .output
        .into_host()
        .expect("nemotron_h prefill CPU eval output is dense");
    let program = staged(&g, target);
    let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let gpu_bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .expect("nemotron_h prefill GPU dispatch")
        .read()
        .expect("read nemotron_h prefill output");
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let gpu_logits = f32_bytes(&gpu_bytes);

    let max_rel = assert_matches(
        &gpu_logits,
        cpu_logits.as_f32().unwrap(),
        1e-4,
        "nemotron_h prefill",
    );
    eprintln!("nemotron_h_prefill_gpu_matches_cpu: max rel diff vs CPU = {max_rel:.2e}");
}

/// Drives the CPU oracle's 5-step decode sequence through [`GpuExecutor::run_resident_kv`], comparing
/// each step's GPU logits against `eval_with_state`. `Graph::state` carries three distinct shapes (conv
/// ring cache, SSM state, attention K/V), each of which must survive the device-resident ping-pong, so a
/// mis-swapped Mamba carrier shows up as divergence at step 1. Skips if no Vulkan/wgpu adapter is present.
#[test]
fn nemotron_h_decode_gpu_matches_cpu_at_every_position() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = tiny_cfg();
    let tokens: [usize; 5] = [1, 4, 7, 10, 2];
    let cap = tokens.len();

    let g = trace_nemotron_h_decode(&cfg, cap);
    g.validate()
        .expect("nemotron_h decode graph should validate");
    // 2 state tensors per Mamba layer (conv ring cache + SSM state), 2 per attention layer (K/V), 0 per
    // MLP layer: the "M-M-M-M*-" toy pattern has 4 Mamba and 1 attention layer.
    assert_eq!(
        g.state.len(),
        10,
        "4 Mamba layers (conv cache + SSM state each) + 1 attention layer (K + V)"
    );
    let weights = synth_weights(&g);

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
                Storage::Slot(other) => {
                    unreachable!("unexpected slot {other:?} in nemotron_h decode")
                }
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
            .expect("nemotron_h decode CPU eval");
        let cpu_logits = r
            .output
            .into_host()
            .expect("nemotron_h decode CPU eval output is dense");
        cpu_caches = r
            .state
            .into_iter()
            .map(|v| {
                v.into_host()
                    .expect("nemotron_h decode CPU eval state is dense")
            })
            .collect();

        let slot_binds = decode_step_slot_binds(&g, &inputs);
        let step_in = step_inputs(&slot_binds);
        let gpu_bytes = exec
            .step(exe, entry, &step_in, &mut NoSync)
            .expect("nemotron_h decode GPU dispatch")
            .read()
            .expect("read nemotron_h decode output");
        let gpu_logits = f32_bytes(&gpu_bytes);

        let rel = assert_matches(
            &gpu_logits,
            cpu_logits.as_f32().unwrap(),
            1e-4,
            &format!("nemotron_h decode step {pos}"),
        );
        max_rel_seen = max_rel_seen.max(rel);
    }
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    eprintln!(
        "nemotron_h_decode_gpu_matches_cpu_at_every_position: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {} steps",
        tokens.len()
    );
}
