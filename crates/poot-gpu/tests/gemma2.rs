//! Gemma2 attention-logit softcap run end to end on a real GPU.
//!
//! No gemma2 checkpoint is available here, so this is a synthetic probe: real Gemma2-2B attention dims
//! (`head_dim=256`, `hidden_size=2304`, 8 query heads / 4 kv heads, as `Dense::new(Family::Gemma2)` configures them), seeded
//! random weights, a small vocab (softcap and flash prefill do not touch lm_head) and 2 layers (1 global,
//! 1 SWA).
//!
//! The graph fuses through `OpKind::FlashAttentionPrefill`, and the planner chooses the imported kernel
//! (`pootc/tests/kernels/flash_prefill.rs`) because the generated region prefill has no softcap (Card
//! 557). `head_dim=256` equals `FLASH_LDS_CAP`, the imported kernel's limit. This is the only GPU test
//! that runs the softcap branch (`if has_softcap_f > 0.5`) of that kernel; the other flash-prefill tests
//! use `has_softcap_f == 0.0`.
use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Executor, HostView, NoSync, StepInputs};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::op::{FusedOp, FusedOperand, OpKind, UnOp};
use poot_graph_ir::{Graph, Slot, SlotKey, Storage};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, ImportedKernel,
    KernelChoice, Partition, StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_graph_plan::{cse, fuse};
use poot_models::model::{LogitRows, Phase};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use poot_test_util::{assert_close_rel, max_abs_error};

/// Compile `g` for `target` with `Submission::Replay` (Card 546b: the contract admits no other
/// submission; fusion is submission-independent, so the structural assertions below see exactly the
/// same graph the old `Submission::Eager` compile produced).
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
    .expect("compile the gemma2 graph")
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

/// Bind every input deterministically (weights seeded by name so CPU and GPU binds agree bit-for-bit).
/// `include_state`: the CPU path (`eval_with_state`) wants `Storage::State` as a zero-seeded input; the
/// GPU path (`GpuExecutor::run_prefill`) zero-seeds state itself and expects `inputs` to hold the rest.
fn bind(
    g: &poot_graph_ir::Graph,
    n: usize,
    tokens: &[u32],
    include_state: bool,
) -> HashMap<usize, Value> {
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for &id in &g.inputs {
        let m = &g.values[id];
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                m.aval.shape.clone(),
                tokens.iter().map(|&t| t as i32).collect(),
            ),
            // Card 550: the global/SWA masks are now graph computations over `Slot::Pos` and `iota`;
            // this one-shot prefill always starts at position 0.
            Storage::Slot(Slot::Pos) => HostTensor::i32(vec![1, n], (0..n as i32).collect()),
            Storage::Const => {
                let name = m.name.as_deref().unwrap_or("");
                let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
                    (h ^ c as u64).wrapping_mul(1099511628211)
                });
                HostTensor::f32(m.aval.shape.clone(), fill(m.aval.numel(), seed))
            }
            Storage::State => {
                if !include_state {
                    continue;
                }
                HostTensor::zeros(m.aval.shape.clone())
            }
            _ => continue,
        };
        inputs.insert(id, t.into());
    }
    inputs
}

/// [`bind`]'s GPU-path split onto the contract: every `Storage::Const` input becomes a named
/// [`WeightStore`] entry (the identical seed-by-name data `bind` uploads for the CPU oracle, so the
/// two paths agree bit for bit), every `Storage::Slot` input becomes a [`StepInputs`] row, and
/// `Storage::State` is left out entirely - the contract zero-seeds a newly allocated state buffer on
/// first use, the same zero seed `GpuExecutor::run_prefill` used to upload by hand.
fn gpu_consts_and_slots(
    g: &poot_graph_ir::Graph,
    n: usize,
    tokens: &[u32],
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for &id in &g.inputs {
        let m = &g.values[id];
        match m.storage {
            Storage::Slot(Slot::Token) => {
                let bytes: Vec<u8> = tokens
                    .iter()
                    .flat_map(|&t| (t as i32).to_le_bytes())
                    .collect();
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: m.aval.shape.clone(),
                    dtype: DType::I32,
                    bytes,
                });
            }
            // Card 550: see `bind`'s identical `Slot::Pos` arm.
            Storage::Slot(Slot::Pos) => {
                let bytes: Vec<u8> = (0..n as i32).flat_map(|p| p.to_le_bytes()).collect();
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: vec![1, n],
                    dtype: DType::I32,
                    bytes,
                });
            }
            Storage::Slot(other) => panic!("unexpected slot {other:?} on gemma2 prefill"),
            Storage::Const => {
                let name = m.name.clone().unwrap();
                let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
                    (h ^ c as u64).wrapping_mul(1099511628211)
                });
                let data = fill(m.aval.numel(), seed);
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dense =
                    DenseWeight::try_new(DType::F32, m.aval.shape.clone(), Arc::from(bytes))
                        .unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
            // The RoPE tables are computed constants the executor materializes itself.
            Storage::State | Storage::Computed(_) => {}
            other => panic!("unexpected storage {other:?} on gemma2 prefill"),
        }
    }
    (builder.build(), slots)
}

/// Gemma2 attention-logit softcap (`Some(50.0)`, gemma2-2b's value) inside the imported flash-prefill
/// kernel on a GPU, compared to the CPU reference (materialized softcap, unoptimized graph). Skips without a GPU.
#[test]
fn gemma2_softcap_flash_prefill_gpu_matches_cpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    // Gemma 2 2B's real attention dims; the vocabulary (lm_head does not touch the kernel under test)
    // and the MLP width shrink for a fast probe.
    let cfg = Dense::new(Family::Gemma2)
        .vocab(256)
        .dims(2304, 256, 2) // 1 sliding-window + 1 global layer.
        .heads(8, 4)
        .head_dim(256)
        .max_positions(8192)
        .with("sliding_window", 4096)
        .with("attn_logit_softcapping", 50.0)
        .with("final_logit_softcapping", 30.0)
        .with("query_pre_attn_scalar", 256.0);
    assert_eq!(
        cfg.head_dim,
        Some(256),
        "gemma2-2b's real head_dim must stay the FLASH_LDS_CAP boundary"
    );

    let n = 24usize;
    let tokens: Vec<u32> = (0..n)
        .map(|i| (i * 7 + 3) as u32 % cfg.vocab as u32)
        .collect();
    let g = plain(
        cfg.f32_model()
            .model
            .trace(Phase::Prefill, step(1, n, n, LogitRows::Last))
            .unwrap(),
    );

    // Card 535b: `compile` (not the raw `optimize()`), like `run_prefill_runs_a_flash_fused_compiled_graph_and_matches_cpu`
    // (qwen2.rs) and `deepseek32_dsa_prefill_gpu_matches_cpu` - the same graph the contract steps below,
    // so the structural assertion below is checked on the actual artifact under test, not a stand-in for
    // it. `compile` runs strictly more than `optimize()` (legalize, dtype passes); this synthetic F32
    // graph needs none of that extra work, so the flash-fusion structure is unaffected.
    let program = staged(&g, target);
    let stage = program.stages().next().expect("single stage").2;
    let opt = stage.graph();
    // Confirm the softcap fusion fires on this graph (as in the CPU-only test in poot-models).
    let flash_eqns: Vec<&poot_graph_ir::Eqn> = opt
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionPrefill { .. }))
        .collect();
    assert_eq!(
        flash_eqns.len(),
        cfg.layers,
        "expected one fused flash-prefill op per layer"
    );
    for eqn in &flash_eqns {
        let OpKind::FlashAttentionPrefill { softcap, .. } = eqn.op else {
            unreachable!()
        };
        assert_eq!(softcap, Some(50.0), "fused op must carry the real softcap");
        // Card 557: the planner records the kernel; the generated region prefill has no softcap, so
        // a softcapped prefill takes the imported kernel card 198 edited.
        assert!(
            matches!(
                stage.kernel_choice(eqn),
                KernelChoice::Imported {
                    kernel: ImportedKernel::FlashPrefill,
                    ..
                }
            ),
            "a softcapped head_dim=256 prefill should plan the imported flash prefill, got {:?}",
            stage.kernel_choice(eqn)
        );
    }

    let cpu_inputs = bind(&g, n, &tokens, true);
    let cpu_eval =
        eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).expect("CPU oracle eval");
    let cpu_logits = cpu_eval.output.into_host().expect("dense output");
    let _cpu_state: Vec<HostTensor> = cpu_eval
        .state
        .into_iter()
        .map(Value::into_host)
        .collect::<Result<Vec<_>, _>>()
        .expect("dense state");

    // card 535b: the contract's own `add_entry`/`step` compiles nothing itself (already done above into
    // `program`), so the caller binds the consts/slots directly against the compiled `opt` graph;
    // `compile` preserves `g`'s input/const identity, so binding against it stays valid. Card 546a's
    // contract zero-seeds a newly allocated state buffer on first use (`Engine`'s own state arena), the
    // same zero seed `GpuExecutor::run_prefill` used to upload by hand.
    let (store, slot_binds) = gpu_consts_and_slots(&g, n, &tokens);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let inputs = step_inputs(&slot_binds);
    let gpu_bytes = exec
        .step(exe, entry, &inputs, &mut NoSync)
        .expect("contract step on the pre-fused (flash-attention) graph")
        .read()
        .expect("read the contract's prefill output");
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let gpu_logits = f32_bytes(&gpu_bytes);

    assert_eq!(g.aval(g.output).shape, cpu_logits.shape());
    assert_eq!(gpu_logits.len(), cpu_logits.as_f32().unwrap().len());
    assert_eq!(cpu_logits.shape(), vec![1, 1, cfg.vocab]);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in gpu_logits.iter().zip(cpu_logits.as_f32().unwrap().iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite logit: GPU {a} vs CPU {b}"
        );
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(1e-3);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    eprintln!(
        "gemma2_softcap_flash_prefill_gpu_matches_cpu: max_abs={max_abs:.3e} max_rel={max_rel:.3e} \
         (n={n}, layers={}, head_dim={:?})",
        cfg.layers, cfg.head_dim
    );
    assert!(
        max_rel <= 5e-3,
        "GPU softcapped flash-prefill vs CPU materialized softcap reference: max relative diff \
         {max_rel} too large"
    );
}

/// Bind a decode graph's one-element slots (`Token`/`Pos`) and Const inputs with deterministic
/// pseudo-random data, keyed by input index. `Storage::State` (the KV cache) is deliberately left
/// unbound here: the resident executor reads carried state from its own `state_in: &[DeviceBuffer]`
/// parameter, not the host `inputs` map (same convention as `poot-gpu/tests/qwen2.rs`'s
/// `qwen2_decode_kv_gpu_resident_matches_cpu`); the CPU oracle side inserts it separately as plain
/// tensors for `eval_with_state`.
fn bind_decode(g: &poot_graph_ir::Graph, pos: usize, token: u32) -> HashMap<usize, Value> {
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let m = g.meta(id);
        match m.storage {
            Storage::State => continue,
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(m.aval.shape.clone(), vec![token as i32]).into(),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(
                    id,
                    HostTensor::i32(m.aval.shape.clone(), vec![pos as i32]).into(),
                );
            }
            Storage::Slot(other) => panic!("unexpected slot {other:?} on gemma2 decode"),
            _ => {
                inputs.insert(
                    id,
                    HostTensor::f32(m.aval.shape.clone(), fill(m.aval.numel(), 100 + i as u64))
                        .into(),
                );
            }
        }
    }
    inputs
}

/// [`bind_decode`]'s GPU-path split onto the contract: slots (`Token`/`Pos`) and every
/// other non-state input keyed by input index, same as `bind_decode`, routed to a [`WeightStore`] entry
/// (`Storage::Const`) or a [`StepInputs`] row (`Storage::Slot`). `Storage::State` is left out: the
/// contract zero-seeds a newly allocated state buffer on first use, the same zero seed this test used to
/// upload to `state_in` by hand.
fn gpu_consts_and_slots_decode(
    g: &poot_graph_ir::Graph,
    pos: usize,
    token: u32,
) -> (WeightStore, Vec<SlotBind>) {
    let mut builder = WeightStore::builder();
    let mut slots = Vec::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let m = g.meta(id);
        match m.storage {
            Storage::State => continue,
            Storage::Slot(kind) => {
                let scalar = match kind {
                    Slot::Token => token as i32,
                    Slot::Pos => pos as i32,
                    other => panic!("unexpected slot {other:?} on gemma2 decode"),
                };
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: m.aval.shape.clone(),
                    dtype: DType::I32,
                    bytes: scalar.to_le_bytes().to_vec(),
                });
            }
            Storage::Const => {
                let name = m.name.clone().unwrap();
                let data = fill(m.aval.numel(), 100 + i as u64);
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dense =
                    DenseWeight::try_new(DType::F32, m.aval.shape.clone(), Arc::from(bytes))
                        .unwrap();
                builder.insert(name, WeightEntry::Dense(dense)).unwrap();
            }
            Storage::Computed(_) => {}
            other => panic!("unexpected storage {other:?} on gemma2 decode"),
        }
    }
    (builder.build(), slots)
}

/// Steps `g` once on a fresh executable of its own (so its KV state starts zero regardless of what any
/// other entry on `exec` has done - Card 546b's state-by-name sharing only applies within
/// one executable, and `base`/`fg` below declare the identical state names since `fuse` never renames
/// state, so without this isolation the second call would read the first call's already-advanced
/// state instead of a fresh zero one).
fn run_decode_once(
    exec: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    store: &Arc<WeightStore>,
    binds: &[SlotBind],
) -> Vec<f32> {
    let program = staged(g, target);
    let exe = exec
        .load_weights(Arc::clone(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let inputs = step_inputs(binds);
    let bytes = exec
        .step(exe, entry, &inputs, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    f32_bytes(&bytes)
}

/// Does this fused region carry the tanh-approximate GELU (`ops::gelu`) of a GeGLU MLP: its `exp` step and
/// its `0.044715` cubic coefficient. Both are required, so splitting the chain at `exp` (a region that
/// keeps only the polynomial half) is not counted.
fn carries_gelu(region: &poot_graph_ir::op::FusedRegion) -> bool {
    let has_exp = region
        .steps
        .iter()
        .any(|s| matches!(s.op, FusedOp::Unary(UnOp::Exp)));
    let has_coeff = region.steps.iter().any(|s| {
        s.inputs
            .contains(&FusedOperand::Lit(poot_graph_ir::Scalar::F32(0.044715)))
    });
    has_exp && has_coeff
}

/// Card 536b (R481-007) SC-003: Gemma2's GeGLU MLP (`Gelu(gate) * up`, `ops::geglu`) is exactly the
/// pointwise chain this card's fusion-legality fix newly admits into one region - `is_fusable` refused
/// `Gelu` before, so every GeGLU MLP paid for two dispatches (the `Gelu` and the `Mul`) instead of one.
/// Real device regression: fuse the decode graph, confirm a `Fused` region actually carries the `Gelu`
/// step (so this row proves the fix, not merely "fusion happens somewhere"), then run both the fused and
/// un-fused graphs on the wgpu Arc through the stateful resident KV path (real carried KV cache, zero
/// seeded) and compare dense-decode logits against each other and against the CPU oracle, within tier 2.
/// Mirrors `poot-gpu/tests/qwen2.rs`'s `fused_decode_matches_unfused_on_gpu` and
/// `qwen2_decode_kv_gpu_resident_matches_cpu`.
///
/// Mutation (recorded in the card's landing note, never left in the tree): drop `Exp` from
/// `poot_graph_plan::FUSABLE_FLOAT_UNARY_OPS` (`fuse.rs`) - the GeGLU region assertion below
/// fails immediately (the GELU chain splits at its `exp`, so no `Fused` region carries the whole GELU), before either device even runs.
#[test]
fn gemma2_geglu_fused_decode_matches_unfused_and_cpu_on_gpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));

    let cfg = Dense::new(Family::Gemma2)
        .vocab(64)
        .dims(64, 128, 4)
        .max_positions(32)
        .with("sliding_window", 4)
        .with("attn_logit_softcapping", 50.0)
        .with("final_logit_softcapping", 30.0)
        .with("query_pre_attn_scalar", serde_json::Value::Null);
    let pos = 3usize;
    let cap = 8usize;
    let g = plain(
        cfg.f32_model()
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap(),
    );
    let inputs = bind_decode(&g, pos, 5);

    let base = cse(&g); // the resident executor's normal pre-pass; fusion groups on top of it.
    let fg = fuse(&base);
    fg.validate().expect("fused graph valid");
    let geglu_region_count = fg
        .eqns
        .iter()
        .filter(|e| match &e.op {
            OpKind::Fused(region) => carries_gelu(region),
            _ => false,
        })
        .count();
    assert_eq!(
        geglu_region_count,
        cfg.layers,
        "expected one Fused region carrying the GeGLU Gelu step per layer: {} eqns {:?}",
        fg.eqns.len(),
        fg.eqns.iter().map(|e| e.op.name()).collect::<Vec<_>>()
    );
    assert!(
        fg.eqns.len() < base.eqns.len(),
        "fewer dispatches after fusion: {} -> {}",
        base.eqns.len(),
        fg.eqns.len()
    );

    // Real carried KV cache, zero-seeded (this is the graph's first decode step in isolation - `pos`
    // names where in the RoPE table/valid-prefix slice the step reads, not how many prior steps ran).
    let zero_state: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    let mut cpu_inputs = inputs.clone();
    for (i, &(si, _)) in g.state.iter().enumerate() {
        cpu_inputs.insert(si, zero_state[i].clone().into());
    }
    let cpu_eval = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let cpu = cpu_eval.output.into_host().unwrap();
    let _cpu_new_state: Vec<HostTensor> = cpu_eval
        .state
        .into_iter()
        .map(Value::into_host)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // `base`/`fg` keep `g`'s own input identity (cse/fuse restructure equations, never input ids), so
    // the consts/slots computed from `g` bind either graph unchanged.
    let (store, slot_binds) = gpu_consts_and_slots_decode(&g, pos, 5);
    let store = Arc::new(store);
    let unfused_gpu = run_decode_once(&mut *exec, target, &base, &store, &slot_binds);
    let fused_gpu = run_decode_once(&mut *exec, target, &fg, &store, &slot_binds);

    assert_eq!(fused_gpu.len(), unfused_gpu.len());
    assert_eq!(fused_gpu.len(), cpu.as_f32().unwrap().len());
    // Fusion moves region intermediates from f32 global buffers to registers, so fused vs un-fused is a
    // tight fp rounding difference (tier 1 territory), while GPU vs CPU is the ordinary tier-2 gap.
    let fused_vs_unfused = max_abs_error(&fused_gpu, &unfused_gpu);
    assert!(
        fused_vs_unfused <= 1e-4,
        "fused GPU decode must match un-fused within fp tolerance: max_abs={fused_vs_unfused:.3e}"
    );
    assert_close_rel(&fused_gpu, cpu.as_f32().unwrap(), 5e-3);
    eprintln!(
        "gemma2_geglu_fused_decode_matches_unfused_and_cpu_on_gpu: fused/unfused max_abs={:.3e}, \
         fused/cpu max_abs={:.3e}",
        fused_vs_unfused,
        max_abs_error(&fused_gpu, cpu.as_f32().unwrap())
    );
}
