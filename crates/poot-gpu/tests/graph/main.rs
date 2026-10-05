//! Run a real decomposed transformer op (RMSNorm) on the GPU graph executor and check it matches the
//! CPU eager reference (executor equivalence on the GPU path). Skips if no Vulkan adapter.
mod bf16_carrier;
mod dense_gemv;
mod flash;
mod gather_scatter;
mod gemm;
mod legalize;
mod lora;
mod misc;
mod moe;
mod norms_rope;
mod packed_kv;
mod plan_selection;
mod prefill_qwen3next;
mod ssm_linear_attn;
mod tiled_dequant;
mod vision;

use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::OpKind;
use poot_graph_ir::Storage;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{attention_masked, rmsnorm};
use poot_graph_ir::types::TensorType;
use poot_graph_plan::Target;
use poot_tensor::HostTensor;
use poot_test_util::{assert_close, max_abs_error};

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Card 546b: the executor-contract replacement for `open_or_skip(DeviceBackend::Wgpu,
/// GpuExecutor::new())`. The device's `target()` must be read before it is moved into the `Engine`.
fn open_engine_or_skip() -> Option<(Engine<WgpuDevice>, Target)> {
    let device = poot_test_util::device_skip::open_or_skip(
        poot_runtime_common::DeviceBackend::Wgpu,
        WgpuDevice::new(),
    )?;
    let target = device.target();
    Some((Engine::new(device), target))
}

/// Card 546b: `GpuExecutor::run`'s executor-contract replacement for this crate's hand-built test
/// graphs. Every `inputs` row is either a `Storage::Const` (becomes a named `ConstFixture`, keyed by
/// `meta.name`) or a `Storage::Slot` (becomes a keyed `StepFixture`); `poot_executor_parity::run_once`
/// binds, compiles, steps and tears the entry back down in one call, the same one-shot shape
/// `GpuExecutor::run` had. The returned bytes become a `HostTensor` in `g.output`'s own declared
/// shape/dtype, so every existing CPU-oracle comparison against `got.shape()`/`got.as_f32()` is unchanged.
fn run_via_contract(
    exec: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
) -> HostTensor {
    let mut consts = Vec::new();
    let mut slots = Vec::new();
    for (&id, t) in inputs {
        let meta = g.meta(id);
        if meta.storage == Storage::State {
            // `add_entry`'s own `state_buffer` zero-allocates a fresh state buffer on first use
            // (`Context::alloc_storage`'s "zero-filled" guarantee), identical to the zero seed every
            // caller here supplies for a graph's first (and, for `run_once`, only) step - so this input
            // needs no bind. A caller that ever needs a genuinely non-zero seeded state must drive
            // `Engine::add_entry`/`step` directly (see `gather_scatter.rs`'s priming-entry pattern).
            assert!(
                t.as_f32().unwrap().iter().all(|&v| v == 0.0),
                "run_via_contract: {id:?} is Storage::State with a non-zero seed; this adapter only \
                 supports the contract's zero-initialized state, not an injected one"
            );
            continue;
        }
        let dtype = meta.aval.dtype;
        // The executor contract is strictly typed: a bound tensor carries exactly the declared dtype
        // (a BF16/F16 const is built from its words, never narrowed from f32 here).
        assert_eq!(
            t.dtype(),
            dtype,
            "run_via_contract: {id:?} is bound as {:?} but declared {dtype:?}",
            t.dtype()
        );
        match meta.storage {
            Storage::Const => {
                let name: &'static str = Box::leak(meta.name.clone().unwrap().into_boxed_str());
                consts.push(poot_executor_parity::ConstFixture {
                    name,
                    tensor: t.clone(),
                });
            }
            Storage::Slot(_) => {
                let key = meta.slot_key().unwrap().clone();
                slots.push(poot_test_util::StepFixture {
                    key,
                    tensor: t.clone(),
                });
            }
            other => panic!(
                "run_via_contract: {id:?} has unsupported storage {other:?} for a test input"
            ),
        }
    }
    poot_executor_parity::run_once(exec, target, g, &consts, &slots)
        .unwrap_or_else(|e| panic!("{e}"))
}

fn staged_replay(
    g: &poot_graph_ir::Graph,
    target: Target,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    staged_replay_with(g, target, poot_graph_plan::FusionPolicy::Full)
}

/// [`staged_replay`] under a chosen fusion policy (Card 557: `FusionPolicy::MoeHangGuard` keeps a
/// matmul off the generated tiled GEMM, the one way to plan the imported tiled GEMM for a comparison).
fn staged_replay_with(
    g: &poot_graph_ir::Graph,
    target: Target,
    fusion: poot_graph_plan::FusionPolicy,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let gc = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &gc,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap()
}

/// Card 546b: `GpuExecutor::run_resident`'s executor-contract replacement for a perf microbench that
/// needs its (const-only) inputs uploaded once and the compiled entry kept live across many timed
/// `step` calls - what `run_resident`'s "far fewer host copies than `run`" bought on the old executor.
/// Every row of `inputs` must be `Storage::Const` (these microbenches bind the same consts to every
/// call; a slot-bearing graph needs its own `add_entry`/`step` loop, as the decode fixtures already
/// do). Returns the live `(ExecutableId, EntryId)`; the caller steps with `StepInputs::new()` in its
/// own timing loop via [`step_resident`] and is responsible for `remove_entry`/`unload` when done.
fn add_resident_entry(
    exec: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
) -> (poot_executor::ExecutableId, poot_executor::EntryId) {
    add_resident_entry_with(exec, target, g, inputs, poot_graph_plan::FusionPolicy::Full)
}

/// [`add_resident_entry`] compiled under a chosen fusion policy (see [`staged_replay_with`]).
fn add_resident_entry_with(
    exec: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
    fusion: poot_graph_plan::FusionPolicy,
) -> (poot_executor::ExecutableId, poot_executor::EntryId) {
    let mut builder = poot_quant::weights::WeightStore::builder();
    for (&id, t) in inputs {
        let meta = g.meta(id);
        assert!(
            matches!(meta.storage, Storage::Const),
            "add_resident_entry: {id:?} is not Storage::Const"
        );
        let name = meta.name.clone().unwrap();
        let bytes: Vec<u8> = t
            .as_f32()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let dense = poot_quant::weights::DenseWeight::try_new(
            DType::F32,
            t.shape().to_vec(),
            Arc::from(bytes),
        )
        .unwrap();
        builder
            .insert(name, poot_quant::weights::WeightEntry::Dense(dense))
            .unwrap();
    }
    let exe = exec
        .load_weights(
            Arc::new(builder.build()),
            poot_executor::WeightSource::ConstNames,
        )
        .unwrap();
    let staged = staged_replay_with(g, target, fusion);
    let entry = exec.add_entry(exe, &staged).unwrap();
    (exe, entry)
}

/// One replayed step of `entry` with no slot inputs, read back as a `Tensor` shaped like `g.output`:
/// the `run_resident`-shaped read for a caller using [`add_resident_entry`].
fn step_resident(
    exec: &mut dyn Executor,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    g: &poot_graph_ir::Graph,
) -> HostTensor {
    let bytes = exec
        .step(
            exe,
            entry,
            &poot_executor::StepInputs::new(),
            &mut poot_executor::NoSync,
        )
        .unwrap()
        .read()
        .unwrap();
    let out = g.aval(g.output);
    HostTensor::f32(
        out.shape.clone(),
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn build_rope_decomposition(
    hq: usize,
    d: usize,
    rot: usize,
) -> (
    poot_graph_ir::Graph,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    use poot_graph_ir::op::{BinOp, UnOp};
    let b = Builder::new();
    let half = rot / 2;
    let last = 3usize;
    let x = b.constant("x", TensorType::f32(vec![1, hq, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![rot]));
    let sin = b.constant("sin", TensorType::f32(vec![rot]));
    let xr = if rot < d { b.slice(x, last, 0, rot) } else { x };
    let x1 = b.slice(xr, last, 0, half);
    let x2 = b.slice(xr, last, half, rot);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rh = b.concat(last, &[neg_x2, x1]);
    let xc = b.binary(BinOp::Mul, xr, cos);
    let rs = b.binary(BinOp::Mul, rh, sin);
    let rotated = b.binary(BinOp::Add, xc, rs);
    let out = if rot < d {
        let xp = b.slice(x, last, rot, d);
        b.concat(last, &[rotated, xp])
    } else {
        rotated
    };
    let (xi, ci, si) = (x.id, cos.id, sin.id);
    (b.finish(out), xi, ci, si)
}

/// Declare an f32 constant and record `(id, shape)` so the test can fill its data afterwards.
fn vconst(
    b: &Builder,
    binds: &mut Vec<(poot_graph_ir::ValueId, Vec<usize>)>,
    name: &str,
    shape: Vec<usize>,
) -> poot_graph_ir::builder::Traced {
    let t = b.constant(name, TensorType::f32(shape.clone()));
    binds.push((t.id, shape));
    t
}

// --- card 158: batched-prefill-only ops, GPU vs CPU, at the real chunk size C=64 ---
//
// generate_prefill on the real Qwen3.6-35B degenerated into repeating garbage after ~5 tokens. The CPU oracle (poot-eval's
// `gdn_prefill_chunked_matches_decode_recurrence_heavy_pad` / `qwen3next_prefill_trace_matches_decode_heavy_pad`) proved
// the graph and math correct at the real config (GDN chunk C=64, heavy zero-padding), so the bug is GPU-specific. These
// tests run each prefill-only op standalone on the GPU against CPU `eval` on the same graph. Landmines in this 64-wide
// regime: the >64-lane cross-wave LDS cap (barrier-in-loop-crashes-spirv) and the tiled-GEMM high-workgroup-count race
// (tiled-gemm-large-workgroup-race). A real divergence would be gross (like the card-128 ROCm dispatch bugs,
// max_abs~156), so the tolerance is generous; the printed max_abs_err is the diagnostic.

/// Element-wise GPU-vs-CPU comparison with a diagnostic print for the card-158 probes. `tol = mul * max(|cpu_elem|, floor)`
/// per element; prints `op` and the overall max abs error first.
fn assert_prefill_op_gpu_matches_cpu(
    op: &str,
    tag: &str,
    got: &[f32],
    cpu: &[f32],
    mul: f32,
    floor: f32,
) {
    assert_eq!(got.len(), cpu.len(), "{op} [{tag}]: length mismatch");
    let max_err = max_abs_error(got, cpu);
    println!("{op} [{tag}]: max_abs_err={max_err:.3e}");
    for (i, (a, b)) in got.iter().zip(cpu.iter()).enumerate() {
        let tol = mul * b.abs().max(floor);
        assert!(
            (a - b).abs() <= tol,
            "{op} [{tag}] elem {i}: gpu {a} vs cpu {b} (tol {tol:.2e}, max_abs_err={max_err:.3e})"
        );
    }
}

/// Realistic GDN-shaped strictly-lower-triangular `[c,c]` attn matrix (mirrors poot-eval's private `realistic_gdn_attn`):
/// `attn[i,j] = beta_i * dot(k_i, k_j)` for `j < i`, `k` rows L2-normalized, `beta_i in (0.05, 0.95)`. Well-conditioned,
/// unlike raw uniform noise.
fn realistic_gdn_attn_c64(c: usize, dk: usize, seed: u64) -> Vec<f32> {
    let raw = |n: usize, s: u64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(2654435761).wrapping_add(s);
                (((x >> 8) % 2000) as f32 / 1000.0) - 1.0
            })
            .collect::<Vec<f32>>()
    };
    let raw_k = raw(c * dk, seed);
    let mut k = vec![0.0f32; c * dk];
    for i in 0..c {
        let row = &raw_k[i * dk..(i + 1) * dk];
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for a in 0..dk {
            k[i * dk + a] = row[a] / norm;
        }
    }
    let beta = raw(c, seed.wrapping_add(1));
    let mut attn = vec![0.0f32; c * c];
    for i in 0..c {
        let beta_i = 0.5 + 0.45 * beta[i]; // (0.05, 0.95)
        for j in 0..i {
            let dot: f32 = (0..dk).map(|a| k[i * dk + a] * k[j * dk + a]).sum();
            attn[i * c + j] = beta_i * dot;
        }
    }
    attn
}

/// GPU-vs-CPU check for `gdn_prefill_chunked` at the real chunk size C=64: builds a small synthetic graph (`H_k=2, H_v=4`,
/// tiled GQA, per-head dim `d`) for length `l`, evaluates it on CPU and the GPU executor, and checks both the output `o`
/// and the carried state `s_out` (a state-carry divergence would corrupt later decode steps). `gpu` is constructed by the
/// caller, which owns the skip-if-absent guard.
fn check_gdn_prefill_chunked_gpu(
    exec: &mut dyn Executor,
    target: Target,
    l: usize,
    chunk: usize,
    d: usize,
) {
    use poot_graph_ir::ops::gdn_prefill_chunked;
    let (h_k, h_v) = (2usize, 4usize);
    let c = chunk;

    let l2norm = |raw: &mut [f32], rows: usize, dim: usize| {
        for r in 0..rows {
            let row = &mut raw[r * dim..(r + 1) * dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    };
    let fillv = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
                ((x >> 8) % 1000) as f32 / 1000.0 - 0.5
            })
            .collect()
    };
    let mut qd = fillv(h_k * l * d, 101);
    l2norm(&mut qd, h_k * l, d);
    let mut kd = fillv(h_k * l * d, 202);
    l2norm(&mut kd, h_k * l, d);
    let vd = fillv(h_v * l * d, 303);
    // g: small negative log-decay (already log-domain; no Log call inside gdn_prefill_chunked).
    let gd: Vec<f32> = (0..h_v * l)
        .map(|i| -0.05 - 0.03 * ((i * 5 + i / l) % 4) as f32)
        .collect();
    // beta in a realistic (0,1) delta-rule-weight range.
    let betad: Vec<f32> = (0..h_v * l)
        .map(|i| 0.3 + 0.5 * (((i * 3 + i / l) % 4) as f32 / 4.0))
        .collect();
    let s0 = fillv(h_v * d * d, 404); // nonzero initial state, exercises both state cross-terms

    let mut tril_incl = vec![0.0f32; c * c]; // lower-triangular INCLUDING the diagonal
    let mut tril_strict = vec![0.0f32; c * c]; // STRICTLY lower-triangular, zero diagonal
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    for (pick, label) in [(0u8, "o"), (1u8, "s_out")] {
        // gdn_prefill_chunked returns two outputs but `GpuExecutor::run` returns only the graph's single `output`, so (as in
        // `causal_conv1d_decode_gpu_matches_cpu`'s pick_state loop) build a fresh Builder per picked output rather than using
        // state_input/finish_with_state (CPU-only `eval_with_state`). `s_in` is bound as a plain constant.
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, h_k, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, h_k, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, h_v, l, d]));
        let gate = b.constant("g", TensorType::f32(vec![1, h_v, l, 1]));
        let beta = b.constant("beta", TensorType::f32(vec![1, h_v, l, 1]));
        let s_in = b.constant("s_in", TensorType::f32(vec![1, h_v, d, d]));
        let ti = b.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
        let ts = b.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
        let ids = [q.id, k.id, v.id, gate.id, beta.id, s_in.id, ti.id, ts.id];
        let (o, s_out) = gdn_prefill_chunked(&b, q, k, v, gate, beta, s_in, ti, ts, c);
        let graph = b.finish(if pick == 0 { o } else { s_out });

        let mut inputs = HashMap::new();
        inputs.insert(ids[0], HostTensor::f32(vec![1, h_k, l, d], qd.clone()));
        inputs.insert(ids[1], HostTensor::f32(vec![1, h_k, l, d], kd.clone()));
        inputs.insert(ids[2], HostTensor::f32(vec![1, h_v, l, d], vd.clone()));
        inputs.insert(ids[3], HostTensor::f32(vec![1, h_v, l, 1], gd.clone()));
        inputs.insert(ids[4], HostTensor::f32(vec![1, h_v, l, 1], betad.clone()));
        inputs.insert(ids[5], HostTensor::f32(vec![1, h_v, d, d], s0.clone()));
        inputs.insert(ids[6], HostTensor::f32(vec![1, 1, c, c], tril_incl.clone()));
        inputs.insert(
            ids[7],
            HostTensor::f32(vec![1, 1, c, c], tril_strict.clone()),
        );

        let eval_inputs: HashMap<poot_graph_ir::ValueId, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        let cpu = eval(
            &graph,
            &eval_inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();
        let got = run_via_contract(exec, target, &graph, &inputs);
        assert_eq!(
            got.shape(),
            cpu.shape(),
            "gdn_prefill_chunked l={l} c={c} [{label}] shape"
        );
        assert_prefill_op_gpu_matches_cpu(
            "gdn_prefill_chunked",
            &format!("l={l} c={c} {label}"),
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            3e-3,
            1e-3,
        );
    }
}

/// Card 557: the kernel the planner chooses for `g`'s first compiled equation whose op matches `op`,
/// after the production `compile` for `target` (`FusionPolicy::Full`). Kernel choice is a plan property,
/// so a test asserts on it here, on the graph `compile` actually plans, never on a retagged op.
fn compiled_choice(
    g: &poot_graph_ir::Graph,
    target: &Target,
    op: impl Fn(&OpKind) -> bool,
) -> poot_graph_plan::KernelChoice {
    let program = compile_full(g, target);
    let eqn = program
        .graph()
        .eqns
        .iter()
        .find(|e| op(&e.op))
        .expect("the compiled graph holds the equation");
    program.kernel_choice(eqn).clone()
}

/// The production `compile` of `g` for `target` under `FusionPolicy::Full`, for a test that reads a compiled
/// equation's plan and choice together (`Program::planned`, `Program::kernel_choice`).
fn compile_full(g: &poot_graph_ir::Graph, target: &Target) -> poot_graph_plan::Program {
    poot_graph_plan::compile(
        g,
        target,
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("the graph compiles")
}

/// A SPIR-V/Vulkan target with the documented fixture caps, for plan assertions that must not depend on
/// the adapter the test happens to open.
fn spirv_fixture_target() -> Target {
    let backend = poot_target::Backend::SpirvVulkan;
    Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

/// One equation's plan and the kernel choice behind it, planned for a single device with no strided views
/// (the planner entry point `compile` itself drives per equation). Planner-selection tests assert on the
/// choice, never on the plan's opaque key.
fn plan_eqn_with_choice(
    g: &poot_graph_ir::Graph,
    eqn: &poot_graph_ir::Eqn,
    backend: poot_target::Backend,
    caps: &poot_target::DeviceCaps,
) -> Result<(poot_graph_plan::Plan, poot_graph_plan::KernelChoice), poot_graph_plan::PlanError> {
    poot_graph_plan::plan_eqn_choice_analyzed(
        &poot_graph_plan::ExactI32StorageAnalysis::new(g),
        g,
        eqn,
        backend,
        1,
        &std::collections::HashMap::new(),
        caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .map(|planned| (planned.plan, planned.choice))
}
