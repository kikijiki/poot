//! Cross-backend correctness: the wgpu backend against the CPU oracle.
//!
//! Each test runs the same graph on both and compares element-wise within f32 tolerance (max_abs < 1e-5 for
//! exact ops, looser for transcendentals and reductions). Skips when no Vulkan adapter is available.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Executor, HostView, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::{attention_masked, rmsnorm, silu};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, SlotKey, Storage, ValueId};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_tensor::HostTensor;

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn try_gpu() -> Option<(Box<dyn Executor>, Target)> {
    let device = poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())?;
    let target = device.target();
    Some((Box::new(poot_executor::Engine::new(device)), target))
}

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
    .expect("compile the graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32_bytes(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

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

/// Every input here is a plain F32 `Storage::Const` (no slots, no state, no packed weights), so this
/// splits a `HashMap<ValueId, Tensor>` bind straight into a one-shot [`WeightStore`] (the `Vec<SlotBind>`
/// is always empty).
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
                let name = m.name.clone().expect("named const");
                let t = inputs
                    .get(&id)
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
                    .unwrap_or_else(|| panic!("missing slot input {id:?}"));
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: t.shape().to_vec(),
                    dtype: DType::F32,
                    bytes: t
                        .as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                });
            }
            Storage::State | Storage::Computed(_) => {}
            Storage::Device => unreachable!(),
        }
    }
    (builder.build(), slots)
}

/// The executor-contract replacement for `GpuExecutor::run`/`.run(&g, &inputs)`: a one-shot,
/// stateless step through `Engine<WgpuDevice>::add_entry`/`step`, torn back down afterward - every
/// test in this file drives exactly one graph once, so there is no entry worth keeping alive across
/// calls.
fn run_once_contract(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
) -> Vec<f32> {
    let program = staged(g, target);
    let (store, slot_binds) = split_consts_and_slots(g, inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    f32_bytes(&bytes)
}

fn run_once_contract_i32(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
) -> Vec<i32> {
    let program = staged(g, target);
    let (store, slot_binds) = split_consts_and_slots(g, inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    i32_bytes(&bytes)
}

/// Deterministic xorshift64 fill.
fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], cpu: &[f32], label: &str, tol: f32) {
    assert_eq!(got.len(), cpu.len(), "{label}: length mismatch");
    for (i, (a, b)) in got.iter().zip(cpu.iter()).enumerate() {
        let t = tol * b.abs().max(1e-3);
        assert!(
            (a - b).abs() <= t,
            "{label} elem {i}: gpu {a} vs cpu {b} (tol={t})"
        );
    }
}

// ── Elementwise ops ──────────────────────────────────────────────────────

#[test]
fn elementwise_add_sub_mul_div_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 64usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let add = b.binary(BinOp::Add, a, c);
    let sub = b.binary(BinOp::Sub, add, c);
    let mul = b.binary(BinOp::Mul, sub, a);
    let out = b.binary(BinOp::Div, mul, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![n], fill(42, n)));
    inputs.insert(
        ci,
        HostTensor::f32(vec![n], fill(99, n).iter().map(|x| x + 2.0).collect()),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "eltwise chain", 1e-5);
}

/// Card 630 SC-005 value table: zero, tiny, unit, mid, saturating (`+-100` overflows the textbook
/// `(e^x - e^-x) / (e^x + e^-x)`) and the half points.
const TRANSCENDENTAL_TABLE: [f32; 21] = [
    0.0, 1e-6, -1e-6, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 5.0, -5.0, 7.0, -7.0, 10.0, -10.0, 88.0,
    -88.0, 100.0, -100.0, 0.1, -0.1,
];

/// ADR-0101 tier 2 for the device `Tanh`/`Erf` expansions against the oracle's libm: `abs + rel * |cpu|`,
/// and the saturated rows (`|x| >= 10`, where the true value rounds to `+-1`) must be exactly `+-1`.
fn assert_transcendental_tier2(got: &[f32], cpu: &[f32], label: &str, saturates_at: f32) {
    assert_eq!(got.len(), cpu.len(), "{label}: length mismatch");
    for (i, (g, c)) in got.iter().zip(cpu).enumerate() {
        let x = TRANSCENDENTAL_TABLE[i];
        assert!(
            !g.is_nan(),
            "{label}(x={x}): device returned NaN, oracle {c}"
        );
        // The absolute slack covers the device `tanh`'s cancellation near zero (`Tanh` doc comment); the
        // 1e-4 relative part covers `exp` differing across the three backends' hardware.
        let tol = 2e-6 + 1e-4 * c.abs();
        assert!(
            (g - c).abs() <= tol,
            "{label}(x={x}): device {g} vs oracle {c} (tol {tol})"
        );
        if x.abs() >= saturates_at {
            assert_eq!(
                *g,
                x.signum(),
                "{label}({x}) must be exactly {}",
                x.signum()
            );
        }
    }
}

/// Card 630 SC-005 (wgpu): device `Tanh` and `Erf` over the first-row table within tier 2 of the oracle,
/// and `tanh(+-100)` exactly `+-1`. Mutation: emit tanh as `(e^x - e^-x) / (e^x + e^-x)` without the `|x|`
/// shift in `poot-kernelgen`'s `emit_scalar_op`; `+-100` returns NaN and this row goes red.
#[test]
fn unary_tanh_erf_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = TRANSCENDENTAL_TABLE.len();
    for (op, label) in [(UnOp::Tanh, "tanh"), (UnOp::Erf, "erf")] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![n]));
        let out = b.unary(op, x);
        let xi = x.id;
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![n], TRANSCENDENTAL_TABLE.to_vec()));
        let cpu = eval(
            &g,
            &inputs
                .iter()
                .map(|(&id, t)| (id, Value::from(t.clone())))
                .collect::<HashMap<_, _>>(),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();
        let got = run_once_contract(&mut *exec, target, &g, &inputs);
        assert_transcendental_tier2(&got, cpu.as_f32().unwrap(), label, 10.0);
    }
}

#[test]
fn unary_neg_exp_log_sqrt_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 32usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let neg = b.unary(UnOp::Neg, x);
    let xi = x.id;
    let g = b.finish(neg);

    let xd: Vec<f32> = (1..=n).map(|i| i as f32 * 0.1).collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![n], xd));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "neg", 1e-6);

    // exp
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::f32(vec![n]));
    let e = b2.unary(UnOp::Exp, x2);
    let g2 = b2.finish(e);
    let mut in2 = HashMap::new();
    in2.insert(x2.id, HostTensor::f32(vec![n], fill(7, n)));
    let cpu2 = eval(
        &g2,
        &in2.iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got2 = run_once_contract(&mut *exec, target, &g2, &in2);
    assert_close(&got2, cpu2.as_f32().unwrap(), "exp", 1e-5);

    // sqrt (positive inputs)
    let b3 = Builder::new();
    let x3 = b3.constant("x", TensorType::f32(vec![n]));
    let s = b3.unary(UnOp::Sqrt, x3);
    let g3 = b3.finish(s);
    let xd3: Vec<f32> = (1..=n).map(|i| i as f32 * 0.5 + 0.1).collect();
    let mut in3 = HashMap::new();
    in3.insert(x3.id, HostTensor::f32(vec![n], xd3));
    let cpu3 = eval(
        &g3,
        &in3.iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got3 = run_once_contract(&mut *exec, target, &g3, &in3);
    assert_close(&got3, cpu3.as_f32().unwrap(), "sqrt", 1e-5);
}

// ── Matmul ───────────────────────────────────────────────────────────────

#[test]
fn matmul_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (m, k, n) = (8usize, 16usize, 12usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![m, k], fill(11, m * k)));
    inputs.insert(wi, HostTensor::f32(vec![k, n], fill(22, k * n)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu.shape());
    assert_eq!(got.len(), cpu.as_f32().unwrap().len());
    // Matmul accumulates K terms; allow reduction-order tolerance.
    assert_close(&got, cpu.as_f32().unwrap(), "matmul", 1e-4);
}

#[test]
fn matmul_bias_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (m, k, n) = (4usize, 8usize, 6usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let mm = b.matmul(a, w);
    let out = b.binary(BinOp::Add, mm, bias);
    let (ai, wi, bi) = (a.id, w.id, bias.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![m, k], fill(33, m * k)));
    inputs.insert(wi, HostTensor::f32(vec![k, n], fill(44, k * n)));
    inputs.insert(bi, HostTensor::f32(vec![n], fill(55, n)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "matmul_bias", 1e-4);
}

// ── Reduce ───────────────────────────────────────────────────────────────

#[test]
fn reduce_sum_last_axis_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols) = (4usize, 16usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Sum, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        HostTensor::f32(vec![rows, cols], fill(101, rows * cols)),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    // Reduction sum order differs; tolerance scales with column count.
    let tol = 1e-4 * (cols as f32).sqrt();
    assert_close(&got, cpu.as_f32().unwrap(), "reduce_sum", tol);
}

#[test]
fn reduce_max_last_axis_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols) = (4usize, 16usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Max, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        HostTensor::f32(vec![rows, cols], fill(202, rows * cols)),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    // Max is exact (order-independent).
    assert_close(&got, cpu.as_f32().unwrap(), "reduce_max", 1e-6);
}

// ── RMSNorm ──────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 32usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = rmsnorm(&b, x, w, 1e-6);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, 1, n], fill(303, n)));
    inputs.insert(
        wi,
        HostTensor::f32(
            vec![n],
            fill(404, n).iter().map(|v| 1.0 + v.abs()).collect(),
        ),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "rmsnorm", 1e-4);
}

// ── Attention ────────────────────────────────────────────────────────────

#[test]
fn masked_attention_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hq, hkv, cap, d) = (2usize, 1usize, 4usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let out = attention_masked(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], fill(501, hq * d)));
    inputs.insert(
        ki,
        HostTensor::f32(vec![1, hkv, cap, d], fill(502, hkv * cap * d)),
    );
    inputs.insert(
        vi,
        HostTensor::f32(vec![1, hkv, cap, d], fill(503, hkv * cap * d)),
    );
    inputs.insert(
        mi,
        HostTensor::f32(vec![1, 1, 1, cap], vec![0.0, 0.0, -1.0e9, -1.0e9]),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    // Softmax + matmul accumulate; tolerance accounts for exp/sum order.
    assert_close(&got, cpu.as_f32().unwrap(), "attention", 5e-3);
}

// ── FFN block ────────────────────────────────────────────────────────────

#[test]
fn ffn_block_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (d, hidden) = (16usize, 32usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, d]));
    let w1 = b.constant("w1", TensorType::f32(vec![d, hidden]));
    let w2 = b.constant("w2", TensorType::f32(vec![hidden, d]));
    let h = b.matmul(x, w1);
    let act = silu(&b, h);
    let out = b.matmul(act, w2);
    let (xi, w1i, w2i) = (x.id, w1.id, w2.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, d], fill(601, d)));
    inputs.insert(w1i, HostTensor::f32(vec![d, hidden], fill(602, d * hidden)));
    inputs.insert(w2i, HostTensor::f32(vec![hidden, d], fill(603, hidden * d)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "ffn", 1e-3);
}

// ── Broadcast binary ─────────────────────────────────────────────────────

#[test]
fn broadcast_binary_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (batch, rows, cols) = (2usize, 4usize, 8usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![batch, rows, cols]));
    let bias = b.constant("bias", TensorType::f32(vec![cols]));
    let out = b.binary(BinOp::Add, a, bias);
    let (ai, bi) = (a.id, bias.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        ai,
        HostTensor::f32(vec![batch, rows, cols], fill(701, batch * rows * cols)),
    );
    inputs.insert(bi, HostTensor::f32(vec![cols], fill(702, cols)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "broadcast_add", 1e-6);
}

// ── Transpose + Slice ────────────────────────────────────────────────────

#[test]
fn transpose_slice_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (r, c) = (6usize, 8usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![r, c]));
    let t = b.transpose(x, vec![1, 0]);
    let s = b.slice(t, 1, 2, 6);
    let xi = x.id;
    let g = b.finish(s);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![r, c], fill(801, r * c)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu.shape());
    assert_eq!(got.len(), cpu.as_f32().unwrap().len());
    assert_close(&got, cpu.as_f32().unwrap(), "transpose_slice", 1e-6);
}

// ── Gather ───────────────────────────────────────────────────────────────

#[test]
fn gather_axis0_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (n, d) = (8usize, 4usize);
    let idx_count = 3usize;
    let b = Builder::new();
    let data = b.constant("data", TensorType::f32(vec![n, d]));
    let idx = b.constant("idx", TensorType::f32(vec![idx_count]));
    let out = b.gather(data, 0, idx);
    let (di, ii) = (data.id, idx.id);
    let g = b.finish(out);

    let dd: Vec<f32> = fill(901, n * d);
    let id: Vec<f32> = vec![3.0, 0.0, 6.0];
    let mut inputs = HashMap::new();
    inputs.insert(di, HostTensor::f32(vec![n, d], dd));
    inputs.insert(ii, HostTensor::f32(vec![idx_count], id));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu.shape());
    assert_eq!(got.len(), cpu.as_f32().unwrap().len());
    // Gather is exact copy.
    assert_close(&got, cpu.as_f32().unwrap(), "gather", 1e-6);
}

// ── Sampling primitives (card 551a) ──

/// Card 551a (SC-001, SC-002, SC-005, SC-007): `OpKind::SampleToken { rule: Greedy }` through the full
/// `compile`/planner/executor contract (not `pootc`'s raw-kernel `import_run.rs` harness, which bypasses
/// `planner::sampling` entirely) matches the CPU oracle bit for bit. Row 0 carries two ties, isolated
/// from any non-finite logit so each tie-break is actually observable in the final token rather than
/// masked by the non-finite override: a per-lane tie (indices 5 and 5+64, same lane 5 two groups over -
/// already resolved by the per-lane scan's own strict `>` before the cross-lane fold ever sees it) and a
/// cross-lane tie (indices 1 and 64, different lanes - the case the cross-lane fold's `ci < bi`
/// tie-break exists for; wants token 1, the lowest of the four tied indices). Row 1 carries the
/// non-finite logit (NaN at the last index) alone.
#[test]
fn sample_token_greedy_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, vocab) = (2usize, 130usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let out = b.sample_token(
        poot_graph_ir::op::SampleRule::Greedy,
        logits,
        None,
        None,
        None,
    );
    let li = logits.id;
    let g = b.finish(out);

    let mut data = vec![0f32; rows * vocab];
    for row in 0..rows {
        for i in 0..vocab {
            data[row * vocab + i] = fill(100 + row as u64, vocab)[i] + (i as f32) * 1e-4;
        }
    }
    let top = data[0..vocab]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        + 1.0;
    data[5] = top; // per-lane tie
    data[5 + 64] = top;
    data[1] = top; // cross-lane tie
    data[64] = top;
    data[vocab + vocab - 1] = f32::NAN; // row 1, isolated from row 0's ties
    let mut inputs = HashMap::new();
    inputs.insert(li, HostTensor::f32(vec![rows, vocab], data));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract_i32(&mut *exec, target, &g, &inputs);
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token Greedy through the full contract must match the CPU oracle, including the \
         per-lane tie, the cross-lane tie (row 0 wants token 1) and the non-finite row"
    );
}

/// Card 551a (SC-001, SC-010): `OpKind::RandomUniform` through the full compile/planner/executor
/// contract matches the CPU oracle bit for bit (tier 1, ADR-0101).
#[test]
fn random_uniform_cross_backend() {
    use poot_graph_ir::{Slot, SlotKey};

    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols) = (3usize, 37usize);
    let b = Builder::new();
    // `seed` is I32, so it rides as a `Slot` (card 551a's `Slot::Sampler`), not a `Storage::Const`:
    // `split_consts_and_slots`'s shared helper hardcodes F32 weight storage for consts (every other
    // cross_backend test is F32-only), so this test builds its own slot bind with the real I32 dtype
    // instead of widening the test helper for this one case.
    let seed = b.slot_named(
        Slot::Sampler,
        "seed",
        TensorType::new(vec![rows], DType::I32),
    );
    let out = b.random_uniform(seed, cols);
    let g = b.finish(out);

    let seed_data: Vec<i32> = vec![0, 1, 0xDEADBEEFu32 as i32];
    let inputs: HashMap<ValueId, HostTensor> =
        HashMap::from([(seed.id, HostTensor::i32(vec![rows], seed_data.clone()))]);
    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();

    let program = staged(&g, target);
    let store = WeightStore::builder().build();
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let seed_bytes: Vec<u8> = seed_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bind = SlotBind {
        key: SlotKey::new(Slot::Sampler, Some("seed")),
        shape: vec![rows],
        dtype: DType::I32,
        bytes: seed_bytes,
    };
    let binds = [bind];
    let step_in = step_inputs(&binds);
    let bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let got = f32_bytes(&bytes);

    assert_close(&got, cpu.as_f32().unwrap(), "random_uniform", 0.0);
}

/// Builds and runs a Gumbel-family `SampleToken` graph through the full contract: `logits`/`noise`/
/// `params` as F32 consts (`split_consts_and_slots` elsewhere in this file hardcodes F32 for consts,
/// which is right for these three), `top_k` (when present) as an I32 `Slot::Sampler` bind - the same
/// dtype problem `random_uniform_cross_backend` documents, handled the same way.
fn run_gumbel_family_contract(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    logits: (ValueId, Vec<usize>, Vec<f32>),
    noise: (ValueId, Vec<usize>, Vec<f32>),
    params: (ValueId, Vec<usize>, Vec<f32>),
    top_k: Option<(ValueId, Vec<usize>, Vec<i32>)>,
) -> Vec<i32> {
    use poot_graph_ir::Slot;

    let program = staged(g, target);
    let mut builder = WeightStore::builder();
    for (id, shape, data) in [&logits, &noise, &params] {
        let name = g.meta(*id).name.clone().unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape.clone(), Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let store = builder.build();
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let mut binds = Vec::new();
    if let Some((_, shape, data)) = &top_k {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        binds.push(SlotBind {
            key: SlotKey::new(Slot::Sampler, Some("top_k")),
            shape: shape.clone(),
            dtype: DType::I32,
            bytes,
        });
    }
    let step_in = step_inputs(&binds);
    let bytes = exec
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    i32_bytes(&bytes)
}

/// Card 551a (SC-001, SC-004, SC-007): `OpKind::SampleToken { rule: Gumbel }` through the full
/// compile/planner/executor contract matches the CPU oracle bit for bit. Row 0 isolates the
/// `inv_temp` multiply (R472-007: "multiplies, not divides"): logit 0 (10.0, noise 0.5) vs logit 1
/// (10.5, noise 0.0) at `inv_temp = 0.1` only out-ranks logit 1 when the temperature SCALES the logit
/// down before adding noise (`1.0+0.5=1.5` vs `1.05+0=1.05`, token 0); dividing instead inflates both
/// logits so noise can no longer compete (`100+0.5` vs `105`, token 1) - an observable, not masked,
/// mutation witness. Row 1 is an unrelated non-finite row (NaN), isolated from row 0 so the override
/// (SC-007: non-finite forces token 0) can't hide row 0's own result.
#[test]
fn sample_token_gumbel_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, vocab) = (2usize, 8usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let noise = b.constant("noise", TensorType::f32(vec![rows, vocab]));
    let params = b.constant("params", TensorType::f32(vec![rows, 3]));
    let out = b.sample_token(
        poot_graph_ir::op::SampleRule::Gumbel,
        logits,
        Some(noise),
        Some(params),
        None,
    );
    let (li, ni, pi) = (logits.id, noise.id, params.id);
    let g = b.finish(out);

    let mut logits_data = vec![-5.0f32; rows * vocab];
    logits_data[0] = 10.0;
    logits_data[1] = 10.5;
    logits_data[vocab + 6] = f32::NAN; // row 1, index 6
    let mut noise_data = vec![0.0f32; rows * vocab];
    noise_data[0] = 0.5;
    let params_data = vec![
        0.1,
        -3.4028235e38,
        1.0, // row 0: inv_temp=0.1, min_p disabled, noise_scale=1
        1.0,
        -3.4028235e38,
        1.0, // row 1: doesn't matter, non-finite overrides
    ];
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            li,
            HostTensor::f32(vec![rows, vocab], logits_data.clone()).into(),
        ),
        (
            ni,
            HostTensor::f32(vec![rows, vocab], noise_data.clone()).into(),
        ),
        (
            pi,
            HostTensor::f32(vec![rows, 3], params_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let got = run_gumbel_family_contract(
        &mut *exec,
        target,
        &g,
        (li, vec![rows, vocab], logits_data),
        (ni, vec![rows, vocab], noise_data),
        (pi, vec![rows, 3], params_data),
        None,
    );
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token Gumbel through the full contract must match the CPU oracle: row 0's inv_temp \
         multiply, row 1's non-finite override"
    );
}

/// Card 551a (SC-001): `OpKind::SampleToken { rule: GumbelTopK }` through the full compile/planner/
/// executor contract matches the CPU oracle bit for bit. `top_k = 1` with `min_p` disabled folds the
/// floor to the row max; two non-zero indices (3, 6) tied exactly at that max, the rest well below -
/// the inclusive filter (`v >= floor`) must keep exactly those two survivors for the tie-break to pick
/// the lower (3); an exclusive filter keeps none and falls back to 0, a different, observable token
/// (same technique as the SC-006 fixture, `imported_sample_topp_gumbel_argmax_batched_matches_cpu_oracle`
/// in `pootc`'s `import_run.rs`, applied here at the contract level instead of the raw-kernel level).
#[test]
fn sample_token_gumbel_topk_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, vocab) = (1usize, 8usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let noise = b.constant("noise", TensorType::f32(vec![rows, vocab]));
    let params = b.constant("params", TensorType::f32(vec![rows, 3]));
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
    };
    let out = b.sample_token(
        poot_graph_ir::op::SampleRule::GumbelTopK,
        logits,
        Some(noise),
        Some(params),
        Some(top_k),
    );
    let (li, ni, pi) = (logits.id, noise.id, params.id);
    let g = b.finish(out);

    let mut logits_data = vec![0.0f32; rows * vocab];
    logits_data[3] = 10.0;
    logits_data[6] = 10.0;
    let noise_data = vec![0.0f32; rows * vocab];
    let params_data = vec![1.0, -3.4028235e38, 0.0]; // inv_temp=1, min_p disabled, noise_scale=0
    let top_k_data = vec![1i32];
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            li,
            HostTensor::f32(vec![rows, vocab], logits_data.clone()).into(),
        ),
        (
            ni,
            HostTensor::f32(vec![rows, vocab], noise_data.clone()).into(),
        ),
        (
            pi,
            HostTensor::f32(vec![rows, 3], params_data.clone()).into(),
        ),
        (
            top_k.id,
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let got = run_gumbel_family_contract(
        &mut *exec,
        target,
        &g,
        (li, vec![rows, vocab], logits_data),
        (ni, vec![rows, vocab], noise_data),
        (pi, vec![rows, 3], params_data),
        Some((top_k.id, vec![rows], top_k_data)),
    );
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token GumbelTopK through the full contract must match the CPU oracle: top_k=1 keeps \
         both tied survivors (3, 6), tie-break picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

/// Card 551a (SC-001): `OpKind::SampleToken { rule: GumbelTopKTopP }` through the full compile/
/// planner/executor contract matches the CPU oracle bit for bit. `top_p = 0` must collapse to top-1
/// (`floor = max_logit`) rather than being treated as disabled (the kernel doc's own warning: "do not
/// take the disabled shortcut: it would leave floor at floor1"); `top_k` is disabled too (0), so
/// nothing else would impose a floor if top_p's own collapse didn't run. Logits 3 and 6 are tied at
/// the row max (10.0, the correct floor); logit 2 sits just below (9.0) but carries a large noise
/// boost (noise_scale=1, noise=5.0) that only wins if the (buggy, undisabled) filter lets it compete:
/// correctly collapsed, index 2 is excluded by `v >= 10.0` and the tie-break over {3, 6} picks 3;
/// left disabled, index 2's perturbed value (9+5=14) beats the tied pair's (10+0=10 each) and wins -
/// an observable, not masked, mutation witness.
#[test]
fn sample_token_gumbel_topk_topp_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, vocab) = (1usize, 8usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let noise = b.constant("noise", TensorType::f32(vec![rows, vocab]));
    let params = b.constant("params", TensorType::f32(vec![rows, 4]));
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
    };
    let out = b.sample_token(
        poot_graph_ir::op::SampleRule::GumbelTopKTopP,
        logits,
        Some(noise),
        Some(params),
        Some(top_k),
    );
    let (li, ni, pi) = (logits.id, noise.id, params.id);
    let g = b.finish(out);

    let mut logits_data = vec![-100.0f32; rows * vocab];
    logits_data[2] = 9.0;
    logits_data[3] = 10.0;
    logits_data[6] = 10.0;
    let mut noise_data = vec![0.0f32; rows * vocab];
    noise_data[2] = 5.0;
    // inv_temp=1, min_p disabled, noise_scale=1, top_p=0 (must collapse to top-1, not disable).
    let params_data = vec![1.0, -3.4028235e38, 1.0, 0.0];
    let top_k_data = vec![0i32]; // top_k disabled; top_p alone must do the filtering
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            li,
            HostTensor::f32(vec![rows, vocab], logits_data.clone()).into(),
        ),
        (
            ni,
            HostTensor::f32(vec![rows, vocab], noise_data.clone()).into(),
        ),
        (
            pi,
            HostTensor::f32(vec![rows, 4], params_data.clone()).into(),
        ),
        (
            top_k.id,
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let got = run_gumbel_family_contract(
        &mut *exec,
        target,
        &g,
        (li, vec![rows, vocab], logits_data),
        (ni, vec![rows, vocab], noise_data),
        (pi, vec![rows, 4], params_data),
        Some((top_k.id, vec![rows], top_k_data)),
    );
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token GumbelTopKTopP through the full contract must match the CPU oracle: top_p=0 \
         collapses to top-1 (excluding index 2's noise-boosted decoy), tie-break over {{3, 6}} picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

/// Card 677 SC-001: `OpKind::SampleToken { rule: GumbelTopKTopP }` with top-p the ONLY active
/// truncation (`top_k = 0`, `floor_offset = -inf` for a disabled min-p) still truncates on both the
/// device and the CPU oracle. Found by card 551b's review: `poot-eval`'s `sample_one_row` used to
/// start its pre-top-p bisection floor at `-inf` whenever neither min-p nor top-k had already raised
/// it off that start, and `(-inf + finite) * 0.5 == -inf` in IEEE float, so the 30-iteration
/// bisection never moved and top-p silently became a no-op.
///
/// Fixture: logits 3 and 6 tied at the row max (10.0); logit 2 sits just below (9.0) but carries a
/// huge noise boost (noise_scale=1, noise=20.0) that makes it the Gumbel-max winner of the
/// UNTRUNCATED row (perturbed 9+20=29 beats the tied pair's 10+0=10). `top_p = 0.7` with `top_k = 0`
/// and min-p disabled gives a quantized nucleus mass of 16 (idx 3) + 16 (idx 6) + 6 (idx 2) = 38 and
/// a threshold of `round(0.7 * 38) = 27`; the tied pair alone already clears it (32 >= 27), so the
/// correct nucleus is exactly {3, 6}, excluding the decoy. A correct bisection converges its floor to
/// `max_logit` (10.0), which still keeps the tied pair since `v >= floor` is `>=`, not `>`; tie-break
/// picks index 3. Left broken (floor pinned at `-inf`), every index competes and the decoy (29) wins
/// - an observable, not masked, mutation witness (reverting `poot-eval`'s fix reproduces this on the
///   CPU oracle; see the card's done-report for the recorded red/green run).
#[test]
fn sample_token_gumbel_topp_only_excludes_untruncated_winner_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, vocab) = (1usize, 8usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let noise = b.constant("noise", TensorType::f32(vec![rows, vocab]));
    let params = b.constant("params", TensorType::f32(vec![rows, 4]));
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
    };
    let out = b.sample_token(
        poot_graph_ir::op::SampleRule::GumbelTopKTopP,
        logits,
        Some(noise),
        Some(params),
        Some(top_k),
    );
    let (li, ni, pi) = (logits.id, noise.id, params.id);
    let g = b.finish(out);

    let mut logits_data = vec![-100.0f32; rows * vocab];
    logits_data[2] = 9.0;
    logits_data[3] = 10.0;
    logits_data[6] = 10.0;
    let mut noise_data = vec![0.0f32; rows * vocab];
    noise_data[2] = 20.0;
    // inv_temp=1, min_p disabled (floor_offset = -inf, matching driver::suffix::row_params), noise_scale=1, top_p=0.7.
    let params_data = vec![1.0, f32::NEG_INFINITY, 1.0, 0.7];
    let top_k_data = vec![0i32]; // top_k disabled; top_p alone must do the filtering
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            li,
            HostTensor::f32(vec![rows, vocab], logits_data.clone()).into(),
        ),
        (
            ni,
            HostTensor::f32(vec![rows, vocab], noise_data.clone()).into(),
        ),
        (
            pi,
            HostTensor::f32(vec![rows, 4], params_data.clone()).into(),
        ),
        (
            top_k.id,
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let got = run_gumbel_family_contract(
        &mut *exec,
        target,
        &g,
        (li, vec![rows, vocab], logits_data),
        (ni, vec![rows, vocab], noise_data),
        (pi, vec![rows, 4], params_data),
        Some((top_k.id, vec![rows], top_k_data)),
    );
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token GumbelTopKTopP through the full contract must match the CPU oracle: top_p=0.7 \
         alone (no min-p/top-k) must exclude the noise-boosted decoy (index 2), tie-break over {{3, 6}} \
         picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

// ── LayerNorm and real-magnitude wide reduces ──

/// Real-magnitude deterministic fill: values in ~[-30, 30], every 7th scaled by 1e-3, to stress float
/// cancellation in a reduce (unlike the tiny [-1,1] `fill`). Same as the ROCm cross_backend helper.
fn fill_real_magnitude(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..count)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            let base = (u * 2.0 - 1.0) * 30.0;
            if i % 7 == 0 { base * 1e-3 } else { base }
        })
        .collect()
}

/// Independent f64 host fold (sum) over the last axis of a row-major `[rows, cols]`, a second reference
/// besides the eval oracle (real-magnitude data against f64, never poot-vs-poot at one precision).
fn f64_row_sum(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for c in 0..cols {
                acc += data[r * cols + c] as f64;
            }
            acc as f32
        })
        .collect()
}

/// Independent f64 host fold (max) over the last axis of a row-major `[rows, cols]`.
fn f64_row_max(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let mut acc = f64::NEG_INFINITY;
            for c in 0..cols {
                let v = data[r * cols + c] as f64;
                if v > acc {
                    acc = v;
                }
            }
            acc as f32
        })
        .collect()
}

fn assert_close_abs_rel(got: &[f32], reference: &[f32], label: &str, abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
    for (i, (a, r)) in got.iter().zip(reference.iter()).enumerate() {
        let t = abs_tol + rel_tol * r.abs();
        assert!(
            (a - r).abs() <= t,
            "{label} elem {i}: gpu {a} vs ref {r} (tol={t})"
        );
    }
}

/// LayerNorm (mean-subtract, variance-normalize, affine) on wgpu vs the CPU oracle at hidden width 896,
/// with real-magnitude input.
#[test]
fn layernorm_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 896usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let out = poot_graph_ir::ops::layernorm(&b, x, w, bias, 1e-6);
    let (xi, wi, bi) = (x.id, w.id, bias.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        HostTensor::f32(vec![1, 1, n], fill_real_magnitude(941, n)),
    );
    inputs.insert(
        wi,
        HostTensor::f32(
            vec![n],
            fill(942, n).iter().map(|v| 1.0 + v.abs()).collect(),
        ),
    );
    inputs.insert(
        bi,
        HostTensor::f32(vec![n], fill(943, n).iter().map(|v| v * 0.1).collect()),
    );

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "layernorm_nonpow2", 1e-4);
}

/// Last-axis SUM reduce over non-power-of-2, real-hidden-dim widths with real-magnitude data. Matches the CPU
/// oracle and an independent f64 host fold at width 896.
#[test]
fn reduce_sum_nonpow2_wide_real_magnitude_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let rows = 3usize;
    for (i, &cols) in [33usize, 896, 1536].iter().enumerate() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Sum, x, 1, false);
        let xi = x.id;
        let g = b.finish(out);

        let data = fill_real_magnitude(820 + i as u64, rows * cols);
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![rows, cols], data.clone()));

        let cpu = eval(
            &g,
            &inputs
                .iter()
                .map(|(&id, t)| (id, Value::from(t.clone())))
                .collect::<HashMap<_, _>>(),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();
        let got = run_once_contract(&mut *exec, target, &g, &inputs);
        let label = format!("wgpu_reduce_sum_nonpow2_wide_cols{cols}");
        let abs_tol = 0.02 * (cols as f32).sqrt();
        assert_close_abs_rel(&got, cpu.as_f32().unwrap(), &label, abs_tol, 1e-4);

        if cols == 896 {
            let want_f64 = f64_row_sum(&data, rows, cols);
            assert_close_abs_rel(
                &got,
                &want_f64,
                "wgpu_reduce_sum_nonpow2_wide_cols896_vs_f64",
                0.05,
                1e-4,
            );
        }
    }
}

/// Last-axis MAX reduce over non-power-of-2, real-hidden-dim widths with real-magnitude data. Max is exact, so
/// it must match the CPU oracle and the f64 fold tightly.
#[test]
fn reduce_max_nonpow2_wide_real_magnitude_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let rows = 3usize;
    for (i, &cols) in [33usize, 896, 1536].iter().enumerate() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Max, x, 1, false);
        let xi = x.id;
        let g = b.finish(out);

        let data = fill_real_magnitude(660 + i as u64, rows * cols);
        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![rows, cols], data.clone()));

        let cpu = eval(
            &g,
            &inputs
                .iter()
                .map(|(&id, t)| (id, Value::from(t.clone())))
                .collect::<HashMap<_, _>>(),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();
        let got = run_once_contract(&mut *exec, target, &g, &inputs);
        let label = format!("wgpu_reduce_max_nonpow2_wide_cols{cols}");
        assert_close_abs_rel(&got, cpu.as_f32().unwrap(), &label, 1e-5, 1e-5);
        let want_f64 = f64_row_max(&data, rows, cols);
        assert_close_abs_rel(&got, &want_f64, &format!("{label}_vs_f64"), 1e-5, 1e-5);
    }
}

// ── DynamicUpdateSlice and edge shapes ──

/// DynamicUpdateSlice (the KV-cache slot write): write one row of `update` into `operand` at a static row
/// index along axis 0, bit-exact against the CPU oracle (pure copy). Guards against an unreachable-DUS-arm bug.
#[test]
fn dynamic_update_slice_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols, index) = (4usize, 4usize, 2usize);
    let b = Builder::new();
    let operand = b.constant("operand", TensorType::f32(vec![rows, cols]));
    let update = b.constant("update", TensorType::f32(vec![1, cols]));
    let out = b.dynamic_update_slice(operand, update, index, 0);
    let (oi, ui) = (operand.id, update.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        oi,
        HostTensor::f32(vec![rows, cols], fill(701, rows * cols)),
    );
    inputs.insert(ui, HostTensor::f32(vec![1, cols], fill(702, cols)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "dus_axis0", 0.0); // pure copy -> bit-exact
}

/// Non-power-of-2 length (33) elementwise add: the tail lane past the last full 32-wide group must be
/// handled.
#[test]
fn non_power_of_2_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 33usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![n], fill(601, n)));
    inputs.insert(ci, HostTensor::f32(vec![n], fill(602, n)));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "npo2_33", 1e-5);
}

/// A single-element tensor: the 1-lane dispatch must produce the right result, not skip or read out of bounds.
#[test]
fn single_element_cross_backend() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1]));
    let c = b.constant("c", TensorType::f32(vec![1]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1], vec![3.5]));
    inputs.insert(ci, HostTensor::f32(vec![1], vec![2.5]));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_close(&got, cpu.as_f32().unwrap(), "single_elem", 1e-6);
}

/// DIAGNOSTIC (card 255 item 6): a large `transpose` + `matmul` of a BLOOM-scale `lm_head` weight
/// ([vocab=250880, hidden=1024]) alone, with no ALiBi mask or BLOOM architecture. `bloom_560m_gpu_reprefill_
/// matches_cpu` hung on this shape (kernel log: `ring gfx_0.0.0 timeout`); if this reproduces it, the bug is
/// in `GpuExecutor::run`'s per-op large-transpose dispatch, not ALiBi/BLOOM.
#[test]
#[ignore = "KNOWN FAILING (docs/updates/0615-alibi-real-gpu-parity-wgpu.md, card 258): DOES hang real GPU \
            hardware (auto-recovers via a ring reset, no reboot needed, but disruptive) - allocates a \
            ~1GB weight; run manually only, with --ignored, once you understand the expected outcome"]
fn large_vocab_lm_head_transpose_matmul_does_not_hang() {
    let Some(_allow_hang) = poot_test_util::require_gpu_hang_opt_in(
        "large_vocab_lm_head_transpose_matmul_does_not_hang",
    ) else {
        return;
    };
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hidden, vocab) = (1024usize, 250880usize);
    let b = Builder::new();
    let hstate = b.constant("hidden", TensorType::f32(vec![1, 1, hidden]));
    let w = b.constant("lm_head.weight", TensorType::f32(vec![vocab, hidden]));
    let wt = b.transpose(w, vec![1, 0]); // [hidden, vocab]
    let out = b.matmul(hstate, wt); // [1,1,vocab]
    let (hi, wi) = (hstate.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(hi, HostTensor::f32(vec![1, 1, hidden], fill(1, hidden)));
    // Weight-sized fill (~1GB); xorshift is cheap on the host.
    inputs.insert(
        wi,
        HostTensor::f32(vec![vocab, hidden], fill(2, vocab * hidden)),
    );

    let t0 = std::time::Instant::now();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    eprintln!("large lm_head transpose+matmul: {:?}", t0.elapsed());
    assert_eq!(got.len(), vocab);
    assert!(got.iter().all(|v| v.is_finite()));
}

/// The bare `transpose` alone (no matmul), same BLOOM-scale shape: separates a hang in the transpose's
/// materialize-copy kernel from one that needs the following matmul.
#[test]
#[ignore = "diagnostic: allocates a ~1GB weight (safe - passes cleanly in ~3.7s, see docs/updates/0615-\
            alibi-real-gpu-parity-wgpu.md); run manually with --ignored"]
fn large_vocab_transpose_alone_does_not_hang() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hidden, vocab) = (1024usize, 250880usize);
    let b = Builder::new();
    let w = b.constant("lm_head.weight", TensorType::f32(vec![vocab, hidden]));
    let wt = b.transpose(w, vec![1, 0]); // [hidden, vocab]
    let wi = w.id;
    let g = b.finish(wt);

    let mut inputs = HashMap::new();
    inputs.insert(
        wi,
        HostTensor::f32(vec![vocab, hidden], fill(2, vocab * hidden)),
    );

    let t0 = std::time::Instant::now();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    eprintln!("large lm_head transpose alone: {:?}", t0.elapsed());
    assert_eq!(got.len(), vocab * hidden);
}

/// The bare `matmul` alone at BLOOM lm_head scale, weight fed already in `[hidden, vocab]` layout (no
/// transpose op).
#[test]
#[ignore = "diagnostic: allocates a ~1GB weight (safe - passes cleanly in ~0.6s, see docs/updates/0615-\
            alibi-real-gpu-parity-wgpu.md); run manually with --ignored"]
fn large_vocab_matmul_alone_does_not_hang() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hidden, vocab) = (1024usize, 250880usize);
    let b = Builder::new();
    let hstate = b.constant("hidden", TensorType::f32(vec![1, 1, hidden]));
    let w = b.constant("lm_head.weight.t", TensorType::f32(vec![hidden, vocab]));
    let out = b.matmul(hstate, w); // [1,1,vocab]
    let (hi, wi) = (hstate.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(hi, HostTensor::f32(vec![1, 1, hidden], fill(1, hidden)));
    inputs.insert(
        wi,
        HostTensor::f32(vec![hidden, vocab], fill(2, hidden * vocab)),
    );

    let t0 = std::time::Instant::now();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    eprintln!("large lm_head matmul alone: {:?}", t0.elapsed());
    assert_eq!(got.len(), vocab);
}

/// Same as `large_vocab_lm_head_transpose_matmul_does_not_hang` at `team-lucid/mptk-1b` scale
/// (`vocab=50432`, `hidden=2048`): checks whether the tied-lm_head transpose+matmul hang is a plain size
/// trigger (MPT ties `lm_head` via `b.transpose(embed, vec![1, 0])` in `crates/poot-models/src/mpt.rs`)
/// without loading the ~5.2GB checkpoint.
#[test]
#[ignore = "diagnostic: allocates a large weight (safe - passes cleanly in ~2s, see docs/updates/0615-\
            alibi-real-gpu-parity-wgpu.md); run manually with --ignored"]
fn mpt_scale_tied_lm_head_transpose_matmul_does_not_hang() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hidden, vocab) = (2048usize, 50432usize);
    let b = Builder::new();
    let hstate = b.constant("hidden", TensorType::f32(vec![1, 1, hidden]));
    let w = b.constant("wte.weight", TensorType::f32(vec![vocab, hidden]));
    let wt = b.transpose(w, vec![1, 0]); // [hidden, vocab] - the tied-lm_head pattern
    let out = b.matmul(hstate, wt); // [1,1,vocab]
    let (hi, wi) = (hstate.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(hi, HostTensor::f32(vec![1, 1, hidden], fill(1, hidden)));
    inputs.insert(
        wi,
        HostTensor::f32(vec![vocab, hidden], fill(2, vocab * hidden)),
    );

    let t0 = std::time::Instant::now();
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    eprintln!(
        "mpt-scale tied lm_head transpose+matmul: {:?}",
        t0.elapsed()
    );
    assert_eq!(got.len(), vocab);
    assert!(got.iter().all(|v| v.is_finite()));
}
