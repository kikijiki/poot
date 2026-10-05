//! Stress tests for GPU backend edge cases (card 035): boundary conditions that could trigger off-by-one,
//! buffer overread, or workgroup sizing bugs (large tensors >64/>256 elements, non-power-of-2 dimensions,
//! single-element tensors, maximum practical dimensions).

use poot_runtime_common::DeviceBackend;
use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Executor, HostView, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_tensor::HostTensor;

use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, SlotKey, Storage};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_test_util::max_abs_error;

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
fn split_consts_and_slots_t(
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
                let dtype = m.aval.dtype;
                let bytes: Vec<u8> = if dtype == DType::I32 {
                    t.as_i32()
                        .expect("I32-declared const needs an I32 HostTensor")
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
                let dense =
                    DenseWeight::try_new(dtype, t.shape().to_vec(), Arc::from(bytes)).unwrap();
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

/// [`split_consts_and_slots_t`]'s `HashMap<ValueId, Value>` sibling, for the handful of tests here that
/// build a `Value`-typed bind directly (`Tensor::int`/`Tensor::new` wrapped with `.into()`).
fn split_consts_and_slots_v(
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
                let dtype = m.aval.dtype;
                let bytes: Vec<u8> = if dtype == DType::I32 {
                    t.as_i32()
                        .expect("I32-declared const needs an I32 HostTensor")
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
                let dense =
                    DenseWeight::try_new(dtype, t.shape().to_vec(), Arc::from(bytes)).unwrap();
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

/// The executor-contract replacement for `GpuExecutor::run`/`run_resident_kv` on a stateless graph: a
/// one-shot step through `Engine<WgpuDevice>::add_entry`/`step`, torn back down afterward.
fn run_once_contract(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
) -> Vec<f32> {
    let program = staged(g, target);
    let (store, slot_binds) = split_consts_and_slots_t(g, inputs);
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

/// [`run_once_contract`]'s `Value`-bind sibling.
fn run_once_contract_v(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Vec<f32> {
    let program = staged(g, target);
    let (store, slot_binds) = split_consts_and_slots_v(g, inputs);
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

// Large tensors

#[test]
fn elementwise_large_256_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 256usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![n], fill(1001, n)));
    inputs.insert(ci, HostTensor::f32(vec![n], fill(1002, n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "add_256", 1e-5);
}

#[test]
fn elementwise_large_1024_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 1024usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Mul, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![n], fill(2001, n)));
    inputs.insert(ci, HostTensor::f32(vec![n], fill(2002, n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "mul_1024", 1e-5);
}

// Non-power-of-2 dimensions

#[test]
fn non_power_of_2_elementwise_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // 7, 13, 33, 50 are all non-power-of-2.
    for &n in &[7usize, 13, 33, 50] {
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![n]));
        let c = b.constant("c", TensorType::f32(vec![n]));
        let out = b.binary(BinOp::Add, a, c);
        let (ai, ci) = (a.id, c.id);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        inputs.insert(ai, HostTensor::f32(vec![n], fill(3000 + n as u64, n)));
        inputs.insert(ci, HostTensor::f32(vec![n], fill(4000 + n as u64, n)));

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
        assert_close(&got, cpu.as_f32().unwrap(), &format!("add_n{n}"), 1e-5);
    }
}

#[test]
fn non_power_of_2_reduce_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols) = (3usize, 7usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Sum, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        HostTensor::f32(vec![rows, cols], fill(5001, rows * cols)),
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
    assert_close(&got, cpu.as_f32().unwrap(), "reduce_3x7", 1e-4);
}

#[test]
fn non_power_of_2_matmul_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (m, k, n) = (3usize, 7usize, 5usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![m, k], fill(6001, m * k)));
    inputs.insert(wi, HostTensor::f32(vec![k, n], fill(6002, k * n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "matmul_3x7x5", 1e-4);
}

// Single-element tensors

#[test]
fn single_element_add_gpu_matches_cpu() {
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
    assert_close(&got, cpu.as_f32().unwrap(), "add_1elem", 1e-6);
}

#[test]
fn single_element_matmul_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // 1x1 @ 1x1 = scalar matmul.
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 1]));
    let w = b.constant("w", TensorType::f32(vec![1, 1]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1, 1], vec![5.0]));
    inputs.insert(wi, HostTensor::f32(vec![1, 1], vec![3.0]));

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
    assert_close(&got, cpu.as_f32().unwrap(), "matmul_1x1", 1e-6);
}

#[test]
fn single_element_reduce_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1]));
    let out = b.reduce(RedOp::Sum, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, 1], vec![42.0]));

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
    assert_close(&got, cpu.as_f32().unwrap(), "reduce_1x1", 1e-6);
}

// Maximum practical dimensions

#[test]
fn large_matmul_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // 64x64 @ 64x64 = 4096-element output, well within practical limits.
    let (m, k, n) = (64usize, 64usize, 64usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![m, k], fill(7001, m * k)));
    inputs.insert(wi, HostTensor::f32(vec![k, n], fill(7002, k * n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "matmul_64x64x64", 1e-3);
}

#[test]
fn large_reduce_wide_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, cols) = (8usize, 256usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Sum, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        HostTensor::f32(vec![rows, cols], fill(8001, rows * cols)),
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
    // Wide reduction: sum order differs, scale tolerance.
    let tol = 1e-4 * (cols as f32).sqrt();
    assert_close(&got, cpu.as_f32().unwrap(), "reduce_8x256", tol);
}

// Dimension=1 edge cases

#[test]
fn row_vector_matmul_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // 1xK @ KxN = row-vector times matrix.
    let (k, n) = (16usize, 8usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1, k], fill(9001, k)));
    inputs.insert(wi, HostTensor::f32(vec![k, n], fill(9002, k * n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "rowvec_matmul", 1e-4);
}

#[test]
fn column_vector_matmul_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // MxK @ Kx1 = matrix times column-vector.
    let (m, k) = (8usize, 16usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, 1]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![m, k], fill(10001, m * k)));
    inputs.insert(wi, HostTensor::f32(vec![k, 1], fill(10002, k)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "colvec_matmul", 1e-4);
}

// Multi-axis stress

#[test]
fn high_dimensional_broadcast_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    // [2,3,4,5] + [5] broadcast.
    let shape = vec![2usize, 3, 4, 5];
    let n: usize = shape.iter().product();
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(shape.clone()));
    let bias = b.constant("bias", TensorType::f32(vec![5]));
    let out = b.binary(BinOp::Add, a, bias);
    let (ai, bi) = (a.id, bias.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(shape, fill(11001, n)));
    inputs.insert(bi, HostTensor::f32(vec![5], fill(11002, 5)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "4d_broadcast", 1e-5);
}

#[test]
fn rmsnorm_non_power_of_2_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    use poot_graph_ir::ops::rmsnorm;
    let n = 33usize; // non-power-of-2
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = rmsnorm(&b, x, w, 1e-6);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, n], fill(12001, n)));
    inputs.insert(
        wi,
        HostTensor::f32(
            vec![n],
            fill(12002, n).iter().map(|v| 1.0 + v.abs()).collect(),
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
    assert_close(&got, cpu.as_f32().unwrap(), "rmsnorm_33", 1e-4);
}

// Fused chain stress

#[test]
fn deep_fused_chain_gpu_matches_cpu() {
    use poot_graph_plan::{cse, fuse};
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = 48usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    // Deep chain: multiple ops that should fuse into one kernel.
    let x = b.binary(BinOp::Add, a, c);
    let x = b.binary(BinOp::Mul, x, a);
    let x = b.unary(UnOp::Neg, x);
    let x = b.binary(BinOp::Add, x, c);
    let x = b.binary(BinOp::Mul, x, x);
    let (ai, ci) = (a.id, c.id);
    let g = fuse(&cse(&b.finish(x)));

    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![n], fill(13001, n)));
    inputs.insert(ci, HostTensor::f32(vec![n], fill(13002, n)));

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
    assert_close(&got, cpu.as_f32().unwrap(), "deep_fused", 1e-4);
}

// Card 159 Inc 3: batched KV-write grid > 65535 workgroups

/// Card 159 Inc 3 synthetic repro: force `scatter_update`'s dispatch grid past wgpu's 65535-per-dim workgroup
/// cap so the `elementwise_2d_grid` 2-D fold (`poot-graph-plan`) engages. At `WORKGROUP_SIZE=256` the fold only
/// kicks in above `out_numel = 65535*256 = 16_776_960`, so `pool` is just over that with `d=1` (~64MB f32).
/// Compares `run_resident_kv` (per-dispatch) bit-exact against a plain CPU `poot_eval::eval`, so a
/// wrong-but-self-consistent GPU result is caught as well as a hang.
#[test]
fn wgpu_scatter_update_grid_over_65535_workgroups() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    // out_numel = pool * d must exceed 65535*256 = 16_776_960.
    let d = 1usize;
    let pool = 16_776_960 / d + 4096; // 16_781_056 elements, ~64MB f32 buffers.
    let n = 4usize; // small src row count.

    let inv_data: Vec<f32> = (0..pool)
        .map(|i| {
            if i % 5 == 0 {
                -1.0
            } else {
                ((i / 5) % n) as f32
            }
        })
        .collect();

    let b = Builder::new();
    let base = b.constant("big_sc_base", TensorType::f32(vec![pool, d]));
    let src = b.constant("big_sc_src", TensorType::f32(vec![n, d]));
    let inv = b.constant("big_sc_inv", TensorType::f32(vec![pool]));
    let out = b.scatter_update(base, src, inv);
    let g = b.finish(out);
    let g = poot_graph_plan::cse(&g);

    let base_data = fill(1, pool * d);
    let src_data = fill(2, n * d);

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        base.id,
        HostTensor::f32(vec![pool, d], base_data.clone()).into(),
    );
    inputs.insert(src.id, HostTensor::f32(vec![n, d], src_data.clone()).into());
    inputs.insert(inv.id, HostTensor::f32(vec![pool], inv_data.clone()).into());

    // CPU oracle first (independent of any GPU dispatch-grid math).
    let cpu_out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("cpu eval scatter_update")
        .output
        .into_host()
        .expect("cpu eval scatter_update output is dense");

    eprintln!(
        "wgpu_scatter_update_grid_over_65535_workgroups: out_numel={} \
         (total_groups={}, expect x_groups=65535 y_groups=2)",
        pool * d,
        (pool * d).div_ceil(256)
    );

    let per_out = run_once_contract_v(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(per_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in per_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            c.to_bits(),
            "scatter_update per-dispatch elem {i}: gpu {a} vs cpu {c}"
        );
    }

    eprintln!(
        "wgpu_scatter_update_grid_over_65535_workgroups: OK ({} elements, bit-exact vs CPU)",
        pool * d
    );
}

/// Card 159 Inc 3 synthetic repro, `dynamic_update_slice` (dynamic buffer-index) sibling of
/// [`wgpu_scatter_update_grid_over_65535_workgroups`]. The same grid-cap logic
/// (`is_elementwise_2d`/`elementwise_2d_grid`) applies to the `DynamicUpdateSlice` runtime-index `ComputeMeta`
/// form.
#[test]
fn wgpu_dyn_update_slice_grid_over_65535_workgroups() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    let d = 1usize;
    let cap = 16_776_960 / d + 4096; // > 65535*256 elements.
    let pos = cap / 2;

    let b = Builder::new();
    let cache = b.constant("big_dus_cache", TensorType::f32(vec![cap, d]));
    let update = b.constant("big_dus_update", TensorType::f32(vec![1, d]));
    let idx_val = b.constant("big_dus_idx", TensorType::f32(vec![]));
    let out = b.dynamic_update_slice_dyn(cache, update, idx_val, 0);
    let g = b.finish(out);
    let g = poot_graph_plan::cse(&g);

    let cache_data = fill(3, cap * d);
    let update_data = fill(4, d);

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(cache.id, HostTensor::f32(vec![cap, d], cache_data).into());
    inputs.insert(update.id, HostTensor::f32(vec![1, d], update_data).into());
    inputs.insert(idx_val.id, HostTensor::scalar(pos as f32).into());

    let cpu_out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("cpu eval dyn_update_slice")
        .output
        .into_host()
        .expect("cpu eval dyn_update_slice output is dense");

    eprintln!(
        "wgpu_dyn_update_slice_grid_over_65535_workgroups: out_numel={} \
         (total_groups={}, expect x_groups=65535 y_groups=2)",
        cap * d,
        (cap * d).div_ceil(256)
    );

    let per_out = run_once_contract_v(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(per_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in per_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            c.to_bits(),
            "dyn_update_slice per-dispatch elem {i}: gpu {a} vs cpu {c}"
        );
    }

    eprintln!(
        "wgpu_dyn_update_slice_grid_over_65535_workgroups: OK ({} elements, bit-exact vs CPU)",
        cap * d
    );
}

/// Card 159 Inc 3 repro #3: the shape that hangs in the batched gemma4-dense serve is not one over-cap
/// `dynamic_update_slice_dyn` dispatch (the single-dispatch repros above pass) but
/// `scatter_shared_pool`'s chain of `batch` such dispatches per layer, each folded onto
/// `pool` and reading the prior dispatch's full output as its `operand` (the `(0..batch).fold(pool, ...)` in
/// `crates/poot-models/src/qwen2.rs`). This mirrors that fold (3-D `[pool_slots, hkv, d]` state, axis-0
/// single-row `dynamic_update_slice_dyn` per row, `batch` rows chained in one graph) at gemma4-dense local-layer
/// sizing (`pool_slots=4097, hkv=16, d=256` -> `16_781_312 > 65535*256`), so every dispatch takes the 2-D fold.
#[test]
fn wgpu_scatter_shared_pool_chained_dus_grid_over_65535_workgroups() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    let pool_slots = 4097usize;
    let hkv = 16usize;
    let d = 256usize;
    let batch = 4usize;
    let out_numel = pool_slots * hkv * d;
    assert!(
        out_numel > 65535 * 256,
        "sizing must force the 2-D fold: {out_numel}"
    );

    let b = Builder::new();
    let pool0 = b.constant("big_pool_kcache", TensorType::f32(vec![pool_slots, hkv, d]));
    let update = b.constant("big_pool_update", TensorType::f32(vec![batch, hkv, 1, d]));
    // one physical slot index per batch row, as a [batch] i32 constant (mirrors scatter_shared_pool's `phys_row`
    // result, baked directly since only the resulting per-row physical index matters here).
    let phys = b.constant("big_pool_phys", TensorType::new(vec![batch], DType::I32));

    let mut pool = pool0;
    for row in 0..batch {
        let update_row = b.slice(update, 0, row, row + 1); // [1, hkv, 1, d]
        let update_row = b.reshape(update_row, vec![1, hkv, d]); // [1, hkv, d]
        let phys_row = b.reshape(b.slice(phys, 0, row, row + 1), vec![]); // scalar physical slot
        pool = b.dynamic_update_slice_dyn(pool, update_row, phys_row, 0);
    }
    let g = b.finish(pool);
    let g = poot_graph_plan::cse(&g);

    let pool_data = fill(1, out_numel);
    let update_data = fill(2, batch * hkv * d);
    // distinct, in-range, non-colliding physical slots for each row.
    let phys_data: Vec<i32> = (0..batch)
        .map(|row| (100 + row * 37) as i32 % pool_slots as i32)
        .collect();

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        pool0.id,
        HostTensor::f32(vec![pool_slots, hkv, d], pool_data).into(),
    );
    inputs.insert(
        update.id,
        HostTensor::f32(vec![batch, hkv, 1, d], update_data).into(),
    );
    inputs.insert(phys.id, HostTensor::i32(vec![batch], phys_data).into());

    let cpu_out = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("cpu eval chained scatter_shared_pool")
        .output
        .into_host()
        .expect("cpu eval chained scatter_shared_pool output is dense");

    eprintln!(
        "wgpu_scatter_shared_pool_chained_dus_grid_over_65535_workgroups: out_numel={out_numel} \
         batch={batch} (total_groups={}, expect x_groups=65535 y_groups=2 PER dispatch)",
        out_numel.div_ceil(256)
    );

    let per_out = run_once_contract_v(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(per_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in per_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            c.to_bits(),
            "chained scatter_shared_pool per-dispatch elem {i}: gpu {a} vs cpu {c}"
        );
    }

    eprintln!(
        "wgpu_scatter_shared_pool_chained_dus_grid_over_65535_workgroups: OK \
         ({out_numel} elements x {batch} chained dispatches, bit-exact vs CPU)"
    );
}

/// Card 159 Inc 3 repro: a `Binary` broadcast op past the 65535-workgroup ceiling. This is the op family that
/// crashed the gemma4-dense batched decode step, via the same `Binary`-broadcast mechanism `attention_masked`
/// composes with (the additive mask add itself is 32768 elements at real gemma4 config, under the cap) and the
/// family `is_elementwise_2d`/`binary_broadcast_dt_views_grid` covers. Shape: `[1, hq, 1, cap] + [1, 1, 1, cap]`
/// at `hq=1024, cap=16384` -> `out_numel = 16_777_216`, one workgroup (256 elements) over the
/// `65535*256 = 16_776_960` ceiling, the same margin as the real crash (`Broadcast`/`Transpose` at
/// `1*32*512*1024` elements, see `poot-graph-plan::is_elementwise_2d`). Small buffers (~64MB).
#[test]
fn wgpu_binary_broadcast_grid_over_65535_workgroups() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    let cap = 16_384usize;
    let hq = 16_777_216 / cap; // 1024; out_numel = hq*cap = 16_777_216 > 65535*256.
    assert!(
        hq * cap > 65_535 * 256,
        "shape must exceed the wg-bump-only ceiling"
    );

    let b = Builder::new();
    let scores = b.constant("big_bin_scores", TensorType::f32(vec![1, hq, 1, cap]));
    let mask = b.constant("big_bin_mask", TensorType::f32(vec![1, 1, 1, cap]));
    let out = b.binary(BinOp::Add, scores, mask);
    let g = b.finish(out);
    let g = poot_graph_plan::cse(&g);

    let scores_data = fill(3, hq * cap);
    let mask_data = fill(4, cap);

    let mut inputs = HashMap::new();
    inputs.insert(scores.id, HostTensor::f32(vec![1, hq, 1, cap], scores_data));
    inputs.insert(mask.id, HostTensor::f32(vec![1, 1, 1, cap], mask_data));

    // CPU oracle first (independent of any GPU dispatch-grid math).
    let cpu_out = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval binary broadcast")
    .output
    .into_host()
    .expect("cpu eval binary broadcast output is dense");

    eprintln!(
        "wgpu_binary_broadcast_grid_over_65535_workgroups: out_numel={} \
         (total_groups={}, expect x_groups=65535 y_groups=2)",
        hq * cap,
        (hq * cap).div_ceil(256)
    );

    let gpu_out = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(gpu_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in gpu_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            c.to_bits(),
            "wgpu_binary_broadcast_grid_over_65535_workgroups elem {i}: gpu {a} vs cpu {c}"
        );
    }

    eprintln!(
        "wgpu_binary_broadcast_grid_over_65535_workgroups: OK ({} elements, bit-exact vs CPU)",
        hq * cap
    );
}

/// Card 159 Inc 3 repro: a `Transpose` past the 65535-workgroup ceiling, feeding a `MatMul` consumer (so
/// `compute_views` cannot promote it to a zero-dispatch view: a `MatMul` operand is not strided-capable, as with
/// the gemma4 K^T transpose before `Q @ K^T`). This is the op+shape that crashed the gemma4-dense batched
/// decode step at `POOT_SLOTS=1`/`POOT_CAP=1024`, found by scanning the real optimized graph (the `Binary`
/// mask add measures only 32768 elements at `Hq=32`; see `poot-graph-plan::is_elementwise_2d`). Shape:
/// `k [1,32,1024,512] -transpose(0,1,3,2)-> kt [1,32,512,1024]` (16_777_216 elements, 256 over the `65535*256`
/// ceiling), then `q [1,32,1,512] @ kt -> [1,32,1,1024]` so the transpose is not dead code.
#[test]
fn wgpu_transpose_grid_over_65535_workgroups_feeding_matmul() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    let (hq, cap, d) = (32usize, 1024usize, 512usize);
    let k_numel = hq * cap * d;
    assert_eq!(
        k_numel, 16_777_216,
        "matches the real gemma4 global-layer K read shape"
    );
    assert!(
        k_numel > 65_535 * 256,
        "shape must exceed the wg-bump-only ceiling"
    );

    let b = Builder::new();
    let k = b.constant("big_tr_k", TensorType::f32(vec![1, hq, cap, d]));
    let q = b.constant("big_tr_q", TensorType::f32(vec![1, hq, 1, d]));
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,hq,d,cap]
    let out = b.matmul(q, kt); // [1,hq,1,cap]
    let g = b.finish(out);
    let g = poot_graph_plan::cse(&g);

    let k_data = fill(5, k_numel);
    let q_data = fill(6, hq * d);

    let mut inputs = HashMap::new();
    inputs.insert(k.id, HostTensor::f32(vec![1, hq, cap, d], k_data));
    inputs.insert(q.id, HostTensor::f32(vec![1, hq, 1, d], q_data));

    // CPU oracle first (independent of any GPU dispatch-grid math).
    let cpu_out = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval transpose+matmul")
    .output
    .into_host()
    .expect("cpu eval transpose+matmul output is dense");

    eprintln!(
        "wgpu_transpose_grid_over_65535_workgroups_feeding_matmul: transpose out_numel={k_numel} \
         (total_groups={}, expect x_groups=65535 y_groups=2)",
        k_numel.div_ceil(256)
    );

    let gpu_out = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(gpu_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in gpu_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        let t = 1e-3 * c.abs().max(1e-3);
        assert!(
            (a - c).abs() <= t,
            "wgpu_transpose_grid_over_65535_workgroups_feeding_matmul elem {i}: gpu {a} vs cpu {c} (tol={t})"
        );
    }

    eprintln!(
        "wgpu_transpose_grid_over_65535_workgroups_feeding_matmul: OK ({k_numel} transpose elements, matches CPU)"
    );
}

/// Card 159 Inc 3 repro: a 2-input `Concat` past the 65535-workgroup ceiling. This crashed the gemma4-dense
/// batched (n_slots=4) decode step's on-device argmax path (`GridCap([65536,1,1])`), not the argmax kernel
/// itself (one workgroup per batch row) but `gather_shared_pool`'s pairwise `concat(acc, row)` fold that
/// assembles each row's shared-pool K/V read into `[batch,Hkv,cap,D]`. Shape matches the gemma4-31B global
/// layer (`Hkv=4, D=512`) at `cap=1024`: two `[1,4,1024,512]` reads concat on axis 0 to `[2,4,1024,512]` =
/// 4_194_304 elements, `groups=65536` at the concat2 kernel's original `WORKGROUP_SIZE=64` (one workgroup over
/// the 65535 cap).
#[test]
fn wgpu_concat2_grid_over_65535_workgroups() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };

    let (hkv, cap, d) = (4usize, 1024usize, 512usize);
    let row_numel = hkv * cap * d;
    let out_numel = 2 * row_numel;
    assert_eq!(
        out_numel, 4_194_304,
        "matches the real gemma4 global-layer shared-pool gather concat fold shape"
    );
    assert!(
        out_numel.div_ceil(64) > 65_535,
        "shape must exceed the default-wg (64) ceiling that the pre-fix concat2 kernel used"
    );

    let b = Builder::new();
    let row0 = b.constant("big_cat_row0", TensorType::f32(vec![1, hkv, cap, d]));
    let row1 = b.constant("big_cat_row1", TensorType::f32(vec![1, hkv, cap, d]));
    let out = b.concat(0, &[row0, row1]); // [2,hkv,cap,d]
    let g = b.finish(out);
    let g = poot_graph_plan::cse(&g);

    let row0_data = fill(7, row_numel);
    let row1_data = fill(8, row_numel);

    let mut inputs = HashMap::new();
    inputs.insert(row0.id, HostTensor::f32(vec![1, hkv, cap, d], row0_data));
    inputs.insert(row1.id, HostTensor::f32(vec![1, hkv, cap, d], row1_data));

    // CPU oracle first (independent of any GPU dispatch-grid math).
    let cpu_out = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval concat2")
    .output
    .into_host()
    .expect("cpu eval concat2 output is dense");

    eprintln!(
        "wgpu_concat2_grid_over_65535_workgroups: out_numel={out_numel} \
         (total_groups={}, expect x_groups=65535 y_groups=1)",
        out_numel.div_ceil(256)
    );

    let gpu_out = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(g.aval(g.output).shape, cpu_out.shape());
    assert_eq!(gpu_out.len(), cpu_out.as_f32().unwrap().len());
    for (i, (a, c)) in gpu_out
        .iter()
        .zip(cpu_out.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            c.to_bits(),
            "wgpu_concat2_grid_over_65535_workgroups elem {i}: gpu {a} vs cpu {c}"
        );
    }

    eprintln!(
        "wgpu_concat2_grid_over_65535_workgroups: OK ({out_numel} elements, bit-exact vs CPU)"
    );
}

/// N that forces `ceil(N / 256) > 65535` after the plan's wg-bump to 256: one workgroup over the
/// device cap, so the host-side `fold_grid` must lay the launch out as `[65535, 2, 1]` and shared
/// `thread_index` reconstruction must rebuild the linear index. Old code (GridCap reject, no fold)
/// fails this size for any family not already plan-folded by `is_elementwise_2d`.
const GRID_FOLD_NUMEL: usize = 65_536 * 256; // 16_777_216 = 2^24

/// Elementwise family past the workgroup cap: `Cast(I32 -> F32)` is one-thread-per-element via
/// `thread_index` and is not covered by `is_elementwise_2d`, so it takes the new host fold + shared
/// reconstruction path. Checks every element including the fold tail.
#[test]
fn wgpu_cast_grid_over_65535_workgroups_every_element() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let n = GRID_FOLD_NUMEL;
    assert!(
        n.div_ceil(256) > 65_535,
        "N={n} must force >65535 workgroups at wg=256 so the old GridCap path would fail"
    );

    let b = Builder::new();
    let src = b.constant("cast_src", TensorType::new(vec![n], DType::I32));
    let out = b.cast(src, DType::F32);
    let g = b.finish(out);

    // Exact for every lane at or below 2^24 (card 372c); n == 2^24 covers lanes 0..=n-1.
    let src_data: Vec<i32> = (0..n as i32).collect();
    let mut inputs = HashMap::new();
    inputs.insert(src.id, HostTensor::i32(vec![n], src_data.clone()));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval cast i32->f32")
    .output
    .into_host()
    .expect("cpu eval cast i32->f32 output is dense");
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(got.len(), n);
    for i in [0usize, 1, 65_534, 65_535, 65_536, n / 2, n - 2, n - 1] {
        let want = src_data[i] as f32;
        assert_eq!(
            got[i].to_bits(),
            want.to_bits(),
            "cast fold tail/elem {i}: gpu {} vs cpu {want}",
            got[i]
        );
    }
    assert_eq!(
        got.as_slice(),
        cpu.as_f32().unwrap(),
        "cast fold must match CPU on every element (including the y-row tail)"
    );
    eprintln!(
        "wgpu_cast_grid_over_65535_workgroups_every_element: OK ({n} elements, host-folded [65535,2,1])"
    );
}

/// Other affected family past the workgroup cap: axis-0 `Gather` is also one-thread-per-element via
/// `thread_index` and not in `is_elementwise_2d`. Shape: `index` of 65536 rows x `rest` 256 =
/// 16_777_216 outputs. Checks every element including the fold tail.
#[test]
fn wgpu_gather_grid_over_65535_workgroups_every_element() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (rows, rest) = (65_536usize, 256usize);
    let n = rows * rest;
    assert_eq!(n, GRID_FOLD_NUMEL);
    assert!(
        n.div_ceil(256) > 65_535,
        "gather N={n} must force >65535 workgroups at wg=256"
    );

    let b = Builder::new();
    // Tiny table (all indices 0): axis-0 gather only needs data[index[s], :].
    let table = b.constant("gather_table", TensorType::f32(vec![1, rest]));
    let index = b.constant("gather_index", TensorType::f32(vec![rows]));
    let out = b.gather(table, 0, index);
    let g = b.finish(out);

    let table_data = fill(11, rest);
    let index_data = vec![0f32; rows]; // every row reads table row 0
    let mut inputs = HashMap::new();
    inputs.insert(table.id, HostTensor::f32(vec![1, rest], table_data.clone()));
    inputs.insert(index.id, HostTensor::f32(vec![rows], index_data));

    let cpu = eval(
        &g,
        &inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect::<HashMap<_, _>>(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval gather")
    .output
    .into_host()
    .expect("cpu eval gather output is dense");
    let got = run_once_contract(&mut *exec, target, &g, &inputs);
    assert_eq!(got.len(), n);
    for i in [0usize, 1, 65_534, 65_535, 65_536, n / 2, n - 2, n - 1] {
        let want = table_data[i % rest];
        assert_eq!(
            got[i].to_bits(),
            want.to_bits(),
            "gather fold tail/elem {i}: gpu {} vs cpu {want}",
            got[i]
        );
    }
    assert_eq!(
        got.as_slice(),
        cpu.as_f32().unwrap(),
        "gather fold must match CPU on every element (including the y-row tail)"
    );
    eprintln!(
        "wgpu_gather_grid_over_65535_workgroups_every_element: OK ({n} elements, host-folded [65535,2,1])"
    );
}

// ---- Card 550 SC-003: family-neutral sliding-window mask-from-Pos fixture ----

/// Card 550 SC-003: a family-neutral fixture (no LLM architecture involved) whose mask is
/// `causal_mask_from_pos` with a real window (`w < l`), fed `Slot::Pos` as a step input rather than a
/// host-built mask constant, through the one executor contract. Matches the CPU oracle on wgpu.
/// Mutation (recorded in the card's final report): building the mask on the host instead of from
/// `Slot::Pos` turns this row red, since the fixture's mask would no longer react to the position input.
#[test]
fn sliding_window_mask_from_pos_gpu_matches_cpu() {
    let _g = gpu_lock();
    let Some((mut exec, target)) = try_gpu() else {
        return;
    };
    let (hq, hkv, l, window) = (2usize, 1usize, 6usize, 3usize);
    let n_rep = hq / hkv;
    let d = 4usize;
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let pos = b.slot(
        poot_graph_ir::Slot::Pos,
        TensorType::new(vec![1, l], DType::I32),
    );
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, l, Some(window));
    let out = poot_graph_ir::ops::attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, posi) = (q.id, k.id, v.id, pos.id);
    let g = b.finish(out);

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        qi,
        HostTensor::f32(vec![1, hq, l, d], fill(20001, hq * l * d)).into(),
    );
    inputs.insert(
        ki,
        HostTensor::f32(vec![1, hkv, l, d], fill(20002, hkv * l * d)).into(),
    );
    inputs.insert(
        vi,
        HostTensor::f32(vec![1, hkv, l, d], fill(20003, hkv * l * d)).into(),
    );
    inputs.insert(
        posi,
        HostTensor::i32(vec![1, l], (0..l as i32).collect()).into(),
    );

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_contract_v(&mut *exec, target, &g, &inputs);
    assert_close(
        &got,
        cpu.as_f32().unwrap(),
        "sliding_window_mask_from_pos",
        1e-4,
    );

    // Sensitivity half of the mutation note: the mask must actually come from `Slot::Pos`, not a
    // host-frozen constant. Rebind the same compiled graph with every row at position 2 instead of
    // the increasing 0..l (collapses the staircase mask into one row's window repeated for every
    // query) and confirm the GPU output moves with it, matching its own fresh CPU oracle. A
    // host-built mask that ignored `Slot::Pos` would return the first run's output unchanged here.
    let mut inputs2 = inputs.clone();
    inputs2.insert(posi, HostTensor::i32(vec![1, l], vec![2i32; l]).into());
    let cpu2 = eval(&g, &inputs2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got2 = run_once_contract_v(&mut *exec, target, &g, &inputs2);
    assert_close(
        &got2,
        cpu2.as_f32().unwrap(),
        "sliding_window_mask_from_pos (shifted Pos)",
        1e-4,
    );
    assert!(
        max_abs_error(&got, &got2) > 1e-3,
        "the fixture's output did not change when Slot::Pos changed; its mask may not be reading Pos"
    );
}
