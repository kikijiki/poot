//! Cross-backend correctness tests for ROCm (card 035): the same graph runs on ROCm and the CPU
//! oracle and outputs are compared element-wise within f32 tolerance. Skips cleanly when ROCm/HSA is
//! unavailable.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::{attention_masked, attention_prefill};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, Storage};
use poot_graph_plan::Target;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_gpu::device::RocmDevice;
use poot_tensor::DType;
use poot_tensor::HostTensor;

fn try_rocm() -> Option<(Target, Engine<RocmDevice>)> {
    let device = poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
    let target = device.target();
    Some((target, Engine::new(device)))
}

/// Run `g` once through the executor contract: every `Storage::Const` input `inputs` names becomes
/// a `WeightStore` entry (Card 546a binds consts from the executable's store, never a per-step
/// value), every other graph input must be a `Storage::Slot` bound through an empty-free `step`
/// (none of this file's fixtures declare one). Mirrors `run_resident`'s old one-shot eager contract
/// on the new one-shot contract-entry shape.
fn run_once_rocm(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> HostTensor {
    let mut builder = WeightStore::builder();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage != Storage::Const {
            panic!(
                "run_once_rocm: fixture graph has a non-Const input {id} ({:?})",
                meta.storage
            );
        }
        let name = meta.name.clone().expect("named const");
        let value = inputs
            .get(&id)
            .unwrap_or_else(|| panic!("run_once_rocm: no input bound for const {name}"));
        let tensor = value.as_host().expect("dense const input");
        let bytes: Arc<[u8]> = Arc::from(
            tensor
                .as_f32()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let dense = DenseWeight::try_new(DType::F32, tensor.shape().to_vec(), bytes).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let store = Arc::new(builder.build());
    let staged = {
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
    };
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();
    let bytes = engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    engine.unload(exe).unwrap();
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    HostTensor::f32(g.aval(g.output).shape.clone(), data)
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
            "{label} elem {i}: rocm {a} vs cpu {b} (tol={t})"
        );
    }
}

/// Absolute+relative tolerance compare, for reference values near zero from cancellation (the
/// relative-only floor of `assert_close` is too tight there).
fn assert_close_abs_rel(got: &[f32], reference: &[f32], label: &str, abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
    for (i, (a, r)) in got.iter().zip(reference.iter()).enumerate() {
        let t = abs_tol + rel_tol * r.abs();
        assert!(
            (a - r).abs() <= t,
            "{label} elem {i}: rocm {a} vs ref {r} (tol={t})"
        );
    }
}

/// Real-magnitude deterministic fill: values spread roughly in [-30, 30], every 7th element scaled
/// by 1e-3 to mix magnitudes (stresses cancellation/precision in a reduce, unlike the tiny [-1, 1]
/// `fill` above).
fn fill_real_magnitude(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..count)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            let base = (u * 2.0 - 1.0) * 30.0; // [-30, 30]
            if i % 7 == 0 { base * 1e-3 } else { base }
        })
        .collect()
}

/// Independent f64 host fold (sum), row-major [rows, cols] reduced over the last axis; a second
/// reference separate from the eval-oracle CPU path.
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

/// Independent f64 host fold (max), row-major [rows, cols] reduced over the last axis.
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

fn make_inputs(ids: &[(ValueId, Vec<usize>, Vec<f32>)]) -> HashMap<ValueId, Value> {
    ids.iter()
        .map(|(id, shape, data)| (*id, HostTensor::f32(shape.clone(), data.clone()).into()))
        .collect()
}

// Elementwise ops

#[test]
fn elementwise_add_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 32usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let ad = fill(42, n);
    let cd = fill(99, n);
    let inputs = make_inputs(&[(ai, vec![n], ad.clone()), (ci, vec![n], cd.clone())]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_add",
        1e-5,
    );
}

#[test]
fn elementwise_mul_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 32usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Mul, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[(ai, vec![n], fill(201, n)), (ci, vec![n], fill(202, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_mul",
        1e-5,
    );
}

#[test]
fn elementwise_neg_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 32usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let out = b.unary(UnOp::Neg, x);
    let xi = x.id;
    let g = b.finish(out);

    let inputs = make_inputs(&[(xi, vec![n], fill(301, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_neg",
        1e-5,
    );
}

// Matmul

#[test]
fn matmul_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    // Small dims that fit one AQL (m*n <= 64).
    let (m, k, n) = (4usize, 8usize, 4usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[
        (ai, vec![m, k], fill(401, m * k)),
        (wi, vec![k, n], fill(402, k * n)),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_matmul",
        1e-4,
    );
}

// Reduce

#[test]
fn reduce_sum_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (rows, cols) = (2usize, 16usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Sum, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let inputs = make_inputs(&[(xi, vec![rows, cols], fill(501, rows * cols))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    let tol = 1e-4 * (cols as f32).sqrt();
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_reduce_sum",
        tol,
    );
}

#[test]
fn reduce_max_last_axis_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (rows, cols) = (2usize, 16usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let out = b.reduce(RedOp::Max, x, 1, false);
    let xi = x.id;
    let g = b.finish(out);

    let inputs = make_inputs(&[(xi, vec![rows, cols], fill(511, rows * cols))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    // Max is exact (order-independent).
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_reduce_max",
        1e-6,
    );
}

// Reduce: non-power-of-2, wide, real-magnitude
//
// Covers two gaps: the ROCm "non_power_of_2" test exercised BinOp::Add, not Reduce, and every
// reduce test used tiny [-1, 1] data that never stresses cancellation. Widths 33, 896, 1536 are
// non-powers-of-2 and match real transformer hidden dims.

#[test]
fn reduce_sum_nonpow2_wide_real_magnitude_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let rows = 3usize;
    for (i, &cols) in [33usize, 896, 1536].iter().enumerate() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Sum, x, 1, false);
        let xi = x.id;
        let g = b.finish(out);

        let data = fill_real_magnitude(820 + i as u64, rows * cols);
        let inputs = make_inputs(&[(xi, vec![rows, cols], data.clone())]);

        let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = run_once_rocm(&mut rocm, target, &g, &inputs);
        let label = format!("rocm_reduce_sum_nonpow2_wide_cols{cols}");
        // Real-magnitude data (up to |30|) with mixed large/small terms: rounding noise is bounded by an
        // absolute+relative tolerance scaling with sqrt(cols), since the relative-only floor is too tight
        // when a row's true sum cancels near 0.
        let abs_tol = 0.02 * (cols as f32).sqrt();
        assert_close_abs_rel(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            &label,
            abs_tol,
            1e-4,
        );

        // Independent f64 host-fold cross-check (not the eval-oracle path) at the real hidden-dim width.
        if cols == 896 {
            let want_f64 = f64_row_sum(&data, rows, cols);
            assert_close_abs_rel(
                got.as_f32().unwrap(),
                &want_f64,
                "rocm_reduce_sum_nonpow2_wide_cols896_vs_f64",
                0.05,
                1e-4,
            );
        }
    }
}

#[test]
fn reduce_max_nonpow2_wide_real_magnitude_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let rows = 3usize;
    for (i, &cols) in [33usize, 896, 1536].iter().enumerate() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Max, x, 1, false);
        let xi = x.id;
        let g = b.finish(out);

        let data = fill_real_magnitude(920 + i as u64, rows * cols);
        let inputs = make_inputs(&[(xi, vec![rows, cols], data.clone())]);

        let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = run_once_rocm(&mut rocm, target, &g, &inputs);
        let label = format!("rocm_reduce_max_nonpow2_wide_cols{cols}");
        // Max is exact (order-independent) regardless of magnitude.
        assert_close_abs_rel(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            &label,
            1e-5,
            1e-6,
        );

        // Independent f64 host-fold cross-check at the real hidden-dim width.
        if cols == 896 {
            let want_f64 = f64_row_max(&data, rows, cols);
            assert_close_abs_rel(
                got.as_f32().unwrap(),
                &want_f64,
                "rocm_reduce_max_nonpow2_wide_cols896_vs_f64",
                1e-5,
                1e-6,
            );
        }
    }
}

// RMSNorm / LayerNorm (RowKernel cross-lane reduce)
//
// These compose Reduce(Sum, keepdim) and route through the RowKernel/LDS cross-lane reduce path on
// ROCm, the path that had the amdgcn cross-wave LDS-visibility bug (see
// amdgcn-barrier-needs-lds-fence). Non-power-of-2 width, real-magnitude input.

#[test]
fn rmsnorm_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 896usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = poot_graph_ir::ops::rmsnorm(&b, x, w, 1e-6);
    let (xi, wi) = (x.id, w.id);
    let g = b.finish(out);

    let xd = fill_real_magnitude(931, n);
    // Weights stay near typical learned-scale magnitude (not +-30).
    let wd: Vec<f32> = fill(932, n).iter().map(|v| 1.0 + v.abs()).collect();
    let inputs = make_inputs(&[(xi, vec![1, 1, n], xd), (wi, vec![n], wd)]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_rmsnorm_nonpow2",
        1e-4,
    );
}

#[test]
fn layernorm_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 896usize;
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let out = poot_graph_ir::ops::layernorm(&b, x, w, bias, 1e-6);
    let (xi, wi, bi) = (x.id, w.id, bias.id);
    let g = b.finish(out);

    let xd = fill_real_magnitude(941, n);
    let wd: Vec<f32> = fill(942, n).iter().map(|v| 1.0 + v.abs()).collect();
    let bd: Vec<f32> = fill(943, n).iter().map(|v| v * 0.1).collect();
    let inputs = make_inputs(&[
        (xi, vec![1, 1, n], xd),
        (wi, vec![n], wd),
        (bi, vec![n], bd),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_layernorm_nonpow2",
        1e-4,
    );
}

// Stress: non-power-of-2

#[test]
fn non_power_of_2_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 33usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[(ai, vec![n], fill(601, n)), (ci, vec![n], fill(602, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_npo2_33",
        1e-5,
    );
}

// Stress: single element

#[test]
fn single_element_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1]));
    let c = b.constant("c", TensorType::f32(vec![1]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[(ai, vec![1], vec![3.5]), (ci, vec![1], vec![2.5])]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_single",
        1e-6,
    );
}

// DynamicUpdateSlice (KV-cache slot write)
//
// Regression guard: a stray `_ => {}` catch-all shadowed the DUS interception in `exec_eqns`, so DUS
// fell through to the general layout path and wrote to the wrong destination. The decode probes
// call `disp_kv_scatter_*` directly, so nothing exercised the graph DUS path. This walks a DUS graph
// through `run_resident` and compares to the CPU oracle.

#[test]
fn dynamic_update_slice_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    // operand [rows, cols]; write one row of `update` at row `index` along axis 0.
    let (rows, cols, index) = (4usize, 4usize, 2usize);
    let b = Builder::new();
    let operand = b.constant("operand", TensorType::f32(vec![rows, cols]));
    let update = b.constant("update", TensorType::f32(vec![1, cols]));
    let out = b.dynamic_update_slice(operand, update, index, 0);
    let (oi, ui) = (operand.id, update.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[
        (oi, vec![rows, cols], fill(701, rows * cols)),
        (ui, vec![1, cols], fill(702, cols)),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    // Bit-exact: DUS is a pure copy, no arithmetic.
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_dus_axis0",
        0.0,
    );
}

// Cross-backend op-coverage parity: ops wgpu tests cross-backend but ROCm did not. Mirrors
// crates/poot-gpu/tests/cross_backend.rs. Each test creates one executor (one try_rocm): a second
// in-process `try_rocm` fails OUT_OF_RESOURCES because the ROCm binary leaks the HSA queue. Run
// each test in its own `cargo test` invocation.

#[test]
fn gather_axis0_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (n, d) = (8usize, 4usize);
    let idx_count = 3usize;
    let b = Builder::new();
    let data = b.constant("data", TensorType::f32(vec![n, d]));
    let idx = b.constant("idx", TensorType::f32(vec![idx_count]));
    let out = b.gather(data, 0, idx);
    let (di, ii) = (data.id, idx.id);
    let g = b.finish(out);
    let inputs = make_inputs(&[
        (di, vec![n, d], fill(901, n * d)),
        (ii, vec![idx_count], vec![3.0, 0.0, 6.0]),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_eq!(got.shape(), cpu.shape());
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_gather",
        1e-6,
    );
}

#[test]
fn matmul_bias_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (m, k, n) = (4usize, 8usize, 4usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let mm = b.matmul(a, w);
    let out = b.binary(BinOp::Add, mm, bias);
    let (ai, wi, bi) = (a.id, w.id, bias.id);
    let g = b.finish(out);
    let inputs = make_inputs(&[
        (ai, vec![m, k], fill(401, m * k)),
        (wi, vec![k, n], fill(402, k * n)),
        (bi, vec![n], fill(403, n)),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_matmul_bias",
        1e-4,
    );
}

// One unary op per test: `run_resident` is single-graph per executor (a second call on the same
// executor returns a zero buffer) and a second `try_rocm` leaks the HSA queue. Known inputs make a
// divergence unambiguous.
fn unary_rocm_case(op: UnOp, xd: Vec<f32>, label: &str) {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = xd.len();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let out = b.unary(op, x);
    let xi = x.id;
    let g = b.finish(out);
    let inputs = make_inputs(&[(xi, vec![n], xd)]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(got.as_f32().unwrap(), cpu.as_f32().unwrap(), label, 1e-4);
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

/// Card 630 SC-005 (ROCm): `Tanh` and `Erf` over the first-row table within tier 2 of the oracle, and
/// `tanh(+-100)` exactly `+-1`. Mutation: emit tanh as `(e^x - e^-x) / (e^x + e^-x)` without the `|x|`
/// shift in `poot-kernelgen`'s `emit_scalar_op`; `+-100` returns NaN and this row goes red.
#[test]
fn unary_tanh_rocm_matches_cpu() {
    unary_transcendental_rocm(UnOp::Tanh, "rocm_tanh", 10.0);
}

#[test]
fn unary_erf_rocm_matches_cpu() {
    unary_transcendental_rocm(UnOp::Erf, "rocm_erf", 10.0);
}

fn unary_transcendental_rocm(op: UnOp, label: &str, saturates_at: f32) {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = TRANSCENDENTAL_TABLE.len();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let out = b.unary(op, x);
    let xi = x.id;
    let g = b.finish(out);
    let inputs = make_inputs(&[(xi, vec![n], TRANSCENDENTAL_TABLE.to_vec())]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_transcendental_tier2(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        label,
        saturates_at,
    );
}

#[test]
fn unary_exp_rocm_matches_cpu() {
    unary_rocm_case(UnOp::Exp, vec![0.0f32, 0.5, 1.0, -1.0], "rocm_exp");
}

#[test]
fn unary_sqrt_rocm_matches_cpu() {
    unary_rocm_case(UnOp::Sqrt, vec![1.0f32, 4.0, 9.0, 0.25], "rocm_sqrt");
}

#[test]
fn unary_log_rocm_matches_cpu() {
    unary_rocm_case(
        UnOp::Log,
        vec![1.0f32, std::f32::consts::E, 7.389056, 0.5],
        "rocm_log",
    );
}

#[test]
fn broadcast_binary_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (batch, rows, cols) = (2usize, 3usize, 4usize);
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![batch, rows, cols]));
    let bias = b.constant("bias", TensorType::f32(vec![cols]));
    let out = b.binary(BinOp::Add, a, bias);
    let (ai, bi) = (a.id, bias.id);
    let g = b.finish(out);
    let inputs = make_inputs(&[
        (ai, vec![batch, rows, cols], fill(601, batch * rows * cols)),
        (bi, vec![cols], fill(602, cols)),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_broadcast_add",
        1e-6,
    );
}

#[test]
fn transpose_slice_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (r, c) = (8usize, 4usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![r, c]));
    let t = b.transpose(x, vec![1, 0]); // [c, r] = [4, 8]
    let s = b.slice(t, 1, 2, 6); // axis 1 has size r=8; 2..6 valid
    let xi = x.id;
    let g = b.finish(s);
    let inputs = make_inputs(&[(xi, vec![r, c], fill(701, r * c))]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_eq!(got.shape(), cpu.shape());
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_transpose_slice",
        1e-6,
    );
}

#[test]
fn elementwise_sub_div_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let n = 16usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let sub = b.binary(BinOp::Sub, a, c);
    let out = b.binary(BinOp::Div, sub, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);
    // c is strictly nonzero to avoid div-by-zero.
    let mut cd = fill(802, n);
    for v in cd.iter_mut() {
        *v = v.abs() + 0.5;
    }
    let inputs = make_inputs(&[(ai, vec![n], fill(801, n)), (ci, vec![n], cd)]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_sub_div",
        1e-5,
    );
}

#[test]
fn masked_attention_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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
    let inputs = make_inputs(&[
        (qi, vec![1, hq, 1, d], fill(501, hq * d)),
        (ki, vec![1, hkv, cap, d], fill(502, hkv * cap * d)),
        (vi, vec![1, hkv, cap, d], fill(503, hkv * cap * d)),
        (mi, vec![1, 1, 1, cap], vec![0.0, 0.0, -1.0e9, -1.0e9]),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    // Softmax + matmul accumulate; tolerance accounts for exp/sum order.
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_attention",
        5e-3,
    );
}

#[test]
fn decomposed_prefill_attention_over_65535_query_groups_rocm_matches_cpu() {
    // Card 1006: the traced prefill attention chain at Hq*L = 66560 > 65535 (the wgpu X grid cap; its
    // row-parallel softmax needs one workgroup per (head, row)). The result matches the CPU oracle.
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (hq, l, d) = (512usize, 130usize, 4usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = attention_prefill(&b, q, k, v, 1, 0.3, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);
    let causal: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let inputs = make_inputs(&[
        (qi, vec![1, hq, l, d], fill(601, hq * l * d)),
        (ki, vec![1, hq, l, d], fill(602, hq * l * d)),
        (vi, vec![1, hq, l, d], fill(603, hq * l * d)),
        (mi, vec![1, 1, l, l], causal),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_once_rocm(&mut rocm, target, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "rocm_decomposed_prefill_attention",
        2e-3,
    );
}

// ── Sampling primitives (card 551a) ──

/// Card 551a (SC-001, SC-002, SC-005, SC-007): `OpKind::SampleToken { rule: Greedy }` through the full
/// compile/planner/executor contract matches the CPU oracle bit for bit on ROCm. Row 0 carries two
/// ties, isolated from any non-finite logit so each tie-break is actually observable in the final
/// token rather than masked by the non-finite override: a per-lane tie (indices 5 and 5+64, same lane
/// 5 two groups over - already resolved by the per-lane scan's own strict `>` before the cross-lane
/// fold ever sees it) and a cross-lane tie (indices 1 and 64, different lanes - the case the
/// cross-lane fold's `ci < bi` tie-break exists for; wants token 1, the lowest of the four tied
/// indices). Row 1 carries the non-finite logit (NaN at the last index) alone. `run_once_rocm` assumes
/// an F32 output, so this test reads back I32 bytes itself.
#[test]
fn sample_token_greedy_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(li, HostTensor::f32(vec![rows, vocab], data).into())]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let mut builder = WeightStore::builder();
    let name = g.meta(li).name.clone().unwrap();
    let tensor = inputs[&li].as_host().unwrap();
    let bytes: Arc<[u8]> = Arc::from(
        tensor
            .as_f32()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let dense = DenseWeight::try_new(DType::F32, tensor.shape().to_vec(), bytes).unwrap();
    builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    let store = Arc::new(builder.build());
    let staged = {
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
    };
    let exe = rocm
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = rocm.add_entry(exe, &staged).unwrap();
    let bytes = rocm
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    rocm.unload(exe).unwrap();
    let got: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token Greedy through the full contract must match the CPU oracle on ROCm"
    );
}

/// Card 551a (SC-001, SC-010): `OpKind::RandomUniform` through the full compile/planner/executor
/// contract matches the CPU oracle bit for bit on ROCm (tier 1, ADR-0101). `seed` is I32, so it rides
/// as a `Slot::Sampler` (not a `Storage::Const`: `run_once_rocm` panics on any non-Const input and
/// hardcodes F32 weight storage, since every other fixture in this file is F32-const-only).
#[test]
fn random_uniform_rocm_matches_cpu() {
    use poot_executor::HostView;
    use poot_graph_ir::{Slot, SlotKey};

    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (rows, cols) = (3usize, 37usize);
    let b = Builder::new();
    let seed = b.slot_named(
        Slot::Sampler,
        "seed",
        TensorType::new(vec![rows], DType::I32),
    );
    let out = b.random_uniform(seed, cols);
    let g = b.finish(out);

    let seed_data: Vec<i32> = vec![0, 1, 0xDEADBEEFu32 as i32];
    let inputs: HashMap<ValueId, Value> = HashMap::from([(
        seed.id,
        HostTensor::i32(vec![rows], seed_data.clone()).into(),
    )]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let staged = {
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
    };
    let store = Arc::new(WeightStore::builder().build());
    let exe = rocm
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = rocm.add_entry(exe, &staged).unwrap();
    let seed_bytes: Vec<u8> = seed_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut step_in = StepInputs::new();
    let seed_shape = [rows];
    step_in.push(
        SlotKey::new(Slot::Sampler, Some("seed")),
        &seed_shape,
        HostView::new(DType::I32, rows, &seed_bytes).unwrap(),
    );
    let bytes = rocm
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    rocm.unload(exe).unwrap();
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_close(&got, cpu.as_f32().unwrap(), "rocm_random_uniform", 0.0);
}

/// Builds and runs a Gumbel-family `SampleToken` graph through the full contract on ROCm: `logits`/
/// `noise`/`params` as F32 consts (`run_once_rocm` panics on any non-Const input and hardcodes F32,
/// as `random_uniform_rocm_matches_cpu` documents), `top_k` (when present) as an I32 `Slot::Sampler`
/// bind, same pattern as that test's `seed`.
fn run_gumbel_family_contract_rocm(
    rocm: &mut Engine<RocmDevice>,
    target: Target,
    g: &Graph,
    logits: (ValueId, Vec<usize>, Vec<f32>),
    noise: (ValueId, Vec<usize>, Vec<f32>),
    params: (ValueId, Vec<usize>, Vec<f32>),
    top_k: Option<(ValueId, Vec<usize>, Vec<i32>)>,
) -> Vec<i32> {
    use poot_executor::HostView;
    use poot_graph_ir::{Slot, SlotKey};

    let mut builder = WeightStore::builder();
    for (id, shape, data) in [&logits, &noise, &params] {
        let name = g.meta(*id).name.clone().unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape.clone(), Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let store = Arc::new(builder.build());
    let staged = {
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
    };
    let exe = rocm
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = rocm.add_entry(exe, &staged).unwrap();
    let mut step_in = StepInputs::new();
    let top_k_bytes;
    if let Some((_, shape, data)) = &top_k {
        top_k_bytes = data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        step_in.push(
            SlotKey::new(Slot::Sampler, Some("top_k")),
            shape,
            HostView::new(DType::I32, shape.iter().product(), &top_k_bytes).unwrap(),
        );
    }
    let bytes = rocm
        .step(exe, entry, &step_in, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    rocm.unload(exe).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Card 551a (SC-001, SC-004, SC-007): `OpKind::SampleToken { rule: Gumbel }` through the full
/// compile/planner/executor contract matches the CPU oracle bit for bit on ROCm. Same fixture as
/// `sample_token_gumbel_cross_backend` (wgpu, `poot-gpu/tests/cross_backend.rs`): row 0 isolates the
/// `inv_temp` multiply (R472-007: "multiplies, not divides"), row 1 is an unrelated non-finite row
/// (SC-007), isolated so the override can't hide row 0's own result.
#[test]
fn sample_token_gumbel_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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

    let got = run_gumbel_family_contract_rocm(
        &mut rocm,
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
        "sample_token Gumbel through the full contract must match the CPU oracle on ROCm: row 0's \
         inv_temp multiply, row 1's non-finite override"
    );
}

/// Card 551a (SC-001): `OpKind::SampleToken { rule: GumbelTopK }` through the full compile/planner/
/// executor contract matches the CPU oracle bit for bit on ROCm. Same fixture as
/// `sample_token_gumbel_topk_cross_backend` (wgpu): `top_k = 1` with `min_p` disabled, two non-zero
/// indices (3, 6) tied exactly at the row max, the inclusive filter must keep exactly those two
/// survivors for the tie-break to pick the lower (3).
#[test]
fn sample_token_gumbel_topk_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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
    let params_data = vec![1.0, -3.4028235e38, 0.0];
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

    let got = run_gumbel_family_contract_rocm(
        &mut rocm,
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
        "sample_token GumbelTopK through the full contract must match the CPU oracle on ROCm: \
         top_k=1 keeps both tied survivors (3, 6), tie-break picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

/// Card 551a (SC-001): `OpKind::SampleToken { rule: GumbelTopKTopP }` through the full compile/
/// planner/executor contract matches the CPU oracle bit for bit on ROCm. Same fixture as
/// `sample_token_gumbel_topk_topp_cross_backend` (wgpu): `top_p = 0` must collapse to top-1 rather
/// than being treated as disabled; index 2's noise-boosted decoy (9.0 logit, 5.0 noise) only wins if
/// that collapse doesn't run.
#[test]
fn sample_token_gumbel_topk_topp_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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
    let params_data = vec![1.0, -3.4028235e38, 1.0, 0.0];
    let top_k_data = vec![0i32];
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

    let got = run_gumbel_family_contract_rocm(
        &mut rocm,
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
        "sample_token GumbelTopKTopP through the full contract must match the CPU oracle on ROCm: \
         top_p=0 collapses to top-1 (excluding index 2's noise-boosted decoy), tie-break picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

/// Card 677 SC-001, ROCm sibling of `sample_token_gumbel_topp_only_excludes_untruncated_winner_cross_backend`
/// (wgpu): top-p the ONLY active truncation (`top_k = 0`, min-p disabled) must still truncate on
/// ROCm. Same fixture: logits 3 and 6 tied at the row max (10.0), logit 2 just below (9.0) with a
/// noise boost (20.0) that makes it the Gumbel-max winner of the UNTRUNCATED row; `top_p = 0.7`
/// gives the tied pair a quantized mass of 32 against a threshold of 27, excluding the decoy. Found
/// by card 551b's review: `poot-eval`'s `sample_one_row` used to start this bisection's floor at
/// `-inf` whenever neither min-p nor top-k had already raised it, pinning it and silently disabling
/// top-p.
#[test]
fn sample_token_gumbel_topp_only_excludes_untruncated_winner_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
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
    let params_data = vec![1.0, f32::NEG_INFINITY, 1.0, 0.7];
    let top_k_data = vec![0i32];
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

    let got = run_gumbel_family_contract_rocm(
        &mut rocm,
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
        "sample_token GumbelTopKTopP through the full contract must match the CPU oracle on ROCm: \
         top_p=0.7 alone (no min-p/top-k) must exclude the noise-boosted decoy (index 2), tie-break \
         over {{3, 6}} picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}
