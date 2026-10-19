//! Cross-backend correctness tests for ROCm (card 035): the same graph runs on ROCm and the CPU
//! oracle and outputs are compared element-wise within f32 tolerance. Skips cleanly when ROCm/HSA is
//! unavailable.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::analysis::Tier2Class;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::{attention_masked, attention_prefill};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, Storage, ValueId};
use poot_graph_plan::Target;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_device::device::RocmDevice;
use poot_tensor::{DType, HostTensor};

fn try_rocm() -> Option<(Target, Engine<RocmDevice>)> {
    let device =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
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
    for &id in g.inputs() {
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
        let bytes = StoredBytes::from(tensor.view().bytes().to_vec());
        let dense = DenseWeight::try_new(tensor.dtype(), tensor.shape().to_vec(), bytes).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let store = Arc::new(builder.build());
    let mut weights = ConstWeights::new(&store);
    let staged = {
        let g = weights.graph(g);
        poot_graph_plan::staged::compile_staged(
            &g,
            &poot_graph_plan::staged::TargetSet::single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
                target,
            ),
            &poot_graph_plan::staged::Partition {
                experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
                devices: poot_graph_plan::staged::DevicePlacement::Single(
                    poot_graph_plan::multi_device::topology::DeviceId(0),
                ),
            },
            &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("compile_staged")
    };
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();
    let bytes = engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    engine.unload(exe).unwrap();
    let data: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    HostTensor::f32(g.aval(g.result()).shape.clone(), data)
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![n])).unwrap();
    let c = b.constant("c", TensorType::f32(vec![n])).unwrap();
    let out = b.binary(BinOp::Add, a, c).unwrap();
    let (ai, ci) = (a.id(), c.id());
    let g = b.finish(out).unwrap();

    let ad = fill(42, n);
    let cd = fill(99, n);
    let inputs = make_inputs(&[(ai, vec![n], ad.clone()), (ci, vec![n], cd.clone())]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![n])).unwrap();
    let c = b.constant("c", TensorType::f32(vec![n])).unwrap();
    let out = b.binary(BinOp::Mul, a, c).unwrap();
    let (ai, ci) = (a.id(), c.id());
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(ai, vec![n], fill(201, n)), (ci, vec![n], fill(202, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let out = b.unary(UnOp::Neg, x).unwrap();
    let xi = x.id();
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(xi, vec![n], fill(301, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(a, w).unwrap();
    let (ai, wi) = (a.id(), w.id());
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[
        (ai, vec![m, k], fill(401, m * k)),
        (wi, vec![k, n], fill(402, k * n)),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, cols])).unwrap();
    let out = b.reduce(RedOp::Sum, x, 1, false).unwrap();
    let xi = x.id();
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(xi, vec![rows, cols], fill(501, rows * cols))]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, cols])).unwrap();
    let out = b.reduce(RedOp::Max, x, 1, false).unwrap();
    let xi = x.id();
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(xi, vec![rows, cols], fill(511, rows * cols))]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, cols])).unwrap();
        let out = b.reduce(RedOp::Sum, x, 1, false).unwrap();
        let xi = x.id();
        let g = b.finish(out).unwrap();

        let data = fill_real_magnitude(820 + i as u64, rows * cols);
        let inputs = make_inputs(&[(xi, vec![rows, cols], data.clone())]);

        let cpu = eval(&g, &inputs, EvalOptions::default())
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
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, cols])).unwrap();
        let out = b.reduce(RedOp::Max, x, 1, false).unwrap();
        let xi = x.id();
        let g = b.finish(out).unwrap();

        let data = fill_real_magnitude(920 + i as u64, rows * cols);
        let inputs = make_inputs(&[(xi, vec![rows, cols], data.clone())]);

        let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![n])).unwrap();
    let out = poot_graph_ir::ops::rmsnorm(&b, x, w, 1e-6).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();

    let xd = fill_real_magnitude(931, n);
    // Weights stay near typical learned-scale magnitude (not +-30).
    let wd: Vec<f32> = fill(932, n).iter().map(|v| 1.0 + v.abs()).collect();
    let inputs = make_inputs(&[(xi, vec![1, 1, n], xd), (wi, vec![n], wd)]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let out = poot_graph_ir::ops::layernorm(&b, x, w, bias, 1e-6).unwrap();
    let (xi, wi, bi) = (x.id(), w.id(), bias.id());
    let g = b.finish(out).unwrap();

    let xd = fill_real_magnitude(941, n);
    let wd: Vec<f32> = fill(942, n).iter().map(|v| 1.0 + v.abs()).collect();
    let bd: Vec<f32> = fill(943, n).iter().map(|v| v * 0.1).collect();
    let inputs = make_inputs(&[
        (xi, vec![1, 1, n], xd),
        (wi, vec![n], wd),
        (bi, vec![n], bd),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![n])).unwrap();
    let c = b.constant("c", TensorType::f32(vec![n])).unwrap();
    let out = b.binary(BinOp::Add, a, c).unwrap();
    let (ai, ci) = (a.id(), c.id());
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(ai, vec![n], fill(601, n)), (ci, vec![n], fill(602, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1])).unwrap();
    let c = b.constant("c", TensorType::f32(vec![1])).unwrap();
    let out = b.binary(BinOp::Add, a, c).unwrap();
    let (ai, ci) = (a.id(), c.id());
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[(ai, vec![1], vec![3.5]), (ci, vec![1], vec![2.5])]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let operand = b
        .constant("operand", TensorType::f32(vec![rows, cols]))
        .unwrap();
    let update = b
        .constant("update", TensorType::f32(vec![1, cols]))
        .unwrap();
    let out = b.dynamic_update_slice(operand, update, index, 0).unwrap();
    let (oi, ui) = (operand.id(), update.id());
    let g = b.finish(out).unwrap();

    let inputs = make_inputs(&[
        (oi, vec![rows, cols], fill(701, rows * cols)),
        (ui, vec![1, cols], fill(702, cols)),
    ]);

    let cpu = eval(&g, &inputs, EvalOptions::default())
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
// crates/poot-wgpu-device/tests/cross_backend.rs. Each test creates one executor (one try_rocm): a second
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
    let b = Builder::for_tests();
    let data = b.constant("data", TensorType::f32(vec![n, d])).unwrap();
    let idx = b
        .constant(
            "idx",
            TensorType::new(vec![idx_count], poot_tensor::DType::I32),
        )
        .unwrap();
    let out = b.gather(data, 0, idx).unwrap();
    let (di, ii) = (data.id(), idx.id());
    let g = b.finish(out).unwrap();
    let mut inputs = make_inputs(&[(di, vec![n, d], fill(901, n * d))]);
    inputs.insert(ii, HostTensor::i32(vec![idx_count], vec![3, 0, 6]).into());
    let cpu = eval(&g, &inputs, EvalOptions::default())
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

/// A `Gather` over an I32 const table whose words need all 32 bits ([`poot_eval::test_support::i32_table_gather`])
/// returns the table's words exactly on ROCm: the table is stored as raw I32.
///
/// Mutation: type the I32 table through `Ty::F32` in `movement_storage`; the output is the table rounded
/// through F32 and this row goes red.
#[test]
fn i32_table_gather_is_exact_beyond_f32_rocm() {
    use poot_eval::test_support::i32_table_gather as fixture;

    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let g = fixture::graph();
    let mut builder = WeightStore::builder();
    let words: Vec<u8> = fixture::table()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    builder
        .insert(
            "table",
            WeightEntry::Dense(
                DenseWeight::try_new(DType::I32, vec![6, 4], StoredBytes::from(words)).unwrap(),
            ),
        )
        .unwrap();
    let store = Arc::new(builder.build());
    let mut weights = ConstWeights::new(&store);
    let staged = {
        let g = weights.graph(&g);
        poot_graph_plan::staged::compile_staged(
            &g,
            &poot_graph_plan::staged::TargetSet::single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
                target,
            ),
            &poot_graph_plan::staged::Partition {
                experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
                devices: poot_graph_plan::staged::DevicePlacement::Single(
                    poot_graph_plan::multi_device::topology::DeviceId(0),
                ),
            },
            &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("compile_staged")
    };
    let exe = rocm
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = rocm.add_entry(exe, &staged).unwrap();
    let tokens = fixture::tokens();
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|w| w.to_le_bytes()).collect();
    let key = g
        .inputs()
        .iter()
        .find_map(|&id| g.meta(id).slot_key().cloned())
        .expect("the token slot");
    let shape = [tokens.len()];
    let mut slots = StepInputs::new();
    slots.push(
        key,
        &shape,
        poot_tensor::HostView::new(DType::I32, tokens.len(), &token_bytes).unwrap(),
    );
    let bytes = rocm
        .step(exe, entry, &slots, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    rocm.unload(exe).unwrap();
    let got: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    fixture::check("rocm", &got).unwrap();
}

#[test]
fn matmul_bias_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (m, k, n) = (4usize, 8usize, 4usize);
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![m, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let mm = b.matmul(a, w).unwrap();
    let out = b.binary(BinOp::Add, mm, bias).unwrap();
    let (ai, wi, bi) = (a.id(), w.id(), bias.id());
    let g = b.finish(out).unwrap();
    let inputs = make_inputs(&[
        (ai, vec![m, k], fill(401, m * k)),
        (wi, vec![k, n], fill(402, k * n)),
        (bi, vec![n], fill(403, n)),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let out = b.unary(op, x).unwrap();
    let xi = x.id();
    let g = b.finish(out).unwrap();
    let inputs = make_inputs(&[(xi, vec![n], xd)]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let out = b.unary(op, x).unwrap();
    let xi = x.id();
    let g = b.finish(out).unwrap();
    let inputs = make_inputs(&[(xi, vec![n], TRANSCENDENTAL_TABLE.to_vec())]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::f32(vec![batch, rows, cols]))
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![cols])).unwrap();
    let out = b.binary(BinOp::Add, a, bias).unwrap();
    let (ai, bi) = (a.id(), bias.id());
    let g = b.finish(out).unwrap();
    let inputs = make_inputs(&[
        (ai, vec![batch, rows, cols], fill(601, batch * rows * cols)),
        (bi, vec![cols], fill(602, cols)),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![r, c])).unwrap();
    let t = b.transpose(x, vec![1, 0]).unwrap(); // [c, r] = [4, 8]
    let s = b.slice(t, 1, 2, 6).unwrap(); // axis 1 has size r=8; 2..6 valid
    let xi = x.id();
    let g = b.finish(s).unwrap();
    let inputs = make_inputs(&[(xi, vec![r, c], fill(701, r * c))]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![n])).unwrap();
    let c = b.constant("c", TensorType::f32(vec![n])).unwrap();
    let sub = b.binary(BinOp::Sub, a, c).unwrap();
    let out = b.binary(BinOp::Div, sub, c).unwrap();
    let (ai, ci) = (a.id(), c.id());
    let g = b.finish(out).unwrap();
    // c is strictly nonzero to avoid div-by-zero.
    let mut cd = fill(802, n);
    for v in cd.iter_mut() {
        *v = v.abs() + 0.5;
    }
    let inputs = make_inputs(&[(ai, vec![n], fill(801, n)), (ci, vec![n], cd)]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
        .unwrap();
    let out = attention_masked(&b, q, k, v, n_rep, scale, mask).unwrap();
    let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
    let g = b.finish(out).unwrap();
    let inputs = make_inputs(&[
        (qi, vec![1, hq, 1, d], fill(501, hq * d)),
        (ki, vec![1, hkv, cap, d], fill(502, hkv * cap * d)),
        (vi, vec![1, hkv, cap, d], fill(503, hkv * cap * d)),
        (mi, vec![1, 1, 1, cap], vec![0.0, 0.0, -1.0e9, -1.0e9]),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, l, l]))
        .unwrap();
    let out = attention_prefill(&b, q, k, v, 1, 0.3, mask).unwrap();
    let (qi, ki, vi, mi) = (q.id(), k.id(), v.id(), mask.id());
    let g = b.finish(out).unwrap();
    let causal: Vec<f32> = (0..l * l)
        .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
        .collect();
    let inputs = make_inputs(&[
        (qi, vec![1, hq, l, d], fill(601, hq * l * d)),
        (ki, vec![1, hq, l, d], fill(602, hq * l * d)),
        (vi, vec![1, hq, l, d], fill(603, hq * l * d)),
        (mi, vec![1, 1, l, l], causal),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let logits = b
        .constant("logits", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::Greedy,
            logits,
            None,
            None,
            None,
        )
        .unwrap();
    let li = logits.id();
    let g = b.finish(out).unwrap();

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
    let cpu = eval(&g, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let mut builder = WeightStore::builder();
    let name = g.meta(li).name.clone().unwrap();
    let tensor = inputs[&li].as_host().unwrap();
    let bytes = StoredBytes::from(
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
    let mut weights = ConstWeights::new(&store);
    let staged = {
        let g = weights.graph(&g);
        poot_graph_plan::staged::compile_staged(
            &g,
            &poot_graph_plan::staged::TargetSet::single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
                target,
            ),
            &poot_graph_plan::staged::Partition {
                experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
                devices: poot_graph_plan::staged::DevicePlacement::Single(
                    poot_graph_plan::multi_device::topology::DeviceId(0),
                ),
            },
            &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("compile_staged")
    };
    let exe = rocm
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = rocm.add_entry(exe, &staged).unwrap();
    let bytes = rocm
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    rocm.unload(exe).unwrap();
    let got: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
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
    use poot_graph_ir::{Slot, SlotKey};
    use poot_tensor::HostView;

    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (rows, cols) = (3usize, 37usize);
    let b = Builder::for_tests();
    let seed = b
        .slot_named(
            Slot::Sampler,
            "seed",
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap();
    let out = b.random_uniform(seed, cols).unwrap();
    let g = b.finish(out).unwrap();

    let seed_data: Vec<i32> = vec![0, 1, 0xDEADBEEFu32 as i32];
    let inputs: HashMap<ValueId, Value> = HashMap::from([(
        seed.id(),
        HostTensor::i32(vec![rows], seed_data.clone()).into(),
    )]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let staged = {
        let g = g.clone();
        poot_graph_plan::staged::compile_staged(
            &g,
            &poot_graph_plan::staged::TargetSet::single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
                target,
            ),
            &poot_graph_plan::staged::Partition {
                experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
                devices: poot_graph_plan::staged::DevicePlacement::Single(
                    poot_graph_plan::multi_device::topology::DeviceId(0),
                ),
            },
            &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("compile_staged")
    };
    let store = Arc::new(WeightStore::builder().build());
    let exe = rocm.load_weights(store, Arc::default()).unwrap();
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
        .as_chunks::<4>()
        .0
        .iter()
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
    use poot_graph_ir::{Slot, SlotKey};
    use poot_tensor::HostView;

    let mut builder = WeightStore::builder();
    for (id, shape, data) in [&logits, &noise, &params] {
        let name = g.meta(*id).name.clone().unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense =
            DenseWeight::try_new(DType::F32, shape.clone(), StoredBytes::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    let store = Arc::new(builder.build());
    let mut weights = ConstWeights::new(&store);
    let staged = {
        let g = weights.graph(g);
        poot_graph_plan::staged::compile_staged(
            &g,
            &poot_graph_plan::staged::TargetSet::single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
                target,
            ),
            &poot_graph_plan::staged::Partition {
                experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
                devices: poot_graph_plan::staged::DevicePlacement::Single(
                    poot_graph_plan::multi_device::topology::DeviceId(0),
                ),
            },
            &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("compile_staged")
    };
    let exe = rocm
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Card 551a (SC-001, SC-004, SC-007): `OpKind::SampleToken { rule: Gumbel }` through the full
/// compile/planner/executor contract matches the CPU oracle bit for bit on ROCm. Same fixture as
/// `sample_token_gumbel_cross_backend` (wgpu, `poot-wgpu-device/tests/cross_backend.rs`): row 0 isolates the
/// `inv_temp` multiply (R472-007: "multiplies, not divides"), row 1 is an unrelated non-finite row
/// (SC-007), isolated so the override can't hide row 0's own result.
#[test]
fn sample_token_gumbel_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let (rows, vocab) = (2usize, 8usize);
    let b = Builder::for_tests();
    let logits = b
        .constant("logits", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let noise = b
        .constant("noise", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let params = b
        .constant("params", TensorType::f32(vec![rows, 3]))
        .unwrap();
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::Gumbel,
            logits,
            Some(noise),
            Some(params),
            None,
        )
        .unwrap();
    let (li, ni, pi) = (logits.id(), noise.id(), params.id());
    let g = b.finish(out).unwrap();

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
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
    let b = Builder::for_tests();
    let logits = b
        .constant("logits", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let noise = b
        .constant("noise", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let params = b
        .constant("params", TensorType::f32(vec![rows, 3]))
        .unwrap();
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap()
    };
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::GumbelTopK,
            logits,
            Some(noise),
            Some(params),
            Some(top_k),
        )
        .unwrap();
    let (li, ni, pi) = (logits.id(), noise.id(), params.id());
    let g = b.finish(out).unwrap();

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
            top_k.id(),
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
        Some((top_k.id(), vec![rows], top_k_data)),
    );
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token GumbelTopK through the full contract must match the CPU oracle on ROCm: \
         top_k=1 keeps both tied survivors (3, 6), tie-break picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

/// POOT-1195 (R2-C1-001) SC-003: the batched top-p kernel picks the same token as the CPU oracle for
/// 1,000 seeded rows on each of three distributions whose real nucleus the old `SCALE = 16` mass
/// collapsed (Zipf s=1.1 and s=0.8 over 32,000 tokens at `top_p = 0.95`; a 0.5 head over a 500-token
/// flat tail at `top_p = 0.9`), on ROCm. Every row shares the logits and draws its own Gumbel noise,
/// so a collapsed device nucleus is observable twice: as a mismatch against the oracle and as a draw that
/// never leaves the first few tokens.
///
/// Mutation: put `SCALE = 16` (`round(exp(..) * 16)`) back in `sample_topp_gumbel_argmax_batched.rs`
/// and regenerate the assets; this row goes red while the host rows stay green.
#[test]
fn sample_token_topp_nucleus_rocm_matches_cpu() {
    let (target, mut rocm) = match try_rocm() {
        Some(p) => p,
        None => return,
    };
    let zipf = |s: f64| -> Vec<f32> {
        (0..32_000)
            .map(|i| (-s * ((i + 1) as f64).ln()) as f32)
            .collect()
    };
    let mut head_tail = vec![0.5f64.ln() as f32];
    head_tail.extend((0..500).map(|j| (0.001f64 * (1.0 - 1.0e-4 * j as f64)).ln() as f32));
    let distributions: [(&str, Vec<f32>, f32); 3] = [
        ("zipf s=1.1", zipf(1.1), 0.95),
        ("zipf s=0.8", zipf(0.8), 0.95),
        ("head + flat tail", head_tail, 0.9),
    ];
    let rows = 1000usize;
    for (label, row_logits, top_p) in distributions {
        let vocab = row_logits.len();
        let b = Builder::for_tests();
        let logits = b
            .constant("logits", TensorType::f32(vec![rows, vocab]))
            .unwrap();
        let noise = b
            .constant("noise", TensorType::f32(vec![rows, vocab]))
            .unwrap();
        let params = b
            .constant("params", TensorType::f32(vec![rows, 4]))
            .unwrap();
        let top_k = {
            use poot_graph_ir::Slot;
            b.slot_named(
                Slot::Sampler,
                "top_k",
                TensorType::new(vec![rows], DType::I32),
            )
            .unwrap()
        };
        let out = b
            .sample_token(
                poot_graph_ir::op::SampleRule::GumbelTopKTopP,
                logits,
                Some(noise),
                Some(params),
                Some(top_k),
            )
            .unwrap();
        let (li, ni, pi) = (logits.id(), noise.id(), params.id());
        let g = b.finish(out).unwrap();

        let logits_data: Vec<f32> = (0..rows).flat_map(|_| row_logits.iter().copied()).collect();
        // Gumbel noise `-ln(-ln(u))`, one xorshift64 stream per row (row index is the seed).
        let noise_data: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ ((r as u64 + 1) << 20);
                (0..vocab).map(move |_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let u = (((s >> 40) as f32) + 0.5) / (1u64 << 24) as f32;
                    -(-u.ln()).ln()
                })
            })
            .collect();
        let params_data: Vec<f32> = (0..rows)
            .flat_map(|_| [1.0, f32::NEG_INFINITY, 1.0, top_p])
            .collect();
        let top_k_data = vec![0i32; rows];
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
                top_k.id(),
                HostTensor::i32(vec![rows], top_k_data.clone()).into(),
            ),
        ]);
        let cpu = eval(&g, &inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let cpu = cpu.as_i32().expect("SampleToken output is I32").to_vec();

        let got: Vec<i32> = run_gumbel_family_contract_rocm(
            &mut rocm,
            target,
            &g,
            (li, vec![rows, vocab], logits_data),
            (ni, vec![rows, vocab], noise_data),
            (pi, vec![rows, 4], params_data),
            Some((top_k.id(), vec![rows], top_k_data)),
        );
        let first_diff = got.iter().zip(&cpu).position(|(g, c)| g != c);
        assert_eq!(
            first_diff,
            None,
            "{label}: device top-p vs the CPU oracle over {rows} seeded rows: first differing word {first_diff:?} \
             (device {:?}, oracle {:?})",
            first_diff.map(|i| got[i]),
            first_diff.map(|i| cpu[i]),
        );
        let farthest = got.chunks(2).map(|row| row[0]).max().unwrap();
        assert!(
            farthest > 100,
            "{label}: a real nucleus reaches far past the first 20 tokens, the farthest draw was {farthest}"
        );
    }
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
    let b = Builder::for_tests();
    let logits = b
        .constant("logits", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let noise = b
        .constant("noise", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let params = b
        .constant("params", TensorType::f32(vec![rows, 4]))
        .unwrap();
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap()
    };
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::GumbelTopKTopP,
            logits,
            Some(noise),
            Some(params),
            Some(top_k),
        )
        .unwrap();
    let (li, ni, pi) = (logits.id(), noise.id(), params.id());
    let g = b.finish(out).unwrap();

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
            top_k.id(),
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
        Some((top_k.id(), vec![rows], top_k_data)),
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
/// gives the tied pair about 84% of the mass (2 / (2 + e^-1)), above the 0.7 threshold, excluding the decoy. Found
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
    let b = Builder::for_tests();
    let logits = b
        .constant("logits", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let noise = b
        .constant("noise", TensorType::f32(vec![rows, vocab]))
        .unwrap();
    let params = b
        .constant("params", TensorType::f32(vec![rows, 4]))
        .unwrap();
    let top_k = {
        use poot_graph_ir::Slot;
        b.slot_named(
            Slot::Sampler,
            "top_k",
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap()
    };
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::GumbelTopKTopP,
            logits,
            Some(noise),
            Some(params),
            Some(top_k),
        )
        .unwrap();
    let (li, ni, pi) = (logits.id(), noise.id(), params.id());
    let g = b.finish(out).unwrap();

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
            top_k.id(),
            HostTensor::i32(vec![rows], top_k_data.clone()).into(),
        ),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::default())
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
        Some((top_k.id(), vec![rows], top_k_data)),
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

// ---- contraction epilogue (SC-001, SC-006) ------------------------------------------------------------
//
// A bias, a pointwise activation and a residual add fuse into the contraction that feeds them, and the fused
// device program meets the CPU oracle of the unfused graph. A narrow head (BF16, F16) rounds to its dtype in
// registers before the epilogue reads it: the fused kernel keeps the narrow and widen a separate store and reload
// would perform.

/// `g` run once on `engine`; every input is a named const of its own dtype and the output is returned as the
/// graph declares it, with the numerics class the compiled program publishes.
fn run_under(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
) -> (HostTensor, Tier2Class) {
    let mut builder = WeightStore::builder();
    for (&id, tensor) in inputs {
        let meta = g.meta(id);
        assert_eq!(meta.storage, Storage::Const, "{id} is a const");
        let dense = DenseWeight::try_new(
            tensor.dtype(),
            tensor.shape().to_vec(),
            StoredBytes::from(tensor.view().bytes().to_vec()),
        )
        .unwrap();
        builder
            .insert(meta.name.clone().unwrap(), WeightEntry::Dense(dense))
            .unwrap();
    }
    let store = Arc::new(builder.build());
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(g);
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let staged = poot_graph_plan::staged::compile_staged(
        &named,
        &poot_graph_plan::staged::TargetSet::single(
            poot_graph_plan::multi_device::topology::DeviceId(0),
            target,
        ),
        &poot_graph_plan::staged::Partition {
            experts: poot_graph_plan::staged::ExpertPlacement::AllResident,
            devices: poot_graph_plan::staged::DevicePlacement::Single(
                poot_graph_plan::multi_device::topology::DeviceId(0),
            ),
        },
        &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile_staged");
    let entry = engine.add_entry(exe, &staged).unwrap();
    let bytes = engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    engine.remove_entry(exe, entry).unwrap();
    engine.unload(exe).unwrap();
    let out = g.aval(g.result());
    let words = || -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let host = match out.dtype {
        DType::F32 => HostTensor::f32(
            out.shape.clone(),
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::BF16 => HostTensor::bf16(out.shape.clone(), words()),
        DType::F16 => HostTensor::f16(out.shape.clone(), words()),
        other => panic!("unexpected output dtype {other:?}"),
    };
    (host, staged.numerics_class())
}

/// The compiled program holds exactly one `ContractionEpilogue`, over `head`.
fn assert_fuses_into_one_epilogue(
    g: &Graph,
    target: &Target,
    head: poot_graph_ir::op::ContractionHead,
) {
    let fused = poot_graph_plan::compile(
        g,
        target,
        &poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("the graph compiles");
    let heads: Vec<_> = fused
        .graph()
        .eqns()
        .iter()
        .filter_map(|eqn| match &eqn.op {
            poot_graph_ir::OpKind::ContractionEpilogue { head, .. } => Some(*head),
            _ => None,
        })
        .collect();
    assert_eq!(
        heads,
        [head],
        "one fused contraction: {:?}",
        fused.graph().eqns()
    );
}

/// The fused device result meets the CPU oracle's tier-2 tolerance (`label` names the row in a failure).
fn assert_fused_matches_oracle(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    g: &Graph,
    inputs: &HashMap<ValueId, HostTensor>,
    label: &str,
) -> HostTensor {
    let (fused, class) = run_under(engine, target, g, inputs);
    let oracle_inputs: HashMap<ValueId, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let want = eval(g, &oracle_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(fused.shape(), want.shape(), "{label}: shape");
    poot_test_util::assert_close_scaled(
        &fused.to_f32().unwrap(),
        &want.to_f32().unwrap(),
        class.tolerance(),
        label,
    );
    fused
}

/// `tanh(head + bias) * 0.5 + residual`: a bias, a representative pointwise activation, a literal scale and a
/// residual add, each value read once, so the whole chain is one epilogue.
fn epilogue_chain(
    b: &Builder,
    head: poot_graph_ir::Traced,
    bias: poot_graph_ir::Traced,
    residual: poot_graph_ir::Traced,
) -> poot_graph_ir::Traced {
    let biased = b.binary(BinOp::Add, head, bias).unwrap();
    let activated = b.unary(UnOp::Tanh, biased).unwrap();
    let scaled = b
        .binary_scalar(
            BinOp::Mul,
            activated,
            poot_graph_ir::types::Scalar::F32(0.5),
        )
        .unwrap();
    b.binary(BinOp::Add, scaled, residual).unwrap()
}

/// SC-001 (ROCm, the decode Gemv): an F32 `[N, K]` projection at `M == 1` with a bias, an activation, a scale
/// and a residual fuses into one `ContractionEpilogue { head: Dense }`; the device result meets the
/// oracle. Mutation: omit the bias or the residual leaf in `emit_epilogue` (`read_leaf` yields zero); the
/// nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_gemv_epilogue_matches_the_oracle_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![n, k])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let residual = b
        .constant("residual", TensorType::f32(vec![1, 1, n]))
        .unwrap();
    let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (x.id(), HostTensor::f32(vec![1, 1, k], fill(1, k))),
        (w.id(), HostTensor::f32(vec![n, k], fill(2, n * k))),
        (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
        (residual.id(), HostTensor::f32(vec![1, 1, n], fill(4, n))),
    ]);
    let g = b.finish(out).unwrap();
    assert_fuses_into_one_epilogue(
        &g,
        &target,
        poot_graph_ir::op::ContractionHead::Dense { weight: DType::F32 },
    );
    assert_fused_matches_oracle(&mut rocm, target, &g, &inputs, "dense gemv epilogue");
}

/// Round an f32 to its BF16 bits, nearest even.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// POOT-1032 (ROCm, the generated tiled GEMM): the dense head at `M > 1` over an F32 and a packed-BF16 `[N, K]`
/// weight, with the same chain, fuses into one `ContractionEpilogue { head: Dense }` at a single partial row-tile
/// (`M == 5`) and at many (`M == 160`); the device result meets the oracle. Mutation: drop the
/// epilogue from the tiled store in `tiled_region`; the nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_tiled_epilogue_matches_the_oracle_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for weight_dtype in [DType::F32, DType::BF16] {
        for m in [5usize, 160] {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
            let w = b
                .constant("w", TensorType::new(vec![n, k], weight_dtype))
                .unwrap();
            let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
            let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
            let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
            let out = epilogue_chain(&b, head, bias, residual);
            let weight_values = fill(2, n * k);
            let weight = match weight_dtype {
                DType::BF16 => HostTensor::bf16(
                    vec![n, k],
                    weight_values.into_iter().map(bf16_bits).collect(),
                ),
                _ => HostTensor::f32(vec![n, k], weight_values),
            };
            let inputs = HashMap::from([
                (x.id(), HostTensor::f32(vec![m, k], fill(1, m * k))),
                (w.id(), weight),
                (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
                (residual.id(), HostTensor::f32(vec![m, n], fill(4, m * n))),
            ]);
            let g = b.finish(out).unwrap();
            assert_fuses_into_one_epilogue(
                &g,
                &target,
                poot_graph_ir::op::ContractionHead::Dense {
                    weight: weight_dtype,
                },
            );
            assert_fused_matches_oracle(
                &mut rocm,
                target,
                &g,
                &inputs,
                &format!("dense tiled epilogue {weight_dtype:?} M={m}"),
            );
        }
    }
}

/// SC-001 (ROCm, the one-thread-per-output body): a batched F32 `MatMul` the planner lowers serially, with the
/// same chain, fuses into `ContractionEpilogue { head: MatMul }` and meets the oracle.
#[test]
fn a_serial_matmul_epilogue_matches_the_oracle_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 2, 3, 4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![1, 1, 4, 5])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![5])).unwrap();
    let residual = b
        .constant("residual", TensorType::f32(vec![1, 2, 3, 5]))
        .unwrap();
    let head = b.matmul(a, w).unwrap();
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (a.id(), HostTensor::f32(vec![1, 2, 3, 4], fill(11, 24))),
        (w.id(), HostTensor::f32(vec![1, 1, 4, 5], fill(12, 20))),
        (bias.id(), HostTensor::f32(vec![5], fill(13, 5))),
        (
            residual.id(),
            HostTensor::f32(vec![1, 2, 3, 5], fill(14, 30)),
        ),
    ]);
    let g = b.finish(out).unwrap();
    assert_fuses_into_one_epilogue(&g, &target, poot_graph_ir::op::ContractionHead::MatMul);
    assert_fused_matches_oracle(&mut rocm, target, &g, &inputs, "serial matmul epilogue");
}

/// POOT-1036 (ROCm, the generated tiled GEMM over an f32 `[K, N]` weight): a plain `MatMul` head at `M > 1`
/// with the same chain fuses into one `ContractionEpilogue { head: MatMul }` at a single partial row-tile
/// (`M == 5`) and at many (`M == 160`); the device result meets the oracle. Mutation: drop the
/// `Tileable` arm of `epilogue_head_lowers`; the head stays apart from its chain and this row goes red.
#[test]
fn a_tiled_matmul_epilogue_matches_the_oracle_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for m in [5usize, 160] {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
        let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
        let head = b.matmul(x, w).unwrap();
        let out = epilogue_chain(&b, head, bias, residual);
        let inputs = HashMap::from([
            (x.id(), HostTensor::f32(vec![m, k], fill(1, m * k))),
            (w.id(), HostTensor::f32(vec![k, n], fill(2, k * n))),
            (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
            (residual.id(), HostTensor::f32(vec![m, n], fill(4, m * n))),
        ]);
        let g = b.finish(out).unwrap();
        assert_fuses_into_one_epilogue(&g, &target, poot_graph_ir::op::ContractionHead::MatMul);
        assert_fused_matches_oracle(
            &mut rocm,
            target,
            &g,
            &inputs,
            &format!("tiled matmul epilogue M={m}"),
        );
    }
}

/// SC-006 (ROCm, BF16 and F16 heads): a narrow head rounds to its dtype before its residual reads it. `A = [1, 1]`
/// against `B = [1, tie]^T`, `tie` half a unit in the last place of 1.0 (`2^-8` BF16, `2^-11` F16), accumulates
/// to exactly halfway between 1.0 and its successor; round-to-nearest-even rounds the head to 1.0 and the residual
/// `-1` leaves zero. The fused device program and the CPU decomposition agree and are zero.
///
/// Mutation: remove the head's narrow and widen from `emit_epilogue` (`round_through` returns its input
/// unrounded); the fused kernel then reads the f32 accumulator, yields `tie`, and this row goes red.
#[test]
fn a_narrow_head_rounds_before_its_residual_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    // (dtype, words of 1.0, words of the tie, words of -1.0)
    let cases = [
        (DType::BF16, 0x3f80u16, 0x3b80u16, 0xbf80u16),
        (DType::F16, 0x3c00, 0x1000, 0xbc00),
    ];
    for (dtype, one, tie, minus_one) in cases {
        let tensor = |shape: &[usize], words: Vec<u16>| match dtype {
            DType::BF16 => HostTensor::bf16(shape.to_vec(), words),
            _ => HostTensor::f16(shape.to_vec(), words),
        };
        let b = Builder::for_tests();
        let a = b.constant("a", TensorType::new(vec![1, 2], dtype)).unwrap();
        let w = b.constant("w", TensorType::new(vec![2, 1], dtype)).unwrap();
        let residual = b
            .constant("residual", TensorType::new(vec![1, 1], dtype))
            .unwrap();
        let out = b
            .binary(BinOp::Add, b.matmul(a, w).unwrap(), residual)
            .unwrap();
        let inputs = HashMap::from([
            (a.id(), tensor(&[1, 2], vec![one, one])),
            (w.id(), tensor(&[2, 1], vec![one, tie])),
            (residual.id(), tensor(&[1, 1], vec![minus_one])),
        ]);
        let g = b.finish(out).unwrap();
        assert_fuses_into_one_epilogue(&g, &target, poot_graph_ir::op::ContractionHead::MatMul);
        let got = assert_fused_matches_oracle(
            &mut rocm,
            target,
            &g,
            &inputs,
            &format!("{dtype:?} narrow head"),
        );
        assert_eq!(got.dtype(), dtype);
        assert_eq!(
            got.to_f32().unwrap().as_ref(),
            [0.0],
            "{dtype:?}: the head rounds to 1.0 before the residual, not to 1 + tie"
        );
    }
}

/// SC-006 (ROCm, multistage narrow pointwise): every step of an epilogue over a BF16 head rounds through BF16.
/// `1.0 + tie` is a tie that rounds to 1.0 at each of two adds, so the chain ends at 1.0; an epilogue that kept
/// its intermediate in f32 would end at `1 + 2 tie`, which BF16 holds exactly. Mutation: skip the per-step
/// rounding in `emit_epilogue` (keep the head's round, drop `round_through` after each step); this row goes red
/// with `1.0078125`.
#[test]
fn every_step_of_a_narrow_epilogue_rounds_through_its_dtype_on_rocm() {
    let Some((target, mut rocm)) = try_rocm() else {
        return;
    };
    let (one, tie) = (0x3f80u16, 0x3b80u16);
    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![1, 1], DType::BF16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![1, 1], DType::BF16))
        .unwrap();
    let leaf = b
        .constant("tie", TensorType::new(vec![1, 1], DType::BF16))
        .unwrap();
    let head = b.matmul(a, w).unwrap();
    let out = b
        .binary(BinOp::Add, b.binary(BinOp::Add, head, leaf).unwrap(), leaf)
        .unwrap();
    let inputs = HashMap::from([
        (a.id(), HostTensor::bf16(vec![1, 1], vec![one])),
        (w.id(), HostTensor::bf16(vec![1, 1], vec![one])),
        (leaf.id(), HostTensor::bf16(vec![1, 1], vec![tie])),
    ]);
    let g = b.finish(out).unwrap();
    assert_fuses_into_one_epilogue(&g, &target, poot_graph_ir::op::ContractionHead::MatMul);
    let got = assert_fused_matches_oracle(&mut rocm, target, &g, &inputs, "bf16 multistage");
    assert_eq!(got.to_f32().unwrap().as_ref(), [1.0]);
}
