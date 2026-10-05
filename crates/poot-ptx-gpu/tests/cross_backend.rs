//! Cross-backend correctness tests for PTX (card 035).
//!
//! Verify that the PTX/NVIDIA GPU backend produces identical results to the CPU oracle
//! for all key operations. Tests run the same graph on both backends and compare
//! outputs element-wise within f32 tolerance.
//!
//! Skips cleanly when CUDA/NVIDIA hardware is unavailable.
//! PTX verification on real hardware requires RunPod.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::{attention_masked, attention_prefill};
use poot_graph_ir::types::TensorType;
use poot_ptx_gpu::PtxDevice;
use poot_tensor::HostTensor;

mod common;

fn try_ptx() -> Option<PtxDevice> {
    poot_test_util::device_skip::open_or_skip(
        poot_runtime_common::DeviceBackend::Ptx,
        PtxDevice::new(),
    )
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
            "{label} elem {i}: ptx {a} vs cpu {b} (tol={t})"
        );
    }
}

/// Absolute+relative tolerance compare, for a reference near zero from cancellation (the relative-only
/// `assert_close` floor is too tight there).
fn assert_close_abs_rel(got: &[f32], reference: &[f32], label: &str, abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
    for (i, (a, r)) in got.iter().zip(reference.iter()).enumerate() {
        let t = abs_tol + rel_tol * r.abs();
        assert!(
            (a - r).abs() <= t,
            "{label} elem {i}: ptx {a} vs ref {r} (tol={t})"
        );
    }
}

/// Real-magnitude deterministic fill: values spread roughly in [-30, 30], with every 7th element scaled by
/// 1e-3 to mix magnitudes (stresses reduce cancellation/precision, unlike the [-1, 1] `fill` above).
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

/// Independent f64 host fold (sum), row-major [rows, cols] reduced over the last axis; a second reference
/// independent of the eval oracle.
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

// ── Elementwise ops ──────────────────────────────────────────────────────

#[test]
fn elementwise_add_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
        None => return,
    };
    let n = 32usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[(ai, vec![n], fill(42, n)), (ci, vec![n], fill(99, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_add",
        1e-5,
    );
}

#[test]
fn elementwise_mul_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_mul",
        1e-5,
    );
}

#[test]
fn elementwise_neg_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_neg",
        1e-5,
    );
}

// ── Matmul ───────────────────────────────────────────────────────────────

#[test]
fn matmul_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
        None => return,
    };
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_matmul",
        1e-4,
    );
}

// ── Reduce ───────────────────────────────────────────────────────────────

#[test]
fn reduce_sum_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    let tol = 1e-4 * (cols as f32).sqrt();
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_reduce_sum",
        tol,
    );
}

#[test]
fn reduce_max_last_axis_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    // Max is exact (order-independent).
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_reduce_max",
        1e-6,
    );
}

// ── Reduce: non-power-of-2, wide, real-magnitude ────────────────────────
//
// Covers non-power-of-2 reduce (the earlier PTX "non_power_of_2" test exercised BinOp::Add, not Reduce) and
// real hidden-dim magnitudes (earlier reduce tests used tiny [-1, 1] data). Widths 33, 896 and 1536 are
// non-powers-of-2 matching real transformer hidden dims.

#[test]
fn reduce_sum_nonpow2_wide_real_magnitude_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
        let got = common::run_resident(&mut ptx, &g, &inputs);
        let label = format!("ptx_reduce_sum_nonpow2_wide_cols{cols}");
        // Real-magnitude data (up to |30|) with mixed large/small terms: bound rounding noise by an
        // absolute+relative tolerance scaling with sqrt(cols), since the relative-only floor is too tight when a
        // row's sum cancels near 0.
        let abs_tol = 0.02 * (cols as f32).sqrt();
        assert_close_abs_rel(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            &label,
            abs_tol,
            1e-4,
        );

        // Independent f64 host-fold cross-check at the real hidden-dim width (real-magnitude vs f64, never
        // poot-vs-poot at one precision).
        if cols == 896 {
            let want_f64 = f64_row_sum(&data, rows, cols);
            assert_close_abs_rel(
                got.as_f32().unwrap(),
                &want_f64,
                "ptx_reduce_sum_nonpow2_wide_cols896_vs_f64",
                0.05,
                1e-4,
            );
        }
    }
}

#[test]
fn reduce_max_nonpow2_wide_real_magnitude_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
        let got = common::run_resident(&mut ptx, &g, &inputs);
        let label = format!("ptx_reduce_max_nonpow2_wide_cols{cols}");
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
                "ptx_reduce_max_nonpow2_wide_cols896_vs_f64",
                1e-5,
                1e-6,
            );
        }
    }
}

// ── RMSNorm / LayerNorm ───────────────────────────────────────────────────
//
// Composes Reduce(Sum, keepdim) internally. Non-power-of-2 width, real-magnitude input.

#[test]
fn rmsnorm_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    // Weights stay near typical learned-scale magnitude (not blown up to +-30).
    let wd: Vec<f32> = fill(932, n).iter().map(|v| 1.0 + v.abs()).collect();
    let inputs = make_inputs(&[(xi, vec![1, 1, n], xd), (wi, vec![n], wd)]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_rmsnorm_nonpow2",
        1e-4,
    );
}

#[test]
fn layernorm_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_layernorm_nonpow2",
        1e-4,
    );
}

// ── Stress: non-power-of-2 ──────────────────────────────────────────────

#[test]
fn non_power_of_2_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_npo2_33",
        1e-5,
    );
}

// ── Stress: single element ───────────────────────────────────────────────

#[test]
fn single_element_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_single",
        1e-6,
    );
}

// ── Stress: large tensor ─────────────────────────────────────────────────

#[test]
fn large_elementwise_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
        None => return,
    };
    let n = 1024usize;
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![n]));
    let c = b.constant("c", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Mul, a, c);
    let (ai, ci) = (a.id, c.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[(ai, vec![n], fill(701, n)), (ci, vec![n], fill(702, n))]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_large_1024",
        1e-5,
    );
}

// Cross-backend op-coverage parity (2026-07-28): ops wgpu tests cross-backend but PTX did not; mirrors the
// ROCm coverage tests. Skip-gated (try_ptx None on no-CUDA).

#[test]
fn gather_axis0_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_eq!(got.shape(), cpu.shape());
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_gather",
        1e-6,
    );
}

#[test]
fn matmul_bias_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_matmul_bias",
        1e-4,
    );
}

#[test]
fn broadcast_binary_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_broadcast_add",
        1e-6,
    );
}

#[test]
fn transpose_slice_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
        None => return,
    };
    let (r, c) = (8usize, 4usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![r, c]));
    let t = b.transpose(x, vec![1, 0]);
    let s = b.slice(t, 1, 2, 6);
    let xi = x.id;
    let g = b.finish(s);
    let inputs = make_inputs(&[(xi, vec![r, c], fill(701, r * c))]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_eq!(got.shape(), cpu.shape());
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_transpose_slice",
        1e-6,
    );
}

#[test]
fn elementwise_sub_div_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_sub_div",
        1e-5,
    );
}

fn unary_ptx_case(op: UnOp, xd: Vec<f32>, label: &str) {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
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

/// Card 630 SC-005 (PTX, pod batch): `Tanh` and `Erf` over the first-row table within tier 2 of the
/// oracle, and `tanh(+-100)` exactly `+-1`.
#[test]
fn unary_tanh_ptx_matches_cpu() {
    unary_transcendental_ptx(UnOp::Tanh, "ptx_tanh");
}

#[test]
fn unary_erf_ptx_matches_cpu() {
    unary_transcendental_ptx(UnOp::Erf, "ptx_erf");
}

fn unary_transcendental_ptx(op: UnOp, label: &str) {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_transcendental_tier2(got.as_f32().unwrap(), cpu.as_f32().unwrap(), label, 10.0);
}

#[test]
fn unary_exp_ptx_matches_cpu() {
    unary_ptx_case(UnOp::Exp, vec![0.0f32, 0.5, 1.0, -1.0], "ptx_exp");
}

#[test]
fn unary_sqrt_ptx_matches_cpu() {
    unary_ptx_case(UnOp::Sqrt, vec![1.0f32, 4.0, 9.0, 0.25], "ptx_sqrt");
}

#[test]
fn unary_log_ptx_matches_cpu() {
    unary_ptx_case(
        UnOp::Log,
        vec![1.0f32, std::f32::consts::E, 7.389056, 0.5],
        "ptx_log",
    );
}

#[test]
fn masked_attention_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_attention",
        5e-3,
    );
}

#[test]
fn decomposed_prefill_attention_over_65535_query_groups_ptx_matches_cpu() {
    // Card 1006: the traced prefill attention chain at Hq*L = 66560 > 65535 (the wgpu X grid cap; its
    // row-parallel softmax needs one workgroup per (head, row)). The result matches the CPU oracle.
    let mut ptx = match try_ptx() {
        Some(g) => g,
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
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_decomposed_prefill_attention",
        2e-3,
    );
}

/// Card 258 cross-backend data point: the PTX twin
/// of `crates/poot-gpu/tests/cross_backend.rs::large_vocab_lm_head_transpose_matmul_does_not_hang`: a bare
/// `transpose([250880,1024]) -> matmul` graph at BLOOM's tied-lm_head shape on the PTX executor. Card 258's
/// hypothesis is a display-watchdog TDR specific to the wgpu/RADV iGPU (a compute-only NVIDIA GPU has none), so
/// this checks whether `poot-graph-plan`'s `is_decode_gemv` plan is slow or wrong at this scale on PTX too
/// (a general planning gap) or fine (the watchdog is the whole story). Run on a RunPod NVIDIA pod
/// through `poot-orchestrator exec`.
#[test]
#[ignore = "diagnostic: allocates a ~1GB weight, mirroring the wgpu twin's own ignore convention (crates/ \
            poot-gpu/tests/cross_backend.rs); run manually with --ignored, ideally on a RunPod NVIDIA pod \
            since no local NVIDIA hardware is assumed"]
fn large_vocab_lm_head_transpose_matmul_does_not_hang_ptx() {
    let mut ptx = match try_ptx() {
        Some(g) => g,
        None => return,
    };
    let (hidden, vocab) = (1024usize, 250880usize);
    let b = Builder::new();
    let hstate = b.constant("hidden", TensorType::f32(vec![1, 1, hidden]));
    let w = b.constant("lm_head.weight", TensorType::f32(vec![vocab, hidden]));
    let wt = b.transpose(w, vec![1, 0]); // [hidden, vocab]
    let out = b.matmul(hstate, wt); // [1,1,vocab]
    let (hi, wi) = (hstate.id, w.id);
    let g = b.finish(out);

    let inputs = make_inputs(&[
        (hi, vec![1, 1, hidden], fill(1, hidden)),
        (wi, vec![vocab, hidden], fill(2, vocab * hidden)),
    ]);

    let t0 = std::time::Instant::now();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    eprintln!("PTX large lm_head transpose+matmul: {:?}", t0.elapsed());
    assert_eq!(got.as_f32().unwrap().len(), vocab);
    assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
}

// ── Sampling primitives (card 551a) ──

/// Card 551a (SC-001, SC-002, SC-005, SC-007): `OpKind::SampleToken { rule: Greedy }` through the full
/// compile/planner/executor contract matches the CPU oracle bit for bit on PTX/NVIDIA. Row 0 carries
/// two ties, isolated from any non-finite logit so each tie-break is actually observable in the final
/// token rather than masked by the non-finite override: a per-lane tie (indices 5 and 5+64, same lane
/// 5 two groups over - already resolved by the per-lane scan's own strict `>` before the cross-lane
/// fold ever sees it) and a cross-lane tie (indices 1 and 64, different lanes - the case the
/// cross-lane fold's `ci < bi` tie-break exists for; wants token 1, the lowest of the four tied
/// indices). Row 1 carries the non-finite logit (NaN at the last index) alone.
#[test]
fn sample_token_greedy_ptx_matches_cpu() {
    let mut ptx = match try_ptx() {
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
    let inputs = make_inputs(&[(li, vec![rows, vocab], data)]);

    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident_i32(&mut ptx, &g, &inputs);
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token Greedy through the full contract must match the CPU oracle on PTX"
    );
}

/// Card 551a (SC-001, SC-010): `OpKind::RandomUniform` through the full compile/planner/executor
/// contract matches the CPU oracle bit for bit on PTX/NVIDIA (tier 1, ADR-0101). `seed` is I32, so it
/// rides as a `Slot::Sampler` (`make_inputs` is F32-only, like every other fixture in this file).
#[test]
fn random_uniform_ptx_matches_cpu() {
    use poot_graph_ir::Slot;

    let mut ptx = match try_ptx() {
        Some(p) => p,
        None => return,
    };
    let (rows, cols) = (3usize, 37usize);
    let b = Builder::new();
    let seed = b.slot_named(
        Slot::Sampler,
        "seed",
        TensorType::new(vec![rows], poot_tensor::DType::I32),
    );
    let out = b.random_uniform(seed, cols);
    let g = b.finish(out);

    let seed_data: Vec<i32> = vec![0, 1, 0xDEADBEEFu32 as i32];
    let inputs: HashMap<ValueId, Value> =
        HashMap::from([(seed.id, HostTensor::i32(vec![rows], seed_data).into())]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident(&mut ptx, &g, &inputs);
    assert_close(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "ptx_random_uniform",
        0.0,
    );
}

/// Card 677 SC-001, PTX leg: `OpKind::SampleToken { rule: GumbelTopKTopP }` with top-p the ONLY
/// active truncation (`top_k = 0`, min-p disabled via `floor_offset = -inf`) must still truncate on
/// PTX/NVIDIA, matching the CPU oracle bit for bit. Same fixture as the wgpu/ROCm siblings
/// (`sample_token_gumbel_topp_only_excludes_untruncated_winner_cross_backend`/`_rocm_matches_cpu`):
/// logits 3 and 6 tied at the row max (10.0); logit 2 sits just below (9.0) but carries a noise boost
/// (20.0) that makes it the Gumbel-max winner of the UNTRUNCATED row. `top_p = 0.7` gives the tied
/// pair a quantized mass of 32 against a threshold of `round(0.7 * 38) = 27`, excluding the decoy.
/// Found by card 551b's review: `poot-eval`'s `sample_one_row` used to start this bisection's floor
/// at `-inf` whenever neither min-p nor top-k had already raised it, pinning it (`(-inf + finite) *
/// 0.5 == -inf`) and silently disabling top-p everywhere, including this device kernel's own oracle
/// comparison.
#[test]
fn sample_token_gumbel_topp_only_excludes_untruncated_winner_ptx_matches_cpu() {
    use poot_graph_ir::Slot;

    let mut ptx = match try_ptx() {
        Some(p) => p,
        None => return,
    };
    let (rows, vocab) = (1usize, 8usize);
    let b = Builder::new();
    let logits = b.constant("logits", TensorType::f32(vec![rows, vocab]));
    let noise = b.constant("noise", TensorType::f32(vec![rows, vocab]));
    let params = b.constant("params", TensorType::f32(vec![rows, 4]));
    let top_k = b.slot_named(
        Slot::Sampler,
        "top_k",
        TensorType::new(vec![rows], poot_tensor::DType::I32),
    );
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
    let top_k_data: Vec<i32> = vec![0]; // top_k disabled; top_p alone must do the filtering
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (li, HostTensor::f32(vec![rows, vocab], logits_data).into()),
        (ni, HostTensor::f32(vec![rows, vocab], noise_data).into()),
        (pi, HostTensor::f32(vec![rows, 4], params_data).into()),
        (top_k.id, HostTensor::i32(vec![rows], top_k_data).into()),
    ]);
    let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = common::run_resident_i32(&mut ptx, &g, &inputs);
    assert_eq!(
        got,
        cpu.as_i32().expect("SampleToken output is I32").to_vec(),
        "sample_token GumbelTopKTopP through the full contract must match the CPU oracle on PTX: \
         top_p=0.7 alone (no min-p/top-k) must exclude the noise-boosted decoy (index 2), tie-break \
         over {{3, 6}} picks 3"
    );
    assert_eq!(got, vec![3, -1]);
}

// `i32_slot_gather_variants_compiled_match_cpu_bit_exact_ptx` (spike 562 F-9, card 643) is not ported to
// the executor contract: its Dus/CastDus variants seed carried state with specific non-zero bytes
// before the one run (`I32SlotGatherVariant::state()`), but `poot_executor::Device`/`Executor` has no
// "seed a state buffer with bytes" primitive - `Engine`'s state is always zero at first declaration
// (Z5), and the only way to make it non-zero is to run a real donating/committing step first. Building
// a generic priming entry for that is more than this card's scope. The bug class itself (an I32
// `Slot::Pos` value that is both a Gather index and feeds another use) is still covered cross-backend by
// `i32_slot_gather_variants_compiled_match_cpu_bit_exact` on wgpu/ROCm (`poot_test_util::
// i32_slot_gather`, shared ground truth); only PTX's own row is missing until the contract gains a
// seeding primitive or a priming-step pattern.
