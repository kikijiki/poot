//! Primitive-op fuzz: reduce, matmul, transpose, broadcast, elementwise, softmax, fused chains/epilogues.

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, UnOp};
use poot_graph_ir::ops::softmax;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error_f64};

// -- card 035 shape/axis fuzzer: independent oracle vs poot-eval for primitive ops --
//
// Each test evaluates a poot-graph via eval() and compares against a hand-written reference that does
// not call poot-eval internals. Traversal orders differ from poot-eval's so a systematic indexing bug
// cannot cancel. CPU-only.

/// Row-major strides for `shape`. The last dimension always has stride 1.
fn fuzz_strides(shape: &[usize]) -> Vec<usize> {
    let r = shape.len();
    let mut s = vec![1usize; r];
    for i in (0..r.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Naive reduce: iterate over all input elements and route each to its output slot by zeroing the
/// reduced axis coordinate. poot-eval iterates over output elements, so an indexing bug cannot cancel.
fn fuzz_naive_reduce(
    x: &[f32],
    shape: &[usize],
    axis: usize,
    keepdim: bool,
    is_sum: bool,
) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let init = if is_sum { 0.0f32 } else { f32::NEG_INFINITY };
    let mut out = vec![init; n_out];
    let st_x = fuzz_strides(shape);
    let st_out = fuzz_strides(&out_shape);
    let n_x: usize = shape.iter().product();
    for (flat_x, &v) in x.iter().enumerate().take(n_x) {
        // unravel the input flat index.
        let mut rem = flat_x;
        let mut idx_x = vec![0usize; shape.len()];
        for d in 0..shape.len() {
            idx_x[d] = rem / st_x[d];
            rem %= st_x[d];
        }
        // compute the output flat index.
        let flat_out: usize = if keepdim {
            let mut io = idx_x.clone();
            io[axis] = 0; // zero the reduced axis
            io.iter().zip(&st_out).map(|(c, s)| c * s).sum()
        } else {
            // drop the axis coord and re-pack the remaining coordinates.
            let io: Vec<usize> = idx_x
                .iter()
                .enumerate()
                .filter(|&(d, _)| d != axis)
                .map(|(_, &v)| v)
                .collect();
            io.iter().zip(&st_out).map(|(c, s)| c * s).sum()
        };
        if is_sum {
            out[flat_out] += v;
        } else if v > out[flat_out] {
            out[flat_out] = v;
        }
    }
    (out, out_shape)
}

/// Naive transpose: direct index-permutation formula. Output index `i` refers to input axis
/// `perm[i]`, so `in_idx[perm[i]] = out_idx[i]`.
fn fuzz_naive_transpose(x: &[f32], shape: &[usize], perm: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let n = x.len();
    let st_x = fuzz_strides(shape);
    let st_out = fuzz_strides(&out_shape);
    let mut out = vec![0.0f32; n];
    for (flat_out, out_val) in out.iter_mut().enumerate() {
        let mut rem = flat_out;
        let mut out_idx = vec![0usize; out_shape.len()];
        for d in 0..out_shape.len() {
            out_idx[d] = rem / st_out[d];
            rem %= st_out[d];
        }
        let mut in_idx = vec![0usize; shape.len()];
        for (i, &p) in perm.iter().enumerate() {
            in_idx[p] = out_idx[i];
        }
        let flat_x: usize = in_idx.iter().zip(&st_x).map(|(c, s)| c * s).sum();
        *out_val = x[flat_x];
    }
    (out, out_shape)
}

/// Naive broadcast: right-aligned numpy rules; a size-1 dim contributes coordinate 0.
fn fuzz_naive_broadcast(x: &[f32], in_shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let n: usize = out_shape.iter().product();
    let st_out = fuzz_strides(out_shape);
    let st_in = fuzz_strides(in_shape);
    let off = out_shape.len() - in_shape.len(); // right-align
    let mut out = vec![0.0f32; n];
    for (flat_out, out_val) in out.iter_mut().enumerate() {
        let mut rem = flat_out;
        let mut out_idx = vec![0usize; out_shape.len()];
        for d in 0..out_shape.len() {
            out_idx[d] = rem / st_out[d];
            rem %= st_out[d];
        }
        let src_flat: usize = in_shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                let coord = if dim == 1 { 0 } else { out_idx[off + i] };
                coord * st_in[i]
            })
            .sum();
        *out_val = x[src_flat];
    }
    out
}

/// Card 035: random-shape reduce correctness vs an independent naive reference.
///
/// `reduce_over_non_last_axis_keepdim_is_correct` covers one fixed [2,3,4] shape, and the reduction
/// fuzzer (`fused_equals_unfused_over_random_reduction_graphs`) only reduces the last axis and checks
/// fusion-invariance. This covers every axis on rank-2/3/4 tensors, keepdim true/false, Sum and Max:
/// 2 rank-2 shapes x 2 axes + 3 rank-3 shapes x 3 axes + 1 rank-4 shape x 4 axes = 17 shape/axis
/// pairs, x 2 keepdim x 2 ops = 68 checks.
#[test]
fn fuzz_reduce_all_axes_and_keepdim() {
    let cases: &[&[usize]] = &[
        // rank 2
        &[3, 4],
        &[7, 5],
        // rank 3
        &[2, 3, 4],
        &[5, 1, 7],
        &[3, 4, 5],
        // rank 4
        &[2, 3, 2, 4],
    ];

    let mut count = 0usize;
    for shape in cases {
        let rank = shape.len();
        let numel: usize = shape.iter().product();
        let xd = fill(numel, 0xDEAD_BEEF ^ numel as u64 ^ rank as u64);

        for axis in 0..rank {
            for &keepdim in &[true, false] {
                for &is_sum in &[true, false] {
                    let op = if is_sum { RedOp::Sum } else { RedOp::Max };

                    // poot-eval path.
                    let b = Builder::new();
                    let x = b.constant("x", TensorType::f32(shape.to_vec()));
                    let r = b.reduce(op, x, axis, keepdim);
                    let g = b.finish(r);
                    let mut inp = HashMap::new();
                    inp.insert(
                        x.id,
                        Value::from(HostTensor::f32(shape.to_vec(), xd.clone())),
                    );
                    let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                        .map(|r| {
                            r.output
                                .into_host()
                                .expect("fuzz_primitives tests evaluate dense graphs")
                        })
                        .unwrap_or_else(|e| {
                            panic!(
                                "fuzz_reduce: shape={shape:?} axis={axis} keepdim={keepdim} \
                             is_sum={is_sum}: eval error {e}"
                            )
                        });

                    // independent naive reference.
                    let (want, want_shape) = fuzz_naive_reduce(&xd, shape, axis, keepdim, is_sum);

                    assert_eq!(
                        got.shape(),
                        want_shape,
                        "fuzz_reduce: shape={shape:?} axis={axis} keepdim={keepdim} \
                         is_sum={is_sum}: output shape mismatch got={:?} want={:?}",
                        got.shape(),
                        want_shape
                    );
                    // 1e-5 relative tolerance catches wrong-axis bugs (many ULPs off) without failing on
                    // benign reordering.
                    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
                    count += 1;
                }
            }
        }
    }
    eprintln!("fuzz_reduce_all_axes_and_keepdim: {count} cases passed");
    assert!(count >= 60, "expected >= 60 cases, got {count}");
}

/// Card 035: matmul correctness at varied shapes (odd M/K/N, K=1, batched rank-3) vs a naive
/// triple-loop reference. `matmul_matches_naive` covers only `[1,1,3]x[3,2]`.
#[test]
fn fuzz_matmul_random_shapes() {
    // (a_shape, b_shape).  a=[..,M,K], b=[..,K,N].
    let cases: &[(&[usize], &[usize])] = &[
        // plain 2-D
        (&[1, 3], &[3, 1]),    // M=1, K=3, N=1
        (&[5, 7], &[7, 3]),    // odd M/N
        (&[7, 1], &[1, 5]),    // K=1
        (&[3, 3], &[3, 3]),    // square
        (&[33, 7], &[7, 11]),  // partial/odd widths
        (&[1, 1], &[1, 1]),    // trivial 1x1
        (&[100, 1], &[1, 50]), // K=1, large M and N
        // rank-3 a, rank-2 b (batched a, broadcast b).
        (&[2, 3, 4], &[4, 5]), // batch=2
        (&[3, 5, 7], &[7, 2]), // batch=3, odd K/N
        (&[1, 7, 3], &[3, 3]), // batch=1 == non-batched
        // rank-3 both (same batch size, no broadcast needed).
        (&[2, 4, 3], &[2, 3, 5]),
        (&[3, 1, 6], &[3, 6, 4]), // M=1
    ];

    for (ci, (a_shape, b_shape)) in cases.iter().enumerate() {
        let seed = ci as u64 * 17 + 0xFEED;
        let na: usize = a_shape.iter().product();
        let nb: usize = b_shape.iter().product();
        let ad = fill(na, seed);
        let bd = fill(nb, seed + 1);

        let bld = Builder::new();
        let av = bld.constant("a", TensorType::f32(a_shape.to_vec()));
        let bv = bld.constant("b", TensorType::f32(b_shape.to_vec()));
        let out = bld.matmul(av, bv);
        let g = bld.finish(out);
        let mut inp = HashMap::new();
        inp.insert(
            av.id,
            Value::from(HostTensor::f32(a_shape.to_vec(), ad.clone())),
        );
        inp.insert(
            bv.id,
            Value::from(HostTensor::f32(b_shape.to_vec(), bd.clone())),
        );
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .unwrap_or_else(|e| panic!("fuzz_matmul case {ci} a={a_shape:?} b={b_shape:?}: {e}"));

        // naive batch-aware triple loop using the existing matmul_ref helper.
        let ra = a_shape.len();
        let rb = b_shape.len();
        let (m, k) = (a_shape[ra - 2], a_shape[ra - 1]);
        let n = b_shape[rb - 1];
        let batch_a: usize = a_shape[..ra - 2].iter().product::<usize>().max(1);
        let batch_b: usize = b_shape[..rb - 2].iter().product::<usize>().max(1);
        let batch = batch_a.max(batch_b);
        let mut want = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            let a_bi = (bi % batch_a) * m * k;
            let b_bi = (bi % batch_b) * k * n;
            let c_bi = bi * m * n;
            let c = matmul_ref(&ad[a_bi..a_bi + m * k], &bd[b_bi..b_bi + k * n], m, k, n);
            want[c_bi..c_bi + m * n].copy_from_slice(&c);
        }

        assert_eq!(
            got.as_f32().unwrap().len(),
            want.len(),
            "fuzz_matmul case {ci} a={a_shape:?} b={b_shape:?}: output size mismatch"
        );
        // use a slightly relaxed tolerance for accumulated sums.
        assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
    }
    eprintln!("fuzz_matmul_random_shapes: {} cases passed", cases.len());
}

/// Card 035: transpose correctness at rank-2/3/4 with various permutations vs a naive index-permutation
/// formula. The movement fuzzer (`fused_equals_unfused_over_random_movement_graphs`) uses eval() as its
/// own reference and cannot detect a systematic bug in eval's transpose handler.
#[test]
fn fuzz_transpose_correctness() {
    // (shape, perm).
    let cases: &[(&[usize], &[usize])] = &[
        // rank-2
        (&[3, 4], &[1, 0]),
        (&[7, 5], &[1, 0]),
        // rank-3: all 6 non-trivial permutations of [2,3,4].
        (&[2, 3, 4], &[1, 0, 2]),
        (&[2, 3, 4], &[2, 1, 0]),
        (&[2, 3, 4], &[0, 2, 1]),
        (&[2, 3, 4], &[1, 2, 0]),
        (&[2, 3, 4], &[2, 0, 1]),
        // rank-3 shapes with size-1 dims (exercises stride-1 paths).
        (&[5, 1, 7], &[2, 0, 1]),
        (&[3, 4, 5], &[1, 0, 2]),
        // rank-4
        (&[2, 3, 4, 5], &[3, 2, 1, 0]),
        (&[2, 3, 4, 5], &[0, 2, 1, 3]),
        (&[2, 3, 4, 5], &[1, 0, 3, 2]),
        (&[1, 4, 3, 2], &[3, 0, 2, 1]),
    ];

    for (ci, (shape, perm)) in cases.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let xd = fill(numel, ci as u64 * 31 + 0xBEEF);

        let bld = Builder::new();
        let x = bld.constant("x", TensorType::f32(shape.to_vec()));
        let t = bld.transpose(x, perm.to_vec());
        let g = bld.finish(t);
        let mut inp = HashMap::new();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(shape.to_vec(), xd.clone())),
        );
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .unwrap_or_else(|e| {
                panic!("fuzz_transpose case {ci} shape={shape:?} perm={perm:?}: {e}")
            });

        let (want, want_shape) = fuzz_naive_transpose(&xd, shape, perm);

        assert_eq!(
            got.shape(),
            want_shape,
            "fuzz_transpose case {ci}: output shape mismatch"
        );
        // transpose is pure index remapping, so must be bit-identical.
        for (i, (&gv, &wv)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
            assert_eq!(
                gv.to_bits(),
                wv.to_bits(),
                "fuzz_transpose case {ci} shape={shape:?} perm={perm:?} elem {i}: got {gv} want {wv}"
            );
        }
    }
    eprintln!("fuzz_transpose_correctness: {} cases passed", cases.len());
}

/// Card 035: broadcast correctness vs a naive right-aligned expansion formula.
///
/// Covers size-1 expansion, multi-axis broadcast, and rank promotion (leading size-1 dims).
#[test]
fn fuzz_broadcast_correctness() {
    // (in_shape, out_shape).
    let cases: &[(&[usize], &[usize])] = &[
        // rank-1
        (&[1], &[5]),
        (&[7], &[7]), // no-op
        // rank-2
        (&[1, 4], &[3, 4]),
        (&[3, 1], &[3, 4]),
        (&[1, 1], &[3, 4]),
        // rank-3
        (&[1, 1, 4], &[2, 3, 4]),
        (&[2, 1, 4], &[2, 3, 4]),
        (&[1, 3, 4], &[2, 3, 4]),
        (&[2, 3, 1], &[2, 3, 5]),
        // rank promotion (leading dims implicitly 1).
        (&[4], &[2, 3, 4]),
        (&[3, 4], &[2, 3, 4]),
    ];

    for (ci, (in_shape, out_shape)) in cases.iter().enumerate() {
        let numel_in: usize = in_shape.iter().product::<usize>().max(1);
        let xd = fill(numel_in, ci as u64 * 41 + 0xCAFE);

        let bld = Builder::new();
        let x = bld.constant("x", TensorType::f32(in_shape.to_vec()));
        let y = bld.broadcast(x, out_shape.to_vec());
        let g = bld.finish(y);
        let mut inp = HashMap::new();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(in_shape.to_vec(), xd.clone())),
        );
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .unwrap_or_else(|e| {
                panic!("fuzz_broadcast case {ci} in={in_shape:?} out={out_shape:?}: {e}")
            });

        let want = fuzz_naive_broadcast(&xd, in_shape, out_shape);

        assert_eq!(
            got.shape(),
            out_shape.to_vec(),
            "fuzz_broadcast case {ci}: output shape mismatch"
        );
        // broadcast is pure index remapping: bit-identical.
        for (i, (&gv, &wv)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
            assert_eq!(
                gv.to_bits(),
                wv.to_bits(),
                "fuzz_broadcast case {ci} in={in_shape:?} out={out_shape:?} elem {i}: got {gv} want {wv}"
            );
        }
    }
    eprintln!("fuzz_broadcast_correctness: {} cases passed", cases.len());
}

/// Card 035: elementwise unary and binary op correctness at partial/odd widths (7, 33, 100). The
/// pointwise fuzzer (`fused_equals_unfused_over_random_pointwise_graphs`) uses fixed shape [1,4,8] and
/// checks fusion-invariance only; these sizes (not multiples of 4 or 8) expose stride/length off-by-one bugs.
#[test]
fn fuzz_elementwise_correctness() {
    // --- unary ops at widths 7, 33, 100 ---
    for (wi, &width) in [7usize, 33, 100].iter().enumerate() {
        let xd = fill(width, wi as u64 * 101 + 0x1234);

        // Neg: exact negation, bit-identical.
        {
            let bld = Builder::new();
            let x = bld.constant("x", TensorType::f32(vec![width]));
            let y = bld.unary(UnOp::Neg, x);
            let g = bld.finish(y);
            let mut inp = HashMap::new();
            inp.insert(x.id, Value::from(HostTensor::f32(vec![width], xd.clone())));
            let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    r.output
                        .into_host()
                        .expect("fuzz_primitives tests evaluate dense graphs")
                })
                .unwrap();
            let want: Vec<f32> = xd.iter().map(|&v| -v).collect();
            for (i, (&gv, &wv)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    gv.to_bits(),
                    wv.to_bits(),
                    "neg width={width} elem {i}: got {gv} want {wv}"
                );
            }
        }

        // Exp: same formula, bit-identical.
        {
            let bld = Builder::new();
            let x = bld.constant("x", TensorType::f32(vec![width]));
            let y = bld.unary(UnOp::Exp, x);
            let g = bld.finish(y);
            let mut inp = HashMap::new();
            inp.insert(x.id, Value::from(HostTensor::f32(vec![width], xd.clone())));
            let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    r.output
                        .into_host()
                        .expect("fuzz_primitives tests evaluate dense graphs")
                })
                .unwrap();
            let want: Vec<f32> = xd.iter().map(|&v| v.exp()).collect();
            for (i, (&gv, &wv)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    gv.to_bits(),
                    wv.to_bits(),
                    "exp width={width} elem {i}: got {gv} want {wv}"
                );
            }
        }

        // Tanh and Erf against the f64 libm functions, within 2 ulp of f32 at unit scale.
        for (name, op, want_fn) in [
            (
                "tanh",
                UnOp::Tanh,
                (|v| libm::tanh(v as f64) as f32) as fn(f32) -> f32,
            ),
            ("erf", UnOp::Erf, |v| libm::erf(v as f64) as f32),
        ] {
            let bld = Builder::new();
            let x = bld.constant("x", TensorType::f32(vec![width]));
            let y = bld.unary(op, x);
            let g = bld.finish(y);
            let mut inp = HashMap::new();
            inp.insert(x.id, Value::from(HostTensor::f32(vec![width], xd.clone())));
            let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    r.output
                        .into_host()
                        .expect("fuzz_primitives tests evaluate dense graphs")
                })
                .unwrap();
            for (i, (&gv, &xv)) in got.as_f32().unwrap().iter().zip(xd.iter()).enumerate() {
                let wv = want_fn(xv);
                assert!(
                    (gv - wv).abs() <= 2.0 * f32::EPSILON * wv.abs().max(1.0),
                    "{name} width={width} elem {i} (x={xv}): got {gv} want {wv}"
                );
            }
        }
    }

    // --- binary ops at widths 7, 33 (same-shape, no broadcast) ---
    let bin_cases = [
        (BinOp::Add, "add"),
        (BinOp::Sub, "sub"),
        (BinOp::Mul, "mul"),
        (BinOp::Max, "max"),
    ];
    for &width in &[7usize, 33] {
        for (opi, (op, op_name)) in bin_cases.iter().enumerate() {
            let seed = (width ^ (opi * 7)) as u64 + 0xAAAA;
            let ad = fill(width, seed);
            let bd = fill(width, seed + 1);

            let bld = Builder::new();
            let av = bld.constant("a", TensorType::f32(vec![width]));
            let bv = bld.constant("b", TensorType::f32(vec![width]));
            let r = bld.binary(*op, av, bv);
            let g = bld.finish(r);
            let mut inp = HashMap::new();
            inp.insert(av.id, Value::from(HostTensor::f32(vec![width], ad.clone())));
            inp.insert(bv.id, Value::from(HostTensor::f32(vec![width], bd.clone())));
            let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| {
                    r.output
                        .into_host()
                        .expect("fuzz_primitives tests evaluate dense graphs")
                })
                .unwrap();

            // direct formula reference.
            let want: Vec<f32> = ad
                .iter()
                .zip(bd.iter())
                .map(|(&a, &b)| match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Max => a.max(b),
                    _ => panic!("unexpected op"),
                })
                .collect();

            // no reassociation in binary ops: bit-identical.
            for (i, (&gv, &wv)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    gv.to_bits(),
                    wv.to_bits(),
                    "binary {op_name} width={width} elem {i}: got {gv} want {wv}"
                );
            }
        }
    }
    eprintln!("fuzz_elementwise_correctness: passed");
}

/// Card 035: softmax graph (ops::softmax = reduce-max/sub/exp/reduce-sum/div) vs an independent two-pass
/// formula applied row-by-row, checking that eval's reduce, broadcast, exp, and div handlers compose.
#[test]
fn fuzz_softmax_vs_independent_formula() {
    let cases: &[&[usize]] = &[&[8], &[3, 8], &[2, 3, 8], &[5, 16], &[1, 4, 32]];

    for (ci, shape) in cases.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let n_last = *shape.last().unwrap();
        let n_rows = numel / n_last;
        let xd = fill(numel, ci as u64 * 7 + 0x9999);

        let bld = Builder::new();
        let x = bld.constant("x", TensorType::f32(shape.to_vec()));
        let s = softmax(&bld, x);
        let g = bld.finish(s);
        let mut inp = HashMap::new();
        inp.insert(
            x.id,
            Value::from(HostTensor::f32(shape.to_vec(), xd.clone())),
        );
        let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .unwrap_or_else(|e| panic!("fuzz_softmax case {ci} shape={shape:?}: {e}"));

        // independent reference: apply softmax_ref to each row.
        let mut want = vec![0.0f32; numel];
        for r in 0..n_rows {
            let row_ref = softmax_ref(&xd[r * n_last..(r + 1) * n_last]);
            want[r * n_last..(r + 1) * n_last].copy_from_slice(&row_ref);
        }

        assert_eq!(
            got.shape(),
            shape.to_vec(),
            "fuzz_softmax case {ci}: shape mismatch"
        );
        assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
    }
    eprintln!(
        "fuzz_softmax_vs_independent_formula: {} cases passed",
        cases.len()
    );
}

/// Real-magnitude check: existing softmax oracles (`fuzz_softmax_vs_independent_formula`, ...) feed
/// scores in roughly [-8, 8], but real attention logits can span much wider (attention sinks, unscaled
/// bias) and naive `exp(x)` overflows f32 above ~88. This feeds scores spanning [-300, 300] through the
/// two-pass `softmax`, checking it against an independent f64 ground truth. The max-shifted `exp(x-m)`
/// is bounded to (0, 1] before the sum, so it should stay tight (a regression guard; compare the GDN
/// cumulative-log-decay cancellation).
#[test]
fn softmax_stays_accurate_at_extreme_score_magnitude() {
    let l = 256usize;
    // scores spanning [-300, 300]: far past the x>~88 naive-exp overflow threshold.
    let scores: Vec<f32> = fill(l, 0xF00D).iter().map(|v| v * 300.0).collect();

    // f64 ground truth (independent of poot's own max-shift implementation).
    let m64 = scores
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps64: Vec<f64> = scores.iter().map(|&s| (s as f64 - m64).exp()).collect();
    let sum64: f64 = exps64.iter().sum();
    let want: Vec<f64> = exps64.iter().map(|&e| e / sum64).collect();

    // two-pass softmax (poot's `softmax` op).
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![l]));
    let s = softmax(&b, x);
    let g = b.finish(s);
    let mut inp = HashMap::new();
    inp.insert(x.id, Value::from(HostTensor::f32(vec![l], scores.clone())));
    let two_pass = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("fuzz_primitives tests evaluate dense graphs")
        })
        .expect("two-pass softmax at extreme magnitude");
    assert!(
        two_pass.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "two-pass softmax produced a non-finite value at extreme score magnitude"
    );
    let worst_two_pass = max_abs_error_f64(two_pass.as_f32().unwrap(), &want);
    assert!(
        worst_two_pass < 1e-6,
        "two-pass softmax vs f64 ground truth at extreme magnitude: err={worst_two_pass:e}"
    );

    eprintln!(
        "softmax at extreme score magnitude ([-300,300]): stayed within 1e-6 of f64 ground truth \
         (as the max-shift design predicts) - permanent regression guard, not a bug."
    );
}
