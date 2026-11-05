//! Contractions: `MatMul` (every contraction composite decomposes into it) and `IndexedMatMul`.
//!
//! Both read their operands as the equation's [`OperandPrecision`] says: each operand enters the products as one
//! term (exact, or narrowed once) or two (a split f32 operand), and a product is summed for every pair of terms,
//! one from each operand, into one f32 accumulator in `k` order.
//!
//! Output rows are independent, so they are spread over a bounded thread pool; each output element is still
//! accumulated by one thread, in `k` order, with a separate multiply and add (no FMA, no split of `k` across
//! threads), so the result is bit-identical to the serial loop whatever the thread count.

use std::sync::OnceLock;

use poot_graph_ir::{ExpertRows, OperandPrecision, ValueId};
use poot_tensor::{DType, HostTensor};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::index::row_experts;
use super::precision::operand_terms;
use crate::{EvalError, broadcast_flat, strides, unravel};

/// The most threads the oracle's contractions use: the box is shared, and a row-parallel loop gains little past
/// the memory bandwidth of a few cores.
const MAX_ORACLE_THREADS: usize = 16;

/// A reader of one term of an operand: the value at a row-major logical flat index.
pub(crate) type TermReader<'a> = dyn Fn(usize) -> f32 + Sync + 'a;

/// The pool every contraction runs on (shared by the whole process, built on first use).
fn oracle_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(MAX_ORACLE_THREADS);
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("poot-eval-matmul-{i}"))
            .build()
            .expect("build the oracle's contraction thread pool")
    })
}

/// `MatMul`, whatever kernel the planner chose for it: an f32 accumulation over operands read
/// through [`HostTensor::to_f32`] (a BF16/F16 operand widens exactly). The walk stores the F32
/// result as the equation's declared dtype, so a BF16/F16 result is narrowed once, on store, as the
/// GPU kernels do.
pub(crate) fn matmul_to(
    a: &HostTensor,
    b: &HostTensor,
    out_shape: &[usize],
    precision: OperandPrecision,
) -> Result<HostTensor, EvalError> {
    matmul(a, b, out_shape, precision)
}

/// `IndexedMatMul`: `out[m, n] = sum_k x[m, k] * W[e(m), k, n]`, each row contracting against its own
/// expert (the eager oracle for the generated indexed contraction). `x` and `w` are read through
/// [`HostTensor::to_f32`] (either may be BF16/F16); the expert operand is the per-row ids or their grouping, as
/// `select` says ([`row_experts`]).
pub(crate) fn indexed_matmul(
    x: &HostTensor,
    w: &HostTensor,
    (operand, select): (&HostTensor, ExpertRows),
    out_shape: &[usize],
    eqn: ValueId,
    precision: OperandPrecision,
) -> Result<HostTensor, EvalError> {
    let (m, k, n) = (x.shape()[0], x.shape()[1], w.shape()[2]);
    let experts = w.shape()[0];
    let idx_rows = row_experts(operand, select, m, experts, eqn)?;
    let (x_view, w_view) = (x.to_f32()?, w.to_f32()?);
    let (x_terms, w_terms) = if precision.is_exact() {
        (vec![x_view.to_vec()], vec![w_view.to_vec()])
    } else {
        (
            operand_terms(&x_view, x.dtype(), precision),
            operand_terms(&w_view, w.dtype(), precision),
        )
    };
    let mut data = vec![0.0f32; m * n];
    if !data.is_empty() {
        oracle_pool().install(|| {
            data.par_chunks_mut(n).enumerate().for_each(|(mm, row)| {
                let ebase = idx_rows[mm] * k * n;
                for (nn, out) in row.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        for xt in &x_terms {
                            for wt in &w_terms {
                                acc += xt[mm * k + kk] * wt[ebase + kk * n + nn];
                            }
                        }
                    }
                    *out = acc;
                }
            });
        });
    }
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

pub(crate) fn matmul(
    a: &HostTensor,
    b: &HostTensor,
    out_shape: &[usize],
    precision: OperandPrecision,
) -> Result<HostTensor, EvalError> {
    let (av, bv) = (a.to_f32()?, b.to_f32()?);
    let mut data = vec![0.0f32; out_shape.iter().product()];
    if precision.is_exact() {
        matmul_accumulate(
            a.shape(),
            &[&|flat| av[flat]],
            b.shape(),
            &[&|flat| bv[flat]],
            out_shape,
            &mut data,
        );
    } else {
        matmul_narrowed(
            (a.shape(), &av, a.dtype()),
            (b.shape(), &bv, b.dtype()),
            out_shape,
            &mut data,
            precision,
        );
    }
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

/// [`matmul_accumulate`] over operands read as a non-exact `precision` says. Each operand is its shape, its f32
/// reading and the dtype it is stored in (an operand already stored in the fragment dtype enters once).
pub(crate) fn matmul_narrowed(
    (a_shape, a_values, a_stored): (&[usize], &[f32], DType),
    (b_shape, b_values, b_stored): (&[usize], &[f32], DType),
    out_shape: &[usize],
    data: &mut [f32],
    precision: OperandPrecision,
) {
    let (a_terms, b_terms) = (
        operand_terms(a_values, a_stored, precision),
        operand_terms(b_values, b_stored, precision),
    );
    let a_reads: Vec<_> = a_terms.iter().map(|t| move |flat: usize| t[flat]).collect();
    let b_reads: Vec<_> = b_terms.iter().map(|t| move |flat: usize| t[flat]).collect();
    matmul_accumulate(
        a_shape,
        &a_reads.iter().map(|r| r as &TermReader).collect::<Vec<_>>(),
        b_shape,
        &b_reads.iter().map(|r| r as &TermReader).collect::<Vec<_>>(),
        out_shape,
        data,
    );
}

/// The eager matmul definition. Each of `a_terms` and `b_terms` reads row-major logical elements of one term of an
/// operand shaped `a_shape` and `b_shape` (one term unless the operand is split); every `k` step adds the product of
/// each pair of terms, `a` terms outermost. `data` is the zeroed row-major output of `out_shape`.
pub(crate) fn matmul_accumulate(
    a_shape: &[usize],
    a_terms: &[&TermReader],
    b_shape: &[usize],
    b_terms: &[&TermReader],
    out_shape: &[usize],
    data: &mut [f32],
) {
    matmul_accumulate_on(
        oracle_pool(),
        (a_shape, a_terms),
        (b_shape, b_terms),
        out_shape,
        data,
    );
}

/// [`matmul_accumulate`] on `pool`. The output rows (`batch x m`) are the unit of parallelism: a row is
/// written by exactly one thread, which accumulates each of its elements over `k` in order.
fn matmul_accumulate_on(
    pool: &ThreadPool,
    (a_shape, a_terms): (&[usize], &[&TermReader]),
    (b_shape, b_terms): (&[usize], &[&TermReader]),
    out_shape: &[usize],
    data: &mut [f32],
) {
    let (ra, rb) = (a_shape.len(), b_shape.len());
    let (m, k) = (a_shape[ra - 2], a_shape[ra - 1]);
    let n = b_shape[rb - 1];
    if data.is_empty() {
        return;
    }
    let a_st = strides(a_shape);
    let b_st = strides(b_shape);
    let batch_rank = out_shape.len() - 2;
    let (a_rs, a_cs) = (a_st[ra - 2], a_st[ra - 1]);
    let (b_rs, b_cs) = (b_st[rb - 2], b_st[rb - 1]);
    // The output is row-major and contiguous, so row `r` of `batch x m` is `data[r * n..][..n]`.
    pool.install(|| {
        data.par_chunks_mut(n).enumerate().for_each(|(row, out)| {
            let (batch, mi) = (row / m, row % m);
            let batch_idx = if batch_rank == 0 {
                vec![]
            } else {
                unravel(batch, &out_shape[..batch_rank])
            };
            // base offsets into a and b for this batch (broadcast batch dims).
            let a_base = broadcast_flat(&extend_idx(&batch_idx, &[0, 0]), a_shape);
            let b_base = broadcast_flat(&extend_idx(&batch_idx, &[0, 0]), b_shape);
            // cache-friendly ikj order: accumulate the output row by scaling contiguous B rows. With the
            // common B layout [K,N] (stride N over K, 1 over N), the inner ni loop walks B contiguously.
            for ki in 0..k {
                // Every product is computed: `0 * inf` and `0 * NaN` are NaN, never a skipped 0 (ADR-0101).
                let b_row = b_base + ki * b_rs;
                for a_at in a_terms {
                    let av = a_at(a_base + mi * a_rs + ki * a_cs);
                    for b_at in b_terms {
                        for (ni, acc) in out.iter_mut().enumerate() {
                            *acc += av * b_at(b_row + ni * b_cs);
                        }
                    }
                }
            }
        });
    });
}

/// append two trailing coords to a batch index (for computing a batch base offset, coords 0,0).
fn extend_idx(batch: &[usize], tail: &[usize]) -> Vec<usize> {
    let mut v = batch.to_vec();
    v.extend_from_slice(tail);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic xorshift stream of f32 operands. `special` mixes in the values a summation order or a skipped
    /// product would change: infinities, NaNs, subnormals, signed zeros and values that cancel.
    fn operand(len: usize, seed: u64, special: bool) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                if !special {
                    return unit * 8.0;
                }
                match state % 11 {
                    0 => f32::INFINITY,
                    1 => f32::NEG_INFINITY,
                    2 => f32::NAN,
                    3 => f32::from_bits(1 + (state >> 33) as u32 % 1000),
                    4 => 0.0,
                    5 => -0.0,
                    6 => f32::MAX * unit,
                    _ => unit * 1e3,
                }
            })
            .collect()
    }

    /// The definition, one output element at a time: the products of every pair of terms (`a` outermost) added
    /// to one f32 accumulator in `k` order, with a separate multiply and add.
    fn serial(
        (a_shape, a_terms): (&[usize], &[Vec<f32>]),
        (b_shape, b_terms): (&[usize], &[Vec<f32>]),
        out_shape: &[usize],
    ) -> Vec<f32> {
        let (ra, rb, ro) = (a_shape.len(), b_shape.len(), out_shape.len());
        let k = a_shape[ra - 1];
        let (a_st, b_st) = (strides(a_shape), strides(b_shape));
        (0..out_shape.iter().product::<usize>())
            .map(|flat| {
                let at = unravel(flat, out_shape);
                let mut batch = at[..ro - 2].to_vec();
                batch.extend([0, 0]);
                let (a_base, b_base) = (
                    broadcast_flat(&batch, a_shape),
                    broadcast_flat(&batch, b_shape),
                );
                let mut acc = 0.0f32;
                for ki in 0..k {
                    for a in a_terms {
                        for b in b_terms {
                            let ai = a_base + at[ro - 2] * a_st[ra - 2] + ki * a_st[ra - 1];
                            let bi = b_base + ki * b_st[rb - 2] + at[ro - 1] * b_st[rb - 1];
                            acc += a[ai] * b[bi];
                        }
                    }
                }
                acc
            })
            .collect()
    }

    /// The bits of `value`, with every NaN as one: IEEE leaves a NaN's sign and payload to the hardware (an
    /// operand-order choice the optimizer may make differently in two compilations of the same sum), and nothing
    /// reads them. Every other value, signed zeros, infinities and subnormals included, compares exactly.
    fn identity_bits(value: f32) -> u32 {
        if value.is_nan() {
            f32::NAN.to_bits()
        } else {
            value.to_bits()
        }
    }

    fn on_pool(
        threads: usize,
        (a_shape, a_terms): (&[usize], &[Vec<f32>]),
        (b_shape, b_terms): (&[usize], &[Vec<f32>]),
        out_shape: &[usize],
    ) -> Vec<f32> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let a_reads: Vec<_> = a_terms.iter().map(|t| move |flat: usize| t[flat]).collect();
        let b_reads: Vec<_> = b_terms.iter().map(|t| move |flat: usize| t[flat]).collect();
        let mut data = vec![0.0f32; out_shape.iter().product()];
        matmul_accumulate_on(
            &pool,
            (
                a_shape,
                &a_reads.iter().map(|r| r as &TermReader).collect::<Vec<_>>(),
            ),
            (
                b_shape,
                &b_reads.iter().map(|r| r as &TermReader).collect::<Vec<_>>(),
            ),
            out_shape,
            &mut data,
        );
        data
    }

    /// The row-parallel matmul is bit-identical to the serial definition on every operand class, shape (plain,
    /// batched, a broadcast batch, one row, one column, `k` of one and of zero) and thread count, with one term
    /// per operand and with split operands of two.
    #[test]
    fn parallel_matmul_is_bit_identical_to_the_serial_loop() {
        let shapes: [(&[usize], &[usize], &[usize]); 7] = [
            (&[1, 1], &[1, 1], &[1, 1]),
            (&[5, 7], &[7, 3], &[5, 3]),
            (&[1, 64], &[64, 33], &[1, 33]),
            (&[17, 5], &[5, 1], &[17, 1]),
            (&[3, 4, 6], &[6, 5], &[3, 4, 5]),
            (&[3, 4, 6], &[2, 1, 6, 5], &[2, 3, 4, 5]),
            (&[4, 0], &[0, 3], &[4, 3]),
        ];
        for (case, (a_shape, b_shape, out_shape)) in shapes.into_iter().enumerate() {
            let (a_len, b_len) = (a_shape.iter().product(), b_shape.iter().product());
            for special in [false, true] {
                for terms in [1u64, 2] {
                    let seed = 100 * case as u64 + 10 * u64::from(special) + terms;
                    let a: Vec<_> = (0..terms)
                        .map(|t| operand(a_len, seed + 7 * t, special))
                        .collect();
                    let b: Vec<_> = (0..terms)
                        .map(|t| operand(b_len, seed + 31 * t + 1, special))
                        .collect();
                    let want = serial((a_shape, &a), (b_shape, &b), out_shape);
                    for threads in [1, 2, 3, 8] {
                        let got = on_pool(threads, (a_shape, &a), (b_shape, &b), out_shape);
                        let (got, want): (Vec<_>, Vec<_>) = (
                            got.iter().copied().map(identity_bits).collect(),
                            want.iter().copied().map(identity_bits).collect(),
                        );
                        assert_eq!(
                            got, want,
                            "{a_shape:?} x {b_shape:?}, special={special}, {terms} term(s), {threads} thread(s)"
                        );
                    }
                }
            }
        }
    }

    /// The test inputs reach the cases the equality guards: the special operands produce NaN and infinite
    /// outputs and the plain ones finite values, so a comparison of all-NaN outputs cannot pass vacuously.
    #[test]
    fn matmul_test_operands_reach_nan_infinity_and_finite_outputs() {
        let (a_shape, b_shape, out_shape): (&[usize], &[usize], &[usize]) =
            (&[5, 7], &[7, 3], &[5, 3]);
        let special = serial(
            (a_shape, &[operand(35, 1, true)]),
            (b_shape, &[operand(21, 2, true)]),
            out_shape,
        );
        assert!(special.iter().any(|v| v.is_nan()), "{special:?}");
        let plain = serial(
            (a_shape, &[operand(35, 1, false)]),
            (b_shape, &[operand(21, 2, false)]),
            out_shape,
        );
        assert!(plain.iter().all(|v| v.is_finite()), "{plain:?}");
    }

    /// ADR-0101 decision 4 (R470-007): the reference `MatMul` computes every product. A zero activation times
    /// an Inf weight is NaN (IEEE), so the output is NaN; skipping the zero activation hid it as 0.
    #[test]
    fn matmul_zero_activation_times_inf_weight_is_nan() {
        let a = HostTensor::f32(vec![1, 2], vec![0.0, 1.0]);
        let b = HostTensor::f32(vec![2, 2], vec![f32::INFINITY, 2.0, 3.0, 4.0]);
        let out = matmul(&a, &b, &[1, 2], OperandPrecision::Exact).unwrap();
        let data = out.as_f32().unwrap();
        assert!(
            data[0].is_nan(),
            "0 * inf + 1 * 3 must be NaN, got {}",
            data[0]
        );
        assert_eq!(data[1], 4.0, "the finite column is 0 * 2 + 1 * 4");
    }

    /// A zero activation times a NaN weight is NaN too, not a silent 0.
    #[test]
    fn matmul_zero_activation_times_nan_weight_is_nan() {
        let a = HostTensor::f32(vec![1, 1], vec![0.0]);
        let b = HostTensor::f32(vec![1, 1], vec![f32::NAN]);
        let out = matmul(&a, &b, &[1, 1], OperandPrecision::Exact).unwrap();
        let data = out.as_f32().unwrap();
        assert!(data[0].is_nan(), "0 * NaN must be NaN, got {}", data[0]);
    }
}
