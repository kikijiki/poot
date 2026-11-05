//! Contractions: `MatMul` (every contraction composite decomposes into it) and `IndexedMatMul`.
//!
//! Both read their operands as the equation's [`OperandPrecision`] says: each operand enters the products as one
//! term (exact, or narrowed once) or two (a split f32 operand), and a product is summed for every pair of terms,
//! one from each operand, into one f32 accumulator in `k` order.

use poot_graph_ir::{OperandPrecision, ValueId};
use poot_tensor::{DType, HostTensor};

use super::index_rule::index_at;
use super::movement::index_values;
use super::precision::operand_terms;
use crate::{EvalError, broadcast_flat, strides, unravel};

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

/// `IndexedMatMul`: `out[m, n] = sum_k x[m, k] * W[idx[m], k, n]`, each row contracting against its own
/// expert (the eager oracle for the generated indexed contraction). `x` and `w` are read through
/// [`HostTensor::to_f32`] (either may be BF16/F16); `idx` is F32 or I32 words.
pub(crate) fn indexed_matmul(
    x: &HostTensor,
    w: &HostTensor,
    idx: &HostTensor,
    out_shape: &[usize],
    eqn: ValueId,
    precision: OperandPrecision,
) -> Result<HostTensor, EvalError> {
    let (m, k, n) = (x.shape()[0], x.shape()[1], w.shape()[2]);
    let experts = w.shape()[0];
    let (x_view, w_view) = (x.to_f32()?, w.to_f32()?);
    let (x_terms, w_terms) = if precision.is_exact() {
        (vec![x_view.to_vec()], vec![w_view.to_vec()])
    } else {
        (
            operand_terms(&x_view, x.dtype(), precision),
            operand_terms(&w_view, w.dtype(), precision),
        )
    };
    let idx_values = index_values(idx)?;
    let idx_rows: Vec<usize> = idx_values
        .iter()
        .enumerate()
        .map(|(position, &value)| index_at(value, position, experts, eqn))
        .collect::<Result<_, _>>()?;
    let mut data = vec![0.0f32; m * n];
    for mm in 0..m {
        let ebase = idx_rows[mm] * k * n;
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                for xt in &x_terms {
                    for wt in &w_terms {
                        acc += xt[mm * k + kk] * wt[ebase + kk * n + nn];
                    }
                }
            }
            data[mm * n + nn] = acc;
        }
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
        &a_reads
            .iter()
            .map(|r| r as &dyn Fn(usize) -> f32)
            .collect::<Vec<_>>(),
        b_shape,
        &b_reads
            .iter()
            .map(|r| r as &dyn Fn(usize) -> f32)
            .collect::<Vec<_>>(),
        out_shape,
        data,
    );
}

/// The eager matmul definition. Each of `a_terms` and `b_terms` reads row-major logical elements of one term of an
/// operand shaped `a_shape` and `b_shape` (one term unless the operand is split); every `k` step adds the product of
/// each pair of terms, `a` terms outermost. `data` is the zeroed row-major output of `out_shape`.
pub(crate) fn matmul_accumulate(
    a_shape: &[usize],
    a_terms: &[&dyn Fn(usize) -> f32],
    b_shape: &[usize],
    b_terms: &[&dyn Fn(usize) -> f32],
    out_shape: &[usize],
    data: &mut [f32],
) {
    let (ra, rb) = (a_shape.len(), b_shape.len());
    let (m, k) = (a_shape[ra - 2], a_shape[ra - 1]);
    let n = b_shape[rb - 1];
    let a_st = strides(a_shape);
    let b_st = strides(b_shape);
    let out_st = strides(out_shape);
    let batch_rank = out_shape.len() - 2;
    // iterate over batch x m x n.
    let batch_count: usize = out_shape[..batch_rank].iter().product::<usize>().max(1);
    for batch in 0..batch_count {
        let batch_idx = if batch_rank == 0 {
            vec![]
        } else {
            unravel(batch, &out_shape[..batch_rank])
        };
        // base offsets into a and b for this batch (broadcast batch dims).
        let a_base = broadcast_flat(&extend_idx(&batch_idx, &[0, 0]), a_shape);
        let b_base = broadcast_flat(&extend_idx(&batch_idx, &[0, 0]), b_shape);
        // cache-friendly ikj order: accumulate each output row by scaling contiguous B rows. With the
        // common B layout [K,N] (stride N over K, 1 over N), the inner ni loop walks B contiguously.
        let out_batch_base: usize = batch_idx.iter().zip(&out_st).map(|(c, st)| c * st).sum();
        let (a_rs, a_cs) = (a_st[ra - 2], a_st[ra - 1]);
        let (b_rs, b_cs) = (b_st[rb - 2], b_st[rb - 1]);
        let out_rs = out_st[out_shape.len() - 2];
        let out_cs = out_st[out_shape.len() - 1];
        for mi in 0..m {
            let out_row = out_batch_base + mi * out_rs;
            for ki in 0..k {
                // Every product is computed: `0 * inf` and `0 * NaN` are NaN, never a skipped 0 (ADR-0101).
                let b_row = b_base + ki * b_rs;
                for a_at in a_terms {
                    let av = a_at(a_base + mi * a_rs + ki * a_cs);
                    for b_at in b_terms {
                        for ni in 0..n {
                            data[out_row + ni * out_cs] += av * b_at(b_row + ni * b_cs);
                        }
                    }
                }
            }
        }
    }
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
