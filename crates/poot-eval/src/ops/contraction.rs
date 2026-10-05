//! Contractions: `MatMul`, `IndexedMatMul`, `DenseContraction` over an F32 or F16 weight,
//! and the F32-activation times BF16-weight `MatMul`/`DenseContraction` of the BF16 table.

use poot_graph_ir::ValueId;
use poot_tensor::HostTensor;

use super::index_rule::index_at;
use super::movement::index_values;
use crate::{EvalError, broadcast_flat, strides, unravel};

/// `MatMul`, whatever kernel the planner chose for it: an f32 accumulation over operands read
/// through [`HostTensor::to_f32`] (a BF16/F16 operand widens exactly). The walk stores the F32
/// result as the equation's declared dtype, so a BF16/F16 result is narrowed once, on store, as the
/// GPU kernels do.
pub(crate) fn matmul_to(
    a: &HostTensor,
    b: &HostTensor,
    out_shape: &[usize],
) -> Result<HostTensor, EvalError> {
    matmul(a, b, out_shape)
}

/// `DenseContraction` over an F32 or (Card 1007) F16 weight held in checkpoint `[N, K]` order:
/// `out[.., m, n] = sum over k of a[.., m, k] * w[n, k]`, exactly `MatMul(a, Transpose([1, 0], w))`, the
/// equation the fold replaces.
pub(crate) fn dense_contraction(
    a: &HostTensor,
    w: &HostTensor,
    out_shape: &[usize],
) -> Result<HostTensor, EvalError> {
    matmul(a, &super::movement::transpose(w, &[1, 0])?, out_shape)
}

/// `IndexedMatMul`: `out[m, n] = sum_k x[m, k] * W[idx[m], k, n]`, each row contracting against its own
/// expert (the eager oracle for `indexed_matmul_dt`). `x` and `w` are read through
/// [`HostTensor::to_f32`] (either may be BF16/F16); `idx` is F32 or I32 words.
pub(crate) fn indexed_matmul(
    x: &HostTensor,
    w: &HostTensor,
    idx: &HostTensor,
    out_shape: &[usize],
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let (m, k, n) = (x.shape()[0], x.shape()[1], w.shape()[2]);
    let experts = w.shape()[0];
    let (x_view, w_view) = (x.to_f32()?, w.to_f32()?);
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
                acc += x_view[mm * k + kk] * w_view[ebase + kk * n + nn];
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
) -> Result<HostTensor, EvalError> {
    let (av, bv) = (a.to_f32()?, b.to_f32()?);
    let mut data = vec![0.0f32; out_shape.iter().product()];
    matmul_accumulate(
        a.shape(),
        |flat| av[flat],
        b.shape(),
        |flat| bv[flat],
        out_shape,
        &mut data,
    );
    Ok(HostTensor::f32(out_shape.to_vec(), data))
}

/// The eager matmul definition. `a_at` and `b_at` read row-major logical elements of operands shaped `a_shape`
/// and `b_shape`; `data` is the zeroed row-major output of `out_shape`.
pub(crate) fn matmul_accumulate(
    a_shape: &[usize],
    a_at: impl Fn(usize) -> f32,
    b_shape: &[usize],
    b_at: impl Fn(usize) -> f32,
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
                let av = a_at(a_base + mi * a_rs + ki * a_cs);
                let b_row = b_base + ki * b_rs;
                for ni in 0..n {
                    data[out_row + ni * out_cs] += av * b_at(b_row + ni * b_cs);
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

    /// Card 645: the F32 `DenseContraction` oracle reads the weight in checkpoint `[N, K]` order, so
    /// `out[m, n] = sum over k of a[m, k] * w[n, k]`. The values are chosen so reading `w` as `[K, N]` cannot
    /// produce them.
    #[test]
    fn dense_contraction_contracts_the_last_axis_of_a_checkpoint_order_weight() {
        let a = HostTensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = HostTensor::f32(
            vec![4, 3],
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        );
        let out = dense_contraction(&a, &w, &[2, 4]).unwrap();
        assert_eq!(out.shape(), [2, 4]);
        assert_eq!(
            out.as_f32().unwrap().to_vec(),
            vec![4.0, 2.0, 12.0, 6.0, 10.0, 5.0, 30.0, 15.0]
        );
    }

    /// Card 1007: an F16 weight (binary16 words 0x3c00 = 1.0, 0x4000 = 2.0) contracts to the same values as
    /// its F32 twin above, read in checkpoint `[N, K]` order and widened exactly.
    #[test]
    fn dense_contraction_reads_an_f16_checkpoint_order_weight() {
        let a = HostTensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (one, two) = (0x3c00u16, 0x4000u16);
        let w = HostTensor::f16(
            vec![4, 3],
            vec![one, 0, one, 0, one, 0, two, two, two, one, one, one],
        );
        let out = dense_contraction(&a, &w, &[2, 4]).unwrap();
        assert_eq!(
            out.as_f32().unwrap().to_vec(),
            vec![4.0, 2.0, 12.0, 6.0, 10.0, 5.0, 30.0, 15.0]
        );
    }

    /// ADR-0101 decision 4 (R470-007): the reference `MatMul` computes every product. A zero activation times
    /// an Inf weight is NaN (IEEE), so the output is NaN; skipping the zero activation hid it as 0.
    #[test]
    fn matmul_zero_activation_times_inf_weight_is_nan() {
        let a = HostTensor::f32(vec![1, 2], vec![0.0, 1.0]);
        let b = HostTensor::f32(vec![2, 2], vec![f32::INFINITY, 2.0, 3.0, 4.0]);
        let out = matmul(&a, &b, &[1, 2]).unwrap();
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
        let out = matmul(&a, &b, &[1, 1]).unwrap();
        let data = out.as_f32().unwrap();
        assert!(data[0].is_nan(), "0 * NaN must be NaN, got {}", data[0]);
    }
}
