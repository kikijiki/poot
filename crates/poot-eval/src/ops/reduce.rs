//! `Reduce` over one axis (F32).

use poot_graph_ir::RedOp;

use poot_tensor::HostTensor;

use super::elementwise::max_number;
use crate::{EvalError, strides, unravel};

pub(crate) fn reduce(
    op: RedOp,
    x: &HostTensor,
    axis: usize,
    keepdim: bool,
) -> Result<HostTensor, EvalError> {
    let x_view = x.to_f32()?;
    let mut out_shape = x.shape().to_vec();
    let axis_len = out_shape[axis];
    if keepdim {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    let n: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; n];
    let x_st = strides(x.shape());
    for (flat, slot) in data.iter_mut().enumerate() {
        // build the source base index from the (reduced) out index.
        let out_idx = unravel(flat, &out_shape);
        let mut src = vec![0usize; x.shape().len()];
        let mut oi = 0;
        for (d, s) in src.iter_mut().enumerate() {
            if d == axis {
                *s = 0;
            } else if keepdim {
                // keepdim: out_shape has full rank (axis dim = 1), so out_idx is index-aligned with src.
                // (`oi` is only correct when the axis dim was removed from out_shape.)
                *s = out_idx[d];
            } else {
                *s = out_idx[oi];
                oi += 1;
            }
        }
        let base: usize = src.iter().zip(&x_st).map(|(c, st)| c * st).sum();
        let mut acc = match op {
            RedOp::Sum => 0.0,
            RedOp::Max => f32::NEG_INFINITY,
        };
        for a in 0..axis_len {
            let v = x_view[base + a * x_st[axis]];
            acc = match op {
                RedOp::Sum => acc + v,
                RedOp::Max => max_number(acc, v),
            };
        }
        *slot = acc;
    }
    Ok(HostTensor::f32(out_shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RedOp::Max` has `BinOp::Max`'s rule: NaN elements are skipped, and a row of only NaN is the identity, negative infinity.
    ///
    /// Mutation: use a propagating max in `max_number`; the first row comes back NaN and this goes red on
    /// `a NaN element must be skipped`.
    #[test]
    fn a_reduced_row_skips_nan_elements_and_only_an_all_nan_row_is_nan() {
        let x = HostTensor::f32(
            vec![3, 3],
            vec![
                1.0,
                f32::NAN,
                3.0,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                1.0,
                2.0,
                3.0,
            ],
        );
        let out = reduce(RedOp::Max, &x, 1, false).unwrap();
        let out = out.as_f32().unwrap();
        assert_eq!(out[0], 3.0, "a NaN element must be skipped, got {}", out[0]);
        assert_eq!(
            out[1],
            f32::NEG_INFINITY,
            "an all-NaN row is -inf, got {}",
            out[1]
        );
        assert_eq!(out[2], 3.0);
    }
}
