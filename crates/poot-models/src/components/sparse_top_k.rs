//! Top-k selection by pairwise rank, for attention-side selectors (DeepSeek sparse attention, the
//! compressed-attention indexers). The router's selection is
//! [`poot_graph_ir::ops::stable_descending_rank`], which breaks ties by index so exactly `k`
//! distinct entries are kept; this rank counts strictly greater entries, so exact ties share a rank
//! and may keep more than `k`. That is the established selector behavior of these attention
//! families, kept distinct from the router's.

use crate::model::TraceError;
use crate::model::last_axis;
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced};

/// Pairwise-`Ge` rank along the trailing axis of `x` (shape `[.., N]`): `rank[i]` is the number of
/// positions strictly greater than `i`, so rank `0` is the maximum. Exact ties share a rank.
pub(crate) fn pairwise_rank(b: &Builder, x: Traced) -> Result<Traced, TraceError> {
    let shape = b.aval(x)?.shape;
    let n = last_axis("pairwise_rank", &shape)?;
    let last = shape.len() - 1;

    let mut pair_shape = shape.clone();
    pair_shape.push(n);
    let mut p_shape = shape.clone();
    p_shape.push(1);
    let p = b.broadcast(b.reshape(x, p_shape)?, pair_shape.clone())?;
    let mut q_shape = shape.clone();
    q_shape[last] = 1;
    q_shape.push(n);
    let q = b.broadcast(b.reshape(x, q_shape)?, pair_shape)?;
    let ge = b.binary(BinOp::Ge, p, q)?;
    let gt = b.binary_scalar(BinOp::Mul, ge, Scalar::F32(-1.0))?;
    let gt = b.binary_scalar(BinOp::Add, gt, Scalar::F32(1.0))?;
    Ok(b.reduce(RedOp::Sum, gt, last + 1, false)?)
}

/// `1.0` where `rank < k` (kept), `0.0` otherwise: turns a [`pairwise_rank`] output into a keep mask.
pub(crate) fn keep_top_k_mask(b: &Builder, rank: Traced, k: usize) -> Result<Traced, TraceError> {
    let rge = b.binary_scalar(BinOp::Ge, rank, Scalar::F32(k as f32))?;
    let keep = b.binary_scalar(BinOp::Mul, rge, Scalar::F32(-1.0))?;
    Ok(b.binary_scalar(BinOp::Add, keep, Scalar::F32(1.0))?)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::Fx;

    const ROWS: usize = 3;
    const N: usize = 7;

    /// Rows with distinct values, exact ties and a negative run.
    fn input() -> Vec<f32> {
        vec![
            0.5, -1.0, 2.0, 0.25, 3.0, -0.5, 1.0, //
            1.0, 1.0, 0.0, 1.0, -2.0, 0.0, 2.0, //
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
        ]
    }

    fn run(x: &[f32], pick: impl Fn(&Builder, Traced) -> Result<Traced, TraceError>) -> Vec<f32> {
        let b = Builder::for_tests();
        let xs = b.constant("x", TensorType::f32(vec![ROWS, N])).unwrap();
        let out = pick(&b, xs).unwrap();
        Fx::new(1)
            .eval(&b.finish(out).unwrap(), &[("x", x)], &[], &[])
            .0
    }

    /// The rank of each entry written from its definition: how many entries of the row are strictly
    /// greater.
    fn reference_rank(x: &[f32]) -> Vec<f32> {
        x.chunks(N)
            .flat_map(|row| (0..N).map(|i| row.iter().filter(|&&v| v > row[i]).count() as f32))
            .collect()
    }

    /// SC-002: the rank of every entry is the count of strictly greater entries (ties share a
    /// rank), and the keep mask of rank `< k` keeps exactly the entries a hand-written selection
    /// keeps: `k` of a distinct row, every tied entry of an all-equal row. Mutation: an off-by-one
    /// in `keep_top_k_mask` (`>= k + 1`); the kept counts change and this row goes red.
    #[test]
    fn rank_counts_strictly_greater_entries_and_the_mask_keeps_rank_below_k() {
        let x = input();
        assert_eq!(run(&x, pairwise_rank), reference_rank(&x));
        for k in 1..=N {
            let got = run(&x, |b, xs| keep_top_k_mask(b, pairwise_rank(b, xs)?, k));
            let want: Vec<f32> = reference_rank(&x)
                .into_iter()
                .map(|r| if r < k as f32 { 1.0 } else { 0.0 })
                .collect();
            assert_eq!(got, want, "k = {k}");
        }
        // The distinct row keeps exactly k; the all-equal row keeps all of them at any k.
        let two = run(&x, |b, xs| keep_top_k_mask(b, pairwise_rank(b, xs)?, 2));
        assert_eq!(two[..N].iter().sum::<f32>(), 2.0);
        assert_eq!(two[2 * N..].iter().sum::<f32>(), N as f32);
    }
}
