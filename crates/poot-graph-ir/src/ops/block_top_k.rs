//! The definition of `OpKind::BlockTopK`: the per-query block selection of a sparse-attention indexer.

use super::{
    BinOp, Builder, BuilderError, RedOp, Scalar, Traced, stable_descending_rank,
    stable_descending_rank_masked, top_k_keep_mask,
};
use crate::types::DType;

/// The packed result `[R, 1, T, K + NB]` I32 of [`crate::OpKind::BlockTopK`] over `scores` `[R, 1, T, NB]` and
/// the queries' positions `pos` `[R, T]`, in primitive operations.
///
/// Block `b` is eligible for the query at `p` when `b * ratio + ratio - 1 <= p`. The eligible blocks
/// rank by canonical score descending, lower block first on equal scores
/// ([`stable_descending_rank_masked`]), and the first `k` of them are kept. The kept blocks are then
/// ranked again by block id ascending (ineligible and dropped blocks after them), and
/// [`crate::OpKind::ArgTopK`] reads the first `k` of that order: the ids, with `-1` past the number kept.
/// Both ranks are pairwise over the `NB` blocks, so this is the quadratic oracle definition, not a
/// schedule.
pub fn block_top_k_packed(
    b: &Builder,
    scores: Traced,
    pos: Traced,
    ratio: usize,
    k: usize,
) -> Result<Traced, BuilderError> {
    let shape = b.aval(scores)?.shape;
    let &[rows, one, tokens, blocks] = shape.as_slice() else {
        unreachable!("the typing rule admits only [R, 1, T, NB] scores")
    };
    let wide = vec![rows, one, tokens, blocks];

    // Complete as of the query: the block's last position is at or before it. Exact `0.0`/`1.0`.
    let pos = b.broadcast(
        b.reshape(b.cast(pos, DType::F32)?, vec![rows, one, tokens, 1])?,
        wide.clone(),
    )?;
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(blocks)?, Scalar::F32(ratio as f32))?,
            vec![1, 1, 1, blocks],
        )?,
        wide,
    )?;
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32))?;
    let eligible = b.binary(BinOp::Ge, pos, last)?;

    let rank = stable_descending_rank_masked(b, scores, eligible)?;
    let keep = b.binary(BinOp::Mul, top_k_keep_mask(b, rank, k)?, eligible)?;

    // Kept blocks by id ascending, then the rest: rank the negated sort key `id + NB * (1 - keep)`.
    let id = b.broadcast(
        b.reshape(b.iota(blocks)?, vec![1, 1, 1, blocks])?,
        b.aval(keep)?.shape,
    )?;
    let dropped = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0))?,
        Scalar::F32(-(blocks as f32)),
    )?;
    let order = stable_descending_rank(
        b,
        b.binary_scalar(
            BinOp::Mul,
            b.binary(BinOp::Add, id, dropped)?,
            Scalar::F32(-1.0),
        )?,
    )?;
    let ids = b.arg_top_k(order, k)?; // [R, 1, T, K] I32

    // Slot `j` holds a block when at least `j + 1` blocks were kept, else `-1`.
    let kept = b.reduce(RedOp::Sum, keep, 3, true)?;
    let slot = b.broadcast(
        b.reshape(b.iota(k)?, vec![1, 1, 1, k])?,
        vec![rows, one, tokens, k],
    )?;
    let filled = b.cast(
        b.binary(
            BinOp::Ge,
            b.broadcast(kept, vec![rows, one, tokens, k])?,
            b.binary_scalar(BinOp::Add, slot, Scalar::F32(1.0))?,
        )?,
        DType::I32,
    )?;
    let none = b.broadcast(
        b.reshape(b.iota_i32(1)?, vec![1, 1, 1, 1])?,
        vec![rows, one, tokens, k],
    )?;
    let ids = b.select(
        filled,
        ids,
        b.binary_scalar(BinOp::Add, none, Scalar::I32(-1))?,
    )?;
    b.concat(3, &[ids, b.cast(keep, DType::I32)?])
}
