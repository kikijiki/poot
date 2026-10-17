//! The definition of `OpKind::SelectedAttention`: dense attention under the mask a block selection makes.

use super::{
    AttentionKeys, BinOp, Builder, BuilderError, RedOp, Scalar, Traced, attention_over_keys,
};
use crate::types::DType;

/// The magnitude of the additive penalty on a position the selection does not read.
const DROPPED: f32 = 1e30;

/// `(indicator - 1) * 1e30` of an exact `0.0`/`1.0` indicator: zero where it is one, `-1e30` where it is zero.
fn penalty(b: &Builder, indicator: Traced) -> Result<Traced, BuilderError> {
    b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, indicator, Scalar::F32(1.0))?,
        Scalar::F32(DROPPED),
    )
}

/// Spread a per-block value `[B, 1, Lq, NB]` over the `ratio` positions of each block and cut it to the
/// first `lk` positions: `[B, 1, Lq, lk]`.
fn per_position(
    b: &Builder,
    blocks: Traced,
    ratio: usize,
    lk: usize,
) -> Result<Traced, BuilderError> {
    let shape = b.aval(blocks)?.shape;
    let &[rows, one, lq, nb] = shape.as_slice() else {
        unreachable!("per_position takes [B, 1, Lq, NB]")
    };
    let wide = b.broadcast(
        b.reshape(blocks, vec![rows, one, lq, nb, 1])?,
        vec![rows, one, lq, nb, ratio],
    )?;
    let flat = b.reshape(wide, vec![rows, one, lq, nb * ratio])?;
    if nb * ratio == lk {
        Ok(flat)
    } else {
        b.slice(flat, 3, 0, lk)
    }
}

/// The additive mask `[B, 1, Lq, Lk]` (`0.0` where a query reads the key, `-1e30` elsewhere) of the queries at
/// `pos` `[B, Lq]` I32 that selected the blocks `ids` `[B, 1, Lq, K]` I32 (`-1` is no block) of `ratio`
/// positions: a key is read when it is causal (`j <= pos`) and in a selected block or in the query's own
/// unfinished block (the one `pos` is in, when `pos` is not its last position).
fn selected_block_mask(
    b: &Builder,
    ids: Traced,
    pos: Traced,
    ratio: usize,
    lk: usize,
) -> Result<Traced, BuilderError> {
    let ids_shape = b.aval(ids)?.shape;
    let &[rows, one, lq, k] = ids_shape.as_slice() else {
        unreachable!("the typing rule admits only [B, 1, Lq, K] ids")
    };
    let nb = lk.div_ceil(ratio);

    // The selected blocks as a 0/1 indicator per block: some id equals the block's index.
    let id = b.broadcast(
        b.reshape(b.cast(ids, DType::F32)?, vec![rows, one, lq, k, 1])?,
        vec![rows, one, lq, k, nb],
    )?;
    let block = b.broadcast(
        b.reshape(b.iota(nb)?, vec![1, 1, 1, 1, nb])?,
        vec![rows, one, lq, k, nb],
    )?;
    let equal = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, id, block)?,
        b.binary(BinOp::Ge, block, id)?,
    )?;
    let selected = penalty(b, b.reduce(RedOp::Max, equal, 3, false)?)?;

    // The query's own unfinished block: it starts at or before the query and is not yet complete.
    let wide = vec![rows, one, lq, nb];
    let at = b.broadcast(
        b.reshape(b.cast(pos, DType::F32)?, vec![rows, one, lq, 1])?,
        wide.clone(),
    )?;
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(nb)?, Scalar::F32(ratio as f32))?,
            vec![1, 1, 1, nb],
        )?,
        wide,
    )?;
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32))?;
    let complete = b.binary(BinOp::Ge, at, last)?;
    let started = b.binary(BinOp::Ge, at, first)?;
    let tail = penalty(b, b.binary(BinOp::Sub, started, complete)?)?;

    let read = b.binary(
        BinOp::Max,
        per_position(b, selected, ratio, lk)?,
        per_position(b, tail, ratio, lk)?,
    )?;

    // Causal: the key is at or before the query.
    let key = b.broadcast(
        b.reshape(b.iota(lk)?, vec![1, 1, 1, lk])?,
        vec![rows, one, lq, lk],
    )?;
    let query = b.broadcast(
        b.reshape(b.cast(pos, DType::F32)?, vec![rows, one, lq, 1])?,
        vec![rows, one, lq, lk],
    )?;
    let causal = penalty(b, b.binary(BinOp::Ge, query, key)?)?;
    b.binary(BinOp::Add, read, causal)
}

/// The output `[B, Hq, Lq, Dv]` of `OpKind::SelectedAttention` over `q`, `k`, `v`, `ids` and `pos`: dense attention
/// under [`selected_block_mask`]. Quadratic in nothing but `Lk`, so a long context costs the oracle a dense
/// attention and one `[B, Lq, K, blocks]` indicator.
#[allow(clippy::too_many_arguments)]
pub fn selected_attention(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    ids: Traced,
    pos: Traced,
    n_rep: usize,
    scale: f32,
    softcap: Option<f32>,
    ratio: usize,
) -> Result<Traced, BuilderError> {
    let lk = b.aval(k)?.shape[2];
    let mask = selected_block_mask(b, ids, pos, ratio, lk)?;
    attention_over_keys(
        b,
        q,
        AttentionKeys::Single(k),
        v,
        n_rep,
        scale,
        mask,
        softcap,
        None,
    )
}
