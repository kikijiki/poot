//! The pass-only composites as ops in a test graph: the tests that exercise `OpKind::BlockTopK` and
//! `OpKind::SelectedAttention` themselves (typing, decomposition, the oracle's bounds, the planner's
//! refusals) need an equation holding the op.
//!
//! A tracer states a selection as its primitive composition ([`crate::ops::block_top_k_packed`],
//! [`crate::ops::selected_attention`]) and `compile` forms the op (`poot-graph-plan`'s selection pass,
//! architecture rule 3), so [`crate::Builder`] has no constructor for either: these functions live here, beside
//! the other fixtures no production code calls, and a tracer cannot reach them (the `compile_fail` examples on
//! [`crate::Builder`]).

use crate::builder::{Builder, Traced};
use crate::error::BuilderError;
use crate::op::{BlockTopKAttrs, OpKind, SelectedAttentionAttrs};

/// The values of a block selection built by [`block_top_k`]: the packed result its one equation defines and the
/// two projections of it.
#[derive(Clone, Copy, Debug)]
pub struct BlockTopKResult {
    /// The packed I32 `[R, 1, T, K + NB]` result.
    pub packed: Traced,
    /// The selected block ids `[R, 1, T, K]`, ascending, `-1` after the last selected.
    pub ids: Traced,
    /// The `0`/`1` keep indicator of each block `[R, 1, T, NB]`.
    pub keep: Traced,
}

/// ONE `OpKind::BlockTopK` equation over `scores` `[R, 1, T, NB]` and the queries' positions `pos` `[R, T]`,
/// returned as its typed values. The two projections are slices of the packed result at the columns its
/// [`crate::BlockTopKLayout`] names.
pub fn block_top_k(
    b: &Builder,
    attrs: BlockTopKAttrs,
    scores: Traced,
    pos: Traced,
) -> Result<BlockTopKResult, BuilderError> {
    let layout = attrs
        .layout(&[b.aval(scores)?, b.aval(pos)?])
        .map_err(|source| BuilderError::Inference {
            equation: b.equation_count(),
            operation: OpKind::BlockTopK(attrs).name(),
            source,
        })?;
    let packed = b.emit(
        OpKind::BlockTopK(attrs),
        vec![b.operand(scores)?, b.operand(pos)?],
    )?;
    let (ids, keep) = (layout.ids(), layout.keep());
    Ok(BlockTopKResult {
        packed,
        ids: b.slice(packed, 3, ids.start, ids.end)?,
        keep: b.slice(packed, 3, keep.start, keep.end)?,
    })
}

/// ONE `OpKind::SelectedAttention` equation over `q`, the cached `k` and `v`, the selected block `ids`
/// `[B, 1, Lq, K]` and the queries' positions `pos` `[B, Lq]`. Output `[B, Hq, Lq, Dv]`.
pub fn selected_attention(
    b: &Builder,
    attrs: SelectedAttentionAttrs,
    [q, k, v, ids, pos]: [Traced; 5],
) -> Result<Traced, BuilderError> {
    b.emit(
        OpKind::SelectedAttention(attrs),
        vec![
            b.operand(q)?,
            b.operand(k)?,
            b.operand(v)?,
            b.operand(ids)?,
            b.operand(pos)?,
        ],
    )
}
