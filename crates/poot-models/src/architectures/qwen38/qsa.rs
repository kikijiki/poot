//! Qwen Sparse Attention (QSA): the block indexer that narrows which cached positions a
//! full-attention layer reads (`Qwen4ExpTextQSAIndexer`). The layer itself is the ordinary gated GQA
//! attention (`components::attention` with the output gate); the indexer only chooses which blocks of
//! cached positions it reads ([`qsa_select`] returns their ids, and the attention reads those blocks and the
//! query's own unfinished block).
//!
//! The indexer scores *blocks* of `ratio` consecutive positions and caches the blocks' keys, not the
//! positions':
//!
//! 1. The layer's normed input projects to a per-head query `q [heads, dim]` and one shared raw key
//!    `k [dim]`. `q` gets a norm and partial RoPE at its own position.
//! 2. The raw keys of a block are mean-pooled, and only then does the pooled key get its own norm and
//!    partial RoPE at the block's first position: pool, then norm, then rope.
//! 3. Query `t` scores block `b` as `sum_h ReLU(q_{t,h} . k_b) / sqrt(dim)`, an unweighted sum over
//!    the heads.
//! 4. Per query, the best `budget / ratio` blocks stay visible, ties going to the lower block (the
//!    budget is exact even on equal scores). A block is eligible once every position of it is at or
//!    before the query. The query's own unfinished block is always visible, unscored. The selection is
//!    the primitive composition [`ops::block_top_k_packed`]; `compile` forms `OpKind::BlockTopK` from it
//!    (a linear radix select on a device, a quadratic rank on the oracle).
//!
//! A completed block's key is a pure function of its `ratio` raw keys and its first position, so it
//! never changes and is computed once. Each layer carries two states: the block keys
//! (`kv.l{layer}.index_blocks`, [`StateRole::Blocked`]) and the raw keys of the unfinished block
//! (`kv.l{layer}.index_tail`, `ratio` slots, slot `j` holding the latest position `= j mod ratio`).
//! A step pools only the blocks it touches (`ceil((tokens + ratio - 1) / ratio)` of them, a trace-time
//! bound), from the tail slots followed by its own raw keys, and writes them at the block index of
//! its first position. A touched block the step leaves unfinished receives a partial key that the
//! eligibility mask hides until a later step rewrites it complete. There is no `if block complete`,
//! so one graph serves a prompt, a chunk and one token.
//!
//! A row may rewind (a rejected speculative draft, a re-run) only to a block boundary or to a position
//! inside the block of the furthest position it has written. Slot `j` of the tail keeps the latest
//! position of its residue, so a rewind into an earlier block than the furthest one, away from a
//! boundary, finds the later block's keys in the slots below the rewind point and pools them into
//! the rewound block. Rewinding to a boundary reads no tail slot, and complete blocks are never touched.
//!
//! Every quantity of the selection is a function of the query's position, so the selection is built in
//! the graph from the step's `Slot::Pos` and one graph serves a prompt, a chunk and one token.

use std::num::NonZeroU32;

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, StateRole, TensorType, Traced, ops};
use poot_quant::weights::{QsaRole, WeightId, WeightMap, WeightRole};

use crate::components::attention::{BlockSelection, first_writes};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormParams, norm};
use crate::components::rope::{RopeParams, apply_rope, rope_rows};
use crate::components::standard::{ParamError, Step};
use crate::model::{ConfigReason, TraceError};

/// Checked QSA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct QsaParams {
    heads: usize,
    dim: usize,
    budget: usize,
    ratio: usize,
    /// `ratio`, as the [`StateRole::Blocked`] granule of the block-key cache.
    granule: NonZeroU32,
    /// `log2(ratio)`: a position's block is `position >> shift`, its place in the block `position & (ratio - 1)`.
    shift: u32,
    rope: RopeParams,
    norm: NormParams,
}

impl QsaParams {
    /// `heads` indexer heads of `dim` width scoring blocks of `ratio` positions, keeping
    /// `budget / ratio` blocks per query. `rope` rotates the leading dims of each head and key (no
    /// wider than `dim`); `norm` is the query and key norm.
    pub(super) fn new(
        heads: usize,
        dim: usize,
        budget: usize,
        ratio: usize,
        rope: RopeParams,
        norm: NormParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("indexer_n_heads", heads),
            ("indexer_head_dim", dim),
            ("indexer_budget", budget),
            ("indexer_compress_ratio", ratio),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        // A budget below one block would keep no block at all.
        if budget < ratio {
            return Err(ParamError::new("indexer_budget", ConfigReason::Unsupported));
        }
        if rope.rotary() > dim {
            return Err(ParamError::new(
                "indexer_head_dim",
                ConfigReason::Exceeds { max: rope.rotary() },
            ));
        }
        // The block of a position is found with a shift and a mask: no device lowers an integer division.
        if !ratio.is_power_of_two() {
            return Err(ParamError::new(
                "indexer_compress_ratio",
                ConfigReason::Unsupported,
            ));
        }
        let granule = u32::try_from(ratio)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| {
                ParamError::new(
                    "indexer_compress_ratio",
                    ConfigReason::Exceeds {
                        max: u32::MAX as usize,
                    },
                )
            })?;
        Ok(Self {
            heads,
            dim,
            budget,
            ratio,
            granule,
            shift: ratio.trailing_zeros(),
            rope,
            norm,
        })
    }

    /// The indexer head width.
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// The rows of the fused query and key projection that are the query's.
    pub(super) fn query_rows(&self) -> usize {
        self.heads * self.dim
    }

    /// The blocks each query keeps.
    fn block_topk(&self) -> usize {
        self.budget / self.ratio
    }
}

/// A layer's indexer weights: the query projection `[heads * dim, width]`, the shared key
/// projection `[dim, width]` and the scales of the query and key norms `[dim]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct QsaWeights {
    q: Weight,
    k: Weight,
    q_norm: Weight,
    k_norm: Weight,
}

impl QsaWeights {
    pub(super) fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &QsaParams,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Qsa(role)), shape)
        };
        Ok(Self {
            q: w(QsaRole::Q, &[p.query_rows(), width])?,
            k: w(QsaRole::K, &[p.dim, width])?,
            q_norm: w(QsaRole::QNorm, &[p.dim])?,
            k_norm: w(QsaRole::KNorm, &[p.dim])?,
        })
    }
}

/// The blocks a step of `tokens` positions can touch: its first position may sit anywhere in a
/// block, so `tokens + ratio - 1` positions span `ceil((tokens + ratio - 1) / ratio)` blocks.
fn touched_blocks(tokens: usize, ratio: usize) -> usize {
    (tokens + 2 * ratio - 2) / ratio
}

/// A rank-1 I32 `[len]` vector holding the scalar `value`.
fn spread(b: &Builder, value: Traced, len: usize) -> Result<Traced, TraceError> {
    Ok(b.broadcast(b.reshape(value, vec![1])?, vec![len])?)
}

/// `layer`'s block keys `[rows, 1, blocks, dim]` as of the end of this step: the cache carried from
/// the last step with the blocks this step touches pooled, normed, rotated and written at the block
/// index of each row's first position. `x` is the layer's normed input `[rows, tokens, width]`.
///
/// Per row, the touched blocks pool a span of `touched * ratio` raw keys that starts at the block
/// boundary at or before the row's first position `pos`: the tail slots below `pos mod ratio`, then
/// this step's keys, then zeros (positions the step has not reached, in a block that stays unfinished).
/// The tail then takes, per slot, the latest position of its residue this step reached.
fn block_keys(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &QsaWeights,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (dim, ratio) = (p.dim, p.ratio);
    let nb = cap / ratio;
    debug_assert_eq!(nb * ratio, cap, "the model checks the capacity granule");
    // A step that fits the capacity touches at most `nb` blocks.
    let touched = touched_blocks(t, ratio).min(nb);
    let span = touched * ratio;
    let ratio_i32 = ratio as i32;

    let keys_in = b.state_input(
        &format!("kv.l{layer}.index_blocks"),
        TensorType::f32(vec![rows, 1, nb, dim]),
        StateRole::Blocked {
            axis: 2,
            ratio: p.granule,
        },
    )?;
    let tail_in = b.state_input(
        &format!("kv.l{layer}.index_tail"),
        TensorType::f32(vec![rows, 1, ratio, dim]),
        StateRole::Recurrent,
    )?;

    let raw = b.reshape(linear(b, x, &w.k, None)?, vec![rows, 1, t, dim])?;
    let zeros = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(1)?, Scalar::F32(0.0))?,
            vec![1, 1, 1, 1],
        )?,
        vec![1, 1, span, dim],
    )?;
    let slot = b.iota_i32(ratio)?;
    let offset_in_span = b.iota_i32(span)?;
    let touched_ids = b.iota_i32(touched)?;
    let rope_capacity = cap + span;

    let mut keys_out = Vec::with_capacity(rows);
    let mut tail_out = Vec::with_capacity(rows);
    for (row, &first) in first_writes(b, step)?.iter().enumerate() {
        let part = |value: Traced| -> Result<Traced, TraceError> {
            Ok(if rows == 1 {
                value
            } else {
                b.slice(value, 0, row, row + 1)?
            })
        };
        // Where the row's first position falls in its block, and the index of that block.
        let skew = b.binary_scalar(BinOp::And, first, Scalar::I32(ratio_i32 - 1))?;
        let first_block = b.binary_scalar(BinOp::Shr, first, Scalar::I32(p.shift as i32))?;

        // The raw keys the span pools. Span offset `i` below the skew reads tail slot `i`; at or past it
        // reads this step's key `i - skew`, which sits at `ratio + i - skew` after the tail.
        let sources = b.concat(2, &[part(tail_in)?, part(raw)?, zeros])?;
        let skew_wide = spread(b, skew, span)?;
        let in_tail = b.binary(
            BinOp::Ge,
            skew_wide,
            b.binary_scalar(BinOp::Add, offset_in_span, Scalar::I32(1))?,
        )?;
        let from = b.binary(
            BinOp::Add,
            offset_in_span,
            b.binary(
                BinOp::Mul,
                b.binary_scalar(BinOp::Sub, skew_wide, Scalar::I32(ratio_i32))?,
                b.binary_scalar(BinOp::Sub, in_tail, Scalar::I32(1))?,
            )?,
        )?;
        let span_keys = b.gather(sources, 2, from)?; // [1, 1, span, dim]

        // Pool each touched block, then norm it, then rotate it at the block's first position.
        let summed = b.reduce(
            RedOp::Sum,
            b.reshape(span_keys, vec![1, 1, touched, ratio, dim])?,
            3,
            true,
        )?;
        let pooled = b.reshape(
            b.binary_scalar(BinOp::Mul, summed, Scalar::F32(1.0 / ratio as f32))?,
            vec![1, 1, touched, dim],
        )?;
        let block_pos = b.reshape(
            b.binary_scalar(
                BinOp::Mul,
                b.binary(BinOp::Add, spread(b, first_block, touched)?, touched_ids)?,
                Scalar::I32(ratio_i32),
            )?,
            vec![1, touched],
        )?;
        let block_k = apply_rope(
            b,
            norm(b, pooled, &w.k_norm, p.norm)?,
            &rope_rows(b, p.rope, block_pos, rope_capacity)?,
        )?; // [1, 1, touched, dim]
        // The write is `touched` blocks long, so a step that ends at the capacity slides it left to stay
        // in bounds; the blocks it then covers before `first_block` are copied unchanged.
        let start = b.binary(
            BinOp::Sub,
            first_block,
            b.binary_scalar(
                BinOp::Max,
                b.binary_scalar(BinOp::Sub, first_block, Scalar::I32((nb - touched) as i32))?,
                Scalar::I32(0),
            )?,
        )?;
        let blocks = b.binary(BinOp::Add, spread(b, start, touched)?, touched_ids)?;
        let ahead = b.binary(BinOp::Sub, blocks, spread(b, first_block, touched)?)?;
        let is_new = b.broadcast(
            b.reshape(
                b.binary_scalar(BinOp::Ge, ahead, Scalar::I32(0))?,
                vec![1, 1, touched, 1],
            )?,
            vec![1, 1, touched, dim],
        )?;
        let cache = part(keys_in)?;
        let written = b.select(
            is_new,
            b.gather(
                block_k,
                2,
                b.binary_scalar(BinOp::Max, ahead, Scalar::I32(0))?,
            )?,
            b.gather(cache, 2, blocks)?,
        )?;
        keys_out.push(b.dynamic_update_slice_dyn(cache, written, start, 2)?);

        // Tail slot `j` takes the latest position of residue `j` below the step's end, if the step
        // wrote one (`reached >= 0`: its key sits `reached` after the step's first); else it keeps its key.
        let end_residue = b.binary_scalar(
            BinOp::And,
            b.binary_scalar(BinOp::Add, skew, Scalar::I32((t - 1) as i32))?,
            Scalar::I32(ratio_i32 - 1),
        )?;
        let behind = b.binary_scalar(
            BinOp::And,
            b.binary(
                BinOp::Sub,
                b.binary_scalar(
                    BinOp::Add,
                    spread(b, end_residue, ratio)?,
                    Scalar::I32(ratio_i32),
                )?,
                slot,
            )?,
            Scalar::I32(ratio_i32 - 1),
        )?;
        let reached = b.binary_scalar(
            BinOp::Mul,
            b.binary_scalar(BinOp::Sub, behind, Scalar::I32((t - 1) as i32))?,
            Scalar::I32(-1),
        )?;
        let wrote = b.binary_scalar(BinOp::Ge, reached, Scalar::I32(0))?;
        let tail_from = b.binary(
            BinOp::Add,
            slot,
            b.binary(
                BinOp::Mul,
                wrote,
                b.binary(
                    BinOp::Sub,
                    b.binary_scalar(BinOp::Add, reached, Scalar::I32(ratio_i32))?,
                    slot,
                )?,
            )?,
        )?;
        tail_out.push(b.gather(sources, 2, tail_from)?); // [1, 1, ratio, dim]
    }
    let joined = |parts: Vec<Traced>| -> Result<Traced, TraceError> {
        Ok(if rows == 1 {
            parts[0]
        } else {
            b.concat(0, &parts)?
        })
    };
    let (keys_out, tail_out) = (joined(keys_out)?, joined(tail_out)?);
    step.carry(keys_in, keys_out);
    step.carry(tail_in, tail_out);
    Ok(keys_out)
}

/// `layer`'s block selection: the blocks of cached positions each query of the step reads, as the ids half of a
/// [`ops::block_top_k_packed`] selection (the attention then reads those blocks and the query's own unfinished block). `x`
/// is the layer's normed input `[rows, tokens, width]`; the layer's block keys and unfinished-block raw keys are
/// carried by `step` ([`block_keys`]). The step's capacity must be a multiple of the block `ratio`.
pub(super) fn qsa_select(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &QsaWeights,
    p: &QsaParams,
) -> Result<BlockSelection, TraceError> {
    let q = query(b, step, x, w, p)?;
    let block_k = block_keys(b, step, layer, x, w, p)?; // [rows, 1, blocks, dim]
    let packed = select_blocks(b, step, q, block_k, p)?;
    Ok(BlockSelection {
        ids: b.slice(packed, 3, 0, kept_blocks(step, p))?,
        ratio: p.ratio,
    })
}

/// The indexer queries `[rows, heads, tokens, dim]`: a norm over each head, then RoPE at its own position.
fn query(
    b: &Builder,
    step: &Step,
    x: Traced,
    w: &QsaWeights,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let q = norm(
        b,
        b.reshape(linear(b, x, &w.q, None)?, vec![rows, t, p.heads, p.dim])?,
        &w.q_norm,
        p.norm,
    )?;
    apply_rope(
        b,
        b.transpose(q, vec![0, 2, 1, 3])?,
        &rope_rows(b, p.rope, step.pos(), cap)?,
    )
}

/// The score of every block for every query: `sum_h ReLU(q . k_block) / sqrt(dim)`, the head axis kept at
/// one: `[rows, 1, tokens, blocks]`.
fn block_scores(b: &Builder, q: Traced, block_k: Traced, dim: usize) -> Result<Traced, TraceError> {
    let dots = b.matmul(q, b.transpose(block_k, vec![0, 1, 3, 2])?)?; // [rows, heads, tokens, blocks]
    Ok(b.binary_scalar(
        BinOp::Mul,
        b.reduce(RedOp::Sum, ops::relu(b, dots)?, 1, true)?,
        Scalar::F32(1.0 / (dim as f32).sqrt()),
    )?)
}

/// The blocks each query keeps: the budget in blocks, or every block when the budget is above the block count.
fn kept_blocks(step: &Step, p: &QsaParams) -> usize {
    p.block_topk().min(step.shape().capacity.get() / p.ratio)
}

/// The budgeted selection of the blocks each query has completed, from the queries `q` and the block keys
/// `block_k` `[rows, 1, blocks, dim]`: scoring, then the best `budget / ratio` blocks, ties to the lower block
/// ([`ops::block_top_k_packed`]); a budget above the block count keeps every complete block. The packed I32
/// result `[rows, 1, tokens, K + blocks]`: the `K` ids, then each block's keep indicator.
fn select_blocks(
    b: &Builder,
    step: &Step,
    q: Traced,
    block_k: Traced,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    debug_assert_eq!(
        (step.shape().capacity.get() / p.ratio) * p.ratio,
        step.shape().capacity.get(),
        "the model checks the capacity granule"
    );
    let scores = block_scores(b, q, block_k, p.dim)?;
    Ok(ops::block_top_k_packed(
        b,
        scores,
        step.pos(),
        p.ratio,
        kept_blocks(step, p),
    )?)
}

#[cfg(test)]
mod tests;
