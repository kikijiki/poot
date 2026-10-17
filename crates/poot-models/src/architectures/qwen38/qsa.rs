//! Qwen Sparse Attention (QSA): the block indexer that narrows which cached positions a
//! full-attention layer reads (`Qwen4ExpTextQSAIndexer`). The layer itself is the ordinary gated GQA
//! attention (`components::attention` with the output gate); the indexer only replaces its
//! causal-over-cache mask by [`qsa_mask`].
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
//!    before the query. The query's own unfinished block is always visible, unscored.
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
//! Every quantity of the selection is a function of the query's position, so the mask is built in
//! the graph from the step's `Slot::Pos`.
//!
//! All masks are `{0, -1e30}` valued, so AND is a plain add (two penalties never overflow `f32`) and
//! OR is a max.

use poot_graph_ir::ops::{stable_descending_rank, top_k_keep_mask};
use std::num::NonZeroU32;

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, StateRole, TensorType, Traced, ops};
use poot_quant::weights::{QsaRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use crate::components::attention::first_writes;
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormParams, norm};
use crate::components::rope::{RopeParams, apply_rope, rope_rows};
use crate::components::standard::{ParamError, Step};
use crate::model::{ConfigReason, TraceError};

/// The additive penalty of a position the selection drops or has not made visible.
const DROPPED: f32 = -1e30;

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

/// `(indicator - 1) * 1e30`: zero where the `{0, 1}` indicator is one, [`DROPPED`] where it is zero.
fn penalty(b: &Builder, indicator: Traced) -> Result<Traced, TraceError> {
    Ok(b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, indicator, Scalar::F32(1.0))?,
        Scalar::F32(-DROPPED),
    )?)
}

/// Spread a per-block value `[rows, 1, tokens, blocks]` over the `ratio` positions of each block:
/// `[rows, 1, tokens, blocks * ratio]`.
fn per_position(b: &Builder, blocks: Traced, ratio: usize) -> Result<Traced, TraceError> {
    let shape = b.aval(blocks)?.shape;
    let [rows, one, t, nb] = shape[..] else {
        return Err(poot_graph_ir::BuilderError::Requirement {
            operation: "per_position",
            requirement: poot_graph_ir::Requirement::Rank {
                expected: 4,
                found: shape,
            },
        }
        .into());
    };
    let wide = b.broadcast(
        b.reshape(blocks, vec![rows, one, t, nb, 1])?,
        vec![rows, one, t, nb, ratio],
    )?;
    Ok(b.reshape(wide, vec![rows, one, t, nb * ratio])?)
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

/// The mask of `layer`'s sparse attention: `causal` (`[rows, 1, tokens, capacity]`) plus
/// [`DROPPED`] on every position outside each query's selection. `x` is the layer's normed input
/// `[rows, tokens, width]`; the layer's block keys and unfinished-block raw keys are carried by `step`
/// ([`block_keys`]). The step's capacity must be a multiple of the block `ratio`.
pub(super) fn qsa_mask(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    causal: Traced,
    w: &QsaWeights,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let q = query(b, step, x, w, p)?;
    let block_k = block_keys(b, step, layer, x, w, p)?; // [rows, 1, blocks, dim]
    selection_mask(b, step, q, block_k, causal, p)
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

/// `causal` plus [`DROPPED`] on every position outside each query's selection, from the queries `q`
/// and the block keys `block_k` `[rows, 1, blocks, dim]`: scoring, eligibility, the budgeted rank.
fn selection_mask(
    b: &Builder,
    step: &Step,
    q: Traced,
    block_k: Traced,
    causal: Traced,
    p: &QsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (dim, ratio) = (p.dim, p.ratio);
    let nb = cap / ratio;
    debug_assert_eq!(nb * ratio, cap, "the model checks the capacity granule");

    // Score: `sum_h ReLU(q . k_block) / sqrt(dim)`, the head axis kept at one.
    let dots = b.matmul(q, b.transpose(block_k, vec![0, 1, 3, 2])?)?; // [rows, heads, tokens, blocks]
    let scores = b.binary_scalar(
        BinOp::Mul,
        b.reduce(RedOp::Sum, ops::relu(b, dots)?, 1, true)?,
        Scalar::F32(1.0 / (dim as f32).sqrt()),
    )?; // [rows, 1, tokens, blocks]

    // Eligibility and the tail, from each query's position and each block's first position.
    let wide = vec![rows, 1, t, nb];
    let pos = b.broadcast(
        b.reshape(b.cast(step.pos(), DType::F32)?, vec![rows, 1, t, 1])?,
        wide.clone(),
    )?;
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(nb)?, Scalar::F32(ratio as f32))?,
            vec![1, 1, 1, nb],
        )?,
        wide.clone(),
    )?;
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32))?;
    // Complete as of the query: its last position is at or before the query.
    let complete = b.binary(BinOp::Ge, pos, last)?;
    let eligible = penalty(b, complete)?;
    // The query's own unfinished block: it starts at or before the query, and is not yet complete.
    let started = b.binary(BinOp::Ge, pos, first)?;
    let tail = penalty(b, b.binary(BinOp::Sub, started, complete)?)?;

    // Rank the sum, so an ineligible block cannot outrank an eligible one, keep the budget exactly,
    // and re-AND eligibility: a budget above the eligible count fills with ineligible blocks.
    let rank = stable_descending_rank(b, b.binary(BinOp::Add, scores, eligible)?)?;
    let kept = penalty(b, top_k_keep_mask(b, rank, p.block_topk())?)?;
    let selected = b.binary(BinOp::Add, eligible, kept)?;

    let selection = b.binary(
        BinOp::Max,
        per_position(b, selected, ratio)?,
        per_position(b, tail, ratio)?,
    )?;
    Ok(b.binary(BinOp::Add, selection, causal)?)
}

#[cfg(test)]
mod tests;
