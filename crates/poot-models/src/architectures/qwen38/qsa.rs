//! Qwen Sparse Attention (QSA): the block indexer that narrows which cached positions a
//! full-attention layer reads (`Qwen4ExpTextQSAIndexer`). The layer itself is the ordinary gated GQA
//! attention (`components::attention` with the output gate); the indexer only replaces its
//! causal-over-cache mask by [`qsa_mask`].
//!
//! The indexer keeps one raw key per cached position (the layer's `index` cache, carried like the
//! attention's own) and scores *blocks* of `ratio` consecutive positions:
//!
//! 1. The layer's normed input projects to a per-head query `q [heads, dim]` and one shared raw key
//!    `k [dim]`. `q` gets a norm and partial RoPE at its own position.
//! 2. The cached raw keys are mean-pooled into blocks aligned to absolute positions, and only then
//!    does the pooled key get its own norm and partial RoPE at the block's first position: pool,
//!    then norm, then rope.
//! 3. Query `t` scores block `b` as `sum_h ReLU(q_{t,h} . k_b) / sqrt(dim)`, an unweighted sum over
//!    the heads.
//! 4. Per query, the best `budget / ratio` blocks stay visible, ties going to the lower block (the
//!    budget is exact even on equal scores). A block is eligible once every position of it is at or
//!    before the query. The query's own unfinished block is always visible, unscored.
//!
//! Every quantity of the selection is a function of the query's position, so the mask is built in
//! the graph from the step's `Slot::Pos` and one graph serves a prompt, a chunk and one token.
//! Pooling reads the whole capacity every step (`capacity` is a multiple of `ratio`): the
//! eligibility mask, not incremental bookkeeping, keeps an unfinished or unwritten block out.
//!
//! All masks are `{0, -1e30}` valued, so AND is a plain add (two penalties never overflow `f32`) and
//! OR is a max.

use poot_graph_ir::ops::{stable_descending_rank, top_k_keep_mask};
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{QsaRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use crate::components::attention::{carried_cache, first_writes};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormParams, norm};
use crate::components::rope::{RopeParams, apply_rope, rope_rows};
use crate::components::standard::{ParamError, Step};
use crate::model::ConfigReason;

/// The additive penalty of a position the selection drops or has not made visible.
const DROPPED: f32 = -1e30;

/// Checked QSA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaParams {
    heads: usize,
    dim: usize,
    budget: usize,
    ratio: usize,
    rope: RopeParams,
    norm: NormParams,
}

impl QsaParams {
    /// `heads` indexer heads of `dim` width scoring blocks of `ratio` positions, keeping
    /// `budget / ratio` blocks per query. `rope` rotates the leading dims of each head and key (no
    /// wider than `dim`); `norm` is the query and key norm.
    pub fn new(
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
        Ok(Self {
            heads,
            dim,
            budget,
            ratio,
            rope,
            norm,
        })
    }

    /// The indexer head width.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The rows of the fused query and key projection that are the query's.
    pub fn query_rows(&self) -> usize {
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
pub struct QsaWeights {
    q: Weight,
    k: Weight,
    q_norm: Weight,
    k_norm: Weight,
}

impl QsaWeights {
    pub fn new(
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
fn penalty(b: &Builder, indicator: Traced) -> Traced {
    b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, indicator, Scalar::F32(1.0)),
        Scalar::F32(-DROPPED),
    )
}

/// Spread a per-block value `[rows, 1, tokens, blocks]` over the `ratio` positions of each block:
/// `[rows, 1, tokens, blocks * ratio]`.
fn per_position(b: &Builder, blocks: Traced, ratio: usize) -> Traced {
    let [rows, one, t, nb] = b.aval(blocks).shape[..] else {
        unreachable!("a per-block mask is [rows, 1, tokens, blocks]")
    };
    let wide = b.broadcast(
        b.reshape(blocks, vec![rows, one, t, nb, 1]),
        vec![rows, one, t, nb, ratio],
    );
    b.reshape(wide, vec![rows, one, t, nb * ratio])
}

/// The mask of `layer`'s sparse attention: `causal` (`[rows, 1, tokens, capacity]`) plus
/// [`DROPPED`] on every position outside each query's selection. `x` is the layer's normed input
/// `[rows, tokens, width]`; the layer's raw indexer keys (`kv.l{layer}.index`) are carried by `step`.
/// The step's capacity must be a multiple of the block `ratio`.
pub fn qsa_mask(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    causal: Traced,
    w: &QsaWeights,
    p: &QsaParams,
) -> Traced {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (heads, dim, ratio) = (p.heads, p.dim, p.ratio);
    let nb = cap / ratio;
    debug_assert_eq!(nb * ratio, cap, "the model checks the capacity granule");

    // The query: a norm over each head, then RoPE at its own position.
    let q = norm(
        b,
        b.reshape(linear(b, x, &w.q, None), vec![rows, t, heads, dim]),
        &w.q_norm,
        p.norm,
    );
    let q = apply_rope(
        b,
        b.transpose(q, vec![0, 2, 1, 3]),
        &rope_rows(b, p.rope, step.pos(), cap),
    ); // [rows, heads, tokens, dim]

    // The raw keys of every cached position, this step's written first.
    let raw = b.reshape(linear(b, x, &w.k, None), vec![rows, 1, t, dim]);
    let write_at = first_writes(b, step);
    let keys = carried_cache(b, step, &write_at, layer, "index", raw, 1, dim); // [rows, 1, cap, dim]

    // Pool each block, then norm it, then rotate it at the block's first position.
    let summed = b.reduce(
        RedOp::Sum,
        b.reshape(keys, vec![rows, 1, nb, ratio, dim]),
        3,
        true,
    );
    let pooled = b.reshape(
        b.binary_scalar(BinOp::Mul, summed, Scalar::F32(1.0 / ratio as f32)),
        vec![rows, 1, nb, dim],
    );
    let starts_i32 = b.binary_scalar(BinOp::Mul, b.iota_i32(nb), Scalar::I32(ratio as i32));
    let block_pos = b.broadcast(b.reshape(starts_i32, vec![1, nb]), vec![rows, nb]);
    let block_k = apply_rope(
        b,
        norm(b, pooled, &w.k_norm, p.norm),
        &rope_rows(b, p.rope, block_pos, cap),
    ); // [rows, 1, blocks, dim]

    // Score: `sum_h ReLU(q . k_block) / sqrt(dim)`, the head axis kept at one.
    let dots = b.matmul(q, b.transpose(block_k, vec![0, 1, 3, 2])); // [rows, heads, tokens, blocks]
    let scores = b.binary_scalar(
        BinOp::Mul,
        b.reduce(RedOp::Sum, ops::relu(b, dots), 1, true),
        Scalar::F32(1.0 / (dim as f32).sqrt()),
    ); // [rows, 1, tokens, blocks]

    // Eligibility and the tail, from each query's position and each block's first position.
    let wide = vec![rows, 1, t, nb];
    let pos = b.broadcast(
        b.reshape(b.cast(step.pos(), DType::F32), vec![rows, 1, t, 1]),
        wide.clone(),
    );
    let first = b.broadcast(
        b.reshape(
            b.binary_scalar(BinOp::Mul, b.iota(nb), Scalar::F32(ratio as f32)),
            vec![1, 1, 1, nb],
        ),
        wide.clone(),
    );
    let last = b.binary_scalar(BinOp::Add, first, Scalar::F32((ratio - 1) as f32));
    // Complete as of the query: its last position is at or before the query.
    let complete = b.binary(BinOp::Ge, pos, last);
    let eligible = penalty(b, complete);
    // The query's own unfinished block: it starts at or before the query, and is not yet complete.
    let started = b.binary(BinOp::Ge, pos, first);
    let tail = penalty(b, b.binary(BinOp::Sub, started, complete));

    // Rank the sum, so an ineligible block cannot outrank an eligible one, keep the budget exactly,
    // and re-AND eligibility: a budget above the eligible count fills with ineligible blocks.
    let rank = stable_descending_rank(b, b.binary(BinOp::Add, scores, eligible));
    let kept = penalty(b, top_k_keep_mask(b, rank, p.block_topk()));
    let selected = b.binary(BinOp::Add, eligible, kept);

    let selection = b.binary(
        BinOp::Max,
        per_position(b, selected, ratio),
        per_position(b, tail, ratio),
    );
    b.binary(BinOp::Add, selection, causal)
}
