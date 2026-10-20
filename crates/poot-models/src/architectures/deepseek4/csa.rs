//! Compressed sparse attention (CSA): overlapping-window compression of the keys, and a lightning
//! indexer that keeps the `index_topk` best complete windows for each query token.
//!
//! Each source token's key and gate projection is twice the entry width: the first half ("Ca")
//! feeds the NEXT window's entry, the second ("Cb") its own. Entry `w` pools the previous window's
//! Ca half and window `w`'s Cb half (`2 * rate` positions); window 0 has no predecessor, so its Ca
//! half is zero keys with a dropped gate. The indexer has a compressor of its own at its own width,
//! scores every window as `sum_h w_h * relu(q_h . k) * scale` and the selection is the top-k of
//! those scores over the windows the query may read.
//!
//! Selection follows the shared `sparse_top_k` rule: a window is kept when fewer than `index_topk`
//! windows score strictly higher, so windows tied at the cut are all kept (a ReLU score of exactly
//! zero is common). That can keep more than `index_topk` windows, unlike an exact top-k.

use poot_graph_ir::ops::MASK_HIDDEN;
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{DsaRole, WeightId, WeightMap, WeightRole};

use super::config::{DeepseekV4Config, LayerKind};
use super::hca::{
    CompressorWeights, Owner, Pooling, complete_windows, entries, softmax_pool, windows_in,
};
use super::sliding::{Compressed, Local, rotate_tail};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::sparse_top_k::{keep_top_k_mask, pairwise_rank};
use crate::components::standard::Step;
use crate::model::{ParamBlock, TraceError, WiringError};

/// A CSA layer's lightning indexer: its query projection from the low-rank query, the per-head
/// weights projection and its own compressor.
#[derive(Debug)]
pub(super) struct IndexerWeights {
    q: Weight,
    weights: Weight,
    compressor: CompressorWeights,
}

impl IndexerWeights {
    pub(super) fn new(
        map: &WeightMap,
        layer: usize,
        cfg: &DeepseekV4Config,
    ) -> Result<Self, WeightError> {
        let ix = *WiringError::require(&cfg.indexer, ParamBlock::Indexer)?;
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Dsa(role)), shape)
        };
        Ok(Self {
            q: w(DsaRole::Q, &[ix.heads * ix.head_dim, cfg.attn.q_lora])?,
            weights: w(DsaRole::Weights, &[ix.heads, cfg.width])?,
            compressor: CompressorWeights::new(
                map,
                layer,
                Owner::Indexer,
                cfg.width,
                cfg.csa_rate,
                ix.head_dim,
                Pooling::Overlapping,
            )?,
        })
    }
}

/// The pooled entries of overlapping windows: `kv` and `gate` (with its position bias) are
/// `[rows, windows, rate, 2 * entry]`; the result is `[rows, windows, entry]`, before the norm.
pub(super) fn overlap_pool(
    b: &Builder,
    kv: Traced,
    gate: Traced,
    entry: usize,
) -> Result<Traced, TraceError> {
    let windows = b.aval(kv)?.shape[1];
    let half =
        |x: Traced, i: usize| Ok::<_, TraceError>(b.slice(x, 3, i * entry, (i + 1) * entry)?);
    // Window 0's predecessor: zero keys and a dropped gate, shaped like one window's Ca half.
    let zero = b.binary_scalar(
        BinOp::Mul,
        b.slice(half(kv, 0)?, 1, 0, 1)?,
        Scalar::F32(0.0),
    )?;
    let dropped = b.binary_scalar(BinOp::Add, zero, Scalar::F32(MASK_HIDDEN))?;
    let previous = |x: Traced, pad: Traced| {
        if windows > 1 {
            Ok::<_, TraceError>(b.concat(1, &[pad, b.slice(half(x, 0)?, 1, 0, windows - 1)?])?)
        } else {
            Ok::<_, TraceError>(pad)
        }
    };
    let kv = b.concat(2, &[previous(kv, zero)?, half(kv, 1)?])?;
    let gate = b.concat(2, &[previous(gate, dropped)?, half(gate, 1)?])?;
    softmax_pool(b, kv, gate)
}

/// The additive mask keeping, per query token, the `topk` best-scoring of the windows it may read
/// (`readable`, `[rows, 1, tokens, windows]`): the scores plus `readable` are ranked, so a window the
/// query cannot read is never picked over one it can, and `readable` stays in the result.
fn select_top_k(
    b: &Builder,
    scores: Traced,
    readable: Traced,
    topk: usize,
) -> Result<Traced, TraceError> {
    let rank = pairwise_rank(b, b.binary(BinOp::Add, scores, readable)?)?;
    let keep = keep_top_k_mask(b, rank, topk)?;
    let dropped = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0))?,
        Scalar::F32(-MASK_HIDDEN),
    )?;
    Ok(b.binary(BinOp::Add, readable, dropped)?)
}

/// A CSA layer's compressed attention inputs: its overlapping-window entries and, per query token,
/// the mask of the windows the indexer keeps. `None` when the cache holds no whole window.
#[expect(clippy::too_many_arguments)]
pub(super) fn compressed(
    b: &Builder,
    step: &Step,
    layer: usize,
    normed: Traced,
    local: &Local,
    compressor: &CompressorWeights,
    indexer: &IndexerWeights,
    cfg: &DeepseekV4Config,
) -> Result<Option<Compressed>, TraceError> {
    let ix = *WiringError::require(&cfg.indexer, ParamBlock::Indexer)?;
    let shape = step.shape();
    let (r, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let rate = cfg.csa_rate;
    let rope = cfg.rope_of(LayerKind::Csa);
    let rope_dim = cfg.attn.rope_dim;

    // Both compressors write their histories whether or not the cache holds a window: the carried
    // state is the same at every capacity.
    let keys = entries(
        b,
        step,
        layer,
        normed,
        &local.write_at,
        compressor,
        "compress",
        rate,
        cfg.attn.head_dim,
        Pooling::Overlapping,
        rope,
        cfg,
    );
    let index_keys = entries(
        b,
        step,
        layer,
        normed,
        &local.write_at,
        &indexer.compressor,
        "indexer",
        rate,
        ix.head_dim,
        Pooling::Overlapping,
        rope,
        cfg,
    );
    let (Some(keys), Some(index_keys)) = (keys?, index_keys?) else {
        return Ok(None);
    };

    // Per indexer head: the query from the low-rank query, rotated at the token's position.
    let q = linear(b, local.q_res, &indexer.q, None)?;
    let q = b.transpose(
        b.reshape(q, vec![r, t, ix.heads, ix.head_dim])?,
        vec![0, 2, 1, 3],
    )?;
    let q = rotate_tail(b, q, &local.rows, rope_dim)?;

    let n = windows_in(cap, rate);
    let keys_per_head = ops::repeat_kv(b, index_keys, ix.heads)?;
    let dots = b.matmul(q, b.transpose(keys_per_head, vec![0, 1, 3, 2])?)?;
    let scored = b.binary_scalar(
        BinOp::Mul,
        ops::relu(b, dots)?,
        Scalar::F32((ix.head_dim as f32).powf(-0.5)),
    )?;
    let head_weights = b.binary_scalar(
        BinOp::Mul,
        b.reshape(
            b.transpose(linear(b, normed, &indexer.weights, None)?, vec![0, 2, 1])?,
            vec![r, ix.heads, t, 1],
        )?,
        Scalar::F32((ix.heads as f32).powf(-0.5)),
    )?;
    let scores = b.reduce(
        RedOp::Sum,
        b.binary(BinOp::Mul, scored, head_weights)?,
        1,
        true,
    )?;

    let readable = complete_windows(b, step, rate, n)?;
    Ok(Some(Compressed {
        entries: keys,
        mask: select_top_k(b, scores, readable, ix.topk.min(n))?,
    }))
}
