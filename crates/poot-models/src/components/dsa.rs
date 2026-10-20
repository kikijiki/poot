//! DeepSeek sparse attention (DSA, DeepSeek-V3.2): the lightning indexer and fine-grained token
//! selection as a mask over multi-head latent attention ([`super::mla`]).
//!
//! The indexer scores every cached key for each query token with
//! `I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])`: `q` per indexer head from the low-rank query,
//! `w` per head from the hidden state, and one key per position shared by the heads, LayerNorm'ed
//! and rotated (half-split, over the leading `qk_rope_head_dim` dims only) before it is cached. The
//! keys the step caches carry the position axis like attention's own ([`carried_cache`]).
//!
//! Selection keeps the `topk` best-scoring causal positions of each query token as an additive
//! mask on top of the causal one, so it composes with the attention core unchanged: ranking the
//! scores plus the causal mask keeps top-k from picking a position the query cannot see.

use poot_graph_ir::ops::MASK_HIDDEN;
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_tensor::DType;
use poot_weights::{AttnRole, CompressRole, DsaRole, MlaRole, WeightId, WeightMap, WeightRole};

use super::attention::{carried_cache, first_writes};
use super::linear::{Weight, WeightError, linear};
use super::mla::Selection;
use super::norm::{NormParams, norm};
use super::rope::{RopeParams, apply_rope, rope_rows};
use super::sparse_top_k::{keep_top_k_mask, pairwise_rank};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;
use crate::model::TraceError;

/// The indexer's key LayerNorm epsilon (`Indexer.k_norm = nn.LayerNorm(head_dim, eps=1e-6)`).
const K_NORM_EPS: f32 = 1e-6;

/// Checked DSA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsaParams {
    heads: usize,
    dim: usize,
    topk: usize,
    rope: RopeParams,
}

impl DsaParams {
    /// `heads` indexer heads of `dim` width keeping the `topk` best positions per token; `rope`
    /// rotates the leading `rope.rotary()` dims of each head, which must not exceed `dim`.
    pub fn new(
        heads: usize,
        dim: usize,
        topk: usize,
        rope: RopeParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("index_n_heads", heads),
            ("index_head_dim", dim),
            ("index_topk", topk),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if rope.rotary() > dim {
            return Err(ParamError::new(
                "index_head_dim",
                ConfigReason::Exceeds { max: rope.rotary() },
            ));
        }
        Ok(Self {
            heads,
            dim,
            topk,
            rope,
        })
    }
}

/// A layer's indexer weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsaWeights {
    q: Weight,
    k: Weight,
    k_norm: Weight,
    k_norm_bias: Weight,
    weights: Weight,
}

impl DsaWeights {
    /// The indexer's weights for a model of `width` hidden units whose low-rank query is `q_rank`
    /// wide.
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        q_rank: usize,
        p: &DsaParams,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Dsa(role)), shape)
        };
        Ok(Self {
            q: w(DsaRole::Q, &[p.heads * p.dim, q_rank])?,
            k: w(DsaRole::K, &[p.dim, width])?,
            k_norm: w(DsaRole::KNorm, &[p.dim])?,
            k_norm_bias: w(DsaRole::KNormBias, &[p.dim])?,
            weights: w(DsaRole::Weights, &[p.heads, width])?,
        })
    }
}

/// The indexer's score of every cached key for each query token of `layer`: `[rows, 1, tokens,
/// capacity]`, `sum_j w[t, j] * ReLU(q[t, j] . k[s])`. `x` is the layer's normalized hidden state
/// `[rows, tokens, width]` (the key and the weights read it) and `qr` the low-rank query `[rows,
/// tokens, q_rank]` (the query reads it). The layer's indexer keys (`kv.l{layer}.index`) are
/// carried by `step`; a position not yet written scores zero.
pub fn dsa_scores(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    qr: Traced,
    w: &DsaWeights,
    p: &DsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (hi, di) = (p.heads, p.dim);
    let rope = rope_rows(b, p.rope, step.pos(), cap)?;

    let q = linear(b, qr, &w.q, None)?;
    let q = b.transpose(b.reshape(q, vec![rows, t, hi, di])?, vec![0, 2, 1, 3])?; // [rows, hi, t, di]
    let q = apply_rope(b, q, &rope)?;

    let k = ops::layernorm(
        b,
        linear(b, x, &w.k, None)?,
        w.k_norm.declare_f32(b)?,
        w.k_norm_bias.declare_f32(b)?,
        K_NORM_EPS,
    )?;
    let k = apply_rope(b, b.reshape(k, vec![rows, 1, t, di])?, &rope)?;
    let write_at = first_writes(b, step)?;
    let keys = carried_cache(b, step, &write_at, layer, "index", k, 1, di)?; // [rows, 1, cap, di]

    let weights = linear(b, x, &w.weights, None)?; // [rows, t, hi]
    let weights = b.reshape(b.transpose(weights, vec![0, 2, 1])?, vec![rows, hi, t, 1])?;

    // The head axis reduced, kept at size one so it adds onto the `[rows, 1, t, cap]` mask.
    let dots = b.matmul(q, b.transpose(keys, vec![0, 1, 3, 2])?)?; // [rows, hi, t, cap]
    let scored = b.binary(BinOp::Mul, ops::relu(b, dots)?, weights)?;
    Ok(b.reduce(RedOp::Sum, scored, 1, true)?)
}

/// The mask of the sparse attention of `layer`: `sel.mask` plus a [`MASK_HIDDEN`] penalty on every
/// position outside each query token's `topk` best by [`dsa_scores`]. `x` is the layer's normalized
/// hidden state `[rows, tokens, width]`; the indexer's query reads the low-rank query in `sel`.
pub fn dsa_select(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    sel: Selection,
    w: &DsaWeights,
    p: &DsaParams,
) -> Result<Traced, TraceError> {
    let scores = dsa_scores(b, step, layer, x, sel.qr, w, p)?;
    let ranked = b.binary(BinOp::Add, scores, sel.mask)?;
    let keep = keep_top_k_mask(b, pairwise_rank(b, ranked)?, p.topk)?; // 1.0 kept, 0.0 dropped
    let drop = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0))?,
        Scalar::F32(-MASK_HIDDEN),
    )?; // 0 kept, MASK_HIDDEN dropped
    Ok(b.binary(BinOp::Add, sel.mask, drop)?)
}

/// The pooled indexer's key LayerNorm epsilon.
const POOLED_K_NORM_EPS: f32 = 1e-6;

/// Checked parameters of GLM-5.3-Flash's DSA layer: no-position multi-head latent attention whose
/// keys and values every query reads through a pooled indexer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PooledDsaParams {
    heads: usize,
    q_rank: usize,
    kv_rank: usize,
    qk_dim: usize,
    v_dim: usize,
    norm: NormParams,
    index_heads: usize,
    index_dim: usize,
    pool: usize,
    pool_topk: usize,
}

impl PooledDsaParams {
    /// `heads` heads of `qk_dim` query/key and `v_dim` value channels, read from a `kv_rank`-wide
    /// latent through a `q_rank`-wide low-rank query, normalized with `norm`. The indexer has
    /// `index_heads` heads of `index_dim` channels, pools `pool` consecutive keys into one and keeps
    /// the best `index_topk / pool` complete pools (so `index_topk` raw tokens) plus the tokens
    /// after the last complete pool.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        heads: usize,
        q_rank: usize,
        kv_rank: usize,
        qk_dim: usize,
        v_dim: usize,
        norm: NormParams,
        index_heads: usize,
        index_dim: usize,
        index_topk: usize,
        pool: usize,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("num_attention_heads", heads),
            ("q_lora_rank", q_rank),
            ("kv_lora_rank", kv_rank),
            ("qk_head_dim", qk_dim),
            ("v_head_dim", v_dim),
            ("index_n_heads", index_heads),
            ("index_head_dim", index_dim),
            ("index_topk", index_topk),
            ("index_kpool", pool),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if !index_topk.is_multiple_of(pool) {
            return Err(ParamError::new(
                "index_topk",
                ConfigReason::NotDivisible { by: pool },
            ));
        }
        Ok(Self {
            heads,
            q_rank,
            kv_rank,
            qk_dim,
            v_dim,
            norm,
            index_heads,
            index_dim,
            pool,
            pool_topk: index_topk / pool,
        })
    }
}

/// A layer's pooled-DSA weights: the latent attention's projections, the indexer's
/// ([`DsaWeights`]) and the pooling's gate and position bias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PooledDsaWeights {
    q_a: Weight,
    q_a_norm: Weight,
    q_b: Weight,
    kv_a: Weight,
    kv_a_norm: Weight,
    kv_b: Weight,
    out: Weight,
    indexer: DsaWeights,
    pool_gate: Weight,
    pool_bias: Weight,
}

impl PooledDsaWeights {
    /// Resolves the layer's latent-attention, indexer and pooling tensors from `map` and checks each
    /// shape against `p` and the model `width`; a missing or mis-shaped tensor is a typed
    /// [`WeightError`].
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &PooledDsaParams,
    ) -> Result<Self, WeightError> {
        let w = |role: WeightRole, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, role), shape)
        };
        let mla = |role, shape: &[usize]| w(WeightRole::Mla(role), shape);
        let (h, qk, v, id) = (p.heads, p.qk_dim, p.v_dim, p.index_dim);
        Ok(Self {
            q_a: mla(MlaRole::QA, &[p.q_rank, width])?,
            q_a_norm: mla(MlaRole::QANorm, &[p.q_rank])?,
            q_b: mla(MlaRole::QB, &[h * qk, p.q_rank])?,
            kv_a: mla(MlaRole::KvA, &[p.kv_rank, width])?,
            kv_a_norm: mla(MlaRole::KvANorm, &[p.kv_rank])?,
            kv_b: mla(MlaRole::KvB, &[h * (qk + v), p.kv_rank])?,
            out: w(WeightRole::Attn(AttnRole::O), &[width, h * v])?,
            indexer: DsaWeights {
                q: w(WeightRole::Dsa(DsaRole::Q), &[p.index_heads * id, p.q_rank])?,
                k: w(WeightRole::Dsa(DsaRole::K), &[id, width])?,
                k_norm: w(WeightRole::Dsa(DsaRole::KNorm), &[id])?,
                k_norm_bias: w(WeightRole::Dsa(DsaRole::KNormBias), &[id])?,
                weights: w(WeightRole::Dsa(DsaRole::Weights), &[p.index_heads, width])?,
            },
            pool_gate: w(
                WeightRole::Compress(CompressRole::IndexerGate),
                &[id, width],
            )?,
            pool_bias: w(
                WeightRole::Compress(CompressRole::IndexerPositionBias),
                &[p.pool, id],
            )?,
        })
    }
}

/// The pooled indexer's keep mask of the cached positions, one row per query token: `[rows, 1,
/// tokens, capacity]`, `0` for a position the token attends and [`MASK_HIDDEN`] for the others.
///
/// The cached indexer keys (and the gates that weight them) of every `pool` consecutive positions
/// pool into one key, the softmax of the gates plus the position bias over the pool's positions
/// weighting the keys per channel. A pool is complete for the token at position `p` when its last
/// position is `<= p`. The token scores each complete pool `sum_j w[t, j] * ReLU(q[t, j] . pooled) /
/// sqrt(dim * heads)` and keeps the `pool_topk` best (equal scores keep the lower pool) with all
/// their positions, plus every position after its last complete pool up to itself. The keys the
/// step caches carry the position axis like attention's own (`kv.l{layer}.index`); a pool not yet
/// written is never complete.
#[expect(clippy::too_many_arguments)]
fn pooled_keep(
    b: &Builder,
    step: &Step,
    write_at: &[Traced],
    layer: usize,
    x: Traced,
    qr: Traced,
    w: &PooledDsaWeights,
    p: &PooledDsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (hi, di, pool) = (p.index_heads, p.index_dim, p.pool);
    let pools = cap / pool;
    let ix = &w.indexer;

    let key = ops::layernorm(
        b,
        linear(b, x, &ix.k, None)?,
        ix.k_norm.declare_f32(b)?,
        ix.k_norm_bias.declare_f32(b)?,
        POOLED_K_NORM_EPS,
    )?;
    let gate = linear(b, x, &w.pool_gate, None)?;
    let new = b.reshape(b.concat(2, &[key, gate])?, vec![rows, 1, t, 2 * di])?;
    // `[rows, 1, cap, 2 * dim]`: the key and the gate of every cached position.
    let cached = carried_cache(b, step, write_at, layer, "index", new, 1, 2 * di)?;

    // The position of every token and of every cached slot, as exact F32 integers.
    let pos = b.reshape(b.cast(step.pos(), DType::F32)?, vec![rows, t, 1])?;
    let slots = b.broadcast(
        b.reshape(b.iota(cap)?, vec![1, 1, cap])?,
        vec![rows, t, cap],
    )?;
    let visible = b.binary(BinOp::Ge, b.broadcast(pos, vec![rows, t, cap])?, slots)?;

    // The positions after the last complete pool, and the positions of the pools kept.
    let (tail_from, kept) = if pools == 0 {
        (b.binary_scalar(BinOp::Mul, pos, Scalar::F32(0.0))?, None)
    } else {
        // Pool `j` ends at position `(j + 1) * pool - 1`: complete once the token reaches it.
        let ends = b.binary_scalar(
            BinOp::Add,
            b.binary_scalar(BinOp::Mul, b.iota(pools)?, Scalar::F32(pool as f32))?,
            Scalar::F32(pool as f32 - 1.0),
        )?;
        let complete = b.binary(
            BinOp::Ge,
            b.broadcast(pos, vec![rows, t, pools])?,
            b.broadcast(b.reshape(ends, vec![1, 1, pools])?, vec![rows, t, pools])?,
        )?; // [rows, t, pools]
        let tail_from = b.binary_scalar(
            BinOp::Mul,
            b.reduce(RedOp::Sum, complete, 2, true)?,
            Scalar::F32(pool as f32),
        )?;

        // Each pool's key: the cached keys weighted by the softmax of the gates over the pool.
        let block = b.reshape(
            b.slice(cached, 2, 0, pools * pool)?,
            vec![rows, pools, pool, 2 * di],
        )?;
        let keys = b.slice(block, 3, 0, di)?;
        let bias = b.broadcast(
            b.reshape(w.pool_bias.declare_f32(b)?, vec![1, 1, pool, di])?,
            vec![rows, pools, pool, di],
        )?;
        let gates = b.binary(BinOp::Add, b.slice(block, 3, di, 2 * di)?, bias)?;
        let weights = ops::softmax(b, b.transpose(gates, vec![0, 1, 3, 2])?)?; // [rows, pools, di, pool]
        let pooled = b.reduce(
            RedOp::Sum,
            b.binary(BinOp::Mul, weights, b.transpose(keys, vec![0, 1, 3, 2])?)?,
            3,
            false,
        )?; // [rows, pools, di]

        let q = b.transpose(
            b.reshape(linear(b, qr, &ix.q, None)?, vec![rows, t, hi, di])?,
            vec![0, 2, 1, 3],
        )?; // [rows, hi, t, di]
        let pooled = b.reshape(
            b.transpose(pooled, vec![0, 2, 1])?,
            vec![rows, 1, di, pools],
        )?;
        let dots = b.binary_scalar(
            BinOp::Mul,
            ops::relu(b, b.matmul(q, pooled)?)?,
            Scalar::F32(1.0 / (di as f32).sqrt()),
        )?; // [rows, hi, t, pools]
        let head_weights = b.binary_scalar(
            BinOp::Mul,
            linear(b, x, &ix.weights, None)?,
            Scalar::F32(1.0 / (hi as f32).sqrt()),
        )?;
        let head_weights = b.reshape(
            b.transpose(head_weights, vec![0, 2, 1])?,
            vec![rows, hi, t, 1],
        )?;
        let scores = b.reduce(
            RedOp::Sum,
            b.binary(BinOp::Mul, dots, head_weights)?,
            1,
            false,
        )?;

        // The best `pool_topk` complete pools, ties to the lower pool: the block selection of
        // `ops::block_top_k_packed` over the pools (`compile` forms `OpKind::BlockTopK` from it); its keep
        // indicator is the chosen flag of each pool.
        let kept_pools = p.pool_topk.min(pools);
        let selection = ops::block_top_k_packed(
            b,
            b.reshape(scores, vec![rows, 1, t, pools])?,
            step.pos(),
            pool,
            kept_pools,
        )?;
        let chosen = b.reshape(
            b.cast(
                b.slice(selection, 3, kept_pools, kept_pools + pools)?,
                DType::F32,
            )?,
            vec![rows, t, pools],
        )?;
        // Spread each pool's flag over its positions: `[pools, cap]` ones where slot `s` is in pool `j`.
        let starts = b.broadcast(
            b.reshape(
                b.binary_scalar(BinOp::Mul, b.iota(pools)?, Scalar::F32(pool as f32))?,
                vec![pools, 1],
            )?,
            vec![pools, cap],
        )?;
        let slot_row = b.broadcast(b.reshape(b.iota(cap)?, vec![1, cap])?, vec![pools, cap])?;
        let in_pool = b.binary(
            BinOp::Mul,
            b.binary(BinOp::Ge, slot_row, starts)?,
            b.binary(
                BinOp::Ge,
                b.binary_scalar(BinOp::Add, starts, Scalar::F32(pool as f32 - 1.0))?,
                slot_row,
            )?,
        )?;
        (tail_from, Some(b.matmul(chosen, in_pool)?)) // [rows, t, cap]
    };
    let tail = b.binary(
        BinOp::Mul,
        b.binary(
            BinOp::Ge,
            slots,
            b.broadcast(tail_from, vec![rows, t, cap])?,
        )?,
        visible,
    )?;
    let keep = match kept {
        Some(kept) => b.binary(BinOp::Add, kept, tail)?,
        None => tail,
    };
    let drop = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0))?,
        Scalar::F32(-MASK_HIDDEN),
    )?; // 0 kept, MASK_HIDDEN dropped
    Ok(b.reshape(drop, vec![rows, 1, t, cap])?)
}

/// GLM-5.3-Flash's DSA layer over `x` (`[rows, tokens, width]`) at layer `layer`: `[rows, tokens,
/// width]`. Latent attention without rotary embedding: the low-rank query, the normalized latent
/// `kv.l{layer}.c` (carried by `step`) expanded through `kv_b` into every head's keys and values, and
/// the attention under the pooled indexer's keep mask ([`pooled_keep`]; its keys are
/// `kv.l{layer}.index`).
pub fn pooled_dsa(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &PooledDsaWeights,
    p: &PooledDsaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (h, qk, vd, rank) = (p.heads, p.qk_dim, p.v_dim, p.kv_rank);

    let qr = norm(b, linear(b, x, &w.q_a, None)?, &w.q_a_norm, p.norm)?;
    let q = b.transpose(
        b.reshape(linear(b, qr, &w.q_b, None)?, vec![rows, t, h, qk])?,
        vec![0, 2, 1, 3],
    )?;

    let latent = norm(b, linear(b, x, &w.kv_a, None)?, &w.kv_a_norm, p.norm)?;
    let write_at = first_writes(b, step)?;
    let c = carried_cache(
        b,
        step,
        &write_at,
        layer,
        "c",
        b.reshape(latent, vec![rows, 1, t, rank])?,
        1,
        rank,
    )?;
    // Every head's keys and values from the whole cached latent.
    let kv = linear(b, b.reshape(c, vec![rows, cap, rank])?, &w.kv_b, None)?;
    let kv = b.transpose(
        b.reshape(kv, vec![rows, cap, h, qk + vd])?,
        vec![0, 2, 1, 3],
    )?;
    let (k, v) = (b.slice(kv, 3, 0, qk)?, b.slice(kv, 3, qk, qk + vd)?);

    let mask = pooled_keep(b, step, &write_at, layer, x, qr, w, p)?;
    let out = ops::attention_masked(b, q, k, v, 1, 1.0 / (qk as f32).sqrt(), mask)?;
    let out = b.reshape(b.transpose(out, vec![0, 2, 1, 3])?, vec![rows, t, h * vd])?;
    linear(b, out, &w.out, None)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::rope_table::RopeFlavor;
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const WIDTH: usize = 6;
    const Q_RANK: usize = 3;
    const HEADS: usize = 2;
    const DIM: usize = 4;
    const ROPE: usize = 2;
    const THETA: f32 = 10_000.0;
    const T: usize = 5;

    fn id(role: DsaRole) -> WeightId {
        WeightId::layer(0, WeightRole::Dsa(role))
    }

    fn params(topk: usize) -> DsaParams {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain).unwrap();
        DsaParams::new(HEADS, DIM, topk, rope).unwrap()
    }

    fn fixture() -> Fx {
        let mut fx = Fx::new(13);
        fx.f32(id(DsaRole::Q), &[HEADS * DIM, Q_RANK]);
        fx.f32(id(DsaRole::K), &[DIM, WIDTH]);
        fx.set(id(DsaRole::KNorm), &[DIM], vec![1.1, 0.9, 1.2, 0.8]);
        fx.f32(id(DsaRole::KNormBias), &[DIM]);
        fx.f32(id(DsaRole::Weights), &[HEADS, WIDTH]);
        fx
    }

    /// The scores `[T, CAP]` (or, with `select`, the selection mask) of `T` tokens from position
    /// 0 over the zero cache.
    fn run(fx: &Fx, topk: usize, x: &[f32], qr: &[f32], select: bool) -> Vec<f32> {
        let p = params(topk);
        let w = DsaWeights::new(&fx.map(), 0, WIDTH, Q_RANK, &p).unwrap();
        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(T, CAP, LogitRows::All)).unwrap();
        let xs = b.constant("x", TensorType::f32(vec![1, T, WIDTH])).unwrap();
        let qrs = b
            .constant("qr", TensorType::f32(vec![1, T, Q_RANK]))
            .unwrap();
        let out = if select {
            let mask = ops::causal_mask_from_pos(&b, step.pos(), CAP, None).unwrap();
            dsa_select(&b, &step, 0, xs, Selection { qr: qrs, mask }, &w, &p).unwrap()
        } else {
            dsa_scores(&b, &step, 0, xs, qrs, &w, &p).unwrap()
        };
        let g = step.finish(b, out).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let ids = vec![0; T];
        let pos: Vec<i32> = (0..T as i32).collect();
        fx.eval(
            &g,
            &[("x", x), ("qr", qr)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &state,
        )
        .0
    }

    /// Half-split rotation of the leading `ROPE` dims of `row` at `pos` in f64: `(x[j], x[j +
    /// ROPE/2])` turn by `pos * theta^(-2j / ROPE)`.
    fn rotate(row: &[f64], pos: usize) -> Vec<f64> {
        let mut out = row.to_vec();
        let half = ROPE / 2;
        for j in 0..half {
            let angle = pos as f64 * f64::from(THETA).powf(-2.0 * j as f64 / ROPE as f64);
            let (s, c) = angle.sin_cos();
            out[j] = row[j] * c - row[j + half] * s;
            out[j + half] = row[j + half] * c + row[j] * s;
        }
        out
    }

    fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..rows)
            .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
            .collect()
    }

    /// `I[t, s]` for every key the step wrote (`s < T`, causal or not: the causal mask is applied after;
    /// unwritten keys score zero), from the definition.
    fn reference_scores(fx: &Fx, x: &[f32], qr: &[f32]) -> Vec<f64> {
        let value = |role| fx.values(id(role));
        let (wk, kn, kb) = (
            value(DsaRole::K),
            value(DsaRole::KNorm),
            value(DsaRole::KNormBias),
        );
        let row = |data: &[f32], t: usize, n: usize| -> Vec<f64> {
            data[t * n..(t + 1) * n].iter().map(|&v| v.into()).collect()
        };
        let keys: Vec<Vec<f64>> = (0..T)
            .map(|s| {
                let raw = matvec(&wk, DIM, WIDTH, &row(x, s, WIDTH));
                let mean = raw.iter().sum::<f64>() / DIM as f64;
                let var = raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / DIM as f64;
                let normed: Vec<f64> = (0..DIM)
                    .map(|j| {
                        (raw[j] - mean) / (var + 1e-6).sqrt() * f64::from(kn[j]) + f64::from(kb[j])
                    })
                    .collect();
                let (head, tail) = normed.split_at(ROPE);
                [rotate(head, s), tail.to_vec()].concat()
            })
            .collect();
        let mut out = Vec::new();
        for t in 0..T {
            let q = matvec(&value(DsaRole::Q), HEADS * DIM, Q_RANK, &row(qr, t, Q_RANK));
            let w = matvec(&value(DsaRole::Weights), HEADS, WIDTH, &row(x, t, WIDTH));
            for s in 0..CAP {
                let mut score = 0.0;
                if let Some(key) = keys.get(s) {
                    for h in 0..HEADS {
                        let head = &q[h * DIM..(h + 1) * DIM];
                        let (lead, tail) = head.split_at(ROPE);
                        let head = [rotate(lead, t), tail.to_vec()].concat();
                        let dot: f64 = head.iter().zip(key).map(|(a, b)| a * b).sum();
                        score += w[h] * dot.max(0.0);
                    }
                }
                out.push(score);
            }
        }
        out
    }

    fn inputs() -> (Vec<f32>, Vec<f32>) {
        let scaled = |n, seed| {
            oracle::random(n, seed)
                .into_iter()
                .map(|v| 3.0 * v)
                .collect::<Vec<_>>()
        };
        (scaled(T * WIDTH, 21), scaled(T * Q_RANK, 22))
    }

    /// SC-002: the indexer's scores equal the f64 definition, each weight matters, and the
    /// selection keeps exactly the `topk` best causal positions of every query token (the others
    /// are pushed to the dropped penalty, the invisible ones were already masked). Mutation: an
    /// off-by-one in `keep_top_k_mask` (`>= k + 1`); a row keeps `topk + 1` positions and the
    /// selection assertion goes red.
    #[test]
    fn the_indexer_scores_and_the_selection_match_the_definition() {
        let fx = fixture();
        let (x, qr) = inputs();
        let scores = run(&fx, 2, &x, &qr, false);
        let want = reference_scores(&fx, &x, &qr);
        oracle::assert_matches_f64(&scores, &want, 1e-4);
        for role in [
            DsaRole::Q,
            DsaRole::K,
            DsaRole::KNorm,
            DsaRole::KNormBias,
            DsaRole::Weights,
        ] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id(role));
            oracle::assert_differs(&run(&perturbed, 2, &x, &qr, false), &scores, id(role));
        }

        // The positions a query cannot see were already at the causal penalty (the selection may
        // keep or drop them: they tie far below every visible score), so only the visible ones
        // are compared: the `topk` best of them stay, the rest drop to the penalty.
        let mask = run(&fx, 2, &x, &qr, true);
        for t in 0..T {
            let visible = &want[t * CAP..t * CAP + t + 1];
            let mut kept = 0;
            for s in 0..=t {
                let rank = visible.iter().filter(|&&v| v > visible[s]).count();
                let got = mask[t * CAP + s];
                if rank < 2 {
                    kept += 1;
                    assert!(got.abs() < 1e-3, "token {t} key {s} should stay: {got}");
                } else {
                    assert_eq!(got, MASK_HIDDEN, "token {t} key {s} should drop: {got}");
                }
            }
            assert_eq!(kept, 2.min(t + 1), "token {t}");
        }
    }

    mod pooled {
        use super::*;
        use poot_weights::{AttnRole, CompressRole, MlaRole};

        const WIDTH: usize = 6;
        const HEADS: usize = 2;
        const Q_RANK: usize = 3;
        const KV_RANK: usize = 4;
        const QK: usize = 3;
        const V: usize = 2;
        const INDEX_HEADS: usize = 2;
        const INDEX_DIM: usize = 4;
        const POOL: usize = 2;
        const POOL_TOPK: usize = 1;
        const EPS: f32 = 1e-5;
        const CAPACITY: usize = 12;
        const TOKENS: usize = 10;

        fn role(layer_role: WeightRole) -> WeightId {
            WeightId::layer(0, layer_role)
        }

        fn params() -> PooledDsaParams {
            PooledDsaParams::new(
                HEADS,
                Q_RANK,
                KV_RANK,
                QK,
                V,
                NormParams::new(EPS).unwrap(),
                INDEX_HEADS,
                INDEX_DIM,
                POOL * POOL_TOPK,
                POOL,
            )
            .unwrap()
        }

        /// Every weight of the layer, by id.
        fn roles() -> Vec<(WeightRole, Vec<usize>)> {
            use WeightRole::*;
            vec![
                (Mla(MlaRole::QA), vec![Q_RANK, WIDTH]),
                (Mla(MlaRole::QANorm), vec![Q_RANK]),
                (Mla(MlaRole::QB), vec![HEADS * QK, Q_RANK]),
                (Mla(MlaRole::KvA), vec![KV_RANK, WIDTH]),
                (Mla(MlaRole::KvANorm), vec![KV_RANK]),
                (Mla(MlaRole::KvB), vec![HEADS * (QK + V), KV_RANK]),
                (Attn(AttnRole::O), vec![WIDTH, HEADS * V]),
                (Dsa(DsaRole::Q), vec![INDEX_HEADS * INDEX_DIM, Q_RANK]),
                (Dsa(DsaRole::K), vec![INDEX_DIM, WIDTH]),
                (Dsa(DsaRole::KNorm), vec![INDEX_DIM]),
                (Dsa(DsaRole::KNormBias), vec![INDEX_DIM]),
                (Dsa(DsaRole::Weights), vec![INDEX_HEADS, WIDTH]),
                (Compress(CompressRole::IndexerGate), vec![INDEX_DIM, WIDTH]),
                (
                    Compress(CompressRole::IndexerPositionBias),
                    vec![POOL, INDEX_DIM],
                ),
            ]
        }

        fn fixture() -> Fx {
            let mut fx = Fx::new(41);
            for (r, shape) in roles() {
                if matches!(
                    r,
                    WeightRole::Mla(MlaRole::QANorm | MlaRole::KvANorm)
                        | WeightRole::Dsa(DsaRole::KNorm)
                ) {
                    let n = shape[0];
                    fx.set(
                        role(r),
                        &shape,
                        (0..n).map(|i| 1.0 + 0.1 * i as f32).collect(),
                    );
                } else if matches!(r, WeightRole::Attn(_)) {
                    fx.bf16(role(r), &shape);
                } else if matches!(r, WeightRole::Dsa(_) | WeightRole::Compress(_)) {
                    // Wide indexer weights, so the ranking of the pools moves with each of them.
                    fx.f32(role(r), &shape);
                    let wide = fx.values(role(r)).into_iter().map(|v| 6.0 * v).collect();
                    fx.set(role(r), &shape, wide);
                } else {
                    fx.f32(role(r), &shape);
                }
            }
            fx
        }

        struct Run {
            out: Vec<f32>,
            state: Vec<Vec<f32>>,
        }

        fn run(fx: &Fx, topk_pools: usize, x: &[f32], start: i32, state: &[Vec<f32>]) -> Run {
            let mut p = params();
            p.pool_topk = topk_pools;
            let w = PooledDsaWeights::new(&fx.map(), 0, WIDTH, &p).unwrap();
            let tokens = x.len() / WIDTH;
            let b = Builder::for_tests();
            let step = Step::new(&b, step_shape(tokens, CAPACITY, LogitRows::All)).unwrap();
            let xs = b
                .constant("x", TensorType::f32(vec![1, tokens, WIDTH]))
                .unwrap();
            let y = pooled_dsa(&b, &step, 0, xs, &w, &p).unwrap();
            let g = step.finish(b, y).unwrap();
            let zeros: Vec<Vec<f32>> = g
                .state()
                .map(|(si, _)| vec![0.0; g.aval(si).numel()])
                .collect();
            let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
                .iter()
                .map(Vec::as_slice)
                .collect();
            let ids = vec![0; tokens];
            let pos: Vec<i32> = (start..).take(tokens).collect();
            let (out, state) = fx.eval(
                &g,
                &[("x", x)],
                &[(Slot::Token, &ids), (Slot::Pos, &pos)],
                &state,
            );
            Run { out, state }
        }

        fn matvec(w: &[f32], rows: usize, x: &[f64]) -> Vec<f64> {
            let k = x.len();
            (0..rows)
                .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
                .collect()
        }

        fn rms(x: &[f64], scale: &[f32]) -> Vec<f64> {
            let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
            let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
            x.iter()
                .zip(scale)
                .map(|(v, s)| v * inv * f64::from(*s))
                .collect()
        }

        /// What the layer computes for the token at each position, and the caches it leaves, written
        /// from the definition in f64 (the restored `reference_step`): the pooled indexer's selection
        /// by an explicit sort, the attention over exactly the selected positions.
        fn reference(fx: &Fx, topk_pools: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let v = |r| fx.values(role(r));
            let mla = |r| v(WeightRole::Mla(r));
            let dsa = |r| v(WeightRole::Dsa(r));
            let gate_w = v(WeightRole::Compress(CompressRole::IndexerGate));
            let ape = v(WeightRole::Compress(CompressRole::IndexerPositionBias));
            type Rows = Vec<Vec<f64>>;
            let (mut latents, mut keys, mut gates) = (Rows::new(), Rows::new(), Rows::new());
            let mut out = Vec::new();
            for (p, token) in x.chunks(WIDTH).enumerate() {
                let row: Vec<f64> = token.iter().map(|&t| t.into()).collect();
                let q_res = rms(
                    &matvec(&mla(MlaRole::QA), Q_RANK, &row),
                    &mla(MlaRole::QANorm),
                );
                let q = matvec(&mla(MlaRole::QB), HEADS * QK, &q_res);
                latents.push(rms(
                    &matvec(&mla(MlaRole::KvA), KV_RANK, &row),
                    &mla(MlaRole::KvANorm),
                ));
                let raw = matvec(&dsa(DsaRole::K), INDEX_DIM, &row);
                let mean = raw.iter().sum::<f64>() / INDEX_DIM as f64;
                let var = raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / INDEX_DIM as f64;
                let (kn, kb) = (dsa(DsaRole::KNorm), dsa(DsaRole::KNormBias));
                keys.push(
                    (0..INDEX_DIM)
                        .map(|j| {
                            (raw[j] - mean) / (var + 1e-6).sqrt() * f64::from(kn[j])
                                + f64::from(kb[j])
                        })
                        .collect(),
                );
                gates.push(matvec(&gate_w, INDEX_DIM, &row));

                // The pooled keys of the pools complete at this position, scored.
                let complete = (p + 1) / POOL;
                let q_idx = matvec(&dsa(DsaRole::Q), INDEX_HEADS * INDEX_DIM, &q_res);
                let head_w = matvec(&dsa(DsaRole::Weights), INDEX_HEADS, &row);
                let scores: Vec<f64> = (0..complete)
                    .map(|j| {
                        let pooled: Vec<f64> = (0..INDEX_DIM)
                            .map(|f| {
                                let logits: Vec<f64> = (0..POOL)
                                    .map(|l| {
                                        gates[j * POOL + l][f] + f64::from(ape[l * INDEX_DIM + f])
                                    })
                                    .collect();
                                let max = logits.iter().cloned().fold(f64::MIN, f64::max);
                                let exps: Vec<f64> =
                                    logits.iter().map(|l| (l - max).exp()).collect();
                                let sum: f64 = exps.iter().sum();
                                (0..POOL)
                                    .map(|l| exps[l] / sum * keys[j * POOL + l][f])
                                    .sum()
                            })
                            .collect();
                        (0..INDEX_HEADS)
                            .map(|h| {
                                let dot: f64 = (0..INDEX_DIM)
                                    .map(|f| q_idx[h * INDEX_DIM + f] * pooled[f])
                                    .sum::<f64>()
                                    / (INDEX_DIM as f64).sqrt();
                                dot.max(0.0) * head_w[h] / (INDEX_HEADS as f64).sqrt()
                            })
                            .sum()
                    })
                    .collect();
                let mut order: Vec<usize> = (0..complete).collect();
                order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                let mut selected: Vec<usize> = order[..topk_pools.min(complete)]
                    .iter()
                    .flat_map(|&j| (0..POOL).map(move |l| j * POOL + l))
                    .collect();
                selected.extend(complete * POOL..=p);

                let kv_b = mla(MlaRole::KvB);
                let up: Vec<Vec<f64>> = selected
                    .iter()
                    .map(|&s| matvec(&kv_b, HEADS * (QK + V), &latents[s]))
                    .collect();
                let mut attended = vec![0.0; HEADS * V];
                for h in 0..HEADS {
                    let at =
                        |u: &'_ [f64]| -> Vec<f64> { u[h * (QK + V)..(h + 1) * (QK + V)].to_vec() };
                    let logits: Vec<f64> = up
                        .iter()
                        .map(|u| {
                            (0..QK).map(|f| q[h * QK + f] * at(u)[f]).sum::<f64>()
                                / (QK as f64).sqrt()
                        })
                        .collect();
                    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
                    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    for (u, e) in up.iter().zip(&exps) {
                        for f in 0..V {
                            attended[h * V + f] += e / sum * at(u)[QK + f];
                        }
                    }
                }
                out.extend(matvec(&v(WeightRole::Attn(AttnRole::O)), WIDTH, &attended));
            }
            let pad = |rows: Vec<Vec<f64>>, width: usize| {
                let mut cache = vec![0.0; CAPACITY * width];
                for (p, row) in rows.iter().enumerate() {
                    cache[p * width..(p + 1) * width].copy_from_slice(row);
                }
                cache
            };
            let index_rows: Vec<Vec<f64>> = keys
                .iter()
                .zip(&gates)
                .map(|(k, g)| [k.clone(), g.clone()].concat())
                .collect();
            (out, pad(latents, KV_RANK), pad(index_rows, 2 * INDEX_DIM))
        }

        fn inputs() -> Vec<f32> {
            oracle::random(TOKENS * WIDTH, 51)
                .into_iter()
                .map(|v| 3.0 * v)
                .collect()
        }

        /// SC-008: the pooled DSA layer equals the f64 definition at every output and both carried
        /// caches (whole, in two chunks and token by token), with the selection live (two of up to
        /// five complete pools are kept, and the tail positions join them), and each weight changes
        /// the output. Mutation: an off-by-one in the kept pool count (`ops::block_top_k_packed` over
        /// `pool_topk + 1`); a third pool joins the attention and the whole-run rows go red.
        #[test]
        fn the_pooled_layer_matches_the_definition_and_continues_across_chunks() {
            let fx = fixture();
            let x = inputs();
            let (want, latent, index) = reference(&fx, POOL_TOPK, &x);
            let check = |run: &Run| {
                oracle::assert_matches_f64(&run.state[0], &latent, 1e-4);
                oracle::assert_matches_f64(&run.state[1], &index, 1e-4);
            };
            let whole = run(&fx, POOL_TOPK, &x, 0, &[]);
            oracle::assert_matches_f64(&whole.out, &want, 1e-4);
            check(&whole);

            let first = run(&fx, POOL_TOPK, &x[..5 * WIDTH], 0, &[]);
            let second = run(&fx, POOL_TOPK, &x[5 * WIDTH..], 5, &first.state);
            oracle::assert_matches_f64(
                &[first.out.clone(), second.out.clone()].concat(),
                &want,
                1e-4,
            );
            check(&second);

            let mut steps = run(&fx, POOL_TOPK, &x[..WIDTH], 0, &[]);
            let mut stepped = steps.out.clone();
            for t in 1..TOKENS {
                steps = run(
                    &fx,
                    POOL_TOPK,
                    &x[t * WIDTH..(t + 1) * WIDTH],
                    t as i32,
                    &steps.state,
                );
                stepped.extend(&steps.out);
            }
            oracle::assert_matches_f64(&stepped, &want, 1e-4);
            check(&steps);

            // A weight of the indexer moves the output only when it reorders the pools (a discrete
            // choice), so a perturbation is repeated, growing, until the ranking moves.
            for (r, _) in roles() {
                let mut perturbed = fx.clone();
                let moved = (0..12).any(|_| {
                    perturbed.perturb(role(r));
                    let out = run(&perturbed, POOL_TOPK, &x, 0, &[]).out;
                    out.iter()
                        .zip(&whole.out)
                        .any(|(a, b)| (a - b).abs() > 1e-6)
                });
                assert!(moved, "the output does not depend on {}", role(r));
            }
        }

        /// The selection decides the output: keeping every complete pool gives other outputs than
        /// keeping two, so a layer that attended everything could not pass the row above.
        #[test]
        fn the_selection_is_live() {
            let fx = fixture();
            let x = inputs();
            let narrow = run(&fx, POOL_TOPK, &x, 0, &[]).out;
            let wide = run(&fx, CAPACITY / POOL, &x, 0, &[]).out;
            oracle::assert_differs(&narrow, &wide, role(WeightRole::Dsa(DsaRole::Q)));
        }

        #[test]
        fn params_refuse_a_budget_that_is_not_whole_pools() {
            let norm = NormParams::new(EPS).unwrap();
            assert_eq!(
                PooledDsaParams::new(HEADS, Q_RANK, KV_RANK, QK, V, norm, 2, 4, 5, 2),
                Err(ParamError::new(
                    "index_topk",
                    ConfigReason::NotDivisible { by: 2 }
                ))
            );
            assert_eq!(
                PooledDsaParams::new(HEADS, Q_RANK, KV_RANK, QK, V, norm, 2, 4, 4, 0),
                Err(ParamError::new("index_kpool", ConfigReason::Zero))
            );
        }
    }

    #[test]
    fn params_refuse_an_impossible_indexer() {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain).unwrap();
        assert_eq!(
            DsaParams::new(0, DIM, 2, rope),
            Err(ParamError::new("index_n_heads", ConfigReason::Zero))
        );
        assert_eq!(
            DsaParams::new(HEADS, 1, 2, rope),
            Err(ParamError::new(
                "index_head_dim",
                ConfigReason::Exceeds { max: ROPE }
            ))
        );
    }
}
