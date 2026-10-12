use super::{BinOp, Builder, BuilderError, DType, RedOp, Scalar, Traced, UnOp, softcap};
use crate::error::Requirement;

/// GQA repeat_kv: broadcast each of H_kv heads to n_rep query heads.
/// `k[bsz,H_kv,S,D]` -> `[bsz,H_kv*n_rep,S,D]`.
pub fn repeat_kv(b: &Builder, k: Traced, n_rep: usize) -> Result<Traced, BuilderError> {
    if n_rep == 1 {
        return Ok(k);
    }
    let s = b.aval(k)?.shape;
    let (bsz, h_kv, seq, d) = (s[0], s[1], s[2], s[3]);
    let r = b.reshape(k, vec![bsz, h_kv, 1, seq, d])?;
    let r = b.broadcast(r, vec![bsz, h_kv, n_rep, seq, d])?;
    b.reshape(r, vec![bsz, h_kv * n_rep, seq, d])
}

/// Tiled GQA repeat: `[bsz, H_kv, seq, D] -> [bsz, H_kv*n_rep, seq, D]` where output head `h` maps to
/// kv-head `h % H_kv` (the kv-head block is tiled `n_rep` times: `[k0..k15, k0..k15]`). Qwen3-Next's
/// Gated-DeltaNet uses this to expand `num_k_heads` q/k to `num_v_heads` (llama.cpp
/// `ggml_repeat_4d`), distinct from the blocked [`repeat_kv`] (`h -> h / n_rep`) used by standard GQA.
/// Inserting the `n_rep` dim before the head dim gives the tile.
pub fn repeat_kv_tiled(b: &Builder, k: Traced, n_rep: usize) -> Result<Traced, BuilderError> {
    if n_rep == 1 {
        return Ok(k);
    }
    let s = b.aval(k)?.shape;
    let (bsz, h_kv, seq, d) = (s[0], s[1], s[2], s[3]);
    let r = b.reshape(k, vec![bsz, 1, h_kv, seq, d])?;
    let r = b.broadcast(r, vec![bsz, n_rep, h_kv, seq, d])?;
    b.reshape(r, vec![bsz, n_rep * h_kv, seq, d])
}

/// Attention-sink prefill core (gpt-oss spec 263, promoted for DeepSeek-V4 reuse in spec 281;
/// derivation in `crates/poot-models/src/gpt_oss.rs`'s module doc): standard scaled-dot-product
/// causal/windowed attention, but each query row's softmax also includes one extra
/// position-independent per-head learned logit (`sinks[h]`) that contributes to the row max and
/// softmax denominator but never to the weighted V sum. `sinks` is `[Hq]`. Implemented without a
/// concat op (the IR has none): folding the sink into the max/sum over the real columns (two extra
/// `Max`/`Add`/`Exp` steps) equals concatenating a sink column.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_with_sink(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    sinks: Traced,
    hq: usize,
    l: usize,
) -> Result<Traced, BuilderError> {
    let k = repeat_kv(b, k, n_rep)?;
    let v = repeat_kv(b, v, n_rep)?;
    let kt = b.transpose(k, vec![0, 1, 3, 2])?; // [1,Hq,D,L]
    let scores = b.matmul(q, kt)?; // [1,Hq,L,L]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale))?;
    let scores = b.binary(BinOp::Add, scores, mask)?; // causal/windowed mask, broadcast over Hq
    let last = b.aval(scores)?.rank() - 1;
    let m_scores = b.reduce(RedOp::Max, scores, last, true)?; // [1,Hq,L,1]
    let sinks_b = b.broadcast(b.reshape(sinks, vec![1, hq, 1, 1])?, vec![1, hq, l, 1])?;
    let m = b.binary(BinOp::Max, m_scores, sinks_b)?; // row max INCLUDES the sink logit
    let shifted = b.binary(BinOp::Sub, scores, m)?;
    let e = b.unary(UnOp::Exp, shifted)?;
    let denom_scores = b.reduce(RedOp::Sum, e, last, true)?;
    let sink_shifted = b.binary(BinOp::Sub, sinks_b, m)?;
    let e_sink = b.unary(UnOp::Exp, sink_shifted)?;
    let denom = b.binary(BinOp::Add, denom_scores, e_sink)?; // sink's mass counted, never gathered into V
    let p = b.binary(BinOp::Div, e, denom)?;
    b.matmul(p, v) // [1,Hq,L,D]
}

/// Decode attention (single query token), GQA, no mask (l=1 attends all cached keys). `q[1,Hq,1,D]`,
/// `k`/`v[1,Hkv,S,D]`. Returns `[1,Hq,1,D]`.
pub fn attention(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
) -> Result<Traced, BuilderError> {
    attention_softcap(b, q, k, v, n_rep, scale, None)
}

/// [`attention`] with optional Gemma2/Grok attention-logit softcapping. `attn_logit_softcap = Some(c)`
/// applies `c * tanh(scores / c)` to the scaled `Q K^T` scores before softmax ([`softcap`]); `None` is
/// byte-identical to [`attention`]. One definition.
pub fn attention_softcap(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    attn_logit_softcap: Option<f32>,
) -> Result<Traced, BuilderError> {
    let k = repeat_kv(b, k, n_rep)?;
    let v = repeat_kv(b, v, n_rep)?;
    let kt = b.transpose(k, vec![0, 1, 3, 2])?; // [1,Hq,D,S]
    let scores = b.matmul(q, kt)?; // [1,Hq,1,S]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale))?;
    let scores = match attn_logit_softcap {
        Some(c) => softcap(b, scores, c)?,
        None => scores,
    };
    let last = b.aval(scores)?.rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true)?;
    let shifted = b.binary(BinOp::Sub, scores, m)?;
    let e = b.unary(UnOp::Exp, shifted)?;
    let denom = b.reduce(RedOp::Sum, e, last, true)?;
    let p = b.binary(BinOp::Div, e, denom)?;
    b.matmul(p, v) // [1,Hq,1,D]
}

/// Decode attention over the full fixed-capacity cache with an additive mask (G3d). Like [`attention`]
/// but reads all `cap` cached slots and adds `mask` before softmax, so the shape is constant across
/// positions (one captured graph replays every token). `q[1,Hq,1,D]`, `k`/`v[1,Hkv,cap,D]`,
/// `mask[1,1,1,cap]` (0 for valid key slots `t <= pos`, large-negative for `t > pos`, broadcast over Hq).
/// Returns `[1,Hq,1,D]`. Masked slots get ~0 weight after softmax, so this equals sliced attention over
/// `[0..=pos]` (FR-001). One definition shared by the tracer + eager eval.
///
/// `mask` may also be `[1,Hq,1,cap]` (one row per head), which is how ALiBi's per-head bias is wired in
/// (spec 254, `poot_llm::graphs::alibi_decode_mask_row`): the visibility term and
/// `-slope[h] * (query_pos - key_pos)` are summed host-side into one mask input.
pub fn attention_masked(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
) -> Result<Traced, BuilderError> {
    attention_masked_softcap(b, q, k, v, n_rep, scale, mask, None)
}

/// [`attention_masked`] with optional Gemma2/Grok attention-logit softcapping. `Some(c)` applies
/// `c * tanh(scores / c)` ([`softcap`]) to the scaled `Q K^T` scores BEFORE the additive mask and softmax
/// (the Gemma2 order); `None` is byte-identical to [`attention_masked`] (no extra eqns). One definition
/// across tracing, the oracle and every lowering.
#[allow(clippy::too_many_arguments)]
pub fn attention_masked_softcap(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    attn_logit_softcap: Option<f32>,
) -> Result<Traced, BuilderError> {
    attention_over_keys(
        b,
        q,
        AttentionKeys::Single(k),
        v,
        n_rep,
        scale,
        mask,
        attn_logit_softcap,
        None,
    )
}

/// The `[B,Hkv,Lk,D]` view of the `[S,Hkv,D]` KV `pool` through the `[B,Lk]` I32 slot map `read`: position `t`
/// of batch row `b` is pool slot `read[b,t]`. It is the one definition of a paged cache's logical view: the
/// models trace it, and [`crate::KeyLayout::Paged`] attention is its fusion into the attention kernel, which
/// reads the pool through `read` instead of materializing the view.
pub fn paged_view(b: &Builder, pool: Traced, read: Traced) -> Result<Traced, BuilderError> {
    let rank = |expected: usize, shape: Vec<usize>| BuilderError::Requirement {
        operation: "paged view",
        requirement: Requirement::Rank {
            expected,
            found: shape,
        },
    };
    let read_shape = b.aval(read)?.shape.clone();
    let &[rows, cap] = read_shape.as_slice() else {
        return Err(rank(2, read_shape));
    };
    let pool_shape = b.aval(pool)?.shape.clone();
    let &[_, hkv, d] = pool_shape.as_slice() else {
        return Err(rank(3, pool_shape));
    };
    let slots = b.reshape(read, vec![rows * cap])?;
    let rows_first = b.reshape(b.gather(pool, 0, slots)?, vec![rows, cap, hkv, d])?;
    b.transpose(rows_first, vec![0, 2, 1, 3])
}

/// The key operand of the attention composite: one dense tensor, or the two parts of a split key.
#[derive(Clone, Copy, Debug)]
pub enum AttentionKeys {
    /// `[B,Hkv,Lk,D]`.
    Single(Traced),
    /// `head [B,Hkv,Lk,D - s]` and `shared [B,1,Lk,s]`: the key is `concat(head, broadcast(shared))` on the
    /// last axis (MLA's `k_nope` and rotary `k_pe`).
    Split {
        /// The per-head part of the key.
        head: Traced,
        /// The part shared across heads, broadcast over `Hkv` and concatenated after `head`.
        shared: Traced,
    },
}

/// What [`attention_live`] returns: the attention output and the length-validation witness the caller
/// declares as a validation output.
#[derive(Clone, Copy, Debug)]
#[cfg(any(test, feature = "test-support"))]
pub struct LiveAttention {
    /// `[B,Hq,Lq,Dv]`.
    pub out: Traced,
    /// Zero when every `len[b]` was in `1..=Lk`; otherwise an executor publishes nothing.
    pub witness: Traced,
}

/// Live-capacity attention: the attention composite over the first `len[b]` of the `Lk` key positions of batch
/// row `b`, with `len` a raw I32 `[B]` that must lie in `1..=Lk`. The range is checked by the canonical
/// index-bounds guard ([`Builder::guard_index_bounds`] on `len - 1`), so an out-of-range length publishes
/// nothing through the validation channel and the graph reads a valid in-range fallback.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "test-support"))]
pub fn attention_live(
    b: &Builder,
    q: Traced,
    keys: AttentionKeys,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    attn_logit_softcap: Option<f32>,
    len: Traced,
) -> Result<LiveAttention, crate::index_guard::IndexGuardError> {
    let lk = b.aval(v)?.shape[2];
    let last = b.binary_scalar(BinOp::Sub, len, Scalar::I32(1))?;
    let guard = b.guard_index_bounds(last, lk)?;
    let live_len = b.binary_scalar(BinOp::Add, guard.guarded, Scalar::I32(1))?;
    let out = attention_over_keys(
        b,
        q,
        keys,
        v,
        n_rep,
        scale,
        mask,
        attn_logit_softcap,
        Some(live_len),
    )?;
    Ok(LiveAttention {
        out,
        witness: guard.witness,
    })
}

/// The one attention definition behind [`attention_masked_softcap`] and [`attention_live`], and the
/// decomposition of `OpKind::Attention` over every key layout and bound.
///
/// With `live_len == None` every key position counts and the mask removes the ones that must not. With
/// `Some(len)` (`len [B]` I32, already in `1..=Lk`) key `j` of batch row `b` is active when `j < len[b]`:
/// inactive K and V rows are *selected* to zero before either matmul (a float `Select`, never a multiplication
/// by zero, so a NaN there cannot survive) and inactive scores are selected to `-inf` after the mask is
/// added, so inactive positions weigh exactly zero. Both parts of a split key are guarded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_over_keys(
    b: &Builder,
    q: Traced,
    keys: AttentionKeys,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    attn_logit_softcap: Option<f32>,
    live_len: Option<Traced>,
) -> Result<Traced, BuilderError> {
    let live = match live_len {
        Some(len) => {
            let v_type = b.aval(v)?;
            Some(LiveKeys::new(b, len, v_type.shape[2], v_type.dtype)?)
        }
        None => None,
    };
    let guard_rows = |x: Traced| match &live {
        Some(live) => b.select(live.rows, x, live.zero_rows),
        None => Ok(x),
    };
    let k = match keys {
        AttentionKeys::Single(k) => guard_rows(k)?,
        AttentionKeys::Split { head, shared } => {
            let head = guard_rows(head)?;
            let shared = guard_rows(shared)?;
            // The shared part reaches every key head through the same `repeat_kv` chain a per-head key does.
            let shared = repeat_kv(b, shared, b.aval(head)?.shape[1])?;
            b.concat(3, &[head, shared])?
        }
    };
    let v = guard_rows(v)?;
    let k = repeat_kv(b, k, n_rep)?;
    let v = repeat_kv(b, v, n_rep)?;
    let kt = b.transpose(k, vec![0, 1, 3, 2])?; // [B,Hq,D,Lk]
    let scores = b.matmul(q, kt)?; // [B,Hq,Lq,Lk]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale))?;
    let scores = match attn_logit_softcap {
        Some(c) => softcap(b, scores, c)?,
        None => scores,
    };
    let scores = b.binary(BinOp::Add, scores, mask)?; // additive mask, broadcast over Hq
    let scores = match &live {
        Some(live) => b.select(live.columns, scores, live.neg_inf_columns)?,
        None => scores,
    };
    let last = b.aval(scores)?.rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true)?;
    let shifted = b.binary(BinOp::Sub, scores, m)?;
    let e = b.unary(UnOp::Exp, shifted)?;
    let denom = b.reduce(RedOp::Sum, e, last, true)?;
    let p = b.binary(BinOp::Div, e, denom)?;
    b.matmul(p, v) // [B,Hq,Lq,Dv]
}

/// The activity conditions and select fallbacks of a live bound, built once from `len [B]`.
struct LiveKeys {
    /// `[B,1,Lk,1]` I32: key row `j` of batch row `b` is active (`j < len[b]`).
    rows: Traced,
    /// `[B,1,1,Lk]` I32: the same condition laid along the score row.
    columns: Traced,
    /// `[1,1,Lk,1]` finite zeros in the operands' dtype, the fallback for an inactive K or V row.
    zero_rows: Traced,
    /// `[1,1,1,Lk]` `-inf` in the operands' dtype, the fallback for an inactive score.
    neg_inf_columns: Traced,
}

impl LiveKeys {
    fn new(b: &Builder, len: Traced, lk: usize, dtype: DType) -> Result<Self, BuilderError> {
        let batch = b.aval(len)?.shape[0];
        let len = b.reshape(len, vec![batch, 1, 1, 1])?;
        // `j < len` is `len >= j + 1`; the key ends `1..=Lk` are built exactly in I32, with no float round trip.
        let ends = b.binary_scalar(BinOp::Add, b.iota_i32(lk)?, Scalar::I32(1))?; // [Lk] i32
        // The select fallbacks come from the F32 range scaled by zero, so they are finite zeros whatever the
        // scalar, and `-inf` is added to a zero rather than multiplied into a possible `0 * inf`.
        let zero = b.binary_scalar(BinOp::Mul, b.iota(lk)?, Scalar::F32(0.0))?;
        let neg_inf = b.binary_scalar(BinOp::Add, zero, Scalar::F32(f32::NEG_INFINITY))?;
        // A select's branches share one dtype: zero and `-inf` are exact in every float dtype.
        let (zero, neg_inf) = if dtype == DType::F32 {
            (zero, neg_inf)
        } else {
            (b.cast(zero, dtype)?, b.cast(neg_inf, dtype)?)
        };
        Ok(Self {
            rows: b.binary(BinOp::Ge, len, b.reshape(ends, vec![1, 1, lk, 1])?)?,
            columns: b.binary(BinOp::Ge, len, b.reshape(ends, vec![1, 1, 1, lk])?)?,
            zero_rows: b.reshape(zero, vec![1, 1, lk, 1])?,
            neg_inf_columns: b.reshape(neg_inf, vec![1, 1, 1, lk])?,
        })
    }
}
