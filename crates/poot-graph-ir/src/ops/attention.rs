use super::*;

/// GQA repeat_kv: broadcast each of H_kv heads to n_rep query heads.
/// `k[bsz,H_kv,S,D]` -> `[bsz,H_kv*n_rep,S,D]`.
pub fn repeat_kv(b: &Builder, k: Traced, n_rep: usize) -> Traced {
    if n_rep == 1 {
        return k;
    }
    let s = b.aval(k).shape;
    let (bsz, h_kv, seq, d) = (s[0], s[1], s[2], s[3]);
    let r = b.reshape(k, vec![bsz, h_kv, 1, seq, d]);
    let r = b.broadcast(r, vec![bsz, h_kv, n_rep, seq, d]);
    b.reshape(r, vec![bsz, h_kv * n_rep, seq, d])
}

/// Tiled GQA repeat: `[bsz, H_kv, seq, D] -> [bsz, H_kv*n_rep, seq, D]` where output head `h` maps to
/// kv-head `h % H_kv` (the kv-head block is tiled `n_rep` times: `[k0..k15, k0..k15]`). Qwen3-Next's
/// Gated-DeltaNet uses this to expand `num_k_heads` q/k to `num_v_heads` (llama.cpp
/// `ggml_repeat_4d`), distinct from the blocked [`repeat_kv`] (`h -> h / n_rep`) used by standard GQA.
/// Inserting the `n_rep` dim before the head dim gives the tile.
pub fn repeat_kv_tiled(b: &Builder, k: Traced, n_rep: usize) -> Traced {
    if n_rep == 1 {
        return k;
    }
    let s = b.aval(k).shape;
    let (bsz, h_kv, seq, d) = (s[0], s[1], s[2], s[3]);
    let r = b.reshape(k, vec![bsz, 1, h_kv, seq, d]);
    let r = b.broadcast(r, vec![bsz, n_rep, h_kv, seq, d]);
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
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D,L]
    let scores = b.matmul(q, kt); // [1,Hq,L,L]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale));
    let scores = b.binary(BinOp::Add, scores, mask); // causal/windowed mask, broadcast over Hq
    let last = b.aval(scores).rank() - 1;
    let m_scores = b.reduce(RedOp::Max, scores, last, true); // [1,Hq,L,1]
    let sinks_b = b.broadcast(b.reshape(sinks, vec![1, hq, 1, 1]), vec![1, hq, l, 1]);
    let m = b.binary(BinOp::Max, m_scores, sinks_b); // row max INCLUDES the sink logit
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom_scores = b.reduce(RedOp::Sum, e, last, true);
    let sink_shifted = b.binary(BinOp::Sub, sinks_b, m);
    let e_sink = b.unary(UnOp::Exp, sink_shifted);
    let denom = b.binary(BinOp::Add, denom_scores, e_sink); // sink's mass counted, never gathered into V
    let p = b.binary(BinOp::Div, e, denom);
    b.matmul(p, v) // [1,Hq,L,D]
}

/// [`attention_prefill_with_sink`]'s single-query decode analog (`L=1`, `mask[1,1,1,cap]`).
#[allow(clippy::too_many_arguments)]
pub fn attention_masked_with_sink(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    sinks: Traced,
    hq: usize,
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D,cap]
    let scores = b.matmul(q, kt); // [1,Hq,1,cap]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale));
    let scores = b.binary(BinOp::Add, scores, mask);
    let last = b.aval(scores).rank() - 1;
    let m_scores = b.reduce(RedOp::Max, scores, last, true); // [1,Hq,1,1]
    let sinks_b = b.reshape(sinks, vec![1, hq, 1, 1]); // L=1: no broadcast needed
    let m = b.binary(BinOp::Max, m_scores, sinks_b);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom_scores = b.reduce(RedOp::Sum, e, last, true);
    let sink_shifted = b.binary(BinOp::Sub, sinks_b, m);
    let e_sink = b.unary(UnOp::Exp, sink_shifted);
    let denom = b.binary(BinOp::Add, denom_scores, e_sink);
    let p = b.binary(BinOp::Div, e, denom);
    b.matmul(p, v) // [1,Hq,1,D]
}

/// Decode attention (single query token), GQA, no mask (l=1 attends all cached keys). `q[1,Hq,1,D]`,
/// `k`/`v[1,Hkv,S,D]`. Returns `[1,Hq,1,D]`.
pub fn attention(b: &Builder, q: Traced, k: Traced, v: Traced, n_rep: usize, scale: f32) -> Traced {
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
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D,S]
    let scores = b.matmul(q, kt); // [1,Hq,1,S]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale));
    let scores = match attn_logit_softcap {
        Some(c) => softcap(b, scores, c),
        None => scores,
    };
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = b.binary(BinOp::Div, e, denom);
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
) -> Traced {
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
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D,cap]
    let scores = b.matmul(q, kt); // [1,Hq,1,cap]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale));
    let scores = match attn_logit_softcap {
        Some(c) => softcap(b, scores, c),
        None => scores,
    };
    let scores = b.binary(BinOp::Add, scores, mask); // additive mask, broadcast over Hq
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = b.binary(BinOp::Div, e, denom);
    b.matmul(p, v) // [1,Hq,1,D]
}
