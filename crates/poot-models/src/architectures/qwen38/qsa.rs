use super::*;
use poot_graph_ir::ops::{stable_descending_rank, top_k_keep_mask};

/// Large additive penalty for "not selected" / "not yet valid" mask positions: `-1e30`, as
/// `crate::deepseek3::deepseek3_router_gate` and `crate::deepseek32`'s `DSA_MASK_NEG` use (zeroes out after
/// softmax, and summing two never overflows `f32`).
pub(crate) const QSA_MASK_NEG: f32 = -1e30;

/// Half-split ("rotate-half") partial RoPE table, `[max_pos, rot]`, each frequency duplicated at `i` and
/// `half+i`, the layout `poot_graph_ir::ops::rope`/`rope_prefill` expect. The real text config has no
/// `rope_scaling`/YaRN entry, so this is the plain formula (unlike
/// `crate::deepseek32::dsa_indexer_rope_tables`, which threads an optional YaRN table).
#[cfg(test)]
pub(crate) fn qwen38_rope_tables(max_pos: usize, rot: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rot / 2;
    let mut cos = vec![0.0f32; max_pos * rot];
    let mut sin = vec![0.0f32; max_pos * rot];
    for pos in 0..max_pos {
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f32 / rot as f32);
            let ang = pos as f32 * inv_freq;
            let (s, c) = ang.sin_cos();
            cos[pos * rot + i] = c;
            cos[pos * rot + half + i] = c;
            sin[pos * rot + i] = s;
            sin[pos * rot + half + i] = s;
        }
    }
    (cos, sin)
}

/// Same half-split RoPE table as [`qwen38_rope_tables`], sampled at micro-block start positions
/// `0, C, 2C, ..`: the indexer's block-key table (pooled keys are RoPE'd at their block's first-token
/// position, spec 282 step 3).
#[cfg(test)]
pub(crate) fn qwen38_block_rope_tables(
    num_blocks: usize,
    compress_ratio: usize,
    rot: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = rot / 2;
    let mut cos = vec![0.0f32; num_blocks * rot];
    let mut sin = vec![0.0f32; num_blocks * rot];
    for b in 0..num_blocks {
        let pos = (b * compress_ratio) as f32;
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f32 / rot as f32);
            let ang = pos * inv_freq;
            let (s, c) = ang.sin_cos();
            cos[b * rot + i] = c;
            cos[b * rot + half + i] = c;
            sin[b * rot + i] = s;
            sin[b * rot + half + i] = s;
        }
    }
    (cos, sin)
}

/// Additive plain causal mask `[1,1,L,L]`: `0.0` on/below the diagonal, [`QSA_MASK_NEG`] above.
#[cfg(test)]
pub(crate) fn causal_mask_data(l: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            m[i * l + j] = if j <= i { 0.0 } else { QSA_MASK_NEG };
        }
    }
    m
}

/// Additive block-eligibility mask `[1,1,L,NB]` (spec 282 FR-002): `0.0` if block `b` (positions
/// `[b*C, b*C+C-1]`) is fully causally complete as of query `i` (`b*C + C - 1 <= i`), else
/// [`QSA_MASK_NEG`].
#[cfg(test)]
pub(crate) fn block_eligible_mask_data(
    l: usize,
    num_blocks: usize,
    compress_ratio: usize,
) -> Vec<f32> {
    let mut m = vec![0.0f32; l * num_blocks];
    for i in 0..l {
        for b in 0..num_blocks {
            let block_last = b * compress_ratio + compress_ratio - 1;
            m[i * num_blocks + b] = if block_last <= i { 0.0 } else { QSA_MASK_NEG };
        }
    }
    m
}

/// Additive tail-force-visible mask `[1,1,L,L]` (spec 282 FR-003): `0.0` for a token `j` in query `i`'s
/// own still-incomplete block, else [`QSA_MASK_NEG`].
#[cfg(test)]
pub(crate) fn tail_mask_data(l: usize, compress_ratio: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; l * l];
    for i in 0..l {
        let block_complete = (i + 1).is_multiple_of(compress_ratio);
        for j in 0..l {
            let same_block = j / compress_ratio == i / compress_ratio;
            m[i * l + j] = if same_block && j <= i && !block_complete {
                0.0
            } else {
                QSA_MASK_NEG
            };
        }
    }
    m
}

/// Mean-pool the indexer's raw (unrotated, unnormalized) keys into fixed `compress_ratio`-token
/// micro-blocks, then apply the indexer's `k_layernorm` and partial RoPE at each block's first-token
/// position (spec 282 step 3: pool, then norm, then rope).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_pool_block_keys(
    b: &Builder,
    raw_k: Traced, // [1, L, Di]
    k_ln_w: Traced,
    block_cos: Traced,
    block_sin: Traced,
    l: usize,
    c: usize,
    di: usize,
    eps: f32,
) -> Traced {
    let nb = l / c;
    let grouped = b.reshape(raw_k, vec![1, nb, c, di]);
    let summed = b.reduce(RedOp::Sum, grouped, 2, true); // [1, NB, 1, Di]
    let pooled = b.binary_scalar(BinOp::Mul, summed, Scalar::F32(1.0 / c as f32));
    let pooled = b.reshape(pooled, vec![1, nb, di]);
    let normed = rmsnorm(b, pooled, k_ln_w, eps);
    let normed4 = b.reshape(normed, vec![1, 1, nb, di]);
    rope_prefill(b, normed4, block_cos, block_sin, nb) // [1, 1, NB, Di]
}

/// QSA indexer score (spec 282 step 4): `sum_h ReLU(q_h . k_block) / sqrt(Di)`, unweighted over the `Hi`
/// indexer heads (real QSA has no analog of DSA's `weights_proj`). Returns `[1,1,Lq,NB]`.
pub(crate) fn qsa_indexer_block_scores(
    b: &Builder,
    q_idx: Traced,   // [1, Hi, Lq, Di] - already RMSNorm'd + RoPE'd at its own position
    block_k: Traced, // [1, 1, NB, Di] - pooled, normed, RoPE'd at block-start position (shared, MQA)
    hi: usize,
    di: usize,
) -> Traced {
    let k_b = repeat_kv(b, block_k, hi); // [1, Hi, NB, Di]
    let k_t = b.transpose(k_b, vec![0, 1, 3, 2]); // [1, Hi, Di, NB]
    let dots = b.matmul(q_idx, k_t); // [1, Hi, Lq, NB]
    let scored = relu(b, dots);
    let summed = b.reduce(RedOp::Sum, scored, 1, true); // [1, 1, Lq, NB]
    let scale = 1.0 / (di as f32).sqrt();
    b.binary_scalar(BinOp::Mul, summed, Scalar::F32(scale))
}

/// Keep mask (`1.0` kept, `0.0` dropped) for exactly `block_topk` blocks along the trailing (block) axis of
/// `ranked`: the highest scores first, and on an exact tie the lower block index. The budget is exact even
/// on ties. ReLU-summed indexer scores tie often, at `0.0` whenever every indexer head's dot product with a
/// block is negative. A shared-rank construction would admit every block tied at the cut, so one query
/// could attend to more than `index_budget` tokens.
fn qsa_block_top_k(b: &Builder, ranked: Traced, block_topk: usize) -> Traced {
    let rank = stable_descending_rank(b, ranked);
    top_k_keep_mask(b, rank, block_topk)
}

/// Combine QSA's block-level top-k selection with the always-visible tail and the plain causal mask into
/// one additive `[1,1,L,L]` mask for [`qwen3next_gated_attention_prefill`]'s `mask` argument (spec 282
/// FR-002..FR-005; the module's central composition).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_combined_mask(
    b: &Builder,
    index_scores: Traced,   // [1,1,L,NB]
    block_eligible: Traced, // [1,1,L,NB]
    causal_mask: Traced,    // [1,1,L,L]
    tail_mask: Traced,      // [1,1,L,L]
    block_topk: usize,
    l: usize,
    nb: usize,
    c: usize,
) -> Traced {
    assert_eq!(
        nb * c,
        l,
        "seq_len must be an exact multiple of compress_ratio (spec 282 Out-of-scope)"
    );
    // Rank the sum (not the raw score): an ineligible block's -1e30 term keeps it from outranking an
    // eligible one.
    let ranked = b.binary(BinOp::Add, index_scores, block_eligible);
    let keep = qsa_block_top_k(b, ranked, block_topk); // 1.0 kept, 0.0 dropped
    let shifted = b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0)); // 0 kept, -1 dropped
    let topk_add = b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-QSA_MASK_NEG));
    // Re-AND eligibility after top-k: when block_topk exceeds the eligible count, the rank fills the
    // remaining budget with ineligible blocks.
    let block_selected = b.binary(BinOp::Add, block_eligible, topk_add); // [1,1,L,NB]

    // Expand the block decision to token granularity: token j inherits block floor(j/C)'s decision.
    let expanded = b.reshape(block_selected, vec![1, 1, l, nb, 1]);
    let expanded = b.broadcast(expanded, vec![1, 1, l, nb, c]);
    let expanded = b.reshape(expanded, vec![1, 1, l, l]);

    // OR with the tail: Max, not Add (see the module docs).
    let selection = b.binary(BinOp::Max, expanded, tail_mask);

    // AND with the plain causal mask (FR-005): both terms are `{0.0, MASK_NEG}`-valued, so plain Add is AND.
    b.binary(BinOp::Add, selection, causal_mask)
}

/// One QSA (full-attention) layer's prefill forward: build the indexer's combined mask, then feed it into
/// [`qwen3next_gated_attention_prefill`] unchanged (no new attention primitive, only a different mask
/// input; spec 282). `x` is the layer's already-normed input. Returns
/// `(layer_output, k_cache_out, v_cache_out)`, as [`qwen3next_gated_attention_prefill`].
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_prefill_layer(
    b: &Builder,
    x: Traced,
    layer_prefix: &str,
    cfg: Qwen4ExpConfig,
    qcfg: QsaConfig,
    l: usize,
    cos: Traced,
    sin: Traced,
    block_cos: Traced,
    block_sin: Traced,
    causal_mask: Traced,
    block_eligible: Traced,
    tail_mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
) -> (Traced, Traced, Traced) {
    let (out, kc, vc, _raw_k) = qsa_attention_prefill_layer_with_raw_k(
        b,
        x,
        layer_prefix,
        cfg,
        qcfg,
        l,
        cos,
        sin,
        block_cos,
        block_sin,
        causal_mask,
        block_eligible,
        tail_mask,
        k_cache_in,
        v_cache_in,
    );
    (out, kc, vc)
}

/// [`qsa_attention_prefill_layer`] plus the indexer's raw keys `[1, 1, L, Di]` (the row the decode
/// path carries as `idx_k_cache`). Prefill-then-decode chaining writes these into the capacity-sized
/// idx cache so both graphs share one state layout (Card 449 H0b).
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_prefill_layer_with_raw_k(
    b: &Builder,
    x: Traced,
    layer_prefix: &str,
    cfg: Qwen4ExpConfig,
    qcfg: QsaConfig,
    l: usize,
    cos: Traced,
    sin: Traced,
    block_cos: Traced,
    block_sin: Traced,
    causal_mask: Traced,
    block_eligible: Traced,
    tail_mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
) -> (Traced, Traced, Traced, Traced) {
    assert_eq!(
        qcfg.index_kv_heads, 1,
        "real Qwen3.8-Flash-Next's indexer is MQA-shaped (indexer_kv_heads=1) - spec 282"
    );
    let (hi, di, c) = (
        qcfg.index_n_heads,
        qcfg.index_head_dim,
        qcfg.index_compress_ratio,
    );
    let nb = l / c;
    let h = cfg.hidden;
    let p = |s: &str| format!("{layer_prefix}.{s}");

    // --- indexer: separate q/k projections (a simplification vs the real source's single fused
    // `index_qk_proj`, a checkpoint-naming detail; out of scope per the spec). ---
    let wiq = b.constant(&p("indexer.wq.weight"), TensorType::f32(vec![h, hi * di]));
    let q_idx_flat = linear(b, x, wiq, None); // [1,L,Hi*Di]
    let q_idx_raw = b.reshape(q_idx_flat, vec![1, l, hi, di]);
    let q_ln_w = b.constant(&p("indexer.q_layernorm.weight"), TensorType::f32(vec![di]));
    let q_idx_normed = rmsnorm(b, q_idx_raw, q_ln_w, cfg.eps); // [1,L,Hi,Di]
    let q_idx_t = b.transpose(q_idx_normed, vec![0, 2, 1, 3]); // [1,Hi,L,Di]
    let q_idx = rope_prefill(b, q_idx_t, cos, sin, l);

    let wik = b.constant(&p("indexer.wk.weight"), TensorType::f32(vec![h, di]));
    let raw_k = linear(b, x, wik, None); // [1,L,Di]
    let raw_k4 = b.reshape(raw_k, vec![1, 1, l, di]); // idx_k_cache layout
    let k_ln_w = b.constant(&p("indexer.k_layernorm.weight"), TensorType::f32(vec![di]));
    let block_k = qsa_pool_block_keys(b, raw_k, k_ln_w, block_cos, block_sin, l, c, di, cfg.eps);

    let scores = qsa_indexer_block_scores(b, q_idx, block_k, hi, di);
    let combined_mask = qsa_combined_mask(
        b,
        scores,
        block_eligible,
        causal_mask,
        tail_mask,
        qcfg.block_topk(),
        l,
        nb,
        c,
    );

    // --- reused gated GQA attention core (unchanged) ---
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let wq = b.constant(&p("q_proj.weight"), TensorType::f32(vec![h, nh * 2 * hd]));
    let wk = b.constant(&p("k_proj.weight"), TensorType::f32(vec![h, nkv * hd]));
    let wv = b.constant(&p("v_proj.weight"), TensorType::f32(vec![h, nkv * hd]));
    let wo = b.constant(&p("o_proj.weight"), TensorType::f32(vec![nh * hd, h]));
    let qn = b.constant(&p("q_norm.weight"), TensorType::f32(vec![hd]));
    let kn = b.constant(&p("k_norm.weight"), TensorType::f32(vec![hd]));

    let (out, kc, vc) = qwen3next_gated_attention_prefill(
        b,
        x,
        wq,
        wk,
        wv,
        wo,
        qn,
        kn,
        cos,
        sin,
        combined_mask,
        k_cache_in,
        v_cache_in,
        nh,
        nkv,
        hd,
        cfg.eps,
    );
    (out, kc, vc, raw_k4)
}

/// QSA decode block-eligibility mask (spec 282): the dynamic-`pos` counterpart of
/// [`block_eligible_mask_data`], for a fixed-KV masked decode step where the query position is a runtime
/// `Traced` value. Block `b` (start `block_positions[b] = b*C`) is eligible iff `pos >= b*C + C - 1`
/// (fully causally complete), the `BinOp::Ge`-against-a-per-slot-constant derivation of
/// `crate::deepseek4::hca_decode_validity_mask` with QSA's block width `C`. Returns `[nb]`, additive
/// (`0.0` eligible, [`QSA_MASK_NEG`] not yet complete).
pub(crate) fn qsa_decode_block_eligible_mask(
    b: &Builder,
    pos_f: Traced,
    block_positions: Traced,
    c: usize,
) -> Traced {
    let close = b.binary_scalar(BinOp::Add, block_positions, Scalar::F32((c - 1) as f32));
    let eligible = b.binary(BinOp::Ge, pos_f, close); // [nb] - 1.0 eligible, 0.0 not yet complete
    let shifted = b.binary_scalar(BinOp::Sub, eligible, Scalar::F32(1.0)); // 0.0 eligible, -1.0 not
    b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-QSA_MASK_NEG)) // 0.0 eligible, QSA_MASK_NEG not
}

/// QSA decode always-visible-tail mask (spec 282): the dynamic-`pos` counterpart of
/// [`tail_mask_data`]'s block-membership test, at block granularity (decode's single query row makes
/// block the natural unit; the caller expands to token granularity as [`qsa_combined_mask`] does). Block
/// `b` is query `pos`'s own in-progress block iff `b*C <= pos < b*C + C - 1`, strictly before the block's
/// last token, matching [`tail_mask_data`]'s `!block_complete` guard: once `pos` reaches the block's last
/// token the block is eligible (see [`qsa_decode_block_eligible_mask`]) and the tail rule stops forcing
/// it, leaving it to the top-k ranking. Using `b*C + C` here would force-include a block on the step it
/// closes and double-count it against `qsa_combined_mask_decode`'s top-k. Computed as
/// `Ge(pos, b*C) - Ge(pos, b*C + C - 1)` (both `{0.0,1.0}`-valued, first always `>=` second). Returns
/// `[nb]`, additive (`0.0` query's own in-progress block, [`QSA_MASK_NEG`] otherwise).
pub(crate) fn qsa_decode_tail_block_mask(
    b: &Builder,
    pos_f: Traced,
    block_positions: Traced,
    c: usize,
) -> Traced {
    let ge_low = b.binary(BinOp::Ge, pos_f, block_positions); // [nb] - 1.0 if pos >= b*C
    let closed = b.binary_scalar(BinOp::Add, block_positions, Scalar::F32((c - 1) as f32));
    let ge_closed = b.binary(BinOp::Ge, pos_f, closed); // [nb] - 1.0 if pos >= b*C + C - 1 (== eligible)
    let in_progress = b.binary(BinOp::Sub, ge_low, ge_closed); // [nb] - 1.0 iff b*C <= pos < b*C+C-1
    let shifted = b.binary_scalar(BinOp::Sub, in_progress, Scalar::F32(1.0)); // 0.0 own block, -1.0 else
    b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-QSA_MASK_NEG))
}

/// [`qsa_combined_mask`]'s decode counterpart (spec 282): combines a single query row's block-topk
/// selection ([`qsa_decode_block_eligible_mask`]) with the always-visible tail block
/// ([`qsa_decode_tail_block_mask`]) and the caller-supplied per-token causal/validity mask (the standard
/// [`poot_graph_ir::Slot::Mask`] `[cap]` row, `decode_mask_row`'s convention, so no QSA-specific bind arm)
/// into one additive `[1,1,1,cap]` mask for [`poot_graph_ir::ops::attention_masked`]. Expands block
/// decisions to token granularity as [`qsa_combined_mask`] does, at `Lq=1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_combined_mask_decode(
    b: &Builder,
    index_scores: Traced,   // [1,1,1,nb]
    block_eligible: Traced, // [1,1,1,nb]
    tail_block: Traced,     // [1,1,1,nb]
    causal_mask: Traced,    // [1,1,1,cap]
    block_topk: usize,
    nb: usize,
    c: usize,
    cap: usize,
) -> Traced {
    assert_eq!(
        nb * c,
        cap,
        "cap must be an exact multiple of compress_ratio (spec 282)"
    );
    // Rank the sum (not the raw score), as `qsa_combined_mask` does.
    let ranked = b.binary(BinOp::Add, index_scores, block_eligible);
    let keep = qsa_block_top_k(b, ranked, block_topk.min(nb)); // 1.0 kept, 0.0 dropped
    let shifted = b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0)); // 0 kept, -1 dropped
    let topk_add = b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-QSA_MASK_NEG));
    // Re-AND eligibility after top-k, as `qsa_combined_mask` does.
    let block_selected = b.binary(BinOp::Add, block_eligible, topk_add); // [1,1,1,nb]

    // Expand the block decision to token granularity: token j inherits block floor(j/C)'s decision.
    let expanded = b.reshape(block_selected, vec![1, 1, 1, nb, 1]);
    let expanded = b.broadcast(expanded, vec![1, 1, 1, nb, c]);
    let expanded = b.reshape(expanded, vec![1, 1, 1, cap]);

    let tail_expanded = b.reshape(tail_block, vec![1, 1, 1, nb, 1]);
    let tail_expanded = b.broadcast(tail_expanded, vec![1, 1, 1, nb, c]);
    let tail_expanded = b.reshape(tail_expanded, vec![1, 1, 1, cap]);

    // OR with the tail (Max, not Add; see `qsa_combined_mask`).
    let selection = b.binary(BinOp::Max, expanded, tail_expanded);

    // AND with the caller's causal/validity mask (both terms `{0.0, MASK_NEG}`-valued, so plain Add is
    // AND). This also excludes any not-yet-written cache slot regardless of the block-level decision:
    // ranking excludes, but the final Add guarantees it (as `dsa_combined_mask`/`csa_topk_mask`).
    b.binary(BinOp::Add, selection, causal_mask)
}

/// One QSA (full-attention) layer's decode step (spec 282): a new token against a fixed-capacity,
/// incrementally-built KV cache; the decode counterpart of [`qsa_attention_prefill_layer`], as
/// `qwen3next_gated_attention` relates to [`qwen3next_gated_attention_prefill`].
///
/// The indexer's raw (unrotated, unnormalized) key history is carried as a third per-layer state tensor
/// (`idx_k_cache`, `[1,1,cap,Di]`) beside the main K/V pair, as DSA's `self_attn.indexer.k_cache`
/// (`crate::deepseek32::trace_deepseek32_dsa_decode`) and CSA's `kv_raw_cache`/`gate_raw_cache`
/// (`crate::deepseek4::deepseek4_csa_layer_decode`). Unlike DSA (which caches the normed, RoPE'd key),
/// this cache holds the raw key, so [`qsa_pool_block_keys`] runs unchanged on the full `cap`-wide cache
/// every step, as CSA's `csa_overlap_pool` does. The causal-completeness mask, not incremental
/// bookkeeping, keeps an incomplete or unwritten block from affecting the result. `cap` must be an exact
/// multiple of `qcfg.index_compress_ratio`.
///
/// Returns `(out [1,1,H], k_cache_out, v_cache_out, idx_k_cache_out)`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_decode_layer(
    b: &Builder,
    x: Traced,
    layer_prefix: &str,
    cfg: Qwen4ExpConfig,
    qcfg: QsaConfig,
    cap: usize,
    pos_slot: Traced,
    pos_f: Traced,
    cos: Traced,
    sin: Traced,
    block_cos: Traced,
    block_sin: Traced,
    block_positions: Traced,
    causal_mask: Traced,
    k_cache_in: Traced,
    v_cache_in: Traced,
    idx_k_cache_in: Traced,
) -> (Traced, Traced, Traced, Traced) {
    assert_eq!(
        qcfg.index_kv_heads, 1,
        "real Qwen3.8-Flash-Next's indexer is MQA-shaped (indexer_kv_heads=1) - spec 282"
    );
    let (hi, di, c) = (
        qcfg.index_n_heads,
        qcfg.index_head_dim,
        qcfg.index_compress_ratio,
    );
    let nb = cap / c;
    let h = cfg.hidden;
    let p = |s: &str| format!("{layer_prefix}.{s}");

    // --- indexer: fresh query for this single token, RoPE'd at `pos_slot`. ---
    let wiq = b.constant(&p("indexer.wq.weight"), TensorType::f32(vec![h, hi * di]));
    let q_idx_flat = linear(b, x, wiq, None); // [1,1,Hi*Di]
    let q_idx_raw = b.reshape(q_idx_flat, vec![1, 1, hi, di]);
    let q_ln_w = b.constant(&p("indexer.q_layernorm.weight"), TensorType::f32(vec![di]));
    let q_idx_normed = rmsnorm(b, q_idx_raw, q_ln_w, cfg.eps); // [1,1,Hi,Di]
    let q_idx_t = b.transpose(q_idx_normed, vec![0, 2, 1, 3]); // [1,Hi,1,Di]
    let q_idx = rope(b, q_idx_t, cos, sin, pos_slot);

    // --- indexer: fresh raw key for this token, written into the raw-key history cache. ---
    let wik = b.constant(&p("indexer.wk.weight"), TensorType::f32(vec![h, di]));
    let raw_k_new = linear(b, x, wik, None); // [1,1,Di]
    let raw_k_new4 = b.reshape(raw_k_new, vec![1, 1, 1, di]);
    let idx_k_cache_out = b.dynamic_update_slice_dyn(idx_k_cache_in, raw_k_new4, pos_slot, 2);

    // Recompute every block's pooled key from the full raw history each step (see the doc comment):
    // `qsa_pool_block_keys` unchanged, called at `l=cap`.
    let raw_k_flat = b.reshape(idx_k_cache_out, vec![1, cap, di]);
    let k_ln_w = b.constant(&p("indexer.k_layernorm.weight"), TensorType::f32(vec![di]));
    let block_k = qsa_pool_block_keys(
        b, raw_k_flat, k_ln_w, block_cos, block_sin, cap, c, di, cfg.eps,
    );

    let scores = qsa_indexer_block_scores(b, q_idx, block_k, hi, di); // [1,1,1,nb]
    let block_eligible = qsa_decode_block_eligible_mask(b, pos_f, block_positions, c);
    let block_eligible = b.reshape(block_eligible, vec![1, 1, 1, nb]);
    let tail_block = qsa_decode_tail_block_mask(b, pos_f, block_positions, c);
    let tail_block = b.reshape(tail_block, vec![1, 1, 1, nb]);
    let combined_mask = qsa_combined_mask_decode(
        b,
        scores,
        block_eligible,
        tail_block,
        causal_mask,
        qcfg.block_topk(),
        nb,
        c,
        cap,
    );

    // --- main gated GQA attention, inlined at Lq=1 (the math of `qwen3next_gated_attention`, but with an
    // already-combined mask via `attention_masked` and a dynamic `pos_slot` write instead of a baked
    // `pos_idx`/valid-prefix slice, so one graph serves every decode step). ---
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let n_rep = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let q_dim = nh * hd;
    let wq = b.constant(&p("q_proj.weight"), TensorType::f32(vec![h, nh * 2 * hd]));
    let wk = b.constant(&p("k_proj.weight"), TensorType::f32(vec![h, nkv * hd]));
    let wv = b.constant(&p("v_proj.weight"), TensorType::f32(vec![h, nkv * hd]));
    let wo = b.constant(&p("o_proj.weight"), TensorType::f32(vec![nh * hd, h]));
    let qn = b.constant(&p("q_norm.weight"), TensorType::f32(vec![hd]));
    let kn = b.constant(&p("k_norm.weight"), TensorType::f32(vec![hd]));

    let qg_flat = linear(b, x, wq, None); // [1,1,nh*2*hd]
    let qg = b.reshape(qg_flat, vec![1, 1, nh, 2 * hd]);
    let q_raw = b.slice(qg, 3, 0, hd);
    let gate_raw = b.slice(qg, 3, hd, 2 * hd);
    let k_flat = linear(b, x, wk, None);
    let v_flat = linear(b, x, wv, None);

    let q_normed = rmsnorm(b, q_raw, qn, cfg.eps);
    let k_shaped = b.reshape(k_flat, vec![1, 1, nkv, hd]);
    let k_normed = rmsnorm(b, k_shaped, kn, cfg.eps);
    let v_shaped = b.reshape(v_flat, vec![1, 1, nkv, hd]);

    let q4 = b.transpose(q_normed, vec![0, 2, 1, 3]); // [1,nh,1,hd]
    let k4 = b.transpose(k_normed, vec![0, 2, 1, 3]); // [1,nkv,1,hd]
    let v4 = b.transpose(v_shaped, vec![0, 2, 1, 3]); // [1,nkv,1,hd]

    let q4 = rope(b, q4, cos, sin, pos_slot);
    let k4 = rope(b, k4, cos, sin, pos_slot);

    let k_cache_out = b.dynamic_update_slice_dyn(k_cache_in, k4, pos_slot, 2);
    let v_cache_out = b.dynamic_update_slice_dyn(v_cache_in, v4, pos_slot, 2);

    let attn = attention_masked(b, q4, k_cache_out, v_cache_out, n_rep, scale, combined_mask);

    let gate4 = b.transpose(gate_raw, vec![0, 2, 1, 3]);
    let gate_sig = sigmoid(b, gate4);
    let attn_gated = b.binary(BinOp::Mul, attn, gate_sig);

    let attn_back = b.transpose(attn_gated, vec![0, 2, 1, 3]);
    let attn_flat = b.reshape(attn_back, vec![1, 1, q_dim]);
    let out = linear(b, attn_flat, wo, None);

    (out, k_cache_out, v_cache_out, idx_k_cache_out)
}

/// A small (`layers`-deep, QSA-only, no FFN) prefill probe: `x = x + qsa_layer(rmsnorm(x))` per layer,
/// final `rmsnorm`. It takes `x0` as a directly-fed `[1,L,H]` constant (no embedding gather) and has no
/// FFN, so the differential test targets the indexer mask construction feeding the reused gated-attention
/// core at every query position, not only the last. Returns `[1,L,H]`.
#[cfg(test)]
pub(crate) fn trace_qwen38_qsa_probe(
    cfg: Qwen4ExpConfig,
    qcfg: QsaConfig,
    layers: usize,
    l: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let c = qcfg.index_compress_ratio;
    let nb = l / c;

    let x0 = b.constant("probe.x0", TensorType::f32(vec![1, l, h]));
    let cos = b.constant("rope.cos", TensorType::f32(vec![l, cfg.rotary_dim]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![l, cfg.rotary_dim]));
    let block_cos = b.constant(
        "qsa.block_rope.cos",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let block_sin = b.constant(
        "qsa.block_rope.sin",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let causal_mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));
    let block_eligible = b.slot_named(
        Slot::Activation,
        "qsa.block_eligible",
        TensorType::f32(vec![1, 1, l, nb]),
    );
    let tail_mask = b.slot_named(
        Slot::Activation,
        "qsa.tail_mask",
        TensorType::f32(vec![1, 1, l, l]),
    );

    let mut x = x0;
    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for li in 0..layers {
        let p = format!("layers.{li}");
        let ln_w = b.constant(&format!("{p}.ln.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln_w, cfg.eps);
        let kc_in = b.state_input(
            &format!("{p}.k_cache"),
            TensorType::f32(vec![1, cfg.n_kv_heads, l, cfg.head_dim]),
            StateRole::Recurrent,
        );
        let vc_in = b.state_input(
            &format!("{p}.v_cache"),
            TensorType::f32(vec![1, cfg.n_kv_heads, l, cfg.head_dim]),
            StateRole::Recurrent,
        );
        let (attn_out, kc_out, vc_out) = qsa_attention_prefill_layer(
            &b,
            normed,
            &p,
            cfg,
            qcfg,
            l,
            cos,
            sin,
            block_cos,
            block_sin,
            causal_mask,
            block_eligible,
            tail_mask,
            kc_in,
            vc_in,
        );
        x = b.binary(BinOp::Add, x, attn_out);
        state.push((kc_in, kc_out));
        state.push((vc_in, vc_out));
    }

    let final_ln = b.constant("final.ln.weight", TensorType::f32(vec![h]));
    let out = rmsnorm(&b, x, final_ln, cfg.eps);
    b.finish_with_state(out, &state)
}

/// [`trace_qwen38_qsa_probe`]'s decode counterpart (spec 282): a `layers`-deep, QSA-only, no-FFN probe over
/// a fixed-capacity cache, composing [`qsa_attention_decode_layer`]. It lets the differential test compare
/// against the same independent reference (`qsa_layer_ref`/`qsa_probe_ref`) as the prefill probe. The
/// pre-attention/final norm use per-layer `ln.weight`/`final.ln.weight` plain RMSNorm gammas (unlike
/// [`trace_qwen38_decode`]'s Hyper-Connections `hc_norm`, `qwen38_gated_residual`), matching
/// `qsa_probe_ref`. `x0` is fed fresh per call (one decode step = one `[1,1,H]` `probe.x0` binding); the
/// caller threads `state` across sequential calls to build a multi-position trajectory.
#[cfg(test)]
pub(crate) fn trace_qwen38_qsa_probe_decode(
    cfg: Qwen4ExpConfig,
    qcfg: QsaConfig,
    layers: usize,
    cap: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let c = qcfg.index_compress_ratio;
    let nb = cap / c;

    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let iota_full = b.iota(cap);
    let pos_f = b.gather_scalar(iota_full, 0, pos_slot);

    let x0 = b.constant("probe.x0", TensorType::f32(vec![1, 1, h]));
    let cos = b.constant("rope.cos", TensorType::f32(vec![cap, cfg.rotary_dim]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cap, cfg.rotary_dim]));
    let block_cos = b.constant(
        "qsa.block_rope.cos",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let block_sin = b.constant(
        "qsa.block_rope.sin",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let block_positions = b.slot_named(
        Slot::Activation,
        "qsa.block_positions",
        TensorType::f32(vec![nb]),
    );

    let mut x = x0;
    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for li in 0..layers {
        let p = format!("layers.{li}");
        let ln_w = b.constant(&format!("{p}.ln.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln_w, cfg.eps);
        let kc_in = b.state_input(
            &format!("{p}.k_cache"),
            TensorType::f32(vec![1, cfg.n_kv_heads, cap, cfg.head_dim]),
            StateRole::Recurrent,
        );
        let vc_in = b.state_input(
            &format!("{p}.v_cache"),
            TensorType::f32(vec![1, cfg.n_kv_heads, cap, cfg.head_dim]),
            StateRole::Recurrent,
        );
        let idxc_in = b.state_input(
            &format!("{p}.idx_k_cache"),
            TensorType::f32(vec![1, 1, cap, qcfg.index_head_dim]),
            StateRole::Recurrent,
        );
        let (attn_out, kc_out, vc_out, idxc_out) = qsa_attention_decode_layer(
            &b,
            normed,
            &p,
            cfg,
            qcfg,
            cap,
            pos_slot,
            pos_f,
            cos,
            sin,
            block_cos,
            block_sin,
            block_positions,
            mask,
            kc_in,
            vc_in,
            idxc_in,
        );
        x = b.binary(BinOp::Add, x, attn_out);
        state.push((kc_in, kc_out));
        state.push((vc_in, vc_out));
        state.push((idxc_in, idxc_out));
    }

    let final_ln = b.constant("final.ln.weight", TensorType::f32(vec![h]));
    let out = rmsnorm(&b, x, final_ln, cfg.eps);
    b.finish_with_state(out, &state)
}
