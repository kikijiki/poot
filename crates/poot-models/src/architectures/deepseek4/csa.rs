#[cfg(test)]
use super::*;

/// CSA's overlapping "Ca/Cb" window construction (paper section 2.3.1). `DeepseekV4CSACompressor` (at
/// `head_dim`) and `DeepseekV4Indexer` (at `index_head_dim`) both use it. Each source token's
/// `kv_proj`/`gate_proj` output is `2*head_dim` wide, split into "Ca" (`[..,:head_dim]`, its contribution to
/// the NEXT window's entry) and "Cb" (`[..,head_dim:]`, its contribution to its OWN window's entry). Entry `w`
/// softmax-pools `2*compress_rate` slots: window `w-1`'s Ca half concatenated with window `w`'s Cb half.
///
/// `raw_kv`/`raw_gate` are `[1, n_windows, compress_rate, 2*head_dim]` (window-reshaped RAW projections).
/// `position_bias` is `[compress_rate, 2*head_dim]`, added to `raw_gate` before the Ca/Cb split, at each
/// window's own slot index and before any window shift. Window `0` has no predecessor: its prior-Ca half is
/// zero kv / [`HCA_MASK_NEG`] gate (softmax weight ~0), as in the reference's first window. Decode and prefill
/// hit the same case because the full history is recomputed each step, so no cross-call overlap state exists.
/// Returns the pooled, `kv_norm`'d entries `[1, n_windows, head_dim]`, before RoPE ([`hca_window_rope`]).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn csa_overlap_pool(
    b: &Builder,
    raw_kv: Traced,
    raw_gate: Traced,
    position_bias: Traced,
    kv_norm_w: Traced,
    n_windows: usize,
    compress_rate: usize,
    head_dim: usize,
    eps: f32,
) -> Traced {
    let pb = b.reshape(position_bias, vec![1, 1, compress_rate, 2 * head_dim]);
    let gate_biased = b.binary(BinOp::Add, raw_gate, pb); // [1,n_windows,cr,2*head_dim], bias at own window

    let ca_kv = b.slice(raw_kv, 3, 0, head_dim); // this window's Ca -> feeds the NEXT window's entry
    let cb_kv = b.slice(raw_kv, 3, head_dim, 2 * head_dim); // this window's own Cb
    let ca_gate = b.slice(gate_biased, 3, 0, head_dim);
    let cb_gate = b.slice(gate_biased, 3, head_dim, 2 * head_dim);

    // Window 0's "prior Ca" pad: zeros (kv) / HCA_MASK_NEG (gate), derived from an existing traced value's shape
    // via `binary_scalar` (as `clamp_max`/`clamp_sym` derive constants) rather than a host-supplied constant.
    let ca_kv_win0 = b.slice(ca_kv, 1, 0, 1); // [1,1,cr,head_dim] - shape template only, value discarded
    let pad_kv = b.binary_scalar(BinOp::Mul, ca_kv_win0, Scalar::F32(0.0));
    let pad_gate = b.binary_scalar(BinOp::Add, pad_kv, Scalar::F32(HCA_MASK_NEG));

    let (prior_ca_kv, prior_ca_gate) = if n_windows > 1 {
        let prev_kv = b.slice(ca_kv, 1, 0, n_windows - 1);
        let prev_gate = b.slice(ca_gate, 1, 0, n_windows - 1);
        (
            b.concat(1, &[pad_kv, prev_kv]),
            b.concat(1, &[pad_gate, prev_gate]),
        )
    } else {
        (pad_kv, pad_gate)
    };

    let new_kv = b.concat(2, &[prior_ca_kv, cb_kv]); // [1,n_windows,2*cr,head_dim]
    let new_gate = b.concat(2, &[prior_ca_gate, cb_gate]);
    softmax_gate_pool_core(b, new_kv, new_gate, kv_norm_w, eps)
}

/// Lightning Indexer score (`DeepseekV4IndexerScorer.forward`, paper section 2.3.1 eqs 13-17):
/// `I[..,Lq,Lk] = softmax_scale * sum_h w_h * ReLU(q_h . k_s)` with `softmax_scale = index_head_dim**-0.5`,
/// `w_h = weights_scaling * weights_proj(hidden_states)_h`, `weights_scaling = index_n_heads**-0.5`. The head
/// reduce is a direct axis-1 reduce, as `crate::deepseek32::dsa_indexer_scores`, but applies the
/// two scale constants, which DSA's scorer never applies. `q_idx` is `[1,Hi,Lq,Di]` (per-head, RoPE'd); `k_idx`
/// is `[1,Lk,Di]` (the indexer's compressed keys, shared across heads via `repeat_kv`); `w_idx` is `[1,Hi,Lq]`.
/// Returns `[1,1,Lq,Lk]`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn csa_indexer_scores(
    b: &Builder,
    q_idx: Traced,
    k_idx: Traced,
    w_idx: Traced,
    hi: usize,
    lq: usize,
    lk: usize,
    di: usize,
    softmax_scale: f32,
    weights_scaling: f32,
) -> Traced {
    let k_idx_h = b.reshape(k_idx, vec![1, 1, lk, di]);
    let k_idx_b = repeat_kv(b, k_idx_h, hi); // [1,Hi,Lk,Di] - one shared compressed-key row per head
    let k_idx_t = b.transpose(k_idx_b, vec![0, 1, 3, 2]); // [1,Hi,Di,Lk]
    let dots = b.matmul(q_idx, k_idx_t); // [1,Hi,Lq,Lk]
    let scored = relu(b, dots);
    let scored = b.binary_scalar(BinOp::Mul, scored, Scalar::F32(softmax_scale));
    let w_r = b.reshape(w_idx, vec![1, hi, lq, 1]);
    let w_scaled = b.binary_scalar(BinOp::Mul, w_r, Scalar::F32(weights_scaling));
    let w_b = b.broadcast(w_scaled, vec![1, hi, lq, lk]);
    let weighted = b.binary(BinOp::Mul, scored, w_b);
    // Reduce the head axis (1), keepdim, directly (card 523b): `compile`'s `lower_nonlast_reduces`
    // legalizes a non-last-axis reduce for every target.
    b.reduce(RedOp::Sum, weighted, 1, true) // [1,1,Lq,Lk]
}

/// CSA's per-query compressed-entry block bias, as the additive-mask top-k trick of
/// `crate::deepseek32::dsa_combined_mask` instead of the reference's `scatter_`/`-1` sentinel. Ranking
/// `index_scores + causal_mask` (not raw `index_scores`) means a slot picked only because fewer than `topk`
/// entries are causally valid still carries `causal_mask`'s [`HCA_MASK_NEG`] in the final sum, matching the
/// reference where sentinel picks land in a discarded padding column and stay `-inf`. `causal_mask` is the
/// window-causal-completeness mask ([`hca_decode_validity_mask`] on decode, a caller-supplied block bias on
/// prefill); indexer selection layers on top of it. `topk` is `min(cfg.index_topk, n_windows)`.
#[cfg(test)]
pub(crate) fn csa_topk_mask(
    b: &Builder,
    index_scores: Traced,
    causal_mask: Traced,
    topk: usize,
) -> Traced {
    let ranked_input = b.binary(BinOp::Add, index_scores, causal_mask);
    let rank = pairwise_rank(b, ranked_input);
    let keep = keep_top_k_mask(b, rank, topk); // 1.0 kept, 0.0 dropped
    let shifted = b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0)); // 0.0 kept, -1.0 dropped
    let topk_add = b.binary_scalar(BinOp::Mul, shifted, Scalar::F32(-HCA_MASK_NEG)); // 0 kept, HCA_MASK_NEG dropped
    b.binary(BinOp::Add, causal_mask, topk_add)
}

/// One `compressed_sparse_attention` layer's DECODE step, shared by [`trace_deepseek4_csa_decode`] and the
/// hybrid stack. Unlike HCA, the final attention mask is layer-specific (the Lightning Indexer's top-`k`
/// selection depends on the layer's weights), so this takes the layer-independent `local_mask` and pre-top-k
/// `csa_causal_mask` and builds its own `combined_mask`. `cos`/`sin` are the `"compress"` rope-type tables.
/// Returns the updated widened residual and the layer's five state pairs (local K==V cache, compressor raw
/// kv/gate history, indexer raw kv/gate history).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_csa_layer_decode(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    cap: usize,
    pos_slot: Traced,
    cos: Traced,
    sin: Traced,
    local_mask: Traced,
    csa_causal_mask: Traced,
    win_idx: Traced,
    n_windows: usize,
    plan: &DeepseekV4SourcePlan,
    dims: V4MoeDims,
    moe: &mut V4MoeContext<'_>,
) -> Result<(Traced, Vec<(Traced, Traced)>), DeepseekV4MoeError> {
    let (h, hq, d, rd) = (
        cfg.hidden,
        cfg.num_heads,
        cfg.head_dim,
        cfg.qk_rope_head_dim,
    );
    let nope_w = d - rd;
    let hc = cfg.hc_mult;
    let scale = cfg.attn_scale();
    let cr = cfg.csa_compress_rate;
    let (hi, di) = (cfg.index_n_heads, cfg.index_head_dim);
    let idx_nope_w = di - rd;
    let softmax_scale = 1.0 / (di as f32).sqrt();
    let weights_scaling = 1.0 / (hi as f32).sqrt();
    let topk = cfg.index_topk.min(n_windows);
    // State tensors are not checkpoint rows; they share the plan's `layers.{i}.` prefix so names read alike.
    let state_name = |s: &str| format!("{}.{s}", sources::v4_layer_prefix(li));
    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(5);

    let ahc_fn = v4_mhc_projection(b, plan.layer_dense(li, V4LayerDense::HcAttnProjection)?);
    let ahc_base = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcAttnBase)?);
    let ahc_scale = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcAttnScale)?);
    let (post, comb, collapsed) = hyper_connection(
        b,
        x,
        ahc_fn,
        ahc_base,
        ahc_scale,
        hc,
        h,
        1,
        cfg.eps,
        cfg.hc_eps,
        cfg.hc_sinkhorn_iters,
    );

    let ln1 = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnNorm)?);
    let normed = rmsnorm(b, collapsed, ln1, cfg.eps);

    let q_a = v4_packed_linear(b, plan, li, V4PackedRole::WqA, normed)?;
    let q_res = rmsnorm(
        b,
        q_a,
        v4_source_f32(b, plan.layer_dense(li, V4LayerDense::QNorm)?),
        cfg.eps,
    );
    let q = v4_packed_linear(b, plan, li, V4PackedRole::WqB, q_res)?;
    let q = b.transpose(b.reshape(q, vec![1, 1, hq, d]), vec![0, 2, 1, 3]); // [1,Hq,1,D]
    let q = rmsnorm_no_weight(b, q, cfg.eps); // q_b_norm

    let q_nope = b.slice(q, 3, 0, nope_w);
    let q_rope = b.slice(q, 3, nope_w, d);
    let q_rope = rope_interleaved_decode(b, q_rope, cos, sin, pos_slot);
    let q = b.concat(3, &[q_nope, q_rope]);

    let kv = v4_packed_linear(b, plan, li, V4PackedRole::Wkv, normed)?; // [1,1,D]
    let kv = rmsnorm(
        b,
        kv,
        v4_source_f32(b, plan.layer_dense(li, V4LayerDense::KvNorm)?),
        cfg.eps,
    );
    let kv = b.reshape(kv, vec![1, 1, 1, d]);
    let kv_nope = b.slice(kv, 3, 0, nope_w);
    let kv_rope = b.slice(kv, 3, nope_w, d);
    let kv_rope = rope_interleaved_decode(b, kv_rope, cos, sin, pos_slot);
    let kv_new = b.concat(3, &[kv_nope, kv_rope]); // [1,1,1,D]

    let kv_cache = b.state_input(
        &state_name("self_attn.kv_cache"),
        TensorType::f32(vec![1, 1, cap, d]),
        StateRole::Recurrent,
    );
    let kv_cache_out = b.dynamic_update_slice_dyn(kv_cache, kv_new, pos_slot, 2);
    state.push((kv_cache, kv_cache_out));

    // CSA compressor's own raw kv/gate history (2*head_dim wide, Ca/Cb layout).
    let cwkv = plan.compressor_dense(li, V4CompressorDense::Kv)?;
    let cwgate = plan.compressor_dense(li, V4CompressorDense::Gate)?;
    let csa_kv_new = b.reshape(v4_source_linear(b, normed, cwkv), vec![1, 1, 1, 2 * d]);
    let csa_gate_new = b.reshape(v4_source_linear(b, normed, cwgate), vec![1, 1, 1, 2 * d]);

    let csa_kv_cache = b.state_input(
        &state_name("self_attn.csa.kv_raw_cache"),
        TensorType::f32(vec![1, 1, cap, 2 * d]),
        StateRole::Recurrent,
    );
    let csa_kv_cache_out = b.dynamic_update_slice_dyn(csa_kv_cache, csa_kv_new, pos_slot, 2);
    state.push((csa_kv_cache, csa_kv_cache_out));
    let csa_gate_cache = b.state_input(
        &state_name("self_attn.csa.gate_raw_cache"),
        TensorType::f32(vec![1, 1, cap, 2 * d]),
        StateRole::Recurrent,
    );
    let csa_gate_cache_out = b.dynamic_update_slice_dyn(csa_gate_cache, csa_gate_new, pos_slot, 2);
    state.push((csa_gate_cache, csa_gate_cache_out));

    let csa_chunk_kv = b.reshape(csa_kv_cache_out, vec![1, n_windows, cr, 2 * d]);
    let csa_chunk_gate = b.reshape(csa_gate_cache_out, vec![1, n_windows, cr, 2 * d]);
    let csa_pb = v4_source_f32(
        b,
        plan.compressor_dense(li, V4CompressorDense::PositionBias)?,
    );
    let csa_kv_norm_w = v4_source_f32(b, plan.compressor_dense(li, V4CompressorDense::Norm)?);
    let csa_pooled = csa_overlap_pool(
        b,
        csa_chunk_kv,
        csa_chunk_gate,
        csa_pb,
        csa_kv_norm_w,
        n_windows,
        cr,
        d,
        cfg.eps,
    );
    let csa_compressed = hca_window_rope(b, csa_pooled, cos, sin, win_idx, nope_w, rd);
    let csa_compressed = b.reshape(csa_compressed, vec![1, 1, n_windows, d]);

    // Lightning Indexer: its own overlapping-window pooled keys at index_head_dim, plus its own query.
    let iwkv = plan.indexer_dense(li, V4IndexerDense::Kv)?;
    let iwgate = plan.indexer_dense(li, V4IndexerDense::Gate)?;
    let idx_kv_new = b.reshape(v4_source_linear(b, normed, iwkv), vec![1, 1, 1, 2 * di]);
    let idx_gate_new = b.reshape(v4_source_linear(b, normed, iwgate), vec![1, 1, 1, 2 * di]);
    let idx_kv_cache = b.state_input(
        &state_name("self_attn.csa.indexer.kv_raw_cache"),
        TensorType::f32(vec![1, 1, cap, 2 * di]),
        StateRole::Recurrent,
    );
    let idx_kv_cache_out = b.dynamic_update_slice_dyn(idx_kv_cache, idx_kv_new, pos_slot, 2);
    state.push((idx_kv_cache, idx_kv_cache_out));
    let idx_gate_cache = b.state_input(
        &state_name("self_attn.csa.indexer.gate_raw_cache"),
        TensorType::f32(vec![1, 1, cap, 2 * di]),
        StateRole::Recurrent,
    );
    let idx_gate_cache_out = b.dynamic_update_slice_dyn(idx_gate_cache, idx_gate_new, pos_slot, 2);
    state.push((idx_gate_cache, idx_gate_cache_out));

    let idx_chunk_kv = b.reshape(idx_kv_cache_out, vec![1, n_windows, cr, 2 * di]);
    let idx_chunk_gate = b.reshape(idx_gate_cache_out, vec![1, n_windows, cr, 2 * di]);
    let idx_pb = v4_source_f32(b, plan.indexer_dense(li, V4IndexerDense::PositionBias)?);
    let idx_kv_norm_w = v4_source_f32(b, plan.indexer_dense(li, V4IndexerDense::Norm)?);
    let idx_pooled = csa_overlap_pool(
        b,
        idx_chunk_kv,
        idx_chunk_gate,
        idx_pb,
        idx_kv_norm_w,
        n_windows,
        cr,
        di,
        cfg.eps,
    );
    let idx_compressed = hca_window_rope(b, idx_pooled, cos, sin, win_idx, idx_nope_w, rd);

    // The indexer's query up projection is an E4M3 packed pair and consumes the same `q_res` as the main Q.
    let q_idx = v4_packed_linear(b, plan, li, V4PackedRole::IndexerWqB, q_res)?; // [1,1,Hi*Di]
    let q_idx = b.transpose(b.reshape(q_idx, vec![1, 1, hi, di]), vec![0, 2, 1, 3]); // [1,Hi,1,Di]
    let q_idx_nope = b.slice(q_idx, 3, 0, idx_nope_w);
    let q_idx_rope = b.slice(q_idx, 3, idx_nope_w, di);
    let q_idx_rope = rope_interleaved_decode(b, q_idx_rope, cos, sin, pos_slot);
    let q_idx = b.concat(3, &[q_idx_nope, q_idx_rope]);

    let w_idx = v4_source_linear(b, normed, plan.indexer_dense(li, V4IndexerDense::Weights)?); // [1,1,Hi]
    let w_idx = b.reshape(w_idx, vec![1, hi, 1]);

    let index_scores = csa_indexer_scores(
        b,
        q_idx,
        idx_compressed,
        w_idx,
        hi,
        1,
        n_windows,
        di,
        softmax_scale,
        weights_scaling,
    );
    let csa_mask = csa_topk_mask(b, index_scores, csa_causal_mask, topk);
    let combined_mask = b.concat(3, &[local_mask, csa_mask]);
    let kv_full = b.concat(2, &[kv_cache_out, csa_compressed]);

    let sinks = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnSink)?);
    let attn =
        attention_masked_with_sink(b, q, kv_full, kv_full, hq, scale, combined_mask, sinks, hq);
    let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,1,Hq,D]
    let attn_nope = b.slice(attn, 3, 0, nope_w);
    let attn_rope = b.slice(attn, 3, nope_w, d);
    let attn_rope = rope_interleaved_decode_inverse(b, attn_rope, cos, sin, pos_slot);
    let attn = b.concat(3, &[attn_nope, attn_rope]);

    // `attn.wo_a` is Card 385's block-diagonal packed linear; `attn.wo_b` is an ordinary packed linear.
    let grouped =
        deepseek4_grouped_out_a(b, plan, li, attn, 1, hq, d, cfg.o_groups, cfg.o_lora_rank)?;
    let attn_out = v4_packed_linear(b, plan, li, V4PackedRole::WoB, grouped)?;

    let mut x = hyper_connection_combine(b, post, comb, attn_out, x, hc, 1, h);

    let fhc_fn = v4_mhc_projection(b, plan.layer_dense(li, V4LayerDense::HcFfnProjection)?);
    let fhc_base = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcFfnBase)?);
    let fhc_scale = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcFfnScale)?);
    let (post2, comb2, collapsed2) = hyper_connection(
        b,
        x,
        fhc_fn,
        fhc_base,
        fhc_scale,
        hc,
        h,
        1,
        cfg.eps,
        cfg.hc_eps,
        cfg.hc_sinkhorn_iters,
    );
    let ln2 = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::FfnNorm)?);
    let normed2 = rmsnorm(b, collapsed2, ln2, cfg.eps);
    let mlp_out = deepseek4_routed_moe_ffn(b, cfg, plan, dims, li, normed2, moe)?;
    x = hyper_connection_combine(b, post2, comb2, mlp_out, x, hc, 1, h);

    Ok((x, state))
}

/// Trace a single CSA (`compressed_sparse_attention`-only) DECODE step. Extends [`trace_deepseek4_hca_decode`]'s
/// per-layer body with CSA's compressed branch: the compressor's overlapping-window pooled KV
/// ([`csa_overlap_pool`] at `head_dim`), plus the Lightning Indexer's overlapping-window pooled keys (again at
/// `index_head_dim`) and top-`k` selection ([`csa_indexer_scores`]/[`csa_topk_mask`]) over the compressed
/// entries.
///
/// Five per-layer state tensors: the local K==V cache, the compressor's raw `kv_proj`/`gate_proj` history
/// (`2*head_dim` wide, Ca/Cb layout), and the indexer's (`2*index_head_dim` wide). All are indexed by absolute
/// position and recomputed every step, as for HCA, so window 0 is the same case in decode and prefill (see
/// [`csa_overlap_pool`]).
///
/// The indexer's query projection consumes `q_res` (the `q_a_norm` output of the main Q path), and its q/k RoPE
/// reuses the layer's `cos`/`sin` table: the table width is fixed by `qk_rope_head_dim` whatever tensor it
/// rotates, so the `[max_pos, rd/2]` table also serves the `index_head_dim`-wide indexer tensors. (DSA's
/// indexer needed a separate half-split table, `crate::deepseek32::dsa_indexer_rope_tables`, because it uses a
/// different RoPE convention.)
///
/// `cap` must be an exact multiple of `cfg.csa_compress_rate`.
#[cfg(test)]
pub(crate) fn trace_deepseek4_csa_decode(
    cfg: DeepseekV4Config,
    cap: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    assert!(
        cap.is_multiple_of(cfg.csa_compress_rate) && cfg.csa_compress_rate > 0,
        "trace_deepseek4_csa_decode: cap must be an exact multiple of csa_compress_rate"
    );
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);
    let cr = cfg.csa_compress_rate;
    let n_windows = cap / cr;

    // `[1]`, not a scalar: Card 372c's guard needs an axis to reduce, and Card 371's exact lane has no I32
    // reshape.
    let token = b.slot(Slot::Token, TensorType::new(vec![1], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Csa);
    let (plan, dims, tokens, mut moe) =
        deepseek4_moe_preamble(&b, &cfg, &schedule, token, V4MoePhase::Decode)?;
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let iota_full = b.iota(cfg.max_pos);
    let iota_cap = b.slice(iota_full, 0, 0, cap);
    let pos_f = b.gather_scalar(iota_full, 0, pos_slot);
    let local_mask =
        gemma4_local_window_floor(&b, mask, pos_f, iota_cap, cfg.sliding_window, 1, cap);

    let cos = b.constant(
        V4_COMPRESS_ROPE_COS,
        TensorType::f32(vec![cfg.max_pos, rd / 2]),
    );
    let sin = b.constant(
        V4_COMPRESS_ROPE_SIN,
        TensorType::f32(vec![cfg.max_pos, rd / 2]),
    );
    let win_positions = b.slot_named(
        Slot::Activation,
        V4_CSA_WINDOW_POSITIONS,
        TensorType::f32(vec![n_windows]),
    );
    let csa_causal_mask = hca_decode_validity_mask(&b, pos_f, win_positions, cr);
    let csa_causal_mask = b.reshape(csa_causal_mask, vec![1, 1, 1, n_windows]);

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let x0 = b.cast(b.gather(embed, 0, tokens), DType::F32);
    let x0 = b.reshape(x0, vec![1, 1, 1, h]);
    let mut x = b.broadcast(x0, vec![1, 1, hc, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(5 * cfg.layers);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        let (new_x, pairs) = deepseek4_csa_layer_decode(
            &b,
            &cfg,
            li,
            x,
            cap,
            pos_slot,
            cos,
            sin,
            local_mask,
            csa_causal_mask,
            win_positions,
            n_windows,
            &plan,
            dims,
            &mut moe.context(),
        )?;
        x = new_x;
        state.extend(pairs);
    }

    let hh_fn = v4_mhc_projection(&b, plan.top_dense(V4TopDense::HcHeadProjection)?);
    let hh_base = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadBase)?);
    let hh_scale = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadScale)?);
    let collapsed_final = hyper_head(
        &b, x, hh_fn, hh_base, hh_scale, hc, h, 1, cfg.eps, cfg.hc_eps,
    );

    let norm = v4_source_f32(&b, plan.top_dense(V4TopDense::FinalNorm)?);
    let xf = rmsnorm(&b, collapsed_final, norm, cfg.eps);
    let logits = v4_source_linear(&b, xf, plan.top_dense(V4TopDense::Head)?);
    moe.finish(b, logits, &state)
}

/// [`deepseek4_csa_layer_decode`]'s PREFILL twin. `mask` and `csa_block_bias` are the layer-independent
/// `[1,1,l,l]`/`[1,1,l,n_windows]` constants; the function builds its own `combined_mask` because the top-k
/// selection is layer-specific.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_csa_layer_prefill(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    l: usize,
    cos: Traced,
    sin: Traced,
    mask: Traced,
    csa_block_bias: Traced,
    win_idx: Traced,
    n_windows: usize,
    plan: &DeepseekV4SourcePlan,
    dims: V4MoeDims,
    moe: &mut V4MoeContext<'_>,
) -> Result<Traced, DeepseekV4MoeError> {
    let (h, hq, d, rd) = (
        cfg.hidden,
        cfg.num_heads,
        cfg.head_dim,
        cfg.qk_rope_head_dim,
    );
    let nope_w = d - rd;
    let hc = cfg.hc_mult;
    let scale = cfg.attn_scale();
    let cr = cfg.csa_compress_rate;
    let (hi, di) = (cfg.index_n_heads, cfg.index_head_dim);
    let idx_nope_w = di - rd;
    let softmax_scale = 1.0 / (di as f32).sqrt();
    let weights_scaling = 1.0 / (hi as f32).sqrt();
    let topk = cfg.index_topk.min(n_windows);

    let ahc_fn = v4_mhc_projection(b, plan.layer_dense(li, V4LayerDense::HcAttnProjection)?);
    let ahc_base = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcAttnBase)?);
    let ahc_scale = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcAttnScale)?);
    let (post, comb, collapsed) = hyper_connection(
        b,
        x,
        ahc_fn,
        ahc_base,
        ahc_scale,
        hc,
        h,
        l,
        cfg.eps,
        cfg.hc_eps,
        cfg.hc_sinkhorn_iters,
    );

    let ln1 = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnNorm)?);
    let normed = rmsnorm(b, collapsed, ln1, cfg.eps);

    let q_a = v4_packed_linear(b, plan, li, V4PackedRole::WqA, normed)?;
    let q_res = rmsnorm(
        b,
        q_a,
        v4_source_f32(b, plan.layer_dense(li, V4LayerDense::QNorm)?),
        cfg.eps,
    );
    let q = v4_packed_linear(b, plan, li, V4PackedRole::WqB, q_res)?;
    let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]); // [1,Hq,L,D]
    let q = rmsnorm_no_weight(b, q, cfg.eps);

    let q_nope = b.slice(q, 3, 0, nope_w);
    let q_rope = b.slice(q, 3, nope_w, d);
    let q_rope = rope_interleaved_prefill(b, q_rope, cos, sin, l);
    let q = b.concat(3, &[q_nope, q_rope]);

    let kv = v4_packed_linear(b, plan, li, V4PackedRole::Wkv, normed)?; // [1,L,D]
    let kv = rmsnorm(
        b,
        kv,
        v4_source_f32(b, plan.layer_dense(li, V4LayerDense::KvNorm)?),
        cfg.eps,
    );
    let kv = b.reshape(kv, vec![1, 1, l, d]);
    let kv_nope = b.slice(kv, 3, 0, nope_w);
    let kv_rope = b.slice(kv, 3, nope_w, d);
    let kv_rope = rope_interleaved_prefill(b, kv_rope, cos, sin, l);
    let kv_full = b.concat(3, &[kv_nope, kv_rope]); // [1,1,L,D]

    let cwkv = plan.compressor_dense(li, V4CompressorDense::Kv)?;
    let cwgate = plan.compressor_dense(li, V4CompressorDense::Gate)?;
    let raw_kv = v4_source_linear(b, normed, cwkv); // [1,L,2*d]
    let raw_gate = v4_source_linear(b, normed, cwgate);
    let csa_chunk_kv = b.reshape(raw_kv, vec![1, n_windows, cr, 2 * d]);
    let csa_chunk_gate = b.reshape(raw_gate, vec![1, n_windows, cr, 2 * d]);
    let csa_pb = v4_source_f32(
        b,
        plan.compressor_dense(li, V4CompressorDense::PositionBias)?,
    );
    let csa_kv_norm_w = v4_source_f32(b, plan.compressor_dense(li, V4CompressorDense::Norm)?);
    let csa_pooled = csa_overlap_pool(
        b,
        csa_chunk_kv,
        csa_chunk_gate,
        csa_pb,
        csa_kv_norm_w,
        n_windows,
        cr,
        d,
        cfg.eps,
    );
    let csa_compressed = hca_window_rope(b, csa_pooled, cos, sin, win_idx, nope_w, rd);
    let csa_compressed = b.reshape(csa_compressed, vec![1, 1, n_windows, d]);

    let iwkv = plan.indexer_dense(li, V4IndexerDense::Kv)?;
    let iwgate = plan.indexer_dense(li, V4IndexerDense::Gate)?;
    let idx_raw_kv = v4_source_linear(b, normed, iwkv); // [1,L,2*di]
    let idx_raw_gate = v4_source_linear(b, normed, iwgate);
    let idx_chunk_kv = b.reshape(idx_raw_kv, vec![1, n_windows, cr, 2 * di]);
    let idx_chunk_gate = b.reshape(idx_raw_gate, vec![1, n_windows, cr, 2 * di]);
    let idx_pb = v4_source_f32(b, plan.indexer_dense(li, V4IndexerDense::PositionBias)?);
    let idx_kv_norm_w = v4_source_f32(b, plan.indexer_dense(li, V4IndexerDense::Norm)?);
    let idx_pooled = csa_overlap_pool(
        b,
        idx_chunk_kv,
        idx_chunk_gate,
        idx_pb,
        idx_kv_norm_w,
        n_windows,
        cr,
        di,
        cfg.eps,
    );
    let idx_compressed = hca_window_rope(b, idx_pooled, cos, sin, win_idx, idx_nope_w, rd);

    // The indexer's query up projection is an E4M3 packed pair and consumes the same `q_res` as the main Q.
    let q_idx = v4_packed_linear(b, plan, li, V4PackedRole::IndexerWqB, q_res)?; // [1,L,Hi*Di]
    let q_idx = b.transpose(b.reshape(q_idx, vec![1, l, hi, di]), vec![0, 2, 1, 3]); // [1,Hi,L,Di]
    let q_idx_nope = b.slice(q_idx, 3, 0, idx_nope_w);
    let q_idx_rope = b.slice(q_idx, 3, idx_nope_w, di);
    let q_idx_rope = rope_interleaved_prefill(b, q_idx_rope, cos, sin, l);
    let q_idx = b.concat(3, &[q_idx_nope, q_idx_rope]);

    let w_idx_raw = v4_source_linear(b, normed, plan.indexer_dense(li, V4IndexerDense::Weights)?); // [1,L,Hi]
    let w_idx = b.transpose(w_idx_raw, vec![0, 2, 1]); // [1,Hi,L]

    let index_scores = csa_indexer_scores(
        b,
        q_idx,
        idx_compressed,
        w_idx,
        hi,
        l,
        n_windows,
        di,
        softmax_scale,
        weights_scaling,
    );
    let csa_mask = csa_topk_mask(b, index_scores, csa_block_bias, topk);
    let combined_mask = b.concat(3, &[mask, csa_mask]);
    let kv_cat = b.concat(2, &[kv_full, csa_compressed]); // [1,1,L+n_windows,D]

    let sinks = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnSink)?);
    let attn =
        attention_prefill_with_sink(b, q, kv_cat, kv_cat, hq, scale, combined_mask, sinks, hq, l);
    let attn_nope = b.slice(attn, 3, 0, nope_w);
    let attn_rope = b.slice(attn, 3, nope_w, d);
    let attn_rope = rope_interleaved_prefill_inverse(b, attn_rope, cos, sin, l);
    let attn = b.concat(3, &[attn_nope, attn_rope]);
    let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [1,L,Hq,D] - token-major for the grouped out-proj

    // `attn.wo_a` is Card 385's block-diagonal packed linear; `attn.wo_b` is an ordinary packed linear.
    let grouped =
        deepseek4_grouped_out_a(b, plan, li, attn, l, hq, d, cfg.o_groups, cfg.o_lora_rank)?;
    let attn_out = v4_packed_linear(b, plan, li, V4PackedRole::WoB, grouped)?;

    let mut x = hyper_connection_combine(b, post, comb, attn_out, x, hc, l, h);

    let fhc_fn = v4_mhc_projection(b, plan.layer_dense(li, V4LayerDense::HcFfnProjection)?);
    let fhc_base = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcFfnBase)?);
    let fhc_scale = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::HcFfnScale)?);
    let (post2, comb2, collapsed2) = hyper_connection(
        b,
        x,
        fhc_fn,
        fhc_base,
        fhc_scale,
        hc,
        h,
        l,
        cfg.eps,
        cfg.hc_eps,
        cfg.hc_sinkhorn_iters,
    );
    let ln2 = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::FfnNorm)?);
    let normed2 = rmsnorm(b, collapsed2, ln2, cfg.eps);
    let mlp_out = deepseek4_routed_moe_ffn(b, cfg, plan, dims, li, normed2, moe)?;
    x = hyper_connection_combine(b, post2, comb2, mlp_out, x, hc, l, h);
    Ok(x)
}

/// Trace a full-sequence CSA PREFILL forward: the per-layer block of [`trace_deepseek4_csa_decode`] without an
/// incremental cache. Local K/V and both compressed branches (compressor and indexer) are computed once.
/// `mask` is a `[1,1,l,l]` local-window causal mask and `csa_block_bias` a `[1,1,l,n_windows]` additive
/// causal-completeness mask, both host-precomputed as in [`trace_deepseek4_hca_prefill`]. The indexer's top-`k`
/// selection ([`csa_topk_mask`]) depends on the weights, so it is computed in-graph on top of `csa_block_bias`.
#[cfg(test)]
pub(crate) fn trace_deepseek4_csa_prefill(
    cfg: DeepseekV4Config,
    seq_len: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    // `seq_len` need not be a multiple of `csa_compress_rate`; see `deepseek4_prefill_pad_len`.
    let l = deepseek4_prefill_pad_len(seq_len, &cfg, false, true);
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);
    let cr = cfg.csa_compress_rate;
    let n_windows = l / cr;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Csa);
    let (plan, dims, tokens, mut moe) =
        deepseek4_moe_preamble(&b, &cfg, &schedule, tokens, V4MoePhase::Prefill)?;
    let cos = b.constant(
        V4_COMPRESS_ROPE_COS,
        TensorType::f32(vec![cfg.max_pos, rd / 2]),
    );
    let sin = b.constant(
        V4_COMPRESS_ROPE_SIN,
        TensorType::f32(vec![cfg.max_pos, rd / 2]),
    );
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));
    let csa_block_bias = b.slot_named(
        Slot::Activation,
        V4_CSA_BLOCK_BIAS,
        TensorType::f32(vec![1, 1, l, n_windows]),
    );
    let win_idx = b.slot_named(
        Slot::Activation,
        V4_CSA_WINDOW_POSITIONS,
        TensorType::f32(vec![n_windows]),
    );

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let emb = b.cast(b.gather(embed, 0, tokens), DType::F32); // [L, hidden]
    let x0 = b.reshape(emb, vec![1, l, 1, h]);
    let mut x = b.broadcast(x0, vec![1, l, hc, h]);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        x = deepseek4_csa_layer_prefill(
            &b,
            &cfg,
            li,
            x,
            l,
            cos,
            sin,
            mask,
            csa_block_bias,
            win_idx,
            n_windows,
            &plan,
            dims,
            &mut moe.context(),
        )?;
    }

    let hh_fn = v4_mhc_projection(&b, plan.top_dense(V4TopDense::HcHeadProjection)?);
    let hh_base = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadBase)?);
    let hh_scale = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadScale)?);
    let collapsed_final = hyper_head(
        &b, x, hh_fn, hh_base, hh_scale, hc, h, l, cfg.eps, cfg.hc_eps,
    );

    let norm = v4_source_f32(&b, plan.top_dense(V4TopDense::FinalNorm)?);
    let xf = rmsnorm(&b, collapsed_final, norm, cfg.eps);
    // Real last token, not the padded one.
    let last = b.slice(xf, 1, seq_len - 1, seq_len);
    let logits = v4_source_linear(&b, last, plan.top_dense(V4TopDense::Head)?);
    moe.finish(b, logits, &[])
}
