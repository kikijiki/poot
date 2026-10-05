#[cfg(test)]
use super::*;

/// Additive mask for causally invalid slots (a compressed slot not yet reachable, or an unwritten padding row);
/// same role and magnitude as `crate::deepseek32::DSA_MASK_NEG`.
#[cfg(test)]
pub(crate) const HCA_MASK_NEG: f32 = -1.0e30;

/// HCA's softmax-gated window pooling (`DeepseekV4HCACompressor.forward`, paper section 2.3.2 eqs 20-23).
/// `chunk_kv`/`chunk_gate` are `[1, n_windows, compress_rate, head_dim]` (window-reshaped; `chunk_gate` is the RAW
/// gate projection, this function adds `position_bias`, `[compress_rate, head_dim]`). Returns
/// `[1, n_windows, head_dim]`, after `kv_norm` and before RoPE (see [`hca_window_rope`]; decode and prefill
/// differ only in which absolute positions each window maps to).
///
/// The softmax runs over the window-position axis independently per `head_dim` channel (`.softmax(dim=2)`),
/// not over the trailing axis. [`poot_graph_ir::ops::softmax`] only normalizes its input's trailing axis, so
/// the shared core transposes the window axis to the end and back around that call; the weighted-sum pool
/// below reduces the window axis directly (card 523b: `compile` legalizes a non-last-axis reduce for every
/// target).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn hca_softmax_gate_pool(
    b: &Builder,
    chunk_kv: Traced,
    chunk_gate: Traced,
    position_bias: Traced,
    kv_norm_w: Traced,
    compress_rate: usize,
    head_dim: usize,
    eps: f32,
) -> Traced {
    let pb = b.reshape(position_bias, vec![1, 1, compress_rate, head_dim]);
    let gate_biased = b.binary(BinOp::Add, chunk_gate, pb); // broadcasts over the n_windows axis
    softmax_gate_pool_core(b, chunk_kv, gate_biased, kv_norm_w, eps)
}

/// Softmax over the window axis, weighted sum, and `kv_norm`, shared by [`hca_softmax_gate_pool`] (bias added at
/// `compress_rate` width) and [`csa_overlap_pool`] (bias added and window doubled to `2*compress_rate` first).
/// `kv`/`gate_biased` are `[1, n_windows, window_width, head_dim]`; `gate_biased` must already carry its
/// position bias. The softmax is per `head_dim` channel over the window axis: `gate_biased` transposes that
/// axis to the trailing position for [`poot_graph_ir::ops::softmax`] and back; the weighted-sum pool then
/// reduces the window axis (2) directly.
#[cfg(test)]
pub(crate) fn softmax_gate_pool_core(
    b: &Builder,
    kv: Traced,
    gate_biased: Traced,
    kv_norm_w: Traced,
    eps: f32,
) -> Traced {
    let gate_t = b.transpose(gate_biased, vec![0, 1, 3, 2]); // [1,n_windows,head_dim,window_width]
    let sm_t = softmax(b, gate_t); // softmax over window_width, now the trailing axis
    let sm = b.transpose(sm_t, vec![0, 1, 3, 2]); // back to [1,n_windows,window_width,head_dim]
    let weighted = b.binary(BinOp::Mul, kv, sm);
    let pooled = b.reduce(RedOp::Sum, weighted, 2, false); // [1,n_windows,head_dim]
    rmsnorm(b, pooled, kv_norm_w, eps)
}

/// Apply the trailing-slice interleaved partial RoPE to a `[1, n, head_dim]` batch of compressed entries at
/// caller-supplied absolute positions (`pos_idx`, a `[n]` constant). Window `w` sits at `w * compress_rate`, the
/// position of its first source token, which is known at trace time for both decode
/// ([`trace_deepseek4_hca_decode`]) and prefill ([`trace_deepseek4_hca_prefill`]), unlike Q/K's runtime decode
/// step (`rotary_emb(compressed, position_ids=positions, layer_type="compress")`).
#[cfg(test)]
pub(crate) fn hca_window_rope(
    b: &Builder,
    pooled: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos_idx: Traced,
    nope_w: usize,
    rd: usize,
) -> Traced {
    let nope = b.slice(pooled, 2, 0, nope_w);
    let rope_part = b.slice(pooled, 2, nope_w, nope_w + rd);
    let cos = b.gather(cos_table, 0, pos_idx); // [n, rd/2]
    let sin = b.gather(sin_table, 0, pos_idx);
    let rotated = rope_interleave_apply(b, rope_part, cos, sin);
    b.concat(2, &[nope, rotated])
}

/// Causal validity of the HCA compressed cache's `n_windows_cap` physical slots at decode step `pos_f` (a traced
/// scalar, so one capture/replay covers every step), as an additive mask (`0.0` valid, `HCA_MASK_NEG` invalid or
/// not yet written), `[n_windows_cap]`.
///
/// The reference uses `causal_threshold = (pos+1) // compress_rate` with window `w` valid iff
/// `w < causal_threshold`. For integers that is `pos >= w*cr + cr - 1`: a window is valid once its LAST source
/// token (`w*cr + cr - 1`) has been processed. This is not the window's first-token position `w*cr` that
/// [`hca_window_rope`] rotates to; conflating the two lets a still-forming window leak into attention. It is one
/// [`BinOp::Ge`] against `win_positions[w] + (compress_rate - 1)` (`poot_graph_ir` has no floor division), and
/// also excludes unwritten physical slots, which have `w*cr + cr - 1 > pos`.
#[cfg(test)]
pub(crate) fn hca_decode_validity_mask(
    b: &Builder,
    pos_f: Traced,
    win_positions: Traced,
    compress_rate: usize,
) -> Traced {
    let close_pos = b.binary_scalar(
        BinOp::Add,
        win_positions,
        Scalar::F32((compress_rate - 1) as f32),
    );
    let valid = b.binary(BinOp::Ge, pos_f, close_pos); // [n] - 1.0 valid, 0.0 invalid
    let invalid = b.binary_scalar(BinOp::Sub, valid, Scalar::F32(1.0)); // 0.0 valid, -1.0 invalid
    b.binary_scalar(BinOp::Mul, invalid, Scalar::F32(-HCA_MASK_NEG)) // 0.0 valid, HCA_MASK_NEG invalid
}

/// One `heavily_compressed_attention` layer's DECODE step, shared by [`trace_deepseek4_hca_decode`] and the
/// hybrid stack. `combined_mask` is `concat(local_mask, hca_mask)` (`[1,1,1,cap+n_windows]`); it needs no
/// per-layer weights, so the caller computes it once for every HCA layer. `cos`/`sin` are the `"compress"`
/// rope-type tables. Returns the updated widened residual and the layer's three state pairs (local K==V cache,
/// HCA raw kv and gate history).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_hca_layer_decode(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    cap: usize,
    pos_slot: Traced,
    cos: Traced,
    sin: Traced,
    combined_mask: Traced,
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
    let cr = cfg.hca_compress_rate;
    // State tensors are not checkpoint rows; they share the plan's `layers.{i}.` prefix so names read alike.
    let state_name = |s: &str| format!("{}.{s}", sources::v4_layer_prefix(li));
    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(3);

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

    // HCA compressed branch: raw kv/gate projections of this token go into an absolute-position history cache,
    // then are pooled over every window the full (capacity-sized, causal-mask-guarded) history admits.
    let hwkv = plan.compressor_dense(li, V4CompressorDense::Kv)?;
    let hwgate = plan.compressor_dense(li, V4CompressorDense::Gate)?;
    let hca_kv_new = b.reshape(v4_source_linear(b, normed, hwkv), vec![1, 1, 1, d]);
    let hca_gate_new = b.reshape(v4_source_linear(b, normed, hwgate), vec![1, 1, 1, d]);

    let hca_kv_cache = b.state_input(
        &state_name("self_attn.hca.kv_raw_cache"),
        TensorType::f32(vec![1, 1, cap, d]),
        StateRole::Recurrent,
    );
    let hca_kv_cache_out = b.dynamic_update_slice_dyn(hca_kv_cache, hca_kv_new, pos_slot, 2);
    state.push((hca_kv_cache, hca_kv_cache_out));
    let hca_gate_cache = b.state_input(
        &state_name("self_attn.hca.gate_raw_cache"),
        TensorType::f32(vec![1, 1, cap, d]),
        StateRole::Recurrent,
    );
    let hca_gate_cache_out = b.dynamic_update_slice_dyn(hca_gate_cache, hca_gate_new, pos_slot, 2);
    state.push((hca_gate_cache, hca_gate_cache_out));

    let chunk_kv = b.reshape(hca_kv_cache_out, vec![1, n_windows, cr, d]);
    let chunk_gate = b.reshape(hca_gate_cache_out, vec![1, n_windows, cr, d]);
    let position_bias = v4_source_f32(
        b,
        plan.compressor_dense(li, V4CompressorDense::PositionBias)?,
    );
    let hca_kv_norm_w = v4_source_f32(b, plan.compressor_dense(li, V4CompressorDense::Norm)?);
    let pooled = hca_softmax_gate_pool(
        b,
        chunk_kv,
        chunk_gate,
        position_bias,
        hca_kv_norm_w,
        cr,
        d,
        cfg.eps,
    );
    let compressed = hca_window_rope(b, pooled, cos, sin, win_idx, nope_w, rd);
    let compressed = b.reshape(compressed, vec![1, 1, n_windows, d]);

    let kv_full = b.concat(2, &[kv_cache_out, compressed]); // [1,1,cap+n_windows,d]

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

/// Trace a single HCA (`heavily_compressed_attention`-only) DECODE step. Extends
/// [`trace_deepseek4_sliding_decode`]'s per-layer body with HCA's compressed branch: two extra per-layer state
/// tensors (`self_attn.hca.kv_raw_cache`/`self_attn.hca.gate_raw_cache`, `[1,1,cap,head_dim]`, the raw per-token
/// `kv_proj`/`gate_proj` history indexed by absolute position), pooled into `n_windows_cap = cap /
/// cfg.hca_compress_rate` compressed entries every step and concatenated onto the local K==V before the
/// sink-attention call (`kv = torch.cat([kv, compressed_kv], dim=2)`). Recomputing from the full history is
/// exact: the causal-validity mask (see [`hca_decode_validity_mask`]) keeps incomplete windows out.
///
/// The whole layer (Q, local K/V, compressed entries, output derotation) uses `cfg.compress_rope_theta`'s
/// table (`rope_layer_type = "compress"` for every non-sliding layer). `cap` must be an exact multiple of
/// `cfg.hca_compress_rate`.
#[cfg(test)]
pub(crate) fn trace_deepseek4_hca_decode(
    cfg: DeepseekV4Config,
    cap: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    assert!(
        cap.is_multiple_of(cfg.hca_compress_rate) && cfg.hca_compress_rate > 0,
        "trace_deepseek4_hca_decode: cap must be an exact multiple of hca_compress_rate"
    );
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);
    let cr = cfg.hca_compress_rate;
    let n_windows = cap / cr;

    // `[1]`, not a scalar: Card 372c's guard needs an axis to reduce, and Card 371's exact lane has no I32
    // reshape.
    let token = b.slot(Slot::Token, TensorType::new(vec![1], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Hca);
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
        V4_HCA_WINDOW_POSITIONS,
        TensorType::f32(vec![n_windows]),
    );
    let hca_mask = hca_decode_validity_mask(&b, pos_f, win_positions, cr);
    let hca_mask = b.reshape(hca_mask, vec![1, 1, 1, n_windows]);
    let combined_mask = b.concat(3, &[local_mask, hca_mask]);

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let x0 = b.cast(b.gather(embed, 0, tokens), DType::F32);
    let x0 = b.reshape(x0, vec![1, 1, 1, h]);
    let mut x = b.broadcast(x0, vec![1, 1, hc, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(3 * cfg.layers);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        let (new_x, pairs) = deepseek4_hca_layer_decode(
            &b,
            &cfg,
            li,
            x,
            cap,
            pos_slot,
            cos,
            sin,
            combined_mask,
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

/// [`deepseek4_hca_layer_decode`]'s PREFILL twin. `combined_mask` is `concat(mask, hca_block_bias)`
/// (`[1,1,l,l+n_windows]`), layer-independent and computed once by the caller.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_hca_layer_prefill(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    l: usize,
    cos: Traced,
    sin: Traced,
    combined_mask: Traced,
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
    let cr = cfg.hca_compress_rate;

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

    let hwkv = plan.compressor_dense(li, V4CompressorDense::Kv)?;
    let hwgate = plan.compressor_dense(li, V4CompressorDense::Gate)?;
    let raw_kv = v4_source_linear(b, normed, hwkv); // [1,L,D]
    let raw_gate = v4_source_linear(b, normed, hwgate);
    let chunk_kv = b.reshape(raw_kv, vec![1, n_windows, cr, d]);
    let chunk_gate = b.reshape(raw_gate, vec![1, n_windows, cr, d]);
    let position_bias = v4_source_f32(
        b,
        plan.compressor_dense(li, V4CompressorDense::PositionBias)?,
    );
    let hca_kv_norm_w = v4_source_f32(b, plan.compressor_dense(li, V4CompressorDense::Norm)?);
    let pooled = hca_softmax_gate_pool(
        b,
        chunk_kv,
        chunk_gate,
        position_bias,
        hca_kv_norm_w,
        cr,
        d,
        cfg.eps,
    );
    let compressed = hca_window_rope(b, pooled, cos, sin, win_idx, nope_w, rd);
    let compressed = b.reshape(compressed, vec![1, 1, n_windows, d]);

    let kv_cat = b.concat(2, &[kv_full, compressed]); // [1,1,L+n_windows,D]

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

/// Trace a full-sequence HCA PREFILL forward: the per-layer block of [`trace_deepseek4_hca_decode`] without an
/// incremental cache. Local K/V and the compressed branch are computed once over the sequence. The
/// windowed-causal `mask.prefill` step input (card 550a) and the `[1,1,l,n_windows]` additive
/// `V4_HCA_BLOCK_BIAS` step input (`0.0`/`HCA_MASK_NEG`) are host-precomputed. `seq_len` need not be a
/// multiple of `cfg.hca_compress_rate`: the graph is traced at the padded length (see
/// [`deepseek4_prefill_pad_len`]).
#[cfg(test)]
pub(crate) fn trace_deepseek4_hca_prefill(
    cfg: DeepseekV4Config,
    seq_len: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    // Trace at the padded length; padding cannot change a real position's output (see
    // `deepseek4_prefill_pad_len`), and the final logits slice below reads the real last token.
    let l = deepseek4_prefill_pad_len(seq_len, &cfg, true, false);
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);
    let cr = cfg.hca_compress_rate;
    let n_windows = l / cr;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Hca);
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
    let hca_block_bias = b.slot_named(
        Slot::Activation,
        V4_HCA_BLOCK_BIAS,
        TensorType::f32(vec![1, 1, l, n_windows]),
    );
    let combined_mask = b.concat(3, &[mask, hca_block_bias]);
    let win_idx = b.slot_named(
        Slot::Activation,
        V4_HCA_WINDOW_POSITIONS,
        TensorType::f32(vec![n_windows]),
    );

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let emb = b.cast(b.gather(embed, 0, tokens), DType::F32); // [L, hidden]
    let x0 = b.reshape(emb, vec![1, l, 1, h]);
    let mut x = b.broadcast(x0, vec![1, l, hc, h]);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        x = deepseek4_hca_layer_prefill(
            &b,
            &cfg,
            li,
            x,
            l,
            cos,
            sin,
            combined_mask,
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
