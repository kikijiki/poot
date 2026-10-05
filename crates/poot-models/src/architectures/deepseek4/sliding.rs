#[cfg(test)]
use super::*;

/// Apply the inverse RoPE rotation (negated `sin`) to `x`'s trailing `rope_d`-wide slice through the same
/// interleaved-pair core as [`rope_interleaved_decode`]: the output-side derotation of
/// `DeepseekV4Attention.forward`. `x` must already be sliced to the rotated portion.
#[cfg(test)]
pub(crate) fn rope_interleaved_decode_inverse(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: Traced,
) -> Traced {
    let cos = b.gather_scalar(cos_table, 0, pos);
    let sin = b.gather_scalar(sin_table, 0, pos);
    let neg_sin = b.unary(UnOp::Neg, sin);
    rope_interleave_apply(b, x, cos, neg_sin)
}

/// [`rope_interleaved_decode_inverse`]'s full-sequence analog.
#[cfg(test)]
pub(crate) fn rope_interleaved_prefill_inverse(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    seq_len: usize,
) -> Traced {
    let cos = b.slice(cos_table, 0, 0, seq_len);
    let sin = b.slice(sin_table, 0, 0, seq_len);
    let neg_sin = b.unary(UnOp::Neg, sin);
    rope_interleave_apply(b, x, cos, neg_sin)
}

/// One `sliding_attention` layer's DECODE step, shared by [`trace_deepseek4_sliding_decode`] and
/// [`trace_deepseek4_hybrid_stack_decode`]. `local_mask` is `[1,1,1,cap]` (already window-floored, shared by
/// every layer kind, since every V4 layer keeps a local K==V cache); `cos`/`sin` are the layer's rope-type
/// tables. Returns the updated widened residual and the layer's state pair (the local K==V cache).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_sliding_layer_decode(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    cap: usize,
    pos_slot: Traced,
    local_mask: Traced,
    cos: Traced,
    sin: Traced,
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
    // State tensors are not checkpoint rows; they share the plan's `layers.{i}.` prefix so names read alike.
    let state_name = |s: &str| format!("{}.{s}", sources::v4_layer_prefix(li));
    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(1);

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

    let sinks = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnSink)?);
    let attn = attention_masked_with_sink(
        b,
        q,
        kv_cache_out,
        kv_cache_out,
        hq,
        scale,
        local_mask,
        sinks,
        hq,
    );
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

/// Trace a single `sliding_attention`-only DECODE step: one new token against a fixed-capacity single-head K==V
/// cache, with mHC at the attention and MLP sites and a final [`hyper_head`] collapse (spec 281 FR-001..FR-006).
/// `cap` is the physical cache capacity and may exceed `cfg.sliding_window`; the window is enforced by
/// [`gemma4_local_window_floor`].
#[cfg(test)]
pub(crate) fn trace_deepseek4_sliding_decode(
    cfg: DeepseekV4Config,
    cap: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);

    // `[1]`, not a scalar: Card 372c's guard needs an axis to reduce, and Card 371's exact lane has no I32
    // reshape.
    let token = b.slot(Slot::Token, TensorType::new(vec![1], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Sliding);
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

    let cos = b.constant(V4_ROPE_COS, TensorType::f32(vec![cfg.max_pos, rd / 2]));
    let sin = b.constant(V4_ROPE_SIN, TensorType::f32(vec![cfg.max_pos, rd / 2]));

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let x0 = b.cast(b.gather(embed, 0, tokens), DType::F32);
    let x0 = b.reshape(x0, vec![1, 1, 1, h]);
    let mut x = b.broadcast(x0, vec![1, 1, hc, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(cfg.layers);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        let (new_x, pairs) = deepseek4_sliding_layer_decode(
            &b,
            &cfg,
            li,
            x,
            cap,
            pos_slot,
            local_mask,
            cos,
            sin,
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

/// [`deepseek4_sliding_layer_decode`]'s PREFILL twin: the same per-layer body with no incremental cache. `mask`
/// is the caller-supplied `[1,1,l,l]` windowed-causal constant. Returns the updated `[1,l,hc,hidden]` residual.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_sliding_layer_prefill(
    b: &Builder,
    cfg: &DeepseekV4Config,
    li: usize,
    x: Traced,
    l: usize,
    mask: Traced,
    cos: Traced,
    sin: Traced,
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

    let sinks = v4_source_f32(b, plan.layer_dense(li, V4LayerDense::AttnSink)?);
    let attn = attention_prefill_with_sink(b, q, kv_full, kv_full, hq, scale, mask, sinks, hq, l);
    // [1,Hq,L,D], head-major. The inverse rope table is `[L,rd/2]` and needs L at axis -2, so derotate before the
    // token-major transpose (`apply_rotary_pos_emb(attn_output.transpose(1,2), cos, -sin).transpose(1,2)`).
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

/// Trace a full-sequence `sliding_attention`-only PREFILL forward: the per-layer block of
/// [`trace_deepseek4_sliding_decode`] without an incremental cache (K==V computed once over the sequence). The
/// `[1,1,l,l]` windowed-causal mask is the `mask.prefill` step input (card 550a), as in
/// `crate::deepseek32::trace_deepseek32_dsa_prefill`.
#[cfg(test)]
pub(crate) fn trace_deepseek4_sliding_prefill(
    cfg: DeepseekV4Config,
    seq_len: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    let b = Builder::new();
    let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Sliding);
    let (plan, dims, tokens, mut moe) =
        deepseek4_moe_preamble(&b, &cfg, &schedule, tokens, V4MoePhase::Prefill)?;
    let cos = b.constant(V4_ROPE_COS, TensorType::f32(vec![cfg.max_pos, rd / 2]));
    let sin = b.constant(V4_ROPE_SIN, TensorType::f32(vec![cfg.max_pos, rd / 2]));
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
    let emb = b.cast(b.gather(embed, 0, tokens), DType::F32); // [L, hidden]
    let x0 = b.reshape(emb, vec![1, l, 1, h]);
    let mut x = b.broadcast(x0, vec![1, l, hc, h]);

    for li in 0..cfg.layers {
        let _layer_scope = b.layer_scope(li);
        x = deepseek4_sliding_layer_prefill(
            &b,
            &cfg,
            li,
            x,
            l,
            mask,
            cos,
            sin,
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
    let last = b.slice(xf, 1, l - 1, l);
    let logits = v4_source_linear(&b, last, plan.top_dense(V4TopDense::Head)?);
    moe.finish(b, logits, &[])
}
