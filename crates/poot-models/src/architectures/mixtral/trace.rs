use super::*;

/// Mixtral's MoE shape, alongside a [`Qwen2Config`]. Unlike [`crate::qwen3moe::Qwen3MoeParams`] there is no
/// `sparse_layer` field: every layer routes through the expert mixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixtralParams {
    /// Total experts per layer (`num_local_experts`; `8` on the 8x7B).
    pub n_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`; `2` on the 8x7B).
    pub top_k: usize,
    /// Per-expert FFN intermediate size (`intermediate_size`; `14336` on the 8x7B). There is no separate dense width.
    pub inter: usize,
}

/// Trace a full-sequence Mixtral prefill forward (the CPU-oracle path): embeddings, `cfg.layers` pre-norm blocks
/// (split Q/K/V, RoPE, causal GQA attention, `o_proj`, top-`mp.top_k`-of-`mp.n_experts` MoE MLP, residuals),
/// final `model.norm`, `lm_head.weight`. No scalar multipliers, no QK-norm; every layer routes through
/// [`poot_graph_ir::ops::moe`].
///
/// Named constants bound at eval time: `model.embed_tokens.weight` `[vocab,hidden]`, `rope.cos`/`rope.sin`
/// `[max_pos,head_dim]` (one shared table; `rope_theta` is a loader concern), per-layer `model.layers.{li}.{input_layernorm,post_attention_layernorm}.weight`,
/// `self_attn.{q,k,v,o}_proj.weight` (no bias), `block_sparse_moe.gate.weight` `[hidden, n_experts]`,
/// `block_sparse_moe.experts.gate_up_proj.weight` `[n_experts,hidden,2*inter]`,
/// `block_sparse_moe.experts.down_proj.weight` `[n_experts,inter,hidden]`, `model.norm.weight`, and
/// `lm_head.weight` (`build_weights` materializes it densely even when untied); plus the `mask.prefill`
/// step input (card 550a) `[1,1,l,l]` (plain causal).
pub fn trace_mixtral_prefill(cfg: Qwen2Config, mp: MixtralParams, seq_len: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );

        let q = linear(&b, normed, wq, None);
        let k = linear(&b, normed, wk, None);
        let v = linear(&b, normed, wv, None);

        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

        let attn = attention_prefill(&b, q, k, v, n_rep, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = mixtral_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// Mixtral single-token fixed-KV masked decode, the decode analog of [`trace_mixtral_prefill`] on
/// `crate::qwen2::trace_decode_kv_masked`'s skeleton. Same block shape as prefill.
pub fn trace_mixtral_decode_kv_masked(cfg: Qwen2Config, mp: MixtralParams, cap: usize) -> Graph {
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let cos = b.constant(
        "rope.cos",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );
    let sin = b.constant(
        "rope.sin",
        TensorType::f32(vec![cfg.max_pos, cfg.rotary_dim]),
    );

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather_scalar(embed, 0, token);
    let mut x = b.reshape(x0, vec![1, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );

        let q = linear(&b, normed, wq, None);
        let k = linear(&b, normed, wk, None);
        let v = linear(&b, normed, wv, None);

        let q = b.transpose(b.reshape(q, vec![1, 1, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, 1, hkv, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, 1, hkv, d]), vec![0, 2, 1, 3]);

        let q = rope(&b, q, cos, sin, pos_slot);
        let k = rope(&b, k, cos, sin, pos_slot);

        let kcache = b.state_input(
            &p("kv.k_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let vcache = b.state_input(
            &p("kv.v_cache"),
            TensorType::f32(vec![1, hkv, cap, d]),
            StateRole::Recurrent,
        );
        let kcache_out = b.dynamic_update_slice_dyn(kcache, k, pos_slot, 2);
        let vcache_out = b.dynamic_update_slice_dyn(vcache, v, pos_slot, 2);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

        let attn = attention_masked(&b, q, kcache_out, vcache_out, n_rep, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = mixtral_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// Mixtral's batched shared-pool decode (spec 266-batched Phase 1). The plain dense `[E,...]`-stacked-weights
/// path, not the expert-pool one: each layer's FFN is the same [`mixtral_ffn`] call as
/// [`trace_mixtral_decode_kv_masked`], re-presented over `batch` concurrently decoding rows through a shared
/// `[pool_slots, Hkv,D]` K/V pool per layer addressed by a `[batch,cap]` global slot map. It calls the generic
/// core [`crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`] that
/// `crate::granite::trace_granite_decode_kv_masked_batched_shared_pool` and
/// `crate::qwen3moe::trace_qwen3_moe_decode_kv_masked_batched_shared_pool` use, closing over Mixtral's
/// `qk_norm=false`/no-scalar-multiplier attention block and [`mixtral_ffn`] as the `ffn` closure
/// (`embed_mult`/`residual_mult`/`logits_mult` all `1.0`).
///
/// Weight-quantized (packed GGUF) experts are not supported, as for the granitemoe/qwen3_moe wrappers; the
/// caller (`Runner::trace_batched_shared_pool_decode`) rejects `quant_kv` for Mixtral first.
pub fn trace_mixtral_decode_kv_masked_batched_shared_pool(
    cfg: Qwen2Config,
    mp: MixtralParams,
    cap: usize,
    batch: usize,
    pool_slots: usize,
) -> Graph {
    let h = cfg.hidden;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool(
        cfg,
        false, // no per-head QK-norm (unlike qwen3-moe)
        1.0,   // no embedding multiplier (unlike granite)
        scale,
        1.0, // no residual multiplier (unlike granite)
        1.0, // no logits scale (unlike granite)
        cap,
        batch,
        pool_slots,
        move |b, normed, li| mixtral_ffn(b, normed, h, &mp, li),
    )
}

/// Mixtral's shared-pool paged prefill (spec 266-batched Phase 1): the prefill half of
/// [`trace_mixtral_decode_kv_masked_batched_shared_pool`], so a loaded Mixtral `Runner` can be admitted into the
/// shared-pool serving path. A thin wrapper around [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`], as
/// `crate::granite::trace_granite_prefill_kv_shared_pool` and
/// `crate::qwen3moe::trace_qwen3_moe_prefill_kv_shared_pool`, closing over Mixtral's `qk_norm=false` attention
/// block and [`mixtral_ffn`].
pub fn trace_mixtral_prefill_kv_shared_pool(
    cfg: Qwen2Config,
    mp: MixtralParams,
    n: usize,
    pool: usize,
) -> Graph {
    let h = cfg.hidden;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    crate::moe_prefill::trace_moe_prefill_kv_shared_pool(
        cfg,
        false, // no per-head QK-norm (unlike qwen3-moe)
        1.0,   // no embedding multiplier (unlike granite)
        scale,
        1.0, // no residual multiplier (unlike granite)
        1.0, // no logits scale (unlike granite)
        n,
        pool,
        move |b, normed, li| mixtral_ffn(b, normed, h, &mp, li),
    )
}

/// One layer's MoE MLP: every Mixtral layer routes through [`poot_graph_ir::ops::moe`] (no dense-layer branch,
/// unlike `crate::qwen3moe::ffn`). Shared by both trace functions so `block_sparse_moe.*` naming lives in one place.
pub(crate) fn mixtral_ffn(
    b: &Builder,
    normed: Traced,
    h: usize,
    mp: &MixtralParams,
    li: usize,
) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let router = b.constant(
        &p("block_sparse_moe.gate.weight"),
        TensorType::f32(vec![h, mp.n_experts]),
    );
    let w_in = b.constant(
        &p("block_sparse_moe.experts.gate_up_proj.weight"),
        TensorType::f32(vec![mp.n_experts, h, 2 * mp.inter]),
    );
    let w_out = b.constant(
        &p("block_sparse_moe.experts.down_proj.weight"),
        TensorType::f32(vec![mp.n_experts, mp.inter, h]),
    );
    moe(
        b,
        normed,
        router,
        w_in,
        w_out,
        mp.n_experts,
        mp.top_k,
        mp.inter,
    )
}
