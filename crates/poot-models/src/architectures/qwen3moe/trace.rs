use super::*;

/// Qwen3-MoE deltas alongside a [`Qwen2Config`] (`qkv_bias: false`, `qk_norm: true`).
#[derive(Clone, Debug)]
pub struct Qwen3MoeParams {
    /// Routed experts per MoE layer (`num_experts`).
    pub n_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`). Softmax is renormalized over these
    /// (`norm_topk_prob`); the shared `moe` op supports only the renormalized form.
    pub top_k: usize,
    /// Per-expert FFN intermediate size (`moe_intermediate_size`), not `Qwen2Config::inter` (the dense size).
    pub inter: usize,
    /// Per layer: `true` routes through the expert mixture, `false` keeps a dense swiglu MLP at `cfg.inter`.
    /// Length must equal `cfg.layers`.
    pub sparse_layer: Vec<bool>,
}

impl Qwen3MoeParams {
    pub(crate) fn is_sparse(&self, li: usize) -> bool {
        self.sparse_layer.get(li).copied().unwrap_or_else(|| {
            panic!("Qwen3MoeParams::sparse_layer is shorter than cfg.layers (missing layer {li})")
        })
    }
}

/// One decoder block's FFN half: dense swiglu (`cfg.inter`) or the routed expert mixture (`mp.inter`),
/// selected per `mp.sparse_layer[li]`.
pub(crate) fn ffn(
    b: &Builder,
    normed: Traced,
    cfg: &Qwen2Config,
    mp: &Qwen3MoeParams,
    li: usize,
) -> Traced {
    let h = cfg.hidden;
    let p = |s: &str| format!("model.layers.{li}.{s}");
    if mp.is_sparse(li) {
        let router = b.constant(
            &p("mlp.gate.weight"),
            TensorType::f32(vec![h, mp.n_experts]),
        );
        let w_in = b.constant(
            &p("mlp.experts.gate_up_proj.weight"),
            TensorType::f32(vec![mp.n_experts, h, 2 * mp.inter]),
        );
        let w_out = b.constant(
            &p("mlp.experts.down_proj.weight"),
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
    } else {
        let wg = b.constant(
            &p("mlp.gate_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wu = b.constant(
            &p("mlp.up_proj.weight"),
            TensorType::f32(vec![h, cfg.inter]),
        );
        let wd = b.constant(
            &p("mlp.down_proj.weight"),
            TensorType::f32(vec![cfg.inter, h]),
        );
        let gate = linear(b, normed, wg, None);
        let up = linear(b, normed, wu, None);
        let act = swiglu(b, gate, up);
        linear(b, act, wd, None)
    }
}

/// Trace a full-sequence Qwen3-MoE prefill forward (CPU coherence path, no KV-cache writes). Mirrors
/// [`crate::granite::trace_granite_prefill`] without the Granite scalars.
pub fn trace_qwen3_moe_prefill(cfg: Qwen2Config, mp: Qwen3MoeParams, seq_len: usize) -> Graph {
    assert_eq!(
        mp.sparse_layer.len(),
        cfg.layers,
        "Qwen3MoeParams::sparse_layer must have one entry per layer"
    );
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
    let emb = b.gather(embed, 0, tokens);
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

        let mut q = b.reshape(q, vec![1, l, hq, d]);
        let mut k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        // per-head QK-norm: RMSNorm over head_dim, before the transpose to [1,H,L,D] and RoPE.
        let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
        let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
        q = rmsnorm(&b, q, qn, cfg.eps);
        k = rmsnorm(&b, k, kn, cfg.eps);

        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

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
        let m = ffn(&b, normed, &cfg, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// Batched prefill with KV-cache writes ([`trace_qwen3_moe_prefill`] on the skeleton of
/// [`crate::qwen2::trace_prefill_kv`]; contiguous cache only). Fills each layer's `[1,Hkv,cap,D]` K/V cache
/// for positions `[0,N)`, with the state names/order [`trace_qwen3_moe_decode_kv_masked`] expects.
pub fn trace_qwen3_moe_prefill_kv(
    cfg: Qwen2Config,
    mp: Qwen3MoeParams,
    n: usize,
    cap: usize,
) -> Graph {
    assert_eq!(
        mp.sparse_layer.len(),
        cfg.layers,
        "Qwen3MoeParams::sparse_layer must have one entry per layer"
    );
    assert!(
        cap >= n,
        "cache capacity {cap} must be at least the prompt length {n}"
    );
    let b = Builder::new();
    let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let n_rep = hq / hkv;
    let q_dim = hq * d;
    let kv_dim = hkv * d;
    let scale = 1.0 / (d as f32).sqrt();
    let l = n;

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
    let emb = b.gather(embed, 0, tokens);
    let mut x = b.reshape(emb, vec![1, l, h]);

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

        let mut q = b.reshape(q, vec![1, l, hq, d]);
        let mut k = b.reshape(k, vec![1, l, hkv, d]);
        let v = b.reshape(v, vec![1, l, hkv, d]);
        let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
        let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
        q = rmsnorm(&b, q, qn, cfg.eps);
        k = rmsnorm(&b, k, kn, cfg.eps);

        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

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
        let kcache_out = b.dynamic_update_slice(kcache, k, 0, 2);
        let vcache_out = b.dynamic_update_slice(vcache, v, 0, 2);
        state.push((kcache, kcache_out));
        state.push((vcache, vcache_out));

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
        let m = ffn(&b, normed, &cfg, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// Shared-pool paged prefill (spec 249): the qwen3-moe equivalent of
/// [`crate::qwen2::trace_prefill_kv_shared_pool_ext`]. Delegates to
/// [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`] with qwen3-moe's per-head QK-norm and the
/// per-layer [`ffn`] switch. `pool` is the shared physical pool size (`pool >= n`).
pub fn trace_qwen3_moe_prefill_kv_shared_pool(
    cfg: Qwen2Config,
    mp: Qwen3MoeParams,
    n: usize,
    pool: usize,
) -> Graph {
    assert_eq!(
        mp.sparse_layer.len(),
        cfg.layers,
        "Qwen3MoeParams::sparse_layer must have one entry per layer"
    );
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    crate::moe_prefill::trace_moe_prefill_kv_shared_pool(
        cfg,
        true, // qwen3: per-head QK-norm
        1.0,  // no embedding multiplier
        scale,
        1.0, // no residual multiplier
        1.0, // no logits scale
        n,
        pool,
        move |b, normed, li| ffn(b, normed, &cfg, &mp, li),
    )
}

/// Batched shared-pool decode (spec 249): the qwen3-moe equivalent of
/// [`crate::qwen2::trace_decode_kv_masked_batched_shared_pool_ext`]. Delegates to
/// [`crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`] with qwen3-moe's per-head QK-norm
/// and the per-layer [`ffn`] switch. `cap` is each row's logical capacity, `batch` the number of
/// concurrently decoding rows, `pool_slots` the shared physical pool size.
pub fn trace_qwen3_moe_decode_kv_masked_batched_shared_pool(
    cfg: Qwen2Config,
    mp: Qwen3MoeParams,
    cap: usize,
    batch: usize,
    pool_slots: usize,
) -> Graph {
    assert_eq!(
        mp.sparse_layer.len(),
        cfg.layers,
        "Qwen3MoeParams::sparse_layer must have one entry per layer"
    );
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool(
        cfg,
        true, // qwen3: per-head QK-norm
        1.0,  // no embedding multiplier
        scale,
        1.0, // no residual multiplier
        1.0, // no logits scale
        cap,
        batch,
        pool_slots,
        move |b, normed, li| ffn(b, normed, &cfg, &mp, li),
    )
}

/// Single-token fixed-KV masked decode, on [`crate::qwen2::trace_decode_kv_masked`]'s skeleton, with the
/// same per-layer dense/MoE FFN switch as the prefill variants.
pub fn trace_qwen3_moe_decode_kv_masked(cfg: Qwen2Config, mp: Qwen3MoeParams, cap: usize) -> Graph {
    assert_eq!(
        mp.sparse_layer.len(),
        cfg.layers,
        "Qwen3MoeParams::sparse_layer must have one entry per layer"
    );
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

        let mut q = b.reshape(q, vec![1, 1, hq, d]);
        let mut k = b.reshape(k, vec![1, 1, hkv, d]);
        let v = b.reshape(v, vec![1, 1, hkv, d]);
        let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![d]));
        let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![d]));
        q = rmsnorm(&b, q, qn, cfg.eps);
        k = rmsnorm(&b, k, kn, cfg.eps);

        let q = b.transpose(q, vec![0, 2, 1, 3]);
        let k = b.transpose(k, vec![0, 2, 1, 3]);
        let v = b.transpose(v, vec![0, 2, 1, 3]);

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
        let m = ffn(&b, normed, &cfg, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}
