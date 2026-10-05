//! gpt-oss tracer (card 135d, `specs/263-gptoss-tracer/spec.md`) for `openai/gpt-oss-20b` and
//! `openai/gpt-oss-120b`. Checked against the published `config.json`, `transformers`'
//! `modeling_gpt_oss.py`, and the tensor names/shapes of the BF16 `tiny-random/gpt-oss` checkpoint.
//!
//! Differences from Mixtral/OlmoE:
//!
//! 1. **Attention sinks.** `GptOssAttention` has a per-query-head learned scalar `self.sinks`. Real
//!    `eager_attention_forward`:
//!    ```text
//!    combined_logits = torch.cat([attn_weights, sinks_broadcast], dim=-1)
//!    combined_logits -= combined_logits.max(dim=-1, keepdim=True).values
//!    probs = softmax(combined_logits, dim=-1)
//!    scores = probs[..., :-1]   # the sink's own probability mass is discarded
//!    ```
//!    The row max and softmax denominator include the sink logit, but the sink adds nothing to the V sum.
//!    poot's IR has no concat op; folding the sink value into the max/sum over the real columns is
//!    equivalent ([`poot_graph_ir::ops::attention_prefill_with_sink`],
//!    [`poot_graph_ir::ops::attention_masked_with_sink`]).
//! 2. **Alternating sliding-window / full attention layers.** The config's `layer_types` is an explicit
//!    per-layer array (strictly alternating on the real 20b/120b, 24 layers, and the 2-layer tiny fixture, but
//!    read from the array); `sliding_window: 128`. Prefill binds two mask step inputs (`mask.prefill` full,
//!    `mask.prefill.local` windowed) and picks per layer. Decode reuses
//!    [`crate::gemma4::gemma4_local_window_floor`] (an in-graph `-1e9 * max(0, (pos-w+1)-t)` floor added onto
//!    the single shared `Slot::Mask`) with a `Vec<bool>` schedule filled from `layer_types`.
//! 3. **YaRN RoPE scaling** (`rope_type: "yarn", factor: 32.0, beta_fast: 32.0, beta_slow: 1.0,
//!    original_max_position_embeddings: 4096`, `rope_theta: 150000`). This is host-side rope-table
//!    computation (`poot_llm::gguf::rope_tables`, shared by every arch's safetensors loader via
//!    `hf.rope_scaling`), not a graph concern: `rope`/`rope_prefill` read whatever `cos`/`sin` the loader filled.
//!
//! **Router.** Real `GptOssTopKRouter.forward`:
//! ```text
//! router_logits = F.linear(hidden_states, self.weight, self.bias)   # with a router bias
//! router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)   # topk on raw logits first
//! router_scores = softmax(router_top_value, dim=1)                  # softmax over the k selected values
//! ```
//! Top-k on raw logits picks the same experts as top-k on softmax probabilities, and softmax over the k
//! selected raw logits is exactly what [`poot_graph_ir::ops::top_k_gate`] computes (the derivation
//! `crate::mixtral` gives). So the router is [`linear`] with a bias feeding the shared `top_k_gate`; unlike
//! OlmoE (`norm_topk_prob: false`) no new composition is needed.
//!
//! **Expert MLP** is a clamped swish-GLU, not SwiGLU (`GptOssExperts._apply_gate`, `self.alpha = 1.702`,
//! `self.limit = config.swiglu_limit = 7.0`):
//! ```text
//! gate, up = gate_up[..., ::2], gate_up[..., 1::2]     # interleaved (even/odd), not first-half/second-half
//! gate = gate.clamp(max=limit)                          # one-sided clamp
//! up = up.clamp(min=-limit, max=limit)                  # two-sided clamp
//! glu = gate * sigmoid(gate * alpha)
//! gated_output = (up + 1) * glu                          # "+1", not plain up * glu
//! ```
//! Both expert projections have biases (`gate_up_proj_bias [E,2I]`, `down_proj_bias [E,H]`). [`gptoss_ffn`]
//! composes this from the primitives `moe_dense` uses (`matmul`/`broadcast`/`binary`/`reduce`/[`sigmoid`])
//! plus `Neg`+`Max` for clamp (`min(a,b) = -max(-a,-b)`); no new graph-IR primitive is added.
//!
//! **Config** (`openai/gpt-oss-20b`): `hidden_size: 2880`, `intermediate_size: 2880` (per-expert width; every
//! layer routes), `num_hidden_layers: 24`, `num_attention_heads: 64`, `num_key_value_heads: 8`,
//! `head_dim: 64` (explicit, not `hidden_size / num_attention_heads = 45`), `num_local_experts: 32`,
//! `num_experts_per_tok: 4` (also given redundantly as `experts_per_token`), `attention_bias: true` (q/k/v/o),
//! `rms_norm_eps: 1e-5`, `swiglu_limit: 7.0`, `sliding_window: 128`, `tie_word_embeddings: false` (the tiny
//! fixture sets `true`). `router_aux_loss_coef` is training-only and not implemented.
//!
//! **Loader layout.** Experts ship already fused per layer into one 3D tensor in poot's `[in,out]` convention:
//! `model.layers.{li}.mlp.experts.gate_up_proj` is `[E, hidden, 2*inter]` and `...down_proj` is
//! `[E, inter, hidden]`. Neither name ends in `.weight`, so `build_weights`' `proj.weight` transpose rule leaves
//! them alone. Only `mlp.router.weight` needs the per-layer transpose every arch's router gets.
//!
//! **Limit.** The real 20b/120b checkpoints ship expert weights MXFP4-quantized
//! (`quantization_config: {quant_method: "mxfp4", modules_to_not_convert: [attn, router, embed, lm_head]}`).
//! This module is verified against the un-quantized `tiny-random/gpt-oss`; a real checkpoint needs an MXFP4
//! dequantizer (see the spec's "Out of scope").
//!
//! Rides on [`crate::qwen2::Qwen2Config`] for the shared attention/RoPE/norm dims (like `crate::mixtral`/
//! `crate::olmoe`) plus [`GptOssParams`] for the MoE shape, sliding-window size, and `layer_types` schedule.

use poot_graph_ir::ops::{
    attention_masked_with_sink, attention_prefill_with_sink, linear, rmsnorm, rope, rope_prefill,
    sigmoid, top_k_gate,
};
use poot_graph_ir::{
    BinOp, Builder, Graph, RedOp, Scalar, Slot, StateRole, TensorType, Traced, UnOp,
};
use poot_tensor::DType;

use crate::gemma4::gemma4_local_window_floor;
use crate::qwen2::Qwen2Config;

/// `GptOssExperts._apply_gate`'s hardcoded sigmoid-gate rescale (`self.alpha = 1.702`); not a config field.
pub const GPTOSS_SWIGLU_ALPHA: f32 = 1.702;

/// gpt-oss's MoE shape and sliding-window schedule, alongside a [`Qwen2Config`] (shared attention/RoPE/norm
/// dims). Unlike Mixtral/OlmoE there is no `norm_topk_prob` flag (the router always matches [`top_k_gate`]),
/// but there is a per-layer sliding-window schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct GptOssParams {
    /// Total experts per layer (`num_local_experts`; `32` on the real models and the tiny fixture).
    pub n_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`; `4`).
    pub top_k: usize,
    /// Per-expert FFN intermediate size (`intermediate_size`; `2880` real / `64` tiny fixture). There is no
    /// separate dense-MLP width.
    pub inter: usize,
    /// Clamp bound of the clamped-GLU activation (`swiglu_limit`; `7.0`).
    pub swiglu_limit: f32,
    /// Window size for layers where `layer_is_sliding[li]` is true (`sliding_window`; `128`).
    pub sliding_window: usize,
    /// Per-layer window schedule, length `cfg.layers`, from the config `layer_types` array
    /// (`"sliding_attention"` -> `true`, `"full_attention"` -> `false`). Not assumed to alternate.
    pub layer_is_sliding: Vec<bool>,
}

/// Trace a full-sequence gpt-oss prefill forward (the CPU-oracle path): embeddings, `cfg.layers` pre-norm blocks
/// (biased Q/K/V, RoPE, causal-or-windowed GQA attention with per-head sink logits, biased `o_proj`, clamped-GLU
/// MoE MLP, residuals), final `model.norm`, `lm_head.weight`.
///
/// Named constants bound at eval time: `model.embed_tokens.weight` `[vocab,hidden]`,
/// `rope.cos`/`rope.sin` `[max_pos,head_dim]`, per-layer `model.layers.{li}.
/// {input_layernorm,post_attention_layernorm}.weight`, `self_attn.{q,k,v,o}_proj.{weight,bias}`,
/// `self_attn.sinks` `[n_heads]`, `mlp.router.{weight,bias}` `[hidden,n_experts]`/`[n_experts]`,
/// `mlp.experts.gate_up_proj` `[n_experts,hidden,2*inter]` (already `[in,out]`, not `.weight`-suffixed),
/// `mlp.experts.gate_up_proj_bias` `[n_experts,2*inter]`,
/// `mlp.experts.down_proj` `[n_experts,inter,hidden]`, `mlp.experts.down_proj_bias` `[n_experts,hidden]`,
/// `model.norm.weight`, and `lm_head.weight`; plus the `mask.prefill`/`mask.prefill.local` step inputs
/// (card 550a) `[1,1,l,l]` (full / windowed by `mp.sliding_window`, selected per layer by
/// `mp.layer_is_sliding`).
pub fn trace_gptoss_prefill(cfg: Qwen2Config, mp: GptOssParams, seq_len: usize) -> Graph {
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
    let mask_full = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));
    let mask_local = b.slot_named(
        Slot::Mask,
        "prefill.local",
        TensorType::f32(vec![1, 1, l, l]),
    );

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let is_local = mp.layer_is_sliding[li];

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let bq = b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]));
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let bk = b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]));
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let bv = b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]));
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );
        let bo = b.constant(&p("self_attn.o_proj.bias"), TensorType::f32(vec![h]));

        let q = linear(&b, normed, wq, Some(bq));
        let k = linear(&b, normed, wk, Some(bk));
        let v = linear(&b, normed, wv, Some(bv));

        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

        let sinks = b.constant(&p("self_attn.sinks"), TensorType::f32(vec![hq]));
        let mask = if is_local { mask_local } else { mask_full };
        let attn = attention_prefill_with_sink(&b, q, k, v, n_rep, scale, mask, sinks, hq, l);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, Some(bo));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = gptoss_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// gpt-oss single-token fixed-KV masked decode, the decode analog of [`trace_gptoss_prefill`] on
/// `crate::qwen2::trace_decode_kv_masked`'s skeleton. Reuses [`gemma4_local_window_floor`] for the window
/// schedule: an in-graph floor added onto the single shared `Slot::Mask`.
pub fn trace_gptoss_decode_kv_masked(cfg: Qwen2Config, mp: GptOssParams, cap: usize) -> Graph {
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

    // Window schedule via the same in-graph floor gemma4's decode tracer uses.
    let iota_full = b.iota(cfg.max_pos);
    let iota_cap = b.slice(iota_full, 0, 0, cap);
    let pos_f = b.gather_scalar(iota_full, 0, pos_slot); // iota[pos] == pos, avoids an i32->f32 Cast
    let local_mask =
        gemma4_local_window_floor(&b, mask, pos_f, iota_cap, mp.sliding_window, 1, cap);

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
        let is_local = mp.layer_is_sliding[li];

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let wq = b.constant(
            &p("self_attn.q_proj.weight"),
            TensorType::f32(vec![h, q_dim]),
        );
        let bq = b.constant(&p("self_attn.q_proj.bias"), TensorType::f32(vec![q_dim]));
        let wk = b.constant(
            &p("self_attn.k_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let bk = b.constant(&p("self_attn.k_proj.bias"), TensorType::f32(vec![kv_dim]));
        let wv = b.constant(
            &p("self_attn.v_proj.weight"),
            TensorType::f32(vec![h, kv_dim]),
        );
        let bv = b.constant(&p("self_attn.v_proj.bias"), TensorType::f32(vec![kv_dim]));
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![q_dim, h]),
        );
        let bo = b.constant(&p("self_attn.o_proj.bias"), TensorType::f32(vec![h]));

        let q = linear(&b, normed, wq, Some(bq));
        let k = linear(&b, normed, wk, Some(bk));
        let v = linear(&b, normed, wv, Some(bv));

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

        let sinks = b.constant(&p("self_attn.sinks"), TensorType::f32(vec![hq]));
        let m = if is_local { local_mask } else { mask };
        let attn =
            attention_masked_with_sink(&b, q, kcache_out, vcache_out, n_rep, scale, m, sinks, hq);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, q_dim]);
        let attn = linear(&b, attn, wo, Some(bo));
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let mo = gptoss_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, mo);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// One layer's MoE MLP. Every gpt-oss layer routes (no dense-layer branch). The router is [`top_k_gate`] on
/// biased logits; the expert FFN is the clamped-GLU-with-bias composition. Dense form (every expert evaluated,
/// non-selected zero-weighted) for prefill and decode.
fn gptoss_ffn(b: &Builder, normed: Traced, h: usize, mp: &GptOssParams, li: usize) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let e = mp.n_experts;
    let k = mp.top_k;
    let inter = mp.inter;
    let limit = mp.swiglu_limit;

    let router_w = b.constant(&p("mlp.router.weight"), TensorType::f32(vec![h, e]));
    let router_b = b.constant(&p("mlp.router.bias"), TensorType::f32(vec![e]));
    // Not `.weight`-suffixed and already [in,out]: no transpose in the loader.
    let w_in = b.constant(
        &p("mlp.experts.gate_up_proj"),
        TensorType::f32(vec![e, h, 2 * inter]),
    );
    let w_in_b = b.constant(
        &p("mlp.experts.gate_up_proj_bias"),
        TensorType::f32(vec![e, 2 * inter]),
    );
    let w_out = b.constant(
        &p("mlp.experts.down_proj"),
        TensorType::f32(vec![e, inter, h]),
    );
    let w_out_b = b.constant(
        &p("mlp.experts.down_proj_bias"),
        TensorType::f32(vec![e, h]),
    );

    let shape = b.aval(normed).shape;
    let l = shape[shape.len() - 2];
    let xm = b.reshape(normed, vec![l, h]);

    // route: topk-then-softmax over the selected logits equals `top_k_gate`'s selected-only normalization;
    // the logits are biased (`Some(router_b)`).
    let logits = linear(b, xm, router_w, Some(router_b));
    let gate = top_k_gate(b, logits, k); // [L, E]

    // Every expert's FFN, batched over the expert axis (same shape-static dense form as moe_dense).
    let x_e = b.broadcast(b.reshape(xm, vec![1, l, h]), vec![e, l, h]); // [E,L,H]
    let gu = b.matmul(x_e, w_in); // [E,L,2I]
    let gu_b = b.broadcast(
        b.reshape(w_in_b, vec![e, 1, 2 * inter]),
        vec![e, l, 2 * inter],
    );
    let gu = b.binary(BinOp::Add, gu, gu_b);

    // Deinterleave: `gate = gu[...,::2]`, `up = gu[...,1::2]`; reshape 2I -> [I,2] and slice index 0 vs 1.
    let gu4 = b.reshape(gu, vec![e, l, inter, 2]);
    let gate_h = b.reshape(b.slice(gu4, 3, 0, 1), vec![e, l, inter]);
    let up_h = b.reshape(b.slice(gu4, 3, 1, 2), vec![e, l, inter]);

    // clamp(gate, max=limit) = -max(-gate, -limit); clamp(up, -limit, limit) via two Max steps (no Min/Clamp
    // IR op; min(a,b) = -max(-a,-b)).
    let neg_gate = b.unary(UnOp::Neg, gate_h);
    let neg_gate_c = b.binary_scalar(BinOp::Max, neg_gate, Scalar::F32(-limit));
    let gate_c = b.unary(UnOp::Neg, neg_gate_c);
    let up_lo = b.binary_scalar(BinOp::Max, up_h, Scalar::F32(-limit));
    let neg_up_lo = b.unary(UnOp::Neg, up_lo);
    let neg_up_c = b.binary_scalar(BinOp::Max, neg_up_lo, Scalar::F32(-limit));
    let up_c = b.unary(UnOp::Neg, neg_up_c);

    // glu = gate * sigmoid(gate * alpha) with alpha = GPTOSS_SWIGLU_ALPHA; gated = (up + 1) * glu.
    let gate_alpha = b.binary_scalar(BinOp::Mul, gate_c, Scalar::F32(GPTOSS_SWIGLU_ALPHA));
    let sig = sigmoid(b, gate_alpha);
    let glu = b.binary(BinOp::Mul, gate_c, sig);
    let up_plus1 = b.binary_scalar(BinOp::Add, up_c, Scalar::F32(1.0));
    let gated = b.binary(BinOp::Mul, up_plus1, glu); // [E,L,I]

    let out = b.matmul(gated, w_out); // [E,L,H]
    let out_b = b.broadcast(b.reshape(w_out_b, vec![e, 1, h]), vec![e, l, h]);
    let out = b.binary(BinOp::Add, out, out_b);

    // Weight each expert's output by its gate value and sum over the expert axis (last-axis reduce only).
    let gate_w = b.reshape(b.transpose(gate, vec![1, 0]), vec![e, l, 1]); // [E,L,1]
    let weighted = b.binary(BinOp::Mul, out, gate_w); // [E,L,H]
    let weighted_t = b.transpose(weighted, vec![1, 2, 0]); // [L,H,E]
    let y = b.reduce(RedOp::Sum, weighted_t, 2, false); // [L,H]
    b.reshape(y, vec![1, l, h])
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    fn tiny_cfg() -> Qwen2Config {
        // Small but non-degenerate dims: GQA, and head_dim decoupled from hidden/n_heads (4 * 6 = 24 != hidden 16,
        // as in the real config). The bias is unconditional in this module, not gated by cfg.qkv_bias.
        Qwen2Config {
            vocab: 24,
            hidden: 16,
            inter: 16, // unused directly by gpt-oss (MoE inter comes from GptOssParams), kept non-zero
            layers: 3,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
            rotary_dim: 6,
            eps: 1e-5,
            max_pos: 32,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        }
    }

    /// `sliding_window` is smaller than the 5-7 token test sequences so the CPU-oracle tests exercise windowing.
    fn tiny_mp(layer_is_sliding: Vec<bool>) -> GptOssParams {
        GptOssParams {
            n_experts: 6,
            top_k: 2,
            inter: 12,
            swiglu_limit: 2.5, // deliberately small (vs the real 7.0) so the CPU-oracle test clamp fires
            sliding_window: 2,
            layer_is_sliding,
        }
    }

    #[test]
    fn gptoss_prefill_validates_bias_everywhere_sinks_and_alternating_window() {
        let cfg = tiny_cfg();
        let mp = tiny_mp(vec![true, false, true]);
        let g = trace_gptoss_prefill(cfg, mp, 5);
        g.validate().expect("gpt-oss prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // FR: bias constants ARE present on every q/k/v/o projection (unlike Mixtral/OlmoE).
        for suf in ["q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias"] {
            let count = g
                .inputs
                .iter()
                .filter(|id| {
                    matches!(g.values[**id].storage, Storage::Const)
                        && g.values[**id]
                            .name
                            .as_deref()
                            .is_some_and(|n| n.ends_with(suf))
                })
                .count();
            assert_eq!(count, cfg.layers, "{suf} must exist on every layer");
        }
        // sinks: one [n_heads] constant per layer.
        let sinks_count = g
            .inputs
            .iter()
            .filter(|id| {
                matches!(g.values[**id].storage, Storage::Const)
                    && g.values[**id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with("self_attn.sinks"))
            })
            .count();
        assert_eq!(sinks_count, cfg.layers);
        let sink_id = g
            .inputs
            .iter()
            .find(|id| {
                g.values[**id]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.ends_with("self_attn.sinks"))
            })
            .unwrap();
        assert_eq!(g.values[*sink_id].aval.shape, vec![cfg.n_heads]);
        // both mask step inputs present (alternating schedule needs both).
        assert!(g.inputs.iter().any(|id| {
            g.values[*id]
                .name
                .as_deref()
                .is_some_and(|n| n == "mask.prefill")
        }));
        assert!(g.inputs.iter().any(|id| {
            g.values[*id]
                .name
                .as_deref()
                .is_some_and(|n| n == "mask.prefill.local")
        }));
        // every layer declares a router constant (MoE unconditional, no dense-layer branch).
        let router_count = g
            .inputs
            .iter()
            .filter(|id| {
                matches!(g.values[**id].storage, Storage::Const)
                    && g.values[**id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with("mlp.router.weight"))
            })
            .count();
        assert_eq!(router_count, cfg.layers, "one router per layer, no skips");
    }

    #[test]
    fn gptoss_decode_kv_masked_validates_state_and_mask_shapes() {
        let cfg = tiny_cfg();
        let mp = tiny_mp(vec![true, false, true]);
        let cap = 16;
        let g = trace_gptoss_decode_kv_masked(cfg, mp, cap);
        g.validate().expect("gpt-oss decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(g.state.len(), 2 * cfg.layers);
        for (si, _so) in &g.state {
            assert_eq!(
                g.aval(*si).shape,
                vec![1, cfg.n_kv_heads, cap, cfg.head_dim]
            );
            assert_eq!(g.values[*si].storage, Storage::State);
        }
        let mask_id = g
            .inputs
            .iter()
            .find(|id| matches!(g.values[**id].storage, Storage::Slot(Slot::Mask)))
            .expect("Slot::Mask present");
        assert_eq!(g.values[*mask_id].aval.shape, vec![cap]);
        // the in-graph windowing floor's range dependency is an `Iota` equation (card 550a), not a
        // bound constant.
        assert!(
            g.eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Iota { .. })),
            "decode graph must compute its causal range in-graph"
        );
        assert!(
            g.inputs
                .iter()
                .all(|id| g.values[*id].name.as_deref() != Some("causal.iota")),
            "causal.iota must be an in-graph iota, not a bound const"
        );
    }

    // ---- CPU-oracle numerics: checked against an independent from-scratch Rust reference forward pass. ----

    mod cpu_oracle {
        use super::*;
        use std::collections::HashMap;

        use poot_test_util::fill;

        use poot_test_util::seed_of;

        fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
            let raw = fill(n, seed_of(name));
            if ln_gamma {
                raw.iter().map(|v| 1.0 + v * 0.05).collect()
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            }
        }

        /// Sink logits get a real-magnitude fill; near-zero sinks would be inert.
        fn sink_weight(name: &str, n: usize) -> Vec<f32> {
            fill(n, seed_of(name))
        }

        fn rope_tables(max_pos: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
            let mut cos = vec![0.0f32; max_pos * d];
            let mut sin = vec![0.0f32; max_pos * d];
            for pos in 0..max_pos {
                for i in 0..d / 2 {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / d as f32);
                    let ang = pos as f32 * freq;
                    let (s, c) = ang.sin_cos();
                    cos[pos * d + i] = c;
                    cos[pos * d + i + d / 2] = c;
                    sin[pos * d + i] = s;
                    sin[pos * d + i + d / 2] = s;
                }
            }
            (cos, sin)
        }

        /// All named constants [`trace_gptoss_prefill`] declares, keyed by exact graph const name.
        fn all_weights(cfg: &Qwen2Config, mp: &GptOssParams) -> HashMap<String, Vec<f32>> {
            let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
            let (q_dim, kv_dim) = (hq * d, hkv * d);
            let mut w = HashMap::new();
            w.insert(
                "model.embed_tokens.weight".to_string(),
                weight("embed", cfg.vocab * h, false),
            );
            let (cos, sin) = rope_tables(cfg.max_pos, d, 150_000.0); // real rope_theta (plain, no YaRN)
            w.insert("rope.cos".to_string(), cos);
            w.insert("rope.sin".to_string(), sin);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1"), h, true));
                w.insert(
                    p("self_attn.q_proj.weight"),
                    weight(&p("qw"), h * q_dim, false),
                );
                w.insert(p("self_attn.q_proj.bias"), weight(&p("qb"), q_dim, false));
                w.insert(
                    p("self_attn.k_proj.weight"),
                    weight(&p("kw"), h * kv_dim, false),
                );
                w.insert(p("self_attn.k_proj.bias"), weight(&p("kb"), kv_dim, false));
                w.insert(
                    p("self_attn.v_proj.weight"),
                    weight(&p("vw"), h * kv_dim, false),
                );
                w.insert(p("self_attn.v_proj.bias"), weight(&p("vb"), kv_dim, false));
                w.insert(
                    p("self_attn.o_proj.weight"),
                    weight(&p("ow"), q_dim * h, false),
                );
                w.insert(p("self_attn.o_proj.bias"), weight(&p("ob"), h, false));
                w.insert(p("self_attn.sinks"), sink_weight(&p("sinks"), hq));
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2"), h, true),
                );
                w.insert(
                    p("mlp.router.weight"),
                    weight(&p("router"), h * mp.n_experts, false),
                );
                w.insert(
                    p("mlp.router.bias"),
                    weight(&p("routerb"), mp.n_experts, false),
                );
                // gate_up_proj: [E, H, 2I], already [in,out]; gate/up interleaved (even=gate, odd=up).
                let mut gate_up = vec![0.0f32; mp.n_experts * h * 2 * mp.inter];
                let mut down = vec![0.0f32; mp.n_experts * mp.inter * h];
                for e in 0..mp.n_experts {
                    let g = weight(&p(&format!("e{e}.gate")), h * mp.inter, false);
                    let u = weight(&p(&format!("e{e}.up")), h * mp.inter, false);
                    let d_ = weight(&p(&format!("e{e}.down")), mp.inter * h, false);
                    for row in 0..h {
                        let dst = (e * h + row) * 2 * mp.inter;
                        for i in 0..mp.inter {
                            gate_up[dst + 2 * i] = g[row * mp.inter + i];
                            gate_up[dst + 2 * i + 1] = u[row * mp.inter + i];
                        }
                    }
                    let dst = e * mp.inter * h;
                    down[dst..dst + mp.inter * h].copy_from_slice(&d_);
                }
                w.insert(p("mlp.experts.gate_up_proj"), gate_up);
                w.insert(
                    p("mlp.experts.gate_up_proj_bias"),
                    weight(&p("gub"), mp.n_experts * 2 * mp.inter, false),
                );
                w.insert(p("mlp.experts.down_proj"), down);
                w.insert(
                    p("mlp.experts.down_proj_bias"),
                    weight(&p("downb"), mp.n_experts * h, false),
                );
            }
            w.insert("model.norm.weight".to_string(), weight("ln_f", h, true));
            w.insert(
                "lm_head.weight".to_string(),
                weight("lm_head", h * cfg.vocab, false),
            );
            w
        }

        use poot_test_util::rmsnorm_ref;

        fn sigmoid_ref(v: f32) -> f32 {
            1.0 / (1.0 + (-v).exp())
        }

        fn clamp_ref(v: f32, lo: Option<f32>, hi: Option<f32>) -> f32 {
            let mut x = v;
            if let Some(lo) = lo {
                x = x.max(lo);
            }
            if let Some(hi) = hi {
                x = x.min(hi);
            }
            x
        }

        /// `y[out] = x @ w[in,out] (+ bias)`, flat row-major.
        fn linear_ref(
            x: &[f32],
            w: &[f32],
            in_dim: usize,
            out_dim: usize,
            bias: Option<&[f32]>,
        ) -> Vec<f32> {
            let mut y = vec![0.0f32; out_dim];
            for o in 0..out_dim {
                let mut acc = bias.map_or(0.0, |b| b[o]);
                for i in 0..in_dim {
                    acc += x[i] * w[i * out_dim + o];
                }
                y[o] = acc;
            }
            y
        }

        use poot_test_util::rope_ref;

        /// Independent top-`top_k` router (topk on raw logits, then softmax over the selected). Weights sum to 1
        /// over the selected experts; gpt-oss has no non-renormalized mode.
        fn top_k_router_ref(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let sel = &idx[..top_k];
            let m = sel.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
            let exp: Vec<f32> = sel.iter().map(|&i| (logits[i] - m).exp()).collect();
            let denom: f32 = exp.iter().sum();
            sel.iter()
                .zip(exp.iter())
                .map(|(&i, &e)| (i, e / denom))
                .collect()
        }

        /// Independent reference forward pass for [`trace_gptoss_prefill`] (last token only), written as direct
        /// loops rather than a copy of `gptoss_ffn`'s decomposition.
        fn gptoss_prefill_ref(
            cfg: &Qwen2Config,
            mp: &GptOssParams,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let (h, d, hq, hkv, l) = (
                cfg.hidden,
                cfg.head_dim,
                cfg.n_heads,
                cfg.n_kv_heads,
                tokens.len(),
            );
            let n_rep = hq / hkv;
            let scale = 1.0 / (d as f32).sqrt();
            let embed = &w["model.embed_tokens.weight"];
            let cos = &w["rope.cos"];
            let sin = &w["rope.sin"];

            let mut x: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| embed[t * h..(t + 1) * h].to_vec())
                .collect();

            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                let is_local = mp.layer_is_sliding[li];
                let ln1w = &w[&p("input_layernorm.weight")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
                    .collect();

                let qw = &w[&p("self_attn.q_proj.weight")];
                let qb = &w[&p("self_attn.q_proj.bias")];
                let kw = &w[&p("self_attn.k_proj.weight")];
                let kb = &w[&p("self_attn.k_proj.bias")];
                let vw = &w[&p("self_attn.v_proj.weight")];
                let vb = &w[&p("self_attn.v_proj.bias")];
                let mut q: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, qw, h, hq * d, Some(qb)))
                    .collect();
                let mut k: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, kw, h, hkv * d, Some(kb)))
                    .collect();
                let v: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, vw, h, hkv * d, Some(vb)))
                    .collect();

                for (pos, qrow) in q.iter_mut().enumerate() {
                    for hh in 0..hq {
                        let rotated = rope_ref(&qrow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                        qrow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
                    }
                }
                for (pos, krow) in k.iter_mut().enumerate() {
                    for hh in 0..hkv {
                        let rotated = rope_ref(&krow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                        krow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
                    }
                }

                let sinks = &w[&p("self_attn.sinks")];
                let mut attn_out = vec![vec![0.0f32; hq * d]; l];
                for qh in 0..hq {
                    let kh = qh / n_rep;
                    for i in 0..l {
                        let start = if is_local {
                            i.saturating_sub(mp.sliding_window - 1)
                        } else {
                            0
                        };
                        let mut scores = vec![0.0f32; i - start + 1];
                        for (jj, sc) in scores.iter_mut().enumerate() {
                            let j = start + jj;
                            let mut s = 0.0f32;
                            for dd in 0..d {
                                s += q[i][qh * d + dd] * k[j][kh * d + dd];
                            }
                            *sc = s * scale;
                        }
                        let sink = sinks[qh];
                        let m = scores.iter().cloned().fold(sink, f32::max);
                        let mut denom = (sink - m).exp();
                        let mut e = vec![0.0f32; scores.len()];
                        for (jj, sc) in scores.iter().enumerate() {
                            e[jj] = (sc - m).exp();
                            denom += e[jj];
                        }
                        for dd in 0..d {
                            let mut acc = 0.0f32;
                            for (jj, ej) in e.iter().enumerate() {
                                let j = start + jj;
                                acc += (ej / denom) * v[j][kh * d + dd];
                            }
                            attn_out[i][qh * d + dd] = acc;
                        }
                    }
                }

                let ow = &w[&p("self_attn.o_proj.weight")];
                let ob = &w[&p("self_attn.o_proj.bias")];
                for i in 0..l {
                    let proj = linear_ref(&attn_out[i], ow, hq * d, h, Some(ob));
                    for c in 0..h {
                        x[i][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                let router_w = &w[&p("mlp.router.weight")];
                let router_b = &w[&p("mlp.router.bias")];
                let gate_up = &w[&p("mlp.experts.gate_up_proj")];
                let gu_bias = &w[&p("mlp.experts.gate_up_proj_bias")];
                let down = &w[&p("mlp.experts.down_proj")];
                let down_bias = &w[&p("mlp.experts.down_proj_bias")];
                for row in x.iter_mut().take(l) {
                    let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let logits = linear_ref(&normed, router_w, h, mp.n_experts, Some(router_b));
                    let selected = top_k_router_ref(&logits, mp.top_k);
                    let mut moe_out = vec![0.0f32; h];
                    for (e, gate_weight) in selected {
                        let e_gate_up = &gate_up[e * h * 2 * mp.inter..(e + 1) * h * 2 * mp.inter];
                        let e_gu_bias = &gu_bias[e * 2 * mp.inter..(e + 1) * 2 * mp.inter];
                        let e_down = &down[e * mp.inter * h..(e + 1) * mp.inter * h];
                        let e_down_bias = &down_bias[e * h..(e + 1) * h];

                        // interleaved gate/up projection weights, de-interleaved into two [H,I] matrices.
                        let mut gate_w = vec![0.0f32; h * mp.inter];
                        let mut up_w = vec![0.0f32; h * mp.inter];
                        for row_i in 0..h {
                            for i in 0..mp.inter {
                                gate_w[row_i * mp.inter + i] =
                                    e_gate_up[row_i * 2 * mp.inter + 2 * i];
                                up_w[row_i * mp.inter + i] =
                                    e_gate_up[row_i * 2 * mp.inter + 2 * i + 1];
                            }
                        }
                        let mut gate_b = vec![0.0f32; mp.inter];
                        let mut up_b = vec![0.0f32; mp.inter];
                        for i in 0..mp.inter {
                            gate_b[i] = e_gu_bias[2 * i];
                            up_b[i] = e_gu_bias[2 * i + 1];
                        }

                        let mut gate = linear_ref(&normed, &gate_w, h, mp.inter, Some(&gate_b));
                        let mut up = linear_ref(&normed, &up_w, h, mp.inter, Some(&up_b));
                        for gv in gate.iter_mut() {
                            *gv = clamp_ref(*gv, None, Some(mp.swiglu_limit));
                        }
                        for uv in up.iter_mut() {
                            *uv = clamp_ref(*uv, Some(-mp.swiglu_limit), Some(mp.swiglu_limit));
                        }
                        let act: Vec<f32> = gate
                            .iter()
                            .zip(up.iter())
                            .map(|(&g_, &u)| {
                                let glu = g_ * sigmoid_ref(g_ * GPTOSS_SWIGLU_ALPHA);
                                (u + 1.0) * glu
                            })
                            .collect();
                        let expert_out = linear_ref(&act, e_down, mp.inter, h, Some(e_down_bias));
                        for c in 0..h {
                            moe_out[c] += gate_weight * expert_out[c];
                        }
                    }
                    for (c, mv) in moe_out.iter().enumerate() {
                        row[c] += mv;
                    }
                }
            }

            let ln_f_w = &w["model.norm.weight"];
            let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
            let lm_head = &w["lm_head.weight"];
            linear_ref(&last, lm_head, h, cfg.vocab, None)
        }

        fn eval_gptoss_prefill(
            g: &Graph,
            tokens: &[usize],
            weights: &HashMap<String, Vec<f32>>,
            sliding_window: usize,
        ) -> poot_tensor::HostTensor {
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let t = match &meta.storage {
                    Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                        vec![tokens.len()],
                        tokens.iter().map(|&t| t as i32).collect(),
                    ),
                    Storage::Slot(Slot::Mask) => {
                        let name = meta.name.as_deref().expect("mask slot without a name");
                        let l = meta.aval.shape[2];
                        let mut m = vec![0.0f32; l * l];
                        if name == "mask.prefill" {
                            for i in 0..l {
                                for j in 0..l {
                                    m[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                                }
                            }
                        } else {
                            assert_eq!(name, "mask.prefill.local", "unexpected mask slot {name}");
                            for i in 0..l {
                                let lo = i.saturating_sub(sliding_window - 1);
                                for j in 0..l {
                                    m[i * l + j] = if j <= i && j >= lo { 0.0 } else { -1.0e30 };
                                }
                            }
                        }
                        poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                    }
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        let data = weights
                            .get(name)
                            .unwrap_or_else(|| panic!("no weight bound for {name}"));
                        poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data.clone())
                    }
                    other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
                };
                inputs.insert(id, t.into());
            }
            poot_eval::eval(
                g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu eval")
            .output
            .into_host()
            .expect("dense output")
        }

        fn assert_matches(got: &poot_tensor::HostTensor, want: &[f32], vocab: usize) {
            assert_eq!(got.shape(), vec![1, 1, vocab]);
            poot_test_util::assert_close_rel(got.as_f32().unwrap(), want, 1e-4);
            assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
        }

        /// `trace_gptoss_prefill` matches an independent hand-loop reference within f32 tolerance;
        /// `sliding_window=2` on a 6-token sequence so the window schedule is exercised.
        #[test]
        fn gptoss_prefill_matches_hand_rolled_reference() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(vec![true, false, true]);
            let tokens = [3usize, 7, 1, 9, 15, 4];
            let weights = all_weights(&cfg, &mp);

            let g = trace_gptoss_prefill(cfg, mp.clone(), tokens.len());
            let got = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);
            let want = gptoss_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// All-full-attention control (every layer `is_sliding=false`) with the same reference/graph pair.
        #[test]
        fn gptoss_prefill_matches_hand_rolled_reference_all_full_attention() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(vec![false, false, false]);
            let tokens = [2usize, 5, 8, 11];
            let weights = all_weights(&cfg, &mp);

            let g = trace_gptoss_prefill(cfg, mp.clone(), tokens.len());
            let got = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);
            let want = gptoss_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }

        /// Perturbing a layer's sink logit changes the output (a dropped sink term would still give finite output).
        #[test]
        fn gptoss_prefill_output_is_sensitive_to_a_perturbed_sink() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(vec![true, false, true]);
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg, &mp);
            let g = trace_gptoss_prefill(cfg, mp.clone(), tokens.len());
            let base = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);

            let key = "model.layers.1.self_attn.sinks".to_string();
            weights.get_mut(&key).unwrap()[0] += 5.0;
            let perturbed = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-6,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// Perturbing the first weight of every expert's layer-1 down-projection changes the output: the
        /// routing/expert chain is live. Every expert is perturbed because which experts three tokens route to
        /// depends on the fixture's seeds; a routed one always moves the output.
        #[test]
        fn gptoss_prefill_output_is_sensitive_to_a_perturbed_expert_weight() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(vec![true, false, true]);
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg, &mp);
            let g = trace_gptoss_prefill(cfg, mp.clone(), tokens.len());
            let base = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);

            let key = "model.layers.1.mlp.experts.down_proj".to_string();
            let down = weights.get_mut(&key).unwrap();
            let per_expert = down.len() / mp.n_experts;
            for expert in 0..mp.n_experts {
                down[expert * per_expert] += 5.0;
            }
            let perturbed = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-4,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }

        // ---- real-checkpoint weight-layout crosswalk: tiny-random/gpt-oss. ----
        //
        // A randomly initialized BF16 `GptOssForCausalLM` export (2 layers, hidden 32, 2 heads/1 kv head,
        // head_dim 32, 32 experts top-4, intermediate 64, swiglu_limit 7.0, sliding_window 128, layer_types
        // [sliding, full], tied embeddings). It proves the loader crosswalk (router transpose, tied lm_head,
        // direct-bind fused `[in,out]` expert tensors, biases) against real HF-shaped tensors. Loads via
        // `poot_load` (poot-llm sits above this crate) and compares the traced graph's CPU eval to
        // `gptoss_prefill_ref` fed the loaded weights.
        //
        // The rope table here is plain-theta (non-YaRN), read by both graph and reference; YaRN is exercised
        // by the full `Runner::load` test in poot-llm. `sliding_window: 128` exceeds every test sequence, so
        // windowing is inert here; the synthetic tests above (`sliding_window: 2`) exercise it.
        #[test]
        fn gptoss_tiny_random_checkpoint_matches_hand_rolled_reference() {
            // Populate with `hf download tiny-random/gpt-oss --local-dir $POOT_MODELS_DIR/gptoss-tiny`.
            let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny"))
            else {
                return;
            };

            let hf =
                poot_load::Qwen2HfConfig::load(dir.join("config.json")).expect("load config.json");
            assert!(hf.is_gpt_oss());
            let st = poot_load::safetensors::load_weight_store(&dir).expect("load safetensors");

            let cfg = Qwen2Config {
                vocab: hf.vocab_size,
                hidden: hf.hidden_size,
                inter: hf.intermediate_size,
                layers: hf.num_hidden_layers,
                n_heads: hf.num_attention_heads,
                n_kv_heads: hf.num_key_value_heads,
                head_dim: hf.head_dim(),
                rotary_dim: hf.rotary_dim(),
                eps: hf.rms_norm_eps,
                max_pos: 8, // only positions 0..4 are ever read; keep the rope table cheap.
                qkv_bias: hf.qkv_bias(),
                qk_norm: hf.qk_norm(),
                ..Default::default()
            };
            let mp = GptOssParams {
                n_experts: hf
                    .num_local_experts
                    .expect("gpt-oss carries num_local_experts"),
                top_k: hf
                    .num_experts_per_tok
                    .expect("gpt-oss carries num_experts_per_tok"),
                inter: hf.intermediate_size,
                swiglu_limit: hf.swiglu_limit.expect("gpt-oss carries swiglu_limit"),
                sliding_window: hf.sliding_window.expect("gpt-oss carries sliding_window"),
                layer_is_sliding: hf
                    .layer_types
                    .clone()
                    .expect("gpt-oss carries layer_types")
                    .iter()
                    .map(|s| s == "sliding_attention")
                    .collect(),
            };
            assert_eq!(mp.n_experts, 32);
            assert_eq!(mp.top_k, 4);
            assert_eq!(mp.layer_is_sliding, vec![true, false]);

            fn transpose2d(data: &[f32], r: usize, c: usize) -> Vec<f32> {
                let mut out = vec![0.0f32; r * c];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = data[i * c + j];
                    }
                }
                out
            }

            let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
            let get2d = |name: &str| -> Vec<f32> {
                let rt = poot_eval::materialize_dense(&st, name)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                transpose2d(rt.as_f32().unwrap(), rt.shape()[0], rt.shape()[1])
            };
            let get_raw = |name: &str| -> Vec<f32> {
                poot_eval::materialize_dense(&st, name)
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                    .as_f32()
                    .unwrap()
                    .to_vec()
            };

            weights.insert(
                "model.embed_tokens.weight".to_string(),
                get_raw("model.embed_tokens.weight"),
            );
            // tie_word_embeddings: true on this fixture: lm_head reuses the transposed embedding.
            weights.insert(
                "lm_head.weight".to_string(),
                transpose2d(&get_raw("model.embed_tokens.weight"), cfg.vocab, cfg.hidden),
            );
            weights.insert(
                "model.norm.weight".to_string(),
                get_raw("model.norm.weight"),
            );
            let (cos, sin) = rope_tables(
                cfg.max_pos,
                cfg.head_dim,
                hf.effective_rope_theta()
                    .expect("config carries rope_theta"),
            ); // plain theta, no YaRN
            weights.insert("rope.cos".to_string(), cos);
            weights.insert("rope.sin".to_string(), sin);

            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                weights.insert(
                    p("input_layernorm.weight"),
                    get_raw(&p("input_layernorm.weight")),
                );
                weights.insert(
                    p("post_attention_layernorm.weight"),
                    get_raw(&p("post_attention_layernorm.weight")),
                );
                weights.insert(
                    p("self_attn.q_proj.weight"),
                    get2d(&p("self_attn.q_proj.weight")),
                );
                weights.insert(
                    p("self_attn.q_proj.bias"),
                    get_raw(&p("self_attn.q_proj.bias")),
                );
                weights.insert(
                    p("self_attn.k_proj.weight"),
                    get2d(&p("self_attn.k_proj.weight")),
                );
                weights.insert(
                    p("self_attn.k_proj.bias"),
                    get_raw(&p("self_attn.k_proj.bias")),
                );
                weights.insert(
                    p("self_attn.v_proj.weight"),
                    get2d(&p("self_attn.v_proj.weight")),
                );
                weights.insert(
                    p("self_attn.v_proj.bias"),
                    get_raw(&p("self_attn.v_proj.bias")),
                );
                weights.insert(
                    p("self_attn.o_proj.weight"),
                    get2d(&p("self_attn.o_proj.weight")),
                );
                weights.insert(
                    p("self_attn.o_proj.bias"),
                    get_raw(&p("self_attn.o_proj.bias")),
                );
                weights.insert(p("self_attn.sinks"), get_raw(&p("self_attn.sinks")));
                // router: HF Linear [out,in]=[E,H] -> transpose to poot's [H,E].
                weights.insert(p("mlp.router.weight"), get2d(&p("mlp.router.weight")));
                weights.insert(p("mlp.router.bias"), get_raw(&p("mlp.router.bias")));
                // experts: already [in,out], no transpose.
                weights.insert(
                    p("mlp.experts.gate_up_proj"),
                    get_raw(&p("mlp.experts.gate_up_proj")),
                );
                weights.insert(
                    p("mlp.experts.gate_up_proj_bias"),
                    get_raw(&p("mlp.experts.gate_up_proj_bias")),
                );
                weights.insert(
                    p("mlp.experts.down_proj"),
                    get_raw(&p("mlp.experts.down_proj")),
                );
                weights.insert(
                    p("mlp.experts.down_proj_bias"),
                    get_raw(&p("mlp.experts.down_proj_bias")),
                );
            }

            let tokens = [1usize, 2, 3, 4];
            let g = trace_gptoss_prefill(cfg, mp.clone(), tokens.len());
            let got = eval_gptoss_prefill(&g, &tokens, &weights, mp.sliding_window);
            let want = gptoss_prefill_ref(&cfg, &mp, &tokens, &weights);

            assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
            assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
            // Vocab is 201088 with a tiny hidden=32 embedding; a few logits sit near zero, where the plain relative
            // check (`abs_diff / max(|a|,|b|,1e-6)`) amplifies f32 summation-order noise (max absolute diff
            // ~6.7e-8 over all logits, vs ~0.1-0.2 logit magnitudes). A 1e-3 absolute floor, still far above the
            // noise, absorbs that without weakening the check elsewhere.
            for (i, (&a, &b)) in got.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
                let denom = a.abs().max(b.abs()).max(1e-3);
                let rel = (a - b).abs() / denom;
                assert!(rel <= 1e-4, "index {i}: {a} vs {b} (rel {rel} > 1e-4)");
            }
        }
    }
}
