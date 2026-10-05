//! OlmoE tracer (card 135d, `specs/262-olmoe-tracer/spec.md`) for `allenai/OLMoE-1B-7B-0924`.
//!
//! OlmoE is a hybrid of existing precedents:
//!
//! - Block topology is standard pre-norm (`OlmoeDecoderLayer.forward`): input layernorm, attention,
//!   residual add, then the same shape for the MLP. This is the opposite of dense olmo2
//!   (`crate::qwen2::trace_olmo2_prefill`), which post-norms each sublayer output.
//! - QK-norm follows olmo2's full-dimension convention: RMSNorm over the whole projection width
//!   (`hidden_size` for q, `head_dim * num_key_value_heads` for k), before the head reshape and RoPE.
//! - The router is not renormalized on the real checkpoint (`"norm_topk_prob": false`).
//!   `OlmoeTopKRouter.forward` takes a softmax over all `num_experts`, selects the top `top_k`, and
//!   divides by the selected sum only when `norm_topk_prob` is set. With it false, the gate weights are
//!   the full-softmax probabilities of the selected experts and do not sum to 1. Mixtral and
//!   [`poot_graph_ir::ops::moe`] always renormalize, and
//!   `poot_load::Qwen2HfConfig::qwen3_moe_norm_topk_prob` rejects `false` for Qwen3-MoE for that reason.
//!   OlmoE cannot reject it because the released checkpoint sets it.
//!
//! The shared MoE op family (`moe`/`moe_dense`/`moe_sparse`/`moe_grouped`/`moe_grouped_prep`) has no
//! non-renormalized form, so this module composes its own router and dispatch
//! ([`olmoe_ffn`]/`olmoe_top_k_gate`) from the same primitives `moe_dense` uses, parameterized by
//! [`OlmoeParams::norm_topk_prob`].
//!
//! Real config: `hidden_size: 2048`, `intermediate_size: 1024` (per-expert FFN width; no dense MLP
//! layer), `num_hidden_layers: 16`, `num_attention_heads: 16`, `num_key_value_heads: 16` (plain MHA;
//! the tracer stays generic over `Qwen2Config::n_kv_heads`), `num_experts: 64`, `num_experts_per_tok: 8`,
//! `norm_topk_prob: false`, `rms_norm_eps` default `1e-05`, `rope_theta: 10000.0`,
//! `attention_bias: false`, `clip_qkv: null` (optional q/k/v clamp, not implemented),
//! `tie_word_embeddings: false`.
//!
//! Safetensors names: `model.layers.{li}.mlp.gate.weight` (router, `[E,H]`),
//! `model.layers.{li}.mlp.experts.{e}.{gate,up,down}_proj.weight` (separate per-expert 2D tensors, same
//! convention as `crate::qwen3moe`, so the loader reuses `fuse_qwen3_moe_experts` in
//! `crates/poot-llm/src/runner.rs::build_weights`), `self_attn.{q,k,v,o}_proj.weight` (no bias), and
//! `self_attn.{q,k}_norm.weight` (full-dim 1D, loaded as plain pass-through constants).
//!
//! The attention/RoPE/norm dims are what [`crate::qwen2::Qwen2Config`] holds, so this module rides on it
//! (like `crate::mixtral`) plus a small [`OlmoeParams`]. Primitive composition only: no new graph-IR op
//! and no hand-fused kernel.

use poot_graph_ir::ops::{
    attention_masked, attention_prefill, canonical_router_scores, linear, rmsnorm, rope,
    rope_prefill, stable_descending_rank, swiglu, top_k_keep_mask,
};
use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Slot, StateRole, TensorType, Traced, UnOp};
use poot_tensor::DType;

use crate::qwen2::Qwen2Config;

/// OlmoE's MoE shape, alongside a [`Qwen2Config`]. Every layer routes (no `sparse_layer` field, unlike
/// [`crate::qwen3moe::Qwen3MoeParams`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OlmoeParams {
    /// total experts per layer (`num_experts`; `64` on the real checkpoint).
    pub n_experts: usize,
    /// experts selected per token (`num_experts_per_tok`; `8` on the real checkpoint).
    pub top_k: usize,
    /// per-expert FFN intermediate size (`intermediate_size`; `1024` on the real checkpoint).
    pub inter: usize,
    /// whether the top-`top_k` gate weights are renormalized to sum to 1 (`norm_topk_prob`; `false` on the
    /// real checkpoint). Chosen at trace time, not a runtime graph branch.
    pub norm_topk_prob: bool,
}

/// Trace a full-sequence OlmoE prefill forward (CPU-oracle path): embeddings, `cfg.layers` pre-norm
/// blocks (input norm, QKV, full-dim QK-norm, RoPE, causal attention, `o_proj` + residual, post-attention
/// norm, top-`mp.top_k`-of-`mp.n_experts` MoE MLP + residual), final `model.norm`, `lm_head.weight`.
///
/// Expects these named constants bound at eval time: `model.embed_tokens.weight` `[vocab,hidden]`,
/// `rope.cos`/`rope.sin` `[max_pos,head_dim]`, per-layer `model.layers.{li}.
/// {input_layernorm,post_attention_layernorm}.weight`, `self_attn.{q,k,v,o}_proj.weight` (no bias),
/// `self_attn.{q,k}_norm.weight` (`[q_dim]`/`[kv_dim]`, full-dim), `mlp.gate.weight` `[hidden,n_experts]`,
/// `mlp.experts.gate_up_proj.weight` `[n_experts,hidden,2*inter]`, `mlp.experts.down_proj.weight`
/// `[n_experts,inter,hidden]`, `model.norm.weight`, and `lm_head.weight` (untied on the real checkpoint),
/// plus the `mask.prefill` step input (card 550a) `[1,1,l,l]`.
pub fn trace_olmoe_prefill(cfg: Qwen2Config, mp: OlmoeParams, seq_len: usize) -> Graph {
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

        // pre-norm: the normed input feeds attention.
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

        // Full-dimension QK-norm, before the head reshape and RoPE.
        let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![q_dim]));
        let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![kv_dim]));
        let q = rmsnorm(&b, q, qn, cfg.eps);
        let k = rmsnorm(&b, k, kn, cfg.eps);

        let q = b.transpose(b.reshape(q, vec![1, l, hq, d]), vec![0, 2, 1, 3]);
        let k = b.transpose(b.reshape(k, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);
        let v = b.transpose(b.reshape(v, vec![1, l, hkv, d]), vec![0, 2, 1, 3]);

        let q = rope_prefill(&b, q, cos, sin, l);
        let k = rope_prefill(&b, k, cos, sin, l);

        let attn = attention_prefill(&b, q, k, v, n_rep, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, q_dim]);
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn); // pre-norm: residual adds the raw sublayer output

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = olmoe_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// OlmoE single-token fixed-KV masked decode, the decode analog of [`trace_olmoe_prefill`] on
/// `crate::qwen2::trace_decode_kv_masked`'s skeleton.
pub fn trace_olmoe_decode_kv_masked(cfg: Qwen2Config, mp: OlmoeParams, cap: usize) -> Graph {
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

        let qn = b.constant(&p("self_attn.q_norm.weight"), TensorType::f32(vec![q_dim]));
        let kn = b.constant(&p("self_attn.k_norm.weight"), TensorType::f32(vec![kv_dim]));
        let q = rmsnorm(&b, q, qn, cfg.eps);
        let k = rmsnorm(&b, k, kn, cfg.eps);

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
        let m = olmoe_ffn(&b, normed, h, &mp, li);
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// OlmoE's top-k router (`OlmoeTopKRouter.forward`): softmax over all `e` experts, select the top `k`,
/// and renormalize the selected values only if `norm_topk_prob`.
///
/// This shares [`poot_graph_ir::ops::top_k_gate`]'s stable rank and exact numerator mask, but with
/// `norm_topk_prob == false` the denominator is the full unmasked softmax sum instead of the masked sum.
/// The wrong denominator gives finite, plausible output, so only a numeric reference comparison (the
/// `cpu_oracle` tests) catches it.
fn olmoe_top_k_gate(b: &Builder, x: Traced, k: usize, norm_topk_prob: bool) -> Traced {
    let shape = b.aval(x).shape;
    let e = *shape.last().expect("router logits have an expert axis");
    let last = shape.len() - 1;
    assert!(
        (1..=e).contains(&k),
        "olmoe_top_k_gate needs 1 <= k <= E, got k={k}, E={e}"
    );

    let scores = canonical_router_scores(b, x);
    let rank = stable_descending_rank(b, scores);
    let keep = top_k_keep_mask(b, rank, k);
    let m = b.reduce(RedOp::Max, scores, last, true);
    let ex_full = b.unary(UnOp::Exp, b.binary(BinOp::Sub, scores, m));
    let ex_masked = b.binary(BinOp::Mul, ex_full, keep);

    let denom = if norm_topk_prob {
        // renormalized (Mixtral/top_k_gate semantics): sum over the selected entries only.
        b.reduce(RedOp::Sum, ex_masked, last, true)
    } else {
        // sum over all E experts (full unmasked softmax denominator).
        b.reduce(RedOp::Sum, ex_full, last, true)
    };
    b.binary(BinOp::Div, ex_masked, denom)
}

/// One layer's MoE MLP with OlmoE's router semantics ([`olmoe_top_k_gate`]); every layer routes.
/// Built from the primitives [`poot_graph_ir::ops::moe_dense`] uses, not a call into it (see the module
/// docs). Dense form (every expert evaluated, non-selected zero-weighted) for prefill and decode, so the
/// shape is static in `L`. No sparse/indexed path (`moe_sparse`/`moe_grouped` lack `norm_topk_prob`).
fn olmoe_ffn(b: &Builder, normed: Traced, h: usize, mp: &OlmoeParams, li: usize) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let e = mp.n_experts;
    let k = mp.top_k;
    let inter = mp.inter;

    let router = b.constant(&p("mlp.gate.weight"), TensorType::f32(vec![h, e]));
    let w_in = b.constant(
        &p("mlp.experts.gate_up_proj.weight"),
        TensorType::f32(vec![e, h, 2 * inter]),
    );
    let w_out = b.constant(
        &p("mlp.experts.down_proj.weight"),
        TensorType::f32(vec![e, inter, h]),
    );

    let shape = b.aval(normed).shape;
    let l = shape[shape.len() - 2];
    let xm = b.reshape(normed, vec![l, h]);

    // route: logits [L,E] -> per-row top-k gate weights [L,E] (may not sum to 1 when !norm_topk_prob).
    let logits = linear(b, xm, router, None);
    let gate = olmoe_top_k_gate(b, logits, k, mp.norm_topk_prob);

    // every expert's swiglu FFN, batched over the expert axis (same dense form as moe_dense).
    let x_e = b.broadcast(b.reshape(xm, vec![1, l, h]), vec![e, l, h]); // [E,L,H]
    let gu = b.matmul(x_e, w_in); // [E,L,2I]
    let g_part = b.slice(gu, 2, 0, inter); // [E,L,I]
    let u_part = b.slice(gu, 2, inter, 2 * inter); // [E,L,I]
    let act = swiglu(b, g_part, u_part); // [E,L,I]
    let out = b.matmul(act, w_out); // [E,L,H]

    // weight by gate value and sum over experts (axis 0).
    let gate_w = b.reshape(b.transpose(gate, vec![1, 0]), vec![e, l, 1]); // [E,L,1]
    let weighted = b.binary(BinOp::Mul, out, gate_w); // [E,L,H]
    let y = b.reduce(RedOp::Sum, weighted, 0, false); // [L,H]
    b.reshape(y, vec![1, l, h])
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    fn tiny_cfg() -> Qwen2Config {
        // small dims with GQA (the real checkpoint is MHA, but the tracer is generic); no cfg.qk_norm since
        // QK-norm is unconditional in this module; no qkv_bias.
        Qwen2Config {
            vocab: 24,
            hidden: 16,
            inter: 16, // unused directly by olmoe (MoE inter comes from OlmoeParams), kept non-zero
            layers: 3,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            rotary_dim: 4,
            eps: 1e-5,
            max_pos: 32,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        }
    }

    fn tiny_mp(norm_topk_prob: bool) -> OlmoeParams {
        OlmoeParams {
            n_experts: 6,
            top_k: 2,
            inter: 12,
            norm_topk_prob,
        }
    }

    #[test]
    fn olmoe_prefill_validates_pre_norm_no_bias_full_dim_qk_norm() {
        let cfg = tiny_cfg();
        let mp = tiny_mp(false);
        let g = trace_olmoe_prefill(cfg, mp, 5);
        g.validate().expect("olmoe prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // no bias constant anywhere.
        for id in &g.inputs {
            if let Storage::Const = g.values[*id].storage {
                let name = g.values[*id].name.as_deref().unwrap_or("");
                assert!(!name.ends_with(".bias"), "OlmoE has no bias, found {name}");
            }
        }
        // full-dim QK-norm constants (full projection width, not per-head).
        let q_norm = g
            .inputs
            .iter()
            .find(|id| {
                g.values[**id]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.ends_with("self_attn.q_norm.weight"))
            })
            .expect("q_norm constant present");
        assert_eq!(
            g.values[*q_norm].aval.shape,
            vec![cfg.n_heads * cfg.head_dim]
        );
        let k_norm = g
            .inputs
            .iter()
            .find(|id| {
                g.values[**id]
                    .name
                    .as_deref()
                    .is_some_and(|n| n.ends_with("self_attn.k_norm.weight"))
            })
            .expect("k_norm constant present");
        assert_eq!(
            g.values[*k_norm].aval.shape,
            vec![cfg.n_kv_heads * cfg.head_dim]
        );
        // every layer declares a router constant (MoE branch is unconditional).
        let router_count = g
            .inputs
            .iter()
            .filter(|id| {
                matches!(g.values[**id].storage, Storage::Const)
                    && g.values[**id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with("mlp.gate.weight"))
            })
            .count();
        assert_eq!(router_count, cfg.layers, "one router per layer, no skips");
        // pre-norm: input_layernorm present (olmo2 has none).
        let has_input_ln = g.inputs.iter().any(|id| {
            g.values[*id]
                .name
                .as_deref()
                .is_some_and(|n| n.ends_with("input_layernorm.weight"))
        });
        assert!(
            has_input_ln,
            "olmoe is pre-norm: input_layernorm must exist"
        );
    }

    #[test]
    fn olmoe_decode_kv_masked_validates_state_and_mask_shapes() {
        let cfg = tiny_cfg();
        let mp = tiny_mp(false);
        let cap = 16;
        let g = trace_olmoe_decode_kv_masked(cfg, mp, cap);
        g.validate().expect("olmoe decode graph should validate");
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
    }

    // ---- CPU-oracle numerics (poot_eval) against an independent Rust reference forward pass. ----

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

        /// All named constants [`trace_olmoe_prefill`] declares, keyed by exact graph const name.
        fn all_weights(cfg: &Qwen2Config, mp: &OlmoeParams) -> HashMap<String, Vec<f32>> {
            let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
            let (q_dim, kv_dim) = (hq * d, hkv * d);
            let mut w = HashMap::new();
            w.insert(
                "model.embed_tokens.weight".to_string(),
                weight("embed", cfg.vocab * h, false),
            );
            let (cos, sin) = rope_tables(cfg.max_pos, d, 10_000.0); // real rope_theta
            w.insert("rope.cos".to_string(), cos);
            w.insert("rope.sin".to_string(), sin);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1"), h, true));
                w.insert(
                    p("self_attn.q_proj.weight"),
                    weight(&p("qw"), h * q_dim, false),
                );
                w.insert(
                    p("self_attn.k_proj.weight"),
                    weight(&p("kw"), h * kv_dim, false),
                );
                w.insert(
                    p("self_attn.v_proj.weight"),
                    weight(&p("vw"), h * kv_dim, false),
                );
                w.insert(
                    p("self_attn.o_proj.weight"),
                    weight(&p("ow"), q_dim * h, false),
                );
                w.insert(p("self_attn.q_norm.weight"), weight(&p("qn"), q_dim, true));
                w.insert(p("self_attn.k_norm.weight"), weight(&p("kn"), kv_dim, true));
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2"), h, true),
                );
                w.insert(
                    p("mlp.gate.weight"),
                    weight(&p("router"), h * mp.n_experts, false),
                );
                // gate_up_proj: [E, H, 2I], gate||up per expert, [in,out].
                let mut gate_up = vec![0.0f32; mp.n_experts * h * 2 * mp.inter];
                let mut down = vec![0.0f32; mp.n_experts * mp.inter * h];
                for e in 0..mp.n_experts {
                    let g = weight(&p(&format!("e{e}.gate")), h * mp.inter, false);
                    let u = weight(&p(&format!("e{e}.up")), h * mp.inter, false);
                    let d_ = weight(&p(&format!("e{e}.down")), mp.inter * h, false);
                    for row in 0..h {
                        let dst = (e * h + row) * 2 * mp.inter;
                        gate_up[dst..dst + mp.inter]
                            .copy_from_slice(&g[row * mp.inter..(row + 1) * mp.inter]);
                        gate_up[dst + mp.inter..dst + 2 * mp.inter]
                            .copy_from_slice(&u[row * mp.inter..(row + 1) * mp.inter]);
                    }
                    let dst = e * mp.inter * h;
                    down[dst..dst + mp.inter * h].copy_from_slice(&d_);
                }
                w.insert(p("mlp.experts.gate_up_proj.weight"), gate_up);
                w.insert(p("mlp.experts.down_proj.weight"), down);
            }
            w.insert("model.norm.weight".to_string(), weight("ln_f", h, true));
            w.insert(
                "lm_head.weight".to_string(),
                weight("lm_head", h * cfg.vocab, false),
            );
            w
        }

        use poot_test_util::rmsnorm_ref;

        use poot_test_util::silu_ref;

        use poot_test_util::linear_ref;

        use poot_test_util::rope_ref;

        /// Independent top-`top_k` router (softmax, topk, optional renormalize), re-derived directly rather
        /// than reusing `olmoe_top_k_gate`. Returns `(expert_id, weight)` pairs; weights sum to 1 only when
        /// `norm_topk_prob`.
        fn top_k_router_ref(
            logits: &[f32],
            top_k: usize,
            norm_topk_prob: bool,
        ) -> Vec<(usize, f32)> {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let sel = &idx[..top_k];
            let m = logits.iter().cloned().fold(f32::MIN, f32::max); // full max over ALL experts
            let exp_sel: Vec<f32> = sel.iter().map(|&i| (logits[i] - m).exp()).collect();
            let denom = if norm_topk_prob {
                exp_sel.iter().sum::<f32>()
            } else {
                logits.iter().map(|&v| (v - m).exp()).sum::<f32>()
            };
            sel.iter()
                .zip(exp_sel.iter())
                .map(|(&i, &e)| (i, e / denom))
                .collect()
        }

        /// Independent Rust reference forward pass for [`trace_olmoe_prefill`] (last token only), written
        /// as direct loops rather than a copy of `olmoe_ffn`.
        fn olmoe_prefill_ref(
            cfg: &Qwen2Config,
            mp: &OlmoeParams,
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
                let ln1w = &w[&p("input_layernorm.weight")];
                let normed: Vec<Vec<f32>> = x
                    .iter()
                    .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
                    .collect();

                let qw = &w[&p("self_attn.q_proj.weight")];
                let kw = &w[&p("self_attn.k_proj.weight")];
                let vw = &w[&p("self_attn.v_proj.weight")];
                let mut q: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, qw, h, hq * d))
                    .collect();
                let mut k: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, kw, h, hkv * d))
                    .collect();
                let v: Vec<Vec<f32>> = normed
                    .iter()
                    .map(|row| linear_ref(row, vw, h, hkv * d))
                    .collect();

                // full-dimension QK-norm, before the per-head RoPE loop.
                let qnw = &w[&p("self_attn.q_norm.weight")];
                let knw = &w[&p("self_attn.k_norm.weight")];
                for qrow in q.iter_mut() {
                    *qrow = rmsnorm_ref(qrow, qnw, hq * d, cfg.eps);
                }
                for krow in k.iter_mut() {
                    *krow = rmsnorm_ref(krow, knw, hkv * d, cfg.eps);
                }

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

                let mut attn_out = vec![vec![0.0f32; h]; l];
                for qh in 0..hq {
                    let kh = qh / n_rep;
                    for i in 0..l {
                        let mut scores = vec![0.0f32; i + 1];
                        for (j, sc) in scores.iter_mut().enumerate() {
                            let mut s = 0.0f32;
                            for dd in 0..d {
                                s += q[i][qh * d + dd] * k[j][kh * d + dd];
                            }
                            *sc = s * scale;
                        }
                        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut denom = 0.0f32;
                        let mut e = vec![0.0f32; scores.len()];
                        for (j, sc) in scores.iter().enumerate() {
                            e[j] = (sc - m).exp();
                            denom += e[j];
                        }
                        for dd in 0..d {
                            let mut acc = 0.0f32;
                            for (j, ej) in e.iter().enumerate() {
                                acc += (ej / denom) * v[j][kh * d + dd];
                            }
                            attn_out[i][qh * d + dd] = acc;
                        }
                    }
                }

                let ow = &w[&p("self_attn.o_proj.weight")];
                for i in 0..l {
                    let proj = linear_ref(&attn_out[i], ow, hq * d, h);
                    for c in 0..h {
                        x[i][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                let router_w = &w[&p("mlp.gate.weight")];
                let gate_up = &w[&p("mlp.experts.gate_up_proj.weight")];
                let down = &w[&p("mlp.experts.down_proj.weight")];
                for row in x.iter_mut().take(l) {
                    let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let logits = linear_ref(&normed, router_w, h, mp.n_experts);
                    let selected = top_k_router_ref(&logits, mp.top_k, mp.norm_topk_prob);
                    let mut moe_out = vec![0.0f32; h];
                    for (e, gate_weight) in selected {
                        let e_gate_up = &gate_up[e * h * 2 * mp.inter..(e + 1) * h * 2 * mp.inter];
                        let mut gate_w = vec![0.0f32; h * mp.inter];
                        let mut up_w = vec![0.0f32; h * mp.inter];
                        for row_i in 0..h {
                            let src = row_i * 2 * mp.inter;
                            gate_w[row_i * mp.inter..(row_i + 1) * mp.inter]
                                .copy_from_slice(&e_gate_up[src..src + mp.inter]);
                            up_w[row_i * mp.inter..(row_i + 1) * mp.inter]
                                .copy_from_slice(&e_gate_up[src + mp.inter..src + 2 * mp.inter]);
                        }
                        let e_down = &down[e * mp.inter * h..(e + 1) * mp.inter * h];
                        let gate = linear_ref(&normed, &gate_w, h, mp.inter);
                        let up = linear_ref(&normed, &up_w, h, mp.inter);
                        let act: Vec<f32> = gate
                            .iter()
                            .zip(up.iter())
                            .map(|(&g, &u)| silu_ref(g) * u)
                            .collect();
                        let expert_out = linear_ref(&act, e_down, mp.inter, h);
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
            linear_ref(&last, lm_head, h, cfg.vocab)
        }

        fn eval_olmoe_prefill(
            g: &Graph,
            tokens: &[usize],
            weights: &HashMap<String, Vec<f32>>,
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
                        assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                        let l = meta.aval.shape[2];
                        let mut m = vec![0.0f32; l * l];
                        for i in 0..l {
                            for j in 0..l {
                                m[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
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

        /// `trace_olmoe_prefill` matches the independent reference within f32 tolerance at
        /// `norm_topk_prob: false` (the real checkpoint setting).
        #[test]
        fn olmoe_prefill_matches_hand_rolled_reference_norm_topk_prob_false() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(false);
            let tokens = [3usize, 7, 1, 9, 15];
            let weights = all_weights(&cfg, &mp);

            let g = trace_olmoe_prefill(cfg, mp, tokens.len());
            let got = eval_olmoe_prefill(&g, &tokens, &weights);
            let want = olmoe_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Same oracle with `norm_topk_prob: true`, covering the other branch of `olmoe_top_k_gate`.
        #[test]
        fn olmoe_prefill_matches_hand_rolled_reference_norm_topk_prob_true() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(true);
            let tokens = [2usize, 4, 6, 8];
            let weights = all_weights(&cfg, &mp);

            let g = trace_olmoe_prefill(cfg, mp, tokens.len());
            let got = eval_olmoe_prefill(&g, &tokens, &weights);
            let want = olmoe_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }

        /// Flipping `norm_topk_prob` changes the traced output for the same weights, so the wrong-denominator
        /// bug would not go undetected.
        #[test]
        fn olmoe_norm_topk_prob_flag_changes_output() {
            let cfg = tiny_cfg();
            let tokens = [1usize, 3, 5];
            let weights = all_weights(&cfg, &tiny_mp(false)); // same weights for both (flag is orthogonal)

            let g_false = trace_olmoe_prefill(cfg, tiny_mp(false), tokens.len());
            let g_true = trace_olmoe_prefill(cfg, tiny_mp(true), tokens.len());
            let out_false = eval_olmoe_prefill(&g_false, &tokens, &weights);
            let out_true = eval_olmoe_prefill(&g_true, &tokens, &weights);

            let max_diff = poot_test_util::max_abs_error(
                out_false.as_f32().unwrap(),
                out_true.as_f32().unwrap(),
            );
            assert!(
                max_diff > 1e-4,
                "norm_topk_prob should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// Perturbing the first weight of every expert's layer-1 down-projection changes the output. Every expert
        /// is perturbed because which experts three tokens route to depends on the fixture's seeds; a routed one
        /// always moves the output.
        #[test]
        fn olmoe_prefill_output_is_sensitive_to_a_perturbed_expert_weight() {
            let cfg = tiny_cfg();
            let mp = tiny_mp(false);
            let tokens = [2usize, 5, 8];
            let mut weights = all_weights(&cfg, &mp);
            let g = trace_olmoe_prefill(cfg, mp, tokens.len());
            let base = eval_olmoe_prefill(&g, &tokens, &weights);

            let key = "model.layers.1.mlp.experts.down_proj.weight".to_string();
            let down = weights.get_mut(&key).unwrap();
            let per_expert = down.len() / mp.n_experts;
            for expert in 0..mp.n_experts {
                down[expert * per_expert] += 5.0;
            }
            let perturbed = eval_olmoe_prefill(&g, &tokens, &weights);

            let max_diff =
                poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
            assert!(
                max_diff > 1e-4,
                "perturbing {key} should change the output; max_diff={max_diff:.2e}"
            );
        }

        // Weight-layout crosswalk against `hf-tiny-v2/tiny-random-OlmoeForCausalLM` (2 layers, hidden 32,
        // 4 heads MHA, 8 experts top-2, intermediate 16, randomly initialized): proves the loader's
        // transpose/fuse of per-expert gate/up/down tensors matches what this tracer binds. Loads via
        // `poot_load` (poot_llm sits above this crate) and compares against `olmoe_prefill_ref`. Its config
        // also sets `norm_topk_prob: false`, so this exercises the non-renormalized router end to end.
        #[test]
        fn olmoe_tiny_random_checkpoint_matches_hand_rolled_reference() {
            // Populate `$POOT_MODELS_DIR/olmoe-tiny` with `hf-tiny-v2/tiny-random-OlmoeForCausalLM`'s
            // config.json/model.safetensors/tokenizer.json.
            let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-tiny"))
            else {
                return;
            };

            let hf =
                poot_load::Qwen2HfConfig::load(dir.join("config.json")).expect("load config.json");
            assert!(hf.is_olmoe());
            assert_eq!(hf.norm_topk_prob, Some(false));
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
            let mp = OlmoeParams {
                n_experts: hf.num_experts.expect("olmoe config carries num_experts"),
                top_k: hf
                    .num_experts_per_tok
                    .expect("olmoe config carries num_experts_per_tok"),
                inter: hf.intermediate_size,
                norm_topk_prob: hf.norm_topk_prob.unwrap_or(false),
            };
            assert_eq!(mp.n_experts, 8);
            assert_eq!(mp.top_k, 2);
            assert!(!mp.norm_topk_prob);

            fn transpose2d(data: &[f32], r: usize, c: usize) -> Vec<f32> {
                let mut out = vec![0.0f32; r * c];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = data[i * c + j];
                    }
                }
                out
            }

            /// Fuse per-expert `{prefix}.{e}.{gate,up,down}_proj.weight` tensors into the fused `[E,H,2I]`
            /// gate||up and `[E,I,H]` down layout `olmoe_ffn` binds (same as
            /// `poot_llm::runner::fuse_qwen3_moe_experts`).
            fn fuse_experts(
                st: &poot_quant::weights::WeightStore,
                prefix: &str,
                n_experts: usize,
                hidden: usize,
                inter: usize,
            ) -> (Vec<f32>, Vec<f32>) {
                let mut gate_up = vec![0.0f32; n_experts * hidden * 2 * inter];
                let mut down = vec![0.0f32; n_experts * inter * hidden];
                for e in 0..n_experts {
                    let g =
                        poot_eval::materialize_dense(st, &format!("{prefix}.{e}.gate_proj.weight"))
                            .unwrap_or_else(|err| panic!("{prefix}.{e}.gate_proj.weight: {err}"));
                    let u =
                        poot_eval::materialize_dense(st, &format!("{prefix}.{e}.up_proj.weight"))
                            .unwrap_or_else(|err| panic!("{prefix}.{e}.up_proj.weight: {err}"));
                    let d_ =
                        poot_eval::materialize_dense(st, &format!("{prefix}.{e}.down_proj.weight"))
                            .unwrap_or_else(|err| panic!("{prefix}.{e}.down_proj.weight: {err}"));
                    // gate_proj/up_proj: HF [inter, hidden] (out,in) -> [hidden, inter] (in,out).
                    let gt = transpose2d(g.as_f32().unwrap(), inter, hidden);
                    let ut = transpose2d(u.as_f32().unwrap(), inter, hidden);
                    for row in 0..hidden {
                        let dst = (e * hidden + row) * 2 * inter;
                        gate_up[dst..dst + inter]
                            .copy_from_slice(&gt[row * inter..(row + 1) * inter]);
                        gate_up[dst + inter..dst + 2 * inter]
                            .copy_from_slice(&ut[row * inter..(row + 1) * inter]);
                    }
                    // down_proj: HF [hidden, inter] (out,in) -> [inter, hidden] (in,out).
                    let dt = transpose2d(d_.as_f32().unwrap(), hidden, inter);
                    let dst = e * inter * hidden;
                    down[dst..dst + inter * hidden].copy_from_slice(&dt);
                }
                (gate_up, down)
            }

            let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
            let get2d = |name: &str| -> Vec<f32> {
                let rt = poot_eval::materialize_dense(&st, name)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                transpose2d(rt.as_f32().unwrap(), rt.shape()[0], rt.shape()[1])
            };
            let get1d = |name: &str| -> Vec<f32> {
                poot_eval::materialize_dense(&st, name)
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                    .as_f32()
                    .unwrap()
                    .to_vec()
            };

            weights.insert(
                "model.embed_tokens.weight".to_string(),
                get1d("model.embed_tokens.weight"),
            );
            // untied (tie_word_embeddings: false).
            weights.insert("lm_head.weight".to_string(), get2d("lm_head.weight"));
            weights.insert("model.norm.weight".to_string(), get1d("model.norm.weight"));
            let (cos, sin) = rope_tables(
                cfg.max_pos,
                cfg.head_dim,
                hf.effective_rope_theta()
                    .expect("config carries rope_theta"),
            );
            weights.insert("rope.cos".to_string(), cos);
            weights.insert("rope.sin".to_string(), sin);

            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                weights.insert(
                    p("input_layernorm.weight"),
                    get1d(&p("input_layernorm.weight")),
                );
                weights.insert(
                    p("post_attention_layernorm.weight"),
                    get1d(&p("post_attention_layernorm.weight")),
                );
                weights.insert(
                    p("self_attn.q_proj.weight"),
                    get2d(&p("self_attn.q_proj.weight")),
                );
                weights.insert(
                    p("self_attn.k_proj.weight"),
                    get2d(&p("self_attn.k_proj.weight")),
                );
                weights.insert(
                    p("self_attn.v_proj.weight"),
                    get2d(&p("self_attn.v_proj.weight")),
                );
                weights.insert(
                    p("self_attn.o_proj.weight"),
                    get2d(&p("self_attn.o_proj.weight")),
                );
                weights.insert(
                    p("self_attn.q_norm.weight"),
                    get1d(&p("self_attn.q_norm.weight")),
                );
                weights.insert(
                    p("self_attn.k_norm.weight"),
                    get1d(&p("self_attn.k_norm.weight")),
                );
                weights.insert(p("mlp.gate.weight"), get2d(&p("mlp.gate.weight")));
                let (gate_up, down) =
                    fuse_experts(&st, &p("mlp.experts"), mp.n_experts, cfg.hidden, mp.inter);
                weights.insert(p("mlp.experts.gate_up_proj.weight"), gate_up);
                weights.insert(p("mlp.experts.down_proj.weight"), down);
            }

            let tokens = [1usize, 2, 3, 4];
            let g = trace_olmoe_prefill(cfg, mp, tokens.len());
            let got = eval_olmoe_prefill(&g, &tokens, &weights);
            let want = olmoe_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }
    }
}
