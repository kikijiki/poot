//! DeepSeek-V3 CPU-oracle tracer, first slice (epic 129's audit; only `crate::deepseek2`'s V2 existed
//! before). Scoped like `crate::qwen3moe`'s first slice (update 0774): config struct, prefill/decode
//! CPU-oracle tracers, bit-exact independent-reference tests, F32 only. No safetensors/GGUF loader,
//! `Runner`/serve wiring, or GPU path.
//!
//! Verified against `deepseek-ai/DeepSeek-V3`'s `config.json` and `modeling_deepseek.py`
//! `MoEGate.forward` (HF `main`).
//!
//! # Identical to DeepSeek-V2 (reused)
//!
//! The attention block is the same MLA as `crate::deepseek2`: low-rank (or plain) Q split into
//! nope/rope per head, a shared low-rank KV latent decompressed through `kv_b_proj` on every forward
//! (no weight absorption, see `crate::deepseek2`), interleaved-pair RoPE, and YaRN with the
//! `mscale`/`mscale_all_dim` softmax-scale correction. V3's `config.json` (`q_lora_rank: 1536`,
//! `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`,
//! `num_attention_heads: 128`, `rope_scaling: {"type": "yarn", "factor": 40, ...}`) has the same fields
//! as `DeepseekV2Config`. This module uses [`crate::deepseek2::DeepseekV2Config`] directly and
//! reuses [`crate::deepseek2::mla_query_proj`], [`crate::deepseek2::rope_interleaved_prefill`],
//! [`crate::deepseek2::rope_interleaved_decode`] and [`crate::deepseek2::deepseek2_dense_ffn`]
//! (`DeepseekV3MLP` equals `DeepseekV2MLP`) unchanged.
//!
//! The dense-vs-routed split is the same `li >= first_k_dense_replace` threshold
//! (`crate::deepseek2::DeepseekV2MoeParams::is_moe_layer`, reimplemented on [`DeepseekV3MoeParams`]),
//! with `first_k_dense_replace` `3` on V3 vs `1` on V2. The shared expert (one SwiGLU MLP, added with
//! no gate) is also the same, with `n_shared_experts: 1` vs V2's `2`
//! (`DeepseekV3MoE.forward`: `hidden_states = expert_output + shared_output`).
//!
//! # Different: the router (`topk_method: "noaux_tc"`, `scoring_func: "sigmoid"`)
//!
//! `MoEGate.forward` from `modeling_deepseek.py`:
//!
//! ```text
//! logits = F.linear(hidden_states, self.weight)
//! scores = logits.sigmoid()                                    # sigmoid, not softmax (V2's scoring_func)
//! scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)   # selection-only additive bias
//! group_scores = scores_for_choice.view(..., n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
//! group_idx = group_scores.topk(topk_group, dim=-1)[1]          # group-limited routing
//! ... mask scores_for_choice to the selected groups, then topk(top_k) over the masked result -> topk_idx
//! topk_weight = scores.gather(1, topk_idx)                      # the weight reads the unbiased sigmoid score
//! if top_k > 1 and norm_topk_prob:
//!     topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)   # renormalized
//! topk_weight = topk_weight * self.routed_scaling_factor
//! ```
//!
//! Differences from V2's `crate::deepseek2::deepseek2_router_gate` (softmax, no bias, raw
//! non-renormalized selected values):
//! 1. **`sigmoid`, not `softmax`.** [`poot_graph_ir::ops::sigmoid`] already exists (Qwen3-Next's
//!    shared-expert gate). Sigmoid scores are independent per expert, so the reference needs an
//!    explicit renormalization after top-k.
//! 2. **`e_score_correction_bias`** (learned per-expert, `[n_routed_experts]`) shifts which experts are
//!    selected but not the weight they contribute: `topk_weight` reads `scores`, not
//!    `scores_for_choice`. This is DeepSeek's auxiliary-loss-free load balancing (`noaux_tc`).
//! 3. **Selected weights are renormalized to sum to `1.0`** before `routed_scaling_factor`
//!    (`norm_topk_prob: true`), the opposite of V2, which keeps raw softmax values.
//!
//! [`deepseek3_router_gate`] implements this, including group-limited routing (`n_group`/`topk_group`;
//! V3 uses `n_group: 8`, `topk_group: 4`, so plain top-`k` over all experts would not reproduce real
//! routing). Group limiting uses the same reduce-then-stable-rank vocabulary one level up: split the
//! expert axis into `n_group` groups, sum each group's top-2 `scores_for_choice`, rank the groups,
//! keep the `topk_group` best, then push `scores_for_choice` to `-1e30` for experts outside them
//! before the expert-level top-`k` rank. No new `poot_graph_ir` primitive; `reshape`/`broadcast`
//! give the `(.., n_group, group_size)` view. `n_group=1` and `topk_group=n_group` are no-ops (the
//! `-1e30` term is `0.0` everywhere); the tests check they match plain top-k, and that V3's real
//! `n_group`/`topk_group` ratio changes the selection.
//!
//! # Scope cuts
//!
//! - **Multi-token prediction (`num_nextn_predict_layers: 1`):** extra training-time heads predicting
//!   token `t+2`, sometimes used for speculative drafting. Standard generation never uses them, so
//!   they are skipped.
//! - **No loader, `poot_llm::runner` wiring, or GPU dispatch.** A real loader would hit the same
//!   `build_weights` generic-transpose gotcha as `kv_a_proj_with_mqa` in `crate::deepseek2` (names not
//!   ending in `"proj"` need an explicit rule), plus `mlp.gate.e_score_correction_bias` (a plain
//!   per-expert vector, no transpose) must not break the generic loop.
//! - **FP8 weights** (`quantization_config: {"quant_method": "fp8", ...}`): irrelevant to an F32
//!   CPU-oracle tracer; a loader would use the existing FP8/quant machinery (`crate::mixtral`/
//!   `crate::gpt_oss` packed-quant pools).
//!
//! # Real config fields (`deepseek-ai/DeepSeek-V3/config.json`)
//!
//! `vocab_size` 129280, `hidden_size` 7168, `num_hidden_layers` 61, `num_attention_heads` 128,
//! `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128,
//! `n_routed_experts` 256, `n_shared_experts` 1, `num_experts_per_tok` 8, `moe_intermediate_size` 2048,
//! `intermediate_size` (dense-layer width) 18432, `first_k_dense_replace` 3, `routed_scaling_factor` 2.5,
//! `topk_method` `"noaux_tc"`, `scoring_func` `"sigmoid"`, `n_group` 8, `topk_group` 4, `norm_topk_prob`
//! `true`, `rms_norm_eps` 1e-6, `attention_bias` `false`, `rope_theta` 10000, `rope_scaling` YaRN
//! (`factor` 40, `beta_fast` 32, `beta_slow` 1, `mscale`/`mscale_all_dim` 1.0,
//! `original_max_position_embeddings` 4096).

use crate::components::{gather_shared_pool, scatter_shared_pool};
use crate::deepseek2::{
    DeepseekV2Config, deepseek2_dense_ffn, mla_query_proj, rope_interleaved_decode,
    rope_interleaved_decode_batched, rope_interleaved_prefill,
};
use poot_graph_ir::ops::{
    attention_masked, attention_prefill, linear, repeat_kv, rmsnorm, sigmoid, swiglu,
};
use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Scalar, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

/// DeepSeek-V3's MoE shape: routed top-k experts selected via sigmoid plus correction bias
/// (`topk_method: "noaux_tc"`), shared expert(s), and the dense-vs-routed layer threshold. A separate
/// struct from [`crate::deepseek2::DeepseekV2MoeParams`] because the router contract differs
/// (sigmoid + selection-only bias + renormalize vs softmax + non-renormalized).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV3MoeParams {
    /// total routed experts per MoE layer (`n_routed_experts`; `256` on V3).
    pub n_routed_experts: usize,
    /// routed experts selected per token (`num_experts_per_tok`; `8` on V3).
    pub top_k: usize,
    /// per-routed-expert FFN intermediate size (`moe_intermediate_size`; `2048` on V3).
    pub moe_inter: usize,
    /// always-active shared experts, combined into one MLP of width `moe_inter * n_shared_experts`
    /// (`n_shared_experts`; `1` on V3). `0` disables the branch.
    pub n_shared_experts: usize,
    /// the dense-layer (`li < first_k_dense_replace`) SwiGLU MLP width (`intermediate_size`; `18432` on V3).
    pub dense_inter: usize,
    /// layers `< first_k_dense_replace` use the dense MLP, later layers route (`3` on V3).
    pub first_k_dense_replace: usize,
    /// equal-size groups the router logit axis splits into (`n_group`; `8` on V3). Must divide
    /// `n_routed_experts`. `1` makes group limiting a no-op (see [`deepseek3_router_gate`]).
    pub n_group: usize,
    /// groups selected per token before the expert-level top-`k` (`topk_group`; `4` on V3).
    /// `topk_group == n_group` is also a no-op.
    pub topk_group: usize,
    /// scalar applied to the renormalized gate weights (`routed_scaling_factor`; `2.5` on V3). Unlike
    /// V2's field of the same name, it scales renormalized values (see [`deepseek3_router_gate`]).
    pub routed_scaling_factor: f32,
}

impl DeepseekV3MoeParams {
    /// Whether layer `li` routes (`true`) or uses the dense MLP (`false`); same threshold as
    /// [`crate::deepseek2::DeepseekV2MoeParams::is_moe_layer`].
    pub fn is_moe_layer(&self, li: usize) -> bool {
        li >= self.first_k_dense_replace
    }
}

/// `Runner`-level bundle of DeepSeek-V3's two param structs (MLA rides on
/// [`crate::deepseek2::DeepseekV2Config`]), mirroring [`crate::deepseek2::DeepseekV2Params`]. Not yet
/// consumed by `Runner` (no loader/serve wiring).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV3Params {
    pub cfg: DeepseekV2Config,
    pub moe: DeepseekV3MoeParams,
}

/// Trace a full-sequence DeepSeek-V3 prefill forward (the CPU-oracle path): embeddings, `cfg.layers`
/// pre-norm blocks (MLA attention identical to [`crate::deepseek2::trace_deepseek2_prefill`],
/// `o_proj` + residual, dense-or-routed+shared MoE MLP + residual), final `model.norm`,
/// `lm_head.weight`. Only the router is new relative to `crate::deepseek2`.
///
/// Expects the same constants as [`crate::deepseek2::trace_deepseek2_prefill`] for attention and
/// dense layers, plus (routed layers) `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` `[E]`
/// (the `noaux_tc` selection bias, see [`deepseek3_router_gate`]),
/// `mlp.experts.{gate_up,down}_proj.weight`, and (if `mp.n_shared_experts>0`)
/// `mlp.shared_experts.{gate,up,down}_proj.weight`.
pub fn trace_deepseek3_prefill(
    cfg: DeepseekV2Config,
    mp: DeepseekV3MoeParams,
    seq_len: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let l = seq_len;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
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

        let (q, _qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let q = b.transpose(b.reshape(q, vec![1, l, hq, qk_head_dim]), vec![0, 2, 1, 3]);
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_prefill(&b, q_pe, cos, sin, l);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None);
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps);

        let k_pe = b.reshape(k_pe_raw, vec![1, 1, l, rope_d]);
        let k_pe = rope_interleaved_prefill(&b, k_pe, cos, sin, l);

        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_kv, w_kvb, None);
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, l, hq, nope + vd]),
            vec![0, 2, 1, 3],
        );
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let k_pe_b = repeat_kv(&b, k_pe, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        let attn = attention_prefill(&b, q_full, k_full, v_h, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            deepseek3_moe_ffn(&b, normed, h, &mp, li)
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish(logits)
}

/// Spec 269 stage 5: shared-pool paged prefill, the admission-time twin of
/// [`trace_deepseek3_decode_kv_masked_batched_shared_pool`], built on [`trace_deepseek3_prefill`]'s
/// attention block. It admits one sequence's whole `n`-token prompt in one forward, like
/// [`crate::moe_prefill::trace_moe_prefill_kv_shared_pool`] and
/// [`crate::mixtral::trace_mixtral_prefill_kv_shared_pool`]. `n` is the prompt length; `pool` is the
/// shared physical pool size (`pool >= n`).
///
/// Addressing mirrors `trace_moe_prefill_kv_shared_pool`'s `PrefillKv::SharedPool` write path,
/// doubled for MLA's two pools (update 0811 "Question 2"): a `Slot::SlotMap` `[pool]` inverse map
/// (`inv[physical] = logical prompt position`, or `-1` to pass another sequence's/zero row through)
/// addresses both `[pool, 1, kv_lora_rank]` (`mla.c_cache`) and `[pool, 1, qk_rope_head_dim]`
/// (`mla.rope_cache`) via `Builder::scatter_update` (`hkv = 1` folded into the pool shape). Both pools
/// use the same map so a position's latent and RoPE key cannot land in different rows (spec 269
/// "Internal correctness"). Attention reads the fresh `l`-position K/V directly, and the cache write
/// only feeds carried state, so logits are bit-exact against [`trace_deepseek3_prefill`] regardless of
/// physical slot layout.
///
/// State names/shapes match [`trace_deepseek3_decode_kv_masked_batched_shared_pool`] when
/// `pool == kv_pool_slots`, so a following batched decode reads the same pool with no reshaping.
pub fn trace_deepseek3_prefill_kv_shared_pool(
    cfg: DeepseekV2Config,
    mp: DeepseekV3MoeParams,
    n: usize,
    pool: usize,
) -> Graph {
    assert!(
        pool >= n,
        "shared pool size {pool} must be at least the prompt length {n}"
    );
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let l = n;

    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    // [pool] inverse slot map: inv[physical] = prompt position whose latent/RoPE-key lands there, or -1
    // to pass the existing row through (as `trace_moe_prefill_kv_shared_pool`).
    let inv = b.slot(Slot::SlotMap, TensorType::new(vec![pool], DType::I32));
    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let emb = b.gather(embed, 0, tokens); // [L, hidden]
    let mut x = b.reshape(emb, vec![1, l, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let (q, _qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let q = b.transpose(b.reshape(q, vec![1, l, hq, qk_head_dim]), vec![0, 2, 1, 3]);
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_prefill(&b, q_pe, cos, sin, l);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None);
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps); // [1, l, kv_rank]

        let k_pe = b.reshape(k_pe_raw, vec![1, 1, l, rope_d]);
        let k_pe = rope_interleaved_prefill(&b, k_pe, cos, sin, l); // [1, 1, l, rope_d]

        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_kv, w_kvb, None);
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, l, hq, nope + vd]),
            vec![0, 2, 1, 3],
        );
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let k_pe_b = repeat_kv(&b, k_pe, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        let attn = attention_prefill(&b, q_full, k_full, v_h, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, l, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        // Shared-pool cache write: scatter this call's l latent and RoPE-key positions into the
        // [pool, 1, D] caches via the inverse map, passing other sequences' slots through (as
        // `trace_moe_prefill_kv_shared_pool`, doubled for MLA's two pools). Placed after attention to
        // mirror the decode tracer; the write reads `c_kv`/`k_pe`, so order does not affect dataflow.
        let c_cache = b.state_input(
            &p("mla.c_cache"),
            TensorType::f32(vec![pool, 1, kv_rank]),
            StateRole::Recurrent,
        );
        let rope_cache = b.state_input(
            &p("mla.rope_cache"),
            TensorType::f32(vec![pool, 1, rope_d]),
            StateRole::Recurrent,
        );
        let c_base = b.reshape(c_cache, vec![pool, kv_rank]);
        let c_src = b.reshape(c_kv, vec![l, kv_rank]);
        let c_cache_out = b.reshape(b.scatter_update(c_base, c_src, inv), vec![pool, 1, kv_rank]);
        let rope_base = b.reshape(rope_cache, vec![pool, rope_d]);
        let rope_src = b.reshape(k_pe, vec![l, rope_d]);
        let rope_cache_out = b.reshape(
            b.scatter_update(rope_base, rope_src, inv),
            vec![pool, 1, rope_d],
        );
        state.push((c_cache, c_cache_out));
        state.push((rope_cache, rope_cache_out));

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            deepseek3_moe_ffn(&b, normed, h, &mp, li)
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let last = b.slice(x, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, last, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// DeepSeek-V3 single-token fixed-KV masked decode, the capture/replay analog of
/// [`trace_deepseek3_prefill`], with [`crate::deepseek2::trace_deepseek2_decode_kv_masked`]'s attention
/// block and compressed-cache shape unchanged.
pub fn trace_deepseek3_decode_kv_masked(
    cfg: DeepseekV2Config,
    mp: DeepseekV3MoeParams,
    cap: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();

    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![cap]));
    let mask = b.reshape(mask, vec![1, 1, 1, cap]);

    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));

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

        let (q, _qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let q = b.transpose(b.reshape(q, vec![1, 1, hq, qk_head_dim]), vec![0, 2, 1, 3]);
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_decode(&b, q_pe, cos, sin, pos_slot);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None);
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv_new = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps);
        let c_kv_new = b.reshape(c_kv_new, vec![1, 1, 1, kv_rank]);

        let k_pe_new = b.reshape(k_pe_raw, vec![1, 1, 1, rope_d]);
        let k_pe_new = rope_interleaved_decode(&b, k_pe_new, cos, sin, pos_slot);

        let c_cache = b.state_input(
            &p("mla.c_cache"),
            TensorType::f32(vec![1, 1, cap, kv_rank]),
            StateRole::Recurrent,
        );
        let rope_cache = b.state_input(
            &p("mla.rope_cache"),
            TensorType::f32(vec![1, 1, cap, rope_d]),
            StateRole::Recurrent,
        );
        let c_cache_out = b.dynamic_update_slice_dyn(c_cache, c_kv_new, pos_slot, 2);
        let rope_cache_out = b.dynamic_update_slice_dyn(rope_cache, k_pe_new, pos_slot, 2);
        state.push((c_cache, c_cache_out));
        state.push((rope_cache, rope_cache_out));

        let c_flat = b.reshape(c_cache_out, vec![cap, kv_rank]);
        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_flat, w_kvb, None);
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, cap, hq, nope + vd]),
            vec![0, 2, 1, 3],
        );
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let k_pe_b = repeat_kv(&b, rope_cache_out, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        let attn = attention_masked(&b, q_full, k_full, v_h, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]);
        let attn = b.reshape(attn, vec![1, 1, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            deepseek3_moe_ffn(&b, normed, h, &mp, li)
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// Spec 269 stage 1: the batched, `Slot::SlotMap`-addressed shared-KV-pool analog of
/// [`trace_deepseek3_decode_kv_masked`]: `batch` concurrently-decoding rows, dense
/// DeepSeek-V3, F32 only. Two physical pools per layer (`mla.c_cache` the latent, `mla.rope_cache`
/// the RoPE key), each `[kv_pool_slots, 1, D]`, addressed by the same `[batch, cap]` `Slot::SlotMap`
/// on write (`scatter_shared_pool`) and read (`gather_shared_pool`) with `hkv = 1` (FR-001), reusing
/// [`crate::components::scatter_shared_pool`]/[`crate::components::gather_shared_pool`] with no new `Slot`
/// variant or `OpKind` (FR-002). One map for both caches keeps a position's latent and RoPE key in the
/// same physical row (spec "Internal correctness").
///
/// Every other per-layer step is [`trace_deepseek3_decode_kv_masked`]'s block generalized to `batch`
/// (FR-003, as corrected by the spec's adversarial review): query projection, interleaved RoPE via
/// [`rope_interleaved_decode_batched`] (not a bare `rope_interleaved_decode`, see its docs) at both the
/// query and cached-key sites, `kv_b_proj` over the gathered `[batch, cap, kv_lora_rank]` latent, masked
/// attention. The dense/MoE switch is unchanged (`mp.is_moe_layer`), except the routed branch passes its
/// `[batch, 1, h]` operand to [`deepseek3_moe_ffn`] as `[1, batch, h]`: B independent single-token rows
/// equal one sequence of B tokens (as `crate::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`),
/// needed because `deepseek3_moe_ffn` reads its row count `L` off the second-to-last axis.
pub fn trace_deepseek3_decode_kv_masked_batched_shared_pool(
    cfg: DeepseekV2Config,
    mp: DeepseekV3MoeParams,
    cap: usize,
    batch: usize,
    kv_pool_slots: usize,
) -> Graph {
    let b = Builder::new();
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();

    let token = b.slot(Slot::Token, TensorType::new(vec![batch], DType::I32));
    let pos_slot = b.slot(Slot::Pos, TensorType::new(vec![batch], DType::I32));
    let _seq_len = b.slot(Slot::SeqLen, TensorType::scalar(DType::I32));
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![batch, cap]));
    let mask = b.reshape(mask, vec![batch, 1, 1, cap]);
    let slot_map = b.slot(Slot::SlotMap, TensorType::new(vec![batch, cap], DType::I32));

    let cos = b.constant("rope.cos", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![cfg.max_pos, rope_d / 2]));

    let embed = b.constant(
        "model.embed_tokens.weight",
        TensorType::f32(vec![cfg.vocab, h]),
    );
    let x0 = b.gather(embed, 0, token); // [batch, h]
    let mut x = b.reshape(x0, vec![batch, 1, h]);

    let mut state: Vec<(Traced, Traced)> = Vec::with_capacity(2 * cfg.layers);

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        let ln1 = b.constant(&p("input_layernorm.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln1, cfg.eps);

        let (q, _qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let q = b.transpose(
            b.reshape(q, vec![batch, 1, hq, qk_head_dim]),
            vec![0, 2, 1, 3],
        ); // [batch, Hq, 1, qk_head_dim]
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_decode_batched(&b, q_pe, cos, sin, pos_slot, batch);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None); // [batch, 1, kv_rank+rope_d]
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv_new = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps);
        let c_kv_new = b.reshape(c_kv_new, vec![batch, 1, 1, kv_rank]);

        let k_pe_new = b.reshape(k_pe_raw, vec![batch, 1, 1, rope_d]);
        let k_pe_new = rope_interleaved_decode_batched(&b, k_pe_new, cos, sin, pos_slot, batch);

        let c_cache = b.state_input(
            &p("mla.c_cache"),
            TensorType::f32(vec![kv_pool_slots, 1, kv_rank]),
            StateRole::Recurrent,
        );
        let rope_cache = b.state_input(
            &p("mla.rope_cache"),
            TensorType::f32(vec![kv_pool_slots, 1, rope_d]),
            StateRole::Recurrent,
        );
        let c_cache_out = scatter_shared_pool(&b, c_cache, c_kv_new, pos_slot, slot_map, batch);
        let rope_cache_out =
            scatter_shared_pool(&b, rope_cache, k_pe_new, pos_slot, slot_map, batch);
        state.push((c_cache, c_cache_out));
        state.push((rope_cache, rope_cache_out));

        let c_read = gather_shared_pool(&b, c_cache_out, slot_map, batch, cap, 1, kv_rank); // [batch,1,cap,kv_rank]
        let c_flat = b.reshape(c_read, vec![batch, cap, kv_rank]);
        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_flat, w_kvb, None); // [batch, cap, hq*(nope+vd)]
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![batch, cap, hq, nope + vd]),
            vec![0, 2, 1, 3],
        ); // [batch, hq, cap, nope+vd]
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope);
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd);

        let rope_read = gather_shared_pool(&b, rope_cache_out, slot_map, batch, cap, 1, rope_d); // [batch,1,cap,rope_d]
        let k_pe_b = repeat_kv(&b, rope_read, hq);
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]);

        let attn = attention_masked(&b, q_full, k_full, v_h, 1, scale, mask);
        let attn = b.transpose(attn, vec![0, 2, 1, 3]); // [batch, 1, Hq, vd]
        let attn = b.reshape(attn, vec![batch, 1, hq * vd]);
        let wo = b.constant(
            &p("self_attn.o_proj.weight"),
            TensorType::f32(vec![hq * vd, h]),
        );
        let attn = linear(&b, attn, wo, None);
        x = b.binary(BinOp::Add, x, attn);

        let ln2 = b.constant(
            &p("post_attention_layernorm.weight"),
            TensorType::f32(vec![h]),
        );
        let normed = rmsnorm(&b, x, ln2, cfg.eps);
        let m = if mp.is_moe_layer(li) {
            let normed_flat = b.reshape(normed, vec![1, batch, h]);
            let out = deepseek3_moe_ffn(&b, normed_flat, h, &mp, li); // [1, batch, h]
            b.reshape(out, vec![batch, 1, h])
        } else {
            deepseek2_dense_ffn(&b, normed, h, mp.dense_inter, li)
        };
        x = b.binary(BinOp::Add, x, m);
    }

    let norm = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let x = rmsnorm(&b, x, norm, cfg.eps);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, cfg.vocab]));
    let logits = linear(&b, x, lm_head, None);
    b.finish_with_state(logits, &state)
}

/// One MoE layer's MLP (`li >= first_k_dense_replace`): [`deepseek3_router_gate`]'s routed top-k
/// experts in dense form (every expert evaluated, non-selected zero-weighted) plus
/// `mp.n_shared_experts` shared experts, summed with no gate. Same as
/// `crate::deepseek2::deepseek2_moe_ffn` except for the router and the `e_score_correction_bias`
/// constant.
pub(crate) fn deepseek3_moe_ffn(
    b: &Builder,
    normed: Traced,
    h: usize,
    mp: &DeepseekV3MoeParams,
    li: usize,
) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let e = mp.n_routed_experts;
    let k = mp.top_k;
    let inter = mp.moe_inter;

    let router_w = b.constant(&p("mlp.gate.weight"), TensorType::f32(vec![h, e]));
    let bias = b.constant(
        &p("mlp.gate.e_score_correction_bias"),
        TensorType::f32(vec![e]),
    );
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

    let logits = linear(b, xm, router_w, None); // [L, E]
    let gate = deepseek3_router_gate(
        b,
        logits,
        bias,
        k,
        mp.n_group,
        mp.topk_group,
        mp.routed_scaling_factor,
    ); // [L, E]

    let x_e = b.broadcast(b.reshape(xm, vec![1, l, h]), vec![e, l, h]); // [E,L,H]
    let gu = b.matmul(x_e, w_in); // [E,L,2I]
    let g_part = b.slice(gu, 2, 0, inter);
    let u_part = b.slice(gu, 2, inter, 2 * inter);
    let act = swiglu(b, g_part, u_part);
    let out = b.matmul(act, w_out); // [E,L,H]

    let gate_w = b.reshape(b.transpose(gate, vec![1, 0]), vec![e, l, 1]); // [E,L,1]
    let weighted = b.binary(BinOp::Mul, out, gate_w);
    let weighted_t = b.transpose(weighted, vec![1, 2, 0]); // [L,H,E]
    let routed = b.reduce(RedOp::Sum, weighted_t, 2, false); // [L,H]
    let routed = b.reshape(routed, vec![1, l, h]);

    match deepseek3_shared_expert_ffn(b, normed, h, mp, li) {
        Some(shared) => b.binary(BinOp::Add, routed, shared),
        None => routed,
    }
}

/// The shared-expert branch (`mp.n_shared_experts` combined into one ungated SwiGLU MLP), factored out
/// of [`deepseek3_moe_ffn`]. Shared experts are ordinary dense weights never looked up by expert id. Returns `None` when
/// `mp.n_shared_experts == 0`.
fn deepseek3_shared_expert_ffn(
    b: &Builder,
    normed: Traced,
    h: usize,
    mp: &DeepseekV3MoeParams,
    li: usize,
) -> Option<Traced> {
    if mp.n_shared_experts == 0 {
        return None;
    }
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let shared_inter = mp.moe_inter * mp.n_shared_experts;
    let wsg = b.constant(
        &p("mlp.shared_experts.gate_proj.weight"),
        TensorType::f32(vec![h, shared_inter]),
    );
    let wsu = b.constant(
        &p("mlp.shared_experts.up_proj.weight"),
        TensorType::f32(vec![h, shared_inter]),
    );
    let wsd = b.constant(
        &p("mlp.shared_experts.down_proj.weight"),
        TensorType::f32(vec![shared_inter, h]),
    );
    let sg = linear(b, normed, wsg, None);
    let su = linear(b, normed, wsu, None);
    let sact = swiglu(b, sg, su);
    Some(linear(b, sact, wsd, None))
}

/// Pairwise-`Ge` rank along the trailing axis of `x` (shape `[.., N]`): `rank[i]` is the number of
/// positions strictly greater than `i`, so rank `0` is the maximum. Exact ties share a rank. This helper
/// retains the established DSA/CSA/QSA attention-selection behavior; MoE router call sites use
/// [`poot_graph_ir::ops::stable_descending_rank`] explicitly.
pub(crate) fn pairwise_rank(b: &Builder, x: Traced) -> Traced {
    let shape = b.aval(x).shape;
    let n = *shape
        .last()
        .expect("pairwise_rank input has a trailing axis");
    let last = shape.len() - 1;

    let mut pair_shape = shape.clone();
    pair_shape.push(n);
    let mut p_shape = shape.clone();
    p_shape.push(1);
    let p = b.broadcast(b.reshape(x, p_shape), pair_shape.clone());
    let mut q_shape = shape.clone();
    q_shape[last] = 1;
    q_shape.push(n);
    let q = b.broadcast(b.reshape(x, q_shape), pair_shape);
    let ge = b.binary(BinOp::Ge, p, q);
    let gt = b.binary_scalar(BinOp::Mul, ge, Scalar::F32(-1.0));
    let gt = b.binary_scalar(BinOp::Add, gt, Scalar::F32(1.0));
    b.reduce(RedOp::Sum, gt, last + 1, false)
}

/// `1.0` where `rank < k` (kept), `0.0` otherwise: turns a [`pairwise_rank`] output into a keep mask.
pub(crate) fn keep_top_k_mask(b: &Builder, rank: Traced, k: usize) -> Traced {
    let rge = b.binary_scalar(BinOp::Ge, rank, Scalar::F32(k as f32));
    let keep = b.binary_scalar(BinOp::Mul, rge, Scalar::F32(-1.0));
    b.binary_scalar(BinOp::Add, keep, Scalar::F32(1.0))
}

/// DeepSeek-V3's routed-expert gate (`topk_method: "noaux_tc"`, `scoring_func: "sigmoid"`, with
/// group-limited routing; see the module docs): `p = sigmoid(logits)` (per-expert independent),
/// `sc = p + bias` (selection-only). Group limiting runs first: split `sc`'s expert axis into
/// `n_group` groups of `n_routed_experts/n_group`, sum each group's top-2 `sc` values (`group_scores`,
/// via stable rank within the group axis), rank the groups and keep the `topk_group` best, then push
/// `sc` to `-1e30` for experts outside them. Expert-level top-`k` stably ranks the masked `sc`, keeps
/// the top-`k` by multiplicatively masking the unbiased `p` to zero elsewhere, renormalizes the kept
/// values to sum to `1.0`, and scales by `routed_scaling_factor`. No `exp` is needed.
///
/// `n_group=1` and `topk_group=n_group` are no-ops (the `-1e30` drop is `0.0` everywhere); the
/// CPU-oracle tests check they match ungated top-k.
fn deepseek3_router_gate(
    b: &Builder,
    logits: Traced,
    bias: Traced,
    k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> Traced {
    let (p, rank) = deepseek3_router_rank(b, logits, bias, n_group, topk_group);
    deepseek3_router_weights(b, p, rank, k, routed_scaling_factor)
}

/// [`deepseek3_router_gate`] plus the rank-ordered expert ids, for model-local packed expert dispatch.
///
/// Only callers that consume the ids use this form, so dense DeepSeek-V3 graphs carry no `ArgTopK`.
#[cfg(test)]
pub(crate) fn deepseek3_router_gate_and_ids(
    b: &Builder,
    logits: Traced,
    bias: Traced,
    k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> (Traced, Traced) {
    let (p, rank) = deepseek3_router_rank(b, logits, bias, n_group, topk_group);
    let ids = b.arg_top_k(rank, k);
    (
        deepseek3_router_weights(b, p, rank, k, routed_scaling_factor),
        ids,
    )
}

/// The router's unbiased sigmoid scores and the stable rank of the group-masked biased scores.
fn deepseek3_router_rank(
    b: &Builder,
    logits: Traced,
    bias: Traced,
    n_group: usize,
    topk_group: usize,
) -> (Traced, Traced) {
    let shape = b.aval(logits).shape;
    let e = *shape.last().expect("router logits have an expert axis");
    let last = shape.len() - 1;
    assert!(
        n_group > 0 && e.is_multiple_of(n_group),
        "n_routed_experts ({e}) must be evenly divisible by n_group ({n_group})"
    );
    let group_size = e / n_group;

    let p = sigmoid(b, logits); // [.., E] - the UNBIASED score used for the final weight

    let mut bias_shape = vec![1usize; shape.len()];
    bias_shape[last] = e;
    let bias_r = b.reshape(bias, bias_shape);
    let bias_b = b.broadcast(bias_r, shape.clone());
    let sc = b.binary(BinOp::Add, p, bias_b); // scores_for_choice: selection-only

    // group_scores = sc.view(.., n_group, group_size).topk(2, -1)[0].sum(-1) (MoEGate.forward)
    let mut group_shape = shape.clone();
    group_shape[last] = n_group;
    group_shape.push(group_size); // [.., n_group, group_size]
    let sc_grouped = b.reshape(sc, group_shape.clone());
    let in_group_rank = poot_graph_ir::ops::stable_descending_rank(b, sc_grouped);
    let top2 = poot_graph_ir::ops::top_k_keep_mask(b, in_group_rank, 2);
    let top2_vals = b.binary(BinOp::Mul, sc_grouped, top2);
    let group_scores = b.reduce(RedOp::Sum, top2_vals, last + 1, false); // [.., n_group]

    // group_idx = group_scores.topk(topk_group, -1)[1]; mask scores_for_choice to the selected groups.
    let group_rank = poot_graph_ir::ops::stable_descending_rank(b, group_scores);
    let group_keep = poot_graph_ir::ops::top_k_keep_mask(b, group_rank, topk_group);

    let mut group_keep_shape = shape.clone();
    group_keep_shape[last] = n_group;
    group_keep_shape.push(1);
    let group_keep_r = b.reshape(group_keep, group_keep_shape);
    let group_keep_b = b.broadcast(group_keep_r, group_shape);
    let expert_keep = b.reshape(group_keep_b, shape.clone()); // [.., E], 1.0 iff expert's group is selected

    let not_keep = b.binary_scalar(BinOp::Mul, expert_keep, Scalar::F32(-1.0));
    let not_keep = b.binary_scalar(BinOp::Add, not_keep, Scalar::F32(1.0));
    let group_drop = b.binary_scalar(BinOp::Mul, not_keep, Scalar::F32(-1.0e30));
    let sc_group_masked = b.binary(BinOp::Add, sc, group_drop); // -1e30 outside the selected groups

    // topk(top_k) over the group-masked scores -> topk_idx
    let rank = poot_graph_ir::ops::stable_descending_rank(b, sc_group_masked); // [.., E]
    (p, rank)
}

/// Keep the top-`k` unbiased scores by rank, renormalize them, and apply the routed scaling factor.
fn deepseek3_router_weights(
    b: &Builder,
    p: Traced,
    rank: Traced,
    k: usize,
    routed_scaling_factor: f32,
) -> Traced {
    let last = b.aval(p).shape.len() - 1;
    let keep = poot_graph_ir::ops::top_k_keep_mask(b, rank, k);

    let masked_p = b.binary(BinOp::Mul, p, keep); // UNBIASED score, zeroed outside top-k

    let denom = b.reduce(RedOp::Sum, masked_p, last, true);
    let denom = b.binary_scalar(BinOp::Add, denom, Scalar::F32(1e-20));
    let normed = b.binary(BinOp::Div, masked_p, denom); // renormalized to sum to 1.0 over the kept experts

    b.binary_scalar(BinOp::Mul, normed, Scalar::F32(routed_scaling_factor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek2::DeepseekV2Yarn;
    use poot_graph_ir::Storage;

    fn tiny_cfg(q_lora_rank: Option<usize>) -> DeepseekV2Config {
        DeepseekV2Config {
            vocab: 24,
            hidden: 16,
            layers: 4,
            n_heads: 4,
            q_lora_rank,
            kv_lora_rank: 6,
            qk_nope_head_dim: 5,
            qk_rope_head_dim: 4,
            v_head_dim: 7,
            eps: 1e-5,
            max_pos: 32,
            rope_theta: 10_000.0,
            yarn: None,
        }
    }

    // V3 shape ratios (8 experts, top_k 3, first_k_dense_replace 2 of 4 layers), larger than V2's tiny
    // fixture for a non-trivial top-k. `n_group: 1, topk_group: 1` is the group-limiting no-op, so
    // full-model tests exercise plain top-k as `sigmoid_bias_router_ref` does; group-limited routing
    // has dedicated router-gate tests below.
    fn tiny_mp(first_k_dense_replace: usize, routed_scaling_factor: f32) -> DeepseekV3MoeParams {
        DeepseekV3MoeParams {
            n_routed_experts: 8,
            top_k: 3,
            moe_inter: 6,
            n_shared_experts: 1,
            dense_inter: 10,
            first_k_dense_replace,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor,
        }
    }

    #[test]
    fn deepseek3_prefill_validates_dense_moe_split_and_shared_expert() {
        let cfg = tiny_cfg(Some(3));
        let mp = tiny_mp(2, 2.5);
        let g = trace_deepseek3_prefill(cfg, mp, 5);
        g.validate()
            .expect("deepseek3 prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        let has_suffix = |suf: &str| {
            g.inputs.iter().any(|id| {
                matches!(g.values[*id].storage, Storage::Const)
                    && g.values[*id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with(suf))
            })
        };
        assert!(has_suffix("mlp.gate.e_score_correction_bias"));
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
        assert_eq!(
            router_count, 2,
            "layers 2,3 route (first_k_dense_replace=2)"
        );
        let dense_count = g
            .inputs
            .iter()
            .filter(|id| {
                matches!(g.values[**id].storage, Storage::Const)
                    && g.values[**id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with("mlp.gate_proj.weight"))
            })
            .count();
        assert_eq!(
            dense_count, 2,
            "layers 0,1 are dense (first_k_dense_replace=2)"
        );
    }

    #[test]
    fn deepseek3_decode_kv_masked_validates_compressed_cache_shapes() {
        let cfg = tiny_cfg(Some(3));
        let mp = tiny_mp(2, 2.5);
        let cap = 16;
        let g = trace_deepseek3_decode_kv_masked(cfg, mp, cap);
        g.validate()
            .expect("deepseek3 decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(g.state.len(), 2 * cfg.layers);
        for (i, (si, _so)) in g.state.iter().enumerate() {
            let shape = g.aval(*si).shape.clone();
            if i % 2 == 0 {
                assert_eq!(shape, vec![1, 1, cap, cfg.kv_lora_rank], "c_cache shape");
            } else {
                assert_eq!(
                    shape,
                    vec![1, 1, cap, cfg.qk_rope_head_dim],
                    "rope_cache shape"
                );
            }
        }
    }

    mod cpu_oracle {
        use super::*;
        use crate::deepseek2::deepseek2_rope_tables_interleaved;
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

        /// The correction bias is an additive shift of realistic scale (0.1-1.0 in published
        /// checkpoints), large enough to flip which experts rank in the top-k on the tiny fixture.
        fn bias_weight(name: &str, n: usize) -> Vec<f32> {
            fill(n, seed_of(name)).iter().map(|v| v * 0.5).collect()
        }

        fn all_weights(
            cfg: &DeepseekV2Config,
            mp: &DeepseekV3MoeParams,
        ) -> HashMap<String, Vec<f32>> {
            let h = cfg.hidden;
            let hq = cfg.n_heads;
            let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
            let qk_head_dim = cfg.qk_head_dim();
            let kv_rank = cfg.kv_lora_rank;
            let mut w = HashMap::new();
            w.insert(
                "model.embed_tokens.weight".to_string(),
                weight("embed", cfg.vocab * h, false),
            );
            let (cos, sin) = deepseek2_rope_tables_interleaved(
                cfg.max_pos,
                rope_d,
                cfg.rope_theta,
                cfg.yarn.as_ref(),
            );
            w.insert("rope.cos".to_string(), cos);
            w.insert("rope.sin".to_string(), sin);
            for li in 0..cfg.layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                w.insert(p("input_layernorm.weight"), weight(&p("ln1"), h, true));
                w.insert(
                    p("post_attention_layernorm.weight"),
                    weight(&p("ln2"), h, true),
                );
                match cfg.q_lora_rank {
                    Some(r) => {
                        w.insert(
                            p("self_attn.q_a_proj.weight"),
                            weight(&p("qa"), h * r, false),
                        );
                        w.insert(
                            p("self_attn.q_a_layernorm.weight"),
                            weight(&p("qaln"), r, true),
                        );
                        w.insert(
                            p("self_attn.q_b_proj.weight"),
                            weight(&p("qb"), r * hq * qk_head_dim, false),
                        );
                    }
                    None => {
                        w.insert(
                            p("self_attn.q_proj.weight"),
                            weight(&p("q"), h * hq * qk_head_dim, false),
                        );
                    }
                }
                w.insert(
                    p("self_attn.kv_a_proj_with_mqa.weight"),
                    weight(&p("kva"), h * (kv_rank + rope_d), false),
                );
                w.insert(
                    p("self_attn.kv_a_layernorm.weight"),
                    weight(&p("kvaln"), kv_rank, true),
                );
                w.insert(
                    p("self_attn.kv_b_proj.weight"),
                    weight(&p("kvb"), kv_rank * hq * (nope + vd), false),
                );
                w.insert(
                    p("self_attn.o_proj.weight"),
                    weight(&p("o"), hq * vd * h, false),
                );
                if mp.is_moe_layer(li) {
                    w.insert(
                        p("mlp.gate.weight"),
                        weight(&p("router"), h * mp.n_routed_experts, false),
                    );
                    w.insert(
                        p("mlp.gate.e_score_correction_bias"),
                        bias_weight(&p("bias"), mp.n_routed_experts),
                    );
                    let e = mp.n_routed_experts;
                    let inter = mp.moe_inter;
                    let mut gate_up = vec![0.0f32; e * h * 2 * inter];
                    let mut down = vec![0.0f32; e * inter * h];
                    for ei in 0..e {
                        let g = weight(&p(&format!("e{ei}.gate")), h * inter, false);
                        let u = weight(&p(&format!("e{ei}.up")), h * inter, false);
                        let d_ = weight(&p(&format!("e{ei}.down")), inter * h, false);
                        for row in 0..h {
                            let dst = (ei * h + row) * 2 * inter;
                            gate_up[dst..dst + inter]
                                .copy_from_slice(&g[row * inter..(row + 1) * inter]);
                            gate_up[dst + inter..dst + 2 * inter]
                                .copy_from_slice(&u[row * inter..(row + 1) * inter]);
                        }
                        let dst = ei * inter * h;
                        down[dst..dst + inter * h].copy_from_slice(&d_);
                    }
                    w.insert(p("mlp.experts.gate_up_proj.weight"), gate_up);
                    w.insert(p("mlp.experts.down_proj.weight"), down);
                    if mp.n_shared_experts > 0 {
                        let shared_inter = mp.moe_inter * mp.n_shared_experts;
                        w.insert(
                            p("mlp.shared_experts.gate_proj.weight"),
                            weight(&p("sg"), h * shared_inter, false),
                        );
                        w.insert(
                            p("mlp.shared_experts.up_proj.weight"),
                            weight(&p("su"), h * shared_inter, false),
                        );
                        w.insert(
                            p("mlp.shared_experts.down_proj.weight"),
                            weight(&p("sd"), shared_inter * h, false),
                        );
                    }
                } else {
                    w.insert(
                        p("mlp.gate_proj.weight"),
                        weight(&p("dg"), h * mp.dense_inter, false),
                    );
                    w.insert(
                        p("mlp.up_proj.weight"),
                        weight(&p("du"), h * mp.dense_inter, false),
                    );
                    w.insert(
                        p("mlp.down_proj.weight"),
                        weight(&p("dd"), mp.dense_inter * h, false),
                    );
                }
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

        fn sigmoid_ref(v: f32) -> f32 {
            1.0 / (1.0 + (-v).exp())
        }

        use poot_test_util::linear_ref;

        // Shared with deepseek2/deepseek32 via `crate::reference_ops` (R474-014).
        use crate::reference_ops::rope_interleaved_ref;

        #[allow(clippy::too_many_arguments)]
        fn mla_attend_ref(
            cfg: &DeepseekV2Config,
            q_full: &[Vec<f32>],
            c_kv_cache: &[Vec<f32>],
            k_pe_cache: &[Vec<f32>],
            kv_b_w: &[f32],
            scale: f32,
        ) -> Vec<f32> {
            let (hq, nope, vd, rope_d) = (
                cfg.n_heads,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
                cfg.qk_rope_head_dim,
            );
            let kv_rank = cfg.kv_lora_rank;
            let s = c_kv_cache.len();
            let mut k_nope = vec![vec![vec![0.0f32; nope]; s]; hq];
            let mut v = vec![vec![vec![0.0f32; vd]; s]; hq];
            for (t, c) in c_kv_cache.iter().enumerate() {
                let expanded = linear_ref(c, kv_b_w, kv_rank, hq * (nope + vd));
                for hh in 0..hq {
                    let base = hh * (nope + vd);
                    k_nope[hh][t].copy_from_slice(&expanded[base..base + nope]);
                    v[hh][t].copy_from_slice(&expanded[base + nope..base + nope + vd]);
                }
            }
            let mut out = vec![0.0f32; hq * vd];
            for hh in 0..hq {
                let mut scores = vec![0.0f32; s];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for i in 0..nope {
                        dot += q_full[hh][i] * k_nope[hh][t][i];
                    }
                    for i in 0..rope_d {
                        dot += q_full[hh][nope + i] * k_pe_cache[t][i];
                    }
                    *sc = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                let mut e = vec![0.0f32; s];
                for (t, sc) in scores.iter().enumerate() {
                    e[t] = (sc - m).exp();
                    denom += e[t];
                }
                for i in 0..vd {
                    let mut acc = 0.0f32;
                    for (t, ei) in e.iter().enumerate() {
                        acc += (ei / denom) * v[hh][t][i];
                    }
                    out[hh * vd + i] = acc;
                }
            }
            out
        }

        /// Independent reference for [`deepseek3_router_gate`]: sigmoid, selection-only additive bias,
        /// top-k by the biased score, weight from the unbiased score, renormalize to sum 1.0, scale by
        /// `routed_scaling_factor`.
        fn sigmoid_bias_router_ref(
            logits: &[f32],
            bias: &[f32],
            top_k: usize,
            routed_scaling_factor: f32,
        ) -> Vec<(usize, f32)> {
            let p: Vec<f32> = logits.iter().map(|&v| sigmoid_ref(v)).collect();
            let sc: Vec<f32> = p.iter().zip(bias.iter()).map(|(&a, &b)| a + b).collect();
            let mut idx: Vec<usize> = (0..sc.len()).collect();
            idx.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
            let sel = &idx[..top_k];
            let denom: f32 = sel.iter().map(|&i| p[i]).sum::<f32>() + 1e-20;
            sel.iter()
                .map(|&i| (i, (p[i] / denom) * routed_scaling_factor))
                .collect()
        }

        /// Independent reference for [`deepseek3_router_gate`]'s group-limited form: scoring as
        /// [`sigmoid_bias_router_ref`], with top-k only over experts in the `topk_group` groups with the
        /// highest sum-of-top-2 `scores_for_choice`.
        fn sigmoid_bias_grouped_router_ref(
            logits: &[f32],
            bias: &[f32],
            n_group: usize,
            topk_group: usize,
            top_k: usize,
            routed_scaling_factor: f32,
        ) -> Vec<(usize, f32)> {
            let e = logits.len();
            assert_eq!(e % n_group, 0);
            let group_size = e / n_group;
            let p: Vec<f32> = logits.iter().map(|&v| sigmoid_ref(v)).collect();
            let sc: Vec<f32> = p.iter().zip(bias.iter()).map(|(&a, &b)| a + b).collect();

            let mut group_scores = vec![0.0f32; n_group];
            for (g, gs) in group_scores.iter_mut().enumerate() {
                let mut vals: Vec<f32> = sc[g * group_size..(g + 1) * group_size].to_vec();
                vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
                *gs = vals.iter().take(2).sum();
            }
            let mut group_idx: Vec<usize> = (0..n_group).collect();
            group_idx.sort_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).unwrap());
            let selected_groups: std::collections::HashSet<usize> =
                group_idx[..topk_group].iter().copied().collect();

            let mut idx: Vec<usize> = (0..e)
                .filter(|&i| selected_groups.contains(&(i / group_size)))
                .collect();
            idx.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
            let sel = &idx[..top_k];
            let denom: f32 = sel.iter().map(|&i| p[i]).sum::<f32>() + 1e-20;
            sel.iter()
                .map(|&i| (i, (p[i] / denom) * routed_scaling_factor))
                .collect()
        }

        fn deepseek3_prefill_ref(
            cfg: &DeepseekV2Config,
            mp: &DeepseekV3MoeParams,
            tokens: &[usize],
            w: &HashMap<String, Vec<f32>>,
        ) -> Vec<f32> {
            let h = cfg.hidden;
            let hq = cfg.n_heads;
            let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
            let qk_head_dim = cfg.qk_head_dim();
            let kv_rank = cfg.kv_lora_rank;
            let scale = cfg.attn_scale();
            let l = tokens.len();
            let embed = &w["model.embed_tokens.weight"];
            let cos = &w["rope.cos"];
            let sin = &w["rope.sin"];
            let half = rope_d / 2;

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

                let q_flat: Vec<Vec<f32>> = match cfg.q_lora_rank {
                    Some(r) => {
                        let wqa = &w[&p("self_attn.q_a_proj.weight")];
                        let wqaln = &w[&p("self_attn.q_a_layernorm.weight")];
                        let wqb = &w[&p("self_attn.q_b_proj.weight")];
                        normed
                            .iter()
                            .map(|row| {
                                let qa = linear_ref(row, wqa, h, r);
                                let qa_n = rmsnorm_ref(&qa, wqaln, r, cfg.eps);
                                linear_ref(&qa_n, wqb, r, hq * qk_head_dim)
                            })
                            .collect()
                    }
                    None => {
                        let wq = &w[&p("self_attn.q_proj.weight")];
                        normed
                            .iter()
                            .map(|row| linear_ref(row, wq, h, hq * qk_head_dim))
                            .collect()
                    }
                };
                let mut q_full: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; qk_head_dim]; hq]; l];
                for (pos, qrow) in q_flat.iter().enumerate() {
                    for (hh, q_slot) in q_full[pos].iter_mut().enumerate() {
                        let base = hh * qk_head_dim;
                        let q_nope = &qrow[base..base + nope];
                        let q_pe = &qrow[base + nope..base + qk_head_dim];
                        let q_pe_rot = rope_interleaved_ref(q_pe, cos, sin, pos, half);
                        q_slot[..nope].copy_from_slice(q_nope);
                        q_slot[nope..].copy_from_slice(&q_pe_rot);
                    }
                }

                let wkva = &w[&p("self_attn.kv_a_proj_with_mqa.weight")];
                let wkvaln = &w[&p("self_attn.kv_a_layernorm.weight")];
                let wkvb = &w[&p("self_attn.kv_b_proj.weight")];
                let mut c_kv_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
                let mut k_pe_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
                for (pos, row) in normed.iter().enumerate() {
                    let kva = linear_ref(row, wkva, h, kv_rank + rope_d);
                    let kv_nope_raw = &kva[..kv_rank];
                    let k_pe_raw = &kva[kv_rank..];
                    let c_kv = rmsnorm_ref(kv_nope_raw, wkvaln, kv_rank, cfg.eps);
                    let k_pe_rot = rope_interleaved_ref(k_pe_raw, cos, sin, pos, half);
                    c_kv_cache.push(c_kv);
                    k_pe_cache.push(k_pe_rot);
                }

                let mut attn_out = vec![vec![0.0f32; hq * vd]; l];
                for qpos in 0..l {
                    let ao = mla_attend_ref(
                        cfg,
                        &q_full[qpos],
                        &c_kv_cache[..=qpos],
                        &k_pe_cache[..=qpos],
                        wkvb,
                        scale,
                    );
                    attn_out[qpos] = ao;
                }
                let wo = &w[&p("self_attn.o_proj.weight")];
                for pos in 0..l {
                    let proj = linear_ref(&attn_out[pos], wo, hq * vd, h);
                    for c in 0..h {
                        x[pos][c] += proj[c];
                    }
                }

                let ln2w = &w[&p("post_attention_layernorm.weight")];
                for row in x.iter_mut().take(l) {
                    let normed2 = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let ffn_out = if mp.is_moe_layer(li) {
                        let router_w = &w[&p("mlp.gate.weight")];
                        let bias = &w[&p("mlp.gate.e_score_correction_bias")];
                        let gate_up = &w[&p("mlp.experts.gate_up_proj.weight")];
                        let down = &w[&p("mlp.experts.down_proj.weight")];
                        let logits = linear_ref(&normed2, router_w, h, mp.n_routed_experts);
                        let selected = sigmoid_bias_router_ref(
                            &logits,
                            bias,
                            mp.top_k,
                            mp.routed_scaling_factor,
                        );
                        let inter = mp.moe_inter;
                        let mut moe_out = vec![0.0f32; h];
                        for (e, gate_weight) in selected {
                            let e_gate_up = &gate_up[e * h * 2 * inter..(e + 1) * h * 2 * inter];
                            let mut gate_w = vec![0.0f32; h * inter];
                            let mut up_w = vec![0.0f32; h * inter];
                            for row_i in 0..h {
                                let src = row_i * 2 * inter;
                                gate_w[row_i * inter..(row_i + 1) * inter]
                                    .copy_from_slice(&e_gate_up[src..src + inter]);
                                up_w[row_i * inter..(row_i + 1) * inter]
                                    .copy_from_slice(&e_gate_up[src + inter..src + 2 * inter]);
                            }
                            let e_down = &down[e * inter * h..(e + 1) * inter * h];
                            let gate = linear_ref(&normed2, &gate_w, h, inter);
                            let up = linear_ref(&normed2, &up_w, h, inter);
                            let act: Vec<f32> = gate
                                .iter()
                                .zip(up.iter())
                                .map(|(&g, &u)| silu_ref(g) * u)
                                .collect();
                            let expert_out = linear_ref(&act, e_down, inter, h);
                            for c in 0..h {
                                moe_out[c] += gate_weight * expert_out[c];
                            }
                        }
                        if mp.n_shared_experts > 0 {
                            let shared_inter = mp.moe_inter * mp.n_shared_experts;
                            let wsg = &w[&p("mlp.shared_experts.gate_proj.weight")];
                            let wsu = &w[&p("mlp.shared_experts.up_proj.weight")];
                            let wsd = &w[&p("mlp.shared_experts.down_proj.weight")];
                            let sg = linear_ref(&normed2, wsg, h, shared_inter);
                            let su = linear_ref(&normed2, wsu, h, shared_inter);
                            let sact: Vec<f32> = sg
                                .iter()
                                .zip(su.iter())
                                .map(|(&g, &u)| silu_ref(g) * u)
                                .collect();
                            let shared_out = linear_ref(&sact, wsd, shared_inter, h);
                            for c in 0..h {
                                moe_out[c] += shared_out[c];
                            }
                        }
                        moe_out
                    } else {
                        let wg = &w[&p("mlp.gate_proj.weight")];
                        let wu = &w[&p("mlp.up_proj.weight")];
                        let wd = &w[&p("mlp.down_proj.weight")];
                        let g = linear_ref(&normed2, wg, h, mp.dense_inter);
                        let u = linear_ref(&normed2, wu, h, mp.dense_inter);
                        let act: Vec<f32> = g
                            .iter()
                            .zip(u.iter())
                            .map(|(&gg, &uu)| silu_ref(gg) * uu)
                            .collect();
                        linear_ref(&act, wd, mp.dense_inter, h)
                    };
                    for c in 0..h {
                        row[c] += ffn_out[c];
                    }
                }
            }

            let ln_f_w = &w["model.norm.weight"];
            let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
            let lm_head = &w["lm_head.weight"];
            linear_ref(&last, lm_head, h, cfg.vocab)
        }

        fn eval_deepseek3_prefill(
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

        /// `trace_deepseek3_prefill` matches an independent reference implementing the
        /// sigmoid/bias/renormalize router.
        #[test]
        fn deepseek3_prefill_matches_hand_rolled_reference() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(2, 2.5);
            let tokens = [3usize, 7, 1, 9, 15];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek3_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek3_prefill(&g, &tokens, &weights);
            let want = deepseek3_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Same oracle with `q_lora_rank: None` (V2-Lite's setting), covering `mla_query_proj`'s other
        /// branch with V3's router.
        #[test]
        fn deepseek3_prefill_matches_hand_rolled_reference_no_q_lora() {
            let cfg = tiny_cfg(None);
            let mp = tiny_mp(1, 1.0);
            let tokens = [2usize, 4, 6, 8];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek3_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek3_prefill(&g, &tokens, &weights);
            let want = deepseek3_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }

        /// Changing `e_score_correction_bias` with `mlp.gate.weight` fixed must change which experts
        /// are chosen and so the output: a selection-only bias still matters because it changes which
        /// experts contribute.
        #[test]
        fn deepseek3_correction_bias_is_load_bearing() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(2, 2.5);
            let tokens = [4usize, 6, 8];
            let mut weights_a = all_weights(&cfg, &mp);
            let mut weights_b = weights_a.clone();
            for li in mp.first_k_dense_replace..cfg.layers {
                let key = format!("model.layers.{li}.mlp.gate.e_score_correction_bias");
                let alt: Vec<f32> = weights_a[&key].iter().map(|v| -v).collect();
                weights_b.insert(key, alt);
            }
            // the two bias sets differ
            assert_ne!(
                weights_a["model.layers.2.mlp.gate.e_score_correction_bias"],
                weights_b["model.layers.2.mlp.gate.e_score_correction_bias"]
            );

            let g = trace_deepseek3_prefill(cfg, mp, tokens.len());
            let out_a = eval_deepseek3_prefill(&g, &tokens, &weights_a);
            let want_a = deepseek3_prefill_ref(&cfg, &mp, &tokens, &weights_a);
            assert_matches(&out_a, &want_a, cfg.vocab);
            let want_b = deepseek3_prefill_ref(&cfg, &mp, &tokens, &weights_b);

            let max_diff = poot_test_util::max_abs_error(&want_a, &want_b);
            assert!(
                max_diff > 1e-3,
                "flipping the correction bias should change the reference output; max_diff={max_diff:.2e}"
            );
            // the traced graph must show the same sensitivity when fed weights_b
            weights_a.extend(weights_b);
            let out_b = eval_deepseek3_prefill(&g, &tokens, &weights_a);
            let graph_diff =
                poot_test_util::max_abs_error(out_a.as_f32().unwrap(), out_b.as_f32().unwrap());
            assert!(
                graph_diff > 1e-3,
                "traced graph should be equally sensitive to the bias; graph_diff={graph_diff:.2e}"
            );
        }

        /// The dense gate discards expert ids, so its graph must not carry the id `ArgTopK` that the packed
        /// dispatch form emits; `Builder::finish` does no dead-code elimination.
        #[test]
        fn deepseek3_router_gate_emits_ids_only_for_the_dispatch_form() {
            let count_arg_top_k = |with_ids: bool| {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, 6]));
                let bias = b.constant("bias", TensorType::f32(vec![6]));
                let gate = if with_ids {
                    deepseek3_router_gate_and_ids(&b, logits, bias, 3, 1, 1, 2.5).0
                } else {
                    deepseek3_router_gate(&b, logits, bias, 3, 1, 1, 2.5)
                };
                b.finish(gate)
                    .eqns
                    .iter()
                    .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::ArgTopK { .. }))
                    .count()
            };
            assert_eq!(count_arg_top_k(false), 0);
            assert_eq!(count_arg_top_k(true), 1);
        }

        /// A token's selected gate weights, divided by `routed_scaling_factor`, must sum to `1.0` (f32
        /// tolerance). V2's `crate::deepseek2::deepseek2_router_gate` does not renormalize, so this
        /// guards against reusing its shape for V3.
        #[test]
        fn deepseek3_router_gate_renormalizes_to_routed_scaling_factor() {
            let b = Builder::new();
            let e = 6usize;
            let k = 3usize;
            let scale = 2.5f32;
            let logits_data = [0.3f32, -1.2, 2.1, -0.4, 0.9, -2.5];
            let bias_data = [0.5f32, 0.0, -1.0, 0.2, -0.3, 1.0];
            let logits = b.constant("logits", TensorType::f32(vec![1, e]));
            let bias = b.constant("bias", TensorType::f32(vec![e]));
            let gate = deepseek3_router_gate(&b, logits, bias, k, 1, 1, scale);
            let g = b.finish(gate);

            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let name = meta.name.as_deref().unwrap();
                let data = if name == "logits" {
                    logits_data.to_vec()
                } else {
                    bias_data.to_vec()
                };
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data).into(),
                );
            }
            let got = poot_eval::eval(
                &g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu eval")
            .output
            .into_host()
            .expect("dense output");
            let sum: f32 = got.as_f32().unwrap().iter().sum();
            assert!(
                (sum - scale).abs() < 1e-4,
                "selected gate weights should sum to routed_scaling_factor: got {sum}, want {scale}"
            );
            let nonzero = got
                .as_f32()
                .unwrap()
                .iter()
                .filter(|&&v| v.abs() > 1e-6)
                .count();
            assert_eq!(
                nonzero, k,
                "exactly top_k experts should carry a nonzero gate weight"
            );

            // cross-check against the independent reference at the same inputs
            let want = sigmoid_bias_router_ref(&logits_data, &bias_data, k, scale);
            let mut want_full = vec![0.0f32; e];
            for (i, wv) in want {
                want_full[i] = wv;
            }
            // graph (actual) vs the independent reference (expected)
            poot_test_util::assert_close(got.as_f32().unwrap(), &want_full, 1e-4);
        }

        /// Runs a `[1, E]`-logits / `[E]`-bias router-gate graph and returns the raw gate weights.
        fn eval_router_gate(
            g: &Graph,
            logits_data: &[f32],
            bias_data: &[f32],
        ) -> poot_tensor::HostTensor {
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let name = meta.name.as_deref().unwrap();
                let data = if name == "logits" {
                    logits_data.to_vec()
                } else {
                    bias_data.to_vec()
                };
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data).into(),
                );
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

        /// `n_group=1` and `topk_group == n_group` are group-limiting no-ops and must match plain
        /// top-k bit-for-bit. Uses the `e=6`/`k=3` fixture of
        /// `deepseek3_router_gate_renormalizes_to_routed_scaling_factor` (which pins `n_group=1,
        /// topk_group=1` as its baseline), so the `n_group=6, topk_group=6` variant (six one-expert
        /// groups, all selected) is checked against the same known-good numbers.
        #[test]
        fn deepseek3_router_gate_group_limited_degenerates_to_ungated_when_unrestricted() {
            let e = 6usize;
            let k = 3usize;
            let scale = 2.5f32;
            let logits_data = [0.3f32, -1.2, 2.1, -0.4, 0.9, -2.5];
            let bias_data = [0.5f32, 0.0, -1.0, 0.2, -0.3, 1.0];

            let eval_with = |n_group: usize, topk_group: usize| -> poot_tensor::HostTensor {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, e]));
                let bias = b.constant("bias", TensorType::f32(vec![e]));
                let gate = deepseek3_router_gate(&b, logits, bias, k, n_group, topk_group, scale);
                let g = b.finish(gate);
                eval_router_gate(&g, &logits_data, &bias_data)
            };

            let one_group = eval_with(1, 1); // n_group=1: single group, trivially selected
            let all_groups = eval_with(6, 6); // n_group=6, topk_group=6: six 1-expert groups, all selected

            for (i, (&a, &b2)) in one_group
                .as_f32()
                .unwrap()
                .iter()
                .zip(all_groups.as_f32().unwrap().iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b2.to_bits(),
                    "index {i}: n_group=1 ({a}) and topk_group=n_group=6 ({b2}) should be bit-exact"
                );
            }

            let want = sigmoid_bias_router_ref(&logits_data, &bias_data, k, scale);
            let mut want_full = vec![0.0f32; e];
            for (i, wv) in want {
                want_full[i] = wv;
            }
            // group-limited-but-unrestricted (actual) vs the ungated reference (expected)
            poot_test_util::assert_close(one_group.as_f32().unwrap(), &want_full, 1e-4);
        }

        /// Negative control: with V3's `n_group`/`topk_group` (`8`/`4`), group-limited routing must
        /// change the selection relative to ungated top-k. Expert 10 is the single highest-scoring
        /// expert but its group partner (expert 11) scores near zero, so the group's sum-of-top-2
        /// `scores_for_choice` falls below the `topk_group=4` cut and expert 10 is excluded. A mask
        /// with no effect would make gated and ungated selections identical.
        #[test]
        fn deepseek3_router_gate_group_limited_routing_changes_selection() {
            let e = 16usize;
            let n_group = 8usize;
            let topk_group = 4usize;
            let k = 5usize;
            let scale = 2.5f32;
            #[rustfmt::skip]
            let logits_data = [
                2.0f32, 1.5, // group 0
                -3.0, -2.5,  // group 1
                0.5, 0.3,    // group 2
                -1.0, -1.2,  // group 3
                1.8, 1.6,    // group 4
                3.0, -10.0,  // group 5 - expert 10 individually highest, but group sum is low
                0.9, 0.7,    // group 6
                -2.0, -1.8,  // group 7
            ];
            let bias_data = [0.0f32; 16];

            let build_and_eval = |n_group: usize, topk_group: usize| -> poot_tensor::HostTensor {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, e]));
                let bias = b.constant("bias", TensorType::f32(vec![e]));
                let gate = deepseek3_router_gate(&b, logits, bias, k, n_group, topk_group, scale);
                let g = b.finish(gate);
                eval_router_gate(&g, &logits_data, &bias_data)
            };

            let gated = build_and_eval(n_group, topk_group);
            let ungated = build_and_eval(1, 1);

            let selected = |t: &poot_tensor::HostTensor| -> Vec<usize> {
                t.as_f32()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v.abs() > 1e-6)
                    .map(|(i, _)| i)
                    .collect()
            };
            let gated_sel = selected(&gated);
            let ungated_sel = selected(&ungated);
            assert_ne!(
                gated_sel, ungated_sel,
                "group-limited masking should change the selected expert set (gated {gated_sel:?} vs ungated {ungated_sel:?})"
            );
            assert!(
                ungated_sel.contains(&10),
                "sanity: expert 10 has the highest individual score, ungated top-k must select it"
            );
            assert!(
                !gated_sel.contains(&10),
                "expert 10's group (sum-of-top-2 rank 5, cut is top-4) should be masked out"
            );

            // cross-check against the independent grouped reference
            let want = sigmoid_bias_grouped_router_ref(
                &logits_data,
                &bias_data,
                n_group,
                topk_group,
                k,
                scale,
            );
            let mut want_full = vec![0.0f32; e];
            for (i, wv) in want {
                want_full[i] = wv;
            }
            // graph (actual) vs the grouped reference (expected)
            poot_test_util::assert_close(gated.as_f32().unwrap(), &want_full, 1e-4);
        }

        /// Decode-vs-prefill consistency, as
        /// `crate::deepseek2::tests::cpu_oracle::deepseek2_decode_matches_prefill_at_every_position`,
        /// with V3's router: checks the compressed-latent cache round-trips through
        /// `dynamic_update_slice_dyn` and per-step `kv_b_proj` decompression.
        #[test]
        fn deepseek3_decode_matches_prefill_at_every_position() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(2, 2.5);
            let tokens = [3usize, 7, 1, 9, 15, 2];
            let weights = all_weights(&cfg, &mp);
            let cap = tokens.len();

            let g_decode = trace_deepseek3_decode_kv_masked(cfg, mp, cap);
            let mut caches: Vec<poot_tensor::HostTensor> = g_decode
                .state
                .iter()
                .map(|&(si, _)| poot_tensor::HostTensor::zeros(g_decode.aval(si).shape.clone()))
                .collect();

            for pos in 0..tokens.len() {
                let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
                for &id in &g_decode.inputs {
                    let meta = g_decode.meta(id);
                    let t = match meta.storage {
                        Storage::Slot(Slot::Token) => {
                            poot_tensor::HostTensor::i32(vec![], vec![tokens[pos] as i32])
                        }
                        Storage::Slot(Slot::Pos) => {
                            poot_tensor::HostTensor::i32(vec![], vec![pos as i32])
                        }
                        Storage::Slot(Slot::SeqLen) => {
                            poot_tensor::HostTensor::i32(vec![], vec![(pos + 1) as i32])
                        }
                        Storage::Slot(Slot::Mask) => {
                            let cap_n = meta.aval.shape.iter().product::<usize>();
                            let m: Vec<f32> = (0..cap_n)
                                .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                                .collect();
                            poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                        }
                        Storage::State => continue,
                        Storage::Const => {
                            let name = meta.name.as_deref().expect("const without a name");
                            let data = weights
                                .get(name)
                                .unwrap_or_else(|| panic!("no weight for {name}"));
                            poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data.clone())
                        }
                        other => panic!("unexpected storage {other:?}"),
                    };
                    inputs.insert(id, t.into());
                }
                for (ci, &(si, _)) in g_decode.state.iter().enumerate() {
                    inputs.insert(si, caches[ci].clone().into());
                }
                let step = poot_eval::eval(
                    &g_decode,
                    &inputs,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .expect("decode step eval");
                let logits = step.output.into_host().expect("dense output");
                caches = step
                    .state
                    .into_iter()
                    .map(|v| v.into_host().expect("dense state"))
                    .collect();

                let prefix: Vec<usize> = tokens[..=pos].to_vec();
                let g_prefill = trace_deepseek3_prefill(cfg, mp, prefix.len());
                let prefill_out = eval_deepseek3_prefill(&g_prefill, &prefix, &weights);

                assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
                // decode (actual) vs prefill (expected) at position `pos`
                poot_test_util::assert_close_rel(
                    logits.as_f32().unwrap(),
                    prefill_out.as_f32().unwrap(),
                    1e-3,
                );
            }
        }

        /// Zeroing `n_shared_experts` (`1` on V3) must change the output, as in V2's tests.
        #[test]
        fn deepseek3_shared_expert_is_load_bearing() {
            let cfg = tiny_cfg(Some(3));
            let mp_with = tiny_mp(2, 2.5);
            let mp_without = DeepseekV3MoeParams {
                n_shared_experts: 0,
                ..mp_with
            };
            let tokens = [4usize, 6, 8];
            let weights = all_weights(&cfg, &mp_with);

            let g_with = trace_deepseek3_prefill(cfg, mp_with, tokens.len());
            let g_without = trace_deepseek3_prefill(cfg, mp_without, tokens.len());
            let out_with = eval_deepseek3_prefill(&g_with, &tokens, &weights);
            let out_without = eval_deepseek3_prefill(&g_without, &tokens, &weights);

            let max_diff = poot_test_util::max_abs_error(
                out_with.as_f32().unwrap(),
                out_without.as_f32().unwrap(),
            );
            assert!(
                max_diff > 1e-4,
                "shared-expert branch should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// YaRN engages as in V2 (shared code path) when combined with V3's router.
        #[test]
        fn deepseek3_yarn_prefill_matches_hand_rolled_reference() {
            let yarn = DeepseekV2Yarn {
                factor: 40.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                attention_factor: 1.0,
                softmax_mscale_sq: 1.5896,
            };
            let cfg = DeepseekV2Config {
                yarn: Some(yarn),
                ..tiny_cfg(None)
            };
            let mp = tiny_mp(1, 2.5);
            let tokens = [3usize, 7, 1, 9];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek3_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek3_prefill(&g, &tokens, &weights);
            let want = deepseek3_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }
    }
}
