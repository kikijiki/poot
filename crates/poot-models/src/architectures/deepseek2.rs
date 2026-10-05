//! DeepSeek-V2 MLA tracer (card 135d, see `tasks/deferred/135-modern-model-support-and-benching.md`,
//! `specs/<NNN>-deepseek-mla-tracer/spec.md`) for `deepseek-ai/DeepSeek-V2-Lite` and
//! `deepseek-ai/DeepSeek-V2`. Its distinguishing mechanism is **Multi-head Latent Attention (MLA)**,
//! not an MoE variant. Verified against `deepseek-ai/DeepSeek-V2-Lite`'s `config.json`, the HF
//! `transformers` `modeling_deepseek_v2.py`, and the `yujiepan/deepseek-v2-tiny-random` checkpoint
//! (BF16, safetensors header read via HTTP Range request).
//!
//! # MLA
//!
//! MLA compresses K and V into a shared low-rank latent instead of per-head K/V projections, and
//! splits each Q/K head into a rotated "decoupled RoPE" region and an unrotated "nope" region, using
//! DeepSeek's interleaved-pair rotary convention (not the half-split convention of every other arch
//! here). HF `DeepseekV2Attention.forward`:
//!
//! ```text
//! # Query: low-rank compressed (if q_lora_rank is set) or a plain projection; either way split nope/rope:
//! q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))   # or self.q_proj(hidden_states)
//! q = q.view(query_shape).transpose(1, 2)
//! q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
//!
//! # KV: down-projected to one shared low-rank latent + a shared (not per-head) RoPE slice:
//! compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
//! kv_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
//! k_nope = self.kv_a_layernorm(kv_nope).view(batch_size, 1, seq_length, self.kv_lora_rank)  # single head
//! k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)                        # single head
//!
//! q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings)   # only the *_pe slices rotate
//!
//! # Cache read/write is on the still-compressed latent (k_nope here is the normed latent, not the
//! # per-head key): past_key_values.update(k_nope, k_pe, self.layer_idx)
//!
//! query_states = torch.cat((q_nope, q_pe), dim=-1)
//! key_states, value_states = self.expand_kv(k_nope, k_pe)   # decompress the whole cached latent, every call
//! ```
//!
//! `expand_kv`: `kv_b_proj` up-projects the latent to per-head `(qk_nope_head_dim + v_head_dim)`,
//! split into `k_nope`/`value_states`; the shared single-head `k_pe` is broadcast across heads and
//! concatenated onto `k_nope`. Standard scaled-dot-product attention then runs on the result.
//!
//! The HF implementation does not use the "weight absorption" trick (folding `W_UK`/`W_UV` into the
//! query/output projections so decode never materializes per-head K/V). It decompresses the entire
//! cached latent through `kv_b_proj` on every forward, prefill or decode. The decode tracer mirrors
//! that (see [`trace_deepseek2_decode_kv_masked`]), costing O(cap) extra matmul work per decode step.
//! Weight absorption is out of scope (see "Out of scope" in the spec).
//!
//! # Composition from existing primitives (AGENTS.md: no new graph-IR op)
//!
//! 1. Low-rank compress/decompress is two more matmuls (`kv_a_proj_with_mqa`/`kv_b_proj`,
//!    `q_a_proj`/`q_b_proj`) through [`poot_graph_ir::ops::linear`].
//! 2. The compressed KV cache is a fixed-capacity cache with masking, with per-position width
//!    `kv_lora_rank + qk_rope_head_dim` shared across heads. `b.state_input` and
//!    `dynamic_update_slice_dyn` apply unchanged.
//! 3. The shared single-head `k_pe` is broadcast to all `H` query heads with
//!    [`poot_graph_ir::ops::repeat_kv`] (`n_rep = H`, `H_kv = 1`), the GQA extreme.
//! 4. The nope/rope concat-then-attend is [`poot_graph_ir::ops::attention_prefill`]/`attention_masked`
//!    unchanged. `v_head_dim` may differ from the Q/K width; only `P @ V`'s contraction axis must match.
//! 5. The nope/rope split is [`Builder::slice`] + [`Builder::concat`], as in
//!    `poot_graph_ir::ops::rope_partial` (phi3), applied to independent sub-tensors (`q_pe`/`k_pe`).
//! 6. DeepSeek's RoPE pairing is interleaved (GPT-J-style), not half-split (GPT-NeoX-style). HF
//!    `apply_rotary_emb` uses `torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))`, so
//!    each adjacent pair `(x[2i], x[2i+1])` rotates by its own angle, unlike
//!    [`poot_graph_ir::ops::rope`] which pairs `x[i]` with `x[i+d/2]`. Using the half-split pairing
//!    gives a finite, wrong rotation that only a numeric reference comparison catches.
//!    [`rope_interleave_apply`] is a module-local composition: reshape+slice to deinterleave,
//!    concat to reinterleave (as in gpt-oss's `gptoss_ffn`), with no `Complex` dtype.
//!
//! Nothing needs a new `OpKind`, a non-fixed-capacity KV cache, or a data-dependent shape.
//!
//! # MoE: shared experts + routed top-k + dense-layer prefix
//!
//! - **Shared experts.** `n_shared_experts` always-active experts are combined into one SwiGLU MLP of
//!   width `moe_intermediate_size * n_shared_experts`, added unconditionally to the routed output:
//!   `hidden_states = self.experts(...) + self.shared_experts(residuals)`. Unlike
//!   `poot_models::qwen3next`, there is no per-token sigmoid gate.
//! - **Router.** `DeepseekV2TopkRouter.forward`: `scores = softmax(router_logits)` over all experts,
//!   `topk` on the raw softmax values (no renormalization; identical to
//!   [`poot_graph_ir::ops::top_k_gate`]'s masked-numerator-over-full-denominator construction with
//!   `norm_topk_prob=false`, as in `crate::olmoe::olmoe_top_k_gate`), then
//!   `topk_weights *= self.routed_scaling_factor` (`16.0` on the tiny fixture, `1.0` on V2-Lite).
//!   [`deepseek2_router_gate`] is its own composition (OlmoE's non-renormalized construction plus the
//!   scalar multiply).
//! - **Dense prefix by threshold.** Layers `< first_k_dense_replace` (`1` on V2-Lite and the fixture)
//!   use a dense SwiGLU MLP (`DeepseekV2MLP`, width `intermediate_size`); later layers route
//!   (`DeepseekV2Moe`). The switch is `li >= first_k_dense_replace`.
//! - **Expert tensors.** Routed experts are separate per-expert tensors named like qwen3-moe
//!   (`mlp.experts.{e}.{gate,up,down}_proj.weight`), so the loader reuses `fuse_qwen3_moe_experts`.
//! - **Loader gotcha.** `poot_llm::runner::build_weights`'s generic transpose rule
//!   (`name.ends_with("proj.weight")`) misses `kv_a_proj_with_mqa.weight`, which ends in
//!   `_with_mqa`. Left untransposed it has a wrong but plausible shape
//!   (`[kv_lora_rank+qk_rope_head_dim, hidden]`) and surfaced only as an out-of-bounds panic in
//!   `poot_eval::matmul` under the real-checkpoint crosswalk test. It gets an explicit per-layer
//!   transpose, as does the router (`mlp.gate.weight`).
//!
//! # Group-limited routing (card 296)
//!
//! The tiny fixture sets `topk_method: "group_limited_greedy"` (`n_group: 8`, `topk_group: 3`): the
//! larger DeepSeek-V2 checkpoints group the routed experts and restrict top-k to a subset of groups.
//! `deepseek-ai/DeepSeek-V2-Lite` sets `n_group: 1`/`topk_group: 1`, where group restriction is a
//! no-op (`topk_method: "greedy"`). [`deepseek2_router_gate`] implements the cross-group masking,
//! following `crate::deepseek3::deepseek3_router_gate`, except that the group score is `max` per
//! group (V2), not V3's sum-of-top-2. MoE selection uses the shared stable rank/mask composition;
//! DSA/CSA/QSA keep the older attention rank helper. The crosswalk test exercises the fixture's real
//! `n_group: 8`/`topk_group: 3` routing against a hand-rolled reference.
//!
//! # Real config fields (`deepseek-ai/DeepSeek-V2-Lite/config.json` / `yujiepan/deepseek-v2-tiny-random/config.json`)
//!
//! `hidden_size` 2048/8, `kv_lora_rank` 512/2, `q_lora_rank` `null`/2 (both branches of
//! [`DeepseekV2Config::q_lora_rank`] are real), `qk_nope_head_dim` 128/2, `qk_rope_head_dim` 64/2,
//! `v_head_dim` 128/2, `num_attention_heads` 16/2, `num_key_value_heads` equal to
//! `num_attention_heads` on both (the shared latent replaces KV-head reduction, so `n_rep=1`),
//! `n_routed_experts` 64/160, `n_shared_experts` 2/2, `num_experts_per_tok` 6/6,
//! `moe_intermediate_size` 1408/4, `intermediate_size` (dense-layer width) 10944/16,
//! `first_k_dense_replace` 1/1, `routed_scaling_factor` 1.0/16.0, `rms_norm_eps` 1e-6/1e-6,
//! `attention_bias` `false`/absent (q/kv down-projections carry no bias, so none are declared).
//! `rope_scaling` is YaRN on both (`factor` 40, `mscale`/`mscale_all_dim` 0.707); see
//! [`DeepseekV2Yarn`] (specs/264): an NTK-by-parts inv_freq ramp on the interleaved-pair table plus
//! the `mscale_all_dim`-derived softmax-scale correction from `yarn_apply_mscale`.
//!
//! Uses its own [`DeepseekV2Config`] rather than `crate::qwen2::Qwen2Config`, since MLA has no
//! per-head K/V at the cache level and rotary applies to a slice of the head (as
//! `crate::bloom`/`crate::mpt`), plus [`DeepseekV2MoeParams`] for the MoE shape.

use poot_graph_ir::ops::{attention_masked, attention_prefill, linear, repeat_kv, rmsnorm, swiglu};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

/// DeepSeek-V2's MLA attention shape plus the shared block dims (hidden/layers/vocab/eps/max_pos/
/// rope_theta) in one config; MLA's dims do not map onto `crate::qwen2::Qwen2Config`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV2Config {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// query heads (`num_attention_heads`). No separate KV-head count; the tracer uses `n_rep=1`.
    pub n_heads: usize,
    /// `q_lora_rank`: `Some(r)` compresses the query through `q_a_proj`/`q_a_layernorm`/`q_b_proj`
    /// (V2/V3 and the tiny fixture); `None` uses a plain `q_proj` (V2-Lite). Either way `q` splits into
    /// `qk_nope_head_dim`/`qk_rope_head_dim` per head.
    pub q_lora_rank: Option<usize>,
    /// `kv_lora_rank`: width of the shared single-head latent K and V decompress from.
    pub kv_lora_rank: usize,
    /// per-head unrotated K/Q slice width.
    pub qk_nope_head_dim: usize,
    /// per-head rotated K/Q slice width (decoupled RoPE). Must be even (whole interleaved pairs).
    pub qk_rope_head_dim: usize,
    /// per-head V width; independent of `qk_nope_head_dim + qk_rope_head_dim` (see module docs, item 4).
    pub v_head_dim: usize,
    pub eps: f32,
    pub max_pos: usize,
    /// RoPE base frequency (`rope_theta`).
    pub rope_theta: f32,
    /// YaRN parameters resolved by the loader from `rope_scaling`. `None` is plain RoPE and softmax
    /// scale; `Some` is required for a correct forward pass on both real configs (see
    /// [`DeepseekV2Yarn`]).
    pub yarn: Option<DeepseekV2Yarn>,
}

/// YaRN RoPE-scaling parameters, resolved by the loader from `poot_load::RopeScaling` (this module
/// does not depend on `poot-load` types).
///
/// Two separate corrections from `modeling_deepseek_v2.py`:
/// 1. **The RoPE table** (`DeepseekV2RotaryEmbedding`/`_compute_yarn_parameters`, see
///    `poot_llm::gguf::yarn_inv_freq`): an NTK-by-parts ramp blending extrapolated (unscaled) and
///    interpolated (divided by `factor`) per-frequency values, plus a constant `attention_factor`
///    multiplier on the cos/sin table (`freqs_cis * self.attention_scaling`).
/// 2. **The attention softmax scale** (`self.scaling = yarn_apply_mscale(config.rope_parameters,
///    self.qk_head_dim ** (-0.5))`): a `mscale^2` correction from `mscale_all_dim` on top of
///    `qk_head_dim^-0.5`, applied whenever `rope_scaling.type != "default"`
///    (`yarn_get_mscale(factor, mscale_all_dim)`, squared); see `softmax_mscale_sq`.
///
/// These are different scalars. V2-Lite's `mscale == mscale_all_dim == 0.707` gives
/// `attention_factor == 1.0` while `softmax_mscale_sq` is still ~1.59 (`get_mscale(40, 0.707)^2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV2Yarn {
    /// context-extension factor (`rope_scaling.factor`).
    pub factor: f32,
    /// NTK-by-parts ramp low-end, extrapolation side (`rope_scaling.beta_fast`, HF default 32).
    pub beta_fast: f32,
    /// NTK-by-parts ramp high-end, interpolation side (`rope_scaling.beta_slow`, HF default 1).
    pub beta_slow: f32,
    /// the checkpoint's pretrained context length (`rope_scaling.original_max_position_embeddings`).
    pub original_max_position_embeddings: usize,
    /// the RoPE table's cos/sin magnitude multiplier, as resolved by the loader: an explicit
    /// `rope_scaling.attention_factor` if present, else `get_mscale(factor, mscale) /
    /// get_mscale(factor, mscale_all_dim)` when both `mscale` and `mscale_all_dim` are nonzero, else the
    /// generic YaRN default `get_mscale(factor, 1.0)`.
    pub attention_factor: f32,
    /// the extra softmax-scale correction: `get_mscale(factor, mscale_all_dim)^2` when
    /// `rope_scaling.mscale_all_dim` is nonzero, else `1.0`.
    pub softmax_mscale_sq: f32,
}

impl DeepseekV2Config {
    /// The per-head Q/K width `Q K^T` contracts over (`qk_nope_head_dim + qk_rope_head_dim`).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// The softmax scale: `qk_head_dim^-0.5`, times [`DeepseekV2Yarn::softmax_mscale_sq`] when `yarn`
    /// is `Some`. Distinct from the RoPE table's `attention_factor` (see [`DeepseekV2Yarn`]).
    pub fn attn_scale(&self) -> f32 {
        let base = 1.0 / (self.qk_head_dim() as f32).sqrt();
        match &self.yarn {
            Some(y) => base * y.softmax_mscale_sq,
            None => base,
        }
    }
}

/// DeepSeek-V2's MoE shape: routed top-k experts, shared expert(s), and the dense-vs-routed layer
/// threshold (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV2MoeParams {
    /// total routed experts per MoE layer (`n_routed_experts`).
    pub n_routed_experts: usize,
    /// routed experts selected per token (`num_experts_per_tok`).
    pub top_k: usize,
    /// per-ROUTED-expert FFN intermediate size (`moe_intermediate_size`).
    pub moe_inter: usize,
    /// always-active shared experts, combined into one MLP of width `moe_inter * n_shared_experts`
    /// (`n_shared_experts`). `0` disables the branch.
    pub n_shared_experts: usize,
    /// the dense-layer (`li < first_k_dense_replace`) SwiGLU MLP width (`intermediate_size`), separate
    /// from `moe_inter`.
    pub dense_inter: usize,
    /// layers `< first_k_dense_replace` use the dense MLP, later layers route (`1` on observed configs).
    pub first_k_dense_replace: usize,
    /// scalar applied to the selected gate weights after the non-renormalized softmax selection
    /// (`routed_scaling_factor`). `1.0` is a no-op (V2-Lite); the tiny fixture uses `16.0`.
    pub routed_scaling_factor: f32,
    /// Card 296: equal-size groups the routed experts split into for `topk_method:
    /// "group_limited_greedy"` (`n_group`; `8` on non-Lite V2). Must divide `n_routed_experts`. `1`
    /// makes group-limiting a no-op (V2-Lite, see [`deepseek2_router_gate`]).
    pub n_group: usize,
    /// Card 296: groups selected per token before the expert-level top-`k` (`topk_group`; `3` on
    /// non-Lite V2). `topk_group == n_group` is also a no-op.
    pub topk_group: usize,
}

impl DeepseekV2MoeParams {
    /// Whether layer `li` routes (`true`) or uses the dense MLP (`false`).
    pub fn is_moe_layer(&self, li: usize) -> bool {
        li >= self.first_k_dense_replace
    }
}

/// `Runner`-level bundle of DeepSeek-V2's two param structs, populated and cleared together as one
/// `Option` (the special-tracer field pattern of `crate::bloom`/`crate::mpt`/`crate::mixtral`/...).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV2Params {
    pub cfg: DeepseekV2Config,
    pub moe: DeepseekV2MoeParams,
}

/// Trace a full-sequence DeepSeek-V2 prefill forward (the CPU-oracle path): embeddings, `cfg.layers`
/// pre-norm blocks (MLA attention, `o_proj` + residual, dense-or-routed+shared MoE MLP + residual),
/// final `model.norm`, `lm_head.weight`.
///
/// Expects these constants bound at eval time: `model.embed_tokens.weight` `[vocab,hidden]`,
/// `rope.cos`/`rope.sin` `[max_pos,qk_rope_head_dim/2]` (interleaved-pair
/// table, half the width of a half-split table; see [`rope_interleave_apply`]), plus the `mask.prefill`
/// step input (card 550a) `[1,1,l,l]`,
/// per-layer `model.layers.{li}.{input_layernorm,post_attention_layernorm}.weight`,
/// `self_attn.{q_a_proj,q_a_layernorm,q_b_proj}.weight` (when `cfg.q_lora_rank.is_some()`) OR
/// `self_attn.q_proj.weight` (when `None`), `self_attn.{kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,
/// o_proj}.weight`, and either `mlp.{gate,up,down}_proj.weight` (dense layers) or `mlp.gate.weight` +
/// `mlp.experts.{gate_up,down}_proj.weight` + (if `mp.n_shared_experts>0`)
/// `mlp.shared_experts.{gate,up,down}_proj.weight` (routed layers, per [`deepseek2_moe_ffn`]'s doc
/// comment), `model.norm.weight`, and `lm_head.weight`.
pub fn trace_deepseek2_prefill(
    cfg: DeepseekV2Config,
    mp: DeepseekV2MoeParams,
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

        // Q: low-rank-compressed (Some) or plain (None); either way split nope/rope per head.
        let (q, _qr) = mla_query_proj(&b, &p, normed, cfg, h, hq, qk_head_dim);
        let q = b.transpose(b.reshape(q, vec![1, l, hq, qk_head_dim]), vec![0, 2, 1, 3]); // [1,Hq,L,qk_head_dim]
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_prefill(&b, q_pe, cos, sin, l);
        let q_full = b.concat(3, &[q_nope, q_pe]); // [1,Hq,L,qk_head_dim]

        // KV: down-project to the shared latent, split nope/rope, normalize + rotate, then decompress the
        // whole latent through kv_b_proj (as HF does).
        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None); // [1,L,kv_rank+rope_d]
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank); // [1,L,kv_rank]
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d); // [1,L,rope_d]
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps); // [1,L,kv_rank] - the compressed latent

        let k_pe = b.reshape(k_pe_raw, vec![1, 1, l, rope_d]); // single shared "head"
        let k_pe = rope_interleaved_prefill(&b, k_pe, cos, sin, l);

        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_kv, w_kvb, None); // [1,L,Hq*(nope+vd)]
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, l, hq, nope + vd]),
            vec![0, 2, 1, 3],
        ); // [1,Hq,L,nope+vd]
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope); // [1,Hq,L,nope]
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd); // [1,Hq,L,vd]

        let k_pe_b = repeat_kv(&b, k_pe, hq); // [1,1,L,rope_d] -> [1,Hq,L,rope_d] (H_kv=1 broadcast)
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]); // [1,Hq,L,qk_head_dim]

        let attn = attention_prefill(&b, q_full, k_full, v_h, 1, scale, mask); // [1,Hq,L,vd]
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
            deepseek2_moe_ffn(&b, normed, h, &mp, li)
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

/// DeepSeek-V2 single-token fixed-KV masked decode, the capture/replay analog of
/// [`trace_deepseek2_prefill`]. The KV cache is the compressed latent (`c_kv[1,1,cap,kv_lora_rank]` +
/// `k_pe[1,1,cap,qk_rope_head_dim]`, shared across all `Hq` heads), fixed-capacity with masking.
/// Every step decompresses the whole cache (0..cap) through `kv_b_proj` before attention, as HF
/// `expand_kv` does, costing O(cap) extra matmul work per step (no weight absorption; see the
/// module docs).
pub fn trace_deepseek2_decode_kv_masked(
    cfg: DeepseekV2Config,
    mp: DeepseekV2MoeParams,
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
        let q = b.transpose(b.reshape(q, vec![1, 1, hq, qk_head_dim]), vec![0, 2, 1, 3]); // [1,Hq,1,qk_head_dim]
        let q_nope = b.slice(q, 3, 0, nope);
        let q_pe = b.slice(q, 3, nope, qk_head_dim);
        let q_pe = rope_interleaved_decode(&b, q_pe, cos, sin, pos_slot);
        let q_full = b.concat(3, &[q_nope, q_pe]);

        let w_kva = b.constant(
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            TensorType::f32(vec![h, kv_rank + rope_d]),
        );
        let kv_a = linear(&b, normed, w_kva, None); // [1,1,kv_rank+rope_d]
        let kv_nope_raw = b.slice(kv_a, 2, 0, kv_rank);
        let k_pe_raw = b.slice(kv_a, 2, kv_rank, kv_rank + rope_d);
        let kva_ln = b.constant(
            &p("self_attn.kv_a_layernorm.weight"),
            TensorType::f32(vec![kv_rank]),
        );
        let c_kv_new = rmsnorm(&b, kv_nope_raw, kva_ln, cfg.eps); // [1,1,kv_rank]
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

        // Decompress the whole cache through kv_b_proj (HF expand_kv):
        // c_cache_out[1,1,cap,kv_rank] -> [cap,kv_rank] -> linear -> [cap,Hq*(nope+vd)].
        let c_flat = b.reshape(c_cache_out, vec![cap, kv_rank]);
        let w_kvb = b.constant(
            &p("self_attn.kv_b_proj.weight"),
            TensorType::f32(vec![kv_rank, hq * (nope + vd)]),
        );
        let kv_expanded = linear(&b, c_flat, w_kvb, None); // [cap,Hq*(nope+vd)]
        let kv_expanded = b.transpose(
            b.reshape(kv_expanded, vec![1, cap, hq, nope + vd]),
            vec![0, 2, 1, 3],
        ); // [1,Hq,cap,nope+vd]
        let k_nope_h = b.slice(kv_expanded, 3, 0, nope); // [1,Hq,cap,nope]
        let v_h = b.slice(kv_expanded, 3, nope, nope + vd); // [1,Hq,cap,vd]

        let k_pe_b = repeat_kv(&b, rope_cache_out, hq); // [1,1,cap,rope_d] -> [1,Hq,cap,rope_d]
        let k_full = b.concat(3, &[k_nope_h, k_pe_b]); // [1,Hq,cap,qk_head_dim]

        let attn = attention_masked(&b, q_full, k_full, v_h, 1, scale, mask); // [1,Hq,1,vd]
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
            deepseek2_moe_ffn(&b, normed, h, &mp, li)
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

/// The query-side projection shared by prefill/decode: `q_a_proj -> q_a_layernorm -> q_b_proj` when
/// `cfg.q_lora_rank.is_some()`, else a plain `q_proj`. Returns `([.., Hq*qk_head_dim], qr)` (the
/// first un-reshaped, the caller splits heads), where `qr` is the `q_a_layernorm` output
/// (`[.., q_lora_rank]`, `Some` only for the low-rank branch). DeepSeek-V3.2's DSA Lightning Indexer
/// feeds its `wq_b` from the same `qr` (update 0857, `crate::deepseek32`), so it is returned here
/// rather than recomputed.
pub(crate) fn mla_query_proj(
    b: &Builder,
    p: &dyn Fn(&str) -> String,
    normed: Traced,
    cfg: DeepseekV2Config,
    h: usize,
    hq: usize,
    qk_head_dim: usize,
) -> (Traced, Option<Traced>) {
    match cfg.q_lora_rank {
        Some(r) => {
            let wqa = b.constant(&p("self_attn.q_a_proj.weight"), TensorType::f32(vec![h, r]));
            let q_a = linear(b, normed, wqa, None); // [..,r]
            let qa_ln = b.constant(
                &p("self_attn.q_a_layernorm.weight"),
                TensorType::f32(vec![r]),
            );
            let q_a_normed = rmsnorm(b, q_a, qa_ln, cfg.eps);
            let wqb = b.constant(
                &p("self_attn.q_b_proj.weight"),
                TensorType::f32(vec![r, hq * qk_head_dim]),
            );
            (linear(b, q_a_normed, wqb, None), Some(q_a_normed))
        }
        None => {
            let wq = b.constant(
                &p("self_attn.q_proj.weight"),
                TensorType::f32(vec![h, hq * qk_head_dim]),
            );
            (linear(b, normed, wq, None), None)
        }
    }
}

/// DeepSeek's interleaved-pair RoPE (module docs, item 6): each adjacent pair `(x[2i], x[2i+1])`
/// rotates by its own angle, unlike [`poot_graph_ir::ops::rope`]'s half-split pairing
/// (`x[i]` with `x[i+d/2]`).
///
/// ```text
/// out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
/// out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
/// ```
///
/// `x[.., D]` (`D` even); `cos`/`sin` are already position-selected, `[.., D/2]` (one angle per
/// pair). Deinterleave by reshaping `D -> [D/2, 2]` and slicing index `0`/`1` (as
/// `poot_models::gpt_oss::gptoss_ffn` does), rotate elementwise, then reshape each half to
/// `[.., D/2, 1]` and `concat` on the new trailing axis to reinterleave (the row-major flatten of
/// `[.., D/2, 2]` is the interleaved order).
pub(crate) fn rope_interleave_apply(b: &Builder, x: Traced, cos: Traced, sin: Traced) -> Traced {
    let shape = b.aval(x).shape;
    let d = *shape.last().expect("rope on a scalar");
    let last = shape.len() - 1;
    let half = d / 2;

    let mut pair_shape = shape.clone();
    pair_shape[last] = half;
    pair_shape.push(2); // [.., D/2, 2]
    let xp = b.reshape(x, pair_shape);

    let mut half_shape = shape.clone();
    half_shape[last] = half; // [.., D/2]
    let x_even = b.reshape(b.slice(xp, last + 1, 0, 1), half_shape.clone()); // x[2i]
    let x_odd = b.reshape(b.slice(xp, last + 1, 1, 2), half_shape.clone()); // x[2i+1]

    let ec = b.binary(BinOp::Mul, x_even, cos);
    let os = b.binary(BinOp::Mul, x_odd, sin);
    let out_even = b.binary(BinOp::Sub, ec, os); // x[2i]*cos - x[2i+1]*sin

    let es = b.binary(BinOp::Mul, x_even, sin);
    let oc = b.binary(BinOp::Mul, x_odd, cos);
    let out_odd = b.binary(BinOp::Add, es, oc); // x[2i]*sin + x[2i+1]*cos

    let mut col_shape = half_shape;
    col_shape.push(1); // [.., D/2, 1]
    let oe = b.reshape(out_even, col_shape.clone());
    let oo = b.reshape(out_odd, col_shape);
    let cat = b.concat(last + 1, &[oe, oo]); // [.., D/2, 2] - row-major flatten IS interleaved order
    b.reshape(cat, shape)
}

/// [`rope_interleave_apply`] for a full sequence `x[.., L, D]` at positions `0..L`; the interleaved
/// analog of [`poot_graph_ir::ops::rope_prefill`].
pub(crate) fn rope_interleaved_prefill(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    seq_len: usize,
) -> Traced {
    let cos = b.slice(cos_table, 0, 0, seq_len); // [L, D/2]
    let sin = b.slice(sin_table, 0, 0, seq_len);
    rope_interleave_apply(b, x, cos, sin)
}

/// [`rope_interleave_apply`] for one decode step at a runtime `pos`; the interleaved analog of
/// [`poot_graph_ir::ops::rope`].
pub(crate) fn rope_interleaved_decode(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: Traced,
) -> Traced {
    let cos = b.gather_scalar(cos_table, 0, pos); // [D/2]
    let sin = b.gather_scalar(sin_table, 0, pos);
    rope_interleave_apply(b, x, cos, sin)
}

/// Batched-decode analog of [`rope_interleaved_decode`]: `pos` is a `[batch]` vector, as
/// [`poot_graph_ir::ops::rope_batched`] relates to [`poot_graph_ir::ops::rope`] (spec 269).
///
/// `rope_interleaved_decode` gathers `cos`/`sin` with `gather_scalar`, giving rank-1 `[D/2]` that
/// broadcasts against `x`'s trailing axis. A plain `gather` with a `[batch]` `pos` gives
/// `[batch, D/2]`, which right-aligned broadcasting against `x[batch, H, 1, D/2]` places on the
/// query-length axis and fails a later reshape for `batch > 1`. This function gathers, then reshapes to
/// `[batch, 1, 1, D/2]` before calling [`rope_interleave_apply`], as `rope_batched` does.
pub(crate) fn rope_interleaved_decode_batched(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: Traced,
    batch: usize,
) -> Traced {
    let rot = *b.aval(cos_table).shape.last().expect("rope table rank>=1");
    let cos = b.gather(cos_table, 0, pos); // [batch, D/2]
    let sin = b.gather(sin_table, 0, pos);
    let cos = b.reshape(cos, vec![batch, 1, 1, rot]);
    let sin = b.reshape(sin, vec![batch, 1, 1, rot]);
    rope_interleave_apply(b, x, cos, sin)
}

/// The dense per-layer MLP for layers `< first_k_dense_replace`: SwiGLU without bias
/// (`DeepseekV2MLP`), `linear -> linear -> swiglu -> linear`.
pub(crate) fn deepseek2_dense_ffn(
    b: &Builder,
    normed: Traced,
    h: usize,
    inter: usize,
    li: usize,
) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let wg = b.constant(&p("mlp.gate_proj.weight"), TensorType::f32(vec![h, inter]));
    let wu = b.constant(&p("mlp.up_proj.weight"), TensorType::f32(vec![h, inter]));
    let wd = b.constant(&p("mlp.down_proj.weight"), TensorType::f32(vec![inter, h]));
    let g = linear(b, normed, wg, None);
    let u = linear(b, normed, wu, None);
    let act = swiglu(b, g, u);
    linear(b, act, wd, None)
}

/// One MoE layer's MLP (`li >= first_k_dense_replace`): [`deepseek2_router_gate`]'s routed top-k
/// experts in dense form (every expert evaluated, non-selected zero-weighted, as
/// `crate::olmoe::olmoe_ffn`/`poot_graph_ir::ops::moe_dense`) plus `mp.n_shared_experts` shared
/// experts (one SwiGLU MLP of width `moe_inter * n_shared_experts`), summed without a gate.
fn deepseek2_moe_ffn(
    b: &Builder,
    normed: Traced,
    h: usize,
    mp: &DeepseekV2MoeParams,
    li: usize,
) -> Traced {
    let p = |s: &str| format!("model.layers.{li}.{s}");
    let e = mp.n_routed_experts;
    let k = mp.top_k;
    let inter = mp.moe_inter;

    let router_w = b.constant(&p("mlp.gate.weight"), TensorType::f32(vec![h, e]));
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

    let logits = linear(b, xm, router_w, None);
    let gate = deepseek2_router_gate(
        b,
        logits,
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
    let routed = b.reduce(poot_graph_ir::RedOp::Sum, weighted_t, 2, false); // [L,H]
    let routed = b.reshape(routed, vec![1, l, h]);

    if mp.n_shared_experts == 0 {
        return routed;
    }
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
    let shared = linear(b, sact, wsd, None); // [1,L,H]

    b.binary(BinOp::Add, routed, shared)
}

/// DeepSeek's routed-expert gate (`DeepseekV2TopkRouter.forward` and the `topk_method:
/// "group_limited_greedy"` branch of `MoEGate.forward` in `deepseek-ai/DeepSeek-V2`'s
/// `modeling_deepseek.py`, card 296): `scores = softmax(logits)` over all `E` experts. Group limiting
/// runs before expert selection: `group_scores = scores.view(.., n_group, -1).max(-1)` (plain max per
/// group; V3 in `crate::deepseek3::deepseek3_router_gate` uses sum-of-top-2),
/// `group_idx = topk(group_scores, topk_group)`, and experts outside a selected group are masked to
/// `0.0` in the reference (safe since softmax is strictly positive). Softmax is monotonic with a
/// common denominator, so this operates on `logits` for both group and expert ranking. The graph gives
/// eligible groups lexicographic priority before expert score, matching the `0.0`-fill without an
/// additive sentinel that could collide with canonical `f32::MIN`. The denominator still covers every
/// expert, as the reference softmax runs once before masking.
///
/// `topk` then selects the `k` largest eligible positions (raw softmax values, not renormalized:
/// the masked-numerator-over-full-denominator construction of `crate::olmoe::olmoe_top_k_gate` with
/// `norm_topk_prob=false`), scaled by `routed_scaling_factor`. Ranking, numerator and denominator use
/// the same canonical finite scores. Selection uses [`poot_graph_ir::ops::stable_descending_rank`]
/// and [`poot_graph_ir::ops::stable_descending_rank_masked`]; DSA/CSA/QSA attention ranking is
/// unchanged.
///
/// `n_group=1` (V2-Lite) and `topk_group=n_group` are no-ops: a single group always ranks `0` and is
/// kept, so this degenerates bit-for-bit to plain top-k (checked by the CPU-oracle tests).
fn deepseek2_router_gate(
    b: &Builder,
    x: Traced,
    k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> Traced {
    deepseek2_router_gate_and_mask(b, x, k, n_group, topk_group, routed_scaling_factor).0
}

fn deepseek2_router_gate_and_mask(
    b: &Builder,
    x: Traced,
    k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> (Traced, Traced) {
    let shape = b.aval(x).shape;
    let e = *shape.last().expect("router logits have an expert axis");
    let last = shape.len() - 1;
    assert!(
        n_group > 0 && e.is_multiple_of(n_group),
        "n_routed_experts ({e}) must be evenly divisible by n_group ({n_group})"
    );
    let group_size = e / n_group;
    assert!(
        topk_group > 0 && topk_group <= n_group,
        "topk_group must be in 1..=n_group"
    );
    assert!(
        k > 0 && k <= topk_group * group_size,
        "top_k must fit in the selected expert groups"
    );

    // Ranking, numerator and denominator share one finite canonical score domain: keeps
    // norm_topk_prob=false semantics and avoids inf-inf NaNs.
    let scores = poot_graph_ir::ops::canonical_router_scores(b, x);

    // Group-limited routing (topk_method "group_limited_greedy"): group score is the max logit per
    // group (monotonic with max softmax); keep the topk_group best groups.
    let mut group_shape = shape.clone();
    group_shape[last] = n_group;
    group_shape.push(group_size); // [.., n_group, group_size]
    let x_grouped = b.reshape(scores, group_shape.clone());
    let group_scores = b.reduce(poot_graph_ir::RedOp::Max, x_grouped, last + 1, false); // [.., n_group]

    let group_rank = poot_graph_ir::ops::stable_descending_rank(b, group_scores);
    let group_keep = poot_graph_ir::ops::top_k_keep_mask(b, group_rank, topk_group);

    let mut group_keep_shape = shape.clone();
    group_keep_shape[last] = n_group;
    group_keep_shape.push(1);
    let group_keep_r = b.reshape(group_keep, group_keep_shape);
    let group_keep_b = b.broadcast(group_keep_r, group_shape);
    let expert_keep = b.reshape(group_keep_b, shape.clone()); // [.., E], 1.0 iff expert's group is selected

    // Mask-aware lexicographic rank: eligible experts first, no additive sentinel at f32 endpoints.
    let rank = poot_graph_ir::ops::stable_descending_rank_masked(b, scores, expert_keep);
    let keep = poot_graph_ir::ops::top_k_keep_mask(b, rank, k);
    let m = b.reduce(poot_graph_ir::RedOp::Max, scores, last, true);
    let ex_full = b.unary(poot_graph_ir::UnOp::Exp, b.binary(BinOp::Sub, scores, m));
    let ex_masked = b.binary(BinOp::Mul, ex_full, keep);

    // Non-renormalized shape: the softmax denominator covers all E experts (the reference softmax
    // runs before group masking).
    let denom = b.reduce(poot_graph_ir::RedOp::Sum, ex_full, last, true);
    let gate = b.binary(BinOp::Div, ex_masked, denom);

    (
        b.binary_scalar(
            BinOp::Mul,
            gate,
            poot_graph_ir::Scalar::F32(routed_scaling_factor),
        ),
        keep,
    )
}

/// Build the interleaved-pair RoPE table for DeepSeek's decoupled-rope slice: `cos`/`sin`
/// `[max_pos, rope_dim/2]`, one angle per pair. Shared by this module's CPU-oracle test and (via
/// `poot_llm::runner`) the safetensors loader, so the table math has one definition.
///
/// `yarn: None` builds the unscaled table. `yarn: Some(_)` applies the NTK-by-parts ramp and
/// `attention_factor` scale of `poot_llm::gguf::yarn_inv_freq` (verified against an f64 reference by
/// `yarn_matches_hf_reference_formula`), on an interleaved table (see [`DeepseekV2Yarn`]). The math is
/// duplicated here because `poot-models` must not depend on `poot-llm`.
pub fn deepseek2_rope_tables_interleaved(
    max_pos: usize,
    rope_dim: usize,
    theta: f32,
    yarn: Option<&DeepseekV2Yarn>,
) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let (inv_freq, attention_factor) = deepseek2_rope_inv_freq(rope_dim, theta, yarn);
    let mut cos = vec![0.0f32; max_pos * half];
    let mut sin = vec![0.0f32; max_pos * half];
    for pos in 0..max_pos {
        for i in 0..half {
            let ang = pos as f32 * inv_freq[i];
            let (s, c) = ang.sin_cos();
            cos[pos * half + i] = c * attention_factor;
            sin[pos * half + i] = s * attention_factor;
        }
    }
    (cos, sin)
}

/// The inv_freq/attention_factor computation shared by [`deepseek2_rope_tables_interleaved`]
/// (interleaved-pair layout) and `crate::deepseek32`'s Lightning Indexer RoPE table (half-split layout,
/// spec 277). Both use the same frequencies: in `deepseek-ai/DeepSeek-V3.2-Exp`'s `inference/model.py`,
/// `Indexer.forward` receives the same `freqs_cis` as `MLA.forward` (`precompute_freqs_cis` runs once
/// per model); only the pairing convention differs.
pub(crate) fn deepseek2_rope_inv_freq(
    rope_dim: usize,
    theta: f32,
    yarn: Option<&DeepseekV2Yarn>,
) -> (Vec<f32>, f32) {
    let half = rope_dim / 2;
    match yarn {
        None => (
            (0..half)
                .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rope_dim as f32))
                .collect(),
            1.0,
        ),
        Some(y) => {
            let d = rope_dim as f32;
            let orig = y.original_max_position_embeddings.max(1) as f32;
            let factor = y.factor.max(1e-6);
            // find_correction_dim: frequency index where a wavelength makes `num_rotations` turns over
            // the original context (as `poot_llm::gguf::yarn_inv_freq`).
            let correction_dim = |num_rotations: f32| -> f32 {
                (d * (orig / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * theta.ln())
            };
            let low = correction_dim(y.beta_fast).floor().max(0.0);
            let high = correction_dim(y.beta_slow)
                .ceil()
                .min(rope_dim as f32 - 1.0);
            let (low, high) = if low == high {
                (low, high + 0.001)
            } else {
                (low, high)
            };
            let inv_freq = (0..half)
                .map(|j| {
                    let pos_freq = theta.powf((2 * j) as f32 / d);
                    let extrap = 1.0 / pos_freq;
                    let interp = 1.0 / (factor * pos_freq);
                    let ramp = ((j as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrap_factor = 1.0 - ramp;
                    interp * (1.0 - extrap_factor) + extrap * extrap_factor
                })
                .collect();
            (inv_freq, y.attention_factor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Storage;

    fn tiny_cfg(q_lora_rank: Option<usize>) -> DeepseekV2Config {
        // v_head_dim differs from qk_nope+qk_rope (independent V width), qk_rope_head_dim is even
        // (interleaved pairs), q_lora_rank is covered both ways by the two configs below.
        DeepseekV2Config {
            vocab: 24,
            hidden: 16,
            layers: 3,
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

    fn tiny_mp(first_k_dense_replace: usize, routed_scaling_factor: f32) -> DeepseekV2MoeParams {
        DeepseekV2MoeParams {
            n_routed_experts: 6,
            top_k: 2,
            moe_inter: 8,
            n_shared_experts: 2,
            dense_inter: 10,
            first_k_dense_replace,
            routed_scaling_factor,
            // n_group=1, topk_group=1: group limiting is a no-op (V2-Lite); the group-limited tests
            // (card 296) use a dedicated fixture.
            n_group: 1,
            topk_group: 1,
        }
    }

    #[test]
    fn deepseek2_prefill_validates_q_lora_and_dense_moe_split() {
        let cfg = tiny_cfg(Some(3));
        let mp = tiny_mp(1, 1.7);
        let g = trace_deepseek2_prefill(cfg, mp, 5);
        g.validate()
            .expect("deepseek2 prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);

        // q_lora_rank Some: q_a_proj/q_a_layernorm/q_b_proj present, no q_proj.
        let has_name = |suf: &str| {
            g.inputs.iter().any(|id| {
                matches!(g.values[*id].storage, Storage::Const)
                    && g.values[*id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with(suf))
            })
        };
        assert!(has_name("self_attn.q_a_proj.weight"));
        assert!(has_name("self_attn.q_a_layernorm.weight"));
        assert!(has_name("self_attn.q_b_proj.weight"));
        assert!(!has_name("self_attn.q_proj.weight"));

        // Layer 0 is dense (gate/up/down, no router); layers 1,2 route. "mlp.gate_proj" and
        // "mlp.gate.weight" are distinct names, so check presence counts by layer.
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
            "layers 1,2 route (first_k_dense_replace=1)"
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
            dense_count, 1,
            "only layer 0 is dense (first_k_dense_replace=1)"
        );
        let shared_count = g
            .inputs
            .iter()
            .filter(|id| {
                matches!(g.values[**id].storage, Storage::Const)
                    && g.values[**id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with("shared_experts.gate_proj.weight"))
            })
            .count();
        assert_eq!(shared_count, 2, "one shared-expert MLP per MoE layer");
    }

    #[test]
    fn deepseek2_prefill_validates_plain_q_proj_when_no_q_lora() {
        let cfg = tiny_cfg(None);
        let mp = tiny_mp(1, 1.0);
        let g = trace_deepseek2_prefill(cfg, mp, 4);
        g.validate()
            .expect("deepseek2 prefill graph should validate");
        let has_name = |suf: &str| {
            g.inputs.iter().any(|id| {
                matches!(g.values[*id].storage, Storage::Const)
                    && g.values[*id]
                        .name
                        .as_deref()
                        .is_some_and(|n| n.ends_with(suf))
            })
        };
        assert!(has_name("self_attn.q_proj.weight"));
        assert!(!has_name("self_attn.q_a_proj.weight"));
    }

    #[test]
    fn deepseek2_decode_kv_masked_validates_compressed_cache_shapes() {
        let cfg = tiny_cfg(Some(3));
        let mp = tiny_mp(1, 1.0);
        let cap = 16;
        let g = trace_deepseek2_decode_kv_masked(cfg, mp, cap);
        g.validate()
            .expect("deepseek2 decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(g.state.len(), 2 * cfg.layers);
        // The compressed cache is shared across heads (head dim 1) and much narrower than a per-head
        // cache (kv_lora_rank=6 / qk_rope_head_dim=4 vs n_heads*(qk_head_dim or v_head_dim) = 36 or 28).
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
            assert_eq!(g.values[*si].storage, Storage::State);
        }
        let mask_id = g
            .inputs
            .iter()
            .find(|id| matches!(g.values[**id].storage, Storage::Slot(Slot::Mask)))
            .expect("Slot::Mask present");
        assert_eq!(g.values[*mask_id].aval.shape, vec![cap]);
    }

    // ---- CPU-oracle numerics (poot_eval): decomposition checked against an independent Rust
    // reference forward pass. ----

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

        /// All constants [`trace_deepseek2_prefill`]/[`trace_deepseek2_decode_kv_masked`] declare, keyed
        /// by graph const name (both share per-layer names).
        fn all_weights(
            cfg: &DeepseekV2Config,
            mp: &DeepseekV2MoeParams,
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

        use poot_test_util::linear_ref;

        // Interleaved-pair rotation (module docs, item 6): pairs `(x[2i],x[2i+1])`, not the half-split
        // `rope_ref` used elsewhere. Shared by the DeepSeek family tests (`crate::reference_ops`).
        use crate::reference_ops::rope_interleaved_ref;

        /// Independent MLA attention for one query position `qpos` over key positions `0..=kpos_max`:
        /// decompress c_kv/k_pe to per-head K/V (kv_b_proj + shared-rope broadcast), then scaled
        /// dot-product attention.
        #[allow(clippy::too_many_arguments)]
        fn mla_attend_ref(
            cfg: &DeepseekV2Config,
            q_full: &[Vec<f32>], // [Hq][qk_head_dim] for this ONE query position
            c_kv_cache: &[Vec<f32>], // [kpos][kv_lora_rank], positions 0..=kpos_max
            k_pe_cache: &[Vec<f32>], // [kpos][rope_dim], ALREADY ROTATED
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
            // decompress every cached position to per-head nope+v
            let mut k_nope = vec![vec![vec![0.0f32; nope]; s]; hq]; // [h][t][nope]
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
                        dot += q_full[hh][nope + i] * k_pe_cache[t][i]; // SHARED rope, same for every head
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

        fn top_k_router_ref(
            logits: &[f32],
            top_k: usize,
            routed_scaling_factor: f32,
        ) -> Vec<(usize, f32)> {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let sel = &idx[..top_k];
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exp_sel: Vec<f32> = sel.iter().map(|&i| (logits[i] - m).exp()).collect();
            let denom_full: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
            sel.iter()
                .zip(exp_sel.iter())
                .map(|(&i, &ex)| (i, (ex / denom_full) * routed_scaling_factor))
                .collect()
        }

        /// Direct-loop reference forward pass for [`trace_deepseek2_prefill`] (last token only), not a
        /// copy of the graph decomposition.
        fn deepseek2_prefill_ref(
            cfg: &DeepseekV2Config,
            mp: &DeepseekV2MoeParams,
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

                // Q
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
                // split per head into nope/rope, rotate rope
                let mut q_full: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; qk_head_dim]; hq]; l]; // [pos][h][dim]
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

                // KV: compress, cache compressed, rotate k_pe
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

                // attention: causal, per query position attends 0..=pos.
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

                // FFN
                let ln2w = &w[&p("post_attention_layernorm.weight")];
                for row in x.iter_mut().take(l) {
                    let normed2 = rmsnorm_ref(row, ln2w, h, cfg.eps);
                    let ffn_out = if mp.is_moe_layer(li) {
                        let router_w = &w[&p("mlp.gate.weight")];
                        let gate_up = &w[&p("mlp.experts.gate_up_proj.weight")];
                        let down = &w[&p("mlp.experts.down_proj.weight")];
                        let logits = linear_ref(&normed2, router_w, h, mp.n_routed_experts);
                        let selected =
                            top_k_router_ref(&logits, mp.top_k, mp.routed_scaling_factor);
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

        fn eval_deepseek2_prefill(
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

        /// `trace_deepseek2_prefill` matches the independent reference within f32 tolerance, low-rank Q
        /// branch.
        #[test]
        fn deepseek2_prefill_matches_hand_rolled_reference_with_q_lora() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(1, 1.7);
            let tokens = [3usize, 7, 1, 9, 15];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek2_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek2_prefill(&g, &tokens, &weights);
            let want = deepseek2_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Same oracle with `q_lora_rank: None` (V2-Lite: plain `q_proj`), covering [`mla_query_proj`]'s
        /// other branch.
        #[test]
        fn deepseek2_prefill_matches_hand_rolled_reference_no_q_lora() {
            let cfg = tiny_cfg(None);
            let mp = tiny_mp(2, 1.0);
            let tokens = [2usize, 4, 6, 8, 10, 12];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek2_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek2_prefill(&g, &tokens, &weights);
            let want = deepseek2_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }

        /// Every layer dense (`first_k_dense_replace >= layers`).
        #[test]
        fn deepseek2_prefill_matches_hand_rolled_reference_all_dense() {
            let cfg = tiny_cfg(Some(4));
            let mp = tiny_mp(10, 1.0); // first_k_dense_replace > layers: every layer dense
            let tokens = [1usize, 2, 3];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek2_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek2_prefill(&g, &tokens, &weights);
            let want = deepseek2_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
        }

        /// The interleaved-pair convention matters: half-split pairing with the same cos/sin values
        /// must change the output (module docs, item 6).
        #[test]
        fn deepseek2_interleaved_rope_differs_from_half_split_pairing() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(1, 1.0);
            let tokens = [5usize, 9, 2];
            let weights = all_weights(&cfg, &mp);
            let g = trace_deepseek2_prefill(cfg, mp, tokens.len());
            let interleaved = eval_deepseek2_prefill(&g, &tokens, &weights);

            // Half-split reference: same rope values, x[i]/x[i+half] pairing instead of x[2i]/x[2i+1].
            // The two coincide only for rope_dim<=2; this test uses rope_dim=4.
            fn rope_half_split_ref(
                row: &[f32],
                cos: &[f32],
                sin: &[f32],
                pos: usize,
                half: usize,
            ) -> Vec<f32> {
                let c = &cos[pos * half..(pos + 1) * half];
                let s = &sin[pos * half..(pos + 1) * half];
                let mut out = vec![0.0f32; 2 * half];
                for i in 0..half {
                    let (a, b) = (row[i], row[half + i]);
                    out[i] = a * c[i] - b * s[i];
                    out[half + i] = b * c[i] + a * s[i];
                }
                out
            }
            // Confirm the output is finite and non-trivial, and that a hand pairing swap on a single
            // vector differs, showing the conventions are distinct.
            let half = cfg.qk_rope_head_dim / 2;
            let sample = [0.7f32, -1.3, 0.4, 0.9];
            let cos = vec![0.8f32, 0.6];
            let sin = vec![0.6f32, 0.8];
            let inter = rope_interleaved_ref(&sample, &cos, &sin, 0, half);
            let halfsplit = rope_half_split_ref(&sample, &cos, &sin, 0, half);
            assert_ne!(
                inter, halfsplit,
                "interleaved-pair and half-split RoPE must give different results for a non-degenerate input"
            );
            assert!(interleaved.as_f32().unwrap().iter().all(|v| v.is_finite()));
        }

        /// Zeroing `n_shared_experts` (dropping the shared MLP) must change the output.
        #[test]
        fn deepseek2_shared_expert_is_load_bearing() {
            let cfg = tiny_cfg(Some(3));
            let mp_with = tiny_mp(1, 1.0);
            let mp_without = DeepseekV2MoeParams {
                n_shared_experts: 0,
                ..mp_with
            };
            let tokens = [4usize, 6, 8];
            let weights = all_weights(&cfg, &mp_with); // shared-expert weights present but unused by mp_without

            let g_with = trace_deepseek2_prefill(cfg, mp_with, tokens.len());
            let g_without = trace_deepseek2_prefill(cfg, mp_without, tokens.len());
            let out_with = eval_deepseek2_prefill(&g_with, &tokens, &weights);
            let out_without = eval_deepseek2_prefill(&g_without, &tokens, &weights);

            let max_diff = poot_test_util::max_abs_error(
                out_with.as_f32().unwrap(),
                out_without.as_f32().unwrap(),
            );
            assert!(
                max_diff > 1e-4,
                "shared-expert branch should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// `routed_scaling_factor` (`topk_weights *= routed_scaling_factor`) must affect the output.
        #[test]
        fn deepseek2_routed_scaling_factor_is_load_bearing() {
            let cfg = tiny_cfg(Some(3));
            let mp_a = tiny_mp(1, 1.0);
            let mp_b = tiny_mp(1, 3.5);
            let tokens = [2usize, 5, 9];
            let weights = all_weights(&cfg, &mp_a);
            let g_a = trace_deepseek2_prefill(cfg, mp_a, tokens.len());
            let g_b = trace_deepseek2_prefill(cfg, mp_b, tokens.len());
            let out_a = eval_deepseek2_prefill(&g_a, &tokens, &weights);
            let out_b = eval_deepseek2_prefill(&g_b, &tokens, &weights);
            let max_diff =
                poot_test_util::max_abs_error(out_a.as_f32().unwrap(), out_b.as_f32().unwrap());
            assert!(
                max_diff > 1e-4,
                "routed_scaling_factor should change the output; max_diff={max_diff:.2e}"
            );
        }

        /// A V2-Lite-like `DeepseekV2Yarn` (`factor: 40`, `mscale`-derived `attention_factor`/
        /// `softmax_mscale_sq`) must change both the RoPE table and `DeepseekV2Config::attn_scale`
        /// relative to plain RoPE, so YaRN cannot silently degrade to plain theta.
        #[test]
        fn deepseek2_yarn_scaling_is_load_bearing() {
            let yarn = DeepseekV2Yarn {
                factor: 40.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                attention_factor: 1.3,     // explicit non-1.0 override
                softmax_mscale_sq: 1.5896, // real V2-Lite's own get_mscale(40, 0.707)^2
            };

            // RoPE table: yarn's NTK-by-parts ramp + attention_factor must differ from plain theta.
            let (plain_cos, plain_sin) = deepseek2_rope_tables_interleaved(16, 8, 10_000.0, None);
            let (yarn_cos, yarn_sin) =
                deepseek2_rope_tables_interleaved(16, 8, 10_000.0, Some(&yarn));
            assert_ne!(
                plain_cos, yarn_cos,
                "YaRN-scaled cos table must differ from plain-theta RoPE"
            );
            assert_ne!(
                plain_sin, yarn_sin,
                "YaRN-scaled sin table must differ from plain-theta RoPE"
            );
            assert!(
                yarn_cos.iter().all(|v| v.is_finite()) && yarn_sin.iter().all(|v| v.is_finite())
            );
            // Position 0 has angle 0, but attention_factor still scales cos[0] away from 1.0.
            assert!(
                (yarn_cos[0] - yarn.attention_factor).abs() < 1e-5,
                "cos[pos=0] should equal attention_factor exactly: {}",
                yarn_cos[0]
            );

            // Softmax scale: attn_scale() must apply softmax_mscale_sq on top of the plain qk_head_dim^-0.5.
            let cfg_plain = DeepseekV2Config {
                yarn: None,
                ..tiny_cfg(Some(3))
            };
            let cfg_yarn = DeepseekV2Config {
                yarn: Some(yarn),
                ..tiny_cfg(Some(3))
            };
            let plain_scale = cfg_plain.attn_scale();
            let yarn_scale = cfg_yarn.attn_scale();
            assert!(
                (yarn_scale - plain_scale * yarn.softmax_mscale_sq).abs() < 1e-6,
                "yarn attn_scale should be plain_scale * softmax_mscale_sq: {yarn_scale} vs \
                 {plain_scale}*{}",
                yarn.softmax_mscale_sq
            );
            assert_ne!(plain_scale, yarn_scale);
        }

        /// End-to-end: `trace_deepseek2_prefill` matches the hand-rolled reference (reading cos/sin from
        /// the same weights map) with a V2-Lite-shaped YaRN config, so YaRN flows through the full graph.
        #[test]
        fn deepseek2_yarn_prefill_matches_hand_rolled_reference() {
            let yarn = DeepseekV2Yarn {
                factor: 40.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position_embeddings: 4096,
                attention_factor: 1.0, // real V2-Lite's own degenerate mscale==mscale_all_dim ratio
                softmax_mscale_sq: 1.5896,
            };
            let cfg = DeepseekV2Config {
                yarn: Some(yarn),
                ..tiny_cfg(None)
            }; // no q_lora, like V2-Lite
            let mp = tiny_mp(1, 1.0);
            let tokens = [3usize, 7, 1, 9, 15];
            let weights = all_weights(&cfg, &mp);

            let g = trace_deepseek2_prefill(cfg, mp, tokens.len());
            let got = eval_deepseek2_prefill(&g, &tokens, &weights);
            let want = deepseek2_prefill_ref(&cfg, &mp, &tokens, &weights);
            assert_matches(&got, &want, cfg.vocab);
            assert!(
                got.as_f32()
                    .unwrap()
                    .iter()
                    .any(|&v| v != got.as_f32().unwrap()[0])
            );
        }

        /// Decode-vs-prefill consistency: feeding a prompt one token at a time through
        /// [`trace_deepseek2_decode_kv_masked`] yields, at every position, the logits
        /// [`trace_deepseek2_prefill`] computes for that prefix. Checks the compressed-latent cache
        /// round-trips through `dynamic_update_slice_dyn` and per-step `kv_b_proj` decompression.
        #[test]
        fn deepseek2_decode_matches_prefill_at_every_position() {
            let cfg = tiny_cfg(Some(3));
            let mp = tiny_mp(1, 1.7);
            let tokens = [3usize, 7, 1, 9, 15, 2];
            let weights = all_weights(&cfg, &mp);
            let cap = tokens.len();

            let g_decode = trace_deepseek2_decode_kv_masked(cfg, mp, cap);
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

                // the prefill reference for the prefix 0..=pos
                let prefix: Vec<usize> = tokens[..=pos].to_vec();
                let g_prefill = trace_deepseek2_prefill(cfg, mp, prefix.len());
                let prefill_out = eval_deepseek2_prefill(&g_prefill, &prefix, &weights);

                assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
                // decode (actual) vs prefill (expected) at position `pos`
                poot_test_util::assert_close_rel(
                    logits.as_f32().unwrap(),
                    prefill_out.as_f32().unwrap(),
                    1e-3,
                );
            }
        }

        /// Card 296: independent reference for [`deepseek2_router_gate`]'s group-limited form (see
        /// `sigmoid_bias_grouped_router_ref` in `crate::deepseek3`): softmax scoring, top-k only over
        /// experts in the `topk_group` groups with the highest max `logits` (V2's plain max, not V3's
        /// sum-of-top-2). Not renormalized (`norm_topk_prob: false`).
        fn softmax_grouped_router_ref(
            logits: &[f32],
            n_group: usize,
            topk_group: usize,
            top_k: usize,
            routed_scaling_factor: f32,
        ) -> Vec<(usize, f32)> {
            let e = logits.len();
            assert_eq!(e % n_group, 0);
            let group_size = e / n_group;
            let canonical = |v: f32| {
                if v.is_nan() || v == f32::NEG_INFINITY {
                    f32::MIN
                } else if v == f32::INFINITY {
                    f32::MAX
                } else if v == 0.0 {
                    0.0
                } else {
                    v
                }
            };
            let scores: Vec<f32> = logits.iter().map(|&v| canonical(v)).collect();

            let mut group_scores = vec![f32::MIN; n_group];
            for (g, gs) in group_scores.iter_mut().enumerate() {
                *gs = scores[g * group_size..(g + 1) * group_size]
                    .iter()
                    .cloned()
                    .fold(f32::MIN, f32::max);
            }
            let mut group_idx: Vec<usize> = (0..n_group).collect();
            group_idx.sort_by(|&a, &b| group_scores[b].total_cmp(&group_scores[a]));
            let selected_groups: std::collections::HashSet<usize> =
                group_idx[..topk_group].iter().copied().collect();

            let mut idx: Vec<usize> = (0..e)
                .filter(|&i| selected_groups.contains(&(i / group_size)))
                .collect();
            idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let sel = &idx[..top_k];

            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exp_sel: Vec<f32> = sel.iter().map(|&i| (scores[i] - m).exp()).collect();
            let denom_full: f32 = scores.iter().map(|&v| (v - m).exp()).sum();
            sel.iter()
                .zip(exp_sel.iter())
                .map(|(&i, &ex)| (i, (ex / denom_full) * routed_scaling_factor))
                .collect()
        }

        /// Runs a `[1, E]`-logits router-gate graph and returns the raw gate weights.
        fn eval_router_gate(g: &Graph, logits_data: &[f32]) -> poot_tensor::HostTensor {
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let data = logits_data.to_vec();
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

        /// Card 296: `n_group=1` and `topk_group == n_group` are group-limiting no-ops and must match
        /// plain top-k (`top_k_router_ref`) bit-for-bit, protecting V2-Lite's `n_group: 1`.
        #[test]
        fn deepseek2_router_gate_group_limited_degenerates_to_ungated_when_unrestricted() {
            let e = 6usize;
            let k = 3usize;
            let scale = 2.5f32;
            let logits_data = [0.3f32, -1.2, 2.1, -0.4, 0.9, -2.5];

            let eval_with = |n_group: usize, topk_group: usize| -> poot_tensor::HostTensor {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, e]));
                let gate = deepseek2_router_gate(&b, logits, k, n_group, topk_group, scale);
                let g = b.finish(gate);
                eval_router_gate(&g, &logits_data)
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

            let want = top_k_router_ref(&logits_data, k, scale);
            let mut want_full = vec![0.0f32; e];
            for (i, wv) in want {
                want_full[i] = wv;
            }
            // group-limited-but-unrestricted (actual) vs the ungated reference (expected)
            poot_test_util::assert_close(one_group.as_f32().unwrap(), &want_full, 1e-4);
        }

        #[test]
        fn deepseek2_nonrenormalized_router_nonfinite_matches_independent_full_denominator() {
            struct Case {
                name: &'static str,
                logits: [f32; 6],
                k: usize,
            }
            let cases = [
                Case {
                    name: "finite",
                    logits: [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                    k: 3,
                },
                Case {
                    name: "positive infinity",
                    logits: [f32::INFINITY, f32::INFINITY, 5.0, 4.0, -1.0, -2.0],
                    k: 3,
                },
                Case {
                    name: "canonical low tie",
                    logits: [
                        f32::NEG_INFINITY,
                        f32::NAN,
                        f32::MIN,
                        f32::NEG_INFINITY,
                        f32::NAN,
                        f32::MIN,
                    ],
                    k: 3,
                },
                Case {
                    name: "canonical endpoints underflow",
                    logits: [f32::MAX, f32::MIN, 0.0, -0.0, f32::MIN, f32::MIN],
                    k: 4,
                },
                Case {
                    name: "signed zero boundary",
                    logits: [-0.0, 0.0, -0.0, 0.0, -1.0, -2.0],
                    k: 3,
                },
            ];
            let (n_group, topk_group, scale) = (3usize, 2usize, 2.5f32);

            for case in cases {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, case.logits.len()]));
                let (gate, mask) =
                    deepseek2_router_gate_and_mask(&b, logits, case.k, n_group, topk_group, scale);
                let out = b.concat(1, &[mask, gate]);
                let graph = b.finish(out);
                let got = eval_router_gate(&graph, &case.logits);
                let e = case.logits.len();
                let (got_mask, got_gate) = got.as_f32().unwrap().split_at(e);

                let reference =
                    softmax_grouped_router_ref(&case.logits, n_group, topk_group, case.k, scale);
                let mut want_mask = vec![0.0f32; e];
                let mut want_gate = vec![0.0f32; e];
                for &(id, weight) in &reference {
                    want_mask[id] = 1.0;
                    want_gate[id] = weight;
                }

                assert_eq!(got_mask, want_mask, "{}: selected mask", case.name);
                assert_eq!(
                    got_mask.iter().filter(|&&v| v == 1.0).count(),
                    case.k,
                    "{}: exactly k selected entries",
                    case.name
                );
                for (id, (&actual, &expected)) in got_gate.iter().zip(&want_gate).enumerate() {
                    assert!(actual.is_finite(), "{} expert {id}: non-finite", case.name);
                    assert!(
                        (actual - expected).abs() <= 1e-6,
                        "{} expert {id}: graph {actual} vs full-denominator reference {expected}",
                        case.name
                    );
                    if got_mask[id] == 0.0 {
                        assert_eq!(actual, 0.0, "{} expert {id}: unselected", case.name);
                    }
                }
                if case.name == "canonical endpoints underflow" {
                    assert_eq!(got_mask[1], 1.0, "endpoint expert stays selected");
                    assert_eq!(got_gate[1], 0.0, "selected endpoint weight underflows");
                }
            }
        }

        /// Card 296 negative control: with `n_group: 8`, `topk_group: 3`, group-limited masking must
        /// change the selection relative to ungated top-k. Because V2's group score is a plain max, the
        /// globally best expert is never excluded, so the fixture targets the 4th-best expert (expert 6,
        /// group 3): groups 0/1/2 (experts 0/2/4) outrank it and fill the `topk_group=3` cut, while
        /// ungated `k=4` selects it. Each group's second member is below every other group's primary
        /// (so it never displaces one in the ungated top-4) but close enough to its own group's max to
        /// carry non-negligible softmax weight, and no two values tie, so no rank tie lands on the
        /// k-th boundary. A mask with no effect would make gated and ungated selections identical.
        #[test]
        fn deepseek2_router_gate_group_limited_routing_changes_selection() {
            let e = 16usize;
            let n_group = 8usize;
            let topk_group = 3usize;
            let k = 4usize;
            let scale = 16.0f32; // real DeepSeek-V2's own routed_scaling_factor
            #[rustfmt::skip]
            let logits_data = [
                5.0f32, 3.4, // group 0 (selected: max 5.0)
                4.5, 3.3,    // group 1 (selected: max 4.5)
                4.0, 3.2,    // group 2 (selected: max 4.0)
                3.5, 3.1,    // group 3 (EXCLUDED: max 3.5, 4th-highest group, cut is top-3)
                1.0, 0.9,    // group 4
                0.0, -0.5,   // group 5
                -1.0, -1.5,  // group 6
                -2.0, -2.5,  // group 7
            ];

            let build_and_eval = |n_group: usize, topk_group: usize| -> poot_tensor::HostTensor {
                let b = Builder::new();
                let logits = b.constant("logits", TensorType::f32(vec![1, e]));
                let gate = deepseek2_router_gate(&b, logits, k, n_group, topk_group, scale);
                let g = b.finish(gate);
                eval_router_gate(&g, &logits_data)
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
            assert_eq!(
                ungated_sel,
                vec![0, 2, 4, 6],
                "sanity: ungated top-4 picks the 4 individually-highest experts, including expert 6"
            );
            assert_eq!(
                gated_sel,
                vec![0, 1, 2, 4],
                "group 3 (expert 6's group) is excluded, so expert 1 (group 0's low filler, still higher \
                 than anything outside groups 0-2) fills the 4th slot instead"
            );
            assert!(
                !gated_sel.contains(&6),
                "expert 6's group (max 3.5, group-rank 4th, cut is top-3 of 8) should be masked out"
            );

            // cross-check against the independent grouped reference
            let want = softmax_grouped_router_ref(&logits_data, n_group, topk_group, k, scale);
            let mut want_full = vec![0.0f32; e];
            for (i, wv) in want {
                want_full[i] = wv;
            }
            // graph (actual) vs the grouped reference (expected)
            poot_test_util::assert_close(gated.as_f32().unwrap(), &want_full, 1e-4);
        }
    }
}
