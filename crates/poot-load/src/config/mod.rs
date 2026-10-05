use super::*;
pub mod quant;
pub mod rope;
pub use quant::{QuantConfig, QuantKind, QuantScheme};
pub use rope::{RopeParameters, RopeScaling};
/// The subset of an HF Qwen2-family `config.json` we need (covers qwen2 and qwen3).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen2HfConfig {
    #[serde(default)]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Card 135d (spec 262): the full-scale `allenai/OLMoE-1B-7B-0924-Instruct` `config.json`
    /// omits this key (the tiny-random fixture sets it), which was a `serde` "missing field"
    /// error. Falls back to `1e-5`, `transformers`' `OlmoeConfig` default.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Flat RoPE base frequency key. Do not read directly: call
    /// [`Qwen2HfConfig::effective_rope_theta`], which also honors the nested `rope_parameters`
    /// key and falls back to the family's reference value. `None` when the key is absent (a checkpoint
    /// using only the nested schema has no top-level key).
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Newer HF schema (seen on `hf-tiny-v2/tiny-random-OlmoeForCausalLM`,
    /// `transformers_version: "5.16.0.dev0"`): `rope_theta` moved into a nested
    /// `rope_parameters: { rope_theta, rope_type }` block. `None` for checkpoints using the flat
    /// key; see [`Qwen2HfConfig::effective_rope_theta`].
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Phi-3/Phi-4: fraction of each head's dims that get RoPE (the rest pass through unrotated). Absent
    /// (None) means full rotary for every other arch; see [`Qwen2HfConfig::rotary_dim`].
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// Phi-3/Phi-4 LongRoPE: the pre-extension context length, used for short/long factor
    /// selection and the attention magnitude factor. Phi puts it at the top level of config.json
    /// (Llama 3 nests it inside `rope_scaling`).
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// q/k/v projection bias. Absent in qwen2 (implicitly true) and in mistral/llama without the
    /// field (false); explicit `false` in qwen3 and llama. `None` means the per-arch default. Call
    /// [`Qwen2HfConfig::qkv_bias`] instead of reading this.
    #[serde(default)]
    pub attention_bias: Option<bool>,
    /// End-of-sequence token. Do not read directly: call [`Qwen2HfConfig::effective_eos_token_id`], which
    /// falls back to the family's reference value or a typed error, never another family's id.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Beginning-of-sequence token. Gemma is trained with BOS always present and degenerates without it;
    /// the tokenizer's `encode(_, false)` does not add it, so the runner prepends it for archs that set it.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// RoPE frequency rescaling. None for qwen2/qwen3 (plain RoPE); `llama3` for Llama 3.x (rescales the
    /// inverse frequencies before the cos/sin tables). Other `rope_type`s are not handled yet.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// Gemma: the softmax denominator base (attn scale = 1/sqrt(this)) instead of head_dim.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Gemma: local (sliding-window) layers use this RoPE base; global layers use `rope_theta`.
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    /// Gemma: every `sliding_window_pattern`-th layer is global (full attn), the rest local.
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// Mistral/Qwen2: uniform sliding-window size (queries see the `sliding_window` most recent
    /// keys); `None` means full causal. For qwen2-family configs call
    /// [`Qwen2HfConfig::effective_sliding_window`]: Qwen2.5 checkpoints carry `sliding_window:
    /// 32768` alongside `use_sliding_window: false`, so the field is dormant.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Qwen2-family: whether `sliding_window` is applied (HF `Qwen2Config` default `false`; Qwen2.5
    /// sets `sliding_window: 32768` and gates it off with this flag). `None` (Mistral/Gemma have no
    /// such flag) passes `sliding_window` through unchanged.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// Gemma2/Grok: per-layer attention-logit softcap coefficient (key name in HF config.json is
    /// `attn_logit_softcapping`). `None` for models that omit it.
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    /// Gemma2/Grok: final-logit softcap coefficient (key name `final_logit_softcapping`). `None`
    /// for models that omit it.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    /// MoE: total experts per layer (GraniteMoE `num_local_experts`).
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    /// MoE: experts selected per token (`num_experts_per_tok`); GraniteMoE and Qwen3-MoE share the key.
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// Qwen3-MoE: total routed experts per layer (`num_experts`); GraniteMoE calls this
    /// `num_local_experts`.
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// Qwen3-MoE: per-expert FFN intermediate size (`moe_intermediate_size`), distinct from the dense
    /// `intermediate_size` used by non-MoE layers (see `mlp_only_layers`/`decoder_sparse_step`).
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// Qwen3-MoE: whether top-k router weights are renormalized to sum to 1. Every real checkpoint
    /// sets `true`; poot's MoE op (`poot_graph_ir::ops::moe`) only computes the renormalized
    /// form, so `Some(false)` is rejected (see [`Qwen2HfConfig::qwen3_moe_norm_topk_prob`]).
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    /// Qwen3-MoE: layer `li` is MoE when `(li+1) % decoder_sparse_step == 0`; other layers keep a
    /// dense swiglu MLP. HF default 1 (every layer MoE) when omitted.
    #[serde(default)]
    pub decoder_sparse_step: Option<usize>,
    /// Qwen3-MoE: layer indices forced dense even if `decoder_sparse_step` would select MoE. HF
    /// default empty.
    #[serde(default)]
    pub mlp_only_layers: Option<Vec<usize>>,
    /// Granite: token-embedding multiplier.
    #[serde(default)]
    pub embedding_multiplier: Option<f32>,
    /// Granite: attention softmax scale (replaces 1/sqrt(head_dim)).
    #[serde(default)]
    pub attention_multiplier: Option<f32>,
    /// Granite: per-sublayer residual multiplier.
    #[serde(default)]
    pub residual_multiplier: Option<f32>,
    /// Granite: final-logits divisor.
    #[serde(default)]
    pub logits_scaling: Option<f32>,
    /// Quantization scheme (GPTQ/AWQ/...) if the checkpoint is quantized; `None` for f32/bf16 models.
    #[serde(default)]
    pub quantization_config: Option<QuantConfig>,
    /// gpt-oss: the clamp bound of the clamped sigmoid-gated activation (`GptOssExperts._apply_gate`'s
    /// `self.limit`); `7.0` on the 20b/120b and the `tiny-random/gpt-oss` fixture.
    #[serde(default)]
    pub swiglu_limit: Option<f32>,
    /// gpt-oss: per-layer attention-window schedule (`"sliding_attention"`/`"full_attention"`),
    /// length `num_hidden_layers`. Read directly rather than assumed to alternate (see
    /// `poot_models::gpt_oss`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// DeepSeek-V2 (card 135d, MLA): the shared low-rank KV latent width (`kv_lora_rank`).
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    /// DeepSeek-V2: low-rank query compression width (`q_lora_rank`). `null`/absent on
    /// `DeepSeek-V2-Lite` (selects a plain `q_proj`); `Some` on `DeepSeek-V2`/`V3` and the
    /// `yujiepan/deepseek-v2-tiny-random` fixture (selects `q_a_proj`/`q_a_layernorm`/`q_b_proj`);
    /// see `poot_models::deepseek2::DeepseekV2Config::q_lora_rank`.
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    /// DeepSeek-V2: per-head non-rotated K/Q slice width (`qk_nope_head_dim`).
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    /// DeepSeek-V2: per-head rotated (decoupled-RoPE) K/Q slice width (`qk_rope_head_dim`).
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    /// DeepSeek-V2: per-head V width (`v_head_dim`) - independent of `qk_nope_head_dim +
    /// qk_rope_head_dim`.
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    /// DeepSeek-V2: routed experts per MoE layer (`n_routed_experts`; Qwen3-MoE's `num_experts`,
    /// GraniteMoE's `num_local_experts`).
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    /// DeepSeek-V2: number of always-active shared experts, combined into ONE MLP
    /// (`n_shared_experts`; `0`/absent disables the shared-expert branch).
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    /// DeepSeek-V2: layers `< first_k_dense_replace` use a plain dense MLP; layers
    /// `>= first_k_dense_replace` route (`first_k_dense_replace`; `1` on every real config observed).
    #[serde(default)]
    pub first_k_dense_replace: Option<usize>,
    /// DeepSeek-V2: scalar applied to the selected routed-expert gate weights after the
    /// non-renormalized softmax selection (`routed_scaling_factor`; see `poot_models::deepseek2`).
    /// V3 uses the same key but `deepseek3_router_gate` applies it after renormalization (see
    /// `poot_models::deepseek3::DeepseekV3MoeParams::routed_scaling_factor`).
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    /// DeepSeek-V2/V3: number of equal-size groups the routed-expert axis splits into for
    /// group-limited routing (`n_group`; `8` on V3 and non-Lite V2). Card 296: read by both the
    /// `is_deepseek2()` path (`topk_method: "group_limited_greedy"`,
    /// `poot_models::deepseek2::deepseek2_router_gate`) and the `is_deepseek3()` path
    /// (`"noaux_tc"`, `poot_models::deepseek3::deepseek3_router_gate`). Same key, different
    /// group score (V2: max per group; V3: sum of top 2).
    #[serde(default)]
    pub n_group: Option<usize>,
    /// DeepSeek-V2/V3: groups selected per token before the expert-level top-`k` (`topk_group`;
    /// `4` on V3, `3` on non-Lite V2). Read by both loader paths (see `n_group`).
    #[serde(default)]
    pub topk_group: Option<usize>,
    /// DeepSeek-V3.2 (DSA, spec 277): Lightning Indexer heads (`index_n_heads`; `64` on
    /// `deepseek-ai/DeepSeek-V3.2-Exp`). Read only by `is_deepseek32()`; V3 configs omit it.
    #[serde(default)]
    pub index_n_heads: Option<usize>,
    /// DeepSeek-V3.2: per-indexer-head query/key width (`index_head_dim`; `128` on the real config).
    #[serde(default)]
    pub index_head_dim: Option<usize>,
    /// DeepSeek-V3.2: top-`k` KV entries the Lightning Indexer selects per query token (`index_topk`;
    /// `2048` on the real config). See `poot_models::deepseek32::DsaConfig::index_topk`'s doc comment.
    #[serde(default)]
    pub index_topk: Option<usize>,
    /// DeepSeek-V4 (spec 281): parallel mHC residual streams (`hc_mult`; `4` on
    /// `deepseek-ai/DeepSeek-V4-Flash-0731`).
    #[serde(default)]
    pub hc_mult: Option<usize>,
    /// DeepSeek-V4: Sinkhorn-Knopp round count for mHC's `comb` doubly-stochastic projection
    /// (`hc_sinkhorn_iters`; `20` on the real config).
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<usize>,
    /// DeepSeek-V4: numerical floor added after mHC's sigmoid/softmax/Sinkhorn steps (`hc_eps`; `1e-6` on
    /// the real config).
    #[serde(default)]
    pub hc_eps: Option<f32>,
    /// DeepSeek-V4: number of independent blocks the grouped output projection splits
    /// `num_attention_heads*head_dim` into (`o_groups`; `8` on the real Flash config, `16` on Pro).
    #[serde(default)]
    pub o_groups: Option<usize>,
    /// DeepSeek-V4: per-group intermediate width of the grouped output projection (`o_lora_rank`; `1024`
    /// on the real config).
    #[serde(default)]
    pub o_lora_rank: Option<usize>,
    /// DeepSeek-V4: RoPE base for CSA/HCA (compressed) layers, used for everything on the layer
    /// (Q, local K/V, compressed entries, output derotation) in place of `rope_theta`
    /// (`compress_rope_theta`; `160000`). `sliding_attention` layers keep `rope_theta`.
    #[serde(default)]
    pub compress_rope_theta: Option<f32>,
    /// DeepSeek-V4: legacy per-layer compression schedule (`compress_ratios`; a flat `Vec<usize>` of
    /// `0`/`csa_compress_rate`/`hca_compress_rate`, one per layer; `0` decodes to
    /// `sliding_attention`). `DeepSeek-V4-Flash-0731` has 46 entries for `num_hidden_layers: 43`:
    /// the trailing 3 (all `0`) cover the separate `mtp.{0,1,2}` multi-token-prediction modules
    /// (layer indices in `model.safetensors.index.json` are exactly `0..=42`). Callers must take
    /// only the first `num_hidden_layers` entries. `swiglu_limit` above is reused for
    /// V4's SwiGLU clamp (`10.0`), the same role as in gpt-oss's `GptOssExperts._apply_gate`.
    #[serde(default)]
    pub compress_ratios: Option<Vec<usize>>,
}

/// The `transformers` config-class default `rope_theta` for `model_type`, when the reference class has one
/// (read from transformers 4.57.6 and 5.17.0, which agree; `deepseek_v32` exists only in 5.17.0).
/// `None` for a `model_type` this loader has no reference for (including the empty string).
fn reference_rope_theta(model_type: &str) -> Option<f32> {
    match model_type {
        "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3_moe" | "phi3" | "granite"
        | "granitemoe" | "olmo2" | "olmoe" | "deepseek_v2" | "deepseek_v3" | "deepseek_v32" => {
            Some(10_000.0)
        }
        "mixtral" | "gemma3_text" => Some(1_000_000.0),
        "smollm3" => Some(2_000_000.0),
        "gpt_oss" => Some(150_000.0),
        _ => None,
    }
}

/// The `transformers` config-class default `eos_token_id` for `model_type`, when the reference class has
/// one (read from transformers 4.57.6 and 5.17.0, which agree; `deepseek_v32` exists only in 5.17.0). These
/// are the class defaults, not any checkpoint's own value. The Qwen and gpt-oss classes declare none, so
/// their checkpoints must name it. `None` for a `model_type` this loader has no reference for (including
/// the empty string).
fn reference_eos_token_id(model_type: &str) -> Option<u32> {
    match model_type {
        "llama" | "mistral" | "mixtral" | "granite" | "granitemoe" => Some(2),
        "gemma3_text" => Some(1),
        "phi3" => Some(32000),
        "olmo2" | "olmoe" => Some(50279),
        "deepseek_v2" => Some(2),
        "deepseek_v3" | "deepseek_v32" => Some(1),
        "smollm3" => Some(128_001),
        _ => None,
    }
}

/// Fallback for `Qwen2HfConfig::rms_norm_eps` when `config.json` omits it (seen on
/// `allenai/OLMoE-1B-7B-0924-Instruct`, card 135d, spec 262): `transformers`' `OlmoeConfig` default.
fn default_rms_norm_eps() -> f32 {
    1e-5
}

impl Qwen2HfConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    /// Effective head_dim (explicit for qwen3, else hidden/heads).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    /// Number of head dims that get RoPE: full head_dim except phi3, where `partial_rotary_factor`
    /// (e.g. 0.75) rotates only the leading fraction. Rounded to even (rotate-half pairs halves).
    pub fn rotary_dim(&self) -> usize {
        match self.partial_rotary_factor {
            Some(f) if f < 1.0 => {
                let r = (self.head_dim() as f32 * f) as usize;
                r - (r % 2)
            }
            _ => self.head_dim(),
        }
    }
    /// The `sliding_window` that applies, honoring `use_sliding_window`. Qwen2.5 checkpoints (e.g.
    /// `Qwen/Qwen2.5-0.5B-Instruct`) set `sliding_window: 32768` with `use_sliding_window: false`,
    /// so the field is a capability, not an active window. Only `Some(false)` suppresses it; `None`
    /// (Mistral/Gemma, no such flag) or `Some(true)` pass it through.
    pub fn effective_sliding_window(&self) -> Option<usize> {
        if self.use_sliding_window == Some(false) {
            None
        } else {
            self.sliding_window
        }
    }
    /// per-head QK-norm: qwen3, qwen3-moe, and gemma3 all carry q_norm/k_norm tensors.
    pub fn qk_norm(&self) -> bool {
        self.model_type == "qwen3"
            || self.model_type == "qwen3_moe"
            || self.model_type == "gemma3_text"
    }
    /// Qwen3-MoE (`qwen3_moe`): the MoE sibling of Qwen3 dense. Same attention flags
    /// (`qkv_bias() == false`, `qk_norm() == true`) plus a per-layer dense/MoE FFN switch; see
    /// [`Qwen2HfConfig::qwen3_moe_layer_is_sparse`].
    pub fn is_qwen3_moe(&self) -> bool {
        self.model_type == "qwen3_moe"
    }
    /// Qwen3-MoE: validates the router-renormalization flag. poot's MoE op
    /// (`poot_graph_ir::ops::moe`/`moe_sparse`/`moe_grouped`) only computes the renormalized top-k
    /// softmax gate (`norm_topk_prob: true`), so an explicit `false` is rejected rather than loaded
    /// with wrong gate weights. Real checkpoints (e.g. Qwen3-30B-A3B) set `true`; absent is
    /// treated as `true` (HF's `false` default only applies to from-scratch configs).
    pub fn qwen3_moe_norm_topk_prob(&self) -> Result<(), LoadError> {
        if self.norm_topk_prob == Some(false) {
            return Err(LoadError::SafeTensors(
                "qwen3_moe config sets norm_topk_prob=false; poot's MoE op (poot_graph_ir::ops::moe) only \
                 supports the renormalized top-k gate (norm_topk_prob=true)"
                    .to_string(),
            ));
        }
        Ok(())
    }
    /// Qwen3-MoE: whether layer `layer_idx` (0-indexed) routes through the expert mixture or keeps
    /// a dense swiglu MLP. Mirrors HF `Qwen3MoeDecoderLayer.__init__`:
    /// `layer_idx not in mlp_only_layers and (layer_idx + 1) % decoder_sparse_step == 0`. With the
    /// HF defaults every layer is MoE (Qwen3-30B-A3B); the `yujiepan/qwen3-moe-tiny-random` fixture
    /// sets `decoder_sparse_step: 2`, so this is evaluated per layer.
    pub fn qwen3_moe_layer_is_sparse(&self, layer_idx: usize) -> bool {
        let step = self.decoder_sparse_step.unwrap_or(1).max(1);
        let mlp_only = self.mlp_only_layers.as_deref().unwrap_or(&[]);
        !mlp_only.contains(&layer_idx) && (layer_idx + 1).is_multiple_of(step)
    }
    /// GraniteMoE (`granitemoe`): Granite's four scalar multipliers on a top-k MoE MLP.
    pub fn is_granite_moe(&self) -> bool {
        self.model_type == "granitemoe"
    }
    /// Mixtral (card 135d, spec 261): a Mistral-shaped GQA+RoPE decoder with a top-k MoE MLP on
    /// every layer (no per-layer switch). Uses `num_local_experts`/`num_experts_per_tok` (as
    /// granitemoe), and `intermediate_size` is the per-expert FFN width (no
    /// `moe_intermediate_size`).
    pub fn is_mixtral(&self) -> bool {
        self.model_type == "mixtral"
    }
    /// OlmoE (card 135d, spec 262): olmo2's full-dimension QK-norm on a standard pre-norm block,
    /// with a top-k MoE MLP on every layer like Mixtral. Uses `num_experts`/`num_experts_per_tok`
    /// (as qwen3-moe), and `intermediate_size` is the per-expert FFN width. Unlike qwen3-moe
    /// (`norm_topk_prob: false` rejected by `qwen3_moe_norm_topk_prob`), OlmoE's `false` is
    /// supported through its own router (`poot_models::olmoe::olmoe_ffn`).
    pub fn is_olmoe(&self) -> bool {
        self.model_type == "olmoe"
    }
    /// gpt-oss (card 135d, spec 263): OpenAI's `gpt-oss-20b`/`-120b`. A top-k MoE on every layer
    /// like Mixtral/OlmoE, but with attention sinks (a per-head learned softmax-competing logit,
    /// `poot_models::gpt_oss::attention_prefill_with_sink`), sliding-window/full-attention layers
    /// per `layer_types` (read per layer), bias on every q/k/v/o projection, the router, and both
    /// expert projections (`attention_bias: true`), and a clamped sigmoid-gated activation
    /// (`swiglu_limit`) instead of plain SwiGLU. The router math is identical to Mixtral's
    /// `poot_graph_ir::ops::top_k_gate` (see `poot_models::gpt_oss`), using `num_local_experts`/
    /// `num_experts_per_tok`.
    pub fn is_gpt_oss(&self) -> bool {
        self.model_type == "gpt_oss"
    }
    /// DeepSeek-V2 (card 135d, MLA): K/V compressed into a shared low-rank latent
    /// (`kv_lora_rank`), with decoupled RoPE (`qk_nope_head_dim`/`qk_rope_head_dim`; only the rope
    /// slice rotates, in DeepSeek's interleaved-pair convention rather than poot's half-split),
    /// and a routed top-k plus always-active shared-experts MoE MLP on layers
    /// `>= first_k_dense_replace` (earlier layers are dense). See `poot_models::deepseek2` for the
    /// derivation from `modeling_deepseek_v2.py`.
    pub fn is_deepseek2(&self) -> bool {
        self.model_type == "deepseek_v2"
    }
    /// DeepSeek-V3 (`poot_models::deepseek3`): V2's MLA attention (same `DeepseekV2Config` /
    /// `is_deepseek2` fields) with a different router (`topk_method: "noaux_tc"`,
    /// `scoring_func: "sigmoid"`, `e_score_correction_bias`, group-limited routing via
    /// `n_group`/`topk_group`). `deepseek-ai/DeepSeek-V3`'s `config.json` has
    /// `model_type: "deepseek_v3"` (with the underscore, unlike GGUF's shared `"deepseek2"`; see
    /// `gguf_deepseek2_is_v3_style` in `crates/poot-llm/src/gguf.rs`).
    pub fn is_deepseek3(&self) -> bool {
        self.model_type == "deepseek_v3"
    }
    /// DeepSeek-V3.2 (`poot_models::deepseek32`, spec 277 DSA): V3's MLA attention and MoE router
    /// (same `DeepseekV2Config`/`DeepseekV3MoeParams` fields; DSA leaves MLA compression and the
    /// router unchanged) plus the Lightning Indexer fields `index_n_heads`/`index_head_dim`/
    /// `index_topk`. `deepseek-ai/DeepSeek-V3.2-Exp` has `model_type: "deepseek_v32"`, distinct
    /// from V3's `"deepseek_v3"`, so `is_deepseek3` and `is_deepseek32` are mutually exclusive.
    pub fn is_deepseek32(&self) -> bool {
        self.model_type == "deepseek_v32"
    }
    /// The effective RoPE base frequency: the nested `rope_parameters.rope_theta` (see
    /// [`RopeParameters`] / [`Qwen2HfConfig::rope_parameters`]) when present, else the flat
    /// `rope_theta`, else the `model_type`'s reference value. Card 135d:
    /// `hf-tiny-v2/tiny-random-OlmoeForCausalLM` carries only the nested key. Errors when neither key is
    /// present and the family has no reference value, or when the resolved value is not positive and
    /// finite (a zero base makes infinite inverse frequencies and NaN RoPE tables).
    pub fn effective_rope_theta(&self) -> Result<f32, LoadError> {
        let theta = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .or_else(|| reference_rope_theta(&self.model_type))
            .ok_or_else(|| LoadError::MissingRopeTheta {
                model_type: self.model_type.clone(),
            })?;
        if theta.is_finite() && theta > 0.0 {
            Ok(theta)
        } else {
            Err(LoadError::InvalidRopeTheta {
                model_type: self.model_type.clone(),
                value: theta,
            })
        }
    }
    /// The effective end-of-sequence token: the config's `eos_token_id`, else the `model_type`'s reference
    /// value. Errors when the key is absent and the family has no reference value.
    pub fn effective_eos_token_id(&self) -> Result<u32, LoadError> {
        self.eos_token_id
            .or_else(|| reference_eos_token_id(&self.model_type))
            .ok_or_else(|| LoadError::MissingEosTokenId {
                model_type: self.model_type.clone(),
            })
    }
    /// The quantization scheme of a supported quantized checkpoint ([`QuantConfig::scheme`]), `None`
    /// for unquantized.
    pub fn quant(&self) -> Result<Option<QuantScheme>, LoadError> {
        self.quantization_config
            .as_ref()
            .map(QuantConfig::scheme)
            .transpose()
    }
    /// Effective q/k/v projection bias: the explicit `attention_bias`, else the per-arch default
    /// (Qwen2 and Qwen-VL text towers implicitly biased; llama/mistral/etc not).
    pub fn qkv_bias(&self) -> bool {
        self.attention_bias.unwrap_or(matches!(
            self.model_type.as_str(),
            "qwen2" | "qwen2_vl" | "qwen2_5_vl"
        ))
    }
    /// The FP8 `(block_out, block_in)` block of a DeepSeek-style checkpoint, from
    /// `quantization_config.weight_block_size` if present, else `[128, 128]` (DeepSeek-V3.2-Exp and
    /// V4-Flash-0731).
    pub fn fp8_block_size(&self) -> (usize, usize) {
        self.quantization_config
            .as_ref()
            .and_then(|q| q.weight_block_size)
            .map(|[bo, bi]| (bo, bi))
            .unwrap_or((128, 128))
    }
}

impl QuantConfig {
    /// The quantization scheme this block names: GPTQ 4-bit sym/no act-order, AWQ 4-bit GEMM (specs
    /// 019-020) or compressed-tensors float-quantized FP8. Errors on unsupported variants.
    pub fn scheme(&self) -> Result<QuantScheme, LoadError> {
        let group_size = self.group_size.unwrap_or(128);
        match self.quant_method.as_str() {
            "gptq" => {
                if self.bits != Some(4) || self.desc_act == Some(true) || self.sym == Some(false) {
                    return Err(LoadError::SafeTensors(format!(
                        "unsupported GPTQ variant (bits={:?} desc_act={:?} sym={:?}); only 4-bit sym no-act-order",
                        self.bits, self.desc_act, self.sym
                    )));
                }
                Ok(QuantScheme {
                    kind: QuantKind::Gptq,
                    group_size,
                })
            }
            "awq" => {
                if self.bits != Some(4) || self.version.as_deref() != Some("gemm") {
                    return Err(LoadError::SafeTensors(format!(
                        "unsupported AWQ variant (bits={:?} version={:?}); only 4-bit gemm",
                        self.bits, self.version
                    )));
                }
                Ok(QuantScheme {
                    kind: QuantKind::Awq,
                    group_size,
                })
            }
            "compressed-tensors" => {
                if self.format.as_deref() != Some("float-quantized") {
                    return Err(LoadError::SafeTensors(format!(
                        "unsupported compressed-tensors format {:?}; only float-quantized (FP8)",
                        self.format
                    )));
                }
                Ok(QuantScheme {
                    kind: QuantKind::Fp8,
                    group_size,
                })
            }
            other => Err(LoadError::SafeTensors(format!(
                "unsupported quant_method {other:?} (gptq, awq, compressed-tensors today)"
            ))),
        }
    }
}

/// The quantization scheme of an HF checkpoint whose raw `config.json` is `config`, `None` when it has
/// no `quantization_config`: read without parsing a family's config, so any family's loader packs its
/// quantized linears the same way.
pub fn quant_scheme(config: &serde_json::Value) -> Result<Option<QuantScheme>, LoadError> {
    config
        .get("quantization_config")
        .map(|block| {
            serde_json::from_value::<QuantConfig>(block.clone())
                .map_err(|e| LoadError::SafeTensors(format!("quantization_config: {e}")))?
                .scheme()
        })
        .transpose()
}
