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
    /// Flat RoPE base frequency key; `None` when the key is absent (a checkpoint using only the
    /// nested `rope_parameters` schema has no top-level key).
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Newer HF schema (seen on `hf-tiny-v2/tiny-random-OlmoeForCausalLM`,
    /// `transformers_version: "5.16.0.dev0"`): `rope_theta` moved into a nested
    /// `rope_parameters: { rope_theta, rope_type }` block. `None` for checkpoints using the flat
    /// key.
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
    /// End-of-sequence token; `None` when the key is absent.
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
    /// form, so `Some(false)` was refused by the Runner (the qwen3-MoE family now reads the flag itself).
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
