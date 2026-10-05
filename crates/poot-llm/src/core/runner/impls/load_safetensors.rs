use std::io::Read;
use std::path::Path;

use poot_load::Qwen2HfConfig;
use poot_load::safetensors;
use poot_models::deepseek2::{DeepseekV2Config, DeepseekV2MoeParams, DeepseekV2Params};
use poot_models::deepseek3::{DeepseekV3MoeParams, DeepseekV3Params};
use poot_models::deepseek32::Deepseek32Params;
use poot_models::gpt_oss::GptOssParams;
use poot_models::granite::{GraniteParams, MoeShape};
use poot_models::mixtral::MixtralParams;
use poot_models::olmoe::OlmoeParams;
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::Qwen3MoeParams;
use poot_quant::weights::WeightEntry;
use poot_tensor::DType;
use tokenizers::Tokenizer;

use super::super::weights::build_weights;
use super::super::yarn::deepseek2_yarn_params;
use super::super::{
    LoraHotState, Runner, refuse_registered, runner_chat_format, validate_safetensors_arch,
};
use crate::architectures::deepseek32_load::{build_deepseek32_weights, deepseek32_config_from_hf};
use crate::architectures::nemotron_h_load::{
    build_nemotron_h_weights, is_nemotron_h, nemotron_h_config_from_json,
};
use crate::architectures::qwen2_hf_config::qwen2_config_from_hf;
use crate::error::{OptionExt, Result, ResultExt, RunnerError};
use crate::text::chat::read_tokenizer_config_chat;
use crate::text::tokenize::{ChatTemplate, TextCodec};

impl Runner {
    /// Loads config.json, model.safetensors, and tokenizer.json from a model directory. A checkpoint of
    /// a registered family is refused ([`RunnerError::RegisteredFamily`]): it loads through
    /// `driver::ModelHandle`. Every dense weight keeps its stored dtype (a bf16/f16 checkpoint's weights are bf16/f16 words); a
    /// GPTQ/AWQ/FP8 checkpoint's quantized linears stay packed.
    ///
    /// Card 537 (ADR-0104 decision 5, R475-016): this used to also select a resident mode via
    /// `POOT_QUANT_RESIDENT=1`, a mode-selecting env read in library code. With no host f32 mirror
    /// of a native weight left to drop, there is no mode to select.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(dir.as_ref())
    }

    pub(crate) fn load_impl(dir: &Path) -> Result<Self> {
        const MAX_CONFIG_BYTES: usize = 16 * 1024 * 1024;

        let mut raw_config_bytes = Vec::new();
        std::fs::File::open(dir.join("config.json"))
            .and_then(|file| {
                file.take(MAX_CONFIG_BYTES as u64 + 1)
                    .read_to_end(&mut raw_config_bytes)
            })
            .map_err(|error| err!("read config.json: {error}"))?;
        if raw_config_bytes.len() > MAX_CONFIG_BYTES {
            bail!("read config.json: larger than {MAX_CONFIG_BYTES} bytes");
        }
        let raw_config: serde_json::Value =
            serde_json::from_slice(&raw_config_bytes).context("parse config.json")?;
        refuse_registered(&poot_models::registry::RawConfig::HfJson {
            config: &raw_config,
            generation: None,
        })?;
        // Nemotron-H (spec 279): peeked at before the flat parse below: its config.json has no
        // `rope_theta`/`max_position_embeddings`/`tie_word_embeddings` (NoPE; `NemotronHHfConfig` reads a
        // disjoint field set) and would fail the `Qwen2HfConfig::load` parse. See the `nemotron_h` field.
        if is_nemotron_h(&raw_config) {
            return Self::load_nemotron_h_impl(dir, &raw_config);
        }
        // Refuse an architecture this loader does not know before the flat `Qwen2HfConfig` parse: a nested
        // VLM-style config (glm5_next, qwen3_5, qwen4_exp) would otherwise fail that parse with an unrelated
        // missing-field error instead of naming the architecture. A config with no string `model_type` is
        // refused too: it must never fall through to the qwen2 tracer (card 190).
        let model_type = raw_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RunnerError::UnsupportedModel {
                model_type: "<none>".to_string(),
                reason: "config.json has no string model_type",
            })?;
        validate_safetensors_arch(model_type)?;
        let hf: Qwen2HfConfig =
            serde_json::from_slice(&raw_config_bytes).context("load config.json")?;
        // DeepSeek-V3.2 DSA (spec 277): unlike Nemotron-H above, a real `deepseek_v32` config.json parses through
        // `Qwen2HfConfig` (it uses the same MLA/MoE fields as DeepSeek-V3; see the `deepseek32` field), so this
        // check runs after the parse. Still an early return, since
        // `deepseek32_load::deepseek32_config_from_hf`/`build_deepseek32_weights` build a complete config/weight
        // map from `hf` directly.
        if hf.is_deepseek32() {
            return Self::load_deepseek32_impl(dir, &hf);
        }
        let mut cfg = qwen2_config_from_hf(&hf);
        let granite_moe = hf.is_granite_moe().then(|| GraniteParams {
            moe: Some(MoeShape {
                n_experts: hf.num_local_experts.unwrap_or(0),
                top_k: hf.num_experts_per_tok.unwrap_or(1),
                inter: cfg.inter,
            }),
            embed_mult: hf.embedding_multiplier.unwrap_or(1.0),
            attn_mult: hf
                .attention_multiplier
                .unwrap_or(1.0 / (cfg.head_dim as f32).sqrt()),
            residual_mult: hf.residual_multiplier.unwrap_or(1.0),
            logits_scale: hf.logits_scaling.unwrap_or(1.0),
        });
        // Card 246: Qwen3-MoE's per-layer dense/routed-expert switch. `norm_topk_prob=false` is rejected
        // (poot's shared moe op has no non-renormalized form; see `qwen3_moe_norm_topk_prob`); the three
        // MoE-shape fields are required on any real qwen3_moe config.json.
        let qwen3_moe = if hf.is_qwen3_moe() {
            hf.qwen3_moe_norm_topk_prob()?;
            Some(Qwen3MoeParams {
                n_experts: hf
                    .num_experts
                    .context("qwen3_moe config missing num_experts")?,
                top_k: hf
                    .num_experts_per_tok
                    .context("qwen3_moe config missing num_experts_per_tok")?,
                inter: hf
                    .moe_intermediate_size
                    .context("qwen3_moe config missing moe_intermediate_size")?,
                sparse_layer: (0..cfg.layers)
                    .map(|li| hf.qwen3_moe_layer_is_sparse(li))
                    .collect(),
            })
        } else {
            None
        };
        // Mixtral (card 135d): every layer routes through a top-k MoE MLP (no per-layer dense/MoE switch).
        // `num_local_experts`/`num_experts_per_tok` are the HF keys granitemoe uses; `intermediate_size`
        // (already `cfg.inter`) is the per-expert FFN width, as Mixtral has no separate
        // `moe_intermediate_size`.
        let mixtral = hf.is_mixtral().then(|| MixtralParams {
            n_experts: hf.num_local_experts.unwrap_or(0),
            top_k: hf.num_experts_per_tok.unwrap_or(1),
            inter: cfg.inter,
        });
        // OlmoE (card 135d): every layer routes through a top-k MoE MLP (like Mixtral).
        // `num_experts`/`num_experts_per_tok` are the HF keys qwen3-moe uses; `intermediate_size` (already
        // `cfg.inter`) is the per-expert FFN width. Unlike qwen3-moe (which rejects `norm_topk_prob: false`),
        // OlmoE's router supports both; the real checkpoint sets `false`, so an absent key defaults to `false`.
        let olmoe = hf.is_olmoe().then(|| OlmoeParams {
            n_experts: hf.num_experts.unwrap_or(0),
            top_k: hf.num_experts_per_tok.unwrap_or(1),
            inter: cfg.inter,
            norm_topk_prob: hf.norm_topk_prob.unwrap_or(false),
        });
        // gpt-oss (card 135d): every layer routes through a top-k MoE MLP (like Mixtral/OlmoE) with attention
        // sinks, alternating sliding windows, and a biased clamped GLU (see poot_models::gpt_oss).
        // `num_local_experts`/`num_experts_per_tok` are the HF keys Mixtral/OlmoE/granitemoe share.
        // `layer_types` is the real per-layer window schedule and is required: a config missing it would
        // mis-window every layer as full-causal (which `layer_is_sliding`'s absence default would hide), so
        // this errors instead of defaulting.
        let gpt_oss = if hf.is_gpt_oss() {
            let layer_types = hf
                .layer_types
                .clone()
                .context("gpt_oss config missing layer_types")?;
            Some(GptOssParams {
                n_experts: hf
                    .num_local_experts
                    .context("gpt_oss config missing num_local_experts")?,
                top_k: hf
                    .num_experts_per_tok
                    .context("gpt_oss config missing num_experts_per_tok")?,
                inter: cfg.inter,
                swiglu_limit: hf.swiglu_limit.unwrap_or(7.0),
                sliding_window: hf.sliding_window.unwrap_or(128),
                layer_is_sliding: layer_types
                    .iter()
                    .map(|s| s == "sliding_attention")
                    .collect(),
            })
        } else {
            None
        };
        let is_gpt_oss = gpt_oss.is_some();
        // DeepSeek-V2 (card 135d, MLA): uses its own `DeepseekV2Config`, not `cfg`/`Qwen2Config` (see the
        // `deepseek2` field). `q_lora_rank` is optional (DeepSeek-V2-Lite omits it: a plain q_proj; the real
        // V2/V3 and the tiny fixture set it: a low-rank q_a_proj/q_a_layernorm/q_b_proj branch), so it is not
        // `.context()`-required. `yarn` resolves the checkpoint's YaRN `rope_scaling` block via
        // `deepseek2_yarn_params` (`None` for plain RoPE, as the tiny fixture). `n_group`/`topk_group` (card
        // 296) default to `1`/`n_group` (the tracer's no-op shape, see `deepseek2_router_gate`) when omitted,
        // as in the `deepseek3` branch below and `gguf_config_deepseek2_moe`.
        let deepseek2 = if hf.is_deepseek2() {
            let dcfg = DeepseekV2Config {
                vocab: cfg.vocab,
                hidden: cfg.hidden,
                layers: cfg.layers,
                n_heads: cfg.n_heads,
                q_lora_rank: hf.q_lora_rank,
                kv_lora_rank: hf
                    .kv_lora_rank
                    .context("deepseek_v2 config missing kv_lora_rank")?,
                qk_nope_head_dim: hf
                    .qk_nope_head_dim
                    .context("deepseek_v2 config missing qk_nope_head_dim")?,
                qk_rope_head_dim: hf
                    .qk_rope_head_dim
                    .context("deepseek_v2 config missing qk_rope_head_dim")?,
                v_head_dim: hf
                    .v_head_dim
                    .context("deepseek_v2 config missing v_head_dim")?,
                eps: cfg.eps,
                max_pos: cfg.max_pos,
                rope_theta: hf.effective_rope_theta()?,
                yarn: deepseek2_yarn_params(&hf),
            };
            let n_group = hf.n_group.unwrap_or(1);
            let topk_group = hf.topk_group.unwrap_or(n_group);
            let dmoe = DeepseekV2MoeParams {
                n_routed_experts: hf
                    .n_routed_experts
                    .context("deepseek_v2 config missing n_routed_experts")?,
                top_k: hf
                    .num_experts_per_tok
                    .context("deepseek_v2 config missing num_experts_per_tok")?,
                moe_inter: hf
                    .moe_intermediate_size
                    .context("deepseek_v2 config missing moe_intermediate_size")?,
                n_shared_experts: hf.n_shared_experts.unwrap_or(0),
                dense_inter: cfg.inter,
                first_k_dense_replace: hf.first_k_dense_replace.unwrap_or(1),
                n_group,
                topk_group,
                routed_scaling_factor: hf.routed_scaling_factor.unwrap_or(1.0),
            };
            Some(DeepseekV2Params {
                cfg: dcfg,
                moe: dmoe,
            })
        } else {
            None
        };
        // DeepSeek-V3 (docs/updates/0791): same `DeepseekV2Config` for MLA attention as V2 above (see
        // `is_deepseek3`), with its own `DeepseekV3MoeParams` router shape (sigmoid + selection-only bias +
        // group-limited routing, see `poot_models::deepseek3`). `hf.model_type` is "deepseek_v2" XOR
        // "deepseek_v3" (unlike the GGUF path's shared "deepseek2" string, see `gguf_deepseek2_is_v3_style`),
        // so `deepseek2` and `deepseek3` stay mutually exclusive. `n_group`/`topk_group` default to `1`/`n_group`
        // (see `deepseek3_router_gate`) when omitted, as `gguf_config_deepseek3_moe` does on the GGUF side.
        let deepseek3 = if hf.is_deepseek3() {
            let dcfg = DeepseekV2Config {
                vocab: cfg.vocab,
                hidden: cfg.hidden,
                layers: cfg.layers,
                n_heads: cfg.n_heads,
                q_lora_rank: hf.q_lora_rank,
                kv_lora_rank: hf
                    .kv_lora_rank
                    .context("deepseek_v3 config missing kv_lora_rank")?,
                qk_nope_head_dim: hf
                    .qk_nope_head_dim
                    .context("deepseek_v3 config missing qk_nope_head_dim")?,
                qk_rope_head_dim: hf
                    .qk_rope_head_dim
                    .context("deepseek_v3 config missing qk_rope_head_dim")?,
                v_head_dim: hf
                    .v_head_dim
                    .context("deepseek_v3 config missing v_head_dim")?,
                eps: cfg.eps,
                max_pos: cfg.max_pos,
                rope_theta: hf.effective_rope_theta()?,
                yarn: deepseek2_yarn_params(&hf),
            };
            let n_group = hf.n_group.unwrap_or(1);
            let topk_group = hf.topk_group.unwrap_or(n_group);
            let dmoe = DeepseekV3MoeParams {
                n_routed_experts: hf
                    .n_routed_experts
                    .context("deepseek_v3 config missing n_routed_experts")?,
                top_k: hf
                    .num_experts_per_tok
                    .context("deepseek_v3 config missing num_experts_per_tok")?,
                moe_inter: hf
                    .moe_intermediate_size
                    .context("deepseek_v3 config missing moe_intermediate_size")?,
                n_shared_experts: hf.n_shared_experts.unwrap_or(0),
                dense_inter: cfg.inter,
                first_k_dense_replace: hf.first_k_dense_replace.unwrap_or(1),
                n_group,
                topk_group,
                routed_scaling_factor: hf.routed_scaling_factor.unwrap_or(1.0),
            };
            Some(DeepseekV3Params {
                cfg: dcfg,
                moe: dmoe,
            })
        } else {
            None
        };
        let quant = hf.quant().context("quantization_config")?;
        // Card 545a: a quantized checkpoint's linears are packed straight from their checkpoint bytes
        // (GPTQ/AWQ/FP8 per-channel), never dequantized; `build_weights` places them packed.
        let store = safetensors::load_weight_store(dir).context("load safetensors")?;
        let mut store = match quant {
            Some(scheme) => safetensors::pack_quantized_linears(&store, scheme)
                .context("pack quantized linears")?,
            None => store,
        };
        // Detect if the projection weights are natively bf16 or f16 (from the safetensors dtype), and if so
        // set proj_dtype so the tracers type those constants as DType::BF16/F16 and GPU backends upload native
        // bytes without an in-graph Cast (spec 135 Phase 2 residency; f16 mirrors card 134's bf16). Probe
        // layer 0's q_proj: the store always carries the checkpoint's own dtype metadata (card 540b), so this
        // needs no `cpu_oracle_only` branch or header re-probe: every entry already knows its `DType`.
        let source_proj_dtype = match store.get("model.layers.0.self_attn.q_proj.weight") {
            Some(WeightEntry::Dense(dense)) => match dense.dtype() {
                DType::BF16 => DType::BF16,
                DType::F16 => DType::F16,
                _ => DType::F32,
            },
            _ => DType::F32,
        };
        cfg.proj_dtype = source_proj_dtype;
        let (weights, formats) = build_weights(&mut store, &cfg, &hf)?;
        // `store` (every tensor's bytes as stored, held whole) is fully consumed by build_weights. Drop it
        // explicitly rather than at end of scope: for a large checkpoint that is tens of GB held through
        // tokenizer/chat-template loading and Runner construction (see docs/updates/, "cpu-eager memory",
        // for DeepSeek-V2-Lite peak numbers).
        drop(store);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        // The model's chat template + bos/eos strings (spec 035), read from tokenizer_config.json when present;
        // absent -> the server falls back to the spec-034 hardcoded ChatFormat.
        let (chat_template, bos_token, eos_token) = read_tokenizer_config_chat(dir);
        let eos = hf.effective_eos_token_id()?;
        Ok(Runner {
            cfg,
            weights,
            text: TextCodec::new(
                tokenizer,
                None, // safetensors path uses the HF tokenizer.json directly (not GGUF SPM)
                None,
                eos,
                runner_chat_format(&hf.model_type),
                ChatTemplate {
                    template: chat_template,
                    bos_token,
                    eos_token,
                },
            ),
            eos,
            arch: hf.model_type.clone(),
            granite_moe,
            qwen3_moe,
            mixtral,
            olmoe,
            gpt_oss,
            deepseek2,
            deepseek3,
            deepseek32: None, // deepseek_v32 detection happens earlier in load_impl and returns via load_deepseek32_impl
            nemotron_h: None,
            formats,
            // gpt-oss needs its per-layer local/global split, not the single uniform
            // `Runner::sliding_window` (the `mask.prefill` / `mask.prefill.local` step-input pair in
            // poot_models::gpt_oss).
            sliding_window: if is_gpt_oss { None } else { cfg.sliding_window },
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        })
    }

    pub(crate) fn load_nemotron_h_impl(dir: &Path, raw_config: &serde_json::Value) -> Result<Self> {
        let (nh_cfg, eos, _bos) =
            nemotron_h_config_from_json(raw_config).context("parse nemotron_h config")?;
        let store = safetensors::load_weight_store(dir).context("load safetensors")?;
        let weights =
            build_nemotron_h_weights(&store, &nh_cfg).context("build nemotron_h weights")?;
        drop(store);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        let (chat_template, bos_token, eos_token) = read_tokenizer_config_chat(dir);
        let cfg = Qwen2Config {
            vocab: nh_cfg.vocab_size,
            hidden: nh_cfg.hidden,
            layers: nh_cfg.pattern.len(),
            eps: nh_cfg.eps,
            ..Default::default()
        };
        Ok(Runner {
            cfg,
            weights,
            text: TextCodec::new(
                tokenizer,
                None,
                None, // BOS convention not verified against a real checkpoint: no forced BOS
                eos,
                runner_chat_format("nemotron_h"),
                ChatTemplate {
                    template: chat_template,
                    bos_token,
                    eos_token,
                },
            ),
            eos,
            arch: "nemotron_h".to_string(),
            granite_moe: None,
            qwen3_moe: None,
            mixtral: None,
            olmoe: None,
            gpt_oss: None,
            deepseek2: None,
            deepseek3: None,
            deepseek32: None,
            nemotron_h: Some(nh_cfg),
            formats: Default::default(),
            sliding_window: None,
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        })
    }

    /// DeepSeek-V3.2 DSA safetensors load (spec 277): builds `deepseek32_load`'s three config structs and
    /// the complete weight map directly from `hf` (`deepseek32_config_from_hf`/`build_deepseek32_weights`;
    /// see the `deepseek32` field for why this is a dedicated early return), and builds a base `Runner` with
    /// `deepseek32: Some(..)`, like `load_nemotron_h_impl`. `self.cfg` gets only
    /// vocab/hidden/layers/eps/max_pos (its attention-shape fields are unused by the DSA tracer).
    /// Tokenizer/chat-template loading is the generic path (DSA changes only attention), so
    /// `Tokenizer::from_file`/`read_tokenizer_config_chat` are reused.
    pub(crate) fn load_deepseek32_impl(dir: &Path, hf: &Qwen2HfConfig) -> Result<Self> {
        let (dcfg, dmoe, dsa) =
            deepseek32_config_from_hf(hf).context("parse deepseek_v32 config")?;
        let store = safetensors::load_weight_store(dir).context("load safetensors")?;
        let (weights, formats) =
            build_deepseek32_weights(&store, &dcfg, &dmoe, hf.fp8_block_size())
                .context("build deepseek_v32 weights")?;
        drop(store);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        let (chat_template, bos_token, eos_token) = read_tokenizer_config_chat(dir);
        let cfg = Qwen2Config {
            vocab: dcfg.vocab,
            hidden: dcfg.hidden,
            layers: dcfg.layers,
            eps: dcfg.eps,
            max_pos: dcfg.max_pos,
            ..Default::default()
        };
        let eos = hf.effective_eos_token_id()?;
        Ok(Runner {
            cfg,
            weights,
            text: TextCodec::new(
                tokenizer,
                None,
                None, // BOS convention not verified against a real checkpoint: no forced BOS
                eos,
                runner_chat_format(&hf.model_type),
                ChatTemplate {
                    template: chat_template,
                    bos_token,
                    eos_token,
                },
            ),
            eos,
            arch: hf.model_type.clone(),
            granite_moe: None,
            qwen3_moe: None,
            mixtral: None,
            olmoe: None,
            gpt_oss: None,
            deepseek2: None,
            deepseek3: None,
            deepseek32: Some(Deepseek32Params {
                cfg: dcfg,
                moe: dmoe,
                dsa,
            }),
            nemotron_h: None,
            formats,
            sliding_window: None,
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        })
    }
}
