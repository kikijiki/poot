use std::fs::File;
use std::path::Path;

use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf};
use poot_models::deepseek2::DeepseekV2Params;
use poot_models::deepseek3::DeepseekV3Params;
use poot_models::gpt_oss::GptOssParams;
use poot_models::granite::{GraniteParams, MoeShape};
use poot_models::olmoe::OlmoeParams;
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::Qwen3MoeParams;

use super::super::{LoraHotState, Runner, refuse_registered, runner_chat_format};
use crate::checkpoint::gguf::arch_config::gguf_weights;
use crate::checkpoint::gguf::deepseek::{
    gguf_config_deepseek2, gguf_config_deepseek2_moe, gguf_config_deepseek3_moe,
    gguf_deepseek2_is_v3_style, gguf_deepseek2_weights, gguf_deepseek3_weights,
};
use crate::checkpoint::gguf::{gguf_config, gguf_f32, gguf_proj_dtype, gguf_u32};
use crate::error::{Result, ResultExt, RunnerError};
use crate::text::tokenize::{ChatTemplate, TextCodec, gguf_spm_data, gguf_tokenizer};

impl Runner {
    /// Loads a model from a single GGUF file (spec 022): config and weights come from the one GGUF
    /// reader (card 543, dquant.md 8.1) - `GgufIndex` (header/metadata) plus a `WeightStore` built by
    /// `read_gguf` (every tensor's native bytes, no widening or transcoding) - and the tokenizer comes
    /// from the GGUF metadata. A GGUF of a registered family is refused
    /// ([`RunnerError::RegisteredFamily`]): it loads through `driver::ModelHandle`.
    ///
    /// Quantized tensors stay packed (card 545a): the family builders place them as packed carriers
    /// plus `WeightFormats`, which `Runner::bind_storage` puts onto every traced graph. The file is
    /// read once, tensor by tensor (`read_gguf`); the tokenizer reads the same header index.
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let g = GgufIndex::open(path).context("open gguf index")?;
        let file = File::open(path).context("open gguf file")?;
        let store = read_gguf(&g, &file, &IdentityNames).context("read gguf weights")?;
        refuse_registered(&poot_models::registry::RawConfig::Gguf(&g))?;
        let arch = g.architecture().unwrap_or("").to_string();
        // DeepSeek-V2 (card 135d, MLA, GGUF): unlike olmoe/qwen3moe below, MLA has no correspondence to `Qwen2Config`'s `n_heads`/`n_kv_heads`/`head_dim`/`rotary_dim` (see the
        // `deepseek2` field), so it needs dedicated GGUF builders (`gguf_config_deepseek2`/
        // `gguf_deepseek2_weights`, `gguf.rs`) rather than the generic qwen2-shaped `gguf_weights`. The llama.cpp
        // GGUF arch string is `"deepseek2"` (`LLM_ARCH_NAMES` in `src/llama-arch.cpp`), unlike the safetensors
        // `hf.model_type` `"deepseek_v2"`; `Runner::arch` is set to `"deepseek_v2"` below so `chat_format()`/
        // diagnostics match the safetensors path.
        if arch == "deepseek2" {
            let dcfg = gguf_config_deepseek2(&g).context("deepseek2 gguf config")?;
            let dense_inter = gguf_u32(&g, "deepseek2.feed_forward_length")
                .context("deepseek2 gguf missing feed_forward_length")?
                as usize;
            // Real V2 and V3 checkpoints share this "deepseek2" arch string (see `gguf_deepseek2_is_v3_style`);
            // only the `noaux_tc` selection-bias tensor tells them apart. Checked before building the MoE
            // config/weights, since the two structs/loaders differ from here.
            let is_v3 = gguf_deepseek2_is_v3_style(&g);
            let ((weights, formats), deepseek2, deepseek3, runner_arch) = if is_v3 {
                let dmoe3 = gguf_config_deepseek3_moe(&g, dense_inter)
                    .context("deepseek2 (v3-style) gguf moe config")?;
                let weights = gguf_deepseek3_weights(&g, &store, &dcfg, &dmoe3)
                    .context("deepseek2 (v3-style) gguf weights")?;
                (
                    weights,
                    None,
                    Some(DeepseekV3Params {
                        cfg: dcfg,
                        moe: dmoe3,
                    }),
                    "deepseek_v3".to_string(),
                )
            } else {
                let dmoe = gguf_config_deepseek2_moe(&g, dense_inter)
                    .context("deepseek2 gguf moe config")?;
                let weights = gguf_deepseek2_weights(&g, &store, &dcfg, &dmoe)
                    .context("deepseek2 gguf weights")?;
                (
                    weights,
                    Some(DeepseekV2Params {
                        cfg: dcfg,
                        moe: dmoe,
                    }),
                    None,
                    "deepseek_v2".to_string(),
                )
            };
            let tokenizer = gguf_tokenizer(&g)?;
            let eos = gguf_u32(&g, "tokenizer.ggml.eos_token_id").unwrap_or(100_001);
            let chat_template = g
                .get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let gguf_token_str = |key: &str| -> Option<String> {
                let id = g.get(key).and_then(|v| v.as_u64())? as usize;
                g.get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())?
                    .get(id)
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            };
            let bos_token = gguf_token_str("tokenizer.ggml.bos_token_id");
            let eos_token = gguf_token_str("tokenizer.ggml.eos_token_id");
            // `self.cfg` is populated generically (vocab/hidden/layers/eps/max_pos, plus a best-effort
            // `inter`/`head_dim`/`rotary_dim`/`n_kv_heads` with no MLA meaning) for bookkeeping only
            // (tokenizer decode); its attention-shape fields are unused by the deepseek2 tracer, as on
            // the safetensors path (see the `deepseek2` field).
            let cfg = Qwen2Config {
                vocab: dcfg.vocab,
                hidden: dcfg.hidden,
                inter: dense_inter,
                layers: dcfg.layers,
                n_heads: dcfg.n_heads,
                n_kv_heads: dcfg.n_heads,
                head_dim: dcfg.qk_head_dim(),
                rotary_dim: dcfg.qk_rope_head_dim,
                eps: dcfg.eps,
                max_pos: dcfg.max_pos,
                ..Default::default()
            };
            return Ok(Runner {
                cfg,
                weights,
                text: TextCodec::new(
                    tokenizer,
                    None,
                    None, // BOS convention not verified against a real checkpoint: no forced BOS
                    eos,
                    runner_chat_format(&runner_arch),
                    ChatTemplate {
                        template: chat_template,
                        bos_token,
                        eos_token,
                    },
                ),
                eos,
                arch: runner_arch,
                granite_moe: None,
                qwen3_moe: None,
                mixtral: None,
                olmoe: None,
                gpt_oss: None,
                deepseek2,
                deepseek3,
                deepseek32: None, // no GGUF loader for deepseek_v32 this round (spec 277 scope cut)
                nemotron_h: None,
                formats,
                sliding_window: None,
                lora_hot: std::sync::RwLock::new(LoraHotState::default()),
            });
        }
        let qkv_bias = match arch.as_str() {
            // qwen3moe (card 247) is the qwen3-attention-shaped MoE arch; llama.cpp's GGUF arch string is
            // "qwen3moe" (no underscore, unlike HF `model_type` "qwen3_moe"). granitemoe and olmoe carry no
            // q/k/v bias either.
            "granitemoe" | "qwen3moe" | "olmoe" => false,
            // gpt-oss (card 135d): q/k/v bias (the HF config has `attention_bias: true`, and llama.cpp's
            // `src/models/openai-moe.cpp` `create_tensor_qkv` creates `attn_{q,k,v}.bias`; `gguf_weights`'s
            // `arch == "gpt-oss"` block handles the o_proj bias this flag does not cover).
            "gpt-oss" => true,
            other => {
                return Err(RunnerError::UnsupportedModel {
                    model_type: other.to_string(),
                    reason: "not a supported GGUF architecture (supported: granitemoe, qwen3moe, \
                             olmoe, gpt-oss, deepseek2)",
                });
            }
        };
        let mut cfg = gguf_config(&g, &arch, qkv_bias)?;
        // Spec 135 Phase 2 / card 145 (the GGUF half of `Self::load`'s `proj_dtype` auto-select): native
        // F16/BF16 projection bytes stay narrow through the `WeightStore`/`materialize_dense`/`transpose2d`
        // path, but need the matching graph dtype before a backend can consume them. `gguf_proj_dtype`
        // probes layer 0's seven projection tensors (all seven, see its doc) and leaves `DType::F32` unless
        // they are uniformly F16 or uniformly BF16, so a quantized (packed) or mixed-dtype GGUF is
        // unchanged. Scoped to this qwen2-shaped path: the DeepSeek-V2 branch above builds its own `Runner`
        // and is not covered by this tensor-name set.
        cfg.proj_dtype = gguf_proj_dtype(&store);
        // Sparse-layer detection needs only `cfg.layers` and tensor presence; computed once and reused below.
        let qwen3moe_sparse_layer: Option<Vec<bool>> = (arch == "qwen3moe")
            .then(|| crate::architectures::qwen3moe::load::qwen3moe_sparse_layers(&g, &cfg));
        // Packed tensors stay packed: `gguf_weights` places them as carriers and records their
        // `WeightFormats`, which `Runner::bind_storage` puts onto every traced graph (card 545a).
        let (weights, formats) = gguf_weights(&g, &store, &cfg, &arch)?;
        // granitemoe deltas: the GGUF carries every Granite scalar (embedding/attention/residual/logit) plus the
        // expert counts, so `GraniteParams` is built straight from metadata. See card 231.
        let granite_moe = (arch == "granitemoe")
            .then(|| -> Result<GraniteParams> {
                Ok(GraniteParams {
                    moe: Some(MoeShape {
                        n_experts: gguf_u32(&g, "granitemoe.expert_count")? as usize,
                        top_k: gguf_u32(&g, "granitemoe.expert_used_count")? as usize,
                        inter: cfg.inter,
                    }),
                    embed_mult: gguf_f32(&g, "granitemoe.embedding_scale")?,
                    attn_mult: gguf_f32(&g, "granitemoe.attention.scale")?,
                    residual_mult: gguf_f32(&g, "granitemoe.residual_scale")?,
                    logits_scale: gguf_f32(&g, "granitemoe.logit_scale")?,
                })
            })
            .transpose()?;
        // qwen3moe deltas (card 247). Unlike the safetensors path (which reads decoder_sparse_step/
        // mlp_only_layers from config.json), the GGUF has no metadata key for either (absent from llama.cpp's
        // `Keys.LLM` in gguf-py/gguf/constants.py; `src/models/qwen3moe.cpp` creates ffn_gate_inp/ffn_*_exps for
        // every layer, so llama.cpp can only run all-routed qwen3moe GGUFs, matching every real release, e.g.
        // Qwen3-30B-A3B with decoder_sparse_step: 1 and mlp_only_layers: []). Rather than assume every layer is
        // sparse (which would mis-load a hypothetical mixed checkpoint), detect per layer from tensor presence:
        // a layer converted with a dense fallback MLP (llama.cpp's converter handles it; see the qwen3moe arm of
        // `gguf_weights` in `crates/poot-llm/src/gguf.rs`) has no `ffn_gate_inp.weight`, only the plain
        // `ffn_gate/up/down.weight` triplet. `sparse_layer` reuses `qwen3moe_sparse_layer` computed early above
        // from the same scan; `.unwrap_or_default()` is unreachable in practice (both are gated on `arch ==
        // "qwen3moe"`) but avoids a panic if that changes.
        let qwen3_moe = (arch == "qwen3moe")
            .then(|| -> Result<Qwen3MoeParams> {
                Ok(Qwen3MoeParams {
                    n_experts: gguf_u32(&g, "qwen3moe.expert_count")? as usize,
                    top_k: gguf_u32(&g, "qwen3moe.expert_used_count")? as usize,
                    inter: gguf_u32(&g, "qwen3moe.expert_feed_forward_length")? as usize,
                    sparse_layer: qwen3moe_sparse_layer.clone().unwrap_or_default(),
                })
            })
            .transpose()?;
        // OlmoE (card 135d, GGUF). Unlike Mixtral, OlmoE has its own GGUF arch string ("olmoe"; see
        // `conversion/olmo.py`'s `OlmoeModel`, confirmed with a `convert_hf_to_gguf.py --outtype f32`
        // conversion of `~/models/olmoe-tiny`), so detection is a plain arch-string match (evidence in
        // `gguf.rs`'s `gguf_weights` arm). Every OlmoE layer routes, so `cfg.inter` (from
        // `olmoe.feed_forward_length`) is the per-expert width directly (the conversion logs "feed forward
        // length = 16" = `intermediate_size`), unlike qwen3moe's separate `expert_feed_forward_length`.
        // `norm_topk_prob` has no GGUF metadata key for OlmoE (absent from `Keys.LLM.EXPERT_WEIGHTS_NORM` and
        // never written by `OlmoeModel.set_gguf_parameters`; llama.cpp's `src/models/olmoe.cpp` passes a
        // hardcoded `false` to `build_moe_ffn`), so default to `false`, matching the real checkpoint's config
        // and llama.cpp (unlike qwen3moe, which has a key and rejects `false`; see
        // `poot_load::Qwen2HfConfig::qwen3_moe_norm_topk_prob`).
        let olmoe = (arch == "olmoe")
            .then(|| -> Result<OlmoeParams> {
                Ok(OlmoeParams {
                    n_experts: gguf_u32(&g, "olmoe.expert_count")? as usize,
                    top_k: gguf_u32(&g, "olmoe.expert_used_count")? as usize,
                    inter: cfg.inter,
                    norm_topk_prob: false,
                })
            })
            .transpose()?;
        // gpt-oss (card 135d, GGUF). Own GGUF arch string ("gpt-oss", with a hyphen, unlike HF `model_type`
        // "gpt_oss"; evidence in `gguf.rs`'s `gguf_weights` arm), so detection is an arch-string match. Every
        // gpt-oss layer routes (like Mixtral/OlmoE), so `cfg.inter` (from `gpt-oss.feed_forward_length`) is the
        // per-expert width: `conversion/gpt_oss.py`'s `GptOssModel.set_gguf_parameters` writes both the generic
        // `feed_forward_length` (from the same `intermediate_size` HF field) and an explicit
        // `gpt-oss.expert_feed_forward_length` with the identical value. `cfg.inter` is used for consistency.
        //
        // `swiglu_limit` has no GGUF metadata key (`GptOssModel.set_gguf_parameters` never writes it; llama.cpp
        // hardcodes `constexpr float limit = 7.0f;` in `LLM_FFN_SWIGLU_OAI_MOE`, `src/llama-graph.cpp`), so this
        // path hardcodes 7.0, matching the real checkpoint and the `swiglu_limit.unwrap_or(7.0)` default in
        // `poot_load::Qwen2HfConfig` on the safetensors side.
        //
        // `layer_is_sliding` (the alternating sliding-window/full schedule) also has no GGUF array metadata,
        // unlike the safetensors `layer_types`. llama.cpp's `src/models/openai-moe.cpp` `load_arch_hparams`
        // defaults `swa_period = 2` and calls `hparams.set_swa_pattern(swa_period)` with `dense_first = false`,
        // which (`is_swa_impl[il] = il % n_pattern < (n_pattern - 1)` in `src/llama-hparams.cpp`) makes even
        // layers (0, 2, 4, ...) sliding and odd layers full, the same pattern as the HF `layer_types` on the
        // tiny fixture and the real 20b/120b (`["sliding_attention", "full_attention", ...]`). This path
        // hardcodes that parity.
        let gpt_oss = (arch == "gpt-oss")
            .then(|| -> Result<GptOssParams> {
                Ok(GptOssParams {
                    n_experts: gguf_u32(&g, "gpt-oss.expert_count")? as usize,
                    top_k: gguf_u32(&g, "gpt-oss.expert_used_count")? as usize,
                    inter: cfg.inter,
                    swiglu_limit: 7.0,
                    sliding_window: cfg.sliding_window.unwrap_or(128),
                    layer_is_sliding: (0..cfg.layers).map(|li| li % 2 == 0).collect(),
                })
            })
            .transpose()?;
        // Override the Runner's `arch` to "gpt_oss" once detected, keeping it consistent with the safetensors
        // loader (`arch: hf.model_type.clone()`): its GGUF arch string ("gpt-oss") differs from its safetensors
        // `model_type` ("gpt_oss"). This must come after every `gguf_config`/`gguf_weights` lookup above, which
        // need the real GGUF arch string; only the struct field changes. Captured before `gpt_oss` moves into
        // the `Runner` literal (needed again for the sliding window, as on the safetensors path).
        let is_gpt_oss = gpt_oss.is_some();
        let arch = if is_gpt_oss {
            "gpt_oss".to_string()
        } else {
            arch
        };
        let tokenizer = gguf_tokenizer(&g)?;
        let eos = gguf_u32(&g, "tokenizer.ggml.eos_token_id").unwrap_or(2);
        // Honor the GGUF's own add_bos_token flag; default false when absent.
        let bos = if g
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            g.get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
        } else {
            None
        };
        // The chat template + bos/eos strings from GGUF metadata (spec 035): the template is under
        // `tokenizer.chat_template`; the bos/eos strings are looked up in the token table at their ids.
        let chat_template = g
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let gguf_token_str = |key: &str| -> Option<String> {
            let id = g.get(key).and_then(|v| v.as_u64())? as usize;
            g.get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())?
                .get(id)
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        let bos_token = gguf_token_str("tokenizer.ggml.bos_token_id");
        let eos_token = gguf_token_str("tokenizer.ggml.eos_token_id");
        let spm = gguf_spm_data(&g);
        Ok(Runner {
            cfg,
            weights,
            text: TextCodec::new(
                tokenizer,
                spm,
                bos,
                eos,
                runner_chat_format(&arch),
                ChatTemplate {
                    template: chat_template,
                    bos_token,
                    eos_token,
                },
            ),
            eos,
            arch,
            granite_moe,
            qwen3_moe, // card 247: GGUF loading now builds this for arch == "qwen3moe"
            // A Mixtral GGUF names `general.architecture = "llama"`, a registered family, so the
            // registry resolve above refuses it before this point (POOT-738 brings the llama family's
            // experts).
            mixtral: None,
            // `Some` when `arch == "olmoe"`; see the `olmoe` binding above.
            olmoe,
            // `Some` when `arch == "gpt-oss"`; see the `gpt_oss` binding above.
            gpt_oss,
            deepseek2: None,
            deepseek3: None,
            deepseek32: None,
            nemotron_h: None,
            formats,
            // gpt-oss needs its per-layer local/global split, not the single uniform
            // `Runner::sliding_window`; see `GptOssParams::layer_is_sliding`.
            sliding_window: if is_gpt_oss { None } else { cfg.sliding_window },
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        })
    }
}
