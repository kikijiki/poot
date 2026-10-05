use poot_load::Qwen2HfConfig;
use poot_models::qwen2::Qwen2Config;

/// Build the shared qwen2-shaped tracer config from an already-parsed HF config.
///
/// This is the field mapping used by the safetensors Runner path. Architecture-specific callers may
/// validate and then patch fields that are not part of ordinary Qwen2/Qwen3 conversion, such as mRoPE
/// section widths for Qwen2.5-VL.
pub(crate) fn qwen2_config_from_hf(hf: &Qwen2HfConfig) -> Qwen2Config {
    Qwen2Config {
        vocab: hf.vocab_size,
        hidden: hf.hidden_size,
        inter: hf.intermediate_size,
        layers: hf.num_hidden_layers,
        n_heads: hf.num_attention_heads,
        n_kv_heads: hf.num_key_value_heads,
        head_dim: hf.head_dim(),
        rotary_dim: hf.rotary_dim(),
        eps: hf.rms_norm_eps,
        max_pos: hf.max_position_embeddings,
        qkv_bias: hf.qkv_bias(),
        qk_norm: hf.qk_norm(),
        // Honor `use_sliding_window: false` (real Qwen2.5 checkpoints set `sliding_window: 32768`
        // unconditionally but gate it off); see `Qwen2HfConfig::effective_sliding_window`.
        sliding_window: hf.effective_sliding_window(),
        attn_logit_softcap: hf.attn_logit_softcapping,
        final_logit_softcap: hf.final_logit_softcapping,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_hf(json: serde_json::Value) -> Qwen2HfConfig {
        serde_json::from_value(json).expect("parse Qwen2HfConfig")
    }

    #[test]
    fn qwen2_config_from_hf_preserves_ordinary_qwen2_and_qwen3_mrope_none() {
        let qwen2 = parse_hf(serde_json::json!({
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
            "sliding_window": 32768,
            "use_sliding_window": false
        }));
        let qwen2_cfg = qwen2_config_from_hf(&qwen2);
        assert_eq!(qwen2_cfg.vocab, 151936);
        assert_eq!(qwen2_cfg.hidden, 896);
        assert_eq!(qwen2_cfg.head_dim, 64);
        assert_eq!(qwen2_cfg.rotary_dim, 64);
        assert!(qwen2_cfg.qkv_bias);
        assert!(!qwen2_cfg.qk_norm);
        assert_eq!(qwen2_cfg.sliding_window, None);
        assert_eq!(qwen2_cfg.mrope_section, None);

        let qwen3 = parse_hf(serde_json::json!({
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960
        }));
        let qwen3_cfg = qwen2_config_from_hf(&qwen3);
        assert_eq!(qwen3_cfg.vocab, 151936);
        assert_eq!(qwen3_cfg.hidden, 1024);
        assert_eq!(qwen3_cfg.head_dim, 128);
        assert_eq!(qwen3_cfg.rotary_dim, 128);
        assert!(!qwen3_cfg.qkv_bias);
        assert!(qwen3_cfg.qk_norm);
        assert_eq!(qwen3_cfg.mrope_section, None);
    }
}
