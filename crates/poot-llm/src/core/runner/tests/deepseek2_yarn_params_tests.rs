use super::super::*;
use poot_load::Qwen2HfConfig;

fn deepseek_v2_lite_config() -> Qwen2HfConfig {
    // A trimmed slice of the real `deepseek-ai/DeepSeek-V2-Lite`/`-Chat` config.json:
    // `mscale == mscale_all_dim == 0.707`, `factor: 40`.
    let json = r#"{
            "model_type": "deepseek_v2", "vocab_size": 102400, "hidden_size": 2048,
            "intermediate_size": 10944, "num_hidden_layers": 27, "num_attention_heads": 16,
            "num_key_value_heads": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 163840,
            "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "rope_scaling": {
                "type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                "mscale": 0.707, "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096
            }
        }"#;
    serde_json::from_str(json).expect("deepseek-v2-lite config must parse")
}

#[test]
fn real_v2_lite_mscale_equals_mscale_all_dim_gives_attention_factor_one() {
    // V2-Lite's degenerate case: mscale == mscale_all_dim makes the attention_factor ratio exactly 1.0
    // (get_mscale(40, 0.707) / get_mscale(40, 0.707)); expected, not a bug (see `deepseek2_yarn_params`).
    let y = deepseek2_yarn_params(&deepseek_v2_lite_config()).expect("yarn params");
    assert!(
        (y.attention_factor - 1.0).abs() < 1e-6,
        "attention_factor={}",
        y.attention_factor
    );
}

#[test]
fn real_v2_lite_softmax_mscale_sq_is_a_genuine_correction() {
    // softmax_mscale_sq is separate from attention_factor and is not 1.0 even though attention_factor
    // degenerates to 1.0: want = get_mscale(40, 0.707)^2 = (0.1*0.707*ln(40)+1.0)^2.
    let y = deepseek2_yarn_params(&deepseek_v2_lite_config()).expect("yarn params");
    let want_mscale = 0.1 * 0.707 * 40f32.ln() + 1.0;
    let want = want_mscale * want_mscale;
    assert!(
        (y.softmax_mscale_sq - want).abs() < 1e-5,
        "softmax_mscale_sq={} want={want}",
        y.softmax_mscale_sq
    );
    assert!(
        y.softmax_mscale_sq > 1.01,
        "must be a correction, not a no-op"
    );
}

#[test]
fn real_v2_lite_beta_fast_slow_and_original_ctx_pass_through() {
    let y = deepseek2_yarn_params(&deepseek_v2_lite_config()).expect("yarn params");
    assert_eq!(y.factor, 40.0);
    assert_eq!(y.beta_fast, 32.0);
    assert_eq!(y.beta_slow, 1.0);
    assert_eq!(y.original_max_position_embeddings, 4096);
}

#[test]
fn non_yarn_or_absent_rope_scaling_returns_none() {
    let mut hf = deepseek_v2_lite_config();
    hf.rope_scaling = None;
    assert!(deepseek2_yarn_params(&hf).is_none());
    let mut hf2 = deepseek_v2_lite_config();
    hf2.rope_scaling.as_mut().unwrap().rope_type = "linear".to_string();
    assert!(deepseek2_yarn_params(&hf2).is_none());
}
