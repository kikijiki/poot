/// The `rope_scaling` block. Schemas consumed: `rope_type: "llama3"` (Llama 3.x NTK-by-parts, HF
/// `_compute_llama3_parameters`; `factor`/`low_freq_factor`/`high_freq_factor`), `type: "longrope"`
/// (Phi-3/Phi-4; per-frequency `long_factor`/`short_factor`), `"linear"` (position interpolation,
/// `factor`), `"dynamic"` (HF `_compute_dynamic_ntk_parameters`: `factor` +
/// `original_max_position_embeddings`), and `"yarn"` (HF `_compute_yarn_parameters`: `factor` +
/// `original_max_position_embeddings`, optionally `beta_fast`/`beta_slow`/`attention_factor`).
/// Phi writes the discriminator as `type`, the rest as `rope_type`; both deserialize into
/// `rope_type` via the alias. The llama3 scalars default to 0.0 (unused unless llama3), the
/// longrope arrays are `None` for non-phi configs, and yarn-only fields are `None` unless present.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    #[serde(default, alias = "type")]
    pub rope_type: String,
    #[serde(default)]
    pub factor: f32,
    #[serde(default)]
    pub low_freq_factor: f32,
    #[serde(default)]
    pub high_freq_factor: f32,
    #[serde(default)]
    pub original_max_position_embeddings: usize,
    /// LongRoPE (phi3): per-frequency inv_freq rescale used above `original_max_position_embeddings`.
    /// Length is `rotary_dim/2`.
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
    /// LongRoPE (phi3): per-frequency inv_freq rescale used at/below the original context length.
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    /// YaRN: NTK-by-parts ramp low-end (num_rotations, extrapolation side). HF default 32.
    #[serde(default)]
    pub beta_fast: Option<f32>,
    /// YaRN: NTK-by-parts ramp high-end (num_rotations, interpolation side). HF default 1.
    #[serde(default)]
    pub beta_slow: Option<f32>,
    /// YaRN: explicit attention (temperature) scaling override. `None` derives it from `factor`
    /// via HF's `get_mscale` (`0.1 * ln(factor) + 1.0` for `factor > 1`), unless both
    /// `mscale`/`mscale_all_dim` are present, in which case HF uses their ratio
    /// (`get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`). DeepSeek-V2-Lite sets
    /// both to `0.707`, making the ratio exactly `1.0`.
    #[serde(default)]
    pub attention_factor: Option<f32>,
    /// YaRN: numerator of the `attention_factor` mscale ratio (DeepSeek-V2's `mscale` key); `None`
    /// on non-DeepSeek YaRN configs.
    #[serde(default)]
    pub mscale: Option<f32>,
    /// YaRN: denominator of the `attention_factor` mscale ratio, and (per DeepSeek
    /// `modeling_deepseek_v2.py`'s `yarn_apply_mscale`) a separate correction
    /// `get_mscale(factor, mscale_all_dim)^2` on the attention softmax scale on top of
    /// `qk_head_dim^-0.5` (see `poot_models::deepseek2::DeepseekV2Config::attn_scale`). DeepSeek-V2's
    /// `mscale_all_dim` key; `None` on non-DeepSeek YaRN configs.
    #[serde(default)]
    pub mscale_all_dim: Option<f32>,
}

/// The nested `rope_parameters` config block (card 135d/OlmoE); see
/// [`super::Qwen2HfConfig::rope_parameters`]. Only `rope_theta` is consumed; other keys are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f32>,
}
