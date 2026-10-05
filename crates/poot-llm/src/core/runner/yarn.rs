use poot_load::Qwen2HfConfig;

/// Resolves DeepSeek-V2's YaRN RoPE-scaling parameters (card 135d, spec 264) from `hf.rope_scaling` into
/// [`poot_models::deepseek2::DeepseekV2Yarn`]'s scalars. The one definition, shared by
/// [`Runner::load_impl`]'s `DeepseekV2Config` construction and [`build_weights`]'s interleaved-pair RoPE
/// table override. Returns `None` when there is no `rope_scaling` block or `rope_type` is not "yarn"
/// (plain RoPE, as the tiny random-init fixture). Mirrors the `original_max_position_embeddings` fold-in
/// of `build_weights`'s generic rope-table call (HF defaults an omitted value to
/// `max_position_embeddings`).
///
/// `pub(crate)` because `crate::deepseek32_load` (DeepSeek-V3.2's standalone loader, spec 277) needs the
/// same resolution: DSA has an identical MLA/YaRN shape.
pub(crate) fn deepseek2_yarn_params(
    hf: &Qwen2HfConfig,
) -> Option<poot_models::deepseek2::DeepseekV2Yarn> {
    let s = hf.rope_scaling.as_ref()?;
    if s.rope_type != "yarn" {
        return None;
    }
    let original_max_position_embeddings = if s.original_max_position_embeddings > 0 {
        s.original_max_position_embeddings
    } else {
        hf.original_max_position_embeddings
            .unwrap_or(hf.max_position_embeddings)
    };
    Some(deepseek2_yarn_resolve(
        s.factor,
        s.beta_fast.unwrap_or(32.0),
        s.beta_slow.unwrap_or(1.0),
        original_max_position_embeddings,
        s.attention_factor,
        s.mscale,
        s.mscale_all_dim,
    ))
}

/// The shared DeepSeek-V2 YaRN resolution core (card 135d, spec 264), extracted from
/// [`deepseek2_yarn_params`] (the safetensors/HF path) so the GGUF metadata parsing in
/// `crates/poot-llm/src/gguf.rs` computes the same [`poot_models::deepseek2::DeepseekV2Yarn`] scalars
/// from `deepseek2.rope.scaling.*` keys. `mscale`/`mscale_all_dim` are `Option<f32>` like
/// `poot_load::RopeScaling`'s fields; the GGUF caller passes the same recovered `mscale_all_dim` for both
/// (see that call site for why GGUF cannot represent HF's `mscale` separately).
pub(crate) fn deepseek2_yarn_resolve(
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position_embeddings: usize,
    attention_factor_override: Option<f32>,
    mscale: Option<f32>,
    mscale_all_dim: Option<f32>,
) -> poot_models::deepseek2::DeepseekV2Yarn {
    let factor = factor.max(1e-6);
    // As `yarn_get_mscale`/HF `_compute_yarn_parameters`'s `get_mscale`: 1.0 at scale<=1, else a log ramp
    // scaled by `mscale`.
    let get_mscale = |scale: f32, mscale: f32| -> f32 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * mscale * scale.ln() + 1.0
        }
    };
    // `attention_factor`: explicit override, else the mscale/mscale_all_dim ratio (real DeepSeek-V2 configs
    // set `mscale == mscale_all_dim == 0.707`, making the ratio exactly 1.0), else the generic
    // single-`mscale` YaRN default used by every other YaRN arch here.
    let attention_factor =
        attention_factor_override.unwrap_or_else(|| match (mscale, mscale_all_dim) {
            (Some(m), Some(mad)) if m != 0.0 && mad != 0.0 => {
                get_mscale(factor, m) / get_mscale(factor, mad)
            }
            _ => get_mscale(factor, 1.0),
        });
    // The separate softmax-scale correction (`yarn_apply_mscale`): `mscale_all_dim`-derived `mscale^2`, or a
    // no-op `1.0` when `mscale_all_dim` is absent.
    let softmax_mscale_sq = match mscale_all_dim {
        Some(mad) if mad != 0.0 => {
            let m = get_mscale(factor, mad);
            m * m
        }
        _ => 1.0,
    };
    poot_models::deepseek2::DeepseekV2Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_max_position_embeddings,
        attention_factor,
        softmax_mscale_sq,
    }
}
