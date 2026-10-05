use poot_eval::materialize_dense;
use poot_graph_ir::rope_table::{RopeFlavor, RopeTables};
use poot_load::RopeScaling;
use poot_load::gguf::GgufIndex;
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;

use crate::error::{Result, ResultExt};

/// A GGUF `{arch}.rope.freq_base` that cannot be a RoPE base: `base^(-2j/d)` is `inf` at `base == 0` and NaN
/// below it, so the whole cos/sin table would be NaN. A missing key is not this error (it takes the
/// architecture's reference default); a present-but-invalid one is a fault in the checkpoint.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("GGUF `{key}` is {value}, not a finite RoPE base frequency > 0")]
pub struct InvalidRopeBase {
    pub key: String,
    pub value: f32,
}

/// A small per-frequency factor tensor (`rope_freqs`, phi3 `rope_factors_*`) as the f32 values the table
/// builder multiplies in: the one explicit widening, whatever float dtype the GGUF stored it in.
fn widen_factors(store: &WeightStore, name: &str) -> Result<Vec<f32>> {
    let tensor = materialize_dense(store, name)?;
    Ok(tensor
        .to_f32()
        .with_context(|| format!("{name}: widening rope factors"))?
        .into_owned())
}

/// Build the `rope.cos`/`rope.sin` tables from a GGUF's rope metadata. Reads only the small rope tensors
/// (llama3 `rope_freqs` / phi3 `rope_factors_{short,long}`), not the model weights. Flavors: plain
/// (qwen2/base-llama), llama3 NTK-by-parts baked as `rope_freqs.weight`, phi3 LongRoPE (short + optional
/// long regime), and linear/YaRN via `{arch}.rope.scaling.type` (GGUF has no representation for
/// dynamic-NTK).
pub fn gguf_rope_tables(
    g: &GgufIndex,
    store: &WeightStore,
    cfg: &Qwen2Config,
    arch: &str,
) -> Result<(HostTensor, HostTensor)> {
    // rope base is arch-keyed; default per arch if the GGUF omits it (qwen2 1e6, llama/phi3 1e4).
    let base_key = format!("{arch}.rope.freq_base");
    let theta = g
        .get(&base_key)
        .and_then(|v| v.as_f32())
        .unwrap_or(if arch == "qwen2" {
            1_000_000.0
        } else {
            10_000.0
        });
    if !(theta.is_finite() && theta > 0.0) {
        return Err(InvalidRopeBase {
            key: base_key,
            value: theta,
        })
        .context("building GGUF rope tables");
    }
    // llama.cpp bakes the llama3 rope rescale into a `rope_freqs.weight` tensor (effective inv_freq = base /
    // factor) instead of carrying the HF rope_scaling params in metadata. Apply it when present.
    let freq_factors = match g.tensors.contains_key("rope_freqs.weight") {
        true => Some(widen_factors(store, "rope_freqs.weight")?),
        false => None,
    };
    // phi3 LongRoPE: llama.cpp stores per-frequency factors as `rope_factors_{short,long}.weight` plus
    // `{arch}.rope.scaling.original_context_length`. Build the same `longrope` RopeScaling as the safetensors
    // path: short-regime factors (always present here) plus long-regime factors when the GGUF ships them
    // (`rope_tables` picks between them once, using the table's capacity as the effective sequence length),
    // and the constant attention magnitude factor (recomputed by rope_tables from cfg.max_pos /
    // original_context_length, matching the GGUF's rope.scaling.attn_factor).
    let scaling = if arch == "phi3" && g.tensors.contains_key("rope_factors_short.weight") {
        let short = widen_factors(store, "rope_factors_short.weight")?;
        let long = match g.tensors.contains_key("rope_factors_long.weight") {
            true => Some(widen_factors(store, "rope_factors_long.weight")?),
            false => None,
        };
        let orig = g
            .get(&format!("{arch}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        Some(RopeScaling {
            rope_type: "longrope".to_string(),
            factor: 0.0,
            low_freq_factor: 0.0,
            high_freq_factor: 0.0,
            original_max_position_embeddings: orig,
            long_factor: long,
            short_factor: Some(short),
            beta_fast: None,
            beta_slow: None,
            attention_factor: None,
            mscale: None,
            mscale_all_dim: None,
        })
    } else {
        // linear / YaRN: llama.cpp's conversion writes these under `{arch}.rope.scaling.type`
        // (`"linear"`/`"yarn"`, gguf-py `Keys.Rope`/`RopeScalingType`) with params under
        // `{arch}.rope.scaling.{factor,attn_factor,original_context_length,yarn_beta_fast,yarn_beta_slow}`
        // (src/llama-arch.cpp `LLM_KV_ROPE_SCALING_*`). There is no GGUF representation for HF's "dynamic"
        // (dynamic-NTK) type: `RopeScalingType` defines only `none`/`linear`/`yarn`/`longrope`, and dynamic NTK is a
        // per-forward-recompute HF construct. A checkpoint that needs it stays a safetensors/HF-config load (card 236).
        let scaling_type = g
            .get(&format!("{arch}.rope.scaling.type"))
            .and_then(|v| v.as_str());
        match scaling_type {
            Some(t @ ("linear" | "yarn")) => {
                let factor = g
                    .get(&format!("{arch}.rope.scaling.factor"))
                    .and_then(|v| v.as_f32())
                    .unwrap_or(1.0);
                // HF defaults an omitted original_max_position_embeddings to the checkpoint's max_position_embeddings;
                // cfg.max_pos (from `{arch}.context_length`) is the GGUF equivalent (as in runner.rs).
                let orig = g
                    .get(&format!("{arch}.rope.scaling.original_context_length"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(cfg.max_pos);
                let attention_factor = g
                    .get(&format!("{arch}.rope.scaling.attn_factor"))
                    .and_then(|v| v.as_f32());
                let beta_fast = g
                    .get(&format!("{arch}.rope.scaling.yarn_beta_fast"))
                    .and_then(|v| v.as_f32());
                let beta_slow = g
                    .get(&format!("{arch}.rope.scaling.yarn_beta_slow"))
                    .and_then(|v| v.as_f32());
                Some(RopeScaling {
                    rope_type: t.to_string(),
                    factor,
                    low_freq_factor: 0.0,
                    high_freq_factor: 0.0,
                    original_max_position_embeddings: orig,
                    long_factor: None,
                    short_factor: None,
                    beta_fast,
                    beta_slow,
                    attention_factor,
                    // GGUF has no representation for DeepSeek's mscale/mscale_all_dim keys (absent from gguf-py
                    // `Keys.Rope`); a GGUF-converted checkpoint always takes the `attn_factor`/default-`get_mscale(factor)`
                    // path.
                    mscale: None,
                    mscale_all_dim: None,
                })
            }
            _ => None,
        }
    };
    Ok(rope_tables(
        cfg,
        theta,
        scaling.as_ref(),
        freq_factors.as_deref(),
    ))
}

/// The [`RopeFlavor`] an HF `rope_scaling` names. YaRN's omitted `beta_fast`/`beta_slow` take HF's
/// defaults (32 and 1); an unknown or absent type is plain RoPE.
fn rope_flavor(scaling: Option<&RopeScaling>, max_positions: usize) -> RopeFlavor<'_> {
    let Some(s) = scaling else {
        return RopeFlavor::Plain;
    };
    let original = s.original_max_position_embeddings;
    match s.rope_type.as_str() {
        "linear" => RopeFlavor::Linear { factor: s.factor },
        "dynamic" => RopeFlavor::DynamicNtk {
            factor: s.factor,
            original,
        },
        "yarn" => RopeFlavor::Yarn {
            factor: s.factor,
            original,
            beta_fast: s.beta_fast.unwrap_or(32.0),
            beta_slow: s.beta_slow.unwrap_or(1.0),
            attention_factor: s.attention_factor,
        },
        "llama3" => RopeFlavor::Llama3 {
            factor: s.factor,
            low_freq_factor: s.low_freq_factor,
            high_freq_factor: s.high_freq_factor,
            original,
        },
        "longrope" => RopeFlavor::LongRope {
            original,
            max_positions,
            short: s.short_factor.as_deref(),
            long: s.long_factor.as_deref(),
        },
        _ => RopeFlavor::Plain,
    }
}

/// The Runner's `rope.cos`/`rope.sin` tables, `[max_pos, rotary_dim]`, from
/// [`poot_graph_ir::rope_table::rope_tables`]: the table's capacity is the model's whole context.
pub(crate) fn rope_tables(
    cfg: &Qwen2Config,
    theta: f32,
    scaling: Option<&RopeScaling>,
    freq_factors: Option<&[f32]>,
) -> (HostTensor, HostTensor) {
    let (d, p) = (cfg.rotary_dim, cfg.max_pos);
    let RopeTables { cos, sin } =
        poot_graph_ir::rope_table::rope_tables(d, p, theta, &rope_flavor(scaling, p), freq_factors);
    (
        HostTensor::f32(vec![p, d], cos),
        HostTensor::f32(vec![p, d], sin),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::gguf::gguf_config;
    use crate::error::RunnerError;
    use poot_load::gguf::{GgufValue, write_gguf};

    /// A one-layer GGUF whose only variable is `{arch}.rope.freq_base` (absent when `None`).
    fn gguf_with_base(arch: &str, base: Option<f32>) -> GgufIndex {
        let keys: Vec<(String, GgufValue)> = [
            ("embedding_length", 8),
            ("block_count", 1),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 2),
            ("feed_forward_length", 16),
            ("context_length", 8),
        ]
        .into_iter()
        .map(|(k, v)| (format!("{arch}.{k}"), GgufValue::U32(v)))
        .chain(base.map(|b| (format!("{arch}.rope.freq_base"), GgufValue::F32(b))))
        .collect();
        let mut kvs = vec![
            ("general.architecture", GgufValue::Str(arch.into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(["a", "b"].map(|s| GgufValue::Str(s.into())).into()),
            ),
        ];
        kvs.extend(keys.iter().map(|(k, v)| (k.as_str(), v.clone())));
        GgufIndex::from_bytes(&write_gguf(&kvs, &[])).expect("parse fixture gguf")
    }

    fn tables(arch: &str, base: Option<f32>) -> Result<(Qwen2Config, HostTensor, HostTensor)> {
        let g = gguf_with_base(arch, base);
        let cfg = gguf_config(&g, arch, false).expect("fixture config");
        let store = poot_quant::weights::WeightStore::default();
        let (cos, sin) = gguf_rope_tables(&g, &store, &cfg, arch)?;
        Ok((cfg, cos, sin))
    }

    /// The typed cause behind the `Context` wrapper `gguf_rope_tables` returns.
    fn invalid_base(e: RunnerError) -> InvalidRopeBase {
        let RunnerError::Context { source, .. } = e else {
            panic!("expected a Context-wrapped InvalidRopeBase, got {e}");
        };
        *source
            .downcast::<InvalidRopeBase>()
            .unwrap_or_else(|s| panic!("cause is not InvalidRopeBase: {s}"))
    }

    /// A missing `freq_base` takes the reference default (llama 1e4, qwen2 1e6), so the table is finite and
    /// its position-1 angle at frequency index 1 is `1 * base^(-2/d)`, computed here in f64.
    #[test]
    fn missing_freq_base_yields_the_arch_default_table() {
        for (arch, default_base) in [("llama", 10_000.0f64), ("qwen2", 1_000_000.0)] {
            let (cfg, cos, sin) =
                tables(arch, None).expect("missing base is the default, not a fault");
            let d = cfg.rotary_dim;
            assert!(
                cos.as_f32()
                    .unwrap()
                    .iter()
                    .chain(sin.as_f32().unwrap().iter())
                    .all(|v| v.is_finite()),
                "{arch}: a default-base table must be finite"
            );
            let ang = default_base.powf(-2.0 / d as f64);
            let (got_c, got_s) = (
                cos.as_f32().unwrap()[d + 1] as f64,
                sin.as_f32().unwrap()[d + 1] as f64,
            );
            assert!(
                (got_c - ang.cos()).abs() < 1e-6 && (got_s - ang.sin()).abs() < 1e-6,
                "{arch}: pos 1 j 1 is (cos, sin) = ({got_c}, {got_s}), expected ({}, {})",
                ang.cos(),
                ang.sin()
            );
        }
    }

    /// A present base that is 0, negative or non-finite is a checkpoint fault: typed error, never a table.
    #[test]
    fn invalid_freq_base_is_a_typed_error_not_a_nan_table() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let e = tables("llama", Some(bad)).expect_err("an invalid base must not build a table");
            let cause = invalid_base(e);
            assert_eq!(cause.key, "llama.rope.freq_base");
            assert_eq!(cause.value.to_bits(), bad.to_bits());
        }
    }
}
