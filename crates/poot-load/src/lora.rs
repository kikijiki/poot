//! LoRA adapter loading (spec 248, epic 129 B8): the PEFT/HF adapter format,
//! `adapter_config.json` + `adapter_model.safetensors` next to (not merged into) a base
//! checkpoint. Format per PEFT's docs and source (citations in spec 248 "Adapter weight loading"):
//!
//! - State dict keys are `base_model.model.{module_path}.lora_A.weight` / `.lora_B.weight`, e.g.
//!   `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` for a causal LM. A
//!   single-adapter save strips the adapter name segment; a multi-adapter save keeps it
//!   (`lora_A.default.weight`, ...). This loader accepts either by locating the `lora_A`/`lora_B`
//!   segment and ignoring anything between it and the trailing `.weight`.
//! - `lora_A.weight` is `[r, in]` and `lora_B.weight` is `[out, r]` (bias-free `nn.Linear`, so the
//!   `[out_features, in_features]` convention of the base model's `.weight` tensors).
//! - `adapter_config.json` carries `r`, `lora_alpha`, `target_modules` (module name suffixes, e.g.
//!   `["q_proj","v_proj"]` or all seven attention+MLP projections), and `use_rslora`
//!   (`alpha/sqrt(r)` instead of `alpha/r`).
//!
//! Embedding-layer LoRA (`lora_embedding_A`/`lora_embedding_B`, a bare `nn.Parameter` pair with a
//! different shape convention) and DoRA's magnitude vector are out of scope (spec 248) and are
//! rejected with an error rather than ignored.

use crate::LoadError;
use crate::safetensors::dense_bytes;
use poot_quant::weights::{DenseWeight, WeightStore};
use poot_tensor::DType;
use std::collections::HashMap;
use std::path::Path;

/// The dense entry `name`, as the adapter file stored it: a float tensor of dtype F32, F16 or BF16.
fn factor(store: &WeightStore, name: &str) -> Result<DenseWeight, LoadError> {
    let dense = dense_bytes(store, name)?;
    match dense.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => Ok(dense.clone()),
        dtype => Err(LoadError::LoraFactorDtype {
            tensor: name.to_string(),
            dtype,
        }),
    }
}

/// The `adapter_config.json` fields this loader needs. PEFT's file has many training and
/// bookkeeping fields; `#[serde(deny_unknown_fields)]` is not set so they are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LoraAdapterConfig {
    /// Adapter rank `r` (LoRA's `A`/`B` inner dimension). Typical range 8-64.
    pub r: usize,
    /// PEFT's `lora_alpha`, the scale numerator (see `scaling`).
    pub lora_alpha: f32,
    /// Module name suffixes the adapter targets, e.g. `["q_proj","v_proj"]`, matched against a
    /// graph's per-projection weight prefix by suffix (`LoraAdapter::weights`), as PEFT matches
    /// `nn.Module` names. PEFT's single-regex-string form is unsupported (rejected by serde's type
    /// mismatch, spec 248).
    pub target_modules: Vec<String>,
    /// Rank-stabilized LoRA (Kalajdzievski 2023): scales by `alpha/sqrt(r)` instead of `alpha/r`.
    #[serde(default)]
    pub use_rslora: bool,
}

/// The [`LoraAdapterConfig`] of an `adapter_config.json`'s bytes. PEFT's per-module overrides
/// (`rank_pattern`, `alpha_pattern`) are not part of the struct, so they are read from the raw
/// document: a non-empty one is refused rather than silently flattened to the one `r`/`lora_alpha`.
fn parse_adapter_config(bytes: &[u8]) -> Result<LoraAdapterConfig, LoadError> {
    let document: serde_json::Value = serde_json::from_slice(bytes)?;
    for field in ["rank_pattern", "alpha_pattern"] {
        let overrides = document.get(field).filter(|value| !value.is_null());
        let empty = overrides.is_none_or(|value| value.as_object().is_some_and(|o| o.is_empty()));
        if !empty {
            return Err(LoadError::UnsupportedLoraConfig { field });
        }
    }
    Ok(serde_json::from_value(document)?)
}

impl LoraAdapterConfig {
    /// PEFT's correction scale: `lora_alpha / r` (standard), or `lora_alpha / sqrt(r)` under rslora.
    pub fn scaling(&self) -> f32 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f32).sqrt()
        } else {
            self.lora_alpha / self.r as f32
        }
    }
}

/// One target module's adapter weights exactly as the adapter file stored them, in their stored dtype
/// (F32, F16 or BF16; never widened): `a` is `[r, in]` and `b` is `[out, r]`, PEFT's `nn.Linear`
/// orientation. The driver stacks these per generation for the LoRA compile pass, reading them
/// transposed into poot's `x @ W` layout (`[in, r]` and `[r, out]`) as it copies them.
#[derive(Debug, Clone)]
pub struct LoraWeight {
    /// `[r, in]`.
    pub a: DenseWeight,
    /// `[out, r]`.
    pub b: DenseWeight,
}

/// A loaded LoRA adapter: its config plus each target module's `A`/`B` pair, keyed by
/// checkpoint module path with the `base_model.model.` prefix and
/// `.lora_A/.lora_B[.<adapter>].weight` suffix stripped (e.g. `model.layers.0.self_attn.q_proj`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// The adapter's `adapter_config.json`.
    pub config: LoraAdapterConfig,
    /// Each target module's `A`/`B` pair, keyed by module path.
    pub weights: HashMap<String, LoraWeight>,
}

impl LoraAdapter {
    /// Load `adapter_config.json` + `adapter_model.safetensors` from a PEFT adapter directory
    /// (`PeftModel.save_pretrained` output). The older pickle `adapter_model.bin` is not supported.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = parse_adapter_config(&std::fs::read(dir.join("adapter_config.json"))?)?;
        let store =
            crate::safetensors::load_weight_store_file(dir.join("adapter_model.safetensors"))?;
        Self::from_store(config, &store)
    }

    /// Parse an already-loaded [`WeightStore`] (the non-file-IO half of `load`, and the test
    /// entry point).
    pub fn from_store(config: LoraAdapterConfig, store: &WeightStore) -> Result<Self, LoadError> {
        let mut a_by_module: HashMap<String, String> = HashMap::new();
        let mut b_by_module: HashMap<String, String> = HashMap::new();
        for key in store.keys() {
            let key = key.as_str();
            // DoRA checkpoints carry full lora_A/lora_B tensors plus a magnitude vector (PEFT's
            // `layer.py::update_layer`), so the empty check below would not catch them. Reject
            // eagerly rather than load with the magnitude renormalization dropped.
            if is_dora_magnitude_key(key) {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {key} is a DoRA magnitude-vector tensor - DoRA adapters are not \
                     supported (the magnitude renormalization would be silently dropped) - see \
                     poot_load::lora module docs"
                )));
            }
            match parse_lora_key(key) {
                Some((module, LoraSide::A)) => {
                    a_by_module.insert(module, key.to_string());
                }
                Some((module, LoraSide::B)) => {
                    b_by_module.insert(module, key.to_string());
                }
                None => {} // not a lora_A/lora_B tensor (or an out-of-scope embedding key); skip
            }
        }
        if a_by_module.is_empty() {
            return Err(LoadError::SafeTensors(
                "lora adapter: no lora_A.*.weight tensors found (embedding-only or DoRA adapters are \
                 not supported - see poot_load::lora module docs)"
                    .into(),
            ));
        }
        let mut weights = HashMap::with_capacity(a_by_module.len());
        for (module, a_key) in a_by_module {
            let b_key = b_by_module.get(&module).ok_or_else(|| {
                LoadError::SafeTensors(format!("lora adapter: {module} has lora_A but no lora_B"))
            })?;
            let a = factor(store, &a_key)?;
            let b = factor(store, b_key)?;
            if a.shape().len() != 2 || b.shape().len() != 2 {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {module} lora_A/lora_B must be 2-D linear weights, got {:?}/{:?}",
                    a.shape(),
                    b.shape()
                )));
            }
            if a.shape()[0] != config.r || b.shape()[1] != config.r {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {module} lora_A shape {:?} / lora_B shape {:?} don't match config r={}",
                    a.shape(),
                    b.shape(),
                    config.r
                )));
            }
            weights.insert(module, LoraWeight { a, b });
        }
        Ok(LoraAdapter { config, weights })
    }
}

enum LoraSide {
    A,
    B,
}

/// Recognize a `lora_A`/`lora_B` state-dict key and return `(module_path, side)`, or `None` for
/// anything else (`lora_embedding_A/B`, a DoRA magnitude key, an unrelated tensor). `module_path`
/// has the `base_model.model.` prefix and any adapter-name segment stripped, so
/// `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` and
/// `...q_proj.lora_A.default.weight` both yield `model.layers.0.self_attn.q_proj`.
fn parse_lora_key(key: &str) -> Option<(String, LoraSide)> {
    let stripped = key.strip_prefix("base_model.model.")?;
    let parts: Vec<&str> = stripped.split('.').collect();
    if parts.last() != Some(&"weight") {
        return None;
    }
    let idx = parts.iter().position(|&p| p == "lora_A" || p == "lora_B")?;
    let side = if parts[idx] == "lora_A" {
        LoraSide::A
    } else {
        LoraSide::B
    };
    Some((parts[..idx].join("."), side))
}

/// True if `key` is a DoRA magnitude-vector tensor, `base_model.model.{module}.lora_magnitude_vector[.<adapter>]`
/// (a `nn.Parameter` with no trailing `.weight`, one per DoRA-adapted module). Kept separate from
/// `parse_lora_key` so `from_safetensors` can tell "not a lora tensor, skip" from "DoRA present,
/// refuse the adapter".
fn is_dora_magnitude_key(key: &str) -> bool {
    match key.strip_prefix("base_model.model.") {
        Some(stripped) => stripped.split('.').any(|p| p == "lora_magnitude_vector"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The little-endian bytes of `values` stored as `dtype` (F32 or BF16; the values the tests use are
    /// exactly representable in both).
    fn encode(dtype: &str, values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| match dtype {
                "BF16" => (v.to_bits() >> 16).to_le_bytes()[..2].to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect()
    }

    /// Minimal `adapter_model.safetensors` blob with one target module's `lora_A`/`lora_B` (r=2,
    /// in=4, out=3) stored as `dtype`, keyed with the PEFT causal-LM prefix (`base_model.model.model.layers...`) and
    /// a `.default.` adapter-name segment, so the stripped and named forms are shown to parse alike.
    fn synthetic_adapter_safetensors(dtype: &str) -> Vec<u8> {
        let (r, in_dim, out_dim) = (2usize, 4usize, 3usize);
        // lora_A [r,in] row-major, lora_B [out,r] row-major.
        let a_data: Vec<f32> = (0..r * in_dim).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..out_dim * r).map(|i| i as f32 * 0.5 + 1.0).collect();
        let a_bytes = encode(dtype, &a_data);
        let b_bytes = encode(dtype, &b_data);
        let mut data = a_bytes.clone();
        let b_off = data.len();
        data.extend_from_slice(&b_bytes);
        let end = data.len();
        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight";
        let key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight";
        let header = serde_json::json!({
            key_a: {"dtype": dtype, "shape": [r, in_dim], "data_offsets": [0, b_off]},
            key_b: {"dtype": dtype, "shape": [out_dim, r], "data_offsets": [b_off, end]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn parses_real_peft_causal_lm_key_shape() {
        let key = "base_model.model.model.layers.5.self_attn.v_proj.lora_A.weight";
        let (module, side) = parse_lora_key(key).expect("should parse");
        assert_eq!(module, "model.layers.5.self_attn.v_proj");
        assert!(matches!(side, LoraSide::A));
    }

    #[test]
    fn parses_named_adapter_segment() {
        let key = "base_model.model.model.layers.5.self_attn.v_proj.lora_B.default.weight";
        let (module, side) = parse_lora_key(key).expect("should parse");
        assert_eq!(module, "model.layers.5.self_attn.v_proj");
        assert!(matches!(side, LoraSide::B));
    }

    #[test]
    fn rejects_embedding_and_unrelated_keys() {
        assert!(parse_lora_key("base_model.model.model.embed_tokens.lora_embedding_A").is_none());
        assert!(parse_lora_key("model.layers.0.self_attn.q_proj.weight").is_none());
    }

    /// The adapter keeps the factors in the layout and dtype the file stored them: `lora_A` `[r, in]`,
    /// `lora_B` `[out, r]`, the file's own bytes. BF16 stays BF16 (POOT-1258 SC-001); nothing is widened.
    #[test]
    fn factors_keep_the_stored_layout_and_dtype() {
        let (r, in_dim, out_dim) = (2usize, 4usize, 3usize);
        for (dtype, expected) in [("F32", DType::F32), ("BF16", DType::BF16)] {
            let store = crate::safetensors::test_only::load_weight_store_bytes(
                &synthetic_adapter_safetensors(dtype),
            )
            .unwrap();
            let config = LoraAdapterConfig {
                r,
                lora_alpha: 8.0,
                target_modules: vec!["q_proj".into()],
                use_rslora: false,
            };
            let adapter = LoraAdapter::from_store(config, &store).unwrap();
            assert_eq!(adapter.config.scaling(), 4.0); // alpha/r = 8/2

            let w = adapter
                .weights
                .get("model.layers.0.self_attn.q_proj")
                .expect("q_proj adapter present");
            assert_eq!(w.a.dtype(), expected, "{dtype}: lora_A dtype");
            assert_eq!(w.b.dtype(), expected, "{dtype}: lora_B dtype");
            assert_eq!(w.a.shape(), [r, in_dim]);
            assert_eq!(w.b.shape(), [out_dim, r]);
            let a_values: Vec<f32> = (0..r * in_dim).map(|i| i as f32 * 0.5).collect();
            assert_eq!(
                w.a.bytes().as_slice(),
                encode(dtype, &a_values).as_slice(),
                "{dtype}: lora_A bytes are the file's own"
            );
            assert!(
                !adapter
                    .weights
                    .contains_key("model.layers.0.self_attn.k_proj")
            );
        }
    }

    /// A non-float adapter tensor (an I32 file) is refused at load, naming the tensor and dtype.
    #[test]
    fn a_non_float_factor_is_refused() {
        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let header = serde_json::json!({
            key_a: {"dtype": "I32", "shape": [2, 4], "data_offsets": [0, 32]},
            key_b: {"dtype": "I32", "shape": [3, 2], "data_offsets": [32, 56]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&[0u8; 56]);
        let store = crate::safetensors::test_only::load_weight_store_bytes(&blob).unwrap();
        let config = LoraAdapterConfig {
            r: 2,
            lora_alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        let error = LoraAdapter::from_store(config, &store).unwrap_err();
        assert!(
            matches!(
                &error,
                LoadError::LoraFactorDtype {
                    dtype: DType::I32,
                    ..
                }
            ),
            "{error}"
        );
    }

    /// POOT-1196 SC-006: a non-empty `alpha_pattern` or `rank_pattern` is refused by name; empty ones
    /// (PEFT writes `{}`) load.
    #[test]
    fn per_module_patterns_are_refused() {
        let config = |extra: serde_json::Value| {
            let mut doc = serde_json::json!({
                "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
                "alpha_pattern": {}, "rank_pattern": {}
            });
            doc.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            parse_adapter_config(&serde_json::to_vec(&doc).unwrap())
        };
        assert!(config(serde_json::json!({})).is_ok());
        for field in ["alpha_pattern", "rank_pattern"] {
            let error = config(serde_json::json!({ field: {"q_proj": 4} })).unwrap_err();
            assert!(
                matches!(&error, LoadError::UnsupportedLoraConfig { field: f } if *f == field),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn rslora_scaling_divides_by_sqrt_r() {
        let config = LoraAdapterConfig {
            r: 16,
            lora_alpha: 16.0,
            target_modules: vec![],
            use_rslora: true,
        };
        assert!((config.scaling() - 4.0).abs() < 1e-6); // 16/sqrt(16) = 4
    }

    #[test]
    fn dora_magnitude_vector_key_is_rejected() {
        // A DoRA checkpoint has full lora_A/lora_B tensors plus a lora_magnitude_vector per adapted
        // module (PEFT's `layer.py::update_layer`). The loader must refuse it rather than drop the
        // magnitude renormalization.
        let (r, in_dim, out_dim, out_features) = (2usize, 4usize, 3usize, 3usize);
        let a_data: Vec<f32> = (0..r * in_dim).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..out_dim * r).map(|i| i as f32 * 0.1 + 1.0).collect();
        let mag_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.01 + 1.0).collect();

        let mut data = Vec::new();
        let a_off = data.len();
        for v in &a_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let b_off = data.len();
        for v in &b_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mag_off = data.len();
        for v in &mag_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();

        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let key_mag =
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default";
        let header = serde_json::json!({
            key_a: {"dtype": "F32", "shape": [r, in_dim], "data_offsets": [a_off, b_off]},
            key_b: {"dtype": "F32", "shape": [out_dim, r], "data_offsets": [b_off, mag_off]},
            key_mag: {"dtype": "F32", "shape": [out_features], "data_offsets": [mag_off, end]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&data);

        let store = crate::safetensors::test_only::load_weight_store_bytes(&blob).unwrap();
        let config = LoraAdapterConfig {
            r,
            lora_alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        let err = LoraAdapter::from_store(config, &store).expect_err(
            "a DoRA checkpoint (lora_A/lora_B present PLUS lora_magnitude_vector) must error, \
                         not silently load without the magnitude renormalization",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("DoRA"),
            "error should name DoRA as the reason, got: {msg}"
        );
    }

    #[test]
    fn missing_lora_b_is_an_error() {
        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let header = serde_json::json!({
            key_a: {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&[0u8; 32]);
        let store = crate::safetensors::test_only::load_weight_store_bytes(&blob).unwrap();
        let config = LoraAdapterConfig {
            r: 2,
            lora_alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        assert!(LoraAdapter::from_store(config, &store).is_err());
    }
}
