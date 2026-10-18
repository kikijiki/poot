//! Resolving a checkpoint's `quantization_config` into a supported [`QuantScheme`], or a typed refusal.
//!
//! The serde shape of the block lives in [`quant`]; model hyperparameters are not read here.

use super::LoadError;
pub mod quant;
pub use quant::{QuantConfig, QuantKind, QuantScheme};
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
                if let Some(format) = self.checkpoint_format.as_deref().filter(|f| *f != "gptq") {
                    return Err(LoadError::UnsupportedQuantConfig {
                        field: "checkpoint_format",
                        value: format.to_string(),
                    });
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
                if self.zero_point == Some(false) {
                    return Err(LoadError::UnsupportedQuantConfig {
                        field: "zero_point",
                        value: "false".to_string(),
                    });
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
            // HF `quant_method: "fp8"` (Qwen3.5-FP8, DeepSeek-V3): E4M3 weights with one `weight_scale_inv`
            // per 128x128 block, the one block `poot_quant`'s `E4m3Block128` describes.
            "fp8" => {
                if self.weight_block_size != Some([128, 128]) {
                    return Err(LoadError::SafeTensors(format!(
                        "unsupported fp8 weight_block_size {:?}; only [128, 128]",
                        self.weight_block_size
                    )));
                }
                Ok(QuantScheme {
                    kind: QuantKind::Fp8,
                    group_size,
                })
            }
            other => Err(LoadError::SafeTensors(format!(
                "unsupported quant_method {other:?} (gptq, awq, compressed-tensors, fp8 today)"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme_of(block: serde_json::Value) -> Result<QuantScheme, LoadError> {
        // Through the production entry: a whole `config.json` with a `quantization_config` block.
        quant_scheme(&serde_json::json!({"quantization_config": block}))
            .map(|scheme| scheme.expect("the block names a scheme"))
    }

    fn refused(result: Result<QuantScheme, LoadError>, field: &str, value: &str) {
        match result {
            Err(LoadError::UnsupportedQuantConfig { field: f, value: v }) => {
                assert_eq!((f, v.as_str()), (field, value));
            }
            other => panic!("expected a refusal naming {field} = {value}, got {other:?}"),
        }
    }

    /// POOT-1196 SC-001: a GPTQModel `gptq_v2` checkpoint is refused by field and value; v1 (named or
    /// absent) loads.
    #[test]
    fn gptq_v2_is_refused_by_checkpoint_format() {
        let block = |extra: serde_json::Value| {
            let mut block = serde_json::json!({"quant_method": "gptq", "bits": 4, "sym": true});
            block
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            block
        };
        refused(
            scheme_of(block(serde_json::json!({"checkpoint_format": "gptq_v2"}))),
            "checkpoint_format",
            "gptq_v2",
        );
        scheme_of(block(serde_json::json!({"checkpoint_format": "gptq"}))).expect("v1 named");
        scheme_of(block(serde_json::json!({}))).expect("v1 by absence");
    }

    /// POOT-1196 SC-002: AWQ without zero points is refused by name; with them (or by default) it loads.
    #[test]
    fn awq_without_zero_points_is_refused() {
        let block = |zero_point: serde_json::Value| serde_json::json!({"quant_method": "awq", "bits": 4, "version": "gemm", "zero_point": zero_point});
        refused(scheme_of(block(false.into())), "zero_point", "false");
        scheme_of(block(true.into())).expect("zero_point true");
        scheme_of(serde_json::json!({"quant_method": "awq", "bits": 4, "version": "gemm"}))
            .expect("zero_point absent");
    }
}
