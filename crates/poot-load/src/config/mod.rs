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
