use super::super::*;

/// The `quantization_config` block. GPTQ 4-bit (sym, no act-order) and AWQ 4-bit (GEMM) are consumed
/// today (specs 019-020).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantConfig {
    pub quant_method: String,
    #[serde(default)]
    pub bits: Option<u32>,
    #[serde(default)]
    pub group_size: Option<usize>,
    #[serde(default)]
    pub desc_act: Option<bool>,
    #[serde(default)]
    pub sym: Option<bool>,
    #[serde(default)]
    pub version: Option<String>,
    /// compressed-tensors: `"float-quantized"` for FP8.
    #[serde(default)]
    pub format: Option<String>,
    /// compressed-tensors groups are keyed by a non-semantic label. Resident E4M3 admission
    /// validates the single group's published JSON contract; the default dense loader is unchanged.
    #[serde(default)]
    pub config_groups: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub ignore: Option<Vec<String>>,
    /// DeepSeek's `quant_method: "fp8"` block-wise scheme (spec 277/281, distinct from
    /// compressed-tensors' per-output-channel scheme): `[block_out, block_in]`, e.g. `[128, 128]`
    /// on `deepseek-ai/DeepSeek-V3.2-Exp` and `deepseek-ai/DeepSeek-V4-Flash-0731`, the one block
    /// `poot_quant`'s packed `E4m3Block128` format describes.
    #[serde(default)]
    pub weight_block_size: Option<[usize; 2]>,
}

/// Which 4-bit weight packing a quantized checkpoint uses (they differ in axis order, bit
/// interleave, and zero-point convention; see `dequant_gptq` / `dequant_awq`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Gptq,
    Awq,
    /// FP8 E4M3, per-output-channel scale (compressed-tensors, float-quantized).
    Fp8,
}

/// A resolved, supported quantization scheme (the kind + its group size).
#[derive(Debug, Clone, Copy)]
pub struct QuantScheme {
    pub kind: QuantKind,
    pub group_size: usize,
}
