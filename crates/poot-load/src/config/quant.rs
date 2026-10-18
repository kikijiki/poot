//! The serde shape of a checkpoint's `quantization_config` block and the resolved scheme (weight
//! packing kind and group size) the loader consumes.

use std::collections::HashMap;
/// The `quantization_config` block. GPTQ 4-bit (sym, no act-order) and AWQ 4-bit (GEMM) are consumed
/// today (specs 019-020).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantConfig {
    /// The scheme family: `"gptq"`, `"awq"`, `"compressed-tensors"` or `"fp8"`.
    pub quant_method: String,
    /// Weight bit width (`4` for GPTQ and AWQ).
    #[serde(default)]
    pub bits: Option<u32>,
    /// Quantization group size along the input axis; `128` when absent.
    #[serde(default)]
    pub group_size: Option<usize>,
    /// GPTQ activation-order flag; `Some(true)` is refused.
    #[serde(default)]
    pub desc_act: Option<bool>,
    /// GPTQ symmetric-quantization flag; `Some(false)` is refused.
    #[serde(default)]
    pub sym: Option<bool>,
    /// GPTQ checkpoint convention (GPTQModel): `"gptq"` (the v1 zero `u4 + 1`, also what an absent key
    /// means) is the only one admitted; `"gptq_v2"` stores zeros the decoder would read one unit off.
    #[serde(default)]
    pub checkpoint_format: Option<String>,
    /// AWQ zero-point flag; `Some(false)` (symmetric, no `qzeros` bias) is refused.
    #[serde(default)]
    pub zero_point: Option<bool>,
    /// AWQ packing version; only `"gemm"` is admitted.
    #[serde(default)]
    pub version: Option<String>,
    /// compressed-tensors: `"float-quantized"` for FP8.
    #[serde(default)]
    pub format: Option<String>,
    /// compressed-tensors groups are keyed by a non-semantic label. Resident E4M3 admission
    /// validates the single group's published JSON contract; the default dense loader is unchanged.
    #[serde(default)]
    pub config_groups: Option<HashMap<String, serde_json::Value>>,
    /// Module names the checkpoint leaves unquantized (compressed-tensors).
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
    /// GPTQ 4-bit packing.
    Gptq,
    /// AWQ 4-bit GEMM packing.
    Awq,
    /// FP8 E4M3, per-output-channel scale (compressed-tensors, float-quantized).
    Fp8,
}

/// A resolved, supported quantization scheme (the kind + its group size).
#[derive(Debug, Clone, Copy)]
pub struct QuantScheme {
    /// Which packing the checkpoint uses.
    pub kind: QuantKind,
    /// Quantization group size along the input axis.
    pub group_size: usize,
}
