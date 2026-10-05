//! GGUF (qwen2-family) config and weights builders. Every tensor read goes through the one reader
//! (`poot_load::gguf::{GgufIndex, read_gguf}`, card 543, dquant.md 8.1) into a `WeightStore`; family
//! code here reads stored bytes from it (`poot_eval::materialize_dense`), never widening or
//! transcoding.

use poot_load::gguf::GgufIndex;
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::WeightStore;
use poot_tensor::DType;

use crate::core::runner::deepseek2_yarn_resolve;
use crate::error::{OptionExt, Result};

/// A required u32 GGUF metadata value.
pub(crate) fn gguf_u32(g: &GgufIndex, key: &str) -> Result<u32> {
    g.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .with_context(|| format!("gguf missing {key}"))
}

pub(crate) fn gguf_f32(g: &GgufIndex, key: &str) -> Result<f32> {
    g.get(key)
        .and_then(|v| v.as_f32())
        .with_context(|| format!("gguf missing {key}"))
}

/// The GGUF tensor names of layer 0's seven projection weights: the set the qwen2-family tracers type with
/// [`Qwen2Config::proj_dtype`] (`poot-models/src/qwen2.rs`: q/k/v/o plus gate/up/down). Kept next to
/// [`gguf_proj_dtype`], its only caller.
const GGUF_LAYER0_PROJECTIONS: [&str; 7] = [
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
];

/// The GGUF half of `Runner::load`'s `proj_dtype` auto-select.
///
/// `Runner::load` probes the safetensors checkpoint's layer-0 q_proj dtype and sets `cfg.proj_dtype =
/// BF16/F16` when the checkpoint is natively 16-bit, so the tracers type those weight constants narrow and
/// backends with native support can upload the two-byte payload instead of a widened f32 mirror. This does
/// the same for natively-F16/BF16 GGUF weights (kept by `read_gguf` as stored bytes, carried through `transpose2d`).
///
/// Stricter than the safetensors probe: mixed per-tensor quant schemes are common in GGUF, so every one of
/// [`GGUF_LAYER0_PROJECTIONS`] must have the same native F16 or BF16 type before a narrow dtype is
/// selected. A missing name (a fused-qkv arch like phi3, a MoE arch whose FFN lives in `ffn_*_exps`), a
/// packed (quantized) tensor, or any mixed/non-native member leaves `DType::F32`. A BF16/F16 mixture
/// cannot share one `proj_dtype`.
pub(crate) fn gguf_proj_dtype(store: &WeightStore) -> DType {
    use poot_quant::weights::WeightEntry;
    let dtype_of = |name: &str| match store.get(name) {
        Some(WeightEntry::Dense(dense)) => Some(dense.dtype()),
        _ => None,
    };
    if GGUF_LAYER0_PROJECTIONS
        .iter()
        .all(|name| dtype_of(name) == Some(DType::F16))
    {
        DType::F16
    } else if GGUF_LAYER0_PROJECTIONS
        .iter()
        .all(|name| dtype_of(name) == Some(DType::BF16))
    {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Build a [`Qwen2Config`] from a GGUF's metadata. GGUF hyperparams are namespaced under the arch string
/// (e.g. `qwen2.embedding_length`, `llama.embedding_length`), so everything keys off `arch`.
pub(crate) fn gguf_config(g: &GgufIndex, arch: &str, qkv_bias: bool) -> Result<Qwen2Config> {
    let k = |s: &str| format!("{arch}.{s}");
    let hidden = gguf_u32(g, &k("embedding_length"))? as usize;
    let n_heads = gguf_u32(g, &k("attention.head_count"))? as usize;
    // head_dim is explicit for archs where it differs from hidden/n_heads (e.g. gemma3: key_length=256 vs
    // 1152/4=288); otherwise hidden/n_heads (qwen2/llama). The GGUF path is full-rotary.
    let head_dim = g
        .get(&k("attention.key_length"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden / n_heads);
    // phi3 carries an explicit rotary width (`phi3.rope.dimension_count`); partial rotary when < head_dim.
    // qwen2/llama omit it (full rotary). A GGUF with rope.dimension_count < head_dim would also carry LongRoPE
    // factors (rope_factors_* tensors), which are not consumed yet.
    let rotary_dim = g
        .get(&k("rope.dimension_count"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    let vocab = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .context("gguf missing tokenizer.ggml.tokens")?
        .len();
    Ok(Qwen2Config {
        vocab,
        hidden,
        inter: gguf_u32(g, &k("feed_forward_length"))? as usize,
        layers: gguf_u32(g, &k("block_count"))? as usize,
        n_heads,
        n_kv_heads: gguf_u32(g, &k("attention.head_count_kv"))? as usize,
        head_dim,
        rotary_dim,
        eps: g
            .get(&k("attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6),
        max_pos: gguf_u32(g, &k("context_length"))? as usize,
        qkv_bias,
        // gemma3/qwen3/qwen3moe (per-head, over head_dim) and olmo2/olmoe (full-projection, over q_dim/kv_dim)
        // apply RMSNorm to q and k before RoPE; qwen2/llama/phi3 do not. qwen3moe has the same per-head QK-norm as
        // qwen3 dense (llama.cpp's `src/models/qwen3moe.cpp` creates attn_q_norm/attn_k_norm at `{n_embd_head_k}`
        // for every layer). olmoe uses olmo2's full-dimension convention (`poot_models::olmoe`; llama.cpp's
        // `src/models/olmoe.cpp` applies `attn_q_norm`/`attn_k_norm` before the head reshape).
        qk_norm: arch == "gemma3"
            || arch == "qwen3"
            || arch == "qwen3moe"
            || arch == "olmo2"
            || arch == "olmoe",
        // Sliding-window attention size (Mistral/Qwen2-family SWA variants): `{arch}.attention.sliding_window`,
        // a u32. Absent (most archs, or a non-SWA checkpoint) -> None (full causal).
        sliding_window: g
            .get(&k("attention.sliding_window"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        ..Default::default()
    })
}

pub mod arch_config;
pub mod deepseek;
pub mod experts;
pub mod permute;
pub mod rope;

pub(crate) use permute::{row_slice, transpose_experts, transpose2d};

/// Resolve DeepSeek-V2's YaRN RoPE-scaling parameters from a GGUF's `deepseek2.rope.scaling.*` metadata
/// into a [`poot_models::deepseek2::DeepseekV2Yarn`], mirroring `runner.rs`'s `deepseek2_yarn_params`
/// (safetensors/HF path) but reading GGUF keys (llama.cpp's
/// `conversion/deepseek.py::DeepseekV2Model.set_gguf_parameters` and
/// `src/models/deepseek2.cpp::load_arch_hparams`):
/// - `deepseek2.rope.scaling.type` == `"yarn"` gates this (else `None`, plain RoPE).
/// - `deepseek2.rope.scaling.{factor,original_context_length,yarn_beta_fast,yarn_beta_slow}` are the generic
///   keys `gguf_rope_tables`'s linear/YaRN branch reads for other archs.
/// - `deepseek2.rope.scaling.attn_factor` is unlikely to be present in a real DeepSeek-V2 GGUF (the base
///   converter writes it only from `rope_params.get("attention_factor")`, and real configs set
///   `mscale`/`mscale_all_dim` instead), so resolution usually falls through to the ratio-based path below.
///   It is still read as an explicit override, for parity with `deepseek2_yarn_params`.
/// - `deepseek2.rope.scaling.yarn_log_multiplier` is DeepSeek-V2's own key (`[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`
///   in `conversion/deepseek.py`; the generic converter's write is commented out in `conversion/base.py`).
///   `DeepseekV2Model.set_gguf_parameters` writes `0.1 * mscale_all_dim`, and `src/models/deepseek2.cpp`
///   divides by `0.1` to recover it, as this function does. GGUF has one `yarn_log_multiplier` key derived
///   from `mscale_all_dim` alone; real DeepSeek-V2 configs set `mscale == mscale_all_dim` (0.707), so the
///   recovered value is passed as both to the shared resolution core, giving `attention_factor == 1.0` as
///   the real checkpoint does.
pub(crate) fn gguf_deepseek2_yarn_params(
    g: &GgufIndex,
    max_pos: usize,
) -> Option<poot_models::deepseek2::DeepseekV2Yarn> {
    let scaling_type = g
        .get("deepseek2.rope.scaling.type")
        .and_then(|v| v.as_str());
    if scaling_type != Some("yarn") {
        return None;
    }
    let factor = g
        .get("deepseek2.rope.scaling.factor")
        .and_then(|v| v.as_f32())
        .unwrap_or(1.0);
    let orig = g
        .get("deepseek2.rope.scaling.original_context_length")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(max_pos);
    let beta_fast = g
        .get("deepseek2.rope.scaling.yarn_beta_fast")
        .and_then(|v| v.as_f32())
        .unwrap_or(32.0);
    let beta_slow = g
        .get("deepseek2.rope.scaling.yarn_beta_slow")
        .and_then(|v| v.as_f32())
        .unwrap_or(1.0);
    let attention_factor = g
        .get("deepseek2.rope.scaling.attn_factor")
        .and_then(|v| v.as_f32());
    let mscale_all_dim = g
        .get("deepseek2.rope.scaling.yarn_log_multiplier")
        .and_then(|v| v.as_f32())
        .map(|v| v / 0.1);
    Some(deepseek2_yarn_resolve(
        factor,
        beta_fast,
        beta_slow,
        orig,
        attention_factor,
        mscale_all_dim,
        mscale_all_dim,
    ))
}

#[cfg(test)]
mod tests;
