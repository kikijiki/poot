//! Dense weight materialization from a [`WeightStore`] into [`HostTensor`] (card 543, the GGUF/
//! safetensors-neutral half of card 540b's loading design): both a GGUF- and a safetensors-sourced
//! dense entry reach a `HostTensor` through this one path. Lives here, not in `poot-load`, because
//! the byte-level lookup/decode (`poot_load::safetensors::dense_bytes`/`decode_dense`) stays in
//! `poot-load` since other stored-dtype dequantizers there (GPTQ/AWQ/FP8) share it.

use poot_tensor::{DType, HostTensor};

use poot_load::LoadError;
use poot_load::safetensors::{decode_dense, dense_bytes};
use poot_quant::weights::{DenseWeight, WeightStore};

/// A stored dense tensor as its own dtype: a BF16/F16 entry keeps its stored words (the GPU
/// zero-copy upload form, never widened here), every other float dtype is decoded to F32.
fn dense_tensor(dense: &DenseWeight) -> Result<HostTensor, LoadError> {
    match dense.dtype() {
        dtype @ (DType::BF16 | DType::F16) => {
            HostTensor::from_le_bytes(dtype, dense.shape().to_vec(), dense.bytes().as_slice())
                .map_err(|error| LoadError::SafeTensors(format!("dense {dtype} tensor: {error}")))
        }
        _ => {
            let (shape, data) = decode_dense(dense)?;
            Ok(HostTensor::f32(shape, data))
        }
    }
}

/// Materialize one dense stored tensor from `store` into a [`HostTensor`] of its stored dtype (BF16
/// and F16 as words, every other float dtype decoded to F32). This is the one place a checkpoint
/// consumer turns a [`WeightStore`] entry into the shape every existing transform helper
/// (`transpose2d`, `row_slice`, ...) already operates on.
pub fn materialize_dense(store: &WeightStore, name: &str) -> Result<HostTensor, LoadError> {
    dense_tensor(dense_bytes(store, name)?)
}
