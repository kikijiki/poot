//! Dense weight materialization from a [`WeightStore`] into [`HostTensor`] (card 543, the GGUF/
//! safetensors-neutral half of card 540b's loading design): both a GGUF- and a safetensors-sourced
//! dense entry reach a `HostTensor` through this one path. Lives here, not in `poot-load`, because
//! the byte-level lookup/decode (`poot_load::safetensors::dense_bytes`/`decode_dense`) stays in
//! `poot-load` since other stored-dtype dequantizers there (GPTQ/AWQ/FP8) share it.

use poot_tensor::{DType, HostTensor};

use poot_load::LoadError;
use poot_load::safetensors::{decode_dense, dense_bytes};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};

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

/// Like [`materialize_dense`], but removes the entry from `store` first: once the returned
/// [`HostTensor`] (or the caller's transform of it) is the only thing referencing this tensor's
/// bytes, the store's own copy is gone rather than staying resident until the whole store drops. A
/// checkpoint loader that reads every tensor exactly once (`build_weights`'s generic loop, e.g.)
/// uses this so the store's footprint shrinks as it drains, keeping peak host RSS near one file-size
/// instead of file-size-plus-final-weight-map-size (card 540b's M4 RSS gate, SC-001).
pub fn take_dense(store: &mut WeightStore, name: &str) -> Result<HostTensor, LoadError> {
    let dense = match store.remove(name) {
        Some(WeightEntry::Dense(dense)) => dense,
        Some(WeightEntry::Packed(_)) => {
            return Err(LoadError::SafeTensors(format!(
                "{name}: expected a dense stored tensor, found a packed entry"
            )));
        }
        None => return Err(LoadError::SafeTensors(format!("missing tensor {name}"))),
    };
    dense_tensor(&dense)
}
