//! Dense weight materialization from a [`WeightStore`] into [`HostTensor`] (card 543, the GGUF/
//! safetensors-neutral half of card 540b's loading design): both a GGUF- and a safetensors-sourced
//! dense entry reach a `HostTensor` through this one path. Lives here, not in `poot-load`, because
//! the target [`HostTensor`] is this layer's carrier; the stored-dtype widening is
//! [`DenseWeight::decode_f32`], owned by `poot-quant` beside the scalar decoders.

use poot_quant::weights::{DenseWeight, UndecodableDtype, WeightEntry, WeightStore};
use poot_tensor::{CarrierError, DType, HostTensor};

/// Why a stored dense tensor did not materialize.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MaterializeError {
    #[error("missing tensor {name}")]
    Missing { name: String },
    #[error("{name}: expected a dense stored tensor, found a packed entry")]
    Packed { name: String },
    #[error("dense {dtype} tensor {name}: {source}")]
    Carrier {
        name: String,
        dtype: DType,
        source: CarrierError,
    },
    #[error("tensor {name}: {source}")]
    Undecodable {
        name: String,
        source: UndecodableDtype,
    },
}

/// A stored dense tensor as its own dtype: a BF16/F16 entry keeps its stored words (the GPU
/// zero-copy upload form, never widened here), every other float dtype is decoded to F32.
fn dense_tensor(name: &str, dense: &DenseWeight) -> Result<HostTensor, MaterializeError> {
    match dense.dtype() {
        dtype @ (DType::BF16 | DType::F16) => {
            HostTensor::from_le_bytes(dtype, dense.shape().to_vec(), dense.bytes().as_slice())
                .map_err(|source| MaterializeError::Carrier {
                    name: name.to_string(),
                    dtype,
                    source,
                })
        }
        _ => {
            let data = dense
                .decode_f32()
                .map_err(|source| MaterializeError::Undecodable {
                    name: name.to_string(),
                    source,
                })?;
            Ok(HostTensor::f32(dense.shape().to_vec(), data))
        }
    }
}

/// Materialize one dense stored tensor from `store` into a [`HostTensor`] of its stored dtype (BF16
/// and F16 as words, every other float dtype decoded to F32). This is the one place a checkpoint
/// consumer turns a [`WeightStore`] entry into the shape every existing transform helper
/// (`transpose2d`, `row_slice`, ...) already operates on.
pub fn materialize_dense(store: &WeightStore, name: &str) -> Result<HostTensor, MaterializeError> {
    match store.get(name) {
        Some(WeightEntry::Dense(dense)) => dense_tensor(name, dense),
        Some(WeightEntry::Packed(_)) => Err(MaterializeError::Packed {
            name: name.to_string(),
        }),
        None => Err(MaterializeError::Missing {
            name: name.to_string(),
        }),
    }
}
