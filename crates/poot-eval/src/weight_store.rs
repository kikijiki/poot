//! Dense weight materialization from a [`WeightStore`] into [`HostTensor`] (card 543, the GGUF/
//! safetensors-neutral half of card 540b's loading design): both a GGUF- and a safetensors-sourced
//! dense entry reach a `HostTensor` through this one path. Lives here, not in `poot-load`, because
//! the target [`HostTensor`] is this layer's carrier; the stored-dtype widening is
//! [`DenseWeight::decode_f32`], owned by `poot-quant` beside the scalar decoders.

#[cfg(any(test, feature = "test-support"))]
use poot_graph_ir::ValueMeta;
use poot_quant::weights::{DenseWeight, UndecodableDtype, WeightEntry, WeightStore};
use poot_tensor::{CarrierError, DType, HostTensor};

/// Why a stored dense tensor did not materialize.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MaterializeError {
    /// The store has no tensor of that name.
    #[error("missing tensor {name}")]
    Missing {
        /// The requested tensor name.
        name: String,
    },
    /// The entry is packed, not dense.
    #[error("{name}: expected a dense stored tensor, found a packed entry")]
    Packed {
        /// The tensor name.
        name: String,
    },
    /// The stored bytes did not form a host tensor of the stored dtype.
    #[error("dense {dtype} tensor {name}: {source}")]
    Carrier {
        /// Tensor name.
        name: String,
        /// Stored dtype.
        dtype: DType,
        /// The carrier's refusal.
        source: CarrierError,
    },
    /// A weight the graph declares F16 over the checkpoint's BF16 holds, at element `index`, a value f16 cannot
    /// hold: the bind refuses it the same way.
    #[error("{name}: element {index} (bf16 bits {bits:#06x}) is outside f16's finite range")]
    PastF16Range {
        /// The tensor name.
        name: String,
        /// The element's row-major index.
        index: usize,
        /// The element's stored bf16 bits.
        bits: u16,
    },
    /// The stored dtype has no f32 decoding.
    #[error("tensor {name}: {source}")]
    Undecodable {
        /// Tensor name.
        name: String,
        /// The decoder's refusal.
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

/// The const `meta` declares, materialized from `store` as the graph declares it: the stored dense tensor,
/// converted where the graph converted the weight ([`declared_weight`]).
#[cfg(any(test, feature = "test-support"))]
pub fn materialize_declared(
    store: &WeightStore,
    meta: &ValueMeta,
) -> Result<HostTensor, MaterializeError> {
    let name = meta.name().expect("a const input carries its name");
    declared_weight(&name, meta, materialize_dense(store, &name)?)
}

/// `stored`, the checkpoint tensor of the const `name` that `meta` declares, as the graph declares it: unchanged
/// when the graph declares the stored dtype, else the one conversion the bind performs
/// ([`ValueMeta::stored_dtype`]): each bf16 element rounded to nearest even f16, a value f16 cannot hold refused.
#[cfg(any(test, feature = "test-support"))]
pub fn declared_weight(
    name: &str,
    meta: &ValueMeta,
    stored: HostTensor,
) -> Result<HostTensor, MaterializeError> {
    if meta.stored_dtype() == meta.aval.dtype {
        return Ok(stored);
    }
    let words = stored
        .as_half()
        .expect("the one weight conversion is BF16 words to F16 words");
    let mut converted = Vec::with_capacity(words.len());
    for (index, &bits) in words.iter().enumerate() {
        let half = poot_quant::scalar::f32_to_f16(poot_quant::scalar::bf16_to_f32(bits));
        if half & 0x7c00 == 0x7c00 {
            return Err(MaterializeError::PastF16Range {
                name: name.to_string(),
                index,
                bits,
            });
        }
        converted.push(half);
    }
    Ok(HostTensor::f16(stored.shape().to_vec(), converted))
}
