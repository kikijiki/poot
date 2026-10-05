//! Host bytes into a planned device lane: the one encoder every device shares, so no backend
//! re-derives how a stored dtype lands in the storage the plan chose (ADR-0103: bytes as stored where
//! the plan keeps them, the plan's widening where it does not).

use poot_target::{BufferStorage, ElementKind, LogicalDType};
use poot_tensor::DType;

/// Why a stored dense weight cannot land in its planned lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EncodeError {
    /// The stored dtype has no encoding into the planned storage.
    NoEncoding,
    /// Card 1007: a packed-F16 weight holds an infinity or NaN (exponent `0x1f`) at element `index`. The
    /// generated bodies decode packed binary16 over its finite domain only (the packed emitter's admitted
    /// domain), so such a weight is refused at upload rather than decoded to a wrong finite value.
    NonFiniteF16 { index: usize, bits: u16 },
}

/// A stored dense weight's bytes as `planned` holds them. An F16 weight a contraction reads lands as stored,
/// natively or packed two elements per `u32` word (Card 1007), and a BF16 weight likewise (Card 1011). A stored
/// dtype is never widened here: the caller refuses a const whose declared dtype differs from the stored one
/// before it gets this far, and a stored dtype with no encoding into the planned storage is `NoEncoding`.
pub(crate) fn encode_stored(
    stored: DType,
    bytes: &[u8],
    planned: BufferStorage,
) -> Result<Vec<u8>, EncodeError> {
    match (stored, planned) {
        (DType::F32, p) if p == BufferStorage::f32() => Ok(bytes.to_vec()),
        (DType::F16, p) if p == BufferStorage::f16() => Ok(bytes.to_vec()),
        (DType::BF16, p) if p == BufferStorage::bf16() => Ok(bytes.to_vec()),
        (DType::I32, p) if p == BufferStorage::i32() => Ok(bytes.to_vec()),
        (DType::I32, p) if p == BufferStorage::i32_f32_mirror() => Ok(i32_to_f32_mirror(bytes)),
        // Two little-endian BF16 elements per u32 word is the stored byte order, padded to a word.
        (DType::BF16, p) if p == BufferStorage::bf16_packed() => Ok(packed_words(bytes)),
        // Card 1007: the same word for F16, element `i` in word `i / 2` at bit `(i % 2) * 16`, once every
        // element is finite.
        (DType::F16, p) if p == BufferStorage::f16_packed() => {
            if let Some((index, bits)) = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .enumerate()
                .find(|&(_, bits)| bits & 0x7c00 == 0x7c00)
            {
                return Err(EncodeError::NonFiniteF16 { index, bits });
            }
            Ok(packed_words(bytes))
        }
        _ => Err(EncodeError::NoEncoding),
    }
}

/// Two-byte elements as stored, zero-padded to a whole `u32` word.
fn packed_words(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out.resize(bytes.len().div_ceil(4) * 4, 0);
    out
}

/// A host view's bytes as `planned` holds them (slots and computed consts).
pub(crate) fn encode_host(dtype: DType, bytes: &[u8], planned: BufferStorage) -> Option<Vec<u8>> {
    match (dtype, planned.element(), planned.dtype()) {
        (DType::F32, ElementKind::F32, LogicalDType::F32) => Some(bytes.to_vec()),
        (DType::I32, ElementKind::I32, LogicalDType::I32) => Some(bytes.to_vec()),
        (DType::I32, ElementKind::F32, LogicalDType::I32) => Some(i32_to_f32_mirror(bytes)),
        _ => None,
    }
}

/// The F32 bit-pattern mirror of declared-I32 values (Card 621).
fn i32_to_f32_mirror(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .flat_map(|c| (i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).to_le_bytes())
        .collect()
}
