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

/// How a stored dense dtype lands in a planned lane, decided from the two types alone: it names the transform
/// without running it, so a planning pass can refuse a weight that has no encoding without reading its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// The stored bytes are the lane's bytes.
    Verbatim,
    /// An `i32` index table whose lane mirrors it as `f32`.
    I32AsF32,
    /// Two-byte elements, zero-padded to a whole `u32` word.
    PackedWords,
    /// Card 1007: as [`Encoding::PackedWords`], once every element is a finite binary16.
    PackedWordsFiniteF16,
    /// POOT-1060: one-byte elements, each row of the last axis padded with zero bytes to a whole `u32` word.
    RowPaddedWords,
}

/// The encoding of a `stored` dtype into the `planned` storage, or `None` when there is none. A stored dtype is
/// never widened here: the caller refuses a const whose declared dtype differs from the stored one before it
/// gets this far.
pub(crate) fn encoding_of(stored: DType, planned: BufferStorage) -> Option<Encoding> {
    match (stored, planned) {
        (DType::F32, p) if p == BufferStorage::f32() => Some(Encoding::Verbatim),
        (DType::F16, p) if p == BufferStorage::f16() => Some(Encoding::Verbatim),
        (DType::BF16, p) if p == BufferStorage::bf16() => Some(Encoding::Verbatim),
        (DType::I32, p) if p == BufferStorage::i32() => Some(Encoding::Verbatim),
        (DType::I32, p) if p == BufferStorage::i32_f32_mirror() => Some(Encoding::I32AsF32),
        // Two little-endian BF16 elements per u32 word is the stored byte order, padded to a word.
        (DType::BF16, p) if p == BufferStorage::bf16_packed() => Some(Encoding::PackedWords),
        // Card 1007: the same word for F16, element `i` in word `i / 2` at bit `(i % 2) * 16`.
        (DType::F16, p) if p == BufferStorage::f16_packed() => Some(Encoding::PackedWordsFiniteF16),
        // POOT-1060: `words_per_row = ceil(width / 4)`, so a row never starts mid-word.
        (DType::E4M3FN, p) if p == BufferStorage::e4m3_packed() => Some(Encoding::RowPaddedWords),
        _ => None,
    }
}

/// A stored dense weight's bytes as `planned` holds them. An F16 weight a contraction reads lands as stored,
/// natively or packed two elements per `u32` word (Card 1007), and a BF16 weight likewise (Card 1011); a stored
/// dtype with no encoding into the planned storage is `NoEncoding`.
pub(crate) fn encode_stored(
    stored: DType,
    shape: &[usize],
    bytes: &[u8],
    planned: BufferStorage,
) -> Result<Vec<u8>, EncodeError> {
    match encoding_of(stored, planned).ok_or(EncodeError::NoEncoding)? {
        Encoding::Verbatim => Ok(bytes.to_vec()),
        Encoding::I32AsF32 => Ok(i32_to_f32_mirror(bytes)),
        Encoding::PackedWords => Ok(packed_words(bytes)),
        Encoding::PackedWordsFiniteF16 => {
            if let Some((index, bits)) = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .enumerate()
                .find(|&(_, bits)| bits & 0x7c00 == 0x7c00)
            {
                return Err(EncodeError::NonFiniteF16 { index, bits });
            }
            Ok(packed_words(bytes))
        }
        Encoding::RowPaddedWords => Ok(row_padded_words(shape, bytes)),
    }
}

/// One-byte elements of `shape`, each row of the last axis zero-padded to a whole `u32` word. A 0-D value
/// is one row of width one.
fn row_padded_words(shape: &[usize], bytes: &[u8]) -> Vec<u8> {
    let width = shape.last().copied().unwrap_or(1).max(1);
    let padded = width.div_ceil(4) * 4;
    let mut out = Vec::with_capacity(bytes.len().div_ceil(width) * padded);
    for row in bytes.chunks(width) {
        out.extend_from_slice(row);
        out.resize(out.len() + padded - row.len(), 0);
    }
    out
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
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|c| (i32::from_le_bytes(*c) as f32).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use poot_target::BufferStorage;
    use poot_tensor::DType;

    use super::{EncodeError, encode_stored, encoding_of};

    /// POOT-1060: a stored E4M3FN table lands as `u32` words with every row of the last axis zero-padded to
    /// a whole word, so a width that is not a multiple of four starts the next row in the next word, and the
    /// byte count is the word count `BufferStorage::device_elems` sizes the buffer with.
    ///
    /// Mutation: pad to `width` instead of `width.div_ceil(4) * 4` in `row_padded_words`; rows 2 and 3 move
    /// and this row goes red.
    #[test]
    fn an_e4m3_table_pads_each_row_to_a_whole_word() {
        let packed = BufferStorage::e4m3_packed();
        let bytes = [1u8, 2, 3, 4, 5, 6];
        let encoded = encode_stored(DType::E4M3FN, &[2, 3], &bytes, packed).unwrap();
        assert_eq!(encoded, [1, 2, 3, 0, 4, 5, 6, 0]);
        assert_eq!(encoded.len(), packed.device_elems(&[2, 3]).unwrap() * 4);
        let wide = encode_stored(DType::E4M3FN, &[1, 5], &[9, 8, 7, 6, 5], packed).unwrap();
        assert_eq!(wide, [9, 8, 7, 6, 5, 0, 0, 0]);
        let aligned = encode_stored(DType::E4M3FN, &[2, 4], &[1, 2, 3, 4, 5, 6, 7, 8], packed);
        assert_eq!(aligned.unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// Dense E4M3FN storage still has no encoding: only the packed lane does.
    #[test]
    fn dense_e4m3_storage_has_no_encoding() {
        let dense = BufferStorage::dense(
            poot_target::ElementKind::RawBytes,
            poot_target::LogicalDType::E4M3FN,
        );
        assert_eq!(
            encode_stored(DType::E4M3FN, &[1, 4], &[1, 2, 3, 4], dense),
            Err(EncodeError::NoEncoding)
        );
    }

    /// POOT-1066: `encoding_of` is the one decision of whether a stored dtype lands in a planned lane, so a
    /// planning pass that only sizes a weight refuses exactly what the encoding upload refuses, and an
    /// encoded weight's length is the lane's device element count (the size a planning pass allocates).
    ///
    /// Mutation: make `encoding_of` return `Some(Encoding::Verbatim)` for every pair; the unencodable pairs
    /// stop being `NoEncoding` and this row goes red.
    #[test]
    fn encoding_of_agrees_with_encode_stored_for_every_dtype_and_lane() {
        let storages = [
            BufferStorage::f32(),
            BufferStorage::f16(),
            BufferStorage::bf16(),
            BufferStorage::i32(),
            BufferStorage::i32_f32_mirror(),
            BufferStorage::bf16_packed(),
            BufferStorage::f16_packed(),
            BufferStorage::e4m3_packed(),
        ];
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::E4M3FN,
        ];
        let shape = [3usize, 5];
        let mut encodable = 0;
        for dtype in dtypes {
            for storage in storages {
                let width = match dtype {
                    DType::F32 | DType::I32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    _ => 1,
                };
                let bytes = vec![0u8; 15 * width];
                match (
                    encoding_of(dtype, storage),
                    encode_stored(dtype, &shape, &bytes, storage),
                ) {
                    (None, Err(EncodeError::NoEncoding)) => {}
                    (Some(_), Ok(encoded)) => {
                        encodable += 1;
                        let elems = storage.device_elems(&shape).expect("a sized lane");
                        assert_eq!(
                            encoded.len(),
                            elems * storage.element().byte_width(),
                            "{dtype:?} into {storage:?}: encoded length vs the lane's device elements"
                        );
                    }
                    (decision, encoded) => panic!(
                        "{dtype:?} into {storage:?}: encoding_of {decision:?} vs encode_stored {encoded:?}"
                    ),
                }
            }
        }
        assert!(encodable >= 8, "the table exercises the encodable pairs");
    }
}
