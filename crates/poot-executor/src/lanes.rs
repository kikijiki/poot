//! Host bytes into a planned device lane: the one encoder every device shares, so no backend
//! re-derives how a stored dtype lands in the storage the plan chose (ADR-0103: bytes as stored where
//! the plan keeps them, the plan's widening where it does not).

use std::ops::Range;

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
        // Two little-endian BF16 elements per u32 word is the stored byte order, padded to a word.
        (DType::BF16, p) if p == BufferStorage::bf16_packed() => Some(Encoding::PackedWords),
        // Card 1007: the same word for F16, element `i` in word `i / 2` at bit `(i % 2) * 16`.
        (DType::F16, p) if p == BufferStorage::f16_packed() => Some(Encoding::PackedWordsFiniteF16),
        // POOT-1060: `words_per_row = ceil(width / 4)`, so a row never starts mid-word.
        (DType::E4M3FN, p) if p == BufferStorage::e4m3_packed() => Some(Encoding::RowPaddedWords),
        _ => None,
    }
}

/// The first element of binary16 `bytes` that is an infinity or NaN (exponent `0x1f`), as `(index, bits)`.
pub(crate) fn first_non_finite_f16(bytes: &[u8]) -> Option<(usize, u16)> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .enumerate()
        .find(|&(_, bits)| bits & 0x7c00 == 0x7c00)
}

/// The width of one row of `shape`'s last axis (a 0-D value is one row of width one) and that row zero-padded
/// to a whole `u32` word.
fn row_widths(shape: &[usize]) -> (usize, usize) {
    let width = shape.last().copied().unwrap_or(1).max(1);
    (width, width.div_ceil(4) * 4)
}

/// The byte length of `stored_len` stored bytes of `shape` once `encoding` has landed them: the bytes the
/// device must be handed. Whatever the lane holds past that (a packed lane's guard word, a stack's last slab
/// padding) is the allocation's own zeros.
pub(crate) fn encoded_len(encoding: Encoding, shape: &[usize], stored_len: usize) -> usize {
    match encoding {
        Encoding::Verbatim => stored_len,
        Encoding::PackedWords | Encoding::PackedWordsFiniteF16 => stored_len.div_ceil(4) * 4,
        Encoding::RowPaddedWords => {
            let (width, padded) = row_widths(shape);
            stored_len.div_ceil(width) * padded
        }
    }
}

/// Bytes `range` of the encoded lane, written into `out` (`out.len() == range.len()`), where `read(r, out)`
/// supplies bytes `r` of the `stored_len` stored bytes. Every encoding is element- or row-local, so any
/// sub-range encodes on its own: this is the one definition of how a stored dtype lands in a lane, and it is
/// what lets a device take a weight in chunks without a host copy of the whole. A byte past the stored data
/// (a word's padding) is zero.
pub(crate) fn encode_range(
    encoding: Encoding,
    shape: &[usize],
    stored_len: usize,
    range: Range<usize>,
    read: &mut dyn FnMut(Range<usize>, &mut [u8]),
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), range.len());
    // Copy stored bytes `from..from + n` to the front of `into`, zero the rest of `into`.
    let copy_clipped =
        |from: usize, into: &mut [u8], read: &mut dyn FnMut(Range<usize>, &mut [u8])| {
            let n = into.len().min(stored_len.saturating_sub(from));
            if n > 0 {
                read(from..from + n, &mut into[..n]);
            }
            into[n..].fill(0);
        };
    match encoding {
        Encoding::Verbatim | Encoding::PackedWords | Encoding::PackedWordsFiniteF16 => {
            copy_clipped(range.start, out, read);
        }
        Encoding::RowPaddedWords => {
            let (width, padded) = row_widths(shape);
            if width == padded {
                copy_clipped(range.start, out, read);
                return;
            }
            // Row `r` is destination bytes `r * padded .. r * padded + width` of data, then padding.
            out.fill(0);
            for row in range.start / padded..range.end.div_ceil(padded) {
                let row_start = row * padded;
                let lo = range.start.max(row_start);
                let hi = range.end.min(row_start + width);
                if lo < hi {
                    let from = row * width + (lo - row_start);
                    copy_clipped(from, &mut out[lo - range.start..hi - range.start], read);
                }
            }
        }
    }
}

/// A stored dense weight's bytes as `planned` holds them. An F16 weight a contraction reads lands as stored,
/// natively or packed two elements per `u32` word (Card 1007), and a BF16 weight likewise (Card 1011); a stored
/// dtype with no encoding into the planned storage is `NoEncoding`. The whole weight at once, for the small
/// tables a host gather keeps; a device upload goes through [`encode_range`] in chunks.
pub(crate) fn encode_stored(
    stored: DType,
    shape: &[usize],
    bytes: &[u8],
    planned: BufferStorage,
) -> Result<Vec<u8>, EncodeError> {
    let encoding = encoding_of(stored, planned).ok_or(EncodeError::NoEncoding)?;
    if encoding == Encoding::PackedWordsFiniteF16
        && let Some((index, bits)) = first_non_finite_f16(bytes)
    {
        return Err(EncodeError::NonFiniteF16 { index, bits });
    }
    let mut out = vec![0u8; encoded_len(encoding, shape, bytes.len())];
    let len = out.len();
    encode_range(
        encoding,
        shape,
        bytes.len(),
        0..len,
        &mut |r, into| into.copy_from_slice(&bytes[r]),
        &mut out,
    );
    Ok(out)
}

/// A host view's bytes as `planned` holds them (slots and computed consts).
pub(crate) fn encode_host(dtype: DType, bytes: &[u8], planned: BufferStorage) -> Option<Vec<u8>> {
    match (dtype, planned.element(), planned.dtype()) {
        (DType::F32, ElementKind::F32, LogicalDType::F32) => Some(bytes.to_vec()),
        (DType::I32, ElementKind::I32, LogicalDType::I32) => Some(bytes.to_vec()),
        // Native F16 words: a target that computes in F16 (an Intel NPU) stores an F16 slot as it is.
        (DType::F16, ElementKind::F16, LogicalDType::F16) => Some(bytes.to_vec()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use poot_target::BufferStorage;
    use poot_tensor::DType;

    use super::{
        EncodeError, Encoding, encode_host, encode_range, encode_stored, encoded_len, encoding_of,
    };

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

    /// A slot or computed const lands in the lane it is planned in, bit for bit: F16 words in a native F16
    /// buffer (an Intel NPU's lane), and never in a lane of another width.
    ///
    /// Mutation: drop the `(F16, F16, F16)` arm of `encode_host`; the native F16 lane has no encoding.
    #[test]
    fn a_host_f16_view_lands_only_in_the_native_f16_lane() {
        let words = [0x00u8, 0x3c, 0x00, 0x40];
        assert_eq!(
            encode_host(DType::F16, &words, BufferStorage::f16()).as_deref(),
            Some(words.as_slice())
        );
        assert_eq!(encode_host(DType::F16, &words, BufferStorage::f32()), None);
        assert_eq!(encode_host(DType::F32, &words, BufferStorage::f16()), None);
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
        assert!(encodable >= 7, "the table exercises the encodable pairs");
    }

    /// The encoders as they were before `encode_range`: one whole-buffer pass each, kept here as the oracle the
    /// chunked encoder is compared with.
    fn oracle_row_padded(shape: &[usize], bytes: &[u8]) -> Vec<u8> {
        let width = shape.last().copied().unwrap_or(1).max(1);
        let padded = width.div_ceil(4) * 4;
        let mut out = Vec::new();
        for row in bytes.chunks(width) {
            out.extend_from_slice(row);
            out.resize(out.len() + padded - row.len(), 0);
        }
        out
    }

    fn oracle_word_padded(bytes: &[u8]) -> Vec<u8> {
        let mut out = bytes.to_vec();
        out.resize(bytes.len().div_ceil(4) * 4, 0);
        out
    }

    /// POOT-1100 LP-3 (model-free half): every encoding, written in chunks of any size, lands the same bytes as
    /// the whole-buffer oracle. Chunk sizes are chosen to cut rows, words and elements in the middle (4 and 12 are
    /// whole words; the rest are odd). Each lane is shown with a shape whose rows are not whole
    /// words, so the row-padded encoding has padding to place.
    /// Mutation: in `encode_range`'s `RowPaddedWords` arm read the row from `row * width + lo` (forgetting the
    /// row start); every chunking that cuts a row goes red.
    #[test]
    fn every_encoding_lands_the_oracle_bytes_in_chunks_of_any_size() {
        let cases: [(DType, BufferStorage, &[usize]); 7] = [
            (DType::F32, BufferStorage::f32(), &[3, 7]),
            (DType::F16, BufferStorage::f16(), &[5, 3]),
            (DType::BF16, BufferStorage::bf16(), &[5, 3]),
            (DType::I32, BufferStorage::i32(), &[3, 5]),
            (DType::BF16, BufferStorage::bf16_packed(), &[5, 3]),
            (DType::F16, BufferStorage::f16_packed(), &[5, 3]),
            (DType::E4M3FN, BufferStorage::e4m3_packed(), &[7, 5]),
        ];
        for (dtype, storage, shape) in cases {
            let elems: usize = shape.iter().product();
            let width = match dtype {
                DType::F32 | DType::I32 => 4,
                DType::F16 | DType::BF16 => 2,
                _ => 1,
            };
            // Finite F16 patterns only: the packed-F16 lane refuses an infinity or a NaN.
            let bytes: Vec<u8> = (0..elems * width)
                .map(|i| {
                    if width == 2 {
                        [(i * 37) as u8, (i as u8) & 0x3b][i % 2]
                    } else {
                        (i * 37 + 11) as u8
                    }
                })
                .collect();
            let encoding = encoding_of(dtype, storage).expect("an encodable pair");
            let oracle = match encoding {
                Encoding::Verbatim => bytes.clone(),
                Encoding::PackedWords | Encoding::PackedWordsFiniteF16 => {
                    oracle_word_padded(&bytes)
                }
                Encoding::RowPaddedWords => oracle_row_padded(shape, &bytes),
            };
            assert_eq!(
                encoded_len(encoding, shape, bytes.len()),
                oracle.len(),
                "{dtype:?} into {storage:?}"
            );
            assert_eq!(
                encode_stored(dtype, shape, &bytes, storage).unwrap(),
                oracle
            );
            for chunk in [4usize, 12, 5, 7, 13, 1000] {
                let mut assembled = vec![0u8; oracle.len()];
                let mut at = 0;
                while at < oracle.len() {
                    let n = chunk.min(oracle.len() - at);
                    encode_range(
                        encoding,
                        shape,
                        bytes.len(),
                        at..at + n,
                        &mut |r, out| out.copy_from_slice(&bytes[r]),
                        &mut assembled[at..at + n],
                    );
                    at += n;
                }
                assert_eq!(
                    assembled, oracle,
                    "{dtype:?} into {storage:?}, chunks of {chunk}"
                );
            }
        }
    }
}
