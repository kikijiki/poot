//! The reference encoder for the block weight formats: the inverse of [`crate::decode`], for fixtures and
//! the GGUF writer.
//!
//! No production load path encodes (ADR-0103 decision 1): checkpoints arrive already packed. The encoder
//! lives beside the decoder so one crate owns both directions of every layout, and a round trip proves the
//! two agree on one descriptor. Each scheme is the simple ggml reference encoder: one block scale from the
//! block's largest magnitude, each element rounded to the nearest code. The scale is stored as the f16 the
//! decoder will read back ([`crate::scalar::f32_to_f16`]), but the codes divide by the unrounded scale, as
//! ggml's `quantize_row_*_ref` do.

use std::fmt;

use crate::format::{Storage, WeightFormat};
use crate::scalar::{e2m1_to_f32, e8m0_to_f32, f32_to_f16};
use crate::{PackedPayload, PackedWeight, PackedWeightError, SourceRole, StoredBytes};

/// Values per block of every scheme [`encode`] supports.
const BLOCK_VALUES: usize = 32;

/// Typed failures of [`encode`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// `format` has no reference encoder (a dense, planar or K-quant/IQ format).
    UnsupportedFormat { format: WeightFormat },
    /// `values` does not hold exactly `rows * cols` values.
    ValueCount {
        rows: usize,
        cols: usize,
        actual: usize,
    },
    /// A NaN or infinite input value: block formats store finite values only.
    NonFiniteValue { index: usize },
    /// The `[rows, cols]` shape is not a valid packed weight of the format.
    Weight(PackedWeightError),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormat { format } => write!(f, "{format:?} has no reference encoder"),
            Self::ValueCount { rows, cols, actual } => {
                write!(
                    f,
                    "[{rows}, {cols}] needs {rows}*{cols} values, got {actual}"
                )
            }
            Self::NonFiniteValue { index } => {
                write!(f, "value {index} is not finite and cannot be block-encoded")
            }
            Self::Weight(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<PackedWeightError> for EncodeError {
    fn from(error: PackedWeightError) -> Self {
        Self::Weight(error)
    }
}

/// Encode row-major `[rows, cols]` `values` into the packed payload of `format`.
///
/// Supports [`WeightFormat::Q8_0`], [`WeightFormat::Q4_0`], [`WeightFormat::Q5_0`] and
/// [`WeightFormat::Mxfp4`]; `cols` must be a whole number of 32-value blocks. The result decodes through
/// [`PackedPayload::decode_row`], the one scalar decoder.
pub fn encode(
    format: WeightFormat,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> Result<PackedPayload, EncodeError> {
    let encode_block: fn(&[f32], &mut Vec<u8>) = match format {
        WeightFormat::Q8_0 => encode_q8_0_block,
        WeightFormat::Q4_0 => encode_q4_0_block,
        WeightFormat::Q5_0 => encode_q5_0_block,
        WeightFormat::Mxfp4 => encode_mxfp4_block,
        _ => return Err(EncodeError::UnsupportedFormat { format }),
    };
    let weight = PackedWeight::try_new(format, [rows, cols])?;
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("{format:?} has a block encoder, so block storage");
    };
    debug_assert_eq!(layout.values, BLOCK_VALUES);
    if values.len() != weight.logical_values() {
        return Err(EncodeError::ValueCount {
            rows,
            cols,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(EncodeError::NonFiniteValue { index });
    }
    let mut bytes = Vec::with_capacity(weight.source_bytes(SourceRole::Blocks));
    for block in values.as_chunks::<BLOCK_VALUES>().0 {
        encode_block(block, &mut bytes);
    }
    Ok(PackedPayload::try_new(
        weight,
        [(SourceRole::Blocks, StoredBytes::from(bytes))],
    )?)
}

/// The largest magnitude of one block.
fn amax(block: &[f32]) -> f32 {
    block.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

/// Q8_0: an f16 scale `d = max|x| / 127`, then 32 `round(x / d)` as i8.
fn encode_q8_0_block(block: &[f32], out: &mut Vec<u8>) {
    let d = amax(block) / 127.0;
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for &x in block {
        let code = if d != 0.0 {
            (x / d).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
        out.push(code as u8);
    }
}

/// Q4_0: an f16 scale `d = max|x| / 8`, then 16 bytes packing `clamp(round(x / d) + 8, 0, 15)` with byte `j`
/// holding element `j` low and element `j + 16` high.
fn encode_q4_0_block(block: &[f32], out: &mut Vec<u8>) {
    let d = amax(block) / 8.0;
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    let nibble = |x: f32| -> u8 {
        if d != 0.0 {
            ((x / d).round() + 8.0).clamp(0.0, 15.0) as u8
        } else {
            8
        }
    };
    for j in 0..BLOCK_VALUES / 2 {
        out.push(nibble(block[j]) | (nibble(block[j + 16]) << 4));
    }
}

/// Q5_0: an f16 scale `d = max|x| / 16`, a `u32` whose bit `j` is the fifth bit of element `j`, then the low
/// four bits packed as Q4_0 does.
fn encode_q5_0_block(block: &[f32], out: &mut Vec<u8>) {
    let d = amax(block) / 16.0;
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    let code = |x: f32| -> u8 {
        if d != 0.0 {
            ((x / d).round() + 16.0).clamp(0.0, 31.0) as u8
        } else {
            16
        }
    };
    let mut high_bits: u32 = 0;
    let mut low = [0u8; BLOCK_VALUES];
    for (j, &x) in block.iter().enumerate() {
        let q = code(x);
        low[j] = q & 0x0f;
        high_bits |= u32::from((q >> 4) & 1) << j;
    }
    out.extend_from_slice(&high_bits.to_le_bytes());
    for j in 0..BLOCK_VALUES / 2 {
        out.push(low[j] | (low[j + 16] << 4));
    }
}

/// MXFP4 as ggml's `quantize_row_mxfp4_ref`: the shared E8M0 exponent `floor(log2(max|x|)) - 2 + 127` (0 for an
/// all-zero block), then per element the E2M1 code whose decoded value is nearest, packed as Q4_0 does.
fn encode_mxfp4_block(block: &[f32], out: &mut Vec<u8>) {
    let amax = amax(block);
    let exponent: u8 = if amax > 0.0 {
        (amax.log2().floor() - 2.0 + 127.0).clamp(0.0, 254.0) as u8
    } else {
        0
    };
    let scale = e8m0_to_f32(exponent);
    let code_of = |x: f32| -> u8 {
        (0..16u8)
            .min_by(|&a, &b| {
                let error = |code| (x - e2m1_to_f32(code) * scale).abs();
                error(a).total_cmp(&error(b))
            })
            .expect("sixteen codes")
    };
    out.push(exponent);
    for j in 0..BLOCK_VALUES / 2 {
        out.push(code_of(block[j]) | (code_of(block[j + 16]) << 4));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::f16_to_f32;

    /// Decode every row of `payload` through the one scalar decoder.
    fn decode(payload: &PackedPayload) -> Vec<f32> {
        let [rows, cols] = payload.weight().shape();
        let mut out = vec![0.0f32; rows * cols];
        for (row, chunk) in out.chunks_exact_mut(cols).enumerate() {
            payload.decode_row(row, chunk).unwrap();
        }
        out
    }

    fn assert_bit_exact(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (index, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: element {index} decoded to {g:?}, want {w:?}"
            );
        }
    }

    /// Two rows of four blocks. Each block's elements are `code * scale` for distinct integer codes, with the
    /// code that fixes the block scale (`pinned`) in element 0: every value sits on its format's grid, so
    /// the encoder is lossless and the decode must return the input bit for bit. Distinct codes in every
    /// position make any nibble, bit or element permutation visible.
    fn on_grid_values(codes: std::ops::RangeInclusive<i32>, pinned: i32) -> Vec<f32> {
        let span = codes.end() - codes.start() + 1;
        (0..8usize)
            .flat_map(|block| {
                let scale = 0.25f32 * (1u32 << block) as f32;
                let codes = codes.clone();
                (0..BLOCK_VALUES).map(move |j| {
                    let code = if j == 0 {
                        pinned
                    } else {
                        codes.start() + ((j as i32 * 7 + block as i32 * 3) % span)
                    };
                    code as f32 * scale
                })
            })
            .collect()
    }

    /// SC-004: Q8_0, Q4_0 and Q5_0 round-trip on-grid values bit-exactly. A block's scale is its largest
    /// magnitude over the top code count, so pinning the most negative code in element 0 fixes it.
    #[test]
    fn uniform_block_schemes_round_trip_bit_exactly() {
        for (format, codes, pinned) in [
            (WeightFormat::Q8_0, -127..=127, -127),
            (WeightFormat::Q4_0, -8..=7, -8),
            (WeightFormat::Q5_0, -16..=15, -16),
        ] {
            let values = on_grid_values(codes, pinned);
            let payload = encode(format, 2, 128, &values).unwrap();
            assert_bit_exact(&decode(&payload), &values, &format!("{format:?}"));
        }
    }

    /// SC-004: MXFP4 round-trips every E2M1 magnitude and sign (except -0, which encodes as +0) times a
    /// per-block power of two, with 6 in element 0 fixing the shared exponent.
    #[test]
    fn mxfp4_round_trips_the_e2m1_grid_bit_exactly() {
        const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let values: Vec<f32> = (0..8usize)
            .flat_map(|block| {
                let scale = 2.0f32.powi(block as i32 - 3);
                (0..BLOCK_VALUES).map(move |j| {
                    let magnitude = if j == 0 {
                        6.0
                    } else {
                        GRID[(j * 5 + block) % 8]
                    };
                    let negative = magnitude != 0.0 && (j + block) % 3 == 0;
                    if negative {
                        -magnitude * scale
                    } else {
                        magnitude * scale
                    }
                })
            })
            .collect();
        let payload = encode(WeightFormat::Mxfp4, 2, 128, &values).unwrap();
        assert_bit_exact(&decode(&payload), &values, "Mxfp4");
    }

    /// Off-grid input: the decode must equal the spec's value for each element, computed here directly in
    /// the decoded domain (`round(x / d) * f16(d)`), so a wrong scale, a wrong rounding or a wrong
    /// offset shows even though both sides read the same bytes.
    #[test]
    fn off_grid_values_decode_to_the_rounded_codes_times_the_f16_scale() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.173).collect();
        for (format, divisor, offset, low, high) in [
            (WeightFormat::Q8_0, 127.0f32, 0.0f32, -127.0f32, 127.0f32),
            (WeightFormat::Q4_0, 8.0, 8.0, 0.0, 15.0),
            (WeightFormat::Q5_0, 16.0, 16.0, 0.0, 31.0),
        ] {
            let payload = encode(format, 1, 64, &values).unwrap();
            let want: Vec<f32> = values
                .chunks(BLOCK_VALUES)
                .flat_map(|block| {
                    let d = amax(block) / divisor;
                    let stored = f16_to_f32(f32_to_f16(d));
                    block.iter().map(move |&x| {
                        let code = ((x / d).round() + offset).clamp(low, high);
                        (code as i32 - offset as i32) as f32 * stored
                    })
                })
                .collect();
            assert_bit_exact(&decode(&payload), &want, &format!("{format:?}"));
        }
    }

    /// Off-grid MXFP4: every decoded value is the E2M1 grid point nearest its source at the block's shared scale
    /// (ggml's `best_index_mxfp4`). The grid is non-uniform (gaps 1, 1, 1, 2, 2, 4 scale units), so a flat
    /// absolute tolerance would be wrong.
    #[test]
    fn mxfp4_off_grid_values_decode_to_the_nearest_grid_point() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.037 - 0.4).collect();
        let payload = encode(WeightFormat::Mxfp4, 1, 32, &values).unwrap();
        let exponent = payload.bytes(SourceRole::Blocks)[0];
        let scale = e8m0_to_f32(exponent);
        let grid: Vec<f32> = (0..16u8).map(|code| e2m1_to_f32(code) * scale).collect();
        for (index, (&got, &source)) in decode(&payload).iter().zip(&values).enumerate() {
            let best = grid
                .iter()
                .copied()
                .min_by(|x, y| (x - source).abs().total_cmp(&(y - source).abs()))
                .unwrap();
            assert_eq!(
                got, best,
                "element {index}: decoded {got}, nearest grid point to {source} at scale {scale} is {best}"
            );
        }
    }

    #[test]
    fn unsupported_and_malformed_requests_are_typed() {
        let values = vec![0.5f32; 32];
        assert_eq!(
            encode(WeightFormat::Q4_K, 1, 32, &values),
            Err(EncodeError::UnsupportedFormat {
                format: WeightFormat::Q4_K
            })
        );
        assert_eq!(
            encode(WeightFormat::Q8_0, 2, 32, &values),
            Err(EncodeError::ValueCount {
                rows: 2,
                cols: 32,
                actual: 32
            })
        );
        assert!(matches!(
            encode(WeightFormat::Q8_0, 1, 33, &[0.0; 33]),
            Err(EncodeError::Weight(PackedWeightError::PartialBlock { .. }))
        ));
        let mut poisoned = values;
        poisoned[7] = f32::NAN;
        assert_eq!(
            encode(WeightFormat::Q8_0, 1, 32, &poisoned),
            Err(EncodeError::NonFiniteValue { index: 7 })
        );
    }
}
