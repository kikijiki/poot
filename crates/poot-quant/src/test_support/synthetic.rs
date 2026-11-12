//! A synthetic block format whose fields have four pieces each, for the rows that pin N-piece fields
//! (POOT-1539 SC-4) in every evaluator: the reference interpreter, the const plan and the device
//! emitter. No GGUF format has a field of more than two pieces yet (IQ1_M's f16 super-scale will be
//! the first), so the descriptor is a fixture, not a format.
//!
//! A block holds 32 values in 24 bytes:
//!
//! ```text
//! bytes 0..8   four u16 words; the top nibble of word k is bits 4k..4k+4 of the f16 scale `d`
//! bytes 8..24  four bit planes of 32 bits; plane k holds bit k of each element's code
//! ```
//!
//! and `w = d * (code - 8)`.

use crate::format::{
    BitLayout, BlockField, BlockLayout, Digit, Field, FieldEncoding, FloatFormat, FormatDescriptor,
    OperandRole, Piece, Storage, ValueMap, WeightFormat,
};
use crate::scalar::f16_to_f32;

const BROADCAST: &[Digit] = &[Digit {
    extent: 32,
    stride: 0,
}];
const ELEMENT_BITS: &[Digit] = &[Digit {
    extent: 32,
    stride: 1,
}];

const fn scale_nibble(word: u32) -> Piece {
    Piece {
        layout: BitLayout {
            base: 16 * word + 12,
            digits: BROADCAST,
        },
        width: 4,
    }
}

const fn code_plane(plane: u32) -> Piece {
    Piece {
        layout: BitLayout {
            base: 64 + 32 * plane,
            digits: ELEMENT_BITS,
        },
        width: 1,
    }
}

/// The descriptor. Its `format` is a label (the descriptor is evaluated directly, never looked up by
/// format).
pub const FOUR_PIECE: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q4_0,
    storage: Storage::Blocks(BlockLayout {
        values: 32,
        bytes: 24,
        fields: &[
            BlockField {
                role: OperandRole::Scale,
                name: "d (four nibbles)",
                field: Field {
                    pieces: &[
                        scale_nibble(0),
                        scale_nibble(1),
                        scale_nibble(2),
                        scale_nibble(3),
                    ],
                    encoding: FieldEncoding::Float(FloatFormat::F16),
                },
            },
            BlockField {
                role: OperandRole::Codes,
                name: "qs (four bit planes)",
                field: Field {
                    pieces: &[code_plane(0), code_plane(1), code_plane(2), code_plane(3)],
                    encoding: FieldEncoding::Unsigned { offset: -8 },
                },
            },
        ],
    }),
    value: ValueMap::Integer,
    min_sign: None,
};

/// `blocks` blocks of pseudo-random bytes whose scale is a finite f16 (its exponent's top bit is
/// clear), so every element decodes to a number.
pub fn four_piece_blocks(blocks: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut bytes: Vec<u8> = (0..blocks * 24)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    for block in bytes.chunks_mut(24) {
        // The scale's bit 14 is the top nibble's bit 2: byte 7, bit 6.
        block[7] &= !0x40;
    }
    bytes
}

/// The decode of one block by direct arithmetic on its bytes, independent of every evaluator.
pub fn four_piece_reference(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 24);
    let bit = |at: usize| u32::from(block[at / 8] >> (at % 8) & 1);
    let scale_bits: u32 = (0..4)
        .map(|word| {
            let nibble = (0..4).map(|k| bit(16 * word + 12 + k) << k).sum::<u32>();
            nibble << (4 * word)
        })
        .sum();
    let scale = f16_to_f32(scale_bits as u16);
    (0..32)
        .map(|i| {
            let code: u32 = (0..4).map(|plane| bit(64 + 32 * plane + i) << plane).sum();
            scale * (code as i32 - 8) as f32
        })
        .collect()
}
