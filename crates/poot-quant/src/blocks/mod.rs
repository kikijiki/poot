//! Descriptors of the block formats: the GGUF quantized blocks, transcribed from ggml's block
//! structs (`ggml/src/ggml-common.h`) and dequantizers (`ggml/src/ggml-quants.c`), and the dense
//! float formats as one-value blocks.
//!
//! The descriptors are split by value family, the way the scalar semantics are
//! ([`crate::families`]): [`legacy`] (the `Q4_0` family), [`kquant`] (the K-quant super-blocks),
//! [`lattice`] (the table formats: `IQ4_NL`, `IQ4_XS` and the `IQ2`/`IQ3` grids) and [`float`] (the
//! dense floats and MXFP4).
//!
//! Bit offsets are from the start of the block; see [`crate::format`] for how a [`BitLayout`] maps
//! an in-block element index to a bit. Each layout's digits name the element index split the way
//! ggml's dequantizer loop walks it.

use crate::format::{BitLayout, BlockLayout, Digit, FieldEncoding, FloatFormat, Piece, Storage};

/// Values in a K-quant super-block (`QK_K`).
pub(crate) const QK_K: usize = 256;

const fn digit(extent: u32, stride: u32) -> Digit {
    Digit { extent, stride }
}

const fn piece(base: u32, digits: &'static [Digit], width: u32) -> Piece {
    Piece {
        layout: BitLayout { base, digits },
        width,
    }
}

const WHOLE_32: &[Digit] = &[digit(32, 0)];
const WHOLE_QK_K: &[Digit] = &[digit(QK_K as u32, 0)];
/// `i = h*16 + l`: nibble `h` of byte `l` (the `Q4_0` family's half/half interleave).
const HALVES_32: &[Digit] = &[digit(2, 4), digit(16, 8)];

const HALF: FieldEncoding = FieldEncoding::Float(FloatFormat::F16);
const UNSIGNED: FieldEncoding = FieldEncoding::Unsigned { offset: 0 };

/// One operand of a block: `field!(Role, "name", encoding, [pieces])`.
macro_rules! field {
    ($role:ident, $name:literal, $encoding:expr, [$($piece:expr),+ $(,)?]) => {
        crate::format::BlockField {
            role: crate::format::OperandRole::$role,
            name: $name,
            field: crate::format::Field {
                pieces: &[$($piece),+],
                encoding: $encoding,
            },
        }
    };
}

const fn blocks(
    values: usize,
    bytes: usize,
    fields: &'static [crate::format::BlockField],
) -> Storage {
    Storage::Blocks(BlockLayout {
        values,
        bytes,
        fields,
    })
}

mod float;
mod kquant;
mod lattice;
mod legacy;

pub(crate) use float::{BF16, F16, F32, MXFP4};
pub(crate) use kquant::{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, TQ2_0};
pub(crate) use lattice::{IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS};
pub(crate) use legacy::{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

#[cfg(test)]
mod tests;
