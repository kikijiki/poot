//! The float families' descriptors: the dense float formats as one-value blocks, and MXFP4.

use super::{HALVES_32, WHOLE_32, blocks, piece};
use crate::format::{FieldEncoding, FloatFormat, FormatDescriptor, ValueMap, WeightFormat};

/// A dense float: a one-value block holding the value itself.
macro_rules! dense {
    ($format:ident, $float:ident, $bytes:literal) => {
        FormatDescriptor {
            format: WeightFormat::$format,
            storage: blocks(
                1,
                $bytes,
                &[field!(
                    Codes,
                    "value",
                    FieldEncoding::Float(FloatFormat::$float),
                    [piece(0, &[], FloatFormat::$float.bits())]
                )],
            ),
            value: ValueMap::Float,
            min_sign: None,
        }
    };
}

pub(crate) const F32: FormatDescriptor = dense!(F32, F32, 4);
pub(crate) const F16: FormatDescriptor = dense!(F16, F16, 2);
pub(crate) const BF16: FormatDescriptor = dense!(Bf16, Bf16, 2);

/// MXFP4 (ggml type 39): `[e u8][qs u8[16]]`, 17 bytes per 32, with the `Q4_0` nibble order.
/// `y = e2m1(q) * 2^(e - 127)`; see the sign-of-zero note in [`crate::format`].
pub(crate) const MXFP4: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Mxfp4,
    storage: blocks(
        32,
        17,
        &[
            field!(
                Scale,
                "e",
                FieldEncoding::Float(FloatFormat::E8m0),
                [piece(0, WHOLE_32, 8)]
            ),
            field!(
                Codes,
                "qs",
                FieldEncoding::Float(FloatFormat::E2m1),
                [piece(8, HALVES_32, 4)]
            ),
        ],
    ),
    value: ValueMap::Float,
    min_sign: None,
};
