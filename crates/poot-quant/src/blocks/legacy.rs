//! The `Q4_0` family: 32-value blocks of one f16 scale (and, for `Q4_1`/`Q5_1`, an f16 min) with
//! integer codes.

use super::{HALF, HALVES_32, UNSIGNED, WHOLE_32, blocks, digit, piece};
use crate::format::{FieldEncoding, FormatDescriptor, MinSign, ValueMap, WeightFormat};

/// `Q4_0` (ggml type 2): `[d f16][qs u8[16]]`, 18 bytes per 32. `y = (q - 8) * d`.
pub(crate) const Q4_0: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q4_0,
    storage: blocks(
        32,
        18,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_32, 16)]),
            field!(
                Codes,
                "qs",
                FieldEncoding::Unsigned { offset: -8 },
                [piece(16, HALVES_32, 4)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};

/// `Q4_1` (ggml type 3): `[d f16][m f16][qs u8[16]]`, 20 bytes per 32. `y = q*d + m`.
pub(crate) const Q4_1: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q4_1,
    storage: blocks(
        32,
        20,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_32, 16)]),
            field!(Min, "m", HALF, [piece(16, WHOLE_32, 16)]),
            field!(Codes, "qs", UNSIGNED, [piece(32, HALVES_32, 4)]),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: Some(MinSign::Add),
};

/// `Q5_0` (ggml type 6): `[d f16][qh u8[4]][qs u8[16]]`, 22 bytes per 32. Element `i`'s fifth bit
/// is bit `i` of the little-endian `qh` (`xh_0 = qh >> j`, `xh_1 = qh >> (j + 12)` for
/// `i = j + 16`). `y = ((q | h << 4) - 16) * d`.
pub(crate) const Q5_0: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q5_0,
    storage: blocks(
        32,
        22,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_32, 16)]),
            field!(
                Codes,
                "qs+qh",
                FieldEncoding::Unsigned { offset: -16 },
                [piece(48, HALVES_32, 4), piece(16, &[digit(32, 1)], 1)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};

/// `Q5_1` (ggml type 7): `[d f16][m f16][qh u8[4]][qs u8[16]]`, 24 bytes per 32.
/// `y = (q | h << 4)*d + m`.
pub(crate) const Q5_1: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q5_1,
    storage: blocks(
        32,
        24,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_32, 16)]),
            field!(Min, "m", HALF, [piece(16, WHOLE_32, 16)]),
            field!(
                Codes,
                "qs+qh",
                UNSIGNED,
                [piece(64, HALVES_32, 4), piece(32, &[digit(32, 1)], 1)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: Some(MinSign::Add),
};

/// `Q8_0` (ggml type 8): `[d f16][qs i8[32]]`, 34 bytes per 32. `y = q * d`.
pub(crate) const Q8_0: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q8_0,
    storage: blocks(
        32,
        34,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_32, 16)]),
            field!(
                Codes,
                "qs",
                FieldEncoding::Signed,
                [piece(16, &[digit(32, 8)], 8)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};
