//! The K-quants: 256-value super-blocks of integer codes with packed sub-block scales and mins, and
//! `TQ2_0`, the 256-value ternary block of one scale.

use super::{HALF, QK_K, UNSIGNED, WHOLE_QK_K, blocks, digit, piece};
use crate::format::{FieldEncoding, FormatDescriptor, MinSign, ValueMap, WeightFormat};

/// `Q2_K` (ggml type 10): `[scales u8[16]][qs u8[64]][d f16][dmin f16]`, 84 bytes per 256.
/// Element `i = n*128 + j*32 + half*16 + l` is bits `2j` of `qs[n*32 + half*16 + l]`; sub-block
/// `i / 16` has scale `scales[is] & 0xF` and min `scales[is] >> 4`.
/// `y = d*sc * q - dmin*m`.
pub(crate) const Q2_K: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q2_K,
    storage: blocks(
        QK_K,
        84,
        &[
            field!(Scale, "d", HALF, [piece(640, WHOLE_QK_K, 16)]),
            field!(Min, "dmin", HALF, [piece(656, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                UNSIGNED,
                [piece(0, &[digit(16, 8), digit(16, 0)], 4)]
            ),
            field!(
                SubMin,
                "scales",
                UNSIGNED,
                [piece(4, &[digit(16, 8), digit(16, 0)], 4)]
            ),
            field!(
                Codes,
                "qs",
                UNSIGNED,
                [piece(
                    128,
                    &[digit(2, 256), digit(4, 2), digit(2, 128), digit(16, 8)],
                    2
                )]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: Some(MinSign::Subtract),
};

/// `Q3_K` (ggml type 11): `[hmask u8[32]][qs u8[64]][scales u8[12]][d f16]`, 110 bytes per 256.
/// Element `i = n*128 + j*32 + half*16 + l`: low two bits from `qs` as in `Q2_K`, high bit `n*4 + j`
/// of `hmask[half*16 + l]`, stored inverted: `q = low2 - (h ? 0 : 4)`, which is the code
/// `low2 | h << 2` minus 4. The sixteen 6-bit scales (`kmask1`/`kmask2` unpack) take their low
/// nibble from byte `is % 8` (high nibble when `is >= 8`) and their high two bits from byte
/// `8 + is % 4`, bits `2 * (is / 4)`. `y = d*(sc - 32) * q`.
pub(crate) const Q3_K: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q3_K,
    storage: blocks(
        QK_K,
        110,
        &[
            field!(Scale, "d", HALF, [piece(864, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                FieldEncoding::Unsigned { offset: -32 },
                [
                    piece(
                        768,
                        &[digit(2, 4), digit(4, 16), digit(2, 8), digit(16, 0)],
                        4
                    ),
                    piece(
                        832,
                        &[
                            digit(2, 4),
                            digit(2, 2),
                            digit(2, 16),
                            digit(2, 8),
                            digit(16, 0)
                        ],
                        2
                    ),
                ]
            ),
            field!(
                Codes,
                "qs+hmask",
                FieldEncoding::Unsigned { offset: -4 },
                [
                    piece(
                        256,
                        &[digit(2, 256), digit(4, 2), digit(2, 128), digit(16, 8)],
                        2
                    ),
                    piece(
                        0,
                        &[digit(2, 4), digit(4, 1), digit(2, 128), digit(16, 8)],
                        1
                    ),
                ]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};

/// The `get_scale_min_k4` split of `Q4_K`/`Q5_K`'s 12 scale bytes at bit `base`, over the
/// sub-block `j = jh*4 + jl` of `i = jh*128 + jl*32 + e` (see [`crate::format`]).
macro_rules! k4_sub_scales {
    ($base:literal) => {
        [
            field!(
                SubScale,
                "scales",
                UNSIGNED,
                [
                    piece($base, &[digit(2, 64), digit(4, 8), digit(32, 0)], 4),
                    piece($base + 4, &[digit(2, 2), digit(4, 8), digit(32, 0)], 2),
                ]
            ),
            field!(
                SubMin,
                "scales",
                UNSIGNED,
                [
                    piece($base + 32, &[digit(2, 36), digit(4, 8), digit(32, 0)], 4),
                    piece($base + 36, &[digit(2, 2), digit(4, 8), digit(32, 0)], 2),
                ]
            ),
        ]
    };
}

const K4_SUB_SCALES: [crate::format::BlockField; 2] = k4_sub_scales!(32);

/// `Q4_K` (ggml type 12): `[d f16][dmin f16][scales u8[12]][qs u8[128]]`, 144 bytes per 256.
/// Element `i = j*64 + h*32 + l` is nibble `h` of `qs[j*32 + l]`.
/// `y = d*sc * q - dmin*m`, with `sc`/`m` from `get_scale_min_k4`.
pub(crate) const Q4_K: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q4_K,
    storage: blocks(
        QK_K,
        144,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_QK_K, 16)]),
            field!(Min, "dmin", HALF, [piece(16, WHOLE_QK_K, 16)]),
            K4_SUB_SCALES[0],
            K4_SUB_SCALES[1],
            field!(
                Codes,
                "qs",
                UNSIGNED,
                [piece(128, &[digit(4, 256), digit(2, 4), digit(32, 8)], 4)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: Some(MinSign::Subtract),
};

/// `Q5_K` (ggml type 13): `[d f16][dmin f16][scales u8[12]][qh u8[32]][qs u8[128]]`, 176 bytes
/// per 256. Like `Q4_K`, plus the fifth bit of element `i = j*64 + h*32 + l` in bit `2j + h` of
/// `qh[l]`. `y = d*sc * (q | h << 4) - dmin*m`.
pub(crate) const Q5_K: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q5_K,
    storage: blocks(
        QK_K,
        176,
        &[
            field!(Scale, "d", HALF, [piece(0, WHOLE_QK_K, 16)]),
            field!(Min, "dmin", HALF, [piece(16, WHOLE_QK_K, 16)]),
            K4_SUB_SCALES[0],
            K4_SUB_SCALES[1],
            field!(
                Codes,
                "qs+qh",
                UNSIGNED,
                [
                    piece(384, &[digit(4, 256), digit(2, 4), digit(32, 8)], 4),
                    piece(128, &[digit(4, 2), digit(2, 1), digit(32, 8)], 1),
                ]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: Some(MinSign::Subtract),
};

/// `Q6_K` (ggml type 14): `[ql u8[128]][qh u8[64]][scales i8[16]][d f16]`, 210 bytes per 256.
/// Element `i = n*128 + s*32 + l` (`s = shi*2 + slo`): low nibble `shi` of
/// `ql[n*64 + slo*32 + l]`, high two bits `2s` of `qh[n*32 + l]`, scale `scales[n*8 + s*2 + l/16]`.
/// `y = d*sc * (q - 32)`.
pub(crate) const Q6_K: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Q6_K,
    storage: blocks(
        QK_K,
        210,
        &[
            field!(Scale, "d", HALF, [piece(1664, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                FieldEncoding::Signed,
                [piece(
                    1536,
                    &[digit(2, 64), digit(4, 16), digit(2, 8), digit(16, 0)],
                    8
                )]
            ),
            field!(
                Codes,
                "ql+qh",
                FieldEncoding::Unsigned { offset: -32 },
                [
                    piece(
                        0,
                        &[digit(2, 512), digit(2, 4), digit(2, 256), digit(32, 8)],
                        4
                    ),
                    piece(
                        1024,
                        &[digit(2, 256), digit(2, 4), digit(2, 2), digit(32, 8)],
                        2
                    ),
                ]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};

/// `TQ2_0` (ggml type 35): `[qs u8[64]][d f16]`, 66 bytes per 256. Element `i = n*128 + l*32 + m` is bits
/// `2l` of `qs[n*32 + m]`. `y = (q - 1) * d`, so a code is one of -1, 0, 1 (and 2, which ggml's quantizer
/// never writes but its dequantizer reads).
pub(crate) const TQ2_0: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Tq2_0,
    storage: blocks(
        QK_K,
        66,
        &[
            field!(Scale, "d", HALF, [piece(512, WHOLE_QK_K, 16)]),
            field!(
                Codes,
                "qs",
                FieldEncoding::Unsigned { offset: -1 },
                [piece(0, &[digit(2, 256), digit(4, 2), digit(32, 8)], 2)]
            ),
        ],
    ),
    value: ValueMap::Integer,
    min_sign: None,
};
