//! The table formats: every descriptor whose codes name a point of a [`Lattice`]. `IQ4_NL` and `IQ4_XS`
//! read 16 one-lane points of f32 cells (ggml's `kvalues_iq4nl`); `IQ1_S` and `IQ1_M` read points of eight
//! ternary cells and add a one-bit shift; `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`
//! and `IQ3_S` read points of four or eight byte lanes and a sign pattern per eight values.
//!
//! The grids are ggml's own tables (`grids.rs`); a lattice's cells are those tables' words, with no
//! per-format expansion.

use super::{HALF, HALVES_32, QK_K, UNSIGNED, WHOLE_32, WHOLE_QK_K, blocks, digit, segment};
use crate::families::lattice::{
    CellTable, LaneMap, Lattice, SignRule, f32_words, ternary_words, trit_words, u64_words,
};
use crate::format::{FieldEncoding, FloatFormat, FormatDescriptor, ValueMap, WeightFormat};
use crate::grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};

/// ggml's `kvalues_iq4nl`: the `IQ4_NL`/`IQ4_XS` codebook.
pub(crate) const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

const IQ4NL_WORDS: [u32; 16] = f32_words(KVALUES_IQ4NL);
const IQ1S_WORDS: [u32; 2048] = ternary_words(&IQ1S_GRID);
const TQ1_WORDS: [u32; 256] = trit_words();
const IQ2XXS_WORDS: [u32; 512] = u64_words(&IQ2XXS_GRID);
const IQ2XS_WORDS: [u32; 1024] = u64_words(&IQ2XS_GRID);
const IQ2S_WORDS: [u32; 2048] = u64_words(&IQ2S_GRID);

/// The `IQ4` codebook as a lattice: 16 points of one f32 cell, no signs.
const IQ4NL_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ4NL_WORDS,
        width: 32,
        encoding: FieldEncoding::Float(FloatFormat::F32),
    },
    lanes: 1,
    lane: LaneMap::cyclic::<1>(),
    signs: None,
};

/// The `IQ1_S` and `IQ1_M` lattice: 2048 points of eight ternary cells (`iq1s_grid`: `-1`, `0`, `1`), four
/// bits each, so a point is one word and the table 8 KiB. No signs: the cells are signed already.
pub(crate) const IQ1S_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ1S_WORDS,
        width: 4,
        encoding: FieldEncoding::Unsigned { offset: -1 },
    },
    lanes: 8,
    lane: LaneMap::cyclic::<8>(),
    signs: None,
};

/// The `TQ1_0` lattice: 256 points, one per byte of `qs` or `qh`, of the five trits that byte holds
/// (`-1`, `0`, `1`; [`trit_words`]), four bits each. A point is padded to eight lanes, one word, so a kernel
/// reads it like any other; the three padding cells are never read. The lane of an element is its trit's
/// position in the byte, which ggml's dequantizer walks as `n` in `i = j + n*32 + m`: five trits of the
/// first 160 elements from `qs[0..32]`, five of the next 80 from `qs[32..48]` over 16 bytes, and four of the
/// last 16 from `qh[0..4]`.
const TQ1_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &TQ1_WORDS,
        width: 4,
        encoding: FieldEncoding::Unsigned { offset: -1 },
    },
    lanes: 8,
    lane: LaneMap {
        segments: &[
            segment(0, 0, &[digit(5, 1), digit(32, 0)]),
            segment(160, 0, &[digit(5, 1), digit(16, 0)]),
            segment(240, 0, &[digit(4, 1), digit(4, 0)]),
        ],
    },
    signs: None,
};

/// The `IQ2_XXS` lattice: 256 points of eight magnitude bytes (`ggml-common.h` `iq2xxs_grid`).
pub(crate) const IQ2XXS_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ2XXS_WORDS,
        width: 8,
        encoding: UNSIGNED,
    },
    lanes: 8,
    lane: LaneMap::cyclic::<8>(),
    signs: Some(SignRule::ParityOfSeven),
};

/// The `IQ2_XS` lattice: 512 points of eight magnitude bytes (`ggml-common.h` `iq2xs_grid`).
pub(crate) const IQ2XS_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ2XS_WORDS,
        width: 8,
        encoding: UNSIGNED,
    },
    lanes: 8,
    lane: LaneMap::cyclic::<8>(),
    signs: Some(SignRule::ParityOfSeven),
};

/// The `IQ2_S` lattice: 1024 points of eight magnitude bytes (`iq2s_grid`), with a full byte of signs
/// per eight values.
pub(crate) const IQ2S_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ2S_WORDS,
        width: 8,
        encoding: UNSIGNED,
    },
    lanes: 8,
    lane: LaneMap::cyclic::<8>(),
    signs: Some(SignRule::Direct),
};

/// The `IQ3_XXS` lattice: 256 points of four magnitude bytes, one word each (`iq3xxs_grid`).
pub(crate) const IQ3XXS_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ3XXS_GRID,
        width: 8,
        encoding: UNSIGNED,
    },
    lanes: 4,
    lane: LaneMap::cyclic::<4>(),
    signs: Some(SignRule::ParityOfSeven),
};

/// The `IQ3_S` lattice: 512 points of four magnitude bytes (`iq3s_grid`), with a full byte of signs
/// per eight values.
pub(crate) const IQ3S_LATTICE: Lattice = Lattice {
    cells: CellTable {
        words: &IQ3S_GRID,
        width: 8,
        encoding: UNSIGNED,
    },
    lanes: 4,
    lane: LaneMap::cyclic::<4>(),
    signs: Some(SignRule::Direct),
};

/// `IQ4_NL` (ggml type 20): `[d f16][qs u8[16]]`, 18 bytes per 32. `y = d * kvalues_iq4nl[q]`.
pub(crate) const IQ4_NL: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq4_Nl,
    storage: blocks(
        32,
        18,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_32, 16)]),
            field!(Codes, "qs", UNSIGNED, [piece!(16, HALVES_32, 4)]),
        ],
    ),
    value: ValueMap::Lattice(&IQ4NL_LATTICE),
    min_sign: None,
};

/// `IQ4_XS` (ggml type 23): `[d f16][scales_h u16][scales_l u8[4]][qs u8[128]]`, 136 bytes per
/// 256. Element `i = ib*32 + h*16 + j` is nibble `h` of `qs[ib*16 + j]`; sub-block `ib`'s 6-bit
/// scale takes nibble `ib % 2` of `scales_l[ib/2]` and bits `2*ib` of `scales_h`.
/// `y = d*(ls - 32) * kvalues_iq4nl[q]`.
pub(crate) const IQ4_XS: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq4_Xs,
    storage: blocks(
        QK_K,
        136,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales_l+scales_h",
                FieldEncoding::Unsigned { offset: -32 },
                [
                    piece!(32, &[digit(8, 4), digit(32, 0)], 4),
                    piece!(16, &[digit(8, 2), digit(32, 0)], 2),
                ]
            ),
            field!(
                Codes,
                "qs",
                UNSIGNED,
                [piece!(64, &[digit(8, 128), digit(2, 4), digit(16, 8)], 4)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ4NL_LATTICE),
    min_sign: None,
};

/// `IQ2_XXS` (ggml type 16): `[d f16][qs u16[32]]`, 66 bytes per 256. Thirty-two-value run `ib` reads
/// the eight bytes `qs[4*ib..4*ib + 4]` (as u16): the first four bytes are the grid points of its four
/// eight-value groups, the last four (a little-endian u32) hold their sign patterns (7 bits each, the
/// eighth sign is the parity) and, in bits 28..32, the half-step scale.
/// `y = d*(s + 1/2)/4 * sign * grid[point][i % 8]`.
pub(crate) const IQ2_XXS: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq2_Xxs,
    storage: blocks(
        QK_K,
        66,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "qs (scale)",
                FieldEncoding::HalfStep { exponent: -2 },
                [piece!(76, &[digit(8, 64), digit(32, 0)], 4)]
            ),
            field!(
                Codes,
                "qs (grid point)",
                UNSIGNED,
                [piece!(16, &[digit(8, 64), digit(4, 8), digit(8, 0)], 8)]
            ),
            field!(
                Signs,
                "qs (signs)",
                UNSIGNED,
                [piece!(48, &[digit(8, 64), digit(4, 7), digit(8, 0)], 7)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ2XXS_LATTICE),
    min_sign: None,
};

/// `IQ2_XS` (ggml type 17): `[d f16][qs u16[32]][scales u8[8]]`, 74 bytes per 256. Element
/// `i = g*8 + l` (eight-value group `g`, lane `l`) reads `qs[g]`: its low 9 bits are the grid point
/// and its high 7 bits the group's sign pattern. Sixteen-value run `i / 16` takes nibble `(i/16) % 2`
/// of `scales[i/32]`. `y = d*(s + 1/2)/4 * sign * grid[point][l]`.
pub(crate) const IQ2_XS: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq2_Xs,
    storage: blocks(
        QK_K,
        74,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                FieldEncoding::HalfStep { exponent: -2 },
                [piece!(528, &[digit(16, 4), digit(16, 0)], 4)]
            ),
            field!(
                Codes,
                "qs (grid point)",
                UNSIGNED,
                [piece!(16, &[digit(32, 16), digit(8, 0)], 9)]
            ),
            field!(
                Signs,
                "qs (signs)",
                UNSIGNED,
                [piece!(25, &[digit(32, 16), digit(8, 0)], 7)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ2XS_LATTICE),
    min_sign: None,
};

/// `IQ3_XXS` (ggml type 18): `[d f16][qs u8[64]][scales_and_signs u32[8]]`, 98 bytes per 256.
/// Element `i` reads grid point `qs[i/4]` (four lanes). Thirty-two-value run `ib` reads
/// `scales_and_signs[ib]`: bits 0..28 are the sign patterns of its four eight-value groups (7 bits
/// each) and bits 28..32 the half-step scale. `y = d*(s + 1/2)/2 * sign * grid[point][i % 4]`.
pub(crate) const IQ3_XXS: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq3_Xxs,
    storage: blocks(
        QK_K,
        98,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales_and_signs (scale)",
                FieldEncoding::HalfStep { exponent: -1 },
                [piece!(556, &[digit(8, 32), digit(32, 0)], 4)]
            ),
            field!(
                Codes,
                "qs",
                UNSIGNED,
                [piece!(16, &[digit(64, 8), digit(4, 0)], 8)]
            ),
            field!(
                Signs,
                "scales_and_signs (signs)",
                UNSIGNED,
                [piece!(528, &[digit(8, 32), digit(4, 7), digit(8, 0)], 7)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ3XXS_LATTICE),
    min_sign: None,
};

/// `IQ3_S` (ggml type 21): `[d f16][qs u8[64]][qh u8[8]][signs u8[32]][scales u8[4]]`, 110 bytes per
/// 256. Element `i` reads grid point `qs[i/4] | bit (i%32)/4 of qh[i/32] << 8` (four lanes), the sign
/// byte `signs[i/8]` (bit `i%8`), and the scale nibble `i/32` of `scales`.
/// `y = d*(2s + 1) * sign * grid[point][i % 4]`.
pub(crate) const IQ3_S: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq3_S,
    storage: blocks(
        QK_K,
        110,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                FieldEncoding::HalfStep { exponent: 1 },
                [piece!(848, &[digit(8, 4), digit(32, 0)], 4)]
            ),
            field!(
                Codes,
                "qs + qh (grid point)",
                UNSIGNED,
                [
                    piece!(16, &[digit(64, 8), digit(4, 0)], 8),
                    piece!(528, &[digit(8, 8), digit(8, 1), digit(4, 0)], 1),
                ]
            ),
            field!(
                Signs,
                "signs",
                UNSIGNED,
                [piece!(592, &[digit(32, 8), digit(8, 0)], 8)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ3S_LATTICE),
    min_sign: None,
};

/// `IQ2_S` (ggml type 22): `[d f16][qs u8[64]][qh u8[8]][scales u8[8]]`, 82 bytes per 256. The first
/// 32 bytes of `qs` are the low grid bits and the last 32 the sign bytes. Element `i = ib*32 + l*8 + j`
/// reads grid point `qs[4*ib + l] | (qh[ib] >> 2*l & 3) << 8`, sign byte `qs[32 + 4*ib + l]` (bit `j`)
/// and nibble `(i/16) % 2` of `scales[i/32]`. `y = d*(s + 1/2)/4 * sign * grid[point][j]`.
pub(crate) const IQ2_S: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq2_S,
    storage: blocks(
        QK_K,
        82,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "scales",
                FieldEncoding::HalfStep { exponent: -2 },
                [piece!(592, &[digit(16, 4), digit(16, 0)], 4)]
            ),
            field!(
                Codes,
                "qs + qh (grid point)",
                UNSIGNED,
                [
                    piece!(16, &[digit(8, 32), digit(4, 8), digit(8, 0)], 8),
                    piece!(528, &[digit(8, 8), digit(4, 2), digit(8, 0)], 2),
                ]
            ),
            field!(
                Signs,
                "qs (signs)",
                UNSIGNED,
                [piece!(272, &[digit(8, 32), digit(4, 8), digit(8, 0)], 8)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ2S_LATTICE),
    min_sign: None,
};

/// `IQ1_S` (ggml type 19): `[d f16][qs u8[32]][qh u16[8]]`, 50 bytes per 256. Thirty-two-value run `ib` reads
/// the u16 `qh[ib]`: bits 0..12 are the three high bits of the grid points of its four eight-value groups
/// (the low eight bits are `qs[4*ib + l]`), bits 12..15 the scale `s` and bit 15 the sign of the shift.
/// `y = d*(2s + 1) * (grid[point][i % 8] + (+/-1/8))`.
pub(crate) const IQ1_S: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq1_S,
    storage: blocks(
        QK_K,
        50,
        &[
            field!(Scale, "d", HALF, [piece!(0, WHOLE_QK_K, 16)]),
            field!(
                SubScale,
                "qh (scale)",
                FieldEncoding::HalfStep { exponent: 1 },
                [piece!(284, &[digit(8, 16), digit(32, 0)], 3)]
            ),
            field!(
                Codes,
                "qs + qh (grid point)",
                UNSIGNED,
                [
                    piece!(16, &[digit(32, 8), digit(8, 0)], 8),
                    piece!(272, &[digit(8, 16), digit(4, 3), digit(8, 0)], 3),
                ]
            ),
            field!(
                Shift,
                "qh (delta)",
                FieldEncoding::SignedPowerOfTwo { exponent: -3 },
                [piece!(287, &[digit(8, 16), digit(32, 0)], 1)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ1S_LATTICE),
    min_sign: None,
};

/// `IQ1_M` (ggml type 29): `[qs u8[32]][qh u8[16]][scales u16[4]]`, 56 bytes per 256, with no `d`: the f16
/// super-scale is the top nibble of each of the four `scales` words, lowest word first. Element
/// `i = g*8 + j` (eight-value group `g`) reads grid point `qs[g] | (3 bits of qh[g/2] at 4*(g%2)) << 8`, the
/// shift sign at bit `3` (even `g`) or `7` (odd `g`) of `qh[g/2]`, and the 3-bit scale `s` of its sixteen
/// values at bit `6*(ib%2) + 3*((i/16)%2)` of `scales[ib/2]` (`ib = i/32`).
/// `y = d*(2s + 1) * (grid[point][j] + (+/-1/8))`.
pub(crate) const IQ1_M: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Iq1_M,
    storage: blocks(
        QK_K,
        56,
        &[
            field!(
                Scale,
                "scales (super-scale)",
                HALF,
                [
                    piece!(396, WHOLE_QK_K, 4),
                    piece!(412, WHOLE_QK_K, 4),
                    piece!(428, WHOLE_QK_K, 4),
                    piece!(444, WHOLE_QK_K, 4),
                ]
            ),
            field!(
                SubScale,
                "scales (scale)",
                FieldEncoding::HalfStep { exponent: 1 },
                [piece!(
                    384,
                    &[digit(4, 16), digit(2, 6), digit(2, 3), digit(16, 0)],
                    3
                )]
            ),
            field!(
                Codes,
                "qs + qh (grid point)",
                UNSIGNED,
                [
                    piece!(0, &[digit(32, 8), digit(8, 0)], 8),
                    piece!(256, &[digit(16, 8), digit(2, 4), digit(8, 0)], 3),
                ]
            ),
            field!(
                Shift,
                "qh (delta)",
                FieldEncoding::SignedPowerOfTwo { exponent: -3 },
                [piece!(259, &[digit(16, 8), digit(2, 4), digit(8, 0)], 1)]
            ),
        ],
    ),
    value: ValueMap::Lattice(&IQ1S_LATTICE),
    min_sign: None,
};

/// `TQ1_0` (ggml type 34): `[qs u8[48]][qh u8[4]][d f16]`, 54 bytes per 256, `d` last. The codes are bytes
/// of five base-3 digits each (`qs`: 5 digits x 32 bytes, then 5 x 16; `qh`: 4 x 4), and element `i` takes
/// the digit its lane map names of the byte its layout names: `y = (digit - 1) * d`.
pub(crate) const TQ1_0: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::Tq1_0,
    storage: blocks(
        QK_K,
        54,
        &[
            field!(
                Codes,
                "qs, qh",
                UNSIGNED,
                [piecewise!(
                    8,
                    [
                        segment(0, 0, &[digit(5, 0), digit(32, 8)]),
                        segment(160, 256, &[digit(5, 0), digit(16, 8)]),
                        segment(240, 384, &[digit(4, 0), digit(4, 8)]),
                    ]
                )]
            ),
            field!(Scale, "d", HALF, [piece!(416, WHOLE_QK_K, 16)]),
        ],
    ),
    value: ValueMap::Lattice(&TQ1_LATTICE),
    min_sign: None,
};
