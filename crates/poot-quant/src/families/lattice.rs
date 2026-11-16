//! The lattice family: a code names a point of a table, the element's lane picks one cell of the
//! point, and an optional sign pattern flips it.
//!
//! `IQ4_NL` and `IQ4_XS` are lattices of 16 one-lane points whose cells are f32 bit patterns; `IQ1_S` and
//! `IQ1_M` are lattices of 4-bit ternary cells without signs;
//! `IQ2_XS`, `IQ3_XXS` and `IQ3_S` are lattices of four or eight lanes of unsigned magnitude bytes
//! with a sign group of eight values; `TQ1_0` is a lattice of 256 points, one per byte, of five trits read
//! through a lane map that differs over three stretches of the block. One statement serves all of them.

use crate::format::{Digit, FieldEncoding, Segment, segments_at};
use crate::scalar::float_to_f32;

/// A table of `width`-bit cells packed little-endian into `u32` words, the form a device kernel reads.
///
/// A cell never straddles a word, so `width` divides 32. Cell `c` is at bit `c * width` of the words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellTable {
    /// The cells, packed low bits first.
    pub words: &'static [u32],
    /// Bits per cell: 1, 2, 4, 8, 16 or 32.
    pub width: u32,
    /// How a cell's bits become a number: `Unsigned` (an integer plus an offset) or `Float`.
    pub encoding: FieldEncoding,
}

impl CellTable {
    /// The bits of cell `index`.
    pub const fn cell(&self, index: usize) -> u32 {
        let bit = index * self.width as usize;
        let word = self.words[bit / 32] >> (bit % 32);
        if self.width == 32 {
            word
        } else {
            word & ((1u32 << self.width) - 1)
        }
    }

    /// The number cell `index` holds.
    pub fn number(&self, index: usize) -> f32 {
        let bits = self.cell(index);
        match self.encoding {
            FieldEncoding::Unsigned { offset } => (bits as i32 + offset) as f32,
            FieldEncoding::Float(format) => float_to_f32(format, bits),
            FieldEncoding::Signed
            | FieldEncoding::HalfStep { .. }
            | FieldEncoding::SignedPowerOfTwo { .. } => {
                unreachable!("descriptor tests: a cell is an unsigned integer or a float")
            }
        }
    }

    /// Number of cells.
    pub const fn cells(&self) -> usize {
        self.words.len() * 32 / self.width as usize
    }
}

/// How a lattice's `Signs` operand states the eight signs of a group of eight values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignRule {
    /// Seven stored bits are signs 0 to 6 and sign 7 is their parity, which is how ggml's
    /// `ksigns_iq2xs` completes a 7-bit index to a byte of even population count (`IQ2_XS`,
    /// `IQ3_XXS`).
    ParityOfSeven,
    /// Eight stored bits, one sign each (`IQ3_S`).
    Direct,
}

impl SignRule {
    /// Whether value `lane` (0 to 7) of a group of eight is negative under the stored `signs`.
    pub const fn negative(self, signs: u32, lane: u32) -> bool {
        let pattern = match self {
            Self::ParityOfSeven => signs | ((signs.count_ones() & 1) << 7),
            Self::Direct => signs,
        };
        pattern >> lane & 1 == 1
    }
}

/// Which cell of its point an element of a block reads: a piecewise mixed-radix affine map from the in-block
/// element index to a lane, the [`Segment`]s of a [`crate::format::BitLayout`] with lanes for values.
/// Most lattices cycle through their lanes ([`Self::cyclic`]); `TQ1_0`'s lane is the digit of the trit within
/// its byte, which ggml lays out differently over three stretches of the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneMap {
    /// The segments in element order; the first starts at element 0.
    pub segments: &'static [Segment],
}

impl LaneMap {
    /// Element `i` reads lane `i % LANES`.
    pub const fn cyclic<const LANES: u32>() -> Self {
        Self {
            segments: &[Segment {
                first: 0,
                base: 0,
                digits: &[Digit {
                    extent: LANES,
                    stride: 1,
                }],
            }],
        }
    }

    /// The lane of in-block element `element`.
    pub const fn lane(self, element: usize) -> usize {
        segments_at(self.segments, element as u32) as usize
    }

    /// The lane every element `i` with `i % m == r` reads, when they all read one: the digits of each
    /// segment that move the lane are fixed by the residue, and every segment agrees. `None` when the lane
    /// depends on more of the index than the residue says, which the emitter then computes at run time.
    pub fn fixed_by(self, (r, m): (usize, usize)) -> Option<usize> {
        let mut lanes = self.segments.iter().map(|segment| {
            // The residue of the element's offset into the segment, and the modulus it is known to.
            let mut residue = (r + m - segment.first as usize % m) % m;
            let mut known = m;
            let mut lane = segment.base as usize;
            for digit in segment.digits.iter().rev() {
                let extent = digit.extent as usize;
                if known % extent == 0 {
                    lane += residue % extent * digit.stride as usize;
                    residue /= extent;
                    known /= extent;
                } else if digit.stride == 0 {
                    (residue, known) = (0, 1);
                } else {
                    return None;
                }
            }
            Some(lane)
        });
        let first = lanes.next()??;
        lanes.all(|lane| lane == Some(first)).then_some(first)
    }
}

/// A table of points, each a run of `lanes` cells, and the sign rule that applies to them.
///
/// Element `i` of a block decodes to `sign(i) * cells[code * lanes + lane(i)]`, where `code` is the
/// `Codes` operand at `i` and `lane` is the [`LaneMap`]. When [`Self::signs`] is set the sign comes from the `Signs` operand at `i`:
/// the signs of the group of eight values `i / 8 * 8 ..` (bit `j` set means value `j` of the group is
/// negative), whatever [`Self::lanes`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lattice {
    /// The cells of every point, row-major: point `g` is cells `g * lanes .. (g + 1) * lanes`.
    pub cells: CellTable,
    /// Cells one point covers: 1 for the `IQ4` codebooks, 8 for `IQ2_XS`, 4 for `IQ3_XXS` and `IQ3_S`, 8 for
    /// `TQ1_0` (five trits padded to a word). A point is a whole number of `u32` words, so a kernel reads a point
    /// without crossing a word.
    pub lanes: usize,
    /// The lane each element of a block reads.
    pub lane: LaneMap,
    /// The sign rule, or `None` for a lattice without a `Signs` operand.
    pub signs: Option<SignRule>,
}

impl Lattice {
    /// Number of points.
    pub const fn points(&self) -> usize {
        self.cells.cells() / self.lanes
    }

    /// The value of in-block element `element` whose `Codes` operand is `code` and whose `Signs`
    /// operand is `signs` (ignored when [`Self::signs`] is `None`).
    #[inline(always)]
    pub fn value(&self, code: u32, element: usize, signs: u32) -> f32 {
        let magnitude = self
            .cells
            .number(code as usize * self.lanes + self.lane.lane(element));
        match self.signs {
            Some(rule) if rule.negative(signs, (element % 8) as u32) => -magnitude,
            _ => magnitude,
        }
    }
}

/// The bit patterns of a table of f32 cells, the word form of a 32-bit float [`CellTable`].
pub(crate) const fn f32_words<const N: usize>(values: [f32; N]) -> [u32; N] {
    let mut words = [0u32; N];
    let mut index = 0;
    while index < N {
        words[index] = values[index].to_bits();
        index += 1;
    }
    words
}

/// A table of 8-lane points of byte cells, each point a little-endian `u64` as ggml writes it, as the two
/// words per point of a 8-bit [`CellTable`]. `WORDS` is twice the number of points.
pub(crate) const fn u64_words<const POINTS: usize, const WORDS: usize>(
    points: &[u64; POINTS],
) -> [u32; WORDS] {
    assert!(WORDS == 2 * POINTS, "two words per 8-lane point");
    let mut words = [0u32; WORDS];
    let mut point = 0;
    while point < POINTS {
        words[2 * point] = points[point] as u32;
        words[2 * point + 1] = (points[point] >> 32) as u32;
        point += 1;
    }
    words
}

/// A table of 8-lane points of signed bytes in `-1..=1`, each point a little-endian `u64` as ggml writes it
/// (`iq1s_grid`), as the one word per point of a 4-bit [`CellTable`] whose cells are the bytes plus one
/// (`Unsigned { offset: -1 }`). `iq1s_grid_gpu` in `ggml-common.h` is this packing.
pub(crate) const fn ternary_words<const POINTS: usize>(points: &[u64; POINTS]) -> [u32; POINTS] {
    let mut words = [0u32; POINTS];
    let mut point = 0;
    while point < POINTS {
        let mut lane = 0;
        while lane < 8 {
            let byte = (points[point] >> (8 * lane)) as u8 as i8;
            assert!(-1 <= byte && byte <= 1, "a ternary grid cell");
            words[point] |= ((byte + 1) as u32) << (4 * lane);
            lane += 1;
        }
        point += 1;
    }
    words
}

/// The table of `TQ1_0`: for each byte `b`, the five trits ggml reads from it, one per four-bit lane as
/// `((uint8_t)(b * 3^n) * 3) >> 8` (`dequantize_row_tq1_0`: `q = qs[j] * pow3[n]` truncated to a byte, then
/// `xi = (q * 3) >> 8`), a digit of 0, 1 or 2 stored as the cell the `Unsigned { offset: -1 }` encoding reads
/// as `-1`, `0` or `1`. Lanes 5 to 7 pad the point to a word and hold the cell of `0`.
pub(crate) const fn trit_words() -> [u32; 256] {
    let mut words = [0u32; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut power = 1u32;
        let mut lane = 0;
        while lane < 8 {
            let cell = if lane < 5 {
                let times_power = (byte as u32 * power) as u8;
                let trit = (times_power as u32 * 3) >> 8;
                power *= 3;
                trit
            } else {
                1
            };
            words[byte] |= cell << (4 * lane);
            lane += 1;
        }
        byte += 1;
    }
    words
}
