//! Whole-tensor decoding of the block formats, compiled from their descriptors at compile time.
//!
//! [`ConstPlan::compile`] is a `const fn` over a [`FormatDescriptor`]: it evaluates the
//! descriptor's [`BitLayout`](crate::format::BitLayout)s for every element and groups the
//! repeating operands (`d`, the sub-block scales and mins) into slots. Each quantized block format
//! gets one instantiation of the same generic decoder, so rustc sees that format's offsets, widths,
//! value count and scale groups as constants. Nothing here states a layout of its own. The
//! interpreter in [`crate::decode`] stays the reference, and the tests pin every format to it bit
//! for bit. Dense formats (one float per block) decode in one loop over the tensor.

use crate::decode::DecodeError;
use crate::format::{
    BlockField, Field, FieldEncoding, FloatFormat, FormatDescriptor, MinSign, OperandRole, Storage,
    ValueMap, WeightFormat,
};
use crate::scalar::float_to_f32;

/// The largest block, plus zero padding so an 8-byte window never leaves the scratch.
const MAX_BLOCK_BYTES: usize = 256;
type Padded = [u8; MAX_BLOCK_BYTES + 8];

/// At most this many distinct slots of one repeating factor per block.
const MAX_SLOTS: usize = 16;

/// One field at one element: up to two pieces, low bits first.
#[derive(Clone, Copy, Debug)]
struct Pieces {
    two: bool,
    byte: [usize; 2],
    shift: [u32; 2],
    mask: [u64; 2],
    low_width: u32,
}

impl Pieces {
    const EMPTY: Self = Self {
        two: false,
        byte: [0; 2],
        shift: [0; 2],
        mask: [0; 2],
        low_width: 0,
    };

    const fn at(field: &Field, index: u32) -> Self {
        assert!(
            !field.pieces.is_empty() && field.pieces.len() <= 2,
            "a field of one or two pieces"
        );
        let mut pieces = Self::EMPTY;
        pieces.two = field.pieces.len() == 2;
        pieces.low_width = field.pieces[0].width;
        let mut slot = 0;
        while slot < field.pieces.len() {
            let piece = field.pieces[slot];
            let bit = piece.layout.bit(index);
            pieces.byte[slot] = (bit / 8) as usize;
            pieces.shift[slot] = bit % 8;
            pieces.mask[slot] = (1u64 << piece.width) - 1;
            slot += 1;
        }
        pieces
    }

    const fn same(&self, other: &Self) -> bool {
        self.two == other.two
            && self.byte[0] == other.byte[0]
            && self.byte[1] == other.byte[1]
            && self.shift[0] == other.shift[0]
            && self.shift[1] == other.shift[1]
            && self.mask[0] == other.mask[0]
            && self.mask[1] == other.mask[1]
    }

    #[inline(always)]
    fn piece(&self, padded: &Padded, slot: usize) -> u32 {
        let byte = self.byte[slot];
        let window = u64::from_le_bytes(padded[byte..byte + 8].try_into().unwrap());
        ((window >> self.shift[slot]) & self.mask[slot]) as u32
    }

    /// The raw bits.
    #[inline(always)]
    fn read(&self, padded: &Padded) -> u32 {
        let low = self.piece(padded, 0);
        if self.two {
            low | (self.piece(padded, 1) << self.low_width)
        } else {
            low
        }
    }
}

/// How raw bits become a number.
#[derive(Clone, Copy, Debug)]
struct Reader {
    encoding: FieldEncoding,
    width: u32,
}

impl Reader {
    const fn of(field: &Field) -> Self {
        Self {
            encoding: field.encoding,
            width: field.width(),
        }
    }

    #[inline(always)]
    fn number(self, bits: u32) -> f32 {
        match self.encoding {
            FieldEncoding::Unsigned { offset } => (bits as i32 + offset) as f32,
            FieldEncoding::Signed => {
                let unused = 32 - self.width;
                (((bits << unused) as i32) >> unused) as f32
            }
            FieldEncoding::Float(format) => float_to_f32(format, bits),
        }
    }

    #[inline(always)]
    fn integer(self, bits: u32) -> i32 {
        match self.encoding {
            FieldEncoding::Unsigned { offset } => bits as i32 + offset,
            FieldEncoding::Signed => {
                let unused = 32 - self.width;
                ((bits << unused) as i32) >> unused
            }
            FieldEncoding::Float(_) => 0,
        }
    }
}

/// A repeating operand (`d`, a sub-block scale): its distinct slots and each element's pick.
#[derive(Clone, Copy, Debug)]
struct Factor<const N: usize> {
    present: bool,
    role: OperandRole,
    reader: Reader,
    slots: [Pieces; MAX_SLOTS],
    count: usize,
    slot_of: [usize; N],
}

impl<const N: usize> Factor<N> {
    const ABSENT: Self = Self {
        present: false,
        role: OperandRole::Scale,
        reader: Reader {
            encoding: FieldEncoding::Unsigned { offset: 0 },
            width: 0,
        },
        slots: [Pieces::EMPTY; MAX_SLOTS],
        count: 0,
        slot_of: [0; N],
    };

    const fn compile(field: Option<&BlockField>) -> Self {
        let Some(field) = field else {
            return Self::ABSENT;
        };
        let mut factor = Self::ABSENT;
        factor.present = true;
        factor.role = field.role;
        factor.reader = Reader::of(&field.field);
        let mut index = 0;
        while index < N {
            let pieces = Pieces::at(&field.field, index as u32);
            let mut slot = 0;
            while slot < factor.count && !factor.slots[slot].same(&pieces) {
                slot += 1;
            }
            if slot == factor.count {
                assert!(factor.count < MAX_SLOTS, "at most 16 slots of one factor");
                factor.slots[slot] = pieces;
                factor.count += 1;
            }
            factor.slot_of[index] = slot;
            index += 1;
        }
        factor
    }

    /// Every slot's value for one block; a NaN is refused.
    #[inline(always)]
    fn load(
        &self,
        padded: &Padded,
        format: WeightFormat,
        values: &mut [f32; MAX_SLOTS],
    ) -> Result<(), DecodeError> {
        let mut nan = false;
        for (value, pieces) in values.iter_mut().zip(&self.slots[..self.count]) {
            *value = self.reader.number(pieces.read(padded));
            nan |= value.is_nan();
        }
        if nan {
            return Err(DecodeError::NotANumber {
                format,
                role: self.role,
            });
        }
        Ok(())
    }
}

/// `unrolled!(N, index => body)`: `body` pasted once per element index below `N` (at most 256),
/// with `index` bound to a literal. After inlining, each copy sees its index as a constant, so a
/// const plan's per-element table entries fold into the code instead of being loaded, which is
/// what lets rustc vectorize a block. (A closure called 256 times defeats the inliner; the body
/// has to be pasted.)
macro_rules! unrolled {
    ($n:expr, $index:ident => $body:block) => {{
        const { assert!($n <= 256, "a block of at most 256 values") };
        macro_rules! element {
            ($at:expr) => {{
                let $index: usize = $at;
                if $index < $n {
                    $body
                }
            }};
        }
        macro_rules! sixteen {
            ($base:expr) => {
                element!($base);
                element!($base + 1);
                element!($base + 2);
                element!($base + 3);
                element!($base + 4);
                element!($base + 5);
                element!($base + 6);
                element!($base + 7);
                element!($base + 8);
                element!($base + 9);
                element!($base + 10);
                element!($base + 11);
                element!($base + 12);
                element!($base + 13);
                element!($base + 14);
                element!($base + 15);
            };
        }
        sixteen!(0);
        sixteen!(16);
        sixteen!(32);
        sixteen!(48);
        sixteen!(64);
        sixteen!(80);
        sixteen!(96);
        sixteen!(112);
        sixteen!(128);
        sixteen!(144);
        sixteen!(160);
        sixteen!(176);
        sixteen!(192);
        sixteen!(208);
        sixteen!(224);
        sixteen!(240);
    }};
}

/// A block format of `N` values compiled at compile time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstPlan<const N: usize> {
    format: WeightFormat,
    bytes: usize,
    value: ValueMap,
    min_sign: Option<MinSign>,
    codes: Reader,
    code_pieces: [Pieces; N],
    /// `Scale`, `SubScale`, `Min`, `SubMin`.
    factors: [Factor<N>; 4],
}

/// Per-block factor slot values.
type Slots = [[f32; MAX_SLOTS]; 4];

impl<const N: usize> ConstPlan<N> {
    pub(crate) const fn compile(descriptor: FormatDescriptor) -> Self {
        let Storage::Blocks(layout) = descriptor.storage else {
            panic!("only block formats have a const plan");
        };
        assert!(
            layout.values == N,
            "the plan's N is the block's value count"
        );
        assert!(
            layout.bytes <= MAX_BLOCK_BYTES,
            "a block of at most 256 bytes"
        );
        assert!(
            layout.const_field(OperandRole::Zero).is_none(),
            "a block format has no zero point"
        );
        let Some(codes) = layout.const_field(OperandRole::Codes) else {
            panic!("every block format has a Codes field");
        };
        let mut code_pieces = [Pieces::EMPTY; N];
        let mut index = 0;
        while index < N {
            code_pieces[index] = Pieces::at(&codes.field, index as u32);
            index += 1;
        }
        let factors = [
            Factor::compile(layout.const_field(OperandRole::Scale)),
            Factor::compile(layout.const_field(OperandRole::SubScale)),
            Factor::compile(layout.const_field(OperandRole::Min)),
            Factor::compile(layout.const_field(OperandRole::SubMin)),
        ];
        Self {
            format: descriptor.format,
            bytes: layout.bytes,
            value: descriptor.value,
            min_sign: descriptor.min_sign,
            codes: Reader::of(&codes.field),
            code_pieces,
            factors,
        }
    }

    /// Load one block and its factor slots.
    #[inline(always)]
    fn load(
        &self,
        block: &[u8],
        padded: &mut Padded,
        slots: &mut Slots,
    ) -> Result<(), DecodeError> {
        padded[..self.bytes].copy_from_slice(block);
        for (factor, values) in self.factors.iter().zip(slots.iter_mut()) {
            if factor.present {
                factor.load(padded, self.format, values)?;
            }
        }
        Ok(())
    }

    /// `(Scale * SubScale, Min * SubMin)` of element `index`, multiplied left to right.
    #[inline(always)]
    fn scale_and_min(&self, slots: &Slots, index: usize) -> (f32, f32) {
        let factor = |which: usize| {
            let factor = &self.factors[which];
            factor.present.then(|| slots[which][factor.slot_of[index]])
        };
        let product = |lhs: Option<f32>, rhs: Option<f32>| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => lhs * rhs,
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => 1.0,
        };
        (product(factor(0), factor(1)), product(factor(2), factor(3)))
    }

    /// Decode a run of whole blocks into `out`.
    #[inline(always)]
    pub(crate) fn decode_blocks(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), DecodeError> {
        let mut padded: Padded = [0; MAX_BLOCK_BYTES + 8];
        let mut slots: Slots = [[0.0; MAX_SLOTS]; 4];
        for (block, out) in bytes.chunks_exact(self.bytes).zip(out.chunks_exact_mut(N)) {
            let out: &mut [f32; N] = out.try_into().expect("chunks of N values");
            self.load(block, &mut padded, &mut slots)?;
            self.decode_block(&padded, &slots, out);
            if matches!(self.value, ValueMap::Float) && out.iter().any(|value| value.is_nan()) {
                return Err(DecodeError::NotANumber {
                    format: self.format,
                    role: OperandRole::Codes,
                });
            }
        }
        Ok(())
    }

    /// Decode one loaded block, element by element through `unrolled!`.
    #[inline(always)]
    fn decode_block(&self, padded: &Padded, slots: &Slots, out: &mut [f32; N]) {
        let has_scale = self.factors[0].present || self.factors[1].present;
        unrolled!(N, index => {
            let bits = self.code_pieces[index].read(padded);
            let decoded = match self.value {
                ValueMap::Integer => self.codes.integer(bits) as f32,
                ValueMap::Codebook(table) => table[(bits & 0xf) as usize],
                ValueMap::Float => self.codes.number(bits),
            };
            let (scale, min) = self.scale_and_min(slots, index);
            let scaled = if has_scale { scale * decoded } else { decoded };
            out[index] = match self.min_sign {
                None => scaled,
                Some(MinSign::Add) => scaled + min,
                Some(MinSign::Subtract) => scaled - min,
            };
        });
    }
}

/// Bind `$plan` to `$format`'s const plan and evaluate `$body`, or `$otherwise` for a format
/// without one.
macro_rules! with_plan {
    ($format:expr, |$plan:ident| $body:expr, otherwise $otherwise:expr) => {{
        use crate::blocks;
        macro_rules! arm {
            ($descriptor:path, $values:expr) => {{
                const PLAN: &ConstPlan<{ $values }> = &ConstPlan::compile($descriptor);
                let $plan = PLAN;
                $body
            }};
        }
        match $format {
            WeightFormat::Q4_0 => arm!(blocks::Q4_0, 32),
            WeightFormat::Q4_1 => arm!(blocks::Q4_1, 32),
            WeightFormat::Q5_0 => arm!(blocks::Q5_0, 32),
            WeightFormat::Q5_1 => arm!(blocks::Q5_1, 32),
            WeightFormat::Q8_0 => arm!(blocks::Q8_0, 32),
            WeightFormat::Iq4_Nl => arm!(blocks::IQ4_NL, 32),
            WeightFormat::Mxfp4 => arm!(blocks::MXFP4, 32),
            WeightFormat::Q2_K => arm!(blocks::Q2_K, blocks::QK_K),
            WeightFormat::Q3_K => arm!(blocks::Q3_K, blocks::QK_K),
            WeightFormat::Q4_K => arm!(blocks::Q4_K, blocks::QK_K),
            WeightFormat::Q5_K => arm!(blocks::Q5_K, blocks::QK_K),
            WeightFormat::Q6_K => arm!(blocks::Q6_K, blocks::QK_K),
            WeightFormat::Iq4_Xs => arm!(blocks::IQ4_XS, blocks::QK_K),
            _ => $otherwise,
        }
    }};
}

/// Check that `bytes` hold `values` values of `format`'s whole blocks.
fn check_blocks(format: WeightFormat, bytes: &[u8], values: usize) -> Result<(), DecodeError> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        return Err(DecodeError::WrongStorage { format });
    };
    if !values.is_multiple_of(layout.values)
        || (values / layout.values).checked_mul(layout.bytes) != Some(bytes.len())
    {
        return Err(DecodeError::RowLength {
            format,
            bytes: bytes.len(),
            values,
        });
    }
    Ok(())
}

/// Decode a run of whole blocks of a block format into `out`.
pub fn decode_blocks(
    format: WeightFormat,
    bytes: &[u8],
    out: &mut [f32],
) -> Result<(), DecodeError> {
    check_blocks(format, bytes, out.len())?;
    with_plan!(format, |plan| plan.decode_blocks(bytes, out), otherwise decode_dense(format, bytes, out))
}

/// A dense format: every one-value block is one little-endian float of the codes field's format.
fn decode_dense(format: WeightFormat, bytes: &[u8], out: &mut [f32]) -> Result<(), DecodeError> {
    let descriptor = format.descriptor();
    let Storage::Blocks(layout) = descriptor.storage else {
        return Err(DecodeError::WrongStorage { format });
    };
    let codes = layout
        .field(OperandRole::Codes)
        .expect("every block format has a Codes field");
    let (1, FieldEncoding::Float(float), None) = (
        layout.values,
        codes.field.encoding,
        layout.field(OperandRole::Scale),
    ) else {
        unreachable!("descriptor tests: every block format without a const plan is dense");
    };
    decode_dense_floats(float, layout.bytes, bytes, out);
    if out.iter().any(|value| value.is_nan()) {
        return Err(DecodeError::NotANumber {
            format,
            role: OperandRole::Codes,
        });
    }
    Ok(())
}

fn decode_dense_floats(float: FloatFormat, width: usize, bytes: &[u8], out: &mut [f32]) {
    for (value, bytes) in out.iter_mut().zip(bytes.chunks_exact(width)) {
        let bits = bytes
            .iter()
            .rev()
            .fold(0u32, |bits, &byte| (bits << 8) | u32::from(byte));
        *value = float_to_f32(float, bits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_FORMATS: [WeightFormat; 16] = [
        WeightFormat::F32,
        WeightFormat::F16,
        WeightFormat::Bf16,
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q8_0,
        WeightFormat::Q2_K,
        WeightFormat::Q3_K,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
        WeightFormat::Iq4_Nl,
        WeightFormat::Iq4_Xs,
        WeightFormat::Mxfp4,
    ];

    /// xorshift64 bytes: deterministic blocks covering every bit pattern of every field, NaN
    /// scales included.
    fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    /// The compiled plan decodes every block exactly as the reference interpreter: same bits, and
    /// the same error for a block holding a NaN operand.
    #[test]
    fn the_block_plan_decodes_every_format_as_the_interpreter() {
        for format in BLOCK_FORMATS {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                unreachable!()
            };
            let mut refused = 0;
            for seed in 1..=512u64 {
                let block = random_bytes(layout.bytes, seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
                let interpreted: Result<Vec<f32>, DecodeError> = (0..layout.values)
                    .map(|index| descriptor.decode_block_value(&block, index))
                    .collect();
                let mut planned = vec![0.0; layout.values];
                let planned = decode_blocks(format, &block, &mut planned).map(|()| planned);
                match (&interpreted, &planned) {
                    (Ok(want), Ok(got)) => {
                        let want: Vec<u32> = want.iter().map(|value| value.to_bits()).collect();
                        let got: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
                        assert_eq!(got, want, "{format:?} block seed {seed}");
                    }
                    (Err(want), Err(got)) => {
                        assert_eq!(got, want, "{format:?} block seed {seed}");
                        refused += 1;
                    }
                    _ => panic!(
                        "{format:?} seed {seed}: plan {planned:?} vs interpreter {interpreted:?}"
                    ),
                }
            }
            assert!(refused < 512, "{format:?}: every random block was refused");
        }
    }

    /// Mutant H2 (mutants-m4.md): `check_blocks -> Ok(())` removes the only length guard, so
    /// `chunks_exact` drops a trailing partial block and an `out` longer than the bytes stays zero.
    /// A row that is not whole blocks, bytes short or bytes long, is a typed refusal.
    #[test]
    fn decode_blocks_refuses_a_row_length_that_is_not_whole_blocks() {
        let format = WeightFormat::Q4_0;
        for (bytes, values) in [(17, 32), (18, 64), (36, 32)] {
            let mut out = vec![0.0f32; values];
            assert_eq!(
                format
                    .descriptor()
                    .decode_blocks(&vec![0u8; bytes], &mut out),
                Err(DecodeError::RowLength {
                    format,
                    bytes,
                    values,
                }),
                "{bytes} bytes into {values} values"
            );
        }
    }
}
