//! One descriptor for every weight storage format poot reads (ADR-0103 decision 1).
//!
//! # Design
//!
//! A [`WeightFormat`] names a format; [`WeightFormat::descriptor`] returns its
//! [`FormatDescriptor`], which states the whole storage layout as data. The one scalar decoder
//! ([`crate::decode`]) interprets that data. Kernel lowering reads the same data, so the layout of
//! a format is written once, here, and nowhere else.
//!
//! ## Value formula
//!
//! Every format decodes one logical value with the same formula over named operand roles
//! ([`OperandRole`]):
//!
//! ```text
//! w = (Scale * SubScale) * value(Codes, Zero) (+|-) (Min * SubMin)
//! ```
//!
//! A role a format does not have is absent from the formula (not multiplied by one): a dense F32
//! weight is `w = value(Codes)`. The products are taken left to right in f32, which is the order
//! `ggml-quants.c` writes them (`d1 = d * sc; y = d1 * q - m1`), so the decoder reproduces the
//! reference rounding. `value` is one of three [`ValueMap`]s:
//!
//! - [`ValueMap::Integer`]: the code as an integer, minus the `Zero` operand when the format has
//!   one (`q - 8` for `Q4_0`, `q - (z + 1)` for GPTQ). Offsets such as `- 8` are part of the
//!   field's [`FieldEncoding`].
//! - [`ValueMap::Codebook`]: a 16-entry table (`IQ4_NL`, `IQ4_XS`).
//! - [`ValueMap::Float`]: the code is a float (F32, F16, BF16, E4M3FN, E2M1).
//!
//! ## Two storage shapes
//!
//! - [`Storage::Blocks`] (GGUF): one byte buffer of self-contained blocks along K. A block holds
//!   [`BlockLayout::values`] consecutive K values in [`BlockLayout::bytes`] bytes; every operand is a
//!   [`Field`] inside the block.
//! - [`Storage::Planar`] (safetensors): one source tensor per operand role ([`PlanarOperand`]).
//!
//! ## Fields inside a block: bit layouts
//!
//! A [`Field`] is read for one in-block element index `i` (`0..values`). It is the concatenation,
//! low bits first, of one or more [`Piece`]s. Each piece reads `width` bits at the bit offset its
//! [`BitLayout`] gives for `i`. A bit layout splits `i` into mixed-radix digits (most significant
//! first) and sums `digit * stride` onto a base bit offset. Bits are numbered little-endian through
//! the block: bit `b` is bit `b % 8` of byte `b / 8`, so a little-endian `u16` or `u32` field is a
//! plain run of bits.
//!
//! A digit with stride 0 is a broadcast: the field does not change along that digit. That is how a
//! per-sub-block operand is indexed by the element index: for `Q4_K`, `i = jh*128 + jl*32 + e`
//! (`jh` 0..2, `jl` 0..4, `e` 0..32), and every scale digit has stride 0 over `e`.
//!
//! ## Super-blocks with packed 6-bit sub-scales and mins
//!
//! `Q4_K` and `Q5_K` pack eight 6-bit scales and eight 6-bit mins into 12 bytes
//! (`get_scale_min_k4` in `ggml-quants.c`). Written as two pieces each, every piece is one affine
//! bit layout over the sub-block index `j = jh*4 + jl`:
//!
//! ```text
//! SubScale low 4 bits  : bit  0 + jh*64 + jl*8    (byte jl for j < 4, low nibble of byte jl+8 for j >= 4)
//! SubScale high 2 bits : bit  4 + jh*2  + jl*8    (bits 4..6 of byte jl for j < 4, bits 6..8 for j >= 4)
//! SubMin   low 4 bits  : bit 32 + jh*36 + jl*8    (byte jl+4 for j < 4, high nibble of byte jl+8 for j >= 4)
//! SubMin   high 2 bits : bit 36 + jh*2  + jl*8
//! ```
//!
//! (offsets relative to the start of `scales[12]`). `Q3_K`'s sixteen 6-bit scales (`kmask1`/`kmask2`
//! shuffle) and `IQ4_XS`'s `scales_l`/`scales_h` split are two affine pieces the same way, with a
//! `-32` offset in their [`FieldEncoding`]. No format is a special case in the decoder.
//!
//! ## Operand roles per format
//!
//! | Format | Scale | SubScale | Codes (value) | Zero | Min | SubMin |
//! |---|---|---|---|---|---|---|
//! | F32, F16, BF16 | - | - | float | - | - | - |
//! | `Q4_0` | `d` f16 | - | u4 - 8 | - | - | - |
//! | `Q4_1` | `d` f16 | - | u4 | - | `+m` f16 | - |
//! | `Q5_0` | `d` f16 | - | u5 - 16 (`qs` + `qh`) | - | - | - |
//! | `Q5_1` | `d` f16 | - | u5 (`qs` + `qh`) | - | `+m` f16 | - |
//! | `Q8_0` | `d` f16 | - | i8 | - | - | - |
//! | `Q2_K` | `d` f16 | u4 | u2 | - | `-dmin` f16 | u4 |
//! | `Q3_K` | `d` f16 | u6 - 32 | u3 - 4 (`qs` + `hmask`) | - | - | - |
//! | `Q4_K` | `d` f16 | u6 | u4 | - | `-dmin` f16 | u6 |
//! | `Q5_K` | `d` f16 | u6 | u5 (`qs` + `qh`) | - | `-dmin` f16 | u6 |
//! | `Q6_K` | `d` f16 | i8 | u6 - 32 (`ql` + `qh`) | - | - | - |
//! | `IQ4_NL` | `d` f16 | - | codebook | - | - | - |
//! | `IQ4_XS` | `d` f16 | u6 - 32 | codebook | - | - | - |
//! | MXFP4 (GGUF) | `e` E8M0 | - | E2M1 | - | - | - |
//! | E4M3 per-channel | f32/bf16/E8M0 per row | - | E4M3FN | - | - | - |
//! | E4M3 128x128 block | f32/bf16/E8M0 per block | - | E4M3FN | - | - | - |
//! | E2M1 row-32 | E8M0 per 32 along K | - | E2M1 | - | - | - |
//! | GPTQ int4 | f16 per group | - | u4 | u4 + 1 per group | - | - |
//! | AWQ int4 | f16 per group | - | u4 | u4 per group | - | - |
//!
//! `Q3_K`'s `hmask` bit is stored inverted (`q = low2 - (h ? 0 : 4)`); as a code it is
//! `low2 | h << 2` with a `-4` offset, which is the same value for all eight codes.
//!
//! ## GPTQ and AWQ
//!
//! Both are [`Storage::Planar`]: `qweight`, `qzeros` and `scales` are three source tensors with the
//! roles `Codes`, `Zero` and `Scale`. A [`PlanarOperand`] states the element grid (how many logical
//! values share one stored element along `[out, K]`), the sub-byte [`Packing`] (values per word, the
//! packed axis and the lane order, [`LaneOrder::Awq`] for AWQ's `[0, 4, 1, 5, 2, 6, 3, 7]`) and the
//! storage [`Major`] (GPTQ and AWQ store K-major, `[K/8, out]` and `[K, out/8]`). The group of a K
//! index is [`Extent::Group`]: `k / group_size`, or, for GPTQ with act-order, the `GroupIndex`
//! operand (`g_idx[k]`), selected by [`GroupMap`].
//!
//! ## Whole-tensor decoding
//!
//! The interpreter in [`crate::decode`] reads one element at a time and is the reference. Loaders
//! decode whole tensors through [`crate::plan`], which is const-evaluated from the same
//! descriptor: every bit layout is evaluated at compile time and each format gets its own
//! instantiation of one generic block decoder. A test pins the plan to the interpreter bit for
//! bit.
//!
//! ## MXFP4 sign of zero
//!
//! E2M1 code `0b1000` decodes to `-0.0`, for GGUF MXFP4 and safetensors E2M1 alike. E2M1 is a
//! sign-magnitude float and the OCP Microscaling v1.0 spec defines that code as negative zero;
//! `-0.0` is the exact conversion, which is what device FP4 conversion produces and what the
//! generated MXFP4 decode already computes (`mag * (1 - 2 * sign)`). ggml's `kvalues_mxfp4` is an
//! `int8_t` table, which cannot hold `-0`, so its `+0` is an artifact of the carrier, not a choice
//! of the format. The two are equal as values (`-0.0 == 0.0`); only the bit pattern differs. The
//! E2M1 values themselves are taken undoubled with the full E8M0 scale `2^(e - 127)`; ggml's doubled
//! table with its halved scale gives the same product bit for bit, including the `e < 2` subnormal
//! scales. E8M0 `0xff` is NaN and is a decode error, never a value.

use std::num::NonZeroUsize;

/// Every weight storage format poot reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[expect(
    non_camel_case_types,
    reason = "the GGUF formats keep ggml's type names (Q4_0, Q4_K, ...)"
)]
pub enum WeightFormat {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Iq4_Nl,
    Iq4_Xs,
    /// GGUF MXFP4 (ggml type 39): 32-value blocks, element `j` in the low nibble of byte `j` and
    /// element `j + 16` in its high nibble.
    Mxfp4,
    /// Safetensors `F8_E4M3` with one scale per output row.
    E4m3PerChannel {
        scale: ScaleEncoding,
    },
    /// Safetensors `F8_E4M3` with one scale per 128x128 `[out, K]` block.
    E4m3Block128 {
        scale: ScaleEncoding,
    },
    /// Safetensors E2M1 with one E8M0 scale per 32 K values; adjacent K values share a byte (even
    /// K in the low nibble).
    E2m1Row32,
    /// GPTQ int4 (AutoGPTQ): `w = scale * (q - (z + 1))`.
    Gptq {
        groups: GroupMap,
    },
    /// AWQ int4 (AutoAWQ GEMM): `w = scale * (q - z)`.
    Awq {
        group_size: NonZeroUsize,
    },
}

/// The encoding of a planar scale tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleEncoding {
    F32,
    Bf16,
    E8m0,
}

/// How a K index selects its quantization group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupMap {
    /// Group `k / size`.
    Contiguous { size: NonZeroUsize },
    /// Group `g_idx[k]`, read from the [`OperandRole::GroupIndex`] operand (GPTQ act-order), with
    /// this many groups.
    Indexed { groups: NonZeroUsize },
}

/// A named operand of the value formula (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// The quantized value codes.
    Codes,
    /// The block or group scale (`d`).
    Scale,
    /// The sub-block scale of a super-block (`sc`).
    SubScale,
    /// The per-group zero point subtracted from an integer code.
    Zero,
    /// The block or super-block min (`m`, `dmin`).
    Min,
    /// The sub-block min of a super-block (`m`).
    SubMin,
    /// The group index of each K value (GPTQ `g_idx`).
    GroupIndex,
}

/// Whether the min term is added (`Q4_1`, `Q5_1`: `x = d*q + m`) or subtracted (K-quants:
/// `x = d*sc*q - dmin*m`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MinSign {
    Add,
    Subtract,
}

/// How the `Codes` operand becomes the value the scale multiplies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueMap {
    /// The code read through its [`FieldEncoding`] as an integer, minus the `Zero` operand when the
    /// format has one.
    Integer,
    /// `table[code]`.
    Codebook(&'static [f32; 16]),
    /// The code is a float; its format is the `Codes` field's [`FieldEncoding::Float`].
    Float,
}

/// The narrow float formats a code or scale can be stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFormat {
    F32,
    F16,
    Bf16,
    /// OCP E4M3FN: no infinities, `0x7f`/`0xff` are NaN.
    E4m3Fn,
    /// OCP E2M1 (FP4): sign-magnitude, code `0b1000` is `-0.0`.
    E2m1,
    /// OCP E8M0: exponent only, `2^(e - 127)`, `0xff` is NaN.
    E8m0,
}

impl FloatFormat {
    /// The stored width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::Bf16 => 16,
            Self::E4m3Fn | Self::E8m0 => 8,
            Self::E2m1 => 4,
        }
    }
}

/// How a field's raw bits become a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldEncoding {
    /// An unsigned integer plus a constant offset (`Q4_0` codes: `offset = -8`).
    Unsigned { offset: i32 },
    /// A two's-complement integer of the field's width.
    Signed,
    /// A float of this format.
    Float(FloatFormat),
}

/// One digit of a [`BitLayout`]: its extent and the bit stride it contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Digit {
    pub extent: u32,
    pub stride: u32,
}

/// A mixed-radix affine map from an in-block element index to a bit offset.
///
/// The index is split into [`Digit`]s, most significant first; the product of the extents is the
/// block's value count. `bit(i) = base + sum(digit_n(i) * stride_n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitLayout {
    pub base: u32,
    pub digits: &'static [Digit],
}

impl BitLayout {
    /// The number of indices the layout covers (the product of its digit extents).
    pub const fn extent(self) -> u32 {
        let mut extent = 1;
        let mut index = 0;
        while index < self.digits.len() {
            extent *= self.digits[index].extent;
            index += 1;
        }
        extent
    }

    /// The bit offset of index `i`. `i` must be below [`Self::extent`].
    pub const fn bit(self, i: u32) -> u32 {
        let mut remaining = i;
        let mut bit = self.base;
        let mut index = self.digits.len();
        while index > 0 {
            index -= 1;
            let digit = self.digits[index];
            bit += (remaining % digit.extent) * digit.stride;
            remaining /= digit.extent;
        }
        bit
    }
}

/// `width` bits at the offset a [`BitLayout`] gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub layout: BitLayout,
    pub width: u32,
}

/// A number read from a block: its pieces concatenated low bits first, then decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub pieces: &'static [Piece],
    pub encoding: FieldEncoding,
}

impl Field {
    /// The total width in bits.
    pub const fn width(self) -> u32 {
        let mut width = 0;
        let mut index = 0;
        while index < self.pieces.len() {
            width += self.pieces[index].width;
            index += 1;
        }
        width
    }
}

/// One operand of a block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockField {
    pub role: OperandRole,
    /// The field's name in ggml's block struct (`d`, `scales`, `qs`), for messages and docs.
    pub name: &'static str,
    pub field: Field,
}

/// A GGUF block format: `values` consecutive K values stored in `bytes` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockLayout {
    pub values: usize,
    pub bytes: usize,
    pub fields: &'static [BlockField],
}

impl BlockLayout {
    /// The field with this role, if the format has one.
    pub fn field(self, role: OperandRole) -> Option<&'static BlockField> {
        self.fields.iter().find(|field| field.role == role)
    }

    /// [`Self::field`], usable from `const fn` context (discriminants compared as `u8` rather than
    /// through `PartialEq`, matching `plan::ConstPlan`'s const-time field lookup).
    pub(crate) const fn const_field(self, role: OperandRole) -> Option<&'static BlockField> {
        let mut index = 0;
        while index < self.fields.len() {
            if self.fields[index].role as u8 == role as u8 {
                return Some(&self.fields[index]);
            }
            index += 1;
        }
        None
    }
}

/// How many logical values share one stored element along one axis of `[out, K]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    /// A fixed count.
    Values(usize),
    /// The whole axis (one element per row, for a per-channel scale).
    Whole,
    /// One element per quantization group along K, as the format's [`GroupMap`] selects.
    Group,
}

/// The logical axis of `[out, K]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Out,
    K,
}

/// The order of the stored element grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Major {
    /// Rows follow `out`: a `[out, K]` grid.
    OutMajor,
    /// Rows follow `K`: a `[K, out]` grid (GPTQ, AWQ).
    KMajor,
}

/// Where value `j` of a packed word sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneOrder {
    /// Value `j` at lane `j` (lowest bits first).
    Natural,
    /// AutoAWQ's order: value `j` at lane `[0, 4, 1, 5, 2, 6, 3, 7][j]`.
    Awq,
}

impl LaneOrder {
    /// The lane of value `j` (`j` below the word's value count).
    pub const fn lane(self, j: usize) -> usize {
        match self {
            Self::Natural => j,
            Self::Awq => [0, 4, 1, 5, 2, 6, 3, 7][j],
        }
    }
}

/// Sub-byte values packed into little-endian words along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Packing {
    pub values_per_word: usize,
    pub word_bytes: usize,
    pub axis: Axis,
    pub lanes: LaneOrder,
}

/// One source tensor of a planar format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanarOperand {
    pub role: OperandRole,
    pub encoding: FieldEncoding,
    /// Stored width of one value in bits.
    pub bits: u32,
    /// Logical values per stored element along `[out, K]`.
    pub grid: [Extent; 2],
    pub packing: Option<Packing>,
    pub major: Major,
}

impl PlanarOperand {
    /// Bytes of one stored element: a packed word when the operand packs sub-byte values,
    /// otherwise one value.
    pub const fn element_bytes(self) -> usize {
        match self.packing {
            Some(packing) => packing.word_bytes,
            None => self.bits as usize / 8,
        }
    }
}

/// A safetensors format: one source tensor per operand role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanarLayout {
    pub operands: &'static [PlanarOperand],
}

/// Where a format's operands are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    Blocks(BlockLayout),
    Planar(PlanarLayout),
}

/// The complete storage and value description of one [`WeightFormat`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormatDescriptor {
    pub format: WeightFormat,
    pub storage: Storage,
    pub value: ValueMap,
    /// Whether the min term is added or subtracted; `None` when the format has no `Min` operand.
    pub min_sign: Option<MinSign>,
}

impl FormatDescriptor {
    /// Whether this format has a `Scale` operand. `false` only for the dense float formats
    /// (F32, F16, BF16): derived from the layout, never a restated list of formats.
    pub fn has_scale(&self) -> bool {
        match self.storage {
            Storage::Blocks(layout) => layout.field(OperandRole::Scale).is_some(),
            Storage::Planar(layout) => layout
                .operands
                .iter()
                .any(|operand| operand.role == OperandRole::Scale),
        }
    }
}

impl WeightFormat {
    /// This format's descriptor.
    pub const fn descriptor(self) -> FormatDescriptor {
        use crate::{blocks, planar};
        match self {
            Self::F32 => blocks::F32,
            Self::F16 => blocks::F16,
            Self::Bf16 => blocks::BF16,
            Self::Q4_0 => blocks::Q4_0,
            Self::Q4_1 => blocks::Q4_1,
            Self::Q5_0 => blocks::Q5_0,
            Self::Q5_1 => blocks::Q5_1,
            Self::Q8_0 => blocks::Q8_0,
            Self::Q2_K => blocks::Q2_K,
            Self::Q3_K => blocks::Q3_K,
            Self::Q4_K => blocks::Q4_K,
            Self::Q5_K => blocks::Q5_K,
            Self::Q6_K => blocks::Q6_K,
            Self::Iq4_Nl => blocks::IQ4_NL,
            Self::Iq4_Xs => blocks::IQ4_XS,
            Self::Mxfp4 => blocks::MXFP4,
            Self::E4m3PerChannel { scale } => planar::e4m3_per_channel(scale),
            Self::E4m3Block128 { scale } => planar::e4m3_block128(scale),
            Self::E2m1Row32 => planar::E2M1_ROW32,
            Self::Gptq { groups } => planar::gptq(groups),
            Self::Awq { group_size } => planar::awq(group_size),
        }
    }
}
