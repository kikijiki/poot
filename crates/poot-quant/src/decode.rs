//! The one scalar decoder: it interprets a [`FormatDescriptor`] and holds no per-format code.
//!
//! Block and planar storage differ only in where an operand's bits are; both feed the same value
//! formula ([`Formula::evaluate`], see [`crate::format`]).

use std::fmt;

use crate::format::{
    Axis, BlockLayout, Extent, Field, FieldEncoding, FormatDescriptor, GroupMap, Major, MinSign,
    OperandRole, PlanarLayout, PlanarOperand, Storage, ValueMap, WeightFormat,
};
use crate::scalar::float_to_f32;

/// Typed failures of the scalar decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The operation needs the other storage shape (blocks versus planar tensors).
    WrongStorage { format: WeightFormat },
    /// A block slice has the wrong length.
    BlockLength {
        format: WeightFormat,
        expected: usize,
        actual: usize,
    },
    /// An in-block element index is outside the block.
    ElementIndex {
        format: WeightFormat,
        index: usize,
        values: usize,
    },
    /// Source bytes do not hold the requested values in whole blocks.
    RowLength {
        format: WeightFormat,
        bytes: usize,
        values: usize,
    },
    /// A logical coordinate is outside the `[out, K]` shape.
    Coordinate {
        format: WeightFormat,
        coordinate: [usize; 2],
        shape: [usize; 2],
    },
    /// A row decode's output slice does not hold exactly one logical row of `K` values.
    RowOutput {
        format: WeightFormat,
        k: usize,
        len: usize,
    },
    /// A planar operand the format needs was not supplied.
    MissingOperand {
        format: WeightFormat,
        role: OperandRole,
    },
    /// A planar operand's source has the wrong byte length for the shape.
    OperandLength {
        format: WeightFormat,
        role: OperandRole,
        expected: usize,
        actual: usize,
    },
    /// A GPTQ `g_idx` entry names no group.
    GroupIndex {
        format: WeightFormat,
        k: usize,
        group: i64,
    },
    /// Deriving a planar operand's extent overflowed `usize`.
    Overflow { format: WeightFormat },
    /// A stored operand is NaN (E8M0 `0xff`, E4M3FN `0x7f`, a NaN float).
    NotANumber {
        format: WeightFormat,
        role: OperandRole,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongStorage { format } => {
                write!(f, "{format:?} does not have this storage shape")
            }
            Self::BlockLength {
                format,
                expected,
                actual,
            } => write!(
                f,
                "{format:?} block is {expected} bytes, got a slice of {actual}"
            ),
            Self::ElementIndex {
                format,
                index,
                values,
            } => write!(
                f,
                "{format:?} element index {index} is outside a block of {values} values"
            ),
            Self::RowLength {
                format,
                bytes,
                values,
            } => write!(
                f,
                "{format:?}: {bytes} source bytes do not hold {values} values in whole blocks"
            ),
            Self::Coordinate {
                format,
                coordinate,
                shape,
            } => write!(
                f,
                "{format:?} coordinate {coordinate:?} is outside shape {shape:?}"
            ),
            Self::RowOutput { format, k, len } => write!(
                f,
                "{format:?}: a row decode needs an output of K = {k} values, got {len}"
            ),
            Self::MissingOperand { format, role } => {
                write!(f, "{format:?} needs a {role:?} operand")
            }
            Self::OperandLength {
                format,
                role,
                expected,
                actual,
            } => write!(
                f,
                "{format:?} {role:?} operand is {expected} bytes for this shape, got {actual}"
            ),
            Self::GroupIndex { format, k, group } => {
                write!(f, "{format:?} g_idx[{k}] = {group} names no group")
            }
            Self::Overflow { format } => {
                write!(f, "{format:?} operand extent overflows usize")
            }
            Self::NotANumber { format, role } => {
                write!(f, "{format:?} stores a NaN {role:?} operand")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Raw operand bits with the encoding that reads them.
#[derive(Clone, Copy)]
struct Raw {
    bits: u32,
    width: u32,
    encoding: FieldEncoding,
}

impl Raw {
    fn integer(self) -> i64 {
        match self.encoding {
            FieldEncoding::Unsigned { offset } => i64::from(self.bits) + i64::from(offset),
            FieldEncoding::Signed => {
                let unused = 32 - self.width;
                i64::from(((self.bits << unused) as i32) >> unused)
            }
            FieldEncoding::Float(_) => unreachable!("descriptor tests: integer operands are ints"),
        }
    }

    fn number(self) -> f32 {
        match self.encoding {
            FieldEncoding::Float(format) => float_to_f32(format, self.bits),
            FieldEncoding::Unsigned { .. } | FieldEncoding::Signed => self.integer() as f32,
        }
    }
}

/// The value formula of [`crate::format`], evaluated over an operand reader.
struct Formula<'d> {
    descriptor: &'d FormatDescriptor,
}

impl Formula<'_> {
    /// Read the scale and min products and the raw code.
    fn parts(
        &self,
        operand: &mut impl FnMut(OperandRole) -> Result<Option<Raw>, DecodeError>,
    ) -> Result<(Option<f32>, Option<f32>, Raw), DecodeError> {
        let format = self.descriptor.format;
        let mut number = |role| -> Result<Option<f32>, DecodeError> {
            let Some(raw) = operand(role)? else {
                return Ok(None);
            };
            let value = raw.number();
            if value.is_nan() {
                return Err(DecodeError::NotANumber { format, role });
            }
            Ok(Some(value))
        };
        let scale = product(number(OperandRole::Scale)?, number(OperandRole::SubScale)?);
        let min = product(number(OperandRole::Min)?, number(OperandRole::SubMin)?);
        let code = operand(OperandRole::Codes)?.ok_or(DecodeError::MissingOperand {
            format,
            role: OperandRole::Codes,
        })?;
        Ok((scale, min, code))
    }

    fn evaluate(
        &self,
        mut operand: impl FnMut(OperandRole) -> Result<Option<Raw>, DecodeError>,
    ) -> Result<f32, DecodeError> {
        let (scale, min, code) = self.parts(&mut operand)?;
        let value = match self.descriptor.value {
            ValueMap::Integer => {
                let zero = operand(OperandRole::Zero)?.map_or(0, Raw::integer);
                (code.integer() - zero) as f32
            }
            ValueMap::Codebook(table) => table[code.bits as usize],
            ValueMap::Float => {
                let value = code.number();
                if value.is_nan() {
                    return Err(DecodeError::NotANumber {
                        format: self.descriptor.format,
                        role: OperandRole::Codes,
                    });
                }
                value
            }
        };
        let scaled = scale.map_or(value, |scale| scale * value);
        Ok(match (min, self.descriptor.min_sign) {
            (Some(min), Some(MinSign::Add)) => scaled + min,
            (Some(min), Some(MinSign::Subtract)) => scaled - min,
            _ => scaled,
        })
    }
}

/// `lhs * rhs` in that order when both are present.
fn product(lhs: Option<f32>, rhs: Option<f32>) -> Option<f32> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs * rhs),
        (lhs, None) => lhs,
        (None, rhs) => rhs,
    }
}

/// `width` (at most 32) bits at bit offset `bit` of a little-endian byte stream.
fn read_bits(bytes: &[u8], bit: usize, width: u32) -> u32 {
    let first = bit / 8;
    let last = (bit + width as usize).div_ceil(8);
    let window = bytes[first..last]
        .iter()
        .rev()
        .fold(0u64, |window, &byte| (window << 8) | u64::from(byte));
    ((window >> (bit % 8)) & ((1u64 << width) - 1)) as u32
}

/// A block field's raw bits for element `index`: its pieces concatenated low bits first.
fn read_field(field: &Field, block: &[u8], index: u32) -> Raw {
    let mut bits = 0u32;
    let mut width = 0;
    for piece in field.pieces {
        let value = read_bits(block, piece.layout.bit(index) as usize, piece.width);
        bits |= value << width;
        width += piece.width;
    }
    Raw {
        bits,
        width,
        encoding: field.encoding,
    }
}

impl FormatDescriptor {
    fn blocks(&self) -> Result<BlockLayout, DecodeError> {
        match self.storage {
            Storage::Blocks(layout) => Ok(layout),
            Storage::Planar(_) => Err(DecodeError::WrongStorage {
                format: self.format,
            }),
        }
    }

    fn planar(&self) -> Result<PlanarLayout, DecodeError> {
        match self.storage {
            Storage::Planar(layout) => Ok(layout),
            Storage::Blocks(_) => Err(DecodeError::WrongStorage {
                format: self.format,
            }),
        }
    }

    /// The first `Float`-encoded block field whose value at some element does not decode finite (its
    /// role and the flat K index of the failing element), or `None` if every one does. `bytes` is a
    /// run of whole blocks (a row or a whole tensor); content validation does not care which row an
    /// element is in, only whether some stored value is non-finite.
    ///
    /// Walks every element of every `Float`-encoded field directly (not through the compiled
    /// [`crate::plan::ConstPlan`], which only groups the *scale* factors): a codes field encoded as
    /// `Float` (MXFP4's E2M1) varies every element, so it cannot be checked once per block.
    pub(crate) fn first_non_finite_block_field(
        &self,
        bytes: &[u8],
    ) -> Option<(OperandRole, usize)> {
        let layout = self.blocks().ok()?;
        for (block_index, block) in bytes.chunks_exact(layout.bytes).enumerate() {
            for field in layout.fields {
                let FieldEncoding::Float(_) = field.field.encoding else {
                    continue;
                };
                for i in 0..layout.values {
                    if read_field(&field.field, block, i as u32)
                        .number()
                        .is_finite()
                    {
                        continue;
                    }
                    return Some((field.role, block_index * layout.values + i));
                }
            }
        }
        None
    }

    /// The first stored element of a `Float`-encoded planar operand that does not decode finite (its
    /// flat index in the operand's own stored grid, lane-major), or `None` if the operand is not
    /// `Float`-encoded or every stored value decodes finite. Walks the operand's own stored elements
    /// once each (never a logical `[out, K]` coordinate, which a `Group`/`Whole` extent or packing can
    /// map many-to-one onto the same stored bits).
    pub(crate) fn first_non_finite_planar_operand(
        &self,
        operand: &PlanarOperand,
        shape: [usize; 2],
        bytes: &[u8],
    ) -> Result<Option<usize>, DecodeError> {
        let FieldEncoding::Float(float) = operand.encoding else {
            return Ok(None);
        };
        let [rows, columns] = self.stored_shape(operand, shape)?;
        let lanes = operand.packing.map_or(1, |packing| packing.values_per_word);
        let element_bits = operand.element_bytes() * 8;
        for row in 0..rows {
            for column in 0..columns {
                let base_bit = (row * columns + column) * element_bits;
                for lane in 0..lanes {
                    let bit = base_bit + lane * operand.bits as usize;
                    let value = float_to_f32(float, read_bits(bytes, bit, operand.bits));
                    if value.is_finite() {
                        continue;
                    }
                    return Ok(Some((row * columns + column) * lanes + lane));
                }
            }
        }
        Ok(None)
    }

    /// The first stored `GroupIndex` value (GPTQ act-order `g_idx`) that names no group - negative,
    /// or `>= groups` - as `(k, value)`, or `None` if every stored value is in range. `operand` must
    /// be the format's own `GroupIndex` operand (`OperandRole::GroupIndex`, act-order GPTQ's only
    /// one); `groups` is the caller's own `GroupMap::Indexed` count. Checked once at construction
    /// (`PackedPayload::try_new`), so no decoder - the CPU oracle's `group_of` or a device kernel's
    /// `emit_group_of` - ever reads an out-of-range group index (dquant.md 4, "kernels decode the
    /// admitted domain only": a device kernel has no such check of its own to fail closed with).
    pub(crate) fn first_invalid_group_index(
        &self,
        operand: &PlanarOperand,
        shape: [usize; 2],
        bytes: &[u8],
        groups: usize,
    ) -> Option<(usize, i64)> {
        let [rows, columns] = self.stored_shape(operand, shape).ok()?;
        for row in 0..rows {
            for column in 0..columns {
                let bit = (row * columns + column) * operand.element_bytes() * 8;
                let bits = read_bits(bytes, bit, operand.bits);
                let value = i64::from(bits as i32);
                if value < 0 || value >= groups as i64 {
                    return Some((row * columns + column, value));
                }
            }
        }
        None
    }

    /// Decode a run of whole blocks (a GGUF row or tensor) into `out`, through the format's
    /// compiled [plan](crate::plan).
    pub fn decode_blocks(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), DecodeError> {
        crate::plan::decode_blocks(self.format, bytes, out)
    }

    /// The `[rows, row_bytes]` grid of the `role` planar operand's source tensor for a logical
    /// `[out, K]` shape: `rows` stored rows, each `row_bytes` bytes. The one place a byte extent is
    /// derived from the stored element grid ([`Self::stored_shape`]) and the element width
    /// ([`PlanarOperand::element_bytes`]); every source-shape or source-byte fact a caller needs
    /// comes from here, never restated.
    pub fn source_shape(
        &self,
        role: OperandRole,
        shape: [usize; 2],
    ) -> Result<[usize; 2], DecodeError> {
        let operand = self.planar_operand(role)?;
        let [rows, columns] = self.stored_shape(&operand, shape)?;
        let row_bytes =
            columns
                .checked_mul(operand.element_bytes())
                .ok_or(DecodeError::Overflow {
                    format: self.format,
                })?;
        Ok([rows, row_bytes])
    }

    /// The byte length of the `role` source tensor for a logical `[out, K]` shape.
    pub fn planar_operand_bytes(
        &self,
        shape: [usize; 2],
        role: OperandRole,
    ) -> Result<usize, DecodeError> {
        let [rows, row_bytes] = self.source_shape(role, shape)?;
        rows.checked_mul(row_bytes).ok_or(DecodeError::Overflow {
            format: self.format,
        })
    }

    /// Decode logical coordinate `[o, k]` of a planar weight of `shape` from its source tensors.
    /// Every operand the format has must be in `sources` with its exact byte length.
    pub fn decode_planar_value(
        &self,
        shape: [usize; 2],
        sources: &[(OperandRole, &[u8])],
        coordinate: [usize; 2],
    ) -> Result<f32, DecodeError> {
        let layout = self.planar()?;
        if coordinate[0] >= shape[0] || coordinate[1] >= shape[1] {
            return Err(DecodeError::Coordinate {
                format: self.format,
                coordinate,
                shape,
            });
        }
        for operand in layout.operands {
            let expected = self.planar_operand_bytes(shape, operand.role)?;
            let actual = self.source(sources, operand.role)?.len();
            if actual != expected {
                return Err(DecodeError::OperandLength {
                    format: self.format,
                    role: operand.role,
                    expected,
                    actual,
                });
            }
        }
        Formula { descriptor: self }.evaluate(|role| {
            layout
                .operands
                .iter()
                .find(|operand| operand.role == role)
                .map(|operand| self.read_planar(operand, shape, sources, coordinate))
                .transpose()
        })
    }

    /// Decode logical row `row` of a planar weight into `out` (`K` values), bit for bit what
    /// [`Self::decode_planar_value`] gives per element, with the source checks done once per row
    /// instead of once per value.
    pub fn decode_planar_row(
        &self,
        shape: [usize; 2],
        sources: &[(OperandRole, &[u8])],
        row: usize,
        out: &mut [f32],
    ) -> Result<(), DecodeError> {
        let layout = self.planar()?;
        if row >= shape[0] {
            return Err(DecodeError::Coordinate {
                format: self.format,
                coordinate: [row, 0],
                shape,
            });
        }
        for operand in layout.operands {
            let expected = self.planar_operand_bytes(shape, operand.role)?;
            let actual = self.source(sources, operand.role)?.len();
            if actual != expected {
                return Err(DecodeError::OperandLength {
                    format: self.format,
                    role: operand.role,
                    expected,
                    actual,
                });
            }
        }
        for (column, value) in out.iter_mut().enumerate().take(shape[1]) {
            *value = Formula { descriptor: self }.evaluate(|role| {
                layout
                    .operands
                    .iter()
                    .find(|operand| operand.role == role)
                    .map(|operand| self.read_planar(operand, shape, sources, [row, column]))
                    .transpose()
            })?;
        }
        Ok(())
    }

    /// The planar operand with this role.
    pub fn planar_operand(&self, role: OperandRole) -> Result<PlanarOperand, DecodeError> {
        self.planar()?
            .operands
            .iter()
            .copied()
            .find(|operand| operand.role == role)
            .ok_or(DecodeError::MissingOperand {
                format: self.format,
                role,
            })
    }

    fn source<'s>(
        &self,
        sources: &[(OperandRole, &'s [u8])],
        role: OperandRole,
    ) -> Result<&'s [u8], DecodeError> {
        sources
            .iter()
            .find(|(source, _)| *source == role)
            .map(|(_, bytes)| *bytes)
            .ok_or(DecodeError::MissingOperand {
                format: self.format,
                role,
            })
    }

    /// The number of quantization groups along K.
    fn groups(&self, k: usize) -> usize {
        match self.format {
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size },
            }
            | WeightFormat::Awq { group_size: size } => k.div_ceil(size.get()),
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups },
            } => groups.get(),
            _ => unreachable!("descriptor tests: only GPTQ and AWQ have Extent::Group"),
        }
    }

    /// The `[rows, columns]` of the stored (packed) grid, in storage order.
    fn stored_shape(
        &self,
        operand: &PlanarOperand,
        shape: [usize; 2],
    ) -> Result<[usize; 2], DecodeError> {
        let mut grid = [0; 2];
        for axis in 0..2 {
            grid[axis] = match operand.grid[axis] {
                Extent::Values(values) => shape[axis].div_ceil(values),
                Extent::Whole => 1,
                Extent::Group => self.groups(shape[1]),
            };
        }
        if let Some(packing) = operand.packing {
            let axis = axis_index(packing.axis);
            grid[axis] = grid[axis].div_ceil(packing.values_per_word);
        }
        Ok(match operand.major {
            Major::OutMajor => grid,
            Major::KMajor => [grid[1], grid[0]],
        })
    }

    fn read_planar(
        &self,
        operand: &PlanarOperand,
        shape: [usize; 2],
        sources: &[(OperandRole, &[u8])],
        [o, k]: [usize; 2],
    ) -> Result<Raw, DecodeError> {
        let mut element = [0; 2];
        for (axis, coordinate) in [(0, o), (1, k)] {
            element[axis] = match operand.grid[axis] {
                Extent::Values(values) => coordinate / values,
                Extent::Whole => 0,
                Extent::Group => self.group_of(shape, sources, k)?,
            };
        }
        let mut lane = 0;
        if let Some(packing) = operand.packing {
            let axis = axis_index(packing.axis);
            let value = element[axis] % packing.values_per_word;
            element[axis] /= packing.values_per_word;
            lane = packing.lanes.lane(value);
        }
        let [_, columns] = self.stored_shape(operand, shape)?;
        let [row, column] = match operand.major {
            Major::OutMajor => element,
            Major::KMajor => [element[1], element[0]],
        };
        let bit =
            (row * columns + column) * operand.element_bytes() * 8 + lane * operand.bits as usize;
        Ok(Raw {
            bits: read_bits(self.source(sources, operand.role)?, bit, operand.bits),
            width: operand.bits,
            encoding: operand.encoding,
        })
    }

    /// The quantization group of K index `k`.
    fn group_of(
        &self,
        shape: [usize; 2],
        sources: &[(OperandRole, &[u8])],
        k: usize,
    ) -> Result<usize, DecodeError> {
        match self.format {
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups },
            } => {
                let operand = self.planar_operand(OperandRole::GroupIndex)?;
                let group = self
                    .read_planar(&operand, shape, sources, [0, k])?
                    .integer();
                usize::try_from(group)
                    .ok()
                    .filter(|&group| group < groups.get())
                    .ok_or(DecodeError::GroupIndex {
                        format: self.format,
                        k,
                        group,
                    })
            }
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size },
            }
            | WeightFormat::Awq { group_size: size } => Ok(k / size.get()),
            _ => unreachable!("descriptor tests: only GPTQ and AWQ have Extent::Group"),
        }
    }
}

const fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::Out => 0,
        Axis::K => 1,
    }
}

// Card 638: a bare `#[cfg(test)] impl` outside a test tree is a
// pattern the gate rejects; a genuine test oracle belongs inside a `mod tests`. Inherent impls
// resolve by crate regardless of which module declares them, so plan.rs, blocks.rs and tests.rs
// keep calling `descriptor.decode_block_value(...)` with no call-site change.
#[cfg(test)]
mod tests {
    use super::*;

    impl FormatDescriptor {
        /// Decode element `index` of one block: the reference interpreter the compiled
        /// [plan](crate::plan) is tested against.
        pub(crate) fn decode_block_value(
            &self,
            block: &[u8],
            index: usize,
        ) -> Result<f32, DecodeError> {
            let layout = self.blocks()?;
            if block.len() != layout.bytes {
                return Err(DecodeError::BlockLength {
                    format: self.format,
                    expected: layout.bytes,
                    actual: block.len(),
                });
            }
            if index >= layout.values {
                return Err(DecodeError::ElementIndex {
                    format: self.format,
                    index,
                    values: layout.values,
                });
            }
            Formula { descriptor: self }.evaluate(|role| {
                Ok(layout
                    .field(role)
                    .map(|field| read_field(&field.field, block, index as u32)))
            })
        }
    }
}
