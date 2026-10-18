//! The packed-weight content check, compiled from a [`FormatDescriptor`].
//!
//! ADR-0103 decision 4: every stored value of a `Float`-encoded field must decode finite. Which
//! patterns are non-finite is [`FloatFormat::non_finite`]; this module only locates the stored bits
//! of each field once per format, so a scan reads each *stored* float once (a per-block scale once
//! per block, not once per value it scales) and never interprets the descriptor per element.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::decode::DecodeError;
use crate::format::{
    BlockLayout, FieldEncoding, FloatFormat, FormatDescriptor, NonFinite, OperandRole,
    PlanarOperand, Storage, WeightFormat,
};

/// Counts the stored floats a scan reads. Production scans pass `()`, which compiles away; the
/// work-bound test passes a counter.
pub(crate) trait ReadTally {
    fn float_read(&mut self);
}

impl ReadTally for () {
    #[inline(always)]
    fn float_read(&mut self) {}
}

/// Little-endian unsigned integer of `width` (1, 2 or 4) bytes at `at`.
#[inline(always)]
fn load(bytes: &[u8], at: usize, width: usize) -> u32 {
    match width {
        1 => u32::from(bytes[at]),
        2 => u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])),
        _ => u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]),
    }
}

/// One stored float of a block, at a fixed byte offset; `element` is the first in-block element
/// whose value it is.
#[derive(Clone, Copy, Debug)]
struct Read {
    byte: usize,
    element: usize,
}

#[derive(Clone, Debug)]
struct CheckedField {
    role: OperandRole,
    width: usize,
    non_finite: NonFinite,
    /// The distinct stored floats of this field in one block, by first element.
    reads: Vec<Read>,
}

/// The content check of a block format: per field with a non-finite pattern, the distinct stored
/// floats of one block.
#[derive(Clone, Debug)]
pub(crate) struct BlockContentCheck {
    block_bytes: usize,
    values: usize,
    fields: Vec<CheckedField>,
}

/// The content check of one planar operand: it is `Some` when the operand's elements can be
/// non-finite.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OperandContentCheck {
    width: usize,
    non_finite: NonFinite,
}

/// The content check of one [`WeightFormat`], compiled once per format for the process.
#[derive(Debug)]
pub(crate) enum ContentCheck {
    Blocks(BlockContentCheck),
    /// One entry per operand of the layout, in layout order.
    Planar(Vec<Option<OperandContentCheck>>),
}

impl ContentCheck {
    fn compile(descriptor: &FormatDescriptor) -> Result<Self, DecodeError> {
        Ok(match descriptor.storage {
            Storage::Blocks(_) => Self::Blocks(descriptor.block_content_check()?),
            Storage::Planar(layout) => Self::Planar(
                layout
                    .operands
                    .iter()
                    .map(|operand| descriptor.operand_content_check(operand))
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// The compiled check of `format`: built on first use, shared afterwards.
    pub(crate) fn of(format: WeightFormat) -> Result<Arc<Self>, DecodeError> {
        static COMPILED: OnceLock<Mutex<HashMap<WeightFormat, Arc<ContentCheck>>>> =
            OnceLock::new();
        let mut compiled = COMPILED
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(check) = compiled.get(&format) {
            return Ok(Arc::clone(check));
        }
        let check = Arc::new(Self::compile(&format.descriptor())?);
        compiled.insert(format, Arc::clone(&check));
        Ok(check)
    }
}

fn unsupported(descriptor: &FormatDescriptor, role: OperandRole) -> DecodeError {
    DecodeError::UnsupportedFloatField {
        format: descriptor.format,
        role,
    }
}

impl FormatDescriptor {
    /// Compile the content check of a block format. A field whose float format has no non-finite
    /// pattern (MXFP4's E2M1 codes) is not scanned at all.
    pub(crate) fn block_content_check(&self) -> Result<BlockContentCheck, DecodeError> {
        let Storage::Blocks(BlockLayout {
            values,
            bytes,
            fields,
        }) = self.storage
        else {
            return Err(DecodeError::WrongStorage {
                format: self.format,
            });
        };
        let mut checked = Vec::new();
        for block_field in fields {
            let FieldEncoding::Float(float) = block_field.field.encoding else {
                continue;
            };
            let Some(non_finite) = float.non_finite() else {
                continue;
            };
            let width = float_width(float);
            let [piece] = block_field.field.pieces else {
                return Err(unsupported(self, block_field.role));
            };
            if piece.width != float.bits() {
                return Err(unsupported(self, block_field.role));
            }
            let mut reads: Vec<Read> = Vec::new();
            for element in 0..values {
                let bit = piece.layout.bit(element as u32) as usize;
                if !bit.is_multiple_of(8) {
                    return Err(unsupported(self, block_field.role));
                }
                let byte = bit / 8;
                if byte + width > bytes {
                    return Err(unsupported(self, block_field.role));
                }
                if !reads.iter().any(|read| read.byte == byte) {
                    reads.push(Read { byte, element });
                }
            }
            checked.push(CheckedField {
                role: block_field.role,
                width,
                non_finite,
                reads,
            });
        }
        Ok(BlockContentCheck {
            block_bytes: bytes,
            values,
            fields: checked,
        })
    }

    /// Compile the content check of a planar operand, or `None` when no stored element can be
    /// non-finite (an integer operand, E2M1 codes).
    pub(crate) fn operand_content_check(
        &self,
        operand: &PlanarOperand,
    ) -> Result<Option<OperandContentCheck>, DecodeError> {
        let FieldEncoding::Float(float) = operand.encoding else {
            return Ok(None);
        };
        let Some(non_finite) = float.non_finite() else {
            return Ok(None);
        };
        if operand.packing.is_some() || operand.bits != float.bits() {
            return Err(unsupported(self, operand.role));
        }
        Ok(Some(OperandContentCheck {
            width: float_width(float),
            non_finite,
        }))
    }
}

fn float_width(float: FloatFormat) -> usize {
    float.bits() as usize / 8
}

impl BlockContentCheck {
    /// The first non-finite stored float of `bytes`, a run of whole blocks: its field's role and
    /// the flat K index of the first element it is the value of. Fields are checked in descriptor
    /// order within a block, so the answer is the first failing element of the first failing
    /// field.
    pub(crate) fn first_non_finite(
        &self,
        bytes: &[u8],
        tally: &mut impl ReadTally,
    ) -> Option<(OperandRole, usize)> {
        for (block_index, block) in bytes.chunks_exact(self.block_bytes).enumerate() {
            for field in &self.fields {
                for read in &field.reads {
                    tally.float_read();
                    if field
                        .non_finite
                        .matches(load(block, read.byte, field.width))
                    {
                        return Some((field.role, block_index * self.values + read.element));
                    }
                }
            }
        }
        None
    }
}

impl OperandContentCheck {
    /// The index of the first non-finite stored element of `bytes` (the operand's own stored
    /// grid, one element per `width` bytes).
    pub(crate) fn first_non_finite(
        &self,
        bytes: &[u8],
        tally: &mut impl ReadTally,
    ) -> Option<usize> {
        bytes.chunks_exact(self.width).position(|element| {
            tally.float_read();
            self.non_finite.matches(load(element, 0, self.width))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::StoredBytes;
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    use super::*;
    use crate::format::{GroupMap, ScaleEncoding, WeightFormat};
    use crate::scalar::float_to_f32;
    use crate::{PackedPayload, PackedWeight, PackedWeightError, SourceRole};

    /// Whether the stored `bits` decode finite under the format's non-finite pattern.
    fn is_finite_bits(float: FloatFormat, bits: u32) -> bool {
        float
            .non_finite()
            .is_none_or(|non_finite| !non_finite.matches(bits))
    }

    const FLOATS: [FloatFormat; 6] = [
        FloatFormat::F32,
        FloatFormat::F16,
        FloatFormat::Bf16,
        FloatFormat::E4m3Fn,
        FloatFormat::E2m1,
        FloatFormat::E8m0,
    ];

    const BLOCK_FORMATS: [WeightFormat; 19] = [
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
        WeightFormat::Iq2_Xs,
        WeightFormat::Iq3_Xxs,
        WeightFormat::Iq3_S,
        WeightFormat::Mxfp4,
    ];

    fn planar_formats() -> Vec<WeightFormat> {
        let size = NonZeroUsize::new(32).unwrap();
        let mut formats = vec![WeightFormat::E2m1Row32];
        for scale in [ScaleEncoding::F32, ScaleEncoding::Bf16, ScaleEncoding::E8m0] {
            formats.push(WeightFormat::E4m3PerChannel { scale });
            formats.push(WeightFormat::E4m3Block128 { scale });
        }
        formats.push(WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size },
        });
        formats.push(WeightFormat::Gptq {
            groups: GroupMap::Indexed {
                groups: NonZeroUsize::new(1).unwrap(),
            },
        });
        formats.push(WeightFormat::Awq { group_size: size });
        formats
    }

    /// The stored patterns of `float` that do not decode finite, found by decoding every pattern
    /// (or, for f32, from the IEEE special values): the independent oracle for the compiled masks.
    fn non_finite_patterns(float: FloatFormat) -> Vec<u32> {
        if float == FloatFormat::F32 {
            return vec![
                f32::INFINITY.to_bits(),
                f32::NEG_INFINITY.to_bits(),
                f32::NAN.to_bits(),
                0xffff_ffff,
            ];
        }
        let all: Vec<u32> = (0..1u32 << float.bits())
            .filter(|&bits| !float_to_f32(float, bits).is_finite())
            .collect();
        match (all.first(), all.last()) {
            (Some(&first), Some(&last)) => vec![first, last],
            _ => Vec::new(),
        }
    }

    /// SC-001: for every `Float` format of at most 16 bits the compiled finiteness predicate equals
    /// `float_to_f32(bits).is_finite()` for every stored bit pattern; f32 over every sign and
    /// exponent with the mantissa edge patterns.
    ///
    /// Mutation (observed): mask `0x7c00` -> `0x7800` for `F16` in `FloatFormat::non_finite` fails
    /// with `F16 0x7c00` (`left: true, right: false`).
    #[test]
    fn the_finiteness_predicate_equals_the_scalar_decoder_for_every_pattern() {
        for float in FLOATS {
            if float == FloatFormat::F32 {
                for sign_exponent in 0..0x200u32 {
                    for mantissa in [0, 1, 0x40_0000, 0x7f_ffff] {
                        let bits = sign_exponent << 23 | mantissa;
                        assert_eq!(
                            is_finite_bits(float, bits),
                            float_to_f32(float, bits).is_finite(),
                            "F32 {bits:#010x}"
                        );
                    }
                }
                continue;
            }
            for bits in 0..1u32 << float.bits() {
                assert_eq!(
                    is_finite_bits(float, bits),
                    float_to_f32(float, bits).is_finite(),
                    "{float:?} {bits:#06x}"
                );
            }
        }
        assert_eq!(FloatFormat::E2m1.non_finite(), None);
    }

    /// `pattern` (little-endian, `width` bytes) written at byte `at`.
    fn plant(bytes: &mut [u8], at: usize, width: usize, pattern: u32) {
        bytes[at..at + width].copy_from_slice(&pattern.to_le_bytes()[..width]);
    }

    /// The first in-block element whose field bits start at `bit`, found by walking the layout.
    fn first_element_at(layout: crate::format::BitLayout, values: usize, bit: u32) -> usize {
        (0..values)
            .find(|&i| layout.bit(i as u32) == bit)
            .expect("the bit belongs to some element")
    }

    /// SC-002 (blocks): every `Float` field with a non-finite pattern of every block format, at the
    /// first, a middle and the last block, is refused naming role and the first element it scales.
    ///
    /// Mutation (observed): `.take(blocks - 1)` on the block iterator of `first_non_finite` fails
    /// with `F32 value block 4 0x7f800000` (`left: None, right: Some((Codes, 4))`).
    #[test]
    fn every_block_format_refuses_a_non_finite_field_at_first_middle_and_last_block() {
        const BLOCKS: usize = 5;
        for format in BLOCK_FORMATS {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                unreachable!()
            };
            let weight = PackedWeight::try_new(format, [1, layout.values * BLOCKS]);
            // A dense format has no scale, so `PackedWeight` refuses it; its check is exercised
            // through the descriptor directly.
            let mut cases = 0;
            for field in layout.fields {
                let FieldEncoding::Float(float) = field.field.encoding else {
                    continue;
                };
                for pattern in non_finite_patterns(float) {
                    let width = float.bits() as usize / 8;
                    let piece = field.field.pieces[0];
                    // Plant at every distinct stored position of the field, one block at a time.
                    let positions: BTreeSet<u32> = (0..layout.values as u32)
                        .map(|i| piece.layout.bit(i))
                        .collect();
                    for block in [0, BLOCKS / 2, BLOCKS - 1] {
                        for &bit in &positions {
                            let mut bytes = vec![0u8; layout.bytes * BLOCKS];
                            plant(
                                &mut bytes,
                                block * layout.bytes + bit as usize / 8,
                                width,
                                pattern,
                            );
                            let expected = (
                                field.role,
                                block * layout.values
                                    + first_element_at(piece.layout, layout.values, bit),
                            );
                            let check = descriptor.block_content_check().unwrap();
                            assert_eq!(
                                check.first_non_finite(&bytes, &mut ()),
                                Some(expected),
                                "{format:?} {} block {block} {pattern:#x}",
                                field.name
                            );
                            if let Ok(weight) = weight {
                                let error = PackedPayload::try_new(
                                    weight,
                                    [(SourceRole::Blocks, StoredBytes::from(bytes))],
                                )
                                .unwrap_err();
                                assert_eq!(
                                    error,
                                    PackedWeightError::NonFiniteField {
                                        format,
                                        role: SourceRole::Blocks,
                                        operand: expected.0,
                                        element: expected.1,
                                    }
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
            let has_pattern = layout.fields.iter().any(|field| {
                matches!(field.field.encoding, FieldEncoding::Float(float) if float.non_finite().is_some())
            });
            assert_eq!(cases > 0, has_pattern, "{format:?}");
        }
    }

    /// SC-002 (planar): the same for every safetensors format: a non-finite stored element of each
    /// `Float` operand with a pattern, at the first, a middle and the last element.
    #[test]
    fn every_planar_format_refuses_a_non_finite_operand_element_at_first_middle_and_last() {
        for format in planar_formats() {
            let weight = PackedWeight::try_new(format, [4, 256]).unwrap();
            let descriptor = format.descriptor();
            let Storage::Planar(layout) = descriptor.storage else {
                unreachable!()
            };
            let mut planted = 0;
            for operand in layout.operands {
                let FieldEncoding::Float(float) = operand.encoding else {
                    continue;
                };
                let role = SourceRole::Planar(operand.role);
                let len = weight.source_bytes(role);
                let width = operand.element_bytes();
                let elements = len / width;
                for pattern in non_finite_patterns(float) {
                    for element in [0, elements / 2, elements - 1] {
                        let sources: Vec<_> = weight
                            .sources()
                            .into_iter()
                            .map(|source| {
                                let mut bytes = vec![0u8; weight.source_bytes(source)];
                                if source == role {
                                    plant(&mut bytes, element * width, width, pattern);
                                }
                                (source, StoredBytes::from(bytes))
                            })
                            .collect();
                        assert_eq!(
                            PackedPayload::try_new(weight, sources).unwrap_err(),
                            PackedWeightError::NonFiniteField {
                                format,
                                role,
                                operand: operand.role,
                                element,
                            },
                            "{format:?} {:?} element {element} {pattern:#x}",
                            operand.role
                        );
                        planted += 1;
                    }
                }
            }
            assert_eq!(
                planted > 0,
                layout.operands.iter().any(|operand| {
                    matches!(operand.encoding, FieldEncoding::Float(float) if float.non_finite().is_some())
                }),
                "{format:?}"
            );
        }
    }

    #[derive(Default)]
    struct Counter(u64);

    impl ReadTally for Counter {
        fn float_read(&mut self) {
            self.0 += 1;
        }
    }

    /// SC-003: a scan reads each stored float of a block once, not once per value it scales: Q4_K's
    /// `d` and `dmin` are two reads per block (not 2 x 256), and every other block format reads
    /// exactly its distinct stored float positions per block.
    ///
    /// Mutation (observed): pushing a `Read` for every element instead of every distinct byte in
    /// `block_content_check` (the per-value walk) fails with `Q4_0` (`left: 32000, right: 1000`).
    #[test]
    fn a_block_scan_reads_each_stored_float_once_per_block() {
        const BLOCKS: usize = 1000;
        for format in BLOCK_FORMATS {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                unreachable!()
            };
            let per_block: usize = layout
                .fields
                .iter()
                .filter_map(|field| match field.field.encoding {
                    FieldEncoding::Float(float) if float.non_finite().is_some() => Some(
                        (0..layout.values as u32)
                            .map(|i| field.field.pieces[0].layout.bit(i))
                            .collect::<BTreeSet<_>>()
                            .len(),
                    ),
                    _ => None,
                })
                .sum();
            let mut counter = Counter::default();
            let bytes = vec![0u8; layout.bytes * BLOCKS];
            let check = descriptor.block_content_check().unwrap();
            assert_eq!(check.first_non_finite(&bytes, &mut counter), None);
            assert_eq!(counter.0 as usize, per_block * BLOCKS, "{format:?}");
        }
        // Per-block scales are one read each, whatever the block's value count.
        for (format, reads) in [
            (WeightFormat::Q4_K, 2),
            (WeightFormat::Q2_K, 2),
            (WeightFormat::Q6_K, 1),
            (WeightFormat::Q8_0, 1),
            (WeightFormat::Mxfp4, 1),
        ] {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                unreachable!()
            };
            let mut counter = Counter::default();
            let bytes = vec![0u8; layout.bytes * BLOCKS];
            descriptor
                .block_content_check()
                .unwrap()
                .first_non_finite(&bytes, &mut counter);
            assert_eq!(counter.0 as usize, reads * BLOCKS, "{format:?}");
        }
    }

    /// A planar operand scan reads each stored element once; an integer operand and E2M1 codes
    /// (no non-finite pattern) are not scanned at all.
    #[test]
    fn a_planar_scan_reads_each_stored_float_once_and_skips_operands_without_a_pattern() {
        let format = WeightFormat::E2m1Row32;
        let descriptor = format.descriptor();
        let Storage::Planar(layout) = descriptor.storage else {
            unreachable!()
        };
        let weight = PackedWeight::try_new(format, [4, 256]).unwrap();
        for operand in layout.operands {
            let check = descriptor.operand_content_check(operand).unwrap();
            match operand.role {
                OperandRole::Codes => assert!(check.is_none()),
                role => {
                    let len = weight.source_bytes(SourceRole::Planar(role));
                    let mut counter = Counter::default();
                    assert_eq!(
                        check
                            .unwrap()
                            .first_non_finite(&vec![0u8; len], &mut counter),
                        None
                    );
                    assert_eq!(counter.0 as usize, len / operand.element_bytes());
                }
            }
        }
    }

    /// Every registered `Float` field with a non-finite pattern is one the compiled check can
    /// locate (no descriptor reaches `UnsupportedFloatField`).
    #[test]
    fn every_registered_format_compiles_a_content_check() {
        for format in BLOCK_FORMATS {
            format.descriptor().block_content_check().unwrap();
        }
        for format in planar_formats() {
            let Storage::Planar(layout) = format.descriptor().storage else {
                unreachable!()
            };
            for operand in layout.operands {
                format.descriptor().operand_content_check(operand).unwrap();
            }
        }
    }
}
