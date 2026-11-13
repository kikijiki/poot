//! Every GGUF block format through the format list and the descriptor-driven fixture (POOT-1514): the fixture
//! reaches every value of every field, and a format's decode of it equals llama.cpp's own dequantizer bit for
//! bit. A new format is covered by adding its descriptor and its reference file, with no test code of its own.

use std::collections::BTreeSet;

use poot_quant::format::{FieldEncoding, OperandRole, Storage, WeightFormat};
use poot_quant::test_support::formats::GGUF_FORMATS;
use poot_quant::test_support::packed::every_field_blocks;

/// The quantized GGUF formats: every listed block format with a `Scale` operand (the dense dtypes have none).
fn quantized_formats() -> impl Iterator<Item = WeightFormat> {
    GGUF_FORMATS.into_iter().filter(|format| {
        matches!(format.descriptor().storage, Storage::Blocks(layout)
            if layout.field(OperandRole::Scale).is_some())
    })
}

/// The reference file of `format` in `tests/data/ggml_reference/`, written by `tests/data/ggml_reference.cpp`
/// from [`every_field_blocks`]: named after the format (`Iq4_Nl` is `iq4_nl.tsv`).
fn reference_of(format: WeightFormat) -> String {
    let name = format!("{format:?}").to_lowercase();
    let path = poot_test_util::manifest_dir()
        .join("tests/data/ggml_reference")
        .join(format!("{name}.tsv"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{format:?} has no llama.cpp reference file {}: {error}; generate it from every_field_blocks",
            path.display()
        )
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// The raw bits of a field at in-block element `index`, read here bit by bit (not through the writer).
fn read_field(block: &[u8], field: &poot_quant::format::Field, index: u32) -> u32 {
    let mut value = 0u32;
    let mut at = 0;
    for piece in field.pieces {
        let start = piece.layout.bit(index) as usize;
        for bit in 0..piece.width as usize {
            let (byte, shift) = ((start + bit) / 8, (start + bit) % 8);
            value |= u32::from((block[byte] >> shift) & 1) << (at + bit);
        }
        at += piece.width as usize;
    }
    value
}

/// In every quantized GGUF format's fixture, every integer field (codes, sub-scales, signs) holds every value
/// its width can, and an 8-bit-or-narrower float code holds every finite pattern; a wider float factor holds
/// more than one value. Mutation: in `every_field_blocks`, build one block whatever the fields need; red
/// with `Q4_0 d: one value`.
#[test]
fn the_fixture_reaches_every_value_of_every_field() {
    for format in GGUF_FORMATS {
        let Storage::Blocks(layout) = format.descriptor().storage else {
            panic!("{format:?} is listed as a GGUF block format");
        };
        if layout.field(OperandRole::Scale).is_none() {
            continue;
        }
        let bytes = every_field_blocks(format);
        assert_eq!(bytes.len() % layout.bytes, 0, "{format:?}");
        for field in layout.fields {
            let seen: BTreeSet<u32> = bytes
                .chunks(layout.bytes)
                .flat_map(|block| {
                    (0..layout.values as u32).map(|index| read_field(block, &field.field, index))
                })
                .collect();
            let width = field.field.width();
            match field.field.encoding {
                FieldEncoding::Float(float) if width > 8 || field.role != OperandRole::Codes => {
                    assert!(seen.len() > 1, "{format:?} {}: one value", field.name);
                    assert!(
                        seen.iter()
                            .all(|&bits| !float.non_finite().is_some_and(|nf| nf.matches(bits))),
                        "{format:?} {}: a non-finite factor",
                        field.name
                    );
                }
                FieldEncoding::Float(float) => {
                    let finite = (0..1u32 << width)
                        .filter(|&bits| !float.non_finite().is_some_and(|nf| nf.matches(bits)))
                        .count();
                    assert_eq!(seen.len(), finite, "{format:?} {}", field.name);
                }
                _ => assert_eq!(seen.len(), 1 << width, "{format:?} {}", field.name),
            }
        }
    }
}

/// SC-2 of POOT-1539: every quantized format's decode of its fixture equals llama.cpp's `to_float`
/// (`dequantize_row_*`, b9190) bit for bit. Each reference file's first row is the fixture's blocks and
/// llama.cpp's output; the blocks must be exactly [`every_field_blocks`], so the reference covers every value of
/// every field. Further rows are extra evidence (real checkpoint blocks) decoded the same way. A row may carry
/// a third column, the positions where poot decodes `-0.0` and ggml's int8 table gives `+0.0` (MXFP4's E2M1
/// code `0b1000`, see `poot_quant::format`); the decode differs at exactly those positions and only there.
/// Mutation: flip one byte of a grid or codebook (`KVALUES_IQ4NL`, an `iq*_grid` entry) and the format's row
/// goes red naming the value; or change the MXFP4 negative-zero positions in `mxfp4.tsv`.
#[test]
fn every_quantized_format_decodes_its_fixture_like_llamacpp() {
    let mut formats = 0;
    for format in quantized_formats() {
        let file = reference_of(format);
        let mut rows = 0;
        for (index, line) in file
            .lines()
            .filter(|line| !line.starts_with('#'))
            .enumerate()
        {
            let mut columns = line.split('\t');
            let (blocks, expected) = (
                unhex(columns.next().unwrap()),
                unhex(columns.next().unwrap()),
            );
            let negative_zeros: Vec<usize> = columns
                .next()
                .map(|positions| positions.split(',').map(|at| at.parse().unwrap()).collect())
                .unwrap_or_default();
            if index == 0 {
                let fixture = every_field_blocks(format);
                assert!(
                    blocks == fixture,
                    "{format:?}: the reference was not generated from the current fixture; regenerate it \
                     from these blocks: {}",
                    hex(&fixture)
                );
            }
            let expected: Vec<f32> = expected
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            let mut decoded = vec![0.0f32; expected.len()];
            poot_quant::plan::decode_blocks(format, &blocks, &mut decoded)
                .unwrap_or_else(|error| panic!("{format:?}: {error}"));
            let differing: Vec<usize> = (0..decoded.len())
                .filter(|&i| decoded[i].to_bits() != expected[i].to_bits())
                .collect();
            if let Some(&at) = differing.iter().find(|at| !negative_zeros.contains(at)) {
                panic!(
                    "{format:?}: value {at} decodes to {} ({:#010x}), llama.cpp says {} ({:#010x})",
                    decoded[at],
                    decoded[at].to_bits(),
                    expected[at],
                    expected[at].to_bits()
                );
            }
            for &at in &negative_zeros {
                assert!(
                    decoded[at].to_bits() == 0x8000_0000 && expected[at].to_bits() == 0,
                    "{format:?}: value {at} is listed as poot's -0.0 against ggml's +0.0 but decodes to {} and ggml says {}",
                    decoded[at],
                    expected[at]
                );
            }
            assert_eq!(
                differing, negative_zeros,
                "{format:?}: the values that differ are exactly the listed negative zeros"
            );
            rows += 1;
        }
        assert!(
            rows >= 1,
            "{format:?}: a reference file holds the fixture row"
        );
        formats += 1;
    }
    assert_eq!(formats, 19, "the nineteen quantized GGUF formats");
}
