//! Packed-linear payloads decode through the one scalar decoder.

use std::sync::Arc;

use super::*;
use crate::format::ScaleEncoding;

fn owner(bytes: &[u8]) -> Arc<[u8]> {
    Arc::from(bytes)
}

fn payload(
    format: WeightFormat,
    shape: [usize; 2],
    weight_bytes: Arc<[u8]>,
    scale_bytes: Arc<[u8]>,
) -> PackedPayload {
    let weight = PackedWeight::try_new(format, shape).unwrap();
    PackedPayload::try_new(
        weight,
        [
            (SourceRole::Planar(OperandRole::Codes), weight_bytes),
            (SourceRole::Planar(OperandRole::Scale), scale_bytes),
        ],
    )
    .unwrap()
}

/// Logical row `row` of `payload`, through [`PackedPayload::decode_row`].
fn decoded_row(payload: &PackedPayload, row: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; payload.weight().shape()[1]];
    payload.decode_row(row, &mut out).unwrap();
    out
}

fn assert_decodes(payload: &PackedPayload, expected: &[([usize; 2], f32)]) {
    for &(coordinate, want) in expected {
        let got = decoded_row(payload, coordinate[0])[coordinate[1]];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{coordinate:?}: {got:?} vs {want:?}"
        );
    }
}

/// A `[129, 130]` E4M3 weight with literal bytes at the 128x128 block corners and ragged edges
/// (every other byte zero), and one scale per block: 1, 2, 4, 8.
fn block_fp8_payload(format: WeightFormat, scales: &[u8]) -> PackedPayload {
    let mut weights = vec![0x00; 129 * 130];
    for (coordinate, byte) in [
        ([0, 0], 0x38),
        ([0, 127], 0x40),
        ([0, 128], 0x42),
        ([127, 129], 0xb8),
        ([128, 0], 0x01),
        ([128, 127], 0x30),
        ([128, 128], 0x3c),
        ([128, 129], 0x7e),
    ] {
        weights[coordinate[0] * 130 + coordinate[1]] = byte;
    }
    payload(format, [129, 130], Arc::from(weights), owner(scales))
}

#[test]
fn block_fp8_payload_decodes_block_corners_and_ragged_edges() {
    let expected = [
        ([0, 0], 1.0),
        ([0, 127], 2.0),
        ([0, 128], 5.0),
        ([127, 129], -2.0),
        ([128, 0], 1.0 / 128.0),
        ([128, 127], 2.0),
        ([128, 128], 12.0),
        ([128, 129], 3584.0),
    ];
    for (format, scales) in [
        (
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            &[0x80, 0x3f, 0x00, 0x40, 0x80, 0x40, 0x00, 0x41][..],
        ),
        (
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            &[
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00,
                0x00, 0x41,
            ][..],
        ),
        (
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            &[0x7f, 0x80, 0x81, 0x82][..],
        ),
    ] {
        assert_decodes(&block_fp8_payload(format, scales), &expected);
    }
}

#[test]
fn block_fp8_payload_reads_one_byte_per_k_in_order() {
    let payload = payload(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        [1, 5],
        owner(&[0x38, 0x40, 0x42, 0xb8, 0x01]),
        owner(&[0x80, 0x3f]),
    );
    assert_decodes(
        &payload,
        &[
            ([0, 0], 1.0),
            ([0, 1], 2.0),
            ([0, 2], 2.5),
            ([0, 3], -1.0),
            ([0, 4], 1.0 / 512.0),
        ],
    );
}

#[test]
fn adjacent_e2m1_payload_decodes_nibble_order_and_ragged_blocks() {
    let mut weights = vec![0; 36];
    weights[0] = 0x51;
    weights[15] = 0xac;
    weights[16] = 0x62;
    weights[17] = 0x03;
    weights[18] = 0xe4;
    weights[33] = 0x19;
    weights[34] = 0xd5;
    weights[35] = 0x04;
    let payload = payload(
        WeightFormat::E2m1Row32,
        [2, 35],
        Arc::from(weights),
        owner(&[0x7f, 0x80, 0x81, 0x7e]),
    );
    assert_decodes(
        &payload,
        &[
            ([0, 0], 0.5),
            ([0, 1], 3.0),
            ([0, 30], -2.0),
            ([0, 31], -1.0),
            ([0, 32], 2.0),
            ([0, 33], 8.0),
            ([0, 34], 3.0),
            ([1, 0], 8.0),
            ([1, 1], -16.0),
            ([1, 30], -2.0),
            ([1, 31], 2.0),
            ([1, 32], 1.5),
            ([1, 33], -1.5),
            ([1, 34], 1.0),
        ],
    );
}

/// The same byte `0x51` is K 0 and K 1 in the adjacent layout, but K 0 and K 16 in GGUF MXFP4's
/// half/half layout.
#[test]
fn adjacent_e2m1_is_not_gguf_mxfp4() {
    let mut weights = [0; 16];
    weights[0] = 0x51;
    let payload = payload(
        WeightFormat::E2m1Row32,
        [1, 32],
        owner(&weights),
        owner(&[0x7f]),
    );
    let mut mxfp4_block = [0; 17];
    mxfp4_block[0] = 0x7f;
    mxfp4_block[1] = 0x51;
    let mxfp4 = WeightFormat::Mxfp4.descriptor();
    assert_eq!(decoded_row(&payload, 0)[1], 3.0);
    assert_eq!(mxfp4.decode_block_value(&mxfp4_block, 1).unwrap(), 0.0);
    assert_eq!(mxfp4.decode_block_value(&mxfp4_block, 16).unwrap(), 3.0);
}

/// The byte offset of `format`'s `Scale` field, asserted byte-aligned (every block-32 scale is a
/// whole f16 or E8M0 byte, never a sub-byte piece).
fn scale_byte_offset(format: WeightFormat) -> usize {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("block-32 formats only")
    };
    let field = layout
        .field(OperandRole::Scale)
        .expect("every block-32 format has a Scale field");
    let bit = field.field.pieces[0].layout.bit(0);
    assert_eq!(
        bit % 8,
        0,
        "{format:?}: the Scale field must be byte-aligned"
    );
    (bit / 8) as usize
}

/// SC-003 (card 542a): `PackedPayload::try_new` refuses a non-finite `Float`-encoded field
/// (ADR-0101 decision 4) for every block-32 format, and accepts an all-zero block (zero
/// is a legitimate, not a refused, scale) of every one.
#[test]
fn block32_formats_refuse_non_finite_fields_and_accept_everything_else() {
    const BLOCK32: [WeightFormat; 7] = [
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q8_0,
        WeightFormat::Iq4_Nl,
        WeightFormat::Mxfp4,
    ];
    for format in BLOCK32 {
        let Storage::Blocks(layout) = format.descriptor().storage else {
            unreachable!()
        };
        let weight = PackedWeight::try_new(format, [1, layout.values]).unwrap();
        let zero_block = vec![0u8; layout.bytes];
        PackedPayload::try_new(weight, [(SourceRole::Blocks, Arc::from(zero_block))])
            .unwrap_or_else(|error| panic!("{format:?}: an all-zero block must be valid: {error}"));
    }

    // MXFP4: `e` = 0xff (E8M0 NaN).
    let mxfp4 = PackedWeight::try_new(WeightFormat::Mxfp4, [1, 32]).unwrap();
    let mut mxfp4_block = vec![0u8; mxfp4.source_bytes(SourceRole::Blocks)];
    mxfp4_block[scale_byte_offset(WeightFormat::Mxfp4)] = 0xff;
    assert_eq!(
        PackedPayload::try_new(mxfp4, [(SourceRole::Blocks, Arc::from(mxfp4_block))]).unwrap_err(),
        PackedWeightError::NonFiniteField {
            format: WeightFormat::Mxfp4,
            role: SourceRole::Blocks,
            operand: OperandRole::Scale,
            element: 0,
        }
    );

    // Q8_0: `d` = f16 +infinity (0x7c00, scalar::tests::f16_decodes_normals_subnormals_and_specials_exactly).
    let q8_0 = PackedWeight::try_new(WeightFormat::Q8_0, [1, 32]).unwrap();
    let mut q8_0_block = vec![0u8; q8_0.source_bytes(SourceRole::Blocks)];
    let d = scale_byte_offset(WeightFormat::Q8_0);
    q8_0_block[d] = 0x00;
    q8_0_block[d + 1] = 0x7c;
    assert_eq!(
        PackedPayload::try_new(q8_0, [(SourceRole::Blocks, Arc::from(q8_0_block))]).unwrap_err(),
        PackedWeightError::NonFiniteField {
            format: WeightFormat::Q8_0,
            role: SourceRole::Blocks,
            operand: OperandRole::Scale,
            element: 0,
        }
    );
}

/// card 542c: a stored act-order `g_idx` value that names no group - negative,
/// or `>= groups` - must be refused at `PackedPayload::try_new`, before any decoder (the CPU oracle
/// or a device kernel) ever reads it. `Codes`/`Zero`/`Scale` are all-zero (a legitimate payload for
/// every other field; only `GroupIndex` is under test here). Mutation: comment out the
/// `first_invalid_group_index` check in `validate_content` (`crates/poot-quant/src/lib.rs`); both
/// assertions below go red (`PackedPayload::try_new` builds instead of refusing). Reverted; see the
/// card's final report for the observed failure.
#[test]
fn gptq_act_order_refuses_negative_and_out_of_range_group_index() {
    let groups = 2usize;
    let format = WeightFormat::Gptq {
        groups: GroupMap::Indexed {
            groups: std::num::NonZeroUsize::new(groups).unwrap(),
        },
    };
    let weight = PackedWeight::try_new(format, [4, 8]).unwrap();

    let codes = vec![0u8; weight.source_bytes(SourceRole::Planar(OperandRole::Codes))];
    let zero = vec![0u8; weight.source_bytes(SourceRole::Planar(OperandRole::Zero))];
    let scale = vec![0u8; weight.source_bytes(SourceRole::Planar(OperandRole::Scale))];
    let valid_g_idx: Vec<u8> = (0..8u32)
        .flat_map(|k| ((k % groups as u32) as i32).to_le_bytes())
        .collect();

    let build = |g_idx: Vec<u8>| {
        PackedPayload::try_new(
            weight,
            [
                (
                    SourceRole::Planar(OperandRole::Codes),
                    Arc::from(codes.clone()),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    Arc::from(zero.clone()),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    Arc::from(scale.clone()),
                ),
                (
                    SourceRole::Planar(OperandRole::GroupIndex),
                    Arc::from(g_idx),
                ),
            ],
        )
    };

    // Sanity: a valid g_idx (every value in 0..groups) is accepted.
    build(valid_g_idx.clone())
        .unwrap_or_else(|error| panic!("valid g_idx must be accepted: {error}"));

    // k=3 -> -1 (negative).
    let mut negative_g_idx = valid_g_idx.clone();
    negative_g_idx[3 * 4..3 * 4 + 4].copy_from_slice(&(-1i32).to_le_bytes());
    assert_eq!(
        build(negative_g_idx).unwrap_err(),
        PackedWeightError::GroupIndexOutOfRange {
            format,
            k: 3,
            value: -1,
            groups,
        }
    );

    // k=5 -> groups (out of range: valid values are 0..groups).
    let mut oob_g_idx = valid_g_idx;
    oob_g_idx[5 * 4..5 * 4 + 4].copy_from_slice(&(groups as i32).to_le_bytes());
    assert_eq!(
        build(oob_g_idx).unwrap_err(),
        PackedWeightError::GroupIndexOutOfRange {
            format,
            k: 5,
            value: groups as i64,
            groups,
        }
    );
}

/// SC-002 (card 540a): [`StoredBytes`] is the one byte-ownership and content-identity type both a
/// dense tensor and a packed source build on. Equality is bytes only; the fingerprint memo is
/// observability, not identity.
#[test]
fn stored_bytes_owns_its_buffer_with_a_stable_content_based_fingerprint() {
    let a = StoredBytes::new(owner(&[1, 2, 3, 4]));
    let b = StoredBytes::new(owner(&[1, 2, 3, 4]));
    assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(a.len(), 4);
    assert!(!a.is_empty());
    assert_eq!(
        a, b,
        "equal content is equal regardless of fingerprint state"
    );

    let first = a.fingerprint();
    let second = a.fingerprint();
    assert_eq!(first, second, "fingerprint is stable across calls");
    assert_eq!(first, b.fingerprint(), "equal bytes fingerprint equal");
}

/// The source fingerprint is the FNV-1a 64 over each role's source bytes, independently per role, and
/// survives a clone (the memo is observability, not identity: [`PackedPayload`]'s `PartialEq` is
/// weight + bytes only).
#[test]
fn source_fingerprint_is_fnv1a_over_each_payload_source() {
    const CODES: SourceRole = SourceRole::Planar(OperandRole::Codes);
    const SCALE: SourceRole = SourceRole::Planar(OperandRole::Scale);

    let weight = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [129, 130],
    )
    .unwrap();
    let mut weight_source = vec![0; 16_770];
    weight_source[0] = 0x38;
    weight_source[16_769] = 0xb8;
    let scale_source = [
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00,
        0x41,
    ];
    let payload = PackedPayload::try_new(
        weight,
        [
            (CODES, Arc::from(weight_source)),
            (SCALE, owner(&scale_source)),
        ],
    )
    .unwrap();

    // The specification fold, written out here rather than called, so this row checks the value
    // against the algorithm instead of against the function under test.
    fn fold(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
        hash
    }

    let weight_fingerprint = payload.source_fingerprint(CODES);
    assert_eq!(weight_fingerprint.byte_len, 16_770);
    assert_eq!(
        weight_fingerprint.fnv1a64,
        fold(payload.bytes(CODES)),
        "weight fingerprint must be FNV-1a 64 over the weight bytes"
    );

    // A second request returns the same value.
    assert_eq!(payload.source_fingerprint(CODES), weight_fingerprint);

    let scale_fingerprint = payload.source_fingerprint(SCALE);
    assert_eq!(scale_fingerprint.byte_len, 16);
    assert_eq!(scale_fingerprint.fnv1a64, fold(payload.bytes(SCALE)));

    // A clone shares the bytes, so it fingerprints the same.
    let cloned = payload.clone();
    assert_eq!(cloned.source_fingerprint(CODES), weight_fingerprint);
    // Identity stays weight + bytes: the memo is observability, not identity.
    assert_eq!(cloned, payload);
}

/// Deterministic bytes for `rows` rows of whole `format` blocks, `row_values` values each, with a
/// distinct finite f16 super-scale `d` per row (and a finite `dmin` for Q4_K). ggml's `block_q8_0`
/// and `block_q4_K` both start with the f16 `d`; `block_q4_K`'s f16 `dmin` follows it. Every other
/// byte is xorshift noise: 8-bit and 4-bit codes and 6-bit sub-scales are finite at every value.
fn distinct_block_rows(format: WeightFormat, rows: usize, row_values: usize) -> Vec<u8> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("block formats only")
    };
    let blocks_per_row = row_values / layout.values;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut bytes: Vec<u8> = (0..rows * blocks_per_row * layout.bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    // f16 1.0, 2.0, -0.5, 4.0, ...: one super-scale per row.
    const D: [u16; 4] = [0x3c00, 0x4000, 0xb800, 0x4400];
    for (index, block) in bytes.chunks_exact_mut(layout.bytes).enumerate() {
        let row = index / blocks_per_row;
        block[0..2].copy_from_slice(&D[row % D.len()].to_le_bytes());
        if format == WeightFormat::Q4_K {
            // dmin = f16 0.25.
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes());
        }
    }
    bytes
}

/// Mutant H0 (mutants-m4.md), now at `PackedPayload::decode_row`'s block-branch row offset
/// (`crates/poot-quant/src/lib.rs:638`, `let start = row * row_bytes;`): made `/`, every row below
/// `row_bytes` decodes row 0. Each row of a multi-row block payload must decode as that row's own
/// bytes through the one block decoder. (The per-element `PackedPayload::decode`, where the design
/// doc found H0, had no production caller and was deleted.)
#[test]
fn block_payload_decode_reads_the_addressed_row() {
    for (format, shape) in [
        (WeightFormat::Q8_0, [3, 64]),
        (WeightFormat::Q4_K, [3, 256]),
    ] {
        let weight = PackedWeight::try_new(format, shape).unwrap();
        let bytes = distinct_block_rows(format, shape[0], shape[1]);
        let row_bytes = bytes.len() / shape[0];
        let payload =
            PackedPayload::try_new(weight, [(SourceRole::Blocks, Arc::from(bytes.clone()))])
                .unwrap();
        let mut rows = Vec::new();
        for (row, row_source) in bytes.chunks_exact(row_bytes).enumerate() {
            let mut want = vec![0.0f32; shape[1]];
            format
                .descriptor()
                .decode_blocks(row_source, &mut want)
                .unwrap();
            let got = decoded_row(&payload, row);
            for (column, (got, want)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{format:?} [{row}, {column}]: {got:?} vs {want:?}"
                );
            }
            rows.push(want);
        }
        assert!(
            rows[0] != rows[1] && rows[1] != rows[2] && rows[0] != rows[2],
            "{format:?}: the fixture rows must decode differently"
        );
    }
}

/// Mutant M2 (mutants-m4.md) was `PackedPayload::decode`'s coordinate bounds check; that method is
/// deleted, and `decode_row` (`crates/poot-quant/src/lib.rs:620` `row >= shape[0]`, `:627`
/// `out.len() != shape[1]`) owns the same guard. A row past the end is `Coordinate`, and an output
/// one value short or long is `RowOutput`, never a slice panic or a partial decode.
#[test]
fn decode_row_refuses_a_row_or_output_outside_the_shape() {
    let q8_0 = WeightFormat::Q8_0;
    let block = PackedPayload::try_new(
        PackedWeight::try_new(q8_0, [3, 64]).unwrap(),
        [(
            SourceRole::Blocks,
            Arc::from(distinct_block_rows(q8_0, 3, 64)),
        )],
    )
    .unwrap();
    let planar_format = WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::E8m0,
    };
    let planar = block_fp8_payload(planar_format, &[0x7f, 0x80, 0x81, 0x82]);
    for (payload, format, shape) in [
        (&block, q8_0, [3, 64]),
        (&planar, planar_format, [129, 130]),
    ] {
        let mut row = vec![0.0f32; shape[1]];
        assert_eq!(
            payload.decode_row(shape[0], &mut row),
            Err(DecodeError::Coordinate {
                format,
                coordinate: [shape[0], 0],
                shape,
            }),
            "{format:?} row {}",
            shape[0]
        );
        for len in [shape[1] - 1, shape[1] + 1] {
            let mut out = vec![0.0f32; len];
            assert_eq!(
                payload.decode_row(0, &mut out),
                Err(DecodeError::RowOutput {
                    format,
                    k: shape[1],
                    len,
                }),
                "{format:?} output of {len}"
            );
        }
    }
}
