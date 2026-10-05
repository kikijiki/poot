use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_quant::decode::DecodeError;
use poot_quant::format::{GroupMap, OperandRole, ScaleEncoding, Storage, WeightFormat};
use poot_quant::{LogicalAxis, PackedPayload, PackedWeight, PackedWeightError, SourceRole};

fn owner(bytes: &[u8]) -> Arc<[u8]> {
    Arc::from(bytes)
}

/// Logical value `[row, column]` of `payload`, read through [`PackedPayload::decode_row`].
fn decoded(payload: &PackedPayload, [row, column]: [usize; 2]) -> f32 {
    let mut out = vec![0.0f32; payload.weight().shape()[1]];
    payload.decode_row(row, &mut out).unwrap();
    out[column]
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

const CODES: SourceRole = SourceRole::Planar(OperandRole::Codes);
const SCALE: SourceRole = SourceRole::Planar(OperandRole::Scale);

/// SC-001: `PackedWeight::try_new` refuses each invalid case with a typed error.
///
/// Mutation row (run 2026-09-29): dropped the partial-block check in `PackedWeight::try_new`
/// (`crates/poot-quant/src/lib.rs`) -> `packed_weight_try_new_rejects_a_partial_block` failed at
/// `assert_eq!(PackedWeight::try_new(WeightFormat::Q4_0, [3, 33]), Err(...))` with `left:
/// Ok(PackedWeight { format: Q4_0, shape: [3, 33] })`. Restored.
#[test]
fn packed_weight_try_new_rejects_a_zero_extent() {
    assert_eq!(
        PackedWeight::try_new(WeightFormat::Q4_0, [0, 32]),
        Err(PackedWeightError::ZeroExtent {
            axis: LogicalAxis::Out
        })
    );
    assert_eq!(
        PackedWeight::try_new(WeightFormat::Q4_0, [1, 0]),
        Err(PackedWeightError::ZeroExtent {
            axis: LogicalAxis::K
        })
    );
}

#[test]
fn packed_weight_try_new_rejects_a_partial_block() {
    assert_eq!(
        PackedWeight::try_new(WeightFormat::Q4_0, [3, 33]),
        Err(PackedWeightError::PartialBlock {
            format: WeightFormat::Q4_0,
            k: 33,
            block_values: 32,
        })
    );
    // A whole number of blocks is accepted.
    assert!(PackedWeight::try_new(WeightFormat::Q4_0, [3, 64]).is_ok());
}

#[test]
fn packed_weight_try_new_rejects_a_dense_format() {
    for format in [WeightFormat::F32, WeightFormat::F16, WeightFormat::Bf16] {
        assert_eq!(
            PackedWeight::try_new(format, [4, 4]),
            Err(PackedWeightError::DenseFormat { format })
        );
    }
}

#[test]
fn packed_weight_try_new_accepts_every_named_scheme() {
    for format in [
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
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E2m1Row32,
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nonzero(8) },
        },
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nonzero(2) },
        },
        WeightFormat::Awq {
            group_size: nonzero(8),
        },
    ] {
        let block_values = match format.descriptor().storage {
            poot_quant::format::Storage::Blocks(layout) => layout.values,
            poot_quant::format::Storage::Planar(_) => 1,
        };
        let shape = [8, block_values.max(32)];
        assert!(
            PackedWeight::try_new(format, shape).is_ok(),
            "{format:?} {shape:?}"
        );
    }
}

/// SC-002: `source_shape`/`source_bytes` equal the descriptor's stored grid, for every scheme
/// [`format::WeightFormat`] names. Every expected `[rows, row_bytes]` pair below is written by hand
/// from the block byte counts in `blocks.rs`'s doc comments and the operand grids in `planar.rs`,
/// not derived from the function under test; several also cross-check against the source byte
/// lengths `planar.rs`'s own reference-vector tests already pin.
///
/// Mutation row (run 2026-09-29): `PackedWeight::try_source_shape`'s `SourceRole::Blocks` arm
/// returned `self.shape` (the logical `[out, K]` shape) instead of `[out, row_bytes]` ->
/// `packed_weight_source_shape_and_bytes_match_the_stored_grid_for_every_scheme` failed at `Q4_0`
/// with `left: [3, 64], right: [3, 36]`. Restored.
#[test]
fn packed_weight_source_shape_and_bytes_match_the_stored_grid_for_every_scheme() {
    // Block formats (GGUF): out = 3, K = two blocks. `values`/`bytes` are `blocks.rs`'s own per-row
    // doc comments (`Q4_0` `18 bytes per 32`, and so on).
    for (format, values, bytes) in [
        (WeightFormat::Q4_0, 32, 18),
        (WeightFormat::Q4_1, 32, 20),
        (WeightFormat::Q5_0, 32, 22),
        (WeightFormat::Q5_1, 32, 24),
        (WeightFormat::Q8_0, 32, 34),
        (WeightFormat::Q2_K, 256, 84),
        (WeightFormat::Q3_K, 256, 110),
        (WeightFormat::Q4_K, 256, 144),
        (WeightFormat::Q5_K, 256, 176),
        (WeightFormat::Q6_K, 256, 210),
        (WeightFormat::Iq4_Nl, 32, 18),
        (WeightFormat::Iq4_Xs, 256, 136),
        (WeightFormat::Mxfp4, 32, 17),
    ] {
        let shape = [3, 2 * values];
        let weight = PackedWeight::try_new(format, shape).unwrap();
        assert_eq!(weight.sources(), vec![SourceRole::Blocks], "{format:?}");
        let expected_shape = [3, 2 * bytes];
        assert_eq!(
            weight.source_shape(SourceRole::Blocks),
            expected_shape,
            "{format:?}"
        );
        assert_eq!(
            weight.source_bytes(SourceRole::Blocks),
            expected_shape[0] * expected_shape[1],
            "{format:?}"
        );
    }

    // Planar formats (safetensors): `[rows, row_bytes]` per role, hand-derived from each operand's
    // `grid`/`packing`/`major`/`bits` in `planar.rs`.
    let e4m3_per_channel = [3, 5];
    assert_planar_shapes(
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        e4m3_per_channel,
        &[(CODES, [3, 5]), (SCALE, [3, 4])],
    );

    let block128 = [129, 130];
    for (scale, row_bytes) in [
        (ScaleEncoding::Bf16, 4),
        (ScaleEncoding::F32, 8),
        (ScaleEncoding::E8m0, 2),
    ] {
        assert_planar_shapes(
            WeightFormat::E4m3Block128 { scale },
            block128,
            &[(CODES, [129, 130]), (SCALE, [2, row_bytes])],
        );
    }

    assert_planar_shapes(
        WeightFormat::E2m1Row32,
        [2, 35],
        &[(CODES, [2, 18]), (SCALE, [2, 2])],
    );

    let gptq_shape = [8, 16];
    assert_planar_shapes(
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nonzero(8) },
        },
        gptq_shape,
        &[
            (CODES, [2, 32]),
            (SourceRole::Planar(OperandRole::Zero), [2, 4]),
            (SCALE, [2, 16]),
        ],
    );
    assert_planar_shapes(
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nonzero(2) },
        },
        gptq_shape,
        &[
            (CODES, [2, 32]),
            (SourceRole::Planar(OperandRole::Zero), [2, 4]),
            (SCALE, [2, 16]),
            (SourceRole::Planar(OperandRole::GroupIndex), [16, 4]),
        ],
    );
    assert_planar_shapes(
        WeightFormat::Awq {
            group_size: nonzero(8),
        },
        gptq_shape,
        &[
            (CODES, [16, 4]),
            (SourceRole::Planar(OperandRole::Zero), [2, 4]),
            (SCALE, [2, 16]),
        ],
    );
}

fn assert_planar_shapes(
    format: WeightFormat,
    shape: [usize; 2],
    expected: &[(SourceRole, [usize; 2])],
) {
    let weight = PackedWeight::try_new(format, shape).unwrap();
    assert_eq!(
        weight.sources(),
        expected.iter().map(|(role, _)| *role).collect::<Vec<_>>(),
        "{format:?}"
    );
    for &(role, expected_shape) in expected {
        assert_eq!(
            weight.source_shape(role),
            expected_shape,
            "{format:?} {role:?}"
        );
        assert_eq!(
            weight.source_bytes(role),
            expected_shape[0] * expected_shape[1],
            "{format:?} {role:?}"
        );
    }
}

/// A GGUF `Q4_0` block decoded through `PackedPayload::decode_row`, proving the block path (not
/// just the planar E4M3/E2M1 path) is real, not only type-level.
#[test]
fn packed_payload_decodes_a_block_format() {
    let weight = PackedWeight::try_new(WeightFormat::Q4_0, [1, 32]).unwrap();
    let mut block = [0u8; 18];
    block[0..2].copy_from_slice(&[0x00, 0x3c]); // f16 1.0
    block[2] = 0x09; // low nibble: element 0, code 9 -> (9 - 8) * 1.0 = 1.0
    // high nibble: element 16, code 0 -> (0 - 8) * 1.0 = -8.0
    let payload = PackedPayload::try_new(weight, [(SourceRole::Blocks, owner(&block))]).unwrap();
    assert_eq!(decoded(&payload, [0, 0]), 1.0);
    assert_eq!(decoded(&payload, [0, 16]), -8.0);
    assert_eq!(decoded(&payload, [0, 1]), -8.0);
}

#[test]
fn packed_payload_try_new_rejects_a_missing_or_unexpected_source() {
    let weight = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [1, 1],
    )
    .unwrap();
    let weight_bytes = owner(&[0x38]);
    let scale_bytes = owner(&[0x00, 0x00, 0x80, 0x3f]);

    assert_eq!(
        PackedPayload::try_new(weight, [(CODES, Arc::clone(&weight_bytes))]).unwrap_err(),
        PackedWeightError::MissingSource {
            format: weight.format(),
            role: SCALE,
        }
    );
    assert_eq!(
        PackedPayload::try_new(
            weight,
            [
                (CODES, Arc::clone(&weight_bytes)),
                (SCALE, Arc::clone(&scale_bytes)),
                (SourceRole::Blocks, Arc::clone(&weight_bytes)),
            ],
        )
        .unwrap_err(),
        PackedWeightError::UnexpectedSource {
            format: weight.format(),
            role: SourceRole::Blocks,
        }
    );
}

#[test]
fn packed_payload_try_new_rejects_wrong_length_or_invalid_content() {
    let one = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [1, 1],
    )
    .unwrap();
    let valid_weight = owner(&[0x38]);
    let valid_f32_scale = owner(&[0x00, 0x00, 0x80, 0x3f]);

    assert_eq!(
        PackedPayload::try_new(
            one,
            [(CODES, owner(&[])), (SCALE, Arc::clone(&valid_f32_scale))]
        )
        .unwrap_err(),
        PackedWeightError::SourceLengthMismatch {
            format: one.format(),
            role: CODES,
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(
        PackedPayload::try_new(
            one,
            [
                (CODES, Arc::clone(&valid_weight)),
                (SCALE, owner(&[0x00, 0x00, 0x80])),
            ],
        )
        .unwrap_err(),
        PackedWeightError::SourceLengthMismatch {
            format: one.format(),
            role: SCALE,
            expected: 4,
            actual: 3,
        }
    );

    let two_weights = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [1, 2],
    )
    .unwrap();
    for nan_byte in [0x7f, 0xff] {
        assert_eq!(
            PackedPayload::try_new(
                two_weights,
                [
                    (CODES, owner(&[0x38, nan_byte])),
                    (SCALE, Arc::clone(&valid_f32_scale)),
                ],
            )
            .unwrap_err(),
            PackedWeightError::NonFiniteField {
                format: two_weights.format(),
                role: CODES,
                operand: OperandRole::Codes,
                element: 1,
            }
        );
    }

    let bf16 = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        [1, 1],
    )
    .unwrap();
    PackedPayload::try_new(
        bf16,
        [
            (CODES, Arc::clone(&valid_weight)),
            (SCALE, owner(&[0x80, 0x3f])),
        ],
    )
    .unwrap();
    // Zero and negative are legitimate scales (finiteness only, not positivity).
    for valid in [[0x00, 0x00], [0x80, 0xbf]] {
        PackedPayload::try_new(
            bf16,
            [(CODES, Arc::clone(&valid_weight)), (SCALE, owner(&valid))],
        )
        .unwrap();
    }
    for invalid in [[0x80, 0x7f], [0xc0, 0x7f]] {
        assert_eq!(
            PackedPayload::try_new(
                bf16,
                [(CODES, Arc::clone(&valid_weight)), (SCALE, owner(&invalid))],
            )
            .unwrap_err(),
            PackedWeightError::NonFiniteField {
                format: bf16.format(),
                role: SCALE,
                operand: OperandRole::Scale,
                element: 0,
            }
        );
    }

    let e8m0 = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [1, 129],
    )
    .unwrap();
    PackedPayload::try_new(
        e8m0,
        [(CODES, owner(&[0; 129])), (SCALE, owner(&[0x00, 0xfe]))],
    )
    .unwrap();
    assert_eq!(
        PackedPayload::try_new(
            e8m0,
            [(CODES, owner(&[0; 129])), (SCALE, owner(&[0x00, 0xff]))],
        )
        .unwrap_err(),
        PackedWeightError::NonFiniteField {
            format: e8m0.format(),
            role: SCALE,
            operand: OperandRole::Scale,
            element: 1,
        }
    );

    let adjacent = PackedWeight::try_new(WeightFormat::E2m1Row32, [2, 35]).unwrap();
    let mut nonzero_padding = vec![0; 36];
    nonzero_padding[17] = 0xa3;
    assert_eq!(
        PackedPayload::try_new(
            adjacent,
            [
                (CODES, Arc::from(nonzero_padding)),
                (SCALE, owner(&[0, 0, 0, 0])),
            ],
        )
        .unwrap_err(),
        PackedWeightError::NonZeroUnusedHighNibble {
            row: 0,
            source_index: 17,
            byte: 0xa3,
        }
    );

    let valid_adjacent = PackedPayload::try_new(
        adjacent,
        [(CODES, owner(&[0; 36])), (SCALE, owner(&[0; 4]))],
    )
    .unwrap();
    let mut row = vec![0.0f32; 35];
    assert_eq!(
        valid_adjacent.decode_row(2, &mut row),
        Err(DecodeError::Coordinate {
            format: WeightFormat::E2m1Row32,
            coordinate: [2, 0],
            shape: [2, 35],
        })
    );
    let mut long_row = vec![0.0f32; 36];
    assert_eq!(
        valid_adjacent.decode_row(0, &mut long_row),
        Err(DecodeError::RowOutput {
            format: WeightFormat::E2m1Row32,
            k: 35,
            len: 36,
        })
    );
}

#[test]
fn e4m3_f32_payload_is_authoritative_shared_and_source_ordered() {
    let weight = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [129, 130],
    )
    .unwrap();
    let mut weight_source = vec![0; 16_770];
    weight_source[0] = 0x38;
    weight_source[127] = 0x40;
    weight_source[128] = 0x42;
    weight_source[129] = 0x40;
    weight_source[130] = 0x42;
    weight_source[16_640] = 0x01;
    weight_source[16_769] = 0xb8;
    let weight_owner: Arc<[u8]> = Arc::from(weight_source);
    let scale_owner = owner(&[
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00,
        0x41,
    ]);
    let weight_pointer = weight_owner.as_ptr();
    let scale_pointer = scale_owner.as_ptr();
    let weight_weak = Arc::downgrade(&weight_owner);
    let scale_weak = Arc::downgrade(&scale_owner);

    let payload = PackedPayload::try_new(
        weight,
        [
            (CODES, Arc::clone(&weight_owner)),
            (SCALE, Arc::clone(&scale_owner)),
        ],
    )
    .unwrap();
    assert_eq!(payload.weight(), weight);
    assert_eq!(payload.bytes(CODES).as_ptr(), weight_pointer);
    assert_eq!(payload.bytes(SCALE).as_ptr(), scale_pointer);
    assert_eq!(payload.bytes(CODES).len(), 16_770);
    assert_eq!(payload.bytes(SCALE).len(), 16);
    assert_eq!(payload.bytes(CODES)[0], 0x38);
    assert_eq!(payload.bytes(CODES)[127], 0x40);
    assert_eq!(payload.bytes(CODES)[128], 0x42);
    assert_eq!(payload.bytes(CODES)[16_769], 0xb8);
    assert_eq!(
        payload.bytes(SCALE),
        &[
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00,
            0x00, 0x41,
        ]
    );

    let cloned = payload.clone();
    assert_eq!(cloned, payload);
    assert_eq!(cloned.bytes(CODES).as_ptr(), weight_pointer);
    assert_eq!(cloned.bytes(SCALE).as_ptr(), scale_pointer);
    drop(weight_owner);
    drop(scale_owner);
    assert!(weight_weak.upgrade().is_some());
    assert!(scale_weak.upgrade().is_some());
    drop(payload);
    assert!(weight_weak.upgrade().is_some());
    assert!(scale_weak.upgrade().is_some());
    drop(cloned);
    assert!(weight_weak.upgrade().is_none());
    assert!(scale_weak.upgrade().is_none());
}

#[test]
fn coordinate_mapping_uses_handwritten_row_and_scale_sentinels() {
    let weight = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [129, 130],
    )
    .unwrap();
    let mut weight_source = vec![0; 16_770];
    weight_source[0] = 0x38;
    weight_source[127] = 0x40;
    weight_source[128] = 0x42;
    weight_source[129] = 0x40;
    weight_source[130] = 0x42;
    weight_source[16_640] = 0x01;
    weight_source[16_769] = 0xb8;
    let payload = PackedPayload::try_new(
        weight,
        [
            (CODES, Arc::from(weight_source)),
            (
                SCALE,
                owner(&[
                    0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00,
                    0x00, 0x00, 0x41,
                ]),
            ),
        ],
    )
    .unwrap();

    // Each value is the E4M3 byte at the coordinate's row-major source index times the F32 scale of
    // its 128x128 block: blocks [0, 0] = 1, [0, 1] = 2, [1, 0] = 4, [1, 1] = 8.
    let sentinels: [([usize; 2], f32); 7] = [
        ([0, 0], 1.0),         // 0x38 = 1.0, block [0, 0]
        ([0, 127], 2.0),       // 0x40 = 2.0, block [0, 0]
        ([0, 128], 5.0),       // 0x42 = 2.5, block [0, 1]
        ([0, 129], 4.0),       // 0x40 = 2.0, block [0, 1]
        ([1, 0], 2.5),         // source index 130: 0x42, block [0, 0]
        ([128, 0], 0.0078125), // source index 16640: 0x01 = 2^-9, block [1, 0]
        ([128, 129], -8.0),    // source index 16769: 0xb8 = -1.0, block [1, 1]
    ];
    for (coordinate, expected) in sentinels {
        assert_eq!(
            decoded(&payload, coordinate).to_bits(),
            expected.to_bits(),
            "{coordinate:?}"
        );
    }
}

#[test]
fn adjacent_e2m1_ragged_padding_and_nibble_mapping_are_exact() {
    let weight = PackedWeight::try_new(WeightFormat::E2m1Row32, [2, 35]).unwrap();
    assert_eq!(weight.source_shape(CODES), [2, 18]);
    assert_eq!(weight.source_shape(SCALE), [2, 2]);

    let mut accepted_weight_source = vec![0; 36];
    accepted_weight_source[0] = 0x51;
    accepted_weight_source[15] = 0xac;
    accepted_weight_source[16] = 0x62;
    accepted_weight_source[17] = 0x03;
    accepted_weight_source[18] = 0xe4;
    let scale_source = [0x7f, 0x80, 0x81, 0x7e];
    let payload = PackedPayload::try_new(
        weight,
        [
            (CODES, Arc::from(accepted_weight_source.clone())),
            (SCALE, owner(&scale_source)),
        ],
    )
    .unwrap();

    // Each value is the E2M1 code in the coordinate's nibble (even K low, odd K high) times the
    // E8M0 scale of its 32-value group: 0x7f = 1, 0x80 = 2, 0x81 = 4.
    let sentinels: [([usize; 2], f32); 8] = [
        ([0, 0], 0.5),   // byte 0 = 0x51, low nibble 1
        ([0, 1], 3.0),   // byte 0 high nibble 5
        ([0, 30], -2.0), // byte 15 = 0xac, low nibble 0xc
        ([0, 31], -1.0), // byte 15 high nibble 0xa
        ([0, 32], 2.0),  // byte 16 = 0x62, low nibble 2, scale 2
        ([0, 33], 8.0),  // byte 16 high nibble 6 (4.0), scale 2
        ([0, 34], 3.0),  // byte 17 = 0x03, low nibble 3 (1.5), scale 2
        ([1, 0], 8.0),   // byte 18 = 0xe4, low nibble 4 (2.0), scale 4
    ];
    for (coordinate, expected) in sentinels {
        assert_eq!(
            decoded(&payload, coordinate).to_bits(),
            expected.to_bits(),
            "{coordinate:?}"
        );
    }

    let mut rejected_weight_source = accepted_weight_source.clone();
    rejected_weight_source[17] = 0xa3;
    assert_eq!(
        PackedPayload::try_new(
            weight,
            [
                (CODES, Arc::from(rejected_weight_source.clone())),
                (SCALE, owner(&scale_source)),
            ],
        )
        .unwrap_err(),
        PackedWeightError::NonZeroUnusedHighNibble {
            row: 0,
            source_index: 17,
            byte: 0xa3,
        }
    );
    rejected_weight_source[17] = 0x03;
    PackedPayload::try_new(
        weight,
        [
            (CODES, Arc::from(rejected_weight_source.clone())),
            (SCALE, owner(&scale_source)),
        ],
    )
    .unwrap();

    rejected_weight_source[35] = 0xb0;
    assert_eq!(
        PackedPayload::try_new(
            weight,
            [
                (CODES, Arc::from(rejected_weight_source)),
                (SCALE, owner(&scale_source)),
            ],
        )
        .unwrap_err(),
        PackedWeightError::NonZeroUnusedHighNibble {
            row: 1,
            source_index: 35,
            byte: 0xb0,
        }
    );
}

/// Card 642 (dquant.md R5): `PackedPayload::decode_row` decodes a whole logical row
/// through the scalar decoder, bit for bit, for every block format and every planar family (E4M3,
/// E2M1, GPTQ contiguous and act-order, AWQ), on a multi-row weight so a wrong row offset cannot
/// hide. The reference is the descriptor itself: `decode_blocks` over the row's bytes, sliced here,
/// for a block format, and per-value `decode_planar_value` for a planar one. A row past `out` and
/// an output slice of the wrong length are refused. Mutation: read every row's bytes from row 0
/// (`start = 0` in the block arm); the Q4_0 `[1,0]` comparison goes red.
#[test]
fn decode_row_matches_the_scalar_decoder_bit_for_bit() {
    let block_formats = [
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q8_0,
        WeightFormat::Iq4_Nl,
        WeightFormat::Mxfp4,
        WeightFormat::Q2_K,
        WeightFormat::Q3_K,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
        WeightFormat::Iq4_Xs,
    ];
    let planar_formats = [
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E2m1Row32,
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nonzero(64) },
        },
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nonzero(4) },
        },
        WeightFormat::Awq {
            group_size: nonzero(64),
        },
    ];
    for (seed, format) in block_formats.into_iter().chain(planar_formats).enumerate() {
        let shape = [3, 512];
        let payload = poot_test_util::packed::random_payload(format, shape, seed as u64);
        let descriptor = format.descriptor();
        let mut row = vec![0.0f32; shape[1]];
        let mut reference = vec![0.0f32; shape[1]];
        for r in 0..shape[0] {
            payload.decode_row(r, &mut row).unwrap();
            if let Storage::Blocks(_) = descriptor.storage {
                let [_, row_bytes] = payload.weight().source_shape(SourceRole::Blocks);
                let bytes = &payload.bytes(SourceRole::Blocks)[r * row_bytes..(r + 1) * row_bytes];
                descriptor.decode_blocks(bytes, &mut reference).unwrap();
            } else {
                let sources: Vec<(OperandRole, &[u8])> = payload
                    .weight()
                    .sources()
                    .into_iter()
                    .map(|role| {
                        let SourceRole::Planar(operand) = role else {
                            unreachable!("a planar weight has only planar sources")
                        };
                        (operand, payload.bytes(role))
                    })
                    .collect();
                for (c, value) in reference.iter_mut().enumerate() {
                    *value = descriptor
                        .decode_planar_value(shape, &sources, [r, c])
                        .unwrap();
                }
            }
            for (c, (value, expected)) in row.iter().zip(&reference).enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "{format:?} [{r},{c}]: row decode {value} vs scalar decoder {expected}"
                );
            }
        }
        assert!(matches!(
            payload.decode_row(shape[0], &mut row),
            Err(DecodeError::Coordinate { .. })
        ));
        assert!(matches!(
            payload.decode_row(0, &mut row[1..]),
            Err(DecodeError::RowOutput {
                k: 512,
                len: 511,
                ..
            })
        ));
    }
}

fn decoded_rows(payload: &PackedPayload) -> Vec<Vec<u32>> {
    let [out, k] = payload.weight().shape();
    (0..out)
        .map(|r| {
            let mut row = vec![0.0f32; k];
            payload.decode_row(r, &mut row).unwrap();
            row.into_iter().map(f32::to_bits).collect()
        })
        .collect()
}

/// The loader's row ops (a fused-projection slice, a q/k un-permute, gate||up concatenation and
/// gpt-oss's gate/up interleave) are one `gather_rows` over the stored blocks: row `i` of the result
/// decodes bit for bit to the stacked parts' row `rows[i]`, for every block format. A planar
/// format, mismatched parts and a row past the stack are refused.
#[test]
fn gather_rows_moves_whole_stored_rows_of_every_block_format() {
    let block_formats = [
        WeightFormat::Q4_0,
        WeightFormat::Q8_0,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
        WeightFormat::Mxfp4,
        WeightFormat::Iq4_Xs,
    ];
    for (seed, format) in block_formats.into_iter().enumerate() {
        let gate = poot_test_util::packed::random_payload(format, [4, 256], 2 * seed as u64);
        let up = poot_test_util::packed::random_payload(format, [3, 256], 2 * seed as u64 + 1);
        let stacked: Vec<Vec<u32>> = decoded_rows(&gate)
            .into_iter()
            .chain(decoded_rows(&up))
            .collect();
        for rows in [
            vec![1, 2],                // a slice
            vec![0, 2, 1, 3],          // a permutation
            (0..7).collect(),          // a concatenation
            vec![0, 4, 1, 5, 2, 6, 3], // an interleave
        ] {
            let gathered = PackedPayload::gather_rows(&[&gate, &up], &rows).unwrap();
            assert_eq!(gathered.weight().shape(), [rows.len(), 256]);
            let got = decoded_rows(&gathered);
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(got[i], stacked[row], "{format:?} rows {rows:?}: row {i}");
            }
        }
        assert_eq!(
            PackedPayload::gather_rows(&[&gate, &up], &[7]),
            Err(PackedWeightError::RowOutOfRange { row: 7, rows: 7 })
        );
    }
    let q8 = poot_test_util::packed::random_payload(WeightFormat::Q8_0, [2, 64], 0);
    let q4 = poot_test_util::packed::random_payload(WeightFormat::Q4_0, [2, 64], 1);
    assert!(matches!(
        PackedPayload::gather_rows(&[&q8, &q4], &[0]),
        Err(PackedWeightError::RowGatherParts { .. })
    ));
    let awq = poot_test_util::packed::random_payload(
        WeightFormat::Awq {
            group_size: nonzero(64),
        },
        [8, 64],
        2,
    );
    assert!(matches!(
        PackedPayload::gather_rows(&[&awq], &[0]),
        Err(PackedWeightError::RowGatherStorage { .. })
    ));
}
