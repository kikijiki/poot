//! Admitted-domain random packed payloads for device and oracle tests (card 642): every stored float
//! factor is finite, so `PackedPayload::try_new`'s content check admits the fixture, and every other
//! bit pattern of every field is exercised.

use std::sync::Arc;

use poot_quant::format::{
    FieldEncoding, FloatFormat, GroupMap, OperandRole, ScaleEncoding, Storage, WeightFormat,
};
use poot_quant::{PackedPayload, PackedWeight, SourceRole};

/// xorshift64 bytes: deterministic, every bit pattern of every field.
pub fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// Clamp one stored float of `float`'s encoding off its non-finite bit patterns.
fn sanitize_float(value: &mut [u8], float: FloatFormat) {
    match float {
        FloatFormat::F16 | FloatFormat::Bf16 => value[1] &= !0x40,
        FloatFormat::E4m3Fn => {
            if value[0] & 0x7f == 0x7f {
                value[0] &= !0x01;
            }
        }
        FloatFormat::E8m0 => value[0] = value[0].min(0xfe),
        FloatFormat::F32 => value[3] &= !0x40,
        FloatFormat::E2m1 => {}
    }
}

/// An admitted-domain payload of `format` at `[out, K]` `shape`: random codes, sanitized float
/// factors, and (act-order GPTQ) an in-range group index.
pub fn random_payload(format: WeightFormat, shape: [usize; 2], seed: u64) -> PackedPayload {
    let weight = PackedWeight::try_new(format, shape)
        .unwrap_or_else(|error| panic!("{format:?} {shape:?}: {error:?}"));
    let descriptor = format.descriptor();
    let sources: Vec<(SourceRole, Arc<[u8]>)> = weight
        .sources()
        .into_iter()
        .enumerate()
        .map(|(index, role)| {
            let seed = seed.wrapping_add(0x9e37_79b9 * (index as u64 + 1));
            let bytes = match (descriptor.storage, role) {
                (Storage::Blocks(layout), SourceRole::Blocks) => {
                    let mut bytes = random_bytes(weight.source_bytes(role), seed);
                    for block in bytes.chunks_mut(layout.bytes) {
                        for field_role in [OperandRole::Scale, OperandRole::Min] {
                            let Some(field) = layout.field(field_role) else {
                                continue;
                            };
                            if let FieldEncoding::Float(float) = field.field.encoding {
                                let byte = (field.field.pieces[0].layout.bit(0) / 8) as usize;
                                sanitize_float(&mut block[byte..], float);
                            }
                        }
                    }
                    bytes
                }
                (Storage::Planar(_), SourceRole::Planar(OperandRole::GroupIndex)) => {
                    let WeightFormat::Gptq {
                        groups: GroupMap::Indexed { groups },
                    } = format
                    else {
                        unreachable!("only act-order GPTQ has a GroupIndex operand")
                    };
                    (0..shape[1])
                        .flat_map(|k| (((k * 7) % groups) as i32).to_le_bytes())
                        .collect()
                }
                (Storage::Planar(_), SourceRole::Planar(operand_role)) => {
                    let mut bytes = random_bytes(weight.source_bytes(role), seed);
                    let operand = descriptor.planar_operand(operand_role).unwrap();
                    if let FieldEncoding::Float(float) = operand.encoding {
                        for value in bytes.chunks_mut(operand.element_bytes()) {
                            sanitize_float(value, float);
                        }
                    }
                    bytes
                }
                (storage, role) => unreachable!("{storage:?} has no {role:?} source"),
            };
            (role, Arc::from(bytes))
        })
        .collect();
    PackedPayload::try_new(weight, sources)
        .unwrap_or_else(|error| panic!("{format:?} {shape:?}: {error:?}"))
}

/// [`random_payload`] for a contraction fixture that must be sensitive to its scales: an E4M3 payload keeps its
/// random codes but gets prescribed nonzero scales (in the format's own BF16 or F32 encoding) cycling through
/// 0.5, 1.0, 1.5 and 2.0, so every output depends on every scale (a random sanitized scale is usually near
/// zero, which hides a wrong scale); a block payload with E8M0 scales (MXFP4) keeps its codes and gets scales
/// cycling through 0.25 ..= 16 (a random E8M0 scale spans `2^-127 ..= 2^127`, so a one-hot row would read
/// `0 * inf`; the cycle has seven values and a seed-dependent phase, so a misread block or expert is visible).
/// Any other format is the plain random payload.
pub fn prescribed_scale_payload(
    format: WeightFormat,
    shape: [usize; 2],
    seed: u64,
) -> PackedPayload {
    let base = random_payload(format, shape, seed);
    if has_e8m0_block_scale(format) {
        // Seven scales (0.25 ..= 16) cycling with a stride that depends on the seed: two blocks four apart, or the
        // same block of two experts, differ, so a scale read from the wrong block moves the decoded value.
        return e8m0_scaled_payload(format, shape, seed, |block| {
            0x7d + ((block * 3 + seed as usize) % 7) as u8
        });
    }
    let (WeightFormat::E4m3PerChannel { scale } | WeightFormat::E4m3Block128 { scale }) = format
    else {
        return base;
    };
    const SCALES: [f32; 4] = [0.5, 1.0, 1.5, 2.0];
    let weight = base.weight();
    let sources: Vec<(SourceRole, Arc<[u8]>)> = weight
        .sources()
        .into_iter()
        .map(|role| {
            let mut bytes = base.bytes(role).to_vec();
            if role == SourceRole::Planar(OperandRole::Scale) {
                let width = match scale {
                    ScaleEncoding::Bf16 => 2,
                    ScaleEncoding::F32 => 4,
                    ScaleEncoding::E8m0 => unreachable!("E4M3 scales are BF16 or F32"),
                };
                for (index, element) in bytes.chunks_mut(width).enumerate() {
                    let bits = SCALES[index % SCALES.len()].to_bits();
                    match scale {
                        ScaleEncoding::Bf16 => {
                            element.copy_from_slice(&((bits >> 16) as u16).to_le_bytes());
                        }
                        _ => element.copy_from_slice(&bits.to_le_bytes()),
                    }
                }
            }
            (role, Arc::from(bytes))
        })
        .collect();
    PackedPayload::try_new(weight, sources)
        .unwrap_or_else(|error| panic!("{format:?} {shape:?}: {error:?}"))
}

fn has_e8m0_block_scale(format: WeightFormat) -> bool {
    matches!(format.descriptor().storage, Storage::Blocks(layout)
        if layout.field(OperandRole::Scale).is_some_and(|scale| scale.field.encoding == FieldEncoding::Float(FloatFormat::E8m0)))
}

/// [`random_payload`] of a block format with an E8M0 scale (MXFP4) whose block `b` carries the scale byte
/// `scale(b)`: the way to put the extreme admitted exponents (`0x00` = `2^-127`, `0xfe` = `2^127`) into a
/// payload. `0xff` (NaN) is not admitted by [`PackedPayload::try_new`], so it cannot be built here.
pub fn e8m0_scaled_payload(
    format: WeightFormat,
    shape: [usize; 2],
    seed: u64,
    scale: impl Fn(usize) -> u8,
) -> PackedPayload {
    let base = random_payload(format, shape, seed);
    let Storage::Blocks(layout) = format.descriptor().storage else {
        panic!("{format:?}: not a block format");
    };
    let field = layout
        .field(OperandRole::Scale)
        .unwrap_or_else(|| panic!("{format:?}: no scale field"));
    assert_eq!(
        field.field.encoding,
        FieldEncoding::Float(FloatFormat::E8m0),
        "{format:?}"
    );
    let at = (field.field.pieces[0].layout.bit(0) / 8) as usize;
    let mut bytes = base.bytes(SourceRole::Blocks).to_vec();
    for (block, bytes) in bytes.chunks_mut(layout.bytes).enumerate() {
        bytes[at] = scale(block);
    }
    PackedPayload::try_new(base.weight(), [(SourceRole::Blocks, Arc::from(bytes))])
        .unwrap_or_else(|error| panic!("{format:?} {shape:?}: {error:?}"))
}
