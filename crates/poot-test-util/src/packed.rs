//! Admitted-domain random packed payloads for device and oracle tests (card 642): every stored float
//! factor is finite, so `PackedPayload::try_new`'s content check admits the fixture, and every other
//! bit pattern of every field is exercised.

use std::sync::Arc;

use poot_quant::format::{
    FieldEncoding, FloatFormat, GroupMap, OperandRole, Storage, WeightFormat,
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
        FloatFormat::F32 | FloatFormat::E2m1 => {}
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
                        .flat_map(|k| (((k * 7) % groups.get()) as i32).to_le_bytes())
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
