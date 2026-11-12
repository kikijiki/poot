//! Packed-linear fixtures at real weight magnitude: one projection per quantization scheme (Q4_0,
//! Q8_0, AWQ, GPTQ, IQ3_S), and the indexed expert contraction over a stacked table of each.
//!
//! The codes are every bit pattern, but each stored scale is a plausible magnitude, so a decoded
//! weight is of the order of a trained one (`|w| <= 0.25`) and the activation is of unit order: a
//! role swapped or a group misread changes the output by far more than the parity tolerance, where a
//! payload of scale bytes drawn from every bit pattern would put outputs at magnitudes no real
//! checkpoint has.

use std::sync::{Arc, LazyLock};

use poot_graph_ir::test_support::packed_graphs::{
    PackedExpertStack, packed_indexed_linear, packed_linear,
};
use poot_graph_ir::{Builder, Slot, TensorType};
use poot_quant::StoredBytes;
use poot_quant::format::{
    FieldEncoding, FloatFormat, GroupMap, OperandRole, PlanarRole, ScaleEncoding,
    Storage as FormatStorage, WeightFormat,
};
use poot_quant::test_support::formats::{GGUF_FORMATS, SAFETENSORS_FORMATS};
use poot_quant::test_support::packed::prescribed_scale_payload;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, SourceRole};
use poot_tensor::HostTensor;

use poot_eval::test_support::StepFixture;

use crate::{Fixture, fill};

/// The contraction length of every fixture: a multiple of every scheme's group (a Q4_0/Q8_0 block of
/// 32, the AWQ group of 64, four GPTQ groups of 64, the IQ3_S super-block of 256) and of the planar pack width.
const K: usize = 256;
/// Output channels of one linear.
const OUT: usize = 8;
/// The largest decoded weight magnitude the payloads are scaled to, unless a scheme states its own.
const PEAK: f32 = 0.25;
/// Experts of a chain, and rows routed through it.
const EXPERTS: usize = 3;
const ROWS: usize = 5;

/// How a scheme's payload is made.
enum Magnitude {
    /// Random codes with the first scale written so the decoded weights peak at `peak`: a lattice quant's typical
    /// weight is a fraction of its largest (a random grid point and sub-scale), so its peak is higher for the outputs
    /// to be of order one. `max_code` is the largest code magnitude the decode multiplies a scale by, so a scale of
    /// `peak / max_code` keeps the decoded weight within `peak`.
    Scaled { peak: f32, max_code: f32 },
    /// The quant crate's admitted-domain payload ([`prescribed_scale_payload`]): random codes, finite stored
    /// factors, and prescribed scales for the formats whose random scales span too many binades (E4M3, E8M0).
    Admitted,
}

/// One packed format the device lanes check: how its payload is made, and the fixture names of its decode
/// (`M = 1`) and few-row (`M = 4`) linears, and of its indexed expert contraction.
struct Scheme {
    format: WeightFormat,
    magnitude: Magnitude,
    linear_m1: &'static str,
    linear_m4: &'static str,
    experts: &'static str,
}

/// One scheme per packed format the device lanes check: every GGUF and safetensors format of the shared
/// format list that is a packed weight at the fixtures' shape (the dense dtypes are not), so a new format is
/// covered by every backend's parity rows with no edit here. `the_schemes_are_every_format_the_lowering_table_admits`
/// fails when a scheme's format has no `PACKED_LOWERING` row.
static SCHEMES: LazyLock<Vec<Scheme>> = LazyLock::new(|| {
    GGUF_FORMATS
        .into_iter()
        .chain(SAFETENSORS_FORMATS)
        .filter(|&format| PackedWeight::try_new(format, [OUT, K]).is_ok())
        .map(Scheme::of)
        .collect()
});

impl Scheme {
    /// The scheme of `format`: its magnitude and fixture names.
    fn of(format: WeightFormat) -> Self {
        let stem = Self::stem(format);
        Self {
            format,
            magnitude: Self::magnitude(format),
            linear_m1: format!("packed_linear_{stem}_m1").leak(),
            linear_m4: format!("packed_linear_{stem}_m4").leak(),
            experts: format!("packed_experts_{stem}").leak(),
        }
    }

    /// The fixture-name fragment of `format`: its ggml name for a block format, the layout and scale for a
    /// planar one.
    fn stem(format: WeightFormat) -> String {
        let scale = |scale| match scale {
            ScaleEncoding::F32 => "f32",
            ScaleEncoding::Bf16 => "bf16",
            ScaleEncoding::E8m0 => "e8m0",
        };
        match format {
            WeightFormat::E4m3PerChannel { scale: encoding } => {
                format!("e4m3_channel_{}", scale(encoding))
            }
            WeightFormat::E4m3Block128 { scale: encoding } => {
                format!("e4m3_block_{}", scale(encoding))
            }
            WeightFormat::E2m1Row32 => "e2m1".to_string(),
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { .. },
            } => "gptq_contiguous".to_string(),
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { .. },
            } => "gptq".to_string(),
            WeightFormat::Awq { .. } => "awq".to_string(),
            block => format!("{block:?}").to_lowercase(),
        }
    }

    /// How `format`'s payload is made: the formats whose random scales would put a decoded weight far above a
    /// trained one state the largest code magnitude their decode multiplies a scale by.
    fn magnitude(format: WeightFormat) -> Magnitude {
        match format {
            WeightFormat::Q4_0 => Magnitude::Scaled {
                peak: PEAK,
                max_code: 8.0,
            },
            WeightFormat::Q8_0 => Magnitude::Scaled {
                peak: PEAK,
                max_code: 127.0,
            },
            // The largest decoded magnitude is a grid point of 15 times an odd-integer sub-scale of 31.
            WeightFormat::Iq3_S => Magnitude::Scaled {
                peak: 2.0 * PEAK,
                max_code: 15.0 * 31.0,
            },
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { .. },
            } => Magnitude::Scaled {
                peak: PEAK,
                max_code: 16.0,
            },
            WeightFormat::Awq { .. } => Magnitude::Scaled {
                peak: PEAK,
                max_code: 15.0,
            },
            _ => Magnitude::Admitted,
        }
    }
}

/// xorshift64 bytes: every bit pattern of every code field.
fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
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

/// The binary16 bits of a positive normal `value`.
pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!(
        (1..31).contains(&exponent),
        "{value} is not a binary16 normal"
    );
    ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// Write the `index`th scale of a payload (in `[0.5, 1.0) * peak / max_code`) into `out`, as `float`.
fn write_scale(float: FloatFormat, out: &mut [u8], index: u64, peak: f32, max_code: f32) {
    let unit = (random_bytes(1, index.wrapping_mul(0x9E37_79B9) | 1)[0] as f32) / 256.0;
    let scale = peak / max_code * (0.5 + 0.5 * unit);
    match float {
        FloatFormat::F32 => out[..4].copy_from_slice(&scale.to_le_bytes()),
        FloatFormat::F16 => out[..2].copy_from_slice(&f16_bits(scale).to_le_bytes()),
        FloatFormat::Bf16 => {
            out[..2].copy_from_slice(&((scale.to_bits() >> 16) as u16).to_le_bytes())
        }
        other => {
            panic!("no real-magnitude scale for {other:?}: the scheme table holds no such format")
        }
    }
}

/// A payload of `scheme`'s format at `[OUT, K]`.
fn payload(scheme: &Scheme, seed: u64) -> Arc<PackedPayload> {
    match scheme.magnitude {
        Magnitude::Scaled { peak, max_code } => {
            real_magnitude_payload(scheme.format, peak, max_code, seed)
        }
        Magnitude::Admitted => Arc::new(prescribed_scale_payload(scheme.format, [OUT, K], seed)),
    }
}

/// A payload of `format` at `[OUT, K]` with random codes and real-magnitude scales, and (for act-order GPTQ) an
/// in-range group index.
fn real_magnitude_payload(
    format: WeightFormat,
    peak: f32,
    max_code: f32,
    seed: u64,
) -> Arc<PackedPayload> {
    let weight = PackedWeight::try_new(format, [OUT, K]).unwrap();
    let descriptor = format.descriptor();
    let sources: Vec<(SourceRole, StoredBytes)> = weight
        .sources()
        .into_iter()
        .enumerate()
        .map(|(index, role)| {
            let seed = seed.wrapping_add(0x9e37_79b9 * (index as u64 + 1));
            let mut bytes = random_bytes(weight.source_bytes(role), seed);
            match (descriptor.storage, role) {
                (FormatStorage::Blocks(layout), SourceRole::Blocks) => {
                    for (block, bytes) in bytes.chunks_mut(layout.bytes).enumerate() {
                        let field = layout
                            .field(OperandRole::Scale)
                            .expect("a block scheme has a scale");
                        let FieldEncoding::Float(float) = field.field.encoding else {
                            panic!("{format:?}: the block scale is not a float");
                        };
                        let byte = (field.field.pieces[0].layout.bit(0) / 8) as usize;
                        write_scale(
                            float,
                            &mut bytes[byte..],
                            seed ^ block as u64,
                            peak,
                            max_code,
                        );
                    }
                }
                (FormatStorage::Planar(_), SourceRole::Planar(PlanarRole::GroupIndex)) => {
                    let GroupMap::Indexed { groups } = (match format {
                        WeightFormat::Gptq { groups } => groups,
                        _ => unreachable!("only GPTQ has a group index"),
                    }) else {
                        unreachable!("only act-order GPTQ has a group index");
                    };
                    bytes = (0..K)
                        .flat_map(|k| (((k * 7) % groups) as i32).to_le_bytes())
                        .collect();
                }
                (FormatStorage::Planar(_), SourceRole::Planar(operand_role)) => {
                    let operand = descriptor.planar_operand(operand_role).unwrap();
                    if let FieldEncoding::Float(float) = operand.encoding {
                        for (element, value) in
                            bytes.chunks_mut(operand.element_bytes()).enumerate()
                        {
                            write_scale(float, value, seed ^ element as u64, peak, max_code);
                        }
                    }
                }
                (storage, role) => unreachable!("{storage:?} has no {role:?} source"),
            }
            (role, StoredBytes::from(bytes))
        })
        .collect();
    let payload = PackedPayload::try_new(weight, sources).unwrap();
    // The fixture's own check: the decoded weights are of trained-weight magnitude, so the
    // activation-times-weight outputs are of order one, not of a random bit pattern's magnitude.
    let mut row = vec![0.0f32; K];
    let mut decoded_peak = 0.0f32;
    for o in 0..OUT {
        payload.decode_row(o, &mut row).unwrap();
        decoded_peak = row.iter().fold(decoded_peak, |most, w| most.max(w.abs()));
    }
    assert!(
        decoded_peak > peak / 8.0 && decoded_peak <= peak * 1.001,
        "{format:?}: decoded peak {decoded_peak} outside ({}, {peak}]",
        peak / 8.0
    );
    Arc::new(payload)
}

/// A store with one packed entry per `(linear id, payload)`.
fn packed_store(entries: &[(String, Arc<PackedPayload>)]) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (linear_id, payload) in entries {
        builder
            .insert(linear_id.clone(), WeightEntry::Packed(Arc::clone(payload)))
            .unwrap();
    }
    builder.build()
}

/// `[rows, width]` activations of unit order, a different draw per step.
fn activations(rows: usize, width: usize, step: u64) -> HostTensor {
    HostTensor::f32(
        vec![rows, width],
        fill(rows * width, 0xAC7 + step)
            .into_iter()
            .map(|v| v * 10.0)
            .collect(),
    )
}

/// One packed linear per scheme at the decode GEMV (`M = 1`) and at a few rows (`M = 4`): `y = x W^T`
/// over a `[OUT, K]` packed weight, two steps with different activations.
pub fn packed_linear_fixtures() -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for (index, scheme) in SCHEMES.iter().enumerate() {
        let payload = payload(scheme, 11 + index as u64);
        for (name, m) in [(scheme.linear_m1, 1), (scheme.linear_m4, 4)] {
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, K]))
                .unwrap();
            let y = packed_linear(&b, x, "layer", payload.weight(), None, None).unwrap();
            let graph = b.finish(y).unwrap();
            let key = graph.meta(x.id()).slot_key().unwrap().clone();
            let steps = (0..2)
                .map(|step| {
                    vec![StepFixture {
                        key: key.clone(),
                        tensor: activations(m, K, step),
                    }]
                })
                .collect();
            fixtures.push(Fixture {
                name,
                graph,
                store: packed_store(&[("layer".to_string(), Arc::clone(&payload))]),
                steps,
            });
        }
    }
    fixtures
}

/// The indexed packed expert contraction per scheme: `EXPERTS` distinct packed experts stacked under the id
/// `experts` (the store holds them as `experts.0`, `experts.1`, ..), `ROWS` activations, row `m` contracting
/// against expert `ids[m]`. Two steps with different activations and different routes.
pub fn packed_expert_fixtures() -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for (index, scheme) in SCHEMES.iter().enumerate() {
        let payloads: Vec<Arc<PackedPayload>> = (0..EXPERTS)
            .map(|e| payload(scheme, 101 + 7 * e as u64 + index as u64))
            .collect();
        let table = PackedExpertStack {
            stack: poot_graph_ir::test_support::packed_graphs::TEST_STACK,
            descriptor: payloads[0].weight(),
            experts: EXPERTS,
        };
        let entries: Vec<(String, Arc<PackedPayload>)> = payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| {
                let stack = poot_graph_ir::test_support::packed_graphs::TEST_STACK.const_name();
                (format!("{stack}.{expert}"), Arc::clone(payload))
            })
            .collect();
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, K]))
            .unwrap();
        let ids = b
            .slot_named_ranged(
                Slot::Activation,
                "moe-ids",
                TensorType::new(vec![ROWS], poot_tensor::DType::I32),
                poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
            )
            .unwrap();
        let y = packed_indexed_linear(&b, x, ids, &table).unwrap();
        let graph = b.finish(y).unwrap();
        let (kx, ki) = (
            graph.meta(x.id()).slot_key().unwrap().clone(),
            graph.meta(ids.id()).slot_key().unwrap().clone(),
        );
        let steps = (0..2usize)
            .map(|step| {
                let routed: Vec<i32> = (0..ROWS)
                    .map(|m| ((m * 2 + 1 + step) % EXPERTS) as i32)
                    .collect();
                vec![
                    StepFixture {
                        key: kx.clone(),
                        tensor: activations(ROWS, K, step as u64),
                    },
                    StepFixture {
                        key: ki.clone(),
                        tensor: HostTensor::i32(vec![ROWS], routed),
                    },
                ]
            })
            .collect();
        fixtures.push(Fixture {
            name: scheme.experts,
            graph,
            store: packed_store(&entries),
            steps,
        });
    }
    fixtures
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_plan::packed_block_float::PACKED_LOWERING;

    /// A scheme's format is lowered when a row of `PACKED_LOWERING` admits its descriptor.
    fn is_lowered(format: WeightFormat) -> bool {
        PACKED_LOWERING
            .iter()
            .any(|row| (row.admits)(&format.descriptor()))
    }

    /// SC-006 (card 1218): every packed format of the shared format list has a scheme, so a device parity row on
    /// every backend that runs these fixtures, and every scheme's format is one a `PACKED_LOWERING` row admits.
    /// A scheme names no format twice. Mutation: remove the `PACKED_LOWERING` row of one format; this goes red
    /// naming the format.
    #[test]
    fn the_schemes_are_every_format_the_lowering_table_admits() {
        let listed: Vec<WeightFormat> = GGUF_FORMATS
            .into_iter()
            .chain(SAFETENSORS_FORMATS)
            .collect();
        for &format in &listed {
            let packed = PackedWeight::try_new(format, [OUT, K]).is_ok();
            let has_scheme = SCHEMES.iter().any(|scheme| scheme.format == format);
            assert_eq!(
                has_scheme, packed,
                "{format:?}: a packed weight has a scheme and a dense dtype has none"
            );
            if has_scheme {
                assert!(
                    is_lowered(format),
                    "{format:?}: has a scheme but no PACKED_LOWERING row admits it"
                );
            }
        }
        for (index, scheme) in SCHEMES.iter().enumerate() {
            assert!(
                listed.contains(&scheme.format),
                "{:?} is not in the format list",
                scheme.format
            );
            assert!(
                !SCHEMES[..index]
                    .iter()
                    .any(|earlier| earlier.format == scheme.format
                        || earlier.experts == scheme.experts),
                "{:?} has two schemes",
                scheme.format
            );
        }
    }

    /// Every scheme builds its payload (the real-magnitude peak check runs) and all three fixtures.
    #[test]
    fn every_scheme_builds_its_fixtures() {
        assert_eq!(packed_linear_fixtures().len(), 2 * SCHEMES.len());
        assert_eq!(packed_expert_fixtures().len(), SCHEMES.len());
    }
}
