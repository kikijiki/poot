//! Packed-linear fixtures at real weight magnitude: one projection per quantization scheme (Q4_0,
//! Q8_0, AWQ, GPTQ, IQ3_S), and the indexed expert contraction over a stacked table of each.
//!
//! The codes are every bit pattern, but each stored scale is a plausible magnitude, so a decoded
//! weight is of the order of a trained one (`|w| <= 0.25`) and the activation is of unit order: a
//! role swapped or a group misread changes the output by far more than the parity tolerance, where a
//! payload of scale bytes drawn from every bit pattern would put outputs at magnitudes no real
//! checkpoint has.

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::test_support::packed_graphs::{
    PackedExpertStack, packed_indexed_linear, packed_linear,
};
use poot_graph_ir::{Builder, Slot, TensorType};
use poot_quant::StoredBytes;
use poot_quant::format::{
    FieldEncoding, FloatFormat, GroupMap, OperandRole, ScaleEncoding, Storage as FormatStorage,
    WeightFormat,
};
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

/// A [`Scheme`] named `stem` in its three fixture names.
macro_rules! scheme {
    ($stem:literal, $format:expr, $magnitude:expr) => {
        Scheme {
            format: $format,
            magnitude: $magnitude,
            linear_m1: concat!("packed_linear_", $stem, "_m1"),
            linear_m4: concat!("packed_linear_", $stem, "_m4"),
            experts: concat!("packed_experts_", $stem),
        }
    };
}

const fn nonzero(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// One scheme per packed format `PACKED_LOWERING` lowers (`the_schemes_are_every_format_the_lowering_table_admits`
/// fails when a format is missing), so each has a device parity row on every backend that runs these fixtures.
const SCHEMES: [Scheme; 26] = [
    scheme!(
        "q4_0",
        WeightFormat::Q4_0,
        Magnitude::Scaled {
            peak: PEAK,
            max_code: 8.0
        }
    ),
    scheme!("q4_1", WeightFormat::Q4_1, Magnitude::Admitted),
    scheme!("q5_0", WeightFormat::Q5_0, Magnitude::Admitted),
    scheme!("q5_1", WeightFormat::Q5_1, Magnitude::Admitted),
    scheme!(
        "q8_0",
        WeightFormat::Q8_0,
        Magnitude::Scaled {
            peak: PEAK,
            max_code: 127.0
        }
    ),
    scheme!("q2_k", WeightFormat::Q2_K, Magnitude::Admitted),
    scheme!("q3_k", WeightFormat::Q3_K, Magnitude::Admitted),
    scheme!("q4_k", WeightFormat::Q4_K, Magnitude::Admitted),
    scheme!("q5_k", WeightFormat::Q5_K, Magnitude::Admitted),
    scheme!("q6_k", WeightFormat::Q6_K, Magnitude::Admitted),
    scheme!("iq4_nl", WeightFormat::Iq4_Nl, Magnitude::Admitted),
    scheme!("iq4_xs", WeightFormat::Iq4_Xs, Magnitude::Admitted),
    scheme!("iq2_xs", WeightFormat::Iq2_Xs, Magnitude::Admitted),
    scheme!("iq3_xxs", WeightFormat::Iq3_Xxs, Magnitude::Admitted),
    // The largest decoded magnitude is a grid point of 15 times an odd-integer sub-scale of 31.
    scheme!(
        "iq3_s",
        WeightFormat::Iq3_S,
        Magnitude::Scaled {
            peak: 2.0 * PEAK,
            max_code: 15.0 * 31.0
        }
    ),
    scheme!("mxfp4", WeightFormat::Mxfp4, Magnitude::Admitted),
    scheme!(
        "e4m3_channel_bf16",
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::Bf16
        },
        Magnitude::Admitted
    ),
    scheme!(
        "e4m3_channel_f32",
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32
        },
        Magnitude::Admitted
    ),
    scheme!(
        "e4m3_block_bf16",
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16
        },
        Magnitude::Admitted
    ),
    scheme!(
        "e4m3_block_f32",
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32
        },
        Magnitude::Admitted
    ),
    scheme!("e2m1", WeightFormat::E2m1Row32, Magnitude::Admitted),
    scheme!(
        "gptq_contiguous",
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nonzero(64) }
        },
        Magnitude::Admitted
    ),
    scheme!(
        "gptq",
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nonzero(4) }
        },
        Magnitude::Scaled {
            peak: PEAK,
            max_code: 16.0
        }
    ),
    scheme!(
        "awq",
        WeightFormat::Awq {
            group_size: nonzero(64)
        },
        Magnitude::Scaled {
            peak: PEAK,
            max_code: 15.0
        }
    ),
    scheme!(
        "e4m3_channel_e8m0",
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::E8m0
        },
        Magnitude::Admitted
    ),
    scheme!(
        "e4m3_block_e8m0",
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0
        },
        Magnitude::Admitted
    ),
];

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
                (FormatStorage::Planar(_), SourceRole::Planar(OperandRole::GroupIndex)) => {
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
            stack_id: "experts".to_string(),
            descriptor: payloads[0].weight(),
            experts: EXPERTS,
        };
        let entries: Vec<(String, Arc<PackedPayload>)> = payloads
            .iter()
            .enumerate()
            .map(|(expert, payload)| (format!("experts.{expert}"), Arc::clone(payload)))
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

    /// Every [`WeightFormat`] variant at least once, with the parameters that change the lowering. The match in
    /// [`variant_is_listed`] has no wildcard: a new variant does not compile until it is added there, and the
    /// reminder is to list it here.
    fn every_weight_format() -> Vec<WeightFormat> {
        let mut formats = vec![WeightFormat::F32, WeightFormat::F16, WeightFormat::Bf16];
        formats.extend([
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
        ]);
        for scale in [ScaleEncoding::F32, ScaleEncoding::Bf16, ScaleEncoding::E8m0] {
            formats.push(WeightFormat::E4m3PerChannel { scale });
            formats.push(WeightFormat::E4m3Block128 { scale });
        }
        formats
    }

    /// Whether the list above names `format`'s variant: an exhaustive match, so it breaks at a new variant.
    fn variant_is_listed(format: WeightFormat) -> bool {
        let variant = std::mem::discriminant(&format);
        every_weight_format()
            .iter()
            .any(|listed| std::mem::discriminant(listed) == variant)
    }

    /// The compile-time guard for [`variant_is_listed`]: no wildcard arm.
    fn variants_are_enumerated(format: WeightFormat) {
        match format {
            WeightFormat::F32
            | WeightFormat::F16
            | WeightFormat::Bf16
            | WeightFormat::Q4_0
            | WeightFormat::Q4_1
            | WeightFormat::Q5_0
            | WeightFormat::Q5_1
            | WeightFormat::Q8_0
            | WeightFormat::Q2_K
            | WeightFormat::Q3_K
            | WeightFormat::Q4_K
            | WeightFormat::Q5_K
            | WeightFormat::Q6_K
            | WeightFormat::Iq4_Nl
            | WeightFormat::Iq4_Xs
            | WeightFormat::Iq2_Xs
            | WeightFormat::Iq3_Xxs
            | WeightFormat::Iq3_S
            | WeightFormat::Mxfp4
            | WeightFormat::E4m3PerChannel { .. }
            | WeightFormat::E4m3Block128 { .. }
            | WeightFormat::E2m1Row32
            | WeightFormat::Gptq { .. }
            | WeightFormat::Awq { .. } => {}
        }
    }

    /// A format is lowered when it is a packed weight at the fixtures' shape and a row of `PACKED_LOWERING`
    /// admits its descriptor. The dense dtypes are not packed weights.
    fn is_lowered(format: WeightFormat) -> bool {
        PackedWeight::try_new(format, [OUT, K]).is_ok()
            && PACKED_LOWERING
                .iter()
                .any(|row| (row.admits)(&format.descriptor()))
    }

    /// SC-006 (card 1218): every packed format the lowering table admits has a scheme, so a device parity row on
    /// every backend that runs these fixtures; and a scheme names no format twice or one the table does not lower.
    /// Mutation: remove one `SCHEMES` entry; this goes red naming the format.
    #[test]
    fn the_schemes_are_every_format_the_lowering_table_admits() {
        for format in every_weight_format() {
            variants_are_enumerated(format);
            assert!(variant_is_listed(format));
            let has_scheme = SCHEMES.iter().any(|scheme| scheme.format == format);
            assert_eq!(
                has_scheme,
                is_lowered(format),
                "{format:?}: {}",
                if has_scheme {
                    "has a scheme but no PACKED_LOWERING row admits it"
                } else {
                    "PACKED_LOWERING admits it but no SCHEMES entry gives it a device parity row"
                }
            );
        }
        for (index, scheme) in SCHEMES.iter().enumerate() {
            assert!(
                !SCHEMES[..index]
                    .iter()
                    .any(|earlier| earlier.format == scheme.format),
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
