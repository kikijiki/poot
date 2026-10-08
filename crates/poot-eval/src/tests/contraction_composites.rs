//! Card 629: `PackedContraction`, `DenseContraction` and `DenseRowGather` are composites, so the oracle
//! has one definition of each: the primitive chain `decompose()` builds (ADR-0101 tier 1).
//!
//! - SC-001: each composite's evaluation equals the output the pre-card oracle published for the same
//!   fixture, recorded below as `to_bits` literals (captured from the deleted `evaluate_packed_contraction`
//!   and the BF16/F32/F16/E4M3 dense lanes before they were removed). The decomposition's `MatMul` sums over
//!   `K` in the order the deleted loop did, so the rows are tier 1 (bit-identical), not tier 2.
//! - SC-004: a real-shape packed contraction evaluates under the budget the CPU coherence tests state and
//!   is a typed `Budget` error under a smaller one.

use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::{
    Builder, Graph, OpKind, Operand, Slot, Storage, TensorType, Traced, ValueId,
    packed_source_constants,
};
use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, PackedWeight, SourceRole};
use poot_tensor::{DType, HostTensor};

use crate::{EvalBudget, EvalError, EvalOptions, Value, eval};

fn nonzero(value: usize) -> std::num::NonZeroUsize {
    std::num::NonZeroUsize::new(value).unwrap()
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// `poot_quant::planar`'s `assert_e4m3` test helper's weight formula: byte `(o*131 + k*7 + 3) % 256`, with
/// the two NaN E4M3 codes (`0x7f`, `0xff`) replaced by zero.
fn e4m3_weight_bytes(shape: [usize; 2]) -> Vec<u8> {
    (0..shape[0])
        .flat_map(|o| (0..shape[1]).map(move |k| (o * 131 + k * 7 + 3) % 256))
        .map(|byte| match byte as u8 {
            0x7f | 0xff => 0,
            byte => byte,
        })
        .collect()
}

/// One packed weight: its format, logical shape and carrier bytes by role.
struct Case {
    label: &'static str,
    format: WeightFormat,
    shape: [usize; 2],
    sources: Vec<(SourceRole, Vec<u8>)>,
}

/// A one-row GGUF block-format case of `values` logical values.
fn block_case(label: &'static str, format: WeightFormat, values: usize, block_hex: &str) -> Case {
    Case {
        label,
        format,
        shape: [1, values],
        sources: vec![(SourceRole::Blocks, hex(block_hex))],
    }
}

/// Every packed scheme: the GGUF block formats, the E4M3 and E2M1 cells, GPTQ and AWQ. The carrier bytes
/// are the fixtures the packed-dequant oracle's literal reference pins value by value.
fn cases() -> Vec<Case> {
    let block_cases = [
        block_case(
            "Q4_0",
            WeightFormat::Q4_0,
            32,
            "042397621649bd7821226d9d3dd30a06d54a",
        ),
        block_case(
            "Q4_1",
            WeightFormat::Q4_1,
            32,
            "0423b8b21d145901fadc73a8ffbf47d8a2a2457b",
        ),
        block_case(
            "Q5_0",
            WeightFormat::Q5_0,
            32,
            "0423ee8e46327f2c16fa536988e9d93c837043d73aeb",
        ),
        block_case(
            "Q5_1",
            WeightFormat::Q5_1,
            32,
            "0423b8b2490a0e50da8a62ee274c95b68d565f37472ea84e",
        ),
        block_case(
            "Q8_0",
            WeightFormat::Q8_0,
            32,
            "04230f9e167c0081016eefe8f408aad58096763bee663bf085c2c81e773207646c57",
        ),
        block_case(
            "Q2_K",
            WeightFormat::Q2_K,
            256,
            concat!(
                "12ff11bdd71c566ee39a03f9ebde5f9e3eeafd5ab4f4aa3511799ed220538bc8",
                "80deddf9d3d7baf93f42bdc17e7977f42729f4435ff7baca9e8328d9fa4e3184",
                "0698d2fd7862a239e04cf9ef504b2044451fc921",
            ),
        ),
        block_case(
            "Q3_K",
            WeightFormat::Q3_K,
            256,
            concat!(
                "513cd71855113ac60e98a370705c0c6cc5a6d6b9e91d0b38a023e0ae014fbda0",
                "4709e8fffcf2cff8d6c7c057de7e9ff123903238bbe563f56ec7b266bff9c474",
                "76287bad196dbc8999aa926c0937bcfeaf0036aa9d9d1ce704c9815872eabf63",
                "c17e9754da3ea7405f12bf6a451f",
            ),
        ),
        block_case(
            "Q4_K",
            WeightFormat::Q4_K,
            256,
            concat!(
                "451fc9219d1b645efb1d0f891f7c9a3575baad5711b8b0937499e433c92297b3",
                "75d815440095827c5450bb1709de3034def600c254b5c9d329f373a8c85aba7e",
                "e354b9a43dae4cf652843c4f9a82d24f47705a7fd09054d8eb19d3fc8eee07ed",
                "1bce13c0b61a602e2962b5d55aa5f7a312de08a01e726c42af359e0fc5d702ae",
                "dbd3a000d371cb5ca84a616b0201a2ac",
            ),
        ),
        block_case(
            "Q5_K",
            WeightFormat::Q5_K,
            256,
            concat!(
                "451fc921c0478de684332890a3cc170b9b6de18c07870e9fad5242c340f1d5f3",
                "571a647f5d2580560c84c9a3c6128d5cb3f309e886381fda2c4f2e5f419e6899",
                "988df40b2002fc09c4d8045f8a98e7cbaf877ca5af50ed49c394b683b8add5de",
                "aaa0e4a1690aef046454c8380c066eb53f0e94b260965b82c75beb9f01c732f2",
                "63394909d8fb84c14094526f8fed8465b7b9bad61847131139407708ebbfa851",
                "469289e2813e364cc0e13624b2ea49ec",
            ),
        ),
        block_case(
            "Q6_K",
            WeightFormat::Q6_K,
            256,
            concat!(
                "4293ba0b32a291d6db143d29f323d53c314f8968cf76ea446fa012c169ad04a3",
                "b59907cf091577286170d493bec0a383ebeac662b4b4425ea26c6c3e7dac9d55",
                "137c0107d87f14ec47749c4ed8f12633cb5508cd1f648aacc09e3857cb28a01e",
                "8a4758942e6d4a7412b3d5e0d295debababe2a918bf1468a85ea8249d7e8ff23",
                "54dd2c0b4c4883524b6155bdd59bad0b037011bec8b1dca798afd5061caae997",
                "29cb881d7ff762bd76afc455feca077095647b733ccbb555ec8a3f9a98d0ef75",
                "6013767cf76acf8d4403a28f52346e83451f",
            ),
        ),
        block_case(
            "Iq4_Nl",
            WeightFormat::Iq4_Nl,
            32,
            "04238760260d2c74c00575dfcf611de25bd8",
        ),
        block_case(
            "Iq4_Xs",
            WeightFormat::Iq4_Xs,
            256,
            concat!(
                "451f40ea9cb8b3a47f4a630b23f32b3622e927be470ce34ff63d9ec4db621023",
                "dfce18a8378c574185433b3128c79dbbefe90a0690397faf39fb09b28497e655",
                "6cf7c412c93121c3192ba819812b1786651383708c1e4f244e9ece2edc660fb4",
                "a25109ca0cf962c13e0018aead9babf9d1e4f9ac07d10b342fa353897ce27c29",
                "1357df59f780d916",
            ),
        ),
        block_case(
            "Mxfp4",
            WeightFormat::Mxfp4,
            32,
            "7c5c44b968cff4eb51b989d24e70cf4430",
        ),
    ];

    let e4m3_per_channel_shape = [3usize, 5];
    let e4m3_block128_shape = [129usize, 130];
    let e4m3_weight_per_channel = e4m3_weight_bytes(e4m3_per_channel_shape);
    let e4m3_weight_block128 = e4m3_weight_bytes(e4m3_block128_shape);

    let planar_cases = vec![
        Case {
            label: "E4m3PerChannel/F32",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::F32,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("0000003f0000e0bfa69b443b"),
                ),
            ],
        },
        Case {
            label: "E4m3PerChannel/Bf16",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::Bf16,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("403f80bf003c")),
            ],
        },
        Case {
            label: "E4m3PerChannel/E8m0",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::E8m0,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7e8278")),
            ],
        },
        Case {
            label: "E4m3Block128/F32",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    // `poot_quant::planar`'s own literal test pins scale index 1 to `-2.0` (negative on
                    // purpose, to prove the scale field's sign math), covered by `[0,128]`, `[0,129]`
                    // and `[64,129]`. `PackedPayload::try_new` additionally enforces the *production*
                    // content invariant that a registered E4M3 cell's scale is finite and positive
                    // (`validate_scale_encodings`), so this test flips that one block's scale to `2.0`
                    // (same magnitude, sign math already pinned by `poot_quant::planar`'s own test) and
                    // drops the three points it decoded, testing the other 8 through the full graph
                    // oracle this test drives.
                    hex("0000803e000000408fc2753c0000e040"),
                ),
            ],
        },
        Case {
            label: "E4m3Block128/Bf16",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    // See the F32 case above: same fix, same three points dropped.
                    hex("803e0040403fe040"),
                ),
            ],
        },
        Case {
            label: "E4m3Block128/E8m0",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7d807981")),
            ],
        },
        Case {
            label: "E2m1Row32",
            format: WeightFormat::E2m1Row32,
            shape: [2, 35],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex("80837d1d0612d0e601c6d1bdf94dad156304c0dca8af7231d39ce31efb211b9916741c0a"),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7f817a02")),
            ],
        },
        Case {
            label: "Gptq/Contiguous",
            format: WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size: nonzero(8) },
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "6187f803f0423141182c8d58e2912c82c2b7f4045a572fb18fe594e3b5e79e9806fada64fa70e9ce9a1e4d9d1e4970acab2968c3f084cc5f1de2b0ccc9cd39c1",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("c0ad6caaba893613"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
            ],
        },
        Case {
            label: "Gptq/Indexed",
            format: WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups: nonzero(2) },
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "6187f803f0423141182c8d58e2912c82c2b7f4045a572fb18fe594e3b5e79e9806fada64fa70e9ce9a1e4d9d1e4970acab2968c3f084cc5f1de2b0ccc9cd39c1",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("c0ad6caaba893613"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
                (
                    SourceRole::Planar(OperandRole::GroupIndex),
                    hex(
                        "01000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000",
                    ),
                ),
            ],
        },
        Case {
            label: "Awq",
            format: WeightFormat::Awq {
                group_size: nonzero(8),
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "81f2205a168cefb5c757127728eb94e5d844c1ef8f9f23928334218150e0849ba6dbea90901a1fcfea2990d41fe247c8da08099c4db67e3cd4c3ce1f96ccacc5",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("d0acaca69a368b13"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
            ],
        },
    ];

    block_cases.into_iter().chain(planar_cases).collect()
}

/// The activation `[rows, k]` every packed case contracts: distinct values per row and column, all exact
/// binary fractions.
fn activation(shape: &[usize]) -> HostTensor {
    let k = *shape.last().unwrap();
    let n: usize = shape.iter().product();
    HostTensor::f32(
        shape.to_vec(),
        (0..n)
            .map(|flat| ((flat / k * 31 + flat % k * 17) % 23) as f32 / 8.0 - 1.25)
            .collect(),
    )
}

/// A one-equation graph holding `PackedContraction { descriptor, blocks }` over an activation of
/// `activation_shape` and the weight's carrier constants, with the carriers bound to `owner`.
fn packed_contraction(
    owner: &Arc<PackedPayload>,
    blocks: usize,
    activation: HostTensor,
) -> (Graph, HashMap<ValueId, Value>) {
    let descriptor = owner.weight();
    let b = Builder::new();
    let x = b.slot_named(
        Slot::Activation,
        "x",
        TensorType::f32(activation.shape().to_vec()),
    );
    let mut plan = b.append_plan(0);
    let mut operands = vec![Operand::Value(x.id)];
    let mut inputs = HashMap::from([(x.id, Value::Host(activation))]);
    for (name, ty) in packed_source_constants("layer", descriptor) {
        let role = name.role();
        let input = plan
            .input(name, ty, Storage::Const)
            .expect("stage a packed carrier");
        inputs.insert(
            input.id,
            Value::Packed(PackedComponentRef::new(Arc::clone(owner), role)),
        );
        operands.push(Operand::Value(input.id));
    }
    let output = plan
        .equation(
            OpKind::PackedContraction {
                descriptor,
                rows: poot_graph_ir::ContractionRows::Blocks(blocks),
            },
            operands,
        )
        .expect("emit PackedContraction");
    plan.declare_result(output).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight the append");
    let id = b.commit_append(&mut prepared).expect("commit the append");
    (b.finish(Traced { id }), inputs)
}

fn payload_of(case: &Case) -> Arc<PackedPayload> {
    let descriptor = PackedWeight::try_new(case.format, case.shape)
        .unwrap_or_else(|error| panic!("{}: {error}", case.label));
    Arc::new(
        PackedPayload::try_new(
            descriptor,
            case.sources
                .iter()
                .map(|(role, bytes)| (*role, Arc::<[u8]>::from(bytes.as_slice()))),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.label)),
    )
}

fn host_bits(value: Value) -> Vec<u32> {
    let Value::Host(tensor) = value else {
        panic!("expected a dense output, got {value:?}");
    };
    tensor
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn evaluate(graph: &Graph, inputs: &HashMap<ValueId, Value>) -> Result<Vec<u32>, EvalError> {
    eval(graph, inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|evaluation| host_bits(evaluation.output))
}

/// A GGUF block-format payload of `shape` with deterministic pseudo-random blocks whose f16 factors are all
/// finite (Q8_0 `d`; Q4_K `d`, `dmin`: the top exponent bit cleared).
fn gguf_case(label: &'static str, format: WeightFormat, shape: [usize; 2]) -> Case {
    let poot_quant::format::Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("GGUF formats are block formats")
    };
    let descriptor = PackedWeight::try_new(format, shape).unwrap();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut bytes: Vec<u8> = (0..descriptor.source_bytes(SourceRole::Blocks))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    for block in bytes.chunks_mut(layout.bytes) {
        for role in [OperandRole::Scale, OperandRole::Min] {
            if let Some(field) = layout.field(role) {
                let byte = (field.field.pieces[0].layout.bit(0) / 8) as usize;
                block[byte + 1] &= !0x40;
            }
        }
    }
    Case {
        label,
        format,
        shape,
        sources: vec![(SourceRole::Blocks, bytes)],
    }
}

/// The E2M1 cell `[8, 35]` the block-diagonal fixtures contract: `blocks = 4` groups its 8 output rows in 4
/// blocks of 2.
fn e2m1_case() -> Case {
    let shape = [8, 35];
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, shape).unwrap();
    let row_bytes = descriptor.source_shape(SourceRole::Planar(OperandRole::Codes))[1];
    let mut codes = vec![0u8; descriptor.source_bytes(SourceRole::Planar(OperandRole::Codes))];
    for row in 0..shape[0] {
        for byte in 0..row_bytes {
            codes[row * row_bytes + byte] =
                (((2 * byte + row + 8) % 16) as u8) << 4 | ((2 * byte + row + 1) % 16) as u8;
        }
        // K is odd: the last byte holds one element, its high nibble unused.
        codes[(row + 1) * row_bytes - 1] &= 0x0f;
    }
    let scales: Vec<u8> = (0..descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)))
        .map(|index| [0x7f, 0x80, 0x81, 0x7e][index % 4])
        .collect();
    Case {
        label: "E2m1Row32 blocked",
        format: WeightFormat::E2m1Row32,
        shape,
        sources: vec![
            (SourceRole::Planar(OperandRole::Codes), codes),
            (SourceRole::Planar(OperandRole::Scale), scales),
        ],
    }
}

fn bf16_words(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

/// `W [3, 4]` in checkpoint order, every value exact in BF16.
const DENSE_WEIGHT: [f32; 12] = [
    1.0, -2.0, 0.5, 0.25, 3.0, -0.75, 1.5, -1.0, 0.125, 2.0, -4.0, 0.375,
];

/// The same `W` as binary16 words.
const DENSE_WEIGHT_F16: [u16; 12] = [
    0x3c00, 0xc000, 0x3800, 0x3400, 0x4200, 0xba00, 0x3e00, 0xbc00, 0x3000, 0x4000, 0xc400, 0x3600,
];

fn dense_contraction(weight: HostTensor) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![2, 4]));
    let w = b.constant(
        "w",
        TensorType::new(weight.shape().to_vec(), weight.dtype()),
    );
    let mut plan = b.append_plan(0);
    let output = plan
        .equation(
            OpKind::DenseContraction {
                weight: weight.dtype(),
            },
            vec![Operand::Value(x.id), Operand::Value(w.id)],
        )
        .expect("emit DenseContraction");
    plan.declare_result(output).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight the append");
    let id = b.commit_append(&mut prepared).expect("commit the append");
    let inputs = HashMap::from([
        (x.id, Value::Host(activation(&[2, 4]))),
        (w.id, Value::Host(weight)),
    ]);
    (b.finish(Traced { id }), inputs)
}

fn dense_row_gather(table: HostTensor, ids: Vec<i32>) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::new();
    let t = b.constant(
        "table",
        TensorType::new(table.shape().to_vec(), table.dtype()),
    );
    let index = b.slot_named(
        Slot::Token,
        "ids",
        TensorType::new(vec![ids.len()], DType::I32),
    );
    let mut plan = b.append_plan(0);
    let output = plan
        .equation(
            OpKind::DenseRowGather {
                source: table.dtype(),
            },
            vec![Operand::Value(t.id), Operand::Value(index.id)],
        )
        .expect("emit DenseRowGather");
    plan.declare_result(output).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight the append");
    let id = b.commit_append(&mut prepared).expect("commit the append");
    let inputs = HashMap::from([
        (t.id, Value::Host(table)),
        (index.id, Value::Host(HostTensor::i32(vec![ids.len()], ids))),
    ]);
    (b.finish(Traced { id }), inputs)
}

/// Every fixture of SC-001 as `(label, graph, inputs)`.
fn fixtures() -> Vec<(String, Graph, HashMap<ValueId, Value>)> {
    let mut rows = Vec::new();
    for case in cases() {
        let (graph, inputs) =
            packed_contraction(&payload_of(&case), 1, activation(&[2, case.shape[1]]));
        rows.push((case.label.to_string(), graph, inputs));
    }
    for case in [
        gguf_case("Q8_0 rows", WeightFormat::Q8_0, [3, 64]),
        gguf_case("Q4_K rows", WeightFormat::Q4_K, [3, 256]),
    ] {
        let (graph, inputs) =
            packed_contraction(&payload_of(&case), 1, activation(&[2, case.shape[1]]));
        rows.push((case.label.to_string(), graph, inputs));
    }
    let blocked = e2m1_case();
    let (graph, inputs) = packed_contraction(&payload_of(&blocked), 4, activation(&[4, 5, 35]));
    rows.push((blocked.label.to_string(), graph, inputs));
    let (graph, inputs) = dense_contraction(HostTensor::f32(vec![3, 4], DENSE_WEIGHT.to_vec()));
    rows.push(("DenseContraction F32".into(), graph, inputs));
    let (graph, inputs) =
        dense_contraction(HostTensor::bf16(vec![3, 4], bf16_words(&DENSE_WEIGHT)));
    rows.push(("DenseContraction BF16".into(), graph, inputs));
    let (graph, inputs) = dense_contraction(HostTensor::f16(vec![3, 4], DENSE_WEIGHT_F16.to_vec()));
    rows.push(("DenseContraction F16".into(), graph, inputs));
    let table = [
        1.0, -2.0, 0.5, 0.25, 3.0, -0.75, 1.5, -1.0, 0.125, 2.0, -4.0, 0.375, -0.5, 6.0, 0.0,
    ];
    let (graph, inputs) = dense_row_gather(
        HostTensor::bf16(vec![5, 3], bf16_words(&table)),
        vec![3, 0, 4, 3],
    );
    rows.push(("DenseRowGather BF16".into(), graph, inputs));
    let e4m3 = [
        0x38, 0x40, 0xb8, 0x01, 0x7e, 0xc0, 0x00, 0x81, 0x44, 0x30, 0xfe, 0x3c,
    ];
    let (graph, inputs) = dense_row_gather(
        crate::fp8::e4m3fn_tensor(vec![4, 3], e4m3.to_vec()).unwrap(),
        vec![2, 0, 2],
    );
    rows.push(("DenseRowGather E4M3".into(), graph, inputs));
    rows
}

/// The pre-card oracle's output bits per fixture, as `(output position, bits)`. Recorded from the deleted
/// `evaluate_packed_contraction` and the BF16/F32/F16 `DenseContraction` and BF16/E4M3 `DenseRowGather`
/// lanes on these exact fixtures, before they were removed.
const PRE_CARD_ORACLE: &[(&str, &[(usize, u32)])] = &[
    ("Q4_0", &[(0, 0x3ef9ee80), (1, 0xbebaca80)]),
    ("Q4_1", &[(0, 0xbde29600), (1, 0xbf559900)]),
    ("Q5_0", &[(0, 0xbf76dcc0), (1, 0x3e0a8f00)]),
    ("Q5_1", &[(0, 0xbfcbcd80), (1, 0xbdd9f600)]),
    ("Q8_0", &[(0, 0xc00513e0), (1, 0x40842558)]),
    ("Q2_K", &[(0, 0x3f1d8dc0), (1, 0xbef6da10)]),
    ("Q3_K", &[(0, 0xc08c731a), (1, 0x400a2d8a)]),
    ("Q4_K", &[(0, 0x41cfdb25), (1, 0x41ea17e6)]),
    ("Q5_K", &[(0, 0x429b1c7d), (1, 0x4265b6c8)]),
    ("Q6_K", &[(0, 0x43870477), (1, 0xc240f1cb)]),
    ("Iq4_Nl", &[(0, 0x405a0850), (1, 0x40207b80)]),
    ("Iq4_Xs", &[(0, 0x42efa402), (1, 0xc350d752)]),
    ("Mxfp4", &[(0, 0xbe200000), (1, 0xc0640000)]),
    (
        "E4m3PerChannel/F32",
        &[
            (0, 0x3d9a8000),
            (1, 0x3eade800),
            (2, 0x3a43d70a),
            (3, 0xba800000),
            (4, 0xbb600000),
            (5, 0xb775c280),
        ],
    ),
    (
        "E4m3PerChannel/Bf16",
        &[
            (0, 0x3de7c000),
            (1, 0x3e46c000),
            (2, 0x3aff0000),
            (3, 0xbac00000),
            (4, 0xbb000000),
            (5, 0xb8200000),
        ],
    ),
    (
        "E4m3PerChannel/E8m0",
        &[
            (0, 0x3d9a8000),
            (1, 0xbfc6c000),
            (2, 0x3aff0000),
            (3, 0xba800000),
            (4, 0x3c800000),
            (5, 0xb8200000),
        ],
    ),
    (
        "E4m3Block128/F32",
        &[
            (0, 0x435140c4),
            (1, 0x4140e0c0),
            (2, 0xc34ab660),
            (64, 0xc2a4ac88),
            (129, 0xc29e0d68),
            (193, 0x42ab4998),
            (256, 0xc29ccf78),
            (257, 0x409ce3e8),
        ],
    ),
    (
        "E4m3Block128/Bf16",
        &[
            (0, 0x435140c4),
            (1, 0x4140e0c0),
            (2, 0xc34ab660),
            (64, 0xc2a4ac88),
            (129, 0xc29e0d68),
            (193, 0x42ab4998),
            (256, 0xc29ccf78),
            (257, 0x436d1a1c),
        ],
    ),
    (
        "E4m3Block128/E8m0",
        &[
            (0, 0x435140c4),
            (1, 0x4140e0c0),
            (2, 0xc34ab660),
            (64, 0xc2a4ac88),
            (129, 0xc29e0d68),
            (193, 0x42ab4998),
            (256, 0xc29ccf78),
            (257, 0x40a0f568),
        ],
    ),
    (
        "E2m1Row32",
        &[
            (0, 0x40aa0000),
            (1, 0x3e9b0000),
            (2, 0xc1210000),
            (3, 0xbfa28000),
        ],
    ),
    (
        "Gptq/Contiguous",
        &[
            (0, 0xbeb66480),
            (1, 0x3dc9a900),
            (2, 0x3eb52d00),
            (4, 0x3eea3b00),
            (8, 0xbd75c000),
            (12, 0xbf17ac80),
            (14, 0x3f6fae80),
            (15, 0x3de37500),
        ],
    ),
    (
        "Gptq/Indexed",
        &[
            (0, 0xbc999800),
            (1, 0x3e71fe00),
            (2, 0x3f2b6280),
            (4, 0x3f251d00),
            (8, 0x3e8e1300),
            (12, 0xbeeffd80),
            (14, 0xbf880e20),
            (15, 0x3f2f3270),
        ],
    ),
    (
        "Awq",
        &[
            (0, 0xbed70800),
            (1, 0x3e048500),
            (2, 0x3eca5f00),
            (4, 0x3eff5980),
            (8, 0xbdb0a200),
            (12, 0xbf145060),
            (14, 0x3f7785c0),
            (15, 0x3dfdaa80),
        ],
    ),
    (
        "Q8_0 rows",
        &[
            (0, 0xc0827d48),
            (1, 0xc0e0a134),
            (2, 0x4237dd48),
            (3, 0x40fe6dc4),
            (4, 0x4133b7c0),
            (5, 0x427c0f6c),
        ],
    ),
    (
        "Q4_K rows",
        &[
            (0, 0xc4106415),
            (1, 0x4622a102),
            (2, 0x44f1b03c),
            (3, 0xc3ea4675),
            (4, 0x463d20a4),
            (5, 0x44e67284),
        ],
    ),
    (
        "E2m1Row32 blocked",
        &[
            (0, 0x414f0000),
            (1, 0x42847000),
            (2, 0x40dc0000),
            (10, 0x404c0000),
            (20, 0x40200000),
            (30, 0xbfd80000),
            (38, 0x413c0000),
            (39, 0x42054000),
        ],
    ),
    (
        "DenseContraction F32",
        &[
            (0, 0xc0460000),
            (1, 0xc0660000),
            (2, 0x3f5c0000),
            (3, 0x401a0000),
            (4, 0x3fa80000),
            (5, 0xc0cc8000),
        ],
    ),
    (
        "DenseContraction BF16",
        &[
            (0, 0xc0460000),
            (1, 0xc0660000),
            (2, 0x3f5c0000),
            (3, 0x401a0000),
            (4, 0x3fa80000),
            (5, 0xc0cc8000),
        ],
    ),
    (
        "DenseContraction F16",
        &[
            (0, 0xc0460000),
            (1, 0xc0660000),
            (2, 0x3f5c0000),
            (3, 0x401a0000),
            (4, 0x3fa80000),
            (5, 0xc0cc8000),
        ],
    ),
    (
        "DenseRowGather BF16",
        &[
            (0, 0x40000000),
            (1, 0xc0800000),
            (2, 0x3ec00000),
            (3, 0x3f800000),
            (6, 0xbf000000),
            (9, 0x40000000),
            (10, 0xc0800000),
            (11, 0x3ec00000),
        ],
    ),
    (
        "DenseRowGather E4M3",
        &[
            (0, 0x00000000),
            (1, 0xbb000000),
            (2, 0x40400000),
            (4, 0x40000000),
            (6, 0x00000000),
            (7, 0xbb000000),
            (8, 0x40400000),
        ],
    ),
];

/// SC-001. Mutation: swap the operands of the decomposition's `MatMul` (`MatMul(Transpose(w), activation)`
/// in `decompose.rs`): the builder refuses the shapes with a trace-time panic (`matmul contracting dim
/// mismatch`) at the `decompose` call, so the test goes red on the first fixture.
#[test]
fn composites_evaluate_to_the_pre_card_oracle_bits() {
    let fixtures = fixtures();
    assert_eq!(
        fixtures.len(),
        PRE_CARD_ORACLE.len(),
        "one literal row per fixture"
    );
    for (label, graph, inputs) in fixtures {
        let (_, expected) = PRE_CARD_ORACLE
            .iter()
            .find(|(name, _)| *name == label)
            .unwrap_or_else(|| panic!("{label}: no recorded oracle output"));
        let bits = evaluate(&graph, &inputs).unwrap_or_else(|error| panic!("{label}: {error}"));
        for &(position, want) in *expected {
            assert_eq!(
                bits[position],
                want,
                "{label} [{position}]: got {:?}, the pre-card oracle published {:?}",
                f32::from_bits(bits[position]),
                f32::from_bits(want)
            );
        }
    }
}

#[derive(Default)]
struct Allocations(Vec<usize>);

impl crate::observer::EvalObserver for Allocations {
    fn allocation(&mut self, _eqn: usize, _out: ValueId, _dtype: DType, bytes: usize) {
        self.0.push(bytes);
    }
}

/// An observer sees ONE allocation event per intermediate of a decomposition: the dequantized `[8, 35]`
/// weight (`PackedDequant`), its transposed copy and the `[2, 8]` product. The `PackedDequant` arm's own
/// charge is not a second event for the same allocation.
///
/// Mutation: remove the `charged_by_composite` early return in `EvalOptions::charge`; the dequantized
/// weight is reported twice and the event list has four entries.
#[test]
fn a_decomposition_reports_one_allocation_per_intermediate() {
    let case = e2m1_case();
    let (graph, inputs) = packed_contraction(&payload_of(&case), 1, activation(&[2, 35]));
    let mut seen = Allocations::default();
    eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).observer(&mut seen),
    )
    .unwrap();
    assert_eq!(seen.0, [8 * 35 * 4, 35 * 8 * 4, 2 * 8 * 4]);
}

/// POOT-1048: `GroupedDenseContraction` over a BF16 `[G * N, K]` weight evaluates, bit for bit, to the product
/// each group's own `[N, K]` row block makes with that group's activation rows, computed here by hand in f64
/// (every term is exact in f32, so the sum is order-free). Group `g` reads rows `g * N ..`, not the first block.
///
/// Mutation: in `decompose`, reshape the weight to `[N, groups, K]` (rows interleaved across groups); every
/// group but the first reads another group's rows and this row goes red.
#[test]
fn a_grouped_dense_contraction_equals_each_groups_own_block_product() {
    let (groups, m, n, k) = (3usize, 2usize, 4usize, 5usize);
    let weight: Vec<f32> = (0..groups * n * k)
        .map(|i| ((i * 7 + 3) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let x = activation(&[groups, m, k]);
    let b = Builder::new();
    let xs = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![groups, m, k]));
    let w = b.constant("w", TensorType::new(vec![groups * n, k], DType::BF16));
    let mut plan = b.append_plan(0);
    let output = plan
        .equation(
            OpKind::GroupedDenseContraction {
                weight: DType::BF16,
                groups,
            },
            vec![Operand::Value(xs.id), Operand::Value(w.id)],
        )
        .expect("emit GroupedDenseContraction");
    plan.declare_result(output).expect("declare the result");
    let mut prepared = b.preflight_append(plan).expect("preflight the append");
    let id = b.commit_append(&mut prepared).expect("commit the append");
    let graph = b.finish(Traced { id });
    let inputs = HashMap::from([
        (xs.id, Value::Host(x.clone())),
        (
            w.id,
            Value::Host(HostTensor::bf16(vec![groups * n, k], bf16_words(&weight))),
        ),
    ]);
    let got = evaluate(&graph, &inputs).unwrap();
    let activations = x.as_f32().expect("an F32 activation");
    let want: Vec<u32> = (0..groups)
        .flat_map(|g| (0..m).flat_map(move |row| (0..n).map(move |col| (g, row, col))))
        .map(|(g, row, col)| {
            let dot: f64 = (0..k)
                .map(|kk| {
                    f64::from(activations[(g * m + row) * k + kk])
                        * f64::from(weight[(g * n + col) * k + kk])
                })
                .sum();
            (dot as f32).to_bits()
        })
        .collect();
    assert_eq!(got, want);
}
