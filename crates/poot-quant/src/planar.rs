//! Descriptors of the planar (safetensors) quantized formats: one source tensor per operand role,
//! and the row concatenation those source tensors admit ([`PackedPayload::concat_rows`]).

use std::sync::Arc;

use crate::format::{
    Axis, Extent, FieldEncoding, FloatFormat, FormatDescriptor, GroupMap, LaneOrder, Major,
    OperandRole, Packing, PlanarLayout, PlanarOperand, ScaleEncoding, Storage, ValueMap,
    WeightFormat,
};
use crate::{PackedPayload, PackedWeight, PackedWeightError, SourceRole};

/// Logical rows per stored row of `operand`, when its stored rows follow `out` in whole blocks:
/// out-major, with a fixed row extent (times the values per word when it packs along `out`).
/// `None` for a K-major operand (GPTQ, AWQ) or a whole-axis grid.
fn row_block(operand: &PlanarOperand) -> Option<usize> {
    if operand.major != Major::OutMajor {
        return None;
    }
    let Extent::Values(rows) = operand.grid[0] else {
        return None;
    };
    match operand.packing {
        Some(Packing {
            axis: Axis::Out,
            values_per_word,
            ..
        }) => Some(rows * values_per_word),
        _ => Some(rows),
    }
}

impl PackedPayload {
    /// `parts` stacked along `out` in order (a fused `gate||up` expert from its `gate_proj` and
    /// `up_proj`), byte for byte and never through a decode. The parts must share one format and
    /// `K`. A block format stacks whole stored rows ([`Self::gather_rows`]). A planar format stacks
    /// each operand's source tensor, which is the stacked weight's source only when every operand
    /// stores its rows out-major and every part is a whole number of that operand's row blocks: an
    /// `E4m3Block128` part must have a multiple of 128 rows, or its last 128x128 scale block would
    /// cover rows of the next part. Both are refused with a typed error before any byte is copied.
    pub fn concat_rows(parts: &[&Self]) -> Result<Self, PackedWeightError> {
        let Some(first) = parts.first().map(|part| part.weight) else {
            return Err(PackedWeightError::RowConcatEmpty);
        };
        let format = first.format();
        if let Some(other) = parts
            .iter()
            .map(|part| part.weight)
            .find(|weight| weight.format() != format || weight.shape()[1] != first.shape()[1])
        {
            return Err(PackedWeightError::RowGatherParts { first, other });
        }
        let rows: usize = parts.iter().map(|part| part.weight.shape()[0]).sum();
        let Storage::Planar(layout) = format.descriptor().storage else {
            return Self::gather_rows(parts, &(0..rows).collect::<Vec<_>>());
        };
        for operand in layout.operands {
            let role = operand.role;
            let block_rows =
                row_block(operand).ok_or(PackedWeightError::RowConcatLayout { format, role })?;
            if let Some(part) = parts
                .iter()
                .find(|part| !part.weight.shape()[0].is_multiple_of(block_rows))
            {
                return Err(PackedWeightError::RowConcatUnaligned {
                    format,
                    role,
                    rows: part.weight.shape()[0],
                    block_rows,
                });
            }
        }
        let weight = PackedWeight::try_new(format, [rows, first.shape()[1]])?;
        let sources: Vec<(SourceRole, Arc<[u8]>)> = layout
            .operands
            .iter()
            .map(|operand| {
                let role = SourceRole::Planar(operand.role);
                let bytes: Vec<&[u8]> = parts.iter().map(|part| part.bytes(role)).collect();
                (role, Arc::from(bytes.concat()))
            })
            .collect();
        Self::try_new(weight, sources)
    }
}

const fn scale_float(scale: ScaleEncoding) -> FloatFormat {
    match scale {
        ScaleEncoding::F32 => FloatFormat::F32,
        ScaleEncoding::Bf16 => FloatFormat::Bf16,
        ScaleEncoding::E8m0 => FloatFormat::E8m0,
    }
}

/// An unpacked operand of one float per stored element, `[out, K]`-major.
const fn float_operand(role: OperandRole, float: FloatFormat, grid: [Extent; 2]) -> PlanarOperand {
    PlanarOperand {
        role,
        encoding: FieldEncoding::Float(float),
        bits: float.bits(),
        grid,
        packing: None,
        major: Major::OutMajor,
    }
}

const E4M3_CODES: PlanarOperand = float_operand(
    OperandRole::Codes,
    FloatFormat::E4m3Fn,
    [Extent::Values(1), Extent::Values(1)],
);

/// The E4M3 codes and a per-block scale of `grid` logical values.
macro_rules! e4m3 {
    ($format:expr, $scale:expr, $grid:expr) => {
        FormatDescriptor {
            format: $format,
            storage: Storage::Planar(PlanarLayout {
                operands: &[
                    E4M3_CODES,
                    float_operand(OperandRole::Scale, scale_float($scale), $grid),
                ],
            }),
            value: ValueMap::Float,
            min_sign: None,
        }
    };
}

/// Safetensors `F8_E4M3` weight `[out, K]` with a `[out, 1]` scale: `w = e4m3(q) * scale[o]`.
pub(crate) const fn e4m3_per_channel(scale: ScaleEncoding) -> FormatDescriptor {
    const PER_ROW: [Extent; 2] = [Extent::Values(1), Extent::Whole];
    const F32: FormatDescriptor = e4m3!(
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32
        },
        ScaleEncoding::F32,
        PER_ROW
    );
    const BF16: FormatDescriptor = e4m3!(
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::Bf16
        },
        ScaleEncoding::Bf16,
        PER_ROW
    );
    const E8M0: FormatDescriptor = e4m3!(
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::E8m0
        },
        ScaleEncoding::E8m0,
        PER_ROW
    );
    match scale {
        ScaleEncoding::F32 => F32,
        ScaleEncoding::Bf16 => BF16,
        ScaleEncoding::E8m0 => E8M0,
    }
}

/// Safetensors `F8_E4M3` weight `[out, K]` with a `[ceil(out/128), ceil(K/128)]` block scale
/// (DeepSeek-V3 style `weight_scale_inv`): `w = e4m3(q) * scale[o/128, k/128]`.
pub(crate) const fn e4m3_block128(scale: ScaleEncoding) -> FormatDescriptor {
    const BLOCK: [Extent; 2] = [Extent::Values(128), Extent::Values(128)];
    const F32: FormatDescriptor = e4m3!(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32
        },
        ScaleEncoding::F32,
        BLOCK
    );
    const BF16: FormatDescriptor = e4m3!(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16
        },
        ScaleEncoding::Bf16,
        BLOCK
    );
    const E8M0: FormatDescriptor = e4m3!(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0
        },
        ScaleEncoding::E8m0,
        BLOCK
    );
    match scale {
        ScaleEncoding::F32 => F32,
        ScaleEncoding::Bf16 => BF16,
        ScaleEncoding::E8m0 => E8M0,
    }
}

/// Safetensors E2M1 weight: two codes per byte along K (even K in the low nibble), stored
/// `[out, ceil(K/2)]`, and an E8M0 scale `[out, ceil(K/32)]`: `w = e2m1(q) * 2^(e - 127)`.
pub(crate) const E2M1_ROW32: FormatDescriptor = FormatDescriptor {
    format: WeightFormat::E2m1Row32,
    storage: Storage::Planar(PlanarLayout {
        operands: &[
            PlanarOperand {
                role: OperandRole::Codes,
                encoding: FieldEncoding::Float(FloatFormat::E2m1),
                bits: 4,
                grid: [Extent::Values(1), Extent::Values(1)],
                packing: Some(Packing {
                    values_per_word: 2,
                    word_bytes: 1,
                    axis: Axis::K,
                    lanes: LaneOrder::Natural,
                }),
                major: Major::OutMajor,
            },
            float_operand(
                OperandRole::Scale,
                FloatFormat::E8m0,
                [Extent::Values(1), Extent::Values(32)],
            ),
        ],
    }),
    value: ValueMap::Float,
    min_sign: None,
};

/// Eight 4-bit values per little-endian `i32` along `axis`.
const fn int4x8(axis: Axis, lanes: LaneOrder) -> Option<Packing> {
    Some(Packing {
        values_per_word: 8,
        word_bytes: 4,
        axis,
        lanes,
    })
}

/// One f16 scale per `[out, group]`, stored `[groups, out]`.
const GROUP_SCALE: PlanarOperand = PlanarOperand {
    role: OperandRole::Scale,
    encoding: FieldEncoding::Float(FloatFormat::F16),
    bits: 16,
    grid: [Extent::Values(1), Extent::Group],
    packing: None,
    major: Major::KMajor,
};

const GPTQ_CODES: PlanarOperand = PlanarOperand {
    role: OperandRole::Codes,
    encoding: FieldEncoding::Unsigned { offset: 0 },
    bits: 4,
    grid: [Extent::Values(1), Extent::Values(1)],
    packing: int4x8(Axis::K, LaneOrder::Natural),
    major: Major::KMajor,
};

const GPTQ_ZERO: PlanarOperand = PlanarOperand {
    role: OperandRole::Zero,
    encoding: FieldEncoding::Unsigned { offset: 1 },
    bits: 4,
    grid: [Extent::Values(1), Extent::Group],
    packing: int4x8(Axis::Out, LaneOrder::Natural),
    major: Major::KMajor,
};

/// GPTQ act-order `g_idx [K]` (i32): the group of each K index.
const GPTQ_GROUP_INDEX: PlanarOperand = PlanarOperand {
    role: OperandRole::GroupIndex,
    encoding: FieldEncoding::Signed,
    bits: 32,
    grid: [Extent::Whole, Extent::Values(1)],
    packing: None,
    major: Major::KMajor,
};

/// GPTQ `qweight [K/8, out]` (eight codes per word along K), `qzeros [groups, out/8]` (along out),
/// `scales [groups, out]`, and with act-order `g_idx [K]`: AutoGPTQ `w = scale * (q - (z + 1))`.
pub(crate) const fn gptq(groups: GroupMap) -> FormatDescriptor {
    let operands: &'static [PlanarOperand] = match groups {
        GroupMap::Contiguous { .. } => &[GPTQ_CODES, GPTQ_ZERO, GROUP_SCALE],
        GroupMap::Indexed { .. } => &[GPTQ_CODES, GPTQ_ZERO, GROUP_SCALE, GPTQ_GROUP_INDEX],
    };
    FormatDescriptor {
        format: WeightFormat::Gptq { groups },
        storage: Storage::Planar(PlanarLayout { operands }),
        value: ValueMap::Integer,
        min_sign: None,
    }
}

/// AWQ GEMM `qweight [K, out/8]` and `qzeros [groups, out/8]` (eight codes per word along out in
/// AutoAWQ's order), `scales [groups, out]`: `w = scale * (q - z)`.
const AWQ_OPERANDS: &[PlanarOperand] = &[
    PlanarOperand {
        role: OperandRole::Codes,
        encoding: FieldEncoding::Unsigned { offset: 0 },
        bits: 4,
        grid: [Extent::Values(1), Extent::Values(1)],
        packing: int4x8(Axis::Out, LaneOrder::Awq),
        major: Major::KMajor,
    },
    PlanarOperand {
        role: OperandRole::Zero,
        encoding: FieldEncoding::Unsigned { offset: 0 },
        bits: 4,
        grid: [Extent::Values(1), Extent::Group],
        packing: int4x8(Axis::Out, LaneOrder::Awq),
        major: Major::KMajor,
    },
    GROUP_SCALE,
];

pub(crate) const fn awq(group_size: std::num::NonZeroUsize) -> FormatDescriptor {
    FormatDescriptor {
        format: WeightFormat::Awq { group_size },
        storage: Storage::Planar(PlanarLayout {
            operands: AWQ_OPERANDS,
        }),
        value: ValueMap::Integer,
        min_sign: None,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    }

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn assert_planar(
        format: WeightFormat,
        shape: [usize; 2],
        sources: &[(OperandRole, Vec<u8>)],
        expected: &[([usize; 2], f32)],
    ) {
        let sources: Vec<(OperandRole, &[u8])> = sources
            .iter()
            .map(|(role, bytes)| (*role, bytes.as_slice()))
            .collect();
        let descriptor = format.descriptor();
        for &(coordinate, want) in expected {
            let got = descriptor
                .decode_planar_value(shape, &sources, coordinate)
                .unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{format:?} {coordinate:?}: {got:?} vs reference {want:?}"
            );
        }
    }

    /// An E4M3 weight whose byte at `[o, k]` is `(131o + 7k + 3) mod 256`, NaN bytes replaced by
    /// zero. Values from the OCP 8-bit floating point spec (E4M3FN: bias 7, subnormals `m * 2^-9`)
    /// times the scale.
    fn assert_e4m3(
        format: WeightFormat,
        shape: [usize; 2],
        scales: &str,
        expected: &[([usize; 2], f32)],
    ) {
        let weight = (0..shape[0])
            .flat_map(|o| (0..shape[1]).map(move |k| (o * 131 + k * 7 + 3) % 256))
            .map(|byte| match byte as u8 {
                0x7f | 0xff => 0,
                byte => byte,
            })
            .collect();
        assert_planar(
            format,
            shape,
            &[
                (OperandRole::Codes, weight),
                (OperandRole::Scale, hex(scales)),
            ],
            expected,
        );
    }

    #[test]
    fn e4m3_per_channel_f32_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::F32,
            },
            [3, 5],
            "0000003f0000e0bfa69b443b",
            &[
                ([0, 0], 0.0029296875),
                ([0, 1], 0.009765625),
                ([0, 2], 0.017578125),
                ([0, 3], 0.03125),
                ([0, 4], 0.05859375),
                ([1, 0], 0.020507812),
                ([1, 1], 0.044433594),
                ([1, 2], 0.08203125),
                ([1, 3], 0.15039062),
                ([1, 4], 0.2734375),
                ([2, 0], 5.2734376e-5),
                ([2, 1], 9.375e-5),
                ([2, 2], 0.00017578126),
                ([2, 3], 0.000328125),
                ([2, 4], 0.000609375),
            ],
        );
    }

    #[test]
    fn e4m3_block128_f32_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            [129, 130],
            "0000803e000000c08fc2753c0000e040",
            &[
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([0, 128], 0.01171875),
                ([0, 129], 0.0390625),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -8.789062e-5),
                ([128, 128], 0.041015625),
                ([128, 129], 0.13671875),
                ([64, 129], -1.0e1),
                ([5, 77], -0.1015625),
            ],
        );
    }

    #[test]
    fn e4m3_per_channel_bf16_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::Bf16,
            },
            [3, 5],
            "403f80bf003c",
            &[
                ([0, 0], 0.0043945312),
                ([0, 1], 0.0146484375),
                ([0, 2], 0.026367188),
                ([0, 3], 0.046875),
                ([0, 4], 0.087890625),
                ([1, 0], 0.01171875),
                ([1, 1], 0.025390625),
                ([1, 2], 0.046875),
                ([1, 3], 0.0859375),
                ([1, 4], 0.15625),
                ([2, 0], 0.0001373291),
                ([2, 1], 0.00024414062),
                ([2, 2], 0.00045776367),
                ([2, 3], 0.0008544922),
                ([2, 4], 0.0015869141),
            ],
        );
    }

    #[test]
    fn e4m3_block128_bf16_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            [129, 130],
            "803e00c0403fe040",
            &[
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([0, 128], 0.01171875),
                ([0, 129], 0.0390625),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -0.0043945312),
                ([128, 128], 0.041015625),
                ([128, 129], 0.13671875),
                ([64, 129], -1.0e1),
                ([5, 77], -0.1015625),
            ],
        );
    }

    #[test]
    fn e4m3_per_channel_e8m0_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::E8m0,
            },
            [3, 5],
            "7e8278",
            &[
                ([0, 0], 0.0029296875),
                ([0, 1], 0.009765625),
                ([0, 2], 0.017578125),
                ([0, 3], 0.03125),
                ([0, 4], 0.05859375),
                ([1, 0], -0.09375),
                ([1, 1], -0.203125),
                ([1, 2], -0.375),
                ([1, 3], -0.6875),
                ([1, 4], -1.25),
                ([2, 0], 0.0001373291),
                ([2, 1], 0.00024414062),
                ([2, 2], 0.00045776367),
                ([2, 3], 0.0008544922),
                ([2, 4], 0.0015869141),
            ],
        );
    }

    #[test]
    fn e4m3_block128_e8m0_scale_decodes_to_the_ocp_reference() {
        assert_e4m3(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            [129, 130],
            "7d807981",
            &[
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([0, 128], -0.01171875),
                ([0, 129], -0.0390625),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -9.1552734e-5),
                ([128, 128], 0.0234375),
                ([128, 129], 0.078125),
                ([64, 129], 1.0e1),
                ([5, 77], -0.1015625),
            ],
        );
    }

    /// Weight `[2, 35]` (odd K: each row's last byte has a zero high nibble), scales `[2, 2]`. Byte 0
    /// holds code 0 at K 0 and code 8 at K 1, which pins `-0.0`.
    #[test]
    fn e2m1_row32_decodes_to_the_ocp_reference() {
        assert_planar(
            WeightFormat::E2m1Row32,
            [2, 35],
            &[
                (
                    OperandRole::Codes,
                    hex("80837d1d0612d0e601c6d1bdf94dad156304c0dca8af7231d39ce31efb211b9916741c0a"),
                ),
                (OperandRole::Scale, hex("7f817a02")),
            ],
            &[
                ([0, 0], 0.0),
                ([0, 1], -0.0),
                ([0, 2], 1.5),
                ([0, 17], 0.0),
                ([0, 30], 3.0),
                ([0, 31], 0.5),
                ([0, 32], 6.0),
                ([0, 33], 16.0),
                ([0, 34], 8.0),
                ([1, 0], 0.0),
                ([1, 1], -0.0625),
                ([1, 2], -0.0625),
                ([1, 17], -0.125),
                ([1, 30], 0.0625),
                ([1, 31], 0.1875),
                ([1, 32], -4.7019774e-38),
                ([1, 33], 1.1754944e-38),
                ([1, 34], -2.3509887e-38),
            ],
        );
    }

    #[test]
    fn gptq_decodes_to_the_autogptq_reference() {
        assert_planar(
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size: nonzero(8) },
            },
            [8, 16],
            &gptq_sources(None),
            &[
                ([0, 0], -0.0),
                ([1, 1], 0.045013428),
                ([7, 7], 0.11248779),
                ([3, 8], 0.053634644),
                ([5, 9], -0.5361023),
                ([7, 15], 0.1462555),
                ([2, 12], -0.11251831),
                ([6, 4], 0.36743164),
            ],
        );
    }

    /// The same tensors with an act-order `g_idx`: K index `k` is in group `(5k + 1) mod 2`.
    #[test]
    fn gptq_act_order_decodes_to_the_autogptq_reference() {
        assert_planar(
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups: nonzero(2) },
            },
            [8, 16],
            &gptq_sources(Some(hex(
                "01000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000",
            ))),
            &[
                ([0, 0], 0.2999878),
                ([1, 1], 0.045013428),
                ([7, 7], 0.11248779),
                ([3, 8], 0.053634644),
                ([5, 9], -0.38989258),
                ([7, 15], -0.056243896),
                ([2, 12], -0.11251831),
                ([6, 4], 0.0),
            ],
        );
    }

    fn gptq_sources(g_idx: Option<Vec<u8>>) -> Vec<(OperandRole, Vec<u8>)> {
        let mut sources = vec![
            (
                OperandRole::Codes,
                hex(
                    "6187f803f0423141182c8d58e2912c82c2b7f4045a572fb18fe594e3b5e79e9806fada64fa70e9ce9a1e4d9d1e4970acab2968c3f084cc5f1de2b0ccc9cd39c1",
                ),
            ),
            (OperandRole::Zero, hex("c0ad6caaba893613")),
            (
                OperandRole::Scale,
                hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
            ),
        ];
        sources.extend(g_idx.map(|g_idx| (OperandRole::GroupIndex, g_idx)));
        sources
    }

    #[test]
    fn awq_decodes_to_the_autoawq_reference() {
        assert_planar(
            WeightFormat::Awq {
                group_size: nonzero(8),
            },
            [8, 16],
            &[
                (
                    OperandRole::Codes,
                    hex(
                        "81f2205a168cefb5c757127728eb94e5d844c1ef8f9f23928334218150e0849ba6dbea90901a1fcfea2990d41fe247c8da08099c4db67e3cd4c3ce1f96ccacc5",
                    ),
                ),
                (OperandRole::Zero, hex("d0acaca69a368b13")),
                (
                    OperandRole::Scale,
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
            ],
            &[
                ([0, 0], -0.02999878),
                ([1, 1], 0.06752014),
                ([7, 7], 0.056243896),
                ([3, 8], 0.06436157),
                ([5, 9], -0.58483887),
                ([7, 15], 0.16088104),
                ([2, 12], -0.15002441),
                ([6, 4], 0.3149414),
            ],
        );
    }

    const BLOCK128_F32: WeightFormat = WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::F32,
    };

    /// An `E4m3Block128` F32-scale payload `[out, k]`: code `[o, k]` is `(131o + 7k + seed) mod 256`
    /// (NaN codes replaced by zero), and scale block `b` is `seed + 0.25 * (b + 1)`, so every block of
    /// every part has its own scale.
    fn block128(out: usize, k: usize, seed: usize) -> PackedPayload {
        let codes: Vec<u8> = (0..out)
            .flat_map(|o| (0..k).map(move |k| ((o * 131 + k * 7 + seed) % 256) as u8))
            .map(|byte| if byte & 0x7f == 0x7f { 0 } else { byte })
            .collect();
        let blocks = out.div_ceil(128) * k.div_ceil(128);
        let scales: Vec<u8> = (0..blocks)
            .flat_map(|b| (seed as f32 + 0.25 * (b + 1) as f32).to_le_bytes())
            .collect();
        let weight = PackedWeight::try_new(BLOCK128_F32, [out, k]).unwrap();
        PackedPayload::try_new(
            weight,
            [
                (SourceRole::Planar(OperandRole::Codes), Arc::from(codes)),
                (SourceRole::Planar(OperandRole::Scale), Arc::from(scales)),
            ],
        )
        .unwrap()
    }

    fn decoded_rows(payload: &PackedPayload) -> Vec<Vec<u32>> {
        let [out, k] = payload.weight().shape();
        (0..out)
            .map(|row| {
                let mut values = vec![0.0f32; k];
                payload.decode_row(row, &mut values).unwrap();
                values.into_iter().map(f32::to_bits).collect()
            })
            .collect()
    }

    /// F545-2: a fused `gate||up` over two `E4m3Block128` payloads (256 and 128 rows, a partial
    /// K block) decodes, row for row and bit for bit, to `gate`'s rows then `up`'s rows through the
    /// one decoder. Mutation: concatenating the scale sources in the wrong order (or the codes of
    /// `up` before `gate`) decodes `gate`'s rows with `up`'s scales and turns this red.
    #[test]
    fn e4m3_block128_row_concat_decodes_to_its_parts_row_for_row() {
        let (gate, up) = (block128(256, 130, 3), block128(128, 130, 11));
        let fused = PackedPayload::concat_rows(&[&gate, &up]).unwrap();
        assert_eq!(fused.weight().shape(), [384, 130]);
        let mut expected = decoded_rows(&gate);
        expected.extend(decoded_rows(&up));
        for (row, (got, want)) in decoded_rows(&fused).iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "fused row {row}");
        }
    }

    /// F545-2: a part whose row count is not a multiple of 128 is refused, naming the scale operand
    /// whose block would straddle the boundary. Mutation: without the row check, two 100-row parts
    /// concatenate into a 200-row weight whose scale source happens to have the right length, and
    /// rows 100..128 silently decode with `gate`'s second scale block instead of `up`'s first.
    #[test]
    fn e4m3_block128_row_concat_refuses_a_part_that_is_not_whole_row_blocks() {
        let (gate, up) = (block128(100, 130, 3), block128(100, 130, 11));
        assert_eq!(
            PackedPayload::concat_rows(&[&gate, &up]),
            Err(PackedWeightError::RowConcatUnaligned {
                format: BLOCK128_F32,
                role: OperandRole::Scale,
                rows: 100,
                block_rows: 128,
            })
        );
    }

    #[test]
    fn row_concat_of_no_parts_is_refused() {
        assert_eq!(
            PackedPayload::concat_rows(&[]),
            Err(PackedWeightError::RowConcatEmpty)
        );
    }

    /// A K-major planar format (GPTQ) cannot be stacked along `out` by concatenating its sources.
    #[test]
    fn row_concat_refuses_a_k_major_format() {
        let format = WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nonzero(8) },
        };
        let gptq = PackedPayload::try_new(
            PackedWeight::try_new(format, [8, 16]).unwrap(),
            gptq_sources(None)
                .into_iter()
                .map(|(role, bytes)| (SourceRole::Planar(role), Arc::from(bytes))),
        )
        .unwrap();
        assert_eq!(
            PackedPayload::concat_rows(&[&gptq, &gptq]),
            Err(PackedWeightError::RowConcatLayout {
                format,
                role: OperandRole::Codes,
            })
        );
    }
}
