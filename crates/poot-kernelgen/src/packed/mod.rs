//! The descriptor-driven packed decode emitter (cards 542a/542b, dquant.md D1-D3, D9):
//! `poot-kernelgen` reads a `poot_quant::format::FormatDescriptor` at generation time and emits one
//! KIR decode per `(format, op)`. No kernel source here names a scheme; every per-format fact comes
//! from the descriptor's own constants. 542a covers the block-32 formats and `Schedule::Gemv`; 542b
//! adds the K-quant super-blocks' packed `SubScale`/`SubMin` sub-factors and `Schedule::RegisterTiled` for
//! `M > 1` (card 658: every `M > 1` block-diagonal split too, never `Schedule::Serial`, deleted).

mod decode_plan;
mod family;
// `crate::helpers::WeightLane` (Card 1007) reads a packed F16 dense weight through the same float
// decode every packed body uses, rather than restating the binary16 layout.
pub(crate) mod float_decode;
mod indexed;
mod kernel;
mod planar;
mod source;

#[cfg(test)]
mod identity_tests;

pub(crate) use indexed::packed_run_carried_registers;
pub use indexed::packed_run_held_registers;
pub(crate) use kernel::packed_contraction;
pub use kernel::{
    PackedKernelOp, PackedKernelSpec, PackedRunSharing, packed_gemv_body_cost, packed_kernel,
    packed_run_sharing, packed_value_bits,
};

#[cfg(test)]
mod float_decode_tests {
    //! SC-002 (card 542a): `emit_float_decode` matches `poot_quant::scalar::float_to_f32`'s `to_bits`
    //! for every admitted bit pattern (every one `FloatFormat::non_finite` does not name) of every
    //! `FloatFormat`, run through the KIR interpreter
    //! (tier 1, GPU-free).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;
    use poot_kernel_ir::{BasicBlock, Body, Operand, Rvalue, Terminator, Ty};
    use poot_quant::format::FloatFormat;
    use poot_quant::scalar::float_to_f32;

    use super::float_decode::emit_float_decode;
    use crate::emit::{Emit, copy, cu};
    use crate::helpers::{Alloc, elem, ld, local, slice_dtype, slice_f32};

    fn decode_body(format: FloatFormat) -> Body {
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
            ld(slice_f32(true), true),
        ]);
        let (input, output) = (local(1), local(2));
        let mut emit = Emit::new(&mut al);
        let zero = emit.let_(Ty::Usize, Rvalue::Use(cu(0)));
        let bits = emit.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(input, zero))));
        let decoded = emit_float_decode(&mut emit, format, bits);
        emit.assign(elem(output, zero), Rvalue::Use(copy(decoded)));
        let block = BasicBlock {
            statements: emit.take(),
            terminator: Terminator::Return,
        };
        Body::new("float_decode_probe", 2, al.locals, vec![block])
    }

    fn decode_at(body: &Body, bits: u32) -> u32 {
        let mut buffers = [Buffer::from_u32s(&[bits]), Buffer::from_f32s(&[0.0])];
        run(body, workgroups_covering(body, 1), &mut buffers).unwrap();
        buffers[1].to_f32s().unwrap()[0].to_bits()
    }

    /// The emitted decode of `format` equals the scalar decoder for every stored pattern of the
    /// admitted domain: every pattern `FloatFormat::non_finite` does not name (a payload never holds
    /// a non-finite one, so the decoders implement no branch for them). The domain is derived from
    /// `non_finite`, never restated here.
    fn assert_matches_scalar_on_the_admitted_domain(format: FloatFormat) {
        let body = decode_body(format);
        let mut admitted = 0u32;
        for bits in 0..1u32 << format.bits() {
            if format.non_finite().is_some_and(|n| n.matches(bits)) {
                continue;
            }
            admitted += 1;
            let want = float_to_f32(format, bits).to_bits();
            let got = decode_at(&body, bits);
            assert_eq!(
                got, want,
                "{format:?} bits {bits:#06x}: got {got:#010x} want {want:#010x}"
            );
        }
        assert!(admitted > 0, "{format:?}");
    }

    #[test]
    fn e8m0_matches_scalar_on_the_admitted_domain() {
        assert_matches_scalar_on_the_admitted_domain(FloatFormat::E8m0);
    }

    #[test]
    fn e2m1_matches_scalar_for_all_16_codes() {
        assert_matches_scalar_on_the_admitted_domain(FloatFormat::E2m1);
    }

    #[test]
    fn e4m3fn_matches_scalar_on_the_admitted_domain() {
        assert_matches_scalar_on_the_admitted_domain(FloatFormat::E4m3Fn);
    }

    #[test]
    fn f16_matches_scalar_on_the_admitted_domain() {
        assert_matches_scalar_on_the_admitted_domain(FloatFormat::F16);
    }

    #[test]
    fn bf16_matches_scalar_on_the_admitted_domain() {
        assert_matches_scalar_on_the_admitted_domain(FloatFormat::Bf16);
    }

    #[test]
    fn f32_is_an_identity_bitcast() {
        let body = decode_body(FloatFormat::F32);
        for bits in [
            0x0000_0000u32,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0x0000_0001,
        ] {
            assert_eq!(decode_at(&body, bits), bits);
        }
    }
}

#[cfg(test)]
mod float_shift_tests {
    //! POOT-1561 SC-f and SC-g (D3 of the POOT-1550 design): the shifted float decode and the planar run's shared
    //! scale, through the production run paths ([`DecodePlan::emit_run`], [`PlanarPlan::emit_run`] and the planar
    //! Gemv body), in the KIR interpreter.

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;
    use poot_kernel_ir::{
        BasicBlock, Body, Local, Operand, Place, Rvalue, Statement, Terminator, Ty,
    };
    use poot_quant::format::{
        FieldEncoding, OperandRole, PlanarRole, ScaleEncoding, Storage, ValueMap, WeightFormat,
    };
    use poot_quant::{PackedWeight, SourceRole};
    use poot_target::RoundingConversions;

    use super::decode_plan::DecodePlan;
    use super::fixtures::{self, GEMV_SCHEDULES};
    use super::planar::PlanarPlan;
    use super::planar_fixtures::{self, every_planar_format, pack_words};
    use super::source::Source;
    use crate::contraction::Schedule;
    use crate::emit::{Emit, copy, cu};
    use crate::helpers::{Alloc, elem, ld, local, slice_dtype, slice_f32};

    /// The body that stores the values `run` emits for the run at `(row 0, k0)` of a weight row of `k_total` values
    /// (`out_total` is one row) to `out[0..]`; `sources` word slices precede `out`.
    fn probe(
        sources: usize,
        (k0, k_total): (usize, usize),
        run: impl FnOnce(&mut Emit, &[Local], [Local; 4]) -> Vec<Local>,
    ) -> Body {
        let mut params = vec![ld(Ty::Unit, false)];
        params.extend((0..sources).map(|_| ld(slice_dtype(Ty::U32, false), false)));
        params.push(ld(slice_f32(true), true));
        let mut al = Alloc::new(params);
        let words: Vec<Local> = (0..sources as u32).map(|index| local(1 + index)).collect();
        let out = local(1 + sources as u32);
        let mut emit = Emit::new(&mut al);
        let [row, k0, k, out_total] =
            [0, k0, k_total, 1].map(|value| emit.let_(Ty::Usize, Rvalue::Use(cu(value))));
        let values = run(&mut emit, &words, [row, k0, k, out_total]);
        for (j, value) in values.iter().enumerate() {
            let at = emit.let_(Ty::Usize, Rvalue::Use(cu(j)));
            emit.assign(elem(out, at), Rvalue::Use(copy(*value)));
        }
        let block = BasicBlock {
            statements: emit.take(),
            terminator: Terminator::Return,
        };
        Body::new("shift_probe", 1 + sources as u32, al.locals, vec![block])
    }

    fn execute(body: &Body, sources: &[Vec<u32>], count: usize) -> Vec<f32> {
        let mut buffers: Vec<Buffer> = sources
            .iter()
            .map(|words| Buffer::from_u32s(words))
            .collect();
        buffers.push(Buffer::from_f32s(&vec![0.0; count]));
        run(body, workgroups_covering(body, 1), &mut buffers).expect("the probe runs");
        buffers.pop().unwrap().to_f32s().unwrap()
    }

    fn has_half_conversion(body: &Body) -> bool {
        body.blocks
            .iter()
            .flat_map(|block| &block.statements)
            .any(|statement| matches!(statement, Statement::Assign(_, Rvalue::FromLowBits { .. })))
    }

    /// One weight row of `k` values of the planar `format`: every admitted code of the float codes (in order, then
    /// cycling), and `scale` (one element's bytes) in every element of the scale plane.
    fn planar_row(
        format: WeightFormat,
        k: usize,
        scale: &[u8],
    ) -> (Vec<(SourceRole, Vec<u8>)>, Vec<f32>) {
        let shape = [1, k];
        let weight = PackedWeight::try_new(format, shape).unwrap();
        let descriptor = format.descriptor();
        let sources: Vec<(SourceRole, Vec<u8>)> = weight
            .sources()
            .into_iter()
            .map(|role| {
                let SourceRole::Planar(operand_role) = role else {
                    unreachable!("a planar source")
                };
                let len = descriptor
                    .planar_operand_bytes(shape, operand_role)
                    .unwrap();
                let bytes = match operand_role {
                    PlanarRole::Codes => {
                        let operand = descriptor.planar_operand(operand_role).unwrap();
                        let FieldEncoding::Float(float) = operand.encoding else {
                            unreachable!("a float planar format has float codes")
                        };
                        let admitted: Vec<u8> = (0..=255u32)
                            .filter(|&code| {
                                float.bits() < 8
                                    || !float.non_finite().is_some_and(|n| n.matches(code))
                            })
                            .map(|code| code as u8)
                            .collect();
                        (0..len).map(|i| admitted[i % admitted.len()]).collect()
                    }
                    PlanarRole::Scale => scale.iter().copied().cycle().take(len).collect(),
                    other => unreachable!("{other:?} is not an operand of a float planar format"),
                };
                (role, bytes)
            })
            .collect();
        let refs: Vec<(PlanarRole, &[u8])> = sources
            .iter()
            .map(|(role, bytes)| {
                let SourceRole::Planar(operand_role) = role else {
                    unreachable!()
                };
                (*operand_role, bytes.as_slice())
            })
            .collect();
        let expected = (0..k)
            .map(|kk| {
                descriptor
                    .decode_planar_value(shape, &refs, [0, kk])
                    .unwrap()
            })
            .collect();
        (sources, expected)
    }

    /// A shifted run decodes exactly what the scalar decoder does: every admitted code of every float format that
    /// embeds in binary16 (E4M3FN and E2M1), under a spread of scales, through a block run (`DecodePlan`) and a
    /// planar run (`PlanarPlan`, as a sliced four-code run and as sixteen codes one by one), equals `float_to_f32 *
    /// scale` bit for bit. The scales reach the largest power of two the fold keeps finite (2^119 for E4M3FN, 2^113 for
    /// E2M1), a binary32 subnormal, zero and a negative scale. The shifted body must really convert through binary16 and
    /// the exact one must not, so a decode that quietly stays exact does not pass.
    /// Mutation: fold `2^(bias - 14)` (`HalfEmbedding::rescale` one power of two short); every value is off by 2x.
    #[test]
    fn float_shift_decode_matches_scalar_on_the_admitted_domain() {
        let f32_scales: Vec<[u8; 4]> = [
            1.0f32,
            -0.0173,
            3.4e-5,
            f32::from_bits(1 << 8),
            f32::from_bits((127 + 119) << 23),
            0.0,
            448.0,
        ]
        .map(f32::to_le_bytes)
        .to_vec();
        let e8m0_scales: Vec<[u8; 1]> = [0x00u8, 0x01, 0x7f, 0x8e, 0xf0].map(|b| [b]).to_vec();
        let planar: Vec<(WeightFormat, Vec<Vec<u8>>)> = vec![
            (
                WeightFormat::E4m3PerChannel {
                    scale: ScaleEncoding::F32,
                },
                f32_scales.iter().map(|s| s.to_vec()).collect(),
            ),
            (
                WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::F32,
                },
                f32_scales.iter().map(|s| s.to_vec()).collect(),
            ),
            (
                WeightFormat::E2m1Row32,
                e8m0_scales.iter().map(|s| s.to_vec()).collect(),
            ),
        ];
        let k = 512;
        let mut compared = 0usize;
        for (format, scales) in planar {
            let plan = PlanarPlan::of(&format.descriptor()).unwrap();
            for scale in scales {
                let (sources, expected) = planar_row(format, k, &scale);
                let words = pack_words(&sources);
                for count in [4, 16] {
                    let body_of = |plan: &PlanarPlan, k0: usize| {
                        probe(
                            plan.source_count(),
                            (k0, k),
                            |e, words, [row, k0, k, out_total]| {
                                let src: Vec<Source> =
                                    words.iter().copied().map(Source::plain).collect();
                                plan.emit_run(e, &src, row, k0, count, k, out_total)
                            },
                        )
                    };
                    let shifted = PlanarPlan::of(&format.descriptor())
                        .unwrap()
                        .on(RoundingConversions::F16);
                    assert!(
                        has_half_conversion(&body_of(&shifted, 0)),
                        "{format:?}: the shifted decode converts through binary16"
                    );
                    assert!(
                        !has_half_conversion(&body_of(&plan, 0)),
                        "{format:?}: the exact decode does not"
                    );
                    for k0 in (0..k).step_by(count) {
                        let got = execute(&body_of(&shifted, k0), &words, count);
                        for (j, value) in got.iter().enumerate() {
                            assert_eq!(
                                value.to_bits(),
                                expected[k0 + j].to_bits(),
                                "{format:?} scale {scale:?} count {count} k {}: got {value} want {}",
                                k0 + j,
                                expected[k0 + j]
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        for format in poot_quant::test_support::formats::GGUF_FORMATS {
            let descriptor = format.descriptor();
            let Storage::Blocks(layout) = descriptor.storage else {
                continue;
            };
            let codes = layout.field(OperandRole::Codes).unwrap().field;
            let embeds =
                matches!(codes.encoding, FieldEncoding::Float(f) if f.half_embedding().is_some());
            if descriptor.value != ValueMap::Float || !embeds {
                continue;
            }
            for seed in 0..3u64 {
                let (k, mut bytes, _) = fixtures::fixture(format, 1, 16, seed);
                for block in bytes.chunks_mut(layout.bytes) {
                    // The fold needs `scale * 2^14` finite: an E8M0 scale up to 2^113.
                    let bit = layout.field(OperandRole::Scale).unwrap().field.pieces[0]
                        .layout
                        .bit(0);
                    block[(bit / 8) as usize] = block[(bit / 8) as usize].min(0xf0);
                }
                let mut expected = vec![0.0f32; k];
                poot_quant::plan::decode_blocks(format, &bytes, &mut expected).unwrap();
                let words = vec![fixtures::pack_words(&bytes)];
                for count in [4, 16] {
                    let plan = DecodePlan::of(&descriptor)
                        .unwrap()
                        .on(RoundingConversions::F16);
                    for k0 in (0..k).step_by(count) {
                        let body = probe(1, (k0, k), |e, words, [row, k0, k, _]| {
                            plan.emit_run(e, Source::plain(words[0]), row, k0, count, k)
                        });
                        assert!(has_half_conversion(&body), "{format:?}");
                        let got = execute(&body, &words, count);
                        for (j, value) in got.iter().enumerate() {
                            assert_eq!(
                                value.to_bits(),
                                expected[k0 + j].to_bits(),
                                "{format:?} seed {seed} count {count} k {}",
                                k0 + j
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        assert!(compared > 20_000, "{compared} values compared");
    }

    /// The loads from `source` a Gemv body issues: scalar and vector reads of that parameter.
    fn reads_of(body: &Body, source: Local) -> usize {
        body.blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Use(Operand::Copy(Place { local, .. })))
                        | Statement::Assign(_, Rvalue::VectorLoad { place: Place { local, .. }, .. })
                        if *local == source
                )
            })
            .count()
    }

    fn planar_gemv(format: WeightFormat, unroll: u32, conversions: RoundingConversions) -> Body {
        super::kernel::packed_contraction(
            "planar_gemv",
            format,
            Schedule::Gemv {
                width: 32,
                cols: 1,
                unroll,
                activation: crate::GemvActivation::Scalar,
                rows: crate::GemvRows::ONE,
                depth: crate::LoadDepth::ONE,
            },
            &[512, 4],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            conversions,
        )
        .unwrap()
    }

    /// POOT-1561 SC-g: a planar run inside one scale group reads the group's scale once, not once per element. The
    /// scale-plane loads of an FP8 block-128 or AWQ Gemv body (the unrolled run plus its one-element tail) do not
    /// grow with the run: a run of 8 issues as many as a run of 1. Act-order GPTQ names a group per element, so its
    /// run still decodes each element's own scale and its loads grow with the run.
    /// Mutation: decode each element's factors in `contraction_gemv_planar` again (`emit_factors` per `j`, the
    /// pre-D3 loop); an FP8 block-128 Gemv issues 4 scale loads at run 1 and 18 at run 8, and the row is red.
    #[test]
    fn a_planar_run_decodes_its_scale_once() {
        let scale_reads = |format: WeightFormat, unroll: u32| {
            let descriptor = format.descriptor();
            let Storage::Planar(layout) = descriptor.storage else {
                unreachable!("planar")
            };
            let scale = layout
                .operands
                .iter()
                .position(|operand| operand.role == PlanarRole::Scale)
                .expect("a Scale operand");
            // Params: the unit, the activation, then one word slice per source in `PackedWeight::sources()` order.
            let source = local(2 + scale as u32);
            reads_of(
                &planar_gemv(format, unroll, RoundingConversions::NONE),
                source,
            )
        };
        for format in [
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            planar_fixtures::awq(),
        ] {
            let per_run: Vec<usize> = [1, 8].map(|unroll| scale_reads(format, unroll)).to_vec();
            assert!(per_run[0] > 0, "{format:?}: the body reads its scale plane");
            assert_eq!(
                per_run[0], per_run[1],
                "{format:?}: a run of 8 reads the scale plane as often as a run of 1 (loads at runs 1 and 8)"
            );
        }
        let act_order = planar_fixtures::gptq_indexed(2);
        assert!(
            scale_reads(act_order, 8) > scale_reads(act_order, 1),
            "act-order GPTQ decodes each element's own scale"
        );
    }

    /// The output of the planar Gemv of `schedule` over the shared fixture of `out_rows` weight rows at `K = k`.
    fn planar_gemv_output(
        format: WeightFormat,
        schedule: Schedule,
        k: usize,
        out_rows: usize,
        conversions: RoundingConversions,
    ) -> Vec<f32> {
        let (sources, _) = planar_fixtures::fixture(format, out_rows, k, 11);
        let words = pack_words(&sources);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let body = super::kernel::packed_contraction(
            &format!("pin_{}", planar_fixtures::format_tag(format)),
            format,
            schedule,
            &[k as u32, out_rows as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            conversions,
        )
        .unwrap();
        let metadata = [k as u32, out_rows as u32, 1, 0, 0];
        let mut buffers: Vec<Buffer> = vec![Buffer::from_f32s(&activation)];
        buffers.extend(words.iter().map(|words| Buffer::from_u32s(words)));
        buffers.push(Buffer::from_u32s(&metadata));
        buffers.push(Buffer::from_f32s(&vec![0.0; out_rows]));
        let workgroups = workgroups_covering(&body, schedule.grid_threads(1, 1, out_rows));
        run(&body, workgroups, &mut buffers).expect("the planar Gemv runs");
        buffers.pop().unwrap().to_f32s().unwrap()
    }

    /// The f32 bits of the planar Gemv over the shared fixture, per format, schedule and `K`, digested.
    fn planar_gemv_digests(conversions: RoundingConversions) -> Vec<String> {
        let mut current = Vec::new();
        for format in every_planar_format() {
            for k in [17, 64] {
                for (index, schedule) in GEMV_SCHEDULES.into_iter().enumerate() {
                    let got = planar_gemv_output(format, schedule, k, 11, conversions);
                    let text: String = got.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
                    current.push(format!(
                        "{format:?} | gemv{index} k={k} | {:016x}",
                        super::identity_tests::digest(&text)
                    ));
                }
            }
        }
        current
    }

    /// POOT-1560 SC-e over the planar bodies (shared scale, sliced field, per-element codes): `rows` and `depth` change
    /// no bit of a row's sum, at `K` 17 (a ragged tail) and 64, over 12 weight rows (a multiple of every row count).
    /// Mutation: in `gemv_body` run the products of rows after the first in reverse order of the run (the `MUTATION`
    /// line); every row count above one goes red for the planar formats too.
    #[test]
    fn rows_and_depth_do_not_change_a_planar_rows_sum() {
        for format in every_planar_format() {
            for k in [17, 64] {
                for launch in [GEMV_SCHEDULES[1], GEMV_SCHEDULES[3]] {
                    let want = planar_gemv_output(format, launch, k, 12, RoundingConversions::NONE);
                    for rows in 1..=4 {
                        for depth in 1..=3 {
                            let schedule = fixtures::with_rows_and_depth(launch, rows, depth);
                            let got = planar_gemv_output(
                                format,
                                schedule,
                                k,
                                12,
                                RoundingConversions::NONE,
                            );
                            assert_eq!(
                                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "{format:?} {launch:?} k={k} rows={rows} depth={depth}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// POOT-1561 SC-g, the output side: sharing a planar run's scale changes no bit of the Gemv's output. The f32
    /// bits of every planar format under every `GEMV_SCHEDULES` entry at `K` 17 and 64 are the ones recorded from the
    /// per-element-factor body before this change.
    /// Mutation: take the scale of `k0 + 1` for the run's shared factors; an FP8 or AWQ row changes.
    #[test]
    fn a_planar_gemv_is_bit_identical_after_sharing_its_scale() {
        let recorded = include_str!("../../tests/data/packed_planar_gemv_output_bits.txt");
        let current = planar_gemv_digests(RoundingConversions::NONE);
        let expected: Vec<&str> = recorded.lines().collect();
        let changed: Vec<String> = current
            .iter()
            .zip(&expected)
            .filter(|(now, was)| now.as_str() != **was)
            .map(|(now, was)| format!("recorded {was}\n     now {now}"))
            .collect();
        assert!(
            changed.is_empty() && current.len() == expected.len(),
            "{} of {} planar Gemv outputs changed ({} recorded):\n{}\n--- full table ---\n{}",
            changed.len(),
            current.len(),
            expected.len(),
            changed.join("\n"),
            current.join("\n")
        );
    }
}

/// Shared fixtures for every packed-kernel device/interpreter test below: the seven block-32
/// formats, a deterministic admitted-domain byte generator, and the D9 word transport. One
/// implementation so every op's test compares against the identical oracle input shape.
#[cfg(test)]
mod fixtures {
    use poot_quant::format::{BlockLayout, OperandRole, Storage, ValueMap, WeightFormat};
    use poot_quant::test_support::formats::GGUF_FORMATS;

    use crate::contraction::{RunLength, Schedule, TileSize};
    use crate::microtile::{LaneRows, MicroCols};

    /// A codegen-safe (LLVM-identifier-safe) name fragment for a schedule: `{schedule:?}` on
    /// `Schedule::Gemv { .. }` contains `{`, `}` and a space, which breaks an AMDGCN/NVPTX global
    /// or symbol name built from it (card 542a ROCm device rows).
    pub(super) fn schedule_tag(schedule: Schedule) -> String {
        match schedule {
            Schedule::Gemv {
                width,
                cols,
                unroll,
                activation,
                rows,
                depth,
            } => {
                let read = match activation {
                    crate::GemvActivation::Scalar => "",
                    crate::GemvActivation::Vec4 => "v4",
                };
                let rows = if rows.get() > 1 {
                    format!("r{rows:?}")
                } else {
                    String::new()
                };
                let depth = if depth.get() > 1 {
                    format!("d{depth:?}")
                } else {
                    String::new()
                };
                format!("gemv{width}c{cols}u{unroll}{read}{rows}{depth}")
            }
            Schedule::RegisterTiled {
                lanes: TileSize(lanes),
                rows,
                cols,
                run,
            } => format!(
                "register_tiled{lanes}x{}x{}r{}",
                rows.get(),
                cols.get(),
                run.get()
            ),
            Schedule::FragmentTiled {
                waves_m,
                waves_n,
                tile,
                run,
                depth,
            } => format!(
                "fragment_tiled{}x{}x{}x{}r{}d{depth}",
                waves_m.get(),
                waves_n.get(),
                tile.rows,
                tile.cols,
                run.get()
            ),
        }
    }

    /// A [`Schedule::RegisterTiled`] of `lanes x lanes` lanes of two rows by four columns each (small tiles, so a few rows span
    /// several) over a weight whose rows hold `k` values: the longest run of consecutive `K` values that divides `k`, as the planner picks it. Block formats' `k` is a
    /// whole number of blocks, so any block width stands for it.
    pub(super) fn register_tiled(lanes: u32, k: usize) -> Schedule {
        Schedule::RegisterTiled {
            lanes: TileSize::new(lanes).unwrap(),
            rows: LaneRows(2),
            cols: MicroCols::new(4),
            run: RunLength::dividing(k),
        }
    }

    /// The Gemv launch shapes every Gemv row runs (card 653): one column per workgroup, several
    /// columns with a four-element run, a three-element run that divides neither a block nor the
    /// planar fixtures' `K = 17`, so the element-by-element tail runs, and a 32-element run that
    /// spans two 16-element K-quant sub-blocks, so a run may share its factors only where
    /// `DecodePlan::factor_run` allows. Every Gemv row's `out_per_block` (11 or 5) is not a
    /// multiple of 2, 4 or 8, so the last workgroup has columns past the output.
    pub(super) const GEMV_SCHEDULES: [Schedule; 4] = [
        Schedule::Gemv {
            width: 32,
            cols: 1,
            unroll: 1,
            activation: crate::GemvActivation::Scalar,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        },
        Schedule::Gemv {
            width: 32,
            cols: 4,
            unroll: 4,
            activation: crate::GemvActivation::Scalar,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        },
        Schedule::Gemv {
            width: 64,
            cols: 8,
            unroll: 3,
            activation: crate::GemvActivation::Scalar,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        },
        Schedule::Gemv {
            width: 32,
            cols: 2,
            unroll: 32,
            activation: crate::GemvActivation::Scalar,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        },
    ];

    /// `schedule`, a Gemv, with `rows` adjacent output columns per lane and `depth` trips of loads in flight.
    pub(super) fn with_rows_and_depth(schedule: Schedule, rows: u32, depth: u32) -> Schedule {
        let Schedule::Gemv {
            width,
            cols,
            unroll,
            activation,
            ..
        } = schedule
        else {
            panic!("{schedule:?} is not a Gemv");
        };
        Schedule::Gemv {
            width,
            cols,
            unroll,
            activation,
            rows: crate::GemvRows::new(rows).unwrap(),
            depth: crate::LoadDepth::new(depth).unwrap(),
        }
    }

    /// The geometries a device row runs a Gemv at beyond one row and one trip: two rows, two trips in flight, and the
    /// deepest of both. The formats are a block-32 one and a K-quant with sub-block factors and a lattice, so the
    /// rotated state carries windows, factors and a constant table.
    pub(super) const DEVICE_ROWS_AND_DEPTH: [(u32, u32); 3] = [(2, 1), (1, 3), (4, 3)];
    pub(super) fn device_rows_and_depth_formats() -> Vec<WeightFormat> {
        vec![
            block32()[0],
            kquant()[0],
            kquant()[kquant().len() - 1],
            lattice()[0],
        ]
    }

    /// The block formats of 32 values (`Q4_0` to `Q8_0`, `IQ4_NL`, MXFP4), from the shared format list.
    pub(super) fn block32() -> Vec<WeightFormat> {
        gguf_blocks_of(32)
    }

    /// The 256-value super-block formats (`blocks::QK_K`): the five K-quants, `IQ4_XS` and the three sign
    /// lattices, whose decode reads packed 6-bit `SubScale`/`SubMin` sub-factors, or a constant table and a
    /// `Signs` field. From the shared format list.
    pub(super) fn kquant() -> Vec<WeightFormat> {
        gguf_blocks_of(256)
    }

    /// The lattice quants with a sign pattern, which decode through a constant grid table and a `Signs` field.
    pub(super) fn lattice() -> Vec<WeightFormat> {
        kquant()
            .into_iter()
            .filter(|format| {
                matches!(
                    format.descriptor().value,
                    ValueMap::Lattice(lattice) if lattice.signs.is_some()
                )
            })
            .collect()
    }

    /// The GGUF block formats with a piecewise layout: a field read through a layout of several segments, or a
    /// lattice whose lane map has several. From the shared format list, by what the descriptor says.
    pub(super) fn piecewise() -> Vec<WeightFormat> {
        GGUF_FORMATS
            .into_iter()
            .filter(|format| {
                let descriptor = format.descriptor();
                let in_layout = matches!(descriptor.storage, Storage::Blocks(layout)
                    if layout.fields.iter().flat_map(|field| field.field.pieces).any(|piece| piece.layout.segments.len() > 1));
                let in_lane_map = matches!(descriptor.value, ValueMap::Lattice(lattice) if lattice.lane.segments.len() > 1);
                in_layout || in_lane_map
            })
            .collect()
    }

    /// The Gemv schedules a piecewise format's runs are checked under: runs of 4, which every segment moves alike,
    /// and of 8 and 16, which they do not (`TQ1_0`'s last stretch has four bytes to a row where the others have 32 and
    /// 16).
    pub(super) fn piecewise_run_schedules() -> [Schedule; 3] {
        [4, 8, 16].map(|unroll| Schedule::Gemv {
            width: 32,
            cols: 2,
            unroll,
            activation: crate::GemvActivation::Scalar,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        })
    }

    /// The GGUF block formats of `values` values per block, in ggml order.
    fn gguf_blocks_of(values: usize) -> Vec<WeightFormat> {
        GGUF_FORMATS
            .into_iter()
            .filter(|format| {
                matches!(format.descriptor().storage, Storage::Blocks(layout) if layout.values == values)
            })
            .collect()
    }

    /// xorshift64 bytes, exactly `plan.rs`'s own fixture generator (card 539a): deterministic and
    /// covers every bit pattern of every field.
    pub(super) fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    /// Native source bytes as zero-filled `u32` words plus one zero guard word (D9): the transport
    /// every packed kernel reads, whatever the byte length.
    pub(super) fn pack_words(bytes: &[u8]) -> Vec<u32> {
        let mut words: Vec<u32> = bytes
            .chunks(4)
            .map(|chunk| {
                let mut word_bytes = [0u8; 4];
                word_bytes[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(word_bytes)
            })
            .collect();
        words.push(0);
        words
    }

    /// A `[out_rows, blocks_per_row*values]` fixture of admitted-domain random bytes, plus the
    /// `decode_blocks` reference for it: shared by the interpreter (tier 1) and device (tier 1/2)
    /// rows, so both compare against the identical oracle input.
    pub(super) fn fixture(
        format: WeightFormat,
        out_rows: usize,
        blocks_per_row: usize,
        seed: u64,
    ) -> (usize, Vec<u8>, Vec<f32>) {
        let Storage::Blocks(layout) = format.descriptor().storage else {
            unreachable!()
        };
        let k = layout.values * blocks_per_row;
        let mut bytes = random_bytes(
            out_rows * blocks_per_row * layout.bytes,
            0x9e37_79b9_7f4a_7c15u64.wrapping_add(seed),
        );
        for block in bytes.chunks_mut(layout.bytes) {
            poot_quant::test_support::packed::sanitize_block_factors(layout, block);
        }
        let mut expected = vec![0.0f32; out_rows * k];
        poot_quant::plan::decode_blocks(format, &bytes, &mut expected)
            .unwrap_or_else(|error| panic!("{format:?}: fixture must decode: {error}"));
        (k, bytes, expected)
    }

    /// Write `value` into `role`'s field of `block` as read for in-block element `index`: the field's
    /// pieces take the value's bits low first (`format::Field`).
    fn write_field(
        layout: BlockLayout,
        block: &mut [u8],
        role: OperandRole,
        index: usize,
        mut value: u32,
    ) {
        let field = layout.field(role).expect("the format has this field");
        for piece in field.field.pieces {
            let start = piece.layout.bit(index as u32) as usize;
            for bit in 0..piece.width as usize {
                let (byte, shift) = ((start + bit) / 8, (start + bit) % 8);
                block[byte] =
                    (block[byte] & !(1 << shift)) | ((((value >> bit) & 1) as u8) << shift);
            }
            value >>= piece.width;
        }
    }

    /// Blocks of a lattice format in which every grid point occurs, every one of the 128 stored sign
    /// patterns occurs and every scale nibble occurs, so a wrong grid word, sign bit or scale in a decode
    /// changes some element: `(rows, K, bytes, decoded)`, one block per row, `decoded` from the CPU oracle
    /// (`decode_blocks`). Grid point `p` sits at consecutive in-block groups across rows, the sign
    /// pattern of eight-value group `g` is `g % 128`, and the scale nibble of each scale run is its index
    /// modulo 16 (16-value runs for `IQ2_XS`, 32-value runs for `IQ3_XXS`). `d` is 1.0.
    pub(super) fn every_grid_point(format: WeightFormat) -> (usize, usize, Vec<u8>, Vec<f32>) {
        let descriptor = format.descriptor();
        let (Storage::Blocks(layout), ValueMap::Lattice(grid)) =
            (descriptor.storage, descriptor.value)
        else {
            panic!("{format:?} is not a lattice format");
        };
        let rows = grid.points() * grid.lanes / layout.values;
        let scale_run = if grid.lanes == 8 { 16 } else { 32 };
        let mut bytes = vec![0u8; rows * layout.bytes];
        for (row, block) in bytes.chunks_mut(layout.bytes).enumerate() {
            write_field(layout, block, OperandRole::Scale, 0, 0x3c00);
            for index in 0..layout.values {
                let element = row * layout.values + index;
                let fields = [
                    (OperandRole::Codes, element / grid.lanes),
                    (OperandRole::Signs, element / 8 % 128),
                    (OperandRole::SubScale, element / scale_run % 16),
                ];
                for (role, value) in fields {
                    write_field(layout, block, role, index, value as u32);
                }
            }
        }
        let mut decoded = vec![0.0f32; rows * layout.values];
        poot_quant::plan::decode_blocks(format, &bytes, &mut decoded)
            .unwrap_or_else(|error| panic!("{format:?}: the grid fixture must decode: {error}"));
        (rows, layout.values, bytes, decoded)
    }

    /// One `RowGather` case: weight bytes, the oracle's decode of them, and the rows to gather.
    pub(super) struct GatherCase {
        pub(super) format: WeightFormat,
        pub(super) label: &'static str,
        pub(super) k: usize,
        pub(super) bytes: Vec<u8>,
        pub(super) ids: Vec<u32>,
        /// The oracle's rows `ids` select, concatenated.
        pub(super) expected: Vec<f32>,
    }

    /// Every [`kquant`] format over random blocks (rows `[3, 0, 4, 3]` of five), and every [`lattice`]
    /// format over [`every_grid_point`] blocks (all rows, reversed).
    pub(super) fn kquant_gather_cases() -> Vec<GatherCase> {
        let select = |decoded: &[f32], k: usize, ids: &[u32]| -> Vec<f32> {
            ids.iter()
                .flat_map(|&id| {
                    decoded[id as usize * k..(id as usize + 1) * k]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let mut cases = Vec::new();
        for (seed, format) in kquant().into_iter().enumerate() {
            let (k, bytes, decoded) = fixture(format, 5, 2, seed as u64);
            let ids = vec![3, 0, 4, 3];
            let expected = select(&decoded, k, &ids);
            cases.push(GatherCase {
                format,
                label: "random blocks",
                k,
                bytes,
                ids,
                expected,
            });
        }
        for format in lattice() {
            let (rows, k, bytes, decoded) = every_grid_point(format);
            let ids: Vec<u32> = (0..rows as u32).rev().collect();
            let expected = select(&decoded, k, &ids);
            cases.push(GatherCase {
                format,
                label: "every grid point",
                k,
                bytes,
                ids,
                expected,
            });
        }
        cases
    }

    /// `activation[rows,K] @ decode(weight)^T` in plain f32, `weight` decoded to `[out_rows,K]` by
    /// `decoded` (row-major): the reference every `Contraction` schedule/blocks combination compares
    /// against. `out_per_block`/`m_per_block` mirror the kernel's own metadata (`blocks == out_rows /
    /// out_per_block`).
    pub(super) fn reference_contraction(
        activation: &[f32],
        decoded: &[f32],
        k: usize,
        out_per_block: usize,
        m_per_block: usize,
        rows: usize,
    ) -> Vec<f32> {
        (0..rows * out_per_block)
            .map(|i| {
                let row_idx = i / out_per_block;
                let col = i % out_per_block;
                let b = row_idx / m_per_block;
                let weight_row = b * out_per_block + col;
                (0..k)
                    .map(|kk| activation[row_idx * k + kk] * decoded[weight_row * k + kk])
                    .sum()
            })
            .collect()
    }

    /// A `Contraction`-shaped fixture: `blocks * out_per_block` weight rows, `blocks * m_per_block`
    /// activation rows, K = `values * 2`, deterministic activation, plus the reference contraction.
    pub(super) fn contraction_fixture(
        format: WeightFormat,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) -> (usize, Vec<u8>, Vec<f32>, Vec<f32>) {
        let blocks_per_row = 2;
        let out_rows = blocks * out_per_block;
        let (k, bytes, decoded) = fixture(format, out_rows, blocks_per_row, 3);
        let rows = blocks * m_per_block;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..rows * k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected =
            reference_contraction(&activation, &decoded, k, out_per_block, m_per_block, rows);
        (k, bytes, activation, expected)
    }

    /// Tier-2 tolerance (ADR-0101): a relative epsilon generous enough for a plain sequential-sum
    /// reference to disagree with the generator's own accumulation order, tight enough to catch a
    /// wrong table row, index or decode.
    pub(super) fn assert_close(format: WeightFormat, what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{format:?} {what}: length");
        for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-3 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "{format:?} {what} element {index}: got {g} want {w} (tol {tol})"
            );
        }
    }
}

/// Shared fixtures for card 542c's planar-storage device/interpreter tests: the nine planar formats
/// (six E4M3 scale encodings, E2M1 row-32, GPTQ contiguous and act-order, AWQ), an admitted-domain
/// per-operand byte generator, and the `decode_planar_value` oracle - the planar counterpart of
/// `fixtures` above, which the block front end's tests use.
#[cfg(test)]
mod planar_fixtures {
    use std::num::NonZeroUsize;

    use poot_quant::format::{
        FieldEncoding, FloatFormat, GroupMap, PlanarRole, ScaleEncoding, WeightFormat,
    };
    use poot_quant::test_support::formats::SAFETENSORS_FORMATS;
    use poot_quant::{PackedWeight, SourceRole};

    use crate::StagedTerms;
    use crate::contraction::{RegisterTiling, Schedule};
    use crate::helpers::WeightLayout;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    pub(super) fn gptq_contiguous() -> WeightFormat {
        WeightFormat::Gptq {
            groups: GroupMap::Contiguous { size: nz(8) },
        }
    }

    /// `groups` groups over a K this fixture's `k` is a whole multiple of.
    pub(super) fn gptq_indexed(groups: usize) -> WeightFormat {
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nz(groups) },
        }
    }

    pub(super) fn awq() -> WeightFormat {
        WeightFormat::Awq { group_size: nz(8) }
    }

    /// Every planar format SC-001 (card 542c) names, from the shared format list: the six E4M3 scale encodings,
    /// E2M1 row-32, GPTQ contiguous and act-order, and AWQ, with the small groups these fixtures' `K` holds.
    pub(super) fn every_planar_format() -> Vec<WeightFormat> {
        SAFETENSORS_FORMATS
            .into_iter()
            .map(|format| match format {
                WeightFormat::Gptq {
                    groups: GroupMap::Contiguous { .. },
                } => gptq_contiguous(),
                WeightFormat::Gptq {
                    groups: GroupMap::Indexed { .. },
                } => gptq_indexed(2),
                WeightFormat::Awq { .. } => awq(),
                other => other,
            })
            .collect()
    }

    /// A codegen-safe (LLVM-identifier-safe) name fragment for a planar format: `{format:?}` on
    /// `E4m3PerChannel { scale: F32 }` contains `{`, `}` and spaces, which break an AMDGCN/NVPTX
    /// global or symbol name built from it (`fixtures::schedule_tag`'s reason, restated for planar
    /// formats).
    pub(super) fn format_tag(format: WeightFormat) -> String {
        match format {
            WeightFormat::E4m3PerChannel { scale } => format!("e4m3_per_channel_{scale:?}"),
            WeightFormat::E4m3Block128 { scale } => format!("e4m3_block128_{scale:?}"),
            WeightFormat::E2m1Row32 => "e2m1_row32".to_string(),
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size },
            } => format!("gptq_contiguous{size}"),
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups },
            } => format!("gptq_indexed{groups}"),
            WeightFormat::Awq { group_size } => format!("awq{group_size}"),
            other => format!("{other:?}"),
        }
    }

    /// xorshift64 bytes (`fixtures::random_bytes`'s twin, kept local so this module has no
    /// dependency on the block front end's fixtures).
    fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    /// Clamp a `Float`-encoded operand's stored bytes off every non-admitted (NaN) bit pattern.
    /// `element_bytes` is the operand's own (unpacked) element width: every `Float`-encoded planar
    /// operand in these nine formats (`Scale`, and `Codes` for the six E4M3 formats) stores one
    /// whole-byte-aligned value per element (`E2M1`'s packed `Codes` is also `Float`-encoded, but
    /// E2M1 has no NaN encoding at all, so it is never sanitized and never reaches this function).
    fn sanitize_float_bytes(bytes: &mut [u8], element_bytes: usize, float: FloatFormat) {
        for chunk in bytes.chunks_mut(element_bytes) {
            match float {
                // Exponent <= 0b0111_1111: clearing bit 6 of the high byte keeps the 8-bit exponent
                // under 0xFF (F16's and BF16's NaN/infinity exponent) for either format.
                FloatFormat::F16 | FloatFormat::Bf16 => chunk[1] &= !0x40,
                // `0x7f`/`0xff` are the two NaN bytes; clearing the mantissa's low bit moves either
                // one to the adjacent (finite, maximum-magnitude) code.
                FloatFormat::E4m3Fn => {
                    if chunk[0] & 0x7f == 0x7f {
                        chunk[0] &= !0x01;
                    }
                }
                // `0xff` is the one non-admitted byte. The fixture also caps the scale at 2^32
                // (`0x9f`): a random E8M0 scale up to 2^127 times E4M3's 448 overflows a
                // contraction's sum to infinity or NaN, and an output that is NaN on both sides
                // checks nothing.
                FloatFormat::E8m0 => chunk[0] = chunk[0].min(0x9f),
                FloatFormat::F32 | FloatFormat::E2m1 => {}
            }
        }
    }

    /// One planar format's source tensors (admitted-domain random bytes, `PackedWeight::sources()`
    /// order) and the `decode_planar_value` reference for every `[out_rows, k]` element: the oracle
    /// every planar interpreter/device test compares against, mirroring `fixtures::fixture`'s role
    /// for the block front end.
    pub(super) fn fixture(
        format: WeightFormat,
        out_rows: usize,
        k: usize,
        seed: u64,
    ) -> (Vec<(SourceRole, Vec<u8>)>, Vec<f32>) {
        let shape = [out_rows, k];
        let weight = PackedWeight::try_new(format, shape)
            .unwrap_or_else(|error| panic!("{format:?} [{out_rows},{k}]: {error}"));
        let descriptor = format.descriptor();
        let sources: Vec<(SourceRole, Vec<u8>)> = weight
            .sources()
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                let SourceRole::Planar(operand_role) = role else {
                    unreachable!("format tests: every planar source is SourceRole::Planar")
                };
                let bytes = if operand_role == PlanarRole::GroupIndex {
                    let WeightFormat::Gptq {
                        groups: GroupMap::Indexed { groups },
                    } = format
                    else {
                        unreachable!("format tests: only act-order GPTQ has a GroupIndex operand")
                    };
                    (0..k)
                        .flat_map(|kk| ((kk % groups) as i32).to_le_bytes())
                        .collect()
                } else {
                    let len = descriptor
                        .planar_operand_bytes(shape, operand_role)
                        .unwrap();
                    let mut bytes =
                        random_bytes(len, seed.wrapping_add(index as u64 * 0x9E37_79B9));
                    let operand = descriptor.planar_operand(operand_role).unwrap();
                    if let FieldEncoding::Float(float) = operand.encoding {
                        sanitize_float_bytes(&mut bytes, operand.element_bytes(), float);
                    }
                    bytes
                };
                (role, bytes)
            })
            .collect();
        let source_refs: Vec<(PlanarRole, &[u8])> = sources
            .iter()
            .map(|(role, bytes)| {
                let SourceRole::Planar(operand_role) = role else {
                    unreachable!()
                };
                (*operand_role, bytes.as_slice())
            })
            .collect();
        let mut expected = vec![0.0f32; out_rows * k];
        for o in 0..out_rows {
            for kk in 0..k {
                expected[o * k + kk] = descriptor
                    .decode_planar_value(shape, &source_refs, [o, kk])
                    .unwrap_or_else(|error| panic!("{format:?} [{o},{kk}]: {error}"));
            }
        }
        (sources, expected)
    }

    /// A prefill-sized planar contraction at the widest geometry `caps` runs (the 16x16 lane grid of 8x8 register
    /// tiles, a 128 x 128 output tile), over more rows and columns than one tile so both edges are ragged.
    /// Every device row that must run the wide workgroup on hardware (the interpreter rows above run the same
    /// shapes) builds its launch from this one case.
    pub(super) struct WideTile {
        pub(super) format: WeightFormat,
        pub(super) m: usize,
        pub(super) out_rows: usize,
        pub(super) schedule: Schedule,
        pub(super) word_bufs: Vec<Vec<u32>>,
        pub(super) activation: Vec<f32>,
        pub(super) expected: Vec<f32>,
    }

    /// The [`WideTile`] for a weight row of `k` values: `E4m3Block128` (the FP8 checkpoints' format), `130` rows
    /// and `130` columns, so each edge of the largest tile the planner weighs (a 16x16 lane grid of 8x8 register
    /// tiles, 128 x 128) is partial. Panics if `caps` does not run that tiling, so the row cannot silently run a
    /// narrower one.
    pub(super) fn wide_tile(k: usize, caps: &poot_target::DeviceCaps) -> WideTile {
        let format = WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        };
        let (m, out_rows) = (130, 130);
        let shape = crate::TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: out_rows,
            k,
            layout: WeightLayout::Nk,
            rows: crate::TiledRows::Dense,
            terms: StagedTerms::ONE,
            weight: crate::StagedWeight::Packed(format),
            activation_bits: 32,
        };
        let tiling = RegisterTiling::candidates(shape, caps)
            .max_by_key(|tiling| {
                let (rows, cols) = tiling.output_tile();
                rows * cols
            })
            .expect("the caps run a register tiling");
        assert_eq!(
            (tiling.workgroup_lanes(), tiling.output_tile()),
            (256, (128, 128)),
            "these caps must run the 16x16 lane grid of 8x8 tiles"
        );
        let (sources, decoded) = fixture(format, out_rows, k, 11);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..m * k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected = (0..m * out_rows)
            .map(|i| {
                let (row, col) = (i / out_rows, i % out_rows);
                (0..k)
                    .map(|kk| activation[row * k + kk] * decoded[col * k + kk])
                    .sum()
            })
            .collect();
        WideTile {
            format,
            m,
            out_rows,
            schedule: tiling.schedule(),
            word_bufs: pack_words(&sources),
            activation,
            expected,
        }
    }

    /// Native source bytes as zero-filled `u32` words plus one zero guard word (D9), one per source
    /// tensor in `fixture`'s order (`fixtures::pack_words`'s twin, one buffer per call there since a
    /// block format has one source).
    pub(super) fn pack_words(sources: &[(SourceRole, Vec<u8>)]) -> Vec<Vec<u32>> {
        sources
            .iter()
            .map(|(_, bytes)| {
                let mut words: Vec<u32> = bytes
                    .chunks(4)
                    .map(|chunk| {
                        let mut word_bytes = [0u8; 4];
                        word_bytes[..chunk.len()].copy_from_slice(chunk);
                        u32::from_le_bytes(word_bytes)
                    })
                    .collect();
                words.push(0);
                words
            })
            .collect()
    }
}

#[cfg(test)]
mod row_gather_tests {
    //! SC-001 (card 542a, RowGather row): `packed_kernel(RowGather)` run in the KIR interpreter
    //! selects the same rows `poot_quant::plan::decode_blocks` decodes (tier 1, GPU-free).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;

    use super::fixtures::{GatherCase, block32, fixture, kquant, kquant_gather_cases, pack_words};
    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_kernel};

    /// POOT-1317: the row gather's block address and bit offset are `usize` arithmetic with no 32-bit step, so
    /// a row whose block lies past `u32::MAX` bits (Flash-Next's PLE table: row `2^26` of a `[_, 160]` `IQ4_NL`
    /// table is bit 4.8e10) is addressed at its true word, not at the word a wrapped offset names. No 28 GB
    /// table is built: the words buffer is four words long, so the gather must fault, and the fault names the
    /// word index it tried to read. A 64-bit offset names word 1509949440 (= `2^26 * 5 * 144 / 32`); a
    /// wrapped one names a word inside `u32::MAX / 32` (which a real, smaller table would silently read).
    ///
    /// Mutation: in `DecodePlan::block_base_bit`, multiply the block in `u32` and widen it back; the fault names
    /// a wrapped word and this goes red.
    #[test]
    fn row_gather_addresses_a_block_past_u32_bits_at_its_true_word() {
        let spec = PackedKernelSpec {
            format: poot_quant::format::WeightFormat::Iq4_Nl,
            op: PackedKernelOp::RowGather,
        };
        let body = packed_kernel("row_gather_wide", spec).unwrap();
        let (k, row) = (160usize, 1i32 << 26);
        let mut buffers = [
            Buffer::from_u32s(&[0; 4]),
            Buffer::from_i32s(&[row]),
            Buffer::from_u32s(&[k as u32]),
            Buffer::from_f32s(&vec![0.0; k]),
        ];
        let error = run(&body, workgroups_covering(&body, k), &mut buffers).expect_err(
            "the table is four words: a row past it must fault, not read a wrapped word",
        );
        let word = (row as u64) * 5 * 144 / 32;
        assert!(
            error
                .to_string()
                .contains(&format!("index {word} is out of bounds")),
            "the gather must read word {word}, got: {error}"
        );
    }

    /// The K-quant sibling of `row_gather_matches_decode_blocks_rows_for_every_block32_format`
    /// (card 542b): exercises `DecodePlan`'s `SubScale`/`SubMin` path through `RowGather` too.
    #[test]
    fn row_gather_matches_decode_blocks_rows_for_every_kquant_format() {
        for GatherCase {
            format,
            label,
            k,
            bytes,
            ids,
            expected,
        } in kquant_gather_cases()
        {
            let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_kquant_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let mut buffers = [
                Buffer::from_u32s(&words),
                Buffer::from_i32s(&ids_i32),
                Buffer::from_u32s(&[k as u32]),
                Buffer::from_f32s(&vec![0.0; ids.len() * k]),
            ];
            run(
                &body,
                workgroups_covering(&body, ids.len() * k),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers[3].to_f32s().unwrap();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?} ({label}): RowGather vs decode_blocks rows"
            );
        }
    }

    #[test]
    fn row_gather_bodies_compile_for_kquant_spirv_and_nvptx() {
        for format in kquant() {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_kquant_{format:?}"), spec).unwrap();
            let _ = crate::unit_helpers::spv(&body, &format!("row_gather_kquant_{format:?}_spv"));
            let _ = crate::unit_helpers::ptx(&body, &format!("row_gather_kquant_{format:?}_ptx"));
        }
    }

    #[test]
    fn row_gather_bodies_compile_for_kquant_amdgcn() {
        use poot_codegen::{Target, artifact_path};
        let arch = poot_target::AmdArch::gfx1151();
        // False once the AMD toolchain proved absent: the SPIR-V and NVPTX compiles still run.
        let mut amd = true;
        for format in kquant() {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_kquant_{format:?}_amdgcn");
            let body = packed_kernel(&name, spec).unwrap();
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            amd = amd && crate::unit_helpers::compile_amdgcn_or_skip(&name, &body, arch, &out);
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_wgpu_kquant() {
        use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
        use poot_wgpu_runtime::{Context, KernelBuffer};

        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            Context::new(),
        ) else {
            return;
        };

        for GatherCase {
            format,
            label,
            k,
            bytes,
            ids,
            expected,
        } in kquant_gather_cases()
        {
            let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_wgpu_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let kernel = kernel_handle(&body, &artifact).unwrap();

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_i32(&ids_i32),
                KernelBuffer::read_only_u32(&[k as u32]),
                KernelBuffer::write_f32(ids.len() * k),
            ];
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [(ids.len() * k) as u32, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers[3].as_f32();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?} ({label}): RowGather (wgpu) vs decode_blocks rows"
            );
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_for_every_block32_format() {
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        for (seed, format) in block32().into_iter().enumerate() {
            let (k, bytes, decoded) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let expected: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    decoded[id as usize * k..(id as usize + 1) * k]
                        .iter()
                        .copied()
                })
                .collect();

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let mut buffers = [
                Buffer::from_u32s(&words),
                Buffer::from_i32s(&ids_i32),
                Buffer::from_u32s(&[k as u32]),
                Buffer::from_f32s(&vec![0.0; ids.len() * k]),
            ];
            run(
                &body,
                workgroups_covering(&body, ids.len() * k),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers[3].to_f32s().unwrap();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: RowGather vs decode_blocks rows"
            );
        }
    }

    /// Sanity only (not acceptance): every
    /// block-32 `RowGather` body lowers to SPIR-V, NVPTX and AMDGCN. Names the SPIR-V/NVPTX artifacts
    /// (`row_gather_<format>_spv`/`_ptx`) for the PTX pod batch (SC-004).
    #[test]
    fn row_gather_bodies_compile_for_spirv_and_nvptx() {
        for format in block32() {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_{format:?}"), spec).unwrap();
            let _ = crate::unit_helpers::spv(&body, &format!("row_gather_{format:?}_spv"));
            let _ = crate::unit_helpers::ptx(&body, &format!("row_gather_{format:?}_ptx"));
        }
    }

    #[test]
    fn row_gather_bodies_compile_for_amdgcn() {
        use poot_codegen::{Target, artifact_path};
        let arch = poot_target::AmdArch::gfx1151();
        // False once the AMD toolchain proved absent: the SPIR-V and NVPTX compiles still run.
        let mut amd = true;
        for format in block32() {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_{format:?}_amdgcn");
            let body = packed_kernel(&name, spec).unwrap();
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            amd = amd && crate::unit_helpers::compile_amdgcn_or_skip(&name, &body, arch, &out);
        }
    }

    /// SC-004 (card 542a, device tier 1): `RowGather` for every block-32 format equals the oracle's
    /// rows on the local wgpu (SPIR-V/RADV) device.
    #[test]
    fn row_gather_matches_decode_blocks_rows_on_wgpu() {
        use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
        use poot_wgpu_runtime::{Context, KernelBuffer};

        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            Context::new(),
        ) else {
            return;
        };

        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        for (seed, format) in block32().into_iter().enumerate() {
            let (k, bytes, decoded) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let expected: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    decoded[id as usize * k..(id as usize + 1) * k]
                        .iter()
                        .copied()
                })
                .collect();

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_wgpu_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let kernel = kernel_handle(&body, &artifact).unwrap();

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_i32(&ids_i32),
                KernelBuffer::read_only_u32(&[k as u32]),
                KernelBuffer::write_f32(ids.len() * k),
            ];
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [(ids.len() * k) as u32, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers[3].as_f32();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: RowGather (wgpu) vs decode_blocks rows"
            );
        }
    }
}

#[cfg(test)]
mod contraction_tests {
    //! SC-005 (card 542a, GPU-free rehearsal): `packed_kernel(Contraction)` (Gemv and RegisterTiled,
    //! blocks 1 and 2) run in the KIR interpreter is within tier-2 tolerance of a
    //! plain f32 reference contraction over `poot_quant::plan::decode_blocks`'s decode. Device rows
    //! (wgpu/ROCm/PTX) are separate, real-hardware evidence (the card's final report).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;

    use super::fixtures::{
        DEVICE_ROWS_AND_DEPTH, GEMV_SCHEDULES, assert_close, block32, contraction_fixture,
        device_rows_and_depth_formats, kquant, pack_words, piecewise, piecewise_run_schedules,
        register_tiled, schedule_tag, with_rows_and_depth,
    };
    use super::kernel::packed_contraction;
    use crate::contraction::{RegisterTiling, Schedule};

    fn run_case(
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        let (got, expected) = interpret(format, schedule, blocks, m_per_block, out_per_block);
        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks}"),
            &got,
            &expected,
        );
    }

    /// `packed_contraction`'s body for `schedule` run on the interpreter over [`contraction_fixture`]: the output
    /// and the plain f32 reference.
    fn interpret(
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;

        let body = packed_contraction(
            &format!("contraction_{format:?}_{}_{blocks}", schedule_tag(schedule)),
            format,
            schedule,
            &[k as u32, out_per_block as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `tiled_body`'s
        // doc). metadata[4] (tile_offset) is 0: each of these runs one unsplit dispatch.
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
        let out_len = rows * out_per_block;
        let mut buffers = [
            Buffer::from_f32s(&activation),
            Buffer::from_u32s(&words),
            Buffer::from_u32s(&metadata),
            Buffer::from_f32s(&vec![0.0; out_len]),
        ];
        // The planner's own launch convention (total threads, `Schedule::grid_threads`), divided by
        // the body's workgroup size.
        let workgroups = workgroups_covering(
            &body,
            schedule.grid_threads(blocks, m_per_block, out_per_block),
        );
        run(&body, workgroups, &mut buffers)
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
        (buffers[3].to_f32s().unwrap(), expected)
    }

    /// POOT-1430: a packed contraction reads its operands as the declared precision says: the activation is rounded
    /// and the weight, decoded to f32 inside the nest, is rounded after the decode (a packed weight is never
    /// stored in the fragment dtype), in the register-tiled schedule, for every block-32 and K-quant format. The decode
    /// Gemv multiplies exact operands and refuses a narrowed precision (`a_gemv_refuses_a_narrowed_precision`). The output equals a host product over the same rounded terms (`poot_eval`'s rounding, the
    /// oracle's definition) to the sum's own rounding error, under the integer rounding and the kept f16
    /// conversion alike (POOT-1458: the decode goes directly to the fragment dtype through the device's rounding).
    /// Mutation: in `packed_contraction` read the weight as `ProductRead::dense(.., WeightLane::Bf16Packed)` (as if
    /// stored narrow, so never rounded); the f16 rows go red.
    #[test]
    fn a_narrowed_packed_contraction_rounds_the_decoded_weight_and_the_activation() {
        use poot_graph_ir::{FragmentDtype, OperandPrecision};
        let round = |dtype, x: f32| poot_eval::round_to_fragment(dtype, x);
        let terms = |precision: OperandPrecision, x: f32| -> Vec<f32> {
            match precision {
                OperandPrecision::Exact => vec![x],
                OperandPrecision::Narrow(d) => vec![round(d, x)],
                OperandPrecision::Split(d) => {
                    let hi = round(d, x);
                    vec![
                        hi,
                        if hi.is_finite() {
                            round(d, x - hi)
                        } else {
                            0.0
                        },
                    ]
                }
            }
        };
        let forms = [
            poot_target::RoundingConversions::NONE,
            poot_target::RoundingConversions::F16,
        ];
        for (precision, conversions) in [
            OperandPrecision::Narrow(FragmentDtype::F16),
            OperandPrecision::Narrow(FragmentDtype::Bf16),
            OperandPrecision::Split(FragmentDtype::F16),
        ]
        .into_iter()
        .flat_map(|precision| forms.map(|conversions| (precision, conversions)))
        // The kept conversion is f16's only: a bf16 precision rounds the same way under either.
        .filter(|(precision, conversions)| {
            *conversions == poot_target::RoundingConversions::NONE
                || precision.fragment_dtype() == Some(FragmentDtype::F16)
        }) {
            for format in block32().into_iter().chain(kquant()) {
                // Two blocks per row with a hand-made tiling, and eight (K = 256 for the 32-value formats) with the
                // tiling the planner's cost model picks for the shape.
                for (blocks_per_row, m, planned) in [(2, 5, false), (8, 4, true)] {
                    let out_per_block = if planned { 8 } else { 7 };
                    let (k, bytes, decoded) =
                        super::fixtures::fixture(format, out_per_block, blocks_per_row, 3);
                    let schedule = if planned {
                        RegisterTiling::choose(
                            crate::TiledShape {
                                blocks: 1,
                                m_per_block: m,
                                out_per_block,
                                k,
                                layout: crate::WeightLayout::Nk,
                                rows: crate::TiledRows::Dense,
                                terms: crate::StagedTerms::ONE,
                                weight: crate::StagedWeight::Packed(format),
                                activation_bits: 32,
                            },
                            &poot_test_util::device_caps::wgpu_rdna3_igpu(),
                        )
                        .schedule()
                    } else {
                        register_tiled(2, k)
                    };
                    let mut state = 0x2545_f491_4f6c_dd1du64;
                    let activation: Vec<f32> = (0..m * k)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            ((state % 2000) as f32 - 1000.0) / 100.0
                        })
                        .collect();
                    let body = packed_contraction(
                        "narrowed",
                        format,
                        schedule,
                        &[k as u32, out_per_block as u32],
                        None,
                        crate::PackedRows::Blocks,
                        precision,
                        conversions,
                    )
                    .unwrap();
                    let metadata = [k as u32, out_per_block as u32, m as u32, 0, 0];
                    let mut buffers = [
                        Buffer::from_f32s(&activation),
                        Buffer::from_u32s(&pack_words(&bytes)),
                        Buffer::from_u32s(&metadata),
                        Buffer::from_f32s(&vec![f32::NAN; m * out_per_block]),
                    ];
                    let workgroups =
                        workgroups_covering(&body, schedule.grid_threads(1, m, out_per_block));
                    run(&body, workgroups, &mut buffers).unwrap();
                    let got = buffers[3].to_f32s().unwrap();
                    for row in 0..m {
                        for col in 0..out_per_block {
                            let (mut want, mut magnitude) = (0.0f32, 0.0f32);
                            for kk in 0..k {
                                for a in terms(precision, activation[row * k + kk]) {
                                    for w in terms(precision, decoded[col * k + kk]) {
                                        want += a * w;
                                        magnitude += (a * w).abs();
                                    }
                                }
                            }
                            let got = got[row * out_per_block + col];
                            // A format whose fixture decodes a non-finite weight (MXFP4's reserved codes) makes the
                            // sum non-finite, and the sign of an infinity or the NaN it meets is the sum order's.
                            assert!(
                                !want.is_finite()
                                    || (got - want).abs() <= 1e-5 * magnitude.max(1.0),
                                "{format:?} {precision} {conversions:?} {schedule:?} [{row}, {col}]: {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// POOT-1405: a Gemv that reads its full runs' activation values as 4-lane vector groups
    /// ([`crate::GemvActivation::Vec4`]) computes the same output bits as the scalar read, for every block-32 and
    /// K-quant format, over the Gemv shapes whose run is a multiple of four (a 4-element run, and a 32-element run
    /// across K-quant sub-blocks), blocks 1 and 2 (activation row `b` starts at `b * K`). Both stay within the
    /// reference's tolerance. Mutation: extract each group's lanes in reverse in `gemv_body`; every row goes red.
    #[test]
    fn a_gemv_refuses_a_narrowed_precision() {
        use poot_graph_ir::{FragmentDtype, OperandPrecision};
        for rows in [crate::PackedRows::Blocks, crate::PackedRows::Indexed] {
            for precision in [
                OperandPrecision::Narrow(FragmentDtype::F16),
                OperandPrecision::Split(FragmentDtype::Bf16),
            ] {
                let error = packed_contraction(
                    "narrowed_gemv",
                    poot_quant::format::WeightFormat::Q8_0,
                    GEMV_SCHEDULES[1],
                    &[64, 8, 4, 0, 0],
                    None,
                    rows,
                    precision,
                    poot_target::RoundingConversions::NONE,
                )
                .unwrap_err();
                assert!(
                    matches!(
                        error,
                        crate::KernelGenError::Unsupported {
                            request: "packed_contraction",
                            ..
                        }
                    ),
                    "{rows:?} {precision}: {error}"
                );
            }
        }
    }

    #[test]
    fn a_vector_activation_read_matches_the_scalar_read_bit_for_bit() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for format in block32().into_iter().chain(kquant()) {
            for scalar in GEMV_SCHEDULES {
                let Schedule::Gemv {
                    width,
                    cols,
                    unroll,
                    ..
                } = scalar
                else {
                    unreachable!("GEMV_SCHEDULES holds Gemv schedules")
                };
                if !unroll.is_multiple_of(4) {
                    continue;
                }
                let vector = Schedule::Gemv {
                    width,
                    cols,
                    unroll,
                    activation: crate::GemvActivation::Vec4,
                    rows: crate::GemvRows::ONE,
                    depth: crate::LoadDepth::ONE,
                };
                for (blocks, out_per_block) in [(1, 11), (2, 5)] {
                    let what = format!("{format:?} {scalar:?} blocks={blocks}");
                    let (want, reference) = interpret(format, scalar, blocks, 1, out_per_block);
                    let (got, _) = interpret(format, vector, blocks, 1, out_per_block);
                    assert_eq!(bits(&got), bits(&want), "{what}");
                    assert_close(format, &what, &got, &reference);
                }
            }
        }
    }

    /// A vector activation read over a run that is not whole groups is refused when the body is built, never
    /// emitted with a short group (the planner never asks for it: [`crate::GemvActivation::Vec4`]'s rule).
    /// Mutation: drop the `unroll % 4` check in `GemvLaunch::new`; the 6-element run builds (or panics) instead.
    #[test]
    fn a_vector_activation_read_over_a_partial_group_run_is_refused() {
        let schedule = Schedule::Gemv {
            width: 32,
            cols: 4,
            unroll: 6,
            activation: crate::GemvActivation::Vec4,
            rows: crate::GemvRows::ONE,
            depth: crate::LoadDepth::ONE,
        };
        let refused = packed_contraction(
            "contraction_vec4_unroll6",
            poot_quant::format::WeightFormat::Q8_0,
            schedule,
            &[],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        );
        assert!(
            matches!(
                refused,
                Err(crate::KernelGenError::NotDivisible {
                    value: 6,
                    divisor: 4,
                    ..
                })
            ),
            "got {:?}",
            refused.err()
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2() {
        for format in block32() {
            for schedule in GEMV_SCHEDULES {
                run_case(format, schedule, 1, 1, 11);
                run_case(format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1542: a format whose layout or lane map is piecewise decodes like the reference under the runs the
    /// planner plans over it. Runs of 4 give every segment the same offsets and share their windows; runs of 8
    /// and 16 do not (the last stretch of `TQ1_0` has four bytes to a row where the others have 32 and 16), so the
    /// emitter addresses the run's first element once, selects the segment it is in once, and reads each element at
    /// that segment's constant offset, and takes the run's lanes the same way. A block of 256 holds all three
    /// stretches of `TQ1_0`, a run of 8 or 16 in each.
    /// Mutation: in `emit_segment_offsets` use the last segment's offsets for every segment; every row of 8 and 16 goes
    /// red naming `Tq1_0`.
    #[test]
    fn runs_over_a_piecewise_layout_decode_like_the_reference() {
        let formats = piecewise();
        assert!(!formats.is_empty(), "a piecewise format is listed");
        for format in formats {
            for schedule in piecewise_run_schedules() {
                run_case(format, schedule, 1, 1, 11);
                run_case(format, schedule, 2, 1, 5);
            }
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_kquant() {
        for format in kquant() {
            for schedule in GEMV_SCHEDULES {
                run_case(format, schedule, 1, 1, 11);
                run_case(format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1559 SC-a: the Gemv reads every bit offset of a run at an alignment proven from the descriptor, and
    /// reads exactly the bits the unaligned windows read, so the output of every GGUF block format under every
    /// `GEMV_SCHEDULES` entry is the one recorded from the body before the static-alignment reader (the f32 bits of
    /// the interpreter's output over the shared fixture, digested).
    /// Mutation: make `PieceAlignment::of` return 64 bits for a 2-byte-aligned format; the interpreter faults the
    /// misaligned vector load, or the Q6_K row reads wrong bits.
    #[test]
    fn every_format_gemv_is_bit_identical_after_static_alignment() {
        let recorded = include_str!("../../tests/data/packed_gemv_output_bits.txt");
        let mut current = Vec::new();
        for format in block32().into_iter().chain(kquant()) {
            for (index, schedule) in GEMV_SCHEDULES.into_iter().enumerate() {
                for (blocks, out_per_block) in [(1, 11), (2, 5)] {
                    let (got, _) = interpret(format, schedule, blocks, 1, out_per_block);
                    let text: String = got.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
                    current.push(format!(
                        "{format:?} | gemv{index} blocks={blocks} | {:016x}",
                        super::identity_tests::digest(&text)
                    ));
                }
            }
        }
        let expected: Vec<&str> = recorded.lines().collect();
        let changed: Vec<String> = current
            .iter()
            .zip(&expected)
            .filter(|(now, was)| now.as_str() != **was)
            .map(|(now, was)| format!("recorded {was}\n     now {now}"))
            .collect();
        assert!(
            changed.is_empty() && current.len() == expected.len(),
            "{} of {} Gemv outputs changed ({} recorded):\n{}\n--- full table ---\n{}",
            changed.len(),
            current.len(),
            expected.len(),
            changed.join("\n"),
            current.join("\n")
        );
    }

    /// POOT-1560 SC-e: a lane that accumulates `rows` adjacent output columns and keeps `depth` trips of loads in flight
    /// computes every column's sum bit for bit as it does at one row and one trip, at the same lanes per column and
    /// run, over every block format and three launches (a run that ends on a block, one that straddles several
    /// trips of a K-quant row, one of 3 values that no block divides), over two blocks of 12 outputs. 12 is a multiple of
    /// every row count, and 24 columns over groups of up to 4 columns of 4 rows leave the last workgroup's tail past the
    /// output.
    /// Mutation: in `gemv_body` run the products of rows after the first in reverse order of the run (the
    /// `MUTATION` line); every row count above one goes red, rows = 1 stays green.
    #[test]
    fn rows_and_depth_do_not_change_a_rows_sum() {
        let launches = [
            GEMV_SCHEDULES[1],
            with_rows_and_depth(GEMV_SCHEDULES[2], 1, 1),
            Schedule::Gemv {
                width: 16,
                cols: 2,
                unroll: 8,
                activation: crate::GemvActivation::Scalar,
                rows: crate::GemvRows::ONE,
                depth: crate::LoadDepth::ONE,
            },
        ];
        for format in block32().into_iter().chain(kquant()) {
            for launch in launches {
                for (blocks, out_per_block) in [(2, 12)] {
                    let (want, _) = interpret(format, launch, blocks, 1, out_per_block);
                    for rows in 1..=4 {
                        for depth in 1..=3 {
                            let schedule = with_rows_and_depth(launch, rows, depth);
                            let (got, _) = interpret(format, schedule, blocks, 1, out_per_block);
                            assert_eq!(
                                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "{format:?} {launch:?} blocks={blocks} rows={rows} depth={depth}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A launch whose lane rows straddle two blocks (`out_per_block` not a multiple of `rows`) would meet one
    /// activation row with weight rows of another; the body computes NaN for every output instead of a sum over the
    /// wrong activation, and the same launch at one row, or at a count that divides `out_per_block`, is finite.
    /// Mutation: drop the poison term from the published partials in `gemv_body`; the straddling launch is finite and
    /// the row goes red.
    #[test]
    fn a_launch_whose_rows_straddle_a_block_computes_nan() {
        let format = block32()[0];
        let launch = GEMV_SCHEDULES[1];
        let (blocks, out_per_block) = (2, 6);
        for (rows, poisoned) in [(1, false), (2, false), (3, false), (4, true)] {
            let (got, _) = interpret(
                format,
                with_rows_and_depth(launch, rows, 1),
                blocks,
                1,
                out_per_block,
            );
            assert!(
                got.iter().all(|v| v.is_nan() == poisoned),
                "rows={rows}: {got:?}"
            );
        }
    }

    /// The weight reads of one run of 16 elements of `format` (`DecodePlan::emit_run`, the statements a Gemv trip
    /// emits for it): the lane counts of its vector loads of the packed words, its scalar loads of them, and the
    /// `min` clamps in the run.
    fn weight_reads(format: poot_quant::format::WeightFormat) -> (Vec<u32>, usize, usize) {
        use poot_kernel_ir::{BinOp, Operand, Rvalue, Statement, Ty};

        use super::decode_plan::DecodePlan;
        use super::source::Source;
        use crate::emit::Emit;
        use crate::helpers::{Alloc, ld, local, slice_dtype};

        let plan = DecodePlan::of(&format.descriptor()).unwrap();
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_dtype(Ty::U32, false), false),
        ]);
        let words = local(1);
        let [row, k0, k] = [(); 3].map(|()| al.add(Ty::Usize, false));
        let mut e = Emit::new(&mut al);
        plan.emit_run(&mut e, Source::plain(words), row, k0, 16, k);
        let statements = e.take();
        let (mut vectors, mut scalars, mut clamps) = (Vec::new(), 0, 0);
        for statement in &statements {
            let Statement::Assign(dest, rvalue) = statement else {
                continue;
            };
            match rvalue {
                Rvalue::VectorLoad { place, .. } if place.local == words => {
                    let Ty::Vec { lanes, .. } = &al.locals[dest.local.index as usize].ty else {
                        unreachable!("a vector load assigns a vector")
                    };
                    vectors.push(*lanes);
                }
                Rvalue::Use(Operand::Copy(place)) if place.local == words => scalars += 1,
                Rvalue::BinaryOp(BinOp::Min, ..) => clamps += 1,
                _ => {}
            }
        }
        (vectors, scalars, clamps)
    }

    /// POOT-1559 SC-b: an aligned run reads exactly its span. A Q4_K run of 16 codes is the same nibble of 16
    /// consecutive bytes, 128 bits starting in a 128-bit-aligned word, so it is one `u32x4` load and no other code
    /// word, with one runtime shift by the nibble; its factors are one narrow word each (`d`, `dmin`, two pieces
    /// of `SubScale`, two of `SubMin`), where the unaligned reader loaded two stretches of code words and 12
    /// scalar header words. Nothing clamps.
    /// Mutation: read one word more than the span touches on the aligned path of `emit_read_run` (the `+1` word the
    /// unaligned reader loads); the vector load is a `u32x4` and a scalar, and the row is red.
    #[test]
    fn an_aligned_run_reads_exactly_its_span() {
        let (vectors, scalars, clamps) = weight_reads(poot_quant::format::WeightFormat::Q4_K);
        assert_eq!(
            (vectors.as_slice(), scalars, clamps),
            (&[4][..], 6, 0),
            "Q4_K run 16: one u32x4 code load, six narrow factor loads, no clamp"
        );
    }

    /// POOT-1559 SC-b, the 2-byte-aligned side: a Q6_K block is 210 bytes, so a run starts on a 16-bit boundary,
    /// possibly 16 bits into a word, and reads the 16 bytes of its `ql` codes and of its `qh` bits from 5 words each
    /// (the span plus one word, the funnel's residue), scalar loads: no vector load can be proven for an
    /// odd block, and no clamp. Beside them one narrow word each for the `i8` sub-scale and `d`.
    /// Mutation: `PieceAlignment::of` reports a zero residue for any format; the codes read a word fewer, and the
    /// load count and SC-a's bits are red.
    #[test]
    fn a_two_byte_aligned_run_reads_its_span_plus_one_word() {
        let (vectors, scalars, clamps) = weight_reads(poot_quant::format::WeightFormat::Q6_K);
        assert_eq!(
            (vectors.as_slice(), scalars, clamps),
            (&[][..], 12, 0),
            "Q6_K run 16: no vector load, 5 + 5 code words and two factor words, no clamp"
        );
    }

    /// The bias, tanh, scale and residual chain of the fuse rule: `tanh(head + bias) * 0.5 + residual`, as the
    /// kernelgen epilogue the planner builds from a `ContractionEpilogue` over a packed head.
    pub(super) fn bias_activation_residual_chain(out: usize) -> crate::contraction::Epilogue {
        bias_activation_residual_chain_over(1, out)
    }

    /// [`bias_activation_residual_chain`] over a `[rows, out]` output: the bias is `[out]`, the residual the output's
    /// own shape.
    pub(super) fn bias_activation_residual_chain_over(
        rows: usize,
        out: usize,
    ) -> crate::contraction::Epilogue {
        use crate::contraction::{
            Epilogue, EpilogueInput as In, EpilogueLeaf, EpilogueStep as Step,
        };
        use crate::fused::FusedScalarOp as Op;
        Epilogue {
            dt: poot_kernel_ir::Ty::F32,
            out_shape: vec![rows, out],
            leaves: [vec![out], vec![rows, out]]
                .into_iter()
                .map(|shape| EpilogueLeaf {
                    shape,
                    storage: None,
                })
                .collect(),
            steps: vec![
                Step {
                    op: Op::Binary(poot_kernel_ir::BinOp::Add),
                    inputs: vec![In::Head, In::Leaf(0)],
                },
                Step {
                    op: Op::Tanh,
                    inputs: vec![In::Step(0)],
                },
                Step {
                    op: Op::Binary(poot_kernel_ir::BinOp::Mul),
                    inputs: vec![In::Step(1), In::Lit(0.5)],
                },
                Step {
                    op: Op::Binary(poot_kernel_ir::BinOp::Add),
                    inputs: vec![In::Step(2), In::Leaf(1)],
                },
            ],
            output: In::Step(3),
        }
    }

    /// The host image of [`bias_activation_residual_chain`] over one head value.
    pub(super) fn apply_chain(head: f32, bias: f32, residual: f32) -> f32 {
        (head + bias).tanh() * 0.5 + residual
    }

    /// A packed Gemv with an epilogue applies it once to each column's completed sum, before the one store:
    /// the result equals the reference contraction with the chain applied on the host, for every Gemv launch
    /// shape (lanes of a column reduce in LDS, so an epilogue applied per lane would add the bias `lanes`
    /// times). The leaves ride between the weight words and the metadata.
    #[test]
    fn contraction_gemv_epilogue_is_applied_once_to_the_completed_sum() {
        let out_per_block = 11;
        let bias: Vec<f32> = (0..out_per_block).map(|i| 0.3 - i as f32 * 0.07).collect();
        let residual: Vec<f32> = (0..out_per_block).map(|i| i as f32 * 0.11 - 0.5).collect();
        for format in [
            poot_quant::format::WeightFormat::Q8_0,
            poot_quant::format::WeightFormat::Q4_K,
        ] {
            let (k, bytes, activation, head) = contraction_fixture(format, 1, 1, out_per_block);
            // The head sums are O(10); scale them so tanh does not saturate every column.
            let activation: Vec<f32> = activation.iter().map(|v| v * 0.02).collect();
            let head: Vec<f32> = head.iter().map(|v| v * 0.02).collect();
            let expected: Vec<f32> = (0..out_per_block)
                .map(|i| apply_chain(head[i], bias[i], residual[i]))
                .collect();
            let epilogue = bias_activation_residual_chain(out_per_block);
            for schedule in GEMV_SCHEDULES {
                let body = packed_contraction(
                    &format!("epilogue_{format:?}_{}", schedule_tag(schedule)),
                    format,
                    schedule,
                    &[],
                    Some(&epilogue),
                    crate::PackedRows::Blocks,
                    poot_graph_ir::OperandPrecision::Exact,
                    poot_target::RoundingConversions::NONE,
                )
                .unwrap_or_else(|error| panic!("{format:?}: {error}"));
                body.verify().expect("the epilogue body verifies");
                let words = pack_words(&bytes);
                let mut buffers = [
                    Buffer::from_f32s(&activation),
                    Buffer::from_u32s(&words),
                    Buffer::from_f32s(&bias),
                    Buffer::from_f32s(&residual),
                    Buffer::from_u32s(&[k as u32, out_per_block as u32]),
                    Buffer::from_f32s(&vec![0.0; out_per_block]),
                ];
                run(
                    &body,
                    workgroups_covering(&body, schedule.grid_threads(1, 1, out_per_block)),
                    &mut buffers,
                )
                .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
                assert_close(
                    format,
                    &format!("epilogue {schedule:?}"),
                    &buffers[5].to_f32s().unwrap(),
                    &expected,
                );
            }
        }
    }

    /// The block-diagonal tiled schedule applies the epilogue once to each completed sum of every output element
    /// (POOT-1626): the result equals the reference contraction with the chain applied on the host, over a ragged row
    /// tile and column edge and over two blocks. The leaves ride between the weight words and the metadata, the
    /// bias along the columns and the residual per output element.
    ///
    /// Mutation: drop the epilogue in `contraction_register_tiled` (`epilogue: None` in its `TiledBody`): the output
    /// is the plain product and this row goes red (`Q8_0 tiled epilogue element 0: got 11.417626 want 0`).
    #[test]
    fn a_tiled_packed_contraction_applies_its_epilogue_once_to_each_completed_sum() {
        let (blocks, m_per_block, out_per_block) = (2, 7, 11);
        let rows = blocks * m_per_block;
        let bias: Vec<f32> = (0..out_per_block).map(|i| 0.3 - i as f32 * 0.07).collect();
        let residual: Vec<f32> = (0..rows * out_per_block)
            .map(|i| i as f32 * 0.011 - 0.5)
            .collect();
        for format in [
            poot_quant::format::WeightFormat::Q8_0,
            poot_quant::format::WeightFormat::Q4_K,
        ] {
            let (k, bytes, activation, head) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            // The head sums are O(10); scale them so tanh does not saturate every element.
            let activation: Vec<f32> = activation.iter().map(|v| v * 0.02).collect();
            let head: Vec<f32> = head.iter().map(|v| v * 0.02).collect();
            // The output is `[rows, out_per_block]` over both blocks: the bias leaf is `[out_per_block]`, one value
            // per column, and the residual the output's own shape.
            let expected: Vec<f32> = (0..rows * out_per_block)
                .map(|i| apply_chain(head[i], bias[i % out_per_block], residual[i]))
                .collect();
            let epilogue = bias_activation_residual_chain_over(rows, out_per_block);
            let schedule = register_tiled(2, 32);
            let body = packed_contraction(
                &format!("tiled_epilogue_{format:?}"),
                format,
                schedule,
                &[k as u32, out_per_block as u32],
                Some(&epilogue),
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{format:?}: {error}"));
            body.verify().expect("the epilogue body verifies");
            let words = pack_words(&bytes);
            let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
            let mut buffers = [
                Buffer::from_f32s(&activation),
                Buffer::from_u32s(&words),
                Buffer::from_f32s(&bias),
                Buffer::from_f32s(&residual),
                Buffer::from_u32s(&metadata),
                Buffer::from_f32s(&vec![0.0; rows * out_per_block]),
            ];
            run(
                &body,
                workgroups_covering(
                    &body,
                    schedule.grid_threads(blocks, m_per_block, out_per_block),
                ),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            assert_close(
                format,
                "tiled epilogue",
                &buffers[5].to_f32s().unwrap(),
                &expected,
            );
        }
    }

    /// SC-002 (card 542b, GPU-free rehearsal): `Contraction` `Schedule::RegisterTiled` (`lanes = 2` of two
    /// rows each: a 4x8 output tile), `blocks = 1`, for every block-32 and K-quant format - over both format
    /// families (the schedule carries no per-format code, so a block-32 format is as good a witness as a
    /// K-quant one). `m_per_block` (`out_per_block` follows it 1:1 in this fixture, `contraction_fixture`'s
    /// doc) of 4 (exactly one tile), 7 and 8 (a second, ragged row-tile and a ragged column edge) and 12 (a
    /// third row-tile and a second column-tile), so the grid split `g -> (trow, tcol)` depends on `tiles_col =
    /// ceil(out / tile_cols)` exactly (mutants-m4 H6: at 12 `ceil(12/8) = 2` where `12/8 = 1`).
    ///
    /// Card 658: `blocks == 1` only through card 542b/545b - a `blocks > 1` (ADR-0109 block-diagonal)
    /// claim could have a row-tile straddle two blocks whenever `m_per_block` was not a multiple of
    /// the tile, so a straddling tile would read the wrong block's weight for some of its rows and the
    /// planner fell back to the per-thread-unbounded `Schedule::Serial` instead (since deleted:
    /// nothing plans it any more). `tiled_body` now lays its tiles out per block (`b = g /
    /// tiles_per_block`, fixed per workgroup), so this test's `blocks == 1` cases stay green unchanged.
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged() {
        for format in block32().into_iter().chain(kquant()) {
            run_case(format, register_tiled(2, 32), 1, 4, 4);
            run_case(format, register_tiled(2, 32), 1, 7, 7);
            run_case(format, register_tiled(2, 32), 1, 8, 8);
            run_case(format, register_tiled(2, 32), 1, 12, 12);
        }
    }

    /// SC-002 (card 658, GPU-free rehearsal): `Contraction` `Schedule::RegisterTiled` over a block-diagonal
    /// claim (`blocks = 3`) whose `m_per_block = 7` is not a multiple of `tile = 4` - the exact shape
    /// `try_plan_contraction` used to refuse to tile and fall back to the per-thread-
    /// unbounded `Schedule::Serial` for (`Serial` since deleted). Every
    /// output is checked against
    /// `contraction_fixture`'s block-diagonal reference (`reference_contraction`'s `b = row_idx /
    /// m_per_block`, a fresh partial sum per block), so a tile that reused another block's decoded
    /// weight row - the straddling hazard card 658 closed - would show up as a
    /// wrong value on every block boundary row, not just an out-of-range access. `out_per_block = 5`
    /// (also not a multiple of `tile`) exercises the ragged column edge at the same time.
    /// Mutation: in `tiled_body`, replace the tile-fixed `b` with the
    /// old per-row derivation (`row.min(rows - 1) / m_per_block`, `row` from the global `(trow, tcol)`
    /// tile index rather than block-major `g`) - the row-tile at the block-2/block-3 boundary mixes
    /// block 1's weight into some of block 2's rows, and this test goes red.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged() {
        for format in block32().into_iter().chain(kquant()) {
            run_case(format, register_tiled(2, 32), 3, 7, 5);
        }
    }

    /// SC-001 (card 658, GPU-free): dispatches the exact folded geometry
    /// `try_plan_contraction` would plan for `blocks = 1, m_per_block = 12, out_per_block = 4`, two lanes
    /// of two rows (a 4x8 output tile), on a device whose grid cap folds `tiles = ceil(12/4)*ceil(4/8) = 3` onto
    /// `(x_groups, y_groups) = (2, 2)` (`metadata[3] = 2`) - `workgroups = [2, 2, 1]`, i.e. 4 workgroups for 3
    /// real tiles. The padding workgroup (`g = 3`) computes `b = g / tiles_per_block >= blocks`
    /// (`tiles_per_block = 3` here, so `g = 3` lands at `b = 1`), and the interpreter - unlike
    /// wgpu/RADV's robust-buffer-access, which silently drops the same out-of-bounds store
    /// (`contraction_tiled_folded_grid_matches_reference_on_wgpu` runs this identical geometry on real
    /// hardware without ever catching it) - hard-faults on it.
    ///
    /// Mutation: drop the `in_range` term from `store_ok` (keep `row_ok & col_ok` alone, as
    /// before card 658's fix). Red: `InterpError::Fault { fault: OutOfBounds { .. }, .. }` ("buffer index 160
    /// is out of bounds for length 48" for a `rows=12, out_per_block=4` Q4_0 fixture). Restored; green.
    #[test]
    fn contraction_tiled_folded_grid_padding_workgroups_stay_in_bounds() {
        let (blocks, m_per_block, out_per_block, tile) = (1usize, 12usize, 4usize, 2u32);
        let (tile_rows, tile_cols) = register_tiled(tile, 32).output_tile().unwrap();
        let tiles = blocks * m_per_block.div_ceil(tile_rows) * out_per_block.div_ceil(tile_cols);
        let (x_groups, y_groups) = (2u32, 2u32);
        assert_eq!(tiles, 3, "fixture shape must need exactly 3 tiles");
        assert!(
            (x_groups as usize) * (y_groups as usize) > tiles,
            "the fold must dispatch at least one padding workgroup for this to be a real test"
        );
        for format in block32().into_iter().chain(kquant()) {
            let (k, bytes, activation, expected) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            let schedule = register_tiled(tile, 32);
            let body = packed_contraction(
                &format!("contraction_folded_interp_{format:?}"),
                format,
                schedule,
                &[k as u32, out_per_block as u32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let metadata = [
                k as u32,
                out_per_block as u32,
                m_per_block as u32,
                x_groups,
                0,
            ];
            let rows = blocks * m_per_block;
            let out_len = rows * out_per_block;
            let mut buffers = [
                Buffer::from_f32s(&activation),
                Buffer::from_u32s(&words),
                Buffer::from_u32s(&metadata),
                Buffer::from_f32s(&vec![0.0; out_len]),
            ];
            run(&body, [x_groups, y_groups, 1], &mut buffers).unwrap_or_else(|error| {
                panic!("{format:?}: interpreter: folded grid [2,2,1] (tiles=3): {error}")
            });
            let got = buffers[3].to_f32s().unwrap();
            assert_close(
                format,
                &format!("Contraction {schedule:?} folded grid=[2,2,1]"),
                &got,
                &expected,
            );
        }
    }

    /// Sanity only (not acceptance): every
    /// block-32 `Contraction` (Gemv) body lowers to SPIR-V, NVPTX and AMDGCN. Names the SPIR-V/NVPTX
    /// artifacts for the PTX pod batch (SC-005).
    #[test]
    fn contraction_bodies_compile_for_spirv_nvptx_and_amdgcn() {
        use poot_codegen::{Target, artifact_path};
        let arch = poot_target::AmdArch::gfx1151();
        // False once the AMD toolchain proved absent: the SPIR-V and NVPTX compiles still run.
        let mut amd = true;
        for format in block32() {
            for schedule in GEMV_SCHEDULES {
                let name = format!("contraction_{format:?}_{}", schedule_tag(schedule));
                let body = packed_contraction(
                    &name,
                    format,
                    schedule,
                    &[],
                    None,
                    crate::PackedRows::Blocks,
                    poot_graph_ir::OperandPrecision::Exact,
                    poot_target::RoundingConversions::NONE,
                )
                .unwrap();
                let _ = crate::unit_helpers::spv(&body, &format!("{name}_spv"));
                let _ = crate::unit_helpers::ptx(&body, &format!("{name}_ptx"));
                let dir = std::env::temp_dir()
                    .join("poot-kernelgen-unit-test")
                    .join(format!("{name}_amdgcn"));
                std::fs::create_dir_all(&dir).unwrap();
                let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
                amd = amd && crate::unit_helpers::compile_amdgcn_or_skip(&name, &body, arch, &out);
            }
        }
    }

    /// The K-quant sibling of `contraction_bodies_compile_for_spirv_nvptx_and_amdgcn`, plus the new
    /// `Schedule::RegisterTiled` on every format (block-32 and K-quant): sanity only, names the SPIR-V/NVPTX
    /// artifacts for the PTX pod batch.
    #[test]
    fn contraction_tiled_and_kquant_bodies_compile_for_spirv_nvptx_and_amdgcn() {
        use poot_codegen::{Target, artifact_path};
        let arch = poot_target::AmdArch::gfx1151();
        // False once the AMD toolchain proved absent: the SPIR-V and NVPTX compiles still run.
        let mut amd = true;
        for format in kquant() {
            for schedule in GEMV_SCHEDULES {
                let name = format!("contraction_kquant_{format:?}_{}", schedule_tag(schedule));
                let body = packed_contraction(
                    &name,
                    format,
                    schedule,
                    &[],
                    None,
                    crate::PackedRows::Blocks,
                    poot_graph_ir::OperandPrecision::Exact,
                    poot_target::RoundingConversions::NONE,
                )
                .unwrap();
                let _ = crate::unit_helpers::spv(&body, &format!("{name}_spv"));
                let _ = crate::unit_helpers::ptx(&body, &format!("{name}_ptx"));
                let dir = std::env::temp_dir()
                    .join("poot-kernelgen-unit-test")
                    .join(format!("{name}_amdgcn"));
                std::fs::create_dir_all(&dir).unwrap();
                let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
                amd = amd && crate::unit_helpers::compile_amdgcn_or_skip(&name, &body, arch, &out);
            }
        }
        for format in block32().into_iter().chain(kquant()) {
            let schedule = register_tiled(2, 32);
            let name = format!("contraction_{format:?}_{}", schedule_tag(schedule));
            let body = packed_contraction(
                &name,
                format,
                schedule,
                &[32, 32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap();
            let _ = crate::unit_helpers::spv(&body, &format!("{name}_spv"));
            let _ = crate::unit_helpers::ptx(&body, &format!("{name}_ptx"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(format!("{name}_amdgcn"));
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            amd = amd && crate::unit_helpers::compile_amdgcn_or_skip(&name, &body, arch, &out);
        }
    }

    /// SC-005 (card 542a, device tier 2): `Contraction` (Gemv, blocks
    /// 1 and 2) is within tier-2 tolerance of the reference on the local wgpu (SPIR-V/RADV) device.
    fn run_wgpu_case(
        ctx: &poot_wgpu_runtime::Context,
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
        use poot_wgpu_runtime::KernelBuffer;

        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;
        let name = format!(
            "contraction_wgpu_{format:?}_{}_{blocks}",
            schedule_tag(schedule)
        );
        let body = packed_contraction(
            &name,
            format,
            schedule,
            &[k as u32, out_per_block as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004")
            .join(&name);
        std::fs::create_dir_all(&dir).unwrap();
        let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
        let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out_path)
            .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
        let kernel = kernel_handle(&body, &artifact).unwrap();

        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `tiled_body`'s
        // doc). metadata[4] (tile_offset) is 0: each of these runs one unsplit dispatch.
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
        let out_len = rows * out_per_block;
        let mut buffers = [
            KernelBuffer::read_only_f32(&activation),
            KernelBuffer::read_only_u32(&words),
            KernelBuffer::read_only_u32(&metadata),
            KernelBuffer::write_f32(out_len),
        ];
        // The launch is in threads (`Schedule::grid_threads`); `Context::dispatch` divides by
        // `body.workgroup_size` internally, matching the ROCm/PTX harnesses' convention.
        let threads = schedule.grid_threads(blocks, m_per_block, out_per_block) as u32;
        ctx.dispatch(
            &name,
            &kernel,
            body.workgroup_size,
            [threads, 1, 1],
            &mut buffers,
        )
        .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
        let got = buffers[3].as_f32();
        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks} (wgpu)"),
            got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in block32() {
            for schedule in GEMV_SCHEDULES {
                run_wgpu_case(&ctx, format, schedule, 1, 1, 11);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1560 (device tier 2): a Gemv whose lanes accumulate several rows and keep several trips of loads in flight
    /// is within tier-2 tolerance of the reference on the local wgpu (SPIR-V/RADV) device, over two blocks of 12
    /// outputs (a multiple of every row count). The interpreter pins these bit for bit against one row and one trip
    /// (`rows_and_depth_do_not_change_a_rows_sum`).
    #[test]
    fn a_gemv_with_rows_and_depth_matches_reference_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in device_rows_and_depth_formats() {
            for (rows, depth) in DEVICE_ROWS_AND_DEPTH {
                let schedule = with_rows_and_depth(GEMV_SCHEDULES[1], rows, depth);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 12);
            }
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_wgpu_kquant() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in kquant() {
            for schedule in GEMV_SCHEDULES {
                run_wgpu_case(&ctx, format, schedule, 1, 1, 11);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1542 (device tier 2): the piecewise formats' runs of 4, 8 and 16 on the local wgpu (SPIR-V/RADV) device, the
    /// counterpart of `runs_over_a_piecewise_layout_decode_like_the_reference`.
    #[test]
    fn runs_over_a_piecewise_layout_decode_like_the_reference_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in piecewise() {
            for schedule in piecewise_run_schedules() {
                run_wgpu_case(&ctx, format, schedule, 1, 1, 11);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2): `Contraction` `Schedule::RegisterTiled` for every block-32 and
    /// K-quant format on the local wgpu (SPIR-V/RADV) device - the real-hardware counterpart of
    /// `contraction_tiled_matches_reference_aligned_and_ragged` (`blocks == 1` only; see its doc).
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_wgpu_case(&ctx, format, register_tiled(2, 32), 1, 4, 4);
            run_wgpu_case(&ctx, format, register_tiled(2, 32), 1, 7, 7);
        }
    }

    /// SC-001/SC-002 (card 658, device tier 2, wgpu/RADV): the real-hardware counterpart of
    /// `contraction_tiled_matches_reference_block_diagonal_ragged` - a block-diagonal `Schedule::RegisterTiled`
    /// claim (`blocks = 3`, `m_per_block = 7` not a multiple of `tile = 4`) completes under the GPU
    /// lock and matches the CPU oracle, proving the shape `try_plan_contraction` used to refuse to
    /// tile (falling back to the per-thread-unbounded `Schedule::Serial`) now dispatches
    /// bounded and correct on real hardware.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_wgpu_case(&ctx, format, register_tiled(2, 32), 3, 7, 5);
        }
    }

    /// SC-001 (card 658 review F1/F2, device tier 2, wgpu/RADV): the real-hardware counterpart of
    /// `packed_dequant::tests::over_cap_tiled_contraction_folds_onto_a_2d_grid_and_keeps_its_workgroup` - a
    /// `Schedule::RegisterTiled` dispatch folded onto a 2-D `(x_groups, y_groups)` grid (`tiles =
    /// ceil(12/4)*ceil(4/8) = 3` over a synthetic `x_groups` cap of 2, folding to `(2, 2)`, the smallest
    /// shape that exercises a real `GroupY > 0` workgroup) matches the CPU oracle on real hardware, not
    /// just the KIR interpreter - proving `tiled_body`'s `g = gy*x_groups+gx` reconstruction
    /// (`metadata[3]`) is correct on the real SPIR-V/RADV lowering of `GroupY`, not only in the
    /// zero-cost interpreter path every other row here exercises at `GroupY == 0`.
    ///
    /// Mutation: in `tiled_body`'s `g = gy*x_groups + gx`, zero the `x_groups` operand (`g =
    /// gy*0 + gx`), dropping `GroupY`'s contribution. Red on real wgpu/RADV: `Q4_0 ... folded
    /// grid=[64,2,1] (wgpu) element 64: got 0 want -55.689743` - element 64 is the first output of the
    /// fold's second row (`GroupY == 1`), which the mutated body never reaches. Restored; green again.
    #[test]
    fn contraction_tiled_folded_grid_matches_reference_on_wgpu() {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            poot_wgpu_runtime::Context::new(),
        ) else {
            return;
        };
        let (blocks, m_per_block, out_per_block, tile) = (1usize, 12usize, 4usize, 2u32);
        let ts = tile as usize;
        let (tile_rows, tile_cols) = register_tiled(tile, 32).output_tile().unwrap();
        let tiles = blocks * m_per_block.div_ceil(tile_rows) * out_per_block.div_ceil(tile_cols);
        let (x_groups, y_groups) = (2u32, 2u32);
        assert_eq!(tiles, 3, "fixture shape must need exactly 3 tiles");
        assert!(
            (x_groups as usize) * (y_groups as usize) >= tiles,
            "the folded grid must address every tile"
        );
        for format in block32().into_iter().chain(kquant()) {
            let (k, bytes, activation, expected) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            let schedule = register_tiled(tile, 32);
            let name = format!("contraction_wgpu_folded_{format:?}");
            let body = packed_contraction(
                &name,
                format,
                schedule,
                &[k as u32, out_per_block as u32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-card658-fold")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path =
                poot_codegen::artifact_path(&dir, &name, poot_codegen::Target::SpirvVulkan);
            let artifact =
                poot_codegen::compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out_path)
                    .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let kernel = poot_codegen::kernel_handle(&body, &artifact).unwrap();

            let words = pack_words(&bytes);
            let metadata = [
                k as u32,
                out_per_block as u32,
                m_per_block as u32,
                x_groups,
                0,
            ];
            let rows = blocks * m_per_block;
            let out_len = rows * out_per_block;
            let mut buffers = [
                poot_wgpu_runtime::KernelBuffer::read_only_f32(&activation),
                poot_wgpu_runtime::KernelBuffer::read_only_u32(&words),
                poot_wgpu_runtime::KernelBuffer::read_only_u32(&metadata),
                poot_wgpu_runtime::KernelBuffer::write_f32(out_len),
            ];
            let threads_x = x_groups * (ts * ts) as u32;
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [threads_x, y_groups, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers[3].as_f32();
            assert_close(
                format,
                &format!("Contraction {schedule:?} folded grid=[{threads_x},{y_groups},1] (wgpu)"),
                got,
                &expected,
            );
        }
    }
}

#[cfg(test)]
mod planar_contraction_tests {
    //! SC-002 (card 542c, GPU-free rehearsal): `packed_kernel(Contraction)` (Gemv and RegisterTiled,
    //! dense rows) for every planar format run in the KIR interpreter is within
    //! tier-2 tolerance of a plain f32 reference contraction over `decode_planar_value` - the planar
    //! counterpart of `contraction_tests`.

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;

    use super::fixtures::{GEMV_SCHEDULES, assert_close, register_tiled};
    use super::kernel::packed_contraction;
    use super::planar_fixtures::{every_planar_format, fixture, pack_words};
    use crate::contraction::{RegisterTiling, Schedule};
    use crate::helpers::WeightLayout;

    /// `K` values of a weight row: 17 leaves no run longer than one value, 18 stages in runs of two, 20 in
    /// runs of four.
    const KS: [usize; 3] = [17, 18, 20];

    fn reference_contraction(
        activation: &[f32],
        decoded: &[f32],
        k: usize,
        out: usize,
    ) -> Vec<f32> {
        let rows = activation.len() / k;
        (0..rows * out)
            .map(|i| {
                let row = i / out;
                let col = i % out;
                (0..k)
                    .map(|kk| activation[row * k + kk] * decoded[col * k + kk])
                    .sum()
            })
            .collect()
    }

    fn run_case(
        format: poot_quant::format::WeightFormat,
        k: usize,
        out_rows: usize,
        m: usize,
        schedule: Schedule,
    ) {
        let (sources, decoded) = fixture(format, out_rows, k, 11);
        let word_bufs = pack_words(&sources);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..m * k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected = reference_contraction(&activation, &decoded, k, out_rows);

        let body = packed_contraction(
            &format!(
                "contraction_planar_{}",
                super::planar_fixtures::format_tag(format)
            ),
            format,
            schedule,
            &[k as u32, out_rows as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

        let metadata = [k as u32, out_rows as u32, m as u32, 0, 0]; // [3] x_groups unused (unfolded), [4] tile_offset 0
        let out_len = m * out_rows;
        let mut buffers: Vec<Buffer> = vec![Buffer::from_f32s(&activation)];
        buffers.extend(word_bufs.iter().map(|words| Buffer::from_u32s(words)));
        buffers.push(Buffer::from_u32s(&metadata));
        buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));

        let workgroups = workgroups_covering(&body, schedule.grid_threads(1, m, out_rows));
        run(&body, workgroups, &mut buffers)
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
        let got = buffers.last().unwrap().to_f32s().unwrap();

        assert_close(
            format,
            &format!("planar Contraction {schedule:?}"),
            &got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_for_every_planar_format() {
        for format in every_planar_format() {
            for k in KS {
                for schedule in GEMV_SCHEDULES {
                    run_case(format, k, 11, 1, schedule);
                }
            }
        }
    }

    /// `out_rows = 5` and `m = 3` fit one 4x8 tile; `8x8`, `12x12` and `20x40` put more than one tile on each
    /// axis, so the grid split `g -> (trow, tcol)` depends on `tiles_col = ceil(out / tile_cols)` exactly
    /// (mutants-m4 H6). `K = 20` stages in four-value runs (the shared-scale and sliced-code path), `K = 18` in
    /// runs of two and `K = 17` in single values.
    #[test]
    fn contraction_tiled_matches_reference_for_every_planar_format() {
        for format in every_planar_format() {
            for k in KS {
                for (out_rows, m) in [(5, 3), (8, 8), (12, 12), (20, 40)] {
                    run_case(format, k, out_rows, m, register_tiled(2, k));
                }
            }
        }
    }

    /// Every geometry the planner weighs ([`RegisterTiling::candidates`]): the 8x8 and 16x16 lane grids with 2, 4
    /// and 8 rows and 4 and 8 columns per lane, over row counts that are not multiples of the tile, for every
    /// planar format and a four-value run (`K = 20`). The 2-lane rows above cover the grid arithmetic cheaply; this one covers the shapes that run.
    #[test]
    fn contraction_tiled_matches_reference_at_the_planned_geometries() {
        let caps = poot_target::DeviceCaps::rocm_default();
        let k = 20;
        for format in every_planar_format() {
            for m in [9, 33, 70, 130] {
                let shape = crate::TiledShape {
                    blocks: 1,
                    m_per_block: m,
                    out_per_block: 6,
                    k,
                    layout: WeightLayout::Nk,
                    rows: crate::TiledRows::Dense,
                    terms: crate::StagedTerms::ONE,
                    weight: crate::StagedWeight::Packed(format),
                    activation_bits: 32,
                };
                for tiling in RegisterTiling::candidates(shape, &caps) {
                    run_case(format, k, 6, m, tiling.schedule());
                }
            }
        }
    }

    /// SC-002 (card 658 F6, GPU-free rehearsal): the planar front end's identical block-major rewrite
    /// (`PlanarPlan` through `tiled_body`) gets the same block-diagonal, ragged-`m_per_block` row the block
    /// front end's `contraction_tiled_matches_reference_block_diagonal_ragged` already has - planar
    /// descriptors are admitted for `Contraction` on every backend and `packed_block_diagonal_linear`
    /// does not care about storage, so a planar block-diagonal claim is constructible in production.
    /// `blocks = 3`, `m_per_block = 7` (not a multiple of `tile = 4`), `out_per_block = 5` (also not a
    /// multiple of `tile`) exercise the same ragged row/column edges at once.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_for_every_planar_format() {
        let (blocks, m_per_block, out_per_block) = (3usize, 7usize, 5usize);
        let out_rows = blocks * out_per_block;
        for (format, k) in every_planar_format()
            .into_iter()
            .flat_map(|format| KS.map(|k| (format, k)))
        {
            let (sources, decoded) = fixture(format, out_rows, k, 11);
            let word_bufs = pack_words(&sources);
            let rows = blocks * m_per_block;
            let mut state = 0x2545_f491_4f6c_dd1du64;
            let activation: Vec<f32> = (0..rows * k)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state % 2000) as f32 - 1000.0) / 500.0
                })
                .collect();
            let expected: Vec<f32> = (0..rows * out_per_block)
                .map(|i| {
                    let row_idx = i / out_per_block;
                    let col = i % out_per_block;
                    let b = row_idx / m_per_block;
                    let weight_row = b * out_per_block + col;
                    (0..k)
                        .map(|kk| activation[row_idx * k + kk] * decoded[weight_row * k + kk])
                        .sum()
                })
                .collect();

            let schedule = register_tiled(2, k);
            let body = packed_contraction(
                &format!(
                    "contraction_planar_blockdiag_{}",
                    super::planar_fixtures::format_tag(format)
                ),
                format,
                schedule,
                &[k as u32, out_per_block as u32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
            let out_len = rows * out_per_block;
            let mut buffers: Vec<Buffer> = vec![Buffer::from_f32s(&activation)];
            buffers.extend(word_bufs.iter().map(|words| Buffer::from_u32s(words)));
            buffers.push(Buffer::from_u32s(&metadata));
            buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));

            let workgroups = workgroups_covering(
                &body,
                schedule.grid_threads(blocks, m_per_block, out_per_block),
            );
            run(&body, workgroups, &mut buffers)
                .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers.last().unwrap().to_f32s().unwrap();

            assert_close(
                format,
                &format!("planar Contraction {schedule:?} blocks={blocks}"),
                &got,
                &expected,
            );
        }
    }
}

#[cfg(test)]
mod rocm_tests {
    //! SC-004/SC-005 (card 542a, device tier 1/2): RowGather and Contraction (Gemv,
    //! blocks 1 and 2) for every block-32 format, dispatched on real
    //! ROCm/HSA hardware (gfx1151) - the AMDGCN counterpart of the wgpu row above, through the same
    //! production path every real caller uses (`poot_codegen::compile_kernel` + `kernel_handle`, card 628's
    //! `no_contract` tests' pattern). Skips (does not fail) with no HSA library or GPU agent, matching
    //! `poot-rocm-runtime`'s own device tests; run serial under `/tmp/poot-gpu.lock`.

    use poot_kernel_ir::Body;
    use std::sync::Arc;

    use poot_codegen::CompiledKernel;
    use poot_rocm_runtime::{
        BoundKernarg, HsacoModule, KernelHandle, RocmBuffer, RocmContext, RocmError,
    };
    use poot_runtime_common::BufferRole;
    use poot_target::ElementKind;

    use super::fixtures::{
        DEVICE_ROWS_AND_DEPTH, GEMV_SCHEDULES, GatherCase, assert_close, block32,
        contraction_fixture, device_rows_and_depth_formats, fixture, kquant, kquant_gather_cases,
        pack_words, piecewise, piecewise_run_schedules, register_tiled, schedule_tag,
        with_rows_and_depth,
    };
    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_contraction, packed_kernel};
    use crate::contraction::Schedule;

    fn open_or_skip(test: &str) -> Option<RocmContext> {
        match RocmContext::new() {
            Ok(ctx) => Some(ctx),
            Err(error @ (RocmError::LibraryNotFound(_) | RocmError::NoGpuAgent)) => {
                poot_runtime_common::device_skip::open_or_skip::<(), _>(
                    poot_runtime_common::DeviceBackend::Rocm,
                    Err(error),
                );
                None
            }
            Err(error) => panic!("{test}: {error}"),
        }
    }

    /// Compile `body` for `ctx`'s real device arch through the same production path every real caller
    /// uses, then load it (mirrors `poot-rocm-runtime`'s own `compile_and_load` test helper).
    fn compile_and_load(
        ctx: &RocmContext,
        name: &str,
        body: &Body,
    ) -> (Arc<HsacoModule>, CompiledKernel) {
        let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
            .unwrap_or_else(|error| panic!("{name}: could not classify device arch: {error}"));
        let target = poot_codegen::Target::AmdGcn(arch);
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004-rocm")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, target);
        let artifact = poot_codegen::compile_kernel(body, poot_target::Backend::AmdGcn(arch), &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        let kernel = poot_codegen::kernel_handle(body, &artifact).unwrap();
        let module = ctx
            .load_hsaco(&kernel)
            .unwrap_or_else(|error| panic!("{name}: load_hsaco: {error}"));
        (module, kernel)
    }

    /// Upload `u32` data through the f32-sized allocator (same 4-byte element width; a raw byte copy):
    /// `poot-rocm-runtime`'s public transfer API has no `u32` variant, only `f32`. `f32::from_bits`/`to_bits`
    /// round-trips every u32 bit pattern exactly (a total, lossless reinterpretation, never a numeric
    /// conversion), so uploading through the f32 path and reading the raw bits back device-side (the
    /// kernel's own `Ty::U32` params) is exact.
    fn upload_u32(ctx: &RocmContext, data: &[u32]) -> RocmBuffer {
        let as_f32: Vec<f32> = data.iter().map(|&word| f32::from_bits(word)).collect();
        ctx.upload_f32(&as_f32, BufferRole::Input)
            .expect("upload u32 buffer")
    }

    /// One `(ptr: u64, len: u64)` kernarg pair per buffer, in the body's declared param order
    /// (`RocmContext::dispatch`'s doc: `(ptr, i64 len, ptr, i64 len, ...)`).
    fn bind_kernarg(
        ctx: &RocmContext,
        kernel: &KernelHandle,
        buffers: &[&RocmBuffer],
    ) -> BoundKernarg {
        let slots: Vec<(&RocmBuffer, u64)> = buffers
            .iter()
            .map(|&buffer| (buffer, buffer.elem_count() as u64))
            .collect();
        ctx.bind_kernarg(kernel, &slots).expect("bind kernarg")
    }

    /// One buffer typed by its element kind, in the body's declared param order; `In` inputs are
    /// uploaded, `Out(len)` is allocated uninitialized and downloaded back after dispatch.
    enum Arg<'a> {
        U32(&'a [u32]),
        F32(&'a [f32]),
        Out(usize),
    }

    /// `(grid, block)` for a "one thread per element" body (`RowGather`): `block` must
    /// be the body's own compiled `workgroup_size` (AMDGCN, like SPIR-V, bakes it into the compiled
    /// kernel; a dispatch with a different block size silently changes what `ThreadIndexCall`'s
    /// global-index formula computes, not just the physical launch shape - card 542a ROCm device rows,
    /// caught by this exact mismatch: `block=(1,1,1)` against a baked `workgroup_size=[64,1,1]` put
    /// every thread's `X` index on a stride of 64, not 1). `grid` is `threads` rounded up to a whole
    /// number of workgroups; the body's own bounds guard masks the over-launched tail.
    fn one_thread_per_element_launch(
        body: &Body,
        threads: usize,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let block_x = body.workgroup_size[0];
        let grid_x = (threads as u32).div_ceil(block_x) * block_x;
        ((grid_x, 1, 1), (block_x, 1, 1))
    }

    /// Compile, load, dispatch and download one packed-kernel body on real ROCm/HSA hardware:
    /// `args` in the body's declared param order (the last entry must be `Arg::Out`).
    fn dispatch_and_download(
        ctx: &RocmContext,
        name: &str,
        body: &Body,
        args: &[Arg<'_>],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Vec<f32> {
        let (module, compiled) = compile_and_load(ctx, name, body);
        let kernel = ctx
            .lookup_kernel(&module, compiled.entry_point())
            .unwrap_or_else(|error| panic!("{name}: lookup_kernel: {error}"));

        let mut owned: Vec<RocmBuffer> = Vec::with_capacity(args.len());
        let mut elements: Vec<ElementKind> = Vec::with_capacity(args.len());
        let mut out_len = 0;
        for arg in args {
            match *arg {
                Arg::U32(data) => {
                    owned.push(upload_u32(ctx, data));
                    elements.push(ElementKind::I32);
                }
                Arg::F32(data) => {
                    owned.push(
                        ctx.upload_f32(data, BufferRole::Input)
                            .expect("upload f32 buffer"),
                    );
                    elements.push(ElementKind::F32);
                }
                Arg::Out(len) => {
                    out_len = len;
                    owned.push(
                        ctx.allocate_f32(len, BufferRole::Output)
                            .expect("allocate out buffer"),
                    );
                    elements.push(ElementKind::F32);
                }
            }
        }
        let refs: Vec<&RocmBuffer> = owned.iter().collect();
        let kernarg = bind_kernarg(ctx, &kernel, &refs);
        ctx.dispatch(&kernel, &kernarg, &elements, grid, block)
            .unwrap_or_else(|error| panic!("{name}: dispatch: {error}"));
        let mut got = vec![0.0f32; out_len];
        ctx.download_f32(owned.last().expect("Arg::Out present"), &mut got)
            .unwrap_or_else(|error| panic!("{name}: download: {error}"));
        got
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_rocm() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_rocm") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        for (seed, format) in block32().into_iter().enumerate() {
            let (k, bytes, decoded) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let expected: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    decoded[id as usize * k..(id as usize + 1) * k]
                        .iter()
                        .copied()
                })
                .collect();
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_rocm_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let words = pack_words(&bytes);
            let (grid, block) = one_thread_per_element_launch(&body, ids.len() * k);
            let got = dispatch_and_download(
                &ctx,
                &name,
                &body,
                &[
                    Arg::U32(&words),
                    Arg::U32(&ids),
                    Arg::U32(&[k as u32]),
                    Arg::Out(ids.len() * k),
                ],
                grid,
                block,
            );
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: RowGather (ROCm) vs decode_blocks rows"
            );
        }
    }

    fn run_contraction_case(
        ctx: &RocmContext,
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        let (_, _, _, expected) = contraction_fixture(format, blocks, m_per_block, out_per_block);
        let got = contraction_on_rocm(
            ctx,
            format,
            schedule,
            poot_graph_ir::OperandPrecision::Exact,
            (blocks, m_per_block, out_per_block),
        );
        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks} (ROCm)"),
            &got,
            &expected,
        );
    }

    /// The outputs of the packed contraction `schedule` (reading its operands at `precision`) over
    /// [`contraction_fixture`] on the device.
    fn contraction_on_rocm(
        ctx: &RocmContext,
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        precision: poot_graph_ir::OperandPrecision,
        (blocks, m_per_block, out_per_block): (usize, usize, usize),
    ) -> Vec<f32> {
        let (k, bytes, activation, _) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;
        let name = format!(
            "contraction_rocm_{format:?}_{}_{blocks}_{}",
            schedule_tag(schedule),
            precision
                .to_string()
                .replace(|c: char| !c.is_ascii_alphanumeric(), "_")
        );
        let body = packed_contraction(
            &name,
            format,
            schedule,
            &[k as u32, out_per_block as u32],
            None,
            crate::PackedRows::Blocks,
            precision,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `tiled_body`'s
        // doc). metadata[4] (tile_offset) is 0: each of these runs one unsplit dispatch.
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
        let out_len = rows * out_per_block;
        let grid = (
            schedule.grid_threads(blocks, m_per_block, out_per_block) as u32,
            1,
            1,
        );
        let block = (body.workgroup_size[0], 1, 1);
        dispatch_and_download(
            ctx,
            &name,
            &body,
            &[
                Arg::F32(&activation),
                Arg::U32(&words),
                Arg::U32(&metadata),
                Arg::Out(out_len),
            ],
            grid,
            block,
        )
    }

    /// POOT-1436 (device tier 2): a packed contraction on the matrix unit. The fragment schedule decodes each
    /// block format into narrow operand tiles and multiplies them on the unit; it must agree with the scalar
    /// tiling of the same `Narrow(bf16)` and `Split(bf16)` contraction (same declared rounding, the scalar
    /// micro-tile's FMA order) to the matrix unit's accumulation order, for every block-32 and K-quant format at
    /// ragged shapes and block-diagonal blocks. Mutation: in `MicroTile::spill`/`elements` swap a lane map
    /// (`WmmaElemRead` index `ev ^ 1`); every format goes red.
    #[test]
    fn packed_fragment_contractions_match_the_scalar_tiling_of_the_same_precision_on_rocm() {
        use crate::TileSize;
        use poot_graph_ir::{FragmentDtype, OperandPrecision};
        use poot_target::{FragmentLayout, LaneMap};
        let Some(ctx) = open_or_skip(
            "packed_fragment_contractions_match_the_scalar_tiling_of_the_same_precision_on_rocm",
        ) else {
            return;
        };
        // The gfx11 matrix unit: the declared RDNA3 lane map. A ROCm device without it has no fragment to run.
        let has_unit = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
            .is_ok_and(|arch| !arch.matrix_units().is_empty());
        if !has_unit {
            poot_test_util::skip_report("poot-kernelgen", &"the device has no matrix unit");
            return;
        }
        let fragment = |k: usize| Schedule::FragmentTiled {
            waves_m: TileSize::new(2).unwrap(),
            waves_n: TileSize::new(1).unwrap(),
            tile: crate::FragmentTile {
                dtype: FragmentDtype::Bf16,
                layout: FragmentLayout::Declared(LaneMap::Rdna3),
                rows: 1,
                cols: 2,
            },
            run: crate::contraction::RunLength::dividing(k),
            depth: 32,
        };
        for format in block32().into_iter().chain(kquant()) {
            for precision in [
                OperandPrecision::Narrow(FragmentDtype::Bf16),
                OperandPrecision::Split(FragmentDtype::Bf16),
            ] {
                for shape in [(1, 19, 21), (3, 7, 5)] {
                    let (k, ..) = contraction_fixture(format, shape.0, shape.1, shape.2);
                    let scalar =
                        contraction_on_rocm(&ctx, format, register_tiled(2, k), precision, shape);
                    let device = contraction_on_rocm(&ctx, format, fragment(k), precision, shape);
                    // The same declared rounding, summed in two orders: the matrix unit's internal accumulation is
                    // coarser than sequential f32 FMAs (the design's section 1.3 measured the same). Measured on
                    // gfx1151: at most 5e-4 under `Narrow`, and 6e-3 under `Split` (Q6_K, one block, where a split
                    // sum cancels to f32 accuracy and so shows the accumulation order most); the dense rows hold
                    // the same two precisions to the declared oracle at 2e-7, so the bound here is 1e-2.
                    let scaled = |values: &[f32]| -> Vec<f32> {
                        values
                            .iter()
                            .zip(&scalar)
                            .map(|(v, s)| v / s.abs().max(1.0))
                            .collect()
                    };
                    let worst = poot_test_util::max_abs_error(&scaled(&device), &scaled(&scalar));
                    eprintln!(
                        "{format:?} {precision} blocks={}: fragment vs scalar {worst:.2e}",
                        shape.0
                    );
                    assert!(
                        worst <= 1e-2,
                        "{format:?} fragment vs scalar {precision} blocks={}: {worst:.2e}",
                        shape.0
                    );
                }
            }
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_rocm() {
        let Some(ctx) = open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_rocm")
        else {
            return;
        };
        for format in block32() {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1560 (device tier 2): the ROCm counterpart of
    /// `contraction_tests::a_gemv_with_rows_and_depth_matches_reference_on_wgpu`.
    #[test]
    fn a_gemv_with_rows_and_depth_matches_reference_on_rocm() {
        let Some(ctx) = open_or_skip("a_gemv_with_rows_and_depth_matches_reference_on_rocm") else {
            return;
        };
        for format in device_rows_and_depth_formats() {
            for (rows, depth) in DEVICE_ROWS_AND_DEPTH {
                let schedule = with_rows_and_depth(GEMV_SCHEDULES[1], rows, depth);
                run_contraction_case(&ctx, format, schedule, 2, 1, 12);
            }
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_rocm_kquant() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_rocm_kquant") else {
            return;
        };
        for GatherCase {
            format,
            label,
            k,
            bytes,
            ids,
            expected,
        } in kquant_gather_cases()
        {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_rocm_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let words = pack_words(&bytes);
            let (grid, block) = one_thread_per_element_launch(&body, ids.len() * k);
            let got = dispatch_and_download(
                &ctx,
                &name,
                &body,
                &[
                    Arg::U32(&words),
                    Arg::U32(&ids),
                    Arg::U32(&[k as u32]),
                    Arg::Out(ids.len() * k),
                ],
                grid,
                block,
            );
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?} ({label}): RowGather (ROCm) vs decode_blocks rows"
            );
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_rocm_kquant() {
        let Some(ctx) =
            open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_rocm_kquant")
        else {
            return;
        };
        for format in kquant() {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1542 (device tier 2): the piecewise formats' runs of 4, 8 and 16 on real ROCm/HSA hardware, the counterpart of
    /// `runs_over_a_piecewise_layout_decode_like_the_reference`.
    #[test]
    fn runs_over_a_piecewise_layout_decode_like_the_reference_on_rocm() {
        let Some(ctx) =
            open_or_skip("runs_over_a_piecewise_layout_decode_like_the_reference_on_rocm")
        else {
            return;
        };
        for format in piecewise() {
            for schedule in piecewise_run_schedules() {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2): `Contraction` `Schedule::RegisterTiled` for every block-32 and
    /// K-quant format on real ROCm/HSA hardware - the ROCm counterpart of
    /// `contraction_tests::contraction_tiled_matches_reference_aligned_and_ragged_on_wgpu`
    /// (`blocks == 1` only; see its doc).
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged_on_rocm() {
        let Some(ctx) =
            open_or_skip("contraction_tiled_matches_reference_aligned_and_ragged_on_rocm")
        else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_contraction_case(&ctx, format, register_tiled(2, 32), 1, 4, 4);
            run_contraction_case(&ctx, format, register_tiled(2, 32), 1, 7, 7);
        }
    }

    /// SC-001/SC-002 (card 658, device tier 2, ROCm/HSA): the real-ROCm-hardware counterpart of
    /// `contraction_tiled_matches_reference_block_diagonal_ragged` (see its doc).
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_on_rocm() {
        let Some(ctx) =
            open_or_skip("contraction_tiled_matches_reference_block_diagonal_ragged_on_rocm")
        else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_contraction_case(&ctx, format, register_tiled(2, 32), 3, 7, 5);
        }
    }

    /// POOT-1626 (device tier 2, ROCm/HSA): the real-hardware counterpart of
    /// `a_tiled_packed_contraction_applies_its_epilogue_once_to_each_completed_sum`: the block-diagonal tiled schedule
    /// with the bias, tanh, scale and residual epilogue equals the reference product with the chain applied on the
    /// host, over two blocks of a ragged row tile and column edge.
    #[test]
    fn a_tiled_packed_contraction_applies_its_epilogue_on_rocm() {
        let Some(ctx) = open_or_skip("a_tiled_packed_contraction_applies_its_epilogue_on_rocm")
        else {
            return;
        };
        let (blocks, m_per_block, out_per_block) = (2, 7, 11);
        let rows = blocks * m_per_block;
        let bias: Vec<f32> = (0..out_per_block).map(|i| 0.3 - i as f32 * 0.07).collect();
        let residual: Vec<f32> = (0..rows * out_per_block)
            .map(|i| i as f32 * 0.011 - 0.5)
            .collect();
        for format in [
            poot_quant::format::WeightFormat::Q8_0,
            poot_quant::format::WeightFormat::Q4_K,
        ] {
            let (k, bytes, activation, head) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            let activation: Vec<f32> = activation.iter().map(|v| v * 0.02).collect();
            let head: Vec<f32> = head.iter().map(|v| v * 0.02).collect();
            let expected: Vec<f32> = (0..rows * out_per_block)
                .map(|i| {
                    super::contraction_tests::apply_chain(
                        head[i],
                        bias[i % out_per_block],
                        residual[i],
                    )
                })
                .collect();
            let epilogue =
                super::contraction_tests::bias_activation_residual_chain_over(rows, out_per_block);
            let schedule = register_tiled(2, 32);
            let name = format!("tiled_epilogue_rocm_{format:?}");
            let body = packed_contraction(
                &name,
                format,
                schedule,
                &[k as u32, out_per_block as u32],
                Some(&epilogue),
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{format:?}: {error}"));
            let words = pack_words(&bytes);
            let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
            let grid = (
                schedule.grid_threads(blocks, m_per_block, out_per_block) as u32,
                1,
                1,
            );
            let got = dispatch_and_download(
                &ctx,
                &name,
                &body,
                &[
                    Arg::F32(&activation),
                    Arg::U32(&words),
                    Arg::F32(&bias),
                    Arg::F32(&residual),
                    Arg::U32(&metadata),
                    Arg::Out(rows * out_per_block),
                ],
                grid,
                (body.workgroup_size[0], 1, 1),
            );
            assert_close(format, "tiled epilogue (ROCm)", &got, &expected);
        }
    }

    /// SC-002 (card 542c, device tier 2): `Contraction` (Gemv, RegisterTiled) for every planar format is
    /// within tier-2 tolerance of the reference on real ROCm/HSA hardware. Each is built twice, as a device that
    /// does not keep the binary16 conversion decodes it and as the ROCm caps' `RoundingConversions::F16` does
    /// (POOT-1561: a float code shifts into binary16 and its scale takes the power of two), and the two outputs are
    /// equal bit for bit.
    /// Mutation: fold `2^(bias - 14)` (`HalfEmbedding::rescale`); the E4M3 and E2M1 rows differ between the builds.
    #[test]
    fn contraction_matches_reference_for_every_planar_format_on_rocm() {
        let Some(ctx) =
            open_or_skip("contraction_matches_reference_for_every_planar_format_on_rocm")
        else {
            return;
        };
        let out_rows = 5;
        for (format, k) in super::planar_fixtures::every_planar_format()
            .into_iter()
            .flat_map(|format| [17usize, 18, 20].map(|k| (format, k)))
        {
            for schedule in GEMV_SCHEDULES.into_iter().chain([register_tiled(2, k)]) {
                let (sources, decoded) = super::planar_fixtures::fixture(format, out_rows, k, 11);
                let word_bufs = super::planar_fixtures::pack_words(&sources);
                let m = match schedule {
                    Schedule::Gemv { .. } => 1,
                    _ => 3,
                };
                let mut state = 0x2545_f491_4f6c_dd1du64;
                let activation: Vec<f32> = (0..m * k)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state % 2000) as f32 - 1000.0) / 500.0
                    })
                    .collect();
                let expected: Vec<f32> = (0..m * out_rows)
                    .map(|i| {
                        let row = i / out_rows;
                        let col = i % out_rows;
                        (0..k)
                            .map(|kk| activation[row * k + kk] * decoded[col * k + kk])
                            .sum()
                    })
                    .collect();

                let mut outputs: Vec<Vec<f32>> = Vec::new();
                for (conversions, tag) in [
                    (poot_target::RoundingConversions::NONE, "exact"),
                    (poot_target::RoundingConversions::F16, "f16"),
                ] {
                    let name = format!(
                        "contraction_planar_rocm_{}_{}_{tag}",
                        super::planar_fixtures::format_tag(format),
                        schedule_tag(schedule)
                    );
                    let body = packed_contraction(
                        &name,
                        format,
                        schedule,
                        &[k as u32, out_rows as u32],
                        None,
                        crate::PackedRows::Blocks,
                        poot_graph_ir::OperandPrecision::Exact,
                        conversions,
                    )
                    .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
                    let metadata = [k as u32, out_rows as u32, m as u32, 0, 0]; // [3] x_groups unused (unfolded), [4] tile_offset 0
                    let out_len = m * out_rows;
                    let grid = (schedule.grid_threads(1, m, out_rows) as u32, 1, 1);
                    let block = (body.workgroup_size[0], 1, 1);
                    let mut args: Vec<Arg<'_>> = vec![Arg::F32(&activation)];
                    args.extend(word_bufs.iter().map(|w| Arg::U32(w)));
                    args.push(Arg::U32(&metadata));
                    args.push(Arg::Out(out_len));
                    let got = dispatch_and_download(&ctx, &name, &body, &args, grid, block);
                    assert_close(
                        format,
                        &format!("planar Contraction {schedule:?} {tag} (ROCm)"),
                        &got,
                        &expected,
                    );
                    outputs.push(got);
                }
                assert_eq!(
                    outputs[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    outputs[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{format:?} {schedule:?}: the decode that keeps the f16 conversion (the shift into binary16 for a float code) is the exact decode, bit for bit"
                );
            }
        }
    }

    /// POOT-1193 (device tier 2): the wide register-tiled GEMM the planner picks for a prefill linear, a 16x16 lane
    /// grid (a 256-lane workgroup, 8 rows per lane, a 128 x 64 tile) of `E4m3Block128` over 130 rows and 70
    /// columns, runs on real ROCm/HSA hardware and matches the reference: `K = 20` (a four-value run, a `K` tail
    /// of 4 past the 16-value stage) and `K = 32` (whole stages). Mutation: `RegisterTiling::choose` never picks
    /// the wide grid and `wide_tile` refuses the caps.
    #[test]
    fn contraction_wide_register_tile_matches_reference_on_rocm() {
        let Some(ctx) = open_or_skip("contraction_wide_register_tile_matches_reference_on_rocm")
        else {
            return;
        };
        for k in [20usize, 32] {
            let case =
                super::planar_fixtures::wide_tile(k, &poot_target::DeviceCaps::rocm_default());
            let name = format!("contraction_wide_rocm_k{k}");
            let body = packed_contraction(
                &name,
                case.format,
                case.schedule,
                &[k as u32, case.out_rows as u32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{:?}: packed_kernel: {error}", case.format));
            let metadata = [k as u32, case.out_rows as u32, case.m as u32, 0, 0]; // [3] x_groups unused (unfolded), [4] tile_offset 0
            let grid = (
                case.schedule.grid_threads(1, case.m, case.out_rows) as u32,
                1,
                1,
            );
            let block = (body.workgroup_size[0], 1, 1);
            assert_eq!(block.0, 256, "the wide grid is a 256-lane workgroup");
            let mut args: Vec<Arg<'_>> = vec![Arg::F32(&case.activation)];
            args.extend(case.word_bufs.iter().map(|w| Arg::U32(w)));
            args.push(Arg::U32(&metadata));
            args.push(Arg::Out(case.m * case.out_rows));
            let got = dispatch_and_download(&ctx, &name, &body, &args, grid, block);
            assert_close(
                case.format,
                &format!("wide register tile K={k} (ROCm)"),
                &got,
                &case.expected,
            );
        }
    }
}

#[cfg(test)]
mod planar_wgpu_tests {
    //! POOT-1193 (device tier 2): the wide register-tiled GEMM a prefill linear plans, on the local wgpu
    //! (SPIR-V/RADV) device - the wgpu counterpart of `rocm_tests`' wide-tile row.

    use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
    use poot_target::Backend;
    use poot_wgpu_runtime::{Context, KernelBuffer};

    use super::fixtures::assert_close;
    use super::kernel::packed_contraction;
    use super::planar_fixtures::wide_tile;

    /// The widest register tiling `caps` runs (see `wide_tile`) of `E4m3Block128` over 130 rows and 130 columns,
    /// `K = 20` (a four-value run and a `K` tail of 4) and `K = 32`, dispatched on the local RADV device as a
    /// workgroup of `lanes` lanes, matches the reference.
    fn wide_tile_matches_reference_on_radv(caps: &poot_target::DeviceCaps, lanes: u32) {
        let Some(ctx) = poot_runtime_common::device_skip::open_or_skip(
            poot_runtime_common::DeviceBackend::Wgpu,
            Context::new(),
        ) else {
            return;
        };
        for k in [20usize, 32] {
            let case = wide_tile(k, caps);
            let name = format!("contraction_wide_wgpu_{lanes}_k{k}");
            let body = packed_contraction(
                &name,
                case.format,
                case.schedule,
                &[k as u32, case.out_rows as u32],
                None,
                crate::PackedRows::Blocks,
                poot_graph_ir::OperandPrecision::Exact,
                poot_target::RoundingConversions::NONE,
            )
            .unwrap_or_else(|error| panic!("{:?}: packed_kernel: {error}", case.format));
            assert_eq!(body.workgroup_size[0], lanes, "the case's lane grid");
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-1193-wide")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            let artifact = compile_kernel(&body, Backend::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{:?}: compile: {error}", case.format));
            let kernel = kernel_handle(&body, &artifact).unwrap();

            let metadata = [k as u32, case.out_rows as u32, case.m as u32, 0, 0]; // [3] x_groups unused (unfolded), [4] tile_offset 0
            let mut buffers = vec![KernelBuffer::read_only_f32(&case.activation)];
            buffers.extend(
                case.word_bufs
                    .iter()
                    .map(|w| KernelBuffer::read_only_u32(w)),
            );
            buffers.push(KernelBuffer::read_only_u32(&metadata));
            buffers.push(KernelBuffer::write_f32(case.m * case.out_rows));
            let threads = case.schedule.grid_threads(1, case.m, case.out_rows) as u32;
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [threads, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{:?}: dispatch: {error}", case.format));
            let got = buffers.last().unwrap().as_f32();
            assert_close(
                case.format,
                &format!("{lanes}-lane register tile K={k} (wgpu)"),
                got,
                &case.expected,
            );
        }
    }

    /// The guard of the RADV wave64 LDS defect (update 0169: a 256-lane LDS-tiled GEMM came back with output rows 8
    /// and up near zero, the upper waves' LDS writes invisible after the barrier), which does not reproduce on this
    /// body: the widest tiling the RADV caps plan, the 16x16 lane grid of 8x8 register tiles, a 256-lane workgroup of
    /// four waves staging 16.5 KiB of LDS, matches the reference on RADV. Mutation (the defect's own shape): in
    /// `tiled_body`'s stage block, stage zeros from the lanes past the first wave64; the upper rows read zeros and
    /// this row goes red ("element 2080: got 0 want 2460.198").
    #[test]
    fn the_register_tiled_body_runs_256_lane_workgroups_on_radv() {
        wide_tile_matches_reference_on_radv(&poot_test_util::device_caps::wgpu_rdna3_igpu(), 256);
    }
}

#[cfg(test)]
mod ptx_tests {
    //! SC-004/SC-005 (card 542a, device tier 1/2, M4 pod batch): RowGather and
    //! Contraction (Gemv, blocks 1 and 2) for every block-32 format,
    //! dispatched on real NVIDIA/PTX hardware, the NVPTX counterpart of `rocm_tests` (same fixtures,
    //! same tier-1/tier-2 comparisons). `PtxContext::new()` implements the project's device-skip
    //! convention: it returns `Err` (this test then skips, does not fail) with no NVIDIA driver, and
    //! panics instead under `POOT_REQUIRE_PTX=1` (the PTX pod runs), so a pod run that was
    //! supposed to exercise these never silently reports a skip as a pass. No local run is possible on
    //! this box (AMD Strix Halo, no NVIDIA device): these are unexercised until the PTX
    //! pod batch runs them; name them there.
    //!
    //! `PtxContext::dispatch_dev` takes `block` (the body's own compiled `workgroup_size`) and
    //! `threads` (the logical element count) and computes the grid itself
    //! (`ceil(threads/block)`, `context.rs:557-561`) - the exact mismatch the ROCm harness above hit
    //! (an arbitrary dispatch-time block size diverging from what AMDGCN bakes into the compiled
    //! kernel) cannot arise here by construction, since there is no separate block argument to get
    //! wrong: `block` must be `body.workgroup_size`.

    use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
    use poot_kernel_ir::Body;
    use poot_ptx_runtime::{PtxBuffer, PtxContext};

    use super::fixtures::{
        DEVICE_ROWS_AND_DEPTH, GEMV_SCHEDULES, GatherCase, assert_close, block32,
        contraction_fixture, device_rows_and_depth_formats, fixture, kquant, kquant_gather_cases,
        pack_words, register_tiled, schedule_tag, with_rows_and_depth,
    };
    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_contraction, packed_kernel};
    use crate::contraction::Schedule;

    fn open_or_skip(test: &str) -> Option<PtxContext> {
        match PtxContext::new() {
            Ok(ctx) => Some(ctx),
            Err(error) => {
                eprintln!("POOT-SKIP poot-kernelgen: {test} (no PTX device): {error}");
                None
            }
        }
    }

    fn compile_and_load(name: &str, body: &Body) -> poot_codegen::CompiledKernel {
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004-ptx")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = artifact_path(&dir, name, Target::Nvptx);
        let artifact = compile_kernel(body, poot_target::Backend::Nvptx, &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        kernel_handle(body, &artifact).unwrap()
    }

    /// Upload `u32` data through the i32-typed allocator (`u32 as i32` is a bit-preserving
    /// reinterpretation between same-width integers, never a numeric conversion):
    /// `poot-ptx-runtime`'s public upload API has no `u32` variant, only `i32`.
    fn upload_u32(ctx: &PtxContext, data: &[u32]) -> PtxBuffer {
        let as_i32: Vec<i32> = data.iter().map(|&word| word as i32).collect();
        ctx.upload_i32(&as_i32).expect("upload u32 buffer")
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_ptx() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_ptx") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        for (seed, format) in block32().into_iter().enumerate() {
            let (k, bytes, decoded) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let expected: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    decoded[id as usize * k..(id as usize + 1) * k]
                        .iter()
                        .copied()
                })
                .collect();
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_ptx_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let words = pack_words(&bytes);
            let words_buf = upload_u32(&ctx, &words);
            let ids_buf = ctx
                .upload_i32(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>())
                .expect("upload ids");
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(ids.len() * k).expect("allocate out");
            ctx.dispatch_dev(
                &kernel,
                body.workgroup_size,
                [(ids.len() * k) as u32, 1, 1],
                &[&words_buf, &ids_buf, &metadata_buf],
                &[
                    words_buf.elem_count(),
                    ids_buf.elem_count(),
                    metadata_buf.elem_count(),
                ],
                &out_buf,
                out_buf.elem_count(),
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = ctx.download_f32(&out_buf).expect("download out");
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: RowGather (PTX) vs decode_blocks rows"
            );
        }
    }

    fn run_contraction_case(
        ctx: &PtxContext,
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;
        let name = format!(
            "contraction_ptx_{format:?}_{}_{blocks}",
            schedule_tag(schedule)
        );
        let body = packed_contraction(
            &name,
            format,
            schedule,
            &[k as u32, out_per_block as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let kernel = compile_and_load(&name, &body);
        let words = pack_words(&bytes);
        let words_buf = upload_u32(ctx, &words);
        let activation_buf = ctx.upload_f32(&activation).expect("upload activation");
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `tiled_body`'s
        // doc). metadata[4] (tile_offset) is 0: each of these runs one unsplit dispatch.
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0, 0];
        let metadata_buf = upload_u32(ctx, &metadata);
        let out_len = rows * out_per_block;
        let out_buf = ctx.alloc_f32(out_len).expect("allocate out");
        // Total work-items (`Schedule::grid_threads`; dispatch_dev divides by body.workgroup_size):
        // a Gemv launches `width` lanes per `cols` outputs, not one thread per output - the exact
        // block/workgroup_size mismatch the ROCm harness caught (see `one_thread_per_element_launch`'s
        // doc comment above).
        let threads = schedule.grid_threads(blocks, m_per_block, out_per_block) as u32;
        ctx.dispatch_dev(
            &kernel,
            body.workgroup_size,
            [threads, 1, 1],
            &[&activation_buf, &words_buf, &metadata_buf],
            &[
                activation_buf.elem_count(),
                words_buf.elem_count(),
                metadata_buf.elem_count(),
            ],
            &out_buf,
            out_buf.elem_count(),
        )
        .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
        let got = ctx.download_f32(&out_buf).expect("download out");
        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks} (PTX)"),
            &got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_ptx() {
        let Some(ctx) = open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_ptx")
        else {
            return;
        };
        for format in block32() {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// POOT-1560 (device tier 2): the PTX counterpart of
    /// `contraction_tests::a_gemv_with_rows_and_depth_matches_reference_on_wgpu`.
    #[test]
    fn a_gemv_with_rows_and_depth_matches_reference_on_ptx() {
        let Some(ctx) = open_or_skip("a_gemv_with_rows_and_depth_matches_reference_on_ptx") else {
            return;
        };
        for format in device_rows_and_depth_formats() {
            for (rows, depth) in DEVICE_ROWS_AND_DEPTH {
                let schedule = with_rows_and_depth(GEMV_SCHEDULES[1], rows, depth);
                run_contraction_case(&ctx, format, schedule, 2, 1, 12);
            }
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_ptx_kquant() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_ptx_kquant") else {
            return;
        };
        for GatherCase {
            format,
            label,
            k,
            bytes,
            ids,
            expected,
        } in kquant_gather_cases()
        {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let name = format!("row_gather_ptx_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let words = pack_words(&bytes);
            let words_buf = upload_u32(&ctx, &words);
            let ids_buf = ctx
                .upload_i32(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>())
                .expect("upload ids");
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(ids.len() * k).expect("allocate out");
            ctx.dispatch_dev(
                &kernel,
                body.workgroup_size,
                [(ids.len() * k) as u32, 1, 1],
                &[&words_buf, &ids_buf, &metadata_buf],
                &[
                    words_buf.elem_count(),
                    ids_buf.elem_count(),
                    metadata_buf.elem_count(),
                ],
                &out_buf,
                out_buf.elem_count(),
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = ctx.download_f32(&out_buf).expect("download out");
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?} ({label}): RowGather (PTX) vs decode_blocks rows"
            );
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_ptx_kquant() {
        let Some(ctx) =
            open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_ptx_kquant")
        else {
            return;
        };
        for format in kquant() {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2, M4 pod batch): `Contraction` `Schedule::RegisterTiled` for every
    /// block-32 and K-quant format, the PTX counterpart of
    /// `rocm_tests::contraction_tiled_matches_reference_aligned_and_ragged_on_rocm` (`blocks == 1`
    /// only; see its doc).
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged_on_ptx() {
        let Some(ctx) =
            open_or_skip("contraction_tiled_matches_reference_aligned_and_ragged_on_ptx")
        else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_contraction_case(&ctx, format, register_tiled(2, 32), 1, 4, 4);
            run_contraction_case(&ctx, format, register_tiled(2, 32), 1, 7, 7);
        }
    }

    /// SC-002/F3 (card 658 review, device tier 2, M4 pod batch): `Contraction` `Schedule::RegisterTiled` over
    /// a block-diagonal claim (`blocks = 3`, `m_per_block = 7` not a multiple of `tile = 4`) on real
    /// NVIDIA/PTX hardware - the PTX counterpart of
    /// `contraction_tests::contraction_tiled_matches_reference_block_diagonal_ragged`/
    /// `rocm_tests::contraction_tiled_matches_reference_block_diagonal_ragged_on_rocm`. Every other
    /// Tiled PTX row here is `blocks == 1`; this is the only one exercising the block-major grid this
    /// card changed for NVIDIA.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_on_ptx() {
        let Some(ctx) =
            open_or_skip("contraction_tiled_matches_reference_block_diagonal_ragged_on_ptx")
        else {
            return;
        };
        for format in block32().into_iter().chain(kquant()) {
            run_contraction_case(&ctx, format, register_tiled(2, 32), 3, 7, 5);
        }
    }
}

#[cfg(test)]
mod planar_ptx_tests {
    //! SC-002 (card 542c, device tier 1/2, M4 pod batch): Contraction (Gemv,
    //! Tiled) for every planar format, dispatched on real NVIDIA/PTX hardware - the planar
    //! counterpart of `ptx_tests` (same skip/require convention, same `PtxContext` harness;
    //! `dispatch_dev`'s source-buffer slice is plain `&[&PtxBuffer]`, so a planar format's variable
    //! source count needs no new harness). No local run is possible on this box (AMD Strix Halo, no
    //! NVIDIA device): unexercised until the PTX pod batch runs them.

    use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
    use poot_kernel_ir::Body;
    use poot_ptx_runtime::{PtxBuffer, PtxContext};

    use super::fixtures::{GEMV_SCHEDULES, register_tiled, schedule_tag};
    use super::kernel::packed_contraction;
    use super::planar_fixtures::{every_planar_format, fixture, format_tag, pack_words};
    use crate::contraction::Schedule;

    /// `K` values of a weight row: 17 leaves no run longer than one value, 18 stages in runs of two, 20 in
    /// runs of four.
    const KS: [usize; 3] = [17, 18, 20];

    fn open_or_skip(test: &str) -> Option<PtxContext> {
        match PtxContext::new() {
            Ok(ctx) => Some(ctx),
            Err(error) => {
                eprintln!("POOT-SKIP poot-kernelgen: {test} (no PTX device): {error}");
                None
            }
        }
    }

    fn compile_and_load(name: &str, body: &Body) -> poot_codegen::CompiledKernel {
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc002-planar-ptx")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = artifact_path(&dir, name, Target::Nvptx);
        let artifact = compile_kernel(body, poot_target::Backend::Nvptx, &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        kernel_handle(body, &artifact).unwrap()
    }

    fn upload_u32(ctx: &PtxContext, data: &[u32]) -> PtxBuffer {
        let as_i32: Vec<i32> = data.iter().map(|&word| word as i32).collect();
        ctx.upload_i32(&as_i32).expect("upload u32 buffer")
    }

    fn reference_contraction(
        activation: &[f32],
        decoded: &[f32],
        k: usize,
        out: usize,
    ) -> Vec<f32> {
        let rows = activation.len() / k;
        (0..rows * out)
            .map(|i| {
                let row = i / out;
                let col = i % out;
                (0..k)
                    .map(|kk| activation[row * k + kk] * decoded[col * k + kk])
                    .sum()
            })
            .collect()
    }

    fn run_case(
        ctx: &PtxContext,
        format: poot_quant::format::WeightFormat,
        k: usize,
        out_rows: usize,
        schedule: Schedule,
    ) {
        let (sources, decoded) = fixture(format, out_rows, k, 11);
        let word_bufs = pack_words(&sources);
        let m = match schedule {
            Schedule::Gemv { .. } => 1,
            _ => 3,
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..m * k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected = reference_contraction(&activation, &decoded, k, out_rows);

        let name = format!(
            "contraction_planar_ptx_{}_{}",
            format_tag(format),
            schedule_tag(schedule)
        );
        let body = packed_contraction(
            &name,
            format,
            schedule,
            &[k as u32, out_rows as u32],
            None,
            crate::PackedRows::Blocks,
            poot_graph_ir::OperandPrecision::Exact,
            poot_target::RoundingConversions::NONE,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let kernel = compile_and_load(&name, &body);
        let source_bufs: Vec<PtxBuffer> = word_bufs
            .iter()
            .map(|words| upload_u32(ctx, words))
            .collect();
        let activation_buf = ctx.upload_f32(&activation).expect("upload activation");
        let metadata = [k as u32, out_rows as u32, m as u32, 0, 0]; // [3] x_groups unused (unfolded), [4] tile_offset 0
        let metadata_buf = upload_u32(ctx, &metadata);
        let out_len = m * out_rows;
        let out_buf = ctx.alloc_f32(out_len).expect("allocate out");
        let mut arg_refs: Vec<&PtxBuffer> = vec![&activation_buf];
        arg_refs.extend(source_bufs.iter());
        arg_refs.push(&metadata_buf);
        let arg_lens: Vec<usize> = arg_refs.iter().map(|b| b.elem_count()).collect();
        let threads = schedule.grid_threads(1, m, out_rows) as u32;
        ctx.dispatch_dev(
            &kernel,
            body.workgroup_size,
            [threads, 1, 1],
            &arg_refs,
            &arg_lens,
            &out_buf,
            out_buf.elem_count(),
        )
        .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
        let got = ctx.download_f32(&out_buf).expect("download out");
        super::fixtures::assert_close(
            format,
            &format!("planar Contraction {schedule:?} (PTX)"),
            &got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_for_every_planar_format_on_ptx() {
        let Some(ctx) =
            open_or_skip("contraction_gemv_matches_reference_for_every_planar_format_on_ptx")
        else {
            return;
        };
        for format in every_planar_format() {
            for k in KS {
                for schedule in GEMV_SCHEDULES {
                    run_case(&ctx, format, k, 11, schedule);
                }
            }
        }
    }

    #[test]
    fn contraction_tiled_matches_reference_for_every_planar_format_on_ptx() {
        let Some(ctx) =
            open_or_skip("contraction_tiled_matches_reference_for_every_planar_format_on_ptx")
        else {
            return;
        };
        for format in every_planar_format() {
            for k in KS {
                run_case(&ctx, format, k, 5, register_tiled(2, k));
            }
        }
    }
}

#[cfg(test)]
mod row_gather_index_width_tests {
    use poot_codegen::{Target, emit_llvm_ir};
    use poot_quant::format::WeightFormat;
    use poot_target::AmdArch;

    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_kernel};

    fn ir(target: Target) -> String {
        let spec = PackedKernelSpec {
            format: WeightFormat::Iq4_Nl,
            op: PackedKernelOp::RowGather,
        };
        let body = packed_kernel("row_gather_width", spec).unwrap();
        emit_llvm_ir(&body, target).unwrap()
    }

    /// POOT-1317: the IQ4_NL row gather's block-to-bit multiply (144 bits per block) and every word index
    /// it loads through are `i64` on the backends whose kernels are 64-bit (PTX, ROCm), with no `trunc` of
    /// the offset chain, so a PLE row past `u32::MAX` bits is addressed at its true word. SPIR-V for Vulkan
    /// lowers `usize` to `i32` (its resource index is `i32`): the same multiply is `i32` there, which is why
    /// the planner refuses a packed source past `u32::MAX` bits on that backend
    /// (`check_source_addressable`) instead of letting it wrap.
    ///
    /// Mutation: lower `usize` to `i32` on NVPTX in `usize_llty_of`; the NVPTX `mul i64` assertion goes red.
    #[test]
    fn the_row_gather_offset_chain_is_64_bit_where_the_backend_is() {
        for (target, label) in [
            (Target::Nvptx, "nvptx"),
            (Target::AmdGcn(AmdArch::gfx1151()), "amdgcn"),
        ] {
            let ir = ir(target);
            let multiplies: Vec<&str> = ir
                .lines()
                .filter(|line| line.contains(" = mul ") && line.trim_end().ends_with(", 144"))
                .collect();
            assert!(
                !multiplies.is_empty(),
                "{label}: the block-to-bit multiply is emitted"
            );
            for line in &multiplies {
                assert!(
                    line.contains("= mul i64 "),
                    "{label}: 32-bit block-to-bit multiply: {line}"
                );
            }
            // The words param (`%p1`) is read through `i64` indices only.
            let word_loads: Vec<&str> = ir
                .lines()
                .filter(|line| line.contains("getelementptr inbounds i32, ptr addrspace(1) %p1,"))
                .collect();
            assert!(!word_loads.is_empty(), "{label}: the packed words are read");
            for line in &word_loads {
                assert!(
                    line.contains(", i64 "),
                    "{label}: a packed word read indexed in 32 bits: {line}"
                );
            }
        }
        let vulkan = ir(Target::SpirvVulkan);
        let multiplies: Vec<&str> = vulkan
            .lines()
            .filter(|line| line.contains(" = mul ") && line.trim_end().ends_with(", 144"))
            .collect();
        assert!(!multiplies.is_empty());
        for line in multiplies {
            assert!(
                line.contains("= mul i32 "),
                "SPIR-V for Vulkan multiplies in 32 bits: {line}"
            );
        }
    }
}
