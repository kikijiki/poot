//! The descriptor-driven packed decode emitter (cards 542a/542b, dquant.md D1-D3, D9):
//! `poot-kernelgen` reads a `poot_quant::format::FormatDescriptor` at generation time and emits one
//! KIR decode per `(format, op)`. No kernel source here names a scheme; every per-format fact comes
//! from the descriptor's own constants. 542a covers the block-32 formats and `Schedule::Gemv`; 542b
//! adds the K-quant super-blocks' packed `SubScale`/`SubMin` sub-factors and `Schedule::Tiled` for
//! `M > 1` (card 658: every `M > 1` block-diagonal split too, never `Schedule::Serial`, deleted).

mod decode_plan;
// `crate::helpers::WeightLane` (Card 1007) reads a packed F16 dense weight through the same float
// decode every packed body uses, rather than restating the binary16 layout.
pub(crate) mod float_decode;
mod kernel;
mod planar;

pub use kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};

#[cfg(test)]
mod float_decode_tests {
    //! SC-002 (card 542a): `emit_float_decode` matches `poot_quant::scalar::float_to_f32`'s `to_bits`
    //! for every admitted bit pattern of every `FloatFormat`, run through the KIR interpreter
    //! (tier 1, GPU-free).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::{BasicBlock, Body, Operand, Rvalue, Terminator, Ty};
    use poot_quant::format::FloatFormat;
    use poot_quant::scalar::float_to_f32;
    use poot_test_util::kernel_fixtures::workgroups_covering;

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

    #[test]
    fn e8m0_matches_scalar_for_the_admitted_domain_0_to_254() {
        let body = decode_body(FloatFormat::E8m0);
        for byte in 0..=254u32 {
            let want = float_to_f32(FloatFormat::E8m0, byte).to_bits();
            let got = decode_at(&body, byte);
            assert_eq!(
                got, want,
                "E8M0 byte {byte:#04x}: got {got:#010x} want {want:#010x}"
            );
        }
    }

    #[test]
    fn e2m1_matches_scalar_for_all_16_codes() {
        let body = decode_body(FloatFormat::E2m1);
        for code in 0..16u32 {
            let want = float_to_f32(FloatFormat::E2m1, code).to_bits();
            let got = decode_at(&body, code);
            assert_eq!(got, want, "E2M1 code {code}");
        }
    }

    #[test]
    fn e4m3fn_matches_scalar_for_every_non_nan_byte() {
        let body = decode_body(FloatFormat::E4m3Fn);
        for byte in 0..=255u32 {
            if byte == 0x7f || byte == 0xff {
                continue; // the two NaN bytes: outside the admitted domain (refused at load).
            }
            let want = float_to_f32(FloatFormat::E4m3Fn, byte).to_bits();
            let got = decode_at(&body, byte);
            assert_eq!(got, want, "E4M3FN byte {byte:#04x}");
        }
    }

    #[test]
    fn f16_matches_scalar_for_every_finite_pattern() {
        let body = decode_body(FloatFormat::F16);
        for bits in 0..=0xffffu32 {
            if (bits >> 10) & 0x1f == 0x1f {
                continue; // infinities and NaN: outside the admitted domain (refused at load).
            }
            let want = float_to_f32(FloatFormat::F16, bits).to_bits();
            let got = decode_at(&body, bits);
            assert_eq!(got, want, "F16 bits {bits:#06x}");
        }
    }

    #[test]
    fn bf16_matches_scalar_for_every_pattern() {
        let body = decode_body(FloatFormat::Bf16);
        for bits in 0..=0xffffu32 {
            let want = float_to_f32(FloatFormat::Bf16, bits).to_bits();
            let got = decode_at(&body, bits);
            assert_eq!(got, want, "BF16 bits {bits:#06x}");
        }
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

/// Shared fixtures for every packed-kernel device/interpreter test below: the seven block-32
/// formats, a deterministic admitted-domain byte generator, and the D9 word transport. One
/// implementation so every op's test compares against the identical oracle input shape.
#[cfg(test)]
mod fixtures {
    use poot_quant::format::{FieldEncoding, FloatFormat, OperandRole, Storage, WeightFormat};

    use crate::contraction::{Schedule, TileSize};

    /// A codegen-safe (LLVM-identifier-safe) name fragment for a schedule: `{schedule:?}` on
    /// `Schedule::Gemv { .. }` contains `{`, `}` and a space, which breaks an AMDGCN/NVPTX global
    /// or symbol name built from it (card 542a ROCm device rows).
    pub(super) fn schedule_tag(schedule: Schedule) -> String {
        match schedule {
            Schedule::Gemv {
                width,
                cols,
                unroll,
            } => format!("gemv{width}c{cols}u{unroll}"),
            Schedule::Tiled {
                tile: TileSize(tile),
            } => format!("tiled{tile}"),
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
        },
        Schedule::Gemv {
            width: 32,
            cols: 4,
            unroll: 4,
        },
        Schedule::Gemv {
            width: 64,
            cols: 8,
            unroll: 3,
        },
        Schedule::Gemv {
            width: 32,
            cols: 2,
            unroll: 32,
        },
    ];

    pub(super) const BLOCK32: [WeightFormat; 7] = [
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q5_0,
        WeightFormat::Q5_1,
        WeightFormat::Q8_0,
        WeightFormat::Iq4_Nl,
        WeightFormat::Mxfp4,
    ];

    /// The six K-quant super-block formats (card 542b): packed 6-bit `SubScale`/`SubMin` sub-factors
    /// (Q2_K, Q4_K, Q5_K two-piece each; Q3_K, IQ4_XS `SubScale` two-piece; Q6_K `SubScale` one
    /// piece), 256-value super-blocks (`blocks::QK_K`).
    pub(super) const KQUANT: [WeightFormat; 6] = [
        WeightFormat::Q2_K,
        WeightFormat::Q3_K,
        WeightFormat::Q4_K,
        WeightFormat::Q5_K,
        WeightFormat::Q6_K,
        WeightFormat::Iq4_Xs,
    ];

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

    /// Force every block's `Scale`/`Min` field into the admitted (non-NaN) domain: xorshift bytes
    /// otherwise land on E8M0 `0xff` or an F16 infinity/NaN roughly one seed in a few hundred, which
    /// `poot_quant::plan::decode_blocks` (correctly) refuses. Codes never need this: none of the
    /// block-32 formats' codes fields can encode a NaN (E2M1 has none; the rest are integers).
    pub(super) fn sanitize_factors(format: WeightFormat, block: &mut [u8]) {
        let Storage::Blocks(layout) = format.descriptor().storage else {
            unreachable!()
        };
        for role in [OperandRole::Scale, OperandRole::Min] {
            let Some(field) = layout.field(role) else {
                continue;
            };
            let bit = field.field.pieces[0].layout.bit(0);
            assert_eq!(bit % 8, 0, "{format:?} {role:?} must be byte-aligned");
            let byte0 = (bit / 8) as usize;
            match field.field.encoding {
                FieldEncoding::Float(FloatFormat::F16) => {
                    // Clear the top exponent bit: exponent <= 0b01111, never 0x1f (inf/NaN).
                    block[byte0 + 1] &= !0x40;
                }
                FieldEncoding::Float(FloatFormat::E8m0) => {
                    // 0xff is the only non-admitted byte, so clamping below it is exact.
                    block[byte0] = block[byte0].min(0xfe);
                }
                _ => {}
            }
        }
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
            sanitize_factors(format, block);
        }
        let mut expected = vec![0.0f32; out_rows * k];
        poot_quant::plan::decode_blocks(format, &bytes, &mut expected)
            .unwrap_or_else(|error| panic!("{format:?}: fixture must decode: {error}"));
        (k, bytes, expected)
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
    #[allow(clippy::type_complexity)]
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
        FieldEncoding, FloatFormat, GroupMap, OperandRole, ScaleEncoding, WeightFormat,
    };
    use poot_quant::{PackedWeight, SourceRole};

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    /// The six `E4M3` scale encodings (card 542c): per-channel and 128x128-block, each F32, BF16 and
    /// E8M0.
    pub(super) const E4M3: [WeightFormat; 6] = [
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::E8m0,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
    ];

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

    /// Every planar format SC-001 (card 542c) names: the six E4M3 scale encodings, E2M1 row-32,
    /// GPTQ contiguous and act-order, and AWQ.
    pub(super) fn every_planar_format() -> Vec<WeightFormat> {
        E4M3.into_iter()
            .chain([WeightFormat::E2m1Row32])
            .chain([gptq_contiguous(), gptq_indexed(2), awq()])
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
                let bytes = if operand_role == OperandRole::GroupIndex {
                    let WeightFormat::Gptq {
                        groups: GroupMap::Indexed { groups },
                    } = format
                    else {
                        unreachable!("format tests: only act-order GPTQ has a GroupIndex operand")
                    };
                    (0..k)
                        .flat_map(|kk| ((kk % groups.get()) as i32).to_le_bytes())
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
        let source_refs: Vec<(OperandRole, &[u8])> = sources
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
mod materialize_tests {
    //! SC-001 (card 542a): `packed_kernel(Materialize)` run in the KIR interpreter equals
    //! `poot_quant::plan::decode_blocks`'s `to_bits` on every element, for every block-32 format
    //! (tier 1, GPU-free).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::fixtures::{BLOCK32, KQUANT, fixture, pack_words};
    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_kernel};

    /// SC-001 (card 542b): `packed_kernel(Materialize)` run in the KIR interpreter equals
    /// `poot_quant::plan::decode_blocks`'s `to_bits` on every element, for every K-quant super-block
    /// format (tier 1, GPU-free). The two-piece `SubScale`/`SubMin` sub-factors are exercised here for
    /// the first time (542a's block-32 formats have none). Mutation: drop `SubMin`'s high piece in
    /// `DecodePlan::emit_factors` (`Field { pieces: &field.pieces[..1], .. }`); the Q4_K and Q5_K rows
    /// go red (their `dmin*m` term reads only the low nibble of an 8-value range, decoding a wrong
    /// min for every sub-block whose true `m` needs the high two bits).
    #[test]
    fn materialize_matches_decode_blocks_for_every_kquant_format() {
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in KQUANT.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let body = packed_kernel(&format!("materialize_kquant_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let mut buffers = [
                Buffer::from_u32s(&words),
                Buffer::from_u32s(&[k as u32]),
                Buffer::from_f32s(&vec![0.0; out_rows * k]),
            ];
            run(
                &body,
                workgroups_covering(&body, out_rows * k),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers[2].to_f32s().unwrap();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize vs decode_blocks"
            );
        }
    }

    /// Sanity only (not acceptance, as the block-32 sibling above): every K-quant `Materialize` body
    /// compiles for SPIR-V and NVPTX.
    #[test]
    fn materialize_bodies_compile_for_kquant_spirv_and_nvptx() {
        for format in KQUANT {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let body = packed_kernel(&format!("materialize_kquant_{format:?}"), spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("materialize_kquant_{format:?}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("materialize_kquant_{format:?}_ptx"));
        }
    }

    #[test]
    fn materialize_bodies_compile_for_kquant_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in KQUANT {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_kquant_{format:?}_amdgcn");
            let body = packed_kernel(&name, spec).unwrap();
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?}: AMDGCN compile: {error}"));
        }
    }

    /// SC-002 (card 542b, device tier 1): `Materialize` for every K-quant format equals the oracle's
    /// `to_bits` on the local wgpu (SPIR-V/RADV) device (the block-32 `materialize_matches_decode_blocks_on_wgpu`
    /// row's K-quant sibling).
    #[test]
    fn materialize_matches_decode_blocks_on_wgpu_kquant() {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::{Context, KernelBuffer};

        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };

        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in KQUANT.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_wgpu_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            compile(&body, Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_u32(&[k as u32]),
                KernelBuffer::write_f32(out_rows * k),
            ];
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * k) as u32, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers[2].as_f32();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (wgpu) vs decode_blocks"
            );
        }
    }

    #[test]
    fn materialize_matches_decode_blocks_for_every_block32_format() {
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in BLOCK32.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let body = packed_kernel(&format!("materialize_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let mut buffers = [
                Buffer::from_u32s(&words),
                Buffer::from_u32s(&[k as u32]),
                Buffer::from_f32s(&vec![0.0; out_rows * k]),
            ];
            run(
                &body,
                workgroups_covering(&body, out_rows * k),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers[2].to_f32s().unwrap();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize vs decode_blocks"
            );
        }
    }

    /// Sanity only (not acceptance): every block-32 `Materialize` body actually lowers through
    /// `poot-codegen` to SPIR-V and NVPTX, so a KIR construct this module relies on (`Ty::U32`
    /// arithmetic, `Bitcast`, the branchless `select_u32` pattern) is not accidentally
    /// interpreter-only. Device execution (SC-004) is separate, GPU/pod evidence.
    #[test]
    fn materialize_bodies_compile_for_spirv_and_nvptx() {
        for format in BLOCK32 {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let body = packed_kernel(&format!("materialize_{format:?}"), spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("materialize_{format:?}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("materialize_{format:?}_ptx"));
        }
    }

    /// Sanity only (as above): every block-32 `Materialize` body also lowers to AMDGCN (HSACO), the
    /// third backend the card's device rows (SC-004) run on.
    #[test]
    fn materialize_bodies_compile_for_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in BLOCK32 {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_{format:?}_amdgcn");
            let body = packed_kernel(&name, spec).unwrap();
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?}: AMDGCN compile: {error}"));
        }
    }

    /// SC-004 (card 542a, device tier 1): `Materialize` for every block-32 format equals the oracle's
    /// `to_bits` on the local wgpu (SPIR-V/RADV) device. Skips (does not fail) with no Vulkan adapter,
    /// matching every other GPU test in this crate (`nix develop` required; run serial under
    /// `/tmp/poot-gpu.lock`).
    #[test]
    fn materialize_matches_decode_blocks_on_wgpu() {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::{Context, KernelBuffer};

        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };

        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in BLOCK32.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_wgpu_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            compile(&body, Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_u32(&[k as u32]),
                KernelBuffer::write_f32(out_rows * k),
            ];
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * k) as u32, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers[2].as_f32();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (wgpu) vs decode_blocks"
            );
        }
    }
}

#[cfg(test)]
mod row_gather_tests {
    //! SC-001 (card 542a, RowGather row): `packed_kernel(RowGather)` run in the KIR interpreter
    //! selects the same rows `poot_quant::plan::decode_blocks` decodes (tier 1, GPU-free).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::fixtures::{BLOCK32, KQUANT, fixture, pack_words};
    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_kernel};

    /// The K-quant sibling of `row_gather_matches_decode_blocks_rows_for_every_block32_format`
    /// (card 542b): exercises `DecodePlan`'s `SubScale`/`SubMin` path through `RowGather` too, not
    /// only `Materialize`.
    #[test]
    fn row_gather_matches_decode_blocks_rows_for_every_kquant_format() {
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in KQUANT.into_iter().enumerate() {
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
            let body = packed_kernel(&format!("row_gather_kquant_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let mut buffers = [
                Buffer::from_u32s(&words),
                Buffer::from_f32s(&ids_f32),
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

    #[test]
    fn row_gather_bodies_compile_for_kquant_spirv_and_nvptx() {
        for format in KQUANT {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_kquant_{format:?}"), spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("row_gather_kquant_{format:?}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("row_gather_kquant_{format:?}_ptx"));
        }
    }

    #[test]
    fn row_gather_bodies_compile_for_kquant_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in KQUANT {
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
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?}: AMDGCN compile: {error}"));
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_wgpu_kquant() {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::{Context, KernelBuffer};

        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };

        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in KQUANT.into_iter().enumerate() {
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
            let name = format!("row_gather_wgpu_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            compile(&body, Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_f32(&ids_f32),
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

    #[test]
    fn row_gather_matches_decode_blocks_rows_for_every_block32_format() {
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in BLOCK32.into_iter().enumerate() {
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
                Buffer::from_f32s(&ids_f32),
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

    /// Sanity only (not acceptance, as `materialize_bodies_compile_for_spirv_and_nvptx`): every
    /// block-32 `RowGather` body lowers to SPIR-V, NVPTX and AMDGCN. Names the SPIR-V/NVPTX artifacts
    /// (`row_gather_<format>_spv`/`_ptx`) for the PTX pod batch (SC-004).
    #[test]
    fn row_gather_bodies_compile_for_spirv_and_nvptx() {
        for format in BLOCK32 {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::RowGather,
            };
            let body = packed_kernel(&format!("row_gather_{format:?}"), spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("row_gather_{format:?}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("row_gather_{format:?}_ptx"));
        }
    }

    #[test]
    fn row_gather_bodies_compile_for_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in BLOCK32 {
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
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?}: AMDGCN compile: {error}"));
        }
    }

    /// SC-004 (card 542a, device tier 1): `RowGather` for every block-32 format equals the oracle's
    /// rows on the local wgpu (SPIR-V/RADV) device (the `materialize_matches_decode_blocks_on_wgpu`
    /// row's sibling).
    #[test]
    fn row_gather_matches_decode_blocks_rows_on_wgpu() {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::{Context, KernelBuffer};

        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };

        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in BLOCK32.into_iter().enumerate() {
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
            compile(&body, Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

            let words = pack_words(&bytes);
            let mut buffers = [
                KernelBuffer::read_only_u32(&words),
                KernelBuffer::read_only_f32(&ids_f32),
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
    //! SC-005 (card 542a, GPU-free rehearsal): `packed_kernel(Contraction)` (Gemv and Tiled,
    //! `RowSelect::Dense`, blocks 1 and 2) run in the KIR interpreter is within tier-2 tolerance of a
    //! plain f32 reference contraction over `poot_quant::plan::decode_blocks`'s decode. Device rows
    //! (wgpu/ROCm/PTX) are separate, real-hardware evidence (the card's final report).

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::fixtures::{
        BLOCK32, GEMV_SCHEDULES, KQUANT, assert_close, contraction_fixture, pack_words,
        schedule_tag,
    };
    use super::kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};
    use crate::contraction::{Schedule, TileSize};

    fn run_case(
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;

        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let body = packed_kernel(
            &format!("contraction_{format:?}_{}_{blocks}", schedule_tag(schedule)),
            spec,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `contraction_tiled`'s
        // doc).
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0];
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
        let got = buffers[3].to_f32s().unwrap();

        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks}"),
            &got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2() {
        for format in BLOCK32 {
            for schedule in GEMV_SCHEDULES {
                run_case(format, schedule, 1, 1, 11);
                run_case(format, schedule, 2, 1, 5);
            }
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_kquant() {
        for format in KQUANT {
            for schedule in GEMV_SCHEDULES {
                run_case(format, schedule, 1, 1, 11);
                run_case(format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, GPU-free rehearsal): `Contraction` `Schedule::Tiled` (`tile = 4`), `blocks =
    /// 1`, for every block-32 and K-quant format - the new schedule 542b adds, over both format
    /// families (`Schedule::Tiled` carries no per-format code, so a block-32 format that never needed
    /// tiling before is as good a witness as a K-quant one). `m_per_block` (`out_per_block` follows it
    /// 1:1 in this fixture, `contraction_fixture`'s doc) of 4 (tile-aligned) and 7 (ragged, three rows
    /// short of two full tiles) against `tile = 4` exercise both the aligned and the ragged row/column
    /// tile edges. 8 and 12 put more than one tile on each axis with `out_per_block` a multiple of
    /// the tile, so the grid split `g -> (trow, tcol)` depends on `tiles_col = ceil(out / ts)`
    /// exactly (mutants-m4 H6: at 4 there is one tile, and at 7 `(7 + 4) / 4 == (7 + 3) / 4`).
    ///
    /// Card 658: `blocks == 1` only through card 542b/545b - a `blocks > 1` (ADR-0109 block-diagonal)
    /// claim could have a row-tile straddle two blocks whenever `m_per_block` was not a multiple of
    /// `tile`, so a straddling tile would read the wrong block's weight for some of its rows and the
    /// planner fell back to the per-thread-unbounded `Schedule::Serial` instead (since deleted:
    /// nothing plans it any more). `contraction_tiled` now lays its tiles out per block (`b = g /
    /// tiles_per_block`, fixed per workgroup), so this test's `blocks == 1` cases stay green unchanged.
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged() {
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_case(
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                4,
                4,
            );
            run_case(
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                7,
                7,
            );
            run_case(
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                8,
                8,
            );
            run_case(
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                12,
                12,
            );
        }
    }

    /// SC-002 (card 658, GPU-free rehearsal): `Contraction` `Schedule::Tiled` over a block-diagonal
    /// claim (`blocks = 3`) whose `m_per_block = 7` is not a multiple of `tile = 4` - the exact shape
    /// `try_plan_contraction` used to refuse to `Schedule::Tiled` and fall back to the per-thread-
    /// unbounded `Schedule::Serial` for (`Serial` since deleted). Every
    /// output is checked against
    /// `contraction_fixture`'s block-diagonal reference (`reference_contraction`'s `b = row_idx /
    /// m_per_block`, a fresh partial sum per block), so a tile that reused another block's decoded
    /// weight row - the straddling hazard `contraction_tiled`'s old doc named - would show up as a
    /// wrong value on every block boundary row, not just an out-of-range access. `out_per_block = 5`
    /// (also not a multiple of `tile`) exercises the ragged column edge at the same time.
    /// Mutation: in `contraction_tiled`/`contraction_tiled_planar`, replace the tile-fixed `b` with the
    /// old per-row derivation (`row.min(rows - 1) / m_per_block`, `row` from the global `(trow, tcol)`
    /// tile index rather than block-major `g`) - the row-tile at the block-2/block-3 boundary mixes
    /// block 1's weight into some of block 2's rows, and this test goes red.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged() {
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_case(
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                3,
                7,
                5,
            );
        }
    }

    /// SC-001 (card 658, GPU-free): dispatches the exact folded geometry
    /// `try_plan_contraction` would plan for `blocks = 1, m_per_block = 20, out_per_block = 4, tile =
    /// 4` on a device whose grid cap folds `tiles = ceil(20/4)*ceil(4/4) = 5` onto `(x_groups,
    /// y_groups) = (4, 2)` (`metadata[3] = 4`) - `workgroups = [4, 2, 1]`, i.e. 8 workgroups for 5
    /// real tiles. The 3 padding workgroups (`g in 5..8`) compute `b = g / tiles_per_block >= blocks`
    /// (`tiles_per_block = 5` here, so `g in {5,6,7}` all land at `b = 1`), and the interpreter - unlike
    /// wgpu/RADV's robust-buffer-access, which silently drops the same out-of-bounds store
    /// (`contraction_tiled_folded_grid_matches_reference_on_wgpu` runs this identical geometry on real
    /// hardware without ever catching it) - hard-faults on it.
    ///
    /// Mutation: drop the `rows_ok` term from `storeok` (restore `storeok = rowok2 & colok2` alone, as
    /// before this fix). Red: `InterpError::Fault { fault: OutOfBounds { .. }, .. }` ("buffer index 80
    /// is out of bounds for length 80" for a `rows=20, out_per_block=4` Q4_0 fixture). Restored; green.
    #[test]
    fn contraction_tiled_folded_grid_padding_workgroups_stay_in_bounds() {
        let (blocks, m_per_block, out_per_block, tile) = (1usize, 20usize, 4usize, 4u32);
        let ts = tile as usize;
        let tiles = blocks * m_per_block.div_ceil(ts) * out_per_block.div_ceil(ts);
        let (x_groups, y_groups) = (4u32, 2u32);
        assert_eq!(tiles, 5, "fixture shape must need exactly 5 tiles");
        assert!(
            (x_groups as usize) * (y_groups as usize) > tiles,
            "the fold must dispatch at least one padding workgroup for this to be a real test"
        );
        for format in BLOCK32.into_iter().chain(KQUANT) {
            let (k, bytes, activation, expected) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            let schedule = Schedule::Tiled {
                tile: TileSize::new(tile).unwrap(),
            };
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Contraction {
                    rows: RowSelect::Dense,
                    schedule,
                },
            };
            let body = packed_kernel(&format!("contraction_folded_interp_{format:?}"), spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let words = pack_words(&bytes);
            let metadata = [k as u32, out_per_block as u32, m_per_block as u32, x_groups];
            let rows = blocks * m_per_block;
            let out_len = rows * out_per_block;
            let mut buffers = [
                Buffer::from_f32s(&activation),
                Buffer::from_u32s(&words),
                Buffer::from_u32s(&metadata),
                Buffer::from_f32s(&vec![0.0; out_len]),
            ];
            run(&body, [x_groups, y_groups, 1], &mut buffers).unwrap_or_else(|error| {
                panic!("{format:?}: interpreter: folded grid [4,2,1] (tiles=5): {error}")
            });
            let got = buffers[3].to_f32s().unwrap();
            assert_close(
                format,
                &format!("Contraction {schedule:?} folded grid=[4,2,1]"),
                &got,
                &expected,
            );
        }
    }

    /// Sanity only (not acceptance, as `materialize_bodies_compile_for_spirv_and_nvptx`): every
    /// block-32 `Contraction` (Gemv) body lowers to SPIR-V, NVPTX and AMDGCN. Names the SPIR-V/NVPTX
    /// artifacts for the PTX pod batch (SC-005).
    #[test]
    fn contraction_bodies_compile_for_spirv_nvptx_and_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in BLOCK32 {
            for schedule in GEMV_SCHEDULES {
                let spec = PackedKernelSpec {
                    format,
                    op: PackedKernelOp::Contraction {
                        rows: RowSelect::Dense,
                        schedule,
                    },
                };
                let name = format!("contraction_{format:?}_{}", schedule_tag(schedule));
                let body = packed_kernel(&name, spec).unwrap();
                let _ = crate::test_support::spv(&body, &format!("{name}_spv"));
                let _ = crate::test_support::ptx(&body, &format!("{name}_ptx"));
                let dir = std::env::temp_dir()
                    .join("poot-kernelgen-unit-test")
                    .join(format!("{name}_amdgcn"));
                std::fs::create_dir_all(&dir).unwrap();
                let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
                compile(&body, Target::AmdGcn(arch), &out).unwrap_or_else(|error| {
                    panic!("{format:?} {schedule:?}: AMDGCN compile: {error}")
                });
            }
        }
    }

    /// The K-quant sibling of `contraction_bodies_compile_for_spirv_nvptx_and_amdgcn`, plus the new
    /// `Schedule::Tiled` on every format (block-32 and K-quant): sanity only, names the SPIR-V/NVPTX
    /// artifacts for the PTX pod batch.
    #[test]
    fn contraction_tiled_and_kquant_bodies_compile_for_spirv_nvptx_and_amdgcn() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in KQUANT {
            for schedule in GEMV_SCHEDULES {
                let spec = PackedKernelSpec {
                    format,
                    op: PackedKernelOp::Contraction {
                        rows: RowSelect::Dense,
                        schedule,
                    },
                };
                let name = format!("contraction_kquant_{format:?}_{}", schedule_tag(schedule));
                let body = packed_kernel(&name, spec).unwrap();
                let _ = crate::test_support::spv(&body, &format!("{name}_spv"));
                let _ = crate::test_support::ptx(&body, &format!("{name}_ptx"));
                let dir = std::env::temp_dir()
                    .join("poot-kernelgen-unit-test")
                    .join(format!("{name}_amdgcn"));
                std::fs::create_dir_all(&dir).unwrap();
                let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
                compile(&body, Target::AmdGcn(arch), &out).unwrap_or_else(|error| {
                    panic!("{format:?} {schedule:?}: AMDGCN compile: {error}")
                });
            }
        }
        for format in BLOCK32.into_iter().chain(KQUANT) {
            let schedule = Schedule::Tiled {
                tile: TileSize::new(4).unwrap(),
            };
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Contraction {
                    rows: RowSelect::Dense,
                    schedule,
                },
            };
            let name = format!("contraction_{format:?}_{}", schedule_tag(schedule));
            let body = packed_kernel(&name, spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("{name}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("{name}_ptx"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(format!("{name}_amdgcn"));
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?} {schedule:?}: AMDGCN compile: {error}"));
        }
    }

    /// SC-005 (card 542a, device tier 2): `Contraction` (Gemv, `RowSelect::Dense`, blocks
    /// 1 and 2) is within tier-2 tolerance of the reference on the local wgpu (SPIR-V/RADV) device
    /// (the `materialize_matches_decode_blocks_on_wgpu` row's sibling).
    fn run_wgpu_case(
        ctx: &poot_runtime::Context,
        format: poot_quant::format::WeightFormat,
        schedule: Schedule,
        blocks: usize,
        m_per_block: usize,
        out_per_block: usize,
    ) {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::KernelBuffer;

        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;
        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let name = format!(
            "contraction_wgpu_{format:?}_{}_{blocks}",
            schedule_tag(schedule)
        );
        let body = packed_kernel(&name, spec)
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004")
            .join(&name);
        std::fs::create_dir_all(&dir).unwrap();
        let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
        compile(&body, Target::SpirvVulkan, &out_path)
            .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
        let spirv_bytes = std::fs::read(&out_path).unwrap();
        let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `contraction_tiled`'s
        // doc).
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0];
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
        let ctx = match poot_runtime::Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };
        for format in BLOCK32 {
            for schedule in GEMV_SCHEDULES {
                run_wgpu_case(&ctx, format, schedule, 1, 1, 11);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_wgpu_kquant() {
        let ctx = match poot_runtime::Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };
        for format in KQUANT {
            for schedule in GEMV_SCHEDULES {
                run_wgpu_case(&ctx, format, schedule, 1, 1, 11);
                run_wgpu_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2): `Contraction` `Schedule::Tiled` for every block-32 and
    /// K-quant format on the local wgpu (SPIR-V/RADV) device - the real-hardware counterpart of
    /// `contraction_tiled_matches_reference_aligned_and_ragged` (`blocks == 1` only; see its doc).
    #[test]
    fn contraction_tiled_matches_reference_aligned_and_ragged_on_wgpu() {
        let ctx = match poot_runtime::Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_wgpu_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                4,
                4,
            );
            run_wgpu_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                7,
                7,
            );
        }
    }

    /// SC-001/SC-002 (card 658, device tier 2, wgpu/RADV): the real-hardware counterpart of
    /// `contraction_tiled_matches_reference_block_diagonal_ragged` - a block-diagonal `Schedule::Tiled`
    /// claim (`blocks = 3`, `m_per_block = 7` not a multiple of `tile = 4`) completes under the GPU
    /// lock and matches the CPU oracle, proving the shape `try_plan_contraction` used to refuse to
    /// `Schedule::Tiled` (falling back to the per-thread-unbounded `Schedule::Serial`) now dispatches
    /// bounded and correct on real hardware.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_on_wgpu() {
        let ctx = match poot_runtime::Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_wgpu_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                3,
                7,
                5,
            );
        }
    }

    /// SC-001 (card 658 review F1/F2, device tier 2, wgpu/RADV): the real-hardware counterpart of
    /// `packed_dequant::tests::over_cap_tiled_contraction_folds_onto_a_2d_grid_not_the_wg_bump` - a
    /// `Schedule::Tiled` dispatch folded onto a 2-D `(x_groups, y_groups)` grid (`tiles =
    /// ceil(20/4)*ceil(4/4) = 5` over a synthetic `x_groups` cap of 4, folding to `(4, 2)`, the smallest
    /// shape that exercises a real `GroupY > 0` workgroup) matches the CPU oracle on real hardware, not
    /// just the KIR interpreter - proving `contraction_tiled`'s `g = gy*x_groups+gx` reconstruction
    /// (`metadata[3]`) is correct on the real SPIR-V/RADV lowering of `GroupY`, not only in the
    /// zero-cost interpreter path every other row here exercises at `GroupY == 0`.
    ///
    /// Mutation: in `contraction_tiled`'s `g = gy*x_groups + gx`, zero the `x_groups` operand (`g =
    /// gy*0 + gx`), dropping `GroupY`'s contribution. Red on real wgpu/RADV: `Q4_0 ... folded
    /// grid=[64,2,1] (wgpu) element 64: got 0 want -55.689743` - element 64 is the first output of the
    /// fold's second row (`GroupY == 1`), which the mutated body never reaches. Restored; green again.
    #[test]
    fn contraction_tiled_folded_grid_matches_reference_on_wgpu() {
        let ctx = match poot_runtime::Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };
        let (blocks, m_per_block, out_per_block, tile) = (1usize, 20usize, 4usize, 4u32);
        let ts = tile as usize;
        let tiles = blocks * m_per_block.div_ceil(ts) * out_per_block.div_ceil(ts);
        let (x_groups, y_groups) = (4u32, 2u32);
        assert_eq!(tiles, 5, "fixture shape must need exactly 5 tiles");
        assert!(
            (x_groups as usize) * (y_groups as usize) >= tiles,
            "the folded grid must address every tile"
        );
        for format in BLOCK32.into_iter().chain(KQUANT) {
            let (k, bytes, activation, expected) =
                contraction_fixture(format, blocks, m_per_block, out_per_block);
            let schedule = Schedule::Tiled {
                tile: TileSize::new(tile).unwrap(),
            };
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Contraction {
                    rows: RowSelect::Dense,
                    schedule,
                },
            };
            let name = format!("contraction_wgpu_folded_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-card658-fold")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path =
                poot_codegen::artifact_path(&dir, &name, poot_codegen::Target::SpirvVulkan);
            poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel =
                poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, spirv_bytes);

            let words = pack_words(&bytes);
            let metadata = [k as u32, out_per_block as u32, m_per_block as u32, x_groups];
            let rows = blocks * m_per_block;
            let out_len = rows * out_per_block;
            let mut buffers = [
                poot_runtime::KernelBuffer::read_only_f32(&activation),
                poot_runtime::KernelBuffer::read_only_u32(&words),
                poot_runtime::KernelBuffer::read_only_u32(&metadata),
                poot_runtime::KernelBuffer::write_f32(out_len),
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
mod planar_materialize_tests {
    //! SC-001 (card 542c): `packed_kernel(Materialize)` run in the KIR interpreter equals
    //! `decode_planar_value` for the six E4M3 variants, E2M1 row-32 (odd K), GPTQ (contiguous and
    //! act-order) and AWQ (tier 1, GPU-free) - the planar counterpart of `materialize_tests`.

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::kernel::{PackedKernelOp, PackedKernelSpec, packed_kernel};
    use super::planar_fixtures::{fixture, pack_words};

    /// `k = 17`: odd (E2M1's packed `Codes` byte holds two K values, so an odd `K` leaves one K
    /// value alone in the last byte - dquant.md 3.3/SC-001's named case), and not a whole multiple
    /// of GPTQ/AWQ's group size (8) or E4M3 block128's block size (128), so every format's ragged
    /// grid edge is exercised by the same fixture shape.
    const K: usize = 17;

    #[test]
    fn materialize_matches_decode_planar_value_for_every_planar_format() {
        let out_rows = 3;
        for (seed, format) in super::planar_fixtures::every_planar_format()
            .into_iter()
            .enumerate()
        {
            let (sources, expected) = fixture(format, out_rows, K, seed as u64);
            let word_bufs = pack_words(&sources);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let body = packed_kernel(
                &format!(
                    "materialize_planar_{}",
                    super::planar_fixtures::format_tag(format)
                ),
                spec,
            )
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let mut buffers: Vec<Buffer> = word_bufs
                .iter()
                .map(|words| Buffer::from_u32s(words))
                .collect();
            buffers.push(Buffer::from_u32s(&[K as u32]));
            buffers.push(Buffer::from_f32s(&vec![0.0; out_rows * K]));
            run(
                &body,
                workgroups_covering(&body, out_rows * K),
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: interpreter: {error}"));
            let got = buffers.last().unwrap().to_f32s().unwrap();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize vs decode_planar_value"
            );
        }
    }

    /// Sanity only (not acceptance, as `materialize_bodies_compile_for_spirv_and_nvptx`): every
    /// planar `Materialize` body lowers to SPIR-V, NVPTX and AMDGCN.
    #[test]
    fn materialize_bodies_compile_for_every_planar_backend() {
        use poot_codegen::{Target, artifact_path, compile};
        let arch = poot_target::AmdArch::gfx1151();
        for format in super::planar_fixtures::every_planar_format() {
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!(
                "materialize_planar_{}",
                super::planar_fixtures::format_tag(format)
            );
            let body = packed_kernel(&name, spec).unwrap();
            let _ = crate::test_support::spv(&body, &format!("{name}_spv"));
            let _ = crate::test_support::ptx(&body, &format!("{name}_ptx"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-unit-test")
                .join(format!("{name}_amdgcn"));
            std::fs::create_dir_all(&dir).unwrap();
            let out = artifact_path(&dir, &name, Target::AmdGcn(arch));
            compile(&body, Target::AmdGcn(arch), &out)
                .unwrap_or_else(|error| panic!("{format:?}: AMDGCN compile: {error}"));
        }
    }

    /// SC-002 (card 542c, device tier 1): `Materialize` for every planar format equals the oracle's
    /// `to_bits` on the local wgpu (SPIR-V/RADV) device.
    #[test]
    fn materialize_matches_decode_planar_value_on_wgpu() {
        use poot_codegen::{Target, artifact_path, compile, kernel_handle};
        use poot_runtime::{Context, KernelBuffer};

        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("no GPU ({error}); skipping");
                return;
            }
        };

        let out_rows = 3;
        for (seed, format) in super::planar_fixtures::every_planar_format()
            .into_iter()
            .enumerate()
        {
            let (sources, expected) = fixture(format, out_rows, K, seed as u64);
            let word_bufs = pack_words(&sources);

            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!(
                "materialize_planar_wgpu_{}",
                super::planar_fixtures::format_tag(format)
            );
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let dir = std::env::temp_dir()
                .join("poot-kernelgen-sc004")
                .join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let out_path = artifact_path(&dir, &name, Target::SpirvVulkan);
            compile(&body, Target::SpirvVulkan, &out_path)
                .unwrap_or_else(|error| panic!("{format:?}: compile: {error}"));
            let spirv_bytes = std::fs::read(&out_path).unwrap();
            let kernel = kernel_handle(&body, Target::SpirvVulkan, spirv_bytes);

            let mut buffers: Vec<KernelBuffer> = word_bufs
                .iter()
                .map(|words| KernelBuffer::read_only_u32(words))
                .collect();
            buffers.push(KernelBuffer::read_only_u32(&[K as u32]));
            buffers.push(KernelBuffer::write_f32(out_rows * K));
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * K) as u32, 1, 1],
                &mut buffers,
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = buffers.last().unwrap().as_f32();

            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (wgpu) vs decode_planar_value"
            );
        }
    }
}

#[cfg(test)]
mod planar_contraction_tests {
    //! SC-002 (card 542c, GPU-free rehearsal): `packed_kernel(Contraction)` (Gemv and Tiled,
    //! `RowSelect::Dense`) for every planar format run in the KIR interpreter is within
    //! tier-2 tolerance of a plain f32 reference contraction over `decode_planar_value` - the planar
    //! counterpart of `contraction_tests`.

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_test_util::kernel_fixtures::workgroups_covering;

    use super::fixtures::{GEMV_SCHEDULES, assert_close};
    use super::kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};
    use super::planar_fixtures::{every_planar_format, fixture, pack_words};
    use crate::contraction::{Schedule, TileSize};

    const K: usize = 17;

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
        out_rows: usize,
        m: usize,
        schedule: Schedule,
    ) {
        let (sources, decoded) = fixture(format, out_rows, K, 11);
        let word_bufs = pack_words(&sources);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..m * K)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected = reference_contraction(&activation, &decoded, K, out_rows);

        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let body = packed_kernel(
            &format!(
                "contraction_planar_{}",
                super::planar_fixtures::format_tag(format)
            ),
            spec,
        )
        .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

        let metadata = [K as u32, out_rows as u32, m as u32, 0]; // metadata[3] (x_groups): unused, unfolded grid
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
            for schedule in GEMV_SCHEDULES {
                run_case(format, 11, 1, schedule);
            }
        }
    }

    /// `out_rows = m = 8` and `12` put more than one tile on each axis with `out_rows` a multiple
    /// of the tile: the grid split `g -> (trow, tcol)` depends on `tiles_col = ceil(out / ts)`
    /// exactly (mutants-m4 H6), which `out_rows = 5` with a single row tile never exposes.
    #[test]
    fn contraction_tiled_matches_reference_for_every_planar_format() {
        for format in every_planar_format() {
            run_case(
                format,
                5,
                3,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
            );
            run_case(
                format,
                8,
                8,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
            );
            run_case(
                format,
                12,
                12,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
            );
        }
    }

    /// SC-002 (card 658 F6, GPU-free rehearsal): the planar front end's identical block-major rewrite
    /// (`contraction_tiled_planar`) gets the same block-diagonal, ragged-`m_per_block` row the block
    /// front end's `contraction_tiled_matches_reference_block_diagonal_ragged` already has - planar
    /// descriptors are admitted for `Contraction` on every backend and `packed_block_diagonal_linear`
    /// does not care about storage, so a planar block-diagonal claim is constructible in production.
    /// `blocks = 3`, `m_per_block = 7` (not a multiple of `tile = 4`), `out_per_block = 5` (also not a
    /// multiple of `tile`) exercise the same ragged row/column edges at once.
    #[test]
    fn contraction_tiled_matches_reference_block_diagonal_ragged_for_every_planar_format() {
        let (blocks, m_per_block, out_per_block) = (3usize, 7usize, 5usize);
        let out_rows = blocks * out_per_block;
        for format in every_planar_format() {
            let (sources, decoded) = fixture(format, out_rows, K, 11);
            let word_bufs = pack_words(&sources);
            let rows = blocks * m_per_block;
            let mut state = 0x2545_f491_4f6c_dd1du64;
            let activation: Vec<f32> = (0..rows * K)
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
                    (0..K)
                        .map(|kk| activation[row_idx * K + kk] * decoded[weight_row * K + kk])
                        .sum()
                })
                .collect();

            let schedule = Schedule::Tiled {
                tile: TileSize::new(4).unwrap(),
            };
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Contraction {
                    rows: RowSelect::Dense,
                    schedule,
                },
            };
            let body = packed_kernel(
                &format!(
                    "contraction_planar_blockdiag_{}",
                    super::planar_fixtures::format_tag(format)
                ),
                spec,
            )
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));

            let metadata = [K as u32, out_per_block as u32, m_per_block as u32, 0];
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
    //! SC-004/SC-005 (card 542a, device tier 1/2): Materialize, RowGather and Contraction (Gemv,
    //! `RowSelect::Dense`, blocks 1 and 2) for every block-32 format, dispatched on real
    //! ROCm/HSA hardware (gfx1151) - the AMDGCN counterpart of the wgpu row above, through the same
    //! production path every real caller uses (`poot_codegen::compile` + `kernel_handle`, card 628's
    //! `no_contract` tests' pattern). Skips (does not fail) with no HSA library or GPU agent, matching
    //! `poot-rocm-runtime`'s own device tests; run serial under `/tmp/poot-gpu.lock`.

    use poot_kernel_ir::Body;
    use poot_rocm_runtime::{BufferRole, HsacoModule, RocmBuffer, RocmContext, RocmError};
    use poot_runtime_common::CompiledKernel;
    use poot_target::ElementKind;

    use super::fixtures::{
        BLOCK32, GEMV_SCHEDULES, KQUANT, assert_close, contraction_fixture, fixture, pack_words,
        schedule_tag,
    };
    use super::kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};
    use crate::contraction::{Schedule, TileSize};

    fn open_or_skip(test: &str) -> Option<RocmContext> {
        match RocmContext::new() {
            Ok(ctx) => Some(ctx),
            Err(RocmError::LibraryNotFound(msg)) => {
                eprintln!("poot-kernelgen: SKIP {test} (no HSA library): {msg}");
                None
            }
            Err(RocmError::NoGpuAgent) => {
                eprintln!("poot-kernelgen: SKIP {test} (no GPU agent)");
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
    ) -> (HsacoModule, CompiledKernel) {
        let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
            .unwrap_or_else(|error| panic!("{name}: could not classify device arch: {error}"));
        let target = poot_codegen::Target::AmdGcn(arch);
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004-rocm")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, target);
        poot_codegen::compile(body, target, &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        let bytes = std::fs::read(&out).unwrap_or_else(|error| panic!("{name}: read: {error}"));
        let kernel = poot_codegen::kernel_handle(body, target, bytes);
        let module = ctx
            .load_hsaco(&kernel)
            .unwrap_or_else(|error| panic!("{name}: load_hsaco: {error}"));
        (module, kernel)
    }

    /// Upload `u32` data through the f32-sized allocator (same 4-byte element width; a raw byte copy):
    /// `poot-rocm-runtime`'s public transfer API has no `u32` variant, only `f32`, and
    /// `write_raw_bytes` is typed to a `RawBytes`-declared buffer only. `f32::from_bits`/`to_bits`
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
    fn write_kernarg(ctx: &RocmContext, kernarg: &RocmBuffer, buffers: &[&RocmBuffer]) {
        let mut karg = vec![0u8; kernarg.byte_capacity()];
        for (slot, buf) in buffers.iter().enumerate() {
            let off = slot * 16;
            karg[off..off + 8].copy_from_slice(&buf.device_ptr().to_le_bytes());
            karg[off + 8..off + 16].copy_from_slice(&u64::from(buf.elem_count()).to_le_bytes());
        }
        ctx.write_raw_bytes(kernarg, 0, &karg)
            .expect("write kernarg");
    }

    /// One buffer typed by its element kind, in the body's declared param order; `In` inputs are
    /// uploaded, `Out(len)` is allocated uninitialized and downloaded back after dispatch.
    enum Arg<'a> {
        U32(&'a [u32]),
        F32(&'a [f32]),
        Out(usize),
    }

    /// `(grid, block)` for a "one thread per element" body (`Materialize`, `RowGather`): `block` must
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
        let kernarg = ctx
            .allocate_kernarg(kernel.kernarg_size() as usize)
            .expect("allocate kernarg");
        write_kernarg(ctx, &kernarg, &refs);
        ctx.dispatch(&kernel, &kernarg, &elements, grid, block)
            .unwrap_or_else(|error| panic!("{name}: dispatch: {error}"));
        let mut got = vec![0.0f32; out_len];
        ctx.download_f32(owned.last().expect("Arg::Out present"), &mut got)
            .unwrap_or_else(|error| panic!("{name}: download: {error}"));
        got
    }

    #[test]
    fn materialize_matches_decode_blocks_on_rocm() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_blocks_on_rocm") else {
            return;
        };
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in BLOCK32.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_rocm_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let words = pack_words(&bytes);
            let (grid, block) = one_thread_per_element_launch(&body, out_rows * k);
            let got = dispatch_and_download(
                &ctx,
                &name,
                &body,
                &[
                    Arg::U32(&words),
                    Arg::U32(&[k as u32]),
                    Arg::Out(out_rows * k),
                ],
                grid,
                block,
            );
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (ROCm) vs decode_blocks"
            );
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_rocm() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_rocm") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in BLOCK32.into_iter().enumerate() {
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
                    Arg::F32(&ids_f32),
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
        let (k, bytes, activation, expected) =
            contraction_fixture(format, blocks, m_per_block, out_per_block);
        let rows = blocks * m_per_block;
        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let name = format!(
            "contraction_rocm_{format:?}_{}_{blocks}",
            schedule_tag(schedule)
        );
        let body = packed_kernel(&name, spec)
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let words = pack_words(&bytes);
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `contraction_tiled`'s
        // doc).
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0];
        let out_len = rows * out_per_block;
        let grid = (
            schedule.grid_threads(blocks, m_per_block, out_per_block) as u32,
            1,
            1,
        );
        let block = (body.workgroup_size[0], 1, 1);
        let got = dispatch_and_download(
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
        );
        assert_close(
            format,
            &format!("Contraction {schedule:?} blocks={blocks} (ROCm)"),
            &got,
            &expected,
        );
    }

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_rocm() {
        let Some(ctx) = open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_rocm")
        else {
            return;
        };
        for format in BLOCK32 {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 1/2): Materialize/RowGather/Contraction (Gemv) for every
    /// K-quant format on real ROCm/HSA hardware.
    #[test]
    fn materialize_matches_decode_blocks_on_rocm_kquant() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_blocks_on_rocm_kquant") else {
            return;
        };
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in KQUANT.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_rocm_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let words = pack_words(&bytes);
            let (grid, block) = one_thread_per_element_launch(&body, out_rows * k);
            let got = dispatch_and_download(
                &ctx,
                &name,
                &body,
                &[
                    Arg::U32(&words),
                    Arg::U32(&[k as u32]),
                    Arg::Out(out_rows * k),
                ],
                grid,
                block,
            );
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (ROCm) vs decode_blocks"
            );
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_rocm_kquant() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_rocm_kquant") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in KQUANT.into_iter().enumerate() {
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
                    Arg::F32(&ids_f32),
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

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_rocm_kquant() {
        let Some(ctx) =
            open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_rocm_kquant")
        else {
            return;
        };
        for format in KQUANT {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2): `Contraction` `Schedule::Tiled` for every block-32 and
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
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                4,
                4,
            );
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                7,
                7,
            );
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
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                3,
                7,
                5,
            );
        }
    }

    /// SC-002 (card 542c, device tier 1): `Materialize` for every planar format equals the oracle's
    /// `to_bits` on real ROCm/HSA hardware - the planar counterpart of
    /// `materialize_matches_decode_blocks_on_rocm`. `dispatch_and_download`'s `args` is a plain
    /// slice, so a planar format's variable source count needs no new harness.
    #[test]
    fn materialize_matches_decode_planar_value_on_rocm() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_planar_value_on_rocm") else {
            return;
        };
        let out_rows = 3;
        let k = 17;
        for (seed, format) in super::planar_fixtures::every_planar_format()
            .into_iter()
            .enumerate()
        {
            let (sources, expected) =
                super::planar_fixtures::fixture(format, out_rows, k, seed as u64);
            let word_bufs = super::planar_fixtures::pack_words(&sources);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!(
                "materialize_planar_rocm_{}",
                super::planar_fixtures::format_tag(format)
            );
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let (grid, block) = one_thread_per_element_launch(&body, out_rows * k);
            let k_meta = [k as u32];
            let mut args: Vec<Arg<'_>> = word_bufs.iter().map(|w| Arg::U32(w)).collect();
            args.push(Arg::U32(&k_meta));
            args.push(Arg::Out(out_rows * k));
            let got = dispatch_and_download(&ctx, &name, &body, &args, grid, block);
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (ROCm) vs decode_planar_value"
            );
        }
    }

    /// SC-002 (card 542c, device tier 2): `Contraction` (Gemv, Tiled) for every planar format is
    /// within tier-2 tolerance of the reference on real ROCm/HSA hardware.
    #[test]
    fn contraction_matches_reference_for_every_planar_format_on_rocm() {
        let Some(ctx) =
            open_or_skip("contraction_matches_reference_for_every_planar_format_on_rocm")
        else {
            return;
        };
        let k = 17;
        let out_rows = 5;
        for format in super::planar_fixtures::every_planar_format() {
            for schedule in GEMV_SCHEDULES.into_iter().chain([Schedule::Tiled {
                tile: TileSize::new(4).unwrap(),
            }]) {
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

                let spec = PackedKernelSpec {
                    format,
                    op: PackedKernelOp::Contraction {
                        rows: RowSelect::Dense,
                        schedule,
                    },
                };
                let name = format!(
                    "contraction_planar_rocm_{}_{}",
                    super::planar_fixtures::format_tag(format),
                    schedule_tag(schedule)
                );
                let body = packed_kernel(&name, spec)
                    .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
                let metadata = [k as u32, out_rows as u32, m as u32, 0]; // metadata[3] (x_groups): unused, unfolded grid
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
                    &format!("planar Contraction {schedule:?} (ROCm)"),
                    &got,
                    &expected,
                );
            }
        }
    }
}

#[cfg(test)]
mod ptx_tests {
    //! SC-004/SC-005 (card 542a, device tier 1/2, M4 pod batch): Materialize, RowGather and
    //! Contraction (Gemv, `RowSelect::Dense`, blocks 1 and 2) for every block-32 format,
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

    use poot_codegen::{Target, artifact_path, compile, kernel_handle};
    use poot_kernel_ir::Body;
    use poot_ptx_runtime::{PtxBuffer, PtxContext};

    use super::fixtures::{
        BLOCK32, GEMV_SCHEDULES, KQUANT, assert_close, contraction_fixture, fixture, pack_words,
        schedule_tag,
    };
    use super::kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};
    use crate::contraction::{Schedule, TileSize};

    fn open_or_skip(test: &str) -> Option<PtxContext> {
        match PtxContext::new() {
            Ok(ctx) => Some(ctx),
            Err(error) => {
                eprintln!("poot-kernelgen: SKIP {test} (no PTX device): {error}");
                None
            }
        }
    }

    fn compile_and_load(name: &str, body: &Body) -> poot_runtime_common::CompiledKernel {
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc004-ptx")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = artifact_path(&dir, name, Target::Nvptx);
        compile(body, Target::Nvptx, &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        let bytes = std::fs::read(&out).unwrap_or_else(|error| panic!("{name}: read: {error}"));
        kernel_handle(body, Target::Nvptx, bytes)
    }

    /// Upload `u32` data through the i32-typed allocator (`u32 as i32` is a bit-preserving
    /// reinterpretation between same-width integers, never a numeric conversion):
    /// `poot-ptx-runtime`'s public upload API has no `u32` variant, only `i32`.
    fn upload_u32(ctx: &PtxContext, data: &[u32]) -> PtxBuffer {
        let as_i32: Vec<i32> = data.iter().map(|&word| word as i32).collect();
        ctx.upload_i32(&as_i32).expect("upload u32 buffer")
    }

    #[test]
    fn materialize_matches_decode_blocks_on_ptx() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_blocks_on_ptx") else {
            return;
        };
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in BLOCK32.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_ptx_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let words = pack_words(&bytes);
            let words_buf = upload_u32(&ctx, &words);
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(out_rows * k).expect("allocate out");
            ctx.dispatch_dev(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * k) as u32, 1, 1],
                &[&words_buf, &metadata_buf],
                &[words_buf.elem_count(), metadata_buf.elem_count()],
                &out_buf,
                out_buf.elem_count(),
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = ctx.download_f32(&out_buf).expect("download out");
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (PTX) vs decode_blocks"
            );
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_ptx() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_ptx") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in BLOCK32.into_iter().enumerate() {
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
            let ids_buf = ctx.upload_f32(&ids_f32).expect("upload ids");
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(ids.len() * k).expect("allocate out");
            ctx.dispatch_dev(
                &name,
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
        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let name = format!(
            "contraction_ptx_{format:?}_{}_{blocks}",
            schedule_tag(schedule)
        );
        let body = packed_kernel(&name, spec)
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let kernel = compile_and_load(&name, &body);
        let words = pack_words(&bytes);
        let words_buf = upload_u32(ctx, &words);
        let activation_buf = ctx.upload_f32(&activation).expect("upload activation");
        // metadata[3] (x_groups): 0 is fine here - every dispatch below is `[threads, 1, 1]`
        // (unfolded), so `GroupY` always reads 0 and `g = gy*x_groups+gx` collapses to `gx`
        // regardless of this value (card 658 review F1/F2's fold-aware grid, `contraction_tiled`'s
        // doc).
        let metadata = [k as u32, out_per_block as u32, m_per_block as u32, 0];
        let metadata_buf = upload_u32(ctx, &metadata);
        let out_len = rows * out_per_block;
        let out_buf = ctx.alloc_f32(out_len).expect("allocate out");
        // Total work-items (`Schedule::grid_threads`; dispatch_dev divides by body.workgroup_size):
        // a Gemv launches `width` lanes per `cols` outputs, not one thread per output - the exact
        // block/workgroup_size mismatch the ROCm harness caught (see `one_thread_per_element_launch`'s
        // doc comment above).
        let threads = schedule.grid_threads(blocks, m_per_block, out_per_block) as u32;
        ctx.dispatch_dev(
            &name,
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
        for format in BLOCK32 {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 1/2, M4 pod batch): Materialize/RowGather/Contraction (Gemv)
    /// for every K-quant format, named for the PTX pod - unexercised locally
    /// (no NVIDIA device on this box), like every other row in this module.
    #[test]
    fn materialize_matches_decode_blocks_on_ptx_kquant() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_blocks_on_ptx_kquant") else {
            return;
        };
        let out_rows = 3;
        let blocks_per_row = 2;
        for (seed, format) in KQUANT.into_iter().enumerate() {
            let (k, bytes, expected) = fixture(format, out_rows, blocks_per_row, seed as u64);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_ptx_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let words = pack_words(&bytes);
            let words_buf = upload_u32(&ctx, &words);
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(out_rows * k).expect("allocate out");
            ctx.dispatch_dev(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * k) as u32, 1, 1],
                &[&words_buf, &metadata_buf],
                &[words_buf.elem_count(), metadata_buf.elem_count()],
                &out_buf,
                out_buf.elem_count(),
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = ctx.download_f32(&out_buf).expect("download out");
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (PTX) vs decode_blocks"
            );
        }
    }

    #[test]
    fn row_gather_matches_decode_blocks_rows_on_ptx_kquant() {
        let Some(ctx) = open_or_skip("row_gather_matches_decode_blocks_rows_on_ptx_kquant") else {
            return;
        };
        let out_rows = 5;
        let blocks_per_row = 2;
        let ids: [u32; 4] = [3, 0, 4, 3];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        for (seed, format) in KQUANT.into_iter().enumerate() {
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
            let name = format!("row_gather_ptx_kquant_{format:?}");
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let words = pack_words(&bytes);
            let words_buf = upload_u32(&ctx, &words);
            let ids_buf = ctx.upload_f32(&ids_f32).expect("upload ids");
            let metadata_buf = upload_u32(&ctx, &[k as u32]);
            let out_buf = ctx.alloc_f32(ids.len() * k).expect("allocate out");
            ctx.dispatch_dev(
                &name,
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

    #[test]
    fn contraction_gemv_matches_reference_blocks_1_and_2_on_ptx_kquant() {
        let Some(ctx) =
            open_or_skip("contraction_gemv_matches_reference_blocks_1_and_2_on_ptx_kquant")
        else {
            return;
        };
        for format in KQUANT {
            for schedule in GEMV_SCHEDULES {
                run_contraction_case(&ctx, format, schedule, 1, 1, 11);
                run_contraction_case(&ctx, format, schedule, 2, 1, 5);
            }
        }
    }

    /// SC-002 (card 542b, device tier 2, M4 pod batch): `Contraction` `Schedule::Tiled` for every
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
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                4,
                4,
            );
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                1,
                7,
                7,
            );
        }
    }

    /// SC-002/F3 (card 658 review, device tier 2, M4 pod batch): `Contraction` `Schedule::Tiled` over
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
        for format in BLOCK32.into_iter().chain(KQUANT) {
            run_contraction_case(
                &ctx,
                format,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
                3,
                7,
                5,
            );
        }
    }
}

#[cfg(test)]
mod planar_ptx_tests {
    //! SC-002 (card 542c, device tier 1/2, M4 pod batch): Materialize and Contraction (Gemv,
    //! Tiled) for every planar format, dispatched on real NVIDIA/PTX hardware - the planar
    //! counterpart of `ptx_tests` (same skip/require convention, same `PtxContext` harness;
    //! `dispatch_dev`'s source-buffer slice is plain `&[&PtxBuffer]`, so a planar format's variable
    //! source count needs no new harness). No local run is possible on this box (AMD Strix Halo, no
    //! NVIDIA device): unexercised until the PTX pod batch runs them.

    use poot_codegen::{Target, artifact_path, compile, kernel_handle};
    use poot_kernel_ir::Body;
    use poot_ptx_runtime::{PtxBuffer, PtxContext};

    use super::fixtures::{GEMV_SCHEDULES, schedule_tag};
    use super::kernel::{PackedKernelOp, PackedKernelSpec, RowSelect, packed_kernel};
    use super::planar_fixtures::{every_planar_format, fixture, format_tag, pack_words};
    use crate::contraction::{Schedule, TileSize};

    const K: usize = 17;

    fn open_or_skip(test: &str) -> Option<PtxContext> {
        match PtxContext::new() {
            Ok(ctx) => Some(ctx),
            Err(error) => {
                eprintln!("poot-kernelgen: SKIP {test} (no PTX device): {error}");
                None
            }
        }
    }

    fn compile_and_load(name: &str, body: &Body) -> poot_runtime_common::CompiledKernel {
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-sc002-planar-ptx")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = artifact_path(&dir, name, Target::Nvptx);
        compile(body, Target::Nvptx, &out)
            .unwrap_or_else(|error| panic!("{name}: compile: {error}"));
        let bytes = std::fs::read(&out).unwrap_or_else(|error| panic!("{name}: read: {error}"));
        kernel_handle(body, Target::Nvptx, bytes)
    }

    fn upload_u32(ctx: &PtxContext, data: &[u32]) -> PtxBuffer {
        let as_i32: Vec<i32> = data.iter().map(|&word| word as i32).collect();
        ctx.upload_i32(&as_i32).expect("upload u32 buffer")
    }

    #[test]
    fn materialize_matches_decode_planar_value_on_ptx() {
        let Some(ctx) = open_or_skip("materialize_matches_decode_planar_value_on_ptx") else {
            return;
        };
        let out_rows = 3;
        for (seed, format) in every_planar_format().into_iter().enumerate() {
            let (sources, expected) = fixture(format, out_rows, K, seed as u64);
            let word_bufs = pack_words(&sources);
            let spec = PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            };
            let name = format!("materialize_planar_ptx_{}", format_tag(format));
            let body = packed_kernel(&name, spec)
                .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
            let kernel = compile_and_load(&name, &body);
            let source_bufs: Vec<PtxBuffer> = word_bufs
                .iter()
                .map(|words| upload_u32(&ctx, words))
                .collect();
            let metadata_buf = upload_u32(&ctx, &[K as u32]);
            let mut arg_refs: Vec<&PtxBuffer> = source_bufs.iter().collect();
            arg_refs.push(&metadata_buf);
            let arg_lens: Vec<u32> = arg_refs.iter().map(|b| b.elem_count()).collect();
            let out_buf = ctx.alloc_f32(out_rows * K).expect("allocate out");
            ctx.dispatch_dev(
                &name,
                &kernel,
                body.workgroup_size,
                [(out_rows * K) as u32, 1, 1],
                &arg_refs,
                &arg_lens,
                &out_buf,
                out_buf.elem_count(),
            )
            .unwrap_or_else(|error| panic!("{format:?}: dispatch: {error}"));
            let got = ctx.download_f32(&out_buf).expect("download out");
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "{format:?}: Materialize (PTX) vs decode_planar_value"
            );
        }
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
        out_rows: usize,
        schedule: Schedule,
    ) {
        let (sources, decoded) = fixture(format, out_rows, K, 11);
        let word_bufs = pack_words(&sources);
        let m = match schedule {
            Schedule::Gemv { .. } => 1,
            _ => 3,
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let activation: Vec<f32> = (0..m * K)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 500.0
            })
            .collect();
        let expected = reference_contraction(&activation, &decoded, K, out_rows);

        let spec = PackedKernelSpec {
            format,
            op: PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule,
            },
        };
        let name = format!(
            "contraction_planar_ptx_{}_{}",
            format_tag(format),
            schedule_tag(schedule)
        );
        let body = packed_kernel(&name, spec)
            .unwrap_or_else(|error| panic!("{format:?}: packed_kernel: {error}"));
        let kernel = compile_and_load(&name, &body);
        let source_bufs: Vec<PtxBuffer> = word_bufs
            .iter()
            .map(|words| upload_u32(ctx, words))
            .collect();
        let activation_buf = ctx.upload_f32(&activation).expect("upload activation");
        let metadata = [K as u32, out_rows as u32, m as u32, 0]; // metadata[3] (x_groups): unused, unfolded grid
        let metadata_buf = upload_u32(ctx, &metadata);
        let out_len = m * out_rows;
        let out_buf = ctx.alloc_f32(out_len).expect("allocate out");
        let mut arg_refs: Vec<&PtxBuffer> = vec![&activation_buf];
        arg_refs.extend(source_bufs.iter());
        arg_refs.push(&metadata_buf);
        let arg_lens: Vec<u32> = arg_refs.iter().map(|b| b.elem_count()).collect();
        let threads = schedule.grid_threads(1, m, out_rows) as u32;
        ctx.dispatch_dev(
            &name,
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
            for schedule in GEMV_SCHEDULES {
                run_case(&ctx, format, 11, schedule);
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
            run_case(
                &ctx,
                format,
                5,
                Schedule::Tiled {
                    tile: TileSize::new(4).unwrap(),
                },
            );
        }
    }
}
