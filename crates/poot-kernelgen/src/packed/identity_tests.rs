//! The emitted-kernel identity of every packed format and op (POOT-1539 SC-1).
//!
//! A refactor of the decode emitter must not change a kernel any existing format emits. The identity
//! of a kernel is the digest of its body's `Debug` text, which is what `poot-codegen`'s `KernelId` hashes
//! (with the target), so two bodies with the same digest here are the same cached kernel. The table in
//! `tests/data/packed_kernel_identities.txt` pins one digest per `(format, op)` row, recorded before the
//! lattice refactor; a row that changes names its format and op.
//!
//! Re-recorded once, for POOT-1558: `VectorLoad` became element-indexed and carries `align_bytes`, so the 29
//! `gemv_vec4` bodies (the only users, one per format) read their activation group at element `act_base + 4g` with
//! a 16-byte alignment. Every other row is unchanged.
//!
//! Re-recorded again for POOT-1559 (static-alignment reads): the block formats' run reads and factor reads no longer
//! load `chunks + 1` words through a clamp and a runtime funnel shift, and read each factor piece as one word where
//! the descriptor proves it never straddles. 213 of 377 rows changed: every row of the 16 block formats (Q4_0, Q4_1,
//! Q5_0, Q5_1, Q8_0, Mxfp4, Iq4_Nl, Q2_K to Q6_K, Iq2_Xs, Iq3_S, Iq3_Xxs, Iq4_Xs) under `row_gather`, `gemv0` to
//! `gemv3`, `gemv0_indexed`, `gemv1_indexed`, `gemv_vec4`, `register_tiled`, `register_tiled_indexed`,
//! `register_tiled_narrow_f16` and `fragment_tiled_split_bf16`, because each emits `DecodePlan::emit_run` or
//! `emit_factors`; and the Gemv rows of the three one-value block formats (F32, F16, Bf16), whose single-element
//! tail run now reads its word aligned. The planar formats and every `run_sharing` row are unchanged. The numerics
//! are bit-identical: `every_format_gemv_is_bit_identical_after_static_alignment` pins the interpreter output of
//! every block format under every Gemv schedule from the body before this change. To re-record, run the test and
//! copy its `full table`.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_quant::format::WeightFormat;
use poot_quant::test_support::formats::{GGUF_FORMATS, SAFETENSORS_FORMATS};
use poot_target::{FragmentLayout, LaneMap, RoundingConversions};

use super::fixtures::{GEMV_SCHEDULES, register_tiled};
use super::{PackedKernelOp, PackedKernelSpec, packed_contraction, packed_kernel};
use crate::PackedRows;
use crate::contraction::{RunLength, Schedule, TileSize};

const K: usize = 2048;
const OUT_PER_BLOCK: u32 = 16;

/// FNV-1a over the body's `Debug` text: a stable digest with no dependency, enough to tell two kernels apart.
pub(super) fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn fragment(dtype: FragmentDtype) -> Schedule {
    Schedule::FragmentTiled {
        waves_m: TileSize::new(2).unwrap(),
        waves_n: TileSize::new(1).unwrap(),
        tile: crate::FragmentTile {
            dtype,
            layout: FragmentLayout::Declared(LaneMap::Rdna3),
            rows: 1,
            cols: 2,
        },
        run: RunLength::dividing(K),
        depth: 32,
    }
}

/// The identity of one body, or the typed refusal that replaces it: a refusal is part of what a format emits.
fn identity(body: Result<poot_kernel_ir::Body, crate::KernelGenError>) -> String {
    match body {
        Ok(body) => format!("{:016x}", digest(&format!("{body:?}"))),
        Err(error) => format!("refused: {error}"),
    }
}

/// Every `(format, op)` row as `format | op | identity`, over every GGUF and safetensors format.
fn rows() -> Vec<String> {
    let meta = [K as u32, OUT_PER_BLOCK, 4, 0, 0];
    let contraction = |format: WeightFormat,
                       schedule: Schedule,
                       rows: PackedRows,
                       precision: OperandPrecision,
                       conversions: RoundingConversions| {
        packed_contraction(
            "identity",
            format,
            schedule,
            &meta,
            None,
            rows,
            precision,
            conversions,
        )
    };
    let exact = OperandPrecision::Exact;
    let none = RoundingConversions::NONE;
    let mut table = Vec::new();
    for format in GGUF_FORMATS.into_iter().chain(SAFETENSORS_FORMATS) {
        let mut row =
            |op: String, body| table.push(format!("{format:?} | {op} | {}", identity(body)));
        row(
            "row_gather".into(),
            packed_kernel(
                "identity",
                PackedKernelSpec {
                    format,
                    op: PackedKernelOp::RowGather,
                },
            ),
        );
        for (index, schedule) in GEMV_SCHEDULES.into_iter().enumerate() {
            row(
                format!("gemv{index}"),
                contraction(format, schedule, PackedRows::Blocks, exact, none),
            );
        }
        for (index, schedule) in GEMV_SCHEDULES.into_iter().take(2).enumerate() {
            row(
                format!("gemv{index}_indexed"),
                contraction(format, schedule, PackedRows::Indexed, exact, none),
            );
        }
        row(
            "gemv_vec4".into(),
            contraction(
                format,
                Schedule::Gemv {
                    width: 32,
                    cols: 2,
                    unroll: 4,
                    activation: crate::GemvActivation::Vec4,
                },
                PackedRows::Blocks,
                exact,
                none,
            ),
        );
        row(
            "register_tiled".into(),
            contraction(
                format,
                register_tiled(2, K),
                PackedRows::Blocks,
                exact,
                none,
            ),
        );
        row(
            "register_tiled_indexed".into(),
            contraction(
                format,
                register_tiled(2, K),
                PackedRows::Indexed,
                exact,
                none,
            ),
        );
        row(
            "register_tiled_narrow_f16".into(),
            contraction(
                format,
                register_tiled(2, K),
                PackedRows::Blocks,
                OperandPrecision::Narrow(FragmentDtype::F16),
                RoundingConversions::F16,
            ),
        );
        row(
            "fragment_tiled_split_bf16".into(),
            contraction(
                format,
                fragment(FragmentDtype::Bf16),
                PackedRows::Blocks,
                OperandPrecision::Split(FragmentDtype::Bf16),
                none,
            ),
        );
        // The planner's two facts about a decode: what a lane run shares and the bits a value occupies.
        let sharing = match super::packed_run_sharing(format) {
            Ok(sharing) => format!("{sharing:?}"),
            Err(error) => format!("refused: {error}"),
        };
        table.push(format!(
            "{format:?} | run_sharing | {sharing} {} bits",
            super::packed_value_bits(format)
        ));
    }
    table
}

/// SC-1: every emitted kernel of every existing format is the one recorded before the lattice refactor.
/// Mutation: change the lattice emitter's cell shift and the rows of the table formats go red by name.
#[test]
fn every_format_emits_the_kernel_recorded_before_the_lattice_refactor() {
    let recorded = include_str!("../../tests/data/packed_kernel_identities.txt");
    let current = rows();
    let expected: Vec<&str> = recorded.lines().collect();
    let changed: Vec<String> = current
        .iter()
        .zip(&expected)
        .filter(|(now, was)| now.as_str() != **was)
        .map(|(now, was)| format!("recorded {was}\n     now {now}"))
        .collect();
    assert!(
        changed.is_empty() && current.len() == expected.len(),
        "{} of {} kernel identities changed ({} recorded):\n{}\n--- full table ---\n{}",
        changed.len(),
        current.len(),
        expected.len(),
        changed.join("\n"),
        current.join("\n")
    );
}
