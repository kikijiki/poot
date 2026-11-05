//! The registers an AMDGCN kernel the planner's tilings produce takes: compiled offline for gfx1151, its code
//! object's metadata must show no spilled VGPR. A spilled register-tiled body moves its accumulators through
//! scratch memory inside the key and depth loops; the tiled attention at its planned tiling did, 60-120 VGPRs
//! of it, until the AMDGCN compile stopped LLVM re-unrolling the generators' own unrolled loops (POOT-1428).

use std::path::Path;

use poot_codegen::{Target, artifact_path};
use poot_target::{AmdArch, BufferStorage, DeviceCaps};

use crate::attention::{
    AttentionMicro, AttentionSchedule, AttentionSink, AttentionSpec, KeyBound, KeyLayout, LaneTile,
    MaskKind, ScoreTransform, WaveTile, attention_body,
};
use crate::contraction::{DenseTiledWeight, RegisterTiling, TiledShape, dense_tiled_contraction};
use crate::helpers::WeightLayout;

/// `(vgprs, spilled vgprs)` of the kernel in code object `hsaco`, from `llvm-readelf --notes` of the toolchain
/// that compiled it (next to `$POOT_AMD_CLANG`), or `None` when that tool is absent.
fn vgprs(hsaco: &Path) -> Option<(u32, u32)> {
    let clang = std::env::var_os("POOT_AMD_CLANG")?;
    let readelf = Path::new(&clang).parent()?.join("llvm-readelf");
    let notes = std::process::Command::new(readelf)
        .arg("--notes")
        .arg(hsaco)
        .output()
        .ok()?;
    let notes = String::from_utf8_lossy(&notes.stdout);
    let field = |name: &str| -> u32 {
        notes
            .lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_else(|| panic!("no {name} in the code object notes:\n{notes}"))
    };
    Some((field(".vgpr_count:"), field(".vgpr_spill_count:")))
}

/// Compile `body` for gfx1151 into a fresh directory and return its registers; `None` (a reported skip) without
/// the AMD toolchain.
fn compiled_registers(body: &poot_kernel_ir::Body) -> Option<(u32, u32)> {
    let arch = AmdArch::gfx1151();
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-unit-test")
        .join(format!("{}_registers", body.name));
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, &body.name, Target::AmdGcn(arch));
    if !crate::unit_helpers::compile_amdgcn_or_skip(&body.name, body, arch, &out) {
        return None;
    }
    let registers = vgprs(&out);
    if registers.is_none() {
        eprintln!(
            "POOT-SKIP poot-kernelgen: {} (no llvm-readelf next to POOT_AMD_CLANG)",
            body.name
        );
    }
    registers
}

/// The 16-head, 8-key/value-head, 128-wide causal prefill attention of 8192 positions under the tiled schedule
/// `br` positions by `bc` keys per block, 256 lanes, with the given score and output lane tiles.
fn tiled_spec(br: usize, bc: usize, score: (usize, usize), out: (usize, usize)) -> AttentionSpec {
    AttentionSpec {
        kv: crate::KvLane::F16,
        batch: 1,
        hq: 16,
        n_rep: 2,
        lq: 8192,
        lk: 8192,
        dk: 128,
        dv: 128,
        key: KeyLayout::Single,
        score: ScoreTransform::Scale {
            scale_bits: 0.088f32.to_bits(),
        },
        mask: MaskKind::Causal { window: None },
        bound: KeyBound::Static,
        sink: AttentionSink::Absent,
        schedule: AttentionSchedule::Tiled {
            br,
            bc,
            lanes: 256,
            micro: AttentionMicro::Scalar {
                score: LaneTile {
                    rows: score.0,
                    cols: score.1,
                },
                out: LaneTile {
                    rows: out.0,
                    cols: out.1,
                },
            },
        },
        precision: poot_graph_ir::OperandPrecision::Exact,
    }
}

/// The tiled attention at every tiling the planner can choose for a 16-head, 8-key/value-head, 128-wide
/// prefill (its rows, key blocks and lane tiles: `attention_tiling`'s candidates) spills no register. Before the
/// AMDGCN compile disabled LLVM's unroller, `br 32, bc 16` (the planner's pick on a 64 KiB LDS) took 256 VGPRs with
/// 68 spilled, `br 16, bc 32` 117 spilled. Mutation: drop `-fno-unroll-loops` from `Target::llc_args`; the
/// `br 32, bc 16` row reads `spilled 68 VGPRs`.
#[test]
fn the_tiled_attention_at_its_planned_tilings_spills_no_register() {
    let tilings = [
        (32usize, 16usize, (2usize, 2usize), (8usize, 4usize)),
        (32, 32, (2, 4), (4, 8)),
        (32, 16, (4, 1), (8, 4)),
        (16, 32, (2, 2), (4, 4)),
        (16, 16, (1, 2), (2, 8)),
    ];
    for (br, bc, score, out) in tilings {
        let spec = tiled_spec(br, bc, score, out);
        let name = format!("budget_attention_br{br}_bc{bc}_s{}x{}", score.0, score.1);
        let body = attention_body(&name, &spec, 0..spec.workgroups(), 1).unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(
            spilled, 0,
            "{name}: spilled {spilled} VGPRs of {vgprs} (the planner's tiling must fit the register file)"
        );
    }
}

/// The fragment tiled attention (f16 operands on the gfx11 matrix unit, a declared lane map) at the tilings the
/// planner chooses for a 16-head, 8-key/value-head, 128-wide prefill spills no register: the score and output
/// accumulator fragments, the operand fragments of a step and the softmax's per-row state fit the file beside
/// each other. Mutation: raise `trip_steps` for a fragment to 8; the hoisted operand loads of eight steps read
/// `spilled`.
#[test]
fn the_fragment_attention_at_its_planned_tilings_spills_no_register() {
    let wave = |(waves_m, waves_n, rows, cols)| WaveTile {
        waves_m,
        waves_n,
        rows,
        cols,
    };
    // (br, bc, lanes, score waves, output waves) of n_rep = 2, each wave `(waves_m, waves_n, rows, cols)`.
    let tilings = [
        (32usize, 32usize, 256usize, (4, 2, 1, 1), (4, 2, 1, 4)),
        (32, 16, 128, (4, 1, 1, 1), (2, 2, 2, 4)),
        (16, 32, 128, (2, 2, 1, 1), (2, 2, 1, 4)),
        // The planner's pick for the bank's 16-head, 8-key/value-head prefill (128 rows over 8 waves).
        (64, 16, 256, (8, 1, 1, 1), (4, 2, 2, 4)),
    ];
    for (br, bc, lanes, score, out) in tilings {
        let mut spec = tiled_spec(br, bc, (1, 1), (1, 1));
        spec.precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::F16);
        spec.schedule = AttentionSchedule::Tiled {
            br,
            bc,
            lanes,
            micro: AttentionMicro::Fragment {
                dtype: poot_target::FragmentDtype::F16,
                layout: poot_target::FragmentLayout::Declared(poot_target::LaneMap::Rdna3),
                score: wave(score),
                out: wave(out),
            },
        };
        let name = format!("budget_fragment_attention_br{br}_bc{bc}");
        let body = attention_body(&name, &spec, 0..spec.workgroups(), 1).unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(
            spilled, 0,
            "{name}: spilled {spilled} VGPRs of {vgprs} (the planner's tiling must fit the register file)"
        );
    }
}

/// The register-tiled dense contraction at the tiling the planner chooses for each projection of a prefill
/// spills no register and keeps its occupancy-friendly register count.
#[test]
fn the_dense_register_tiled_contraction_spills_no_register() {
    let caps = DeviceCaps::rocm_default();
    for (label, m, n, k) in [
        ("qkv", 8192usize, 4096usize, 2560usize),
        ("down", 8192, 2560, 9728),
    ] {
        let shape = TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            segments: 1,
            terms: crate::precision::StagedTerms::ONE,
        };
        let tiling = RegisterTiling::choose(shape, &caps);
        let name = format!("budget_dense_{label}");
        let body = dense_tiled_contraction(
            &name,
            DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
            tiling.schedule(),
            k,
            None,
            poot_graph_ir::OperandPrecision::Exact,
        )
        .unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(spilled, 0, "{name}: spilled {spilled} VGPRs of {vgprs}");
    }
}

/// The fragment dense contraction at the tiling the planner chooses for each narrowed prefill projection (bf16
/// operands on the gfx11 matrix unit) spills no register: its accumulator fragments, the operand fragments of a step
/// and the staging addresses fit the file. Mutation: raise `MAX_WAVE_FRAGMENTS` to 16; the planner's pick for the
/// long projections holds 128 accumulator registers beside its operands and the row reads `spilled`.
#[test]
fn the_fragment_dense_contraction_spills_no_register() {
    let caps = DeviceCaps {
        matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::amd("gfx1151", 32)),
        ..DeviceCaps::rocm_default()
    };
    let precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16);
    for (label, m, n, k) in [
        ("qkv", 8192usize, 4096usize, 2560usize),
        ("down", 8192, 2560, 9728),
        ("short", 128, 2048, 2048),
    ] {
        let shape = TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            segments: 1,
            terms: crate::precision::StagedTerms::of(
                precision,
                None,
                Some(poot_graph_ir::FragmentDtype::Bf16),
            ),
        };
        let tiling = RegisterTiling::choose(shape, &caps);
        assert!(
            matches!(tiling.schedule(), crate::Schedule::FragmentTiled { .. }),
            "{label}: {:?}",
            tiling.schedule()
        );
        let name = format!("budget_fragment_{label}");
        let body = dense_tiled_contraction(
            &name,
            DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
            tiling.schedule(),
            k,
            None,
            precision,
        )
        .unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(spilled, 0, "{name}: spilled {spilled} VGPRs of {vgprs}");
    }
}

/// A body whose generator unrolled it carries the no-unroll loop metadata on AMDGCN; the same body on NVPTX and a
/// body the backend may unroll (the streaming attention a decode runs, which LLVM unrolls to good effect: the
/// low-head decode attention took 2.3x longer without it) carry none. Mutation: make `emit_terminator` skip
/// the `Unrolling::Generator` test; the streaming body gains the metadata and its row goes red.
#[test]
fn only_a_body_the_generator_unrolled_carries_the_no_unroll_metadata() {
    let arch = AmdArch::gfx1151();
    let tiled = tiled_spec(32, 16, (2, 2), (8, 4));
    let tiled_body = attention_body("tiled", &tiled, 0..tiled.workgroups(), 1).unwrap();
    let ir = poot_codegen::emit_llvm_ir(&tiled_body, Target::AmdGcn(arch)).unwrap();
    assert!(
        ir.contains(", !llvm.loop !0"),
        "no loop carries the metadata"
    );
    assert!(ir.contains("!1 = !{!\"llvm.loop.unroll.disable\"}"));
    let nvptx = poot_codegen::emit_llvm_ir(&tiled_body, Target::Nvptx).unwrap();
    assert!(!nvptx.contains("llvm.loop"), "NVPTX takes no loop metadata");

    let streaming = AttentionSpec {
        lq: 1,
        schedule: AttentionSchedule::Streaming { lanes: 64 },
        ..tiled
    };
    let streaming_body =
        attention_body("streaming", &streaming, 0..streaming.workgroups(), 1).unwrap();
    let ir = poot_codegen::emit_llvm_ir(&streaming_body, Target::AmdGcn(arch)).unwrap();
    assert!(
        !ir.contains("llvm.loop"),
        "the backend unrolls a streaming body"
    );
}
