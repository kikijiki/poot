//! POOT-1489: where a per-row indexed contraction stops being a Gemv is the cost model's, not a row count. The Gemv
//! reads each row's expert straight from global memory; the register-tiled schedule over the rows grouped by expert
//! shares an expert's weight tile between its rows but pays for tiles that the rows do not fill. Which costs less
//! depends on the rows, the experts, the weight and the device (`RegisterTiling::gemv_beats_grouped`), and the
//! `CompileOptions::indexed_rows_pin` names one schedule for the calibration sweep.
//!
//! Every row drives the production `compile` and reads the indexed request the plan holds.

use poot_graph_ir::{Builder, Graph, IndexRange, MatmulPrecision, OpKind, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, IndexedRows, KernelChoice, Program, Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_target::{AmdArch, Backend};
use poot_tensor::DType;

/// The depth and columns of one expert's weight: the square projection of the microbench corpus.
const K: usize = 1024;
const N: usize = 1024;

fn rocm() -> Target {
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    Target::Gpu {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// `IndexedMatMul(x [m, K], w [E, N, K] bf16 behind a transpose, ids [m])`: a routed expert projection over a
/// checkpoint-order bf16 stack, the form the microbench times.
fn routed(m: usize, experts: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![experts, N, K], DType::BF16))
        .unwrap();
    let w = b.transpose(w, vec![0, 2, 1]).unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![m], DType::I32),
            IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let out = b.indexed_matmul(x, w, ids).unwrap();
    b.finish(out).unwrap()
}

fn compiled(m: usize, experts: usize, target: &Target, pin: Option<IndexedRows>) -> Program {
    let options = CompileOptions {
        indexed_rows_pin: pin,
        ..CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact)
    };
    compile(&routed(m, experts), target, &options).expect("the routed projection compiles")
}

/// The schedule of the program's one indexed contraction.
fn schedule_of(program: &Program) -> Schedule {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| {
            matches!(
                eqn.op,
                OpKind::IndexedMatMul { .. } | OpKind::ContractionEpilogue { .. }
            )
        })
        .expect("the indexed equation is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Indexed(request))) => {
            request.schedule
        }
        other => panic!("expected a generated indexed contraction, got {other:?}"),
    }
}

/// How the program runs the rows of its one indexed contraction.
fn rows_of(program: &Program) -> IndexedRows {
    match schedule_of(program) {
        Schedule::Gemv { .. } => IndexedRows::Gemv,
        Schedule::RegisterTiled { .. } | Schedule::FragmentTiled { .. } => IndexedRows::Grouped,
    }
}

fn grouping_equations(program: &Program) -> usize {
    program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::ExpertGroup { .. }))
        .count()
}

/// The cost model's boundary on gfx1151, measured by `calibrate-launch indexed` (POOT-1489): few experts and many rows share each expert's weight tile, so 256 rows
/// over 8 experts are grouped (the grouped tiling timed 0.6x the Gemv); many experts and the same rows share nothing,
/// so 256 rows over 128 experts are a Gemv (the grouped tiling timed 2.1x the Gemv, which the retired 32-row switch
/// would have planned), and so are 16 rows over 8. Past the model's boundary for 128 experts, 4096 rows are grouped
/// again. ROCm and wgpu agree on every row.
///
/// Mutation: make `gemv_wins` answer `true` whenever a Gemv launch exists (the Gemv for every row count); the
/// 256-rows-over-8-experts and 4096-rows rows go red naming the schedule. Make it answer `false`; every Gemv row goes
/// red.
#[test]
fn the_cost_model_decides_where_an_indexed_contraction_stops_being_a_gemv() {
    for (label, target) in [("rocm", rocm()), ("wgpu", wgpu())] {
        for (m, experts, want) in [
            (16, 8, IndexedRows::Gemv),
            (256, 8, IndexedRows::Grouped),
            (256, 128, IndexedRows::Gemv),
            (4096, 128, IndexedRows::Grouped),
        ] {
            let program = compiled(m, experts, &target, None);
            assert_eq!(
                rows_of(&program),
                want,
                "{label}: {m} rows over {experts} experts"
            );
            assert_eq!(
                grouping_equations(&program),
                usize::from(want == IndexedRows::Grouped),
                "{label}: {m} rows over {experts} experts: one grouping for the tiled schedule, none for the Gemv"
            );
        }
    }
}

/// The pin names the schedule whatever the model would choose, at one shape the model answers either way: 256 rows
/// over 128 experts are a Gemv unpinned and grouped pinned, 256 rows over 8 experts the reverse. Mutation: ignore the
/// pin in `gemv_wins`; the pinned rows plan the model's schedule and go red.
#[test]
fn a_pin_names_the_schedule_the_model_would_not_choose() {
    for (label, target) in [("rocm", rocm()), ("wgpu", wgpu())] {
        for (m, experts, model, pinned) in [
            (256, 128, IndexedRows::Gemv, IndexedRows::Grouped),
            (256, 8, IndexedRows::Grouped, IndexedRows::Gemv),
        ] {
            assert_eq!(
                rows_of(&compiled(m, experts, &target, None)),
                model,
                "{label}: the model's schedule for {m} rows over {experts} experts"
            );
            let program = compiled(m, experts, &target, Some(pinned));
            assert_eq!(
                rows_of(&program),
                pinned,
                "{label}: {m} rows over {experts} experts pinned"
            );
            assert_eq!(
                grouping_equations(&program),
                usize::from(pinned == IndexedRows::Grouped),
                "{label}: the grouping pass follows the pin"
            );
        }
    }
}

/// `matmul(x, transpose(w))` with a bf16 `[N, K]` checkpoint weight: the dense fold turns it into one
/// `DenseContraction` over the weight as stored.
fn dense(m: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// The register tiling `target` plans for the dense contraction of `m` rows: `(lanes edge, rows, cols)`.
fn dense_tiling(m: usize, target: &Target) -> (u32, u32, u32) {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    let program = compile(&dense(m), target, &options).expect("the dense contraction compiles");
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::DenseContraction { .. }))
        .expect("the dense contraction is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(spec))) => {
            match spec.schedule {
                Schedule::RegisterTiled {
                    lanes, rows, cols, ..
                } => (lanes.get(), rows.get(), cols.get()),
                other => panic!("expected a register tiling, got {other:?}"),
            }
        }
        other => panic!("expected a generated tiled contraction, got {other:?}"),
    }
}

/// The two backends of one silicon plan with their own cost ratios (POOT-1489): ROCm's identified LDS read and staged
/// value (9.88 and 8.43, POOT-1494), wgpu's portable prior (1.0 and 8.0, not identified behind RADV). The 256-row
/// `1024 x 1024` projection of the 1494 sweep is the shape where the ROCm fit moves the tiling (4 x 4 to 4 x 8
/// register tiles, 95 to 85 us on ROCm); wgpu keeps the 4 x 4. Mutation: give the wgpu device preset
/// (`poot_test_util::device_caps::wgpu_rdna3_igpu`) the `STRIX_HALO_HSA` calibration; the two targets plan alike and
/// this row goes red.
#[test]
fn the_two_backends_of_gfx1151_plan_with_their_own_ratios() {
    assert_eq!(dense_tiling(256, &rocm()), (16, 4, 8), "rocm");
    assert_eq!(dense_tiling(256, &wgpu()), (16, 4, 4), "wgpu");
}

/// A Gemv whose workgroups would exceed the device's X grid is widened (more columns per workgroup) until they fit,
/// not refused: 256 rows of 1024 columns are 262,144 outputs, which at the 4 columns of the dense table are 65,536
/// workgroups, one past wgpu's 65,535, and run at 8. (The retired 32-row switch never planned a Gemv this large, so
/// the dense indexed Gemv had no widening; the packed one always did.) Mutation: return the launch unwidened from
/// `widened_to_fit_grid`; the pinned Gemv is refused as an ungrouped contraction and this row goes red.
#[test]
fn a_gemv_whose_workgroups_exceed_the_grid_is_widened_not_refused() {
    let program = compiled(256, 8, &wgpu(), Some(IndexedRows::Gemv));
    let Schedule::Gemv { width, cols, .. } = schedule_of(&program) else {
        panic!("expected the pinned Gemv");
    };
    assert_eq!((width, cols), (256, 8));
}

fn nvptx(calibrated: poot_target::calibration::Calibrated) -> Target {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: poot_target::DeviceCaps {
            perf_calibration: calibrated,
            ..caps
        },
    }
}

/// The Gemv's direct value is the device's: an sm_120 part, measured to price a global read at 14.15 FMAs, groups the
/// rows of every cell the portable prior (7.6) groups and of cells beyond them, because the dearer Gemv loses sooner.
/// Mutation: spell `SM_120`'s `direct_value` as the portable prior's; no cell differs and the second assertion goes red.
#[test]
fn the_direct_value_of_the_device_moves_the_gemv_boundary() {
    let (portable, sm_120) = (
        nvptx(poot_target::calibration::Calibrated::Portable),
        nvptx(poot_target::calibration::Calibrated::SM_120),
    );
    let mut moved = Vec::new();
    for experts in [8, 32, 64, 128] {
        for m in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let on = |target: &Target| rows_of(&compiled(m, experts, target, None));
            let (prior, measured) = (on(&portable), on(&sm_120));
            if prior == IndexedRows::Grouped {
                assert_eq!(
                    measured,
                    IndexedRows::Grouped,
                    "{m} rows over {experts} experts: grouped on the prior, so on the dearer Gemv"
                );
            }
            if prior != measured {
                moved.push((m, experts));
            }
        }
    }
    assert!(
        !moved.is_empty(),
        "no cell of the grid moved between direct values 7.6 and 14.15"
    );
}
