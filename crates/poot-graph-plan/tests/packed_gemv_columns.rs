//! POOT-1319: the columns per workgroup of a packed decode Gemv are the device's measured figure, not a prior borrowed
//! from another part. A projection over one activation row drives the production `compile` on the PTX target of each
//! calibration and the plan's Gemv holds the columns of that calibration's table: wide while the outputs fill the
//! device (two workgroups per compute unit), narrow below.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::calibration::Calibrated;
use poot_target::{Backend, Queried};

/// The compute units of the RTX PRO 6000 Blackwell Server.
const COMPUTE_UNITS: u32 = 188;
/// The depth of the projection: a whole number of Q4_K super-blocks.
const K: usize = 2560;

fn nvptx(calibrated: Calibrated) -> Target {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: poot_target::DeviceCaps {
            perf_calibration: calibrated,
            compute_units: Queried::Known(COMPUTE_UNITS),
            ..caps
        },
    }
}

/// A Q4_K projection `[n, K]` over one activation row.
fn projection(n: usize) -> Graph {
    let descriptor = PackedWeight::try_new(WeightFormat::Q4_K, [n, K]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, K])).unwrap();
    let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
    b.finish(out).unwrap()
}

/// The columns of the program's one packed Gemv.
fn columns_of(program: &Program) -> u32 {
    let schedules: Vec<Schedule> = program
        .planned()
        .filter_map(|(eqn, _)| match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect();
    let [Schedule::Gemv { cols, .. }] = schedules[..] else {
        panic!("expected one packed Gemv, got {schedules:?}");
    };
    cols
}

fn planned_columns(n: usize, calibrated: Calibrated) -> u32 {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    columns_of(
        &compile(&projection(n), &nvptx(calibrated), &options)
            .expect("the packed projection compiles"),
    )
}

/// The sm_120 table is `{4, 2}` and the portable prior (Strix Halo's) `{16, 8}`: 17,408 outputs fill the device on
/// both (4,352 and 1,088 workgroups against 376), 1,024 outputs fill it on neither (256 and 64).
///
/// Mutation: spell `SM_120`'s `packed_gemv_cols` as `PORTABLE`'s; the sm_120 rows read 16 and 8.
#[test]
fn a_packed_gemv_takes_the_columns_of_its_devices_calibration() {
    for (n, sm_120, portable) in [(17_408, 4, 16), (1_024, 2, 8)] {
        assert_eq!(
            planned_columns(n, Calibrated::SM_120),
            sm_120,
            "sm_120, {n} outputs"
        );
        assert_eq!(
            planned_columns(n, Calibrated::Portable),
            portable,
            "portable, {n} outputs"
        );
    }
}
