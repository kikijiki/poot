//! POOT-1613 (C2 of the POOT-1609 design): the packed Gemv's launch figures of sm_89 and sm_86 are those SM's own
//! structural facts, four warp schedulers and 1536 resident lanes (CUDA programming guide, compute capability 8.6 and
//! 8.9), not the RDNA3.5 prior (issue 2, 1024 lanes) both planned with before. A projection over one activation row
//! drives the production `compile` on each part and the plan's Gemv geometry is compared with a literal pin and with
//! what the prior plans on the same part.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::calibration::Calibrated;
use poot_target::{Backend, DeviceCaps, Queried};

/// `(cols, rows, run, depth)` of a planned Gemv.
type Geometry = (u32, u32, u32, u32);

/// An NVPTX target of `compute_units` SMs planning with `calibrated`.
fn target(calibrated: Calibrated, compute_units: u32) -> Target {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: DeviceCaps {
            perf_calibration: calibrated,
            compute_units: Queried::Known(compute_units),
            ..caps
        },
    }
}

/// A packed projection `[n, k]` of `format` over one activation row.
fn projection(format: WeightFormat, n: usize, k: usize) -> Graph {
    let descriptor = PackedWeight::try_new(format, [n, k]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, k])).unwrap();
    let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
    b.finish(out).unwrap()
}

/// The plan's one packed Gemv geometry of the projection.
fn planned(target: &Target, format: WeightFormat, (n, k): (usize, usize)) -> Geometry {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    let program = compile(&projection(format, n, k), target, &options)
        .expect("the packed projection compiles");
    let schedules: Vec<Schedule> = program
        .planned()
        .filter_map(|(eqn, _)| match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect();
    let [
        Schedule::Gemv {
            cols,
            rows,
            unroll,
            depth,
            ..
        },
    ] = schedules[..]
    else {
        panic!("expected one packed Gemv, got {schedules:?}");
    };
    (cols, rows.get(), unroll, depth.get())
}

/// The RTX 4090's SM count, from the POOT-1609 design's device table (`data/devices.tsv`). Only the count shapes the
/// rounds; the plan names no part.
const SM_89_UNITS: u32 = 128;
/// The sm_86 part's SM count in the same table.
const SM_86_UNITS: u32 = 84;

/// The fixtures `(format, (n, k))`: the POOT-1609 sweep's cells that the corrected facts move.
const CELLS: [(WeightFormat, (usize, usize)); 6] = [
    (WeightFormat::Q4_K, (5120, 17408)),
    (WeightFormat::Q4_K, (17408, 5120)),
    (WeightFormat::Q4_K, (1024, 2560)),
    (WeightFormat::Q5_K, (17408, 5120)),
    (WeightFormat::Iq4_Xs, (17408, 5120)),
    (WeightFormat::Q8_0, (17408, 5120)),
];

/// The geometry each fixture plans on sm_89 and on sm_86 under the structural facts. sm_89's rates are the L1 rate its
/// corpus identified (POOT-1614) and priors; sm_86's are all priors, so its pins are the model's argmin at priors until a
/// pod sweep identifies them (`the_sm_89_gemv_geometry_is_within_the_measured_regret_of_the_fastest` is the measured
/// evidence for sm_89).
const PLANNED: [(&str, Calibrated, u32, [Geometry; 6]); 2] = [
    (
        "sm_89",
        Calibrated::SM_89,
        SM_89_UNITS,
        [
            (4, 2, 16, 1),
            (4, 2, 16, 1),
            (4, 2, 16, 3),
            (4, 2, 16, 1),
            (4, 4, 16, 1),
            (4, 4, 16, 1),
        ],
    ),
    (
        "sm_86",
        Calibrated::SM_86,
        SM_86_UNITS,
        [
            (4, 4, 16, 1),
            (4, 1, 16, 1),
            (8, 1, 16, 3),
            (4, 1, 16, 1),
            (4, 4, 16, 1),
            (4, 4, 16, 1),
        ],
    ),
];

/// Every fixture plans, on each of the two parts, the geometry the model picks with the SM's own issue rate and
/// resident lanes.
///
/// Mutation: set `SM_86_89_ISSUE_PER_CYCLE` to 2.0 and `SM_86_89_RESIDENT_LANES` to 1024 (the RDNA3.5 prior);
/// the fixtures plan the prior's geometries and the pins go red.
#[test]
fn the_gemv_geometry_of_sm_89_and_sm_86_follows_the_sms_structural_facts() {
    for (name, calibrated, units, pinned) in PLANNED {
        let target = target(calibrated, units);
        for ((format, shape), pin) in CELLS.into_iter().zip(pinned) {
            assert_eq!(
                planned(&target, format, shape),
                pin,
                "{name} {format:?} {shape:?}"
            );
        }
    }
}

/// The pins above are not inert: on the same part the RDNA3.5 prior plans another geometry in every fixture but
/// at most one, so a pin that held for both sets of facts would fail here.
#[test]
fn the_prior_plans_other_geometries_on_the_same_parts() {
    for (name, _, units, pinned) in PLANNED {
        let prior = target(Calibrated::Portable, units);
        let moved = CELLS
            .into_iter()
            .zip(pinned)
            .filter(|&((format, shape), pin)| planned(&prior, format, shape) != pin)
            .count();
        assert!(
            moved >= 5,
            "{name}: the prior's facts plan {moved} of {} fixtures differently",
            CELLS.len()
        );
    }
}
