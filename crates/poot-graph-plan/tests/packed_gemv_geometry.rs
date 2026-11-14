//! POOT-1562 (W5 of the POOT-1550 design): the geometry of a packed decode Gemv, `(cols, rows, run, depth)`, is the
//! minimum of the planner's cost model over the body the format's descriptor emits, scored with the device's
//! calibrated figures. No per-device table of columns or runs is consulted: none exists. A projection over one
//! activation row drives the production `compile` on the PTX target (sm_120's calibration) and the ROCm target
//! (gfx1151's, whose prior is the portable one), and the plan's Gemv is compared with an argmin this file computes
//! itself from the design's formula (`receipts/1550-design/README.md` section 3) over every geometry the generator
//! admits, and with the literal geometry each fixture planned when the constants were set.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{
    ContractionSpec, GemvRows, KernelRequest, LaneMap, LoadDepth, Schedule, packed_gemv_body_cost,
    packed_run_sharing,
};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::calibration::Calibrated;
use poot_target::{AmdArch, Backend, DeviceCaps, Queried};

/// What the model scores for one `(cols, rows, run, depth)`, in the planner's order of preference on a tie.
type Geometry = (u32, u32, u32, u32);

fn target(backend: Backend, calibrated: Calibrated, compute_units: u32) -> Target {
    let caps = poot_test_util::device_caps::default_caps_for(backend);
    Target::Gpu {
        backend,
        caps: DeviceCaps {
            perf_calibration: calibrated,
            compute_units: Queried::Known(compute_units),
            ..caps
        },
    }
}

/// The RTX PRO 6000 Blackwell Server: 188 SMs, its calibration.
fn sm_120() -> Target {
    target(Backend::Nvptx, Calibrated::SM_120, 188)
}

/// Strix Halo through HSA: 40 compute units, its calibration.
fn gfx1151() -> Target {
    target(
        Backend::AmdGcn(AmdArch::gfx1151()),
        Calibrated::STRIX_HALO_HSA,
        40,
    )
}

/// A Q4_K projection `[n, k]` over one activation row.
fn projection(n: usize, k: usize) -> Graph {
    let descriptor = PackedWeight::try_new(WeightFormat::Q4_K, [n, k]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, k])).unwrap();
    let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
    b.finish(out).unwrap()
}

/// The plan's one packed Gemv.
fn planned_gemv(program: &Program) -> Schedule {
    let schedules: Vec<Schedule> = program
        .planned()
        .filter_map(|(eqn, _)| match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect();
    let [schedule @ Schedule::Gemv { .. }] = schedules[..] else {
        panic!("expected one packed Gemv, got {schedules:?}");
    };
    schedule
}

fn plan(target: &Target, n: usize, k: usize) -> Schedule {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    planned_gemv(
        &compile(&projection(n, k), target, &options).expect("the packed projection compiles"),
    )
}

fn geometry_of(schedule: Schedule) -> Geometry {
    let Schedule::Gemv {
        cols,
        rows,
        unroll,
        depth,
        ..
    } = schedule
    else {
        panic!("{schedule:?} is not a Gemv");
    };
    (cols, rows.get(), unroll, depth.get())
}

/// The design's model, restated: a dispatch lasts its fixed cost plus its rounds of trips; a trip lasts the larger of
/// its work (the unit's issue of the warps' instructions, its L1's service of their lines) and its latency spread over
/// the trips a lane keeps in flight, `(latency + work) / depth`.
fn modelled_cycles(
    target: &Target,
    planned: Schedule,
    (n, k): (usize, usize),
    (cols, rows, run, depth): Geometry,
) -> Option<f64> {
    let Target::Gpu { caps, .. } = target else {
        unreachable!("a GPU target");
    };
    let Schedule::Gemv {
        width, activation, ..
    } = planned
    else {
        unreachable!("a Gemv");
    };
    let schedule = Schedule::Gemv {
        width,
        cols,
        unroll: run,
        activation,
        rows: GemvRows::new(rows).unwrap(),
        depth: LoadDepth::new(depth).unwrap(),
    };
    let body = packed_gemv_body_cost(WeightFormat::Q4_K, schedule, caps).ok()?;
    let device = caps.gemm_launch().calibrated.figures().packed_gemv;
    let units = f64::from(caps.compute_units.known().unwrap());
    let warp = match caps.subgroup {
        Queried::Known(poot_target::SubgroupSupport::Present { max_size, .. }) => max_size,
        _ => 32,
    };
    let lanes_per_col = width / cols;
    let trips = k.div_ceil((lanes_per_col * run) as usize) as f64;
    let launched = n.div_ceil((cols * rows) as usize) as f64 * f64::from(width);
    let rounds = (launched / (units * f64::from(device.resident_lanes)))
        .ceil()
        .max(1.0);
    let warps = launched / rounds / units / f64::from(warp);
    let lines = body.lines_per_trip(LaneMap {
        warp_lanes: warp,
        lanes_per_col,
    });
    let work = (warps * f64::from(body.warp_instructions) / device.issue_per_cycle)
        .max(warps * lines / device.l1_lines_per_cycle);
    let trip = work.max((device.dram_latency_cycles + work) / f64::from(depth));
    Some(device.dispatch_cycles + rounds * trips * trip)
}

/// Every geometry of Q4_K (factor run 32) the generator admits at `shape` on `target`, scored.
fn scored(target: &Target, planned: Schedule, shape: (usize, usize)) -> Vec<(f64, Geometry)> {
    assert_eq!(
        packed_run_sharing(WeightFormat::Q4_K).unwrap(),
        poot_kernelgen::PackedRunSharing::FactorRun { factor_run: 32 }
    );
    let mut all = Vec::new();
    for rows in 1..=GemvRows::MAX {
        for depth in 1..=LoadDepth::MAX {
            for run in [32, 16, 8, 4] {
                for cols in [1, 2, 4, 8, 16, 32] {
                    let geometry = (cols, rows, run, depth);
                    if let Some(cycles) = modelled_cycles(target, planned, shape, geometry) {
                        all.push((cycles, geometry));
                    }
                }
            }
        }
    }
    all
}

/// The fixture shapes `(n, k)`: a down and a gate projection of a 27B FFN, and a small one that does not fill sm_120.
const SHAPES: [(usize, usize); 3] = [(5120, 17408), (17408, 5120), (1024, 2560)];

/// The geometry each fixture plans on each calibration, `(cols, rows, run, depth)`: the model's minimum under the
/// constants of `SM_120_GEMV_DEVICE` and `PORTABLE_GEMV_DEVICE` (priors until POOT-1563 measures them).
const PLANNED: [(&str, usize, Geometry); 6] = [
    ("sm_120", 0, (32, 1, 32, 3)),
    ("sm_120", 1, (32, 1, 32, 2)),
    ("sm_120", 2, (4, 2, 16, 3)),
    ("gfx1151", 0, (32, 2, 32, 1)),
    ("gfx1151", 1, (32, 2, 32, 1)),
    ("gfx1151", 2, (32, 1, 32, 2)),
];

/// The geometry the planner picks is the model's minimum, and it is the one each fixture planned when the constants
/// were set. The first half is the model: the plan's geometry costs no more than any geometry the generator admits, by
/// this file's own restatement of the formula. The second pins the result, so a constant that changes the choice goes
/// red here and its provenance row is reread.
///
/// Mutation: swap `dram_latency_cycles` and `dispatch_cycles` in `SM_120_GEMV_DEVICE`; the sm_120 rows plan another
/// geometry and the pins go red.
#[test]
fn the_gemv_geometry_is_the_cost_model_minimum() {
    let devices = [("sm_120", sm_120()), ("gfx1151", gfx1151())];
    for (device, target) in &devices {
        for (index, &shape) in SHAPES.iter().enumerate() {
            let planned = plan(target, shape.0, shape.1);
            let chosen = geometry_of(planned);
            let all = scored(target, planned, shape);
            let (least, _) = all
                .iter()
                .copied()
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .expect("a geometry fits");
            let (cost, _) = all
                .iter()
                .copied()
                .find(|&(_, geometry)| geometry == chosen)
                .unwrap_or_else(|| {
                    panic!("{device} {shape:?}: {chosen:?} is not an admitted geometry")
                });
            assert!(
                cost <= least * (1.0 + 1e-9),
                "{device} {shape:?}: planned {chosen:?} at {cost:.0} cycles, the model's minimum is {least:.0}"
            );
            let (_, _, pinned) = PLANNED
                .iter()
                .copied()
                .find(|&(name, at, _)| name == *device && at == index)
                .expect("a pin per fixture");
            assert_eq!(chosen, pinned, "{device} {shape:?}");
        }
    }
}
