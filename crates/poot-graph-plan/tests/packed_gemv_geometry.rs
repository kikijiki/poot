//! POOT-1562 (W5 of the POOT-1550 design): the geometry of a packed decode Gemv, `(cols, rows, run, depth)`, is the
//! minimum of the planner's cost model over the body the format's descriptor emits, scored with the device's
//! calibrated figures. No per-device table of columns or runs is consulted: none exists. A projection over one
//! activation row drives the production `compile` on the PTX target (sm_120's calibration) and the ROCm target
//! (gfx1151's, whose prior is the portable one), and the plan's Gemv is compared with an argmin this file computes
//! itself from the design's formula (`receipts/1550-design/README.md` section 3) over every geometry the generator
//! admits, and with the literal geometry each fixture planned when the constants were set.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, KernelChoice, PackedGemvPin, Program, Target, compile,
};
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

/// The RTX 4090: 128 SMs, its calibration.
fn sm_89() -> Target {
    target(Backend::Nvptx, Calibrated::SM_89, 128)
}

/// Strix Halo through HSA: 40 compute units, its calibration.
fn gfx1151() -> Target {
    target(
        Backend::AmdGcn(AmdArch::gfx1151()),
        Calibrated::STRIX_HALO_HSA,
        40,
    )
}

/// A `format` projection `[n, k]` over one activation row.
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

fn plan(target: &Target, format: WeightFormat, n: usize, k: usize) -> Schedule {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    planned_gemv(
        &compile(&projection(format, n, k), target, &options)
            .expect("the packed projection compiles"),
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

/// The model of POOT-1611 (the POOT-1609 design's T1-T3), restated: a dispatch lasts its fixed cost plus its rounds of
/// trips; the busiest unit holds whole workgroups; a trip lasts the larger of its work (the busiest unit's issue of its
/// warps' instructions, its L1's service of their lines, every column group's activation read paid) and one warp's own
/// chain, the DRAM latency plus its dependent instructions, shared over the trips a lane keeps in flight.
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
    let units = caps.compute_units.known().unwrap() as usize;
    let warp = match caps.subgroup {
        Queried::Known(poot_target::SubgroupSupport::Present { max_size, .. }) => max_size,
        _ => 32,
    };
    let lanes_per_col = width / cols;
    let trips = k.div_ceil((lanes_per_col * run) as usize) as f64;
    let workgroups = n.div_ceil((cols * rows) as usize);
    let per_unit = (device.resident_lanes / width).max(1) as usize;
    let rounds = workgroups.div_ceil(units * per_unit).max(1);
    let busiest = workgroups.div_ceil(rounds).div_ceil(units);
    let warps = (busiest * width as usize) as f64 / f64::from(warp);
    let lines = body.lines_per_trip(LaneMap {
        warp_lanes: warp,
        lanes_per_col,
    });
    let work = (warps * f64::from(body.warp_instructions) / device.issue_per_cycle)
        .max(warps * lines / device.l1_lines_per_cycle);
    let chain = (device.dram_latency_cycles
        + f64::from(body.warp_instructions) * device.dependent_issue_cycles)
        / f64::from(depth);
    Some(device.dispatch_cycles + rounds as f64 * trips * work.max(chain))
}

/// Every geometry of Q4_K (factor run 32) the generator admits at `shape` on `target`, scored.
fn scored(target: &Target, planned: Schedule, shape: (usize, usize)) -> Vec<(f64, Geometry)> {
    assert_eq!(
        packed_run_sharing(WeightFormat::Q4_K).unwrap(),
        poot_kernelgen::PackedRunSharing::FactorRun { factor_run: 32 }
    );
    let mut all = Vec::new();
    for rows in (1..=GemvRows::MAX).filter(|&rows| shape.0.is_multiple_of(rows as usize)) {
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
/// constants of `SM_120_GEMV_DEVICE` (fitted, POOT-1611) and `PORTABLE_GEMV_DEVICE` (a prior: gfx1151's picks are unmeasured, rows 4 included, until it has a corpus).
const PLANNED: [(&str, usize, Geometry); 6] = [
    ("sm_120", 0, (4, 2, 16, 1)),
    ("sm_120", 1, (4, 4, 16, 1)),
    ("sm_120", 2, (1, 2, 16, 1)),
    ("gfx1151", 0, (4, 4, 16, 1)),
    ("gfx1151", 1, (4, 4, 16, 1)),
    ("gfx1151", 2, (8, 2, 16, 1)),
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
            let planned = plan(target, WeightFormat::Q4_K, shape.0, shape.1);
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

/// The plan of the Q4_K projection `[n, k]` with the packed Gemv pinned to `pin`: its Gemv, or the compile's refusal.
fn plan_pinned(
    target: &Target,
    (n, k): (usize, usize),
    pin: PackedGemvPin,
) -> Result<Schedule, String> {
    let options = CompileOptions {
        packed_gemv_pin: Some(pin),
        ..CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact)
    };
    let program = compile(&projection(WeightFormat::Q4_K, n, k), target, &options)
        .map_err(|error| format!("{error:?}"))?;
    Ok(planned_gemv(&program))
}

/// POOT-1608: the calibration sweep times one geometry of the planner's own search space, so the pin plans exactly the
/// geometry it names, whatever the model would choose, and a geometry the generator refuses is its typed refusal, never
/// another geometry. Without the pin the committed model plans (4, 2, 16, 1) for this shape (`PLANNED` above).
///
/// Mutation: drop the pin test from `packed_gemv_geometry`'s candidate loop; the first assertion plans the model's
/// geometry (32, 1, 32, 3) instead of the pinned (8, 1, 16, 1) and goes red.
#[test]
fn a_pinned_packed_gemv_geometry_is_the_geometry_planned() {
    let target = sm_120();
    let shape = SHAPES[0];
    let pin = PackedGemvPin {
        cols: 8,
        rows: 1,
        run: 16,
        depth: 1,
    };
    let planned = plan_pinned(&target, shape, pin).expect("the pinned geometry plans");
    assert_eq!(geometry_of(planned), (8, 1, 16, 1), "{planned:?}");
    // Four rows of a 32-value run at depth 3 hold more registers than the generator's bound: refused by name.
    let refused = PackedGemvPin {
        rows: 4,
        run: 32,
        depth: 3,
        ..pin
    };
    let error = plan_pinned(&target, shape, refused)
        .expect_err("the generator refuses the pinned geometry");
    assert!(
        error.contains("ExceedsBound") && error.contains("registers a Gemv lane holds"),
        "{error}"
    );
}

/// The measured geometry sweep of the RTX PRO 6000 (sm_120): POOT-1608 and POOT-1610, one row per geometry,
/// `format NxK cols rows run depth device_us ..`.
const CORPUS: &str = include_str!("../../../benchmarks/calibration/ptx-sm120-packed-gemv.tsv");

/// The body facts of the generator for each geometry of [`CORPUS`], `format NxK cols rows run depth warp_instructions
/// lines`, the file `fit_packed_gemv.py` reads.
const BODY_FACTS: &str =
    include_str!("../../../benchmarks/calibration/ptx-sm120-packed-gemv-body.tsv");

/// The worst regret the planner's sm_120 geometry achieves over [`CORPUS`]'s cells, rounded up to 0.01: the pick of the
/// fitted model (`fit_packed_gemv.py`) is within this of the fastest measured geometry of every cell. The remaining
/// regret is rows 4 (Q4_K and IQ4_XS 17408x5120, Q8_0), which sm_120 measures slower than rows 2 and which register
/// residency (POOT-1609 T4) is to remove, and depth 2 on the smallest cell (Q6_K 1024x2560, 1.183), which the model
/// prices below depth 1 and sm_120 does not.
const MEASURED_REGRET_BOUND: f64 = 1.19;

/// One cell of [`CORPUS`]: a format and the shape `[n, k]`, its measured geometries by `(cols, rows, run, depth)`.
struct Cell {
    format: WeightFormat,
    name: String,
    shape: (usize, usize),
    measured: std::collections::BTreeMap<Geometry, f64>,
}

fn format_named(name: &str) -> WeightFormat {
    match name {
        "q4_k" => WeightFormat::Q4_K,
        "q5_k" => WeightFormat::Q5_K,
        "q6_k" => WeightFormat::Q6_K,
        "q8_0" => WeightFormat::Q8_0,
        "iq4_xs" => WeightFormat::Iq4_Xs,
        other => panic!("the corpus names an unknown format {other}"),
    }
}

/// The data rows of a TSV: its non-comment lines split on tabs.
fn tsv_rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.split('\t').collect())
}

fn geometry_in(fields: &[&str]) -> Geometry {
    let at = |i: usize| fields[i].parse::<u32>().expect("a geometry number");
    (at(2), at(3), at(4), at(5))
}

fn corpus_cells() -> Vec<Cell> {
    cells_of(CORPUS)
}

/// The cells of a corpus TSV.
fn cells_of(corpus: &'static str) -> Vec<Cell> {
    let mut cells: Vec<Cell> = Vec::new();
    for fields in tsv_rows(corpus) {
        let (n, k) = fields[1].split_once('x').expect("an NxK cell");
        let name = format!("{} {}", fields[0], fields[1]);
        let index = cells
            .iter()
            .position(|cell| cell.name == name)
            .unwrap_or_else(|| {
                cells.push(Cell {
                    format: format_named(fields[0]),
                    name,
                    shape: (n.parse().unwrap(), k.parse().unwrap()),
                    measured: Default::default(),
                });
                cells.len() - 1
            });
        cells[index]
            .measured
            .insert(geometry_in(&fields), fields[6].parse().expect("device_us"));
    }
    cells
}

/// POOT-1611: the geometry the production planner picks on sm_120, with no pin, is within the measured regret of the
/// fastest geometry the sweep timed, in every cell of the corpus (fifteen cells, five formats, 2440 geometries). The
/// planner's pick must itself be a measured geometry: a pick the corpus never timed fails, so the sweep covers whatever
/// the planner can choose. This is the evidence the model's structure and figures are right where the other tests in
/// this file only restate them: the modelled side (the planner) and the measured side (the corpus) differ.
///
/// Mutations, each run red then restored to green (POOT-1611 receipt): price the activation read once per warp
/// (`Stream::Activation => group_lines(..)` without `groups *` in `GemvBodyCost::lines_per_trip`), which brings run-32
/// picks back; and the retired latency arm `(latency + work) / depth` in `gemv_geometry.rs`'s `modelled_cycles`.
#[test]
fn the_sm_120_gemv_geometry_is_within_the_measured_regret_of_the_fastest() {
    let target = sm_120();
    let mut failures = Vec::new();
    let mut worst: f64 = 1.0;
    let cells = corpus_cells();
    assert_eq!(cells.len(), 15, "the corpus has fifteen cells");
    for cell in &cells {
        let (n, k) = cell.shape;
        let planned = geometry_of(plan(&target, cell.format, n, k));
        let Some(&time) = cell.measured.get(&planned) else {
            failures.push(format!(
                "planner chose (cols, rows, run, depth) {planned:?} for {}, which the corpus did not measure",
                cell.name
            ));
            continue;
        };
        let (&fastest, &best) = cell
            .measured
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("a cell has measured geometries");
        let regret = time / best;
        worst = worst.max(regret);
        if regret > MEASURED_REGRET_BOUND {
            failures.push(format!(
                "{}: planned {planned:?} {time:.1} us is {regret:.3} of the fastest {fastest:?} {best:.1} us, over the bound {MEASURED_REGRET_BOUND}",
                cell.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        worst > MEASURED_REGRET_BOUND - 0.01,
        "the bound {MEASURED_REGRET_BOUND} is not the worst achieved ({worst:.4}) rounded up to 0.01"
    );
}

/// The body facts file the fit reads is the generator's: for every geometry of the corpus the instructions and cache
/// lines `packed_gemv_body_cost` reads off the kernel on the PTX target equal the committed file's, so a change to the
/// generator or to `lines_per_trip` fails here until the file is regenerated and the fit re-run.
///
/// Mutation: restore the sharing credit in `lines_per_trip`; the geometries with several column groups per warp differ.
#[test]
fn the_corpus_body_facts_are_the_generators() {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    let mut rows = 0;
    let mut differing = Vec::new();
    for fields in tsv_rows(BODY_FACTS) {
        let (cols, rows_, run, depth) = geometry_in(&fields);
        let schedule = Schedule::Gemv {
            width: 256,
            cols,
            unroll: run,
            activation: poot_kernelgen::GemvActivation::Vec4,
            rows: GemvRows::new(rows_).unwrap(),
            depth: LoadDepth::new(depth).unwrap(),
        };
        let body = packed_gemv_body_cost(format_named(fields[0]), schedule, &caps)
            .expect("the corpus geometry is one the generator admits");
        let lines = body.lines_per_trip(LaneMap {
            warp_lanes: 32,
            lanes_per_col: 256 / cols,
        });
        let (instructions, file_lines): (u32, f64) =
            (fields[6].parse().unwrap(), fields[7].parse().unwrap());
        if (body.warp_instructions, lines) != (instructions, file_lines) {
            differing.push(format!(
                "{} {} {:?}: generator {} instructions {lines} lines, file {instructions} {file_lines}",
                fields[0],
                fields[1],
                (cols, rows_, run, depth),
                body.warp_instructions
            ));
        }
        rows += 1;
    }
    assert_eq!(
        rows,
        tsv_rows(CORPUS).count(),
        "a body-facts row per corpus row"
    );
    assert!(
        differing.is_empty(),
        "{} differ:\n{}",
        differing.len(),
        differing.join("\n")
    );
}

/// The measured geometry sweep of the RTX 4090 (sm_89), POOT-1614: five cells, 852 geometries, the same columns as
/// [`CORPUS`] (the lines and instructions columns are the generator's).
const CORPUS_SM_89: &str = include_str!("../../../benchmarks/calibration/ptx-sm89-packed-gemv.tsv");

/// The worst regret the planner's sm_89 geometry achieves over [`CORPUS_SM_89`]'s cells, rounded up to 0.01 (the worst
/// achieved is Q4_K 65536x5120 at 1.399, the rows-4 pick register residency, POOT-1609 T4, is to remove). It is above
/// the corpus's own run-to-run spread, 0.7-1.3 % (168 repeated rows).
const MEASURED_REGRET_BOUND_SM_89: f64 = 1.40;

/// Where the corpus measures run 32 slower than the best geometry by at least this factor, the planner must not pick
/// run 32. Every cell of [`CORPUS_SM_89`] has its best run-32 geometry 1.14-1.32 slower than its fastest, so 1.10 is
/// clear of the repeat spread.
const RUN_32_LOSES_BY: f64 = 1.10;

/// POOT-1613: the geometry the production planner picks on sm_89 with no pin is a measured geometry within the
/// measured regret of the fastest of its cell, and is not run 32 where the corpus says run 32 loses, in every cell of
/// the RTX 4090's corpus. With the RDNA3.5 prior that part planned with before (1024 resident lanes, issue 2) it picked
/// run 32 in all five cells (worst regret 1.462, POOT-1614). The modelled side (the planner) and the measured side
/// (the corpus) differ; the row restates no model.
///
/// Mutations, each run red then restored to green (POOT-1613 receipt): restore `SM_89`'s portable packed Gemv figures
/// (`packed_gemv: PORTABLE_GEMV_DEVICE`); and remove the L1 term from `modelled_cycles` in `gemv_geometry.rs`
/// (`service` of 0).
#[test]
fn the_sm_89_gemv_geometry_is_within_the_measured_regret_of_the_fastest() {
    let target = sm_89();
    let cells = cells_of(CORPUS_SM_89);
    assert_eq!(cells.len(), 5, "the corpus has five cells");
    let mut failures = Vec::new();
    let mut worst: f64 = 1.0;
    for cell in &cells {
        let (n, k) = cell.shape;
        let planned = geometry_of(plan(&target, cell.format, n, k));
        let Some(&time) = cell.measured.get(&planned) else {
            failures.push(format!(
                "planner chose (cols, rows, run, depth) {planned:?} for {}, which the corpus did not measure",
                cell.name
            ));
            continue;
        };
        let (&fastest, &best) = cell
            .measured
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("a cell has measured geometries");
        let regret = time / best;
        worst = worst.max(regret);
        if regret > MEASURED_REGRET_BOUND_SM_89 {
            failures.push(format!(
                "{}: planned {planned:?} {time:.1} us is {regret:.3} of the fastest {fastest:?} {best:.1} us, over the bound {MEASURED_REGRET_BOUND_SM_89}",
                cell.name
            ));
        }
        let best_run_32 = cell
            .measured
            .iter()
            .filter(|(geometry, _)| geometry.2 == 32)
            .map(|(_, &us)| us)
            .fold(f64::INFINITY, f64::min);
        if best_run_32 / best >= RUN_32_LOSES_BY && planned.2 == 32 {
            failures.push(format!(
                "{}: planned run 32 {planned:?}, where the best run-32 geometry takes {best_run_32:.1} us against {best:.1} us",
                cell.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        worst > MEASURED_REGRET_BOUND_SM_89 - 0.01,
        "the bound {MEASURED_REGRET_BOUND_SM_89} is not the worst achieved ({worst:.4}) rounded up to 0.01"
    );
}
