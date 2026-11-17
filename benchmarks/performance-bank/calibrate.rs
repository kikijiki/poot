//! `calibrate-launch`: fit the register-tiled contraction cost model's ratios to this device's measured time
//! (design POOT-1479 section 3.4, the POOT-1403 method), and write a [`PerfCalibration`] cost entry for review.
//!
//! Tiling modes, so no device is opened twice in one process (an HSA executor cannot be reopened after a drop):
//! - `shapes`: open the device and print the corpus, one `MxKxN<TAB>index` line per tiling the device runs for the
//!   shape (`RegisterTiling::candidates`, in order); with `--shape MxKxN`, that shape's candidates alone, one
//!   `index<TAB>schedule` line each;
//! - `measure --bank B --shape MxKxN --tiling INDEX`: open the device, pin that candidate tiling
//!   (`CompileOptions::tiling_pin`), time the plan of that dense contraction on the device clock and print one TSV
//!   row: the shape, the median device microseconds, and the cost-model terms of the pinned tiling;
//! - `fit --rows F --out E`: regress the rows' times (per `K` value) on their terms and write the entry (the ratios, the fit's
//!   quality and the rows it rests on) to `E`. It opens no device.
//!
//! Indexed modes (POOT-1489), for the Gemv-versus-grouped comparison of an indexed contraction:
//! - `indexed-cells`: print the corpus, one `ExMxKxN` line per cell (experts, routed rows, depth, columns);
//! - `indexed-measure --bank B --cell ExMxKxN --rows gemv|grouped`: pin the rows to that schedule
//!   (`CompileOptions::indexed_rows_pin`), time the plan on the device clock and print one TSV row with the cost-model
//!   terms of the schedule that ran;
//! - `indexed-choice`: plan every cell of the corpus unpinned (no run) and print the production planner's choice;
//! - `indexed-fit --rows F --choices C --out E`: sweep the Gemv's direct value, score the model's choice per cell against
//!   the measured faster schedule and the retired 32-row switch, and tabulate the production planner's choice. It opens
//!   no device.
//!
//! Gemv columns modes (POOT-1490), for the dense decode Gemv's columns per workgroup:
//! - `gemv-cells`: print the corpus, one `NxK<TAB>cols` line per (output count, columns) pair;
//! - `gemv-measure --bank B --cell NxK --cols C`: pin the Gemv's columns (`CompileOptions::gemv_cols_pin`), time the plan of
//!   that decode contraction on the device clock and print one TSV row: the cell, the pinned columns, the median
//!   device microseconds and the columns the planner chooses unpinned;
//! - `gemv-fit --rows F --out E`: for each output count, the columns that ran fastest and by how much, and which
//!   figures the sweep identifies. It opens no device.
//!
//! Packed Gemv geometry modes (POOT-1608), for the decode Gemv over one packed `[N, K]` weight whose launch geometry
//! `(cols, rows, run, depth)` the planner scores from the device's `GemvDevice` figures:
//! - `packed-gemv-cells --bank B`: one `format<TAB>NxK<TAB>cols<TAB>rows<TAB>run<TAB>depth` line per geometry of
//!   [`PACKED_GEMV_CORPUS`] the format's decode admits and the lane's registers hold;
//! - `packed-gemv-measure --bank B --format F --cell NxK --cols C --rows R --run U --depth D`: pin that geometry
//!   (`CompileOptions::packed_gemv_pin`), refuse unless the planner planned exactly it, time the plan on the device
//!   clock (the weights held as the bank's rotation copies, so a step streams them from memory) and print one TSV row:
//!   the cell, the geometry, the median device microseconds and the body facts the model scores that geometry with
//!   (warp instructions and cache lines per trip, the device's compute units and warp lanes, the weight bytes);
//! - the fit is `benchmarks/calibration/fit_packed_gemv.py`, which reads those rows. It opens no device.
//!
//! Packed grouped modes (POOT-1505), for a routed projection over a packed expert stack (a `packed_grouped_experts`
//! bank case) at several prompt lengths:
//! - `packed-cells --bank B --case ID`: one `tokens<TAB>precision<TAB>index` line per tiling the device runs for the
//!   case's rows at each of [`PACKED_TOKENS`], exact and narrow, each grouped at its own output-tile rows;
//! - `packed-measure --bank B --case ID --tokens T --precision exact|narrow|gemv --tiling INDEX|chosen`: pin that tiling
//!   (or none), time the plan on the device clock and print one TSV row with the tiling's cost-model terms;
//!
//! Each packed row prints the median device microseconds of the step, of its `expert_group` dispatch and of its
//! contraction dispatch, then the cost-model terms of the schedule that ran. `scripts/calibrate-launch rocm fragment`
//! measures the packed and dense rows `benchmarks/calibration/<device>.tsv` holds, which
//! `benchmarks/calibration/fit_fragment_cost.py` fits (POOT-1547). The precision `gemv` is the per-row Gemv.
//!
//! Dense modes (POOT-1547), the same row for a dense contraction:
//! - `dense-cells --bank B`: one `MxKxN<TAB>precision<TAB>index` line per tiling of the dense corpus, exact and narrow,
//!   and a `gemv` line for each one-row shape;
//! - `dense-measure --bank B --shape MxKxN --precision exact|narrow|gemv --tiling INDEX|chosen`: one TSV row.
//!
//! `scripts/calibrate-launch <backend>` runs them in order under the GPU lock. The corpus is a grid of shapes,
//! never a model's: one run calibrates every shape, and no shape-specific figure enters the table.
//!
//! The fit regresses over every tiling the device runs for every shape, each pinned so the planner runs exactly
//! that one. The tilings differ in tile geometry whatever the shape, so the corpus determines the ratios; the tool
//! refuses a corpus that does not.

use std::path::PathBuf;

use poot_executor::Device;
use poot_graph_plan::staged::StagedProgram;
use poot_graph_plan::{IndexedRows, KernelChoice};
use poot_kernelgen::{ContractionSpec, GemvShape, KernelRequest, Schedule};

use crate::runner::{Operands, facts_of, measure, prepare, staged};
use poot_graph_plan::Target;
use poot_kernelgen::{
    CostTerms, RegisterTiling, StagedTerms, TiledRows, TiledShape, fit_cost_ratios,
};
use poot_target::calibration::{Calibrated, CostRatios};
use serde_json::Value;

/// The `[N, K]` weights of the corpus: a square and a gated MLP's down projection at a 1024-wide hidden state.
const WEIGHTS: [(usize, usize); 2] = [(1024, 1024), (1024, 3072)];
/// The activation rows of the corpus: from a short prompt to a long one. Every tiling the device runs is timed for
/// each shape, one device process per tiling, so the corpus is kept to 8 shapes (80 timed rows on gfx1151).
const ROWS: [usize; 4] = [16, 64, 256, 1024];

/// The dense, exact, bf16-weight shape the planner's contraction tiling is chosen for (`planner/packed.rs`).
fn tiled_shape((m, k, n): (usize, usize, usize)) -> TiledShape {
    TiledShape {
        blocks: 1,
        m_per_block: m,
        out_per_block: n,
        k,
        layout: poot_kernelgen::WeightLayout::Nk,
        rows: TiledRows::Dense,
        terms: StagedTerms::ONE,
        weight: poot_kernelgen::StagedWeight::Dense { bits: 16 },
        activation_bits: 32,
    }
}

fn parse_shape(text: &str) -> Result<(usize, usize, usize), String> {
    let dims: Vec<usize> = text
        .split('x')
        .map(|d| d.parse().map_err(|e| format!("shape {text:?}: {e}")))
        .collect::<Result<_, _>>()?;
    match dims[..] {
        [m, k, n] => Ok((m, k, n)),
        _ => Err(format!("shape {text:?} is not MxKxN")),
    }
}

/// The value of `--flag` in the arguments.
fn flag(args: &[String], name: &str) -> Result<String, String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or_else(|| format!("{name} is required"))
}

/// The schedules of the register-tiled dense contractions `program` plans, in plan order.
fn dense_tiled_schedules_of(program: &StagedProgram) -> Vec<Schedule> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(
                spec,
            ))) => Some(spec.schedule),
            _ => None,
        })
        .collect()
}

/// The same schedules as their `Debug` text.
fn tiled_schedules_of(program: &StagedProgram) -> Vec<String> {
    dense_tiled_schedules_of(program)
        .into_iter()
        .map(|schedule| format!("{schedule:?}"))
        .collect()
}

/// The median device time, in microseconds, of the dense contraction `m x k x n` (an f32 activation against a bf16
/// `[N, K]` weight, exact operands) on `device` under `tiling`, and the schedules the planner gives it. The case is the bank's
/// `prefill-dense-bf16-nk` with its shape replaced, so the fixtures, the reference check and the timing policy are
/// the bank's own; a device that reports no timestamps is an error, never a fabricated time.
fn dense_contraction_device_us<D>(
    device: D,
    bank: &Value,
    (m, k, n): (usize, usize, usize),
    tiling: RegisterTiling,
) -> Result<(f64, Vec<String>), String>
where
    D: Device + 'static,
{
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == "prefill-dense-bf16-nk"))
        .ok_or("the bank has no case prefill-dense-bf16-nk")?
        .clone();
    case["shape"] = serde_json::json!({ "m": m, "k": k, "n": n });
    let ops = Operands {
        narrow: None,
        tiling: Some(tiling),
        indexed_rows: None,
        gemv_cols: None,
        packed_gemv: None,
    };
    let prepared = prepare(bank, &case, ops)?;
    let schedules = tiled_schedules_of(&staged(
        &prepared.graphs[0],
        device.target(),
        ops,
        &facts_of(&device),
    )?);
    let measured = measure(device, bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{m}x{k}x{n}: the output disagrees with the reference (worst {})",
            measured.worst
        ));
    }
    let mut samples = measured
        .device_us
        .ok_or("the device reports no timestamps; build it with device timing")?;
    samples.sort_by(f64::total_cmp);
    Ok((samples[samples.len() / 2], schedules))
}

/// The example's whole `main`: `device` opens a device with timestamps, for `shapes` and `measure`.
pub(crate) fn run<D>(
    backend: &str,
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("shapes") if flag(&args, "--shape").is_ok() => {
            // One shape's candidates, for an A/B of two tilings of a plan: `index<TAB>schedule`.
            let shape = parse_shape(&flag(&args, "--shape")?)?;
            let device = device()?;
            let Target::Gpu { caps, .. } = device.target() else {
                return Err("the device is not a GPU target".into());
            };
            for (index, tiling) in RegisterTiling::candidates(tiled_shape(shape), &caps).enumerate()
            {
                println!("{index}\t{:?}", tiling.schedule());
            }
            Ok(())
        }
        Some("shapes") => {
            let device = device()?;
            let Target::Gpu { caps, .. } = device.target() else {
                return Err("the device is not a GPU target".into());
            };
            for (n, k) in WEIGHTS {
                for m in ROWS {
                    for index in
                        0..RegisterTiling::candidates(tiled_shape((m, k, n)), &caps).count()
                    {
                        println!("{m}x{k}x{n}\t{index}");
                    }
                }
            }
            Ok(())
        }
        Some("measure") => {
            let bank_path = PathBuf::from(flag(&args, "--bank")?);
            let shape = parse_shape(&flag(&args, "--shape")?)?;
            let bank: Value = serde_json::from_slice(
                &std::fs::read(&bank_path)
                    .map_err(|e| format!("read {}: {e}", bank_path.display()))?,
            )
            .map_err(|e| format!("parse {}: {e}", bank_path.display()))?;
            let device = device()?;
            let Target::Gpu { caps, .. } = device.target() else {
                return Err("the device is not a GPU target".into());
            };
            let index: usize = flag(&args, "--tiling")?
                .parse()
                .map_err(|e| format!("--tiling: {e}"))?;
            let tiling = RegisterTiling::candidates(tiled_shape(shape), &caps)
                .nth(index)
                .ok_or_else(|| format!("{shape:?} has no candidate tiling {index}"))?;
            let terms = tiling.cost_terms(tiled_shape(shape), &caps);
            let (device_us, planned) = dense_contraction_device_us(device, &bank, shape, tiling)?;
            // Every dispatch (a large shape splits into several) must run the pinned tiling, or the row pairs a time with the
            // wrong terms.
            let schedule = format!("{:?}", tiling.schedule());
            if planned.is_empty() || planned.iter().any(|ran| *ran != schedule) {
                return Err(format!(
                    "{backend} {shape:?}: the planner ran {planned:?}, not the pinned tiling {schedule}"
                ));
            }
            println!(
                "{}x{}x{}\t{device_us}\t{}\t{}\t{}\t{}\t{}\t{schedule}",
                shape.0,
                shape.1,
                shape.2,
                terms.fma,
                terms.lds_reads,
                terms.staged_per_lane,
                terms.rounding,
                terms.occupancy,
            );
            Ok(())
        }
        Some("fit") => {
            let rows =
                std::fs::read_to_string(flag(&args, "--rows")?).map_err(|e| e.to_string())?;
            let out = PathBuf::from(flag(&args, "--out")?);
            let device_name = flag(&args, "--device")?;
            let mut samples = Vec::new();
            let mut table = Vec::new();
            for line in rows.lines().filter(|l| !l.trim().is_empty()) {
                let f: Vec<&str> = line.split('\t').collect();
                let number = |i: usize| {
                    f.get(i)
                        .and_then(|v| v.parse::<f64>().ok())
                        .ok_or_else(|| format!("bad row {line:?}"))
                };
                let terms = CostTerms {
                    fma: number(2)?,
                    lds_reads: number(3)?,
                    staged_per_lane: number(4)?,
                    direct_per_lane: 0.0,
                    rounding: number(5)?,
                    occupancy: number(6)?,
                    ..CostTerms::default()
                };
                // The model's time is per staged `K` value (`K` is common to every tiling of one shape, so it is left
                // out of the terms); the corpus varies `K`, so the measured time is put on the same footing.
                let k = parse_shape(f[0])?.1 as f64;
                samples.push((terms, number(1)? / k));
                table.push(format!(
                    "//   {} {:.1} us {}",
                    f[0],
                    number(1)?,
                    f.get(7).unwrap_or(&"")
                ));
            }
            let fit = fit_cost_ratios(&samples).ok_or(
                "the corpus does not determine the ratios (fewer than three rows, or one tile geometry)",
            )?;
            let entry = format!(
                "// calibrate-launch on {device_name} ({backend}), {} rows. Review before adding to `PerfCalibration`.\n\
                 // Fit quality: R^2 = {:.4}; one FMA issue of the model = {:.4e} us per K value.\n\
                 // The Gemv's direct_value comes from `indexed-fit`.\n\
                 cost: CostRatios {{\n    lds_read: {:.3},\n    staged_value: {:.3},\n    ..\n}},\n\
                 // Rows (shape, median device time, pinned schedule):\n{}\n",
                samples.len(),
                fit.r_squared,
                fit.seconds_per_fma,
                fit.lds_read,
                fit.staged_value,
                table.join("\n"),
            );
            std::fs::write(&out, &entry).map_err(|e| format!("write {}: {e}", out.display()))?;
            print!("{entry}");
            Ok(())
        }
        Some("indexed-cells") => {
            for cell in indexed_cells() {
                println!("{}", cell_text(cell));
            }
            Ok(())
        }
        Some("indexed-measure") => indexed_measure(&args, device),
        Some("indexed-choice") => indexed_choice(device),
        Some("indexed-fit") => indexed_fit(backend, &args),
        Some("gemv-cells") => {
            for cell in gemv_cells() {
                println!("{}\t{}", gemv_cell_text(cell.outputs), cell.cols);
            }
            Ok(())
        }
        Some("gemv-measure") => gemv_measure(&args, device),
        Some("gemv-fit") => gemv_fit(backend, &args),
        Some("dense-cells") => dense_cells(device),
        Some("dense-measure") => dense_measure(&args, device),
        Some("packed-gemv-cells") => packed_gemv_cells(&args, device),
        Some("packed-gemv-measure") => packed_gemv_measure(&args, device),
        Some("packed-cells") => packed_cells(&args, device),
        Some("packed-measure") => packed_measure(&args, device),
        other => Err(format!(
            "usage: calibrate_launch shapes|measure|fit|indexed-cells|indexed-measure|indexed-choice|indexed-fit|gemv-cells|gemv-measure|gemv-fit|dense-cells|dense-measure|packed-gemv-cells|packed-gemv-measure|packed-cells|packed-measure, got {other:?}"
        )),
    }
}

/// The experts of the indexed corpus: a small mixture, and the largest the checkpoints use.
const EXPERTS: [usize; 4] = [8, 32, 64, 128];
/// The routed rows of the indexed corpus: a decode step's selected experts, a few steps batched, and prefill chunks up to
/// 4096 routed rows (a 512-token chunk routed to 8 experts each).
const ROUTED_ROWS: [usize; 11] = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
/// The `(K, N)` of an expert's weight in the indexed corpus: a square projection and an up projection.
const EXPERT_WEIGHTS: [(usize, usize); 2] = [(1024, 1024), (2048, 768)];
/// The routed rows the retired `INDEXED_GEMV_MAX_ROWS` ran as the Gemv, kept here as the baseline the model is held to.
const RETIRED_GEMV_MAX_ROWS: usize = 32;

/// A cell of the indexed corpus: experts, routed rows, depth `K` and columns `N` of one expert's weight.
type Cell = (usize, usize, usize, usize);

fn parse_cell(text: &str) -> Result<Cell, String> {
    let dims: Vec<usize> = text
        .split('x')
        .map(|d| d.parse().map_err(|e| format!("cell {text:?}: {e}")))
        .collect::<Result<_, _>>()?;
    match dims[..] {
        [e, m, k, n] => Ok((e, m, k, n)),
        _ => Err(format!("cell {text:?} is not ExMxKxN")),
    }
}

fn cell_text((e, m, k, n): Cell) -> String {
    format!("{e}x{m}x{k}x{n}")
}

/// Every cell of the indexed corpus.
fn indexed_cells() -> impl Iterator<Item = Cell> {
    EXPERT_WEIGHTS.into_iter().flat_map(|(k, n)| {
        EXPERTS
            .into_iter()
            .flat_map(move |e| ROUTED_ROWS.into_iter().map(move |m| (e, m, k, n)))
    })
}

/// The bank's routed expert projection with its shape replaced by `cell`: `m` rows, each routed to one of `e`
/// experts, over a checkpoint-order bf16 stack rotated beyond the last-level cache, exact operands.
fn indexed_case(bank: &Value, (e, m, k, n): Cell) -> Result<Value, String> {
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|c| c["id"] == "expert-prefill-qwen3moe-up-narrow")
        })
        .ok_or("the bank has no case expert-prefill-qwen3moe-up-narrow")?
        .clone();
    case["shape"] = serde_json::json!({
        "tokens": m, "top_k": 1, "experts": e, "k": k, "n": n, "order": "routed"
    });
    case["cache"] = "weights-rotated-beyond-cache".into();
    Ok(case)
}

/// The routed expert projection of `cell` as a graph with no data, for planning alone.
fn indexed_graph((e, m, k, n): Cell) -> Result<poot_graph_ir::Graph, String> {
    let b = poot_graph_ir::Builder::new(poot_graph_ir::TraceLimits::for_tests());
    let x = b
        .constant("x", poot_graph_ir::TensorType::f32(vec![m, k]))
        .map_err(|e| e.to_string())?;
    let stack = b
        .constant(
            "w",
            poot_graph_ir::TensorType::new(vec![e, n, k], poot_tensor::DType::BF16),
        )
        .map_err(|e| e.to_string())?;
    let weight = b
        .transpose(stack, vec![0, 2, 1])
        .map_err(|e| e.to_string())?;
    let ids = b
        .slot_named_ranged(
            poot_graph_ir::Slot::Activation,
            "ids",
            poot_graph_ir::TensorType::new(vec![m], poot_tensor::DType::I32),
            poot_graph_ir::IndexRange::below(e).ok_or("no experts")?,
        )
        .map_err(|e| e.to_string())?;
    let y = b
        .indexed_matmul(x, weight, ids)
        .map_err(|e| e.to_string())?;
    b.finish(y).map_err(|e| e.to_string())
}

/// The schedules of the indexed contractions `program` plans, in plan order.
fn indexed_schedules_of(program: &StagedProgram) -> Vec<Schedule> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Indexed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect()
}

/// How `schedule` runs the rows of an indexed contraction: the Gemv, or a tiling of the rows grouped by expert.
fn rows_of(schedule: Schedule) -> IndexedRows {
    match schedule {
        Schedule::Gemv { .. } => IndexedRows::Gemv,
        _ => IndexedRows::Grouped,
    }
}

fn rows_name(rows: IndexedRows) -> &'static str {
    match rows {
        IndexedRows::Gemv => "gemv",
        IndexedRows::Grouped => "grouped",
    }
}

/// The cost-model terms of `schedule` over `cell` on `caps`.
fn indexed_terms(
    schedule: Schedule,
    (e, m, k, n): Cell,
    caps: &poot_target::DeviceCaps,
) -> Result<CostTerms, String> {
    match schedule {
        // The planner's Gemv shape (`GemvLaunch::cost_shape`): a workgroup owns `cols` column groups of `rows` rows each.
        Schedule::Gemv {
            width, cols, rows, ..
        } => Ok(GemvShape {
            outputs: m * n,
            k,
            width,
            cols: cols * rows.get(),
        }
        .cost_terms(caps)),
        other => {
            let tiling = RegisterTiling::of(other)
                .ok_or_else(|| format!("the schedule {other:?} is no tiling"))?;
            let shape = TiledShape {
                blocks: 1,
                m_per_block: m,
                out_per_block: n,
                k,
                layout: poot_kernelgen::WeightLayout::Nk,
                rows: TiledRows::Grouped {
                    groups: e + 1,
                    block: tiling.output_tile().0,
                },
                terms: StagedTerms::ONE,
                weight: poot_kernelgen::StagedWeight::Dense { bits: 16 },
                activation_bits: 32,
            };
            Ok(tiling.cost_terms(shape, caps))
        }
    }
}

/// The median device time, in microseconds, of the indexed contraction of `cell` on `device` with its rows pinned to
/// `rows`, and the schedules the planner gives it.
fn indexed_device_us<D>(
    device: D,
    bank: &Value,
    cell: Cell,
    rows: IndexedRows,
) -> Result<(f64, Vec<Schedule>), String>
where
    D: Device + 'static,
{
    let case = indexed_case(bank, cell)?;
    let ops = Operands {
        narrow: None,
        tiling: None,
        indexed_rows: Some(rows),
        gemv_cols: None,
        packed_gemv: None,
    };
    let prepared = prepare(bank, &case, ops)?;
    let schedules = indexed_schedules_of(&staged(
        &prepared.graphs[0],
        device.target(),
        ops,
        &facts_of(&device),
    )?);
    let measured = measure(device, bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{}: the output disagrees with the reference (worst {})",
            cell_text(cell),
            measured.worst
        ));
    }
    let mut samples = measured
        .device_us
        .ok_or("the device reports no timestamps; build it with device timing")?;
    samples.sort_by(f64::total_cmp);
    Ok((samples[samples.len() / 2], schedules))
}

/// `indexed-measure`: one row of the indexed dataset.
fn indexed_measure<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank_path = PathBuf::from(flag(args, "--bank")?);
    let bank: Value = serde_json::from_slice(
        &std::fs::read(&bank_path).map_err(|e| format!("read {}: {e}", bank_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", bank_path.display()))?;
    let cell = parse_cell(&flag(args, "--cell")?)?;
    let rows = match flag(args, "--rows")?.as_str() {
        "gemv" => IndexedRows::Gemv,
        "grouped" => IndexedRows::Grouped,
        other => return Err(format!("--rows is gemv or grouped, not {other:?}")),
    };
    let device = device()?;
    let Target::Gpu { caps, .. } = device.target() else {
        return Err("the device is not a GPU target".into());
    };
    let (device_us, schedules) = indexed_device_us(device, &bank, cell, rows)?;
    // Every dispatch must run the pinned rows, or the row pairs a time with the wrong terms.
    let Some(&schedule) = schedules.first() else {
        return Err(format!(
            "{}: no indexed contraction planned",
            cell_text(cell)
        ));
    };
    if schedules.iter().any(|ran| *ran != schedule) || rows_of(schedule) != rows {
        return Err(format!(
            "{}: the planner ran {schedules:?}, not the pinned {}",
            cell_text(cell),
            rows_name(rows)
        ));
    }
    let terms = indexed_terms(schedule, cell, &caps)?;
    println!(
        "{}\t{}\t{device_us}\t{}\t{}\t{}\t{}\t{}\t{}\t{schedule:?}",
        cell_text(cell),
        rows_name(rows),
        terms.fma,
        terms.lds_reads,
        terms.staged_per_lane,
        terms.direct_per_lane,
        terms.rounding,
        terms.occupancy,
    );
    Ok(())
}

/// `indexed-choice`: the production planner's choice for every cell of the corpus, unpinned and unrun.
fn indexed_choice<D>(device: impl FnOnce() -> Result<D, String>) -> Result<(), String>
where
    D: Device + 'static,
{
    let device = device()?;
    let ops = Operands {
        narrow: None,
        tiling: None,
        indexed_rows: None,
        gemv_cols: None,
        packed_gemv: None,
    };
    for cell in indexed_cells() {
        let schedules = indexed_schedules_of(&staged(
            &indexed_graph(cell)?,
            device.target(),
            ops,
            &facts_of(&device),
        )?);
        let Some(&schedule) = schedules.first() else {
            return Err(format!(
                "{}: no indexed contraction planned",
                cell_text(cell)
            ));
        };
        println!("{}\t{}", cell_text(cell), rows_name(rows_of(schedule)));
    }
    Ok(())
}

/// One measured row of the indexed dataset.
struct IndexedRow {
    cell: Cell,
    rows: IndexedRows,
    device_us: f64,
    terms: CostTerms,
}

fn parse_indexed_rows(text: &str) -> Result<Vec<IndexedRow>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let number = |i: usize| {
                f.get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .ok_or_else(|| format!("bad row {line:?}"))
            };
            let cell = parse_cell(f.first().ok_or("empty row")?)?;
            let rows = match f.get(1).copied() {
                Some("gemv") => IndexedRows::Gemv,
                Some("grouped") => IndexedRows::Grouped,
                other => return Err(format!("bad rows {other:?} in {line:?}")),
            };
            let fma = number(3)?;
            Ok(IndexedRow {
                cell,
                rows,
                device_us: number(2)?,
                terms: CostTerms {
                    fma,
                    lds_reads: number(4)?,
                    staged_per_lane: number(5)?,
                    direct_per_lane: number(6)?,
                    rounding: number(7)?,
                    occupancy: number(8)?,
                    // A Gemv lane's FMAs per `K` value are its share of its workgroup's dot products
                    // (`ContractionCost::gemv_terms`), so the row's dots follow from them and the cell's `K`.
                    dots_per_k: match rows {
                        IndexedRows::Gemv => fma / cell.2.max(1) as f64,
                        IndexedRows::Grouped => 0.0,
                    },
                    ..CostTerms::default()
                },
            })
        })
        .collect()
}

/// How well a direct value's choices serve the corpus.
struct Sweep {
    direct_value: f64,
    /// The worst time of the model's choice over the time of the retired switch's, over every cell.
    worst_vs_retired: f64,
    /// The mean over the cells of the model's time over the faster schedule's.
    mean_regret: f64,
    /// The cells where the model's choice is not the faster schedule.
    wrong: usize,
}

/// The model's choice of `cell` under `ratios`: the Gemv unless the grouped tiling costs strictly less.
fn model_choice(gemv: &IndexedRow, grouped: &IndexedRow, ratios: CostRatios) -> IndexedRows {
    if gemv.terms.modelled_time(ratios) <= grouped.terms.modelled_time(ratios) {
        IndexedRows::Gemv
    } else {
        IndexedRows::Grouped
    }
}

/// The `(gemv, grouped)` rows of every cell of the corpus that has both.
fn indexed_pairs(rows: &[IndexedRow]) -> Vec<(Cell, &IndexedRow, &IndexedRow)> {
    indexed_cells()
        .filter_map(|cell| {
            let of = |kind| rows.iter().find(|r| r.cell == cell && r.rows == kind);
            Some((cell, of(IndexedRows::Gemv)?, of(IndexedRows::Grouped)?))
        })
        .collect()
}

fn time_of(gemv: &IndexedRow, grouped: &IndexedRow, kind: IndexedRows) -> f64 {
    match kind {
        IndexedRows::Gemv => gemv.device_us,
        IndexedRows::Grouped => grouped.device_us,
    }
}

/// The retired `INDEXED_GEMV_MAX_ROWS` switch's schedule for `cell`.
fn retired_choice((_, m, _, _): Cell) -> IndexedRows {
    if m <= RETIRED_GEMV_MAX_ROWS {
        IndexedRows::Gemv
    } else {
        IndexedRows::Grouped
    }
}

/// How `direct_value` serves the corpus `pairs`, the other ratios being `base`'s.
fn score_direct_value(
    pairs: &[(Cell, &IndexedRow, &IndexedRow)],
    base: CostRatios,
    direct_value: f64,
) -> Sweep {
    let ratios = CostRatios {
        direct_value,
        ..base
    };
    let (mut worst, mut regret, mut wrong) = (0.0f64, 0.0, 0);
    for &(cell, gemv, grouped) in pairs {
        let choice = time_of(gemv, grouped, model_choice(gemv, grouped, ratios));
        let best = gemv.device_us.min(grouped.device_us);
        worst = worst.max(choice / time_of(gemv, grouped, retired_choice(cell)));
        regret += choice / best;
        wrong += usize::from(choice > best * 1.0001);
    }
    Sweep {
        direct_value,
        worst_vs_retired: worst,
        mean_regret: regret / pairs.len() as f64,
        wrong,
    }
}

/// The slowest the model's choice may be against the retired switch's on any cell and still count as no slower: the
/// owner's acceptance (POOT-1489) is that it is never slower, so only the rounding of the measured microseconds.
const NO_SLOWER: f64 = 1.001;

/// `indexed-fit`: sweep the Gemv's direct value over the dataset, name the range in which the model's choices are
/// never slower than the retired switch's on any cell ([`NO_SLOWER`]) and within 5 % of the best mean regret, take its
/// geometric centre, and tabulate every cell with the production planner's choice.
fn indexed_fit(backend: &str, args: &[String]) -> Result<(), String> {
    let rows = parse_indexed_rows(
        &std::fs::read_to_string(flag(args, "--rows")?).map_err(|e| e.to_string())?,
    )?;
    let choices: Vec<(Cell, IndexedRows)> = std::fs::read_to_string(flag(args, "--choices")?)
        .map_err(|e| e.to_string())?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let (cell, rows) = line.split_once('\t').ok_or("bad choice row")?;
            let rows = match rows {
                "gemv" => IndexedRows::Gemv,
                "grouped" => IndexedRows::Grouped,
                other => return Err(format!("bad choice {other:?}")),
            };
            Ok((parse_cell(cell)?, rows))
        })
        .collect::<Result<_, String>>()?;
    let out = PathBuf::from(flag(args, "--out")?);
    let base: CostRatios = if backend == "rocm" {
        Calibrated::STRIX_HALO_HSA
    } else {
        Calibrated::STRIX_HALO_VULKAN
    }
    .figures()
    .cost;
    let pairs = indexed_pairs(&rows);
    if pairs.is_empty() {
        return Err("the dataset has no cell with both schedules".into());
    }
    let sweep: Vec<Sweep> = std::iter::successors(Some(0.5f64), |v| Some(v * 1.12))
        .take_while(|v| *v <= 64.0)
        .map(|direct_value| score_direct_value(&pairs, base, direct_value))
        .collect();
    let best = sweep
        .iter()
        .filter(|s| s.worst_vs_retired <= NO_SLOWER)
        .map(|s| s.mean_regret)
        .min_by(f64::total_cmp)
        .ok_or("no direct value keeps the model within the noise of the retired switch")?;
    let admitted: Vec<&Sweep> = sweep
        .iter()
        .filter(|s| s.worst_vs_retired <= NO_SLOWER && s.mean_regret <= best * 1.05)
        .collect();
    let (low, high) = (
        admitted.first().map_or(0.0, |s| s.direct_value),
        admitted.last().map_or(0.0, |s| s.direct_value),
    );
    let centre = (low * high).sqrt();
    let at_centre = score_direct_value(&pairs, base, centre);
    let retired_regret = pairs
        .iter()
        .map(|&(cell, gemv, grouped)| {
            time_of(gemv, grouped, retired_choice(cell)) / gemv.device_us.min(grouped.device_us)
        })
        .sum::<f64>()
        / pairs.len() as f64;
    let mut report = format!(
        "// calibrate-launch indexed on {backend}, {} rows over {} cells. Review before adding to `PerfCalibration`.\n\
         // direct_value {centre:.2}: the geometric centre of {low:.2} to {high:.2}, the values at which the model's choice is\n\
         // never slower than the retired 32-row switch's on any cell and the mean regret is within 5 % of the best ({best:.4}).\n\
         // At the centre: worst model/retired {:.3}, mean regret {:.4} ({} cells where the faster schedule is not chosen);\n\
         // the retired switch: mean regret {retired_regret:.4}.\n\
         // lds_read {:.3}, staged_value {:.3} are the entry's (the tiling fit).\n\
         // sweep: direct_value, worst model/retired, mean regret, cells wrong\n",
        rows.len(),
        pairs.len(),
        at_centre.worst_vs_retired,
        at_centre.mean_regret,
        at_centre.wrong,
        base.lds_read,
        base.staged_value,
    );
    for s in &sweep {
        report.push_str(&format!(
            "// {:.2}\t{:.3}\t{:.4}\t{}\n",
            s.direct_value, s.worst_vs_retired, s.mean_regret, s.wrong
        ));
    }
    report.push_str(
        "// cell, t_gemv us, t_grouped us, faster, model at the centre, planner, retired 32-switch, planner/faster, planner/retired\n",
    );
    let centre_ratios = CostRatios {
        direct_value: centre,
        ..base
    };
    let (mut worst_planner, mut planner_slower) = (0.0f64, 0);
    for &(cell, gemv, grouped) in &pairs {
        let faster = if gemv.device_us <= grouped.device_us {
            IndexedRows::Gemv
        } else {
            IndexedRows::Grouped
        };
        let model = model_choice(gemv, grouped, centre_ratios);
        let retired = retired_choice(cell);
        let planner = choices
            .iter()
            .find(|(c, _)| *c == cell)
            .map_or(model, |&(_, kind)| kind);
        let vs_retired = time_of(gemv, grouped, planner) / time_of(gemv, grouped, retired);
        worst_planner = worst_planner.max(vs_retired);
        planner_slower += usize::from(vs_retired > NO_SLOWER);
        report.push_str(&format!(
            "{}\t{:.1}\t{:.1}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\n",
            cell_text(cell),
            gemv.device_us,
            grouped.device_us,
            rows_name(faster),
            rows_name(model),
            rows_name(planner),
            rows_name(retired),
            time_of(gemv, grouped, planner) / gemv.device_us.min(grouped.device_us),
            vs_retired,
        ));
    }
    report.push_str(&format!(
        "// {} cells; the planner's choice is slower than the retired 32-switch's in {planner_slower}; worst planner/retired = {worst_planner:.3}\n",
        pairs.len(),
    ));
    std::fs::write(&out, &report).map_err(|e| format!("write {}: {e}", out.display()))?;
    print!("{report}");
    Ok(())
}

/// The depth of the Gemv corpus: a hidden state of a mid-size model. The sweep varies the output count, which sets
/// whether the wide columns still fill the device.
const GEMV_K: usize = 2048;
/// The output counts of the Gemv corpus: from a launch that starves any device to one that floods it.
const GEMV_OUTPUTS: [usize; 5] = [512, 2048, 8192, 32768, 131072];
/// The columns per workgroup the sweep pins: every power of two the 256-lane Gemv width divides up to a 16-column group.
const GEMV_COLS: [u32; 5] = [1, 2, 4, 8, 16];
/// The margin by which the fastest columns must beat the runner-up for a cell to name them: below it the cell is
/// a tie and identifies nothing.
const GEMV_MARGIN: f64 = 1.05;

/// A cell of the Gemv corpus: `outputs` dot products of `GEMV_K` values, launched at `cols` columns per workgroup.
#[derive(Clone, Copy)]
struct GemvCell {
    outputs: usize,
    cols: u32,
}

fn gemv_cell_text(outputs: usize) -> String {
    format!("{outputs}x{GEMV_K}")
}

fn gemv_cells() -> impl Iterator<Item = GemvCell> {
    GEMV_OUTPUTS.into_iter().flat_map(|outputs| {
        GEMV_COLS
            .into_iter()
            .map(move |cols| GemvCell { outputs, cols })
    })
}

/// The Gemv schedules of the dense decode contractions `program` plans, in plan order.
fn dense_gemv_schedules_of(program: &StagedProgram) -> Vec<Schedule> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
                schedule,
                ..
            })) => Some(*schedule),
            _ => None,
        })
        .collect()
}

/// The columns of the Gemv `schedules` launch, `None` when one is not a Gemv or they disagree.
fn gemv_cols_of(schedules: &[Schedule]) -> Option<u32> {
    let mut cols = schedules.iter().map(|s| match s {
        Schedule::Gemv { cols, .. } => Some(*cols),
        _ => None,
    });
    let first = cols.next()??;
    cols.all(|c| c == Some(first)).then_some(first)
}

/// The bank's dense `[N, K]` bf16 projection as one decode row of `outputs` columns, so its fixtures, reference
/// check and timing policy are the bank's own.
fn gemv_case(bank: &Value, outputs: usize) -> Result<Value, String> {
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == "prefill-dense-bf16-nk"))
        .ok_or("the bank has no case prefill-dense-bf16-nk")?
        .clone();
    case["shape"] = serde_json::json!({ "m": 1, "k": GEMV_K, "n": outputs });
    case["cache"] = "weights-rotated-beyond-cache".into();
    Ok(case)
}

/// `gemv-measure`: one row of the Gemv dataset.
fn gemv_measure<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank_path = PathBuf::from(flag(args, "--bank")?);
    let bank: Value = serde_json::from_slice(
        &std::fs::read(&bank_path).map_err(|e| format!("read {}: {e}", bank_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", bank_path.display()))?;
    let cell = flag(args, "--cell")?;
    let outputs = cell
        .strip_suffix(&format!("x{GEMV_K}"))
        .and_then(|n| n.parse::<usize>().ok())
        .ok_or_else(|| format!("cell {cell:?} is not NxK with K = {GEMV_K}"))?;
    let cols: u32 = flag(args, "--cols")?
        .parse()
        .map_err(|e| format!("--cols: {e}"))?;
    let device = device()?;
    let case = gemv_case(&bank, outputs)?;
    let unpinned = Operands {
        narrow: None,
        tiling: None,
        indexed_rows: None,
        gemv_cols: None,
        packed_gemv: None,
    };
    let prepared = prepare(&bank, &case, unpinned)?;
    let chosen = gemv_cols_of(&dense_gemv_schedules_of(&staged(
        &prepared.graphs[0],
        device.target(),
        unpinned,
        &facts_of(&device),
    )?));
    let ops = Operands {
        gemv_cols: Some(cols),
        ..unpinned
    };
    let ran = gemv_cols_of(&dense_gemv_schedules_of(&staged(
        &prepared.graphs[0],
        device.target(),
        ops,
        &facts_of(&device),
    )?));
    if ran != Some(cols) {
        return Err(format!(
            "{cell}: the planner launched {ran:?} columns, not the pinned {cols}"
        ));
    }
    let measured = measure(device, &bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{cell}: the output disagrees with the reference (worst {})",
            measured.worst
        ));
    }
    let mut samples = measured
        .device_us
        .ok_or("the device reports no timestamps; build it with device timing")?;
    samples.sort_by(f64::total_cmp);
    let chosen = chosen.ok_or_else(|| format!("{cell}: the unpinned plan is no Gemv"))?;
    println!("{cell}\t{cols}\t{}\t{chosen}", samples[samples.len() / 2]);
    Ok(())
}

/// The measured rows of one Gemv cell: the columns the planner chooses unpinned, and each pinned columns' median time.
struct GemvTimes {
    cell: String,
    chosen: u32,
    times: Vec<(u32, f64)>,
}

/// `gemv-fit`: per output count, the fastest columns and its margin over the runner-up, and the columns the planner chose.
fn gemv_fit(backend: &str, args: &[String]) -> Result<(), String> {
    let rows = std::fs::read_to_string(flag(args, "--rows")?).map_err(|e| e.to_string())?;
    let out = PathBuf::from(flag(args, "--out")?);
    let mut cells: Vec<GemvTimes> = Vec::new();
    for line in rows.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let [cell, cols, us, chosen] = f[..] else {
            return Err(format!("bad row {line:?}"));
        };
        let bad = |e: std::num::ParseFloatError| format!("bad row {line:?}: {e}");
        let (cols, chosen): (u32, u32) = (
            cols.parse().map_err(|_| format!("bad row {line:?}"))?,
            chosen.parse().map_err(|_| format!("bad row {line:?}"))?,
        );
        let us: f64 = us.parse().map_err(bad)?;
        match cells.iter_mut().find(|c| c.cell == cell) {
            Some(known) => known.times.push((cols, us)),
            None => cells.push(GemvTimes {
                cell: cell.to_string(),
                chosen,
                times: vec![(cols, us)],
            }),
        }
    }
    if cells.is_empty() {
        return Err("no rows".into());
    }
    let mut report = format!(
        "// calibrate-launch gemv-fit ({backend}), {} cells. Review before editing `PerfCalibration::dense_gemv_cols`.\n\
         // A cell names its fastest columns only when they beat the runner-up by {GEMV_MARGIN}x; else it is a tie.\n\
         // cell, planner cols, fastest cols, runner-up/fastest, then cols=us per pinned columns:\n",
        cells.len()
    );
    for GemvTimes {
        cell,
        chosen,
        times,
    } in &mut cells
    {
        times.sort_by(|a, b| a.1.total_cmp(&b.1));
        let margin = times.get(1).map_or(1.0, |second| second.1 / times[0].1);
        let named = if margin >= GEMV_MARGIN {
            times[0].0.to_string()
        } else {
            "tie".to_string()
        };
        let mut by_cols = times.clone();
        by_cols.sort_by_key(|t| t.0);
        let listed: Vec<String> = by_cols
            .iter()
            .map(|(c, us)| format!("{c}={us:.1}"))
            .collect();
        report += &format!(
            "//   {cell}\t{chosen}\t{named}\t{margin:.3}\t{}\n",
            listed.join(" ")
        );
    }
    std::fs::write(&out, &report).map_err(|e| format!("write {}: {e}", out.display()))?;
    print!("{report}");
    Ok(())
}

/// The cells of the packed Gemv corpus (POOT-1608, POOT-1610): a bank `storage` name and the `[N, K]` weight of a decode
/// projection (a gate or up, a down, and the head of the 27B; a short-`K` and a small-`N` projection of the 4B, and POOT-1611's three more small decode cells of the 4B: its
/// attention output and query projections and a Q6_K value projection, the latency-bound end of the model), each
/// swept over the whole geometry space the planner searches. The formats are the ones whose decode admits a 32-value
/// run (Q4_K, Q5_K, IQ4_XS: the planner's run 32 of POOT-1606) and two it does not (Q6_K, Q8_0), so a figure that fits
/// the first cannot flip the second unseen.
const PACKED_GEMV_CORPUS: [(&str, usize, usize); 15] = [
    ("q4_k", 17408, 5120),
    ("q4_k", 5120, 17408),
    ("q4_k", 2560, 9728),
    ("q4_k", 65536, 5120),
    ("q4_k", 9728, 2560),
    ("q4_k", 1024, 2560),
    ("q5_k", 17408, 5120),
    ("q5_k", 5120, 17408),
    ("iq4_xs", 17408, 5120),
    ("iq4_xs", 5120, 17408),
    ("q6_k", 17408, 5120),
    ("q8_0", 17408, 5120),
    ("q4_k", 2560, 4096),
    ("q4_k", 4096, 2560),
    ("q6_k", 1024, 2560),
];
/// The columns, runs, rows and depths a cell sweeps: the planner's `COLS` and runs, `GemvRows` (those dividing the
/// cell's `N`, as the planner's) and `LoadDepth`.
const PACKED_GEMV_COLS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const PACKED_GEMV_RUNS: [u32; 4] = [32, 16, 8, 4];

/// The `(rows, depth)` pairs a cell with `n` outputs sweeps.
fn rows_and_depths(n: usize) -> Vec<(u32, u32)> {
    (1..=poot_kernelgen::GemvRows::MAX)
        .filter(|&rows| n.is_multiple_of(rows as usize))
        .flat_map(|rows| (1..=poot_kernelgen::LoadDepth::MAX).map(move |depth| (rows, depth)))
        .collect()
}

/// The Gemv schedule of a geometry over a weight of `k` values on the device `caps`, as the planner builds it for the
/// Vec4 activation read every run of this corpus takes.
fn packed_gemv_schedule(
    caps: &poot_target::DeviceCaps,
    (cols, rows, run, depth): (u32, u32, u32, u32),
) -> Result<Schedule, String> {
    Ok(Schedule::Gemv {
        width: caps.gemm_launch().gemv_width,
        cols,
        unroll: run,
        activation: poot_kernelgen::GemvActivation::Vec4,
        rows: poot_kernelgen::GemvRows::new(rows).map_err(|e| format!("rows {rows}: {e:?}"))?,
        depth: poot_kernelgen::LoadDepth::new(depth)
            .map_err(|e| format!("depth {depth}: {e:?}"))?,
    })
}

/// `packed-gemv-cells`: the geometries of the corpus the format's decode admits and the lane's registers hold.
fn packed_gemv_cells<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let _ = read_bank(args)?;
    let device = device()?;
    let Target::Gpu { caps, .. } = device.target() else {
        return Err("the device is not a GPU target".into());
    };
    for (storage, n, k) in PACKED_GEMV_CORPUS {
        let format = crate::runner::packed_format(storage)?;
        let sharing = poot_kernelgen::packed_run_sharing(format).map_err(|e| format!("{e:?}"))?;
        for run in PACKED_GEMV_RUNS {
            let admitted = match sharing {
                poot_kernelgen::PackedRunSharing::FactorRun { factor_run } => {
                    factor_run.is_multiple_of(run as usize)
                }
                _ => true,
            };
            if !admitted {
                continue;
            }
            for (rows, depth) in rows_and_depths(n) {
                for cols in PACKED_GEMV_COLS {
                    let schedule = packed_gemv_schedule(&caps, (cols, rows, run, depth))?;
                    if poot_kernelgen::packed_gemv_body_cost(format, schedule, &caps).is_ok() {
                        println!("{storage}\t{n}x{k}\t{cols}\t{rows}\t{run}\t{depth}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// The packed Gemv of the plan `program` holds, in plan order.
fn packed_gemv_schedules_of(program: &StagedProgram) -> Vec<Schedule> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect()
}

/// `packed-gemv-measure`: one row of the packed Gemv corpus.
fn packed_gemv_measure<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank = read_bank(args)?;
    let storage = flag(args, "--format")?;
    let cell = flag(args, "--cell")?;
    let (n, k) = cell
        .split_once('x')
        .and_then(|(n, k)| Some((n.parse::<usize>().ok()?, k.parse::<usize>().ok()?)))
        .ok_or_else(|| format!("cell {cell:?} is not NxK"))?;
    let number = |name: &str| -> Result<u32, String> {
        flag(args, name)?
            .parse()
            .map_err(|e| format!("{name}: {e}"))
    };
    let geometry = (
        number("--cols")?,
        number("--rows")?,
        number("--run")?,
        number("--depth")?,
    );
    let format = crate::runner::packed_format(&storage)?;
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == "packed-decode-gemv"))
        .ok_or("the bank has no case packed-decode-gemv")?
        .clone();
    case["storage"] = storage.clone().into();
    case["shape"] = serde_json::json!({ "m": 1, "k": k, "n": n });
    let device = device()?;
    let Target::Gpu { caps, .. } = device.target() else {
        return Err("the device is not a GPU target".into());
    };
    let (cols, rows, run, depth) = geometry;
    let ops = Operands {
        packed_gemv: Some(poot_graph_plan::PackedGemvPin {
            cols,
            rows,
            run,
            depth,
        }),
        ..Operands::of(device.target(), &case)?
    };
    let prepared = prepare(&bank, &case, ops)?;
    let ran = packed_gemv_schedules_of(&staged(
        &prepared.graphs[0],
        device.target(),
        ops,
        &facts_of(&device),
    )?);
    let wanted = packed_gemv_schedule(&caps, geometry)?;
    let [
        Schedule::Gemv {
            width,
            cols: ran_cols,
            unroll,
            rows: ran_rows,
            depth: ran_depth,
            ..
        },
    ] = ran[..]
    else {
        return Err(format!(
            "{storage} {cell}: the planner ran {ran:?}, not a Gemv"
        ));
    };
    if (ran_cols, ran_rows.get(), unroll, ran_depth.get()) != geometry {
        return Err(format!(
            "{storage} {cell}: the planner ran {ran:?}, not the pinned {wanted:?}"
        ));
    }
    let body = poot_kernelgen::packed_gemv_body_cost(format, ran[0], &caps)
        .map_err(|e| format!("{storage} {cell}: {e:?}"))?;
    let warp =
        match caps.subgroup {
            poot_target::Queried::Known(poot_target::SubgroupSupport::Present {
                max_size, ..
            }) if max_size > 0 => max_size,
            _ => 32,
        };
    let lines = body.lines_per_trip(poot_kernelgen::LaneMap {
        warp_lanes: warp,
        lanes_per_col: width / cols,
    });
    let units = caps
        .compute_units
        .known()
        .ok_or("the device reports no compute units")?;
    let descriptor = poot_quant::PackedWeight::try_new(format, [n, k])
        .map_err(|e| format!("{storage} {cell}: {e:?}"))?;
    let weight_bytes: usize = descriptor
        .sources()
        .into_iter()
        .map(|role| descriptor.slab_bytes(role))
        .sum();
    let measured = measure(device, &bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{storage} {cell}: the output disagrees with the reference (worst {})",
            measured.worst
        ));
    }
    let mut samples = measured
        .device_us
        .ok_or("the device reports no timestamps; build it with device timing")?;
    samples.sort_by(f64::total_cmp);
    println!(
        "{storage}\t{cell}\t{cols}\t{rows}\t{run}\t{depth}\t{}\t{}\t{lines}\t{units}\t{warp}\t{width}\t{weight_bytes}",
        samples[samples.len() / 2],
        body.warp_instructions,
    );
    Ok(())
}

/// The token counts of the packed grouped corpus (POOT-1505): a short prompt to a long prefill chunk, so the routed
/// rows per expert of a wide mixture run from a few (each grouped tile mostly padding) to many.
const PACKED_TOKENS: [usize; 3] = [128, 512, 2048];

/// The bank case `id` with its token count replaced.
fn packed_case(bank: &Value, id: &str, tokens: usize) -> Result<Value, String> {
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == id))
        .ok_or_else(|| format!("the bank has no case {id}"))?
        .clone();
    if case["producer"] != "packed_grouped_experts" {
        return Err(format!("{id} is not a packed_grouped_experts case"));
    }
    case["shape"]["tokens"] = tokens.into();
    Ok(case)
}

/// The precision a `--precision` names, and the operands of a measurement at it on `target` with its rows grouped:
/// the bank's own narrow operands ([`Operands::of`]).
fn packed_operands(name: &str, target: Target) -> Result<crate::runner::Operands, String> {
    let (precision, rows) = match name {
        "exact" | "narrow" => (name, IndexedRows::Grouped),
        // The per-row Gemv the exact twin runs where it wins: exact operands, no grouping.
        "gemv" => ("exact", IndexedRows::Gemv),
        other => {
            return Err(format!(
                "--precision is exact, narrow or gemv, not {other:?}"
            ));
        }
    };
    let case = serde_json::json!({ "numerics": { "operand_precision": precision } });
    Ok(Operands {
        indexed_rows: Some(rows),
        ..Operands::of(target, &case)?
    })
}

/// The grouped tiled shape of the packed case at `tokens` under `ops`, its block the tile rows the caller pins (the
/// planner's own `packed_indexed_shape`, from the case's dimensions).
fn packed_grouped_shape(
    case: &Value,
    ops: crate::runner::Operands,
    block: usize,
) -> Result<TiledShape, String> {
    let shape = &case["shape"];
    let dim = |name: &str| {
        shape[name]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("case shape has no {name}"))
    };
    let format = crate::runner::packed_format(case["storage"].as_str().unwrap_or_default())?;
    let precision = match ops.narrow {
        None => poot_graph_ir::OperandPrecision::Exact,
        Some(dtype) => poot_graph_ir::OperandPrecision::Narrow(dtype),
    };
    Ok(TiledShape {
        blocks: 1,
        m_per_block: dim("tokens")? * shape["top_k"].as_u64().unwrap_or(1) as usize,
        out_per_block: dim("n")?,
        k: dim("k")?,
        layout: poot_kernelgen::WeightLayout::Nk,
        rows: TiledRows::Grouped {
            groups: dim("experts")? + 1,
            block,
        },
        terms: StagedTerms::of(precision, None, None),
        weight: poot_kernelgen::StagedWeight::Packed(format),
        activation_bits: poot_kernelgen::activation_bits(None),
    })
}

/// The tilings the device runs for the packed case's rows at any block: the candidates of the shape with its rows
/// free, each one grouping at its own output-tile rows.
fn packed_candidates(
    case: &Value,
    ops: crate::runner::Operands,
    caps: &poot_target::DeviceCaps,
) -> Result<Vec<RegisterTiling>, String> {
    let free = TiledShape {
        rows: TiledRows::Dense,
        ..packed_grouped_shape(case, ops, 0)?
    };
    Ok(RegisterTiling::candidates(free, caps)
        .filter(|tiling| {
            packed_grouped_shape(case, ops, tiling.output_tile().0)
                .is_ok_and(|grouped| tiling.runs_over(grouped, caps))
        })
        .collect())
}

/// `packed-cells --bank B --case ID`: one `tokens<TAB>precision<TAB>index` line per tiling of the corpus, the
/// planner's own choice first marked by the line `tokens<TAB>precision<TAB>chosen`.
fn packed_cells<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank = read_bank(args)?;
    let id = flag(args, "--case")?;
    let device = device()?;
    let Target::Gpu { caps, .. } = device.target() else {
        return Err("the device is not a GPU target".into());
    };
    for tokens in PACKED_TOKENS {
        let case = packed_case(&bank, &id, tokens)?;
        for precision in ["exact", "narrow"] {
            let ops = packed_operands(precision, device.target())?;
            for index in 0..packed_candidates(&case, ops, &caps)?.len() {
                println!("{tokens}\t{precision}\t{index}");
            }
        }
        println!("{tokens}\tgemv\tchosen");
    }
    Ok(())
}

fn read_bank(args: &[String]) -> Result<Value, String> {
    let bank_path = PathBuf::from(flag(args, "--bank")?);
    serde_json::from_slice(
        &std::fs::read(&bank_path).map_err(|e| format!("read {}: {e}", bank_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", bank_path.display()))
}

/// The tiled schedules of the indexed packed contractions `program` plans, in plan order.
fn packed_schedules_of(program: &StagedProgram) -> Vec<Schedule> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) => {
                Some(spec.schedule)
            }
            _ => None,
        })
        .collect()
}

/// `packed-measure --bank B --case ID --tokens T --precision exact|narrow --tiling INDEX|chosen`: one TSV row of
/// the packed grouped dataset: the cell, the median device microseconds, the cost-model terms of the tiling that ran
/// (its rows grouped at its own tile), its modelled cost under the device's ratios and its schedule. `chosen` runs
/// the planner's own tiling (no pin) with the rows grouped.
fn packed_measure<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank = read_bank(args)?;
    let id = flag(args, "--case")?;
    let tokens: usize = flag(args, "--tokens")?
        .parse()
        .map_err(|e| format!("--tokens: {e}"))?;
    let precision = flag(args, "--precision")?;
    let device = device()?;
    let target = device.target();
    let Target::Gpu { caps, .. } = target else {
        return Err("the device is not a GPU target".into());
    };
    let case = packed_case(&bank, &id, tokens)?;
    let mut ops = packed_operands(&precision, target)?;
    let tiling = match flag(args, "--tiling")?.as_str() {
        "chosen" => None,
        index => {
            let index: usize = index.parse().map_err(|e| format!("--tiling: {e}"))?;
            Some(
                packed_candidates(&case, ops, &caps)?
                    .get(index)
                    .copied()
                    .ok_or_else(|| format!("no candidate tiling {index}"))?,
            )
        }
    };
    ops.tiling = tiling;
    let prepared = crate::runner::prepare(&bank, &case, ops)?;
    let schedules = packed_schedules_of(&staged(
        &prepared.graphs[0],
        target,
        ops,
        &facts_of(&device),
    )?);
    let measured = measure(device, &bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{id} at {tokens}: the output disagrees with the reference (worst {})",
            measured.worst
        ));
    }
    let Some(&schedule) = schedules.first() else {
        return Err(format!("{id} at {tokens}: no packed contraction planned"));
    };
    if schedules.iter().any(|ran| *ran != schedule)
        || tiling.is_some_and(|pin| pin.schedule() != schedule)
    {
        return Err(format!(
            "{id} at {tokens}: the planner ran {schedules:?}, not the pinned {tiling:?}"
        ));
    }
    let shape = &case["shape"];
    let dim = |name: &str| shape[name].as_u64().map_or(0, |v| v as usize);
    let rows = dim("tokens") * dim("top_k").max(1);
    let step_us = median(
        measured
            .device_us
            .clone()
            .ok_or("the device reports no timestamps; build it with device timing")?,
    );
    let group_us = dispatch_us(&measured, "expert_group")?.unwrap_or(0.0);
    let contraction_us = dispatch_us(&measured, "contraction:packed")?
        .ok_or_else(|| format!("{id} at {tokens}: no packed contraction dispatch"))?;
    // The terms of the schedule that ran: a Gemv's, or the tiling's with its rows grouped at its own tile, and the
    // tiles it launches and the `K` rounds each runs (a Gemv has neither).
    let (terms, tiles, rounds, tile, lanes, depth) = match schedule {
        // The planner's Gemv shape (`GemvLaunch::cost_shape`): a workgroup owns `cols` column groups of `gemv_rows`
        // rows each.
        Schedule::Gemv {
            width,
            cols,
            rows: gemv_rows,
            ..
        } => (
            GemvShape {
                outputs: rows * dim("n"),
                k: dim("k"),
                width,
                cols: cols * gemv_rows.get(),
            }
            .cost_terms(&caps),
            0,
            0,
            "gemv".to_string(),
            width as usize,
            0,
        ),
        other => {
            let ran = RegisterTiling::of(other).ok_or("the grouped rows ran no tiling")?;
            let grouped = packed_grouped_shape(&case, ops, ran.output_tile().0)?;
            let (tile_rows, tile_cols) = ran.output_tile();
            (
                ran.cost_terms(grouped, &caps),
                grouped.blocks
                    * grouped.working_row_tiles(tile_rows)
                    * grouped.out_per_block.div_ceil(tile_cols),
                grouped.k.div_ceil(ran.depth()),
                format!("{tile_rows}x{tile_cols}"),
                ran.workgroup_lanes(),
                ran.depth(),
            )
        }
    };
    let modelled = terms.modelled_time(caps.gemm_launch().calibrated.figures().cost);
    println!(
        "{tokens}\t{precision}\t{step_us}\t{group_us}\t{contraction_us}\t{}\t{}\t{}\t{}\t{}\t{}\t{modelled}\t{tile}\t{lanes}\t{depth}\t{tiles}\t{rounds}\t{}\t{schedule:?}",
        terms.fma,
        terms.lds_reads,
        terms.staged_per_lane,
        terms.direct_per_lane,
        terms.rounding,
        terms.occupancy,
        flag(args, "--tiling")?,
    );
    Ok(())
}

/// The median over the timed repeats of the device time of the dispatches whose plan entry starts with `prefix`
/// (summed when a contraction splits into several), or `None` when the plan has none.
fn dispatch_us(measured: &crate::runner::Measured, prefix: &str) -> Result<Option<f64>, String> {
    let positions: Vec<usize> = measured
        .plan
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.starts_with(prefix))
        .map(|(position, _)| position)
        .collect();
    if positions.is_empty() {
        return Ok(None);
    }
    let per_repeat = measured
        .dispatch_us
        .iter()
        .map(|repeat| {
            positions
                .iter()
                .map(|&position| {
                    repeat
                        .get(position)
                        .copied()
                        .ok_or_else(|| "the device reports no per-dispatch time".to_string())
                })
                .sum::<Result<f64, String>>()
        })
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(Some(median(per_repeat)))
}

/// The median of `samples`.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// The dense shapes of the fragment corpus (POOT-1547), as `MxKxN`: prefill chunks of the corpus's projections, and
/// the one-row decode of a vocabulary head, where the planner must keep the Gemv.
const DENSE_CELLS: [(usize, usize, usize); 6] = [
    (64, 1024, 1024),
    (512, 1024, 3072),
    (2048, 1024, 3072),
    (512, 3072, 1024),
    (1, 1024, 65_536),
    (1, 2560, 151_936),
];

/// The operands of a dense measurement at `precision` (`exact`, `narrow`, or `gemv`: the exact decode Gemv).
fn dense_operands(precision: &str, target: Target) -> Result<Operands, String> {
    let name = match precision {
        "exact" | "gemv" => "exact",
        "narrow" => "narrow",
        other => {
            return Err(format!(
                "--precision is exact, narrow or gemv, not {other:?}"
            ));
        }
    };
    Operands::of(
        target,
        &serde_json::json!({ "numerics": { "operand_precision": name } }),
    )
}

/// The dense tiled shape of `shape` under `ops`: the narrowed terms when the operands are narrow.
fn dense_shape(shape: (usize, usize, usize), ops: Operands) -> TiledShape {
    let precision = match ops.narrow {
        None => poot_graph_ir::OperandPrecision::Exact,
        Some(dtype) => poot_graph_ir::OperandPrecision::Narrow(dtype),
    };
    TiledShape {
        terms: StagedTerms::of(precision, None, None),
        ..tiled_shape(shape)
    }
}

/// `dense-cells`: one `MxKxN<TAB>precision<TAB>index` line per tiling of the dense corpus, exact and narrow, and the
/// `gemv` line of each one-row shape.
fn dense_cells<D>(device: impl FnOnce() -> Result<D, String>) -> Result<(), String>
where
    D: Device + 'static,
{
    let device = device()?;
    let Target::Gpu { caps, .. } = device.target() else {
        return Err("the device is not a GPU target".into());
    };
    for shape in DENSE_CELLS {
        let text = format!("{}x{}x{}", shape.0, shape.1, shape.2);
        for precision in ["exact", "narrow"] {
            let ops = dense_operands(precision, device.target())?;
            for index in 0..RegisterTiling::candidates(dense_shape(shape, ops), &caps).count() {
                println!("{text}\t{precision}\t{index}");
            }
        }
        if shape.0 == 1 {
            println!("{text}\tgemv\tchosen");
        }
    }
    Ok(())
}

/// `dense-measure --bank B --shape MxKxN --precision exact|narrow|gemv --tiling INDEX|chosen`: one row in the packed
/// grouped dataset's columns (`packed-measure`) for a dense contraction, its `group` column zero.
fn dense_measure<D>(
    args: &[String],
    device: impl FnOnce() -> Result<D, String>,
) -> Result<(), String>
where
    D: Device + 'static,
{
    let bank = read_bank(args)?;
    let text = flag(args, "--shape")?;
    let shape = parse_shape(&text)?;
    let precision = flag(args, "--precision")?;
    let device = device()?;
    let target = device.target();
    let Target::Gpu { caps, .. } = target else {
        return Err("the device is not a GPU target".into());
    };
    let mut ops = dense_operands(&precision, target)?;
    let tiling = match flag(args, "--tiling")?.as_str() {
        "chosen" => None,
        index => {
            let index: usize = index.parse().map_err(|e| format!("--tiling: {e}"))?;
            Some(
                RegisterTiling::candidates(dense_shape(shape, ops), &caps)
                    .nth(index)
                    .ok_or_else(|| format!("{text} has no candidate tiling {index}"))?,
            )
        }
    };
    ops.tiling = tiling;
    let template = if shape.0 == 1 {
        "dense-decode-head-bf16"
    } else {
        "prefill-dense-bf16-nk"
    };
    let mut case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == template))
        .ok_or_else(|| format!("the bank has no case {template}"))?
        .clone();
    case["shape"] = serde_json::json!({ "m": shape.0, "k": shape.1, "n": shape.2 });
    let prepared = prepare(&bank, &case, ops)?;
    let program = staged(&prepared.graphs[0], target, ops, &facts_of(&device))?;
    let tiled = dense_tiled_schedules_of(&program);
    let gemv = dense_gemv_schedules_of(&program);
    let measured = measure(device, &bank, &prepared, ops)?;
    if !measured.agrees {
        return Err(format!(
            "{text}: the output disagrees with the reference (worst {})",
            measured.worst
        ));
    }
    let step_us = median(
        measured
            .device_us
            .clone()
            .ok_or("the device reports no timestamps; build it with device timing")?,
    );
    let contraction_us = dispatch_us(&measured, "contraction:dense")?
        .ok_or_else(|| format!("{text}: no dense contraction dispatch"))?;
    let (terms, tiles, rounds, tile, lanes, depth) = match (gemv.first(), tiled.first()) {
        // The planner's Gemv shape (`GemvLaunch::cost_shape`): a workgroup owns `cols` column groups of `rows` rows each.
        (
            Some(&Schedule::Gemv {
                width, cols, rows, ..
            }),
            None,
        ) => (
            GemvShape {
                outputs: shape.0 * shape.2,
                k: shape.1,
                width,
                cols: cols * rows.get(),
            }
            .cost_terms(&caps),
            0,
            0,
            "gemv".to_string(),
            width as usize,
            0,
        ),
        (None, Some(&ran)) => {
            let tiling = RegisterTiling::of(ran).ok_or("the dense contraction ran no tiling")?;
            let dense = dense_shape(shape, ops);
            let (tile_rows, tile_cols) = tiling.output_tile();
            if tiled.iter().any(|schedule| *schedule != ran)
                || ops.tiling.is_some_and(|pin| pin.schedule() != ran)
            {
                return Err(format!(
                    "{text}: the planner ran {tiled:?}, not the pinned {:?}",
                    ops.tiling
                ));
            }
            (
                tiling.cost_terms(dense, &caps),
                dense.blocks
                    * dense.working_row_tiles(tile_rows)
                    * dense.out_per_block.div_ceil(tile_cols),
                dense.k.div_ceil(tiling.depth()),
                format!("{tile_rows}x{tile_cols}"),
                tiling.workgroup_lanes(),
                tiling.depth(),
            )
        }
        other => {
            return Err(format!(
                "{text}: the plan holds {other:?}, not one schedule"
            ));
        }
    };
    let modelled = terms.modelled_time(caps.gemm_launch().calibrated.figures().cost);
    println!(
        "{}\t{precision}\t{step_us}\t0\t{contraction_us}\t{}\t{}\t{}\t{}\t{}\t{}\t{modelled}\t{tile}\t{lanes}\t{depth}\t{tiles}\t{rounds}\t{}\t{text}",
        shape.0,
        terms.fma,
        terms.lds_reads,
        terms.staged_per_lane,
        terms.direct_per_lane,
        terms.rounding,
        terms.occupancy,
        flag(args, "--tiling")?,
    );
    Ok(())
}
