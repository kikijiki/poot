//! `calibrate-launch`: fit the register-tiled contraction cost model's ratios to this device's measured time
//! (design POOT-1479 section 3.4, the POOT-1403 method), and write a [`PerfCalibration`] cost entry for review.
//!
//! Three modes, so no device is opened twice in one process (an HSA executor cannot be reopened after a drop):
//! - `shapes`: open the device and print the corpus, one `MxKxN<TAB>index` line per tiling the device runs for the
//!   shape (`RegisterTiling::candidates`, in order);
//! - `measure --bank B --shape MxKxN --tiling INDEX`: open the device, pin that candidate tiling
//!   (`CompileOptions::tiling_pin`), time the plan of that dense contraction on the device clock and print one TSV
//!   row: the shape, the median device microseconds, and the cost-model terms of the pinned tiling;
//! - `fit --rows F --out E`: regress the rows' times (per `K` value) on their terms and write the entry (the ratios, the fit's
//!   quality and the rows it rests on) to `E`. It opens no device.
//!
//! `scripts/calibrate-launch <backend>` runs them in order under the GPU lock. The corpus is a grid of shapes,
//! never a model's: one run calibrates every shape, and no shape-specific figure enters the table.
//!
//! The fit regresses over every tiling the device runs for every shape, each pinned so the planner runs exactly
//! that one. The tilings differ in tile geometry whatever the shape, so the corpus determines the ratios; the tool
//! refuses a corpus that does not.

use std::path::PathBuf;

use poot_executor::Device;
use poot_graph_plan::KernelChoice;
use poot_graph_plan::staged::StagedProgram;
use poot_kernelgen::{ContractionSpec, KernelRequest};

use crate::runner::{Operands, measure, prepare, staged};
use poot_graph_plan::Target;
use poot_kernelgen::{
    CostTerms, RegisterTiling, StagedTerms, TiledRows, TiledShape, fit_cost_ratios,
};
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
        weight_bits: 16,
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

/// The schedules of the register-tiled dense contractions `program` plans, in plan order, as their `Debug` text.
fn tiled_schedules_of(program: &StagedProgram) -> Vec<String> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| plan.dispatches().iter())
        .filter_map(|d| match d.choice() {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(
                spec,
            ))) => Some(format!("{:?}", spec.schedule)),
            _ => None,
        })
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
    };
    let prepared = prepare(bank, &case, ops)?;
    let schedules = tiled_schedules_of(&staged(&prepared.graphs[0], device.target(), ops)?);
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
                    rounding: number(5)?,
                    occupancy: number(6)?,
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
                 // The current Strix Halo entry is lds_read 1.0, staged_value 8.0.\n\
                 cost: CostRatios {{\n    lds_read: {:.3},\n    staged_value: {:.3},\n}},\n\
                 // Rows (shape, median device time, pinned schedule):\n{}\n",
                samples.len(),
                fit.r_squared,
                fit.seconds_per_fma,
                fit.ratios.lds_read,
                fit.ratios.staged_value,
                table.join("\n"),
            );
            std::fs::write(&out, &entry).map_err(|e| format!("write {}: {e}", out.display()))?;
            print!("{entry}");
            Ok(())
        }
        other => Err(format!(
            "usage: calibrate_launch shapes|measure|fit, got {other:?}"
        )),
    }
}
