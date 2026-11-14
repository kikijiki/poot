//! The packed decode Gemv's launch geometry as the argmin of a cost model (POOT-1550 D4, POOT-1562).
//!
//! A geometry is `(cols, rows, run, depth)` over the device's workgroup width. The planner enumerates every one the
//! format's decode admits and the lane's registers hold ([`poot_kernelgen::packed_gemv_body_cost`] refuses the rest by
//! the generator's own bound), scores each with the model below and takes the cheapest. Nothing is keyed on a model, a
//! shape or a format: the format enters only through its descriptor's run sharing and the cost of the body its
//! descriptor emits, and the device only through [`poot_target::calibration::GemvDevice`].
//!
//! A Gemv is a loop of trips, each a lane's run of `K` elements of each of its rows. For a grid of at most one
//! residency of the device every workgroup is resident at once, so the dispatch lasts its fixed cost plus its trips, and
//! a trip costs the larger of two things:
//!
//! - its work, the larger of the unit's issue of the warps' instructions and its L1's service of their cache lines;
//! - its latency, spread over the `depth` trips a lane keeps in flight: with one trip in flight a lane waits the DRAM
//!   latency before it can use what it loaded, so the trip lasts `latency + work`; at depth `d` those `d` trips share
//!   that wait, `(latency + work) / d`, which cannot go below the work itself.
//!
//! A grid past one residency runs in as many rounds as it takes.

use poot_kernelgen::{GemvRows, LaneMap, LoadDepth, PackedRunSharing};
use poot_quant::format::WeightFormat;
use poot_target::calibration::GemvDevice;
use poot_target::{Backend, DeviceCaps, Queried, SubgroupSupport};

use crate::predicates::GemvLaunch;
use crate::refusal::Site;
use crate::{Capability, PlanError};

/// The refusal of a launch no geometry of which fits: with no workgroup limit it is every geometry exceeding the lane's
/// registers.
pub(crate) const NO_GEOMETRY: poot_kernelgen::KernelGenError =
    poot_kernelgen::KernelGenError::Unsupported {
        request: "packed_gemv_geometry",
        reason: "no Gemv geometry fits the launch",
    };

/// The lane runs a decode Gemv offers, longest first: the runs the generator's register bound sweeps (a vector
/// activation read needs a multiple of four).
const RUNS: [u32; 4] = [32, 16, 8, 4];

/// The columns per workgroup a Gemv offers: each a power of two the workgroup width divides. A weight row is
/// contiguous along `K`, so fewer columns give each row more lanes.
const COLS: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// What a packed Gemv launch is over: the dot products it computes and the extents they share.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemvSite {
    /// The dot products launched.
    pub(crate) outputs: usize,
    /// The outputs of one block (one activation row): a lane's rows lie within one, so its rows divide it.
    pub(crate) out_per_block: usize,
    /// The depth of each dot product.
    pub(crate) k: usize,
    /// The most workgroups the launch may have, where a launch past it is refused rather than folded.
    pub(crate) workgroup_limit: Option<usize>,
}

/// The lane runs the decode of a format that shares `sharing` across a run takes: a decode that shares a run's factors
/// up to a factor run takes only the runs that divide it (a run that does not decodes every element's factors, a cliff
/// the model does not see), any other the whole set. A factor run no run divides takes one element.
fn runs_of(sharing: PackedRunSharing) -> Vec<u32> {
    let runs: Vec<u32> = RUNS
        .into_iter()
        .filter(|&run| match sharing {
            PackedRunSharing::FactorRun { factor_run } => factor_run.is_multiple_of(run as usize),
            PackedRunSharing::WholeRow | PackedRunSharing::Elementwise => true,
        })
        .collect();
    if runs.is_empty() { vec![1] } else { runs }
}

/// The lanes one warp issues together: the largest subgroup the device reports, else the 32 lanes of the narrowest.
fn warp_lanes(caps: &DeviceCaps) -> u32 {
    match caps.subgroup {
        Queried::Known(SubgroupSupport::Present { max_size, .. }) if max_size > 0 => max_size,
        _ => 32,
    }
}

/// The modelled cycles of a Gemv `geometry` over `site` with trips costing `body` on `device`, `units` compute units.
fn modelled_cycles(
    geometry: &Geometry,
    site: GemvSite,
    body: &poot_kernelgen::GemvBodyCost,
    (device, units, warp): (GemvDevice, u32, u32),
) -> f64 {
    let lanes_per_col = geometry.width / geometry.cols;
    let trips = site
        .k
        .div_ceil(lanes_per_col as usize * geometry.run as usize) as f64;
    let workgroups = site
        .outputs
        .div_ceil((geometry.cols * geometry.rows) as usize);
    let launched = workgroups as f64 * f64::from(geometry.width);
    let capacity = f64::from(units) * f64::from(device.resident_lanes);
    let rounds = (launched / capacity).ceil().max(1.0);
    let warps_per_unit = launched / rounds / f64::from(units) / f64::from(warp);
    let lines = body.lines_per_trip(LaneMap {
        warp_lanes: warp,
        lanes_per_col,
    });
    let issue = warps_per_unit * f64::from(body.warp_instructions) / device.issue_per_cycle;
    let service = warps_per_unit * lines / device.l1_lines_per_cycle;
    let work = issue.max(service);
    let trip = work.max((device.dram_latency_cycles + work) / f64::from(geometry.depth));
    device.dispatch_cycles + rounds * trips * trip
}

/// One geometry the model scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    width: u32,
    cols: u32,
    rows: u32,
    run: u32,
    depth: u32,
}

/// The launch of the cheapest geometry the model scores for a `format` Gemv over `site`, or `None` when no geometry
/// fits the workgroup limit. Ties go to the geometry that holds fewer registers (fewer rows, then a shallower
/// pipeline), so a geometry is deepened only for a modelled gain.
///
/// A candidate the generator refuses for registers is not a candidate; any other refusal (a descriptor the decode
/// cannot state) is the site's.
pub(crate) fn packed_gemv_geometry(
    site: Site<'_>,
    shape: GemvSite,
    format: WeightFormat,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Option<GemvLaunch>, PlanError> {
    let launch = caps.gemm_launch();
    let device = launch.calibrated.figures().packed_gemv;
    let units = caps.compute_units.known().map_or(1, |units| units.max(1));
    let warp = warp_lanes(caps);
    let width = launch.gemv_width;
    let sharing = poot_kernelgen::packed_run_sharing(format)
        .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
    let mut best: Option<(f64, Geometry)> = None;
    let mut refusal = None;
    for rows in 1..=GemvRows::MAX {
        if !shape.out_per_block.is_multiple_of(rows as usize) {
            continue;
        }
        for depth in 1..=LoadDepth::MAX {
            for run in runs_of(sharing) {
                for cols in COLS.into_iter().filter(|&cols| width.is_multiple_of(cols)) {
                    let geometry = Geometry {
                        width,
                        cols,
                        rows,
                        run,
                        depth,
                    };
                    let fits = shape.workgroup_limit.is_none_or(|limit| {
                        shape.outputs.div_ceil((cols * rows) as usize) <= limit
                    });
                    if !fits {
                        continue;
                    }
                    let candidate = launch_of(geometry);
                    let body = match poot_kernelgen::packed_gemv_body_cost(
                        format,
                        candidate.schedule(backend, shape.k),
                        caps,
                    ) {
                        Ok(body) => body,
                        Err(error) => {
                            refusal = Some(error);
                            continue;
                        }
                    };
                    let cycles = modelled_cycles(&geometry, shape, &body, (device, units, warp));
                    if best.is_none_or(|(least, _)| cycles < least * (1.0 - 1e-9)) {
                        best = Some((cycles, geometry));
                    }
                }
            }
        }
    }
    match (best, refusal) {
        (Some((_, geometry)), _) => Ok(Some(launch_of(geometry))),
        (None, Some(error)) => Err(site.refuse(Capability::KernelGen(error))),
        (None, None) => Ok(None),
    }
}

/// The launch of `geometry`.
fn launch_of(geometry: Geometry) -> GemvLaunch {
    GemvLaunch {
        width: geometry.width,
        cols: geometry.cols,
        unroll: geometry.run,
        rows: GemvRows::new(geometry.rows).expect("a geometry's rows are within the generator's"),
        depth: LoadDepth::new(geometry.depth)
            .expect("a geometry's depth is within the generator's"),
    }
}
