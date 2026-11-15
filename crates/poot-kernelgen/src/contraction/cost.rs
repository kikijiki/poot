//! The one cost model of a contraction schedule (POOT-1476, POOT-1479
//! sections 3.5 and 7): the modelled time of a register or fragment tiling and of the per-row Gemv, in the same unit,
//! so the planner can ask which exact or narrowed schedule is cheapest and never ranks two schedules by two models.
//!
//! The unit is the FMA issues of one lane per `K` value (`K` is common to every schedule, so it is left out), over the
//! launch's lanes against the device's saturating lanes, but never less than one lane's own sequence of steps.
//!
//! Calibration: every ratio is the device's ([`poot_target::calibration::PerfCalibration::cost`]; the tiling ratios of
//! gfx1151 through HSA were fitted by POOT-1494, the Gemv's direct value by the POOT-1489 microbench; on sm_89 the
//! direct value and the Gemv's per-dot-product cost together by POOT-1602). A Gemv pays what its dot products cost
//! whatever their `K` beside its per-`K` reads, which is what lets a short-`K` Gemv lose to a tile that its long-`K`
//! twin of the same work beats.
//!
//! One model ranks the tilings of one contraction ([`ContractionCost::tiled`]), the Gemv against a matrix unit
//! ([`ContractionCost::gemv`]) and the per-row Gemv of an indexed contraction against the tiling of its rows grouped
//! by expert ([`crate::RegisterTiling::gemv_beats_grouped`]). It counts issues only. A global-byte roofline over the
//! same terms was measured and left out (POOT-1489): over the 88 cells of the indexed microbench it classified no cell
//! better than the issues alone, and its byte ratio was not identified by them.

use poot_target::calibration::CostRatios;

use super::{RegisterTiling, TiledRows, TiledShape};
use crate::data_movement::expert_group_scan_steps;
use crate::microtile::{FRAGMENT_LANES, MicroTile};

/// The issue cost, in FMAs, of what one lane does per staged `K` value: an FMA per accumulator (the unit), an LDS
/// read per operand value of the step and a staged value (a global load, its LDS write and its share of the round's
/// two barriers). The last two are the device's measured [`CostRatios`] (`PerfCalibration::cost`).
const FMA_COST: f64 = 1.0;
/// The reads a lane issues to load one operand fragment, 16 values of 2 bytes adjacent in an `X`-major staged tile
/// (every fragment operand is staged so, [`MicroTile::staged_major`]): two 16-byte reads.
const FRAGMENT_LOAD_READS: f64 = 2.0;
/// The lane FMAs of one fragment multiply before the matrix unit's speed credit: a 16x16x16 multiply is 4096 FMAs
/// spread over 32 lanes. The unit runs [`poot_target::FragmentConfig::fma_rate`] times the FMA rate (1 where the
/// device does not report it: no credit).
const FRAGMENT_MULTIPLY_FMAS: f64 = 4096.0 / FRAGMENT_LANES as f64;

/// The terms of a tiling's modelled time on a device, before the device's [`CostRatios`] weigh them: one lane's
/// issues per staged `K` value and the occupancy factor that turns them into the launch's time. Linear in the
/// ratios, which is what lets a measured device time be regressed on them (`fit_cost_ratios`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostTerms {
    /// The multiply-accumulates one lane issues, in FMAs.
    pub fma: f64,
    /// The LDS reads one lane issues (weighed by [`CostRatios::lds_read`]).
    pub lds_reads: f64,
    /// The values one lane stages (weighed by [`CostRatios::staged_value`]).
    pub staged_per_lane: f64,
    /// The values one lane reads straight from global memory (weighed by [`CostRatios::direct_value`]): a Gemv
    /// lane's, none for a tile.
    pub direct_per_lane: f64,
    /// The extra FMA issues a staged value takes to be rounded to a narrow dtype.
    pub rounding: f64,
    /// The launch's lanes over the device's saturating lanes, never below 1.
    pub occupancy: f64,
    /// For a fragment tiling over rows grouped by expert, the `K` rounds a lane runs per `K` value (one over the
    /// round's depth), which a device that identified the fragment cost weighs by
    /// [`poot_target::calibration::CostRatios::fragment_round`]; `None` for any other tiling and for a Gemv.
    pub rounds_per_k: Option<f64>,
    /// The scan steps of the expert grouping this contraction's rows need, per `K` value (the grouping runs once, in
    /// one workgroup, so the occupancy does not scale it): none for dense rows and for a Gemv.
    pub grouping_per_k: f64,
    /// The dot products one lane finishes per `K` value (its share of its workgroup's outputs over `K`), which a
    /// device that identified it weighs by [`CostRatios::gemv_dot`]: a Gemv's, none for a tiling.
    pub dots_per_k: f64,
}

impl CostTerms {
    /// The modelled time under `ratios`, in FMA issues of one saturated lane per staged `K` value.
    pub fn modelled_time(self, ratios: CostRatios) -> f64 {
        let lane_step = match (self.rounds_per_k, ratios.fragment_round) {
            // A fragment tiling on a device that identified it: the multiplies and the rounds, and no staging term.
            (Some(rounds), Some(round)) => self.fma + rounds * round,
            _ => {
                self.fma
                    + self.lds_reads * ratios.lds_read
                    + self.staged_per_lane * (ratios.staged_value + self.rounding)
                    + self.direct_per_lane * ratios.direct_value
            }
        } + ratios.gemv_dot.map_or(0.0, |dot| self.dots_per_k * dot);
        lane_step * self.occupancy
            + ratios
                .group_scan
                .map_or(0.0, |scan| self.grouping_per_k * scan)
    }
}

/// The launch of a per-row Gemv (`Schedule::Gemv`) over `outputs` dot products of depth `k`: what its cost needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvShape {
    /// Dot products launched: the rows times the output columns.
    pub outputs: usize,
    /// The depth of each dot product.
    pub k: usize,
    /// Lanes of a workgroup, a multiple of `cols`.
    pub width: u32,
    /// Output columns a workgroup owns (its column groups times the rows of each).
    pub cols: u32,
}

impl GemvShape {
    /// The terms of this launch's modelled time on `caps`, before the device's ratios weigh them: what the
    /// calibration fit regresses a measured time on.
    #[cfg(any(test, feature = "test-support"))]
    pub fn cost_terms(self, caps: &poot_target::DeviceCaps) -> CostTerms {
        ContractionCost::gemv_terms(self, caps)
    }
}

/// The cost functions of every contraction schedule.
pub(super) struct ContractionCost;

impl ContractionCost {
    /// The modelled time of `shape` under `tiling`.
    ///
    /// One lane's work per `K` value is its [`FMA_COST`]s (a scalar tile) or its share of the fragment multiplies
    /// ([`FRAGMENT_MULTIPLY_FMAS`] per 16 values over the unit's `fma_rate`), its LDS reads and its share of the
    /// tile's staged values (both at the device's calibrated [`CostRatios`]), over the tiles that compute
    /// ([`TiledShape::working_row_tiles`]: a grouped tile is one expert's, whole tiles per bucket). The device runs
    /// `caps.compute_units` units, each saturated by `caps.gemm_launch().saturating_lanes` resident lanes: the
    /// launch's lanes (padded up to whole tiles: a ragged edge computes the full tile) share them, so the time is
    /// the lane step times the launch's lanes over the device's saturating lanes, but never less than one lane's
    /// own sequence of steps. This is what makes a short contraction take tiles small enough to fill the device,
    /// and a long one the large tiles that stage least per FMA.
    pub(super) fn tiled(
        tiling: RegisterTiling,
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
    ) -> f64 {
        let ratios = caps.gemm_launch().calibrated.figures().cost;
        Self::tiled_terms(tiling, shape, caps).modelled_time(ratios)
    }

    /// The terms [`Self::tiled`] weighs for `shape` under `tiling` on `caps`.
    pub(super) fn tiled_terms(
        tiling: RegisterTiling,
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
    ) -> CostTerms {
        let (tile_rows, tile_cols) = tiling.output_tile();
        let lane_count = tiling.workgroup_lanes() as f64;
        let staged_per_lane = (tile_rows + tile_cols) as f64 / lane_count;
        let (fma, lds_reads) = match tiling.micro {
            MicroTile::Scalar { rows, cols } => {
                let (rows, cols) = (f64::from(rows.get()), f64::from(cols.get()));
                (rows * cols * FMA_COST, rows + cols)
            }
            MicroTile::Fragment(tile) => {
                // A fragment multiply per 16 values, and a lane loads 16 values of each operand fragment in
                // `FRAGMENT_LOAD_READS` reads.
                let kappa = tiling.micro.kappa() as f64;
                let loads = f64::from(tile.rows + tile.cols) * FRAGMENT_LOAD_READS;
                let fma_rate = caps
                    .matrix_units
                    .known()
                    .and_then(|units| units.config(tile.dtype)?.fma_rate.known())
                    .map_or(1.0, |rate| f64::from(rate.max(1)));
                (
                    f64::from(tile.rows * tile.cols) * FRAGMENT_MULTIPLY_FMAS / fma_rate / kappa,
                    loads / kappa,
                )
            }
        };
        // A staged value of a narrowed operand also takes its rounding (and, into a fragment tile, the narrowing cast).
        let rounding = shape.terms.rounding_ops(caps.rounding_conversions) as f64
            + f64::from(u8::from(
                matches!(tiling.micro, MicroTile::Fragment(_)) && shape.terms.fragment.is_some(),
            ));
        let tiles = shape.blocks
            * shape.working_row_tiles(tile_rows)
            * shape.out_per_block.div_ceil(tile_cols);
        let grouping_per_k = match shape.rows {
            TiledRows::Dense => 0.0,
            TiledRows::Grouped { groups, .. } => {
                expert_group_scan_steps(groups, shape.blocks * shape.m_per_block) as f64
                    / shape.k.max(1) as f64
            }
        };
        CostTerms {
            fma,
            lds_reads,
            staged_per_lane,
            direct_per_lane: 0.0,
            rounding,
            occupancy: Self::fill(tiles as f64 * lane_count, caps),
            rounds_per_k: (matches!(tiling.micro, MicroTile::Fragment(_))
                && matches!(shape.rows, TiledRows::Grouped { .. }))
            .then(|| 1.0 / tiling.depth() as f64),
            grouping_per_k,
            dots_per_k: 0.0,
        }
    }

    /// The modelled time of a Gemv launch, in the unit of [`Self::tiled`]. A workgroup of `width` lanes shares the
    /// `K` values of `cols` output columns, so per `K` value a lane issues `1 / width` of the FMAs of its columns
    /// and reads their weight values and the activation value straight from global memory
    /// ([`CostRatios::direct_value`]), with no LDS stage or barrier. The launch is `outputs / cols` workgroups. Each
    /// dot product also costs what does not scale with `K` ([`CostRatios::gemv_dot`]): a lane's share of its
    /// workgroup's `cols` dot products, over `K`.
    pub(super) fn gemv(gemv: GemvShape, caps: &poot_target::DeviceCaps) -> f64 {
        let ratios = caps.gemm_launch().calibrated.figures().cost;
        Self::gemv_terms(gemv, caps).modelled_time(ratios)
    }

    /// The terms [`Self::gemv`] weighs for `gemv` on `caps`.
    pub(super) fn gemv_terms(gemv: GemvShape, caps: &poot_target::DeviceCaps) -> CostTerms {
        let (width, cols) = (f64::from(gemv.width.max(1)), f64::from(gemv.cols.max(1)));
        let lanes = gemv.outputs.div_ceil(gemv.cols.max(1) as usize) as f64 * width;
        CostTerms {
            fma: cols * FMA_COST / width,
            lds_reads: 0.0,
            staged_per_lane: 0.0,
            direct_per_lane: (cols + 1.0) / width,
            rounding: 0.0,
            occupancy: Self::fill(lanes, caps),
            rounds_per_k: None,
            grouping_per_k: 0.0,
            dots_per_k: cols / width / gemv.k.max(1) as f64,
        }
    }

    /// The launch's lanes over the device's saturating lanes, never below one lane's own sequence of steps. A
    /// device with no known compute-unit count is costed as one unit: no schedule is credited with filling lanes it
    /// cannot show exist, so the schedules rank by the work they issue.
    fn fill(lanes: f64, caps: &poot_target::DeviceCaps) -> f64 {
        let device_lanes = f64::from(caps.compute_units.known().map_or(1, |units| units.max(1)))
            * f64::from(caps.gemm_launch().saturating_lanes.max(1));
        (lanes / device_lanes).max(1.0)
    }
}
