//! The contraction lowering (Card 727): one loop structure ([`Schedule`]) per
//! contraction shape, and one body per schedule that serves every operand load. A load supplies only the
//! read of a run of weight values: the dense loader (an f32, f16-packed or bf16-packed
//! `[N, K]` weight) here, the descriptor-driven block decode in `crate::packed`. The planner picks the schedule and
//! its launch from shapes and the device ([`poot_target::DeviceCaps::compute_units`]); the generator only
//! implements it.

mod dense;
mod epilogue;
mod gemv;
mod indexed;
mod tiled;

pub use dense::DenseTiledWeight;
pub(crate) use dense::{dense_contraction, dense_tiled_contraction};
pub(crate) use epilogue::{BoundEpilogue, emit_epilogue};
pub use epilogue::{Epilogue, EpilogueInput, EpilogueLeaf, EpilogueStep};
pub(crate) use gemv::{GemvLaunch, gemv_body};
pub(crate) use indexed::{
    META_EXPERTS, META_N, META_PACKED_SLAB_WORDS, RowExpert, emit_poison, emit_row_expert,
    indexed_contraction,
};

pub(crate) use tiled::{ExpertWalk, TiledBody, TiledLaunch, TiledWeight, tiled_body};

use crate::KernelGenError;
use crate::helpers::WeightLayout;
use crate::microtile::{LaneRows, MicroCols, MicroTile};

/// How a [`Schedule::Gemv`] lane reads the `unroll` activation values of a full run. Both read the same values
/// and feed them to the same products in the same order, so the output is bit-identical; they differ only in the
/// loads the target issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemvActivation {
    /// One scalar load per value.
    Scalar,
    /// One 4-lane [`poot_kernel_ir::Rvalue::VectorLoad`] per four values of a full run; the element-by-element tail
    /// past the last full run stays scalar, so the activation param is read both ways. The planner picks it only for
    /// a target whose codegen lowers that (`poot_codegen::Capability::mixed_vector_reads`) and only when every full
    /// run starts on a 4-element boundary of a 16-byte-aligned buffer: `K % 4 == 0` and `unroll % 4 == 0`, with the
    /// executor's binding alignment (`poot_executor`'s binding contract). [`Schedule::Gemv`] refuses it for an
    /// `unroll` that is not a multiple of four.
    Vec4,
}

impl GemvActivation {
    /// The values one [`GemvActivation::Vec4`] group read carries: four f32, the 16 bytes of the executors' binding
    /// alignment.
    pub const VEC4_LANES: usize = 4;
}

/// The contraction's loop structure, chosen by the planner from shapes, never from the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    /// `M == 1`: one workgroup of `width` lanes owns `cols` adjacent output columns (weight rows),
    /// `width / cols` lanes per column; each lane reads `unroll` contiguous `K` elements per trip,
    /// and one LDS tree folds each column's lanes. `cols` is the launch-grid knob the
    /// planner picks per shape; `width` must be a multiple of `cols`.
    Gemv {
        /// Lanes in the workgroup; a multiple of `cols`.
        width: u32,
        /// Adjacent output columns the workgroup owns.
        cols: u32,
        /// Contiguous `K` elements each lane reads per trip.
        unroll: u32,
        /// How a full run reads its activation values.
        activation: GemvActivation,
    },
    /// `M > 1`, every contraction (block-diagonal, or indexed with each row reading its own expert's weight):
    /// LDS-staged tiles with a register tile per lane. A workgroup
    /// of `lanes x lanes` lanes computes the [`Schedule::output_tile`] its lanes interleave, each accumulating a
    /// `rows x cols` part of it, so one staged weight value feeds `rows` FMAs and one activation value `cols`. `K`
    /// is staged in runs of `run` consecutive values, which the weight's decode shares an address chain, word load
    /// and scale across. [`RegisterTiling::choose`] picks the geometry from the shapes and the device.
    RegisterTiled {
        /// The edge of the square lane grid one workgroup runs.
        lanes: TileSize,
        /// Output rows each lane accumulates.
        rows: LaneRows,
        /// Output columns each lane accumulates.
        cols: MicroCols,
        /// Consecutive `K` values one lane loads and decodes at a time; divides `K`.
        run: RunLength,
    },
}

/// The run of consecutive `K` values a [`Schedule::RegisterTiled`] lane stages at a time, built only by
/// [`RunLength::dividing`]: the longest run that divides `K`, so every run of a row is whole.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunLength(u32);

impl RunLength {
    /// The longest run (at most four values: one word of byte codes) that divides `k`.
    pub fn dividing(k: usize) -> Self {
        Self(match k {
            k if k.is_multiple_of(4) => 4,
            k if k.is_multiple_of(2) => 2,
            _ => 1,
        })
    }

    /// The run's length, at least 1.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The bare length, so a schedule's `Debug` (part of the planner's kernel key) reads `run: 4`.
impl std::fmt::Debug for RunLength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The lane-grid edge of a [`Schedule::RegisterTiled`] contraction, built only by [`TileSize::new`]. Card 658: a
/// row-tile is laid out per block ([`Schedule::grid_threads`], `tiled_body`'s grid derivation), so it never
/// straddles two blocks' weights and the schedule serves any `blocks >= 1`: `m_per_block` need not be a multiple
/// of the tile.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileSize(pub(crate) u32);

impl TileSize {
    /// A `tile x tile` schedule: refused unless `tile >= 1`.
    pub fn new(tile: u32) -> Result<Self, KernelGenError> {
        if tile == 0 {
            return Err(KernelGenError::BelowMinimum {
                generator: "contraction_tiled",
                what: "tile".to_string(),
                value: 0,
                min: 1,
            });
        }
        Ok(Self(tile))
    }

    /// The tile edge, at least 1.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Schedule {
    /// The launch extent, in total threads (executors divide by the body's `workgroup_size`), of a
    /// contraction over `blocks` block-diagonal blocks of `m_per_block` activation rows each and
    /// `out_per_block` weight rows per block: `width` lanes per `cols` outputs for `Gemv`, `lanes^2` lanes per
    /// output tile for `RegisterTiled`, laid out per block (`blocks * ceil(m_per_block / tile_rows) *
    /// ceil(out_per_block / tile_cols)` tiles), card 658, so a row-tile never spans two blocks. The one owner of each
    /// body's grid convention.
    pub fn grid_threads(self, blocks: usize, m_per_block: usize, out_per_block: usize) -> usize {
        let rows = blocks * m_per_block;
        let outputs = rows * out_per_block;
        match self {
            Schedule::Gemv { width, cols, .. } => outputs.div_ceil(cols as usize) * width as usize,
            Schedule::RegisterTiled {
                lanes, rows, cols, ..
            } => {
                let (tile_rows, tile_cols) = RegisterTiling::output_tile_of(lanes, rows, cols);
                let lanes = lanes.get() as usize;
                blocks
                    * m_per_block.div_ceil(tile_rows)
                    * out_per_block.div_ceil(tile_cols)
                    * lanes
                    * lanes
            }
        }
    }

    /// The `(rows, columns)` of the output tile one workgroup of a tiled schedule computes, or `None` for the Gemv.
    pub fn output_tile(self) -> Option<(usize, usize)> {
        match self {
            Schedule::Gemv { .. } => None,
            Schedule::RegisterTiled {
                lanes, rows, cols, ..
            } => Some(RegisterTiling::output_tile_of(lanes, rows, cols)),
        }
    }
}

/// The register tiles a [`Schedule::RegisterTiled`] lane may accumulate: rows and columns per lane.
const LANE_ROWS: [u32; 3] = [2, 4, 8];
const LANE_COLS: [u32; 2] = [4, 8];

/// The most accumulators one lane holds (an 8x8 tile): with the `rows + cols` operand values of a step and the
/// staging addresses, 149 VGPRs on gfx1151, within the 256 a wave32 lane holds, so several waves stay resident.
const MAX_LANE_ACCUMULATORS: u32 = 64;

/// The issue cost, in FMAs, of what one lane does per staged `K` value: an FMA per accumulator, an LDS read per
/// operand value of the step, and a staged value (a global load, its LDS write and its share of the round's two
/// barriers). Calibrated on gfx1151 (ROCm): over every tiling of a qwen3-0.6b prefill at 512 and 2048 tokens, the
/// contraction device time follows these ratios within a few percent (POOT-1403 receipt).
const FMA_COST: f64 = 1.0;
const LDS_READ_COST: f64 = 1.0;
const STAGED_VALUE_COST: f64 = 8.0;

/// The shape of an `M > 1` contraction: `blocks` independent products of `m_per_block` activation rows against
/// `out_per_block` weight rows of `k` values (one block is a plain GEMM), the weight read in `layout`, the rows of a
/// block in `segments` contiguous runs that each read their own weight (an indexed contraction's expert-sorted
/// rows: `segments` is the expert count; 1 for a block-diagonal one). What [`RegisterTiling::choose`] sizes a tiling
/// for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiledShape {
    /// Independent products, one weight block (or slab) each.
    pub blocks: usize,
    /// Activation rows per block.
    pub m_per_block: usize,
    /// Output columns (weight rows) per block.
    pub out_per_block: usize,
    /// The contraction depth.
    pub k: usize,
    /// How the weight is stored, which decides how its tile is staged.
    pub layout: WeightLayout,
    /// Contiguous row runs of a block that read different weights: an expert walk's keys (1 for none).
    pub segments: usize,
}

impl TiledShape {
    /// The expected `K` passes of one tile of `tile_rows` rows: a tile spans its first segment plus every segment
    /// boundary inside it, and `segments` runs over `m_per_block` sorted rows put a boundary after about every
    /// `m_per_block / segments` rows, so a tile takes `1 + (tile_rows - 1) * (segments - 1) / m_per_block` passes,
    /// never more than its rows or the segments. 1 for a block-diagonal contraction. An expert walk's steps for
    /// expert-sorted rows; the model of their cost, from the shape alone.
    pub fn passes_per_tile(self, tile_rows: usize) -> f64 {
        let segments = self.segments.max(1);
        if segments == 1 {
            return 1.0;
        }
        let boundaries =
            (tile_rows.saturating_sub(1) * (segments - 1)) as f64 / self.m_per_block.max(1) as f64;
        (1.0 + boundaries).min(tile_rows.min(segments) as f64)
    }
}

/// The geometry of a [`Schedule::RegisterTiled`] contraction, chosen from the shapes and the device by
/// [`RegisterTiling::choose`]. It is the schedule's fields before they are wrapped, so a planner reads the lane
/// grid and output tile it launches without destructuring the [`Schedule`] again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterTiling {
    lanes: TileSize,
    rows: LaneRows,
    cols: MicroCols,
    run: RunLength,
}

impl RegisterTiling {
    /// The tiling of `shape` on `caps`, from an occupancy cost model over every geometry the device can run: the
    /// narrow and wide lane grids of `caps.gemm_launch()`, each with every register tile of [`LANE_ROWS`] x
    /// [`LANE_COLS`] up to [`MAX_LANE_ACCUMULATORS`], whose workgroup fits the device, whose two staged tiles fit
    /// its LDS and stage in whole slots ([`tiled::stages_evenly`]). Each is costed by [`Self::cost`] and the
    /// cheapest wins, a tie going to the larger output tile. Chosen from shapes and device limits alone, never
    /// from the format or a model; the run is the longest that divides `k`.
    pub fn choose(shape: TiledShape, caps: &poot_target::DeviceCaps) -> Self {
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(shape.k);
        let mut best: Option<(f64, Self)> = None;
        for tiling in Self::candidates(shape.k, shape.layout, caps) {
            let cost = tiling.cost(shape, caps);
            let better = best.is_none_or(|(best_cost, best_tiling)| {
                cost < best_cost
                    || (cost == best_cost
                        && tiling.outputs_per_tile() > best_tiling.outputs_per_tile())
            });
            if better {
                best = Some((cost, tiling));
            }
        }
        // The narrow grid's smallest tile stages evenly at any run and fits every device that runs a 64-lane
        // workgroup; it is the tiling of a device the filter leaves nothing for.
        best.map_or(
            Self {
                lanes: TileSize(launch.register_lane_edge),
                rows: LaneRows(LANE_ROWS[0]),
                cols: MicroCols::new(LANE_COLS[0]),
                run,
            },
            |(_, tiling)| tiling,
        )
    }

    /// Every tiling [`Self::choose`] weighs for a weight of `k`-value rows stored in `layout` on `caps`.
    pub(crate) fn candidates(
        k: usize,
        layout: WeightLayout,
        caps: &poot_target::DeviceCaps,
    ) -> impl Iterator<Item = Self> + '_ {
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(k);
        [launch.register_lane_edge, launch.register_lane_edge_wide]
            .into_iter()
            .flat_map(move |lanes| {
                LANE_ROWS.into_iter().flat_map(move |rows| {
                    LANE_COLS.into_iter().map(move |cols| Self {
                        lanes: TileSize(lanes),
                        rows: LaneRows(rows),
                        cols: MicroCols::new(cols),
                        run,
                    })
                })
            })
            .filter(move |tiling| tiling.runs_on(layout, caps))
    }

    /// Whether `caps` runs this tiling over a weight stored in `layout`: its workgroup fits, its two staged tiles
    /// fit the LDS and stage in whole slots, and its register tile is at most [`MAX_LANE_ACCUMULATORS`].
    fn runs_on(self, layout: WeightLayout, caps: &poot_target::DeviceCaps) -> bool {
        let lanes = self.lanes();
        let lane_count = lanes * lanes;
        let micro = self.micro_tile();
        self.rows.get() * self.cols.get() <= MAX_LANE_ACCUMULATORS
            && lane_count <= caps.max_workgroup_invocations as usize
            && lane_count <= caps.max_workgroup_size[0] as usize
            && tiled::lds_bytes(lanes, micro) <= caps.lds_bytes as usize
            && tiled::stages_evenly(lanes, micro, self.run.get() as usize, layout)
    }

    /// The modelled time of `shape` under this tiling, in FMA issues of one lane per staged `K` value (`K` is
    /// common to every tiling, so it is left out).
    ///
    /// One lane's work per `K` value is its [`FMA_COST`]s, [`LDS_READ_COST`]s and its share of the tile's
    /// staged values ([`STAGED_VALUE_COST`]), once per `K` pass of the tile ([`TiledShape::passes_per_tile`]: more
    /// than one for a tile whose rows read several experts). The device runs `caps.compute_units` units, each saturated by
    /// `caps.gemm_launch().saturating_lanes` resident lanes: the launch's lanes (padded up to whole tiles: a
    /// ragged edge computes the full tile) share them, so the time is the lane step times the launch's lanes over
    /// the device's saturating lanes, but never less than one lane's own sequence of steps. This is what makes a
    /// short contraction take tiles small enough to fill the device, and a long one the large tiles that stage
    /// least per FMA.
    fn cost(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> f64 {
        let (tile_rows, tile_cols) = self.output_tile();
        let lane_count = (self.lanes() * self.lanes()) as f64;
        let (rows, cols) = (f64::from(self.rows.get()), f64::from(self.cols.get()));
        let staged_per_lane = (tile_rows + tile_cols) as f64 / lane_count;
        let lane_step = (rows * cols * FMA_COST
            + (rows + cols) * LDS_READ_COST
            + staged_per_lane * STAGED_VALUE_COST)
            * shape.passes_per_tile(tile_rows);
        let tiles = shape.blocks
            * shape.m_per_block.div_ceil(tile_rows)
            * shape.out_per_block.div_ceil(tile_cols);
        let device_lanes = f64::from(caps.compute_units.max(1))
            * f64::from(caps.gemm_launch().saturating_lanes.max(1));
        lane_step * (tiles as f64 * lane_count / device_lanes).max(1.0)
    }

    /// The schedule that runs this tiling.
    pub fn schedule(self) -> Schedule {
        Schedule::RegisterTiled {
            lanes: self.lanes,
            rows: self.rows,
            cols: self.cols,
            run: self.run,
        }
    }

    /// The edge of the square lane grid one workgroup runs.
    pub fn lanes(self) -> usize {
        self.lanes.get() as usize
    }

    /// The `(rows, columns)` of the output tile one workgroup computes.
    pub fn output_tile(self) -> (usize, usize) {
        Self::output_tile_of(self.lanes, self.rows, self.cols)
    }

    fn outputs_per_tile(self) -> usize {
        let (rows, cols) = self.output_tile();
        rows * cols
    }

    fn micro_tile(self) -> MicroTile {
        Self::micro_tile_of(self.rows, self.cols)
    }

    fn output_tile_of(lanes: TileSize, rows: LaneRows, cols: MicroCols) -> (usize, usize) {
        let lanes = lanes.get() as usize;
        Self::micro_tile_of(rows, cols).output_tile(lanes, lanes)
    }

    /// The micro-tile each lane of a tiling with `rows x cols` per lane accumulates.
    pub(super) const fn micro_tile_of(rows: LaneRows, cols: MicroCols) -> MicroTile {
        MicroTile::Scalar { rows, cols }
    }
}

/// The bare edge, so a schedule's `Debug` (part of the planner's kernel key) reads `Tiled { tile: 8 }`.
impl std::fmt::Debug for TileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: usize, k: usize, n: usize) -> TiledShape {
        TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            segments: 1,
        }
    }

    /// A long prefill projection (`[2048, 1024] x [3072, 1024]^T`) fills the device many times over, so the
    /// tiling that stages least per FMA wins: the wide lane grid of 8x8 register tiles, a 128 x 128 output tile,
    /// on the ROCm, PTX and wgpu caps. Mutation: cost the lane step alone (drop the launch-lanes factor in
    /// `RegisterTiling::cost`); the smallest register tile wins and this row goes red.
    #[test]
    fn a_long_prefill_takes_the_widest_tile() {
        for caps in [
            poot_target::DeviceCaps::rocm_default(),
            poot_target::DeviceCaps::ptx_default(),
            poot_target::DeviceCaps::wgpu_rdna3_igpu(),
        ] {
            let tiling = RegisterTiling::choose(shape(2048, 1024, 3072), &caps);
            assert_eq!(
                (tiling.lanes(), tiling.output_tile()),
                (16, (128, 128)),
                "{tiling:?}"
            );
        }
    }

    /// Expert-sorted rows put a segment boundary after about every `m / segments` rows, so a tile's expected `K`
    /// passes grow with its height: one for a block-diagonal contraction at any height, `1 + 127 * 63 / 4096` for a
    /// 128-row tile of 4096 rows over 64 experts, and never more than the tile's rows or the segments.
    #[test]
    fn a_tiles_passes_count_the_segment_boundaries_it_spans() {
        let walk = |segments| TiledShape {
            segments,
            ..shape(4096, 2048, 1024)
        };
        assert_eq!(walk(1).passes_per_tile(128), 1.0);
        assert_eq!(walk(64).passes_per_tile(1), 1.0);
        assert!((walk(64).passes_per_tile(128) - (1.0 + 127.0 * 63.0 / 4096.0)).abs() < 1e-12);
        let dense_segments = TiledShape {
            segments: 4096,
            ..shape(64, 2048, 1024)
        };
        assert_eq!(dense_segments.passes_per_tile(16), 16.0);
    }

    /// An expert walk pays a `K` pass per expert a tile's rows span, so with few rows per expert it takes shorter
    /// tiles than the block-diagonal contraction of the same shape: 512 rows over 128 experts (four rows each) on the
    /// ROCm caps. Mutation: drop the `passes_per_tile` factor in `RegisterTiling::cost`; the walk takes the dense
    /// shape's tile and this row goes red.
    #[test]
    fn an_expert_walk_takes_shorter_tiles_than_the_same_dense_shape() {
        let caps = poot_target::DeviceCaps::rocm_default();
        let dense = RegisterTiling::choose(shape(512, 2048, 1024), &caps);
        let walk = RegisterTiling::choose(
            TiledShape {
                segments: 128,
                ..shape(512, 2048, 1024)
            },
            &caps,
        );
        assert!(
            walk.output_tile().0 < dense.output_tile().0,
            "walk {walk:?} vs dense {dense:?}"
        );
    }

    /// A contraction too small to fill the device costs one lane's own steps whatever its tile, so the cheapest
    /// lane step wins: sixteen rows of 1024 columns take the narrow grid's smallest tile (16 x 32), not the
    /// 128 x 128 tile whose lanes would each do four times the work. Mutation: drop the `max(1.0)` floor in
    /// `RegisterTiling::cost`; every tiling then scales with its padded lanes, the 16 x 64 tile wins and this
    /// row goes red.
    #[test]
    fn a_contraction_below_the_device_takes_the_cheapest_lane_step() {
        let tiling = RegisterTiling::choose(
            shape(16, 1024, 1024),
            &poot_target::DeviceCaps::rocm_default(),
        );
        assert_eq!(
            (tiling.lanes(), tiling.output_tile()),
            (8, (16, 32)),
            "{tiling:?}"
        );
    }

    /// The candidates are the tilings the device runs: a device whose workgroup holds 64 invocations is offered
    /// only the narrow (8x8) lane grid, and one whose LDS holds 8 KiB none whose two staged tiles need more.
    /// Mutation: drop the `lds_bytes` clause of `RegisterTiling::runs_on`; the 64 x 64 tiles (8.3 KiB) are
    /// offered on the small-LDS device and this row goes red.
    #[test]
    fn the_candidates_are_the_tilings_the_device_runs() {
        let small_workgroup = poot_target::DeviceCaps {
            max_workgroup_invocations: 64,
            ..poot_target::DeviceCaps::rocm_default()
        };
        assert!(
            RegisterTiling::candidates(1024, WeightLayout::Nk, &small_workgroup)
                .all(|tiling| tiling.lanes() == 8)
        );
        let small_lds = poot_target::DeviceCaps {
            lds_bytes: 8 * 1024,
            ..poot_target::DeviceCaps::rocm_default()
        };
        let offered: Vec<RegisterTiling> =
            RegisterTiling::candidates(1024, WeightLayout::Nk, &small_lds).collect();
        assert!(!offered.is_empty());
        for tiling in offered {
            assert!(
                tiled::lds_bytes(tiling.lanes(), tiling.micro_tile()) <= 8 * 1024,
                "{tiling:?}"
            );
        }
    }
}
