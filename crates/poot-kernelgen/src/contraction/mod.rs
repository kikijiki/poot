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

use poot_graph_ir::FragmentDtype;
use poot_target::FragmentLayout;

use crate::KernelGenError;
use crate::helpers::WeightLayout;
use crate::microtile::{FRAGMENT_LANES, FragmentTile, LaneRows, MicroCols, MicroTile};
use crate::precision::StagedTerms;

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
    /// `M > 1` on matrix units (POOT-1436): the [`Schedule::RegisterTiled`] loop nest with a [`MicroTile::Fragment`].
    /// A workgroup of `waves_m x waves_n` waves of 32 lanes computes the [`Schedule::output_tile`] its waves
    /// interleave, each wave accumulating `tile.rows x tile.cols` fragments of 16x16 outputs from narrow operand
    /// tiles in workgroup memory. The request's operand precision must narrow or split to `tile.dtype`
    /// ([`TiledLaunch::of`]). `run` is as for `RegisterTiled`. [`RegisterTiling::choose`] picks it where the
    /// device has a matrix unit for the operands.
    FragmentTiled {
        /// Waves along the output rows.
        waves_m: TileSize,
        /// Waves along the output columns.
        waves_n: TileSize,
        /// What each wave accumulates, and the fragment dtype and layout.
        tile: FragmentTile,
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
            Schedule::RegisterTiled { .. } | Schedule::FragmentTiled { .. } => {
                let tiling = RegisterTiling::of(self).expect("a tiled schedule has a tiling");
                let (tile_rows, tile_cols) = tiling.output_tile();
                blocks
                    * m_per_block.div_ceil(tile_rows)
                    * out_per_block.div_ceil(tile_cols)
                    * tiling.workgroup_lanes()
            }
        }
    }

    /// The `(rows, columns)` of the output tile one workgroup of a tiled schedule computes, or `None` for the Gemv.
    pub fn output_tile(self) -> Option<(usize, usize)> {
        RegisterTiling::of(self).map(RegisterTiling::output_tile)
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
/// The vector reads a lane issues to load one operand fragment: its 16 values of 2 bytes, as two 16-byte reads.
const FRAGMENT_LOAD_READS: f64 = 2.0;

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
    /// The f32 tiles staged per operand: more than one for a split operand, which stages its `hi` and `lo` parts.
    pub terms: StagedTerms,
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

/// The geometry of a [`Schedule::RegisterTiled`] or [`Schedule::FragmentTiled`] contraction, chosen from the shapes
/// and the device by [`RegisterTiling::choose`]. It is the schedule's fields before they are wrapped, so a planner
/// reads the workgroup and output tile it launches without destructuring the [`Schedule`] again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterTiling {
    /// Owners along the output rows and columns: lanes for a scalar micro-tile, waves for a fragment one.
    owners_m: TileSize,
    owners_n: TileSize,
    micro: MicroTile,
    run: RunLength,
}

/// The wave grids a fragment tiling may run, `(waves along rows, waves along columns)`: one to eight waves, with the
/// long side along the rows (an activation tile is staged once for all the wave columns that read it).
const WAVE_GRIDS: [(u32, u32); 6] = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4)];
/// The fragments a wave may accumulate along each axis.
const WAVE_FRAGMENTS: [u32; 3] = [1, 2, 4];
/// The most accumulator fragments one wave holds (8 f32 registers each): with the operand fragments of a step
/// (8 registers each) and the staging addresses, within the 256 registers a wave32 lane holds.
const MAX_WAVE_FRAGMENTS: u32 = 8;
/// The issue cost, in lane FMAs, of one fragment multiply per lane: a 16x16x16 multiply is 4096 FMAs spread over 32
/// lanes, and the matrix unit runs 4x the FMA rate (the measured gfx1151 peaks, 40-45 against 10.8 TFLOP/s:
/// `receipts/1429-narrow-operands/design.md` section 1.1).
const WMMA_COST: f64 = 4096.0 / FRAGMENT_LANES as f64 / 4.0;

impl RegisterTiling {
    /// The tiling of `shape` on `caps`, from an occupancy cost model over every geometry the device can run.
    ///
    /// - Scalar: the narrow and wide lane grids of `caps.gemm_launch()`, each with every register tile of
    ///   [`LANE_ROWS`] x [`LANE_COLS`] up to [`MAX_LANE_ACCUMULATORS`].
    /// - Fragment, where the operands narrow to a dtype the device has a matrix unit for
    ///   ([`StagedTerms::fragment`]): every wave grid of [`WAVE_GRIDS`] with every [`WAVE_FRAGMENTS`] tile of at
    ///   most [`MAX_WAVE_FRAGMENTS`] fragments. An expert walk asks for scalar tiles only
    ///   ([`StagedTerms::scalar_only`]): the rows of a fragment must share one weight tile.
    ///
    /// A candidate must have a workgroup that fits the device, two staged tiles that fit its LDS and stage in whole
    /// slots ([`tiled::stages_evenly`]). Each is costed by [`Self::cost`] and the cheapest wins, a tie going to the
    /// larger output tile. Chosen from shapes and device limits alone, never from the format or a model; the run is
    /// the longest that divides `k`.
    pub fn choose(shape: TiledShape, caps: &poot_target::DeviceCaps) -> Self {
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(shape.k);
        let mut best: Option<(f64, Self)> = None;
        for tiling in Self::candidates(shape, caps) {
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
            Self::scalar(
                TileSize(launch.register_lane_edge),
                LaneRows(LANE_ROWS[0]),
                MicroCols::new(LANE_COLS[0]),
                run,
            ),
            |(_, tiling)| tiling,
        )
    }

    fn scalar(lanes: TileSize, rows: LaneRows, cols: MicroCols, run: RunLength) -> Self {
        Self {
            owners_m: lanes,
            owners_n: lanes,
            micro: MicroTile::Scalar { rows, cols },
            run,
        }
    }

    /// The tiling a tiled schedule runs, or `None` for a Gemv.
    pub fn of(schedule: Schedule) -> Option<Self> {
        match schedule {
            Schedule::Gemv { .. } => None,
            Schedule::RegisterTiled {
                lanes,
                rows,
                cols,
                run,
            } => Some(Self::scalar(lanes, rows, cols, run)),
            Schedule::FragmentTiled {
                waves_m,
                waves_n,
                tile,
                run,
            } => Some(Self {
                owners_m: waves_m,
                owners_n: waves_n,
                micro: MicroTile::Fragment(tile),
                run,
            }),
        }
    }

    /// The fragment config `caps` offers for operands of `dtype`: a 16x16x16 multiply held across the 32 lanes of a
    /// wave, the one shape and collective every fragment micro-tile lowers. Its layout, and the most waves of one
    /// workgroup.
    fn fragment_config(
        dtype: FragmentDtype,
        caps: &poot_target::DeviceCaps,
    ) -> Option<(FragmentLayout, u32)> {
        let units = caps.matrix_units.known()?;
        let config = units.config(dtype)?;
        (config.shape == poot_target::FragmentShape::M16N16K16
            && config.subgroup as usize == FRAGMENT_LANES)
            .then_some((config.layout, config.waves))
    }

    /// Every tiling [`Self::choose`] weighs for `shape` on `caps`.
    pub(crate) fn candidates(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
    ) -> impl Iterator<Item = Self> + '_ {
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(shape.k);
        let scalar = [launch.register_lane_edge, launch.register_lane_edge_wide]
            .into_iter()
            .flat_map(move |lanes| {
                LANE_ROWS.into_iter().flat_map(move |rows| {
                    LANE_COLS.into_iter().map(move |cols| {
                        Self::scalar(TileSize(lanes), LaneRows(rows), MicroCols::new(cols), run)
                    })
                })
            });
        let fragment = shape
            .terms
            .fragment
            .and_then(|dtype| Some((dtype, Self::fragment_config(dtype, caps)?)))
            .into_iter()
            .flat_map(move |(dtype, (layout, waves))| {
                WAVE_GRIDS.into_iter().flat_map(move |(waves_m, waves_n)| {
                    WAVE_FRAGMENTS.into_iter().flat_map(move |rows| {
                        WAVE_FRAGMENTS.into_iter().filter_map(move |cols| {
                            (rows * cols <= MAX_WAVE_FRAGMENTS && waves_m * waves_n <= waves)
                                .then_some(Self {
                                    owners_m: TileSize(waves_m),
                                    owners_n: TileSize(waves_n),
                                    micro: MicroTile::Fragment(FragmentTile {
                                        dtype,
                                        layout,
                                        rows,
                                        cols,
                                    }),
                                    run,
                                })
                        })
                    })
                })
            });
        scalar
            .chain(fragment)
            .filter(move |tiling| tiling.runs_on(shape, caps))
    }

    /// Whether `caps` runs this tiling over `shape`: its workgroup fits, its staged tiles (and the scratch an
    /// `Opaque` fragment spills to) fit the LDS and stage in whole slots, and a scalar register tile is at most
    /// [`MAX_LANE_ACCUMULATORS`].
    fn runs_on(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> bool {
        let lane_count = self.workgroup_lanes();
        let accumulators = match self.micro {
            MicroTile::Scalar { rows, cols } => rows.get() * cols.get(),
            MicroTile::Fragment(_) => 0,
        };
        accumulators <= MAX_LANE_ACCUMULATORS
            && lane_count <= caps.max_workgroup_invocations as usize
            && lane_count <= caps.max_workgroup_size[0] as usize
            && tiled::lds_bytes(self.launch_shape(), shape.layout, shape.terms)
                <= caps.lds_bytes as usize
            && tiled::stages_evenly(self.launch_shape(), self.run.get() as usize, shape.layout)
    }

    fn run(self) -> RunLength {
        self.run
    }

    fn launch_shape(self) -> tiled::LaunchShape {
        tiled::LaunchShape {
            owners: (self.owners_m.get() as usize, self.owners_n.get() as usize),
            micro: self.micro,
        }
    }

    /// The modelled time of `shape` under this tiling, in FMA issues of one lane per staged `K` value (`K` is
    /// common to every tiling, so it is left out).
    ///
    /// One lane's work per `K` value is its [`FMA_COST`]s (a scalar tile) or its share of the fragment multiplies
    /// ([`WMMA_COST`] per 16 values), its [`LDS_READ_COST`]s and its share of the tile's staged values
    /// ([`STAGED_VALUE_COST`]), once per `K` pass of the tile ([`TiledShape::passes_per_tile`]: more than one for a
    /// tile whose rows read several experts). The device runs `caps.compute_units` units, each saturated by
    /// `caps.gemm_launch().saturating_lanes` resident lanes: the launch's lanes (padded up to whole tiles: a
    /// ragged edge computes the full tile) share them, so the time is the lane step times the launch's lanes over
    /// the device's saturating lanes, but never less than one lane's own sequence of steps. This is what makes a
    /// short contraction take tiles small enough to fill the device, and a long one the large tiles that stage
    /// least per FMA.
    fn cost(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> f64 {
        let (tile_rows, tile_cols) = self.output_tile();
        let lane_count = self.workgroup_lanes() as f64;
        let staged_per_lane = (tile_rows + tile_cols) as f64 / lane_count;
        let (compute, reads) = match self.micro {
            MicroTile::Scalar { rows, cols } => {
                let (rows, cols) = (f64::from(rows.get()), f64::from(cols.get()));
                (rows * cols * FMA_COST, (rows + cols) * LDS_READ_COST)
            }
            MicroTile::Fragment(tile) => {
                // A fragment multiply per 16 values, and a lane loads 16 values of each operand fragment: two
                // vector reads per fragment.
                let kappa = self.micro.kappa() as f64;
                let loads = f64::from(tile.rows + tile.cols) * FRAGMENT_LOAD_READS;
                (
                    f64::from(tile.rows * tile.cols) * WMMA_COST / kappa,
                    loads * LDS_READ_COST / kappa,
                )
            }
        };
        let lane_step = (compute + reads + staged_per_lane * STAGED_VALUE_COST)
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
        match self.micro {
            MicroTile::Scalar { rows, cols } => Schedule::RegisterTiled {
                lanes: self.owners_m,
                rows,
                cols,
                run: self.run,
            },
            MicroTile::Fragment(tile) => Schedule::FragmentTiled {
                waves_m: self.owners_m,
                waves_n: self.owners_n,
                tile,
                run: self.run,
            },
        }
    }

    /// The lanes of the workgroup that computes one output tile.
    pub fn workgroup_lanes(self) -> usize {
        self.owners_m.get() as usize * self.owners_n.get() as usize * self.micro.owner_lanes()
    }

    /// The `(rows, columns)` of the output tile one workgroup computes.
    pub fn output_tile(self) -> (usize, usize) {
        self.micro
            .output_tile(self.owners_m.get() as usize, self.owners_n.get() as usize)
    }

    fn outputs_per_tile(self) -> usize {
        let (rows, cols) = self.output_tile();
        rows * cols
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
            terms: crate::precision::StagedTerms::ONE,
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
                (tiling.workgroup_lanes(), tiling.output_tile()),
                (256, (128, 128)),
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
            (tiling.workgroup_lanes(), tiling.output_tile()),
            (64, (16, 32)),
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
            RegisterTiling::candidates(shape(512, 1024, 1024), &small_workgroup)
                .all(|tiling| tiling.workgroup_lanes() == 64)
        );
        let small_lds = poot_target::DeviceCaps {
            lds_bytes: 8 * 1024,
            ..poot_target::DeviceCaps::rocm_default()
        };
        let offered: Vec<RegisterTiling> =
            RegisterTiling::candidates(shape(512, 1024, 1024), &small_lds).collect();
        assert!(!offered.is_empty());
        for tiling in offered {
            assert!(
                tiled::lds_bytes(tiling.launch_shape(), WeightLayout::Nk, StagedTerms::ONE)
                    <= 8 * 1024,
                "{tiling:?}"
            );
        }
    }
}
