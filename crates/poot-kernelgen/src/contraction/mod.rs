//! The contraction lowering (Card 727): one loop structure ([`Schedule`]) per
//! contraction shape, and one body per schedule that serves every operand load. A load supplies only the
//! read of a run of weight values: the dense loader (an f32, f16-packed or bf16-packed
//! `[N, K]` weight) here, the descriptor-driven block decode in `crate::packed`. The planner picks the schedule and
//! its launch from shapes and the device ([`poot_target::DeviceCaps::compute_units`]); the generator only
//! implements it.

mod cost;
mod dense;
mod epilogue;
#[cfg(any(test, feature = "test-support"))]
mod fit;
mod gemv;
mod indexed;
mod tiled;

use cost::ContractionCost;
pub use cost::CostTerms;
pub use cost::GemvShape;
pub use dense::DenseTiledWeight;
pub(crate) use dense::{TiledRead, dense_contraction, dense_tiled_contraction};
pub(crate) use epilogue::{BoundEpilogue, emit_epilogue};
pub use epilogue::{Epilogue, EpilogueInput, EpilogueLeaf, EpilogueStep};
#[cfg(any(test, feature = "test-support"))]
pub use fit::{CostFit, fit_cost_ratios};
pub(crate) use gemv::{GemvLaunch, gemv_body};
pub(crate) use indexed::{
    META_EXPERTS, META_N, META_PACKED_SLAB_WORDS, RowExpert, emit_poison, emit_row_expert,
    indexed_contraction,
};

pub(crate) use tiled::{
    GroupedRows, TiledBody, TiledExtents, TiledLaunch, TiledWeight, tiled_body,
};

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
        /// The `K` values one barrier round stages: a whole number of fragment steps, sized so a round reads whole
        /// global requests of every operand row ([`RegisterTiling::choose`]).
        depth: u32,
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

/// The bytes one global memory request reads: the cache line of every target poot generates for (the L2 and L0/L1
/// lines of AMD gfx9-12, NVIDIA's lines). A round whose segment of an operand row is shorter pays a whole request
/// for part of a line, and the vector memory path is bound by requests, not bytes (POOT-1445:
/// halving the bytes of every activation request left the cycles where they were; halving the requests per multiply
/// took a third of them).
const GLOBAL_REQUEST_BYTES: usize = 128;

/// The round depths a fragment tiling may stage, shallowest first: one and two pairs of fragment steps. The
/// pipelined loop holds one round's loaded values in registers while it multiplies the previous one, so a deeper
/// round holds more: at 64 the 128 x 128 tile of a narrowed bf16 projection is 225 VGPRs on gfx1151, near the 256 a
/// wave32 lane holds (`register_budget` guards the planner's tilings against spills).
const FRAGMENT_DEPTHS: [usize; 2] = [32, 64];

/// How the activation rows of a [`TiledShape`]'s blocks meet the tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiledRows {
    /// A block's rows are contiguous activation rows, every tile reading its block's weight.
    Dense,
    /// One block of rows grouped by expert in tiles of `block` positions (POOT-1451,
    /// [`poot_graph_ir::ExpertGrouping`]): `groups` buckets (the experts and the invalid one), each tile one bucket's.
    /// Only a tiling whose output tile is `block` rows runs it.
    Grouped {
        /// The buckets, `experts + 1`.
        groups: usize,
        /// The positions of one tile.
        block: usize,
    },
}

/// The shape of an `M > 1` contraction: `blocks` independent products of `m_per_block` activation rows against
/// `out_per_block` weight rows of `k` values (one block is a plain GEMM), the weight read in `layout`, and how the
/// rows meet the tiles (`rows`: contiguous, or an indexed contraction's rows grouped by expert). What
/// [`RegisterTiling::choose`] sizes a tiling for.
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
    /// How the rows meet the tiles.
    pub rows: TiledRows,
    /// The f32 tiles staged per operand: more than one for a split operand, which stages its `hi` and `lo` parts.
    pub terms: StagedTerms,
    /// The bits one weight element occupies in global memory, rounded up: 32 for f32, 16 for a packed half, a
    /// packed format's bits per value ([`crate::packed_value_bits`]). What decides how deep a fragment round must
    /// be to read whole requests of a weight row.
    pub weight_bits: u32,
    /// The bits one activation element occupies in global memory: 32 for f32, 16 for an activation the planner
    /// stored in its fragment dtype (POOT-1448). The activation's side of the same round depth rule.
    pub activation_bits: u32,
}

impl TiledShape {
    /// Whether a tiling whose output tile is `tile_rows` rows runs these rows: any for dense rows, exactly `block`
    /// for grouped ones (a tile of the grouping is one output tile).
    pub fn admits_tile_rows(self, tile_rows: usize) -> bool {
        match self.rows {
            TiledRows::Dense => true,
            TiledRows::Grouped { block, .. } => tile_rows == block,
        }
    }

    /// The rows of one block's tile grid: its activation rows, or the grouping's padded positions (the bound the
    /// launch covers; a tile past the padded total returns at once).
    pub fn grid_rows(self) -> usize {
        match self.rows {
            TiledRows::Dense => self.m_per_block,
            TiledRows::Grouped { groups, block } => poot_graph_ir::ExpertGrouping {
                rows: self.m_per_block,
                experts: groups.saturating_sub(1),
                block,
            }
            .positions(),
        }
    }

    /// The row tiles of one block that compute, at `tile_rows` rows a tile: the rows' tiles, or for grouped rows
    /// spread evenly over `min(groups, rows)` non-empty buckets, each bucket's whole tiles. The model of the cost,
    /// from the shape alone.
    pub fn working_row_tiles(self, tile_rows: usize) -> usize {
        match self.rows {
            TiledRows::Dense => self.m_per_block.div_ceil(tile_rows),
            TiledRows::Grouped { groups, .. } => {
                let buckets = groups.min(self.m_per_block).max(1);
                buckets * self.m_per_block.div_ceil(buckets).div_ceil(tile_rows)
            }
        }
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
    /// The `K` values one barrier round stages ([`Self::depth`]).
    depth: usize,
}

/// The wave grids a fragment tiling may run, `(waves along rows, waves along columns)`: one to eight waves. Two
/// grids of one output tile load alike (every fragment operand is staged `X`-major), so they cost the same and the
/// earlier grid here wins the tie.
const WAVE_GRIDS: [(u32, u32); 6] = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4)];
/// The fragments a wave may accumulate along each axis.
const WAVE_FRAGMENTS: [u32; 3] = [1, 2, 4];
/// The f32 registers one lane of a wave holds for an accumulator fragment: the 16x16 values of its tile over the
/// 32 lanes of the wave.
const FRAGMENT_ACCUMULATOR_REGISTERS: u32 = 16 * 16 / FRAGMENT_LANES as u32;
impl RegisterTiling {
    /// The tiling of `shape` on `caps`, from an occupancy cost model over every geometry the device can run.
    ///
    /// - Scalar: the narrow and wide lane grids of `caps.gemm_launch()`, each with every register tile of
    ///   [`LANE_ROWS`] x [`LANE_COLS`].
    /// - Fragment, where the operands narrow to a dtype the device has a matrix unit for
    ///   ([`StagedTerms::fragment`]): every wave grid of [`WAVE_GRIDS`] with every [`WAVE_FRAGMENTS`] tile. Its
    ///   round depth is the
    ///   shallowest of [`FRAGMENT_DEPTHS`] at which one round's segment of every operand row is a whole global
    ///   request ([`GLOBAL_REQUEST_BYTES`]; the deepest when none is), or the deepest shallower one its LDS holds
    ///   ([`Self::fragment_depth`]).
    ///
    /// A candidate must have a workgroup that fits the device, accumulators within the lane's register budget
    /// ([`poot_target::RegisterBudget`]), two staged tiles that fit its LDS and stage in whole
    /// slots ([`tiled::stages_evenly`]), and an output tile the shape's rows admit (a grouping's block,
    /// [`TiledShape::admits_tile_rows`]). Each is costed by [`ContractionCost::tiled`] and the cheapest wins, a tie going to the
    /// larger output tile and then to the earlier candidate (the scalar grids, then [`WAVE_GRIDS`] in order: two wave
    /// grids of one output tile load alike and tie). Chosen from shapes, storage widths and device limits alone, never
    /// from the format or a model; the run is the longest that divides `k`.
    pub fn choose(shape: TiledShape, caps: &poot_target::DeviceCaps) -> Self {
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(shape.k);
        // The narrow grid's smallest tile stages evenly at any run and fits every device that runs a 64-lane
        // workgroup; it is the tiling of a device the filter leaves nothing for (a grouped shape's planner then
        // refuses it unless its rows are the block).
        Self::cheapest(shape, caps).map_or(
            Self::scalar(
                TileSize(launch.register_lane_edge),
                LaneRows(LANE_ROWS[0]),
                MicroCols::new(LANE_COLS[0]),
                run,
            ),
            |(_, tiling)| tiling,
        )
    }

    /// The cheapest candidate of `shape` on `caps` and its modelled cost, as [`Self::choose`] ranks them; `None`
    /// when the device runs none.
    fn cheapest(shape: TiledShape, caps: &poot_target::DeviceCaps) -> Option<(f64, Self)> {
        let mut best: Option<(f64, Self)> = None;
        for tiling in Self::candidates(shape, caps) {
            let cost = ContractionCost::tiled(tiling, shape, caps);
            let better = best.is_none_or(|(best_cost, best_tiling)| {
                cost < best_cost
                    || (cost == best_cost
                        && tiling.outputs_per_tile() > best_tiling.outputs_per_tile())
            });
            if better {
                best = Some((cost, tiling));
            }
        }
        best
    }

    /// The block of one expert grouping several contractions read (POOT-1451): the output-tile rows for which the
    /// sum of the contractions' cheapest tilings with exactly that many rows ([`Self::choose`]'s cost model over
    /// each `shape` grouped at that block) is least, a tie going to the larger block (fewer tiles). Only rows every
    /// contraction has a tiling for are weighed; `None` when there are none. Each shape's `rows` gives the buckets
    /// (`TiledRows::Grouped { groups, .. }`; a dense shape counts one bucket); its block is replaced.
    pub fn grouped_block(shapes: &[TiledShape], caps: &poot_target::DeviceCaps) -> Option<usize> {
        let grouped = |shape: TiledShape, block: usize| TiledShape {
            rows: TiledRows::Grouped {
                groups: match shape.rows {
                    TiledRows::Grouped { groups, .. } => groups,
                    TiledRows::Dense => 1,
                },
                block,
            },
            ..shape
        };
        let mut blocks: Vec<usize> = shapes
            .iter()
            .flat_map(|&shape| {
                let free = TiledShape {
                    rows: TiledRows::Dense,
                    ..shape
                };
                Self::candidates(free, caps).map(|tiling| tiling.output_tile().0)
            })
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        let mut best: Option<(f64, usize)> = None;
        for block in blocks {
            let total = shapes.iter().try_fold(0.0, |total, &shape| {
                Self::cheapest(grouped(shape, block), caps).map(|(cost, _)| total + cost)
            });
            if let Some(total) = total
                && best.is_none_or(|(best_total, _)| total <= best_total)
            {
                best = Some((total, block));
            }
        }
        best.map(|(_, block)| block)
    }

    fn scalar(lanes: TileSize, rows: LaneRows, cols: MicroCols, run: RunLength) -> Self {
        let micro = MicroTile::Scalar { rows, cols };
        Self {
            owners_m: lanes,
            owners_n: lanes,
            micro,
            run,
            depth: micro.block_k(),
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
                depth,
            } => Some(Self {
                owners_m: waves_m,
                owners_n: waves_n,
                micro: MicroTile::Fragment(tile),
                run,
                depth: depth as usize,
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
                            (waves_m * waves_n <= waves)
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
                                    depth: FRAGMENT_DEPTHS[0],
                                })
                                .and_then(|tiling| tiling.fragment_depth(shape, caps))
                        })
                    })
                })
            });
        // The units follow the precision (POOT-1476): an exact shape runs on lanes, a narrowed or split one on a
        // matrix unit. A narrowed op never has a scalar tiling, which would round for no speed.
        let narrowed = shape.terms.fragment.is_some();
        scalar
            .filter(move |_| !narrowed)
            .chain(fragment)
            .filter(move |tiling| {
                tiling.runs_on(shape, caps) && shape.admits_tile_rows(tiling.output_tile().0)
            })
    }

    /// The cheapest fragment tiling of a narrowed or split `shape`, or `None` when the device has none for it
    /// (POOT-1476). A narrowed contraction plans only on a matrix unit, so this is the planner's hold: it never
    /// compares with an exact schedule, which is the pass's question ([`Self::matrix_unit_wins`]).
    pub fn fragment_tiling(shape: TiledShape, caps: &poot_target::DeviceCaps) -> Option<Self> {
        shape.terms.fragment?;
        Self::cheapest(shape, caps).map(|(_, tiling)| tiling)
    }

    /// Whether a matrix unit is the cheapest schedule of a narrowed or split `shape` (POOT-1476,
    /// POOT-1471): the cheapest fragment tiling, rounding
    /// terms included, costs less than the cheapest exact alternative, a scalar tiling of the same shape read exactly
    /// or the per-row Gemv `gemv` where the planner has one. All are ranked by the one [`ContractionCost`] model
    /// over the ungrouped rows, so the answer is a property of the dimensions and the device, and a grouping of the
    /// rows later (`grouped_block`) cannot reverse it.
    pub fn matrix_unit_wins(
        shape: TiledShape,
        gemv: Option<GemvShape>,
        caps: &poot_target::DeviceCaps,
    ) -> bool {
        if shape.terms.fragment.is_none() {
            return false;
        }
        let dense = TiledShape {
            rows: TiledRows::Dense,
            ..shape
        };
        let exact = TiledShape {
            terms: StagedTerms::ONE,
            ..dense
        };
        let Some((fragment_cost, _)) = Self::cheapest(dense, caps) else {
            return false;
        };
        [
            Self::cheapest(exact, caps).map(|(cost, _)| cost),
            gemv.map(|gemv| ContractionCost::gemv(gemv, caps)),
        ]
        .into_iter()
        .flatten()
        .all(|exact| fragment_cost < exact)
    }

    /// The `K` values one barrier round of this tiling stages.
    pub fn depth(self) -> usize {
        self.depth
    }

    /// This fragment geometry at its round depth, or `None` when the device runs it at none: the shallowest of
    /// [`FRAGMENT_DEPTHS`] whose round segment of every operand row (the activation's rows at
    /// `shape.activation_bits` per value, the weight's at `shape.weight_bits`) is at least one
    /// [`GLOBAL_REQUEST_BYTES`] request, the deepest when none is; and if [`Self::runs_on`] refuses that depth (its
    /// staged tiles outgrow the LDS), the deepest shallower one it runs. A 32-deep round of a bf16 row reads half a
    /// line per request: the vector memory path pays a request for it, so the round doubles; an f32 activation row
    /// is a whole line at 32 already.
    fn fragment_depth(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> Option<Self> {
        let narrowest = shape.activation_bits.min(shape.weight_bits).max(1) as usize;
        let whole = FRAGMENT_DEPTHS
            .iter()
            .position(|&depth| depth * narrowest >= GLOBAL_REQUEST_BYTES * 8)
            .unwrap_or(FRAGMENT_DEPTHS.len() - 1);
        FRAGMENT_DEPTHS[..=whole]
            .iter()
            .rev()
            .map(|&depth| Self { depth, ..self })
            .find(|tiling| tiling.runs_on(shape, caps))
    }

    /// The f32 registers each lane of this tiling holds across the `K` loop: a scalar tile's accumulators, a
    /// fragment tile's accumulator fragments.
    pub(crate) fn accumulator_registers(self) -> u32 {
        match self.micro {
            MicroTile::Scalar { rows, cols } => rows.get() * cols.get(),
            MicroTile::Fragment(tile) => tile.rows * tile.cols * FRAGMENT_ACCUMULATOR_REGISTERS,
        }
    }

    /// Whether `caps` runs this tiling over `shape`: its workgroup fits, a lane's accumulators fit its register
    /// budget (`caps.gemm_launch().register_budget`, a quarter of what a lane of this workgroup keeps resident),
    /// its staged tiles (and the scratch an `Opaque` fragment spills to) fit the LDS and stage in whole slots.
    fn runs_on(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> bool {
        let lane_count = self.workgroup_lanes();
        let budget = caps.gemm_launch().register_budget;
        u32::try_from(lane_count)
            .is_ok_and(|lanes| self.accumulator_registers() <= budget.accumulators(lanes))
            && lane_count <= caps.max_workgroup_invocations as usize
            && lane_count <= caps.max_workgroup_size[0] as usize
            && tiled::lds_bytes(self.launch_shape(), shape.terms) <= caps.lds_bytes as usize
            && tiled::stages_evenly(
                self.launch_shape(),
                self.run.get() as usize,
                shape.layout,
                shape.out_per_block,
            )
    }

    fn run(self) -> RunLength {
        self.run
    }

    fn launch_shape(self) -> tiled::LaunchShape {
        tiled::LaunchShape {
            owners: (self.owners_m.get() as usize, self.owners_n.get() as usize),
            micro: self.micro,
            depth: self.depth,
        }
    }

    /// The terms of this tiling's modelled time for `shape` on `caps`, before the device's
    /// [`poot_target::calibration::CostRatios`] weigh them: what the calibration fit regresses a measured time on.
    #[cfg(any(test, feature = "test-support"))]
    pub fn cost_terms(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> CostTerms {
        ContractionCost::tiled_terms(self, shape, caps)
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
                depth: self.depth as u32,
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
    use poot_target::FragmentConfig;

    fn shape(m: usize, k: usize, n: usize) -> TiledShape {
        TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            rows: crate::TiledRows::Dense,
            terms: crate::precision::StagedTerms::ONE,
            weight_bits: 32,
            activation_bits: 32,
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
            poot_test_util::device_caps::wgpu_rdna3_igpu(),
        ] {
            let tiling = RegisterTiling::choose(shape(2048, 1024, 3072), &caps);
            assert_eq!(
                (tiling.workgroup_lanes(), tiling.output_tile()),
                (256, (128, 128)),
                "{tiling:?}"
            );
        }
    }

    /// Grouped rows compute whole tiles per bucket and launch the grouping's padded positions: 4096 rows over 64
    /// experts (65 buckets) at a 128-row block are 65 working tiles (63 rows a bucket, one tile each), against 32 for
    /// the same rows dense, and a grid of `(4096 + 65 * 127) / 128 * 128` positions. Mutation: count a grouped
    /// shape's working tiles as the dense rows' in `TiledShape::working_row_tiles`; this row goes red.
    #[test]
    fn grouped_rows_count_whole_tiles_per_bucket() {
        let grouped = TiledShape {
            rows: TiledRows::Grouped {
                groups: 65,
                block: 128,
            },
            ..shape(4096, 2048, 1024)
        };
        assert_eq!(grouped.working_row_tiles(128), 65);
        assert_eq!(shape(4096, 2048, 1024).working_row_tiles(128), 32);
        assert_eq!(grouped.grid_rows(), (4096 + 65 * 127) / 128 * 128);
        assert!(grouped.admits_tile_rows(128) && !grouped.admits_tile_rows(64));
    }

    /// A grouping's block is the tile rows its contractions run cheapest together, and every tiling of a grouped
    /// shape has exactly those rows: with four rows per expert (512 rows over 128 experts) the block is shorter than
    /// the dense shape's tile, since a taller tile computes mostly padding. Mutation: as above (count a grouped
    /// shape's working tiles as the dense rows'); the block becomes the dense tile and this row goes red.
    #[test]
    fn a_grouping_with_few_rows_per_expert_takes_a_short_block() {
        let caps = poot_target::DeviceCaps::rocm_default();
        let dense = RegisterTiling::choose(shape(512, 2048, 1024), &caps);
        let shapes = [shape(512, 2048, 1024), shape(512, 1024, 2048)].map(|shape| TiledShape {
            rows: TiledRows::Grouped {
                groups: 129,
                block: 0,
            },
            ..shape
        });
        let block = RegisterTiling::grouped_block(&shapes, &caps).expect("a block");
        assert!(
            block < dense.output_tile().0,
            "block {block} vs dense {dense:?}"
        );
        for shape in shapes {
            let tiling = RegisterTiling::choose(
                TiledShape {
                    rows: TiledRows::Grouped { groups: 129, block },
                    ..shape
                },
                &caps,
            );
            assert_eq!(tiling.output_tile().0, block, "{tiling:?}");
        }
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

    /// A narrowed contraction on a device whose fragment workgroup is one wave (wgpu: the runtime cannot make the
    /// pipeline's subgroups 32 lanes wide) does not win on a matrix unit, and the same device with several waves per
    /// workgroup does: a one-wave tile is small, so it stages many values per multiply, and each is rounded to the
    /// narrow dtype. The planner's hold still has a fragment tiling for the one-wave device (a declared narrowing runs
    /// there), and it is not a scalar one. Mutation: drop the `narrowing` term from `ContractionCost::tiled`; the
    /// one-wave device wins on a matrix unit and this row goes red.
    #[test]
    fn a_fragment_tile_of_one_wave_loses_to_the_exact_tiling() {
        use poot_target::{CooperativeMatrixConfig, FragmentShape, MatrixUnits, Queried};
        let device = |waves| poot_target::DeviceCaps {
            matrix_units: Queried::Known(MatrixUnits::cooperative_matrix(
                [CooperativeMatrixConfig {
                    shape: FragmentShape::M16N16K16,
                    ab_f16: true,
                    result_f32: true,
                }],
                32,
                waves,
                Queried::Known(poot_target::calibration::GFX11_MATRIX_FMA_RATE),
            )),
            ..poot_test_util::device_caps::wgpu_rdna3_igpu()
        };
        let narrow = TiledShape {
            terms: crate::precision::StagedTerms {
                fragment: Some(FragmentDtype::F16),
                ..crate::precision::StagedTerms::ONE
            },
            ..shape(2048, 1024, 3072)
        };
        assert!(!RegisterTiling::matrix_unit_wins(narrow, None, &device(1)));
        let held =
            RegisterTiling::fragment_tiling(narrow, &device(1)).expect("a declared narrowing runs");
        assert!(matches!(held.micro, MicroTile::Fragment(_)), "{held:?}");
        let many_waves = FragmentConfig::ANY_WAVES;
        let tiling =
            RegisterTiling::fragment_tiling(narrow, &device(many_waves)).expect("fragments");
        assert!(
            matches!(tiling.micro, MicroTile::Fragment(_))
                && tiling.workgroup_lanes() > FRAGMENT_LANES,
            "{tiling:?}"
        );
    }

    /// The matrix unit is the cheapest schedule of a prefill contraction on gfx1151, and the per-row Gemv is the
    /// cheapest of a decode row: the Gemv's cost (`ContractionCost::gemv`, the same unit as the tilings') beats the
    /// fragment tiling of one row, and loses to it once the launch covers enough rows to fill the device. A narrowed
    /// shape has no scalar candidate. Mutation: drop the Gemv alternative from `matrix_unit_wins`; the one-row
    /// contraction wins on a matrix unit and this row goes red.
    #[test]
    fn the_gemv_beats_a_matrix_unit_at_one_row_and_loses_at_many() {
        let caps = poot_target::DeviceCaps {
            matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::amd("gfx1151", 32)),
            compute_units: poot_target::Queried::Known(40),
            ..poot_target::DeviceCaps::rocm_default()
        };
        let narrow = |m| TiledShape {
            terms: crate::precision::StagedTerms {
                fragment: Some(FragmentDtype::F16),
                ..crate::precision::StagedTerms::ONE
            },
            weight_bits: 16,
            ..shape(m, 2048, 2048)
        };
        let gemv = |rows: usize| GemvShape {
            outputs: rows * 2048,
            width: 256,
            cols: 4,
        };
        assert!(!RegisterTiling::matrix_unit_wins(
            narrow(1),
            Some(gemv(1)),
            &caps
        ));
        assert!(
            RegisterTiling::matrix_unit_wins(narrow(1), None, &caps),
            "the Gemv term is what keeps one row exact"
        );
        assert!(RegisterTiling::matrix_unit_wins(
            narrow(2048),
            Some(gemv(2048)),
            &caps
        ));
        assert!(
            RegisterTiling::candidates(narrow(512), &caps)
                .all(|tiling| matches!(tiling.micro, MicroTile::Fragment(_))),
            "a narrowed shape tiles on fragments alone"
        );
    }

    /// The matrix unit's rate is the device's, read from its fragment config: the same fragment tiling on the same
    /// shape issues a quarter of the multiply FMAs at rate 4 that it does at rate 1, and an unknown rate gives the
    /// unit no credit (rate 1). The scalar tilings' terms do not move with it.
    /// Mutation: read a constant 4 instead of `fma_rate` in `cost_terms`; the rate-1 and rate-16 rows go red.
    #[test]
    fn the_matrix_units_rate_discounts_the_fragment_multiply() {
        use poot_target::{FragmentLayout, LaneMap, MatrixUnits, Queried};
        let caps_with = |fma_rate| poot_target::DeviceCaps {
            matrix_units: Queried::Known(MatrixUnits::of([FragmentConfig::m16n16k16(
                FragmentDtype::Bf16,
                FragmentLayout::Declared(LaneMap::Rdna3),
                32,
                FragmentConfig::ANY_WAVES,
                fma_rate,
            )])),
            ..poot_target::DeviceCaps::rocm_default()
        };
        let narrow = TiledShape {
            terms: crate::precision::StagedTerms {
                fragment: Some(FragmentDtype::Bf16),
                ..crate::precision::StagedTerms::ONE
            },
            ..shape(2048, 1024, 3072)
        };
        let fragment_fma = |rate, caps: &poot_target::DeviceCaps| {
            let tiling = RegisterTiling::candidates(narrow, caps)
                .find(|t| matches!(t.micro, MicroTile::Fragment(_)))
                .expect("a fragment candidate");
            (tiling, tiling.cost_terms(narrow, caps).fma, rate)
        };
        let (tiling, unknown, _) =
            fragment_fma(Queried::<u8>::Unknown, &caps_with(Queried::Unknown));
        let (_, one, _) = fragment_fma(Queried::Known(1), &caps_with(Queried::Known(1)));
        let (same, four, _) = fragment_fma(Queried::Known(4), &caps_with(Queried::Known(4)));
        let (_, sixteen, _) = fragment_fma(Queried::Known(16), &caps_with(Queried::Known(16)));
        assert_eq!(tiling, same, "the same candidate is costed at every rate");
        assert_eq!(unknown, one, "an unknown rate is no credit");
        assert_eq!(four * 4.0, one);
        assert_eq!(sixteen * 16.0, one);
        // A narrowed read takes fragment tilings only, so the scalar tilings are the exact shape's.
        let exact = shape(2048, 1024, 3072);
        let scalar = |caps: &poot_target::DeviceCaps| {
            RegisterTiling::candidates(exact, caps)
                .find(|t| matches!(t.micro, MicroTile::Scalar { .. }))
                .expect("a scalar candidate")
                .cost_terms(exact, caps)
        };
        assert_eq!(
            scalar(&caps_with(Queried::Known(1))),
            scalar(&caps_with(Queried::Known(16)))
        );
    }

    /// A fragment round stages whole global requests of every operand row: over a bf16 `[N, K]` weight (64 bytes a
    /// row at 32 values) the round is 64 deep, over an f32 `[K, N]` weight (128 bytes at 32) it stays 32; and a
    /// device whose LDS cannot hold the 64-deep tiles takes the deepest that fits. The planner's long-prefill tile is
    /// the same 128 x 128 of 4 x 2 waves for both layouts: every fragment operand is staged `X`-major, so the two
    /// grids of that tile load alike and the first wins. An activation the planner stored bf16 (POOT-1448) is a
    /// 64-byte row at 32 values too, so over an f32 `[N, K]` weight its round is 64 deep. Mutation: size the depth from
    /// the activation alone (drop `shape.weight_bits` from `fragment_depth`'s `narrowest`); the bf16 weight row takes
    /// 32 and goes red. Mutation: size it from the weight alone (drop `shape.activation_bits`); the bf16 activation
    /// row takes 32 and goes red.
    #[test]
    fn a_fragment_round_stages_whole_requests_of_every_operand_row() {
        let caps = poot_target::DeviceCaps {
            matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::amd("gfx1151", 32)),
            ..poot_target::DeviceCaps::rocm_default()
        };
        let narrow = |stored| {
            crate::precision::StagedTerms::of(
                poot_graph_ir::OperandPrecision::Narrow(FragmentDtype::Bf16),
                None,
                stored,
            )
        };
        let planned =
            |layout, (weight_bits, activation_bits), stored, caps: &poot_target::DeviceCaps| {
                RegisterTiling::choose(
                    TiledShape {
                        layout,
                        terms: narrow(stored),
                        weight_bits,
                        activation_bits,
                        ..shape(2048, 1024, 3072)
                    },
                    caps,
                )
            };
        for (layout, bits, stored, depth) in [
            (WeightLayout::Nk, (16, 32), Some(FragmentDtype::Bf16), 64),
            (WeightLayout::Kn, (32, 32), None, 32),
            (WeightLayout::Nk, (32, 16), None, 64),
        ] {
            let tiling = planned(layout, bits, stored, &caps);
            let MicroTile::Fragment(tile) = tiling.micro else {
                panic!("{layout:?}: {tiling:?}");
            };
            assert_eq!(
                (
                    tiling.depth(),
                    (tiling.owners_m.get(), tiling.owners_n.get()),
                    (tile.rows, tile.cols)
                ),
                (depth, (4, 2), (2, 4)),
                "{layout:?}: {tiling:?}"
            );
        }
        // The 36 KiB of a 128 x 128 tile 64 deep do not fit a 32 KiB LDS; its 20 KiB 32 deep do.
        let small_lds = poot_target::DeviceCaps {
            lds_bytes: 32 * 1024,
            ..caps
        };
        let tiling = planned(
            WeightLayout::Nk,
            (16, 32),
            Some(FragmentDtype::Bf16),
            &small_lds,
        );
        assert!(
            matches!(tiling.micro, MicroTile::Fragment(_)) && tiling.depth() == 32,
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
                tiled::lds_bytes(tiling.launch_shape(), StagedTerms::ONE) <= 8 * 1024,
                "{tiling:?}"
            );
        }
    }
}
