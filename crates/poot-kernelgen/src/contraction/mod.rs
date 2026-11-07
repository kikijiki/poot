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
mod probe;
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
pub use probe::{Feasibility, SpillCheck, probe_body};

pub(crate) use tiled::{
    GroupedRows, TiledBody, TiledExtents, TiledLaunch, TiledWeight, held_registers, tiled_body,
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
/// wave32 lane holds (`RegisterTiling::loop_registers` bounds what a round holds). A shape that holds no fragment
/// tiling at either stages one fragment step a round ([`RegisterTiling::candidates`]).
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

/// What a tiled contraction's weight values are staged from (POOT-1505): what decides how deep a round reads whole
/// requests of a weight row and what one staged run holds in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagedWeight {
    /// A dense element of `bits` in global memory, rounded up: 32 for f32, 16 for a packed half. A staged value is
    /// the value its load reads.
    Dense {
        /// The bits one element occupies.
        bits: u32,
    },
    /// A packed weight of this format, decoded as it is staged ([`crate::packed_value_bits`] bits per value).
    Packed(poot_quant::format::WeightFormat),
}

impl StagedWeight {
    /// The bits one weight value occupies in global memory, rounded up.
    pub fn bits(self) -> u32 {
        match self {
            Self::Dense { bits } => bits,
            Self::Packed(format) => crate::packed_value_bits(format),
        }
    }

    /// The 32-bit registers one staged run of `run` weight values holds in a tiled round: a dense run its loaded
    /// bits; a packed run the reads its decode issues ahead of the round's multiply
    /// ([`crate::packed_run_held_registers`]) and the `run` f32 values the decode then computes from them, which
    /// stay live until the round's staged write. `None` for a format the decode emitter cannot state.
    pub fn run_registers(self, run: usize) -> Option<u32> {
        match self {
            Self::Dense { bits } => Some((run as u32 * bits).div_ceil(32)),
            Self::Packed(format) => crate::packed_run_held_registers(format, run)
                .ok()
                .map(|held| held + run as u32),
        }
    }
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
    /// What the weight's staged values are read from: a dense element of some bits, or a packed format's decode.
    /// Its bits decide how deep a fragment round must be to read whole requests of a weight row, and its read
    /// decides the registers a round holds ([`StagedWeight::run_registers`]).
    pub weight: StagedWeight,
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
    /// spread evenly over the buckets the rows are expected to touch ([`expected_touched_buckets`]), each bucket's
    /// whole tiles. The model of the cost, from the shape alone.
    pub fn working_row_tiles(self, tile_rows: usize) -> usize {
        match self.rows {
            TiledRows::Dense => self.m_per_block.div_ceil(tile_rows),
            TiledRows::Grouped { groups, .. } => {
                let buckets = expected_touched_buckets(groups, self.m_per_block);
                buckets * self.m_per_block.div_ceil(buckets).div_ceil(tile_rows)
            }
        }
    }
}

/// The buckets `rows` rows are expected to touch among `groups` when each picks one uniformly: `G (1 - (1 - 1/G)^M)`,
/// rounded up, at least one and at most `rows`. A shape carries no routing, so uniform routing is the documented prior
/// (llama.cpp sizes an RDNA MoE tile against the average tokens per expert, `mmq.cu:299-304`; vLLM uses `M / E`): rows
/// share buckets long before `M` reaches `G`, at `M = G` about 37 % of them do.
pub(crate) fn expected_touched_buckets(groups: usize, rows: usize) -> usize {
    let groups = groups.max(1) as f64;
    let touched = groups * (1.0 - (1.0 - 1.0 / groups).powf(rows as f64));
    // Round off the float noise of a whole count before taking the ceiling.
    ((touched - 1e-9).ceil() as usize).clamp(1, rows.max(1))
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

/// The 32-bit registers one staged run of each operand of a shape holds in a tiled round (POOT-1505), at the run
/// every tiling of the shape stages: an activation run its loaded bits, a weight run what
/// [`StagedWeight::run_registers`] says. Computed once per shape for [`RegisterTiling::candidates`]; `None` for a
/// weight whose decode the emitter cannot state, which no tiling runs.
#[derive(Clone, Copy, Debug)]
struct RunRegisters {
    activation: u32,
    weight: u32,
}

impl RunRegisters {
    fn of(shape: TiledShape, run: RunLength) -> Option<Self> {
        let run = run.get() as usize;
        Some(Self {
            activation: (run as u32 * shape.activation_bits).div_ceil(32),
            weight: shape.weight.run_registers(run)?,
        })
    }
}

/// The register bound a candidate's count is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    /// The register model decides: accumulators within their share, the loop's registers within the held share.
    HeldShare,
    /// A compiler decides: the loop's registers within the lane's whole file, the least a lane could hold.
    Lane,
}

/// The schedule [`RegisterTiling::select`] chose for a contraction, which also says at which precision it reads its
/// operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Selection {
    /// A scalar tiling of the exact read (for grouped rows, its output-tile rows are the block it was costed at).
    Exact(RegisterTiling),
    /// The per-row Gemv of the exact read.
    Gemv,
    /// A fragment tiling of the narrowed or split read (for grouped rows, its output-tile rows are the block).
    Narrowed(RegisterTiling),
}

impl Selection {
    /// Whether the contraction reads its operands narrowed: its fragment tiling was the cheapest schedule.
    pub fn narrows(self) -> bool {
        matches!(self, Self::Narrowed(_))
    }
}

impl RegisterTiling {
    /// The tiling of `shape` on `caps`, from an occupancy cost model over every geometry the device can run.
    ///
    /// - Scalar: the narrow and wide lane grids of `caps.gemm_launch()`, each with every register tile of
    ///   [`LANE_ROWS`] x [`LANE_COLS`].
    /// - Fragment, where the operands narrow to a dtype the device has a matrix unit for
    ///   ([`StagedTerms::fragment`]): every wave grid of [`WAVE_GRIDS`] with every [`WAVE_FRAGMENTS`] tile. Its
    ///   round depth is the
    ///   shallowest of [`FRAGMENT_DEPTHS`] at which one round's segment of every operand row is a whole global
    ///   request ([`GLOBAL_REQUEST_BYTES`]; the deepest when none is), or the deepest shallower one it runs
    ///   ([`Self::fragment_depth`]); where no geometry runs at either depth, every geometry at one fragment step.
    ///
    /// A candidate must have a workgroup that fits the device, accumulators and the registers it holds across the
    /// loop within the lane's register budget ([`poot_target::RegisterBudget`], [`Self::loop_registers`]), two staged
    /// tiles that fit its LDS and stage in whole
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
        Self::cheapest(shape, caps, Feasibility::Model).map_or(
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
    /// when the device runs none. Under [`Feasibility::Compiled`] the candidates are compiled in cost order (a tie
    /// going to the larger output tile, then the earlier candidate) and the first whose probe holds its registers is
    /// the cheapest: usually the first compile, the kernel the plan runs.
    fn cheapest(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        feasibility: Feasibility<'_>,
    ) -> Option<(f64, Self)> {
        let mut ranked: Vec<(f64, Self)> = Self::prefiltered(shape, caps, feasibility)
            .into_iter()
            .map(|tiling| (ContractionCost::tiled(tiling, shape, caps), tiling))
            .collect();
        // Stable: equal costs and tiles keep the candidates' order.
        ranked.sort_by(|(a_cost, a), (b_cost, b)| {
            a_cost
                .total_cmp(b_cost)
                .then(b.outputs_per_tile().cmp(&a.outputs_per_tile()))
        });
        match feasibility {
            Feasibility::Model => ranked.into_iter().next(),
            Feasibility::Compiled(check) => ranked.into_iter().find(|&(_, tiling)| {
                probe_body(shape, tiling, caps.rounding_conversions)
                    .is_ok_and(|body| check.holds(&body))
            }),
        }
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
        let mut best: Option<(f64, usize)> = None;
        for block in Self::blocks_of(shapes, caps) {
            let total = shapes.iter().try_fold(0.0, |total, &shape| {
                Self::cheapest(grouped(shape, block), caps, Feasibility::Model)
                    .map(|(cost, _)| total + cost)
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

    /// Every tiling [`Self::choose`] weighs for `shape` on `caps` under the register model: the tilings the device
    /// can run, which a pin ([`Self::runs_over`]) and the calibration sweep enumerate.
    pub fn candidates(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
    ) -> impl Iterator<Item = Self> + '_ {
        Self::prefiltered(shape, caps, Feasibility::Model).into_iter()
    }

    /// The candidates `feasibility` weighs: under [`Feasibility::Model`] each fragment geometry at the depth
    /// [`Self::fragment_depth`] gives it and every tiling within the held share; under [`Feasibility::Compiled`] each
    /// geometry at every whole-request depth, deepest first, and every tiling whose count fits the lane's registers at
    /// all (the compiler decides the rest, [`Self::cheapest`]).
    fn prefiltered(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        feasibility: Feasibility<'_>,
    ) -> Vec<Self> {
        let bound = match feasibility {
            Feasibility::Model => Bound::HeldShare,
            Feasibility::Compiled(_) => Bound::Lane,
        };
        let launch = caps.gemm_launch();
        let run = RunLength::dividing(shape.k);
        let runs = RunRegisters::of(shape, run);
        let scalar = [launch.register_lane_edge, launch.register_lane_edge_wide]
            .into_iter()
            .flat_map(move |lanes| {
                LANE_ROWS.into_iter().flat_map(move |rows| {
                    LANE_COLS.into_iter().map(move |cols| {
                        Self::scalar(TileSize(lanes), LaneRows(rows), MicroCols::new(cols), run)
                    })
                })
            });
        // Every fragment geometry the device runs, its depth chosen by `fragment_depth` (or, for the fallback below, set).
        let geometries = move || {
            shape
                .terms
                .fragment
                .and_then(|dtype| Some((dtype, Self::fragment_config(dtype, caps)?)))
                .into_iter()
                .flat_map(move |(dtype, (layout, waves))| {
                    WAVE_GRIDS.into_iter().flat_map(move |(waves_m, waves_n)| {
                        WAVE_FRAGMENTS.into_iter().flat_map(move |rows| {
                            WAVE_FRAGMENTS.into_iter().filter_map(move |cols| {
                                (waves_m * waves_n <= waves).then_some(Self {
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
                            })
                        })
                    })
                })
        };
        let admitted = move |tiling: &Self| {
            runs.is_some_and(|runs| tiling.runs_on(shape, caps, runs, bound))
                && shape.admits_tile_rows(tiling.output_tile().0)
        };
        let mut fragment: Vec<Self> = match bound {
            Bound::HeldShare => geometries()
                .filter_map(|tiling| tiling.fragment_depth(shape, caps, runs))
                .filter(admitted)
                .collect(),
            Bound::Lane => geometries()
                .flat_map(|tiling| {
                    Self::whole_request_depths(shape)
                        .iter()
                        .rev()
                        .map(move |&depth| Self { depth, ..tiling })
                })
                .filter(admitted)
                .collect(),
        };
        // A shape no fragment tiling holds at a whole-request depth stages one fragment step a round (POOT-1505): the
        // round holds half of what the shallowest whole-request one does, at twice its barriers, which the cost model
        // does not price, so it is the round only of a shape that has no deeper one (a one-wave tile over an act-order
        // GPTQ weight).
        if fragment.is_empty() {
            fragment = geometries()
                .map(|tiling| Self {
                    depth: tiling.micro.kappa(),
                    ..tiling
                })
                .filter(admitted)
                .collect();
        }
        // The units follow the precision (POOT-1476): an exact shape runs on lanes, a narrowed or split one on a
        // matrix unit. A narrowed op never has a scalar tiling, which would round for no speed.
        let narrowed = shape.terms.fragment.is_some();
        scalar
            .filter(move |_| !narrowed)
            .filter(admitted)
            .chain(fragment)
            .collect()
    }

    /// Whether `caps` runs this tiling over `shape`: it is one of [`Self::candidates`]. A caller that pins a tiling
    /// (the calibration sweep) asks this and refuses by name when it is not.
    pub fn runs_over(self, shape: TiledShape, caps: &poot_target::DeviceCaps) -> bool {
        Self::candidates(shape, caps).any(|tiling| tiling == self)
    }

    /// The cheapest tiling of `shape` at its own precision and rows (the scalar tilings of an exact shape, the
    /// fragment tilings of a narrowed or split one), or `None` when the device runs none. What the planner plans a
    /// contraction with once [`Self::select`] has chosen its precision: the precision is that choice's output, so a
    /// narrowed contraction is one whose fragment tiling won.
    pub fn tiling_of(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        feasibility: Feasibility<'_>,
    ) -> Option<Self> {
        Self::cheapest(shape, caps, feasibility).map(|(_, tiling)| tiling)
    }

    /// The schedule of a contraction (POOT-1505): the cheapest, by the
    /// one [`ContractionCost`] model, of every candidate the device runs:
    /// - the scalar tilings of `shape` read exactly;
    /// - the per-row Gemv `gemv`, where the planner has one;
    /// - the fragment tilings of `shape` at its narrowed or split precision, where `shape.terms` permits one.
    ///
    /// Every tiling passes the one feasibility rule, `feasibility` (the register model, or the device's compiler on
    /// the candidate's probe, [`Feasibility`]). Each is costed over `shape`'s rows as
    /// the plan runs them: a grouped shape at every block a tiling of its precision has, the block chosen with the
    /// tiling, so the empty rows of a short expert tile are in every cost. A tie goes
    /// to the exact schedule: a narrowing rounds, so it must win. `None` when the device runs no candidate.
    ///
    /// The precision is this choice's output: a contraction reads its operands narrowed exactly when the choice is
    /// [`Selection::Narrowed`]. A precision policy that permits narrowing never obliges it.
    pub fn select(
        shape: TiledShape,
        gemv: Option<GemvShape>,
        caps: &poot_target::DeviceCaps,
        feasibility: Feasibility<'_>,
    ) -> Option<Selection> {
        let exact = TiledShape {
            terms: StagedTerms::ONE,
            ..shape
        };
        let tiled = Self::cheapest_grouping(exact, caps, feasibility)
            .map(|(cost, tiling)| (cost, Selection::Exact(tiling)));
        let gemv = gemv.map(|gemv| (ContractionCost::gemv(gemv, caps), Selection::Gemv));
        let narrowed = shape.terms.fragment.and_then(|_| {
            Self::cheapest_grouping(shape, caps, feasibility)
                .map(|(cost, tiling)| (cost, Selection::Narrowed(tiling)))
        });
        Self::first_cheapest([tiled, gemv, narrowed].into_iter().flatten())
    }

    /// The cheapest of `ranked` (cost, schedule) pairs, a tie going to the earlier one: [`Self::select`] lists the
    /// exact schedules first, so a tie stays exact.
    fn first_cheapest(ranked: impl Iterator<Item = (f64, Selection)>) -> Option<Selection> {
        ranked
            .reduce(|best, next| if next.0 < best.0 { next } else { best })
            .map(|(_, selection)| selection)
    }

    /// The cheapest tiling of `shape` and its cost: over its rows as given when they are dense, and for grouped rows
    /// at every block a tiling of its precision has ([`Self::blocks_of`]), as a grouping of this contraction alone
    /// would take it (a tie going to the larger block, which runs fewer tiles). `None` when the device runs none.
    fn cheapest_grouping(
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        feasibility: Feasibility<'_>,
    ) -> Option<(f64, Self)> {
        let TiledRows::Grouped { groups, .. } = shape.rows else {
            return Self::cheapest(shape, caps, feasibility);
        };
        Self::blocks_of(&[shape], caps)
            .into_iter()
            .filter_map(|block| {
                Self::cheapest(
                    TiledShape {
                        rows: TiledRows::Grouped { groups, block },
                        ..shape
                    },
                    caps,
                    feasibility,
                )
            })
            .reduce(|best, next| if next.0 <= best.0 { next } else { best })
    }

    /// The output-tile rows some tiling of any of `shapes` has when its rows are free: the blocks a grouping can take,
    /// ascending.
    fn blocks_of(shapes: &[TiledShape], caps: &poot_target::DeviceCaps) -> Vec<usize> {
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
        blocks
    }

    /// Whether the per-row Gemv `gemv` of an indexed contraction costs no more than the tiling of the same rows
    /// grouped by expert (`grouped`: its `rows` are `TiledRows::Grouped`, its block replaced by the cheapest one for
    /// this contraction alone, as a grouping of it by itself would take). Both are costed by the one
    /// [`ContractionCost`] model, the grouped one over the buckets the rows are expected to touch
    /// ([`TiledShape::working_row_tiles`]); a grouped shape the device runs no tiling for loses to any Gemv. The one
    /// question that decides where an indexed contraction stops being a Gemv, from the dimensions and the device
    /// alone.
    pub fn gemv_beats_grouped(
        gemv: GemvShape,
        grouped: TiledShape,
        caps: &poot_target::DeviceCaps,
    ) -> bool {
        if !matches!(grouped.rows, TiledRows::Grouped { .. }) {
            return false;
        }
        Self::cheapest_grouping(grouped, caps, Feasibility::Model)
            .is_none_or(|(grouped, _)| ContractionCost::gemv(gemv, caps) <= grouped)
    }

    /// The `K` values one barrier round of this tiling stages.
    pub fn depth(self) -> usize {
        self.depth
    }

    /// This fragment geometry at its round depth, or `None` when the device runs it at none: the shallowest of
    /// [`FRAGMENT_DEPTHS`] whose round segment of every operand row (the activation's rows at
    /// `shape.activation_bits` per value, the weight's at `shape.weight`'s bits) is at least one
    /// [`GLOBAL_REQUEST_BYTES`] request, the deepest when none is; and if [`Self::runs_on`] refuses that depth (its
    /// staged tiles outgrow the LDS, or the round it holds outgrows the lane's registers), the deepest shallower one
    /// it runs. A 32-deep round of a bf16 row reads half a line per request: the vector memory path pays a request
    /// for it, so the round doubles; an f32 activation row is a whole line at 32 already.
    fn fragment_depth(
        self,
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        runs: Option<RunRegisters>,
    ) -> Option<Self> {
        Self::whole_request_depths(shape)
            .iter()
            .rev()
            .map(|&depth| Self { depth, ..self })
            .find(|tiling| {
                runs.is_some_and(|runs| tiling.runs_on(shape, caps, runs, Bound::HeldShare))
            })
    }

    /// [`FRAGMENT_DEPTHS`] up to the shallowest at which one round's segment of every operand row of `shape` is a
    /// whole [`GLOBAL_REQUEST_BYTES`] request (all of them when none is): the depths a fragment geometry may run.
    fn whole_request_depths(shape: TiledShape) -> &'static [usize] {
        let narrowest = shape.activation_bits.min(shape.weight.bits()).max(1) as usize;
        let whole = FRAGMENT_DEPTHS
            .iter()
            .position(|&depth| depth * narrowest >= GLOBAL_REQUEST_BYTES * 8)
            .unwrap_or(FRAGMENT_DEPTHS.len() - 1);
        &FRAGMENT_DEPTHS[..=whole]
    }

    /// The f32 registers each lane of this tiling holds across the `K` loop: a scalar tile's accumulators, a
    /// fragment tile's accumulator fragments.
    pub(crate) fn accumulator_registers(self) -> u32 {
        match self.micro {
            MicroTile::Scalar { rows, cols } => rows.get() * cols.get(),
            MicroTile::Fragment(tile) => tile.rows * tile.cols * FRAGMENT_ACCUMULATOR_REGISTERS,
        }
    }

    /// The 32-bit registers each lane of this tiling holds at once across its `K` loop (POOT-1505): its
    /// accumulators, and its share of a round of both operands, `runs` registers per staged run of each. The loop is
    /// software-pipelined: a round's reads are issued before the previous round's multiply and decoded after it, so
    /// a lane holds the next round's reads through the multiply, then those reads and the values they decode to
    /// until the round's staged write. A lane stages `rows * depth / lanes` activation values and
    /// `cols * depth / lanes` weight values of a round (`(rows, cols)` the output tile).
    fn loop_registers(self, runs: RunRegisters) -> u32 {
        let lanes = self.workgroup_lanes();
        let (rows, cols) = self.output_tile();
        let run = self.run.get() as usize;
        let lane_runs = |extent: usize| (extent * self.depth).div_ceil(lanes).div_ceil(run) as u32;
        self.accumulator_registers()
            + lane_runs(rows) * runs.activation
            + lane_runs(cols) * runs.weight
    }

    /// Whether `caps` runs this tiling over `shape`: its workgroup fits, a lane's accumulators fit its register
    /// budget (`caps.gemm_launch().register_budget`, a quarter of what a lane of this workgroup keeps resident) and
    /// so do all the registers it holds across the loop ([`Self::loop_registers`] within the budget's
    /// [`poot_target::RegisterBudget::loop_registers`], half of it: a tiling over it spills, POOT-1505), its staged
    /// tiles (and the scratch an `Opaque` fragment spills to) fit the LDS and stage in whole slots. Under
    /// [`Bound::Lane`] (a compiler decides the spill) the registers it holds need only fit the lane's whole file.
    fn runs_on(
        self,
        shape: TiledShape,
        caps: &poot_target::DeviceCaps,
        runs: RunRegisters,
        bound: Bound,
    ) -> bool {
        let lane_count = self.workgroup_lanes();
        let budget = caps.gemm_launch().register_budget;
        u32::try_from(lane_count).is_ok_and(|lanes| match bound {
            Bound::HeldShare => {
                self.accumulator_registers() <= budget.accumulators(lanes)
                    && self.loop_registers(runs) <= budget.loop_registers(lanes)
            }
            Bound::Lane => self.loop_registers(runs) <= budget.lane_registers(lanes),
        }) && lane_count <= caps.max_workgroup_invocations as usize
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

    impl RegisterTiling {
        /// [`Self::loop_registers`] of this tiling over `shape`, `None` for a weight whose decode the emitter cannot
        /// state: what the register spill rows (`register_budget`) hold the compiler's count against.
        pub(crate) fn loop_registers_over(self, shape: TiledShape) -> Option<u32> {
            RunRegisters::of(shape, self.run).map(|runs| self.loop_registers(runs))
        }
    }

    fn shape(m: usize, k: usize, n: usize) -> TiledShape {
        TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            rows: crate::TiledRows::Dense,
            terms: crate::precision::StagedTerms::ONE,
            weight: crate::StagedWeight::Dense { bits: 32 },
            activation_bits: 32,
        }
    }

    /// Whether the selection over `shape`'s exact and narrowed schedules chooses its fragment tiling.
    fn narrows(shape: TiledShape, gemv: Option<GemvShape>, caps: &poot_target::DeviceCaps) -> bool {
        RegisterTiling::select(shape, gemv, caps, Feasibility::Model)
            .is_some_and(Selection::narrows)
    }

    /// A one-row dense decode contraction stays on the Gemv on gfx1151 through HSA at every output count, the vocabulary
    /// head's (150 000 columns and up) included: past the point where both launches fill the device, the Gemv costs
    /// `1 + (1 + 1 / cols) * direct_value` issues per output and a fragment tiling a fixed 20, so the Gemv's direct
    /// value must stay under the crossover (15.2 for four columns per workgroup). The direct value POOT-1489 fitted
    /// on the indexed corpus alone (16.8) sat above it, so the head moved onto a matrix unit that reads the same
    /// weight bytes through an LDS stage and decode went 8 % slower (POOT-1503). Mutation: restore
    /// `STRIX_HALO_HSA_COST_RATIOS.direct_value` to 16.8; the rows from 16 384 columns up go red (observed: `[1, 1024] x [16384, 1024]^T moved onto a matrix unit`).
    #[test]
    fn a_one_row_dense_decode_stays_on_the_gemv_at_every_output_count() {
        let caps = poot_target::DeviceCaps {
            matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::amd("gfx1151", 32)),
            compute_units: poot_target::Queried::Known(40),
            perf_calibration: poot_target::calibration::Calibrated::STRIX_HALO_HSA,
            ..poot_target::DeviceCaps::rocm_default()
        };
        for (n, k) in [
            (1024, 1024),
            (16384, 1024),
            (65536, 1024),
            (128_256, 4096),
            (151_936, 1024),
            (151_936, 2560),
            (262_144, 2048),
        ] {
            let narrow = TiledShape {
                terms: crate::precision::StagedTerms {
                    fragment: Some(FragmentDtype::F16),
                    ..crate::precision::StagedTerms::ONE
                },
                weight: crate::StagedWeight::Dense { bits: 16 },
                ..shape(1, k, n)
            };
            let gemv = GemvShape {
                outputs: n,
                width: 128,
                cols: 4,
            };
            assert!(
                !narrows(narrow, Some(gemv), &caps),
                "[1, {k}] x [{n}, {k}]^T moved onto a matrix unit"
            );
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

    /// Rows spread uniformly over buckets touch `G (1 - (1 - 1/G)^M)` of them, rounded up: 82 of 129 buckets at
    /// `M = 129` (rows share long before `M` reaches `G`), every bucket at `M >> G`, and never more than the rows or
    /// fewer than one. Mutation: answer `groups.min(rows)` (the old even spread); the `M = G` row goes red.
    #[test]
    fn rows_expected_to_touch_fewer_buckets_than_they_have() {
        assert_eq!(expected_touched_buckets(129, 129), 82);
        assert_eq!(expected_touched_buckets(129, 4), 4);
        assert_eq!(expected_touched_buckets(129, 4096), 129);
        assert_eq!(expected_touched_buckets(1, 5), 1);
        assert_eq!(expected_touched_buckets(129, 0), 1);
        // The working tiles follow: 129 rows over 129 buckets are 82 buckets of 2 rows, each its own 16-row tile.
        let grouped = TiledShape {
            rows: TiledRows::Grouped {
                groups: 129,
                block: 16,
            },
            ..shape(129, 2048, 1024)
        };
        assert_eq!(grouped.working_row_tiles(16), 82);
    }

    /// A routed projection of `rows` rows over `experts` experts of a bf16 `[N, K]` stack: the grouped tiled shape and
    /// the Gemv launch the planner would cost against it.
    fn routed(rows: usize, experts: usize) -> (GemvShape, TiledShape) {
        let grouped = TiledShape {
            blocks: 1,
            m_per_block: rows,
            out_per_block: 1024,
            k: 1024,
            layout: WeightLayout::Nk,
            rows: TiledRows::Grouped {
                groups: experts + 1,
                block: 0,
            },
            terms: crate::precision::StagedTerms::ONE,
            weight: crate::StagedWeight::Dense { bits: 16 },
            activation_bits: 32,
        };
        let gemv = GemvShape {
            outputs: rows * 1024,
            width: 256,
            cols: 4,
        };
        (gemv, grouped)
    }

    /// Where an indexed contraction stops being a Gemv is the cost model's (POOT-1489), and on gfx1151 through HSA it
    /// is where `calibrate-launch indexed` measured it for a `1024 x 1024` bf16 expert: 16 rows over 8 experts are a
    /// Gemv (the grouping timed 2.1x slower), 256 over 8 are grouped (it timed 0.6x), 256 over 128 are a Gemv (2.1x),
    /// 4096 over 128 grouped again (0.16x). The retired 32-row switch had the third row wrong. Mutation: compare
    /// `gemv > grouped` in `gemv_beats_grouped`; every row inverts and goes red.
    #[test]
    fn the_gemv_gives_way_to_the_grouping_where_the_rows_share_their_experts() {
        let backend = poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151());
        let caps = poot_test_util::device_caps::default_caps_for(backend);
        for (rows, experts, gemv_wins) in [
            (16, 8, true),
            (256, 8, false),
            (256, 128, true),
            (4096, 128, false),
        ] {
            let (gemv, grouped) = routed(rows, experts);
            assert_eq!(
                RegisterTiling::gemv_beats_grouped(gemv, grouped, &caps),
                gemv_wins,
                "{rows} rows over {experts} experts"
            );
        }
    }

    /// A grouped shape the device runs no tiling for loses to any Gemv, and a dense shape is never the grouping's:
    /// the comparison answers only for rows grouped by expert. Mutation: answer `true` for a dense shape; the second
    /// assertion goes red.
    #[test]
    fn the_comparison_is_for_grouped_rows_on_a_device_that_tiles_them() {
        let caps = poot_test_util::device_caps::wgpu_rdna3_igpu();
        let (gemv, grouped) = routed(256, 8);
        let dense = TiledShape {
            rows: TiledRows::Dense,
            ..grouped
        };
        assert!(!RegisterTiling::gemv_beats_grouped(gemv, dense, &caps));
        let no_tiling = poot_target::DeviceCaps {
            max_workgroup_invocations: 1,
            max_workgroup_size: [1, 1, 1],
            ..caps
        };
        assert!(
            RegisterTiling::gemv_beats_grouped(gemv, grouped, &no_tiling),
            "a device that runs no tile of the grouping runs the Gemv"
        );
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
        assert!(!narrows(narrow, None, &device(1)));
        let held = RegisterTiling::tiling_of(narrow, &device(1), Feasibility::Model)
            .expect("a declared narrowing runs");
        assert!(matches!(held.micro, MicroTile::Fragment(_)), "{held:?}");
        let many_waves = FragmentConfig::ANY_WAVES;
        let tiling = RegisterTiling::tiling_of(narrow, &device(many_waves), Feasibility::Model)
            .expect("fragments");
        assert!(
            matches!(tiling.micro, MicroTile::Fragment(_))
                && tiling.workgroup_lanes() > FRAGMENT_LANES,
            "{tiling:?}"
        );
    }

    /// The matrix unit is the cheapest schedule of a prefill contraction on gfx1151, and the per-row Gemv is the
    /// cheapest of a decode row: the Gemv's cost (`ContractionCost::gemv`, the same unit as the tilings') beats the
    /// fragment tiling of one row, and loses to it once the launch covers enough rows to fill the device. A narrowed
    /// shape has no scalar candidate. Mutation: drop the Gemv alternative from `RegisterTiling::select`; the one-row
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
            weight: crate::StagedWeight::Dense { bits: 16 },
            ..shape(m, 2048, 2048)
        };
        let gemv = |rows: usize| GemvShape {
            outputs: rows * 2048,
            width: 256,
            cols: 4,
        };
        assert!(!narrows(narrow(1), Some(gemv(1)), &caps));
        assert!(
            narrows(narrow(1), None, &caps),
            "the Gemv term is what keeps one row exact"
        );
        assert!(narrows(narrow(2048), Some(gemv(2048)), &caps));
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
    /// the activation alone (drop `shape.weight.bits()` from `fragment_depth`'s `narrowest`); the bf16 weight row takes
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
                        weight: StagedWeight::Dense { bits: weight_bits },
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

    /// POOT-1505: a fragment tiling of a short-prompt projection over a packed expert stack holds no more registers
    /// across its loop than the lane's held share. The Flash-Next expert up projection at 128 tokens (1280 IQ2_XS rows
    /// routed to 512 experts, K 2560, N 640) narrowed to f16 on gfx1151 grouped in 16-row tiles: the 16 x 128 tile at
    /// a 64-deep round the planner took before (two waves; 432 registers a lane holds across the loop against the
    /// held share of 128, 720 VGPRs spilled when compiled; the 128-token first token took twice the exact plan's,
    /// POOT-1504) is no candidate, and neither is its 32-deep round (232). A 16-row fragment tiling remains (the
    /// 16 x 64 tile at a 32-deep round, 120), so the contraction still narrows. Mutation: drop the
    /// `loop_registers` clause of `RegisterTiling::runs_on`; the 16 x 128 tile at 64 is a candidate again and the
    /// row names it.
    #[test]
    fn a_packed_expert_round_stays_within_the_lanes_held_registers() {
        let backend = poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151());
        let caps = poot_test_util::device_caps::default_caps_for(backend);
        let precision = poot_graph_ir::OperandPrecision::Narrow(FragmentDtype::F16);
        let grouped = TiledShape {
            blocks: 1,
            m_per_block: 1280,
            out_per_block: 640,
            k: 2560,
            layout: WeightLayout::Nk,
            rows: TiledRows::Grouped {
                groups: 513,
                block: 16,
            },
            terms: crate::precision::StagedTerms::of(precision, None, None),
            weight: StagedWeight::Packed(poot_quant::format::WeightFormat::Iq2_Xs),
            activation_bits: 32,
        };
        let (layout, _) = RegisterTiling::fragment_config(FragmentDtype::F16, &caps)
            .expect("gfx1151 has an f16 matrix unit");
        let tile = |cols, depth| RegisterTiling {
            owners_m: TileSize(1),
            owners_n: TileSize(2),
            micro: MicroTile::Fragment(FragmentTile {
                dtype: FragmentDtype::F16,
                layout,
                rows: 1,
                cols,
            }),
            run: RunLength::dividing(2560),
            depth,
        };
        let candidates: Vec<RegisterTiling> = RegisterTiling::candidates(grouped, &caps).collect();
        for refused in [tile(4, 64), tile(4, 32)] {
            assert!(
                !candidates.contains(&refused),
                "a tiling that spills its round is a candidate: {:?}",
                refused.schedule()
            );
        }
        assert!(
            candidates.contains(&tile(2, 32)),
            "the 16 x 64 tile at 32 holds its round: {candidates:?}"
        );
        assert!(RegisterTiling::tiling_of(grouped, &caps, Feasibility::Model).is_some());
    }

    /// POOT-1505: a shape no fragment tiling holds at a whole-request depth stages one fragment step a round, and only
    /// such a shape. On the wgpu caps (a one-wave fragment workgroup) a narrowed packed linear of 4 rows over an
    /// act-order GPTQ weight (8 columns of 256, 4 groups) holds 136 registers in the 16 x 16 tile at a 32-deep round
    /// against the held share of 128, and plans at 16 (72); the same linear over Q8_0 holds a 64-deep round (104) and
    /// keeps it. Mutation: drop the one-step fallback of `RegisterTiling::candidates`; the GPTQ row has no fragment
    /// tiling and reads `None`.
    #[test]
    fn a_shape_no_whole_round_holds_stages_one_fragment_step() {
        use poot_quant::format::{GroupMap, WeightFormat};
        let caps = poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan);
        let linear = |format| TiledShape {
            blocks: 1,
            m_per_block: 4,
            out_per_block: 8,
            k: 256,
            layout: WeightLayout::Nk,
            rows: TiledRows::Dense,
            terms: crate::precision::StagedTerms::of(
                poot_graph_ir::OperandPrecision::Narrow(FragmentDtype::F16),
                None,
                None,
            ),
            weight: StagedWeight::Packed(format),
            activation_bits: 32,
        };
        let act_order = WeightFormat::Gptq {
            groups: GroupMap::Indexed {
                groups: std::num::NonZeroUsize::new(4).unwrap(),
            },
        };
        let depth = |format| {
            RegisterTiling::tiling_of(linear(format), &caps, Feasibility::Model)
                .map(RegisterTiling::depth)
        };
        assert_eq!(
            depth(act_order),
            Some(16),
            "the act-order GPTQ linear: one 16-deep fragment step"
        );
        assert_eq!(depth(WeightFormat::Q8_0), Some(64), "the Q8_0 linear");
    }

    /// The selection costs a grouped contraction at the blocks its grouping can take, not over its ungrouped rows
    /// (POOT-1505): the Flash-Next expert projection at a 128-token prompt (IQ2_XS, `K` 2560, `N` 640, 1280 rows
    /// routed over 513 buckets) on gfx1151 narrows onto a 16-row fragment tile, the block its buckets fill, and the
    /// grouped plan runs that tiling at that block. Over 1280 dense rows the cheapest fragment tile is 128 rows tall,
    /// which no grouping of these buckets would plan. Mutation: cost the narrowed candidates over `TiledRows::Dense` in
    /// `RegisterTiling::select`; the selection reads a 128-row tile (observed: `selected FragmentTiled { waves_m: 4, waves_n: 2,
    /// .. rows: 2, cols: 4 }, run: 4, depth: 32 }`, left 128, right 16).
    #[test]
    fn a_grouped_contraction_is_selected_at_the_block_its_grouping_takes() {
        let caps = poot_target::DeviceCaps {
            matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::amd("gfx1151", 32)),
            register_file: poot_target::Queried::Known(
                poot_target::RegisterFile::amd("gfx1151", 32).expect("gfx1151's file"),
            ),
            compute_units: poot_target::Queried::Known(40),
            perf_calibration: poot_target::calibration::Calibrated::STRIX_HALO_HSA,
            ..poot_target::DeviceCaps::rocm_default()
        };
        let grouped = |block| TiledShape {
            blocks: 1,
            m_per_block: 1280,
            out_per_block: 640,
            k: 2560,
            layout: WeightLayout::Nk,
            rows: TiledRows::Grouped { groups: 513, block },
            terms: crate::precision::StagedTerms::of(
                poot_graph_ir::OperandPrecision::Narrow(FragmentDtype::F16),
                None,
                None,
            ),
            weight: StagedWeight::Packed(poot_quant::format::WeightFormat::Iq2_Xs),
            activation_bits: 32,
        };
        let Some(Selection::Narrowed(tiling)) =
            RegisterTiling::select(grouped(16), None, &caps, Feasibility::Model)
        else {
            panic!("the expert projection stays exact");
        };
        let rows = tiling.output_tile().0;
        assert_eq!(rows, 16, "selected {:?}", tiling.schedule());
        assert!(
            tiling.runs_over(grouped(rows), &caps),
            "{:?}",
            tiling.schedule()
        );
        assert_eq!(
            RegisterTiling::tiling_of(grouped(rows), &caps, Feasibility::Model),
            Some(tiling)
        );
    }

    /// A check that holds every probe but counts them, and refuses the probes whose names contain `refused`.
    struct Counted {
        refused: Option<String>,
        asked: std::cell::Cell<usize>,
    }

    impl SpillCheck for Counted {
        fn holds(&self, probe: &poot_kernel_ir::Body) -> bool {
            self.asked.set(self.asked.get() + 1);
            self.refused
                .as_ref()
                .is_none_or(|refused| !probe.name.contains(refused.as_str()))
        }
    }

    /// The gfx1151 ROCm caps with the agent's register file.
    fn gfx1151() -> poot_target::DeviceCaps {
        poot_test_util::device_caps::default_caps_for(poot_target::Backend::AmdGcn(
            poot_target::AmdArch::gfx1151(),
        ))
    }

    /// The narrowed bf16 prefill projection of the bank (2048 rows of a 3072 x 1024 weight).
    fn narrowed_prefill() -> TiledShape {
        TiledShape {
            terms: crate::precision::StagedTerms {
                fragment: Some(FragmentDtype::F16),
                ..crate::precision::StagedTerms::ONE
            },
            weight: StagedWeight::Dense { bits: 16 },
            ..shape(2048, 1024, 3072)
        }
    }

    /// Under compile feedback the candidates are compiled in cost order and the first that holds is the tiling: a
    /// compiler that holds everything is asked once, and one that spills that tiling is asked again and gets the next
    /// candidate, which costs no less. Mutation: in `RegisterTiling::cheapest`, take the first candidate in the
    /// candidates' order instead of the cost order; the first compile is not the model's cheapest.
    #[test]
    fn a_compiled_selection_takes_the_cheapest_candidate_that_holds() {
        let caps = gfx1151();
        let shape = narrowed_prefill();
        let all = Counted {
            refused: None,
            asked: Default::default(),
        };
        let (first_cost, first) =
            RegisterTiling::cheapest(shape, &caps, Feasibility::Compiled(&all))
                .expect("a tiling holds");
        assert_eq!(all.asked.get(), 1, "the cheapest holds: one compile");
        let lane = RegisterTiling::prefiltered(shape, &caps, Feasibility::Compiled(&all));
        let least = lane
            .iter()
            .map(|&tiling| ContractionCost::tiled(tiling, shape, &caps))
            .fold(f64::INFINITY, f64::min);
        assert_eq!(
            first_cost, least,
            "the first compile is the cheapest candidate"
        );
        let name = probe_body(shape, first, caps.rounding_conversions)
            .unwrap()
            .name;
        let spills_first = Counted {
            refused: Some(name),
            asked: Default::default(),
        };
        let (next_cost, next) =
            RegisterTiling::cheapest(shape, &caps, Feasibility::Compiled(&spills_first))
                .expect("another tiling holds");
        assert_ne!(next, first);
        assert!(next_cost >= first_cost);
        assert!(spills_first.asked.get() >= 2);
    }

    /// The compiler, not the held share, decides: the Flash-Next expert projection's one-wave 16 x 16 tile at a 64-deep
    /// round counts 136 registers on an sm_120 lane, over the held share (127), so the register model never weighs
    /// it, and under compile feedback it is a candidate (POOT-1509 measured it at 2.00 ms against the exact 2.35 ms,
    /// unspilled). Mutation: hold `Bound::Lane` to `RegisterBudget::loop_registers` in `RegisterTiling::runs_on`; the
    /// tile is no candidate.
    #[test]
    fn compile_feedback_weighs_a_tiling_the_held_share_refuses() {
        let caps = poot_target::DeviceCaps {
            matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::nvidia((12, 0))),
            register_file: poot_target::RegisterFile::nvidia(12)
                .map_or(poot_target::Queried::Unknown, poot_target::Queried::Known),
            ..poot_target::DeviceCaps::ptx_default()
        };
        let grouped = TiledShape {
            blocks: 1,
            m_per_block: 1280,
            out_per_block: 640,
            k: 2560,
            layout: WeightLayout::Nk,
            rows: TiledRows::Grouped {
                groups: 513,
                block: 16,
            },
            terms: crate::precision::StagedTerms::of(
                poot_graph_ir::OperandPrecision::Narrow(FragmentDtype::F16),
                None,
                None,
            ),
            weight: StagedWeight::Packed(poot_quant::format::WeightFormat::Iq2_Xs),
            activation_bits: 32,
        };
        let layout = caps
            .matrix_units
            .known()
            .and_then(|units| Some(units.config(FragmentDtype::F16)?.layout))
            .unwrap();
        let tile = RegisterTiling {
            owners_m: TileSize(1),
            owners_n: TileSize(1),
            micro: MicroTile::Fragment(FragmentTile {
                dtype: FragmentDtype::F16,
                layout,
                rows: 1,
                cols: 1,
            }),
            run: RunLength::dividing(2560),
            depth: 64,
        };
        assert!(!RegisterTiling::candidates(grouped, &caps).any(|t| t == tile));
        let all = Counted {
            refused: None,
            asked: Default::default(),
        };
        assert!(
            RegisterTiling::prefiltered(grouped, &caps, Feasibility::Compiled(&all))
                .contains(&tile),
            "the compiler is asked about the 64-deep 16 x 16 tile"
        );
    }

    /// The probe of a tiling is a body for each kind of contraction the planner tiles: a dense weight read as `[N, K]`
    /// or grouped by expert, and a packed weight over plain or grouped rows. Mutation: return an error for the
    /// grouped dense arm of `probe_body`; that row reads `Err`.
    #[test]
    fn every_tiled_contraction_has_a_probe() {
        let caps = gfx1151();
        let grouped = |shape: TiledShape| TiledShape {
            rows: TiledRows::Grouped {
                groups: 9,
                block: 16,
            },
            ..shape
        };
        let packed = TiledShape {
            weight: StagedWeight::Packed(poot_quant::format::WeightFormat::Q4_K),
            ..shape(512, 2560, 768)
        };
        for shape in [
            narrowed_prefill(),
            grouped(shape(512, 1024, 512)),
            packed,
            grouped(packed),
        ] {
            let tiling = RegisterTiling::tiling_of(shape, &caps, Feasibility::Model)
                .expect("the model tiles it");
            assert!(
                probe_body(shape, tiling, caps.rounding_conversions).is_ok(),
                "{shape:?} under {:?}",
                tiling.schedule()
            );
        }
    }

    /// A tie between an exact schedule and the narrowed one stays exact: a narrowing rounds, so it must win.
    /// Mutation: let a later candidate replace the best on `<=` in `RegisterTiling::first_cheapest`; the narrowed
    /// tiling is chosen.
    #[test]
    fn a_tie_between_exact_and_narrowed_stays_exact() {
        let tiling =
            RegisterTiling::choose(shape(64, 64, 64), &poot_target::DeviceCaps::rocm_default());
        let ranked = [
            (2.0, Selection::Exact(tiling)),
            (1.0, Selection::Gemv),
            (1.0, Selection::Narrowed(tiling)),
        ];
        assert_eq!(
            RegisterTiling::first_cheapest(ranked.into_iter()),
            Some(Selection::Gemv)
        );
    }
}
