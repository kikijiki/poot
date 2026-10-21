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

pub(crate) use dense::dense_contraction;
pub(crate) use epilogue::emit_epilogue;
pub use epilogue::{Epilogue, EpilogueInput, EpilogueLeaf, EpilogueStep};
pub(crate) use gemv::{GemvEpilogue, GemvLaunch, gemv_body};
pub(crate) use indexed::{
    IndexedTiledBody, IndexedTiledWeight, META_EXPERTS, META_N, META_PACKED_SLAB_WORDS, RowExpert,
    emit_poison, emit_row_expert, indexed_contraction, indexed_tiled_body,
};

pub(crate) use tiled::{TiledBody, TiledLaunch, tiled_body};

use crate::KernelGenError;

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
    },
    /// `M > 1`, an indexed (row-selected) contraction: LDS-staged tiles (542b), each lane computing one output of
    /// the `tile x tile` tile its workgroup owns.
    Tiled {
        /// The edge of the square output tile one workgroup computes.
        tile: TileSize,
    },
    /// `M > 1`, a plain (block-diagonal) contraction: LDS-staged tiles with a register tile per lane. A workgroup
    /// of `lanes x lanes` lanes computes the [`Schedule::output_tile`] its lanes interleave, each accumulating a
    /// `rows x 4` part of it, so one decoded weight value feeds `rows` FMAs per LDS read instead of one. `K` is
    /// staged in runs of `run` consecutive values, which the weight's decode shares an address chain, word load
    /// and scale across. [`RegisterTiling::choose`] picks the geometry from the shapes.
    RegisterTiled {
        /// The edge of the square lane grid one workgroup runs.
        lanes: TileSize,
        /// Output rows each lane accumulates.
        rows: LaneRows,
        /// Consecutive `K` values one lane loads and decodes at a time; divides `K`.
        run: RunLength,
    },
}

/// The output rows a [`Schedule::RegisterTiled`] lane accumulates, built only by [`LaneRows::covering`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneRows(pub(crate) u32);

impl LaneRows {
    /// The rows per lane for a workgroup of `lanes x lanes` lanes over `m_per_block` rows: 8 once the rows fill
    /// a tile of 8 per lane, 4 for half of that, else 2, so a tile never computes many more rows than there are.
    fn covering(m_per_block: usize, lanes: usize) -> Self {
        Self(match m_per_block.div_ceil(lanes) {
            rows if rows >= 8 => 8,
            rows if rows >= 4 => 4,
            _ => 2,
        })
    }

    /// The rows per lane, at least 1.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The bare count, so a schedule's `Debug` (part of the planner's kernel key) reads `rows: 8`.
impl std::fmt::Debug for LaneRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
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

/// The tile edge of a [`Schedule::Tiled`] contraction (or the lane-grid edge of a [`Schedule::RegisterTiled`] one),
/// built only by [`TileSize::new`]. Card 658: a
/// row-tile is laid out per block ([`Schedule::grid_threads`], `tiled_body`'s grid derivation),
/// so it never straddles two blocks' weights and this schedule serves any `blocks >= 1` - `m_per_block`
/// need not be a multiple of `tile`.
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
    /// `out_per_block` weight rows per block: `width` lanes per `cols` outputs for `Gemv`, `tile^2`
    /// lanes per `tile x tile` output block for `Tiled` - laid out per block (`blocks *
    /// ceil(m_per_block/tile) * ceil(out_per_block/tile)` tiles), card 658, so a row-tile never spans
    /// two blocks and `m_per_block` need not be a multiple of `tile`. The one owner of each body's
    /// grid convention.
    pub fn grid_threads(self, blocks: usize, m_per_block: usize, out_per_block: usize) -> usize {
        let rows = blocks * m_per_block;
        let outputs = rows * out_per_block;
        match self {
            Schedule::Gemv { width, cols, .. } => outputs.div_ceil(cols as usize) * width as usize,
            Schedule::Tiled { tile } => {
                let ts = tile.get() as usize;
                blocks * m_per_block.div_ceil(ts) * out_per_block.div_ceil(ts) * ts * ts
            }
            Schedule::RegisterTiled { lanes, rows, .. } => {
                let (lanes, rows) = (lanes.get() as usize, rows.get() as usize);
                blocks
                    * m_per_block.div_ceil(tiled::tile_rows(lanes, rows))
                    * out_per_block.div_ceil(tiled::tile_cols(lanes))
                    * lanes
                    * lanes
            }
        }
    }

    /// The `(rows, columns)` of the output tile one workgroup of a tiled schedule computes, or `None` for the Gemv.
    pub fn output_tile(self) -> Option<(usize, usize)> {
        match self {
            Schedule::Gemv { .. } => None,
            Schedule::Tiled { tile } => Some((tile.get() as usize, tile.get() as usize)),
            Schedule::RegisterTiled { lanes, rows, .. } => {
                Some(RegisterTiling::output_tile_of(lanes, rows))
            }
        }
    }
}

/// Rows per lane of the wide register tile that [`RegisterTiling::choose`] sizes: with the 16-lane wide edge it is
/// the 128-row tile (card 542b / POOT-1193, the shape the register-tiled kernel was swept at).
const WIDE_TILE_LANE_ROWS: usize = 8;

/// The geometry of a [`Schedule::RegisterTiled`] contraction, chosen from the shapes and the device by
/// [`RegisterTiling::choose`]. It is the schedule's fields before they are wrapped, so a planner reads the lane
/// grid and output tile it launches without destructuring the [`Schedule`] again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterTiling {
    lanes: TileSize,
    rows: LaneRows,
    run: RunLength,
}

impl RegisterTiling {
    /// The tiling for a block-diagonal contraction of `m_per_block` activation rows per block over a weight whose
    /// rows hold `k` values, on `caps`: the wide lane grid (`caps.gemm_launch`, a 128-row tile at 16x16) once there
    /// are enough rows and the device runs its workgroups with room for the two tiles, else the narrow grid; rows
    /// per lane so the tile does not far exceed the rows ([`LaneRows::covering`]); and the longest run that
    /// divides `k`. Chosen from shapes and device limits alone, never from the format.
    pub fn choose(m_per_block: usize, k: usize, caps: &poot_target::DeviceCaps) -> Self {
        let launch = caps.gemm_launch();
        let wide = launch.register_lane_edge_wide as usize;
        let narrow = launch.register_lane_edge as usize;
        let wide_rows = tiled::tile_rows(wide, WIDE_TILE_LANE_ROWS);
        let fits = wide * wide <= caps.max_workgroup_invocations as usize
            && wide * wide <= caps.max_workgroup_size[0] as usize
            && tiled::lds_bytes(wide, WIDE_TILE_LANE_ROWS) <= caps.lds_bytes as usize;
        let lanes = if m_per_block >= wide_rows && fits {
            wide
        } else {
            narrow
        };
        Self {
            lanes: TileSize(lanes as u32),
            rows: LaneRows::covering(m_per_block, lanes),
            run: RunLength::dividing(k),
        }
    }

    /// The schedule that runs this tiling.
    pub fn schedule(self) -> Schedule {
        Schedule::RegisterTiled {
            lanes: self.lanes,
            rows: self.rows,
            run: self.run,
        }
    }

    /// The edge of the square lane grid one workgroup runs.
    pub fn lanes(self) -> usize {
        self.lanes.get() as usize
    }

    /// The `(rows, columns)` of the output tile one workgroup computes.
    pub fn output_tile(self) -> (usize, usize) {
        Self::output_tile_of(self.lanes, self.rows)
    }

    fn output_tile_of(lanes: TileSize, rows: LaneRows) -> (usize, usize) {
        let (lanes, rows) = (lanes.get() as usize, rows.get() as usize);
        (tiled::tile_rows(lanes, rows), tiled::tile_cols(lanes))
    }
}

/// The bare edge, so a schedule's `Debug` (part of the planner's kernel key) reads `Tiled { tile: 8 }`.
impl std::fmt::Debug for TileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
