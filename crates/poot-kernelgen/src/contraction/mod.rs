//! The contraction lowering (Card 727): one loop structure ([`Schedule`]) per
//! contraction shape, and one body per schedule that serves every operand load. A load supplies only the
//! read of a run of weight values: the dense loader ([`DenseLoad`], an f32, f16-packed or bf16-packed
//! weight) here, the descriptor-driven block decode in `crate::packed`. The planner picks the schedule and
//! its launch from shapes and the device ([`poot_target::DeviceCaps::compute_units`]); the generator only
//! implements it.

mod dense;
mod gemv;

pub(crate) use dense::DenseLoad;
pub(crate) use dense::dense_contraction;
pub(crate) use gemv::{GemvLaunch, gemv_body};

use crate::KernelGenError;

/// The contraction's loop structure, chosen by the planner from shapes, never from the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    /// `M == 1`: one workgroup of `width` lanes owns `cols` adjacent output columns (weight rows),
    /// `width / cols` lanes per column; each lane reads `unroll` contiguous `K` elements per trip,
    /// and one LDS tree folds each column's lanes. `cols` is the launch-grid knob the
    /// planner picks per shape; `width` must be a multiple of `cols`.
    Gemv { width: u32, cols: u32, unroll: u32 },
    /// `M > 1`: LDS-staged tiles (542b), for a contraction with no block-diagonal split.
    Tiled { tile: TileSize },
}

/// The tile edge of a [`Schedule::Tiled`] contraction, built only by [`TileSize::new`]. Card 658: a
/// row-tile is laid out per block ([`Schedule::grid_threads`], `contraction_tiled`'s grid derivation),
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
        }
    }
}

/// The bare edge, so a schedule's `Debug` (part of the planner's kernel key) reads `Tiled { tile: 8 }`.
impl std::fmt::Debug for TileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
