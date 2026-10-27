//! The [`super::Schedule::RegisterTiled`] loop nest: the one `M > 1` contraction body, for every operand load.
//! A front end supplies only how the weight tile is read ([`TiledWeight`]): runs of decoded values along a
//! weight row for an `[N, K]` weight (the dense loader in every storage, the packed descriptor decode), or the
//! elements of a `[K, N]` weight, staged along `N`. An indexed contraction (each row reading its own expert's
//! weight) is the same nest inside an [`ExpertWalk`] over the experts its tile's rows select.
//!
//! A workgroup of `lanes x lanes` lanes owns a `(lanes * rows) x (lanes * cols)` output tile, and each lane
//! accumulates a `rows x cols` register tile of it ([`MicroTile::Scalar`], which owns the dot step, the owner map
//! and the element view the store chain is written against). The registers are the point: a lane that owns one
//! output decodes one weight element and reads two LDS values per FMA, so the decode and the LDS traffic bound
//! the kernel; a lane that owns a register tile decodes each staged weight element once for all the tile's rows
//! and reads `rows + cols` LDS values per `rows * cols` FMAs. `K` is staged `BLOCK_K` at a time. The activation
//! (and an `[N, K]` weight) is staged as runs of consecutive `K` values: a lane loads `run` activation values and
//! decodes `run` weight values per slot, so a weight run's address chain, word loads and shared scale are
//! emitted once. A `[K, N]` weight is staged one element per slot with adjacent lanes adjacent in `N`, its
//! coalesced direction.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, LocalDecl, Operand, Place, Rvalue,
    Terminator, Ty, WorkgroupLocalDecl,
};

use super::epilogue::BoundEpilogue;
use super::indexed::{RowExpert, emit_poison, emit_row_expert, emit_row_key};
use super::{RegisterTiling, RunLength, TileSize};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu, emit_read_meta, select_u32};
use crate::helpers::{Alloc, WeightLane, WeightLayout, elem, guard};
use crate::microtile::{LaneRows, MicroCols, MicroTile, Staged, StoreAt};

/// `K` elements staged in LDS per barrier round, and run in unrolled outer-product steps.
pub(super) const BLOCK_K: usize = 16;

/// The CFG [`tiled_body`] builds: the `GroupX`, `GroupY` and `LocalX` reads, the setup block, the K loop (guard,
/// staged load, dot, latch) and then the store chain.
const GROUP_Y: u32 = 1;
const LANE: u32 = 2;
const SETUP: u32 = 3;
const LOOP_GUARD: u32 = 4;
const STAGE: u32 = 5;
const DOT: u32 = 6;
const LATCH: u32 = 7;
/// The first block of the store chain ([`crate::microtile::Elements::store`]): one guard block and one store
/// block per register-tile element.
const STORES: u32 = 8;

/// The workgroup-local bytes one workgroup of `lanes x lanes` lanes of `micro` tiles stages its two tiles in.
pub(super) const fn lds_bytes(lanes: usize, micro: MicroTile) -> usize {
    let (tile_rows, tile_cols) = micro.output_tile(lanes, lanes);
    BLOCK_K * (tile_rows + 1 + tile_cols + 1) * std::mem::size_of::<f32>()
}

/// Whether a workgroup of `lanes x lanes` lanes of `micro` tiles stages both tiles in whole slots: every lane
/// stages the same number of runs of the activation tile and of an `[N, K]` weight tile (runs of `run` values),
/// or of single values of a `[K, N]` one.
pub(super) const fn stages_evenly(
    lanes: usize,
    micro: MicroTile,
    run: usize,
    layout: WeightLayout,
) -> bool {
    let (tile_rows, tile_cols) = micro.output_tile(lanes, lanes);
    let lane_count = lanes * lanes;
    let weight_slots = match layout {
        WeightLayout::Nk => tile_cols * (BLOCK_K / run),
        WeightLayout::Kn => tile_cols * BLOCK_K,
    };
    (tile_rows * (BLOCK_K / run)).is_multiple_of(lane_count)
        && weight_slots.is_multiple_of(lane_count)
}

/// A [`super::Schedule::RegisterTiled`]'s launch shape, checked once before emission ([`stages_evenly`]).
#[derive(Clone, Copy)]
pub(crate) struct TiledLaunch {
    lanes: usize,
    micro: MicroTile,
    run: usize,
    layout: WeightLayout,
}

impl TiledLaunch {
    pub(crate) fn new(
        lanes: TileSize,
        rows: LaneRows,
        cols: MicroCols,
        run: RunLength,
        layout: WeightLayout,
    ) -> Result<Self, KernelGenError> {
        let micro = RegisterTiling::micro_tile_of(rows, cols);
        let (lanes, run) = (lanes.get() as usize, run.get() as usize);
        if !stages_evenly(lanes, micro, run, layout) {
            let (tile_rows, tile_cols) = micro.output_tile(lanes, lanes);
            return Err(KernelGenError::NotDivisible {
                generator: "contraction_register_tiled",
                dim: format!("staged slots of a {tile_rows}x{tile_cols} tile in runs of {run}"),
                value: (tile_rows + tile_cols) * BLOCK_K,
                divisor: lanes * lanes,
            });
        }
        Ok(Self {
            lanes,
            micro,
            run,
            layout,
        })
    }

    /// Runs per row of a staged run tile: a row's `BLOCK_K` values are `BLOCK_K / run` runs.
    fn runs_per_row(self) -> usize {
        BLOCK_K / self.run
    }

    /// Runs of a staged run tile of `edge` rows each lane loads per round.
    fn slots(self, edge: usize) -> usize {
        edge * self.runs_per_row() / (self.lanes * self.lanes)
    }
}

/// The metadata words every [`tiled_body`] reads, in order. An expert walk ([`ExpertWalk`]) reads its expert count
/// from a word of its own past these.
pub(crate) const META_K: usize = 0;
pub(crate) const META_OUT_PER_BLOCK: usize = 1;
pub(crate) const META_M_PER_BLOCK: usize = 2;
pub(crate) const META_X_GROUPS: usize = 3;
pub(crate) const META_TILE_OFFSET: usize = 4;

/// The read of one run of an `[N, K]` weight row's values, as [`TiledWeight::Rows`] describes it.
pub(crate) type WeightRun<'f> =
    dyn FnMut(&mut Emit, &[Local], Local, Local, usize, Local, Local) -> Vec<Local> + 'f;

/// How [`tiled_body`] reads the weight tile, which must be the launch's [`WeightLayout`].
pub(crate) enum TiledWeight<'f> {
    /// An `[N, K]` weight: `run(e, step, weight_row, k0, count, k, weight_rows)` emits the `count` logical f32
    /// weight values `k0 .. k0 + count` of row `weight_row` of a weight with `weight_rows` rows of `k` values, `k0` a
    /// multiple of `count`; it is only ever called with the whole run inside `0..k` (`k` is a multiple of the
    /// launch's `run`, so a run is whole or wholly past the end). `step` is what the body's [`ExpertWalk`] emitted
    /// for the current expert (empty without a walk), and `weight_row` is then a row of that expert's `[N, K]`
    /// slab.
    Rows(&'f mut WeightRun<'f>),
    /// A `[K, N]` weight in `weight`, read through `lane`: block `b`'s `[K, out_per_block]` slab starts at element
    /// `b * K * out_per_block`, and under an [`ExpertWalk`] the current expert's does.
    Columns { weight: Local, lane: WeightLane },
}

/// What one key of an [`ExpertWalk`] emits for its `expert`: the weight state its runs read.
pub(crate) type WalkStep<'f> = dyn FnMut(&mut Emit, Local) -> Vec<Local> + 'f;

/// The segmentation of an indexed contraction: output row `m` reads the weight of expert `ids[m]`, so one tile's
/// rows may select several experts. The body then walks the distinct expert keys among its tile's rows in
/// ascending order ([`emit_row_key`]: a row's expert, or `E` for a row past the end or with an invalid id), and per
/// key runs the whole `K` loop over that expert's weight tile with the activation rows of every other key staged
/// as exact zeros, so a row's sum is its own expert's product. The accumulators carry across the walk: a row adds
/// exact zero products on every other key's pass. The body has one block (`blocks == 1`); `metadata` word
/// `experts_word` is `E`.
pub(crate) struct ExpertWalk<'f> {
    /// The `I32` expert id of each output row.
    pub(crate) ids: Local,
    /// The metadata word holding the expert count `E`.
    pub(crate) experts_word: usize,
    /// What one key of the walk needs from the weight, emitted once per step for that `expert` and handed to
    /// every [`TiledWeight::Rows`] run of the step (a packed stack's per-source slab offsets; the slab's first row
    /// for a dense one).
    pub(crate) step: &'f mut WalkStep<'f>,
}

/// The body's buffers: the full parameter list (`unit`, the activation, the weight sources, the `ids` of a walk,
/// epilogue leaves, `metadata`, `out`), where the ones the loop nest itself reads sit, the epilogue each output runs
/// through before its store, and the expert walk of an indexed contraction.
pub(crate) struct TiledBody<'a> {
    pub(crate) name: &'a str,
    pub(crate) params: Vec<LocalDecl>,
    pub(crate) activation: Local,
    pub(crate) metadata: Local,
    pub(crate) out: Local,
    pub(crate) epilogue: Option<BoundEpilogue<'a>>,
    pub(crate) walk: Option<ExpertWalk<'a>>,
}

/// One staged slot of one operand's tile: where its values land in LDS and where they come from.
struct StagedSlot {
    /// The slot's first `K` offset within the round.
    k_offset: Local,
    /// The LDS slot of the first value; value `j` of a run is `j` rows (`stride` slots each) further.
    lds_idx: Local,
    /// The activation row's first element (A), the weight row (`[N, K]` B), or the weight column (`[K, N]` B).
    source: Local,
    /// The buffer row the slot stands for before the clamp: past the end for a slot of an edge tile's missing
    /// rows (an expert walk keys the activation rows by it).
    row: Local,
}

/// The [`super::Schedule::RegisterTiled`] loop nest, shared by every operand load.
///
/// `metadata` is `[K, out_per_block, m_per_block, x_groups, tile_offset]` ([`META_K`]..); `activation` is `[rows,
/// K]` and `out` is `[rows, out_per_block]` with `rows = blocks * m_per_block` (block-diagonal: block `b`'s rows read
/// weight rows `b * out_per_block ..` of an `[N, K]` weight, or the `b`th slab of a `[K, N]` one). A flat tile
/// index `g = tile_offset + GroupY * x_groups + GroupX` (`tile_offset` is the first tile of the range this dispatch
/// covers: a contraction over the device's watchdog budget or tiled-GEMM workgroup ceiling is several dispatches
/// over disjoint tile ranges, 0 when unsplit) is laid out block-major, so one workgroup's staged weight tile is
/// always one block's, and every edge read is clamped in range, never branched on inside the barrier'd loop: a
/// row or column past the block re-reads a real one (its output is never stored, and an output depends on its
/// own row and column alone), and a `K` tail re-reads the last run or element, which the weight tile zeroes. A
/// workgroup past the last tile of a folded grid reads the last real block and never stores.
///
/// The `K` tail is zeroed on the weight tile only, so an activation `Inf`/`NaN` in the last real run reaches the
/// tail slots as `Inf * 0 = NaN`; that output was already non-finite for any nonzero weight there.
///
/// With an [`ExpertWalk`] the `K` loop runs once per distinct expert key among the tile's rows (see there), and
/// each stored output adds its row's poison (`0.0`, or NaN for an id outside `0..E`) before the epilogue. The walk
/// reads the tile's ids at tile-uniform indexes, so it is uniform control flow and the loop's barriers stay legal
/// inside it.
pub(crate) fn tiled_body(
    spec: TiledBody<'_>,
    launch: TiledLaunch,
    weight: TiledWeight<'_>,
) -> Body {
    let TiledBody {
        name,
        params,
        activation,
        metadata,
        out,
        epilogue,
        mut walk,
    } = spec;
    debug_assert_eq!(
        matches!(weight, TiledWeight::Columns { .. }),
        launch.layout == WeightLayout::Kn,
        "the weight read is the launch's layout"
    );
    let (lanes, micro, run) = (launch.lanes, launch.micro, launch.run);
    let (edge_m, edge_n) = micro.output_tile(lanes, lanes);
    // One LDS row of a staged tile is padded by one word, so the cooperative stores (adjacent lanes adjacent in
    // the tile's rows or columns) and the dot reads (adjacent lanes adjacent in a register tile) both spread
    // over the banks.
    let (stride_a, stride_b) = (edge_m + 1, edge_n + 1);
    let lane_count = lanes * lanes;
    // The block indexes past the store chain: its return, then the walk's own blocks.
    let ret = STORES + 2 * micro.accumulators() as u32;
    let (walk_head, walk_step, walk_next) = (ret + 1, ret + 2, ret + 3);
    let n_inputs = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);
    let kt = al.add(Ty::Usize, true);
    // The weight slab a `[K, N]` tile reads: its block's, or the walk's current expert's.
    let b_slab = al.add(Ty::Usize, true);
    // The walk's current key and the least key it has not taken yet.
    let next = al.add(Ty::Usize, true);
    let lo = al.add(Ty::Usize, true);
    let mut acc = micro.accumulator(&mut al);

    let thread_index = |destination: Local, dim: IndexAxis, target: u32| BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(destination),
            dim,
            target: BlockId { index: target },
        },
    };
    let goto = |target: u32| Terminator::Goto {
        target: BlockId { index: target },
    };
    let barrier = |target: u32| Terminator::Barrier {
        target: BlockId { index: target },
    };

    // --- setup: tile coordinates, the loop-invariant parts of every staged slot, the accumulators ---
    let mut setup = Emit::new(&mut al);
    let k = emit_read_meta(&mut setup, metadata, META_K);
    let out_per_block = emit_read_meta(&mut setup, metadata, META_OUT_PER_BLOCK);
    let m_per_block = emit_read_meta(&mut setup, metadata, META_M_PER_BLOCK);
    let x_groups = emit_read_meta(&mut setup, metadata, META_X_GROUPS);
    let g = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(gy), copy(x_groups));
    let g = bin(&mut setup, Ty::Usize, BinOp::Add, copy(g), copy(gx));
    let tile_offset = emit_read_meta(&mut setup, metadata, META_TILE_OFFSET);
    let g = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(g),
        copy(tile_offset),
    );

    // `rows` and `blocks` come from `out`'s own length, never a metadata word: `out` is exactly
    // `[rows, out_per_block]`.
    let out_len = setup.let_(Ty::Usize, Rvalue::Len(Place::local(out)));
    let rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(out_len),
        copy(out_per_block),
    );
    let blocks = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(rows),
        copy(m_per_block),
    );
    let weight_rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(blocks),
        copy(out_per_block),
    );
    let max_weight_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(weight_rows), cu(1));
    let max_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(rows), cu(1));
    // The last run start: `k` is a multiple of `run`, so every run is whole or wholly past the end, and a run past
    // the end re-reads the last whole one (its values are masked to zero).
    let last_run = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(k), cu(run));

    let ceil_div = |e: &mut Emit, value: Local, divisor: usize| {
        let padded = bin(e, Ty::Usize, BinOp::Add, copy(value), cu(divisor - 1));
        bin(e, Ty::Usize, BinOp::Div, copy(padded), cu(divisor))
    };
    let tiles_col = ceil_div(&mut setup, out_per_block, edge_n);
    let tiles_row = ceil_div(&mut setup, m_per_block, edge_m);
    let kt_count = ceil_div(&mut setup, k, BLOCK_K);
    let tiles_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tiles_row),
        copy(tiles_col),
    );
    let block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(g),
        copy(tiles_per_block),
    );
    let tile_in_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(g),
        copy(tiles_per_block),
    );
    let tile_row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let tile_col = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tile_row),
        cu(edge_m),
    );
    let col_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tile_col),
        cu(edge_n),
    );
    let block_row0 = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(block),
        copy(m_per_block),
    );
    let block_weight_row0 = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(block),
        copy(out_per_block),
    );

    // The owner map: which tile row and column each of this lane's accumulator elements is.
    let geometry = micro.geometry(&mut setup, l, lanes, lanes);

    // Staged run `q` of a run tile of `edge` rows: run `r = q * lane_count + l` is row `r / runs_per_row`, `K`
    // offset `(r % runs_per_row) * run` (adjacent lanes adjacent in `K`: the coalesced direction of a row-major
    // operand). `base` is the tile's first row in its block; `source_row0` and `source_stride` turn the row into
    // the buffer row the run reads, clamped to the last real one. A row past the block (or the buffer) re-reads a
    // real row: its output is never stored, and each output depends on its own row and column alone, so no edge
    // needs a mask.
    let stage_runs = |setup: &mut Emit,
                      edge: usize,
                      stride: usize,
                      base: Local,
                      source_row0: Local,
                      max_source_row: Local,
                      source_stride: Option<Local>|
     -> Vec<StagedSlot> {
        let runs_per_row = launch.runs_per_row();
        (0..launch.slots(edge))
            .map(|q| {
                let r = bin(setup, Ty::Usize, BinOp::Add, copy(l), cu(q * lane_count));
                let k_chunk = bin(setup, Ty::Usize, BinOp::Rem, copy(r), cu(runs_per_row));
                let k_offset = bin(setup, Ty::Usize, BinOp::Mul, copy(k_chunk), cu(run));
                let tile_idx = bin(setup, Ty::Usize, BinOp::Div, copy(r), cu(runs_per_row));
                let lds_row = bin(setup, Ty::Usize, BinOp::Mul, copy(k_offset), cu(stride));
                let lds_idx = bin(setup, Ty::Usize, BinOp::Add, copy(lds_row), copy(tile_idx));

                let in_block = bin(setup, Ty::Usize, BinOp::Add, copy(base), copy(tile_idx));
                let row = bin(
                    setup,
                    Ty::Usize,
                    BinOp::Add,
                    copy(source_row0),
                    copy(in_block),
                );
                let source_row = bin(
                    setup,
                    Ty::Usize,
                    BinOp::Min,
                    copy(row),
                    copy(max_source_row),
                );
                let source = match source_stride {
                    Some(stride) => {
                        bin(setup, Ty::Usize, BinOp::Mul, copy(source_row), copy(stride))
                    }
                    None => source_row,
                };
                StagedSlot {
                    k_offset,
                    lds_idx,
                    source,
                    row,
                }
            })
            .collect()
    };
    let a_runs = stage_runs(
        &mut setup,
        edge_m,
        stride_a,
        row_base,
        block_row0,
        max_row,
        Some(k),
    );
    let b_slots = match weight {
        TiledWeight::Rows(_) => stage_runs(
            &mut setup,
            edge_n,
            stride_b,
            col_base,
            block_weight_row0,
            max_weight_row,
            None,
        ),
        // Staged value `q` of a `[K, N]` tile: value `r = q * lane_count + l` is column `r % edge_n`, `K` offset
        // `r / edge_n` (adjacent lanes adjacent in `N`). `source` is the column, clamped to the block's last real
        // one; the slab it sits in is `b_slab`: the block's (a padding workgroup's block clamped to the last real
        // one), or under a walk the step's expert's.
        TiledWeight::Columns { .. } => {
            let max_block = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(blocks), cu(1));
            let real_block = bin(
                &mut setup,
                Ty::Usize,
                BinOp::Min,
                copy(block),
                copy(max_block),
            );
            let slab = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(real_block), copy(k));
            let slab = bin(
                &mut setup,
                Ty::Usize,
                BinOp::Mul,
                copy(slab),
                copy(out_per_block),
            );
            setup.assign(Place::local(b_slab), Rvalue::Use(copy(slab)));
            let max_col = bin(
                &mut setup,
                Ty::Usize,
                BinOp::Sub,
                copy(out_per_block),
                cu(1),
            );
            (0..edge_n * BLOCK_K / lane_count)
                .map(|q| {
                    let setup = &mut setup;
                    let r = bin(setup, Ty::Usize, BinOp::Add, copy(l), cu(q * lane_count));
                    let col = bin(setup, Ty::Usize, BinOp::Rem, copy(r), cu(edge_n));
                    let k_offset = bin(setup, Ty::Usize, BinOp::Div, copy(r), cu(edge_n));
                    let lds_row = bin(setup, Ty::Usize, BinOp::Mul, copy(k_offset), cu(stride_b));
                    let lds_idx = bin(setup, Ty::Usize, BinOp::Add, copy(lds_row), copy(col));
                    let col = bin(setup, Ty::Usize, BinOp::Add, copy(col_base), copy(col));
                    let source = bin(setup, Ty::Usize, BinOp::Min, copy(col), copy(max_col));
                    StagedSlot {
                        k_offset,
                        lds_idx,
                        source,
                        row: col,
                    }
                })
                .collect()
        }
    };

    // The walk's per-slot keys (each staged activation row's expert, or `E`), loop-invariant.
    let walk_keys = walk.as_ref().map(|walk| {
        let experts = emit_read_meta(&mut setup, metadata, walk.experts_word);
        let keys: Vec<Local> = a_runs
            .iter()
            .map(|slot| emit_row_key(&mut setup, walk.ids, slot.row, (rows, experts)))
            .collect();
        (experts, keys)
    });

    micro.zero(&mut setup, &mut acc);
    let setup_block = match walk_keys {
        Some(_) => {
            setup.assign(Place::local(lo), Rvalue::Use(cu(0)));
            BasicBlock {
                statements: setup.take(),
                terminator: goto(walk_head),
            }
        }
        None => {
            setup.assign(Place::local(kt), Rvalue::Use(cu(0)));
            BasicBlock {
                statements: setup.take(),
                terminator: goto(LOOP_GUARD),
            }
        }
    };

    // --- the walk: the least key at or above `lo` among the tile's rows, read from `ids` at tile-uniform indexes;
    // `E` when none is left. Then one step: the expert's weight state, the `K` loop from 0, and `lo` past it ---
    let mut walk_blocks = Vec::new();
    let mut step_state = Vec::new();
    if let (Some(walk), Some((experts, _))) = (walk.as_mut(), walk_keys.as_ref()) {
        let mut head = Emit::new(&mut al);
        let mut least = head.let_(Ty::Usize, Rvalue::Use(copy(*experts)));
        for r in 0..edge_m {
            let tile_row = bin(&mut head, Ty::Usize, BinOp::Add, copy(row_base), cu(r));
            let key = emit_row_key(&mut head, walk.ids, tile_row, (rows, *experts));
            let at_or_above = bin(&mut head, Ty::Bool, BinOp::Ge, copy(key), copy(lo));
            let above = head.let_(
                Ty::Usize,
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(at_or_above),
                },
            );
            let distance = bin(&mut head, Ty::Usize, BinOp::Sub, copy(*experts), copy(key));
            let pulled = bin(
                &mut head,
                Ty::Usize,
                BinOp::Mul,
                copy(above),
                copy(distance),
            );
            let candidate = bin(
                &mut head,
                Ty::Usize,
                BinOp::Sub,
                copy(*experts),
                copy(pulled),
            );
            least = bin(
                &mut head,
                Ty::Usize,
                BinOp::Min,
                copy(least),
                copy(candidate),
            );
        }
        head.assign(Place::local(next), Rvalue::Use(copy(least)));
        let more = bin(&mut head, Ty::Bool, BinOp::Lt, copy(next), copy(*experts));
        walk_blocks.push(BasicBlock {
            statements: head.take(),
            terminator: guard(more, STORES, walk_step),
        });

        let mut step = Emit::new(&mut al);
        step_state = (walk.step)(&mut step, next);
        if matches!(weight, TiledWeight::Columns { .. }) {
            let slab = bin(&mut step, Ty::Usize, BinOp::Mul, copy(next), copy(k));
            let slab = bin(
                &mut step,
                Ty::Usize,
                BinOp::Mul,
                copy(slab),
                copy(out_per_block),
            );
            step.assign(Place::local(b_slab), Rvalue::Use(copy(slab)));
        }
        step.assign(Place::local(kt), Rvalue::Use(cu(0)));
        walk_blocks.push(BasicBlock {
            statements: step.take(),
            terminator: goto(LOOP_GUARD),
        });

        let mut after = Emit::new(&mut al);
        let lo_next = bin(&mut after, Ty::Usize, BinOp::Add, copy(next), cu(1));
        after.assign(Place::local(lo), Rvalue::Use(copy(lo_next)));
        walk_blocks.push(BasicBlock {
            statements: after.take(),
            terminator: goto(walk_head),
        });
    }

    let mut guard_e = Emit::new(&mut al);
    let more = bin(&mut guard_e, Ty::Bool, BinOp::Lt, copy(kt), copy(kt_count));
    let loop_done = if walk_keys.is_some() {
        walk_next
    } else {
        STORES
    };
    let loop_guard = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(more, loop_done, STAGE),
    };

    // --- stage: every lane loads its activation runs and reads or decodes its weight slots into LDS ---
    let mut load = Emit::new(&mut al);
    let kt_base = bin(&mut load, Ty::Usize, BinOp::Mul, copy(kt), cu(BLOCK_K));
    // A run's start, clamped to the last whole run: a run past the end of `K` re-reads it. The weight tile is
    // zeroed there (`k_mask`), so the activation tile needs no mask of its own.
    let run_start = |load: &mut Emit, k_offset: Local| {
        let start = bin(load, Ty::Usize, BinOp::Add, copy(kt_base), copy(k_offset));
        let clamped = bin(load, Ty::Usize, BinOp::Min, copy(start), copy(last_run));
        (start, clamped)
    };
    // 1.0 while `k_at` is inside `K`, else 0.0: the mask of a weight value at `K` offset `k_at`.
    let k_mask = |load: &mut Emit, k_at: Local| {
        let in_range = bin(load, Ty::Bool, BinOp::Lt, copy(k_at), copy(k));
        load.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(in_range),
            },
        )
    };
    for (slot, staged) in a_runs.iter().enumerate() {
        let (_, start) = run_start(&mut load, staged.k_offset);
        let first = bin(
            &mut load,
            Ty::Usize,
            BinOp::Add,
            copy(staged.source),
            copy(start),
        );
        // Under a walk, a row of another key stages exact zeros: a bit select, so a non-finite activation of
        // another expert's row never reaches this step's products.
        let mine = walk_keys
            .as_ref()
            .map(|(_, keys)| bin(&mut load, Ty::Bool, BinOp::Eq, copy(keys[slot]), copy(next)));
        for j in 0..run {
            let idx = bin(&mut load, Ty::Usize, BinOp::Add, copy(first), cu(j));
            let raw = load.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(activation, idx))));
            let value = match mine {
                Some(mine) => zero_unless(&mut load, mine, raw),
                None => raw,
            };
            let slot = bin(
                &mut load,
                Ty::Usize,
                BinOp::Add,
                copy(staged.lds_idx),
                cu(j * stride_a),
            );
            load.write_lds(0, copy(slot), copy(value));
        }
    }
    match weight {
        TiledWeight::Rows(weight_run) => {
            for staged in &b_slots {
                let (unclamped, start) = run_start(&mut load, staged.k_offset);
                let mask = k_mask(&mut load, unclamped);
                let values = weight_run(
                    &mut load,
                    &step_state,
                    staged.source,
                    start,
                    run,
                    k,
                    weight_rows,
                );
                for (j, &decoded) in values.iter().enumerate() {
                    let value = bin(&mut load, Ty::F32, BinOp::Mul, copy(decoded), copy(mask));
                    let slot = bin(
                        &mut load,
                        Ty::Usize,
                        BinOp::Add,
                        copy(staged.lds_idx),
                        cu(j * stride_b),
                    );
                    load.write_lds(1, copy(slot), copy(value));
                }
            }
        }
        TiledWeight::Columns { weight, lane } => {
            let max_k = bin(&mut load, Ty::Usize, BinOp::Sub, copy(k), cu(1));
            for staged in &b_slots {
                let k_at = bin(
                    &mut load,
                    Ty::Usize,
                    BinOp::Add,
                    copy(kt_base),
                    copy(staged.k_offset),
                );
                let mask = k_mask(&mut load, k_at);
                let k_read = bin(&mut load, Ty::Usize, BinOp::Min, copy(k_at), copy(max_k));
                let row = bin(
                    &mut load,
                    Ty::Usize,
                    BinOp::Mul,
                    copy(k_read),
                    copy(out_per_block),
                );
                let in_slab = bin(
                    &mut load,
                    Ty::Usize,
                    BinOp::Add,
                    copy(row),
                    copy(staged.source),
                );
                let idx = bin(
                    &mut load,
                    Ty::Usize,
                    BinOp::Add,
                    copy(b_slab),
                    copy(in_slab),
                );
                let raw = lane.emit_elem(&mut load, weight, idx);
                let value = bin(&mut load, Ty::F32, BinOp::Mul, copy(raw), copy(mask));
                load.write_lds(1, copy(staged.lds_idx), copy(value));
            }
        }
    }
    let stage = BasicBlock {
        statements: load.take(),
        terminator: barrier(DOT),
    };

    // --- dot: `BLOCK_K` unrolled micro-tile steps over the two staged tiles ---
    let (staged_a, staged_b) = (
        Staged {
            array: 0,
            stride: stride_a,
        },
        Staged {
            array: 1,
            stride: stride_b,
        },
    );
    let mut dot = Emit::new(&mut al);
    for k in 0..BLOCK_K {
        micro.step(&mut dot, &geometry, &mut acc, staged_a, staged_b, k);
    }
    let dot = BasicBlock {
        statements: dot.take(),
        terminator: goto(LATCH),
    };

    let mut latch = Emit::new(&mut al);
    let kt_next = bin(&mut latch, Ty::Usize, BinOp::Add, copy(kt), cu(1));
    latch.assign(Place::local(kt), Rvalue::Use(copy(kt_next)));
    let latch = BasicBlock {
        statements: latch.take(),
        terminator: barrier(LOOP_GUARD),
    };

    // --- stores: one guarded store per register-tile element, through the row's poison under a walk and the
    // epilogue if any; a row or column past the block, or a block past the last (a folded grid's padding
    // workgroups), stores nothing ---
    let mut blocks_out = vec![
        thread_index(gx, IndexAxis::GroupX, GROUP_Y),
        thread_index(gy, IndexAxis::GroupY, LANE),
        thread_index(l, IndexAxis::LocalX, SETUP),
        setup_block,
        loop_guard,
        stage,
        dot,
        latch,
    ];
    debug_assert_eq!(blocks_out.len() as u32, STORES);
    let poison_of = walk
        .as_ref()
        .zip(walk_keys.as_ref())
        .map(|(walk, (experts, _))| (walk.ids, *experts));
    let stores = micro.elements(&geometry, acc).store(
        &mut al,
        STORES,
        out,
        out_per_block,
        |cond, at| {
            let row_in_block = bin(cond, Ty::Usize, BinOp::Add, copy(row_base), copy(at.row));
            let col = bin(cond, Ty::Usize, BinOp::Add, copy(col_base), copy(at.col));
            let row = bin(
                cond,
                Ty::Usize,
                BinOp::Add,
                copy(block_row0),
                copy(row_in_block),
            );
            let row_ok = bin(
                cond,
                Ty::Bool,
                BinOp::Lt,
                copy(row_in_block),
                copy(m_per_block),
            );
            let col_ok = bin(cond, Ty::Bool, BinOp::Lt, copy(col), copy(out_per_block));
            let in_range = bin(cond, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
            let tile_ok = bin(cond, Ty::Bool, BinOp::BitAnd, copy(row_ok), copy(col_ok));
            let ok = bin(cond, Ty::Bool, BinOp::BitAnd, copy(tile_ok), copy(in_range));
            StoreAt { ok, row, col }
        },
        |store, row, out_idx, sum| {
            let sum = match poison_of {
                Some((ids, experts)) => {
                    let RowExpert { valid, .. } = emit_row_expert(store, ids, row, experts);
                    let poison = emit_poison(store, valid);
                    bin(store, Ty::F32, BinOp::Add, copy(sum), copy(poison))
                }
                None => sum,
            };
            match epilogue {
                Some(epilogue) => epilogue.emit(store, out_idx, sum),
                None => sum,
            }
        },
    );
    blocks_out.extend(stores);
    debug_assert_eq!(blocks_out.len() as u32, ret);
    blocks_out.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    blocks_out.extend(walk_blocks);

    let mut body = Body::new(name, n_inputs, al.locals, blocks_out);
    body.workgroup_size = [lane_count as u32, 1, 1];
    body.workgroup_locals = [stride_a, stride_b]
        .into_iter()
        .map(|stride| WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (BLOCK_K * stride) as u32,
        })
        .collect();
    body
}

/// `value`, or exactly `0.0` unless `keep`: a bit-pattern select, so a masked value never multiplies into a sum
/// whatever it holds.
fn zero_unless(e: &mut Emit, keep: Local, value: Local) -> Local {
    let bits = e.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: copy(value),
        },
    );
    let zero = e.let_(Ty::U32, Rvalue::Use(c32(0)));
    let kept = select_u32(e, keep, bits, zero);
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(kept),
        },
    )
}
