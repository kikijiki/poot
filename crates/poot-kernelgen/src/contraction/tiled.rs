//! The [`super::Schedule::RegisterTiled`] loop nest: the one `M > 1` contraction body, for every operand load.
//! A front end supplies only how the weight tile is read ([`TiledWeight`]): runs of decoded values along a
//! weight row for an `[N, K]` weight (the dense loader in every storage, the packed descriptor decode), or the
//! elements of a `[K, N]` weight, staged along `N`. An indexed contraction (each row reading its own expert's
//! weight) is the same nest over its rows grouped by expert ([`GroupedRows`]): a tile is one expert's.
//!
//! A workgroup of `lanes x lanes` lanes owns a `(lanes * rows) x (lanes * cols)` output tile, and each lane
//! accumulates a `rows x cols` register tile of it ([`MicroTile::Scalar`], which owns the dot step, the owner map
//! and the element view the store chain is written against). The registers are the point: a lane that owns one
//! output decodes one weight element and reads two LDS values per FMA, so the decode and the LDS traffic bound
//! the kernel; a lane that owns a register tile decodes each staged weight element once for all the tile's rows
//! and reads `rows + cols` LDS values per `rows * cols` FMAs. `K` is staged a round (the launch's depth) at a
//! time. The activation (and an `[N, K]` weight) is staged as runs of consecutive `K` values: a lane loads `run`
//! activation values and decodes `run` weight values per slot, so a weight run's address chain, word loads and
//! shared scale are emitted once. A `[K, N]` weight is staged in blocks of adjacent columns with adjacent lanes
//! adjacent in `N`, its coalesced direction ([`ColumnBlock`]): one value per slot into a scalar tile, a 4 x 4 block
//! transposed into a fragment tile.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, LocalDecl, Operand, Place, Rvalue,
    Statement, Terminator, Ty, WorkgroupLocalDecl,
};

use poot_graph_ir::{FragmentDtype, OperandPrecision};

use super::epilogue::BoundEpilogue;
use super::indexed::{RowExpert, emit_poison, emit_row_expert};
use super::{RegisterTiling, Schedule};
use crate::KernelGenError;
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, WeightLane, WeightLayout, elem, guard};
use crate::microtile::{MicroTile, Staged, StagedMajor, StoreAt};
use crate::precision::{ProductRead, StagedTerms, store_value, stored_ty};

/// The CFG [`tiled_body`] builds: the `GroupX`, `GroupY` and `LocalX` reads, the setup block, the pipelined `K`
/// loop (its prologue and first multiply, then per round: use, write, issue, multiply), the drain (an `Opaque`
/// fragment's spill and barrier) and the finish (the element read), and then the store chain.
const GROUP_Y: u32 = 1;
const LANE: u32 = 2;
const SETUP: u32 = 3;
const PROLOGUE: u32 = 4;
const FIRST_MULTIPLY: u32 = 5;
const USE: u32 = 6;
const WRITE: u32 = 7;
const ISSUE: u32 = 8;
const MULTIPLY: u32 = 9;
const DRAIN: u32 = 10;
const FINISH: u32 = 11;
/// The first block of the store chain ([`crate::microtile::Elements::store`]): one guard block and one store
/// block per owned accumulator element.
const STORES: u32 = 12;

/// The workgroup an owner grid of `owners` makes out of `micro` tiles, staging `depth` values of `K` per round: what
/// decides the output tile, the staged tiles and how they are staged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaunchShape {
    /// Owners along the output rows and columns (lanes for a scalar tile, waves for a fragment one).
    pub(super) owners: (usize, usize),
    pub(super) micro: MicroTile,
    /// The `K` values one barrier round stages: a whole number of `micro`'s steps.
    pub(super) depth: usize,
}

impl LaunchShape {
    const fn lane_count(self) -> usize {
        self.owners.0 * self.owners.1 * self.micro.owner_lanes()
    }

    /// The `(activation, weight)` staged tiles: the output tile rows and columns, and the stride, major and
    /// element count of each. Both are staged in the micro-tile's major, whatever the weight's global order.
    const fn tiles(self) -> [StagedTile; 2] {
        let (tile_rows, tile_cols) = self.micro.output_tile(self.owners.0, self.owners.1);
        [
            StagedTile::of(self, tile_rows),
            StagedTile::of(self, tile_cols),
        ]
    }
}

/// One operand's staged tile: `edge` output rows (activation) or columns (weight) by the round's depth.
#[derive(Clone, Copy)]
struct StagedTile {
    major: StagedMajor,
    stride: usize,
    /// Elements of one part's array.
    len: usize,
}

impl StagedTile {
    const fn of(shape: LaunchShape, edge: usize) -> Self {
        let major = shape.micro.staged_major();
        let stride = shape.micro.staged_stride(major, edge, shape.depth);
        let rows = match major {
            StagedMajor::K => shape.depth,
            StagedMajor::X => edge,
        };
        Self {
            major,
            stride,
            len: rows * stride,
        }
    }
}

/// The workgroup-local bytes one workgroup of `shape` stages its two tiles in (and spills an `Opaque` fragment
/// accumulator to).
pub(super) const fn lds_bytes(shape: LaunchShape, terms: StagedTerms) -> usize {
    let [a, b] = shape.tiles();
    let staged = a.len * terms.activation + b.len * terms.weight;
    staged * shape.micro.staged_elem_bytes()
        + shape.micro.scratch_words(shape.owners.0 * shape.owners.1) * std::mem::size_of::<f32>()
}

/// Whether a workgroup of `shape` stages both tiles in whole slots: every lane stages the same number of runs of
/// the activation tile and of an `[N, K]` weight tile (runs of `run` values), or of [`ColumnBlock`]s of a `[K, N]`
/// one whose blocks are `out_per_block` wide.
pub(super) const fn stages_evenly(
    shape: LaunchShape,
    run: usize,
    layout: WeightLayout,
    out_per_block: usize,
) -> bool {
    let (tile_rows, tile_cols) = shape.micro.output_tile(shape.owners.0, shape.owners.1);
    let depth = shape.depth;
    let lane_count = shape.lane_count();
    let weight_slots = match layout {
        WeightLayout::Nk => tile_cols * (depth / run),
        WeightLayout::Kn => {
            let block = ColumnBlock::of(shape, out_per_block);
            (tile_cols / block.cols) * (depth / block.deep)
        }
    };
    depth.is_multiple_of(run)
        && (tile_rows * (depth / run)).is_multiple_of(lane_count)
        && weight_slots.is_multiple_of(lane_count)
}

/// How one slot of a `[K, N]` weight tile is staged: `cols` adjacent columns (one vector read of a `K` row; adjacent
/// lanes adjacent in `N`, the weight's coalesced direction) by `deep` consecutive `K` values (one vector write of
/// a column's `X`-major row). A fragment tile transposes the weight as it stages it, so a fragment lane loads its
/// column's 16 depth values in two reads; four columns when the block's `out_per_block` columns are a multiple of
/// four (an edge block is then wholly in range or wholly past it), else one. A scalar tile stages `K`-major, one
/// value per slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ColumnBlock {
    cols: usize,
    deep: usize,
}

impl ColumnBlock {
    /// The vector width of a `[K, N]` block read and of an `X`-major column write: four 4-byte values, one
    /// 16-byte read, and four 2-byte staged values, one 8-byte write.
    const VECTOR: usize = 4;

    pub(super) const fn of(shape: LaunchShape, out_per_block: usize) -> Self {
        match shape.micro.staged_major() {
            StagedMajor::K => Self { cols: 1, deep: 1 },
            StagedMajor::X => Self {
                cols: if out_per_block.is_multiple_of(Self::VECTOR) {
                    Self::VECTOR
                } else {
                    1
                },
                deep: Self::VECTOR,
            },
        }
    }
}

/// The extents a tiled body's request fixes when it is generated: the contraction depth `K` and the outputs per
/// block, the first two words of its metadata ([`META_K`], [`META_OUT_PER_BLOCK`]). The body still reads both from
/// the metadata; these decide only what it emits (whole rounds, the `[K, N]` block width), and the request that
/// carries them is the kernel's key.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TiledExtents {
    pub(crate) k: usize,
    pub(crate) out_per_block: usize,
}

impl TiledExtents {
    /// The extents a tiled request's metadata names.
    pub(crate) fn of_meta(generator: &'static str, meta: &[u32]) -> Result<Self, KernelGenError> {
        match meta {
            [k, out_per_block, ..] => Ok(Self {
                k: *k as usize,
                out_per_block: *out_per_block as usize,
            }),
            _ => Err(KernelGenError::Unsupported {
                request: generator,
                reason: "the tiled metadata starts with K and the outputs per block",
            }),
        }
    }
}

/// A tiled schedule's launch shape, checked once before emission ([`stages_evenly`]).
#[derive(Clone, Copy)]
pub(crate) struct TiledLaunch {
    shape: LaunchShape,
    run: usize,
    layout: WeightLayout,
    /// How a `[K, N]` weight tile is staged per slot.
    columns: ColumnBlock,
    /// Whether `K` is a whole number of rounds: then no run, element or round of the loop reaches past `K` but the
    /// one past the last that the pipeline issues and never uses, and the body emits no tail mask or clamp.
    whole_rounds: bool,
}

impl TiledLaunch {
    /// The launch of a tiled `schedule` over a weight stored in `layout` whose operands are read at `precision`,
    /// for a request of `extents`. A fragment schedule multiplies operands narrowed to its dtype, so the precision
    /// must say that dtype.
    pub(crate) fn of(
        schedule: Schedule,
        layout: WeightLayout,
        precision: OperandPrecision,
        extents: TiledExtents,
    ) -> Result<Self, KernelGenError> {
        const GENERATOR: &str = "contraction_register_tiled";
        let Some(tiling) = RegisterTiling::of(schedule) else {
            return Err(KernelGenError::Unsupported {
                request: GENERATOR,
                reason: "an M > 1 contraction runs a tiled schedule",
            });
        };
        let (shape, run) = (tiling.launch_shape(), tiling.run());
        if let MicroTile::Fragment(tile) = shape.micro
            && precision.fragment_dtype() != Some(tile.dtype)
        {
            return Err(KernelGenError::Unsupported {
                request: GENERATOR,
                reason: "a fragment schedule needs operands narrowed or split to its dtype",
            });
        }
        let run = run.get() as usize;
        if !stages_evenly(shape, run, layout, extents.out_per_block) {
            let (tile_rows, tile_cols) = shape.micro.output_tile(shape.owners.0, shape.owners.1);
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: format!("staged slots of a {tile_rows}x{tile_cols} tile in runs of {run}"),
                value: (tile_rows + tile_cols) * shape.depth,
                divisor: shape.lane_count(),
            });
        }
        Ok(Self {
            shape,
            run,
            layout,
            columns: ColumnBlock::of(shape, extents.out_per_block),
            whole_rounds: extents.k.is_multiple_of(shape.depth),
        })
    }

    /// The consecutive `K` values a lane stages at a time.
    pub(crate) fn run(self) -> usize {
        self.run
    }

    /// Runs per row of a staged run tile: a row's depth is `depth / run` runs.
    fn runs_per_row(self) -> usize {
        self.shape.depth / self.run
    }

    /// Runs of a staged run tile of `edge` rows each lane loads per round.
    fn slots(self, edge: usize) -> usize {
        edge * self.runs_per_row() / self.shape.lane_count()
    }
}

/// The metadata words every [`tiled_body`] reads, in order. A grouped body ([`GroupedRows`]) reads its expert count
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
    /// launch's `run`, so a run is whole or wholly past the end). `step` is what the body's [`GroupedRows`] emitted
    /// for the tile's expert (empty without a grouping), and `weight_row` is then a row of that expert's `[N, K]`
    /// slab.
    Rows(&'f mut WeightRun<'f>),
    /// A `[K, N]` weight in `weight`, read through `lane`: block `b`'s `[K, out_per_block]` slab starts at element
    /// `b * K * out_per_block`, and under a [`GroupedRows`] the tile's expert's does.
    Columns { weight: Local, lane: WeightLane },
}

/// What a grouped tile emits for its `expert`: the weight state its runs read.
pub(crate) type ExpertStep<'f> = dyn FnMut(&mut Emit, Local) -> Vec<Local> + 'f;

/// The rows of an indexed contraction grouped by expert (POOT-1451, [`poot_graph_ir::ExpertGrouping`]): the grid's
/// rows are the grouping's `P` padded positions (`metadata` word [`META_M_PER_BLOCK`]) in one block, the output is
/// `[M, out_per_block]` in row order, and one output tile of rows is one tile of the grouping (the launch's tile
/// rows are the grouping's block), so all its rows read one expert's weight and the tile runs one `K` loop. Each
/// staged activation slot reads the row its position holds (a sentinel position re-reads a real row, whose output is
/// never stored), each store writes the row its position holds and skips a sentinel, and a tile whose first
/// position is the sentinel is past every bucket, so its workgroup returns before it stages anything. `metadata`
/// word `experts_word` is `E`; a tile whose expert word is outside `0..E` (the bucket of invalid ids) stores NaN in
/// each of its rows.
pub(crate) struct GroupedRows<'f> {
    /// The `I32` grouping: the padded order, then each tile's expert.
    pub(crate) groups: Local,
    /// The metadata word holding the expert count `E`.
    pub(crate) experts_word: usize,
    /// What the tile's expert needs from the weight, emitted once and handed to every [`TiledWeight::Rows`] run (a
    /// packed stack's per-source slab offsets; the slab's first row for a dense one).
    pub(crate) step: &'f mut ExpertStep<'f>,
}

/// The body's buffers: the full parameter list (`unit`, the activation, the weight sources, the grouping of an
/// indexed contraction, epilogue leaves, `metadata`, `out`), where the ones the loop nest itself reads sit, the
/// epilogue each output runs through before its store, and the expert grouping of an indexed contraction.
pub(crate) struct TiledBody<'a> {
    pub(crate) name: &'a str,
    pub(crate) params: Vec<LocalDecl>,
    pub(crate) activation: Local,
    pub(crate) metadata: Local,
    pub(crate) out: Local,
    pub(crate) epilogue: Option<BoundEpilogue<'a>>,
    pub(crate) grouped: Option<GroupedRows<'a>>,
    /// How the two staged tiles are read: the declared operand precision, applied where each value is staged.
    pub(crate) read: ProductRead,
}

/// What the setup of a grouped body ([`GroupedRows`]) knows about its tile, loop-invariant.
#[derive(Clone, Copy)]
struct GroupedTile {
    /// The grouping.
    groups: Local,
    /// The last padded position, `P - 1`: every position read is clamped to it.
    last_position: Local,
    /// The word of the tile's expert, `P + tile` (the tile clamped to the last).
    expert_at: Local,
    /// Whether the workgroup is a tile of the grid's one block, not a folded grid's padding.
    in_grid: Local,
    /// Whether the tile holds a row: it is in the grid and its first position is not the sentinel.
    occupied: Local,
}

/// The row padded position `position` of `groups` holds, as an index: the sentinel reads as the row count.
fn emit_grouped_row(e: &mut Emit, groups: Local, position: Local) -> Local {
    let word = e.let_(Ty::I32, Rvalue::Use(Operand::Copy(elem(groups, position))));
    e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(word),
        },
    )
}

/// One staged slot of one operand's tile: where its values land in LDS and where they come from.
struct StagedSlot {
    /// The slot's first `K` offset within the round.
    k_offset: Local,
    /// The LDS slot of the first value; value `j` of a run is `j` rows (`stride` slots each) further.
    lds_idx: Local,
    /// The activation row's first element (A), the weight row (`[N, K]` B), or the weight column (`[K, N]` B).
    source: Local,
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
/// With a [`GroupedRows`] the rows are the grouping's positions (see there): each stored output adds its tile's
/// poison (`0.0`, or NaN for an expert outside `0..E`) before the epilogue. The early return of an empty tile reads
/// the grouping at a tile-uniform index, so it is uniform control flow ahead of the loop's barriers.
pub(crate) fn tiled_body(
    spec: TiledBody<'_>,
    launch: TiledLaunch,
    mut weight: TiledWeight<'_>,
) -> Body {
    let TiledBody {
        name,
        params,
        activation,
        metadata,
        out,
        epilogue,
        grouped,
        read,
    } = spec;
    debug_assert_eq!(
        matches!(weight, TiledWeight::Columns { .. }),
        launch.layout == WeightLayout::Kn,
        "the weight read is the launch's layout"
    );
    let (shape, run) = (launch.shape, launch.run);
    let (owners, micro) = (shape.owners, shape.micro);
    let (edge_m, edge_n) = micro.output_tile(owners.0, owners.1);
    // One LDS row of a staged tile is padded, so the cooperative stores (adjacent lanes adjacent in the tile's
    // rows or columns, or in a row's depth) and the dot reads (adjacent lanes adjacent in a register tile, or a
    // fragment lane reading its row) spread over the banks.
    let [a_tile, b_tile] = shape.tiles();
    let (stride_a, stride_b) = (a_tile.stride, b_tile.stride);
    let lane_count = shape.lane_count();
    let depth = shape.depth;
    // The block past the store chain: its return, where an empty grouped tile also goes.
    let ret = STORES + 2 * micro.accumulators() as u32;
    let n_inputs = (params.len() - 1) as u32;
    // The staged tiles' LDS arrays: the activation's and the weight's (`hi`) parts, then the `lo` part of each
    // operand that is split ([`StagedTerms`]), then the scratch an `Opaque` fragment spills to.
    let terms = read.staged_terms();
    let scratch = micro
        .needs_spill()
        .then_some((terms.activation + terms.weight) as u8);
    let a_array = |term: usize| if term == 0 { 0 } else { 2 };
    let b_array = |term: usize| {
        if term == 0 {
            1
        } else {
            1 + terms.activation as u8
        }
    };
    let mut al = Alloc::new(params);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);
    let kt = al.add(Ty::Usize, true);
    // The weight slab a `[K, N]` tile reads: its block's, or a grouped tile's expert's.
    let b_slab = al.add(Ty::Usize, true);
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
    // `[rows, out_per_block]`. A grouped body is one block whose `m_per_block` grid rows are the grouping's
    // positions, not the output's rows.
    let out_len = setup.let_(Ty::Usize, Rvalue::Len(Place::local(out)));
    let rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(out_len),
        copy(out_per_block),
    );
    let blocks = match grouped {
        Some(_) => setup.let_(Ty::Usize, Rvalue::Use(cu(1))),
        None => bin(
            &mut setup,
            Ty::Usize,
            BinOp::Div,
            copy(rows),
            copy(m_per_block),
        ),
    };
    let weight_rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(blocks),
        copy(out_per_block),
    );
    let max_weight_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(weight_rows), cu(1));
    let max_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(rows), cu(1));
    // With a `K` tail, the last run start: `k` is a multiple of `run`, so every run is whole or wholly past the end,
    // and a run past the end re-reads the last whole one (its values are masked to zero). None with whole rounds.
    let last_run =
        (!launch.whole_rounds).then(|| bin(&mut setup, Ty::Usize, BinOp::Sub, copy(k), cu(run)));

    let ceil_div = |e: &mut Emit, value: Local, divisor: usize| {
        let padded = bin(e, Ty::Usize, BinOp::Add, copy(value), cu(divisor - 1));
        bin(e, Ty::Usize, BinOp::Div, copy(padded), cu(divisor))
    };
    let tiles_col = ceil_div(&mut setup, out_per_block, edge_n);
    let tiles_row = ceil_div(&mut setup, m_per_block, edge_m);
    let kt_count = ceil_div(&mut setup, k, depth);
    // The last round: the pipeline issues the reads of one round past it, never used, and reads the last round
    // again in its place, so no address of that round leaves `K` (one workgroup-uniform clamp, not one per value).
    let last_round = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(kt_count), cu(1));
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
    let geometry = micro.geometry(&mut setup, l, owners.0, owners.1);

    // A grouped tile: tile `tile_row` of the grouping, whose `edge_m` positions start at `row_base` (the grid's one
    // block; a folded grid's padding workgroup, past it, stores nothing and its reads are clamped in range).
    let grouped_tile = grouped.as_ref().map(|grouped| {
        let groups = grouped.groups;
        let last_position = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(m_per_block), cu(1));
        let last_tile = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(tiles_row), cu(1));
        let tile = bin(
            &mut setup,
            Ty::Usize,
            BinOp::Min,
            copy(tile_row),
            copy(last_tile),
        );
        let expert_at = bin(
            &mut setup,
            Ty::Usize,
            BinOp::Add,
            copy(m_per_block),
            copy(tile),
        );
        let in_grid = bin(&mut setup, Ty::Bool, BinOp::Eq, copy(block), cu(0));
        let head_at = bin(
            &mut setup,
            Ty::Usize,
            BinOp::Min,
            copy(row_base),
            copy(last_position),
        );
        let head = emit_grouped_row(&mut setup, groups, head_at);
        let real = bin(&mut setup, Ty::Bool, BinOp::Lt, copy(head), copy(rows));
        let occupied = bin(
            &mut setup,
            Ty::Bool,
            BinOp::BitAnd,
            copy(in_grid),
            copy(real),
        );
        GroupedTile {
            groups,
            last_position,
            expert_at,
            in_grid,
            occupied,
        }
    });
    let row_map = grouped_tile.map(|tile| (tile.groups, tile.last_position));

    // Staged run `q` of a run tile of `edge` rows: run `r = q * lane_count + l` is row `r / runs_per_row`, `K`
    // offset `(r % runs_per_row) * run` (adjacent lanes adjacent in `K`: the coalesced direction of a row-major
    // operand). `base` is the tile's first row in its block; `source_row0` and `source_stride` turn the row into
    // the buffer row the run reads, clamped to the last real one. A row past the block (or the buffer) re-reads a
    // real row: its output is never stored, and each output depends on its own row and column alone, so no edge
    // needs a mask. Under `row_map` (a grouping and its last position) the row is a padded position, and the run
    // reads the row that position holds (a sentinel clamps to the last real row like any edge row).
    let stage_runs = |setup: &mut Emit,
                      edge: usize,
                      (stride, major): (usize, StagedMajor),
                      base: Local,
                      source_row0: Local,
                      max_source_row: Local,
                      source_stride: Option<Local>,
                      row_map: Option<(Local, Local)>|
     -> Vec<StagedSlot> {
        let runs_per_row = launch.runs_per_row();
        (0..launch.slots(edge))
            .map(|q| {
                let r = bin(setup, Ty::Usize, BinOp::Add, copy(l), cu(q * lane_count));
                let k_chunk = bin(setup, Ty::Usize, BinOp::Rem, copy(r), cu(runs_per_row));
                let k_offset = bin(setup, Ty::Usize, BinOp::Mul, copy(k_chunk), cu(run));
                let tile_idx = bin(setup, Ty::Usize, BinOp::Div, copy(r), cu(runs_per_row));
                let lds_idx = staged_word(setup, major, stride, k_offset, tile_idx);

                let in_block = bin(setup, Ty::Usize, BinOp::Add, copy(base), copy(tile_idx));
                let row = bin(
                    setup,
                    Ty::Usize,
                    BinOp::Add,
                    copy(source_row0),
                    copy(in_block),
                );
                let row = match row_map {
                    Some((groups, last_position)) => {
                        let position =
                            bin(setup, Ty::Usize, BinOp::Min, copy(row), copy(last_position));
                        emit_grouped_row(setup, groups, position)
                    }
                    None => row,
                };
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
                }
            })
            .collect()
    };
    let a_runs = stage_runs(
        &mut setup,
        edge_m,
        (stride_a, a_tile.major),
        row_base,
        block_row0,
        max_row,
        Some(k),
        row_map,
    );
    let b_slots = match weight {
        TiledWeight::Rows(_) => stage_runs(
            &mut setup,
            edge_n,
            (stride_b, b_tile.major),
            col_base,
            block_weight_row0,
            max_weight_row,
            None,
            None,
        ),
        // Staged block `q` of a `[K, N]` tile ([`ColumnBlock`]: `cols` columns by `deep` `K` values): block
        // `r = q * lane_count + l` starts at column `(r % (edge_n / cols)) * cols` and `K` offset
        // `(r / (edge_n / cols)) * deep` (adjacent lanes adjacent in `N`). `source` is its first column, clamped so
        // the block ends at the block's last real column: `out_per_block` is a multiple of `cols`, so a block is
        // wholly in range or wholly past it, and one past it re-reads a real one (its outputs are never stored).
        // The slab it sits in is `b_slab`: the block's (a padding workgroup's block clamped to the last real one),
        // or under a grouping the tile's expert's (set with the expert, below).
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
            let ColumnBlock { cols, deep } = launch.columns;
            let max_col = bin(
                &mut setup,
                Ty::Usize,
                BinOp::Sub,
                copy(out_per_block),
                cu(cols),
            );
            let groups = edge_n / cols;
            (0..groups * (depth / deep) / lane_count)
                .map(|q| {
                    let setup = &mut setup;
                    let r = bin(setup, Ty::Usize, BinOp::Add, copy(l), cu(q * lane_count));
                    let group = bin(setup, Ty::Usize, BinOp::Rem, copy(r), cu(groups));
                    let col = bin(setup, Ty::Usize, BinOp::Mul, copy(group), cu(cols));
                    let k_group = bin(setup, Ty::Usize, BinOp::Div, copy(r), cu(groups));
                    let k_offset = bin(setup, Ty::Usize, BinOp::Mul, copy(k_group), cu(deep));
                    let lds_idx = staged_word(setup, b_tile.major, stride_b, k_offset, col);
                    let col = bin(setup, Ty::Usize, BinOp::Add, copy(col_base), copy(col));
                    let source = bin(setup, Ty::Usize, BinOp::Min, copy(col), copy(max_col));
                    StagedSlot {
                        k_offset,
                        lds_idx,
                        source,
                    }
                })
                .collect()
        }
    };

    // A grouped tile's expert: its weight state, the `[K, N]` slab it reads, and the poison its stores add.
    let mut step_state = Vec::new();
    let poison = grouped_tile.map(|tile| {
        let grouped = grouped.expect("a grouped tile has its grouping");
        let experts = emit_read_meta(&mut setup, metadata, grouped.experts_word);
        let RowExpert { expert, valid } =
            emit_row_expert(&mut setup, tile.groups, tile.expert_at, experts);
        step_state = (grouped.step)(&mut setup, expert);
        if matches!(weight, TiledWeight::Columns { .. }) {
            let slab = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(expert), copy(k));
            let slab = bin(
                &mut setup,
                Ty::Usize,
                BinOp::Mul,
                copy(slab),
                copy(out_per_block),
            );
            setup.assign(Place::local(b_slab), Rvalue::Use(copy(slab)));
        }
        emit_poison(&mut setup, valid)
    });

    micro.zero(&mut setup, &mut acc);
    setup.assign(Place::local(kt), Rvalue::Use(cu(0)));
    let setup_block = BasicBlock {
        statements: setup.take(),
        terminator: match grouped_tile {
            Some(tile) => guard(tile.occupied, ret, PROLOGUE),
            None => goto(PROLOGUE),
        },
    };

    // --- stage: every lane loads its activation runs and reads or decodes its weight slots of round `round`; the
    // values and the LDS slots they are written to come back, so a pipelined round writes them after its dot ---
    let mut stage_round = |load: &mut Emit, round: Local| -> Vec<StagedWrite> {
        let mut writes = Vec::new();
        let read_round = bin(load, Ty::Usize, BinOp::Min, copy(round), copy(last_round));
        let kt_base = bin(load, Ty::Usize, BinOp::Mul, copy(read_round), cu(depth));
        // A run's start. With a `K` tail it is clamped to the last whole run (a run past the end of `K` re-reads it),
        // and the weight tile is zeroed there (`k_mask`), so the activation tile needs no mask of its own; with
        // whole rounds no run reaches past `K`.
        let run_start = |load: &mut Emit, k_offset: Local| {
            let start = bin(load, Ty::Usize, BinOp::Add, copy(kt_base), copy(k_offset));
            let clamped = match last_run {
                Some(last_run) => bin(load, Ty::Usize, BinOp::Min, copy(start), copy(last_run)),
                None => start,
            };
            (start, clamped)
        };
        // With a `K` tail, 1.0 while `k_at` is inside `K`, else 0.0: the mask of a weight value at `K` offset
        // `k_at`. None with whole rounds.
        let k_mask = |load: &mut Emit, k_at: Local| {
            last_run.is_some().then(|| {
                let in_range = bin(load, Ty::Bool, BinOp::Lt, copy(k_at), copy(k));
                load.let_(
                    Ty::F32,
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: copy(in_range),
                    },
                )
            })
        };
        let masked = |load: &mut Emit, value: Local, mask: Option<Local>| match mask {
            Some(mask) => bin(load, Ty::F32, BinOp::Mul, copy(value), copy(mask)),
            None => value,
        };
        for staged in &a_runs {
            let (_, start) = run_start(load, staged.k_offset);
            let first = bin(
                load,
                Ty::Usize,
                BinOp::Add,
                copy(staged.source),
                copy(start),
            );
            for j in 0..run {
                let idx = bin(load, Ty::Usize, BinOp::Add, copy(first), cu(j));
                let value = read_activation(load, read.activation_stored, activation, idx);
                let slot = bin(
                    load,
                    Ty::Usize,
                    BinOp::Add,
                    copy(staged.lds_idx),
                    cu(j * a_tile.major.k_step(stride_a)),
                );
                for (term, part) in read.activation_terms(load, value).into_iter().enumerate() {
                    let part = micro.stage_value(load, part);
                    writes.push((a_array(term), slot, part));
                }
            }
        }
        match &mut weight {
            TiledWeight::Rows(weight_run) => {
                for staged in &b_slots {
                    let (unclamped, start) = run_start(load, staged.k_offset);
                    let mask = k_mask(load, unclamped);
                    let values =
                        weight_run(load, &step_state, staged.source, start, run, k, weight_rows);
                    for (j, &decoded) in values.iter().enumerate() {
                        let slot = bin(
                            load,
                            Ty::Usize,
                            BinOp::Add,
                            copy(staged.lds_idx),
                            cu(j * b_tile.major.k_step(stride_b)),
                        );
                        for (term, part) in read.weight_terms(load, decoded).into_iter().enumerate()
                        {
                            let value = masked(load, part, mask);
                            let value = micro.stage_value(load, value);
                            writes.push((b_array(term), slot, value));
                        }
                    }
                }
            }
            // A block's `deep` rows of `K`, each a read of its `cols` adjacent columns; value `(kk, c)` lands at
            // `K` offset `k_offset + kk` of column `col + c` of the staged tile.
            TiledWeight::Columns { weight, lane } => {
                let ColumnBlock { cols, deep } = launch.columns;
                let max_k = last_run
                    .is_some()
                    .then(|| bin(load, Ty::Usize, BinOp::Sub, copy(k), cu(1)));
                for staged in &b_slots {
                    for kk in 0..deep {
                        let k_at = bin(
                            load,
                            Ty::Usize,
                            BinOp::Add,
                            copy(kt_base),
                            copy(staged.k_offset),
                        );
                        let k_at = bin(load, Ty::Usize, BinOp::Add, copy(k_at), cu(kk));
                        let mask = k_mask(load, k_at);
                        let k_read = match max_k {
                            Some(max_k) => {
                                bin(load, Ty::Usize, BinOp::Min, copy(k_at), copy(max_k))
                            }
                            None => k_at,
                        };
                        let row = bin(
                            load,
                            Ty::Usize,
                            BinOp::Mul,
                            copy(k_read),
                            copy(out_per_block),
                        );
                        let in_slab =
                            bin(load, Ty::Usize, BinOp::Add, copy(row), copy(staged.source));
                        let first = bin(load, Ty::Usize, BinOp::Add, copy(b_slab), copy(in_slab));
                        for c in 0..cols {
                            let idx = bin(load, Ty::Usize, BinOp::Add, copy(first), cu(c));
                            let raw = lane.emit_elem(load, read.conversions, *weight, idx);
                            let at = kk * b_tile.major.k_step(stride_b)
                                + c * b_tile.major.x_step(stride_b);
                            let slot =
                                bin(load, Ty::Usize, BinOp::Add, copy(staged.lds_idx), cu(at));
                            for (term, part) in read.weight_terms(load, raw).into_iter().enumerate()
                            {
                                let value = masked(load, part, mask);
                                let value = micro.stage_value(load, value);
                                writes.push((b_array(term), slot, value));
                            }
                        }
                    }
                }
            }
        }
        writes
    };
    let write_staged = |e: &mut Emit, writes: &[StagedWrite]| {
        for &(array, slot, value) in writes {
            e.write_lds(array, copy(slot), copy(value));
        }
    };

    // --- dot: the round's unrolled micro-tile steps over the two staged tiles ---
    let staged_a = |term: usize| Staged {
        array: a_array(term),
        stride: stride_a,
        major: a_tile.major,
    };
    let staged_b = |term: usize| Staged {
        array: b_array(term),
        stride: stride_b,
        major: b_tile.major,
    };
    let (a_parts, b_parts): (Vec<usize>, Vec<usize>) =
        ((0..terms.activation).collect(), (0..terms.weight).collect());
    let mut emit_dot = |dot: &mut Emit| {
        for k in (0..depth).step_by(micro.kappa()) {
            // A split operand adds a product per pair of parts, `a` parts outermost: the oracle's order.
            for (a, b) in a_parts
                .iter()
                .flat_map(|&a| b_parts.iter().map(move |&b| (a, b)))
            {
                micro.step(dot, &geometry, &mut acc, staged_a(a), staged_b(b), k);
            }
        }
    };

    // --- the rounds: a software-pipelined loop ---
    //
    // A round's staging is split into the part that issues its loads (and the address arithmetic they read) and the
    // part that turns the loaded values into staged ones (decode, rounding, narrowing) ([`split_reads`]). Round
    // `kt + 1`'s loads are issued, a scheduling boundary keeps them ahead of the multiplies, round `kt` is multiplied
    // from the tiles, and only then are the loaded values used, at the top of the loop header (the use block), a
    // block with two predecessors (the first multiply and the back edge): no compiler pass sinks a load into it or
    // hoists its arithmetic above the multiplies, so the loads are in flight during them whatever the target's
    // scheduler does with the boundary.
    //
    // The prologue stages round 0 and makes it visible; the first multiply issues round 1 and multiplies round 0.
    // Per round: the use block turns round `kt + 1`'s loaded values into staged ones and waits for every wave to
    // have read round `kt`; the write block stores round `kt + 1` and steps `kt` to it, and a barrier makes it
    // visible; the issue block issues the next round's loads; the multiply block multiplies round `kt` and leaves
    // when no next round exists. Two barriers per round. The last round's loads re-read the clamped tail, which
    // every round may read, and are not used.
    let loop_done = DRAIN;
    // The round whose loads are in flight: `kt + 1`, set where they are issued.
    let ahead = al.add(Ty::Usize, true);
    let mut staged_e = Emit::new(&mut al);
    let writes = stage_round(&mut staged_e, ahead);
    let (issue, consume) = split_reads(staged_e.take());
    let issue: Vec<Statement> = std::iter::once(Statement::Assign(
        Place::local(ahead),
        Rvalue::BinaryOp(BinOp::Add, copy(kt), cu(1)),
    ))
    .chain(issue)
    .collect();
    // A scheduling boundary that keeps the issued loads ahead, round `kt`'s multiplies, and the way out while no
    // round `ahead` exists.
    let mut multiply = |al: &mut Alloc| {
        let mut e = Emit::new(al);
        e.push(Statement::SchedulingBoundary);
        emit_dot(&mut e);
        let more = bin(&mut e, Ty::Bool, BinOp::Lt, copy(ahead), copy(kt_count));
        BasicBlock {
            statements: e.take(),
            terminator: guard(more, loop_done, USE),
        }
    };
    let mut prologue_e = Emit::new(&mut al);
    let first = stage_round(&mut prologue_e, kt);
    write_staged(&mut prologue_e, &first);
    let prologue = BasicBlock {
        statements: prologue_e.take(),
        terminator: barrier(FIRST_MULTIPLY),
    };
    let mut first_multiply = multiply(&mut al);
    first_multiply
        .statements
        .splice(0..0, issue.iter().cloned());
    let use_block = BasicBlock {
        statements: consume,
        terminator: barrier(WRITE),
    };
    let mut write_e = Emit::new(&mut al);
    write_staged(&mut write_e, &writes);
    write_e.assign(Place::local(kt), Rvalue::Use(copy(ahead)));
    let write_block = BasicBlock {
        statements: write_e.take(),
        terminator: barrier(ISSUE),
    };
    let issue_block = BasicBlock {
        statements: issue,
        terminator: goto(MULTIPLY),
    };
    let multiply_block = multiply(&mut al);

    // --- drain and finish: an `Opaque` fragment accumulator spills to workgroup memory and, once every wave has,
    // is read back element by element; any other accumulator is read in place ---
    let mut drain = Emit::new(&mut al);
    micro.spill(&mut drain, &geometry, &acc, scratch);
    let drain = BasicBlock {
        statements: drain.take(),
        terminator: if micro.needs_spill() {
            barrier(FINISH)
        } else {
            goto(FINISH)
        },
    };
    let mut finish = Emit::new(&mut al);
    let elements = micro.elements(&mut finish, &geometry, acc, scratch);
    let finish = BasicBlock {
        statements: finish.take(),
        terminator: goto(STORES),
    };

    // --- stores: one guarded store per accumulator element, through the row's poison under a walk and the
    // epilogue if any; a row or column past the block, or a block past the last (a folded grid's padding
    // workgroups), stores nothing ---
    let mut blocks_out = vec![
        thread_index(gx, IndexAxis::GroupX, GROUP_Y),
        thread_index(gy, IndexAxis::GroupY, LANE),
        thread_index(l, IndexAxis::LocalX, SETUP),
        setup_block,
        prologue,
        first_multiply,
        use_block,
        write_block,
        issue_block,
        multiply_block,
        drain,
        finish,
    ];
    debug_assert_eq!(blocks_out.len() as u32, STORES);
    let stores = elements.store(
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
            // A grouped position stores the row it holds; a sentinel's row is `rows`, out of range. A padding
            // workgroup past the grid's one block stores nothing.
            let (row, in_grid) = match grouped_tile {
                Some(tile) => {
                    let position = bin(
                        cond,
                        Ty::Usize,
                        BinOp::Min,
                        copy(row),
                        copy(tile.last_position),
                    );
                    (
                        emit_grouped_row(cond, tile.groups, position),
                        Some(tile.in_grid),
                    )
                }
                None => (row, None),
            };
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
            let ok = match in_grid {
                Some(in_grid) => bin(cond, Ty::Bool, BinOp::BitAnd, copy(ok), copy(in_grid)),
                None => ok,
            };
            StoreAt { ok, row, col }
        },
        |store, _row, out_idx, sum| {
            let sum = match poison {
                Some(poison) => bin(store, Ty::F32, BinOp::Add, copy(sum), copy(poison)),
                None => sum,
            };
            match epilogue {
                Some(epilogue) => {
                    let value = epilogue.emit(store, read.conversions, out_idx, sum);
                    store_value(store, epilogue.epilogue.store(), value)
                }
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

    let mut body = Body::new(name, n_inputs, al.locals, blocks_out);
    body.workgroup_size = [lane_count as u32, 1, 1];
    // Array order: activation, weight, then the activation's `lo` part and the weight's, when split, then the
    // f32 scratch of an `Opaque` fragment.
    let arrays = [
        (a_tile.len, 1),
        (b_tile.len, 1),
        (a_tile.len, terms.activation - 1),
        (b_tile.len, terms.weight - 1),
    ];
    body.workgroup_locals = arrays
        .into_iter()
        .flat_map(|(len, count)| std::iter::repeat_n(len, count))
        .map(|len| WorkgroupLocalDecl {
            elem_ty: micro.staged_ty(),
            len: len as u32,
        })
        .collect();
    let scratch_words = micro.scratch_words(owners.0 * owners.1);
    if scratch.is_some() {
        body.workgroup_locals.push(WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: scratch_words as u32,
        });
    }
    body
}

/// One value a round stages: the LDS array, the slot in it and the value written there.
type StagedWrite = (u8, Local, Local);

/// `statements` (one straight-line run) split into the part that issues its memory reads and the part that uses
/// what they read, each in its original order: the first holds every statement that reads no loaded value (the
/// address arithmetic and the reads themselves), the second everything that does (decode, rounding,
/// narrowing). Running the first part, other statements, then the second computes what `statements` computes: a
/// statement moves ahead of the second part only if it is a plain assignment of a local with no side effect,
/// reads no local the second part defines, and defines no local an earlier statement of the second part reads or
/// defines.
pub(crate) fn split_reads(statements: Vec<Statement>) -> (Vec<Statement>, Vec<Statement>) {
    use std::collections::HashSet;
    let (mut issue, mut consume) = (Vec::new(), Vec::new());
    // Locals holding a loaded value or one computed from it; locals the second part reads or defines.
    let mut loaded: HashSet<Local> = HashSet::new();
    let mut kept: HashSet<Local> = HashSet::new();
    for statement in statements {
        let mut reads_loaded = false;
        statement.for_each_read(|l| reads_loaded |= loaded.contains(&l));
        let mut redefines_kept = false;
        statement.for_each_def(|l| redefines_kept |= kept.contains(&l));
        let read = match &statement {
            Statement::Assign(place, rvalue) if place.projection.is_empty() => reads_memory(rvalue),
            _ => None,
        };
        match read {
            Some(loads) if !reads_loaded && !redefines_kept => {
                if loads {
                    statement.for_each_def(|l| {
                        loaded.insert(l);
                    });
                }
                issue.push(statement);
            }
            _ => {
                statement.for_each_def(|l| {
                    loaded.insert(l);
                    kept.insert(l);
                });
                statement.for_each_read(|l| {
                    kept.insert(l);
                });
                consume.push(statement);
            }
        }
    }
    (issue, consume)
}

/// The 32-bit registers the reads of `statements` hold until the rest of them runs (POOT-1505): split as a pipelined
/// round's staging is ([`split_reads`]), every local the issued part defines and the rest reads stays live across the
/// round's multiply. Each counts by its width ([`register_words`]); `locals` declares them.
pub(crate) fn held_registers(statements: Vec<Statement>, locals: &[LocalDecl]) -> u32 {
    use std::collections::HashSet;
    let (issue, consume) = split_reads(statements);
    // Only a read's value is held: the address arithmetic the issued part also computes is rebuilt from the round's
    // few base registers wherever the compiler needs the register, which a loaded value cannot be.
    let mut defined: HashSet<Local> = HashSet::new();
    for statement in &issue {
        if let Statement::Assign(place, rvalue) = statement
            && reads_memory(rvalue) == Some(true)
        {
            defined.insert(place.local);
        }
    }
    let mut held: HashSet<Local> = HashSet::new();
    for statement in &consume {
        statement.for_each_read(|l| {
            if defined.contains(&l) {
                held.insert(l);
            }
        });
    }
    held.into_iter()
        .map(|l| register_words(&locals[l.index as usize].ty))
        .sum()
}

/// The 32-bit registers a value of `ty` occupies: one for a scalar of at most 32 bits (a 16-bit value takes a whole
/// register), two for a 64-bit one, a vector's lanes packed, nothing for a type no register holds.
fn register_words(ty: &Ty) -> u32 {
    match ty {
        Ty::Bool | Ty::Usize | Ty::I32 | Ty::U32 | Ty::F32 | Ty::F16 | Ty::BF16 => 1,
        Ty::F64 => 2,
        Ty::Vec { elem, lanes } => match elem.as_ref() {
            Ty::F16 | Ty::BF16 => lanes.div_ceil(2),
            elem => lanes * register_words(elem),
        },
        _ => 0,
    }
}

/// Whether `rvalue` reads memory (a buffer or private array element, a vector of them, a constant table word or
/// a workgroup-local element): `Some(true)`, `Some(false)` for one that reads only its operands, `None` for one
/// with a side effect (an atomic), which never moves.
fn reads_memory(rvalue: &Rvalue) -> Option<bool> {
    match rvalue {
        Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) => {
            Some(!place.projection.is_empty())
        }
        Rvalue::VectorLoad { .. }
        | Rvalue::ConstantTableRead { .. }
        | Rvalue::WorkgroupLocalRead { .. } => Some(true),
        Rvalue::WorkgroupLocalAtomic { .. }
        | Rvalue::GlobalAtomic { .. }
        | Rvalue::WorkgroupLocalCompareExchange { .. }
        | Rvalue::GlobalCompareExchange { .. } => None,
        _ => Some(false),
    }
}

/// The word of element `(k, x)` of a staged tile of `stride`: `k` and `x` are run-time locals ([`StagedMajor`]).
fn staged_word(e: &mut Emit, major: StagedMajor, stride: usize, k: Local, x: Local) -> Local {
    let (outer, inner) = match major {
        StagedMajor::K => (k, x),
        StagedMajor::X => (x, k),
    };
    let row = bin(e, Ty::Usize, BinOp::Mul, copy(outer), cu(stride));
    bin(e, Ty::Usize, BinOp::Add, copy(row), copy(inner))
}

/// Activation element `idx` as the f32 value the products read: the f32 element, or the element of the fragment
/// dtype the planner stored the activation in (POOT-1448), widened exactly.
fn read_activation(
    e: &mut Emit,
    stored: Option<FragmentDtype>,
    activation: Local,
    idx: Local,
) -> Local {
    let raw = e.let_(
        stored_ty(stored),
        Rvalue::Use(Operand::Copy(elem(activation, idx))),
    );
    match stored {
        None => raw,
        Some(_) => e.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(raw),
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_kernel_ir::{AtomicOp, Constant, ProjectionElem};

    fn l(index: u32) -> Local {
        Local { index }
    }

    fn set(local: u32, rvalue: Rvalue) -> Statement {
        Statement::Assign(Place::local(l(local)), rvalue)
    }

    fn read(buffer: u32, index: u32) -> Rvalue {
        Rvalue::Use(Operand::Copy(Place {
            local: l(buffer),
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(index))],
        }))
    }

    fn add(a: u32, b: Operand) -> Rvalue {
        Rvalue::BinaryOp(BinOp::Add, copy(l(a)), b)
    }

    /// The split moves the address arithmetic and the reads ahead and keeps every use of a read value, in order:
    /// a value computed from a read stays, and so does a statement that would move ahead of a kept statement whose
    /// operand it overwrites (`_12`), and an atomic, which never moves. Mutation: drop the `redefines_kept` test
    /// in `split_reads`; the overwrite of `_12` moves ahead of the statement that reads it and this row goes red.
    #[test]
    fn a_split_moves_the_reads_ahead_and_keeps_their_uses() {
        // `_1` a buffer; `_10`.. locals.
        let one = Operand::Const(Constant::Usize(1));
        let statements = vec![
            set(10, Rvalue::Use(one.clone())), // address
            set(11, read(1, 10)),              // read
            set(12, add(11, one.clone())),     // use of the read
            set(13, add(10, one.clone())),     // address
            set(14, read(1, 13)),              // read
            set(15, add(12, copy(l(14)))),     // use
            set(12, Rvalue::Use(one.clone())), // overwrites an operand a kept statement read
            set(
                16,
                Rvalue::GlobalAtomic {
                    place: Place::local(l(1)),
                    value: one.clone(),
                    op: AtomicOp::Add,
                },
            ),
        ];
        let (issue, consume) = split_reads(statements.clone());
        let at = |indexes: &[usize]| {
            indexes
                .iter()
                .map(|&i| statements[i].clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(issue, at(&[0, 1, 3, 4]));
        assert_eq!(consume, at(&[2, 5, 6, 7]));
    }

    /// POOT-1505: the registers a round's reads hold are the read values its decode uses, by width: two reads of an
    /// f32 and a four-lane f32 vector that the decode reads hold six, an issued read nobody reads after the split
    /// and the address arithmetic the decode also reads (rebuilt where it is needed) hold none. Mutation: count every
    /// local the issued part defines (drop the `reads_memory` test in `held_registers`); the address `_10` is counted
    /// and the row reads 7.
    #[test]
    fn a_round_holds_the_reads_its_decode_uses() {
        let one = Operand::Const(Constant::Usize(1));
        let mut locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            };
            20
        ];
        for (index, ty) in [
            (10, Ty::Usize),
            (11, Ty::F32),
            (12, Ty::F32),
            (13, Ty::F32),
            (14, Ty::F32),
            (15, Ty::F32),
            (17, Ty::F32),
        ] {
            locals[index].ty = ty;
        }
        locals[16].ty = Ty::Vec {
            elem: Box::new(Ty::F32),
            lanes: 4,
        };
        let statements = vec![
            set(10, Rvalue::Use(one.clone())), // address, read by the decode
            set(11, read(1, 10)),              // read, decoded
            set(12, read(1, 10)),              // read, never used
            set(16, read(2, 10)),              // vector read, decoded
            set(13, read(1, 10)),              // read, decoded
            set(14, add(11, copy(l(13)))),     // decode
            set(15, add(10, copy(l(14)))),     // decode reads the address
            set(17, add(16, one.clone())),     // decode of the vector
        ];
        assert_eq!(held_registers(statements, &locals), 1 + 4 + 1);
    }
}
