//! The indexed contraction: `out[m, n] = sum over k of x[m, k] * W[ids[m], k, n]` over a stacked `[E, K, N]`
//! weight, where output row `m` reads the expert slab `ids[m]` (the MoE expert matmul, and every per-row
//! adapter). Row selection is a property of the contraction's index map, not of a loader: both schedules read
//! the expert through [`emit_row_expert`], and the schedule only decides the loop order.
//!
//! - [`Schedule::Gemv`] (decode, few rows): the shared Gemv loop nest ([`gemv_body`]) with an indexed run
//!   read. Flat output column `m * N + n` is the "weight row"; its run read takes the expert slab of row `m`
//!   and the `[K, N]` weight at stride `N`.
//! - [`Schedule::Tiled`] (prefill, many rows) is *segmented*: the rows of one tile may select different
//!   experts, so the tile walks the distinct expert ids among its rows in ascending order. Each step stages that
//!   expert's weight tile once through LDS and the rows that selected it take the sum, so the weight tile is
//!   shared by every row of an expert-sorted tile (the whole gain of grouping) and the result is correct for
//!   any id order.
//!
//! A [`WeightLayout::Nk`] weight is the same stack with each expert stored `[N, K]` (the checkpoint order): the
//! slab stride is still `K * N` and only the in-slab index swaps, so no transpose of the experts ever runs.
//!
//! An expert id outside `0..E`, or not an integer (NaN included), is not read: its output row is NaN on every
//! schedule (R-632-4), and the slab it would have addressed is clamped in range so no read leaves the buffer.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, LocalDecl, MathOp, Operand,
    Place, Rvalue, Statement, Terminator, Ty, WorkgroupLocalDecl,
};

use poot_target::BufferStorage;

use super::epilogue::{Epilogue, emit_epilogue};
use super::gemv::{GemvEpilogue, GemvLaunch, gemv_body};
use super::{Schedule, TileSize};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu, emit_read_meta, select_u32};
use crate::helpers::{
    Alloc, WeightLane, WeightLayout, elem, guard, ld, local, slice_dtype, slice_f32,
};

/// The launch metadata words of an indexed contraction, in order: `K`, `N`, `E` (the expert count), `M`
/// (the activation rows) and `x_groups` (the Tiled grid's X extent; the Gemv ignores it).
pub(crate) const META_K: usize = 0;
pub(crate) const META_N: usize = 1;
pub(crate) const META_EXPERTS: usize = 2;
pub(crate) const META_ROWS: usize = 3;
pub(crate) const META_X_GROUPS: usize = 4;
/// The first word past the shared layout: a packed table's per-source slab strides start here.
pub(crate) const META_PACKED_SLAB_WORDS: usize = 5;

/// The IEEE-754 quiet NaN the fail-closed rows carry.
const NAN_BITS: u32 = 0x7FC0_0000;

/// The expert row `m`'s id names, checked.
pub(crate) struct RowExpert {
    /// The expert, clamped into `0..E` so an invalid id never addresses outside the weight.
    pub(crate) expert: Local,
    /// Whether the id is an integer in `0..E`.
    pub(crate) valid: Local,
}

/// What row `m`'s expert id selects: the slab it reads, and whether it may.
struct RowSlab {
    /// `expert * stride`: the element offset of the slab in the stacked weight.
    base: Local,
    /// `0.0` for a valid id and NaN otherwise; added to the row's sum so an invalid row's output is NaN.
    poison: Local,
}

/// `ids[row]` as an expert: valid when it is an integer in `0..experts` (ordered comparisons, so NaN fails each),
/// with the expert taken from the id clamped after the cast, so whatever the cast of a non-finite or negative id
/// yields, the expert stays inside the table.
pub(crate) fn emit_row_expert(e: &mut Emit, ids: Local, row: Local, experts: Local) -> RowExpert {
    let id = e.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(ids, row))));
    let experts_f = e.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(experts),
        },
    );
    let non_negative = bin(
        e,
        Ty::Bool,
        BinOp::Ge,
        copy(id),
        Operand::Const(Constant::F32(0.0)),
    );
    let in_range = bin(e, Ty::Bool, BinOp::Lt, copy(id), copy(experts_f));
    let floor = e.let_(Ty::F32, Rvalue::MathUnary(MathOp::Floor, copy(id)));
    let integral = bin(e, Ty::Bool, BinOp::Eq, copy(floor), copy(id));
    let bounded = bin(
        e,
        Ty::Bool,
        BinOp::BitAnd,
        copy(non_negative),
        copy(in_range),
    );
    let valid = bin(e, Ty::Bool, BinOp::BitAnd, copy(bounded), copy(integral));
    let raw = e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(id),
        },
    );
    let last = bin(e, Ty::Usize, BinOp::Sub, copy(experts), cu(1));
    let expert = bin(e, Ty::Usize, BinOp::Min, copy(raw), copy(last));
    RowExpert { expert, valid }
}

/// `0.0` for a valid expert id and NaN otherwise: added to the first term of a row's sum so an invalid row's
/// output is NaN (R-632-4), the one definition of the fail-closed row every indexed schedule shares.
pub(crate) fn emit_poison(e: &mut Emit, valid: Local) -> Local {
    let clean = e.let_(Ty::U32, Rvalue::Use(c32(0)));
    let nan = e.let_(Ty::U32, Rvalue::Use(c32(NAN_BITS)));
    let bits = select_u32(e, valid, clean, nan);
    e.let_(
        Ty::F32,
        Rvalue::Bitcast {
            to: Ty::F32,
            operand: copy(bits),
        },
    )
}

/// The one place a row's expert id is turned into a slab: [`emit_row_expert`] plus the slab base and the NaN
/// poison of an invalid id.
fn emit_row_slab(e: &mut Emit, ids: Local, row: Local, experts: Local, stride: Local) -> RowSlab {
    let RowExpert { expert, valid } = emit_row_expert(e, ids, row, experts);
    let base = bin(e, Ty::Usize, BinOp::Mul, copy(expert), copy(stride));
    let poison = emit_poison(e, valid);
    RowSlab { base, poison }
}

/// The key a row walks under: its expert when `row` is a real row (`< rows`) with a valid id, else `experts`,
/// which no step selects. Reads `ids[min(row, rows - 1)]`, a read at an index every lane of a tile computes alike
/// when `row` is tile-uniform.
fn emit_row_key(e: &mut Emit, ids: Local, row: Local, (rows, experts): (Local, Local)) -> Local {
    let last_row = bin(e, Ty::Usize, BinOp::Sub, copy(rows), cu(1));
    let row_c = bin(e, Ty::Usize, BinOp::Min, copy(row), copy(last_row));
    let row_ok = bin(e, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
    let RowExpert { expert, valid } = emit_row_expert(e, ids, row_c, experts);
    let real = bin(e, Ty::Bool, BinOp::BitAnd, copy(row_ok), copy(valid));
    let real_u = e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(real),
        },
    );
    let gap = bin(e, Ty::Usize, BinOp::Sub, copy(experts), copy(expert));
    let pull = bin(e, Ty::Usize, BinOp::Mul, copy(real_u), copy(gap));
    bin(e, Ty::Usize, BinOp::Sub, copy(experts), copy(pull))
}

/// The element offset of weight element `(k, n)` inside one expert slab: `k * N + n` for a [`WeightLayout::Kn`]
/// slab and `n * K + k` for an [`WeightLayout::Nk`] one (the checkpoint's `[N, K]` expert, read in place). The
/// slab stride is `K * N` either way.
fn emit_in_slab(
    e: &mut Emit,
    layout: WeightLayout,
    (k_at, n_at): (Local, Local),
    (k, n): (Local, Local),
) -> Local {
    let (major, extent, minor) = match layout {
        WeightLayout::Kn => (k_at, n, n_at),
        WeightLayout::Nk => (n_at, k, k_at),
    };
    let row = bin(e, Ty::Usize, BinOp::Mul, copy(major), copy(extent));
    bin(e, Ty::Usize, BinOp::Add, copy(row), copy(minor))
}

/// The body of an indexed contraction over f32 activation and output and a stacked weight held in `weight`'s
/// storage (f32, or BF16 packed two per `u32` word and widened in-register: no f32 copy of an expert stack ever
/// exists) under `schedule`, with the weight's experts read as `layout` says, params `(x, w, ids, epilogue leaves..,
/// metadata,
/// out)`; `ids` is f32 and `metadata` is `[K, N, E, M, x_groups]` ([`META_K`]..). `epilogue` (f32) runs over each
/// output element's sum before its store.
pub(crate) fn indexed_contraction(
    name: &str,
    schedule: Schedule,
    layout: WeightLayout,
    weight: BufferStorage,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let lane = match WeightLane::of("indexed_contraction", weight)? {
        WeightLane::F16Packed => {
            return Err(KernelGenError::Unsupported {
                request: "indexed_contraction",
                reason: "an indexed weight is read as f32 or bf16-packed storage only",
            });
        }
        lane => lane,
    };
    if let Some(epilogue) = epilogue {
        epilogue.require_f32("indexed_contraction")?;
    }
    match schedule {
        Schedule::Gemv {
            width,
            cols,
            unroll,
        } => indexed_gemv(
            name,
            GemvLaunch::new(width, cols, unroll)?,
            layout,
            lane,
            epilogue,
        ),
        Schedule::Tiled { tile } => indexed_tiled(name, tile, layout, lane, epilogue),
    }
}

/// The indexed Gemv: [`gemv_body`] with the run read of one flat output column `m * N + n`, which takes row
/// `m`'s expert slab and reads `W[slab + k * N + n]`.
fn indexed_gemv(
    name: &str,
    launch: GemvLaunch,
    layout: WeightLayout,
    lane: WeightLane,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("indexed_contraction")?,
        None => Vec::new(),
    };
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        lane.param(),
        ld(slice_f32(false), false),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let al = Alloc::new(params);
    let (activation, weight, ids) = (local(1), local(2), local(3));
    let leaves: Vec<Local> = (0..leaf_count).map(|j| local(4 + j as u32)).collect();
    let (metadata, out) = (local(4 + leaf_count as u32), local(5 + leaf_count as u32));
    let read_run =
        |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, _out_total: Local| {
            let n = emit_read_meta(e, metadata, META_N);
            let experts = emit_read_meta(e, metadata, META_EXPERTS);
            let m = bin(e, Ty::Usize, BinOp::Div, copy(row), copy(n));
            let col = bin(e, Ty::Usize, BinOp::Rem, copy(row), copy(n));
            let stride = bin(e, Ty::Usize, BinOp::Mul, copy(k), copy(n));
            let slab = emit_row_slab(e, ids, m, experts, stride);
            (0..count)
                .map(|j| {
                    let k_at = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                    let in_slab = emit_in_slab(e, layout, (k_at, col), (k, n));
                    let at = bin(e, Ty::Usize, BinOp::Add, copy(slab.base), copy(in_slab));
                    let value = lane.emit_elem(e, weight, at);
                    if j == 0 {
                        bin(e, Ty::F32, BinOp::Add, copy(value), copy(slab.poison))
                    } else {
                        value
                    }
                })
                .collect()
        };
    let epilogue = epilogue.map(|epilogue| GemvEpilogue {
        epilogue,
        leaves: &leaves,
        lanes: &lanes,
    });
    Ok(gemv_body(
        name,
        al,
        4 + leaf_count as u32 + 1,
        [activation, metadata, out],
        launch,
        epilogue,
        read_run,
    ))
}

// Block indices of the segmented Tiled body (block 0 reads `GroupX`).
const BB_GROUP_Y: u32 = 1;
const BB_LANE: u32 = 2;
const BB_SETUP: u32 = 3;
const BB_WALK: u32 = 4;
const BB_SLAB: u32 = 5;
const BB_K_HEADER: u32 = 6;
const BB_LOAD: u32 = 7;
const BB_DOT: u32 = 8;
const BB_LATCH: u32 = 9;
const BB_AFTER: u32 = 10;
const BB_TAKE: u32 = 11;
const BB_STORE_GUARD: u32 = 12;
const BB_STORE: u32 = 13;
const BB_RETURN: u32 = 14;

/// The segmented Tiled indexed contraction: one `tile x tile` workgroup owns a block of output rows and
/// columns, one output element per lane, on the grid [`Schedule::grid_threads`] sizes (folded over `GroupY`
/// above the device's X extent, `x_groups` in the metadata).
///
/// Each row has a key: its expert, or `E` for an invalid or past-the-end row ([`emit_row_key`]). Every lane
/// walks the distinct keys among the tile's rows in ascending order: the next key is the least one at or above
/// `lo`, computed from the tile's ids read straight from `ids` at tile-uniform indexes, so the walk is uniform
/// control flow and the K loop's barriers are legal inside it (the verifier proves it: an LDS read is never
/// uniform). A step stages that expert's activation and weight tiles per K sub-tile (a branchless masked load,
/// never an `if` around a barrier), runs the unrolled `tile`-wide dot into a step accumulator, and the rows whose
/// key is the step's expert keep it.
/// A row takes a step only for its own expert, so its sum is exactly the sequential `K` sum of its expert.
fn indexed_tiled(
    name: &str,
    tile: TileSize,
    layout: WeightLayout,
    lane: WeightLane,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("indexed_contraction")?,
        None => Vec::new(),
    };
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        lane.param(),
        ld(slice_f32(false), false),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let leaves: Vec<Local> = (0..leaf_count).map(|j| local(4 + j as u32)).collect();
    Ok(indexed_tiled_body(
        IndexedTiledBody {
            name,
            tile,
            params,
            x: local(1),
            ids: local(3),
            metadata: local(4 + leaf_count as u32),
            out: local(5 + leaf_count as u32),
            epilogue: epilogue.map(|epilogue| GemvEpilogue {
                epilogue,
                leaves: &leaves,
                lanes: &lanes,
            }),
        },
        &DenseExperts {
            layout,
            lane,
            weight: local(2),
        },
    ))
}

/// A stacked dense `[E, K, N]` weight (or `[E, N, K]`, [`WeightLayout`]) as the segmented Tiled body reads it.
struct DenseExperts {
    layout: WeightLayout,
    lane: WeightLane,
    weight: Local,
}

impl IndexedTiledWeight for DenseExperts {
    /// The element offset of the expert's slab: `expert * K * N`.
    fn step(&self, e: &mut Emit, expert: Local, (k, n): (Local, Local)) -> Vec<Local> {
        let stride = bin(e, Ty::Usize, BinOp::Mul, copy(k), copy(n));
        vec![bin(e, Ty::Usize, BinOp::Mul, copy(expert), copy(stride))]
    }

    fn load(
        &self,
        e: &mut Emit,
        step: &[Local],
        at: (Local, Local),
        extents: (Local, Local),
    ) -> Local {
        let in_slab = emit_in_slab(e, self.layout, at, extents);
        let at = bin(e, Ty::Usize, BinOp::Add, copy(step[0]), copy(in_slab));
        self.lane.emit_elem(e, self.weight, at)
    }
}

/// How the segmented Tiled body reads one expert's weight, the only part that differs between the dense stack
/// and a packed one: everything else (the key walk, the staging, the dot, the poison and the epilogue) is
/// [`indexed_tiled_body`].
pub(crate) trait IndexedTiledWeight {
    /// What the walk computes once per step for the step's `expert` (its slab offset in each buffer), given the
    /// weight's `(K, N)`; handed back to every [`Self::load`] of the step.
    fn step(&self, e: &mut Emit, expert: Local, extents: (Local, Local)) -> Vec<Local>;
    /// Element `(k, n)` (`at`, both inside the weight) of the step's expert, for a weight of `extents` `(K, N)`.
    fn load(
        &self,
        e: &mut Emit,
        step: &[Local],
        at: (Local, Local),
        extents: (Local, Local),
    ) -> Local;
}

/// What [`indexed_tiled_body`] needs besides the weight read: the body's parameters (`unit, x, .., metadata,
/// out`, with the `ids` and any epilogue leaves among them) and where each of those sits.
pub(crate) struct IndexedTiledBody<'a> {
    pub(crate) name: &'a str,
    pub(crate) tile: TileSize,
    pub(crate) params: Vec<LocalDecl>,
    pub(crate) x: Local,
    pub(crate) ids: Local,
    pub(crate) metadata: Local,
    pub(crate) out: Local,
    pub(crate) epilogue: Option<GemvEpilogue<'a>>,
}

/// The segmented Tiled body, over whichever stacked weight `weight` reads (see [`indexed_tiled`]).
pub(crate) fn indexed_tiled_body(
    spec: IndexedTiledBody<'_>,
    weight: &impl IndexedTiledWeight,
) -> Body {
    let IndexedTiledBody {
        name,
        tile,
        params,
        x,
        ids,
        metadata,
        out,
        epilogue,
    } = spec;
    let ts = tile.get() as usize;
    let n_inputs = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);
    let acc = al.add(Ty::F32, true);
    let part = al.add(Ty::F32, true);
    let kt = al.add(Ty::Usize, true);
    let lo = al.add(Ty::Usize, true);

    let goto = |index: u32| Terminator::Goto {
        target: BlockId { index },
    };
    let barrier = |index: u32| Terminator::Barrier {
        target: BlockId { index },
    };
    let load = |e: &mut Emit, buffer: Local, at: Local| -> Local {
        e.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(buffer, at))))
    };
    // `value` or exactly 0.0, by `keep`: a bit-pattern select, so a masked-off lane never multiplies a
    // non-finite clamped read into the tile.
    let masked = |e: &mut Emit, keep: Local, value: Local| -> Local {
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
    };

    let bb_group_x = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: BB_GROUP_Y },
        },
    };
    let bb_group_y = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: BB_LANE },
        },
    };
    let bb_lane = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(l),
            dim: IndexAxis::LocalX,
            target: BlockId { index: BB_SETUP },
        },
    };

    // Setup: tile geometry, this lane's row check and key, the key's LDS slot.
    let mut s = Emit::new(&mut al);
    let k = emit_read_meta(&mut s, metadata, META_K);
    let n = emit_read_meta(&mut s, metadata, META_N);
    let experts = emit_read_meta(&mut s, metadata, META_EXPERTS);
    let rows = emit_read_meta(&mut s, metadata, META_ROWS);
    let x_groups = emit_read_meta(&mut s, metadata, META_X_GROUPS);
    let g_base = bin(&mut s, Ty::Usize, BinOp::Mul, copy(gy), copy(x_groups));
    let g = bin(&mut s, Ty::Usize, BinOp::Add, copy(g_base), copy(gx));
    let cols_padded = bin(&mut s, Ty::Usize, BinOp::Add, copy(n), cu(ts - 1));
    let tiles_col = bin(&mut s, Ty::Usize, BinOp::Div, copy(cols_padded), cu(ts));
    let k_padded = bin(&mut s, Ty::Usize, BinOp::Add, copy(k), cu(ts - 1));
    let kt_count = bin(&mut s, Ty::Usize, BinOp::Div, copy(k_padded), cu(ts));
    let tr = bin(&mut s, Ty::Usize, BinOp::Div, copy(l), cu(ts));
    let tc = bin(&mut s, Ty::Usize, BinOp::Rem, copy(l), cu(ts));
    let trow = bin(&mut s, Ty::Usize, BinOp::Div, copy(g), copy(tiles_col));
    let tcol = bin(&mut s, Ty::Usize, BinOp::Rem, copy(g), copy(tiles_col));
    let row_base = bin(&mut s, Ty::Usize, BinOp::Mul, copy(trow), cu(ts));
    let row = bin(&mut s, Ty::Usize, BinOp::Add, copy(row_base), copy(tr));
    let col_base = bin(&mut s, Ty::Usize, BinOp::Mul, copy(tcol), cu(ts));
    let col = bin(&mut s, Ty::Usize, BinOp::Add, copy(col_base), copy(tc));
    let last_row = bin(&mut s, Ty::Usize, BinOp::Sub, copy(rows), cu(1));
    let row_c = bin(&mut s, Ty::Usize, BinOp::Min, copy(row), copy(last_row));
    let last_col = bin(&mut s, Ty::Usize, BinOp::Sub, copy(n), cu(1));
    let col_c = bin(&mut s, Ty::Usize, BinOp::Min, copy(col), copy(last_col));
    let last_k = bin(&mut s, Ty::Usize, BinOp::Sub, copy(k), cu(1));
    let row_ok = bin(&mut s, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
    let col_ok = bin(&mut s, Ty::Bool, BinOp::Lt, copy(col), copy(n));
    let RowExpert { valid, .. } = emit_row_expert(&mut s, ids, row_c, experts);
    let poison = emit_poison(&mut s, valid);
    let key = emit_row_key(&mut s, ids, row, (rows, experts));
    let tr_ts = bin(&mut s, Ty::Usize, BinOp::Mul, copy(tr), cu(ts));
    let mut setup = s.take();
    setup.extend([
        Statement::Assign(
            Place::local(acc),
            Rvalue::Use(Operand::Const(Constant::F32(0.0))),
        ),
        Statement::Assign(Place::local(lo), Rvalue::Use(cu(0))),
    ]);
    let bb_setup = BasicBlock {
        statements: setup,
        terminator: goto(BB_WALK),
    };

    // Walk: the least key at or above `lo` among the tile's rows; `E` when none is left. Every lane reads the
    // tile's `ts` ids straight from `ids` at tile-uniform indexes (a `GroupX`/`GroupY`-derived base), so the
    // result is provably the same for every lane of the workgroup: the barriers in the K loop below sit under
    // a uniform branch.
    let mut s = Emit::new(&mut al);
    let mut next = s.let_(Ty::Usize, Rvalue::Use(copy(experts)));
    for r in 0..ts {
        let tile_row = bin(&mut s, Ty::Usize, BinOp::Add, copy(row_base), cu(r));
        let key_r = emit_row_key(&mut s, ids, tile_row, (rows, experts));
        let at_or_above = bin(&mut s, Ty::Bool, BinOp::Ge, copy(key_r), copy(lo));
        let above_u = s.let_(
            Ty::Usize,
            Rvalue::Cast {
                to: Ty::Usize,
                operand: copy(at_or_above),
            },
        );
        let distance = bin(&mut s, Ty::Usize, BinOp::Sub, copy(experts), copy(key_r));
        let pulled = bin(&mut s, Ty::Usize, BinOp::Mul, copy(above_u), copy(distance));
        let candidate = bin(&mut s, Ty::Usize, BinOp::Sub, copy(experts), copy(pulled));
        next = bin(&mut s, Ty::Usize, BinOp::Min, copy(next), copy(candidate));
    }
    let more = bin(&mut s, Ty::Bool, BinOp::Lt, copy(next), copy(experts));
    let bb_walk = BasicBlock {
        statements: s.take(),
        terminator: guard(more, BB_STORE_GUARD, BB_SLAB),
    };

    // Step setup: this step's expert slab (a real key, so in range) and a zeroed step accumulator.
    let mut s = Emit::new(&mut al);
    let step = weight.step(&mut s, next, (k, n));
    let mut slab_stmts = s.take();
    slab_stmts.extend([
        Statement::Assign(
            Place::local(part),
            Rvalue::Use(Operand::Const(Constant::F32(0.0))),
        ),
        Statement::Assign(Place::local(kt), Rvalue::Use(cu(0))),
    ]);
    let bb_slab = BasicBlock {
        statements: slab_stmts,
        terminator: goto(BB_K_HEADER),
    };

    let mut s = Emit::new(&mut al);
    let kt_in = bin(&mut s, Ty::Bool, BinOp::Lt, copy(kt), copy(kt_count));
    let bb_k_header = BasicBlock {
        statements: s.take(),
        terminator: guard(kt_in, BB_AFTER, BB_LOAD),
    };

    // Load: this lane's activation element and weight element of K sub-tile `kt`, branchless and masked.
    let mut s = Emit::new(&mut al);
    let k_off = bin(&mut s, Ty::Usize, BinOp::Mul, copy(kt), cu(ts));
    let a_k = bin(&mut s, Ty::Usize, BinOp::Add, copy(k_off), copy(tc));
    let b_k = bin(&mut s, Ty::Usize, BinOp::Add, copy(k_off), copy(tr));
    let a_k_ok = bin(&mut s, Ty::Bool, BinOp::Lt, copy(a_k), copy(k));
    let b_k_ok = bin(&mut s, Ty::Bool, BinOp::Lt, copy(b_k), copy(k));
    let a_k_c = bin(&mut s, Ty::Usize, BinOp::Min, copy(a_k), copy(last_k));
    let b_k_c = bin(&mut s, Ty::Usize, BinOp::Min, copy(b_k), copy(last_k));
    let a_row = bin(&mut s, Ty::Usize, BinOp::Mul, copy(row_c), copy(k));
    let a_at = bin(&mut s, Ty::Usize, BinOp::Add, copy(a_row), copy(a_k_c));
    let a_raw = load(&mut s, x, a_at);
    let a_keep = bin(&mut s, Ty::Bool, BinOp::BitAnd, copy(row_ok), copy(a_k_ok));
    let a_val = masked(&mut s, a_keep, a_raw);
    s.write_lds(0, copy(l), copy(a_val));
    let b_raw = weight.load(&mut s, &step, (b_k_c, col_c), (k, n));
    let b_keep = bin(&mut s, Ty::Bool, BinOp::BitAnd, copy(col_ok), copy(b_k_ok));
    let b_val = masked(&mut s, b_keep, b_raw);
    s.write_lds(1, copy(l), copy(b_val));
    let bb_load = BasicBlock {
        statements: s.take(),
        terminator: barrier(BB_DOT),
    };

    // Dot: `part += As[tr, t] * Bs[t, tc]` for `t` ascending, so `part` sums `K` in order across sub-tiles.
    let mut s = Emit::new(&mut al);
    let mut sum = part;
    for t in 0..ts {
        let a_slot = bin(&mut s, Ty::Usize, BinOp::Add, copy(tr_ts), cu(t));
        let b_slot = bin(&mut s, Ty::Usize, BinOp::Add, copy(tc), cu(t * ts));
        let a = s.read_lds(Ty::F32, 0, copy(a_slot));
        let b = s.read_lds(Ty::F32, 1, copy(b_slot));
        let product = bin(&mut s, Ty::F32, BinOp::Mul, copy(a), copy(b));
        sum = bin(&mut s, Ty::F32, BinOp::Add, copy(sum), copy(product));
    }
    s.assign(Place::local(part), Rvalue::Use(copy(sum)));
    let bb_dot = BasicBlock {
        statements: s.take(),
        terminator: goto(BB_LATCH),
    };

    let mut s = Emit::new(&mut al);
    let kt_next = bin(&mut s, Ty::Usize, BinOp::Add, copy(kt), cu(1));
    s.assign(Place::local(kt), Rvalue::Use(copy(kt_next)));
    let bb_latch = BasicBlock {
        statements: s.take(),
        terminator: barrier(BB_K_HEADER),
    };

    // After a step: move `lo` past it, and the rows that selected it keep the step's sum.
    let mut s = Emit::new(&mut al);
    let lo_next = bin(&mut s, Ty::Usize, BinOp::Add, copy(next), cu(1));
    s.assign(Place::local(lo), Rvalue::Use(copy(lo_next)));
    let mine = bin(&mut s, Ty::Bool, BinOp::Eq, copy(key), copy(next));
    let bb_after = BasicBlock {
        statements: s.take(),
        terminator: guard(mine, BB_WALK, BB_TAKE),
    };
    let bb_take = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(acc),
            Rvalue::Use(copy(part)),
        )],
        terminator: goto(BB_WALK),
    };

    let mut s = Emit::new(&mut al);
    let stores = bin(&mut s, Ty::Bool, BinOp::BitAnd, copy(row_ok), copy(col_ok));
    let bb_store_guard = BasicBlock {
        statements: s.take(),
        terminator: guard(stores, BB_RETURN, BB_STORE),
    };

    // Store: the sum plus the row's poison (NaN for an invalid id, whose sum is the untouched 0), through
    // the epilogue, narrowed to `dt`.
    let mut s = Emit::new(&mut al);
    let out_row = bin(&mut s, Ty::Usize, BinOp::Mul, copy(row), copy(n));
    let out_at = bin(&mut s, Ty::Usize, BinOp::Add, copy(out_row), copy(col));
    let total = bin(&mut s, Ty::F32, BinOp::Add, copy(acc), copy(poison));
    let value = match epilogue {
        Some(GemvEpilogue {
            epilogue,
            leaves,
            lanes,
        }) => emit_epilogue(&mut s, epilogue, out_at, leaves, lanes, total),
        None => total,
    };
    s.assign(elem(out, out_at), Rvalue::Use(copy(value)));
    let bb_store = BasicBlock {
        statements: s.take(),
        terminator: Terminator::Return,
    };
    let bb_return = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    let mut body = Body::new(
        name,
        n_inputs,
        al.locals,
        vec![
            bb_group_x,
            bb_group_y,
            bb_lane,
            bb_setup,
            bb_walk,
            bb_slab,
            bb_k_header,
            bb_load,
            bb_dot,
            bb_latch,
            bb_after,
            bb_take,
            bb_store_guard,
            bb_store,
            bb_return,
        ],
    );
    body.workgroup_size = [(ts * ts) as u32, 1, 1];
    let lds = |len: usize| WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: len as u32,
    };
    body.workgroup_locals = vec![lds(ts * ts), lds(ts * ts)];
    body
}
