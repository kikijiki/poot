//! The indexed contraction: `out[m, n] = sum over k of x[m, k] * W[ids[m], k, n]` over a stacked `[E, K, N]`
//! weight, where output row `m` reads the expert slab `ids[m]` (the MoE expert matmul, and every per-row
//! adapter). Row selection is a property of the contraction's index map, not of a loader: both schedules read
//! the expert through [`emit_row_expert`], and the schedule only decides the loop order.
//!
//! - [`Schedule::Gemv`] (decode, few rows): the shared Gemv loop nest ([`gemv_body`]) with an indexed run
//!   read. Flat output column `m * N + n` is the "weight row"; its run read takes the expert slab of row `m`
//!   and the `[K, N]` weight at stride `N`.
//! - [`Schedule::RegisterTiled`] (prefill, many rows) is *segmented*: the rows of one tile may select different
//!   experts, so the tile walks the distinct expert ids among its rows in ascending order ([`ExpertWalk`]). Each
//!   step stages that expert's weight tile once through LDS with the other experts' rows zeroed, so the weight tile
//!   is shared by every row of an expert-sorted tile (the whole gain of grouping) and the result is correct for
//!   any id order.
//!
//! A [`WeightLayout::Nk`] weight is the same stack with each expert stored `[N, K]` (the checkpoint order): the
//! slab stride is still `K * N` and only the in-slab index swaps, so no transpose of the experts ever runs.
//!
//! An expert id outside `0..E` is not read: its output row is NaN on every
//! schedule (R-632-4), and the slab it would have addressed is clamped in range so no read leaves the buffer.

use poot_kernel_ir::{BinOp, Body, Constant, Local, Operand, Rvalue, Ty};

use poot_target::BufferStorage;

use super::Schedule;
use super::epilogue::{BoundEpilogue, Epilogue};
use super::gemv::{GemvLaunch, gemv_body};
use super::tiled::{ExpertWalk, TiledBody, TiledLaunch, TiledWeight, WalkStep, tiled_body};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu, emit_read_meta, select_u32};
use crate::helpers::{Alloc, WeightLane, WeightLayout, elem, ld, local, slice_dtype, slice_f32};

/// The launch metadata words of an indexed contraction, in order: the register-tiled body's own five (`K`, `N`, `M`
/// the activation rows, `x_groups` and `tile_offset`; [`super::tiled::META_K`]..), then `E` (the expert count). The
/// Gemv reads `K`, `N` and `E` and ignores the rest.
pub(crate) const META_N: usize = super::tiled::META_OUT_PER_BLOCK;
pub(crate) const META_EXPERTS: usize = 5;
/// The first word past the shared layout: a packed table's per-source slab strides start here.
pub(crate) const META_PACKED_SLAB_WORDS: usize = 6;

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

/// `ids[row]` as an expert: valid when it is in `0..experts`, with the expert taken from the id clamped after
/// the cast, so whatever the cast of a negative id yields, the expert stays inside the table.
pub(crate) fn emit_row_expert(e: &mut Emit, ids: Local, row: Local, experts: Local) -> RowExpert {
    let id = e.let_(Ty::I32, Rvalue::Use(Operand::Copy(elem(ids, row))));
    let experts_i = e.let_(
        Ty::I32,
        Rvalue::Cast {
            to: Ty::I32,
            operand: copy(experts),
        },
    );
    let non_negative = bin(
        e,
        Ty::Bool,
        BinOp::Ge,
        copy(id),
        Operand::Const(Constant::I32(0)),
    );
    let in_range = bin(e, Ty::Bool, BinOp::Lt, copy(id), copy(experts_i));
    let valid = bin(
        e,
        Ty::Bool,
        BinOp::BitAnd,
        copy(non_negative),
        copy(in_range),
    );
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
pub(crate) fn emit_row_key(
    e: &mut Emit,
    ids: Local,
    row: Local,
    (rows, experts): (Local, Local),
) -> Local {
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
/// metadata, out)`; `ids` is `I32` and `metadata` is `[K, N, M, x_groups, tile_offset, E]`. `k` is the `K` the
/// planner generates for (it decides only whether a packed row's runs start on a word boundary). `epilogue` (f32)
/// runs over each output element's sum before its store.
pub(crate) fn indexed_contraction(
    name: &str,
    schedule: Schedule,
    layout: WeightLayout,
    weight: BufferStorage,
    k: usize,
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
            activation,
        } => indexed_gemv(
            name,
            GemvLaunch::new(width, cols, unroll, activation)?,
            layout,
            lane,
            epilogue,
        ),
        Schedule::RegisterTiled {
            lanes,
            rows,
            cols,
            run,
        } => indexed_register_tiled(
            name,
            TiledLaunch::new(lanes, rows, cols, run, layout)?,
            layout,
            lane,
            k,
            epilogue,
        ),
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
        ld(slice_dtype(Ty::I32, false), false),
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
    let epilogue = epilogue.map(|epilogue| BoundEpilogue {
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

/// The register-tiled indexed contraction (prefill, many rows): [`tiled_body`] with one block of `M` rows over
/// `N` weight rows, inside an [`ExpertWalk`] over the experts the tile's rows select. Each walk step reads its
/// expert's slab: an `[N, K]` expert's rows `expert * N ..` of the stack read as `[E * N, K]` rows, or a `[K, N]`
/// one's slab at `expert * K * N`. Params as [`indexed_gemv`]'s.
fn indexed_register_tiled(
    name: &str,
    launch: TiledLaunch,
    layout: WeightLayout,
    lane: WeightLane,
    k: usize,
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
        ld(slice_dtype(Ty::I32, false), false),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let (weight, ids) = (local(2), local(3));
    let leaves: Vec<Local> = (0..leaf_count).map(|j| local(4 + j as u32)).collect();
    let metadata = local(4 + leaf_count as u32);
    // An `[N, K]` step reads from its expert's first weight row, `expert * N` of the stack read as `[E * N, K]`
    // rows; a `[K, N]` step's slab offset is the body's own.
    let mut first_row = |e: &mut Emit, expert: Local| {
        let n = emit_read_meta(e, metadata, META_N);
        vec![bin(e, Ty::Usize, BinOp::Mul, copy(expert), copy(n))]
    };
    let mut no_state = |_: &mut Emit, _: Local| Vec::new();
    let word_aligned = k.is_multiple_of(lane.per_word());
    let mut row_run = |e: &mut Emit,
                       step: &[Local],
                       row: Local,
                       k0: Local,
                       count: usize,
                       k: Local,
                       _weight_rows: Local| {
        let row = bin(e, Ty::Usize, BinOp::Add, copy(step[0]), copy(row));
        let row_base = bin(e, Ty::Usize, BinOp::Mul, copy(row), copy(k));
        let first = bin(e, Ty::Usize, BinOp::Add, copy(row_base), copy(k0));
        lane.emit_run(e, weight, first, count, word_aligned)
    };
    let (step, weight_read): (&mut WalkStep<'_>, _) = match layout {
        WeightLayout::Nk => (&mut first_row, TiledWeight::Rows(&mut row_run)),
        WeightLayout::Kn => (&mut no_state, TiledWeight::Columns { weight, lane }),
    };
    let spec = TiledBody {
        name,
        params,
        activation: local(1),
        metadata,
        out: local(5 + leaf_count as u32),
        epilogue: epilogue.map(|epilogue| BoundEpilogue {
            epilogue,
            leaves: &leaves,
            lanes: &lanes,
        }),
        walk: Some(ExpertWalk {
            ids,
            experts_word: META_EXPERTS,
            step,
        }),
    };
    Ok(tiled_body(spec, launch, weight_read))
}
