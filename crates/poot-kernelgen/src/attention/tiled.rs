//! The [`AttentionSchedule::Tiled`] body: FlashAttention-2's loop nest (Dao, arXiv 2307.08691, algorithm 1) over
//! the micro-tile interface ([`crate::microtile`], `designs/prefill-microtile-1393.md`).
//!
//! One workgroup serves a block of `br` query positions of one `(batch, key/value head)` for each of the head's
//! `n_rep` query heads: its `rows = br * n_rep` tile rows are row `r = head * br + position`, query head
//! `kv_head * n_rep + r / br` at query row `first + r % br`. A padded row of the last block (past `lq`) reads the
//! last real query and is never stored.
//!
//! The query tile is staged in workgroup memory once, transposed (`Q^T [dk][rows + 1]`). A fragment schedule that
//! holds Q ([`QOperand::Held`], POOT-1447) instead stages it through the key/value array a chunk of channels at a
//! time before the walk, each wave loading its A fragments of the chunk into registers it keeps for the whole walk;
//! its score product then loads only K. The keys are walked in blocks of `bc`; per block:
//!
//! 1. stage `K^T` (`[dk][bc + 1]`, or `[bc][dk + 8]` for fragments); barrier;
//! 2. `S = Q K^T` into each lane's score register tile; score transform and additive mask; on a block some row
//!    cannot see all of (the causal diagonal, a window's edge, the live end, the capacity tail) the hidden
//!    scores become `-inf`; each owned row's maximum is published; barrier;
//! 3. stage `V [bc][dv]` over the keys' storage; one lane per row folds the published maxima into the running
//!    maximum (`m_new = max(m, maxima)`) and publishes `m_new` and `alpha = exp(m - m_new)`; barrier; a score
//!    lane stages `P = exp(S - m_new)` transposed (`P^T [bc][rows + 1]`) and publishes each owned row's sum, an
//!    output lane rescales its rows by `alpha`; barrier;
//! 4. every lane accumulates `O += P V`; one lane per row moves the row's state on (`l = l * alpha + sum`,
//!    `m = m_new`); barrier.
//!
//! The two products run on the micro-tile of the schedule ([`AttentionMicro`]): register tiles per lane, or
//! matrix fragments per wave (POOT-1437). A fragment tile stages Q, K, V and P in the operand dtype, and a `Declared`
//! fragment layout reads its accumulator elements in place. An `Opaque` one spills each accumulator to a scratch
//! array and reads it back after a barrier (the scores), and rescales the output by spill, scale and reload, which
//! adds two barriers to step 3.
//!
//! After the last block every lane divides its output elements by their row's `l` and stores the real rows.
//!
//! Every barrier sits in the key loop, whose bounds are uniform over the workgroup: they come from the block's
//! query positions (all of its rows), the live length and the capacity. The loop starts at the block of the
//! earliest key a row's window shows and ends at the last key a row sees. A key past that end is staged as zero
//! and reads nothing, so no read passes the live length, the capacity or the slot map. Unlike the streaming
//! schedule, a key one row of the block sees is read by every row of it: a non-finite value there reaches a row
//! that cannot see it as `0 * inf`, NaN. Real key and value rows are finite.
//!
//! A score of `-inf` (hidden, or an additive `-inf`) weighs exactly zero, and a row that has seen only such
//! scores keeps `m = -inf` and is routed around `-inf - -inf`; a row with no visible key at all divides zero by
//! zero, the oracle's NaN, as the streaming schedule does.

use poot_kernel_ir::{
    BinOp, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue, Ty, Unrolling,
    WorkgroupLocalDecl,
};

use super::{
    AttentionMicro, AttentionQueryScale, AttentionSchedule, AttentionSink, AttentionSpec,
    GENERATOR, KeyBound, KeyLayout, KvLane, LaneTile, MaskKind, QOperand, WaveTile, count,
    count_to, emit_score, f32_array, f32c, local_op,
};
use crate::KernelGenError;
use crate::data_movement::Cfg;
use crate::emit::{Emit, bin, cu};
use crate::helpers::{elem, ld, slice_dtype, slice_f32};
use crate::microtile::{
    Acc, Coord, FragmentTile, Geometry, LaneRows, MicroCols, MicroTile, Staged, StagedMajor,
};
use crate::precision::narrowed;
use poot_graph_ir::OperandPrecision;
use poot_target::RoundingConversions;

/// `K^T` (element `(d, key j)` at `d * (bc + 1) + j`) for the score product, then `V` (element `(j, channel c)`
/// at `j * dv + c`) for the value product: one region, staged in turn. A fragment tile has `K` as `[key][d]` and `V`
/// as `[channel][key]`, the depth of each product adjacent. A held Q ([`QOperand::Held`]) passes through it in
/// depth chunks before the key loop.
const KV_ARRAY: u8 = 0;
/// `P^T`: element `(key j, row r)` at `j * (rows + 1) + r`; a fragment tile has the keys of a row adjacent, at
/// `r * (bc + 8) + j`.
const P_ARRAY: u8 = 1;
/// The rows' partial maxima, then (once the new maxima are folded and read) their partial sums, of a block: row
/// `r`'s owner `o` at `r * owners + o`. The two are never live together, so they share one array.
const RED_ARRAY: u8 = 2;
/// The running maximum `m` and the running sum `l` of each row.
const M_ARRAY: u8 = 3;
const L_ARRAY: u8 = 4;
/// The running maximum each row moves to in a block and the factor its state is rescaled by.
const NEW_M_ARRAY: u8 = 5;
const ALPHA_ARRAY: u8 = 6;
/// The arrays every tiled body declares; a staged Q tile ([`TiledLayout::q_array`]) and the scratch of an `Opaque`
/// fragment accumulator ([`TiledLayout::scratch_array`]) follow, each only where the schedule has it.
const FIXED_ARRAYS: u8 = 7;

/// Depth steps one trip of a scalar product's run-time loop unrolls.
const UNROLL: usize = 8;
/// Words a fragment tile's row is padded past its depth: 16 bytes of the 2-byte operand, so every row stays
/// aligned for a vector load and the rows spread over the banks.
const FRAGMENT_PAD: usize = 8;

/// A tiled schedule's shape, checked once: the tile rows, the key block, and the two register tiles with the
/// lane grids they form.
#[derive(Clone, Copy, Debug)]
pub(super) struct TiledLayout {
    rows: usize,
    br: usize,
    bc: usize,
    lanes: usize,
    score: Grid,
    out: Grid,
    q: QOperand,
}

/// One product's register tile and its `lanes_m x lanes_n` lane grid.
#[derive(Clone, Copy, Debug)]
struct Grid {
    micro: MicroTile,
    lanes_m: usize,
    lanes_n: usize,
}

impl Grid {
    /// The grid `tile` forms over a `rows x cols` tile in a workgroup of `lanes` lanes.
    fn scalar(
        what: &str,
        tile: LaneTile,
        (rows, cols): (usize, usize),
        lanes: usize,
    ) -> Result<Self, KernelGenError> {
        for (dim, value, divisor) in [("rows", rows, tile.rows), ("columns", cols, tile.cols)] {
            if divisor == 0 {
                return Err(super::bad(&format!("{what} lane-tile {dim}"), 0, 1));
            }
            if !value.is_multiple_of(divisor) {
                return Err(KernelGenError::NotDivisible {
                    generator: GENERATOR,
                    dim: format!("{what} tile {dim}"),
                    value,
                    divisor,
                });
            }
        }
        let (lanes_m, lanes_n) = (rows / tile.rows, cols / tile.cols);
        if lanes_m * lanes_n != lanes {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: format!(
                    "workgroup lanes over the {what} tile's {lanes_m} x {lanes_n} lane grid"
                ),
                value: lanes,
                divisor: lanes_m * lanes_n,
            });
        }
        let narrow = |value: usize| {
            u32::try_from(value).map_err(|_| KernelGenError::ExceedsBound {
                generator: GENERATOR,
                what: format!("{what} lane tile"),
                value,
                bound: u32::MAX as usize,
            })
        };
        Ok(Self {
            micro: MicroTile::Scalar {
                rows: LaneRows(narrow(tile.rows)?),
                cols: MicroCols::new(narrow(tile.cols)?),
            },
            lanes_m,
            lanes_n,
        })
    }

    /// The grid of waves `wave` forms over a `rows x cols` tile in a workgroup of `lanes` lanes.
    fn fragment(
        what: &str,
        tile: FragmentTile,
        wave: WaveTile,
        (rows, cols): (usize, usize),
        lanes: usize,
    ) -> Result<Self, KernelGenError> {
        for (dim, value) in [
            ("waves along rows", wave.waves_m),
            ("waves along columns", wave.waves_n),
            ("fragments along rows", tile.rows as usize),
            ("fragments along columns", tile.cols as usize),
        ] {
            if value == 0 {
                return Err(super::bad(&format!("{what} {dim}"), 0, 1));
            }
        }
        let micro = MicroTile::Fragment(tile);
        let covered = micro.output_tile(wave.waves_m, wave.waves_n);
        for (dim, value, covered) in [("rows", rows, covered.0), ("columns", cols, covered.1)] {
            if value != covered {
                return Err(KernelGenError::NotDivisible {
                    generator: GENERATOR,
                    dim: format!("{what} tile {dim} over its wave grid of whole fragments"),
                    value,
                    divisor: covered,
                });
            }
        }
        let wave_lanes = wave.waves_m * wave.waves_n * micro.owner_lanes();
        if wave_lanes != lanes {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: format!("workgroup lanes over the {what} tile's wave grid"),
                value: lanes,
                divisor: wave_lanes,
            });
        }
        Ok(Self {
            micro,
            lanes_m: wave.waves_m,
            lanes_n: wave.waves_n,
        })
    }

    fn geometry(self, e: &mut Emit, lane: Local) -> Geometry {
        self.micro.geometry(e, lane, self.lanes_m, self.lanes_n)
    }

    /// The lanes that own elements of one row of this product's tile (the partials a row publishes).
    fn row_owners(self) -> usize {
        self.micro.row_owners(self.lanes_n)
    }

    /// The f32 words of scratch this product's `Opaque` accumulator spills to, 0 for any other.
    fn scratch_words(self) -> usize {
        self.micro.scratch_words(self.lanes_m * self.lanes_n)
    }
}

impl TiledLayout {
    /// The layout of `spec`'s tiled schedule, or the typed reason it cannot be built.
    pub(super) fn of(spec: &AttentionSpec) -> Result<Self, KernelGenError> {
        let AttentionSchedule::Tiled {
            br,
            bc,
            lanes,
            micro,
        } = spec.schedule
        else {
            unreachable!("only a tiled schedule has a tiled layout")
        };
        for (what, value) in [("br", br), ("bc", bc)] {
            if value == 0 {
                return Err(super::bad(what, value, 1));
            }
        }
        if let MaskKind::Blocks { .. } = spec.mask {
            return Err(KernelGenError::Unsupported {
                request: "attention",
                reason: "the tiled schedule walks dense key blocks, not a block selection",
            });
        }
        let rows = br * spec.n_rep;
        // Every lane stages the same share of each tile ([`stage_strided`]).
        let shared_dim = match spec.key {
            KeyLayout::Split { shared_dim } => shared_dim,
            KeyLayout::Single | KeyLayout::Paged => 0,
        };
        for (what, total) in [
            ("staged query", rows * spec.dk),
            ("staged key", bc * (spec.dk - shared_dim)),
            ("staged shared key", bc * shared_dim),
            ("staged value", bc * spec.dv),
        ] {
            if lanes == 0 || !total.is_multiple_of(lanes) {
                return Err(KernelGenError::NotDivisible {
                    generator: GENERATOR,
                    dim: format!("{what} elements"),
                    value: total,
                    divisor: lanes,
                });
            }
        }
        let (score, out, q) = match micro {
            AttentionMicro::Scalar { score, out } => (
                Grid::scalar("score", score, (rows, bc), lanes)?,
                Grid::scalar("output", out, (rows, spec.dv), lanes)?,
                QOperand::Staged,
            ),
            AttentionMicro::Fragment {
                dtype,
                layout,
                score,
                out,
                q,
            } => {
                // The operands are staged already narrowed, so the dtype the fragments multiply in is the
                // precision's.
                if spec.precision != OperandPrecision::Narrow(dtype) {
                    return Err(KernelGenError::Unsupported {
                        request: "attention_tiled",
                        reason: "a fragment attention narrows its operands to the fragment dtype",
                    });
                }
                let tile = |wave: WaveTile| FragmentTile {
                    dtype,
                    layout,
                    rows: wave.rows,
                    cols: wave.cols,
                };
                (
                    Grid::fragment("score", tile(score), score, (rows, bc), lanes)?,
                    Grid::fragment("output", tile(out), out, (rows, spec.dv), lanes)?,
                    q,
                )
            }
        };
        // The score product's depth is the head width, the value product's the key block: whole steps of each
        // micro-tile.
        for (what, depth, micro) in [
            ("head width", spec.dk, score.micro),
            ("key block", bc, out.micro),
        ] {
            if !depth.is_multiple_of(micro.kappa()) {
                return Err(KernelGenError::NotDivisible {
                    generator: GENERATOR,
                    dim: format!("{what} over the micro-tile's depth step"),
                    value: depth,
                    divisor: micro.kappa(),
                });
            }
        }
        let layout = Self {
            rows,
            br,
            bc,
            lanes,
            score,
            out,
            q,
        };
        if q == QOperand::Held && layout.q_chunk(spec.dk, spec.dv).is_none() {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: "a held query's depth chunks through the key/value array".into(),
                value: spec.dk,
                divisor: micro_edge(),
            });
        }
        Ok(layout)
    }

    /// The depth a held Q is staged in per pass through the key/value array before the key loop: the widest
    /// multiple of the fragment depth dividing `dk` whose `rows x (chunk + FRAGMENT_PAD)` elements fit the array and
    /// split evenly over the lanes. `None` for a staged Q, or where not even one fragment depth fits.
    fn q_chunk(&self, dk: usize, dv: usize) -> Option<usize> {
        if self.q != QOperand::Held {
            return None;
        }
        let edge = micro_edge();
        let capacity = self.kv_len(dk, dv);
        (1..=dk / edge)
            .rev()
            .map(|steps| steps * edge)
            .find(|&chunk| {
                dk.is_multiple_of(chunk)
                    && self.rows * (chunk + FRAGMENT_PAD) <= capacity
                    && (self.rows * chunk).is_multiple_of(self.lanes)
            })
    }

    /// The elements of the key/value array: the larger of the staged `K` and `V` tiles.
    fn kv_len(&self, dk: usize, dv: usize) -> usize {
        let Self { bc, score, .. } = *self;
        match score.micro {
            MicroTile::Scalar { .. } => (dk * (bc + 1)).max(bc * dv),
            MicroTile::Fragment(_) => (bc * (dk + FRAGMENT_PAD)).max(dv * (bc + FRAGMENT_PAD)),
        }
    }

    /// The array of a staged Q tile, `None` for a held one.
    fn q_array(&self) -> Option<u8> {
        (self.q == QOperand::Staged).then_some(FIXED_ARRAYS)
    }

    /// The array an `Opaque` fragment accumulator spills to, after a staged Q's; `None` for any other micro-tile.
    fn scratch_array(&self) -> Option<u8> {
        (self.score.scratch_words().max(self.out.scratch_words()) > 0)
            .then(|| FIXED_ARRAYS + u8::from(self.q_array().is_some()))
    }

    /// The staged operand tiles of the two products and the elements of each one's array.
    ///
    /// A scalar tile is `K`-major throughout, rows padded by one word where lanes read adjacent ones. A fragment
    /// lane loads 16 consecutive depth values of its row (or column), so every tile is `X`-major: Q and K (whose
    /// rows are the query and key positions, the depth the head width), and P (a row per query, the keys along it)
    /// and V (a row per channel, the keys along it), whose depth is the key position. A `K`-major P and V would
    /// read each fragment lane's 16 depth values as 16 separate 2-byte loads. Every row is padded by
    /// [`FRAGMENT_PAD`]. A held Q is staged in depth chunks through the key/value array with K's layout
    /// ([`Self::q_chunk`]).
    fn tiles(&self, dk: usize, dv: usize) -> StagedTiles {
        let Self {
            rows, bc, score, ..
        } = *self;
        let x_major = |array, stride| Staged {
            array,
            stride,
            major: StagedMajor::X,
        };
        let kv_len = self.kv_len(dk, dv);
        match score.micro {
            MicroTile::Scalar { .. } => StagedTiles {
                q: QTile::Staged(Staged::k_major(FIXED_ARRAYS, rows + 1)),
                k: Staged::k_major(KV_ARRAY, bc + 1),
                p: Staged::k_major(P_ARRAY, rows + 1),
                v: Staged::k_major(KV_ARRAY, dv),
                q_len: dk * (rows + 1),
                kv_len,
                p_len: bc * (rows + 1),
            },
            MicroTile::Fragment(_) => StagedTiles {
                q: match self.q_chunk(dk, dv) {
                    None => QTile::Staged(x_major(FIXED_ARRAYS, dk + FRAGMENT_PAD)),
                    Some(chunk) => QTile::Held {
                        through: x_major(KV_ARRAY, chunk + FRAGMENT_PAD),
                        chunk,
                    },
                },
                k: x_major(KV_ARRAY, dk + FRAGMENT_PAD),
                p: x_major(P_ARRAY, bc + FRAGMENT_PAD),
                v: x_major(KV_ARRAY, bc + FRAGMENT_PAD),
                q_len: rows * (dk + FRAGMENT_PAD),
                kv_len,
                p_len: rows * (bc + FRAGMENT_PAD),
            },
        }
    }

    /// The workgroup-memory arrays the body declares, in array order: the key/value and P tiles (in the
    /// micro-tile's operand type), the rows' partial maxima and sums (one array), the running `m` and `l`, the new
    /// maxima and rescale factors, then a staged Q tile and the scratch of an `Opaque` fragment accumulator where
    /// the schedule has them.
    fn locals(&self, dk: usize, dv: usize) -> Vec<WorkgroupLocalDecl> {
        let Self {
            rows, score, out, ..
        } = *self;
        let tiles = self.tiles(dk, dv);
        let staged = |len: usize| WorkgroupLocalDecl {
            elem_ty: score.micro.staged_ty(),
            len: len as u32,
        };
        let mut locals = vec![
            staged(tiles.kv_len),
            staged(tiles.p_len),
            f32_array(rows * score.row_owners()),
            f32_array(rows),
            f32_array(rows),
            f32_array(rows),
            f32_array(rows),
        ];
        if self.q_array().is_some() {
            locals.push(staged(tiles.q_len));
        }
        if self.scratch_array().is_some() {
            locals.push(f32_array(score.scratch_words().max(out.scratch_words())));
        }
        locals
    }
}

/// The workgroup-memory bytes the tiled schedule of `spec` declares, or the typed reason the schedule cannot be
/// built.
pub(super) fn lds_bytes(spec: &AttentionSpec) -> Result<usize, KernelGenError> {
    let layout = TiledLayout::of(spec)?;
    Ok(layout
        .locals(spec.dk, spec.dv)
        .iter()
        .map(|local| {
            local.len as usize
                * match local.elem_ty {
                    Ty::F32 => std::mem::size_of::<f32>(),
                    Ty::F16 | Ty::BF16 => 2,
                    _ => unreachable!("the body declares only f32 and 16-bit float arrays"),
                }
        })
        .sum())
}

/// The staged operand tiles of the two products ([`TiledLayout::tiles`]) and the elements of each array.
struct StagedTiles {
    q: QTile,
    k: Staged,
    p: Staged,
    v: Staged,
    q_len: usize,
    kv_len: usize,
    p_len: usize,
}

/// Where the score product reads Q from.
#[derive(Clone, Copy)]
enum QTile {
    /// Its own array, for the whole key loop.
    Staged(Staged),
    /// Held fragments, staged `chunk` channels at a time through `through` (the key/value array, K's layout)
    /// before the key loop.
    Held { through: Staged, chunk: usize },
}

/// The depth one fragment step consumes (every fragment config lowers 16x16x16).
const fn micro_edge() -> usize {
    poot_target::FragmentShape::M16N16K16.k as usize
}

fn op(local: Local) -> Operand {
    local_op(local)
}

/// A usize `a op n`.
fn un(e: &mut Emit, op_: BinOp, a: Local, n: usize) -> Local {
    bin(e, Ty::Usize, op_, op(a), cu(n))
}

/// A usize `a op b`.
fn uu(e: &mut Emit, op_: BinOp, a: Local, b: Local) -> Local {
    bin(e, Ty::Usize, op_, op(a), op(b))
}

/// `if cond { a } else { b }`.
fn select(e: &mut Emit, ty: Ty, cond: Local, a: Operand, b: Operand) -> Local {
    e.let_(
        ty,
        Rvalue::Select {
            condition: op(cond),
            if_true: a,
            if_false: b,
        },
    )
}

/// The `I32` word `buf[at]` as an index, a negative one read as 0.
fn non_negative_index(e: &mut Emit, buf: Local, at: Local) -> Local {
    let raw = e.let_(Ty::I32, Rvalue::Use(crate::helpers::copy(elem(buf, at))));
    let at_least_zero = bin(
        e,
        Ty::I32,
        BinOp::Max,
        op(raw),
        Operand::Const(Constant::I32(0)),
    );
    e.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: op(at_least_zero),
        },
    )
}

/// The softmax weight of score `s` under the row maximum `m`: `exp(s - m)`, exactly 0 for a `-inf` score (whose
/// row maximum may itself be `-inf`).
fn weight(e: &mut Emit, s: Local, m: Local) -> Local {
    let below = bin(e, Ty::F32, BinOp::Sub, op(s), op(m));
    let w = e.let_(
        Ty::F32,
        Rvalue::MathUnary(poot_kernel_ir::MathOp::Exp, op(below)),
    );
    let hidden = bin(e, Ty::Bool, BinOp::Eq, op(s), f32c(f32::NEG_INFINITY));
    select(e, Ty::F32, hidden, f32c(0.0), op(w))
}

/// Row `row`'s running maximum before and after this block: `m` from workgroup memory, and `max(m, ..)` of the
/// block's partial maxima its `owners` lanes published.
fn running_max(e: &mut Emit, row: Local, owners: usize) -> (Local, Local) {
    let old = e.read_lds(Ty::F32, M_ARRAY, op(row));
    let base = un(e, BinOp::Mul, row, owners);
    let new = (0..owners).fold(old, |max, o| {
        let at = un(e, BinOp::Add, base, o);
        let part = e.read_lds(Ty::F32, RED_ARRAY, op(at));
        bin(e, Ty::F32, BinOp::Max, op(max), op(part))
    });
    (old, new)
}

/// The rescale of what a row held when its maximum moves from `old` to `new`: `exp(old - new)`, exactly 0 for a
/// row that has seen only `-inf` scores (whose state is empty).
fn rescale(e: &mut Emit, old: Local, new: Local) -> Local {
    let behind = bin(e, Ty::F32, BinOp::Sub, op(old), op(new));
    let factor = e.let_(
        Ty::F32,
        Rvalue::MathUnary(poot_kernel_ir::MathOp::Exp, op(behind)),
    );
    let empty = bin(e, Ty::Bool, BinOp::Eq, op(old), f32c(f32::NEG_INFINITY));
    select(e, Ty::F32, empty, f32c(0.0), op(factor))
}

/// Where a workgroup's tile rows sit in the operands.
#[derive(Clone, Copy)]
struct TileRows {
    b: Local,
    first_row: Local,
    first_head: Local,
    br: usize,
    hq: usize,
    lq: usize,
}

impl TileRows {
    /// Tile row `r`'s query head.
    fn head(self, e: &mut Emit, r: Local) -> Local {
        let within = un(e, BinOp::Div, r, self.br);
        uu(e, BinOp::Add, self.first_head, within)
    }

    /// Tile row `r`'s query row (unclamped: a padded row is past `lq`).
    fn query(self, e: &mut Emit, r: Local) -> Local {
        let within = un(e, BinOp::Rem, r, self.br);
        uu(e, BinOp::Add, self.first_row, within)
    }

    /// Tile row `r`'s query row, a padded one read as the last real row.
    fn clamped_query(self, e: &mut Emit, r: Local) -> Local {
        let i = self.query(e, r);
        un(e, BinOp::Min, i, self.lq - 1)
    }

    /// The `(batch, head, query)` row of `q` and `out`.
    fn flat(self, e: &mut Emit, h: Local, i: Local) -> Local {
        let by_batch = un(e, BinOp::Mul, self.b, self.hq);
        let bh = uu(e, BinOp::Add, by_batch, h);
        let by_row = un(e, BinOp::Mul, bh, self.lq);
        uu(e, BinOp::Add, by_row, i)
    }

    /// The query's index among the `[B, Lq]` positions.
    fn position_index(self, e: &mut Emit, i: Local) -> Local {
        let by_batch = un(e, BinOp::Mul, self.b, self.lq);
        uu(e, BinOp::Add, by_batch, i)
    }
}

/// A `depth`-deep product `acc += A[:, ..depth] * B[..depth, :]` in the micro-tile's steps of [`MicroTile::kappa`]
/// values: a run-time loop of trips of [`trip_steps`] steps, then the remainder unrolled.
fn product(
    c: &mut Cfg,
    micro: MicroTile,
    g: &Geometry,
    acc: &mut Acc,
    (a, b): (Staged, Staged),
    depth: usize,
) {
    let kappa = micro.kappa();
    let steps = depth / kappa;
    let trip = trip_steps(micro);
    let whole = steps / trip * trip;
    if whole > 0 {
        count(c, cu(0), whole, trip, |c, s0| {
            c.emit(|e| {
                for u in 0..trip {
                    let step = un(e, BinOp::Add, s0, u);
                    let k = un(e, BinOp::Mul, step, kappa);
                    micro.step_at(e, g, acc, a, b, k);
                }
            });
        });
    }
    c.emit(|e| {
        for step in whole..steps {
            micro.step(e, g, acc, a, b, step * kappa);
        }
    });
}

/// The steps one trip of a product's run-time loop unrolls: [`UNROLL`] scalar steps, or the two fragment steps
/// of one staging round ([`MicroTile::block_k`]). A fragment step loads its operand fragments before its
/// multiplies, and a longer trip lets the scheduler hoist the loads of every step ahead of the first multiply.
fn trip_steps(micro: MicroTile) -> usize {
    match micro {
        MicroTile::Scalar { .. } => UNROLL,
        MicroTile::Fragment(_) => micro.block_k() / micro.kappa(),
    }
}

/// The word of element `(k, x)` (staged depth `k`, the tile's row or column `x`) of `tile`.
fn staged_word(e: &mut Emit, tile: Staged, k: Local, x: Local) -> Local {
    let (outer, inner) = match tile.major {
        StagedMajor::K => (k, x),
        StagedMajor::X => (x, k),
    };
    let row = un(e, BinOp::Mul, outer, tile.stride);
    uu(e, BinOp::Add, row, inner)
}

/// The keys one workgroup walks, all uniform over it: `[first, end)` is every key some row of the block sees (from
/// the first key block a row's window reaches), `[full_start, full_end)` the keys every row sees, and `live_end`
/// the capacity or the live length.
struct KeyRange {
    first: Local,
    end: Local,
    full_start: Local,
    full_end: Local,
    live_end: Local,
}

/// Build the tiled attention kernel `spec` describes over its row groups `rows`, on the same 2-D `x_groups` fold
/// and with the same parameters as the streaming body ([`super::attention_body`]).
pub(super) fn tiled_body(
    name: &str,
    spec: &AttentionSpec,
    rows: std::ops::Range<usize>,
    x_groups: usize,
    conversions: RoundingConversions,
) -> Result<Body, KernelGenError> {
    let layout = TiledLayout::of(spec)?;
    let TiledLayout {
        rows: tile_rows,
        br,
        bc,
        lanes,
        score: score_grid,
        out: out_grid,
        q: _,
    } = layout;
    let AttentionSpec {
        batch: _,
        hq,
        n_rep,
        lq,
        lk,
        dk,
        dv,
        key,
        score,
        mask: mask_kind,
        bound,
        sink,
        query,
        schedule: _,
        kv,
        precision,
    } = *spec;
    // One staged tile per operand: a narrowed operand is rounded as it is staged (Q, K, V and P), and a split one,
    // which has a `lo` tile as well, is the streaming schedule's.
    if let OperandPrecision::Split(_) = precision {
        return Err(KernelGenError::Unsupported {
            request: "attention_tiled",
            reason: "a split operand precision has no tiled schedule",
        });
    }
    let tiles = layout.tiles(dk, dv);
    let scratch = layout.scratch_array();
    let hkv = hq / n_rep;
    let blocks_per_head = lq.div_ceil(br);
    let shared_dim = match key {
        KeyLayout::Single | KeyLayout::Paged => 0,
        KeyLayout::Split { shared_dim } => shared_dim,
    };
    let head_dim = dk - shared_dim;

    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        kv.param(),                  // 2 k
        kv.param(),                  // 3 v
        // 4 the query positions or the additive mask
        match mask_kind {
            MaskKind::Causal { .. } => ld(slice_dtype(Ty::I32, false), false),
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Blocks { .. } => {
                ld(slice_f32(false), false)
            }
        },
    ];
    let mut next_param = |decl| {
        params.push(decl);
        Local {
            index: params.len() as u32 - 1,
        }
    };
    let k_shared = match key {
        KeyLayout::Split { .. } => Some(next_param(kv.param())),
        KeyLayout::Single | KeyLayout::Paged => None,
    };
    let read = match key {
        KeyLayout::Paged => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
        KeyLayout::Single | KeyLayout::Split { .. } => None,
    };
    let len = match bound {
        KeyBound::Live => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
        KeyBound::Static => None,
    };
    let sinks = match sink {
        AttentionSink::PerHead => Some(next_param(ld(slice_f32(false), false))),
        AttentionSink::Absent => None,
    };
    let query_scales = match query {
        AttentionQueryScale::PerRow => Some(next_param(ld(slice_f32(false), false))),
        AttentionQueryScale::Absent => None,
    };
    let out = next_param(ld(slice_f32(true), true));
    let (q, k, v, mask) = (
        Local { index: 1 },
        Local { index: 2 },
        Local { index: 3 },
        Local { index: 4 },
    );
    let mut c = Cfg::new(params);

    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    let place = c.emit(|e| {
        let by_row = un(e, BinOp::Mul, group_y, x_groups);
        uu(e, BinOp::Add, by_row, group_x)
    });
    let in_range = c.bin(Ty::Bool, BinOp::Lt, op(place), cu(rows.len()));

    c.if_then(in_range, |c| {
        // The row group `((b * hkv) + kv_head) * blocks_per_head + block`.
        let (tile, kv_head) = c.emit(|e| {
            let g = un(e, BinOp::Add, place, rows.start);
            let block = un(e, BinOp::Rem, g, blocks_per_head);
            let by_head = un(e, BinOp::Div, g, blocks_per_head);
            let kv_head = un(e, BinOp::Rem, by_head, hkv);
            let b = un(e, BinOp::Div, by_head, hkv);
            let tile = TileRows {
                b,
                first_row: un(e, BinOp::Mul, block, br),
                first_head: un(e, BinOp::Mul, kv_head, n_rep),
                br,
                hq,
                lq,
            };
            (tile, kv_head)
        });

        // Channel `d` of tile row `r`'s query, rounded to the operand precision.
        let load_q = |e: &mut Emit, r: Local, d: Local| {
            let h = tile.head(e, r);
            let i = tile.clamped_query(e, r);
            let row = tile.flat(e, h, i);
            let base = un(e, BinOp::Mul, row, dk);
            let at = uu(e, BinOp::Add, base, d);
            let raw = e.let_(Ty::F32, Rvalue::Use(crate::helpers::copy(elem(q, at))));
            narrowed(e, precision, None, conversions, raw)
        };
        // Stage Q once, adjacent lanes on adjacent channels of a query row, into its own array for the whole loop. A
        // held Q is staged after the geometry below.
        if let QTile::Staged(staged) = tiles.q {
            stage_strided(
                c,
                (lane, lanes),
                (tile_rows, dk),
                std::mem::size_of::<f32>(),
                StageOrder::AlongColumns,
                load_q,
                |e, r, d, value| {
                    let slot = staged_word(e, staged, d, r);
                    let value = score_grid.micro.stage_value(e, value);
                    e.write_lds(staged.array, op(slot), op(value));
                },
            );
        }
        // Each row's running state: empty, or the head's sink logit with its unit mass.
        count(c, op(lane), tile_rows, lanes, |c, r| {
            c.emit(|e| {
                let (m0, l0) = match sinks {
                    None => (f32c(f32::NEG_INFINITY), f32c(0.0)),
                    Some(sinks) => {
                        let h = tile.head(e, r);
                        let logit =
                            e.let_(Ty::F32, Rvalue::Use(crate::helpers::copy(elem(sinks, h))));
                        (op(logit), f32c(1.0))
                    }
                };
                e.write_lds(M_ARRAY, op(r), m0);
                e.write_lds(L_ARRAY, op(r), l0);
            });
        });

        let keys = key_range(c, (tile, mask, len), (mask_kind, lk, bc));
        let (gs, go) = c.emit(|e| (score_grid.geometry(e, lane), out_grid.geometry(e, lane)));
        // A held Q is staged through the key/value array a chunk of channels at a time, each wave loading its A
        // fragments of the chunk's depth steps before the next chunk overwrites it.
        let held = match tiles.q {
            QTile::Staged(_) => None,
            QTile::Held { through, chunk } => {
                let kappa = score_grid.micro.kappa();
                let mut held = Vec::with_capacity(dk / kappa);
                for d0 in (0..dk).step_by(chunk) {
                    stage_strided(
                        c,
                        (lane, lanes),
                        (tile_rows, chunk),
                        std::mem::size_of::<f32>(),
                        StageOrder::AlongColumns,
                        |e, r, d| {
                            let channel = un(e, BinOp::Add, d, d0);
                            load_q(e, r, channel)
                        },
                        |e, r, d, value| {
                            let slot = staged_word(e, through, d, r);
                            let value = score_grid.micro.stage_value(e, value);
                            e.write_lds(through.array, op(slot), op(value));
                        },
                    );
                    c.barrier();
                    held.extend(c.emit(|e| {
                        (0..chunk / kappa)
                            .map(|step| score_grid.micro.hold_a(e, &gs, through, step * kappa))
                            .collect::<Vec<_>>()
                    }));
                    // The last chunk's loads are ordered before the first key block's staging by the barrier
                    // that publishes the rows' state below.
                    if d0 + chunk < dk {
                        c.barrier();
                    }
                }
                Some(held)
            }
        };
        // The score rows' query positions (causal) or additive-mask rows (`(plane * lq + i) * lk`).
        let row_positions = match mask_kind {
            MaskKind::Causal { .. } => Some(c.emit(|e| {
                gs.row_values(e, |e, r| {
                    let i = tile.clamped_query(e, r);
                    let at = tile.position_index(e, i);
                    non_negative_index(e, mask, at)
                })
            })),
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Blocks { .. } => None,
        };
        let mask_rows = match mask_kind {
            MaskKind::Broadcast | MaskKind::PerHead => Some(c.emit(|e| {
                gs.row_values(e, |e, r| {
                    let i = tile.clamped_query(e, r);
                    let plane = match mask_kind {
                        MaskKind::PerHead => {
                            let by_batch = un(e, BinOp::Mul, tile.b, hq);
                            let h = tile.head(e, r);
                            uu(e, BinOp::Add, by_batch, h)
                        }
                        _ => tile.b,
                    };
                    let by_plane = un(e, BinOp::Mul, plane, lq);
                    let row = uu(e, BinOp::Add, by_plane, i);
                    un(e, BinOp::Mul, row, lk)
                })
            })),
            MaskKind::Causal { .. } | MaskKind::Blocks { .. } => None,
        };
        // The score rows' query scales, read once for the whole key walk.
        let row_scales = query_scales.map(|scales| {
            c.emit(|e| {
                gs.row_values(e, |e, r| {
                    let h = tile.head(e, r);
                    let i = tile.clamped_query(e, r);
                    let row = tile.flat(e, h, i);
                    e.let_(
                        Ty::F32,
                        Rvalue::Use(crate::helpers::copy(elem(scales, row))),
                    )
                })
            })
        });

        let mut acc_o = c.emit(|e| {
            let mut acc = out_grid.micro.accumulator(e.parts().0);
            out_grid.micro.zero(e, &mut acc);
            acc
        });
        // Q^T and the row state are visible to every lane.
        c.barrier();

        // The storage rows of key `t`: of the key's own part, its value, and (dense) its shared part.
        let kv_plane = c.emit(|e| {
            let by_batch = un(e, BinOp::Mul, tile.b, hkv);
            uu(e, BinOp::Add, by_batch, kv_head)
        });
        let key_rows = |e: &mut Emit, t: Local| -> (Local, Local) {
            match read {
                None => {
                    let plane = un(e, BinOp::Mul, kv_plane, lk);
                    let row = uu(e, BinOp::Add, plane, t);
                    let by_batch = un(e, BinOp::Mul, tile.b, lk);
                    (row, uu(e, BinOp::Add, by_batch, t))
                }
                Some(read) => {
                    let base = un(e, BinOp::Mul, tile.b, lk);
                    let at = uu(e, BinOp::Add, base, t);
                    let word = e.let_(Ty::I32, Rvalue::Use(crate::helpers::copy(elem(read, at))));
                    let slot = e.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: op(word),
                        },
                    );
                    let by_slot = un(e, BinOp::Mul, slot, hkv);
                    let pool_row = uu(e, BinOp::Add, by_slot, kv_head);
                    (pool_row, pool_row)
                }
            }
        };
        let owners = gs.row_owners();
        count_to(c, op(keys.first), op(keys.end), bc, |c, k0| {
            // 1. K^T.
            let stage = StageKeys {
                k0,
                end: keys.end,
                lane,
                lanes,
                bc,
                kv,
            };
            stage.run(
                c,
                (k, head_dim),
                StageOrder::AlongColumns,
                |e, t| key_rows(e, t).0,
                |e, j, d, value| {
                    let slot = staged_word(e, tiles.k, d, j);
                    let value = narrowed(e, precision, kv.fragment(), conversions, value);
                    let value = score_grid.micro.stage_value(e, value);
                    e.write_lds(KV_ARRAY, op(slot), op(value));
                },
            );
            if let Some(k_shared) = k_shared {
                stage.run(
                    c,
                    (k_shared, shared_dim),
                    StageOrder::AlongColumns,
                    |e, t| key_rows(e, t).1,
                    |e, j, d, value| {
                        let channel = un(e, BinOp::Add, d, head_dim);
                        let slot = staged_word(e, tiles.k, channel, j);
                        let value = narrowed(e, precision, kv.fragment(), conversions, value);
                        let value = score_grid.micro.stage_value(e, value);
                        e.write_lds(KV_ARRAY, op(slot), op(value));
                    },
                );
            }
            c.barrier();

            // 2. S = Q K^T, scored and masked; the rows' maxima.
            let mut acc_s = c.emit(|e| {
                let mut acc = score_grid.micro.accumulator(e.parts().0);
                score_grid.micro.zero(e, &mut acc);
                acc
            });
            match (&held, tiles.q) {
                (None, QTile::Staged(q_tile)) => {
                    product(c, score_grid.micro, &gs, &mut acc_s, (q_tile, tiles.k), dk);
                }
                // A held fragment is a register: its depth step is a compile-time one, so the product is unrolled.
                (Some(held), QTile::Held { .. }) => c.emit(|e| {
                    let kappa = score_grid.micro.kappa();
                    for (step, a) in held.iter().enumerate() {
                        score_grid
                            .micro
                            .step_held(e, &gs, &mut acc_s, a, tiles.k, step * kappa);
                    }
                }),
                _ => unreachable!("Q is held exactly when its tile is"),
            }
            // An opaque fragment's elements are read back from workgroup memory, once every wave has stored its own.
            if score_grid.micro.needs_spill() {
                c.emit(|e| score_grid.micro.spill(e, &gs, &acc_s, scratch));
                c.barrier();
            }
            let mut s = c.emit(|e| {
                let mut s = score_grid.micro.elements(e, &gs, acc_s, scratch);
                match &row_scales {
                    Some(scales) => {
                        s.map_rows(e, scales, |e, _, scale, dot| {
                            emit_score(e, score, Some(scale), dot)
                        });
                    }
                    None => s.map(e, |e, _, dot| emit_score(e, score, None, dot)),
                }
                if let Some(mask_rows) = &mask_rows {
                    s.map_rows(e, mask_rows, |e, coord, row, value| {
                        let t = uu(e, BinOp::Add, k0, coord.col);
                        let t = un(e, BinOp::Min, t, lk - 1);
                        let at = uu(e, BinOp::Add, row, t);
                        let added =
                            e.let_(Ty::F32, Rvalue::Use(crate::helpers::copy(elem(mask, at))));
                        bin(e, Ty::F32, BinOp::Add, op(value), op(added))
                    });
                }
                s
            });
            // Only a block some row cannot see whole hides scores.
            let partial = c.emit(|e| {
                let block_end = un(e, BinOp::Add, k0, bc);
                let past_full = bin(e, Ty::Bool, BinOp::Gt, op(block_end), op(keys.full_end));
                let before_full = bin(e, Ty::Bool, BinOp::Lt, op(k0), op(keys.full_start));
                bin(e, Ty::Bool, BinOp::BitOr, op(past_full), op(before_full))
            });
            c.if_then(partial, |c| {
                c.emit(|e| {
                    let hide = |e: &mut Emit,
                                coord: Coord,
                                position: Option<Local>,
                                value: Local| {
                        let t = uu(e, BinOp::Add, k0, coord.col);
                        let mut visible = bin(e, Ty::Bool, BinOp::Lt, op(t), op(keys.live_end));
                        if let (Some(position), MaskKind::Causal { window }) = (position, mask_kind)
                        {
                            let seen = bin(e, Ty::Bool, BinOp::Le, op(t), op(position));
                            visible = bin(e, Ty::Bool, BinOp::BitAnd, op(visible), op(seen));
                            if let Some(window) = window {
                                // position - t < window, i.e. t + window > position
                                let reach = un(e, BinOp::Add, t, window);
                                let inside = bin(e, Ty::Bool, BinOp::Gt, op(reach), op(position));
                                visible = bin(e, Ty::Bool, BinOp::BitAnd, op(visible), op(inside));
                            }
                        }
                        select(e, Ty::F32, visible, op(value), f32c(f32::NEG_INFINITY))
                    };
                    match &row_positions {
                        Some(positions) => s.map_rows(e, positions, |e, coord, position, value| {
                            hide(e, coord, Some(position), value)
                        }),
                        None => s.map(e, |e, coord, value| hide(e, coord, None, value)),
                    }
                });
            });
            c.emit(|e| s.row_reduce(e, BinOp::Max).publish(e, &gs, RED_ARRAY));
            c.barrier();

            // 3. V over K's storage. Every lane folds the published maxima of the rows it owns into their running
            // maximum: the score lanes weigh their scores against it, stage P and publish the rows' sums; the
            // output lanes rescale their rows.
            stage.run(
                c,
                (v, dv),
                // A fragment tile stores V transposed, `[channel][key]`.
                match tiles.v.major {
                    StagedMajor::X => StageOrder::AlongRows,
                    StagedMajor::K => StageOrder::AlongColumns,
                },
                |e, t| key_rows(e, t).0,
                |e, j, d, value| {
                    let slot = staged_word(e, tiles.v, j, d);
                    let value = narrowed(e, precision, kv.fragment(), conversions, value);
                    let value = out_grid.micro.stage_value(e, value);
                    e.write_lds(KV_ARRAY, op(slot), op(value));
                },
            );
            // One lane per row folds the published partial maxima into the row's running maximum and publishes the
            // new maximum and the factor the row's state is rescaled by, so that the `rows x owners` partials are
            // read once per row and not once per element a lane holds.
            count(c, op(lane), tile_rows, lanes, |c, r| {
                c.emit(|e| {
                    let (old, new) = running_max(e, r, owners);
                    let alpha = rescale(e, old, new);
                    e.write_lds(NEW_M_ARRAY, op(r), op(new));
                    e.write_lds(ALPHA_ARRAY, op(r), op(alpha));
                });
            });
            c.barrier();
            let alphas = c.emit(|e| {
                let maxima = gs.row_values(e, |e, row| e.read_lds(Ty::F32, NEW_M_ARRAY, op(row)));
                s.map_rows(e, &maxima, |e, _, m, value| weight(e, value, m));
                s.stage(e, tiles.p, |e, p| {
                    let rounded = narrowed(e, precision, None, conversions, p);
                    out_grid.micro.stage_value(e, rounded)
                });
                s.row_reduce(e, BinOp::Add).publish(e, &gs, RED_ARRAY);
                go.row_values(e, |e, row| e.read_lds(Ty::F32, ALPHA_ARRAY, op(row)))
            });
            if out_grid.micro.needs_spill() {
                // An opaque accumulator has no element access to rescale in place: spill it, scale the spilled
                // elements, reload it. P and V are visible after the first barrier, the scaled elements after
                // the second.
                c.emit(|e| out_grid.micro.spill(e, &go, &acc_o, scratch));
                c.barrier();
                c.emit(|e| out_grid.micro.scale_spilled(e, &go, &alphas, scratch));
                c.barrier();
                c.emit(|e| out_grid.micro.reload(e, &go, &mut acc_o, scratch));
            } else {
                c.emit(|e| out_grid.micro.scale_rows(e, &mut acc_o, &alphas));
                c.barrier();
            }

            // 4. O += P V; one lane per row moves the row's running state on.
            product(c, out_grid.micro, &go, &mut acc_o, (tiles.p, tiles.v), bc);
            count(c, op(lane), tile_rows, lanes, |c, r| {
                c.emit(|e| {
                    let new = e.read_lds(Ty::F32, NEW_M_ARRAY, op(r));
                    let alpha = e.read_lds(Ty::F32, ALPHA_ARRAY, op(r));
                    let l_old = e.read_lds(Ty::F32, L_ARRAY, op(r));
                    let decayed = bin(e, Ty::F32, BinOp::Mul, op(l_old), op(alpha));
                    let base = un(e, BinOp::Mul, r, owners);
                    let sum = (0..owners).fold(decayed, |sum, o| {
                        let at = un(e, BinOp::Add, base, o);
                        let part = e.read_lds(Ty::F32, RED_ARRAY, op(at));
                        bin(e, Ty::F32, BinOp::Add, op(sum), op(part))
                    });
                    e.write_lds(L_ARRAY, op(r), op(sum));
                    e.write_lds(M_ARRAY, op(r), op(new));
                });
            });
            c.barrier();
        });

        // O / l, stored for the real rows.
        if out_grid.micro.needs_spill() {
            c.emit(|e| out_grid.micro.spill(e, &go, &acc_o, scratch));
            c.barrier();
        }
        let o = c.emit(|e| {
            let sums = go.row_values(e, |e, row| e.read_lds(Ty::F32, L_ARRAY, op(row)));
            let mut o = out_grid.micro.elements(e, &go, acc_o, scratch);
            o.map_rows(e, &sums, |e, _, l, value| {
                bin(e, Ty::F32, BinOp::Div, op(value), op(l))
            });
            o
        });
        o.for_each(|coord, value| {
            let (i, real) = c.emit(|e| {
                let i = tile.query(e, coord.row);
                (i, bin(e, Ty::Bool, BinOp::Lt, op(i), cu(lq)))
            });
            c.if_then(real, |c| {
                c.emit(|e| {
                    let h = tile.head(e, coord.row);
                    let row = tile.flat(e, h, i);
                    let base = un(e, BinOp::Mul, row, dv);
                    let at = uu(e, BinOp::Add, base, coord.col);
                    e.assign(elem(out, at), Rvalue::Use(op(value)));
                });
            });
        });
    });

    let mut body = c.finish(name);
    body.workgroup_size = [lanes as u32, 1, 1];
    body.unrolling = Unrolling::Generator;
    body.workgroup_locals = layout.locals(dk, dv);
    Ok(body)
}

/// The cooperative staging of one key block: every lane loads a strided share of its `bc` keys' channels,
/// adjacent lanes on adjacent channels of a key.
#[derive(Clone, Copy)]
struct StageKeys {
    k0: Local,
    end: Local,
    lane: Local,
    lanes: usize,
    bc: usize,
    kv: KvLane,
}

impl StageKeys {
    /// Stage `width` channels of every key of the block from `part`, whose row of key `t` is `row_of(t)` (`width`
    /// values per row): `write(e, j, d, value)` places channel `d` of the block's key `j`. A key at or past the
    /// walk's end is zero and reads nothing.
    ///
    /// A key past the end reads the block's first key (always real: the walk only enters a block that starts
    /// before its end) and is selected away, so the loads carry no branch and a batch of them is in flight at
    /// once ([`stage_strided`]).
    fn run(
        self,
        c: &mut Cfg,
        (part, width): (Local, usize),
        order: StageOrder,
        row_of: impl Fn(&mut Emit, Local) -> Local,
        write: impl Fn(&mut Emit, Local, Local, Local),
    ) {
        stage_strided(
            c,
            (self.lane, self.lanes),
            (self.bc, width),
            self.kv.bytes(),
            order,
            |e, j, d| {
                let t = uu(e, BinOp::Add, self.k0, j);
                let real = bin(e, Ty::Bool, BinOp::Lt, op(t), op(self.end));
                let safe = select(e, Ty::Usize, real, op(t), op(self.k0));
                let row = row_of(e, safe);
                let base = un(e, BinOp::Mul, row, width);
                let at = uu(e, BinOp::Add, base, d);
                let loaded = self.kv.read(e, part, at);
                select(e, Ty::F32, real, op(loaded), f32c(0.0))
            },
            write,
        );
    }
}

/// The most bytes one lane holds in flight in [`stage_strided`]: 8 f32 or 16 two-byte elements. Measured on the bank's
/// `prefill-attention-8192` fragment cell (ROCm, gfx1151, POOT-1440): 16 two-byte loads in flight are 10% faster than
/// 4, where 16 f32 loads are 12% slower than 4.
const STAGE_BYTES: usize = 32;

/// Which of a staged tile's axes is contiguous in workgroup memory, and so which way one lane's batch of elements
/// runs (adjacent lanes stay on adjacent columns either way, so the global loads coalesce).
#[derive(Clone, Copy, PartialEq, Eq)]
enum StageOrder {
    /// The columns are contiguous where they are stored: a lane's batch strides the rows, adjacent lanes store
    /// adjacent words.
    AlongColumns,
    /// The rows are contiguous where they are stored (a tile staged transposed, like a fragment's `V`
    /// `[channel][key]`): a lane's batch is a run of consecutive rows of its column, so its stores are adjacent
    /// and aligned and can merge into one wide store per 16 bytes, instead of one 2-byte store per element at a
    /// lane stride that lands lanes on a repeating subset of the LDS banks.
    AlongRows,
}

/// Where a lane's elements of a `rows x width` tile sit when `lanes` lanes stage it, lane `l` taking flat elements
/// `l, l + lanes, ..` (adjacent lanes on adjacent columns, so the loads coalesce). Its `u`-th element is a
/// lane-dependent base plus a compile-time offset, so the addresses of a batch share one or two registers.
#[derive(Clone, Copy)]
enum Strided {
    /// `lanes` is a multiple of `width` and the order is [`StageOrder::AlongRows`]: the lane keeps column `col`;
    /// its rows are `start + n * groups + u` for its batch `n` (a multiple of the batch) and `u` below the batch,
    /// `start` its row group (of `groups = lanes / width`) times the batch.
    Runs {
        start: Local,
        col: Local,
        groups: usize,
    },
    /// `lanes` is a multiple of `width`: the lane keeps column `col` and steps `rows_per_trip` rows.
    Column {
        row: Local,
        col: Local,
        rows_per_trip: usize,
    },
    /// `width` is a multiple of `lanes`: element `u` is row `u / per_row`, column `col + lanes * (u % per_row)`.
    Row { col: Local, per_row: usize },
    /// Neither divides the other: the flat index is split per element.
    Flat { lane: Local },
}

impl Strided {
    fn new(
        e: &mut Emit,
        (lane, lanes): (Local, usize),
        width: usize,
        (order, batch): (StageOrder, usize),
    ) -> Self {
        if order == StageOrder::AlongRows && lanes.is_multiple_of(width) {
            let group = un(e, BinOp::Div, lane, width);
            Self::Runs {
                start: un(e, BinOp::Mul, group, batch),
                col: un(e, BinOp::Rem, lane, width),
                groups: lanes / width,
            }
        } else if lanes.is_multiple_of(width) {
            Self::Column {
                row: un(e, BinOp::Div, lane, width),
                col: un(e, BinOp::Rem, lane, width),
                rows_per_trip: lanes / width,
            }
        } else if width.is_multiple_of(lanes) {
            Self::Row {
                col: lane,
                per_row: width / lanes,
            }
        } else {
            Self::Flat { lane }
        }
    }

    /// The `(row, column)` of the lane's element `first + u`, `first` a run-time multiple of the batch (`None`
    /// for 0).
    fn at(
        self,
        e: &mut Emit,
        (lanes, width): (usize, usize),
        first: Option<Local>,
        u: usize,
    ) -> (Local, Local) {
        // The element's trip `n = first + u`, scaled by `step`, added to `base`.
        let advance = |e: &mut Emit, base: Local, step: usize| match first {
            None => un(e, BinOp::Add, base, u * step),
            Some(first) => {
                let trip = un(e, BinOp::Add, first, u);
                let by = un(e, BinOp::Mul, trip, step);
                uu(e, BinOp::Add, base, by)
            }
        };
        match self {
            Self::Runs { start, col, groups } => {
                // The batch's first row, then the element's compile-time offset in the run.
                let first_row = match first {
                    None => start,
                    Some(first) => {
                        let by = un(e, BinOp::Mul, first, groups);
                        uu(e, BinOp::Add, by, start)
                    }
                };
                (un(e, BinOp::Add, first_row, u), col)
            }
            Self::Column {
                row,
                col,
                rows_per_trip,
            } => (advance(e, row, rows_per_trip), col),
            Self::Row { col, per_row } => {
                let n = match first {
                    None => e.let_(Ty::Usize, Rvalue::Use(cu(u))),
                    Some(first) => un(e, BinOp::Add, first, u),
                };
                let row = un(e, BinOp::Div, n, per_row);
                let within = un(e, BinOp::Rem, n, per_row);
                let shift = un(e, BinOp::Mul, within, lanes);
                (row, uu(e, BinOp::Add, col, shift))
            }
            Self::Flat { lane } => {
                let idx = advance(e, lane, lanes);
                (un(e, BinOp::Div, idx, width), un(e, BinOp::Rem, idx, width))
            }
        }
    }
}

/// Stage a `rows x width` tile cooperatively over `lanes` lanes ([`Strided`]), in batches of up to
/// [`STAGE_BYTES`] bytes of `elem_bytes`-byte elements whose loads are all emitted before their stores, so their
/// latencies overlap.
/// `load(e, row, col)` must be safe for every element (no branch guards it); `store(e, row, col, value)` places it.
/// The tile is a multiple of `lanes` elements ([`TiledLayout::of`] checks it), so every lane stages the same
/// count and no element needs a guard.
fn stage_strided(
    c: &mut Cfg,
    (lane, lanes): (Local, usize),
    (rows, width): (usize, usize),
    elem_bytes: usize,
    order: StageOrder,
    load: impl Fn(&mut Emit, Local, Local) -> Local,
    store: impl Fn(&mut Emit, Local, Local, Local),
) {
    let per_lane = rows * width / lanes;
    let batch = (1..=(STAGE_BYTES / elem_bytes).min(per_lane))
        .rev()
        .find(|b| per_lane.is_multiple_of(*b))
        .expect("1 divides every count");
    let strided = c.emit(|e| Strided::new(e, (lane, lanes), width, (order, batch)));
    let emit_batch = |e: &mut Emit, first: Option<Local>| {
        let places: Vec<(Local, Local)> = (0..batch)
            .map(|u| strided.at(e, (lanes, width), first, u))
            .collect();
        let values: Vec<Local> = places.iter().map(|&(r, col)| load(e, r, col)).collect();
        for (&(r, col), &value) in places.iter().zip(&values) {
            store(e, r, col, value);
        }
    };
    if per_lane == batch {
        c.emit(|e| emit_batch(e, None));
        return;
    }
    count(c, cu(0), per_lane, batch, |c, n0| {
        c.emit(|e| emit_batch(e, Some(n0)));
    });
}

/// The keys the workgroup over `tile` walks (see [`KeyRange`]). A causal block reads its rows' positions at
/// workgroup-uniform indices, so the bounds are uniform and the barriers in the walk stay on one path.
fn key_range(
    c: &mut Cfg,
    (tile, mask, len): (TileRows, Local, Option<Local>),
    (mask_kind, lk, bc): (MaskKind, usize, usize),
) -> KeyRange {
    let live_end = match len {
        None => c.usize_const(lk),
        Some(len) => c.emit(|e| {
            let at_least_zero = non_negative_index(e, len, tile.b);
            un(e, BinOp::Min, at_least_zero, lk)
        }),
    };
    let MaskKind::Causal { window } = mask_kind else {
        let zero = c.usize_const(0);
        return KeyRange {
            first: zero,
            end: live_end,
            full_start: zero,
            full_end: live_end,
            live_end,
        };
    };
    let lowest = c.var(Ty::Usize, Rvalue::Use(cu(usize::MAX)));
    let highest = c.var(Ty::Usize, Rvalue::Use(cu(0)));
    count(c, cu(0), tile.br, 1, |c, u| {
        let (i, real) = c.emit(|e| {
            let i = uu(e, BinOp::Add, tile.first_row, u);
            (i, bin(e, Ty::Bool, BinOp::Lt, op(i), cu(tile.lq)))
        });
        c.if_then(real, |c| {
            c.emit(|e| {
                let at = tile.position_index(e, i);
                let position = non_negative_index(e, mask, at);
                let low = uu(e, BinOp::Min, lowest, position);
                e.assign(Place::local(lowest), Rvalue::Use(op(low)));
                let high = uu(e, BinOp::Max, highest, position);
                e.assign(Place::local(highest), Rvalue::Use(op(high)));
            });
        });
    });
    c.emit(|e| {
        let past_highest = un(e, BinOp::Add, highest, 1);
        let past_lowest = un(e, BinOp::Add, lowest, 1);
        // The first key a row at `position` sees: `position + 1 - window`, or 0.
        let window_start = |e: &mut Emit, past: Local| match window {
            None => e.let_(Ty::Usize, Rvalue::Use(cu(0))),
            Some(window) => {
                let lifted = un(e, BinOp::Max, past, window);
                un(e, BinOp::Sub, lifted, window)
            }
        };
        let earliest = window_start(e, past_lowest);
        let first_block = un(e, BinOp::Div, earliest, bc);
        KeyRange {
            first: un(e, BinOp::Mul, first_block, bc),
            end: uu(e, BinOp::Min, past_highest, live_end),
            full_start: window_start(e, past_highest),
            full_end: uu(e, BinOp::Min, past_lowest, live_end),
            live_end,
        }
    })
}
