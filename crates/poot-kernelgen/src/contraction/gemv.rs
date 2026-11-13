//! The [`super::Schedule::Gemv`] loop nest over a weight whose rows are the output columns (`[N, K]`, `K`
//! contiguous): one body for every operand load. The packed front ends and the dense loader
//! ([`super::dense`]) each supply only the read of a run of one row's weight values, in two phases
//! ([`RunRead`]) so the nest can issue a later trip's loads before this trip's decode.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty,
};

use super::epilogue::{BoundEpilogue, emit_epilogue};
use super::{GemvActivation, GemvRows, LoadDepth};
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, elem, guard};
use poot_target::RoundingConversions;

/// A [`super::Schedule::Gemv`]'s launch shape, checked once before emission: every extent at least one,
/// `width` a multiple of `cols`, so each column group owns exactly `width / cols` lanes, and a vector activation
/// read only over runs of whole groups.
#[derive(Clone, Copy)]
pub(crate) struct GemvLaunch {
    width: usize,
    cols: usize,
    unroll: usize,
    activation: GemvActivation,
    rows: usize,
    depth: usize,
}

/// The lanes of one [`GemvActivation::Vec4`] group read.
const ACTIVATION_GROUP: usize = GemvActivation::VEC4_LANES;

/// The alignment, in bytes, of one f32 activation group: its address is a multiple of the group's size.
const ACTIVATION_GROUP_ALIGN: u32 = (ACTIVATION_GROUP * size_of::<f32>()) as u32;

/// The bits of the NaN a launch with rows that straddle two activation rows computes.
const POISON_BITS: u32 = 0x7fc0_0000;

impl GemvLaunch {
    pub(crate) fn new(
        width: u32,
        cols: u32,
        unroll: u32,
        activation: GemvActivation,
        rows: GemvRows,
        depth: LoadDepth,
    ) -> Result<Self, KernelGenError> {
        for (what, value) in [("width", width), ("cols", cols), ("unroll", unroll)] {
            if value == 0 {
                return Err(KernelGenError::BelowMinimum {
                    generator: "contraction_gemv",
                    what: what.to_string(),
                    value: 0,
                    min: 1,
                });
            }
        }
        if !width.is_multiple_of(cols) {
            return Err(KernelGenError::NotDivisible {
                generator: "contraction_gemv",
                dim: "width".to_string(),
                value: width as usize,
                divisor: cols as usize,
            });
        }
        if activation == GemvActivation::Vec4 && !(unroll as usize).is_multiple_of(ACTIVATION_GROUP)
        {
            return Err(KernelGenError::NotDivisible {
                generator: "contraction_gemv",
                dim: "unroll (vector activation)".to_string(),
                value: unroll as usize,
                divisor: ACTIVATION_GROUP,
            });
        }
        Ok(Self {
            width: width as usize,
            cols: cols as usize,
            unroll: unroll as usize,
            activation,
            rows: rows.get() as usize,
            depth: depth.get() as usize,
        })
    }

    /// Lanes that share one output column.
    fn lanes_per_col(self) -> usize {
        self.width / self.cols
    }
}

/// Where one lane reads a run of one weight row: the `count` logical values `k0 .. k0 + count` of row `row`, over a
/// contraction of depth `k` and `out_total` output columns. `k0` is a multiple of `count` (a full run, or a
/// one-element tail step).
#[derive(Clone, Copy)]
pub(crate) struct RunAt {
    pub(crate) row: Local,
    pub(crate) k0: Local,
    pub(crate) count: usize,
    pub(crate) k: Local,
    pub(crate) out_total: Local,
}

/// How a front end reads a run, in the two phases [`gemv_body`] separates.
///
/// - `load` emits the run's weight loads and whatever address and factor arithmetic they need, and returns the
///   locals holding what it read: the run's state in flight. It is called with the run it loads, which at a
///   [`super::LoadDepth`] above one is a later trip's than the one decoded next.
/// - `decode(at, loaded)` turns the state `load` returned for `at` into the run's `count` f32 values, in order.
///
/// A returned state is scalar locals (`Usize`, `U32`, `I32`, `F32`, `Bool`) of a number and types that depend only
/// on the run's `count`, so the loop can carry it across trips.
pub(crate) trait RunRead {
    fn load(&mut self, e: &mut Emit, at: RunAt) -> Vec<Local>;
    fn decode(&mut self, e: &mut Emit, at: RunAt, loaded: &[Local]) -> Vec<Local>;
}

/// A front end that reads a run in two phases, each a closure.
pub(crate) struct Phased<Load, Decode> {
    pub(crate) load: Load,
    pub(crate) decode: Decode,
}

impl<Load, Decode> RunRead for Phased<Load, Decode>
where
    Load: FnMut(&mut Emit, RunAt) -> Vec<Local>,
    Decode: FnMut(&mut Emit, RunAt, &[Local]) -> Vec<Local>,
{
    fn load(&mut self, e: &mut Emit, at: RunAt) -> Vec<Local> {
        (self.load)(e, at)
    }

    fn decode(&mut self, e: &mut Emit, at: RunAt, loaded: &[Local]) -> Vec<Local> {
        (self.decode)(e, at, loaded)
    }
}

/// A front end whose run cannot issue its loads apart from its decode (each element's address, the factors of a
/// planar scale, are computed where they are read): it loads nothing ahead and reads the whole run in `decode`. A
/// [`super::LoadDepth`] above one carries nothing across trips for it.
pub(crate) struct Undivided<Read>(pub(crate) Read);

impl<Read: FnMut(&mut Emit, RunAt) -> Vec<Local>> RunRead for Undivided<Read> {
    fn load(&mut self, _: &mut Emit, _: RunAt) -> Vec<Local> {
        Vec::new()
    }

    fn decode(&mut self, e: &mut Emit, at: RunAt, _: &[Local]) -> Vec<Local> {
        (self.0)(e, at)
    }
}

/// The zero a carried local starts as: the loop reads every slot of the rotation, so none may be unassigned when a
/// short row never loads it.
fn zero(ty: &Ty) -> Constant {
    match ty {
        Ty::Usize => Constant::Usize(0),
        Ty::U32 => Constant::U32(0),
        Ty::I32 => Constant::I32(0),
        Ty::F32 => Constant::F32(0.0),
        Ty::Bool => Constant::Bool(false),
        other => panic!("a Gemv run carries scalars across trips, not {other:?}"),
    }
}

/// The [`super::Schedule::Gemv`] loop nest, shared by every operand load; `read` supplies the `count` logical
/// weight values `k0 .. k0 + count` of one row ([`RunRead`]).
///
/// Workgroup `g` owns `cols` column groups of `rows` adjacent output columns each, columns
/// `(g*cols + c)*rows + r`. Lane `lane` serves group `c = lane / L` (`L = width / cols` lanes per group) as
/// sub-lane `s = lane % L`, so a group's lanes are adjacent and read one contiguous run of each of its `rows` weight
/// rows per trip: sub-lane `s` reads the `unroll` elements `k0 .. k0 + unroll` with
/// `k0 = trip * L * unroll + s * unroll`, and meets each activation value once for all its rows. Full runs take the
/// unrolled body; the one partial run a `K` that is not a multiple of `L * unroll` leaves is finished element by
/// element, so no `K` is refused. A column past the output (the last workgroup's tail) reads the last real column
/// instead and never stores, keeping every lane at every barrier.
///
/// At depth `d` the loads of the next `d - 1` full runs of each row are in flight: a prologue loads the first
/// `d - 1`, and each trip loads the run `d - 1` trips ahead before it decodes the current one, so the loop
/// issues loads only for runs it will decode (a run that does not fit is not loaded). Every row's sum sees the
/// same products in the same order at any `rows` and `depth`.
///
/// Row `r`'s partials land in workgroup array `r` at `s * cols + c`, so the per-column fold is a tree of stride
/// `cols` over the `L` sub-lanes and leaves column `c`'s sum at `LDS[r][c]`; `epilogue`, when given, runs over
/// that sum (at output element `store_col`) before the store.
#[expect(clippy::too_many_arguments)] // the name, locals and launch shape of one body plus the operand read and epilogue
pub(crate) fn gemv_body<R: RunRead>(
    name: &str,
    mut al: Alloc,
    param_count: u32,
    [activation, metadata, out]: [Local; 3],
    launch: GemvLaunch,
    conversions: RoundingConversions,
    epilogue: Option<BoundEpilogue>,
    mut read: R,
) -> Body {
    let GemvLaunch {
        width,
        cols,
        unroll,
        activation: activation_read,
        rows,
        depth,
    } = launch;
    let lanes_per_col = launch.lanes_per_col();
    let trip_stride = lanes_per_col * unroll;
    let ahead = depth - 1;
    let group = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(group),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };

    let mut setup = Emit::new(&mut al);
    let k = emit_read_meta(&mut setup, metadata, 0);
    let out_per_block = emit_read_meta(&mut setup, metadata, 1);
    let out_total = setup.let_(Ty::Usize, Rvalue::Len(Place::local(out)));
    let col_in_group = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(lane),
        cu(lanes_per_col),
    );
    let sub_lane = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(lane),
        cu(lanes_per_col),
    );
    let group_base = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(group), cu(cols));
    let col_flat = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(group_base),
        copy(col_in_group),
    );
    let last_col = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(out_total), cu(1));
    // The lane's rows are the `rows` adjacent output columns of its group, each clamped to the last real one.
    let weight_rows: Vec<Local> = (0..rows)
        .map(|r| {
            let out_row = row_of(&mut setup, col_flat, rows, r);
            bin(
                &mut setup,
                Ty::Usize,
                BinOp::Min,
                copy(out_row),
                copy(last_col),
            )
        })
        .collect();
    let b = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(weight_rows[0]),
        copy(out_per_block),
    );
    let act_row_base = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(b), copy(k));
    // The rows share the activation row of the first, which holds only while they lie in one block.
    let poison = (rows > 1).then(|| {
        let remainder = bin(
            &mut setup,
            Ty::Usize,
            BinOp::Rem,
            copy(out_per_block),
            cu(rows),
        );
        let straddles = bin(&mut setup, Ty::Bool, BinOp::Ne, copy(remainder), cu(0));
        let flag = setup.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(straddles),
            },
        );
        let mask = bin(&mut setup, Ty::U32, BinOp::Sub, c32(0), copy(flag));
        let bits = bin(
            &mut setup,
            Ty::U32,
            BinOp::BitAnd,
            copy(mask),
            c32(POISON_BITS),
        );
        setup.let_(
            Ty::F32,
            Rvalue::Bitcast {
                to: Ty::F32,
                operand: copy(bits),
            },
        )
    });
    let lds_slot = {
        let strided = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(sub_lane), cu(cols));
        bin(
            &mut setup,
            Ty::Usize,
            BinOp::Add,
            copy(strided),
            copy(col_in_group),
        )
    };
    let first_k = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(sub_lane),
        cu(unroll),
    );
    let mut init_stmts = setup.take();

    let partials: Vec<Local> = (0..rows).map(|_| al.add(Ty::F32, true)).collect();
    let k0 = al.add(Ty::Usize, true);
    for &partial in &partials {
        init_stmts.push(Statement::Assign(
            Place::local(partial),
            Rvalue::Use(Operand::Const(Constant::F32(0.0))),
        ));
    }
    init_stmts.push(Statement::Assign(
        Place::local(k0),
        Rvalue::Use(copy(first_k)),
    ));

    let at = |row: Local, k0: Local, count: usize| RunAt {
        row,
        k0,
        count,
        k,
        out_total,
    };

    // Block positions: 0..=2 setup, then `ahead` prologue steps of two blocks (guard, load), the loop head, a
    // prefetch guard and load when the pipeline is deeper than one trip, the body, the tail and the fold.
    let prologue_start = 3u32;
    let head = prologue_start + 2 * ahead as u32;
    let (body, prefetch) = if ahead == 0 {
        (head + 1, None)
    } else {
        (head + 3, Some((head + 1, head + 2)))
    };
    let tail_guard = body + 1;
    let tail_block = body + 2;
    let lds_write = body + 3;
    let round_start = body + 4;

    // The carried runs: `carried[slot][row]` are the locals slot `slot`'s loaded state of row `row` rotates through,
    // declared when the first load shows their types.
    let mut carried: Vec<Vec<Vec<Local>>> = Vec::new();
    let declare = |al: &mut Alloc, carried: &mut Vec<Vec<Vec<Local>>>, loaded: &[Vec<Local>]| {
        if carried.is_empty() {
            *carried = (0..depth)
                .map(|_| {
                    loaded
                        .iter()
                        .map(|state| {
                            state
                                .iter()
                                .map(|local| {
                                    let ty = al.locals[local.index as usize].ty.clone();
                                    al.add(ty, true)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
        }
    };
    // Statements for the carried slot `slot` taking `loaded`.
    let store_slot = |e: &mut Emit, slot: &[Vec<Local>], loaded: &[Vec<Local>]| {
        for (dst_row, src_row) in slot.iter().zip(loaded) {
            for (&dst, &src) in dst_row.iter().zip(src_row) {
                e.assign(Place::local(dst), Rvalue::Use(copy(src)));
            }
        }
    };
    let fits = |e: &mut Emit, run_k0: Local| {
        let run_end = bin(e, Ty::Usize, BinOp::Add, copy(run_k0), cu(unroll));
        bin(e, Ty::Bool, BinOp::Le, copy(run_end), copy(k))
    };

    // The prologue: the first `ahead` runs, each loaded when it fits.
    let mut prologue_blocks = Vec::new();
    for step in 0..ahead {
        let guard_idx = prologue_start + 2 * step as u32;
        let next = if step + 1 < ahead {
            guard_idx + 2
        } else {
            head
        };
        let mut check = Emit::new(&mut al);
        let run_k0 = if step == 0 {
            k0
        } else {
            bin(
                &mut check,
                Ty::Usize,
                BinOp::Add,
                copy(k0),
                cu(step * trip_stride),
            )
        };
        let run_fits = fits(&mut check, run_k0);
        prologue_blocks.push(BasicBlock {
            statements: check.take(),
            terminator: guard(run_fits, next, guard_idx + 1),
        });
        let mut issue = Emit::new(&mut al);
        let loaded: Vec<Vec<Local>> = weight_rows
            .iter()
            .map(|&row| read.load(&mut issue, at(row, run_k0, unroll)))
            .collect();
        declare(issue.parts().0, &mut carried, &loaded);
        store_slot(&mut issue, &carried[step], &loaded);
        prologue_blocks.push(BasicBlock {
            statements: issue.take(),
            terminator: Terminator::Goto {
                target: BlockId { index: next },
            },
        });
    }
    for state in carried.iter().flatten().flatten() {
        let ty = al.locals[state.index as usize].ty.clone();
        init_stmts.push(Statement::Assign(
            Place::local(*state),
            Rvalue::Use(Operand::Const(zero(&ty))),
        ));
    }
    let bb2 = BasicBlock {
        statements: init_stmts,
        terminator: Terminator::Goto {
            target: BlockId {
                index: if ahead == 0 { head } else { prologue_start },
            },
        },
    };

    // `partial += activation[b, k] * w[k]` over the `count` elements `k0 ..`, left to right, for each row. A full run
    // under `GemvActivation::Vec4` reads its activation values in groups of four (`act_base` is a multiple of four
    // there: `K`, `k0` and `unroll` all are); the values and the order they meet the weights are the same. Each
    // activation value is read once and meets every row's weight.
    let accumulate = |e: &mut Emit, read: &mut R, count: usize, loaded: &[Vec<Local>]| {
        let values: Vec<Vec<Local>> = weight_rows
            .iter()
            .zip(loaded)
            .map(|(&row, state)| read.decode(e, at(row, k0, count), state))
            .collect();
        let act_base = bin(e, Ty::Usize, BinOp::Add, copy(act_row_base), copy(k0));
        // The vector groups are read before the run's products; a scalar run reads each value as it meets it.
        let grouped: Option<Vec<Local>> =
            (activation_read == GemvActivation::Vec4 && count == unroll).then(|| {
                (0..count / ACTIVATION_GROUP)
                    .flat_map(|g| {
                        // The group starts at element `act_base + 4g`, a multiple of four; the activation binding
                        // starts on `BUFFER_PARAM_ALIGN`, so the group's address is 16-byte aligned.
                        let group_at = bin(
                            e,
                            Ty::Usize,
                            BinOp::Add,
                            copy(act_base),
                            cu(g * ACTIVATION_GROUP),
                        );
                        let group = e.let_(
                            Ty::Vec {
                                elem: Box::new(Ty::F32),
                                lanes: ACTIVATION_GROUP as u32,
                            },
                            Rvalue::VectorLoad {
                                place: elem(activation, group_at),
                                align_bytes: ACTIVATION_GROUP_ALIGN,
                            },
                        );
                        (0..ACTIVATION_GROUP as u32)
                            .map(|lane| {
                                e.let_(
                                    Ty::F32,
                                    Rvalue::VectorExtract {
                                        vector: copy(group),
                                        lane,
                                    },
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            });
        let mut sums = partials.clone();
        for j in 0..count {
            let act = match &grouped {
                Some(acts) => acts[j],
                None => {
                    let act_idx = bin(e, Ty::Usize, BinOp::Add, copy(act_base), cu(j));
                    e.let_(
                        Ty::F32,
                        Rvalue::Use(Operand::Copy(elem(activation, act_idx))),
                    )
                }
            };
            for (sum, row_values) in sums.iter_mut().zip(&values) {
                let product = bin(e, Ty::F32, BinOp::Mul, copy(act), copy(row_values[j]));
                *sum = bin(e, Ty::F32, BinOp::Add, copy(*sum), copy(product));
            }
        }
        for (&partial, &sum) in partials.iter().zip(&sums) {
            e.assign(Place::local(partial), Rvalue::Use(copy(sum)));
        }
    };

    // The loop head: full runs of `unroll` elements while `k0 + unroll <= K`.
    let mut guard_e = Emit::new(&mut al);
    let run_fits = fits(&mut guard_e, k0);
    let head_block = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(
            run_fits,
            tail_guard,
            prefetch.map_or(body, |(guard_idx, _)| guard_idx),
        ),
    };
    // The prefetch: the run `ahead` trips on is loaded into the farthest slot when it fits.
    let prefetch_blocks = prefetch.map(|(guard_idx, load_idx)| {
        let mut check = Emit::new(&mut al);
        let far_k0 = bin(
            &mut check,
            Ty::Usize,
            BinOp::Add,
            copy(k0),
            cu(ahead * trip_stride),
        );
        let far_fits = fits(&mut check, far_k0);
        let check_block = BasicBlock {
            statements: check.take(),
            terminator: guard(far_fits, body, load_idx),
        };
        let mut issue = Emit::new(&mut al);
        let loaded: Vec<Vec<Local>> = weight_rows
            .iter()
            .map(|&row| read.load(&mut issue, at(row, far_k0, unroll)))
            .collect();
        store_slot(&mut issue, &carried[ahead], &loaded);
        debug_assert_eq!(guard_idx + 1, load_idx);
        let load_block = BasicBlock {
            statements: issue.take(),
            terminator: Terminator::Goto {
                target: BlockId { index: body },
            },
        };
        [check_block, load_block]
    });
    // The body: one full run decoded and summed, the carried runs rotated down a slot.
    let mut run = Emit::new(&mut al);
    if ahead == 0 {
        let loaded: Vec<Vec<Local>> = weight_rows
            .iter()
            .map(|&row| read.load(&mut run, at(row, k0, unroll)))
            .collect();
        accumulate(&mut run, &mut read, unroll, &loaded);
    } else {
        accumulate(&mut run, &mut read, unroll, &carried[0]);
        for slot in 0..ahead {
            let (current, next) = (&carried[slot], &carried[slot + 1]);
            store_slot(&mut run, current, next);
        }
    }
    let next_k0 = bin(&mut run, Ty::Usize, BinOp::Add, copy(k0), cu(trip_stride));
    run.assign(Place::local(k0), Rvalue::Use(copy(next_k0)));
    let body_block = BasicBlock {
        statements: run.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: head },
        },
    };

    // The partial run (fewer than `unroll` elements) past the last full one, if any.
    let mut tail_check = Emit::new(&mut al);
    let in_k = bin(&mut tail_check, Ty::Bool, BinOp::Lt, copy(k0), copy(k));
    let tail_guard_block = BasicBlock {
        statements: tail_check.take(),
        terminator: guard(in_k, lds_write, tail_block),
    };
    let mut tail = Emit::new(&mut al);
    let loaded: Vec<Vec<Local>> = weight_rows
        .iter()
        .map(|&row| read.load(&mut tail, at(row, k0, 1)))
        .collect();
    accumulate(&mut tail, &mut read, 1, &loaded);
    let next_k0 = bin(&mut tail, Ty::Usize, BinOp::Add, copy(k0), cu(1));
    tail.assign(Place::local(k0), Rvalue::Use(copy(next_k0)));
    let tail_run_block = BasicBlock {
        statements: tail.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: tail_guard },
        },
    };

    let final_branch = round_start + 3 * column_tree_rounds(lanes_per_col).len() as u32;
    let mut publish = Emit::new(&mut al);
    for (r, &partial) in partials.iter().enumerate() {
        let value = match poison {
            Some(poison) => bin(
                &mut publish,
                Ty::F32,
                BinOp::Add,
                copy(partial),
                copy(poison),
            ),
            None => partial,
        };
        publish.write_lds(r as u8, copy(lds_slot), copy(value));
    }
    let lds_write_block = BasicBlock {
        statements: publish.take(),
        terminator: Terminator::Barrier {
            target: BlockId { index: round_start },
        },
    };

    let mut blocks = vec![bb0, bb1, bb2];
    blocks.extend(prologue_blocks);
    blocks.push(head_block);
    blocks.extend(prefetch_blocks.into_iter().flatten());
    blocks.extend([
        body_block,
        tail_guard_block,
        tail_run_block,
        lds_write_block,
    ]);
    debug_assert_eq!(blocks.len() as u32, round_start);
    emit_column_tree(
        &mut al,
        &mut blocks,
        lane,
        launch,
        round_start,
        final_branch,
    );

    // final_branch: lane `c < cols` stores column `(group*cols + c)*rows + r` for each row `r` that is a real
    // output. Per row a guard block then a store block; a guard that fails falls to the next row's.
    let returns = final_branch + 2 * rows as u32;
    for r in 0..rows {
        let guard_idx = final_branch + 2 * r as u32;
        let mut store_guard = Emit::new(&mut al);
        let column = bin(
            &mut store_guard,
            Ty::Usize,
            BinOp::Add,
            copy(group_base),
            copy(lane),
        );
        let store_col = row_of(&mut store_guard, column, rows, r);
        let is_column_lane = bin(&mut store_guard, Ty::Bool, BinOp::Lt, copy(lane), cu(cols));
        let in_out = bin(
            &mut store_guard,
            Ty::Bool,
            BinOp::Lt,
            copy(store_col),
            copy(out_total),
        );
        let stores = bin(
            &mut store_guard,
            Ty::Bool,
            BinOp::BitAnd,
            copy(is_column_lane),
            copy(in_out),
        );
        let after = if r + 1 < rows { guard_idx + 2 } else { returns };
        blocks.push(BasicBlock {
            statements: store_guard.take(),
            terminator: guard(stores, after, guard_idx + 1),
        });
        let mut store = Emit::new(&mut al);
        let sum = store.read_lds(Ty::F32, r as u8, copy(lane));
        let value = match epilogue {
            Some(BoundEpilogue {
                epilogue,
                leaves,
                lanes,
            }) => emit_epilogue(
                &mut store,
                epilogue,
                conversions,
                store_col,
                leaves,
                lanes,
                sum,
            ),
            None => sum,
        };
        store.assign(elem(out, store_col), Rvalue::Use(copy(value)));
        blocks.push(BasicBlock {
            statements: store.take(),
            terminator: if r + 1 < rows {
                Terminator::Goto {
                    target: BlockId { index: after },
                }
            } else {
                Terminator::Return
            },
        });
    }
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });

    let mut body = Body::new(name, param_count, al.locals, blocks);
    body.workgroup_size = [width as u32, 1, 1];
    body.workgroup_locals = (0..rows)
        .map(|_| poot_kernel_ir::WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: width as u32,
        })
        .collect();
    body
}
/// Output column `column * rows + r`: row `r` of group-column `column`, as a local.
fn row_of(e: &mut Emit, column: Local, rows: usize, r: usize) -> Local {
    if rows == 1 {
        return column;
    }
    let first = bin(e, Ty::Usize, BinOp::Mul, copy(column), cu(rows));
    if r == 0 {
        first
    } else {
        bin(e, Ty::Usize, BinOp::Add, copy(first), cu(r))
    }
}

/// The `(half, thresh)` rounds of a ceil-halving tree over `live` partials: round `i` folds
/// partial `p + half` into partial `p` for every `p < thresh`, leaving `half` live, until one is
/// left. Exact for any `live`, not just a power of two (the unpaired partial rides forward).
fn column_tree_rounds(live: usize) -> Vec<(usize, usize)> {
    let mut rounds = Vec::new();
    let mut live = live;
    while live > 1 {
        let half = live.div_ceil(2);
        rounds.push((half, live - half));
        live = half;
    }
    rounds
}

/// [`gemv_body`]'s per-column fold: `cols` independent trees over the `L = width / cols` sub-lane
/// partials, which sit at LDS stride `cols` (`LDS[s * cols + c]`), so every column folds in the
/// same round with one shared barrier. Round `(half, thresh)` has thread `t < thresh * cols` add
/// `LDS[t + half * cols]` into `LDS[t]` (the same column, sub-lane `s + half`); after the last round
/// `LDS[c]` holds column `c`'s sum and control reaches `final_branch`. Each of a lane's `rows` rows folds in its own
/// workgroup array, in the same round. Three blocks per round (guard, combine, barrier); every lane reaches every
/// barrier.
fn emit_column_tree(
    al: &mut Alloc,
    blocks: &mut Vec<BasicBlock>,
    lane: Local,
    launch: GemvLaunch,
    round_start: u32,
    final_branch: u32,
) {
    let rounds = column_tree_rounds(launch.lanes_per_col());
    for (i, &(half, thresh)) in rounds.iter().enumerate() {
        let guard_idx = round_start + 3 * i as u32;
        let next = if i + 1 < rounds.len() {
            guard_idx + 3
        } else {
            final_branch
        };
        let mut check = Emit::new(al);
        let active = bin(
            &mut check,
            Ty::Bool,
            BinOp::Lt,
            copy(lane),
            cu(thresh * launch.cols),
        );
        blocks.push(BasicBlock {
            statements: check.take(),
            terminator: guard(active, guard_idx + 2, guard_idx + 1),
        });
        let mut combine = Emit::new(al);
        let other_idx = bin(
            &mut combine,
            Ty::Usize,
            BinOp::Add,
            copy(lane),
            cu(half * launch.cols),
        );
        for array in 0..launch.rows as u8 {
            let mine = combine.read_lds(Ty::F32, array, copy(lane));
            let other = combine.read_lds(Ty::F32, array, copy(other_idx));
            let sum = bin(&mut combine, Ty::F32, BinOp::Add, copy(mine), copy(other));
            combine.write_lds(array, copy(lane), copy(sum));
        }
        blocks.push(BasicBlock {
            statements: combine.take(),
            terminator: Terminator::Goto {
                target: BlockId {
                    index: guard_idx + 2,
                },
            },
        });
        blocks.push(BasicBlock {
            statements: vec![],
            terminator: Terminator::Barrier {
                target: BlockId { index: next },
            },
        });
    }
}
