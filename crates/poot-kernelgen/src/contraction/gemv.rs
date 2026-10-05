//! The [`super::Schedule::Gemv`] loop nest over a weight whose rows are the output columns (`[N, K]`, `K`
//! contiguous): one body for every operand load. The packed front ends and the dense loader
//! ([`super::dense`]) each supply only the read of a run of one row's weight values.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, Operand, Place, Rvalue, Statement,
    Terminator, Ty,
};

use crate::KernelGenError;
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, elem, guard};

/// A [`super::Schedule::Gemv`]'s launch shape, checked once before emission: every extent at least one and
/// `width` a multiple of `cols`, so each column owns exactly `width / cols` lanes.
#[derive(Clone, Copy)]
pub(crate) struct GemvLaunch {
    width: usize,
    cols: usize,
    unroll: usize,
}

impl GemvLaunch {
    pub(crate) fn new(width: u32, cols: u32, unroll: u32) -> Result<Self, KernelGenError> {
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
        Ok(Self {
            width: width as usize,
            cols: cols as usize,
            unroll: unroll as usize,
        })
    }

    /// Lanes that share one output column.
    fn lanes_per_col(self) -> usize {
        self.width / self.cols
    }
}

/// The [`super::Schedule::Gemv`] loop nest, shared by every operand load; `decode_run(e, weight_row,
/// k0, count, k, out_total)` emits the `count` logical weight values `k0 .. k0 + count` of one row,
/// and is only ever called with `k0` a multiple of `count` (a full run, or a one-element tail step).
///
/// Workgroup `g` owns columns `g*cols .. g*cols + cols`. Lane `lane` serves column `c = lane / L`
/// (`L = width / cols` lanes per column) as sub-lane `s = lane % L`, so a column's lanes are
/// adjacent and read one contiguous run of its weight row per trip: sub-lane `s` reads the `unroll`
/// elements `k0 .. k0 + unroll` with `k0 = trip * L * unroll + s * unroll`. Full runs take the
/// unrolled body; the one partial run a `K` that is not a multiple of `L * unroll` leaves is
/// finished element by element, so no `K` is refused. A column past the output (the last
/// workgroup's tail) reads the last real column instead and never stores, keeping every lane at
/// every barrier. Partials land in LDS at `s * cols + c`, so the per-column fold is a tree of
/// stride `cols` over the `L` sub-lanes and leaves column `c`'s sum at `LDS[c]`.
pub(crate) fn gemv_body(
    name: &str,
    mut al: Alloc,
    param_count: u32,
    [activation, metadata, out]: [Local; 3],
    launch: GemvLaunch,
    mut decode_run: impl FnMut(&mut Emit, Local, Local, usize, Local, Local) -> Vec<Local>,
) -> Body {
    let GemvLaunch {
        width,
        cols,
        unroll,
    } = launch;
    let lanes_per_col = launch.lanes_per_col();
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
    let weight_row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Min,
        copy(col_flat),
        copy(last_col),
    );
    let b = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(weight_row),
        copy(out_per_block),
    );
    let act_row_base = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(b), copy(k));
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

    let partial = al.add(Ty::F32, true);
    let k0 = al.add(Ty::Usize, true);
    init_stmts.push(Statement::Assign(
        Place::local(partial),
        Rvalue::Use(Operand::Const(poot_kernel_ir::Constant::F32(0.0))),
    ));
    init_stmts.push(Statement::Assign(
        Place::local(k0),
        Rvalue::Use(copy(first_k)),
    ));
    let bb2 = BasicBlock {
        statements: init_stmts,
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };

    // `partial += activation[b, k] * w[k]` over the `count` elements `k0 ..`, left to right.
    let mut accumulate = |e: &mut Emit, count: usize| {
        let values = decode_run(e, weight_row, k0, count, k, out_total);
        let act_base = bin(e, Ty::Usize, BinOp::Add, copy(act_row_base), copy(k0));
        let mut sum = partial;
        for (j, value) in values.into_iter().enumerate() {
            let act_idx = bin(e, Ty::Usize, BinOp::Add, copy(act_base), cu(j));
            let act = e.let_(
                Ty::F32,
                Rvalue::Use(Operand::Copy(elem(activation, act_idx))),
            );
            let product = bin(e, Ty::F32, BinOp::Mul, copy(act), copy(value));
            sum = bin(e, Ty::F32, BinOp::Add, copy(sum), copy(product));
        }
        e.assign(Place::local(partial), Rvalue::Use(copy(sum)));
    };

    // bb3/bb4: full runs of `unroll` elements while `k0 + unroll <= K`.
    let mut guard_e = Emit::new(&mut al);
    let run_end = bin(&mut guard_e, Ty::Usize, BinOp::Add, copy(k0), cu(unroll));
    let run_fits = bin(&mut guard_e, Ty::Bool, BinOp::Le, copy(run_end), copy(k));
    let bb3 = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(run_fits, 5, 4),
    };
    let mut run = Emit::new(&mut al);
    accumulate(&mut run, unroll);
    let next_k0 = bin(
        &mut run,
        Ty::Usize,
        BinOp::Add,
        copy(k0),
        cu(lanes_per_col * unroll),
    );
    run.assign(Place::local(k0), Rvalue::Use(copy(next_k0)));
    let bb4 = BasicBlock {
        statements: run.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };

    // bb5/bb6: the partial run (fewer than `unroll` elements) past the last full one, if any.
    let mut tail_guard = Emit::new(&mut al);
    let in_k = bin(&mut tail_guard, Ty::Bool, BinOp::Lt, copy(k0), copy(k));
    let bb5 = BasicBlock {
        statements: tail_guard.take(),
        terminator: guard(in_k, 7, 6),
    };
    let mut tail = Emit::new(&mut al);
    accumulate(&mut tail, 1);
    let next_k0 = bin(&mut tail, Ty::Usize, BinOp::Add, copy(k0), cu(1));
    tail.assign(Place::local(k0), Rvalue::Use(copy(next_k0)));
    let bb6 = BasicBlock {
        statements: tail.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: 5 },
        },
    };

    let round_start: u32 = 8;
    let final_branch = round_start + 3 * column_tree_rounds(lanes_per_col).len() as u32;
    let store_idx = final_branch + 1;
    let early_return_idx = final_branch + 2;
    let bb7 = BasicBlock {
        statements: vec![Statement::WorkgroupLocalWrite {
            idx: copy(lds_slot),
            value: copy(partial),
            array: 0,
        }],
        terminator: Terminator::Barrier {
            target: BlockId { index: round_start },
        },
    };
    let mut blocks = vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7];
    emit_column_tree(
        &mut al,
        &mut blocks,
        lane,
        launch,
        round_start,
        final_branch,
    );

    // final_branch: lane `c < cols` stores column `group*cols + c` when it is a real output.
    let mut store_guard = Emit::new(&mut al);
    let store_col = bin(
        &mut store_guard,
        Ty::Usize,
        BinOp::Add,
        copy(group_base),
        copy(lane),
    );
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
    blocks.push(BasicBlock {
        statements: store_guard.take(),
        terminator: guard(stores, early_return_idx, store_idx),
    });
    let mut store = Emit::new(&mut al);
    let sum = store.read_lds(Ty::F32, 0, copy(lane));
    store.assign(elem(out, store_col), Rvalue::Use(copy(sum)));
    blocks.push(BasicBlock {
        statements: store.take(),
        terminator: Terminator::Return,
    });
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });

    let mut body = Body::new(name, param_count, al.locals, blocks);
    body.workgroup_size = [width as u32, 1, 1];
    body.workgroup_locals = vec![poot_kernel_ir::WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: width as u32,
    }];
    body
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
/// `LDS[c]` holds column `c`'s sum and control reaches `final_branch`. Three blocks per round
/// (guard, combine, barrier); every lane reaches every barrier.
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
        let mine = combine.read_lds(Ty::F32, 0, copy(lane));
        let other = combine.read_lds(Ty::F32, 0, copy(other_idx));
        let sum = bin(&mut combine, Ty::F32, BinOp::Add, copy(mine), copy(other));
        combine.write_lds(0, copy(lane), copy(sum));
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
