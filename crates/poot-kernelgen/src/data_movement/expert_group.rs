use poot_kernel_ir::WorkgroupLocalDecl;

use super::{
    Alloc, BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty, copy, elem, guard, ld, local, slice_dtype,
};

/// The lanes of the one workgroup that groups the rows.
const LANES: usize = 256;

/// The block-aligned grouping of `ids [M]` (the MoE rows' expert ids, POOT-1451) into `out`, the
/// [`poot_graph_ir::ExpertGrouping`] layout of `M` rows, `experts` experts and tiles of `block` positions: the padded
/// order (bucket by bucket, each bucket's rows in row order and its run padded with the sentinel `M` to whole
/// tiles, every position past the last run the sentinel), then each tile's bucket (`experts` for the bucket of ids
/// outside `0..experts` and for the empty tiles past the last run). `out`'s length fixes `T = len / (block + 1)`
/// tiles and `P = T * block` positions.
///
/// One workgroup of [`LANES`] lanes, the buckets (`experts + 1`) strided over them: each lane counts its buckets'
/// rows (a scan of the ids per bucket) into workgroup memory; lane 0 turns the counts into each bucket's first
/// position (the padded sizes' prefix sum, and the padded total after the last); each lane then places its buckets'
/// rows in row order (a second scan per bucket, the llama.cpp `mm_ids_helper` scheme), pads their runs and writes
/// their tiles; finally the lanes fill the positions and tiles past the padded total. No atomics: every write has
/// one owner, so the grouping is the same on every device and every run.
pub(crate) fn expert_group(name: &str, experts: usize, block: usize) -> Body {
    // _1 ids [M] (i32), _2 out [P + T] (i32)
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::I32, false), false),
        ld(slice_dtype(Ty::I32, true), true),
    ]);
    let (ids, out) = (local(1), local(2));
    let buckets = experts + 1;
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let set = |l: Local, rvalue: Rvalue| Statement::Assign(Place::local(l), rvalue);
    let bin = |op, a: Local, b: Operand| Rvalue::BinaryOp(op, copy(Place::local(a)), b);
    let use_ = |l: Local| Operand::Copy(Place::local(l));
    let cast = |to: Ty, l: Local| Rvalue::Cast {
        to,
        operand: copy(Place::local(l)),
    };
    let goto = |target: u32| Terminator::Goto {
        target: BlockId { index: target },
    };
    let barrier = |target: u32| Terminator::Barrier {
        target: BlockId { index: target },
    };
    const COUNTS: u8 = 0;
    const STARTS: u8 = 1;

    let lane = al.add(Ty::Usize, false);
    let rows = al.add(Ty::Usize, false);
    let words = al.add(Ty::Usize, false);
    let tiles = al.add(Ty::Usize, false);
    let positions = al.add(Ty::Usize, false);
    let sentinel = al.add(Ty::I32, false);
    let b = al.add(Ty::Usize, true); // the bucket under the lane
    let b_i = al.add(Ty::I32, true);
    let j = al.add(Ty::Usize, true); // the row under a scan
    let count = al.add(Ty::Usize, true);
    let acc = al.add(Ty::Usize, true);
    let pos = al.add(Ty::Usize, true);
    let end = al.add(Ty::Usize, true);
    let t = al.add(Ty::Usize, true);
    let t_end = al.add(Ty::Usize, true);
    let total = al.add(Ty::Usize, true);
    let more = al.add(Ty::Bool, false);
    let own = al.add(Ty::Bool, false);
    let word = al.add(Ty::U32, false);
    let at = al.add(Ty::Usize, false);
    let value = al.add(Ty::I32, false);

    // The bucket of row `j` as an I32: its id when in `0..experts`, else `experts`.
    let key_of = |al: &mut Alloc, statements: &mut Vec<Statement>| -> Local {
        let id = al.add(Ty::I32, false);
        let negative = al.add(Ty::Bool, false);
        let over = al.add(Ty::Bool, false);
        let bad = al.add(Ty::Bool, false);
        let bad_i = al.add(Ty::I32, false);
        let gap = al.add(Ty::I32, false);
        let shift = al.add(Ty::I32, false);
        let key = al.add(Ty::I32, false);
        let ci = |x: usize| Operand::Const(Constant::I32(x as i32));
        statements.extend([
            set(id, Rvalue::Use(copy(elem(ids, j)))),
            set(negative, bin(BinOp::Lt, id, ci(0))),
            set(over, bin(BinOp::Ge, id, ci(experts))),
            set(bad, bin(BinOp::BitOr, negative, use_(over))),
            set(bad_i, cast(Ty::I32, bad)),
            set(gap, Rvalue::BinaryOp(BinOp::Sub, ci(experts), use_(id))),
            set(shift, bin(BinOp::Mul, bad_i, use_(gap))),
            set(key, bin(BinOp::Add, id, use_(shift))),
        ]);
        key
    };
    // `count` rounded up to whole tiles, into `into`.
    let padded = |al: &mut Alloc, into: Local, count: Local| -> Vec<Statement> {
        let up = al.add(Ty::Usize, false);
        let whole = al.add(Ty::Usize, false);
        vec![
            set(up, bin(BinOp::Add, count, cu(block - 1))),
            set(whole, bin(BinOp::Div, up, cu(block))),
            set(into, bin(BinOp::Mul, whole, cu(block))),
        ]
    };
    let lds_read = |into: Local, array: u8, idx: Local| -> Vec<Statement> {
        vec![
            set(
                word,
                Rvalue::WorkgroupLocalRead {
                    idx: use_(idx),
                    array,
                },
            ),
            set(into, cast(Ty::Usize, word)),
        ]
    };
    let lds_write = |array: u8, idx: Operand, from: Local| -> Vec<Statement> {
        vec![
            set(word, cast(Ty::U32, from)),
            Statement::WorkgroupLocalWrite {
                idx,
                value: use_(word),
                array,
            },
        ]
    };
    let store = |index: Local, from: Local| -> Vec<Statement> {
        vec![
            set(value, cast(Ty::I32, from)),
            Statement::Assign(elem(out, index), Rvalue::Use(use_(value))),
        ]
    };

    let mut blocks = Vec::new();
    let mut push = |statements: Vec<Statement>, terminator: Terminator| {
        blocks.push(BasicBlock {
            statements,
            terminator,
        });
    };
    // 0: the lane.
    push(
        vec![],
        Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 1 },
        },
    );
    // 1: the extents; the lane's first bucket.
    push(
        vec![
            set(rows, Rvalue::Len(Place::local(ids))),
            set(words, Rvalue::Len(Place::local(out))),
            set(tiles, bin(BinOp::Div, words, cu(block + 1))),
            set(positions, bin(BinOp::Mul, tiles, cu(block))),
            set(sentinel, cast(Ty::I32, rows)),
            set(b, Rvalue::Use(use_(lane))),
        ],
        goto(2),
    );
    // 2-6: count each of the lane's buckets.
    push(
        vec![set(more, bin(BinOp::Lt, b, cu(buckets)))],
        guard(more, 7, 3),
    );
    push(
        vec![
            set(count, Rvalue::Use(cu(0))),
            set(j, Rvalue::Use(cu(0))),
            set(b_i, cast(Ty::I32, b)),
        ],
        goto(4),
    );
    push(
        vec![set(more, bin(BinOp::Lt, j, use_(rows)))],
        guard(more, 6, 5),
    );
    let mut probe = Vec::new();
    let key = key_of(&mut al, &mut probe);
    let own_u = al.add(Ty::Usize, false);
    probe.extend([
        set(own, bin(BinOp::Eq, key, use_(b_i))),
        set(own_u, cast(Ty::Usize, own)),
        set(count, bin(BinOp::Add, count, use_(own_u))),
        set(j, bin(BinOp::Add, j, cu(1))),
    ]);
    push(probe, goto(4));
    let mut stored = lds_write(COUNTS, use_(b), count);
    stored.push(set(b, bin(BinOp::Add, b, cu(LANES))));
    push(stored, goto(2));
    // 7: every count is written.
    push(vec![], barrier(8));
    // 8-12: lane 0 turns the counts into each bucket's first position and the padded total.
    push(
        vec![set(own, bin(BinOp::Eq, lane, cu(0)))],
        guard(own, 13, 9),
    );
    push(
        vec![set(acc, Rvalue::Use(cu(0))), set(b, Rvalue::Use(cu(0)))],
        goto(10),
    );
    push(
        vec![set(more, bin(BinOp::Lt, b, cu(buckets)))],
        guard(more, 12, 11),
    );
    let mut prefix = lds_write(STARTS, use_(b), acc);
    prefix.extend(lds_read(count, COUNTS, b));
    prefix.extend(padded(&mut al, end, count));
    prefix.extend([
        set(acc, bin(BinOp::Add, acc, use_(end))),
        set(b, bin(BinOp::Add, b, cu(1))),
    ]);
    push(prefix, goto(10));
    push(lds_write(STARTS, cu(buckets), acc), goto(13));
    // 13: every first position is written.
    push(vec![], barrier(14));
    // 14-26: place each of the lane's buckets: its rows in row order, its padding, its tiles.
    push(vec![set(b, Rvalue::Use(use_(lane)))], goto(15));
    push(
        vec![set(more, bin(BinOp::Lt, b, cu(buckets)))],
        guard(more, 27, 16),
    );
    let mut start = lds_read(pos, STARTS, b);
    start.extend(lds_read(count, COUNTS, b));
    start.extend(padded(&mut al, end, count));
    start.extend([
        set(end, bin(BinOp::Add, end, use_(pos))),
        set(t, bin(BinOp::Div, pos, cu(block))),
        set(j, Rvalue::Use(cu(0))),
        set(b_i, cast(Ty::I32, b)),
    ]);
    push(start, goto(17));
    push(
        vec![set(more, bin(BinOp::Lt, j, use_(rows)))],
        guard(more, 21, 18),
    );
    let mut probe = Vec::new();
    let key = key_of(&mut al, &mut probe);
    probe.push(set(own, bin(BinOp::Eq, key, use_(b_i))));
    push(probe, guard(own, 20, 19));
    let mut place = store(pos, j);
    place.push(set(pos, bin(BinOp::Add, pos, cu(1))));
    push(place, goto(20));
    push(vec![set(j, bin(BinOp::Add, j, cu(1)))], goto(17));
    push(
        vec![set(more, bin(BinOp::Lt, pos, use_(end)))],
        guard(more, 23, 22),
    );
    push(
        vec![
            Statement::Assign(elem(out, pos), Rvalue::Use(use_(sentinel))),
            set(pos, bin(BinOp::Add, pos, cu(1))),
        ],
        goto(21),
    );
    push(vec![set(t_end, bin(BinOp::Div, end, cu(block)))], goto(24));
    push(
        vec![set(more, bin(BinOp::Lt, t, use_(t_end)))],
        guard(more, 26, 25),
    );
    let mut tile = vec![set(at, bin(BinOp::Add, positions, use_(t)))];
    tile.extend(store(at, b));
    tile.push(set(t, bin(BinOp::Add, t, cu(1))));
    push(tile, goto(24));
    push(vec![set(b, bin(BinOp::Add, b, cu(LANES)))], goto(15));
    // 27-32: the positions and tiles past the padded total, strided over the lanes.
    push(
        vec![
            set(
                word,
                Rvalue::WorkgroupLocalRead {
                    idx: cu(buckets),
                    array: STARTS,
                },
            ),
            set(total, cast(Ty::Usize, word)),
            set(pos, bin(BinOp::Add, total, use_(lane))),
        ],
        goto(28),
    );
    push(
        vec![set(more, bin(BinOp::Lt, pos, use_(positions)))],
        guard(more, 30, 29),
    );
    push(
        vec![
            Statement::Assign(elem(out, pos), Rvalue::Use(use_(sentinel))),
            set(pos, bin(BinOp::Add, pos, cu(LANES))),
        ],
        goto(28),
    );
    push(
        vec![
            set(t, bin(BinOp::Div, total, cu(block))),
            set(t, bin(BinOp::Add, t, use_(lane))),
        ],
        goto(31),
    );
    push(
        vec![set(more, bin(BinOp::Lt, t, use_(tiles)))],
        guard(more, 33, 32),
    );
    let mut empty = vec![set(at, bin(BinOp::Add, positions, use_(t)))];
    empty.extend([
        set(
            value,
            Rvalue::Use(Operand::Const(Constant::I32(experts as i32))),
        ),
        Statement::Assign(elem(out, at), Rvalue::Use(use_(value))),
        set(t, bin(BinOp::Add, t, cu(LANES))),
    ]);
    push(empty, goto(31));
    // 33
    push(vec![], Terminator::Return);

    let mut body = Body::new(name, 2, al.locals, blocks);
    body.workgroup_size = [LANES as u32, 1, 1];
    body.workgroup_locals = vec![
        WorkgroupLocalDecl {
            elem_ty: Ty::U32,
            len: buckets as u32,
        },
        WorkgroupLocalDecl {
            elem_ty: Ty::U32,
            len: buckets as u32 + 1,
        },
    ];
    body
}

/// The lanes [`expert_group`]'s one workgroup runs: the launch's total threads.
pub(crate) const fn expert_group_lanes() -> usize {
    LANES
}
