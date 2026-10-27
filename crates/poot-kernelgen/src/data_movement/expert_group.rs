use super::{
    Alloc, BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty, copy, elem, guard, ld, local, slice_dtype,
};

/// The expert-grouping permutation of `ids [M]` (the MoE rows' expert ids): `out [2 * M]`, `out[r]` the row
/// that sorted position `r` holds (`order`) and `out[M + m]` the sorted position of row `m` (`dest`, the
/// inverse), rows sorted by expert ascending and by row within one expert. A row whose id is outside
/// `0..experts` falls in a last bucket after every expert, so both halves stay permutations and an invalid id
/// reaches the indexed contraction that fails it closed (NaN row) instead of being dropped or misrouted.
///
/// One thread per bucket (`experts + 1`): count the rows of lower buckets (the bucket's start offset), then
/// walk the rows in order and place those of its own bucket. No atomics and no barriers; the rows are read
/// `2 * M` times per thread, the llama.cpp `mm_ids_helper` scheme (one block per expert scanning every id).
///
/// Measured cost (RTX PRO 6000 Blackwell, PTX, Flash-Next Q2 with 512 experts, a 1024-token chunk so `M = 10240`):
/// 0.74 ms per launch, 35.6 ms per chunk over the 48 layers, 1.0 percent of a 3.48 s chunk. The serial id reads
/// per thread are not the bottleneck at that size; they grow with `M`, so a chunk several times larger would
/// want the scan split over a workgroup per expert.
pub(crate) fn expert_group(name: &str, experts: usize) -> Body {
    // _1 ids [M] (i32), _2 out [2 * M] (i32)
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::I32, false), false),
        ld(slice_dtype(Ty::I32, true), true),
    ]);
    let (ids, out) = (local(1), local(2));
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let ci = |x: usize| Operand::Const(Constant::I32(x as i32));

    let bucket = al.add(Ty::Usize, false);
    let in_range = al.add(Ty::Bool, false);
    let rows = al.add(Ty::Usize, false);
    let bucket_i = al.add(Ty::I32, false);
    let start = al.add(Ty::Usize, true); // the next sorted position this bucket fills
    let j = al.add(Ty::Usize, true); // the row under the scan
    let more = al.add(Ty::Bool, false);
    let less = al.add(Ty::Bool, false);
    let less_u = al.add(Ty::Usize, false);
    let start_new = al.add(Ty::Usize, false);
    let j_new = al.add(Ty::Usize, false);
    let own = al.add(Ty::Bool, false);
    let row_i = al.add(Ty::I32, false);
    let start_i = al.add(Ty::I32, false);
    let dest_at = al.add(Ty::Usize, false);

    // The bucket of row `j`: its id when in `0..experts`, else `experts`.
    let key_of = |al: &mut Alloc, statements: &mut Vec<Statement>| -> Local {
        let id = al.add(Ty::I32, false);
        let negative = al.add(Ty::Bool, false);
        let over = al.add(Ty::Bool, false);
        let bad = al.add(Ty::Bool, false);
        let bad_i = al.add(Ty::I32, false);
        let gap = al.add(Ty::I32, false);
        let shift = al.add(Ty::I32, false);
        let key = al.add(Ty::I32, false);
        let bin = |op, a: Operand, b: Operand| Rvalue::BinaryOp(op, a, b);
        statements.extend([
            Statement::Assign(Place::local(id), Rvalue::Use(copy(elem(ids, j)))),
            Statement::Assign(
                Place::local(negative),
                bin(BinOp::Lt, copy(Place::local(id)), ci(0)),
            ),
            Statement::Assign(
                Place::local(over),
                bin(BinOp::Ge, copy(Place::local(id)), ci(experts)),
            ),
            Statement::Assign(
                Place::local(bad),
                bin(
                    BinOp::BitOr,
                    copy(Place::local(negative)),
                    copy(Place::local(over)),
                ),
            ),
            Statement::Assign(
                Place::local(bad_i),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(bad)),
                },
            ),
            Statement::Assign(
                Place::local(gap),
                bin(BinOp::Sub, ci(experts), copy(Place::local(id))),
            ),
            Statement::Assign(
                Place::local(shift),
                bin(
                    BinOp::Mul,
                    copy(Place::local(bad_i)),
                    copy(Place::local(gap)),
                ),
            ),
            Statement::Assign(
                Place::local(key),
                bin(
                    BinOp::Add,
                    copy(Place::local(id)),
                    copy(Place::local(shift)),
                ),
            ),
        ]);
        key
    };

    // bb0: bucket = thread index.
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(bucket),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    // bb1: lanes past the last bucket return before any access.
    let bb1 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(in_range),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(bucket)), cu(experts + 1)),
        )],
        terminator: guard(in_range, 10, 2),
    };
    // bb2: rows = M; start = 0; j = 0.
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(rows), Rvalue::Len(Place::local(ids))),
            Statement::Assign(
                Place::local(bucket_i),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(bucket)),
                },
            ),
            Statement::Assign(Place::local(start), Rvalue::Use(cu(0))),
            Statement::Assign(Place::local(j), Rvalue::Use(cu(0))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb3/bb4: start = the number of rows whose bucket is below this one.
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(more),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), copy(Place::local(rows))),
        )],
        terminator: guard(more, 5, 4),
    };
    let mut count = Vec::new();
    let key = key_of(&mut al, &mut count);
    count.extend([
        Statement::Assign(
            Place::local(less),
            Rvalue::BinaryOp(
                BinOp::Lt,
                copy(Place::local(key)),
                copy(Place::local(bucket_i)),
            ),
        ),
        Statement::Assign(
            Place::local(less_u),
            Rvalue::Cast {
                to: Ty::Usize,
                operand: copy(Place::local(less)),
            },
        ),
        Statement::Assign(
            Place::local(start_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(start)),
                copy(Place::local(less_u)),
            ),
        ),
        Statement::Assign(
            Place::local(start),
            Rvalue::Use(copy(Place::local(start_new))),
        ),
        Statement::Assign(
            Place::local(j_new),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
        ),
        Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
    ]);
    let bb4 = BasicBlock {
        statements: count,
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb5: restart the scan to place this bucket's rows.
    let bb5 = BasicBlock {
        statements: vec![Statement::Assign(Place::local(j), Rvalue::Use(cu(0)))],
        terminator: Terminator::Goto {
            target: BlockId { index: 6 },
        },
    };
    let bb6 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(more),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), copy(Place::local(rows))),
        )],
        terminator: guard(more, 10, 7),
    };
    // bb7: test the row's bucket; bb8 places it; bb9 advances.
    let mut probe = Vec::new();
    let key = key_of(&mut al, &mut probe);
    probe.push(Statement::Assign(
        Place::local(own),
        Rvalue::BinaryOp(
            BinOp::Eq,
            copy(Place::local(key)),
            copy(Place::local(bucket_i)),
        ),
    ));
    let bb7 = BasicBlock {
        statements: probe,
        terminator: guard(own, 9, 8),
    };
    let bb8 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(row_i),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(j)),
                },
            ),
            Statement::Assign(elem(out, start), Rvalue::Use(copy(Place::local(row_i)))),
            Statement::Assign(
                Place::local(start_i),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(start)),
                },
            ),
            Statement::Assign(
                Place::local(dest_at),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(rows)), copy(Place::local(j))),
            ),
            Statement::Assign(elem(out, dest_at), Rvalue::Use(copy(Place::local(start_i)))),
            Statement::Assign(
                Place::local(start_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(start)), cu(1)),
            ),
            Statement::Assign(
                Place::local(start),
                Rvalue::Use(copy(Place::local(start_new))),
            ),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 9 },
        },
    };
    let bb9 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(j_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 6 },
        },
    };
    let bb10 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(
        name,
        2,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10],
    )
}
