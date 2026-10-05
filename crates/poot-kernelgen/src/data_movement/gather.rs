use super::*;

/// Axis-0 gather with independently typed data and index storage. Exact-I32 producer chains use an I32
/// index; legacy token/index graphs use the f32 convention (`index_dt = Ty::F32`, card 671: the
/// `gather_axis0_dt` wrapper for that case moved to `poot_test_util::kernel_fixtures`, its only caller).
pub fn gather_axis0_index_dt(name: &str, dt: Ty, index_dt: Ty, rest: usize) -> Body {
    // _1 data, _2 index, _3 out
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // data (dt)
        ld(slice_dtype(index_dt.clone(), false), false), // index
        ld(slice_dtype(dt.clone(), true), true),   // out (dt)
    ]);
    let (data, index, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let s = al.add(Ty::Usize, false);
    let r = al.add(Ty::Usize, false);
    let index_value = al.add(index_dt, false);
    let row = al.add(Ty::Usize, false);
    let base = al.add(Ty::Usize, false);
    let src = al.add(Ty::Usize, false);
    let val = al.add(dt.clone(), false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(s),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(Place::local(index_value), Rvalue::Use(copy(elem(index, s)))),
            Statement::Assign(
                Place::local(row),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(index_value)),
                },
            ),
            Statement::Assign(
                Place::local(base),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(src),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(base)), copy(Place::local(r))),
            ),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(data, src)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3])
}

/// General-axis gather with independently typed data and index storage; see [`gather_axis0_index_dt`]
/// for the index contract.
pub fn gather_axis_index_dt(
    name: &str,
    dt: Ty,
    index_dt: Ty,
    inner: usize,
    axis_len: usize,
    idx_numel: usize,
) -> Body {
    // _1 data, _2 index, _3 out
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // data (dt)
        ld(slice_dtype(index_dt.clone(), false), false), // index
        ld(slice_dtype(dt.clone(), true), true),   // out (dt)
    ]);
    let (data, index, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let inner_i = al.add(Ty::Usize, false);
    let mid = al.add(Ty::Usize, false);
    let gpos = al.add(Ty::Usize, false);
    let outer_i = al.add(Ty::Usize, false);
    let index_value = al.add(index_dt, false);
    let axis_pos = al.add(Ty::Usize, false);
    let src = al.add(Ty::Usize, false);
    let t = al.add(Ty::Usize, false);
    let val = al.add(dt.clone(), false);
    let outer_stride = axis_len * inner;
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            // inner_i = i % inner ; mid = i / inner
            Statement::Assign(
                Place::local(inner_i),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(inner)),
            ),
            Statement::Assign(
                Place::local(mid),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(inner)),
            ),
            // gpos = mid % idx_numel ; outer_i = mid / idx_numel
            Statement::Assign(
                Place::local(gpos),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(mid)), cu(idx_numel)),
            ),
            Statement::Assign(
                Place::local(outer_i),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(mid)), cu(idx_numel)),
            ),
            // axis_pos = (usize) index[gpos]
            Statement::Assign(
                Place::local(index_value),
                Rvalue::Use(copy(elem(index, gpos))),
            ),
            Statement::Assign(
                Place::local(axis_pos),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(index_value)),
                },
            ),
            // src = outer_i*outer_stride + axis_pos*inner + inner_i
            Statement::Assign(
                Place::local(src),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(outer_i)), cu(outer_stride)),
            ),
            Statement::Assign(
                Place::local(t),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(axis_pos)), cu(inner)),
            ),
            Statement::Assign(
                Place::local(src),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(src)), copy(Place::local(t))),
            ),
            Statement::Assign(
                Place::local(src),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(src)),
                    copy(Place::local(inner_i)),
                ),
            ),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(data, src)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3])
}
