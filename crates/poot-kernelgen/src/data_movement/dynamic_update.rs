use super::*;

/// DynamicUpdateSlice: `out = operand` with `update` written at `[.., idx..idx+extent, ..]` on `axis`.
/// `operand` (input 0) and `out` share `out_shape`; `update` (input 1) matches it except axis extent =
/// `extent`. One thread per output element: if its axis coord is in `[idx, idx+extent)` it reads from
/// `update` (coord shifted by `idx`), else it copies `operand[i]`. The KV-cache slot write.
///
/// `dt` is the storage dtype (spec 024; pure element copy, no casts).
pub fn dyn_update_slice_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    axis: usize,
    idx: usize,
    extent: usize,
) -> Body {
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let mut update_shape = out_shape.to_vec();
    update_shape[axis] = extent;
    let upd_strides = row_major_strides(&update_shape);

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 operand
        ld(slice_dtype(dt.clone(), false), false), // 2 update
        ld(slice_dtype(dt.clone(), true), true),   // 3 out
    ]);
    let (operand, update, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let coord: Vec<Local> = (0..r).map(|_| al.add(Ty::Usize, false)).collect();
    let lt_lo = al.add(Ty::Bool, false);
    let lt_hi = al.add(Ty::Bool, false);
    let acc = al.add(Ty::Usize, true);
    let term = al.add(Ty::Usize, false);
    let acc_new = al.add(Ty::Usize, false);
    let uaxis = al.add(Ty::Usize, false);
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
        terminator: guard(cmp, 6, 2), // out-of-bounds -> bare return (bb6)
    };
    // bb2: coords; lt_lo = coord[axis] < idx. lt_lo==0 (coord>=idx) -> bb3 (check upper) else bb4 (operand)
    let mut s2 = Vec::new();
    for (d, &cd) in coord.iter().enumerate() {
        s2.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
        ));
        s2.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(cd)), cu(out_shape[d])),
        ));
    }
    s2.push(Statement::Assign(
        Place::local(lt_lo),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(coord[axis])), cu(idx)),
    ));
    let bb2 = BasicBlock {
        statements: s2,
        terminator: guard(lt_lo, 3, 4),
    };
    // bb3: lt_hi = coord[axis] < idx+extent. lt_hi==0 (coord>=hi) -> bb4 (operand) else bb5 (update)
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(lt_hi),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(coord[axis])), cu(idx + extent)),
        )],
        terminator: guard(lt_hi, 4, 5),
    };
    // bb4: operand path. operand and out share strides, so out[i] = operand[i].
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(operand, i)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    // bb5: update path. uaxis = coord[axis]-idx; upd_flat = sum coord[d]*upd_strides[d] (axis -> uaxis).
    let mut s5 = vec![
        Statement::Assign(
            Place::local(uaxis),
            Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(coord[axis])), cu(idx)),
        ),
        Statement::Assign(Place::local(acc), Rvalue::Use(cu(0))),
    ];
    for (d, &cd) in coord.iter().enumerate() {
        let src = if d == axis { uaxis } else { cd };
        s5.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(src)), cu(upd_strides[d])),
        ));
        s5.push(Statement::Assign(
            Place::local(acc_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(acc)),
                copy(Place::local(term)),
            ),
        ));
        s5.push(Statement::Assign(
            Place::local(acc),
            Rvalue::Use(copy(Place::local(acc_new))),
        ));
    }
    s5.push(Statement::Assign(
        Place::local(val),
        Rvalue::Use(copy(elem(update, acc))),
    ));
    s5.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(val))),
    ));
    let bb5 = BasicBlock {
        statements: s5,
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6])
}

/// `dyn_update_slice_dynamic_dt` in storage dtype `dt` (spec 024; pure copy, no casts). `index` stays f32.
pub fn dyn_update_slice_dynamic_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    axis: usize,
    extent: usize,
) -> Body {
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let mut update_shape = out_shape.to_vec();
    update_shape[axis] = extent;
    let upd_strides = row_major_strides(&update_shape);

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 operand
        ld(slice_dtype(dt.clone(), false), false), // 2 update
        ld(slice_f32(false), false),               // 3 index (runtime slot index, f32[1])
        ld(slice_dtype(dt.clone(), true), true),   // 4 out
    ]);
    let (operand, update, index, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let zero = al.add(Ty::Usize, false);
    let idxf = al.add(Ty::F32, false);
    let idx = al.add(Ty::Usize, false); // runtime slot index
    let hi = al.add(Ty::Usize, false); // idx + extent
    let coord: Vec<Local> = (0..r).map(|_| al.add(Ty::Usize, false)).collect();
    let lt_lo = al.add(Ty::Bool, false);
    let lt_hi = al.add(Ty::Bool, false);
    let acc = al.add(Ty::Usize, true);
    let term = al.add(Ty::Usize, false);
    let acc_new = al.add(Ty::Usize, false);
    let uaxis = al.add(Ty::Usize, false);
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
        terminator: guard(cmp, 6, 2),
    };
    // bb2: read the runtime index (index[0] as usize), hi = idx + extent, coords, lt_lo = coord[axis] < idx.
    let mut s2 = vec![
        Statement::Assign(Place::local(zero), Rvalue::Use(cu(0))),
        Statement::Assign(Place::local(idxf), Rvalue::Use(copy(elem(index, zero)))),
        Statement::Assign(
            Place::local(idx),
            Rvalue::Cast {
                to: Ty::Usize,
                operand: copy(Place::local(idxf)),
            },
        ),
        Statement::Assign(
            Place::local(hi),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(idx)), cu(extent)),
        ),
    ];
    for (d, &cd) in coord.iter().enumerate() {
        s2.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
        ));
        s2.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(cd)), cu(out_shape[d])),
        ));
    }
    s2.push(Statement::Assign(
        Place::local(lt_lo),
        Rvalue::BinaryOp(
            BinOp::Lt,
            copy(Place::local(coord[axis])),
            copy(Place::local(idx)),
        ),
    ));
    let bb2 = BasicBlock {
        statements: s2,
        terminator: guard(lt_lo, 3, 4),
    };
    // bb3: lt_hi = coord[axis] < hi (= idx+extent).
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(lt_hi),
            Rvalue::BinaryOp(
                BinOp::Lt,
                copy(Place::local(coord[axis])),
                copy(Place::local(hi)),
            ),
        )],
        terminator: guard(lt_hi, 4, 5),
    };
    // bb4: operand path (out[i] = operand[i]).
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(operand, i)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    // bb5: update path. uaxis = coord[axis]-idx; upd_flat = sum coord[d]*upd_strides[d] (axis -> uaxis).
    let mut s5 = vec![
        Statement::Assign(
            Place::local(uaxis),
            Rvalue::BinaryOp(
                BinOp::Sub,
                copy(Place::local(coord[axis])),
                copy(Place::local(idx)),
            ),
        ),
        Statement::Assign(Place::local(acc), Rvalue::Use(cu(0))),
    ];
    for (d, &cd) in coord.iter().enumerate() {
        let src = if d == axis { uaxis } else { cd };
        s5.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(src)), cu(upd_strides[d])),
        ));
        s5.push(Statement::Assign(
            Place::local(acc_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(acc)),
                copy(Place::local(term)),
            ),
        ));
        s5.push(Statement::Assign(
            Place::local(acc),
            Rvalue::Use(copy(Place::local(acc_new))),
        ));
    }
    s5.push(Statement::Assign(
        Place::local(val),
        Rvalue::Use(copy(elem(update, acc))),
    ));
    s5.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(val))),
    ));
    let bb5 = BasicBlock {
        statements: s5,
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6])
}
