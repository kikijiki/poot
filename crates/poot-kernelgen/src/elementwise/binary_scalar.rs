use super::*;

/// Elementwise binary with a baked scalar literal: `c[i] = a[i] op lit`.
pub fn binary_scalar(name: &str, op: BinOp, lit: f32) -> Body {
    binary_scalar_dt(name, op, Ty::F32, lit)
}

/// `binary_scalar` in storage dtype `dt`, computing in f32 (spec 024): bf16 input widens, the op runs in
/// f32 against the literal, the result narrows to `dt`.
pub fn binary_scalar_dt(name: &str, op: BinOp, dt: Ty, lit: f32) -> Body {
    binary_scalar_dt_grid(name, op, dt, lit, None)
}

/// [`binary_scalar_dt`] with an optional 2-D grid fold; see [`build_unary_dt_grid`] (card 181 Bug 2: the
/// attention-score `scores * scale` hits the same over-cap grid at long ISL as the `Exp` after it).
/// `x_groups: None` is identical to [`binary_scalar_dt`].
pub fn binary_scalar_dt_grid(
    name: &str,
    op: BinOp,
    dt: Ty,
    lit: f32,
    x_groups: Option<usize>,
) -> Body {
    binary_scalar_typed_grid(name, op, dt, Constant::F32(lit), x_groups, false)
}

/// Exact signed I32 scalar binary operation. The literal is an I32 immediate, not converted through f32.
pub fn binary_scalar_i32_grid(name: &str, op: BinOp, lit: i32, x_groups: Option<usize>) -> Body {
    binary_scalar_typed_grid(name, op, Ty::I32, Constant::I32(lit), x_groups, false)
}

/// Exact unsigned `>=` over an I32 buffer and one I32 bit-pattern literal.
pub fn binary_scalar_i32_geu_grid(name: &str, lit: i32, x_groups: Option<usize>) -> Body {
    binary_scalar_typed_grid(
        name,
        BinOp::Ge,
        Ty::I32,
        Constant::U32(lit as u32),
        x_groups,
        true,
    )
}

/// Exact unsigned remainder over an I32 buffer and one I32 bit-pattern literal (the divisor).
pub fn binary_scalar_i32_remu_grid(name: &str, lit: i32, x_groups: Option<usize>) -> Body {
    binary_scalar_typed_grid(
        name,
        BinOp::Rem,
        Ty::I32,
        Constant::U32(lit as u32),
        x_groups,
        true,
    )
}

fn binary_scalar_typed_grid(
    name: &str,
    op: BinOp,
    dt: Ty,
    lit: Constant,
    x_groups: Option<usize>,
    unsigned_operands: bool,
) -> Body {
    let narrow_float = matches!(dt, Ty::BF16 | Ty::F16);
    let compute_ty = if narrow_float { Ty::F32 } else { dt.clone() };
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false),
        ld(slice_dtype(dt.clone(), true), true),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(compute_ty.clone(), false),
    ]);
    let (a, c, i, len, cmp, r) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let af = if narrow_float {
        Some(al.add(Ty::F32, false))
    } else {
        None
    };
    let au = unsigned_operands.then(|| al.add(Ty::U32, false));
    let unsigned_rem = (unsigned_operands && op == BinOp::Rem).then(|| al.add(Ty::U32, false));
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    // bb2: optionally widen a bf16 input to f32, compute `r = src op lit` (i1->f32 cast for a comparison,
    // reusing the dead `cmp` local), then store r (narrowing to dt for bf16).
    let mut bb2_stmts = Vec::new();
    let src = match af {
        None => copy(elem(a, i)),
        Some(af) => {
            bb2_stmts.push(Statement::Assign(
                Place::local(af),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(a, i)),
                },
            ));
            copy(Place::local(af))
        }
    };
    let src = if let Some(au) = au {
        bb2_stmts.push(Statement::Assign(
            Place::local(au),
            Rvalue::Bitcast {
                to: Ty::U32,
                operand: src,
            },
        ));
        copy(Place::local(au))
    } else {
        src
    };
    let lit_op = Operand::Const(lit.clone());
    if is_cmp(op) {
        bb2_stmts.push(Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(op, src, lit_op),
        ));
        bb2_stmts.push(Statement::Assign(
            Place::local(r),
            Rvalue::Cast {
                to: compute_ty,
                operand: copy(Place::local(cmp)),
            },
        ));
    } else if let Some(rem_local) = unsigned_rem {
        // `src` is already the U32 bitcast and `lit` is a U32 divisor; rem, then bitcast back.
        bb2_stmts.push(Statement::Assign(
            Place::local(rem_local),
            Rvalue::BinaryOp(BinOp::Rem, src, lit_op),
        ));
        bb2_stmts.push(Statement::Assign(
            Place::local(r),
            Rvalue::Bitcast {
                to: compute_ty,
                operand: copy(Place::local(rem_local)),
            },
        ));
    } else if matches!(op, BinOp::Shl | BinOp::Shr) && dt == Ty::I32 {
        let masked = match lit {
            Constant::I32(v) => Operand::Const(Constant::I32(v & 31)),
            other => Operand::Const(other),
        };
        if op == BinOp::Shr {
            let rv = emit_i32_shift(op, src, masked, &mut al, &mut bb2_stmts);
            bb2_stmts.push(Statement::Assign(Place::local(r), rv));
        } else {
            bb2_stmts.push(Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(op, src, masked),
            ));
        }
    } else {
        bb2_stmts.push(Statement::Assign(
            Place::local(r),
            Rvalue::BinaryOp(op, src, lit_op),
        ));
    }
    bb2_stmts.push(Statement::Assign(
        elem(c, i),
        if narrow_float {
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(r)),
            }
        } else {
            Rvalue::Use(copy(Place::local(r)))
        },
    ));
    let bb2 = BasicBlock {
        statements: bb2_stmts,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    match x_groups {
        None => {
            // Flat 1-D grid: `i = ThreadIndexCall(X)`.
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(i),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            };
            Body::new(name, 2, al.locals, vec![bb0, bb1, bb2, bb3])
        }
        Some(xg) => {
            // 2-D grid fold (card 181 Bug 2), same as `build_unary_dt_grid`: gx=GroupX -> gy=GroupY ->
            // group_id=gy*xg+gx, lane=LocalX -> i=group_id*wg+lane -> bb1. The caller must bake
            // `body.workgroup_size[0] = ELEMENTWISE_2D_WORKGROUP_SIZE`.
            let wg = ELEMENTWISE_2D_WORKGROUP_SIZE;
            let gx = al.add(Ty::Usize, false);
            let gy = al.add(Ty::Usize, false);
            let group_id = al.add(Ty::Usize, false);
            let lane = al.add(Ty::Usize, false);
            let gy_idx: u32 = 4;
            let lane_idx: u32 = 5;
            let combine_idx: u32 = 6;
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(gx),
                    dim: IndexAxis::GroupX,
                    target: BlockId { index: gy_idx },
                },
            };
            let bb_gy = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(gy),
                    dim: IndexAxis::GroupY,
                    target: BlockId { index: lane_idx },
                },
            };
            let bb_lane = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(lane),
                    dim: IndexAxis::LocalX,
                    target: BlockId { index: combine_idx },
                },
            };
            let bb_combine = BasicBlock {
                statements: vec![
                    Statement::Assign(
                        Place::local(group_id),
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(xg)),
                    ),
                    Statement::Assign(
                        Place::local(group_id),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(group_id)),
                            copy(Place::local(gx)),
                        ),
                    ),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(group_id)), cu(wg)),
                    ),
                    Statement::Assign(
                        Place::local(i),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(i)),
                            copy(Place::local(lane)),
                        ),
                    ),
                ],
                terminator: Terminator::Goto {
                    target: BlockId { index: 1 },
                },
            };
            Body::new(
                name,
                2,
                al.locals,
                vec![bb0, bb1, bb2, bb3, bb_gy, bb_lane, bb_combine],
            )
        }
    }
}
