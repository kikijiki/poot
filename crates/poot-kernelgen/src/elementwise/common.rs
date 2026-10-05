use super::*;

/// Workgroup width for [`binary_broadcast_dt_views_grid`]'s 2-D-grid-fold path. Mirrors
/// `poot-graph-plan::ELEMENTWISE_2D_WIDTH` (kernelgen sits below poot-graph-plan, so it cannot reference
/// it). `plan_eqn` bakes `body.workgroup_size[0]` to that constant, and the two must agree for the
/// dispatch to reconstruct the same `i`; a unit test cross-checks them.
pub const ELEMENTWISE_2D_WORKGROUP_SIZE: usize = 256;

/// Shared one-input elementwise skeleton; `make_rv` builds the rvalue computing the result from `x[i]`.
/// Locals: _1 x, _2 y, _3 i, _4 len, _5 cmp, _6 r.
/// Narrow floats (bf16/f16) widen to f32, run the op, then narrow back (spec 024). F32 and I32 compute in
/// their storage type, so an exact-I32 `Not`/`Clz` body never recasts words through f32.
pub fn build_unary_dt(name: &str, dt: Ty, make_rv: impl Fn(Place) -> Rvalue) -> Body {
    build_unary_dt_grid(name, dt, make_rv, None)
}

/// [`build_unary_dt`] with an optional 2-D grid fold; see [`binary_broadcast_dt_views_grid`] for `x_groups`.
/// Card 181 Bug 2: a non-flash prefill's attention-score chain (scale `Mul`, `Exp`) runs one thread per
/// element of the full `[1,Hq,L,L]` scores tensor, which at long ISL exceeds the `65535*256` ceiling
/// (ISL=2048, Hq=14: 58,720,256 elements, `GridCap([229376,1,1])`). `x_groups: None` keeps the flat
/// `ThreadIndexCall(X)`.
pub fn build_unary_dt_grid(
    name: &str,
    dt: Ty,
    make_rv: impl Fn(Place) -> Rvalue,
    x_groups: Option<usize>,
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
        ld(compute_ty, false),
    ]);
    let (x, y, i, len, cmp, r) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let xf = if narrow_float {
        Some(al.add(Ty::F32, false))
    } else {
        None
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: match xf {
            None => vec![
                Statement::Assign(Place::local(r), make_rv(elem(x, i))),
                Statement::Assign(elem(y, i), Rvalue::Use(copy(Place::local(r)))),
            ],
            Some(xf) => vec![
                Statement::Assign(
                    Place::local(xf),
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: copy(elem(x, i)),
                    },
                ),
                Statement::Assign(Place::local(r), make_rv(Place::local(xf))),
                Statement::Assign(
                    elem(y, i),
                    Rvalue::Cast {
                        to: dt.clone(),
                        operand: copy(Place::local(r)),
                    },
                ),
            ],
        },
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
            // 2-D grid fold (card 181 Bug 2, as in `binary_broadcast_dt_views_grid`): gx=GroupX ->
            // gy=GroupY -> group_id=gy*xg+gx, lane=LocalX -> i=group_id*wg+lane -> bb1 (tail-length guard).
            // The caller must bake `body.workgroup_size[0] = ELEMENTWISE_2D_WORKGROUP_SIZE`.
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

/// [`build_unary_dt`] generalized to read `x` through an arbitrary physical [`Layout`] (spec 132: a
/// Transpose/Slice/Broadcast feeding a bare Neg/Recip/Sqrt/Round/Exp/Log, e.g. the RoPE `rotate_half`
/// negate). A unary op never broadcasts, so the operand shape is `out_shape`. With an optional 2-D grid
/// fold (card 181 Bug 2): `dispatch_grid`/`is_elementwise_2d` cannot see whether an operand was promoted
/// to a view (a planning-time fact in `poot-graph-plan`'s `views` map), so both the plain and view-reading
/// bodies must fold identically whenever `is_elementwise_2d` says so.
pub fn build_unary_dt_views_grid(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    layout: &Layout,
    make_rv: impl Fn(Place) -> Rvalue,
    x_groups: Option<usize>,
) -> Body {
    let (eff, base) = view_eff_strides(out_shape, out_shape, layout);
    let out_strides = row_major_strides(out_shape);
    if eff == out_strides.as_slice() && base == 0 {
        // plain contiguous buffer: same as `build_unary_dt`.
        return build_unary_dt_grid(name, dt, make_rv, x_groups);
    }
    let narrow_float = matches!(dt, Ty::BF16 | Ty::F16);
    let compute_ty = if narrow_float { Ty::F32 } else { dt.clone() };
    let mut locals = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false),
        ld(slice_dtype(dt.clone(), true), true),
        ld(Ty::Usize, false),  // 3 i
        ld(Ty::Usize, false),  // 4 len
        ld(Ty::Bool, false),   // 5 cmp
        ld(compute_ty, false), // 6 r
        ld(Ty::Usize, true),   // 7 off (the view-mapped read index)
    ];
    let (x, y, i, len, cmp, r, off) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
    );
    let mut n = 8u32;
    let mut fresh = |locals: &mut Vec<LocalDecl>, ty: Ty| {
        locals.push(ld(ty, false));
        let l = local(n);
        n += 1;
        l
    };
    let xf = if narrow_float {
        Some(fresh(&mut locals, Ty::F32))
    } else {
        None
    };
    // scratch for the unravel.
    let div = fresh(&mut locals, Ty::Usize);
    let md = fresh(&mut locals, Ty::Usize);
    let term = fresh(&mut locals, Ty::Usize);
    let off_new = fresh(&mut locals, Ty::Usize);
    let rem = fresh(&mut locals, Ty::Usize);
    let cu = |v: usize| Operand::Const(Constant::Usize(v as u64));

    // off = the view-mapped read index for output element `i`: same div+mul+sub unravel as
    // `binary_broadcast_dt` (never `urem`, which NVPTX miscompiles for a large i64 dividend), seeded at `base`.
    let mut off_stmts: Vec<Statement> = Vec::new();
    if eff == out_strides.as_slice() {
        // plain strides with a nonzero base (e.g. a Slice view): off = i + base.
        off_stmts.push(Statement::Assign(
            Place::local(off_new),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), cu(base)),
        ));
        off_stmts.push(Statement::Assign(
            Place::local(off),
            Rvalue::Use(copy(Place::local(off_new))),
        ));
    } else {
        off_stmts.push(Statement::Assign(Place::local(off), Rvalue::Use(cu(base))));
        off_stmts.push(Statement::Assign(
            Place::local(rem),
            Rvalue::Use(copy(Place::local(i))),
        ));
        for d in 0..out_shape.len() {
            off_stmts.push(Statement::Assign(
                Place::local(div),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(rem)), cu(out_strides[d])),
            ));
            if eff[d] != 0 {
                off_stmts.push(Statement::Assign(
                    Place::local(term),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(eff[d])),
                ));
                off_stmts.push(Statement::Assign(
                    Place::local(off_new),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(off)),
                        copy(Place::local(term)),
                    ),
                ));
                off_stmts.push(Statement::Assign(
                    Place::local(off),
                    Rvalue::Use(copy(Place::local(off_new))),
                ));
            }
            if d + 1 < out_shape.len() {
                off_stmts.push(Statement::Assign(
                    Place::local(md),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(out_strides[d])),
                ));
                off_stmts.push(Statement::Assign(
                    Place::local(off_new),
                    Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(rem)), copy(Place::local(md))),
                ));
                off_stmts.push(Statement::Assign(
                    Place::local(rem),
                    Rvalue::Use(copy(Place::local(off_new))),
                ));
            }
        }
    }

    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let mut bb2_stmts = off_stmts;
    match xf {
        None => {
            bb2_stmts.push(Statement::Assign(Place::local(r), make_rv(elem(x, off))));
            bb2_stmts.push(Statement::Assign(
                elem(y, i),
                Rvalue::Use(copy(Place::local(r))),
            ));
        }
        Some(xf) => {
            bb2_stmts.push(Statement::Assign(
                Place::local(xf),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(x, off)),
                },
            ));
            bb2_stmts.push(Statement::Assign(
                Place::local(r),
                make_rv(Place::local(xf)),
            ));
            bb2_stmts.push(Statement::Assign(
                elem(y, i),
                Rvalue::Cast {
                    to: dt.clone(),
                    operand: copy(Place::local(r)),
                },
            ));
        }
    }
    let bb2 = BasicBlock {
        statements: bb2_stmts,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
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
            Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
        }
        Some(xg) => {
            // 2-D grid fold (card 181 Bug 2), as in `build_unary_dt_grid`. Extra blocks go after bb3 (index 4+).
            let wg = ELEMENTWISE_2D_WORKGROUP_SIZE;
            let gx = fresh(&mut locals, Ty::Usize);
            let gy = fresh(&mut locals, Ty::Usize);
            let group_id = fresh(&mut locals, Ty::Usize);
            let lane = fresh(&mut locals, Ty::Usize);
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
                locals,
                vec![bb0, bb1, bb2, bb3, bb_gy, bb_lane, bb_combine],
            )
        }
    }
}
