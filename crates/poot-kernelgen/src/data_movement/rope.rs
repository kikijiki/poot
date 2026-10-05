use super::*;

/// Fused RoPE (rotary position embedding): one dispatch replacing the `slice/slice/neg/concat/mul/mul/add`
/// (`+slice/concat` for partial) chain traced by `ops::rope_partial`. f32; see [`rope_dt`].
pub fn rope(
    name: &str,
    x_shape: &[usize],
    cos_shape: &[usize],
    rot: usize,
) -> Result<Body, KernelGenError> {
    rope_dt(name, Ty::F32, x_shape, cos_shape, rot)
}

/// [`rope`] in storage dtype `dt`. Inputs `[x, cos, sin]`, output shape == `x_shape`. One thread per output
/// element `i` (1-D grid, wg=1 like [`concat2_dt`]). Let `j = coord[last]` be the element's position in the
/// head dim, `half = rot/2`. GPT-NeoX / HF half-split rotate:
///   `j >= rot`      : `out[i] = x[i]`                          (passthrough, only when `rot < D`)
///   `j <  half`     : `out[i] = x[i]*cos[cj] + (-x[i+half])*sin[cj]`
///   `half <= j < rot`: `out[i] = x[i]*cos[cj] +   x[i-half] *sin[cj]`
/// where `cj` is `cos`/`sin`'s broadcast-into-`x` index (leading dims broadcast; last dim indexed by `j`).
/// The float ops mirror the decomposition (`xc = x*cos`, `rs = rotate_half*sin`, `xc + rs`), so the result
/// matches the un-fused chain to float-reorder tolerance. `x`'s last-axis stride is 1, so the partner
/// element is at linear index `i +/- half`. Coord unravel uses div/mul/sub, never `urem` (the NVPTX i64
/// remainder miscompiles for a large dividend; see `binary_broadcast_dt_views`).
pub fn rope_dt(
    name: &str,
    dt: Ty,
    x_shape: &[usize],
    cos_shape: &[usize],
    rot: usize,
) -> Result<Body, KernelGenError> {
    let r = x_shape.len();
    if r < 1 {
        return Err(KernelGenError::BelowMinimum {
            generator: "rope_dt",
            what: "x_shape rank".to_string(),
            value: r,
            min: 1,
        });
    }
    let last = r - 1;
    let d = x_shape[last];
    if rot > d {
        return Err(KernelGenError::ExceedsBound {
            generator: "rope_dt",
            what: "rot".to_string(),
            value: rot,
            bound: d,
        });
    }
    if !rot.is_multiple_of(2) {
        return Err(KernelGenError::NotDivisible {
            generator: "rope_dt",
            dim: "rot".to_string(),
            value: rot,
            divisor: 2,
        });
    }
    let half = rot / 2;
    let out_strides = row_major_strides(x_shape);
    let cos_strides = row_major_strides(cos_shape);
    let cos_last_stride = *cos_strides.last().unwrap_or(&1);
    let cos_rank = cos_shape.len();
    if cos_rank > r {
        return Err(KernelGenError::ExceedsBound {
            generator: "rope_dt",
            what: "cos_shape rank".to_string(),
            value: cos_rank,
            bound: r,
        });
    }
    let align = r - cos_rank; // cos is right-aligned to x's trailing dims
    // Broadcast-effective stride of cos/sin into x for the leading dims 0..last only. The last dim is
    // indexed by j (width `rot`, which need not equal D), so leaf_eff[last] is unused.
    let leaf_eff: Vec<usize> = (0..r)
        .map(|dd| {
            if dd < align {
                0
            } else {
                let cd = dd - align;
                if cos_shape[cd] == 1 && x_shape[dd] > 1 {
                    0
                } else {
                    cos_strides[cd]
                }
            }
        })
        .collect();

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 x
        ld(slice_dtype(dt.clone(), false), false), // 2 cos
        ld(slice_dtype(dt.clone(), false), false), // 3 sin
        ld(slice_dtype(dt.clone(), true), true),   // 4 out
    ]);
    let (x, cos, sin, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp_len = al.add(Ty::Bool, false);
    let rem = al.add(Ty::Usize, true);
    let cos_off = al.add(Ty::Usize, true);
    let j = al.add(Ty::Usize, false);
    let div = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let acc_new = al.add(Ty::Usize, false);
    let cmp_pass = al.add(Ty::Bool, false);
    let cmp_lo = al.add(Ty::Bool, false);
    let jl = al.add(Ty::Usize, false);
    let cos_idx = al.add(Ty::Usize, false);
    let partner = al.add(Ty::Usize, false);
    let xv = al.add(Ty::F32, false);
    let cosv = al.add(Ty::F32, false);
    let sinv = al.add(Ty::F32, false);
    let xc = al.add(Ty::F32, false);
    let pv = al.add(Ty::F32, false);
    let negv = al.add(Ty::F32, false);
    let rs = al.add(Ty::F32, false);
    let res = al.add(Ty::F32, false);
    let cu = |v: usize| Operand::Const(Constant::Usize(v as u64));
    let widen = |src: Place| -> Rvalue {
        if dt == Ty::F32 {
            Rvalue::Use(copy(src))
        } else {
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(src),
            }
        }
    };
    let narrow = |val: Local| -> Rvalue {
        if dt == Ty::F32 {
            Rvalue::Use(copy(Place::local(val)))
        } else {
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(val)),
            }
        }
    };

    // bb0: global thread index.
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    // bb1: bounds guard (i < out.len()).
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp_len),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp_len, 7, 2),
    };
    // bb2: unravel i -> cos_off (leading dims) + j (last coord); branch on j < rot.
    let mut s2 = vec![
        Statement::Assign(Place::local(rem), Rvalue::Use(copy(Place::local(i)))),
        Statement::Assign(Place::local(cos_off), Rvalue::Use(cu(0))),
    ];
    for dd in 0..r {
        s2.push(Statement::Assign(
            Place::local(div),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(rem)), cu(out_strides[dd])),
        ));
        if dd == last {
            // last coord j == rem (out_strides[last] == 1), captured via the div above.
            s2.push(Statement::Assign(
                Place::local(j),
                Rvalue::Use(copy(Place::local(div))),
            ));
        } else {
            if leaf_eff[dd] != 0 {
                s2.push(Statement::Assign(
                    Place::local(term),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(leaf_eff[dd])),
                ));
                s2.push(Statement::Assign(
                    Place::local(acc_new),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(cos_off)),
                        copy(Place::local(term)),
                    ),
                ));
                s2.push(Statement::Assign(
                    Place::local(cos_off),
                    Rvalue::Use(copy(Place::local(acc_new))),
                ));
            }
            // rem -= div * out_strides[dd]
            s2.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(out_strides[dd])),
            ));
            s2.push(Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Sub,
                    copy(Place::local(rem)),
                    copy(Place::local(term)),
                ),
            ));
            s2.push(Statement::Assign(
                Place::local(rem),
                Rvalue::Use(copy(Place::local(acc_new))),
            ));
        }
    }
    s2.push(Statement::Assign(
        Place::local(cmp_pass),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(rot)),
    ));
    let bb2 = BasicBlock {
        statements: s2,
        terminator: guard(cmp_pass, 6, 3), // j >= rot -> passthrough(6); j < rot -> rotate(3)
    };
    // bb3: rotate. cos_idx = cos_off + j*cos_last_stride; xc = x[i]*cos; branch on j < half.
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(jl),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(j)), cu(cos_last_stride)),
            ),
            Statement::Assign(
                Place::local(cos_idx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(cos_off)),
                    copy(Place::local(jl)),
                ),
            ),
            Statement::Assign(Place::local(xv), widen(elem(x, i))),
            Statement::Assign(Place::local(cosv), widen(elem(cos, cos_idx))),
            Statement::Assign(Place::local(sinv), widen(elem(sin, cos_idx))),
            Statement::Assign(
                Place::local(xc),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(xv)), copy(Place::local(cosv))),
            ),
            Statement::Assign(
                Place::local(cmp_lo),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(half)),
            ),
        ],
        terminator: guard(cmp_lo, 5, 4), // j >= half -> hi(5); j < half -> lo(4)
    };
    // bb4: lo half (j < half). rotate_half elem = -x[i+half]; out = xc + (-partner)*sin.
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(partner),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), cu(half)),
            ),
            Statement::Assign(Place::local(pv), widen(elem(x, partner))),
            Statement::Assign(
                Place::local(negv),
                Rvalue::UnaryOp(UnOp::Neg, copy(Place::local(pv))),
            ),
            Statement::Assign(
                Place::local(rs),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(negv)),
                    copy(Place::local(sinv)),
                ),
            ),
            Statement::Assign(
                Place::local(res),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(xc)), copy(Place::local(rs))),
            ),
            Statement::Assign(elem(out, i), narrow(res)),
        ],
        terminator: Terminator::Return,
    };
    // bb5: hi half (half <= j < rot). rotate_half elem = x[i-half]; out = xc + partner*sin.
    let bb5 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(partner),
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(i)), cu(half)),
            ),
            Statement::Assign(Place::local(pv), widen(elem(x, partner))),
            Statement::Assign(
                Place::local(rs),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(pv)), copy(Place::local(sinv))),
            ),
            Statement::Assign(
                Place::local(res),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(xc)), copy(Place::local(rs))),
            ),
            Statement::Assign(elem(out, i), narrow(res)),
        ],
        terminator: Terminator::Return,
    };
    // bb6: passthrough (only reached when rot < D and j >= rot). out[i] = x[i] (storage dtype copy).
    let bb6 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, i),
            Rvalue::Use(copy(elem(x, i))),
        )],
        terminator: Terminator::Return,
    };
    // bb7: out-of-bounds return.
    let bb7 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(Body::new(
        name,
        4,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7],
    ))
}
