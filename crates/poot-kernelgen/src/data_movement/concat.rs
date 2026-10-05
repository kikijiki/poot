use super::*;

/// Concatenate two tensors along `axis` in storage dtype `dt` (spec 024; pure element copy, no casts):
/// `out` = `[a; b]` (a and b match `out` except in `axis`). One thread per output element; a branch
/// routes each to a or b. Covers RoPE rotate-half + KV append.
pub fn concat2_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    axis: usize,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Body {
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);
    let a_axis_len = a_shape[axis];

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 a
        ld(slice_dtype(dt.clone(), false), false), // 2 b
        ld(slice_dtype(dt.clone(), true), true),   // 3 out
    ]);
    let (a, b, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let coord: Vec<Local> = (0..r).map(|_| al.add(Ty::Usize, false)).collect();
    let in_a = al.add(Ty::Bool, false);
    let acc = al.add(Ty::Usize, true);
    let term = al.add(Ty::Usize, false);
    let acc_new = al.add(Ty::Usize, false);
    let baxis = al.add(Ty::Usize, false);
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
        terminator: guard(cmp, 5, 2),
    };
    // bb2: coords + in_a = coord[axis] < a_axis_len
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
        Place::local(in_a),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(coord[axis])), cu(a_axis_len)),
    ));
    let bb2 = BasicBlock {
        statements: s2,
        terminator: guard(in_a, 4, 3),
    }; // in_a==0 -> b(bb4) else a(bb3)
    // bb3: a path. a_flat = sum coord[d]*a_strides[d]; out[i]=a[a_flat]
    let mut sa = vec![Statement::Assign(Place::local(acc), Rvalue::Use(cu(0)))];
    for (d, &cd) in coord.iter().enumerate() {
        sa.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(cd)), cu(a_strides[d])),
        ));
        sa.push(Statement::Assign(
            Place::local(acc_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(acc)),
                copy(Place::local(term)),
            ),
        ));
        sa.push(Statement::Assign(
            Place::local(acc),
            Rvalue::Use(copy(Place::local(acc_new))),
        ));
    }
    sa.push(Statement::Assign(
        Place::local(val),
        Rvalue::Use(copy(elem(a, acc))),
    ));
    sa.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(val))),
    ));
    let bb3 = BasicBlock {
        statements: sa,
        terminator: Terminator::Return,
    };
    // bb4: b path. b_axis = coord[axis]-a_axis_len; b_flat with axis replaced.
    let mut sb = vec![
        Statement::Assign(
            Place::local(baxis),
            Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(coord[axis])), cu(a_axis_len)),
        ),
        Statement::Assign(Place::local(acc), Rvalue::Use(cu(0))),
    ];
    for (d, &cd) in coord.iter().enumerate() {
        let src = if d == axis { baxis } else { cd };
        sb.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(src)), cu(b_strides[d])),
        ));
        sb.push(Statement::Assign(
            Place::local(acc_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(acc)),
                copy(Place::local(term)),
            ),
        ));
        sb.push(Statement::Assign(
            Place::local(acc),
            Rvalue::Use(copy(Place::local(acc_new))),
        ));
    }
    sb.push(Statement::Assign(
        Place::local(val),
        Rvalue::Use(copy(elem(b, acc))),
    ));
    sb.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(val))),
    ));
    let bb4 = BasicBlock {
        statements: sb,
        terminator: Terminator::Return,
    };
    let bb5 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5])
}

/// Concatenate N tensors along `axis` (the `N != 2` case; [`concat2_dt`] handles 2 inputs), keeping the
/// op on-device (card 043). `out` matches every input except along `axis`, where the input axis sizes sum.
/// Inputs are `_1.._N`, `out` is `_{N+1}`. One thread per output element `i`: decode the output coordinate,
/// read every input at a clamped in-bounds index, and keep the owning segment's value via a 0/1 mask
/// (`out = sum_k mask_k * in_k`, `mask_k = 1` iff `cum[k] <= coord[axis] < cum[k+1]`). Branchless because
/// a nested N-way segment-guard ladder makes the LLVM->SPIR-V structurizer emit control flow that faults
/// RADV past ~2 nesting levels; this keeps a single selection (the grid guard). Exact: one segment owns
/// each coord and non-owner terms are 0.0. dt-generic pure copy.
pub fn concat_n_dt(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    axis: usize,
    in_shapes: &[&[usize]],
) -> Result<Body, KernelGenError> {
    let n = in_shapes.len();
    if n < 1 {
        return Err(KernelGenError::BelowMinimum {
            generator: "concat_n_dt",
            what: "in_shapes count".to_string(),
            value: n,
            min: 1,
        });
    }
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let in_strides: Vec<Vec<usize>> = in_shapes.iter().map(|s| row_major_strides(s)).collect();
    // cum[k] = sum of in_shapes[0..k][axis]; cum[N] = out_shape[axis].
    let mut cum = vec![0usize; n + 1];
    for k in 0..n {
        cum[k + 1] = cum[k] + in_shapes[k][axis];
    }

    let mut params = vec![ld(Ty::Unit, false)];
    for _ in 0..n {
        params.push(ld(slice_dtype(dt.clone(), false), false)); // input k
    }
    params.push(ld(slice_dtype(dt.clone(), true), true)); // out
    let mut al = Alloc::new(params);
    let ins: Vec<Local> = (0..n).map(|k| local((1 + k) as u32)).collect();
    let out = local((1 + n) as u32);
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let coord: Vec<Local> = (0..r).map(|_| al.add(Ty::Usize, false)).collect();
    // per-input scratch, reused across the unrolled k loop.
    let base = al.add(Ty::Usize, false);
    let local0 = al.add(Ty::Usize, false);
    let laxis = al.add(Ty::Usize, false);
    let acc = al.add(Ty::Usize, true);
    let term = al.add(Ty::Usize, false);
    let geb = al.add(Ty::Bool, false);
    let ltb = al.add(Ty::Bool, false);
    let gef = al.add(dt.clone(), false);
    let ltf = al.add(dt.clone(), false);
    let mask = al.add(dt.clone(), false);
    let vk = al.add(dt.clone(), false);
    let termv = al.add(dt.clone(), false);
    let out_val = al.add(dt.clone(), true);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    // Branchless (see the fn doc): nested selections fault RADV at depth > ~2. Layout: 0 bb0, 1 bb1
    // (guard), 2 body (straight-line), 3 END.
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
    let mut s = Vec::new();
    // coord[d] = (i / out_strides[d]) % out_shape[d]
    for (d, &cd) in coord.iter().enumerate() {
        s.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
        ));
        s.push(Statement::Assign(
            Place::local(cd),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(cd)), cu(out_shape[d])),
        ));
    }
    // out_val = 0 in dt.
    s.push(Statement::Assign(
        Place::local(out_val),
        Rvalue::Cast {
            to: dt.clone(),
            operand: cu(0),
        },
    ));
    for k in 0..n {
        let sz = in_shapes[k][axis];
        // Clamp the local axis coord into [0, sz-1] so the read is in bounds even for masked-out threads:
        // base = max(coord[axis], cum[k]); laxis = min(base - cum[k], sz-1).
        s.push(Statement::Assign(
            Place::local(base),
            Rvalue::BinaryOp(BinOp::Max, copy(Place::local(coord[axis])), cu(cum[k])),
        ));
        s.push(Statement::Assign(
            Place::local(local0),
            Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(base)), cu(cum[k])),
        ));
        s.push(Statement::Assign(
            Place::local(laxis),
            Rvalue::BinaryOp(BinOp::Min, copy(Place::local(local0)), cu(sz - 1)),
        ));
        // flat = sum_d coord'[d] * in_strides[k][d] (axis dim uses the clamped laxis).
        s.push(Statement::Assign(Place::local(acc), Rvalue::Use(cu(0))));
        for (d, &cd) in coord.iter().enumerate() {
            let src_coord = if d == axis { laxis } else { cd };
            s.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(src_coord)),
                    cu(in_strides[k][d]),
                ),
            ));
            s.push(Statement::Assign(
                Place::local(acc),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(term)),
                ),
            ));
        }
        s.push(Statement::Assign(
            Place::local(vk),
            Rvalue::Use(copy(elem(ins[k], acc))),
        ));
        // mask = (coord[axis] >= cum[k]) * (coord[axis] < cum[k+1]) in dt (1.0 for the owner, else 0.0).
        s.push(Statement::Assign(
            Place::local(geb),
            Rvalue::BinaryOp(BinOp::Ge, copy(Place::local(coord[axis])), cu(cum[k])),
        ));
        s.push(Statement::Assign(
            Place::local(ltb),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(coord[axis])), cu(cum[k + 1])),
        ));
        s.push(Statement::Assign(
            Place::local(gef),
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(geb)),
            },
        ));
        s.push(Statement::Assign(
            Place::local(ltf),
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(ltb)),
            },
        ));
        s.push(Statement::Assign(
            Place::local(mask),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gef)), copy(Place::local(ltf))),
        ));
        // out_val += mask * vk (only the owner's term is nonzero).
        s.push(Statement::Assign(
            Place::local(termv),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(mask)), copy(Place::local(vk))),
        ));
        s.push(Statement::Assign(
            Place::local(out_val),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(out_val)),
                copy(Place::local(termv)),
            ),
        ));
    }
    s.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(out_val))),
    ));
    let body = BasicBlock {
        statements: s,
        terminator: Terminator::Return,
    };
    let end = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    Ok(Body::new(
        name,
        (n + 1) as u32,
        al.locals,
        vec![bb0, bb1, body, end],
    ))
}
