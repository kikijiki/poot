use super::*;

/// Scatter rows along axis 0 in storage dtype `dt`, the inverse of [`super::gather_axis0_index_dt`] (at
/// `index_dt = Ty::F32`):
/// `out[index[s], r] = src[s, r]` where `s = i / rest`, `r = i % rest`. `index` is a `[N]` permutation of
/// `0..N` (f32, cast to usize), `src`/`out` carry `dt`. One thread per source element `i`; every `out`
/// slot is written exactly once (no pre-zeroing, no races). Card 034.
pub fn scatter_axis0_dt(name: &str, dt: Ty, rest: usize) -> Body {
    // _1 src, _2 index, _3 out
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // src (dt)
        ld(slice_f32(false), false),               // index (f32 permutation)
        ld(slice_dtype(dt.clone(), true), true),   // out (dt)
    ]);
    let (src, index, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let s = al.add(Ty::Usize, false);
    let r = al.add(Ty::Usize, false);
    let idxf = al.add(Ty::F32, false);
    let row = al.add(Ty::Usize, false);
    let base = al.add(Ty::Usize, false);
    let dst = al.add(Ty::Usize, false);
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
            // guard over the source length (== out length for a permutation).
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(src))),
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
            Statement::Assign(Place::local(idxf), Rvalue::Use(copy(elem(index, s)))),
            Statement::Assign(
                Place::local(row),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(idxf)),
                },
            ),
            Statement::Assign(
                Place::local(base),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(dst),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(base)), copy(Place::local(r))),
            ),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(src, i)))),
            Statement::Assign(elem(out, dst), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3])
}

/// Scatter-update rows into a base buffer (card 059, the multi-token paged KV write): `out[p, :] =
/// inv[p] >= 0 ? src[inv[p], :] : base[p, :]`. `base` is `[POOL, rest]`, `src` is `[N, rest]`, `inv` is the
/// `[POOL]` inverse map (`inv[p]` = the source row written to physical slot `p`, or `-1` to keep `base[p]`),
/// read as f32 and cast to i32. Unlike [`scatter_axis0_dt`] (a permutation), this writes only the `N`
/// mapped rows and passes the rest through from `base`, landing a prefill's tokens in non-contiguous paged
/// blocks. One thread per output element; the host builds the inverse map. dt-generic pure copy.
pub fn scatter_update_dt(name: &str, dt: Ty, rest: usize) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 base [POOL*rest] (dt)
        ld(slice_dtype(dt.clone(), false), false), // 2 src [N*rest] (dt)
        ld(slice_f32(false), false),               // 3 inv [POOL] (f32: source row or -1)
        ld(slice_dtype(dt.clone(), true), true),   // 4 out [POOL*rest] (dt)
    ]);
    let (base, src, inv, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let row = al.add(Ty::Usize, false); // i / rest
    let r = al.add(Ty::Usize, false); // i % rest
    let invf = al.add(Ty::F32, false);
    let j = al.add(Ty::I32, false); // (i32) inv[row]
    let neg = al.add(Ty::Bool, false); // j < 0
    let ju = al.add(Ty::Usize, false); // (usize) j
    let term = al.add(Ty::Usize, false); // ju * rest
    let sidx = al.add(Ty::Usize, false); // ju*rest + r
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
        terminator: guard(cmp, 5, 2), // out-of-bounds -> bare return (bb5)
    };
    // bb2: row=i/rest; r=i%rest; j=(i32)inv[row]; neg = j<0. neg==0 (j>=0) -> bb3 (src); else bb4 (base).
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(row),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(Place::local(invf), Rvalue::Use(copy(elem(inv, row)))),
            Statement::Assign(
                Place::local(j),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(invf)),
                },
            ),
            Statement::Assign(
                Place::local(neg),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    copy(Place::local(j)),
                    Operand::Const(Constant::I32(0)),
                ),
            ),
        ],
        terminator: guard(neg, 3, 4),
    };
    // bb3: src path. ju=(usize)j; sidx=ju*rest+r; out[i]=src[sidx].
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(ju),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(j)),
                },
            ),
            Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(ju)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(sidx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(term)), copy(Place::local(r))),
            ),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(src, sidx)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    // bb4: base path. out[i] = base[i].
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(base, i)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    let bb5 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5])
}

/// Chunked sibling of [`scatter_update_dt`] (card 159 Inc3): the same `out[p,:] = inv[p]>=0 ?
/// src[inv[p],:] : base[p,:]` over the full `[POOL,rest]` output, with a baked `elem_offset` added to the
/// thread index before the bounds check. The launch spans only
/// `elem_offset..elem_offset+chunk_elems`, so a wide pool (gemma4's shared-pool KV write,
/// up to several million elements) splits into dispatches that each stay under the wgpu display-watchdog
/// `flush_work` cap. `out`/`base` stay bound at full `POOL*rest` length; chunks write disjoint ranges, so
/// dispatch order does not matter (as with other `Plan::ComputeChunks` bodies).
pub fn scatter_update_chunked_dt(name: &str, dt: Ty, rest: usize, elem_offset: usize) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 base [POOL*rest] (dt)
        ld(slice_dtype(dt.clone(), false), false), // 2 src [N*rest] (dt)
        ld(slice_f32(false), false),               // 3 inv [POOL] (f32: source row or -1)
        ld(slice_dtype(dt.clone(), true), true),   // 4 out [POOL*rest] (dt)
    ]);
    let (base, src, inv, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, true);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let row = al.add(Ty::Usize, false); // i / rest
    let r = al.add(Ty::Usize, false); // i % rest
    let invf = al.add(Ty::F32, false);
    let j = al.add(Ty::I32, false); // (i32) inv[row]
    let neg = al.add(Ty::Bool, false); // j < 0
    let ju = al.add(Ty::Usize, false); // (usize) j
    let term = al.add(Ty::Usize, false); // ju * rest
    let sidx = al.add(Ty::Usize, false); // ju*rest + r
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
    let mut bb1_stmts = Vec::new();
    if elem_offset != 0 {
        bb1_stmts.push(Statement::Assign(
            Place::local(i),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), cu(elem_offset)),
        ));
    }
    bb1_stmts.push(Statement::Assign(
        Place::local(len),
        Rvalue::Len(Place::local(out)),
    ));
    bb1_stmts.push(Statement::Assign(
        Place::local(cmp),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
    ));
    let bb1 = BasicBlock {
        statements: bb1_stmts,
        terminator: guard(cmp, 5, 2), // out-of-bounds -> bare return (bb5)
    };
    // bb2: row=i/rest; r=i%rest; j=(i32)inv[row]; neg = j<0. neg==0 (j>=0) -> bb3 (src); else bb4 (base).
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(row),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(rest)),
            ),
            Statement::Assign(Place::local(invf), Rvalue::Use(copy(elem(inv, row)))),
            Statement::Assign(
                Place::local(j),
                Rvalue::Cast {
                    to: Ty::I32,
                    operand: copy(Place::local(invf)),
                },
            ),
            Statement::Assign(
                Place::local(neg),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    copy(Place::local(j)),
                    Operand::Const(Constant::I32(0)),
                ),
            ),
        ],
        terminator: guard(neg, 3, 4),
    };
    // bb3: src path. ju=(usize)j; sidx=ju*rest+r; out[i]=src[sidx].
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(ju),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(j)),
                },
            ),
            Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(ju)), cu(rest)),
            ),
            Statement::Assign(
                Place::local(sidx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(term)), copy(Place::local(r))),
            ),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(src, sidx)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    // bb4: base path. out[i] = base[i].
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(base, i)))),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(val)))),
        ],
        terminator: Terminator::Return,
    };
    let bb5 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5])
}
