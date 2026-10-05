use super::*;

/// Batched argtop-k index extraction (spec 136 P2, naive scalar body, FR-003): `rank [rows,e] -> out
/// [rows,k]`, `out[l,r]` = the index `i` in `0..e` with `rank[l,i] == r`. One thread per output element
/// `idx = l*k + r`: `l = idx/k`, `r = idx%k`. Scans the row's `e` entries with a fixed serial loop and no
/// early exit. `rank` and `out` are always F32 (no `dt` parameter). See bb4 for the accumulation rule.
pub fn arg_top_k_dt(name: &str, e: usize, k: usize) -> Body {
    // _1 rank [rows*e], _2 out [rows*k]
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // rank (f32)
        ld(slice_f32(true), true),   // out (f32)
    ]);
    let (rank, out) = (local(1), local(2));
    let i = al.add(Ty::Usize, false); // flat output index l*k + r
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let l = al.add(Ty::Usize, false);
    let r = al.add(Ty::Usize, false);
    let lbase = al.add(Ty::Usize, false); // l * e
    let rf = al.add(Ty::F32, false); // r cast to f32 (the rank value we are looking for)
    let acc = al.add(Ty::F32, true); // running sum_i i * eq(rank[l,i], r)
    let ee = al.add(Ty::Usize, true); // loop var over the expert axis
    let ee_lt = al.add(Ty::Bool, false);
    let roff = al.add(Ty::Usize, false); // lbase + ee
    let rv = al.add(Ty::F32, false); // rank[l, ee]
    let eqb = al.add(Ty::Bool, false); // rv == rf
    let eqf = al.add(Ty::F32, false); // cast eqb to f32
    let eif = al.add(Ty::F32, false); // ee cast to f32
    let diff = al.add(Ty::F32, false); // eif - acc (last-wins select delta)
    let contrib = al.add(Ty::F32, false); // eqf * diff (0 when rank!=r, else eif-acc)
    let acc_new = al.add(Ty::F32, false);
    let ee_new = al.add(Ty::Usize, false);
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
        // Card 165: out-of-range lanes (i >= len, the tail workgroup when out_numel % 64 != 0) must jump
        // straight to the bare-return bb6. Targeting bb3 skips bb2's init of `lbase`/`ee`/`acc`, so idle
        // lanes issued an unguarded `rank` load from uninitialized state: tolerated on L40S and wgpu/RADV,
        // but faults with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on ROCm gfx1151 and
        // CUDA_ERROR_ILLEGAL_ADDRESS on A100 sm_80.
        terminator: guard(cmp, 6, 2),
    };
    // bb2: l = i/k; r = i%k; lbase = l*e; rf = (f32) r; acc = 0; ee = 0.
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(l),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(k)),
            ),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(k)),
            ),
            Statement::Assign(
                Place::local(lbase),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(l)), cu(e)),
            ),
            Statement::Assign(
                Place::local(rf),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(Place::local(r)),
                },
            ),
            Statement::Assign(
                Place::local(acc),
                Rvalue::Use(Operand::Const(Constant::F32(0.0))),
            ),
            Statement::Assign(Place::local(ee), Rvalue::Use(cu(0))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(ee_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(ee)), cu(e)),
        )],
        terminator: guard(ee_lt, 5, 4),
    };
    // bb4: roff = lbase + ee; rv = rank[roff]; eqb = (rv == rf); eqf = (f32) eqb; eif = (f32) ee;
    //      LAST-WINS SELECT (matches the eval oracle `arg_top_k`, a scatter `out[rank[i]]=i`): on a
    //      match keep the largest such ee, so acc = eqf ? eif : acc = acc + eqf*(eif-acc). A plain sum
    //      (`acc += eqf*eif`) adds every tied id, giving out-of-range expert ids that index the dequant-matmul
    //      out of bounds (`expert*N*nword`) and lose the device (card 158). The select stays in [0,E).
    //      diff = eif - acc; contrib = eqf * diff; acc += contrib; ee += 1.
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(roff),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(lbase)),
                    copy(Place::local(ee)),
                ),
            ),
            Statement::Assign(Place::local(rv), Rvalue::Use(copy(elem(rank, roff)))),
            Statement::Assign(
                Place::local(eqb),
                Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(rv)), copy(Place::local(rf))),
            ),
            Statement::Assign(
                Place::local(eqf),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(Place::local(eqb)),
                },
            ),
            Statement::Assign(
                Place::local(eif),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(Place::local(ee)),
                },
            ),
            Statement::Assign(
                Place::local(diff),
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(eif)), copy(Place::local(acc))),
            ),
            Statement::Assign(
                Place::local(contrib),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(eqf)),
                    copy(Place::local(diff)),
                ),
            ),
            Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(contrib)),
                ),
            ),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(ee_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(ee)), cu(1)),
            ),
            Statement::Assign(Place::local(ee), Rvalue::Use(copy(Place::local(ee_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb5: out[i] = acc.
    let bb5 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, i),
            Rvalue::Use(copy(Place::local(acc))),
        )],
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6])
}
