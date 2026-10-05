use crate::KernelGenError;
use crate::helpers::{Alloc, WeightLane, WeightLayout, copy, elem, guard, ld, local, slice_f32};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty, WorkgroupLocalDecl,
};
use poot_target::BufferStorage;

/// The synthesizer's element-by-element tiled-GEMM lowering: the `TiledRegion` tile IR lowered to a `Body`
/// from a small descriptor (M, K, N, tile size), instead of selecting a hand-authored kernel. It generates the
/// LDS-staged tiled GEMM with the same structure as `tiled_gemm_dt` (same coarsened grid, same RADV-safe
/// patterns), restricted to the f32 / 2-D / single-dispatch / no-bias case; the batch / bias / dtype /
/// chunk-offset generalizations stay in `tiled_gemm_dt`.
///
/// RADV-safety is by construction, as in `tiled_gemm_dt`: the K-tile loop body is branchless (min-clamped
/// indices + 0/1 validity masks, never an `if` in the barrier'd loop); the inner dot is unrolled (no nested
/// loop); the workgroup is `ts*ts = 64` lanes (one wave); the store guards are post-loop. Each lane computes 2
/// output rows (thread-coarsened, `row1 = row0 + ts`) so the grid matches `dispatch_grid`'s
/// `ceil(M/2ts) * ceil(N/ts)` workgroups. Params: `_1 a` \[M*K\] f32, `_2 b` \[K*N\] (read as `layout` says, in
/// the `weight` storage: `BufferStorage::f32()` or, Card 1007, `BufferStorage::f16_packed()`), `out` \[M*N\].
#[allow(clippy::too_many_arguments)] // one baked dimension or mode per argument, as in the sibling generators
pub fn tiled_region(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    ts: usize,
    tile_offset: usize,
    layout: WeightLayout,
    weight: BufferStorage,
) -> Result<Body, KernelGenError> {
    let weight_lane = WeightLane::of("tiled_region", weight)?;
    let ts2 = ts * ts;
    let tiles_n = n.div_ceil(ts);
    let kt_count = k.div_ceil(ts);
    // K-unroll: widen the per-barrier K-chunk from `ts` to `K_UNROLL*ts`, so the K-loop pays its two barriers
    // once per K_UNROLL sub-tiles instead of once per sub-tile (K=1536, ts=8 drops ~384 barriers/workgroup to
    // ~96). Falls back to u_count=1 whenever K isn't a multiple of K_UNROLL*ts, so kt_count is always
    // divisible by u_count.
    const K_UNROLL: usize = 4;
    let u_count = if k.is_multiple_of(K_UNROLL * ts) {
        K_UNROLL
    } else {
        1
    };

    // params: a [M*K] f32, b [K*N] (the weight lane), out [M*N] f32 (last).
    let params = vec![
        ld(Ty::Unit, false),         // 0
        ld(slice_f32(false), false), // 1 a [M*K]
        weight_lane.param(),         // 2 b [K*N]
        ld(slice_f32(true), true),   // 3 out [M*N]
    ];
    let n_inputs = (params.len() - 1) as u32;
    let c_idx = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let a = local(1);
    let b = local(2);
    let c = local(c_idx);

    let g = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);
    let tr = al.add(Ty::Usize, false);
    let tc = al.add(Ty::Usize, false);
    let trow = al.add(Ty::Usize, false);
    let tcol = al.add(Ty::Usize, false);
    let row = al.add(Ty::Usize, false);
    let row1 = al.add(Ty::Usize, false);
    let col = al.add(Ty::Usize, false);
    let tmp = al.add(Ty::Usize, false);
    let acc = al.add(Ty::F32, true);
    let acc1 = al.add(Ty::F32, true);
    let kt = al.add(Ty::Usize, true);
    let kt_lt = al.add(Ty::Bool, false);
    let a_k = al.add(Ty::Usize, false);
    let b_k = al.add(Ty::Usize, false);
    let rowok = al.add(Ty::Bool, false);
    let row1ok = al.add(Ty::Bool, false);
    let akok = al.add(Ty::Bool, false);
    let colok = al.add(Ty::Bool, false);
    let bkok = al.add(Ty::Bool, false);
    let rmask = al.add(Ty::F32, false);
    let rmask1 = al.add(Ty::F32, false);
    let kmask = al.add(Ty::F32, false);
    let cmask = al.add(Ty::F32, false);
    let bkmask = al.add(Ty::F32, false);
    let aidx = al.add(Ty::Usize, false);
    let bidx = al.add(Ty::Usize, false);
    let val = al.add(Ty::F32, false);
    let as_idx = al.add(Ty::Usize, false);
    // `as_idx0` is the K-unroll row0 LDS write slot (`as_idx` doubles as the row1 slot, reused since each uu
    // iteration fully overwrites it before use); `bs_slot` is the per-uu Bs write slot (`l + uu*ts2`).
    let as_idx0 = al.add(Ty::Usize, false);
    let bs_slot = al.add(Ty::Usize, false);
    let bs_idx = al.add(Ty::Usize, false);
    let as_v = al.add(Ty::F32, false);
    let bs_v = al.add(Ty::F32, false);
    let tr_base = al.add(Ty::Usize, false);
    let prod = al.add(Ty::F32, false);
    let acc_new = al.add(Ty::F32, false);
    let kt_new = al.add(Ty::Usize, false);
    let wrok = al.add(Ty::Bool, false);
    let colok2 = al.add(Ty::Bool, false);
    let storeok = al.add(Ty::Bool, false);
    let wrok1 = al.add(Ty::Bool, false);
    let storeok1 = al.add(Ty::Bool, false);
    let cidx = al.add(Ty::Usize, false);

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |bb: u32| Terminator::Goto {
        target: BlockId { index: bb },
    };
    let mulc = |dst: Local, x: Local, cc: usize| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(x)), cu(cc)),
        )
    };
    let addl = |dst: Local, x: Local, y: Local| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(x)), copy(Place::local(y))),
        )
    };
    // dst = x + (constant) cc, used for the per-uu K-unroll offsets (`+uu*ts`, `+uu*ts2`, `+uu*2*ts2`), all
    // baked-in constants at generation time (uu is a Rust `for` index, not a MIR value).
    let addc = |dst: Local, x: Local, cc: usize| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(x)), cu(cc)),
        )
    };
    let ltc = |dst: Local, x: Local, cc: usize| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(x)), cu(cc)),
        )
    };
    let castf = |dst: Local, src: Local| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(Place::local(src)),
            },
        )
    };
    let fmul = |dst: Local, x: Local, y: Local| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(x)), copy(Place::local(y))),
        )
    };
    let minc = |dst: Local, x: Local, cc: usize| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Min, copy(Place::local(x)), cu(cc)),
        )
    };

    // SEAM: fetch one activation element at the precomputed flat index `idx`, clamp it in-bounds, load f32, and
    // zero it out of range via the two 0/1 masks, writing the result into LDS array `lds` at `slot`. The weight
    // load below is the same shape around its lane's read.
    let load_masked = |buf: Local,
                       idx: Local,
                       max_idx: usize,
                       maskx: Local,
                       masky: Local,
                       lds: u8,
                       slot: Local|
     -> Vec<Statement> {
        vec![
            minc(idx, idx, max_idx),
            Statement::Assign(Place::local(val), Rvalue::Use(copy(elem(buf, idx)))),
            fmul(val, val, maskx),
            fmul(val, val, masky),
            Statement::WorkgroupLocalWrite {
                idx: copy(Place::local(slot)),
                value: copy(Place::local(val)),
                array: lds,
            },
        ]
    };

    // bb0: g = GroupX
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(g),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    // bb1: l = LocalX
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(l),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };
    // bb2: decode tile coords; row0 = trow*2ts + tr, row1 = row0 + ts, col = tcol*ts + tc; init acc/acc1/kt.
    // Chunking: a baked `g += tile_offset` (before the tile decode) maps this chunk's workgroups 0..count to the
    // global tile range [tile_offset, tile_offset+count), so a GEMM with >= 2^15 tiles can be dispatched as
    // several sub-2^15 chunks. tile_offset == 0 is the normal single dispatch.
    let mut bb2_stmts = Vec::new();
    if tile_offset > 0 {
        bb2_stmts.push(Statement::Assign(
            Place::local(g),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(g)), cu(tile_offset)),
        ));
    }
    bb2_stmts.extend([
        Statement::Assign(
            Place::local(tr),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(l)), cu(ts)),
        ),
        Statement::Assign(
            Place::local(tc),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(l)), cu(ts)),
        ),
        Statement::Assign(
            Place::local(trow),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(g)), cu(tiles_n)),
        ),
        Statement::Assign(
            Place::local(tcol),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(g)), cu(tiles_n)),
        ),
        mulc(tmp, trow, 2 * ts),
        addl(row, tmp, tr),
        Statement::Assign(
            Place::local(row1),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(row)), cu(ts)),
        ),
        mulc(tmp, tcol, ts),
        addl(col, tmp, tc),
        Statement::Assign(Place::local(acc), Rvalue::Use(cf(0.0))),
        Statement::Assign(Place::local(acc1), Rvalue::Use(cf(0.0))),
        Statement::Assign(Place::local(kt), Rvalue::Use(cu(0))),
    ]);
    let bb2 = BasicBlock {
        statements: bb2_stmts,
        terminator: goto(3),
    };
    // bb3 (k-tile header): kt_lt = kt < kt_count; if kt>=count -> store guard(7), else load(4).
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(kt_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(kt)), cu(kt_count)),
        )],
        terminator: guard(kt_lt, 7, 4),
    };
    // bb4 (branchless cooperative load + LDS write): a_k = kt*ts + uu*ts + tc, b_k = kt*ts + uu*ts + tr; masks;
    // then the two A rows + the B element via the `load_masked` seam. Barrier before the dot.
    //
    // The per-sub-tile body below is emitted `u_count` times (a generation-time `for uu in 0..u_count` unroll, not
    // a MIR loop), so bb4 loads `u_count` consecutive K sub-tiles into distinct LDS regions before its single
    // load-barrier (the `Barrier { target: 5 }` terminator below, emitted once). Each uu's K sub-tile is `kt+uu`,
    // so every `kt*ts` offset gets a constant `+uu*ts`; As writes land at `uu*2*ts2 + ...` (row0) /
    // `uu*2*ts2 + ts2 + ...` (row1); Bs writes land at `uu*ts2 + ...` (via the shared `bs_slot` local).
    let mut bb4_stmts: Vec<Statement> = Vec::new();
    for uu in 0..u_count {
        bb4_stmts.extend([
            mulc(tmp, kt, ts),
            addc(tmp, tmp, uu * ts),
            addl(a_k, tmp, tc),
            mulc(tmp, kt, ts),
            addc(tmp, tmp, uu * ts),
            addl(b_k, tmp, tr),
            ltc(rowok, row, m),
            ltc(row1ok, row1, m),
            ltc(akok, a_k, k),
            ltc(colok, col, n),
            ltc(bkok, b_k, k),
            castf(rmask, rowok),
            castf(rmask1, row1ok),
            castf(kmask, akok),
            castf(cmask, colok),
            castf(bkmask, bkok),
        ]);
        // A row0 -> As[uu*2*ts2 + l]; A row1 -> As[uu*2*ts2 + ts2 + l]; B -> Bs[uu*ts2 + l] (bs_slot).
        bb4_stmts.extend([mulc(tmp, row, k), addl(aidx, tmp, a_k)]);
        bb4_stmts.extend([addc(as_idx0, l, uu * 2 * ts2)]);
        bb4_stmts.extend(load_masked(a, aidx, m * k - 1, rmask, kmask, 0, as_idx0));
        bb4_stmts.extend([
            mulc(tmp, row1, k),
            addl(aidx, tmp, a_k),
            addc(as_idx, l, uu * 2 * ts2 + ts2),
        ]);
        bb4_stmts.extend(load_masked(a, aidx, m * k - 1, rmask1, kmask, 0, as_idx));
        bb4_stmts.extend([addc(bs_slot, l, uu * ts2)]);
        // B (weight) -> Bs[l]: the load_masked seam's clamp and masks around the weight lane's read (an f32 load,
        // or a packed-F16 word load and in-register decode). Kn reads `b_k * N + col`, Nk (checkpoint order)
        // reads `col * K + b_k`; both stay inside `k * n` elements, so the clamp is the same.
        bb4_stmts.extend(match layout {
            WeightLayout::Kn => [mulc(tmp, b_k, n), addl(bidx, tmp, col)],
            WeightLayout::Nk => [mulc(tmp, col, k), addl(bidx, tmp, b_k)],
        });
        bb4_stmts.push(minc(bidx, bidx, k * n - 1));
        bb4_stmts.extend(weight_lane.read(&mut al, b, bidx, val));
        bb4_stmts.extend([
            fmul(val, val, cmask),
            fmul(val, val, bkmask),
            Statement::WorkgroupLocalWrite {
                idx: copy(Place::local(bs_slot)),
                value: copy(Place::local(val)),
                array: 1,
            },
        ]);
    }
    let bb4 = BasicBlock {
        statements: bb4_stmts,
        terminator: Terminator::Barrier {
            target: BlockId { index: 5 },
        },
    };
    // bb5 (unrolled inner dot, the accumulator seam): tr_base = tr*ts; for uu in 0..u_count, for tt in 0..ts
    // both rows accumulate As[..]*Bs[uu*ts2+tt*ts+tc] (the same Bs value, the coarsening reuse). The outer `uu`
    // loop reads the K_UNROLL slabs bb4 packed into LDS at `uu*2*ts2` (As)/`uu*ts2` (Bs). Summation order is slab
    // (uu) ascending, then tt ascending, identical to the kt-major-then-tt-major order over the full K range, so
    // `acc`/`acc1` accumulate the same products in the same order (bit-exact).
    let mut bb5_stmts = vec![mulc(tr_base, tr, ts)];
    for uu in 0..u_count {
        for tt in 0..ts {
            // Bs[uu*ts2 + tt*ts + tc]
            bb5_stmts.push(Statement::Assign(
                Place::local(bs_idx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(tc)), cu(uu * ts2 + tt * ts)),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(bs_v),
                Rvalue::WorkgroupLocalRead {
                    idx: copy(Place::local(bs_idx)),
                    array: 1,
                },
            ));
            // row0: acc += As[uu*2*ts2 + tr_base + tt] * bs_v
            bb5_stmts.push(Statement::Assign(
                Place::local(as_idx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(tr_base)),
                    cu(uu * 2 * ts2 + tt),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(as_v),
                Rvalue::WorkgroupLocalRead {
                    idx: copy(Place::local(as_idx)),
                    array: 0,
                },
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(as_v)),
                    copy(Place::local(bs_v)),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(prod)),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(acc),
                Rvalue::Use(copy(Place::local(acc_new))),
            ));
            // row1: acc1 += As[uu*2*ts2 + ts2 + tr_base + tt] * bs_v
            bb5_stmts.push(Statement::Assign(
                Place::local(as_idx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(tr_base)),
                    cu(uu * 2 * ts2 + ts2 + tt),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(as_v),
                Rvalue::WorkgroupLocalRead {
                    idx: copy(Place::local(as_idx)),
                    array: 0,
                },
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(as_v)),
                    copy(Place::local(bs_v)),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc1)),
                    copy(Place::local(prod)),
                ),
            ));
            bb5_stmts.push(Statement::Assign(
                Place::local(acc1),
                Rvalue::Use(copy(Place::local(acc_new))),
            ));
        }
    }
    let bb5 = BasicBlock {
        statements: bb5_stmts,
        terminator: goto(6),
    };
    // bb6 (latch): kt += u_count (advance by the whole unrolled slab); barrier so all lanes finish reading LDS
    // before the next slab overwrites it.
    let bb6 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(kt_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(kt)), cu(u_count)),
            ),
            Statement::Assign(Place::local(kt), Rvalue::Use(copy(Place::local(kt_new)))),
        ],
        terminator: Terminator::Barrier {
            target: BlockId { index: 3 },
        },
    };
    // store one output row: cidx = row_l*N + col; out[cidx] = acc_l.
    let store_block = |row_l: Local, acc_l: Local, next: u32, ret: bool| -> BasicBlock {
        BasicBlock {
            statements: vec![
                mulc(tmp, row_l, n),
                addl(cidx, tmp, col),
                Statement::Assign(elem(c, cidx), Rvalue::Use(copy(Place::local(acc_l)))),
            ],
            terminator: if ret { Terminator::Return } else { goto(next) },
        }
    };
    // bb7 (row0 store guard): storeok = (row0<M) & (col<N); else -> row1 guard(9), then store row0 (8).
    let bb7 = BasicBlock {
        statements: vec![
            ltc(wrok, row, m),
            ltc(colok2, col, n),
            Statement::Assign(
                Place::local(storeok),
                Rvalue::BinaryOp(
                    BinOp::BitAnd,
                    copy(Place::local(wrok)),
                    copy(Place::local(colok2)),
                ),
            ),
        ],
        terminator: guard(storeok, 9, 8),
    };
    // bb8 (store row0): out[row0] = acc; fall through to the row1 guard (9).
    let bb8 = store_block(row, acc, 9, false);
    // bb9 (row1 store guard): storeok1 = (row1<M) & (col<N) (col<N reuses colok2); else -> ret(11).
    let bb9 = BasicBlock {
        statements: vec![
            ltc(wrok1, row1, m),
            Statement::Assign(
                Place::local(storeok1),
                Rvalue::BinaryOp(
                    BinOp::BitAnd,
                    copy(Place::local(wrok1)),
                    copy(Place::local(colok2)),
                ),
            ),
        ],
        terminator: guard(storeok1, 11, 10),
    };
    // bb10 (store row1): out[row1] = acc1; return.
    let bb10 = store_block(row1, acc1, 0, true);
    // bb11: return.
    let bb11 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    let mut body = Body::new(
        name,
        n_inputs,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11],
    );
    body.workgroup_size = [ts2 as u32, 1, 1];
    body.workgroup_locals = vec![
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            // As: 2*ts x ts (2 rows per lane) x u_count K-unroll slabs. u_count=4 -> 512 f32 = 2KB.
            len: (2 * ts2 * u_count) as u32,
        },
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            // Bs: ts x ts x u_count K-unroll slabs. u_count=4 -> 256 f32 = 1KB.
            len: (ts2 * u_count) as u32,
        },
    ];
    Ok(body)
}
