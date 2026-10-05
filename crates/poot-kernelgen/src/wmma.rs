use crate::KernelGenError;
use crate::helpers::{
    Alloc, batch_eff, copy, elem, guard, ld, local, row_major_strides, slice_bf16, slice_dtype,
    slice_f32,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty, WmmaDtype, WmmaShape,
};

/// Probe: WMMA-load both A and B fragments from a bf16 LDS tile (`WmmaLoadLds`), mma, store to a global f32 C.
/// Exercises the `WmmaLoadLds` capability the fused tensor-core dequant gemm needs (it dequants a packed int4
/// B-tile into a bf16 LDS tile, then WMMA-loads it). Compile-only: the LDS is zero-initialized here, so it is a
/// codegen check (does `wmma.load` from `.shared` lower), not a value test. NVPTX-only (see `WmmaLoadLds`'s doc).
#[cfg(test)]
pub(crate) fn wmma_loadlds_probe(name: &str) -> Body {
    use poot_kernel_ir::{WmmaMat, WorkgroupLocalDecl};
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(true), true), // 1 c (f32 out [16x16])
    ]);
    let c = local(1);
    let idx0 = al.add(Ty::Usize, false);
    let af = al.add(Ty::I32, false);
    let bf = al.add(Ty::I32, false);
    let acc = al.add(Ty::F32, false);

    let stmts = vec![
        Statement::Assign(
            Place::local(idx0),
            Rvalue::Use(Operand::Const(Constant::Usize(0))),
        ),
        Statement::WmmaZero {
            dtype: WmmaDtype::Bf16,
            shape: WmmaShape::M16N16K16,
            dst: acc,
        },
        Statement::WmmaLoadLds {
            which: WmmaMat::A,
            dtype: WmmaDtype::Bf16,
            shape: WmmaShape::M16N16K16,
            array: 0,
            stride: 16,
            dst: af,
        },
        Statement::WmmaLoadLds {
            which: WmmaMat::B,
            dtype: WmmaDtype::Bf16,
            shape: WmmaShape::M16N16K16,
            array: 0,
            stride: 16,
            dst: bf,
        },
        Statement::WmmaMma {
            dtype: WmmaDtype::Bf16,
            shape: WmmaShape::M16N16K16,
            a: af,
            b: bf,
            c: acc,
            dst: acc,
        },
        Statement::WmmaStore {
            dtype: WmmaDtype::Bf16,
            shape: WmmaShape::M16N16K16,
            tile: elem(c, idx0),
            stride: 16,
            src: acc,
        },
    ];
    let bb0 = BasicBlock {
        statements: stmts,
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 1, al.locals, vec![bb0]);
    body.workgroup_size = [32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::BF16,
        len: 256,
    }];
    body
}

/// Tiled tensor-core gemm: `C = A @ B` over `m16n16k16` WMMA tiles, bf16 inputs / f32 accumulate, output
/// storage dtype `dt` (f32 or bf16). One block of 256 threads computes one 16x16 output tile: warp 0 runs the
/// WMMA K-loop, loading the A and B fragments directly from global memory (the WMMA load takes a
/// (pointer, leading-dim-stride)). There is no LDS input staging; multi-warp LDS tile sharing is a deferred
/// perf follow-on. The f32 accumulator fragment updates in place (`c` and `dst` are the same `Local`) across
/// the K-loop, so it loop-carries the ordinary way (an alloca'd local, see `poot_codegen::emit`'s
/// `frag_layout` doc) with no extra copy statement.
///
/// Bf16 (not the target-neutral `wmma_tile`'s F16): the production dense/dequant path keeps weights
/// bf16-resident, and this generator stays AMDGCN/NVPTX-only (`WmmaDtype::Bf16` has no SPIR-V lowering, see
/// `WmmaDtype`'s doc) - a second, K==16-only sibling for SPIR-V coopmat is `matmul_tensorcore_coopmat`: a
/// coopmat accumulator fragment cannot loop-carry across a back-edge without a `phi` node this codegen does
/// not insert, so the two generators stay separate rather than one shared K-loop body (card 530's "one
/// generator" unification applies to the straight-line fragment op, not to this loop-carrying control flow,
/// which is a real per-target SSA constraint, not an op-shape difference).
///
/// Output epilogue depends on `dt`: f32 stores the D fragment straight to C (`WmmaStore`); bf16 stages the D
/// fragment to a per-block f32 LDS tile (`WmmaStoreLds`), barriers, then the block's 256 threads narrow-copy the
/// 256 tile elements to the bf16 output (one per thread), since the WMMA D-store is f32-only.
///
/// Shapes are the same batched layout as `matmul_batched_dt`: out `[..batch, M, N]`, a `[..batch, M, K]`, b
/// `[..batch, K, N]`, all row-major. M, N, K must be multiples of 16 (the planner checks; non-aligned shapes fall
/// back to the serial `matmul_batched_dt`). Launch with `out_numel` threads, wg `[256,1,1]`, so blocks =
/// `out_numel/256 = num_tiles = prod(batch) * (M/16) * (N/16)` (`dispatch_grid`'s default `out_numel` is
/// correct). NVPTX-only in scope today (WMMA has no SPIR-V equivalent).
pub fn matmul_tensorcore(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<Body, KernelGenError> {
    use poot_kernel_ir::{WmmaMat, WorkgroupLocalDecl};
    let bf16 = dt != Ty::F32;
    let r = out_shape.len();
    let m = out_shape[r - 2];
    let n = out_shape[r - 1];
    let k = a_shape[a_shape.len() - 1];
    for (dim, value) in [("M", m), ("N", n), ("K", k)] {
        if !value.is_multiple_of(16) {
            return Err(KernelGenError::NotDivisible {
                generator: "matmul_tensorcore",
                dim: dim.to_string(),
                value,
                divisor: 16,
            });
        }
    }
    let tiles_m = m / 16;
    let tiles_n = n / 16;
    let batch_shape = &out_shape[..r - 2];
    let batch_strides = row_major_strides(batch_shape); // empty if r == 2
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);
    let a_beff = batch_eff(r - 2, a_shape, &a_strides);
    let b_beff = batch_eff(r - 2, b_shape, &b_strides);
    let batch_count: usize = batch_shape.iter().product::<usize>().max(1);
    let num_tiles = batch_count * tiles_m * tiles_n;
    // leading-dimension strides (full matrix row width) for the WMMA (ptr, stride) loads/stores.
    let a_ld = k as u32; // A row-major [..,M,K]
    let b_ld = n as u32; // B row-major [..,K,N]
    let c_ld = n as u32; // C row-major [..,M,N]

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_bf16(false), false),            // 1 a (bf16)
        ld(slice_bf16(false), false),            // 2 b (bf16)
        ld(slice_dtype(dt.clone(), true), true), // 3 c (dt out)
    ]);
    let (a, b, c) = (local(1), local(2), local(3));
    let tid = al.add(Ty::Usize, false);
    let block = al.add(Ty::Usize, false);
    let localid = al.add(Ty::Usize, false);
    let is_w0 = al.add(Ty::Bool, false);
    let cmp = al.add(Ty::Bool, false);
    let tn = al.add(Ty::Usize, false);
    let tmp = al.add(Ty::Usize, false);
    let tm = al.add(Ty::Usize, false);
    let bidx = al.add(Ty::Usize, false);
    let cd = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let tmpoff = al.add(Ty::Usize, false);
    let a_bb = al.add(Ty::Usize, true);
    let b_bb = al.add(Ty::Usize, true);
    let m0 = al.add(Ty::Usize, false);
    let n0 = al.add(Ty::Usize, false);
    let a_row_base = al.add(Ty::Usize, false);
    let b_col_base = al.add(Ty::Usize, false);
    let c_off = al.add(Ty::Usize, true);
    let k0 = al.add(Ty::Usize, true);
    let k_lt = al.add(Ty::Bool, false);
    let kn = al.add(Ty::Usize, false);
    let a_off = al.add(Ty::Usize, false);
    let b_off = al.add(Ty::Usize, false);
    let af = al.add(Ty::I32, false);
    let bf = al.add(Ty::I32, false);
    let acc = al.add(Ty::F32, true); // accumulator fragment: zero-seeded, then updated in place each K-step
    // bf16 narrow epilogue temps.
    let row = al.add(Ty::Usize, false);
    let col = al.add(Ty::Usize, false);
    let gidx = al.add(Ty::Usize, false);
    let ldsv = al.add(Ty::F32, false);

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let set = |l: Local, rv: Rvalue| Statement::Assign(Place::local(l), rv);
    let binc =
        |op: BinOp, src: Local, x: usize| Rvalue::BinaryOp(op, copy(Place::local(src)), cu(x));
    let bin = |op: BinOp, x: Local, y: Local| {
        Rvalue::BinaryOp(op, copy(Place::local(x)), copy(Place::local(y)))
    };

    // block indices differ between the f32 (no epilogue) and bf16 (LDS-stage + barrier + narrow) tails.
    let (kloop, kbody, store) = (3u32, 4u32, 5u32);
    let (ret, barrier, narrow) = if bf16 {
        (8u32, 6u32, 7u32)
    } else {
        (6u32, 6u32, 6u32)
    };

    // bb1: block = tid / 256 (one block per tile); valid = block < num_tiles.
    let bb1 = BasicBlock {
        statements: vec![
            set(block, binc(BinOp::Div, tid, 256)),
            set(cmp, binc(BinOp::Lt, block, num_tiles)),
        ],
        terminator: guard(cmp, ret, 2),
    };

    // bb2: decode the tile coords (from `block`), batch bases, tile bases, c_off; localid + warp-0 flag; zero
    // the accumulator fragment; k0 = 0. Non-warp-0 threads skip the WMMA: bf16 -> barrier (they help narrow),
    // f32 -> return.
    let mut pre = vec![
        set(a_bb, Rvalue::Use(cu(0))),
        set(b_bb, Rvalue::Use(cu(0))),
        set(localid, binc(BinOp::Rem, tid, 256)),
        set(is_w0, binc(BinOp::Lt, localid, 32)),
        set(tn, binc(BinOp::Rem, block, tiles_n)),
        set(tmp, binc(BinOp::Div, block, tiles_n)),
        set(tm, binc(BinOp::Rem, tmp, tiles_m)),
        set(bidx, binc(BinOp::Div, tmp, tiles_m)),
    ];
    // batch offsets: for each out batch dim, coord = (bidx / batch_strides[d]) % batch_shape[d].
    for d in 0..(r - 2) {
        if a_beff[d] == 0 && b_beff[d] == 0 {
            continue;
        }
        pre.push(set(term, binc(BinOp::Div, bidx, batch_strides[d])));
        pre.push(set(cd, binc(BinOp::Rem, term, batch_shape[d])));
        if a_beff[d] != 0 {
            pre.push(set(term, binc(BinOp::Mul, cd, a_beff[d])));
            pre.push(set(tmpoff, bin(BinOp::Add, a_bb, term)));
            pre.push(set(a_bb, Rvalue::Use(copy(Place::local(tmpoff)))));
        }
        if b_beff[d] != 0 {
            pre.push(set(term, binc(BinOp::Mul, cd, b_beff[d])));
            pre.push(set(tmpoff, bin(BinOp::Add, b_bb, term)));
            pre.push(set(b_bb, Rvalue::Use(copy(Place::local(tmpoff)))));
        }
    }
    pre.push(set(m0, binc(BinOp::Mul, tm, 16)));
    pre.push(set(n0, binc(BinOp::Mul, tn, 16)));
    // a_row_base = a_bb + m0*K (A tile top-left at K-step 0); in the loop a_off = a_row_base + k0.
    pre.push(set(term, binc(BinOp::Mul, m0, k)));
    pre.push(set(a_row_base, bin(BinOp::Add, a_bb, term)));
    // b_col_base = b_bb + n0; in the loop b_off = b_col_base + k0*N.
    pre.push(set(b_col_base, bin(BinOp::Add, b_bb, n0)));
    // c_off = bidx*(M*N) + m0*N + n0 (C contiguous; the tile's top-left flat index).
    pre.push(set(c_off, binc(BinOp::Mul, bidx, m * n)));
    pre.push(set(term, binc(BinOp::Mul, m0, n)));
    pre.push(set(c_off, bin(BinOp::Add, c_off, term)));
    pre.push(set(c_off, bin(BinOp::Add, c_off, n0)));
    pre.push(Statement::WmmaZero {
        dtype: WmmaDtype::Bf16,
        shape: WmmaShape::M16N16K16,
        dst: acc,
    });
    pre.push(set(k0, Rvalue::Use(cu(0))));

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(tid),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb2 = BasicBlock {
        statements: pre,
        // warp 0 -> the K loop; others -> barrier (bf16, to help narrow) or return (f32).
        terminator: guard(is_w0, barrier, kloop),
    };
    let bb3 = BasicBlock {
        statements: vec![set(k_lt, binc(BinOp::Lt, k0, k))],
        terminator: guard(k_lt, store, kbody),
    };
    // bb4: load A/B fragments at the current K-step from global, mma-accumulate in place, advance k0 by 16.
    let mut body4 = vec![set(a_off, bin(BinOp::Add, a_row_base, k0))];
    body4.push(Statement::WmmaLoad {
        dtype: WmmaDtype::Bf16,
        shape: WmmaShape::M16N16K16,
        which: WmmaMat::A,
        tile: elem(a, a_off),
        stride: a_ld,
        dst: af,
    });
    body4.push(set(kn, binc(BinOp::Mul, k0, n)));
    body4.push(set(b_off, bin(BinOp::Add, b_col_base, kn)));
    body4.push(Statement::WmmaLoad {
        dtype: WmmaDtype::Bf16,
        shape: WmmaShape::M16N16K16,
        which: WmmaMat::B,
        tile: elem(b, b_off),
        stride: b_ld,
        dst: bf,
    });
    body4.push(Statement::WmmaMma {
        dtype: WmmaDtype::Bf16,
        shape: WmmaShape::M16N16K16,
        a: af,
        b: bf,
        c: acc,
        dst: acc,
    });
    body4.push(set(k0, binc(BinOp::Add, k0, 16)));
    let bb4 = BasicBlock {
        statements: body4,
        terminator: Terminator::Goto {
            target: BlockId { index: kloop },
        },
    };

    let mut blocks = vec![bb0, bb1, bb2, bb3, bb4];
    if bf16 {
        // bb5: warp 0 stages the D fragment into the per-block f32 LDS tile -> barrier.
        blocks.push(BasicBlock {
            statements: vec![Statement::WmmaStoreLds {
                shape: WmmaShape::M16N16K16,
                array: 0,
                stride: 16,
                src: acc,
            }],
            terminator: Terminator::Goto {
                target: BlockId { index: barrier },
            },
        });
        // bb6: barrier (all 256 threads of the block) -> narrow.
        blocks.push(BasicBlock {
            statements: vec![],
            terminator: Terminator::Barrier {
                target: BlockId { index: narrow },
            },
        });
        // bb7: each thread narrows LDS[localid] (row=localid/16, col=localid%16) to C[c_off + row*N + col].
        blocks.push(BasicBlock {
            statements: vec![
                set(row, binc(BinOp::Div, localid, 16)),
                set(col, binc(BinOp::Rem, localid, 16)),
                set(term, binc(BinOp::Mul, row, n)),
                set(gidx, bin(BinOp::Add, c_off, term)),
                set(gidx, bin(BinOp::Add, gidx, col)),
                set(
                    ldsv,
                    Rvalue::WorkgroupLocalRead {
                        idx: copy(Place::local(localid)),
                        array: 0,
                    },
                ),
                Statement::Assign(
                    elem(c, gidx),
                    Rvalue::Cast {
                        to: dt.clone(),
                        operand: copy(Place::local(ldsv)),
                    },
                ),
            ],
            terminator: Terminator::Return,
        });
        // bb8: return (the invalid-block / never-narrow exit).
        blocks.push(BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        });
    } else {
        // bb5: warp 0 stores the D fragment straight to C (f32).
        blocks.push(BasicBlock {
            statements: vec![Statement::WmmaStore {
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                tile: elem(c, c_off),
                stride: c_ld,
                src: acc,
            }],
            terminator: Terminator::Return,
        });
        // bb6: return (invalid block / non-warp-0 f32 exit).
        blocks.push(BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        });
    }

    let mut body = Body::new(name, 3, al.locals, blocks);
    body.workgroup_size = [256, 1, 1]; // one block (8 warps) per output tile; warp 0 does the WMMA
    if bf16 {
        body.workgroup_locals = vec![WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: 256, // one 16x16 f32 tile staged by warp 0, narrowed by the block
        }];
    }
    Ok(body)
}

/// SPIR-V cooperative-matrix tiled matmul: `out[M,N] = A[M,K] @ B[K,N]`, F16 inputs / F32 accumulate, over
/// independent `m16n16k16` coopmat tiles; the [`matmul_tensorcore`] sibling for RADV coopmat (see that
/// function's doc for why the two stay separate generators rather than one shared K-loop body). One
/// subgroup (32 invocations, gfx1151's wave32) computes one 16x16 output tile collectively (coopmat's
/// Subgroup scope means every invocation cooperates, so there is no "warp 0 does the WMMA, the rest
/// barrier" split).
///
/// Unlike `matmul_tensorcore` there is no K-loop: `K` must be exactly 16 (a single coopmat tile). A coopmat value
/// is one opaque SSA register (see `poot_kernel_ir::WmmaDtype`'s doc) and cannot be carried across a loop
/// back-edge without a `phi` node this codegen cannot insert; `WmmaZero`'s `OpCompositeConstruct` zero-seed
/// avoids that only because it is used once, straight-line, per tile. M/N can be multi-tile (tiles are
/// independent); only the last two dims of `out_shape`/`a_shape`/`b_shape` are read (leading dims must all be
/// size 1: the caller's `matmul_spirv_coopmat_eligible` gate enforces `batch_dims_product == 1`; see its doc in
/// `poot-graph-plan` for why real batching is deferred).
///
/// Output is always F32 (RADV coopmat config 13 has no f16-accumulate variant); an F16-output matmul would need
/// an extra narrowing store, as with [`matmul_tensorcore`]'s AMD arm.
pub fn matmul_tensorcore_coopmat(
    name: &str,
    out_shape: &[usize],
    a_shape: &[usize],
    _b_shape: &[usize],
) -> Result<Body, KernelGenError> {
    use poot_kernel_ir::WmmaMat;
    let r = out_shape.len();
    let (m, n) = (out_shape[r - 2], out_shape[r - 1]);
    let k = a_shape[a_shape.len() - 1];
    for (dim, value) in [("M", m), ("N", n)] {
        if !value.is_multiple_of(16) {
            return Err(KernelGenError::NotDivisible {
                generator: "matmul_tensorcore_coopmat",
                dim: dim.to_string(),
                value,
                divisor: 16,
            });
        }
    }
    if k != 16 {
        return Err(KernelGenError::NotEqualTo {
            generator: "matmul_tensorcore_coopmat",
            dim: "K".to_string(),
            value: k,
            expected: 16,
        });
    }
    let tiles_n = n / 16;
    let num_tiles = (m / 16) * tiles_n;
    let (a_ld, b_ld, c_ld) = (k as u32, n as u32, n as u32);

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::F16, false), false), // 1 a (f16, row-major MxK)
        ld(slice_dtype(Ty::F16, false), false), // 2 b (f16, row-major KxN)
        ld(slice_f32(true), true),              // 3 out (f32, row-major MxN)
    ]);
    let (a, b, out) = (local(1), local(2), local(3));
    let tid = al.add(Ty::Usize, false);
    let block = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let tn = al.add(Ty::Usize, false);
    let tm = al.add(Ty::Usize, false);
    let m0 = al.add(Ty::Usize, false);
    let n0 = al.add(Ty::Usize, false);
    let a_off = al.add(Ty::Usize, false);
    let b_off = al.add(Ty::Usize, false);
    let c_off = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let af = al.add(Ty::F16, false); // fragment handle (A)
    let bf = al.add(Ty::F16, false); // fragment handle (B)
    let acc = al.add(Ty::F32, false); // fragment handle (zero-seeded accumulator, updated in place)

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let set = |l: Local, rv: Rvalue| Statement::Assign(Place::local(l), rv);
    let binc =
        |op: BinOp, src: Local, x: usize| Rvalue::BinaryOp(op, copy(Place::local(src)), cu(x));
    let bin = |op: BinOp, x: Local, y: Local| {
        Rvalue::BinaryOp(op, copy(Place::local(x)), copy(Place::local(y)))
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(tid),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    // bb1: block = tid / 32 (one subgroup per tile); valid = block < num_tiles.
    let bb1 = BasicBlock {
        statements: vec![
            set(block, binc(BinOp::Div, tid, 32)),
            set(cmp, binc(BinOp::Lt, block, num_tiles)),
        ],
        terminator: guard(cmp, 3, 2),
    };
    // bb2: decode the tile coords, compute the A/B/out tile offsets (K == 16, so K-step 0 IS the
    // whole K range - a_off is the tile's top-left, no per-step advance), load A/B, zero-seed the
    // accumulator, mma, store D straight to `out`.
    let body_stmts = vec![
        set(tn, binc(BinOp::Rem, block, tiles_n)),
        set(tm, binc(BinOp::Div, block, tiles_n)),
        set(m0, binc(BinOp::Mul, tm, 16)),
        set(n0, binc(BinOp::Mul, tn, 16)),
        set(a_off, binc(BinOp::Mul, m0, k)), // A[m0, 0]
        set(b_off, Rvalue::Use(copy(Place::local(n0)))), // B[0, n0]
        set(term, binc(BinOp::Mul, m0, n)),
        set(c_off, bin(BinOp::Add, term, n0)), // out[m0, n0]
        Statement::WmmaLoad {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            which: WmmaMat::A,
            tile: elem(a, a_off),
            stride: a_ld,
            dst: af,
        },
        Statement::WmmaLoad {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            which: WmmaMat::B,
            tile: elem(b, b_off),
            stride: b_ld,
            dst: bf,
        },
        Statement::WmmaZero {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            dst: acc,
        },
        Statement::WmmaMma {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            a: af,
            b: bf,
            c: acc,
            dst: acc,
        },
        Statement::WmmaStore {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            tile: elem(out, c_off),
            stride: c_ld,
            src: acc,
        },
    ];
    let bb2 = BasicBlock {
        statements: body_stmts,
        terminator: Terminator::Return,
    };
    // bb3: return (invalid-block exit).
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    let mut body = Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3]);
    body.workgroup_size = [32, 1, 1]; // one subgroup (gfx1151 wave32) per output tile
    Ok(body)
}

#[cfg(test)]
mod tests {
    use poot_codegen::{Target, compile};

    use super::wmma_loadlds_probe;

    /// WmmaLoadLds (WMMA-load a fragment from a bf16 LDS tile) lowers to NVPTX with the tensor-core
    /// load reading `.shared`. Split out of `tests/bf16_probe.rs` (card 546c: its only other caller,
    /// poot-rocm-gpu's schedule-batch atomicity test, was deleted with the resident E4M3 region's
    /// `compile_schedule` stack, which made this an external-only consumer that `pub(crate)` cannot
    /// serve). The enabling capability for the fused tensor-core dequant gemm.
    #[test]
    #[ignore = "probe: compiles wmma_loadlds_probe, checks the LDS WMMA load"]
    fn wmma_loadlds_nvptx() {
        let dir = std::env::temp_dir().join("poot-wmma-lds");
        std::fs::create_dir_all(&dir).unwrap();
        let body = wmma_loadlds_probe("k");
        let out = dir.join("wmma_loadlds.ptx");
        let _ = compile(&body, Target::Nvptx, &out)
            .unwrap_or_else(|e| panic!("wmma_loadlds compile failed: {e}"));
        let ptx = std::fs::read_to_string(&out).unwrap();
        let n_load = ptx.matches("wmma.load").count();
        let shared = ptx.contains(".shared") || ptx.contains("shared");
        eprintln!("wmma_loadlds PTX: load={n_load} shared={shared}");
        assert!(n_load >= 2, "expected 2 wmma.load (A+B from LDS):\n{ptx}");
        assert!(shared, "expected the fragment load to read .shared (LDS)");
        assert!(compile(&body, Target::SpirvVulkan, &dir.join("x.spv")).is_err());
    }
}
