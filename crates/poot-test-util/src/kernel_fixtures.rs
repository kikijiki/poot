//! Kernel-IR `Body` fixtures, the reference interpreter's buffer helpers, the kernelgen generators with no
//! production caller, and the golden-IR emit helper (card 671: these had no production caller in their
//! declaring crates, only cross-crate kernel tests - dead under the strict dead-pub rule: a
//! test reference is never a use).
//!
//! `Buffer::from_f32s`/`from_u32s`/`to_f32s` read/write the struct's private fields and `Scalar` payload (the
//! type's only way to guarantee `elem` and `data` agree), so they cannot move bodily: ADR-0113 keeps them in
//! `poot_kernel_ir::interp` under `#[cfg(any(test, feature = "test-support"))]`, and callers here use that
//! path directly - no forwarder (a forwarder whose only purpose is to be a non-test reference is laundering,
//! ADR-0113's Context). Every other item here is its own complete implementation:
//! [`gather_axis0_dt`]/[`reduce_last`]/[`binary_broadcast`]/[`binary_broadcast_dt`]/
//! [`binary_broadcast_dt_views`]/[`fused_i32_views`] delegate to the still-pub, still-production-used
//! generators one layer down (`gather_axis0_index_dt`, `reduce_last_dt`, `binary_broadcast_dt_views_grid`,
//! `fused_i32_views_grid`); [`wg_sum`], [`workgroup_fence_spin_wait`] and [`wmma_tile`] are reconstructed here
//! because the small `Local`/`Place`/`Ty` constructors they need (`local`, `elem`, `copy`, `guard`, `ld`,
//! `slice_dtype`) are `pub(crate)` in `poot-kernelgen`'s own `helpers` module - each is a one-line wrapper over
//! an already-public `poot_kernel_ir` type, duplicated here rather than widened to `pub` crate-wide for a
//! handful of dead fixtures.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, Fp8Format, IndexAxis, Local, LocalDecl, Operand,
    Place, ProjectionElem, Rvalue, Statement, SwitchTargets, Terminator, Ty, WmmaDtype, WmmaMat,
    WmmaShape, WorkgroupLocalDecl,
};

// --- small IR-builder primitives, duplicated from `poot_kernelgen`'s crate-private `helpers` module ---------

fn local(i: u32) -> Local {
    Local { index: i }
}

fn elem(slice: Local, idx: Local) -> Place {
    Place {
        local: slice,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(idx)],
    }
}

fn copy(p: Place) -> Operand {
    Operand::Copy(p)
}

fn ld(ty: Ty, mutable: bool) -> LocalDecl {
    LocalDecl { ty, mutable }
}

fn guard(cmp: Local, zero_target: u32, otherwise: u32) -> Terminator {
    Terminator::SwitchInt {
        discr: copy(Place::local(cmp)),
        targets: SwitchTargets {
            branches: vec![(0, BlockId { index: zero_target })],
            otherwise: BlockId { index: otherwise },
        },
    }
}

fn slice_dtype(elem: Ty, mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(elem))),
    }
}

fn slice_f32(mutable: bool) -> Ty {
    slice_dtype(Ty::F32, mutable)
}

// --- poot-kernel-ir Body fixtures (ex poot-kernel-ir/src/fixtures.rs) ---------------------------------------

fn fp8_conversion_kernel(encode: bool) -> Body {
    let (input_ty, output_ty) = if encode {
        (Ty::F32, Ty::U32)
    } else {
        (Ty::U32, Ty::F32)
    };
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(input_ty.clone(), false), false),
        ld(slice_dtype(output_ty.clone(), true), true),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(input_ty, false),
        ld(output_ty, false),
    ];
    let (input, output, i, len, in_bounds, source, converted) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
    );
    let conversion = if encode {
        Rvalue::Fp8Encode {
            format: Fp8Format::E4M3Fn,
            operand: copy(Place::local(source)),
        }
    } else {
        Rvalue::Fp8Decode {
            format: Fp8Format::E4M3Fn,
            operand: copy(Place::local(source)),
        }
    };
    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(in_bounds),
                    Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
                ),
            ],
            terminator: guard(in_bounds, 3, 2),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(source), Rvalue::Use(copy(elem(input, i)))),
                Statement::Assign(Place::local(converted), conversion),
                Statement::Assign(elem(output, i), Rvalue::Use(copy(Place::local(converted)))),
            ],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Body::new(
        if encode {
            "e4m3fn_encode"
        } else {
            "e4m3fn_decode"
        },
        2,
        locals,
        blocks,
    )
}

/// Software fixture for `f32 -> E4M3FN byte`. Each output u32 carries one byte in its low lane.
pub fn e4m3fn_encode_kernel() -> Body {
    fp8_conversion_kernel(true)
}

/// Software fixture for `E4M3FN byte -> f32`. Each input u32 carries one byte in its low lane.
pub fn e4m3fn_decode_kernel() -> Body {
    fp8_conversion_kernel(false)
}

/// `vadd_loop(a: Slice<f32>, b: Slice<f32>, c: SliceMut<f32>)`:
/// `let mut i = 0; while i < c.len() { c[i] = a[i] + b[i]; i += 1; }`. The sequential form of
/// `poot_kernel_ir::fixtures::add_kernel`: one core walks the tile in a loop. This is the shape a single AIE
/// core runs (the XDNA2 NPU core ISA has no SPMD thread-index; the IRON harness tiles the data). Used by the
/// AIE-core (Peano `aie2p`) codegen path.
pub fn vadd_loop_kernel() -> Body {
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_f32(false), false),
        ld(slice_f32(true), true),
        ld(Ty::Usize, true),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(Ty::F32, false),
        ld(Ty::Usize, false),
    ];
    let (a, b, c, i, len, cmp, sum, i_new) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
        local(8),
    );

    let bb0 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(i),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
        )],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(sum),
                Rvalue::BinaryOp(BinOp::Add, copy(elem(a, i)), copy(elem(b, i))),
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(sum)))),
            Statement::Assign(
                Place::local(i_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(i)),
                    Operand::Const(Constant::Usize(1)),
                ),
            ),
            Statement::Assign(Place::local(i), Rvalue::Use(copy(Place::local(i_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 1 },
        },
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    Body::new("vadd_loop", 3, locals, vec![bb0, bb1, bb2, bb3])
}

/// Scalar GEMV `out[M] = mat[M,K] @ vec[K]`, all row-major f32, dims baked in as IR constants. Sequential
/// nested loops (outer row `i`, inner `j` with an f32 accumulator), no SPMD grid, as an AIE core runs it.
/// Beyond [`vadd_loop_kernel`] it exercises nested back-edged loops, an f32 accumulator carried across the
/// inner loop, a multiply, and `i*K + j` flat indexing.
pub fn gemv_loop_kernel(m: usize, k: usize) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),         // 0 ret
        ld(slice_f32(false), false), // 1 mat [M*K]
        ld(slice_f32(false), false), // 2 vec [K]
        ld(slice_f32(true), true),   // 3 out [M]
        ld(Ty::Usize, true),         // 4 i
        ld(Ty::Bool, false),         // 5 i_lt
        ld(Ty::F32, true),           // 6 acc
        ld(Ty::Usize, true),         // 7 j
        ld(Ty::Bool, false),         // 8 j_lt
        ld(Ty::Usize, false),        // 9 base = i*K
        ld(Ty::Usize, false),        // 10 aidx = base+j
        ld(Ty::F32, false),          // 11 va
        ld(Ty::F32, false),          // 12 vb
        ld(Ty::F32, false),          // 13 prod
        ld(Ty::F32, false),          // 14 acc_new
        ld(Ty::Usize, false),        // 15 j_new
        ld(Ty::Usize, false),        // 16 i_new
    ];
    let (mat, vec_, out, i, i_lt, acc, j, j_lt) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
        local(8),
    );
    let (base, aidx, va, vb, prod, acc_new, j_new, i_new) = (
        local(9),
        local(10),
        local(11),
        local(12),
        local(13),
        local(14),
        local(15),
        local(16),
    );
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cp = Place::local;
    let bin = |op, x, y| Rvalue::BinaryOp(op, x, y);
    let assign = |p, rv| Statement::Assign(p, rv);
    let sw = |discr, zero: u32, otherwise: u32| Terminator::SwitchInt {
        discr,
        targets: SwitchTargets {
            branches: vec![(0, BlockId { index: zero })],
            otherwise: BlockId { index: otherwise },
        },
    };
    let goto = |t: u32| Terminator::Goto {
        target: BlockId { index: t },
    };

    let bb0 = BasicBlock {
        statements: vec![assign(cp(i), Rvalue::Use(cu(0)))],
        terminator: goto(1),
    };
    let bb1 = BasicBlock {
        statements: vec![assign(cp(i_lt), bin(BinOp::Lt, copy(cp(i)), cu(m)))],
        terminator: sw(copy(cp(i_lt)), 6, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            assign(cp(acc), Rvalue::Use(Operand::Const(Constant::F32(0.0)))),
            assign(cp(j), Rvalue::Use(cu(0))),
        ],
        terminator: goto(3),
    };
    let bb3 = BasicBlock {
        statements: vec![assign(cp(j_lt), bin(BinOp::Lt, copy(cp(j)), cu(k)))],
        terminator: sw(copy(cp(j_lt)), 5, 4),
    };
    let bb4 = BasicBlock {
        statements: vec![
            assign(cp(base), bin(BinOp::Mul, copy(cp(i)), cu(k))),
            assign(cp(aidx), bin(BinOp::Add, copy(cp(base)), copy(cp(j)))),
            assign(cp(va), Rvalue::Use(copy(elem(mat, aidx)))),
            assign(cp(vb), Rvalue::Use(copy(elem(vec_, j)))),
            assign(cp(prod), bin(BinOp::Mul, copy(cp(va)), copy(cp(vb)))),
            assign(cp(acc_new), bin(BinOp::Add, copy(cp(acc)), copy(cp(prod)))),
            assign(cp(acc), Rvalue::Use(copy(cp(acc_new)))),
            assign(cp(j_new), bin(BinOp::Add, copy(cp(j)), cu(1))),
            assign(cp(j), Rvalue::Use(copy(cp(j_new)))),
        ],
        terminator: goto(3),
    };
    let bb5 = BasicBlock {
        statements: vec![
            assign(elem(out, i), Rvalue::Use(copy(cp(acc)))),
            assign(cp(i_new), bin(BinOp::Add, copy(cp(i)), cu(1))),
            assign(cp(i), Rvalue::Use(copy(cp(i_new)))),
        ],
        terminator: goto(1),
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    Body::new(
        "gemv_loop",
        3,
        locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6],
    )
}

/// `len_probe(len_src: Slice<f32>, out: SliceMut<f32>)`: `let i = thread_index(); if i < out.len() {
/// out[i] = len_src.len() as f32; }` - card 547b review F1's "small generated probe that writes the
/// length it received into its output": a shape-generic body that genuinely reads `len_src`'s runtime
/// `Len()` (never a value baked into the compiled code, unlike `poot-kernelgen`'s fixed-shape
/// `Add`/`Reduce` bodies), so dispatching it with `Arg::elems` smaller than `len_src`'s own device
/// buffer capacity (an arena slot reused from a larger occupant) directly exposes whether a runtime
/// publishes the dispatch's own logical length or the buffer's capacity.
pub fn len_probe_kernel() -> Body {
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 len_src
        ld(slice_f32(true), true),   // 2 out
        ld(Ty::Usize, false),        // 3 i
        ld(Ty::Usize, false),        // 4 out_len
        ld(Ty::Bool, false),         // 5 cmp
        ld(Ty::Usize, false),        // 6 probed_len = Len(len_src)
        ld(Ty::F32, false),          // 7 probed_len as f32
    ];
    let (len_src, out, i, out_len, cmp, probed, probed_f32) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
    );
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
            Statement::Assign(Place::local(out_len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    copy(Place::local(i)),
                    copy(Place::local(out_len)),
                ),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(probed), Rvalue::Len(Place::local(len_src))),
            Statement::Assign(
                Place::local(probed_f32),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(Place::local(probed)),
                },
            ),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(probed_f32)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new("len_probe", 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// `scale(x: Slice<f32>, y: SliceMut<f32>)`: `let i = thread_index(); if i < y.len() { y[i] = x[i] * x[i]; }`.
/// Single-input elementwise kernel (one read slot, a self-multiply).
pub fn square_kernel() -> Body {
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_f32(true), true),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(Ty::F32, false),
    ];
    let (x, y, i, len, cmp, sq) = (local(1), local(2), local(3), local(4), local(5), local(6));
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
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
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
                Place::local(sq),
                Rvalue::BinaryOp(BinOp::Mul, copy(elem(x, i)), copy(elem(x, i))),
            ),
            Statement::Assign(elem(y, i), Rvalue::Use(copy(Place::local(sq)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new("square", 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// Naive (non-tiled) matmul `C[M,N] = A[M,K] @ B[K,N]`, all row-major f32. Dims are baked in as IR
/// constants, so it needs no scalar params, LDS, or transcendentals: a 2-D grid (col=x, row=y), a K-loop
/// with an accumulator, and flat indexing.
pub fn matmul_kernel(m: usize, n: usize, k: usize) -> Body {
    let f32_ro = slice_f32(false);
    let f32_rw = slice_f32(true);
    let locals = vec![
        ld(Ty::Unit, false),
        ld(f32_ro.clone(), false), // 1 a [M*K]
        ld(f32_ro, false),         // 2 b [K*N]
        ld(f32_rw, true),          // 3 c [M*N]
        ld(Ty::Usize, false),      // 4 col (x)
        ld(Ty::Usize, false),      // 5 row (y)
        ld(Ty::Bool, false),       // 6 col_ok
        ld(Ty::Bool, false),       // 7 row_ok
        ld(Ty::F32, true),         // 8 acc
        ld(Ty::Usize, true),       // 9 kk
        ld(Ty::Bool, false),       // 10 k_lt
        ld(Ty::Usize, false),      // 11 rowK
        ld(Ty::Usize, false),      // 12 flat_a
        ld(Ty::Usize, false),      // 13 kN
        ld(Ty::Usize, false),      // 14 flat_b
        ld(Ty::F32, false),        // 15 va
        ld(Ty::F32, false),        // 16 vb
        ld(Ty::F32, false),        // 17 prod
        ld(Ty::F32, false),        // 18 acc_new
        ld(Ty::Usize, false),      // 19 k_new
        ld(Ty::Usize, false),      // 20 rowN
        ld(Ty::Usize, false),      // 21 flat_c
    ];
    let (a, b, c) = (local(1), local(2), local(3));
    let (col, row, col_ok, row_ok) = (local(4), local(5), local(6), local(7));
    let (acc, kk, k_lt) = (local(8), local(9), local(10));
    let (rowk, flat_a, kn, flat_b, va, vb, prod, acc_new, k_new, rown, flat_c) = (
        local(11),
        local(12),
        local(13),
        local(14),
        local(15),
        local(16),
        local(17),
        local(18),
        local(19),
        local(20),
        local(21),
    );
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cp = Place::local;
    let assign = |p, rv| Statement::Assign(p, rv);
    let bin = |op, x, y| Rvalue::BinaryOp(op, x, y);
    let sw = |discr, zero_target: u32, otherwise: u32| Terminator::SwitchInt {
        discr,
        targets: SwitchTargets {
            branches: vec![(0, BlockId { index: zero_target })],
            otherwise: BlockId { index: otherwise },
        },
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: cp(col),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: cp(row),
            dim: IndexAxis::Y,
            target: BlockId { index: 2 },
        },
    };
    let bb2 = BasicBlock {
        statements: vec![assign(cp(col_ok), bin(BinOp::Lt, copy(cp(col)), cu(n)))],
        terminator: sw(copy(cp(col_ok)), 8, 3),
    };
    let bb3 = BasicBlock {
        statements: vec![assign(cp(row_ok), bin(BinOp::Lt, copy(cp(row)), cu(m)))],
        terminator: sw(copy(cp(row_ok)), 8, 4),
    };
    let bb4 = BasicBlock {
        statements: vec![
            assign(cp(acc), Rvalue::Use(Operand::Const(Constant::F32(0.0)))),
            assign(cp(kk), Rvalue::Use(cu(0))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 5 },
        },
    };
    let bb5 = BasicBlock {
        statements: vec![assign(cp(k_lt), bin(BinOp::Lt, copy(cp(kk)), cu(k)))],
        terminator: sw(copy(cp(k_lt)), 7, 6),
    };
    let bb6 = BasicBlock {
        statements: vec![
            assign(cp(rowk), bin(BinOp::Mul, copy(cp(row)), cu(k))),
            assign(cp(flat_a), bin(BinOp::Add, copy(cp(rowk)), copy(cp(kk)))),
            assign(cp(kn), bin(BinOp::Mul, copy(cp(kk)), cu(n))),
            assign(cp(flat_b), bin(BinOp::Add, copy(cp(kn)), copy(cp(col)))),
            assign(cp(va), Rvalue::Use(copy(elem(a, flat_a)))),
            assign(cp(vb), Rvalue::Use(copy(elem(b, flat_b)))),
            assign(cp(prod), bin(BinOp::Mul, copy(cp(va)), copy(cp(vb)))),
            assign(cp(acc_new), bin(BinOp::Add, copy(cp(acc)), copy(cp(prod)))),
            assign(cp(acc), Rvalue::Use(copy(cp(acc_new)))),
            assign(cp(k_new), bin(BinOp::Add, copy(cp(kk)), cu(1))),
            assign(cp(kk), Rvalue::Use(copy(cp(k_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 5 },
        },
    };
    let bb7 = BasicBlock {
        statements: vec![
            assign(cp(rown), bin(BinOp::Mul, copy(cp(row)), cu(n))),
            assign(cp(flat_c), bin(BinOp::Add, copy(cp(rown)), copy(cp(col)))),
            assign(elem(c, flat_c), Rvalue::Use(copy(cp(acc)))),
        ],
        terminator: Terminator::Return,
    };
    let bb8 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    let mut body = Body::new(
        "matmul",
        3,
        locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8],
    );
    body.workgroup_size = [16, 16, 1];
    body
}

/// `out[i] = a[i]*b[i] - c[i]*d[i]` over four parallel f32 slices (card 628, SC-001/SC-002 - the design's
/// synthetic detector body, dquant.md 3.3's `a*b - c*d` shape). The final subtract (`m1 - m2`) is
/// `Rvalue::BinaryOpNoContract` when `marked`, plain `Rvalue::BinaryOp` otherwise - the SC-001 mutation is
/// exactly `marked: true` -> `marked: false`.
pub fn no_contract_probe_kernel(marked: bool) -> Body {
    let f32_ro = slice_f32(false);
    let locals = vec![
        ld(Ty::Unit, false),
        ld(f32_ro.clone(), false), // 1 a
        ld(f32_ro.clone(), false), // 2 b
        ld(f32_ro.clone(), false), // 3 c
        ld(f32_ro, false),         // 4 d
        ld(slice_f32(true), true), // 5 out
        ld(Ty::Usize, false),      // 6 i
        ld(Ty::Usize, false),      // 7 len
        ld(Ty::Bool, false),       // 8 cmp
        ld(Ty::F32, false),        // 9 m1 = a*b
        ld(Ty::F32, false),        // 10 m2 = c*d
        ld(Ty::F32, false),        // 11 r = m1 - m2
    ];
    let (a, b, c, d, out, i, len, cmp, m1, m2, r) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
        local(8),
        local(9),
        local(10),
        local(11),
    );

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
    let sub = if marked {
        Rvalue::BinaryOpNoContract(BinOp::Sub, copy(Place::local(m1)), copy(Place::local(m2)))
    } else {
        Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(m1)), copy(Place::local(m2)))
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(m1),
                Rvalue::BinaryOp(BinOp::Mul, copy(elem(a, i)), copy(elem(b, i))),
            ),
            Statement::Assign(
                Place::local(m2),
                Rvalue::BinaryOp(BinOp::Mul, copy(elem(c, i)), copy(elem(d, i))),
            ),
            Statement::Assign(Place::local(r), sub),
            Statement::Assign(elem(out, i), Rvalue::Use(copy(Place::local(r)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    Body::new("no_contract_probe", 5, locals, vec![bb0, bb1, bb2, bb3])
}

// --- poot-kernel-ir interp helpers (ex poot-kernel-ir/src/interp.rs) ----------------------------------------

/// The workgroup counts that cover `threads` global `X` indices of a body with a one-dimensional workgroup.
/// Lanes past `threads` run too, as on a device: the body's own guard must keep them out of bounds.
pub fn workgroups_covering(body: &Body, threads: usize) -> [u32; 3] {
    [
        threads.div_ceil(body.workgroup_size[0] as usize) as u32,
        1,
        1,
    ]
}

// --- poot-kernelgen generators with no production caller ----------------------------------------------------

/// f32 gather over axis 0 with an f32 index cast to usize internally. Thin wrapper over the still-pub,
/// still-production-used [`poot_kernelgen::gather_axis0_index_dt`] with `index_dt = Ty::F32`.
pub fn gather_axis0_dt(name: &str, dt: Ty, rest: usize) -> Body {
    poot_kernelgen::gather_axis0_index_dt(name, dt, Ty::F32, rest)
}

/// Workgroup-sum of the first `w` elements of `a` into `out[0]`, via LDS + a barrier. Each lane writes
/// `a[lane]` to `LDS[lane]`, a workgroup barrier, then lane 0 sums `LDS[0..w]`. Exercises the shared-memory
/// + barrier codegen. Launch one workgroup of `w` threads.
pub fn wg_sum(name: &str, w: usize) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 a
        ld(slice_f32(true), true),   // 2 out
        ld(Ty::Usize, false),        // 3 lane
        ld(Ty::F32, false),          // 4 v
        ld(Ty::Bool, false),         // 5 cmp
        ld(Ty::F32, true),           // 6 acc
        ld(Ty::Usize, true),         // 7 j
        ld(Ty::Bool, false),         // 8 j_lt
        ld(Ty::F32, false),          // 9 lds_v
        ld(Ty::F32, false),          // 10 acc_new
        ld(Ty::Usize, false),        // 11 j_new
    ];
    let (a, out) = (local(1), local(2));
    let lane = local(3);
    let (v, cmp, acc, j, j_lt, lds_v, acc_new, j_new) = (
        local(4),
        local(5),
        local(6),
        local(7),
        local(8),
        local(9),
        local(10),
        local(11),
    );
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(v), Rvalue::Use(copy(elem(a, lane)))),
            Statement::WorkgroupLocalWrite {
                idx: copy(Place::local(lane)),
                value: copy(Place::local(v)),
                array: 0,
            },
        ],
        terminator: Terminator::Barrier {
            target: BlockId { index: 2 },
        },
    };
    let bb2 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(lane)), cu(0)),
        )],
        terminator: guard(cmp, 7, 3),
    };
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(acc),
                Rvalue::Use(Operand::Const(Constant::F32(0.0))),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(cu(0))),
        ],
        terminator: goto(4),
    };
    let bb4 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(j_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(w)),
        )],
        terminator: guard(j_lt, 6, 5),
    };
    let bb5 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(lds_v),
                Rvalue::WorkgroupLocalRead {
                    idx: copy(Place::local(j)),
                    array: 0,
                },
            ),
            Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(lds_v)),
                ),
            ),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(j_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
        ],
        terminator: goto(4),
    };
    let bb6 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, lane),
            Rvalue::Use(copy(Place::local(acc))),
        )],
        terminator: Terminator::Return,
    };
    let bb7 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    let mut body = Body::new(
        name,
        2,
        locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7],
    );
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: w as u32,
    }];
    body
}

/// Serial last-axis reduction, f32 storage. Thin wrapper over the still-pub, still-production-used
/// [`poot_kernelgen::reduce_last_dt`] with `dt = Ty::F32`.
pub fn reduce_last(name: &str, op: BinOp, cols: usize, init: f32) -> Body {
    poot_kernelgen::reduce_last_dt(name, Ty::F32, op, cols, init)
}

/// numpy-broadcasting elementwise binary `c = a op b`, f32 storage. Thin wrapper over
/// [`binary_broadcast_dt`] with `dt = Ty::F32`.
pub fn binary_broadcast(
    name: &str,
    op: BinOp,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Body {
    binary_broadcast_dt(name, op, Ty::F32, out_shape, a_shape, b_shape)
}

/// [`binary_broadcast`] with operands/output in storage dtype `dt`. Thin wrapper over
/// [`binary_broadcast_dt_views`] with contiguous layouts.
pub fn binary_broadcast_dt(
    name: &str,
    op: BinOp,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Body {
    binary_broadcast_dt_views(
        name,
        op,
        dt,
        out_shape,
        a_shape,
        &poot_kernelgen::Layout::contiguous(a_shape),
        b_shape,
        &poot_kernelgen::Layout::contiguous(b_shape),
    )
}

/// [`binary_broadcast_dt`] generalized to read `a`/`b` through an arbitrary physical [`poot_kernelgen::Layout`].
/// Thin wrapper over the still-pub, still-production-used
/// [`poot_kernelgen::binary_broadcast_dt_views_grid`] with `x_groups = None`.
#[allow(clippy::too_many_arguments)]
pub fn binary_broadcast_dt_views(
    name: &str,
    op: BinOp,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    a_layout: &poot_kernelgen::Layout,
    b_shape: &[usize],
    b_layout: &poot_kernelgen::Layout,
) -> Body {
    poot_kernelgen::binary_broadcast_dt_views_grid(
        name, op, dt, out_shape, a_shape, a_layout, b_shape, b_layout, None,
    )
}

/// An exact-I32 fused elementwise body over strided-view leaves. Thin wrapper over the still-pub,
/// still-production-used [`poot_kernelgen::fused_i32_views_grid`] with `x_groups = None`.
pub fn fused_i32_views(
    name: &str,
    out_shape: &[usize],
    leaf_shapes: &[&[usize]],
    leaf_layouts: &[poot_kernelgen::Layout],
    k: &poot_kernelgen::FusedKernel,
) -> Result<Body, poot_kernelgen::KernelGenError> {
    poot_kernelgen::fused_i32_views_grid(name, out_shape, leaf_shapes, leaf_layouts, k, None)
}

/// A workgroup-scope producer/consumer fence handoff (`Statement::Fence{Workgroup, ..}`), locally verifiable
/// on RADV. Thread 0 (producer): LDS `data[0] = 42`, `Fence{Workgroup, Release}`, bumps an LDS flag. Every
/// other thread (consumer): spin-reads the flag until it observes it set, `Fence{Workgroup, Acquire}`, then
/// reads `data[0]` into `out[tid]`. Dispatched as exactly one workgroup.
pub fn workgroup_fence_spin_wait(name: &str, threads: u32) -> Body {
    use poot_kernel_ir::{AtomicOp, MemoryOrdering, MemoryScope};

    let locals = vec![
        ld(Ty::Unit, false),       // 0 ret
        ld(slice_f32(true), true), // 1 out
        ld(Ty::Usize, false),      // 2 tid
        ld(Ty::Usize, false),      // 3 zero
        ld(Ty::Bool, false),       // 4 is_producer (tid==0)
        ld(Ty::F32, false),        // 5 fval (atomic read / producer's old)
        ld(Ty::Bool, false),       // 6 not_ready (fval < 1.0)
        ld(Ty::F32, false),        // 7 dval
    ];
    let (out, tid, zero, isp, fval, notready, dval) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
    );
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let assign = |d: Local, rv: Rvalue| Statement::Assign(Place::local(d), rv);

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(tid),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            assign(zero, Rvalue::Use(Operand::Const(Constant::Usize(0)))),
            assign(
                isp,
                Rvalue::BinaryOp(
                    BinOp::Eq,
                    copy(Place::local(tid)),
                    Operand::Const(Constant::Usize(0)),
                ),
            ),
        ],
        terminator: guard(isp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::WorkgroupLocalWrite {
                idx: Operand::Const(Constant::Usize(0)),
                value: cf(42.0),
                array: 0,
            },
            Statement::Fence {
                scope: MemoryScope::Workgroup,
                ordering: MemoryOrdering::Release,
            },
            assign(
                fval,
                Rvalue::WorkgroupLocalAtomic {
                    idx: Operand::Const(Constant::Usize(0)),
                    value: cf(1.0),
                    op: AtomicOp::Add,
                    array: 1,
                },
            ),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![
            assign(
                fval,
                Rvalue::WorkgroupLocalAtomic {
                    idx: Operand::Const(Constant::Usize(0)),
                    value: cf(0.0),
                    op: AtomicOp::Add,
                    array: 1,
                },
            ),
            assign(
                notready,
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(fval)), cf(1.0)),
            ),
        ],
        terminator: guard(notready, 4, 3),
    };
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Fence {
                scope: MemoryScope::Workgroup,
                ordering: MemoryOrdering::Acquire,
            },
            assign(
                dval,
                Rvalue::WorkgroupLocalRead {
                    idx: Operand::Const(Constant::Usize(0)),
                    array: 0,
                },
            ),
            Statement::Assign(elem(out, tid), Rvalue::Use(copy(Place::local(dval)))),
        ],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 1, locals, vec![bb0, bb1, bb2, bb3, bb4]);
    body.workgroup_size = [threads, 1, 1];
    body.workgroup_locals = vec![
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: 1,
        }, // array 0: data
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: 1,
        }, // array 1: flag
    ];
    body
}

/// Single tensor-core tile: `C[16x16] = A[16x16] @ B[16x16]`, F16 inputs / f32 accumulate, via one warp's (or
/// subgroup's) WMMA/cooperative-matrix `m16n16k16` load/mma/store. A/B are row-major F16 slices (stride 16);
/// C a row-major f32 slice. Launched with one warp/subgroup (workgroup_size [32,1,1]).
pub fn wmma_tile(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::F16, false), false), // 1 a (f16, row-major 16x16)
        ld(slice_dtype(Ty::F16, false), false), // 2 b
        ld(slice_f32(true), true),              // 3 c (f32 out)
        ld(Ty::Usize, false),                   // 4 idx0
        ld(Ty::F16, false),                     // 5 af (fragment handle, A)
        ld(Ty::F16, false),                     // 6 bf (fragment handle, B)
        ld(Ty::F32, false),                     // 7 acc (fragment handle, accumulator)
    ];
    let (a, b, c) = (local(1), local(2), local(3));
    let (idx0, af, bf, acc) = (local(4), local(5), local(6), local(7));
    let tile = |s: Local| elem(s, idx0);

    let stmts = vec![
        Statement::Assign(
            Place::local(idx0),
            Rvalue::Use(Operand::Const(Constant::Usize(0))),
        ),
        Statement::WmmaZero {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            dst: acc,
        },
        Statement::WmmaLoad {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            which: WmmaMat::A,
            tile: tile(a),
            stride: 16,
            dst: af,
        },
        Statement::WmmaLoad {
            dtype: WmmaDtype::F16,
            shape: WmmaShape::M16N16K16,
            which: WmmaMat::B,
            tile: tile(b),
            stride: 16,
            dst: bf,
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
            tile: tile(c),
            stride: 16,
            src: acc,
        },
    ];
    let bb0 = BasicBlock {
        statements: stmts,
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 3, locals, vec![bb0]);
    body.workgroup_size = [32, 1, 1];
    body
}

// --- poot-codegen golden-IR helper (ex poot-codegen/src/emit.rs's public wrapper) ---------------------------

/// `Body` -> target-shaped textual LLVM IR, discarding the `NoContraction` ordinals
/// [`poot_codegen::emit_llvm_ir_marked`] returns alongside it: every caller here only inspects the IR text
/// (golden-IR comparisons, `llc` round-trips), never the ordinals `compile_with` uses in production.
pub fn emit_llvm_ir(
    body: &Body,
    target: poot_codegen::Target,
) -> Result<String, poot_codegen::EmitError> {
    Ok(poot_codegen::emit_llvm_ir_marked(body, target)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Card 671: moved from `poot-kernel-ir`'s own `well_formed_bodies_verify` and
    /// `no_contract_probe_kernel_verifies_marked_and_unmarked` - every fixture Body this module builds
    /// must type-check under the reference verifier that owns the invariant these fixtures exist to
    /// exercise in the first place.
    #[test]
    fn fixture_bodies_verify() {
        for body in [
            e4m3fn_encode_kernel(),
            e4m3fn_decode_kernel(),
            vadd_loop_kernel(),
            gemv_loop_kernel(4, 8),
            square_kernel(),
            matmul_kernel(2, 3, 4),
            no_contract_probe_kernel(true),
            no_contract_probe_kernel(false),
            len_probe_kernel(),
        ] {
            body.verify()
                .unwrap_or_else(|e| panic!("fixture {} rejected: {e}", body.name));
        }
    }
}
