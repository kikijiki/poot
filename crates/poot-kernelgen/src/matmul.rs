use crate::KernelGenError;
use crate::elementwise::ELEMENTWISE_2D_WORKGROUP_SIZE;
use crate::helpers::{
    Alloc, WeightLane, batch_eff, copy, elem, guard, ld, local, row_major_strides, slice_dtype,
    slice_f32,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty,
};
use poot_target::BufferStorage;

/// Batched matmul `C[..,M,N] = A[..,M,K] @ B[..,K,N]` (row-major), batch dims broadcast. One thread per
/// output element; a serial K-loop accumulates. Dims (M,N,K) and all strides are baked from the static
/// shapes. Covers decode gemv (B 2-D, batch \[1\]) and batched attention (\[1,H,..\]).
///
/// Batched matmul with operands/output in storage dtype `dt` (f32 or bf16) and f32 accumulate. bf16 operands
/// widen to f32 on load, the dot product accumulates in an f32 register, and the result narrows back to `dt`
/// on store. For `dt = Ty::F32` the casts are no-ops, so the kernel is byte-identical to the f32 matmul (same
/// cache key). This is the naive serial one-thread-per-output kernel `MatMul` falls back to when the tiled
/// paths (`is_tiled_gemm`/`is_batched_tiled_gemm_in`) decline, e.g. a batched attention-score `Q @ K^T` at
/// long ISL: `is_batched_tiled_gemm_in` rejects it once the tile count would cross the device's
/// known-miscompile ceiling (`caps.known_miscompiles.tiled_gemm_max_workgroups`, 2^15 on the RADV default,
/// card 095), a much lower bar than the `65535*256` wg-bump ceiling on output elements. Example:
/// Qwen2.5-0.5B batched prefill at ISL=2048 computes `q[1,14,2048,64] @ kt[1,14,64,2048] -> [1,14,2048,2048]`
/// (58,720,256 elements), `GridCap([229376,1,1])`. `x_groups: None` is the plain (non-folded) grid.
pub fn matmul_batched_dt_grid(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
    x_groups: Option<usize>,
) -> Body {
    matmul_batched_impl(
        name,
        dt,
        out_shape,
        a_shape,
        b_shape,
        WeightRead::Kn,
        false,
        x_groups,
    )
}

/// Batched matmul with a fused bias epilogue: `C[..,m,n] = (A@B)[..,m,n] + bias[n]` in one kernel, `bias`
/// an `[N]` f32 vector (the 3rd input, before the output). Folds `linear`'s post-matmul bias add into the
/// matmul (one dispatch instead of two; the q/k/v projection bias adds). See [`matmul_batched_dt_grid`]'s
/// doc for the dtype/fallback rationale. `x_groups: None` is the plain (non-folded) grid.
pub fn matmul_batched_bias_dt_grid(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
    x_groups: Option<usize>,
) -> Body {
    matmul_batched_impl(
        name,
        dt,
        out_shape,
        a_shape,
        b_shape,
        WeightRead::Kn,
        true,
        x_groups,
    )
}

/// [`matmul_batched_dt_grid`] over a weight held in checkpoint `[N, K]` order (`w_shape` is `[N, K]`, rank 2):
/// `C[..,M,N] = A[..,M,K] @ W[N,K]^T`, without materializing the transpose. The one-thread-per-output
/// fallback of `DenseContraction` where the tiled GEMM is not taken. The activation and output are f32; the
/// weight is read in its planned `weight` storage (`BufferStorage::f32()` or, Card 1007,
/// `BufferStorage::f16_packed()`).
pub(crate) fn matmul_batched_nk_grid(
    name: &str,
    weight: BufferStorage,
    out_shape: &[usize],
    a_shape: &[usize],
    w_shape: &[usize],
    x_groups: Option<usize>,
) -> Result<Body, KernelGenError> {
    let lane = WeightLane::of("matmul_batched_nk_grid", weight)?;
    Ok(matmul_batched_impl(
        name,
        Ty::F32,
        out_shape,
        a_shape,
        w_shape,
        WeightRead::Nk(lane),
        false,
        x_groups,
    ))
}

/// How [`matmul_batched_impl`] reads its weight operand.
#[derive(Clone, Copy)]
enum WeightRead {
    /// `[K, N]` (the `MatMul` weight) in the body's own storage dtype `dt`.
    Kn,
    /// `[N, K]` (the checkpoint orientation) through a weight lane; the body's `dt` is f32.
    Nk(WeightLane),
}

#[allow(clippy::too_many_arguments)] // shapes plus the bias/weight/fold modes the public entry points fix
fn matmul_batched_impl(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
    weight: WeightRead,
    bias: bool,
    x_groups: Option<usize>,
) -> Body {
    let r = out_shape.len();
    let k = a_shape[a_shape.len() - 1];
    let out_strides = row_major_strides(out_shape);
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);
    // a's M-stride (the dim before K) and b's N-stride (the last dim).
    let a_m_stride = a_strides[a_shape.len() - 2];
    // b's K-stride and N-stride: `[K, N]` has K first, the checkpoint `[N, K]` has N first.
    let (b_k_stride, b_n_stride) = match weight {
        WeightRead::Kn => (b_strides[b_shape.len() - 2], b_strides[b_shape.len() - 1]),
        WeightRead::Nk(_) => (b_strides[b_shape.len() - 1], b_strides[b_shape.len() - 2]),
    };
    // batch-effective strides aligned to out's batch dims [0..r-2].
    let a_beff = batch_eff(r - 2, a_shape, &a_strides);
    let b_beff = batch_eff(r - 2, b_shape, &b_strides);

    // params: _0 ret, _1 a, _2 b, [_3 bias (f32 [N], read-only)], c (output, LAST - inputs bind first).
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false),
        match weight {
            WeightRead::Kn => ld(slice_dtype(dt.clone(), false), false),
            WeightRead::Nk(lane) => lane.param(),
        },
    ];
    if bias {
        params.push(ld(slice_f32(false), false));
    }
    params.push(ld(slice_dtype(dt.clone(), true), true));
    let mut al = Alloc::new(params);
    let a = local(1);
    let b = local(2);
    let bias_l = if bias { Some(local(3)) } else { None };
    let c = local(if bias { 4 } else { 3 });
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let mrow = al.add(Ty::Usize, false);
    let ncol = al.add(Ty::Usize, false);
    let a_base = al.add(Ty::Usize, true);
    let b_base = al.add(Ty::Usize, true);
    let div = al.add(Ty::Usize, false);
    let md = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let off_new = al.add(Ty::Usize, false);
    let acc = al.add(Ty::F32, true);
    let kk = al.add(Ty::Usize, true);
    let k_lt = al.add(Ty::Bool, false);
    let pa = al.add(Ty::Usize, false);
    let kn = al.add(Ty::Usize, false);
    let pb = al.add(Ty::Usize, false);
    let va = al.add(Ty::F32, false);
    let vb = al.add(Ty::F32, false);
    let prod = al.add(Ty::F32, false);
    let acc_new = al.add(Ty::F32, false);
    let k_new = al.add(Ty::Usize, false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let coord = |dst: Local, d: usize, body: &mut Vec<Statement>| {
        // dst = (i / out_strides[d]) % out_shape[d]
        body.push(Statement::Assign(
            Place::local(div),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
        ));
        body.push(Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(out_shape[d])),
        ));
    };

    // bb2: compute m,n + batch offsets + a_base(row) + b_base(col) + acc=0,kk=0 -> loop
    let mut pre = vec![
        Statement::Assign(Place::local(a_base), Rvalue::Use(cu(0))),
        Statement::Assign(Place::local(b_base), Rvalue::Use(cu(0))),
    ];
    coord(mrow, r - 2, &mut pre);
    coord(ncol, r - 1, &mut pre);
    // accumulate batch offsets
    for d in 0..(r - 2) {
        if a_beff[d] != 0 {
            coord(md, d, &mut pre); // md = batch coord at dim d (reuses div/md temps)
            pre.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(a_beff[d])),
            ));
            pre.push(Statement::Assign(
                Place::local(off_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(a_base)),
                    copy(Place::local(term)),
                ),
            ));
            pre.push(Statement::Assign(
                Place::local(a_base),
                Rvalue::Use(copy(Place::local(off_new))),
            ));
        }
        if b_beff[d] != 0 {
            coord(md, d, &mut pre);
            pre.push(Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(b_beff[d])),
            ));
            pre.push(Statement::Assign(
                Place::local(off_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(b_base)),
                    copy(Place::local(term)),
                ),
            ));
            pre.push(Statement::Assign(
                Place::local(b_base),
                Rvalue::Use(copy(Place::local(off_new))),
            ));
        }
    }
    // a_row_base = a_base + mrow * a_m_stride  (reuse a_base in place)
    pre.push(Statement::Assign(
        Place::local(term),
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(mrow)), cu(a_m_stride)),
    ));
    pre.push(Statement::Assign(
        Place::local(off_new),
        Rvalue::BinaryOp(
            BinOp::Add,
            copy(Place::local(a_base)),
            copy(Place::local(term)),
        ),
    ));
    pre.push(Statement::Assign(
        Place::local(a_base),
        Rvalue::Use(copy(Place::local(off_new))),
    ));
    // b_col_base = b_base + ncol * b_n_stride
    pre.push(Statement::Assign(
        Place::local(term),
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(ncol)), cu(b_n_stride)),
    ));
    pre.push(Statement::Assign(
        Place::local(off_new),
        Rvalue::BinaryOp(
            BinOp::Add,
            copy(Place::local(b_base)),
            copy(Place::local(term)),
        ),
    ));
    pre.push(Statement::Assign(
        Place::local(b_base),
        Rvalue::Use(copy(Place::local(off_new))),
    ));
    pre.push(Statement::Assign(
        Place::local(acc),
        Rvalue::Use(Operand::Const(Constant::F32(0.0))),
    ));
    pre.push(Statement::Assign(Place::local(kk), Rvalue::Use(cu(0))));

    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 6, 2),
    };
    let bb2 = BasicBlock {
        statements: pre,
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(k_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(kk)), cu(k)),
        )],
        terminator: guard(k_lt, 5, 4),
    };
    // bb4: pa = a_base + kk*a_k_stride; kn = kk*b_k_stride; pb = b_base + kn; acc += a[pa]*b[pb]; kk++
    // The weight read: a native `dt` element, widened when `dt` is bf16 (a no-op for f32), or the checkpoint
    // weight's lane read.
    let read_vb = match weight {
        WeightRead::Kn => vec![Statement::Assign(
            Place::local(vb),
            if dt == Ty::F32 {
                Rvalue::Use(copy(elem(b, pb)))
            } else {
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(b, pb)),
                }
            },
        )],
        WeightRead::Nk(lane) => lane.read(&mut al, b, pb, vb),
    };
    let bb4 = BasicBlock {
        statements: [
            // a's K-stride is 1 (K is A's last dim), so pa = a_base + kk.
            Statement::Assign(
                Place::local(pa),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(a_base)),
                    copy(Place::local(kk)),
                ),
            ),
            Statement::Assign(
                Place::local(kn),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(kk)), cu(b_k_stride)),
            ),
            Statement::Assign(
                Place::local(pb),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(b_base)),
                    copy(Place::local(kn)),
                ),
            ),
            // widen the (possibly bf16) operand to f32 for the f32-accumulate dot product; a no-op for f32.
            Statement::Assign(
                Place::local(va),
                if dt == Ty::F32 {
                    Rvalue::Use(copy(elem(a, pa)))
                } else {
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: copy(elem(a, pa)),
                    }
                },
            ),
        ]
        .into_iter()
        .chain(read_vb)
        .chain([
            Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(va)), copy(Place::local(vb))),
            ),
            Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(prod)),
                ),
            ),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(k_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(kk)), cu(1)),
            ),
            Statement::Assign(Place::local(kk), Rvalue::Use(copy(Place::local(k_new)))),
        ])
        .collect(),
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb5: store. For a bias matmul, fold `+ bias[ncol]` into the accumulator before the (optional) narrow.
    let biased = if bias_l.is_some() {
        Some(al.add(Ty::F32, false))
    } else {
        None
    };
    let mut bb5_stmts = Vec::new();
    let to_store = match (bias_l, biased) {
        (Some(bl), Some(biased)) => {
            bb5_stmts.push(Statement::Assign(
                Place::local(biased),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(acc)), copy(elem(bl, ncol))),
            ));
            biased
        }
        _ => acc,
    };
    bb5_stmts.push(Statement::Assign(
        elem(c, i),
        // narrow the f32 accumulator back to the storage dtype on store; a no-op for f32.
        if dt == Ty::F32 {
            Rvalue::Use(copy(Place::local(to_store)))
        } else {
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(to_store)),
            }
        },
    ));
    let bb5 = BasicBlock {
        statements: bb5_stmts,
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let n_params = if bias { 4 } else { 3 };
    match x_groups {
        None => {
            // the flat 1-D grid: `i = ThreadIndexCall(X)`
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(i),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            };
            Body::new(
                name,
                n_params,
                al.locals,
                vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6],
            )
        }
        Some(xg) => {
            // 2-D grid fold, same convention as `build_unary_dt_grid`/`binary_broadcast_dt_views_grid`: gx=GroupX ->
            // gy=GroupY -> group_id=gy*xg+gx, lane=LocalX -> i=group_id*WORKGROUP_SIZE+lane -> bb1 (the tail-length
            // guard). This kernel already uses blocks 0-6, so the fold blocks land at 7-9. The caller MUST bake
            // `body.workgroup_size[0] = ELEMENTWISE_2D_WORKGROUP_SIZE`.
            let wg = ELEMENTWISE_2D_WORKGROUP_SIZE;
            let gx = al.add(Ty::Usize, false);
            let gy = al.add(Ty::Usize, false);
            let group_id = al.add(Ty::Usize, false);
            let lane = al.add(Ty::Usize, false);
            let gy_idx: u32 = 7;
            let lane_idx: u32 = 8;
            let combine_idx: u32 = 9;
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
                n_params,
                al.locals,
                vec![
                    bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb_gy, bb_lane, bb_combine,
                ],
            )
        }
    }
}

/// Indexed GEMM for gather-free sparse MoE: `out[m, n] = sum_k x[m, k] * W[e_m, k, n]` where the per-row
/// expert `e_m = (usize) idx[m]` selects one `[K, N]` weight matrix from the stacked `W [E, K, N]` inside the
/// kernel, with no gather copy of the selected experts' weights first. The batched MoE case: `M` tokens, each
/// routed to its own expert; `M = 1` is the decode-GEMV case. One thread per output element `i = m*N + n`; the
/// dot reads `x`'s row `m` and the selected expert's column `n` directly. dt-generic: reads widen to f32, the
/// f32 accumulator narrows on store (a no-op for f32). Top-k is this summed over the k selected experts; this is
/// the top-1 primitive.
///
/// Params: `_1 x` \[M*K\] (dt), `_2 w` \[E*K*N\] flat row-major (dt), `_3 idx` \[M\] (f32 expert ids), `out` \[M*N\]
/// (dt).
pub fn indexed_matmul_dt(name: &str, dt: Ty, m_dim: usize, k: usize, n: usize) -> Body {
    let _ = m_dim; // M is implied by the output length; kept in the signature for call-site clarity.
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // x (dt) [M*K]
        ld(slice_dtype(dt.clone(), false), false), // w (dt) [E*K*N]
        ld(slice_f32(false), false),               // idx (f32) [M]
        ld(slice_dtype(dt.clone(), true), true),   // out (dt) [M*N]
    ]);
    let (x, w, idx, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, false); // the flat output index m*N + n
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let mrow = al.add(Ty::Usize, false); // i / N
    let ncol = al.add(Ty::Usize, false); // i % N
    let xbase = al.add(Ty::Usize, false); // mrow * K
    let idxf = al.add(Ty::F32, false);
    let expert = al.add(Ty::Usize, false);
    let ebase = al.add(Ty::Usize, false); // expert * (K*N)
    let acc = al.add(Ty::F32, true);
    let kk = al.add(Ty::Usize, true);
    let k_lt = al.add(Ty::Bool, false);
    let kn = al.add(Ty::Usize, false); // kk * N
    let woff0 = al.add(Ty::Usize, false); // ebase + kk*N
    let woff = al.add(Ty::Usize, false); // ebase + kk*N + ncol
    let xoff = al.add(Ty::Usize, false); // xbase + kk
    let vx = al.add(Ty::F32, false);
    let vw = al.add(Ty::F32, false);
    let prod = al.add(Ty::F32, false);
    let acc_new = al.add(Ty::F32, false);
    let k_new = al.add(Ty::Usize, false);
    let cu = |v: usize| Operand::Const(Constant::Usize(v as u64));
    let widen = |slot: Local, at: Local| {
        if dt == Ty::F32 {
            Rvalue::Use(copy(elem(slot, at)))
        } else {
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(elem(slot, at)),
            }
        }
    };

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
    // bb2: mrow = i/N; ncol = i%N; xbase = mrow*K; expert = (usize) idx[mrow]; ebase = expert*(K*N); acc=0; kk=0.
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(mrow),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(n)),
            ),
            Statement::Assign(
                Place::local(ncol),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(n)),
            ),
            Statement::Assign(
                Place::local(xbase),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(mrow)), cu(k)),
            ),
            Statement::Assign(Place::local(idxf), Rvalue::Use(copy(elem(idx, mrow)))),
            Statement::Assign(
                Place::local(expert),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(idxf)),
                },
            ),
            Statement::Assign(
                Place::local(ebase),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(expert)), cu(k * n)),
            ),
            Statement::Assign(
                Place::local(acc),
                Rvalue::Use(Operand::Const(Constant::F32(0.0))),
            ),
            Statement::Assign(Place::local(kk), Rvalue::Use(cu(0))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(k_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(kk)), cu(k)),
        )],
        terminator: guard(k_lt, 5, 4),
    };
    // bb4: woff = ebase + kk*N + ncol; xoff = xbase + kk; acc += x[xoff] * w[woff]; kk++.
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(kn),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(kk)), cu(n)),
            ),
            Statement::Assign(
                Place::local(woff0),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(ebase)),
                    copy(Place::local(kn)),
                ),
            ),
            Statement::Assign(
                Place::local(woff),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(woff0)),
                    copy(Place::local(ncol)),
                ),
            ),
            Statement::Assign(
                Place::local(xoff),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(xbase)),
                    copy(Place::local(kk)),
                ),
            ),
            Statement::Assign(Place::local(vx), widen(x, xoff)),
            Statement::Assign(Place::local(vw), widen(w, woff)),
            Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(vx)), copy(Place::local(vw))),
            ),
            Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(acc)),
                    copy(Place::local(prod)),
                ),
            ),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(k_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(kk)), cu(1)),
            ),
            Statement::Assign(Place::local(kk), Rvalue::Use(copy(Place::local(k_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb5: store out[i] = acc (narrow to dt on store; a no-op for f32).
    let bb5 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, i),
            if dt == Ty::F32 {
                Rvalue::Use(copy(Place::local(acc)))
            } else {
                Rvalue::Cast {
                    to: dt.clone(),
                    operand: copy(Place::local(acc)),
                }
            },
        )],
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6])
}
