use crate::KernelGenError;
use crate::helpers::{
    Alloc, WeightLane, WeightLayout, copy, elem, emit_lds_tree_blocks, guard, ld,
    lds_tree_block_count, local, slice_f32,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Operand, Place, Rvalue, Statement,
    Terminator, Ty, WorkgroupLocalDecl,
};
use poot_target::BufferStorage;

/// Decode GEMV via an LDS reduction: a shared-weight M=1 matmul `out[b,n] = sum_k a[b,k] * W[k*N + n]`,
/// single-sequence (`a [1,1,K]`, out `[1,1,N]`) or batched (`a [B,1,K]`, out `[B,1,N]`; `W [K,N]` is shared
/// across the batch). One workgroup of `w` lanes per output element `col = GroupX` in `0..B*N`
/// (`b_row = col/N`, `n_col = col%N`): each lane sums a strided subset of the K products, writes its partial to
/// LDS, a barrier, then a log2-tree of `w` lanes combines the partials (see [`emit_lds_tree_blocks`]) before
/// lane 0 writes `out[col]`. Same shape as `reduce_last_lds` with a multiply-in. Compared with the naive
/// one-thread-per-output matmul this raises parallelism to `B*N*w` and shortens each loop to `K/w`. Launch
/// `B*N*w` threads (see `dispatch_grid`). f32 only (the wgpu decode dtype). The reduction order differs from the
/// naive serial sum by fp rounding, within the executor-equivalence tolerance.
///
/// `x_groups` is the number of workgroups the launch places along the X grid dim; the flat output column is
/// `col = GroupY * x_groups + GroupX + elem_offset`, which lets a large output (the ~150k-vocab `lm_head`)
/// spill onto the Y grid dim and clear the wgpu 65535-per-dim cap. `dispatch_grid` launches the matching 2D
/// grid; a `col < len(out)` guard drops the over-dispatched tail of the last Y row. For an output that fits on
/// X alone, `x_groups == B*N` and `GroupY == 0`.
///
/// `layout` is how `W` is stored: [`WeightLayout::Kn`] reads `W[k*N + n]` (the matmul weight), [`WeightLayout::Nk`]
/// reads `W[n*K + k]` (the checkpoint orientation, so adjacent lanes read adjacent weight elements and no
/// `transpose` of the weight runs).
///
/// `weight` is the weight buffer's planned storage (Card 1007): `BufferStorage::f32()`, or
/// `BufferStorage::f16_packed()` for a checkpoint F16 weight decoded in-register from its packed words. Any other
/// storage is [`KernelGenError::Unsupported`].
///
/// `elem_offset` (card 258): the absolute output-element index this dispatch's `GroupX==0,GroupY==0`
/// workgroup starts at. Watchdog-safety chunking (`is_decode_gemv`'s plan, `poot-graph-plan`) splits one huge
/// decode GEMV (the wide-vocab tied `lm_head`, `N` up to ~250k) into several smaller dispatches, each covering a
/// disjoint `[elem_offset, elem_offset+chunk_elems)` slice of the same shared output buffer. `x_groups`
/// and the grid size are the chunk's own; only the final `col` takes the absolute offset, since
/// `n_col = col % N` and `tot = len(out)`
/// already generalize to any starting `col`. Zero when unchunked; the add is emitted only if `!= 0`, so
/// zero-offset callers get unchanged IR.
#[allow(clippy::too_many_arguments)] // one baked dimension or mode per argument, as in the sibling generators
pub fn gemv_lds(
    name: &str,
    k: usize,
    n: usize,
    w: usize,
    x_groups: usize,
    elem_offset: usize,
    layout: WeightLayout,
    weight: BufferStorage,
) -> Result<Body, KernelGenError> {
    let weight_lane = WeightLane::of("gemv_lds", weight)?;
    // params: _0 ret, _1 a, _2 b (weight), _3 out (LAST).
    let params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 a [B, K]
        weight_lane.param(),         // 2 b (weight) [K, N] or [N, K]
        ld(slice_f32(true), true),   // 3 out [B, N]
    ];
    let mut al = Alloc::new(params);
    let (a, b) = (local(1), local(2));
    let out = local(3);
    let col = al.add(Ty::Usize, false);
    let gx = al.add(Ty::Usize, false); // GroupX
    let gy = al.add(Ty::Usize, false); // GroupY (>0 only when the output spills past the X grid cap)
    let tot = al.add(Ty::Usize, false); // len(out) = B*N (the tail guard)
    let inb = al.add(Ty::Bool, false); // col < tot
    let lane = al.add(Ty::Usize, false);
    let n_col = al.add(Ty::Usize, false); // col % N  (the output column within a row)
    let b_off = al.add(Ty::Usize, false); // (col / N) * K  (the activation row's base, batched)
    let aidx = al.add(Ty::Usize, false);
    let partial = al.add(Ty::F32, true);
    let j = al.add(Ty::Usize, true);
    let j_lt = al.add(Ty::Bool, false);
    let bidx = al.add(Ty::Usize, false);
    let av = al.add(Ty::F32, false);
    let bv = al.add(Ty::F32, false);
    let prod = al.add(Ty::F32, false);
    let part_new = al.add(Ty::F32, false);
    let j_new = al.add(Ty::Usize, false);
    let cmp0 = al.add(Ty::Bool, false); // lane == 0 (final store guard, after the log2 tree)
    let acc = al.add(Ty::F32, false); // final reduced sum, read from LDS[0] once the tree completes
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |bb: u32| Terminator::Goto {
        target: BlockId { index: bb },
    };

    // Block layout:
    // 0..5: gx placeholder-target below, lane, K-accumulation loop, LDS write + barrier.
    // 6..round_start+3*rounds: the unrolled tree rounds (emit_lds_tree_blocks), 3 blocks/round.
    // final_branch: lane==0 guard -> store (lane 0) / early-return (other lanes).
    // final_branch+3/+4/+5: the 2-D grid GroupY-read / col-compute+tail-check / tail-return tail, placed last
    // so their indices don't shift as the tree's block count varies with `w`.
    let round_start: u32 = 6;
    let final_branch: u32 = round_start + lds_tree_block_count(w);
    let store_idx = final_branch + 1;
    let early_return_idx = final_branch + 2;
    let gy_idx = final_branch + 3;
    let colcalc_idx = final_branch + 4;
    let tail_return_idx = final_branch + 5;

    // bb0: gx = GroupX, then gy_idx reads GroupY and colcalc_idx forms the flat col (2-D grid for the large
    // lm_head).
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: gy_idx },
        },
    };
    // bb1: lane = LocalX
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };
    // bb2: n_col = col % N; b_off = (col / N) * K (batched: col indexes B*N outputs row-major); partial=0; j=lane
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(n_col),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(col)), cu(n)),
            ),
            Statement::Assign(
                Place::local(b_off),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(col)), cu(n)),
            ),
            Statement::Assign(
                Place::local(b_off),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(b_off)), cu(k)),
            ),
            Statement::Assign(Place::local(partial), Rvalue::Use(cf(0.0))),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(lane)))),
        ],
        terminator: goto(3),
    };
    // bb3 (acc hdr): j_lt = j < K; if j>=K -> LDS write(5), else body(4)
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(j_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(k)),
        )],
        terminator: guard(j_lt, 5, 4),
    };
    // bb4 (acc body): partial += a[b_off + j] * b[idx(j, n_col)]; j += w
    let read_bv = weight_lane.read(&mut al, b, bidx, bv);
    let bb4 = BasicBlock {
        statements: [
            Statement::Assign(
                Place::local(aidx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(b_off)), copy(Place::local(j))),
            ),
            Statement::Assign(Place::local(av), Rvalue::Use(copy(elem(a, aidx)))),
            // Kn: bidx = j * N + n_col. Nk: bidx = n_col * K + j.
            Statement::Assign(
                Place::local(bidx),
                match layout {
                    WeightLayout::Kn => Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(j)), cu(n)),
                    WeightLayout::Nk => {
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(n_col)), cu(k))
                    }
                },
            ),
            Statement::Assign(
                Place::local(bidx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(bidx)),
                    copy(Place::local(match layout {
                        WeightLayout::Kn => n_col,
                        WeightLayout::Nk => j,
                    })),
                ),
            ),
        ]
        .into_iter()
        .chain(read_bv)
        .chain([
            Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(av)), copy(Place::local(bv))),
            ),
            Statement::Assign(
                Place::local(part_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(partial)),
                    copy(Place::local(prod)),
                ),
            ),
            Statement::Assign(
                Place::local(partial),
                Rvalue::Use(copy(Place::local(part_new))),
            ),
            Statement::Assign(
                Place::local(j_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(w)),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
        ])
        .collect(),
        terminator: goto(3),
    };
    // bb5: LDS[lane] = partial; barrier -> the first tree round's guard (round_start), or straight to
    // final_branch if `w == 1` (no rounds - round_start == final_branch in that degenerate case).
    let bb5 = BasicBlock {
        statements: vec![Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(lane)),
            value: copy(Place::local(partial)),
            array: 0,
        }],
        terminator: Terminator::Barrier {
            target: BlockId { index: round_start },
        },
    };
    let mut blocks = vec![bb0, bb1, bb2, bb3, bb4, bb5];
    emit_lds_tree_blocks(&mut al, &mut blocks, lane, w, round_start, final_branch);
    // final_branch: if lane != 0 -> early return, else store (lane 0 only; LDS[0] now holds the full sum).
    blocks.push(BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp0),
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(lane)), cu(0)),
        )],
        terminator: guard(cmp0, early_return_idx, store_idx),
    });
    // store_idx: acc = LDS[0]; out[col] = acc; return
    blocks.push(BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(acc),
                Rvalue::WorkgroupLocalRead {
                    idx: cu(0),
                    array: 0,
                },
            ),
            Statement::Assign(elem(out, col), Rvalue::Use(copy(Place::local(acc)))),
        ],
        terminator: Terminator::Return,
    });
    // early_return_idx: non-lane-0 threads return without writing.
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    // gy_idx: gy = GroupY
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: colcalc_idx },
        },
    });
    // colcalc_idx: col = gy*x_groups + gx [+ elem_offset]; tot = len(out); if col >= tot -> tail return,
    // else continue(bb1). `elem_offset` is only emitted when non-zero.
    let mut colcalc_stmts = vec![
        Statement::Assign(
            Place::local(col),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(x_groups)),
        ),
        Statement::Assign(
            Place::local(col),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(col)), copy(Place::local(gx))),
        ),
    ];
    if elem_offset != 0 {
        colcalc_stmts.push(Statement::Assign(
            Place::local(col),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(col)), cu(elem_offset)),
        ));
    }
    colcalc_stmts.push(Statement::Assign(
        Place::local(tot),
        Rvalue::Len(Place::local(out)),
    ));
    colcalc_stmts.push(Statement::Assign(
        Place::local(inb),
        Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(col)), copy(Place::local(tot))),
    ));
    blocks.push(BasicBlock {
        statements: colcalc_stmts,
        terminator: guard(inb, tail_return_idx, 1),
    });
    // tail_return_idx: over-dispatched tail (col past the output) -> return without writing
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    let mut body = Body::new(
        name, 3, // data params: a, b, out
        al.locals, blocks,
    );
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: w as u32,
    }];
    Ok(body)
}

/// The decode attention `scores @ V` matmul via an LDS reduction: a batched, per-row matmul
/// `out[row, d] = sum_t probs[row, t] * v[row, t, d]` where `row` ranges over the flattened batch/head dims
/// (`B*Hq`; `probs [1,Hq,1,cap]`, `v [1,Hq,cap,D]` -> `out [1,Hq,1,D]`). GQA is already expanded to `Hq` by
/// `repeat_kv` upstream (`poot-graph-ir::ops::repeat_kv` materializes a Broadcast copy), so no head-group math
/// is needed here. Unlike [`gemv_lds`] the second operand is not shared across rows: row `row`'s `D`-wide slab
/// is at `v[row*cap*D ..]`, so each row reads its own `probs` slice and its own `v` slab.
///
/// Same design as `gemv_lds`: one workgroup of `w` lanes per output element `col` in `0..rows*D`
/// (`row = col/D`, `d_col = col%D`); each lane sums a strided subset of the `cap` products, writes its partial
/// to LDS, a barrier, then a log2-tree combines the `w` partials and lane 0 writes `out[col]`. `scores @ V` is a
/// large reduction (`cap`, the growing KV length) into a small output (`Hq*D`), which the naive
/// one-thread-per-output kernel (`matmul_batched_dt`) handles poorly. `Q @ K^T` has the opposite shape (small
/// reduction `D`, large output `Hq*cap`) and is deliberately not routed here; `poot-graph-plan::is_decode_attn_gemv`
/// tells them apart via the Transpose-producer check (a size heuristic on `cap` vs `D` is unreliable at small
/// `cap`, e.g. the first decode step after a 1-token prompt).
///
/// The `x_groups`/2-D grid spill uses the same convention as `gemv_lds` (`col = GroupY*x_groups + GroupX`) so
/// both share one grid-forming convention, though `rows*D` never nears the 65535-per-dim cap in practice.
/// f32 only (the wgpu decode dtype); the reduction order differs from the naive serial sum by fp rounding,
/// within the executor-equivalence tolerance.
pub fn attn_scores_v_gemv_lds(name: &str, cap: usize, d: usize, w: usize, x_groups: usize) -> Body {
    // params: _0 ret, _1 probs [rows, cap], _2 v [rows, cap, d], out [rows, d] (LAST).
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 probs [rows, cap]
        ld(slice_f32(false), false), // 2 v [rows, cap, d]
        ld(slice_f32(true), true),   // out [rows, d]
    ]);
    let (probs, v) = (local(1), local(2));
    let out = local(3);
    let col = al.add(Ty::Usize, false);
    let gx = al.add(Ty::Usize, false); // GroupX
    let gy = al.add(Ty::Usize, false); // GroupY (>0 only if rows*d overflows the X grid cap)
    let tot = al.add(Ty::Usize, false); // len(out) = rows*d (the tail guard)
    let inb = al.add(Ty::Bool, false); // col < tot
    let lane = al.add(Ty::Usize, false);
    let d_col = al.add(Ty::Usize, false); // col % d (the output column within a row)
    let row = al.add(Ty::Usize, false); // col / d
    let a_base = al.add(Ty::Usize, false); // row * cap  (probs's row base)
    let b_base = al.add(Ty::Usize, false); // row * cap * d  (v's row-slab base)
    let aidx = al.add(Ty::Usize, false);
    let partial = al.add(Ty::F32, true);
    let j = al.add(Ty::Usize, true);
    let j_lt = al.add(Ty::Bool, false);
    let bidx = al.add(Ty::Usize, false);
    let bidx2 = al.add(Ty::Usize, false);
    let av = al.add(Ty::F32, false);
    let bv = al.add(Ty::F32, false);
    let prod = al.add(Ty::F32, false);
    let part_new = al.add(Ty::F32, false);
    let j_new = al.add(Ty::Usize, false);
    let cmp0 = al.add(Ty::Bool, false); // lane == 0 (final store guard, after the log2 tree)
    let acc = al.add(Ty::F32, false); // final reduced sum, read from LDS[0] once the tree completes
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |bb: u32| Terminator::Goto {
        target: BlockId { index: bb },
    };

    // Block layout: see the matching comment in gemv_lds.
    let round_start: u32 = 6;
    let final_branch: u32 = round_start + lds_tree_block_count(w);
    let store_idx = final_branch + 1;
    let early_return_idx = final_branch + 2;
    let gy_idx = final_branch + 3;
    let colcalc_idx = final_branch + 4;
    let tail_return_idx = final_branch + 5;

    // bb0: gx = GroupX, then gy_idx reads GroupY and colcalc_idx forms the flat col (2-D grid, gemv_lds
    // convention).
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: gy_idx },
        },
    };
    // bb1: lane = LocalX
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };
    // bb2: d_col = col % d; row = col / d; a_base = row*cap; b_base = row*cap*d; partial=0; j=lane
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(d_col),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(col)), cu(d)),
            ),
            Statement::Assign(
                Place::local(row),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(col)), cu(d)),
            ),
            Statement::Assign(
                Place::local(a_base),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(cap)),
            ),
            Statement::Assign(
                Place::local(b_base),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(cap * d)),
            ),
            Statement::Assign(Place::local(partial), Rvalue::Use(cf(0.0))),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(lane)))),
        ],
        terminator: goto(3),
    };
    // bb3 (acc hdr): j_lt = j < cap; if j>=cap -> LDS write(5), else body(4)
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(j_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(cap)),
        )],
        terminator: guard(j_lt, 5, 4),
    };
    // bb4 (acc body): partial += probs[a_base+j] * v[b_base + j*d + d_col]; j += w
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(aidx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(a_base)),
                    copy(Place::local(j)),
                ),
            ),
            Statement::Assign(Place::local(av), Rvalue::Use(copy(elem(probs, aidx)))),
            Statement::Assign(
                Place::local(bidx),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(j)), cu(d)),
            ),
            Statement::Assign(
                Place::local(bidx),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(b_base)),
                    copy(Place::local(bidx)),
                ),
            ),
            Statement::Assign(
                Place::local(bidx2),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(bidx)),
                    copy(Place::local(d_col)),
                ),
            ),
            Statement::Assign(Place::local(bv), Rvalue::Use(copy(elem(v, bidx2)))),
            Statement::Assign(
                Place::local(prod),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(av)), copy(Place::local(bv))),
            ),
            Statement::Assign(
                Place::local(part_new),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(partial)),
                    copy(Place::local(prod)),
                ),
            ),
            Statement::Assign(
                Place::local(partial),
                Rvalue::Use(copy(Place::local(part_new))),
            ),
            Statement::Assign(
                Place::local(j_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(w)),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
        ],
        terminator: goto(3),
    };
    // bb5: LDS[lane] = partial; barrier -> the first tree round's guard (round_start).
    let bb5 = BasicBlock {
        statements: vec![Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(lane)),
            value: copy(Place::local(partial)),
            array: 0,
        }],
        terminator: Terminator::Barrier {
            target: BlockId { index: round_start },
        },
    };
    let mut blocks = vec![bb0, bb1, bb2, bb3, bb4, bb5];
    emit_lds_tree_blocks(&mut al, &mut blocks, lane, w, round_start, final_branch);
    // final_branch: if lane != 0 -> early return, else store (lane 0 only; LDS[0] now holds the full sum).
    blocks.push(BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp0),
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(lane)), cu(0)),
        )],
        terminator: guard(cmp0, early_return_idx, store_idx),
    });
    // store_idx: acc = LDS[0]; out[col] = acc; return
    blocks.push(BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(acc),
                Rvalue::WorkgroupLocalRead {
                    idx: cu(0),
                    array: 0,
                },
            ),
            Statement::Assign(elem(out, col), Rvalue::Use(copy(Place::local(acc)))),
        ],
        terminator: Terminator::Return,
    });
    // early_return_idx: non-lane-0 threads return without writing.
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    // gy_idx: gy = GroupY
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: colcalc_idx },
        },
    });
    // colcalc_idx: col = gy*x_groups + gx; tot = len(out); if col >= tot -> tail return, else continue(bb1)
    blocks.push(BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(col),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(x_groups)),
            ),
            Statement::Assign(
                Place::local(col),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(col)), copy(Place::local(gx))),
            ),
            Statement::Assign(Place::local(tot), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(inb),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(col)), copy(Place::local(tot))),
            ),
        ],
        terminator: guard(inb, tail_return_idx, 1),
    });
    // tail_return_idx: over-dispatched tail (col past the output) -> return without writing
    blocks.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });
    let mut body = Body::new(
        name, 3, // data params: probs, v, out
        al.locals, blocks,
    );
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: w as u32,
    }];
    body
}
