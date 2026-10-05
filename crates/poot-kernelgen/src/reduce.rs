#[cfg(test)]
use crate::helpers::{Alloc, arr_elem, slice_f32};
use crate::helpers::{copy, elem, guard, ld, local, slice_dtype};
#[cfg(test)]
use poot_kernel_ir::Local;
#[cfg(test)]
use poot_kernel_ir::WorkgroupLocalDecl;
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Operand, Place, Rvalue, Statement,
    Terminator, Ty,
};

/// Parallel last-axis reduction via LDS: one workgroup of `w` lanes per row. Each lane reduces a strided
/// subset of the row into a register partial, writes it to LDS, a barrier, then lane 0 combines the `w`
/// partials and writes `out[row]`. The parallel-then-small-serial-combine shape generalizes a plain
/// workgroup-sum over a single row's first `w` elements. `op`/`init` select sum (Add, 0) or max (Max,
/// -inf). The reduction order differs from the serial [`reduce_last_dt`], so results are close to it, not
/// bit-identical. Launch `num_rows * w` threads with `wg = [w, 1, 1]`.
#[cfg(test)]
fn reduce_last_lds(name: &str, op: BinOp, cols: usize, init: f32, w: usize) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 in
        ld(slice_f32(true), true),   // 2 out
    ]);
    let (inp, out) = (local(1), local(2));
    let row = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let partial = al.add(Ty::F32, true);
    let j = al.add(Ty::Usize, true);
    let j_lt = al.add(Ty::Bool, false);
    let idx = al.add(Ty::Usize, true);
    let v = al.add(Ty::F32, false);
    let part_new = al.add(Ty::F32, false);
    let j_new = al.add(Ty::Usize, false);
    let cmp0 = al.add(Ty::Bool, false);
    let acc = al.add(Ty::F32, true);
    let k = al.add(Ty::Usize, true);
    let k_lt = al.add(Ty::Bool, false);
    let lv = al.add(Ty::F32, false);
    let acc_new = al.add(Ty::F32, false);
    let k_new = al.add(Ty::Usize, false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };
    let red = |dst: Local, a: Local, b: Local| {
        Statement::Assign(
            Place::local(dst),
            Rvalue::BinaryOp(op, copy(Place::local(a)), copy(Place::local(b))),
        )
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(row),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };
    // bb2: partial = init; j = lane -> bb3
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(partial), Rvalue::Use(cf(init))),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(lane)))),
        ],
        terminator: goto(3),
    };
    // bb3 (acc hdr): j_lt = j < cols; if j_lt==0 (j>=cols) -> write LDS(5), else body(4)
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(j_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(cols)),
        )],
        terminator: guard(j_lt, 5, 4),
    };
    // bb4 (acc body): idx = row*cols + j; partial op= in[idx]; j += w -> bb3
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(idx),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(cols)),
            ),
            Statement::Assign(
                Place::local(idx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(idx)), copy(Place::local(j))),
            ),
            Statement::Assign(Place::local(v), Rvalue::Use(copy(elem(inp, idx)))),
            red(part_new, partial, v),
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
    // bb5: LDS[lane] = partial; barrier -> bb6
    let bb5 = BasicBlock {
        statements: vec![Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(lane)),
            value: copy(Place::local(partial)),
            array: 0,
        }],
        terminator: Terminator::Barrier {
            target: BlockId { index: 6 },
        },
    };
    // bb6: cmp0 = (lane == 0); if cmp0==0 (lane != 0) -> ret(11), else combine(7)
    let bb6 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp0),
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(lane)), cu(0)),
        )],
        terminator: guard(cmp0, 11, 7),
    };
    // bb7: acc = init; k = 0 -> bb8
    let bb7 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(acc), Rvalue::Use(cf(init))),
            Statement::Assign(Place::local(k), Rvalue::Use(cu(0))),
        ],
        terminator: goto(8),
    };
    // bb8 (combine hdr): k_lt = k < w; if k_lt==0 -> store(10), else body(9)
    let bb8 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(k_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(k)), cu(w)),
        )],
        terminator: guard(k_lt, 10, 9),
    };
    // bb9 (combine body): acc op= LDS[k]; k += 1 -> bb8
    let bb9 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(lv),
                Rvalue::WorkgroupLocalRead {
                    idx: copy(Place::local(k)),
                    array: 0,
                },
            ),
            red(acc_new, acc, lv),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(k_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(k)), cu(1)),
            ),
            Statement::Assign(Place::local(k), Rvalue::Use(copy(Place::local(k_new)))),
        ],
        terminator: goto(8),
    };
    // bb10 (store): out[row] = acc; return
    let bb10 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, row),
            Rvalue::Use(copy(Place::local(acc))),
        )],
        terminator: Terminator::Return,
    };
    let bb11 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(
        name,
        2,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11],
    );
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: w as u32,
    }];
    body
}

/// Probe for the private fixed-size array (the flash-attention foundation): each thread fills
/// a per-thread `arr[n]` from its row of `in` (doubling), then reads it back to sum into `out[t]`. Both the
/// store and the load are dynamically indexed - the exact pattern that lowers on NVPTX (`alloca [n x f32]`)
/// but crashes the LLVM SPIR-V backend, so `compile(.., SpirvVulkan)` rejects it.
#[cfg(test)]
fn array_sum_probe(name: &str, n: usize) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 in
        ld(slice_f32(true), true),   // 2 out
    ]);
    let (inp, out) = (local(1), local(2));
    let arr = al.add(
        Ty::Array {
            elem: Box::new(Ty::F32),
            len: n as u32,
        },
        true,
    );
    let t = al.add(Ty::Usize, false);
    let i = al.add(Ty::Usize, true);
    let cmp = al.add(Ty::Bool, false);
    let idx = al.add(Ty::Usize, true);
    let v = al.add(Ty::F32, false);
    let v2 = al.add(Ty::F32, false);
    let acc = al.add(Ty::F32, true);
    let av = al.add(Ty::F32, false);
    let i_new = al.add(Ty::Usize, false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };

    // bb0: t = thread X -> bb1
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(t),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    // bb1: i = 0 -> bb2
    let bb1 = BasicBlock {
        statements: vec![Statement::Assign(Place::local(i), Rvalue::Use(cu(0)))],
        terminator: goto(2),
    };
    // bb2 (fill hdr): cmp = i<n; if cmp==0 -> sum-init(4), else fill-body(3)
    let bb2 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), cu(n)),
        )],
        terminator: guard(cmp, 4, 3),
    };
    // bb3 (fill body): idx = t*n + i; v = in[idx]; arr[i] = v*2; i += 1 -> bb2
    let bb3 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(idx),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(t)), cu(n)),
            ),
            Statement::Assign(
                Place::local(idx),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(idx)), copy(Place::local(i))),
            ),
            Statement::Assign(Place::local(v), Rvalue::Use(copy(elem(inp, idx)))),
            Statement::Assign(
                Place::local(v2),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(v)),
                    Operand::Const(Constant::F32(2.0)),
                ),
            ),
            Statement::Assign(arr_elem(arr, i), Rvalue::Use(copy(Place::local(v2)))),
            Statement::Assign(
                Place::local(i_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), cu(1)),
            ),
            Statement::Assign(Place::local(i), Rvalue::Use(copy(Place::local(i_new)))),
        ],
        terminator: goto(2),
    };
    // bb4 (sum init): acc = 0; i = 0 -> bb5
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(acc),
                Rvalue::Use(Operand::Const(Constant::F32(0.0))),
            ),
            Statement::Assign(Place::local(i), Rvalue::Use(cu(0))),
        ],
        terminator: goto(5),
    };
    // bb5 (sum hdr): cmp = i<n; if cmp==0 -> store(7), else sum-body(6)
    let bb5 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(cmp),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), cu(n)),
        )],
        terminator: guard(cmp, 7, 6),
    };
    // bb6 (sum body): av = arr[i]; acc += av; i += 1 -> bb5
    let bb6 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(av), Rvalue::Use(copy(arr_elem(arr, i)))),
            Statement::Assign(
                Place::local(acc),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(acc)), copy(Place::local(av))),
            ),
            Statement::Assign(
                Place::local(i_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), cu(1)),
            ),
            Statement::Assign(Place::local(i), Rvalue::Use(copy(Place::local(i_new)))),
        ],
        terminator: goto(5),
    };
    // bb7 (store): out[t] = acc -> return
    let bb7 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, t),
            Rvalue::Use(copy(Place::local(acc))),
        )],
        terminator: Terminator::Return,
    };
    Body::new(
        name,
        2,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7],
    )
}

/// Serial reduce over the last axis: `out[r] = reduce_j x[r*cols + j]` for `cols` columns, storage dtype
/// `dt`, accumulating in f32 (spec 024): each bf16 element widens to f32, the reduce runs in an f32
/// accumulator, the result narrows to `dt` on store (byte-identical for `dt = Ty::F32`, the convenience
/// that `reduce_last` used to expose before it moved to `poot_test_util::kernel_fixtures`, card 671, its
/// only caller). `op` is the reduce op (`Add` for sum, `Max`). `init` seeds the accumulator (0.0 for sum,
/// a large negative for max). The output length (rows) comes from `Len(out)`; `cols` is baked in (the
/// reduced extent).
pub fn reduce_last_dt(name: &str, dt: Ty, op: BinOp, cols: usize, init: f32) -> Body {
    // _0 ret, _1 x, _2 out, _3 r(row), _4 rows, _5 r_ok, _6 acc, _7 j, _8 j_lt, _9 base, _10 pos,
    // _11 v, _12 acc_new, _13 j_new
    let bf16 = dt != Ty::F32;
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false),
        ld(slice_dtype(dt.clone(), true), true),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(Ty::F32, true),
        ld(Ty::Usize, true),
        ld(Ty::Bool, false),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::F32, false),
        ld(Ty::F32, false),
        ld(Ty::Usize, false),
    ];
    let x = local(1);
    let out = local(2);
    let (row, rows, r_ok, acc, j, j_lt) =
        (local(3), local(4), local(5), local(6), local(7), local(8));
    let (base, pos, v, acc_new, j_new) = (local(9), local(10), local(11), local(12), local(13));
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(row),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(rows), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(r_ok),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(row)), copy(Place::local(rows))),
            ),
        ],
        terminator: guard(r_ok, 6, 2), // out-of-bounds row -> bare return (bb6), not the store (bb5)
    };
    // bb2: acc=init; j=0 -> bb3
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(acc),
                Rvalue::Use(Operand::Const(Constant::F32(init))),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(cu(0))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb3 (loop hdr): j_lt = j < cols; if !j_lt -> bb5(store) else bb4
    let bb3 = BasicBlock {
        statements: vec![Statement::Assign(
            Place::local(j_lt),
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(j)), cu(cols)),
        )],
        terminator: guard(j_lt, 5, 4),
    };
    // bb4 (body): base=row*cols; pos=base+j; v=x[pos]; acc=acc op v; j=j+1 -> bb3
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(base),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(cols)),
            ),
            Statement::Assign(
                Place::local(pos),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(base)), copy(Place::local(j))),
            ),
            // widen the (possibly bf16) element to f32 for the f32 accumulate; a no-op for f32.
            Statement::Assign(
                Place::local(v),
                if bf16 {
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: copy(elem(x, pos)),
                    }
                } else {
                    Rvalue::Use(copy(elem(x, pos)))
                },
            ),
            Statement::Assign(
                Place::local(acc_new),
                Rvalue::BinaryOp(op, copy(Place::local(acc)), copy(Place::local(v))),
            ),
            Statement::Assign(Place::local(acc), Rvalue::Use(copy(Place::local(acc_new)))),
            Statement::Assign(
                Place::local(j_new),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(j)), cu(1)),
            ),
            Statement::Assign(Place::local(j), Rvalue::Use(copy(Place::local(j_new)))),
        ],
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };
    // bb5 (store): out[row] = acc (narrowed to dt for bf16); return
    let bb5 = BasicBlock {
        statements: vec![Statement::Assign(
            elem(out, row),
            if bf16 {
                Rvalue::Cast {
                    to: dt.clone(),
                    operand: copy(Place::local(acc)),
                }
            } else {
                Rvalue::Use(copy(Place::local(acc)))
            },
        )],
        terminator: Terminator::Return,
    };
    let bb6 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6])
}

#[cfg(test)]
mod tests {
    use poot_codegen::{Target, compile};
    use poot_runtime::KernelBuffer;

    use super::*;
    use crate::test_support::{ctx, spv, stage_nvptx};

    #[test]
    fn reduce_last_lds_parallel() {
        // parallel LDS reduction: 3 rows x 100 cols, one workgroup of 32 lanes per row. Sum and max, each
        // bit-close to a serial reference (the parallel reduction reorders the accumulation).
        let Some(ctx) = ctx() else { return };
        let (rows, cols, w) = (3usize, 100usize, 32usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32) * 0.1 - 5.0)
            .collect();

        // sum
        let body = reduce_last_lds("rsum", BinOp::Add, cols, 0.0, w);
        stage_nvptx(&body);
        let s = spv(&body, "rsum_lds");
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::write_f32(rows),
        ];
        ctx.dispatch(
            "t",
            &s,
            [w as u32, 1, 1],
            [(rows * w) as u32, 1, 1],
            &mut bufs,
        )
        .unwrap();
        let got = bufs[1].as_f32();
        for r in 0..rows {
            let want: f32 = x[r * cols..r * cols + cols].iter().sum();
            assert!(
                (got[r] - want).abs() <= 1e-3,
                "sum row {r}: {} vs {want}",
                got[r]
            );
        }

        // max
        let s = spv(
            &reduce_last_lds("rmax", BinOp::Max, cols, f32::NEG_INFINITY, w),
            "rmax_lds",
        );
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::write_f32(rows),
        ];
        ctx.dispatch(
            "t",
            &s,
            [w as u32, 1, 1],
            [(rows * w) as u32, 1, 1],
            &mut bufs,
        )
        .unwrap();
        let got = bufs[1].as_f32();
        for r in 0..rows {
            let want = x[r * cols..r * cols + cols]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(got[r], want, "max row {r}");
        }
    }

    #[test]
    fn array_sum_probe_emits_valid_nvptx() {
        // the private fixed-size array for flash attention's o[D] accumulator: dynamically-indexed store +
        // load into a per-thread alloca [n x f32], NVPTX-only.
        stage_nvptx(&array_sum_probe("array_sum_probe", 8));
    }

    #[test]
    fn array_sum_probe_rejected_on_spirv() {
        // a per-thread private array crashes the LLVM SPIR-V backend, so codegen must reject it on
        // SpirvVulkan with a named diagnostic rather than emit a crashing module.
        let dir = std::env::temp_dir().join("poot-kernelgen-unit-test");
        std::fs::create_dir_all(&dir).unwrap();
        let body = array_sum_probe("array_sum_probe_spv", 8);
        let out = dir.join("array_sum_probe_spv.spv");
        let err = compile(&body, Target::SpirvVulkan, &out)
            .expect_err("a private array must be rejected on SpirvVulkan");
        let msg = err.to_string();
        assert!(
            msg.contains("private array") && msg.contains("NVPTX-only"),
            "unexpected rejection message: {msg}"
        );
    }
}
