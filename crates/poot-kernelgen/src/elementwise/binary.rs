use super::*;

/// numpy-broadcasting elementwise binary `c = a op b`, fully general over static shapes, in storage dtype
/// `dt` (f32 or bf16), computing in f32 (spec 024): bf16 operands widen on load and the result narrows on
/// store (the casts are no-ops for `dt = Ty::F32` - the convenience that `binary_broadcast`/
/// `binary_broadcast_dt`/`binary_broadcast_dt_views` used to expose before they moved to
/// `poot_test_util::kernel_fixtures`, card 671, their only caller). For each output element `i`, the
/// operand offsets are computed from `i` via the (constant) output strides and each operand's
/// broadcast-effective strides (0 on a broadcast dim). Subsumes the same-shape case. Shapes are
/// right-aligned (leading-1 padded), as in `poot_graph_ir`'s broadcast rule.
///
/// Reads `a`/`b` through an arbitrary physical [`Layout`] (spec 132: a transpose/slice/broadcast read
/// directly by a strided-capable consumer, no materialize copy). Output is identical for both operands
/// `Layout::contiguous`.
///
/// `x_groups`: `None` keeps the flat `ThreadIndexCall(X)` (one dispatch dimension, capped at
/// `65535 * workgroup_size` threads by wgpu's `gridDim.x` limit). `Some(xg)` folds the launch onto a 2-D
/// grid (card 159 Inc 3): read `GroupX`/`GroupY`, form `group_id = GroupY*xg + GroupX`, then
/// `i = group_id*workgroup_size + LocalX`. This matches the reconstruction in
/// `pootc/tests/kernels/scatter_update.rs`/`dyn_update_slice.rs`/`index_remap.rs`, so `dispatch_grid`'s
/// `elementwise_2d_grid(out_numel)` launches the matching grid. The caller must bake
/// `workgroup_size = [ELEMENTWISE_2D_WIDTH, 1, 1]` on the returned `Body`; this function leaves the
/// default `[64,1,1]` because it cannot see `poot-graph-plan`'s constant.
#[allow(clippy::too_many_arguments)]
pub fn binary_broadcast_dt_views_grid(
    name: &str,
    op: BinOp,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    a_layout: &Layout,
    b_shape: &[usize],
    b_layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    binary_broadcast_typed_views_grid(
        name, op, dt, out_shape, a_shape, a_layout, b_shape, b_layout, x_groups, false,
    )
}

/// Exact unsigned comparison over I32 storage bit patterns. Operands are bitcast to U32 for the
/// comparison; the bool result is written as I32 zero or one.
#[allow(clippy::too_many_arguments)]
pub fn binary_broadcast_i32_geu_views_grid(
    name: &str,
    out_shape: &[usize],
    a_shape: &[usize],
    a_layout: &Layout,
    b_shape: &[usize],
    b_layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    binary_broadcast_typed_views_grid(
        name,
        BinOp::Ge,
        Ty::I32,
        out_shape,
        a_shape,
        a_layout,
        b_shape,
        b_layout,
        x_groups,
        true,
    )
}

/// Exact unsigned remainder over I32 storage bit patterns. Operands are bitcast to U32 for the
/// remainder; the u32 result is bitcast back to the I32 storage pattern.
#[allow(clippy::too_many_arguments)]
pub fn binary_broadcast_i32_remu_views_grid(
    name: &str,
    out_shape: &[usize],
    a_shape: &[usize],
    a_layout: &Layout,
    b_shape: &[usize],
    b_layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    binary_broadcast_typed_views_grid(
        name,
        BinOp::Rem,
        Ty::I32,
        out_shape,
        a_shape,
        a_layout,
        b_shape,
        b_layout,
        x_groups,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn binary_broadcast_typed_views_grid(
    name: &str,
    op: BinOp,
    dt: Ty,
    out_shape: &[usize],
    a_shape: &[usize],
    a_layout: &Layout,
    b_shape: &[usize],
    b_layout: &Layout,
    x_groups: Option<usize>,
    unsigned_operands: bool,
) -> Body {
    let r = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let (a_eff, a_base) = view_eff_strides(out_shape, a_shape, a_layout);
    let (b_eff, b_base) = view_eff_strides(out_shape, b_shape, b_layout);

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),                       // 0 ret
        ld(slice_dtype(dt.clone(), false), false), // 1 a
        ld(slice_dtype(dt.clone(), false), false), // 2 b
        ld(slice_dtype(dt.clone(), true), true),   // 3 c
    ]);
    let (a, b, c) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let a_off = al.add(Ty::Usize, true);
    let b_off = al.add(Ty::Usize, true);
    let div = al.add(Ty::Usize, false);
    let md = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let off_new = al.add(Ty::Usize, false);
    let rem = al.add(Ty::Usize, false);
    let narrow_float = matches!(dt, Ty::BF16 | Ty::F16);
    let compute_ty = if narrow_float { Ty::F32 } else { dt.clone() };
    let va = al.add(compute_ty.clone(), false);
    let vb = al.add(compute_ty.clone(), false);
    let res = al.add(compute_ty.clone(), false);
    let rescmp = al.add(Ty::Bool, false); // i1 result of a comparison op, before conversion to storage type
    let unsigned = unsigned_operands.then(|| (al.add(Ty::U32, false), al.add(Ty::U32, false)));
    let unsigned_rem = (unsigned_operands && op == BinOp::Rem).then(|| al.add(Ty::U32, false));
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));

    // Seed a_off/b_off with each operand's layout base offset (0 unless it is a view), then index math, op, store.
    let mut body = vec![
        Statement::Assign(Place::local(a_off), Rvalue::Use(cu(a_base))),
        Statement::Assign(Place::local(b_off), Rvalue::Use(cu(b_base))),
    ];
    // Unravel the linear index `i` into per-dim coords by a running remainder (div + mul + sub), never
    // `urem`: NVPTX software-emulates the i64 remainder and miscompiles it for a large dividend, which
    // corrupted broadcast reads at large N (prefill garbled at n=864; power-of-2 dims lower to AND and were
    // safe). coord[d]=rem/stride[d]; rem-=coord*stride[d].
    let accumulate = |off: Local, eff: &[usize], body: &mut Vec<Statement>| {
        body.push(Statement::Assign(
            Place::local(rem),
            Rvalue::Use(copy(Place::local(i))),
        ));
        for d in 0..r {
            // coord = rem / out_strides[d]
            body.push(Statement::Assign(
                Place::local(div),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(rem)), cu(out_strides[d])),
            ));
            if eff[d] != 0 {
                // off += coord * eff[d]
                body.push(Statement::Assign(
                    Place::local(term),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(eff[d])),
                ));
                body.push(Statement::Assign(
                    Place::local(off_new),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(off)),
                        copy(Place::local(term)),
                    ),
                ));
                body.push(Statement::Assign(
                    Place::local(off),
                    Rvalue::Use(copy(Place::local(off_new))),
                ));
            }
            if d + 1 < r {
                // rem -= coord * out_strides[d]
                body.push(Statement::Assign(
                    Place::local(md),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(div)), cu(out_strides[d])),
                ));
                body.push(Statement::Assign(
                    Place::local(off_new),
                    Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(rem)), copy(Place::local(md))),
                ));
                body.push(Statement::Assign(
                    Place::local(rem),
                    Rvalue::Use(copy(Place::local(off_new))),
                ));
            }
        }
    };
    // A non-broadcast operand (`eff == out_strides`) has linear offset `i` plus its layout base (already
    // seeded in `off`): emit `off = i + off` instead of a div/rem chain.
    let set_off = |off: Local, eff: &[usize], base: usize, body: &mut Vec<Statement>| {
        if eff == out_strides.as_slice() {
            if base == 0 {
                body.push(Statement::Assign(
                    Place::local(off),
                    Rvalue::Use(copy(Place::local(i))),
                ));
            } else {
                body.push(Statement::Assign(
                    Place::local(off_new),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(i)), copy(Place::local(off))),
                ));
                body.push(Statement::Assign(
                    Place::local(off),
                    Rvalue::Use(copy(Place::local(off_new))),
                ));
            }
        } else {
            accumulate(off, eff, body);
        }
    };
    set_off(a_off, &a_eff, a_base, &mut body);
    set_off(b_off, &b_eff, b_base, &mut body);
    // Widen narrow floats to f32; F32 and I32 compute in their storage type.
    let widen = |src: Place| -> Rvalue {
        if narrow_float {
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(src),
            }
        } else {
            Rvalue::Use(copy(src))
        }
    };
    body.push(Statement::Assign(Place::local(va), widen(elem(a, a_off))));
    body.push(Statement::Assign(Place::local(vb), widen(elem(b, b_off))));
    let (cmp_a, cmp_b) = if let Some((ua, ub)) = unsigned {
        body.push(Statement::Assign(
            Place::local(ua),
            Rvalue::Bitcast {
                to: Ty::U32,
                operand: copy(Place::local(va)),
            },
        ));
        body.push(Statement::Assign(
            Place::local(ub),
            Rvalue::Bitcast {
                to: Ty::U32,
                operand: copy(Place::local(vb)),
            },
        ));
        (copy(Place::local(ua)), copy(Place::local(ub)))
    } else {
        (copy(Place::local(va)), copy(Place::local(vb)))
    };
    if is_cmp(op) {
        // Comparison yields i1; convert to the output scalar type (1 / 0).
        body.push(Statement::Assign(
            Place::local(rescmp),
            Rvalue::BinaryOp(op, cmp_a, cmp_b),
        ));
        body.push(Statement::Assign(
            Place::local(res),
            Rvalue::Cast {
                to: compute_ty.clone(),
                operand: copy(Place::local(rescmp)),
            },
        ));
    } else if let Some(rem_local) = unsigned_rem {
        // Unsigned remainder on the already-bitcast U32 operands; bitcast the u32 result back to the
        // I32 storage pattern.
        body.push(Statement::Assign(
            Place::local(rem_local),
            Rvalue::BinaryOp(BinOp::Rem, cmp_a, cmp_b),
        ));
        body.push(Statement::Assign(
            Place::local(res),
            Rvalue::Bitcast {
                to: compute_ty.clone(),
                operand: copy(Place::local(rem_local)),
            },
        ));
    } else if matches!(op, BinOp::Shl | BinOp::Shr) && dt == Ty::I32 {
        let rv = emit_i32_shift(
            op,
            copy(Place::local(va)),
            copy(Place::local(vb)),
            &mut al,
            &mut body,
        );
        body.push(Statement::Assign(Place::local(res), rv));
    } else {
        body.push(Statement::Assign(
            Place::local(res),
            Rvalue::BinaryOp(op, copy(Place::local(va)), copy(Place::local(vb))),
        ));
    }
    // Narrow back to narrow-float storage; F32 and I32 store directly.
    body.push(Statement::Assign(
        elem(c, i),
        if narrow_float {
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(res)),
            }
        } else {
            Rvalue::Use(copy(Place::local(res)))
        },
    ));

    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: body,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    match x_groups {
        None => {
            // Flat 1-D grid: `i = ThreadIndexCall(X)` (global thread id, `group*wg+local` folded in by the backend).
            let bb0 = BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(i),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            };
            Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3])
        }
        Some(xg) => {
            // 2-D grid fold (card 159 Inc 3): gx=GroupX -> gy=GroupY -> group_id=gy*xg+gx,
            // lane=LocalX -> i=group_id*wg+lane -> bb1 (tail-length guard). The caller must bake
            // `body.workgroup_size[0]` to the same literal `wg`.
            let wg = ELEMENTWISE_2D_WORKGROUP_SIZE;
            let gx = al.add(Ty::Usize, false);
            let gy = al.add(Ty::Usize, false);
            let group_id = al.add(Ty::Usize, false);
            let lane = al.add(Ty::Usize, false);
            let gy_idx: u32 = 4;
            let lane_idx: u32 = 5;
            let combine_idx: u32 = 6;
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
                3,
                al.locals,
                vec![bb0, bb1, bb2, bb3, bb_gy, bb_lane, bb_combine],
            )
        }
    }
}

/// Minimal `Ty::Vec` exerciser (spec 134 P1): each thread vec4-loads group `g` from `a`, adds a broadcast
/// constant `add`, and vec4-stores to `out`: `out[4g+l] = a[4g+l] + add`, lane `l` in `0..4`. Exercises
/// `Rvalue::VectorLoad`, `Rvalue::VectorSplat`, `BinaryOp` lanewise on `Ty::Vec`, and
/// `Statement::VectorStore`. No bounds-check block: the dispatch is sized exactly to the buffer.
#[cfg(test)]
pub(crate) fn vec4_add_splat(name: &str, add: f32) -> Body {
    // _0 ret, _1 a: &[f32], _2 out: &mut [f32], _3 i (vector-group index), _4 v (loaded vec4),
    // _5 splat (broadcast add), _6 sum (v + splat).
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_f32(true), true),
        ld(Ty::Usize, false),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
        ld(
            Ty::Vec {
                elem: Box::new(Ty::F32),
                lanes: 4,
            },
            false,
        ),
    ];
    let (a, out, i, v, splat, sum) = (local(1), local(2), local(3), local(4), local(5), local(6));
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
            Statement::Assign(Place::local(v), Rvalue::VectorLoad { place: elem(a, i) }),
            Statement::Assign(
                Place::local(splat),
                Rvalue::VectorSplat(Operand::Const(Constant::F32(add))),
            ),
            Statement::Assign(
                Place::local(sum),
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(v)), copy(Place::local(splat))),
            ),
            Statement::VectorStore {
                place: elem(out, i),
                value: copy(Place::local(sum)),
            },
        ],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1])
}

/// Elementwise binary `c[i] = a[i] op b[i]` over `n` elements (a, b same length as c).
pub fn binary(name: &str, op: BinOp) -> Body {
    // _0 ret, _1 a, _2 b, _3 c, _4 i, _5 len, _6 cmp, _7 r
    let locals = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_f32(false), false),
        ld(slice_f32(true), true),
        ld(Ty::Usize, false),
        ld(Ty::Usize, false),
        ld(Ty::Bool, false),
        ld(Ty::F32, false),
    ];
    let (a, b, c, i, len, cmp, r) = (
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
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
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
                Place::local(r),
                Rvalue::BinaryOp(op, copy(elem(a, i)), copy(elem(b, i))),
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(r)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 3, locals, vec![bb0, bb1, bb2, bb3])
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use poot_codegen::{Target, compile};
    use poot_runtime::{Context, KernelBuffer};

    use super::vec4_add_splat;

    fn spv(body: &poot_kernel_ir::Body, name: &str) -> poot_runtime::CompiledKernel {
        let dir: PathBuf = std::env::temp_dir().join("poot-kernelgen-test").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        compile(body, Target::SpirvVulkan, &out).expect("compile");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, Target::SpirvVulkan, bytes)
    }

    fn ctx() -> Option<Context> {
        match Context::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                None
            }
        }
    }

    /// Is `tool` on PATH? (small local helper for the optional spirv-val check.)
    fn which(tool: &str) -> bool {
        std::process::Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// `vec4_add_splat` end to end (SpirvVulkan only), split out of `tests/run.rs` (card 546c: its
    /// only other caller, poot-ptx-gpu's schedule-batch atomicity test, was deleted with the resident
    /// E4M3 region's `compile_schedule`/`compiled_batch` stack, which made this an external-only
    /// consumer that `pub(crate)` cannot serve): (1) `spirv-val` the module (vec4 buffer resource,
    /// splat and vec4 store must be strictly valid, not just RADV-tolerated); (2) dispatch on the
    /// Arc/RADV and compare every lane to a scalar CPU reference - one workgroup of 64 threads, each
    /// vec4-loading its own 4-float group (256 floats).
    #[test]
    fn vec4_add_splat_validates_and_matches_cpu() {
        let add = 1.5f32;
        let body = vec4_add_splat("vec4add", add);
        let s = spv(&body, "vec4add");

        // strict SPIR-V validity (skip only if spirv-val is absent).
        let dir = std::env::temp_dir()
            .join("poot-kernelgen-test")
            .join("vec4add");
        let spv_path = poot_codegen::artifact_path(&dir, "vec4add", Target::SpirvVulkan);
        if which("spirv-val") {
            let v = std::process::Command::new("spirv-val")
                .arg("--target-env")
                .arg("vulkan1.3")
                .arg(&spv_path)
                .output()
                .expect("run spirv-val");
            assert!(
                v.status.success(),
                "spirv-val failed for vec4_add_splat:\n{}",
                String::from_utf8_lossy(&v.stderr)
            );
        } else {
            eprintln!("spirv-val not on PATH; skipping the validity check");
        }

        let Some(ctx) = ctx() else { return };
        // 64 vec4 groups (one per thread in the single workgroup) * 4 lanes = 256 floats.
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(n)];
        ctx.dispatch("test", &s, [64, 1, 1], [64, 1, 1], &mut bufs)
            .unwrap();
        let got = bufs[1].as_f32();
        let want: Vec<f32> = a.iter().map(|x| x + add).collect();
        assert_eq!(
            got,
            &want[..],
            "vec4_add_splat: gpu output must match the scalar CPU reference lane-for-lane"
        );
    }
}
