//! Atomic counters, cross-block handoffs, and fence smoke kernels.

#[cfg(test)]
use crate::helpers::slice_dtype;
use crate::helpers::{copy, elem, guard, ld, local, slice_f32};
#[cfg(test)]
use poot_kernel_ir::WorkgroupLocalDecl;
use poot_kernel_ir::{
    AtomicOp, BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, MemoryOrdering,
    MemoryScope, Operand, Place, Rvalue, Statement, Terminator, Ty,
};

/// Every dispatched thread atomically adds 1.0 to `out[0]`: the MPK event-counter primitive (a dependency
/// counter a producing task bumps and a consuming task spin-waits on). Relaxed ordering: correct for a counter
/// read after the dispatch completes; the cross-CTA acquire/release spin-wait is a separate concern. With N
/// threads dispatched, `out[0] == N`.
#[cfg(test)]
fn global_atomic_add_counter(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),       // 0 ret
        ld(slice_f32(true), true), // 1 out (mut slice)
        ld(Ty::Usize, false),      // 2 zero (the index 0)
        ld(Ty::F32, false),        // 3 old (the atomic's returned old value, unused)
    ];
    let (out, zero, old) = (local(1), local(2), local(3));
    let bb0 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(zero),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            Statement::Assign(
                Place::local(old),
                Rvalue::GlobalAtomic {
                    place: elem(out, zero),
                    value: Operand::Const(Constant::F32(1.0)),
                    op: AtomicOp::Add,
                },
            ),
        ],
        terminator: Terminator::Return,
    };
    Body::new(name, 1, locals, vec![bb0])
}

/// Smoke kernel for the workgroup-local (LDS) atomic codegen: atomic-add 1.0 to LDS `counter[0]`, store the
/// returned old value to `out[0]`. The LDS-scoped sibling of [`global_atomic_add_counter`]; it confirms the
/// addrspace(3) atomic emits valid SPIR-V and PTX (the runtime atomicrmw is the same instruction the global
/// counter proves on GPU). Used for in-tile reductions and histograms.
#[cfg(test)]
fn workgroup_atomic_add_smoke(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),       // 0 ret
        ld(slice_f32(true), true), // 1 out
        ld(Ty::F32, false),        // 2 old (the atomic's returned old value)
        ld(Ty::Usize, false),      // 3 zero (out index)
    ];
    let (out, old, zero) = (local(1), local(2), local(3));
    let bb0 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(zero),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            Statement::Assign(
                Place::local(old),
                Rvalue::WorkgroupLocalAtomic {
                    idx: Operand::Const(Constant::Usize(0)),
                    value: Operand::Const(Constant::F32(1.0)),
                    op: AtomicOp::Add,
                    array: 0,
                },
            ),
            Statement::Assign(elem(out, zero), Rvalue::Use(copy(Place::local(old)))),
        ],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 1, locals, vec![bb0]);
    body.workgroup_size = [64, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: 1,
    }];
    body
}

/// Compile-only smoke kernel for the workgroup-local (LDS) CAS codegen: a single CAS on LDS `counter[0]`
/// (`0 -> 1`), storing the returned old value to `out[0]`. Same `cmpxchg` lowering as a global-memory CAS,
/// only the address space differs, so a compile check on both backends suffices (no retry loop needed).
#[cfg(test)]
fn workgroup_compare_exchange_smoke(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),                  // 0 ret
        ld(slice_dtype(Ty::U32, true), true), // 1 out
        ld(Ty::U32, false),                   // 2 old (the CAS's returned old value)
        ld(Ty::Usize, false),                 // 3 zero (out index)
    ];
    let (out, old, zero) = (local(1), local(2), local(3));
    let bb0 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(zero),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            Statement::Assign(
                Place::local(old),
                Rvalue::WorkgroupLocalCompareExchange {
                    idx: Operand::Const(Constant::Usize(0)),
                    expected: Operand::Const(Constant::U32(0)),
                    desired: Operand::Const(Constant::U32(1)),
                    array: 0,
                },
            ),
            Statement::Assign(elem(out, zero), Rvalue::Use(copy(Place::local(old)))),
        ],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 1, locals, vec![bb0]);
    body.workgroup_size = [64, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::U32,
        len: 1,
    }];
    body
}

/// Cross-block producer/consumer handoff for a cooperative grid, with the ordering made explicit via
/// `Statement::Fence` pairs instead of relying on an atomic's monotonic ordering alone. Producer:
/// `data[0]=42`, `Fence{Device,Release}`, then bump `flag[0]` (still a monotonic atomic; the fence
/// carries the ordering). Consumer: spin on `flag[0]` (unfenced atomic load), and once ready,
/// `Fence{Device,Acquire}` before reading `data[0]`; only then is the read guaranteed to observe the
/// producer's release. Needs every block co-resident on a real NVIDIA GPU (RADV cannot prove cross-CTA
/// forward progress), so it is compile-checked locally (PTX `fence`) only; see
/// `poot-codegen/tests/llc.rs`'s AMDGCN compile-only coverage of the same body.
pub fn cross_block_fence_spin_wait_coop(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),       // 0 ret
        ld(slice_f32(true), true), // 1 data
        ld(slice_f32(true), true), // 2 flag
        ld(slice_f32(true), true), // 3 result
        ld(Ty::Usize, false),      // 4 block (GroupX)
        ld(Ty::Usize, false),      // 5 zero
        ld(Ty::Bool, false),       // 6 is_producer (block==0)
        ld(Ty::F32, false),        // 7 fval (atomic read / producer's old)
        ld(Ty::Bool, false),       // 8 not_ready (fval < 1.0)
        ld(Ty::F32, false),        // 9 dval
    ];
    let (data, flag, result, block, zero, isp, fval, notready, dval) = (
        local(1),
        local(2),
        local(3),
        local(4),
        local(5),
        local(6),
        local(7),
        local(8),
        local(9),
    );
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let assign = |d: Local, rv: Rvalue| Statement::Assign(Place::local(d), rv);
    // bb0: block = GroupX
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(block),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    // bb1: zero=0; is_producer = block==0; producer -> bb2, else consumer spin -> bb3
    let bb1 = BasicBlock {
        statements: vec![
            assign(zero, Rvalue::Use(Operand::Const(Constant::Usize(0)))),
            assign(
                isp,
                Rvalue::BinaryOp(
                    BinOp::Eq,
                    copy(Place::local(block)),
                    Operand::Const(Constant::Usize(0)),
                ),
            ),
        ],
        terminator: guard(isp, 3, 2),
    };
    // bb2 (producer): data[0]=42; Fence{Device,Release}; atomic_add(flag[0], 1.0); return
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(elem(data, zero), Rvalue::Use(cf(42.0))),
            Statement::Fence {
                scope: MemoryScope::Device,
                ordering: MemoryOrdering::Release,
            },
            assign(
                fval,
                Rvalue::GlobalAtomic {
                    place: elem(flag, zero),
                    value: cf(1.0),
                    op: AtomicOp::Add,
                },
            ),
        ],
        terminator: Terminator::Return,
    };
    // bb3 (consumer spin head, SINGLE-EXIT self-loop): fval = atomic_add(flag[0], 0.0);
    //   not_ready = fval < 1.0; if not_ready loop to bb3 else done (bb4)
    let bb3 = BasicBlock {
        statements: vec![
            assign(
                fval,
                Rvalue::GlobalAtomic {
                    place: elem(flag, zero),
                    value: cf(0.0),
                    op: AtomicOp::Add,
                },
            ),
            assign(
                notready,
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(fval)), cf(1.0)),
            ),
        ],
        terminator: guard(notready, 4, 3),
    };
    // bb4 (done): Fence{Device,Acquire}; result[0] = data[0]; return. The fence must come before the read, or the
    // read is not guaranteed to observe the producer's release.
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Fence {
                scope: MemoryScope::Device,
                ordering: MemoryOrdering::Acquire,
            },
            assign(dval, Rvalue::Use(copy(elem(data, zero)))),
            Statement::Assign(elem(result, zero), Rvalue::Use(copy(Place::local(dval)))),
        ],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 3, locals, vec![bb0, bb1, bb2, bb3, bb4]);
    body.workgroup_size = [1, 1, 1];
    body
}

#[cfg(test)]
mod tests {
    use poot_codegen::{Target, compile};
    use poot_runtime::KernelBuffer;

    use super::*;
    use crate::test_support::{ctx, ptx, spirv_word_count, spv, which};

    #[test]
    fn workgroup_local_atomic_compiles_to_both_backends() {
        // LDS-scoped sibling of the global atomic counter (in-tile reductions): must emit valid SPIR-V
        // (RADV) and a PTX atom.* with no unresolved intrinsic. The runtime atomicrmw is the same as the
        // global counter's (addrspace(3) vs addrspace(1)), so a compile check suffices.
        let body = workgroup_atomic_add_smoke("wgatom");
        let s = spv(&body, "wgatom");
        assert!(
            spirv_word_count(&s) > 0,
            "workgroup-local atomic must emit valid SPIR-V"
        );
        let p = ptx(&body, "wgatom");
        assert!(
            p.contains("atom") && !p.contains(".extern .func"),
            "workgroup-local atomic must emit a PTX atom.* with no unresolved intrinsic"
        );
    }

    #[test]
    fn global_atomic_add_counter_sums_across_threads() {
        // Event-counter primitive: every thread atomically adds 1.0 to out[0]; with N threads the result
        // must be exactly N (a plain `+= 1.0` would race and undercount). out[0] is zero-initialized by
        // write_f32. The atomic must also lower to a PTX atom instruction with no unresolved externs.
        let p = ptx(&global_atomic_add_counter("atomctr"), "atomctr");
        assert!(
            p.contains("atom") && !p.contains(".extern .func"),
            "global atomic must emit a PTX atom.* with no unresolved intrinsic"
        );

        let Some(ctx) = ctx() else { return };
        let s = spv(&global_atomic_add_counter("atomctr"), "atomctr");
        // wg must match the Body's workgroup_size (64); 256 threads = 4 workgroups, exercising
        // cross-workgroup atomic contention.
        let n = 256u32;
        let mut bufs = [KernelBuffer::write_f32(1)];
        ctx.dispatch("test", &s, [64, 1, 1], [n, 1, 1], &mut bufs)
            .unwrap();
        assert_eq!(
            bufs[0].as_f32(),
            &[n as f32],
            "the global atomic counter must equal the thread count (no lost updates)"
        );
    }

    /// `Rvalue::GlobalAtomic`'s SPIR-V lowering must pass `spirv-val --target-env vulkan1.3` (RADV dispatch
    /// tolerates invalid SPIR-V). Without an explicit `syncscope`, LLVM's SPIR-V backend emits
    /// `OpAtomicFAddEXT`'s Memory Scope as `CrossDevice` (0), rejected by
    /// VUID-StandaloneSpirv-None-04638. `Emitter::atomic_syncscope` (poot-codegen/src/emit.rs) emits
    /// `syncscope("device")` on SpirvVulkan so the scope becomes `Device` (1). Compile + spirv-val only, no
    /// dispatch. Skips the assertion (not the test) if spirv-val is absent.
    #[test]
    fn global_atomic_add_counter_validates_as_spirv() {
        let s = spv(&global_atomic_add_counter("atomctr_val"), "atomctr_val");
        assert!(
            spirv_word_count(&s) > 0,
            "compile must produce a non-empty SPIR-V module"
        );

        let dir = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join("atomctr_val");
        let spv_path = poot_codegen::artifact_path(&dir, "atomctr_val", Target::SpirvVulkan);
        if which("spirv-val") {
            let v = std::process::Command::new("spirv-val")
                .arg("--target-env")
                .arg("vulkan1.3")
                .arg(&spv_path)
                .output()
                .expect("run spirv-val");
            assert!(
                v.status.success(),
                "spirv-val failed for global_atomic_add_counter:\n{}",
                String::from_utf8_lossy(&v.stderr)
            );
        } else {
            eprintln!("spirv-val not on PATH; skipping the validity check");
        }
    }

    /// Compile-only check for the LDS `WorkgroupLocalCompareExchange` sibling (mirrors
    /// `workgroup_local_atomic_compiles_to_both_backends`): the addrspace(3) CAS emits a PTX atom.*.cas,
    /// and SpirvVulkan rejects it the same way.
    #[test]
    fn workgroup_compare_exchange_compiles_to_ptx_and_rejects_spirv() {
        let body = workgroup_compare_exchange_smoke("wgcas");

        let p = ptx(&body, "wgcas");
        assert!(
            p.contains("atom") && !p.contains(".extern .func"),
            "workgroup-local compare-exchange must emit a PTX atom.*.cas with no unresolved intrinsic"
        );

        let dir = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join("wgcas_spv_reject");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, "wgcas", Target::SpirvVulkan);
        let err = compile(&body, Target::SpirvVulkan, &out);
        assert!(
            err.is_err(),
            "workgroup-local compare-exchange on SpirvVulkan must be a named rejection, not a \
             silently-accepted compile"
        );
    }
}
