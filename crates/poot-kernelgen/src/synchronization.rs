//! Atomic counters and workgroup atomic smoke kernels (this crate's own tests; the cross-block fence kernels are
//! `poot_kernel_ir::test_support`).

use crate::helpers::{copy, elem, ld, local, slice_dtype, slice_f32};
use poot_kernel_ir::WorkgroupLocalDecl;
use poot_kernel_ir::{
    AtomicOp, BasicBlock, Body, Constant, Operand, Place, Rvalue, Statement, Terminator, Ty,
};

/// Every dispatched thread atomically adds 1.0 to `out[0]`: the MPK event-counter primitive (a dependency
/// counter a producing task bumps and a consuming task spin-waits on). Relaxed ordering: correct for a counter
/// read after the dispatch completes; the cross-CTA acquire/release spin-wait is a separate concern. With N
/// threads dispatched, `out[0] == N`.
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

#[cfg(test)]
mod tests {
    use poot_codegen::{Target, compile};
    use poot_wgpu_runtime::KernelBuffer;

    use super::*;
    use crate::unit_helpers::{ctx, ptx, spirv_word_count, spv, which};

    /// The LDS atomic orders at the workgroup, never the device: only the workgroup's threads reach LDS, and a
    /// device-scope `atomicrmw` on addrspace(3) asks the hardware for ordering nobody can observe. The
    /// global-memory atomic on the same target keeps its own (device or default) scope. Checked on the emitted
    /// LLVM IR per target. Mutation: have `WorkgroupLocalAtomic` call `atomic_syncscope` instead of
    /// `workgroup_atomic_syncscope`; the SPIR-V rows turn red.
    #[test]
    fn workgroup_local_atomic_is_ordered_at_workgroup_scope() {
        let body = workgroup_atomic_add_smoke("wgatom_scope");
        let lds_atomic = |target: Target| {
            let ir = poot_codegen::emit_llvm_ir(&body, target).unwrap();
            ir.lines()
                .find(|line| line.contains("atomicrmw") && line.contains("addrspace(3)"))
                .unwrap_or_else(|| panic!("no LDS atomicrmw in:\n{ir}"))
                .to_string()
        };
        for (target, scope) in [
            (Target::SpirvVulkan, "workgroup"),
            (Target::SpirvOpenCl, "workgroup"),
            (Target::Nvptx, "block"),
            (Target::AmdGcn(poot_target::AmdArch::gfx1151()), "workgroup"),
        ] {
            let line = lds_atomic(target);
            assert!(
                line.contains(&format!("syncscope(\"{scope}\")")),
                "{target:?}: expected syncscope(\"{scope}\") in {line}"
            );
        }
    }

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
