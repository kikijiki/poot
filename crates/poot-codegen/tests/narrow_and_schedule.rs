//! `Rvalue::Narrow` and `Statement::SchedulingBoundary` (POOT-1441) per target: the exact narrowing never emits a
//! rounding sequence (AMDGCN bf16 takes the top half of the f32, every other narrowing is one `fptrunc`), and the
//! scheduling boundary is `llvm.amdgcn.sched.barrier(0)` on AMDGCN and nothing on the targets whose compiler takes
//! no such order. The compile half skips (passes) when a toolchain is absent, as `mul_add.rs` does.

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{Target, compile, emit_llvm_ir};
use poot_kernel_ir::{
    BasicBlock, BlockId, Body, IndexAxis, Local, LocalDecl, Operand, Place, ProjectionElem, Rvalue,
    Statement, Terminator, Ty,
};

const SCHED_BARRIER: &str = "call void @llvm.amdgcn.sched.barrier(i32 0)";

fn l(index: u32) -> Local {
    Local { index }
}

fn copy(local: u32) -> Operand {
    Operand::Copy(Place::local(l(local)))
}

fn element(param: u32, index: u32) -> Place {
    Place {
        local: l(param),
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(index))],
    }
}

fn buffer(elem: Ty, mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(elem))),
    }
}

/// `out[i] = narrow(input[i])` for thread `i`, with a scheduling boundary between the read and the narrowing:
/// `_1` input (f32), `_2` out (`to`), `_3` thread, `_4` value, `_5` narrowed.
fn narrow_kernel(to: Ty) -> Body {
    let locals = [
        Ty::Unit,
        buffer(Ty::F32, false),
        buffer(to.clone(), true),
        Ty::Usize,
        Ty::F32,
        to.clone(),
    ]
    .into_iter()
    .map(|ty| LocalDecl { ty, mutable: true })
    .collect();
    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(l(3)),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(l(4)),
                    Rvalue::Use(Operand::Copy(element(1, 3))),
                ),
                Statement::SchedulingBoundary,
                Statement::Assign(
                    Place::local(l(5)),
                    Rvalue::Narrow {
                        to,
                        operand: copy(4),
                    },
                ),
                Statement::Assign(element(2, 3), Rvalue::Use(copy(5))),
            ],
            terminator: Terminator::Return,
        },
    ];
    let mut body = Body::new("narrow_probe", 2, locals, blocks);
    body.workgroup_size = [64, 1, 1];
    body
}

/// The narrow dtypes each target stores: bf16 is AMDGCN and NVPTX only (SPIR-V has no bf16 type).
fn cases() -> Vec<(Target, Ty)> {
    let amd = Target::AmdGcn(poot_target::AmdArch::gfx1151());
    vec![
        (amd, Ty::BF16),
        (amd, Ty::F16),
        (Target::Nvptx, Ty::BF16),
        (Target::Nvptx, Ty::F16),
        (Target::SpirvVulkan, Ty::F16),
    ]
}

/// The emitted IR: AMDGCN carries one `sched.barrier(0)` and its declaration, every other target none; an AMDGCN
/// bf16 narrowing is the shift of the f32's top half with no rounding bias (`32767`, the bias of the rounding
/// `Cast` lowering), every other narrowing one `fptrunc`. Mutation: lower `Rvalue::Narrow` through `emit_cast`;
/// the AMDGCN bf16 row emits the rounding bias and this test goes red. Mutation: emit nothing for the boundary on
/// AMDGCN; its row finds no `sched.barrier` and goes red.
#[test]
fn a_narrowing_never_rounds_and_a_boundary_is_a_sched_barrier_only_on_amdgcn() {
    for (target, to) in cases() {
        let ir = emit_llvm_ir(&narrow_kernel(to.clone()), target)
            .unwrap_or_else(|e| panic!("{target:?} {to:?}: {e}"));
        let amdgcn = matches!(target, Target::AmdGcn(_));
        assert_eq!(
            ir.matches(SCHED_BARRIER).count(),
            usize::from(amdgcn),
            "{target:?} {to:?}:\n{ir}"
        );
        assert_eq!(
            ir.contains("declare void @llvm.amdgcn.sched.barrier(i32 immarg)"),
            amdgcn,
            "{target:?}:\n{ir}"
        );
        assert!(!ir.contains("32767"), "{target:?} {to:?} rounds:\n{ir}");
        if amdgcn && to == Ty::BF16 {
            assert!(ir.contains("lshr i32") && !ir.contains("fptrunc"), "{ir}");
        } else {
            assert_eq!(
                ir.matches("fptrunc float").count(),
                1,
                "{target:?} {to:?}:\n{ir}"
            );
        }
    }
}

fn out_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join("poot-codegen-test").join(tag);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn have(bin: &str) -> bool {
    Command::new(bin).arg("--version").output().is_ok()
}

/// Every case compiles with its target's toolchain (and the SPIR-V one is `spirv-val` clean), so the boundary and
/// the narrowing are lowerings each backend accepts, not only IR text.
#[test]
fn the_narrowing_and_the_boundary_compile_on_every_gpu_target() {
    let dir = out_dir("narrow-and-schedule");
    for (target, to) in cases() {
        let available = match target {
            Target::AmdGcn(_) => std::env::var("POOT_AMD_CLANG")
                .is_ok_and(|clang| Command::new(clang).arg("--version").output().is_ok()),
            Target::Nvptx => have("llc"),
            Target::SpirvVulkan => have("llc") && have("spirv-val"),
            Target::AieCore => false,
        };
        if !available {
            eprintln!("POOT-SKIP poot-codegen: no toolchain for {target:?}");
            continue;
        }
        let name = format!("narrow_{:?}", to).to_lowercase();
        let out = poot_codegen::artifact_path(&dir, &name, target);
        compile(&narrow_kernel(to.clone()), target, &out)
            .unwrap_or_else(|e| panic!("{target:?} {to:?}: {e}"));
        if target == Target::SpirvVulkan {
            let val = Command::new("spirv-val")
                .args(["--target-env", "vulkan1.3"])
                .arg(&out)
                .output()
                .unwrap();
            assert!(
                val.status.success(),
                "spirv-val:\n{}",
                String::from_utf8_lossy(&val.stderr)
            );
        }
    }
}
