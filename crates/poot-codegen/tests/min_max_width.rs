//! POOT-1167: integer and float `Min`/`Max` name the LLVM intrinsic of the operand's own width, on every target.
//! A fixed `.i32`/`.f32` suffix declared `i64 @llvm.umin.i32(i64, i64)` for `usize` on the 64-bit targets, which
//! `llc` rejects (and an `f64` max silently narrowed to the wrong overload). These tests read the emitted IR, so they
//! need no toolchain.

use poot_codegen::{Target, emit_llvm_ir};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, LocalDecl, Operand, Place, ProjectionElem,
    Rvalue, Statement, Terminator, Ty,
};

fn l(index: u32) -> Local {
    Local { index }
}

fn slice(ty: &Ty, mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(ty.clone()))),
    }
}

/// `k(a: &[T], out: &mut [T])`: `let i = thread_index(); out[i] = op(a[i], a[i]);`
fn elementwise_kernel(ty: &Ty, op: BinOp) -> Body {
    let decl = |ty: Ty, mutable| LocalDecl { ty, mutable };
    let locals = vec![
        decl(Ty::Unit, false),
        decl(slice(ty, false), false),
        decl(slice(ty, true), true),
        decl(Ty::Usize, false),
        decl(ty.clone(), false),
        decl(ty.clone(), false),
    ];
    let (a, out, i, x, r) = (l(1), l(2), l(3), l(4), l(5));
    let elem = |p: Local| Place {
        local: p,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
    };
    let entry = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let body = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(x), Rvalue::Use(Operand::Copy(elem(a)))),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(
                    op,
                    Operand::Copy(Place::local(x)),
                    Operand::Copy(Place::local(x)),
                ),
            ),
            Statement::Assign(elem(out), Rvalue::Use(Operand::Copy(Place::local(r)))),
        ],
        terminator: Terminator::Return,
    };
    Body::new("min_max", 2, locals, vec![entry, body])
}

/// `k(a: &[f32])`: `let i = thread_index(); let n = a.len(); let _ = op(i, n);`, the `usize` operands.
fn usize_kernel(op: BinOp) -> Body {
    let decl = |ty: Ty| LocalDecl { ty, mutable: false };
    let locals = vec![
        decl(Ty::Unit),
        decl(slice(&Ty::F32, false)),
        decl(Ty::Usize),
        decl(Ty::Usize),
        decl(Ty::Usize),
    ];
    let (a, i, n, r) = (l(1), l(2), l(3), l(4));
    let entry = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let body = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(n), Rvalue::Len(Place::local(a))),
            Statement::Assign(
                Place::local(r),
                Rvalue::BinaryOp(
                    op,
                    Operand::Copy(Place::local(i)),
                    Operand::Copy(Place::local(n)),
                ),
            ),
        ],
        terminator: Terminator::Return,
    };
    Body::new("min_max_usize", 1, locals, vec![entry, body])
}

fn targets() -> [Target; 3] {
    [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(poot_target::AmdArch::gfx1151()),
    ]
}

fn assert_intrinsic(ir: &str, target: Target, intrinsic: &str, llvm_ty: &str) {
    let declare = format!("declare {llvm_ty} @{intrinsic}({llvm_ty}, {llvm_ty})");
    assert!(
        ir.contains(&declare),
        "{target:?}: expected `{declare}` in the emitted IR:\n{ir}"
    );
    let call = format!("call {llvm_ty} @{intrinsic}({llvm_ty} ");
    assert!(
        ir.contains(&call),
        "{target:?}: expected a `{call}` call in the emitted IR:\n{ir}"
    );
}

#[test]
fn integer_min_max_name_the_operand_width() {
    for target in targets() {
        for (ty, min, max) in [
            (Ty::I32, "llvm.smin.i32", "llvm.smax.i32"),
            (Ty::U32, "llvm.umin.i32", "llvm.umax.i32"),
        ] {
            for (op, intrinsic) in [(BinOp::Min, min), (BinOp::Max, max)] {
                let ir = emit_llvm_ir(&elementwise_kernel(&ty, op), target).unwrap();
                assert_intrinsic(&ir, target, intrinsic, "i32");
            }
        }
    }
}

/// `usize` is i32 on SPIR-V and i64 on NVPTX and AMDGCN.
#[test]
fn usize_min_max_follow_the_target_pointer_width() {
    for target in targets() {
        let width = if target == Target::SpirvVulkan {
            "i32"
        } else {
            "i64"
        };
        for (op, name) in [(BinOp::Min, "umin"), (BinOp::Max, "umax")] {
            let ir = emit_llvm_ir(&usize_kernel(op), target).unwrap();
            assert_intrinsic(&ir, target, &format!("llvm.{name}.{width}"), width);
        }
    }
}

#[test]
fn float_min_max_name_the_operand_width() {
    for (ty, llvm_ty, suffix) in [(Ty::F32, "float", "f32"), (Ty::F64, "double", "f64")] {
        for target in [
            Target::Nvptx,
            Target::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            for (op, name) in [(BinOp::Min, "minnum"), (BinOp::Max, "maxnum")] {
                let ir = emit_llvm_ir(&elementwise_kernel(&ty, op), target).unwrap();
                assert_intrinsic(&ir, target, &format!("llvm.{name}.{suffix}"), llvm_ty);
            }
        }
    }
}
