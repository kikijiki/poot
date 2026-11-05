//! POOT-1460: a float `Ne` lowers to the unordered `fcmp une` on every target, so `NaN != x` is true as it is in
//! the kernel interpreter (IEEE 754). `fcmp one` is ordered and false for a `NaN`; SPIR-V's translation of it is
//! `FOrdNotEqual`. The ordered comparisons keep their ordered predicates (`NaN == x` and `NaN < x` are false).
//! These tests read the emitted IR, so they need no toolchain.

use poot_codegen::{Target, emit_llvm_ir};
use poot_kernel_ir::BinOp;
use poot_kernel_ir::test_support::float_ne_kernel;
use poot_kernel_ir::{Rvalue, Statement};

fn targets() -> [Target; 3] {
    [
        Target::SpirvVulkan,
        Target::Nvptx,
        Target::AmdGcn(poot_target::AmdArch::gfx1151()),
    ]
}

#[test]
fn float_ne_lowers_to_the_unordered_compare() {
    for target in targets() {
        let ir = emit_llvm_ir(&float_ne_kernel(), target).unwrap();
        assert!(
            ir.contains("fcmp une float"),
            "{target:?}: float Ne must be `fcmp une` (true for NaN):\n{ir}"
        );
        assert!(
            !ir.contains("fcmp one"),
            "{target:?}: `fcmp one` is false for NaN:\n{ir}"
        );
    }
}

/// The same kernel with `Eq`: the equality stays ordered, so the fix is not a blanket change of predicates.
#[test]
fn float_eq_stays_ordered() {
    let mut body = float_ne_kernel();
    let mut swapped = 0;
    for statement in body.blocks.iter_mut().flat_map(|b| &mut b.statements) {
        if let Statement::Assign(_, Rvalue::BinaryOp(op @ BinOp::Ne, _, _)) = statement {
            *op = BinOp::Eq;
            swapped += 1;
        }
    }
    assert_eq!(swapped, 1, "the fixture holds one float Ne");
    for target in targets() {
        let ir = emit_llvm_ir(&body, target).unwrap();
        assert!(
            ir.contains("fcmp oeq float"),
            "{target:?}: float Eq must be `fcmp oeq`:\n{ir}"
        );
    }
}
