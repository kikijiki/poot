//! `Rvalue::SubgroupShuffleXor` is subgroup-collective like a matrix-fragment op: every lane of the aligned
//! 32-lane run executes it together or none does, and its mask stays inside that run. These Bodies drive the
//! verifier with independent control flow; no generator, interpreter or device is involved.
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, LocalDecl, Operand, Place,
    Rvalue, Statement, SwitchTargets, Terminator, Ty, VerifyErrorKind,
};

fn l(index: u32) -> Local {
    Local { index }
}

fn bb(index: u32) -> BlockId {
    BlockId { index }
}

fn usize_const(value: u64) -> Operand {
    Operand::Const(Constant::Usize(value))
}

fn copy(local: u32) -> Operand {
    Operand::Copy(Place::local(l(local)))
}

fn assign(local: u32, rvalue: Rvalue) -> Statement {
    Statement::Assign(Place::local(l(local)), rvalue)
}

fn binary(local: u32, op: BinOp, lhs: Operand, rhs: Operand) -> Statement {
    assign(local, Rvalue::BinaryOp(op, lhs, rhs))
}

const LANE: u32 = 1;
const SELECTOR: u32 = 2;
const COND: u32 = 3;
const VALUE: u32 = 4;
const RECEIVED: u32 = 5;

/// bb0 reads `LocalX`; bb1 computes `selector` and branches on it; bb2 shuffles `value_ty` by `mask` (taken
/// when the condition is true); bb3 returns. The shuffle's operand is local 4, typed `value_ty`.
fn guarded(selector: Vec<Statement>, mask: u32, value_ty: Ty) -> Body {
    let locals = [
        Ty::Unit,
        Ty::Usize,
        Ty::Usize,
        Ty::Bool,
        value_ty.clone(),
        value_ty,
    ]
    .into_iter()
    .map(|ty| LocalDecl { ty, mutable: true })
    .collect();
    let mut body = Body::new(
        "shuffle",
        0,
        locals,
        vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(l(LANE)),
                    dim: IndexAxis::LocalX,
                    target: bb(1),
                },
            },
            BasicBlock {
                statements: selector,
                terminator: Terminator::SwitchInt {
                    discr: copy(COND),
                    targets: SwitchTargets {
                        branches: vec![(0, bb(3))],
                        otherwise: bb(2),
                    },
                },
            },
            BasicBlock {
                statements: vec![assign(
                    RECEIVED,
                    Rvalue::SubgroupShuffleXor {
                        value: copy(VALUE),
                        mask,
                    },
                )],
                terminator: Terminator::Return,
            },
            BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            },
        ],
    );
    // The shuffle's operand is never defined by a branch-free path; seed it in the entry block.
    body.blocks[0].statements.push(assign(
        VALUE,
        match &body.locals[VALUE as usize].ty {
            Ty::F32 => Rvalue::Use(Operand::Const(Constant::F32(1.0))),
            Ty::U32 => Rvalue::Use(Operand::Const(Constant::U32(1))),
            Ty::I32 => Rvalue::Use(Operand::Const(Constant::I32(1))),
            Ty::Bool => Rvalue::Use(Operand::Const(Constant::Bool(true))),
            other => panic!("no seed for {other:?}"),
        },
    ));
    body.workgroup_size = [64, 1, 1];
    body
}

/// `LocalX % 3 == 0`: a third of the lanes of every subgroup.
fn every_third_lane() -> Vec<Statement> {
    vec![
        binary(SELECTOR, BinOp::Rem, copy(LANE), usize_const(3)),
        binary(COND, BinOp::Eq, copy(SELECTOR), usize_const(0)),
    ]
}

/// `LocalX / 32 == 0`: subgroup 0 of the workgroup.
fn first_subgroup() -> Vec<Statement> {
    vec![
        binary(SELECTOR, BinOp::Div, copy(LANE), usize_const(32)),
        binary(COND, BinOp::Eq, copy(SELECTOR), usize_const(0)),
    ]
}

#[test]
fn a_shuffle_under_a_branch_that_splits_a_subgroup_is_rejected_and_one_under_a_whole_subgroup_is_accepted()
 {
    let divergent = guarded(every_third_lane(), 1, Ty::U32)
        .verify()
        .expect_err("a shuffle under `LocalX % 3` must be rejected")
        .kind;
    assert_eq!(
        divergent,
        VerifyErrorKind::DivergentCollectiveParticipation {
            branch: bb(1),
            op: bb(2),
            // Both arms end in their own `Return`: they only meet at the body's exit.
            reconverge: None,
        }
    );
    guarded(first_subgroup(), 1, Ty::U32).verify().unwrap();
}

#[test]
fn a_shuffle_mask_below_the_collective_width_is_accepted_and_the_width_itself_is_rejected() {
    guarded(first_subgroup(), 31, Ty::U32).verify().unwrap();
    let error = guarded(first_subgroup(), 32, Ty::U32)
        .verify()
        .expect_err("mask 32 leaves the aligned run");
    assert_eq!(
        error.kind,
        VerifyErrorKind::ShuffleMaskOutOfRange { mask: 32 }
    );
    assert_eq!(
        error.to_string(),
        "invalid kernel body at bb2 statement 0: subgroup shuffle mask 32 is not below 32"
    );
}

#[test]
fn a_shuffle_takes_f32_i32_and_u32_and_nothing_else() {
    for ty in [Ty::F32, Ty::I32, Ty::U32] {
        guarded(first_subgroup(), 1, ty).verify().unwrap();
    }
    let error = guarded(first_subgroup(), 1, Ty::Bool)
        .verify()
        .expect_err("a bool is not a shuffled scalar");
    assert!(
        matches!(error.kind, VerifyErrorKind::InvalidOperandType { .. }),
        "{error}"
    );
}
