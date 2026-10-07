//! Matrix-fragment ops are subgroup-collective: every lane of the 32-lane group executes one together or none
//! does. These Bodies drive the verifier's collective-participation check with independent control flow; no
//! generator or device is involved.
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, FragmentTy, IndexAxis, Local, LocalDecl, Operand,
    Place, ProjectionElem, Rvalue, Statement, SwitchTargets, Terminator, Ty, VerifyErrorKind,
    WmmaDtype, WmmaMat, WmmaShape,
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

fn element(param: u32) -> Place {
    Place {
        local: l(param),
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(4))],
    }
}

fn buffer(elem: Ty, mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(elem))),
    }
}

const SHAPE: WmmaShape = WmmaShape::M16N16K16;
const DTYPE: WmmaDtype = WmmaDtype::F16;
/// The index every selector reads: the thread index.
const LANE: u32 = 8;
const SELECTOR: u32 = 9;
const COND: u32 = 10;

/// Params 1/2 f16 inputs, 3 f32 output; 4 tile index; 5/6/7 the A/B/accumulator fragments; 8 the thread index;
/// 9 a selector temporary; 10 the branch condition. `dim` is the thread-index axis written to local 8.
fn locals() -> Vec<LocalDecl> {
    [
        Ty::Unit,
        buffer(Ty::F16, false),
        buffer(Ty::F16, false),
        buffer(Ty::F32, true),
        Ty::Usize,
        Ty::Fragment(FragmentTy::matrix(WmmaMat::A, DTYPE, SHAPE)),
        Ty::Fragment(FragmentTy::matrix(WmmaMat::B, DTYPE, SHAPE)),
        Ty::Fragment(FragmentTy::accumulator(DTYPE, SHAPE)),
        Ty::Usize,
        Ty::Usize,
        Ty::Bool,
    ]
    .into_iter()
    .map(|ty| LocalDecl { ty, mutable: true })
    .collect()
}

/// One complete matrix tile: load A and B, seed the accumulator, multiply, store.
fn matrix_work() -> Vec<Statement> {
    let load = |which, param, dst| Statement::WmmaLoad {
        which,
        dtype: DTYPE,
        shape: SHAPE,
        tile: element(param),
        stride: 16,
        dst: l(dst),
    };
    vec![
        load(WmmaMat::A, 1, 5),
        load(WmmaMat::B, 2, 6),
        Statement::WmmaZero {
            dtype: DTYPE,
            shape: SHAPE,
            dst: l(7),
        },
        Statement::WmmaMma {
            dtype: DTYPE,
            shape: SHAPE,
            a: l(5),
            b: l(6),
            c: l(7),
            dst: l(7),
        },
        Statement::WmmaStore {
            dtype: DTYPE,
            shape: SHAPE,
            tile: element(3),
            stride: 16,
            src: l(7),
        },
    ]
}

fn branch_on_cond(then_block: u32, else_block: u32) -> Terminator {
    Terminator::SwitchInt {
        discr: copy(COND),
        targets: SwitchTargets {
            branches: vec![(0, bb(else_block))],
            otherwise: bb(then_block),
        },
    }
}

/// bb0 reads the thread index along `dim`; bb1 computes `selector` and branches on it; bb2 holds the matrix
/// work (taken when the condition is true); bb3 returns.
fn guarded(dim: IndexAxis, workgroup: [u32; 3], selector: Vec<Statement>) -> Body {
    let mut body = Body::new(
        "collective",
        3,
        locals(),
        vec![
            BasicBlock {
                statements: vec![assign(4, Rvalue::Use(usize_const(0)))],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(l(LANE)),
                    dim,
                    target: bb(1),
                },
            },
            BasicBlock {
                statements: selector,
                terminator: branch_on_cond(2, 3),
            },
            BasicBlock {
                statements: matrix_work(),
                terminator: Terminator::Return,
            },
            BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            },
        ],
    );
    body.workgroup_size = workgroup;
    body
}

fn lane() -> Operand {
    copy(LANE)
}

fn selector() -> Operand {
    copy(SELECTOR)
}

/// `(lane % 256) < 32`: the warp-0 selection `matmul_tensorcore` uses.
fn warp_zero_by_remainder() -> Vec<Statement> {
    vec![
        binary(SELECTOR, BinOp::Rem, lane(), usize_const(256)),
        binary(COND, BinOp::Lt, selector(), usize_const(32)),
    ]
}

/// `lane / 32 == 0`.
fn warp_zero_by_quotient() -> Vec<Statement> {
    vec![
        binary(SELECTOR, BinOp::Div, lane(), usize_const(32)),
        binary(COND, BinOp::Eq, selector(), usize_const(0)),
    ]
}

fn divergent_error(body: &Body) -> VerifyErrorKind {
    body.verify()
        .expect_err("matrix work under a lane-divergent branch must be rejected")
        .kind
}

#[test]
fn whole_subgroup_selection_is_accepted() {
    for selector in [warp_zero_by_remainder(), warp_zero_by_quotient()] {
        guarded(IndexAxis::X, [256, 1, 1], selector)
            .verify()
            .unwrap();
    }
    // The dispatch id and the local id are the same lane index at a workgroup offset that is a multiple of 32.
    guarded(IndexAxis::LocalX, [64, 1, 1], warp_zero_by_remainder())
        .verify()
        .unwrap();
}

#[test]
fn a_workgroup_uniform_condition_is_accepted() {
    let selector = vec![binary(COND, BinOp::Eq, lane(), usize_const(0))];
    guarded(IndexAxis::GroupX, [256, 1, 1], selector)
        .verify()
        .unwrap();
}

#[test]
fn a_selector_constant_per_row_is_accepted_when_rows_are_whole_subgroups() {
    let selector = vec![binary(COND, BinOp::Eq, lane(), usize_const(0))];
    guarded(IndexAxis::LocalY, [32, 4, 1], selector)
        .verify()
        .unwrap();
}

#[test]
fn a_lane_prefix_shorter_than_a_subgroup_is_rejected() {
    let selector = vec![binary(COND, BinOp::Lt, lane(), usize_const(16))];
    let body = guarded(IndexAxis::X, [256, 1, 1], selector);
    assert_eq!(
        divergent_error(&body),
        VerifyErrorKind::DivergentFragmentParticipation {
            branch: bb(1),
            op: bb(2),
            // Both arms end in their own `Return`: they only meet at the body's exit.
            reconverge: None,
        }
    );
}

#[test]
fn even_lanes_only_is_rejected() {
    let selector = vec![
        binary(SELECTOR, BinOp::Rem, lane(), usize_const(2)),
        binary(COND, BinOp::Eq, selector(), usize_const(0)),
    ];
    let body = guarded(IndexAxis::X, [256, 1, 1], selector);
    assert!(matches!(
        divergent_error(&body),
        VerifyErrorKind::DivergentFragmentParticipation { .. }
    ));
}

#[test]
fn a_lane_dependent_modulus_that_is_not_a_subgroup_multiple_is_rejected() {
    // lane % 48 == lane % 16 inside each 48-lane stretch: subgroups straddle it.
    let selector = vec![
        binary(SELECTOR, BinOp::Rem, lane(), usize_const(48)),
        binary(COND, BinOp::Lt, selector(), usize_const(32)),
    ];
    let body = guarded(IndexAxis::X, [96, 1, 1], selector);
    assert!(matches!(
        divergent_error(&body),
        VerifyErrorKind::DivergentFragmentParticipation { .. }
    ));
}

/// A workgroup whose X extent is not a multiple of 32 has subgroups that straddle rows, so even the warp-zero
/// selection `lane < 32` can split one.
#[test]
fn subgroups_that_straddle_workgroup_rows_are_rejected() {
    let body = guarded(
        IndexAxis::X,
        [48, 2, 1],
        vec![binary(COND, BinOp::Lt, lane(), usize_const(32))],
    );
    assert!(matches!(
        divergent_error(&body),
        VerifyErrorKind::DivergentFragmentParticipation { .. }
    ));
}

#[test]
fn a_lane_varying_loop_around_matrix_work_is_rejected() {
    // bb1: i = 0 -> bb2; bb2 (header): cond = i < lane (varying trip count); bb3 (body): matrix work, i += 1.
    let mut body = guarded(IndexAxis::X, [256, 1, 1], vec![]);
    body.blocks = vec![
        BasicBlock {
            statements: vec![assign(4, Rvalue::Use(usize_const(0)))],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(l(LANE)),
                dim: IndexAxis::X,
                target: bb(1),
            },
        },
        BasicBlock {
            statements: vec![assign(SELECTOR, Rvalue::Use(usize_const(0)))],
            terminator: Terminator::Goto { target: bb(2) },
        },
        BasicBlock {
            statements: vec![binary(COND, BinOp::Lt, selector(), lane())],
            terminator: branch_on_cond(3, 4),
        },
        BasicBlock {
            statements: {
                let mut work = matrix_work();
                work.push(binary(SELECTOR, BinOp::Add, selector(), usize_const(1)));
                work
            },
            terminator: Terminator::Goto { target: bb(2) },
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    assert_eq!(
        divergent_error(&body),
        VerifyErrorKind::DivergentFragmentParticipation {
            branch: bb(2),
            op: bb(3),
            reconverge: Some(bb(4)),
        }
    );
}

/// A divergent branch whose arms rejoin before any matrix op does not split the collective: the matrix work
/// sits at the reconvergence point, where every lane is back.
#[test]
fn matrix_work_after_the_arms_reconverge_is_accepted() {
    let mut body = guarded(IndexAxis::X, [256, 1, 1], vec![]);
    body.blocks = vec![
        BasicBlock {
            statements: vec![assign(4, Rvalue::Use(usize_const(0)))],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(l(LANE)),
                dim: IndexAxis::X,
                target: bb(1),
            },
        },
        BasicBlock {
            statements: vec![binary(COND, BinOp::Lt, lane(), usize_const(16))],
            terminator: branch_on_cond(2, 3),
        },
        BasicBlock {
            statements: vec![assign(SELECTOR, Rvalue::Use(usize_const(1)))],
            terminator: Terminator::Goto { target: bb(4) },
        },
        BasicBlock {
            statements: vec![assign(SELECTOR, Rvalue::Use(usize_const(2)))],
            terminator: Terminator::Goto { target: bb(4) },
        },
        BasicBlock {
            statements: matrix_work(),
            terminator: Terminator::Return,
        },
    ];
    body.verify().unwrap();
}
