//! Exact-I32 generation: the scalar, compare, shift, remainder and fused bodies keep I32 buffers, literals
//! and arithmetic as I32 (unsigned operations bitcast through U32), and a packed I32 leaf is read lane by
//! lane. These check the emitted `Body` structure; no external tool runs.

use poot_kernel_ir::{BinOp, Body, Constant, Operand, Rvalue, Statement, Ty, UnOp};
use poot_kernelgen::{FusedInput, FusedKernel, FusedScalarOp, FusedStep, LeafId, LeafTy, StepId};

/// The exact-I32 body of one `op` step over a `[4]` leaf, with the literal as its second operand when given: the
/// region every I32 unary, binary and scalar-literal equation lowers to.
fn one_step(op: FusedScalarOp, lit: Option<i32>) -> Body {
    let mut inputs = vec![FusedInput::Leaf(LeafId::new(0))];
    inputs.extend(lit.map(FusedInput::LitI32));
    poot_kernelgen::fused_i32_views_grid(
        "k",
        &[4],
        &[&[4]],
        &[poot_kernelgen::Layout::contiguous(&[4])],
        &FusedKernel {
            leaf_tys: vec![LeafTy::I32],
            steps: vec![FusedStep { op, inputs }],
            output: FusedInput::Step(StepId::new(0)),
        },
        None,
    )
    .expect("an exact-I32 step generates")
}

fn statements(body: &Body) -> Vec<&Statement> {
    body.blocks
        .iter()
        .flat_map(|block| &block.statements)
        .collect()
}

/// Whether `body` assigns `rvalue`-shaped statement matching `pattern` anywhere.
fn any_rvalue(body: &Body, pattern: impl Fn(&Rvalue) -> bool) -> bool {
    statements(body)
        .into_iter()
        .any(|statement| matches!(statement, Statement::Assign(_, rvalue) if pattern(rvalue)))
}

#[test]
fn exact_i32_scalar_body_keeps_i32_buffers_literals_and_arithmetic() {
    let body = one_step(FusedScalarOp::Binary(BinOp::Add), Some(-1));
    assert_eq!(body.param_count, 2);
    for local in &body.locals[1..=2] {
        let Ty::Ref { pointee, .. } = &local.ty else {
            panic!("parameter is not a slice ref: {:?}", local.ty);
        };
        assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
    }
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(BinOp::Add, _, Operand::Const(Constant::I32(-1)))
    )));
}

#[test]
fn exact_i32_unsigned_compare_body_uses_u32_bitcasts() {
    let body = one_step(FusedScalarOp::GeU, Some(i32::MIN));
    // The literal is bitcast to U32 like the value, so both sides compare as unsigned words.
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: Operand::Const(Constant::I32(i32::MIN)),
        }
    )));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: Operand::Copy(_),
        }
    )));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(BinOp::Ge, Operand::Copy(_), Operand::Copy(_))
    )));
}

#[test]
fn exact_i32_unary_not_and_clz_keep_i32_storage() {
    for body in [
        one_step(FusedScalarOp::Unary(UnOp::Not), None),
        one_step(FusedScalarOp::Clz, None),
    ] {
        for local in &body.locals[1..=2] {
            let Ty::Ref { pointee, .. } = &local.ty else {
                panic!("unary parameter is not a slice ref: {:?}", local.ty);
            };
            assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
        }
        assert!(!any_rvalue(&body, |rvalue| matches!(
            rvalue,
            Rvalue::Cast { to: Ty::F32, .. }
        )));
    }
}

#[test]
fn exact_i32_shift_masks_the_amount() {
    let body = one_step(FusedScalarOp::Binary(BinOp::Shl), Some(33));
    // The amount is masked with 31 before the shift, so a shift by 33 is a shift by 1.
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(
            BinOp::BitAnd,
            Operand::Const(Constant::I32(33)),
            Operand::Const(Constant::I32(31))
        )
    )));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(BinOp::Shl, _, Operand::Copy(_))
    )));
}

#[test]
fn exact_i32_shr_bitcasts_through_u32_for_logical_shift() {
    let body = one_step(FusedScalarOp::Binary(BinOp::Shr), Some(1));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::Bitcast { to: Ty::U32, .. }
    )));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(BinOp::Shr, Operand::Copy(_), Operand::Copy(_))
    )));
}

#[test]
fn exact_i32_fused_body_keeps_i32_locals_and_literals() {
    let body = poot_kernelgen::fused_i32_views_grid(
        "k",
        &[2],
        &[&[2], &[2]],
        &[
            poot_kernelgen::Layout::contiguous(&[2]),
            poot_kernelgen::Layout::contiguous(&[2]),
        ],
        &poot_kernelgen::FusedKernel {
            leaf_tys: vec![poot_kernelgen::LeafTy::I32; 2],
            steps: vec![
                poot_kernelgen::FusedStep {
                    op: poot_kernelgen::FusedScalarOp::Binary(BinOp::BitAnd),
                    inputs: vec![
                        poot_kernelgen::FusedInput::Leaf(poot_kernelgen::LeafId::new(0)),
                        poot_kernelgen::FusedInput::LitI32(-1),
                    ],
                },
                poot_kernelgen::FusedStep {
                    op: poot_kernelgen::FusedScalarOp::Clz,
                    inputs: vec![poot_kernelgen::FusedInput::Step(
                        poot_kernelgen::StepId::new(0),
                    )],
                },
            ],
            output: poot_kernelgen::FusedInput::Step(poot_kernelgen::StepId::new(1)),
        },
        None,
    )
    .expect("fused_i32_views precondition");
    for local in &body.locals[1..=3] {
        let Ty::Ref { pointee, .. } = &local.ty else {
            panic!("parameter is not a slice ref: {:?}", local.ty);
        };
        assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
    }
    let statements: Vec<_> = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .collect();
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::IntScalarUnary(poot_kernel_ir::IntScalarOp::LeadingZeros, _)
            )
        )
    }));
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(
                    BinOp::BitAnd,
                    _,
                    poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32(-1))
                )
            )
        )
    }));
}

#[test]
fn packed_i32_fused_kernel_guards_with_output_len_over_lanes() {
    let body = poot_kernelgen::fused_i32_views_grid_pack(
        "k",
        &[0, 2],
        &[&[0, 2]],
        &[poot_kernelgen::Layout::contiguous(&[0, 2])],
        &poot_kernelgen::FusedKernel {
            leaf_tys: vec![poot_kernelgen::LeafTy::I32; 1],
            steps: vec![],
            output: poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(0),
            },
        },
        None,
        &[
            poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(0),
            },
            poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(1),
            },
        ],
    )
    .expect("fused_i32_views_grid_pack precondition");
    let guard = &body.blocks[1].statements;
    assert!(
        guard
            .iter()
            .any(|statement| matches!(statement, Statement::Assign(_, Rvalue::Len(_)))),
        "packed fused kernels must read the output length"
    );
    assert!(
        guard.iter().any(|statement| matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(
                    BinOp::Div,
                    _,
                    poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::Usize(2))
                )
            )
        )),
        "packed fused kernels must divide length by pack width"
    );
    assert!(
        !guard.iter().any(|statement| matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::Use(poot_kernel_ir::Operand::Const(
                    poot_kernel_ir::Constant::Usize(1)
                ))
            )
        )),
        "empty packed prefix must not bake a one-thread loop bound"
    );
}

#[test]
fn packed_i32_leaf_lane_broadcasts_a_unit_prefix() {
    let body = poot_kernelgen::fused_i32_views_grid_pack(
        "k",
        &[4, 2],
        &[&[1, 2]],
        &[poot_kernelgen::Layout::contiguous(&[1, 2])],
        &poot_kernelgen::FusedKernel {
            leaf_tys: vec![poot_kernelgen::LeafTy::I32; 1],
            steps: vec![],
            output: poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(0),
            },
        },
        None,
        &[
            poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(0),
            },
            poot_kernelgen::FusedInput::LeafLane {
                leaf: poot_kernelgen::LeafId::new(0),
                lane: poot_kernelgen::LaneId::new(1),
            },
        ],
    )
    .expect("fused_i32_views_grid_pack precondition");
    let load = &body.blocks[2].statements;
    let scaled = load
        .iter()
        .filter(|statement| {
            matches!(
                statement,
                Statement::Assign(
                    _,
                    Rvalue::BinaryOp(
                        BinOp::Mul,
                        _,
                        poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::Usize(2))
                    )
                )
            )
        })
        .count();
    assert_eq!(
        scaled, 2,
        "broadcast packed leaves must not scale i by pack width; only packed stores do"
    );
}

#[test]
fn exact_i32_unsigned_remainder_bitcasts_through_u32() {
    let body = one_step(FusedScalarOp::RemU, Some(0x7fff_ffff));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: Operand::Const(Constant::I32(0x7fff_ffff)),
        }
    )));
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::BinaryOp(BinOp::Rem, Operand::Copy(_), Operand::Copy(_))
    )));
    // The u32 result is bitcast back to the I32 storage pattern.
    assert!(any_rvalue(&body, |rvalue| matches!(
        rvalue,
        Rvalue::Bitcast { to: Ty::I32, .. }
    )));
}
