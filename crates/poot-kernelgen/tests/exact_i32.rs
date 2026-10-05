use poot_kernel_ir::{BinOp, Rvalue, Statement, Ty};

#[test]
fn exact_i32_scalar_body_keeps_i32_buffers_literals_and_arithmetic() {
    let body = poot_kernelgen::binary_scalar_i32_grid("k", BinOp::Add, -1, None);
    assert_eq!(body.param_count, 2);
    for local in &body.locals[1..=2] {
        let Ty::Ref { pointee, .. } = &local.ty else {
            panic!("parameter is not a slice ref: {:?}", local.ty);
        };
        assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
    }
    assert!(
        body.blocks
            .iter()
            .flat_map(|block| &block.statements)
            .any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(
                        _,
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            _,
                            poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32(-1))
                        )
                    )
                )
            })
    );
}

#[test]
fn exact_i32_unsigned_compare_body_uses_u32_bitcasts() {
    let body = poot_kernelgen::binary_scalar_i32_geu_grid("k", i32::MIN, None);
    let statements: Vec<_> = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .collect();
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(_, Rvalue::Bitcast { to: Ty::U32, .. })
        )
    }));
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(
                    BinOp::Ge,
                    _,
                    poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::U32(0x8000_0000))
                )
            )
        )
    }));
}

#[test]
fn exact_i32_unary_not_and_clz_keep_i32_storage() {
    for body in [
        poot_kernelgen::unary_dt_grid("k", Ty::I32, poot_kernel_ir::UnOp::Not, None),
        poot_kernelgen::unary_i32_clz_grid("k", None),
    ] {
        for local in &body.locals[1..=2] {
            let Ty::Ref { pointee, .. } = &local.ty else {
                panic!("unary parameter is not a slice ref: {:?}", local.ty);
            };
            assert_eq!(**pointee, Ty::Slice(Box::new(Ty::I32)));
        }
        assert!(
            !body
                .blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|statement| matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Cast { to: Ty::F32, .. })
                ))
        );
    }
}

#[test]
fn exact_i32_shift_masks_the_amount() {
    let body = poot_kernelgen::binary_scalar_i32_grid("k", BinOp::Shl, 33, None);
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
                Rvalue::BinaryOp(
                    BinOp::Shl,
                    _,
                    poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32(1))
                )
            )
        )
    }));
}

#[test]
fn exact_i32_shr_bitcasts_through_u32_for_logical_shift() {
    let body = poot_kernelgen::binary_scalar_i32_grid("k", BinOp::Shr, 1, None);
    let statements: Vec<_> = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .collect();
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(_, Rvalue::Bitcast { to: Ty::U32, .. })
        )
    }));
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(BinOp::Shr, _, poot_kernel_ir::Operand::Copy(_))
            )
        )
    }));
}

#[test]
fn exact_i32_fused_body_keeps_i32_locals_and_literals() {
    let body = poot_test_util::kernel_fixtures::fused_i32_views(
        "k",
        &[2],
        &[&[2], &[2]],
        &[
            poot_kernelgen::Layout::contiguous(&[2]),
            poot_kernelgen::Layout::contiguous(&[2]),
        ],
        &poot_kernelgen::FusedKernel {
            n_leaves: 2,
            steps: vec![
                poot_kernelgen::FusedStep {
                    op: poot_kernelgen::FusedScalarOp::Binary(BinOp::BitAnd),
                    inputs: vec![
                        poot_kernelgen::FusedInput::Leaf(0),
                        poot_kernelgen::FusedInput::LitI32(-1),
                    ],
                },
                poot_kernelgen::FusedStep {
                    op: poot_kernelgen::FusedScalarOp::Clz,
                    inputs: vec![poot_kernelgen::FusedInput::Step(0)],
                },
            ],
            output: poot_kernelgen::FusedInput::Step(1),
        },
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
            n_leaves: 1,
            steps: vec![],
            output: poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 0 },
        },
        None,
        &[
            poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 0 },
            poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 1 },
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
            n_leaves: 1,
            steps: vec![],
            output: poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 0 },
        },
        None,
        &[
            poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 0 },
            poot_kernelgen::FusedInput::LeafLane { leaf: 0, lane: 1 },
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
    let body = poot_kernelgen::binary_scalar_i32_remu_grid("k", 0x7fff_ffff, None);
    let statements: Vec<_> = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .collect();
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(_, Rvalue::Bitcast { to: Ty::U32, .. })
        )
    }));
    assert!(statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(
                    BinOp::Rem,
                    _,
                    poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::U32(0x7fff_ffff))
                )
            )
        )
    }));
    assert!(!statements.iter().any(|statement| {
        matches!(
            statement,
            Statement::Assign(_, Rvalue::Cast { to: Ty::F32, .. })
        )
    }));
}
