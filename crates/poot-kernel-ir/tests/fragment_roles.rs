//! Independent Bodies for the existing producer-role contract. No generator or device required.
use poot_kernel_ir::{
    BasicBlock, BlockId, Body, Constant, Local, LocalDecl, Operand, Place, ProjectionElem, Rvalue,
    Site, Statement, SwitchTargets, Terminator, Ty, VerifyErrorKind, WmmaDtype, WmmaMat, WmmaShape,
    WorkgroupLocalDecl,
};

fn l(index: u32) -> Local {
    Local { index }
}

fn bb(index: u32) -> BlockId {
    BlockId { index }
}

fn site(block: u32, index: usize) -> Site {
    Site::Statement {
        block: bb(block),
        index,
    }
}

fn element(param: u32) -> Place {
    Place {
        local: l(param),
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(4))],
    }
}

// Fixed local identities: 1/2 inputs, 3 output, 4 index, 5 A, 6 B, 7 accumulator.
fn tile(dtype: WmmaDtype, lds: bool) -> Body {
    let elem = match dtype {
        WmmaDtype::F16 => Ty::F16,
        WmmaDtype::Bf16 => Ty::BF16,
    };
    let buffer = |ty, mutable| Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(ty))),
    };
    let locals = [
        Ty::Unit,
        buffer(elem.clone(), false),
        buffer(elem.clone(), false),
        buffer(Ty::F32, true),
        Ty::Usize,
        Ty::I32,
        Ty::I32,
        Ty::F32,
    ]
    .into_iter()
    .map(|ty| LocalDecl { ty, mutable: true })
    .collect();
    let shape = WmmaShape::M16N16K16;
    let load = |which, param, dst| {
        if lds {
            Statement::WmmaLoadLds {
                which,
                dtype,
                shape,
                array: 0,
                stride: 16,
                dst: l(dst),
            }
        } else {
            Statement::WmmaLoad {
                which,
                dtype,
                shape,
                tile: element(param),
                stride: 16,
                dst: l(dst),
            }
        }
    };
    let store = if lds {
        Statement::WmmaStoreLds {
            shape,
            array: 1,
            stride: 16,
            src: l(7),
        }
    } else {
        Statement::WmmaStore {
            dtype,
            shape,
            tile: element(3),
            stride: 16,
            src: l(7),
        }
    };
    let mut body = Body::new(
        "fragment_roles",
        3,
        locals,
        vec![BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(l(4)),
                    Rvalue::Use(Operand::Const(Constant::Usize(0))),
                ),
                load(WmmaMat::A, 1, 5),
                load(WmmaMat::B, 2, 6),
                Statement::WmmaZero {
                    dtype,
                    shape,
                    dst: l(7),
                },
                Statement::WmmaMma {
                    dtype,
                    shape,
                    a: l(5),
                    b: l(6),
                    c: l(7),
                    dst: l(7),
                },
                store,
            ],
            terminator: Terminator::Return,
        }],
    );
    body.workgroup_size = [32, 1, 1];
    if lds {
        body.workgroup_locals = vec![
            WorkgroupLocalDecl {
                elem_ty: elem,
                len: 256,
            },
            WorkgroupLocalDecl {
                elem_ty: Ty::F32,
                len: 256,
            },
        ];
    }
    body
}

fn variants() -> impl Iterator<Item = (WmmaDtype, bool)> {
    [WmmaDtype::F16, WmmaDtype::Bf16]
        .into_iter()
        .flat_map(|dtype| [false, true].map(|lds| (dtype, lds)))
}

#[test]
fn swapped_a_b_are_rejected() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
        let Statement::WmmaMma { a, b, .. } = &mut body.blocks[0].statements[4] else {
            unreachable!()
        };
        std::mem::swap(a, b);
        let error = body
            .verify()
            .expect_err("swapped known A/B handles must refuse");
        assert_eq!(error.site, site(0, 4));
        assert_eq!(
            error.kind,
            VerifyErrorKind::FragmentRoleMismatch {
                local: l(6),
                expected_matrix: Some(WmmaMat::A),
                found_matrix: Some(WmmaMat::B),
            }
        );
    }
}

#[test]
fn a_used_as_accumulator_is_rejected() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
        let Statement::WmmaMma { c, .. } = &mut body.blocks[0].statements[4] else {
            unreachable!()
        };
        *c = l(5);
        let error = body.verify().expect_err("A supplied as C must refuse");
        assert_eq!(error.site, site(0, 4));
        assert_eq!(
            error.kind,
            VerifyErrorKind::FragmentRoleMismatch {
                local: l(5),
                expected_matrix: None,
                found_matrix: Some(WmmaMat::A),
            }
        );
    }
}

#[test]
fn matrix_fragments_cannot_be_stored_as_accumulators() {
    for (dtype, lds) in variants() {
        for input in [5, 6] {
            let mut body = tile(dtype, lds);
            match &mut body.blocks[0].statements[5] {
                Statement::WmmaStore { src, .. } | Statement::WmmaStoreLds { src, .. } => {
                    *src = l(input)
                }
                _ => unreachable!(),
            }
            let error = body
                .verify()
                .expect_err("matrix source at accumulator store must refuse");
            assert_eq!(error.site, site(0, 5));
            assert_eq!(
                error.kind,
                VerifyErrorKind::FragmentRoleMismatch {
                    local: l(input),
                    expected_matrix: None,
                    found_matrix: Some(if input == 5 { WmmaMat::A } else { WmmaMat::B }),
                }
            );
        }
    }
}

#[test]
fn conflicting_producer_roles_are_rejected() {
    for (dtype, lds) in variants() {
        let original = tile(dtype, lds);
        for index in [2, 3, 4] {
            let mut body = original.clone();
            let mut conflict = body.blocks[0].statements[index].clone();
            match &mut conflict {
                Statement::WmmaLoad { dst, .. }
                | Statement::WmmaLoadLds { dst, .. }
                | Statement::WmmaZero { dst, .. }
                | Statement::WmmaMma { dst, .. } => *dst = l(5),
                _ => unreachable!(),
            }
            body.blocks[0].statements.push(conflict);
            let error = body
                .verify()
                .expect_err("conflicting producer roles must refuse");
            assert_eq!(error.site, site(0, 6));
            assert_eq!(
                error.kind,
                VerifyErrorKind::FragmentProducerRoleConflict {
                    local: l(5),
                    previous: site(0, 1),
                    previous_matrix: Some(WmmaMat::A),
                    found_matrix: if index == 2 { Some(WmmaMat::B) } else { None },
                }
            );
        }
    }
}

fn producers_after_consumer_block(swapped: bool) -> Body {
    let mut body = tile(WmmaDtype::F16, false);
    let mut statements = body.blocks.remove(0).statements;
    if swapped {
        let Statement::WmmaMma { a, b, .. } = &mut statements[4] else {
            unreachable!()
        };
        std::mem::swap(a, b);
    }
    let consumers = statements.split_off(4);
    body.blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Goto { target: bb(2) },
        },
        BasicBlock {
            statements: consumers,
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements,
            terminator: Terminator::Goto { target: bb(1) },
        },
    ];
    body
}

#[test]
fn known_roles_do_not_depend_on_block_storage_order() {
    producers_after_consumer_block(false).verify().unwrap();
    let error = producers_after_consumer_block(true)
        .verify()
        .expect_err("later-stored producers must still constrain consumers");
    assert_eq!(error.site, site(1, 0));
    assert_eq!(
        error.kind,
        VerifyErrorKind::FragmentRoleMismatch {
            local: l(6),
            expected_matrix: Some(WmmaMat::A),
            found_matrix: Some(WmmaMat::B),
        }
    );
}

#[test]
fn accumulator_cannot_supply_either_matrix_operand() {
    for (dtype, lds) in variants() {
        for which in [WmmaMat::A, WmmaMat::B] {
            let mut body = tile(dtype, lds);
            let Statement::WmmaMma { a, b, .. } = &mut body.blocks[0].statements[4] else {
                unreachable!()
            };
            match which {
                WmmaMat::A => *a = l(7),
                WmmaMat::B => *b = l(7),
            }
            let error = body.verify().unwrap_err();
            assert_eq!(error.site, site(0, 4));
            assert_eq!(
                error.kind,
                VerifyErrorKind::FragmentRoleMismatch {
                    local: l(7),
                    expected_matrix: Some(which),
                    found_matrix: None,
                }
            );
        }
    }
}

#[test]
fn scalar_marker_and_unknown_producer_behavior_is_unchanged() {
    let mut body = tile(WmmaDtype::F16, false);
    // This bounded check only constrains known producers. Scalar assignment is deliberately not
    // interpreted as a fragment declaration or prohibited here; complete typing is separate work.
    body.blocks[0].statements[1] = Statement::Assign(
        Place::local(l(5)),
        Rvalue::Use(Operand::Const(Constant::I32(0))),
    );
    body.verify().unwrap();
}

#[test]
fn valid_global_and_lds_fragments_preserve_accumulator_reuse() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
        body.verify().unwrap();
        // Repeated zero seeds and same-role loads are legal, as are repeated in-place MMAs.
        let seed = body.blocks[0].statements[3].clone();
        body.blocks[0].statements.insert(3, seed);
        let load = body.blocks[0].statements[1].clone();
        body.blocks[0].statements.insert(1, load);
        let mma = body.blocks[0].statements[6].clone();
        assert!(matches!(mma, Statement::WmmaMma { .. }));
        body.blocks[0].statements.insert(7, mma);
        body.verify().unwrap();
    }
}

#[test]
fn loop_carried_accumulator_role_is_preserved() {
    let mut body = tile(WmmaDtype::Bf16, false);
    let statements = body.blocks.remove(0).statements;
    body.blocks = vec![
        BasicBlock {
            statements: statements[..4].to_vec(),
            terminator: Terminator::Goto { target: bb(1) },
        },
        BasicBlock {
            statements: vec![statements[4].clone()],
            terminator: Terminator::SwitchInt {
                discr: Operand::Const(Constant::Bool(false)),
                targets: SwitchTargets {
                    branches: vec![(0, bb(2))],
                    otherwise: bb(1),
                },
            },
        },
        BasicBlock {
            statements: vec![statements[5].clone()],
            terminator: Terminator::Return,
        },
    ];
    body.verify().unwrap();
}

#[test]
fn existing_definition_and_shape_errors_remain_typed() {
    let mut undefined = tile(WmmaDtype::F16, false);
    undefined.blocks[0].statements.remove(2);
    assert_eq!(
        undefined.verify().unwrap_err().kind,
        VerifyErrorKind::UseBeforeDefinition(l(6))
    );
    let mut undeclared = tile(WmmaDtype::F16, false);
    let Statement::WmmaMma { a, .. } = &mut undeclared.blocks[0].statements[4] else {
        unreachable!()
    };
    *a = l(99);
    assert_eq!(
        undeclared.verify().unwrap_err().kind,
        VerifyErrorKind::UndeclaredLocal(l(99))
    );
    let mut shape = tile(WmmaDtype::F16, false);
    let Statement::WmmaMma { shape: size, .. } = &mut shape.blocks[0].statements[4] else {
        unreachable!()
    };
    size.k = 32;
    assert!(matches!(
        shape.verify().unwrap_err().kind,
        VerifyErrorKind::UnsupportedWmmaShape { .. }
    ));
}
