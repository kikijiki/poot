//! Independent Bodies for the declared fragment-type contract (role, operand dtype and tile are part of each
//! fragment local's `Ty::Fragment`). No generator or device required.
use poot_kernel_ir::{
    BasicBlock, BlockId, Body, Constant, FragmentTy, Local, LocalDecl, Operand, Place,
    ProjectionElem, Rvalue, Site, Statement, SwitchTargets, Terminator, Ty, VerifyErrorKind,
    WmmaDtype, WmmaMat, WmmaShape, WorkgroupLocalDecl,
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
        Ty::Fragment(FragmentTy::matrix(WmmaMat::A, dtype, WmmaShape::M16N16K16)),
        Ty::Fragment(FragmentTy::matrix(WmmaMat::B, dtype, WmmaShape::M16N16K16)),
        Ty::Fragment(FragmentTy::accumulator(dtype, WmmaShape::M16N16K16)),
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
            dtype,
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

fn shape() -> WmmaShape {
    WmmaShape::M16N16K16
}

fn mismatch(local: u32, expected: FragmentTy, found: impl Into<Ty>) -> VerifyErrorKind {
    VerifyErrorKind::FragmentTypeMismatch {
        local: l(local),
        expected,
        found: found.into(),
    }
}

#[test]
fn valid_global_and_lds_fragments_verify() {
    for (dtype, lds) in variants() {
        tile(dtype, lds).verify().unwrap();
    }
}

#[test]
fn swapped_a_b_are_rejected() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
        let Statement::WmmaMma { a, b, .. } = &mut body.blocks[0].statements[4] else {
            unreachable!()
        };
        std::mem::swap(a, b);
        let error = body.verify().expect_err("swapped A/B handles must refuse");
        assert_eq!(error.site, site(0, 4));
        assert_eq!(
            error.kind,
            mismatch(
                6,
                FragmentTy::matrix(WmmaMat::A, dtype, shape()),
                FragmentTy::matrix(WmmaMat::B, dtype, shape())
            )
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
            mismatch(
                5,
                FragmentTy::accumulator(dtype, shape()),
                FragmentTy::matrix(WmmaMat::A, dtype, shape())
            )
        );
    }
}

#[test]
fn matrix_fragments_cannot_be_stored_as_accumulators() {
    for (dtype, lds) in variants() {
        for (input, which) in [(5, WmmaMat::A), (6, WmmaMat::B)] {
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
                mismatch(
                    input,
                    FragmentTy::accumulator(dtype, shape()),
                    FragmentTy::matrix(which, dtype, shape())
                )
            );
        }
    }
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
                mismatch(
                    7,
                    FragmentTy::matrix(which, dtype, shape()),
                    FragmentTy::accumulator(dtype, shape())
                )
            );
        }
    }
}

/// A producer whose destination is declared with another role is refused at the producer, wherever the
/// consumers sit.
#[test]
fn producer_destination_must_be_declared_with_its_role() {
    for (dtype, lds) in variants() {
        // Statement 2 loads B into local 6; point it at A's local 5 instead.
        let mut body = tile(dtype, lds);
        match &mut body.blocks[0].statements[2] {
            Statement::WmmaLoad { dst, .. } | Statement::WmmaLoadLds { dst, .. } => *dst = l(5),
            _ => unreachable!(),
        }
        let error = body
            .verify()
            .expect_err("B load into an A local must refuse");
        assert_eq!(error.site, site(0, 2));
        assert_eq!(
            error.kind,
            mismatch(
                5,
                FragmentTy::matrix(WmmaMat::B, dtype, shape()),
                FragmentTy::matrix(WmmaMat::A, dtype, shape())
            )
        );
        // Statement 3 seeds the accumulator; point the zero seed at the B local.
        let mut body = tile(dtype, lds);
        let Statement::WmmaZero { dst, .. } = &mut body.blocks[0].statements[3] else {
            unreachable!()
        };
        *dst = l(6);
        let error = body
            .verify()
            .expect_err("zero seed into a B local must refuse");
        assert_eq!(error.site, site(0, 3));
    }
}

#[test]
fn operation_dtype_must_match_the_declared_fragment_dtype() {
    for lds in [false, true] {
        let mut body = tile(WmmaDtype::F16, lds);
        let Statement::WmmaMma { dtype, .. } = &mut body.blocks[0].statements[4] else {
            unreachable!()
        };
        *dtype = WmmaDtype::Bf16;
        let error = body.verify().expect_err("dtype disagreement must refuse");
        assert_eq!(error.site, site(0, 4));
        assert_eq!(
            error.kind,
            mismatch(
                5,
                FragmentTy::matrix(WmmaMat::A, WmmaDtype::Bf16, shape()),
                FragmentTy::matrix(WmmaMat::A, WmmaDtype::F16, shape())
            )
        );
    }
}

#[test]
fn a_scalar_declared_fragment_local_is_rejected() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
        body.locals[7].ty = Ty::F32;
        let error = body
            .verify()
            .expect_err("scalar-typed accumulator must refuse");
        assert_eq!(error.site, site(0, 3));
        assert_eq!(
            error.kind,
            mismatch(7, FragmentTy::accumulator(dtype, shape()), Ty::F32)
        );
    }
}

#[test]
fn fragments_cannot_be_assigned_like_scalars() {
    let mut body = tile(WmmaDtype::F16, false);
    body.blocks[0].statements.insert(
        4,
        Statement::Assign(
            Place::local(l(5)),
            Rvalue::Use(Operand::Copy(Place::local(l(5)))),
        ),
    );
    let error = body.verify().expect_err("fragment assignment must refuse");
    assert_eq!(error.site, site(0, 4));
    assert!(matches!(
        error.kind,
        VerifyErrorKind::InvalidOperandType {
            what: "assignment destination",
            ..
        }
    ));
}

#[test]
fn repeated_seeds_loads_and_in_place_mma_stay_legal() {
    for (dtype, lds) in variants() {
        let mut body = tile(dtype, lds);
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
fn loop_carried_accumulator_verifies() {
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
    let mut bad_shape = tile(WmmaDtype::F16, false);
    let Statement::WmmaMma { shape: size, .. } = &mut bad_shape.blocks[0].statements[4] else {
        unreachable!()
    };
    size.k = 32;
    assert!(matches!(
        bad_shape.verify().unwrap_err().kind,
        VerifyErrorKind::UnsupportedWmmaShape { .. }
    ));
}
