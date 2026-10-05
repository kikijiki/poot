//! Complete, valid `Body` fixtures for codegen tests: every local is declared and the body is well-typed.

use crate::*;

fn slice_f32(mutable: bool) -> Ty {
    Ty::Ref {
        mutable,
        pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
    }
}

/// `add(a: Slice<f32>, b: Slice<f32>, c: SliceMut<f32>)`:
/// `let i = thread_index(); if i < c.len() { c[i] = a[i] + b[i]; }`.
pub fn add_kernel() -> Body {
    // locals: _0 ret(unit), _1 a, _2 b, _3 c, _4 i, _5 len, _6 cmp, _7 sum.
    let locals = vec![
        LocalDecl {
            ty: Ty::Unit,
            mutable: false,
        },
        LocalDecl {
            ty: slice_f32(false),
            mutable: false,
        },
        LocalDecl {
            ty: slice_f32(false),
            mutable: false,
        },
        LocalDecl {
            ty: slice_f32(true),
            mutable: true,
        },
        LocalDecl {
            ty: Ty::Usize,
            mutable: false,
        },
        LocalDecl {
            ty: Ty::Usize,
            mutable: false,
        },
        LocalDecl {
            ty: Ty::Bool,
            mutable: false,
        },
        LocalDecl {
            ty: Ty::F32,
            mutable: false,
        },
    ];
    let l = |i| Local { index: i };
    let (a, b, c, i, len, cmp, sum) = (l(1), l(2), l(3), l(4), l(5), l(6), l(7));
    let elem = |p: Local| Place {
        local: p,
        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    Operand::Copy(Place::local(i)),
                    Operand::Copy(Place::local(len)),
                ),
            ),
        ],
        // if cmp == 0 -> bb3 (exit), else (true) -> bb2 (body)
        terminator: Terminator::SwitchInt {
            discr: Operand::Copy(Place::local(cmp)),
            targets: SwitchTargets {
                branches: vec![(0, BlockId { index: 3 })],
                otherwise: BlockId { index: 2 },
            },
        },
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(sum),
                Rvalue::BinaryOp(BinOp::Add, Operand::Copy(elem(a)), Operand::Copy(elem(b))),
            ),
            Statement::Assign(elem(c), Rvalue::Use(Operand::Copy(Place::local(sum)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };

    Body::new("add", 3, locals, vec![bb0, bb1, bb2, bb3])
}
