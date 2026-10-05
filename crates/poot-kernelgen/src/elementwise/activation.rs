use super::*;

/// Hyperbolic tangent in storage dtype `dt`, computing in f32; the shared [`FusedScalarOp::Tanh`]
/// expansion.
pub fn tanh_dt(name: &str, dt: Ty) -> Body {
    scalar_expansion_dt(name, dt, FusedScalarOp::Tanh)
}

/// Error function in storage dtype `dt`, computing in f32; the shared [`FusedScalarOp::Erf`] expansion.
pub fn erf_dt(name: &str, dt: Ty) -> Body {
    scalar_expansion_dt(name, dt, FusedScalarOp::Erf)
}

/// Elementwise `y[i] = f(x[i])` in storage dtype `dt`, computing in f32 (spec 024): a bf16 input widens,
/// `op`'s scalar expansion runs in f32, the result narrows to `dt`.
fn scalar_expansion_dt(name: &str, dt: Ty, op: FusedScalarOp) -> Body {
    let bf16 = dt != Ty::F32;
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false),
        ld(slice_dtype(dt.clone(), true), true),
    ]);
    let (x, y) = (local(1), local(2));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let r = al.add(Ty::F32, false);
    let xf = if bf16 {
        Some(al.add(Ty::F32, false))
    } else {
        None
    };
    let xsrc = match xf {
        Some(xf) => Place::local(xf),
        None => elem(x, i),
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
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(y))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let mut s2 = Vec::new();
    if let Some(xf) = xf {
        s2.push(Statement::Assign(
            Place::local(xf),
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(elem(x, i)),
            },
        ));
    }
    emit_scalar_op(r, op, &[copy(xsrc)], &mut al, &mut s2);
    s2.push(Statement::Assign(
        elem(y, i),
        if bf16 {
            Rvalue::Cast {
                to: dt.clone(),
                operand: copy(Place::local(r)),
            }
        } else {
            Rvalue::Use(copy(Place::local(r)))
        },
    ));
    let bb2 = BasicBlock {
        statements: s2,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, al.locals, vec![bb0, bb1, bb2, bb3])
}
