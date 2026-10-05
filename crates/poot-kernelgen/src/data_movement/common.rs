use super::*;

/// Shared skeleton for a one-input index-remap copy kernel: `out[i] = in[src(i)]`, where `src` is built
/// from the output coords by `src_terms` (each `(out_dim, stride)` contributes `coord(out_dim)*stride`)
/// plus a constant `src_base`. Covers broadcast / transpose / slice.
pub(crate) fn index_remap_copy(
    name: &str,
    dt: Ty,
    out_strides: &[usize],
    out_shape: &[usize],
    src_terms: &[(usize, usize)],
    src_base: usize,
) -> Body {
    // Pure element copy (transpose/slice/broadcast): the value carries the storage dtype, no widen/narrow (spec 024).
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(dt.clone(), false), false), // 1 in
        ld(slice_dtype(dt.clone(), true), true),   // 2 out
    ]);
    let (inp, out) = (local(1), local(2));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);
    let src = al.add(Ty::Usize, true);
    let div = al.add(Ty::Usize, false);
    let md = al.add(Ty::Usize, false);
    let term = al.add(Ty::Usize, false);
    let src_new = al.add(Ty::Usize, false);
    let val = al.add(dt.clone(), false);
    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let mut body = vec![Statement::Assign(
        Place::local(src),
        Rvalue::Use(cu(src_base)),
    )];
    for &(d, stride) in src_terms {
        if stride == 0 {
            continue;
        }
        body.push(Statement::Assign(
            Place::local(div),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(out_strides[d])),
        ));
        body.push(Statement::Assign(
            Place::local(md),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(div)), cu(out_shape[d])),
        ));
        body.push(Statement::Assign(
            Place::local(term),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(md)), cu(stride)),
        ));
        body.push(Statement::Assign(
            Place::local(src_new),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(src)),
                copy(Place::local(term)),
            ),
        ));
        body.push(Statement::Assign(
            Place::local(src),
            Rvalue::Use(copy(Place::local(src_new))),
        ));
    }
    body.push(Statement::Assign(
        Place::local(val),
        Rvalue::Use(copy(elem(inp, src))),
    ));
    body.push(Statement::Assign(
        elem(out, i),
        Rvalue::Use(copy(Place::local(val))),
    ));
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
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: body,
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, al.locals, vec![bb0, bb1, bb2, bb3])
}
