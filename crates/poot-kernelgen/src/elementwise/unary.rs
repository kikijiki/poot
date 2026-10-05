use super::*;

/// Elementwise math-unary `y[i] = f(x[i])` (sqrt/abs/floor/..., non-transcendental), in storage dtype `dt`
/// (spec 024; f32 compute). Only this crate's own tests exercise the plain (non-grid, non-view) form
/// directly today; production callers go through [`math_unary_dt_grid`]/[`math_unary_dt_views_grid`].
#[cfg(test)]
fn math_unary_dt(name: &str, dt: Ty, op: MathOp) -> Body {
    build_unary_dt(name, dt, |x_i| Rvalue::MathUnary(op, copy(x_i)))
}

/// [`math_unary_dt`] with an optional 2-D grid fold; see [`build_unary_dt_grid`]. `x_groups: None` is
/// identical to [`math_unary_dt`].
pub fn math_unary_dt_grid(name: &str, dt: Ty, op: MathOp, x_groups: Option<usize>) -> Body {
    build_unary_dt_grid(name, dt, |x_i| Rvalue::MathUnary(op, copy(x_i)), x_groups)
}

/// Math-unary `dt` over a view layout with an optional 2-D grid fold; see `build_unary_dt_views_grid`.
pub fn math_unary_dt_views_grid(
    name: &str,
    dt: Ty,
    op: MathOp,
    out_shape: &[usize],
    layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    build_unary_dt_views_grid(
        name,
        dt,
        out_shape,
        layout,
        |x_i| Rvalue::MathUnary(op, copy(x_i)),
        x_groups,
    )
}

/// Reciprocal `dt` with an optional 2-D grid fold; see `build_unary_dt_grid`.
pub fn recip_dt_grid(name: &str, dt: Ty, x_groups: Option<usize>) -> Body {
    build_unary_dt_grid(
        name,
        dt,
        |x_i| Rvalue::BinaryOp(BinOp::Div, Operand::Const(Constant::F32(1.0)), copy(x_i)),
        x_groups,
    )
}

/// Reciprocal `dt` over a view layout with an optional 2-D grid fold; see `build_unary_dt_views_grid`.
pub fn recip_dt_views_grid(
    name: &str,
    dt: Ty,
    out_shape: &[usize],
    layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    build_unary_dt_views_grid(
        name,
        dt,
        out_shape,
        layout,
        |x_i| Rvalue::BinaryOp(BinOp::Div, Operand::Const(Constant::F32(1.0)), copy(x_i)),
        x_groups,
    )
}

/// Elementwise unary `y[i] = op(x[i])` (neg/not).
pub fn unary(name: &str, op: UnOp) -> Body {
    unary_dt(name, Ty::F32, op)
}

/// `unary` in storage dtype `dt` (spec 024; f32 compute).
pub fn unary_dt(name: &str, dt: Ty, op: UnOp) -> Body {
    build_unary_dt(name, dt, |x_i| Rvalue::UnaryOp(op, copy(x_i)))
}

/// [`unary_dt`] with an optional 2-D grid fold; see [`build_unary_dt_grid`].
pub fn unary_dt_grid(name: &str, dt: Ty, op: UnOp, x_groups: Option<usize>) -> Body {
    build_unary_dt_grid(name, dt, |x_i| Rvalue::UnaryOp(op, copy(x_i)), x_groups)
}

/// Exact I32 leading-zero count of the unsigned 32-bit pattern, returning `0..=32`.
pub fn unary_i32_clz_grid(name: &str, x_groups: Option<usize>) -> Body {
    build_unary_dt_grid(
        name,
        Ty::I32,
        |x_i| Rvalue::IntScalarUnary(IntScalarOp::LeadingZeros, copy(x_i)),
        x_groups,
    )
}

/// Exact I32 leading-zero count reading `x` through `layout`.
pub fn unary_i32_clz_views_grid(
    name: &str,
    out_shape: &[usize],
    layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    build_unary_dt_views_grid(
        name,
        Ty::I32,
        out_shape,
        layout,
        |x_i| Rvalue::IntScalarUnary(IntScalarOp::LeadingZeros, copy(x_i)),
        x_groups,
    )
}

/// [`unary_dt`] generalized to read `x` through a physical [`Layout`] (spec 132 phase 1). Only this
/// crate's own tests exercise the plain (non-grid) form directly today; production callers go through
/// [`unary_dt_views_grid`].
#[cfg(test)]
fn unary_dt_views(name: &str, dt: Ty, op: UnOp, out_shape: &[usize], layout: &Layout) -> Body {
    build_unary_dt_views_grid(
        name,
        dt,
        out_shape,
        layout,
        |x_i| Rvalue::UnaryOp(op, copy(x_i)),
        None,
    )
}

/// [`unary_dt_views`] with an optional 2-D grid fold; see [`build_unary_dt_views_grid`].
pub fn unary_dt_views_grid(
    name: &str,
    dt: Ty,
    op: UnOp,
    out_shape: &[usize],
    layout: &Layout,
    x_groups: Option<usize>,
) -> Body {
    build_unary_dt_views_grid(
        name,
        dt,
        out_shape,
        layout,
        |x_i| Rvalue::UnaryOp(op, copy(x_i)),
        x_groups,
    )
}

#[cfg(test)]
mod tests {
    use poot_runtime::KernelBuffer;

    use super::*;
    use crate::test_support::{ctx, spv};

    #[test]
    fn math_sqrt() {
        let Some(ctx) = ctx() else { return };
        let s = spv(&math_unary_dt("sqrt", Ty::F32, MathOp::Sqrt), "sqrt");
        let x = [1.0f32, 4.0, 9.0, 16.0];
        let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(4)];
        ctx.dispatch("test", &s, [64, 1, 1], [4, 1, 1], &mut bufs)
            .unwrap();
        assert_eq!(bufs[1].as_f32(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn math_exp_matches_reference() {
        let Some(ctx) = ctx() else { return };
        let x = [0.0f32, 1.0, -1.0, 2.0];
        let s = spv(&math_unary_dt("exp", Ty::F32, MathOp::Exp), "exp");
        let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(4)];
        ctx.dispatch("test", &s, [64, 1, 1], [4, 1, 1], &mut bufs)
            .unwrap();
        for (g, v) in bufs[1].as_f32().iter().zip(x.iter()) {
            let want = v.exp();
            assert!(
                (g - want).abs() <= 1e-4 * want.max(1.0),
                "exp: {g} vs {want}"
            );
        }
    }

    #[test]
    fn unary_neg_view_reads_a_transposed_input() {
        // `unary_dt_views` reads its operand through a physical Layout. x is [2,3] row-major; the Layout is
        // what `compute_views` derives for `transpose(x, [1,0])` (out_shape [3,2]): strides [1,3], offset 0.
        // The kernel must compute `neg(transpose(x))` straight from x's untransposed buffer, with no
        // materialize/copy kernel.
        let Some(ctx) = ctx() else { return };
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let layout = Layout {
            strides: vec![1, 3],
            offset: 0,
        };
        let body = unary_dt_views("negview", Ty::F32, UnOp::Neg, &[3, 2], &layout);
        let s = spv(&body, "negview");
        let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(6)];
        ctx.dispatch("test", &s, [64, 1, 1], [6, 1, 1], &mut bufs)
            .unwrap();
        // transpose(x) = [[1,4],[2,5],[3,6]] -> neg -> [-1,-4,-2,-5,-3,-6].
        assert_eq!(bufs[1].as_f32(), &[-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
    }

    #[test]
    fn unary_neg_view_contiguous_is_byte_identical_to_unary_dt() {
        // `unary_dt_views` with `Layout::contiguous` must generate the exact same Body as `unary_dt`, so
        // existing (non-view) callers are untouched.
        let shape = vec![4usize, 5];
        let a = unary_dt("k", Ty::F32, UnOp::Neg);
        let b = unary_dt_views("k", Ty::F32, UnOp::Neg, &shape, &Layout::contiguous(&shape));
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}
