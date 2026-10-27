//! A unary, binary or scalar-literal op is a pointwise region of one step. These rows run such regions on
//! the reference interpreter over views, broadcasts and the 2-D grid fold, and check the two-byte float
//! bodies' structure (widen, compute in F32, round back).

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::{BinOp, Body, Constant, MathOp, Operand, Rvalue, Statement, Ty, UnOp};
use poot_kernelgen::{
    FusedInput, FusedKernel, FusedScalarOp, Layout, LeafId, LeafTy, fused, fused_views,
};

fn leaves(count: usize) -> Vec<FusedInput> {
    (0..count)
        .map(|leaf| FusedInput::Leaf(LeafId::new(leaf)))
        .collect()
}

fn kernel(op: FusedScalarOp, leaf_tys: Vec<LeafTy>, lit: Option<f32>) -> FusedKernel {
    let mut inputs = leaves(leaf_tys.len());
    inputs.extend(lit.map(FusedInput::Lit));
    FusedKernel::one_step(op, leaf_tys, inputs)
}

/// Run `body` once per workgroup of `groups` over `inputs`, into an output of `out_len` f32s.
fn run_f32(body: &Body, groups: [u32; 3], inputs: &[&[f32]], out_len: usize) -> Vec<f32> {
    let mut buffers: Vec<Buffer> = inputs.iter().map(|data| Buffer::from_f32s(data)).collect();
    buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));
    run(body, groups, &mut buffers).expect("the body runs");
    buffers.pop().unwrap().to_f32s().unwrap()
}

fn groups_for(body: &Body, threads: usize) -> [u32; 3] {
    [
        threads.div_ceil(body.workgroup_size[0] as usize) as u32,
        1,
        1,
    ]
}

/// `neg(transpose(x))` read straight from `x`'s untransposed buffer through the view's strides, with no copy.
///
/// Mutation: in `emit_view_offset`, drop the `off += coord * stride` term; the transposed read returns the
/// untransposed order and this row goes red.
#[test]
fn a_unary_op_reads_a_transposed_view() {
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
    let layout = Layout {
        strides: vec![1, 3],
        offset: 0,
    };
    let body = fused_views(
        "neg_view",
        &[3, 2],
        &[&[3, 2]],
        &[layout],
        &kernel(FusedScalarOp::Unary(UnOp::Neg), vec![LeafTy::F32], None),
        None,
    )
    .unwrap();
    let got = run_f32(&body, groups_for(&body, 6), &[&x], 6);
    // transpose(x) = [[1,4],[2,5],[3,6]] -> neg.
    assert_eq!(got, [-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
}

/// Every leaf contiguous and no fold: the view-reading form is the plain form, byte for byte.
#[test]
fn contiguous_layouts_generate_the_plain_body() {
    let shape = [4usize, 5];
    let k = kernel(FusedScalarOp::Unary(UnOp::Neg), vec![LeafTy::F32], None);
    let plain = fused("k", &shape, &[&shape], &k).unwrap();
    let viewed = fused_views(
        "k",
        &shape,
        &[&shape],
        &[Layout::contiguous(&shape)],
        &k,
        None,
    )
    .unwrap();
    assert_eq!(format!("{plain:?}"), format!("{viewed:?}"));
}

/// numpy broadcasting over mixed axes, a viewed operand with a base offset, and a literal: each output
/// element reads the right element of each operand.
#[test]
fn a_binary_op_broadcasts_and_reads_a_view_with_an_offset() {
    let (a_shape, b_shape, out_shape) = ([2usize, 1, 4], [3usize, 1], [2usize, 3, 4]);
    let a: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    // `b` is a [3, 1] column read from the middle of a bigger buffer: row stride 2, offset 3.
    let b_buffer: Vec<f32> = (0..10).map(|i| 10.0 * i as f32).collect();
    let b_layout = Layout {
        strides: vec![2, 1],
        offset: 3,
    };
    for (op, f) in [
        (BinOp::Add, (|x: f32, y: f32| x + y) as fn(f32, f32) -> f32),
        (BinOp::Mul, |x, y| x * y),
        (BinOp::Sub, |x, y| x - y),
        (BinOp::Max, f32::max),
    ] {
        let body = fused_views(
            "bcast",
            &out_shape,
            &[&a_shape, &b_shape],
            &[Layout::contiguous(&a_shape), b_layout.clone()],
            &kernel(FusedScalarOp::Binary(op), vec![LeafTy::F32; 2], None),
            None,
        )
        .unwrap();
        let got = run_f32(&body, groups_for(&body, 24), &[&a, &b_buffer], 24);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let want = f(a[i * 4 + k], b_buffer[3 + 2 * j]);
                    assert_eq!(got[(i * 3 + j) * 4 + k], want, "{op:?} [{i},{j},{k}]");
                }
            }
        }
    }
}

/// The scalar-literal form: `x * 2.5`, and a comparison against a literal writes 1.0 or 0.0.
#[test]
fn a_scalar_literal_is_an_inline_operand() {
    let x = [-1.0f32, 0.0, 1.0, 2.0];
    let mul = fused(
        "mul",
        &[4],
        &[&[4]],
        &kernel(
            FusedScalarOp::Binary(BinOp::Mul),
            vec![LeafTy::F32],
            Some(2.5),
        ),
    )
    .unwrap();
    assert_eq!(
        run_f32(&mul, groups_for(&mul, 4), &[&x], 4),
        [-2.5, 0.0, 2.5, 5.0]
    );
    let ge = fused(
        "ge",
        &[4],
        &[&[4]],
        &kernel(
            FusedScalarOp::Binary(BinOp::Ge),
            vec![LeafTy::F32],
            Some(0.0),
        ),
    )
    .unwrap();
    assert_eq!(
        run_f32(&ge, groups_for(&ge, 4), &[&x], 4),
        [0.0, 1.0, 1.0, 1.0]
    );
}

/// The unary math and expansion ops of one step: sqrt, exp, reciprocal, tanh and erf against the host.
#[test]
fn unary_math_ops_match_the_host() {
    let x = [0.25f32, 1.0, 4.0, 9.0];
    for (op, host) in [
        (
            FusedScalarOp::Math(MathOp::Sqrt),
            (|v: f32| v.sqrt()) as fn(f32) -> f32,
        ),
        (FusedScalarOp::Math(MathOp::Exp), |v| v.exp()),
        (FusedScalarOp::Recip, |v| 1.0 / v),
        (FusedScalarOp::Tanh, |v| v.tanh()),
    ] {
        let body = fused("u", &[4], &[&[4]], &kernel(op, vec![LeafTy::F32], None)).unwrap();
        let got = run_f32(&body, groups_for(&body, 4), &[&x], 4);
        for (g, v) in got.iter().zip(x) {
            let want = host(v);
            assert!(
                (g - want).abs() <= 1e-6 * want.abs().max(1.0),
                "{op:?}({v}): {g} vs {want}"
            );
        }
    }
}

/// Past the device's X grid cap the launch folds onto a 2-D grid of 256-lane workgroups; every element is
/// still written exactly once.
///
/// Mutation: in `pointwise_body`'s fold, compute `i = group_id * wg` without the lane; the tail elements stay
/// zero and this row goes red.
#[test]
fn the_2d_grid_fold_covers_every_element() {
    let numel = 700usize;
    let x: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let k = kernel(
        FusedScalarOp::Binary(BinOp::Add),
        vec![LeafTy::F32],
        Some(1.0),
    );
    let x_groups = 2;
    let mut body = fused_views(
        "fold",
        &[numel],
        &[&[numel]],
        &[Layout::contiguous(&[numel])],
        &k,
        Some(x_groups),
    )
    .unwrap();
    body.workgroup_size = [256, 1, 1];
    let total = numel.div_ceil(256);
    let groups = [x_groups as u32, total.div_ceil(x_groups) as u32, 1];
    let got = run_f32(&body, groups, &[&x], numel);
    assert!(got.iter().zip(&x).all(|(g, v)| *g == v + 1.0));
}

fn statements(body: &Body) -> impl Iterator<Item = &Statement> {
    body.blocks.iter().flat_map(|block| &block.statements)
}

/// A two-byte float step widens each value operand to F32, computes there, and rounds the result back: the
/// body reads and writes BF16 or F16 slices and computes on F32. A literal stays an F32 constant.
#[test]
fn a_two_byte_float_step_computes_in_f32_and_rounds_back() {
    for (leaf, ty) in [(LeafTy::BF16, Ty::BF16), (LeafTy::F16, Ty::F16)] {
        let body = fused(
            "narrow",
            &[4],
            &[&[4], &[4]],
            &kernel(FusedScalarOp::Binary(BinOp::Add), vec![leaf; 2], None),
        )
        .unwrap();
        let slice = |mutable| Ty::Ref {
            mutable,
            pointee: Box::new(Ty::Slice(Box::new(ty.clone()))),
        };
        assert_eq!(body.locals[1].ty, slice(false));
        assert_eq!(body.locals[2].ty, slice(false));
        assert_eq!(body.locals[3].ty, slice(true));
        let widened = statements(&body)
            .filter(|s| matches!(s, Statement::Assign(_, Rvalue::Cast { to: Ty::F32, .. })))
            .count();
        assert_eq!(widened, 2, "{ty:?}: both operands widen");
        assert_eq!(
            statements(&body)
                .filter(|s| matches!(s, Statement::Assign(_, Rvalue::Cast { to, .. }) if *to == ty))
                .count(),
            1,
            "{ty:?}: the result rounds back once"
        );
        assert!(statements(&body).any(|s| matches!(
            s,
            Statement::Assign(
                _,
                Rvalue::BinaryOp(BinOp::Add, Operand::Copy(_), Operand::Copy(_))
            )
        )));
    }
    // With a literal operand only the value widens.
    let scaled = fused(
        "scaled",
        &[4],
        &[&[4]],
        &kernel(
            FusedScalarOp::Binary(BinOp::Mul),
            vec![LeafTy::BF16],
            Some(0.5),
        ),
    )
    .unwrap();
    assert!(statements(&scaled).any(|s| matches!(
        s,
        Statement::Assign(
            _,
            Rvalue::BinaryOp(BinOp::Mul, Operand::Copy(_), Operand::Const(Constant::F32(v)))
        ) if *v == 0.5
    )));
}

/// Operands of different widths in one arithmetic step have no float meaning: a typed refusal.
#[test]
fn mixed_width_operands_are_refused() {
    let refused = fused(
        "mixed",
        &[4],
        &[&[4], &[4]],
        &kernel(
            FusedScalarOp::Binary(BinOp::Add),
            vec![LeafTy::BF16, LeafTy::F32],
            None,
        ),
    );
    assert!(refused.is_err(), "{refused:?}");
}
