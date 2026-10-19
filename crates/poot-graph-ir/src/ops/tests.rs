use super::*;
use crate::op::OpKind;

/// The equation kinds `f` emits over one `[2, 3]` input, in trace order.
fn trace_kinds(f: fn(&Builder, Traced) -> Result<Traced, BuilderError>) -> Vec<OpKind> {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let out = f(&b, x).unwrap();
    b.finish(out)
        .unwrap()
        .eqns()
        .iter()
        .map(|e| e.op.clone())
        .collect()
}

/// Card 630: the activations are primitive compositions, not primitives. `silu` is the single division
/// `x / (1 + exp(-x))`, `gelu` the sigmoid form over `exp` with no `Tanh`, and `gelu_erf` is built on the
/// `Erf` primitive. Mutation: emit `silu` as `x * sigmoid(x)`; the `Div` is gone and the row goes red.
#[test]
fn activations_are_compositions_over_the_transcendental_primitives() {
    use BinOp::{Add, Div, Mul};
    let un = OpKind::Unary;
    let bin = OpKind::Binary;
    assert_eq!(
        trace_kinds(silu),
        [un(UnOp::Neg), un(UnOp::Exp), bin(Add), bin(Div)]
    );
    assert_eq!(
        trace_kinds(gelu),
        [
            bin(Mul),
            bin(Mul),
            bin(Mul),
            bin(Add),
            bin(Mul),
            bin(Mul),
            un(UnOp::Exp),
            bin(Add),
            bin(Div),
        ]
    );
    assert_eq!(
        trace_kinds(gelu_erf),
        [bin(Mul), un(UnOp::Erf), bin(Add), bin(Mul), bin(Mul)]
    );
    assert_eq!(trace_kinds(tanh), [un(UnOp::Tanh)]);
}

/// `swiglu`/`geglu` are the gate's activation equations plus one trailing `Mul` by `up`.
#[test]
fn swiglu_and_geglu_emit_activation_then_mul() {
    let cases = [
        (
            silu as fn(&Builder, Traced) -> Result<Traced, BuilderError>,
            swiglu as fn(&Builder, Traced, Traced) -> Result<Traced, BuilderError>,
        ),
        (gelu, geglu),
    ];
    for (act, f) in cases {
        let b = Builder::for_tests();
        let gate = b.constant("gate", TensorType::f32(vec![2, 3])).unwrap();
        let up = b.constant("up", TensorType::f32(vec![2, 3])).unwrap();
        let out = f(&b, gate, up).unwrap();
        let kinds: Vec<OpKind> = b
            .finish(out)
            .unwrap()
            .eqns()
            .iter()
            .map(|e| e.op.clone())
            .collect();
        let mut want = trace_kinds(act);
        want.push(OpKind::Binary(BinOp::Mul));
        assert_eq!(kinds, want);
    }
}

/// Test-only alternative UT-transform via block-recursive 2x2 halving (spec 139 review C6).
/// It is intentionally retained only as an equation-count cross-check for the production
/// Neumann-doubling implementation.
fn ut_inverse_block_recursive(b: &Builder, attn: Traced, c: usize, rank: usize) -> Traced {
    if c == 1 {
        let zero = b.binary(BinOp::Sub, attn, attn).unwrap();
        return b.binary_scalar(BinOp::Add, zero, Scalar::F32(1.0)).unwrap();
    }
    assert_eq!(
        c % 2,
        0,
        "unit_lower_triangular_inverse: block-recursive halving needs C a power of two (got block size {c})"
    );
    let half = c / 2;
    let (ax0, ax1) = (rank - 2, rank - 1);
    let a_attn = b
        .slice(b.slice(attn, ax0, 0, half).unwrap(), ax1, 0, half)
        .unwrap();
    let d_attn = b
        .slice(b.slice(attn, ax0, half, c).unwrap(), ax1, half, c)
        .unwrap();
    let b_block = b
        .slice(b.slice(attn, ax0, half, c).unwrap(), ax1, 0, half)
        .unwrap();

    let a_inv = ut_inverse_block_recursive(b, a_attn, half, rank);
    let d_inv = ut_inverse_block_recursive(b, d_attn, half, rank);
    let ba = b.matmul(b_block, a_inv).unwrap();
    let dba = b.matmul(d_inv, ba).unwrap();
    let neg_dba = b.unary(UnOp::Neg, dba).unwrap();
    let zeros = b.binary(BinOp::Sub, a_inv, a_inv).unwrap();

    let top = b.concat(ax1, &[a_inv, zeros]).unwrap();
    let bottom = b.concat(ax1, &[neg_dba, d_inv]).unwrap();
    b.concat(ax0, &[top, bottom]).unwrap()
}

/// The Neumann-doubling UT-transform (the default) keeps eqn count logarithmic in C where
/// block-recursive halving was linear: at C=64 it emits ~45 eqns vs 947 (card 158). Guards against
/// a regression that would reinflate the GDN-prefill dispatch count.
#[test]
fn ut_transform_neumann_doubling_is_log_op_count() {
    for (c, max_eqns) in [(8usize, 30), (16, 40), (32, 45), (64, 55)] {
        let b = Builder::for_tests();
        let attn = b.constant("attn", TensorType::f32(vec![c, c])).unwrap();
        let t = unit_lower_triangular_inverse(&b, attn).unwrap();
        let g = b.finish(t).unwrap();
        assert!(
            g.eqns().len() <= max_eqns,
            "UT-transform C={c} emitted {} eqns (expected <= {max_eqns}, LOG in C)",
            g.eqns().len()
        );
    }
    // and the old block-recursive (kept as a cross-check) is still ~20x heavier at C=64.
    let b = Builder::for_tests();
    let attn = b.constant("attn", TensorType::f32(vec![64, 64])).unwrap();
    let t = ut_inverse_block_recursive(&b, attn, 64, 2);
    let g = b.finish(t).unwrap();
    assert!(
        g.eqns().len() > 900,
        "block-recursive C=64 should stay LINEAR (~947 eqns), got {}",
        g.eqns().len()
    );
}

/// POOT-1041: a live bound types at every float dtype the capacity bound does: its decomposition selects inactive
/// keys with the bit-moving `Select`, which has a BF16 and an F16 form. BF16 and F16 queries type as `Live`
/// with the operands' dtype, and an I32 query is still the typed `DtypeOp` error under both bounds. Mutation:
/// reinstate `(live && q.dtype != DType::F32)` in `infer_attention`; the BF16 and F16 live rows fail to type.
#[test]
fn a_live_attention_bound_types_at_every_float_dtype() {
    use crate::error::ShapeError;
    use crate::op::{AttentionAttrs, KvBound};
    use crate::types::DType;
    let (hq, lk, d) = (2, 4, 8);
    let operands = |dtype| {
        vec![
            TensorType::new(vec![1, hq, 1, d], dtype),
            TensorType::new(vec![1, hq, lk, d], dtype),
            TensorType::new(vec![1, hq, lk, d], dtype),
            TensorType::new(vec![1, 1, 1, lk], dtype),
        ]
    };
    let attrs = |bound| AttentionAttrs {
        bound,
        ..AttentionAttrs::new(1, 0.5, None)
    };
    let len = TensorType::new(vec![1], DType::I32);
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let capacity = OpKind::Attention(attrs(KvBound::Capacity)).infer(&operands(dtype));
        assert_eq!(
            capacity.unwrap().dtype,
            dtype,
            "{dtype:?} types as capacity"
        );
        let mut live_operands = operands(dtype);
        live_operands.push(len.clone());
        let live = OpKind::Attention(attrs(KvBound::Live)).infer(&live_operands);
        assert_eq!(
            live.unwrap().dtype,
            dtype,
            "{dtype:?} types as a live bound"
        );
    }
    let mut integer_live = operands(DType::I32);
    integer_live.push(len);
    for (bound, operands) in [
        (KvBound::Capacity, operands(DType::I32)),
        (KvBound::Live, integer_live),
    ] {
        let typed = OpKind::Attention(attrs(bound)).infer(&operands);
        assert!(
            matches!(
                typed,
                Err(ShapeError::DtypeOp {
                    op: "Attention",
                    dtype: DType::I32
                })
            ),
            "an I32 query is a typed error under {bound:?}, got {typed:?}"
        );
    }
}

/// `paged_view` over a read map or a pool of the wrong rank is a typed builder refusal naming the expected
/// rank, not an index panic; the well-formed pair builds the `[B, Hkv, Lk, D]` view.
///
/// Mutation: require a rank-3 read map in the `let &[rows, cap]` guard; the well-formed row is refused with
/// `Rank { expected: 2, found: [2, 3] }`.
#[test]
fn a_paged_view_over_a_mis_ranked_map_or_pool_is_a_typed_refusal() {
    use crate::types::TensorType;
    let b = Builder::for_tests();
    let pool = b.constant("pool", TensorType::f32(vec![8, 2, 4])).unwrap();
    let flat_pool = b
        .constant("flat_pool", TensorType::f32(vec![8, 8]))
        .unwrap();
    let i32_map = |name: &str, shape: Vec<usize>| {
        b.constant(name, TensorType::new(shape, DType::I32))
            .unwrap()
    };
    let read = i32_map("read", vec![2, 3]);
    let flat_read = i32_map("flat_read", vec![6]);

    let view = paged_view(&b, pool, read).expect("a rank-3 pool and a rank-2 map build a view");
    assert_eq!(b.aval(view).unwrap().shape, [2, 2, 3, 4]);
    for (what, result, expected, found) in [
        (
            "a rank-1 read map",
            paged_view(&b, pool, flat_read),
            2,
            vec![6],
        ),
        (
            "a rank-2 pool",
            paged_view(&b, flat_pool, read),
            3,
            vec![8, 8],
        ),
    ] {
        match result {
            Err(BuilderError::Requirement {
                operation: "paged view",
                requirement:
                    Requirement::Rank {
                        expected: want,
                        found: got,
                    },
            }) => assert_eq!((want, got), (expected, found), "{what}"),
            other => panic!("{what}: expected a typed rank refusal, got {other:?}"),
        }
    }
}
