use super::*;
use crate::op::OpKind;
use crate::packed_source::PackedSourceName;
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;

/// The equation kinds `f` emits over one `[2, 3]` input, in trace order.
fn trace_kinds(f: fn(&Builder, Traced) -> Traced) -> Vec<OpKind> {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 3]));
    let out = f(&b, x);
    b.finish(out).eqns.into_iter().map(|e| e.op).collect()
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
            silu as fn(&Builder, Traced) -> Traced,
            swiglu as fn(&Builder, Traced, Traced) -> Traced,
        ),
        (gelu, geglu),
    ];
    for (act, f) in cases {
        let b = Builder::new();
        let gate = b.constant("gate", TensorType::f32(vec![2, 3]));
        let up = b.constant("up", TensorType::f32(vec![2, 3]));
        let out = f(&b, gate, up);
        let kinds: Vec<OpKind> = b.finish(out).eqns.into_iter().map(|e| e.op).collect();
        let mut want = trace_kinds(act);
        want.push(OpKind::Binary(BinOp::Mul));
        assert_eq!(kinds, want);
    }
}

#[test]
fn packed_linear_append_is_transactional() -> Result<(), BuilderAppendError> {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 35]));
    let bias = b.constant("bias", TensorType::f32(vec![3]));
    let output = packed_linear(&b, x, "layer", descriptor, None, Some(bias))?;
    let graph = b.finish(output);
    assert_eq!(graph.values.len(), 8);
    assert_eq!(graph.inputs.len(), 4);
    assert_eq!(graph.consts.len(), 4);
    assert!(graph.slots.is_empty());
    let [weight_name, scale_name] = PackedSourceName::pair("layer");
    assert_eq!(
        graph
            .inputs
            .iter()
            .map(|&value| graph.meta(value).name.as_deref())
            .collect::<Vec<_>>(),
        vec![
            Some("x"),
            Some("bias"),
            Some(weight_name.as_str()),
            Some(scale_name.as_str()),
        ]
    );
    assert!(matches!(
        graph.eqns.iter().map(|eqn| &eqn.op).collect::<Vec<_>>().as_slice(),
        [
            OpKind::PackedDequant { .. },
            OpKind::Transpose { perm },
            OpKind::MatMul,
            OpKind::Binary(BinOp::Add),
        ] if perm.as_slice() == [1, 0]
    ));
    assert_eq!(
        graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
            .count(),
        1
    );
    Ok(())
}

/// Card 642: `packed_linear` stages one `I8` constant per descriptor source, in
/// `PackedWeight::sources` role order, and its `PackedDequant` reads exactly those constants in that
/// order - one for a GGUF block format, three for AWQ, four for act-order GPTQ. The recognized
/// `PackedContraction` keeps the same carrier list behind the activation. Mutation: stage the sources
/// in reverse; the AWQ row goes red (`PackedDequantOperand { role: Planar(Codes), field: "shape" }`,
/// the op's own role-typed inference refuses the swapped carriers).
#[test]
fn packed_linear_stages_every_source_in_role_order() -> Result<(), BuilderAppendError> {
    use std::num::NonZeroUsize;
    let nz = |n| NonZeroUsize::new(n).unwrap();
    let cases = [
        (WeightFormat::Q4_K, 1),
        (WeightFormat::Awq { group_size: nz(32) }, 3),
        (
            WeightFormat::Gptq {
                groups: poot_quant::format::GroupMap::Indexed { groups: nz(8) },
            },
            4,
        ),
    ];
    for (format, source_count) in cases {
        let descriptor = PackedWeight::try_new(format, [8, 256]).unwrap();
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 256]));
        let output = packed_linear(&b, x, "layer", descriptor, None, None)?;
        let graph = b.finish(output);
        let roles = descriptor.sources();
        assert_eq!(roles.len(), source_count, "{format:?}");
        let OpKind::PackedDequant { .. } = graph.eqns[0].op else {
            panic!("{format:?}: first equation is {:?}", graph.eqns[0].op);
        };
        let carriers: Vec<crate::ValueId> = graph.eqns[0]
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => *value,
                other => panic!("{format:?}: carrier {other:?}"),
            })
            .collect();
        let staged: Vec<(poot_quant::SourceRole, DType)> = carriers
            .iter()
            .map(|&carrier| {
                let meta = graph.meta(carrier);
                let name = PackedSourceName::parse(meta.name.as_deref().unwrap()).unwrap();
                assert_eq!(name.linear_id(), "layer");
                assert_eq!(meta.storage, Storage::Const);
                (name.role(), graph.aval(carrier).dtype)
            })
            .collect();
        let expected: Vec<(poot_quant::SourceRole, DType)> =
            roles.iter().map(|&role| (role, DType::I8)).collect();
        assert_eq!(staged, expected, "{format:?}: carriers out of role order");

        // The claim that `poot_graph_plan::recognize_packed_contractions` reads this `PackedDequant`'s
        // carriers in the same order used to live here too (card 626: that pass moved to
        // `poot-graph-plan`, a downstream consumer of this crate - a dev-dependency back onto it from
        // here would recompile this crate twice, `--cfg test` and not, and hand the pass a `Graph` from
        // the wrong compilation of it). Covered instead by `poot-graph-plan`'s own
        // `planner::packed_dequant` tests.
    }
    Ok(())
}

#[test]
fn packed_linear_error_preserves_graph() {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();

    let collision_builder = Builder::new();
    let x = collision_builder.constant("x", TensorType::f32(vec![2, 35]));
    let taken = PackedSourceName::scale("layer");
    collision_builder.constant(taken.as_str(), TensorType::new(vec![1, 1], DType::I8));
    let before = collision_builder.generation();
    assert!(matches!(
        packed_linear(&collision_builder, x, "layer", descriptor, None, None),
        Err(BuilderAppendError::NameCollision {
            name,
            requested: crate::BuilderValueNamespace::Constant,
            existing: crate::BuilderValueNamespace::Constant,
        }) if name == taken.as_str()
    ));
    assert_eq!(collision_builder.generation(), before);
    let graph = collision_builder.finish(x);
    assert_eq!(graph.values.len(), 2);
    assert_eq!(graph.inputs.len(), 2);
    assert_eq!(graph.consts.len(), 2);
    assert!(graph.slots.is_empty());
    assert!(graph.eqns.is_empty());

    let inference_builder = Builder::new();
    let wrong_x = inference_builder.constant("wrong_x", TensorType::f32(vec![2, 34]));
    let before = inference_builder.generation();
    assert!(matches!(
        packed_linear(
            &inference_builder,
            wrong_x,
            "layer",
            descriptor,
            None,
            None,
        ),
        Err(BuilderAppendError::Inference {
            operation,
            source: crate::ShapeError::MatMulContract { .. },
            ..
        }) if operation == "matmul"
    ));
    assert_eq!(inference_builder.generation(), before);
    let graph = inference_builder.finish(wrong_x);
    assert_eq!(graph.values.len(), 1);
    assert_eq!(graph.inputs, vec![wrong_x.id]);
    assert_eq!(graph.consts, vec![wrong_x.id]);
    assert!(graph.slots.is_empty());
    assert!(graph.eqns.is_empty());
}

#[test]
fn packed_linear_late_error_reaches_caller() {
    packed_linear_error_preserves_graph();
}

#[test]
fn packed_block_diagonal_linear_builds_expected_chain() -> Result<(), BuilderAppendError> {
    // out=8, k=35, blocks=4 -> block_out=2 (mirrors the real attn.wo_a geometry at a small scale:
    // o_groups*o_lora_rank stored rows, a stored k, grouped into o_groups blocks).
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 5, 35]));
    let output = packed_block_diagonal_linear(&b, x, "layer", descriptor, 4, None)?;
    let graph = b.finish(output);
    assert_eq!(graph.aval(graph.output).shape, vec![4, 5, 2]);
    assert!(matches!(
        graph.eqns.iter().map(|eqn| &eqn.op).collect::<Vec<_>>().as_slice(),
        [
            OpKind::PackedDequant { .. },
            OpKind::Reshape { shape },
            OpKind::Transpose { perm },
            OpKind::MatMul,
        ] if shape.as_slice() == [4, 2, 35] && perm.as_slice() == [0, 2, 1]
    ));
    Ok(())
}

#[test]
fn packed_block_diagonal_linear_rejects_indivisible_blocks() {
    let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![3, 5, 35]));
    let before = b.generation();
    assert!(matches!(
        packed_block_diagonal_linear(&b, x, "layer", descriptor, 3, None),
        Err(BuilderAppendError::PackedBlockDiagonalBlocks { out: 8, blocks: 3 })
    ));
    assert_eq!(
        b.generation(),
        before,
        "a rejected call must not mutate the builder"
    );
    let graph = b.finish(x);
    assert_eq!(graph.values.len(), 1);
    assert!(graph.eqns.is_empty(), "no packed equations were staged");
}

fn operand_value(eqn: &crate::Eqn, position: usize) -> crate::ValueId {
    match eqn.inputs[position] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!("literal fixture expects a value edge"),
    }
}

/// SC-001 (IR half): `packed_indexed_linear` over every packed layout the engine serves builds exactly one
/// claimed `PackedContraction { rows: Indexed }`, no `PackedDequant`, no `Materialize`-shaped chain and no
/// sort, whose operands are `[x, ids, carriers..]`, each carrier one stacked, word-padded constant named
/// after the stack id. Mutation: stage one carrier per expert (the removed W12 spelling); the equation count
/// and the carrier shapes go red.
#[test]
fn packed_indexed_linear_is_one_claimed_contraction_over_stacked_carriers()
-> Result<(), BuilderAppendError> {
    use poot_quant::format::GroupMap;
    use std::num::NonZeroUsize;
    let formats = [
        WeightFormat::Q8_0,
        WeightFormat::Q4_K,
        WeightFormat::Awq {
            group_size: NonZeroUsize::new(32).unwrap(),
        },
        WeightFormat::Gptq {
            groups: GroupMap::Indexed {
                groups: NonZeroUsize::new(2).unwrap(),
            },
        },
        WeightFormat::Mxfp4,
    ];
    for format in formats {
        let k = match format {
            WeightFormat::Q4_K => 256,
            _ => 64,
        };
        let descriptor = PackedWeight::try_new(format, [3, k]).unwrap();
        let table = PackedExpertStack {
            stack_id: "layer.experts".to_string(),
            descriptor,
            experts: 3,
        };
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, k]));
        let selector = b.constant("selector", TensorType::f32(vec![4]));
        let output = packed_indexed_linear(&b, x, selector, &table)?;
        let graph = b.finish(output);

        let [eqn] = graph.eqns.as_slice() else {
            panic!("{format:?}: expected one equation, got {:?}", graph.eqns);
        };
        assert_eq!(
            eqn.op,
            OpKind::PackedContraction {
                descriptor,
                rows: crate::op::ContractionRows::Indexed { experts: 3 },
            },
            "{format:?}"
        );
        assert_eq!(
            eqn.inputs.len(),
            2 + descriptor.sources().len(),
            "{format:?}"
        );
        assert_eq!(
            [operand_value(eqn, 0), operand_value(eqn, 1)],
            [x.id, selector.id]
        );
        for (position, role) in descriptor.sources().into_iter().enumerate() {
            let carrier = operand_value(eqn, 2 + position);
            let expected = PackedSourceName::new("layer.experts", role);
            assert_eq!(graph.meta(carrier).name.as_deref(), Some(expected.as_str()));
            assert_eq!(
                graph.aval(carrier),
                &TensorType::new(vec![3, descriptor.slab_bytes(role)], DType::I8),
                "{format:?} {role:?}"
            );
        }
        assert_eq!(graph.output, eqn.out);
    }
    Ok(())
}

/// A stack with no id, no experts, or operands of the wrong type is refused without touching the builder.
/// Each operand case has exactly one operand wrong and is refused as that operand's `OperandType`.
#[test]
fn packed_indexed_linear_refuses_a_bad_table_or_operand_without_mutation() {
    let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
    let table = PackedExpertStack {
        stack_id: "experts".to_string(),
        descriptor,
        experts: 2,
    };
    let (m, k) = (4, 32);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let selector = b.constant("selector", TensorType::f32(vec![m]));
    let before = b.generation();
    let unnamed = PackedExpertStack {
        stack_id: String::new(),
        ..table.clone()
    };
    assert!(matches!(
        packed_indexed_linear(&b, x, selector, &unnamed),
        Err(BuilderAppendError::EmptyPackedLinearId { index: 0 })
    ));
    let empty = PackedExpertStack {
        experts: 0,
        ..table.clone()
    };
    assert!(packed_indexed_linear(&b, x, selector, &empty).is_err());
    assert_eq!(
        b.generation(),
        before,
        "a rejected table mutated the builder"
    );

    let cases = [
        (
            "x f16",
            TensorType::new(vec![m, k], DType::F16),
            TensorType::f32(vec![m]),
            true,
        ),
        (
            "x rank 3",
            TensorType::f32(vec![m, k, 1]),
            TensorType::f32(vec![m]),
            true,
        ),
        (
            "x K + 1",
            TensorType::f32(vec![m, k + 1]),
            TensorType::f32(vec![m]),
            true,
        ),
        (
            "selector m + 1",
            TensorType::f32(vec![m, k]),
            TensorType::f32(vec![m + 1]),
            false,
        ),
        (
            "selector i32",
            TensorType::f32(vec![m, k]),
            TensorType::new(vec![m], DType::I32),
            false,
        ),
    ];
    for (label, x_type, selector_type, x_is_wrong) in &cases {
        let b = Builder::new();
        let x = b.constant("x", x_type.clone());
        let selector = b.constant("selector", selector_type.clone());
        let wrong = if *x_is_wrong { x.id } else { selector.id };
        match packed_indexed_linear(&b, x, selector, &table) {
            Err(BuilderAppendError::OperandType { value, .. }) if value == wrong => {}
            other => panic!("{label}: expected OperandType on {wrong}, got {other:?}"),
        }
    }
}

/// Test-only alternative UT-transform via block-recursive 2x2 halving (spec 139 review C6).
/// It is intentionally retained only as an equation-count cross-check for the production
/// Neumann-doubling implementation.
fn ut_inverse_block_recursive(b: &Builder, attn: Traced, c: usize, rank: usize) -> Traced {
    if c == 1 {
        let zero = b.binary(BinOp::Sub, attn, attn);
        return b.binary_scalar(BinOp::Add, zero, Scalar::F32(1.0));
    }
    assert_eq!(
        c % 2,
        0,
        "unit_lower_triangular_inverse: block-recursive halving needs C a power of two (got block size {c})"
    );
    let half = c / 2;
    let (ax0, ax1) = (rank - 2, rank - 1);
    let a_attn = b.slice(b.slice(attn, ax0, 0, half), ax1, 0, half);
    let d_attn = b.slice(b.slice(attn, ax0, half, c), ax1, half, c);
    let b_block = b.slice(b.slice(attn, ax0, half, c), ax1, 0, half);

    let a_inv = ut_inverse_block_recursive(b, a_attn, half, rank);
    let d_inv = ut_inverse_block_recursive(b, d_attn, half, rank);
    let ba = b.matmul(b_block, a_inv);
    let dba = b.matmul(d_inv, ba);
    let neg_dba = b.unary(UnOp::Neg, dba);
    let zeros = b.binary(BinOp::Sub, a_inv, a_inv);

    let top = b.concat(ax1, &[a_inv, zeros]);
    let bottom = b.concat(ax1, &[neg_dba, d_inv]);
    b.concat(ax0, &[top, bottom])
}

/// The Neumann-doubling UT-transform (the default) keeps eqn count logarithmic in C where
/// block-recursive halving was linear: at C=64 it emits ~45 eqns vs 947 (card 158). Guards against
/// a regression that would reinflate the GDN-prefill dispatch count.
#[test]
fn ut_transform_neumann_doubling_is_log_op_count() {
    for (c, max_eqns) in [(8usize, 30), (16, 40), (32, 45), (64, 55)] {
        let b = Builder::new();
        let attn = b.constant("attn", TensorType::f32(vec![c, c]));
        let t = unit_lower_triangular_inverse(&b, attn);
        let g = b.finish(t);
        assert!(
            g.eqns.len() <= max_eqns,
            "UT-transform C={c} emitted {} eqns (expected <= {max_eqns}, LOG in C)",
            g.eqns.len()
        );
    }
    // and the old block-recursive (kept as a cross-check) is still ~20x heavier at C=64.
    let b = Builder::new();
    let attn = b.constant("attn", TensorType::f32(vec![64, 64]));
    let t = ut_inverse_block_recursive(&b, attn, 64, 2);
    let g = b.finish(t);
    assert!(
        g.eqns.len() > 900,
        "block-recursive C=64 should stay LINEAR (~947 eqns), got {}",
        g.eqns.len()
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
