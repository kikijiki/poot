use super::*;
use crate::op::OpKind;
use crate::packed_source::PackedSourceName;
use poot_quant::PackedWeight;
use poot_quant::format::{ScaleEncoding, WeightFormat};

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

fn packed_rows(descriptor: PackedWeight) -> Vec<PackedLinearGraphRow> {
    (0..2)
        .map(|ordinal| PackedLinearGraphRow {
            ordinal,
            linear_id: format!("layer.expert.{ordinal}"),
            descriptor,
        })
        .collect()
}

fn operand_value(eqn: &crate::Eqn, position: usize) -> crate::ValueId {
    match eqn.inputs[position] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!("literal fixture expects a value edge"),
    }
}

#[test]
fn packed_indexed_linear_is_card356_canonical() -> Result<(), BuilderAppendError> {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [3, 5],
    )
    .unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 5]));
    let selector = b.constant("selector", TensorType::f32(vec![4]));
    let output = packed_indexed_linear(&b, x, selector, &packed_rows(descriptor))?;
    let graph = b.finish(output);

    assert_eq!(graph.eqns.len(), 8);
    assert!(matches!(
        graph.eqns.iter().map(|eqn| &eqn.op).collect::<Vec<_>>().as_slice(),
        [
            OpKind::PackedDequant { descriptor: first },
            OpKind::Transpose { perm: first_perm },
            OpKind::Reshape { shape: first_shape },
            OpKind::PackedDequant { descriptor: second },
            OpKind::Transpose { perm: second_perm },
            OpKind::Reshape { shape: second_shape },
            OpKind::Concat { axis: 0 },
            OpKind::IndexedMatMul,
        ] if *first == descriptor
            && *second == descriptor
            && first_perm == &[1, 0]
            && second_perm == &[1, 0]
            && first_shape == &[1, 5, 3]
            && second_shape == &[1, 5, 3]
    ));
    let first_carriers = [
        operand_value(&graph.eqns[0], 0),
        operand_value(&graph.eqns[0], 1),
    ];
    let second_carriers = [
        operand_value(&graph.eqns[3], 0),
        operand_value(&graph.eqns[3], 1),
    ];
    let [first_weight, first_scale] = PackedSourceName::pair("layer.expert.0");
    let [second_weight, second_scale] = PackedSourceName::pair("layer.expert.1");
    assert_eq!(
        first_carriers.map(|value| graph.meta(value).name.as_deref()),
        [Some(first_weight.as_str()), Some(first_scale.as_str())]
    );
    assert_eq!(
        second_carriers.map(|value| graph.meta(value).name.as_deref()),
        [Some(second_weight.as_str()), Some(second_scale.as_str())]
    );
    assert_eq!(
        graph.eqns[6]
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => *value,
                Operand::Lit(_) => usize::MAX,
            })
            .collect::<Vec<_>>(),
        vec![graph.eqns[2].out, graph.eqns[5].out]
    );
    assert_eq!(
        graph.eqns[7]
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => *value,
                Operand::Lit(_) => usize::MAX,
            })
            .collect::<Vec<_>>(),
        vec![x.id, graph.eqns[6].out, selector.id]
    );
    assert_eq!(graph.output, graph.eqns[7].out);
    Ok(())
}

#[test]
fn packed_grouped_linear_is_card356_canonical() -> Result<(), BuilderAppendError> {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [3, 5],
    )
    .unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![4, 5]));
    let selector = b.constant("selector", TensorType::f32(vec![4]));
    let output = packed_grouped_linear(&b, x, selector, &packed_rows(descriptor))?;
    let graph = b.finish(output);

    // The leading two equations are the computed row/expert iotas.
    assert!(matches!(graph.eqns[0].op, OpKind::Iota { len: 4 }));
    assert!(matches!(graph.eqns[1].op, OpKind::Iota { len: 3 }));
    let expert_iota = graph.eqns[1].out;
    assert_eq!(graph.eqns.len(), 40);
    let perm = graph.eqns[20].out;
    assert!(matches!(
        graph.eqns[20].op,
        OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false
        }
    ));
    assert_eq!(
        [
            operand_value(&graph.eqns[6], 0),
            operand_value(&graph.eqns[6], 1)
        ],
        [graph.eqns[5].out, graph.eqns[3].out]
    );
    assert_eq!(
        [
            operand_value(&graph.eqns[9], 0),
            operand_value(&graph.eqns[9], 1)
        ],
        [graph.eqns[3].out, graph.eqns[5].out]
    );
    assert_eq!(
        [
            operand_value(&graph.eqns[15], 0),
            operand_value(&graph.eqns[15], 1)
        ],
        [graph.eqns[14].out, graph.eqns[12].out]
    );
    for (equation, source) in [(21, x.id), (22, selector.id)] {
        assert!(matches!(
            graph.eqns[equation].op,
            OpKind::Scatter { axis: 0 }
        ));
        assert_eq!(
            [
                operand_value(&graph.eqns[equation], 0),
                operand_value(&graph.eqns[equation], 1)
            ],
            [source, perm]
        );
    }
    assert_eq!(operand_value(&graph.eqns[23], 0), selector.id);
    assert_eq!(operand_value(&graph.eqns[25], 0), expert_iota);
    assert!(matches!(
        graph.eqns[30].op,
        OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false
        }
    ));
    assert!(matches!(graph.eqns[37].op, OpKind::Concat { axis: 0 }));
    assert!(matches!(graph.eqns[38].op, OpKind::IndexedMatMul));
    assert_eq!(
        [
            operand_value(&graph.eqns[38], 0),
            operand_value(&graph.eqns[38], 1),
            operand_value(&graph.eqns[38], 2),
        ],
        [graph.eqns[21].out, graph.eqns[37].out, graph.eqns[22].out]
    );
    assert!(matches!(graph.eqns[39].op, OpKind::Gather { axis: 0 }));
    assert_eq!(
        [
            operand_value(&graph.eqns[39], 0),
            operand_value(&graph.eqns[39], 1)
        ],
        [graph.eqns[38].out, perm]
    );
    assert_eq!(graph.output, graph.eqns[39].out);
    Ok(())
}

#[test]
fn packed_expert_builders_reject_invalid_tables_without_mutation() {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [3, 5],
    )
    .unwrap();
    let other = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [3, 5],
    )
    .unwrap();
    let builder = Builder::new();
    let x = builder.constant("x", TensorType::f32(vec![4, 5]));
    let selector = builder.constant("selector", TensorType::f32(vec![4]));
    let before = builder.generation();
    let cases = [
        (
            vec![PackedLinearGraphRow {
                ordinal: 1,
                linear_id: "expert.0".to_string(),
                descriptor,
            }],
            "ordinal",
        ),
        (
            vec![PackedLinearGraphRow {
                ordinal: 0,
                linear_id: String::new(),
                descriptor,
            }],
            "empty id",
        ),
        (
            vec![
                PackedLinearGraphRow {
                    ordinal: 0,
                    linear_id: "duplicate".to_string(),
                    descriptor,
                },
                PackedLinearGraphRow {
                    ordinal: 1,
                    linear_id: "duplicate".to_string(),
                    descriptor,
                },
            ],
            "duplicate id",
        ),
        (
            vec![
                PackedLinearGraphRow {
                    ordinal: 0,
                    linear_id: "expert.0".to_string(),
                    descriptor,
                },
                PackedLinearGraphRow {
                    ordinal: 1,
                    linear_id: "expert.1".to_string(),
                    descriptor: other,
                },
            ],
            "descriptor",
        ),
    ];
    assert!(matches!(
        packed_indexed_linear(&builder, x, selector, &[]),
        Err(BuilderAppendError::EmptyPackedRows)
    ));
    for (rows, label) in cases {
        let error = packed_indexed_linear(&builder, x, selector, &rows);
        assert!(error.is_err(), "{label} must be rejected");
        assert_eq!(builder.generation(), before, "{label} mutated the builder");
    }
    let graph = builder.finish(x);
    assert_eq!(graph.values.len(), 2);
    assert!(graph.eqns.is_empty());
}

/// Mutant M7 (mutants-m4.md): `validate_indexed_inputs`'s `x` check (`dtype != F32 || rank != 2
/// || K != k`, either `||` made `&&`) and `require_operand_type -> Ok(())` were never tested one
/// condition at a time. Each row has exactly one operand wrong and must be refused as that
/// operand's `OperandType`, by both the indexed and the grouped builder.
#[test]
fn packed_indexed_inputs_refuse_one_wrong_operand_at_a_time() {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [3, 5],
    )
    .unwrap();
    let rows = packed_rows(descriptor);
    let (m, k) = (4, 5);
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
    type Build =
        fn(&Builder, Traced, Traced, &[PackedLinearGraphRow]) -> Result<Traced, BuilderAppendError>;
    let builders: [(&str, Build); 2] = [
        ("packed_indexed_linear", packed_indexed_linear),
        ("packed_grouped_linear", packed_grouped_linear),
    ];
    for (name, build) in builders {
        for (label, x_type, selector_type, x_is_wrong) in &cases {
            let b = Builder::new();
            let x = b.constant("x", x_type.clone());
            let selector = b.constant("selector", selector_type.clone());
            let wrong = if *x_is_wrong { x.id } else { selector.id };
            match build(&b, x, selector, &rows) {
                Err(BuilderAppendError::OperandType { value, .. }) if value == wrong => {}
                other => panic!("{name} {label}: expected OperandType on {wrong}, got {other:?}"),
            }
        }
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let selector = b.constant("selector", TensorType::f32(vec![m]));
        build(&b, x, selector, &rows)
            .unwrap_or_else(|error| panic!("{name}: the well-typed operands build: {error}"));
    }
}

/// Mutant M8 (mutants-m4.md): `packed_grouped_linear`'s row-count guard `value > F32_EXACT_INT_MAX`
/// made `>=` or `==` moves the boundary. The sort computes row indices in f32, exact up to `2^24`:
/// `m = 2^24` builds and `m = 2^24 + 1` is refused. Builder-only: no value is allocated.
#[test]
fn packed_grouped_linear_row_count_boundary_is_f32_exact() {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        [3, 5],
    )
    .unwrap();
    let rows = packed_rows(descriptor);
    let build = |m: usize| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, 5]));
        let selector = b.constant("selector", TensorType::f32(vec![m]));
        packed_grouped_linear(&b, x, selector, &rows)
    };
    build(1 << 24).unwrap_or_else(|error| panic!("m = 2^24 must build: {error}"));
    assert!(matches!(
        build((1 << 24) + 1),
        Err(BuilderAppendError::F32ExactIntegerRange {
            field: "row count",
            value,
            max,
        }) if value == (1 << 24) + 1 && max == 1 << 24
    ));
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
