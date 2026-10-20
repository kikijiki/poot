//! Packed-linear graph fixtures: the packed weight compositions (`PackedDequant` over staged carrier constants,
//! a claimed indexed `PackedContraction`) that a graph, planner, oracle or device test builds directly. No
//! production caller builds these: a model declares dense weights and `bind_packed_weights` rewrites them from
//! the loader's `WeightFormats`. They live here, beside the other fixtures with no production caller, not among
//! the ops a tracer composes.

use crate::builder::{Builder, BuilderAppendPlan, Traced};
use crate::error::BuilderError;
use crate::graph::{Operand, Storage};
use crate::op::BinOp;
use crate::packed_source::packed_source_constants;
use crate::types::{DType, TensorType};
use crate::weight_ref::WeightRef;
use poot_weights::WeightId;

/// Shared rank-2 packed linear composition. Bias remains an ordinary Binary(Add) outside
/// the packed-contraction candidate.
pub fn packed_linear(
    b: &Builder,
    x: Traced,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
    alias_weight_shape: Option<Vec<usize>>,
    bias: Option<Traced>,
) -> Result<Traced, BuilderError> {
    assert!(
        !linear_id.is_empty(),
        "a packed linear needs an id to name its carriers"
    );
    let mut plan = b.append_plan(0)?;
    let weight = stage_packed_dequant(&mut plan, linear_id, descriptor)?;
    let weight = plan.equation(
        crate::op::OpKind::Transpose { perm: vec![1, 0] },
        vec![Operand::Value(weight.id())],
    )?;
    let weight = match alias_weight_shape {
        Some(shape) => plan.equation(
            crate::op::OpKind::Reshape { shape },
            vec![Operand::Value(weight.id())],
        )?,
        None => weight,
    };
    let output = plan.equation(
        crate::op::OpKind::MatMul,
        vec![Operand::Value(x.id()), Operand::Value(weight.id())],
    )?;
    let output = match bias {
        Some(bias) => plan_binary(&mut plan, BinOp::Add, output, bias)?,
        None => output,
    };
    finish_packed_plan(b, plan, output)
}

/// A packed embedding lookup (card 545a): `decode(table)[ids]` along axis 0, the table a
/// packed `[rows, K]` weight (a quantized GGUF `token_embd`). `ids` is any-shape I32 row ids; the
/// result is `ids.shape ++ [K]`. `compile` claims the `PackedDequant -> Gather` pair as one
/// `PackedRowGather`, so only the gathered rows are ever decoded.
pub fn packed_embedding(
    b: &Builder,
    ids: Traced,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
) -> Result<Traced, BuilderError> {
    assert!(
        !linear_id.is_empty(),
        "a packed linear needs an id to name its carriers"
    );
    let mut plan = b.append_plan(0)?;
    let table = stage_packed_dequant(&mut plan, linear_id, descriptor)?;
    let rows = plan.equation(
        crate::op::OpKind::Gather { axis: 0 },
        vec![Operand::Value(table.id()), Operand::Value(ids.id())],
    )?;
    finish_packed_plan(b, plan, rows)
}

/// Block-diagonal packed linear (Card 385): the stored `[out, k]` pair is grouped into `blocks`
/// contiguous row-blocks before the canonical transpose, so each activation block contracts only
/// against its own weight block. `packed_linear` cannot express this: the grouped view needs a
/// reshape (`[blocks, out/blocks, k]`) and then a 3-axis transpose (`[0,2,1]`) to bring `k`
/// innermost. `x` is `[blocks, m, k]`; batched-matmul broadcast gives each block its own
/// `[m, out/blocks]` slab, so no new primitive is needed. [`crate::op::OpKind::PackedContraction`]'s
/// [`crate::op::ContractionRows::Blocks`] recognizes the folded form for lowering.
///
/// First target: DeepSeek-V4's `attn.wo_a`, an E4M3/E8M0 packed pair stored
/// `[o_groups*o_lora_rank, in_per_group]` whose block contracts against
/// `[o_groups, in_per_group, o_lora_rank]`.
pub fn packed_block_diagonal_linear(
    b: &Builder,
    x: Traced,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
    blocks: usize,
    bias: Option<Traced>,
) -> Result<Traced, BuilderError> {
    assert!(
        !linear_id.is_empty(),
        "a packed linear needs an id to name its carriers"
    );
    let [out, k] = descriptor.shape();
    assert!(
        blocks != 0 && out % blocks == 0,
        "packed block-diagonal out={out} is not divisible by blocks={blocks}"
    );
    let block_out = out / blocks;
    let mut plan = b.append_plan(0)?;
    let weight = stage_packed_dequant(&mut plan, linear_id, descriptor)?;
    let weight = plan.equation(
        crate::op::OpKind::Reshape {
            shape: vec![blocks, block_out, k],
        },
        vec![Operand::Value(weight.id())],
    )?;
    let weight = plan.equation(
        crate::op::OpKind::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![Operand::Value(weight.id())],
    )?;
    let output = plan.equation(
        crate::op::OpKind::MatMul,
        vec![Operand::Value(x.id()), Operand::Value(weight.id())],
    )?;
    let output = match bias {
        Some(bias) => plan_binary(&mut plan, BinOp::Add, output, bias)?,
        None => output,
    };
    finish_packed_plan(b, plan, output)
}

/// The stacked table of one routed projection: `experts` packed weights of one `descriptor`, staged as one
/// carrier constant per source role of `stack` (a [`WeightRef`] with that source), each
/// `[experts, slab_bytes]` ([`crate::packed_source::packed_stack_source_type`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedExpertStack {
    /// The stack's own weight; each source role's carrier constant is a part of it.
    pub stack: WeightId,
    /// The packed layout every expert in the stack shares.
    pub descriptor: poot_quant::PackedWeight,
    /// The number of experts stacked along the carrier's leading axis.
    pub experts: usize,
}

/// The stack the packed fixtures share: the weight a fixture's expert carriers are parts of when it needs no
/// particular id.
pub const TEST_STACK: WeightId = WeightId::layer(
    0,
    poot_weights::WeightRole::Moe(poot_weights::MoeRole::Stack(poot_weights::ExpertProj::Up)),
);

/// A routed packed projection: `x [M, K]` contracted per row against the expert `selector [M]` names in
/// `table`, `[M, out]`. One claimed [`crate::op::OpKind::PackedContraction`] with
/// [`crate::op::ContractionRows::Indexed`] over one stacked carrier per source role: no expert is dequantized
/// outside the contraction, and a row reads only its own expert's blocks.
pub fn packed_indexed_linear(
    b: &Builder,
    x: Traced,
    selector: Traced,
    table: &PackedExpertStack,
) -> Result<Traced, BuilderError> {
    let mut plan = b.append_plan(0)?;
    let [_, k] = table.descriptor.shape();
    let x_type = plan
        .type_of(x.id())
        .cloned()
        .ok_or(BuilderError::InvalidOperand {
            equation: 0,
            operation: "packed_indexed_linear".to_string(),
            value: x.id(),
        })?;
    if x_type.dtype != DType::F32 || x_type.shape.len() != 2 || x_type.shape[1] != k {
        return Err(BuilderError::OperandType {
            value: x.id(),
            expected: TensorType::f32(vec![x_type.shape.first().copied().unwrap_or(0), k]),
            actual: x_type,
        });
    }
    require_operand_type(
        &plan,
        selector,
        TensorType::new(vec![x_type.shape[0]], DType::I32),
    )?;
    let mut operands = vec![Operand::Value(x.id()), Operand::Value(selector.id())];
    for role in table.descriptor.sources() {
        let carrier = plan.weight_input(
            WeightRef::whole(table.stack).with_source(role),
            crate::packed_source::packed_stack_source_type(table.descriptor, table.experts, role),
        )?;
        operands.push(Operand::Value(carrier.id()));
    }
    let output = plan.equation(
        crate::op::OpKind::PackedContraction {
            descriptor: table.descriptor,
            rows: crate::op::ContractionRows::Indexed {
                experts: table.experts,
            },
        },
        operands,
    )?;
    finish_packed_plan(b, plan, output)
}

pub(crate) fn finish_packed_plan(
    b: &Builder,
    mut plan: BuilderAppendPlan,
    output: Traced,
) -> Result<Traced, BuilderError> {
    plan.declare_result(output)?;
    let mut prepared = b.preflight_append(plan)?;
    b.commit_append(&mut prepared)
}

/// Stage every source constant of one packed weight and emit its `PackedDequant` over them, in
/// [`poot_quant::PackedWeight::sources`] role order: one carrier for a block format (GGUF), one per
/// planar operand otherwise (AWQ three, GPTQ four, the E4M3/E2M1 cells two).
pub(crate) fn stage_packed_dequant(
    plan: &mut BuilderAppendPlan,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
) -> Result<Traced, BuilderError> {
    let mut sources = Vec::new();
    for (name, tensor_type) in packed_source_constants(linear_id, descriptor) {
        let source = plan.input(name, tensor_type, Storage::Const)?;
        sources.push(Operand::Value(source.id()));
    }
    plan.equation(crate::op::OpKind::PackedDequant { descriptor }, sources)
}

pub(crate) fn require_operand_type(
    plan: &BuilderAppendPlan,
    value: Traced,
    expected: TensorType,
) -> Result<(), BuilderError> {
    let actual = plan
        .type_of(value.id())
        .cloned()
        .ok_or(BuilderError::InvalidOperand {
            equation: 0,
            operation: "packed_linear operand".to_string(),
            value: value.id(),
        })?;
    if actual != expected {
        return Err(BuilderError::OperandType {
            value: value.id(),
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn plan_binary(
    plan: &mut BuilderAppendPlan,
    op: BinOp,
    left: Traced,
    right: Traced,
) -> Result<Traced, BuilderError> {
    plan.equation(
        crate::op::OpKind::Binary(op),
        vec![Operand::Value(left.id()), Operand::Value(right.id())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operand;
    use crate::Storage;
    use crate::op::OpKind;
    use crate::packed_source::PackedSourceName;
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;

    #[test]
    fn packed_linear_append_is_transactional() -> Result<(), BuilderError> {
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 35]).unwrap();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 35])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![3])).unwrap();
        let output = packed_linear(&b, x, "layer", descriptor, None, Some(bias))?;
        let graph = b.finish(output).unwrap();
        assert_eq!(graph.values().len(), 8);
        assert_eq!(graph.inputs().len(), 4);
        assert_eq!(graph.consts().len(), 4);
        assert!(graph.slots().is_empty());
        let [weight_name, scale_name] = PackedSourceName::pair("layer");
        assert_eq!(
            graph
                .inputs()
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
            graph.eqns().iter().map(|eqn| &eqn.op).collect::<Vec<_>>().as_slice(),
            [
                OpKind::PackedDequant { .. },
                OpKind::Transpose { perm },
                OpKind::MatMul,
                OpKind::Binary(BinOp::Add),
            ] if perm.as_slice() == [1, 0]
        ));
        assert_eq!(
            graph
                .eqns()
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
    fn packed_linear_stages_every_source_in_role_order() -> Result<(), BuilderError> {
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
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![2, 256])).unwrap();
            let output = packed_linear(&b, x, "layer", descriptor, None, None)?;
            let graph = b.finish(output).unwrap();
            let roles = descriptor.sources();
            assert_eq!(roles.len(), source_count, "{format:?}");
            let OpKind::PackedDequant { .. } = graph.eqns()[0].op else {
                panic!("{format:?}: first equation is {:?}", graph.eqns()[0].op);
            };
            let carriers: Vec<crate::ValueId> = graph.eqns()[0]
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

        let collision_builder = Builder::for_tests();
        let x = collision_builder
            .constant("x", TensorType::f32(vec![2, 35]))
            .unwrap();
        let taken = PackedSourceName::scale("layer");
        collision_builder
            .constant(taken.as_str(), TensorType::new(vec![1, 1], DType::I8))
            .unwrap();
        let before = collision_builder.generation();
        assert!(matches!(
            packed_linear(&collision_builder, x, "layer", descriptor, None, None),
            Err(BuilderError::NameCollision {
                name,
                requested: crate::BuilderValueNamespace::Constant,
                existing: crate::BuilderValueNamespace::Constant,
            }) if name == taken.as_str()
        ));
        assert_eq!(collision_builder.generation(), before);
        let graph = collision_builder.finish(x).unwrap();
        assert_eq!(graph.values().len(), 2);
        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.consts().len(), 2);
        assert!(graph.slots().is_empty());
        assert!(graph.eqns().is_empty());

        let inference_builder = Builder::for_tests();
        let wrong_x = inference_builder
            .constant("wrong_x", TensorType::f32(vec![2, 34]))
            .unwrap();
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
            Err(BuilderError::Inference {
                operation,
                source: crate::ShapeError::MatMulContract { .. },
                ..
            }) if operation == "matmul"
        ));
        assert_eq!(inference_builder.generation(), before);
        let graph = inference_builder.finish(wrong_x).unwrap();
        assert_eq!(graph.values().len(), 1);
        assert_eq!(graph.inputs(), vec![wrong_x.id()]);
        assert_eq!(graph.consts(), vec![wrong_x.id()]);
        assert!(graph.slots().is_empty());
        assert!(graph.eqns().is_empty());
    }

    #[test]
    fn packed_linear_late_error_reaches_caller() {
        packed_linear_error_preserves_graph();
    }

    #[test]
    fn packed_block_diagonal_linear_builds_expected_chain() -> Result<(), BuilderError> {
        // out=8, k=35, blocks=4 -> block_out=2 (mirrors the real attn.wo_a geometry at a small scale:
        // o_groups*o_lora_rank stored rows, a stored k, grouped into o_groups blocks).
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4, 5, 35])).unwrap();
        let output = packed_block_diagonal_linear(&b, x, "layer", descriptor, 4, None)?;
        let graph = b.finish(output).unwrap();
        assert_eq!(graph.aval(graph.result()).shape, vec![4, 5, 2]);
        assert!(matches!(
            graph.eqns().iter().map(|eqn| &eqn.op).collect::<Vec<_>>().as_slice(),
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
    #[should_panic(expected = "is not divisible by blocks=3")]
    fn packed_block_diagonal_linear_rejects_indivisible_blocks() {
        let descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [8, 35]).unwrap();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![3, 5, 35])).unwrap();
        let _ = packed_block_diagonal_linear(&b, x, "layer", descriptor, 3, None);
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
    -> Result<(), BuilderError> {
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
                stack: TEST_STACK,
                descriptor,
                experts: 3,
            };
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![4, k])).unwrap();
            let selector = b
                .constant("selector", TensorType::new(vec![4], DType::I32))
                .unwrap();
            let output = packed_indexed_linear(&b, x, selector, &table)?;
            let graph = b.finish(output).unwrap();

            let [eqn] = graph.eqns() else {
                panic!("{format:?}: expected one equation, got {:?}", graph.eqns());
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
                [x.id(), selector.id()]
            );
            for (position, role) in descriptor.sources().into_iter().enumerate() {
                let carrier = operand_value(eqn, 2 + position);
                let expected = WeightRef::whole(TEST_STACK).with_source(role);
                assert_eq!(
                    graph.meta(carrier).name.as_deref(),
                    Some(expected.name().as_str())
                );
                assert_eq!(graph.meta(carrier).weight(), Some(expected));
                assert_eq!(
                    graph.aval(carrier),
                    &TensorType::new(vec![3, descriptor.slab_bytes(role)], DType::I8),
                    "{format:?} {role:?}"
                );
            }
            assert_eq!(graph.result(), eqn.out);
        }
        Ok(())
    }

    /// A stack with no id, no experts, or operands of the wrong type is refused without touching the builder.
    /// Each operand case has exactly one operand wrong and is refused as that operand's `OperandType`.
    #[test]
    fn packed_indexed_linear_refuses_a_bad_table_or_operand_without_mutation() {
        let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [3, 32]).unwrap();
        let table = PackedExpertStack {
            stack: TEST_STACK,
            descriptor,
            experts: 2,
        };
        let (m, k) = (4, 32);
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let selector = b
            .constant("selector", TensorType::new(vec![m], DType::I32))
            .unwrap();
        let before = b.generation();
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
                TensorType::new(vec![m], DType::I32),
                true,
            ),
            (
                "x rank 3",
                TensorType::f32(vec![m, k, 1]),
                TensorType::new(vec![m], DType::I32),
                true,
            ),
            (
                "x K + 1",
                TensorType::f32(vec![m, k + 1]),
                TensorType::new(vec![m], DType::I32),
                true,
            ),
            (
                "selector m + 1",
                TensorType::f32(vec![m, k]),
                TensorType::new(vec![m + 1], DType::I32),
                false,
            ),
            (
                "selector f32",
                TensorType::f32(vec![m, k]),
                TensorType::f32(vec![m]),
                false,
            ),
        ];
        for (label, x_type, selector_type, x_is_wrong) in &cases {
            let b = Builder::for_tests();
            let x = b.constant("x", x_type.clone()).unwrap();
            let selector = b.constant("selector", selector_type.clone()).unwrap();
            let wrong = if *x_is_wrong { x.id() } else { selector.id() };
            match packed_indexed_linear(&b, x, selector, &table) {
                Err(BuilderError::OperandType { value, .. }) if value == wrong => {}
                other => panic!("{label}: expected OperandType on {wrong}, got {other:?}"),
            }
        }
    }
}
