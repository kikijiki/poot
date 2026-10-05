//! Composite ops defined by their decomposition (ADR-0101 tier 1, Card 556).
//!
//! A composite ([`OpClass::Composite`](crate::OpClass)) has no definition of its own: its semantics are
//! the primitive graph [`decompose`] builds, from the same `ops::` functions a tracer calls. The CPU
//! oracle evaluates that graph, so a pass that replaces a primitive chain with the composite preserves
//! the oracle's result bit for bit, and a kernel for the composite is compared with the decomposition at
//! tier 2. `OpKind::class() == Composite` exactly when [`decompose`] returns `Some` for the op's operand
//! types (the consistency row in this file's tests).

use crate::builder::{Builder, BuilderAppendPlan, Traced};
use crate::error::BuilderAppendError;
use crate::graph::{Graph, Operand, ValueId};
use crate::op::{FusedOp, FusedOperand, FusedRegion, OpKind, RowRegion, RowStep};
use crate::ops;
use crate::types::TensorType;

/// A composite's primitive definition: `graph` computes the composite's result as its output from the
/// constants `operands`, one per equation operand, in operand order. An evaluator binds the equation's
/// operand values to `operands` and evaluates `graph`.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub graph: Graph,
    pub operands: Vec<ValueId>,
}

/// The decomposition of `op` over operands of types `operands`, or `None` when `op` is a primitive or
/// `op` does not type over `operands` (`OpKind::infer` refuses them).
///
/// Every operand is a [`Builder::constant`] placeholder; the composite's parameters are baked into the
/// primitive equations exactly as the tracer's `ops::` function bakes them:
///
/// - `FlashAttentionDecode { n_rep, scale }`: [`ops::attention_masked_softcap`] with no softcap.
/// - `FlashAttentionPrefill { n_rep, scale, softcap }`: [`ops::attention_prefill_softcap`].
/// - `Rope { rot }`: [`ops::rope_partial`] over `x`'s last axis.
/// - `MatMulBias`: [`ops::linear`] with the bias.
/// - `Fused(region)` / `FusedRow(region)`: the region's steps as equations, operands in place (a
///   `PackLane` leaf is the unit `Slice` of its packed input, a non-empty `pack` the last-axis `Concat`).
pub fn decompose(op: &OpKind, operands: &[TensorType]) -> Option<Decomposition> {
    let out = op.infer(operands).ok()?;
    let b = Builder::new();
    let inputs: Vec<Traced> = operands
        .iter()
        .enumerate()
        .map(|(position, ty)| b.constant(&format!("operand.{position}"), ty.clone()))
        .collect();
    let result = match op {
        OpKind::FlashAttentionDecode { n_rep, scale } => {
            let &[q, k, v, mask] = inputs.as_slice() else {
                return None;
            };
            ops::attention_masked_softcap(&b, q, k, v, *n_rep, *scale, mask, None)
        }
        OpKind::FlashAttentionPrefill {
            n_rep,
            scale,
            softcap,
        } => {
            let &[q, k, v, mask] = inputs.as_slice() else {
                return None;
            };
            ops::attention_prefill_softcap(&b, q, k, v, *n_rep, *scale, mask, *softcap)
        }
        OpKind::Rope { rot } => {
            let &[x, cos, sin] = inputs.as_slice() else {
                return None;
            };
            let shape = &operands[0].shape;
            let last = shape.len() - 1;
            ops::rope_partial(&b, x, cos, sin, shape[last], *rot, last)
        }
        OpKind::MatMulBias => {
            let &[x, w, bias] = inputs.as_slice() else {
                return None;
            };
            ops::linear(&b, x, w, Some(bias))
        }
        OpKind::Fused(region) => fused(&b, region, &inputs).ok()??,
        OpKind::FusedRow(region) => fused_row(&b, region, &inputs).ok()??,
        OpKind::Unary(_)
        | OpKind::Binary(_)
        | OpKind::Select
        | OpKind::Reduce { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Cast { .. }
        | OpKind::Reshape { .. }
        | OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Concat { .. }
        | OpKind::Iota { .. }
        | OpKind::Gather { .. }
        | OpKind::Scatter { .. }
        | OpKind::ScatterUpdate
        | OpKind::MatMul
        | OpKind::PackedDequant { .. }
        | OpKind::PackedContraction { .. }
        | OpKind::PackedRowGather { .. }
        | OpKind::DenseContraction { .. }
        | OpKind::DenseRowGather { .. }
        | OpKind::ArgTopK { .. }
        | OpKind::IndexedMatMul
        | OpKind::DynamicUpdateSlice { .. }
        | OpKind::PackI8
        | OpKind::UnpackI8 { .. }
        | OpKind::AllReduce { .. }
        | OpKind::AllGather { .. }
        | OpKind::RandomUniform { .. }
        | OpKind::SampleToken { .. } => return None,
    };
    // The composite's typing rule and its definition must agree, or the evaluator would publish a value
    // of another type than the equation declares.
    if b.aval(result) != out {
        return None;
    }
    Some(Decomposition {
        operands: inputs.iter().map(|input| input.id).collect(),
        graph: b.finish(result),
    })
}

/// One region local as an equation operand: an input or step result, or a literal kept in place.
fn region_operand(
    plan: &mut BuilderAppendPlan,
    locals: &[Traced],
    operand: &FusedOperand,
) -> Result<Option<Operand>, BuilderAppendError> {
    Ok(Some(match operand {
        FusedOperand::Local(local) => {
            let Some(value) = locals.get(*local) else {
                return Ok(None);
            };
            Operand::Value(value.id)
        }
        FusedOperand::Lit(scalar) => Operand::Lit(*scalar),
        FusedOperand::PackLane { input, lane } => {
            let Some(packed) = locals.get(*input).map(|value| value.id) else {
                return Ok(None);
            };
            let Some(ty) = plan.type_of(packed) else {
                return Ok(None);
            };
            let Some(axis) = ty.rank().checked_sub(1) else {
                return Ok(None);
            };
            let slice = OpKind::Slice {
                axis,
                start: *lane,
                end: lane + 1,
            };
            Operand::Value(plan.equation(slice, vec![Operand::Value(packed)])?.id)
        }
    }))
}

fn pointwise(op: FusedOp) -> OpKind {
    match op {
        FusedOp::Unary(op) => OpKind::Unary(op),
        FusedOp::Binary(op) => OpKind::Binary(op),
        FusedOp::Select => OpKind::Select,
    }
}

/// Stage the equations of a region whose steps `stage` emits, then commit them; `Ok(None)` when a
/// region operand names no local.
fn commit_region(
    b: &Builder,
    inputs: &[Traced],
    stage: impl FnOnce(
        &mut BuilderAppendPlan,
        &mut Vec<Traced>,
    ) -> Result<Option<Traced>, BuilderAppendError>,
) -> Result<Option<Traced>, BuilderAppendError> {
    let mut plan = b.append_plan(0);
    let mut locals = inputs.to_vec();
    let Some(result) = stage(&mut plan, &mut locals)? else {
        return Ok(None);
    };
    // A region whose output is one of its inputs has no equation to commit.
    if inputs.iter().any(|input| input.id == result.id) {
        return Ok(Some(result));
    }
    plan.declare_result(result)?;
    let mut prepared = b.preflight_append(plan)?;
    Ok(Some(Traced {
        id: b.commit_append(&mut prepared)?,
    }))
}

/// A pointwise region's steps as equations, then its output local or its last-axis `pack`.
fn fused(
    b: &Builder,
    region: &FusedRegion,
    inputs: &[Traced],
) -> Result<Option<Traced>, BuilderAppendError> {
    commit_region(b, inputs, |plan, locals| {
        for step in &region.steps {
            let mut operands = Vec::with_capacity(step.inputs.len());
            for operand in &step.inputs {
                let Some(operand) = region_operand(plan, locals, operand)? else {
                    return Ok(None);
                };
                operands.push(operand);
            }
            locals.push(plan.equation(pointwise(step.op), operands)?);
        }
        if region.pack.is_empty() {
            return Ok(locals.get(region.output).copied());
        }
        let mut parts = Vec::with_capacity(region.pack.len());
        for local in &region.pack {
            let Some(part) = locals.get(*local) else {
                return Ok(None);
            };
            parts.push(Operand::Value(part.id));
        }
        let Some(axis) = plan
            .type_of(locals[region.pack[0]].id)
            .and_then(|ty| ty.rank().checked_sub(1))
        else {
            return Ok(None);
        };
        Ok(Some(plan.equation(OpKind::Concat { axis }, parts)?))
    })
}

/// A row region's steps as equations: pointwise steps in place, each reduction a keepdim `Reduce` over
/// the region's row axis.
fn fused_row(
    b: &Builder,
    region: &RowRegion,
    inputs: &[Traced],
) -> Result<Option<Traced>, BuilderAppendError> {
    commit_region(b, inputs, |plan, locals| {
        for step in &region.steps {
            let (op, step_operands) = match step {
                RowStep::Pointwise { op, inputs } => (pointwise(*op), inputs.as_slice()),
                RowStep::Reduce { op, input } => (
                    OpKind::Reduce {
                        op: *op,
                        axis: region.axis,
                        keepdim: true,
                    },
                    std::slice::from_ref(input),
                ),
            };
            let mut operands = Vec::with_capacity(step_operands.len());
            for operand in step_operands {
                let Some(operand) = region_operand(plan, locals, operand)? else {
                    return Ok(None);
                };
                operands.push(operand);
            }
            locals.push(plan.equation(op, operands)?);
        }
        Ok(locals.get(region.output).copied())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{
        BinOp, FusedStep, OpClass, PackedWeight, RedOp, SampleRule, UnOp, WeightFormat,
    };
    use crate::types::{DType, Scalar};

    fn f32(shape: &[usize]) -> TensorType {
        TensorType::f32(shape.to_vec())
    }

    fn typed(shape: &[usize], dtype: DType) -> TensorType {
        TensorType::new(shape.to_vec(), dtype)
    }

    /// The position of `op`'s variant among every `OpKind` variant. Exhaustive with no wildcard, so a
    /// new variant cannot compile until it is placed here, and [`VARIANTS`] then demands a row for it.
    fn variant(op: &OpKind) -> usize {
        match op {
            OpKind::Unary(_) => 0,
            OpKind::Binary(_) => 1,
            OpKind::Select => 2,
            OpKind::Reduce { .. } => 3,
            OpKind::Broadcast { .. } => 4,
            OpKind::Cast { .. } => 5,
            OpKind::Reshape { .. } => 6,
            OpKind::Transpose { .. } => 7,
            OpKind::Slice { .. } => 8,
            OpKind::Concat { .. } => 9,
            OpKind::Iota { .. } => 10,
            OpKind::Gather { .. } => 11,
            OpKind::Scatter { .. } => 12,
            OpKind::ScatterUpdate => 13,
            OpKind::MatMul => 14,
            OpKind::PackedDequant { .. } => 15,
            OpKind::PackedContraction { .. } => 16,
            OpKind::PackedRowGather { .. } => 17,
            OpKind::DenseContraction { .. } => 18,
            OpKind::DenseRowGather { .. } => 19,
            OpKind::MatMulBias => 20,
            OpKind::ArgTopK { .. } => 21,
            OpKind::IndexedMatMul => 22,
            OpKind::DynamicUpdateSlice { .. } => 23,
            OpKind::PackI8 => 24,
            OpKind::UnpackI8 { .. } => 25,
            OpKind::Fused(_) => 26,
            OpKind::FusedRow(_) => 27,
            OpKind::FlashAttentionDecode { .. } => 28,
            OpKind::FlashAttentionPrefill { .. } => 29,
            OpKind::Rope { .. } => 30,
            OpKind::AllReduce { .. } => 31,
            OpKind::AllGather { .. } => 32,
            OpKind::RandomUniform { .. } => 33,
            OpKind::SampleToken { .. } => 34,
        }
    }

    const VARIANTS: usize = 35;

    /// One instance of every `OpKind` variant over canonical operand types (each row types under
    /// `infer`), the composites in each form the evaluator meets (prefill with and without softcap, a
    /// partial rope, an F32 and an I32 pack-lane region).
    fn canonical_rows() -> Vec<(OpKind, Vec<TensorType>)> {
        let packed = PackedWeight::try_new(WeightFormat::E2m1Row32, [4, 32]).unwrap();
        let sources: Vec<TensorType> = packed
            .sources()
            .iter()
            .map(|role| typed(&packed.source_shape(*role), DType::I8))
            .collect();
        let with_sources = |mut operands: Vec<TensorType>, before: bool| {
            if before {
                operands.splice(0..0, sources.iter().cloned());
            } else {
                operands.extend(sources.iter().cloned());
            }
            operands
        };
        let local = FusedOperand::Local;
        let step = |op, inputs| FusedStep { op, inputs };
        let f32_region = FusedRegion {
            n_inputs: 2,
            steps: vec![
                step(FusedOp::Unary(UnOp::Exp), vec![local(0)]),
                step(
                    FusedOp::Binary(BinOp::Add),
                    vec![local(2), FusedOperand::Lit(Scalar::F32(1.0))],
                ),
                step(FusedOp::Binary(BinOp::Div), vec![local(1), local(3)]),
            ],
            output: 4,
            pack: Vec::new(),
        };
        let lane = |lane| FusedOperand::PackLane { input: 0, lane };
        let i32_region = FusedRegion {
            n_inputs: 1,
            steps: vec![
                step(FusedOp::Binary(BinOp::Add), vec![lane(0), lane(1)]),
                step(FusedOp::Unary(UnOp::Not), vec![local(1)]),
            ],
            output: 2,
            pack: vec![1, 2],
        };
        let softmax = RowRegion {
            n_inputs: 1,
            axis: 1,
            steps: vec![
                RowStep::Reduce {
                    op: RedOp::Max,
                    input: local(0),
                },
                RowStep::Pointwise {
                    op: FusedOp::Binary(BinOp::Sub),
                    inputs: vec![local(0), local(1)],
                },
                RowStep::Pointwise {
                    op: FusedOp::Unary(UnOp::Exp),
                    inputs: vec![local(2)],
                },
                RowStep::Reduce {
                    op: RedOp::Sum,
                    input: local(3),
                },
                RowStep::Pointwise {
                    op: FusedOp::Binary(BinOp::Div),
                    inputs: vec![local(3), local(4)],
                },
            ],
            output: 5,
        };
        let i32 = DType::I32;
        let decode = vec![
            f32(&[1, 4, 1, 8]),
            f32(&[1, 2, 6, 8]),
            f32(&[1, 2, 6, 8]),
            f32(&[1, 1, 1, 6]),
        ];
        let prefill = vec![
            f32(&[1, 4, 5, 8]),
            f32(&[1, 2, 5, 8]),
            f32(&[1, 2, 5, 8]),
            f32(&[1, 4, 5, 5]),
        ];
        vec![
            (OpKind::Unary(UnOp::Neg), vec![f32(&[2, 3])]),
            (OpKind::Binary(BinOp::Add), vec![f32(&[2, 3]), f32(&[3])]),
            (
                OpKind::Select,
                vec![typed(&[2], i32), typed(&[2], i32), typed(&[2], i32)],
            ),
            (
                OpKind::Reduce {
                    op: RedOp::Sum,
                    axis: 1,
                    keepdim: true,
                },
                vec![f32(&[2, 3])],
            ),
            (OpKind::Broadcast { shape: vec![2, 3] }, vec![f32(&[1, 3])]),
            (OpKind::Cast { to: DType::BF16 }, vec![f32(&[2])]),
            (OpKind::Reshape { shape: vec![3, 2] }, vec![f32(&[2, 3])]),
            (OpKind::Transpose { perm: vec![1, 0] }, vec![f32(&[2, 3])]),
            (
                OpKind::Slice {
                    axis: 1,
                    start: 0,
                    end: 2,
                },
                vec![f32(&[2, 3])],
            ),
            (OpKind::Concat { axis: 0 }, vec![f32(&[2, 3]), f32(&[1, 3])]),
            (OpKind::Iota { len: 4 }, Vec::new()),
            (
                OpKind::Gather { axis: 0 },
                vec![f32(&[4, 3]), typed(&[2], i32)],
            ),
            (
                OpKind::Scatter { axis: 0 },
                vec![f32(&[3, 2]), typed(&[3], i32)],
            ),
            (
                OpKind::ScatterUpdate,
                vec![f32(&[4, 3]), f32(&[2, 3]), typed(&[4], i32)],
            ),
            (OpKind::MatMul, vec![f32(&[2, 3]), f32(&[3, 4])]),
            (
                OpKind::PackedDequant { descriptor: packed },
                with_sources(Vec::new(), false),
            ),
            (
                OpKind::PackedContraction {
                    descriptor: packed,
                    blocks: 1,
                },
                with_sources(vec![f32(&[2, 32])], false),
            ),
            (
                OpKind::PackedRowGather { descriptor: packed },
                with_sources(vec![typed(&[2], i32)], true),
            ),
            (
                OpKind::DenseContraction {
                    weight: DType::BF16,
                },
                vec![f32(&[2, 4]), typed(&[3, 4], DType::BF16)],
            ),
            (
                OpKind::DenseRowGather {
                    source: DType::BF16,
                },
                vec![typed(&[5, 4], DType::BF16), typed(&[2], i32)],
            ),
            (
                OpKind::MatMulBias,
                vec![f32(&[2, 3]), f32(&[3, 4]), f32(&[4])],
            ),
            (OpKind::ArgTopK { k: 2 }, vec![f32(&[2, 5])]),
            (
                OpKind::IndexedMatMul,
                vec![f32(&[2, 3]), f32(&[4, 3, 5]), typed(&[2], i32)],
            ),
            (
                OpKind::DynamicUpdateSlice { axis: 0 },
                vec![f32(&[4, 3]), f32(&[1, 3]), typed(&[], i32)],
            ),
            (OpKind::PackI8, vec![f32(&[2, 8])]),
            (OpKind::UnpackI8 { len: 8 }, vec![typed(&[2, 2], i32)]),
            (OpKind::Fused(f32_region), vec![f32(&[3, 4]), f32(&[3, 4])]),
            (OpKind::Fused(i32_region), vec![typed(&[3, 2], i32)]),
            (OpKind::FusedRow(softmax), vec![f32(&[3, 7])]),
            (
                OpKind::FlashAttentionDecode {
                    n_rep: 2,
                    scale: 0.5,
                },
                decode,
            ),
            (
                OpKind::FlashAttentionPrefill {
                    n_rep: 2,
                    scale: 0.5,
                    softcap: None,
                },
                prefill.clone(),
            ),
            (
                OpKind::FlashAttentionPrefill {
                    n_rep: 2,
                    scale: 0.5,
                    softcap: Some(30.0),
                },
                prefill,
            ),
            (
                OpKind::Rope { rot: 8 },
                vec![f32(&[1, 2, 3, 8]), f32(&[3, 8]), f32(&[3, 8])],
            ),
            (
                OpKind::Rope { rot: 4 },
                vec![f32(&[1, 2, 3, 8]), f32(&[3, 4]), f32(&[3, 4])],
            ),
            (
                OpKind::AllReduce {
                    op: RedOp::Sum,
                    axis: 0,
                },
                vec![f32(&[2])],
            ),
            (OpKind::AllGather { axis: 0 }, vec![f32(&[2])]),
            (OpKind::RandomUniform { cols: 4 }, vec![typed(&[2], i32)]),
            (
                OpKind::SampleToken {
                    rule: SampleRule::Greedy,
                },
                vec![f32(&[2, 5])],
            ),
        ]
    }

    /// Card 556 SC-004: for every `OpKind` variant, `class() == Composite` exactly when `decompose`
    /// returns `Some` for the variant's canonical operand types, and a composite's decomposition is
    /// primitive equations that compute the composite's declared type from one placeholder per
    /// operand. The class is a property of the op, not of which producer built the equation: these
    /// rows build no graph at all.
    #[test]
    fn class_is_composite_exactly_when_decompose_is_some() {
        let rows = canonical_rows();
        let mut covered = [false; VARIANTS];
        for (op, operands) in &rows {
            covered[variant(op)] = true;
            let declared = op.infer(operands).unwrap_or_else(|error| {
                panic!("{}: canonical operands must type: {error}", op.name())
            });
            let decomposition = decompose(op, operands);
            assert_eq!(
                op.class() == OpClass::Composite,
                decomposition.is_some(),
                "{}: class() is {:?} but decompose() is {}",
                op.name(),
                op.class(),
                if decomposition.is_some() {
                    "Some"
                } else {
                    "None"
                }
            );
            if let Some(decomposition) = decomposition {
                let graph = &decomposition.graph;
                assert_eq!(graph.aval(graph.output), &declared, "{}", op.name());
                assert_eq!(
                    decomposition.operands.len(),
                    operands.len(),
                    "{}",
                    op.name()
                );
                for (placeholder, operand) in decomposition.operands.iter().zip(operands) {
                    assert_eq!(graph.aval(*placeholder), operand, "{}", op.name());
                }
                assert!(
                    graph
                        .eqns
                        .iter()
                        .all(|eqn| eqn.op.class() == OpClass::Primitive),
                    "{}: a decomposition is primitive equations",
                    op.name()
                );
            }
        }
        let missing: Vec<usize> = (0..VARIANTS).filter(|&index| !covered[index]).collect();
        assert!(
            missing.is_empty(),
            "variants with no canonical row: {missing:?}"
        );
    }

    /// A composite whose operands do not type has no decomposition, rather than a panic in the
    /// tracer: the evaluator turns that into a typed refusal.
    #[test]
    fn an_ill_typed_composite_has_no_decomposition() {
        let rope = OpKind::Rope { rot: 4 };
        assert!(decompose(&rope, &[f32(&[1, 2, 3, 8]), f32(&[3, 6]), f32(&[3, 6])]).is_none());
        let i32_rope = [
            typed(&[3, 8], DType::I32),
            typed(&[3, 4], DType::I32),
            typed(&[3, 4], DType::I32),
        ];
        assert!(decompose(&rope, &i32_rope).is_none());
        let bias = [f32(&[2, 3]), f32(&[3, 4]), f32(&[1, 4])];
        assert!(decompose(&OpKind::MatMulBias, &bias).is_none());
    }
}
