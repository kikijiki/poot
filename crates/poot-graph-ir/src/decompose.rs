//! Composite ops defined by their decomposition (ADR-0101 tier 1, Card 556).
//!
//! A composite ([`OpClass::Composite`](crate::OpClass)) has no definition of its own: its semantics are
//! the primitive graph [`decompose`] builds, from the same `ops::` functions a tracer calls. The CPU
//! oracle evaluates that graph, so a pass that replaces a primitive chain with the composite preserves
//! the oracle's result bit for bit, and a kernel for the composite is compared with the decomposition at
//! tier 2. `OpKind::class() == Composite` exactly when [`decompose`] returns `Some` for the op's operand
//! types (the consistency row in this file's tests).

use crate::budget::TraceLimits;
use crate::builder::{Builder, BuilderAppendPlan, Traced};
use crate::error::BuilderError;
use crate::graph::{Graph, Operand, ValueId};
use crate::op::{
    ContractionHead, ContractionRows, FusedOperand, FusedRegion, KeyLayout, KvBound, MaskOperand,
    OpKind, RowRegion, RowStep, ScanFamily, SinkOperand,
};
use crate::ops;
use crate::packed_source::packed_source_constants;
use crate::types::{DType, TensorType};
use poot_quant::PackedWeight;

/// A composite's primitive definition: `graph` computes the composite's result as its output from the
/// constants `operands`, one per equation operand, in operand order. An evaluator binds the equation's
/// operand values to `operands` and evaluates `graph`.
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// The primitive graph; its single output is the composite's result.
    pub graph: Graph,
    /// The constants of `graph` standing for the composite's equation operands, one per operand in operand order.
    pub operands: Vec<ValueId>,
}

/// The decomposition of `op` over operands of types `operands`, or `None` when `op` is a primitive or
/// `op` does not type over `operands` (`OpKind::infer` refuses them).
///
/// Every operand is a [`Builder::constant`] placeholder; the composite's parameters are baked into the
/// primitive equations exactly as the tracer's `ops::` function bakes them:
///
/// - `Attention(attrs)`: the one attention composite, `ops::attention_over_keys` (the body of
///   [`ops::attention_masked_softcap`] and [`ops::attention_live`]), for every phase, key layout and bound.
/// - `Rope { rot }`: [`ops::rope_partial`] over `x`'s last axis.
/// - `ContractionEpilogue { head, region }`: the head op over its two operands, then the region's steps with
///   local 0 the head's result (the dtype the head declares: a narrow head has already rounded).
/// - `PackedContraction { descriptor, rows: Blocks(blocks) }`: `PackedDequant` of the carriers, then the `MatMul`
///   of `packed_graphs::packed_linear` (`blocks == 1`) or `packed_graphs::packed_block_diagonal_linear` (`blocks > 1`). An
///   `Indexed` one is a primitive: the oracle decodes only the experts the row ids select.
/// - `DenseContraction { .. }`: `MatMul(activation, Transpose([1, 0], w))`, the chain the dense fold replaces.
/// - `GroupedDenseContraction { groups, .. }`: `MatMul(activation, Transpose([0, 2, 1], Reshape([groups, N, K], w)))`,
///   the chain the grouped fold replaces.
/// - `DenseRowGather { .. }`: `Cast { to: F32 }` of `Gather { axis: 0 }`, the pair the row-gather fold
///   replaces.
/// - `Fused(region)` / `FusedRow(region)`: the region's steps as equations, operands in place (a
///   `PackLane` leaf is the unit `Slice` of its packed input, a non-empty `pack` the last-axis `Concat`).
///
/// `Ok(None)` is "this operation has no decomposition (or its operands do not type)". `Err` is a
/// failure to build the one that exists: a decomposition that does not fit `limits`, or a defect of the
/// definition itself.
pub fn decompose(
    op: &OpKind,
    operands: &[TensorType],
    limits: TraceLimits,
) -> Result<Option<Decomposition>, BuilderError> {
    let Ok(out) = op.infer(operands) else {
        return Ok(None);
    };
    let b = Builder::new(limits);
    // A packed carrier is bound by its role-carrying source name, which the evaluator checks.
    let carrier_names: Vec<String> = match op {
        OpKind::PackedContraction { descriptor, .. } => {
            packed_source_constants("operand", *descriptor)
                .into_iter()
                .map(|(name, _)| name.as_str().to_owned())
                .collect()
        }
        _ => Vec::new(),
    };
    let inputs: Vec<Traced> = operands
        .iter()
        .enumerate()
        .map(|(position, ty)| {
            let name = match position.checked_sub(1).and_then(|i| carrier_names.get(i)) {
                Some(carrier) => carrier.clone(),
                None => format!("operand.{position}"),
            };
            b.constant(&name, ty.clone())
        })
        .collect::<Result<_, _>>()?;
    let result = match op {
        OpKind::Attention(attrs) => {
            let Some((&[q, k, v, mask], extra)) = inputs.split_first_chunk::<4>() else {
                return Ok(None);
            };
            let (extra, sinks) = match (attrs.sink, extra.split_last()) {
                (SinkOperand::Absent, _) => (extra, None),
                (SinkOperand::PerHead, Some((&sinks, rest))) => (rest, Some(sinks)),
                (SinkOperand::PerHead, None) => return Ok(None),
            };
            let (extra_key, len) = match (attrs.bound, extra) {
                (KvBound::Capacity, []) => (None, None),
                (KvBound::Capacity, &[operand]) => (Some(operand), None),
                (KvBound::Live, &[len]) => (None, Some(len)),
                (KvBound::Live, &[operand, len]) => (Some(operand), Some(len)),
                _ => return Ok(None),
            };
            // The paged form is the dense one over the views its slot map gathers out of the pools.
            let (keys, v) = match (attrs.key, extra_key) {
                (KeyLayout::Single, None) => (ops::AttentionKeys::Single(k), v),
                (KeyLayout::Split { .. }, Some(shared)) => {
                    (ops::AttentionKeys::Split { head: k, shared }, v)
                }
                (KeyLayout::Paged, Some(read)) => (
                    ops::AttentionKeys::Single(ops::paged_view(&b, k, read)?),
                    ops::paged_view(&b, v, read)?,
                ),
                _ => return Ok(None),
            };
            // A causal mask is built from the positions over the key axis the scores have.
            let mask = match attrs.mask {
                MaskOperand::Additive => mask,
                MaskOperand::Causal { window } => {
                    let lk = b.aval(v)?.shape[2];
                    ops::causal_mask_from_pos(&b, mask, lk, window)?
                }
            };
            ops::attention_over_keys(
                &b,
                q,
                keys,
                v,
                attrs.n_rep,
                attrs.scale,
                mask,
                attrs.softcap,
                len,
                sinks,
            )?
        }
        OpKind::Rope { rot } => {
            let &[x, cos, sin] = inputs.as_slice() else {
                return Ok(None);
            };
            let shape = &operands[0].shape;
            let last = shape.len() - 1;
            ops::rope_partial(&b, x, cos, sin, shape[last], *rot, last)?
        }
        OpKind::SelectedAttention(attrs) => {
            let &[q, k, v, ids, pos] = inputs.as_slice() else {
                return Ok(None);
            };
            ops::selected_attention(
                &b,
                q,
                k,
                v,
                ids,
                pos,
                attrs.n_rep,
                attrs.scale,
                attrs.softcap,
                attrs.ratio,
            )?
        }
        OpKind::BlockTopK(attrs) => {
            let &[scores, pos] = inputs.as_slice() else {
                return Ok(None);
            };
            ops::block_top_k_packed(&b, scores, pos, attrs.ratio, attrs.k)?
        }
        OpKind::Scan(attrs) => {
            // The definition unrolls one chunk body per chunk; the CPU oracle never evaluates it whole but one
            // chunk window at a time (`poot_eval`), so the sequence length has no ceiling there.
            if attrs.layout(operands).is_err() {
                return Ok(None);
            }
            match (attrs.family, inputs.as_slice()) {
                (ScanFamily::DeltaRule, &[q, k, v, g, beta, s_in]) => {
                    ops::scan_delta_rule_packed(&b, q, k, v, g, beta, s_in, attrs.chunk)?
                }
                (ScanFamily::Ssd, &[x, b_in, c_in, delta, a, d, h_in]) => {
                    ops::scan_ssd_packed(&b, x, b_in, c_in, delta, a, d, h_in, attrs.chunk)?
                }
                _ => return Ok(None),
            }
        }
        OpKind::ContractionEpilogue { head, region } => {
            let (head_operands, extras) = inputs.split_at(head.operands());
            let staged = contraction_epilogue(&b, *head, region, head_operands, extras);
            match ill_typed_is_none(staged)? {
                Some(result) => result,
                None => return Ok(None),
            }
        }
        OpKind::Fused(region) => match ill_typed_is_none(fused(&b, region, &inputs))? {
            Some(result) => result,
            None => return Ok(None),
        },
        OpKind::FusedRow(region) => match ill_typed_is_none(fused_row(&b, region, &inputs))? {
            Some(result) => result,
            None => return Ok(None),
        },
        OpKind::PackedContraction {
            descriptor,
            rows: ContractionRows::Blocks(blocks),
        } => {
            let [activation, carriers @ ..] = inputs.as_slice() else {
                return Ok(None);
            };
            packed_contraction(&b, *descriptor, *blocks, *activation, carriers)?
        }
        OpKind::PackedContraction {
            rows: ContractionRows::Indexed { .. },
            ..
        } => return Ok(None),
        OpKind::DenseContraction { .. } => {
            let &[activation, weight] = inputs.as_slice() else {
                return Ok(None);
            };
            b.matmul(activation, b.transpose(weight, vec![1, 0])?)?
        }
        OpKind::GroupedDenseContraction { groups, .. } => {
            let &[activation, weight] = inputs.as_slice() else {
                return Ok(None);
            };
            let [rows, k] = b.aval(weight)?.shape[..] else {
                return Ok(None);
            };
            let blocks = b.reshape(weight, vec![*groups, rows / groups, k])?;
            b.matmul(activation, b.transpose(blocks, vec![0, 2, 1])?)?
        }
        OpKind::DenseRowGather { .. } => {
            let &[table, index] = inputs.as_slice() else {
                return Ok(None);
            };
            b.cast(b.gather(table, 0, index)?, DType::F32)?
        }
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
        | OpKind::PackedRowGather { .. }
        | OpKind::ArgTopK { .. }
        | OpKind::IndexedMatMul
        | OpKind::DynamicUpdateSlice { .. }
        | OpKind::PackI8
        | OpKind::UnpackI8 { .. }
        | OpKind::AllReduce { .. }
        | OpKind::AllGather { .. }
        | OpKind::RandomUniform { .. }
        | OpKind::SampleToken { .. } => return Ok(None),
    };
    // The composite's typing rule and its definition must agree, or the evaluator would publish a value
    // of another type than the equation declares.
    if b.aval(result)? != out {
        return Ok(None);
    }
    Ok(Some(Decomposition {
        operands: inputs.iter().map(|input| input.id()).collect(),
        graph: b.finish(result)?,
    }))
}

/// A fragment that does not stage because the composite is ill typed has no decomposition (`Ok(None)`);
/// a request the budget or an allocation refused is a failure the caller must see.
fn ill_typed_is_none(
    staged: Result<Option<Traced>, BuilderError>,
) -> Result<Option<Traced>, BuilderError> {
    match staged {
        Err(
            error @ (BuilderError::Budget(_)
            | BuilderError::Reserve { .. }
            | BuilderError::SizeOverflow { .. }
            | BuilderError::GenerationOverflow),
        ) => Err(error),
        Err(_) => Ok(None),
        Ok(result) => Ok(result),
    }
}

/// A packed contraction: the weight dequantized to `[out, k]` F32, then the activation contracted against
/// it. `blocks == 1` is `packed_graphs::packed_linear`'s chain, `MatMul(activation, Transpose([1, 0], w))`.
/// `blocks > 1` states `packed_graphs::packed_block_diagonal_linear`'s semantics: the weight is grouped into
/// `[blocks, out / blocks, k]` before the `[0, 2, 1]` transpose, so each `[blocks, M, k]` activation block
/// contracts only against its own weight block. The dequantized weight is a real intermediate (an
/// `[out, k]` F32 tensor), which the evaluator charges to its budget.
fn packed_contraction(
    b: &Builder,
    descriptor: PackedWeight,
    blocks: usize,
    activation: Traced,
    carriers: &[Traced],
) -> Result<Traced, BuilderError> {
    let weight = b.packed_dequant(carriers, descriptor)?;
    let [out, k] = descriptor.shape();
    let weight = if blocks == 1 {
        b.transpose(weight, vec![1, 0])?
    } else {
        let grouped = b.reshape(weight, vec![blocks, out / blocks, k])?;
        b.transpose(grouped, vec![0, 2, 1])?
    };
    b.matmul(activation, weight)
}

/// One region local as an equation operand: an input or step result, or a literal kept in place.
fn region_operand(
    plan: &mut BuilderAppendPlan,
    locals: &[Traced],
    operand: &FusedOperand,
) -> Result<Option<Operand>, BuilderError> {
    Ok(Some(match operand {
        FusedOperand::Local(local) => {
            let Some(value) = locals.get(*local) else {
                return Ok(None);
            };
            Operand::Value(value.id())
        }
        FusedOperand::Lit(scalar) => Operand::Lit(*scalar),
        FusedOperand::PackLane { input, lane } => {
            let Some(packed) = locals.get(*input).map(|value| value.id()) else {
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
            Operand::Value(plan.equation(slice, vec![Operand::Value(packed)])?.id())
        }
    }))
}

/// Stage the equations of a region whose steps `stage` emits, then commit them; `Ok(None)` when a
/// region operand names no local.
fn commit_region(
    b: &Builder,
    inputs: &[Traced],
    stage: impl FnOnce(&mut BuilderAppendPlan, &mut Vec<Traced>) -> Result<Option<Traced>, BuilderError>,
) -> Result<Option<Traced>, BuilderError> {
    let mut plan = b.append_plan(0)?;
    let mut locals = inputs.to_vec();
    let Some(result) = stage(&mut plan, &mut locals)? else {
        return Ok(None);
    };
    // A region whose output is one of its inputs has no equation to commit.
    if inputs.iter().any(|input| input.id() == result.id()) {
        return Ok(Some(result));
    }
    plan.declare_result(result)?;
    let mut prepared = b.preflight_append(plan)?;
    Ok(Some(b.commit_append(&mut prepared)?))
}

/// A pointwise region's steps as equations over `locals`, then its output local or its last-axis `pack`.
fn stage_region(
    plan: &mut BuilderAppendPlan,
    locals: &mut Vec<Traced>,
    region: &FusedRegion,
) -> Result<Option<Traced>, BuilderError> {
    for step in &region.steps {
        let mut operands = Vec::with_capacity(step.inputs.len());
        for operand in &step.inputs {
            let Some(operand) = region_operand(plan, locals, operand)? else {
                return Ok(None);
            };
            operands.push(operand);
        }
        locals.push(plan.equation(step.op.kind(), operands)?);
    }
    if region.pack.is_empty() {
        return Ok(locals.get(region.output).copied());
    }
    let mut parts = Vec::with_capacity(region.pack.len());
    for local in &region.pack {
        let Some(part) = locals.get(*local) else {
            return Ok(None);
        };
        parts.push(Operand::Value(part.id()));
    }
    let Some(axis) = plan
        .type_of(locals[region.pack[0]].id())
        .and_then(|ty| ty.rank().checked_sub(1))
    else {
        return Ok(None);
    };
    Ok(Some(plan.equation(OpKind::Concat { axis }, parts)?))
}

/// A pointwise region's steps as equations, then its output local or its last-axis `pack`.
fn fused(
    b: &Builder,
    region: &FusedRegion,
    inputs: &[Traced],
) -> Result<Option<Traced>, BuilderError> {
    commit_region(b, inputs, |plan, locals| stage_region(plan, locals, region))
}

/// A contraction epilogue: the head op over `head_operands`, then `region` over the head's result (local 0)
/// and the `extras` (locals `1..`). The head is an equation of the same plan, so the whole composite commits
/// as one append.
fn contraction_epilogue(
    b: &Builder,
    head: ContractionHead,
    region: &FusedRegion,
    head_operands: &[Traced],
    extras: &[Traced],
) -> Result<Option<Traced>, BuilderError> {
    let mut plan = b.append_plan(0)?;
    let operands = head_operands
        .iter()
        .map(|operand| Operand::Value(operand.id()))
        .collect();
    let mut locals = vec![plan.equation(head.op(), operands)?];
    locals.extend_from_slice(extras);
    let Some(result) = stage_region(&mut plan, &mut locals, region)? else {
        return Ok(None);
    };
    plan.declare_result(result)?;
    let mut prepared = b.preflight_append(plan)?;
    Ok(Some(b.commit_append(&mut prepared)?))
}

/// A row region's steps as equations: pointwise steps in place, each reduction a keepdim `Reduce` over
/// the region's row axis.
fn fused_row(
    b: &Builder,
    region: &RowRegion,
    inputs: &[Traced],
) -> Result<Option<Traced>, BuilderError> {
    commit_region(b, inputs, |plan, locals| {
        for step in &region.steps {
            let (op, step_operands) = match step {
                RowStep::Pointwise { op, inputs } => (op.kind(), inputs.as_slice()),
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
    use crate::op::{AttentionAttrs, BinOp, FusedOp, FusedStep, OpClass, RedOp, SampleRule, UnOp};
    use crate::types::{DType, Scalar};
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;

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
            OpKind::ContractionEpilogue { .. } => 20,
            OpKind::ArgTopK { .. } => 21,
            OpKind::IndexedMatMul => 22,
            OpKind::DynamicUpdateSlice { .. } => 23,
            OpKind::PackI8 => 24,
            OpKind::UnpackI8 { .. } => 25,
            OpKind::Fused(_) => 26,
            OpKind::FusedRow(_) => 27,
            OpKind::Attention(_) => 28,
            OpKind::Rope { .. } => 29,
            OpKind::Scan(_) => 34,
            OpKind::BlockTopK(_) => 36,
            OpKind::SelectedAttention(_) => 37,
            OpKind::GroupedDenseContraction { .. } => 35,
            OpKind::AllReduce { .. } => 30,
            OpKind::AllGather { .. } => 31,
            OpKind::RandomUniform { .. } => 32,
            OpKind::SampleToken { .. } => 33,
        }
    }

    const VARIANTS: usize = 38;

    /// The epilogue region `(head + bias) + residual`: local 0 the head result, 1 the bias, 2 the residual.
    fn bias_residual_region() -> FusedRegion {
        FusedRegion {
            n_inputs: 3,
            steps: vec![
                FusedStep {
                    op: FusedOp::Binary(BinOp::Add),
                    inputs: vec![FusedOperand::Local(0), FusedOperand::Local(1)],
                },
                FusedStep {
                    op: FusedOp::Binary(BinOp::Add),
                    inputs: vec![FusedOperand::Local(3), FusedOperand::Local(2)],
                },
            ],
            output: 4,
            pack: Vec::new(),
        }
    }

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
            (
                OpKind::Iota {
                    len: 4,
                    dtype: DType::F32,
                },
                Vec::new(),
            ),
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
                vec![f32(&[4, 3]), f32(&[2, 3]), typed(&[2], i32)],
            ),
            (OpKind::MatMul, vec![f32(&[2, 3]), f32(&[3, 4])]),
            (
                OpKind::PackedDequant { descriptor: packed },
                with_sources(Vec::new(), false),
            ),
            (
                OpKind::PackedContraction {
                    descriptor: packed,
                    rows: ContractionRows::Blocks(1),
                },
                with_sources(vec![f32(&[2, 32])], false),
            ),
            (
                OpKind::PackedContraction {
                    descriptor: packed,
                    rows: ContractionRows::Blocks(2),
                },
                with_sources(vec![f32(&[2, 3, 32])], false),
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
                OpKind::DenseContraction { weight: DType::F32 },
                vec![f32(&[2, 3, 4]), f32(&[5, 4])],
            ),
            (
                OpKind::DenseContraction { weight: DType::F16 },
                vec![f32(&[2, 4]), typed(&[3, 4], DType::F16)],
            ),
            (
                OpKind::GroupedDenseContraction {
                    weight: DType::BF16,
                    groups: 2,
                },
                vec![f32(&[2, 3, 4]), typed(&[10, 4], DType::BF16)],
            ),
            (
                OpKind::DenseRowGather {
                    source: DType::BF16,
                },
                vec![typed(&[5, 4], DType::BF16), typed(&[2], i32)],
            ),
            (
                OpKind::DenseRowGather {
                    source: DType::E4M3FN,
                },
                vec![typed(&[5, 4], DType::E4M3FN), typed(&[2, 3], i32)],
            ),
            (
                OpKind::ContractionEpilogue {
                    head: ContractionHead::MatMul,
                    region: bias_residual_region(),
                },
                vec![f32(&[2, 3]), f32(&[3, 4]), f32(&[4]), f32(&[2, 4])],
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
            (OpKind::Attention(AttentionAttrs::new(2, 0.5, None)), decode),
            (
                OpKind::Attention(AttentionAttrs::new(2, 0.5, None)),
                prefill.clone(),
            ),
            (
                OpKind::Attention(AttentionAttrs::new(2, 0.5, Some(30.0))),
                prefill,
            ),
            (
                // A per-head sink: the `[Hq]` logits are the last operand.
                OpKind::Attention(AttentionAttrs {
                    sink: crate::op::SinkOperand::PerHead,
                    ..AttentionAttrs::new(2, 0.5, None)
                }),
                vec![
                    f32(&[2, 4, 1, 8]),
                    f32(&[2, 2, 6, 8]),
                    f32(&[2, 2, 6, 8]),
                    f32(&[2, 1, 1, 6]),
                    f32(&[4]),
                ],
            ),
            (
                // A windowed causal mask: the fourth operand is the `[B, Lq]` I32 positions.
                OpKind::Attention(AttentionAttrs {
                    mask: MaskOperand::Causal { window: Some(3) },
                    ..AttentionAttrs::new(2, 0.5, None)
                }),
                vec![
                    f32(&[2, 4, 1, 8]),
                    f32(&[2, 2, 6, 8]),
                    f32(&[2, 2, 6, 8]),
                    typed(&[2, 1], i32),
                ],
            ),
            (
                // A paged key: two `[S, Hkv, D]` pools read through the `[B, Lk]` slot map.
                OpKind::Attention(AttentionAttrs {
                    key: KeyLayout::Paged,
                    ..AttentionAttrs::new(2, 0.5, None)
                }),
                vec![
                    f32(&[2, 4, 1, 8]),
                    f32(&[9, 2, 8]),
                    f32(&[9, 2, 8]),
                    f32(&[2, 1, 1, 6]),
                    typed(&[2, 6], i32),
                ],
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
                OpKind::Scan(crate::op::ScanAttrs {
                    family: ScanFamily::DeltaRule,
                    chunk: 4,
                }),
                vec![
                    f32(&[1, 2, 6, 3]),
                    f32(&[1, 2, 6, 3]),
                    f32(&[1, 2, 6, 3]),
                    f32(&[1, 2, 6, 1]),
                    f32(&[1, 2, 6, 1]),
                    f32(&[1, 2, 3, 3]),
                ],
            ),
            (
                OpKind::Scan(crate::op::ScanAttrs {
                    family: ScanFamily::Ssd,
                    chunk: 4,
                }),
                vec![
                    f32(&[1, 2, 7, 3]),
                    f32(&[1, 2, 7, 5]),
                    f32(&[1, 2, 7, 5]),
                    f32(&[1, 2, 7, 1]),
                    f32(&[1, 2, 1, 1]),
                    f32(&[1, 2, 1, 1]),
                    f32(&[1, 2, 5, 3]),
                ],
            ),
            (
                OpKind::SelectedAttention(crate::op::SelectedAttentionAttrs {
                    n_rep: 2,
                    scale: 0.5,
                    softcap: None,
                    ratio: 2,
                    k: 2,
                }),
                vec![
                    f32(&[2, 4, 1, 8]),
                    f32(&[2, 2, 6, 8]),
                    f32(&[2, 2, 6, 8]),
                    typed(&[2, 1, 1, 2], i32),
                    typed(&[2, 1], i32),
                ],
            ),
            (
                OpKind::BlockTopK(crate::op::BlockTopKAttrs { ratio: 2, k: 3 }),
                vec![f32(&[2, 1, 3, 6]), typed(&[2, 3], i32)],
            ),
            (
                OpKind::AllReduce {
                    op: RedOp::Sum,
                    axis: 0,
                },
                vec![f32(&[2])],
            ),
            (
                OpKind::AllGather {
                    axis: 0,
                    world: std::num::NonZeroU32::MIN,
                },
                vec![f32(&[2])],
            ),
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
            let decomposition = decompose(op, operands, TraceLimits::for_tests()).unwrap();
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
                assert_eq!(graph.aval(graph.result()), &declared, "{}", op.name());
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
                        .eqns()
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
        assert!(
            decompose(
                &rope,
                &[f32(&[1, 2, 3, 8]), f32(&[3, 6]), f32(&[3, 6])],
                TraceLimits::for_tests()
            )
            .unwrap()
            .is_none()
        );
        let i32_rope = [
            typed(&[3, 8], DType::I32),
            typed(&[3, 4], DType::I32),
            typed(&[3, 4], DType::I32),
        ];
        assert!(
            decompose(&rope, &i32_rope, TraceLimits::for_tests())
                .unwrap()
                .is_none()
        );
        // A bias of width 5 does not broadcast against the head's `[2, 4]` result.
        let epilogue = OpKind::ContractionEpilogue {
            head: ContractionHead::MatMul,
            region: bias_residual_region(),
        };
        let wide_bias = [f32(&[2, 3]), f32(&[3, 4]), f32(&[5]), f32(&[2, 4])];
        assert!(
            decompose(&epilogue, &wide_bias, TraceLimits::for_tests())
                .unwrap()
                .is_none()
        );
    }
}
