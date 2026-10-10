use super::*;

/// Linear: matmul against a `[in, out]` weight, with an optional `[out]` bias broadcast-add. `compile`'s
/// contraction-epilogue fuse rule folds the add into the matmul kernel (Card 557); the trace states only
/// the primitives.
pub fn linear(b: &Builder, x: Traced, w: Traced, bias: Option<Traced>) -> Traced {
    let y = b.matmul(x, w);
    match bias {
        Some(bias) => b.binary(BinOp::Add, y, bias),
        None => y,
    }
}

/// Shared rank-2 packed linear composition. Bias remains an ordinary Binary(Add) outside
/// the packed-contraction candidate.
pub fn packed_linear(
    b: &Builder,
    x: Traced,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
    alias_weight_shape: Option<Vec<usize>>,
    bias: Option<Traced>,
) -> Result<Traced, BuilderAppendError> {
    if linear_id.is_empty() {
        return Err(BuilderAppendError::EmptyPackedLinearId { index: 0 });
    }
    let mut plan = b.append_plan(0);
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
/// packed `[rows, K]` weight (a quantized GGUF `token_embd`). `ids` is any-shape F32/I32 row ids; the
/// result is `ids.shape ++ [K]`. `compile` claims the `PackedDequant -> Gather` pair as one
/// `PackedRowGather`, so only the gathered rows are ever decoded.
pub fn packed_embedding(
    b: &Builder,
    ids: Traced,
    linear_id: &str,
    descriptor: poot_quant::PackedWeight,
) -> Result<Traced, BuilderAppendError> {
    if linear_id.is_empty() {
        return Err(BuilderAppendError::EmptyPackedLinearId { index: 0 });
    }
    let mut plan = b.append_plan(0);
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
) -> Result<Traced, BuilderAppendError> {
    if linear_id.is_empty() {
        return Err(BuilderAppendError::EmptyPackedLinearId { index: 0 });
    }
    let [out, k] = descriptor.shape();
    if blocks == 0 || out % blocks != 0 {
        return Err(BuilderAppendError::PackedBlockDiagonalBlocks { out, blocks });
    }
    let block_out = out / blocks;
    let mut plan = b.append_plan(0);
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
/// carrier constant per source role named after `stack_id` (the stack's own `WeightId::const_name`), each
/// `[experts, slab_bytes]` ([`crate::packed_source::packed_stack_source_type`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedExpertStack {
    pub stack_id: String,
    pub descriptor: poot_quant::PackedWeight,
    pub experts: usize,
}

/// A routed packed projection: `x [M, K]` contracted per row against the expert `selector [M]` names in
/// `table`, `[M, out]`. One claimed [`crate::op::OpKind::PackedContraction`] with
/// [`crate::op::ContractionRows::Indexed`] over one stacked carrier per source role: no expert is dequantized
/// outside the contraction, and a row reads only its own expert's blocks.
pub fn packed_indexed_linear(
    b: &Builder,
    x: Traced,
    selector: Traced,
    table: &PackedExpertStack,
) -> Result<Traced, BuilderAppendError> {
    if table.stack_id.is_empty() {
        return Err(BuilderAppendError::EmptyPackedLinearId { index: 0 });
    }
    let mut plan = b.append_plan(0);
    let [_, k] = table.descriptor.shape();
    let x_type = plan
        .type_of(x.id())
        .cloned()
        .ok_or(BuilderAppendError::InvalidOperand {
            equation: 0,
            operation: "packed_indexed_linear".to_string(),
            value: x.id(),
        })?;
    if x_type.dtype != DType::F32 || x_type.shape.len() != 2 || x_type.shape[1] != k {
        return Err(BuilderAppendError::OperandType {
            value: x.id(),
            expected: TensorType::f32(vec![x_type.shape.first().copied().unwrap_or(0), k]),
            actual: x_type,
        });
    }
    require_operand_type(&plan, selector, TensorType::f32(vec![x_type.shape[0]]))?;
    let mut operands = vec![Operand::Value(x.id()), Operand::Value(selector.id())];
    for (name, tensor_type) in crate::packed_source::packed_stack_source_constants(
        &table.stack_id,
        table.descriptor,
        table.experts,
    ) {
        let carrier = plan.input(name, tensor_type, Storage::Const)?;
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
) -> Result<Traced, BuilderAppendError> {
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
) -> Result<Traced, BuilderAppendError> {
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
) -> Result<(), BuilderAppendError> {
    let actual = plan
        .type_of(value.id())
        .cloned()
        .ok_or(BuilderAppendError::InvalidOperand {
            equation: 0,
            operation: "packed_linear operand".to_string(),
            value: value.id(),
        })?;
    if actual != expected {
        return Err(BuilderAppendError::OperandType {
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
) -> Result<Traced, BuilderAppendError> {
    plan.equation(
        crate::op::OpKind::Binary(op),
        vec![Operand::Value(left.id()), Operand::Value(right.id())],
    )
}
