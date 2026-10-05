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
        vec![Operand::Value(weight.id)],
    )?;
    let weight = match alias_weight_shape {
        Some(shape) => plan.equation(
            crate::op::OpKind::Reshape { shape },
            vec![Operand::Value(weight.id)],
        )?,
        None => weight,
    };
    let output = plan.equation(
        crate::op::OpKind::MatMul,
        vec![Operand::Value(x.id), Operand::Value(weight.id)],
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
        vec![Operand::Value(table.id), Operand::Value(ids.id)],
    )?;
    finish_packed_plan(b, plan, rows)
}

/// Block-diagonal packed linear (Card 385): the stored `[out, k]` pair is grouped into `blocks`
/// contiguous row-blocks before the canonical transpose, so each activation block contracts only
/// against its own weight block. `packed_linear` cannot express this: the grouped view needs a
/// reshape (`[blocks, out/blocks, k]`) and then a 3-axis transpose (`[0,2,1]`) to bring `k`
/// innermost. `x` is `[blocks, m, k]`; batched-matmul broadcast gives each block its own
/// `[m, out/blocks]` slab, so no new primitive is needed. [`crate::op::OpKind::PackedContraction`]'s
/// `blocks` field recognizes the folded form for lowering.
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
        vec![Operand::Value(weight.id)],
    )?;
    let weight = plan.equation(
        crate::op::OpKind::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![Operand::Value(weight.id)],
    )?;
    let output = plan.equation(
        crate::op::OpKind::MatMul,
        vec![Operand::Value(x.id), Operand::Value(weight.id)],
    )?;
    let output = match bias {
        Some(bias) => plan_binary(&mut plan, BinOp::Add, output, bias)?,
        None => output,
    };
    finish_packed_plan(b, plan, output)
}

/// One deterministic expert-table row for the canonical packed indexed/grouped builders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearGraphRow {
    pub ordinal: usize,
    pub linear_id: String,
    pub descriptor: poot_quant::PackedWeight,
}

/// Build the exact packed branch table and ordinary `IndexedMatMul` carrier recognized by Card 356.
pub fn packed_indexed_linear(
    b: &Builder,
    x: Traced,
    selector: Traced,
    rows: &[PackedLinearGraphRow],
) -> Result<Traced, BuilderAppendError> {
    validate_packed_rows(rows)?;
    let mut plan = b.append_plan(rows.len());
    validate_indexed_inputs(&plan, x, selector, rows[0].descriptor)?;
    let output = stage_packed_indexed_linear(&mut plan, x, selector, rows)?;
    finish_packed_plan(b, plan, output)
}

/// Wrap the canonical indexed carrier in Card 356's stable sort, paired scatters, per-expert offsets,
/// and inverse gather.
///
/// The row iota (`0..M`) and expert iota (`0..=E`) are computed with `iota` (card 558a), not bound
/// constants. The final row-restoring `Gather` depends only on `row_iota`/`selector`. Nothing reads
/// `expert_iota`'s `_offsets` reduction: the packed-block-float plan that recognized it was deleted
/// (card 573b), and `compile`'s DCE drops it. Card 388 tracks giving `_offsets` a consumer or removing
/// it.
pub fn packed_grouped_linear(
    b: &Builder,
    x: Traced,
    selector: Traced,
    rows: &[PackedLinearGraphRow],
) -> Result<Traced, BuilderAppendError> {
    const F32_EXACT_INT_MAX: usize = 1 << 24;

    validate_packed_rows(rows)?;
    let mut plan = b.append_plan(rows.len());
    validate_indexed_inputs(&plan, x, selector, rows[0].descriptor)?;
    let x_type = plan
        .type_of(x.id)
        .cloned()
        .ok_or(BuilderAppendError::InvalidOperand {
            equation: 0,
            operation: "packed_grouped_linear".to_string(),
            value: x.id,
        })?;
    let m = x_type.shape[0];
    let e = rows.len();
    for (field, value) in [("row count", m), ("expert count", e)] {
        if value > F32_EXACT_INT_MAX {
            return Err(BuilderAppendError::F32ExactIntegerRange {
                field,
                value,
                max: F32_EXACT_INT_MAX,
            });
        }
    }
    let expert_iota_len = e.checked_add(1).ok_or(BuilderAppendError::SizeOverflow {
        collection: BuilderCollection::Values,
        current: e,
        additional: 1,
    })?;
    // The two ranges are computed, not bound: `Iota` folds to a compiler-computed constant before
    // planning, so no named per-expert-range binder supplies them (card 558a).
    let row_iota = plan.equation(crate::op::OpKind::Iota { len: m }, Vec::new())?;
    let expert_iota = plan.equation(
        crate::op::OpKind::Iota {
            len: expert_iota_len,
        },
        Vec::new(),
    )?;

    let a = plan.equation(
        crate::op::OpKind::Reshape { shape: vec![m, 1] },
        vec![Operand::Value(selector.id)],
    )?;
    let a = plan.equation(
        crate::op::OpKind::Broadcast { shape: vec![m, m] },
        vec![Operand::Value(a.id)],
    )?;
    let bb = plan.equation(
        crate::op::OpKind::Reshape { shape: vec![1, m] },
        vec![Operand::Value(selector.id)],
    )?;
    let bb = plan.equation(
        crate::op::OpKind::Broadcast { shape: vec![m, m] },
        vec![Operand::Value(bb.id)],
    )?;
    let ge_ba = plan_binary(&mut plan, BinOp::Ge, bb, a)?;
    let less_e = plan_binary_scalar(&mut plan, BinOp::Mul, ge_ba, Scalar::F32(-1.0))?;
    let less_e = plan_binary_scalar(&mut plan, BinOp::Add, less_e, Scalar::F32(1.0))?;
    let ge_ab = plan_binary(&mut plan, BinOp::Ge, a, bb)?;
    let eq = plan_binary(&mut plan, BinOp::Mul, ge_ab, ge_ba)?;
    let rm = plan.equation(
        crate::op::OpKind::Reshape { shape: vec![m, 1] },
        vec![Operand::Value(row_iota.id)],
    )?;
    let rm = plan.equation(
        crate::op::OpKind::Broadcast { shape: vec![m, m] },
        vec![Operand::Value(rm.id)],
    )?;
    let rp = plan.equation(
        crate::op::OpKind::Reshape { shape: vec![1, m] },
        vec![Operand::Value(row_iota.id)],
    )?;
    let rp = plan.equation(
        crate::op::OpKind::Broadcast { shape: vec![m, m] },
        vec![Operand::Value(rp.id)],
    )?;
    let ge_rp_rm = plan_binary(&mut plan, BinOp::Ge, rp, rm)?;
    let less_row = plan_binary_scalar(&mut plan, BinOp::Mul, ge_rp_rm, Scalar::F32(-1.0))?;
    let less_row = plan_binary_scalar(&mut plan, BinOp::Add, less_row, Scalar::F32(1.0))?;
    let eq_and_row = plan_binary(&mut plan, BinOp::Mul, eq, less_row)?;
    let contrib = plan_binary(&mut plan, BinOp::Add, less_e, eq_and_row)?;
    let perm = plan.equation(
        crate::op::OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false,
        },
        vec![Operand::Value(contrib.id)],
    )?;
    let sorted_x = plan.equation(
        crate::op::OpKind::Scatter { axis: 0 },
        vec![Operand::Value(x.id), Operand::Value(perm.id)],
    )?;
    let sorted_selector = plan.equation(
        crate::op::OpKind::Scatter { axis: 0 },
        vec![Operand::Value(selector.id), Operand::Value(perm.id)],
    )?;

    let idx_b = plan.equation(
        crate::op::OpKind::Reshape { shape: vec![1, m] },
        vec![Operand::Value(selector.id)],
    )?;
    let idx_b = plan.equation(
        crate::op::OpKind::Broadcast {
            shape: vec![expert_iota_len, m],
        },
        vec![Operand::Value(idx_b.id)],
    )?;
    let expert_b = plan.equation(
        crate::op::OpKind::Reshape {
            shape: vec![expert_iota_len, 1],
        },
        vec![Operand::Value(expert_iota.id)],
    )?;
    let expert_b = plan.equation(
        crate::op::OpKind::Broadcast {
            shape: vec![expert_iota_len, m],
        },
        vec![Operand::Value(expert_b.id)],
    )?;
    // Card 388: `_offsets[e] = count(selector < expert_iota[e])`, the cumulative row count per expert
    // boundary. This graph never reads it; `perm` is derived from `row_iota`/`selector` alone.
    let ge_ie = plan_binary(&mut plan, BinOp::Ge, idx_b, expert_b)?;
    let lt_ie = plan_binary_scalar(&mut plan, BinOp::Mul, ge_ie, Scalar::F32(-1.0))?;
    let lt_ie = plan_binary_scalar(&mut plan, BinOp::Add, lt_ie, Scalar::F32(1.0))?;
    let _offsets = plan.equation(
        crate::op::OpKind::Reduce {
            op: RedOp::Sum,
            axis: 1,
            keepdim: false,
        },
        vec![Operand::Value(lt_ie.id)],
    )?;

    let sorted = stage_packed_indexed_linear(&mut plan, sorted_x, sorted_selector, rows)?;
    let output = plan.equation(
        crate::op::OpKind::Gather { axis: 0 },
        vec![Operand::Value(sorted.id), Operand::Value(perm.id)],
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
    let id = b.commit_append(&mut prepared)?;
    Ok(Traced { id })
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
        sources.push(Operand::Value(source.id));
    }
    plan.equation(crate::op::OpKind::PackedDequant { descriptor }, sources)
}

pub(crate) fn stage_packed_indexed_linear(
    plan: &mut BuilderAppendPlan,
    x: Traced,
    selector: Traced,
    rows: &[PackedLinearGraphRow],
) -> Result<Traced, BuilderAppendError> {
    let [out, k] = rows[0].descriptor.shape();
    let mut branches = Vec::with_capacity(rows.len());
    for row in rows {
        let weight = stage_packed_dequant(plan, &row.linear_id, row.descriptor)?;
        let weight = plan.equation(
            crate::op::OpKind::Transpose { perm: vec![1, 0] },
            vec![Operand::Value(weight.id)],
        )?;
        branches.push(plan.equation(
            crate::op::OpKind::Reshape {
                shape: vec![1, k, out],
            },
            vec![Operand::Value(weight.id)],
        )?);
    }
    let weights = plan.equation(
        crate::op::OpKind::Concat { axis: 0 },
        branches
            .iter()
            .map(|branch| Operand::Value(branch.id))
            .collect(),
    )?;
    plan.equation(
        crate::op::OpKind::IndexedMatMul,
        vec![
            Operand::Value(x.id),
            Operand::Value(weights.id),
            Operand::Value(selector.id),
        ],
    )
}

pub(crate) fn validate_packed_rows(
    rows: &[PackedLinearGraphRow],
) -> Result<(), BuilderAppendError> {
    if rows.is_empty() {
        return Err(BuilderAppendError::EmptyPackedRows);
    }
    let mut ids = std::collections::HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        if row.ordinal != index {
            return Err(BuilderAppendError::PackedRowOrdinal {
                index,
                expected: index,
                actual: row.ordinal,
            });
        }
        if row.linear_id.is_empty() {
            return Err(BuilderAppendError::EmptyPackedLinearId { index });
        }
        if let Some(first) = ids.insert(row.linear_id.as_str(), index) {
            return Err(BuilderAppendError::DuplicatePackedLinearId {
                linear_id: row.linear_id.clone(),
                first,
                second: index,
            });
        }
        if row.descriptor != rows[0].descriptor {
            return Err(BuilderAppendError::PackedDescriptorMismatch { index });
        }
    }
    Ok(())
}

pub(crate) fn validate_indexed_inputs(
    plan: &BuilderAppendPlan,
    x: Traced,
    selector: Traced,
    descriptor: poot_quant::PackedWeight,
) -> Result<(), BuilderAppendError> {
    let x_type = plan
        .type_of(x.id)
        .cloned()
        .ok_or(BuilderAppendError::InvalidOperand {
            equation: 0,
            operation: "packed_indexed_linear".to_string(),
            value: x.id,
        })?;
    let [_, k] = descriptor.shape();
    if x_type.dtype != DType::F32 || x_type.shape.len() != 2 || x_type.shape[1] != k {
        return Err(BuilderAppendError::OperandType {
            value: x.id,
            expected: TensorType::f32(vec![x_type.shape.first().copied().unwrap_or(0), k]),
            actual: x_type,
        });
    }
    require_operand_type(plan, selector, TensorType::f32(vec![x_type.shape[0]]))
}

pub(crate) fn require_operand_type(
    plan: &BuilderAppendPlan,
    value: Traced,
    expected: TensorType,
) -> Result<(), BuilderAppendError> {
    let actual = plan
        .type_of(value.id)
        .cloned()
        .ok_or(BuilderAppendError::InvalidOperand {
            equation: 0,
            operation: "packed_linear operand".to_string(),
            value: value.id,
        })?;
    if actual != expected {
        return Err(BuilderAppendError::OperandType {
            value: value.id,
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
        vec![Operand::Value(left.id), Operand::Value(right.id)],
    )
}

pub(crate) fn plan_binary_scalar(
    plan: &mut BuilderAppendPlan,
    op: BinOp,
    value: Traced,
    scalar: Scalar,
) -> Result<Traced, BuilderAppendError> {
    plan.equation(
        crate::op::OpKind::Binary(op),
        vec![Operand::Value(value.id), Operand::Lit(scalar)],
    )
}

/// LoRA-adapted linear: `linear(x, w, bias) + scaling * (x @ lora_a) @ lora_b`, the base projection
/// plus a low-rank correction (PEFT's `W' = W + scaling*B@A` in poot's `x @ W` convention).
///
/// `lora_a` is `[in, r]` and `lora_b` is `[r, out]`, already transposed to the matmul layout by
/// `poot_load::lora` at load time (HF/PEFT store `lora_A.weight [r,in]`, `lora_B.weight [out,r]`).
/// `scaling` is PEFT's `lora_alpha / r` (or `lora_alpha / sqrt(r)` for rslora; see
/// `poot_load::lora::LoraAdapterConfig::scaling`).
///
/// Composition of `linear` plus two small `MatMul`s, no new `OpKind` (spec 248). Per-row multi-adapter
/// routing is [`lora_linear_batched`].
pub fn lora_linear(
    b: &Builder,
    x: Traced,
    w: Traced,
    bias: Option<Traced>,
    lora_a: Traced,
    lora_b: Traced,
    scaling: f32,
) -> Traced {
    let base = linear(b, x, w, bias);
    let xa = b.matmul(x, lora_a); // [..,r]
    let xab = b.matmul(xa, lora_b); // [..,out]
    let correction = b.binary_scalar(BinOp::Mul, xab, Scalar::F32(scaling));
    b.binary(BinOp::Add, base, correction)
}

/// Batched multi-adapter LoRA linear (spec 248 Phase 2): `M` rows in one dispatch, each row `m` with
/// its own adapter id `idx[m]`. Reuses [`Builder::indexed_matmul`] twice (once per low-rank factor),
/// so there is no new `OpKind` and no gather/copy of the selected adapters' weights.
///
/// Shapes (all rank-2, matching `IndexedMatMul`'s `x[M,K], W[E,K,N], idx[M] -> [M,N]` contract):
/// - `x [M, in]`: one row per request.
/// - `w [in, out]`, `bias [out]?`: the shared base weight; only the LoRA correction is per-row routed.
/// - `lora_a_stacked [n_adapters, in, r]`, `lora_b_stacked [n_adapters, r, out]`: every adapter's
///   `A`/`B` stacked on a new leading axis and zero-padded to a common `r` (see
///   `poot_load::lora::LoraAdapterPool`; padding is exact). Pool index 0 is reserved as the "no
///   adapter" sentinel: an all-zero `A` makes `x @ A_stacked[0] = 0`, so the correction is inert
///   regardless of `B_stacked[0]` or `scaling_vec[0]`.
/// - `scaling_vec [n_adapters]`: each adapter's own `lora_alpha/r` (or rslora `lora_alpha/sqrt(r)`).
/// - `idx [M]` (f32, `IndexedMatMul`'s convention): the adapter id per row, used for both
///   `indexed_matmul` calls (a row's `A` and `B` come from the same adapter) and for the
///   `scaling_vec` gather.
///
/// `out[m,:] = x[m,:]@w + bias + scaling_vec[idx[m]] * (x[m,:]@A_stacked[idx[m]])@B_stacked[idx[m]]`.
/// Row `m` depends only on `x[m,:]` and `idx[m]` (see
/// `lora_linear_batched_matches_sequential_single_adapter_dispatch` in `poot-eval`).
#[allow(clippy::too_many_arguments)]
pub fn lora_linear_batched(
    b: &Builder,
    x: Traced,
    w: Traced,
    bias: Option<Traced>,
    lora_a_stacked: Traced,
    lora_b_stacked: Traced,
    scaling_vec: Traced,
    idx: Traced,
) -> Traced {
    let base = linear(b, x, w, bias); // [M, out]
    let xa = b.indexed_matmul(x, lora_a_stacked, idx); // [M, r]: row m against A_stacked[idx[m]]
    let xab = b.indexed_matmul(xa, lora_b_stacked, idx); // [M, out]: row m against B_stacked[idx[m]], SAME idx
    let m = b.aval(x).shape[0];
    let scale_row = b.gather(scaling_vec, 0, idx); // [M]: row m's own adapter's scaling
    let scale_col = b.reshape(scale_row, vec![m, 1]); // [M,1], broadcasts against [M,out]
    let correction = b.binary(BinOp::Mul, xab, scale_col);
    b.binary(BinOp::Add, base, correction)
}
