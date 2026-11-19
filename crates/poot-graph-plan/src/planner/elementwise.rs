//! Elementwise, reduction, fused-region and fused-RoPE equation planning.

use super::{
    BinOp, BodyLimits, Capability, ContractionHead, DType, Eqn, GBinOp, GUnOp, KernelRequest,
    Lowering, OpKind, Operand, PlanError, Planned, PointwiseForm, PointwiseSpec, RedOp, RopeSpec,
    RowSpec, RowStep, Scalar, ValueId, ViewOperand, fty_ty, kg, value_ids,
};
use crate::Views;
use crate::dtype_widen::ConstLanes;
use crate::kernel_mapping::{
    check_widening_step, float_region_leaf_storage, map_fused_input, map_fused_input_i32,
    map_fused_local, map_fused_op, map_fused_op_i32, map_mixed_input_i32, map_pointwise_step,
    map_row_value, map_unary, narrowing_store_op,
};
use crate::planner::{GraphTables, Layout, StorageDtypes};
use crate::predicates::{LayoutDemand, elementwise_fold, row_parallel_width};
use crate::refusal::Site;
use poot_graph_ir::LocalId;
use poot_graph_ir::op::{FusedOp, FusedOperand};

/// Whether this family lowers `op` over these storage dtypes. A packed E4M3FN payload is never an
/// elementwise operand; a 2-byte float output is wired for the pointwise and reduce bodies but not for the
/// fused-region, fused-row and RoPE bodies, which compute over f32 storage.
pub(super) fn admits_dtypes(op: &OpKind, dtypes: StorageDtypes) -> Result<(), Capability> {
    if dtypes.touches_e4m3fn() {
        return Err(Capability::E4m3Storage);
    }
    match op {
        // A row or pointwise region whose last step narrows its f32 result stores a two-byte float (POOT-1448); its
        // lowering decides whether the target has the store.
        OpKind::FusedRow(region) if region.narrowed_output() == Some(dtypes.out) => Ok(()),
        OpKind::Fused(region) if region.narrowed_output() == Some(dtypes.out) => Ok(()),
        OpKind::Rope { .. } | OpKind::Fused(_) | OpKind::FusedRow(_) if dtypes.out_is_half() => {
            Err(Capability::DtypeLowering)
        }
        _ => Ok(()),
    }
}

/// What this family needs of the layout it reads operand `position` (the value `value`) through. Every pointwise
/// op reads its leaves through their layouts (the pointwise synthesizer, `fused_views`): an elementwise `Binary`,
/// a `Unary`, a `Fused` region of f32 or exact-I32 leaves, a `FusedRow` of f32 leaves. A two-byte leaf, a
/// `Select` and a `Rope` read flat, and so does a reduction.
pub(super) fn layout_demand(g: GraphTables<'_>, eqn: &Eqn, value: ValueId) -> LayoutDemand {
    let dtype = g.aval(value).dtype;
    let words = matches!(dtype, DType::F32 | DType::I32);
    match &eqn.op {
        OpKind::Binary(_) | OpKind::Unary(_) | OpKind::Fused(_) if words => LayoutDemand::Any,
        OpKind::FusedRow(_) if dtype == DType::F32 => LayoutDemand::Any,
        _ => LayoutDemand::Dense,
    }
}

/// The element type a one-step pointwise kernel reads its leaves as: the equation's own dtype (an exact-I32
/// word, an F32, or a two-byte float computed in F32 and rounded back). Any other dtype has no lowering.
fn leaf_ty_of(site: Site<'_>, dtype: DType) -> Result<kg::LeafTy, PlanError> {
    match dtype {
        DType::F32 => Ok(kg::LeafTy::F32),
        DType::I32 => Ok(kg::LeafTy::I32),
        DType::BF16 => Ok(kg::LeafTy::BF16),
        DType::F16 => Ok(kg::LeafTy::F16),
        _ => Err(site.refuse(Capability::DtypeLowering)),
    }
}

/// Plan one elementwise-family equation (Binary/Select/Unary/Rope/Reduce/Fused/FusedRow).
#[expect(
    clippy::too_many_arguments,
    reason = "a per-op planner arm: the graph, equation and output facts plus the lowering context (backend, device caps, limits, pins, constant lanes) every arm receives alike from the dispatch in `planner.rs`"
)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    lowering: Lowering<'_>,
    views: &Views,
    odt: DType,
    limits: BodyLimits,
    const_lanes: &ConstLanes,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, lowering.id(), limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let layout_of = |vid: ValueId| -> Layout {
        // The residual layout: the dispatch binds the operand's root at the view's own `bound`.
        views
            .get(&vid)
            .map(|view| view.layout.clone())
            .unwrap_or_else(|| Layout::contiguous(&shape(vid)))
    };
    // Every input read through its layout (a strided view, or the row-major default).
    let leaves = |ids: &[ValueId]| -> Vec<ViewOperand> {
        ids.iter()
            .map(|&v| ViewOperand {
                shape: shape(v),
                layout: layout_of(v),
            })
            .collect()
    };
    let ids = value_ids(eqn);
    let fty = fty_ty;
    // The launch decision (card 159, card 181 Bug 2): past the grid cap at the narrow workgroup, a
    // one-thread-per-element kernel runs at the wide one, and past it at the wide one (`is_elementwise_2d`)
    // it folds onto a 2-D grid, because a flat X dispatch cannot clear wgpu's `gridDim.x` cap however large
    // `workgroup_size` gets. The fold carries the width, so `generate` bakes the body's workgroup and derives
    // the matching `x_groups` and grid from the same decision.
    let fold = match lowering {
        Lowering::Gpu { caps, .. } => elementwise_fold(g, eqn, out_shape, out_numel, caps),
        // A sequential loop has no grid to fold onto.
        Lowering::Aie(_) => kg::Fold::flat(out_numel),
    };
    let generate = |request: KernelRequest| Planned::lowered(site, request, lowering);
    let planned = match &eqn.op {
        OpKind::Binary(bop) => {
            match eqn.inputs.as_slice() {
                [Operand::Value(a), Operand::Value(b)] if g.aval(*a).dtype != g.aval(*b).dtype => {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
                [Operand::Value(_), Operand::Value(_) | Operand::Lit(_)] => {}
                [Operand::Lit(_), _] => {
                    return Err(site.refuse(Capability::LiteralFirstOperand));
                }
                inputs => {
                    return Err(site.refuse(Capability::OperandCount {
                        expected: 2,
                        actual: inputs.len(),
                    }));
                }
            }
            let exact_i32 = odt == DType::I32;
            if odt == DType::I32 && *bop == GBinOp::Div {
                return Err(site.refuse(Capability::ExactI32Op));
            }
            if odt == DType::I32
                && *bop == GBinOp::RemU
                && matches!(eqn.inputs.as_slice(), [_, Operand::Lit(Scalar::I32(0))])
            {
                return Err(site.refuse(Capability::ExactI32Op));
            }
            // The equation is one pointwise step: `op` over the first operand and the second, which is another
            // leaf (broadcast into the output, each read through its layout) or an inline literal (card 181 Bug 2:
            // the scalar form, e.g. the attention-score `* scale`, folds onto the same 2-D grid).
            let op = if exact_i32 {
                map_fused_op_i32(site, FusedOp::Binary(*bop))?
            } else {
                map_fused_op(site, FusedOp::Binary(*bop))?
            };
            let leaf_ty = leaf_ty_of(site, odt)?;
            let (second, leaf_count) = match &eqn.inputs[1] {
                Operand::Lit(Scalar::F32(_)) if exact_i32 => {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
                Operand::Lit(Scalar::F32(v)) => (kg::FusedInput::Lit(*v), 1),
                Operand::Lit(Scalar::I32(v)) if exact_i32 => (kg::FusedInput::LitI32(*v), 1),
                Operand::Lit(Scalar::I32(v)) => (kg::FusedInput::Lit(*v as f32), 1),
                Operand::Value(_) => (kg::FusedInput::Leaf(kg::LeafId::new(1)), 2),
            };
            // Layout safety (spec 132): each operand's strides and offset are baked into the generated body,
            // so they are part of the request: two equations with the same shapes but a different source
            // layout (one a plain buffer, one a view) are different kernels.
            generate(KernelRequest::Pointwise(PointwiseSpec {
                out_shape: out_shape.to_vec(),
                numel: out_numel,
                leaves: leaves(&ids[..leaf_count]),
                kernel: kg::FusedKernel::one_step(
                    op,
                    vec![leaf_ty; leaf_count],
                    vec![kg::FusedInput::Leaf(kg::LeafId::new(0)), second],
                ),
                form: if exact_i32 {
                    PointwiseForm::ExactI32 { fold }
                } else {
                    PointwiseForm::Float { fold }
                },
            }))?
        }
        OpKind::Select => {
            // The condition leaf is an I32 word; the branches are I32 (exact-I32 form), F32 or a two-byte
            // float (float form). A two-byte branch is moved as stored, so it needs a native two-byte lane:
            // SPIR-V has none (its BF16 and F16 values are packed words), so it keeps the typed refusal.
            let branch = match odt {
                DType::I32 => kg::LeafTy::I32,
                DType::F32 => kg::LeafTy::F32,
                // SPIR-V has no native two-byte lane, and neither has an AIE core.
                DType::BF16 | DType::F16
                    if matches!(
                        lowering,
                        Lowering::Gpu { caps, .. } if caps.native_two_byte_floats
                    ) =>
                {
                    if odt == DType::BF16 {
                        kg::LeafTy::BF16
                    } else {
                        kg::LeafTy::F16
                    }
                }
                _ => return Err(site.refuse(Capability::DtypeLowering)),
            };
            // A value operand is a leaf of the kernel; a branch literal is inline in the step. The condition is
            // always a leaf: a literal condition is a constant branch, which the planner's grammar has no
            // kernel for.
            let mut leaf_tys = vec![kg::LeafTy::I32];
            let mut select_inputs = vec![kg::FusedInput::Leaf(kg::LeafId::new(0))];
            let (condition, branches) = eqn.inputs.split_first().ok_or_else(|| {
                site.refuse(Capability::OperandCount {
                    expected: 3,
                    actual: 0,
                })
            })?;
            if matches!(condition, Operand::Lit(_)) {
                return Err(site.refuse(Capability::LiteralFirstOperand));
            }
            for operand in branches {
                select_inputs.push(match operand {
                    Operand::Value(_) => {
                        leaf_tys.push(branch);
                        kg::FusedInput::Leaf(kg::LeafId::new(leaf_tys.len() - 1))
                    }
                    Operand::Lit(Scalar::I32(word)) if odt == DType::I32 => {
                        kg::FusedInput::LitI32(*word)
                    }
                    Operand::Lit(Scalar::F32(value)) if odt == DType::F32 => {
                        kg::FusedInput::Lit(*value)
                    }
                    // Graph inference types a branch literal as its branch.
                    Operand::Lit(_) => return Err(site.refuse(Capability::DtypeLowering)),
                });
            }
            let kernel = kg::FusedKernel {
                leaf_tys,
                steps: vec![kg::FusedStep {
                    op: kg::FusedScalarOp::Select,
                    inputs: select_inputs,
                }],
                output: kg::FusedInput::Step(kg::StepId::new(0)),
            };
            generate(KernelRequest::Pointwise(PointwiseSpec {
                out_shape: out_shape.to_vec(),
                numel: out_numel,
                leaves: leaves(&ids),
                kernel,
                form: if odt == DType::I32 {
                    PointwiseForm::ExactI32 { fold }
                } else {
                    PointwiseForm::Float { fold }
                },
            }))?
        }
        OpKind::Unary(uop) => {
            let op = if odt == DType::I32 {
                match uop {
                    GUnOp::Not | GUnOp::Clz => map_fused_op_i32(site, FusedOp::Unary(*uop))?,
                    GUnOp::Neg
                    | GUnOp::Recip
                    | GUnOp::Sqrt
                    | GUnOp::Exp
                    | GUnOp::Log
                    | GUnOp::Tanh
                    | GUnOp::Erf => {
                        return Err(site.refuse(Capability::ExactI32Op));
                    }
                }
            } else {
                map_unary(site, *uop)?
            };
            // A unary op never broadcasts: its operand has the output's shape, read through its layout.
            generate(KernelRequest::Pointwise(PointwiseSpec {
                out_shape: out_shape.to_vec(),
                numel: out_numel,
                leaves: leaves(&ids[..1]),
                kernel: kg::FusedKernel::one_step(
                    op,
                    vec![leaf_ty_of(site, odt)?],
                    vec![kg::FusedInput::Leaf(kg::LeafId::new(0))],
                ),
                form: if odt == DType::I32 {
                    PointwiseForm::ExactI32 { fold }
                } else {
                    PointwiseForm::Float { fold }
                },
            }))?
        }
        OpKind::Rope { rot } => {
            // Fused RoPE (transform::rope_fusion): one dispatch replacing the rotate-half
            // slice/slice/neg/concat/mul/mul/add chain. Inputs [x, cos, sin]; out shape == x. One
            // thread per output element, like concat2.
            generate(KernelRequest::Rope(RopeSpec {
                dt: fty(odt),
                x_shape: shape(ids[0]),
                cos_shape: shape(ids[1]),
                rot: *rot,
                numel: out_numel,
                workgroup: lowering.workgroup(out_numel),
            }))?
        }
        OpKind::Reduce { op, axis, .. } => {
            let a = shape(ids[0]);
            if *axis != a.len() - 1 {
                return Err(site.refuse(Capability::NonLastAxisReduce { axis: *axis }));
            }
            let (kb, init) = match op {
                RedOp::Sum => (BinOp::Add, 0.0f32),
                RedOp::Max => (BinOp::Max, f32::NEG_INFINITY),
            };
            generate(KernelRequest::Row(RowSpec::ReduceLast {
                dt: fty(odt),
                op: kb,
                cols: a[*axis],
                init_bits: init.to_bits(),
                numel: out_numel,
                workgroup: lowering.workgroup(out_numel),
            }))?
        }
        OpKind::Fused(region) => {
            // A region with an I32 output is exact-I32 end to end (and may pack its words). A float region
            // may still compute I32 steps: an exact producer chain, then the cast to f32 that feeds the
            // arithmetic after it.
            let exact_region = odt == DType::I32;
            let input_tys: Vec<_> = ids.iter().map(|&id| g.aval(id).clone()).collect();
            let locals = region
                .local_types(&input_tys)
                .expect("a validated fused region infers");
            let local_dtype = |local: LocalId| locals[local.index()].dtype;
            // A last step that narrows the region's f32 result is the body's narrow store (POOT-1448), which needs
            // native two-byte elements.
            let narrowed = region.narrowed_output();
            let stores_narrow =
                matches!(lowering, Lowering::Gpu { caps, .. } if caps.native_two_byte_floats);
            let last_step = region.steps.len().checked_sub(1);
            let steps = region
                .steps
                .iter()
                .enumerate()
                .map(|(j, st)| {
                    let exact_step = local_dtype(LocalId::new(region.n_inputs + j)) == DType::I32;
                    let narrows_f32 = matches!(
                        st.inputs.as_slice(),
                        [FusedOperand::Local(input)] if local_dtype(*input) == DType::F32
                    );
                    let op = match narrowed {
                        Some(to) if Some(j) == last_step && narrows_f32 => {
                            narrowing_store_op(site, to, stores_narrow)?
                        }
                        _ => map_pointwise_step(site, st.op, &st.inputs, exact_step, local_dtype)?,
                    };
                    Ok(kg::FusedStep {
                        op,
                        inputs: st
                            .inputs
                            .iter()
                            .map(|o| match (exact_region, exact_step) {
                                (true, _) => map_fused_input_i32(site, o, region.n_inputs),
                                (false, true) => map_mixed_input_i32(site, o, region.n_inputs),
                                (false, false) => map_fused_input(site, o, region.n_inputs),
                            })
                            .collect::<Result<Vec<_>, PlanError>>()?,
                    })
                })
                .collect::<Result<Vec<_>, PlanError>>()?;
            // An I32 leaf is a select condition, a cast operand or an exact-I32 step's word (a float region),
            // or one of an exact-I32 region's words.
            let leaf_tys = ids
                .iter()
                .map(|&id| match g.aval(id).dtype {
                    DType::I32 => kg::LeafTy::I32,
                    _ => kg::LeafTy::F32,
                })
                .collect();
            let kernel = kg::FusedKernel {
                leaf_tys,
                steps,
                output: map_fused_local(region.output, region.n_inputs),
            };
            // Card 159's row-fold pack (`region.pack`) launches one thread per packed lane.
            let form = if exact_region {
                let packed_threads = kg::packed_threads(out_numel, region.pack.len());
                PointwiseForm::ExactI32Packed {
                    pack: region
                        .pack
                        .iter()
                        .map(|&id| map_fused_local(id, region.n_inputs))
                        .collect(),
                    workgroup: lowering.workgroup(packed_threads),
                }
            } else {
                PointwiseForm::Float { fold }
            };
            generate(KernelRequest::Pointwise(PointwiseSpec {
                out_shape: out_shape.to_vec(),
                numel: out_numel,
                leaves: leaves(&ids),
                kernel,
                form,
            }))?
        }
        OpKind::FusedRow(region) => {
            // A last step that narrows the region's f32 result is the body's narrow store (POOT-1448), which needs
            // native two-byte elements.
            let narrowed = region.narrowed_output();
            let stores_narrow =
                matches!(lowering, Lowering::Gpu { caps, .. } if caps.native_two_byte_floats);
            let last_step = region.steps.len().checked_sub(1);
            let steps = region
                .steps
                .iter()
                .enumerate()
                .map(|(j, st)| {
                    Ok(match st {
                        RowStep::Pointwise { op, inputs } => kg::RowOp::Pointwise {
                            op: match narrowed {
                                Some(to) if Some(j) == last_step => {
                                    narrowing_store_op(site, to, stores_narrow)?
                                }
                                _ => {
                                    check_widening_step(site, *op, inputs, |id| {
                                        id.index() < region.n_inputs
                                    })?;
                                    map_fused_op(site, *op)?
                                }
                            },
                            inputs: inputs
                                .iter()
                                .map(|o| map_fused_input(site, o, region.n_inputs))
                                .collect::<Result<Vec<_>, PlanError>>()?,
                        },
                        RowStep::Reduce { op, input } => kg::RowOp::Reduce {
                            op: match op {
                                RedOp::Sum => kg::RowReduce::Sum,
                                RedOp::Max => kg::RowReduce::Max,
                            },
                            input: map_fused_input(site, input, region.n_inputs)?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, PlanError>>()?;
            let leaf_storage = ids
                .iter()
                .enumerate()
                .map(|(j, &id)| {
                    let packed = const_lanes.packed(j);
                    float_region_leaf_storage(site, g.aval(id).dtype, packed)
                })
                .collect::<Result<Vec<_>, PlanError>>()?;
            // The kernel walks the region's rows, whatever the shape of what it writes: the result and each side
            // output is a row or a per-row scalar of them (POOT-1456). A side output is stored f32.
            let operand_types: Vec<_> = eqn
                .inputs
                .iter()
                .map(|operand| match operand {
                    Operand::Value(v) => g.aval(*v).clone(),
                    Operand::Lit(scalar) => scalar.ty(),
                })
                .collect();
            let row_shape = region
                .row_shape(&operand_types)
                .ok()
                .flatten()
                .ok_or_else(|| site.refuse(Capability::RowRegionWithoutReduction))?;
            let side_types = region
                .side_output_types(&operand_types)
                .map_err(|_| site.refuse(Capability::RowRegionWithoutReduction))?;
            if side_types.iter().any(|ty| ty.dtype != DType::F32) {
                return Err(site.refuse(Capability::DtypeLowering));
            }
            let n_cols = row_shape[region.axis];
            let kernel = kg::RowKernel {
                n_leaves: region.n_inputs,
                leaf_storage,
                n_cols,
                steps,
                output: map_row_value(region.output, region.n_inputs),
                side_outputs: region
                    .side_outputs
                    .iter()
                    .map(|&side| map_row_value(side, region.n_inputs))
                    .collect(),
            };
            generate(KernelRequest::Row(RowSpec::Fused {
                numel: row_shape.iter().product::<usize>().max(1),
                row_shape,
                leaves: leaves(&ids),
                kernel,
                // One workgroup of `width` lanes per row (spec 013); an AIE core is one lane, the form of
                // the generator without workgroup memory.
                width: match lowering {
                    Lowering::Gpu { .. } => row_parallel_width(n_cols),
                    Lowering::Aie(_) => 1,
                },
            }))?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | side_output_ops!()
        | packed_ops!()
        | moe_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | stable_rank_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only elementwise ops here"),
    };
    Ok(planned)
}
