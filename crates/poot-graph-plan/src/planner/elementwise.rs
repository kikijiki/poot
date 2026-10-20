//! Elementwise, reduction, fused-region and fused-RoPE equation planning.

use super::{
    BinOp, BodyLimits, Capability, ContractionHead, DType, ElementwiseSpec, Eqn, GBinOp, GUnOp,
    HashMap, I32Binary, KernelRequest, Lowering, OpKind, Operand, PlanError, Planned,
    PointwiseForm, PointwiseSpec, RedOp, RopeSpec, RowSpec, RowStep, Scalar, Ty, UnOp, UnaryOp,
    ValueBinary, ValueId, ViewOperand, fty_ty, kg, value_ids,
};
use crate::kernel_mapping::{
    check_widening_step, float_region_leaf_storage, map_binop, map_fused_input,
    map_fused_input_i32, map_fused_local, map_fused_op, map_mixed_input_i32, map_pointwise_step,
    map_unary,
};
use crate::planner::{GraphTables, Layout};
use crate::predicates::{ELEMENTWISE_2D_WIDTH, elementwise_fold, row_parallel_width};
use crate::refusal::Site;
use poot_graph_ir::LocalId;

/// Plan one elementwise-family equation (Binary/Select/Unary/Rope/Reduce/Fused/FusedRow).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    lowering: Lowering<'_>,
    views: &HashMap<ValueId, Layout>,
    odt: DType,
    limits: BodyLimits,
    packed_operands: &[bool],
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, lowering.id(), limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let layout_of = |vid: ValueId| -> Layout {
        views
            .get(&vid)
            .cloned()
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
    // The 2-D fold decision (card 159, card 181 Bug 2): past the wg-bump's own ceiling
    // (`is_elementwise_2d`), a one-thread-per-element kernel folds onto the same 2-D grid the imported
    // scatter/DUS/index_remap kernels use; a flat X dispatch cannot clear wgpu's `gridDim.x` cap however
    // large `workgroup_size` gets. `finalize` bakes `workgroup_size[0] = ELEMENTWISE_2D_WIDTH` whenever
    // the predicate holds; `generate` derives the matching `x_groups` and grid from the same decision.
    let fold = match lowering {
        Lowering::Gpu { caps, .. } => elementwise_fold(g, eqn, out_shape, out_numel, caps),
        // A sequential loop has no grid to fold onto.
        Lowering::Aie(_) => kg::Fold {
            numel: out_numel,
            two_d: false,
            width: ELEMENTWISE_2D_WIDTH,
        },
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
            let kb = (!matches!(*bop, GBinOp::GeU | GBinOp::RemU)).then(|| map_binop(*bop));
            match &eqn.inputs[1] {
                Operand::Lit(Scalar::F32(_)) if exact_i32 => {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
                // card 181 Bug 2: the scalar/literal form (e.g. the attention-score `* scale`) gets the
                // same 2-D fold as the broadcast-value form below.
                Operand::Lit(Scalar::F32(v)) => {
                    generate(KernelRequest::Elementwise(ElementwiseSpec::ScalarFloat {
                        op: kb.expect("float GeU/RemU is rejected by graph inference"),
                        dt: fty(odt),
                        lit_bits: v.to_bits(),
                        fold,
                    }))?
                }
                Operand::Lit(Scalar::I32(v)) if exact_i32 => {
                    generate(KernelRequest::Elementwise(ElementwiseSpec::ScalarI32 {
                        op: match bop {
                            GBinOp::GeU => I32Binary::GeU,
                            GBinOp::RemU => I32Binary::RemU,
                            _ => I32Binary::Basic(
                                kb.expect("non-unsigned I32 op has a direct kernel op"),
                            ),
                        },
                        lit: *v,
                        fold,
                    }))?
                }
                Operand::Lit(Scalar::I32(v)) => {
                    generate(KernelRequest::Elementwise(ElementwiseSpec::ScalarFloat {
                        op: kb.expect("float GeU/RemU is rejected by graph inference"),
                        dt: fty(odt),
                        lit_bits: (*v as f32).to_bits(),
                        fold,
                    }))?
                }
                Operand::Value(_) => {
                    // Layout safety (spec 132): a's/b's strides and offset are baked into the generated
                    // body, so they are part of the request: two equations with the same shapes but a
                    // different source layout (one a plain buffer, one a view) are different kernels.
                    let mut operands = leaves(&ids[..2]).into_iter();
                    let (a, b) = (operands.next().unwrap(), operands.next().unwrap());
                    let op = match bop {
                        GBinOp::GeU => {
                            debug_assert!(
                                exact_i32,
                                "float GeU/RemU is rejected by graph inference"
                            );
                            ValueBinary::GeU
                        }
                        GBinOp::RemU => {
                            debug_assert!(exact_i32, "float RemU is rejected by graph inference");
                            ValueBinary::RemU
                        }
                        _ => ValueBinary::Basic(
                            kb.expect("unsigned I32 binary has a typed kernel path"),
                            if exact_i32 { Ty::I32 } else { fty(odt) },
                        ),
                    };
                    generate(KernelRequest::Elementwise(ElementwiseSpec::Binary {
                        op,
                        out_shape: out_shape.to_vec(),
                        a,
                        b,
                        fold,
                    }))?
                }
            }
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
            let kernel = kg::FusedKernel {
                leaf_tys: vec![kg::LeafTy::I32, branch, branch],
                steps: vec![kg::FusedStep {
                    op: kg::FusedScalarOp::Select,
                    inputs: vec![
                        kg::FusedInput::Leaf(0),
                        kg::FusedInput::Leaf(1),
                        kg::FusedInput::Leaf(2),
                    ],
                }],
                output: kg::FusedInput::Step(0),
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
            let view = views.get(&ids[0]).map(|layout| ViewOperand {
                shape: out_shape.to_vec(),
                layout: layout.clone(),
            });
            let (op, dt) = if odt == DType::I32 {
                match uop {
                    GUnOp::Not => (UnaryOp::Basic(UnOp::Not), Ty::I32),
                    GUnOp::Clz => (UnaryOp::ClzI32, Ty::I32),
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
                (map_unary(site, *uop)?, fty(odt))
            };
            generate(KernelRequest::Elementwise(ElementwiseSpec::Unary {
                op,
                dt,
                view,
                fold,
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
            let steps = region
                .steps
                .iter()
                .enumerate()
                .map(|(j, st)| {
                    let exact_step = local_dtype(LocalId::new(region.n_inputs + j)) == DType::I32;
                    Ok(kg::FusedStep {
                        op: map_pointwise_step(site, st.op, &st.inputs, exact_step, local_dtype)?,
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
                PointwiseForm::ExactI32Packed {
                    pack: region
                        .pack
                        .iter()
                        .map(|&id| map_fused_local(id, region.n_inputs))
                        .collect(),
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
            let steps = region
                .steps
                .iter()
                .map(|st| {
                    Ok(match st {
                        RowStep::Pointwise { op, inputs } => kg::RowOp::Pointwise {
                            op: {
                                check_widening_step(site, *op, inputs, |id| {
                                    id.index() < region.n_inputs
                                })?;
                                map_fused_op(site, *op)?
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
                    let packed = packed_operands.get(j).copied().unwrap_or(false);
                    float_region_leaf_storage(site, g.aval(id).dtype, packed)
                })
                .collect::<Result<Vec<_>, PlanError>>()?;
            let kernel = kg::RowKernel {
                n_leaves: region.n_inputs,
                leaf_storage,
                n_cols: out_shape[region.axis],
                steps,
                // the region's output local id uses the same 0..n_inputs leaves, n_inputs+s steps
                // numbering as the kernelgen recipe.
                output: region.output.index(),
            };
            generate(KernelRequest::Row(RowSpec::Fused {
                out_shape: out_shape.to_vec(),
                numel: out_numel,
                leaves: leaves(&ids),
                kernel,
                // One workgroup of `width` lanes per row (spec 013); an AIE core is one lane, the form of
                // the generator without workgroup memory.
                width: match lowering {
                    Lowering::Gpu { .. } => row_parallel_width(out_shape[region.axis]),
                    Lowering::Aie(_) => 1,
                },
            }))?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only elementwise ops here"),
    };
    Ok(planned)
}
