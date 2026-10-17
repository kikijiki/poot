//! The Intel NPU (VPU) arm of the planner (POOT-1159, design npu-targets-1093 sections 4.1 and 4.7).
//!
//! An Intel NPU runs whole graphs a vendor compiler builds, so an equation does not lower to a `Body`: it
//! lowers to one OpenVINO opset op ([`VpuOp`]) with the same semantics, and a region of such equations is
//! one vendor-compiled graph. Support is still decided per equation, here, with typed refusals that name
//! the equation (ADR-0104): [`lower_eqn`] is the one table, and [`plan_vpu_region`] only assembles the
//! lowered equations into a [`VpuRegion`] for `poot-codegen`'s emitter. Nothing here recognizes a composite
//! (attention stays its primitives and the vendor compiler fuses what it can): a recognizer would be a second
//! semantic definition of the op.
//!
//! Numerics. An Intel NPU computes in FP16. An F32 or BF16 arithmetic equation is a narrowing of its
//! semantics, not a schedule reassociation, so it is refused with [`Capability::Arithmetic`] instead of
//! being computed narrower. Data movement is bit exact in any width the device carries.

use std::collections::HashMap;

use poot_codegen::{VpuElem, VpuInput, VpuNode, VpuOp, VpuRegion, VpuTensor, VpuValue};
use poot_graph_ir::op::{BinOp, ContractionHead, RedOp, UnOp};
use poot_graph_ir::{Eqn, Graph, OpKind, Operand, ValueId};
use poot_target::{VpuArithmetic, VpuCaps};
use poot_tensor::DType;

use crate::compile::whole_graph_region;
use crate::planner::Planned;
use crate::predicates::GraphTables;
use crate::refusal::{StorageGap, refusal_at};
use crate::{Capability, PlanError};

/// The plan of `eqn` on an Intel NPU: [`Plan::Graph`] when the lowering table has an opset op for it, else the
/// equation's typed refusal. An equation launches nothing of its own: it is a node of the program's one
/// graph artifact, which [`plan_vpu_region`] assembles from the lowered equations.
pub(super) fn plan_eqn(
    g: GraphTables<'_>,
    eqn: &Eqn,
    caps: &VpuCaps,
) -> Result<Planned, PlanError> {
    lower_eqn(g, eqn, caps).map(|_| Planned::graph())
}

/// The VPU element type of `dtype`, if the device carries it.
const fn elem_of(dtype: DType) -> Option<VpuElem> {
    match dtype {
        DType::F16 => Some(VpuElem::F16),
        DType::F32 => Some(VpuElem::F32),
        DType::I32 => Some(VpuElem::I32),
        _ => None,
    }
}

/// Whether the target computes `dtype` natively.
fn computes(caps: &VpuCaps, dtype: DType) -> Result<(), Capability> {
    match (caps.arithmetic(), dtype) {
        (VpuArithmetic::F16, DType::F16) => Ok(()),
        (VpuArithmetic::F16, DType::F32 | DType::BF16 | DType::F64) => {
            Err(Capability::Arithmetic { dtype })
        }
        (VpuArithmetic::F16, _) => Err(Capability::DtypeLowering),
    }
}

/// One lowered equation: its opset op, the value operands it reads and the type of the value it defines.
struct Lowered {
    op: VpuOp,
    operands: Vec<ValueId>,
    out: VpuTensor,
}

/// One equation's lowering, or the typed refusal.
fn lower_eqn(g: GraphTables<'_>, eqn: &Eqn, caps: &VpuCaps) -> Result<Lowered, PlanError> {
    let refuse = |missing| refusal_at(g, eqn, caps.target, missing);
    let operands = value_operands(eqn).ok_or_else(|| refuse(Capability::LiteralOperand))?;
    let out = g.aval(eqn.out);
    let out_elem = elem_of(out.dtype);
    let tensor = |elem: Option<VpuElem>| {
        elem.map(|elem| VpuTensor {
            elem,
            shape: out.shape.clone(),
        })
        .ok_or_else(|| refuse(Capability::DtypeLowering))
    };
    // Arithmetic: the output and every operand compute natively.
    let arithmetic = |op: VpuOp| -> Result<Lowered, PlanError> {
        for dtype in operands
            .iter()
            .map(|&id| g.aval(id).dtype)
            .chain([out.dtype])
        {
            computes(caps, dtype).map_err(refuse)?;
        }
        Ok(Lowered {
            op,
            operands: operands.clone(),
            out: tensor(out_elem)?,
        })
    };
    // Data movement: bit exact, in any width the device carries.
    let movement = |op: VpuOp| -> Result<Lowered, PlanError> {
        for dtype in operands.iter().map(|&id| g.aval(id).dtype) {
            if elem_of(dtype).is_none() {
                return Err(refuse(Capability::DtypeLowering));
            }
        }
        Ok(Lowered {
            op,
            operands: operands.clone(),
            out: tensor(out_elem)?,
        })
    };
    let Some((op, class)) = opset_op(&eqn.op) else {
        return Err(refuse(unsupported(&eqn.op)));
    };
    match class {
        OpClass::Arithmetic => arithmetic(op),
        OpClass::Movement => movement(op),
        OpClass::Gather => {
            if g.aval(operands[1]).dtype != DType::I32 {
                return Err(refuse(Capability::DtypeLowering));
            }
            movement(op)
        }
        OpClass::Convert => {
            // F16 <-> F32 only: a conversion through an integer or BF16 changes the value.
            let float = |dtype| matches!(dtype, DType::F16 | DType::F32);
            if !float(g.aval(operands[0]).dtype) || !float(out.dtype) {
                return Err(refuse(Capability::DtypeLowering));
            }
            Ok(Lowered {
                op,
                operands: operands.clone(),
                out: tensor(out_elem)?,
            })
        }
    }
}

/// How an opset op's dtypes are checked.
enum OpClass {
    /// Computes in the device's arithmetic width: the output and every operand must be native.
    Arithmetic,
    /// Moves bits: any width the device carries.
    Movement,
    /// A `Gather`: movement whose index operand is I32.
    Gather,
    /// A `Convert`: between F16 and F32.
    Convert,
}

/// The one opset op `op` lowers to, if the table has one. The ops are the design's primitive list; anything
/// else is refused by [`unsupported`], never approximated.
fn opset_op(op: &OpKind) -> Option<(VpuOp, OpClass)> {
    Some(match op {
        OpKind::Binary(BinOp::Add) => (VpuOp::Add, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Mul) => (VpuOp::Multiply, OpClass::Arithmetic),
        OpKind::Unary(UnOp::Exp) => (VpuOp::Exp, OpClass::Arithmetic),
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            keepdim,
        } => (
            VpuOp::ReduceSum {
                axis: *axis,
                keepdim: *keepdim,
            },
            OpClass::Arithmetic,
        ),
        OpKind::MatMul => (VpuOp::MatMul, OpClass::Arithmetic),
        OpKind::Gather { axis } => (VpuOp::Gather { axis: *axis }, OpClass::Gather),
        OpKind::Reshape { .. } => (VpuOp::Reshape, OpClass::Movement),
        OpKind::Transpose { perm } => (VpuOp::Transpose { perm: perm.clone() }, OpClass::Movement),
        OpKind::Broadcast { .. } => (VpuOp::Broadcast, OpClass::Movement),
        OpKind::Slice { axis, start, end } => (
            VpuOp::Slice {
                axis: *axis,
                start: *start,
                end: *end,
            },
            OpClass::Movement,
        ),
        OpKind::Concat { axis } => (VpuOp::Concat { axis: *axis }, OpClass::Movement),
        OpKind::Cast { .. } => (VpuOp::Convert, OpClass::Convert),
        _ => return None,
    })
}

/// Why an op [`opset_op`] has no lowering for is refused. Exhaustive over the planner's op families, so a new
/// `OpKind` variant fails to compile until it is assigned one.
fn unsupported(op: &OpKind) -> Capability {
    match op {
        OpKind::Iota { .. } => Capability::UnfoldedIota,
        elementwise_ops!()
        | imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | matmul_ops!()
        | attention_ops!()
        | scan_ops!()
        | block_top_k_ops!()
        | cast_ops!()
        | sampling_ops!() => Capability::VpuOpset,
        // A collective is a cut between stage programs on every target.
        movement_ops!() => match op {
            OpKind::AllReduce { .. } | OpKind::AllGather { .. } => Capability::CollectiveInStage,
            _ => Capability::VpuOpset,
        },
    }
}

/// The value operands of `eqn`; `None` when any operand is an inline literal.
fn value_operands(eqn: &Eqn) -> Option<Vec<ValueId>> {
    eqn.inputs
        .iter()
        .map(|operand| match operand {
            Operand::Value(id) => Some(*id),
            Operand::Lit(_) => None,
        })
        .collect()
}

/// Plans every equation of `graph` for the Intel NPU `caps` and assembles them into one region: the graph's
/// equations in program order, every value defined outside them a runtime argument (the values
/// [`package`](crate::compile) reads from outside the whole graph: weights, activations and state, in order
/// of first use), and the values the graph publishes or nothing reads its results. The argument and result
/// lists are the program's one [`Region`](crate::Region)'s, so the artifact's ABI and the engine's bindings
/// are the same lists by construction.
///
/// # Errors
///
/// The first equation's typed refusal ([`PlanError::Refused`]), or [`PlanError::UnrepresentableValue`] when a
/// region argument has a dtype the device does not carry.
pub(crate) fn plan_vpu_region(graph: &Graph, caps: &VpuCaps) -> Result<VpuRegion, PlanError> {
    let g = GraphTables::from(graph);
    let lowered = graph
        .eqns()
        .iter()
        .map(|eqn| lower_eqn(g, eqn, caps))
        .collect::<Result<Vec<_>, _>>()?;
    let region = whole_graph_region(graph);
    // Values are numbered inputs first, then each node's output, in order.
    let mut numbering: HashMap<ValueId, VpuValue> = HashMap::new();
    let mut inputs: Vec<VpuInput> = Vec::with_capacity(region.reads().len());
    for &id in region.reads() {
        numbering.insert(id, VpuValue(inputs.len() as u32));
        inputs.push(input_of(g, id)?);
    }
    let mut nodes = Vec::with_capacity(lowered.len());
    for (eqn, lowered) in graph.eqns().iter().zip(lowered) {
        let value = VpuValue((inputs.len() + nodes.len()) as u32);
        nodes.push(VpuNode {
            origin: eqn.out.index() as u32,
            op: lowered.op,
            inputs: lowered.operands.iter().map(|id| numbering[id]).collect(),
            out: lowered.out,
        });
        numbering.insert(eqn.out, value);
    }
    let outputs = region.writes().iter().map(|id| numbering[id]).collect();
    Ok(VpuRegion {
        name: "poot_region".into(),
        inputs,
        nodes,
        outputs,
    })
}

/// The region argument for value `id`.
fn input_of(g: GraphTables<'_>, id: ValueId) -> Result<VpuInput, PlanError> {
    let aval = g.aval(id);
    let elem = elem_of(aval.dtype).ok_or(PlanError::UnrepresentableValue {
        value: id,
        dtype: aval.dtype,
        gap: StorageGap::VpuDtype,
    })?;
    Ok(VpuInput {
        name: id.to_string(),
        tensor: VpuTensor {
            elem,
            shape: aval.shape.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use poot_codegen::{VpuRegion, emit_vpu};
    use poot_graph_ir::{BinOp, Builder, Graph, RedOp, Scalar, TensorType, Traced, UnOp};
    use poot_target::{Queried, VendorCompilerId, VpuTargetId};

    use super::*;
    use crate::production_plan::Submission;
    use crate::{CompileError, CompileLimits, CompileOptions, Plan, Refusal, Target};

    fn caps() -> VpuCaps {
        // SYNTHETIC: an NPU4 with a made-up driver compiler version, not a device measurement.
        VpuCaps {
            target: VpuTargetId::Npu4,
            compiler: Queried::Known(VendorCompilerId { major: 7, minor: 5 }),
            memory_budget_bytes: Queried::Unknown,
        }
    }

    fn graph_of(build: impl FnOnce(&Builder) -> Traced) -> Graph {
        let b = Builder::for_tests();
        let out = build(&b);
        b.finish(out).unwrap()
    }

    fn f(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType::new(shape.to_vec(), dtype)
    }

    fn refusal(result: Result<VpuRegion, PlanError>) -> Refusal {
        match result {
            Err(PlanError::Refused(refusal)) => *refusal,
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }

    /// `sum(x * w + x, axis 1)` over F16 `[2, 3]`: the region's IR text, pinned. Every layer is named after
    /// its equation (`v2` is the multiply, `v3` the add, `v4` the reduction), the two runtime arguments are
    /// the constants `v0` and `v1`, the only result is the reduction (the intermediates have none), and the
    /// reduction's axis is an `i64` constant at offset 0 of the weight blob.
    /// Mutation: lower `Multiply` as `Subtract`; the pinned text no longer matches.
    #[test]
    fn an_f16_chain_emits_the_pinned_openvino_ir() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[2, 3])).unwrap();
            let w = b.constant("w", f(DType::F16, &[2, 3])).unwrap();
            let scaled = b.binary(BinOp::Mul, x, w).unwrap();
            let shifted = b.binary(BinOp::Add, scaled, x).unwrap();
            b.reduce(RedOp::Sum, shifted, 1, false).unwrap()
        });
        let region = plan_vpu_region(&g, &caps()).unwrap();
        let ir = emit_vpu(&region).unwrap();
        let expected = concat!(
            r#"<?xml version="1.0"?><net name="poot_region" version="11"><layers>"#,
            r#"<layer id="0" name="v0" type="Parameter" version="opset1"><data shape="2,3" element_type="f16"/><output><port id="0" precision="FP16" names="v0"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="1" name="v1" type="Parameter" version="opset1"><data shape="2,3" element_type="f16"/><output><port id="0" precision="FP16" names="v1"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="2" name="v2" type="Multiply" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="FP16"><dim>2</dim><dim>3</dim></port></input><output><port id="2" precision="FP16" names="v2"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="3" name="v3" type="Add" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="FP16"><dim>2</dim><dim>3</dim></port></input><output><port id="2" precision="FP16" names="v3"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="4" name="aux_4" type="Const" version="opset1"><data element_type="i64" shape="1" offset="0" size="8"/><output><port id="0" precision="I64"><dim>1</dim></port></output></layer>"#,
            r#"<layer id="5" name="v4" type="ReduceSum" version="opset1"><data keep_dims="false"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="I64"><dim>1</dim></port></input><output><port id="2" precision="FP16" names="v4"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="6" name="result_0" type="Result" version="opset1"><input><port id="0" precision="FP16"><dim>2</dim></port></input></layer>"#,
            r#"</layers><edges>"#,
            r#"<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>"#,
            r#"<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/><edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>"#,
            r#"<edge from-layer="3" from-port="2" to-layer="5" to-port="0"/><edge from-layer="4" from-port="0" to-layer="5" to-port="1"/>"#,
            r#"<edge from-layer="5" from-port="2" to-layer="6" to-port="0"/>"#,
            r#"</edges></net>"#,
        );
        assert_eq!(ir.xml, expected);
        assert_eq!(ir.weights, 1i64.to_le_bytes());
    }

    /// F32 and BF16 arithmetic are refused with the dtype named, not computed in FP16 (ADR-0112 decision 2).
    /// Mutation: drop the arithmetic check in `lower_eqn`; the F32 add lowers and this row fails.
    #[test]
    fn f32_and_bf16_arithmetic_are_refused_with_the_dtype() {
        for dtype in [DType::F32, DType::BF16] {
            let g = graph_of(|b| {
                let x = b.constant("x", f(dtype, &[2, 3])).unwrap();
                let y = b.constant("y", f(dtype, &[2, 3])).unwrap();
                b.binary(BinOp::Add, x, y).unwrap()
            });
            let refused = refusal(plan_vpu_region(&g, &caps()));
            assert_eq!(refused.missing, Capability::Arithmetic { dtype });
            assert_eq!(refused.eqn, g.eqns()[0].out);
            assert_eq!(
                refused.target,
                poot_target::TargetId::Vpu(VpuTargetId::Npu4)
            );
        }
    }

    /// An op outside the lowering table is a typed refusal, never an approximation; a literal operand and
    /// an integer conversion are refused for their own reasons.
    #[test]
    fn ops_without_an_opset_lowering_are_refused_by_name() {
        let sub = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            let y = b.constant("y", f(DType::F16, &[4])).unwrap();
            b.binary(BinOp::Sub, x, y).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&sub, &caps())).missing,
            Capability::VpuOpset
        );
        let silu = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            b.unary(UnOp::Neg, x).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&silu, &caps())).missing,
            Capability::VpuOpset
        );
        let literal = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0)).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&literal, &caps())).missing,
            Capability::LiteralOperand
        );
        let to_bf16 = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            b.cast(x, DType::BF16).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&to_bf16, &caps())).missing,
            Capability::DtypeLowering
        );
    }

    /// An Intel NPU's artifact is the whole graph: its vendor compiler takes every equation at once, so
    /// `package` groups nothing and the granularity names no region limits.
    /// Mutation: map `Target::Vpu` to `ArtifactGranularity::Kernel` in `Target::granularity`; this row fails.
    #[test]
    fn an_intel_npu_target_has_whole_graph_granularity() {
        let target = Target::Vpu {
            target: VpuTargetId::Npu4,
            caps: caps(),
        };
        assert_eq!(
            target.granularity(),
            crate::compile::ArtifactGranularity::Graph
        );
    }

    /// The production entry: `compile` for an Intel NPU target plans every supported equation as a node of the one
    /// graph artifact and refuses the first equation the lowering has no op for, by name.
    /// Mutation: make the `Target::Vpu` arm of `plan_eqn_views_impl` skip `lower_eqn`; the F32 row then
    /// compiles instead of reporting `Arithmetic`.
    #[test]
    fn compile_for_an_intel_npu_plans_one_graph_region_and_refuses_what_it_cannot_lower() {
        let target = Target::Vpu {
            target: VpuTargetId::Npu4,
            caps: caps(),
        };
        let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
        let compiled = |dtype: DType| {
            let g = graph_of(|b| {
                let x = b.constant("x", f(dtype, &[2, 3])).unwrap();
                let y = b.constant("y", f(dtype, &[2, 3])).unwrap();
                b.binary(BinOp::Add, x, y).unwrap()
            });
            crate::compile(&g, &target, &options)
        };
        let program = compiled(DType::F16).unwrap();
        assert_eq!(
            program.regions().len(),
            1,
            "the whole graph is one artifact"
        );
        assert_eq!(program.regions()[0].members(), 0..1);
        assert!(matches!(program.planned().next().unwrap().1, Plan::Graph));
        let Err(CompileError::Plan(error)) = compiled(DType::F32) else {
            panic!("an F32 add must be refused on an NPU");
        };
        let PlanError::Refused(refusal) = *error else {
            panic!("expected a typed refusal, got {error:?}");
        };
        assert_eq!(
            refusal.missing,
            Capability::Arithmetic { dtype: DType::F32 }
        );
    }

    /// A scalar-index gather is `Reshape(index -> [1]) -> Gather -> Reshape`, since the NPU faults on a
    /// rank-0 index; a vector-index gather is the bare op. Both define the output shape poot's `Gather`
    /// infers.
    /// Mutation: emit the scalar gather as the bare op; the layer-type sequence differs.
    #[test]
    fn a_scalar_index_gather_is_reshaped_around_the_op() {
        let layers = |index_shape: &[usize]| {
            let g = graph_of(|b| {
                let table = b.constant("table", f(DType::F16, &[5, 4])).unwrap();
                let ids = b.constant("ids", f(DType::I32, index_shape)).unwrap();
                b.gather(table, 0, ids).unwrap()
            });
            let ir = emit_vpu(&plan_vpu_region(&g, &caps()).unwrap()).unwrap();
            ir.xml
                .match_indices(" type=\"")
                .map(|(at, _)| {
                    let rest = &ir.xml[at + 7..];
                    rest[..rest.find('"').unwrap()].to_string()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            layers(&[]),
            [
                "Parameter",
                "Parameter",
                "Const",
                "Const",
                "Reshape",
                "Gather",
                "Const",
                "Reshape",
                "Result"
            ]
        );
        assert_eq!(
            layers(&[3]),
            ["Parameter", "Parameter", "Const", "Gather", "Result"]
        );
    }

    /// Movement ops keep their dtype bit for bit, including F32 and I32, and a Convert is the only way to
    /// change width: F32 -> F16 lowers, and the region's argument stays F32.
    #[test]
    fn movement_and_convert_carry_their_dtypes() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F32, &[2, 3])).unwrap();
            let narrowed = b.cast(x, DType::F16).unwrap();
            let t = b.transpose(narrowed, vec![1, 0]).unwrap();
            b.slice(t, 0, 1, 3).unwrap()
        });
        let region = plan_vpu_region(&g, &caps()).unwrap();
        assert_eq!(region.inputs[0].tensor.elem, poot_codegen::VpuElem::F32);
        let ops: Vec<_> = region.nodes.iter().map(|n| n.op.clone()).collect();
        assert_eq!(
            ops,
            [
                VpuOp::Convert,
                VpuOp::Transpose { perm: vec![1, 0] },
                VpuOp::Slice {
                    axis: 0,
                    start: 1,
                    end: 3
                }
            ]
        );
        assert_eq!(region.nodes[2].out.shape, [2, 2]);
        assert_eq!(region.nodes[0].out.elem, poot_codegen::VpuElem::F16);
    }
}
