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
//! Numerics (POOT-1178, POOT-ADR-0103). An Intel NPU computes in FP16, and a vendor compiler handed an F32
//! graph narrows it silently. Poot makes that step its own: when the target records a
//! [`NarrowingContract`], [`narrow_to_device_precision`] plans the whole region over F16 values, converts each
//! F32 region input to F16 on entry and each F32 result back on exit, and the contract bounds the error of the
//! region as a whole (interior values are not converted op by op). With no contract an F32 equation is refused
//! with [`Capability::NarrowingContract`], and BF16 or F64 arithmetic, which no contract covers, with
//! [`Capability::Arithmetic`]. Data movement is bit exact in any width the device carries.

use std::collections::HashMap;

use poot_codegen::{VpuElem, VpuInput, VpuNode, VpuOp, VpuRegion, VpuScalar, VpuTensor, VpuValue};
use poot_graph_ir::op::{BinOp, ContractionHead, RedOp, UnOp};
use poot_graph_ir::{Eqn, Graph, OpKind, Operand, Scalar, ValueId};
use poot_target::{NarrowingContract, VpuArithmetic, VpuCaps};
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

/// Every equation of `graph` the Intel NPU `caps` refuses, in program order, each with its typed refusal.
/// [`plan_vpu_region`] stops at the first refusal; this plans every equation on its own, so a caller sees
/// the whole list of what stands between a graph and the NPU (the first refusal alone hides every other
/// op class).
///
/// # Errors
///
/// A planning failure that is not a refusal.
#[cfg(any(test, feature = "test-support"))]
pub fn vpu_refusals(graph: &Graph, caps: &VpuCaps) -> Result<Vec<crate::Refusal>, PlanError> {
    let g = GraphTables::from(graph);
    let mut refused = Vec::new();
    for eqn in graph.eqns() {
        match lower_eqn(g, eqn, caps) {
            Ok(_) => {}
            Err(PlanError::Refused(refusal)) => refused.push(*refusal),
            Err(other) => return Err(other),
        }
    }
    Ok(refused)
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

/// The contract under which the region computes `dtype` values narrower, if the target records one for them.
fn narrowing_of(caps: &VpuCaps, dtype: DType) -> Option<NarrowingContract> {
    match (caps.arithmetic(), dtype) {
        (VpuArithmetic::F16, DType::F32) => caps.narrowing,
        _ => None,
    }
}

/// The dtype a value of `dtype` has inside the region: F16 for an F32 value the contract narrows, else its own.
fn device_dtype(caps: &VpuCaps, dtype: DType) -> DType {
    if narrowing_of(caps, dtype).is_some() {
        match caps.arithmetic() {
            VpuArithmetic::F16 => DType::F16,
        }
    } else {
        dtype
    }
}

/// Whether the target computes `dtype` values, natively or narrowed under the target's contract.
fn computes(caps: &VpuCaps, dtype: DType) -> Result<(), Capability> {
    match (caps.arithmetic(), dtype) {
        (VpuArithmetic::F16, DType::F16) => Ok(()),
        (VpuArithmetic::F16, DType::F32) => narrowing_of(caps, dtype)
            .map(|_| ())
            .ok_or(Capability::NarrowingContract { dtype }),
        (VpuArithmetic::F16, DType::BF16 | DType::F64) => Err(Capability::Arithmetic { dtype }),
        (VpuArithmetic::F16, _) => Err(Capability::DtypeLowering),
    }
}

/// The largest finite FP16 value.
const F16_MAX: f32 = 65504.0;

/// The largest integer below which every integer is exact in FP16 (11 significand bits).
const F16_EXACT_INTEGERS: u32 = 2048;

/// One operand of a lowered equation: a value, or a scalar constant the region defines for it.
enum LoweredOperand {
    Value(ValueId),
    Scalar(VpuScalar),
}

/// One lowered equation: its opset op, the operands it reads and the type of the value it defines.
struct Lowered {
    op: VpuOp,
    operands: Vec<LoweredOperand>,
    out: VpuTensor,
}

/// An inline literal's value as the float it takes part in arithmetic as.
fn literal_value(scalar: Scalar) -> f32 {
    match scalar {
        Scalar::F32(value) => value,
        Scalar::I32(value) => value as f32,
    }
}

/// The constant an inline literal lowers to when it takes part in arithmetic in `dtype`: the value rounded
/// to the device's arithmetic width. A literal the width cannot hold is refused, not rounded to infinity, and a
/// NaN literal is refused. The one exception is `saturate`, set only for a finite F32 literal of an additive
/// equation (`Add`, `Sub`) under the narrowing contract: there a magnitude past the range is a sentinel that
/// acts as an infinity (a mask), so it saturates to the largest finite value. A multiplicative use would
/// silently change the result (`x * 1e9` is not `x * 65504`), so it stays refused.
fn arithmetic_scalar(
    scalar: Scalar,
    dtype: DType,
    saturate: bool,
) -> Result<VpuScalar, Capability> {
    let value = literal_value(scalar);
    match dtype {
        DType::F16 => {
            let value = if saturate && value.is_finite() {
                value.clamp(-F16_MAX, F16_MAX)
            } else {
                value
            };
            let bits = poot_quant::scalar::f32_to_f16(value);
            if !value.is_nan()
                && poot_quant::scalar::f16_to_f32(bits).is_finite() == value.is_finite()
            {
                Ok(VpuScalar::F16(bits))
            } else {
                Err(Capability::DtypeLowering)
            }
        }
        DType::F32 => Ok(VpuScalar::F32(value.to_bits())),
        _ => Err(Capability::DtypeLowering),
    }
}

/// The constant an inline literal lowers to when it only moves (an index): its own value and dtype.
fn carried_scalar(scalar: Scalar) -> VpuScalar {
    match scalar {
        Scalar::F32(value) => VpuScalar::F32(value.to_bits()),
        Scalar::I32(value) => VpuScalar::I32(value),
    }
}

/// One equation's lowering, or the typed refusal.
fn lower_eqn(g: GraphTables<'_>, eqn: &Eqn, caps: &VpuCaps) -> Result<Lowered, PlanError> {
    let refuse = |missing| refusal_at(g, eqn, caps.target, missing);
    let out = g.aval(eqn.out);
    let device_out = device_dtype(caps, out.dtype);
    let out_elem = elem_of(device_out);
    let operand_dtype = |operand: &Operand| match operand {
        Operand::Value(id) => g.aval(*id).dtype,
        Operand::Lit(scalar) => scalar.dtype(),
    };
    let tensor = |elem: Option<VpuElem>| {
        elem.map(|elem| VpuTensor {
            elem,
            shape: out.shape.clone(),
        })
        .ok_or_else(|| refuse(Capability::DtypeLowering))
    };
    // Arithmetic: the output and every value operand compute natively or narrowed under the contract; a
    // literal takes the width the output has inside the region.
    let arithmetic = |op: VpuOp| -> Result<Lowered, PlanError> {
        for dtype in eqn
            .inputs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(id) => Some(g.aval(*id).dtype),
                Operand::Lit(_) => None,
            })
            .chain([out.dtype])
        {
            computes(caps, dtype).map_err(refuse)?;
        }
        let saturate =
            narrowing_of(caps, out.dtype).is_some() && matches!(op, VpuOp::Add | VpuOp::Subtract);
        // The contract's range holds for every literal of a narrowed equation but those of an additive one,
        // which `arithmetic_scalar` brings into range on purpose (and refuses when NaN): anything else past the
        // range would narrow to infinity.
        if let Some(contract) = narrowing_of(caps, out.dtype).filter(|_| !saturate) {
            for operand in &eqn.inputs {
                if let Operand::Lit(scalar) = operand
                    && !contract.admits(literal_value(*scalar))
                {
                    return Err(refuse(Capability::NarrowingRange { dtype: out.dtype }));
                }
            }
        }
        let operands = eqn
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(id) => Ok(LoweredOperand::Value(*id)),
                Operand::Lit(scalar) => arithmetic_scalar(*scalar, device_out, saturate)
                    .map(LoweredOperand::Scalar)
                    .map_err(refuse),
            })
            .collect::<Result<_, _>>()?;
        Ok(Lowered {
            op,
            operands,
            out: tensor(out_elem)?,
        })
    };
    // Data movement: bit exact, in any width the device carries; a literal keeps its own value and dtype.
    let movement = |op: VpuOp| -> Result<Lowered, PlanError> {
        for operand in &eqn.inputs {
            if elem_of(operand_dtype(operand)).is_none() {
                return Err(refuse(Capability::DtypeLowering));
            }
        }
        let operands = eqn
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(id) => LoweredOperand::Value(*id),
                Operand::Lit(scalar) => LoweredOperand::Scalar(carried_scalar(*scalar)),
            })
            .collect();
        Ok(Lowered {
            op,
            operands,
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
            if operand_dtype(&eqn.inputs[1]) != DType::I32 {
                return Err(refuse(Capability::DtypeLowering));
            }
            movement(op)
        }
        OpClass::UpdateSlice => {
            // The update has the operand's dtype and the index is an I32 word.
            if operand_dtype(&eqn.inputs[0]) != operand_dtype(&eqn.inputs[1])
                || operand_dtype(&eqn.inputs[2]) != DType::I32
            {
                return Err(refuse(Capability::DtypeLowering));
            }
            movement(op)
        }
        OpClass::Convert => {
            // F16 <-> F32 only: a conversion through an integer or BF16 changes the value. Inside a narrowed
            // region both sides are F16 and the conversion is the identity the contract already covers. An
            // integer source is refused with its reason: FP16 holds integers exactly only up to 2048 and the
            // value of an index or position is not bounded by anything the planner reads.
            let source = operand_dtype(&eqn.inputs[0]);
            if source == DType::I32 && device_out == DType::F16 {
                return Err(refuse(Capability::IntegerToFloat {
                    to: device_out,
                    exact_limit: F16_EXACT_INTEGERS,
                }));
            }
            let float = |dtype| matches!(dtype, DType::F16 | DType::F32);
            if !float(operand_dtype(&eqn.inputs[0])) || !float(out.dtype) {
                return Err(refuse(Capability::DtypeLowering));
            }
            movement(op)
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
    /// A `DynamicUpdateSlice`: movement of one dtype whose index operand is I32.
    UpdateSlice,
    /// A `Convert`: between F16 and F32.
    Convert,
}

/// The one opset op `op` lowers to, if the table has one. The ops are the design's primitive list; anything
/// else is refused by [`unsupported`], never approximated.
fn opset_op(op: &OpKind) -> Option<(VpuOp, OpClass)> {
    Some(match op {
        OpKind::Binary(BinOp::Add) => (VpuOp::Add, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Sub) => (VpuOp::Subtract, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Mul) => (VpuOp::Multiply, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Div) => (VpuOp::Divide, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Max) => (VpuOp::Maximum, OpClass::Arithmetic),
        OpKind::Binary(BinOp::Ge) => (VpuOp::GreaterEqual, OpClass::Arithmetic),
        OpKind::Unary(UnOp::Exp) => (VpuOp::Exp, OpClass::Arithmetic),
        OpKind::Unary(UnOp::Neg) => (VpuOp::Negative, OpClass::Arithmetic),
        OpKind::Unary(UnOp::Sqrt) => (VpuOp::Sqrt, OpClass::Arithmetic),
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
        OpKind::Reduce {
            op: RedOp::Max,
            axis,
            keepdim,
        } => (
            VpuOp::ReduceMax {
                axis: *axis,
                keepdim: *keepdim,
            },
            OpClass::Arithmetic,
        ),
        OpKind::MatMul => (VpuOp::MatMul, OpClass::Arithmetic),
        OpKind::Gather { axis } => (VpuOp::Gather { axis: *axis }, OpClass::Gather),
        OpKind::DynamicUpdateSlice { axis } => (
            VpuOp::DynamicUpdateSlice { axis: *axis },
            OpClass::UpdateSlice,
        ),
        OpKind::Iota { len, .. } => (VpuOp::Iota { len: *len }, OpClass::Movement),
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
        OpKind::Iota { .. }
        | elementwise_ops!()
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
        inputs.push(input_of(g, caps, id)?);
    }
    let mut nodes = Vec::with_capacity(lowered.len());
    for (eqn, lowered) in graph.eqns().iter().zip(lowered) {
        let origin = eqn.out.index() as u32;
        let mut inputs = Vec::with_capacity(lowered.operands.len());
        for operand in lowered.operands {
            inputs.push(match operand {
                LoweredOperand::Value(id) => numbering[&id],
                // A scalar constant is a node of its own, defined just before its consumer.
                LoweredOperand::Scalar(scalar) => {
                    let value = VpuValue((region.reads().len() + nodes.len()) as u32);
                    nodes.push(VpuNode {
                        origin,
                        op: VpuOp::Scalar(scalar),
                        inputs: Vec::new(),
                        out: VpuTensor {
                            elem: scalar.elem(),
                            shape: Vec::new(),
                        },
                    });
                    value
                }
            });
        }
        let value = VpuValue((region.reads().len() + nodes.len()) as u32);
        nodes.push(VpuNode {
            origin,
            op: lowered.op,
            inputs,
            out: lowered.out,
        });
        numbering.insert(eqn.out, value);
    }
    let outputs = region.writes().iter().map(|id| numbering[id]).collect();
    let interior = VpuRegion {
        name: "poot_region".into(),
        inputs,
        nodes,
        outputs,
    };
    Ok(narrow_to_device_precision(g, caps, &region, interior))
}

/// The planner pass that makes the narrowing of F32 values to the device's arithmetic width an explicit step
/// of the region (POOT-1178). `interior` is the region as [`lower_eqn`] planned it, every value the contract
/// narrows already an F16 one; the pass gives it the program's ABI back. Each narrowed region input is bound as
/// the F32 the program stores and followed by a [`VpuOp::NarrowEntry`] conversion, each narrowed result is
/// followed by a [`VpuOp::WidenExit`] conversion back to F32, and nothing in between is converted: the
/// contract bounds the region's error as a whole, so an interior reduction runs in the device's width once.
/// A value the contract does not narrow (every F16 and I32 one) crosses the boundary untouched.
fn narrow_to_device_precision(
    g: GraphTables<'_>,
    caps: &VpuCaps,
    region: &crate::Region,
    interior: VpuRegion,
) -> VpuRegion {
    let narrowed = |id: ValueId| narrowing_of(caps, g.aval(id).dtype).is_some();
    let entries: Vec<(usize, ValueId)> = region
        .reads()
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, id)| narrowed(id))
        .collect();
    // Node values are numbered after the inputs, so the entry conversions placed first push every interior
    // node value up by their count; an input that is converted is read through its conversion.
    let input_count = interior.inputs.len() as u32;
    let shift = entries.len() as u32;
    let entry_value: HashMap<u32, u32> = entries
        .iter()
        .enumerate()
        .map(|(n, &(input, _))| (input as u32, input_count + n as u32))
        .collect();
    let remap = |value: VpuValue| {
        if value.0 < input_count {
            VpuValue(entry_value.get(&value.0).copied().unwrap_or(value.0))
        } else {
            VpuValue(value.0 + shift)
        }
    };
    let mut inputs = interior.inputs;
    let mut nodes =
        Vec::with_capacity(entries.len() + interior.nodes.len() + region.writes().len());
    for &(input, id) in &entries {
        let narrowed_tensor = inputs[input].tensor.clone();
        inputs[input].tensor.elem =
            elem_of(g.aval(id).dtype).expect("an F32 value has a VPU element");
        nodes.push(VpuNode {
            origin: id.index() as u32,
            op: VpuOp::NarrowEntry,
            inputs: vec![VpuValue(input as u32)],
            out: narrowed_tensor,
        });
    }
    for mut node in interior.nodes {
        for operand in &mut node.inputs {
            *operand = remap(*operand);
        }
        nodes.push(node);
    }
    let mut outputs = Vec::with_capacity(interior.outputs.len());
    for (&id, &value) in region.writes().iter().zip(&interior.outputs) {
        let value = remap(value);
        if !narrowed(id) {
            outputs.push(value);
            continue;
        }
        let aval = g.aval(id);
        outputs.push(VpuValue(input_count + nodes.len() as u32));
        nodes.push(VpuNode {
            origin: id.index() as u32,
            op: VpuOp::WidenExit,
            inputs: vec![value],
            out: VpuTensor {
                elem: elem_of(aval.dtype).expect("an F32 value has a VPU element"),
                shape: aval.shape.clone(),
            },
        });
    }
    VpuRegion {
        name: interior.name,
        inputs,
        nodes,
        outputs,
    }
}

/// The region argument for value `id`, typed as the region's interior sees it (see
/// [`narrow_to_device_precision`] for the ABI type).
fn input_of(g: GraphTables<'_>, caps: &VpuCaps, id: ValueId) -> Result<VpuInput, PlanError> {
    let aval = g.aval(id);
    let elem = elem_of(device_dtype(caps, aval.dtype)).ok_or(PlanError::UnrepresentableValue {
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
            narrowing: None,
        }
    }

    /// [`caps`] with the numerical contract F32 values are narrowed under. SYNTHETIC like the rest: the bound is
    /// a placeholder, the planner reads only that a contract exists.
    fn narrowing_caps() -> VpuCaps {
        VpuCaps {
            narrowing: Some(NarrowingContract {
                max_relative_error_ppm: 10_000,
                max_magnitude: 65_504,
            }),
            ..caps()
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

    /// BF16 arithmetic is refused with the dtype named, and F32 arithmetic with the missing contract named,
    /// not computed in FP16 (ADR-0103, POOT-1178). A contract does not cover BF16.
    /// Mutation: drop the arithmetic check in `lower_eqn`; the BF16 add lowers and this row fails.
    #[test]
    fn bf16_arithmetic_is_refused_and_f32_arithmetic_names_the_missing_contract() {
        for (dtype, missing, caps) in [
            (
                DType::F32,
                Capability::NarrowingContract { dtype: DType::F32 },
                caps(),
            ),
            (
                DType::BF16,
                Capability::Arithmetic { dtype: DType::BF16 },
                caps(),
            ),
            (
                DType::BF16,
                Capability::Arithmetic { dtype: DType::BF16 },
                narrowing_caps(),
            ),
        ] {
            let g = graph_of(|b| {
                let x = b.constant("x", f(dtype, &[2, 3])).unwrap();
                let y = b.constant("y", f(dtype, &[2, 3])).unwrap();
                b.binary(BinOp::Add, x, y).unwrap()
            });
            let refused = refusal(plan_vpu_region(&g, &caps));
            assert_eq!(refused.missing, missing);
            assert_eq!(refused.eqn, g.eqns()[0].out);
            assert_eq!(
                refused.target,
                poot_target::TargetId::Vpu(VpuTargetId::Npu4)
            );
        }
    }

    /// An op outside the lowering table is a typed refusal, never an approximation; an integer conversion is
    /// refused for its own reason.
    #[test]
    fn ops_without_an_opset_lowering_are_refused_by_name() {
        let log = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            b.unary(UnOp::Log, x).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&log, &caps())).missing,
            Capability::VpuOpset
        );
        let recip = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[4])).unwrap();
            b.unary(UnOp::Recip, x).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&recip, &caps())).missing,
            Capability::VpuOpset
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

    /// The IR layers of `graph` planned for the NPU, as `type/version` in layer order.
    fn layer_kinds(graph: &Graph) -> Vec<String> {
        let ir = emit_vpu(&plan_vpu_region(graph, &caps()).unwrap()).unwrap();
        ir.xml
            .match_indices("<layer ")
            .map(|(at, _)| {
                let rest = &ir.xml[at..];
                let attr = |name: &str| {
                    let start = rest.find(name).unwrap() + name.len();
                    rest[start..start + rest[start..].find('"').unwrap()].to_string()
                };
                format!("{}/{}", attr("type=\""), attr("version=\""))
            })
            .collect()
    }

    /// Each newly lowered elementwise primitive is exactly one opset layer between its parameters and the
    /// result, with the opset's name for it.
    /// Mutation: lower `Sub` as `Add` in `opset_op`; the `Subtract` row fails.
    #[test]
    fn each_elementwise_primitive_lowers_to_its_opset_layer() {
        for (op, layer) in [
            (BinOp::Sub, "Subtract"),
            (BinOp::Div, "Divide"),
            (BinOp::Max, "Maximum"),
        ] {
            let g = graph_of(|b| {
                let x = b.constant("x", f(DType::F16, &[2, 3])).unwrap();
                let y = b.constant("y", f(DType::F16, &[2, 3])).unwrap();
                b.binary(op, x, y).unwrap()
            });
            assert_eq!(
                layer_kinds(&g),
                [
                    "Parameter/opset1".to_string(),
                    "Parameter/opset1".to_string(),
                    format!("{layer}/opset1"),
                    "Result/opset1".to_string(),
                ],
                "{op:?}"
            );
        }
        for (op, layer) in [(UnOp::Neg, "Negative"), (UnOp::Sqrt, "Sqrt")] {
            let g = graph_of(|b| {
                let x = b.constant("x", f(DType::F16, &[2, 3])).unwrap();
                b.unary(op, x).unwrap()
            });
            assert_eq!(
                layer_kinds(&g),
                [
                    "Parameter/opset1".to_string(),
                    format!("{layer}/opset1"),
                    "Result/opset1".to_string(),
                ],
                "{op:?}"
            );
        }
    }

    /// `Ge` is poot's `1.0` or `0.0` in the operands' dtype, so it is a boolean `GreaterEqual` followed by a
    /// `Convert` back to F16: the IR text is pinned.
    /// Mutation: spell the `Convert`'s destination type as `f32` in the `GreaterEqual` emission; the pinned text differs.
    #[test]
    fn ge_is_a_comparison_converted_to_the_operand_dtype() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[2])).unwrap();
            let y = b.constant("y", f(DType::F16, &[2])).unwrap();
            b.binary(BinOp::Ge, x, y).unwrap()
        });
        let ir = emit_vpu(&plan_vpu_region(&g, &caps()).unwrap()).unwrap();
        let expected = concat!(
            r#"<?xml version="1.0"?><net name="poot_region" version="11"><layers>"#,
            r#"<layer id="0" name="v0" type="Parameter" version="opset1"><data shape="2" element_type="f16"/><output><port id="0" precision="FP16" names="v0"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="1" name="v1" type="Parameter" version="opset1"><data shape="2" element_type="f16"/><output><port id="0" precision="FP16" names="v1"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="2" name="v2_flags" type="GreaterEqual" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim></port><port id="1" precision="FP16"><dim>2</dim></port></input><output><port id="2" precision="BOOL" names="v2_flags"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="3" name="v2" type="Convert" version="opset1"><data destination_type="f16"/><input><port id="0" precision="BOOL"><dim>2</dim></port></input><output><port id="1" precision="FP16" names="v2"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="4" name="result_0" type="Result" version="opset1"><input><port id="0" precision="FP16"><dim>2</dim></port></input></layer>"#,
            r#"</layers><edges>"#,
            r#"<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>"#,
            r#"<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/><edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>"#,
            r#"</edges></net>"#,
        );
        assert_eq!(ir.xml, expected);
    }

    /// `max` over an axis is `ReduceMax` with the axis as an `i64` constant, as `ReduceSum` is.
    /// Mutation: lower `RedOp::Max` as `ReduceSum`; the layer type differs.
    #[test]
    fn a_max_reduction_lowers_to_reduce_max() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[2, 3])).unwrap();
            b.reduce(RedOp::Max, x, 1, true).unwrap()
        });
        let ir = emit_vpu(&plan_vpu_region(&g, &caps()).unwrap()).unwrap();
        assert!(
            ir.xml
                .contains(r#"type="ReduceMax" version="opset1"><data keep_dims="true"/>"#),
            "{}",
            ir.xml
        );
        assert_eq!(ir.weights, 1i64.to_le_bytes());
    }

    /// An inline scalar literal is a rank-0 constant of the arithmetic width, defined just before its
    /// consumer: `x * 0.125` over F16 is `Const f16 0x3000` feeding `Multiply`, not a refusal. The IR text and
    /// the weight blob are pinned.
    /// Mutation: lower a literal as `VpuScalar::F32`; the constant would be f32 beside an f16 operand.
    #[test]
    fn a_scalar_literal_lowers_to_a_rank_zero_constant_of_the_operands_width() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F16, &[2])).unwrap();
            b.binary_scalar(BinOp::Mul, x, Scalar::F32(0.125)).unwrap()
        });
        let region = plan_vpu_region(&g, &caps()).unwrap();
        assert_eq!(region.nodes.len(), 2);
        assert_eq!(
            region.nodes[0].op,
            VpuOp::Scalar(poot_codegen::VpuScalar::F16(0x3000))
        );
        assert_eq!(region.nodes[1].op, VpuOp::Multiply);
        let ir = emit_vpu(&region).unwrap();
        let expected = concat!(
            r#"<?xml version="1.0"?><net name="poot_region" version="11"><layers>"#,
            r#"<layer id="0" name="v0" type="Parameter" version="opset1"><data shape="2" element_type="f16"/><output><port id="0" precision="FP16" names="v0"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="1" name="aux_1" type="Const" version="opset1"><data element_type="f16" shape="" offset="0" size="2"/><output><port id="0" precision="FP16"></port></output></layer>"#,
            r#"<layer id="2" name="v1" type="Multiply" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim></port><port id="1" precision="FP16"></port></input><output><port id="2" precision="FP16" names="v1"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="3" name="result_0" type="Result" version="opset1"><input><port id="0" precision="FP16"><dim>2</dim></port></input></layer>"#,
            r#"</layers><edges>"#,
            r#"<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>"#,
            r#"<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>"#,
            r#"</edges></net>"#,
        );
        assert_eq!(ir.xml, expected);
        assert_eq!(ir.weights, 0x3000u16.to_le_bytes());
        // The constant is no layer the compiler is asked about: only the equation's own layers are reported.
        assert_eq!(region.layers().collect::<Vec<_>>(), [("v1".to_string(), 1)]);
    }

    /// A literal the device's width cannot hold is refused, never saturated to infinity; an F32 literal beside
    /// an F32 tensor is still the F32 arithmetic refusal.
    /// Mutation: drop the finiteness check in `arithmetic_scalar`; the F16 row lowers `inf`.
    #[test]
    fn a_literal_the_arithmetic_width_cannot_hold_is_refused() {
        let scaled = |dtype: DType, value: f32| {
            graph_of(|b| {
                let x = b.constant("x", f(dtype, &[2])).unwrap();
                b.binary_scalar(BinOp::Mul, x, Scalar::F32(value)).unwrap()
            })
        };
        assert_eq!(
            refusal(plan_vpu_region(&scaled(DType::F16, 1.0e10), &caps())).missing,
            Capability::DtypeLowering
        );
        assert_eq!(
            refusal(plan_vpu_region(&scaled(DType::F32, 2.0), &caps())).missing,
            Capability::NarrowingContract { dtype: DType::F32 }
        );
        assert!(plan_vpu_region(&scaled(DType::F16, 65504.0), &caps()).is_ok());
    }

    /// Under the narrowing contract a finite F32 literal past FP16's range saturates to the largest finite FP16
    /// value in an additive position (`Add`, `Sub`: the mask sentinel); an infinite literal is exact and stays
    /// infinite; a NaN literal is refused.
    /// Mutation: drop the `saturate` clamp in `arithmetic_scalar`; the `1e9` literal is refused.
    #[test]
    fn a_finite_additive_literal_past_f16_saturates_under_the_narrowing_contract() {
        let lowered = |op: BinOp, value: f32| {
            let g = graph_of(|b| {
                let x = b.constant("x", f(DType::F32, &[2])).unwrap();
                b.binary_scalar(op, x, Scalar::F32(value)).unwrap()
            });
            plan_vpu_region(&g, &narrowing_caps())
        };
        let constant = |op: BinOp, value: f32| {
            lowered(op, value)
                .unwrap()
                .nodes
                .iter()
                .find_map(|node| match node.op {
                    VpuOp::Scalar(scalar) => Some(scalar),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(
            constant(BinOp::Sub, 1.0e9),
            poot_codegen::VpuScalar::F16(0x7BFF)
        );
        assert_eq!(
            constant(BinOp::Add, -1.0e9),
            poot_codegen::VpuScalar::F16(0xFBFF)
        );
        assert_eq!(
            constant(BinOp::Add, f32::NEG_INFINITY),
            poot_codegen::VpuScalar::F16(0xFC00)
        );
        assert_eq!(
            refusal(lowered(BinOp::Add, f32::NAN)).missing,
            Capability::DtypeLowering
        );
    }

    /// A literal past FP16's range in a multiplicative position is refused even under the contract, because
    /// `x * 1e9` saturated to `x * 65504` is a silent wrong result.
    /// Mutation: make `saturate` true for every op in `lower_eqn`; `Mul` and `Div` by `1e9` lower.
    #[test]
    fn a_multiplicative_literal_past_f16_is_refused_under_the_narrowing_contract() {
        for op in [BinOp::Mul, BinOp::Div] {
            let g = graph_of(|b| {
                let x = b.constant("x", f(DType::F32, &[2])).unwrap();
                b.binary_scalar(op, x, Scalar::F32(1.0e9)).unwrap()
            });
            assert_eq!(
                refusal(plan_vpu_region(&g, &narrowing_caps())).missing,
                Capability::NarrowingRange { dtype: DType::F32 },
                "{op:?}"
            );
        }
    }

    /// The contract's range check on literals, at the edge and against POOT-1180's saturation: a literal within
    /// +-65504 plans in any position; past it, a multiplicative or divisive one is a typed `NarrowingRange` refusal, while the additive literals 1180 saturates (`Add`, `Sub`,
    /// finite or infinite) are exempt and plan.
    /// Mutation: drop the `admits` check in `lower_eqn`; 65505 in `Mul` plans (F16 rounds it to 65504). Make
    /// the check ignore `saturate`; the additive rows are refused.
    #[test]
    fn a_literal_outside_the_contracts_range_is_refused_unless_it_saturates() {
        let with = |op: BinOp, value: f32| {
            plan_vpu_region(
                &graph_of(|b| {
                    let x = b.constant("x", f(DType::F32, &[2])).unwrap();
                    b.binary_scalar(op, x, Scalar::F32(value)).unwrap()
                }),
                &narrowing_caps(),
            )
        };
        for op in [BinOp::Mul, BinOp::Div, BinOp::Add, BinOp::Sub] {
            assert!(with(op, 65_504.0).is_ok(), "{op:?} at the edge");
            assert!(with(op, -65_504.0).is_ok(), "{op:?} at the negative edge");
        }
        for op in [BinOp::Mul, BinOp::Div] {
            for value in [65_505.0, -65_505.0, 1.0e10, f32::INFINITY] {
                assert_eq!(
                    refusal(with(op, value)).missing,
                    Capability::NarrowingRange { dtype: DType::F32 },
                    "{op:?} {value}"
                );
            }
        }
        for op in [BinOp::Add, BinOp::Sub] {
            for value in [65_505.0, -1.0e10, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(with(op, value).is_ok(), "{op:?} {value} saturates");
            }
        }
    }

    /// An integer converts to a float inside a narrowed region only if the integer is exact in FP16, which the
    /// planner cannot show for a position or an index: the refusal names the exact-integer limit.
    /// Mutation: delete the I32-source check in the `Convert` arm; the cast lowers and the refusal is gone.
    #[test]
    fn an_integer_to_float_cast_is_refused_with_its_exact_range() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::I32, &[2])).unwrap();
            b.cast(x, DType::F32).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&g, &narrowing_caps())).missing,
            Capability::IntegerToFloat {
                to: DType::F16,
                exact_limit: 2048
            }
        );
    }

    /// A nullary range is a constant of its element type; the blob holds the values.
    /// Mutation: start the I32 range at 1 in the `Iota` emission; the blob bytes differ.
    #[test]
    fn an_iota_lowers_to_a_constant_range() {
        let g = graph_of(|b| b.iota_i32(4).unwrap());
        let ir = emit_vpu(&plan_vpu_region(&g, &caps()).unwrap()).unwrap();
        assert!(
            ir.xml.contains(
                r#"<layer id="0" name="aux_0" type="Const" version="opset1"><data element_type="i32" shape="4" offset="0" size="16"/>"#
            ),
            "{}",
            ir.xml
        );
        let words: Vec<u8> = (0..4i32).flat_map(i32::to_le_bytes).collect();
        assert_eq!(ir.weights, words);
    }

    /// `DynamicUpdateSlice` is a `ScatterUpdate` at `range + index`: a literal index is a constant beside the
    /// range, a runtime index is a region argument, and the layer sequence differs by exactly that constant.
    /// Mutation: drop the index `Add` and scatter at the bare range; the sequence loses `Add`.
    #[test]
    fn a_dynamic_update_slice_scatters_at_the_range_plus_the_index() {
        let literal = graph_of(|b| {
            let cache = b.constant("cache", f(DType::F16, &[6, 4])).unwrap();
            let row = b.constant("row", f(DType::F16, &[2, 4])).unwrap();
            b.dynamic_update_slice(cache, row, 3, 0).unwrap()
        });
        let literal_region = plan_vpu_region(&literal, &caps()).unwrap();
        assert_eq!(
            literal_region.nodes[0].op,
            VpuOp::Scalar(poot_codegen::VpuScalar::I32(3))
        );
        assert_eq!(
            layer_kinds(&literal),
            [
                "Parameter/opset1",
                "Parameter/opset1",
                "Const/opset1",
                "Const/opset1",
                "Add/opset1",
                "Const/opset1",
                "ScatterUpdate/opset3",
                "Result/opset1",
            ]
        );
        let runtime = graph_of(|b| {
            let cache = b.constant("cache", f(DType::F16, &[6, 4])).unwrap();
            let row = b.constant("row", f(DType::F16, &[2, 4])).unwrap();
            let pos = b.constant("pos", f(DType::I32, &[])).unwrap();
            b.dynamic_update_slice_dyn(cache, row, pos, 0).unwrap()
        });
        let ir = emit_vpu(&plan_vpu_region(&runtime, &caps()).unwrap()).unwrap();
        assert_eq!(
            ir.xml.matches("type=\"Const\"").count(),
            2,
            "the range and the axis, no literal: {}",
            ir.xml
        );
        // The range 0..2 is the first blob entry; the axis follows as an i64.
        let mut blob: Vec<u8> = (0..2i32).flat_map(i32::to_le_bytes).collect();
        blob.extend_from_slice(&0i64.to_le_bytes());
        assert_eq!(ir.weights, blob);
    }

    /// A scatter of rows of one dtype by an I32 index only: the update must share the operand's dtype.
    #[test]
    fn a_dynamic_update_slice_with_an_f32_update_over_an_f16_operand_is_refused() {
        let g = graph_of(|b| {
            let cache = b.constant("cache", f(DType::F16, &[6, 4])).unwrap();
            let row = b.constant("row", f(DType::F32, &[2, 4])).unwrap();
            let pos = b.constant("pos", f(DType::I32, &[])).unwrap();
            b.dynamic_update_slice_dyn(cache, row, pos, 0).unwrap()
        });
        assert_eq!(
            refusal(plan_vpu_region(&g, &caps())).missing,
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
            Capability::NarrowingContract { dtype: DType::F32 }
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

    /// `sum(x * w + x, axis 1)` over F32 `[2, 3]` under a narrowing contract: the region's IR text, pinned. The
    /// two F32 arguments are narrowed by a `Convert` each on entry (`v0_narrow`, `v1_narrow`), the whole
    /// interior (`v2` multiply, `v3` add, `v4` sum) runs in FP16 with nothing converted between, and the F32
    /// result is widened by one `Convert` (`v4_widen`) that is the only layer the `Result` reads.
    /// Mutation: skip the `WidenExit` push in `narrow_to_device_precision`; the result is FP16 and the text differs.
    #[test]
    fn an_f32_chain_is_narrowed_at_the_region_boundary_only() {
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F32, &[2, 3])).unwrap();
            let w = b.constant("w", f(DType::F32, &[2, 3])).unwrap();
            let scaled = b.binary(BinOp::Mul, x, w).unwrap();
            let shifted = b.binary(BinOp::Add, scaled, x).unwrap();
            b.reduce(RedOp::Sum, shifted, 1, false).unwrap()
        });
        let region = plan_vpu_region(&g, &narrowing_caps()).unwrap();
        let ir = emit_vpu(&region).unwrap();
        let expected = concat!(
            r#"<?xml version="1.0"?><net name="poot_region" version="11"><layers>"#,
            r#"<layer id="0" name="v0" type="Parameter" version="opset1"><data shape="2,3" element_type="f32"/><output><port id="0" precision="FP32" names="v0"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="1" name="v1" type="Parameter" version="opset1"><data shape="2,3" element_type="f32"/><output><port id="0" precision="FP32" names="v1"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="2" name="v0_narrow" type="Convert" version="opset1"><data destination_type="f16"/><input><port id="0" precision="FP32"><dim>2</dim><dim>3</dim></port></input><output><port id="1" precision="FP16" names="v0_narrow"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="3" name="v1_narrow" type="Convert" version="opset1"><data destination_type="f16"/><input><port id="0" precision="FP32"><dim>2</dim><dim>3</dim></port></input><output><port id="1" precision="FP16" names="v1_narrow"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="4" name="v2" type="Multiply" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="FP16"><dim>2</dim><dim>3</dim></port></input><output><port id="2" precision="FP16" names="v2"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="5" name="v3" type="Add" version="opset1"><data auto_broadcast="numpy"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="FP16"><dim>2</dim><dim>3</dim></port></input><output><port id="2" precision="FP16" names="v3"><dim>2</dim><dim>3</dim></port></output></layer>"#,
            r#"<layer id="6" name="aux_6" type="Const" version="opset1"><data element_type="i64" shape="1" offset="0" size="8"/><output><port id="0" precision="I64"><dim>1</dim></port></output></layer>"#,
            r#"<layer id="7" name="v4" type="ReduceSum" version="opset1"><data keep_dims="false"/><input><port id="0" precision="FP16"><dim>2</dim><dim>3</dim></port><port id="1" precision="I64"><dim>1</dim></port></input><output><port id="2" precision="FP16" names="v4"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="8" name="v4_widen" type="Convert" version="opset1"><data destination_type="f32"/><input><port id="0" precision="FP16"><dim>2</dim></port></input><output><port id="1" precision="FP32" names="v4_widen"><dim>2</dim></port></output></layer>"#,
            r#"<layer id="9" name="result_0" type="Result" version="opset1"><input><port id="0" precision="FP32"><dim>2</dim></port></input></layer>"#,
            r#"</layers><edges>"#,
            r#"<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>"#,
            r#"<edge from-layer="2" from-port="1" to-layer="4" to-port="0"/><edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>"#,
            r#"<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/><edge from-layer="2" from-port="1" to-layer="5" to-port="1"/>"#,
            r#"<edge from-layer="5" from-port="2" to-layer="7" to-port="0"/><edge from-layer="6" from-port="0" to-layer="7" to-port="1"/>"#,
            r#"<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/><edge from-layer="8" from-port="1" to-layer="9" to-port="0"/>"#,
            r#"</edges></net>"#,
        );
        assert_eq!(ir.xml, expected);
        assert_eq!(region.output_names().collect::<Vec<_>>(), ["v4_widen"]);
    }

    /// Only a value the contract narrows crosses the boundary through a conversion: an F16 argument and an F16
    /// result are bound as they are, an F32 argument beside them is narrowed once, and the interior reads each
    /// through the right value (the argument order does not decide which one is converted).
    /// Mutation: convert every region input in `narrow_to_device_precision`; the F16 argument gains a
    /// `NarrowEntry` and the op list differs.
    #[test]
    fn a_value_the_contract_does_not_narrow_crosses_the_boundary_untouched() {
        let g = graph_of(|b| {
            let a = b.constant("a", f(DType::F16, &[4])).unwrap();
            let c = b.constant("c", f(DType::F32, &[4])).unwrap();
            let doubled = b.binary(BinOp::Add, a, a).unwrap();
            let c16 = b.cast(c, DType::F16).unwrap();
            b.binary(BinOp::Add, doubled, c16).unwrap()
        });
        let region = plan_vpu_region(&g, &narrowing_caps()).unwrap();
        let elems: Vec<_> = region
            .inputs
            .iter()
            .map(|input| input.tensor.elem)
            .collect();
        assert_eq!(elems, [VpuElem::F16, VpuElem::F32]);
        let ops: Vec<_> = region.nodes.iter().map(|node| node.op.clone()).collect();
        assert_eq!(
            ops,
            [VpuOp::NarrowEntry, VpuOp::Add, VpuOp::Convert, VpuOp::Add]
        );
        // Values 0 and 1 are the arguments, the node values follow: the F16 argument is read as it is, the F32
        // one only through its conversion (value 2).
        let reads: Vec<_> = region
            .nodes
            .iter()
            .map(|node| node.inputs.iter().map(|v| v.0).collect::<Vec<_>>())
            .collect();
        assert_eq!(reads, [vec![1], vec![0, 0], vec![2], vec![3, 4]]);
        assert_eq!(region.outputs, [poot_codegen::VpuValue(5)]);
        assert!(emit_vpu(&region).is_ok());
    }

    /// The production entry: `compile` for an Intel NPU target that records a contract plans the F32 graph that the
    /// contract-less target refuses, and the program's one graph carries the boundary conversions.
    /// Mutation: make `computes` ignore the contract (`.ok_or(..)` on `None`); the compile is refused again.
    #[test]
    fn compile_for_an_intel_npu_with_a_contract_narrows_an_f32_graph() {
        let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
        let g = graph_of(|b| {
            let x = b.constant("x", f(DType::F32, &[2, 3])).unwrap();
            let y = b.constant("y", f(DType::F32, &[2, 3])).unwrap();
            b.binary(BinOp::Add, x, y).unwrap()
        });
        let target = |caps| Target::Vpu {
            target: VpuTargetId::Npu4,
            caps,
        };
        let program = crate::compile(&g, &target(narrowing_caps()), &options).unwrap();
        let ops: Vec<_> = program
            .vpu_region()
            .unwrap()
            .nodes
            .iter()
            .map(|node| node.op.clone())
            .collect();
        assert_eq!(
            ops,
            [
                VpuOp::NarrowEntry,
                VpuOp::NarrowEntry,
                VpuOp::Add,
                VpuOp::WidenExit
            ]
        );
        assert!(matches!(
            crate::compile(&g, &target(caps()), &options),
            Err(CompileError::Plan(_))
        ));
    }
}
