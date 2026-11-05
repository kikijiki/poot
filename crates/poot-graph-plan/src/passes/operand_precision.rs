//! `select_operand_precision` (POOT-1430): the one pass that writes a non-exact [`OperandPrecision`].
//!
//! The pass reads three things and nothing else: the driver's [`MatmulPrecision`] policy, the float dtypes the
//! target's matrix units take ([`FragmentUnits`]), and the dtypes of the op's operands. It never reads a model, a
//! shape, a benchmark or an environment variable. The policy is a permission: a target with no matrix unit for the
//! op's operands leaves the op `Exact` under any policy, because narrowing there would lose accuracy for no speed.
//! What the planner then makes of a narrowed op (a fragment micro-tile, a scalar micro-tile, a GEMV) is the
//! planner's business; every schedule computes the same declared rounding.

use poot_graph_ir::graph::{EditError, Eqn, Graph, Operand};
use poot_graph_ir::op::{ContractionHead, OpKind};
use poot_graph_ir::{FragmentDtype, MatmulPrecision, OperandPrecision};
use poot_target::{DeviceCaps, FragmentShape, MatrixUnits};

use crate::Target;
use poot_tensor::DType;

/// The matrix units a target reports (`DeviceCaps::matrix_units`), as the pass reads them: it takes a fragment
/// dtype when the device has a 16x16x16 config for it, the one shape the kernels lower. A device that cannot say
/// what it has (`Queried::Unknown`) has no unit the pass will narrow for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FragmentUnits(MatrixUnits);

impl FragmentUnits {
    /// The units `caps` reports. This is the one place the planner reads the device's matrix hardware for
    /// operand precision.
    pub(crate) fn of(caps: &DeviceCaps) -> Self {
        Self(caps.matrix_units.known().unwrap_or(MatrixUnits::NONE))
    }

    fn takes(self, dtype: FragmentDtype) -> bool {
        self.0
            .config(dtype)
            .is_some_and(|config| config.shape == FragmentShape::M16N16K16)
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says on `target`: the
/// graph `compile`'s `select_operand_precision` pass writes, before the passes that fold and fuse it. It is the
/// graph a CPU oracle evaluates to hold a device that compiled `g` under `policy` to the declared semantics (a
/// program narrowed by the planner is compared with the oracle of the narrowed ops, not of the exact ones). A
/// target with no matrix unit, or the policy `Exact`, returns `g` unchanged.
#[cfg(any(test, feature = "test-support"))]
pub fn with_operand_precision(
    g: &Graph,
    target: &Target,
    policy: MatmulPrecision,
) -> Result<Graph, EditError> {
    match target {
        Target::Gpu { caps, .. } => select_operand_precision(g, policy, FragmentUnits::of(caps)),
        Target::Npu(_) => Ok(g.clone()),
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says, where `units` admit
/// it. The policy `Exact` returns `g` unchanged.
pub(crate) fn select_operand_precision(
    g: &Graph,
    policy: MatmulPrecision,
    units: FragmentUnits,
) -> Result<Graph, EditError> {
    if policy == MatmulPrecision::Exact {
        return Ok(g.clone());
    }
    let mut edit = g.edit();
    for eqn in edit.eqns_mut() {
        if eqn.op.operand_precision() != Some(OperandPrecision::Exact) {
            continue;
        }
        if let Some(dtype) = fragment_dtype(g, eqn, units) {
            eqn.op = eqn
                .op
                .with_operand_precision(policy.operand_precision(dtype));
        }
    }
    edit.commit()
}

/// The fragment dtype `eqn`'s products run in on `units`, or `None` when no unit admits the equation.
///
/// - A contraction takes bf16 where the target has it, else f16 (bf16 keeps the f32 exponent range, so narrowing
///   an activation can never overflow). An operand stored as f16 forces f16: rounding it to bf16 would lose
///   bits it holds.
/// - Attention takes the dtype its keys are stored in when the target has a fragment for it (f16 is the default
///   KV dtype), so K and V enter exactly and only Q and P round. A bf16 cache takes bf16 where there is a unit
///   and f16 otherwise; an f32 cache takes f16.
fn fragment_dtype(g: &Graph, eqn: &Eqn, units: FragmentUnits) -> Option<FragmentDtype> {
    let candidates: &[FragmentDtype] = match &eqn.op {
        OpKind::Attention(_) | OpKind::SelectedAttention(_) => match value_dtype(g, eqn, 1)? {
            DType::BF16 => &[FragmentDtype::Bf16, FragmentDtype::F16],
            _ => &[FragmentDtype::F16],
        },
        _ => {
            let operands = contraction_operands(g, eqn)?;
            if operands.contains(&DType::F16) {
                &[FragmentDtype::F16]
            } else {
                &[FragmentDtype::Bf16, FragmentDtype::F16]
            }
        }
    };
    candidates.iter().copied().find(|&dtype| units.takes(dtype))
}

/// The dtypes of the float operands a contraction multiplies: its activation and, where the weight is a graph
/// operand of a float dtype, its weight. A packed weight is decoded inside the kernel and has no dtype of its
/// own. `None` when an operand is not a float, which no matrix unit takes.
fn contraction_operands(g: &Graph, eqn: &Eqn) -> Option<Vec<DType>> {
    let head = match &eqn.op {
        OpKind::ContractionEpilogue { head, .. } => *head,
        op => head_of(op)?,
    };
    let dtypes: Vec<DType> = match head {
        ContractionHead::MatMul { .. } => vec![value_dtype(g, eqn, 0)?, value_dtype(g, eqn, 1)?],
        ContractionHead::Dense { weight, .. } => vec![value_dtype(g, eqn, 0)?, weight],
        ContractionHead::Packed { .. } => vec![value_dtype(g, eqn, 0)?],
        ContractionHead::Indexed { .. } => vec![value_dtype(g, eqn, 0)?, value_dtype(g, eqn, 1)?],
    };
    dtypes
        .iter()
        .all(|dtype| matches!(dtype, DType::F32 | DType::BF16 | DType::F16))
        .then_some(dtypes)
}

/// The head a bare contraction equation stands for.
fn head_of(op: &OpKind) -> Option<ContractionHead> {
    let precision = op.operand_precision()?;
    Some(match op {
        OpKind::MatMul { .. } => ContractionHead::MatMul { precision },
        OpKind::DenseContraction { weight, .. }
        | OpKind::GroupedDenseContraction { weight, .. } => ContractionHead::Dense {
            weight: *weight,
            precision,
        },
        OpKind::PackedContraction {
            descriptor, rows, ..
        } => ContractionHead::Packed {
            descriptor: *descriptor,
            rows: *rows,
            precision,
        },
        OpKind::IndexedMatMul { .. } => ContractionHead::Indexed { precision },
        _ => return None,
    })
}

fn value_dtype(g: &Graph, eqn: &Eqn, position: usize) -> Option<DType> {
    match eqn.inputs.get(position)? {
        Operand::Value(value) => Some(g.aval(*value).dtype),
        Operand::Lit(_) => None,
    }
}
