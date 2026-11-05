//! `select_operand_precision` (POOT-1430): the one pass that writes a non-exact [`OperandPrecision`].
//!
//! The pass reads three things and nothing else: the driver's [`MatmulPrecision`] policy, the float dtypes the
//! target's matrix units take ([`FragmentUnits`]), and the dtypes of the op's operands. It never reads a model, a
//! shape, a benchmark or an environment variable. The policy is a permission: a target with no matrix unit for the
//! op's operands leaves the op `Exact` under any policy, because narrowing there would lose accuracy for no speed.
//! What the planner then makes of a narrowed op (a fragment micro-tile, a scalar micro-tile, a GEMV) is the
//! planner's business; every schedule computes the same declared rounding.
//!
//! No operand is range-scaled: a value past f16's range rounds to infinity and reaches the logits as a non-finite
//! value, which serving refuses with a typed error (`receipts/1471-narrow-without-range-guards/design.md`).

use poot_graph_ir::graph::{EditError, Eqn, Graph, Operand};
use poot_graph_ir::op::{ContractionHead, OpKind};
use poot_graph_ir::{FragmentDtype, KeyLayout, MatmulPrecision, OperandPrecision};
use poot_target::{DeviceCaps, FragmentShape, MatrixUnits};

#[cfg(any(test, feature = "test-support"))]
use super::weight_format::{store_weights_f16, weights_are_f16};
#[cfg(any(test, feature = "test-support"))]
use crate::Target;
#[cfg(any(test, feature = "test-support"))]
use crate::dtype_widen::{fold_dense_bf16_row_gathers, fold_dense_contractions};
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

    /// These units cut down to the `dtype` config alone, or to none when the device has no such config.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn only(self, dtype: FragmentDtype) -> Self {
        Self(MatrixUnits::of(self.0.config(dtype).cloned()))
    }

    pub(super) fn takes(self, dtype: FragmentDtype) -> bool {
        self.0
            .config(dtype)
            .is_some_and(|config| config.shape == FragmentShape::M16N16K16)
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says on `target`: the
/// graph `compile` hands the planner with the weight format and precision selected, before the passes that fuse it. It is the
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
        Target::Gpu { caps, .. } => {
            let units = FragmentUnits::of(caps);
            // The weight format is a property of the folded graph (`compile` converts the weights the folds
            // expose as dense readers), so the oracle's graph folds them the same way before it converts.
            if weights_are_f16(policy, units) {
                let folded = fold_dense_bf16_row_gathers(&fold_dense_contractions(g)?)?;
                select_operand_precision(&store_weights_f16(&folded)?, policy, units)
            } else {
                select_operand_precision(g, policy, units)
            }
        }
        Target::Npu(_) => Ok(g.clone()),
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says, where `units` admit
/// it. The policy `Exact` returns `g` unchanged. No operand is scaled: a value past f16's range rounds to
/// infinity and the step's logits are refused (`receipts/1471-narrow-without-range-guards/design.md` section 4).
pub(crate) fn select_operand_precision(
    g: &Graph,
    policy: MatmulPrecision,
    units: FragmentUnits,
) -> Result<Graph, EditError> {
    if policy == MatmulPrecision::Exact {
        return Ok(g.clone());
    }
    let mut edit = g.edit();
    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns().len());
    for eqn in g.eqns() {
        let mut eqn = eqn.clone();
        if eqn.op.operand_precision() == Some(OperandPrecision::Exact)
            && let Some(dtype) = fragment_dtype(g, &eqn, units)
        {
            eqn.op = eqn
                .op
                .with_operand_precision(policy.operand_precision(dtype));
        }
        eqns.push(eqn);
    }
    *edit.eqns_mut() = eqns;
    edit.commit()
}

/// The fragment dtype `eqn`'s products run in on `units`, or `None` when no unit admits the equation.
///
/// Every contraction and attention takes f16 where the target has an f16 fragment, else bf16, whatever its operand
/// dtypes: f16 has three more mantissa bits than bf16 (24-41x lower mean KL,
/// `receipts/1454-narrow-dtype-policy/design.md` section 1) and costs at most 3% of TTFT and TPOT over bf16 on ROCm
/// (`receipts/1458-f16-rounding/design.md` section 6). An operand stored as f16 forces f16 (rounding it to bf16
/// would lose bits it holds). An operand past f16's range becomes infinity; nothing scales it.
fn fragment_dtype(g: &Graph, eqn: &Eqn, units: FragmentUnits) -> Option<FragmentDtype> {
    use FragmentDtype::{Bf16, F16};
    let candidates: &[FragmentDtype] = match &eqn.op {
        OpKind::Attention(_) | OpKind::SelectedAttention(_) => {
            let mut operands = key_value_dtypes(g, eqn)?;
            operands.push(value_dtype(g, eqn, 0)?);
            if !operands
                .iter()
                .all(|dtype| matches!(dtype, DType::F32 | DType::BF16 | DType::F16))
            {
                return None;
            }
            if operands.contains(&DType::F16) {
                &[F16]
            } else {
                &[F16, Bf16]
            }
        }
        _ => {
            if contraction_operands(g, eqn)?.contains(&DType::F16) {
                &[F16]
            } else {
                &[F16, Bf16]
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

/// The dtypes an attention's keys and values are stored in: the key, the value and a split key's shared part
/// (a paged key's operand 4 is its I32 slot map).
fn key_value_dtypes(g: &Graph, eqn: &Eqn) -> Option<Vec<DType>> {
    let shared =
        matches!(&eqn.op, OpKind::Attention(attrs) if matches!(attrs.key, KeyLayout::Split { .. }));
    let positions: &[usize] = if shared { &[1, 2, 4] } else { &[1, 2] };
    positions
        .iter()
        .map(|&position| value_dtype(g, eqn, position))
        .collect()
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
        OpKind::IndexedMatMul { rows, .. } => ContractionHead::Indexed {
            rows: *rows,
            precision,
        },
        _ => return None,
    })
}

fn value_dtype(g: &Graph, eqn: &Eqn, position: usize) -> Option<DType> {
    match eqn.inputs.get(position)? {
        Operand::Value(value) => Some(g.aval(*value).dtype),
        Operand::Lit(_) => None,
    }
}
