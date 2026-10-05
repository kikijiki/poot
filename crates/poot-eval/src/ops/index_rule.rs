//! The one index-validity rule every index-consuming op shares (R470-008, R470-011): `Gather`,
//! `Scatter`, `ScatterUpdate`'s non-keep-base rows, `DynamicUpdateSlice`'s runtime start, `ArgTopK`'s
//! rank and `IndexedMatMul`'s expert id all read one row-selecting integer through [`index_at`]
//! instead of each restating its own rounding and bounds check.

use poot_graph_ir::ValueId;

use crate::EvalError;

/// An index value as the graph's two index dtypes carry it ([`poot_tensor::DType::is_index_operand`]):
/// an authoritative I32 word, or the dense walk's F32 index convention (an exact integer encoded as
/// f32, never pre-rounded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexValue {
    I32(i32),
    F32(f32),
}

impl std::fmt::Display for IndexValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexValue::I32(v) => write!(f, "{v}"),
            IndexValue::F32(v) => write!(f, "{v}"),
        }
    }
}

/// Why [`index_at`] refused a value, checked in this order: a non-finite F32 value is [`Self::NotFinite`]
/// before its sign or fraction are inspected; a finite value with a nonzero fraction is
/// [`Self::NotIntegral`] before its sign; a finite integral negative value (I32 or F32) is
/// [`Self::Negative`]; only a finite, integral, non-negative value can reach [`Self::OutOfRange`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFaultKind {
    /// An F32 value is NaN or infinite.
    NotFinite,
    /// An F32 value has a nonzero fractional part.
    NotIntegral,
    /// The value is negative.
    Negative,
    /// The value is a non-negative integer but `>= len`.
    OutOfRange,
}

/// One index read refused: `value` at `position` of the index operand cannot select a row of
/// `0..len` (deval 4). The oracle's one index-validity refusal; every index-consuming op raises this
/// instead of restating its own rounding or bounds check.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("index v{eqn} position {position} value {value} is not a valid row of 0..{len} ({kind:?})")]
pub struct IndexFault {
    pub eqn: ValueId,
    pub position: usize,
    pub value: IndexValue,
    pub len: usize,
    pub kind: IndexFaultKind,
}

/// The one row-selection rule (deval 4): `value` at `position` of an index operand selects a row of
/// `0..len`, or [`EvalError::Index`] names why not.
///
/// I32: `0 <= v < len`. F32 (the dense walk's index convention, still admitted by
/// [`poot_tensor::DType::is_index_operand`] until 558b): finite, integral, non-negative, in range.
pub(crate) fn index_at(
    value: IndexValue,
    position: usize,
    len: usize,
    eqn: ValueId,
) -> Result<usize, EvalError> {
    let fault = |kind: IndexFaultKind| {
        EvalError::Index(Box::new(IndexFault {
            eqn,
            position,
            value,
            len,
            kind,
        }))
    };
    let selected = match value {
        IndexValue::I32(v) => {
            if v < 0 {
                return Err(fault(IndexFaultKind::Negative));
            }
            v as usize
        }
        IndexValue::F32(v) => {
            if !v.is_finite() {
                return Err(fault(IndexFaultKind::NotFinite));
            }
            if v.fract() != 0.0 {
                return Err(fault(IndexFaultKind::NotIntegral));
            }
            if v < 0.0 {
                return Err(fault(IndexFaultKind::Negative));
            }
            v as usize
        }
    };
    if selected >= len {
        return Err(fault(IndexFaultKind::OutOfRange));
    }
    Ok(selected)
}
