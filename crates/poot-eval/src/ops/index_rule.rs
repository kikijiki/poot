//! The one index-validity rule every index-consuming op shares (R470-008, R470-011): `Gather`,
//! `ScatterUpdate`'s non-padding slots, `DynamicUpdateSlice`'s runtime start, `ArgTopK`'s
//! rank and `IndexedMatMul`'s expert id all read one row-selecting integer through [`index_at`]
//! instead of each restating its own rounding and bounds check.

use poot_graph_ir::ValueId;

use crate::EvalError;

/// A value the index rule checks: an I32 index word (the one index dtype), or the F32 rank value `ArgTopK`
/// inverts (an integral count computed by float arithmetic, never pre-rounded). No index operand is F32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexValue {
    /// An I32 index word, the one index dtype.
    I32(i32),
    /// The F32 rank value `ArgTopK` inverts.
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
/// [`Self::NotIntegral`] before its sign; a finite integral negative value (I32 or F32 rank) is
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
    /// A row-writing index names a row an earlier position already named: two source rows would write one
    /// slot (see [`claim_row`]).
    Duplicate,
}

/// One index read refused: `value` at `position` of the index operand cannot select a row of
/// `0..len` (deval 4). The oracle's one index-validity refusal; every index-consuming op raises this
/// instead of restating its own rounding or bounds check.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("index {eqn} position {position} value {value} is not a valid row of 0..{len} ({kind:?})")]
pub struct IndexFault {
    /// The equation that read the index.
    pub eqn: ValueId,
    /// Position within the index operand, row-major.
    pub position: usize,
    /// The refused value.
    pub value: IndexValue,
    /// The selectable row count: valid values are `0..len`.
    pub len: usize,
    /// Which check the value failed.
    pub kind: IndexFaultKind,
}

/// The one row-selection rule (deval 4): `value` at `position` of an index operand selects a row of
/// `0..len`, or [`EvalError::Index`] names why not.
///
/// I32: `0 <= v < len`. F32 (an `ArgTopK` rank value): finite, integral, non-negative, in range.
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

/// Mark `row` (selected by `value` at `position` of an index operand over `len` rows) as written, or
/// refuse with [`IndexFaultKind::Duplicate`] when an earlier position already wrote it. `claimed` holds one
/// flag per row. The row-writing ops whose result would depend on write order are refused here, so a
/// plausible last-writer-wins value never stands in for a write race.
pub(crate) fn claim_row(
    claimed: &mut [bool],
    row: usize,
    value: IndexValue,
    position: usize,
    eqn: ValueId,
) -> Result<(), EvalError> {
    if std::mem::replace(&mut claimed[row], true) {
        return Err(EvalError::Index(Box::new(IndexFault {
            eqn,
            position,
            value,
            len: claimed.len(),
            kind: IndexFaultKind::Duplicate,
        })));
    }
    Ok(())
}
