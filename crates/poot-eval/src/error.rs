//! The one evaluator's typed errors.
//!
//! One walk, one error type: every refusal names the equation, the admission gap, or the fail-closed
//! bind check that produced it. Card 626 deleted the one untyped `String` variant,
//! `Executor`: its 14 executor-built sites died with the pre-contract device executors (Cards 546b/549).

use poot_graph_ir::{ExecutionValidationFailure, GraphValidationError, TensorType, ValueId};
use poot_tensor::DType;

use crate::cast_authority::CastAuthorityError;
use crate::ops::index_rule::IndexFault;
use crate::{PackedEvalError, exact_dense};

/// A cast's source or result value, literal, for a [`CastFault`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CastOperand {
    I32(i32),
    F32(f32),
}

impl std::fmt::Display for CastOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastOperand::I32(v) => write!(f, "{v}"),
            CastOperand::F32(v) => write!(f, "{v}"),
        }
    }
}

/// `Cast` admits the pair (`infer` never refuses one), but the value at `index` cannot round-trip:
/// X3's I32<->F32 exact-integer range, F32->I32 non-finite/fractional, or I32->I8 out of range.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("cast v{eqn} ({from} -> {to}) word {index} value {value} does not round-trip")]
pub struct CastFault {
    pub eqn: ValueId,
    pub from: DType,
    pub to: DType,
    pub index: usize,
    pub value: CastOperand,
}

/// One evaluator, one error type. Every walk refusal is one of these.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("invalid graph: {0}")]
    InvalidGraph(#[from] GraphValidationError),
    #[error(transparent)]
    Validation(#[from] ExecutionValidationFailure),
    #[error("validation output v{value} is not a complete dense f32 tensor")]
    ValidationValueNotHost { value: ValueId },
    #[error("validation output v{value} has shape {actual:?}; expected {expected:?}")]
    ValidationShape {
        value: ValueId,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("validation packet has {actual} lanes; the graph layout requires {expected}")]
    ValidationPacketLength { expected: usize, actual: usize },
    #[error("no value bound for input v{0}")]
    MissingInput(ValueId),
    #[error("operand v{0} used before it was computed")]
    UseBeforeDef(ValueId),
    #[error(transparent)]
    Packed(#[from] PackedEvalError),
    #[error(transparent)]
    ExactDense(#[from] exact_dense::ExactDenseError),
    #[error(transparent)]
    ExactBf16(#[from] crate::exact_bf16::ExactBf16Error),
    #[error(transparent)]
    Fp8(#[from] crate::fp8::Fp8Error),
    /// The one index-validity refusal (deval 4): `Gather`, `Scatter`, `ScatterUpdate`'s
    /// non-keep-base rows, `DynamicUpdateSlice`'s runtime start, `ArgTopK`'s rank and
    /// `IndexedMatMul`'s expert id all raise this through [`crate::ops::index_rule::index_at`] instead
    /// of each restating their own rounding and bounds check.
    #[error(transparent)]
    Index(Box<IndexFault>),
    /// `shape`'s element count overflows `usize` (554c/554d: the one lane-neutral
    /// check every `element_count`/`row_major_strides`/`checked_numel` call site shares, instead of
    /// a per-lane copy that mislabels the overflow as its own lane's error).
    #[error("shape {shape:?} overflows the host element count")]
    ElementCountOverflow { shape: Vec<usize> },
    /// `Value::into_host` asked for a host tensor, but the carrier is another kind.
    #[error("the {what} is not a host tensor")]
    NotHost { what: &'static str },
    #[error(transparent)]
    Carrier(#[from] poot_tensor::CarrierError),
    /// `resolve` found no admitted function for this equation's `(op, in dtypes, out dtype)` (the
    /// declared `REFUSED` cast cell, or another combination `infer` admits but this
    /// evaluator does not), or a deeper op-internal refusal with no total definition for its operands
    /// (`detail` then names the specific gap; `eqn` is `0` when no equation id is in scope at that
    /// depth).
    #[error("eqn v{eqn} ({op}): {detail}")]
    Unsupported {
        eqn: ValueId,
        op: &'static str,
        detail: String,
    },
    /// A bind-time fail-closed check: the bound carrier does not match what `expected` requires
    /// (an I32 input without authoritative words, an E4M3FN input bound as dense, an owner view
    /// consumed by a non-admitted equation, ...).
    #[error("input v{value} expected {expected}, got {got}")]
    Input {
        value: ValueId,
        expected: TensorType,
        got: &'static str,
    },
    #[error(transparent)]
    Cast(Box<CastFault>),
    /// A `PackedContraction`/packed table read, or a materialization, exceeded `EvalOptions`'s budget.
    #[error("eqn v{eqn} needs {needed} {resource}, budget allows {limit}")]
    Budget {
        eqn: ValueId,
        resource: &'static str,
        needed: usize,
        limit: usize,
    },
    #[error(transparent)]
    CastAuthority(#[from] Box<CastAuthorityError>),
}

impl EvalError {
    /// A runtime refusal with no equation id in scope (a low-level op helper, deep under `walk.rs`):
    /// `eqn` reads as `0`, the graph's first value id, rather than the real equation. Every refusal
    /// `walk.rs` itself raises carries the real id instead (`EvalError::Unsupported { eqn, .. }`
    /// built directly).
    pub(crate) fn unsupported(op: &'static str, detail: impl Into<String>) -> Self {
        EvalError::Unsupported {
            eqn: 0,
            op,
            detail: detail.into(),
        }
    }
}

impl From<CastFault> for EvalError {
    fn from(value: CastFault) -> Self {
        EvalError::Cast(Box::new(value))
    }
}

impl From<IndexFault> for EvalError {
    fn from(value: IndexFault) -> Self {
        EvalError::Index(Box::new(value))
    }
}

impl From<CastAuthorityError> for EvalError {
    fn from(value: CastAuthorityError) -> Self {
        EvalError::CastAuthority(Box::new(value))
    }
}
