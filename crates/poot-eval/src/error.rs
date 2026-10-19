//! The one evaluator's typed errors.
//!
//! One walk, one error type: every refusal names the equation, the admission gap, or the fail-closed
//! bind check that produced it. Card 626 deleted the one untyped `String` variant,
//! `Executor`: its 14 executor-built sites died with the pre-contract device executors (Cards 546b/549).

use poot_graph_ir::{ExecutionValidationFailure, IndexRange, TensorType, ValueId};
use poot_tensor::DType;

use crate::PackedEvalError;
use crate::ops::index_rule::IndexFault;

/// A cast's source or result value, literal, for a [`CastFault`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CastOperand {
    /// An `I32` word, printed as the integer.
    I32(i32),
    /// An `F32` value, printed with Rust's shortest round-trip float format.
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
#[error("cast {eqn} ({from} -> {to}) word {index} value {value} does not round-trip")]
pub struct CastFault {
    /// The `Cast` equation that refused the value.
    pub eqn: ValueId,
    /// Source dtype of the cast.
    pub from: DType,
    /// Result dtype of the cast.
    pub to: DType,
    /// Row-major element index of the offending word in the cast's operand.
    pub index: usize,
    /// The offending source word, as the literal it held.
    pub value: CastOperand,
}

/// A slot input holds the word `word` at `index`, outside the `range` its graph declares for the slot
/// (POOT-726: `Token` in `[0, vocab)`, `Pos` in `[0, capacity)`). The declaration is a fact every
/// consumer of the slot may rely on, so a binder refuses the step instead of running it.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("input {value} word {index} is {word}, outside its declared range {range}")]
pub struct SlotRangeFault {
    /// The slot input value whose bound words broke the declaration.
    pub value: ValueId,
    /// Row-major element index of the offending word within the bound input.
    pub index: usize,
    /// The offending word.
    pub word: i32,
    /// The range the graph declares for the slot.
    pub range: IndexRange,
}

/// One evaluator, one error type. Every walk refusal is one of these.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// The execution-validation packet failed; the failure is carried through unchanged.
    #[error(transparent)]
    Validation(#[from] ExecutionValidationFailure),
    /// A validation output is bound to something other than a complete dense f32 host tensor.
    #[error("validation output {value} is not a complete dense f32 tensor")]
    ValidationValueNotHost {
        /// The validation output value.
        value: ValueId,
    },
    /// A validation output has a shape other than the one its declaration requires.
    #[error("validation output {value} has shape {actual:?}; expected {expected:?}")]
    ValidationShape {
        /// The validation output value.
        value: ValueId,
        /// Shape the declaration requires.
        expected: Vec<usize>,
        /// Shape the walk produced.
        actual: Vec<usize>,
    },
    /// The validation packet's lane count disagrees with the graph's validation layout.
    #[error("validation packet has {actual} lanes; the graph layout requires {expected}")]
    ValidationPacketLength {
        /// Lane count the graph's validation layout requires.
        expected: usize,
        /// Lane count the packet holds.
        actual: usize,
    },
    /// A graph input (const or slot) was not bound by the caller.
    #[error("no value bound for input {0}")]
    MissingInput(ValueId),
    /// An equation read an operand the walk had not computed yet (an ill-ordered graph).
    #[error("operand {0} used before it was computed")]
    UseBeforeDef(ValueId),
    /// A packed-carrier refusal, carried through unchanged.
    #[error(transparent)]
    Packed(#[from] PackedEvalError),
    /// An exact BF16 carrier refusal, carried through unchanged.
    #[error(transparent)]
    ExactBf16(#[from] crate::exact_bf16::ExactBf16Error),
    /// An E4M3FN carrier refusal, carried through unchanged.
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
    ElementCountOverflow {
        /// The shape whose element count overflowed.
        shape: Vec<usize>,
    },
    /// `Value::into_host` asked for a host tensor, but the carrier is another kind.
    #[error("the {what} is not a host tensor")]
    NotHost {
        /// Which carrier the value was (`"exact value"`, `"packed component"`, ...).
        what: &'static str,
    },
    /// A host-tensor carrier refusal (dtype/payload mismatch), carried through unchanged.
    #[error(transparent)]
    Carrier(#[from] poot_tensor::CarrierError),
    /// `resolve` found no admitted function for this equation's `(op, in dtypes, out dtype)` (the
    /// declared `REFUSED` cast cell, or another combination `infer` admits but this
    /// evaluator does not), or a deeper op-internal refusal with no total definition for its operands
    /// (`detail` then names the specific gap; `eqn` is `0` when no equation id is in scope at that
    /// depth).
    #[error("eqn {eqn} ({op}): {detail}")]
    Unsupported {
        /// The refused equation (`0` when no equation id was in scope).
        eqn: ValueId,
        /// The op's name, as `OpKind::name` reports it.
        op: &'static str,
        /// What exactly is not admitted or not defined.
        detail: String,
    },
    /// The decomposition a composite is evaluated through could not be built: the tracer refused it
    /// (it does not fit the decomposition limits, or its definition is defective).
    #[error("eqn {eqn} ({op}): its decomposition could not be built: {source}")]
    Decomposition {
        /// The composite equation whose decomposition failed.
        eqn: ValueId,
        /// The composite's op name.
        op: &'static str,
        /// The tracer's refusal.
        source: Box<poot_graph_ir::BuilderError>,
    },
    /// A bind-time fail-closed check: the bound carrier does not match what `expected` requires
    /// (an I32 input without authoritative words, an E4M3FN input bound as dense, an owner view
    /// consumed by a non-admitted equation, ...).
    #[error("input {value} expected {expected}, got {got}")]
    Input {
        /// The input value the bind check refused.
        value: ValueId,
        /// The type the graph declares for that input.
        expected: TensorType,
        /// What the caller bound instead.
        got: &'static str,
    },
    /// A cast whose value cannot round-trip; boxed to keep the error small.
    #[error(transparent)]
    Cast(Box<CastFault>),
    /// A slot input holds a word outside the range its graph declares for it (POOT-726): the same
    /// refusal a device executor gives at bind, raised before any equation runs.
    #[error(transparent)]
    SlotRange(Box<SlotRangeFault>),
}

impl EvalError {
    /// A runtime refusal with no equation id in scope (a low-level op helper, deep under `walk.rs`):
    /// `eqn` reads as `0`, the graph's first value id, rather than the real equation. Every refusal
    /// `walk.rs` itself raises carries the real id instead (`EvalError::Unsupported { eqn, .. }`
    /// built directly).
    pub(crate) fn unsupported(op: &'static str, detail: impl Into<String>) -> Self {
        EvalError::Unsupported {
            eqn: ValueId::new(0),
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

impl From<SlotRangeFault> for EvalError {
    fn from(value: SlotRangeFault) -> Self {
        EvalError::SlotRange(Box::new(value))
    }
}
