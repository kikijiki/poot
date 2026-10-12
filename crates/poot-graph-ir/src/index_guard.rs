//! The canonical in-graph index-bounds guard (card 372c).
//!
//! A selector produced on a device has no host value to check before the consuming kernel runs. Rather than
//! read it back or evaluate a host prefix, the graph proves the bounds itself: ordinary primitives compute a
//! per-lane out-of-range flag, downstream work reads an inert in-range fallback, and the flag is reduced into a
//! validation output so an executor publishes nothing when a lane is out of range (spec 375).
//!
//! This module emits that shape ([`Builder::guard_index_bounds`]). Nothing recognizes it afterward: the guarded
//! selector's `[0, upper_exclusive)` range follows from the `GeU` and `Select` rules of [`crate::value_ranges`],
//! the same as for any other graph code (POOT-726).
//!
//! The emitted shape, for a raw I32 selector `raw` with a static `upper_exclusive`:
//!
//! ```text
//! invalid     = Binary(GeU)(raw, Lit::I32(upper_exclusive))   [n] I32, every lane 0 or 1
//! zero        = Binary(Sub)(raw, raw)                         [n] I32, every lane 0
//! guarded     = Select(invalid, zero, raw)                    [n] I32, in range in every lane
//! invalid_f32 = Cast { to: F32 }(invalid)                     [n] F32
//! witness     = Reduce { Sum, last axis, keepdim }(invalid_f32)  [.., 1] F32
//! ```
//!
//! Three details are forced:
//!
//! - One unsigned comparison decides both bounds. `GeU` compares the I32 lanes as u32, so a negative id
//!   becomes a very large unsigned value and fails the same test as an id at or above `upper_exclusive`.
//!   There is no lower-bound comparison to keep in step with the upper one.
//! - The condition is the *invalid* flag, not `valid`. The planner rejects a literal-first `Binary`, so
//!   neither `GeU(Lit, raw)` nor `Sub(Lit(1), invalid)` can be planned. `Select` picks `if_true`
//!   where the condition word is nonzero, so `Select(invalid, zero, raw)` is the same function of
//!   `raw` that `Select(valid, raw, 0)` would be.
//! - `zero` is a value, not a literal operand. The planner's `Select` builds a three-leaf kernel from the
//!   equation's *value* operands, so a literal `0` would not produce three leaves. `Sub(raw, raw)` keeps
//!   the guard self-contained: no caller-bound constant, exact on every lane, and already wired on device.
//!
//! The guard condition and the witness observation are the same value, so nothing has to be tied together
//! after the fact.
//!
//! One emitter ([`emit_index_guard`]) writes the shape for any equation sink: [`Builder::guard_index_bounds`]
//! appends it to a traced graph, and a graph pass appends it to the graph it rewrites. The one recognizer
//! of a guard's witness ([`is_guard_witness`]) lives beside it and reads exactly this shape, so a consumer
//! that must know a clamped lane is observed (the attention matcher, the expert-id check) cannot drift from
//! the emitter.

use std::collections::HashMap;

use crate::builder::{Builder, Traced};
use crate::error::BuilderError;
use crate::graph::{Eqn, Graph, Operand, ValueId};
use crate::op::{BinOp, OpKind, RedOp};
use crate::types::{DType, Scalar, TensorType};

/// Why a guard cannot be emitted for the requested selector and bound.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexGuardError {
    /// The selector is not I32, the only dtype the guard's compare and clamp operate on.
    #[error("selector {raw} has dtype {dtype}; an index-bounds guard needs I32")]
    RawNotI32 {
        /// The selector value that was refused.
        raw: ValueId,
        /// The dtype it has.
        dtype: DType,
    },
    /// The selector is a scalar, so it has no axis to reduce the out-of-range count over.
    #[error("selector {raw} is rank 0; an index-bounds guard needs an axis to reduce")]
    RawNotRanked {
        /// The rank-0 selector value that was refused.
        raw: ValueId,
    },
    /// The exclusive upper bound exceeds `i32::MAX`, so it cannot be written as the I32 literal the guard compares against.
    #[error("bound {upper_exclusive} does not fit in i32")]
    BoundNotRepresentable {
        /// The requested exclusive upper bound.
        upper_exclusive: usize,
    },
    /// Appending one of the guard's equations failed.
    #[error(transparent)]
    Build(#[from] BuilderError),
}

/// What [`Builder::guard_index_bounds`] appended: the selector downstream work reads, and the value to
/// declare as a validation output.
#[derive(Clone, Copy, Debug)]
pub struct GuardedIndex {
    /// Read this instead of the raw selector. Every lane is in `[0, upper_exclusive)`.
    pub guarded: Traced,
    /// Declare this as a validation output. Zero when every lane was in range.
    pub witness: Traced,
}

/// What [`emit_index_guard`] emitted, by value id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardValues {
    /// Read this instead of the raw selector. Every lane is in `[0, upper_exclusive)`.
    pub guarded: ValueId,
    /// Declare this as a validation output. Zero when every lane was in range. The last equation emitted.
    pub witness: ValueId,
}

/// Why [`emit_index_guard`] failed: the guard's own preconditions, or the sink refusing an equation.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexGuardEmitError<E: std::error::Error + 'static> {
    /// The selector or bound cannot be guarded.
    #[error(transparent)]
    Guard(IndexGuardError),
    /// The caller's `emit` callback refused one of the guard's equations.
    #[error("emitting an equation of the index guard: {0}")]
    Emit(#[source] E),
}

/// Emit the canonical guard (see the [module docs](self)) over `raw`, of type `raw_type`, through `emit`:
/// each call appends one equation with the given operands and returns its output value. The five equations
/// are emitted in order and the last is the witness.
///
/// This is the one definition of the shape; [`Builder::guard_index_bounds`] and graph passes both call it.
pub fn emit_index_guard<E: std::error::Error + 'static>(
    raw: ValueId,
    raw_type: &TensorType,
    upper_exclusive: usize,
    mut emit: impl FnMut(OpKind, Vec<Operand>) -> Result<ValueId, E>,
) -> Result<GuardValues, IndexGuardEmitError<E>> {
    if raw_type.dtype != DType::I32 {
        return Err(IndexGuardEmitError::Guard(IndexGuardError::RawNotI32 {
            raw,
            dtype: raw_type.dtype,
        }));
    }
    let Some(axis) = raw_type.shape.len().checked_sub(1) else {
        return Err(IndexGuardEmitError::Guard(IndexGuardError::RawNotRanked {
            raw,
        }));
    };
    let bound = i32::try_from(upper_exclusive).map_err(|_| {
        IndexGuardEmitError::Guard(IndexGuardError::BoundNotRepresentable { upper_exclusive })
    })?;
    let value = Operand::Value;
    let mut step = |op, inputs| emit(op, inputs).map_err(IndexGuardEmitError::Emit);
    let invalid = step(
        OpKind::Binary(BinOp::GeU),
        vec![value(raw), Operand::Lit(Scalar::I32(bound))],
    )?;
    let zero = step(OpKind::Binary(BinOp::Sub), vec![value(raw), value(raw)])?;
    let guarded = step(
        OpKind::Select,
        vec![value(invalid), value(zero), value(raw)],
    )?;
    let invalid_f32 = step(OpKind::Cast { to: DType::F32 }, vec![value(invalid)])?;
    let witness = step(
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            keepdim: true,
        },
        vec![value(invalid_f32)],
    )?;
    Ok(GuardValues { guarded, witness })
}

/// Whether `witness` is the witness [`emit_index_guard`] emits for the guard whose invalid flag is
/// `condition`: a keepdim last-axis `Sum` of the F32 cast of `condition`. Anything else (a witness over
/// another value, a reduction over another axis) does not observe every clamped lane. `producer` maps each
/// value to its equation's index in `g`.
pub fn is_guard_witness(
    g: &Graph,
    producer: &HashMap<ValueId, usize>,
    witness: ValueId,
    condition: ValueId,
) -> bool {
    let of =
        |value: ValueId| -> Option<&Eqn> { producer.get(&value).map(|&index| &g.eqns()[index]) };
    let Some(sum) = of(witness) else {
        return false;
    };
    let (
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            keepdim: true,
        },
        [Operand::Value(flag)],
    ) = (&sum.op, sum.inputs.as_slice())
    else {
        return false;
    };
    if *axis + 1 != g.aval(*flag).shape.len() {
        return false;
    }
    matches!(
        of(*flag),
        Some(Eqn { op: OpKind::Cast { to: DType::F32 }, inputs, .. })
            if matches!(inputs.as_slice(), [Operand::Value(observed)] if *observed == condition)
    )
}

impl Builder {
    /// Emit the canonical index-bounds guard for `raw` and return the guarded selector plus the witness
    /// to declare as a validation output. See the [module docs](self) for the shape and why it is forced.
    pub fn guard_index_bounds(
        &self,
        raw: Traced,
        upper_exclusive: usize,
    ) -> Result<GuardedIndex, IndexGuardError> {
        let aval = self.aval(raw)?;
        let values = emit_index_guard(raw.id(), &aval, upper_exclusive, |op, inputs| {
            self.emit(op, inputs).map(|traced| traced.id())
        })
        .map_err(|error| match error {
            IndexGuardEmitError::Guard(error) => error,
            IndexGuardEmitError::Emit(error) => IndexGuardError::Build(error),
        })?;
        Ok(GuardedIndex {
            guarded: self.handle(values.guarded),
            witness: self.handle(values.witness),
        })
    }
}

#[cfg(test)]
mod tests;
