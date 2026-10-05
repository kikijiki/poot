//! The canonical in-graph index-bounds guard (card 372c).
//!
//! A selector produced on a device has no host value to check before the consuming kernel runs. Rather than
//! read it back or evaluate a host prefix, the graph proves the bounds itself: ordinary primitives compute a
//! per-lane out-of-range flag, downstream work reads an inert in-range fallback, and the flag is reduced into a
//! validation output so an executor publishes nothing when a lane is out of range (spec 375).
//!
//! This module owns both halves: [`Builder::guard_index_bounds`] emits the shape and
//! [`recognize_index_bounds_guard`] accepts exactly it. They live in `poot-graph-ir` because the pattern is
//! graph structure with no model or backend meaning (`upper_exclusive` is a bound, not an expert count) and
//! because it is the only crate both `poot-graph-plan` (authorization) and `poot-eval` (the CPU oracle) depend
//! on. The recognizer re-derives the shape from the graph and never trusts the helper.
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
//!   neither `GeU(Lit, raw)` nor `Sub(Lit(1), invalid)` can be planned. `Select` computes
//!   `if_false + cond * (if_true - if_false)`, so `Select(invalid, zero, raw)` is the same function of
//!   `raw` that `Select(valid, raw, 0)` would be.
//! - `zero` is a value, not a literal operand. The planner's `Select` builds a three-leaf kernel from the
//!   equation's *value* operands, so a literal `0` would not produce three leaves. `Sub(raw, raw)` keeps
//!   the guard self-contained: no caller-bound constant, exact on every lane, and already wired on device.
//!
//! The guard condition and the witness observation are the same value, so nothing has to be tied together
//! after the fact: a guard whose witness observes some other value is not this shape.

use crate::builder::{Builder, Traced};
use crate::graph::{Graph, Operand, ValidationChannel, ValueId};
use crate::op::{BinOp, OpKind, RedOp};
use crate::types::{DType, Scalar};

/// The values of one canonical index-bounds guard, as recognized from a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBoundsGuard {
    /// The unguarded selector the guard bounds.
    pub raw: ValueId,
    /// The per-lane out-of-range flag: the guard condition and the witness observation.
    pub invalid: ValueId,
    /// The all-zero fallback the guard selects for an out-of-range lane.
    pub zero: ValueId,
    /// The guarded selector downstream work reads.
    pub guarded: ValueId,
    /// The F32 flag the witness reduces.
    pub invalid_f32: ValueId,
    /// The validation value: the count of out-of-range lanes.
    pub witness: ValueId,
    /// The exclusive upper bound the guard enforces.
    pub upper_exclusive: i32,
}

/// Why a pair of values is not a canonical index-bounds guard and its witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexGuardRejection {
    #[error("guarded value v{guarded} is not produced by a Select")]
    GuardedNotSelect { guarded: ValueId },
    #[error("guard v{guarded} has a literal operand; every Select operand must be a value")]
    GuardedLiteralOperand { guarded: ValueId },
    #[error("guard condition v{invalid} is not an unsigned upper-bound comparison")]
    ConditionNotUnsignedBound { invalid: ValueId },
    #[error("guard condition v{invalid} does not compare v{expected}")]
    ConditionComparesAnotherValue {
        invalid: ValueId,
        expected: ValueId,
        actual: ValueId,
    },
    #[error("guard condition v{invalid} does not bound by an I32 literal")]
    BoundNotI32Literal { invalid: ValueId },
    #[error("guard fallback v{zero} is not Sub(v{expected}, v{expected})")]
    FallbackNotZeroOfSource { zero: ValueId, expected: ValueId },
    #[error("witness v{witness} is not a keepdim last-axis Sum of a Cast to F32")]
    WitnessNotSumOfCast { witness: ValueId },
    #[error("witness v{witness} observes v{observed}, not the guard condition v{invalid}")]
    WitnessObservesAnotherValue {
        witness: ValueId,
        observed: ValueId,
        invalid: ValueId,
    },
}

/// Why a guard cannot be emitted for the requested selector and bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexGuardError {
    #[error("selector v{raw} has dtype {dtype}; an index-bounds guard needs I32")]
    RawNotI32 { raw: ValueId, dtype: DType },
    #[error("selector v{raw} is rank 0; an index-bounds guard needs an axis to reduce")]
    RawNotRanked { raw: ValueId },
    #[error("bound {upper_exclusive} does not fit in i32")]
    BoundNotRepresentable { upper_exclusive: usize },
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

impl Builder {
    /// Emit the canonical index-bounds guard for `raw` and return the guarded selector plus the witness
    /// to declare as a validation output. See the [module docs](self) for the shape and why it is forced.
    pub fn guard_index_bounds(
        &self,
        raw: Traced,
        upper_exclusive: usize,
    ) -> Result<GuardedIndex, IndexGuardError> {
        let aval = self.aval(raw);
        if aval.dtype != DType::I32 {
            return Err(IndexGuardError::RawNotI32 {
                raw: raw.id,
                dtype: aval.dtype,
            });
        }
        let Some(axis) = aval.shape.len().checked_sub(1) else {
            return Err(IndexGuardError::RawNotRanked { raw: raw.id });
        };
        let bound = i32::try_from(upper_exclusive)
            .map_err(|_| IndexGuardError::BoundNotRepresentable { upper_exclusive })?;
        let invalid = self.binary_scalar(BinOp::GeU, raw, Scalar::I32(bound));
        let zero = self.binary(BinOp::Sub, raw, raw);
        let guarded = self.select(invalid, zero, raw);
        let invalid_f32 = self.cast(invalid, DType::F32);
        let witness = self.reduce(RedOp::Sum, invalid_f32, axis, true);
        Ok(GuardedIndex { guarded, witness })
    }
}

/// Recognize the canonical guard that produced `guarded`, requiring `witness` to observe its condition.
///
/// This is the single reader of the shape [`Builder::guard_index_bounds`] writes. It re-derives every
/// edge from the graph, so a helper that emitted a different comparison, a different operand order, or a
/// witness over some other value is rejected here rather than trusted.
pub fn recognize_index_bounds_guard<V: ValidationChannel>(
    graph: &Graph<V>,
    guarded: ValueId,
    witness: ValueId,
) -> Result<IndexBoundsGuard, IndexGuardRejection> {
    let producers = producer_table(graph);
    let select = producer(&producers, graph, guarded)
        .filter(|eqn| matches!(eqn.op, OpKind::Select))
        .ok_or(IndexGuardRejection::GuardedNotSelect { guarded })?;
    let [condition, if_true, if_false] = select.inputs.as_slice() else {
        return Err(IndexGuardRejection::GuardedNotSelect { guarded });
    };
    let (&Operand::Value(invalid), &Operand::Value(zero), &Operand::Value(raw)) =
        (condition, if_true, if_false)
    else {
        return Err(IndexGuardRejection::GuardedLiteralOperand { guarded });
    };

    let bound = producer(&producers, graph, invalid)
        .filter(|eqn| matches!(eqn.op, OpKind::Binary(BinOp::GeU)))
        .ok_or(IndexGuardRejection::ConditionNotUnsignedBound { invalid })?;
    let [compared, limit] = bound.inputs.as_slice() else {
        return Err(IndexGuardRejection::ConditionNotUnsignedBound { invalid });
    };
    let &Operand::Value(compared) = compared else {
        return Err(IndexGuardRejection::ConditionNotUnsignedBound { invalid });
    };
    if compared != raw {
        return Err(IndexGuardRejection::ConditionComparesAnotherValue {
            invalid,
            expected: raw,
            actual: compared,
        });
    }
    let &Operand::Lit(Scalar::I32(upper_exclusive)) = limit else {
        return Err(IndexGuardRejection::BoundNotI32Literal { invalid });
    };

    let fallback = producer(&producers, graph, zero)
        .filter(|eqn| matches!(eqn.op, OpKind::Binary(BinOp::Sub)))
        .filter(|eqn| {
            matches!(
                eqn.inputs.as_slice(),
                [Operand::Value(left), Operand::Value(right)] if *left == raw && *right == raw
            )
        })
        .is_some();
    if !fallback {
        return Err(IndexGuardRejection::FallbackNotZeroOfSource {
            zero,
            expected: raw,
        });
    }

    let invalid_f32 = witness_source(&producers, graph, witness)?;
    let cast = producer(&producers, graph, invalid_f32)
        .filter(|eqn| matches!(eqn.op, OpKind::Cast { to: DType::F32 }))
        .ok_or(IndexGuardRejection::WitnessNotSumOfCast { witness })?;
    let [Operand::Value(observed)] = cast.inputs.as_slice() else {
        return Err(IndexGuardRejection::WitnessNotSumOfCast { witness });
    };
    if *observed != invalid {
        return Err(IndexGuardRejection::WitnessObservesAnotherValue {
            witness,
            observed: *observed,
            invalid,
        });
    }

    Ok(IndexBoundsGuard {
        raw,
        invalid,
        zero,
        guarded,
        invalid_f32,
        witness,
        upper_exclusive,
    })
}

/// The value a canonical witness reduces: a keepdim `Sum` over the last axis of its operand.
fn witness_source<V: ValidationChannel>(
    producers: &[Option<usize>],
    graph: &Graph<V>,
    witness: ValueId,
) -> Result<ValueId, IndexGuardRejection> {
    let reduce = producer(producers, graph, witness)
        .ok_or(IndexGuardRejection::WitnessNotSumOfCast { witness })?;
    let OpKind::Reduce {
        op: RedOp::Sum,
        axis,
        keepdim: true,
    } = &reduce.op
    else {
        return Err(IndexGuardRejection::WitnessNotSumOfCast { witness });
    };
    let [Operand::Value(source)] = reduce.inputs.as_slice() else {
        return Err(IndexGuardRejection::WitnessNotSumOfCast { witness });
    };
    // Only a last-axis reduce covers every lane of the flag in one value; any other axis would leave
    // whole lanes out of the packet, so an out-of-range id could pass.
    if *axis + 1 != graph.aval(*source).shape.len() {
        return Err(IndexGuardRejection::WitnessNotSumOfCast { witness });
    }
    Ok(*source)
}

fn producer_table<V: ValidationChannel>(graph: &Graph<V>) -> Vec<Option<usize>> {
    let mut producers = vec![None; graph.values.len()];
    for (index, eqn) in graph.eqns.iter().enumerate() {
        producers[eqn.out] = Some(index);
    }
    producers
}

fn producer<'g, V: ValidationChannel>(
    producers: &[Option<usize>],
    graph: &'g Graph<V>,
    value: ValueId,
) -> Option<&'g crate::graph::Eqn> {
    producers
        .get(value)
        .copied()
        .flatten()
        .map(|index| &graph.eqns[index])
}

#[cfg(test)]
mod tests;
