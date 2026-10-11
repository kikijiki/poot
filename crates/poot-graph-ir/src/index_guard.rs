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

use crate::builder::{Builder, Traced};
use crate::graph::ValueId;
use crate::op::{BinOp, RedOp};
use crate::types::{DType, Scalar};

/// Why a guard cannot be emitted for the requested selector and bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IndexGuardError {
    #[error("selector {raw} has dtype {dtype}; an index-bounds guard needs I32")]
    RawNotI32 { raw: ValueId, dtype: DType },
    #[error("selector {raw} is rank 0; an index-bounds guard needs an axis to reduce")]
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
                raw: raw.id(),
                dtype: aval.dtype,
            });
        }
        let Some(axis) = aval.shape.len().checked_sub(1) else {
            return Err(IndexGuardError::RawNotRanked { raw: raw.id() });
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

#[cfg(test)]
mod tests;
