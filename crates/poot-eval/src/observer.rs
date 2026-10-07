//! `EvalOptions::observer`: a caller hook into the walk's per-equation publication, for allocation
//! accounting and NaN diagnosis. Budgets and observers are options on the one walk, not separate
//! walks.

use poot_graph_ir::ValueId;
use poot_tensor::DType;

use crate::Value;

/// A caller hook into one evaluation. Both methods default to doing nothing, so a caller that only
/// wants `allocation` (an accounting test) need not implement `value`, and vice versa.
pub trait EvalObserver {
    /// An equation's output was reserved: `eqn` is its position in `Graph::eqns`, `out` its value id.
    ///
    /// `EvalBudget`/this hook are a materialization ceiling for the packed, cast and Spec-376 BF16
    /// arms (`walk.rs`'s `opts.charge` call sites: `PackedDequant`,
    /// `PackedRowGather`, `Cast`'s `I32 -> F32` and BF16 widens, and the Spec-376 `Widen`/`Gather`/
    /// `WeightMatMul` arms) - the paths where one equation can
    /// lazily materialize an allocation whose size depends on a checkpoint table far larger than the
    /// graph's own declared shapes would suggest. Every intermediate of a
    /// composite's decomposition charges too (Card 556), attributed to the composite equation: a
    /// decomposed prefill attention materializes `[1, Hq, L, L]` scores behind a `[1, Hq, L, D]`
    /// result. It is not a universal per-equation accounting: the dense arithmetic path (`Binary`,
    /// `Reduce`, `MatMul`, `Gather`, ...) allocates a fresh buffer for nearly every equation too, but
    /// never calls `charge`/this hook outside a decomposition, so `max_work_elements` bounds nothing
    /// there and this fires only for the arms listed above.
    fn allocation(&mut self, eqn: usize, out: ValueId, dtype: DType, bytes: usize) {
        let _ = (eqn, out, dtype, bytes);
    }

    /// An equation's value, right after it was computed, before the walk's publication gate runs
    /// (card 537's `POOT_DEBUG_NAN` successor): a tap sees every intermediate even when the gate later
    /// refuses to publish anything.
    fn value(&mut self, eqn: usize, out: ValueId, value: &Value) {
        let _ = (eqn, out, value);
    }
}

/// A no-op observer: the default when a caller passes none.
impl EvalObserver for () {}
