//! `EvalOptions::observer`: a caller hook into the walk's per-equation publication, for allocation
//! accounting and NaN diagnosis. An observer is an option on the one walk, not a separate walk.

use poot_graph_ir::ValueId;
use poot_tensor::DType;

use crate::Value;

/// What the walk hands the caller once validation has passed: the one set of copies that leave the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Publication {
    /// The graph's primary result.
    Output(ValueId),
    /// A carried state output.
    State(ValueId),
    /// The whole environment, under `EvalOptions::keep_environment`.
    Environment,
}

/// A caller hook into one evaluation. Every method defaults to doing nothing, so a caller that only
/// wants `allocation` (an accounting test) need not implement the others.
pub trait EvalObserver {
    /// An equation's output was reserved: `eqn` is its position in `Graph::eqns`, `out` its value id.
    ///
    /// This hook reports the materialization of the packed, cast and Spec-376 BF16 arms
    /// (`walk.rs`'s `opts.allocated` call sites: `PackedDequant`, `PackedRowGather`, an indexed
    /// `PackedContraction` (one `[out, K]` report per expert decoded), `Cast`'s `I32 -> F32` and BF16 widens,
    /// and the Spec-376 `Widen`/`Gather`/`WeightMatMul` arms) - the paths where one equation can lazily
    /// materialize an allocation whose size depends on a checkpoint table far larger than the graph's own
    /// declared shapes would suggest. Every intermediate of a composite's decomposition reports too (Card
    /// 556), attributed to the composite equation: a decomposed prefill attention materializes
    /// `[1, Hq, L, L]` scores behind a `[1, Hq, L, D]` result. It is not a universal per-equation
    /// accounting: the dense arithmetic path (`Binary`, `Reduce`, `MatMul`, `Gather`, ...) allocates a fresh
    /// buffer for nearly every equation too, but never calls this hook outside a decomposition.
    fn allocation(&mut self, eqn: usize, out: ValueId, dtype: DType, bytes: usize) {
        let _ = (eqn, out, dtype, bytes);
    }

    /// An equation's value, right after it was computed, before the walk's publication gate runs
    /// (card 537's `POOT_DEBUG_NAN` successor): a tap sees every intermediate even when the gate later
    /// refuses to publish anything.
    fn value(&mut self, eqn: usize, out: ValueId, value: &Value) {
        let _ = (eqn, out, value);
    }

    /// A value was copied out into the caller-visible result. Fires only after the validation gate
    /// passed (ADR-0090), so a failed validation reports no publication.
    fn published(&mut self, publication: Publication) {
        let _ = publication;
    }
}

/// A no-op observer: the default when a caller passes none.
impl EvalObserver for () {}
