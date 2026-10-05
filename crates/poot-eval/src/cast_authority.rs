//! The exact-I32 cast authority: per-family authorization for `Cast(I32 -> F32)`.
//!
//! `EvalOptions::cast_authority` stays an option until Card 558c replaces it with IR value ranges;
//! this whole file is deleted then. The authority answers one question per `Cast(I32 ->
//! F32)` equation: what role is this selector, so the walk can check the right shape of bound.

use poot_graph_ir::index_guard::recognize_index_bounds_guard;
use poot_graph_ir::{Graph, OpKind, ValidationChannel, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactI32CastBounds {
    pub selected_element_ceiling: usize,
    pub expert_count: usize,
}

/// What an authorized `Cast I32 -> F32` is, and therefore which lane checks the walk owes.
///
/// Every such `Cast` needs an authorization when `opts.cast_authority` is set. The walk checks each
/// kind against the graph itself, so a caller cannot get a weaker check by naming a stronger role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactI32CastRole {
    /// A selector whose lanes are validated here, from a `Gather` result.
    HostCheckedSelector(ExactI32CastBounds),
    /// Card 372c: a selector an in-graph bounds guard clamps, whose declared witness reports the
    /// out-of-range lanes.
    ///
    /// Only the `[0, experts)` lane check is skipped: the role carries no expert count, and keeping it
    /// would make the CPU report `ExpertIdRange` where a device reports `ExecutionValidationFailure`
    /// for the same graph. The exact-f32 range is still checked.
    GuardedSelector { selected_element_ceiling: usize },
    /// Card 372c: a witness lane flag on its way into the validation packet.
    Witness { selected_element_ceiling: usize },
}

impl ExactI32CastRole {
    pub(crate) fn selected_element_ceiling(&self) -> usize {
        match *self {
            ExactI32CastRole::HostCheckedSelector(bounds) => bounds.selected_element_ceiling,
            ExactI32CastRole::GuardedSelector {
                selected_element_ceiling,
            }
            | ExactI32CastRole::Witness {
                selected_element_ceiling,
            } => selected_element_ceiling,
        }
    }
}

/// `opts.cast_authority`'s function object: answer the role of `cast`, or `None` to leave it
/// unauthorized (`MissingCastAuthorization`).
pub trait CastAuthority {
    fn role(&mut self, cast: ValueId) -> Option<ExactI32CastRole>;
}

impl<F: FnMut(ValueId) -> Option<ExactI32CastRole>> CastAuthority for F {
    fn role(&mut self, cast: ValueId) -> Option<ExactI32CastRole> {
        self(cast)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CastAuthorityError {
    #[error("Cast v{cast} has no exact I32 authorization")]
    MissingCastAuthorization { cast: ValueId },
    #[error("Cast v{cast} input v{input} is not a Gather result")]
    CastSourceNotGather { cast: ValueId, input: ValueId },
    #[error("Cast v{cast} input v{input} is not a canonical index-bounds guard over a Gather")]
    CastSourceNotGuardedGather { cast: ValueId, input: ValueId },
    #[error("Cast v{cast} input v{input} is not an ordered comparison")]
    CastSourceNotObservation { cast: ValueId, input: ValueId },
    #[error("Cast v{cast} selected {actual} elements, ceiling {limit}")]
    SelectedElementLimit {
        cast: ValueId,
        actual: usize,
        limit: usize,
    },
    #[error("Cast v{cast} word {index} expert {value} outside [0,{experts})")]
    ExpertIdRange {
        cast: ValueId,
        index: usize,
        value: i32,
        experts: usize,
    },
}

/// Each role names a graph shape, and the walk checks it rather than trusting the caller, so naming
/// `GuardedSelector` or `Witness` for a plain `Gather` is rejected.
///
/// The guard's bound (`IndexBoundsGuard::upper_exclusive`) is not read here and the role carries no
/// expert count, which is why `GuardedSelector` skips `ExpertIdRange`; the exact-f32 range is checked
/// for every role in `ops::cast`. The planner proves the bound equals the claim's expert count
/// (`poot_graph_plan`'s `GuardBoundMismatch`).
pub(crate) fn validate_cast_source<V: ValidationChannel>(
    graph: &Graph<V>,
    producers: &[Option<usize>],
    cast: ValueId,
    source: ValueId,
    role: ExactI32CastRole,
) -> Result<(), CastAuthorityError> {
    let producer = producers[source].and_then(|index| graph.eqns.get(index));
    match role {
        ExactI32CastRole::HostCheckedSelector(_) => producer
            .filter(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .map(|_| ())
            .ok_or(CastAuthorityError::CastSourceNotGather {
                cast,
                input: source,
            }),
        ExactI32CastRole::GuardedSelector { .. } => {
            let guarded = graph
                .validation_outputs()
                .iter()
                .find_map(|validation| {
                    recognize_index_bounds_guard(graph, source, validation.value).ok()
                })
                .filter(|guard| {
                    producers[guard.raw]
                        .and_then(|index| graph.eqns.get(index))
                        .is_some_and(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
                });
            guarded
                .map(|_| ())
                .ok_or(CastAuthorityError::CastSourceNotGuardedGather {
                    cast,
                    input: source,
                })
        }
        ExactI32CastRole::Witness { .. } => producer
            .filter(|eqn| {
                matches!(
                    eqn.op,
                    OpKind::Binary(poot_graph_ir::BinOp::Ge | poot_graph_ir::BinOp::GeU)
                )
            })
            .map(|_| ())
            .ok_or(CastAuthorityError::CastSourceNotObservation {
                cast,
                input: source,
            }),
    }
}
