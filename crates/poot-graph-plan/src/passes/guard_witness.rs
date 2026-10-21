//! Which index guards a value's proven range rests on, and whether each is observed.
//!
//! A range that exists only because an index guard replaced the lanes outside it ([`RangeFacts::clamps`])
//! is a fact about the lanes the guard let through: a consumer that must not run on a silently clamped lane
//! (the fused attention's live length, an indexed contraction's expert ids) requires that every such guard's
//! witness is a declared validation output. What a witness reports is a proven fact of the value
//! ([`RangeFacts::observes`]), not a shape this module matches.

use std::collections::HashMap;

use poot_graph_ir::{Graph, RangeFacts, ValueId};

/// The producer index of every value defined by an equation.
pub(crate) fn producer_map(g: &Graph) -> HashMap<ValueId, usize> {
    g.eqns()
        .iter()
        .enumerate()
        .map(|(index, eqn)| (eqn.out, index))
        .collect()
}

/// The condition of the first index guard `value`'s range rests on whose witness no declared validation
/// output observes, `None` when every guard it rests on is observed (or it rests on none).
pub(crate) fn first_unobserved_clamp(
    g: &Graph,
    facts: &RangeFacts,
    value: ValueId,
) -> Option<ValueId> {
    facts.clamps(value).iter().copied().find(|&condition| {
        !g.witnesses()
            .any(|witness| facts.observes(witness.value).contains(&condition))
    })
}
