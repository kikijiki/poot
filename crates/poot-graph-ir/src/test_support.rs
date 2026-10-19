//! Graph fixtures with no production caller, for the graph, planner, executor and device tests across the
//! workspace (card 671: dead under the strict dead-pub rule, a test reference is never a use), and the
//! well-formedness verdict this crate's own unit tests share.
//!
//! [`graph_fixtures`] reimplements the `Builder` convenience methods card 671 deleted from the public
//! `Builder`/`Graph` methods; [`scan`] and [`const_weights`] are the recurrence-scan fixture and the named
//! weights of a toy graph.

pub mod block_top_k;
pub mod const_weights;
pub mod graph_fixtures;
pub mod packed_graphs;
pub mod scan;
pub mod selected_attention;

#[cfg(test)]
use crate::error::GraphValidationError;
#[cfg(test)]
use crate::graph::{EditError, Graph};

/// The well-formedness verdict on `graph` as an edit commit gives it: `Ok` when re-committing the graph
/// is accepted, the typed [`GraphValidationError`] when the draft is refused as malformed. Any other
/// refusal (a changed interface) is a test bug and panics.
#[cfg(test)]
pub(crate) fn verdict(graph: &Graph) -> Result<(), GraphValidationError> {
    match graph.edit().commit() {
        Ok(_) => Ok(()),
        Err(EditError::Invalid(error)) => Err(error),
        Err(other) => panic!("expected a validity verdict, got {other}"),
    }
}
