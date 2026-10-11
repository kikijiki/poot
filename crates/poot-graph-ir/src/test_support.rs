//! Test-only reconstructions of the `Builder` convenience methods card 671 deleted
//! (`i32_constant`, `finish_with_validations`, `finish_with_state_and_validations`: dead in
//! production, moved to `poot_test_util::graph_fixtures`), shared by this crate's own
//! `#[cfg(test)]` modules (`builder::tests`, `graph::tests`, `index_guard::tests`).
//!
//! Duplicated here rather than imported from `poot-test-util`: that crate's `graph-fixtures`
//! feature regular-depends on `poot-graph-ir` (for these exact fixtures), so this crate's own
//! unit tests dev-depending on it back would compile two mismatched instances of `Builder`/
//! `Traced`/`Graph` (the same reasoning as `poot-kernel-ir`'s and `poot-eval`'s equivalents - see
//! their commit messages). Every function here is built only from already-public `Builder`/
//! `Graph` methods, exactly as `poot_test_util::graph_fixtures` builds its own copy.

use crate::Traced;
use crate::builder::Builder;
use crate::builder::Outputs;
use crate::error::{BuilderAppendError, GraphValidationError};
use crate::graph::{EditError, Graph, ValidationId};
use crate::types::{DType, TensorType};

pub(crate) fn i32_constant(
    b: &Builder,
    name: &str,
    shape: Vec<usize>,
) -> Result<Traced, BuilderAppendError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| BuilderAppendError::ElementCountOverflow {
            name: name.to_string(),
            shape: shape.clone(),
        })?;
    let mut plan = b.append_plan(0);
    plan.input_result(
        name.to_string(),
        TensorType::new(shape, DType::I32),
        crate::graph::Storage::Const,
    )?;
    let mut prepared = b.preflight_append(plan)?;
    b.commit_append(&mut prepared)
}

pub(crate) fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph, GraphValidationError> {
    finish_with_state_and_validations(b, out, &[], validations)
}

pub(crate) fn finish_with_state_and_validations(
    b: Builder,
    out: Traced,
    state: &[(Traced, Traced)],
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph, GraphValidationError> {
    let outputs = validations.iter().fold(
        Outputs::new(out).with_state(state),
        |outputs, &(id, name, value)| outputs.with_witness(id, name, value),
    );
    b.finish(outputs).map_err(|error| match error {
        EditError::Invalid(error) => error,
        other => panic!("a fresh builder has no baseline to break: {other}"),
    })
}

/// The well-formedness verdict on `graph` as an edit commit gives it: `Ok` when re-committing the graph
/// is accepted, the typed [`GraphValidationError`] when the draft is refused as malformed. Any other
/// refusal (a changed interface) is a test bug and panics.
pub(crate) fn verdict(graph: &Graph) -> Result<(), GraphValidationError> {
    match graph.edit().commit() {
        Ok(_) => Ok(()),
        Err(EditError::Invalid(error)) => Err(error),
        Err(other) => panic!("expected a validity verdict, got {other}"),
    }
}
