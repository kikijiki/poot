//! Test-only reconstructions of `poot-graph-ir`'s `Builder` convenience methods card 671 deleted
//! (`finish_with_validations`, `finish_with_state_and_validations`: dead in production, moved to
//! `poot_test_util::graph_fixtures`), shared by this crate's own `#[cfg(test)]` modules.
//!
//! Duplicated here rather than imported from `poot-test-util`: that crate's `graph-fixtures`
//! feature regular-depends on `poot-graph-plan` (for its own graph fixtures), so this crate's own
//! unit tests dev-depending on it back would compile two mismatched instances of `poot-graph-ir`'s
//! `Builder`/`Traced`/`Graph` (the same reasoning as `poot-graph-ir`'s and `poot-eval`'s
//! equivalents - see their commit messages). Every function here is built only from
//! `poot-graph-ir`'s already-public `Builder`/`Graph` methods, exactly as
//! `poot_test_util::graph_fixtures` builds its own copy.

use poot_graph_ir::{
    Builder, Graph, GraphValidationError, Traced, ValidationId, ValidationOutput, ValidationOutputs,
};

pub(crate) fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<ValidationOutputs>, GraphValidationError> {
    finish_with_state_and_validations(b, out, &[], validations)
}

pub(crate) fn finish_with_state_and_validations(
    b: Builder,
    out: Traced,
    state: &[(Traced, Traced)],
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<ValidationOutputs>, GraphValidationError> {
    let g = b.finish_with_state(out, state).with_validations(
        validations
            .iter()
            .map(|&(id, name, value)| ValidationOutput {
                id,
                name: name.to_owned(),
                value: value.id,
            })
            .collect(),
    );
    g.validate()?;
    Ok(g)
}
