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
use crate::error::{BuilderAppendError, GraphValidationError};
use crate::graph::{Graph, ValidationId, ValidationOutput, ValidationOutputs};
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
    let value = plan.input_result(
        name.to_string(),
        TensorType::new(shape, DType::I32),
        crate::graph::Storage::Const,
    )?;
    let mut prepared = b.preflight_append(plan)?;
    let id = b.commit_append(&mut prepared)?;
    debug_assert_eq!(id, value.id);
    Ok(value)
}

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
