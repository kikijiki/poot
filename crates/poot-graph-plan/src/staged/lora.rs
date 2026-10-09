//! Step 2 of [`super::compile_staged`]: the shared whole-graph LoRA rewrite. It runs once on the
//! uncut graph; the stage compiles that follow receive the adapted graph and never apply it again,
//! so each adapter's projection and layer metadata are rewritten exactly one time.

use std::borrow::Cow;

use poot_graph_ir::{Graph, ValidationChannel};

use super::StagedCompileError;
use crate::passes::GraphBudget;
use crate::{CompileError, CompileOptions, PlanError, adapt};

pub(super) fn apply<'g, V: ValidationChannel>(
    g: &'g Graph<V>,
    options: &CompileOptions,
) -> Result<Cow<'g, Graph<V>>, StagedCompileError> {
    g.validate()
        .map_err(PlanError::from)
        .map_err(CompileError::from)?;
    let budget = GraphBudget::of(&options.limits);
    budget
        .admit("compile input", g)
        .map_err(CompileError::from)?;
    Ok(match &options.lora {
        Some(plan) => Cow::Owned(adapt(g, plan, &budget)?),
        None => Cow::Borrowed(g),
    })
}
