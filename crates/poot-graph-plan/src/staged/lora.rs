//! Step 2 of [`super::compile_staged`]: the shared whole-graph LoRA rewrite. It runs once on the
//! uncut graph; the stage compiles that follow receive the adapted graph and never apply it again,
//! so each adapter's projection and layer metadata are rewritten exactly one time.

use crate::compile::adapt;
use std::borrow::Cow;

use poot_graph_ir::Graph;

use super::StagedCompileError;
use crate::passes::GraphBudget;
use crate::{CompileError, CompileOptions};

pub(super) fn apply<'g>(
    g: &'g Graph,
    options: &CompileOptions,
) -> Result<Cow<'g, Graph>, StagedCompileError> {
    let budget = GraphBudget::of(&options.limits);
    budget
        .admit("compile input", g)
        .map_err(CompileError::from)?;
    Ok(match &options.lora {
        Some(plan) => Cow::Owned(adapt(g, plan, &budget)?),
        None => Cow::Borrowed(g),
    })
}
