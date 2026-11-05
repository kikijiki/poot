//! Step 3 of [`super::compile_staged`]: the expert placement. A resident placement leaves the graph
//! as it is. A pooled one rewrites every routed selection to read a resident pool
//! ([`crate::passes::pool_experts`]) and cuts the program at each translating gather, so the host
//! can make the selected experts resident between the stage that computes the ids and the stage that
//! reads the pool.

use std::borrow::Cow;

use poot_graph_ir::Graph;
use poot_graph_ir::analysis::NumericsProperty;

use super::{ExpertPlacement, StagedCompileError};
use crate::passes::{GraphBudget, PoolSite, pool_experts};

/// The graph after the expert placement and the routed selections it cut at, in equation order.
pub(super) struct Placed<'g> {
    pub(super) graph: Cow<'g, Graph>,
    /// Each pooled selection and the equation that opens the stage after its sync.
    pub(super) sites: Vec<(PoolSite, usize)>,
    /// What the placement did to the graph's arithmetic: nothing for a resident placement.
    pub(super) numerics: NumericsProperty,
}

pub(super) fn apply<'g>(
    graph: Cow<'g, Graph>,
    experts: ExpertPlacement,
    budget: &GraphBudget,
) -> Result<Placed<'g>, StagedCompileError> {
    match experts {
        ExpertPlacement::AllResident => Ok(Placed {
            graph,
            sites: Vec::new(),
            numerics: NumericsProperty::Exact,
        }),
        ExpertPlacement::Pooled { slots_per_layer } => {
            let pooled = pool_experts(&graph, slots_per_layer, budget)?;
            let sites = pooled.spec.sites.into_iter().zip(pooled.cuts).collect();
            Ok(Placed {
                graph: Cow::Owned(pooled.graph),
                sites,
                numerics: pooled.numerics,
            })
        }
    }
}
