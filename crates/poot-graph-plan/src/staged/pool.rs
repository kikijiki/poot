//! Step 3 of [`super::compile_staged`]: the expert placement. A resident placement leaves the graph
//! as it is. A pooled one rewrites every routed selection to read a resident pool
//! ([`crate::passes::pool_experts`]) and cuts the program at each translating gather, so the host
//! can make the selected experts resident between the stage that computes the ids and the stage that
//! reads the pool.

use std::borrow::Cow;

use poot_graph_ir::Graph;

use super::{ExpertPlacement, StagedCompileError};
use crate::passes::{GraphBudget, PoolSite, StageSplitError, pool_experts};

/// The graph after the expert placement and the routed selections it cut at, in equation order.
pub(super) struct Placed<'g> {
    pub(super) graph: Cow<'g, Graph>,
    /// Each pooled selection and the equation that opens the stage after its sync.
    pub(super) sites: Vec<(PoolSite, usize)>,
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
        }),
        ExpertPlacement::Pooled { slots_per_layer } => {
            let pooled = pool_experts(&graph, slots_per_layer, budget)?;
            let sites = pooled.spec.sites.into_iter().zip(pooled.cuts).collect();
            Ok(Placed {
                graph: Cow::Owned(pooled.graph),
                sites,
            })
        }
    }
}

/// One cut of a pooled program: where the next stage opens and which selection syncs there (`None`
/// for a layer cut of a pipeline placement).
pub(super) struct Cut {
    pub(super) position: usize,
    pub(super) site: Option<usize>,
}

/// The cuts of a pooled graph in stage order: its sync cuts and the placement's layer cuts, both
/// resolved to equation positions of `graph` (the pooled graph, whose layer tags the gathers carry).
/// Two cuts at one position are one stage boundary.
pub(super) fn cuts(
    graph: &Graph,
    after_layers: &[usize],
    sites: &[(PoolSite, usize)],
) -> Result<Vec<Cut>, StageSplitError> {
    let mut cuts: Vec<Cut> = sites
        .iter()
        .enumerate()
        .map(|(site, (_, position))| Cut {
            position: *position,
            site: Some(site),
        })
        .collect();
    if !after_layers.is_empty() {
        let runs = graph.layer_runs()?;
        for &layer in after_layers {
            let position = runs.runs.get(layer + 1).map(|run| run.start).ok_or(
                StageSplitError::LayerCutOutOfRange {
                    cut: layer,
                    layers: runs.layers,
                },
            )?;
            if !cuts.iter().any(|cut| cut.position == position) {
                cuts.push(Cut {
                    position,
                    site: None,
                });
            }
        }
    }
    cuts.sort_by_key(|cut| cut.position);
    Ok(cuts)
}
