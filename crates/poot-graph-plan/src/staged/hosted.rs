//! Step 2b of [`super::compile_staged`]: the hosted gather sites. Every gather-only table leaves the graph
//! ([`crate::passes::host_gather_tables`]); the executor copies the stored rows a step gathers from the
//! checkpoint mapping. A site whose ids the device computes is cut right before the first equation that reads
//! its rows, so the host reads the ids and copies the rows between the stage that computes them and the stage
//! that reads them (a [`super::SyncPoint`] with [`super::SyncAction::GatherRows`]).

use std::borrow::Cow;

use poot_graph_ir::graph::{BoundaryId, Export, Graph, Operand};

use super::StagedCompileError;
use crate::passes::{HostedSite, host_gather_tables};

/// The graph with its gather-only tables hosted.
pub(super) struct Placed<'g> {
    pub(super) graph: Cow<'g, Graph>,
    pub(super) sites: Vec<HostedSite>,
}

pub(super) fn apply<'g>(
    graph: Cow<'g, Graph>,
    aliases: &poot_weights::WeightAliases,
) -> Result<Placed<'g>, StagedCompileError> {
    let hosted = host_gather_tables(&graph, aliases)
        .map_err(|error| StagedCompileError::Hosted(Box::new(error)))?;
    if hosted.sites.is_empty() {
        return Ok(Placed {
            graph,
            sites: hosted.sites,
        });
    }
    // The host reads a device-computed index at its sync, so the uncut graph exports it: the split makes it a
    // boundary of the stage that produces it though no later stage reads it.
    let mut edit = hosted.graph.edit();
    for site in &hosted.sites {
        if let Some(index) = site.device_index {
            edit.push_export(Export {
                id: BoundaryId(index),
                value: index,
            });
        }
    }
    let graph = edit
        .commit()
        .map_err(|error| StagedCompileError::Hosted(Box::new(error)))?;
    Ok(Placed {
        graph: Cow::Owned(graph),
        sites: hosted.sites,
    })
}

/// The equation position right before the first equation that reads a row of `site` in `graph`: where the
/// program is cut, the latest point after the index is known. `None` when nothing reads the rows.
pub(super) fn cut_before_first_reader(graph: &Graph, site: &HostedSite) -> Option<usize> {
    graph.eqns().iter().position(|eqn| {
        eqn.inputs
            .iter()
            .any(|operand| matches!(operand, Operand::Value(value) if site.values.contains(value)))
    })
}
