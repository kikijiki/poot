//! Step 4 of [`super::compile_staged`], the cut list: where the program is cut into stages, and which host
//! syncs sit at each cut.

use poot_graph_ir::Graph;

use crate::passes::{PoolSite, StageSplitError};

/// What the host does at a cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CutSync {
    /// Make the experts of pooled selection `n` resident.
    Experts(usize),
    /// Copy the rows of hosted gather site `n`.
    Gather(usize),
}

/// One cut of a program: where the next stage opens and which host syncs run there (none for a layer cut of a
/// pipeline placement).
pub(super) struct Cut {
    pub(super) position: usize,
    pub(super) syncs: Vec<CutSync>,
}

/// The cuts of a graph in stage order: its sync cuts (a pooled selection's translating gather, a hosted
/// gather's first row reader) and the placement's layer cuts, all resolved to equation positions of `graph`
/// (the pooled graph, whose layer tags the gathers carry). Cuts at one position are one stage boundary, whose
/// syncs run in the order given.
pub(super) fn cuts(
    graph: &Graph,
    after_layers: &[usize],
    pooled: &[(PoolSite, usize)],
    gathers: &[usize],
) -> Result<Vec<Cut>, StageSplitError> {
    let mut cuts: Vec<Cut> = Vec::new();
    let mut at = |position: usize, sync: Option<CutSync>| match cuts
        .iter_mut()
        .find(|cut| cut.position == position)
    {
        Some(cut) => cut.syncs.extend(sync),
        None => cuts.push(Cut {
            position,
            syncs: sync.into_iter().collect(),
        }),
    };
    for (site, (_, position)) in pooled.iter().enumerate() {
        at(*position, Some(CutSync::Experts(site)));
    }
    for (site, &position) in gathers.iter().enumerate() {
        at(position, Some(CutSync::Gather(site)));
    }
    if !after_layers.is_empty() {
        let runs = graph.layer_runs()?;
        for &layer in after_layers {
            let position = runs.runs.get(layer + 1).map(|run| run.start).ok_or(
                StageSplitError::LayerCutOutOfRange {
                    cut: layer,
                    layers: runs.layers,
                },
            )?;
            at(position, None);
        }
    }
    cuts.sort_by_key(|cut| cut.position);
    Ok(cuts)
}
