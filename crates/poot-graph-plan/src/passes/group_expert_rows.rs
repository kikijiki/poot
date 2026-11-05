//! `group_expert_rows` (POOT-1451, `receipts/1451-moe-block-aligned-grouping/design.md`): the rows of every indexed
//! contraction the planner tiles are grouped by expert, block-aligned, so every tile of the contraction is one
//! expert's rows and runs one `K` loop on the matrix units.
//!
//! A tracer emits the semantics only: `IndexedMatMul(x, W, ids)` (or, after `bind_packed_weights`, the indexed
//! `PackedContraction`) over the rows in row order. For each such contraction the target would tile (not the
//! Gemv, the planner's own predicates say which: `candidate`), the pass groups its rows: the contractions reading one
//! ids value form a site (a layer's gate, up and down projections), the site's block is the tile rows that run them
//! cheapest together (`block_of`, the tiling cost model), one `ExpertGroup { experts, block }` of the ids is inserted
//! before the site's first contraction, and each contraction reads it instead of the ids
//! ([`ExpertRows::Grouped`]). The result stays in row order, so nothing else in the graph changes.
//!
//! The pass reads the graph and the planner, and nothing else: no model, family, shape or threshold. A decode
//! (Gemv rows) is left per-row; a site with no block the device can tile is left as it is, and the planner refuses
//! its contractions by name.

use super::{EditError, Eqn, Graph, HashMap, OpKind, Operand, Storage, ValueId, ValueMeta};
use poot_graph_ir::op::{ContractionRows, ExpertGrouping, ExpertRows};
use poot_graph_ir::types::TensorType;
use poot_kernelgen::{TiledRows, TiledShape};
use poot_tensor::DType;

/// The operand position of a per-row indexed contraction's ids: 2 for `IndexedMatMul` (`[x, w, ids]`), 1 for an
/// indexed `PackedContraction` (`[x, ids, carriers..]`).
fn ids_position(op: &OpKind) -> Option<usize> {
    match op {
        OpKind::IndexedMatMul {
            rows: ExpertRows::PerRow,
            ..
        } => Some(2),
        OpKind::PackedContraction {
            rows:
                ContractionRows::Indexed {
                    rows: ExpertRows::PerRow,
                    ..
                },
            ..
        } => Some(1),
        _ => None,
    }
}

/// `op` reading rows grouped in tiles of `block` positions.
fn grouped(op: &OpKind, block: usize) -> OpKind {
    let mut op = op.clone();
    match &mut op {
        OpKind::IndexedMatMul { rows, .. }
        | OpKind::PackedContraction {
            rows: ContractionRows::Indexed { rows, .. },
            ..
        } => *rows = ExpertRows::Grouped { block },
        _ => unreachable!("only a per-row indexed contraction is grouped"),
    }
    op
}

/// `g` with the rows of every indexed contraction `candidate` names (its tiled shape on the target, grouped rows
/// over its buckets) grouped by expert in tiles of the block `block_of` chooses for its site (the contractions
/// reading one ids value over one expert count).
pub(crate) fn group_expert_rows(
    g: &Graph,
    candidate: impl Fn(&Graph, &Eqn) -> Option<TiledShape>,
    block_of: impl Fn(&[TiledShape]) -> Option<usize>,
) -> Result<Graph, EditError> {
    // The sites in the order their first contraction appears: (ids, experts) and each contraction with its shape.
    type Site = ((ValueId, usize), Vec<(usize, TiledShape)>);
    let mut sites: Vec<Site> = Vec::new();
    let mut site_of: HashMap<(ValueId, usize), usize> = HashMap::new();
    for (e, eqn) in g.eqns().iter().enumerate() {
        let Some(position) = ids_position(&eqn.op) else {
            continue;
        };
        let (Some(Operand::Value(ids)), Some(shape)) =
            (eqn.inputs.get(position), candidate(g, eqn))
        else {
            continue;
        };
        let TiledRows::Grouped { groups, .. } = shape.rows else {
            continue;
        };
        let key = (*ids, groups - 1);
        let index = *site_of.entry(key).or_insert_with(|| {
            sites.push((key, Vec::new()));
            sites.len() - 1
        });
        sites[index].1.push((e, shape));
    }

    let mut edit = g.edit();
    // Each grouping's equation and the index it goes in front of, inserted last to first afterwards.
    let mut groupings: Vec<(usize, Eqn)> = Vec::new();
    for ((ids, experts), members) in sites {
        let shapes: Vec<TiledShape> = members.iter().map(|&(_, shape)| shape).collect();
        let Some(block) = block_of(&shapes) else {
            continue;
        };
        let rows = g.aval(ids).shape.first().copied().unwrap_or(0);
        let layout = ExpertGrouping {
            rows,
            experts,
            block,
        };
        let group = edit.values_mut().push(ValueMeta::new(
            TensorType::new(vec![layout.len()], DType::I32),
            Storage::Device,
            None,
        ));
        let first = members[0].0;
        for &(e, _) in &members {
            let eqn = &mut edit.eqns_mut()[e];
            let position =
                ids_position(&eqn.op).expect("a member is a per-row indexed contraction");
            eqn.op = grouped(&eqn.op, block);
            eqn.inputs[position] = Operand::Value(group);
        }
        groupings.push((
            first,
            Eqn {
                op: OpKind::ExpertGroup { experts, block },
                inputs: vec![Operand::Value(ids)],
                out: group,
                layer: g.eqns()[first].layer,
            },
        ));
    }
    groupings.sort_by_key(|&(first, _)| std::cmp::Reverse(first));
    for (first, eqn) in groupings {
        edit.eqns_mut().insert(first, eqn);
    }
    edit.commit()
}
