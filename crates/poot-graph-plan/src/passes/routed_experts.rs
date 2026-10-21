//! Routed-expert sites: the graph shape cards 186 and 192 recorded as hanging ROCm and PTX under
//! flash-attention synthesis, fusion and the generated tiled GEMM (POOT-1038 measured it completing on both
//! with every pass on, so no compile reads a site to skip one).
//!
//! A site is a router-logits contraction whose `ArgTopK` expert ids feed indexed contractions. The
//! shape is what decides, not the family: an `ArgTopK` that feeds nothing indexed (a sampler's top-k)
//! has no site, and every routed family has one at prefill and at decode, however it spells its router.
//!
//! The walk's production caller is the expert-pool transform (`pool`), which pages the tables the
//! site's indexed contractions read.

use std::collections::HashMap;

use poot_graph_ir::graph::{Graph, Operand, ValueId};
use poot_graph_ir::op::{ContractionRows, OpKind};

/// One routed-expert site, as equation indices into the graph's `eqns`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub struct RoutedExpertSite {
    /// The router-logits contraction (`MatMul`).
    pub router: usize,
    /// The `ArgTopK` selecting each row's experts from the router's scores.
    pub select: usize,
    /// Every indexed contraction whose per-row expert index the selection feeds, in equation order:
    /// the up and down projections of a layer share one selection.
    pub experts: Vec<usize>,
}

fn value_inputs(inputs: &[Operand]) -> impl Iterator<Item = ValueId> + '_ {
    inputs.iter().filter_map(|operand| match operand {
        Operand::Value(id) => Some(*id),
        Operand::Lit(_) => None,
    })
}

/// Every routed-expert site of `g`.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn routed_expert_sites(g: &Graph) -> Vec<RoutedExpertSite> {
    let producer: HashMap<ValueId, usize> = g
        .eqns()
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let mut consumers: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, eqn) in g.eqns().iter().enumerate() {
        for input in value_inputs(&eqn.inputs) {
            consumers.entry(input).or_default().push(i);
        }
    }

    let mut sites = Vec::new();
    for (select, eqn) in g.eqns().iter().enumerate() {
        if !matches!(eqn.op, OpKind::ArgTopK { .. }) {
            continue;
        }
        let Some(router) = upstream_router(g, &producer, select) else {
            continue;
        };
        let experts = downstream_indexed(g, &consumers, select);
        if experts.is_empty() {
            continue;
        }
        sites.push(RoutedExpertSite {
            router,
            select,
            experts,
        });
    }
    sites
}

/// The nearest `MatMul` feeding equation `start`: the walk stops at the first contraction on every
/// path, so it never reaches an earlier layer's.
fn upstream_router(g: &Graph, producer: &HashMap<ValueId, usize>, start: usize) -> Option<usize> {
    let mut seen = vec![false; g.eqns().len()];
    let mut stack = vec![start];
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut seen[i], true) {
            continue;
        }
        let eqn = &g.eqns()[i];
        if i != start && eqn.op.is_plain_matmul() {
            return Some(i);
        }
        stack.extend(value_inputs(&eqn.inputs).filter_map(|v| producer.get(&v).copied()));
    }
    None
}

/// The operand position of an indexed contraction's per-row expert index, `None` for any other op.
fn expert_index_position(op: &OpKind) -> Option<usize> {
    match op {
        OpKind::IndexedMatMul => Some(2),
        OpKind::PackedContraction {
            rows: ContractionRows::Indexed { .. },
            ..
        } => Some(1),
        _ => None,
    }
}

/// Every indexed contraction whose per-row expert index is reached from the output of equation
/// `start`, in equation order. The walk does not continue through a contraction: its result is an
/// activation, never an index, so a later layer's selection is not this one's.
fn downstream_indexed(
    g: &Graph,
    consumers: &HashMap<ValueId, Vec<usize>>,
    start: usize,
) -> Vec<usize> {
    let mut seen = vec![false; g.eqns().len()];
    let mut found = Vec::new();
    let mut stack = vec![g.eqns()[start].out];
    while let Some(value) = stack.pop() {
        for &i in consumers.get(&value).into_iter().flatten() {
            let eqn = &g.eqns()[i];
            if let Some(position) = expert_index_position(&eqn.op) {
                if matches!(eqn.inputs.get(position), Some(Operand::Value(idx)) if *idx == value)
                    && !found.contains(&i)
                {
                    found.push(i);
                }
                continue;
            }
            if !std::mem::replace(&mut seen[i], true) {
                stack.push(eqn.out);
            }
        }
    }
    found.sort_unstable();
    found
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, TensorType, ops};

    use super::*;

    /// `rows` tokens of width 4 through a router over 3 experts, the `ArgTopK` of its ranks as per-row
    /// expert ids; `consume` decides whether an indexed contraction reads them.
    fn routed(rows: usize, consume: bool) -> Graph {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, 4])).unwrap();
        let router = b.constant("router", TensorType::f32(vec![4, 3])).unwrap();
        let logits = b.matmul(x, router).unwrap();
        let ids = b
            .reshape(
                b.arg_top_k(ops::stable_descending_rank(&b, logits).unwrap(), 1)
                    .unwrap(),
                vec![rows],
            )
            .unwrap();
        let out = if consume {
            let experts = b
                .constant("experts", TensorType::f32(vec![3, 4, 2]))
                .unwrap();
            b.indexed_matmul(x, experts, ids).unwrap()
        } else {
            b.reshape(ids, vec![rows, 1]).unwrap()
        };
        b.finish(out).unwrap()
    }

    fn sites(g: &Graph) -> usize {
        routed_expert_sites(g).len()
    }

    /// A router whose selection feeds an indexed contraction is a site, at prefill and at decode; a
    /// selection nothing indexed reads (a sampler's top-k) and an `ArgTopK` with no router contraction
    /// behind it are not.
    ///
    /// Mutation (observed red): match every `ArgTopK` in `routed_expert_sites`; the unconsumed and the
    /// router-less rows each report a site.
    #[test]
    fn a_site_is_a_multi_row_router_whose_selection_feeds_an_indexed_contraction() {
        assert_eq!(sites(&routed(4, true)), 1, "a prefill router");
        assert_eq!(sites(&routed(1, true)), 1, "a one-row router is a decode");
        assert_eq!(
            sites(&routed(4, false)),
            0,
            "the selection feeds nothing indexed"
        );

        let b = Builder::for_tests();
        let scores = b.constant("scores", TensorType::f32(vec![4, 3])).unwrap();
        let ids = b.arg_top_k(scores, 1).unwrap();
        assert_eq!(
            sites(&b.finish(ids).unwrap()),
            0,
            "no router contraction behind the selection"
        );
    }

    /// The site names the router, the selection and the indexed contraction it found.
    #[test]
    fn a_site_names_its_three_equations() {
        let g = routed(4, true);
        let [ref site] = routed_expert_sites(&g)[..] else {
            panic!("one site");
        };
        assert!(matches!(g.eqns()[site.router].op, OpKind::MatMul { .. }));
        assert!(matches!(g.eqns()[site.select].op, OpKind::ArgTopK { .. }));
        assert_eq!(site.experts.len(), 1);
        assert!(matches!(
            g.eqns()[site.experts[0]].op,
            OpKind::IndexedMatMul
        ));
    }
}
