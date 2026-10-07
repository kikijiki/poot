//! Routed-expert sites: the graph shape cards 186 and 192 recorded as hanging ROCm and PTX under
//! flash-attention synthesis, fusion and the generated tiled GEMM.
//!
//! A site is a router-logits contraction over more than one row whose `ArgTopK` expert ids feed an
//! indexed contraction. The shape is what decides, not the family: a one-row decode has no site, a
//! `ArgTopK` that feeds nothing indexed (a sampler's top-k) has none, and every routed family has one
//! at prefill, however it spells its router.
//!
//! [`compile`](crate::compile) asks for the sites only when the target carries
//! `KnownMiscompiles::routed_expert_tiling_hang`; POOT-575 deletes both the fact and this walk.

use std::collections::HashMap;

use poot_graph_ir::graph::{Graph, Operand, ValidationChannel, ValueId};
use poot_graph_ir::op::OpKind;

/// One routed-expert site, as equation indices into the graph's `eqns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedExpertSite {
    /// The router-logits contraction (`MatMul`) over more than one row.
    pub router: usize,
    /// The `ArgTopK` selecting each row's experts from the router's scores.
    pub select: usize,
    /// The indexed contraction whose per-row expert index the selection feeds.
    pub experts: usize,
}

fn value_inputs(inputs: &[Operand]) -> impl Iterator<Item = ValueId> + '_ {
    inputs.iter().filter_map(|operand| match operand {
        Operand::Value(id) => Some(*id),
        Operand::Lit(_) => None,
    })
}

/// Every routed-expert site of `g`.
pub fn routed_expert_sites<V: ValidationChannel>(g: &Graph<V>) -> Vec<RoutedExpertSite> {
    let producer: HashMap<ValueId, usize> = g
        .eqns
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let mut consumers: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, eqn) in g.eqns.iter().enumerate() {
        for input in value_inputs(&eqn.inputs) {
            consumers.entry(input).or_default().push(i);
        }
    }

    let mut sites = Vec::new();
    for (select, eqn) in g.eqns.iter().enumerate() {
        if !matches!(eqn.op, OpKind::ArgTopK { .. }) {
            continue;
        }
        let Some(router) = upstream_router(g, &producer, select) else {
            continue;
        };
        let Some(experts) = downstream_indexed(g, &consumers, select) else {
            continue;
        };
        sites.push(RoutedExpertSite {
            router,
            select,
            experts,
        });
    }
    sites
}

/// The nearest `MatMul` over more than one row feeding equation `start`: the walk stops at the first
/// contraction on every path, so it never reaches an earlier layer's.
fn upstream_router<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &HashMap<ValueId, usize>,
    start: usize,
) -> Option<usize> {
    let mut seen = vec![false; g.eqns.len()];
    let mut stack = vec![start];
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut seen[i], true) {
            continue;
        }
        let eqn = &g.eqns[i];
        if i != start && matches!(eqn.op, OpKind::MatMul) {
            let shape = &g.aval(eqn.out).shape;
            let rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();
            if rows > 1 {
                return Some(i);
            }
            continue;
        }
        stack.extend(value_inputs(&eqn.inputs).filter_map(|v| producer.get(&v).copied()));
    }
    None
}

/// The first `IndexedMatMul` whose per-row expert index (its third operand) is reached from the
/// output of equation `start`.
fn downstream_indexed<V: ValidationChannel>(
    g: &Graph<V>,
    consumers: &HashMap<ValueId, Vec<usize>>,
    start: usize,
) -> Option<usize> {
    let mut seen = vec![false; g.eqns.len()];
    let mut stack = vec![g.eqns[start].out];
    while let Some(value) = stack.pop() {
        for &i in consumers.get(&value).into_iter().flatten() {
            let eqn = &g.eqns[i];
            if matches!(eqn.op, OpKind::IndexedMatMul)
                && matches!(eqn.inputs.get(2), Some(Operand::Value(idx)) if *idx == value)
            {
                return Some(i);
            }
            if !std::mem::replace(&mut seen[i], true) {
                stack.push(eqn.out);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, TensorType, ops};

    use super::*;

    /// `rows` tokens of width 4 through a router over 3 experts, the `ArgTopK` of its ranks as per-row
    /// expert ids; `consume` decides whether an indexed contraction reads them.
    fn routed(rows: usize, consume: bool) -> Graph {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, 4]));
        let router = b.constant("router", TensorType::f32(vec![4, 3]));
        let logits = b.matmul(x, router);
        let ids = b.reshape(
            b.arg_top_k(ops::stable_descending_rank(&b, logits), 1),
            vec![rows],
        );
        let out = if consume {
            let experts = b.constant("experts", TensorType::f32(vec![3, 4, 2]));
            b.indexed_matmul(x, experts, ids)
        } else {
            b.reshape(ids, vec![rows, 1])
        };
        b.finish(out)
    }

    fn sites(g: &Graph) -> usize {
        routed_expert_sites(g).len()
    }

    /// A router over several rows whose selection feeds an indexed contraction is a site; the same
    /// router over one row (a decode), a selection nothing indexed reads (a sampler's top-k), and an
    /// `ArgTopK` with no router contraction behind it are not.
    ///
    /// Mutation (observed red): match every `ArgTopK` in `routed_expert_sites`; the one-row, the
    /// unconsumed and the router-less rows each report a site.
    #[test]
    fn a_site_is_a_multi_row_router_whose_selection_feeds_an_indexed_contraction() {
        assert_eq!(sites(&routed(4, true)), 1, "a prefill router");
        assert_eq!(sites(&routed(1, true)), 0, "a one-row router is a decode");
        assert_eq!(
            sites(&routed(4, false)),
            0,
            "the selection feeds nothing indexed"
        );

        let b = Builder::new();
        let scores = b.constant("scores", TensorType::f32(vec![4, 3]));
        let ids = b.arg_top_k(scores, 1);
        assert_eq!(
            sites(&b.finish(ids)),
            0,
            "no router contraction behind the selection"
        );
    }

    /// The site names the router, the selection and the indexed contraction it found.
    #[test]
    fn a_site_names_its_three_equations() {
        let g = routed(4, true);
        let [site] = routed_expert_sites(&g)[..] else {
            panic!("one site");
        };
        assert!(matches!(g.eqns[site.router].op, OpKind::MatMul));
        assert!(matches!(g.eqns[site.select].op, OpKind::ArgTopK { .. }));
        assert!(matches!(g.eqns[site.experts].op, OpKind::IndexedMatMul));
    }
}
