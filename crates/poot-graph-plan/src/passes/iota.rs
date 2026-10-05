//! `fold_iota`: lower the nullary [`OpKind::Iota`] to a compiler-computed graph constant (R466-019).
//!
//! An `Iota` has a static shape and no operands, so its value is a pure constant known at compile time.
//! Folding it into a [`Storage::Computed`] value means the planner never lowers an iota kernel and no
//! checkpoint binder supplies the range: the fold computes the definition, and an executor materializes
//! it from that definition like a slot from its kind. A graph that reaches the planner with an unfolded
//! `Iota` is refused with a typed error, so the pass is the one route to a planned iota.
//!
//! Value-preserving: every consumer of the folded value reads the same elements it read before.

use poot_graph_ir::graph::{ComputedConst, Graph, Storage, ValidationChannel};
use poot_graph_ir::op::OpKind;

/// Rewrite every `OpKind::Iota { len }` into a [`Storage::Computed`] graph constant and drop its
/// equation. The folded value stays the same [`ValueId`](poot_graph_ir::ValueId), so consumers need no rewiring.
///
/// A graph with no `Iota` is returned unchanged (a clone-cost no-op callers still pay; the pass is
/// written for the pipeline's simple `Graph -> Graph` shape).
pub fn fold_iota<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut g = g.clone();
    let mut folded: Vec<(usize, usize)> = Vec::new();
    for eqn in &g.eqns {
        if let OpKind::Iota { len } = &eqn.op {
            folded.push((eqn.out, *len));
        }
    }
    if folded.is_empty() {
        return g;
    }
    for &(value, len) in &folded {
        g.values[value].storage = Storage::Computed(ComputedConst::Iota { len });
    }
    g.eqns.retain(|eqn| !matches!(eqn.op, OpKind::Iota { .. }));
    for &(value, _) in &folded {
        g.inputs.push(value);
        g.consts.push(value);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Builder;
    use poot_graph_ir::graph::Operand;
    use poot_graph_ir::op::BinOp;

    fn iota_count<V: ValidationChannel>(g: &Graph<V>, len: usize) -> usize {
        g.eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::Iota { len: l } if l == len))
            .count()
    }

    fn computed_count<V: ValidationChannel>(g: &Graph<V>) -> usize {
        g.inputs
            .iter()
            .filter(|&&id| matches!(g.meta(id).storage, Storage::Computed(_)))
            .count()
    }

    /// `Iota` is a pure nullary op: its whole identity is `len` (the dtype is fixed F32), so CSE must
    /// merge two calls with the same length, and only those. `optimize` runs `cse` before `fold_iota`.
    #[test]
    fn cse_merges_identical_iotas_and_keeps_distinct_lengths() {
        let b = Builder::new();
        let a = b.iota(4);
        let c = b.iota(4);
        let out = b.binary(BinOp::Add, a, c);
        let graph = b.finish(out);

        let merged = crate::passes::cse(&graph);
        assert_eq!(iota_count(&merged, 4), 1, "identical iotas must merge");
        let add = merged
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Binary(BinOp::Add)))
            .expect("the Add survives");
        assert!(
            matches!(add.inputs.as_slice(), [Operand::Value(l), Operand::Value(r)] if l == r),
            "the merged Add must read one iota value twice"
        );
        assert_eq!(
            computed_count(&fold_iota(&merged)),
            1,
            "folding the merged graph yields one computed constant"
        );
        assert_eq!(
            computed_count(&fold_iota(&crate::passes::cse(&graph))),
            1,
            "cse then fold_iota yields one computed constant"
        );

        let b = Builder::new();
        let a = b.iota(4);
        let c = b.iota(5);
        let out = b.concat(0, &[a, c]);
        let graph = b.finish(out);

        let merged = crate::passes::cse(&graph);
        assert_eq!(iota_count(&merged, 4), 1);
        assert_eq!(iota_count(&merged, 5), 1);
        assert_eq!(
            computed_count(&fold_iota(&merged)),
            2,
            "distinct lengths stay distinct through the fold"
        );
        assert_eq!(
            computed_count(&fold_iota(&crate::passes::cse(&graph))),
            2,
            "distinct lengths stay distinct through cse then fold_iota"
        );
    }
}
