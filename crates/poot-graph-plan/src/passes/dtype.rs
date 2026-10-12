use super::{EditError, Eqn, Graph, OpKind, Operand, dce, with_eqns};
use poot_graph_ir::ValueTable;

/// A `Transpose` is a pure data no-op (the row-major flat buffer is unchanged) iff its non-unit input dims
/// keep their relative order under the permutation - only size-1 axes are shuffled. Then `transpose(perm)`
/// equals `reshape(out_shape)`. The common case is decode (seq == 1): the attention reshapes like
/// `[1,H,1,D] -> [1,1,H,D]` move no data but otherwise emit a full copy dispatch.
pub(crate) fn transpose_is_noop(in_shape: &[usize], perm: &[usize]) -> bool {
    let mut last: Option<usize> = None;
    for &p in perm {
        if in_shape[p] > 1 {
            if let Some(l) = last
                && p < l
            {
                return false;
            }
            last = Some(p);
        }
    }
    true
}

/// Collapse chains of `Reshape`s: `reshape(reshape(x))` is `reshape(x)`, since both only reinterpret the
/// same contiguous element sequence, so only the final shape matters. Each reshape's input is rewritten
/// past any producing reshape(s) to the earliest equivalent source, then the dead reshapes are DCE'd.
/// Shape- and value-preserving.
///
/// Fires after [`elide_noop_transposes`]: at seq 1 the attention `reshape -> transpose -> ...` for q/k/v
/// (and the `transpose -> reshape` on the output) becomes `reshape -> reshape` once the no-op transpose
/// is a reshape, in every layer. Reshapes are free buffer aliases, so this removes no dispatches; it
/// cuts node count, and so the per-token host capture/plan work, on the launch-bound decode path.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn collapse_reshape_chains(g: &Graph) -> Result<Graph, EditError> {
    let mut producer: ValueTable<Option<usize>> = ValueTable::filled(g.values().len(), None);
    for (i, e) in g.eqns().iter().enumerate() {
        producer[e.out] = Some(i);
    }
    let is_reshape = |i: usize| matches!(g.eqns()[i].op, OpKind::Reshape { .. });
    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .map(|e| {
            if !matches!(e.op, OpKind::Reshape { .. }) {
                return e.clone();
            }
            let Operand::Value(orig) = e.inputs[0] else {
                return e.clone();
            };
            // chase through producing reshapes to the earliest equivalent source operand.
            let mut src = e.inputs[0];
            let mut cur = orig;
            loop {
                match producer[cur] {
                    Some(p) if is_reshape(p) => {
                        src = g.eqns()[p].inputs[0];
                        match src {
                            Operand::Value(v) => cur = v,
                            Operand::Lit(_) => break,
                        }
                    }
                    _ => break,
                }
            }
            if matches!(src, Operand::Value(s) if s == orig) {
                return e.clone(); // no producing reshape to skip
            }
            Eqn {
                inputs: vec![src],
                ..e.clone()
            }
        })
        .collect();
    dce(&with_eqns(g, eqns)?)
}

/// Rewrite data-no-op `Transpose`s to `Reshape`s. A `Reshape` plans as a free buffer alias (no
/// dispatch) where a `Transpose` is a Compute dispatch, so this cuts launch overhead on the
/// launch-bound wgpu decode (the seq-1 attention transposes move no data). Shape- and
/// value-preserving, so it composes anywhere.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn elide_noop_transposes(g: &Graph) -> Result<Graph, EditError> {
    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .map(|e| {
            if let OpKind::Transpose { perm } = &e.op
                && let Operand::Value(v) = e.inputs[0]
                && transpose_is_noop(&g.aval(v).shape, perm)
            {
                return Eqn {
                    op: OpKind::Reshape {
                        shape: g.aval(e.out).shape.clone(),
                    },
                    ..e.clone()
                };
            }
            e.clone()
        })
        .collect();
    with_eqns(g, eqns)
}
