//! `slice_of_concat`: a `Slice` whose range lies inside one part of the `Concat` it reads reads that part
//! instead (POOT-1564, D7 of the POOT-1550 design).
//!
//! `slice(concat(parts, axis), axis, start..end)` with `start..end` inside part `i`'s span is the slice of
//! that part at the part-relative range, and the part itself when the range is the whole part. The pass is
//! exact: it moves no arithmetic and picks the same elements, so `eval(pass(g)) == eval(g)` bit for bit. It
//! removes the concat's consumers one by one; the concat itself goes when the last consumer does (the
//! pipeline's `dce`). The conv window is the case it exists for: the tap slices of `window ++ x` are slices
//! of `window` and `x`, so no tap waits on the concat's copy.
//!
//! A range that straddles two parts keeps reading the concat. A slice that is a pinned value (the graph's
//! output or a state commit) keeps its own equation when it is the whole part: it must keep its own
//! buffer, so it cannot become an alias of the part.

use std::collections::{HashMap, HashSet};

use poot_graph_ir::graph::{EditError, Eqn, Graph, Operand, ValueId};
use poot_graph_ir::op::OpKind;

use super::with_eqns;

/// Rewrite every `Slice` of a `Concat` whose range lies in one part into a read of that part.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn slice_of_concat(g: &Graph) -> Result<Graph, EditError> {
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    let concats: HashMap<ValueId, &Eqn> = g
        .eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::Concat { .. }))
        .map(|eqn| (eqn.out, eqn))
        .collect();
    // A slice that became its part: its readers read the part.
    let mut forwarded: HashMap<ValueId, ValueId> = HashMap::new();
    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns().len());
    for eqn in g.eqns() {
        let mut eqn = eqn.clone();
        for input in &mut eqn.inputs {
            if let Operand::Value(value) = input {
                *value = forwarded.get(value).copied().unwrap_or(*value);
            }
        }
        if let (OpKind::Slice { axis, start, end }, [Operand::Value(source)]) =
            (&eqn.op, eqn.inputs.as_slice())
            && let Some(concat) = concats.get(source)
            && let OpKind::Concat { axis: concat_axis } = concat.op
            && concat_axis == *axis
            && start < end
            && let Some((part, from)) = part_containing(g, concat, *axis, *start..*end)
        {
            let whole = g.aval(part).shape[*axis] == end - start;
            if !whole {
                eqn.op = OpKind::Slice {
                    axis: *axis,
                    start: start - from,
                    end: end - from,
                };
                eqn.inputs = vec![Operand::Value(part)];
            } else if !pinned.contains(&eqn.out) {
                forwarded.insert(eqn.out, part);
                continue;
            }
        }
        eqns.push(eqn);
    }
    with_eqns(g, eqns)
}

/// The part of `concat` whose span on `axis` holds `range`, and where that span starts.
fn part_containing(
    g: &Graph,
    concat: &Eqn,
    axis: usize,
    range: std::ops::Range<usize>,
) -> Option<(ValueId, usize)> {
    let mut from = 0;
    for input in &concat.inputs {
        let Operand::Value(part) = input else {
            return None;
        };
        let to = from + g.aval(*part).shape[axis];
        if from <= range.start && range.end <= to {
            return Some((*part, from));
        }
        from = to;
    }
    None
}
