//! Communication: ordered transfer/reduction/exchange rows over the extended [`crate::CollectiveKind`],
//! their explicit producer/consumer dependency edges, and per-row link/operation capability admission
//! (FR-005, FR-008).
//!
//! Every row's participants, byte accounting, and dependency edges are exactly what the caller declared.
//! Card 360 never infers a route count, a producer, or a consumer from model-specific structure (FR-004).

use std::collections::HashSet;

use poot_graph_ir::RedOp;

use crate::CollectiveKind;
use crate::multi_device::topology::{DeviceId, Topology};

/// A caller-assigned stable identity for one node in the communication dependency graph: a communication
/// row, or an opaque upstream producer / downstream consumer (a compute dispatch, a placement, or
/// anything else outside this module's territory - Card 360 only carries the id, never its meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanNodeId(pub u32);

/// One ordered communication row: a transfer, gather, reduction, or exchange over `kind`, with its exact
/// participants, byte accounting, and explicit upstream/downstream dependency edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommunicationRow {
    pub id: PlanNodeId,
    pub kind: CollectiveKind,
    /// Meaningful for `AllReduce` only, mirroring `Plan::Collective`'s own field.
    pub op: RedOp,
    /// Ordered device group. For `PointToPoint` this is exactly `[from, to]`; for `AllGather`/
    /// `AllReduce`/`AllToAll` it is every participating device.
    pub participants: Vec<DeviceId>,
    pub byte_len: usize,
    /// Bounded temporary storage this row needs beyond its participants' resident buffers (FR-005).
    pub temp_bytes: usize,
    /// Upstream nodes that must complete before this row starts.
    pub producers: Vec<PlanNodeId>,
    /// Downstream nodes that must wait for this row to complete.
    pub consumers: Vec<PlanNodeId>,
    /// Populated only for a row built by `plan_routed_exchange`: `route_counts[i][j]` is the caller's
    /// declared item count from `participants[i]` to `participants[j]`, carried byte-for-byte with no
    /// reordering, aggregation, or inference. Empty for every other row.
    pub route_counts: Vec<Vec<usize>>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CommunicationError {
    #[error("routed exchange has {participant_count} participants but a {rows}x{cols} route table")]
    RouteShapeMismatch {
        participant_count: usize,
        rows: usize,
        cols: usize,
    },
    #[error("routed exchange total byte length overflowed")]
    ByteLenOverflow,
}

/// A caller-declared set of communication rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommunicationPlan {
    pub rows: Vec<CommunicationRow>,
}

/// One directed dependency edge: `from` must complete before `to` starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DependencyEdge {
    pub from: PlanNodeId,
    pub to: PlanNodeId,
}

/// Every producer-to-row and row-to-consumer edge declared across the plan, in row-then-producer/consumer
/// order (FR-005).
///
/// This flattens each row's own declared `producers`/`consumers`; it is not re-derived from a
/// topological sort or reachability walk, so a bug that visits only the first producer (dropping every
/// other declared edge) shows up as a missing [`DependencyEdge`].
pub fn communication_dependency_edges(plan: &CommunicationPlan) -> Vec<DependencyEdge> {
    let mut edges = Vec::new();
    for row in &plan.rows {
        for &producer in &row.producers {
            edges.push(DependencyEdge {
                from: producer,
                to: row.id,
            });
        }
        for &consumer in &row.consumers {
            edges.push(DependencyEdge {
                from: row.id,
                to: consumer,
            });
        }
    }
    edges
}

/// Build the `AllToAll` row for one caller-declared routed exchange (FR-004/FR-005). `route_counts` is
/// carried exactly as supplied: it may be empty, uneven between participants, repeated, or nonmonotonic
/// in either axis, and this function never reorders, sorts, or aggregates it before storing it on the
/// returned row.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn plan_routed_exchange(
    id: PlanNodeId,
    participants: Vec<DeviceId>,
    route_counts: Vec<Vec<usize>>,
    byte_len_per_item: usize,
    temp_bytes: usize,
    producers: Vec<PlanNodeId>,
    consumers: Vec<PlanNodeId>,
) -> Result<CommunicationRow, CommunicationError> {
    if route_counts.len() != participants.len()
        || route_counts
            .iter()
            .any(|row| row.len() != participants.len())
    {
        return Err(CommunicationError::RouteShapeMismatch {
            participant_count: participants.len(),
            rows: route_counts.len(),
            cols: route_counts.first().map_or(0, Vec::len),
        });
    }
    let total_items = route_counts
        .iter()
        .flatten()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or(CommunicationError::ByteLenOverflow)?;
    let byte_len = total_items
        .checked_mul(byte_len_per_item)
        .ok_or(CommunicationError::ByteLenOverflow)?;
    Ok(CommunicationRow {
        id,
        kind: CollectiveKind::AllToAll,
        op: RedOp::Sum,
        participants,
        byte_len,
        temp_bytes,
        producers,
        consumers,
        route_counts,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("no accepted capability row for {from:?} -> {to:?} performing {kind:?}")]
pub struct CapabilityError {
    pub from: DeviceId,
    pub to: DeviceId,
    pub kind: CollectiveKind,
}

/// Admit every communication row against the topology's explicit link-capability rows (FR-008): for
/// a directed `PointToPoint` row requires only its declared `participants[0] -> participants[1]` link;
/// group collectives require every ordered pair of distinct participants. Nothing is inferred from
/// reachability: a link accepted for one `CollectiveKind` does not admit a different one between the
/// same two devices. Participant shape is validated by the caller before this capability-only check.
pub fn admit_communication_capability(
    topology: &Topology,
    rows: &[CommunicationRow],
) -> Result<(), CapabilityError> {
    let accepted: HashSet<(DeviceId, DeviceId, CollectiveKind)> = topology
        .links
        .iter()
        .map(|link| (link.from, link.to, link.kind))
        .collect();
    for row in rows {
        if row.kind == CollectiveKind::PointToPoint && row.participants.len() == 2 {
            let from = row.participants[0];
            let to = row.participants[1];
            if !accepted.contains(&(from, to, row.kind)) {
                return Err(CapabilityError {
                    from,
                    to,
                    kind: row.kind,
                });
            }
            continue;
        }
        for &from in &row.participants {
            for &to in &row.participants {
                if from != to && !accepted.contains(&(from, to, row.kind)) {
                    return Err(CapabilityError {
                        from,
                        to,
                        kind: row.kind,
                    });
                }
            }
        }
    }
    Ok(())
}
