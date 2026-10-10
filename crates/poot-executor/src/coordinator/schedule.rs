//! The host-side admission of a program's communication rows: which rows the coordinator can carry,
//! in what order, and between which devices.
//!
//! This reads only the program's [`CommunicationPlan`]. It issues no device work, and everything is
//! checked before a coordinator opens a route or records a stage, so no prefix of a plan that later
//! fails admission is ever submitted. Every row is a point-to-point transfer: the planner lowers the
//! reductions and gathers of tensor parallelism to exchanges between the ranks' stages.

use std::collections::{HashMap, HashSet};

use poot_graph_plan::multi_device::{
    CommunicationPlan, CommunicationRow, DependencyEdge, DeviceId, PlanNodeId,
    communication_dependency_edges,
};

/// One admitted row: `bytes` of the producer's exported value move from `from` to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduledTransfer {
    pub row: PlanNodeId,
    pub from: DeviceId,
    pub to: DeviceId,
    pub bytes: usize,
}

/// The admitted rows in a stable order that puts every row after the rows it depends on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Schedule {
    rows: Vec<ScheduledTransfer>,
}

impl Schedule {
    #[cfg(test)]
    pub(crate) fn rows(&self) -> &[ScheduledTransfer] {
        &self.rows
    }

    pub(crate) fn row(&self, id: PlanNodeId) -> Option<&ScheduledTransfer> {
        self.rows.iter().find(|row| row.row == id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleError {
    #[error("communication row id {0:?} is declared more than once")]
    DuplicateRowId(PlanNodeId),
    #[error("communication-row dependency subgraph contains a cycle among rows {rows:?}")]
    Cycle { rows: Vec<PlanNodeId> },
    #[error("communication row {row:?} has {actual} participants; a transfer has exactly 2")]
    ParticipantCount { row: PlanNodeId, actual: usize },
    #[error("communication row {row:?} names participant {device:?} more than once")]
    DuplicateParticipant { row: PlanNodeId, device: DeviceId },
}

/// Admit every row of `plan`, or refuse the whole plan.
pub(crate) fn admit(plan: &CommunicationPlan) -> Result<Schedule, ScheduleError> {
    let indices = row_indices(plan)?;
    for row in &plan.rows {
        validate_row(row)?;
    }
    let edges = communication_dependency_edges(plan);
    let order = stable_communication_order(plan, &indices, &edges)?;
    let rows = order
        .into_iter()
        .map(|index| {
            let row = &plan.rows[index];
            ScheduledTransfer {
                row: row.id,
                from: row.participants[0],
                to: row.participants[1],
                bytes: row.byte_len,
            }
        })
        .collect();
    Ok(Schedule { rows })
}

fn row_indices(plan: &CommunicationPlan) -> Result<HashMap<PlanNodeId, usize>, ScheduleError> {
    let mut indices = HashMap::with_capacity(plan.rows.len());
    for (index, row) in plan.rows.iter().enumerate() {
        if indices.insert(row.id, index).is_some() {
            return Err(ScheduleError::DuplicateRowId(row.id));
        }
    }
    Ok(indices)
}

fn validate_row(row: &CommunicationRow) -> Result<(), ScheduleError> {
    if row.participants.len() != 2 {
        return Err(ScheduleError::ParticipantCount {
            row: row.id,
            actual: row.participants.len(),
        });
    }
    if row.participants[0] == row.participants[1] {
        return Err(ScheduleError::DuplicateParticipant {
            row: row.id,
            device: row.participants[0],
        });
    }
    Ok(())
}

/// Stable topological order over communication rows only. Edges to opaque compute nodes are not
/// followed: they cannot form a cycle among rows.
fn stable_communication_order(
    plan: &CommunicationPlan,
    indices: &HashMap<PlanNodeId, usize>,
    edges: &[DependencyEdge],
) -> Result<Vec<usize>, ScheduleError> {
    let mut successors = vec![Vec::new(); plan.rows.len()];
    let mut indegree = vec![0usize; plan.rows.len()];
    let mut seen_edges = HashSet::new();
    for edge in edges {
        let (Some(&from), Some(&to)) = (indices.get(&edge.from), indices.get(&edge.to)) else {
            continue;
        };
        if seen_edges.insert((from, to)) {
            successors[from].push(to);
            indegree[to] += 1;
        }
    }

    let mut emitted = vec![false; plan.rows.len()];
    let mut order = Vec::with_capacity(plan.rows.len());
    while order.len() < plan.rows.len() {
        let Some(next) =
            (0..plan.rows.len()).find(|&index| !emitted[index] && indegree[index] == 0)
        else {
            let rows = plan
                .rows
                .iter()
                .enumerate()
                .filter_map(|(index, row)| (!emitted[index]).then_some(row.id))
                .collect();
            return Err(ScheduleError::Cycle { rows });
        };
        emitted[next] = true;
        order.push(next);
        for &successor in &successors[next] {
            indegree[successor] -= 1;
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::RedOp;
    use poot_graph_plan::CollectiveKind;

    use super::*;

    fn row(
        id: u32,
        kind: CollectiveKind,
        participants: &[u32],
        byte_len: usize,
    ) -> CommunicationRow {
        CommunicationRow {
            id: PlanNodeId(id),
            kind,
            op: RedOp::Sum,
            participants: participants
                .iter()
                .map(|&device| DeviceId(device))
                .collect(),
            byte_len,
            temp_bytes: 0,
            producers: Vec::new(),
            consumers: Vec::new(),
            route_counts: Vec::new(),
        }
    }

    fn plan(rows: Vec<CommunicationRow>) -> CommunicationPlan {
        CommunicationPlan { rows }
    }

    #[test]
    fn point_to_point_rows_are_admitted_in_dependency_order() {
        let mut first = row(7, CollectiveKind::PointToPoint, &[0, 1], 64);
        let mut second = row(3, CollectiveKind::PointToPoint, &[1, 0], 32);
        // Row 3 consumes what row 7 produces, though it is listed first.
        first.consumers = vec![PlanNodeId(3)];
        second.producers = vec![PlanNodeId(7)];
        let schedule = admit(&plan(vec![second, first])).unwrap();
        assert_eq!(
            schedule.rows(),
            [
                ScheduledTransfer {
                    row: PlanNodeId(7),
                    from: DeviceId(0),
                    to: DeviceId(1),
                    bytes: 64
                },
                ScheduledTransfer {
                    row: PlanNodeId(3),
                    from: DeviceId(1),
                    to: DeviceId(0),
                    bytes: 32
                },
            ]
        );
    }

    #[test]
    fn a_cycle_among_rows_refuses_the_whole_plan() {
        let mut a = row(1, CollectiveKind::PointToPoint, &[0, 1], 4);
        let mut b = row(2, CollectiveKind::PointToPoint, &[1, 0], 4);
        a.consumers = vec![PlanNodeId(2)];
        b.consumers = vec![PlanNodeId(1)];
        assert_eq!(
            admit(&plan(vec![a, b])),
            Err(ScheduleError::Cycle {
                rows: vec![PlanNodeId(1), PlanNodeId(2)]
            })
        );
    }

    #[test]
    fn malformed_participants_and_duplicate_ids_are_refused_by_name() {
        assert_eq!(
            admit(&plan(vec![row(1, CollectiveKind::PointToPoint, &[0], 4)])),
            Err(ScheduleError::ParticipantCount {
                row: PlanNodeId(1),
                actual: 1
            })
        );
        assert_eq!(
            admit(&plan(vec![row(
                1,
                CollectiveKind::PointToPoint,
                &[0, 0],
                4
            )])),
            Err(ScheduleError::DuplicateParticipant {
                row: PlanNodeId(1),
                device: DeviceId(0)
            })
        );
        assert_eq!(
            admit(&plan(vec![
                row(1, CollectiveKind::PointToPoint, &[0, 1], 4),
                row(1, CollectiveKind::PointToPoint, &[1, 0], 4),
            ])),
            Err(ScheduleError::DuplicateRowId(PlanNodeId(1)))
        );
    }
}
