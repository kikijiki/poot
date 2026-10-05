//! Host-side communication schedule admission for Card 408's captured multi-device PTX executor.
//!
//! The schedule compiler consumes Card 360's backend-neutral [`CommunicationPlan`] without issuing CUDA
//! work. It is the fail-closed boundary before [`PtxCapturedMultiDeviceExecutor`]: participant and
//! capability checks finish first, unsupported reductions are rejected by name, communication-row
//! dependencies are put in stable topological order, and direct transfers receive normalized
//! source/destination byte ranges. The executor submodule owns stable buffers, distinct per-device CUDA
//! graphs, external-event ordering, and cold/warm replay counters on one host thread.

use std::collections::{HashMap, HashSet};

use poot_graph_ir::RedOp;
use poot_graph_plan::CollectiveKind;
use poot_graph_plan::multi_device::{
    CapabilityError, CommunicationPlan, CommunicationRow, DependencyEdge, DeviceId, PlanNodeId,
    Topology, TopologyError, admit_communication_capability, communication_dependency_edges,
    validate_topology,
};
use poot_target::Backend;

mod error;
mod executor;
mod replay_inputs;
mod segment;
pub use executor::{
    CaptureBufferView, CapturedMultiDeviceError, DeviceCommunicationBuffers,
    PtxCapturedMultiDeviceExecutor, PtxCommunicationBufferBinding, PtxComputeSegment,
    PtxDeviceCapture, PtxDeviceSegment, PtxMultiDeviceReplayCounters, PtxMultiDeviceReplayReport,
    SegmentCapture, SegmentProgram, StaticBufferUpload,
};

/// One normalized byte range selected from a communication row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledTransfer {
    pub from: DeviceId,
    pub to: DeviceId,
    pub src_byte_offset: usize,
    pub dst_byte_offset: usize,
    /// Exact bytes for this pair. This is never converted to an f32 element count.
    pub byte_len: usize,
    /// The caller-declared route-table cell for `AllToAll`; absent for `PointToPoint`.
    pub route: Option<(usize, usize)>,
}

/// One rank's required source or destination extent in a normalized all-to-all layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankByteExtent {
    pub device: DeviceId,
    pub byte_len: usize,
}

/// Row-major source packing and column-major destination packing for one all-to-all row.
///
/// `remote_transfers` contains only distinct device pairs. Nonzero diagonal cells are retained as
/// `local_copies`, so they still occupy and move their declared ranges when source and destination are
/// separate persistent buffers. The complete executor may implement a local copy without P2P.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedAllToAll {
    pub bytes_per_item: usize,
    pub source_extents: Vec<RankByteExtent>,
    pub destination_extents: Vec<RankByteExtent>,
    pub remote_transfers: Vec<ScheduledTransfer>,
    pub local_copies: Vec<ScheduledTransfer>,
}

/// The work retained for one admitted communication row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduledCommunication {
    PointToPoint(ScheduledTransfer),
    AllToAll(NormalizedAllToAll),
    /// The existing P2P ring supplies this operation; Card 408 later captures it per device.
    AllReduceSum,
    /// Card 408 later supplies a captured all-gather schedule over the existing transport.
    AllGather,
}

/// One row in communication-subgraph dependency order. `wait_for` and `signal_to` retain both
/// communication-row and opaque compute/placement node ids for the complete executor to resolve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledCommunicationRow {
    pub id: PlanNodeId,
    pub participants: Vec<DeviceId>,
    pub byte_len: usize,
    pub temp_bytes: usize,
    pub wait_for: Vec<PlanNodeId>,
    pub signal_to: Vec<PlanNodeId>,
    pub communication: ScheduledCommunication,
}

/// An admitted host schedule. Constructing this value performs no device work. `dependency_edges`
/// preserves Card 360's complete declared edge list, including edges to opaque non-communication nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommunicationSchedule {
    pub rows: Vec<ScheduledCommunicationRow>,
    pub dependency_edges: Vec<DependencyEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CommunicationScheduleError {
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    #[error("communication row id {0:?} is declared more than once")]
    DuplicateRowId(PlanNodeId),
    #[error("communication-row dependency subgraph contains a cycle among rows {rows:?}")]
    CommunicationSubgraphCycle { rows: Vec<PlanNodeId> },
    #[error(
        "communication row {row:?} kind {kind:?} has {actual} participants; expected {expected}"
    )]
    InvalidParticipantCount {
        row: PlanNodeId,
        kind: CollectiveKind,
        actual: usize,
        expected: &'static str,
    },
    #[error("communication row {row:?} names unknown participant {device:?}")]
    UnknownParticipant { row: PlanNodeId, device: DeviceId },
    #[error("communication row {row:?} names participant {device:?} more than once")]
    DuplicateParticipant { row: PlanNodeId, device: DeviceId },
    #[error(
        "communication row {row:?} participant {device:?} uses {backend:?}, expected Backend::Nvptx"
    )]
    NonNvptxParticipant {
        row: PlanNodeId,
        device: DeviceId,
        backend: Backend,
    },
    #[error(
        "all-to-all row {row:?} has {participants} participants but a {rows}x{cols} route table"
    )]
    AllToAllRouteShape {
        row: PlanNodeId,
        participants: usize,
        rows: usize,
        cols: usize,
    },
    #[error(
        "all-to-all row {row:?} declares {byte_len} bytes for {items} items; no positive integral byte width exists"
    )]
    AllToAllByteWidth {
        row: PlanNodeId,
        byte_len: usize,
        items: usize,
    },
    #[error("all-to-all row {row:?} byte accounting overflowed")]
    AllToAllByteOverflow { row: PlanNodeId },
    #[error(
        "all-to-all row {row:?} normalized to source={source_bytes} and destination={destination_bytes} bytes, expected {declared_bytes}"
    )]
    AllToAllLayoutMismatch {
        row: PlanNodeId,
        declared_bytes: usize,
        source_bytes: usize,
        destination_bytes: usize,
    },
    #[error("all-reduce max is unsupported for communication row {row:?}")]
    UnsupportedAllReduceMax { row: PlanNodeId },
}

/// Validate and compile Card 360 communication rows into the host schedule consumed by Card 408's
/// later device executor.
///
/// All checks finish before the schedule is returned, so callers cannot issue a prefix of transfers
/// from a plan that later fails participant, capability, reduction, route-layout, or dependency admission.
pub fn compile_communication_schedule(
    topology: &Topology,
    plan: &CommunicationPlan,
) -> Result<CommunicationSchedule, CommunicationScheduleError> {
    validate_topology(topology)?;

    let row_indices = row_indices(plan)?;
    let devices: HashMap<_, _> = topology
        .devices
        .iter()
        .map(|device| (device.id, device.backend))
        .collect();
    for row in &plan.rows {
        validate_row(row, &devices)?;
    }
    admit_communication_capability(topology, &plan.rows)?;

    let dependency_edges = communication_dependency_edges(plan);
    let order = stable_communication_order(plan, &row_indices, &dependency_edges)?;
    let mut rows = Vec::with_capacity(plan.rows.len());
    for index in order {
        let row = &plan.rows[index];
        rows.push(ScheduledCommunicationRow {
            id: row.id,
            participants: row.participants.clone(),
            byte_len: row.byte_len,
            temp_bytes: row.temp_bytes,
            wait_for: adjacent_nodes(row.id, &dependency_edges, true),
            signal_to: adjacent_nodes(row.id, &dependency_edges, false),
            communication: schedule_row(row)?,
        });
    }
    Ok(CommunicationSchedule {
        rows,
        dependency_edges,
    })
}

fn row_indices(
    plan: &CommunicationPlan,
) -> Result<HashMap<PlanNodeId, usize>, CommunicationScheduleError> {
    let mut indices = HashMap::with_capacity(plan.rows.len());
    for (index, row) in plan.rows.iter().enumerate() {
        if indices.insert(row.id, index).is_some() {
            return Err(CommunicationScheduleError::DuplicateRowId(row.id));
        }
    }
    Ok(indices)
}

fn validate_row(
    row: &CommunicationRow,
    devices: &HashMap<DeviceId, Backend>,
) -> Result<(), CommunicationScheduleError> {
    let (valid_count, expected) = match row.kind {
        CollectiveKind::PointToPoint => (row.participants.len() == 2, "exactly 2"),
        _ => (row.participants.len() >= 2, "at least 2"),
    };
    if !valid_count {
        return Err(CommunicationScheduleError::InvalidParticipantCount {
            row: row.id,
            kind: row.kind,
            actual: row.participants.len(),
            expected,
        });
    }

    let mut seen = HashSet::with_capacity(row.participants.len());
    for &participant in &row.participants {
        if !seen.insert(participant) {
            return Err(CommunicationScheduleError::DuplicateParticipant {
                row: row.id,
                device: participant,
            });
        }
        let Some(&backend) = devices.get(&participant) else {
            return Err(CommunicationScheduleError::UnknownParticipant {
                row: row.id,
                device: participant,
            });
        };
        if backend != Backend::Nvptx {
            return Err(CommunicationScheduleError::NonNvptxParticipant {
                row: row.id,
                device: participant,
                backend,
            });
        }
    }

    match row.kind {
        CollectiveKind::AllToAll => {
            let participants = row.participants.len();
            if row.route_counts.len() != participants
                || row
                    .route_counts
                    .iter()
                    .any(|counts| counts.len() != participants)
            {
                return Err(CommunicationScheduleError::AllToAllRouteShape {
                    row: row.id,
                    participants,
                    rows: row.route_counts.len(),
                    cols: row.route_counts.first().map_or(0, Vec::len),
                });
            }
            let items = total_route_items(row)?;
            if (items == 0 && row.byte_len != 0)
                || (items != 0 && (row.byte_len == 0 || !row.byte_len.is_multiple_of(items)))
            {
                return Err(CommunicationScheduleError::AllToAllByteWidth {
                    row: row.id,
                    byte_len: row.byte_len,
                    items,
                });
            }
            Ok(())
        }
        CollectiveKind::AllReduce if row.op == RedOp::Max => {
            Err(CommunicationScheduleError::UnsupportedAllReduceMax { row: row.id })
        }
        _ => Ok(()),
    }
}

fn total_route_items(row: &CommunicationRow) -> Result<usize, CommunicationScheduleError> {
    row.route_counts
        .iter()
        .flatten()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or(CommunicationScheduleError::AllToAllByteOverflow { row: row.id })
}

/// Stable topological order over communication rows only. Edges to opaque compute/placement nodes are
/// retained in the returned schedule but cannot form a cycle claim in this communication-only slice.
fn stable_communication_order(
    plan: &CommunicationPlan,
    indices: &HashMap<PlanNodeId, usize>,
    edges: &[DependencyEdge],
) -> Result<Vec<usize>, CommunicationScheduleError> {
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
            return Err(CommunicationScheduleError::CommunicationSubgraphCycle { rows });
        };
        emitted[next] = true;
        order.push(next);
        for &successor in &successors[next] {
            indegree[successor] -= 1;
        }
    }
    Ok(order)
}

fn adjacent_nodes(id: PlanNodeId, edges: &[DependencyEdge], incoming: bool) -> Vec<PlanNodeId> {
    let mut seen = HashSet::new();
    edges
        .iter()
        .filter_map(|edge| {
            let adjacent = if incoming && edge.to == id {
                Some(edge.from)
            } else if !incoming && edge.from == id {
                Some(edge.to)
            } else {
                None
            };
            adjacent.filter(|node| seen.insert(*node))
        })
        .collect()
}

fn schedule_row(
    row: &CommunicationRow,
) -> Result<ScheduledCommunication, CommunicationScheduleError> {
    match row.kind {
        CollectiveKind::PointToPoint => {
            Ok(ScheduledCommunication::PointToPoint(ScheduledTransfer {
                from: row.participants[0],
                to: row.participants[1],
                src_byte_offset: 0,
                dst_byte_offset: 0,
                byte_len: row.byte_len,
                route: None,
            }))
        }
        CollectiveKind::AllToAll => {
            Ok(ScheduledCommunication::AllToAll(normalize_all_to_all(row)?))
        }
        CollectiveKind::AllReduce => Ok(ScheduledCommunication::AllReduceSum),
        CollectiveKind::AllGather => Ok(ScheduledCommunication::AllGather),
    }
}

fn normalize_all_to_all(
    row: &CommunicationRow,
) -> Result<NormalizedAllToAll, CommunicationScheduleError> {
    let total_items = total_route_items(row)?;
    let bytes_per_item = row.byte_len.checked_div(total_items).unwrap_or_default();
    let mut destination_offsets = vec![0usize; row.participants.len()];
    let mut source_extents = Vec::with_capacity(row.participants.len());
    let mut remote_transfers = Vec::new();
    let mut local_copies = Vec::new();

    for (from_index, counts) in row.route_counts.iter().enumerate() {
        let mut source_offset = 0usize;
        for (to_index, &items) in counts.iter().enumerate() {
            let byte_len = items
                .checked_mul(bytes_per_item)
                .ok_or(CommunicationScheduleError::AllToAllByteOverflow { row: row.id })?;
            let transfer = ScheduledTransfer {
                from: row.participants[from_index],
                to: row.participants[to_index],
                src_byte_offset: source_offset,
                dst_byte_offset: destination_offsets[to_index],
                byte_len,
                route: Some((from_index, to_index)),
            };
            source_offset = source_offset
                .checked_add(byte_len)
                .ok_or(CommunicationScheduleError::AllToAllByteOverflow { row: row.id })?;
            destination_offsets[to_index] = destination_offsets[to_index]
                .checked_add(byte_len)
                .ok_or(CommunicationScheduleError::AllToAllByteOverflow { row: row.id })?;
            if byte_len != 0 {
                if from_index == to_index {
                    local_copies.push(transfer);
                } else {
                    remote_transfers.push(transfer);
                }
            }
        }
        source_extents.push(RankByteExtent {
            device: row.participants[from_index],
            byte_len: source_offset,
        });
    }

    let destination_extents: Vec<_> = row
        .participants
        .iter()
        .copied()
        .zip(destination_offsets)
        .map(|(device, byte_len)| RankByteExtent { device, byte_len })
        .collect();
    let source_bytes = checked_extent_sum(row.id, &source_extents)?;
    let destination_bytes = checked_extent_sum(row.id, &destination_extents)?;
    if source_bytes != row.byte_len || destination_bytes != row.byte_len {
        return Err(CommunicationScheduleError::AllToAllLayoutMismatch {
            row: row.id,
            declared_bytes: row.byte_len,
            source_bytes,
            destination_bytes,
        });
    }

    Ok(NormalizedAllToAll {
        bytes_per_item,
        source_extents,
        destination_extents,
        remote_transfers,
        local_copies,
    })
}

fn checked_extent_sum(
    row: PlanNodeId,
    extents: &[RankByteExtent],
) -> Result<usize, CommunicationScheduleError> {
    extents
        .iter()
        .try_fold(0usize, |total, extent| total.checked_add(extent.byte_len))
        .ok_or(CommunicationScheduleError::AllToAllByteOverflow { row })
}

#[cfg(test)]
mod tests {
    use poot_graph_plan::multi_device::{DeviceSpec, LinkCapability};

    use super::*;

    fn devices(backends: &[Backend]) -> Vec<DeviceSpec> {
        backends
            .iter()
            .enumerate()
            .map(|(id, &backend)| DeviceSpec {
                id: DeviceId(id as u32),
                backend,
                usable_bytes: 1 << 30,
            })
            .collect()
    }

    fn complete_links(device_count: u32, kinds: &[CollectiveKind]) -> Vec<LinkCapability> {
        let mut links = Vec::new();
        for &kind in kinds {
            for from in 0..device_count {
                for to in 0..device_count {
                    if from != to {
                        links.push(LinkCapability {
                            from: DeviceId(from),
                            to: DeviceId(to),
                            kind,
                        });
                    }
                }
            }
        }
        links
    }

    fn nvptx_topology(device_count: u32, kinds: &[CollectiveKind]) -> Topology {
        Topology {
            devices: devices(&vec![Backend::Nvptx; device_count as usize]),
            links: complete_links(device_count, kinds),
        }
    }

    fn row(
        id: u32,
        kind: CollectiveKind,
        participants: Vec<DeviceId>,
        byte_len: usize,
    ) -> CommunicationRow {
        CommunicationRow {
            id: PlanNodeId(id),
            kind,
            op: RedOp::Sum,
            participants,
            byte_len,
            temp_bytes: 0,
            producers: Vec::new(),
            consumers: Vec::new(),
            route_counts: Vec::new(),
        }
    }

    #[test]
    fn point_to_point_accepts_one_way_link_and_keeps_one_4097_byte_range() {
        let topology = Topology {
            devices: devices(&[Backend::Nvptx, Backend::Nvptx, Backend::Nvptx]),
            links: vec![LinkCapability {
                from: DeviceId(0),
                to: DeviceId(1),
                kind: CollectiveKind::PointToPoint,
            }],
        };
        let plan = CommunicationPlan {
            rows: vec![row(
                7,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(1)],
                4097,
            )],
        };

        let schedule = compile_communication_schedule(&topology, &plan).unwrap();
        assert_eq!(schedule.rows.len(), 1);
        assert_eq!(
            schedule.rows[0].communication,
            ScheduledCommunication::PointToPoint(ScheduledTransfer {
                from: DeviceId(0),
                to: DeviceId(1),
                src_byte_offset: 0,
                dst_byte_offset: 0,
                byte_len: 4097,
                route: None,
            })
        );
    }

    #[test]
    fn point_to_point_rejects_missing_forward_link_even_when_reverse_exists() {
        let topology = Topology {
            devices: devices(&[Backend::Nvptx, Backend::Nvptx]),
            links: vec![LinkCapability {
                from: DeviceId(1),
                to: DeviceId(0),
                kind: CollectiveKind::PointToPoint,
            }],
        };
        let plan = CommunicationPlan {
            rows: vec![row(
                8,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(1)],
                33,
            )],
        };

        assert_eq!(
            compile_communication_schedule(&topology, &plan),
            Err(CommunicationScheduleError::Capability(CapabilityError {
                from: DeviceId(0),
                to: DeviceId(1),
                kind: CollectiveKind::PointToPoint,
            }))
        );
    }

    #[test]
    fn all_to_all_normalizes_asymmetric_routes_and_nonzero_diagonal() {
        let topology = nvptx_topology(3, &[CollectiveKind::AllToAll]);
        let mut exchange = row(
            11,
            CollectiveKind::AllToAll,
            vec![DeviceId(0), DeviceId(1), DeviceId(2)],
            232,
        );
        exchange.route_counts = vec![vec![2, 3, 0], vec![5, 7, 11], vec![13, 0, 17]];

        let schedule = compile_communication_schedule(
            &topology,
            &CommunicationPlan {
                rows: vec![exchange],
            },
        )
        .unwrap();
        let ScheduledCommunication::AllToAll(layout) = &schedule.rows[0].communication else {
            unreachable!()
        };
        assert_eq!(layout.bytes_per_item, 4);
        assert_eq!(
            layout.source_extents,
            vec![
                RankByteExtent {
                    device: DeviceId(0),
                    byte_len: 20,
                },
                RankByteExtent {
                    device: DeviceId(1),
                    byte_len: 92,
                },
                RankByteExtent {
                    device: DeviceId(2),
                    byte_len: 120,
                },
            ]
        );
        assert_eq!(
            layout.destination_extents,
            vec![
                RankByteExtent {
                    device: DeviceId(0),
                    byte_len: 80,
                },
                RankByteExtent {
                    device: DeviceId(1),
                    byte_len: 40,
                },
                RankByteExtent {
                    device: DeviceId(2),
                    byte_len: 112,
                },
            ]
        );
        assert_eq!(
            layout.remote_transfers,
            vec![
                ScheduledTransfer {
                    from: DeviceId(0),
                    to: DeviceId(1),
                    src_byte_offset: 8,
                    dst_byte_offset: 0,
                    byte_len: 12,
                    route: Some((0, 1)),
                },
                ScheduledTransfer {
                    from: DeviceId(1),
                    to: DeviceId(0),
                    src_byte_offset: 0,
                    dst_byte_offset: 8,
                    byte_len: 20,
                    route: Some((1, 0)),
                },
                ScheduledTransfer {
                    from: DeviceId(1),
                    to: DeviceId(2),
                    src_byte_offset: 48,
                    dst_byte_offset: 0,
                    byte_len: 44,
                    route: Some((1, 2)),
                },
                ScheduledTransfer {
                    from: DeviceId(2),
                    to: DeviceId(0),
                    src_byte_offset: 0,
                    dst_byte_offset: 28,
                    byte_len: 52,
                    route: Some((2, 0)),
                },
            ]
        );
        assert_eq!(
            layout.local_copies,
            vec![
                ScheduledTransfer {
                    from: DeviceId(0),
                    to: DeviceId(0),
                    src_byte_offset: 0,
                    dst_byte_offset: 0,
                    byte_len: 8,
                    route: Some((0, 0)),
                },
                ScheduledTransfer {
                    from: DeviceId(1),
                    to: DeviceId(1),
                    src_byte_offset: 20,
                    dst_byte_offset: 12,
                    byte_len: 28,
                    route: Some((1, 1)),
                },
                ScheduledTransfer {
                    from: DeviceId(2),
                    to: DeviceId(2),
                    src_byte_offset: 52,
                    dst_byte_offset: 44,
                    byte_len: 68,
                    route: Some((2, 2)),
                },
            ]
        );
    }

    #[test]
    fn dependency_edges_reorder_rows_and_retain_incoming_outgoing_and_opaque_edges() {
        let topology = nvptx_topology(3, &[CollectiveKind::PointToPoint]);
        let producer_id = PlanNodeId(90);
        let consumer_id = PlanNodeId(10);
        let independent_id = PlanNodeId(40);
        let external_in = PlanNodeId(1000);
        let external_out = PlanNodeId(1001);
        let mut consumer = row(
            consumer_id.0,
            CollectiveKind::PointToPoint,
            vec![DeviceId(1), DeviceId(2)],
            13,
        );
        consumer.producers = vec![producer_id, external_in];
        let independent = row(
            independent_id.0,
            CollectiveKind::PointToPoint,
            vec![DeviceId(0), DeviceId(2)],
            17,
        );
        let mut producer = row(
            producer_id.0,
            CollectiveKind::PointToPoint,
            vec![DeviceId(0), DeviceId(1)],
            19,
        );
        producer.consumers = vec![external_out];

        let schedule = compile_communication_schedule(
            &topology,
            &CommunicationPlan {
                rows: vec![consumer, independent, producer],
            },
        )
        .unwrap();
        let ids: Vec<_> = schedule.rows.iter().map(|row| row.id).collect();
        assert_eq!(ids, vec![independent_id, producer_id, consumer_id]);
        assert_eq!(schedule.rows[1].signal_to, vec![consumer_id, external_out]);
        assert_eq!(schedule.rows[2].wait_for, vec![producer_id, external_in]);
        assert_eq!(
            schedule.dependency_edges,
            vec![
                DependencyEdge {
                    from: producer_id,
                    to: consumer_id,
                },
                DependencyEdge {
                    from: external_in,
                    to: consumer_id,
                },
                DependencyEdge {
                    from: producer_id,
                    to: external_out,
                },
            ]
        );
    }

    #[test]
    fn opaque_round_trip_is_not_claimed_as_a_communication_subgraph_cycle() {
        let topology = nvptx_topology(2, &[CollectiveKind::PointToPoint]);
        let external = PlanNodeId(700);
        let mut transfer = row(
            41,
            CollectiveKind::PointToPoint,
            vec![DeviceId(0), DeviceId(1)],
            21,
        );
        transfer.producers.push(external);
        transfer.consumers.push(external);

        let schedule = compile_communication_schedule(
            &topology,
            &CommunicationPlan {
                rows: vec![transfer],
            },
        )
        .unwrap();
        assert_eq!(schedule.rows[0].wait_for, vec![external]);
        assert_eq!(schedule.rows[0].signal_to, vec![external]);
        assert_eq!(schedule.dependency_edges.len(), 2);
    }

    #[test]
    fn all_reduce_max_is_a_named_pre_device_rejection() {
        let topology = nvptx_topology(2, &[CollectiveKind::AllReduce]);
        let mut max = row(
            23,
            CollectiveKind::AllReduce,
            vec![DeviceId(0), DeviceId(1)],
            128,
        );
        max.op = RedOp::Max;

        assert_eq!(
            compile_communication_schedule(&topology, &CommunicationPlan { rows: vec![max] }),
            Err(CommunicationScheduleError::UnsupportedAllReduceMax {
                row: PlanNodeId(23)
            })
        );
    }

    #[test]
    fn unknown_participant_is_a_typed_rejection() {
        let topology = nvptx_topology(2, &[CollectiveKind::PointToPoint]);
        let plan = CommunicationPlan {
            rows: vec![row(
                24,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(9)],
                8,
            )],
        };
        assert_eq!(
            compile_communication_schedule(&topology, &plan),
            Err(CommunicationScheduleError::UnknownParticipant {
                row: PlanNodeId(24),
                device: DeviceId(9),
            })
        );
    }

    #[test]
    fn duplicate_participant_is_a_typed_rejection() {
        let topology = nvptx_topology(2, &[CollectiveKind::PointToPoint]);
        let plan = CommunicationPlan {
            rows: vec![row(
                25,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(0)],
                8,
            )],
        };
        assert_eq!(
            compile_communication_schedule(&topology, &plan),
            Err(CommunicationScheduleError::DuplicateParticipant {
                row: PlanNodeId(25),
                device: DeviceId(0),
            })
        );
    }

    #[test]
    fn invalid_participant_counts_are_typed_for_direct_and_collective_rows() {
        let topology = nvptx_topology(
            2,
            &[CollectiveKind::PointToPoint, CollectiveKind::AllGather],
        );
        for (id, kind, expected) in [
            (26, CollectiveKind::PointToPoint, "exactly 2"),
            (27, CollectiveKind::AllGather, "at least 2"),
        ] {
            let plan = CommunicationPlan {
                rows: vec![row(id, kind, vec![DeviceId(0)], 8)],
            };
            assert_eq!(
                compile_communication_schedule(&topology, &plan),
                Err(CommunicationScheduleError::InvalidParticipantCount {
                    row: PlanNodeId(id),
                    kind,
                    actual: 1,
                    expected,
                })
            );
        }
    }

    #[test]
    fn point_to_point_rejects_three_participants_even_with_complete_links() {
        let topology = nvptx_topology(3, &[CollectiveKind::PointToPoint]);
        let plan = CommunicationPlan {
            rows: vec![row(
                33,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(1), DeviceId(2)],
                8,
            )],
        };
        assert_eq!(
            compile_communication_schedule(&topology, &plan),
            Err(CommunicationScheduleError::InvalidParticipantCount {
                row: PlanNodeId(33),
                kind: CollectiveKind::PointToPoint,
                actual: 3,
                expected: "exactly 2",
            })
        );
    }

    #[test]
    fn invalid_topologies_are_rejected_before_schedule_admission() {
        let duplicate_device = DeviceSpec {
            id: DeviceId(0),
            backend: Backend::Nvptx,
            usable_bytes: 1 << 30,
        };
        let duplicate = Topology {
            devices: vec![duplicate_device.clone(), duplicate_device],
            links: vec![],
        };
        let unknown_link = Topology {
            devices: devices(&[Backend::Nvptx]),
            links: vec![LinkCapability {
                from: DeviceId(0),
                to: DeviceId(9),
                kind: CollectiveKind::PointToPoint,
            }],
        };
        for (topology, expected) in [
            (duplicate, TopologyError::DuplicateDevice(DeviceId(0))),
            (
                unknown_link,
                TopologyError::UnknownDeviceInLink(DeviceId(9)),
            ),
        ] {
            assert_eq!(
                compile_communication_schedule(&topology, &CommunicationPlan::default()),
                Err(CommunicationScheduleError::Topology(expected))
            );
        }
    }

    #[test]
    fn non_nvptx_participant_is_a_typed_rejection() {
        let topology = Topology {
            devices: devices(&[Backend::Nvptx, Backend::SpirvVulkan]),
            links: vec![LinkCapability {
                from: DeviceId(0),
                to: DeviceId(1),
                kind: CollectiveKind::PointToPoint,
            }],
        };
        let plan = CommunicationPlan {
            rows: vec![row(
                28,
                CollectiveKind::PointToPoint,
                vec![DeviceId(0), DeviceId(1)],
                8,
            )],
        };
        assert_eq!(
            compile_communication_schedule(&topology, &plan),
            Err(CommunicationScheduleError::NonNvptxParticipant {
                row: PlanNodeId(28),
                device: DeviceId(1),
                backend: Backend::SpirvVulkan,
            })
        );
    }

    #[test]
    fn valid_collective_groups_retain_sum_and_gather_kinds() {
        let topology = nvptx_topology(3, &[CollectiveKind::AllReduce, CollectiveKind::AllGather]);
        let plan = CommunicationPlan {
            rows: vec![
                row(
                    29,
                    CollectiveKind::AllReduce,
                    vec![DeviceId(0), DeviceId(1), DeviceId(2)],
                    96,
                ),
                row(
                    30,
                    CollectiveKind::AllGather,
                    vec![DeviceId(0), DeviceId(1), DeviceId(2)],
                    48,
                ),
            ],
        };
        let schedule = compile_communication_schedule(&topology, &plan).unwrap();
        assert_eq!(
            schedule.rows[0].communication,
            ScheduledCommunication::AllReduceSum
        );
        assert_eq!(
            schedule.rows[1].communication,
            ScheduledCommunication::AllGather
        );
    }

    #[test]
    fn cyclic_communication_rows_are_rejected_instead_of_partially_ordered() {
        let topology = nvptx_topology(2, &[CollectiveKind::PointToPoint]);
        let mut first = row(
            31,
            CollectiveKind::PointToPoint,
            vec![DeviceId(0), DeviceId(1)],
            7,
        );
        let mut second = row(
            32,
            CollectiveKind::PointToPoint,
            vec![DeviceId(1), DeviceId(0)],
            9,
        );
        first.producers.push(second.id);
        second.producers.push(first.id);

        assert_eq!(
            compile_communication_schedule(
                &topology,
                &CommunicationPlan {
                    rows: vec![first, second]
                }
            ),
            Err(CommunicationScheduleError::CommunicationSubgraphCycle {
                rows: vec![PlanNodeId(31), PlanNodeId(32)]
            })
        );
    }
}
