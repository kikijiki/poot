//! Single-host-thread ownership and replay of Card 408's per-device communication graphs.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;

use cudarc::driver::sys;
use poot_graph_plan::CollectiveKind;
use poot_graph_plan::multi_device::replay::{ReplayContract, StaticBufferId, StaticBufferKind};
use poot_graph_plan::multi_device::{
    ByteCategories, DeviceId, OwnerId, PlacementRow, PlacementSet, PlanNodeId, Topology,
    admit_per_device_capacity, placement_device_bytes, validate_placement, validate_placement_set,
};
use poot_target::Backend;

use super::{
    CommunicationSchedule, ScheduledCommunication, ScheduledCommunicationRow, ScheduledTransfer,
    compile_communication_schedule,
};
use crate::p2p::{CapturableEvent, PtxP2PTransport, RankByteBuffer};

pub use super::error::CapturedMultiDeviceError;
pub use super::segment::{
    CaptureBufferView, PtxComputeSegment, PtxDeviceSegment, SegmentCapture, SegmentProgram,
};

/// Host bytes for one [`ReplayContract`] buffer. Its device comes from the containing PTX capture;
/// content identity remains the contract's caller-supplied fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticBufferUpload {
    pub id: StaticBufferId,
    pub bytes: Vec<u8>,
}

/// PTX-owned resources for one topology device capture. This is execution input, not a second Card 360
/// plan: topology, placement, communication, accounting, and replay remain their original shared types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PtxDeviceCapture {
    pub device: DeviceId,
    pub graph_identity: String,
    pub static_uploads: Vec<StaticBufferUpload>,
}

/// Source and destination buffer ids used by one communication row on one participant. Point-to-point
/// needs only the source on its first participant and destination on its second; all-to-all needs both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCommunicationBuffers {
    pub device: DeviceId,
    pub source: Option<StaticBufferId>,
    pub destination: Option<StaticBufferId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PtxCommunicationBufferBinding {
    pub row: PlanNodeId,
    pub devices: Vec<DeviceCommunicationBuffers>,
}

/// Counts calls and captured transfer nodes produced by the real PTX owner. Setup work is carried into
/// the first replay report; an unchanged second replay has zero allocations, static uploads, graph
/// uploads, captured-node rebuilds, and carrier rebuilds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PtxMultiDeviceReplayCounters {
    /// Resident [`ReplayContract`] buffer allocations (not driver-internal graph/event objects).
    pub allocations: u64,
    /// Static-buffer upload operations.
    pub uploads: u64,
    pub carrier_rebuilds: u64,
    pub graph_instantiations: u64,
    pub graph_uploads: u64,
    pub captured_transfer_nodes: u64,
    pub graph_launches: u64,
    pub transfers: u64,
    /// Card 360's declared dispatch count replayed by the per-device graph set. CUDA host calls are
    /// reported separately as `graph_launches`.
    pub planned_dispatches: u64,
}

impl PtxMultiDeviceReplayCounters {
    fn add(&mut self, rhs: Self) {
        self.allocations += rhs.allocations;
        self.uploads += rhs.uploads;
        self.carrier_rebuilds += rhs.carrier_rebuilds;
        self.graph_instantiations += rhs.graph_instantiations;
        self.graph_uploads += rhs.graph_uploads;
        self.captured_transfer_nodes += rhs.captured_transfer_nodes;
        self.graph_launches += rhs.graph_launches;
        self.transfers += rhs.transfers;
        self.planned_dispatches += rhs.planned_dispatches;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtxMultiDeviceReplayReport {
    pub replay_index: u64,
    pub delta: PtxMultiDeviceReplayCounters,
    pub total: PtxMultiDeviceReplayCounters,
}

#[derive(Clone)]
struct PreparedBuffer {
    id: StaticBufferId,
    device: DeviceId,
    rank: usize,
    kind: StaticBufferKind,
    bytes: Vec<u8>,
}

pub(super) struct CaptureInputs {
    pub(super) topology: Topology,
    pub(super) placements: Vec<PlacementRow>,
    pub(super) expected_owners: Vec<OwnerId>,
    pub(super) device_usage: HashMap<DeviceId, ByteCategories>,
    pub(super) replay: ReplayContract,
    pub(super) device_captures: Vec<PtxDeviceCapture>,
    pub(super) communication_buffers: Vec<PtxCommunicationBufferBinding>,
}

pub(super) struct AdmittedCapture {
    inputs: CaptureInputs,
    placement_set: PlacementSet,
    schedule: CommunicationSchedule,
    pub(super) execution_order: Vec<PlanNodeId>,
    pub(super) rank_by_device: HashMap<DeviceId, usize>,
    buffers: Vec<PreparedBuffer>,
    row_bindings: HashMap<PlanNodeId, HashMap<DeviceId, DeviceCommunicationBuffers>>,
}

/// Private recorder segmentation, not a second semantic plan. Card 360 still owns every node, edge,
/// route, and byte range; this only makes the CUDA event epoch explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CapturePhase {
    WaitForNode {
        producer: PlanNodeId,
        consumer_ranks: Vec<usize>,
    },
    RemoteTransfer {
        row: PlanNodeId,
        transfer: ScheduledTransfer,
    },
    LocalCopy {
        row: PlanNodeId,
        transfer: ScheduledTransfer,
    },
    RecordNode {
        node: PlanNodeId,
        ranks: Vec<usize>,
    },
    RecordIdentity {
        ranks: Vec<usize>,
    },
}

/// Perform every model-free Card 360 check needed by the PTX owner, plus the ADR-0099 segment
/// admission. This function has no CUDA side effects and is deliberately called before
/// `PtxP2PTransport::new`.
pub(super) fn admit_captured_multi_device_plan(
    inputs: CaptureInputs,
    segments: &[PtxDeviceSegment],
) -> Result<AdmittedCapture, CapturedMultiDeviceError> {
    let schedule = compile_communication_schedule(&inputs.topology, &inputs.replay.communication)?;
    if inputs.topology.devices.len() < 2 {
        return Err(CapturedMultiDeviceError::InvalidTopologyDeviceCount {
            actual: inputs.topology.devices.len(),
        });
    }
    let rank_by_device: HashMap<_, _> = inputs
        .topology
        .devices
        .iter()
        .enumerate()
        .map(|(rank, device)| (device.id, rank))
        .collect();

    for device in &inputs.topology.devices {
        if device.backend != Backend::Nvptx {
            return Err(CapturedMultiDeviceError::NonNvptxDevice {
                device: device.id,
                backend: device.backend,
            });
        }
    }
    // Set-level owner assignment first: every expected owner exactly once, no unexpected owner.
    // Row-level defects still report their index through the per-row `validate_placement` loop.
    let placement_set = validate_placement_set(&inputs.placements, &inputs.expected_owners)
        .map_err(|source| CapturedMultiDeviceError::PlacementSet {
            source: Box::new(source),
        })?;
    let mut required_placement_bytes = HashMap::<DeviceId, u64>::new();
    for (index, placement) in placement_set.rows().iter().enumerate() {
        validate_placement(placement).map_err(|source| CapturedMultiDeviceError::Placement {
            index,
            source: Box::new(source),
        })?;
        let devices: Vec<_> = match placement {
            PlacementRow::Replicated { residents, .. } => residents.clone(),
            PlacementRow::Partitioned { ranges, .. } => {
                ranges.iter().map(|(device, _)| *device).collect()
            }
        };
        for device in devices {
            require_known_device(&rank_by_device, "placement", device)?;
        }
        for (device, bytes) in placement_device_bytes(placement) {
            let total = required_placement_bytes.entry(device).or_default();
            *total = total
                .checked_add(bytes)
                .ok_or(CapturedMultiDeviceError::PlacementBytesOverflow { device })?;
        }
    }
    for &device in inputs.device_usage.keys() {
        require_known_device(&rank_by_device, "device accounting", device)?;
    }
    for (&device, &required) in &required_placement_bytes {
        let accounted = inputs
            .device_usage
            .get(&device)
            .map_or(0, |usage| usage.source_or_carrier);
        if accounted < required {
            return Err(CapturedMultiDeviceError::PlacementAccountingUnderflow {
                device,
                accounted,
                required,
            });
        }
    }
    admit_per_device_capacity(&inputs.topology.devices, &inputs.device_usage)?;

    let mut graph_by_device = HashMap::with_capacity(inputs.device_captures.len());
    for graph in &inputs.device_captures {
        require_known_device(&rank_by_device, "captured graph", graph.device)?;
        if graph_by_device.insert(graph.device, graph).is_some() {
            return Err(CapturedMultiDeviceError::DuplicateDeviceGraph {
                device: graph.device,
            });
        }
    }
    for device in &inputs.topology.devices {
        if !graph_by_device.contains_key(&device.id) {
            return Err(CapturedMultiDeviceError::MissingDeviceGraph { device: device.id });
        }
    }

    let mut contract_buffers = HashMap::with_capacity(inputs.replay.buffers.len());
    for buffer in &inputs.replay.buffers {
        if contract_buffers.insert(buffer.id.clone(), buffer).is_some() {
            return Err(CapturedMultiDeviceError::DuplicateContractBuffer {
                id: buffer.id.clone(),
            });
        }
    }
    let mut seen_uploads = HashSet::with_capacity(contract_buffers.len());
    let mut buffers = Vec::with_capacity(contract_buffers.len());
    for device in &inputs.topology.devices {
        let graph = graph_by_device[&device.id];
        for upload in &graph.static_uploads {
            if !seen_uploads.insert(upload.id.clone()) {
                return Err(CapturedMultiDeviceError::DuplicateBufferUpload {
                    id: upload.id.clone(),
                });
            }
            let contract = contract_buffers.get(&upload.id).ok_or_else(|| {
                CapturedMultiDeviceError::UnknownBufferUpload {
                    id: upload.id.clone(),
                }
            })?;
            if contract.bytes != upload.bytes.len() as u64 {
                return Err(CapturedMultiDeviceError::BufferSizeMismatch {
                    id: upload.id.clone(),
                    declared: contract.bytes,
                    actual: upload.bytes.len(),
                });
            }
            buffers.push(PreparedBuffer {
                id: upload.id.clone(),
                device: device.id,
                rank: rank_by_device[&device.id],
                kind: contract.kind,
                bytes: upload.bytes.clone(),
            });
        }
    }
    for id in contract_buffers.keys() {
        if !seen_uploads.contains(id) {
            return Err(CapturedMultiDeviceError::MissingBufferUpload { id: id.clone() });
        }
    }

    let schedule_rows: HashMap<_, _> = schedule.rows.iter().map(|row| (row.id, row)).collect();
    let mut row_bindings = HashMap::with_capacity(inputs.communication_buffers.len());
    for binding in &inputs.communication_buffers {
        if !schedule_rows.contains_key(&binding.row) {
            return Err(CapturedMultiDeviceError::UnknownCommunicationBinding { row: binding.row });
        }
        let mut devices = HashMap::with_capacity(binding.devices.len());
        for device in &binding.devices {
            if devices.insert(device.device, device.clone()).is_some() {
                return Err(CapturedMultiDeviceError::DuplicateCommunicationDevice {
                    row: binding.row,
                    device: device.device,
                });
            }
        }
        if row_bindings.insert(binding.row, devices).is_some() {
            return Err(CapturedMultiDeviceError::DuplicateCommunicationBinding {
                row: binding.row,
            });
        }
    }
    let buffer_owners: HashMap<_, _> = buffers
        .iter()
        .map(|buffer| (buffer.id.clone(), (buffer.device, buffer.bytes.len())))
        .collect();
    for row in &schedule.rows {
        let bindings = row_bindings
            .get(&row.id)
            .ok_or(CapturedMultiDeviceError::MissingCommunicationBinding { row: row.id })?;
        validate_row_bindings(row, bindings, &buffer_owners)?;
    }
    let execution_order = captured_execution_order(&schedule)?;

    // ADR-0099 item 1: a segment may bind only an opaque node of THIS schedule, on a device the
    // topology declares, and every program it may launch must be well formed. Communication rows
    // carry the executor's own transfers, so a segment there would be a second owner of the row.
    for segment in segments {
        require_known_device(&rank_by_device, "compute segment", segment.device)?;
        if schedule.rows.iter().any(|row| row.id == segment.node) {
            return Err(CapturedMultiDeviceError::SegmentOnCommunicationRow { node: segment.node });
        }
        if !execution_order.contains(&segment.node) {
            return Err(CapturedMultiDeviceError::SegmentUnknownNode { node: segment.node });
        }
        for (index, program) in segment.programs.iter().enumerate() {
            if program.elem_bytes == 0 {
                return Err(CapturedMultiDeviceError::SegmentProgramElementBytes {
                    node: segment.node,
                    program: index,
                    elem_bytes: program.elem_bytes,
                });
            }
            if program.block.contains(&0) {
                return Err(CapturedMultiDeviceError::SegmentProgramBlockShape {
                    node: segment.node,
                    program: index,
                    block: program.block,
                });
            }
        }
    }

    Ok(AdmittedCapture {
        inputs,
        placement_set,
        schedule,
        execution_order,
        rank_by_device,
        buffers,
        row_bindings,
    })
}

fn captured_execution_order(
    schedule: &CommunicationSchedule,
) -> Result<Vec<PlanNodeId>, CapturedMultiDeviceError> {
    let mut nodes: Vec<_> = schedule.rows.iter().map(|row| row.id).collect();
    let mut seen: HashSet<_> = nodes.iter().copied().collect();
    for edge in &schedule.dependency_edges {
        for node in [edge.from, edge.to] {
            if seen.insert(node) {
                nodes.push(node);
            }
        }
    }
    let indices: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let mut successors = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    let mut seen_edges = HashSet::new();
    for edge in &schedule.dependency_edges {
        let from = indices[&edge.from];
        let to = indices[&edge.to];
        if seen_edges.insert((from, to)) {
            successors[from].push(to);
            indegree[to] += 1;
        }
    }
    let mut emitted = vec![false; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    while order.len() < nodes.len() {
        let Some(next) = (0..nodes.len()).find(|&index| !emitted[index] && indegree[index] == 0)
        else {
            return Err(CapturedMultiDeviceError::DependencyCycle {
                nodes: nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &node)| (!emitted[index]).then_some(node))
                    .collect(),
            });
        };
        emitted[next] = true;
        order.push(nodes[next]);
        for &successor in &successors[next] {
            indegree[successor] -= 1;
        }
    }
    Ok(order)
}

fn capture_phase_plan(admitted: &AdmittedCapture) -> Vec<CapturePhase> {
    let world_size = admitted.inputs.topology.devices.len();
    let all_ranks: Vec<_> = (0..world_size).collect();
    let mut phases = Vec::new();
    for &node in &admitted.execution_order {
        if let Some(row) = admitted
            .schedule
            .rows
            .iter()
            .find(|candidate| candidate.id == node)
        {
            let participant_ranks: Vec<_> = row
                .participants
                .iter()
                .map(|device| admitted.rank_by_device[device])
                .collect();
            for &producer in &row.wait_for {
                phases.push(CapturePhase::WaitForNode {
                    producer,
                    consumer_ranks: participant_ranks.clone(),
                });
            }
            match &row.communication {
                ScheduledCommunication::PointToPoint(transfer) => {
                    phases.push(CapturePhase::RemoteTransfer {
                        row: row.id,
                        transfer: *transfer,
                    });
                }
                ScheduledCommunication::AllToAll(layout) => {
                    phases.extend(layout.remote_transfers.iter().copied().map(|transfer| {
                        CapturePhase::RemoteTransfer {
                            row: row.id,
                            transfer,
                        }
                    }));
                    phases.extend(layout.local_copies.iter().copied().map(|transfer| {
                        CapturePhase::LocalCopy {
                            row: row.id,
                            transfer,
                        }
                    }));
                }
                ScheduledCommunication::AllReduceSum | ScheduledCommunication::AllGather => {
                    unreachable!()
                }
            }
            phases.push(CapturePhase::RecordNode {
                node: row.id,
                ranks: participant_ranks,
            });
        } else {
            for edge in admitted
                .schedule
                .dependency_edges
                .iter()
                .filter(|edge| edge.to == node)
            {
                phases.push(CapturePhase::WaitForNode {
                    producer: edge.from,
                    consumer_ranks: all_ranks.clone(),
                });
            }
            phases.push(CapturePhase::RecordNode {
                node,
                ranks: all_ranks.clone(),
            });
        }
    }
    phases.push(CapturePhase::RecordIdentity { ranks: all_ranks });
    phases
}

fn remote_transfer_submission_ranks(
    admitted: &AdmittedCapture,
    transfer: &ScheduledTransfer,
) -> [usize; 2] {
    [
        admitted.rank_by_device[&transfer.from],
        admitted.rank_by_device[&transfer.to],
    ]
}

fn require_known_device(
    ranks: &HashMap<DeviceId, usize>,
    role: &'static str,
    device: DeviceId,
) -> Result<(), CapturedMultiDeviceError> {
    if ranks.contains_key(&device) {
        Ok(())
    } else {
        Err(CapturedMultiDeviceError::UnknownDevice { role, device })
    }
}

fn validate_row_bindings(
    row: &ScheduledCommunicationRow,
    bindings: &HashMap<DeviceId, DeviceCommunicationBuffers>,
    buffer_owners: &HashMap<StaticBufferId, (DeviceId, usize)>,
) -> Result<(), CapturedMultiDeviceError> {
    for &device in bindings.keys() {
        if !row.participants.contains(&device) {
            return Err(CapturedMultiDeviceError::UnexpectedCommunicationDevice {
                row: row.id,
                device,
            });
        }
    }
    match &row.communication {
        ScheduledCommunication::PointToPoint(transfer) => {
            validate_buffer_role(
                row.id,
                transfer.from,
                "source",
                bindings,
                buffer_owners,
                transfer.byte_len,
                true,
            )?;
            validate_buffer_role(
                row.id,
                transfer.to,
                "destination",
                bindings,
                buffer_owners,
                transfer.byte_len,
                false,
            )?;
            reject_alias(row.id, transfer.from, transfer.to, bindings)
        }
        ScheduledCommunication::AllToAll(layout) => {
            for extent in &layout.source_extents {
                validate_buffer_role(
                    row.id,
                    extent.device,
                    "source",
                    bindings,
                    buffer_owners,
                    extent.byte_len,
                    true,
                )?;
            }
            for extent in &layout.destination_extents {
                validate_buffer_role(
                    row.id,
                    extent.device,
                    "destination",
                    bindings,
                    buffer_owners,
                    extent.byte_len,
                    false,
                )?;
                reject_alias(row.id, extent.device, extent.device, bindings)?;
            }
            Ok(())
        }
        ScheduledCommunication::AllReduceSum => {
            Err(CapturedMultiDeviceError::UnsupportedCommunicationKind {
                row: row.id,
                kind: CollectiveKind::AllReduce,
            })
        }
        ScheduledCommunication::AllGather => {
            Err(CapturedMultiDeviceError::UnsupportedCommunicationKind {
                row: row.id,
                kind: CollectiveKind::AllGather,
            })
        }
    }
}

fn validate_buffer_role(
    row: PlanNodeId,
    device: DeviceId,
    role: &'static str,
    bindings: &HashMap<DeviceId, DeviceCommunicationBuffers>,
    buffer_owners: &HashMap<StaticBufferId, (DeviceId, usize)>,
    required: usize,
    source: bool,
) -> Result<(), CapturedMultiDeviceError> {
    let binding = bindings
        .get(&device)
        .ok_or(CapturedMultiDeviceError::MissingCommunicationBuffer { row, device, role })?;
    let id = if source {
        binding.source.as_ref()
    } else {
        binding.destination.as_ref()
    }
    .ok_or(CapturedMultiDeviceError::MissingCommunicationBuffer { row, device, role })?;
    let &(actual_device, actual) = buffer_owners
        .get(id)
        .ok_or_else(|| CapturedMultiDeviceError::UnknownBufferUpload { id: id.clone() })?;
    if actual_device != device {
        return Err(CapturedMultiDeviceError::CommunicationBufferDevice {
            row,
            role,
            id: id.clone(),
            actual: actual_device,
            expected: device,
        });
    }
    if actual < required {
        return Err(CapturedMultiDeviceError::CommunicationBufferExtent {
            row,
            role,
            id: id.clone(),
            actual,
            required,
        });
    }
    Ok(())
}

fn reject_alias(
    row: PlanNodeId,
    source_device: DeviceId,
    destination_device: DeviceId,
    bindings: &HashMap<DeviceId, DeviceCommunicationBuffers>,
) -> Result<(), CapturedMultiDeviceError> {
    let source = bindings
        .get(&source_device)
        .and_then(|binding| binding.source.as_ref());
    let destination = bindings
        .get(&destination_device)
        .and_then(|binding| binding.destination.as_ref());
    if let (Some(source), Some(destination)) = (source, destination)
        && source == destination
    {
        return Err(CapturedMultiDeviceError::AliasedCommunicationBuffer {
            row,
            id: source.clone(),
        });
    }
    Ok(())
}

pub(super) struct ResidentBuffer {
    pub(super) device: DeviceId,
    pub(super) buffer: RankByteBuffer,
}

/// One host-thread owner for all rank contexts, stable buffers, and distinct per-device CUDA graph
/// segment sequences. `Rc` phantom state makes moving it to another thread a compile-time error.
pub struct PtxCapturedMultiDeviceExecutor {
    pub(super) transport: PtxP2PTransport,
    topology: Topology,
    placements: PlacementSet,
    device_usage: HashMap<DeviceId, ByteCategories>,
    pub(super) contract: ReplayContract,
    #[cfg(test)]
    graph_identities: Vec<(DeviceId, String)>,
    pub(super) buffers: HashMap<StaticBufferId, ResidentBuffer>,
    transfers_per_replay: u64,
    pending: PtxMultiDeviceReplayCounters,
    total: PtxMultiDeviceReplayCounters,
    replay_index: u64,
    _single_thread: PhantomData<Rc<()>>,
}

impl PtxCapturedMultiDeviceExecutor {
    /// Admit Card 360's shared values unchanged, materialize the PTX-only buffer bindings, and capture
    /// one CUDA graph per topology device. No aggregate replacement "multi-device plan" is introduced.
    /// `expected_owners` is the caller's full owner list: `placements` must name each of them exactly
    /// once before any device work starts.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        topology: Topology,
        placements: Vec<PlacementRow>,
        expected_owners: Vec<OwnerId>,
        device_usage: HashMap<DeviceId, ByteCategories>,
        replay: ReplayContract,
        device_captures: Vec<PtxDeviceCapture>,
        communication_buffers: Vec<PtxCommunicationBufferBinding>,
    ) -> Result<Self, CapturedMultiDeviceError> {
        let mut counters = PtxMultiDeviceReplayCounters::default();
        Self::capture_with_counters(
            topology,
            placements,
            expected_owners,
            device_usage,
            replay,
            device_captures,
            communication_buffers,
            &mut counters,
        )
    }

    /// Capture while exposing setup calls even when capture fails. Admission errors, including capability
    /// and `AllReduce::Max`, leave every counter at zero because admission precedes transport creation.
    /// This is `Self::capture_with_segments` with no segment, which is unchanged pre-ADR-0099
    /// behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_counters(
        topology: Topology,
        placements: Vec<PlacementRow>,
        expected_owners: Vec<OwnerId>,
        device_usage: HashMap<DeviceId, ByteCategories>,
        replay: ReplayContract,
        device_captures: Vec<PtxDeviceCapture>,
        communication_buffers: Vec<PtxCommunicationBufferBinding>,
        counters: &mut PtxMultiDeviceReplayCounters,
    ) -> Result<Self, CapturedMultiDeviceError> {
        Self::capture_inputs_with_counters(
            CaptureInputs {
                topology,
                placements,
                expected_owners,
                device_usage,
                replay,
                device_captures,
                communication_buffers,
            },
            Vec::new(),
            counters,
        )
    }

    pub(super) fn capture_inputs_with_counters(
        inputs: CaptureInputs,
        mut segments: Vec<PtxDeviceSegment>,
        counters: &mut PtxMultiDeviceReplayCounters,
    ) -> Result<Self, CapturedMultiDeviceError> {
        *counters = PtxMultiDeviceReplayCounters::default();
        let admitted = admit_captured_multi_device_plan(inputs, &segments)?;
        let world_size = admitted.inputs.topology.devices.len();
        let mut transport = PtxP2PTransport::new(world_size)?;
        transport.prepare_capturable_peer_access(&declared_peer_table(&admitted))?;
        let mut resident = HashMap::with_capacity(admitted.buffers.len());
        for prepared in &admitted.buffers {
            let buffer = transport.upload_rank_bytes(prepared.rank, &prepared.bytes)?;
            counters.allocations += 1;
            counters.uploads += 1;
            if prepared.kind == StaticBufferKind::Carrier {
                counters.carrier_rebuilds += 1;
            }
            resident.insert(
                prepared.id.clone(),
                ResidentBuffer {
                    device: prepared.device,
                    buffer,
                },
            );
        }

        let transfer_counts =
            transfer_event_counts(world_size, &admitted.schedule, &admitted.rank_by_device);
        transport.prepare_capturable_transfer_events(&transfer_counts)?;
        let mut row_events = HashMap::new();
        for row in &admitted.schedule.rows {
            for &device in &row.participants {
                let rank = admitted.rank_by_device[&device];
                row_events.insert((row.id, rank), transport.create_capturable_event(rank)?);
            }
        }
        let row_ids: HashSet<_> = admitted.schedule.rows.iter().map(|row| row.id).collect();
        let opaque_nodes: Vec<_> = admitted
            .execution_order
            .iter()
            .copied()
            .filter(|node| !row_ids.contains(node))
            .collect();
        let mut opaque_events = HashMap::new();
        for node in opaque_nodes {
            for rank in 0..world_size {
                opaque_events.insert((node, rank), transport.create_capturable_event(rank)?);
            }
        }
        let mut identity_events = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            identity_events.push(transport.create_capturable_event(rank)?);
        }

        // ADR-0099 item 1: a segment's programs must be resident in its rank's context before any
        // stream enters capture mode, so they load here, never inside a phase.
        let mut segment_functions: Vec<Vec<sys::CUfunction>> = Vec::with_capacity(segments.len());
        for segment in &segments {
            let rank = admitted.rank_by_device[&segment.device];
            let mut functions = Vec::with_capacity(segment.programs.len());
            for program in &segment.programs {
                functions.push(transport.load_rank_program(rank, &program.ptx, &program.entry)?);
            }
            segment_functions.push(functions);
        }

        let phases = capture_phase_plan(&admitted);
        let transfers_per_replay = capture_schedule(
            &mut transport,
            &admitted,
            &mut segments,
            &segment_functions,
            &phases,
            &resident,
            &row_events,
            &opaque_events,
            &identity_events,
        )?;
        counters.captured_transfer_nodes = transfers_per_replay;
        counters.graph_instantiations = transport.captured_graph_count() as u64;
        counters.graph_uploads = transport.upload_captured_graphs()? as u64;

        #[cfg(test)]
        let graph_by_device: HashMap<_, _> = admitted
            .inputs
            .device_captures
            .iter()
            .map(|graph| (graph.device, graph.graph_identity.clone()))
            .collect();
        #[cfg(test)]
        let graph_identities = admitted
            .inputs
            .topology
            .devices
            .iter()
            .map(|device| (device.id, graph_by_device[&device.id].clone()))
            .collect();
        let pending = *counters;
        Ok(Self {
            transport,
            topology: admitted.inputs.topology,
            placements: admitted.placement_set,
            device_usage: admitted.inputs.device_usage,
            contract: admitted.inputs.replay,
            #[cfg(test)]
            graph_identities,
            buffers: resident,
            transfers_per_replay,
            pending,
            total: PtxMultiDeviceReplayCounters::default(),
            replay_index: 0,
            _single_thread: PhantomData,
        })
    }

    pub fn replay(
        &mut self,
        contract: &ReplayContract,
    ) -> Result<PtxMultiDeviceReplayReport, CapturedMultiDeviceError> {
        if contract != &self.contract {
            return Err(CapturedMultiDeviceError::ReplayContractChanged);
        }
        let graph_launches = self.transport.launch_captured_graphs()? as u64;
        self.transport.barrier()?;
        Ok(record_successful_replay(
            &mut self.pending,
            &mut self.total,
            &mut self.replay_index,
            graph_launches,
            self.transfers_per_replay,
            self.contract.dispatch_count as u64,
        ))
    }

    pub fn download(&self, id: &StaticBufferId) -> Result<Vec<u8>, CapturedMultiDeviceError> {
        let resident = self
            .buffers
            .get(id)
            .ok_or_else(|| CapturedMultiDeviceError::UnknownBufferUpload { id: id.clone() })?;
        Ok(self.transport.download_rank_bytes(&resident.buffer)?)
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn placements(&self) -> &[PlacementRow] {
        self.placements.rows()
    }

    pub fn device_usage(&self) -> &HashMap<DeviceId, ByteCategories> {
        &self.device_usage
    }

    #[cfg(test)]
    pub(crate) fn device_graph_identities(&self) -> &[(DeviceId, String)] {
        &self.graph_identities
    }

    pub fn counters(&self) -> PtxMultiDeviceReplayCounters {
        let mut counters = self.total;
        counters.add(self.pending);
        counters
    }
}

fn record_successful_replay(
    pending: &mut PtxMultiDeviceReplayCounters,
    total: &mut PtxMultiDeviceReplayCounters,
    replay_index: &mut u64,
    graph_launches: u64,
    transfers: u64,
    planned_dispatches: u64,
) -> PtxMultiDeviceReplayReport {
    let mut delta = std::mem::take(pending);
    delta.graph_launches = graph_launches;
    delta.transfers = transfers;
    delta.planned_dispatches = planned_dispatches;
    total.add(delta);
    *replay_index += 1;
    PtxMultiDeviceReplayReport {
        replay_index: *replay_index,
        delta,
        total: *total,
    }
}

fn declared_peer_table(admitted: &AdmittedCapture) -> Vec<Vec<bool>> {
    let world_size = admitted.inputs.topology.devices.len();
    let mut declared = vec![vec![false; world_size]; world_size];
    for link in &admitted.inputs.topology.links {
        if matches!(
            link.kind,
            CollectiveKind::PointToPoint | CollectiveKind::AllToAll
        ) {
            let from = admitted.rank_by_device[&link.from];
            let to = admitted.rank_by_device[&link.to];
            declared[from][to] = true;
        }
    }
    declared
}

fn transfer_event_counts(
    world_size: usize,
    schedule: &CommunicationSchedule,
    ranks: &HashMap<DeviceId, usize>,
) -> Vec<usize> {
    let mut counts = vec![0usize; world_size];
    for row in &schedule.rows {
        match &row.communication {
            ScheduledCommunication::PointToPoint(transfer) => counts[ranks[&transfer.from]] += 1,
            ScheduledCommunication::AllToAll(layout) => {
                for transfer in &layout.remote_transfers {
                    counts[ranks[&transfer.from]] += 1;
                }
            }
            ScheduledCommunication::AllReduceSum | ScheduledCommunication::AllGather => {}
        }
    }
    counts
}

#[allow(clippy::too_many_arguments)]
fn capture_schedule(
    transport: &mut PtxP2PTransport,
    admitted: &AdmittedCapture,
    segments: &mut [PtxDeviceSegment],
    segment_functions: &[Vec<sys::CUfunction>],
    phases: &[CapturePhase],
    resident: &HashMap<StaticBufferId, ResidentBuffer>,
    row_events: &HashMap<(PlanNodeId, usize), CapturableEvent>,
    opaque_events: &HashMap<(PlanNodeId, usize), CapturableEvent>,
    identity_events: &[CapturableEvent],
) -> Result<u64, CapturedMultiDeviceError> {
    let mut transfers = 0u64;
    for phase in phases {
        match phase {
            CapturePhase::WaitForNode {
                producer,
                consumer_ranks,
            } => {
                capture_one_phase(transport, consumer_ranks, |transport| {
                    for &consumer_rank in consumer_ranks {
                        wait_for_node(
                            transport,
                            admitted,
                            *producer,
                            consumer_rank,
                            row_events,
                            opaque_events,
                        )?;
                    }
                    Ok(())
                })?;
            }
            CapturePhase::RemoteTransfer { row, transfer } => {
                let bindings = &admitted.row_bindings[row];
                let ranks = remote_transfer_submission_ranks(admitted, transfer);
                capture_one_phase(transport, &ranks, |transport| {
                    capture_remote_transfer(transport, transfer, bindings, resident)
                })?;
                transfers += 1;
            }
            CapturePhase::LocalCopy { row, transfer } => {
                let bindings = &admitted.row_bindings[row];
                let source = bound_buffer(transfer.from, true, bindings, resident);
                let destination = bound_buffer(transfer.to, false, bindings, resident);
                let ranks = [source.buffer.rank()];
                capture_one_phase(transport, &ranks, |transport| {
                    transport.copy_bytes_capturable_local(
                        source.buffer.rank(),
                        source.buffer.ptr(),
                        source.buffer.byte_len(),
                        transfer.src_byte_offset,
                        destination.buffer.ptr(),
                        destination.buffer.byte_len(),
                        transfer.dst_byte_offset,
                        transfer.byte_len,
                    )?;
                    Ok(())
                })?;
                transfers += 1;
            }
            CapturePhase::RecordNode { node, ranks } => {
                capture_one_phase(transport, ranks, |transport| {
                    // ADR-0099 item 1: a segment for this opaque node records here, inside the
                    // phase, after the waits above and before the node's event below. The event
                    // still lands on every rank, because a consumer waits on the producer's event
                    // on each of them.
                    for (index, binding) in segments.iter_mut().enumerate() {
                        if binding.node != *node {
                            continue;
                        }
                        let Some(&rank) = admitted.rank_by_device.get(&binding.device) else {
                            continue;
                        };
                        if !ranks.contains(&rank) {
                            continue;
                        }
                        let PtxDeviceSegment {
                            device,
                            node: bound,
                            programs,
                            segment,
                        } = binding;
                        let mut capture = SegmentCapture::new(
                            transport,
                            &admitted.inputs.replay,
                            rank,
                            *device,
                            *bound,
                            programs,
                            &segment_functions[index],
                            resident,
                        );
                        segment.record(&mut capture)?;
                    }
                    for &rank in ranks {
                        let event = if admitted
                            .schedule
                            .rows
                            .iter()
                            .any(|candidate| candidate.id == *node)
                        {
                            row_events[&(*node, rank)].clone()
                        } else {
                            opaque_events[&(*node, rank)].clone()
                        };
                        transport.record_plan_event(rank, event)?;
                    }
                    Ok(())
                })?;
            }
            CapturePhase::RecordIdentity { ranks } => {
                capture_one_phase(transport, ranks, |transport| {
                    for &rank in ranks {
                        transport.record_plan_event(rank, identity_events[rank].clone())?;
                    }
                    Ok(())
                })?;
            }
        }
    }
    Ok(transfers)
}

fn capture_one_phase(
    transport: &mut PtxP2PTransport,
    ranks: &[usize],
    record: impl FnOnce(&PtxP2PTransport) -> Result<(), CapturedMultiDeviceError>,
) -> Result<usize, CapturedMultiDeviceError> {
    transport.begin_capturable_phase(ranks)?;
    if let Err(error) = record(transport) {
        transport.abort_capturable_phase();
        return Err(error);
    }
    Ok(transport.end_capturable_phase()?)
}

fn wait_for_node(
    transport: &PtxP2PTransport,
    admitted: &AdmittedCapture,
    producer: PlanNodeId,
    consumer_rank: usize,
    row_events: &HashMap<(PlanNodeId, usize), CapturableEvent>,
    opaque_events: &HashMap<(PlanNodeId, usize), CapturableEvent>,
) -> Result<(), CapturedMultiDeviceError> {
    if let Some(row) = admitted.schedule.rows.iter().find(|row| row.id == producer) {
        for &producer_device in &row.participants {
            let producer_rank = admitted.rank_by_device[&producer_device];
            transport.wait_plan_event(
                consumer_rank,
                row_events[&(producer, producer_rank)].clone(),
            )?;
        }
    } else {
        for producer_rank in 0..transport.world_size() {
            transport.wait_plan_event(
                consumer_rank,
                opaque_events[&(producer, producer_rank)].clone(),
            )?;
        }
    }
    Ok(())
}

fn capture_remote_transfer(
    transport: &PtxP2PTransport,
    transfer: &ScheduledTransfer,
    bindings: &HashMap<DeviceId, DeviceCommunicationBuffers>,
    resident: &HashMap<StaticBufferId, ResidentBuffer>,
) -> Result<(), CapturedMultiDeviceError> {
    let source = bound_buffer(transfer.from, true, bindings, resident);
    let destination = bound_buffer(transfer.to, false, bindings, resident);
    transport.move_bytes_capturable(
        source.buffer.rank(),
        source.buffer.ptr(),
        source.buffer.byte_len(),
        transfer.src_byte_offset,
        destination.buffer.rank(),
        destination.buffer.ptr(),
        destination.buffer.byte_len(),
        transfer.dst_byte_offset,
        transfer.byte_len,
    )?;
    Ok(())
}

fn bound_buffer<'a>(
    device: DeviceId,
    source: bool,
    bindings: &HashMap<DeviceId, DeviceCommunicationBuffers>,
    resident: &'a HashMap<StaticBufferId, ResidentBuffer>,
) -> &'a ResidentBuffer {
    let binding = &bindings[&device];
    let id = if source {
        binding.source.as_ref().unwrap()
    } else {
        binding.destination.as_ref().unwrap()
    };
    let buffer = &resident[id];
    debug_assert_eq!(buffer.device, device);
    buffer
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::CommunicationScheduleError;
    use poot_graph_ir::RedOp;
    use poot_graph_plan::multi_device::replay::StaticBuffer;
    use poot_graph_plan::multi_device::{
        CommunicationPlan, CommunicationRow, DeviceSpec, HostOwner, LinkCapability, OwnerId,
        PlacementError,
    };

    use super::*;

    pub(crate) fn topology(kinds: &[CollectiveKind]) -> Topology {
        let devices = vec![
            DeviceSpec {
                id: DeviceId(41),
                backend: Backend::Nvptx,
                usable_bytes: 1 << 20,
            },
            DeviceSpec {
                id: DeviceId(7),
                backend: Backend::Nvptx,
                usable_bytes: 1 << 20,
            },
        ];
        let mut links = Vec::new();
        for &kind in kinds {
            for &(from, to) in &[(DeviceId(41), DeviceId(7)), (DeviceId(7), DeviceId(41))] {
                links.push(LinkCapability { from, to, kind });
            }
        }
        Topology { devices, links }
    }

    pub(crate) fn communication_row(
        id: u32,
        kind: CollectiveKind,
        byte_len: usize,
    ) -> CommunicationRow {
        CommunicationRow {
            id: PlanNodeId(id),
            kind,
            op: RedOp::Sum,
            participants: vec![DeviceId(41), DeviceId(7)],
            byte_len,
            temp_bytes: 0,
            producers: Vec::new(),
            consumers: Vec::new(),
            route_counts: Vec::new(),
        }
    }

    pub(crate) fn static_buffer(id: &str, kind: StaticBufferKind, bytes: usize) -> StaticBuffer {
        StaticBuffer {
            id: StaticBufferId(id.into()),
            kind,
            bytes: bytes as u64,
            fingerprint: id.bytes().fold(0u64, |hash, byte| {
                hash.wrapping_mul(16777619) ^ u64::from(byte)
            }),
            per_replay_input: false,
        }
    }

    pub(crate) fn replay_input_buffer(id: &str, bytes: usize) -> StaticBuffer {
        StaticBuffer {
            per_replay_input: true,
            ..static_buffer(id, StaticBufferKind::DenseConstant, bytes)
        }
    }

    pub(crate) fn p2p_inputs() -> CaptureInputs {
        let row = communication_row(5, CollectiveKind::PointToPoint, 4097);
        let source = static_buffer("p2p-source", StaticBufferKind::Carrier, 4097);
        let destination = static_buffer("p2p-destination", StaticBufferKind::Communication, 4097);
        CaptureInputs {
            topology: topology(&[CollectiveKind::PointToPoint]),
            placements: Vec::new(),
            expected_owners: Vec::new(),
            device_usage: HashMap::new(),
            replay: ReplayContract {
                buffers: vec![source.clone(), destination.clone()],
                dispatch_count: 11,
                communication: CommunicationPlan { rows: vec![row] },
            },
            // Deliberately reverse declaration order: topology order remains authoritative.
            device_captures: vec![
                PtxDeviceCapture {
                    device: DeviceId(7),
                    graph_identity: "destination-graph".into(),
                    static_uploads: vec![StaticBufferUpload {
                        id: destination.id.clone(),
                        bytes: vec![0; 4097],
                    }],
                },
                PtxDeviceCapture {
                    device: DeviceId(41),
                    graph_identity: "source-graph".into(),
                    static_uploads: vec![StaticBufferUpload {
                        id: source.id.clone(),
                        bytes: (0..4097).map(|index| (index % 251) as u8).collect(),
                    }],
                },
            ],
            communication_buffers: vec![PtxCommunicationBufferBinding {
                row: PlanNodeId(5),
                devices: vec![
                    DeviceCommunicationBuffers {
                        device: DeviceId(41),
                        source: Some(source.id),
                        destination: None,
                    },
                    DeviceCommunicationBuffers {
                        device: DeviceId(7),
                        source: None,
                        destination: Some(destination.id),
                    },
                ],
            }],
        }
    }

    fn phase_only_admitted(
        rows: Vec<CommunicationRow>,
        kinds: &[CollectiveKind],
    ) -> AdmittedCapture {
        let topology = topology(kinds);
        let communication = CommunicationPlan { rows };
        let schedule = compile_communication_schedule(&topology, &communication).unwrap();
        let execution_order = captured_execution_order(&schedule).unwrap();
        let rank_by_device = topology
            .devices
            .iter()
            .enumerate()
            .map(|(rank, device)| (device.id, rank))
            .collect();
        AdmittedCapture {
            inputs: CaptureInputs {
                topology,
                placements: Vec::new(),
                expected_owners: Vec::new(),
                device_usage: HashMap::new(),
                replay: ReplayContract {
                    buffers: Vec::new(),
                    dispatch_count: 0,
                    communication,
                },
                device_captures: Vec::new(),
                communication_buffers: Vec::new(),
            },
            placement_set: validate_placement_set(&[], &[]).unwrap(),
            schedule,
            execution_order,
            rank_by_device,
            buffers: Vec::new(),
            row_bindings: HashMap::new(),
        }
    }

    fn capture_inputs(
        inputs: CaptureInputs,
        counters: &mut PtxMultiDeviceReplayCounters,
    ) -> Result<PtxCapturedMultiDeviceExecutor, CapturedMultiDeviceError> {
        let CaptureInputs {
            topology,
            placements,
            expected_owners,
            device_usage,
            replay,
            device_captures,
            communication_buffers,
        } = inputs;
        PtxCapturedMultiDeviceExecutor::capture_with_counters(
            topology,
            placements,
            expected_owners,
            device_usage,
            replay,
            device_captures,
            communication_buffers,
            counters,
        )
    }

    #[test]
    fn captured_executor_admission_preserves_topology_order_and_4097_byte_binding() {
        let admitted = admit_captured_multi_device_plan(p2p_inputs(), &[]).unwrap();
        assert_eq!(admitted.rank_by_device[&DeviceId(41)], 0);
        assert_eq!(admitted.rank_by_device[&DeviceId(7)], 1);
        let ScheduledCommunication::PointToPoint(transfer) =
            &admitted.schedule.rows[0].communication
        else {
            unreachable!()
        };
        assert_eq!(transfer.byte_len, 4097);
        assert_eq!(admitted.buffers[0].device, DeviceId(41));
        assert_eq!(admitted.buffers[1].device, DeviceId(7));
        assert_eq!(
            declared_peer_table(&admitted),
            vec![vec![false, true], vec![true, false]]
        );
    }

    #[test]
    fn bidirectional_all_to_all_uses_separate_source_first_event_epochs() {
        let mut row = communication_row(8, CollectiveKind::AllToAll, 8);
        row.route_counts = vec![vec![0, 3], vec![5, 0]];
        let admitted = phase_only_admitted(vec![row], &[CollectiveKind::AllToAll]);
        let phases = capture_phase_plan(&admitted);
        let directions: Vec<_> = phases
            .iter()
            .filter_map(|phase| match phase {
                CapturePhase::RemoteTransfer { transfer, .. } => {
                    Some(remote_transfer_submission_ranks(&admitted, transfer))
                }
                _ => None,
            })
            .collect();

        assert_eq!(directions, vec![[0, 1], [1, 0]]);
        assert!(matches!(
            phases.as_slice(),
            [
                CapturePhase::RemoteTransfer { .. },
                CapturePhase::RemoteTransfer { .. },
                CapturePhase::RecordNode { ranks, .. },
                CapturePhase::RecordIdentity { .. }
            ] if ranks == &[0, 1]
        ));
    }

    #[test]
    fn dependency_phase_records_producer_before_consumer_wait() {
        let mut producer = communication_row(10, CollectiveKind::PointToPoint, 5);
        producer.consumers.push(PlanNodeId(11));
        let mut consumer = communication_row(11, CollectiveKind::PointToPoint, 5);
        consumer.producers.push(PlanNodeId(10));
        let admitted =
            phase_only_admitted(vec![consumer, producer], &[CollectiveKind::PointToPoint]);
        let phases = capture_phase_plan(&admitted);
        let producer_record = phases
            .iter()
            .position(|phase| {
                matches!(
                    phase,
                    CapturePhase::RecordNode {
                        node: PlanNodeId(10),
                        ..
                    }
                )
            })
            .unwrap();
        let consumer_wait = phases
            .iter()
            .position(|phase| {
                matches!(
                    phase,
                    CapturePhase::WaitForNode {
                        producer: PlanNodeId(10),
                        ..
                    }
                )
            })
            .unwrap();
        let consumer_transfer = phases
            .iter()
            .position(|phase| {
                matches!(
                    phase,
                    CapturePhase::RemoteTransfer {
                        row: PlanNodeId(11),
                        ..
                    }
                )
            })
            .unwrap();

        assert!(producer_record < consumer_wait);
        assert!(consumer_wait < consumer_transfer);
    }

    #[test]
    fn captured_executor_retains_card360_placement_and_opaque_edges() {
        let mut inputs = p2p_inputs();
        inputs.replay.communication.rows[0].producers = vec![PlanNodeId(1001)];
        inputs.replay.communication.rows[0].consumers = vec![PlanNodeId(1002)];
        inputs.placements.push(PlacementRow::Replicated {
            owner: HostOwner {
                owner: OwnerId(77),
                total_bytes: 4097,
                encoding_unit_bytes: 1,
            },
            residents: vec![DeviceId(41), DeviceId(7)],
        });
        inputs.expected_owners = vec![OwnerId(77)];
        for device in [DeviceId(41), DeviceId(7)] {
            inputs.device_usage.insert(
                device,
                ByteCategories {
                    source_or_carrier: 4097,
                    ..ByteCategories::default()
                },
            );
        }
        let expected_placements = inputs.placements.clone();
        let expected_usage = inputs.device_usage.clone();
        let admitted = admit_captured_multi_device_plan(inputs, &[]).unwrap();
        assert_eq!(admitted.inputs.placements, expected_placements);
        assert_eq!(
            admitted.placement_set.rows(),
            expected_placements.as_slice()
        );
        assert_eq!(admitted.inputs.device_usage, expected_usage);
        assert_eq!(admitted.schedule.rows[0].wait_for, vec![PlanNodeId(1001)]);
        assert_eq!(admitted.schedule.rows[0].signal_to, vec![PlanNodeId(1002)]);
    }

    #[test]
    fn captured_executor_rejects_duplicate_placement_owner_with_zero_counters() {
        let mut inputs = p2p_inputs();
        inputs.expected_owners = vec![OwnerId(77), OwnerId(78)];
        let duplicate = PlacementRow::Replicated {
            owner: HostOwner {
                owner: OwnerId(77),
                total_bytes: 4097,
                encoding_unit_bytes: 1,
            },
            residents: vec![DeviceId(41), DeviceId(7)],
        };
        inputs.placements.push(duplicate.clone());
        inputs.placements.push(duplicate);
        for device in [DeviceId(41), DeviceId(7)] {
            inputs.device_usage.insert(
                device,
                ByteCategories {
                    source_or_carrier: 8194,
                    ..ByteCategories::default()
                },
            );
        }
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(inputs, &mut counters).err().unwrap();
        match &error {
            CapturedMultiDeviceError::PlacementSet { source } => match &**source {
                PlacementError::DuplicateOwner { owner, indices } => {
                    assert_eq!(*owner, OwnerId(77));
                    assert_eq!(indices, &[0, 1]);
                }
                other => panic!("expected DuplicateOwner, got {other:?}"),
            },
            other => panic!("expected PlacementSet, got {other:?}"),
        }
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    #[test]
    fn captured_executor_rejects_missing_and_unexpected_placement_owners_with_zero_counters() {
        let mut missing = p2p_inputs();
        missing.expected_owners = vec![OwnerId(77)];
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(missing, &mut counters).err().unwrap();
        assert!(
            matches!(
                &error,
                CapturedMultiDeviceError::PlacementSet { source }
                    if matches!(**source, PlacementError::MissingOwner { owner: OwnerId(77) })
            ),
            "expected MissingOwner, got {error:?}"
        );
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());

        let mut unexpected = p2p_inputs();
        unexpected.expected_owners = Vec::new();
        unexpected.placements.push(PlacementRow::Replicated {
            owner: HostOwner {
                owner: OwnerId(99),
                total_bytes: 4097,
                encoding_unit_bytes: 1,
            },
            residents: vec![DeviceId(41)],
        });
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(unexpected, &mut counters).err().unwrap();
        assert!(
            matches!(
                &error,
                CapturedMultiDeviceError::PlacementSet { source }
                    if matches!(
                        **source,
                        PlacementError::UnexpectedOwner {
                            owner: OwnerId(99),
                            index: 0,
                        }
                    )
            ),
            "expected UnexpectedOwner, got {error:?}"
        );
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    #[test]
    fn captured_executor_all_reduce_max_rejects_with_zero_device_counters() {
        let mut row = communication_row(9, CollectiveKind::AllReduce, 64);
        row.op = RedOp::Max;
        let inputs = CaptureInputs {
            topology: topology(&[CollectiveKind::AllReduce]),
            placements: Vec::new(),
            expected_owners: Vec::new(),
            device_usage: HashMap::new(),
            replay: ReplayContract {
                buffers: Vec::new(),
                dispatch_count: 0,
                communication: CommunicationPlan { rows: vec![row] },
            },
            device_captures: Vec::new(),
            communication_buffers: Vec::new(),
        };
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(inputs, &mut counters).err().unwrap();
        assert!(matches!(
            error,
            CapturedMultiDeviceError::Schedule(source)
                if matches!(
                    *source,
                    CommunicationScheduleError::UnsupportedAllReduceMax {
                        row: PlanNodeId(9)
                    }
                )
        ));
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    #[test]
    fn captured_executor_capability_rejects_with_zero_transfers() {
        let mut inputs = p2p_inputs();
        inputs.topology.links.clear();
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(inputs, &mut counters).err().unwrap();
        assert!(matches!(
            error,
            CapturedMultiDeviceError::Schedule(source)
                if matches!(*source, CommunicationScheduleError::Capability(_))
        ));
        assert_eq!(counters.transfers, 0);
        assert_eq!(counters.captured_transfer_nodes, 0);
        assert_eq!(counters.allocations, 0);
        assert_eq!(counters.uploads, 0);
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    #[test]
    fn replay_counter_delta_keeps_warm_setup_zero_and_dispatch_shape_constant() {
        let cold_setup = PtxMultiDeviceReplayCounters {
            allocations: 6,
            uploads: 6,
            carrier_rebuilds: 1,
            graph_instantiations: 18,
            graph_uploads: 18,
            captured_transfer_nodes: 3,
            ..PtxMultiDeviceReplayCounters::default()
        };
        let mut pending = cold_setup;
        let mut total = PtxMultiDeviceReplayCounters::default();
        let mut replay_index = 0;
        let first =
            record_successful_replay(&mut pending, &mut total, &mut replay_index, 18, 3, 11);
        let second =
            record_successful_replay(&mut pending, &mut total, &mut replay_index, 18, 3, 11);
        assert_eq!(second.delta.allocations, 0);
        assert_eq!(second.delta.uploads, 0);
        assert_eq!(second.delta.carrier_rebuilds, 0);
        assert_eq!(second.delta.graph_instantiations, 0);
        assert_eq!(second.delta.graph_uploads, 0);
        assert_eq!(second.delta.captured_transfer_nodes, 0);
        assert_eq!(first.delta.captured_transfer_nodes, 3);
        assert_eq!(first.delta.graph_launches, second.delta.graph_launches);
        assert_eq!(first.delta.transfers, second.delta.transfers);
        assert_eq!(
            first.delta.planned_dispatches,
            second.delta.planned_dispatches
        );
    }

    /// Card 408 SC-001 through SC-005 and SC-007 in one 2-GPU capture. Run only in the scoped hardware
    /// lane; the ordinary host suite must not open CUDA devices.
    #[test]
    #[ignore = "requires two peer-capable NVIDIA GPUs"]
    fn probe_captured_multi_device_p2p_all_to_all_and_warm_replay() {
        let device_count = crate::p2p::device_count().expect("Card 408 requires the NVIDIA driver");
        assert!(
            device_count >= 2,
            "Card 408 requires two NVIDIA GPUs, found {device_count}"
        );

        let point_to_point = communication_row(20, CollectiveKind::PointToPoint, 4097);
        let dependency_source = communication_row(21, CollectiveKind::PointToPoint, 5);
        let mut all_to_all = communication_row(22, CollectiveKind::AllToAll, 8);
        all_to_all.producers.push(PlanNodeId(21));
        all_to_all.route_counts = vec![vec![0, 3], vec![5, 0]];

        let definitions = [
            (
                "p2p-src",
                StaticBufferKind::Carrier,
                (0..4097).map(|i| (i % 251) as u8).collect(),
            ),
            ("p2p-dst", StaticBufferKind::Communication, vec![0; 4097]),
            (
                "dependency-src",
                StaticBufferKind::Communication,
                vec![30, 31, 32, 33, 34],
            ),
            (
                "a2a-src-0",
                StaticBufferKind::Communication,
                vec![10, 11, 12],
            ),
            ("a2a-dst-0", StaticBufferKind::Communication, vec![0; 5]),
            (
                "a2a-src-1",
                StaticBufferKind::Communication,
                vec![20, 21, 22, 23, 24],
            ),
            ("a2a-dst-1", StaticBufferKind::Communication, vec![0; 3]),
        ];
        let static_buffers: Vec<_> = definitions
            .iter()
            .map(|(id, kind, bytes)| static_buffer(id, *kind, bytes.len()))
            .collect();
        let upload = |index: usize| StaticBufferUpload {
            id: static_buffers[index].id.clone(),
            bytes: definitions[index].2.clone(),
        };
        let id = |index: usize| static_buffers[index].id.clone();
        let contract = ReplayContract {
            buffers: static_buffers.clone(),
            dispatch_count: 11,
            communication: CommunicationPlan {
                // The consumer is deliberately declared first. Its edge must move it behind row 21;
                // deleting that edge copies rank 1's initial [20..24] instead of [30..34].
                rows: vec![all_to_all, point_to_point, dependency_source],
            },
        };
        let inputs = CaptureInputs {
            topology: topology(&[CollectiveKind::PointToPoint, CollectiveKind::AllToAll]),
            placements: Vec::new(),
            expected_owners: Vec::new(),
            device_usage: HashMap::new(),
            replay: contract.clone(),
            device_captures: vec![
                PtxDeviceCapture {
                    device: DeviceId(41),
                    graph_identity: "rank-0-source-and-a2a".into(),
                    static_uploads: vec![upload(0), upload(2), upload(3), upload(4)],
                },
                PtxDeviceCapture {
                    device: DeviceId(7),
                    graph_identity: "rank-1-destination-and-a2a".into(),
                    static_uploads: vec![upload(1), upload(5), upload(6)],
                },
            ],
            communication_buffers: vec![
                PtxCommunicationBufferBinding {
                    row: PlanNodeId(20),
                    devices: vec![
                        DeviceCommunicationBuffers {
                            device: DeviceId(41),
                            source: Some(id(0)),
                            destination: None,
                        },
                        DeviceCommunicationBuffers {
                            device: DeviceId(7),
                            source: None,
                            destination: Some(id(1)),
                        },
                    ],
                },
                PtxCommunicationBufferBinding {
                    row: PlanNodeId(21),
                    devices: vec![
                        DeviceCommunicationBuffers {
                            device: DeviceId(41),
                            source: Some(id(2)),
                            destination: None,
                        },
                        DeviceCommunicationBuffers {
                            device: DeviceId(7),
                            source: None,
                            destination: Some(id(5)),
                        },
                    ],
                },
                PtxCommunicationBufferBinding {
                    row: PlanNodeId(22),
                    devices: vec![
                        DeviceCommunicationBuffers {
                            device: DeviceId(41),
                            source: Some(id(3)),
                            destination: Some(id(4)),
                        },
                        DeviceCommunicationBuffers {
                            device: DeviceId(7),
                            source: Some(id(5)),
                            destination: Some(id(6)),
                        },
                    ],
                },
            ],
        };

        let mut capture_counters = PtxMultiDeviceReplayCounters::default();
        let mut executor = capture_inputs(inputs, &mut capture_counters).unwrap();
        assert_ne!(
            executor.device_graph_identities()[0].1,
            executor.device_graph_identities()[1].1
        );
        let first = executor.replay(&contract).unwrap();
        assert_eq!(executor.download(&id(1)).unwrap(), definitions[0].2);
        assert_eq!(executor.download(&id(4)).unwrap(), definitions[2].2);
        assert_eq!(executor.download(&id(6)).unwrap(), definitions[3].2);
        assert_eq!(first.delta.allocations, 7);
        assert_eq!(first.delta.uploads, 7);
        assert_eq!(first.delta.carrier_rebuilds, 1);
        assert_eq!(first.delta.graph_instantiations, 18);
        assert_eq!(first.delta.graph_uploads, 18);
        assert_eq!(first.delta.captured_transfer_nodes, 4);
        assert_eq!(first.delta.graph_launches, 18);
        assert_eq!(first.delta.planned_dispatches, 11);
        let second = executor.replay(&contract).unwrap();
        assert_eq!(second.delta.allocations, 0);
        assert_eq!(second.delta.uploads, 0);
        assert_eq!(second.delta.carrier_rebuilds, 0);
        assert_eq!(second.delta.graph_instantiations, 0);
        assert_eq!(second.delta.graph_uploads, 0);
        assert_eq!(second.delta.captured_transfer_nodes, 0);
        assert_eq!(first.delta.graph_launches, second.delta.graph_launches);
        assert_eq!(first.delta.transfers, 4);
        assert_eq!(first.delta.transfers, second.delta.transfers);
        assert_eq!(
            first.delta.planned_dispatches,
            second.delta.planned_dispatches
        );
        assert_eq!(executor.download(&id(1)).unwrap(), definitions[0].2);
        assert_eq!(executor.download(&id(4)).unwrap(), definitions[2].2);
        assert_eq!(executor.download(&id(6)).unwrap(), definitions[3].2);

        let mut rejected = p2p_inputs();
        rejected.topology.links.clear();
        let mut rejected_counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_inputs(rejected, &mut rejected_counters)
            .err()
            .unwrap();
        assert!(matches!(
            error,
            CapturedMultiDeviceError::Schedule(source)
                if matches!(*source, CommunicationScheduleError::Capability(_))
        ));
        assert_eq!(rejected_counters.transfers, 0);
        assert_eq!(rejected_counters.captured_transfer_nodes, 0);
        assert_eq!(rejected_counters, PtxMultiDeviceReplayCounters::default());
    }
}
