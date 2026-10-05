//! Typed failure of Card 408's captured multi-device executor.
//!
//! ADR-0099 S1 moved the enum here so `executor.rs` stays the plan-and-capture module; the
//! variants and their `From` impls are byte-identical to where they lived before.

use poot_graph_plan::CollectiveKind;
use poot_graph_plan::multi_device::replay::StaticBufferId;
use poot_graph_plan::multi_device::{AccountingError, DeviceId, PlacementError, PlanNodeId};
use poot_target::Backend;

use super::CommunicationScheduleError;
use crate::p2p::P2pError;

#[derive(Debug, thiserror::Error)]
pub enum CapturedMultiDeviceError {
    #[error(transparent)]
    Schedule(Box<CommunicationScheduleError>),
    #[error("placement row {index} is invalid: {source}")]
    Placement {
        index: usize,
        #[source]
        source: Box<PlacementError>,
    },
    #[error("placement set is invalid: {source}")]
    PlacementSet {
        #[source]
        source: Box<PlacementError>,
    },
    #[error("placement bytes overflowed for device {device:?}")]
    PlacementBytesOverflow { device: DeviceId },
    #[error(
        "device {device:?} accounts {accounted} source/carrier bytes but placements require {required}"
    )]
    PlacementAccountingUnderflow {
        device: DeviceId,
        accounted: u64,
        required: u64,
    },
    #[error(transparent)]
    Accounting(Box<AccountingError>),
    #[error("{role} names device {device:?}, which is absent from the topology")]
    UnknownDevice {
        role: &'static str,
        device: DeviceId,
    },
    #[error("captured PTX device {device:?} uses {backend:?}, expected Backend::Nvptx")]
    NonNvptxDevice { device: DeviceId, backend: Backend },
    #[error("captured multi-device execution needs at least two topology devices, got {actual}")]
    InvalidTopologyDeviceCount { actual: usize },
    #[error("device {device:?} has more than one captured graph plan")]
    DuplicateDeviceGraph { device: DeviceId },
    #[error("device {device:?} has no captured graph plan")]
    MissingDeviceGraph { device: DeviceId },
    #[error("static buffer {id:?} appears more than once in the replay contract")]
    DuplicateContractBuffer { id: StaticBufferId },
    #[error("static buffer {id:?} is uploaded more than once")]
    DuplicateBufferUpload { id: StaticBufferId },
    #[error("static buffer upload {id:?} is absent from the replay contract")]
    UnknownBufferUpload { id: StaticBufferId },
    #[error("replay-contract static buffer {id:?} has no device upload")]
    MissingBufferUpload { id: StaticBufferId },
    #[error("static buffer {id:?} declares {declared} bytes but upload supplies {actual}")]
    BufferSizeMismatch {
        id: StaticBufferId,
        declared: u64,
        actual: usize,
    },
    #[error("communication row {row:?} has more than one buffer binding")]
    DuplicateCommunicationBinding { row: PlanNodeId },
    #[error("communication buffer binding names unknown row {row:?}")]
    UnknownCommunicationBinding { row: PlanNodeId },
    #[error("communication row {row:?} has no buffer binding")]
    MissingCommunicationBinding { row: PlanNodeId },
    #[error("communication row {row:?} binds device {device:?} more than once")]
    DuplicateCommunicationDevice { row: PlanNodeId, device: DeviceId },
    #[error("communication row {row:?} binds non-participant device {device:?}")]
    UnexpectedCommunicationDevice { row: PlanNodeId, device: DeviceId },
    #[error("communication row {row:?} has no {role} buffer for device {device:?}")]
    MissingCommunicationBuffer {
        row: PlanNodeId,
        device: DeviceId,
        role: &'static str,
    },
    #[error(
        "communication row {row:?} {role} buffer {id:?} belongs to {actual:?}, expected {expected:?}"
    )]
    CommunicationBufferDevice {
        row: PlanNodeId,
        role: &'static str,
        id: StaticBufferId,
        actual: DeviceId,
        expected: DeviceId,
    },
    #[error(
        "communication row {row:?} {role} buffer {id:?} has {actual} bytes, needs at least {required}"
    )]
    CommunicationBufferExtent {
        row: PlanNodeId,
        role: &'static str,
        id: StaticBufferId,
        actual: usize,
        required: usize,
    },
    #[error("communication row {row:?} aliases source and destination buffer {id:?}")]
    AliasedCommunicationBuffer { row: PlanNodeId, id: StaticBufferId },
    #[error(
        "communication row {row:?} kind {kind:?} is not implemented by the captured byte executor"
    )]
    UnsupportedCommunicationKind {
        row: PlanNodeId,
        kind: CollectiveKind,
    },
    #[error("captured dependency graph contains a cycle among nodes {nodes:?}")]
    DependencyCycle { nodes: Vec<PlanNodeId> },
    #[error("replay contract changed after capture; construct a new captured executor")]
    ReplayContractChanged,
    #[error(
        "compute segment targets node {node:?}, which is a communication row; segments belong to \
         opaque plan nodes only"
    )]
    SegmentOnCommunicationRow { node: PlanNodeId },
    #[error(
        "compute segment targets node {node:?}, which no communication row or dependency edge declares"
    )]
    SegmentUnknownNode { node: PlanNodeId },
    #[error(
        "compute segment program {program} declares elem_bytes {elem_bytes}; the element size must \
         be non-zero"
    )]
    SegmentProgramElementBytes {
        node: PlanNodeId,
        program: usize,
        elem_bytes: u32,
    },
    #[error(
        "compute segment program {program} declares block {block:?}; every workgroup extent must be \
         non-zero"
    )]
    SegmentProgramBlockShape {
        node: PlanNodeId,
        program: usize,
        block: [u32; 3],
    },
    #[error(
        "compute segment launch names program {index}, but the segment declares {programs} program(s)"
    )]
    SegmentProgramIndex {
        node: PlanNodeId,
        index: usize,
        programs: usize,
    },
    #[error(
        "compute segment buffer {id:?} lives on device {actual:?}, but the segment runs on \
         {expected:?}"
    )]
    SegmentBufferDevice {
        node: PlanNodeId,
        id: StaticBufferId,
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error(
        "compute segment buffer {id:?} is {bytes} bytes, not a whole number of {elem_bytes}-byte \
         elements"
    )]
    SegmentBufferElementSize {
        node: PlanNodeId,
        id: StaticBufferId,
        bytes: u64,
        elem_bytes: u32,
    },
    #[error(
        "compute segment launch extent {threads:?} has a zero workgroup count after rounding up to \
         the program's block shape"
    )]
    SegmentLaunchExtent { node: PlanNodeId, threads: [u32; 3] },
    #[error(
        "replay input {id:?} is not declared a per-replay input by the replay contract; only a \
         per-replay input may be written between replays"
    )]
    ReplayInputNotPerReplay { id: StaticBufferId },
    #[error(
        "replay input {id:?} writes {actual} bytes but the replay contract declares {declared}"
    )]
    ReplayInputSizeMismatch {
        id: StaticBufferId,
        declared: u64,
        actual: usize,
    },
    #[error(transparent)]
    P2p(Box<P2pError>),
}

impl From<CommunicationScheduleError> for CapturedMultiDeviceError {
    fn from(error: CommunicationScheduleError) -> Self {
        Self::Schedule(Box::new(error))
    }
}

impl From<AccountingError> for CapturedMultiDeviceError {
    fn from(error: AccountingError) -> Self {
        Self::Accounting(Box::new(error))
    }
}

impl From<P2pError> for CapturedMultiDeviceError {
    fn from(error: P2pError) -> Self {
        Self::P2p(Box::new(error))
    }
}
