//! Card 360: packed multi-device topology and communication planner (planner half).
//!
//! Model-neutral placement of Card 359 source owners across a caller-declared device topology, explicit
//! communication over the extended [`crate::CollectiveKind`], and checked per-device admission. Every
//! type and function here is proved against a model-free virtual topology (see
//! `specs/360-packed-multi-device-topology-capture/spec.md`). This module never derives a partition
//! axis, device count, or communication intent from model-specific naming or role: the caller supplies
//! each explicitly (FR-004).
//!
//! It does not drive a real device: the captured coordinator in `poot-executor` runs the rows
//! `compile_staged` plans from it.
//!
//! Layout: `topology` (devices, links, backend), `placement` (one Card 359 host owner referenced by
//! replicated or partitioned device residents), `communication` (ordered transfer/reduction/exchange
//! rows and their explicit dependency edges), `accounting` (checked per-device byte categories and
//! admission), and `stage` (ADR-0099 S2: a stage split's boundary
//! descriptors as communication rows plus one opaque compute node per stage).

pub mod accounting;
pub mod communication;
pub mod placement;
pub mod stage;
pub mod topology;

pub use accounting::{AccountingError, ByteCategories, admit_per_device_capacity};
pub use communication::{
    CapabilityError, CommunicationPlan, CommunicationRow, DependencyEdge, PlanNodeId,
    admit_communication_capability, admit_group_capability, communication_dependency_edges,
};
pub use placement::{
    ByteRange, HostOwner, OwnerId, PlacementError, PlacementRow, PlacementSet,
    placement_device_bytes, validate_placement_set, validate_stage_placement,
};
pub use stage::StagePlanError;
pub(crate) use stage::{StagePlan, plan_stage_communication};
pub use topology::{
    DeviceId, DeviceSpec, LinkCapability, Topology, TopologyError, validate_topology,
};

#[cfg(test)]
mod tests;
