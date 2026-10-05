//! Card 360: packed multi-device topology and communication planner (planner half).
//!
//! Model-neutral placement of Card 359 source owners across a caller-declared device topology, explicit
//! communication over the extended [`crate::CollectiveKind`], checked per-device admission, and a
//! replay-contract cache identity. Every type and function here is proved against a model-free virtual
//! topology and a fake runtime (see `specs/360-packed-multi-device-topology-capture/spec.md`). This
//! module never derives a partition axis, device count, or communication intent from model-specific
//! naming or role: the caller supplies each explicitly (FR-004).
//!
//! Card 360 does not drive a real device. 408
//! owns proving a real captured executor honors the replay contract this module defines.
//!
//! Layout: `topology` (devices, links, backend), `placement` (one Card 359 host owner referenced by
//! replicated or partitioned device residents), `communication` (ordered transfer/reduction/exchange
//! rows and their explicit dependency edges), `accounting` (checked per-device byte categories and
//! admission), `replay` (the fake-runtime replay contract: cold plan vs warm replay), `cache_key`
//! (the order-sensitive replay-contract identity), and `stage` (ADR-0099 S2: a stage split's boundary
//! descriptors as communication rows plus one opaque compute node per stage).

pub mod accounting;
#[cfg(test)]
mod cache_key;
pub mod communication;
pub mod placement;
pub mod replay;
pub mod stage;
pub mod topology;

pub use accounting::{AccountingError, ByteCategories, admit_per_device_capacity};
pub use communication::{
    CapabilityError, CommunicationPlan, CommunicationRow, DependencyEdge, PlanNodeId,
    admit_communication_capability, communication_dependency_edges,
};
pub use placement::{
    ByteRange, HostOwner, OwnerId, PlacementError, PlacementRow, PlacementSet,
    placement_device_bytes, validate_placement, validate_placement_set,
};
pub use stage::StagePlanError;
#[cfg(test)]
pub(crate) use stage::plan_stage_communication;
pub use topology::{
    DeviceId, DeviceSpec, LinkCapability, Topology, TopologyError, validate_topology,
};

#[cfg(test)]
mod tests;
