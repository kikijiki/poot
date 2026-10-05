//! The replay-contract cache-key identity (FR-010): graph/body identity, topology, placement,
//! communication schedule, backend targets, metadata schema, and codegen epoch. Tensor contents never
//! participate, only owner-bound structure and identity.
//!
//! The key is a plain `String` from each field's `Debug` output, in a fixed field order.
//! `Topology::devices` is formatted in the
//! caller's `Vec` order, never sorted: sorting would make the key sensitive only to the device set, not
//! the order (see `specs/360-packed-multi-device-topology-capture/spec.md` FR-010).

use crate::multi_device::communication::CommunicationPlan;
use crate::multi_device::placement::PlacementRow;
use crate::multi_device::topology::Topology;
use poot_target::Backend;

/// Everything a replay-contract cache key must be sensitive to (FR-010).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MultiDeviceCacheIdentity<'a> {
    /// The graph/body identity this plan was built from (e.g. a Card 356 plan key); Card 360 does not
    /// compute this itself, only carries it as opaque text.
    pub graph_identity: &'a str,
    pub topology: &'a Topology,
    pub placements: &'a [PlacementRow],
    pub communication: &'a CommunicationPlan,
    pub targets: &'a [Backend],
    pub metadata_schema: &'a str,
    pub codegen_epoch: u64,
}

/// Build the order-sensitive replay-contract cache key for one multi-device plan identity.
pub(crate) fn multi_device_plan_cache_key(identity: &MultiDeviceCacheIdentity<'_>) -> String {
    let mut key = format!("multi_device_plan:v1:graph={}", identity.graph_identity);
    key.push_str(&format!(":devices={:?}", identity.topology.devices));
    key.push_str(&format!(":links={:?}", identity.topology.links));
    key.push_str(&format!(":placements={:?}", identity.placements));
    key.push_str(&format!(":communication={:?}", identity.communication.rows));
    key.push_str(&format!(":targets={:?}", identity.targets));
    key.push_str(&format!(
        ":schema={}:epoch={}",
        identity.metadata_schema, identity.codegen_epoch
    ));
    key
}
