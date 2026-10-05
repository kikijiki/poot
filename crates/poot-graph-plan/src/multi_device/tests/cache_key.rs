//! Card 360 acceptance table row "Cache identity" (N=3, one non-reversal reordering).
//!
//! Mutation that must fail (two, tested separately): reorder two devices without changing the device set,
//! and separately remove one identity field from the key. At N=2 the only possible reorder is a full
//! reversal, which a key built over a sorted set (not the caller's sequence) would still pass unchanged.
//! That is the bug this row exists to catch, and it needs N>=3 to be distinguishable from "the set
//! changed".

use crate::multi_device::cache_key::{MultiDeviceCacheIdentity, multi_device_plan_cache_key};
use crate::multi_device::communication::CommunicationPlan;
use crate::multi_device::placement::PlacementRow;
use crate::multi_device::tests::fixtures::device;
use crate::multi_device::topology::Topology;
use poot_target::Backend;

fn topology_with_order(order: [u32; 3]) -> Topology {
    Topology {
        devices: order.iter().map(|&id| device(id, 1 << 30)).collect(),
        links: vec![],
    }
}

#[test]
fn multi_device_cache_key_covers_topology_device_order() {
    // Swap positions 2 and 3 (1-based; index 1 and 2), leaving position 1 (index 0, device 0) fixed:
    // a non-reversal reorder of the same three-device set.
    let before = topology_with_order([0, 1, 2]);
    let after = topology_with_order([0, 2, 1]);
    assert_ne!(
        before.devices, after.devices,
        "sanity: the Vec order differs"
    );

    let placements: Vec<PlacementRow> = vec![];
    let communication = CommunicationPlan::default();
    let targets = vec![Backend::SpirvVulkan];

    let identity_before = MultiDeviceCacheIdentity {
        graph_identity: "g",
        topology: &before,
        placements: &placements,
        communication: &communication,
        targets: &targets,
        metadata_schema: "v1",
        codegen_epoch: 1,
    };
    let identity_after = MultiDeviceCacheIdentity {
        topology: &after,
        ..identity_before
    };

    assert_ne!(
        multi_device_plan_cache_key(&identity_before),
        multi_device_plan_cache_key(&identity_after),
        "reordering two devices without changing the device set must change the cache key"
    );
}

#[test]
fn multi_device_cache_key_covers_topology_every_declared_field() {
    let topology = topology_with_order([0, 1, 2]);
    let placements: Vec<PlacementRow> = vec![];
    let communication = CommunicationPlan::default();
    let targets = vec![Backend::SpirvVulkan];

    let base = MultiDeviceCacheIdentity {
        graph_identity: "g",
        topology: &topology,
        placements: &placements,
        communication: &communication,
        targets: &targets,
        metadata_schema: "v1",
        codegen_epoch: 1,
    };
    let changed_epoch = MultiDeviceCacheIdentity {
        codegen_epoch: 2,
        ..base
    };
    let changed_schema = MultiDeviceCacheIdentity {
        metadata_schema: "v2",
        ..base
    };
    let changed_graph = MultiDeviceCacheIdentity {
        graph_identity: "g2",
        ..base
    };

    let base_key = multi_device_plan_cache_key(&base);
    assert_ne!(base_key, multi_device_plan_cache_key(&changed_epoch));
    assert_ne!(base_key, multi_device_plan_cache_key(&changed_schema));
    assert_ne!(base_key, multi_device_plan_cache_key(&changed_graph));
}
