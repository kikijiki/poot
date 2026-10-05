//! Card 360 acceptance table row "Capability" (N=2): every required link/operation has an accepted row.
//!
//! Mutation that must fail: admit an unsupported transfer or reduction. Per-row admission is independent
//! of device count, so N=2 already exercises it.

use poot_graph_ir::RedOp;

use crate::CollectiveKind;
use crate::multi_device::communication::{
    CapabilityError, CommunicationRow, PlanNodeId, admit_communication_capability,
};
use crate::multi_device::tests::fixtures::device;
use crate::multi_device::topology::{DeviceId, LinkCapability, Topology};

fn row(id: u32, kind: CollectiveKind, participants: Vec<DeviceId>) -> CommunicationRow {
    CommunicationRow {
        id: PlanNodeId(id),
        kind,
        op: RedOp::Sum,
        participants,
        byte_len: 8,
        temp_bytes: 0,
        producers: vec![],
        consumers: vec![],
        route_counts: vec![],
    }
}

#[test]
fn multi_device_communication_capability_is_admitted_per_row() {
    let dev0 = DeviceId(0);
    let dev1 = DeviceId(1);
    // PointToPoint is directed: only the declared dev0 -> dev1 link exists.
    let topology = Topology {
        devices: vec![device(0, 1 << 30), device(1, 1 << 30)],
        links: vec![LinkCapability {
            from: dev0,
            to: dev1,
            kind: CollectiveKind::PointToPoint,
        }],
    };

    let supported = row(1, CollectiveKind::PointToPoint, vec![dev0, dev1]);
    assert!(admit_communication_capability(&topology, &[supported]).is_ok());

    let reverse = row(3, CollectiveKind::PointToPoint, vec![dev1, dev0]);
    assert_eq!(
        admit_communication_capability(&topology, &[reverse]),
        Err(CapabilityError {
            from: dev1,
            to: dev0,
            kind: CollectiveKind::PointToPoint,
        })
    );

    // An AllReduce over the same connected pair has no accepted row for that operation.
    let unsupported = row(2, CollectiveKind::AllReduce, vec![dev0, dev1]);
    assert_eq!(
        admit_communication_capability(&topology, &[unsupported]),
        Err(CapabilityError {
            from: dev0,
            to: dev1,
            kind: CollectiveKind::AllReduce,
        })
    );
}
