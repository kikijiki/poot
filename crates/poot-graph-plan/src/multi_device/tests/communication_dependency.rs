//! Card 360 acceptance table row "Communication dependency" (N>=3, non-linear DAG: one transfer depends
//! on two independent producers).
//!
//! Mutation that must fail: remove one named edge (e.g. producer-A-to-transfer, not producer-B-to-
//! transfer). At N=2 there is exactly one non-trivial order and its reversal, so dropping an edge can look
//! identical to reversing the sequence; a non-linear 3+ DAG lets the test name which edge is missing.

use poot_graph_ir::RedOp;

use crate::CollectiveKind;
use crate::multi_device::communication::{
    CommunicationPlan, CommunicationRow, DependencyEdge, PlanNodeId, communication_dependency_edges,
};
use crate::multi_device::topology::DeviceId;

#[test]
fn multi_device_communication_preserves_dependencies() {
    let producer_a = PlanNodeId(1);
    let producer_b = PlanNodeId(2);
    let consumer = PlanNodeId(3);
    let transfer = PlanNodeId(10);

    // Non-linear: `transfer` has two independent producers (A and B produce on separate devices and
    // neither depends on the other) plus one downstream consumer.
    let row = CommunicationRow {
        id: transfer,
        kind: CollectiveKind::AllReduce,
        op: RedOp::Sum,
        participants: vec![DeviceId(0), DeviceId(1), DeviceId(2)],
        byte_len: 1024,
        temp_bytes: 0,
        producers: vec![producer_a, producer_b],
        consumers: vec![consumer],
        route_counts: Vec::new(),
    };
    let plan = CommunicationPlan { rows: vec![row] };
    let edges = communication_dependency_edges(&plan);

    assert!(edges.contains(&DependencyEdge {
        from: producer_a,
        to: transfer
    }));
    assert!(edges.contains(&DependencyEdge {
        from: producer_b,
        to: transfer
    }));
    assert!(edges.contains(&DependencyEdge {
        from: transfer,
        to: consumer
    }));
    assert_eq!(edges.len(), 3, "no edge is invented or dropped");
}
