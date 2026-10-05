//! Card 360 "Routed exchange" row (N>=3, uneven/repeated/nonmonotonic route counts).
//!
//! Fails if a device's declared route counts are sorted or otherwise canonicalized. Nonmonotonic needs
//! >=3 points per row, since any 2-element sequence is monotonic.

use crate::multi_device::communication::{CommunicationError, PlanNodeId, plan_routed_exchange};
use crate::multi_device::topology::DeviceId;

#[test]
fn multi_device_routed_exchange_preserves_declared_routes() {
    let participants = vec![DeviceId(0), DeviceId(1), DeviceId(2)];
    // Device 1 sends nothing; device 0's row [0, 2, 1] is nonmonotonic; device 2's [1, 1, 0] repeats then drops.
    let route_counts = vec![vec![0, 2, 1], vec![0, 0, 0], vec![1, 1, 0]];

    let row = plan_routed_exchange(
        crate::multi_device::communication::PlanNodeId(1),
        participants.clone(),
        route_counts.clone(),
        /* byte_len_per_item */ 8,
        /* temp_bytes */ 0,
        vec![],
        vec![],
    )
    .unwrap();

    // Carried exactly: no sorting, deduplication, or aggregation.
    assert_eq!(row.route_counts, route_counts);
    let total_items: usize = route_counts.iter().flatten().sum();
    assert_eq!(row.byte_len, total_items * 8);
}

#[test]
fn multi_device_routed_exchange_rejects_total_item_overflow() {
    assert_eq!(
        plan_routed_exchange(
            PlanNodeId(2),
            vec![DeviceId(0), DeviceId(1)],
            vec![vec![usize::MAX, 1], vec![0, 0]],
            1,
            0,
            vec![],
            vec![],
        ),
        Err(CommunicationError::ByteLenOverflow)
    );
}
