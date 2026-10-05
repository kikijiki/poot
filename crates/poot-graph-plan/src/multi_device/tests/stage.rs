//! ADR-0099 S2: the exact `CommunicationRow` set a stage split builds, for a two-stage and a
//! three-stage split. Sizes and ids are literals, so a descriptor that loses a value or a builder
//! that sizes a row from the wrong quantity reddens here.

use crate::{StageAssignment, split_stages};
use poot_graph_ir::RedOp;
use poot_graph_ir::Slot;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::BinOp;
use poot_graph_ir::types::TensorType;

use crate::CollectiveKind;
use crate::multi_device::{
    CommunicationPlan, CommunicationRow, DeviceId, PlanNodeId, StagePlanError,
    plan_stage_communication,
};

fn device(id: u32) -> DeviceId {
    DeviceId(id)
}

/// A two-stage split with one crossing value: `t0 = x + x` over `[2, 3]` F32 (6 elements, 24 bytes)
/// is produced in stage 0 and multiplied in stage 1.
fn two_stage_split() -> (Vec<crate::BoundaryDescriptor>, Vec<DeviceId>) {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![2, 3]));
    let t0 = b.binary(BinOp::Add, x, x);
    let out = b.binary(BinOp::Mul, t0, t0);
    let graph = b.finish(out);
    let assignment = StageAssignment::from_cuts(&[1], graph.eqns.len()).expect("the cut is legal");
    let split = split_stages(&graph, &assignment).expect("the graph splits");
    assert_eq!(split.boundaries.len(), 1, "one value crosses this cut");
    (split.boundaries, vec![device(0), device(1)])
}

/// A three-stage split with one fan-out value: stage 0's `t0` is read by stage 1 (24 bytes) and
/// directly by stage 2, and stage 1's BF16 `t1` (12 bytes) is read by stage 2.
fn three_stage_split() -> Vec<crate::BoundaryDescriptor> {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![2, 3]));
    let t0 = b.binary(BinOp::Add, x, x);
    let t1 = b.cast(t0, poot_tensor::DType::BF16);
    let back = b.cast(t1, poot_tensor::DType::F32);
    let out = b.binary(BinOp::Add, back, t0);
    let graph = b.finish(out);
    let assignment =
        StageAssignment::from_cuts(&[1, 2], graph.eqns.len()).expect("the cuts are legal");
    let split = split_stages(&graph, &assignment).expect("the graph splits");
    assert_eq!(
        split
            .boundaries
            .iter()
            .map(|boundary| boundary.byte_len)
            .collect::<Vec<_>>(),
        vec![24, 12],
        "t0 is 6 F32 elements and t1 is 6 BF16 elements"
    );
    split.boundaries
}

fn row(
    id: u32,
    participants: Vec<DeviceId>,
    byte_len: usize,
    producers: Vec<u32>,
    consumers: Vec<u32>,
) -> CommunicationRow {
    CommunicationRow {
        id: PlanNodeId(id),
        kind: CollectiveKind::PointToPoint,
        op: RedOp::Sum,
        participants,
        byte_len,
        temp_bytes: 0,
        producers: producers.into_iter().map(PlanNodeId).collect(),
        consumers: consumers.into_iter().map(PlanNodeId).collect(),
        route_counts: Vec::new(),
    }
}

#[test]
fn adr0099_s2_two_stage_split_builds_the_exact_row_set() {
    let (boundaries, stage_devices) = two_stage_split();
    let plan =
        plan_stage_communication(&boundaries, &stage_devices).expect("both stages are placed");

    assert_eq!(plan.stage_nodes, vec![PlanNodeId(0), PlanNodeId(1)]);
    assert_eq!(
        plan.communication,
        CommunicationPlan {
            rows: vec![row(2, vec![device(0), device(1)], 24, vec![0], vec![1])]
        }
    );

    // A hop to the same device is local: no row, but the compute node still exists.
    let local = plan_stage_communication(&boundaries, &[device(0), device(0)])
        .expect("both stages are placed");
    assert_eq!(local.stage_nodes, vec![PlanNodeId(0), PlanNodeId(1)]);
    assert!(local.communication.rows.is_empty());

    // A stage with no device is a typed error, not a row to a default device.
    assert_eq!(
        plan_stage_communication(&boundaries, &[device(0)]).expect_err("stage 1 is unplaced"),
        StagePlanError::StageUnplaced { stage: 1, count: 1 }
    );
}

#[test]
fn adr0099_s2_three_stage_split_builds_the_exact_row_set() {
    let boundaries = three_stage_split();
    let stage_devices = vec![device(0), device(1), device(2)];
    let plan =
        plan_stage_communication(&boundaries, &stage_devices).expect("all stages are placed");

    assert_eq!(
        plan.stage_nodes,
        vec![PlanNodeId(0), PlanNodeId(1), PlanNodeId(2)]
    );
    assert_eq!(
        plan.communication,
        CommunicationPlan {
            rows: vec![
                // t0 (stage 0) read by stage 1, then by stage 2: one row per hop.
                row(3, vec![device(0), device(1)], 24, vec![0], vec![1]),
                row(4, vec![device(0), device(2)], 24, vec![0], vec![2]),
                // t1 (stage 1) read by stage 2.
                row(5, vec![device(1), device(2)], 12, vec![1], vec![2]),
            ]
        }
    );
}
