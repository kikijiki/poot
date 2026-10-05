//! Card 360 acceptance table row "Model neutrality": the caller supplies every partition axis; the
//! planner never derives one from a model, tensor, layer, or expert name or from a device's position.
//!
//! Both tests plan a real layered graph end to end: the tracer-side layer tags
//! ([`poot_graph_ir::Builder::layer_scope`]) give the stage cuts, `split_stages_after_layers` turns
//! them into boundary descriptors, and `plan_stage_communication` turns those and a caller-declared
//! device placement into communication rows. The layer tags and the placement are the only inputs
//! that may move the rows.
//!
//! Mutations that must fail (observed red, card 520a):
//! - `multi_device_plan_uses_explicit_axes_only`: `stage.rs` `device_of` returns
//!   `DeviceId(stage as u32)` (a device derived from a stage index).
//! - `multi_device_substrate_is_model_neutral`: a planner or split step that branches on a value
//!   name, so two namings of the same graph plan differently.

use crate::{BoundaryDescriptor, split_stages_after_layers};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::BinOp;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{RedOp, Slot};
use poot_tensor::DType;

use crate::CollectiveKind;
use crate::multi_device::placement::{
    ByteRange, HostOwner, OwnerId, PlacementRow, placement_device_bytes,
};
use crate::multi_device::topology::DeviceId;
use crate::multi_device::{CommunicationRow, PlanNodeId, plan_stage_communication};

/// One layered graph over `[2, 3]` F32 activations. Layer 2 hands its result over as BF16, so the
/// value crossing a cut after layer 2 is 12 bytes and the value crossing a cut after layer 0 or 1 is
/// 24 bytes: the boundary size tells which layer annotation the cut followed. `weight_name(layer)`
/// names each layer's weight constant and is the only thing the callers vary between namings.
fn four_layer_graph(weight_name: impl Fn(usize) -> String) -> poot_graph_ir::Graph {
    let b = Builder::new();
    let mut h = b.slot(Slot::Activation, TensorType::f32(vec![2, 3]));
    for layer in 0..4 {
        let _scope = b.layer_scope(layer);
        let weight = b.constant(&weight_name(layer), TensorType::f32(vec![2, 3]));
        h = match layer {
            0 => b.binary(BinOp::Add, h, weight),
            1 => b.binary(BinOp::Mul, h, weight),
            2 => {
                let sum = b.binary(BinOp::Add, h, weight);
                b.cast(sum, DType::BF16)
            }
            _ => {
                let wide = b.cast(h, DType::F32);
                b.binary(BinOp::Add, wide, weight)
            }
        };
    }
    b.finish(h)
}

fn row(
    id: u32,
    participants: [u32; 2],
    byte_len: usize,
    producer: u32,
    consumer: u32,
) -> CommunicationRow {
    CommunicationRow {
        id: PlanNodeId(id),
        kind: CollectiveKind::PointToPoint,
        op: RedOp::Sum,
        participants: participants.into_iter().map(DeviceId).collect(),
        byte_len,
        temp_bytes: 0,
        producers: vec![PlanNodeId(producer)],
        consumers: vec![PlanNodeId(consumer)],
        route_counts: Vec::new(),
    }
}

fn plan_rows(
    graph: &poot_graph_ir::Graph,
    after_layers: &[usize],
    stage_devices: &[u32],
) -> (Vec<BoundaryDescriptor>, Vec<CommunicationRow>) {
    let split = split_stages_after_layers(graph, after_layers).expect("the layer cuts are legal");
    let devices: Vec<DeviceId> = stage_devices.iter().copied().map(DeviceId).collect();
    let plan =
        plan_stage_communication(&split.boundaries, &devices).expect("every stage is placed");
    (split.boundaries, plan.communication.rows)
}

#[test]
fn multi_device_plan_uses_explicit_axes_only() {
    let graph = four_layer_graph(|layer| format!("w{layer}"));

    // Cuts after layers 0 and 2, three stages on devices 7, 2, 9. None of them is the stage index,
    // and they are neither sorted nor dense, so a device derived from a stage or list position
    // lands on a different id than the caller declared.
    let (boundaries, rows) = plan_rows(&graph, &[0, 2], &[7, 2, 9]);
    assert_eq!(
        boundaries.iter().map(|b| b.byte_len).collect::<Vec<_>>(),
        vec![24, 12],
        "the first cut follows layer 0's F32 result, the second layer 2's BF16 result"
    );
    assert_eq!(
        rows,
        vec![row(3, [7, 2], 24, 0, 1), row(4, [2, 9], 12, 1, 2)],
        "each row's participants are the declared devices of its two stages"
    );

    // The same graph and the same devices under a different layer annotation: the cut is the
    // caller's `after_layer`, so moving it moves the boundary value and its size.
    let (_, after_layer_one) = plan_rows(&graph, &[1], &[7, 2]);
    assert_eq!(after_layer_one, vec![row(2, [7, 2], 24, 0, 1)]);
    let (_, after_layer_two) = plan_rows(&graph, &[2], &[7, 2]);
    assert_eq!(after_layer_two, vec![row(2, [7, 2], 12, 0, 1)]);

    // The same graph and cuts under a different placement: only the participants move.
    let (_, swapped) = plan_rows(&graph, &[0, 2], &[9, 7, 2]);
    assert_eq!(
        swapped,
        vec![row(3, [9, 7], 24, 0, 1), row(4, [7, 2], 12, 1, 2)]
    );

    // Byte-range ownership on a partitioned owner follows the declared device of each range, not
    // the range's position in the row.
    let owner = HostOwner {
        owner: OwnerId(1),
        total_bytes: 100,
        encoding_unit_bytes: 1,
    };
    let placement = PlacementRow::Partitioned {
        owner,
        ranges: vec![
            (DeviceId(5), ByteRange { start: 0, end: 60 }),
            (
                DeviceId(3),
                ByteRange {
                    start: 60,
                    end: 100,
                },
            ),
        ],
    };
    let bytes = placement_device_bytes(&placement);
    assert_eq!(bytes.len(), 2);
    assert_eq!(bytes[&DeviceId(5)], 60);
    assert_eq!(bytes[&DeviceId(3)], 40);
}

#[test]
fn multi_device_substrate_is_model_neutral() {
    let plain = four_layer_graph(|layer| format!("w{layer}"));
    let expected = plan_rows(&plain, &[0, 2], &[7, 2, 9]);

    // The same graph with every weight named for a different model family, and with names that
    // carry the role words a planner might be tempted to read.
    type Naming = (&'static str, fn(usize) -> String);
    let namings: [Naming; 4] = [
        ("qwen", |layer| format!("qwen2.layers.{layer}.mlp.weight")),
        ("deepseek", |layer| {
            format!("deepseek.blocks.{layer}.expert_{layer}.w")
        }),
        ("glm", |layer| format!("glm.layer_idx.{layer}.tensor_name")),
        ("minimax", |layer| format!("minimax.expert_role.{layer}")),
    ];
    for (family, weight_name) in namings {
        let renamed = four_layer_graph(weight_name);
        assert_eq!(
            plan_rows(&renamed, &[0, 2], &[7, 2, 9]),
            expected,
            "a graph named for {family} must split and plan exactly like the plainly named one"
        );
    }
}
