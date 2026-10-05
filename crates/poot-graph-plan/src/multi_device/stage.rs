//! Stage boundaries: ADR-0099 S2's boundary-to-communication builder.
//!
//! A stage split (ADR-0099 S2, `split_stages`) hands over one
//! [`BoundaryDescriptor`] per value that crosses a stage cut, plus one graph per stage. This module
//! turns those descriptors and a device placement (one device per stage) into Card 360
//! [`CommunicationRow`]s and the opaque compute-node ids the rows depend on. It never reads a
//! tensor, a dtype, or a model name: every size comes from the descriptor, every device from the
//! placement.
//!
//! Plan-node ids are a pure function of the placement and the descriptor order, so the same split
//! and placement always produce the same ids (what a replay contract's cache identity needs):
//! stage `s` is [`PlanNodeId`]`(`s`)`, and row `j` is [`PlanNodeId`]`(`stage count + j`)`.

use crate::passes::BoundaryDescriptor;
use poot_graph_ir::RedOp;

use super::communication::{CommunicationPlan, CommunicationRow, PlanNodeId};
use super::topology::DeviceId;
use crate::CollectiveKind;

/// The plan shape of one stage split: an opaque compute node per stage, in stage order, plus the
/// boundary transfer rows.
///
/// Each stage node is the id Card 408's executor records the stage's compute segment against; the
/// rows order the transfers between those nodes.
///
/// No caller outside this crate's own tests today: held for POOT-747 (formerly 584a), which names
/// `plan_stage_communication` below as the function `compile` calls to produce a multi-device
/// `Program`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StagePlan {
    pub stage_nodes: Vec<PlanNodeId>,
    pub communication: CommunicationPlan,
}

/// Typed failure of `plan_stage_communication`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StagePlanError {
    #[error("stage {stage} has no device in a placement of {count}")]
    StageUnplaced { stage: usize, count: usize },
    #[error("the stage plan needs {nodes} plan nodes, more than u32::MAX can carry")]
    PlanNodeIdOverflow { nodes: usize },
}

/// Build one transfer row per boundary hop from `boundaries` under `stage_devices`.
///
/// `stage_devices[stage]` is that stage's device. A hop whose two stages sit on the same device is
/// local and builds no row. Every other hop becomes one [`CollectiveKind::PointToPoint`] row with
/// `byte_len` carried from the descriptor, the producing stage as its `producers` entry, and the
/// consuming stage as its `consumers` entry. A linear pipeline split gives one consumer per boundary
/// value, so it gives one row per boundary value; a value read by two later stages builds one row
/// per hop, because a `PointToPoint` row has exactly `[from, to]` participants.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-747 (formerly 584a)")
)]
pub(crate) fn plan_stage_communication(
    boundaries: &[BoundaryDescriptor],
    stage_devices: &[DeviceId],
) -> Result<StagePlan, StagePlanError> {
    let mut stage_nodes = Vec::with_capacity(stage_devices.len());
    for stage in 0..stage_devices.len() {
        stage_nodes.push(plan_node_id(stage)?);
    }

    let mut rows = Vec::new();
    for boundary in boundaries {
        let from = device_of(stage_devices, boundary.producing_stage)?;
        for &consumer in &boundary.consuming_stages {
            let to = device_of(stage_devices, consumer)?;
            if from == to {
                continue;
            }
            rows.push(CommunicationRow {
                id: plan_node_id(stage_devices.len() + rows.len())?,
                kind: CollectiveKind::PointToPoint,
                op: RedOp::Sum,
                participants: vec![from, to],
                byte_len: boundary.byte_len,
                temp_bytes: 0,
                producers: vec![stage_nodes[boundary.producing_stage]],
                consumers: vec![stage_nodes[consumer]],
                route_counts: Vec::new(),
            });
        }
    }

    Ok(StagePlan {
        stage_nodes,
        communication: CommunicationPlan { rows },
    })
}

fn plan_node_id(index: usize) -> Result<PlanNodeId, StagePlanError> {
    u32::try_from(index)
        .map(PlanNodeId)
        .map_err(|_| StagePlanError::PlanNodeIdOverflow {
            nodes: index.saturating_add(1),
        })
}

fn device_of(stage_devices: &[DeviceId], stage: usize) -> Result<DeviceId, StagePlanError> {
    stage_devices
        .get(stage)
        .copied()
        .ok_or(StagePlanError::StageUnplaced {
            stage,
            count: stage_devices.len(),
        })
}
