//! What one step of a compiled program reads from device memory: the roofline numerator.
//!
//! A decode step is memory-bound, so the bytes it must stream are the quantity a measured step time
//! is judged against (the fraction of a calibrated bandwidth ceiling). The count comes from the plan,
//! never from a checkpoint's file size: each weight is counted at the bytes its planned storage lane
//! holds on the device, once however many equations read it, and a value only a dispatch reads
//! counts (an equation planned as an alias or a view moves nothing).

use std::collections::{HashMap, HashSet};

use poot_graph_ir::{OpKind, StateRole, Storage, ValueId};

use super::devices::value_bytes;
use crate::{Plan, Program, value_ids};

/// The bytes one step of a program reads from its weights and carried state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadTraffic {
    /// Weight bytes at their planned device storage, each weight once. A table only a row gather reads
    /// (an untied embedding) is not here: the step reads the gathered rows, a handful of bytes.
    pub weight_bytes: u64,
    /// Carried state a step folds in whole (conv windows, recurrent matrices), read once per step.
    pub recurrent_state_bytes: u64,
    /// Bytes of positional cache (KV) one cached position of one row holds across every layer. A step
    /// reads this many bytes per key position each of its tokens attends to.
    pub kv_bytes_per_position: u64,
}

impl ReadTraffic {
    /// The bytes of a step whose tokens attend to `positions` key positions in total (one per token
    /// per attended position: a decode token at position `p` attends `p + 1`).
    pub fn step_bytes(&self, positions: u64) -> u64 {
        self.weight_bytes
            .saturating_add(self.recurrent_state_bytes)
            .saturating_add(self.kv_bytes_per_position.saturating_mul(positions))
    }

    fn add(&mut self, other: Self) {
        self.weight_bytes = self.weight_bytes.saturating_add(other.weight_bytes);
        self.recurrent_state_bytes = self
            .recurrent_state_bytes
            .saturating_add(other.recurrent_state_bytes);
        self.kv_bytes_per_position = self
            .kv_bytes_per_position
            .saturating_add(other.kv_bytes_per_position);
    }
}

/// Whether `op` reads input 0 as a table it takes rows of, not a tensor it streams.
fn takes_rows_of_table(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Gather { .. } | OpKind::DenseRowGather { .. } | OpKind::PackedRowGather { .. }
    )
}

/// The traffic of one program: every `Const` and `State` value a dispatch reads, resolved through the
/// aliases and views between the value and its buffer.
pub(super) fn read_traffic(program: &Program) -> ReadTraffic {
    let graph = program.graph();
    let mut source: HashMap<ValueId, ValueId> = HashMap::new();
    for (eqn, plan) in program.planned() {
        match plan {
            Plan::Alias(src) | Plan::View { src, .. } => {
                source.insert(eqn.out, *src);
            }
            Plan::Dispatches(_) => {}
        }
    }
    let root = |mut id: ValueId| {
        while let Some(&src) = source.get(&id) {
            id = src;
        }
        id
    };
    let mut streamed: HashSet<ValueId> = HashSet::new();
    for (eqn, plan) in program.planned() {
        if !matches!(plan, Plan::Dispatches(_)) {
            continue;
        }
        for (index, id) in value_ids(eqn).into_iter().enumerate() {
            if index == 0 && takes_rows_of_table(&eqn.op) {
                continue;
            }
            streamed.insert(root(id));
        }
    }
    let mut traffic = ReadTraffic::default();
    for id in streamed {
        let meta = graph.meta(id);
        let bytes = value_bytes(&meta.aval, program.storage().storage(id).buffer_storage());
        match meta.storage {
            Storage::Const => traffic.weight_bytes = traffic.weight_bytes.saturating_add(bytes),
            Storage::State => match graph.state_role(id) {
                Some(StateRole::Positional { axis }) => {
                    let extent = meta.aval.shape.get(usize::from(axis)).copied().unwrap_or(1);
                    // A contiguous cache leads with the row axis; a paged pool's positions are axis 0.
                    let rows = if axis == 0 {
                        1
                    } else {
                        meta.aval.shape.first().copied().unwrap_or(1)
                    };
                    let per_position = bytes / (extent.max(1) as u64 * rows.max(1) as u64);
                    traffic.kv_bytes_per_position =
                        traffic.kv_bytes_per_position.saturating_add(per_position);
                }
                Some(StateRole::Blocked { axis, ratio }) => {
                    // An entry covers `ratio` positions, so a position's share is the entry's over `ratio`.
                    let extent = meta.aval.shape.get(usize::from(axis)).copied().unwrap_or(1);
                    let rows = meta.aval.shape.first().copied().unwrap_or(1);
                    let positions = extent.max(1) as u64 * u64::from(ratio.get());
                    let per_position = bytes / (positions * rows.max(1) as u64);
                    traffic.kv_bytes_per_position =
                        traffic.kv_bytes_per_position.saturating_add(per_position);
                }
                Some(StateRole::Recurrent) | None => {
                    traffic.recurrent_state_bytes =
                        traffic.recurrent_state_bytes.saturating_add(bytes)
                }
                Some(StateRole::Handoff) => {}
            },
            Storage::Device | Storage::Computed(_) | Storage::Slot(_) => {}
        }
    }
    traffic
}

/// The traffic of a staged program: its stages' traffic summed (a step runs every stage).
pub(super) fn staged_traffic<'a>(programs: impl Iterator<Item = &'a Program>) -> ReadTraffic {
    let mut total = ReadTraffic::default();
    for program in programs {
        total.add(read_traffic(program));
    }
    total
}
