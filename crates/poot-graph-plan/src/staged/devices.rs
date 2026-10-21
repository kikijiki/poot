//! Step 1 of [`super::compile_staged`]: which device each stage runs on and, when the stages sit on
//! distinct devices, what that placement costs: the communication rows that carry each boundary value
//! across, the weight owners each device holds, and the bytes each device must fit.
//!
//! Everything here is host-only planning. A placement the target set cannot carry is refused before a
//! stage is compiled, so no device is touched.

use crate::refusal::RefusalDtypes;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use poot_graph_ir::{
    BoundaryId, ComputedConst, Graph, OpKind, Operand, Storage, TensorType, ValueId,
};
use poot_target::BufferStorage;

use super::{
    DeviceId, DevicePlacement, Partition, StageId, StagedCompileError, TargetSet, Transfer,
};
use crate::multi_device::topology::target_link_kinds;
use crate::multi_device::{
    AccountingError, ByteCategories, CommunicationPlan, DeviceSpec, HostOwner, LinkCapability,
    OwnerId, PlacementRow, StagePlan, Topology, admit_communication_capability,
    admit_group_capability, admit_per_device_capacity, plan_stage_communication,
    validate_stage_placement,
};
use crate::passes::BoundaryDescriptor;
use crate::predicates::GraphTables;
use crate::{Capability, CollectiveKind, Program, Refusal, ShardedGraph, Target};

/// What the partition asked for that this slice cannot do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// Experts paged through a pool on stages that sit on distinct devices: a sync point reads one
    /// device's ids and writes that device's pool.
    PooledAcrossDevices,
}

/// The device and target one stage runs on.
#[derive(Clone, Copy)]
pub(super) struct StagePlacement {
    pub(super) device: DeviceId,
    pub(super) target: Target,
}

/// One entry per stage, in stage order, and the layer cuts between them (none for a single stage).
pub(super) struct Placement {
    pub(super) stages: Vec<StagePlacement>,
    pub(super) after_layers: Vec<usize>,
}

impl Placement {
    /// Stage a pool's sync cuts add to a single-device program run on that device: pad the placement to
    /// `count` stages with the last one's device and target. A pipeline across devices never pools
    /// (`compile_staged` refuses it), so every stage shares one placement.
    pub(super) fn extend_to(&mut self, count: usize) {
        if let Some(&last) = self.stages.last() {
            self.stages.resize(count.max(self.stages.len()), last);
        }
    }

    fn stage_devices(&self) -> Vec<DeviceId> {
        self.stages.iter().map(|stage| stage.device).collect()
    }

    /// Whether the stages name more than one distinct device.
    pub(super) fn spans_devices(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| stage.device != self.stages[0].device)
    }

    /// The devices of the placement and the links their backends' runtimes provide between them.
    fn topology(&self) -> Topology {
        let distinct = self.distinct_devices();
        Topology {
            // Capability admission reads the link rows only; the budget is read, and an unknown one
            // refused, by `admit_capacity` once the stages are compiled.
            devices: distinct
                .iter()
                .map(|stage| DeviceSpec {
                    id: stage.device,
                    target: stage.target.id(),
                    usable_bytes: stage.target.memory_budget_bytes().unwrap_or(0),
                })
                .collect(),
            links: distinct
                .iter()
                .flat_map(|from| distinct.iter().map(move |to| (*from, *to)))
                .filter(|(from, to)| from.device != to.device)
                .flat_map(|(from, to)| {
                    let to_kinds = target_link_kinds(&to.target);
                    target_link_kinds(&from.target)
                        .iter()
                        .filter(move |kind| to_kinds.contains(kind))
                        .map(move |&kind| LinkCapability {
                            from: from.device,
                            to: to.device,
                            kind,
                        })
                })
                .collect(),
        }
    }

    /// Each distinct device once, in first-use order, with its target.
    fn distinct_devices(&self) -> Vec<&StagePlacement> {
        let mut seen = HashSet::new();
        self.stages
            .iter()
            .filter(|stage| seen.insert(stage.device))
            .collect()
    }
}

pub(super) fn resolve(
    targets: &TargetSet,
    partition: &Partition,
) -> Result<Placement, StagedCompileError> {
    let (devices, after_layers) = match &partition.devices {
        DevicePlacement::Single(device) => (vec![*device], Vec::new()),
        // One entry per rank: the stages of a tensor-parallel program are placed after the sharding
        // pass has said which rank each one is.
        DevicePlacement::Tensor { devices, .. } => (devices.clone(), Vec::new()),
        DevicePlacement::Pipeline {
            after_layers,
            devices,
        } => {
            let stages = after_layers.len() + 1;
            if devices.len() != stages {
                return Err(StagedCompileError::DeviceCount {
                    cuts: after_layers.len(),
                    stages,
                    devices: devices.len(),
                });
            }
            (devices.clone(), after_layers.clone())
        }
    };
    let stages = devices
        .into_iter()
        .map(|device| {
            targets
                .devices()
                .iter()
                .find(|(id, _)| *id == device)
                .map(|(_, target)| StagePlacement {
                    device,
                    target: *target,
                })
                .ok_or(StagedCompileError::UnknownDevice(device))
        })
        .collect::<Result<_, _>>()?;
    Ok(Placement {
        stages,
        after_layers,
    })
}

/// The communication rows of the cut, refused when the target set has no link that carries them.
///
/// `boundaries` are sized by the uncut graph's logical bytes here: capability depends on which devices a
/// value crosses, not on how many bytes, so the refusal comes before any stage is compiled.
/// [`communication`] then rebuilds the rows with the exact planned bytes.
pub(super) fn admit_links(
    adapted: &Graph,
    boundaries: &[BoundaryDescriptor],
    placement: &Placement,
) -> Result<(), StagedCompileError> {
    if !placement.spans_devices() {
        return Ok(());
    }
    let plan = plan_stage_communication(boundaries, &placement.stage_devices())?;
    let topology = placement.topology();
    let Err(missing) = admit_communication_capability(&topology, &plan.communication.rows) else {
        return Ok(());
    };
    let transfer = plan
        .communication
        .rows
        .iter()
        .zip(&plan.transfers)
        .find(|(row, _)| row.kind == missing.kind && row.participants == [missing.from, missing.to])
        .map(|(_, transfer)| transfer)
        .expect("a refused row is one of the plan's rows");
    let eqn = adapted
        .eqns()
        .iter()
        .find(|eqn| eqn.out == transfer.boundary.0)
        .expect("a boundary value is an equation result of the uncut graph");
    let target = placement.stages[transfer.producing_stage].target.id();
    Err(StagedCompileError::Refused(Box::new(Refusal {
        eqn: eqn.out,
        op: eqn.op.clone(),
        dtypes: RefusalDtypes::of(GraphTables::from(adapted), eqn),
        target,
        missing: Capability::Collective(missing.kind),
    })))
}

/// The collectives of a sharded graph, refused when the ranks' devices have no link that performs
/// them. `placement` has one entry per rank. The refusal names the collective equation and
/// [`Capability::Collective`], before the program is lowered to exchanges or any stage is compiled.
pub(super) fn admit_collectives(
    sharded: &ShardedGraph,
    placement: &Placement,
) -> Result<(), StagedCompileError> {
    let topology = placement.topology();
    let ranks = placement.stage_devices();
    for eqn in sharded.graph.eqns() {
        let kind = match eqn.op {
            OpKind::AllReduce { .. } => CollectiveKind::AllReduce,
            OpKind::AllGather { .. } => CollectiveKind::AllGather,
            _ => continue,
        };
        let Err(missing) = admit_group_capability(&topology, kind, &ranks) else {
            continue;
        };
        let target = placement
            .stages
            .iter()
            .find(|stage| stage.device == missing.from)
            .expect("a refused link starts at a placed device")
            .target
            .id();
        return Err(StagedCompileError::Refused(Box::new(Refusal {
            eqn: eqn.out,
            op: eqn.op.clone(),
            dtypes: RefusalDtypes::of(GraphTables::from(&sharded.graph), eqn),
            target,
            missing: Capability::Collective(missing.kind),
        })));
    }
    Ok(())
}

/// The bytes a value occupies on a device in `storage`: native elements at the element width, or the
/// logical size for a lane no executor device-encodes.
pub(super) fn value_bytes(aval: &TensorType, storage: BufferStorage) -> u64 {
    let bytes = match storage.device_elems(&aval.shape) {
        Some(elems) => elems.saturating_mul(storage.element().byte_width()),
        None => aval
            .shape
            .iter()
            .fold(1usize, |count, &extent| count.saturating_mul(extent))
            .saturating_mul(aval.dtype.byte_size()),
    };
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

/// The communication plan with the bytes each row actually moves: the producer's planned storage of the
/// exported value, which differs from the logical size for a packed representation.
pub(super) fn communication(
    boundaries: &[BoundaryDescriptor],
    exact_bytes: &HashMap<BoundaryId, usize>,
    placement: &Placement,
) -> Result<StagePlan, StagedCompileError> {
    let sized: Vec<BoundaryDescriptor> = boundaries
        .iter()
        .map(|boundary| BoundaryDescriptor {
            byte_len: exact_bytes
                .get(&boundary.id)
                .copied()
                .unwrap_or(boundary.byte_len),
            ..boundary.clone()
        })
        .collect();
    Ok(plan_stage_communication(
        &sized,
        &placement.stage_devices(),
    )?)
}

/// Every boundary hop is covered by exactly one link (same device) or one transfer whose row exists in
/// `comm` with the hop's own producer, consumer and bytes. Names the first hop that is not.
pub(super) fn check_bound(
    boundaries: &[BoundaryDescriptor],
    links: &[super::Link],
    transfers: &[Transfer],
    comm: &CommunicationPlan,
    stage_nodes: &[crate::multi_device::PlanNodeId],
) -> Result<(), StagedCompileError> {
    for boundary in boundaries {
        let from = StageId(boundary.producing_stage as u32);
        for &consumer in &boundary.consuming_stages {
            let to = StageId(consumer as u32);
            let linked = links
                .iter()
                .any(|link| link.id == boundary.id && link.from == from && link.to == to);
            let carried = transfers
                .iter()
                .filter(|transfer| {
                    transfer.id == boundary.id && transfer.from == from && transfer.to == to
                })
                .any(|transfer| {
                    comm.rows.iter().any(|row| {
                        row.id == transfer.row
                            && row.producers == [stage_nodes[boundary.producing_stage]]
                            && row.consumers == [stage_nodes[consumer]]
                            && row.byte_len as u64 == transfer.bytes
                    })
                });
            if !linked && !carried {
                return Err(StagedCompileError::UnboundBoundary {
                    id: boundary.id,
                    stage: to,
                });
            }
        }
    }
    Ok(())
}

/// The named constants `g`'s equations read, with their types, in first-read order.
fn read_weights(g: &Graph) -> Vec<(String, TensorType)> {
    let read: HashSet<ValueId> = g
        .eqns()
        .iter()
        .flat_map(|eqn| &eqn.inputs)
        .filter_map(|operand| match operand {
            Operand::Value(value) => Some(*value),
            Operand::Lit(_) => None,
        })
        .collect();
    g.consts()
        .iter()
        .copied()
        .filter(|value| read.contains(value))
        .filter_map(|value| {
            let meta = g.meta(value);
            match (&meta.storage, meta.name()) {
                (Storage::Const, Some(name)) => Some((name.into_owned(), meta.aval.clone())),
                _ => None,
            }
        })
        .collect()
}

/// Every weight owner the stage split must place, and where each device holds it.
///
/// An owner is a store key. The uncut graph names the owners that must be placed; each stage graph
/// names the owners it reads, and a key read by stages on several devices is one replicated row whose
/// residents are those devices. Two stages that bind one key to tensors of different types are two
/// rows for one owner, which [`validate_stage_placement`] refuses as a duplicate.
///
/// Missing and off-stage owners are structural invariants of this derivation, not refusals a caller's
/// input can reach: `compile_staged` receives a graph, targets, a partition and options, and no
/// independent placement or weight schema. The compiled programs are not an independent source either:
/// `legalize` renames an oversized constant to `{name}.chunk{n}`, so reading owners back from them would
/// refuse a legitimate chunked weight. [`validate_stage_placement`] still checks both, and its own rows
/// drive them with placements the derivation would never build.
pub(super) fn place_weights(
    adapted: &Graph,
    stage_graphs: &[&Graph],
    placement: &Placement,
) -> Result<(), StagedCompileError> {
    let expected_keys: Vec<String> = read_weights(adapted)
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    let per_stage: Vec<Vec<(String, TensorType)>> =
        stage_graphs.iter().map(|g| read_weights(g)).collect();
    let owners: BTreeMap<&str, OwnerId> = expected_keys
        .iter()
        .map(String::as_str)
        .chain(per_stage.iter().flatten().map(|(key, _)| key.as_str()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, OwnerId(index as u64)))
        .collect();

    // (owner, type) -> the devices that read it, in stage order.
    let mut groups: Vec<(OwnerId, TensorType, Vec<DeviceId>)> = Vec::new();
    let mut readers: HashMap<OwnerId, Vec<(usize, DeviceId)>> = HashMap::new();
    for (stage, weights) in per_stage.iter().enumerate() {
        let device = placement.stages[stage].device;
        for (key, aval) in weights {
            let owner = owners[key.as_str()];
            readers.entry(owner).or_default().push((stage, device));
            match groups
                .iter_mut()
                .find(|(seen, seen_aval, _)| *seen == owner && seen_aval == aval)
            {
                Some((_, _, devices)) => {
                    if !devices.contains(&device) {
                        devices.push(device);
                    }
                }
                None => groups.push((owner, aval.clone(), vec![device])),
            }
        }
    }
    let rows: Vec<PlacementRow> = groups
        .into_iter()
        .map(|(owner, aval, residents)| PlacementRow::Replicated {
            owner: HostOwner {
                owner,
                total_bytes: aval.numel().saturating_mul(aval.dtype.byte_size()),
                encoding_unit_bytes: 1,
            },
            residents,
        })
        .collect();
    let expected: Vec<OwnerId> = expected_keys
        .iter()
        .map(|key| owners[key.as_str()])
        .collect();
    validate_stage_placement(&rows, &expected, &readers)?;
    Ok(())
}

/// What two stages on one device share one buffer of: a store key, or a computed constant's payload. A
/// constant with no key is local to its stage, so it is never merged with another stage's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SharedBuffer {
    Key(String),
    Computed(ComputedConst),
    Local(StageId, ValueId),
}

/// The bytes each distinct device holds for `stages` and the transfers between them: unique weights and
/// computed constants (a store key a device's stages share is one buffer), carried state, every stage's
/// retained arena, the stage inputs that are not boundary values, and the boundary buffers a transfer
/// delivers plus each row's declared scratch.
pub(super) fn footprints(
    stages: &[(StageId, DeviceId, Program)],
    transfers: &[Transfer],
    comm: &CommunicationPlan,
    links: &[super::Link],
) -> Vec<(DeviceId, ByteCategories)> {
    let mut usage: BTreeMap<DeviceId, ByteCategories> = BTreeMap::new();
    let mut shared: HashSet<(DeviceId, SharedBuffer)> = HashSet::new();
    for (stage, device, program) in stages {
        let graph = program.graph();
        let boundary_inputs: HashSet<ValueId> = links
            .iter()
            .map(|link| (link.id, link.to))
            .chain(transfers.iter().map(|transfer| (transfer.id, transfer.to)))
            .filter(|(_, to)| to == stage)
            .filter_map(|(id, _)| program.boundary_input(id))
            .collect();
        let mut categories = usage.get(device).copied().unwrap_or_default();
        for &input in graph.inputs() {
            let meta = graph.meta(input);
            let bytes = value_bytes(
                &meta.aval,
                program.storage().storage(input).buffer_storage(),
            );
            match meta.storage {
                Storage::Const => {
                    let key = match meta.name() {
                        Some(name) => SharedBuffer::Key(name.into_owned()),
                        None => SharedBuffer::Local(*stage, input),
                    };
                    if shared.insert((*device, key)) {
                        categories.dense_constants =
                            categories.dense_constants.saturating_add(bytes);
                    }
                }
                Storage::Computed(computed) => {
                    if shared.insert((*device, SharedBuffer::Computed(computed))) {
                        categories.dense_constants =
                            categories.dense_constants.saturating_add(bytes);
                    }
                }
                Storage::State => categories.state = categories.state.saturating_add(bytes),
                Storage::Slot(_) | Storage::Device => {
                    if !boundary_inputs.contains(&input) {
                        categories.activations = categories.activations.saturating_add(bytes);
                    }
                }
            }
        }
        categories.activations = categories
            .activations
            .saturating_add(program.buffer_plan().arena_bytes() as u64);
        usage.insert(*device, categories);
    }
    for transfer in transfers {
        let (_, device, _) = &stages[transfer.to.0 as usize];
        let categories = usage.entry(*device).or_default();
        categories.communication = categories.communication.saturating_add(transfer.bytes);
    }
    for row in &comm.rows {
        for &device in &row.participants {
            let categories = usage.entry(device).or_default();
            categories.communication = categories
                .communication
                .saturating_add(row.temp_bytes as u64);
        }
    }
    usage.into_iter().collect()
}

/// Admit each device's footprint against its own measured budget. A device with no known budget is
/// refused: an unknown budget is never an unlimited one.
pub(super) fn admit_capacity(
    placement: &Placement,
    usage: &[(DeviceId, ByteCategories)],
) -> Result<(), StagedCompileError> {
    let devices: Vec<DeviceSpec> = placement
        .distinct_devices()
        .into_iter()
        .map(|stage| {
            stage
                .target
                .memory_budget_bytes()
                .map(|usable_bytes| DeviceSpec {
                    id: stage.device,
                    target: stage.target.id(),
                    usable_bytes,
                })
                .ok_or(StagedCompileError::UnknownBudget(stage.device))
        })
        .collect::<Result<_, _>>()?;
    admit_per_device_capacity(&devices, &usage.iter().copied().collect())
        .map_err(|error: AccountingError| StagedCompileError::Capacity(error))
}

#[cfg(test)]
mod tests;
