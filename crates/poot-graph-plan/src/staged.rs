//! The staged program an executor loads: one or more [`Program`]s, each on one device, joined by
//! [`Link`]s that carry a boundary value from the stage that exports it to every stage that reads it.
//!
//! [`compile_staged`] is the one way to build one, as a fixed step list with one module per step so
//! the expert-pool and multi-device cards own disjoint files:
//!
//! 1. `devices`: the partition names devices the target set contains; each stage has its device and
//!    target.
//! 2. `lora`: the whole-graph LoRA rewrite, applied once to the uncut graph.
//! 3. `pool`: the expert placement. Pooled experts rewrite each routed layer to read a resident pool
//!    and add one cut and one [`SyncPoint`] per routed selection (the host makes the selected experts
//!    resident there).
//!    A tensor-parallel partition shards the adapted graph instead: [`crate::shard`] gives every rank
//!    its share of the weights and the collectives between them, and the collectives are lowered to
//!    point-to-point exchanges between the ranks' stages before the cut (see `passes/sharding.rs`).
//! 4. cuts and stage extraction ([`split_stages`]): every crossing value becomes an export of its
//!    producer and a keyed input of each consumer. When the stages sit on distinct devices, each hop
//!    that crosses devices is a communication row, refused with [`Capability::Collective`] before any
//!    stage is compiled when the target set has no link that carries it.
//! 5. per-stage [`crate::compile`] on the stage's own target, never with LoRA again, then the link and
//!    transfer schemas are checked. Across devices the weight owners are placed and each device's exact
//!    footprint is admitted against its budget.

use crate::passes::{CutSite, StageAssignment, split_stages};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::{BoundaryId, Graph, TensorType, ValueId};
use poot_quant::weights::{WeightId, WeightMap};
use poot_target::BufferStorage;

pub use crate::multi_device::DeviceId;
use crate::multi_device::{
    AccountingError, ByteCategories, CommunicationPlan, PlacementError, PlanNodeId, StagePlanError,
    TopologyError,
};
use crate::passes::{GraphBudget, PoolError, PoolSite, ShardError, StageSplitError, expand_ranks};
use crate::{CompileError, CompileOptions, Program, Refusal, ShardRule, Target, shard};

mod devices;
mod lora;
mod pool;
#[cfg(test)]
mod tests;
mod traffic;

pub use devices::Unsupported;
pub use traffic::ReadTraffic;

/// One stage of a [`StagedProgram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StageId(pub u32);

/// One host synchronization point inside a step (a program with no pooled experts has none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncId(pub u32);

/// A host sync of a step: after stage `after` completes, the host reads `reads` (boundary values
/// that stage exported), may write the resident slots of `site`, and only then the next stage runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncPoint {
    /// The sync's identity within the step.
    pub id: SyncId,
    /// The stage whose completion triggers the sync.
    pub after: StageId,
    /// The exported boundary values the host reads at the sync.
    pub reads: Vec<BoundaryId>,
    /// The routed selection whose experts the host makes resident here.
    pub site: PoolSite,
}

/// The devices an executor exposes, each with the target `compile` plans for.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetSet {
    devices: Vec<(DeviceId, Target)>,
}

impl TargetSet {
    /// A target set of exactly one device.
    pub fn single(device: DeviceId, target: Target) -> Self {
        Self {
            devices: vec![(device, target)],
        }
    }

    /// Several devices, each with the target `compile` plans for it. No production caller yet: held for
    /// POOT-748, whose coordinator opens one engine per device.
    pub fn new(devices: Vec<(DeviceId, Target)>) -> Result<Self, TopologyError> {
        for (index, (device, _)) in devices.iter().enumerate() {
            if devices[..index].iter().any(|(seen, _)| seen == device) {
                return Err(TopologyError::DuplicateDevice(*device));
            }
        }
        Ok(Self { devices })
    }

    /// Every device with its target, in the order the set was built.
    pub fn devices(&self) -> &[(DeviceId, Target)] {
        &self.devices
    }
}

/// Where routed experts live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPlacement {
    /// Every routed expert table stays resident; nothing is paged.
    AllResident,
    /// Every routed expert table is paged through a pool of `slots_per_layer` rows per selection:
    /// the experts each step needs are made resident at a [`SyncPoint`] before they are read.
    Pooled {
        /// Rows of the pool for each routed selection.
        slots_per_layer: usize,
    },
}

/// Which devices the stages run on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DevicePlacement {
    /// One stage on one device.
    Single(DeviceId),
    /// A layer pipeline: stage `i` runs on `devices[i]`, cut after each of `after_layers`. Stages on
    /// distinct devices exchange their boundary values over communication rows.
    Pipeline {
        /// Layers after which the graph is cut.
        after_layers: Vec<usize>,
        /// The device of each stage, in stage order.
        devices: Vec<DeviceId>,
    },
    /// Tensor parallelism: every device is one rank of the group and runs every layer over its share
    /// of the weights, which `rule` assigns by the roles `weights` names. Ranks exchange what a
    /// collective carries over communication rows.
    Tensor {
        /// How weights are divided over the ranks.
        rule: ShardRule,
        /// The device of each rank, in rank order.
        devices: Vec<DeviceId>,
        /// The weight map the rule assigns shares from, by role.
        weights: Arc<WeightMap>,
    },
}

/// How a graph is cut into stages. Stated by the caller; no `Default` (ADR-0104).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// Where routed experts live.
    pub experts: ExpertPlacement,
    /// How the stages are placed on devices.
    pub devices: DevicePlacement,
}

/// One boundary value crossing from the stage that exports it to a stage that reads it. The value
/// travels in the producer's buffer: the consumer binds it with no copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    /// The boundary value's name: its id in the uncut graph.
    pub id: BoundaryId,
    /// The value's dtype and shape.
    pub aval: TensorType,
    /// The stage that exports the value.
    pub from: StageId,
    /// The stage that reads it.
    pub to: StageId,
}

/// One boundary value crossing devices: the producer's export and the consumer's import, joined by one
/// communication row, which the captured coordinator executes. Unlike a [`Link`], the consumer does not bind the producer's buffer: the row
/// moves `bytes` of the exported value's planned storage into the consumer device's own buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transfer {
    /// The boundary value's name: its id in the uncut graph.
    pub id: BoundaryId,
    /// The value's dtype and shape.
    pub aval: TensorType,
    /// The stage that exports the value.
    pub from: StageId,
    /// The stage that reads it, on another device.
    pub to: StageId,
    /// The row of [`StagedProgram::comm`] that carries it.
    pub row: PlanNodeId,
    /// The producer program's value that holds the boundary.
    pub export: ValueId,
    /// The consumer program's input the boundary binds.
    pub import: ValueId,
    /// The storage both sides agreed on.
    pub storage: BufferStorage,
    /// The bytes the row moves: the exported value's planned storage, not its logical size.
    pub bytes: u64,
}

/// How a tensor-parallel program divides the weights: rank `r` is `devices[r]` and reads each weight
/// of `shares` cut to its `r`-th equal part of the named axis of the weight's mapped logical shape
/// ([`WeightMap::rank_view`] builds that view); every other weight is read whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorParallel {
    /// The device of each rank, in rank order.
    pub devices: Vec<DeviceId>,
    /// Each weight the rank views cut, with the axis of its mapped logical shape that is divided.
    pub shares: Vec<(WeightId, usize)>,
}

/// A graph compiled into stages, each placed on one device, joined by [`Link`]s on one device and by
/// [`Transfer`]s over communication rows between devices.
#[derive(Debug)]
pub struct StagedProgram {
    stages: Vec<(StageId, DeviceId, Program)>,
    links: Vec<Link>,
    transfers: Vec<Transfer>,
    comm: CommunicationPlan,
    footprints: Vec<(DeviceId, ByteCategories)>,
    sync_points: Vec<SyncPoint>,
    tensor: Option<TensorParallel>,
}

impl StagedProgram {
    /// One stage on one device, no links.
    pub fn single(program: Program, device: DeviceId) -> Self {
        Self {
            stages: vec![(StageId(0), device, program)],
            links: Vec::new(),
            transfers: Vec::new(),
            comm: CommunicationPlan::default(),
            footprints: Vec::new(),
            sync_points: Vec::new(),
            tensor: None,
        }
    }

    /// Every stage with its device and program, in stage order.
    pub fn stages(&self) -> impl Iterator<Item = (StageId, DeviceId, &Program)> {
        self.stages.iter().map(|(s, d, p)| (*s, *d, p))
    }

    /// Take the program apart into its stages and links. With [`Self::from_parts`] this is how the
    /// executor's load-time invariant rows build programs `compile_staged` would never produce
    /// (ADR-0113: gated, no production caller).
    #[cfg(feature = "test-support")]
    pub fn into_parts(self) -> StagedParts {
        StagedParts {
            stages: self.stages,
            links: self.links,
            sync_points: self.sync_points,
        }
    }

    /// Assemble a staged program from parts without any of `compile_staged`'s checks. Test-only, see
    /// [`Self::into_parts`].
    #[cfg(feature = "test-support")]
    pub fn from_parts(parts: StagedParts) -> Self {
        Self {
            stages: parts.stages,
            links: parts.links,
            transfers: Vec::new(),
            comm: CommunicationPlan::default(),
            footprints: Vec::new(),
            sync_points: parts.sync_points,
            tensor: None,
        }
    }

    /// Every boundary value a consumer on the producer's device reads, and the stage pair it crosses, in
    /// producing order.
    pub fn links(&self) -> &[Link] {
        &self.links
    }

    /// Every boundary value that crosses devices, in producing order; the coordinator builds one hop per transfer. Together with [`Self::links`] it
    /// binds every stage's boundary input exactly once.
    pub fn transfers(&self) -> &[Transfer] {
        &self.transfers
    }

    /// The communication rows carrying [`Self::transfers`]; empty on one device; the coordinator admits and schedules these rows.
    pub fn comm(&self) -> &CommunicationPlan {
        &self.comm
    }

    /// What each device holds, by category, when the stages sit on distinct devices; empty on one
    /// device. The footprint admission checked against each device's budget at plan time; the plan tests read it back.
    pub fn footprints(&self) -> &[(DeviceId, ByteCategories)] {
        &self.footprints
    }

    /// What one step of this program reads from weights and carried state, from the planned storage of
    /// each value a dispatch reads (the roofline numerator; see [`ReadTraffic`]).
    pub fn read_traffic(&self) -> ReadTraffic {
        traffic::staged_traffic(self.stages.iter().map(|(_, _, program)| program))
    }

    /// How the weights are divided over the ranks, for a tensor-parallel program.
    pub fn tensor(&self) -> Option<&TensorParallel> {
        self.tensor.as_ref()
    }

    /// The step's host sync points, in stage order (empty unless experts are pooled).
    pub fn sync_points(&self) -> &[SyncPoint] {
        &self.sync_points
    }
}

/// A [`StagedProgram`] taken apart: see [`StagedProgram::into_parts`].
#[cfg(feature = "test-support")]
#[derive(Debug)]
pub struct StagedParts {
    /// Every stage with its device and program.
    pub stages: Vec<(StageId, DeviceId, Program)>,
    /// The links between stages on one device.
    pub links: Vec<Link>,
    /// The host sync points of a step.
    pub sync_points: Vec<SyncPoint>,
}

/// The representation of one side of a link: what [`StagedCompileError::LinkSchema`] compares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundarySchema {
    /// The boundary value's dtype and shape.
    pub aval: TensorType,
    /// The storage the stage compiled the value into.
    pub storage: BufferStorage,
}

/// Why [`compile_staged`] produced no staged program.
#[derive(Debug, thiserror::Error)]
pub enum StagedCompileError {
    /// The LoRA rewrite or a stage's own compile failed.
    #[error(transparent)]
    Compile(#[from] CompileError),
    /// The expert pool pass refused the graph.
    #[error(transparent)]
    Pool(#[from] PoolError),
    /// The sharding pass refused the graph.
    #[error(transparent)]
    Shard(#[from] ShardError),
    /// Cutting the graph into stages failed.
    #[error(transparent)]
    Split(Box<StageSplitError>),
    /// The partition names a device the target set does not contain.
    #[error("partition names device {0:?}, which the target set does not contain")]
    UnknownDevice(DeviceId),
    /// A pipeline partition names a device count that is not its stage count.
    #[error(
        "a pipeline cut after {cuts} layer(s) has {stages} stage(s) but names {devices} device(s)"
    )]
    DeviceCount {
        /// Number of layer cuts.
        cuts: usize,
        /// Number of stages those cuts make.
        stages: usize,
        /// Number of devices the partition names.
        devices: usize,
    },
    /// The partition is a combination the compiler does not support.
    #[error("unsupported partition: {0:?}")]
    Unsupported(Unsupported),
    /// The target set cannot carry a stage cut: the value the refused equation produces crosses devices
    /// and no link of the target set performs the operation. Raised before any stage is compiled.
    #[error(transparent)]
    Refused(Box<Refusal>),
    /// Planning the communication rows between devices failed.
    #[error(transparent)]
    Communication(#[from] StagePlanError),
    /// Placing the weight owners across devices failed.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// A device's footprint does not fit its budget.
    #[error(transparent)]
    Capacity(AccountingError),
    /// The device reported no memory budget.
    #[error("device {0:?} has no measured memory budget, so its footprint cannot be admitted")]
    UnknownBudget(DeviceId),
    /// A boundary hop is bound by neither a link nor a communication row.
    #[error("boundary {id:?} read by stage {stage:?} is bound by neither a link nor a row")]
    UnboundBoundary {
        /// The boundary value's name.
        id: BoundaryId,
        /// The stage that reads it.
        stage: StageId,
    },
    /// A boundary's two sides compiled to different representations.
    #[error(
        "boundary {id:?} is compiled to a different representation by stage {from:?} than stage \
         {to:?} reads: {producer:?} vs {consumer:?}"
    )]
    LinkSchema {
        /// The boundary value's name.
        id: BoundaryId,
        /// The stage that exports it.
        from: StageId,
        /// The stage that reads it.
        to: StageId,
        /// The producer's representation.
        producer: Box<BoundarySchema>,
        /// The consumer's representation.
        consumer: Box<BoundarySchema>,
    },
    /// A stage's compiled program no longer holds a boundary value.
    #[error("stage {stage:?} lost the program value of boundary {id:?} in compilation")]
    BoundaryLost {
        /// The boundary value's name.
        id: BoundaryId,
        /// The stage that lost it.
        stage: StageId,
    },
    /// A boundary is, after planning, a view of a carried-state input the stage commits: a later
    /// stage would read the committed value, not the one the uncut graph reads.
    #[error(
        "boundary {id:?} of stage {stage:?} is a view of carried-state input {state}, which the \
         stage commits"
    )]
    BoundaryAliasesState {
        /// The boundary value's name.
        id: BoundaryId,
        /// The stage that exports it.
        stage: StageId,
        /// The carried-state input the boundary views.
        state: poot_graph_ir::ValueId,
    },
}

impl From<StageSplitError> for StagedCompileError {
    fn from(error: StageSplitError) -> Self {
        Self::Split(Box::new(error))
    }
}

/// Compile `g` for the partition's devices: the fixed step list in the module doc.
pub fn compile_staged(
    g: &Graph,
    targets: &TargetSet,
    partition: &Partition,
    options: &CompileOptions,
) -> Result<StagedProgram, StagedCompileError> {
    let mut placement = devices::resolve(targets, partition)?;
    let adapted = lora::apply(g, options)?;
    if matches!(partition.experts, ExpertPlacement::Pooled { .. }) && placement.spans_devices() {
        return Err(StagedCompileError::Unsupported(
            Unsupported::PooledAcrossDevices,
        ));
    }
    let budget = GraphBudget::of(&options.limits);
    let stage_options = CompileOptions {
        lora: None,
        ..options.clone()
    };

    // A tensor-parallel group is sharded after the adapter rewrite and before any cut: each rank's
    // equations become its own stages, in segments between the collectives.
    let (placed, tensor, rank_cuts) = match &partition.devices {
        DevicePlacement::Tensor {
            rule,
            devices,
            weights,
        } => {
            let world = u32::try_from(devices.len())
                .ok()
                .and_then(std::num::NonZeroU32::new)
                .expect("a placement names at least one device");
            let sharded = shard(&adapted, weights, *rule, world)?;
            devices::admit_collectives(&sharded, &placement)?;
            let ranked = expand_ranks(&sharded)?;
            let rank_devices = placement.stages.clone();
            placement.stages = ranked
                .stage_ranks
                .iter()
                .map(|&rank| rank_devices[rank])
                .collect();
            let tensor = TensorParallel {
                devices: devices.clone(),
                shares: sharded.weights,
            };
            let placed = pool::Placed {
                graph: Cow::Owned(ranked.graph),
                sites: Vec::new(),
            };
            (placed, Some(tensor), Some(ranked.cuts))
        }
        _ => (
            pool::apply(adapted, partition.experts, &budget)?,
            None,
            None,
        ),
    };
    let adapted = placed.graph;

    let uncut = match &rank_cuts {
        Some(cuts) => cuts.is_empty(),
        None => placement.after_layers.is_empty() && placed.sites.is_empty(),
    };
    if uncut {
        let only = &placement.stages[0];
        let mut program = StagedProgram::single(
            crate::compile(&adapted, &only.target, &stage_options)?,
            only.device,
        );
        program.tensor = tensor;
        return Ok(program);
    }

    // Pooled cuts and layer cuts are both equation positions of the pooled graph; a layer-only
    // pipeline keeps cutting by layer.
    let (cut_sites, cuts) = if let Some(rank_cuts) = &rank_cuts {
        (
            rank_cuts.iter().map(|&at| CutSite::Before(at)).collect(),
            Vec::new(),
        )
    } else if placed.sites.is_empty() {
        let sites: Vec<CutSite> = placement
            .after_layers
            .iter()
            .map(|&layer| CutSite::AfterLayer(layer))
            .collect();
        (sites, Vec::new())
    } else {
        let cuts = pool::cuts(&adapted, &placement.after_layers, &placed.sites)?;
        let sites = cuts
            .iter()
            .map(|cut| CutSite::Before(cut.position))
            .collect();
        (sites, cuts)
    };
    let split = split_stages(
        &adapted,
        &StageAssignment::from_sites(&adapted, &cut_sites)?,
    )?;
    // A pool's sync cuts add stages on the one device the program runs on.
    placement.extend_to(split.stages.len());
    devices::admit_links(&adapted, &split.boundaries, &placement)?;
    let mut stages = Vec::with_capacity(split.stages.len());
    for (index, (stage, placed)) in split.stages.iter().zip(&placement.stages).enumerate() {
        let program = crate::compile(&stage.graph, &placed.target, &stage_options)?;
        stages.push((StageId(index as u32), placed.device, program));
    }

    for (index, (stage, _, program)) in stages.iter().enumerate() {
        debug_assert_eq!(stage.0 as usize, index);
        if let Some((id, state)) = program.export_aliasing_state() {
            return Err(StagedCompileError::BoundaryAliasesState {
                id,
                stage: *stage,
                state,
            });
        }
    }

    // Each boundary's schema is the producer's planned storage of the exported value; a hop is a link
    // when both stages share a device and a transfer when they do not.
    let mut exact_bytes = HashMap::new();
    let mut hops = Vec::new();
    for boundary in &split.boundaries {
        let from = StageId(boundary.producing_stage as u32);
        for &consumer in &boundary.consuming_stages {
            let to = StageId(consumer as u32);
            let sides = check_link(
                boundary.id,
                (from, &stages[boundary.producing_stage].2),
                (to, &stages[consumer].2),
            )?;
            exact_bytes.insert(
                boundary.id,
                devices::value_bytes(&sides.aval, sides.storage) as usize,
            );
            hops.push((boundary, to, sides));
        }
    }
    let plan = devices::communication(&split.boundaries, &exact_bytes, &placement)?;
    let mut links = Vec::new();
    let mut transfers = Vec::new();
    for (boundary, to, sides) in hops {
        let from = StageId(boundary.producing_stage as u32);
        let crosses = stages[boundary.producing_stage].1 != stages[to.0 as usize].1;
        if !crosses {
            links.push(Link {
                id: boundary.id,
                aval: boundary.aval.clone(),
                from,
                to,
            });
            continue;
        }
        // A hop across devices is carried by its row. A row the plan lacks leaves the hop bound by
        // nothing, which `check_bound` names.
        let Some(row) = plan
            .transfers
            .iter()
            .find(|transfer| {
                transfer.boundary == boundary.id
                    && transfer.producing_stage == boundary.producing_stage
                    && transfer.consuming_stage == to.0 as usize
            })
            .map(|transfer| transfer.row)
        else {
            continue;
        };
        transfers.push(Transfer {
            id: boundary.id,
            aval: boundary.aval.clone(),
            from,
            to,
            row,
            export: sides.export,
            import: sides.import,
            storage: sides.storage,
            bytes: devices::value_bytes(&sides.aval, sides.storage),
        });
    }
    devices::check_bound(
        &split.boundaries,
        &links,
        &transfers,
        &plan.communication,
        &plan.stage_nodes,
    )?;

    let mut footprints = Vec::new();
    if placement.spans_devices() {
        let stage_graphs: Vec<&Graph> = split.stages.iter().map(|stage| &stage.graph).collect();
        devices::place_weights(&adapted, &stage_graphs, &placement)?;
        footprints = devices::footprints(&stages, &transfers, &plan.communication, &links);
        devices::admit_capacity(&placement, &footprints)?;
    }

    // One sync per pooled selection, after the stage that computes its ids. The ids must be a value
    // that stage exports, or the host would have nothing to read.
    let mut sync_points = Vec::with_capacity(placed.sites.len());
    for (stage, cut) in cuts.iter().enumerate() {
        let Some(site) = cut.site else { continue };
        let site = placed.sites[site].0.clone();
        let from = StageId(stage as u32);
        if !stages[stage].2.exports().any(|e| e.id == site.ids) {
            return Err(StagedCompileError::BoundaryLost {
                id: site.ids,
                stage: from,
            });
        }
        sync_points.push(SyncPoint {
            id: SyncId(sync_points.len() as u32),
            after: from,
            reads: vec![site.ids],
            site,
        });
    }
    Ok(StagedProgram {
        stages,
        links,
        transfers,
        comm: plan.communication,
        footprints,
        sync_points,
        tensor,
    })
}

/// One boundary as both programs compiled it: the producer's export, the consumer's import and the
/// storage the two agreed on.
struct LinkSides {
    export: ValueId,
    import: ValueId,
    aval: TensorType,
    storage: BufferStorage,
}

/// The two sides of one link, as each program compiled them, must agree on logical dtype and shape
/// and on the exact storage (so bytes and logical extent follow). An equal byte count is not a
/// representation: a mismatch is refused here, before any executor allocates.
fn check_link(
    id: BoundaryId,
    (from, producer): (StageId, &Program),
    (to, consumer): (StageId, &Program),
) -> Result<LinkSides, StagedCompileError> {
    let exported = producer
        .exports()
        .find(|export| export.id == id)
        .ok_or(StagedCompileError::BoundaryLost { id, stage: from })?
        .value;
    let input = consumer
        .boundary_input(id)
        .ok_or(StagedCompileError::BoundaryLost { id, stage: to })?;
    let schema = |program: &Program, value| BoundarySchema {
        aval: program.graph().aval(value).clone(),
        storage: program.storage().storage(value).buffer_storage(),
    };
    let (producer_side, consumer_side) = (schema(producer, exported), schema(consumer, input));
    if producer_side != consumer_side {
        return Err(StagedCompileError::LinkSchema {
            id,
            from,
            to,
            producer: Box::new(producer_side),
            consumer: Box::new(consumer_side),
        });
    }
    Ok(LinkSides {
        export: exported,
        import: input,
        aval: producer_side.aval,
        storage: producer_side.storage,
    })
}
