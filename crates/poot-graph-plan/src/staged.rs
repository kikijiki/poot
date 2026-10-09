//! The staged program an executor loads: one or more [`Program`]s, each on one device, joined by
//! [`Link`]s that carry a boundary value from the stage that exports it to every stage that reads it.
//!
//! [`compile_staged`] is the one way to build one, as a fixed step list with one module per step so
//! the expert-pool and multi-device cards own disjoint files:
//!
//! 1. `devices`: the partition names devices the target set contains and, in this slice, one
//!    distinct device.
//! 2. `lora`: the whole-graph LoRA rewrite, applied once to the uncut graph.
//! 3. `pool`: the expert placement (only the resident one exists yet).
//! 4. cuts and stage extraction ([`split_stages`]): every crossing value becomes an export of its
//!    producer and a keyed input of each consumer.
//! 5. per-stage [`crate::compile`], never with LoRA again, then the link schemas are checked.

use poot_graph_ir::{BoundaryId, Graph, TensorType, ValidationChannel};
use poot_target::BufferStorage;

pub use crate::multi_device::DeviceId;
use crate::passes::StageSplitError;
use crate::{
    CompileError, CompileOptions, CutSite, Program, StageAssignment, Target, split_stages,
};

mod devices;
mod lora;
mod pool;
#[cfg(test)]
mod tests;

pub use devices::Unsupported;

/// One stage of a [`StagedProgram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StageId(pub u32);

/// One host synchronization point inside a step (a single device pipeline has none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncId(pub u32);

/// The devices an executor exposes, each with the target `compile` plans for.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetSet {
    devices: Vec<(DeviceId, Target)>,
}

impl TargetSet {
    pub fn single(device: DeviceId, target: Target) -> Self {
        Self {
            devices: vec![(device, target)],
        }
    }

    pub fn devices(&self) -> &[(DeviceId, Target)] {
        &self.devices
    }
}

/// Where routed experts live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPlacement {
    AllResident,
    /// Experts paged through a resident pool. Refused until the pool transform lands.
    Pooled,
}

/// Which devices the stages run on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DevicePlacement {
    /// One stage on one device.
    Single(DeviceId),
    /// A layer pipeline: stage `i` runs on `devices[i]`, cut after each of `after_layers`. More than
    /// one distinct device is refused until multi-device placement lands.
    Pipeline {
        after_layers: Vec<usize>,
        devices: Vec<DeviceId>,
    },
}

/// How a graph is cut into stages. Stated by the caller; no `Default` (ADR-0104).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub experts: ExpertPlacement,
    pub devices: DevicePlacement,
}

/// One boundary value crossing from the stage that exports it to a stage that reads it. The value
/// travels in the producer's buffer: the consumer binds it with no copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub id: BoundaryId,
    pub aval: TensorType,
    pub from: StageId,
    pub to: StageId,
}

/// A graph compiled into stages, each placed on one device, joined by [`Link`]s.
#[derive(Debug)]
pub struct StagedProgram<V: ValidationChannel> {
    stages: Vec<(StageId, DeviceId, Program<V>)>,
    links: Vec<Link>,
}

impl<V: ValidationChannel> StagedProgram<V> {
    /// One stage on one device, no links.
    pub fn single(program: Program<V>, device: DeviceId) -> Self {
        Self {
            stages: vec![(StageId(0), device, program)],
            links: Vec::new(),
        }
    }

    pub fn stages(&self) -> impl Iterator<Item = (StageId, DeviceId, &Program<V>)> {
        self.stages.iter().map(|(s, d, p)| (*s, *d, p))
    }

    /// Take the program apart into its stages and links. With [`Self::from_parts`] this is how the
    /// executor's load-time invariant rows build programs `compile_staged` would never produce
    /// (ADR-0113: gated, no production caller).
    #[cfg(feature = "test-support")]
    pub fn into_parts(self) -> StagedParts<V> {
        StagedParts {
            stages: self.stages,
            links: self.links,
        }
    }

    /// Assemble a staged program from parts without any of `compile_staged`'s checks. Test-only, see
    /// [`Self::into_parts`].
    #[cfg(feature = "test-support")]
    pub fn from_parts(parts: StagedParts<V>) -> Self {
        Self {
            stages: parts.stages,
            links: parts.links,
        }
    }

    /// Every boundary value and the stage pair it crosses, in producing order.
    pub fn links(&self) -> &[Link] {
        &self.links
    }
}

/// A [`StagedProgram`] taken apart: see [`StagedProgram::into_parts`].
#[cfg(feature = "test-support")]
#[derive(Debug)]
pub struct StagedParts<V: ValidationChannel> {
    pub stages: Vec<(StageId, DeviceId, Program<V>)>,
    pub links: Vec<Link>,
}

/// The representation of one side of a link: what [`StagedCompileError::LinkSchema`] compares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundarySchema {
    pub aval: TensorType,
    pub storage: BufferStorage,
}

/// Why [`compile_staged`] produced no staged program.
#[derive(Debug, thiserror::Error)]
pub enum StagedCompileError {
    #[error(transparent)]
    Compile(#[from] CompileError),
    #[error(transparent)]
    Split(Box<StageSplitError>),
    #[error("partition names device {0:?}, which the target set does not contain")]
    UnknownDevice(DeviceId),
    #[error(
        "a pipeline cut after {cuts} layer(s) has {stages} stage(s) but names {devices} device(s)"
    )]
    DeviceCount {
        cuts: usize,
        stages: usize,
        devices: usize,
    },
    #[error("unsupported partition: {0:?}")]
    Unsupported(Unsupported),
    #[error(
        "boundary {id:?} is compiled to a different representation by stage {from:?} than stage \
         {to:?} reads: {producer:?} vs {consumer:?}"
    )]
    LinkSchema {
        id: BoundaryId,
        from: StageId,
        to: StageId,
        producer: Box<BoundarySchema>,
        consumer: Box<BoundarySchema>,
    },
    #[error("stage {stage:?} lost the program value of boundary {id:?} in compilation")]
    BoundaryLost { id: BoundaryId, stage: StageId },
    /// A boundary is, after planning, a view of a carried-state input the stage commits: a later
    /// stage would read the committed value, not the one the uncut graph reads.
    #[error(
        "boundary {id:?} of stage {stage:?} is a view of carried-state input v{state}, which the \
         stage commits"
    )]
    BoundaryAliasesState {
        id: BoundaryId,
        stage: StageId,
        state: poot_graph_ir::ValueId,
    },
}

impl From<StageSplitError> for StagedCompileError {
    fn from(error: StageSplitError) -> Self {
        Self::Split(Box::new(error))
    }
}

/// Compile `g` for the partition's devices: the fixed step list in the module doc.
pub fn compile_staged<V: ValidationChannel>(
    g: &Graph<V>,
    targets: &TargetSet,
    partition: &Partition,
    options: &CompileOptions,
) -> Result<StagedProgram<V>, StagedCompileError> {
    let placement = devices::resolve(targets, partition)?;
    let adapted = lora::apply(g, options)?;
    pool::apply(partition.experts)?;
    let stage_options = CompileOptions {
        lora: None,
        ..options.clone()
    };
    let target = placement.target;
    let device = placement.device;

    if placement.after_layers.is_empty() {
        return Ok(StagedProgram::single(
            crate::compile(&adapted, &target, &stage_options)?,
            device,
        ));
    }

    let sites: Vec<CutSite> = placement
        .after_layers
        .iter()
        .map(|&layer| CutSite::AfterLayer(layer))
        .collect();
    let split = split_stages(&adapted, &StageAssignment::from_sites(&adapted, &sites)?)?;
    let mut stages = Vec::with_capacity(split.stages.len());
    for (index, stage) in split.stages.iter().enumerate() {
        let program = crate::compile(&stage.graph, &target, &stage_options)?;
        stages.push((StageId(index as u32), device, program));
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

    let mut links = Vec::new();
    for boundary in &split.boundaries {
        let from = StageId(boundary.producing_stage as u32);
        for &consumer in &boundary.consuming_stages {
            let to = StageId(consumer as u32);
            check_link(
                boundary.id,
                (from, &stages[boundary.producing_stage].2),
                (to, &stages[consumer].2),
            )?;
            links.push(Link {
                id: boundary.id,
                aval: boundary.aval.clone(),
                from,
                to,
            });
        }
    }
    Ok(StagedProgram { stages, links })
}

/// The two sides of one link, as each program compiled them, must agree on logical dtype and shape
/// and on the exact storage (so bytes and logical extent follow). An equal byte count is not a
/// representation: a mismatch is refused here, before any executor allocates.
fn check_link<V: ValidationChannel>(
    id: BoundaryId,
    (from, producer): (StageId, &Program<V>),
    (to, consumer): (StageId, &Program<V>),
) -> Result<(), StagedCompileError> {
    let exported = producer
        .exports()
        .iter()
        .find(|export| export.id == id)
        .ok_or(StagedCompileError::BoundaryLost { id, stage: from })?
        .value;
    let input = consumer
        .boundary_input(id)
        .ok_or(StagedCompileError::BoundaryLost { id, stage: to })?;
    let schema = |program: &Program<V>, value| BoundarySchema {
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
    Ok(())
}
