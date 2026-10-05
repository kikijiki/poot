//! The staged program an executor loads (Card 546a): one or more [`Program`]s, each on
//! one device. This card lands the single-stage shape only; multi-stage cuts, links and sync points
//! are Card 581a's.

use poot_graph_ir::{Graph, ValidationChannel};

pub use crate::multi_device::DeviceId;
use crate::{CompileError, CompileOptions, Program, Target};

/// One stage of a [`StagedProgram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StageId(pub u32);

/// One host synchronization point inside a step (Card 581a places them; a single stage has none).
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

/// Where routed experts live. The one M3 variant keeps every expert resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPlacement {
    AllResident,
}

/// Which devices the stages run on. The one M3 variant is a single device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePlacement {
    Single(DeviceId),
}

/// How a graph is cut into stages. Stated by the caller; no `Default` (ADR-0104).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub experts: ExpertPlacement,
    pub devices: DevicePlacement,
}

/// A graph compiled into stages, each placed on one device.
#[derive(Debug)]
pub struct StagedProgram<V: ValidationChannel> {
    stages: Vec<(StageId, DeviceId, Program<V>)>,
}

impl<V: ValidationChannel> StagedProgram<V> {
    /// One stage on one device, no links, no syncs.
    pub fn single(program: Program<V>, device: DeviceId) -> Self {
        Self {
            stages: vec![(StageId(0), device, program)],
        }
    }

    pub fn stages(&self) -> impl Iterator<Item = (StageId, DeviceId, &Program<V>)> {
        self.stages.iter().map(|(s, d, p)| (*s, *d, p))
    }
}

/// Why [`compile_staged`] produced no staged program.
#[derive(Debug, thiserror::Error)]
pub enum StagedCompileError {
    #[error(transparent)]
    Compile(#[from] CompileError),
    #[error("partition names device {0:?}, which the target set does not contain")]
    UnknownDevice(DeviceId),
}

/// Compile `g` for the partition's devices. For the one M3 partition this is `compile` plus
/// [`StagedProgram::single`].
pub fn compile_staged<V: ValidationChannel>(
    g: &Graph<V>,
    targets: &TargetSet,
    partition: &Partition,
    options: &CompileOptions,
) -> Result<StagedProgram<V>, StagedCompileError> {
    let Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Single(device),
    } = *partition;
    let target = targets
        .devices
        .iter()
        .find(|(id, _)| *id == device)
        .map(|(_, target)| *target)
        .ok_or(StagedCompileError::UnknownDevice(device))?;
    Ok(StagedProgram::single(
        crate::compile(g, &target, options)?,
        device,
    ))
}
