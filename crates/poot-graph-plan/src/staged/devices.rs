//! Step 1 of [`super::compile_staged`]: which device the stages run on. This slice admits one
//! distinct device; more is a typed refusal until multi-device placement lands.

use super::{DeviceId, DevicePlacement, Partition, StagedCompileError, TargetSet};
use crate::Target;

/// What the partition asked for that this slice cannot do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// The stages name more than one distinct device.
    MultiDevice,
    /// Experts paged through a resident pool.
    ExpertPool,
}

/// The one device, its target and the layer cuts (none for a single stage).
pub(super) struct Placement {
    pub(super) device: DeviceId,
    pub(super) target: Target,
    pub(super) after_layers: Vec<usize>,
}

pub(super) fn resolve(
    targets: &TargetSet,
    partition: &Partition,
) -> Result<Placement, StagedCompileError> {
    let (device, after_layers) = match &partition.devices {
        DevicePlacement::Single(device) => (*device, Vec::new()),
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
            let first = devices[0];
            if devices.iter().any(|&device| device != first) {
                return Err(StagedCompileError::Unsupported(Unsupported::MultiDevice));
            }
            (first, after_layers.clone())
        }
    };
    let target = targets
        .devices()
        .iter()
        .find(|(id, _)| *id == device)
        .map(|(_, target)| *target)
        .ok_or(StagedCompileError::UnknownDevice(device))?;
    Ok(Placement {
        device,
        target,
        after_layers,
    })
}
