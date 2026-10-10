//! Device topology: stable identities, usable bytes, backend, and link capability rows.
//!
//! The caller supplies every device and link explicitly (FR-001, FR-004); nothing is inferred from a
//! topology shape. `Topology::devices` order is meaningful (FR-010): the same device set in a different
//! order is a different plan.

use crate::CollectiveKind;
use poot_target::Backend;

/// Caller-assigned stable device identity. Never reordered, merged, or reassigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u32);

/// One device's stable identity, backend, and usable capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSpec {
    pub id: DeviceId,
    pub backend: Backend,
    /// Usable bytes on this device, after any reserve the caller already excluded. The
    /// [`crate::multi_device::accounting`] reserve category is separate.
    pub usable_bytes: u64,
}

/// One accepted (from, to, operation) row: `from` can perform `kind` targeting `to`. A group operation
/// (`AllReduce`/`AllGather`/`AllToAll`) needs a row for every ordered pair of participants (see
/// [`crate::multi_device::communication::admit_communication_capability`]). Compute capability is
/// carried by [`DeviceSpec::backend`]; every cross-device capability is a link row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinkCapability {
    pub from: DeviceId,
    pub to: DeviceId,
    pub kind: CollectiveKind,
}

/// A caller-declared, model-neutral device topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub devices: Vec<DeviceSpec>,
    pub links: Vec<LinkCapability>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TopologyError {
    #[error("device {0:?} is declared more than once in the topology")]
    DuplicateDevice(DeviceId),
    #[error("link names unknown device {0:?}")]
    UnknownDeviceInLink(DeviceId),
}

/// Validate stable device identity: no duplicate device id, and every link names a declared device.
pub fn validate_topology(topology: &Topology) -> Result<(), TopologyError> {
    let mut seen = std::collections::HashSet::with_capacity(topology.devices.len());
    for device in &topology.devices {
        if !seen.insert(device.id) {
            return Err(TopologyError::DuplicateDevice(device.id));
        }
    }
    for link in &topology.links {
        if !seen.contains(&link.from) {
            return Err(TopologyError::UnknownDeviceInLink(link.from));
        }
        if !seen.contains(&link.to) {
            return Err(TopologyError::UnknownDeviceInLink(link.to));
        }
    }
    Ok(())
}

/// The cross-device operations `backend`'s runtime performs between two of its own devices. A link row
/// between two devices exists only for a kind both devices' backends implement.
///
/// NVIDIA's runtime has the peer-to-peer transport (point-to-point copies and the ring collectives built
/// on it). The wgpu, ROCm and Vulkan runtimes have no cross-device transport, so a stage cut between two
/// of their devices has nothing to carry its boundary. The match is exhaustive: a new backend states its
/// answer here.
pub(crate) fn backend_link_kinds(backend: Backend) -> &'static [CollectiveKind] {
    match backend {
        Backend::Nvptx => &[
            CollectiveKind::PointToPoint,
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::AllToAll,
        ],
        Backend::SpirvVulkan | Backend::AmdGcn(_) => &[],
    }
}
