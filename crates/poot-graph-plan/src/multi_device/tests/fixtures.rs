//! Small shared fixture builders for more than one row's test. Model-free: no shape, dtype, or name is
//! tied to any real checkpoint.

use crate::multi_device::topology::{DeviceId, DeviceSpec};
use poot_target::Backend;

pub fn device(id: u32, usable_bytes: u64) -> DeviceSpec {
    DeviceSpec {
        id: DeviceId(id),
        backend: Backend::SpirvVulkan,
        usable_bytes,
    }
}
