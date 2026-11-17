//! POOT-1273 SC-004: on a unified-memory box (Strix Halo) the ROCm device reports the `Host` memory domain
//! through `Device::memory_topology`, derived from the HSA pool its buffers live in, with no share mechanism and
//! an unmeasured staging cost until POOT-1279 and POOT-1280.

use poot_executor::Device;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::{MemoryDomain, MemoryTopology};

/// Mutation: make `memory_domain_for_pool` ignore the pool and return `Local`; this row goes red on the iGPU.
#[test]
fn the_apu_reports_the_host_domain_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    // The APU query is independent of the pool's location: the agent says its memory is the host's RAM.
    if !device.unified_memory() {
        eprintln!("POOT-SKIP: the ROCm device is not an APU");
        return;
    }
    assert_eq!(
        device.memory_topology(),
        MemoryTopology::unmeasured(MemoryDomain::Host),
        "an APU is Host, shares nothing and has no measured staging cost"
    );
}
