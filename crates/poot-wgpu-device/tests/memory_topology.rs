//! POOT-1273 SC-004: on a unified-memory box (Strix Halo) the wgpu device reports the `Host` memory domain
//! through `Device::memory_topology`, derived from the adapter's heaps and device type, with no share mechanism
//! and an unmeasured staging cost until POOT-1279 and POOT-1280.

use poot_executor::Device;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::dram_channels::DramChannels;
use poot_target::{MemoryDomain, MemoryTopology};
use poot_wgpu_device::device::WgpuDevice;

/// Mutation: make `Context::memory_topology` report `MemoryDomain::Local`; this row goes red on the iGPU.
#[test]
fn the_unified_memory_adapter_reports_the_host_domain_on_wgpu() {
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let adapter = device.identity().expect("wgpu names its adapter").adapter;
    if DramChannels::of_device_name(&adapter).is_none() {
        eprintln!("POOT-SKIP: {adapter} is not a known unified-memory part (Strix Halo)");
        return;
    }
    assert_eq!(
        device.memory_topology(),
        MemoryTopology::unmeasured(MemoryDomain::Host),
        "{adapter}: a unified-memory adapter is Host, shares nothing and has no measured staging cost"
    );
}
