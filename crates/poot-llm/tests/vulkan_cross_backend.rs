//! Raw Vulkan against wgpu: the one executor-contract row that opens both devices. It lives in poot-llm's
//! tests because poot-vulkan-device and poot-wgpu-device are same-rank siblings and neither may depend on the other,
//! even for tests. A row skips when a device does not open (`POOT_REQUIRE_VULKAN=1` and `POOT_REQUIRE_WGPU=1`
//! turn that into a failure); rows run serially under `/tmp/poot-gpu.lock`.

use std::sync::Arc;

use poot_executor::{Device, Engine, Executor};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_vulkan_device::VulkanDevice;

fn vulkan() -> Option<VulkanDevice> {
    open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())
}

/// The Weight and State roles charge the same live bytes on raw Vulkan and wgpu for the same decode
/// entry (the device's own counters, not the engine's tally).
///
/// Mutation: charge every allocation as `Activation` in `Context::alloc_storage`'s caller
/// (`VulkanDevice::allocate` passes a fixed role); the Weight row reads 0 on Vulkan and goes red.
#[test]
fn memory_counters_charge_the_roles_a_decode_entry_allocates_as_wgpu_does() {
    let Some(device) = vulkan() else {
        return;
    };
    let Some(wgpu) = open_or_skip(
        DeviceBackend::Wgpu,
        poot_wgpu_device::device::WgpuDevice::new(),
    ) else {
        return;
    };
    let fixture = poot_models::test_support::fixtures::qwen2_decode_fixture(8);
    let load = |target, device: &mut dyn Executor| {
        let exe = device
            .load_weights(
                Arc::new(fixture.store.clone()),
                poot_executor_parity::fixture_weights(&fixture),
            )
            .unwrap();
        device
            .add_entry(exe, &poot_executor_parity::staged(&fixture, target))
            .unwrap();
        device.stats().memory
    };
    let (vulkan_target, wgpu_target) = (device.target(), wgpu.target());
    let (mut vulkan_engine, mut wgpu_engine) = (Engine::new(device), Engine::new(wgpu));
    let on_vulkan = load(vulkan_target, &mut vulkan_engine);
    let on_wgpu = load(wgpu_target, &mut wgpu_engine);
    let live = |memory: &[(BufferRole, poot_runtime_common::MemoryCounterSnapshot)], role| {
        memory
            .iter()
            .find(|(r, _)| *r == role)
            .map_or(0, |(_, s)| s.live_bytes)
    };
    for role in [BufferRole::Weight, BufferRole::State] {
        assert!(
            live(&on_vulkan, role) > 0,
            "{role:?} is charged: {on_vulkan:?}"
        );
        assert_eq!(
            live(&on_vulkan, role),
            live(&on_wgpu, role),
            "{role:?} live bytes: vulkan {on_vulkan:?} wgpu {on_wgpu:?}"
        );
    }
}
