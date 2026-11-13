//! POOT-1571: raw-Vulkan per-dispatch device timing reaches the engine's profile labels. A row skips when no Vulkan
//! device opens (`POOT_REQUIRE_VULKAN=1` turns that into a failure); rows run serially under `/tmp/poot-gpu.lock`.

use poot_executor::{DetailedTiming, Device, Engine, TimingOptions};
use poot_executor_parity::commit_labels::run_dispatch_labels_survive_the_stage_commit;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_vulkan_device::VulkanDevice;

/// POOT-1571: a step whose recording holds a state commit between two stages' dispatches times each dispatch under
/// its own plan label: the device gives no copy a dispatch index, so no row of the step reads `unknown`.
/// Mutation: in `Context::record_copy` wrap the copy in `timestamp_before`/`timestamp_after`; the commit takes a dispatch index,
/// the second stage's labels shift and the trailing rows read `unknown` (observed: red).
#[test]
fn vulkan_dispatch_times_keep_their_plan_labels_across_a_stage_commit() {
    let Some(device) = open_or_skip(
        DeviceBackend::Vulkan,
        VulkanDevice::new_with_device_timing(),
    ) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::with_timing(
        device,
        TimingOptions::Detailed(DetailedTiming::every_step(16, 64, 4)),
    );
    run_dispatch_labels_survive_the_stage_commit(&mut engine, target).unwrap();
}
