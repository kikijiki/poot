//! POOT-1206: a ROCm recording owns the data buffers its packets address, so the caller dropping its own
//! handles cannot free memory a replay still reads.

use poot_executor::Device;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::BufferStorage;

fn live_activation_bytes(device: &RocmDevice) -> u64 {
    device
        .memory()
        .into_iter()
        .find(|(role, _)| *role == BufferRole::Activation)
        .map_or(0, |(_, snapshot)| snapshot.live_bytes)
}

/// SC-002: dropping the buffers a recorded copy binds keeps their allocations live until the recording
/// drops, and the recording still replays. Mutation: build the bound kernarg in `bind_kernarg` without
/// the clones of the bound buffers; the first live-bytes check sees both allocations already freed.
#[test]
fn dropping_a_bound_buffer_keeps_its_allocation_until_the_recording_drops() {
    let Some(mut device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let before = live_activation_bytes(&device);
    let src = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 64)
        .expect("allocate src");
    let dst = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 64)
        .expect("allocate dst");
    let bound = live_activation_bytes(&device);
    assert_eq!(bound, before + 2 * 64 * 4, "both buffers are live");

    device.begin().expect("begin");
    device.copy(&src, &dst).expect("record the copy");
    let recording = device.finish().expect("finish");
    drop(src);
    drop(dst);

    assert_eq!(
        live_activation_bytes(&device),
        bound,
        "the recording must keep the dropped buffers' allocations live"
    );
    device
        .replay(&recording)
        .expect("replay after the caller dropped its buffers");
    device.synchronize().expect("synchronize");

    drop(recording);
    assert_eq!(
        live_activation_bytes(&device),
        before,
        "dropping the recording releases the allocations"
    );
}
