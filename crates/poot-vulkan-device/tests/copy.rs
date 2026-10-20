//! POOT-1235: `Device::copy` on Vulkan, driven through the production `begin`/`copy`/`finish`/`replay`. The
//! probes are `poot_executor_parity::copy_probe`, the same on every backend: a copy of a byte length that is
//! not a whole number of `u32` words arrives whole, and a copy between buffers of different byte lengths is
//! refused when recorded.

use poot_executor_parity::copy_probe;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_vulkan_device::VulkanDevice;

/// SC-001. Mutation: give `Context::alloc_storage` an unrounded `byte_len`; `vkCmdCopyBuffer` of a 6-byte range is invalid under the validation layer.
#[test]
fn copy_of_an_odd_byte_length_is_exact_on_vulkan() {
    let Some(mut device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    copy_probe::odd_width_copy_is_exact(&mut device);
}

/// A copy between buffers of different byte lengths is a device error at record time, not at replay.
/// Mutation: drop the `byte_len` comparison in `Context::record_copy`.
#[test]
fn copy_between_unequal_lengths_is_refused_on_vulkan() {
    let Some(mut device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    copy_probe::copy_of_unequal_lengths_is_refused(&mut device);
}

/// Misuse of the recording transaction is a typed error, never a panic. Mutation: restore
/// `assert!(self.open.is_none(), ..)` in `VulkanDevice::begin`; the second `begin` panics.
#[test]
fn transaction_misuse_is_a_typed_error_on_vulkan() {
    use poot_executor::Device;
    use poot_runtime_common::BufferRole;
    use poot_target::BufferStorage;
    use poot_vulkan_device::VulkanGpuError;

    let Some(mut device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let src = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 4)
        .expect("allocate src");
    let dst = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 4)
        .expect("allocate dst");
    assert!(matches!(
        device.copy(&src, &dst),
        Err(VulkanGpuError::NotRecording { op: "copy" })
    ));
    assert!(matches!(
        device.finish(),
        Err(VulkanGpuError::NotRecording { op: "finish" })
    ));
    device.begin().expect("begin");
    assert!(matches!(
        device.begin(),
        Err(VulkanGpuError::AlreadyRecording)
    ));
    device
        .finish()
        .expect("the open recording survived the misuse");
}
