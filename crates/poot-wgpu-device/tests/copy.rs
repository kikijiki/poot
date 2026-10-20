//! POOT-1235: `Device::copy` on Wgpu, driven through the production `begin`/`copy`/`finish`/`replay`. The
//! probes are `poot_executor_parity::copy_probe`, the same on every backend: a copy of a byte length that is
//! not a whole number of `u32` words arrives whole, and a copy between buffers of different byte lengths is
//! refused when recorded.

use poot_executor_parity::copy_probe;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_wgpu_device::device::WgpuDevice;

/// SC-001. Mutation: make `Context::copy_bytes` return `elem_count * width` unrounded; wgpu rejects the 6-byte `copy_buffer_to_buffer`.
#[test]
fn copy_of_an_odd_byte_length_is_exact_on_wgpu() {
    let Some(mut device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    copy_probe::odd_width_copy_is_exact(&mut device);
}

/// A copy between buffers of different byte lengths is a device error at record time, not at replay.
/// Mutation: drop the `copy_bytes` call from `WgpuDevice::copy`.
#[test]
fn copy_between_unequal_lengths_is_refused_on_wgpu() {
    let Some(mut device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    copy_probe::copy_of_unequal_lengths_is_refused(&mut device);
}
