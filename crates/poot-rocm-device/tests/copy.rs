//! POOT-1235: `Device::copy` on Rocm, driven through the production `begin`/`copy`/`finish`/`replay`. The
//! probes are `poot_executor_parity::copy_probe`, the same on every backend: a copy of a byte length that is
//! not a whole number of `u32` words arrives whole, and a copy between buffers of different byte lengths is
//! refused when recorded.

use poot_executor_parity::copy_probe;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

/// SC-001. Mutation: stop rounding ROCm allocations to whole words (`allocate_uninit`'s `next_multiple_of(4)`
/// in `poot-rocm-runtime`); the 6-byte buffer is then refused as `CopyNotWordAligned` and this goes red. (Restoring
/// the old multiple-of-4 refusal alone is inert: with the rounding in place no buffer reaches it.)
#[test]
fn copy_of_an_odd_byte_length_is_exact_on_rocm() {
    let Some(mut device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    copy_probe::odd_width_copy_is_exact(&mut device);
}

/// A copy between buffers of different byte lengths is a device error at record time, not at replay.
/// Mutation: drop the byte-capacity comparison in `RocmDevice::copy`.
#[test]
fn copy_between_unequal_lengths_is_refused_on_rocm() {
    let Some(mut device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    copy_probe::copy_of_unequal_lengths_is_refused(&mut device);
}

/// SC-002: misuse of the recording transaction is a typed error, never a panic, and leaves the device
/// usable. Mutation: restore `assert!(self.open.is_none(), ..)` in `RocmDevice::begin` (the second `begin`
/// panics) or `.expect("finish without begin")` in `finish`.
#[test]
fn transaction_misuse_is_a_typed_error_on_rocm() {
    use poot_executor::Device;
    use poot_rocm_device::RocmGpuError;
    use poot_runtime_common::BufferRole;
    use poot_target::BufferStorage;

    let Some(mut device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
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
        Err(RocmGpuError::NotRecording { op: "copy" })
    ));
    assert!(matches!(
        device.finish(),
        Err(RocmGpuError::NotRecording { op: "finish" })
    ));

    device.begin().expect("begin");
    assert!(matches!(
        device.begin(),
        Err(RocmGpuError::AlreadyRecording)
    ));
    assert!(
        matches!(
            device.copy(&src, &dst),
            Err(RocmGpuError::CopyKernelMissing)
        ),
        "a copy before the engine's kernel was loaded is refused"
    );
    // The refused calls left the open recording intact: it still finishes.
    device
        .finish()
        .expect("the open recording survived the misuse");
}
