//! The byte-length contract of `Device::copy`, the same probe on every backend: a copy moves every byte of
//! its source whatever the length, including a length that is not a whole number of `u32` words (3 bf16
//! values are 6 bytes, 5 raw bytes are 5), and a copy between buffers of different byte capacity is a device
//! error. The destination starts as a pattern the copy must overwrite, so a copy that stops at the last whole
//! word, or does not run, leaves bytes behind and the read-back differs.

use poot_codegen::{KernelCache, Target as CodegenTarget};
use poot_executor::{CopyKernel, CopyMechanism, Device};
use poot_graph_plan::CompileLimits;
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

/// Hand `device` the engine's copy kernel when it copies with one ([`CopyMechanism::Kernel`]), compiled for its
/// target the way the engine compiles it; a device that copies natively is left alone. A test that drives a
/// device without an engine calls this once before its first `copy`.
pub fn load_copy_kernel<D: Device>(device: &mut D) {
    let target_id = device.target().id();
    let CopyMechanism::Kernel(mut sink) = device.copy_mechanism() else {
        return;
    };
    let target = CodegenTarget::try_from(target_id).expect("a kernel-copy device runs bodies");
    let body = CopyKernel::body();
    let artifact = KernelCache::open(target)
        .load_or_compile(body, CompileLimits::STANDARD.max_artifact_bytes)
        .expect("compile the copy kernel");
    let compiled =
        poot_codegen::kernel_handle(body, &artifact).expect("the artifact is the body's");
    sink.load(CopyKernel::new(compiled))
        .expect("load the copy kernel");
}

/// Copy a buffer of `elems` elements of `storage` onto a buffer of the same shape through the production
/// `begin`/`copy`/`finish`/`replay`, and return the destination's bytes.
fn copied_bytes<D: Device>(
    device: &mut D,
    storage: BufferStorage,
    elems: usize,
    bytes: usize,
) -> Vec<u8> {
    let src = device
        .allocate(BufferRole::Activation, storage, elems)
        .expect("allocate src");
    let dst = device
        .allocate(BufferRole::Activation, storage, elems)
        .expect("allocate dst");
    device
        .write_with(&src, 0, bytes, &mut |range, out| {
            for (offset, byte) in out.iter_mut().enumerate() {
                *byte = 1 + (range.start + offset) as u8;
            }
        })
        .expect("write src");
    device
        .write_with(&dst, 0, bytes, &mut |_, out| out.fill(0xEE))
        .expect("write dst");
    device.begin().expect("begin");
    device.copy(&src, &dst).expect("copy");
    let recording = device.finish().expect("finish");
    device.begin_step().expect("begin_step");
    device.replay(&recording).expect("replay");
    device.synchronize().expect("synchronize");
    let mut read = vec![0u8; bytes];
    device.read(&dst, &mut read).expect("read dst");
    read
}

/// Copy buffers whose byte length is not a multiple of 4 and check every byte arrived.
pub fn odd_width_copy_is_exact<D: Device>(device: &mut D) {
    load_copy_kernel(device);
    for (storage, elems) in [
        (BufferStorage::bf16(), 3),
        (BufferStorage::bf16(), 1),
        (BufferStorage::raw_bytes(), 5),
        (BufferStorage::raw_bytes(), 7),
    ] {
        let bytes = elems * storage.element().byte_width();
        let expected: Vec<u8> = (0..bytes).map(|i| 1 + i as u8).collect();
        assert_eq!(
            copied_bytes(device, storage, elems, bytes),
            expected,
            "a copy of {bytes} bytes ({elems} x {:?}) moves every byte",
            storage.element()
        );
    }
}

/// A copy between buffers that hold different numbers of bytes is refused with a device error, and the
/// recording it was in can still be abandoned.
pub fn copy_of_unequal_lengths_is_refused<D: Device>(device: &mut D) {
    load_copy_kernel(device);
    let src = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 8)
        .expect("allocate src");
    let dst = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 16)
        .expect("allocate dst");
    device.begin().expect("begin");
    let refused = device.copy(&src, &dst);
    assert!(
        refused.is_err(),
        "a 32-byte buffer does not copy into a 64-byte one"
    );
    device.abort().expect("abort");
}
