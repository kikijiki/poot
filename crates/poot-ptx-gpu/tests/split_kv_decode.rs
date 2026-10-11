//! POOT-1113 on PTX (POOT-1124): the split-KV decode attention through the executor contract.
//!
//! One fixture serves every backend: [`poot_executor_parity::attention`] holds the low-head live decode
//! (2 query rows over a 1024-key cache of one key/value head), its live lengths and its tolerance, and
//! the wgpu, ROCm and Vulkan device rows call the same two helpers. The planner must pick the split plan
//! on the device's own caps; a device that plans the streaming schedule fails the row with the planned
//! launch counts rather than passing on the wrong kernel.

use poot_executor::{Device, Engine, Executor};
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;

/// The split-KV decode stepped at live lengths 1024, 5, 400, 1, 1024 over one executable matches the CPU
/// oracle's evaluation of each length within `SPLIT_KV_DEVICE_TOLERANCE`. The scratch slot is stale from
/// the previous step at every step but the first (length 5 leaves seven of the eight splits with no key),
/// so a split that did not publish its own neutral partial reads the first step's.
///
/// Mutation: in `poot-kernelgen`'s `attention_body`, skip the partial writes of a split with no key; the
/// length-5 and length-1 steps go red. Dropping one split's partial in the combine (`count` starting at
/// 1) goes red on every step.
#[test]
fn split_kv_decode_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::attention::assert_split_kv_decode_matches_the_oracle(
        exec.as_mut(),
        target,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

/// The split-KV decode stepped through its live lengths twice over one entry: the second pass replays
/// the captured two-launch plan as a CUDA graph, and every step's output equals the first pass's bit for
/// bit (the first pass's first step is the recording step's own execution).
///
/// This row guards pass-to-pass consistency of the captured plan, not the kernel: a fault that is the same on
/// every pass (the dropped-split mutations of the row above) leaves it green, and the row above is the one that
/// goes red for those.
///
/// Mutation: in `PtxDevice::replay`, count each recording's replays and launch nothing past the fifth (the first
/// pass's launches); the second pass differs from the first at the length-5 step ("split-KV replay of live
/// length 5 is not bit-identical to the first pass"). The count is per recording, so another row's replays in
/// the same process cannot consume it.
#[test]
fn split_kv_decode_replay_equals_the_recording_step_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::attention::assert_split_kv_decode_replays_bit_identically(
        exec.as_mut(),
        target,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}
