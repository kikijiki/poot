//! POOT-725 SC-001 on wgpu: `Gather` with I32 indices (axis 0 and axis 1; F32 and I32 data; repeated and
//! edge indices) compiled for the device and run through the executor contract equals the CPU oracle and the
//! plain-Rust ground truth word for word ([`poot_test_util::i32_gather`]).
//!
//! BF16 table consts are packed `u32` lanes on wgpu, which a plain `Gather` has no body for (a typed
//! `DtypeLowering` refusal, `bf16_const_reads_refuse_a_reader_without_the_lane`), so the BF16 cases run on
//! the backends with a native BF16 lane.
//!
//! Mutation (never left in the tree): read the index param as `Slice<f32>` in the movement generator
//! (`poot-kernelgen/src/data_movement/generate.rs`, `reads_index` param type). Every case goes red: `compile`
//! refuses with `I32ReadAsF32` before any dispatch, and with that check removed the kernel reads index words as
//! denormals and gathers row 0.

use poot_executor::{Device, Engine};
use poot_gpu::device::WgpuDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn gather_with_i32_indices_matches_the_oracle_bit_for_bit_on_wgpu() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec = Engine::new(device);
    poot_executor_parity::i32_gather::check("wgpu", false, &mut exec, target);
}
