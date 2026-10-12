//! POOT-725 SC-001 on ROCm: `Gather` with I32 indices (axis 0 and axis 1; F32, BF16 and I32 data; repeated
//! and edge indices) compiled for the device and run through the executor contract equals the CPU oracle and
//! the plain-Rust ground truth word for word ([`poot_eval::test_support::i32_gather`]).
//!
//! Mutation (never left in the tree): read the index param as `Slice<f32>` in the movement generator
//! (`poot-kernelgen/src/data_movement/generate.rs`, `reads_index` param type). Every case goes red.

use poot_executor::{Device, Engine};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn gather_with_i32_indices_matches_the_oracle_bit_for_bit_on_rocm() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec = Engine::new(device);
    poot_executor_parity::i32_gather::check("rocm", true, &mut exec, target);
}
