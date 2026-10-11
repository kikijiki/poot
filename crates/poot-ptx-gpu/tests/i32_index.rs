//! POOT-725 SC-001 on PTX: `Gather` with I32 indices (axis 0 and axis 1; F32, BF16 and I32 data; repeated
//! and edge indices) compiled for the device and run through the executor contract equals the CPU oracle and
//! the plain-Rust ground truth word for word ([`poot_test_util::i32_gather`]). Runs on a pod
//! (`just test-device-ptx`).

use poot_executor::{Device, Engine};
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn gather_with_i32_indices_matches_the_oracle_bit_for_bit_on_ptx() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec = Engine::new(device);
    poot_executor_parity::i32_gather::check("ptx", true, &mut exec, target);
}
