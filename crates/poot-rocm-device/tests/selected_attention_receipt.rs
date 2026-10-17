//! The perf receipt of the POOT-1143 kernels on ROCm (HSA dispatch time, not host wall); see the wgpu twin
//! (`poot-wgpu-device/tests/selected_attention_receipt.rs`) for what it times. Prints the TSV lines of
//! `poot_executor_parity::selected_attention::print_receipt` and asserts that every fixture was measured.

use poot_executor::{Device, Engine};
use poot_executor_parity::selected_attention::{print_attention_receipt, print_top_k_receipt};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

/// Mutation: none (a receipt); the row goes red if the device reports no timestamps for a step.
#[test]
fn block_top_k_and_selected_attention_device_times() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new_with_device_timing())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let mut lines = print_top_k_receipt(&mut engine, target, "rocm");
    lines += print_attention_receipt(&mut engine, target, "rocm", 32_768, true);
    lines += print_attention_receipt(&mut engine, target, "rocm", 131_072, true);
    assert_eq!(lines, 7);
}
