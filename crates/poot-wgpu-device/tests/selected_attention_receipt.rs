//! The perf receipt of the POOT-1143 kernels on wgpu (device timestamps, not host wall): the block top-k
//! radix select at 2048, 8192 and 32768 blocks (the design's three-pass claim against this kernel's seven
//! global-read passes) and the selected attention against the dense attention over the same deep decode (24
//! query heads over 2 key/value heads of 256, 128K stored positions, 512 blocks of four selected). The test
//! prints the TSV lines of `poot_executor_parity::selected_attention::print_receipt` and asserts only that the
//! device measured every fixture; the numbers are recorded in
//! `benchmarks/performance-bank/candidates/POOT-1143`.

use std::num::NonZeroUsize;

use poot_executor::{Device, Engine};
use poot_executor_parity::selected_attention::{
    print_attention_receipt, print_prefill_receipt, print_top_k_receipt,
};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_wgpu_device::device::WgpuDevice;

/// Mutation: none (a receipt); the row goes red if the device reports no timestamps for a step.
#[test]
fn block_top_k_and_selected_attention_device_times() {
    let Some(device) = open_or_skip(
        DeviceBackend::Wgpu,
        WgpuDevice::new_with_device_timing(NonZeroUsize::new(64).unwrap()),
    ) else {
        return;
    };
    if !device.has_device_timestamps() {
        poot_runtime_common::device_skip::skip_unsupported(DeviceBackend::Wgpu, "TIMESTAMP_QUERY");
        return;
    }
    let target = device.target();
    let mut engine = Engine::new(device);
    let mut lines = print_top_k_receipt(&mut engine, target, "wgpu");
    // The dense attention over 128K positions lost the RADV context on this box (see the receipt), so the dense
    // row is timed at 32K only; the selected attention is timed at both depths.
    lines += print_attention_receipt(&mut engine, target, "wgpu", 32_768, true);
    lines += print_attention_receipt(&mut engine, target, "wgpu", 131_072, false);
    // POOT-1338: the chunk of a long prompt, at the 8K of the benchmark and at 32K, where it is flat.
    lines += print_prefill_receipt(&mut engine, target, "wgpu", 8_192);
    lines += print_prefill_receipt(&mut engine, target, "wgpu", 32_768);
    assert_eq!(lines, 8);
}
