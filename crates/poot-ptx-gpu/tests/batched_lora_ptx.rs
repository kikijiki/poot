//! The batched-LoRA compile pass on PTX/NVIDIA (POOT-579, spec 248 phase 2 PTX port, epic 129 B8): the
//! same row as the wgpu and ROCm twins (`poot_executor_parity::lora`): each of the seven targetable
//! projections is a graph compiled with a `LoraPlan`, two adapters on rows 0 and 1 and none on row 2, at
//! one and two tokens per row, every row against the host's merged-weight reference within `1e-3`
//! relative (CUDA `IndexedMatMul`/`Gather`/`MatMul` kernels through the executor contract).
//!
//! Skips cleanly with no NVIDIA GPU. Real-hardware runs go through RunPod as a compiled test binary
//! (`cargo test -p poot-ptx-gpu --test batched_lora_ptx --no-run`, then run the binary on the pod).

use poot_executor::{Device, Engine};
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn batched_lora_decode_ptx_matches_per_row_merged_weight_reference() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    poot_executor_parity::lora::assert_batched_lora_matches_merged_weights(
        &mut engine,
        target,
        1e-3,
    );
}
