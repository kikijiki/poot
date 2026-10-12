//! The batched-LoRA compile pass on ROCm/AMD (POOT-579, spec 248 phase 2, epic 129 B8): the same row as
//! the wgpu and PTX twins (`poot_executor_parity::lora`): each of the seven targetable projections is a
//! graph compiled with a `LoraPlan`, two adapters on rows 0 and 1 and none on row 2, at one and two tokens
//! per row, every row against the host's merged-weight reference within `1e-3` relative.
//!
//! Skips cleanly with no AMD GPU/HSA runtime.

use poot_executor::{Device, Engine};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn batched_lora_decode_rocm_matches_per_row_merged_weight_reference() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
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
