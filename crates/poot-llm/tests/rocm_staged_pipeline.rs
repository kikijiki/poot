//! A staged pipeline on ROCm/AMD (POOT-745): the same rows as the wgpu and PTX twins
//! (`poot_models::test_support::pipeline`). A program cut into stages on one device agrees with the
//! one-stage program on every output and carried buffer, writes the same host bytes per step, and
//! records each stage once. `RocmDevice::replay` only enqueues, so a three-stage step waits on the
//! host once, counted at the HSA completion-signal wait and not only at `Device::synchronize`.
//!
//! Skips cleanly with no AMD GPU/HSA runtime.

use poot_executor::{Device, Engine};
use poot_models::test_support::pipeline::{
    Agreement, Counting, assert_export_fixture_matches_single_stage, assert_pipeline_waits_once,
    assert_qwen2_pipeline_matches_single_stage,
};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn qwen2_pipeline_cut_after_layers_zero_and_one_matches_the_single_stage_on_rocm() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_qwen2_pipeline_matches_single_stage(&mut engine, target, Agreement::BitIdentical);
}

#[test]
fn exported_boundaries_reach_every_consumer_on_rocm() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_export_fixture_matches_single_stage(&mut engine, target, Agreement::BitIdentical);
}

/// A three-stage step waits for the device once, where the single stage also waits once. Counted twice:
/// at `Device::synchronize` and at the HSA completion-signal wait of an enqueued batch, which is
/// where a `replay` that blocked on its own batch would show.
///
/// Mutation observed red (2026-10-09): `RocmContext::enqueue` ending in
/// `self.complete_pending(tag)?` (block per batch again) fails with "the 3-stage step waits on the
/// device once, not once per stage" (left: 3, right: 1).
#[test]
fn a_pipeline_step_waits_once_on_rocm() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let batch_waits = device.context().batch_waits();
    let (device, counts) = Counting::new(device);
    let mut engine = Engine::new(device);
    assert_pipeline_waits_once(
        &mut engine,
        &counts,
        &|| batch_waits.load(std::sync::atomic::Ordering::Relaxed),
        target,
        Agreement::BitIdentical,
    );
}
