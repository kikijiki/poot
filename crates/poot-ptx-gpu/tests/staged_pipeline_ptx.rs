//! A staged pipeline on PTX/NVIDIA (POOT-745): the same rows as the wgpu and ROCm twins
//! (`poot_executor_parity::pipeline`). A program cut into stages on one device agrees with the
//! one-stage program on every output and carried buffer, writes the same host bytes per step, and
//! records each stage once.
//!
//! Skips cleanly with no NVIDIA GPU.

use poot_executor::{Device, Engine};
use poot_executor_parity::pipeline::{
    Agreement, Counting, assert_export_fixture_matches_single_stage, assert_pipeline_waits_once,
    assert_qwen2_pipeline_matches_single_stage,
};
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn qwen2_pipeline_cut_after_layers_zero_and_one_matches_the_single_stage_on_ptx() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_qwen2_pipeline_matches_single_stage(&mut engine, target, Agreement::BitIdentical);
}

#[test]
fn exported_boundaries_reach_every_consumer_on_ptx() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_export_fixture_matches_single_stage(&mut engine, target, Agreement::BitIdentical);
}

#[test]
fn a_pipeline_step_waits_once_on_ptx() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let target = device.target();
    let (device, counts) = Counting::new(device);
    let mut engine = Engine::new(device);
    assert_pipeline_waits_once(
        &mut engine,
        &counts,
        &|| counts.waits.get(),
        target,
        Agreement::BitIdentical,
    );
}
