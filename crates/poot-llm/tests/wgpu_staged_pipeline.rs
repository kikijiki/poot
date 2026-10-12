//! A staged pipeline on wgpu (POOT-745): the same rows as the ROCm and PTX twins
//! (`poot_models::test_support::pipeline`). A program cut into stages on one device agrees with the
//! one-stage program on every output and carried buffer, writes the same host bytes per step, and
//! records each stage once. Twins: `rocm_staged_pipeline.rs`,
//! `ptx_staged_pipeline.rs`.

use poot_executor::{Device, Engine};
use poot_graph_plan::Target;
use poot_models::test_support::pipeline::{
    Agreement, Counting, assert_export_fixture_matches_single_stage, assert_pipeline_waits_once,
    assert_qwen2_pipeline_matches_single_stage,
};
use poot_wgpu_device::device::WgpuDevice;

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// The wgpu device behind an `Engine`, with the target it compiles for read before the device moves, or `None`
/// (a skip) without an adapter.
fn open_engine_or_skip() -> Option<(Engine<WgpuDevice>, Target)> {
    let device = poot_runtime_common::device_skip::open_or_skip(
        poot_runtime_common::DeviceBackend::Wgpu,
        WgpuDevice::new(),
    )?;
    let target = device.target();
    Some((Engine::new(device), target))
}

#[test]
fn qwen2_pipeline_cut_after_layers_zero_and_one_matches_the_single_stage_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };
    assert_qwen2_pipeline_matches_single_stage(&mut exec, target, Agreement::BitIdentical);
}

#[test]
fn exported_boundaries_reach_every_consumer_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };
    assert_export_fixture_matches_single_stage(&mut exec, target, Agreement::BitIdentical);
}

#[test]
fn a_pipeline_step_waits_once_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some(device) = poot_runtime_common::device_skip::open_or_skip(
        poot_runtime_common::DeviceBackend::Wgpu,
        WgpuDevice::new(),
    ) else {
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
