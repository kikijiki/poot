//! Device timing across the stages of one step (POOT-745, SC-006), on the device twins the staged
//! pipeline rows share (`poot_executor_parity::pipeline`). One `replay` per stage, one timing per
//! step: a three-stage step reports every stage's dispatches (wgpu, per dispatch) or one span around
//! all its launches (PTX), never only the last stage's.
//!
//! The model-free half is `poot-executor`'s `staged_pipeline` fake-device row.

use std::num::NonZeroUsize;

use poot_executor::{DetailedTiming, Device, DrainPolicy, Engine, TimingOptions};
use poot_executor_parity::pipeline::{
    Counting, assert_detailed_timing_reports_every_stage,
    assert_pipeline_step_reports_a_device_span,
};
use poot_gpu::device::WgpuDevice;
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;

/// Mutation: discard the device timing in `WgpuDevice::replay`, as before the step hook; only the
/// last stage's records survive and the pipeline's per-kind dispatch counts fall short.
#[test]
fn a_detailed_timing_pipeline_step_reports_every_stage_wgpu() {
    const QUERIES: usize = 64;
    let Some(device) = open_or_skip(
        DeviceBackend::Wgpu,
        WgpuDevice::new_with_device_timing(NonZeroUsize::new(QUERIES).unwrap()),
    ) else {
        return;
    };
    if !device.has_device_timestamps() {
        eprintln!("skip: the adapter has no TIMESTAMP_QUERY");
        return;
    }
    let target = device.target();
    let (device, counts) = Counting::new(device);
    let mut engine = Engine::with_timing(
        device,
        TimingOptions::Detailed(DetailedTiming {
            sample_every: 1,
            max_retained_steps: 8,
            max_retained_dispatch_records: 4096,
            max_in_flight_queries: QUERIES,
            drain_policy: DrainPolicy::WaitForCapacity,
        }),
    );
    assert_detailed_timing_reports_every_stage(&mut engine, &counts, target);
}

/// Mutation: have `PtxDevice::begin_step` skip clearing the spans, or `replay` keep only the latest
/// span; the span then covers a single stage (or a stale step) and the nonzero check still holds, so
/// the engine's fake-device row is the one that pins accumulation.
#[test]
fn a_timed_pipeline_step_reports_a_device_span_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new_with_device_timing()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_pipeline_step_reports_a_device_span(&mut engine, target);
}
