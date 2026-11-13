//! POOT-1072: device-side timing reachable from outside the crate. `PtxDevice::new_with_device_timing`
//! is the public constructor, and `StepOutputs::device_time` reports a nonzero CUDA-event span for a
//! known kernel (the delta-rule scan); the plain `PtxDevice::new` reports the typed `Unknown`.

use std::time::Duration;

use poot_executor::{DetailedTiming, Device, DeviceTime, Engine, TimingOptions};
use poot_executor_parity::commit_labels::run_dispatch_labels_survive_the_stage_commit;
use poot_executor_parity::scan::device_time_of_step;
use poot_graph_ir::test_support::scan::ScanFixture;
use poot_ptx_device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

fn scan() -> ScanFixture {
    ScanFixture {
        len: 64,
        ..ScanFixture::delta_rule(2, 0, 128, 64)
    }
}

/// A timing-enabled device reads a nonzero event-pair span. Mutation: have `PtxDevice::device_time`
/// report `Duration::ZERO` for the span; the nonzero assertion goes red.
#[test]
fn timed_ptx_device_reads_a_nonzero_device_time_for_a_known_kernel() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new_with_device_timing()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let DeviceTime::Measured(measured) = device_time_of_step(&mut engine, target, &scan()) else {
        panic!("a timing-enabled ptx device must report Measured");
    };
    let span = measured.device_span.expect("span");
    assert!(span > Duration::ZERO, "device span is zero");
}

/// A device built without timing reports `Unknown`, never a fabricated host wall time. Mutation: have
/// `PtxDevice::device_time` return `Measured` from a host clock when no span is pending; red.
#[test]
fn plain_ptx_device_reports_unknown_device_time() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_eq!(
        device_time_of_step(&mut engine, target, &scan()),
        DeviceTime::Unknown
    );
}

/// A timed device reports each dispatch's own duration, contiguous in submission order, and they add up to the
/// step's device span: the marks the capture recorded partition the replay. Mutation: have
/// `PtxContext::mark_capture` record nothing; the dispatch list is empty and the row goes red.
#[test]
fn timed_ptx_device_reads_each_dispatch_of_a_captured_step() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new_with_device_timing()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let DeviceTime::Measured(measured) = device_time_of_step(&mut engine, target, &scan()) else {
        panic!("a timing-enabled ptx device must report Measured");
    };
    assert!(
        !measured.dispatches.is_empty(),
        "a timed capture reports its dispatches, not only the span"
    );
    for (position, dispatch) in measured.dispatches.iter().enumerate() {
        assert_eq!(dispatch.index, position, "dispatches keep submission order");
        assert!(
            dispatch.duration > Duration::ZERO,
            "dispatch {position} has no duration"
        );
    }
    let sum: Duration = measured.dispatches.iter().map(|d| d.duration).sum();
    assert_eq!(measured.sum_of_dispatch_durations, Some(sum));
    assert!(
        sum <= measured.device_span.expect("span"),
        "the dispatch windows lie inside the step's span"
    );
}

/// POOT-1571: a step whose recording holds a state commit between two stages' dispatches times each dispatch under
/// its own plan label: the device gives no copy a dispatch index, so no row of the step reads `unknown`.
/// Mutation: in `PtxContext::copy_buffer` call `self.mark_capture()` after the memcpy; the commit takes a mark, so a row reads `unknown`.
#[test]
fn ptx_dispatch_times_keep_their_plan_labels_across_a_stage_commit() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new_with_device_timing()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::with_timing(
        device,
        TimingOptions::Detailed(DetailedTiming::every_step(16, 64, 4)),
    );
    run_dispatch_labels_survive_the_stage_commit(&mut engine, target).unwrap();
}
