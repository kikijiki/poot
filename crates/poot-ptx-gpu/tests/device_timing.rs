//! POOT-1072: device-side timing reachable from outside the crate. `PtxDevice::new_with_device_timing`
//! is the public constructor, and `StepOutputs::device_time` reports a nonzero CUDA-event span for a
//! known kernel (the delta-rule scan); the plain `PtxDevice::new` reports the typed `Unknown`.

use std::time::Duration;

use poot_executor::{Device, DeviceTime, Engine};
use poot_executor_parity::scan::device_time_of_step;
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::scan::ScanFixture;

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
