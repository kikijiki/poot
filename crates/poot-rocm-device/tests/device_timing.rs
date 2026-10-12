//! POOT-1079: ROCm device-side timing reachable from outside the crate. `RocmDevice::new_with_device_timing`
//! is the public constructor, and `StepOutputs::device_time` reports a nonzero HSA dispatch time for a
//! known kernel (the delta-rule scan); the plain `RocmDevice::new` reports the typed `Unknown`, and
//! timing leaves the step's result unchanged.

use poot_executor::{Device, DeviceTime, Engine};
use poot_executor_parity::scan::{device_time_of_step, timed_step};
use poot_graph_ir::test_support::scan::ScanFixture;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

fn scan() -> ScanFixture {
    ScanFixture {
        len: 64,
        ..ScanFixture::delta_rule(2, 0, 128, 64)
    }
}

/// A timing-enabled device reads a nonzero sum and span, and lists the dispatches. Mutation: have
/// `RocmDevice::device_time` report `Duration::ZERO` for both measures (or have `read_dispatch_span`
/// return a zero span); the nonzero assertions go red.
#[test]
fn timed_rocm_device_reads_a_nonzero_device_time_for_a_known_kernel() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new_with_device_timing())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let DeviceTime::Measured(measured) = device_time_of_step(&mut engine, target, &scan()) else {
        panic!("a timing-enabled rocm device must report Measured");
    };
    let sum = measured.sum_of_dispatch_durations.expect("sum");
    let span = measured.device_span.expect("span");
    assert!(!sum.is_zero(), "sum of dispatch durations is zero");
    assert!(!span.is_zero(), "device span is zero");
    assert!(
        span >= measured
            .dispatches
            .iter()
            .map(|d| d.duration)
            .max()
            .unwrap(),
        "the span covers its longest dispatch"
    );
    assert!(
        !measured.dispatches.is_empty(),
        "a timed step lists its dispatches"
    );
}

/// A device built without timing reports `Unknown`, never a fabricated zero or host wall time.
/// Mutation: have `RocmDevice::device_time` answer `Measured(Default::default())` when no span was
/// recorded; this row then sees `Measured` and goes red.
#[test]
fn plain_rocm_device_reports_unknown_device_time() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_eq!(
        device_time_of_step(&mut engine, target, &scan()),
        DeviceTime::Unknown
    );
}

/// Profiling must not change results: the timed and untimed devices produce the identical packed
/// result for the same scan. Mutation: have `RocmDevice::replay` drop its first dispatch when the
/// context is timed; the timed result reads all zeros and differs from the untimed one.
#[test]
fn timing_does_not_change_the_rocm_scan_result() {
    let Some(timed) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new_with_device_timing())
    else {
        return;
    };
    let Some(plain) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let (timed_target, plain_target) = (timed.target(), plain.target());
    let (time, with_timing) = timed_step(&mut Engine::new(timed), timed_target, &scan());
    let (_, without) = timed_step(&mut Engine::new(plain), plain_target, &scan());
    assert!(matches!(time, DeviceTime::Measured(_)));
    assert_eq!(with_timing, without);
}
