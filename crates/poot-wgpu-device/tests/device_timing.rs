//! POOT-1072: device-side timing reachable from outside the crate. `WgpuDevice::new_with_device_timing`
//! is the public constructor, and `StepOutputs::device_time` reports a nonzero device time for a known
//! kernel (the delta-rule scan); the counters-only `WgpuDevice::new` reports the typed `Unknown`.

use std::num::NonZeroUsize;

use poot_executor::{Device, DeviceTime, Engine};
use poot_executor_parity::scan::device_time_of_step;
use poot_graph_ir::test_support::scan::ScanFixture;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_wgpu_device::device::WgpuDevice;

fn scan() -> ScanFixture {
    ScanFixture {
        len: 64,
        ..ScanFixture::delta_rule(2, 0, 128, 64)
    }
}

/// A timing-enabled device reads a nonzero sum and span for the scan's dispatches. Mutation: have
/// `WgpuDevice::device_time` return `Duration::ZERO` for both measures; the nonzero assertions go red.
#[test]
fn timed_wgpu_device_reads_a_nonzero_device_time_for_a_known_kernel() {
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
    let DeviceTime::Measured(measured) = device_time_of_step(&mut engine, target, &scan()) else {
        panic!("a timing-enabled wgpu device must report Measured");
    };
    let sum = measured.sum_of_dispatch_durations.expect("sum");
    let span = measured.device_span.expect("span");
    assert!(!sum.is_zero(), "sum of dispatch durations is zero");
    assert!(!span.is_zero(), "device span is zero");
    assert!(
        !measured.dispatches.is_empty(),
        "a timed step lists its dispatches"
    );
}

/// The counters-only device has no timestamps: `Unknown`, never a fabricated zero. Mutation: have
/// `WgpuDevice::device_time` answer `Measured(Default::default())` when nothing was drained; this row
/// then sees `Measured` and goes red.
#[test]
fn counters_only_wgpu_device_reports_unknown_device_time() {
    let Some(device) = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    assert_eq!(
        device_time_of_step(&mut engine, target, &scan()),
        DeviceTime::Unknown
    );
}
