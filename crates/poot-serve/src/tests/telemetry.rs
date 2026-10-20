//! The executor-timing export (POOT-590): what the engine loop publishes to the metrics and how `/metrics`
//! renders it. Counters are exact, device time is `null`/absent when no device measured it, and the
//! detail coverage says whether the retained records are everything the step count implies.

use std::time::Duration;

use poot_profile::{DeviceCoverage, HostTiming, MeasuredDevice, TimingSnapshot};
use poot_runtime_common::Coverage;

use super::recording_engine::{JobSpec, RecordingEngine};
use super::*;
use crate::metrics::TimingExport;

fn host(wall_ms: u64) -> HostTiming {
    HostTiming {
        encode: Duration::from_millis(1),
        wait: Duration::from_millis(2),
        wall: Duration::from_millis(wall_ms),
    }
}

fn measured(ms: u64) -> DeviceCoverage {
    DeviceCoverage::Measured(MeasuredDevice {
        sum_of_dispatch_durations: Some(Duration::from_millis(ms)),
        device_span: Some(Duration::from_millis(ms)),
    })
}

/// A snapshot of the SPIR-V target that keeps `ring` steps and `records` dispatch records.
fn empty_snapshot(ring: usize, records: usize) -> TimingSnapshot {
    TimingSnapshot::new(
        poot_target::TargetId::Gpu(poot_target::Backend::SpirvVulkan),
        poot_profile::BuildIdentity::current(),
        ring,
        records,
    )
}

/// `steps` steps of one entry, each with one dispatch of detail, into a ring that keeps `ring` steps.
fn snapshot(ring: usize, steps: u64, device: DeviceCoverage) -> TimingSnapshot {
    use poot_profile::DispatchTiming;
    let mut snapshot = empty_snapshot(ring, ring * 4);
    for replay in 0..steps {
        let dispatches = if ring == 0 {
            Vec::new()
        } else {
            vec![DispatchTiming {
                dispatch_index: 0,
                kind_name: "matmul",
                plan_label: None,
                device: matches!(device, DeviceCoverage::Measured(_))
                    .then(|| Duration::from_millis(3)),
            }]
        };
        snapshot.record(1, replay, host(10), device, dispatches);
    }
    snapshot
}

fn served_timing(timing: Option<TimingSnapshot>, telemetry: TelemetryConfig) -> Arc<Metrics> {
    let mut engine = RecordingEngine::new(1);
    engine.timing = timing;
    let mut metrics = test_metrics("timing");
    metrics.timing = TimingMetrics::new(telemetry);
    let metrics = Arc::new(metrics);
    let (tx, rx) = mpsc::channel();
    let (job, submitted) = JobSpec::new("hello", 2).build();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    drop(tx);
    assert_eq!(run_engine(&mut engine, &rx, &metrics), Exit::Shutdown);
    assert!(matches!(
        submitted.replies.try_iter().last(),
        Some(GenEvent::Done(..))
    ));
    metrics
}

/// Counters only: no step is measured and no detail retained. The export says so: device time is
/// absent (Unknown, not zero), detail is unavailable, nothing is perturbed. Mutation: export
/// `device_sum.unwrap_or_default()` as a number; the JSON `device_time_ms` becomes `0.0` and this goes red.
#[test]
fn counters_only_exports_unknown_device_time_not_zero() {
    let metrics = served_timing(
        Some(snapshot(0, 6, DeviceCoverage::Unknown)),
        TelemetryConfig::CountersOnly,
    );
    let export = metrics.timing.export();
    assert_eq!(
        export.steps, 6,
        "cumulative counts are exact in counters mode"
    );
    assert_eq!(export.device_sum, None);
    assert_eq!(export.detail, Coverage::Unavailable);
    assert!(!export.perturbed);

    let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
    let timing = &json["device_timing"];
    assert_eq!(timing["mode"], "counters_only");
    assert_eq!(timing["steps"], 6);
    assert!(timing["device_time_ms"].is_null(), "got {timing}");
    assert_eq!(timing["detail_coverage"]["kind"], "unavailable");
    assert!(
        !metrics
            .prometheus()
            .contains("poot_executor_device_seconds_total"),
        "an unmeasured device exports no device-time series"
    );
}

/// Detailed mode over a ring smaller than the run: the step count and the device sum stay exact, the
/// coverage reports the truncation and the perturbation, and the retained detail never exceeds its
/// bound however long the run. Mutation: report `Coverage::Complete` for a truncated window.
#[test]
fn detailed_export_reports_truncated_coverage_and_perturbation() {
    let detail = poot_executor::DetailedTiming::every_step(4, 16, 2);
    let metrics = served_timing(
        Some(snapshot(4, 50, measured(5))),
        TelemetryConfig::Detailed(detail),
    );
    let export = metrics.timing.export();
    assert_eq!(export.steps, 50);
    assert_eq!(export.device_sum, Some(Duration::from_millis(250)));
    assert_eq!(
        export.detail,
        Coverage::Truncated {
            retained: 4,
            dropped: 46
        }
    );
    assert!(export.perturbed);

    let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
    let timing = &json["device_timing"];
    assert_eq!(timing["mode"], "detailed");
    assert_eq!(timing["device_time_ms"], 250.0);
    assert_eq!(timing["detail_coverage"]["kind"], "truncated");
    assert_eq!(timing["detail_coverage"]["retained"], 4);
    assert_eq!(timing["detail_coverage"]["dropped"], 46);
    assert_eq!(timing["perturbed"], true);
    let prometheus = metrics.prometheus();
    assert!(prometheus.contains("poot_executor_steps_total 50"));
    assert!(prometheus.contains("detail_coverage=\"truncated\""));
    assert!(prometheus.contains("poot_executor_device_seconds_total 0.25"));
}

/// A step the sampling policy skipped reports `Unknown` for itself; the window is `sampled`, not
/// complete. Mutation: count unmeasured steps as sampled.
#[test]
fn sampled_detail_is_reported_as_sampled() {
    let mut snapshot = empty_snapshot(16, 64);
    for replay in 0..6u64 {
        let device = if replay % 3 == 0 {
            measured(2)
        } else {
            DeviceCoverage::Unknown
        };
        snapshot.record(1, replay, host(10), device, Vec::new());
    }
    assert_eq!(
        TimingExport::of(&snapshot).detail,
        Coverage::Unsampled {
            sampled: 2,
            total: 6
        }
    );
}

/// An engine with no executor publishes nothing: the export stays the empty one.
#[test]
fn an_engine_without_timing_publishes_nothing() {
    let metrics = served_timing(None, TelemetryConfig::CountersOnly);
    let export = metrics.timing.export();
    assert_eq!(export.steps, 0);
    assert_eq!(export.detail, Coverage::Unavailable);
}

/// The engine thread copies the executor's retained timing and summarizes it every publish. At the most
/// the flags accept (16384 steps, 262144 dispatch records) that work stays cheap: the bound that makes
/// detailed timing a bounded diagnostic and not a service cost. The budget is a generous 250 ms in a
/// debug build, the publish interval; a release build measures a small fraction of it. Mutation: raise a
/// maximum in `config.rs` tenfold and fill the ring to it; the copy plus summary crosses the budget.
#[test]
fn publishing_the_largest_accepted_detail_stays_cheap() {
    use poot_profile::DispatchTiming;

    const STEPS: u64 = 1 << 14;
    const RECORDS_PER_STEP: usize = 16; // 16384 steps x 16 = 262144 records, the dispatch maximum
    let mut snapshot = empty_snapshot(STEPS as usize, STEPS as usize * RECORDS_PER_STEP);
    for replay in 0..STEPS {
        let dispatches = (0..RECORDS_PER_STEP)
            .map(|dispatch_index| DispatchTiming {
                dispatch_index,
                kind_name: "matmul",
                plan_label: None,
                device: Some(Duration::from_micros(3)),
            })
            .collect();
        snapshot.record(1, replay, host(1), measured(1), dispatches);
    }
    let started = Instant::now();
    let copy = snapshot.clone();
    let export = TimingExport::of(&copy);
    let cost = started.elapsed();
    assert_eq!(export.steps, STEPS);
    assert_eq!(export.detail, Coverage::Complete);
    eprintln!("publish cost at the largest accepted detail: {cost:?}");
    assert!(
        cost < Duration::from_millis(250),
        "copying and summarizing the largest accepted detail took {cost:?}"
    );
}
