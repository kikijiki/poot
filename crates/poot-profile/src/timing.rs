//! Typed, bounded device-time records and the windowed report (Card 552).
//!
//! `poot-executor` depends on this module to build [`TimingSnapshot`] from its own
//! per-step measurements and expose it through `ExecutorStats`; this module never depends back on
//! `poot-executor` (`poot-executor` depends on `poot-profile`, never the reverse).
//!
//! [`TimingSnapshot`] keeps two kinds of state with different lifetimes:
//! - exact, never-evicted cumulative counters per entry (step/dispatch counts, host-time sums, a
//!   device-duration sum when measured) - these answer "how many" and "how long in total" for any
//!   window, even one whose detail has since been evicted;
//! - a bounded ring of recent [`StepTiming`] detail (per-dispatch records, kind_name buckets), which
//!   a long run truncates rather than growing without bound (Card 552 SC-006).
//!
//! A [`Mark`] is a monotonic identity (a sequence number plus the cumulative counters observed at
//! that moment), not an index into the bounded ring: [`Report::window`] can always report exact
//! step/host/device-sum totals for a window, and separately and honestly reports whether the
//! per-dispatch detail for that window is still fully retained ([`Coverage`]).

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

pub use poot_runtime_common::Coverage;

/// Host wall time partitioned into the intervals the engine itself measures: encoding/submitting
/// work (`encode`), blocking on the device (`wait`), and the step's total wall (`wall`). `other` (the
/// unattributed residual) is always `wall - encode - wait`, computed, never independently measured
/// and never derived by subtracting device time (Card 552 SC-001): a backend whose device time
/// overlaps host wait time must never show up as a smaller `other`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostTiming {
    pub encode: Duration,
    pub wait: Duration,
    pub wall: Duration,
}

impl HostTiming {
    /// The residual `wall - encode - wait`, in nanoseconds, signed: small measurement/clock-skew
    /// noise can make `encode + wait` exceed `wall` by a few microseconds, and printing a clamped
    /// zero there would hide that noise rather than report it (Card 552 scope: "the signed residual
    /// printed", no clamp).
    pub fn other_nanos(&self) -> i64 {
        self.wall.as_nanos() as i64 - self.encode.as_nanos() as i64 - self.wait.as_nanos() as i64
    }

    fn accumulate(&mut self, other: HostTiming) {
        self.encode += other.encode;
        self.wait += other.wait;
        self.wall += other.wall;
    }
}

/// One dispatch's device-time record, retained only in detailed mode (Card 552: counters-only mode
/// keeps `StepTiming::dispatches` empty and reports [`Coverage::Unavailable`] for detail, never a
/// fabricated empty-but-"complete" list).
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchTiming {
    /// Order within its step's recording/replay (stable across replays of the same entry).
    pub dispatch_index: usize,
    /// The op's semantic report key (`poot_graph_ir::OpKind::kind_name`): the aggregation key for
    /// [`WindowReport::buckets`]. Never the planner's kernel key (Card 552 SC-003).
    pub kind_name: &'static str,
    /// An optional display-only label (the plan's kernel key or similar), filled by Card 560a.
    /// Printed as an extra column when present; never part of the bucket key.
    pub plan_label: Option<String>,
    /// This dispatch's own device duration, when the backend measured it.
    pub device: Option<Duration>,
}

/// One step's device-time coverage: either the device cannot report timestamps at all right now
/// (`Unknown`, never silently treated as a zero-duration measurement - Card 552 SC-004), or at least
/// one of the two independent measures below is available.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeasuredDevice {
    /// Sum of each dispatch's own `[start, end)` device duration. Can exceed `device_span` under
    /// device-side concurrency (a backend with overlapping queues); never collapsed into one number
    /// with `device_span`, and never compared against host wall as if it were the critical path
    /// (Card 552 SC-001/SC-005).
    pub sum_of_dispatch_durations: Option<Duration>,
    /// First dispatch's start to last dispatch's end, on this step's own device clock domain.
    pub device_span: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DeviceCoverage {
    #[default]
    Unknown,
    Measured(MeasuredDevice),
}

/// One `step()` call's full timing record: identity (which entry, which replay of it, which
/// device/backend), the host-wall partition, the device-time coverage, and (detailed mode only) the
/// per-dispatch breakdown.
#[derive(Clone, Debug, PartialEq)]
pub struct StepTiming {
    /// Monotonic identity assigned by [`TimingSnapshot::record`]; never reused, never an index into
    /// the bounded detail ring (a mark taken against this value survives the ring evicting this very
    /// step).
    pub seq: u64,
    /// The program this step belongs to (an opaque per-executable identity, e.g.
    /// `poot_executor::EntryId::raw()`); `poot-profile` assigns no semantic meaning to the value.
    pub entry: u64,
    /// The entry's replay sequence number this step corresponds to.
    pub replay: u64,
    /// Which device/backend produced this record (a debug label, e.g. `"SpirvVulkan"`).
    pub device_label: String,
    pub host: HostTiming,
    pub device: DeviceCoverage,
    /// Per-dispatch detail, in dispatch order; empty unless the collecting `TimingOptions` requested
    /// detailed per-dispatch retention for this step.
    pub dispatches: Vec<DispatchTiming>,
    /// Card 552: whether collecting this step's own detail changed submission grouping
    /// (e.g. one native submit per dispatch instead of one for the whole batch) - true exactly
    /// when `dispatches` is non-empty, since that is what requires per-dispatch attribution.
    /// `Report::window` surfaces this so a figure built from a perturbed window is never presented
    /// as unperturbed throughput evidence.
    pub perturbed: bool,
}

/// Cumulative, never-evicted counters for one entry: exact regardless of whatever detail the bounded
/// ring has since dropped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntryCounters {
    pub steps: u64,
    pub dispatches: u64,
    pub host: HostTiming,
    /// Sum of every step's `sum_of_dispatch_durations`, only across steps where it was measured;
    /// `None` if this entry has never had a measured step (Card 552 SC-004: never defaults to zero).
    pub device_sum: Option<Duration>,
}

impl EntryCounters {
    fn accumulate_step(&mut self, step: &StepTiming) {
        self.steps += 1;
        self.dispatches += step.dispatches.len() as u64;
        self.host.accumulate(step.host);
        if let DeviceCoverage::Measured(MeasuredDevice {
            sum_of_dispatch_durations: Some(d),
            ..
        }) = step.device
        {
            self.device_sum = Some(self.device_sum.unwrap_or_default() + d);
        }
    }

    /// The exact delta `self - earlier`: `self` must be a later (or equal) cumulative snapshot of
    /// the same entry.
    fn since(&self, earlier: &EntryCounters) -> EntryCounters {
        EntryCounters {
            steps: self.steps - earlier.steps,
            dispatches: self.dispatches - earlier.dispatches,
            host: HostTiming {
                encode: self.host.encode - earlier.host.encode,
                wait: self.host.wait - earlier.host.wait,
                wall: self.host.wall - earlier.host.wall,
            },
            device_sum: match (self.device_sum, earlier.device_sum) {
                (Some(a), Some(b)) => Some(a - b),
                (Some(a), None) => Some(a),
                (None, _) => None,
            },
        }
    }
}

/// A monotonic window boundary: the sequence number and the per-entry cumulative counters observed
/// at that moment (Card 552 scope: "window marks are monotonic identities, not indices into an
/// unbounded vector").
#[derive(Clone, Debug, Default)]
pub struct Mark {
    seq: u64,
    dropped_steps: u64,
    by_entry: BTreeMap<u64, EntryCounters>,
}

fn entry_counters_for(map: &BTreeMap<u64, EntryCounters>, entry: u64) -> EntryCounters {
    map.get(&entry).cloned().unwrap_or_default()
}

/// A bounded, typed timing collector (Card 552). Exact cumulative counters never evict; the detail
/// ring (`StepTiming`, retained for [`Report::window`]'s per-dispatch buckets) is bounded by
/// `max_retained_steps` and `max_retained_dispatch_records`, oldest evicted first.
#[derive(Clone, Debug, PartialEq)]
pub struct TimingSnapshot {
    cumulative: BTreeMap<u64, EntryCounters>,
    retained: VecDeque<StepTiming>,
    retained_dispatch_count: usize,
    dropped_steps: u64,
    next_seq: u64,
    max_retained_steps: usize,
    max_retained_dispatch_records: usize,
}

impl TimingSnapshot {
    /// `max_retained_steps == 0` is the counters-only configuration: every cumulative counter still
    /// works exactly, but no per-dispatch detail is ever retained ([`Coverage::Unavailable`] for
    /// every window's detail).
    pub fn new(max_retained_steps: usize, max_retained_dispatch_records: usize) -> Self {
        Self {
            cumulative: BTreeMap::new(),
            retained: VecDeque::new(),
            retained_dispatch_count: 0,
            dropped_steps: 0,
            next_seq: 0,
            max_retained_steps,
            max_retained_dispatch_records,
        }
    }

    /// Record one step's timing. `entry`/`replay`/`device_label` identify it; `dispatches` is empty
    /// unless detailed collection is on for this step. Evicts the oldest retained step(s) first when
    /// either bound would otherwise be exceeded; never evicts or truncates to make room for a step
    /// that has not finished (there is no such state here: a step is recorded only once complete).
    pub fn record(
        &mut self,
        entry: u64,
        replay: u64,
        device_label: impl Into<String>,
        host: HostTiming,
        device: DeviceCoverage,
        dispatches: Vec<DispatchTiming>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let perturbed = !dispatches.is_empty();
        let step = StepTiming {
            seq,
            entry,
            replay,
            device_label: device_label.into(),
            host,
            device,
            dispatches,
            perturbed,
        };
        self.cumulative
            .entry(entry)
            .or_default()
            .accumulate_step(&step);

        if self.max_retained_steps > 0 {
            // Bound the ring by step count first, independent of dispatch-record fitting: this
            // never evicts an unrelated already-retained step just because the *incoming* step's
            // own detail cannot fit (review F7).
            while self.retained.len() >= self.max_retained_steps {
                let Some(evicted) = self.retained.pop_front() else {
                    break;
                };
                self.retained_dispatch_count -= evicted.dispatches.len();
                self.dropped_steps += 1;
            }
            let mut step = step;
            if self.retained_dispatch_count + step.dispatches.len()
                > self.max_retained_dispatch_records
            {
                // Review F7: `max_retained_dispatch_records == 0` means "never retain per-dispatch
                // detail" and falls out of this same comparison (`0 + len > 0` whenever `len > 0`) -
                // no special case, and no longer silently treated as unbounded. A nonzero bound this
                // step's own detail cannot fit inside takes the same path: keep the step's identity/
                // host/device totals (the ring entry) but drop its per-dispatch detail rather than
                // growing past the configured bound.
                step.dispatches.clear();
                self.dropped_steps += 1;
            } else {
                self.retained_dispatch_count += step.dispatches.len();
            }
            self.retained.push_back(step);
        } else {
            self.dropped_steps += 1;
        }
        seq
    }

    /// A mark usable with [`Report::window`]: everything recorded after this call is "in" any
    /// window taken against it.
    pub fn mark(&self) -> Mark {
        Mark {
            seq: self.next_seq,
            dropped_steps: self.dropped_steps,
            by_entry: self.cumulative.clone(),
        }
    }

    pub fn cumulative(&self, entry: u64) -> EntryCounters {
        entry_counters_for(&self.cumulative, entry)
    }

    /// Every entry with at least one step recorded after `mark` (Card 552): the discovery path a
    /// caller uses when it cannot name entry ids ahead of time (e.g. the bench runner, whose decode
    /// entry is created fresh per generation call - `poot-llm/src/backends/gpu_generate.rs`'s
    /// `generate_kv_gpu_cached_sampled` calls `add_entry` on every generation call and removes it on
    /// every exit path) instead of an explicit `entries` filter.
    ///
    /// Review F5: this is a workaround for that per-call churn, not a general discovery API this
    /// card owns. Once a future card (563a's prepared catalog) keeps one resident decode entry
    /// across calls, its caller should name that entry explicitly and this method should be
    /// deleted (`poot/AGENTS.md`: "delete what the replacement makes dead") rather than kept as a
    /// second way to build an `entries` filter.
    pub fn entries_with_activity_since(&self, mark: &Mark) -> Vec<u64> {
        self.cumulative
            .iter()
            .filter(|&(entry, now)| now.steps > entry_counters_for(&mark.by_entry, *entry).steps)
            .map(|(&entry, _)| entry)
            .collect()
    }
}

impl Default for TimingSnapshot {
    /// Counters-only (Card 552: "`TimingOptions` defaults to counters-only"): no detail ring.
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// One `kind_name` bucket's aggregated detail within a window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpBucket {
    pub dispatch_count: u64,
    /// `None` only if not a single contributing dispatch had a measured device duration.
    pub device: Option<Duration>,
}

/// The device-time side of a window: exact cumulative sums, plus a diagnostic span that is only
/// ever present when every retained step in the window had its own measured span (never summed
/// across concurrent/overlapping scopes as if it were one interval).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceWindow {
    /// Exact sum of every window step's `sum_of_dispatch_durations`; `None` only if the window
    /// contains zero steps with a measured device time (never a bare zero).
    pub sum_of_dispatch_durations: Option<Duration>,
    /// Sum of each retained step's own `device_span`, present only when `detail` is
    /// [`Coverage::Complete`] and every included step measured a span. A diagnostic overlap figure:
    /// never compared against [`WindowReport::host`]'s wall as if it were the critical path.
    pub device_span: Option<Duration>,
}

/// A windowed report over a [`TimingSnapshot`] (Card 552 SC-002): exact step/host/device-sum totals
/// for `entries` recorded after `mark`, plus `kind_name` buckets built from whatever per-dispatch
/// detail is still retained, with that detail's own [`Coverage`] reported alongside it rather than
/// implied.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowReport {
    pub entries: Vec<u64>,
    pub steps: u64,
    pub host: HostTiming,
    pub device: DeviceWindow,
    /// Coverage of `buckets`/per-dispatch detail specifically (the cumulative totals above are
    /// always exact regardless of this value).
    pub detail: Coverage,
    pub buckets: BTreeMap<&'static str, OpBucket>,
    /// Card 552: true when any retained step in this window changed submission grouping
    /// to collect its detail. A caller must not present this window's per-token device-time
    /// figures as unperturbed throughput evidence when this is set (a separate, unprofiled/
    /// counters-only run is the throughput evidence); [`Self::render`] labels the window
    /// accordingly. Computed only from retained steps - a window whose perturbed steps were all
    /// since evicted (`detail` would already read `Truncated`) cannot see this either.
    pub perturbed: bool,
}

impl WindowReport {
    /// A plain-text rendering (Card 552 scope): the engine's own step-sum host wall partitioned
    /// into encode/wait/other with the signed residual printed unclamped (`encode + wait + other`
    /// equals that step-sum wall exactly, by construction - Card 552 SC-001), device duration
    /// sum/span reported as a labelled overlap (never subtracted from wall, never mixed into the
    /// same figure as any other wall measurement), the per-backend fallback "device time unknown on
    /// this backend" when nothing was measured, independent cumulative counts, and a
    /// `kind_name`-bucket table noting the detail [`Coverage`] rather than implying completeness.
    ///
    /// `measured_wall` is the caller's own, separately measured wall-clock for the window (e.g. the
    /// bench runner's `Instant::now().elapsed()` around the whole timed loop): a distinct, usually
    /// larger number than the step-sum wall (it also covers host time between steps - sampling,
    /// detokenization, the caller's own callback), printed on its own line and used only to
    /// normalize the device-time-per-token figure, never substituted for the step-sum wall above.
    pub fn render(&self, tokens: usize, measured_wall: Duration) -> String {
        let toks = tokens.max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let mut out = String::new();
        out.push_str(&format!(
            "\n=== poot timing window ({} step(s), {tokens} token(s)) ===\n",
            self.steps
        ));
        out.push_str(&format!(
            "measured wall (caller's own clock, context only): {:.3} ms\n",
            ms(measured_wall),
        ));
        out.push_str(&format!(
            "step-sum wall: {:.3} ms  (encode {:.3} ms + wait {:.3} ms + other {:.3} ms)\n",
            ms(self.host.wall),
            ms(self.host.encode),
            ms(self.host.wait),
            self.other_nanos() as f64 / 1.0e6,
        ));
        if self.perturbed {
            out.push_str(
                "DIAGNOSTIC / PERTURBED WINDOW: collecting per-dispatch device-time detail \
                 changed submission grouping for at least one step here. Do not present the \
                 figures below as unperturbed throughput; take a separate counters-only run for \
                 that.\n",
            );
        }
        match self.device.sum_of_dispatch_durations {
            Some(sum) => {
                out.push_str(&format!(
                    "device time (overlap, not subtracted from wall): sum_of_dispatch_durations {:.3} ms/tok",
                    ms(sum) / toks,
                ));
                match self.device.device_span {
                    Some(span) => out.push_str(&format!("  device_span {:.3} ms\n", ms(span))),
                    None => out.push('\n'),
                }
            }
            None => out.push_str("device time unknown on this backend\n"),
        }
        out.push_str(&format!(
            "per-dispatch detail coverage: {:?}\n",
            self.detail
        ));
        if !self.buckets.is_empty() {
            out.push_str(&format!(
                "{:<20} {:>10} {:>14}\n",
                "kind_name", "dispatches", "device ms"
            ));
            for (kind, bucket) in &self.buckets {
                out.push_str(&format!(
                    "{:<20} {:>10} {:>14}\n",
                    kind,
                    bucket.dispatch_count,
                    bucket
                        .device
                        .map(|d| format!("{:.3}", ms(d)))
                        .unwrap_or_else(|| "unknown".to_string()),
                ));
            }
        }
        out
    }

    /// `host.wall - host.encode - host.wait`, signed, never clamped (Card 552 SC-001).
    pub fn other_nanos(&self) -> i64 {
        self.host.other_nanos()
    }
}

pub struct Report;

impl Report {
    /// Build a [`WindowReport`] for every step recorded in `snapshot` after `mark` whose `entry` is
    /// one of `entries`. Step/host/device-sum totals come from the exact cumulative counters (never
    /// from the bounded detail ring, so an evicted step still counts correctly); `kind_name` buckets
    /// and `device_span` come from whatever detail the ring still retains, with [`Coverage`]
    /// reporting honestly whether that is everything the totals say should be there.
    pub fn window(snapshot: &TimingSnapshot, mark: &Mark, entries: &[u64]) -> WindowReport {
        let mut steps = 0u64;
        let mut host = HostTiming::default();
        let mut device_sum: Option<Duration> = None;
        for &entry in entries {
            let now = entry_counters_for(&snapshot.cumulative, entry);
            let before = entry_counters_for(&mark.by_entry, entry);
            let delta = now.since(&before);
            steps += delta.steps;
            host.accumulate(delta.host);
            if let Some(d) = delta.device_sum {
                device_sum = Some(device_sum.unwrap_or_default() + d);
            }
        }

        let in_window: Vec<&StepTiming> = snapshot
            .retained
            .iter()
            .filter(|s| s.seq >= mark.seq && entries.contains(&s.entry))
            .collect();

        let detail = if snapshot.max_retained_steps == 0 {
            Coverage::Unavailable
        } else if in_window.len() as u64 != steps || snapshot.dropped_steps != mark.dropped_steps {
            // Review F3: ring/record-budget eviction (steps the configured bound dropped), always
            // reported before sampling - an evicted step's own sampling state is unknowable.
            Coverage::Truncated {
                retained: in_window.len() as u64,
                dropped: steps.saturating_sub(in_window.len() as u64),
            }
        } else {
            // Every step the exact cumulative count says should be here is here. Within that,
            // count how many actually carry measured device detail - a step the engine's
            // `sample_every` policy skipped (or a backend/mode that never measures) reports
            // `DeviceCoverage::Unknown` for itself, same shape either way (Card 552).
            let sampled = in_window
                .iter()
                .filter(|s| matches!(s.device, DeviceCoverage::Measured(_)))
                .count() as u64;
            let total = in_window.len() as u64;
            if sampled < total {
                Coverage::Unsampled { sampled, total }
            } else {
                Coverage::Complete
            }
        };

        let mut buckets: BTreeMap<&'static str, OpBucket> = BTreeMap::new();
        let mut span_sum: Option<Duration> = None;
        let mut span_complete = detail.is_complete() && !in_window.is_empty();
        let perturbed = in_window.iter().any(|s| s.perturbed);
        for step in &in_window {
            match step.device {
                DeviceCoverage::Measured(MeasuredDevice {
                    device_span: Some(span),
                    ..
                }) => span_sum = Some(span_sum.unwrap_or_default() + span),
                _ => span_complete = false,
            }
            for d in &step.dispatches {
                let bucket = buckets.entry(d.kind_name).or_default();
                bucket.dispatch_count += 1;
                if let Some(dur) = d.device {
                    bucket.device = Some(bucket.device.unwrap_or_default() + dur);
                }
            }
        }

        WindowReport {
            entries: entries.to_vec(),
            steps,
            host,
            device: DeviceWindow {
                sum_of_dispatch_durations: device_sum,
                device_span: if span_complete { span_sum } else { None },
            },
            detail,
            buckets,
            perturbed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(encode_ms: u64, wait_ms: u64, wall_ms: u64) -> HostTiming {
        HostTiming {
            encode: Duration::from_millis(encode_ms),
            wait: Duration::from_millis(wait_ms),
            wall: Duration::from_millis(wall_ms),
        }
    }

    /// SC-001 shape: encode+wait+other must equal wall exactly, and a nonzero device time must
    /// never be subtracted into `other`.
    #[test]
    fn host_partition_sums_to_wall_without_touching_device() {
        let mut snap = TimingSnapshot::new(16, 64);
        let mark = snap.mark();
        snap.record(
            1,
            1,
            "wgpu",
            host(3, 5, 10),
            DeviceCoverage::Measured(MeasuredDevice {
                sum_of_dispatch_durations: Some(Duration::from_millis(50)),
                device_span: Some(Duration::from_millis(9)),
            }),
            vec![],
        );
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.host.encode.as_millis(), 3);
        assert_eq!(report.host.wait.as_millis(), 5);
        assert_eq!(report.host.wall.as_millis(), 10);
        assert_eq!(report.other_nanos(), 2_000_000); // 10 - 3 - 5 = 2ms, in ns
        // device time is reported, never subtracted from the host partition above.
        assert_eq!(
            report.device.sum_of_dispatch_durations,
            Some(Duration::from_millis(50))
        );
    }

    /// A negative residual (clock noise) prints signed, not clamped to zero.
    #[test]
    fn other_nanos_can_be_negative() {
        let mut snap = TimingSnapshot::new(4, 16);
        let mark = snap.mark();
        snap.record(
            1,
            1,
            "wgpu",
            host(6, 6, 10),
            DeviceCoverage::Unknown,
            vec![],
        );
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.other_nanos(), -2_000_000);
    }

    /// SC-002 shape: a mark taken after warmup, windowed to one entry, excludes every step of
    /// another entry and every step before the mark.
    #[test]
    fn window_excludes_other_entries_and_steps_before_the_mark() {
        let mut snap = TimingSnapshot::new(16, 64);
        // "prefill" warmup, before the mark.
        snap.record(1, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let mark = snap.mark();
        // timed decode steps, after the mark.
        for i in 0..5u64 {
            snap.record(2, i, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        }
        // more prefill after the mark too (must still be excluded by the entry filter).
        snap.record(1, 2, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);

        let report = Report::window(&snap, &mark, &[2]);
        assert_eq!(report.steps, 5);
        assert_eq!(report.host.wall, Duration::from_millis(10));
    }

    /// SC-004 shape: `DeviceCoverage::Unknown` contributes no device bucket/sum, never a fabricated
    /// zero.
    #[test]
    fn unknown_device_time_contributes_no_sum() {
        let mut snap = TimingSnapshot::new(4, 16);
        let mark = snap.mark();
        snap.record(1, 1, "rocm", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.device.sum_of_dispatch_durations, None);
    }

    /// SC-003 shape: two different kind_names sharing a dispatch bucket key stay separate buckets;
    /// repeats of the same kind_name merge into one bucket regardless of their `plan_label`.
    #[test]
    fn buckets_key_on_kind_name_not_plan_label() {
        let mut snap = TimingSnapshot::new(4, 64);
        let mark = snap.mark();
        snap.record(
            1,
            1,
            "wgpu",
            host(1, 1, 2),
            DeviceCoverage::Unknown,
            vec![
                DispatchTiming {
                    dispatch_index: 0,
                    kind_name: "matmul",
                    plan_label: Some("matmul:k1536".into()),
                    device: Some(Duration::from_micros(10)),
                },
                DispatchTiming {
                    dispatch_index: 1,
                    kind_name: "matmul",
                    plan_label: Some("matmul:k4096".into()),
                    device: Some(Duration::from_micros(20)),
                },
                DispatchTiming {
                    dispatch_index: 2,
                    kind_name: "reduce",
                    plan_label: None,
                    device: Some(Duration::from_micros(5)),
                },
            ],
        );
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.buckets.len(), 2);
        let matmul = &report.buckets["matmul"];
        assert_eq!(matmul.dispatch_count, 2);
        assert_eq!(matmul.device, Some(Duration::from_micros(30)));
        assert_eq!(report.buckets["reduce"].dispatch_count, 1);
    }

    /// SC-006 shape: a bound ring evicts the oldest step first, and a window that reaches past the
    /// eviction reports `Truncated` detail while its exact totals (`steps`) stay correct.
    #[test]
    fn bounded_ring_evicts_oldest_and_reports_truncated_coverage() {
        let mut snap = TimingSnapshot::new(2, 64);
        let mark = snap.mark();
        for i in 0..5u64 {
            snap.record(1, i, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        }
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.steps, 5, "exact cumulative count survives eviction");
        assert!(
            matches!(report.detail, Coverage::Truncated { .. }),
            "{:?}",
            report.detail
        );
    }

    /// Review F4: a window with at least one perturbed step (detail collection changed submission
    /// grouping) is labelled as such, and the `render` banner appears; an unperturbed window never
    /// claims otherwise.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552): in `Report::window`,
    /// change `let perturbed = in_window.iter().any(|s| s.perturbed);` to `let perturbed = false;`
    /// (never surface it, the pre-fix bug). Result: RED - this test's
    /// `assert!(report.perturbed)` fails. Reverted: GREEN.
    #[test]
    fn window_is_perturbed_when_any_retained_step_collected_detail() {
        let mut snap = TimingSnapshot::new(4, 64);
        let mark = snap.mark();
        snap.record(1, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        snap.record(
            1,
            2,
            "wgpu",
            host(1, 1, 2),
            DeviceCoverage::Measured(MeasuredDevice {
                sum_of_dispatch_durations: Some(Duration::from_micros(5)),
                device_span: Some(Duration::from_micros(5)),
            }),
            vec![DispatchTiming {
                dispatch_index: 0,
                kind_name: "matmul",
                plan_label: None,
                device: Some(Duration::from_micros(5)),
            }],
        );
        let report = Report::window(&snap, &mark, &[1]);
        assert!(
            report.perturbed,
            "the second step collected per-dispatch detail"
        );
        let text = report.render(1, Duration::from_millis(1));
        assert!(text.contains("DIAGNOSTIC / PERTURBED WINDOW"));

        // A window with no detail-collecting steps is never perturbed.
        let mark2 = snap.mark();
        snap.record(1, 3, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let unperturbed = Report::window(&snap, &mark2, &[1]);
        assert!(!unperturbed.perturbed);
        assert!(
            !unperturbed
                .render(1, Duration::from_millis(1))
                .contains("PERTURBED")
        );
    }

    /// Review F7: `max_retained_dispatch_records == 0` (with `max_retained_steps > 0`) means "never
    /// retain per-dispatch detail", never "unbounded" - a step with real dispatches is still
    /// retained as a ring entry (identity/host/device totals), but with its dispatch list cleared.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552): in
    /// `TimingSnapshot::record`, change the fitting check from `self.retained_dispatch_count +
    /// step.dispatches.len() > self.max_retained_dispatch_records` back to the pre-fix `(...) >
    /// self.max_retained_dispatch_records && self.max_retained_dispatch_records != 0` (the special
    /// case that silently read 0 as unbounded). Result: RED - this test's
    /// `assert!(retained.dispatches.is_empty())` fails because the step's 2 dispatches are kept in
    /// full instead of cleared. Reverted: GREEN.
    #[test]
    fn zero_max_dispatch_records_means_no_dispatch_detail_not_unbounded() {
        let mut snap = TimingSnapshot::new(4, 0);
        let mark = snap.mark();
        snap.record(
            1,
            1,
            "wgpu",
            host(1, 1, 2),
            DeviceCoverage::Unknown,
            vec![
                DispatchTiming {
                    dispatch_index: 0,
                    kind_name: "matmul",
                    plan_label: None,
                    device: None,
                },
                DispatchTiming {
                    dispatch_index: 1,
                    kind_name: "matmul",
                    plan_label: None,
                    device: None,
                },
            ],
        );
        let report = Report::window(&snap, &mark, &[1]);
        // The exact step count stays correct...
        assert_eq!(report.steps, 1);
        // ...but no per-dispatch detail was retained, and the ring still holds the step itself
        // (not evicted - `max_retained_steps` is a separate bound from `max_retained_dispatch_
        // records`).
        assert!(report.buckets.is_empty());
        assert_eq!(
            report.detail,
            Coverage::Truncated {
                retained: 1,
                dropped: 0
            },
            "the step is retained, but with its detail dropped, not a complete 0-dispatch report"
        );
    }

    /// A caller that cannot name entry ids ahead of time (Card 552: the bench runner, whose decode
    /// entry id churns per generation call) discovers them from the snapshot itself.
    #[test]
    fn entries_with_activity_since_finds_only_entries_touched_after_the_mark() {
        let mut snap = TimingSnapshot::new(8, 64);
        snap.record(1, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let mark = snap.mark();
        snap.record(2, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        snap.record(3, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let mut found = snap.entries_with_activity_since(&mark);
        found.sort();
        assert_eq!(found, vec![2, 3], "entry 1's step was before the mark");
    }

    /// A caller's own measured wall-clock (larger than the step-sum wall: it also covers host time
    /// between steps) must never be substituted for the step-sum wall in `render`'s encode/wait/
    /// other breakdown - the regression this guards is exactly that substitution, which silently
    /// broke the printed `encode + wait + other == wall` identity while the underlying data stayed
    /// correct.
    #[test]
    fn render_never_lets_the_callers_measured_wall_corrupt_the_step_sum_identity() {
        let mut snap = TimingSnapshot::new(4, 16);
        let mark = snap.mark();
        snap.record(
            1,
            1,
            "wgpu",
            host(3, 5, 10),
            DeviceCoverage::Unknown,
            vec![],
        );
        let report = Report::window(&snap, &mark, &[1]);
        // A measured wall much larger than the 10ms step-sum wall (as a real multi-step window's
        // external clock, which also covers inter-step host time, would be).
        let text = report.render(1, Duration::from_millis(500));
        assert!(text.contains("step-sum wall: 10.000 ms"), "{text}");
        assert!(text.contains("encode 3.000 ms"), "{text}");
        assert!(text.contains("wait 5.000 ms"), "{text}");
        assert!(text.contains("other 2.000 ms"), "{text}");
        assert!(text.contains("measured wall"), "{text}");
        assert!(text.contains("500.000 ms"), "{text}");
    }

    /// `render` prints the literal per-backend fallback when nothing was measured, never a "0 ms"
    /// line (Card 552 SC-004's rendering half).
    #[test]
    fn render_prints_the_unknown_fallback_not_a_zero_line() {
        let mut snap = TimingSnapshot::new(4, 16);
        let mark = snap.mark();
        snap.record(1, 1, "rocm", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let report = Report::window(&snap, &mark, &[1]);
        let text = report.render(1, Duration::from_millis(2));
        assert!(text.contains("device time unknown on this backend"));
        assert!(!text.contains("sum_of_dispatch_durations"));
    }

    /// Counters-only (`max_retained_steps == 0`, the default) never retains detail, and reports it
    /// as `Unavailable`, not `Complete` over an empty bucket set.
    #[test]
    fn counters_only_reports_detail_unavailable() {
        let mut snap = TimingSnapshot::default();
        let mark = snap.mark();
        snap.record(1, 1, "wgpu", host(1, 1, 2), DeviceCoverage::Unknown, vec![]);
        let report = Report::window(&snap, &mark, &[1]);
        assert_eq!(report.steps, 1);
        assert_eq!(report.detail, Coverage::Unavailable);
        assert!(report.buckets.is_empty());
    }
}
