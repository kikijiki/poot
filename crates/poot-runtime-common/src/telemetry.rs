//! Typed execution-counter units, purposes and coverage (Card 552, R-546-11/R-552-3): the one shape
//! every backend's physical-call and transfer counters report, so cross-backend equality is not equal
//! by construction (548/549 integrate their own native producers against this same shape). Cumulative
//! counts here are always complete and independent of whatever bounded detailed timing is configured
//! ([`Coverage`] describes that separate, bounded *detail* retention, never these totals).

use std::collections::BTreeMap;

/// What one native API call was for. A physical call is tagged with exactly one purpose at its call
/// site; a call that serves several logical needs (e.g. a shared readback that also staged a
/// validation witness) still counts once in the total under its one declared purpose (Card 552
/// SC-007), never split or duplicated across purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallPurpose {
    /// Compute dispatch submission (the launch-tax signal).
    Compute,
    /// Host-to-device upload (weight/const/slot writes).
    Upload,
    /// Device-to-host result readback.
    Readback,
    /// Kernel-assert / validation-packet staging and readback.
    Validation,
    /// Timestamp-query resolve/readback (the timing mechanism's own cost, kept separate so it never
    /// hides inside `Compute`).
    Timing,
}

impl CallPurpose {
    /// Every purpose, in the fixed order a report iterates them.
    pub const ALL: [Self; 5] = [
        Self::Compute,
        Self::Upload,
        Self::Readback,
        Self::Validation,
        Self::Timing,
    ];
}

/// One purpose's native submit/wait counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallCounts {
    /// Native command-buffer submissions (`queue.submit`-equivalent calls), never a dispatch/replay
    /// proxy: a batched replay of many dispatches is still one submission per flush.
    pub submits: u64,
    /// Blocking waits for submitted work (device poll/sync-equivalent calls).
    pub waits: u64,
}

/// Transfer direction for byte/operation counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

/// One (purpose, direction) pair's transfer counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferCounts {
    /// Distinct transfer operations (one `copy`/`write`/`read` call, independent of its byte size).
    pub operations: u64,
    pub bytes: u64,
}

/// Cumulative execution counters for one backend/context (Card 552 scope: "cumulative execution
/// counts do not depend on timestamp coverage"). Logical counts are graph-level (one per equation
/// dispatch, one per captured-step replay); they are never derived from, or compared for equality
/// against, the physical native-call counts below, which are a backend's own driver-level truth. A
/// native graph launch is one launch, never its node count; a backend that cannot see internal driver
/// batching leaves that physical count absent rather than inventing a number.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecutionCounters {
    /// Logical dispatch count: one per planned equation's dispatch, at the point the engine walks it
    /// into the device, independent of how many times its step later replays.
    pub logical_dispatches: u64,
    /// Logical replay count: one per captured step replayed (Card 546a's capture/replay contract).
    pub logical_replays: u64,
    /// Total native calls by purpose. Every call site records into exactly one purpose, so
    /// [`Self::total_submits`]/[`Self::total_waits`] never double-count a mixed-purpose call.
    pub calls: BTreeMap<CallPurpose, CallCounts>,
    /// Transfer operations/bytes by (purpose, direction).
    pub transfers: BTreeMap<(CallPurpose, TransferDirection), TransferCounts>,
}

impl ExecutionCounters {
    pub fn calls(&self, purpose: CallPurpose) -> CallCounts {
        self.calls.get(&purpose).copied().unwrap_or_default()
    }

    pub fn transfer(&self, purpose: CallPurpose, direction: TransferDirection) -> TransferCounts {
        self.transfers
            .get(&(purpose, direction))
            .copied()
            .unwrap_or_default()
    }

    /// Record one native submission under `purpose`.
    pub fn record_submit(&mut self, purpose: CallPurpose) {
        self.calls.entry(purpose).or_default().submits += 1;
    }

    /// Record one blocking wait under `purpose`.
    pub fn record_wait(&mut self, purpose: CallPurpose) {
        self.calls.entry(purpose).or_default().waits += 1;
    }

    /// Record one transfer operation of `bytes` under `(purpose, direction)`.
    pub fn record_transfer(
        &mut self,
        purpose: CallPurpose,
        direction: TransferDirection,
        bytes: u64,
    ) {
        let entry = self.transfers.entry((purpose, direction)).or_default();
        entry.operations += 1;
        entry.bytes += bytes;
    }

    /// Total native submissions across every purpose: the one physical-submission total (Card 552
    /// SC-007's "a known mixed-purpose submission counts once in the total" holds by construction,
    /// since every call site records into exactly one purpose bucket, never several).
    pub fn total_submits(&self) -> u64 {
        self.calls.values().map(|c| c.submits).sum()
    }

    pub fn total_waits(&self) -> u64 {
        self.calls.values().map(|c| c.waits).sum()
    }
}

/// Coverage of a bounded, retained window of *detail* (per-dispatch timing records, query
/// resources): never a property of [`ExecutionCounters`], which is always complete. A report that
/// carries detail must carry this alongside it, so sampled or truncated detail can never be read as
/// a complete per-dispatch account (Card 552 scope: "a report cannot silently claim complete
/// per-dispatch coverage from a sampled or truncated window").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every eligible record in the requested window was retained and attributed.
    Complete,
    /// Sampling was configured on: only `sampled` of `total` eligible records were ever collected,
    /// by policy (not because of a resource bound).
    Unsampled { sampled: u64, total: u64 },
    /// A record/byte/in-flight-resource bound was hit: `dropped` records beyond `retained` were
    /// evicted or never retained, not sampled out by policy.
    Truncated { retained: u64, dropped: u64 },
    /// No detail was collected for this window at all (counters-only mode, or the device/backend
    /// cannot measure it).
    Unavailable,
}

impl Coverage {
    pub fn is_complete(self) -> bool {
        matches!(self, Coverage::Complete)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_purpose_calls_count_once_each_in_the_total() {
        let mut c = ExecutionCounters::default();
        c.record_submit(CallPurpose::Compute);
        c.record_submit(CallPurpose::Compute);
        c.record_submit(CallPurpose::Readback);
        c.record_wait(CallPurpose::Readback);
        c.record_transfer(CallPurpose::Readback, TransferDirection::DeviceToHost, 128);
        c.record_transfer(CallPurpose::Upload, TransferDirection::HostToDevice, 64);

        assert_eq!(c.calls(CallPurpose::Compute).submits, 2);
        assert_eq!(c.calls(CallPurpose::Readback).submits, 1);
        assert_eq!(c.calls(CallPurpose::Readback).waits, 1);
        assert_eq!(c.total_submits(), 3);
        assert_eq!(c.total_waits(), 1);
        assert_eq!(
            c.transfer(CallPurpose::Readback, TransferDirection::DeviceToHost)
                .bytes,
            128
        );
        assert_eq!(
            c.transfer(CallPurpose::Upload, TransferDirection::HostToDevice)
                .bytes,
            64
        );
        // A purpose that recorded no transfer reads as zero, not a missing/panicking lookup.
        assert_eq!(
            c.transfer(CallPurpose::Compute, TransferDirection::HostToDevice)
                .bytes,
            0
        );
    }

    #[test]
    fn logical_counts_are_independent_of_physical_calls() {
        let mut c = ExecutionCounters {
            logical_dispatches: 500,
            logical_replays: 1,
            ..Default::default()
        };
        // A batched replay issues far fewer physical submits than logical dispatches.
        c.record_submit(CallPurpose::Compute);
        assert_eq!(c.logical_dispatches, 500);
        assert_eq!(c.total_submits(), 1);
    }

    #[test]
    fn coverage_complete_is_the_only_complete_variant() {
        assert!(Coverage::Complete.is_complete());
        assert!(!Coverage::Unavailable.is_complete());
        assert!(
            !Coverage::Truncated {
                retained: 1,
                dropped: 1
            }
            .is_complete()
        );
        assert!(
            !Coverage::Unsampled {
                sampled: 1,
                total: 2
            }
            .is_complete()
        );
    }
}
