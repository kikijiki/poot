//! The one memory service's role taxonomy and per-role counters (Card 547a, L6): every runtime keeps
//! one [`MemoryCounters`] instead of its own ad hoc tally, so `Engine::stats().memory` (poot-executor)
//! reads identical counter shapes from every backend.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// What a buffer is for. Each runtime keeps one counter set per role (moved from the executor-contract
/// spike's `device.rs`, Card 546a).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferRole {
    Weight,
    State,
    Activation,
    Input,
    Output,
    Meta,
    /// Direct result/validation readback staging and timestamp buffers (Card 547a scope: "include
    /// direct result/validation staging, timestamp buffers and allocator pools, not only `alloc_f32`").
    /// On ROCm this is the role of a fine-pool allocation whose whole purpose is bridging a transfer -
    /// not the coarse-pool-to-fine-pool DMA bridge itself, which never allocates: a coarse write/read
    /// (Card 547a) goes through `RocmContext::dma_staging`, pinned host memory owned by the
    /// transfer helper, not a `MemoryCounters`-charged buffer of any role.
    Staging,
}

impl BufferRole {
    /// Every role, in the fixed order [`MemoryCounters::snapshot_all`] reports them.
    pub const ALL: [Self; 7] = [
        Self::Weight,
        Self::State,
        Self::Activation,
        Self::Input,
        Self::Output,
        Self::Meta,
        Self::Staging,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Weight => 0,
            Self::State => 1,
            Self::Activation => 2,
            Self::Input => 3,
            Self::Output => 4,
            Self::Meta => 5,
            Self::Staging => 6,
        }
    }
}

/// Live and peak bytes plus allocation count of one role, at the moment of the snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryCounterSnapshot {
    /// Bytes currently allocated and not yet released (SC-002/SC-003: reaches 0 once every buffer of
    /// this role has dropped, except a buffer retained past a poisoned device or an unproven pending
    /// completion, SC-004).
    pub live_bytes: u64,
    /// The highest `live_bytes` this role has ever reached.
    pub peak_bytes: u64,
    /// Distinct allocations made so far (monotonic; never decremented).
    pub allocations: u64,
}

#[derive(Debug, Default)]
struct RoleCounter {
    live_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    allocations: AtomicU64,
}

impl RoleCounter {
    fn snapshot(&self) -> MemoryCounterSnapshot {
        MemoryCounterSnapshot {
            live_bytes: self.live_bytes.load(Ordering::Acquire),
            peak_bytes: self.peak_bytes.load(Ordering::Acquire),
            allocations: self.allocations.load(Ordering::Acquire),
        }
    }
}

/// One role's counter set behind a shared handle, so an [`AllocGuard`] born from a clone of this
/// [`MemoryCounters`] can still decrement the same live-bytes total after the runtime that allocated it
/// has been dropped (mirroring ROCm's existing `RocmAllocationCounter` pattern, generalized to every role
/// and every runtime, Card 547a: "one memory service instead of four").
#[derive(Clone, Debug)]
pub struct MemoryCounters {
    roles: Arc<[RoleCounter; 7]>,
}

impl Default for MemoryCounters {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCounters {
    pub fn new() -> Self {
        Self {
            roles: Arc::new(Default::default()),
        }
    }

    /// Record one new allocation of `bytes` under `role`, bumping live/peak/count, and return the guard
    /// that decrements `live_bytes` back when the allocation is released. The guard is the only way
    /// `live_bytes` moves down: a runtime holds it for the allocation's whole lifetime (typically inside
    /// the handle's `Drop`), and a poisoned or still-pending allocation that must stay charged (SC-004)
    /// simply never drops its guard (`std::mem::forget` it, or hold it behind `ManuallyDrop` and skip the
    /// drop call).
    pub fn record_alloc(&self, role: BufferRole, bytes: u64) -> AllocGuard {
        let counter = &self.roles[role.index()];
        counter.allocations.fetch_add(1, Ordering::Relaxed);
        let live = counter.live_bytes.fetch_add(bytes, Ordering::AcqRel) + bytes;
        counter.peak_bytes.fetch_max(live, Ordering::AcqRel);
        AllocGuard {
            bytes,
            live_bytes: Arc::clone(&self.roles),
            index: role.index(),
        }
    }

    /// One role's current counters.
    pub fn snapshot(&self, role: BufferRole) -> MemoryCounterSnapshot {
        self.roles[role.index()].snapshot()
    }

    /// Every role with at least one allocation ever recorded, in [`BufferRole::ALL`] order.
    pub fn snapshot_all(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        BufferRole::ALL
            .into_iter()
            .map(|role| (role, self.snapshot(role)))
            .filter(|(_, snap)| snap.allocations > 0)
            .collect()
    }

    /// Total live bytes across every role.
    pub fn total_live_bytes(&self) -> u64 {
        BufferRole::ALL
            .into_iter()
            .map(|role| self.snapshot(role).live_bytes)
            .sum()
    }
}

/// Decrements its role's `live_bytes` by its recorded size, exactly once, when dropped. Born from
/// [`MemoryCounters::record_alloc`]; a runtime stores one inside its buffer handle's cleanup path. Never
/// decrements twice: there is no `Clone`, and a caller that must keep a release from firing (a poisoned
/// device, an unproven pending completion, SC-004) holds this behind `std::mem::ManuallyDrop` and skips
/// the drop call rather than calling it twice.
pub struct AllocGuard {
    bytes: u64,
    live_bytes: Arc<[RoleCounter; 7]>,
    index: usize,
}

impl Drop for AllocGuard {
    fn drop(&mut self) {
        self.live_bytes[self.index]
            .live_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_alloc_bumps_live_peak_and_count() {
        let counters = MemoryCounters::new();
        let a = counters.record_alloc(BufferRole::Weight, 100);
        assert_eq!(
            counters.snapshot(BufferRole::Weight),
            MemoryCounterSnapshot {
                live_bytes: 100,
                peak_bytes: 100,
                allocations: 1
            }
        );
        let b = counters.record_alloc(BufferRole::Weight, 50);
        assert_eq!(
            counters.snapshot(BufferRole::Weight),
            MemoryCounterSnapshot {
                live_bytes: 150,
                peak_bytes: 150,
                allocations: 2
            }
        );
        drop(a);
        let after_a = counters.snapshot(BufferRole::Weight);
        assert_eq!(
            after_a.live_bytes, 50,
            "dropping one allocation frees its bytes"
        );
        assert_eq!(after_a.peak_bytes, 150, "peak never drops");
        assert_eq!(after_a.allocations, 2, "allocation count never drops");
        drop(b);
        let after_b = counters.snapshot(BufferRole::Weight);
        assert_eq!(after_b.live_bytes, 0, "dropping every buffer reads zero");
    }

    #[test]
    fn roles_are_independent() {
        let counters = MemoryCounters::new();
        let _a = counters.record_alloc(BufferRole::Weight, 10);
        let _b = counters.record_alloc(BufferRole::Activation, 20);
        assert_eq!(counters.snapshot(BufferRole::Weight).live_bytes, 10);
        assert_eq!(counters.snapshot(BufferRole::Activation).live_bytes, 20);
        assert_eq!(counters.snapshot(BufferRole::State).live_bytes, 0);
    }

    #[test]
    fn snapshot_all_only_lists_roles_with_an_allocation() {
        let counters = MemoryCounters::new();
        let _a = counters.record_alloc(BufferRole::Meta, 4);
        let all = counters.snapshot_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, BufferRole::Meta);
    }

    /// A `MemoryCounters` clone and the original share the same underlying atomics: this is what lets a
    /// runtime hand out a cheap clone to a test or a diagnostic while every allocation's guard still
    /// decrements the one shared total.
    #[test]
    fn clones_share_the_same_counters() {
        let counters = MemoryCounters::new();
        let clone = counters.clone();
        let guard = counters.record_alloc(BufferRole::Input, 8);
        assert_eq!(clone.snapshot(BufferRole::Input).live_bytes, 8);
        drop(guard);
        assert_eq!(clone.snapshot(BufferRole::Input).live_bytes, 0);
    }

    /// A guard that is forgotten (the SC-004 "leak on poison/pending" pattern) never decrements: the
    /// allocation it was born from stays charged forever, exactly as a real poisoned device's never-freed
    /// buffer would.
    #[test]
    fn a_forgotten_guard_leaves_the_allocation_charged() {
        let counters = MemoryCounters::new();
        let guard = counters.record_alloc(BufferRole::State, 32);
        std::mem::forget(guard);
        assert_eq!(counters.snapshot(BufferRole::State).live_bytes, 32);
    }
}
