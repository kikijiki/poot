//! Timing collection options (Card 552): typed configuration, never an environment switch
//! (`poot/AGENTS.md`'s "execution choices are typed configuration, never env vars in library
//! code"). `TimingOptions` governs two independent things:
//!
//! - how much [`crate::Engine`] retains in its own bounded [`poot_profile::TimingSnapshot`]
//!   (`max_retained_steps`/`max_retained_dispatch_records`), which never perturbs execution: it is
//!   host-side bookkeeping over numbers the step already produces;
//! - whether the concrete [`crate::Device`] was constructed to request per-dispatch device
//!   timestamps at all (a backend-specific capability, decided when the `Device` is built, e.g.
//!   `WgpuDevice::new_with_timing`), which `TimingOptions::Detailed` documents the intent for but
//!   does not itself switch on a running device.
//!
//! The default, [`TimingOptions::CountersOnly`], retains no per-dispatch detail and asks nothing
//! extra of the device: cumulative counters (step/dispatch counts, host-wall partitions) are always
//! produced regardless of this setting, since measuring them costs nothing beyond `Instant::now()`.

/// How much bounded per-dispatch detail [`crate::Engine`] retains.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TimingOptions {
    /// No per-dispatch detail retained. Cumulative step/host/device-sum counters remain exact and
    /// available (Card 552 scope: "cumulative execution counts do not depend on timestamp
    /// coverage").
    #[default]
    CountersOnly,
    /// Bounded per-dispatch device-time detail. Retrieving it on a real device typically requires
    /// requesting timestamps on the replayed work, which some backends can only do without an
    /// extra, undisclosed synchronization by riding the step's own existing `synchronize()` call;
    /// a caller that cannot arrange that must label the run diagnostic/perturbed rather than use it
    /// as unprofiled throughput evidence (Card 552 scope).
    Detailed(DetailedTiming),
}

/// Bounds and policy for detailed per-dispatch retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetailedTiming {
    /// Collect detail for 1 of every `sample_every` steps (1 = every step). Must be at least 1.
    pub sample_every: u32,
    /// Maximum retained [`poot_profile::StepTiming`] records; older ones evict first.
    pub max_retained_steps: usize,
    /// Maximum retained per-dispatch records across every retained step combined.
    pub max_retained_dispatch_records: usize,
    /// Maximum number of in-flight device query resources (e.g. timestamp query sets) a `Device`
    /// may hold at once while collecting this detail.
    pub max_in_flight_queries: usize,
    /// What a `Device` does once `max_in_flight_queries` would otherwise be exceeded.
    pub drain_policy: DrainPolicy,
}

impl DetailedTiming {
    /// Every step, bounded to `max_retained_steps` records and `max_retained_dispatch_records`
    /// dispatch records, at most `max_in_flight_queries` query resources in flight at once.
    pub fn every_step(
        max_retained_steps: usize,
        max_retained_dispatch_records: usize,
        max_in_flight_queries: usize,
    ) -> Self {
        Self {
            sample_every: 1,
            max_retained_steps,
            max_retained_dispatch_records,
            max_in_flight_queries,
            drain_policy: DrainPolicy::WaitForCapacity,
        }
    }
}

/// What a `Device` does once its in-flight query-resource bound would otherwise be exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrainPolicy {
    /// Wait for the oldest in-flight query resource to complete before issuing another past the
    /// bound; a resource is reused or freed only after completion is proven, never while pending
    /// (Card 552 scope: "dropping detail must not free pending resources or block execution" - this
    /// policy only ever waits for resources this same step is about to replace, not an unrelated
    /// consumer's pending read).
    WaitForCapacity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_counters_only() {
        assert_eq!(TimingOptions::default(), TimingOptions::CountersOnly);
    }

    #[test]
    fn every_step_has_sample_every_one() {
        let d = DetailedTiming::every_step(8, 256, 4);
        assert_eq!(d.sample_every, 1);
        assert_eq!(d.max_retained_steps, 8);
        assert_eq!(d.max_retained_dispatch_records, 256);
        assert_eq!(d.max_in_flight_queries, 4);
        assert_eq!(d.drain_policy, DrainPolicy::WaitForCapacity);
    }
}
