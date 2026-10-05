//! Card 360 acceptance/mutation tests, one file per row of the card's own acceptance table
//! Every row's fixture device count
//! and required mutation are named in that table; these tests honor it directly rather than a uniform
//! two-device default (see the card's own review note).

use std::collections::HashMap;

use crate::multi_device::replay::{ReplayContract, StaticBufferId, StaticBufferKind};

/// Counted calls the model-free fake runtime observes. Static setup counters stay at zero on a warm
/// replay, while dispatches advance on every run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReplayCounters {
    allocations: u64,
    uploads: u64,
    carrier_rebuilds: u64,
    dispatches: u64,
}

/// Test-only bookkeeping for the replay contract; this is not a backend implementation.
#[derive(Debug, Default)]
struct FakeReplayRuntime {
    resident: HashMap<StaticBufferId, u64>,
    counters: ReplayCounters,
}

impl FakeReplayRuntime {
    fn new() -> Self {
        Self::default()
    }

    fn run(&mut self, contract: &ReplayContract) {
        for buffer in &contract.buffers {
            match self.resident.get(&buffer.id) {
                None => {
                    self.counters.allocations += 1;
                    self.counters.uploads += 1;
                    if buffer.kind == StaticBufferKind::Carrier {
                        self.counters.carrier_rebuilds += 1;
                    }
                    self.resident.insert(buffer.id.clone(), buffer.fingerprint);
                }
                Some(&fingerprint) if fingerprint == buffer.fingerprint => {}
                Some(_) => {
                    self.counters.uploads += 1;
                    if buffer.kind == StaticBufferKind::Carrier {
                        self.counters.carrier_rebuilds += 1;
                    }
                    self.resident.insert(buffer.id.clone(), buffer.fingerprint);
                }
            }
        }
        self.counters.dispatches += contract.dispatch_count as u64;
    }
}

mod accounting;
mod cache_key;
mod capability;
mod communication_dependency;
mod model_neutral;
mod packed_residency;
mod partitioned;
mod placement_set;
mod replay;
mod replicated;
mod routed_exchange;
mod stage;

/// Small shared fixture builders reused by more than one row's test.
mod fixtures;
