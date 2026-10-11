//! The expert-residency manager (POOT-746, ADR-0072 amended): which experts of each pooled site are
//! resident in which pool rows, as a [`HostSync`] the executor calls at every routed sync point.
//!
//! At a sync the manager reads the step's expert ids, pins the experts the step needs, picks rows for
//! the misses (free rows first, then the least recently used row no needed expert holds) and answers
//! with the rows to refill and, when the assignment changed, the whole `[E]` map. The executor
//! writes the stored bytes of each named expert into its row ([`SyncWrites`]); this type never holds
//! weight bytes. State is keyed by the canonical [`SiteIdentity`] the executor reports, so a prefill, a
//! decode and a chunk entry that page the same tables share one LRU, two tables of one layer do not,
//! and a same-named weight of another store generation starts empty.
//!
//! One manager serves one executable's resident buffers. Resetting the executable's state clears
//! them on the device, so the driver resets the manager in the same call ([`ExpertResidency::reset`]):
//! a manager that kept its map after the device lost the rows would serve stale experts.

use std::collections::HashMap;

use poot_executor::{ExecError, HostSync, RowFill, SiteIdentity, SyncReads, SyncWrites};
use poot_graph_plan::SyncPoint;

/// What the manager has done since its last reset: its own count of the syncs it served, kept as a
/// cross-check on the engine's. The source of truth for refill traffic is
/// [`poot_executor::ExecutorStats`] (`syncs`, `sync_read_bytes`, `pool_rows_written`,
/// `pool_refill_bytes`), which the engine produces from what it read and wrote; these counters are the
/// manager's decisions and are cleared by [`ExpertResidency::reset`] and by a failed step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyCounters {
    /// Sync points served (all-hit ones included).
    pub syncs: u64,
    /// Expert ids read back across those syncs.
    pub id_reads: u64,
    /// Experts written into a pool row.
    pub refills: u64,
    /// Refills that displaced another expert.
    pub evictions: u64,
    /// Device bytes those refills wrote, over every table of the site.
    pub refill_bytes: u64,
}

/// Why the manager could not serve a sync point.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    #[error("the sync point reads no expert ids")]
    NoIds,
    #[error("expert id {id} is not one of the site's {experts} experts")]
    ExpertId { id: i32, experts: usize },
    #[error("a step needs {needed} experts but the pool has {slots} rows")]
    PoolTooSmall { needed: usize, slots: usize },
}

struct Site {
    experts: usize,
    /// The expert each pool row holds.
    row_expert: Vec<Option<usize>>,
    /// The pool row each expert sits in.
    expert_row: Vec<Option<usize>>,
    /// The sync at which each row was last needed.
    used: Vec<u64>,
}

/// What one sync did to a site: the `(row, expert)` refills and how many displaced another expert.
struct Served {
    fills: Vec<(usize, usize)>,
    evicted: u64,
}

impl Site {
    /// Make `needed` (distinct experts, at most one per row) resident at sync `now`. The needed
    /// experts already resident are pinned first, so no victim is one of them; a miss takes a free row
    /// before the least recently used one.
    fn serve(&mut self, needed: &[usize], now: u64) -> Served {
        for &expert in needed {
            if let Some(row) = self.expert_row[expert] {
                self.used[row] = now;
            }
        }
        let mut fills = Vec::new();
        let mut evicted = 0u64;
        for &expert in needed {
            if self.expert_row[expert].is_some() {
                continue;
            }
            let row = (0..self.row_expert.len())
                .filter(|&row| self.used[row] != now)
                .min_by_key(|&row| (self.row_expert[row].is_some(), self.used[row]))
                .expect("a pool with a row per needed expert has an unpinned row");
            if let Some(old) = self.row_expert[row].take() {
                self.expert_row[old] = None;
                evicted += 1;
            }
            self.row_expert[row] = Some(expert);
            self.expert_row[expert] = Some(row);
            self.used[row] = now;
            fills.push((row, expert));
        }
        Served { fills, evicted }
    }

    fn new(experts: usize, slots: usize) -> Self {
        Self {
            experts,
            row_expert: vec![None; slots],
            expert_row: vec![None; experts],
            used: vec![0; slots],
        }
    }
}

/// LRU residency per pooled site.
#[derive(Default)]
pub struct ExpertResidency {
    sites: HashMap<SiteIdentity, Site>,
    clock: u64,
    counters: ResidencyCounters,
}

impl ExpertResidency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn counters(&self) -> ResidencyCounters {
        self.counters
    }

    /// Forget every assignment, the LRU order and the counters, together. Called when the executable's
    /// resident buffers are cleared or lost.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl HostSync for ExpertResidency {
    fn sync(&mut self, point: &SyncPoint, reads: &SyncReads) -> Result<SyncWrites, ExecError> {
        let ids = point
            .reads
            .first()
            .and_then(|&id| reads.values(id))
            .ok_or_else(|| ExecError::Host(Box::new(ResidencyError::NoIds)))?;
        let host = |error: ResidencyError| ExecError::Host(Box::new(error));
        let experts = point.site.experts;
        let slots = point.site.slots;

        // The distinct experts the step selects, in first-seen order.
        let mut needed: Vec<usize> = Vec::new();
        for &id in ids {
            let Some(id) = usize::try_from(id).ok().filter(|&id| id < experts) else {
                return Err(host(ResidencyError::ExpertId { id, experts }));
            };
            if !needed.contains(&id) {
                needed.push(id);
            }
        }
        if needed.len() > slots {
            return Err(host(ResidencyError::PoolTooSmall {
                needed: needed.len(),
                slots,
            }));
        }

        self.clock += 1;
        let now = self.clock;
        let site = self
            .sites
            .entry(reads.site().clone())
            .or_insert_with(|| Site::new(experts, slots));
        debug_assert_eq!((site.experts, site.row_expert.len()), (experts, slots));
        let Served { fills, evicted } = site.serve(&needed, now);

        let mut writes = SyncWrites::default();
        if !fills.is_empty() {
            // Every table of the site gets the same expert in the same row.
            for table in 0..point.site.weights.len() {
                for &(row, expert) in &fills {
                    writes.rows.push(RowFill {
                        table,
                        first_row: row,
                        experts: vec![expert],
                    });
                }
            }
            writes.map = Some(
                site.expert_row
                    .iter()
                    .map(|row| row.map_or(-1, |row| row as i32))
                    .collect(),
            );
        }
        self.counters.syncs += 1;
        self.counters.id_reads += ids.len() as u64;
        self.counters.refills += fills.len() as u64;
        self.counters.evictions += evicted;
        self.counters.refill_bytes +=
            fills.len() as u64 * reads.row_bytes().iter().map(|&b| b as u64).sum::<u64>();
        Ok(writes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site() -> Site {
        Site::new(6, 3)
    }

    /// A miss takes a free row, a hit refills nothing, and an all-hit sync writes no row and no map.
    ///
    /// Mutation: in `serve`, refill needed experts that are already resident; the all-hit sync reports
    /// fills and this row goes red.
    #[test]
    fn a_hit_refills_nothing() {
        let mut site = site();
        assert_eq!(site.serve(&[4, 1], 1).fills.len(), 2);
        let again = site.serve(&[1, 4], 2);
        assert!(again.fills.is_empty());
        assert_eq!(again.evicted, 0);
    }

    /// The least recently used row is the victim, and a needed expert is never one: with rows holding
    /// 0 (sync 1), 1 (sync 2) and 2 (sync 3), a step needing {0, 5} evicts expert 1, not the older 0.
    ///
    /// Mutation: pick the victim before pinning the hits; expert 0 (the oldest) is evicted for 5 and
    /// this row goes red.
    #[test]
    fn the_victim_is_the_least_recently_used_row_a_needed_expert_does_not_hold() {
        let mut site = site();
        site.serve(&[0], 1);
        site.serve(&[1], 2);
        site.serve(&[2], 3);
        let served = site.serve(&[5, 0], 4);
        assert_eq!(served.evicted, 1);
        assert!(site.expert_row[0].is_some(), "the pinned hit stayed");
        assert!(site.expert_row[1].is_none(), "the older of the others went");
        assert!(site.expert_row[2].is_some());
        assert!(site.expert_row[5].is_some());
    }

    /// The map the manager publishes is a function of the assignment: each expert's row or none.
    #[test]
    fn rows_and_experts_stay_inverse_maps() {
        let mut site = site();
        for (now, needed) in [[0, 1, 2], [3, 4, 5], [0, 5, 2]].iter().enumerate() {
            site.serve(needed, now as u64 + 1);
            for (row, expert) in site.row_expert.iter().enumerate() {
                if let Some(expert) = expert {
                    assert_eq!(site.expert_row[*expert], Some(row));
                }
            }
            assert_eq!(
                site.expert_row.iter().flatten().count(),
                site.row_expert.iter().flatten().count()
            );
        }
    }

    /// A reset forgets the clock and the counters together with the sites.
    #[test]
    fn a_reset_forgets_the_clock_and_the_counters() {
        let mut manager = ExpertResidency::new();
        manager.clock = 7;
        manager.counters.refills = 2;
        manager.reset();
        assert_eq!(manager.counters(), ResidencyCounters::default());
        assert_eq!(manager.clock, 0);
        assert!(manager.sites.is_empty());
    }
}
