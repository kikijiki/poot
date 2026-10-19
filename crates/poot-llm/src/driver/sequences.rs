//! The open sequences of a serving driver: the KV store they keep their K/V in, their slots and the
//! state each one carries between steps. [`Serving`] is the state a [`Driver`](super::Driver) holds once
//! `prepare` has admitted a layout; a driver that has not been prepared has none, and so none of the
//! methods that use one.

use std::collections::HashMap;

use poot_models::io::RequestInputs;
use poot_models::model::KvLayout;

use super::block_table::PagedKvCache;
use super::caps::{Layout, PoolShape};
use super::lora::AdapterRef;
use super::prefix_cache::PrefixIdentity;
use super::step::SeqId;

/// One open sequence.
#[derive(Debug)]
pub(super) struct Seq {
    /// The sequence's slot in the paged cache: its block table.
    pub(super) slot: usize,
    /// The prompt, then the committed generated tokens.
    pub(super) tokens: Vec<u32>,
    pub(super) prompt_len: usize,
    /// Prompt positions whose K/V is resident: reused at open, then advanced by prefill.
    pub(super) done: usize,
    /// Positions the block table covers: the prompt and `max_new`.
    pub(super) reserved: usize,
    pub(super) adapter: AdapterRef,
    pub(super) identity: PrefixIdentity,
    pub(super) seeded: bool,
    /// The last step's picks, until `commit` takes them.
    pub(super) pending: Vec<u32>,
    /// The tokens of `pending` a recurrent state already holds: a verified row's result, whose kept
    /// prefix the round folded (or replayed) into it. Zero for a row whose state folds nothing ahead.
    pub(super) folded: usize,
    /// A commit kept fewer tokens than `folded`, so the recurrent state holds tokens the sequence
    /// dropped. Only a release, or a resume that resets the row, ends it.
    pub(super) state_ahead: bool,
    /// The values the request supplied for its model's inputs, bound on the steps that carry them.
    /// Immutable and shared: a fork of the sequence holds the same value.
    pub(super) inputs: RequestInputs,
}

/// Where the open sequences keep their K/V.
#[derive(Debug)]
pub(super) enum KvStore {
    /// The one private cache of a single sequence: no block table, no prefix reuse.
    Contiguous,
    /// The shared paged pool.
    Paged {
        pool: PoolShape,
        cache: PagedKvCache,
    },
}

/// The KV store, the open sequences and their slots.
#[derive(Debug)]
pub struct Serving {
    pub(super) kv: KvStore,
    pub(super) seqs: HashMap<SeqId, Seq>,
    pub(super) free_slots: Vec<usize>,
    pub(super) next_seq: u64,
}

impl Serving {
    pub(super) fn new(layout: Layout) -> Self {
        let (kv, max_seqs) = match layout {
            Layout::Contiguous => (KvStore::Contiguous, 1),
            Layout::Paged(pool) => (
                KvStore::Paged {
                    pool,
                    cache: PagedKvCache::new(pool.max_seqs.get(), pool.blocks.get()),
                },
                pool.max_seqs.get(),
            ),
        };
        Self {
            kv,
            seqs: HashMap::new(),
            free_slots: (0..max_seqs).rev().collect(),
            next_seq: 0,
        }
    }

    /// The layout `prepare` admitted.
    pub(super) fn layout(&self) -> Layout {
        match &self.kv {
            KvStore::Contiguous => Layout::Contiguous,
            KvStore::Paged { pool, .. } => Layout::Paged(*pool),
        }
    }

    /// The layout every entry this driver runs has.
    pub(super) fn kv(&self) -> KvLayout {
        match &self.kv {
            KvStore::Contiguous => KvLayout::Contiguous,
            KvStore::Paged { pool, .. } => pool.kv(),
        }
    }

    /// Blocks free in the paged pool; `None` under the contiguous layout.
    pub(super) fn free_blocks(&self) -> Option<usize> {
        match &self.kv {
            KvStore::Contiguous => None,
            KvStore::Paged { cache, .. } => Some(cache.free_blocks()),
        }
    }

    /// Drop the cached prefix blocks of an unloaded adapter generation.
    pub(super) fn evict_identities(&mut self, identities: &[PrefixIdentity]) {
        if let KvStore::Paged { cache, .. } = &mut self.kv {
            cache.evict_identities(identities);
        }
    }
}
