//! The prefix cache: KV blocks of a shared prompt prefix reused across sequences (spec 046c, Card 735).
//! Keyed by the chained hash of full `BLOCK_SIZE`-token prefix blocks and the [`PrefixIdentity`] of the
//! weights that produced them. Nothing here touches a device.

use super::block_table::{BLOCK_SIZE, BlockPool};

/// The identity of the effective model weights a cached K/V block was computed under (spec 248 paged
/// prefix reuse, ADR-0030). Token hashes alone are unsafe to share across adapters: a LoRA adapter
/// changes the effective weights, so its K/V differs from the base model's or another adapter's for the
/// same tokens. `BASE` is the unadapted base model; a registered adapter's id is a monotonically
/// increasing value that changes on every (re)registration, so hot-unload-then-reload into the same pool
/// index never aliases the previous adapter's K/V. The representation deliberately makes the base model
/// distinct from every constructible adapter identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixIdentity(Option<std::num::NonZeroU64>);

impl PrefixIdentity {
    /// The unadapted base model's identity (no LoRA correction applied).
    pub const BASE: Self = PrefixIdentity(None);

    /// One immutable adapter incarnation. Pool-generated incarnation ids are never zero; enforcing that
    /// here prevents an adapter from ever aliasing [`Self::BASE`], even if a future caller bypasses the
    /// normal lease path.
    pub fn adapter(incarnation: u64) -> Self {
        Self(Some(
            std::num::NonZeroU64::new(incarnation)
                .expect("adapter prefix identity must be non-zero"),
        ))
    }
}

/// FNV-1a chained content hash: `h_0 = fnv(seed_off, block_0)`, `h_i = fnv(h_{i-1}, block_i)`. It is
/// position-aware (a block matches only when its whole preceding prefix matches), so two sequences share
/// a physical block iff they share that whole prefix (block `i`'s KV depends on every earlier token).
fn chain_hash(prev: u64, block: &[u32]) -> u64 {
    let mut h = prev ^ 0x100000001b3u64.wrapping_mul(0x517cc1b727220a95); // mix `prev` in
    for &t in block {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// What `release` did with a finished sequence's prompt prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixOutcome {
    /// The prompt's filled full blocks are cached under the sequence's identity: `tokens` of them.
    Registered {
        tokens: usize,
    },
    Skipped(PrefixSkip),
}

/// Why a released sequence registered no prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixSkip {
    /// Recurrent state folds every token into a value a cached block does not hold, so a later
    /// request reusing the blocks would run without that state.
    RecurrentState,
    /// The contiguous layout is one private cache with no block table: there are no blocks to cache.
    NoBlockTable,
    /// A preempted sequence is recomputed; its blocks go back to the pool.
    Preempted,
}

/// Automatic prefix cache (card 046c): the KV blocks of a shared prompt prefix are reused across
/// sequences instead of recomputed. Keyed by the pair of ([`chain_hash`] of the full
/// `BLOCK_SIZE`-token prefix blocks, [`PrefixIdentity`] of the effective model weights): a block matches
/// only when both the preceding tokens AND the weights that produced its K/V agree, so the base model and
/// each adapter keep separate entries even for identical tokens. The partial tail block a sequence is
/// still writing is never cached, so no copy-on-write is needed (shared blocks are complete and
/// read-only). The cache holds one reference to each cached block, so it survives after the producing
/// sequence frees, and releases it on eviction. This is the allocator/policy layer; skipping prefill
/// compute for reused blocks and the engine wiring are separate.
#[derive(Debug, Default)]
pub struct PrefixCache {
    /// (chained hash of a full prefix block, effective-weights identity) -> the physical block holding
    /// its KV.
    cached: std::collections::HashMap<(u64, PrefixIdentity), u32>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached prefix blocks.
    pub fn len(&self) -> usize {
        self.cached.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cached.is_empty()
    }

    /// Match (at admit): the longest cached prefix of `tokens` computed under `identity`. Walks the full
    /// `BLOCK_SIZE`-token blocks in order; for each whose (chained hash, identity) is cached, reuse that
    /// physical block (incref: this sequence now references it) and continue, stopping at the first miss.
    /// Returns the reused physical block ids and how many leading prompt tokens are already resident
    /// (`reused * BLOCK_SIZE`, the prefill the caller can skip). Allocates and registers nothing: the
    /// caller allocates the rest, fills `[skip, n)`, then (at finish, once those blocks hold valid KV)
    /// calls [`Self::register_for`]. Splitting match from register keeps it concurrency-correct: a block
    /// is reusable only after a finished sequence filled and registered it, so a still-filling sequence's
    /// blocks are never reused.
    pub fn match_prefix_for(
        &mut self,
        tokens: &[u32],
        identity: PrefixIdentity,
        pool: &mut BlockPool,
    ) -> (Vec<u32>, usize) {
        let mut out = Vec::new();
        let mut chain = 0xcbf29ce484222325u64; // FNV offset basis
        for block in tokens.chunks_exact(BLOCK_SIZE) {
            chain = chain_hash(chain, block);
            match self.cached.get(&(chain, identity)) {
                Some(&blk) => {
                    pool.incref(blk);
                    out.push(blk);
                }
                None => break, // the first uncached block ends the matched prefix
            }
        }
        let skip = out.len() * BLOCK_SIZE;
        (out, skip)
    }

    /// Register (at finish): make a sequence's filled full prefix blocks reusable under `identity`. For
    /// each full `BLOCK_SIZE`-token block `i` of `tokens`, if its (chained hash, identity) is not already
    /// cached, cache `physical[i]` (the cache takes a reference so the block survives after the sequence
    /// frees). Call only once the blocks hold valid KV (after the prefill). Idempotent; skips blocks
    /// already cached and any past the end of `physical`.
    pub fn register_for(
        &mut self,
        tokens: &[u32],
        physical: &[u32],
        identity: PrefixIdentity,
        pool: &mut BlockPool,
    ) {
        let mut chain = 0xcbf29ce484222325u64;
        for (i, block) in tokens.chunks_exact(BLOCK_SIZE).enumerate() {
            chain = chain_hash(chain, block);
            if i >= physical.len() {
                break;
            }
            if let std::collections::hash_map::Entry::Vacant(e) =
                self.cached.entry((chain, identity))
            {
                e.insert(physical[i]);
                pool.incref(physical[i]); // the cache's own reference (only for newly-cached blocks)
            }
        }
    }

    /// Drop a sequence's references to the blocks it held (call when the sequence ends). Registered
    /// (cached) blocks survive (the cache holds its own reference); unregistered blocks (the partial tail,
    /// generation blocks) hit refcount 0 and return to the pool.
    pub fn release(&self, blocks: &[u32], pool: &mut BlockPool) {
        for &b in blocks {
            pool.free_block(b);
        }
    }

    /// Evict every cached block keyed on one of `identities` (an unloaded adapter generation's): no
    /// live sequence can hold them, so each frees back to the pool. Returns the number freed.
    pub fn evict_identities(
        &mut self,
        identities: &[PrefixIdentity],
        pool: &mut BlockPool,
    ) -> usize {
        let stale: Vec<(u64, PrefixIdentity)> = self
            .cached
            .keys()
            .filter(|(_, identity)| identities.contains(identity))
            .copied()
            .collect();
        for key in &stale {
            let blk = self.cached.remove(key).expect("collected from the map");
            pool.free_block(blk);
        }
        stale.len()
    }

    /// Evict cached prefix blocks that only the cache still references (no live sequence uses them),
    /// freeing them back to the pool. The reclaim policy for when the pool is tight; in-use prefixes stay
    /// cached.
    pub fn evict_unused(&mut self, pool: &mut BlockPool) -> usize {
        let stale: Vec<(u64, PrefixIdentity)> = self
            .cached
            .iter()
            .filter(|&(_, &blk)| pool.refcount(blk) == 1)
            .map(|(&h, _)| h)
            .collect();
        for h in &stale {
            let blk = self.cached.remove(h).unwrap();
            pool.free_block(blk); // releases the cache's last reference -> back to the free list
        }
        stale.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `n` distinct token ids starting at `base` (a stand-in for a real prompt span).
    fn toks(base: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| base + i).collect()
    }
    #[test]
    fn prefix_cache_match_register_shares_prefix() {
        // card 046c: seq A fills + REGISTERS a 2-block prefix; seq B sharing it MATCHES (reuses) those blocks
        // and allocates only its divergent suffix block. match (at admit) + register (at finish, after fill)
        // are split so only filled blocks are reusable - concurrency-correct.
        let mut pool = BlockPool::new(8);
        let mut pc = PrefixCache::new();
        let a_tok = toks(0, 3 * BLOCK_SIZE); // 3 full blocks
        let mut b_tok = toks(0, 2 * BLOCK_SIZE); // shares the first 2 blocks
        b_tok.extend(toks(500, BLOCK_SIZE)); // a different 3rd block

        // seq A (cold): match reuses nothing; allocate its 3 blocks, "fill", register.
        let (a_reused, a_skip) = pc.match_prefix_for(&a_tok, PrefixIdentity::BASE, &mut pool);
        assert!(a_reused.is_empty() && a_skip == 0, "A is cold");
        let a_blocks: Vec<u32> = (0..3).map(|_| pool.alloc().unwrap()).collect();
        pc.register_for(&a_tok, &a_blocks, PrefixIdentity::BASE, &mut pool);
        assert_eq!(pool.available(), 8 - 3);

        // seq B: match reuses A's first 2 prefix blocks; allocate 1 fresh for its 3rd.
        let (b_reused, b_skip) = pc.match_prefix_for(&b_tok, PrefixIdentity::BASE, &mut pool);
        assert_eq!(b_reused.len(), 2, "B reuses the 2 shared prefix blocks");
        assert_eq!(b_skip, 2 * BLOCK_SIZE);
        assert_eq!(
            b_reused,
            a_blocks[..2],
            "same physical blocks for the shared prefix"
        );
        let b_third = pool.alloc().unwrap();
        assert_ne!(b_third, a_blocks[2], "B's divergent block is fresh");
        assert_eq!(
            pool.capacity() - pool.available(),
            4,
            "only ONE new block for B"
        );
        // a shared block: A's alloc ref + the cache's register ref + B's match ref.
        assert_eq!(pool.refcount(a_blocks[0]), 3);
    }

    #[test]
    fn prefix_cache_match_only_reuses_registered_blocks() {
        // concurrency-correctness: a sequence that has MATCHED but not yet registered (its prefix is still
        // filling) is NOT reusable - a second sequence with the same prefix matches nothing until the first
        // registers (at finish, after fill).
        let mut pool = BlockPool::new(8);
        let mut pc = PrefixCache::new();
        let tok = toks(0, BLOCK_SIZE); // 1 full block

        // A admits (cold), allocates, but has NOT registered yet (still "filling").
        let (a_reused, _) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool);
        assert!(a_reused.is_empty());
        let a_blk = pool.alloc().unwrap();
        // B admits with the same prefix BEFORE A registered -> reuses nothing (A's block isn't filled).
        let (b_reused, b_skip) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool);
        assert!(
            b_reused.is_empty() && b_skip == 0,
            "unregistered prefix is not reused"
        );

        // A finishes (fills) and registers; now a later sequence C reuses it.
        pc.register_for(&tok, &[a_blk], PrefixIdentity::BASE, &mut pool);
        let (c_reused, c_skip) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool);
        assert_eq!(c_reused, [a_blk]);
        assert_eq!(c_skip, BLOCK_SIZE);
    }

    #[test]
    fn prefix_cache_identity_separates_base_and_adapters() {
        // Spec 248: a token prefix is reusable only under the SAME effective weights. The base model and
        // an adapter, and two different adapters, must never share K/V even for identical tokens.
        let mut pool = BlockPool::new(16);
        let mut pc = PrefixCache::new();
        let tok = toks(0, BLOCK_SIZE);
        let adapter_a = PrefixIdentity::adapter(1);
        let adapter_b = PrefixIdentity::adapter(2);

        // Adapter A fills and registers its prefix.
        let (_a, _) = pc.match_prefix_for(&tok, adapter_a, &mut pool);
        let a_blk = pool.alloc().unwrap();
        pc.register_for(&tok, &[a_blk], adapter_a, &mut pool);

        // The base model and adapter B must miss; adapter A (unchanged) must hit.
        let (base_reused, base_skip) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool);
        assert!(
            base_reused.is_empty() && base_skip == 0,
            "base model must not reuse an adapter's K/V"
        );
        let (b_reused, b_skip) = pc.match_prefix_for(&tok, adapter_b, &mut pool);
        assert!(
            b_reused.is_empty() && b_skip == 0,
            "a different adapter must not reuse adapter A's K/V"
        );
        let (a_reused, a_skip) = pc.match_prefix_for(&tok, adapter_a, &mut pool);
        assert_eq!(a_reused, [a_blk], "the same adapter reuses its own prefix");
        assert_eq!(a_skip, BLOCK_SIZE);

        // The base model can cache independently under BASE without colliding with adapter A.
        let base_blk = pool.alloc().unwrap();
        assert_ne!(base_blk, a_blk, "distinct identities get distinct blocks");
        pc.register_for(&tok, &[base_blk], PrefixIdentity::BASE, &mut pool);
        let (base_reused, _) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool);
        assert_eq!(base_reused, [base_blk]);
        assert_eq!(pc.len(), 2, "one entry per (tokens, identity)");
        pc.release(&[a_blk, base_blk], &mut pool);
    }

    #[test]
    fn prefix_cache_identity_change_on_reload_never_hits_the_old_adapter() {
        // Spec 248 FR/pool-index reuse: a hot-unload then hot-load reuses the same pool index but the
        // adapter is different, so its identity changes. The new identity must never match K/V registered
        // by the previous occupant of the index.
        let mut pool = BlockPool::new(16);
        let mut pc = PrefixCache::new();
        let tok = toks(0, 2 * BLOCK_SIZE);
        let old_adapter = PrefixIdentity::adapter(7);
        let new_adapter = PrefixIdentity::adapter(8); // reloaded into the SAME pool index

        let (_a, _) = pc.match_prefix_for(&tok[..BLOCK_SIZE], old_adapter, &mut pool);
        let old_blk = pool.alloc().unwrap();
        pc.register_for(&tok[..BLOCK_SIZE], &[old_blk], old_adapter, &mut pool);

        let (new_reused, new_skip) =
            pc.match_prefix_for(&tok[..BLOCK_SIZE], new_adapter, &mut pool);
        assert!(
            new_reused.is_empty() && new_skip == 0,
            "a reloaded adapter in the same pool index must never reuse the prior adapter's K/V"
        );
        // The old identity is no longer selectable (its lease would have blocked unload), so only the
        // eviction path can reclaim it; the new adapter simply caches separately.
        let new_blk = pool.alloc().unwrap();
        assert_ne!(new_blk, old_blk);
        pc.register_for(&tok[..BLOCK_SIZE], &[new_blk], new_adapter, &mut pool);
        assert_eq!(
            pc.match_prefix_for(&tok[..BLOCK_SIZE], new_adapter, &mut pool)
                .0,
            [new_blk]
        );
        pc.release(&[old_blk, new_blk], &mut pool);
    }

    #[test]
    fn prefix_cache_refcount_keeps_shared_blocks_alive() {
        // freeing one sequence must NOT free a block the cache (or another sequence) still references; the
        // block frees only when the last holder releases and the cache evicts it.
        let mut pool = BlockPool::new(8);
        let mut pc = PrefixCache::new();
        let tok = toks(0, BLOCK_SIZE); // 1 full block

        let (_a, _) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool); // cold
        let blkid = pool.alloc().unwrap(); // A's ref (1)
        pc.register_for(&tok, &[blkid], PrefixIdentity::BASE, &mut pool); // + cache ref (2)
        let (b, _) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool); // + B's ref (3)
        assert_eq!(b, [blkid]);
        assert_eq!(pool.refcount(blkid), 3); // A + cache + B

        pc.release(&[blkid], &mut pool); // A releases -> 2
        assert_eq!(pool.refcount(blkid), 2);
        pc.release(&b, &mut pool); // B releases -> 1 (cache only)
        assert_eq!(pool.refcount(blkid), 1);
        assert_eq!(
            pool.available(),
            pool.capacity() - 1,
            "cached block not yet freed"
        );

        let n = pc.evict_unused(&mut pool);
        assert_eq!(n, 1);
        assert!(pc.is_empty());
        assert_eq!(
            pool.available(),
            pool.capacity(),
            "evicted block returned to the pool"
        );
    }

    #[test]
    fn evict_unused_keeps_a_block_a_live_sequence_still_holds() {
        // Safety invariant: evict_unused must reclaim ONLY cache-exclusive blocks (refcount == 1). A cached
        // block a LIVE sequence still references (refcount > 1) must survive - freeing it would be a
        // use-after-free of that sequence's KV. The existing lifecycle test only evicts once the block is
        // already cache-exclusive; this pins the no-op-while-held case.
        let mut pool = BlockPool::new(8);
        let mut pc = PrefixCache::new();
        let tok = toks(0, BLOCK_SIZE); // 1 full block

        pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool); // producer admits cold
        let blk = pool.alloc().unwrap(); // producer's ref (1)
        pc.register_for(&tok, &[blk], PrefixIdentity::BASE, &mut pool); // + cache ref (2)
        let (b, _) = pc.match_prefix_for(&tok, PrefixIdentity::BASE, &mut pool); // a live consumer reuses it (3)
        assert_eq!(b, [blk]);
        assert_eq!(pool.refcount(blk), 3);

        // Producer finishes, but consumer B is STILL live. evict must free NOTHING.
        pc.release(&[blk], &mut pool); // -> refcount 2 (cache + live B)
        assert_eq!(pool.refcount(blk), 2);
        assert_eq!(
            pc.evict_unused(&mut pool),
            0,
            "a cached block a live sequence still holds must NOT be evicted"
        );
        assert_eq!(
            pool.refcount(blk),
            2,
            "still held by cache + the live sequence"
        );
        assert!(!pc.is_empty(), "the cache entry survives");
        assert_eq!(
            pool.available(),
            pool.capacity() - 1,
            "the held block did not return to the pool"
        );

        // Once B releases too (cache-exclusive), evict reclaims it.
        pc.release(&b, &mut pool); // -> refcount 1 (cache only)
        assert_eq!(pc.evict_unused(&mut pool), 1);
        assert_eq!(pool.available(), pool.capacity(), "now reclaimed");
    }
}
