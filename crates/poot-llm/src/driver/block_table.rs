//! Paged KV block management (spec 045 / card 046a, moved into the driver by Card 735): a fixed pool of
//! fixed-size KV blocks and a per-sequence block table mapping logical token positions to physical block
//! slots. The driver owns them; nothing here touches the GPU.
//! Paged KV stays static-capacity (a fixed pool), so capture/replay is unchanged: the block table is a
//! device input and the K/V fetch is indirected through it (see the spec).
//!
//! The block-table approach follows vLLM's PagedAttention (not copied).

use super::prefix_cache::{PrefixCache, PrefixIdentity};

/// The block pool has no free block: the caller evicts a cached prefix or preempts a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("paged KV block pool exhausted")]
pub struct PoolExhausted;

/// Tokens of K (and V) per physical block. vLLM uses 16; small enough to limit waste, large enough to keep
/// the gather cheap.
pub const BLOCK_SIZE: usize = 16;

/// A free-list allocator over a fixed number of physical KV blocks (sized once, like the contiguous KV
/// buffer, so paged KV stays static-capacity). `alloc` hands out a physical block id; `free` returns it
/// for reuse.
#[derive(Debug)]
pub struct BlockPool {
    /// available physical block ids (a stack; `alloc` pops, `free` pushes).
    free: Vec<u32>,
    /// per-block reference count: how many holders (sequences + the prefix cache) reference each block. A
    /// block returns to `free` only when its count hits 0. Plain alloc/free is the refcount-1 case;
    /// counts > 1 let a shared prefix block (card 046c) be referenced by several sequences at once.
    refcount: Vec<u16>,
    num_blocks: usize,
}

impl BlockPool {
    /// A pool of `num_blocks` physical blocks, all initially free.
    pub fn new(num_blocks: usize) -> Self {
        // store descending so `pop` (alloc) hands out ascending ids first (deterministic, easy to test)
        BlockPool {
            free: (0..num_blocks as u32).rev().collect(),
            refcount: vec![0; num_blocks],
            num_blocks,
        }
    }

    /// Total physical blocks in the pool.
    pub fn capacity(&self) -> usize {
        self.num_blocks
    }

    /// Currently free (allocatable) blocks.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Take a free physical block id (refcount 1), or `None` if the pool is exhausted.
    pub fn alloc(&mut self) -> Option<u32> {
        let id = self.free.pop()?;
        debug_assert_eq!(self.refcount[id as usize], 0, "alloc of a referenced block");
        self.refcount[id as usize] = 1;
        Some(id)
    }

    /// Add a reference to an already-allocated block (a second sequence reuses a shared prefix block).
    pub fn incref(&mut self, id: u32) {
        debug_assert!(self.refcount[id as usize] > 0, "incref of a free block");
        self.refcount[id as usize] += 1;
    }

    /// The reference count of a block (for tests / eviction policy).
    pub fn refcount(&self, id: u32) -> u16 {
        self.refcount[id as usize]
    }

    /// Drop one reference to a physical block; return it to the pool only when the last holder releases it.
    pub fn free_block(&mut self, id: u32) {
        debug_assert!((id as usize) < self.num_blocks, "block id out of range");
        let rc = &mut self.refcount[id as usize];
        debug_assert!(*rc > 0, "free of an unreferenced block");
        *rc -= 1;
        if *rc == 0 {
            self.free.push(id);
        }
    }
}

/// A per-sequence block table: logical block index -> physical block id, plus the token count. As the
/// sequence grows, new physical blocks are pulled from a [`BlockPool`]; freeing returns them.
#[derive(Debug, Default, Clone)]
pub struct BlockTable {
    /// logical block `i` -> physical block id; `blocks[pos / BLOCK_SIZE]` holds token `pos`.
    blocks: Vec<u32>,
    /// number of tokens currently stored.
    len: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tokens stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The physical block ids for this sequence (logical order). This is what the decode graph supplies as
    /// the per-sequence block-table device input.
    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// Build a table from already-assigned physical blocks (e.g. the reused full prefix blocks a
    /// [`PrefixCache`] hands back), holding `blocks.len() * BLOCK_SIZE` tokens. Further [`Self::append`]s
    /// continue from there, drawing fresh blocks from the pool. The adopted blocks must already be
    /// referenced (by the cache/caller); `append`/`free` manage only the new ones. Card 046c.
    pub fn adopt(blocks: Vec<u32>) -> Self {
        let len = blocks.len() * BLOCK_SIZE;
        BlockTable { blocks, len }
    }

    /// Append one token, allocating a fresh physical block from `pool` when crossing a block boundary.
    /// Fails (without growing) if the pool is exhausted - that is the scheduler's problem (046b), never a
    /// silent overwrite of another sequence's block (FR-004).
    pub fn append(&mut self, pool: &mut BlockPool) -> Result<(), PoolExhausted> {
        if self.len.is_multiple_of(BLOCK_SIZE) {
            let id = pool.alloc().ok_or(PoolExhausted)?;
            self.blocks.push(id);
        }
        self.len += 1;
        Ok(())
    }

    /// Append `n` tokens, allocating blocks as needed (all-or-nothing: on pool exhaustion the table is left
    /// exactly as it was so the caller can retry after eviction).
    pub fn extend(&mut self, n: usize, pool: &mut BlockPool) -> Result<(), PoolExhausted> {
        let blocks_before = self.blocks.len();
        let len_before = self.len;
        for _ in 0..n {
            if let Err(e) = self.append(pool) {
                // roll back any blocks we grabbed this call.
                for &id in &self.blocks[blocks_before..] {
                    pool.free_block(id);
                }
                self.blocks.truncate(blocks_before);
                self.len = len_before;
                return Err(e);
            }
        }
        Ok(())
    }

    /// The flat physical slot (in units of tokens, into the block pool) for a logical token position:
    /// `physical_block * BLOCK_SIZE + offset_within_block`. `None` if the position is out of range. This is
    /// the index the paged-attention gather uses to fetch the K/V row for key position `logical_pos`.
    pub(crate) fn physical_slot(&self, logical_pos: usize) -> Option<usize> {
        if logical_pos >= self.len {
            return None;
        }
        let phys_block = self.blocks[logical_pos / BLOCK_SIZE] as usize;
        Some(phys_block * BLOCK_SIZE + logical_pos % BLOCK_SIZE)
    }

    /// The `[cap]`-length gather index vector the paged decode graph binds to its slot-map input: logical
    /// key position `t` maps to the flat physical slot holding its K/V (`physical_slot(t)`) for `t < len`.
    /// Positions `t >= len` are not written yet and are masked out of attention, so they get slot 0 (any
    /// valid index works - they are never read). The decode graph gathers the physical-order K/V
    /// cache by this vector to reconstruct logical order, then runs the usual masked attention.
    pub(crate) fn slot_mapping(&self, cap: usize) -> Vec<u32> {
        (0..cap)
            .map(|t| self.physical_slot(t).unwrap_or(0) as u32)
            .collect()
    }

    /// Return all of this sequence's blocks to the pool and reset to empty.
    pub fn free(&mut self, pool: &mut BlockPool) {
        for &id in &self.blocks {
            pool.free_block(id);
        }
        self.blocks.clear();
        self.len = 0;
    }
}

/// A shared paged-KV cache across several decode slots (card 046b): one block pool shared by `n_slots`
/// per-sequence block tables. Concurrent sequences draw blocks from the shared pool as they grow, so the
/// pool is sized for aggregate usage, not `n_slots * max_seq`: the memory win over the contiguous scheme,
/// where every slot reserves `max_seq`. Allocation can fail (pool exhausted), which signals the scheduler
/// to evict; the eviction policy lives in the serving engine (poot-serve).
#[derive(Debug)]
pub struct PagedKvCache {
    pool: BlockPool,
    tables: Vec<BlockTable>,
    /// automatic prefix cache (card 046c) over the SAME pool: shared prompt prefixes reuse blocks across slots.
    prefix: PrefixCache,
}

impl PagedKvCache {
    /// A cache of `n_slots` empty sequences sharing a pool of `num_blocks` physical blocks.
    pub fn new(n_slots: usize, num_blocks: usize) -> Self {
        PagedKvCache {
            pool: BlockPool::new(num_blocks),
            tables: (0..n_slots).map(|_| BlockTable::new()).collect(),
            prefix: PrefixCache::new(),
        }
    }

    /// Admit a sequence into slot `s` with identity-scoped prefix-cache reuse (card 046c + spec 248):
    /// reuse the blocks of any cached matching prompt prefix registered under the SAME effective-weights
    /// `identity` and allocate fresh blocks for the rest of the prompt + `max_new` generated tokens.
    /// Returns `skip`, the number of leading prompt tokens already resident (the prefill the caller skips
    /// by starting the slot's decode at `pos = skip`). Guaranteed `skip < prompt.len()`: the final prompt
    /// token is never served from cache, so the caller has a token left to consume for the frontier
    /// logits. All-or-nothing: on pool exhaustion the slot is left empty (reused refs released) and the
    /// caller can evict + retry. The slot must be free (call after `reset`/`register_and_reset`).
    pub fn admit_prefix_for(
        &mut self,
        s: usize,
        prompt: &[u32],
        max_new: usize,
        identity: PrefixIdentity,
    ) -> Result<usize, PoolExhausted> {
        // Match against the prompt minus its final token: the caller resumes decode at `pos = skip` and
        // consumes `prompt[pos]` for the frontier logits, so one token must stay un-skipped. A full-prompt
        // hit (a BLOCK_SIZE-multiple prompt with every block cached) would return `skip == prompt.len()`
        // and the engine would index out of bounds. Slicing off the last token keeps every full chunk's
        // boundary (and chained hash) identical; the final block is never matchable here and is
        // recomputed into fresh blocks by the resumed decode.
        let matchable = &prompt[..prompt.len().saturating_sub(1)];
        let (reused, skip) = self
            .prefix
            .match_prefix_for(matchable, identity, &mut self.pool);
        let mut table = BlockTable::adopt(reused);
        // the non-cached rest of the prompt + the generation draw fresh blocks.
        let need = (prompt.len() - skip) + max_new;
        if let Err(e) = table.extend(need, &mut self.pool) {
            self.prefix.release(table.blocks(), &mut self.pool); // give back the reused refs
            return Err(e);
        }
        self.tables[s] = table;
        Ok(skip)
    }

    /// Finish slot `s`: register its (now-filled) prompt prefix blocks for future reuse under the
    /// effective-weights `identity`, then free the slot's references (registered prefix blocks survive in
    /// the cache; the tail + generation blocks return to the pool). `prompt` is the slot's prompt tokens.
    /// Card 046c: the finish side of [`Self::admit_prefix_for`].
    pub fn register_and_reset_for(&mut self, s: usize, prompt: &[u32], identity: PrefixIdentity) {
        self.prefix
            .register_for(prompt, self.tables[s].blocks(), identity, &mut self.pool);
        self.tables[s].free(&mut self.pool);
    }

    /// Evict cached prefix blocks no live slot references, reclaiming pool space (the scheduler's signal when
    /// admission fails). Returns the number freed.
    pub fn evict_prefix(&mut self) -> usize {
        self.prefix.evict_unused(&mut self.pool)
    }

    /// Drop the cached prefix blocks of `identities` (see [`PrefixCache::evict_identities`]).
    pub fn evict_identities(&mut self, identities: &[PrefixIdentity]) -> usize {
        self.prefix.evict_identities(identities, &mut self.pool)
    }

    /// Free blocks remaining in the shared pool.
    pub fn free_blocks(&self) -> usize {
        self.pool.available()
    }

    /// Tokens currently held by slot `s`.
    pub fn seq_len(&self, s: usize) -> usize {
        self.tables[s].len()
    }

    /// Grow slot `s`'s sequence by `n` tokens, drawing blocks from the shared pool as boundaries are
    /// crossed. All-or-nothing: on pool exhaustion the slot is unchanged and the caller can evict + retry.
    pub fn append(&mut self, s: usize, n: usize) -> Result<(), PoolExhausted> {
        self.tables[s].extend(n, &mut self.pool)
    }

    /// The `[cap]` gather index vector for slot `s` (what the paged decode graph binds as its slot map).
    pub(crate) fn slot_mapping(&self, s: usize, cap: usize) -> Vec<u32> {
        self.tables[s].slot_mapping(cap)
    }

    /// Free slot `s`'s blocks back to the shared pool and clear it (on request completion / eviction).
    pub fn reset(&mut self, s: usize) {
        self.tables[s].free(&mut self.pool);
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
    fn pool_alloc_free_roundtrip() {
        let mut pool = BlockPool::new(3);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.available(), 3);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_eq!((a, b), (0, 1)); // ascending ids first
        assert_eq!(pool.available(), 1);
        pool.free_block(a);
        assert_eq!(pool.available(), 2);
        // exhaust
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_none(), "pool exhausted");
    }

    #[test]
    fn append_allocates_blocks_on_boundary() {
        let mut pool = BlockPool::new(8);
        let mut t = BlockTable::new();
        // first token allocates the first block.
        t.append(&mut pool).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.blocks().len(), 1);
        // fill the rest of block 0 (BLOCK_SIZE tokens total) - still one block.
        for _ in 1..BLOCK_SIZE {
            t.append(&mut pool).unwrap();
        }
        assert_eq!(t.len(), BLOCK_SIZE);
        assert_eq!(
            t.blocks().len(),
            1,
            "exactly one block for BLOCK_SIZE tokens"
        );
        // the (BLOCK_SIZE+1)th token crosses the boundary -> a second block.
        t.append(&mut pool).unwrap();
        assert_eq!(t.blocks().len(), 2);
        assert_eq!(pool.available(), 8 - 2);
    }

    #[test]
    fn physical_slot_maps_through_noncontiguous_blocks() {
        // two sequences interleave so seqA gets NON-contiguous physical blocks (0 then 2).
        let mut pool = BlockPool::new(3);
        let mut a = BlockTable::new();
        let mut b = BlockTable::new();
        a.extend(BLOCK_SIZE, &mut pool).unwrap(); // a: block 0
        b.extend(BLOCK_SIZE, &mut pool).unwrap(); // b: block 1
        a.append(&mut pool).unwrap(); // a crosses -> block 2
        assert_eq!(a.blocks(), &[0, 2]);
        // logical pos 0 -> phys block 0, offset 0.
        assert_eq!(a.physical_slot(0), Some(0));
        // last token of block 0: logical 15 -> phys 0*16+15.
        assert_eq!(a.physical_slot(BLOCK_SIZE - 1), Some(BLOCK_SIZE - 1));
        // first token of the second logical block: logical 16 -> phys block 2, offset 0 = 32.
        assert_eq!(a.physical_slot(BLOCK_SIZE), Some(2 * BLOCK_SIZE));
        // out of range.
        assert_eq!(a.physical_slot(BLOCK_SIZE + 1), None);
    }

    #[test]
    fn extend_is_all_or_nothing_on_exhaustion() {
        let mut pool = BlockPool::new(1); // room for exactly one block (BLOCK_SIZE tokens)
        let mut t = BlockTable::new();
        // asking for BLOCK_SIZE+1 tokens needs 2 blocks but only 1 is available -> error, table unchanged.
        let r = t.extend(BLOCK_SIZE + 1, &mut pool);
        assert!(r.is_err(), "pool exhausted");
        assert_eq!(t.len(), 0, "rolled back");
        assert_eq!(t.blocks().len(), 0);
        assert_eq!(pool.available(), 1, "grabbed block returned on rollback");
    }

    #[test]
    fn slot_mapping_reconstructs_logical_order_through_blocks() {
        // seqA gets non-contiguous physical blocks [0, 2] (b holds block 1 between them).
        let mut pool = BlockPool::new(3);
        let mut a = BlockTable::new();
        let mut b = BlockTable::new();
        a.extend(BLOCK_SIZE, &mut pool).unwrap(); // a: block 0
        b.extend(BLOCK_SIZE, &mut pool).unwrap(); // b: block 1
        a.append(&mut pool).unwrap(); // a: + block 2, now len BLOCK_SIZE+1
        assert_eq!(a.blocks(), &[0, 2]);

        let cap = 3 * BLOCK_SIZE;
        let map = a.slot_mapping(cap);
        assert_eq!(map.len(), cap);
        // logical 0..BLOCK_SIZE -> physical block 0 (slots 0..BLOCK_SIZE).
        for (t, &slot) in map.iter().enumerate().take(BLOCK_SIZE) {
            assert_eq!(slot, t as u32);
        }
        // logical BLOCK_SIZE -> physical block 2, offset 0 = slot 2*BLOCK_SIZE.
        assert_eq!(map[BLOCK_SIZE], (2 * BLOCK_SIZE) as u32);
        // unwritten positions (t >= len) are padded with slot 0 (masked out of attention).
        assert!(map[BLOCK_SIZE + 1..].iter().all(|&s| s == 0));
    }

    #[test]
    fn shared_cache_draws_from_one_pool_across_slots() {
        let mut cache = PagedKvCache::new(4, 6); // 4 slots, 6 shared blocks
        assert_eq!(cache.free_blocks(), 6);
        // grow two slots; each short sequence (< BLOCK_SIZE) uses one block from the SHARED pool.
        cache.append(0, 5).unwrap();
        cache.append(1, 10).unwrap();
        assert_eq!(cache.seq_len(0), 5);
        assert_eq!(cache.seq_len(1), 10);
        assert_eq!(
            cache.free_blocks(),
            4,
            "two slots took one block each from the shared pool"
        );
        // freeing a slot returns its block(s) to the shared pool for another slot.
        cache.reset(0);
        assert_eq!(cache.free_blocks(), 5);
        assert_eq!(cache.seq_len(0), 0);
    }

    #[test]
    fn shared_pool_serves_more_short_sequences_than_contiguous_reserves() {
        // the paged win: with a max_seq of 1024 (64 blocks), the contiguous scheme reserves 64 blocks PER
        // slot - 8 slots would need 512 blocks. Paged shares one pool: 8 SHORT sequences (<= BLOCK_SIZE
        // tokens) fit in 8 blocks. Here a pool of 8 blocks serves all 8 slots; the contiguous scheme could
        // not (it would need 8 * 64).
        let mut cache = PagedKvCache::new(8, 8);
        for s in 0..8 {
            cache.append(s, 12).unwrap(); // each slot: 12 tokens -> 1 block
        }
        assert_eq!(
            cache.free_blocks(),
            0,
            "8 short sequences exactly fill the 8-block pool"
        );
        for s in 0..8 {
            assert_eq!(cache.seq_len(s), 12);
        }
        // a 9th token on any slot would need a new block (boundary not crossed yet at 12) - but growing one
        // slot past its block boundary now fails: the shared pool is exhausted (the scheduler's evict signal).
        assert!(
            cache.append(0, BLOCK_SIZE).is_err(),
            "pool exhausted -> evict signal"
        );
    }

    #[test]
    fn slot_mapping_per_slot_is_independent() {
        let mut cache = PagedKvCache::new(2, 4);
        cache.append(0, 3).unwrap();
        cache.append(1, 3).unwrap();
        // distinct slots get distinct physical blocks, so their slot maps point at different slots.
        let m0 = cache.slot_mapping(0, BLOCK_SIZE);
        let m1 = cache.slot_mapping(1, BLOCK_SIZE);
        assert_ne!(
            m0[0], m1[0],
            "the two slots occupy different physical blocks"
        );
    }

    #[test]
    fn free_returns_blocks_for_reuse() {
        let mut pool = BlockPool::new(2);
        let mut t = BlockTable::new();
        t.extend(BLOCK_SIZE + 1, &mut pool).unwrap(); // 2 blocks
        assert_eq!(pool.available(), 0);
        t.free(&mut pool);
        assert_eq!(pool.available(), 2, "blocks returned to the pool");
        assert!(t.is_empty());
        // the pool is fully reusable.
        let mut t2 = BlockTable::new();
        t2.extend(BLOCK_SIZE, &mut pool).unwrap();
        assert_eq!(t2.blocks().len(), 1);
    }

    #[test]
    fn paged_cache_prefix_admit_reuses_registered_prefix() {
        // card 046c engine integration: slot 0 admits cold + finishes (registering its prompt prefix); slot 1
        // admitting the same prompt REUSES the registered prefix blocks (skip > 0), so the engine skips that
        // prefill. This is what `batch_engine_loop` calls (admit_prefix at admit, register_and_reset at finish).
        let mut c = PagedKvCache::new(2, 8); // 2 slots, 8 blocks
        let prompt = toks(0, 2 * BLOCK_SIZE + 3); // 2 full blocks + a partial tail

        let skip0 = c
            .admit_prefix_for(0, &prompt, 4, PrefixIdentity::BASE)
            .unwrap();
        assert_eq!(skip0, 0, "slot 0 is cold (nothing cached)");
        c.register_and_reset_for(0, &prompt, PrefixIdentity::BASE); // registers the 2 filled prompt blocks, frees slot 0

        let skip1 = c
            .admit_prefix_for(1, &prompt, 4, PrefixIdentity::BASE)
            .unwrap();
        assert_eq!(
            skip1,
            2 * BLOCK_SIZE,
            "slot 1 reuses the registered 2-block prefix"
        );
        // the two cached prefix blocks are shared, so fewer pool blocks are used than two cold admits.
        c.register_and_reset_for(1, &prompt, PrefixIdentity::BASE);
    }

    #[test]
    fn prefix_cache_admit_reuses_only_the_adapter_that_registered_it() {
        // The engine-facing path: slot 0 admits and finishes under adapter A; slot 1 admitting the same
        // prompt under adapter B gets no skip, while a third admit under adapter A does.
        let mut c = PagedKvCache::new(3, 16);
        let prompt = toks(0, 2 * BLOCK_SIZE + 3);
        let adapter_a = PrefixIdentity::adapter(11);
        let adapter_b = PrefixIdentity::adapter(12);

        assert_eq!(
            c.admit_prefix_for(0, &prompt, 4, adapter_a).unwrap(),
            0,
            "slot 0 is cold"
        );
        c.register_and_reset_for(0, &prompt, adapter_a);

        assert_eq!(
            c.admit_prefix_for(1, &prompt, 4, adapter_b).unwrap(),
            0,
            "adapter B must not reuse adapter A's prefill"
        );
        c.register_and_reset_for(1, &prompt, adapter_b);

        assert_eq!(
            c.admit_prefix_for(2, &prompt, 4, adapter_a).unwrap(),
            2 * BLOCK_SIZE,
            "adapter A reuses the prefix it registered"
        );
        c.register_and_reset_for(2, &prompt, adapter_a);
    }

    #[test]
    fn paged_cache_prefix_full_hit_leaves_a_token_to_decode() {
        // Regression: a prompt whose length is an exact BLOCK_SIZE multiple, served twice, used to match
        // all its blocks, so `admit_prefix` returned `skip == prompt.len()` and the next decode step's
        // `tokens[pos]` indexed out of bounds (killing the engine thread). The block covering the final
        // prompt token must never be matchable: the engine needs a token to consume to recompute the
        // frontier logits.
        let mut c = PagedKvCache::new(2, 16);
        let prompt = toks(0, 2 * BLOCK_SIZE); // exact multiple of BLOCK_SIZE
        let skip0 = c
            .admit_prefix_for(0, &prompt, 4, PrefixIdentity::BASE)
            .unwrap();
        assert_eq!(skip0, 0, "cold admit");
        c.register_and_reset_for(0, &prompt, PrefixIdentity::BASE);
        let skip1 = c
            .admit_prefix_for(1, &prompt, 4, PrefixIdentity::BASE)
            .unwrap();
        assert!(
            skip1 < prompt.len(),
            "a full-prompt hit must leave >= 1 token to decode (got skip {skip1} of {})",
            prompt.len()
        );
        assert_eq!(
            skip1, BLOCK_SIZE,
            "the final block is excluded from matching; earlier blocks still reuse"
        );
        c.register_and_reset_for(1, &prompt, PrefixIdentity::BASE);
    }

    #[test]
    fn admit_failure_recovers_after_evict_prefix() {
        // The serve engine's evict-and-retry: each distinct registered prefix leaves one cache-held block
        // behind, so without eviction the pool drains until no admission can succeed. `evict_prefix`
        // reclaims cache-exclusive blocks; a failed admit retried after eviction must succeed.
        let mut c = PagedKvCache::new(1, 4); // 1 slot, tiny 4-block pool
        // three distinct prompts (1 full block + tail each; 2 blocks live at admit) leave 3 cached blocks.
        for base in [0u32, 1000, 2000] {
            let p = toks(base, BLOCK_SIZE + 3);
            c.admit_prefix_for(0, &p, 4, PrefixIdentity::BASE).unwrap();
            c.register_and_reset_for(0, &p, PrefixIdentity::BASE);
        }
        assert_eq!(
            c.free_blocks(),
            1,
            "three distinct prefixes pinned 3 of 4 blocks"
        );
        // a fourth distinct prompt needs 2 blocks but only 1 is free -> the evict signal.
        let p = toks(3000, BLOCK_SIZE + 3);
        assert!(
            c.admit_prefix_for(0, &p, 4, PrefixIdentity::BASE).is_err(),
            "pool exhausted"
        );
        assert!(
            c.evict_prefix() > 0,
            "cache-exclusive prefix blocks reclaimed"
        );
        let skip = c
            .admit_prefix_for(0, &p, 4, PrefixIdentity::BASE)
            .expect("admission succeeds after eviction");
        assert_eq!(skip, 0, "the evicted prefixes are gone; this admit is cold");
    }
}
