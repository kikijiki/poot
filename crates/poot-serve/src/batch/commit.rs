use std::sync::atomic::Ordering;
use std::time::Instant;

use poot_llm::Runner;
use poot_llm::driver::block_table::PagedKvCache;

use crate::batch::slot::Slot;
use crate::metrics::{CacheMetrics, Metrics};
use crate::types::{GenEvent, PerTokenInfo};

/// Free slot `j`'s KV blocks, registering its filled prompt prefix for reuse first. An mRoPE request
/// bypasses the token-only prefix cache, so it only frees.
pub(crate) fn release_slot_kv(paged_cache: &mut PagedKvCache, j: usize, slot: &Slot) {
    if slot.mrope.is_some() {
        paged_cache.reset(j);
    } else {
        // Register under this request's effective-weights identity (base, or its adapter): a later
        // request only reuses the prefix under the same weights (spec 248, ADR-0030).
        paged_cache.register_and_reset_for(
            j,
            &slot.tokens[..slot.prompt_len],
            slot.lora_adapter.prefix_identity(),
        );
    }
}

/// Prefix-cache telemetry for every non-mRoPE admission, including drivers whose extra state forced a
/// fresh reservation. mRoPE bypasses the token-only cache, so neither side of the hit-rate fraction moves.
pub(crate) fn record_non_mrope_prefix_cache_admission(
    cache: &CacheMetrics,
    prompt_len: usize,
    skip: usize,
    is_mrope: bool,
) {
    if !is_mrope {
        cache
            .prefix_query_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);
        cache
            .prefix_hit_tokens
            .fetch_add(skip as u64, Ordering::Relaxed);
    }
}

/// Commit one newly decided token `next` to slot `j`: EOS check, push and stream (with `PerTokenInfo`),
/// stop-sequence/`max_new` check, and prefix-cache registration plus slot eviction on any terminal
/// condition (EOS, stop, max_new, client cancellation), under one telemetry/cancellation/prefix-cache
/// contract. Returns `false` when the slot was evicted; the caller must stop feeding tokens to slot `j`.
pub(crate) fn commit_token(
    runner: &Runner,
    metrics: &Metrics,
    paged_cache: &mut PagedKvCache,
    num_blocks: usize,
    slots: &mut [Option<Slot>],
    j: usize,
    next: u32,
) -> bool {
    if next == runner.eos() {
        let mut slot = slots[j].take().unwrap();
        tracing::debug!(
            slot = j,
            generated = slot.generated,
            finish = "eos",
            "slot done"
        );
        // Register the filled prompt prefix for reuse, then free the blocks.
        release_slot_kv(paged_cache, j, &slot);
        let lp = slot.sampler.take_logprobs();
        // EOS -> natural stop ("stop"). `deliver_done` publishes the outcome count before the event
        // can be observed, as every terminal `Done` in the server does.
        GenEvent::deliver_done(metrics, &slot.reply, slot.tokens, lp, true);
        return false;
    }
    {
        let sl = slots[j].as_mut().unwrap();
        // push BEFORE streaming so the incremental detokenizer sees `next` with its left context - per-
        // token `decode(&[next])` strips leading spaces on SentencePiece models (update 0102).
        sl.tokens.push(next);
        if sl.stream
            && let Ok(piece) = runner.stream_piece(&sl.tokens, sl.prompt_len)
        {
            // card 030: engine-produced per-token timing. kv_used_ratio is the CURRENT global pool
            // occupancy (used / total blocks), read here on the engine thread.
            let now = Instant::now();
            let since_prev_ms = sl
                .last_token
                .map(|t| (now - t).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let decode_start = sl.decode_start.get_or_insert(now);
            let cumulative_ms = (now - *decode_start).as_secs_f64() * 1000.0;
            let kv_used_ratio = if num_blocks > 0 {
                (num_blocks - paged_cache.free_blocks()) as f64 / num_blocks as f64
            } else {
                0.0
            };
            let info = PerTokenInfo {
                token_idx: sl.generated,
                since_prev_ms,
                cumulative_ms,
                kv_used_ratio,
            };
            sl.last_token = Some(now);
            // a failed send means the connection thread dropped its receiver (the client
            // disconnected): evict the slot to stop generating and free it (cancellation).
            if sl.reply.send(GenEvent::Token(piece, info)).is_err() {
                // client gone: register the filled prompt prefix, then free the slot.
                metrics.requests_cancelled.fetch_add(1, Ordering::Relaxed);
                let slot = slots[j].take().unwrap();
                release_slot_kv(paged_cache, j, &slot);
                return false;
            }
        }
        // advance the sampler's per-token state: the guided-decoding constraint trie and the generated-
        // history penalties (the sequential paths call this inside their loops).
        sl.sampler.observe(next);
        sl.generated += 1;
    }
    metrics.completion_tokens.fetch_add(1, Ordering::Relaxed);
    // Halt on max_new or a stop sequence in the generated text (early stop; the connection thread trims
    // the stop string from the returned text).
    let sl = slots[j].as_ref().unwrap();
    let hit_stop = !sl.stop.is_empty()
        && runner
            .decode(&sl.tokens[sl.prompt_len..])
            .map(|t| {
                sl.stop
                    .iter()
                    .any(|s| !s.is_empty() && t.contains(s.as_str()))
            })
            .unwrap_or(false);
    if sl.generated >= sl.max_new || hit_stop {
        let mut slot = slots[j].take().unwrap();
        tracing::debug!(
            slot = j,
            generated = slot.generated,
            finish = if hit_stop { "stop" } else { "max_new" },
            "slot done"
        );
        release_slot_kv(paged_cache, j, &slot);
        let lp = slot.sampler.take_logprobs();
        // a stop sequence is a natural stop ("stop"); hitting max_new is "length".
        GenEvent::deliver_done(metrics, &slot.reply, slot.tokens, lp, hit_stop);
        return false;
    }
    true
}
