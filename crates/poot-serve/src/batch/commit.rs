use std::sync::atomic::Ordering;

use poot_llm::driver::PrefixOutcome;

use crate::batch::engine::{EngineError, RowResult, ServeEngine};
use crate::batch::slot::Slot;
use crate::metrics::{CacheMetrics, Metrics};
use crate::types::FinishReason;

/// What a request's commit decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotFate {
    /// The request goes on to its next step.
    Continue,
    /// The request ended: an EOS, a stop string, or `max_new`.
    Finished { reason: FinishReason },
    /// The client is gone: a streamed piece found no receiver.
    Cancelled,
}

/// Route what a release did with a finished prompt's prefix into the cache telemetry.
pub(crate) fn record_prefix_outcome(cache: &CacheMetrics, outcome: PrefixOutcome) {
    match outcome {
        PrefixOutcome::Registered { tokens } => {
            cache
                .prefix_registered_tokens
                .fetch_add(tokens as u64, Ordering::Relaxed);
        }
        PrefixOutcome::Skipped(_) => {
            cache
                .prefix_skipped_releases
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Prefix-cache telemetry for one admission: the prompt tokens looked up and the leading ones found
/// resident.
pub(crate) fn record_prefix_admission(cache: &CacheMetrics, prompt_len: usize, reused: usize) {
    cache
        .prefix_query_tokens
        .fetch_add(prompt_len as u64, Ordering::Relaxed);
    cache
        .prefix_hit_tokens
        .fetch_add(reused as u64, Ordering::Relaxed);
}

/// Commit one step's picks for `slot`: keep tokens until one ends the request (an EOS from the model's
/// set, a stop string, `max_new`, a gone client), stream each kept token's released text, then commit
/// exactly the kept prefix with the engine. A token after the ending one is never kept, so it enters
/// neither the sequence nor the sampler's history.
pub(crate) fn commit_tokens<E: ServeEngine>(
    engine: &mut E,
    metrics: &Metrics,
    slot: &mut Slot<E::Seq>,
    result: &RowResult<E::Seq>,
) -> Result<SlotFate, EngineError> {
    let mut fate = SlotFate::Continue;
    let mut kept = 0;
    for &next in &result.tokens {
        if engine.eos().contains(&next) {
            fate = SlotFate::Finished {
                reason: FinishReason::Eos,
            };
            break;
        }
        kept += 1;
        // Push BEFORE decoding the piece: the incremental detokenizer needs `next` with its left context
        // (a per-token decode strips leading spaces on SentencePiece models).
        slot.tokens.push(next);
        let piece = if slot.tracks_text() {
            engine
                .piece(&slot.tokens, slot.prompt_len)
                .unwrap_or_else(|error| {
                    tracing::warn!(error = %error, "detokenize a generated token");
                    String::new()
                })
        } else {
            String::new()
        };
        let released = slot.stops.push(&piece);
        slot.generated += 1;
        metrics.completion_tokens.fetch_add(1, Ordering::Relaxed);
        if !released.is_empty() {
            slot.text.push_str(&released);
            if slot.stream && !slot.send_piece(released, engine.kv_used_ratio()) {
                fate = SlotFate::Cancelled;
                break;
            }
        }
        if slot.stops.hit() {
            fate = SlotFate::Finished {
                reason: FinishReason::Stop,
            };
            break;
        }
        if slot.generated >= slot.max_new {
            fate = SlotFate::Finished {
                reason: FinishReason::Length,
            };
            break;
        }
    }
    slot.logprobs
        .extend(result.logprobs.iter().take(kept).cloned());
    if matches!(fate, SlotFate::Finished { .. }) {
        flush_text(engine, slot, &mut fate);
    }
    engine.commit(slot.seq, kept, &mut slot.sampler)?;
    Ok(fate)
}

/// A finished request's text: what the stop scanner still holds goes out as its last piece (no stop
/// string completed on it), and a request that tracked no pieces has its text decoded once.
fn flush_text<E: ServeEngine>(engine: &mut E, slot: &mut Slot<E::Seq>, fate: &mut SlotFate) {
    if !slot.tracks_text() {
        slot.text = engine
            .decode(&slot.tokens[slot.prompt_len..])
            .unwrap_or_else(|error| {
                tracing::warn!(error = %error, "detokenize a finished request");
                String::new()
            });
        return;
    }
    let tail = slot.stops.finish();
    if tail.is_empty() {
        return;
    }
    slot.text.push_str(&tail);
    if slot.stream && !slot.send_piece(tail, engine.kv_used_ratio()) {
        *fate = SlotFate::Cancelled;
    }
}
