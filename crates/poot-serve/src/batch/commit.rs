use std::sync::atomic::Ordering;
use std::time::Instant;

use poot_llm::driver::PrefixOutcome;

use crate::batch::engine::{EngineError, RowResult, ServeEngine};
use crate::batch::slot::Slot;
use crate::metrics::{CacheMetrics, Metrics};
use crate::types::{GenEvent, PerTokenInfo};

/// What a request's commit decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotFate {
    /// The request goes on to its next step.
    Continue,
    /// The request ended: `natural` is an EOS or a stop string (OpenAI `"stop"`), else `max_new` ("length").
    Finished { natural: bool },
    /// The client is gone: a streamed piece found no receiver.
    Cancelled,
}

/// The stop strings of one request, matched against the generated text as it grows. Only a bounded tail
/// is kept: the last `longest - 1` bytes (moved back to a char boundary), which is all a match that
/// straddles into the next piece can need.
#[derive(Debug)]
pub(crate) struct StopScanner {
    strings: Vec<String>,
    longest: usize,
    tail: String,
}

impl StopScanner {
    pub(crate) fn new(strings: Vec<String>) -> Self {
        let strings: Vec<String> = strings.into_iter().filter(|s| !s.is_empty()).collect();
        let longest = strings.iter().map(String::len).max().unwrap_or(0);
        Self {
            strings,
            longest,
            tail: String::new(),
        }
    }

    /// Whether the request has any stop string to scan for.
    pub(crate) fn active(&self) -> bool {
        self.longest > 0
    }

    /// Record `piece`, the text the latest token added; whether a stop string is now complete.
    pub(crate) fn push(&mut self, piece: &str) -> bool {
        if self.longest == 0 {
            return false;
        }
        self.tail.push_str(piece);
        if self.strings.iter().any(|s| self.tail.contains(s.as_str())) {
            return true;
        }
        let mut from = self.tail.len().saturating_sub(self.longest - 1);
        while !self.tail.is_char_boundary(from) {
            from -= 1;
        }
        self.tail.drain(..from);
        false
    }
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
/// set, a stop string, `max_new`, a gone client), stream each kept token, then commit exactly the kept
/// prefix with the engine. A token after the ending one is never kept, so it enters neither the
/// sequence nor the sampler's history.
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
            fate = SlotFate::Finished { natural: true };
            break;
        }
        kept += 1;
        // Push BEFORE decoding the piece: the incremental detokenizer needs `next` with its left context
        // (a per-token decode strips leading spaces on SentencePiece models).
        slot.tokens.push(next);
        let piece = if slot.stream || slot.stops.active() {
            engine
                .piece(&slot.tokens, slot.prompt_len)
                .unwrap_or_else(|error| {
                    tracing::warn!(error = %error, "detokenize a generated token");
                    String::new()
                })
        } else {
            String::new()
        };
        let stopped = slot.stops.push(&piece);
        slot.generated += 1;
        metrics.completion_tokens.fetch_add(1, Ordering::Relaxed);
        if slot.stream {
            let now = Instant::now();
            let decode_start = *slot.decode_start.get_or_insert(now);
            let info = PerTokenInfo {
                token_idx: slot.generated - 1,
                since_prev_ms: slot
                    .last_token
                    .map_or(0.0, |t| (now - t).as_secs_f64() * 1000.0),
                cumulative_ms: (now - decode_start).as_secs_f64() * 1000.0,
                kv_used_ratio: engine.kv_used_ratio(),
            };
            slot.last_token = Some(now);
            if slot.reply.send(GenEvent::Token(piece, info)).is_err() {
                fate = SlotFate::Cancelled;
                break;
            }
        }
        if stopped {
            fate = SlotFate::Finished { natural: true };
            break;
        }
        if slot.generated >= slot.max_new {
            fate = SlotFate::Finished { natural: false };
            break;
        }
    }
    slot.logprobs
        .extend(result.logprobs.iter().take(kept).cloned());
    engine.commit(slot.seq, kept, &mut slot.sampler)?;
    Ok(fate)
}
