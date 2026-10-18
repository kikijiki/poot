use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use poot_llm::driver::error::FailureClass;
use poot_llm::{Sampler, TokenLogprob};

use crate::metrics::Metrics;
use crate::types::{
    ChoiceReply, Completion, FinishReason, GenEvent, Job, PerTokenInfo, PromptInput,
};
use poot_llm::driver::stops::StopScanner;

/// A request that holds an engine sequence: its committed tokens, its sampler and the channel back to
/// its connection thread.
pub(crate) struct Slot<S> {
    pub(crate) seq: S,
    /// The prompt, then the generated tokens committed so far.
    pub(crate) tokens: Vec<u32>,
    pub(crate) prompt_len: usize,
    /// Prompt tokens the engine has consumed, counting a reused prefix.
    pub(crate) prefilled: usize,
    pub(crate) generated: usize,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    pub(crate) stops: StopScanner,
    pub(crate) stream: bool,
    /// The generated text released so far, cut before a stop string.
    pub(crate) text: String,
    pub(crate) reply: ChoiceReply,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) logprobs: Vec<TokenLogprob>,
    /// Set on the first emitted token; `cumulative_ms` in `PerTokenInfo` is measured from it.
    pub(crate) decode_start: Option<Instant>,
    pub(crate) last_token: Option<Instant>,
}

impl<S> Slot<S> {
    /// Open `job` as `seq`, whose first `reused` prompt tokens are already resident.
    pub(crate) fn new(seq: S, job: Job, reused: usize, max_new: usize) -> Self {
        let (PromptInput::Tokens(prompt) | PromptInput::Multimodal { tokens: prompt, .. }) =
            job.prompt;
        Self {
            seq,
            prompt_len: prompt.len(),
            tokens: prompt,
            prefilled: reused,
            generated: 0,
            max_new,
            sampler: job.sampler,
            stops: StopScanner::new(job.stop),
            stream: job.stream,
            text: String::new(),
            reply: job.reply,
            cancel: job.cancel,
            logprobs: Vec::new(),
            decode_start: None,
            last_token: None,
        }
    }

    /// Whether the prompt still has tokens to feed.
    pub(crate) fn prefilling(&self) -> bool {
        self.prefilled < self.prompt_len
    }

    /// The client stopped waiting for this request.
    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Whether the loop detokenizes this request's tokens one at a time: it streams them, or scans them
    /// for a stop string. Any other request has its text decoded once, when it ends.
    pub(crate) fn tracks_text(&self) -> bool {
        self.stream || self.stops.active()
    }

    /// Stream `piece` with its timing; false when the receiver is gone.
    pub(crate) fn send_piece(&mut self, piece: String, kv_used_ratio: f64) -> bool {
        let now = Instant::now();
        let decode_start = *self.decode_start.get_or_insert(now);
        let info = PerTokenInfo {
            token_idx: self.generated - 1,
            since_prev_ms: self
                .last_token
                .map_or(0.0, |t| (now - t).as_secs_f64() * 1000.0),
            cumulative_ms: (now - decode_start).as_secs_f64() * 1000.0,
            kv_used_ratio,
        };
        self.last_token = Some(now);
        self.reply.send(GenEvent::Token(piece, info)).is_ok()
    }

    /// The finished request's result.
    pub(crate) fn into_completion(self, finish: FinishReason) -> (ChoiceReply, Completion) {
        let Self {
            mut tokens,
            prompt_len,
            text,
            logprobs,
            reply,
            ..
        } = self;
        let generated = tokens.split_off(prompt_len);
        (
            reply,
            Completion {
                prompt_tokens: prompt_len,
                tokens: generated,
                text,
                logprobs,
                finish,
            },
        )
    }
}

/// Reject work before admission. The job is consumed so its adapter lease is released whether or not
/// the failure event is delivered.
pub(crate) fn reject_job(
    job: Job,
    metrics: &Metrics,
    class: FailureClass,
    reason: impl Into<String>,
) {
    let _ = job.reply.send(GenEvent::failed(metrics, class, reason));
}
