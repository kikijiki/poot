use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use poot_llm::{Sampler, TokenLogprob};

use crate::batch::commit::StopScanner;
use crate::metrics::Metrics;
use crate::types::{GenEvent, Job};

/// A request the loop has tokenized but not yet opened on the engine.
pub(crate) struct Waiting {
    pub(crate) job: Job,
    pub(crate) prompt: Vec<u32>,
}

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
    pub(crate) reply: Sender<GenEvent>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) logprobs: Vec<TokenLogprob>,
    /// Set on the first emitted token; `cumulative_ms` in `PerTokenInfo` is measured from it.
    pub(crate) decode_start: Option<Instant>,
    pub(crate) last_token: Option<Instant>,
}

impl<S> Slot<S> {
    /// Open `waiting` as `seq`, whose first `reused` prompt tokens are already resident.
    pub(crate) fn new(seq: S, waiting: Waiting, reused: usize, max_new: usize) -> Self {
        let Waiting { job, prompt } = waiting;
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
}

/// Reject work before admission. The job is consumed so its adapter lease is released whether or not
/// the failure event is delivered.
pub(crate) fn reject_job(job: Job, metrics: &Metrics, reason: impl Into<String>) {
    let _ = job.reply.send(GenEvent::failed(metrics, reason));
}
