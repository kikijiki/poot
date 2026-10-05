//! Shared job, event, and backend types used across the server modules.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::time::Instant;

use poot_llm::encoder::{CrossEncoderRunner, EncoderRunner};
use poot_llm::{LoraAdapterLease, MropePositionIds, Runner, Sampler, TokenLogprob};

#[cfg(test)]
use crate::metrics::Metrics;

/// Global shutdown flag: set by the SIGINT/SIGTERM handler so the accept loop exits cleanly.
pub(crate) static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// A generation job submitted to the engine: the spec plus a reply channel the engine streams events back on.
#[allow(
    dead_code,
    reason = "the fields are the engine's input; no engine reads them yet"
)]
pub(crate) struct Job {
    pub(crate) prompt: String,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    /// stop sequences: the batched engine halts a slot as soon as its generated text contains one.
    pub(crate) stop: Vec<String>,
    /// stream token pieces as generated (SSE) vs only the final token list.
    pub(crate) stream: bool,
    pub(crate) reply: Sender<GenEvent>,
    /// Enqueue time, for the queue-wait histogram (`now - submitted` at admit).
    pub(crate) submitted: Instant,
    /// This request's LoRA adapter lease, acquired by the HTTP layer before enqueueing. Held through
    /// queue, slot, and preemption state so the adapter cannot be unloaded or reused until dropped.
    /// The inert form means "no adapter".
    pub(crate) lora_adapter: LoraAdapterLease,
    /// Prompt positions for an mRoPE Runner. The HTTP layer cannot build these yet; `None` gets a named
    /// admission rejection if the selected graph requires mRoPE.
    pub(crate) mrope_positions: Option<MropePositionIds>,
}

/// Engine-produced per-token data sent with each streamed token. Timing is measured inside the engine
/// (not client-side), so it is vendor-agnostic.
/// token_idx: 0-based position in the generated sequence (0 = first token after the prompt).
/// since_prev_ms: milliseconds since the previous emitted token (0.0 for the first token).
/// cumulative_ms: milliseconds since this request's first generated token.
/// kv_used_ratio: global KV block utilization at this step (used/total).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PerTokenInfo {
    pub(crate) token_idx: usize,
    pub(crate) since_prev_ms: f64,
    pub(crate) cumulative_ms: f64,
    pub(crate) kv_used_ratio: f64,
}

/// What the engine sends back per job: token pieces (streaming), then a terminal `Done` (the full token
/// sequence plus any recorded per-token logprobs) or `Failed`.
pub(crate) enum GenEvent {
    /// A generated token piece plus engine-produced per-token timing/state data.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "produced by the scheduling loop (card 588a)")
    )]
    Token(String, PerTokenInfo),
    /// The full token sequence + per-token logprobs, and `natural_stop`: true if generation ended on EOS
    /// or a stop sequence (OpenAI `"stop"`), false if it hit max_tokens / the KV cap (`"length"`).
    /// EOS is not appended to the sequence, so handlers cannot infer this from the tokens.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "produced by the scheduling loop (card 588a)")
    )]
    Done(Vec<u32>, Vec<TokenLogprob>, bool),
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "produced by the scheduling loop (card 588a)")
    )]
    Failed(AccountedGenerationFailure),
}

impl GenEvent {
    /// The only production constructor of a failure event. Building the `AccountedGenerationFailure`
    /// records the producer-side terminal error first, even if the receiver is gone.
    #[cfg(test)]
    pub(crate) fn failed(metrics: &Metrics, message: impl Into<String>) -> Self {
        metrics.record_terminal_failure();
        GenEvent::Failed(AccountedGenerationFailure {
            message: message.into(),
        })
    }

    /// The `GenEvent`-typed wrapper over [`deliver_terminal`]: build the `Done`, publish its outcome
    /// count, then deliver it, so a consumer that reads `Done` and then scrapes `/metrics` never sees
    /// a stale count. Every commit path calls this.
    #[cfg(test)]
    pub(crate) fn deliver_done(
        metrics: &Metrics,
        reply: &Sender<GenEvent>,
        tokens: Vec<u32>,
        logprobs: Vec<TokenLogprob>,
        natural: bool,
    ) -> bool {
        deliver_terminal(metrics, reply, GenEvent::Done(tokens, logprobs, natural))
    }
}

/// Publish a terminal `Done`'s outcome count, then deliver it: the order [`GenEvent::failed`] uses for
/// failures, so a consumer that reads the event and then scrapes `/metrics` never sees a stale count.
/// Card 051's rule: delivered -> `requests_completed`, undelivered (the reply receiver is already gone)
/// -> `requests_cancelled`. Delivery is read from the channel first because the count has to exist
/// before the event can be observed; the one outcome that read can miss - a receiver dropped between it
/// and the send - cannot be retracted from a monotonic counter, so it stays a completion and is logged.
/// Returns whether the event was delivered. Every terminal `Done` send in the server goes through this,
/// so the ordering and card 051's classification exist once instead of per call site.
#[cfg(test)]
pub(crate) fn deliver_terminal<T>(metrics: &Metrics, reply: &Sender<T>, event: T) -> bool {
    let predicted = !reply.is_disconnected();
    metrics.record_terminal(predicted);
    #[cfg(test)]
    terminal_delivery_probe::record(metrics);
    let delivered = reply.send(event).is_ok();
    if predicted != delivered {
        tracing::debug!(
            predicted,
            "terminal delivery disagreed with the channel liveness check"
        );
    }
    delivered
}

/// A producer failure whose `requests_errored` increment already happened. The private field and sole
/// constructor on [`GenEvent`] keep uncounted failures out; the distinct type stops the response side
/// counting it again.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AccountedGenerationFailure {
    message: String,
}

impl std::fmt::Display for AccountedGenerationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

/// A generation submission's terminal outcome when it is not [`GenEvent::Done`]: no engine, transport
/// loss, an engine-reported failure, or client cancellation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum GenerationError {
    /// No generation engine is loaded. The request was valid and is refused; nothing was queued.
    #[error(
        "no generation engine is loaded: this server parses and bounds requests but cannot generate \
         until the serving loop lands"
    )]
    NoEngine,
    #[error("engine thread is gone")]
    EngineUnavailable,
    #[error("{0}")]
    EngineFailed(AccountedGenerationFailure),
    #[error("generation reply channel closed before Done")]
    ReplyChannelClosed,
    #[error("generation cancelled by the client")]
    ClientCancelled,
    #[error("generation cancelled because server drain expired")]
    ServerDrainExpired,
}

impl GenerationError {
    /// Producer-side failures and client cancellation are counted by the producer. The two transport
    /// errors have no live producer, so the response side counts them.
    pub(crate) fn needs_response_error_metric(&self) -> bool {
        matches!(
            self,
            GenerationError::EngineUnavailable | GenerationError::ReplyChannelClosed
        )
    }

    /// The HTTP status of the failure: the response status when it is found before any header is
    /// written, else the status carried in an SSE error event after the 200 headers are sent.
    pub(crate) fn stream_status(&self) -> u16 {
        match self {
            GenerationError::NoEngine
            | GenerationError::EngineUnavailable
            | GenerationError::ReplyChannelClosed => 503,
            GenerationError::EngineFailed(_) => 500,
            GenerationError::ClientCancelled => 499,
            GenerationError::ServerDrainExpired => 503,
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            GenerationError::NoEngine => "no_engine",
            GenerationError::EngineUnavailable => "engine_unavailable",
            GenerationError::EngineFailed(_) => "engine_failed",
            GenerationError::ReplyChannelClosed => "reply_channel_closed",
            GenerationError::ClientCancelled => "client_cancelled",
            GenerationError::ServerDrainExpired => "server_drain_expired",
        }
    }
}

/// The model being served, auto-detected from `config.json` at startup (card 057). A causal
/// `Decoder` does decoder-pooling embeddings/rerank and, once an engine is loaded, generation; a BERT
/// `Encoder` does embeddings and bi-encoder rerank; a `CrossEncoder` does cross-encoder rerank only.
/// Exactly one is loaded (the checkpoints have disjoint heads). Endpoints a model cannot serve
/// return 400.
pub(crate) enum Backend {
    Decoder(Arc<Runner>),
    Encoder(Arc<EncoderRunner>),
    CrossEncoder(Arc<CrossEncoderRunner>),
}

/// Test-only record of what `/metrics` would report at the instant [`deliver_terminal`] publishes a
/// terminal outcome and sends its event. A consumer that reads the event and then scrapes
/// `/metrics` sees exactly this snapshot, so a test can assert the count was already published at
/// delivery instead of racing the delivering thread's next instruction; a path that bypasses
/// `deliver_terminal` records nothing and the test's `expect` fails. Snapshots are tagged with the
/// recording thread, so a test reads its own producer's record even when other tests record into
/// the same process. Like `startup::select_serve_route_probe`, it compiles out of production builds.
#[cfg(test)]
pub(crate) mod terminal_delivery_probe {
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::thread::ThreadId;

    use crate::metrics::Metrics;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Snapshot {
        pub(crate) completed: u64,
        pub(crate) cancelled: u64,
        pub(crate) errored: u64,
    }

    static RECORD: Mutex<Vec<(ThreadId, Snapshot)>> = Mutex::new(Vec::new());

    /// Record the outcome counters as they stand right now, on the calling thread.
    pub(crate) fn record(metrics: &Metrics) {
        let snapshot = Snapshot {
            completed: metrics.requests_completed.load(Ordering::Relaxed),
            cancelled: metrics.requests_cancelled.load(Ordering::Relaxed),
            errored: metrics.requests_errored.load(Ordering::Relaxed),
        };
        RECORD
            .lock()
            .expect("terminal delivery probe lock")
            .push((std::thread::current().id(), snapshot));
    }

    /// The most recent snapshot `thread` recorded at a `Done` delivery.
    pub(crate) fn at_delivery(thread: ThreadId) -> Option<Snapshot> {
        RECORD
            .lock()
            .expect("terminal delivery probe lock")
            .iter()
            .rev()
            .find(|(recorded, _)| *recorded == thread)
            .map(|(_, snapshot)| *snapshot)
    }

    /// Drop every snapshot `thread` recorded, so its next delivery must record one: a path that
    /// bypasses `deliver_done` then yields `None` from [`at_delivery`] instead of a stale pass.
    pub(crate) fn reset(thread: ThreadId) {
        RECORD
            .lock()
            .expect("terminal delivery probe lock")
            .retain(|(recorded, _)| *recorded != thread);
    }
}
