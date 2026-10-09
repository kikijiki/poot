//! Shared job, event, and engine-view types used across the server modules.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use poot_llm::Sampler;
use poot_llm::TokenLogprob;
use poot_llm::driver::ModelHandle;
use poot_llm::driver::error::FailureClass;
use poot_llm::driver::{AdapterRef, DriverCaps, Encoded, PromptEmbeds};
use poot_llm::encoders::Pooling;

use crate::metrics::Metrics;
use crate::vision::ImageClient;

/// Global shutdown flag: set by the SIGINT/SIGTERM handler so the accept loop exits cleanly.
pub(crate) static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// The prompt of a generation job, tokenized on the connection thread.
#[derive(Clone)]
pub(crate) enum PromptInput {
    /// A text prompt as token ids.
    Tokens(Vec<u32>),
    /// A multimodal prompt: the token ids, and the embedding rows that stand in for the tokens at their
    /// positions (an image's rows at its placeholders). The sequence is opened with the rows, and no
    /// prefix is matched or registered for it.
    Multimodal {
        tokens: Vec<u32>,
        embeds: PromptEmbeds,
    },
}

impl PromptInput {
    pub(crate) fn tokens(&self) -> &[u32] {
        match self {
            PromptInput::Tokens(tokens) | PromptInput::Multimodal { tokens, .. } => tokens,
        }
    }
}

/// A generation job submitted to the engine: the spec plus a reply channel the engine streams events back on.
pub(crate) struct Job {
    pub(crate) prompt: PromptInput,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    /// stop sequences: the loop ends a request as soon as its generated text contains one, and the text
    /// it returns stops before the match.
    pub(crate) stop: Vec<String>,
    /// stream token pieces as generated (SSE) vs only the final completion.
    pub(crate) stream: bool,
    pub(crate) reply: Sender<GenEvent>,
    /// Enqueue time, for the queue-wait histogram (`now - submitted` at admit).
    pub(crate) submitted: Instant,
    /// This request's adapter selection, resolved by the HTTP layer before enqueueing: the base weights
    /// or a captured adapter set's row. [`AdapterRef::Base`] means "no adapter".
    pub(crate) lora_adapter: AdapterRef,
    /// Set by the submitting connection when it stops waiting (its [`crate::lifecycle::JobGuard`] drops):
    /// the client disconnected or the drain expired. The engine releases the request at its next step
    /// boundary, streaming or not.
    pub(crate) cancel: Arc<AtomicBool>,
}

/// One token row of an encode job: its ids and each id's segment (empty for all zero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EncodeRow {
    pub(crate) ids: Vec<u32>,
    pub(crate) types: Vec<u32>,
}

/// An encode job: token rows run through one `Encode` step on the engine thread (embeddings, rerank).
pub(crate) struct EncodeJob {
    pub(crate) rows: Vec<EncodeRow>,
    pub(crate) reply: Sender<EncodeReply>,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// What an encode job answers: the hidden states or scores, or the failure of the step.
pub(crate) type EncodeReply = Result<Encoded, AccountedGenerationFailure>;

/// The LoRA administration operations the engine thread carries out on the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdminRequest {
    List,
    Load { name: String, path: String },
    Unload { name: String },
}

/// One loaded adapter as `GET /v1/lora_adapters` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdapterInfo {
    pub(crate) name: String,
    pub(crate) index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdminReply {
    Adapters(Vec<AdapterInfo>),
    Loaded(AdapterInfo),
    Unloaded(AdapterInfo),
}

/// Why an administration operation did not happen. Nothing was applied in any variant.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AdminError {
    /// The driver or the request refused it: `class` decides the status.
    #[error("{reason}")]
    Refused { class: FailureClass, reason: String },
    #[error("lora adapter {0:?} is not registered")]
    NotFound(String),
    /// The adapter is held by running requests.
    #[error("{0}")]
    Busy(String),
    #[error("engine thread is gone")]
    EngineUnavailable,
}

impl AdminError {
    pub(crate) fn status(&self) -> u16 {
        match self {
            AdminError::Refused { class, .. } => class_status(*class),
            AdminError::NotFound(_) => 404,
            AdminError::Busy(_) => 409,
            AdminError::EngineUnavailable => 503,
        }
    }
}

/// An administration job: the operation and where its outcome goes.
pub(crate) struct AdminJob {
    pub(crate) request: AdminRequest,
    pub(crate) reply: Sender<Result<AdminReply, AdminError>>,
}

/// What a connection thread sends the engine thread. Every route that runs the model sends one of
/// these: the driver is `&mut`, not `Sync`, so the engine thread is the only place it is used.
pub(crate) enum EngineMsg {
    Generate(Box<Job>),
    Encode(EncodeJob),
    Admin(AdminJob),
}

/// The HTTP status a failure class is answered with: a request the driver refuses is the client's (400),
/// one this prepared engine cannot serve is the server's to refuse (503), a fault is a 500.
pub(crate) fn class_status(class: FailureClass) -> u16 {
    match class {
        FailureClass::Request => 400,
        FailureClass::Unsupported => 503,
        FailureClass::Step | FailureClass::Fault => 500,
    }
}

/// What the engine thread publishes for the connection threads to read: the caps the driver serves
/// now (an adapter load or a first encode changes them) and the loaded adapters' names.
#[derive(Clone, Debug)]
pub(crate) struct EngineView(Arc<RwLock<ViewState>>);

#[derive(Debug)]
struct ViewState {
    caps: DriverCaps,
    adapters: BTreeMap<String, AdapterRef>,
}

impl EngineView {
    pub(crate) fn new(caps: DriverCaps) -> Self {
        Self(Arc::new(RwLock::new(ViewState {
            caps,
            adapters: BTreeMap::new(),
        })))
    }

    pub(crate) fn caps(&self) -> DriverCaps {
        self.0.read().expect("engine view lock").caps.clone()
    }

    /// The adapter `name` selects, if it is loaded.
    pub(crate) fn adapter(&self, name: &str) -> Option<AdapterRef> {
        self.0
            .read()
            .expect("engine view lock")
            .adapters
            .get(name)
            .copied()
    }

    pub(crate) fn publish(&self, caps: DriverCaps, adapters: BTreeMap<String, AdapterRef>) {
        let mut state = self.0.write().expect("engine view lock");
        state.caps = caps;
        state.adapters = adapters;
    }
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

/// Why a generation ended, as the API reports it: `Eos` and `Stop` are OpenAI `"stop"`, `Length`
/// (`max_tokens`, or the positions the engine holds) is `"length"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinishReason {
    /// The model sampled one of its end-of-sequence tokens.
    Eos,
    /// A stop string appeared in the generated text.
    Stop,
    /// `max_new` tokens were generated, or the positions ran out.
    Length,
}

impl FinishReason {
    pub(crate) fn api_name(self) -> &'static str {
        match self {
            FinishReason::Eos | FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// A finished generation: what the request produced and why it ended.
#[derive(Debug)]
pub(crate) struct Completion {
    /// Tokens in the prompt, the shared prefix included.
    pub(crate) prompt_tokens: usize,
    /// The generated tokens, the one that completed a stop string among them and an end-of-sequence
    /// token not.
    pub(crate) tokens: Vec<u32>,
    /// The generated text, cut before a stop string.
    pub(crate) text: String,
    /// One record per generated token when the request's sampler records logprobs.
    pub(crate) logprobs: Vec<TokenLogprob>,
    pub(crate) finish: FinishReason,
}

/// What a connection thread holds of the engine: the loaded model for the request-side text services
/// (tokenization, token counts, chat rendering, guided constraints), the queue every route that runs the
/// model sends through, and the view the engine thread publishes.
pub(crate) struct Served {
    pub(crate) model: Arc<ModelHandle>,
    pub(crate) engine: Sender<EngineMsg>,
    pub(crate) view: EngineView,
    /// How a sentence vector is pooled from hidden states, as the checkpoint declares.
    pub(crate) pooling: Pooling,
    /// The image front end of a vision-language checkpoint; `None` serves text only.
    pub(crate) images: Option<ImageClient>,
}

/// What the engine sends back per job: token pieces (streaming), then a terminal `Done` or `Failed`.
pub(crate) enum GenEvent {
    /// A generated token piece plus engine-produced per-token timing/state data. A piece the stop
    /// scanner holds back, or that completes a stop string, is not sent.
    Token(String, PerTokenInfo),
    Done(Completion),
    Failed(AccountedGenerationFailure),
}

impl GenEvent {
    /// The only production constructor of a failure event. Building the `AccountedGenerationFailure`
    /// records the producer-side terminal error first, even if the receiver is gone.
    pub(crate) fn failed(
        metrics: &Metrics,
        class: FailureClass,
        message: impl Into<String>,
    ) -> Self {
        GenEvent::Failed(AccountedGenerationFailure::new(metrics, class, message))
    }

    /// The `GenEvent`-typed wrapper over [`deliver_terminal`]: build the `Done`, publish its outcome
    /// count, then deliver it, so a consumer that reads `Done` and then scrapes `/metrics` never sees
    /// a stale count. Every commit path calls this.
    pub(crate) fn deliver_done(
        metrics: &Metrics,
        reply: &Sender<GenEvent>,
        completion: Completion,
    ) -> bool {
        deliver_terminal(metrics, reply, GenEvent::Done(completion))
    }
}

/// Publish a terminal `Done`'s outcome count, then deliver it: the order [`GenEvent::failed`] uses for
/// failures, so a consumer that reads the event and then scrapes `/metrics` never sees a stale count.
/// Card 051's rule: delivered -> `requests_completed`, undelivered (the reply receiver is already gone)
/// -> `requests_cancelled`. Delivery is read from the channel first because the count has to exist
/// before the event can be observed; the one outcome that read can miss - a receiver dropped between it
/// and the send - cannot be retracted from a monotonic counter, so it stays a completion and is logged.
/// Returns whether the event was delivered. Every terminal success send in the server goes through this,
/// so the ordering and card 051's classification exist once instead of per call site.
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

/// A producer failure whose `requests_errored` increment already happened. The private fields and sole
/// constructor keep uncounted failures out; the distinct type stops the response side counting it again.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AccountedGenerationFailure {
    class: FailureClass,
    message: String,
}

impl AccountedGenerationFailure {
    /// Count the failure, then build it: the count exists before the failure can be observed.
    pub(crate) fn new(metrics: &Metrics, class: FailureClass, message: impl Into<String>) -> Self {
        metrics.record_terminal_failure();
        Self {
            class,
            message: message.into(),
        }
    }

    pub(crate) fn class(&self) -> FailureClass {
        self.class
    }
}

impl std::fmt::Display for AccountedGenerationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

/// A generation submission's terminal outcome when it is not [`GenEvent::Done`]: transport loss, an
/// engine-reported failure, or client cancellation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum GenerationError {
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
    /// written, else the status carried in an SSE error event after the 200 headers are sent. An
    /// engine failure is answered by its class: a refusal is the client's 400 or the server's 503, a
    /// fault is a 500.
    pub(crate) fn stream_status(&self) -> u16 {
        match self {
            GenerationError::EngineUnavailable
            | GenerationError::ReplyChannelClosed
            | GenerationError::ServerDrainExpired => 503,
            GenerationError::EngineFailed(failure) => class_status(failure.class()),
            GenerationError::ClientCancelled => 499,
        }
    }

    /// The OpenAI error `type` of [`Self::stream_status`].
    pub(crate) fn error_type(&self) -> &'static str {
        if self.stream_status() < 500 {
            "invalid_request_error"
        } else {
            "server_error"
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            GenerationError::EngineUnavailable => "engine_unavailable",
            GenerationError::EngineFailed(_) => "engine_failed",
            GenerationError::ReplyChannelClosed => "reply_channel_closed",
            GenerationError::ClientCancelled => "client_cancelled",
            GenerationError::ServerDrainExpired => "server_drain_expired",
        }
    }
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
