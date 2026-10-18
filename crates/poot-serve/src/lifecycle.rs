//! Bounded ownership for server-created threads and accepted response sockets.

use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use poot_llm::driver::{AdapterRef, Encoded};
use poot_llm::{GenerationControl, Sampler};

use crate::metrics::Metrics;
use crate::types::{
    AdminError, AdminJob, AdminReply, AdminRequest, ChoiceReply, Completion, EncodeJob,
    EncodeReply, EncodeRow, EngineMsg, GenEvent, GenerationError, Job, PerTokenInfo, PromptInput,
};

const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(10);
pub(crate) const CANCELLATION_SETTLE_TIMEOUT: Duration = Duration::from_secs(1);
pub(crate) const MAX_DRAIN_TIMEOUT_SECS: u64 = 24 * 60 * 60;

static DRAIN_CANCELLATION: AtomicBool = AtomicBool::new(false);

/// A positive graceful-drain duration capped at one day, so an `Instant` deadline is never built from an unchecked value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ServerDrainTimeout(Duration);

/// A `--drain-timeout-secs` value outside the accepted range.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum DrainTimeoutError {
    #[error("must be greater than zero, got 0")]
    Zero,
    #[error("must be at most {MAX_DRAIN_TIMEOUT_SECS} seconds (one day), got {seconds}")]
    TooLong { seconds: u64 },
}

impl ServerDrainTimeout {
    /// The drain duration of `seconds`, which must be positive and at most one day.
    pub(crate) fn from_secs(seconds: u64) -> Result<Self, DrainTimeoutError> {
        match seconds {
            0 => Err(DrainTimeoutError::Zero),
            seconds if seconds > MAX_DRAIN_TIMEOUT_SECS => {
                Err(DrainTimeoutError::TooLong { seconds })
            }
            seconds => Ok(Self(Duration::from_secs(seconds))),
        }
    }

    pub(crate) fn duration(self) -> Duration {
        self.0
    }

    pub(crate) fn deadline_from(self, start: Instant) -> Instant {
        deadline_after(start, self.0)
    }
}

/// Return a deadline without panicking; an unrepresentable deadline expires immediately.
pub(crate) fn deadline_after(start: Instant, duration: Duration) -> Instant {
    start.checked_add(duration).unwrap_or(start)
}

pub(crate) fn reset_drain_cancellation() {
    DRAIN_CANCELLATION.store(false, Ordering::SeqCst);
}

pub(crate) fn drain_cancellation_requested() -> bool {
    DRAIN_CANCELLATION.load(Ordering::SeqCst)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerKind {
    /// A thread the server itself starts, such as the engine thread.
    Server,
    Connection,
}

struct OwnedWorker {
    id: u64,
    name: String,
    kind: WorkerKind,
    response_socket: Option<TcpStream>,
    join: Option<JoinHandle<()>>,
}

struct JoinRequest {
    id: u64,
    join: JoinHandle<()>,
}

struct JoinedWorker {
    id: u64,
    panicked: bool,
}

struct CompletionGuard {
    completed: Sender<()>,
}

impl Drop for CompletionGuard {
    fn drop(&mut self) {
        let _ = self.completed.send(());
    }
}

/// Ownership table for all top-level server workers. A completion guard wakes the owner even if a
/// worker unwinds; the notice is only a hint, so `is_finished` is checked before join dispatch. Joins run
/// on a dedicated reaper thread, never on the deadline-owning controller.
pub(crate) struct WorkerRegistry {
    next_id: u64,
    completed_tx: Sender<()>,
    completed_rx: Receiver<()>,
    join_tx: Sender<JoinRequest>,
    joined_rx: Receiver<JoinedWorker>,
    _join_reaper: JoinHandle<()>,
    workers: Vec<OwnedWorker>,
}

impl WorkerRegistry {
    pub(crate) fn new() -> Self {
        let (completed_tx, completed_rx) = mpsc::channel();
        let (join_tx, join_rx) = mpsc::channel::<JoinRequest>();
        let (joined_tx, joined_rx) = mpsc::channel();
        let join_reaper = thread::Builder::new()
            .name("poot-worker-join-reaper".to_string())
            .spawn(move || {
                for request in join_rx {
                    let result = JoinedWorker {
                        id: request.id,
                        panicked: request.join.join().is_err(),
                    };
                    if joined_tx.send(result).is_err() {
                        return;
                    }
                }
            })
            .expect("spawn server worker join reaper");
        Self {
            next_id: 0,
            completed_tx,
            completed_rx,
            join_tx,
            joined_rx,
            _join_reaper: join_reaper,
            workers: Vec::new(),
        }
    }

    pub(crate) fn spawn(
        &mut self,
        name: impl Into<String>,
        work: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<()> {
        self.spawn_owned(name.into(), WorkerKind::Server, None, work)
    }

    pub(crate) fn spawn_connection(
        &mut self,
        name: impl Into<String>,
        stream: TcpStream,
        work: impl FnOnce(TcpStream) + Send + 'static,
    ) -> std::io::Result<()> {
        let response_socket = stream.try_clone()?;
        self.spawn_owned(
            name.into(),
            WorkerKind::Connection,
            Some(response_socket),
            move || work(stream),
        )
    }

    fn spawn_owned(
        &mut self,
        name: String,
        kind: WorkerKind,
        response_socket: Option<TcpStream>,
        work: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<()> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("server worker id space exhausted"))?;
        let completed = self.completed_tx.clone();
        let guard = CompletionGuard { completed };
        let join = thread::Builder::new().name(name.clone()).spawn(move || {
            let _completion = guard;
            work();
        })?;
        self.workers.push(OwnedWorker {
            id,
            name,
            kind,
            response_socket,
            join: Some(join),
        });
        Ok(())
    }

    pub(crate) fn reap_ready(&mut self) {
        while self.completed_rx.try_recv().is_ok() {}
        self.dispatch_finished_joins();
        while let Ok(joined) = self.joined_rx.try_recv() {
            let Some(index) = self
                .workers
                .iter()
                .position(|worker| worker.id == joined.id)
            else {
                continue;
            };
            let worker = self.workers.swap_remove(index);
            if joined.panicked {
                tracing::error!(worker = worker.name, "server worker panicked");
            }
        }
    }

    /// A handle can report finished while thread-local destructors still run, so even a guarded join
    /// can block; it therefore runs on the reaper, not the shutdown controller.
    fn dispatch_finished_joins(&mut self) {
        for worker in &mut self.workers {
            let Some(join) = worker.join.as_ref() else {
                continue;
            };
            if !join.is_finished() {
                continue;
            }
            let request = JoinRequest {
                id: worker.id,
                join: worker.join.take().expect("checked as present"),
            };
            if let Err(error) = self.join_tx.send(request) {
                worker.join = Some(error.0.join);
                return;
            }
        }
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.workers
            .iter()
            .filter(|worker| worker.kind == WorkerKind::Connection)
            .count()
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub(crate) fn wait_for_connections_until(&mut self, deadline: Instant) -> bool {
        self.wait_until(deadline, |registry| registry.connection_count() == 0)
    }

    pub(crate) fn wait_for_all_until(&mut self, deadline: Instant) -> bool {
        self.wait_until(deadline, |registry| registry.workers.is_empty())
    }

    fn wait_until(&mut self, deadline: Instant, done: impl Fn(&Self) -> bool) -> bool {
        loop {
            self.reap_ready();
            if done(self) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(WORKER_POLL_INTERVAL);
            match self.completed_rx.recv_timeout(wait) {
                Ok(()) => self.reap_ready(),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return done(self),
            }
        }
    }

    /// End the graceful phase. The cancellation flag is set before the sockets close so handlers blocked
    /// on an engine reply and handlers blocked in socket I/O both make progress.
    pub(crate) fn cancel_connections(&self) -> usize {
        DRAIN_CANCELLATION.store(true, Ordering::SeqCst);
        let mut closed = 0;
        for worker in &self.workers {
            if let Some(socket) = &worker.response_socket {
                let _ = socket.shutdown(Shutdown::Both);
                closed += 1;
            }
        }
        closed
    }
}

/// Whether the peer of `stream` has closed its end: an end-of-file read that consumes nothing. The
/// request has been read in full by now, so a readable socket with no byte is a client that hung up (or
/// half-closed, which an HTTP client does not do while it waits for a response).
pub(crate) fn peer_closed(stream: &TcpStream) -> bool {
    let mut byte = 0u8;
    // SAFETY: `recv` writes at most one byte into `byte`, which outlives the call; the descriptor is
    // owned by `stream`, and MSG_DONTWAIT keeps the shared socket's blocking mode untouched.
    let read = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            (&raw mut byte).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    read == 0
}

/// Sets a job's `cancel` flag when the connection stops waiting for it, however it stops: a result, an
/// error, a client that hung up or a write that failed. The engine releases the request at its next step
/// boundary.
pub(crate) struct JobGuard(Arc<AtomicBool>);

impl JobGuard {
    pub(crate) fn new() -> (Self, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        (Self(Arc::clone(&cancel)), cancel)
    }
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Whether the client that waits for a job has gone: polled while the job waits.
pub(crate) type ClientGone<'a> = &'a dyn Fn() -> bool;

/// What one generation job asks of the engine: the request-to-job conversion every route shares.
pub(crate) struct GenerationRequest {
    pub(crate) prompt: PromptInput,
    pub(crate) max_new: usize,
    pub(crate) sampler: Sampler,
    pub(crate) stop: Vec<String>,
    /// Stream the pieces as they are generated (SSE) instead of only the final completion.
    pub(crate) stream: bool,
    /// The request-owned selection from `resolve_lora_adapter`.
    pub(crate) lora_adapter: AdapterRef,
}

/// The next reply of a job, or why none will come: the drain expired, the client hung up (`client_gone`
/// is polled while the job waits, so a client that hangs up on a request that streams nothing is
/// noticed too), or the engine dropped the reply channel.
fn next_reply<T>(
    rx: &mpsc::Receiver<T>,
    client_gone: ClientGone<'_>,
) -> std::result::Result<T, GenerationError> {
    loop {
        if drain_cancellation_requested() {
            return Err(GenerationError::ServerDrainExpired);
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(reply) => return Ok(reply),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if client_gone() {
                    return Err(GenerationError::ClientCancelled);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(GenerationError::ReplyChannelClosed);
            }
        }
    }
}

/// How long the wait for a request's replies parks on their channel; also how often the wait asks whether
/// the client has hung up.
const REPLY_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// One choice of a generation request while it runs on the engine. Dropping it cancels the job.
struct ChoiceInFlight {
    started: Instant,
    ttft_ms: Option<f64>,
    completion: Option<Completion>,
    _guard: JobGuard,
}

impl ChoiceInFlight {
    /// Take one event of choice `index`: a piece goes to `on_token`, a terminal event ends the choice.
    fn accept(
        &mut self,
        index: usize,
        event: GenEvent,
        metrics: &Metrics,
        on_token: &mut impl FnMut(usize, &str, PerTokenInfo) -> GenerationControl,
    ) -> std::result::Result<(), GenerationError> {
        match event {
            GenEvent::Token(piece, info) => {
                self.ttft_ms
                    .get_or_insert_with(|| self.started.elapsed().as_secs_f64() * 1000.0);
                if on_token(index, &piece, info).is_break() {
                    return Err(GenerationError::ClientCancelled);
                }
            }
            GenEvent::Done(completion) => {
                metrics.record_request(self.started.elapsed().as_secs_f64() * 1000.0, self.ttft_ms);
                self.completion = Some(completion);
            }
            GenEvent::Failed(failure) => return Err(GenerationError::EngineFailed(failure)),
        }
        Ok(())
    }
}

/// Submit one generation job per request (the `n` choices of one HTTP request) and block until every
/// one finishes, returning their [`Completion`]s in request order. The choices share one reply channel,
/// so this thread waits on a single receiver however many there are. `on_token` receives each streamed
/// piece with its choice index and [`PerTokenInfo`]; non-streaming callers return
/// `GenerationControl::Continue(())`. `Break(())` (e.g. after a client socket write fails) makes `submit`
/// return early, as does the first choice that fails. However `submit` returns, every job's [`JobGuard`]
/// drops and the engine releases the requests within one step. A job the engine has not opened within
/// `max_queue_wait` is failed by the engine as unavailable.
///
/// A finished choice is recorded in the request metrics here, once, whatever route it came from.
pub(crate) fn submit(
    tx: &Sender<EngineMsg>,
    metrics: &Metrics,
    max_queue_wait: Duration,
    requests: Vec<GenerationRequest>,
    client_gone: ClientGone<'_>,
    mut on_token: impl FnMut(usize, &str, PerTokenInfo) -> GenerationControl,
) -> std::result::Result<Vec<Completion>, GenerationError> {
    let (replies, events) = ChoiceReply::channel(requests.len());
    let mut choices = Vec::with_capacity(requests.len());
    for (request, reply) in requests.into_iter().zip(replies) {
        let GenerationRequest {
            prompt,
            max_new,
            sampler,
            stop,
            stream,
            lora_adapter,
        } = request;
        let (guard, cancel) = JobGuard::new();
        let started = Instant::now();
        // A choice whose send fails returns here, and the guards of the choices already queued cancel them.
        tx.send(EngineMsg::Generate(Box::new(Job {
            prompt,
            max_new,
            sampler,
            stop,
            stream,
            reply,
            submitted: started,
            start_by: started + max_queue_wait,
            lora_adapter,
            cancel,
        })))
        .map_err(|_| GenerationError::EngineUnavailable)?;
        choices.push(ChoiceInFlight {
            started,
            ttft_ms: None,
            completion: None,
            _guard: guard,
        });
    }
    let mut running = choices.len();
    while running > 0 {
        if drain_cancellation_requested() {
            return Err(GenerationError::ServerDrainExpired);
        }
        match events.recv_timeout(REPLY_POLL_INTERVAL) {
            Ok((index, event)) => {
                let choice = &mut choices[index];
                choice.accept(index, event, metrics, &mut on_token)?;
                if choice.completion.is_some() {
                    running -= 1;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if client_gone() {
                    return Err(GenerationError::ClientCancelled);
                }
            }
            // Every choice holds a sender until the engine finishes with it, so a closed channel with
            // choices still running is an engine that dropped them.
            Err(RecvTimeoutError::Disconnected) => {
                return Err(GenerationError::ReplyChannelClosed);
            }
        }
    }
    Ok(choices
        .into_iter()
        .filter_map(|choice| choice.completion)
        .collect())
}

/// Submit an encode job over token `rows` and block until the engine thread answers. A finished
/// request is recorded in the request metrics here, with the tokens it encoded.
pub(crate) fn submit_encode(
    tx: &Sender<EngineMsg>,
    metrics: &Metrics,
    rows: Vec<EncodeRow>,
    client_gone: ClientGone<'_>,
) -> std::result::Result<Encoded, GenerationError> {
    let tokens: usize = rows.iter().map(|row| row.ids.len()).sum();
    let (reply, rx) = mpsc::channel::<EncodeReply>();
    let (_guard, cancel) = JobGuard::new();
    let start = Instant::now();
    tx.send(EngineMsg::Encode(EncodeJob {
        rows,
        reply,
        cancel,
    }))
    .map_err(|_| GenerationError::EngineUnavailable)?;
    match next_reply(&rx, client_gone)? {
        Ok(encoded) => {
            metrics
                .prompt_tokens
                .fetch_add(tokens as u64, Ordering::Relaxed);
            metrics.record_request(start.elapsed().as_secs_f64() * 1000.0, None);
            Ok(encoded)
        }
        Err(failure) => Err(GenerationError::EngineFailed(failure)),
    }
}

/// Run an administration `request` on the engine thread and block until it answers.
pub(crate) fn submit_admin(
    tx: &Sender<EngineMsg>,
    request: AdminRequest,
) -> std::result::Result<AdminReply, AdminError> {
    let (reply, rx) = mpsc::channel();
    tx.send(EngineMsg::Admin(AdminJob { request, reply }))
        .map_err(|_| AdminError::EngineUnavailable)?;
    match next_reply(&rx, &|| false) {
        Ok(outcome) => outcome,
        Err(_) => Err(AdminError::EngineUnavailable),
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl WorkerRegistry {
        pub(crate) fn notify_completion_for_test(&self) {
            self.completed_tx.send(()).unwrap();
        }

        pub(crate) fn joins_in_flight_for_test(&self) -> usize {
            self.workers
                .iter()
                .filter(|worker| worker.join.is_none())
                .count()
        }
    }
}
