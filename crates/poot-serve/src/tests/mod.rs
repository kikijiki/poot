use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use poot_llm::driver::{
    AdapterRef, DriverCaps, HeadSet, InputSet, LoadOptions, ModelHandle, RowSpeculation,
    StateResidency,
};
use poot_llm::encoders::Pooling;
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::KvLayout;
use poot_models::registry::Registry;

use crate::admin::LoraAdminPolicy;
use crate::api::{
    ChatReq, CompletionReq, GuidedDecodeError, MAX_CHOICES, MAX_LOGPROBS, MAX_NEW_TOKENS_CAP,
    MAX_STOP_SEQUENCES, ResponseFormatConstraint, Stop, StreamOptions, UnsupportedCapabilityError,
    clamp_logprobs, clamp_n, classify_handler_error, default_max, parse_tool_calls,
    reject_unsupported_modality, resolve_forced_tools, response_format_constraint, sampler_of,
    suppress_stream_usage, tool_call_stream_deltas,
};
use crate::batch::commit::{SlotFate, commit_tokens};
use crate::batch::engine::{
    EngineError, Exit, OpenRequest, Opening, Row, RowResult, ServeEngine, Work, run_engine,
    run_engine_thread,
};
use crate::batch::schedule::schedule_prefill_admissions;
use crate::batch::slot::{Slot, reject_job};
use crate::config::TelemetryConfig;
use crate::engine_plan::EngineHealth;
use crate::handlers::{
    account_generation_error, handle_chat, sse_write_probe, stream_chat, stream_completion,
    stream_receipt,
};
use crate::http::test_only::{RequestRead, read_request, read_request_from};
use crate::http::{
    ConnGuard, MAX_BODY_BYTES, MAX_HEADER_BYTES, RateLimiter, body_len_ok,
    configure_response_writes, error_body, handle, model_object, read_request_for_routing, respond,
    respond_error, respond_typed,
};
use crate::lifecycle::{
    CANCELLATION_SETTLE_TIMEOUT, GenerationRequest, WorkerRegistry, deadline_after,
    drain_cancellation_requested, reset_drain_cancellation, submit,
};
use crate::metrics::{CacheMetrics, Histogram, Metrics, SpeculativeMetrics, TimingMetrics};
use crate::types::{
    Completion, EngineMsg, EngineView, FinishReason, GenEvent, GenerationError, Job, PerTokenInfo,
    PromptInput, SHUTDOWN, Served, terminal_delivery_probe,
};
use poot_llm::driver::stops::StopScanner;

mod admin;
mod config;
mod drain;
pub(crate) mod encoder_routes;
mod engine_loop;
mod http_parse;
mod lifecycle;
mod other;
mod recording_engine;
mod scheduling;
mod streaming;
mod telemetry;
mod terminal_done;
mod write_deadline;

use recording_engine::{
    Event, FakeSeq, JobSpec, RecordingEngine, Submitted, outcome, refused, serve, tokens_of,
};

/// What a scripted generation of the test double returns: the generated tokens and text, and why it
/// ended.
type Generated = (Vec<u32>, String, FinishReason);

/// The test double at the engine boundary: drains the queue and runs each generation job through
/// `generate`, streaming `Token` events when the job asked to stream and a terminal `Done`/`Failed`, with
/// the same outcome accounting the batched commit path has (a terminal `Done` through
/// `GenEvent::deliver_done`, a piece that cannot be delivered is a cancellation). It stands in for the
/// serving loop so the request side (`submit`, the handlers, write deadlines) is tested without a model
/// or a device.
fn test_engine_loop<F>(rx: Receiver<EngineMsg>, metrics: Arc<Metrics>, mut generate: F)
where
    F: FnMut(
        &[u32],
        usize,
        &mut Sampler,
        &[String],
        &mut dyn FnMut(&str) -> GenerationControl,
    ) -> Result<Generated>,
{
    for msg in rx {
        let EngineMsg::Generate(job) = msg else {
            panic!("the double serves generation jobs only");
        };
        let Job {
            prompt,
            max_new,
            mut sampler,
            stop,
            stream,
            reply,
            ..
        } = *job;
        let decode_start = Instant::now();
        let mut last_token: Option<Instant> = None;
        let mut token_idx = 0usize;
        let mut cancelled = false;
        let prompt_tokens = prompt.tokens().to_vec();
        let res = generate(
            &prompt_tokens,
            max_new,
            &mut sampler,
            &stop,
            &mut |piece: &str| {
                let now = Instant::now();
                let info = PerTokenInfo {
                    token_idx,
                    since_prev_ms: last_token
                        .map(|t| (now - t).as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    cumulative_ms: (now - decode_start).as_secs_f64() * 1000.0,
                    kv_used_ratio: 0.0,
                };
                last_token = Some(now);
                token_idx += 1;
                if stream
                    && reply
                        .send(GenEvent::Token(piece.to_string(), info))
                        .is_err()
                {
                    cancelled = true;
                    return GenerationControl::Break(());
                }
                GenerationControl::Continue(())
            },
        );
        if cancelled {
            metrics.requests_cancelled.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        match res {
            Ok((tokens, text, finish)) => {
                let completion = Completion {
                    prompt_tokens: prompt_tokens.len(),
                    tokens,
                    text,
                    logprobs: sampler.take_logprobs(),
                    finish,
                };
                GenEvent::deliver_done(&metrics, &reply, completion);
            }
            Err(e) => {
                let class = poot_llm::driver::error::FailureClass::Fault;
                let _ = reply.send(GenEvent::failed(&metrics, class, format!("{e:#}")));
            }
        }
    }
}

/// A caps the fakes publish: `rows` rows of a contiguous-free pool, tokens-only inputs, no adapter.
pub(crate) fn test_caps(rows: usize) -> DriverCaps {
    let rows = std::num::NonZeroUsize::new(rows).expect("at least one row");
    let width = std::num::NonZeroUsize::new(8).expect("nonzero");
    DriverCaps {
        rows,
        row_counts: vec![rows],
        widths: vec![width],
        max_seq: std::num::NonZeroUsize::new(64).expect("nonzero"),
        kv: KvLayout::Contiguous,
        kv_blocks: 0,
        state: StateResidency::KvOnly,
        heads: HeadSet::default(),
        lora: None,
        windows: Vec::new(),
        speculation: None,
        inputs: InputSet {
            tokens: true,
            embeds: false,
            encode: false,
        },
    }
}

/// What a connection thread holds, over `model` and a queue the test reads: the double's engine side.
pub(crate) fn served_over(
    model: Arc<ModelHandle>,
    rows: usize,
) -> (Arc<Served>, Receiver<EngineMsg>) {
    let (engine, queue) = mpsc::channel();
    let served = Served {
        model,
        engine,
        view: EngineView::new(test_caps(rows)),
        pooling: Pooling::Mean,
        images: None,
    };
    (Arc::new(served), queue)
}

/// A recording fake running the production loop on its own thread, behind a [`Served`]: what a request
/// over a socket reaches. The loop's counters and the handlers' are the same `metrics`.
pub(crate) struct EngineThread {
    join: std::thread::JoinHandle<RecordingEngine>,
}

impl EngineThread {
    /// Close the queue (the caller's `served` is the last sender) and take the fake back.
    pub(crate) fn finish(self, served: Arc<Served>) -> RecordingEngine {
        drop(served);
        self.join.join().expect("the engine thread ends cleanly")
    }
}

pub(crate) fn serve_with_fake(
    engine: RecordingEngine,
    model: Arc<ModelHandle>,
    metrics: &Arc<Metrics>,
) -> (Arc<Served>, EngineThread) {
    let (served, queue) = served_over(model, engine.rows);
    let health = EngineHealth::serving_for_test("fake", engine.rows, engine.max_seq);
    let metrics = Arc::clone(metrics);
    let join = std::thread::spawn(move || run_engine_thread(engine, queue, &metrics, &health));
    (served, EngineThread { join })
}

/// The model of the tiny GGUF fixture, loaded and its file removed.
pub(crate) fn tiny_model() -> Arc<ModelHandle> {
    let path = write_tiny_gguf();
    let model = load_handle(&path);
    let _ = std::fs::remove_file(&path);
    model
}

/// A generation request over the tokens of `prompt` (one token per character).
pub(crate) fn generation_request(prompt: &str, max_new: usize, stream: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: PromptInput::Tokens(tokens_of(prompt)),
        max_new,
        sampler: Sampler::greedy(),
        stop: Vec::new(),
        stream,
        lora_adapter: AdapterRef::Base,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationFailureAccountingCase {
    DeliveredDirect,
    DroppedDirect,
    DeliveredRejection,
    DroppedRejection,
    ActiveFailure,
    QueuedShutdown,
    ActiveShutdown,
}

#[derive(Clone, Copy, Debug)]
enum GenerationStreamEndpoint {
    Completion,
    Chat,
}

impl GenerationStreamEndpoint {
    fn name(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Chat => "chat",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GenerationStreamTerminal {
    FailedBeforeToken,
    FailedAfterHeldToken,
    EofBeforeToken,
    EofAfterHeldToken,
    DoneAfterHeldToken,
    Cancelled,
}

fn terminal_token_info() -> PerTokenInfo {
    PerTokenInfo {
        token_idx: 0,
        since_prev_ms: 0.0,
        cumulative_ms: 0.0,
        kv_used_ratio: 0.0,
    }
}

fn run_generation_stream_handler(
    endpoint: GenerationStreamEndpoint,
    terminal: GenerationStreamTerminal,
    runner: Arc<ModelHandle>,
) -> (String, Arc<Metrics>) {
    use std::net::Shutdown;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind terminal handler socket");
    let addr = listener
        .local_addr()
        .expect("terminal handler socket address");
    let mut client = TcpStream::connect(addr).expect("connect terminal handler client");
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("bound terminal handler read");
    let (server, _) = listener.accept().expect("accept terminal handler client");
    server
        .set_write_timeout(Some(Duration::from_secs(5)))
        .expect("bound terminal handler write");
    let write_control = server
        .try_clone()
        .expect("clone terminal handler write control");
    let metrics = Arc::new(test_metrics(endpoint.name()));
    let handler_metrics = Arc::clone(&metrics);
    let (served, job_rx) = served_over(runner, 1);

    let handler = std::thread::spawn(move || {
        let mut server = server;
        match endpoint {
            GenerationStreamEndpoint::Completion => {
                let req: CompletionReq = serde_json::from_str(
                    r#"{"prompt":"a","max_tokens":1,"stream":true,"stop":"ab"}"#,
                )
                .expect("completion terminal request");
                stream_completion(&mut server, &served, &handler_metrics, &req)
            }
            GenerationStreamEndpoint::Chat => {
                let req: ChatReq = serde_json::from_str(
                    r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":1,"stream":true,"stop":"ab"}"#,
                )
                .expect("chat terminal request");
                stream_chat(&mut server, &served, &handler_metrics, &req)
            }
        }
    });

    match terminal {
        GenerationStreamTerminal::FailedBeforeToken
        | GenerationStreamTerminal::FailedAfterHeldToken
        | GenerationStreamTerminal::DoneAfterHeldToken
        | GenerationStreamTerminal::Cancelled => {
            let engine_metrics = Arc::clone(&metrics);
            let (ready_tx, ready_rx) = mpsc::channel();
            let (proceed_tx, proceed_rx) = mpsc::channel();
            let engine = std::thread::spawn(move || {
                let cancel_metrics = Arc::clone(&engine_metrics);
                test_engine_loop(
                    job_rx,
                    engine_metrics,
                    move |_prompt, _max_new, _sampler, _stops, on_token| {
                        ready_tx.send(()).expect("terminal handler job admitted");
                        proceed_rx.recv().expect("terminal handler proceeds");
                        match terminal {
                            GenerationStreamTerminal::FailedBeforeToken => {
                                bail!("terminal handler failure")
                            }
                            GenerationStreamTerminal::FailedAfterHeldToken => {
                                let _ = on_token("a");
                                bail!("terminal handler failure")
                            }
                            GenerationStreamTerminal::DoneAfterHeldToken => {
                                let _ = on_token("a");
                                Ok((vec![0], "a".to_string(), FinishReason::Length))
                            }
                            GenerationStreamTerminal::Cancelled => {
                                let _ = on_token("c");
                                let deadline = Instant::now() + Duration::from_secs(5);
                                while cancel_metrics.inflight.load(Ordering::Relaxed) != 0 {
                                    assert!(
                                        Instant::now() < deadline,
                                        "cancelled handler did not drop its in-flight guard"
                                    );
                                    std::thread::yield_now();
                                }
                                Ok((vec![3], "c".to_string(), FinishReason::Length))
                            }
                            GenerationStreamTerminal::EofBeforeToken
                            | GenerationStreamTerminal::EofAfterHeldToken => {
                                unreachable!("EOF rows own the job directly")
                            }
                        }
                    },
                );
            });
            ready_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("handler submitted its engine job");
            if matches!(terminal, GenerationStreamTerminal::Cancelled) {
                write_control
                    .shutdown(Shutdown::Write)
                    .expect("force handler write cancellation");
            }
            proceed_tx
                .send(())
                .expect("release terminal handler engine");
            engine.join().expect("terminal handler engine");
        }
        GenerationStreamTerminal::EofBeforeToken | GenerationStreamTerminal::EofAfterHeldToken => {
            let EngineMsg::Generate(job) = job_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("handler submitted its EOF fixture job")
            else {
                panic!("a generation request sends a generation job");
            };
            if matches!(terminal, GenerationStreamTerminal::EofAfterHeldToken) {
                job.reply
                    .send(GenEvent::Token("a".into(), terminal_token_info()))
                    .expect("send held terminal token");
            }
            drop(job);
        }
    }

    handler
        .join()
        .expect("terminal handler thread")
        .expect("terminal handler result");
    drop(write_control);
    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .expect("read bounded terminal handler response");
    (response, metrics)
}

pub(crate) fn unique_fixture_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{n}", std::process::id())
}

#[path = "../../tests/support/tiny_gguf.rs"]
mod tiny_gguf;

/// A tiny all-F32 deepseek2-architecture GGUF (two layers, a 4-token vocabulary): a decoder the
/// registry loads as a `ModelHandle`, the way the server loads one.
pub(crate) fn write_tiny_gguf() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "poot_serve_deepseek2_fixture_{}.gguf",
        unique_fixture_suffix()
    ));
    tiny_gguf::write_tiny_deepseek2_gguf(&path, 64);
    path
}

/// The model at `path` (an HF directory or a GGUF) through the family registry, as the server loads it.
pub(crate) fn load_handle(path: &std::path::Path) -> Arc<ModelHandle> {
    Arc::new(
        ModelHandle::load(
            path,
            &Registry::builtin().expect("shipped families register disjoint keys"),
            &LoadOptions::default(),
        )
        .expect("load the model"),
    )
}

/// A minimal `Metrics` for the card 201 tests: only `model` varies; every counter/histogram starts at zero. Mirrors the inline construction in `prometheus_exposition_is_well_formed`.
pub(crate) fn test_metrics(model: &str) -> Metrics {
    Metrics {
        started: Instant::now(),
        requests: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        prompt_tokens: AtomicU64::new(0),
        requests_completed: AtomicU64::new(0),
        requests_cancelled: AtomicU64::new(0),
        requests_errored: AtomicU64::new(0),
        inflight: AtomicU64::new(0),
        ttft: Histogram::new(),
        itl: Histogram::new(),
        e2e: Histogram::new(),
        queue_wait: Histogram::new(),
        batch_size: AtomicU64::new(0),
        slot_capacity: 1,
        backend: "spirv-vulkan".to_string(),
        model: model.into(),
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics::default(),
        speculative: SpeculativeMetrics::default(),
        timing: TimingMetrics::new(TelemetryConfig::CountersOnly),
    }
}

// ---- card 315: LoRA administration trust boundary ----

#[derive(Clone, Copy, Debug)]
enum AdminOperation {
    List,
    Load,
    Unload,
}

impl AdminOperation {
    /// The status the recording fake answers an authorized request with: nothing is loaded, so
    /// removing an adapter is a 404.
    fn allowed_status(self) -> u16 {
        match self {
            Self::List | Self::Load => 200,
            Self::Unload => 404,
        }
    }

    /// What its body holds.
    fn allowed_body_marker(self) -> &'static str {
        match self {
            Self::List => "\"object\":\"list\"",
            Self::Load => "\"name\":\"loaded\"",
            Self::Unload => "is not registered",
        }
    }

    fn request(self, path_marker: &str) -> (&'static str, &'static str, String) {
        match self {
            Self::List => ("GET", "/v1/lora_adapters", String::new()),
            Self::Load => (
                "POST",
                "/v1/lora_adapters",
                serde_json::json!({ "name": "loaded", "path": path_marker }).to_string(),
            ),
            Self::Unload => ("DELETE", "/v1/lora_adapters/base", String::new()),
        }
    }
}

fn run_lora_admin_socket(
    policy: &LoraAdminPolicy,
    served: &Served,
    method: &str,
    path: &str,
    authorization: &[&str],
    body: &str,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind admin trust-boundary socket");
    let addr = listener.local_addr().expect("admin socket address");
    let mut client = TcpStream::connect(addr).expect("connect admin trust-boundary client");
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("bound admin response read");
    let (mut server, _) = listener
        .accept()
        .expect("accept admin trust-boundary client");

    let authorization_headers = authorization
        .iter()
        .map(|value| format!("Authorization: {value}\r\n"))
        .collect::<String>();
    write!(
        client,
        "{method} {path} HTTP/1.1\r\nContent-Length: {}\r\n{authorization_headers}\r\n{body}",
        body.len()
    )
    .expect("write admin request");
    client.flush().expect("flush admin request");

    assert!(
        read_request_for_routing(&mut server, served, policy)
            .expect("stage and handle admin request")
            .is_none(),
        "an admin target must be consumed before ordinary routing"
    );
    drop(server);

    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .expect("read admin response");
    response
}

/// Send a complete protected request head with a nonzero declared body, but deliberately withhold every
/// body byte. A denial can only arrive if production authorizes before body reading.
fn run_denied_lora_admin_head_without_body(
    policy: LoraAdminPolicy,
    served: Arc<Served>,
    method: &str,
    path: &str,
    authorization: &[&str],
    content_length: usize,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind head-first admin socket");
    let addr = listener.local_addr().expect("head-first socket address");
    let mut client = TcpStream::connect(addr).expect("connect head-first admin client");
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("bound immediate denial read");
    let (mut server, _) = listener.accept().expect("accept head-first admin client");
    server
        .set_read_timeout(Some(Duration::from_millis(500)))
        .expect("make an accidental body read fail promptly");

    let handler =
        std::thread::spawn(move || read_request_for_routing(&mut server, &served, &policy));

    let authorization_headers = authorization
        .iter()
        .map(|value| format!("Authorization: {value}\r\n"))
        .collect::<String>();
    write!(
        client,
        "{method} {path} HTTP/1.1\r\nContent-Length: {content_length}\r\n{authorization_headers}\r\n"
    )
    .expect("write only unauthorized admin request head");
    client.flush().expect("flush unauthorized admin head");

    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .expect("denial must arrive while the declared body is withheld");
    assert!(
        handler
            .join()
            .expect("join head-first handler")
            .expect("head-first request handling")
            .is_none(),
        "a denied admin request is fully handled"
    );
    response
}

#[cfg(unix)]
fn set_write_deadline_socket_buffer(stream: &TcpStream, option: libc::c_int) {
    use std::os::fd::AsRawFd;

    let bytes: libc::c_int = 4 * 1024;
    // SAFETY: `stream` owns a live socket fd, and `bytes` is passed with its exact size for the integer
    // SO_SNDBUF/SO_RCVBUF contract. This changes only the local test socket's kernel buffer capacity.
    let result = unsafe {
        libc::setsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            option,
            (&bytes as *const libc::c_int).cast(),
            std::mem::size_of_val(&bytes) as libc::socklen_t,
        )
    };
    assert_eq!(
        result,
        0,
        "set small socket buffer: {}",
        std::io::Error::last_os_error()
    );
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteDeadlineClientCase {
    NeverReads,
    HealthyStream,
    DisconnectedBeforeEvent,
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteDeadlineEngineOutcome {
    Cancelled,
    Completed,
}

/// Drives the production timeout configurator, `ConnGuard` and `submit` over a real loopback socket. The mock engine sends through the same unbounded reply channel as the GPU engine: reaching `producer_progress_rx` shows it did not wait for the connection thread's blocked socket write, and a later send failure shows the timed-out handler dropped its response subscription.
#[cfg(unix)]
fn run_write_deadline_stream_case(case: WriteDeadlineClientCase) {
    let write_timeout = Duration::from_millis(150);
    let completion_budget = Duration::from_secs(3);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind write-deadline fixture");
    let addr = listener.local_addr().expect("write-deadline address");
    let mut client = Some(TcpStream::connect(addr).expect("connect write-deadline client"));
    set_write_deadline_socket_buffer(client.as_ref().unwrap(), libc::SO_RCVBUF);
    client
        .as_ref()
        .unwrap()
        .set_read_timeout(Some(completion_budget))
        .expect("bound fixture client read");
    let (server, _) = listener.accept().expect("accept write-deadline client");
    set_write_deadline_socket_buffer(&server, libc::SO_SNDBUF);
    configure_response_writes(&server, write_timeout)
        .expect("configure production response timeout");

    let live_connections = Arc::new(AtomicUsize::new(1));
    let handler_connections = Arc::clone(&live_connections);
    let (job_tx, job_rx) = mpsc::channel::<EngineMsg>();
    let (headers_tx, headers_rx) = mpsc::channel();
    let metrics = Arc::new(test_metrics("write-deadline"));
    let (proceed_tx, proceed_rx) = mpsc::channel();
    let (handler_done_tx, handler_done_rx) = mpsc::channel();
    let handler = std::thread::Builder::new()
        .name(format!("write-deadline-handler-{case:?}"))
        .spawn(move || {
            let started = Instant::now();
            let outcome = {
                let _guard = ConnGuard(handler_connections);
                let mut server = server;
                let header_result = server
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                    )
                    .and_then(|_| server.flush());
                headers_tx
                    .send(header_result.as_ref().map(|_| ()).map_err(|e| e.kind()))
                    .expect("announce SSE header result");
                header_result.map_err(|error| error.to_string()).and_then(|_| {
                    submit(
                        &job_tx,
                        &metrics,
                        generation_request("write deadline", 2, true),
                        &|| false,
                        |piece, _| {
                            if write!(server, "data: {piece}\n\n")
                                .and_then(|_| server.flush())
                                .is_ok()
                            {
                                GenerationControl::Continue(())
                            } else {
                                GenerationControl::Break(())
                            }
                        },
                    )
                    .map(|_| ())
                    .map_err(|error| error.to_string())
                })
            };
            handler_done_tx
                .send((outcome, started.elapsed()))
                .expect("report write-deadline handler completion");
        })
        .expect("spawn write-deadline handler");

    let (producer_progress_tx, producer_progress_rx) = mpsc::channel();
    let (engine_done_tx, engine_done_rx) = mpsc::channel();
    let engine = std::thread::Builder::new()
        .name(format!("write-deadline-engine-{case:?}"))
        .spawn(move || {
            let EngineMsg::Generate(job) = job_rx
                .recv_timeout(completion_budget)
                .expect("stream handler submits response subscription")
            else {
                panic!("a stream handler sends a generation job");
            };
            proceed_rx
                .recv_timeout(completion_budget)
                .expect("release fixture engine");
            if case == WriteDeadlineClientCase::HealthyStream {
                job.reply
                    .send(GenEvent::Token("one".into(), terminal_token_info()))
                    .expect("send first healthy event");
                job.reply
                    .send(GenEvent::Token("two".into(), terminal_token_info()))
                    .expect("send second healthy event");
                let done = Completion {
                    prompt_tokens: 1,
                    tokens: vec![1, 2],
                    text: "onetwo".into(),
                    logprobs: vec![],
                    finish: FinishReason::Eos,
                };
                job.reply
                    .send(GenEvent::Done(done))
                    .expect("send healthy terminal event");
                engine_done_tx
                    .send(WriteDeadlineEngineOutcome::Completed)
                    .expect("report healthy completion");
                return;
            }

            let oversized_event = "x".repeat(8 * 1024 * 1024);
            job.reply
                .send(GenEvent::Token(oversized_event, terminal_token_info()))
                .expect("queue buffer-saturating event");
            producer_progress_tx
                .send(())
                .expect("prove producer progressed past large event");
            loop {
                if job
                    .reply
                    .send(GenEvent::Token("after".into(), terminal_token_info()))
                    .is_err()
                {
                    engine_done_tx
                        .send(WriteDeadlineEngineOutcome::Cancelled)
                        .expect("report response subscription cancellation");
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        })
        .expect("spawn write-deadline engine");

    let header_result = headers_rx
        .recv_timeout(completion_budget)
        .expect("SSE headers finish within fixture budget");
    assert!(header_result.is_ok(), "{case:?}: SSE headers must fit");
    let (healthy_response_tx, healthy_response_rx) = mpsc::channel();
    let healthy_reader = match case {
        WriteDeadlineClientCase::HealthyStream => {
            let mut healthy_client = client.take().expect("healthy fixture client");
            Some(std::thread::spawn(move || {
                let mut response = String::new();
                let result = healthy_client
                    .read_to_string(&mut response)
                    .map(|_| response);
                healthy_response_tx
                    .send(result)
                    .expect("report healthy streaming response");
            }))
        }
        WriteDeadlineClientCase::DisconnectedBeforeEvent => {
            client.take();
            None
        }
        WriteDeadlineClientCase::NeverReads => None,
    };
    proceed_tx.send(()).expect("start fixture response events");

    if case != WriteDeadlineClientCase::HealthyStream {
        producer_progress_rx
            .recv_timeout(Duration::from_millis(500))
            .expect("engine producer must not block on the slow socket consumer");
    }
    let expected_engine = if case == WriteDeadlineClientCase::HealthyStream {
        WriteDeadlineEngineOutcome::Completed
    } else {
        WriteDeadlineEngineOutcome::Cancelled
    };
    assert_eq!(
        engine_done_rx
            .recv_timeout(completion_budget)
            .expect("engine observes terminal response state"),
        expected_engine,
        "{case:?}"
    );
    let (handler_result, elapsed) = handler_done_rx
        .recv_timeout(completion_budget)
        .expect("handler terminates within the configured write budget");
    assert!(
        elapsed <= completion_budget,
        "{case:?}: handler elapsed {elapsed:?} exceeds {completion_budget:?}"
    );
    assert_eq!(
        live_connections.load(Ordering::SeqCst),
        0,
        "{case:?}: ConnGuard must release the connection-count slot"
    );

    match case {
        WriteDeadlineClientCase::HealthyStream => {
            handler_result.expect("healthy stream handler result");
            let response = healthy_response_rx
                .recv_timeout(completion_budget)
                .expect("healthy client reads while events stream")
                .expect("read healthy stream response");
            assert!(
                response.contains("data: one\n\ndata: two\n\n"),
                "ordered SSE events: {response}"
            );
        }
        WriteDeadlineClientCase::NeverReads | WriteDeadlineClientCase::DisconnectedBeforeEvent => {
            let error = handler_result.expect_err("unwritable stream must cancel submit");
            assert!(error.contains("cancelled"), "{case:?}: {error}");
        }
    }

    handler.join().expect("join completed response handler");
    engine.join().expect("join completed fixture engine");
    if let Some(reader) = healthy_reader {
        reader.join().expect("join healthy streaming client");
    }
}
