use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use poot_llm::driver::block_table::{BLOCK_SIZE, PagedKvCache};
use poot_llm::driver::prefix_cache::PrefixIdentity;
use poot_llm::encoder::EncoderPooling;
use poot_llm::{
    GenerationControl, LoraAdapterLease, MropeDecodeState, MropePosition, MropePositionIds, Runner,
    Sampler,
};

use crate::admin::{LoraAdminBackend, LoraAdminBackendResolver, LoraAdminPolicy};
use crate::api::{
    ChatReq, CompletionReq, GuidedDecodeError, MAX_CHOICES, MAX_LOGPROBS, MAX_NEW_TOKENS_CAP,
    MAX_STOP_SEQUENCES, ResponseFormatConstraint, Stop, StreamOptions, UnsupportedCapabilityError,
    apply_stop, clamp_logprobs, clamp_n, classify_handler_error, default_max, parse_tool_calls,
    reject_unsupported_modality, resolve_forced_tools, response_format_constraint, sampler_of,
    suppress_stream_usage, tool_call_stream_deltas,
};
use crate::batch::commit::{
    commit_token, record_non_mrope_prefix_cache_admission, release_slot_kv,
};
use crate::batch::kv_budget::{
    KV_POOL_VRAM_MULTIPLIER, KV_POOL_VRAM_MULTIPLIER_INPLACE,
    ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN,
    WGPU_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN, fit_kv_to_budget,
};
use crate::batch::preempt::{
    PreemptedSeq, admit_prefix_with_preemption, preempt_slot,
    reserve_without_prefix_with_preemption, resume_preempted, select_preemption_victim,
};
use crate::batch::schedule::{
    drain_ready_jobs, resume_into_free_slots, schedule_prefill_admissions,
};
use crate::batch::slot::{
    Slot, admit_job_to_slot, advance_slot_position, build_mrope_rows, fail_active_slots,
    reject_job, shutdown_batch_requests,
};
use crate::handlers::{
    StreamStops, account_generation_error, embedding_value, encoder_pooling_override, handle_chat,
    sse_write_probe, stream_chat, stream_completion, stream_receipt,
};
use crate::http::{
    ConnGuard, MAX_BODY_BYTES, MAX_HEADER_BYTES, RateLimiter, RequestRead,
    ResponseWriteTimeoutError, body_len_ok, configure_response_writes, error_body, handle,
    max_connections_from, model_object, read_request, read_request_for_routing, read_request_from,
    request_read_timeout_from, respond, respond_error, respond_typed, response_write_timeout_from,
};
use crate::lifecycle::{
    CANCELLATION_SETTLE_TIMEOUT, MAX_DRAIN_TIMEOUT_SECS, WorkerRegistry, deadline_after,
    drain_cancellation_requested, reset_drain_cancellation, server_drain_timeout_from, submit,
};
use crate::metrics::{CacheMetrics, Histogram, Metrics, SpeculativeMetrics};
use crate::types::{
    Backend, GenEvent, GenerationError, Job, PerTokenInfo, SHUTDOWN, terminal_delivery_probe,
};

mod admin;
mod drain;
mod http_parse;
mod no_engine;
mod other;
mod preemption;
mod scheduling;
mod streaming;
mod terminal_done;
mod write_deadline;

/// The test double at the engine boundary: drains the job queue and runs each job through `generate`,
/// streaming `Token` events when the job asked to stream and a terminal `Done`/`Failed`, with the same
/// outcome accounting the batched commit path has (a terminal `Done` through `GenEvent::deliver_done`, a
/// piece that cannot be delivered is a cancellation). It stands in for the serving loop so the request
/// side (`submit`, the handlers, write deadlines) is tested without a model or a device.
fn test_engine_loop<F>(rx: Receiver<Job>, metrics: Arc<Metrics>, mut generate: F)
where
    F: FnMut(
        &str,
        usize,
        &mut Sampler,
        &[String],
        &mut dyn FnMut(&str) -> GenerationControl,
    ) -> Result<(Vec<u32>, bool)>,
{
    for job in rx {
        let Job {
            prompt,
            max_new,
            mut sampler,
            stop,
            stream,
            reply,
            ..
        } = job;
        let decode_start = Instant::now();
        let mut last_token: Option<Instant> = None;
        let mut token_idx = 0usize;
        let mut cancelled = false;
        let res = generate(&prompt, max_new, &mut sampler, &stop, &mut |piece: &str| {
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
        });
        if cancelled {
            metrics.requests_cancelled.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        match res {
            Ok((tokens, natural)) => {
                GenEvent::deliver_done(&metrics, &reply, tokens, sampler.take_logprobs(), natural);
            }
            Err(e) => {
                let _ = reply.send(GenEvent::failed(&metrics, format!("{e:#}")));
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GenerationFailureAccountingCase {
    DeliveredDirect,
    DroppedDirect,
    DeliveredRejection,
    DroppedRejection,
    ActiveFailure,
    QueuedShutdown,
    ActiveShutdown,
    PreemptedShutdown,
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
    runner: Arc<Runner>,
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
    let handler_runner = Arc::clone(&runner);
    let (job_tx, job_rx) = mpsc::channel::<Job>();

    let handler = std::thread::spawn(move || {
        let mut server = server;
        match endpoint {
            GenerationStreamEndpoint::Completion => {
                let req: CompletionReq = serde_json::from_str(
                    r#"{"prompt":"a","max_tokens":1,"stream":true,"stop":"ab"}"#,
                )
                .expect("completion terminal request");
                stream_completion(
                    &mut server,
                    Some(&job_tx),
                    &handler_runner,
                    &handler_metrics,
                    &req,
                )
            }
            GenerationStreamEndpoint::Chat => {
                let req: ChatReq = serde_json::from_str(
                    r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":1,"stream":true,"stop":"ab"}"#,
                )
                .expect("chat terminal request");
                stream_chat(
                    &mut server,
                    Some(&job_tx),
                    &handler_runner,
                    &handler_metrics,
                    &req,
                )
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
                                Ok((vec![0], false))
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
                                Ok((vec![3], false))
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
            let job = job_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("handler submitted its EOF fixture job");
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

fn distinct_mrope_layouts() -> (MropePositionIds, MropePositionIds) {
    let collapsed =
        poot_llm::build_mrope_position_ids(&[poot_llm::MropeSegment::Text(20)], 2).unwrap();
    let sectioned = poot_llm::build_mrope_position_ids(
        &[
            poot_llm::MropeSegment::Text(1),
            poot_llm::MropeSegment::Image(poot_llm::MropeGrid::new(1, 2, 36)),
            poot_llm::MropeSegment::Text(1),
        ],
        2,
    )
    .unwrap();
    assert_eq!(collapsed.len(), sectioned.len());
    assert_ne!(
        collapsed.position(2).unwrap(),
        sectioned.position(2).unwrap()
    );
    (collapsed, sectioned)
}

// Recompute preemption (spec 237): `select_preemption_victim`, `preempt_slot`, `admit_prefix_with_preemption` and `resume_preempted` are host logic over `PagedKvCache` (no GPU types), so the bookkeeping is tested without a device.

/// `n` distinct token ids starting at `base` (a copy of the `toks` helper in `poot_llm::driver::block_table`'s tests).
fn preempt_toks(base: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

/// A `Slot` at `pos` with `generated` tokens produced: the two fields the preemption machinery reads. Other fields are placeholders.
fn preempt_test_slot(tokens: Vec<u32>, pos: usize, generated: usize, max_new: usize) -> Slot {
    let (reply, _rx) = mpsc::channel();
    // `saturating_sub`: callers pass arbitrary short token vectors; only `pos`/`generated` matter.
    let prompt_len = tokens.len().saturating_sub(generated);
    Slot {
        tokens,
        prompt_len,
        pos,
        generated,
        max_new,
        sampler: Sampler::greedy(),
        stop: vec![],
        stream: false,
        reply,
        prefilled: true,
        decode_start: None,
        last_token: None,
        lora_adapter: LoraAdapterLease::none(),
        mrope: None,
    }
}

// Helper: feed pieces through a StreamStops, returning the concatenated emitted text + whether a stop hit.
fn run_stream_stops(stops: &[&str], pieces: &[&str]) -> (String, bool) {
    let mut buf = StreamStops::new(stops.iter().map(|s| s.to_string()).collect());
    let mut out = String::new();
    for p in pieces {
        out.push_str(&buf.push(p));
    }
    out.push_str(&buf.finish());
    (out, buf.hit)
}

fn unique_fixture_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{n}", std::process::id())
}

/// A tiny all-F32 granitemoe GGUF (one layer, two experts, 4-token vocab): a decoder the Runner loads.
/// granitemoe is not a registered family, so it stays on the Runner until POOT-738; a registered
/// (dense) family is refused by `Runner::load_gguf`.
pub(crate) fn write_tiny_granitemoe_gguf() -> std::path::PathBuf {
    use poot_load::gguf::{GgufValue, write_gguf};
    let (h, qd, kvd, inter, vocab, experts) = (32usize, 32usize, 16usize, 64usize, 4usize, 2usize);
    let f32_bytes = |n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (((i % 7) as f32) * 0.1 - 0.3).to_le_bytes())
            .collect()
    };
    const F32: u32 = 0;
    // ggml dims are reversed: a `[out, in]` weight is `[in, out]` here, an `[E, out, in]` expert stack
    // `[in, out, E]`.
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "token_embd.weight",
            vec![h as u64, vocab as u64],
            F32,
            f32_bytes(h * vocab),
        ),
        (
            "output.weight",
            vec![h as u64, vocab as u64],
            F32,
            f32_bytes(h * vocab),
        ),
        ("output_norm.weight", vec![h as u64], F32, f32_bytes(h)),
        (
            "blk.0.attn_q.weight",
            vec![h as u64, qd as u64],
            F32,
            f32_bytes(h * qd),
        ),
        (
            "blk.0.attn_k.weight",
            vec![h as u64, kvd as u64],
            F32,
            f32_bytes(h * kvd),
        ),
        (
            "blk.0.attn_v.weight",
            vec![h as u64, kvd as u64],
            F32,
            f32_bytes(h * kvd),
        ),
        (
            "blk.0.attn_output.weight",
            vec![qd as u64, h as u64],
            F32,
            f32_bytes(qd * h),
        ),
        ("blk.0.attn_norm.weight", vec![h as u64], F32, f32_bytes(h)),
        ("blk.0.ffn_norm.weight", vec![h as u64], F32, f32_bytes(h)),
        (
            "blk.0.ffn_gate_inp.weight",
            vec![h as u64, experts as u64],
            F32,
            f32_bytes(h * experts),
        ),
        (
            "blk.0.ffn_gate_exps.weight",
            vec![h as u64, inter as u64, experts as u64],
            F32,
            f32_bytes(h * inter * experts),
        ),
        (
            "blk.0.ffn_up_exps.weight",
            vec![h as u64, inter as u64, experts as u64],
            F32,
            f32_bytes(h * inter * experts),
        ),
        (
            "blk.0.ffn_down_exps.weight",
            vec![inter as u64, h as u64, experts as u64],
            F32,
            f32_bytes(inter * h * experts),
        ),
    ];
    let kvs = vec![
        ("general.architecture", GgufValue::Str("granitemoe".into())),
        ("granitemoe.embedding_length", GgufValue::U32(h as u32)),
        ("granitemoe.block_count", GgufValue::U32(1)),
        ("granitemoe.attention.head_count", GgufValue::U32(2)),
        ("granitemoe.attention.head_count_kv", GgufValue::U32(1)),
        (
            "granitemoe.feed_forward_length",
            GgufValue::U32(inter as u32),
        ),
        ("granitemoe.context_length", GgufValue::U32(64)),
        ("granitemoe.expert_count", GgufValue::U32(experts as u32)),
        ("granitemoe.expert_used_count", GgufValue::U32(1)),
        ("granitemoe.embedding_scale", GgufValue::F32(1.0)),
        ("granitemoe.attention.scale", GgufValue::F32(0.25)),
        ("granitemoe.residual_scale", GgufValue::F32(1.0)),
        ("granitemoe.logit_scale", GgufValue::F32(1.0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(99)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "ab", "c"]
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let path = std::env::temp_dir().join(format!(
        "poot_serve_granitemoe_fixture_{}.gguf",
        unique_fixture_suffix()
    ));
    std::fs::write(&path, write_gguf(&kvs, &tensors)).unwrap();
    path
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
        requests_preempted: AtomicU64::new(0),
        inflight: AtomicU64::new(0),
        ttft: Histogram::new(),
        itl: Histogram::new(),
        e2e: Histogram::new(),
        queue_wait: Histogram::new(),
        batch_size: AtomicU64::new(0),
        slot_capacity: 1,
        on_gpu: false,
        profile: false,
        model: model.into(),
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics::default(),
        speculative: SpeculativeMetrics::default(),
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FakeLoraAdminSnapshot {
    list_calls: usize,
    load_paths: Vec<String>,
    lookup_calls: usize,
    unload_calls: usize,
    adapters: Vec<String>,
}

struct FakeLoraAdmin {
    state: Mutex<FakeLoraAdminSnapshot>,
}

impl FakeLoraAdmin {
    fn with_base_adapter() -> Self {
        Self {
            state: Mutex::new(FakeLoraAdminSnapshot {
                adapters: vec!["base".into()],
                ..FakeLoraAdminSnapshot::default()
            }),
        }
    }

    fn snapshot(&self) -> FakeLoraAdminSnapshot {
        self.state.lock().unwrap().clone()
    }
}

impl LoraAdminBackend for FakeLoraAdmin {
    fn list(&self) -> Vec<(String, usize, usize)> {
        let mut state = self.state.lock().unwrap();
        state.list_calls += 1;
        state
            .adapters
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index + 1, 0))
            .collect()
    }

    fn hot_load_dir(&self, name: &str, path: &str) -> Result<usize> {
        let mut state = self.state.lock().unwrap();
        state.load_paths.push(path.into());
        state.adapters.push(name.into());
        Ok(state.adapters.len())
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        let mut state = self.state.lock().unwrap();
        state.lookup_calls += 1;
        state
            .adapters
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| index + 1)
    }

    fn hot_unload(&self, name: &str) -> Result<usize> {
        let mut state = self.state.lock().unwrap();
        state.unload_calls += 1;
        let index = state
            .adapters
            .iter()
            .position(|candidate| candidate == name)
            .context("fake adapter must exist before unload")?;
        state.adapters.remove(index);
        Ok(index + 1)
    }
}

struct FakeLoraAdminResolver {
    admin: FakeLoraAdmin,
    resolve_calls: AtomicUsize,
}

impl FakeLoraAdminResolver {
    fn with_base_adapter() -> Self {
        Self {
            admin: FakeLoraAdmin::with_base_adapter(),
            resolve_calls: AtomicUsize::new(0),
        }
    }
}

impl LoraAdminBackendResolver for FakeLoraAdminResolver {
    fn resolve_lora_admin_backend(&self) -> Option<&dyn LoraAdminBackend> {
        self.resolve_calls.fetch_add(1, Ordering::SeqCst);
        Some(&self.admin)
    }
}

fn run_fake_lora_admin_socket(
    policy: &LoraAdminPolicy,
    resolver: &FakeLoraAdminResolver,
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
        read_request_for_routing(&mut server, resolver, policy)
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
/// body byte. A denial can only arrive if production authorizes before body reading. The independent lazy
/// resolver count proves it also authorizes before discriminating the production backend.
fn run_denied_lora_admin_head_without_body(
    policy: LoraAdminPolicy,
    method: &str,
    path: &str,
    authorization: &[&str],
    content_length: usize,
) -> (String, Arc<FakeLoraAdminResolver>) {
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

    let resolver = Arc::new(FakeLoraAdminResolver::with_base_adapter());
    let resolver_for_handler = Arc::clone(&resolver);
    let handler = std::thread::spawn(move || {
        read_request_for_routing(&mut server, resolver_for_handler.as_ref(), &policy)
    });

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
    (response, resolver)
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
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let (headers_tx, headers_rx) = mpsc::channel();
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
                        "write deadline",
                        2,
                        Sampler::greedy(),
                        vec![],
                        true,
                        LoraAdapterLease::none(),
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
            let job = job_rx
                .recv_timeout(completion_budget)
                .expect("stream handler submits response subscription");
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
                job.reply
                    .send(GenEvent::Done(vec![1, 2], vec![], true))
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
