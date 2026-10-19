//! Process configuration, model startup, listener orchestration, and graceful shutdown.

use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use poot_llm::driver::error::DriverError;
use poot_llm::driver::{
    DraftPolicy, LoadOptions, ModelHandle, open_executor, open_group, open_timed_executor,
};
use poot_llm::encoders::Pooling;
use poot_llm::{ImageCheckpoint, read_image_checkpoint};
use poot_models::registry::Registry;

use crate::admin::LoraAdminPolicyMode;
use crate::batch::engine::{DriverEngine, run_engine_thread};
use crate::config::{PlacementSpec, ServeConfig, TelemetryConfig};
use crate::engine_plan::{
    Coordinated, EngineArgs, EngineHealth, EnginePlan, EngineState, Opened, Started, StartupError,
    device_budget,
};
use crate::http::{
    BodyCaps, ConnGuard, HostPolicy, Limits, RateLimiter, RequestQueue, configure_response_writes,
    handle,
};
use crate::lifecycle::{
    CANCELLATION_SETTLE_TIMEOUT, ServerDrainTimeout, WorkerRegistry, deadline_after,
    reset_drain_cancellation,
};
use crate::metrics::{CacheMetrics, Histogram, Metrics, SpeculativeMetrics, TimingMetrics};
use crate::types::{EngineMsg, EngineView, SHUTDOWN, Served};
use crate::vision::spawn_image_front_end;

/// No model path was given and `POOT_MODELS_DIR` is unset or empty.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model path given and POOT_MODELS_DIR is not set: pass a model path or set POOT_MODELS_DIR to the directory that holds `{DEFAULT_MODEL}`"
)]
pub(crate) struct ModelDirUnset;

/// The model directory under `POOT_MODELS_DIR` used when no path is given.
const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

/// The values the accept loop and the drain run on, built once from the configuration and the address the
/// listener bound: what bounds a request (rate, queue, body caps, origin, arrival time), the connection
/// cap, the write bound and the drain bound.
pub(crate) struct Transport {
    pub(crate) limits: Arc<Limits>,
    pub(crate) max_connections: usize,
    pub(crate) write_timeout: Duration,
    pub(crate) drain_timeout: ServerDrainTimeout,
}

impl Transport {
    pub(crate) fn of(config: &ServeConfig, bound: SocketAddr) -> Self {
        const MIB: usize = 1024 * 1024;
        Self {
            limits: Arc::new(Limits {
                limiter: RateLimiter::new(config.rate_limit_rpm),
                queue: RequestQueue::new(config.max_pending_requests),
                receiving: RequestQueue::new(config.max_receiving_requests),
                bodies: BodyCaps {
                    text: config.max_body_mib * MIB,
                    images: config.max_image_body_mib * MIB,
                },
                hosts: HostPolicy::for_listener(bound),
                request_timeout: config.request_timeout_secs,
            }),
            max_connections: config.max_connections,
            write_timeout: config.write_timeout_secs,
            drain_timeout: config.drain_timeout_secs,
        }
    }
}

/// What the operator asked of the engine plan: the backend, the placement, the draft policy and the
/// expert pool. A device group records no device timing, so asking for both is refused here, before
/// anything loads.
pub(crate) fn engine_args(config: &ServeConfig, draft: DraftPolicy) -> Result<EngineArgs> {
    anyhow::ensure!(
        config.placement == PlacementSpec::Single || !config.device_timing,
        "--device-timing is not recorded over a device group: use it with --placement single"
    );
    Ok(EngineArgs::new(config.backend, draft)
        .with_shape(config.rows, config.max_seq)
        .with_placement(config.placement.clone())
        .with_expert_pool(config.expert_pool))
}

/// The draft policy the flags name: `Off` with none of the three. A `--draft-model` checkpoint is loaded
/// here, so a path that does not load is a startup error before the listener binds.
pub(crate) fn draft_policy(config: &ServeConfig) -> Result<DraftPolicy> {
    match (&config.draft_lookup, &config.draft_model, config.draft_mtp) {
        (None, None, None) => Ok(DraftPolicy::Off),
        (Some(lookup), None, None) => Ok(DraftPolicy::PromptLookup {
            ngram: lookup.ngram,
            k: lookup.k,
        }),
        (None, Some(draft), None) => Ok(DraftPolicy::DraftModel {
            k: draft.k,
            handle: load_model(&draft.path, &LoadOptions::default())
                .context("load the draft model")?,
        }),
        (None, None, Some(k)) => Ok(DraftPolicy::Mtp { k }),
        _ => anyhow::bail!(
            "--draft-lookup, --draft-model and --draft-mtp are exclusive: name one policy"
        ),
    }
}

/// The model path: the first positional argument, else `DEFAULT_MODEL` under `POOT_MODELS_DIR`.
fn model_dir_or_env(positional: Option<&String>) -> Result<String, ModelDirUnset> {
    if let Some(path) = positional {
        return Ok(path.clone());
    }
    match std::env::var("POOT_MODELS_DIR") {
        Ok(dir) if !dir.is_empty() => Ok(std::path::Path::new(&dir)
            .join(DEFAULT_MODEL)
            .to_string_lossy()
            .into_owned()),
        _ => Err(ModelDirUnset),
    }
}

/// Signal handler for graceful shutdown: sets the global SHUTDOWN flag so the accept loop exits.
pub(crate) extern "C" fn sigint_handler(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Served model name: the model-dir basename, or the GGUF file stem; echoed in responses and /v1/models.
fn served_model_name(model_dir: &str) -> String {
    std::path::Path::new(model_dir)
        .file_stem()
        .filter(|_| model_dir.ends_with(".gguf"))
        .or_else(|| std::path::Path::new(model_dir).file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "poot-model".to_string())
}

/// RUST_LOG controls verbosity (default "info"). ANSI colors only when stdout is a terminal, so piped
/// logs (supervisors, containers, test harnesses) stay greppable.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
}

/// SIGINT/SIGTERM set SHUTDOWN so the accept loop exits.
fn install_signal_handlers() {
    // SAFETY: `sigint_handler` is an `extern "C" fn(c_int)` that only stores to an atomic, which is
    // async-signal-safe, and it stays valid for the life of the process. `signal` takes no pointers
    // the handler could outlive.
    unsafe {
        libc::signal(
            libc::SIGINT,
            sigint_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            sigint_handler as *const () as libc::sighandler_t,
        );
    }
}

/// Load the checkpoint. Every failure is a startup error, and `run` has not bound the listener yet.
fn load_model(model_dir: &str, options: &LoadOptions) -> Result<Arc<ModelHandle>> {
    tracing::info!(model_dir, "loading model");
    let model = Arc::new(
        ModelHandle::load(
            std::path::Path::new(model_dir),
            &Registry::builtin().context("build the family registry")?,
            options,
        )
        .context("load model")?,
    );
    let config = model.config();
    tracing::info!(
        family = %config.family,
        vocab = config.vocab,
        max_positions = config.max_positions,
        "model config"
    );
    Ok(model)
}

/// The served model, and the image front end of a vision-language checkpoint.
type LoadedModel = (Arc<ModelHandle>, Option<Arc<dyn ImageCheckpoint>>);

/// Load the served model. A vision-language checkpoint (SmolVLM, Qwen2.5-VL) is served by its text tower,
/// with its image front end beside it; any other checkpoint is the model itself.
fn load_served(model_dir: &str, options: &LoadOptions) -> Result<LoadedModel> {
    let path = std::path::Path::new(model_dir);
    let registry = Registry::builtin().context("build the family registry")?;
    let checkpoint =
        read_image_checkpoint(path, &registry).context("load vision-language model")?;
    match checkpoint {
        Some(checkpoint) => {
            anyhow::ensure!(
                options.chat_template.is_none(),
                "--chat-template is not supported for a vision-language checkpoint: its image chats render through the checkpoint's own template"
            );
            tracing::info!(model_dir, "loaded vision-language model");
            Ok((Arc::clone(checkpoint.text()), Some(checkpoint)))
        }
        None => Ok((load_model(model_dir, options)?, None)),
    }
}

/// Start the engine thread and wait for it to report its plan. The device is opened, the workload fitted
/// and prepared, and the loop run on that one thread: an executor is not `Send`, so the driver cannot be
/// built elsewhere and moved in. A startup failure comes back here as the error, before the listener
/// binds; a started engine returns the metrics its loop updates and the view of itself it publishes for the
/// connection threads.
fn spawn_engine(
    workers: &mut WorkerRegistry,
    model: Arc<ModelHandle>,
    args: EngineArgs,
    telemetry: TelemetryConfig,
    model_name: String,
    health: &EngineHealth,
    jobs: mpsc::Receiver<EngineMsg>,
) -> Result<(Arc<Metrics>, EngineView)> {
    type Startup = Result<(Arc<Metrics>, EngineView), StartupError>;
    let (started_tx, started_rx) = mpsc::channel::<Startup>();
    let health = health.clone();
    workers
        .spawn("poot-engine", move || {
            let open = |args: &EngineArgs| -> Result<Opened, DriverError> {
                match &args.placement {
                    PlacementSpec::Single => {
                        let executor = match telemetry {
                            TelemetryConfig::CountersOnly => open_executor(args.backend),
                            TelemetryConfig::Detailed(detail) => {
                                open_timed_executor(args.backend, detail)
                            }
                        }?;
                        Ok(Opened::One(executor))
                    }
                    PlacementSpec::Pipeline { ordinals } => Ok(Opened::Pipeline(Box::new(
                        Coordinated(open_group(args.backend, ordinals)?),
                    ))),
                    PlacementSpec::Tensor { ordinals } => Ok(Opened::Tensor(Box::new(
                        Coordinated(open_group(args.backend, ordinals)?),
                    ))),
                }
            };
            let started = EnginePlan::build(&args, &model, open, device_budget, &health);
            let Started { plan, driver } = match started {
                Ok(started) => started,
                Err(error) => {
                    let _ = started_tx.send(Err(error));
                    return;
                }
            };
            tracing::info!(
                backend = plan.backend_label(),
                devices = ?plan.devices,
                placement = ?plan.placement,
                model = plan.model,
                rows = plan.rows(),
                workload = ?plan.workload,
                budget = ?plan.budget.devices,
                "engine plan"
            );
            let metrics = Arc::new(new_metrics(
                model_name,
                &plan,
                &driver.live_role_bytes(),
                telemetry,
            ));
            let view = EngineView::new(plan.caps.clone());
            let started = (Arc::clone(&metrics), view.clone());
            if started_tx.send(Ok(started)).is_err() {
                return;
            }
            let engine = run_engine_thread(
                DriverEngine::new(driver, model, view),
                jobs,
                &metrics,
                &health,
            );
            retain_dead_engine(engine, &health);
        })
        .context("spawn the engine thread")?;
    started_rx
        .recv()
        .context("the engine thread ended during startup")?
        .context("start the engine")
}

/// A dead engine's device buffers stay owned until the process shuts down: the device may still be
/// executing work whose owners a drop would free.
fn retain_dead_engine(engine: DriverEngine, health: &EngineHealth) {
    if matches!(health.state(), EngineState::Dead(_)) {
        while !SHUTDOWN.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    drop(engine);
}

fn new_metrics(
    model_name: String,
    plan: &EnginePlan,
    driver_bytes: &[(poot_runtime_common::BufferRole, u64)],
    telemetry: TelemetryConfig,
) -> Metrics {
    let live = |role| {
        driver_bytes
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map_or(0, |&(_, bytes)| bytes)
    };
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
        slot_capacity: plan.rows() as u64,
        backend: plan.backend_label(),
        model: model_name,
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics {
            gpu_weight_bytes: AtomicU64::new(live(poot_runtime_common::BufferRole::Weight)),
            gpu_kv_bytes: AtomicU64::new(live(poot_runtime_common::BufferRole::State)),
            kv_total_blocks: AtomicU64::new(plan.caps.kv_blocks as u64),
            gpu_vram_total_bytes: AtomicU64::new(plan.budget.available_bytes().unwrap_or(0)),
            ..CacheMetrics::default()
        },
        speculative: SpeculativeMetrics::default(),
        timing: TimingMetrics::new(telemetry),
    }
}

pub(crate) fn run() -> Result<()> {
    reset_drain_cancellation();
    SHUTDOWN.store(false, Ordering::SeqCst);
    init_logging();
    install_signal_handlers();

    // Every setting parses before the model is detected or any loader runs, so a bad value fails fast.
    let config = ServeConfig::parse();
    let model_dir = model_dir_or_env(config.model.as_ref())?;
    let load_options = config.load_options();
    let telemetry = config.telemetry();
    let model_name = served_model_name(&model_dir);

    // The engine plan decides the backend, fits the workload to the device and prepares it. A refused
    // model or an unopenable device is a typed startup error and the listener never binds. The prepared
    // driver moves to the engine thread, which runs the one scheduling loop over it: generation,
    // embeddings, rerank and LoRA administration are messages on its one queue.
    let health = EngineHealth::starting();
    let (model, checkpoint) = load_served(&model_dir, &load_options)?;
    let mut workers = WorkerRegistry::new();
    let (job_tx, job_rx) = mpsc::channel::<EngineMsg>();
    let draft = draft_policy(&config)?;
    // The vision tower opens first, so the engine fits its workload to the memory the tower left.
    let images = checkpoint
        .map(|checkpoint| spawn_image_front_end(&mut workers, &health, checkpoint, config.backend))
        .transpose()?;
    let args = engine_args(&config, draft)?;
    let (metrics, view) = spawn_engine(
        &mut workers,
        Arc::clone(&model),
        if images.is_some() {
            args.with_embeds()
        } else {
            args
        },
        telemetry,
        model_name,
        &health,
        job_rx,
    )?;
    let served = Arc::new(Served {
        pooling: Pooling::read(std::path::Path::new(&model_dir)),
        model,
        engine: job_tx,
        view,
        images,
        max_queue_wait: config.max_queue_wait_secs,
    });

    let addr = &config.addr;
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    let Transport {
        limits,
        max_connections: max_conns,
        write_timeout,
        drain_timeout,
    } = Transport::of(&config, listener.local_addr()?);
    if limits.limiter.rpm > 0 {
        tracing::info!(
            rpm = limits.limiter.rpm,
            "per-key rate limit active (--rate-limit-rpm)"
        );
    }
    let lora_admin_policy = Arc::new(config.lora_admin_policy(listener.local_addr()?.ip())?);
    match lora_admin_policy.mode() {
        LoraAdminPolicyMode::LocalOnly => {
            tracing::info!("LoRA administration enabled under the loopback-only local trust policy")
        }
        LoraAdminPolicyMode::BearerRequired => {
            tracing::info!("LoRA administration enabled with bearer authentication")
        }
        LoraAdminPolicyMode::Disabled => tracing::warn!(
            "LoRA administration disabled on the non-loopback listener; ordinary inference remains unauthenticated"
        ),
    }
    // Non-blocking accept so the shutdown flag is checked between connections.
    listener.set_nonblocking(true)?;
    let live_conns = Arc::new(AtomicUsize::new(0));
    tracing::info!(%addr, max_connections = max_conns, write_timeout_secs = write_timeout.as_secs(), drain_timeout_secs = drain_timeout.duration().as_secs(), "poot-serve listening (POST /v1/completions, /v1/chat/completions; GET /health, /metrics, /metrics/prometheus)");

    loop {
        workers.reap_ready();
        if SHUTDOWN.load(Ordering::SeqCst) {
            tracing::info!("shutting down (signal received)");
            health.begin_draining();
            break;
        }
        match listener.accept() {
            Ok((s, _)) => {
                let accepted = Instant::now();
                // A signal may land after the top-of-loop check while a queued socket makes `accept` succeed; the signal is
                // the admission boundary, so close that socket rather than start one more request.
                if SHUTDOWN.load(Ordering::SeqCst) {
                    drop(s);
                    tracing::info!("shutting down (signal received during accept)");
                    health.begin_draining();
                    break;
                }
                // Cap live connections: one thread per connection with no cap lets a client exhaust thread/memory limits
                // before the rate limiter or any size cap runs.
                if max_conns > 0 && live_conns.load(Ordering::SeqCst) >= max_conns {
                    drop(s);
                    continue;
                }
                // Disable Nagle: SSE writes one small chunk + flush per token, and Nagle + delayed-ACK stalls each write
                // (tens of ms/token even on localhost).
                let _ = s.set_nodelay(true);
                // Bound every response write; if the timeout cannot be installed, close the connection.
                if let Err(e) = configure_response_writes(&s, write_timeout) {
                    tracing::error!(error = %e, "configure connection write timeout");
                    continue;
                }
                let served = Arc::clone(&served);
                let health = health.clone();
                let metrics = Arc::clone(&metrics);
                let limits = Arc::clone(&limits);
                let lora_admin_policy = Arc::clone(&lora_admin_policy);
                live_conns.fetch_add(1, Ordering::SeqCst);
                let conn_counter = Arc::clone(&live_conns);
                // One thread per connection: parse and shape concurrently.
                if let Err(e) = workers.spawn_connection("poot-conn", s, move |s| {
                    let _guard = ConnGuard(conn_counter);
                    if let Err(e) = handle(
                        s,
                        accepted,
                        &served,
                        &health,
                        &metrics,
                        &limits,
                        &lora_admin_policy,
                    ) {
                        tracing::error!(error = format!("{e:#}"), "request error");
                    }
                }) {
                    tracing::error!(error = %e, "spawn connection thread");
                    // The closure (and its ConnGuard) never ran; undo the increment by hand.
                    live_conns.fetch_sub(1, Ordering::SeqCst);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(e) => tracing::error!(error = %e, "accept error"),
        }
    }
    // The signal closes admission first; accepted connections keep their sockets and job senders so they can
    // finish.
    drop(listener);
    let graceful_deadline = drain_timeout.deadline_from(Instant::now());
    let mut graceful = workers.wait_for_connections_until(graceful_deadline);
    if !graceful {
        let closed = workers.cancel_connections();
        tracing::warn!(
            outstanding_connections = workers.connection_count(),
            closed_response_sockets = closed,
            "graceful drain expired; cancelling accepted work"
        );
    }

    // Connection-owned clones of `served` are gone after graceful completion or cancellation; dropping
    // main's owner closes the queue, and the engine thread ends once its work is done.
    drop(served);

    if graceful {
        graceful = workers.wait_for_all_until(graceful_deadline);
        if !graceful {
            let _ = workers.cancel_connections();
            tracing::warn!(
                outstanding_workers = workers.worker_count(),
                "graceful drain expired while stopping server workers"
            );
        }
    }

    if !graceful {
        let settle_deadline = deadline_after(Instant::now(), CANCELLATION_SETTLE_TIMEOUT);
        let settled = workers.wait_for_all_until(settle_deadline);
        if !settled {
            tracing::error!(
                outstanding_workers = workers.worker_count(),
                "server cancellation settle timeout expired; exiting without waiting further"
            );
        }
    }
    tracing::info!(graceful, "server stopped");
    Ok(())
}
