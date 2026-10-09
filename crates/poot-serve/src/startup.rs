//! Process configuration, model startup, listener orchestration, and graceful shutdown.

use std::net::TcpListener;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use poot_llm::driver::{BackendChoice, DraftPolicy, LoadOptions, ModelHandle, open_executor};
use poot_llm::encoders::Pooling;
use poot_llm::front_end::{Idefics3Checkpoint, is_idefics3_dir};
use poot_models::registry::Registry;

use crate::admin::{LoraAdminPolicy, LoraAdminPolicyMode};
use crate::batch::engine::{DriverEngine, run_engine_thread};
use crate::engine_plan::{
    EngineArgs, EngineHealth, EnginePlan, EngineState, Started, StartupError, device_budget,
};
use crate::http::{
    ConnGuard, RateLimiter, configure_response_writes, handle, max_connections,
    request_read_timeout, response_write_timeout,
};
use crate::lifecycle::{
    CANCELLATION_SETTLE_TIMEOUT, Card327AcceptBarrier, WorkerRegistry, deadline_after,
    reset_drain_cancellation, server_drain_timeout,
};
use crate::metrics::{CacheMetrics, Histogram, Metrics, SpeculativeMetrics};
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

/// The command line. The LoRA flags of the earlier serve modes stay refused (clap rejects an unknown
/// option) until the typed serve configuration (POOT-590) replaces this parser.
#[derive(Debug, Parser)]
#[command(
    name = "poot-serve",
    about = "An OpenAI-compatible server over the poot driver"
)]
struct Cli {
    /// Model directory or GGUF file; default: `qwen2.5-0.5b` under `POOT_MODELS_DIR`.
    model: Option<String>,
    /// Listen address.
    #[arg(default_value = "127.0.0.1:8080")]
    addr: String,
    /// The device backend: wgpu, rocm, ptx, vulkan, or auto (PTX if a CUDA device opens, else ROCm if an
    /// HSA GPU agent opens, else wgpu). A model the chosen backend refuses fails startup; no other
    /// backend is tried.
    #[arg(long, default_value = "auto", value_parser = parse_backend)]
    backend: BackendChoice,
    /// Speculate by prompt lookup: propose the K tokens that followed the latest earlier occurrence of
    /// the sequence's last NGRAM tokens, written `NGRAM:K` (split at the first colon). Exclusive with
    /// `--draft-model`; `draft_policy` refuses both.
    #[arg(long, value_name = "NGRAM:K")]
    draft_lookup: Option<String>,
    /// Speculate with a second model that proposes K tokens a round, written `PATH:K` (split at the last
    /// colon, so the path may contain colons). Its vocabulary must agree with the served model's.
    #[arg(long, value_name = "PATH:K")]
    draft_model: Option<String>,
}

/// The draft policy the flags name: `Off` with neither flag. A `--draft-model` checkpoint is loaded
/// here, so a path that does not load is a startup error before the listener binds.
fn draft_policy(lookup: Option<&str>, model: Option<&str>) -> Result<DraftPolicy> {
    fn count(text: &str, what: &str) -> Result<NonZeroUsize> {
        text.parse::<NonZeroUsize>()
            .with_context(|| format!("{what} {text:?} is not a positive integer"))
    }
    match (lookup, model) {
        (None, None) => Ok(DraftPolicy::Off),
        (Some(spec), None) => {
            let (ngram, k) = spec
                .split_once(':')
                .with_context(|| format!("--draft-lookup {spec:?}: expected NGRAM:K"))?;
            Ok(DraftPolicy::PromptLookup {
                ngram: count(ngram, "the n-gram length")?,
                k: count(k, "the draft length")?,
            })
        }
        (None, Some(spec)) => {
            let (path, k) = spec
                .rsplit_once(':')
                .with_context(|| format!("--draft-model {spec:?}: expected PATH:K"))?;
            Ok(DraftPolicy::DraftModel {
                k: count(k, "the draft length")?,
                handle: load_model(path).context("load the draft model")?,
            })
        }
        (Some(_), Some(_)) => anyhow::bail!("--draft-lookup and --draft-model are exclusive"),
    }
}

fn parse_backend(name: &str) -> Result<BackendChoice, String> {
    match name {
        "wgpu" => Ok(BackendChoice::Wgpu),
        "rocm" => Ok(BackendChoice::Rocm),
        "ptx" => Ok(BackendChoice::Ptx),
        "vulkan" => Ok(BackendChoice::Vulkan),
        "auto" => Ok(BackendChoice::Auto),
        other => Err(format!(
            "unknown backend {other:?}: expected wgpu, rocm, ptx, vulkan or auto"
        )),
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
fn load_model(model_dir: &str) -> Result<Arc<ModelHandle>> {
    tracing::info!(model_dir, "loading model");
    let model = Arc::new(
        ModelHandle::load(
            std::path::Path::new(model_dir),
            &Registry::builtin().context("build the family registry")?,
            &LoadOptions::default(),
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

/// Load the served model. A SmolVLM (idefics3) checkpoint is served by its text tower, with its image
/// front end beside it; any other checkpoint is the model itself.
fn load_served(model_dir: &str) -> Result<(Arc<ModelHandle>, Option<Arc<Idefics3Checkpoint>>)> {
    let path = std::path::Path::new(model_dir);
    if !is_idefics3_dir(path) {
        return Ok((load_model(model_dir)?, None));
    }
    tracing::info!(model_dir, "loading vision-language model");
    let checkpoint = Idefics3Checkpoint::read(
        path,
        &Registry::builtin().context("build the family registry")?,
    )
    .context("load vision-language model")?;
    Ok((Arc::clone(checkpoint.text()), Some(Arc::new(checkpoint))))
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
    model_name: String,
    health: &EngineHealth,
    jobs: mpsc::Receiver<EngineMsg>,
) -> Result<(Arc<Metrics>, EngineView)> {
    type Startup = Result<(Arc<Metrics>, EngineView), StartupError>;
    let (started_tx, started_rx) = mpsc::channel::<Startup>();
    let health = health.clone();
    workers
        .spawn("poot-engine", move || {
            let started = EnginePlan::build(&args, &model, open_executor, device_budget, &health);
            let Started { plan, driver } = match started {
                Ok(started) => started,
                Err(error) => {
                    let _ = started_tx.send(Err(error));
                    return;
                }
            };
            tracing::info!(
                backend = plan.backend_label(),
                device = ?plan.device,
                model = plan.model,
                rows = plan.rows(),
                workload = ?plan.workload,
                planned_bytes = ?plan.budget.planned_bytes,
                discovered_bytes = ?plan.budget.discovered_bytes,
                available_bytes = ?plan.budget.available_bytes,
                "engine plan"
            );
            let metrics = Arc::new(new_metrics(model_name, &plan, &driver.live_role_bytes()));
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
    driver_bytes: &[(poot_executor::BufferRole, u64)],
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
            gpu_weight_bytes: AtomicU64::new(live(poot_executor::BufferRole::Weight)),
            gpu_kv_bytes: AtomicU64::new(live(poot_executor::BufferRole::State)),
            kv_total_blocks: AtomicU64::new(plan.caps.kv_blocks as u64),
            gpu_vram_total_bytes: AtomicU64::new(plan.budget.available_bytes.unwrap_or(0)),
            ..CacheMetrics::default()
        },
        speculative: SpeculativeMetrics::default(),
    }
}

pub(crate) fn run() -> Result<()> {
    reset_drain_cancellation();
    SHUTDOWN.store(false, Ordering::SeqCst);
    init_logging();

    // Validate transport config before model detection or any loader call, so a bad value fails fast.
    let write_timeout = response_write_timeout()?;
    install_signal_handlers();

    let cli = Cli::parse();
    let model_dir = model_dir_or_env(cli.model.as_ref())?;
    let addr = cli.addr;
    let model_name = served_model_name(&model_dir);

    // The engine plan decides the backend, fits the workload to the device and prepares it. A refused
    // model or an unopenable device is a typed startup error and the listener never binds. The prepared
    // driver moves to the engine thread, which runs the one scheduling loop over it: generation,
    // embeddings, rerank and LoRA administration are messages on its one queue.
    let health = EngineHealth::starting();
    let (model, checkpoint) = load_served(&model_dir)?;
    let mut workers = WorkerRegistry::new();
    let (job_tx, job_rx) = mpsc::channel::<EngineMsg>();
    let draft = draft_policy(cli.draft_lookup.as_deref(), cli.draft_model.as_deref())?;
    // The vision tower opens first, so the engine fits its workload to the memory the tower left.
    let images = checkpoint
        .map(|checkpoint| spawn_image_front_end(&mut workers, checkpoint, cli.backend))
        .transpose()?;
    let args = EngineArgs::new(cli.backend, draft);
    let (metrics, view) = spawn_engine(
        &mut workers,
        Arc::clone(&model),
        if images.is_some() {
            args.with_embeds()
        } else {
            args
        },
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
    });

    let limiter = Arc::new(RateLimiter::from_env());
    if limiter.rpm > 0 {
        tracing::info!(
            rpm = limiter.rpm,
            "per-key rate limit active (POOT_RATE_LIMIT_RPM)"
        );
    }

    let listener = TcpListener::bind(&addr).with_context(|| format!("bind {addr}"))?;
    let lora_admin_policy = Arc::new(LoraAdminPolicy::from_env(listener.local_addr()?.ip())?);
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
    let mut card327_accept_barrier = Card327AcceptBarrier::from_env()?;
    let read_timeout = request_read_timeout();
    let drain_timeout = server_drain_timeout();
    let max_conns = max_connections();
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
                card327_accept_barrier.after_accept()?;
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
                // Bound slowloris: a stalled client's initial read times out instead of blocking the thread forever.
                let _ = s.set_read_timeout(read_timeout);
                // Bound every response write; if the timeout cannot be installed, close the connection.
                if let Err(e) = configure_response_writes(&s, write_timeout) {
                    tracing::error!(error = %e, "configure connection write timeout");
                    continue;
                }
                let served = Arc::clone(&served);
                let health = health.clone();
                let metrics = Arc::clone(&metrics);
                let limiter = Arc::clone(&limiter);
                let lora_admin_policy = Arc::clone(&lora_admin_policy);
                live_conns.fetch_add(1, Ordering::SeqCst);
                let conn_counter = Arc::clone(&live_conns);
                // One thread per connection: parse and shape concurrently.
                if let Err(e) = workers.spawn_connection("poot-conn", s, move |s| {
                    let _guard = ConnGuard(conn_counter);
                    if let Err(e) =
                        handle(s, &served, &health, &metrics, &limiter, &lora_admin_policy)
                    {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every backend name the flag documents parses to its choice, and nothing else does.
    #[test]
    fn parse_backend_accepts_each_documented_name_and_rejects_others() {
        for (name, choice) in [
            ("wgpu", BackendChoice::Wgpu),
            ("rocm", BackendChoice::Rocm),
            ("ptx", BackendChoice::Ptx),
            ("vulkan", BackendChoice::Vulkan),
            ("auto", BackendChoice::Auto),
        ] {
            assert_eq!(parse_backend(name), Ok(choice));
        }
        assert!(parse_backend("cuda").is_err());
        assert!(parse_backend("Rocm").is_err());
    }

    /// The flags parse into the policy they name; a malformed or zero value is a startup error.
    #[test]
    fn draft_policy_parses_lookup_and_refuses_malformed_values() {
        assert!(matches!(draft_policy(None, None), Ok(DraftPolicy::Off)));
        match draft_policy(Some("3:4"), None).unwrap() {
            DraftPolicy::PromptLookup { ngram, k } => {
                assert_eq!((ngram.get(), k.get()), (3, 4));
            }
            other => panic!("expected prompt lookup, got {other:?}"),
        }
        for bad in ["3", "3:0", "0:4", "x:4", "3:"] {
            assert!(
                draft_policy(Some(bad), None).is_err(),
                "{bad:?} must refuse"
            );
        }
        assert!(draft_policy(Some("3:4"), Some("m:2")).is_err());
        assert!(draft_policy(None, Some("no-colon")).is_err());
    }
}
