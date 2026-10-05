//! Process configuration, model startup, listener orchestration, and graceful shutdown.

use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use poot_llm::Runner;
use poot_llm::encoder::{BertKind, CrossEncoderRunner, EncoderRunner, bert_kind};

use crate::admin::{LoraAdminPolicy, LoraAdminPolicyMode};
use crate::http::{
    ConnGuard, RateLimiter, configure_response_writes, handle, max_connections,
    request_read_timeout, response_write_timeout,
};
use crate::lifecycle::{
    CANCELLATION_SETTLE_TIMEOUT, Card327AcceptBarrier, WorkerRegistry, deadline_after,
    reset_drain_cancellation, server_drain_timeout,
};
use crate::metrics::{CacheMetrics, Histogram, Metrics, SpeculativeMetrics};
use crate::types::{Backend, SHUTDOWN};

/// The options `poot-serve` takes; each is followed by a value.
const KNOWN_FLAGS: [&str; 2] = ["--lora-adapter", "--lora-pool-capacity"];

/// No model path was given and `POOT_MODELS_DIR` is unset or empty.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model path given and POOT_MODELS_DIR is not set: pass a model path or set POOT_MODELS_DIR to the directory that holds `{DEFAULT_MODEL}`"
)]
pub(crate) struct ModelDirUnset;

/// The model directory under `POOT_MODELS_DIR` used when no path is given.
const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

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

/// Parse one `--lora-adapter NAME=DIR` flag value into `(name, dir)`. `NAME` is the pool name a request's
/// `lora_adapter` field selects; `DIR` is a PEFT adapter directory. Splits on the first `=` only.
pub(crate) fn parse_lora_adapter_flag(raw: &str) -> Result<(String, String)> {
    let (name, dir) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("--lora-adapter {raw:?}: expected NAME=DIR"))?;
    if name.is_empty() || dir.is_empty() {
        bail!("--lora-adapter {raw:?}: NAME and DIR must both be non-empty");
    }
    Ok((name.to_string(), dir.to_string()))
}

/// Parse `--lora-pool-capacity N`: the total number of pool slots to provision at startup, including the
/// `--lora-adapter` flags. Extra slots are registered as empty placeholders so a later
/// `Runner::hot_load_lora_adapter*` can take one over without re-tracing the serving graph (its stacked-const
/// slot count is fixed at trace time). `None` means no headroom: hot-load can replace an existing adapter's
/// weights but cannot add a new name. `N` must be at least the number of `--lora-adapter` flags.
pub(crate) fn parse_lora_pool_capacity_flag(all_args: &[String]) -> Result<Option<usize>> {
    let Some(raw) = all_args.windows(2).find_map(|w| {
        if w[0] == "--lora-pool-capacity" {
            Some(w[1].clone())
        } else {
            None
        }
    }) else {
        return Ok(None);
    };
    let n: usize = raw
        .parse()
        .map_err(|_| anyhow!("--lora-pool-capacity {raw:?}: expected a non-negative integer"))?;
    Ok(Some(n))
}

pub(crate) fn run() -> Result<()> {
    reset_drain_cancellation();
    SHUTDOWN.store(false, Ordering::SeqCst);
    // RUST_LOG controls verbosity (default "info"). ANSI colors only when stdout is a terminal, so piped
    // logs (supervisors, containers, test harnesses) stay greppable.
    tracing_subscriber::fmt()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Validate transport config before model detection or any loader call, so a bad value fails fast.
    let write_timeout = response_write_timeout()?;

    // SIGINT/SIGTERM set SHUTDOWN so the accept loop exits.
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

    let all_args: Vec<String> = std::env::args().collect();
    // The only options are the LoRA pool flags. A flag of a removed serve mode (`--backend`,
    // `--tensor-parallel`, `--profile`) is refused by name: a valued one would otherwise leave its
    // value behind as the model directory.
    if let Some(flag) = all_args
        .iter()
        .skip(1)
        .find(|a| a.starts_with("--") && !KNOWN_FLAGS.contains(&a.as_str()))
    {
        bail!(
            "unknown option {flag}: the options are {}",
            KNOWN_FLAGS.join(", ")
        );
    }
    // Repeatable `--lora-adapter NAME=DIR` flags, validated before any model loads.
    let lora_adapter_flags: Vec<(String, String)> = all_args
        .windows(2)
        .filter(|w| w[0] == "--lora-adapter")
        .map(|w| parse_lora_adapter_flag(&w[1]))
        .collect::<Result<Vec<_>>>()?;
    // Optional reserved pool capacity for later `hot_load_lora_adapter*` calls.
    let lora_pool_capacity_flag = parse_lora_pool_capacity_flag(&all_args)?;
    if let Some(cap) = lora_pool_capacity_flag
        && cap < lora_adapter_flags.len()
    {
        bail!(
            "--lora-pool-capacity {cap} is smaller than the {} --lora-adapter flag(s) given - it \
             must be at least that many",
            lora_adapter_flags.len()
        );
    }
    // Positional args (model dir, then addr) plus flags anywhere. Valued flags consume the following argv
    // entry, which must be skipped or it leaks into `positional`.
    let positional: Vec<String> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in all_args.into_iter().skip(1) {
            if skip {
                skip = false;
                continue;
            }
            if a.starts_with("--") {
                skip = true;
                continue;
            }
            out.push(a);
        }
        out
    };

    let model_dir = model_dir_or_env(positional.first())?;
    let addr = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());

    // Served model name: the model-dir basename, or the GGUF file stem; echoed in responses and /v1/models.
    let model_name = std::path::Path::new(&model_dir)
        .file_stem()
        .filter(|_| model_dir.ends_with(".gguf"))
        .or_else(|| std::path::Path::new(&model_dir).file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "poot-model".to_string());

    // Auto-detect the model kind from config.json: BERT encoder and cross-encoder on CPU eval; anything
    // else loads as a causal decoder `Runner`. No generation engine is started for a decoder: the
    // generation routes refuse with a typed 503 until the one serving loop over the driver lands
    // (POOT-753). A registered family runs only on the driver, so `Runner::load*` refuses it with the
    // typed `RunnerError::RegisteredFamily` and the server does not start.
    let mut gpu_weight_bytes_at_load: u64 = 0;
    let backend: Backend = match bert_kind(&model_dir) {
        Some(BertKind::CrossEncoder) => {
            tracing::info!(
                model_dir,
                "loading cross-encoder reranker (BertForSequenceClassification)"
            );
            let ce = CrossEncoderRunner::load(&model_dir).context("load cross-encoder model")?;
            tracing::info!(
                "cross-encoder loaded (/v1/rerank only; embeddings + generation return 400)"
            );
            Backend::CrossEncoder(Arc::new(ce))
        }
        Some(BertKind::Embedding) => {
            tracing::info!(model_dir, "loading encoder embedding model (BERT-class)");
            let enc = EncoderRunner::load(&model_dir).context("load encoder model")?;
            tracing::info!(
                dim = enc.dim(),
                "encoder model loaded (embeddings + rerank only; generation endpoints return 400)"
            );
            Backend::Encoder(Arc::new(enc))
        }
        None => {
            // A `.gguf` path loads a self-contained GGUF (config, quantized weights, tokenizer); otherwise the path is
            // a safetensors model directory.
            let mut runner = if model_dir.ends_with(".gguf") {
                tracing::info!(model_dir, "loading model (GGUF)");
                Runner::load_gguf(&model_dir).context("load gguf model")?
            } else {
                tracing::info!(model_dir, "loading model (safetensors)");
                Runner::load(&model_dir).context("load model")?
            };
            tracing::info!(
                hidden = runner.config().hidden,
                layers = runner.config().layers,
                vocab = runner.config().vocab,
                "model config"
            );
            // Register every `--lora-adapter NAME=DIR` flag into the Runner's multi-adapter pool, then rebind once: the
            // tracer's stacked constants need real values in `Runner::weights` before `batch_engine_loop` traces the
            // LoRA-batched graph (see `Runner::rebind_lora_pool`). `register_lora_adapter_dir` enforces the q_proj/v_proj
            // and architecture gate, so a bad adapter fails at startup.
            for (name, dir) in &lora_adapter_flags {
                runner
                    .register_lora_adapter_dir(name, dir)
                    .with_context(|| format!("register --lora-adapter {name}={dir}"))?;
                tracing::info!(name, dir, "registered lora adapter");
            }
            // Reserve empty (all-zero, target-nothing) placeholder slots up to `--lora-pool-capacity` via
            // `register_lora_adapter`. They occupy real pool slots from the start, so the batched-pool graph traces with
            // room to spare and a later `Runner::hot_load_lora_adapter*` can take one over without a re-trace.
            if let Some(cap) = lora_pool_capacity_flag {
                for i in lora_adapter_flags.len()..cap {
                    let placeholder = poot_load::lora::LoraAdapter {
                        config: poot_load::lora::LoraAdapterConfig {
                            r: 0,
                            lora_alpha: 0.0,
                            target_modules: Vec::new(),
                            use_rslora: false,
                        },
                        weights: HashMap::new(),
                    };
                    runner
                        .register_lora_adapter(format!("__reserved_{i}"), placeholder)
                        .context("register reserved lora pool placeholder slot")?;
                }
                tracing::info!(
                    capacity = cap,
                    startup_adapters = lora_adapter_flags.len(),
                    "reserved lora pool capacity for future hot-loads"
                );
            }
            if !lora_adapter_flags.is_empty() {
                runner
                    .rebind_lora_pool()
                    .context("rebind lora adapter pool")?;
            }
            tracing::info!(
                "decoder loaded (no generation engine: /v1/completions and /v1/chat/completions \
                 return 503; embeddings, rerank and LoRA administration are served)"
            );
            gpu_weight_bytes_at_load = runner.weight_bytes() as u64;
            Backend::Decoder(Arc::new(runner))
        }
    };

    let metrics = Arc::new(Metrics {
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
        // No engine: no decode slots and no device.
        slot_capacity: 0,
        on_gpu: false,
        profile: false,
        model: model_name,
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics {
            gpu_weight_bytes: AtomicU64::new(gpu_weight_bytes_at_load),
            ..CacheMetrics::default()
        },
        speculative: SpeculativeMetrics::default(),
    });

    let backend = Arc::new(backend);
    let mut workers = WorkerRegistry::new();

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
                let backend = Arc::clone(&backend);
                let metrics = Arc::clone(&metrics);
                let limiter = Arc::clone(&limiter);
                let lora_admin_policy = Arc::clone(&lora_admin_policy);
                live_conns.fetch_add(1, Ordering::SeqCst);
                let conn_counter = Arc::clone(&live_conns);
                // One thread per connection: parse and shape concurrently.
                if let Err(e) = workers.spawn_connection("poot-conn", s, move |s| {
                    let _guard = ConnGuard(conn_counter);
                    if let Err(e) =
                        handle(s, None, &backend, &metrics, &limiter, &lora_admin_policy)
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

    // Connection-owned Backend clones are gone after graceful completion or cancellation; drop main's owner.
    drop(backend);

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
