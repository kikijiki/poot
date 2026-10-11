//! The typed serve configuration (POOT-590): every setting the process reads, parsed strictly once.
//!
//! The command line is the only input. There is no environment switch for any mode or limit: a flag that
//! is absent takes its documented default, and a value that does not parse is a startup error, never a
//! silent fallback to the default. The one environment read left in the binary, `POOT_MODELS_DIR`, only
//! locates the default model directory (`startup::model_dir_or_env`); it selects no mode.

use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use poot_executor::DetailedTiming;
use poot_llm::ChatTemplateOverride;
use poot_llm::driver::{BackendChoice, LoadOptions};

use crate::admin::{AdminKey, LoraAdminPolicy};
use crate::lifecycle::ServerDrainTimeout;

/// Connections served at once when `--max-connections` is not given.
const DEFAULT_MAX_CONNECTIONS: usize = 4096;

/// The most steps `--timing-max-steps` accepts. The engine thread copies the retained detail each time it
/// publishes (at most every 250 ms); these maxima keep that copy to a few milliseconds
/// (`tests::telemetry::publishing_the_largest_accepted_detail_stays_cheap`).
const MAX_TIMING_STEPS: usize = 1 << 14;
/// The most per-dispatch records `--timing-max-dispatch-records` accepts.
const MAX_TIMING_DISPATCH_RECORDS: usize = 1 << 18;
/// The most timestamp queries `--timing-max-in-flight-queries` accepts.
const MAX_TIMING_QUERIES: usize = 4096;
/// The largest sampling stride `--timing-sample-every` accepts.
const MAX_TIMING_SAMPLE_EVERY: u32 = 1 << 20;

/// How much device timing the server asks of its executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TelemetryConfig {
    /// Cumulative counters only: no device timestamp is requested, no per-dispatch detail is retained
    /// and no timestamp query is allocated or waited on. The default.
    CountersOnly,
    /// Bounded per-dispatch device timing. The bounds are finite and validated at parse time; the run
    /// perturbs submission grouping and is diagnostic, never throughput evidence.
    Detailed(DetailedTiming),
}

/// Prompt-lookup speculation: propose the `k` tokens that followed the latest earlier occurrence of the
/// last `ngram` tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptLookup {
    pub(crate) ngram: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
}

/// Speculation with a second model that proposes `k` tokens a round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DraftModel {
    pub(crate) path: String,
    pub(crate) k: NonZeroUsize,
}

/// Where the served model runs: the one default device of the backend, or a group of GPUs the operator
/// names by ordinal, in rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlacementSpec {
    /// The backend's own default device.
    Single,
    /// One pipeline stage per GPU, in this order: a layer cut balanced over their memory.
    Pipeline { ordinals: Vec<usize> },
    /// Every GPU is one rank of a tensor-parallel group.
    Tensor { ordinals: Vec<usize> },
}

/// The idle bound on reading a request, or none (`--read-timeout-secs 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadTimeout(pub(crate) Option<Duration>);

/// The command line, parsed into typed values. Every field's parser is strict.
#[derive(Debug, Parser)]
#[command(
    name = "poot-serve",
    about = "An OpenAI-compatible server over the poot driver"
)]
pub(crate) struct ServeConfig {
    /// Model directory or GGUF file; default: `qwen2.5-0.5b` under `POOT_MODELS_DIR`.
    pub(crate) model: Option<String>,
    /// Listen address.
    #[arg(default_value = "127.0.0.1:8080")]
    pub(crate) addr: String,
    /// The device backend: wgpu, rocm, ptx, vulkan, or auto (PTX if a CUDA device opens, else ROCm if an
    /// HSA GPU agent opens, else wgpu). A model the chosen backend refuses fails startup; no other
    /// backend is tried.
    #[arg(long, default_value = "auto", value_parser = parse_backend)]
    pub(crate) backend: BackendChoice,
    /// Where the model runs: `single` (the backend's default device), `pipeline:ORD,ORD,...` (one layer
    /// stage per GPU ordinal, in order) or `tensor:ORD,ORD,...` (one tensor-parallel rank per ordinal).
    /// A group needs at least two distinct ordinals and the ptx backend; a backend that cannot group
    /// fails startup.
    #[arg(long, default_value = "single", value_name = "PLACEMENT", value_parser = parse_placement)]
    pub(crate) placement: PlacementSpec,
    /// Speculate by prompt lookup: propose the K tokens that followed the latest earlier occurrence of
    /// the sequence's last NGRAM tokens, written `NGRAM:K` (split at the first colon). Exclusive with
    /// `--draft-model`.
    #[arg(
        long,
        value_name = "NGRAM:K",
        value_parser = parse_draft_lookup,
        conflicts_with_all = ["draft_model", "draft_mtp"]
    )]
    pub(crate) draft_lookup: Option<PromptLookup>,
    /// Speculate with a second model that proposes K tokens a round, written `PATH:K` (split at the last
    /// colon, so the path may contain colons). Its vocabulary must agree with the served model's.
    #[arg(
        long,
        value_name = "PATH:K",
        value_parser = parse_draft_model,
        conflicts_with = "draft_mtp"
    )]
    pub(crate) draft_model: Option<DraftModel>,
    /// Speculate with the served model's own multi-token-prediction layer, K tokens a round. There is
    /// no default K (1 is the usual start; 2 to 3 are worth measuring); a model with no such layer, or
    /// one whose layer carries recurrent state, is refused at startup. Exclusive with `--draft-lookup`
    /// and `--draft-model`, and with adapters, beam search and a device group.
    #[arg(long, value_name = "K", value_parser = parse_draft_mtp)]
    pub(crate) draft_mtp: Option<NonZeroUsize>,
    /// `--expert-pool ROWS` pages every routed expert table through a pool of ROWS resident rows per
    /// layer, instead of holding every expert on the device. At least the experts one token selects
    /// and below the model's expert count; a model with no routed experts is refused at startup.
    #[arg(long, value_name = "ROWS", value_parser = parse_expert_pool)]
    pub(crate) expert_pool: Option<NonZeroUsize>,
    /// Render chats through this jinja template instead of the model's own. A value with no jinja
    /// syntax names a built-in template, which poot lacks, and is refused.
    #[arg(long, value_name = "JINJA", conflicts_with = "chat_template_file")]
    pub(crate) chat_template: Option<String>,
    /// Render chats through the jinja template in this file instead of the model's own.
    #[arg(long, value_name = "FILE")]
    pub(crate) chat_template_file: Option<PathBuf>,
    /// Requests per minute allowed per client address on the model routes; 0 disables the limit.
    #[arg(long, value_name = "N", default_value_t = 0, value_parser = parse_count)]
    pub(crate) rate_limit_rpm: usize,
    /// Most simultaneous connections; 0 means unlimited.
    #[arg(
        long,
        value_name = "N",
        default_value_t = DEFAULT_MAX_CONNECTIONS,
        value_parser = parse_count
    )]
    pub(crate) max_connections: usize,
    /// Idle seconds a request read may stall before the connection closes; 0 disables the bound.
    #[arg(
        long,
        value_name = "SECS",
        default_value = "30",
        value_parser = parse_read_timeout
    )]
    pub(crate) read_timeout_secs: ReadTimeout,
    /// Idle seconds a response write may block before the connection closes; at least 1.
    #[arg(
        long,
        value_name = "SECS",
        default_value = "30",
        value_parser = parse_write_timeout
    )]
    pub(crate) write_timeout_secs: Duration,
    /// Seconds in-flight requests get to finish after a shutdown signal; 1 to 86400.
    #[arg(
        long,
        value_name = "SECS",
        default_value = "30",
        value_parser = parse_drain_timeout
    )]
    pub(crate) drain_timeout_secs: ServerDrainTimeout,
    /// File holding the bearer key every `/v1/lora_adapters` request must present. The key is one
    /// RFC 6750 token on its own line; a file keeps it out of `ps`. Without it, administration is open
    /// on a loopback listener and disabled on any other.
    #[arg(long, value_name = "FILE", value_parser = AdminKey::from_file)]
    pub(crate) lora_admin_key_file: Option<AdminKey>,
    /// Request device-side timing from the executor and retain bounded per-dispatch detail. Diagnostic:
    /// it perturbs submission grouping. Off by default, when the server keeps cumulative counters only.
    #[arg(long)]
    pub(crate) device_timing: bool,
    /// With `--device-timing`: most timing steps retained (default 4096).
    #[arg(long, value_name = "N", requires = "device_timing", value_parser = parse_timing_steps)]
    pub(crate) timing_max_steps: Option<usize>,
    /// With `--device-timing`: most per-dispatch records retained across those steps (default 65536).
    #[arg(
        long,
        value_name = "N",
        requires = "device_timing",
        value_parser = parse_timing_dispatch_records
    )]
    pub(crate) timing_max_dispatch_records: Option<usize>,
    /// With `--device-timing`: most timestamp queries in flight at once (default 64).
    #[arg(
        long,
        value_name = "N",
        requires = "device_timing",
        value_parser = parse_timing_queries
    )]
    pub(crate) timing_max_in_flight_queries: Option<usize>,
    /// With `--device-timing`: collect detail for one of every N steps (default 1, every step).
    #[arg(
        long,
        value_name = "N",
        requires = "device_timing",
        value_parser = parse_timing_sample_every
    )]
    pub(crate) timing_sample_every: Option<u32>,
}

impl ServeConfig {
    /// The load options the flags name: the chat template override, when given.
    pub(crate) fn load_options(&self) -> LoadOptions {
        LoadOptions {
            chat_template: ChatTemplateOverride::from_flags(
                self.chat_template.clone(),
                self.chat_template_file.clone(),
            ),
            ..LoadOptions::default()
        }
    }

    /// The LoRA administration policy for a listener bound to `listener_ip`: the key `--lora-admin-key-file`
    /// named, else loopback-only access, else disabled.
    pub(crate) fn lora_admin_policy(&self, listener_ip: IpAddr) -> anyhow::Result<LoraAdminPolicy> {
        LoraAdminPolicy::from_configured_key(
            listener_ip,
            self.lora_admin_key_file.as_ref().map(AdminKey::as_str),
        )
    }

    /// The timing the executor is opened with: counters only unless `--device-timing` is given.
    pub(crate) fn telemetry(&self) -> TelemetryConfig {
        if !self.device_timing {
            return TelemetryConfig::CountersOnly;
        }
        let default = DetailedTiming::default_diagnostic();
        TelemetryConfig::Detailed(DetailedTiming {
            sample_every: self.timing_sample_every.unwrap_or(default.sample_every),
            max_retained_steps: self.timing_max_steps.unwrap_or(default.max_retained_steps),
            max_retained_dispatch_records: self
                .timing_max_dispatch_records
                .unwrap_or(default.max_retained_dispatch_records),
            max_in_flight_queries: self
                .timing_max_in_flight_queries
                .unwrap_or(default.max_in_flight_queries),
            drain_policy: default.drain_policy,
        })
    }
}

fn parse_placement(spec: &str) -> Result<PlacementSpec, String> {
    if spec == "single" {
        return Ok(PlacementSpec::Single);
    }
    let (kind, list) = spec.split_once(':').ok_or_else(|| {
        format!("{spec:?}: expected single, pipeline:ORD,ORD,... or tensor:ORD,ORD,...")
    })?;
    let mut ordinals = Vec::new();
    for ordinal in list.split(',') {
        let ordinal: usize = ordinal
            .trim()
            .parse()
            .map_err(|_| format!("the GPU ordinal {ordinal:?} is not a whole number"))?;
        if ordinals.contains(&ordinal) {
            return Err(format!("GPU {ordinal} is named twice"));
        }
        ordinals.push(ordinal);
    }
    if ordinals.len() < 2 {
        return Err(format!(
            "a {kind} placement needs at least two GPUs, got {}",
            ordinals.len()
        ));
    }
    match kind {
        "pipeline" => Ok(PlacementSpec::Pipeline { ordinals }),
        "tensor" => Ok(PlacementSpec::Tensor { ordinals }),
        other => Err(format!(
            "unknown placement {other:?}: expected single, pipeline or tensor"
        )),
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

/// A non-negative whole number. A sign, a fraction, text and an overflow each get their own message, so a
/// bad value is never read as some other number.
fn whole_number(raw: &str) -> Result<u64, String> {
    let value = raw.trim();
    if value.starts_with('-') {
        return Err(format!("must be a non-negative whole number, got {raw:?}"));
    }
    value.parse::<u64>().map_err(|error| match error.kind() {
        std::num::IntErrorKind::PosOverflow => format!("is too large, got {raw:?}"),
        _ => format!("must be a whole number, got {raw:?}"),
    })
}

fn parse_count(raw: &str) -> Result<usize, String> {
    let value = whole_number(raw)?;
    usize::try_from(value).map_err(|_| format!("is too large, got {raw:?}"))
}

fn parse_read_timeout(raw: &str) -> Result<ReadTimeout, String> {
    let secs = whole_number(raw)?;
    Ok(ReadTimeout((secs > 0).then(|| Duration::from_secs(secs))))
}

fn parse_write_timeout(raw: &str) -> Result<Duration, String> {
    match whole_number(raw)? {
        0 => Err("must be greater than zero, got 0".to_string()),
        secs => Ok(Duration::from_secs(secs)),
    }
}

fn parse_drain_timeout(raw: &str) -> Result<ServerDrainTimeout, String> {
    ServerDrainTimeout::from_secs(whole_number(raw)?).map_err(|error| error.to_string())
}

/// A positive count no larger than `max`, for the timing bounds: zero would disable what the flag
/// bounds and an unbounded value would defeat the bound.
fn bounded_count(raw: &str, max: usize) -> Result<usize, String> {
    let value = parse_count(raw)?;
    match value {
        0 => Err("must be greater than zero, got 0".to_string()),
        value if value > max => Err(format!("must be at most {max}, got {value}")),
        value => Ok(value),
    }
}

fn parse_timing_steps(raw: &str) -> Result<usize, String> {
    bounded_count(raw, MAX_TIMING_STEPS)
}

fn parse_timing_dispatch_records(raw: &str) -> Result<usize, String> {
    bounded_count(raw, MAX_TIMING_DISPATCH_RECORDS)
}

fn parse_timing_queries(raw: &str) -> Result<usize, String> {
    bounded_count(raw, MAX_TIMING_QUERIES)
}

fn parse_timing_sample_every(raw: &str) -> Result<u32, String> {
    let value = bounded_count(raw, MAX_TIMING_SAMPLE_EVERY as usize)?;
    Ok(u32::try_from(value).expect("bounded by u32"))
}

fn positive(text: &str, what: &str) -> Result<NonZeroUsize, String> {
    text.trim()
        .parse::<NonZeroUsize>()
        .map_err(|_| format!("{what} {text:?} is not a positive integer"))
}

fn parse_expert_pool(raw: &str) -> Result<NonZeroUsize, String> {
    positive(raw, "the pool row count")
}

fn parse_draft_mtp(raw: &str) -> Result<NonZeroUsize, String> {
    positive(raw, "the draft length")
}

fn parse_draft_lookup(spec: &str) -> Result<PromptLookup, String> {
    let (ngram, k) = spec
        .split_once(':')
        .ok_or_else(|| format!("{spec:?}: expected NGRAM:K"))?;
    Ok(PromptLookup {
        ngram: positive(ngram, "the n-gram length")?,
        k: positive(k, "the draft length")?,
    })
}

fn parse_draft_model(spec: &str) -> Result<DraftModel, String> {
    let (path, k) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("{spec:?}: expected PATH:K"))?;
    if path.is_empty() {
        return Err(format!("{spec:?}: the model path is empty"));
    }
    Ok(DraftModel {
        path: path.to_string(),
        k: positive(k, "the draft length")?,
    })
}
