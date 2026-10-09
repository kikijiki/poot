//! poot's runner for the cross-framework benchmark suite.
//!
//! One process measures one cell and prints one JSON line (the suite runner contract) as the last line of
//! stdout; logs go to stderr. Every result is that line: the harness turns it into a stored `results.jsonl`
//! row, and the runner has no flag that reports anything else.
//!
//! Two modes, one measure loop each, both written once over a backend:
//!
//! - `--mode single`: `--warmup` untimed generations, then `--iters` timed ones over a fixed
//!   `(prompt, gen-tokens)` scenario. Per iteration: TTFT is the time to the first generated token, e2e the
//!   whole generation, TPOT is the mean inter-token latency; finish is completion minus last token.
//! - `--mode decode-curve`: for each `--isl-list` length, a synthesized context of exactly that size is
//!   filled in one batched prefill and `--osl` tokens are decoded with EOS ignored, timing every token. The
//!   raw per-iteration samples (ttft, e2e, inter-token gaps) go out for the harness to aggregate.
//!
//! `--backend ptx|rocm|wgpu` picks the executor the driver opens; the measure loop never branches on it.
//! `--gguf <path>` (or a `.gguf` `--model-dir`) loads the quant-resident weights; that decode exists on
//! wgpu and rocm only.
//!
//! A checkpoint of a family `Registry::builtin()` holds loads as a `driver::ModelHandle` and runs on the
//! one `poot_llm::driver::Driver` loop; a family it does not hold is a recorded refusal. The result
//! line records the engine. The driver prefills a decode-curve context in the pieces of the chunk
//! plan and a single-mode prompt in one forward under `--prefill`, token by token without it.
//!
//! Only `--precision f32|bf16|f16` are accepted: the engine loads bf16/f16 weights and computes in f32.
//!
//! When the runner refuses a request (see `refusal`: an architecture the loader does not serve, or a
//! precision it cannot load), the line reports
//! `"status":"unsupported"` with the refusal's own text as `reason`; any other failure exits non-zero.
//! Built with a plain `cargo build --release -p poot-bench-runner` (kernels are generated at runtime).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::{json, Value};

use poot_graph_plan::{CompileOptions, Submission};
use poot_llm::driver::error::DriverError;
#[cfg(test)]
use poot_llm::driver::error::Unsupported;
use poot_llm::driver::{
    open_executor, open_timed_executor, BackendChoice, Driver, DriverOptions, DriverStats,
    GenerateRequest, Head, Layout, ModelHandle, PreparedSetLimits, ServingShapes, Warm,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::Registry;
use poot_quant::weights::WeightEntry;

/// The git sha this binary was built from (`build.rs`); the harness records it with every result.
const BUILD_SHA: &str = env!("POOT_BUILD_SHA");

/// The context every decode-curve point cycles to reach its exact length.
const CURVE_BASE_PHRASE: &str = "The quick brown fox jumps over the lazy dog. ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Ptx,
    Rocm,
    Wgpu,
}

impl Backend {
    const ALL: [Backend; 3] = [Backend::Ptx, Backend::Rocm, Backend::Wgpu];

    fn label(self) -> &'static str {
        match self {
            Backend::Ptx => "ptx",
            Backend::Rocm => "rocm",
            Backend::Wgpu => "wgpu",
        }
    }

    fn parse(name: &str) -> anyhow::Result<Self> {
        Self::ALL
            .into_iter()
            .find(|backend| backend.label() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown backend {name:?} (supported: ptx, rocm, wgpu)"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Single,
    DecodeCurve,
}

impl Mode {
    const ALL: [Mode; 2] = [Mode::Single, Mode::DecodeCurve];

    fn label(self) -> &'static str {
        match self {
            Mode::Single => "single",
            Mode::DecodeCurve => "decode-curve",
        }
    }

    fn parse(name: &str) -> anyhow::Result<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.label() == name)
            .ok_or_else(|| {
                anyhow::anyhow!("unknown mode {name:?} (supported: single, decode-curve)")
            })
    }
}

/// Which preparation state the invoker declares for this process (the receipt's label for what the
/// preparation counters must show; the harness validates the counters against it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparedState {
    /// A fresh process over an empty artifact cache: every artifact is built.
    ColdEmptyArtifacts,
    /// A fresh process over a populated artifact cache: artifacts are served from it.
    ColdPopulatedArtifacts,
    /// A driver already prepared by a previous pass of the same declared set: preparing again does nothing.
    ReusedPreparedDriver,
}

impl PreparedState {
    const ALL: [PreparedState; 3] = [
        PreparedState::ColdEmptyArtifacts,
        PreparedState::ColdPopulatedArtifacts,
        PreparedState::ReusedPreparedDriver,
    ];

    fn label(self) -> &'static str {
        match self {
            PreparedState::ColdEmptyArtifacts => "cold-process-empty-artifacts",
            PreparedState::ColdPopulatedArtifacts => "cold-process-populated-artifacts",
            PreparedState::ReusedPreparedDriver => "reused-prepared-driver",
        }
    }

    fn parse(name: &str) -> anyhow::Result<Self> {
        Self::ALL
            .into_iter()
            .find(|state| state.label() == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown prepared state {name:?} (supported: {})",
                    Self::ALL.map(Self::label).join(", ")
                )
            })
    }

    /// Warmup passes of the declared set: a reused driver is prepared once before the pass that is counted.
    fn warmup_passes(self) -> usize {
        match self {
            PreparedState::ReusedPreparedDriver => 2,
            _ => 1,
        }
    }
}

/// What generated a result: the driver's one loop. It is recorded, never chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Driver,
}

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Engine::Driver => "driver",
        }
    }
}

enum Weights {
    /// A safetensors checkpoint directory.
    Safetensors(String),
    /// A GGUF file, decoded from its packed quant weights.
    Gguf(String),
}

struct Config {
    weights: Weights,
    prompt_file: Option<String>,
    gen_tokens: usize,
    warmup: usize,
    iters: usize,
    precision: String,
    backend: Backend,
    /// Single mode: fill the prompt's KV in one batched forward instead of one token at a time.
    prefill: bool,
    json: bool,
    profile: bool,
    mode: Mode,
    isl_list: Vec<usize>,
    osl: usize,
    prepared_state: Option<PreparedState>,
    /// Artifacts the cache directory held before this process opened anything (`None`: no directory named).
    artifact_entries_at_start: Option<u64>,
}

enum Invocation {
    Version,
    Run(Config),
}

/// Options that take a value and switches that do not. Everything else is an error: a flag the runner
/// ignores would report a measurement the caller did not ask for. `--synthetic` is part of the suite's
/// runner contract (the harness passes it to every runner); decode-curve contexts are always synthetic.
const VALUE_FLAGS: [&str; 12] = [
    "--model-dir",
    "--gguf",
    "--prompt-file",
    "--gen-tokens",
    "--warmup",
    "--iters",
    "--precision",
    "--backend",
    "--mode",
    "--isl-list",
    "--osl",
    "--prepared-state",
];
const SWITCHES: [&str; 5] = [
    "--prefill",
    "--json",
    "--profile",
    "--synthetic",
    "--version",
];

fn parse_args(args: &[String]) -> anyhow::Result<Invocation> {
    let mut values: HashMap<&str, &str> = HashMap::new();
    let mut switches: HashSet<&str> = HashSet::new();
    let mut rest = args.iter().skip(1).map(String::as_str);
    while let Some(arg) = rest.next() {
        if SWITCHES.contains(&arg) {
            switches.insert(arg);
        } else if VALUE_FLAGS.contains(&arg) {
            let value = rest
                .next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?;
            values.insert(arg, value);
        } else {
            anyhow::bail!("unknown argument {arg:?}");
        }
    }
    if switches.contains("--version") {
        return Ok(Invocation::Version);
    }

    let number = |flag: &str, default: usize| -> anyhow::Result<usize> {
        values.get(flag).map_or(Ok(default), |value| {
            value
                .parse()
                .with_context(|| format!("{flag} must be a non-negative integer; got {value:?}"))
        })
    };
    let weights = match (values.get("--gguf"), values.get("--model-dir")) {
        (Some(path), _) => Weights::Gguf((*path).to_string()),
        (None, Some(path)) if path.ends_with(".gguf") => Weights::Gguf((*path).to_string()),
        (None, Some(dir)) => Weights::Safetensors((*dir).to_string()),
        (None, None) => anyhow::bail!("--model-dir or --gguf required"),
    };
    let isl_list = values
        .get("--isl-list")
        .map(|list| {
            list.split(',')
                .map(|isl| {
                    isl.trim()
                        .parse()
                        .with_context(|| format!("--isl-list entry {isl:?} is not an integer"))
                })
                .collect::<anyhow::Result<Vec<usize>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let config = Config {
        weights,
        prompt_file: values.get("--prompt-file").map(|f| (*f).to_string()),
        gen_tokens: number("--gen-tokens", 128)?,
        warmup: number("--warmup", 2)?,
        iters: number("--iters", 5)?,
        precision: values
            .get("--precision")
            .copied()
            .unwrap_or("f32")
            .to_string(),
        backend: Backend::parse(values.get("--backend").copied().unwrap_or("ptx"))?,
        prefill: switches.contains("--prefill"),
        json: switches.contains("--json"),
        profile: switches.contains("--profile"),
        mode: Mode::parse(values.get("--mode").copied().unwrap_or("single"))?,
        isl_list,
        osl: number("--osl", 128)?,
        prepared_state: values
            .get("--prepared-state")
            .map(|name| PreparedState::parse(name))
            .transpose()?,
        artifact_entries_at_start: artifact_cache_entries(),
    };

    anyhow::ensure!(config.iters > 0, "--iters must be positive");
    match config.mode {
        Mode::Single => anyhow::ensure!(config.gen_tokens > 0, "--gen-tokens must be positive"),
        Mode::DecodeCurve => {
            anyhow::ensure!(config.osl > 0, "--osl must be positive");
            anyhow::ensure!(
                !config.isl_list.is_empty(),
                "--mode decode-curve needs a non-empty --isl-list"
            );
        }
    }
    Ok(Invocation::Run(config))
}

/// A profiled window: the rendered report and the engine-produced device time of the same window.
struct Profile {
    text: String,
    /// `{"kind": "measured", ...}` or `{"kind": "unknown"}`: see [`device_time_receipt`].
    device_time: Value,
}

/// The window's device time as the result line carries it: measured milliseconds from the engine's own
/// device timestamps, or the typed `unknown` of a device that reports none (ROCm, an untimed device) -
/// never a zero and never a host-observed substitute.
fn device_time_receipt(window: &poot_profile::WindowReport) -> Value {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let sum = window.device.sum_of_dispatch_durations.map(ms);
    let span = window.device.device_span.map(ms);
    if sum.is_none() && span.is_none() {
        return json!({ "kind": "unknown" });
    }
    json!({ "kind": "measured", "sum_of_dispatch_ms": sum, "device_span_ms": span })
}

/// A loaded model on one backend, as the measure loop drives it; the loop is written once against this
/// trait.
trait Generate {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;

    /// Fill `context`'s KV in one batched prefill, then decode up to `max_new` tokens with EOS ignored.
    /// `on_token` runs once per generated token.
    fn generate_context(
        &mut self,
        context: &[u32],
        max_new: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()>;

    /// Generate from `prompt`; `prefill` fills its KV in one batched forward instead of token by token.
    fn generate_prompt(
        &mut self,
        prompt: &str,
        max_new: usize,
        prefill: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()>;

    /// Drop what a profiled executor recorded so far (compiles and cache fills before the timed window).
    fn reset_profile(&mut self);

    fn profile_report(&self, tokens: usize, wall: Duration) -> Option<Profile>;

    /// What the engine itself counted, for the result line: the driver's prepared set, compile and
    /// retention counters. `None` for an engine with nothing of its own to report.
    fn engine_receipt(&self) -> Option<Value> {
        None
    }

    /// Executor steps taken so far (prefill pieces and decode steps alike): the counter a request's cost in
    /// steps is the difference of, so a token delivered without a step inside its request is visible.
    fn steps(&self) -> u64;

    /// What the engine counts per preparation layer; a layer it has no producer counter for is Unknown.
    fn counters(&self) -> PreparationCounters {
        PreparationCounters::default()
    }

    /// The counters of the fresh engine, before the declared set was prepared: where a cold process's
    /// preparation count starts, since the engine prepares while it opens.
    fn initial_counters(&self) -> PreparationCounters {
        PreparationCounters::default()
    }

    /// What generates.
    fn engine(&self) -> Engine;
}

/// The driver engine on one backend (Card 734): [`Driver::generate`] behind the same [`Generate`] seam, so
/// the measure loop never branches on the engine. One `Driver` serves the whole cell: its KV holds the
/// longest request the cell's mode makes, and its prefill chunk is that mode's (the whole prompt for
/// `--prefill`, one token for token by token, the longest context for decode-curve, where a shorter
/// context plans as the descending powers of two its chunk plan admits). A second `Driver` would open a
/// second executor after dropping the first, which the ROCm runtime does not survive in one process.
struct DriverCell {
    handle: Arc<ModelHandle>,
    driver: Driver,
    /// The driver's counters before `Driver::prepare` ran.
    initial: PreparationCounters,
    /// The executor's timing snapshot when `reset_profile` last ran: the start of the profiled window.
    mark: Option<poot_profile::Mark>,
}

/// The preparation layers the driver and its executor count.
fn driver_counters(driver: &Driver) -> PreparationCounters {
    counters_from(&driver.stats(), driver.executor_stats().recordings)
}

/// Each driver stat in its own layer: programs compiled are the plan, entries added are the entries, and
/// the executor's recordings are the recordings. Every other layer has no producer counter.
fn counters_from(stats: &DriverStats, recordings: u64) -> PreparationCounters {
    PreparationCounters {
        plan: Some(stats.compiles),
        entries: Some(stats.entries_added),
        recordings: Some(recordings),
        ..PreparationCounters::default()
    }
}

fn driver_options(capacity: NonZeroUsize, chunk: usize) -> anyhow::Result<DriverOptions> {
    let compile = CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD);
    Ok(DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity,
        prefill_chunk: NonZeroUsize::new(chunk).context("a prefill chunk holds a token")?,
        max_trace_tokens: capacity,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(8 << 30).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
        draft: poot_llm::driver::DraftPolicy::Off,
    })
}

impl DriverCell {
    fn run(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        ignore_eos: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        let request = GenerateRequest {
            prompt: prompt.to_vec(),
            max_new,
            sampler: Sampler::greedy(),
            stops: Vec::new(),
            ignore_eos,
        };
        self.driver.generate(request, &mut |_: u32, piece: &str| {
            on_token(piece);
            GenerationControl::Continue(())
        })?;
        Ok(())
    }
}

impl Generate for DriverCell {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.handle.text().encode(text)?)
    }

    fn generate_context(
        &mut self,
        context: &[u32],
        max_new: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        self.run(context, max_new, true, on_token)
    }

    fn generate_prompt(
        &mut self,
        prompt: &str,
        max_new: usize,
        _prefill: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        let tokens = self.encode(prompt)?;
        self.run(&tokens, max_new, false, on_token)
    }

    fn reset_profile(&mut self) {
        self.mark = Some(self.driver.executor_stats().timing.mark());
    }

    fn profile_report(&self, tokens: usize, wall: Duration) -> Option<Profile> {
        let mark = self.mark.as_ref()?;
        let timing = self.driver.executor_stats().timing;
        let entries = timing.entries_with_activity_since(mark);
        let window = poot_profile::Report::window(&timing, mark, &entries);
        Some(Profile {
            text: window.render(tokens, wall),
            device_time: device_time_receipt(&window),
        })
    }

    fn engine(&self) -> Engine {
        Engine::Driver
    }

    fn steps(&self) -> u64 {
        self.driver.stats().steps
    }

    fn counters(&self) -> PreparationCounters {
        driver_counters(&self.driver)
    }

    fn initial_counters(&self) -> PreparationCounters {
        self.initial
    }

    fn engine_receipt(&self) -> Option<Value> {
        let stats = self.driver.stats();
        let executor = self.driver.executor_stats();
        Some(json!({
            "prepared_entries": self.driver.prepared_entries(),
            "retained_bytes": self.driver.retained_bytes(),
            "compiles": stats.compiles,
            "steps": stats.steps,
            "host_picks": stats.host_picks,
            "recordings": executor.recordings,
            "replays": executor.replays,
        }))
    }
}

/// The precision a loaded GGUF reports: the format most of its packed weights are stored in.
fn dominant_packed_format(handle: &ModelHandle, path: &str) -> anyhow::Result<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for (_, entry) in handle.store().iter() {
        if let WeightEntry::Packed(payload) = entry {
            *counts
                .entry(format!("{:?}", payload.weight().format()).to_lowercase())
                .or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, n)| n)
        .map(|(format, _)| format)
        .ok_or_else(|| anyhow::anyhow!("{path}: no weight is stored packed"))
}

/// The driver engine over a loaded registered family: size every driver's KV to the longest request the
/// mode makes, and name the precision to report.
fn open_driver(
    config: &Config,
    handle: ModelHandle,
) -> anyhow::Result<(Box<dyn Generate>, String)> {
    let handle = Arc::new(handle);
    let precision = match &config.weights {
        Weights::Safetensors(_) => config.precision.clone(),
        Weights::Gguf(path) => dominant_packed_format(&handle, path)?,
    };
    let backend = match config.backend {
        Backend::Ptx => BackendChoice::Ptx,
        Backend::Rocm => BackendChoice::Rocm,
        Backend::Wgpu => BackendChoice::Wgpu,
    };
    let (longest, chunk, prompts) = match config.mode {
        Mode::Single => {
            let prompt = handle.text().encode(&single_prompt(config)?)?.len();
            (
                prompt + config.gen_tokens,
                if config.prefill { prompt } else { 1 },
                vec![prompt],
            )
        }
        Mode::DecodeCurve => {
            let isl = config.isl_list.iter().max().copied().unwrap_or(0);
            (isl + config.osl, isl, config.isl_list.clone())
        }
    };
    let capacity = NonZeroUsize::new(longest).context("the longest request is empty")?;
    let executor = if config.profile {
        open_timed_executor(backend)
    } else {
        open_executor(backend)
    }
    .context("open the executor")?;
    let mut driver = Driver::new(
        Arc::clone(&handle),
        executor,
        driver_options(capacity, chunk)?,
    )
    .context("open the driver")?;
    let initial = driver_counters(&driver);
    // Every entry the cell's prompts plan to is compiled and added before the first timed token.
    driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&prompts),
            windows: &[],
            adapters: &[],
        })
        .context("prepare the driver")?;
    Ok((
        Box::new(DriverCell {
            handle,
            driver,
            initial,
            mark: None,
        }),
        precision,
    ))
}

/// A cell that opened: the engine, the precision to report (the requested one for a checkpoint, the
/// dominant stored packed format for a GGUF) and the registry family the checkpoint resolved to.
struct Opened {
    engine: Box<dyn Generate>,
    precision: String,
    family: String,
}

/// A cell that did not open. `family` is the registry family the checkpoint resolved to, `None` when
/// the failure came before the registry resolved one (a precision the runner declines, an unregistered
/// family, an unreadable config): a refusal row for a cell that did resolve joins the feature matrix.
struct OpenFailure {
    family: Option<String>,
    error: anyhow::Error,
}

impl From<anyhow::Error> for OpenFailure {
    fn from(error: anyhow::Error) -> Self {
        Self {
            family: None,
            error,
        }
    }
}

/// Load the model and open the backend's executor.
fn open(config: &Config) -> Result<Opened, OpenFailure> {
    // poot loads f32/bf16/f16 safetensors and computes in f32. Asking for another precision is asked
    // before any weight is read, so a 20B checkpoint is not loaded to be turned down.
    if !matches!(config.precision.as_str(), "f32" | "bf16" | "f16") {
        return Err(anyhow::Error::from(Declined(format!(
            "poot bench runner loads f32/bf16/f16 checkpoints (computes f32); got {:?}. \
             Quantized formats (fp8/gptq/awq/mxfp4) are on the roadmap.",
            config.precision
        )))
        .into());
    }
    let path = match &config.weights {
        Weights::Safetensors(path) | Weights::Gguf(path) => path.as_str(),
    };
    // One registry resolve decides: a registered family runs on the driver, any other is a refusal.
    let registry = Registry::builtin().context("build the family registry")?;
    let handle = ModelHandle::load(
        Path::new(path),
        &registry,
        &poot_llm::driver::LoadOptions::default(),
    )
    .map_err(|error| anyhow::Error::from(error).context(format!("load {path}")))?;
    let family = handle.config().family.to_string();
    match open_driver(config, handle) {
        Ok((engine, precision)) => Ok(Opened {
            engine,
            precision,
            family,
        }),
        Err(error) => Err(OpenFailure {
            family: Some(family),
            error,
        }),
    }
}

/// One timed generation: time to the first token, the whole generation, and the gaps between tokens.
struct Run {
    ttft_ms: f64,
    e2e_ms: f64,
    first_token_ms: Option<f64>,
    last_token_ms: Option<f64>,
    finish_ms: f64,
    tokens: usize,
    itl_ms: Vec<f64>,
    token_pieces: Vec<String>,
    /// Executor steps this request cost (set by the caller that owns the engine's counter).
    steps: u64,
}

impl Run {
    /// Decode time per token after the first: the mean ITL. Undefined (`None`) below two tokens, where
    /// there is no token-to-token gap.
    fn tpot_ms(&self) -> Option<f64> {
        (self.tokens > 1).then(|| self.itl_ms.iter().sum::<f64>() / (self.tokens - 1) as f64)
    }

    /// Decode rate over the gaps; undefined (`None`) below two tokens or with no elapsed decode time.
    fn decode_tok_s(&self) -> Option<f64> {
        let decode_ms = self.itl_ms.iter().sum::<f64>();
        (decode_ms > 0.0 && self.tokens > 1)
            .then(|| (self.tokens - 1) as f64 / (decode_ms / 1000.0))
    }
}

/// Time one generation: `generate` calls the callback once per generated token.
fn timed(generate: impl FnOnce(&mut dyn FnMut(&str)) -> anyhow::Result<()>) -> anyhow::Result<Run> {
    let start = Instant::now();
    timed_with_clock(generate, || start.elapsed())
}

/// The clock returns elapsed request time, including work before token 1 and after the last token.
fn timed_with_clock(
    generate: impl FnOnce(&mut dyn FnMut(&str)) -> anyhow::Result<()>,
    mut clock: impl FnMut() -> Duration,
) -> anyhow::Result<Run> {
    let mut stamps = Vec::new();
    let mut token_pieces = Vec::new();
    generate(&mut |piece| {
        stamps.push(clock());
        token_pieces.push(piece.to_owned());
    })?;
    let complete = clock();
    let first_token_ms = stamps.first().copied().map(ms);
    let last_token_ms = stamps.last().copied().map(ms);
    Ok(Run {
        // With no tokens there is no first/last boundary: all request time is finish.
        ttft_ms: first_token_ms.unwrap_or(0.0),
        e2e_ms: ms(complete),
        first_token_ms,
        last_token_ms,
        finish_ms: ms(complete - stamps.last().copied().unwrap_or_default()),
        token_pieces,
        tokens: stamps.len(),
        itl_ms: stamps.windows(2).map(|w| ms(w[1] - w[0])).collect(),
        steps: 0,
    })
}

impl Run {
    fn receipt(&self) -> Value {
        json!({
            "tokens": self.tokens,
            "token_pieces": self.token_pieces,
            "first_token_ms": self.first_token_ms,
            "last_token_ms": self.last_token_ms,
            "complete_ms": self.e2e_ms,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "finish_ms": self.finish_ms,
            "steps": self.steps,
            "itl_ms": self.itl_ms,
            "first_gap_ms": self.itl_ms.first(),
            "tpot_ms": self.tpot_ms(),
            "decode_tok_s": self.decode_tok_s(),
        })
    }
}

/// The preparation layers a receipt reports, in order. A layer no producer counter exposes is `None`
/// (Unknown), never zero: the harness treats an Unknown layer as incomplete evidence.
const PREPARATION_LAYERS: [&str; 8] = [
    "plan",
    "entries",
    "generated_bodies",
    "artifact_builds",
    "artifact_hits",
    "runtime_pipelines",
    "recordings",
    "graph_instantiations",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PreparationCounters {
    /// Programs the driver compiled (trace and plan).
    plan: Option<u64>,
    /// Entries the driver added to the executor.
    entries: Option<u64>,
    generated_bodies: Option<u64>,
    artifact_builds: Option<u64>,
    artifact_hits: Option<u64>,
    runtime_pipelines: Option<u64>,
    recordings: Option<u64>,
    graph_instantiations: Option<u64>,
}

impl PreparationCounters {
    fn layers(self) -> [Option<u64>; 8] {
        [
            self.plan,
            self.entries,
            self.generated_bodies,
            self.artifact_builds,
            self.artifact_hits,
            self.runtime_pipelines,
            self.recordings,
            self.graph_instantiations,
        ]
    }
}

/// Files under the artifact cache directory `POOT_KERNEL_CACHE_DIR` names: each built artifact is one file,
/// so the count is the number of artifacts the cache holds. `None` when no directory is named.
fn artifact_cache_entries() -> Option<u64> {
    fn count(dir: &Path) -> u64 {
        std::fs::read_dir(dir).map_or(0, |entries| {
            entries
                .flatten()
                .map(|entry| match entry.file_type() {
                    Ok(kind) if kind.is_dir() => count(&entry.path()),
                    Ok(_) => 1,
                    Err(_) => 0,
                })
                .sum()
        })
    }
    std::env::var_os("POOT_KERNEL_CACHE_DIR").map(|dir| count(Path::new(&dir)))
}

/// The counters now: what the engine counts, plus the artifact builds the cache directory shows as
/// files written since the process began (`since`).
fn snapshot(engine: &dyn Generate, entries_at_start: Option<u64>) -> [Option<u64>; 8] {
    let mut counters = engine.counters();
    counters.artifact_builds = artifact_cache_entries()
        .zip(entries_at_start)
        .map(|(now, start)| now.saturating_sub(start));
    counters.layers()
}

/// Preparation work per layer: what preparing the declared set cost (`prepare`) and what repeating it added
/// (`repeat`). A layer is Unknown if any snapshot of it was.
struct PreparationProbe {
    state: Option<PreparedState>,
    artifact_entries_at_start: Option<u64>,
    mark: [Option<u64>; 8],
    prepare: [Option<u64>; 8],
    repeat: [Option<u64>; 8],
    /// No window has begun: the first cold window starts from the engine's initial counters.
    first: bool,
}

impl PreparationProbe {
    fn new(state: Option<PreparedState>, artifact_entries_at_start: Option<u64>) -> Self {
        Self {
            state,
            artifact_entries_at_start,
            mark: [None; 8],
            prepare: [Some(0); 8],
            repeat: [Some(0); 8],
            first: true,
        }
    }

    fn passes(&self) -> usize {
        self.state.map_or(1, PreparedState::warmup_passes)
    }

    /// Start counting a window. The first window of a cold process starts from the fresh engine's counters,
    /// so the preparation the engine did while opening is counted; every other window starts from now.
    fn begin(&mut self, engine: &dyn Generate) {
        let cold_start = self.first && self.passes() == 1;
        self.first = false;
        self.mark = if cold_start {
            let mut counters = engine.initial_counters();
            counters.artifact_builds = self.artifact_entries_at_start.map(|_| 0);
            counters.layers()
        } else {
            snapshot(engine, self.artifact_entries_at_start)
        };
    }

    fn advance(total: &mut [Option<u64>; 8], mark: &mut [Option<u64>; 8], now: [Option<u64>; 8]) {
        for ((total, mark), now) in total.iter_mut().zip(mark.iter_mut()).zip(now) {
            *total = match (*total, *mark, now) {
                (Some(total), Some(mark), Some(now)) => Some(total + now.saturating_sub(mark)),
                _ => None,
            };
            *mark = now;
        }
    }

    /// The declared set is prepared: work since `begin` is the preparation.
    fn prepared(&mut self, engine: &dyn Generate) {
        let now = snapshot(engine, self.artifact_entries_at_start);
        Self::advance(&mut self.prepare, &mut self.mark, now);
    }

    /// The timed window ran: work since `prepared` is what repeating the set added.
    fn repeated(&mut self, engine: &dyn Generate) {
        let now = snapshot(engine, self.artifact_entries_at_start);
        Self::advance(&mut self.repeat, &mut self.mark, now);
    }

    fn receipt(&self) -> Value {
        let layers: serde_json::Map<String, Value> = PREPARATION_LAYERS
            .iter()
            .zip(self.prepare.iter().zip(&self.repeat))
            .map(|(layer, (prepare, repeat))| {
                (
                    (*layer).to_string(),
                    json!({ "prepare": prepare, "repeat": repeat }),
                )
            })
            .collect();
        json!({
            "state": self.state.map(PreparedState::label),
            "artifact_entries_at_start": self.artifact_entries_at_start,
            "layers": layers,
        })
    }
}

/// One timed request on `engine`, with the executor steps it cost.
fn timed_request(
    engine: &mut dyn Generate,
    generate: impl FnOnce(&mut dyn Generate, &mut dyn FnMut(&str)) -> anyhow::Result<()>,
) -> anyhow::Result<Run> {
    let before = engine.steps();
    let mut run = timed(|on_token| generate(&mut *engine, on_token))?;
    run.steps = engine.steps() - before;
    Ok(run)
}

/// The one measure loop: run the configured mode on `engine` and return the result object.
fn measure(
    config: &Config,
    engine: &mut dyn Generate,
    precision: &str,
    family: &str,
) -> anyhow::Result<Value> {
    let mut probe = PreparationProbe::new(config.prepared_state, config.artifact_entries_at_start);
    let body = match config.mode {
        Mode::Single => measure_single(config, engine, &mut probe)?,
        Mode::DecodeCurve => measure_curve(config, engine, &mut probe)?,
    };
    let mut line = identity(
        Some(engine.engine()),
        Some(family),
        config.backend,
        config.mode,
        precision,
    );
    line["profiled"] = json!(config.profile);
    line["preparation"] = probe.receipt();
    if let Some(receipt) = engine.engine_receipt() {
        line["driver"] = receipt;
    }
    line["timing_contract"] =
        json!("request-start/token-callbacks/completion; TPOT excludes finish");
    line.as_object_mut()
        .expect("identity is an object")
        .extend(body.as_object().cloned().expect("a mode returns an object"));
    Ok(line)
}

/// What every result line carries: which build produced it, with which engine (when one was chosen), on
/// which backend, in which mode. The harness checks `backend` against the one it requested and refuses a
/// result with no `build_sha`. `family` is the registry family the checkpoint resolved to, as the registry
/// reports it (never a mapping from the manifest); `null` on a refusal that came before one resolved.
fn identity(
    engine: Option<Engine>,
    family: Option<&str>,
    backend: Backend,
    mode: Mode,
    precision: &str,
) -> Value {
    let mut line = json!({
        "framework": "poot",
        "build_sha": BUILD_SHA,
        "family": family,
        "backend": backend.label(),
        "mode": mode.label(),
        "precision": precision,
    });
    if let Some(engine) = engine {
        line["engine"] = json!(engine.label());
    }
    line
}

/// Single mode's prompt: the `--prompt-file`'s text, or the default.
fn single_prompt(config: &Config) -> anyhow::Result<String> {
    Ok(match &config.prompt_file {
        Some(file) => std::fs::read_to_string(file)
            .with_context(|| format!("read --prompt-file {file}"))?
            .trim_end()
            .to_string(),
        None => "The capital of France is".to_string(),
    })
}

fn measure_single(
    config: &Config,
    engine: &mut dyn Generate,
    probe: &mut PreparationProbe,
) -> anyhow::Result<Value> {
    let prompt = single_prompt(config)?;
    if !config.json {
        eprintln!(
            "poot-bench: precision={} gen_tokens={} warmup={} iters={} prefill={} backend={}",
            config.precision,
            config.gen_tokens,
            config.warmup,
            config.iters,
            config.prefill,
            config.backend.label()
        );
    }
    let run_once = |engine: &mut dyn Generate| {
        timed_request(engine, |engine, on_token| {
            engine.generate_prompt(&prompt, config.gen_tokens, config.prefill, on_token)
        })
    };

    let mut warmup_runs = Vec::new();
    let passes = probe.passes();
    for pass in 0..passes {
        if pass + 1 == passes {
            probe.begin(&*engine);
        }
        for w in 0..config.warmup {
            warmup_runs.push(run_once(engine)?.receipt());
            if !config.json {
                eprintln!("poot-bench: warmup {}/{} done", w + 1, config.warmup);
            }
        }
    }
    probe.prepared(&*engine);
    // Reset after warmup (compiles, cache fills) so the report covers only the timed iterations.
    if config.profile {
        engine.reset_profile();
    }

    let mut runs = Vec::with_capacity(config.iters);
    let timed_start = Instant::now();
    for it in 0..config.iters {
        let run = run_once(engine)?;
        if !config.json {
            eprintln!(
                "poot-bench: iter {}/{}  TTFT {:.0}ms  TPOT {}ms  {} tok/s",
                it + 1,
                config.iters,
                run.ttft_ms,
                run.tpot_ms().map_or("-".to_string(), |v| format!("{v:.1}")),
                run.decode_tok_s()
                    .map_or("-".to_string(), |v| format!("{v:.1}"))
            );
        }
        runs.push(run);
    }
    let timed_wall = timed_start.elapsed();
    probe.repeated(&*engine);

    let device_time = if config.profile {
        let tokens = runs.iter().map(|run| run.tokens).sum();
        print_profile(engine.profile_report(tokens, timed_wall))
    } else {
        Value::Null
    };

    // The summary medians are over the requests that define the metric; the result contract wants a number,
    // so a cell where none does (fewer than two tokens) summarizes as 0 while every request says null.
    let column =
        |value: fn(&Run) -> Option<f64>| runs.iter().filter_map(value).collect::<Vec<f64>>();
    let ttft = column(|run| Some(run.ttft_ms));
    let tok_s = column(Run::decode_tok_s);
    Ok(json!({
        "warmup_runs": warmup_runs,
        "runs": runs.iter().map(Run::receipt).collect::<Vec<_>>(),
        "prompt_tokens": engine.encode(&prompt)?.len(),
        "gen_tokens": runs.last().map_or(0, |run| run.tokens),
        "iterations": config.iters,
        "device_time": device_time,
        "ttft_ms": round3(median(&ttft)),
        "tpot_ms": round3(median(&column(Run::tpot_ms))),
        "e2e_ms": round3(median(&column(|run| Some(run.e2e_ms)))),
        "decode_tok_s": round3(median(&tok_s)),
        "ttft_ms_stdev": round3(stdev(&ttft)),
        "decode_tok_s_stdev": round3(stdev(&tok_s)),
    }))
}

/// One ISL's measurement: warmup, then `iters` timed runs, as the curve-point object. The context is the
/// base phrase's tokens cycled to exactly `isl`.
fn measure_isl(
    config: &Config,
    engine: &mut dyn Generate,
    base: &[u32],
    isl: usize,
    probe: &mut PreparationProbe,
) -> anyhow::Result<Value> {
    let context: Vec<u32> = (0..isl).map(|i| base[i % base.len()]).collect();
    let run_once = |engine: &mut dyn Generate| {
        timed_request(engine, |engine, on_token| {
            engine.generate_context(&context, config.osl, on_token)
        })
    };
    let mut warmup_runs = Vec::new();
    let passes = probe.passes();
    for pass in 0..passes {
        if pass + 1 == passes {
            probe.begin(&*engine);
        }
        for _ in 0..config.warmup {
            warmup_runs.push(run_once(engine)?.receipt());
        }
    }
    probe.prepared(&*engine);
    // Card 552: a fresh mark after this ISL's own warmup (not once for the whole sweep), since each
    // ISL's prefill/warmup cost differs and must not leak into another ISL's window.
    if config.profile {
        engine.reset_profile();
    }
    let timed_start = Instant::now();
    let mut iters = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        let run = run_once(engine)?;
        iters.push(run.receipt());
    }
    probe.repeated(&*engine);
    let device_time = if config.profile {
        let tokens: usize = iters
            .iter()
            .map(|iter| iter["itl_ms"].as_array().map_or(0, Vec::len))
            .sum();
        eprintln!("poot-bench: isl={isl} profile window:");
        print_profile(engine.profile_report(tokens, timed_start.elapsed()))
    } else {
        Value::Null
    };
    Ok(json!({
        "isl": isl,
        "prompt_tokens": isl,
        "warmup_runs": warmup_runs,
        "iters": iters,
        "device_time": device_time,
    }))
}

/// decode-curve. A per-ISL failure after the first keeps the measured points and records a caveat: the
/// largest contexts fault first, and larger ones would fault too. A failure at the first ISL measured
/// nothing, so it is the cell's error (a refusal stays a refusal).
///
/// Batched prefill materializes the `[Hq,N,N]` attention scores, so a very large ISL on a large model can
/// run out of memory in f32; that surfaces as such a caveat.
fn measure_curve(
    config: &Config,
    engine: &mut dyn Generate,
    probe: &mut PreparationProbe,
) -> anyhow::Result<Value> {
    let base = engine
        .encode(CURVE_BASE_PHRASE)
        .context("encode base context")?;
    anyhow::ensure!(!base.is_empty(), "base context encoded to zero tokens");
    eprintln!(
        "poot-bench: decode-curve isls={:?} osl={} warmup={} iters={} backend={}",
        config.isl_list,
        config.osl,
        config.warmup,
        config.iters,
        config.backend.label()
    );
    let mut curve: Vec<Value> = Vec::new();
    let mut caveats: Vec<String> = Vec::new();
    // Card 552: each ISL takes its own mark and prints its own profile window inside
    // `measure_isl` (a fresh mark after that ISL's own warmup); there is no whole-sweep window.
    for (index, &isl) in config.isl_list.iter().enumerate() {
        // An ISL beyond a model's `max_position_embeddings` (e.g. olmo2-1b's 4096) is a trace-time
        // shape-contract violation that `poot-graph-ir::Builder::emit` panics on by design, and the ISL list
        // is not scoped per model. An uncaught panic would abort the process and discard every smaller
        // measured ISL, so it takes the same path as any per-ISL error.
        let point = std::panic::catch_unwind(AssertUnwindSafe(|| {
            measure_isl(config, engine, &base, isl, probe)
        }))
        .unwrap_or_else(|payload| {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic (non-string payload)".to_string());
            Err(anyhow::anyhow!("{message}"))
        });
        match point {
            Ok(point) => {
                curve.push(point);
                // The harness curve-progress relay parses `isl=<N> done` (regex `isl=(\d+)\s+done`).
                eprintln!(
                    "poot-bench: isl={isl} done ({}/{}), osl={}",
                    index + 1,
                    config.isl_list.len(),
                    config.osl
                );
            }
            Err(error) if index == 0 => return Err(error),
            Err(error) => {
                let reason = error
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or("error")
                    .to_string();
                eprintln!(
                    "poot-bench: isl={isl} FAILED ({reason}); emitting partial curve ({index}/{} isls)",
                    config.isl_list.len()
                );
                caveats.push(format!("isl {isl}+ omitted: {reason}"));
                break;
            }
        }
    }
    Ok(json!({ "osl": config.osl, "caveats": caveats, "curve": curve }))
}

/// Print the window and return its device time for the result line (`null` when nothing was profiled).
fn print_profile(profile: Option<Profile>) -> Value {
    let Some(profile) = profile else {
        return Value::Null;
    };
    eprintln!("=== PROFILE ===");
    eprintln!("{}", profile.text);
    profile.device_time
}

/// A request the runner itself does not serve (a precision it cannot load). Reported like any refusal.
#[derive(Debug)]
struct Declined(String);

impl std::fmt::Display for Declined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Declined {}

/// The refusal a failure carries, if it is one, as its own text. This is the runner's only classification
/// of "poot does not run this": [`Declined`], or the driver's [`DriverError::Unsupported`]
/// (a family the registry does not hold, a shape the family or the planner refuses), found anywhere in
/// the error chain.
fn refusal(error: &anyhow::Error) -> Option<(Option<Engine>, String)> {
    error.chain().find_map(|cause| {
        if let Some(DriverError::Unsupported(_)) = cause.downcast_ref::<DriverError>() {
            return Some((Some(Engine::Driver), cause.to_string()));
        }
        cause
            .downcast_ref::<Declined>()
            .map(|declined| (None, declined.to_string()))
    })
}

/// The result line for `outcome`: the measurement, or the refusal recorded as `unsupported` with the
/// refusal's own text. Any other failure is returned to `main` and exits non-zero.
fn result_line(
    config: &Config,
    family: Option<&str>,
    outcome: anyhow::Result<Value>,
) -> anyhow::Result<Value> {
    match outcome {
        Err(error) => match refusal(&error) {
            Some((engine, reason)) => {
                let mut line = identity(
                    engine,
                    family,
                    config.backend,
                    config.mode,
                    &config.precision,
                );
                let fields = line.as_object_mut().expect("identity is an object");
                fields.insert("status".to_string(), json!("unsupported"));
                fields.insert("reason".to_string(), json!(reason));
                Ok(line)
            }
            None => Err(error),
        },
        ok => ok,
    }
}

/// The cell's result line: open the model, measure it, and record a refusal as `unsupported` with the
/// registry family the checkpoint resolved to (when it resolved one).
fn run(config: &Config) -> anyhow::Result<Value> {
    let open_start = Instant::now();
    let (family, outcome) = match open(config) {
        Ok(mut opened) => {
            let open_ms = round3(ms(open_start.elapsed()));
            let outcome = measure(
                config,
                opened.engine.as_mut(),
                &opened.precision,
                &opened.family,
            )
            .map(|mut line| {
                // Loading is timed apart from every request: it happens before the first one.
                line["preparation"]["open_ms"] = json!(open_ms);
                line
            });
            (Some(opened.family), outcome)
        }
        Err(failure) => (failure.family, Err(failure.error)),
    };
    result_line(config, family.as_deref(), outcome)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = match parse_args(&args)? {
        Invocation::Version => {
            println!("poot-bench-runner {BUILD_SHA}");
            return Ok(());
        }
        Invocation::Run(config) => config,
    };
    println!("{}", run(&config)?);
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Timings go out with microsecond resolution; more digits are noise in a stored row.
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn stdev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Write;
    use std::process::{Command, Stdio};

    use super::*;

    /// `CARGO_MANIFEST_DIR`, read at runtime (not baked via `env!`): the compile-time macro's value is
    /// embedded into this test binary's compiled object code at build time, and a shared compile cache
    /// (kache) that reuses that object across worktrees by source-content hash would then serve whichever
    /// worktree's path happened to compile it first (card 530's build.rs fix; card 543 review).
    /// `std::env::var` reads the environment cargo sets fresh for every test-binary invocation, so it is
    /// correct regardless of which worktree compiled the binary.
    fn manifest_dir() -> &'static str {
        static DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries")
        })
        .as_str()
    }

    #[derive(Debug, PartialEq)]
    enum Event {
        Prompt { max_new: usize, prefill: bool },
        Context { len: usize, max_new: usize },
        ResetProfile,
        ProfileReport { tokens: usize },
    }

    /// A backend that generates instantly and records what the measure loop asked of it.
    struct Stub {
        events: RefCell<Vec<Event>>,
        /// Fail every context generation of this length, with `error`.
        fail_at_isl: Option<usize>,
        /// Fail every prompt generation, with `error`.
        fail_prompt: bool,
        error: fn() -> anyhow::Error,
        /// Executor steps the stub has "taken": one per generated token (prefill yields the first).
        steps: std::cell::Cell<u64>,
        /// What `counters` reports, advanced by `advance_counters`.
        counters: std::cell::Cell<PreparationCounters>,
    }

    impl Stub {
        fn new() -> Self {
            Stub {
                events: RefCell::new(Vec::new()),
                fail_at_isl: None,
                fail_prompt: false,
                error: || anyhow::anyhow!("stub failure"),
                steps: std::cell::Cell::new(0),
                counters: std::cell::Cell::new(PreparationCounters::default()),
            }
        }

        fn emit(max_new: usize, on_token: &mut dyn FnMut(&str)) {
            for _ in 0..max_new {
                // A token takes at least a millisecond, so every measured run has a nonzero decode time.
                std::thread::sleep(Duration::from_millis(1));
                on_token("t");
            }
        }
    }

    impl Generate for Stub {
        fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            Ok((0..text.split_whitespace().count() as u32).collect())
        }

        fn generate_context(
            &mut self,
            context: &[u32],
            max_new: usize,
            on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            self.events.borrow_mut().push(Event::Context {
                len: context.len(),
                max_new,
            });
            if self.fail_at_isl == Some(context.len()) {
                return Err((self.error)());
            }
            self.steps.set(self.steps.get() + max_new as u64);
            Stub::emit(max_new, on_token);
            Ok(())
        }

        fn generate_prompt(
            &mut self,
            _prompt: &str,
            max_new: usize,
            prefill: bool,
            on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            self.events
                .borrow_mut()
                .push(Event::Prompt { max_new, prefill });
            if self.fail_prompt {
                return Err((self.error)());
            }
            self.steps.set(self.steps.get() + max_new as u64);
            Stub::emit(max_new, on_token);
            Ok(())
        }

        fn reset_profile(&mut self) {
            self.events.borrow_mut().push(Event::ResetProfile);
        }

        fn profile_report(&self, tokens: usize, _wall: Duration) -> Option<Profile> {
            self.events
                .borrow_mut()
                .push(Event::ProfileReport { tokens });
            Some(Profile {
                text: "stub window".to_string(),
                device_time: json!({ "kind": "unknown" }),
            })
        }

        fn engine(&self) -> Engine {
            Engine::Driver
        }

        fn steps(&self) -> u64 {
            self.steps.get()
        }

        fn counters(&self) -> PreparationCounters {
            self.counters.get()
        }
    }

    fn config(extra: &[&str]) -> Config {
        let mut args = vec!["poot-bench-runner", "--model-dir", "/nonexistent"];
        args.extend_from_slice(extra);
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        match parse_args(&args).expect("test arguments parse") {
            Invocation::Run(config) => config,
            Invocation::Version => unreachable!("no --version"),
        }
    }

    fn parse_error(extra: &[&str]) -> String {
        let mut args = vec!["poot-bench-runner", "--model-dir", "/nonexistent"];
        args.extend_from_slice(extra);
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        match parse_args(&args) {
            Ok(_) => panic!("{extra:?} parsed, expected an error"),
            Err(error) => format!("{error:#}"),
        }
    }

    fn window(sum: Option<Duration>, span: Option<Duration>) -> poot_profile::WindowReport {
        poot_profile::WindowReport {
            entries: vec![0],
            steps: 1,
            host: poot_profile::HostTiming::default(),
            device: poot_profile::DeviceWindow {
                sum_of_dispatch_durations: sum,
                device_span: span,
            },
            detail: poot_profile::Coverage::Unavailable,
            buckets: Default::default(),
            perturbed: false,
        }
    }

    /// POOT-1072: a window the engine measured reports its milliseconds, and a window with no device time
    /// (ROCm, an untimed device) reports the typed `unknown`, never a zero. Mutation: make
    /// `device_time_receipt` answer `measured` with the (absent) sum read as zero; the unknown case goes red.
    #[test]
    fn a_window_reports_measured_milliseconds_or_the_typed_unknown() {
        let measured = device_time_receipt(&window(
            Some(Duration::from_micros(1500)),
            Some(Duration::from_micros(2000)),
        ));
        assert_eq!(
            measured,
            json!({ "kind": "measured", "sum_of_dispatch_ms": 1.5, "device_span_ms": 2.0 })
        );
        assert_eq!(
            device_time_receipt(&window(None, None)),
            json!({ "kind": "unknown" })
        );
    }

    /// POOT-1072: `--profile` parses on every backend, ROCm included: it no longer refuses a backend with no
    /// device timestamps, it reports `unknown`. Mutation: restore the ROCm refusal in `parse_args`.
    #[test]
    fn profile_is_accepted_on_every_backend() {
        for backend in Backend::ALL {
            let config = config(&["--profile", "--backend", backend.label()]);
            assert!(config.profile, "{}", backend.label());
            assert_eq!(config.backend, backend);
        }
    }

    fn refused() -> anyhow::Error {
        anyhow::Error::from(DriverError::Unsupported(Unsupported::Backend {
            backend: "stub_backend",
        }))
        .context("generate")
    }

    fn prompt_events(count: usize, max_new: usize, prefill: bool) -> Vec<Event> {
        (0..count)
            .map(|_| Event::Prompt { max_new, prefill })
            .collect()
    }

    fn context_events(count: usize, len: usize, max_new: usize) -> Vec<Event> {
        (0..count)
            .map(|_| Event::Context { len, max_new })
            .collect()
    }

    /// SC-001: the warmup, the measured runs and the curve rows are the same for every backend value.
    #[test]
    fn every_backend_runs_the_same_measure_protocol() {
        for backend in Backend::ALL {
            let label = backend.label();

            let mut single = config(&[
                "--warmup",
                "3",
                "--iters",
                "4",
                "--gen-tokens",
                "6",
                "--profile",
                "--prefill",
            ]);
            single.backend = backend;
            let mut stub = Stub::new();
            let line = measure(&single, &mut stub, "bf16", "stub_family").unwrap();
            let mut expected = prompt_events(3, 6, true);
            expected.push(Event::ResetProfile);
            expected.extend(prompt_events(4, 6, true));
            expected.push(Event::ProfileReport { tokens: 4 * 6 });
            assert_eq!(*stub.events.borrow(), expected, "{label}: single protocol");
            assert_eq!(line["backend"], label);
            assert_eq!(line["iterations"], 4, "{label}");
            assert_eq!(line["gen_tokens"], 6, "{label}");

            let mut curve = config(&[
                "--mode",
                "decode-curve",
                "--isl-list",
                "8,16,32",
                "--osl",
                "5",
                "--warmup",
                "2",
                "--iters",
                "3",
            ]);
            curve.backend = backend;
            let mut stub = Stub::new();
            let line = measure(&curve, &mut stub, "bf16", "stub_family").unwrap();
            let mut expected = Vec::new();
            for isl in [8, 16, 32] {
                expected.extend(context_events(2 + 3, isl, 5));
            }
            assert_eq!(*stub.events.borrow(), expected, "{label}: curve protocol");
            assert_eq!(line["backend"], label);
            let points = line["curve"].as_array().unwrap();
            let isls: Vec<u64> = points.iter().map(|p| p["isl"].as_u64().unwrap()).collect();
            assert_eq!(isls, [8, 16, 32], "{label}: one curve row per ISL");
            for point in points {
                let iters = point["iters"].as_array().unwrap();
                assert_eq!(iters.len(), 3, "{label}: timed runs per ISL");
                for iter in iters {
                    assert_eq!(
                        iter["itl_ms"].as_array().unwrap().len(),
                        4,
                        "{label}: osl - 1 gaps"
                    );
                }
            }
        }
    }

    /// Run the harness's own runner-result validation over `line`, as the harness would before it turns
    /// the line into a row. The runner and the harness must agree on one contract, and the harness owns it.
    fn harness_accepts(line: &Value, curve: bool, gen_tokens: usize, backend: Backend) {
        let harness = format!("{}/../../harness", manifest_dir());
        let script = format!(
            "import json, sys\n\
             sys.path.insert(0, {harness:?})\n\
             from result_contract import validate_runner_result\n\
             scenario = {{'kind': 'decode-curve', 'osl': 5, 'isl': [8, 16]}} if {curve} else {{'id': 'single'}}\n\
             validate_runner_result(json.load(sys.stdin), framework='poot', precision='bf16', \
             scenario=scenario, gen_tokens={gen_tokens}, backend={backend:?})\n",
            curve = if curve { "True" } else { "False" },
            backend = backend.label(),
        );
        let mut child = Command::new("python3")
            .args(["-c", &script])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("python3 is on PATH in the dev shell");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(line.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "the harness rejects the runner line {line}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// SC-003: each mode ends in exactly one result line the harness accepts, and the runner accepts no
    /// argument that would report anything else.
    #[test]
    fn each_mode_prints_one_line_the_harness_turns_into_a_row() {
        for mode in Mode::ALL {
            let mut config = config(&[
                "--mode",
                mode.label(),
                "--isl-list",
                "8,16",
                "--osl",
                "5",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gen-tokens",
                "6",
            ]);
            config.backend = Backend::Rocm;
            let outcome = measure(&config, &mut Stub::new(), "bf16", "stub_family");
            let line = result_line(&config, None, outcome).unwrap();
            let text = line.to_string();
            assert!(!text.contains('\n'), "{mode:?}: the result is one line");
            assert_eq!(line["mode"], mode.label());
            assert_eq!(line["framework"], "poot");
            assert_eq!(line["build_sha"], BUILD_SHA);
            assert_eq!(line["family"], "stub_family");
            harness_accepts(&line, mode == Mode::DecodeCurve, 6, Backend::Rocm);
        }
    }

    /// POOT-1072: a profiled run puts the window's device time in the result line (the row's for a single run,
    /// each ISL point's for a curve), the line passes the harness's contract, and an unprofiled run carries
    /// `null`. Mutation: drop `"device_time"` from the single run's `json!` (or the curve point's); the key is
    /// missing and this row goes red.
    #[test]
    fn a_profiled_run_reports_its_device_time_in_a_line_the_harness_accepts() {
        for mode in Mode::ALL {
            for profile in [true, false] {
                let mut args = vec![
                    "--mode",
                    mode.label(),
                    "--isl-list",
                    "8,16",
                    "--osl",
                    "5",
                    "--warmup",
                    "1",
                    "--iters",
                    "2",
                    "--gen-tokens",
                    "6",
                ];
                if profile {
                    args.push("--profile");
                }
                let mut config = config(&args);
                config.backend = Backend::Rocm;
                let outcome = measure(&config, &mut Stub::new(), "bf16", "stub_family");
                let line = result_line(&config, None, outcome).unwrap();
                let expected = if profile {
                    json!({ "kind": "unknown" })
                } else {
                    Value::Null
                };
                let carriers: Vec<&Value> = match mode {
                    Mode::Single => vec![&line],
                    Mode::DecodeCurve => line["curve"].as_array().unwrap().iter().collect(),
                };
                assert!(!carriers.is_empty());
                for carrier in carriers {
                    assert_eq!(
                        carrier.get("device_time"),
                        Some(&expected),
                        "{mode:?} profile={profile}"
                    );
                }
                harness_accepts(&line, mode == Mode::DecodeCurve, 6, Backend::Rocm);
            }
        }
    }

    #[test]
    fn no_flag_reports_anything_but_a_measurement() {
        for flag in [
            "--card277-bf16-generation",
            "--card278-bf16-residency",
            "--probe-bf16",
            "--probe-tc-matmul",
            "--bench-tc",
            "--probe-dequant-gptq",
            "--batched",
            "--captured",
            "--prefill-check",
            "--generate-check-packed",
            "--decode-check-bf16",
            "--decode-compare-bf16",
            "--prefill-compare",
            "--prefill-scan",
            "--probe-gather",
            "--eager",
            "--precompile",
            "--precompile-packed",
            "--show",
        ] {
            let error = parse_error(&[flag]);
            assert!(
                error.contains("unknown argument") && error.contains(flag),
                "{flag}: {error}"
            );
        }
    }

    /// The reason the harness records for `line`, read through the harness's own `runner_refusal`.
    fn harness_refusal(line: &Value, backend: Backend) -> String {
        let harness = format!("{}/../../harness", manifest_dir());
        let script = format!(
            "import json, sys\n\
             sys.path.insert(0, {harness:?})\n\
             from result_contract import runner_refusal\n\
             print(runner_refusal(json.load(sys.stdin), framework='poot', backend={backend:?}))\n",
            backend = backend.label(),
        );
        let mut child = Command::new("python3")
            .args(["-c", &script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("python3 is on PATH in the dev shell");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(line.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "the harness rejects the refusal {line}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .trim_end()
            .to_string()
    }

    /// SC-002 (runner half, stubbed backend): a refusal from generation is recorded with the refusal's own
    /// text; any other failure is an error, never an `unsupported` row.
    #[test]
    fn a_refusal_is_recorded_as_the_runners_own_text() {
        let mut config = config(&["--warmup", "1", "--iters", "1"]);
        config.backend = Backend::Wgpu;
        let mut stub = Stub::new();
        stub.fail_prompt = true;
        stub.error = refused;
        let line = result_line(
            &config,
            Some("stub_family"),
            measure(&config, &mut stub, "bf16", "stub_family"),
        )
        .unwrap();
        assert_eq!(line["status"], "unsupported");
        assert!(
            line["reason"].as_str().unwrap().contains("stub_backend"),
            "{}",
            line["reason"]
        );
        assert_eq!(line["backend"], "wgpu");
        assert_eq!(line["build_sha"], BUILD_SHA);

        stub.error = || anyhow::anyhow!("out of memory");
        let error = result_line(
            &config,
            Some("stub_family"),
            measure(&config, &mut stub, "bf16", "stub_family"),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "out of memory");
    }

    /// A checkpoint directory holding only a config.json, in a fresh temp directory.
    fn config_only_checkpoint(name: &str, config_json: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("poot-bench-runner-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config_json).unwrap();
        dir
    }

    /// SC-002 (real entry): a model the registry does not hold is refused by `ModelHandle::load`
    /// itself, and the refusal reaches the result line with its text and passes the harness's contract.
    /// Nothing here builds the error by hand, so the test cannot pass against a type the bench path never
    /// raises.
    #[test]
    fn a_model_the_loader_does_not_serve_is_a_recorded_refusal() {
        let dir = config_only_checkpoint("unserved", r#"{"model_type": "stub_arch"}"#);
        let mut config = config(&["--precision", "bf16"]);
        config.weights = Weights::Safetensors(dir.display().to_string());
        config.backend = Backend::Rocm;
        let line = run(&config).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(line["status"], "unsupported");
        let reason = line["reason"].as_str().unwrap();
        assert!(
            reason.contains("no family is registered") && reason.contains("stub_arch"),
            "{reason}"
        );
        assert_eq!(harness_refusal(&line, Backend::Rocm), reason);
        // The registry resolved no family, so the row names none.
        assert_eq!(line["family"], Value::Null);
    }

    #[test]
    fn a_load_failure_that_is_not_a_refusal_stays_an_error() {
        let mut config = config(&[]);
        config.weights = Weights::Safetensors("/nonexistent/checkpoint".to_string());
        let error = run(&config).unwrap_err();
        assert!(
            format!("{error:#}").contains("No such file or directory"),
            "{error:#}"
        );
    }

    /// A precision the runner cannot load is a refusal reported before any weight is read: the directory
    /// here does not exist, so a load attempt would be a different (error) outcome.
    #[test]
    fn an_unloadable_precision_is_a_refusal_before_the_model_is_read() {
        let config = config(&["--precision", "mxfp4"]);
        let line = run(&config).unwrap();
        assert_eq!(line["status"], "unsupported");
        assert_eq!(line["precision"], "mxfp4");
        let reason = line["reason"].as_str().unwrap();
        assert!(
            reason.contains("loads f32/bf16/f16 checkpoints") && reason.contains("mxfp4"),
            "{reason}"
        );
        assert_eq!(harness_refusal(&line, Backend::Ptx), reason);
    }

    #[test]
    fn a_curve_keeps_measured_points_but_a_first_point_failure_is_the_cells_failure() {
        let mut config = config(&[
            "--mode",
            "decode-curve",
            "--isl-list",
            "8,16",
            "--osl",
            "3",
            "--warmup",
            "0",
            "--iters",
            "1",
        ]);
        config.backend = Backend::Ptx;
        let mut stub = Stub::new();
        stub.fail_at_isl = Some(16);
        stub.error = || anyhow::anyhow!("CUDA out of memory\nbacktrace");
        let line = measure(&config, &mut stub, "bf16", "stub_family").unwrap();
        assert_eq!(line["curve"].as_array().unwrap().len(), 1);
        assert_eq!(
            line["caveats"],
            json!(["isl 16+ omitted: CUDA out of memory"])
        );

        stub.fail_at_isl = Some(8);
        stub.error = refused;
        let outcome = measure(&config, &mut stub, "bf16", "stub_family");
        let line = result_line(&config, None, outcome).unwrap();
        assert_eq!(line["status"], "unsupported");
    }

    #[test]
    fn a_panic_at_one_isl_is_that_isls_caveat() {
        struct Panics;
        impl Generate for Panics {
            fn encode(&self, _: &str) -> anyhow::Result<Vec<u32>> {
                Ok(vec![1, 2])
            }
            fn generate_context(
                &mut self,
                context: &[u32],
                max_new: usize,
                on_token: &mut dyn FnMut(&str),
            ) -> anyhow::Result<()> {
                assert!(
                    context.len() < 16,
                    "position 16 exceeds max_position_embeddings"
                );
                Stub::emit(max_new, on_token);
                Ok(())
            }
            fn generate_prompt(
                &mut self,
                _: &str,
                _: usize,
                _: bool,
                _: &mut dyn FnMut(&str),
            ) -> anyhow::Result<()> {
                unreachable!("decode-curve only")
            }
            fn reset_profile(&mut self) {}
            fn profile_report(&self, _: usize, _: Duration) -> Option<Profile> {
                None
            }
            fn engine(&self) -> Engine {
                Engine::Driver
            }
            fn steps(&self) -> u64 {
                0
            }
        }
        let config = config(&[
            "--mode",
            "decode-curve",
            "--isl-list",
            "8,16",
            "--osl",
            "3",
            "--warmup",
            "0",
            "--iters",
            "1",
        ]);
        let line = measure(&config, &mut Panics, "bf16", "stub_family").unwrap();
        assert_eq!(line["curve"].as_array().unwrap().len(), 1);
        let caveat = line["caveats"][0].as_str().unwrap();
        assert!(caveat.starts_with("isl 16+ omitted: "), "{caveat}");
        assert!(caveat.contains("max_position_embeddings"), "{caveat}");
    }

    #[test]
    fn arguments_are_parsed_strictly() {
        let defaults = config(&[]);
        assert_eq!(
            (
                defaults.gen_tokens,
                defaults.warmup,
                defaults.iters,
                defaults.osl
            ),
            (128, 2, 5, 128)
        );
        assert_eq!(defaults.backend, Backend::Ptx);
        assert_eq!(defaults.mode, Mode::Single);
        assert!(matches!(defaults.weights, Weights::Safetensors(_)));

        assert!(matches!(
            config(&["--gguf", "/m/model.gguf"]).weights,
            Weights::Gguf(_)
        ));
        let gguf_dir = ["poot-bench-runner", "--model-dir", "/m/model.gguf"].map(String::from);
        match parse_args(&gguf_dir).unwrap() {
            Invocation::Run(config) => assert!(matches!(config.weights, Weights::Gguf(_))),
            Invocation::Version => unreachable!(),
        }

        // These used to fall back to the default silently and measure something else.
        assert!(parse_error(&["--iters", "five"]).contains("--iters must be"));
        assert!(parse_error(&["--isl-list", "128,x"]).contains("not an integer"));
        assert!(parse_error(&["--iters"]).contains("needs a value"));
        assert!(parse_error(&["--iters", "0"]).contains("--iters must be positive"));
        assert!(parse_error(&["--backend", "cuda"]).contains("unknown backend"));
        assert!(parse_error(&["--mode", "decode-curve"]).contains("non-empty --isl-list"));
        assert!(parse_error(&["--engine", "driver"]).contains("unknown argument"));
    }

    /// The registry decides the engine: a registered family's checkpoint opens through the driver (here
    /// the config-only qwen2 directory fails reading its weights, as `ModelHandle::load` does), and the
    /// result line names the engine that generated.
    #[test]
    fn a_registered_family_opens_on_the_driver_and_the_line_names_the_engine() {
        let dir = config_only_checkpoint("registered", r#"{"model_type": "qwen2"}"#);
        let mut config = config(&["--precision", "bf16"]);
        config.weights = Weights::Safetensors(dir.display().to_string());
        let error = open(&config).map(|_| ()).unwrap_err().error;
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            error.chain().any(|cause| matches!(
                cause.downcast_ref::<DriverError>(),
                Some(DriverError::Load(_))
            )),
            "{error:#}"
        );

        let line = identity(
            Some(Engine::Driver),
            Some("qwen2"),
            Backend::Wgpu,
            Mode::Single,
            "bf16",
        );
        assert_eq!(line["engine"], "driver");
        assert_eq!(line["family"], "qwen2");
        let declined = identity(None, None, Backend::Wgpu, Mode::Single, "bf16");
        assert!(declined.get("engine").is_none());
        assert_eq!(declined["family"], Value::Null);
    }

    /// A family or shape the driver cannot serve is a recorded refusal (`unsupported`); a load failure
    /// is still an error.
    #[test]
    fn a_driver_refusal_is_a_recorded_refusal() {
        let unregistered =
            DriverError::Unsupported(poot_llm::driver::error::Unsupported::Backend {
                backend: "rocm",
            });
        let error = anyhow::Error::from(unregistered).context("load /m");
        assert!(refusal(&error).is_some_and(
            |(engine, reason)| engine == Some(Engine::Driver) && reason.contains("rocm")
        ));

        // A refusal after the registry resolved the family records that family: the feature matrix joins
        // the row to the family's compile cell.
        let config = config(&["--precision", "bf16"]);
        let line = result_line(&config, Some("qwen2"), Err(error)).unwrap();
        assert_eq!(line["status"], "unsupported");
        assert_eq!(line["family"], "qwen2");
    }

    #[test]
    fn median_and_stdev() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
        assert!((stdev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) - 2.138_089_935).abs() < 1e-8);
        assert_eq!(stdev(&[5.0]), 0.0);
    }
}

#[cfg(test)]
mod request_timing_tests {
    use super::*;

    fn request(stamps: &[u64], complete: u64) -> Run {
        let mut times = stamps.iter().copied().chain([complete]);
        timed_with_clock(
            |sink| {
                for _ in stamps {
                    sink("t");
                }
                Ok(())
            },
            || Duration::from_millis(times.next().unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn request_windows_account_for_setup_tokens_and_delayed_finish() {
        let run = request(&[40, 50, 65], 72);
        assert_eq!(run.first_token_ms, Some(40.0));
        assert_eq!(run.last_token_ms, Some(65.0));
        assert_eq!(run.itl_ms, [10.0, 15.0]);
        assert_eq!(run.finish_ms, 7.0);
        assert_eq!(
            run.e2e_ms,
            run.ttft_ms + run.itl_ms.iter().sum::<f64>() + run.finish_ms
        );
        let delayed = request(&[40, 50, 65], 172);
        assert_eq!(delayed.tpot_ms(), Some(12.5));
        assert_eq!(delayed.decode_tok_s(), Some(80.0));
        assert_eq!(delayed.finish_ms, 107.0);
        assert_eq!(delayed.e2e_ms - run.e2e_ms, 100.0);
        let relocated = request(&[60, 70, 85], 92);
        assert_eq!(relocated.e2e_ms - run.e2e_ms, 20.0);
    }

    #[test]
    fn a_delayed_token_changes_that_gap_and_only_that_gap() {
        let run = request(&[40, 50, 65], 72);
        let late = request(&[40, 50, 85], 92);
        assert_eq!(run.itl_ms, [10.0, 15.0]);
        assert_eq!(late.itl_ms, [10.0, 35.0]);
        assert_eq!(late.finish_ms, run.finish_ms);
        assert_eq!(late.e2e_ms - run.e2e_ms, 20.0);
    }

    #[test]
    fn zero_and_one_token_windows_are_defined() {
        let zero = request(&[], 25);
        assert_eq!((zero.first_token_ms, zero.last_token_ms), (None, None));
        assert_eq!(
            (zero.ttft_ms, zero.finish_ms, zero.e2e_ms),
            (0.0, 25.0, 25.0)
        );
        let one = request(&[10], 25);
        assert_eq!((one.ttft_ms, one.finish_ms, one.e2e_ms), (10.0, 15.0, 25.0));
        for run in [zero, one] {
            assert_eq!((run.tpot_ms(), run.decode_tok_s()), (None, None));
            assert_eq!(run.receipt()["tpot_ms"], Value::Null);
            assert!(run.itl_ms.is_empty());
        }
    }
}

#[cfg(test)]
mod preparation_tests {
    use super::*;
    use std::cell::Cell;

    /// An engine whose counters move the way a driver's do: `cold_work` on the first request, then
    /// `late_work` on every request (a pipeline built, or a recording redone, inside the timed window).
    struct Counting {
        requests: Cell<u64>,
        counters: Cell<PreparationCounters>,
        cold_work: u64,
        late_work: u64,
        /// Recordings redone on every later request while the plan layer stays quiet.
        late_recordings: u64,
    }

    impl Counting {
        fn new(cold_work: u64, late_work: u64) -> Self {
            Self {
                requests: Cell::new(0),
                counters: Cell::new(PreparationCounters {
                    plan: Some(0),
                    entries: Some(0),
                    recordings: Some(0),
                    ..PreparationCounters::default()
                }),
                cold_work,
                late_work,
                late_recordings: 0,
            }
        }

        /// The engine prepared `work` units while opening, before any request (as a driver does).
        fn prepared_while_opening(self, work: u64) -> Self {
            let mut counters = self.counters.get();
            counters.plan = Some(work);
            counters.recordings = Some(work);
            self.counters.set(counters);
            self
        }

        fn redoing_recordings(mut self, per_request: u64) -> Self {
            self.late_recordings = per_request;
            self
        }

        fn request(&self, max_new: usize, on_token: &mut dyn FnMut(&str)) {
            let first = self.requests.get() == 0;
            self.requests.set(self.requests.get() + 1);
            let work = if first {
                self.cold_work
            } else {
                self.late_work
            };
            let mut counters = self.counters.get();
            let redone = if first { 0 } else { self.late_recordings };
            counters.plan = counters.plan.map(|n| n + work);
            counters.recordings = counters.recordings.map(|n| n + work + redone);
            self.counters.set(counters);
            for _ in 0..max_new {
                std::thread::sleep(Duration::from_millis(1));
                on_token("t");
            }
        }
    }

    impl Generate for Counting {
        fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            Ok((0..text.split_whitespace().count() as u32).collect())
        }
        fn generate_context(
            &mut self,
            _: &[u32],
            max_new: usize,
            on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            self.request(max_new, on_token);
            Ok(())
        }
        fn generate_prompt(
            &mut self,
            _: &str,
            max_new: usize,
            _: bool,
            on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            self.request(max_new, on_token);
            Ok(())
        }
        fn reset_profile(&mut self) {}
        fn profile_report(&self, _: usize, _: Duration) -> Option<Profile> {
            None
        }
        fn engine(&self) -> Engine {
            Engine::Driver
        }
        fn steps(&self) -> u64 {
            self.requests.get() * 3
        }
        fn counters(&self) -> PreparationCounters {
            self.counters.get()
        }
        fn initial_counters(&self) -> PreparationCounters {
            PreparationCounters {
                plan: Some(0),
                entries: Some(0),
                recordings: Some(0),
                ..PreparationCounters::default()
            }
        }
    }

    fn config(extra: &[&str]) -> Config {
        let mut args = vec!["poot-bench-runner", "--model-dir", "/nonexistent"];
        args.extend_from_slice(extra);
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        match parse_args(&args).expect("test arguments parse") {
            Invocation::Run(config) => config,
            Invocation::Version => unreachable!("no --version"),
        }
    }

    fn layer(line: &Value, name: &str) -> (Value, Value) {
        let entry = &line["preparation"]["layers"][name];
        (entry["prepare"].clone(), entry["repeat"].clone())
    }

    #[test]
    fn preparation_separates_what_the_set_cost_from_what_repeating_it_added() {
        let config = config(&["--warmup", "2", "--iters", "3", "--gen-tokens", "3"]);
        let line = measure(&config, &mut Counting::new(4, 0), "bf16", "f").unwrap();

        assert_eq!(layer(&line, "plan"), (json!(4), json!(0)));
        assert_eq!(layer(&line, "recordings"), (json!(4), json!(0)));
        // No producer counts these: Unknown (null), never a zero.
        for unknown in [
            "generated_bodies",
            "artifact_hits",
            "runtime_pipelines",
            "graph_instantiations",
        ] {
            assert_eq!(
                layer(&line, unknown),
                (Value::Null, Value::Null),
                "{unknown}"
            );
        }
    }

    #[test]
    fn work_inside_the_timed_window_is_counted_as_repeat_work() {
        let config = config(&["--warmup", "1", "--iters", "3", "--gen-tokens", "3"]);
        let line = measure(&config, &mut Counting::new(4, 1), "bf16", "f").unwrap();

        // Three timed requests each redo one unit: the layer is not warm.
        assert_eq!(layer(&line, "plan"), (json!(4), json!(3)));
    }

    #[test]
    fn a_cold_process_counts_the_preparation_the_engine_did_while_opening() {
        let args = ["--warmup", "1", "--iters", "2", "--gen-tokens", "3"];
        let cold = config(&args);
        let mut reused = config(&args);
        reused.prepared_state = Some(PreparedState::ReusedPreparedDriver);

        let cold_line = measure(
            &cold,
            &mut Counting::new(0, 0).prepared_while_opening(5),
            "bf16",
            "f",
        )
        .unwrap();
        let reused_line = measure(
            &reused,
            &mut Counting::new(0, 0).prepared_while_opening(5),
            "bf16",
            "f",
        )
        .unwrap();

        // Prepared during open: the cold process paid it; a reused driver, prepared already, adds none.
        assert_eq!(layer(&cold_line, "plan"), (json!(5), json!(0)));
        assert_eq!(layer(&reused_line, "plan"), (json!(0), json!(0)));
    }

    #[test]
    fn each_driver_stat_lands_in_its_own_layer() {
        let stats = DriverStats {
            compiles: 7,
            entries_added: 11,
            steps: 17,
            host_picks: 19,
            ..DriverStats::default()
        };
        let counters = counters_from(&stats, 13);

        assert_eq!(counters.plan, Some(7));
        assert_eq!(counters.entries, Some(11));
        assert_eq!(counters.recordings, Some(13));
        // Steps and host picks are not preparation, and no producer counts the other layers.
        assert_eq!(
            counters
                .layers()
                .iter()
                .filter(|layer| layer.is_some())
                .count(),
            3
        );
    }

    #[test]
    fn each_layer_is_held_to_its_own_counter() {
        let config = config(&["--warmup", "1", "--iters", "3", "--gen-tokens", "3"]);
        // Recordings are redone in every timed request while the plan layer does no work: a plan counter
        // that stood for every layer would call this driver warm.
        let line = measure(
            &config,
            &mut Counting::new(4, 0).redoing_recordings(1),
            "bf16",
            "f",
        )
        .unwrap();

        assert_eq!(layer(&line, "plan"), (json!(4), json!(0)));
        assert_eq!(layer(&line, "recordings"), (json!(4), json!(3)));
    }

    #[test]
    fn a_reused_driver_is_prepared_before_the_counted_pass() {
        let cold = config(&[
            "--warmup",
            "1",
            "--iters",
            "2",
            "--gen-tokens",
            "3",
            "--prepared-state",
            "cold-process-populated-artifacts",
        ]);
        let reused = config(&[
            "--warmup",
            "1",
            "--iters",
            "2",
            "--gen-tokens",
            "3",
            "--prepared-state",
            "reused-prepared-driver",
        ]);

        let cold_line = measure(&cold, &mut Counting::new(4, 0), "bf16", "f").unwrap();
        let reused_line = measure(&reused, &mut Counting::new(4, 0), "bf16", "f").unwrap();

        assert_eq!(layer(&cold_line, "plan").0, json!(4));
        assert_eq!(layer(&reused_line, "plan").0, json!(0));
        assert_eq!(
            reused_line["preparation"]["state"],
            "reused-prepared-driver"
        );
        assert_eq!(reused_line["runs"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn curve_preparation_sums_every_contexts_window() {
        let config = config(&[
            "--mode",
            "decode-curve",
            "--isl-list",
            "4,8",
            "--osl",
            "3",
            "--warmup",
            "1",
            "--iters",
            "2",
        ]);
        // Each context's first request is cold in this engine only once; later requests add nothing.
        let line = measure(&config, &mut Counting::new(2, 0), "bf16", "f").unwrap();
        assert_eq!(layer(&line, "plan"), (json!(2), json!(0)));
    }

    #[test]
    fn the_prepared_state_flag_takes_only_the_declared_states() {
        for state in PreparedState::ALL {
            assert_eq!(PreparedState::parse(state.label()).unwrap(), state);
        }
        let error = PreparedState::parse("warm-ish").unwrap_err().to_string();
        assert!(error.contains("cold-process-empty-artifacts"), "{error}");
        assert!(config(&[]).prepared_state.is_none());
    }

    /// The labels and layer names the runner prints are the ones the harness validates against.
    #[test]
    fn the_harness_and_the_runner_agree_on_states_and_layers() {
        let harness = format!("{}/../../harness", env!("CARGO_MANIFEST_DIR"));
        let script = format!(
            "import json, sys\nsys.path.insert(0, {harness:?})\nimport performance_receipt as r\n\
             print(json.dumps([list(r.PREPARATION_STATES), list(r.PREPARATION_LAYERS)]))\n"
        );
        let output = std::process::Command::new("python3")
            .args(["-c", &script])
            .output()
            .expect("python3 is on PATH in the dev shell");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let harness: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            harness[0],
            json!(PreparedState::ALL.map(PreparedState::label))
        );
        assert_eq!(harness[1], json!(PREPARATION_LAYERS));
    }
}
