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
//! `--precision` names what the caller expects the checkpoint to store (`bf16`, `f16`, `q8_0`, ...). The
//! result's `precision` is what the driver's store holds (the dtype most of its bytes are stored in), read
//! after the load, never the request echoed back: a request the store does not hold fails the cell. The
//! engine computes in `COMPUTE_DTYPE`, reported as `compute_dtype`.
//!
//! When the runner refuses a request (see `refusal`: an architecture or shape the loader or planner does
//! not serve), the line reports `"status":"unsupported"` with the refusal's own text as `reason`; any
//! other failure exits non-zero. Every result names its build (`build_sha`, `build_profile`,
//! `build_features`, `rustc_version`); a measurement also names the adapter and driver it ran on and the
//! digest of the checkpoint's stored weights. The runner measures only when built with the release profile.
//! Built with a plain `cargo build --release -p poot-bench-runner` (kernels are generated at runtime).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::{json, Value};

use poot_executor::DetailedTiming;
use poot_graph_plan::CompileOptions;
use poot_llm::driver::error::DriverError;
use poot_llm::driver::{
    open_executor, open_timed_executor, BackendChoice, Driver, DriverOptions, DriverStats,
    GenerateRequest, Head, Layout, ModelHandle, PhaseTiming, PreparedSetLimits, Serving,
    ServingShapes, State, Warm,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::Registry;
use poot_profile::{Ceiling, Fit, GB};
use poot_quant::weights::{WeightEntry, WeightStore};

/// The git sha this binary was built from (`build.rs`); the harness records it with every result.
const BUILD_SHA: &str = env!("POOT_BUILD_SHA");
/// The cargo profile (`release` or `debug`), the runner's enabled features (comma-separated) and the compiler
/// this binary was built with (`build.rs`); recorded with every result beside the sha.
const BUILD_PROFILE: &str = env!("POOT_BUILD_PROFILE");
const BUILD_FEATURES: &str = env!("POOT_BUILD_FEATURES");
const RUSTC_VERSION: &str = env!("POOT_RUSTC_VERSION");

/// The dtype the engine computes in whatever the weights are stored in: matmuls accumulate in f32 and the
/// activations are f32. Reported beside the resident weight dtype so a row never reads a stored dtype as a
/// compute one.
const COMPUTE_DTYPE: &str = "f32";

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
    /// Report each cell's roofline: the driver then times its phases, which a plain cell does not pay for.
    roofline: bool,
    mode: Mode,
    isl_list: Vec<usize>,
    osl: usize,
    /// Decode-curve mode: the longest slice of a context one prefill forward takes. A longer context is
    /// prefilled in several forwards of at most this many tokens (the driver's chunk plan); `None` takes each
    /// context whole. A typed driver option, recorded in the result: a device whose watchdog kills a
    /// dispatch that runs too long needs it to bound the per-dispatch work.
    prefill_chunk: Option<usize>,
    prepared_state: Option<PreparedState>,
    /// Artifacts the cache directory held before this process opened anything (`None`: no directory named).
    artifact_entries_at_start: Option<u64>,
}

enum Invocation {
    Version,
    /// `--roofline-fit <result lines...>`: the cross-model fit over lines earlier runs printed.
    RooflineFit(Vec<String>),
    Run(Config),
}

/// Options that take a value and switches that do not. Everything else is an error: a flag the runner
/// ignores would report a measurement the caller did not ask for. `--synthetic` is part of the suite's
/// runner contract (the harness passes it to every runner); decode-curve contexts are always synthetic.
const VALUE_FLAGS: [&str; 14] = [
    "--roofline-fit",
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
    "--prefill-chunk",
    "--prepared-state",
];
const SWITCHES: [&str; 6] = [
    "--roofline",
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
    if let Some(files) = values.get("--roofline-fit") {
        return Ok(Invocation::RooflineFit(
            files.split(',').map(str::to_string).collect(),
        ));
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
        roofline: switches.contains("--roofline"),
        mode: Mode::parse(values.get("--mode").copied().unwrap_or("single"))?,
        isl_list,
        osl: number("--osl", 128)?,
        prefill_chunk: values
            .get("--prefill-chunk")
            .map(|_| number("--prefill-chunk", 0))
            .transpose()?,
        prepared_state: values
            .get("--prepared-state")
            .map(|name| PreparedState::parse(name))
            .transpose()?,
        artifact_entries_at_start: artifact_cache_entries(),
    };

    anyhow::ensure!(config.iters > 0, "--iters must be positive");
    anyhow::ensure!(
        config.prefill_chunk.is_none_or(|chunk| chunk > 0),
        "--prefill-chunk must be positive"
    );
    anyhow::ensure!(
        config.prefill_chunk.is_none() || config.mode == Mode::DecodeCurve,
        "--prefill-chunk applies to --mode decode-curve only"
    );
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

    /// The decode work this engine has done so far, as it counted it: tokens, its own step-clock wall
    /// and the bytes its plan read ([`PhaseTiming`]). A window's roofline is the difference of two
    /// readings. Zero for an engine that counts none.
    fn decode_phase(&self) -> PhaseTiming {
        PhaseTiming::default()
    }

    /// The pinned bandwidth ceiling of the device this engine runs on; `None` for a device nobody
    /// calibrated.
    fn ceiling(&self) -> Option<Ceiling> {
        None
    }

    /// What generates.
    fn engine(&self) -> Engine;
}

/// `ms_per_token = a * GB_per_token + b` over the measured rooflines in `rooflines` (a decode-curve's
/// points, or the single-mode lines of a set of models): how the decode step's time grows with the bytes
/// it reads, as the engine produced both axes ([`Fit::over`]). The typed `unknown` below two measured
/// points or with no spread in bytes.
fn roofline_fit<'a>(rooflines: impl IntoIterator<Item = &'a Value>) -> Value {
    let points: Vec<(f64, f64)> = rooflines
        .into_iter()
        .filter_map(|roofline| {
            Some((
                roofline["bytes_per_token"].as_f64()? / GB,
                roofline["ms_per_token"].as_f64()?,
            ))
        })
        .collect();
    match Fit::over(&points) {
        Some(fit) => json!({
            "kind": "measured",
            "points": points.len(),
            "ms_per_gb": fit.ms_per_gb,
            "fixed_ms": fit.fixed_ms,
            "r2": fit.r2,
        }),
        None => json!({
            "kind": "unknown",
            "reason": "fewer than two measured points or no spread in bytes",
        }),
    }
}

/// The cross-model fit over result lines earlier runs printed (`--roofline-fit a.json,b.json`): each
/// file's last line is one result line, whose `roofline` is the model's measured decode window.
fn fit_over_result_files(files: &[String]) -> anyhow::Result<Value> {
    let mut rooflines = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(file).with_context(|| format!("read {file}"))?;
        let line = text
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .with_context(|| format!("{file} holds no result line"))?;
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("{file}: the last line is not a result line"))?;
        rooflines.push(value["roofline"].clone());
    }
    Ok(roofline_fit(&rooflines))
}

/// The roofline of the decode work between two readings of [`Generate::decode_phase`], as the result
/// line carries it: the engine's own `ms_per_token` and bytes per token, and the fraction of the
/// device's pinned ceiling those amount to. The typed `unknown` of a window with no decoded token, no
/// byte count or an uncalibrated device: never a zero and never a figure from another clock.
fn roofline_receipt(engine: &dyn Generate, before: PhaseTiming) -> Value {
    let window = engine.decode_phase() - before;
    let Some(ceiling) = engine.ceiling() else {
        return json!({ "kind": "unknown", "reason": "no pinned ceiling for this device" });
    };
    match window.roofline(ceiling) {
        Some(roofline) => json!({
            "kind": "measured",
            "decode_tokens": window.tokens,
            "ms_per_token": roofline.ms_per_token,
            "bytes_per_token": roofline.bytes_per_token,
            "achieved_gb_per_s": roofline.achieved_gb_per_s,
            "fraction_of_ceiling": roofline.fraction_of_ceiling,
            "ceiling_gb_per_s": roofline.ceiling.gb_per_s,
            "ceiling_target": roofline.ceiling.target,
            "ceiling_receipt": roofline.ceiling.receipt,
        }),
        None => json!({ "kind": "unknown", "reason": "no decoded token or no measured time" }),
    }
}

/// The driver engine on one backend (Card 734): [`Driver::generate`] behind the same [`Generate`] seam, so
/// the measure loop never branches on the engine. One `Driver` serves the whole cell: its KV holds the
/// longest request the cell's mode makes, and its prefill chunk is that mode's (the whole prompt for
/// `--prefill`, one token for token by token, the longest context for decode-curve, where a shorter
/// context plans as the descending powers of two its chunk plan admits). A second `Driver` would open a
/// second executor after dropping the first, which the ROCm runtime does not survive in one process.
struct DriverCell {
    handle: Arc<ModelHandle>,
    driver: Driver<Serving>,
    /// The driver's counters before `Driver::prepare` ran.
    initial: PreparationCounters,
    /// The executor's timing snapshot when `reset_profile` last ran: the start of the profiled window.
    mark: Option<poot_profile::Mark>,
    /// The decode phases of every generation so far, as the driver timed them.
    decode: PhaseTiming,
}

/// The preparation layers the driver and its executor count.
fn driver_counters<S: State>(driver: &Driver<S>) -> PreparationCounters {
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

/// The prefill chunk a decode-curve cell opens its driver with: the longest context whole, or `limit` tokens
/// when the caller bounds one forward (a context shorter than `limit` is still prefilled whole).
fn curve_prefill_chunk(limit: Option<usize>, longest: usize) -> usize {
    limit.map_or(longest, |limit| limit.min(longest))
}

fn driver_options(capacity: NonZeroUsize, chunk: usize) -> anyhow::Result<DriverOptions> {
    let compile = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    Ok(DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity,
        prefill_chunk: NonZeroUsize::new(chunk).context("a prefill chunk holds a token")?,
        trace: poot_llm::driver::serving_trace_limits(capacity),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(8 << 30).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
        draft: poot_llm::driver::DraftPolicy::Off,
        hidden_window: None,
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
        let generation = self.driver.generate(request, &mut |_: u32, piece: &str| {
            on_token(piece);
            GenerationControl::Continue(())
        })?;
        if let Some(timing) = generation.timing {
            self.decode = self.decode + timing.decode;
        }
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

    fn decode_phase(&self) -> PhaseTiming {
        self.decode
    }

    fn ceiling(&self) -> Option<Ceiling> {
        self.driver.ceiling()
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

/// A stored weight's dtype as a precision name: the checkpoint dtype of a dense tensor (`bf16`), the format of
/// a packed one without its parameters (`q8_0`, `e4m3block128`).
fn stored_name(entry: &WeightEntry) -> String {
    match entry {
        WeightEntry::Dense(dense) => dense.dtype().to_string(),
        WeightEntry::Packed(payload) => {
            let format = format!("{:?}", payload.weight().format());
            format
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_lowercase()
        }
    }
}

/// The precision the store holds: the dtype most of its stored bytes are in.
fn resident_precision(store: &WeightStore) -> anyhow::Result<String> {
    let mut bytes = BTreeMap::<String, usize>::new();
    for (_, entry) in store.iter() {
        *bytes.entry(stored_name(entry)).or_default() += entry.stored_bytes();
    }
    bytes
        .into_iter()
        .max_by_key(|&(_, stored)| stored)
        .map(|(name, _)| name)
        .ok_or_else(|| anyhow::anyhow!("the store holds no weight"))
}

/// The cell's precision is the store's: a request for another dtype is the caller's mistake, and reporting the
/// request back would label the measurement with a precision it did not run.
fn ensure_store_holds(requested: &str, store: &WeightStore) -> anyhow::Result<String> {
    let resident = resident_precision(store)?;
    anyhow::ensure!(
        resident == requested,
        "--precision {requested} but the checkpoint's store holds {resident}"
    );
    Ok(resident)
}

/// A digest of the checkpoint's content as the driver holds it: every weight's name, shape, stored dtype and
/// stored bytes in key order. Two checkpoints with one name but different weights digest differently.
fn checkpoint_digest(store: &WeightStore) -> String {
    let mut hasher = blake3::Hasher::new();
    for (key, entry) in store.iter() {
        hasher.update(key.to_string().as_bytes());
        hasher.update(format!("{:?}", entry.shape()).as_bytes());
        hasher.update(stored_name(entry).as_bytes());
        match entry {
            WeightEntry::Dense(dense) => {
                hasher.update(dense.bytes().as_slice());
            }
            WeightEntry::Packed(payload) => {
                for role in payload.weight().sources() {
                    hasher.update(payload.bytes(role));
                }
            }
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// The driver engine over a loaded registered family: size every driver's KV to the longest request the
/// mode makes, and name what the cell ran on: the store's precision, the adapter and driver version the
/// executor reports, and the checkpoint's digest.
fn open_driver(
    config: &Config,
    handle: ModelHandle,
    family: &str,
) -> anyhow::Result<(Box<dyn Generate>, Resolved)> {
    let handle = Arc::new(handle);
    // Before an executor opens: a request the checkpoint does not hold needs no device.
    let precision = ensure_store_holds(&config.precision, handle.store())?;
    let checkpoint_digest = checkpoint_digest(handle.store());
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
            (
                isl + config.osl,
                curve_prefill_chunk(config.prefill_chunk, isl),
                config.isl_list.clone(),
            )
        }
    };
    let capacity = NonZeroUsize::new(longest).context("the longest request is empty")?;
    let executor = if config.profile {
        open_timed_executor(backend, DetailedTiming::default_diagnostic())
    } else {
        open_executor(backend)
    }
    .context("open the executor")?;
    let device = executor
        .device_identity()
        .context("the device cannot name its adapter and driver")?;
    let driver = Driver::new(
        Arc::clone(&handle),
        executor,
        driver_options(capacity, chunk)?,
    )
    .context("open the driver")?;
    let driver = if config.roofline {
        driver.with_phase_timing()
    } else {
        driver
    };
    let initial = driver_counters(&driver);
    // Every entry the cell's prompts plan to is compiled and added before the first timed token.
    let driver = driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&prompts),
            windows: &[],
            adapters: &[],
        })
        .map_err(DriverError::from)
        .context("prepare the driver")?;
    Ok((
        Box::new(DriverCell {
            handle,
            driver,
            initial,
            mark: None,
            decode: PhaseTiming::default(),
        }),
        Resolved {
            family: family.to_string(),
            precision,
            adapter: device.adapter,
            driver_version: device.driver,
            checkpoint_digest,
        },
    ))
}

/// What an opened cell ran on, as the opened driver reports it rather than as the request named it: the
/// registry family the checkpoint resolved to, the precision its store holds, the adapter and driver version
/// of the device and the digest of the weights it loaded.
struct Resolved {
    family: String,
    precision: String,
    adapter: String,
    driver_version: String,
    checkpoint_digest: String,
}

/// A cell that opened: the engine and what it ran on.
struct Opened {
    engine: Box<dyn Generate>,
    resolved: Resolved,
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
    match open_driver(config, handle, &family) {
        Ok((engine, resolved)) => Ok(Opened { engine, resolved }),
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
    resolved: &Resolved,
) -> anyhow::Result<Value> {
    let mut probe = PreparationProbe::new(config.prepared_state, config.artifact_entries_at_start);
    let body = match config.mode {
        Mode::Single => measure_single(config, engine, &mut probe)?,
        Mode::DecodeCurve => measure_curve(config, engine, &mut probe)?,
    };
    let mut line = identity(
        Some(engine.engine()),
        Some(&resolved.family),
        config.backend,
        config.mode,
        &resolved.precision,
    );
    line["compute_dtype"] = json!(COMPUTE_DTYPE);
    line["adapter"] = json!(resolved.adapter);
    line["driver_version"] = json!(resolved.driver_version);
    line["checkpoint_digest"] = json!(resolved.checkpoint_digest);
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

/// What every result line carries: which build produced it (sha, profile, features, compiler), with which
/// engine (when one was chosen), on which backend, in which mode. The harness checks `backend` against the one
/// it requested and refuses a result with no build identity. `family` is the registry family the checkpoint
/// resolved to, as the registry reports it (never a mapping from the manifest); `null` on a refusal that came
/// before one resolved. A measurement adds what only an opened cell knows (see [`measure`]).
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
        "build_profile": BUILD_PROFILE,
        "build_features": BUILD_FEATURES,
        "rustc_version": RUSTC_VERSION,
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
    let decode_before = engine.decode_phase();
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
    let roofline = config
        .roofline
        .then(|| roofline_receipt(&*engine, decode_before));

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
        "roofline": roofline,
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
    let decode_before = engine.decode_phase();
    let timed_start = Instant::now();
    let mut iters = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        let run = run_once(engine)?;
        iters.push(run.receipt());
    }
    probe.repeated(&*engine);
    let roofline = config
        .roofline
        .then(|| roofline_receipt(&*engine, decode_before));
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
        "roofline": roofline,
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
    let fit = config
        .roofline
        .then(|| roofline_fit(curve.iter().map(|point| &point["roofline"])));
    Ok(json!({
        "osl": config.osl,
        "prefill_chunk": config.prefill_chunk,
        "caveats": caveats,
        "curve": curve,
        "roofline_fit": fit,
    }))
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

/// The refusal a failure carries, if it is one, as its own text. This is the runner's only classification
/// of "poot does not run this": the driver's [`DriverError::Unsupported`] (a family the registry does not
/// hold, a shape the family or the planner refuses), found anywhere in the error chain.
fn refusal(error: &anyhow::Error) -> Option<(Option<Engine>, String)> {
    error
        .chain()
        .find_map(|cause| match cause.downcast_ref::<DriverError>() {
            Some(DriverError::Unsupported(_)) => Some((Some(Engine::Driver), cause.to_string())),
            _ => None,
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
            let outcome =
                measure(config, opened.engine.as_mut(), &opened.resolved).map(|mut line| {
                    // Loading is timed apart from every request: it happens before the first one.
                    line["preparation"]["open_ms"] = json!(open_ms);
                    line
                });
            (Some(opened.resolved.family), outcome)
        }
        Err(failure) => (failure.family, Err(failure.error)),
    };
    result_line(config, family.as_deref(), outcome)
}

/// Why a binary built with `profile` is not one to measure, or `None` for a release build. A debug build
/// reports times of code nobody ships; the harness would record them as a result, so the runner refuses
/// before it opens anything.
fn non_release_build(profile: &str, debug_assertions: bool) -> Option<String> {
    if profile != "release" {
        return Some(format!(
            "this runner is a {profile} build; benchmark results come from `cargo build --release` only"
        ));
    }
    debug_assertions.then(|| {
        "this runner is built with debug assertions; benchmark results come from `cargo build --release` only"
            .to_string()
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = match parse_args(&args)? {
        Invocation::RooflineFit(files) => {
            println!("{}", fit_over_result_files(&files)?);
            return Ok(());
        }
        Invocation::Version => {
            println!("poot-bench-runner {BUILD_SHA}");
            return Ok(());
        }
        Invocation::Run(config) => config,
    };
    if let Some(reason) = non_release_build(BUILD_PROFILE, cfg!(debug_assertions)) {
        anyhow::bail!(reason);
    }
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
mod stub_cell {
    use super::Resolved;

    /// What a stub engine's cell ran on: a fixed store precision, device and digest.
    pub(super) fn resolved(family: &str) -> Resolved {
        Resolved {
            family: family.to_string(),
            precision: "bf16".to_string(),
            adapter: "stub adapter".to_string(),
            driver_version: "stub driver 1.0".to_string(),
            checkpoint_digest: format!("blake3:{}", "0".repeat(64)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Write;
    use std::process::{Command, Stdio};

    use poot_llm::driver::error::Unsupported;

    use super::stub_cell::resolved;
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
            Invocation::Version | Invocation::RooflineFit(_) => unreachable!("no --version"),
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
            detail: poot_runtime_common::Coverage::Unavailable,
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
            let line = measure(&single, &mut stub, &resolved("stub_family")).unwrap();
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
            let line = measure(&curve, &mut stub, &resolved("stub_family")).unwrap();
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

    /// `line` as a release runner would print it. These tests run in a dev-profile binary, which the harness's
    /// contract refuses as a result: the line must name that profile (it does), and only the profile is then
    /// stated as the release one so the rest of the contract is judged. That a debug runner measures nothing is
    /// `only_a_release_build_without_debug_assertions_measures` and `tests/release_only.rs`.
    fn as_release_line(line: &Value) -> Value {
        assert_eq!(line["build_profile"], BUILD_PROFILE);
        let mut release = line.clone();
        release["build_profile"] = json!("release");
        release
    }

    /// Run the harness's own runner-result validation over `line`, as the harness would before it turns
    /// the line into a row. The runner and the harness must agree on one contract, and the harness owns it.
    fn harness_accepts(line: &Value, curve: bool, gen_tokens: usize, backend: Backend) {
        let line = &as_release_line(line);
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
            let outcome = measure(&config, &mut Stub::new(), &resolved("stub_family"));
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
                let outcome = measure(&config, &mut Stub::new(), &resolved("stub_family"));
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
        let line = &as_release_line(line);
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
            measure(&config, &mut stub, &resolved("stub_family")),
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
            measure(&config, &mut stub, &resolved("stub_family")),
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

    /// A tiny qwen2 checkpoint (config, word-level tokenizer, BF16 weights) in a fresh temp directory.
    fn tiny_bf16_checkpoint(name: &str) -> std::path::PathBuf {
        use poot_models::test_support::weight_map::{qwen2, Projections, DIMS};
        use tokenizers::models::wordlevel::WordLevel;

        let dir = config_only_checkpoint(
            name,
            &json!({
                "model_type": "qwen2",
                "vocab_size": DIMS.vocab,
                "hidden_size": DIMS.hidden,
                "intermediate_size": DIMS.inter,
                "num_hidden_layers": DIMS.layers,
                "num_attention_heads": DIMS.heads,
                "num_key_value_heads": DIMS.kv_heads,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "eos_token_id": DIMS.vocab - 1,
                "bos_token_id": 1,
            })
            .to_string(),
        );
        let vocab = (0..DIMS.vocab)
            .map(|id| (format!("w{id}"), id as u32))
            .collect();
        let tokenizer = tokenizers::Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("w0".to_string())
                .build()
                .unwrap(),
        );
        tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
        let mapped = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);
        poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &mapped.store);
        dir
    }

    /// SC-002 (real entry): the cell's precision is what the loaded store holds. A BF16 checkpoint asked for as
    /// F16 fails inside `open`, before any device opens, with both sides named; it is not a refusal, and no
    /// result line is printed for it. Mutation: report `config.precision` instead of the store's (the request
    /// echoed back): the F16 request opens and this goes red.
    #[test]
    fn a_request_the_checkpoint_does_not_hold_fails_before_a_device_opens() {
        let dir = tiny_bf16_checkpoint("store-precision");
        let mut config = config(&["--precision", "f16"]);
        config.weights = Weights::Safetensors(dir.display().to_string());
        let error = run(&config).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(
            format!("{error:#}").contains("--precision f16 but the checkpoint's store holds bf16"),
            "{error:#}"
        );
        assert!(
            refusal(&error).is_none(),
            "a precision disagreement is a failure, not a refusal"
        );
    }

    /// The store a Q8_0-projection checkpoint holds reports `q8_0`, so a `bf16` request against it fails and a
    /// `q8_0` request does not; a BF16 store reports `bf16`. The same function `open_driver` calls decides.
    #[test]
    fn the_store_names_its_dominant_dtype() {
        use poot_models::test_support::weight_map::{qwen2, Projections, DIMS};

        let bf16 = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);
        let q8_0 = qwen2(DIMS, Projections::Q8_0, poot_tensor::DType::BF16);

        assert_eq!(ensure_store_holds("bf16", &bf16.store).unwrap(), "bf16");
        assert_eq!(ensure_store_holds("q8_0", &q8_0.store).unwrap(), "q8_0");
        let error = ensure_store_holds("bf16", &q8_0.store).unwrap_err();
        assert_eq!(
            error.to_string(),
            "--precision bf16 but the checkpoint's store holds q8_0"
        );
    }

    /// The checkpoint digest follows the stored bytes: the same store digests the same, a store differing in
    /// one weight byte does not, and the digest names its algorithm.
    #[test]
    fn the_checkpoint_digest_follows_the_stored_bytes() {
        use poot_models::test_support::weight_map::{qwen2, Projections, DIMS};

        let bf16 = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);
        let f32_embed = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::F32);
        let again = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);

        let digest = checkpoint_digest(&bf16.store);
        assert!(digest.starts_with("blake3:") && digest.len() == "blake3:".len() + 64);
        assert_eq!(digest, checkpoint_digest(&again.store));
        assert_ne!(digest, checkpoint_digest(&f32_embed.store));
    }

    /// The runner measures release builds only: a debug profile, or debug assertions in a release one, is
    /// refused with the reason. Mutation: return `None` for every profile; the debug case goes red.
    #[test]
    fn only_a_release_build_without_debug_assertions_measures() {
        assert_eq!(non_release_build("release", false), None);
        assert!(non_release_build("debug", false).is_some_and(|why| why.contains("debug build")));
        assert!(
            non_release_build("release", true).is_some_and(|why| why.contains("debug assertions"))
        );
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
        let line = measure(&config, &mut stub, &resolved("stub_family")).unwrap();
        assert_eq!(line["curve"].as_array().unwrap().len(), 1);
        assert_eq!(
            line["caveats"],
            json!(["isl 16+ omitted: CUDA out of memory"])
        );

        stub.fail_at_isl = Some(8);
        stub.error = refused;
        let outcome = measure(&config, &mut stub, &resolved("stub_family"));
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
        let line = measure(&config, &mut Panics, &resolved("stub_family")).unwrap();
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
            Invocation::Version | Invocation::RooflineFit(_) => unreachable!(),
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
    use super::stub_cell::resolved;
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
            Invocation::Version | Invocation::RooflineFit(_) => unreachable!("no --version"),
        }
    }

    fn layer(line: &Value, name: &str) -> (Value, Value) {
        let entry = &line["preparation"]["layers"][name];
        (entry["prepare"].clone(), entry["repeat"].clone())
    }

    #[test]
    fn preparation_separates_what_the_set_cost_from_what_repeating_it_added() {
        let config = config(&["--warmup", "2", "--iters", "3", "--gen-tokens", "3"]);
        let line = measure(&config, &mut Counting::new(4, 0), &resolved("f")).unwrap();

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
        let line = measure(&config, &mut Counting::new(4, 1), &resolved("f")).unwrap();

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
            &resolved("f"),
        )
        .unwrap();
        let reused_line = measure(
            &reused,
            &mut Counting::new(0, 0).prepared_while_opening(5),
            &resolved("f"),
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
            &resolved("f"),
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

        let cold_line = measure(&cold, &mut Counting::new(4, 0), &resolved("f")).unwrap();
        let reused_line = measure(&reused, &mut Counting::new(4, 0), &resolved("f")).unwrap();

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
        let line = measure(&config, &mut Counting::new(2, 0), &resolved("f")).unwrap();
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

#[cfg(test)]
mod roofline_tests {
    use super::*;

    /// An engine with a settled decode window and (maybe) a pinned ceiling; nothing else is read.
    struct Timed {
        decode: PhaseTiming,
        ceiling: Option<Ceiling>,
    }

    impl Generate for Timed {
        fn encode(&self, _text: &str) -> anyhow::Result<Vec<u32>> {
            unreachable!("the roofline receipt reads the decode window only")
        }

        fn generate_context(
            &mut self,
            _context: &[u32],
            _max_new: usize,
            _on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            unreachable!("the roofline receipt reads the decode window only")
        }

        fn generate_prompt(
            &mut self,
            _prompt: &str,
            _max_new: usize,
            _prefill: bool,
            _on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<()> {
            unreachable!("the roofline receipt reads the decode window only")
        }

        fn reset_profile(&mut self) {}

        fn profile_report(&self, _tokens: usize, _wall: Duration) -> Option<Profile> {
            None
        }

        fn steps(&self) -> u64 {
            0
        }

        fn decode_phase(&self) -> PhaseTiming {
            self.decode
        }

        fn ceiling(&self) -> Option<Ceiling> {
            self.ceiling
        }

        fn engine(&self) -> Engine {
            Engine::Driver
        }
    }

    fn ceiling() -> Ceiling {
        Ceiling::pinned("amdgcn-gfx1151").expect("the checked-in table pins gfx1151")
    }

    fn window(tokens: usize, ms: u64, bytes: u64) -> PhaseTiming {
        PhaseTiming {
            tokens,
            wall: Duration::from_millis(ms),
            bytes,
        }
    }

    fn kind(value: &Value) -> &str {
        value["kind"].as_str().expect("a typed receipt")
    }

    /// A window the engine measured is `measured` with the engine's own figures; an uncalibrated
    /// device, a window with no decoded token and one with no bytes are each the typed `unknown`, never
    /// a zero. Mutation: report `measured` whenever a ceiling exists.
    #[test]
    fn the_roofline_receipt_is_measured_only_for_a_measured_window_on_a_calibrated_device() {
        // 10 tokens, 100 ms, 1 GB: 10 ms/token and 100 MB/token = 10 GB/s.
        let measured = Timed {
            decode: window(10, 100, 1_000_000_000),
            ceiling: Some(ceiling()),
        };
        let receipt = roofline_receipt(&measured, PhaseTiming::default());
        assert_eq!(kind(&receipt), "measured");
        assert_eq!(receipt["decode_tokens"], 10);
        assert!((receipt["ms_per_token"].as_f64().unwrap() - 10.0).abs() < 1e-9);
        assert!((receipt["achieved_gb_per_s"].as_f64().unwrap() - 10.0).abs() < 1e-9);

        let uncalibrated = Timed {
            ceiling: None,
            ..measured
        };
        assert_eq!(
            kind(&roofline_receipt(&uncalibrated, PhaseTiming::default())),
            "unknown"
        );
        // The window is the difference of two readings: nothing decoded since `before`.
        let before = window(10, 100, 1_000_000_000);
        assert_eq!(kind(&roofline_receipt(&measured, before)), "unknown");
        let no_bytes = Timed {
            decode: window(10, 100, 0),
            ceiling: Some(ceiling()),
        };
        assert_eq!(
            kind(&roofline_receipt(&no_bytes, PhaseTiming::default())),
            "unknown"
        );
    }

    /// The fit is measured over two or more spread points and unknown otherwise (one point, no spread in
    /// bytes, a point with no roofline). Mutation: report `measured` for a single point.
    #[test]
    fn the_fit_is_measured_over_spread_points_and_unknown_otherwise() {
        let point = |gb: f64, ms: f64| json!({ "bytes_per_token": gb * GB, "ms_per_token": ms });
        let line = [point(1.0, 5.0), point(2.0, 8.0), point(4.0, 14.0)];
        let fit = roofline_fit(&line);
        assert_eq!(kind(&fit), "measured");
        assert_eq!(fit["points"], 3);
        assert!((fit["ms_per_gb"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((fit["fixed_ms"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((fit["r2"].as_f64().unwrap() - 1.0).abs() < 1e-9);

        assert_eq!(kind(&roofline_fit(&line[..1])), "unknown");
        let flat = [point(1.0, 5.0), point(1.0, 6.0)];
        assert_eq!(kind(&roofline_fit(&flat)), "unknown");
        let unmeasured = [point(1.0, 5.0), json!({ "kind": "unknown" })];
        assert_eq!(kind(&roofline_fit(&unmeasured)), "unknown");
    }

    /// `--roofline-fit` reads each file's last result line and fits their rooflines: two models on a
    /// known line `ms = 3 * GB + 2` recover it. Mutation: read the first line of each file.
    #[test]
    fn the_fit_producer_reads_the_last_result_line_of_each_file() {
        let dir = std::env::temp_dir().join(format!("poot-roofline-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, gb: f64, ms: f64| {
            let path = dir.join(name);
            let line = json!({ "roofline": { "bytes_per_token": gb * GB, "ms_per_token": ms } });
            std::fs::write(&path, format!("{{\"log\": \"not a result\"}}\n{line}\n")).unwrap();
            path.display().to_string()
        };
        let files = [write("a.json", 1.0, 5.0), write("b.json", 3.0, 11.0)];
        let fit = fit_over_result_files(&files).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(kind(&fit), "measured");
        assert!((fit["ms_per_gb"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((fit["fixed_ms"].as_f64().unwrap() - 2.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod prefill_chunk_tests {
    use super::*;

    fn parse(extra: &[&str]) -> anyhow::Result<Invocation> {
        let mut args = vec!["poot-bench-runner", "--model-dir", "/nonexistent"];
        args.extend_from_slice(extra);
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        parse_args(&args)
    }

    #[test]
    fn a_bounded_curve_prefills_a_long_context_in_chunks_and_a_short_one_whole() {
        assert_eq!(curve_prefill_chunk(Some(512), 2048), 512);
        assert_eq!(curve_prefill_chunk(Some(4096), 2048), 2048);
        assert_eq!(curve_prefill_chunk(None, 2048), 2048);
    }

    #[test]
    fn the_prefill_chunk_flag_is_read_and_limited_to_decode_curve() {
        let curve = ["--mode", "decode-curve", "--isl-list", "128"];
        let mut with_chunk = curve.to_vec();
        with_chunk.extend(["--prefill-chunk", "512"]);
        match parse(&with_chunk).expect("a bounded curve parses") {
            Invocation::Run(config) => assert_eq!(config.prefill_chunk, Some(512)),
            _ => unreachable!("a run"),
        }
        assert!(parse(&["--prefill-chunk", "512"]).is_err(), "single mode");
        let mut zero = curve.to_vec();
        zero.extend(["--prefill-chunk", "0"]);
        assert!(parse(&zero).is_err(), "a chunk holds a token");
    }
}
