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
//! `--backend ptx|rocm|wgpu` picks the executor. A backend is a [`Generate`] implementation: it names which
//! `Runner` generation entry the backend uses and nothing else, so the measure loop never branches on it.
//! `--gguf <path>` (or a `.gguf` `--model-dir`) loads the quant-resident weights; that decode exists on
//! wgpu and rocm only.
//!
//! The registry decides what generates: a checkpoint of a family `Registry::builtin()` holds loads as a
//! `driver::ModelHandle` and runs on the one `poot_llm::driver::Driver` loop; a family it does not hold
//! (the MoE and hybrid families, POOT-738) loads as a `Runner` and runs on its entry points. The result
//! line records which (`engine`). The driver prefills a decode-curve context in the pieces of the chunk
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
use std::ops::ControlFlow;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::{json, Value};

use poot_executor::Executor as _;
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission};
use poot_llm::driver::error::{DriverError, Unsupported};
use poot_llm::driver::{
    open_executor, BackendChoice, Driver, DriverOptions, GenerateRequest, Head, Layout,
    ModelHandle, PreparedSetLimits, ServingShapes, Warm,
};
use poot_llm::{GenerationControl, Runner, RunnerError, Sampler};
use poot_models::registry::{Registry, RegistryError};
use poot_ptx_gpu::PtxDevice;
use poot_quant::weights::WeightEntry;
use poot_rocm_gpu::device::RocmDevice;

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

/// What generated a result: the Runner's entry points, or the driver's one loop. The registry decides
/// it from the checkpoint; it is recorded, never chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Runner,
    Driver,
}

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Engine::Runner => "runner",
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
}

enum Invocation {
    Version,
    Run(Config),
}

/// Options that take a value and switches that do not. Everything else is an error: a flag the runner
/// ignores would report a measurement the caller did not ask for. `--synthetic` is part of the suite's
/// runner contract (the harness passes it to every runner); decode-curve contexts are always synthetic.
const VALUE_FLAGS: [&str; 11] = [
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
    };

    anyhow::ensure!(config.iters > 0, "--iters must be positive");
    anyhow::ensure!(
        !(config.profile && config.backend == Backend::Rocm),
        "--profile is not supported on --backend rocm: RocmDevice has no HSA timestamp binding \
         yet (Card 548), so device_time() is always Unknown and there is nothing to \
         report - this used to silently print no profile section instead of refusing"
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

/// A loaded model on one backend, as the measure loop drives it. The methods are the `Runner` generation
/// entries the backend uses (poot-llm owns them); the loop is written once against this trait.
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

    fn profile_report(&self, tokens: usize, wall: Duration) -> Option<String>;

    /// What the engine itself counted, for the result line: the driver's prepared set, compile and
    /// retention counters. `None` for an engine with nothing of its own to report.
    fn engine_receipt(&self) -> Option<Value> {
        None
    }

    /// What generates.
    fn engine(&self) -> Engine {
        Engine::Runner
    }
}

fn keep_going(on_token: &mut dyn FnMut(&str)) -> impl FnMut(&str) -> ControlFlow<()> + '_ {
    move |piece| {
        on_token(piece);
        ControlFlow::Continue(())
    }
}

struct Ptx {
    runner: Runner,
    /// Card 549: the executor contract, used by every arm now that `PtxGraphExecutor` is deleted.
    exec: poot_executor::Engine<poot_ptx_gpu::PtxDevice>,
    exe: poot_executor::ExecutableId,
    /// Card 552: a mark taken at `reset_profile` time, against `exec`'s own typed `TimingSnapshot`
    /// (mirrors `Wgpu`'s identical field; `PtxDevice` carries no legacy label profiler to reset).
    mark: Option<poot_profile::Mark>,
}

impl Generate for Ptx {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.runner.encode(text)?)
    }

    fn generate_context(
        &mut self,
        context: &[u32],
        max_new: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        self.runner.generate_kv_ptx_prefilled_tokens(
            context,
            max_new,
            &mut self.exec,
            self.exe,
            true,
            keep_going(on_token),
        )?;
        Ok(())
    }

    fn generate_prompt(
        &mut self,
        prompt: &str,
        max_new: usize,
        prefill: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        let on_token = keep_going(on_token);
        if prefill {
            self.runner.generate_kv_ptx_prefilled(
                prompt,
                max_new,
                &mut self.exec,
                self.exe,
                on_token,
            )?;
        } else {
            self.runner.generate_kv_gpu_cached_sampled(
                prompt,
                max_new,
                &mut self.exec,
                self.exe,
                &mut Sampler::greedy(),
                &[],
                on_token,
            )?;
        }
        Ok(())
    }

    fn reset_profile(&mut self) {
        self.mark = Some(self.exec.stats().timing.mark());
    }

    fn profile_report(&self, tokens: usize, wall: Duration) -> Option<String> {
        let mark = self.mark.as_ref()?;
        let stats = self.exec.stats();
        let entries = stats.timing.entries_with_activity_since(mark);
        if entries.is_empty() {
            return None;
        }
        let report = poot_profile::Report::window(&stats.timing, mark, &entries);
        (report.steps > 0).then(|| report.render(tokens, wall))
    }
}

struct Rocm {
    runner: Runner,
    /// Card 548: the executor contract is this backend's only path (RocmGraphExecutor and its
    /// eager run loop are deleted). One executable loaded once at `open()` time; every mode/arm
    /// below drives it through `&mut dyn Executor` + `ExecutableId`.
    executor: poot_executor::Engine<RocmDevice>,
    exe: poot_executor::ExecutableId,
}

impl Generate for Rocm {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.runner.encode(text)?)
    }

    fn generate_context(
        &mut self,
        context: &[u32],
        max_new: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        self.runner.generate_kv_rocm_prefilled_tokens(
            context,
            max_new,
            &mut self.executor,
            self.exe,
            true,
            keep_going(on_token),
        )?;
        Ok(())
    }

    fn generate_prompt(
        &mut self,
        prompt: &str,
        max_new: usize,
        prefill: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        let on_token = keep_going(on_token);
        if prefill {
            self.runner.generate_kv_rocm_prefilled(
                prompt,
                max_new,
                &mut self.executor,
                self.exe,
                on_token,
            )?;
        } else {
            self.runner.generate_kv_gpu_cached(
                prompt,
                max_new,
                &mut self.executor,
                self.exe,
                on_token,
            )?;
        }
        Ok(())
    }

    fn reset_profile(&mut self) {
        // Card 548: RocmDevice::device_time() is always Unknown (no HSA timestamp
        // binding yet) and the pre-contract executor's label profiler is gone with it - nothing to
        // reset.
    }

    fn profile_report(&self, _tokens: usize, _wall: Duration) -> Option<String> {
        None
    }
}

struct Wgpu {
    runner: Runner,
    /// Card 546a/546b: the backend-neutral executor contract drives every arm (prefill and cached
    /// decode alike) on this one executable.
    executor: poot_executor::Engine<poot_gpu::device::WgpuDevice>,
    exe: poot_executor::ExecutableId,
    /// Card 552: a mark taken at `reset_profile` time, against `executor`'s own typed
    /// `TimingSnapshot`. `None` until the first `reset_profile` call (or when `--profile` is off).
    mark: Option<poot_profile::Mark>,
}

impl Generate for Wgpu {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.runner.encode(text)?)
    }

    fn generate_context(
        &mut self,
        context: &[u32],
        max_new: usize,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        self.runner.generate_kv_gpu_prefilled_tokens(
            context,
            max_new,
            &mut self.executor,
            self.exe,
            true,
            keep_going(on_token),
        )?;
        Ok(())
    }

    fn generate_prompt(
        &mut self,
        prompt: &str,
        max_new: usize,
        prefill: bool,
        on_token: &mut dyn FnMut(&str),
    ) -> anyhow::Result<()> {
        let on_token = keep_going(on_token);
        if prefill {
            self.runner.generate_kv_gpu_prefilled(
                prompt,
                max_new,
                &mut self.executor,
                self.exe,
                on_token,
            )?;
        } else {
            self.runner.generate_kv_gpu_cached(
                prompt,
                max_new,
                &mut self.executor,
                self.exe,
                on_token,
            )?;
        }
        Ok(())
    }

    fn reset_profile(&mut self) {
        self.mark = Some(self.executor.stats().timing.mark());
    }

    fn profile_report(&self, tokens: usize, wall: Duration) -> Option<String> {
        // Card 552: the typed window over the entries this run actually touched (their ids churn
        // per generation call, so they are discovered from the snapshot rather than named ahead of
        // time).
        let mark = self.mark.as_ref()?;
        let stats = self.executor.stats();
        let entries = stats.timing.entries_with_activity_since(mark);
        if entries.is_empty() {
            return None;
        }
        let report = poot_profile::Report::window(&stats.timing, mark, &entries);
        if report.steps == 0 {
            return None;
        }
        let mut text = report.render(tokens, wall);
        // Card 552 SC-007: the typed, purpose-tagged native call counters, read from the same
        // `WgpuDevice`'s own context - independent of the step/dispatch timing above, and never
        // reset by it.
        let calls = self.executor.device().context().execution_counters();
        text.push_str(&format!(
            "native calls: {} submits, {} waits (logical dispatches: {})\n",
            calls.total_submits(),
            calls.total_waits(),
            calls.logical_dispatches,
        ));
        Some(text)
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
    driver: Driver,
}

fn driver_options(capacity: NonZeroUsize, chunk: usize) -> anyhow::Result<DriverOptions> {
    let compile = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    Ok(DriverOptions {
        prefill: compile,
        decode: compile,
        capacity,
        prefill_chunk: NonZeroUsize::new(chunk).context("a prefill chunk holds a token")?,
        max_trace_tokens: capacity,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(8 << 30).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
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

    fn reset_profile(&mut self) {}

    fn profile_report(&self, _tokens: usize, _wall: Duration) -> Option<String> {
        None
    }

    fn engine(&self) -> Engine {
        Engine::Driver
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
    anyhow::ensure!(
        !config.profile,
        "--profile is not supported on the driver (a registered family): it opens counters-only executors"
    );
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
    let executor = open_executor(backend).context("open the executor")?;
    let mut driver = Driver::new(
        Arc::clone(&handle),
        executor,
        driver_options(capacity, chunk)?,
    )
    .context("open the driver")?;
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
    Ok((Box::new(DriverCell { handle, driver }), precision))
}

/// Load the model and open the backend's executor. Returns the engine and the precision to report: the
/// requested one for a checkpoint, the dominant stored packed format for a GGUF.
fn open(config: &Config) -> anyhow::Result<(Box<dyn Generate>, String)> {
    // poot loads f32/bf16/f16 safetensors and computes in f32. Asking for another precision is asked
    // before any weight is read, so a 20B checkpoint is not loaded to be turned down.
    if !matches!(config.precision.as_str(), "f32" | "bf16" | "f16") {
        return Err(Declined(format!(
            "poot bench runner loads f32/bf16/f16 checkpoints (computes f32); got {:?}. \
             Quantized formats (fp8/gptq/awq/mxfp4) are on the roadmap.",
            config.precision
        ))
        .into());
    }
    let path = match &config.weights {
        Weights::Safetensors(path) | Weights::Gguf(path) => path.as_str(),
    };
    // One registry resolve decides: a registered family runs on the driver, any other on the Runner.
    let registry = Registry::builtin().context("build the family registry")?;
    match ModelHandle::load(Path::new(path), &registry) {
        Ok(handle) => open_driver(config, handle),
        Err(DriverError::Unsupported(Unsupported::Registry(RegistryError::Unregistered {
            ..
        }))) => open_runner(config),
        Err(error) => Err(anyhow::Error::from(error).context(format!("load {path}"))),
    }
}

/// The Runner engine: the families the registry does not hold yet.
fn open_runner(config: &Config) -> anyhow::Result<(Box<dyn Generate>, String)> {
    // The typed `RunnerError` stays in the chain (`refusal` reads it); only context is added.
    let runner = match &config.weights {
        Weights::Safetensors(dir) => Runner::load(dir).with_context(|| format!("load {dir}"))?,
        Weights::Gguf(path) => {
            Runner::load_gguf(path).with_context(|| format!("load gguf {path}"))?
        }
    };
    let precision = match &config.weights {
        Weights::Safetensors(_) => config.precision.clone(),
        // A quantized GGUF stays packed (card 545a); it reports the format most of its packed weights
        // are stored in.
        Weights::Gguf(path) => {
            let mut counts = std::collections::BTreeMap::<String, usize>::new();
            for (_, packed) in runner.weight_formats().iter() {
                *counts
                    .entry(format!("{:?}", packed.weight.format()).to_lowercase())
                    .or_default() += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, n)| n)
                .map(|(format, _)| format)
                .ok_or_else(|| anyhow::anyhow!("{path}: no weight is stored packed"))?
        }
    };
    let engine: Box<dyn Generate> = match config.backend {
        Backend::Ptx => {
            let device = if config.profile {
                PtxDevice::new_with_device_timing()
            } else {
                PtxDevice::new()
            }
            .context("init PTX device (libcuda?)")?;
            let mut exec = poot_executor::Engine::new(device);
            let exe = runner.load_on(&mut exec).context("ptx load_on")?;
            Box::new(Ptx {
                runner,
                exec,
                exe,
                mark: None,
            })
        }
        Backend::Rocm => {
            let device =
                RocmDevice::new().context("init ROCm device (card 546a executor contract)")?;
            let mut executor = poot_executor::Engine::new(device);
            let exe = runner
                .load_on(&mut executor)
                .context("load_on (card 546a executor contract)")?;
            Box::new(Rocm {
                runner,
                executor,
                exe,
            })
        }
        Backend::Wgpu => {
            // Card 552: `--profile` drives the executor with bounded detailed device-time
            // collection (the same declared bound feeds both the device's in-flight-query cap and
            // the engine's own retention); otherwise it stays the counters-only default (no query
            // resources are ever created, no added synchronization).
            let detail = poot_executor::DetailedTiming::every_step(4096, 1 << 20, 4);
            let device = if config.profile {
                poot_gpu::device::WgpuDevice::new_with_timing(detail.max_in_flight_queries)
            } else {
                poot_gpu::device::WgpuDevice::new()
            }
            .context("init wgpu device (card 546a executor contract)")?;
            let timing_options = if config.profile {
                poot_executor::TimingOptions::Detailed(detail)
            } else {
                poot_executor::TimingOptions::CountersOnly
            };
            let mut executor = poot_executor::Engine::with_timing(device, timing_options);
            let exe = runner
                .load_on(&mut executor)
                .context("load_on (card 546a executor contract)")?;
            Box::new(Wgpu {
                runner,
                executor,
                exe,
                mark: None,
            })
        }
    };
    Ok((engine, precision))
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
}

impl Run {
    /// Decode time per token after the first.
    fn tpot_ms(&self) -> f64 {
        if self.tokens > 1 {
            self.itl_ms.iter().sum::<f64>() / (self.tokens - 1) as f64
        } else {
            0.0
        }
    }

    fn decode_tok_s(&self) -> f64 {
        let decode_ms = self.itl_ms.iter().sum::<f64>();
        if decode_ms > 0.0 && self.tokens > 1 {
            (self.tokens - 1) as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        }
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
            "itl_ms": self.itl_ms,
            "first_gap_ms": self.itl_ms.first(),
            "tpot_ms": self.tpot_ms(),
            "decode_tok_s": self.decode_tok_s(),
        })
    }
}

/// The one measure loop: run the configured mode on `engine` and return the result object.
fn measure(config: &Config, engine: &mut dyn Generate, precision: &str) -> anyhow::Result<Value> {
    let body = match config.mode {
        Mode::Single => measure_single(config, engine)?,
        Mode::DecodeCurve => measure_curve(config, engine)?,
    };
    let mut line = identity(
        Some(engine.engine()),
        config.backend,
        config.mode,
        precision,
    );
    line["profiled"] = json!(config.profile);
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
/// result with no `build_sha`.
fn identity(engine: Option<Engine>, backend: Backend, mode: Mode, precision: &str) -> Value {
    let mut line = json!({
        "framework": "poot",
        "build_sha": BUILD_SHA,
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

fn measure_single(config: &Config, engine: &mut dyn Generate) -> anyhow::Result<Value> {
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
        timed(|on_token| {
            engine.generate_prompt(&prompt, config.gen_tokens, config.prefill, on_token)
        })
    };

    let mut warmup_runs = Vec::new();
    for w in 0..config.warmup {
        warmup_runs.push(run_once(engine)?.receipt());
        if !config.json {
            eprintln!("poot-bench: warmup {}/{} done", w + 1, config.warmup);
        }
    }
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
                "poot-bench: iter {}/{}  TTFT {:.0}ms  TPOT {:.1}ms  {:.1} tok/s",
                it + 1,
                config.iters,
                run.ttft_ms,
                run.tpot_ms(),
                run.decode_tok_s()
            );
        }
        runs.push(run);
    }
    let timed_wall = timed_start.elapsed();

    if config.profile {
        let tokens = runs.iter().map(|run| run.tokens).sum();
        print_profile(engine.profile_report(tokens, timed_wall));
    }

    let column = |value: fn(&Run) -> f64| runs.iter().map(value).collect::<Vec<f64>>();
    let ttft = column(|run| run.ttft_ms);
    let tok_s = column(|run| run.decode_tok_s());
    Ok(json!({
        "warmup_runs": warmup_runs,
        "runs": runs.iter().map(Run::receipt).collect::<Vec<_>>(),
        "prompt_tokens": engine.encode(&prompt)?.len(),
        "gen_tokens": runs.last().map_or(0, |run| run.tokens),
        "iterations": config.iters,
        "ttft_ms": round3(median(&ttft)),
        "tpot_ms": round3(median(&column(Run::tpot_ms))),
        "e2e_ms": round3(median(&column(|run| run.e2e_ms))),
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
) -> anyhow::Result<Value> {
    let context: Vec<u32> = (0..isl).map(|i| base[i % base.len()]).collect();
    let run_once = |engine: &mut dyn Generate| {
        timed(|on_token| engine.generate_context(&context, config.osl, on_token))
    };
    let mut warmup_runs = Vec::new();
    for _ in 0..config.warmup {
        warmup_runs.push(run_once(engine)?.receipt());
    }
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
    if config.profile {
        let tokens: usize = iters
            .iter()
            .map(|iter| iter["itl_ms"].as_array().map_or(0, Vec::len))
            .sum();
        eprintln!("poot-bench: isl={isl} profile window:");
        print_profile(engine.profile_report(tokens, timed_start.elapsed()));
    }
    Ok(json!({ "isl": isl, "prompt_tokens": isl, "warmup_runs": warmup_runs, "iters": iters }))
}

/// decode-curve. A per-ISL failure after the first keeps the measured points and records a caveat: the
/// largest contexts fault first, and larger ones would fault too. A failure at the first ISL measured
/// nothing, so it is the cell's error (a refusal stays a refusal).
///
/// Batched prefill materializes the `[Hq,N,N]` attention scores, so a very large ISL on a large model can
/// run out of memory in f32; that surfaces as such a caveat.
fn measure_curve(config: &Config, engine: &mut dyn Generate) -> anyhow::Result<Value> {
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
        let point =
            std::panic::catch_unwind(AssertUnwindSafe(|| measure_isl(config, engine, &base, isl)))
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

fn print_profile(report: Option<String>) {
    if let Some(report) = report {
        eprintln!("=== PROFILE ===");
        eprintln!("{report}");
    }
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
/// of "poot does not run this": [`Declined`], the [`RunnerError::UnsupportedModel`] the safetensors
/// loader raises for an architecture outside its allowlist, or the driver's [`DriverError::Unsupported`]
/// (a family the registry does not hold, a shape the family or the planner refuses), found anywhere in
/// the error chain. Card 577's
/// compile-generated support list adds its typed planner refusal as one more arm here.
fn refusal(error: &anyhow::Error) -> Option<(Option<Engine>, String)> {
    error.chain().find_map(|cause| {
        if let Some(RunnerError::UnsupportedModel { .. }) = cause.downcast_ref::<RunnerError>() {
            return Some((Some(Engine::Runner), cause.to_string()));
        }
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
fn result_line(config: &Config, outcome: anyhow::Result<Value>) -> anyhow::Result<Value> {
    match outcome {
        Err(error) => match refusal(&error) {
            Some((engine, reason)) => {
                let mut line = identity(engine, config.backend, config.mode, &config.precision);
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = match parse_args(&args)? {
        Invocation::Version => {
            println!("poot-bench-runner {BUILD_SHA}");
            return Ok(());
        }
        Invocation::Run(config) => config,
    };
    let outcome = open(&config)
        .and_then(|(mut engine, precision)| measure(&config, engine.as_mut(), &precision));
    println!("{}", result_line(&config, outcome)?);
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
    }

    impl Stub {
        fn new() -> Self {
            Stub {
                events: RefCell::new(Vec::new()),
                fail_at_isl: None,
                fail_prompt: false,
                error: || anyhow::anyhow!("stub failure"),
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
            Stub::emit(max_new, on_token);
            Ok(())
        }

        fn reset_profile(&mut self) {
            self.events.borrow_mut().push(Event::ResetProfile);
        }

        fn profile_report(&self, tokens: usize, _wall: Duration) -> Option<String> {
            self.events
                .borrow_mut()
                .push(Event::ProfileReport { tokens });
            None
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

    fn refused() -> anyhow::Error {
        anyhow::Error::from(RunnerError::UnsupportedModel {
            model_type: "stub_model".to_string(),
            reason: "not a supported safetensors architecture",
        })
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
            let line = measure(&single, &mut stub, "bf16").unwrap();
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
            let line = measure(&curve, &mut stub, "bf16").unwrap();
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
            let outcome = measure(&config, &mut Stub::new(), "bf16");
            let line = result_line(&config, outcome).unwrap();
            let text = line.to_string();
            assert!(!text.contains('\n'), "{mode:?}: the result is one line");
            assert_eq!(line["mode"], mode.label());
            assert_eq!(line["framework"], "poot");
            assert_eq!(line["build_sha"], BUILD_SHA);
            harness_accepts(&line, mode == Mode::DecodeCurve, 6, Backend::Rocm);
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
        let line = result_line(&config, measure(&config, &mut stub, "bf16")).unwrap();
        assert_eq!(line["status"], "unsupported");
        assert_eq!(
            line["reason"],
            "stub_model is not supported by the generic Runner: not a supported safetensors architecture"
        );
        assert_eq!(line["backend"], "wgpu");
        assert_eq!(line["build_sha"], BUILD_SHA);

        stub.error = || anyhow::anyhow!("out of memory");
        let error = result_line(&config, measure(&config, &mut stub, "bf16")).unwrap_err();
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

    /// SC-002 (runner half, real entry): a model the loader does not serve is refused by `Runner::load`
    /// itself, and the refusal reaches the result line with its text and passes the harness's contract.
    /// Nothing here builds the error by hand, so the test cannot pass against a type the bench path never
    /// raises.
    #[test]
    fn a_model_the_loader_does_not_serve_is_a_recorded_refusal() {
        let dir = config_only_checkpoint("unserved", r#"{"model_type": "stub_arch"}"#);
        let mut config = config(&["--precision", "bf16"]);
        config.weights = Weights::Safetensors(dir.display().to_string());
        config.backend = Backend::Rocm;
        let outcome = open(&config).map(|_| Value::Null);
        std::fs::remove_dir_all(&dir).unwrap();

        let line = result_line(&config, outcome).unwrap();
        assert_eq!(line["status"], "unsupported");
        let reason = line["reason"].as_str().unwrap();
        assert!(
            reason.starts_with("stub_arch is not supported by the generic Runner"),
            "{reason}"
        );
        assert_eq!(harness_refusal(&line, Backend::Rocm), reason);
    }

    #[test]
    fn a_load_failure_that_is_not_a_refusal_stays_an_error() {
        let mut config = config(&[]);
        config.weights = Weights::Safetensors("/nonexistent/checkpoint".to_string());
        let outcome = open(&config).map(|_| Value::Null);

        let error = result_line(&config, outcome).unwrap_err();
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
        let outcome = open(&config).map(|_| Value::Null);

        let line = result_line(&config, outcome).unwrap();
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
        let line = measure(&config, &mut stub, "bf16").unwrap();
        assert_eq!(line["curve"].as_array().unwrap().len(), 1);
        assert_eq!(
            line["caveats"],
            json!(["isl 16+ omitted: CUDA out of memory"])
        );

        stub.fail_at_isl = Some(8);
        stub.error = refused;
        let outcome = measure(&config, &mut stub, "bf16");
        let line = result_line(&config, outcome).unwrap();
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
            fn profile_report(&self, _: usize, _: Duration) -> Option<String> {
                None
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
        let line = measure(&config, &mut Panics, "bf16").unwrap();
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
    /// the config-only qwen2 directory fails reading its weights, as `ModelHandle::load`, not as the
    /// Runner's registered-family refusal), and the result line names the engine that generated.
    #[test]
    fn a_registered_family_opens_on_the_driver_and_the_line_names_the_engine() {
        let dir = config_only_checkpoint("registered", r#"{"model_type": "qwen2"}"#);
        let mut config = config(&["--precision", "bf16"]);
        config.weights = Weights::Safetensors(dir.display().to_string());
        let error = open(&config).map(|_| ()).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            error.chain().any(|cause| matches!(
                cause.downcast_ref::<DriverError>(),
                Some(DriverError::Load(_))
            )),
            "{error:#}"
        );

        let line = identity(Some(Engine::Driver), Backend::Wgpu, Mode::Single, "bf16");
        assert_eq!(line["engine"], "driver");
        assert_eq!(
            identity(Some(Engine::Runner), Backend::Wgpu, Mode::Single, "bf16")["engine"],
            "runner"
        );
        assert!(identity(None, Backend::Wgpu, Mode::Single, "bf16")
            .get("engine")
            .is_none());
    }

    /// A family or shape the driver cannot serve is a recorded refusal (`unsupported`), the same as the
    /// Runner's own refusals; a load failure is still an error.
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
        assert_eq!(delayed.tpot_ms(), 12.5);
        assert_eq!(delayed.decode_tok_s(), 80.0);
        assert_eq!(delayed.finish_ms, 107.0);
        assert_eq!(delayed.e2e_ms - run.e2e_ms, 100.0);
        let relocated = request(&[60, 70, 85], 92);
        assert_eq!(relocated.e2e_ms - run.e2e_ms, 20.0);
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
            assert_eq!((run.tpot_ms(), run.decode_tok_s()), (0.0, 0.0));
            assert!(run.itl_ms.is_empty());
        }
    }
}
