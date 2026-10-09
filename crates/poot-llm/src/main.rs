//! poot-generate: load a checkpoint and generate text from a prompt, with the flags of llama.cpp's
//! `llama-cli` where poot has the capability.
//!
//! A checkpoint loads as a `driver::ModelHandle` and generates through the driver on the device
//! `--backend` names (`wgpu`, `rocm`, `ptx`, `vulkan`, or `auto`, the default). A family the registry
//! does not know is a typed error.
//!
//! A model that ships a chat template answers the prompt as a user turn of it (after `--system-prompt`),
//! and prints the reply only; `--no-chat-template`, or a model with no template (a base model), continues
//! the prompt as written and prints it first. `-n` defaults to -1: run to EOS, a `-r` string or the end
//! of the context window (`-c`, default the model's own). `--temp` defaults to 0, greedy, which is
//! poot's exact-token convention (llama-cli samples at 0.80).
//!
//! Without `--model` the model is `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither,
//! it exits with an error naming the variable. ROCm is a build feature: `--features cli,rocm`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, ValueEnum};
use poot_llm::driver::{
    BackendChoice, Generation, ModelHandle, PhaseTiming, Sampling, TextRequest, generate_text,
    open_executor,
};
use poot_models::registry::Registry;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Auto,
    Wgpu,
    Rocm,
    Ptx,
    Vulkan,
}

impl From<Backend> for BackendChoice {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Auto => Self::Auto,
            Backend::Wgpu => Self::Wgpu,
            Backend::Rocm => Self::Rocm,
            Backend::Ptx => Self::Ptx,
            Backend::Vulkan => Self::Vulkan,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "poot-generate", version, about)]
#[command(group(ArgGroup::new("input").required(true).args(["prompt", "file"])))]
struct Args {
    /// The checkpoint: a safetensors directory or a `.gguf` file. Default: `qwen2.5-0.5b` under
    /// `POOT_MODELS_DIR`.
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,
    /// The prompt.
    #[arg(short = 'p', long)]
    prompt: Option<String>,
    /// Read the prompt from a file.
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,
    /// A system turn ahead of the prompt, through the model's chat template. Unlike llama-cli it has no
    /// short form (`-sys`): clap takes single-character shorts only.
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,
    /// Use the prompt as written even when the model ships a chat template.
    #[arg(long = "no-chat-template")]
    no_chat_template: bool,
    /// New tokens to generate at most; -1 runs to EOS, a reverse prompt or the end of the context window.
    #[arg(
        short = 'n',
        long = "predict",
        default_value_t = -1,
        allow_negative_numbers = true,
        value_parser = clap::value_parser!(i64).range(-1..)
    )]
    predict: i64,
    /// The context window in tokens (prompt plus new tokens); 0 takes the model's own.
    #[arg(short = 'c', long = "ctx-size", default_value_t = 0)]
    ctx_size: usize,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    #[arg(long = "top-k", default_value_t = 40)]
    top_k: usize,
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,
    #[arg(long = "min-p", default_value_t = 0.05)]
    min_p: f32,
    /// Repetition penalty over the prompt and the output; 1 is off.
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,
    /// Sampling seed; default a fresh one per run.
    #[arg(short = 's', long)]
    seed: Option<u64>,
    /// Stop when this string is generated (repeatable); it is printed with the token that completes it.
    #[arg(short = 'r', long = "reverse-prompt")]
    reverse_prompt: Vec<String>,
    /// Generate through the model's end-of-sequence token.
    #[arg(long = "ignore-eos")]
    ignore_eos: bool,
    /// The device.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Print the engine-measured prefill and decode tokens per second to stderr (the default).
    #[arg(long = "show-timings", overrides_with = "no_show_timings")]
    show_timings: bool,
    /// Do not print the timings.
    #[arg(long = "no-show-timings", overrides_with = "show_timings")]
    no_show_timings: bool,
    /// Log what the run does (loading, the model's configuration, the seed) to stderr.
    #[arg(short = 'v', long)]
    verbose: bool,
}

/// No model path was given and `POOT_MODELS_DIR` is unset or empty.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model given and POOT_MODELS_DIR is not set: pass --model or set POOT_MODELS_DIR to the directory that holds `{DEFAULT_MODEL}`"
)]
struct ModelDirUnset;

/// The model directory under `POOT_MODELS_DIR` used when no path is given.
const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

/// The model path: `--model`, else `DEFAULT_MODEL` under `POOT_MODELS_DIR`.
fn model_path_or_env(flag: Option<&Path>) -> Result<PathBuf, ModelDirUnset> {
    if let Some(path) = flag {
        return Ok(path.to_path_buf());
    }
    match std::env::var("POOT_MODELS_DIR") {
        Ok(dir) if !dir.is_empty() => Ok(Path::new(&dir).join(DEFAULT_MODEL)),
        _ => Err(ModelDirUnset),
    }
}

fn fresh_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64)
}

impl Args {
    /// Timings print unless `--no-show-timings` came last.
    fn timings(&self) -> bool {
        self.show_timings || !self.no_show_timings
    }

    fn request(&self) -> Result<TextRequest> {
        let prompt = match (&self.prompt, &self.file) {
            (Some(prompt), _) => prompt.clone(),
            (None, Some(file)) => std::fs::read_to_string(file)
                .with_context(|| format!("reading the prompt file {}", file.display()))?,
            (None, None) => unreachable!("clap requires -p or -f"),
        };
        let seed = self.seed.unwrap_or_else(fresh_seed);
        tracing::info!(seed, "sampling seed");
        Ok(TextRequest {
            prompt,
            system: self.system_prompt.clone(),
            raw: self.no_chat_template,
            stops: self.reverse_prompt.clone(),
            max_new: usize::try_from(self.predict).ok(),
            ctx: (self.ctx_size != 0).then_some(self.ctx_size),
            sampling: Sampling {
                temperature: self.temp,
                top_k: self.top_k,
                top_p: self.top_p,
                min_p: self.min_p,
                repeat_penalty: self.repeat_penalty,
                seed,
            },
            ignore_eos: self.ignore_eos,
        })
    }
}

fn report(phase: &str, timing: PhaseTiming) {
    match timing.tokens_per_second() {
        Some(rate) => eprintln!(
            "{phase}: {} tokens in {:.1} ms ({rate:.1} tokens/s)",
            timing.tokens,
            timing.wall.as_secs_f64() * 1e3
        ),
        None => eprintln!("{phase}: {} tokens, no engine time measured", timing.tokens),
    }
}

fn report_timings(generation: &Generation) {
    // `generate_text` always runs a driver that measures its phases.
    if let Some(timing) = generation.timing {
        report("prefill", timing.prefill);
        report("decode", timing.decode);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let level = if args.verbose { "info" } else { "warn" };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        .init();
    let request = args.request()?;
    let model_path = model_path_or_env(args.model.as_deref())?;

    // A backend this build lacks, or a host without the device, fails before the model loads.
    let executor = open_executor(args.backend.into())?;
    tracing::info!(model_path = %model_path.display(), "loading model");
    let registry = Registry::builtin()?;
    let handle = Arc::new(ModelHandle::load(
        &model_path,
        &registry,
        &poot_llm::driver::LoadOptions::default(),
    )?);
    let config = handle.config();
    tracing::info!(
        family = %config.family,
        vocab = config.vocab,
        max_positions = config.max_positions,
        backend = ?args.backend,
        "model config"
    );
    // Generated text is program output (stdout), distinct from status logs (stderr).
    let generation = generate_text(&handle, executor, &request, &mut |piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    })?;
    println!();
    tracing::info!(
        tokens = generation.tokens.len(),
        finish = ?generation.finish,
        "generation complete"
    );
    if args.timings() {
        report_timings(&generation);
    }
    Ok(())
}
