//! qwen2-generate: load a checkpoint and generate text greedily.
//!
//! A checkpoint loads as a `driver::ModelHandle` and generates through the driver on the device
//! `--backend` names (`wgpu`, `rocm`, `ptx`, `vulkan`, or `auto`, the default). A family the registry
//! does not know is a typed error.
//! Usage:
//!   qwen2-generate \[MODEL_PATH\] \[PROMPT\] \[MAX_NEW_TOKENS\] [--backend B]
//!
//! Without `MODEL_PATH` the model is `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither, it
//! exits with an error naming the variable.

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::GenerationControl;
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, ModelHandle, PreparedSetLimits,
    open_executor,
};
use poot_models::registry::Registry;

/// The value after a `--name` flag, if present (e.g. `--backend rocm`).
fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn backend_choice(flag: Option<&str>) -> Result<BackendChoice> {
    Ok(match flag {
        None | Some("auto") => BackendChoice::Auto,
        Some("wgpu") => BackendChoice::Wgpu,
        Some("rocm") => BackendChoice::Rocm,
        Some("ptx") => BackendChoice::Ptx,
        Some("vulkan") => BackendChoice::Vulkan,
        Some(other) => bail!("--backend {other}: expected wgpu, rocm, ptx, vulkan or auto"),
    })
}

fn emit(piece: &str) {
    print!("{piece}");
    std::io::stdout().flush().ok();
}

/// Generate through the driver: the whole prompt in prefill chunks, then greedy decode.
fn generate_on_driver(
    handle: ModelHandle,
    choice: BackendChoice,
    prompt: &str,
    max_new: usize,
) -> Result<()> {
    let handle = Arc::new(handle);
    let config = handle.config();
    let tokens = handle.text().encode(prompt)?;
    let granule = config.prefill_granule.get();
    let chunk = 128usize.div_ceil(granule) * granule;
    let capacity = NonZeroUsize::new(tokens.len() + max_new).context("an empty prompt")?;
    tracing::info!(
        family = %config.family,
        vocab = config.vocab,
        max_positions = config.max_positions,
        ?choice,
        "model config (driver)"
    );
    let compile = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    let options = DriverOptions {
        prefill: compile,
        decode: compile,
        capacity,
        prefill_chunk: NonZeroUsize::new(chunk).expect("a granule multiple is nonzero"),
        max_trace_tokens: NonZeroUsize::new(chunk).expect("a granule multiple is nonzero"),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
        draft: poot_llm::driver::DraftPolicy::Off,
    };
    let mut driver = Driver::new(Arc::clone(&handle), open_executor(choice)?, options)?;
    // Generated text is program output (stdout), distinct from status logs (tracing).
    emit(prompt);
    let generation = driver.generate(
        GenerateRequest {
            prompt: tokens,
            max_new,
            sampler: Sampler::greedy(),
            stops: Vec::new(),
            ignore_eos: false,
        },
        &mut |_: u32, piece: &str| {
            emit(piece);
            GenerationControl::Continue(())
        },
    )?;
    println!();
    tracing::info!(
        tokens = generation.tokens.len(),
        finish = ?generation.finish,
        "generation complete"
    );
    Ok(())
}

/// No model path was given and `POOT_MODELS_DIR` is unset or empty.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model path given and POOT_MODELS_DIR is not set: pass a model path or set POOT_MODELS_DIR to the directory that holds `{DEFAULT_MODEL}`"
)]
struct ModelDirUnset;

/// The model directory under `POOT_MODELS_DIR` used when no path is given.
const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

/// The model path: the first positional argument, else `DEFAULT_MODEL` under `POOT_MODELS_DIR`.
fn model_path_or_env(positional: Option<&String>) -> Result<String, ModelDirUnset> {
    if let Some(path) = positional {
        return Ok(path.clone());
    }
    match std::env::var("POOT_MODELS_DIR") {
        Ok(dir) if !dir.is_empty() => Ok(Path::new(&dir)
            .join(DEFAULT_MODEL)
            .to_string_lossy()
            .into_owned()),
        _ => Err(ModelDirUnset),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().collect();
    // positional args skip the flags (anything starting with '-' and the value after a valued flag).
    let valued: [&str; 1] = ["--backend"];
    let positional: Vec<String> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in &args[1..] {
            if skip {
                skip = false;
                continue;
            }
            if a.starts_with("--") {
                skip = valued.contains(&a.as_str());
                continue;
            }
            out.push(a.clone());
        }
        out
    };
    let model_path = model_path_or_env(positional.first())?;
    let prompt = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_new: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let choice = backend_choice(flag_val(&args, "--backend").as_deref())?;

    tracing::info!(model_path, "loading model");
    let path = Path::new(&model_path);
    let registry = Registry::builtin()?;
    let handle = ModelHandle::load(path, &registry, &poot_llm::driver::LoadOptions::default())?;
    generate_on_driver(handle, choice, &prompt, max_new)
}
