//! qwen2-generate: load a checkpoint and generate text greedily.
//!
//! A checkpoint of a registered family loads as a `driver::ModelHandle` and generates through the
//! driver on the device `--backend` names (`wgpu`, `rocm`, `ptx`, `vulkan`, or `auto`, the default). A checkpoint
//! of a family not yet registered (the MoE and hybrid families) loads as a `Runner`:
//!   (default)         CPU eager reference executor
//!   --gpu             cached decode on wgpu
//!   --ptx             captured PTX decode (NVPTX kernels, cudarc; requires an NVIDIA GPU)
//! The registry decides which: a family it does not know continues on the Runner.
//! Usage:
//!   qwen2-generate \[MODEL_PATH\] \[PROMPT\] \[MAX_NEW_TOKENS\] [--backend B] [--gpu|--ptx]
//!
//! Without `MODEL_PATH` the model is `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither, it
//! exits with an error naming the variable.

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions, FusionPolicy, Submission};
use poot_llm::driver::error::{DriverError, Unsupported};
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, ModelHandle, PreparedSetLimits,
    open_executor,
};
use poot_llm::{GenerationControl, Runner, Sampler};
use poot_models::registry::{Registry, RegistryError};

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
        fusion: FusionPolicy::Full,
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

/// Generate on the Runner: the families the registry does not know yet.
fn generate_on_runner(path: &Path, args: &[String], prompt: &str, max_new: usize) -> Result<()> {
    let runner = if path.is_dir() {
        Runner::load(path)?
    } else {
        Runner::load_gguf(path)?
    };
    let use_gpu = args.iter().any(|a| a == "--gpu");
    let use_ptx = args.iter().any(|a| a == "--ptx");
    let config = runner.config();
    tracing::info!(
        hidden = config.hidden,
        layers = config.layers,
        n_heads = config.n_heads,
        n_kv_heads = config.n_kv_heads,
        head_dim = config.head_dim,
        vocab = config.vocab,
        "model config (runner)"
    );

    emit(prompt);
    let on_piece = |piece: &str| {
        emit(piece);
        GenerationControl::Continue(())
    };
    let tokens = if use_gpu {
        let mut engine = poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new()?);
        let exe = runner.load_on(&mut engine)?;
        runner.generate_kv_gpu_cached(prompt, max_new, &mut engine, exe, on_piece)?
    } else if use_ptx {
        // captured PTX decode; needs an NVIDIA GPU at runtime (kernels JIT'd here if llc is present).
        let mut engine = poot_executor::Engine::new(
            poot_ptx_gpu::PtxDevice::new().context("init PtxDevice (libcuda present?)")?,
        );
        let exe = runner.load_on(&mut engine)?;
        runner.generate_kv_gpu_cached_sampled(
            prompt,
            max_new,
            &mut engine,
            exe,
            &mut Sampler::greedy(),
            &[],
            on_piece,
        )?
    } else {
        runner.generate(prompt, max_new, on_piece)?
    };
    println!();
    tracing::info!(full = %runner.decode(&tokens)?, "generation complete");
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
    // POOT_DEBUG_NAN=1 logs the first non-finite forward eqn: a typed choice this binary makes
    // (ADR-0104 decision 5), read here rather than inside poot-llm's own library code.
    poot_llm::arm_nan_tap(std::env::var("POOT_DEBUG_NAN").is_ok_and(|v| v == "1"));
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
    match ModelHandle::load(path, &registry) {
        Ok(handle) => generate_on_driver(handle, choice, &prompt, max_new),
        Err(DriverError::Unsupported(Unsupported::Registry(RegistryError::Unregistered {
            ..
        }))) => generate_on_runner(path, &args, &prompt, max_new),
        Err(error) => Err(error.into()),
    }
}
