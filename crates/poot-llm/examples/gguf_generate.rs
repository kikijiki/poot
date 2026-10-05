//! Load a model from a single GGUF file and generate greedily through the driver. The config, weights (kept
//! packed as stored) and tokenizer all come from the one `.gguf`; the family comes from its
//! `general.architecture` through the model registry (e.g. Qwen2.5, Llama-3.2, SmolLM2 across the common
//! quant types). The device is the first that opens: PTX, then ROCm, then wgpu.
//!
//! Run: `cargo run -p poot-llm --release --example gguf_generate -- /path/to/model.gguf ["a prompt"]`

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_graph_plan::{CompileLimits, CompileOptions, FusionPolicy, Submission};
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, ModelHandle, PreparedSetLimits,
    open_executor,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::Registry;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gguf_generate <model.gguf> [prompt]");
        eprintln!("  e.g. a Qwen2.5 or Llama-3.2 instruct GGUF (Q4_K_M, Q8_0, ...)");
        return Ok(());
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());

    // Everything (config + weights + tokenizer) is read from the GGUF.
    let handle = Arc::new(ModelHandle::load(Path::new(&path), &Registry::builtin()?)?);
    let tokens = handle.text().encode(&prompt)?;
    let max_new = 32;
    let compile = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    };
    let granule = handle.config().prefill_granule.get();
    let chunk = NonZeroUsize::new(128usize.div_ceil(granule) * granule).expect("nonzero");
    let options = DriverOptions {
        prefill: compile,
        decode: compile,
        capacity: NonZeroUsize::new(tokens.len() + max_new).expect("nonzero"),
        prefill_chunk: chunk,
        max_trace_tokens: chunk,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
    };
    let mut driver = Driver::new(
        Arc::clone(&handle),
        open_executor(BackendChoice::Auto)?,
        options,
    )?;

    print!("{prompt}");
    let _ = std::io::stdout().flush();
    // stream each new token as it is decoded.
    let generation = driver.generate(
        GenerateRequest {
            prompt: tokens,
            max_new,
            sampler: Sampler::greedy(),
            stops: Vec::new(),
            ignore_eos: false,
        },
        &mut |_: u32, piece: &str| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            GenerationControl::Continue(())
        },
    )?;
    println!();
    eprintln!("[{} tokens generated]", generation.tokens.len());
    Ok(())
}
