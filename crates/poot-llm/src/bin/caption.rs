//! poot-caption: caption an image with a vision-language model. Loads a SmolVLM / idefics3 checkpoint
//! (its vision tower and its text tower, each on a driver), decodes the image and prints a caption.
//!
//! Usage: `poot-caption <model_dir> <image_path> [question] [--backend B]`
//!
//! `--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto` (the default).

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::GenerationControl;
use poot_llm::driver::{
    BackendChoice, DraftPolicy, DriverOptions, PreparedSetLimits, open_executor, program_retention,
};
use poot_llm::front_end::{CaptionRequest, Idefics3, decode_image};
use poot_models::registry::Registry;

/// The driver options of one tower: `capacity` positions, prefill in pieces of at most `chunk`.
fn options(capacity: usize, chunk: usize) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let nz = |n: usize| NonZeroUsize::new(n).expect("a positive size");
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        max_trace_tokens: nz(chunk),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
    }
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let backend = args
        .iter()
        .position(|a| a == "--backend")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    let positional: Vec<&String> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in &args[1..] {
            if skip {
                skip = false;
            } else if a == "--backend" {
                skip = true;
            } else {
                out.push(a);
            }
        }
        out
    };
    let model_dir = positional
        .first()
        .context("usage: poot-caption <model_dir> <image_path> [question] [--backend B]")?;
    let image_path = positional.get(1).context("need an image path")?;
    let question = positional
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Can you describe this image?");
    let choice = backend_choice(backend)?;

    let bytes = std::fs::read(image_path).with_context(|| format!("read {image_path}"))?;
    let image = decode_image(&bytes).context("decode image")?;
    eprintln!(
        "loaded {image_path} ({}x{}); loading model from {model_dir}...",
        image.width, image.height
    );

    // The text tower holds the whole prompt (up to 17 frames of 64 rows each and the question) and the
    // 64 new tokens; the vision tower is one frame per step.
    let mut vlm = Idefics3::load(
        Path::new(model_dir),
        &Registry::builtin()?,
        (open_executor(choice)?, options(1024, 1024)),
        (open_executor(choice)?, options(8192, 256)),
    )
    .context("load the VLM")?;
    let generation = vlm
        .caption(
            &CaptionRequest {
                images: &[image.as_image()],
                question,
                max_new: 64,
            },
            &mut |_: u32, piece: &str| {
                print!("{piece}");
                std::io::stdout().flush().ok();
                GenerationControl::Continue(())
            },
        )
        .context("caption")?;
    println!();
    tracing::info!(tokens = generation.tokens.len(), finish = ?generation.finish, "caption complete");
    Ok(())
}
