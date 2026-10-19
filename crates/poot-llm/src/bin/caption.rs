//! poot-caption: caption an image with a vision-language model. Loads the checkpoint (its text decoder
//! and the vision towers its registry entry holds, each on a driver), decodes the image and prints a
//! caption. A checkpoint whose front end takes video (Qwen2.5-VL) also takes an animated PNG as a video,
//! sampled at the processor's default two frames per second.
//!
//! Usage: `poot-caption <model_dir> <image_path> [question] [--backend B]`
//!
//! `--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto` (the default).

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, open_executor, program_retention,
};
use poot_llm::{
    Apng, ApngError, FrontEndSession, GenerationControl, Sampler, decode_image, tower_options,
};
use poot_models::multimodal::front_end::{ChatTurn, Media, Video};
use poot_models::registry::Registry;

/// The driver options of one tower: `capacity` positions, prefill in pieces of at most `chunk`.
fn options(capacity: usize, chunk: usize) -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let nz = |n: usize| NonZeroUsize::new(n).expect("a positive size");
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        trace: poot_llm::driver::serving_trace_limits(nz(chunk)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
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

    let handle = Arc::new(
        ModelHandle::load(
            Path::new(model_dir.as_str()),
            &Registry::builtin()?,
            &LoadOptions::default(),
        )
        .context("load the checkpoint")?,
    );
    let mut session = FrontEndSession::open(Arc::clone(&handle), |_, positions| {
        Ok::<_, anyhow::Error>((open_executor(choice)?, tower_options(positions)))
    })
    .context("open the vision towers")?
    .context("the checkpoint is not a vision-language model: its family has no front end")?;

    let bytes = std::fs::read(image_path).with_context(|| format!("read {image_path}"))?;
    eprintln!("loaded {image_path}; loading model from {model_dir}...");
    // An animated PNG is a video; any other file is one image.
    let apng = match Apng::from_bytes(&bytes) {
        Ok(apng) => Some(apng),
        Err(ApngError::StaticPng) => None,
        Err(error) => return Err(error).with_context(|| format!("probe {image_path}")),
    };
    let image;
    let mut frames;
    let mut media = match &apng {
        Some(apng) => {
            frames = apng.frame_source().context("open the video frames")?;
            vec![Media::Video(Video {
                metadata: apng.metadata(),
                frames: &mut frames,
            })]
        }
        None => {
            image = decode_image(&bytes).context("decode image")?;
            vec![Media::Image(&image)]
        }
    };
    let prepared = session
        .prepare(&[ChatTurn::user(1, question)], &mut media)
        .context("prepare the prompt")?;

    // The text decoder holds the whole prompt (every frame's rows and the question) and the 64 new tokens.
    let mut text = Driver::new(handle, open_executor(choice)?, options(8192, 256))
        .context("open the text decoder")?
        .serve(Layout::Contiguous);
    let generation = text
        .generate_with_inputs(
            GenerateRequest {
                prompt: prepared.prompt,
                max_new: 64,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            prepared.inputs,
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
