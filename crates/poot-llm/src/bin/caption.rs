//! poot-caption: caption an image with a vision-language model. Loads a SmolVLM / idefics3 or a
//! Qwen2.5-VL checkpoint (its vision tower and its text tower, each on a driver), decodes the image
//! and prints a caption. A Qwen2.5-VL checkpoint also takes an animated PNG as a video, sampled at
//! the processor's default two frames per second.
//!
//! Usage: `poot-caption <model_dir> <image_path> [question] [--backend B]`
//!
//! `--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto` (the default).

use std::io::Write;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, DriverOptions, PreparedSetLimits, open_executor, program_retention,
};
use poot_llm::front_end::{CaptionRequest, Idefics3, decode_image};
use poot_llm::qwen_vl::{Qwen25Vl, Qwen25VlGenerate};
use poot_llm::{
    ChatTurn, GenerationControl, Qwen25VlApng, Qwen25VlApngError, Qwen25VlApngFrameSource,
    Qwen25VlChatContent, Qwen25VlChatMessage, Qwen25VlChatMessageContent, Sampler,
};
use poot_models::mrope::{MropeProcessorFpsInput, QWEN2_5_VL_PROCESSOR_DEFAULT_FPS};
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

/// Caption `media` with the Qwen2.5-VL checkpoint `model_dir`: an animated PNG is a video, anything
/// else an image.
fn caption_qwen_vl(
    model_dir: &str,
    media: &str,
    question: &str,
    choice: BackendChoice,
) -> Result<()> {
    let bytes = std::fs::read(media).with_context(|| format!("read {media}"))?;
    let mut vlm = Qwen25Vl::load(
        Path::new(model_dir),
        &Registry::builtin()?,
        (open_executor(choice)?, options(4096, 4096)),
        (open_executor(choice)?, options(8192, 256)),
    )
    .context("load the VLM")?;
    let apng = match Qwen25VlApng::from_bytes(&bytes) {
        Ok(apng) => Some(apng),
        Err(Qwen25VlApngError::StaticPng) => None,
        Err(error) => return Err(error).with_context(|| format!("probe {media}")),
    };
    let media_path = Path::new(media);
    let (part, fps, mut sources): (_, _, Vec<Qwen25VlApngFrameSource<'_>>) = match &apng {
        Some(apng) => (
            Qwen25VlChatContent::Video(apng.metadata()),
            Some(MropeProcessorFpsInput::Scalar(
                QWEN2_5_VL_PROCESSOR_DEFAULT_FPS,
            )),
            vec![apng.frame_source().context("open the video frames")?],
        ),
        None => (Qwen25VlChatContent::ImagePath(media_path), None, Vec::new()),
    };
    let messages = [Qwen25VlChatMessage {
        role: "user",
        content: Qwen25VlChatMessageContent::Parts(vec![part, Qwen25VlChatContent::Text(question)]),
    }];
    let generation = vlm
        .generate_chat(
            &messages,
            fps,
            &mut sources,
            Qwen25VlGenerate {
                max_new: 64,
                sampler: Sampler::greedy(),
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

    let config = std::fs::read(Path::new(model_dir.as_str()).join("config.json"))
        .with_context(|| format!("read {model_dir}/config.json"))?;
    let config: serde_json::Value = serde_json::from_slice(&config).context("parse config.json")?;
    if config["model_type"] == "qwen2_5_vl" {
        return caption_qwen_vl(model_dir, image_path, question, choice);
    }

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
                turns: &[ChatTurn::user(1, question)],
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
