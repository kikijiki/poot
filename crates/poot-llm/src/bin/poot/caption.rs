//! `poot caption`: caption an image with a vision-language model. Loads the checkpoint (its text decoder
//! and the vision towers its registry entry holds, each on a driver), decodes the image and prints a
//! caption. A checkpoint whose front end takes video (Qwen2.5-VL) also takes an animated PNG as a video,
//! sampled at the processor's default two frames per second.

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use poot_llm::driver::{
    Driver, DriverOptions, GenerateRequest, Layout, LoadOptions, ModelHandle, open_executor,
};
use poot_llm::{Apng, ApngError, FrontEndSession, GenerationControl, Sampler, decode_image};
use poot_models::multimodal::front_end::{ChatTurn, Media, Video};
use poot_models::registry::Registry;

use crate::DeviceArgs;

/// The text decoder's window: the whole prompt (every frame's rows and the question) and the 64 new tokens.
const TEXT_CAPACITY: usize = 8192;
/// The text decoder prefills in pieces of at most this many tokens.
const TEXT_PREFILL_CHUNK: usize = 256;

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// The checkpoint directory.
    model_dir: PathBuf,
    /// The image, or an animated PNG as a video.
    image: PathBuf,
    /// What to ask about it.
    #[arg(default_value = "Can you describe this image?")]
    question: String,
    #[command(flatten)]
    device: DeviceArgs,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let Args {
        model_dir,
        image: image_path,
        question,
        device,
    } = args;
    let choice = device.backend;

    let handle = Arc::new(
        ModelHandle::load(&model_dir, &Registry::builtin()?, &LoadOptions::default())
            .context("load the checkpoint")?,
    );
    let mut session = FrontEndSession::open(Arc::clone(&handle), |_, positions| {
        Ok::<_, anyhow::Error>((open_executor(choice)?, DriverOptions::tower(positions)))
    })
    .context("open the vision towers")?
    .context("the checkpoint is not a vision-language model: its family has no front end")?;

    let bytes =
        std::fs::read(&image_path).with_context(|| format!("read {}", image_path.display()))?;
    eprintln!(
        "loaded {}; loading model from {}...",
        image_path.display(),
        model_dir.display()
    );
    // An animated PNG is a video; any other file is one image.
    let apng = match Apng::from_bytes(&bytes) {
        Ok(apng) => Some(apng),
        Err(ApngError::StaticPng) => None,
        Err(error) => return Err(error).with_context(|| format!("probe {}", image_path.display())),
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
        .prepare(&[ChatTurn::user(1, &question)], &mut media)
        .context("prepare the prompt")?;

    // The text decoder holds the whole prompt (every frame's rows and the question) and the 64 new tokens.
    let options = DriverOptions::generation(
        NonZeroUsize::new(TEXT_CAPACITY).expect("nonzero"),
        NonZeroUsize::new(TEXT_PREFILL_CHUNK).expect("nonzero"),
    );
    let mut text = Driver::new(handle, open_executor(choice)?, options)
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
