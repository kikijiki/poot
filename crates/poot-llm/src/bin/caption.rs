//! poot-caption: caption an image with a vision-language model. Loads a SmolVLM / idefics3 checkpoint, decodes
//! the image, and prints a caption.
//!
//! Usage: `poot-caption <model_dir> <image_path> [question]`

use anyhow::{Context, Result};
use poot_llm::vlm::{VlmRunner, decode_image};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .context("usage: poot-caption <model_dir> <image_path> [question]")?;
    let image_path = args.get(2).context("need an image path")?;
    let question = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("Can you describe this image?");

    let bytes = std::fs::read(image_path).with_context(|| format!("read {image_path}"))?;
    let (rgb, h, w) = decode_image(&bytes).context("decode image")?;
    eprintln!("loaded {image_path} ({w}x{h}); loading model from {model_dir}...");

    let vlm = VlmRunner::load(model_dir).context("load VLM")?;
    let mut engine =
        poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().context("init GPU")?);
    let exe = vlm.load_on(&mut engine).context("vlm load_on")?;
    let caption = vlm
        .caption(&rgb, h, w, question, 64, &mut engine, exe)
        .context("caption")?;
    println!("{}", caption.trim());
    Ok(())
}
