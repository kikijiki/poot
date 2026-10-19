//! poot-embed: sentence embeddings and reranking with a BERT-class model on a device. Loads an embedding
//! encoder (all-MiniLM, E5, BGE) or a cross-encoder (ms-marco-MiniLM) through the driver's `Encode` step
//! and prints cosine similarities to the first text, or the ranking of the documents against a query.
//!
//! Usage:
//!   `poot-embed embed <encoder_dir> <text> <text>... [--backend B]`: the cosine of each later text to the first.
//!   `poot-embed rerank <encoder_dir> <query> <doc> <doc>... [--backend B]`: bi-encoder rerank by cosine.
//!   `poot-embed cross <cross_encoder_dir> <query> <doc> <doc>... [--backend B]`: cross-encoder rerank by
//!   the raw relevance logit.
//!
//! `--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto` (the default).

use std::path::Path;

use anyhow::{Context, Result, bail};
use poot_llm::driver::{BackendChoice, open_executor};
use poot_llm::encoders::{CrossEncoder, SentenceEncoder};

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

/// The rows one step is prepared for: as many as the texts on the command line.
fn rows_of(count: usize) -> Result<std::num::NonZeroUsize> {
    std::num::NonZeroUsize::new(count).context("no text to encode")
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut positional, mut backend) = (Vec::new(), None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if arg == "--backend" {
            backend = Some(it.next().context("--backend needs a value")?.as_str());
        } else {
            positional.push(arg.as_str());
        }
    }
    let usage = "usage: poot-embed <embed|rerank|cross> <model_dir> <text>... [--backend B]";
    let [mode, dir, rest @ ..] = positional.as_slice() else {
        bail!(usage);
    };
    let (dir, executor) = (Path::new(dir), open_executor(backend_choice(backend)?)?);
    match *mode {
        "embed" => {
            let mut encoder = SentenceEncoder::load(dir, executor, rows_of(rest.len())?)?;
            println!("pooling {:?}", encoder.pooling());
            let vectors = encoder.embed(rest)?;
            for (text, vector) in rest.iter().zip(&vectors).skip(1) {
                let cosine: f32 = vectors[0].iter().zip(vector).map(|(a, b)| a * b).sum();
                println!("{cosine:+.3}  {text}");
            }
        }
        "rerank" | "cross" => {
            let [query, docs @ ..] = rest else {
                bail!(usage);
            };
            let ranked = if *mode == "rerank" {
                SentenceEncoder::load(dir, executor, rows_of(rest.len())?)?.rerank(query, docs)?
            } else {
                CrossEncoder::load(dir, executor, rows_of(docs.len())?)?.rerank(query, docs)?
            };
            for (rank, (index, score)) in ranked.iter().enumerate() {
                println!("{}. {score:+.3}  {}", rank + 1, docs[*index]);
            }
        }
        _ => bail!(usage),
    }
    Ok(())
}
