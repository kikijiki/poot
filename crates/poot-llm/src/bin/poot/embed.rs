//! `poot embed`: sentence embeddings and reranking with a BERT-class model on a device. Loads an embedding
//! encoder (all-MiniLM, E5, BGE) or a cross-encoder (ms-marco-MiniLM) through the driver's `Encode` step
//! and prints cosine similarities to the first text, or the ranking of the documents against a query.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::ValueEnum;
use poot_llm::driver::open_executor;
use poot_llm::encoders::{CrossEncoder, SentenceEncoder};

use crate::DeviceArgs;

/// What to do with the texts.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    /// The cosine of each later text to the first.
    Similarity,
    /// Bi-encoder rerank by cosine: the first text is the query, the rest are documents.
    Rerank,
    /// Cross-encoder rerank by the raw relevance logit: the first text is the query.
    Cross,
}

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// What to compute.
    #[arg(value_enum)]
    mode: Mode,
    /// The encoder checkpoint directory (a cross-encoder for `cross`).
    model_dir: PathBuf,
    /// The texts: for `rerank` and `cross`, the query then the documents.
    #[arg(required = true, num_args = 1..)]
    texts: Vec<String>,
    #[command(flatten)]
    device: DeviceArgs,
}

/// The rows one step is prepared for: as many as the texts on the command line.
fn rows_of(count: usize) -> Result<NonZeroUsize> {
    NonZeroUsize::new(count).context("no text to encode")
}

pub(crate) fn run(args: Args) -> Result<()> {
    let Args {
        mode,
        model_dir,
        texts,
        device,
    } = args;
    let executor = open_executor(device.backend)?;
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    match mode {
        Mode::Similarity => {
            let mut encoder = SentenceEncoder::load(&model_dir, executor, rows_of(texts.len())?)?;
            println!("pooling {:?}", encoder.pooling());
            let vectors = encoder.embed(&texts)?;
            for (text, vector) in texts.iter().zip(&vectors).skip(1) {
                let cosine: f32 = vectors[0].iter().zip(vector).map(|(a, b)| a * b).sum();
                println!("{cosine:+.3}  {text}");
            }
        }
        Mode::Rerank | Mode::Cross => {
            let [query, docs @ ..] = texts.as_slice() else {
                unreachable!("clap requires at least one text");
            };
            let ranked = if matches!(mode, Mode::Rerank) {
                SentenceEncoder::load(&model_dir, executor, rows_of(texts.len())?)?
                    .rerank(query, docs)?
            } else {
                CrossEncoder::load(&model_dir, executor, rows_of(docs.len())?)?
                    .rerank(query, docs)?
            };
            for (rank, (index, score)) in ranked.iter().enumerate() {
                println!("{}. {score:+.3}  {}", rank + 1, docs[*index]);
            }
        }
    }
    Ok(())
}
