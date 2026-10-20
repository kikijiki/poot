//! `poot`: the command line over the driver. One binary, one `--backend` parser, three commands:
//! `generate` (text from a prompt), `caption` (an image through a vision-language model) and `embed`
//! (sentence embeddings and reranking).

mod caption;
mod embed;
mod generate;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use poot_llm::driver::BackendChoice;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "poot", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate text from a prompt, or hold a conversation.
    Generate(generate::Args),
    /// Caption an image with a vision-language model.
    Caption(caption::Args),
    /// Sentence embeddings and reranking with an encoder.
    Embed(embed::Args),
}

/// The device a command runs on.
#[derive(Debug, Args)]
struct DeviceArgs {
    /// The device: wgpu, rocm, ptx, vulkan, or auto (PTX if a CUDA device opens, else ROCm if an HSA GPU
    /// agent opens, else wgpu).
    #[arg(long, default_value_t = BackendChoice::Auto)]
    backend: BackendChoice,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = matches!(&cli.command, Command::Generate(args) if args.verbose());
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| if verbose { "info" } else { "warn" }.into()),
        )
        .init();
    match cli.command {
        Command::Generate(args) => generate::run(args),
        Command::Caption(args) => caption::run(args),
        Command::Embed(args) => embed::run(args),
    }
}
