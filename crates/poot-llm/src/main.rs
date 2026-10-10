//! poot-generate: load a checkpoint and generate text from a prompt, with the flags of llama.cpp's
//! `llama-cli` where poot has the capability.
//!
//! A checkpoint loads as a `driver::ModelHandle` and generates through the driver on the device
//! `--backend` names (`wgpu`, `rocm`, `ptx`, `vulkan`, or `auto`, the default). A family the registry
//! does not know is a typed error.
//!
//! A model that ships a chat template answers the prompt as a user turn of it (after `--system-prompt`),
//! and prints the reply only; `--no-chat-template`, or a model with no template (a base model), continues
//! the prompt as written and prints it first. `--chat-template` (jinja text) or `--chat-template-file`
//! renders through that template instead of the model's own, so a base model answers turns too. `-n`
//! defaults to -1: run to EOS, a `-r` string or the end
//! of the context window (`-c`, default the model's own). `--temp` defaults to 0, greedy, which is
//! poot's exact-token convention (llama-cli samples at 0.80).
//!
//! `--conversation` answers a chat of turns instead of one prompt: each line of stdin is a user turn,
//! the reply streams to stdout, and the model keeps the K/V of the conversation so far (a turn prefills
//! its new tokens only). `-p`/`-f` is then the first user turn, answered before stdin is read, and
//! `--single-turn` ends the conversation after one reply (the turn from `-p`/`-f`, else the first line of
//! stdin). Unlike llama-cli they have no `-cnv`/`-st` short forms: clap takes single-character shorts only.
//! A model with no chat template cannot hold a conversation.
//!
//! Without `--model` the model is `qwen2.5-0.5b` under the `POOT_MODELS_DIR` directory; with neither,
//! it exits with an error naming the variable. ROCm is a build feature: `--features cli,rocm`.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, ValueEnum};
use poot_llm::ChatTemplateOverride;
use poot_llm::driver::{
    BackendChoice, ChatOptions, ChatSession, Generation, ModelHandle, PhaseTiming, Sampling,
    TextRequest, generate_text, open_executor,
};
use poot_models::registry::Registry;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Auto,
    Wgpu,
    Rocm,
    Ptx,
    Vulkan,
}

impl From<Backend> for BackendChoice {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Auto => Self::Auto,
            Backend::Wgpu => Self::Wgpu,
            Backend::Rocm => Self::Rocm,
            Backend::Ptx => Self::Ptx,
            Backend::Vulkan => Self::Vulkan,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "poot-generate", version, about)]
#[command(group(ArgGroup::new("input").args(["prompt", "file"])))]
struct Args {
    /// The checkpoint: a safetensors directory or a `.gguf` file. Default: `qwen2.5-0.5b` under
    /// `POOT_MODELS_DIR`.
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,
    /// The prompt; with `--conversation`, the first user turn.
    #[arg(short = 'p', long, required_unless_present_any = ["file", "conversation", "single_turn"])]
    prompt: Option<String>,
    /// Read the prompt from a file.
    #[arg(short = 'f', long, required_unless_present_any = ["prompt", "conversation", "single_turn"])]
    file: Option<PathBuf>,
    /// A system turn ahead of the prompt, through the model's chat template. Unlike llama-cli it has no
    /// short form (`-sys`): clap takes single-character shorts only.
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,
    /// Use the prompt as written even when the model ships a chat template.
    #[arg(long = "no-chat-template", conflicts_with_all = ["conversation", "single_turn"])]
    no_chat_template: bool,
    /// Render chats through this jinja template instead of the model's own. Unlike llama-cli it takes
    /// no built-in template name: a value with no jinja syntax is refused.
    #[arg(
        long = "chat-template",
        value_name = "JINJA",
        conflicts_with_all = ["no_chat_template", "chat_template_file"]
    )]
    chat_template: Option<String>,
    /// Render chats through the jinja template in this file instead of the model's own.
    #[arg(
        long = "chat-template-file",
        value_name = "FILE",
        conflicts_with = "no_chat_template"
    )]
    chat_template_file: Option<PathBuf>,
    /// Converse: each line of stdin is a user turn, answered through the chat template with the earlier
    /// turns kept in the context. `-p`/`-f` is the first turn.
    #[arg(long)]
    conversation: bool,
    /// A conversation of one turn: answer `-p`/`-f`, else the first line of stdin, and exit.
    #[arg(long = "single-turn")]
    single_turn: bool,
    /// New tokens to generate at most; -1 runs to EOS, a reverse prompt or the end of the context window.
    #[arg(
        short = 'n',
        long = "predict",
        default_value_t = -1,
        allow_negative_numbers = true,
        value_parser = clap::value_parser!(i64).range(-1..)
    )]
    predict: i64,
    /// The context window in tokens (prompt plus new tokens); 0 takes the model's own.
    #[arg(short = 'c', long = "ctx-size", default_value_t = 0)]
    ctx_size: usize,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    #[arg(long = "top-k", default_value_t = 40)]
    top_k: usize,
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,
    #[arg(long = "min-p", default_value_t = 0.05)]
    min_p: f32,
    /// Repetition penalty over the prompt and the output; 1 is off.
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,
    /// Sampling seed; default a fresh one per run.
    #[arg(short = 's', long)]
    seed: Option<u64>,
    /// Stop when this string is generated (repeatable); it is printed with the token that completes it.
    #[arg(short = 'r', long = "reverse-prompt")]
    reverse_prompt: Vec<String>,
    /// Generate through the model's end-of-sequence token.
    #[arg(long = "ignore-eos")]
    ignore_eos: bool,
    /// The device.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Print the engine-measured prefill and decode tokens per second to stderr (the default).
    #[arg(long = "show-timings", overrides_with = "no_show_timings")]
    show_timings: bool,
    /// Do not print the timings.
    #[arg(long = "no-show-timings", overrides_with = "show_timings")]
    no_show_timings: bool,
    /// Log what the run does (loading, the model's configuration, the seed) to stderr.
    #[arg(short = 'v', long)]
    verbose: bool,
}

/// No model path was given and `POOT_MODELS_DIR` is unset or empty.
#[derive(Debug, thiserror::Error)]
#[error(
    "no model given and POOT_MODELS_DIR is not set: pass --model or set POOT_MODELS_DIR to the directory that holds `{DEFAULT_MODEL}`"
)]
struct ModelDirUnset;

/// The model directory under `POOT_MODELS_DIR` used when no path is given.
const DEFAULT_MODEL: &str = "qwen2.5-0.5b";

/// The model path: `--model`, else `DEFAULT_MODEL` under `POOT_MODELS_DIR`.
fn model_path_or_env(flag: Option<&Path>) -> Result<PathBuf, ModelDirUnset> {
    if let Some(path) = flag {
        return Ok(path.to_path_buf());
    }
    match std::env::var("POOT_MODELS_DIR") {
        Ok(dir) if !dir.is_empty() => Ok(Path::new(&dir).join(DEFAULT_MODEL)),
        _ => Err(ModelDirUnset),
    }
}

fn fresh_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64)
}

impl Args {
    /// Timings print unless `--no-show-timings` came last.
    fn timings(&self) -> bool {
        self.show_timings || !self.no_show_timings
    }

    /// A conversation, not a single prompt.
    fn conversing(&self) -> bool {
        self.conversation || self.single_turn
    }

    /// The text of `-p` or `-f`.
    fn prompt_text(&self) -> Result<Option<String>> {
        match (&self.prompt, &self.file) {
            (Some(prompt), _) => Ok(Some(prompt.clone())),
            (None, Some(file)) => std::fs::read_to_string(file)
                .with_context(|| format!("reading the prompt file {}", file.display()))
                .map(Some),
            (None, None) => Ok(None),
        }
    }

    /// The chat template `--chat-template` or `--chat-template-file` gives.
    fn chat_template_override(&self) -> Option<ChatTemplateOverride> {
        ChatTemplateOverride::from_flags(
            self.chat_template.clone(),
            self.chat_template_file.clone(),
        )
    }

    fn sampling(&self) -> Sampling {
        let seed = self.seed.unwrap_or_else(fresh_seed);
        tracing::info!(seed, "sampling seed");
        Sampling {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            seed,
        }
    }

    fn request(&self, prompt: String) -> TextRequest {
        TextRequest {
            prompt,
            system: self.system_prompt.clone(),
            raw: self.no_chat_template,
            stops: self.reverse_prompt.clone(),
            max_new: usize::try_from(self.predict).ok(),
            ctx: (self.ctx_size != 0).then_some(self.ctx_size),
            sampling: self.sampling(),
            ignore_eos: self.ignore_eos,
        }
    }

    fn chat_options(&self) -> ChatOptions {
        ChatOptions {
            system: self.system_prompt.clone(),
            stops: self.reverse_prompt.clone(),
            max_new: usize::try_from(self.predict).ok(),
            ctx: (self.ctx_size != 0).then_some(self.ctx_size),
            sampling: self.sampling(),
            ignore_eos: self.ignore_eos,
        }
    }
}

/// The user turns of a conversation, in order: `first` (the `-p`/`-f` text) if there is one, then one
/// per nonblank line of `input`. `single_turn` ends after the first.
fn user_turns(
    first: Option<String>,
    input: impl BufRead,
    single_turn: bool,
) -> impl Iterator<Item = std::io::Result<String>> {
    let turns = first.map(Ok).into_iter().chain(
        input
            .lines()
            .filter(|line| line.as_ref().map_or(true, |line| !line.trim().is_empty())),
    );
    turns.take(if single_turn { 1 } else { usize::MAX })
}

/// A conversation on `session` over `turns`: each reply streams to stdout.
fn converse(
    session: &mut ChatSession,
    turns: impl Iterator<Item = std::io::Result<String>>,
    args: &Args,
) -> Result<()> {
    let interactive = std::io::stdin().is_terminal();
    for turn in turns {
        let turn = turn.context("reading a user turn from stdin")?;
        let generation = session.say(&turn, &mut |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        })?;
        println!();
        tracing::info!(
            tokens = generation.tokens.len(),
            finish = ?generation.finish,
            "turn complete"
        );
        if args.timings() {
            report_timings(&generation);
        }
        if interactive {
            eprint!("> ");
        }
    }
    Ok(())
}

fn report(phase: &str, timing: PhaseTiming) {
    match timing.tokens_per_second() {
        Some(rate) => eprintln!(
            "{phase}: {} tokens in {:.1} ms ({rate:.1} tokens/s)",
            timing.tokens,
            timing.wall.as_secs_f64() * 1e3
        ),
        None => eprintln!("{phase}: {} tokens, no engine time measured", timing.tokens),
    }
}

fn report_timings(generation: &Generation) {
    // `generate_text` always runs a driver that measures its phases.
    if let Some(timing) = generation.timing {
        report("prefill", timing.prefill);
        report("decode", timing.decode);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let level = if args.verbose { "info" } else { "warn" };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into()),
        )
        .init();
    let first = args.prompt_text()?;
    let model_path = model_path_or_env(args.model.as_deref())?;

    // A backend this build lacks, or a host without the device, fails before the model loads.
    let executor = open_executor(args.backend.into())?;
    tracing::info!(model_path = %model_path.display(), "loading model");
    let registry = Registry::builtin()?;
    let handle = Arc::new(ModelHandle::load(
        &model_path,
        &registry,
        &poot_llm::driver::LoadOptions {
            chat_template: args.chat_template_override(),
            ..Default::default()
        },
    )?);
    let config = handle.config();
    tracing::info!(
        family = %config.family,
        vocab = config.vocab,
        max_positions = config.max_positions,
        backend = ?args.backend,
        "model config"
    );
    if args.conversing() {
        let mut session = ChatSession::new(&handle, executor, args.chat_options())?;
        if std::io::stdin().is_terminal() && first.is_none() {
            eprint!("> ");
        }
        let turns = user_turns(first, std::io::stdin().lock(), args.single_turn);
        converse(&mut session, turns, &args)?;
        session.finish()?;
        return Ok(());
    }
    let request = args.request(first.expect("clap requires -p or -f without a conversation"));
    // Generated text is program output (stdout), distinct from status logs (stderr).
    let generation = generate_text(&handle, executor, &request, &mut |piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    })?;
    println!();
    tracing::info!(
        tokens = generation.tokens.len(),
        finish = ?generation.finish,
        "generation complete"
    );
    if args.timings() {
        report_timings(&generation);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(first: Option<&str>, input: &str, single_turn: bool) -> Vec<String> {
        user_turns(first.map(str::to_string), input.as_bytes(), single_turn)
            .collect::<std::io::Result<_>>()
            .unwrap()
    }

    /// A conversation's turns are the `-p`/`-f` text, then each nonblank line of stdin, in order.
    /// Mutation: keep blank lines.
    #[test]
    fn the_prompt_is_the_first_turn_and_each_nonblank_stdin_line_the_next() {
        assert_eq!(
            turns(Some("one"), "two\n\n  \nthree\n", false),
            ["one", "two", "three"]
        );
        assert_eq!(turns(None, "two\nthree", false), ["two", "three"]);
        assert!(turns(None, "", false).is_empty());
    }

    /// `--single-turn` takes the first turn only: the prompt if there is one, else the first stdin line, and
    /// reads nothing past it. Mutation: drop the `take`.
    #[test]
    fn a_single_turn_takes_one_turn_and_reads_no_further() {
        assert_eq!(turns(Some("one"), "two\n", true), ["one"]);
        assert_eq!(turns(None, "\ntwo\nthree\n", true), ["two"]);
        let mut input = "two\nthree\n".as_bytes();
        let taken: Vec<_> = user_turns(None, &mut input, true).collect();
        assert_eq!(taken.len(), 1);
        assert_eq!(input, b"three\n", "stdin was read past the one turn");
    }
}
