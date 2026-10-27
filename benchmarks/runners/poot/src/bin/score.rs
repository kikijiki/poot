//! The next-token distribution at the positions of a fixed token sequence, and the greedy continuation of it,
//! from the production driver: poot's side of the llama.cpp correctness comparison (POOT-1333), and the KV cache
//! dtype drift measurement (POOT-1395, two runs that differ only in `--kv-dtype`, compared by
//! `scripts/kv-drift/compare.py`).
//!
//!   poot-score --model <dir|gguf> (--tokens <ids.txt> | --text <file>) --start <S> [--top 20] [--greedy 64]
//!              [--backend ptx] [--kv-dtype f16|bf16|f32] --out <json>
//!
//! `ids.txt` holds whitespace-separated token ids (the ones `llama-perplexity --save-all-logits` wrote into its
//! reference, see `scripts/flash-next-correctness/compare.py`); `--text` encodes a file with the checkpoint's own
//! tokenizer instead. `--kv-dtype` is the driver's explicit KV cache dtype; without it the driver resolves the
//! model's preference or the engine default, and the report names the dtype it ran. The sequence is fed
//! teacher-forced through
//! [`Driver::generate_turn`]: the turn for position `p` has the prompt `tokens[..=p]`, so the K/V of the
//! earlier positions stays resident and only the new token is prefilled, and the one token it asks for carries
//! the top `--top` log-probabilities of the distribution of the token at `p + 1`. The first turn prefills
//! `tokens[..=S]` in the chunks the device budget fits (`fit_prefill_chunk`, as the benchmark runner opens it),
//! so a model whose whole-prompt arena does not fit is scored through the chunked prefill. The greedy
//! continuation is a fresh generation from the whole sequence with EOS ignored.
//!
//! Output (one JSON object): the device, the KV dtype, the chunk, the scored token `ids` (a `--tokens` file of
//! them replays the run), `positions` (`pos`, `top` as `[token, logprob]` pairs, most
//! probable first) and `greedy` (`ids`, `text`).

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use serde_json::json;

use poot_llm::driver::capacity::{chunk_candidates, fit_prefill_chunk, Observation};
use poot_llm::driver::error::DriverError;
use poot_llm::driver::{
    open_executor, BackendChoice, Conversation, Driver, DriverOptions, GenerateRequest, Head,
    Layout, LoadOptions, ModelHandle, PreparedSetLimits, ServingShapes,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::KvDtype;
use poot_models::registry::Registry;

/// Where the scored sequence comes from.
enum Sequence {
    /// Whitespace-separated token ids.
    Ids(PathBuf),
    /// Text the checkpoint's tokenizer encodes.
    Text(PathBuf),
}

struct Args {
    model: PathBuf,
    sequence: Sequence,
    kv_dtype: Option<KvDtype>,
    start: usize,
    top: usize,
    greedy: usize,
    backend: BackendChoice,
    out: PathBuf,
}

/// Every flag takes a value. Anything else is an error: a typo (`--gredy 0`) must not run with the default.
const FLAGS: [&str; 9] = [
    "--model",
    "--tokens",
    "--text",
    "--kv-dtype",
    "--start",
    "--top",
    "--greedy",
    "--backend",
    "--out",
];

fn parse_args() -> anyhow::Result<Args> {
    let mut values = std::collections::HashMap::new();
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        anyhow::ensure!(
            FLAGS.contains(&flag.as_str()),
            "unknown argument {flag:?} (flags: {})",
            FLAGS.join(", ")
        );
        let value = rest
            .next()
            .with_context(|| format!("{flag} needs a value"))?;
        values.insert(flag, value);
    }
    let take = |flag: &str| -> anyhow::Result<String> {
        values
            .get(flag)
            .cloned()
            .with_context(|| format!("{flag} is required"))
    };
    let number = |flag: &str, default: usize| -> anyhow::Result<usize> {
        values.get(flag).map_or(Ok(default), |value| {
            value
                .parse()
                .with_context(|| format!("{flag} must be an integer; got {value:?}"))
        })
    };
    let sequence = match (values.get("--tokens"), values.get("--text")) {
        (Some(ids), None) => Sequence::Ids(ids.into()),
        (None, Some(text)) => Sequence::Text(text.into()),
        _ => anyhow::bail!("exactly one of --tokens and --text is required"),
    };
    let args = Args {
        model: take("--model")?.into(),
        sequence,
        kv_dtype: values
            .get("--kv-dtype")
            .map(|name| name.parse::<KvDtype>())
            .transpose()?,
        start: take("--start")?.parse().context("--start")?,
        top: number("--top", 20)?,
        greedy: number("--greedy", 64)?,
        backend: values
            .get("--backend")
            .map_or("ptx", String::as_str)
            .parse()?,
        out: take("--out")?.into(),
    };
    anyhow::ensure!(args.top > 0, "--top must be positive");
    Ok(args)
}

fn read_tokens(path: &Path) -> anyhow::Result<Vec<u32>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .split_whitespace()
        .map(|id| {
            id.parse()
                .with_context(|| format!("token id {id:?} is not an integer"))
        })
        .collect()
}

fn request(prompt: &[u32], max_new: usize, sampler: Sampler, ignore_eos: bool) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_vec(),
        max_new,
        sampler,
        stops: Vec::new(),
        ignore_eos,
    }
}

fn main() -> anyhow::Result<()> {
    // `RUST_LOG=poot_llm::device_memory=info` with `POOT_DEVICE_MEMORY_LOG=1` prints the device memory per phase.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let args = parse_args()?;
    let registry = Registry::builtin().context("build the family registry")?;
    let handle = Arc::new(
        ModelHandle::load(&args.model, &registry, &LoadOptions::default())
            .with_context(|| format!("load {}", args.model.display()))?,
    );
    let tokens = match &args.sequence {
        Sequence::Ids(path) => read_tokens(path)?,
        Sequence::Text(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            handle.text().encode(&text).context("encode --text")?
        }
    };
    anyhow::ensure!(
        args.start + 1 < tokens.len(),
        "--start {} leaves no scored position in {} tokens",
        args.start,
        tokens.len()
    );
    let capacity = NonZeroUsize::new(tokens.len() + args.greedy).expect("nonzero");
    let executor = open_executor(args.backend).context("open the executor")?;
    let device = executor
        .device_identity()
        .context("the device cannot name its adapter and driver")?;

    let scored = || Sampler::greedy().with_logprobs(args.top);
    let wanted = NonZeroUsize::new(tokens.len()).expect("nonzero");
    let options = DriverOptions {
        trace: poot_llm::driver::serving_trace_limits(capacity),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(8 << 30).expect("nonzero"),
        },
        kv_dtype: args.kv_dtype,
        ..DriverOptions::generation(capacity, wanted)
    };
    // The turns extend the prepared set to the admitted pieces of both heads: the chunk is fitted to that set,
    // so the first turn allocates nothing the fit did not count (POOT-1347).
    let heads = [Head::serving(&scored()), Head::GREEDY];
    let shapes = ServingShapes::turn(Layout::Contiguous, &heads);
    let fitted = fit_prefill_chunk(
        &handle,
        &executor.target_set(),
        || Observation::of(&*executor),
        &options,
        &shapes,
        &chunk_candidates(wanted, handle.config().prefill_granule),
    )
    .context("fit the prefill chunk to the device")?;
    eprintln!(
        "poot-score: prefill chunk {} of {} tokens",
        fitted.chunk, wanted
    );
    let options = DriverOptions {
        prefill_chunk: fitted.chunk,
        ..options
    };
    let mut driver = Driver::new(Arc::clone(&handle), executor, options)
        .context("open the driver")?
        .prepare(&shapes)
        .map_err(DriverError::from)
        .context("prepare the driver")?;

    let mut positions = Vec::new();
    let mut conversation = Conversation::default();
    for position in args.start..tokens.len() - 1 {
        let generation = driver.generate_turn(
            &mut conversation,
            request(&tokens[..=position], 1, scored(), true),
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )?;
        let record = generation
            .logprobs
            .first()
            .with_context(|| format!("no log-probabilities for position {position}"))?;
        positions.push(json!({ "pos": position, "top": record.top }));
        if (position - args.start) % 128 == 0 {
            eprintln!("poot-score: position {position}/{}", tokens.len() - 1);
        }
    }
    driver.end_conversation(conversation)?;

    let mut ids = Vec::new();
    let mut text = String::new();
    driver.generate(
        request(&tokens, args.greedy, Sampler::greedy(), true),
        &mut |id: u32, piece: &str| {
            ids.push(id);
            text.push_str(piece);
            GenerationControl::Continue(())
        },
    )?;

    let report = json!({
        "adapter": device.adapter,
        "driver": device.driver,
        "kv_dtype": driver.kv_dtype().as_str(),
        "prefill_chunk": fitted.chunk.get(),
        "tokens": tokens.len(),
        "ids": tokens,
        "start": args.start,
        "positions": positions,
        "greedy": { "ids": ids, "text": text },
    });
    std::fs::write(&args.out, serde_json::to_vec(&report)?)
        .with_context(|| format!("write {}", args.out.display()))?;
    Ok(())
}
