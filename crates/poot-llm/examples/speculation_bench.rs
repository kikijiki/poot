//! Measure speculative decoding on a real checkpoint: one process runs one draft policy over one prompt
//! and prints one line, so `off`, `lookup`, `model` and `mtp` run on the same binary and prompt are comparable.
//!
//! The decode rate is timed from the token callback (first token to last, prefill excluded). The
//! acceptance numbers are the driver's own counters (`Driver::stats()`), summed over the measured
//! iterations: `acceptance` is accepted over proposed drafts, `pos_accept` the same per draft position
//! (`kept/proposed` for position 1, 2, ...), `p` the share of verified rounds that rejected a draft and
//! so restored and replayed a recurrent target. `hash` is an FNV-1a digest of the generated tokens: a
//! greedy run under any policy must print the digest `off` prints.
//!
//! `--timing 1` builds the driver with phase timing and adds the engine's own clock per verified
//! round, in ms: the draft passes of the decode window (an MTP layer's catch-up and chain, which read the output
//! head; the catch-up over the prompt belongs to prefill and is not counted),
//! the verify window, the snapshot, restore and replay copies, and `host` (the callback span per round
//! less those, so it also holds sampling and any step the counters do not cover). Timing snapshots the
//! executor at each step, so take `decode_tok_s` from a run without it.
//!
//! Run: `cargo run -p poot-llm --release --example speculation_bench -- --target DIR --policy off|lookup|model|mtp
//! [--draft DIR] [--backend wgpu|rocm|ptx|vulkan|auto] [--prompt-file F] [--new N] [--k K] [--ngram N]
//! [--warmup N] [--iters N] [--timing 0|1]`. `--draft` is required by (and only read under)
//! `--policy model`; its vocabulary and token ids must match the target's. `--policy mtp` drafts with
//! the target's own multi-token-prediction layer, `--k` tokens a round (a target with no such layer
//! is refused). `speculation_prompts/` holds a mixed-text and a repetitive-text prompt of about 570
//! tokens for `--prompt-file`. Use a fresh process per policy: a ROCm executor cannot be reopened
//! after a drop.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, open_executor,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::Registry;

/// A passage with the repetition real text has (names, phrases and clauses recur), for prompt lookup.
const DEFAULT_PROMPT: &str = "Summarize the following passage in one paragraph, then list its three \
main claims.\n\nThe lighthouse at Cape Verrin was built in 1851 to warn ships away from the Verrin \
shoals. The lighthouse keeper, Anna Holt, kept the lighthouse lamp burning every night for thirty \
years. In 1881 the lighthouse lamp was replaced with a lens from Paris, and the lighthouse keeper \
wrote that the lens made the lighthouse visible from twenty miles. After Anna Holt retired, the \
lighthouse was automated, and the lighthouse keeper's cottage became a museum about the Verrin \
shoals and the ships the lighthouse saved.\n\nSummary:";

fn flags(args: &[String]) -> Result<HashMap<&str, &str>> {
    let mut values = HashMap::new();
    let mut rest = args.iter().map(String::as_str);
    while let Some(flag) = rest.next() {
        let value = rest
            .next()
            .with_context(|| format!("{flag} needs a value"))?;
        values.insert(flag, value);
    }
    Ok(values)
}

fn number(values: &HashMap<&str, &str>, flag: &str, default: usize) -> Result<usize> {
    values.get(flag).map_or(Ok(default), |value| {
        value
            .parse()
            .with_context(|| format!("{flag} must be an integer; got {value:?}"))
    })
}

fn backend(name: Option<&str>) -> Result<BackendChoice> {
    Ok(match name {
        None | Some("auto") => BackendChoice::Auto,
        Some("wgpu") => BackendChoice::Wgpu,
        Some("rocm") => BackendChoice::Rocm,
        Some("ptx") => BackendChoice::Ptx,
        Some("vulkan") => BackendChoice::Vulkan,
        Some(other) => bail!("--backend {other}: expected wgpu, rocm, ptx, vulkan or auto"),
    })
}

fn load(dir: &str) -> Result<Arc<ModelHandle>> {
    Ok(Arc::new(ModelHandle::load(
        Path::new(dir),
        &Registry::builtin()?,
        &LoadOptions::default(),
    )?))
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn fnv1a(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |hash, token| {
        token.to_le_bytes().iter().fold(hash, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let values = flags(&args)?;
    let target_dir = values.get("--target").context("--target DIR is required")?;
    let k = NonZeroUsize::new(number(&values, "--k", 4)?).context("--k must be positive")?;
    let ngram =
        NonZeroUsize::new(number(&values, "--ngram", 2)?).context("--ngram must be positive")?;
    let new = number(&values, "--new", 128)?;
    let timed = number(&values, "--timing", 0)? != 0;
    let (warmup, iters) = (
        number(&values, "--warmup", 1)?,
        number(&values, "--iters", 3)?,
    );
    anyhow::ensure!(
        new > 1 && iters > 0,
        "--new must exceed 1 and --iters be positive"
    );

    let target = load(target_dir)?;
    let policy_name = values.get("--policy").context("--policy is required")?;
    let policy = match *policy_name {
        "off" => DraftPolicy::Off,
        "lookup" => DraftPolicy::PromptLookup { ngram, k },
        "model" => DraftPolicy::DraftModel {
            handle: load(
                values
                    .get("--draft")
                    .context("--policy model needs --draft DIR")?,
            )?,
            k,
        },
        "mtp" => DraftPolicy::Mtp { k },
        other => bail!("--policy {other}: expected off, lookup, model or mtp"),
    };
    let prompt = match values.get("--prompt-file") {
        Some(file) => std::fs::read_to_string(file).with_context(|| format!("reading {file}"))?,
        None => DEFAULT_PROMPT.to_string(),
    };
    let prompt_tokens = target.text().encode(&prompt)?;

    let granule = target.config().prefill_granule.get();
    let chunk = NonZeroUsize::new(128usize.div_ceil(granule) * granule).expect("nonzero");
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let options = DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: NonZeroUsize::new(prompt_tokens.len() + new).expect("nonzero"),
        prefill_chunk: chunk,
        trace: poot_llm::driver::serving_trace_limits(chunk),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: poot_llm::driver::program_retention,
        hidden_window: policy.carry_window(),
        draft: policy,
    };
    let mut driver = Driver::new(
        Arc::clone(&target),
        open_executor(backend(values.get("--backend").copied())?)?,
        options,
    )?;
    if timed {
        driver = driver.with_phase_timing();
    }
    let mut driver = driver.serve(Layout::Contiguous);

    let mut rates = Vec::new();
    let mut hash = 0;
    let mut before = driver.stats();
    let mut span = std::time::Duration::ZERO;
    for run in 0..warmup + iters {
        if run == warmup {
            before = driver.stats();
            span = std::time::Duration::ZERO;
        }
        let mut stamps = Vec::with_capacity(new);
        let generation = driver.generate(
            GenerateRequest {
                prompt: prompt_tokens.clone(),
                max_new: new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| {
                stamps.push(Instant::now());
                GenerationControl::Continue(())
            },
        )?;
        hash = fnv1a(&generation.tokens);
        if run < warmup {
            continue;
        }
        let run_span = stamps[stamps.len() - 1] - stamps[0];
        span += run_span;
        rates.push((stamps.len() - 1) as f64 / run_span.as_secs_f64());
    }
    let after = driver.stats();
    let measured = after.speculation.since(&before.speculation);
    let positions = after.positions.since(&before.positions);
    let rollback = after.rollback.since(before.rollback);
    let passes = after.passes.since(before.passes);
    let acceptance = if measured.proposed > 0 {
        format!("{:.3}", measured.accepted as f64 / measured.proposed as f64)
    } else {
        "n/a".to_string()
    };
    let by_position = if positions.proposed.is_empty() {
        "n/a".to_string()
    } else {
        positions
            .accepted
            .iter()
            .zip(&positions.proposed)
            .map(|(kept, proposed)| format!("{kept}/{proposed}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let p = if measured.rounds > 0 {
        format!("{:.3}", measured.restored as f64 / measured.rounds as f64)
    } else {
        "n/a".to_string()
    };
    println!(
        "policy={policy_name} prompt_tokens={} new={new} decode_tok_s={:.1} acceptance={acceptance} \
         rounds={} proposed={} accepted={} skipped_no_room={} skipped_no_proposal={} restored={} \
         replayed={} pos_accept={by_position} p={p} hash={hash:016x}",
        prompt_tokens.len(),
        median(rates),
        measured.rounds,
        measured.proposed,
        measured.accepted,
        measured.skipped_no_room,
        measured.skipped_no_proposal,
        measured.restored,
        measured.replayed,
    );
    if timed && measured.rounds > 0 {
        let per_round = |time: poot_llm::driver::StepTime| {
            time.wall.as_secs_f64() * 1e3 / measured.rounds as f64
        };
        let engine = [
            passes.drafts,
            passes.verifies,
            rollback.snapshots,
            rollback.restores,
            rollback.replays,
        ]
        .map(per_round);
        let host = span.as_secs_f64() * 1e3 / measured.rounds as f64 - engine.iter().sum::<f64>();
        println!(
            "round_ms draft={:.2} verify={:.2} snapshot={:.2} restore={:.2} replay={:.2} host={host:.2} \
             draft_passes_per_round={:.2}",
            engine[0],
            engine[1],
            engine[2],
            engine[3],
            engine[4],
            passes.drafts.steps as f64 / measured.rounds as f64,
        );
    }
    Ok(())
}
