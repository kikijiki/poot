//! POOT-1312: poot's LongRoPE tables against llama.cpp's, on the real Phi-4-mini and Phi-3.5-mini.
//!
//! Both models are `Phi3ForCausalLM` with a `longrope` block: a short and a long per-frequency divisor table
//! and an attention scale. llama.cpp picks the table from the context size (`n_ctx > n_ctx_orig`, 4096 for both
//! models), poot from the step capacity; the attention scale applies in both regimes. So the comparison runs in
//! two regimes, each a teacher-forced window scored by `llama-perplexity` on an F16 GGUF of the same
//! checkpoint and by poot (the CPU oracle, or the ROCm executor) on the HF safetensors:
//!
//! - a short window of [`SHORT_CTX`] tokens (the short table on both sides), and
//! - a long window of [`LONG_CTX`] tokens, past the original context with a table capacity above it, whose
//!   scored half is positions 3072 to 6142, of which 2047 lie beyond 4096 (the long table on both sides).
//!
//! The reference is `llama.cpp`'s own `convert_hf_to_gguf.py --outtype f16` of the same safetensors
//! (`Phi-4-mini-instruct-F16.gguf`, `Phi-3.5-mini-instruct-F16.gguf` next to the Q4_K_M files in the models
//! directory), not a quantized file: an F16 GGUF and poot's bf16-as-f32 path are both near exact, so the
//! comparison tests the tables and not quantization noise. (Against the Q4_K_M files the same windows score
//! mean KL 0.02 to 0.07 with single positions at 2.5 to 3.4, which says nothing about the tables.) The row
//! fails under `POOT_REQUIRE_MODELS=1` when the F16 file is absent; `scripts/make-f16-ggufs.sh` produces both
//! into `$POOT_MODELS_DIR`.
//!
//! Measured on ROCm against llama.cpp `d812350`, every scored position of every window:
//!
//! | window, model              | positions | mean KL | max KL  | NLL delta | top-1  | top-5 overlap | max abs log-prob |
//! |----------------------------|-----------|---------|---------|-----------|--------|---------------|------------------|
//! | short x5, Phi-4-mini       | 315       | 1.0e-6  | 6.0e-6  | -1.3e-5   | 1.0    | 1.0           | 0.026            |
//! | short x5, Phi-3.5-mini     | 315       | 5.9e-7  | 3.5e-6  | +1.6e-5   | 1.0    | 1.0           | 0.030            |
//! | long, Phi-4-mini           | 3071      | 1.4e-6  | 2.7e-4  | -1.7e-5   | 0.9997 | 0.9991        | 0.128            |
//! | long, Phi-3.5-mini         | 3071      | 1.2e-6  | 1.2e-4  | +6.1e-5   | 1.0    | 0.9993        | 0.125            |
//! | long, Phi-4-mini, swapped  | 3071      | 4.14    | 29.7    | +4.01     | 0.445  | 0.457         | 32.1             |
//!
//! The ROCm executor against poot's CPU oracle on the first short window: largest logit difference 2.2e-4,
//! top-1 and top-5 identical (`phi_4_mini_short_window_rocm_matches_the_cpu_oracle`).
//!
//! [`GATE`] allows a few times the worst right-table value: the remaining difference is f16 weight rounding
//! against f32 accumulation and the executor's summation order (2.2e-4 in a logit), which flips an arg-max only
//! at a near-tie (one position of 3071 did) and moves a log-probability by at most 0.13 (the reference stores
//! log-probabilities in 16 bits). The swapped-tables control is five orders of magnitude outside the mean-KL
//! gate; [`SWAP_FACTOR`] requires one.
//!
//! On the short window the factor table matters little (the short factors are all 1.0, so that window tests
//! mainly the attention scale: dropping it scores mean KL 0.358 against a Q4_K_M reference); past 4096 the
//! table is the whole difference.
//!
//! Mutation: swap `short_factor` and `long_factor` where `hf_rope_flavor` reads them
//! (`crates/poot-models/src/components/rope.rs`); the long-window row then scores the swapped numbers above and
//! is red at its mean-KL term.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{Graph, ValueId};
use poot_graph_plan::{WeightFormats, bind_packed_weights};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::registry::Registry;
use poot_test_util::{Checkpoint, REQUIRE_MODELS_VARIABLE};

use poot_executor::Executor;

use super::llamacpp_bisect_tests::{oracle_inputs, slot_inputs};
use super::llamacpp_kl_tests::{
    Agreement, CORPUS, Gate, Reference, TeacherForced, compare_chunks, run_reference, top5_overlap,
};
use super::{BackendChoice, ModelHandle, open_executor, serving_trace_limits};

/// The short regime: well inside the 4096 original context.
const SHORT_CTX: usize = 128;

/// The long regime: past the 4096 original context, so llama.cpp and poot both take the long table. The
/// scored half starts at 3072.
const LONG_CTX: usize = 6144;

/// Windows scored in the short regime on the device (the corpus holds five).
const SHORT_WINDOWS: usize = 5;

/// Windows the CPU oracle scores: one is about half an hour on a single thread, five are hours.
const ORACLE_WINDOWS: usize = 1;

/// Windows scored in the long regime: one, since a window is a 6144-token prefill on the CPU.
const LONG_WINDOWS: usize = 1;

/// The bound on poot against an F16 llama.cpp: a few times the worst right-table measurement (see the module
/// comment). The terms hold on every position of every window, short and long, for both models. The NLL
/// delta bound is four steps of the reference file's 16-bit log-probability code (16 nats over 65535 codes,
/// 2.4e-4 each): the one-window CPU row measured -2.4e-4, one step, over 63 positions.
const GATE: Gate = Gate {
    max_mean_kl: 1e-5,
    max_max_kl: 2e-3,
    max_abs_nll_delta: 1e-3,
    min_top1: 0.995,
};

/// The lowest top-5 overlap with llama.cpp.
const MIN_TOP5_OVERLAP: f64 = 0.995;

/// The largest difference of one log-probability from llama.cpp's (16-bit coded) reference.
const MAX_ABS_LOGPROB: f64 = 0.5;

/// How many times the gate's mean KL the swapped tables must score.
const SWAP_FACTOR: f64 = 10.0;

/// The executor noise floor: ROCm against the CPU oracle on the first short window measured 2.2e-4 in the
/// largest logit difference, top-1 1.0 and top-5 1.0; four times the difference and a tie-break's worth of
/// agreement.
const DEVICE_MAX_ABS_LOGIT: f64 = 1e-3;
const DEVICE_MIN_TOP1: f64 = 0.99;
const DEVICE_MIN_TOP5: f64 = 0.99;

const PHI_4_MINI: (Checkpoint, Checkpoint) = (
    poot_test_util::checkpoint!("phi-4-mini"),
    poot_test_util::checkpoint!("phi-4-mini-gguf/Phi-4-mini-instruct-F16.gguf"),
);

const PHI_3_5_MINI: (Checkpoint, Checkpoint) = (
    poot_test_util::checkpoint!("phi-3.5-mini"),
    poot_test_util::checkpoint!("phi-3.5-mini-gguf/Phi-3.5-mini-instruct-F16.gguf"),
);

/// poot's CPU oracle over teacher-forced windows: one traced, packed-bound prefill with logits at every row,
/// its weights built once, only the step slots replaced per window.
struct OracleForced {
    graph: Graph,
    inputs: HashMap<ValueId, Value>,
    n_ctx: usize,
}

impl OracleForced {
    fn new(handle: &ModelHandle, first_window: &[i32]) -> Self {
        let n_ctx = first_window.len();
        let model = handle.model();
        let n = NonZeroUsize::new(n_ctx).unwrap();
        let shape = StepShape::new(NonZeroUsize::new(1).unwrap(), n, n).with_logits(LogitRows::All);
        let graph = model
            .trace(&StepSpec::step(shape), serving_trace_limits(n))
            .expect("trace the prefill");
        let graph = bind_packed_weights(&graph, &WeightFormats::from_weight_map(model.weights()))
            .expect("bind the packed weights");
        let inputs = oracle_inputs(handle, &graph, first_window);
        Self {
            graph,
            inputs,
            n_ctx,
        }
    }

    /// The logits of every position of `window`, `[n_ctx, n_vocab]` row-major.
    fn logits(&mut self, window: &[i32]) -> Vec<f32> {
        assert_eq!(window.len(), self.n_ctx);
        self.inputs.extend(slot_inputs(&self.graph, window));
        eval(&self.graph, &self.inputs, EvalOptions::default())
            .expect("evaluate the prefill")
            .output
            .into_host()
            .expect("dense logits")
            .as_f32()
            .expect("f32 logits")
            .to_vec()
    }
}

/// A copy of the HF checkpoint at `source` (its files linked, not copied) whose `config.json` is `edit` of the
/// original: the controls that prove the comparison can see a table or the attention scale.
fn with_edited_config(
    source: &Path,
    edit: impl FnOnce(&mut serde_json::Value),
) -> poot_test_util::UniqueTempPath {
    let dir = poot_test_util::unique_temp_path("edited-checkpoint");
    std::fs::create_dir_all(&*dir).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "config.json" {
            std::os::unix::fs::symlink(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
    edit(&mut config);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    dir
}

/// The `longrope` short and long tables exchanged.
fn swap_factor_tables(config: &mut serde_json::Value) {
    let scaling = config["rope_scaling"]
        .as_object_mut()
        .expect("a rope_scaling block");
    let short = scaling.remove("short_factor").expect("short_factor");
    let long = scaling.remove("long_factor").expect("long_factor");
    scaling.insert("short_factor".into(), long);
    scaling.insert("long_factor".into(), short);
}

/// The maximum context cut to the original one, so the attention scale `sqrt(1 + ln(max / original) /
/// ln(original))` is 1 (and the short regime, whose factors are all 1.0, is plain RoPE).
fn drop_attention_scale(config: &mut serde_json::Value) {
    config["max_position_embeddings"] = config["original_max_position_embeddings"].clone();
}

/// Whether a missing tool, model or device fails the row (`POOT_REQUIRE_MODELS=1`, the checkpoint lanes)
/// instead of skipping it.
fn required() -> bool {
    std::env::var(REQUIRE_MODELS_VARIABLE).as_deref() == Ok("1")
}

/// Which engine computes poot's logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    /// The CPU oracle: f32 activations, the strict reference, a single thread.
    Oracle,
    /// The ROCm executor on the local GPU. The caller holds `/tmp/poot-gpu.lock`.
    Rocm,
}

/// An engine ready to run: the oracle needs nothing, ROCm a device that has opened.
enum Opened {
    Oracle,
    Rocm(Box<dyn Executor>),
}

/// Open `engine` before anything slow runs (the HF load, the llama.cpp reference). `None` after saying why
/// when the ROCm device is absent: a failure when [`required`], a skip otherwise.
fn open(engine: Engine) -> Option<Opened> {
    match engine {
        Engine::Oracle => Some(Opened::Oracle),
        Engine::Rocm => match open_executor(BackendChoice::Rocm) {
            Ok(exec) => Some(Opened::Rocm(exec)),
            Err(error) => {
                assert!(
                    !required(),
                    "no ROCm device ({error}) but {REQUIRE_MODELS_VARIABLE}=1: the checkpoint lane fails on a \
                     missing device"
                );
                eprintln!("SKIP: ROCm device unavailable: {error}");
                None
            }
        },
    }
}

/// A window's logits, `[n_ctx, n_vocab]` row-major.
type WindowLogits = Box<dyn FnMut(&[i32]) -> Vec<f32>>;

/// poot's logits for whole windows on an opened engine.
fn forward(opened: Opened, handle: &ModelHandle, first_window: &[i32]) -> WindowLogits {
    match opened {
        Opened::Oracle => {
            let mut oracle = OracleForced::new(handle, first_window);
            Box::new(move |window| oracle.logits(window))
        }
        Opened::Rocm(exec) => {
            let mut forced = TeacherForced::new(handle, exec, first_window.len());
            Box::new(move |window| forced.logits(window))
        }
    }
}

fn load(dir: &Path) -> ModelHandle {
    ModelHandle::load(dir, &Registry::builtin().unwrap(), &Default::default())
        .expect("load the HF checkpoint")
}

/// llama.cpp's reference over `text` on `gguf`, parsed from `bytes` the caller keeps alive; `None` after saying
/// why when `llama-perplexity` is not on `PATH` (a failure when [`required`]).
fn llama_reference(gguf: &Path, text: &str, n_ctx: usize) -> Option<Vec<u8>> {
    let scratch = poot_test_util::unique_temp_path("reference.bin");
    let corpus = scratch.with_file_name("corpus.txt");
    std::fs::write(&corpus, text).unwrap();
    if run_reference(gguf, &corpus, n_ctx, &scratch).is_none() {
        assert!(
            !required(),
            "llama-perplexity is not on PATH but {REQUIRE_MODELS_VARIABLE}=1: the checkpoint lane fails on a \
             missing tool"
        );
        return None;
    }
    Some(std::fs::read(&scratch).unwrap())
}

/// poot's agreement with llama.cpp over `windows` windows of `n_ctx` tokens of `text`, scoring the
/// checkpoint at `hf`.
fn measure(
    models: (Checkpoint, Checkpoint),
    engine: Engine,
    n_ctx: usize,
    windows: usize,
    text: impl FnOnce(&Path) -> String,
) -> Option<Agreement> {
    let hf = poot_test_util::model_path(models.0)?;
    measure_dir(models, &hf, engine, n_ctx, windows, text)
}

/// [`measure`] with the HF directory poot loads named: the control scores a modified copy against the
/// reference of the real GGUF.
fn measure_dir(
    (hf, gguf): (Checkpoint, Checkpoint),
    poot_dir: &Path,
    engine: Engine,
    n_ctx: usize,
    windows: usize,
    text: impl FnOnce(&Path) -> String,
) -> Option<Agreement> {
    let hf = poot_test_util::model_path(hf)?;
    let gguf = poot_test_util::model_path(gguf)?;
    let opened = open(engine)?;
    let bytes = llama_reference(&gguf, &text(&hf), n_ctx)?;
    let reference = Reference::parse(&bytes).expect("parse the reference");
    assert_eq!(reference.n_ctx, n_ctx);
    assert!(
        reference.chunks >= windows,
        "llama.cpp scored {} windows of {n_ctx} tokens, fewer than the {windows} the row compares",
        reference.chunks
    );
    let handle = load(poot_dir);
    assert_eq!(
        handle.config().vocab,
        reference.n_vocab,
        "the two engines share one vocabulary"
    );
    let mut logits = forward(opened, &handle, reference.window_tokens(0));
    let agreement = compare_chunks(&reference, windows, |window| logits(window));
    assert_eq!(
        agreement.positions,
        windows * (n_ctx - 1 - n_ctx / 2),
        "every scored position is compared"
    );
    Some(agreement)
}

/// How far two engines' logits are apart over the scored positions of a window.
#[derive(Debug)]
struct LogitDiff {
    /// Largest difference of one logit.
    max_abs: f64,
    /// Fraction of positions where both engines rank the same token first.
    top1: f64,
    /// Mean fraction of one engine's five most likely tokens that the other's five hold.
    top5_overlap: f64,
}

fn logit_diff(a: &[f32], b: &[f32], n_ctx: usize, n_vocab: usize) -> LogitDiff {
    assert_eq!(a.len(), n_ctx * n_vocab);
    assert_eq!(b.len(), a.len());
    let (mut max_abs, mut agree, mut overlap, mut positions) = (0.0f64, 0usize, 0.0, 0usize);
    for position in n_ctx / 2..n_ctx - 1 {
        let row = |x: &[f32]| -> Vec<f64> {
            x[position * n_vocab..][..n_vocab]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let (ra, rb) = (row(a), row(b));
        max_abs = ra
            .iter()
            .zip(&rb)
            .fold(max_abs, |m, (x, y)| m.max((x - y).abs()));
        agree += usize::from(argmax_of(&ra) == argmax_of(&rb));
        overlap += top5_overlap(&ra, &rb);
        positions += 1;
    }
    LogitDiff {
        max_abs,
        top1: agree as f64 / positions as f64,
        top5_overlap: overlap / positions as f64,
    }
}

fn argmax_of(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn check_gate(model: &str, agreement: &Agreement) {
    eprintln!("{model}: llama.cpp agreement {agreement:?}");
    if let Err(failure) = GATE.check(agreement) {
        panic!("{model}: {failure} ({agreement:?})");
    }
    assert!(
        agreement.top5_overlap >= MIN_TOP5_OVERLAP && agreement.max_abs_logprob <= MAX_ABS_LOGPROB,
        "{model}: top-5 overlap {} (gate {MIN_TOP5_OVERLAP}) or largest log-probability difference {} (gate \
         {MAX_ABS_LOGPROB}) is outside the gate ({agreement:?})",
        agreement.top5_overlap,
        agreement.max_abs_logprob
    );
}

/// The long-window text: the checkpoint's own `modeling_phi3.py` and README, enough for two windows of
/// [`LONG_CTX`] tokens (llama.cpp scores only whole windows and needs at least two).
fn long_text(hf: &Path) -> String {
    let text = std::fs::read_to_string(hf.join("modeling_phi3.py")).expect("modeling_phi3.py");
    let readme = std::fs::read_to_string(hf.join("README.md")).unwrap_or_default();
    format!("{text}\n{readme}")
}

fn short_row(models: (Checkpoint, Checkpoint), engine: Engine, windows: usize, name: &str) {
    let Some(run) = measure(models, engine, SHORT_CTX, windows, |_| CORPUS.to_string()) else {
        return;
    };
    check_gate(name, &run);
}

#[test]
#[ignore = "loads Phi-4-mini (bf16, 3.8B) on the CPU oracle, about half an hour; the checkpoint lane runs it by name"]
fn phi_4_mini_short_window_follows_llamacpp() {
    short_row(
        PHI_4_MINI,
        Engine::Oracle,
        ORACLE_WINDOWS,
        "Phi-4-mini, short window, CPU",
    );
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads Phi-4-mini (bf16, 3.8B); the checkpoint-rocm lane runs it by name"]
fn phi_4_mini_short_window_follows_llamacpp_rocm() {
    short_row(
        PHI_4_MINI,
        Engine::Rocm,
        SHORT_WINDOWS,
        "Phi-4-mini, short window, ROCm",
    );
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads Phi-3.5-mini (bf16, 3.8B); the checkpoint-rocm lane runs it by name"]
fn phi_3_5_mini_short_window_follows_llamacpp_rocm() {
    short_row(
        PHI_3_5_MINI,
        Engine::Rocm,
        SHORT_WINDOWS,
        "Phi-3.5-mini, short window, ROCm",
    );
}

/// The executor's noise floor: the ROCm logits of the first short window against the CPU oracle's on the
/// same tokens. Both run f32 activations; the device differs in accumulation order and in how it reads the
/// bf16 weights.
#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; one CPU-oracle window (~30 min) plus the device; the checkpoint-rocm lane runs it by name"]
fn phi_4_mini_short_window_rocm_matches_the_cpu_oracle() {
    let (Some(hf), Some(gguf)) = (
        poot_test_util::model_path(PHI_4_MINI.0),
        poot_test_util::model_path(PHI_4_MINI.1),
    ) else {
        return;
    };
    let Some(device) = open(Engine::Rocm) else {
        return;
    };
    let Some(bytes) = llama_reference(&gguf, CORPUS, SHORT_CTX) else {
        return;
    };
    let reference = Reference::parse(&bytes).expect("parse the reference");
    let window = reference.window_tokens(0);
    let handle = load(&hf);
    let mut on_device = forward(device, &handle, window);
    let on_device = on_device(window);
    let on_cpu = forward(Opened::Oracle, &handle, window)(window);
    let diff = logit_diff(&on_device, &on_cpu, SHORT_CTX, reference.n_vocab);
    eprintln!("Phi-4-mini, short window, ROCm against the CPU oracle: {diff:?}");
    assert!(
        diff.max_abs <= DEVICE_MAX_ABS_LOGIT
            && diff.top1 >= DEVICE_MIN_TOP1
            && diff.top5_overlap >= DEVICE_MIN_TOP5,
        "the executor is further from the oracle than its noise floor: {diff:?}"
    );
}

/// The control for the attention scale: with the maximum context cut to the original one the scale is 1 and
/// the short regime is plain RoPE, which falls outside the gate on the short window (against a Q4_K_M
/// reference the same window scored mean KL 0.358).
#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads Phi-4-mini (bf16, 3.8B); the checkpoint-rocm lane runs it by name"]
fn phi_4_mini_short_window_without_the_attention_scale_leaves_the_gate_rocm() {
    let Some(hf) = poot_test_util::model_path(PHI_4_MINI.0) else {
        return;
    };
    let unscaled = with_edited_config(&hf, drop_attention_scale);
    let Some(run) = measure_dir(
        PHI_4_MINI,
        &unscaled,
        Engine::Rocm,
        SHORT_CTX,
        SHORT_WINDOWS,
        |_| CORPUS.to_string(),
    ) else {
        return;
    };
    eprintln!("Phi-4-mini, short window, ROCm, no attention scale: {run:?}");
    assert!(
        GATE.check(&run).is_err() && run.mean_kl > SWAP_FACTOR * GATE.max_mean_kl,
        "without the attention scale the run must score at least {SWAP_FACTOR} times the gate's mean KL: {run:?}"
    );
}

/// Past the original context both engines take the long table: poot on ROCm follows llama.cpp within the
/// gate over a 6144-token window whose scored half starts at 3072.
#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; a 6144-token prefill on Phi-4-mini; the checkpoint-rocm lane runs it by name"]
fn phi_4_mini_long_window_follows_llamacpp_rocm() {
    let Some(run) = measure(PHI_4_MINI, Engine::Rocm, LONG_CTX, LONG_WINDOWS, long_text) else {
        return;
    };
    check_gate("Phi-4-mini, long window, ROCm", &run);
}

/// The control: the same window with the short and long tables exchanged falls outside the gate, so the
/// agreement above is the checkpoint's long table, not noise. A separate row because the ROCm runtime opens one
/// executor per process.
#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; a 6144-token prefill on Phi-4-mini; the checkpoint-rocm lane runs it by name"]
fn phi_4_mini_long_window_with_swapped_tables_leaves_the_gate_rocm() {
    let Some(hf) = poot_test_util::model_path(PHI_4_MINI.0) else {
        return;
    };
    let swapped = with_edited_config(&hf, swap_factor_tables);
    let Some(run) = measure_dir(
        PHI_4_MINI,
        &swapped,
        Engine::Rocm,
        LONG_CTX,
        LONG_WINDOWS,
        long_text,
    ) else {
        return;
    };
    eprintln!("Phi-4-mini, long window, ROCm, swapped tables: {run:?}");
    assert!(
        GATE.check(&run).is_err() && run.mean_kl > SWAP_FACTOR * GATE.max_mean_kl,
        "the swapped tables must score at least {SWAP_FACTOR} times the gate's mean KL: {run:?}"
    );
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; a 6144-token prefill on Phi-3.5-mini; the checkpoint-rocm lane runs it by name"]
fn phi_3_5_mini_long_window_follows_llamacpp_rocm() {
    let Some(run) = measure(
        PHI_3_5_MINI,
        Engine::Rocm,
        LONG_CTX,
        LONG_WINDOWS,
        long_text,
    ) else {
        return;
    };
    check_gate("Phi-3.5-mini, long window, ROCm", &run);
}
