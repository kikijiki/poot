//! Driver rows on a real device (Card 734): the qwen2 fixture the registry ships, through
//! `Registry::builtin()` -> `ModelHandle` -> `Driver::generate` on the wgpu and ROCm executors. Each row
//! skips when its device does not open (`POOT_REQUIRE_WGPU=1` / `POOT_REQUIRE_ROCM=1` turn that into a
//! failure); the `_ptx` rows run on a pod (`POOT_REQUIRE_PTX=1`). Device rows run serially under
//! `/tmp/poot-gpu.lock`.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_graph_plan::{CompileOptions, Submission};
use poot_models::registry::{RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

use crate::GenerationControl;
use crate::core::sampler::Sampler;
use crate::driver::error::{DriverError, InvalidRequest};
use crate::driver::step::PickPolicy;
use crate::driver::{
    AdapterRef, Admission, BackendChoice, Driver, DriverOptions, GenerateRequest, Head, Layout,
    ModelHandle, PoolShape, PrefixOutcome, PreparedSetLimits, Release, RowWork, SeqId, SeqRequest,
    ServingShapes, StepRow, Warm, open_executor,
};
use crate::text::tokenize::{ChatTemplate, TextCodec};

pub(super) const VOCAB: usize = 48;
/// The fixture's declared end-of-sequence id, and one outside its vocabulary for rows that must not
/// end early on a sampled token.
pub(super) const FIXTURE_EOS: u32 = 47;
pub(super) const NO_EOS: u32 = 999;

pub(super) fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// The fixture's text services: 48 single-character words (`a`..`z`, `0`..`9`, then `p0`..`p11`), so
/// a regex over letters masks exactly the tokens it names.
pub(super) fn fixture_text() -> TextCodec {
    text_of(VOCAB)
}

/// [`fixture_text`] over `vocab` words.
fn text_of(vocab: usize) -> TextCodec {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..vocab - words.len()).map(|i| format!("p{i}")));
    assert_eq!(words.len(), vocab);
    let vocab: HashMap<String, u32> = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("a".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    TextCodec::new(
        tokenizer,
        None,
        None,
        FIXTURE_EOS,
        poot_models::chat::ChatFormat::ChatML,
        ChatTemplate::default(),
    )
}

/// The fixture with its LM head zeroed: every logit is exactly 0, so greedy ties across the whole
/// vocabulary (and must take the first index) and a temperature draw is uniform over it.
fn zero_head(store: &WeightStore) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (key, entry) in store.iter() {
        let entry = match (key.as_str(), entry) {
            ("lm_head.weight", WeightEntry::Dense(dense)) => {
                let zeros = vec![0u8; dense.bytes().len()];
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), zeros.into())
                        .unwrap(),
                )
            }
            _ => entry.clone(),
        };
        builder.insert(key.clone(), entry).unwrap();
    }
    builder.build()
}

#[derive(Clone, Copy)]
enum Weights {
    Random,
    ZeroHead,
}

fn handle(weights: Weights, eos: u32) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let fixture = (registry.entries()[0].fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(eos);
    let store = match weights {
        Weights::Random => fixture.store.clone(),
        Weights::ZeroHead => zero_head(&fixture.store),
    };
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    Arc::new(ModelHandle::from_checkpoint(&raw, store, &registry, |_| Ok(fixture_text())).unwrap())
}

pub(super) fn options(chunk: usize, capacity: usize) -> DriverOptions {
    let compile = CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    DriverOptions {
        prefill: compile,
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        max_trace_tokens: nz(64),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: crate::driver::program_retention,
        draft: crate::driver::DraftPolicy::Off,
    }
}

pub(super) fn open(choice: BackendChoice) -> Option<Box<dyn poot_executor::Executor>> {
    let backend = match choice {
        BackendChoice::Wgpu => DeviceBackend::Wgpu,
        BackendChoice::Rocm => DeviceBackend::Rocm,
        BackendChoice::Ptx => DeviceBackend::Ptx,
        BackendChoice::Vulkan => DeviceBackend::Vulkan,
        BackendChoice::Auto => unreachable!("rows name their backend"),
    };
    poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
}

fn driver(
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    chunk: usize,
    capacity: usize,
) -> Option<Driver> {
    let executor = open(choice)?;
    Some(Driver::new(Arc::clone(handle), executor, options(chunk, capacity)).unwrap())
}

pub(super) fn generate(
    driver: &mut Driver,
    prompt: &[u32],
    max_new: usize,
    sampler: Sampler,
) -> Vec<u32> {
    driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new,
                sampler,
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap()
        .tokens
}

/// `max_new` greedy tokens and the log-probability the head gave each chosen token: a host-path
/// request, so the head's output row is read back, not just its argmax.
fn generate_traced(driver: &mut Driver, prompt: &[u32], max_new: usize) -> (Vec<u32>, Vec<f32>) {
    let generated = driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new,
                sampler: Sampler::greedy().with_logprobs(0),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap();
    let logprobs = generated.logprobs.iter().map(|l| l.logprob).collect();
    (generated.tokens, logprobs)
}

/// A packed model and its decoded twin trace the same greedy run: the same tokens, and each chosen
/// token's log-probability within `1e-3` (the packed kernels and the dense matmul round differently,
/// so the bound is a tolerance; a placement or decode bug moves a log-probability by far more). A
/// constant token sequence cannot pass on the tokens alone, so the log-probabilities carry the claim.
fn assert_same_trace(
    label: &str,
    packed: &mut Driver,
    twin: &mut Driver,
    prompt: &[u32],
    n: usize,
) {
    let (want_tokens, want_logprobs) = generate_traced(twin, prompt, n);
    let (tokens, logprobs) = generate_traced(packed, prompt, n);
    assert_eq!(tokens, want_tokens, "{label}: the packed tokens");
    assert_eq!(logprobs.len(), want_logprobs.len(), "{label}");
    for (step, (got, want)) in logprobs.iter().zip(&want_logprobs).enumerate() {
        assert!(
            (got - want).abs() <= 1e-3,
            "{label} step {step}: packed logprob {got} vs twin {want}"
        );
    }
    assert!(
        want_logprobs.iter().any(|&l| l != want_logprobs[0]),
        "{label}: the twin's logprobs are not constant: {want_logprobs:?}"
    );
}

/// Room for the longest row: a 7-token prompt and 24 new tokens.
const CAPACITY: usize = 32;

const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];

/// Declare a row once and run it per backend; each is its own test so a lane can name it.
macro_rules! per_backend {
    ($row:ident, $wgpu:ident, $rocm:ident, $ptx:ident) => {
        #[test]
        fn $wgpu() {
            $row(BackendChoice::Wgpu);
        }

        #[test]
        fn $ptx() {
            $row(BackendChoice::Ptx);
        }

        #[cfg(feature = "rocm")]
        #[test]
        fn $rocm() {
            $row(BackendChoice::Rocm);
        }
    };
}

/// POOT-574 SC-002 (R476-001 on Gemma4's path): the Gemma4 fixture with a 64-token vocabulary and its
/// head zeroed (every logit exactly 0, so the distribution is uniform) answers a seeded temperature
/// request with the sampler's tokens, not the greedy ones. A 16-token sampled run equal to the greedy
/// run has probability 64^-16 = 2^-96, so the row also draws from a fresh seed and requires the same.
/// Mutation: ignore the sampler in the head (`rule_of` always answers `Greedy`); the draw becomes
/// the greedy run and the row fails.
fn gemma4_sampled_differs_from_greedy(choice: BackendChoice) {
    const WIDE: usize = 64;
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "gemma4")
        .unwrap();
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    config["text_config"]["vocab_size"] = serde_json::json!(WIDE);
    let mut store = WeightStore::builder();
    for (key, entry) in fixture.store.iter() {
        let entry = match (key.as_str(), entry) {
            ("model.language_model.embed_tokens.weight", WeightEntry::Dense(dense)) => {
                // The embedding's rows again for the added tokens.
                let row = dense.bytes().len() / dense.shape()[0];
                let mut bytes = dense.bytes().as_slice().to_vec();
                bytes.extend_from_slice(&dense.bytes().as_slice()[..(WIDE - VOCAB) * row]);
                let shape = vec![WIDE, dense.shape()[1]];
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), shape, bytes.into()).unwrap(),
                )
            }
            ("lm_head.weight", WeightEntry::Dense(dense)) => {
                let shape = vec![WIDE, dense.shape()[1]];
                let zeros = vec![0u8; dense.bytes().len() / VOCAB * WIDE];
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), shape, zeros.into()).unwrap(),
                )
            }
            _ => entry.clone(),
        };
        store.insert(key.clone(), entry).unwrap();
    }
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let handle = Arc::new(
        ModelHandle::from_checkpoint(&raw, store.build(), &registry, |_| Ok(text_of(WIDE)))
            .unwrap(),
    );
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let prompt = [1, 3, 17, 40, 8, 25, 11, 2];
    let greedy = generate(&mut driver, &prompt, 16, Sampler::greedy());
    let sampled = generate(&mut driver, &prompt, 16, Sampler::new(1.0, 0, 1.0, 42));
    let again = generate(&mut driver, &prompt, 16, Sampler::new(1.0, 0, 1.0, 42));
    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let fresh = generate(&mut driver, &prompt, 16, Sampler::new(1.0, 0, 1.0, entropy));
    assert_eq!(greedy, [0; 16], "a tie takes the first index on every step");
    assert_ne!(sampled, greedy, "the sampler's tokens, not greedy ones");
    assert_eq!(sampled, again, "the same seed draws the same tokens");
    assert_ne!(
        fresh, greedy,
        "a fresh seed ({entropy}) draws the sampler's tokens"
    );
}
per_backend!(
    gemma4_sampled_differs_from_greedy,
    gemma4_sampled_differs_from_greedy_wgpu,
    gemma4_sampled_differs_from_greedy_rocm,
    gemma4_sampled_differs_from_greedy_ptx
);

/// SC-001: a seeded temperature draw and a greedy request on a uniform fixture (every logit 0) give
/// different tokens: the draw reaches the sampler head. With 48 equal logits a sampled run equal to the
/// greedy one has probability 48^-24 < 2^-96. Mutation: ignore the sampler in the head (`rule_of`
/// always answers `Greedy`); the draw becomes the greedy run and the row fails.
fn sampled_differs_from_greedy(choice: BackendChoice) {
    let handle = handle(Weights::ZeroHead, NO_EOS);
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let greedy = generate(&mut driver, &PROMPT, 24, Sampler::greedy());
    let sampled = generate(&mut driver, &PROMPT, 24, Sampler::new(1.0, 0, 1.0, 42));
    let again = generate(&mut driver, &PROMPT, 24, Sampler::new(1.0, 0, 1.0, 42));
    assert_eq!(greedy, [0; 24], "a tie takes the first index on every step");
    assert_ne!(sampled, greedy, "the sampler's tokens, not greedy ones");
    assert_eq!(sampled, again, "the same seed draws the same tokens");
}
per_backend!(
    sampled_differs_from_greedy,
    sampled_differs_from_greedy_wgpu,
    sampled_differs_from_greedy_rocm,
    sampled_differs_from_greedy_ptx
);

/// The regex naming the 16 tokens `a`..`p`, 25 times over: a mask the sampling suffix cannot express,
/// that no 24-token run completes (so EOS is never allowed).
pub(super) const SIXTEEN: &str = "[a-p]{25}";

/// SC-001 and SC-006: a guided request with a DFA mask over 16 tokens falls back to the host sampling
/// path and gets masked tokens; a seeded draw over the masked uniform logits differs from the greedy
/// guided run (probability 16^-24 = 2^-96 of equality) and every token is allowed by the mask.
/// Mutation: ignore the sampler in the host path (take the argmax of the masked row); the draw equals
/// the greedy guided run.
fn guided_requests_sample_inside_the_mask(choice: BackendChoice) {
    let handle = handle(Weights::ZeroHead, NO_EOS);
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let guided = |sampler: Sampler| {
        let constraint = handle.text().build_regex_constraint(SIXTEEN).unwrap();
        sampler.with_constraint(constraint)
    };
    let greedy = generate(&mut driver, &PROMPT, 24, guided(Sampler::greedy()));
    let sampled = generate(
        &mut driver,
        &PROMPT,
        24,
        guided(Sampler::new(1.0, 0, 1.0, 42)),
    );
    assert_eq!(greedy, [0; 24]);
    assert!(sampled.iter().all(|&t| t < 16), "{sampled:?}");
    assert_ne!(sampled, greedy);
    assert_eq!(
        driver.stats().host_picks,
        48,
        "every guided pick took the host path"
    );
    let before = driver.stats().host_picks;
    generate(&mut driver, &PROMPT, 4, Sampler::greedy());
    assert_eq!(
        driver.stats().host_picks,
        before,
        "an ordinary request never takes the host path"
    );
}
per_backend!(
    guided_requests_sample_inside_the_mask,
    guided_requests_sample_inside_the_mask_wgpu,
    guided_requests_sample_inside_the_mask_rocm,
    guided_requests_sample_inside_the_mask_ptx
);

/// SC-004 (c): the fixture's greedy tokens equal the literals recorded when this card landed, for the
/// whole prompt in one prefill and in chunks of 5 (internal regression literals Card 737 keeps), and a
/// host-path greedy request (logprobs force it) agrees with the device suffix. Mutation: flip the
/// greedy tie-break to the last index; the host leg of the tie row below (and this row's host leg
/// when two logits tie) takes a different token.
const FIXTURE_GREEDY: [u32; 24] = [
    33, 37, 41, 3, 41, 46, 41, 3, 8, 37, 41, 32, 41, 32, 44, 3, 8, 37, 3, 41, 41, 41, 41, 17,
];

fn fixture_tokens_equal_the_recorded_literals(choice: BackendChoice) {
    let handle = handle(Weights::Random, NO_EOS);
    // Both drivers open before either runs: the ROCm runtime does not survive opening a second
    // executor after dropping the first in one process.
    let Some(mut whole) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let Some(mut chunked) = driver(choice, &handle, 5, CAPACITY) else {
        return;
    };
    // Neither driver drops before both ran: dropping one PTX executor invalidates the other's context.
    for (chunk, driver) in [(8, &mut whole), (5, &mut chunked)] {
        let device = generate(driver, &PROMPT, 24, Sampler::greedy());
        let host = generate(driver, &PROMPT, 24, Sampler::greedy().with_logprobs(0));
        assert_eq!(device, FIXTURE_GREEDY, "chunk {chunk}");
        assert_eq!(
            host, device,
            "chunk {chunk}: the host path agrees with the suffix"
        );
    }
}
per_backend!(
    fixture_tokens_equal_the_recorded_literals,
    fixture_tokens_equal_the_recorded_literals_wgpu,
    fixture_tokens_equal_the_recorded_literals_rocm,
    fixture_tokens_equal_the_recorded_literals_ptx
);

/// SC-005 (device side): the capacity only sizes buffers. A request at 8x the live length generates
/// the tokens of the exact-capacity run. Mutation: fill `Slot::Pos` from the capacity; the tokens
/// diverge.
fn capacity_does_not_change_the_tokens(choice: BackendChoice) {
    let handle = handle(Weights::Random, NO_EOS);
    // live length 8 (4 prompt + 4 new), capacity 64: 8x.
    let prompt = &PROMPT[..4];
    let Some(mut exact) = driver(choice, &handle, 4, 8) else {
        return;
    };
    let Some(mut wide) = driver(choice, &handle, 4, 64) else {
        return;
    };
    let want = generate(&mut exact, prompt, 4, Sampler::greedy().with_logprobs(0));
    let got = generate(&mut wide, prompt, 4, Sampler::greedy().with_logprobs(0));
    assert_eq!(got, want);
}
per_backend!(
    capacity_does_not_change_the_tokens,
    capacity_does_not_change_the_tokens_wgpu,
    capacity_does_not_change_the_tokens_rocm,
    capacity_does_not_change_the_tokens_ptx
);

/// SC-008: prefill in chunks of 3 then decode equals the recorded token-by-token run, and the executor
/// holds one upload of the weights however many entries share the executable: the chunked driver's
/// prefill and decode entries (three or more) upload what a driver with a single entry does.
/// Mutation: give the prefill entries their own executable (their own state set and weights); the
/// decode steps then attend an empty cache, so the tokens diverge from the literal, and the weight
/// allocations double against the single-entry driver.
fn chunked_prefill_equals_token_by_token(choice: BackendChoice) {
    let handle = handle(Weights::Random, NO_EOS);
    // Every executor opens before any runs (see `fixture_tokens_equal_the_recorded_literals`).
    let Some(mut chunked) = driver(choice, &handle, 3, CAPACITY) else {
        return;
    };
    let Some(mut baseline) = driver(choice, &handle, 1, CAPACITY) else {
        return;
    };
    let tokens = generate(&mut chunked, &PROMPT, 12, Sampler::greedy());
    generate(&mut baseline, &PROMPT[..1], 1, Sampler::greedy());
    assert_eq!(baseline.prepared_entries(), 1);
    assert_eq!(
        chunked.prepared_entries(),
        2,
        "the 7-token prompt plans 3 + 3 + 1: prefill 3, and the one-token step the tail and decode share"
    );
    let uploads = |driver: &Driver| {
        driver
            .executor_stats()
            .memory
            .iter()
            .find(|(role, _)| *role == poot_executor::BufferRole::Weight)
            .map(|(_, snapshot)| snapshot.allocations)
            .unwrap()
    };
    assert_eq!(
        uploads(&chunked),
        uploads(&baseline),
        "one weight upload however many entries share the executable"
    );
    assert_eq!(
        tokens,
        FIXTURE_GREEDY[..12],
        "the token-by-token run's tokens"
    );
}
per_backend!(
    chunked_prefill_equals_token_by_token,
    chunked_prefill_equals_token_by_token_wgpu,
    chunked_prefill_equals_token_by_token_rocm,
    chunked_prefill_equals_token_by_token_ptx
);

/// SC-004 tie-break row: with every logit equal, greedy takes the first index on the device suffix and
/// on the host path alike. Mutation: flip the greedy tie-break to the last index; the host leg picks the
/// last token (47) and the row fails.
fn greedy_ties_take_the_first_index(choice: BackendChoice) {
    let handle = handle(Weights::ZeroHead, NO_EOS);
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let device = generate(&mut driver, &PROMPT, 8, Sampler::greedy());
    let host = generate(&mut driver, &PROMPT, 8, Sampler::greedy().with_logprobs(0));
    assert_eq!(device, [0; 8]);
    assert_eq!(host, [0; 8]);
}
per_backend!(
    greedy_ties_take_the_first_index,
    greedy_ties_take_the_first_index_wgpu,
    greedy_ties_take_the_first_index_rocm,
    greedy_ties_take_the_first_index_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-737 SC-001: every dense family's builtin fixture generates through the driver with the tokens
// the Runner generated for it before the dense families left the Runner (recorded on the base commit
// `2ef7ea834` with the Runner's `generate_kv_gpu_cached` on wgpu and ROCm, and its CPU `generate`:
// all three agreed, and the driver agreed with them). Literals, never recomputed (ADR-0101, tier 3).
// ---------------------------------------------------------------------------------------------

/// What the Runner was fed for a family's fixture: its prompt, and whether its head is tied. The
/// Runner always tied BLOOM's and MPT's head to the embedding, so their rows drop the fixture's
/// explicit `lm_head.weight` (the driver then ties it too); gemma3's tokenizer prepended BOS (1).
struct RecordedFamily {
    family: &'static str,
    prompt: &'static [u32],
    tied_head: bool,
    tokens: [u32; 24],
}

const GEMMA3_PROMPT: [u32; 8] = [1, 3, 17, 40, 8, 25, 11, 2];

const RECORDED: &[RecordedFamily] = &[
    RecordedFamily {
        family: "qwen2",
        prompt: &PROMPT,
        tied_head: false,
        tokens: FIXTURE_GREEDY,
    },
    RecordedFamily {
        family: "gemma3",
        prompt: &GEMMA3_PROMPT,
        tied_head: false,
        tokens: [
            15, 0, 41, 5, 20, 24, 41, 23, 13, 25, 15, 28, 30, 20, 40, 22, 41, 24, 24, 40, 30, 16,
            34, 34,
        ],
    },
    RecordedFamily {
        family: "granite",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            24, 17, 38, 41, 4, 17, 36, 47, 38, 41, 0, 24, 3, 4, 13, 25, 35, 17, 41, 4, 13, 34, 3,
            27,
        ],
    },
    RecordedFamily {
        family: "olmo2",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            40, 38, 27, 47, 46, 29, 14, 36, 9, 9, 9, 27, 38, 21, 21, 21, 21, 0, 44, 44, 14, 44, 14,
            14,
        ],
    },
    RecordedFamily {
        family: "bloom",
        prompt: &PROMPT,
        tied_head: true,
        tokens: [
            46, 7, 46, 29, 46, 29, 46, 29, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
            46, 46,
        ],
    },
    RecordedFamily {
        family: "mpt",
        prompt: &PROMPT,
        tied_head: true,
        tokens: [
            4, 35, 15, 38, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 15, 15,
            15, 15,
        ],
    },
    RecordedFamily {
        family: "smollm3",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            45, 37, 6, 43, 6, 0, 20, 20, 20, 20, 6, 43, 43, 13, 20, 14, 31, 20, 14, 31, 29, 0, 13,
            20,
        ],
    },
    RecordedFamily {
        family: "llama",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            46, 15, 25, 46, 32, 46, 41, 46, 8, 39, 45, 38, 47, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            32, 33,
        ],
    },
    RecordedFamily {
        family: "qwen3",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            15, 15, 15, 40, 10, 10, 10, 11, 24, 10, 10, 10, 11, 24, 6, 10, 24, 17, 17, 16, 36, 17,
            41, 8,
        ],
    },
    RecordedFamily {
        family: "phi3",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            36, 7, 7, 16, 36, 26, 32, 46, 10, 3, 46, 2, 24, 45, 31, 25, 20, 0, 25, 13, 38, 46, 17,
            32,
        ],
    },
    // Recorded with the CPU oracle (`run_step` over the fixture, a prefill and 23 decode steps) on
    // this card's base; the f64 reference test in `gemma4.rs` pins the oracle itself (tier 3).
    RecordedFamily {
        family: "gemma4",
        prompt: &GEMMA3_PROMPT,
        tied_head: false,
        tokens: [
            29, 29, 28, 29, 31, 10, 44, 14, 40, 9, 13, 47, 22, 43, 42, 11, 24, 10, 10, 29, 16, 8,
            45, 44,
        ],
    },
    RecordedFamily {
        family: "gemma4/moe",
        prompt: &GEMMA3_PROMPT,
        tied_head: false,
        tokens: [
            34, 12, 5, 47, 33, 43, 13, 45, 36, 7, 22, 32, 2, 1, 10, 43, 8, 21, 41, 24, 11, 6, 20,
            33,
        ],
    },
    // The five MoE rows below (POOT-738) were recorded with the Runner's CPU `generate` on each family's
    // fixture at the base commit. All 24 tokens agreed: the driver on wgpu and on ROCm produce exactly the
    // recorded ids, in one whole-prompt prefill and in chunks of four, so no row has a divergence index or
    // a tier-3 top-two margin to record. (A token flip on a tiny random fixture would be judged by the
    // reference's top-two logit margin, ADR-0101.) The first-token figures of the router-id-swap mutation
    // (qwen3moe 10 vs 28, Mixtral 38 vs 40, granitemoe 43 vs 40, gpt-oss 6 vs 11, olmoe 11 vs 11 with a
    // later split) are the red outcome of that mutation, not Runner disagreements.
    RecordedFamily {
        family: "qwen3moe",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            28, 28, 28, 28, 28, 38, 27, 15, 34, 15, 42, 18, 15, 16, 38, 27, 44, 21, 28, 38, 40, 15,
            38, 40,
        ],
    },
    RecordedFamily {
        family: "olmoe",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            11, 11, 11, 11, 11, 11, 11, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 27, 11, 22, 22, 27,
            11, 27,
        ],
    },
    RecordedFamily {
        family: "llama/mixtral",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            40, 44, 24, 11, 24, 11, 44, 24, 11, 46, 21, 31, 11, 24, 42, 40, 28, 14, 8, 44, 40, 38,
            24, 21,
        ],
    },
    RecordedFamily {
        family: "granite/granitemoe",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            40, 38, 31, 17, 28, 40, 5, 43, 10, 20, 38, 31, 20, 20, 38, 19, 45, 14, 10, 20, 20, 20,
            20, 20,
        ],
    },
    RecordedFamily {
        family: "gpt_oss",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            11, 35, 8, 18, 31, 41, 42, 14, 3, 19, 44, 25, 30, 12, 10, 18, 19, 3, 35, 10, 31, 37,
            47, 3,
        ],
    },
    // minimax_m2 (POOT-570) has no Runner recording: its tokens are the independent f64 reference's greedy
    // decode (`minimax_m2::tests::the_driver_literals_are_the_references_greedy_decode`, smallest top-two
    // logit margin 1.2e-4), so the device is compared with the reference, not with an earlier run.
    RecordedFamily {
        family: "minimax_m2",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            40, 38, 40, 38, 10, 21, 20, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 24, 46, 21,
            46, 21,
        ],
    },
    // The four group-B rows below (POOT-740) were recorded with the Runner's CPU `generate` on each
    // family's fixture at the base commit `64602bced`, over the HF checkpoint the fixture writes (eos
    // unreachable): MLA (deepseek2 with YaRN, group-limited softmax routing and two shared experts;
    // deepseek3 with the sigmoid router; deepseek32 with the sparse-attention indexer keeping three
    // positions) and the NemotH hybrid (Mamba2, attention, MLP).
    RecordedFamily {
        family: "deepseek2",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            11, 32, 18, 32, 18, 32, 29, 38, 32, 20, 6, 19, 20, 6, 19, 20, 6, 19, 20, 6, 19, 20, 6,
            19,
        ],
    },
    RecordedFamily {
        family: "deepseek3",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            11, 2, 11, 2, 11, 2, 11, 32, 35, 5, 15, 32, 23, 12, 2, 20, 18, 32, 27, 2, 38, 12, 2, 20,
        ],
    },
    RecordedFamily {
        family: "deepseek32",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            14, 20, 2, 14, 9, 37, 19, 35, 11, 34, 33, 14, 17, 33, 14, 20, 46, 3, 17, 0, 1, 33, 14,
            8,
        ],
    },
    RecordedFamily {
        family: "nemotron_h",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            28, 31, 1, 1, 1, 27, 2, 0, 36, 25, 22, 25, 28, 31, 1, 1, 1, 27, 32, 34, 31, 1, 25, 28,
        ],
    },
];

/// `family`'s builtin fixture as the Runner was fed it (see [`RecordedFamily`]), with no reachable
/// end-of-sequence id.
fn recorded_family_handle(recorded: &RecordedFamily) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    // `family/variant` names one of the family's option fixtures (llama's Mixtral).
    let (name, variant) = recorded
        .family
        .split_once('/')
        .unwrap_or((recorded.family, ""));
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == name)
        .unwrap_or_else(|| panic!("no builtin family {name}"));
    let fixture = match variant {
        "" => (entry.fixture)(),
        variant => {
            let (_, make) = entry
                .variants
                .iter()
                .find(|(n, _)| *n == variant)
                .unwrap_or_else(|| panic!("{name} has no variant {variant}"));
            make()
        }
    };
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let mut store = WeightStore::builder();
    for (key, weight) in fixture.store.iter() {
        if !(recorded.tied_head && key.as_str() == "lm_head.weight") {
            store.insert(key.clone(), weight.clone()).unwrap();
        }
    }
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, store.build(), &registry, |_| Ok(fixture_text()))
            .unwrap_or_else(|e| panic!("{}: {e}", recorded.family)),
    )
}

/// SC-001: `family`'s fixture generates its recorded 24 greedy tokens through the driver, with the
/// whole prompt in one prefill and in chunks of four. Mutation: register gemma3 with granite's
/// `build` (the gemma fixture traced by the granite tracer); the gemma3 row goes red.
fn recorded_family_tokens(family: &str, choice: BackendChoice) {
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == family)
        .unwrap_or_else(|| panic!("no recorded literals for {family}"));
    let handle = recorded_family_handle(recorded);
    // Both drivers open before either runs (see `fixture_tokens_equal_the_recorded_literals`).
    let Some(mut whole) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let Some(mut chunked) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    for (chunk, driver) in [(8, &mut whole), (4, &mut chunked)] {
        let got = generate(driver, recorded.prompt, 24, Sampler::greedy());
        assert_eq!(
            got, recorded.tokens,
            "{family} on {choice:?}, chunk {chunk}"
        );
    }
}

/// One row per family and backend, each its own test (and process: the ROCm runtime does not survive
/// reopening an executor in one process).
macro_rules! recorded_family_rows {
    ($($family:literal => $wgpu:ident, $rocm:ident, $ptx:ident;)*) => {
        $(
            #[test]
            fn $wgpu() {
                recorded_family_tokens($family, BackendChoice::Wgpu);
            }

            #[cfg(feature = "rocm")]
            #[test]
            fn $rocm() {
                recorded_family_tokens($family, BackendChoice::Rocm);
            }

            #[test]
            fn $ptx() {
                recorded_family_tokens($family, BackendChoice::Ptx);
            }
        )*
    };
}

recorded_family_rows! {
    "qwen2" => recorded_qwen2_tokens_wgpu, recorded_qwen2_tokens_rocm, recorded_qwen2_tokens_ptx;
    "gemma3" => recorded_gemma3_tokens_wgpu, recorded_gemma3_tokens_rocm, recorded_gemma3_tokens_ptx;
    "gemma4" => recorded_gemma4_tokens_wgpu, recorded_gemma4_tokens_rocm, recorded_gemma4_tokens_ptx;
    "gemma4/moe" => recorded_gemma4_moe_tokens_wgpu, recorded_gemma4_moe_tokens_rocm, recorded_gemma4_moe_tokens_ptx;
    "granite" => recorded_granite_tokens_wgpu, recorded_granite_tokens_rocm, recorded_granite_tokens_ptx;
    "olmo2" => recorded_olmo2_tokens_wgpu, recorded_olmo2_tokens_rocm, recorded_olmo2_tokens_ptx;
    "bloom" => recorded_bloom_tokens_wgpu, recorded_bloom_tokens_rocm, recorded_bloom_tokens_ptx;
    "mpt" => recorded_mpt_tokens_wgpu, recorded_mpt_tokens_rocm, recorded_mpt_tokens_ptx;
    "smollm3" => recorded_smollm3_tokens_wgpu, recorded_smollm3_tokens_rocm, recorded_smollm3_tokens_ptx;
    "llama" => recorded_llama_tokens_wgpu, recorded_llama_tokens_rocm, recorded_llama_tokens_ptx;
    "qwen3" => recorded_qwen3_tokens_wgpu, recorded_qwen3_tokens_rocm, recorded_qwen3_tokens_ptx;
    "phi3" => recorded_phi3_tokens_wgpu, recorded_phi3_tokens_rocm, recorded_phi3_tokens_ptx;
    "qwen3moe" => recorded_qwen3moe_tokens_wgpu, recorded_qwen3moe_tokens_rocm, recorded_qwen3moe_tokens_ptx;
    "olmoe" => recorded_olmoe_tokens_wgpu, recorded_olmoe_tokens_rocm, recorded_olmoe_tokens_ptx;
    "llama/mixtral" => recorded_mixtral_tokens_wgpu, recorded_mixtral_tokens_rocm, recorded_mixtral_tokens_ptx;
    "granite/granitemoe" => recorded_granitemoe_tokens_wgpu, recorded_granitemoe_tokens_rocm, recorded_granitemoe_tokens_ptx;
    "gpt_oss" => recorded_gpt_oss_tokens_wgpu, recorded_gpt_oss_tokens_rocm, recorded_gpt_oss_tokens_ptx;
    "minimax_m2" => recorded_minimax_m2_tokens_wgpu, recorded_minimax_m2_tokens_rocm, recorded_minimax_m2_tokens_ptx;
    "deepseek2" => recorded_deepseek2_tokens_wgpu, recorded_deepseek2_tokens_rocm, recorded_deepseek2_tokens_ptx;
    "deepseek3" => recorded_deepseek3_tokens_wgpu, recorded_deepseek3_tokens_rocm, recorded_deepseek3_tokens_ptx;
    "deepseek32" => recorded_deepseek32_tokens_wgpu, recorded_deepseek32_tokens_rocm, recorded_deepseek32_tokens_ptx;
    "nemotron_h" => recorded_nemotron_h_tokens_wgpu, recorded_nemotron_h_tokens_rocm, recorded_nemotron_h_tokens_ptx;
}

// ---------------------------------------------------------------------------------------------
// POOT-737 SC-002 and SC-004: a step shape a dense family's trace refuses is a typed refusal before
// any work starts, the same variant for every family and backend.
// ---------------------------------------------------------------------------------------------

/// A driver whose prefill chunk (16 tokens) exceeds its KV capacity (8 positions): the full-chunk
/// prefill piece is a step of more new tokens than its cache holds, which every dense family's trace
/// refuses (`ShapeReason::TokensAboveCapacity`).
fn oversized_chunk_driver(executor: Box<dyn poot_executor::Executor>, family: &str) -> Driver {
    let handle = family_handle(family, NO_EOS);
    Driver::new(handle, executor, options(16, 8)).unwrap()
}

/// Admit every prompt piece of the contiguous layout: the full chunk first.
fn prepare_admitted(driver: &mut Driver) -> Result<(), DriverError> {
    driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .map(|_| ())
}

/// The refusal `family`'s trace gives for the oversized chunk, as the driver returns it.
fn assert_refused_by_the_trace(error: &DriverError, family: &str) {
    use crate::driver::error::Unsupported;
    use poot_models::model::{ShapeReason, TraceError};
    match error {
        DriverError::Unsupported(Unsupported::Trace(TraceError::ShapeUnsupported {
            family: refused,
            shape,
            reason: ShapeReason::TokensAboveCapacity,
        })) => {
            assert_eq!(refused.as_str(), family);
            assert_eq!((shape.tokens.get(), shape.capacity.get()), (16, 8));
        }
        other => panic!("{family}: expected the trace's typed refusal, got {other:?}"),
    }
}

/// SC-002: granite's trace refuses a step of 16 new tokens into an 8-position cache. The driver
/// returns `DriverError::Unsupported(Trace)` from `prepare`, before it compiled, added or dispatched
/// anything: the real executor recorded and replayed nothing (R476-013: no failure after work
/// started). Mutation: drop `check_step`'s `TokensAboveCapacity` refusal, so the shape fails at run
/// time instead; the row goes red.
fn a_refused_shape_is_typed_before_any_dispatch(choice: BackendChoice) {
    let Some(executor) = open(choice) else {
        return;
    };
    let mut driver = oversized_chunk_driver(executor, "granite");
    let error = prepare_admitted(&mut driver).unwrap_err();
    assert_refused_by_the_trace(&error, "granite");
    assert_eq!(driver.stats().compiles, 0, "nothing compiled");
    assert_eq!(driver.stats().entries_added, 0, "nothing added");
    let executor = driver.executor_stats();
    assert_eq!(
        (executor.recordings, executor.replays),
        (0, 0),
        "the executor dispatched nothing"
    );
}
per_backend!(
    a_refused_shape_is_typed_before_any_dispatch,
    a_refused_shape_is_typed_before_any_dispatch_wgpu,
    a_refused_shape_is_typed_before_any_dispatch_rocm,
    a_refused_shape_is_typed_before_any_dispatch_ptx
);

/// SC-004: the same unsupported request (a prefill step larger than the cache) crosses the driver as
/// the same `DriverError` variant for two families, granite and bloom, on each backend. Mutation:
/// return bloom's refusal as a formatted string error (`DriverError::Text` of a message); bloom's
/// match fails and the row goes red.
fn an_unsupported_request_is_the_same_variant_for_two_families(choice: BackendChoice) {
    // Both executors open before either driver runs (the ROCm runtime does not reopen one).
    let Some(first) = open(choice) else {
        return;
    };
    let Some(second) = open(choice) else {
        return;
    };
    let mut drivers = [
        ("granite", oversized_chunk_driver(first, "granite")),
        ("bloom", oversized_chunk_driver(second, "bloom")),
    ];
    for (family, driver) in &mut drivers {
        let error = prepare_admitted(driver).unwrap_err();
        assert_refused_by_the_trace(&error, family);
    }
}
per_backend!(
    an_unsupported_request_is_the_same_variant_for_two_families,
    an_unsupported_request_is_the_same_variant_for_two_families_wgpu,
    an_unsupported_request_is_the_same_variant_for_two_families_rocm,
    an_unsupported_request_is_the_same_variant_for_two_families_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-737 SC-003: packed checkpoints of non-qwen2 dense families load through the registry as stored.
// ---------------------------------------------------------------------------------------------

/// One tensor of a written safetensors archive.
struct ArchiveTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// An HF checkpoint directory: `config`, one safetensors archive of `tensors`, the fixture tokenizer.
fn write_hf_checkpoint(
    label: &str,
    config: &serde_json::Value,
    tensors: &[ArchiveTensor],
) -> poot_test_util::UniqueTempPath {
    let dir = poot_test_util::unique_temp_path(format!("poot_card737_{label}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.bytes);
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut archive = (header.len() as u64).to_le_bytes().to_vec();
    archive.extend_from_slice(&header);
    archive.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), archive).unwrap();
    fixture_text()
        .tokenizer
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    dir
}

/// `fixture` written as an HF checkpoint directory, every tensor as stored.
fn fixture_dir(
    label: &str,
    fixture: &poot_models::registry::Fixture,
) -> poot_test_util::UniqueTempPath {
    let tensors: Vec<ArchiveTensor> = fixture
        .store
        .iter()
        .map(|(key, weight)| {
            let WeightEntry::Dense(dense) = weight else {
                panic!("the fixture is dense")
            };
            ArchiveTensor {
                name: key.as_str().to_string(),
                dtype: "BF16",
                shape: dense.shape().to_vec(),
                bytes: dense.bytes().as_slice().to_vec(),
            }
        })
        .collect();
    write_hf_checkpoint(label, &fixture.config, &tensors)
}

/// `family`'s builtin fixture written as an HF checkpoint directory, every tensor as stored.
fn builtin_fixture_dir(family: &str) -> poot_test_util::UniqueTempPath {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap();
    fixture_dir(&format!("{family}_fixture"), &(entry.fixture)())
}

/// `ModelHandle::load` resolves the family before it reads a tensor: a checkpoint whose config names an
/// unregistered family is `Unsupported::Registry` even when its safetensors file is corrupt (reading it
/// would be a load error). Mutation: drop the resolve from `load_hf`, so the weights are read first; the
/// corrupt file's load error comes back instead and the row goes red.
#[test]
fn an_unregistered_family_is_refused_before_any_tensor_is_read() {
    use crate::driver::error::Unsupported;
    let dir = poot_test_util::unique_temp_path("poot_card737_unregistered");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({ "model_type": "no-such-family" }).to_string(),
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), b"not a safetensors archive").unwrap();
    match ModelHandle::load(
        &dir,
        &Registry::builtin().unwrap(),
        &crate::driver::LoadOptions::default(),
    ) {
        Err(DriverError::Unsupported(Unsupported::Registry(_))) => {}
        other => panic!("expected the registry refusal before any tensor read, got {other:?}"),
    }
}

/// POOT-570 SC-002: a MiniMax-M2 checkpoint directory whose config the old pinned counts refused (one
/// layer, four experts, no manifest anywhere) loads through `ModelHandle::load`, resolves to the
/// `minimax_m2` family and keeps the config's own counts. Mutation: reinstate a pinned layer count in
/// `MiniMaxM2Config::from_raw`; the load is refused and the row goes red.
#[test]
fn a_finetune_like_minimax_checkpoint_loads_through_model_handle_load() {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "minimax_m2")
        .unwrap();
    let (_, finetune) = entry
        .variants
        .iter()
        .find(|(name, _)| *name == "finetune")
        .expect("minimax_m2 ships a finetune-like fixture");
    let fixture = finetune();
    let dir = fixture_dir("minimax_m2_finetune", &fixture);
    let handle =
        ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default()).unwrap();
    assert_eq!(handle.config().family, entry.family);
    assert_eq!(handle.config().vocab, VOCAB);
}

/// Gemma is trained with a mandatory BOS: an HF gemma3 checkpoint's prompts encode with its BOS id
/// first (the fixture config names 1), as the Runner's loader did; a family without one (qwen2) encodes
/// the prompt as the tokenizer does. Mutation: drop gemma3's `prompt_bos` (`None`); the gemma3 encode
/// loses its BOS and the row goes red.
#[test]
fn an_hf_gemma3_checkpoint_encodes_with_its_mandatory_bos() {
    let registry = Registry::builtin().unwrap();
    let gemma3 = ModelHandle::load(
        &builtin_fixture_dir("gemma3"),
        &registry,
        &crate::driver::LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(gemma3.config().prompt_bos, Some(1));
    assert_eq!(
        gemma3.text().encode("d r p4").unwrap(),
        [1, 3, 17, 40],
        "gemma3 prepends its BOS"
    );
    let qwen2 = ModelHandle::load(
        &builtin_fixture_dir("qwen2"),
        &registry,
        &crate::driver::LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(qwen2.config().prompt_bos, None);
    assert_eq!(qwen2.text().encode("d r p4").unwrap(), [3, 17, 40]);
}

/// The compressed-tensors FP8 (per-output-channel E4M3) `quantization_config` block.
fn compressed_tensors_fp8() -> serde_json::Value {
    serde_json::json!({
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": true,
                    "dynamic": false,
                },
            },
        },
        "ignore": ["lm_head"],
    })
}

/// The builtin llama fixture as a compressed-tensors FP8 checkpoint (every attention and MLP
/// projection stored as E4M3 codes with an F32 scale per output row, everything else as stored), and
/// its decoded twin (each projection as the F32 values `e4m3(code) * scale[row]`).
fn llama_fp8_and_decoded_twin() -> (Vec<ArchiveTensor>, Vec<ArchiveTensor>) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "llama")
        .unwrap();
    let fixture = (entry.fixture)();
    // Finite E4M3 codes of both signs, magnitude 2^-6 .. 2^0.
    let codes = [0x18u8, 0x20, 0x28, 0x30, 0x98, 0xa0, 0xa8, 0xb0];
    let (mut packed, mut twin) = (Vec::new(), Vec::new());
    for (key, weight) in fixture.store.iter() {
        let poot_quant::weights::WeightEntry::Dense(dense) = weight else {
            panic!("the fixture is dense")
        };
        let name = key.as_str().to_string();
        let shape = dense.shape().to_vec();
        if !name.ends_with("_proj.weight") {
            for archive in [&mut packed, &mut twin] {
                archive.push(ArchiveTensor {
                    name: name.clone(),
                    dtype: "BF16",
                    shape: shape.clone(),
                    bytes: dense.bytes().as_slice().to_vec(),
                });
            }
            continue;
        }
        let (rows, cols) = (shape[0], shape[1]);
        let salt = name.len();
        let code_bytes: Vec<u8> = (0..rows * cols)
            .map(|i| codes[(i * 7 + salt) % codes.len()])
            .collect();
        let scales: Vec<f32> = (0..rows).map(|r| 0.25 + r as f32 * 0.0078125).collect();
        let decoded: Vec<u8> = code_bytes
            .iter()
            .enumerate()
            .flat_map(|(i, &code)| {
                (poot_quant::scalar::e4m3fn_to_f32(code) * scales[i / cols]).to_le_bytes()
            })
            .collect();
        packed.push(ArchiveTensor {
            name: name.clone(),
            dtype: "F8_E4M3",
            shape: shape.clone(),
            bytes: code_bytes,
        });
        packed.push(ArchiveTensor {
            name: format!("{name}_scale"),
            dtype: "F32",
            shape: vec![rows, 1],
            bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        twin.push(ArchiveTensor {
            name,
            dtype: "F32",
            shape,
            bytes: decoded,
        });
    }
    (packed, twin)
}

/// The GGUF type ids of F32 and of the quantized formats the packed-twin rows store.
const GGML_F32: u32 = 0;
const GGML_Q5_0: u32 = 6;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;

/// A tiny one-layer GGUF whose seven projections (and, with `packed_embedding`, its embedding and
/// tied head) are random `format` payloads and the rest F32, and its dense twin: the same file with
/// every packed tensor replaced by its decoded F32 values. Hidden size `width` (a K-quant row is 256
/// values, a Q8_0 or Q5_0 row 32), two heads over one KV head, vocabulary 8.
struct PackedGguf {
    packed: poot_test_util::UniqueTempPath,
    twin: poot_test_util::UniqueTempPath,
    /// Every packed payload by its GGUF tensor name.
    stored: HashMap<String, poot_quant::PackedPayload>,
}

fn packed_gguf(
    arch: &str,
    format: poot_quant::format::WeightFormat,
    ggml_type: u32,
    width: usize,
    packed_embedding: bool,
) -> PackedGguf {
    use poot_load::gguf::{GgufValue, write_gguf};
    use poot_quant::SourceRole;
    let (h, kv, inter, vocab) = (width, width / 2, width, 8usize);
    let dense = |n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (((i % 11) as f32) * 0.02 - 0.1).to_le_bytes())
            .collect()
    };
    let ones = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
    let mut stored = HashMap::new();
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let f32_tensor = |name: &str, dims: &[usize], bytes: Vec<u8>| {
        (
            name.to_string(),
            dims.iter().rev().map(|&d| d as u64).collect::<Vec<_>>(),
            GGML_F32,
            bytes,
        )
    };
    // Packed tensors in GGUF dims order (`[k, out]`), seeded by their position.
    let mut packed =
        |tensors: &mut Vec<(String, Vec<u64>, u32, Vec<u8>)>, name: &str, out: usize, k: usize| {
            let payload =
                poot_test_util::packed::random_payload(format, [out, k], (stored.len() + 1) as u64);
            tensors.push((
                name.to_string(),
                vec![k as u64, out as u64],
                ggml_type,
                payload.bytes(SourceRole::Blocks).to_vec(),
            ));
            stored.insert(name.to_string(), payload);
        };
    if packed_embedding {
        packed(&mut tensors, "token_embd.weight", vocab, h);
    } else {
        for (seed, name) in ["token_embd.weight", "output.weight"]
            .into_iter()
            .enumerate()
        {
            let values = poot_test_util::fill(h * vocab, 100 + seed as u64);
            tensors.push(f32_tensor(
                name,
                &[vocab, h],
                poot_test_util::f32_bytes(&values),
            ));
        }
    }
    tensors.push(f32_tensor("output_norm.weight", &[h], ones(h)));
    tensors.push(f32_tensor("blk.0.attn_norm.weight", &[h], ones(h)));
    tensors.push(f32_tensor("blk.0.ffn_norm.weight", &[h], ones(h)));
    if arch == "qwen2" {
        tensors.push(f32_tensor("blk.0.attn_q.bias", &[h], dense(h)));
        tensors.push(f32_tensor("blk.0.attn_k.bias", &[kv], dense(kv)));
        tensors.push(f32_tensor("blk.0.attn_v.bias", &[kv], dense(kv)));
    }
    for (name, out, k) in [
        ("blk.0.attn_q.weight", h, h),
        ("blk.0.attn_k.weight", kv, h),
        ("blk.0.attn_v.weight", kv, h),
        ("blk.0.attn_output.weight", h, h),
        ("blk.0.ffn_gate.weight", inter, h),
        ("blk.0.ffn_up.weight", inter, h),
        ("blk.0.ffn_down.weight", h, inter),
    ] {
        packed(&mut tensors, name, out, k);
    }

    let u = GgufValue::U32;
    let mut kvs = vec![
        (
            "general.architecture".to_string(),
            GgufValue::Str(arch.into()),
        ),
        (format!("{arch}.embedding_length"), u(h as u32)),
        (format!("{arch}.feed_forward_length"), u(inter as u32)),
        (format!("{arch}.block_count"), u(1)),
        (format!("{arch}.attention.head_count"), u(2)),
        (format!("{arch}.attention.head_count_kv"), u(1)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(1e-6),
        ),
        (format!("{arch}.context_length"), u(64)),
        (format!("{arch}.rope.freq_base"), GgufValue::F32(10_000.0)),
        ("tokenizer.ggml.eos_token_id".to_string(), u(NO_EOS)),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(
                ["a", "b", "c", "d", "e", "f", "g", "ab"]
                    .into_iter()
                    .map(|t| GgufValue::Str(t.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    if arch == "granite" {
        kvs.extend([
            (format!("{arch}.embedding_scale"), GgufValue::F32(1.0)),
            (format!("{arch}.attention.scale"), GgufValue::F32(0.088)),
            (format!("{arch}.residual_scale"), GgufValue::F32(1.0)),
            (format!("{arch}.logit_scale"), GgufValue::F32(1.0)),
        ]);
    }
    let write = |label: &str, tensors: &[(String, Vec<u64>, u32, Vec<u8>)]| {
        let kvs: Vec<(&str, GgufValue)> =
            kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
            .iter()
            .map(|(name, dims, ty, bytes)| (name.as_str(), dims.clone(), *ty, bytes.clone()))
            .collect();
        let path = poot_test_util::unique_temp_path(format!(
            "poot_card1018_{arch}_{ggml_type}_{label}.gguf"
        ));
        std::fs::write(&path, write_gguf(&kvs, &tensors)).unwrap();
        path
    };
    let twin_tensors: Vec<_> = tensors
        .iter()
        .map(|(name, dims, ty, bytes)| match stored.get(name) {
            Some(payload) => {
                let [out, k] = payload.weight().shape();
                let mut values = Vec::with_capacity(out * k * 4);
                let mut row = vec![0.0f32; k];
                for r in 0..out {
                    payload.decode_row(r, &mut row).unwrap();
                    values.extend(row.iter().flat_map(|v| v.to_le_bytes()));
                }
                (name.clone(), dims.clone(), GGML_F32, values)
            }
            None => (name.clone(), dims.clone(), *ty, bytes.clone()),
        })
        .collect();
    PackedGguf {
        packed: write("packed", &tensors),
        twin: write("twin", &twin_tensors),
        stored,
    }
}

/// The packed source bytes `store` holds for `name`, or a panic naming how it is stored.
fn packed_bytes(store: &WeightStore, name: &str, role: poot_quant::SourceRole) -> Vec<u8> {
    match store.get(name) {
        Some(WeightEntry::Packed(payload)) => payload.bytes(role).to_vec(),
        Some(WeightEntry::Dense(dense)) => panic!("{name} is stored dense as {:?}", dense.dtype()),
        None => panic!("{name} is not in the store"),
    }
}

/// SC-003: a compressed-tensors FP8 checkpoint of llama (not qwen2) loads through the registry with
/// every projection a packed E4M3 handle whose codes are the checkpoint's own bytes (no F32 copy in
/// the store), and generates on the device exactly what its decoded F32 twin generates; a Q4_K GGUF of
/// granite loads with its seven projections packed as stored and generates. Mutation: reinstate a
/// `model_type == "qwen2"` gate on packing quantized linears in `ModelHandle::load`; the llama load
/// is refused and the row goes red.
fn packed_checkpoints_load_as_stored_through_the_registry(choice: BackendChoice) {
    use poot_quant::format::WeightFormat;
    use poot_quant::{OperandRole, SourceRole};
    let registry = Registry::builtin().unwrap();
    let base = {
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.family.as_str() == "llama")
            .unwrap();
        let mut config = (entry.fixture)().config;
        config["eos_token_id"] = serde_json::json!(NO_EOS);
        config
    };
    let (packed, twin) = llama_fp8_and_decoded_twin();
    let mut fp8_config = base.clone();
    fp8_config["quantization_config"] = compressed_tensors_fp8();
    let fp8_dir = write_hf_checkpoint("llama_fp8", &fp8_config, &packed);
    let twin_dir = write_hf_checkpoint("llama_fp8_twin", &base, &twin);
    let fp8 = Arc::new(
        ModelHandle::load(&fp8_dir, &registry, &crate::driver::LoadOptions::default())
            .expect("the FP8 llama loads"),
    );
    let dense = Arc::new(
        ModelHandle::load(&twin_dir, &registry, &crate::driver::LoadOptions::default()).unwrap(),
    );

    // The packed handle each stored E4M3 tensor maps to.
    let handle_of = |name: &str| {
        fp8.model()
            .weights()
            .iter()
            .find(|(_, view, _)| {
                matches!(view, poot_quant::weights::WeightView::Stored(key) if key.as_str() == name)
            })
            .map(|(_, _, handle)| handle.format)
            .unwrap_or_else(|| panic!("{name} is not mapped"))
    };
    let mut projections = 0;
    for tensor in packed.iter().filter(|t| t.dtype == "F8_E4M3") {
        projections += 1;
        let codes = packed_bytes(
            fp8.store(),
            &tensor.name,
            SourceRole::Planar(OperandRole::Codes),
        );
        assert_eq!(
            codes, tensor.bytes,
            "{}: resident codes are the stored bytes",
            tensor.name
        );
        match handle_of(&tensor.name) {
            poot_quant::weights::HandleFormat::Packed(weight) => assert!(
                matches!(weight.format(), WeightFormat::E4m3PerChannel { .. }),
                "{}: {:?}",
                tensor.name,
                weight.format()
            ),
            dense => panic!("{} maps to {dense:?}, not a packed handle", tensor.name),
        }
    }
    let layers = base["num_hidden_layers"].as_u64().unwrap() as usize;
    assert_eq!(projections, 7 * layers, "seven projections per layer");
    assert!(
        poot_graph_plan::WeightFormats::from_weight_map(dense.model().weights()).is_empty(),
        "the twin is dense"
    );

    let gguf = packed_gguf("granite", WeightFormat::Q4_K, GGML_Q4_K, 256, false);
    let granite = Arc::new(
        ModelHandle::load(
            &gguf.packed,
            &registry,
            &crate::driver::LoadOptions::default(),
        )
        .expect("the Q4_K granite loads"),
    );
    let granite_twin = Arc::new(
        ModelHandle::load(
            &gguf.twin,
            &registry,
            &crate::driver::LoadOptions::default(),
        )
        .unwrap(),
    );
    assert!(
        poot_graph_plan::WeightFormats::from_weight_map(granite_twin.model().weights()).is_empty(),
        "the granite twin is dense"
    );
    let granite_formats =
        poot_graph_plan::WeightFormats::from_weight_map(granite.model().weights());
    for (name, payload) in &gguf.stored {
        assert_eq!(
            packed_bytes(granite.store(), name, SourceRole::Blocks),
            payload.bytes(SourceRole::Blocks),
            "{name}"
        );
    }
    assert_eq!(
        granite_formats
            .iter()
            .filter(|(_, packed)| packed.weight.format() == WeightFormat::Q4_K)
            .count(),
        7,
        "every granite projection is a packed Q4_K handle"
    );

    // Every executor opens before any runs (see `fixture_tokens_equal_the_recorded_literals`).
    let Some(first) = open(choice) else {
        return;
    };
    let mut fp8_driver = Driver::new(Arc::clone(&fp8), first, options(8, CAPACITY)).unwrap();
    let mut dense_driver = Driver::new(
        Arc::clone(&dense),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap();
    let mut granite_driver = Driver::new(
        Arc::clone(&granite),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap();
    let mut granite_twin_driver = Driver::new(
        Arc::clone(&granite_twin),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap();
    let got = generate(&mut fp8_driver, &PROMPT, 16, Sampler::greedy());
    let want = generate(&mut dense_driver, &PROMPT, 16, Sampler::greedy());
    assert_eq!(
        got, want,
        "the FP8 llama generates its decoded twin's tokens"
    );
    assert!(
        got.iter().any(|&t| t != got[0]),
        "the twin's sequence is not constant: {got:?}"
    );
    assert_same_trace(
        "the Q4_K granite",
        &mut granite_driver,
        &mut granite_twin_driver,
        &[1, 2, 3],
        8,
    );
}
per_backend!(
    packed_checkpoints_load_as_stored_through_the_registry,
    packed_checkpoints_load_as_stored_through_the_registry_wgpu,
    packed_checkpoints_load_as_stored_through_the_registry_rocm,
    packed_checkpoints_load_as_stored_through_the_registry_ptx
);

/// Card 1018 (the Runner's `a_packed_gguf_prefills_like_its_dense_twin` and
/// `a_packed_gguf_decodes_on_wgpu_through_compile_like_its_dense_twin`): a qwen2 and a llama GGUF whose
/// projections, embedding and tied head are packed `format` payloads load with every one of them a
/// packed handle (the dense twin has none), and the driver generates the twin's tokens: alone with a
/// chunked prefill (the contiguous entry), and through the serving surface, two sequences stepping
/// together over the paged pool (the shared-pool batched decode graph, which for Q5_0 is the graph
/// ROCm's gfx1151 compile must claim every projection of). Mutations: run
/// `recognize_packed_contractions` in place of `recognize_packed_row_gathers` in `compile` (the packed
/// embedding is refused: `PackedDequant(Unmatched { format: Q8_0 | Q5_0, logical_shape: [8, 32] })`);
/// flip a bit in every seventh byte of a packed GGUF tensor's read (the stored bytes leave the file's,
/// and with that assertion off the packed logprobs and tokens leave the twin's).
fn packed_gguf_generates_like_its_dense_twin(
    choice: BackendChoice,
    format: poot_quant::format::WeightFormat,
    ggml_type: u32,
) {
    use poot_quant::SourceRole;
    let registry = Registry::builtin().unwrap();
    struct Case {
        arch: &'static str,
        packed: Arc<ModelHandle>,
        twin: Arc<ModelHandle>,
        _files: PackedGguf,
    }
    let cases: Vec<Case> = ["qwen2", "llama"]
        .into_iter()
        .map(|arch| {
            let files = packed_gguf(arch, format, ggml_type, 32, true);
            let packed = Arc::new(
                ModelHandle::load(
                    &files.packed,
                    &registry,
                    &crate::driver::LoadOptions::default(),
                )
                .unwrap(),
            );
            let twin = Arc::new(
                ModelHandle::load(
                    &files.twin,
                    &registry,
                    &crate::driver::LoadOptions::default(),
                )
                .unwrap(),
            );
            let formats = poot_graph_plan::WeightFormats::from_weight_map(packed.model().weights());
            // Seven projections, the embedding gather and the tied head (the same payload twice).
            assert_eq!(
                formats.iter().count(),
                9,
                "{arch}: every weight is a packed handle"
            );
            for (name, payload) in &files.stored {
                assert_eq!(
                    packed_bytes(packed.store(), name, SourceRole::Blocks),
                    payload.bytes(SourceRole::Blocks),
                    "{arch} {name}: the packed bytes are the file's"
                );
            }
            assert!(
                poot_graph_plan::WeightFormats::from_weight_map(twin.model().weights()).is_empty(),
                "{arch}: the twin is dense"
            );
            Case {
                arch,
                packed,
                twin,
                _files: files,
            }
        })
        .collect();

    // Every executor opens before any runs (see `fixture_tokens_equal_the_recorded_literals`).
    let mut drivers = Vec::new();
    // The packed model runs twice: alone on the contiguous layout, and on the paged pool.
    for case in &cases {
        for handle in [&case.packed, &case.twin, &case.packed] {
            let Some(executor) = open(choice) else {
                return;
            };
            drivers.push(
                Driver::new(
                    Arc::clone(handle),
                    executor,
                    options(SERVING_CHUNK, SERVING_CAPACITY),
                )
                .unwrap(),
            );
        }
    }
    let long: [u32; 7] = [0, 3, 1, 2, 5, 7, 4];
    let short: [u32; 3] = [6, 1, 2];
    let mut drivers = drivers.iter_mut();
    for case in &cases {
        let (packed, twin, paged) = (
            drivers.next().unwrap(),
            drivers.next().unwrap(),
            drivers.next().unwrap(),
        );
        assert_same_trace(case.arch, packed, twin, &long, 16);
        let want = generate(twin, &long, 16, Sampler::greedy());

        // Two sequences through the paged pool: the batched decode graph.
        let want_short = generate(twin, &short, 12, Sampler::greedy());
        prepare_serving(paged, pool_shape(8, 4), &[2], &[Head::GREEDY], &[]);
        let mut lanes = vec![
            Lane::open(paged, &long, 16, Sampler::greedy()).0,
            Lane::open(paged, &short, 12, Sampler::greedy()).0,
        ];
        run_to_end(paged, &mut lanes);
        assert_eq!(lanes[0].out, want, "{}: long lane", case.arch);
        assert_eq!(lanes[1].out, want_short, "{}: short lane", case.arch);
        for lane in &lanes {
            paged.release(lane.seq, Release::Finished).unwrap();
        }
    }
}
fn packed_q8_0_gguf_generates_like_its_dense_twin(choice: BackendChoice) {
    packed_gguf_generates_like_its_dense_twin(
        choice,
        poot_quant::format::WeightFormat::Q8_0,
        GGML_Q8_0,
    );
}
per_backend!(
    packed_q8_0_gguf_generates_like_its_dense_twin,
    packed_q8_0_gguf_generates_like_its_dense_twin_wgpu,
    packed_q8_0_gguf_generates_like_its_dense_twin_rocm,
    packed_q8_0_gguf_generates_like_its_dense_twin_ptx
);
fn packed_q5_0_gguf_generates_like_its_dense_twin(choice: BackendChoice) {
    packed_gguf_generates_like_its_dense_twin(
        choice,
        poot_quant::format::WeightFormat::Q5_0,
        GGML_Q5_0,
    );
}
per_backend!(
    packed_q5_0_gguf_generates_like_its_dense_twin,
    packed_q5_0_gguf_generates_like_its_dense_twin_wgpu,
    packed_q5_0_gguf_generates_like_its_dense_twin_rocm,
    packed_q5_0_gguf_generates_like_its_dense_twin_ptx
);

/// Real checkpoints through `Registry::builtin()` -> `ModelHandle::load` -> `Driver::generate`, against
/// the tokens the Runner generated for them before this card (recorded on the base commit `2ef7ea834`
/// with `generate_kv_gpu_prefilled`, identical on wgpu and ROCm; the driver equalled them there), and
/// rule 18's coherence references. Checkpoint-device rows under rule 18's memory gates, `#[ignore]`d.
mod checkpoint {
    use super::*;

    /// Qwen2.5-0.5B's 24 greedy tokens after "The capital of Germany is Berlin. The capital of France
    /// is" (bf16 safetensors and the Q8_0 GGUF agree).
    const QWEN2_5_0_5B: [u32; 24] = [
        12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 6323, 374, 26194, 13,
        576, 6722, 315, 279, 3639, 4180, 374, 6515,
    ];
    const QWEN2_5_0_5B_PROMPT: [u32; 12] = [
        785, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315, 9625, 374,
    ];
    /// `gemma4-moe-tiny` (the multimodal `Gemma4ForConditionalGeneration` wrapper, 128 experts): the
    /// CPU oracle's 24 greedy tokens after this prompt (POOT-574 rule 18, ADR-0101 tier 3).
    const GEMMA4_MOE_TINY_PROMPT: [u32; 8] = [2, 105, 2364, 7, 11, 300, 1044, 9];
    const GEMMA4_MOE_TINY: [u32; 24] = [
        121029, 109844, 185071, 99208, 185071, 99208, 185071, 99208, 185071, 99208, 185071, 99208,
        185071, 99208, 185071, 99208, 185071, 99208, 185071, 99208, 185071, 99208, 185071, 99208,
    ];
    fn load(path: &std::path::Path) -> Arc<ModelHandle> {
        Arc::new(
            ModelHandle::load(
                path,
                &Registry::builtin().unwrap(),
                &crate::driver::LoadOptions::default(),
            )
            .unwrap(),
        )
    }

    /// The recorded tokens for `prompt` at the whole prompt and at `prefill_chunk = 5`.
    fn recorded(
        path: &std::path::Path,
        prompt: &[u32],
        expected: &[u32; 24],
        choice: BackendChoice,
    ) {
        let handle = load(path);
        // Every executor opens before any runs (see `fixture_tokens_equal_the_recorded_literals`).
        let chunks = [prompt.len(), 5];
        let Some(first) = open(choice) else {
            return;
        };
        let mut drivers =
            vec![Driver::new(Arc::clone(&handle), first, options(chunks[0], 64)).unwrap()];
        drivers.push(
            Driver::new(
                Arc::clone(&handle),
                open(choice).unwrap(),
                options(chunks[1], 64),
            )
            .unwrap(),
        );
        for (chunk, driver) in chunks.iter().zip(&mut drivers) {
            let got = generate(driver, prompt, 24, Sampler::greedy());
            assert_eq!(&got[..], expected, "{}: chunk {chunk}", path.display());
        }
    }

    /// Rule 18's qwen2 reference: the first greedy token after "The capital of France is" is " Paris"
    /// and the continuation starts with the documented reference text.
    fn qwen2_reference(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        let Some(executor) = open(choice) else {
            return;
        };
        let mut driver = Driver::new(Arc::clone(&handle), executor, options(8, 64)).unwrap();
        let prompt = "The capital of France is";
        let ids = handle.text().encode(prompt).unwrap();
        let tokens = generate(&mut driver, &ids, 10, Sampler::greedy());
        assert_eq!(handle.text().decode(&tokens[..1]).unwrap(), " Paris");
        let text = format!("{prompt}{}", handle.text().decode(&tokens).unwrap());
        assert!(
            text.starts_with("The capital of France is Paris. It is the largest city in Europe"),
            "continuation was: {text:?}"
        );
    }

    /// Card 1018 (`qwen2_fixed_kv_decode_matches_reference`, `qwen2_masked_decode_matches_reference`):
    /// the driver's decode is the constant-shape masked decode over a fixed-capacity cache. On the real
    /// checkpoint, a request at the exact capacity (the prompt and its ten new tokens fill every
    /// position) and one at ten times it produce the same tokens, and they are the documented
    /// continuation. Mutation: fill `Slot::Pos` from the capacity; the large-capacity run diverges.
    fn masked_decode_is_capacity_independent(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        let prompt = "The capital of France is";
        let ids = handle.text().encode(prompt).unwrap();
        let exact = ids.len() + 10;
        // Both executors open before either runs (see `fixture_tokens_equal_the_recorded_literals`).
        let Some(first) = open(choice) else {
            return;
        };
        let mut drivers = vec![Driver::new(Arc::clone(&handle), first, options(8, exact)).unwrap()];
        drivers.push(
            Driver::new(
                Arc::clone(&handle),
                open(choice).unwrap(),
                options(8, 10 * exact),
            )
            .unwrap(),
        );
        let runs: Vec<Vec<u32>> = drivers
            .iter_mut()
            .map(|driver| generate(driver, &ids, 10, Sampler::greedy()))
            .collect();
        assert_eq!(runs[0], runs[1], "the capacity only sizes buffers");
        let text = format!("{prompt}{}", handle.text().decode(&runs[0]).unwrap());
        assert!(
            text.starts_with("The capital of France is Paris. It is the largest city in Europe"),
            "continuation was: {text:?}"
        );
    }

    /// Card 1018 (`qwen2_cached_sampled_decode_is_deterministic_for_a_fixed_seed_on_gpu`): a seeded
    /// temperature draw on the real checkpoint is deterministic: the same seed on one driver twice and
    /// on a second driver gives the same tokens, and another seed gives different ones. Mutation: seed
    /// the draw from the clock; the repeat differs.
    fn sampled_decode_is_deterministic_for_a_fixed_seed(
        path: &std::path::Path,
        choice: BackendChoice,
    ) {
        let handle = load(path);
        let ids = handle.text().encode("The capital of France is").unwrap();
        let Some(mut first) = open_driver(&handle, choice) else {
            return;
        };
        let mut second = open_driver(&handle, choice).unwrap();
        let draw = |driver: &mut Driver, seed: u64| {
            generate(driver, &ids, 16, Sampler::new(0.8, 40, 1.0, seed))
        };
        let a = draw(&mut first, 7);
        assert_eq!(draw(&mut first, 7), a, "the same seed on one driver");
        assert_eq!(draw(&mut second, 7), a, "the same seed on another driver");
        assert_ne!(draw(&mut first, 8), a, "another seed draws other tokens");
    }

    /// Rule 18's packed decodes: a GPTQ or AWQ checkpoint loads with its linears packed and answers
    /// "Paris".
    fn packed_answers_paris(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        let probe = "model.layers.0.self_attn.q_proj";
        assert!(
            handle
                .store()
                .iter()
                .any(|(key, entry)| key.as_str().starts_with(probe)
                    && matches!(entry, poot_quant::weights::WeightEntry::Packed(_))),
            "{probe} loads packed"
        );
        let Some(executor) = open(choice) else {
            return;
        };
        let mut driver = Driver::new(Arc::clone(&handle), executor, options(8, 64)).unwrap();
        let prompt = "The capital of France is";
        let ids = handle.text().encode(prompt).unwrap();
        let tokens = generate(&mut driver, &ids, 8, Sampler::greedy());
        let text = handle.text().decode(&tokens).unwrap();
        assert!(
            text.to_lowercase().contains("paris"),
            "{}: {text:?}",
            path.display()
        );
    }

    /// A [`Driver`] over `handle` on `choice` with room for a long prompt, or `None` after a reported
    /// skip (the device does not open).
    fn open_driver(handle: &Arc<ModelHandle>, choice: BackendChoice) -> Option<Driver> {
        let executor = open(choice)?;
        Some(Driver::new(Arc::clone(handle), executor, options(8, 128)).unwrap())
    }

    const CAPITAL: &str = "The capital of France is";

    /// The greedy continuation of `prompt`, as text, or `None` after a reported skip.
    fn continuation(
        path: &std::path::Path,
        choice: BackendChoice,
        prompt: &str,
        max_new: usize,
    ) -> Option<String> {
        let handle = load(path);
        let mut driver = open_driver(&handle, choice)?;
        let ids = handle.text().encode(prompt).unwrap();
        let tokens = generate(&mut driver, &ids, max_new, Sampler::greedy());
        let text = handle.text().decode(&tokens).unwrap();
        eprintln!("{}: {text:?}", path.display());
        Some(text)
    }

    /// The old Runner coherence rows: the greedy continuation of `prompt` names `needle`, which the
    /// garbled output of a wrong norm, scale, mask or placement does not.
    fn continues_with(
        path: &std::path::Path,
        choice: BackendChoice,
        prompt: &str,
        max_new: usize,
        needle: &str,
    ) {
        let Some(text) = continuation(path, choice, prompt, max_new) else {
            return;
        };
        assert!(
            text.to_lowercase().contains(needle),
            "{}: the continuation of {prompt:?} was {text:?}",
            path.display()
        );
    }

    /// A secret word stated once at the start of ~5900 tokens of filler, asked for at the end. The
    /// rescaled low-frequency band (`rope_freqs.weight`) is what carries that long-range lookup:
    /// plain RoPE answers a distractor word instead (measured, Card 1019). The filler carries digits and
    /// the prompt gets its BOS from `encode`, both as llama.cpp tokenizes them.
    fn recalls_a_word_across_a_long_context(path: &std::path::Path, choice: BackendChoice) {
        const WORDS: [&str; 8] = [
            "amber", "birch", "cedar", "dune", "ember", "fjord", "grove", "harbor",
        ];
        let handle = load(path);
        let Some(mut driver) = driver(choice, &handle, 64, 8192) else {
            return;
        };
        let filler: String = (0..730)
            .map(|i| {
                format!(
                    "The {} crate no. {i} holds {} bolts. ",
                    WORDS[i % 8],
                    WORDS[(i * 3 + 1) % 8]
                )
            })
            .collect();
        let prompt = format!(
            "The secret word is pelican. {filler}\nQuestion: what is the secret word?\nAnswer: The secret word is"
        );
        let ids = handle.text().encode(&prompt).unwrap();
        eprintln!("prompt tokens: {}", ids.len());
        let tokens = generate(&mut driver, &ids, 8, Sampler::greedy());
        let text = handle.text().decode(&tokens).unwrap();
        eprintln!("{}: {text:?}", path.display());
        assert!(text.contains("pelican"), "the word was lost: {text:?}");
    }

    fn names_paris(path: &std::path::Path, choice: BackendChoice) {
        continues_with(path, choice, CAPITAL, 12, "paris");
    }

    /// A coherent English continuation of a prompt with no single right answer: ASCII, several words
    /// and a very common function word. The garbled failure of a wrong norm fold is non-ASCII script
    /// fragments and token salad with neither.
    fn continues_in_english(path: &std::path::Path, choice: BackendChoice) {
        let prompt = "Once upon a time";
        let Some(text) = continuation(path, choice, prompt, 16) else {
            return;
        };
        let cont = text.strip_prefix(prompt).unwrap_or(&text).trim();
        assert!(cont.is_ascii(), "ASCII English expected: {cont:?}");
        assert!(
            cont.split_whitespace().count() >= 5,
            "a multi-word continuation expected: {cont:?}"
        );
        let low = format!(" {} ", cont.to_lowercase());
        assert!(
            low.contains(" a ") || low.contains(" the ") || low.contains(" in "),
            "common English words expected: {cont:?}"
        );
    }

    /// MPT-1B is a Korean-primary 1.3B model: it continues "The capital of France is" with a wrong
    /// fact in fluent text. A wiring bug gives token salad, so the gate is a second prompt's fluent
    /// narrative continuation.
    fn mptk_is_fluent(path: &std::path::Path, choice: BackendChoice) {
        continues_with(
            path,
            choice,
            "Once upon a time, there was a",
            15,
            "beautiful",
        );
    }

    /// OLMo-2-1B's checkpoint is natively bf16: every projection stays a BF16 constant (never widened
    /// to F32 in the store) and the device still prefills and decodes it, which needs `compile` to
    /// fold the `F32 x BF16` matmul pairing every projection is. Mutation: make `fold_dense_contractions`
    /// and `fold_dense_bf16_row_gathers` return the graph unchanged; the load is refused with
    /// `Plan(Refusal { op: MatMul, operands: [F32, BF16], missing: DtypeLowering })` on wgpu and ROCm.
    fn bf16_checkpoint_decodes(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        let probe = "model.layers.0.self_attn.q_proj.weight";
        match handle.store().get(probe) {
            Some(WeightEntry::Dense(dense)) => assert_eq!(
                dense.dtype(),
                poot_tensor::DType::BF16,
                "{probe}: the test is only meaningful on a natively-bf16 checkpoint"
            ),
            other => panic!("{probe} is {:?}", other.map(|_| "a packed weight")),
        }
        let Some(mut driver) = open_driver(&handle, choice) else {
            return;
        };
        let ids = handle.text().encode(CAPITAL).unwrap();
        let tokens = generate(&mut driver, &ids, 8, Sampler::greedy());
        let text = handle.text().decode(&tokens).unwrap();
        eprintln!("{}: {text:?}", path.display());
        assert_eq!(tokens.len(), 8, "decode produced new tokens: {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "{}: {text:?}",
            path.display()
        );
    }

    /// Card 1018 (`post_prefill_prepare_real_gguf_matches_cpu_gpu`): a real qwen2 GGUF answers a chat
    /// prompt, cold and again on the warm driver, with the same tokens, and the answer names Paris. The
    /// Runner compared against its CPU eager path; the driver has no CPU path, so the independent claim
    /// is the answer and the repeat is the prepare-then-reuse claim.
    fn chat_answer_repeats(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        let Some(mut driver) = open_driver(&handle, choice) else {
            return;
        };
        let prompt =
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
        let ids = handle.text().encode(prompt).unwrap();
        let cold = generate(&mut driver, &ids, 8, Sampler::greedy());
        let warm = generate(&mut driver, &ids, 8, Sampler::greedy());
        let text = handle.text().decode(&cold).unwrap();
        eprintln!("{} ids={ids:?} -> {cold:?} {text:?}", path.display());
        assert_eq!(cold, warm, "{}: cold vs repeated", path.display());
        assert!(
            text.to_lowercase().contains("paris"),
            "{}: the answer was {text:?}",
            path.display()
        );
    }

    /// Guided generation on a real model: the constraint (built from the model's own text codec) holds
    /// the output to its language whatever the model would freely say. A guided request takes the host
    /// pick path, so these rows also lock that the constraint reaches the device generate loop.
    fn guided_text(
        path: &std::path::Path,
        choice: BackendChoice,
        prompt: &str,
        max_new: usize,
        build: impl FnOnce(&TextCodec) -> crate::Result<crate::Constraint>,
    ) -> Option<(String, usize)> {
        let handle = load(path);
        let mut driver = open_driver(&handle, choice)?;
        let ids = handle.text().encode(prompt).unwrap();
        let constraint = build(handle.text()).unwrap();
        let tokens = generate(
            &mut driver,
            &ids,
            max_new,
            Sampler::greedy().with_constraint(constraint),
        );
        let text = handle.text().decode(&tokens).unwrap();
        eprintln!("{}: {text:?}", path.display());
        Some((text, tokens.len()))
    }

    fn guided_choice(path: &std::path::Path, choice: BackendChoice) {
        let choices = ["positive".to_string(), "negative".to_string()];
        let Some((out, n)) = guided_text(
            path,
            choice,
            "Review: I absolutely loved this movie, it was wonderful. Sentiment:",
            16,
            |text| text.build_choice_constraint(&choices),
        ) else {
            return;
        };
        assert!(choices.contains(&out), "{out:?} is one of {choices:?}");
        assert!(n < 16, "stopped at the choice end, not at max_new");
    }

    /// GBNF `root ::= "yes" | "no"`: exactly one of the two, then the constraint forces the end.
    fn guided_gbnf_sentence(path: &std::path::Path, choice: BackendChoice) {
        let Some((out, n)) = guided_text(
            path,
            choice,
            "Is fire hot? Answer with one word: ",
            16,
            |text| text.build_grammar_constraint(r#"root ::= "yes" | "no""#),
        ) else {
            return;
        };
        assert!(out == "yes" || out == "no", "grammar output {out:?}");
        assert!(n < 16, "stopped at the sentence end");
    }

    /// GBNF `{m,n}` bounded repetition: `[0-9]{4}` forces exactly four digits.
    fn guided_gbnf_bounded_repetition(path: &std::path::Path, choice: BackendChoice) {
        let Some((out, _)) = guided_text(path, choice, "Pick a 4-digit code: ", 16, |text| {
            text.build_grammar_constraint("root ::= [0-9]{4}")
        }) else {
            return;
        };
        assert!(
            out.len() == 4 && out.chars().all(|c| c.is_ascii_digit()),
            "bounded-repetition output {out:?}"
        );
    }

    /// A forced tool call is a complete ChatML `<tool_call>{json}</tool_call>` block with the forced
    /// name and a schema-conforming argument (a bounded enum, so greedy decoding terminates).
    fn guided_forced_tool_call(path: &std::path::Path, choice: BackendChoice) {
        let params = serde_json::json!({
            "type": "object",
            "properties": { "city": { "enum": ["Paris", "London"] } },
            "required": ["city"],
        });
        let forced = [("get_weather".to_string(), params)];
        let Some((out, _)) =
            guided_text(path, choice, "What's the weather in Paris? ", 48, |text| {
                text.build_tool_call_constraint(&forced)
            })
        else {
            return;
        };
        let body = out
            .strip_prefix("<tool_call>")
            .and_then(|s| s.trim().strip_suffix("</tool_call>"))
            .unwrap_or_else(|| panic!("{out:?} is not a complete <tool_call> block"))
            .trim();
        let call: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("tool-call body {body:?} is not valid JSON: {e}"));
        assert_eq!(call["name"], "get_weather", "forced tool name");
        assert!(
            call["arguments"]["city"] == "Paris" || call["arguments"]["city"] == "London",
            "the schema's enum argument: {call}"
        );
    }

    /// An object-level `const` schema: the output is exactly the fixed object, as valid JSON.
    fn guided_json_const_object(path: &std::path::Path, choice: BackendChoice) {
        let schema = serde_json::json!({ "const": { "status": "ok", "code": 200 } });
        let Some((out, _)) = guided_text(
            path,
            choice,
            "Return the fixed status object: ",
            48,
            |text| text.build_json_constraint(&schema),
        ) else {
            return;
        };
        let value: serde_json::Value = serde_json::from_str(out.trim())
            .unwrap_or_else(|e| panic!("{out:?} is not valid JSON: {e}"));
        assert_eq!(value, serde_json::json!({ "status": "ok", "code": 200 }));
    }

    /// A multi-property schema of bounded fields: the output is complete JSON conforming to it, with
    /// no stall on optional whitespace between structural tokens.
    fn guided_json_multi_field(path: &std::path::Path, choice: BackendChoice) {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "sentiment": { "enum": ["positive", "negative"] },
                "confident": { "type": "boolean" },
                "score": { "type": "integer", "minimum": 0, "maximum": 9 },
            },
            "required": ["sentiment", "confident", "score"],
        });
        let Some((out, _)) = guided_text(
            path,
            choice,
            "Rate this review as JSON: I loved it. ",
            64,
            |text| text.build_json_constraint(&schema),
        ) else {
            return;
        };
        let value: serde_json::Value = serde_json::from_str(out.trim())
            .unwrap_or_else(|e| panic!("{out:?} is not complete valid JSON: {e}"));
        assert!(
            value["sentiment"] == "positive" || value["sentiment"] == "negative",
            "sentiment enum: {value}"
        );
        assert!(value["confident"].is_boolean(), "confident bool: {value}");
        assert!(
            value["score"]
                .as_i64()
                .is_some_and(|n| (0..=9).contains(&n)),
            "score in range: {value}"
        );
    }

    /// Free `json_object`: the output is never free text. It is a whole JSON value or the valid
    /// prefix of one (cut off at `max_new`, which serde reports as an end-of-input error and not a
    /// syntax error), and when generation ended before `max_new` it is complete valid JSON.
    fn guided_json_object_is_not_degenerate(path: &std::path::Path, choice: BackendChoice) {
        let max_new = 40;
        let Some((out, n)) = guided_text(
            path,
            choice,
            "Give me a JSON object describing a cat: ",
            max_new,
            |text| text.build_json_object_constraint(),
        ) else {
            return;
        };
        let trimmed = out.trim();
        assert!(!trimmed.is_empty(), "json_object output {out:?} is blank");
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(_) => {}
            Err(error) => {
                assert!(
                    error.is_eof() && n == max_new,
                    "json_object output {out:?} (ended early: {}) is not a JSON value or its prefix: {error}",
                    n < max_new
                );
            }
        }
    }

    /// The old Runner MoE coherence rows (qwen3-moe, mixtral, olmoe, gpt-oss, granitemoe tiny checkpoints):
    /// the checkpoint loads through the registry and decodes through the driver, a prefill split into
    /// chunks of four generates the tokens a whole-prompt prefill does, and the continuation is not empty.
    /// The Runner compared the device with its CPU re-prefill; the driver has no CPU path, so the
    /// independent claim is the chunking invariance of the routed prefill.
    fn moe_generates(path: &std::path::Path, choice: BackendChoice) {
        let handle = load(path);
        // Every executor opens before any runs (see `fixture_tokens_equal_the_recorded_literals`).
        let Some(mut whole) = driver(choice, &handle, 64, 128) else {
            return;
        };
        let Some(mut chunked) = driver(choice, &handle, 4, 128) else {
            return;
        };
        let ids = handle.text().encode(CAPITAL).unwrap();
        let tokens = generate(&mut whole, &ids, 12, Sampler::greedy());
        let split = generate(&mut chunked, &ids, 12, Sampler::greedy());
        let text = handle.text().decode(&tokens).unwrap();
        eprintln!("{}: {text:?}", path.display());
        assert!(!tokens.is_empty(), "{}: nothing generated", path.display());
        assert_eq!(tokens, split, "{}: prefill chunking", path.display());
    }

    /// One `#[ignore]`d row per checkpoint and backend (each its own test and process: the ROCm runtime
    /// does not survive reopening an executor in one process). The `ptx` rows run on a pod that holds
    /// the model.
    macro_rules! rows {
        ($($name:ident: $checkpoint:literal, $body:expr, $why:literal;)*) => {
            $(
                mod $name {
                    use super::*;

                    fn run(choice: BackendChoice) {
                        let Some(path) =
                            poot_test_util::model_path(poot_test_util::checkpoint!($checkpoint))
                        else {
                            return;
                        };
                        let body: fn(&std::path::Path, BackendChoice) = $body;
                        body(&path, choice);
                    }

                    #[test]
                    #[ignore = $why]
                    fn wgpu() {
                        run(BackendChoice::Wgpu);
                    }

                    #[cfg(feature = "rocm")]
                    #[test]
                    #[ignore = $why]
                    fn rocm() {
                        run(BackendChoice::Rocm);
                    }

                    #[test]
                    #[ignore = $why]
                    fn ptx() {
                        run(BackendChoice::Ptx);
                    }
                }
            )*
        };
    }

    rows! {
        qwen2_bf16_recorded: "qwen2.5-0.5b",
            |path, choice| recorded(path, &QWEN2_5_0_5B_PROMPT, &QWEN2_5_0_5B, choice),
            "loads the real qwen2.5-0.5b bf16 checkpoint onto the GPU; needs >= 40Gi available";
        qwen2_q8_0_recorded: "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf",
            |path, choice| recorded(path, &QWEN2_5_0_5B_PROMPT, &QWEN2_5_0_5B, choice),
            "loads the real qwen2.5-0.5b Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        gemma4_moe_tiny_recorded: "gemma4-moe-tiny",
            |path, choice| recorded(path, &GEMMA4_MOE_TINY_PROMPT, &GEMMA4_MOE_TINY, choice),
            "loads the real gemma4-moe-tiny checkpoint onto the GPU; needs >= 40Gi available";
        qwen2_greedy_reference: "qwen2.5-0.5b", qwen2_reference,
            "loads the real qwen2.5-0.5b bf16 checkpoint onto the GPU; needs >= 40Gi available";
        qwen2_masked_decode_capacity_independent: "qwen2.5-0.5b", masked_decode_is_capacity_independent,
            "loads the real qwen2.5-0.5b bf16 checkpoint onto the GPU; needs >= 40Gi available";
        qwen2_sampled_decode_seeded: "qwen2.5-0.5b", sampled_decode_is_deterministic_for_a_fixed_seed,
            "loads the real qwen2.5-0.5b bf16 checkpoint onto the GPU; needs >= 40Gi available";
        gptq_packed_decode: "qwen2.5-0.5b-gptq", packed_answers_paris,
            "loads qwen2.5-0.5b-gptq packed onto the GPU; needs >= 40Gi available";
        awq_packed_decode: "qwen2.5-0.5b-awq", packed_answers_paris,
            "loads qwen2.5-0.5b-awq packed onto the GPU; needs >= 40Gi available";
        fp8_packed_decode: "qwen2.5-0.5b-fp8", packed_answers_paris,
            "loads qwen2.5-0.5b-fp8 packed onto the GPU; needs >= 40Gi available";
        qwen2_q4_k_m_coherent: "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf", names_paris,
            "loads the real qwen2.5-0.5b Q4_K_M GGUF (Q4_K, Q5_0, Q6_K, Q8_0 tensors) onto the GPU; needs >= 40Gi available";
        qwen2_q4_k_m_chat_answer_repeats: "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            chat_answer_repeats,
            "loads the real qwen2.5-0.5b Q4_K_M GGUF onto the GPU; needs >= 40Gi available";
        qwen2_q8_0_chat_answer_repeats: "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf",
            chat_answer_repeats,
            "loads the real qwen2.5-0.5b Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        lite_mistral_coherent: "lite-mistral-150m", names_paris,
            "loads lite-mistral-150m onto the GPU; needs >= 40Gi available";
        tinyllama_coherent: "tinyllama-1.1b-chat", names_paris,
            "loads tinyllama-1.1b onto the GPU; needs >= 40Gi available";
        llama3_gguf_coherent: "llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf", names_paris,
            "loads the Llama-3.2-1B Q4_K_M GGUF (llama3 rope_freqs.weight) onto the GPU; needs >= 40Gi available";
        llama3_gguf_long_context: "llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            recalls_a_word_across_a_long_context,
            "loads the Llama-3.2-1B Q4_K_M GGUF onto the GPU; needs >= 40Gi available";
        smollm2_gguf_coherent: "smollm2-135m-gguf/SmolLM2-135M-Instruct-Q8_0.gguf", names_paris,
            "loads the SmolLM2-135M Q8_0 GGUF (llama arch) onto the GPU; needs >= 40Gi available";
        gemma3_coherent: "gemma-3-1b-it", names_paris,
            "loads gemma-3-1b-it onto the GPU; needs >= 40Gi available";
        gemma3_gguf_coherent: "gemma-3-1b-gguf/gemma-3-1b-it-Q4_K_M.gguf", continues_in_english,
            "loads the gemma-3-1b Q4_K_M GGUF onto the GPU; needs >= 40Gi available";
        olmo2_bf16_decodes: "olmo2-1b", bf16_checkpoint_decodes,
            "loads the natively bf16 OLMo-2-1B onto the GPU; needs >= 40Gi available";
        olmo2_gguf_coherent: "olmo2-1b-gguf/OLMo-2-0425-1B-Instruct-Q8_0.gguf", names_paris,
            "loads the OLMo-2-1B Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        granite_dense_coherent: "granite-3.1-2b", names_paris,
            "loads granite-3.1-2b onto the GPU; needs >= 40Gi available";
        phi3_gguf_coherent: "phi-3-mini-gguf/Phi-3-mini-4k-instruct-q4.gguf", names_paris,
            "loads the Phi-3-mini-4k Q4 GGUF (fused qkv and gate|up) onto the GPU; needs >= 40Gi available";
        qwen3_coherent: "qwen3-0.6b", names_paris,
            "loads qwen3-0.6b onto the GPU; needs >= 40Gi available";
        qwen3_gguf_coherent: "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", names_paris,
            "loads the Qwen3-0.6B Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        bloom_coherent: "bloom-560m",
            |path, choice| continues_with(path, choice, CAPITAL, 20, "paris"),
            "loads bloom-560m onto the GPU; needs >= 40Gi available";
        bloom_gguf_coherent: "bloom-560m-gguf/bloom-560m.Q8_0.gguf",
            |path, choice| continues_with(path, choice, CAPITAL, 20, "paris"),
            "loads the bloom-560m Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        mpt_coherent: "mptk-1b", mptk_is_fluent,
            "loads mptk-1b (MPT-1.3B, f32) onto the GPU; needs >= 40Gi available";
        smollm3_coherent: "smollm3-3b",
            |path, choice| continues_with(path, choice, CAPITAL, 20, "paris"),
            "loads the bf16 SmolLM3-3B onto the GPU; needs >= 40Gi available";
        qwen3_moe_tiny: "qwen3-moe-tiny", moe_generates,
            "loads qwen3-moe-tiny onto the GPU; needs >= 40Gi available";
        qwen3_moe_sparse: "qwen3-moe-tiny-sparse", moe_generates,
            "loads qwen3-moe-tiny-sparse onto the GPU; needs >= 40Gi available";
        qwen3_moe_sparse_gguf: "qwen3-moe-tiny-sparse-gguf/model.gguf", moe_generates,
            "loads the qwen3-moe-tiny-sparse GGUF onto the GPU; needs >= 40Gi available";
        mixtral_tiny: "mixtral-tiny", moe_generates,
            "loads mixtral-tiny (~945MB f32) onto the GPU; needs >= 40Gi available";
        mixtral_tiny_gguf: "mixtral-tiny/mixtral-tiny-f32.gguf", moe_generates,
            "loads the mixtral-tiny f32 GGUF (a llama-architecture GGUF with experts) onto the GPU; needs >= 40Gi available";
        olmoe_tiny: "olmoe-tiny", moe_generates,
            "loads olmoe-tiny onto the GPU; needs >= 40Gi available";
        olmoe_tiny_gguf: "olmoe-tiny/olmoe-tiny-f32.gguf", moe_generates,
            "loads the olmoe-tiny f32 GGUF onto the GPU; needs >= 40Gi available";
        gptoss_tiny: "gptoss-tiny", moe_generates,
            "loads gptoss-tiny onto the GPU; needs >= 40Gi available";
        gptoss_tiny_gguf: "gptoss-tiny/gptoss-tiny-f32.gguf", moe_generates,
            "loads the gptoss-tiny f32 GGUF onto the GPU; needs >= 40Gi available";
        granitemoe_tiny: "granitemoe-tiny", moe_generates,
            "loads granitemoe-tiny onto the GPU; needs >= 40Gi available";
        granite_moe_1b_gguf: "granite-moe-1b-gguf/granite-3.0-1b-a400m-instruct-Q8_0.gguf", moe_generates,
            "loads the granite-3.0-1b-a400m Q8_0 GGUF onto the GPU; needs >= 40Gi available";
        deepseek2_tiny: "deepseek2-tiny", moe_generates,
            "loads deepseek2-tiny (the MLA checkpoint, safetensors) onto the GPU; needs >= 40Gi available";
        deepseek2_tiny_gguf: "deepseek2-tiny/deepseek2-tiny-f32.gguf", moe_generates,
            "loads the deepseek2-tiny f32 GGUF (split per-head attn_k_b and attn_v_b) onto the GPU; needs >= 40Gi available";
        deepseek3_tiny: "deepseek3-tiny", moe_generates,
            "loads deepseek3-tiny (the sigmoid-router MLA checkpoint) onto the GPU; needs >= 40Gi available";
        deepseek3_tiny_gguf: "deepseek3-tiny/deepseek3-tiny-f32.gguf", moe_generates,
            "loads the deepseek3-tiny f32 GGUF (a deepseek2-architecture GGUF with the sigmoid gating function) onto the GPU; needs >= 40Gi available";
        deepseek2_lite: "deepseek2-lite", moe_generates,
            "loads the real ~31GB bf16 DeepSeek-V2-Lite (16B) onto the GPU; needs the 40Gi gate";
        olmoe_1b_7b: "olmoe-1b-7b", moe_generates,
            "loads the real ~14GB bf16 OLMoE-1B-7B onto the GPU; needs the 40Gi gate";
        gptoss_20b_gguf: "gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf", moe_generates,
            "loads the real 20B gpt-oss MXFP4 GGUF; runs on a pod, never locally";
        mixtral_8x7b_gguf: "mixtral-8x7b-instruct-gguf/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", moe_generates,
            "loads the real Mixtral-8x7B Q4_K_M GGUF; runs on a pod, never locally";
        guided_choice_on_qwen2: "qwen2.5-0.5b", guided_choice,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_gbnf_sentence_on_qwen2: "qwen2.5-0.5b", guided_gbnf_sentence,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_gbnf_repetition_on_qwen2: "qwen2.5-0.5b", guided_gbnf_bounded_repetition,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_tool_call_on_qwen2: "qwen2.5-0.5b", guided_forced_tool_call,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_json_const_on_qwen2: "qwen2.5-0.5b", guided_json_const_object,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_json_multi_field_on_qwen2: "qwen2.5-0.5b", guided_json_multi_field,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
        guided_json_object_on_qwen2: "qwen2.5-0.5b", guided_json_object_is_not_degenerate,
            "loads the real qwen2.5-0.5b onto the GPU; needs >= 40Gi available";
    }

    /// Card 545a (A545-3, R2), on the driver's path: a safetensors GPTQ checkpoint's linears load packed
    /// from their checkpoint bytes, mapped as packed GPTQ handles, with no dense copy of any of them in
    /// the store. CPU-only; the packed decode is `gptq_packed_decode`.
    #[test]
    fn a_gptq_checkpoint_loads_its_linears_packed_with_no_dense_copy() {
        let Some(path) =
            poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b-gptq"))
        else {
            return;
        };
        let handle = load(&path);
        let mut packed = 0;
        for (_, view, mapped) in handle.model().weights().iter() {
            let poot_quant::weights::WeightView::Stored(key) = view else {
                continue;
            };
            if !key.as_str().ends_with("_proj.weight") {
                continue;
            }
            match mapped.format {
                poot_quant::weights::HandleFormat::Packed(weight) => assert!(
                    matches!(
                        weight.format(),
                        poot_quant::format::WeightFormat::Gptq { .. }
                    ),
                    "{key}: {:?}",
                    weight.format()
                ),
                dense => panic!("{key} maps to {dense:?}"),
            }
            assert!(
                matches!(
                    handle.store().get(key.as_str()),
                    Some(WeightEntry::Packed(_))
                ),
                "{key} is stored packed"
            );
            packed += 1;
        }
        // 24 layers x 7 projections.
        assert_eq!(packed, 24 * 7);
    }
}

// ---------------------------------------------------------------------------------------------
// The serving surface (Card 735): `open`/`step`/`commit`/`release` over the paged pool, each row
// compared with what `Driver::generate` produces for the same sequence alone (internal driver/batch
// token parity; independent tier-3 provenance is POOT-567). Builtin qwen2 and bloom fixtures.
// ---------------------------------------------------------------------------------------------

/// Room for the longest row: a 13-token prompt and 12 new tokens, with slack.
const SERVING_CAPACITY: usize = 64;
/// Prefill chunk: a 13-token prompt prefills as 4, 4, 4 and 1.
const SERVING_CHUNK: usize = 4;
const LONG: [u32; 13] = [3, 17, 40, 8, 25, 11, 2, 30, 14, 9, 21, 5, 36];
const MEDIUM: [u32; 5] = [7, 22, 31, 12, 44];
const SHORT: [u32; 3] = [19, 6, 28];

fn family_handle(family: &str, eos: u32) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("no builtin family {family}"));
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(eos);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, fixture.store.clone(), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap(),
    )
}

fn serving_driver(choice: BackendChoice, handle: &Arc<ModelHandle>) -> Option<Driver> {
    let executor = open(choice)?;
    Some(
        Driver::new(
            Arc::clone(handle),
            executor,
            options(SERVING_CHUNK, SERVING_CAPACITY),
        )
        .unwrap(),
    )
}

fn pool_shape(blocks: usize, max_seqs: usize) -> PoolShape {
    PoolShape {
        blocks: nz(blocks),
        max_seqs: nz(max_seqs),
    }
}

/// Prepare the serving shapes the rows draw from.
fn prepare_serving(
    driver: &mut Driver,
    pool: PoolShape,
    rows: &[usize],
    heads: &[Head],
    windows: &[usize],
) {
    let rows: Vec<_> = rows.iter().map(|&r| nz(r)).collect();
    let windows: Vec<_> = windows.iter().map(|&w| nz(w)).collect();
    driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool),
            rows: &rows,
            heads,
            warm: Warm::Admitted,
            windows: &windows,
            adapters: &[],
        })
        .unwrap();
}

/// One sequence a row drives to completion through the serving surface.
struct Lane {
    seq: SeqId,
    prompt: Vec<u32>,
    max_new: usize,
    sampler: Sampler,
    prefilled: usize,
    out: Vec<u32>,
}

impl Lane {
    fn open(
        driver: &mut Driver,
        prompt: &[u32],
        max_new: usize,
        sampler: Sampler,
    ) -> (Self, usize) {
        let admission = driver
            .open(SeqRequest {
                prompt,
                max_new,
                adapter: AdapterRef::Base,
            })
            .unwrap();
        let Admission::Opened { seq, reused } = admission else {
            panic!("no room: {admission:?}");
        };
        let lane = Lane {
            seq,
            prompt: prompt.to_vec(),
            max_new,
            sampler,
            prefilled: reused,
            out: Vec::new(),
        };
        (lane, reused)
    }

    fn done(&self) -> bool {
        self.out.len() == self.max_new
    }

    fn work(&self) -> RowWork<'static> {
        if self.prefilled < self.prompt.len() {
            RowWork::Prefill {
                upto: self.prompt.len(),
            }
        } else {
            RowWork::Decode {
                ahead: &[],
                pick: PickPolicy::Greedy,
            }
        }
    }
}

/// One step over `lanes[active]`, committing every token it returns. Returns whether the step mixed a
/// prefilling row with at least two decoding rows.
fn step_lanes(driver: &mut Driver, lanes: &mut [Lane], active: &[usize]) -> bool {
    let mut rows: Vec<StepRow<'_>> = lanes
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| active.contains(i))
        .map(|(_, lane)| StepRow {
            seq: lane.seq,
            work: lane.work(),
            sampler: &mut lane.sampler,
        })
        .collect();
    let prefilling = rows
        .iter()
        .filter(|row| matches!(row.work, RowWork::Prefill { .. }))
        .count();
    let decoding = rows.len() - prefilling;
    let results = driver.step(&mut rows).unwrap();
    drop(rows);
    for result in results {
        let lane = lanes.iter_mut().find(|l| l.seq == result.seq).unwrap();
        lane.prefilled += result.consumed;
        let kept = result.tokens.len();
        lane.out.extend(&result.tokens);
        driver.commit(result.seq, kept, &mut lane.sampler).unwrap();
    }
    prefilling >= 1 && decoding >= 2
}

/// Run every lane to its `max_new` tokens, one step per round over the unfinished ones.
fn run_to_end(driver: &mut Driver, lanes: &mut [Lane]) {
    while lanes.iter().any(|l| !l.done()) {
        let active: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].done()).collect();
        step_lanes(driver, lanes, &active);
    }
}

const SERVING_MAX_NEW: usize = 12;

/// SC-001: one prefill row and two decode rows in one step produce, for every row, the tokens
/// `Driver::generate` produces for that sequence alone. The 13-token prompt prefills in four pieces
/// while the two short sequences decode beside it, so at least one step mixes the three rows.
/// Mutation: swap two rows' block tables (row `i` reads row `i + 1`'s table); the row goes red.
fn mixed_step_matches_generate(choice: BackendChoice, family: &str) {
    let handle = family_handle(family, NO_EOS);
    let Some(mut driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = serving_driver(choice, &handle) else {
        return;
    };
    let prompts: [&[u32]; 3] = [&LONG, &MEDIUM, &SHORT];
    let want: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| generate(&mut reference, p, SERVING_MAX_NEW, Sampler::greedy()))
        .collect();
    prepare_serving(&mut driver, pool_shape(8, 4), &[3], &[Head::GREEDY], &[]);
    let mut lanes: Vec<Lane> = prompts
        .iter()
        .map(|p| Lane::open(&mut driver, p, SERVING_MAX_NEW, Sampler::greedy()).0)
        .collect();
    // The two short sequences prefill and commit their first token, then decode while the long one
    // prefills beside them.
    while lanes[1].prefilled < lanes[1].prompt.len() || lanes[2].prefilled < lanes[2].prompt.len() {
        step_lanes(&mut driver, &mut lanes, &[1, 2]);
    }
    let mut mixed = 0;
    while lanes.iter().any(|l| !l.done()) {
        let active: Vec<usize> = (0..3).filter(|&i| !lanes[i].done()).collect();
        mixed += usize::from(step_lanes(&mut driver, &mut lanes, &active));
    }
    assert!(
        mixed >= 1,
        "a step mixed a prefill row with two decode rows"
    );
    for (i, lane) in lanes.iter().enumerate() {
        assert_eq!(lane.out, want[i], "{family} row {i}");
    }
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
}
per_backend!(
    mixed_step_qwen2,
    mixed_step_qwen2_wgpu,
    mixed_step_qwen2_rocm,
    mixed_step_qwen2_ptx
);
fn mixed_step_qwen2(choice: BackendChoice) {
    mixed_step_matches_generate(choice, "qwen2");
}
per_backend!(
    mixed_step_bloom,
    mixed_step_bloom_wgpu,
    mixed_step_bloom_rocm,
    mixed_step_bloom_ptx
);
fn mixed_step_bloom(choice: BackendChoice) {
    mixed_step_matches_generate(choice, "bloom");
}

/// A prompt of two full blocks and a tail, and one that shares the first two blocks.
fn shared_prompts() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let a: Vec<u32> = (0..35).map(|i| (i * 7 + 3) % 40).collect();
    let mut b = a[..32].to_vec();
    b.extend([41, 42, 43]);
    let c: Vec<u32> = (0..50).map(|i| (i * 5 + 11) % 40).collect();
    (a, b, c)
}

/// SC-002: two requests sharing a prompt prefix reuse cached blocks (the hit count equals the shared
/// block count) and produce the tokens of the unshared run; the reuse survives another request taking
/// every free block. Mutation: drop the refcount increment on a hit; the finished second request frees
/// the shared blocks, the third request overwrites them, and the fourth reuses corrupted blocks, so its
/// tokens differ.
fn prefix_reuse_hits_and_matches_the_unshared_run(choice: BackendChoice) {
    let handle = family_handle("qwen2", NO_EOS);
    let Some(mut driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = serving_driver(choice, &handle) else {
        return;
    };
    let (a, b, c) = shared_prompts();
    let (want_a, want_b) = (
        generate(&mut reference, &a, 4, Sampler::greedy()),
        generate(&mut reference, &b, 4, Sampler::greedy()),
    );
    // Six blocks: A (39 positions) takes three, C (54) takes four of what remains.
    prepare_serving(&mut driver, pool_shape(6, 2), &[1], &[Head::GREEDY], &[]);
    let run = |driver: &mut Driver, prompt: &[u32]| {
        let (lane, reused) = Lane::open(driver, prompt, 4, Sampler::greedy());
        let mut lanes = [lane];
        run_to_end(driver, &mut lanes);
        let outcome = driver.release(lanes[0].seq, Release::Finished).unwrap();
        (lanes[0].out.clone(), reused, outcome)
    };
    let (tokens, reused, outcome) = run(&mut driver, &a);
    assert_eq!((tokens, reused), (want_a.clone(), 0));
    assert_eq!(outcome, PrefixOutcome::Registered { tokens: 32 });
    let (tokens, reused, _) = run(&mut driver, &b);
    assert_eq!(reused, 32, "the hit count equals the two shared blocks");
    assert_eq!(
        tokens, want_b,
        "the shared prefix's tokens equal the unshared run"
    );
    run(&mut driver, &c);
    let (tokens, reused, _) = run(&mut driver, &a);
    assert_eq!(
        reused, 32,
        "the pinned prefix survived the unrelated request"
    );
    assert_eq!(
        tokens, want_a,
        "reuse after another request filled the pool"
    );
}
per_backend!(
    prefix_reuse_hits_and_matches_the_unshared_run,
    prefix_reuse_hits_and_matches_the_unshared_run_wgpu,
    prefix_reuse_hits_and_matches_the_unshared_run_rocm,
    prefix_reuse_hits_and_matches_the_unshared_run_ptx
);

/// SC-004: `commit(seq, 1)` on a two-token result advances the sequence by exactly one token. The
/// window feeds a wrong draft, so the second pick and the K/V written at the draft's position are
/// stale; the next steps must produce `generate`'s tokens anyway, and a correct draft verifies both
/// picks. Mutation: `commit` by the result's length; the stale pick joins the sequence and the
/// following tokens diverge.
fn commit_keeps_fewer_than_returned(choice: BackendChoice) {
    let handle = family_handle("qwen2", NO_EOS);
    let Some(mut driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = serving_driver(choice, &handle) else {
        return;
    };
    let want = generate(&mut reference, &MEDIUM, 8, Sampler::greedy());
    prepare_serving(&mut driver, pool_shape(4, 1), &[1], &[Head::GREEDY], &[2]);
    let (lane, _) = Lane::open(&mut driver, &MEDIUM, 8, Sampler::greedy());
    let mut lanes = [lane];
    while lanes[0].out.is_empty() {
        step_lanes(&mut driver, &mut lanes, &[0]);
    }
    assert_eq!(lanes[0].out, want[..1]);
    let seq = lanes[0].seq;
    let window = |driver: &mut Driver, lane: &mut Lane, ahead: &[u32]| {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Decode {
                ahead,
                pick: PickPolicy::Greedy,
            },
            sampler: &mut lane.sampler,
        }];
        driver.step(&mut rows).unwrap().remove(0).tokens
    };
    let wrong = (want[1] + 1) % 40;
    let picks = window(&mut driver, &mut lanes[0], &[wrong]);
    assert_eq!(picks.len(), 2, "a window returns a pick per fed position");
    assert_eq!(
        picks[0], want[1],
        "the first pick does not depend on the draft"
    );
    let err = driver.commit(seq, 3, &mut lanes[0].sampler).unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::CommitBeyondResult {
                kept: 3,
                produced: 2
            })
        ),
        "{err:?}"
    );
    driver.commit(seq, 1, &mut lanes[0].sampler).unwrap();
    lanes[0].out.push(picks[0]);
    // A correct draft verifies both picks against `generate`'s continuation.
    let picks = window(&mut driver, &mut lanes[0], &[want[2]]);
    assert_eq!(
        picks,
        want[2..4],
        "both picks equal the tokens generate produced"
    );
    driver.commit(seq, 2, &mut lanes[0].sampler).unwrap();
    lanes[0].out.extend(&picks);
    run_to_end(&mut driver, &mut lanes);
    assert_eq!(lanes[0].out, want);
}
per_backend!(
    commit_keeps_fewer_than_returned,
    commit_keeps_fewer_than_returned_wgpu,
    commit_keeps_fewer_than_returned_rocm,
    commit_keeps_fewer_than_returned_ptx
);

/// SC-006: a step mixing one greedy row and one guided row runs the logits entry for the whole step and
/// returns the sampler's token for both rows, matching `generate` for each row alone, and no step
/// compiles. Mutation: drop the logits branch and always run the plain suffix; the guided row's token
/// is the unmasked argmax and diverges.
fn a_guided_row_takes_the_whole_step_to_the_logits_entry(choice: BackendChoice) {
    let handle = family_handle("qwen2", NO_EOS);
    let Some(mut driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = serving_driver(choice, &handle) else {
        return;
    };
    let guided = |sampler: Sampler| {
        let constraint = handle.text().build_regex_constraint(SIXTEEN).unwrap();
        sampler.with_constraint(constraint)
    };
    let want_plain = generate(&mut reference, &MEDIUM, SERVING_MAX_NEW, Sampler::greedy());
    let want_guided = generate(
        &mut reference,
        &SHORT,
        SERVING_MAX_NEW,
        guided(Sampler::greedy()),
    );
    assert!(
        want_guided.iter().all(|&t| t < 16),
        "the mask holds the guided run to a..p: {want_guided:?}"
    );
    assert_ne!(
        want_guided,
        generate(&mut reference, &SHORT, SERVING_MAX_NEW, Sampler::greedy()),
        "the mask changes the guided row's tokens, so a plain suffix cannot match"
    );
    prepare_serving(
        &mut driver,
        pool_shape(8, 2),
        &[2],
        &[Head::GREEDY, Head::Logits],
        &[],
    );
    let compiled = driver.compile_count();
    let host_before = driver.stats().host_picks;
    let (plain, _) = Lane::open(&mut driver, &MEDIUM, SERVING_MAX_NEW, Sampler::greedy());
    let (masked, _) = Lane::open(
        &mut driver,
        &SHORT,
        SERVING_MAX_NEW,
        guided(Sampler::greedy()),
    );
    let mut lanes = [plain, masked];
    // The plain row alone first: the suffix entry; then both: the logits entry.
    while lanes[0].out.len() < 3 {
        step_lanes(&mut driver, &mut lanes, &[0]);
    }
    assert_eq!(
        driver.stats().host_picks,
        host_before,
        "an ordinary row stays on the device suffix"
    );
    run_to_end(&mut driver, &mut lanes);
    assert_eq!(
        lanes[0].out, want_plain,
        "the greedy row beside a guided one"
    );
    assert_eq!(lanes[1].out, want_guided, "the guided row");
    assert!(
        driver.stats().host_picks > host_before,
        "the guided row took the host path"
    );
    assert_eq!(
        driver.compile_count(),
        compiled,
        "changing the row composition between prepared entries compiles nothing"
    );
}
per_backend!(
    a_guided_row_takes_the_whole_step_to_the_logits_entry,
    a_guided_row_takes_the_whole_step_to_the_logits_entry_wgpu,
    a_guided_row_takes_the_whole_step_to_the_logits_entry_rocm,
    a_guided_row_takes_the_whole_step_to_the_logits_entry_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-738 SC-004 (F4): a NaN in one router weight is a typed validation fault naming the witness
// and its layer, through the driver, not a published token.
// ---------------------------------------------------------------------------------------------

/// `family`'s fixture with one BF16 element of the tensor `key` set to NaN.
fn poisoned_router_handle(family: &str, key_to_poison: &str) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap();
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let mut store = WeightStore::builder();
    for (key, weight) in fixture.store.iter() {
        let weight = match weight {
            WeightEntry::Dense(dense) if key.as_str() == key_to_poison => {
                let mut bytes = dense.bytes().as_slice().to_vec();
                bytes[..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
                        .unwrap(),
                )
            }
            other => other.clone(),
        };
        store.insert(key.clone(), weight).unwrap();
    }
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, store.build(), &registry, |_| Ok(fixture_text()))
            .unwrap(),
    )
}

/// SC-004: the poisoned router stops generation with `ExecError::Validation` carrying the
/// `RouterLogitsFinite` witness of layer 1. Mutation: drop the router's witness emission in
/// `components::router`; the NaN is canonicalized by the rank and tokens publish, so the row goes red.
fn a_nan_router_weight_is_a_typed_validation_fault(choice: BackendChoice) {
    a_nan_router_weight_is_a_typed_validation_fault_of(
        choice,
        "qwen3moe",
        "model.layers.1.mlp.gate.weight",
    );
}

fn a_nan_router_weight_is_a_typed_validation_fault_of(
    choice: BackendChoice,
    family: &str,
    router: &str,
) {
    use poot_executor::ExecError;
    use poot_graph_ir::ValidationPacketError;
    use poot_models::components::witness::WitnessKind;
    let handle = poisoned_router_handle(family, router);
    let Some(mut driver) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let error = driver
        .generate(
            GenerateRequest {
                prompt: PROMPT.to_vec(),
                max_new: 4,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .expect_err("a NaN router weight must not publish tokens");
    let DriverError::Device(ExecError::Validation(failure)) = error else {
        panic!("expected a validation fault, got {error:?}");
    };
    let ValidationPacketError::Failure(failure) = *failure else {
        panic!("expected one failing lane, got {failure:?}");
    };
    assert_eq!(
        WitnessKind::decode(failure.id),
        Some((1, WitnessKind::RouterLogitsFinite)),
        "{failure}"
    );
}
per_backend!(
    a_nan_router_weight_is_a_typed_validation_fault,
    a_nan_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_router_weight_is_a_typed_validation_fault_ptx
);

/// POOT-570 SC-008: MiniMax-M2 uses the same component witness: a NaN in layer 1's router (the tensor
/// `block_sparse_moe.gate`) is `RouterLogitsFinite` naming layer 1 through the driver. Mutation: drop the
/// router's witness emission in `components::router`; the NaN publishes tokens and the row goes red.
fn a_nan_minimax_router_weight_is_a_typed_validation_fault(choice: BackendChoice) {
    a_nan_router_weight_is_a_typed_validation_fault_of(
        choice,
        "minimax_m2",
        "model.layers.1.block_sparse_moe.gate.weight",
    );
}
per_backend!(
    a_nan_minimax_router_weight_is_a_typed_validation_fault,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_ptx
);

/// POOT-570 SC-005: three positions of the MiniMax-M2 fixture through the driver against the reference:
/// the greedy tokens after a three-token prompt and the log-probability of each (within 1e-3), which
/// read the key and value caches of every earlier position. Literals are the reference's
/// (`minimax_m2::tests::the_driver_literals_are_the_references_greedy_decode`). Mutation: swap K and V in
/// the cache write (`components::attention`); the first log-probability moves and the row goes red.
fn minimax_three_positions_match_the_reference(choice: BackendChoice) {
    const PROMPT3: [u32; 3] = [3, 17, 40];
    const TOKENS: [u32; 3] = [40, 40, 38];
    const LOGPROBS: [f32; 3] = [-0.006_896_56, -0.145_637_38, -0.826_256_9];
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "minimax_m2")
        .unwrap();
    let handle = recorded_family_handle(recorded);
    let Some(mut driver) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let (tokens, logprobs) = generate_traced(&mut driver, &PROMPT3, 3);
    assert_eq!(tokens, TOKENS, "minimax_m2 on {choice:?}");
    for (step, (got, want)) in logprobs.iter().zip(LOGPROBS).enumerate() {
        assert!(
            (got - want).abs() <= 1e-3,
            "minimax_m2 on {choice:?}, step {step}: log-probability {got}, the reference gives {want}"
        );
    }
}
per_backend!(
    minimax_three_positions_match_the_reference,
    minimax_three_positions_match_the_reference_wgpu,
    minimax_three_positions_match_the_reference_rocm,
    minimax_three_positions_match_the_reference_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-738 SC-001 (ADR-0101 tier 2): the first step of each MoE family on the device equals the CPU
// oracle on the same traced graph.
// ---------------------------------------------------------------------------------------------

/// The oracle's top five `(token, log-probability)` for the last prompt position of `handle`'s model,
/// over its own traced prefill graph and weight map.
fn oracle_top5(handle: &ModelHandle) -> Vec<(u32, f64)> {
    use poot_graph_ir::{Slot, SlotKey};
    use poot_models::model::{KvLayout, LogitRows, Phase, StepShape};
    let n = PROMPT.len();
    let shape = StepShape {
        rows: nz(1),
        tokens: nz(n),
        capacity: nz(CAPACITY),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    };
    let model = handle.model();
    let graph = model.trace(Phase::Prefill, shape).unwrap();
    let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
    let graph = poot_graph_plan::bind_packed_weights(&graph, &formats).unwrap();
    let ids: Vec<i32> = PROMPT.iter().map(|&t| t as i32).collect();
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            poot_tensor::HostTensor::i32(vec![1, n], ids),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            poot_tensor::HostTensor::i32(vec![1, n], (0..n as i32).collect()),
        ),
    ];
    let zeros: Vec<poot_tensor::HostTensor> = graph
        .state
        .iter()
        .map(|&(si, _)| poot_tensor::HostTensor::zeros(graph.aval(si).shape.clone()))
        .collect();
    let (out, _) = poot_test_util::weight_map_oracle::eval_mapped(
        &graph,
        handle.store(),
        model.weights(),
        &slots,
        &zeros,
    );
    let logits: Vec<f64> = out
        .as_f32()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let max = logits.iter().copied().fold(f64::MIN, f64::max);
    let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
    let mut ranked: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(t, l)| (t as u32, l - log_sum))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(5);
    ranked
}

/// SC-001, tier 2: the device's first generated token and its top-five log-probabilities equal the
/// oracle's over the same graph (token ids exact, log-probabilities within `1e-3`). Mutation: swap two
/// experts' router ids in `components::router`; the family's first token or a log-probability
/// moves and the row goes red.
fn moe_first_step_matches_the_oracle(family: &str, choice: BackendChoice) {
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == family)
        .unwrap_or_else(|| panic!("no recorded row for {family}"));
    let handle = recorded_family_handle(recorded);
    let want = oracle_top5(&handle);
    let Some(mut driver) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let generated = driver
        .generate(
            GenerateRequest {
                prompt: recorded.prompt.to_vec(),
                max_new: 1,
                sampler: Sampler::greedy().with_logprobs(5),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap();
    assert_eq!(
        generated.tokens,
        [want[0].0],
        "{family} on {choice:?}: first token"
    );
    let got = &generated.logprobs[0].top;
    assert_eq!(
        got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
        want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
        "{family} on {choice:?}: the five most likely tokens"
    );
    for ((token, got), (_, want)) in got.iter().zip(&want) {
        assert!(
            (f64::from(*got) - want).abs() <= 1e-3,
            "{family} on {choice:?}: token {token} has log-probability {got}, the oracle gives {want}"
        );
    }
}

macro_rules! moe_oracle_rows {
    ($($family:literal => $wgpu:ident, $rocm:ident;)*) => {
        $(
            #[test]
            fn $wgpu() {
                moe_first_step_matches_the_oracle($family, BackendChoice::Wgpu);
            }

            #[cfg(feature = "rocm")]
            #[test]
            fn $rocm() {
                moe_first_step_matches_the_oracle($family, BackendChoice::Rocm);
            }
        )*
    };
}

moe_oracle_rows! {
    "qwen3moe" => qwen3moe_first_step_matches_the_oracle_wgpu, qwen3moe_first_step_matches_the_oracle_rocm;
    "olmoe" => olmoe_first_step_matches_the_oracle_wgpu, olmoe_first_step_matches_the_oracle_rocm;
    "llama/mixtral" => mixtral_first_step_matches_the_oracle_wgpu, mixtral_first_step_matches_the_oracle_rocm;
    "granite/granitemoe" => granitemoe_first_step_matches_the_oracle_wgpu, granitemoe_first_step_matches_the_oracle_rocm;
    "gpt_oss" => gpt_oss_first_step_matches_the_oracle_wgpu, gpt_oss_first_step_matches_the_oracle_rocm;
    "minimax_m2" => minimax_m2_first_step_matches_the_oracle_wgpu, minimax_m2_first_step_matches_the_oracle_rocm;
    "deepseek2" => deepseek2_first_step_matches_the_oracle_wgpu, deepseek2_first_step_matches_the_oracle_rocm;
    "deepseek3" => deepseek3_first_step_matches_the_oracle_wgpu, deepseek3_first_step_matches_the_oracle_rocm;
    "deepseek32" => deepseek32_first_step_matches_the_oracle_wgpu, deepseek32_first_step_matches_the_oracle_rocm;
    "nemotron_h" => nemotron_h_first_step_matches_the_oracle_wgpu, nemotron_h_first_step_matches_the_oracle_rocm;
}
