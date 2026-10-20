//! Driver rows on a real device (Card 734): the qwen2 fixture the registry ships, through
//! `Registry::builtin()` -> `ModelHandle` -> `Driver::generate` on the wgpu and ROCm executors. Each row
//! skips when its device does not open (`POOT_REQUIRE_WGPU=1` / `POOT_REQUIRE_ROCM=1` turn that into a
//! failure); the `_ptx` rows run on a pod (`POOT_REQUIRE_PTX=1`). Device rows run serially under
//! `/tmp/poot-gpu.lock`.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_graph_plan::CompileOptions;
use poot_models::registry::{RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

use crate::GenerationControl;
use crate::core::sampler::Sampler;
use crate::driver::caps::PrepareRefused;
use crate::driver::error::{DriverError, InvalidRequest};
use crate::driver::{
    AdapterRef, Admission, BackendChoice, Driver, DriverOptions, GenerateRequest, Head, Layout,
    ModelHandle, PoolShape, PrefixOutcome, PrefixSkip, PreparedSetLimits, Release, RowWork, SeqId,
    SeqRequest, Serving, ServingShapes, State, StepRow, Warm, open_executor,
};
use poot_test_util::per_backend;
use poot_text::{ChatTemplate, TextCodec};

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
    text_of(VOCAB, ChatTemplate::default())
}

/// [`fixture_text`] over `vocab` words.
fn text_of(vocab: usize, chat_template: ChatTemplate) -> TextCodec {
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
    TextCodec::with_tokenizer(
        tokenizer,
        None,
        FIXTURE_EOS,
        poot_text::ChatFormat::ChatML,
        chat_template,
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

/// The BF16 store with every tensor but the norm scales multiplied by `factor`. The registry's
/// fixtures are small enough that a tiny tower's greedy decode falls into one repeated token whatever
/// its prompt rows are; a larger scale makes the continuation depend on them.
fn scaled(store: &WeightStore, factor: f32) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (key, entry) in store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture is dense")
        };
        let bytes: Vec<u8> = if key.as_str().contains("norm") {
            dense.bytes().as_slice().to_vec()
        } else {
            dense
                .bytes()
                .as_slice()
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&pair| {
                    let value = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                    (((value * factor).to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect()
        };
        let dense =
            DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into()).unwrap();
        builder
            .insert(key.clone(), WeightEntry::Dense(dense))
            .unwrap();
    }
    builder.build()
}

#[derive(Clone, Copy)]
pub(super) enum Weights {
    Random,
    ZeroHead,
    /// Random weights scaled so a greedy continuation depends on its prompt (see [`scaled`]).
    Scaled,
}

pub(super) fn handle(weights: Weights, eos: u32) -> Arc<ModelHandle> {
    handle_with(weights, eos, ChatTemplate::default())
}

/// [`handle`] for a model that ships `chat_template`.
fn handle_with(weights: Weights, eos: u32, chat_template: ChatTemplate) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let fixture = (registry.entries()[0].fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(eos);
    let store = match weights {
        Weights::Random => fixture.store.clone(),
        Weights::ZeroHead => zero_head(&fixture.store),
        Weights::Scaled => scaled(&fixture.store, 4.0),
    };
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, store, &registry, |_| {
            Ok(text_of(VOCAB, chat_template))
        })
        .unwrap(),
    )
}

pub(super) fn options(chunk: usize, capacity: usize) -> DriverOptions {
    let compile = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: crate::driver::program_retention,
        draft: crate::driver::DraftPolicy::Off,
        hidden_window: None,
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
    poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
}

/// A driver nothing has prepared, for rows that encode, prepare their own shapes or time phases.
fn bare_driver(
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    chunk: usize,
    capacity: usize,
) -> Option<Driver> {
    let executor = open(choice)?;
    Some(Driver::new(Arc::clone(handle), executor, options(chunk, capacity)).unwrap())
}

/// A driver serving the contiguous layout with nothing prepared: `generate` prepares what its request
/// needs.
fn driver(
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    chunk: usize,
    capacity: usize,
) -> Option<Driver<Serving>> {
    Some(bare_driver(choice, handle, chunk, capacity)?.serve(Layout::Contiguous))
}

/// A driver with the token-row encode shapes prepared: four rows, every length up to `capacity`.
fn token_encoder(
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    chunk: usize,
    capacity: usize,
) -> Option<Driver> {
    let mut driver = bare_driver(choice, handle, chunk, capacity)?;
    driver
        .prepare_encode(&crate::driver::EncodeShapes {
            rows: nz(4),
            max_len: nz(capacity),
        })
        .unwrap();
    Some(driver)
}

pub(super) fn generate(
    driver: &mut Driver<Serving>,
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
fn generate_traced(
    driver: &mut Driver<Serving>,
    prompt: &[u32],
    max_new: usize,
) -> (Vec<u32>, Vec<f32>) {
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
    packed: &mut Driver<Serving>,
    twin: &mut Driver<Serving>,
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
        preprocessor: None,
    };
    let handle = Arc::new(
        ModelHandle::from_checkpoint(&raw, store.build(), &registry, |_| {
            Ok(text_of(WIDE, ChatTemplate::default()))
        })
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
    BackendChoice,
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
    BackendChoice,
    sampled_differs_from_greedy,
    sampled_differs_from_greedy_wgpu,
    sampled_differs_from_greedy_rocm,
    sampled_differs_from_greedy_ptx
);

/// POOT-575 SC-004 (R477-006 on a hybrid family's path; POOT-572 for Qwen3.5): the `family` fixture
/// with its head zeroed (every logit exactly 0, so the distribution is uniform) answers a seeded
/// temperature request with the sampler's tokens, not the greedy ones: a sampled run equal to the greedy
/// one has probability 48^-24 < 2^-96. Mutation: ignore the sampler in the head (`rule_of` always answers
/// `Greedy`); the draw becomes the greedy run and the row fails.
fn family_sampled_differs_from_greedy(family: &str, choice: BackendChoice) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap();
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    let handle = Arc::new(
        ModelHandle::from_checkpoint(&raw, zero_head(&fixture.store), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap(),
    );
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

/// POOT-578 SC-003 (R488-002): decode takes its position from the step's `Slot::Pos`, so one compiled
/// program serves every position. The `family` fixture decodes at position 3 (a three-token prompt, one
/// new token after the prefill's) and then at positions 3..=7 (the same prompt, five decode steps);
/// `Driver::compile_count()` does not move between the two runs, and the tokens equal `expected`
/// (recorded at this card's base on wgpu, the same on every backend). Mutation: key the compiled
/// entry on the step's start position; the second run compiles at position 4 and the counter moves.
fn family_decode_reuses_one_program_across_positions(
    family: &str,
    choice: BackendChoice,
    expected: [u32; 6],
) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap();
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    let handle = Arc::new(
        ModelHandle::from_checkpoint(&raw, scaled(&fixture.store, 4.0), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap(),
    );
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let prompt = &PROMPT[..3];
    let at_three = generate(&mut driver, prompt, 2, Sampler::greedy());
    let compiled = driver.compile_count();
    let through_seven = generate(&mut driver, prompt, 6, Sampler::greedy());
    assert_eq!(
        driver.compile_count(),
        compiled,
        "{family}: decoding at positions 4..=7 compiled a program"
    );
    assert_eq!(through_seven, expected, "{family}: the decoded tokens");
    assert_eq!(at_three, through_seven[..2], "{family}: position 3's token");
}

fn qwen35_decode_reuses_one_program_across_positions(choice: BackendChoice) {
    family_decode_reuses_one_program_across_positions("qwen35", choice, [14, 28, 25, 3, 11, 6]);
}
per_backend!(
    BackendChoice,
    qwen35_decode_reuses_one_program_across_positions,
    qwen35_decode_reuses_one_program_across_positions_wgpu,
    qwen35_decode_reuses_one_program_across_positions_rocm,
    qwen35_decode_reuses_one_program_across_positions_ptx
);

fn qwen3next_decode_reuses_one_program_across_positions(choice: BackendChoice) {
    family_decode_reuses_one_program_across_positions("qwen3next", choice, [45, 10, 8, 28, 34, 22]);
}
per_backend!(
    BackendChoice,
    qwen3next_decode_reuses_one_program_across_positions,
    qwen3next_decode_reuses_one_program_across_positions_wgpu,
    qwen3next_decode_reuses_one_program_across_positions_rocm,
    qwen3next_decode_reuses_one_program_across_positions_ptx
);

fn qwen3next_sampled_differs_from_greedy(choice: BackendChoice) {
    family_sampled_differs_from_greedy("qwen3next", choice);
}
per_backend!(
    BackendChoice,
    qwen3next_sampled_differs_from_greedy,
    qwen3next_sampled_differs_from_greedy_wgpu,
    qwen3next_sampled_differs_from_greedy_rocm,
    qwen3next_sampled_differs_from_greedy_ptx
);

fn qwen35_sampled_differs_from_greedy(choice: BackendChoice) {
    family_sampled_differs_from_greedy("qwen35", choice);
}
per_backend!(
    BackendChoice,
    qwen35_sampled_differs_from_greedy,
    qwen35_sampled_differs_from_greedy_wgpu,
    qwen35_sampled_differs_from_greedy_rocm,
    qwen35_sampled_differs_from_greedy_ptx
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
    BackendChoice,
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
    BackendChoice,
    fixture_tokens_equal_the_recorded_literals,
    fixture_tokens_equal_the_recorded_literals_wgpu,
    fixture_tokens_equal_the_recorded_literals_rocm,
    fixture_tokens_equal_the_recorded_literals_ptx
);

/// The Qwen2.5-VL text tower over the qwen2 fixture's own tensors.
fn vl_handle() -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let fixture = poot_models::test_support::vlm::qwen_vl_fixture();
    let mut config = fixture.config.clone();
    // Without a `vision_config` the checkpoint serves as its text model alone.
    config.as_object_mut().unwrap().remove("vision_config");
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, fixture.store, &registry, |_| Ok(fixture_text()))
            .unwrap(),
    )
}

/// A text-only Qwen2.5-VL prompt generates the plain qwen2 reference's tokens: the driver binds
/// `Slot::MropePosition` with every token at its text position on the temporal, height and width
/// axes, which is plain RoPE, so the tower over the same tensors runs the recorded literals of
/// `fixture_tokens_equal_the_recorded_literals` (whole and chunked prefill, suffix and host path).
/// Mutation: bind `Slot::MropePosition` as zeros (`StepData::text_mrope`); every position rotates
/// as position 0 and the tokens diverge from the literal.
fn qwen25_vl_text_only_tokens_equal_the_plain_reference(choice: BackendChoice) {
    let handle = vl_handle();
    let Some(mut whole) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let Some(mut chunked) = driver(choice, &handle, 5, CAPACITY) else {
        return;
    };
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
    BackendChoice,
    qwen25_vl_text_only_tokens_equal_the_plain_reference,
    qwen25_vl_text_only_tokens_equal_the_plain_reference_wgpu,
    qwen25_vl_text_only_tokens_equal_the_plain_reference_rocm,
    qwen25_vl_text_only_tokens_equal_the_plain_reference_ptx
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
    BackendChoice,
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
    let uploads = |driver: &Driver<Serving>| {
        driver
            .executor_stats()
            .memory
            .iter()
            .find(|(role, _)| *role == poot_runtime_common::BufferRole::Weight)
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
    BackendChoice,
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
    BackendChoice,
    greedy_ties_take_the_first_index,
    greedy_ties_take_the_first_index_wgpu,
    greedy_ties_take_the_first_index_rocm,
    greedy_ties_take_the_first_index_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-737 SC-001: every dense family's builtin fixture generates through the driver with the tokens
// the pre-driver engine generated for it before the dense families moved to the driver (recorded on the
// base commit `2ef7ea834` with its cached device generate on wgpu and ROCm, and its CPU generate:
// all three agreed, and the driver agreed with them). Literals, never recomputed (ADR-0101, tier 3).
// ---------------------------------------------------------------------------------------------

/// What the pre-driver engine was fed for a family's fixture: its prompt, and whether its head is tied. The
/// pre-driver engine always tied BLOOM's and MPT's head to the embedding, so their rows drop the fixture's
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
    // The five MoE rows below (POOT-738) were recorded with the pre-driver engine's CPU generate on each family's
    // fixture at the base commit. All 24 tokens agreed: the driver on wgpu and on ROCm produce exactly the
    // recorded ids, in one whole-prompt prefill and in chunks of four, so no row has a divergence index or
    // a tier-3 top-two margin to record. (A token flip on a tiny random fixture would be judged by the
    // reference's top-two logit margin, ADR-0101.) The first-token figures of the router-id-swap mutation
    // (qwen3moe 10 vs 28, Mixtral 38 vs 40, granitemoe 43 vs 40, gpt-oss 6 vs 11, olmoe 11 vs 11 with a
    // later split) are the red outcome of that mutation, not disagreements with the pre-driver engine.
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
    // minimax_m2 (POOT-570) has no pre-driver recording: its tokens are the independent f64 reference's greedy
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
    // The four group-B rows below (POOT-740) were recorded with the pre-driver engine's CPU generate on each
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
    // deepseek4 (POOT-568) has no recording from the deleted `Runner`: its tokens are the independent f64 reference's
    // greedy decode (`deepseek4::tests::the_driver_literals_are_the_references_greedy_decode`,
    // smallest top-two logit margin 1.5e-2). The 7-token prompt is not a multiple of the family's
    // prefill granule (4), so each run is a prefill of four tokens and three decode steps.
    RecordedFamily {
        family: "deepseek4",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            28, 31, 6, 41, 20, 26, 15, 17, 9, 32, 33, 13, 40, 26, 9, 41, 42, 30, 11, 15, 32, 32,
            31, 7,
        ],
    },
    // glm53_flash (POOT-569) has no recording from the deleted `Runner`: its tokens are the independent f64 reference's
    // greedy decode (`glm53_flash::tests::the_driver_literals_are_the_references_greedy_decode`,
    // smallest top-two logit margin 5.7e-3). The family has no prefill granule, so the 7-token prompt
    // runs as prefill pieces (8 or 4 tokens per chunk) and decode steps for no remainder.
    RecordedFamily {
        family: "glm53_flash",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            29, 46, 25, 32, 34, 21, 17, 7, 2, 31, 37, 12, 30, 0, 4, 8, 4, 24, 21, 43, 9, 31, 24, 21,
        ],
    },
    // qwen3next (POOT-575) has no recording from the deleted `Runner`: its tokens are the CPU oracle's greedy decode over the
    // fixture (`qwen3next::tests::the_fixture_generates_the_recorded_tokens`; the GDN, gated attention and
    // gated shared expert are pinned against f64 references in their own modules, tier 3).
    RecordedFamily {
        family: "qwen3next",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            20, 2, 20, 3, 6, 27, 25, 17, 18, 24, 20, 18, 45, 34, 45, 5, 45, 12, 35, 40, 5, 46, 35,
            29,
        ],
    },
    // The three Qwen4Exp rows (POOT-571) are the independent reference's greedy tokens
    // (`qwen38::tests::the_fixtures_generate_the_recorded_tokens`: the plain-loop f64 reference over the
    // checkpoint bytes, re-run over the whole sequence per token), which the CPU oracle's token-by-token
    // decode reproduces. The 7-token prompt is not a whole number of the family's blocks (two for the
    // fixtures, four for the finetune-sized one), so each driver run is whole-granule prefill plus decode
    // steps and equals the token-by-token run (SC-008).
    RecordedFamily {
        family: "qwen4exp",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            38, 45, 2, 44, 16, 27, 35, 25, 34, 10, 28, 16, 14, 10, 15, 35, 36, 3, 38, 30, 4, 39,
            18, 21,
        ],
    },
    RecordedFamily {
        family: "qwen4exp/packed",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            22, 22, 46, 11, 6, 18, 7, 22, 26, 28, 45, 9, 11, 13, 30, 33, 28, 30, 12, 31, 11, 5, 11,
            11,
        ],
    },
    RecordedFamily {
        family: "qwen4exp/finetune",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            12, 6, 34, 37, 30, 7, 38, 40, 18, 38, 7, 40, 7, 6, 34, 21, 38, 46, 27, 7, 43, 31, 28,
            28,
        ],
    },
    // qwen35 (POOT-572) has no recording from the deleted `Runner`: its tokens are the CPU oracle's greedy decode over the
    // fixture (`qwen38_27b::tests::the_fixture_generates_the_recorded_tokens`; its logits and every
    // carried state element are pinned against the independent plain-loop reference in the same module,
    // tier 2).
    RecordedFamily {
        family: "qwen35",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            4, 11, 45, 13, 11, 14, 33, 6, 28, 2, 39, 25, 1, 2, 35, 35, 29, 37, 26, 40, 23, 10, 27,
            24,
        ],
    },
    // The same family stored as `Qwen/Qwen3.8-27B-FP8` stores it: every projection a packed block-scaled
    // E4M3 payload. The tokens are the CPU oracle's over the packed store
    // (`qwen38_27b::tests::the_fp8_fixture_generates_the_recorded_tokens`).
    RecordedFamily {
        family: "qwen35/fp8",
        prompt: &PROMPT,
        tied_head: false,
        tokens: [
            31, 6, 9, 22, 7, 12, 6, 6, 30, 9, 12, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        ],
    },
];

/// `family`'s builtin fixture as the pre-driver engine was fed it (see [`RecordedFamily`]), with no reachable
/// end-of-sequence id.
fn recorded_family_handle(recorded: &RecordedFamily) -> Arc<ModelHandle> {
    recorded_family_handle_with(recorded, &|key, weight| {
        vec![(key.to_string(), weight.clone())]
    })
}

/// How a checkpoint stores a tensor: the named tensors that replace the one the fixture holds.
type Restore = dyn Fn(&str, &WeightEntry) -> Vec<(String, WeightEntry)>;

/// [`recorded_family_handle`] with each stored tensor replaced by the tensors `restore` returns for it
/// (a checkpoint that stores a weight another way).
fn recorded_family_handle_with(recorded: &RecordedFamily, restore: &Restore) -> Arc<ModelHandle> {
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
            for (key, weight) in restore(key.as_str(), weight) {
                store.insert(key, weight).unwrap();
            }
        }
    }
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
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
    "deepseek4" => recorded_deepseek4_tokens_wgpu, recorded_deepseek4_tokens_rocm, recorded_deepseek4_tokens_ptx;
    "qwen3next" => recorded_qwen3next_tokens_wgpu, recorded_qwen3next_tokens_rocm, recorded_qwen3next_tokens_ptx;
    "qwen4exp" => recorded_qwen4exp_tokens_wgpu, recorded_qwen4exp_tokens_rocm, recorded_qwen4exp_tokens_ptx;
    "qwen4exp/packed" => recorded_qwen4exp_packed_tokens_wgpu, recorded_qwen4exp_packed_tokens_rocm, recorded_qwen4exp_packed_tokens_ptx;
    "qwen4exp/finetune" => recorded_qwen4exp_finetune_tokens_wgpu, recorded_qwen4exp_finetune_tokens_rocm, recorded_qwen4exp_finetune_tokens_ptx;
    "qwen35" => recorded_qwen35_tokens_wgpu, recorded_qwen35_tokens_rocm, recorded_qwen35_tokens_ptx;
    "qwen35/fp8" => recorded_qwen35_fp8_tokens_wgpu, recorded_qwen35_fp8_tokens_rocm, recorded_qwen35_fp8_tokens_ptx;
    "glm53_flash" => recorded_glm53_flash_tokens_wgpu, recorded_glm53_flash_tokens_rocm, recorded_glm53_flash_tokens_ptx;
}

// ---------------------------------------------------------------------------------------------
// POOT-746: routed experts paged through a pool smaller than the table.
// ---------------------------------------------------------------------------------------------

/// The Mixtral fixture has four experts and routes two per token: a pool of three rows is smaller than the
/// table and serves one token per step.
const MIXTRAL_POOL_ROWS: usize = 3;

/// The Mixtral fixture's handle, for the capacity rows.
pub(super) fn mixtral_handle() -> Arc<ModelHandle> {
    recorded_family_handle(mixtral_recorded())
}

fn mixtral_recorded() -> &'static RecordedFamily {
    RECORDED
        .iter()
        .find(|recorded| recorded.family == "llama/mixtral")
        .expect("recorded Mixtral literals")
}

/// SC-002: Mixtral decodes its recorded 24 tokens with `P = 3 < E = 4`, evicting (evictions above zero),
/// every stage of every entry recording once and replaying every step. Mutation: in `ExpertResidency`,
/// skip the refill of a miss (answer no rows); the pool serves another expert's bytes and the tokens
/// diverge. Mutation: in the engine's step, replay the next stage before applying the sync writes; the
/// stage reads the previous step's rows and the tokens diverge.
fn pooled_mixtral_decodes_the_recorded_tokens_with_evictions(choice: BackendChoice) {
    let recorded = mixtral_recorded();
    let handle = recorded_family_handle(recorded);
    let Some(executor) = open(choice) else {
        return;
    };
    let mut driver = Driver::new(handle, executor, options(8, CAPACITY))
        .unwrap()
        .with_expert_pool(nz(MIXTRAL_POOL_ROWS))
        .unwrap()
        .serve(Layout::Contiguous);
    let got = generate(&mut driver, recorded.prompt, 24, Sampler::greedy());
    assert_eq!(got, recorded.tokens, "pooled Mixtral on {choice:?}");
    let counters = driver.stats().residency;
    eprintln!("residency counters: {counters:?}");
    assert!(
        counters.refills > 0 && counters.evictions > 0,
        "{counters:?}"
    );
    let stats = driver.executor_stats();
    assert!(stats.syncs > 0 && stats.sync_read_bytes > 0, "{stats:?}");
    assert!(
        stats.stages.iter().all(|stage| stage.recordings == 1),
        "every stage records once: {:?}",
        stats.stages
    );
}
per_backend!(
    BackendChoice,
    pooled_mixtral_decodes_the_recorded_tokens_with_evictions,
    pooled_mixtral_decodes_the_recorded_tokens_with_evictions_wgpu,
    pooled_mixtral_decodes_the_recorded_tokens_with_evictions_rocm,
    pooled_mixtral_decodes_the_recorded_tokens_with_evictions_ptx
);

/// SC-003 and SC-005 (driver half): a 20-token prompt runs as one-token chunks through the pool, and
/// prompts of different lengths interleaved through the same pooled sites, with a complete reset between,
/// each equal the unpooled driver's tokens on the same weights. The reset clears the manager with the
/// device pool: its counters read zero and the next run still matches. Mutation: in
/// `Driver::reset_exe` reset the executable but keep the manager; the run after the reset trusts rows the
/// device no longer holds and the tokens diverge.
fn pooled_mixtral_interleaves_chunks_prompts_and_resets(choice: BackendChoice) {
    let recorded = mixtral_recorded();
    let handle = recorded_family_handle(recorded);
    // Both executors open before either driver runs (the ROCm runtime does not reopen one).
    let Some(plain_executor) = open(choice) else {
        return;
    };
    let Some(pooled_executor) = open(choice) else {
        return;
    };
    let mut plain = Driver::new(Arc::clone(&handle), plain_executor, options(8, CAPACITY))
        .unwrap()
        .serve(Layout::Contiguous);
    let mut pooled = Driver::new(handle, pooled_executor, options(8, CAPACITY))
        .unwrap()
        .with_expert_pool(nz(MIXTRAL_POOL_ROWS))
        .unwrap()
        .serve(Layout::Contiguous);
    let long: Vec<u32> = (0..20).map(|i| PROMPT[i % PROMPT.len()]).collect();
    let short: Vec<u32> = PROMPT.to_vec();
    let round =
        |label: &str, prompt: &[u32], pooled: &mut Driver<Serving>, plain: &mut Driver<Serving>| {
            let want = generate(plain, prompt, 6, Sampler::greedy());
            let got = generate(pooled, prompt, 6, Sampler::greedy());
            assert_eq!(got, want, "{label} on {choice:?}");
        };
    round("long prompt", &long, &mut pooled, &mut plain);
    round("short prompt", &short, &mut pooled, &mut plain);
    round("long prompt again", &long, &mut pooled, &mut plain);
    pooled.reset_executables().unwrap();
    assert_eq!(
        pooled.stats().residency,
        crate::driver::ResidencyCounters::default(),
        "the reset cleared the manager's counters with the device pool"
    );
    round(
        "short prompt after a reset",
        &short,
        &mut pooled,
        &mut plain,
    );
    assert!(pooled.stats().residency.refills > 0);
}
per_backend!(
    BackendChoice,
    pooled_mixtral_interleaves_chunks_prompts_and_resets,
    pooled_mixtral_interleaves_chunks_prompts_and_resets_wgpu,
    pooled_mixtral_interleaves_chunks_prompts_and_resets_rocm,
    pooled_mixtral_interleaves_chunks_prompts_and_resets_ptx
);

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
fn prepare_admitted(driver: Driver) -> Result<Driver<Serving>, Box<PrepareRefused>> {
    driver.prepare(&ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    })
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
    let refused = prepare_admitted(oversized_chunk_driver(executor, "granite")).unwrap_err();
    assert_refused_by_the_trace(&refused.error, "granite");
    let driver = refused.driver;
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
    BackendChoice,
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
    let drivers = [
        ("granite", oversized_chunk_driver(first, "granite")),
        ("bloom", oversized_chunk_driver(second, "bloom")),
    ];
    for (family, driver) in drivers {
        let refused = prepare_admitted(driver).unwrap_err();
        assert_refused_by_the_trace(&refused.error, family);
    }
}
per_backend!(
    BackendChoice,
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
        .tokenizer()
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
                dtype: match dense.dtype() {
                    poot_tensor::DType::BF16 => "BF16",
                    poot_tensor::DType::F32 => "F32",
                    poot_tensor::DType::I64 => "I64",
                    other => panic!("the fixture stores {other:?}"),
                },
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

/// A Qwen2.5-VL checkpoint whose `config.json` cannot be read or parsed is refused as a load error
/// naming the file's path (the config is the first thing `load` reads, whatever the family).
/// Mutation: report `generation_config.json` as the path in `read_json`'s parse error; the parse row
/// goes red.
#[test]
fn an_unreadable_or_malformed_config_is_attributed_to_its_path() {
    use crate::driver::error::LoadError;
    let registry = Registry::builtin().unwrap();
    let load = |dir: &std::path::Path| {
        ModelHandle::load(dir, &registry, &crate::driver::LoadOptions::default())
    };
    let dir = poot_test_util::unique_temp_path("poot_card1077_config_path");
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.json");
    match load(&dir) {
        Err(DriverError::Load(LoadError::Io { path, .. })) => assert_eq!(path, config),
        other => panic!("expected the missing config named by path, got {other:?}"),
    }
    std::fs::write(&config, r#"{"model_type": "tiny", "hidden_size": "#).unwrap();
    match load(&dir) {
        Err(DriverError::Load(LoadError::Json { path, .. })) => assert_eq!(path, config),
        other => panic!("expected the malformed config named by path, got {other:?}"),
    }
}

/// A `config.json` that repeats a key is refused on the real load path, naming the key and the file:
/// `serde_json::Value` alone keeps the last of a repeated key, so a second `rope_scaling`,
/// `rope_parameters` or `mrope_section` would silently replace the first. Mutation: parse the config
/// with `serde_json::from_slice` in `read_json`; each case is then not refused as a duplicate and the
/// row goes red.
#[test]
fn a_config_repeating_a_rope_key_is_refused_by_model_handle_load() {
    use crate::driver::error::LoadError;
    let registry = Registry::builtin().unwrap();
    let dir = poot_test_util::unique_temp_path("poot_card1077_duplicate_key");
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.json");
    for (key, text) in [
        (
            "rope_scaling",
            r#"{"model_type": "tiny", "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 3, 4]}}"#,
        ),
        (
            "rope_parameters",
            r#"{"model_type": "tiny", "rope_parameters": {"rope_type": "mrope", "mrope_section": [2, 3, 3]},
                "rope_parameters": {"rope_type": "mrope", "mrope_section": [1, 3, 4]}}"#,
        ),
        (
            "mrope_section",
            r#"{"model_type": "tiny",
                "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3], "mrope_section": [1, 3, 4]}}"#,
        ),
    ] {
        std::fs::write(&config, text).unwrap();
        let loaded = ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default());
        match loaded {
            Err(DriverError::Load(LoadError::Json { path, source })) => {
                assert_eq!(path, config);
                assert!(
                    source
                        .to_string()
                        .contains(&format!("duplicate JSON key \"{key}\"")),
                    "{key}: {source}"
                );
            }
            other => panic!("expected the repeated {key} refused as a JSON error, got {other:?}"),
        }
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

/// POOT-572 SC-002: a Qwen3.5 checkpoint directory whose config the old pinned counts refused (two
/// layers, other widths and head counts, no manifest anywhere) loads through `ModelHandle::load`,
/// resolves to the `qwen35` family and keeps the config's own counts. Its numerics are the in-crate
/// row's (`qwen38_27b::tests::a_finetune_shaped_config_builds_and_matches_its_reference`). Mutation:
/// reinstate a pinned layer count in `Qwen35Params::from_raw`; the load is refused and the row goes red.
#[test]
fn a_finetune_like_qwen35_checkpoint_loads_through_model_handle_load() {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen35")
        .unwrap();
    let (_, finetune) = entry
        .variants
        .iter()
        .find(|(name, _)| *name == "finetune")
        .expect("qwen35 ships a finetune-like fixture");
    let fixture = finetune();
    assert_eq!(fixture.config["text_config"]["num_hidden_layers"], 2);
    let dir = fixture_dir("qwen35_finetune", &fixture);
    let handle =
        ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default()).unwrap();
    assert_eq!(handle.config().family, entry.family);
    assert_eq!(handle.config().vocab, VOCAB);
}

/// POOT-572 SC-007, the loading half on a fixture: the Qwen3.5 fixture written the way
/// `Qwen/Qwen3.8-27B-FP8` stores it (E4M3 `weight` and a BF16 `weight_scale_inv` per 128x128 block for
/// every text projection, a `quantization_config` naming `quant_method: "fp8"`, the rest BF16) loads
/// through `ModelHandle::load` with its projections packed as stored: the payload bytes equal the
/// written ones, none decoded. Mutation: drop the `"fp8"` arm of `QuantConfig::scheme`; the load is
/// refused as an unknown `quant_method` and the row goes red.
#[test]
fn a_block_fp8_qwen35_checkpoint_loads_packed_through_model_handle_load() {
    use poot_quant::format::{ScaleEncoding, WeightFormat};
    use poot_quant::weights::{HandleFormat, WeightEntry};
    use poot_quant::{OperandRole, SourceRole};
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen35")
        .unwrap();
    let (_, fp8) = entry
        .variants
        .iter()
        .find(|(name, _)| *name == "fp8")
        .expect("qwen35 ships an FP8 fixture");
    let fixture = fp8();
    let mut config = fixture.config.clone();
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    });
    let mut tensors = Vec::new();
    for (key, weight) in fixture.store.iter() {
        match weight {
            WeightEntry::Dense(dense) => tensors.push(ArchiveTensor {
                name: key.as_str().to_string(),
                dtype: "BF16",
                shape: dense.shape().to_vec(),
                bytes: dense.bytes().as_slice().to_vec(),
            }),
            WeightEntry::Packed(payload) => {
                let [rows, k] = payload.weight().shape();
                let stem = key.as_str().strip_suffix(".weight").unwrap();
                tensors.push(ArchiveTensor {
                    name: key.as_str().to_string(),
                    dtype: "F8_E4M3",
                    shape: vec![rows, k],
                    bytes: payload
                        .bytes(SourceRole::Planar(OperandRole::Codes))
                        .to_vec(),
                });
                tensors.push(ArchiveTensor {
                    name: format!("{stem}.weight_scale_inv"),
                    dtype: "BF16",
                    shape: vec![rows.div_ceil(128), k.div_ceil(128)],
                    bytes: payload
                        .bytes(SourceRole::Planar(OperandRole::Scale))
                        .to_vec(),
                });
            }
        }
    }
    let dir = write_hf_checkpoint("qwen35_fp8", &config, &tensors);
    let handle =
        ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default()).unwrap();
    assert_eq!(handle.config().family, entry.family);
    let mut packed = 0;
    for (key, want) in fixture.store.iter() {
        let WeightEntry::Packed(want) = want else {
            continue;
        };
        let Some(WeightEntry::Packed(got)) = handle.store().get(key.as_str()) else {
            panic!("{key} did not load packed");
        };
        assert_eq!(
            got.weight().format(),
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16
            }
        );
        for role in [OperandRole::Codes, OperandRole::Scale] {
            assert_eq!(
                got.bytes(SourceRole::Planar(role)),
                want.bytes(SourceRole::Planar(role)),
                "{key}: {role:?} is stored as written"
            );
        }
        packed += 1;
    }
    // Three GDN layers' qkv, z and out, the attention layer's four projections, every layer's MLP.
    assert_eq!(packed, 3 * 3 + 4 + 4 * 3);
    let handles = handle
        .model()
        .weights()
        .iter()
        .filter(|(_, _, h)| matches!(h.format, HandleFormat::Packed(_)))
        .count();
    assert_eq!(handles, packed, "every packed projection is mapped packed");
}

/// Gemma is trained with a mandatory BOS: an HF gemma3 checkpoint's prompts encode with its BOS id
/// first (the fixture config names 1), as the pre-driver loader did; a family without one (qwen2) encodes
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
            let payload = poot_quant::test_support::packed::random_payload(
                format,
                [out, k],
                (stored.len() + 1) as u64,
            );
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
    let mut fp8_driver = Driver::new(Arc::clone(&fp8), first, options(8, CAPACITY))
        .unwrap()
        .serve(Layout::Contiguous);
    let mut dense_driver = Driver::new(
        Arc::clone(&dense),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap()
    .serve(Layout::Contiguous);
    let mut granite_driver = Driver::new(
        Arc::clone(&granite),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap()
    .serve(Layout::Contiguous);
    let mut granite_twin_driver = Driver::new(
        Arc::clone(&granite_twin),
        open(choice).unwrap(),
        options(8, CAPACITY),
    )
    .unwrap()
    .serve(Layout::Contiguous);
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
    BackendChoice,
    packed_checkpoints_load_as_stored_through_the_registry,
    packed_checkpoints_load_as_stored_through_the_registry_wgpu,
    packed_checkpoints_load_as_stored_through_the_registry_rocm,
    packed_checkpoints_load_as_stored_through_the_registry_ptx
);

/// Card 1018 (the pre-driver engine's packed-GGUF prefill and wgpu compiled-decode rows, each against its
/// dense twin): a qwen2 and a llama GGUF whose
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
    let mut drivers = drivers.into_iter();
    for case in &cases {
        let (packed, twin, paged) = (
            drivers.next().unwrap(),
            drivers.next().unwrap(),
            drivers.next().unwrap(),
        );
        let (mut packed, mut twin) = (
            packed.serve(Layout::Contiguous),
            twin.serve(Layout::Contiguous),
        );
        assert_same_trace(case.arch, &mut packed, &mut twin, &long, 16);
        let want = generate(&mut twin, &long, 16, Sampler::greedy());

        // Two sequences through the paged pool: the batched decode graph.
        let want_short = generate(&mut twin, &short, 12, Sampler::greedy());
        let mut paged = prepare_serving(paged, pool_shape(8, 4), &[2], &[Head::GREEDY], &[]);
        let mut lanes = vec![
            Lane::open(&mut paged, &long, 16, Sampler::greedy()).0,
            Lane::open(&mut paged, &short, 12, Sampler::greedy()).0,
        ];
        run_to_end(&mut paged, &mut lanes);
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
    BackendChoice,
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
    BackendChoice,
    packed_q5_0_gguf_generates_like_its_dense_twin,
    packed_q5_0_gguf_generates_like_its_dense_twin_wgpu,
    packed_q5_0_gguf_generates_like_its_dense_twin_rocm,
    packed_q5_0_gguf_generates_like_its_dense_twin_ptx
);

/// Real checkpoints through `Registry::builtin()` -> `ModelHandle::load` -> `Driver::generate`, against
/// the tokens the pre-driver engine generated for them before this card (recorded on the base commit
/// `2ef7ea834` with its prefilled device generate, identical on wgpu and ROCm; the driver equalled them there), and
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
        let mut drivers = vec![
            Driver::new(Arc::clone(&handle), first, options(chunks[0], 64))
                .unwrap()
                .serve(Layout::Contiguous),
        ];
        drivers.push(
            Driver::new(
                Arc::clone(&handle),
                open(choice).unwrap(),
                options(chunks[1], 64),
            )
            .unwrap()
            .serve(Layout::Contiguous),
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
        let mut driver = Driver::new(Arc::clone(&handle), executor, options(8, 64))
            .unwrap()
            .serve(Layout::Contiguous);
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
        let mut drivers = vec![
            Driver::new(Arc::clone(&handle), first, options(8, exact))
                .unwrap()
                .serve(Layout::Contiguous),
        ];
        drivers.push(
            Driver::new(
                Arc::clone(&handle),
                open(choice).unwrap(),
                options(8, 10 * exact),
            )
            .unwrap()
            .serve(Layout::Contiguous),
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
        let draw = |driver: &mut Driver<Serving>, seed: u64| {
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
        let mut driver = Driver::new(Arc::clone(&handle), executor, options(8, 64))
            .unwrap()
            .serve(Layout::Contiguous);
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
    fn open_driver(handle: &Arc<ModelHandle>, choice: BackendChoice) -> Option<Driver<Serving>> {
        let executor = open(choice)?;
        Some(
            Driver::new(Arc::clone(handle), executor, options(8, 128))
                .unwrap()
                .serve(Layout::Contiguous),
        )
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

    /// The old pre-driver coherence rows: the greedy continuation of `prompt` names `needle`, which the
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

    /// A secret word stated once at the start of ~5800 tokens of filler (480 sentences of about 12 tokens each), asked for at the end. The
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
        let filler: String = (0..480)
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
    /// pre-driver engine compared against its CPU eager path; the driver has no CPU path, so the independent claim
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
        build: impl FnOnce(&TextCodec) -> poot_text::Result<poot_text::Constraint>,
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
                text.forced_tool_constraint(&forced)
                    .map(|constraint| constraint.expect("a ChatML model forces a call"))
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

    /// The old pre-driver MoE coherence rows (qwen3-moe, mixtral, olmoe, gpt-oss, granitemoe tiny checkpoints):
    /// the checkpoint loads through the registry and decodes through the driver, a prefill split into
    /// chunks of four generates the tokens a whole-prompt prefill does, and the continuation is not empty.
    /// The pre-driver engine compared the device with its CPU re-prefill; the driver has no CPU path, so the
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

    /// What the device half of a real-checkpoint receipt produced.
    struct DeviceReceipt {
        handle: Arc<ModelHandle>,
        ids: Vec<u32>,
        capacity: usize,
        generated: crate::driver::Generation,
    }

    /// The device half of POOT-572 SC-007 (M5's one real-dimensions receipt): the checkpoint loads through
    /// the registry, generates through the driver, and its first token is the documented answer to the
    /// prompt, " Paris". A wrong weight-name mapping shared by the device and the oracle (they read one
    /// name table) cannot pass it. Prints the `RECEIPT` lines a run records. `None` when the backend is
    /// absent. The driver is dropped before returning, so a checkpoint that nearly fills the machine is
    /// never held on the device and on the CPU at once.
    fn generates_paris(path: &std::path::Path, choice: BackendChoice) -> Option<DeviceReceipt> {
        let handle = load(path);
        let ids = handle.text().encode(CAPITAL).unwrap();
        let capacity = 64;
        let executor = open(choice)?;
        let mut limits = options(8, capacity);
        limits.prepared.max_retained_bytes = NonZeroU64::new(1 << 38).unwrap();
        let mut driver = Driver::new(Arc::clone(&handle), executor, limits)
            .unwrap()
            .serve(Layout::Contiguous);
        let generated = driver
            .generate(
                GenerateRequest {
                    prompt: ids.clone(),
                    max_new: 8,
                    sampler: Sampler::greedy().with_logprobs(5),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .unwrap();
        // The device's weights are not needed again; the oracle below reads the host store. Releasing them
        // keeps the two residencies from adding up on a machine whose device memory is its host memory.
        drop(driver);
        let text = handle.text().decode(&generated.tokens).unwrap();
        eprintln!("RECEIPT prompt {CAPITAL:?} ids {ids:?}");
        eprintln!("RECEIPT generated {:?} text {text:?}", generated.tokens);
        eprintln!(
            "RECEIPT device first-token top5 {:?}",
            generated.logprobs[0].top
        );
        let first = handle.text().decode(&generated.tokens[..1]).unwrap();
        assert_eq!(
            first, " Paris",
            "the documented answer; the text was {text:?}"
        );
        Some(DeviceReceipt {
            handle,
            ids,
            capacity,
            generated,
        })
    }

    /// The process's peak resident memory, as a `RECEIPT` line.
    fn print_peak_host_memory() {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines().filter(|l| l.starts_with("VmHWM")) {
            eprintln!("RECEIPT peak host memory ({line})");
        }
    }

    /// POOT-572 SC-007, the oracle half on top of [`generates_paris`]: the first token's top-five
    /// log-probabilities equal the CPU oracle's over the same traced graph and weight map (token ids
    /// exact, log-probabilities within `1e-3`, ADR-0101 tier 2). Runs the oracle locally, so it is for
    /// checkpoints whose oracle fits the host; the 27B FP8 checkpoint's oracle needs more than 64 GB and
    /// is compared against a recorded fixture instead (POOT-1184).
    fn generates_and_matches_the_oracle(path: &std::path::Path, choice: BackendChoice) -> Vec<u32> {
        let Some(DeviceReceipt {
            handle,
            ids,
            capacity,
            generated,
        }) = generates_paris(path, choice)
        else {
            return Vec::new();
        };
        let got = &generated.logprobs[0].top;
        let want = oracle_top5(&handle, &ids, capacity);
        eprintln!("RECEIPT oracle first-token top5 {want:?}");
        assert_eq!(generated.tokens[0], want[0].0, "first token");
        assert_eq!(
            got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            "the five most likely tokens"
        );
        for ((token, got), (_, want)) in got.iter().zip(&want) {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-3,
                "token {token} has log-probability {got}, the oracle gives {want}"
            );
        }
        print_peak_host_memory();
        generated.tokens
    }

    /// POOT-135: the SmolLM3-3B GGUF's eight-token greedy continuation equals the one the safetensors
    /// checkpoint produces on every backend (" Paris. It is the largest city in", updates 0620, 0646 and
    /// 0655), on top of the oracle comparison. The oracle reads the same weight map as the device, so a
    /// GGUF layout error they share (the Q/K row order, the NoPE layer schedule) moves both and only this
    /// independent literal catches it. Mutation: in `Smollm3Params::from_raw`, skip
    /// `with_interleaved_rope` for a GGUF; the continuation becomes " Paris, which is the capital of France".
    fn smollm3_gguf_continues_like_the_safetensors_checkpoint(
        path: &std::path::Path,
        choice: BackendChoice,
    ) {
        const SAFETENSORS_CONTINUATION: [u32; 8] = [12366, 13, 1102, 374, 279, 7928, 3363, 304];
        let tokens = generates_and_matches_the_oracle(path, choice);
        if !tokens.is_empty() {
            assert_eq!(tokens, SAFETENSORS_CONTINUATION, "{choice:?}");
        }
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
        // Device half only: the checkpoint's CPU oracle needs more than 64 GB of host RAM, so its comparison
        // (top-five ids and log-probabilities within 1e-3) is a separate row that waits for the oracle
        // fixture recorded on the PTX pod (POOT-1184).
        qwen35_27b_fp8_receipt: "qwen3.8-27b-fp8",
            |path, choice| {
                generates_paris(path, choice);
                print_peak_host_memory();
            },
            "loads the real Qwen3.8-27B-FP8 checkpoint (about 28 GB) onto the GPU; needs >= 40Gi available and all of the lane's real-checkpoint guards";
        smollm3_gguf_oracle: "smollm3-3b-gguf/smollm3-3b-bf16.gguf",
            smollm3_gguf_continues_like_the_safetensors_checkpoint,
            "loads the real SmolLM3-3B bf16 GGUF (6 GB, f32 on the CPU oracle) onto the GPU; needs >= 40Gi available";
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
        gemma2_coherent: "gemma-2-2b-it", names_paris,
            "loads gemma-2-2b-it (softcapped attention, GeGLU) onto the GPU; needs >= 40Gi available";
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
        mpt_gguf_coherent: "mptk-1b-gguf/mptk-1b-f16.gguf", mptk_is_fluent,
            "loads the mptk-1b F16 GGUF (llama.cpp convert_hf_to_gguf.py; tied embeddings, no output.weight) onto the GPU; needs >= 40Gi available";
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
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, fixture.store.clone(), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap(),
    )
}

/// [`family_handle`] with its weights scaled up, so a hybrid fixture's recurrent state decides its tokens
/// (at the fixture's own scale the greedy picks do not depend on the state, and a wrong state row would
/// pass).
fn recurrent_family_handle(family: &str) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("no builtin family {family}"));
    let fixture = (entry.fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, scaled(&fixture.store, 4.0), &registry, |_| {
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

/// A contiguous-layout driver generating one sequence alone, the reference the batched rows equal.
fn solo_driver(choice: BackendChoice, handle: &Arc<ModelHandle>) -> Option<Driver<Serving>> {
    Some(serving_driver(choice, handle)?.serve(Layout::Contiguous))
}

/// Prepare the serving shapes the rows draw from.
fn prepare_serving(
    driver: Driver,
    pool: PoolShape,
    rows: &[usize],
    heads: &[Head],
    windows: &[usize],
) -> Driver<Serving> {
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
        .unwrap()
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
        driver: &mut Driver<Serving>,
        prompt: &[u32],
        max_new: usize,
        sampler: Sampler,
    ) -> (Self, usize) {
        Self::open_with(driver, prompt, max_new, sampler, RequestInputs::none())
    }

    /// A lane whose prompt carries embedding rows in place of some tokens' embeddings.
    fn open_embeds(
        driver: &mut Driver<Serving>,
        prompt: &[u32],
        embeds: RequestInputs,
        max_new: usize,
    ) -> (Self, usize) {
        Self::open_with(driver, prompt, max_new, Sampler::greedy(), embeds)
    }

    fn open_with(
        driver: &mut Driver<Serving>,
        prompt: &[u32],
        max_new: usize,
        sampler: Sampler,
        inputs: RequestInputs,
    ) -> (Self, usize) {
        let request = SeqRequest::new(prompt, max_new, AdapterRef::Base).with_inputs(inputs);
        let admission = driver.open(request).unwrap();
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
            RowWork::Decode { ahead: &[] }
        }
    }
}

/// One step over `lanes[active]`, committing every token it returns. Returns whether the step mixed a
/// prefilling row with at least two decoding rows.
fn step_lanes(driver: &mut Driver<Serving>, lanes: &mut [Lane], active: &[usize]) -> bool {
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
fn run_to_end(driver: &mut Driver<Serving>, lanes: &mut [Lane]) {
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
    mixed_step_matches_generate_of(choice, family, family_handle(family, NO_EOS), None);
}

/// [`mixed_step_matches_generate`] over `handle`. With `recorded`, the solo runs (the reference) must
/// first equal the tokens recorded before the batching change, so a fault in the shared tracer, which
/// the batched and the solo run would both inherit, still turns the row red.
fn mixed_step_matches_generate_of(
    choice: BackendChoice,
    family: &str,
    handle: Arc<ModelHandle>,
    recorded: Option<&[Vec<u32>; 3]>,
) {
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = solo_driver(choice, &handle) else {
        return;
    };
    let prompts: [&[u32]; 3] = [&LONG, &MEDIUM, &SHORT];
    let want: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| generate(&mut reference, p, SERVING_MAX_NEW, Sampler::greedy()))
        .collect();
    if let Some(recorded) = recorded {
        assert_eq!(
            &want, recorded,
            "{family}: the solo runs equal the recorded tokens"
        );
    }
    let mut driver = prepare_serving(driver, pool_shape(8, 4), &[3], &[Head::GREEDY], &[]);
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
    BackendChoice,
    mixed_step_qwen2,
    mixed_step_qwen2_wgpu,
    mixed_step_qwen2_rocm,
    mixed_step_qwen2_ptx
);
fn mixed_step_qwen2(choice: BackendChoice) {
    mixed_step_matches_generate(choice, "qwen2");
}
per_backend!(
    BackendChoice,
    mixed_step_bloom,
    mixed_step_bloom_wgpu,
    mixed_step_bloom_rocm,
    mixed_step_bloom_ptx
);
fn mixed_step_bloom(choice: BackendChoice) {
    mixed_step_matches_generate(choice, "bloom");
}

/// POOT-1074 SC-001: the mixed step above on a hybrid or SSM family; `recorded` is what `Driver::generate`
/// produced for the three prompts at the commit before batching existed (the solo runs on the contiguous
/// layout, which batching leaves as it was). The three sequences share one
/// recurrent state pool of three rows, a step's rows are the sequences in whatever order the lanes list
/// them, the long prompt's four-token pieces run beside one-token decodes (so the driver runs one entry
/// per width), and every row still produces what `Driver::generate` produces for that sequence alone.
/// Mutation: bind `state_read[i] = i` instead of the sequence's slot in `Driver::run_plans`; the rows
/// whose slot differs from their position continue another sequence's state and diverge.
fn mixed_step_recurrent(choice: BackendChoice, family: &str, recorded: &[Vec<u32>; 3]) {
    mixed_step_matches_generate_of(
        choice,
        family,
        recurrent_family_handle(family),
        Some(recorded),
    );
}
per_backend!(
    BackendChoice,
    mixed_step_qwen3next,
    mixed_step_qwen3next_wgpu,
    mixed_step_qwen3next_rocm,
    mixed_step_qwen3next_ptx
);
fn mixed_step_qwen3next(choice: BackendChoice) {
    mixed_step_recurrent(
        choice,
        "qwen3next",
        &[
            vec![36, 37, 44, 37, 34, 11, 22, 14, 17, 21, 7, 25],
            vec![20, 0, 17, 20, 0, 34, 2, 5, 3, 3, 2, 11],
            vec![37, 14, 34, 17, 16, 29, 6, 3, 37, 22, 46, 15],
        ],
    );
}
per_backend!(
    BackendChoice,
    mixed_step_qwen35,
    mixed_step_qwen35_wgpu,
    mixed_step_qwen35_rocm,
    mixed_step_qwen35_ptx
);
fn mixed_step_qwen35(choice: BackendChoice) {
    mixed_step_recurrent(
        choice,
        "qwen35",
        &[
            vec![43, 10, 29, 45, 38, 18, 10, 11, 15, 34, 38, 38],
            vec![38, 9, 20, 17, 25, 3, 20, 12, 38, 1, 45, 37],
            vec![11, 21, 20, 21, 45, 37, 10, 34, 5, 40, 28, 3],
        ],
    );
}
per_backend!(
    BackendChoice,
    mixed_step_nemotron_h,
    mixed_step_nemotron_h_wgpu,
    mixed_step_nemotron_h_rocm,
    mixed_step_nemotron_h_ptx
);
fn mixed_step_nemotron_h(choice: BackendChoice) {
    mixed_step_recurrent(
        choice,
        "nemotron_h",
        &[
            vec![22, 27, 8, 38, 26, 41, 37, 26, 10, 33, 19, 22],
            vec![10, 47, 8, 47, 8, 36, 45, 27, 33, 8, 33, 17],
            vec![2, 29, 11, 20, 11, 38, 41, 20, 30, 27, 27, 36],
        ],
    );
}

/// POOT-1074 SC-002: release and re-admit on a hybrid or SSM family, three sequences through a pool of
/// two rows. The long and the short sequence fill the pool; the short one ends and is released; the
/// third is admitted into its slot (slot 1) while the long one is mid-decode. The third must start from
/// zero state (it equals its solo run) and the long one must be untouched by the admission (it equals
/// its solo run, state and KV pool included).
/// Mutation: drop the reset in `Driver::open`; `third` continues `first`'s state and diverges. Mutation:
/// reset `StateScope::All` there; `second`'s state is zeroed mid-sequence and it diverges.
fn readmit_matches_generate(choice: BackendChoice, family: &str) {
    let handle = recurrent_family_handle(family);
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    let Some(mut reference) = solo_driver(choice, &handle) else {
        return;
    };
    let (long_prompt, short_prompt, third_prompt): (&[u32], &[u32], &[u32]) =
        (&LONG, &MEDIUM, &SHORT);
    let short_new = 3;
    let solo = |driver: &mut Driver<Serving>, prompt: &[u32], new| {
        generate(driver, prompt, new, Sampler::greedy())
    };
    let want_long = solo(&mut reference, long_prompt, SERVING_MAX_NEW);
    let want_short = solo(&mut reference, short_prompt, short_new);
    let want_third = solo(&mut reference, third_prompt, SERVING_MAX_NEW);
    let mut driver = prepare_serving(driver, pool_shape(8, 2), &[2], &[Head::GREEDY], &[]);
    // The long sequence takes slot 0 and the short one slot 1: the slot that is recycled is not the
    // first, so the row a re-admission zeroes is not the row (or the KV pool position) 0.
    let mut lanes = vec![
        Lane::open(&mut driver, long_prompt, SERVING_MAX_NEW, Sampler::greedy()).0,
        Lane::open(&mut driver, short_prompt, short_new, Sampler::greedy()).0,
    ];
    while !lanes[1].done() {
        step_lanes(&mut driver, &mut lanes, &[0, 1]);
    }
    assert_eq!(lanes[1].out, want_short, "{family}: the short sequence");
    driver.release(lanes[1].seq, Release::Finished).unwrap();
    assert!(
        !lanes[0].done() && lanes[0].prefilled > 0,
        "the long sequence is mid-flight when the third is admitted"
    );
    lanes[1] = Lane::open(
        &mut driver,
        third_prompt,
        SERVING_MAX_NEW,
        Sampler::greedy(),
    )
    .0;
    run_to_end(&mut driver, &mut lanes);
    assert_eq!(
        lanes[0].out, want_long,
        "{family}: the sequence beside the admission"
    );
    assert_eq!(
        lanes[1].out, want_third,
        "{family}: the re-admitted sequence"
    );
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
}
per_backend!(
    BackendChoice,
    readmit_qwen3next,
    readmit_qwen3next_wgpu,
    readmit_qwen3next_rocm,
    readmit_qwen3next_ptx
);
fn readmit_qwen3next(choice: BackendChoice) {
    readmit_matches_generate(choice, "qwen3next");
}
per_backend!(
    BackendChoice,
    readmit_nemotron_h,
    readmit_nemotron_h_wgpu,
    readmit_nemotron_h_rocm,
    readmit_nemotron_h_ptx
);
fn readmit_nemotron_h(choice: BackendChoice) {
    readmit_matches_generate(choice, "nemotron_h");
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
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = solo_driver(choice, &handle) else {
        return;
    };
    let (a, b, c) = shared_prompts();
    let (want_a, want_b) = (
        generate(&mut reference, &a, 4, Sampler::greedy()),
        generate(&mut reference, &b, 4, Sampler::greedy()),
    );
    // Six blocks: A (39 positions) takes three, C (54) takes four of what remains.
    let mut driver = prepare_serving(driver, pool_shape(6, 2), &[1], &[Head::GREEDY], &[]);
    let run = |driver: &mut Driver<Serving>, prompt: &[u32]| {
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
    BackendChoice,
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
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = solo_driver(choice, &handle) else {
        return;
    };
    let want = generate(&mut reference, &MEDIUM, 8, Sampler::greedy());
    let mut driver = prepare_serving(driver, pool_shape(4, 1), &[1], &[Head::GREEDY], &[2]);
    let (lane, _) = Lane::open(&mut driver, &MEDIUM, 8, Sampler::greedy());
    let mut lanes = [lane];
    while lanes[0].out.is_empty() {
        step_lanes(&mut driver, &mut lanes, &[0]);
    }
    assert_eq!(lanes[0].out, want[..1]);
    let seq = lanes[0].seq;
    let window = |driver: &mut Driver<Serving>, lane: &mut Lane, ahead: &[u32]| {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Decode { ahead },
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
    BackendChoice,
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
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    // The reference is the contiguous single row; both executors open before either runs.
    let Some(mut reference) = solo_driver(choice, &handle) else {
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
    let mut driver = prepare_serving(
        driver,
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
    BackendChoice,
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
        preprocessor: None,
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
        1,
    );
}

/// `family`'s fixture with NaN in the router tensor `router` of layer `layer` stops generation with
/// the `RouterLogitsFinite` witness of that layer.
fn a_nan_router_weight_is_a_typed_validation_fault_of(
    choice: BackendChoice,
    family: &str,
    router: &str,
    layer: usize,
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
        Some((layer, WitnessKind::RouterLogitsFinite)),
        "{failure}"
    );
}
per_backend!(
    BackendChoice,
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
        1,
    );
}
per_backend!(
    BackendChoice,
    a_nan_minimax_router_weight_is_a_typed_validation_fault,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_minimax_router_weight_is_a_typed_validation_fault_ptx
);

/// POOT-568 SC-006 (F4): a NaN in the router of a score-routed layer (layer 1, the first CSA layer) and in
/// the router of the hash-routed layer 0 is `RouterLogitsFinite` naming that layer through the driver, not a
/// published token: the hash layer declares the same three witnesses as the shared router. Mutation: drop
/// the witness emission from `hash_routes` (layer 0 row) or from `components::router` (layer 1 row); the NaN
/// publishes tokens and the row goes red.
fn a_nan_deepseek4_router_weight_is_a_typed_validation_fault(choice: BackendChoice) {
    a_nan_router_weight_is_a_typed_validation_fault_of(
        choice,
        "deepseek4",
        "layers.1.ffn.gate.weight",
        1,
    );
}
per_backend!(
    BackendChoice,
    a_nan_deepseek4_router_weight_is_a_typed_validation_fault,
    a_nan_deepseek4_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_deepseek4_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_deepseek4_router_weight_is_a_typed_validation_fault_ptx
);

fn a_nan_deepseek4_hash_router_weight_is_a_typed_validation_fault(choice: BackendChoice) {
    a_nan_router_weight_is_a_typed_validation_fault_of(
        choice,
        "deepseek4",
        "layers.0.ffn.gate.weight",
        0,
    );
}
per_backend!(
    BackendChoice,
    a_nan_deepseek4_hash_router_weight_is_a_typed_validation_fault,
    a_nan_deepseek4_hash_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_deepseek4_hash_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_deepseek4_hash_router_weight_is_a_typed_validation_fault_ptx
);

/// POOT-568 SC-007 (F3): the DeepSeek-V4 fixture's prefill granule is 4 (compression rates 2 and 4). A
/// ten-token prompt runs as one prefill of eight tokens and two decode steps, and the eight tokens it then
/// generates are the independent reference's token-by-token decode (the reference reads no prefill at all):
/// the family's trace refuses a prefill that is not a multiple of four, so a driver that chunked at six
/// would fail the run instead. Literals: `deepseek4::tests::a_granule_four_prompt_decodes_as_the_reference`.
/// Mutation: chunk at 6 in the driver (`ChunkPolicy::new` accepting a chunk that is no multiple of the
/// granule); the run is refused and the row goes red.
fn a_deepseek4_prompt_of_ten_runs_as_prefill_eight_and_two_decode_steps(choice: BackendChoice) {
    const PROMPT10: [u32; 10] = [3, 17, 40, 8, 25, 9, 31, 2, 44, 12];
    const TOKENS: [u32; 8] = [36, 13, 21, 9, 32, 15, 32, 2];
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "deepseek4")
        .unwrap();
    let handle = recorded_family_handle(recorded);
    assert_eq!(handle.config().prefill_granule.get(), 4);
    let Some(mut driver) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    assert_eq!(
        generate(&mut driver, &PROMPT10, 8, Sampler::greedy()),
        TOKENS,
        "deepseek4 on {choice:?}"
    );
}
per_backend!(
    BackendChoice,
    a_deepseek4_prompt_of_ten_runs_as_prefill_eight_and_two_decode_steps,
    a_deepseek4_prompt_of_ten_runs_as_prefill_eight_and_two_decode_steps_wgpu,
    a_deepseek4_prompt_of_ten_runs_as_prefill_eight_and_two_decode_steps_rocm,
    a_deepseek4_prompt_of_ten_runs_as_prefill_eight_and_two_decode_steps_ptx
);

/// POOT-568 SC-007: a prefill chunk of six is not a multiple of the family's granule: the driver refuses
/// it with the typed option error before it loads any executable (the chunk is the number of new tokens of
/// a step, so the six-token step the trace refuses is never built). Mutation: drop the granule check from
/// `ChunkPolicy::new`; the driver constructs and the row goes red.
fn a_deepseek4_chunk_that_is_not_a_granule_multiple_is_refused_before_any_executable(
    choice: BackendChoice,
) {
    use crate::driver::error::InvalidOptions;
    let handle = recorded_family_handle(
        RECORDED
            .iter()
            .find(|recorded| recorded.family == "deepseek4")
            .unwrap(),
    );
    let Some(executor) = open(choice) else {
        return;
    };
    match Driver::new(handle, executor, options(6, CAPACITY)) {
        Err(DriverError::InvalidOptions(InvalidOptions::ChunkNotGranuleAligned {
            chunk: 6,
            granule: 4,
        })) => {}
        other => panic!("expected the granule refusal, got {:?}", other.map(|_| ())),
    }
}
per_backend!(
    BackendChoice,
    a_deepseek4_chunk_that_is_not_a_granule_multiple_is_refused_before_any_executable,
    a_deepseek4_chunk_that_is_not_a_granule_multiple_is_refused_before_any_executable_wgpu,
    a_deepseek4_chunk_that_is_not_a_granule_multiple_is_refused_before_any_executable_rocm,
    a_deepseek4_chunk_that_is_not_a_granule_multiple_is_refused_before_any_executable_ptx
);

/// POOT-568 SC-002 (F5): a DeepSeek-V4 checkpoint directory a pinned config refused (three layers in
/// another order, other compression rates, two hash layers, six experts) and no manifest anywhere loads
/// through `ModelHandle::load`, resolves to the `deepseek4` family and keeps the config's own counts.
/// Mutation: reinstate a pinned layer count in `DeepseekV4Config::from_raw`; the load is refused and the
/// row goes red.
#[test]
fn a_finetune_like_deepseek4_checkpoint_loads_through_model_handle_load() {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "deepseek4")
        .unwrap();
    let (_, finetune) = entry
        .variants
        .iter()
        .find(|(name, _)| *name == "finetune")
        .expect("deepseek4 ships a finetune-like fixture");
    let fixture = finetune();
    let dir = fixture_dir("deepseek4_finetune", &fixture);
    let handle =
        ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default()).unwrap();
    assert_eq!(handle.config().family, entry.family);
    assert_eq!(handle.config().vocab, VOCAB);
    assert_eq!(handle.config().prefill_granule.get(), 4);
}

/// POOT-569 SC-002 (F5): a GLM-5.3-Flash checkpoint directory a pinned config refused (three layers in
/// another order, four experts, other heads and pool size, no multi-token-prediction layer) and no
/// manifest anywhere loads through `ModelHandle::load`, resolves to the `glm53_flash` family and keeps
/// the config's own counts. Mutation: reinstate a pinned layer count in `Glm53Config::from_raw`; the
/// load is refused and the row goes red.
#[test]
fn a_finetune_like_glm53_flash_checkpoint_loads_through_model_handle_load() {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "glm53_flash")
        .unwrap();
    let (_, finetune) = entry
        .variants
        .iter()
        .find(|(name, _)| *name == "finetune")
        .expect("glm53_flash ships a finetune-like fixture");
    let fixture = finetune();
    let dir = fixture_dir("glm53_flash_finetune", &fixture);
    let handle =
        ModelHandle::load(&dir, &registry, &crate::driver::LoadOptions::default()).unwrap();
    assert_eq!(handle.config().family, entry.family);
    assert_eq!(handle.config().vocab, VOCAB);
    assert_eq!(handle.config().prefill_granule.get(), 1);
}

/// POOT-569 SC-001: a NaN in layer 1's router (the first routed layer) is `RouterLogitsFinite` naming
/// layer 1 through the driver, not a published token. Mutation: drop the router's witness emission in
/// `components::router`; the NaN publishes tokens and the row goes red.
fn a_nan_glm53_flash_router_weight_is_a_typed_validation_fault(choice: BackendChoice) {
    a_nan_router_weight_is_a_typed_validation_fault_of(
        choice,
        "glm53_flash",
        "model.language_model.layers.1.mlp.gate.weight",
        1,
    );
}
per_backend!(
    BackendChoice,
    a_nan_glm53_flash_router_weight_is_a_typed_validation_fault,
    a_nan_glm53_flash_router_weight_is_a_typed_validation_fault_wgpu,
    a_nan_glm53_flash_router_weight_is_a_typed_validation_fault_rocm,
    a_nan_glm53_flash_router_weight_is_a_typed_validation_fault_ptx
);

/// The independent f64 reference's log-probabilities over the GLM-5.3-Flash fixture's whole vocabulary
/// at the 24 positions the recorded tokens are sampled at, one line per position
/// (`glm53_flash::tests::the_device_logprob_literals_are_the_references` pins the file to the reference).
const GLM53_FLASH_REFERENCE_LOGPROBS: &str =
    include_str!("device_tests/glm53_flash_reference_logprobs.txt");

/// POOT-1056 (POOT-569 SC-001/005/006, device half, tier 3): the GLM-5.3-Flash fixture's whole
/// distribution at every generated position through the driver equals the independent f64 reference's
/// (each of the 48 log-probabilities within `1e-3`), whether the 7-token prompt prefills in one piece
/// (`chunk` 8) or in pieces of two: the KDA recurrence and the DSA pool state carry across the pieces and then across 23
/// decode steps. The device exposes no state readback, so a wrong carried KDA or DSA element is
/// observed in every later position's distribution. Mutation: zero the carried KDA state
/// (`step.carry(state, s)` in `components::kda` carries `s * 0`); the second position's distribution
/// moves and the row goes red.
fn glm53_flash_every_position_matches_the_reference_with(choice: BackendChoice, chunk: usize) {
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "glm53_flash")
        .unwrap();
    let handle = recorded_family_handle(recorded);
    let vocab = handle.config().vocab;
    let want: Vec<Vec<f64>> = GLM53_FLASH_REFERENCE_LOGPROBS
        .lines()
        .map(|line| line.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), recorded.tokens.len());
    let Some(mut driver) = driver(choice, &handle, chunk, CAPACITY) else {
        return;
    };
    let generated = driver
        .generate(
            GenerateRequest {
                prompt: recorded.prompt.to_vec(),
                max_new: recorded.tokens.len(),
                sampler: Sampler::greedy().with_logprobs(vocab),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap();
    assert_eq!(
        generated.tokens, recorded.tokens,
        "glm53_flash on {choice:?}, prefill pieces of {chunk}: tokens"
    );
    assert_eq!(generated.logprobs.len(), want.len());
    for (position, (got, want)) in generated.logprobs.iter().zip(&want).enumerate() {
        assert_eq!(
            got.top.len(),
            vocab,
            "position {position}: the whole vocabulary is recorded"
        );
        for &(token, got) in &got.top {
            let want = want[token as usize];
            assert!(
                (f64::from(got) - want).abs() <= 1e-3,
                "glm53_flash on {choice:?}, prefill pieces of {chunk}, generated position \
                 {position}: token {token} has log-probability {got}, the reference gives {want}"
            );
        }
    }
}
fn glm53_flash_every_position_matches_the_reference(choice: BackendChoice) {
    glm53_flash_every_position_matches_the_reference_with(choice, 8);
}
per_backend!(
    BackendChoice,
    glm53_flash_every_position_matches_the_reference,
    glm53_flash_every_position_matches_the_reference_wgpu,
    glm53_flash_every_position_matches_the_reference_rocm,
    glm53_flash_every_position_matches_the_reference_ptx
);

/// The same row with the prompt prefilled in pieces of two (a driver per row: a ROCm executor cannot
/// be reopened in one process after a drop).
fn glm53_flash_every_position_matches_the_reference_in_pieces_of_two(choice: BackendChoice) {
    glm53_flash_every_position_matches_the_reference_with(choice, 2);
}
per_backend!(
    BackendChoice,
    glm53_flash_every_position_matches_the_reference_in_pieces_of_two,
    glm53_flash_every_position_matches_the_reference_in_pieces_of_two_wgpu,
    glm53_flash_every_position_matches_the_reference_in_pieces_of_two_rocm,
    glm53_flash_every_position_matches_the_reference_in_pieces_of_two_ptx
);

/// The independent f64 reference's logits of the GLM-5.3-Flash fixture's MTP chain (`k = 3` over a
/// six-token prefix and a bonus token), one line per draft step
/// (`glm53_flash::tests::the_device_mtp_logit_literals_are_the_references_chain` pins the file to the
/// reference).
const GLM53_FLASH_REFERENCE_MTP_LOGITS: &str =
    include_str!("device_tests/glm53_flash_reference_mtp_logits.txt");

/// POOT-1110: the `draft` program of the GLM-5.3-Flash fixture executes on the device through the
/// executor contract (compile, load, add entry, step; no driver role) over the carry the target's own
/// prefill wrote into the executable's shared state. The target prefills six tokens with a hidden-state
/// carry; the MTP layer then catches up over all six positions with the token that follows each (the
/// last the bonus), and two one-token chain steps each feed the greedy draft of the step before and
/// read the post-norm state that step left in the carry. Every draft step's whole logit row equals the
/// f64 reference chain's within `2e-3` and every draft token is the reference's. Mutation: swap the
/// concatenation order in `glm53_flash::mtp::MtpLayer::trace`; the first logit row leaves the reference
/// and the row goes red.
fn glm53_flash_mtp_chain_matches_the_reference(choice: BackendChoice) {
    use poot_executor::{NoSync, StepInputs};
    use poot_graph_ir::{Slot, SlotKey};
    use poot_graph_plan::{
        WeightFormats, bind_packed_weights, multi_device::topology::DeviceId,
        staged::DevicePlacement, staged::ExpertPlacement, staged::Partition,
        staged::compile_staged,
    };
    use poot_models::components::standard::HIDDEN_CARRY_STATE;
    use poot_models::io::ProgramId;
    use poot_models::model::{HiddenCarry, StepShape, StepSpec};
    use poot_tensor::HostView;

    const PREFIX: [i32; 6] = [3, 17, 40, 8, 25, 9];
    const BONUS: i32 = 31;
    const K: usize = 3;
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "glm53_flash")
        .unwrap();
    let handle = recorded_family_handle(recorded);
    let model = handle.model();
    let want: Vec<Vec<f64>> = GLM53_FLASH_REFERENCE_MTP_LOGITS
        .lines()
        .map(|line| line.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), K);
    let Some(mut exec) = open(choice) else {
        return;
    };
    let carry = HiddenCarry {
        positions: nz((K + 1).max(PREFIX.len())),
    };
    let shape = |tokens: usize| StepShape::new(nz(1), nz(tokens), nz(16)).with_hidden(Some(carry));
    let formats = WeightFormats::from_weight_map(model.weights());
    let exe = exec
        .load_weights(
            Arc::new(handle.store().clone()),
            Arc::new(model.weights().clone()),
        )
        .unwrap();
    // One entry per program, all on the one executable: the carry is shared by its name and aval.
    let mut entry = |program: ProgramId, tokens: usize| {
        let graph = model
            .trace(
                &StepSpec::new(program, shape(tokens)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        assert!(
            graph
                .state()
                .any(|(input, _)| graph.meta(input).name.as_deref() == Some(HIDDEN_CARRY_STATE)),
            "{program} at {tokens} tokens declares the carry"
        );
        let graph = bind_packed_weights(&graph, &formats).unwrap();
        let program = compile_staged(
            &graph,
            &exec.target_set(),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .unwrap_or_else(|e| panic!("{program} at {tokens} tokens on {choice:?}: {e}"));
        exec.add_entry(exe, &program).unwrap()
    };
    let prefill = entry(ProgramId::STEP, PREFIX.len());
    let catch_up = entry(ProgramId::DRAFT, PREFIX.len());
    let chain = entry(ProgramId::DRAFT, 1);

    let mut run = |entry, tokens: &[i32], start: i32| -> Vec<f32> {
        let shape = [1, tokens.len()];
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let pos_bytes: Vec<u8> = (start..)
            .take(tokens.len())
            .flat_map(|p| p.to_le_bytes())
            .collect();
        let mut inputs = StepInputs::new();
        for (slot, bytes) in [(Slot::Token, &token_bytes), (Slot::Pos, &pos_bytes)] {
            inputs.push(
                SlotKey::new(slot, None),
                &shape,
                HostView::new(poot_tensor::DType::I32, tokens.len(), bytes).unwrap(),
            );
        }
        let bytes = exec
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("step on {choice:?}: {e}"))
            .read()
            .unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    run(prefill, &PREFIX, 0);
    let following: Vec<i32> = PREFIX[1..].iter().copied().chain([BONUS]).collect();
    let n = PREFIX.len();
    let mut steps = vec![run(catch_up, &following, 0)];
    for i in 1..K {
        let previous = &steps[i - 1];
        let token = previous
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as i32;
        steps.push(run(chain, &[token], (n - 1 + i) as i32));
    }
    let argmax = |row: &[f64]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    for (step, (got, want)) in steps.iter().zip(&want).enumerate() {
        assert_eq!(got.len(), want.len(), "the whole vocabulary");
        for (token, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (f64::from(got) - want).abs() <= 2e-3 * (1.0 + want.abs()),
                "glm53_flash MTP on {choice:?}, draft step {step}, token {token}: logit {got}, \
                 the reference gives {want}"
            );
        }
        let got_row: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
        assert_eq!(
            argmax(&got_row),
            argmax(want),
            "draft step {step}: the draft token"
        );
    }
}
per_backend!(
    BackendChoice,
    glm53_flash_mtp_chain_matches_the_reference,
    glm53_flash_mtp_chain_matches_the_reference_wgpu,
    glm53_flash_mtp_chain_matches_the_reference_rocm,
    glm53_flash_mtp_chain_matches_the_reference_ptx
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
    BackendChoice,
    minimax_three_positions_match_the_reference,
    minimax_three_positions_match_the_reference_wgpu,
    minimax_three_positions_match_the_reference_rocm,
    minimax_three_positions_match_the_reference_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-738 SC-001 (ADR-0101 tier 2): the first step of each MoE family on the device equals the CPU
// oracle on the same traced graph.
// ---------------------------------------------------------------------------------------------

/// The oracle's last-position logits after each step of `steps` (first position, length) over
/// `tokens`, on `handle`'s model, its own traced graph (a cache of `capacity` positions) and weight
/// map; the carried state flows from step to step. One row per step.
fn oracle_step_logits(
    handle: &ModelHandle,
    tokens: &[u32],
    steps: &[(usize, usize)],
    capacity: usize,
) -> Vec<Vec<f64>> {
    let model = handle.model();
    let mut state: Vec<poot_tensor::HostTensor> = Vec::new();
    let mut rows = Vec::new();
    for &(start, len) in steps {
        let shape = StepShape::new(nz(1), nz(len), nz(capacity));
        let graph = model
            .trace(
                &StepSpec::step(shape),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
        let graph = poot_graph_plan::bind_packed_weights(&graph, &formats).unwrap();
        let ids: Vec<i32> = tokens[start..start + len]
            .iter()
            .map(|&t| t as i32)
            .collect();
        let slots = [
            (
                SlotKey::new(Slot::Token, None),
                poot_tensor::HostTensor::i32(vec![1, len], ids),
            ),
            (
                SlotKey::new(Slot::Pos, None),
                poot_tensor::HostTensor::i32(vec![1, len], (start as i32..).take(len).collect()),
            ),
        ];
        if state.is_empty() {
            // A carried token history is I32; every other state starts as zeros.
            state = graph
                .state()
                .map(|(si, _)| {
                    let aval = graph.aval(si);
                    match aval.dtype {
                        poot_tensor::DType::I32 => {
                            poot_tensor::HostTensor::i32(aval.shape.clone(), vec![0; aval.numel()])
                        }
                        _ => poot_tensor::HostTensor::zeros(aval.shape.clone()),
                    }
                })
                .collect();
        }
        let (logits, next) = poot_eval::test_support::weight_map_oracle::eval_mapped(
            &graph,
            handle.store(),
            model.weights(),
            &slots,
            &state,
        );
        state = next;
        rows.push(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        );
    }
    rows
}

/// `logits` as log-probabilities.
fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::MIN, f64::max);
    let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
    logits.iter().map(|l| l - log_sum).collect()
}

/// The oracle's top five `(token, log-probability)` for the last position of `prompt` on `handle`'s
/// model, over its own traced prefill graph (a cache of `capacity` positions) and weight map.
fn oracle_top5(handle: &ModelHandle, prompt: &[u32], capacity: usize) -> Vec<(u32, f64)> {
    let n = prompt.len();
    // The prompt as the driver runs it: one prefill of the largest multiple of the model's granule
    // (none when the prompt is shorter than it) and a decode step for each remaining token.
    let granule = handle.model().config().prefill_granule.get();
    let aligned = n / granule * granule;
    let steps: Vec<(usize, usize)> = (aligned > 0)
        .then_some((0, aligned))
        .into_iter()
        .chain((aligned..n).map(|at| (at, 1)))
        .collect();
    let rows = oracle_step_logits(handle, prompt, &steps, capacity);
    let last = rows.last().expect("a prompt has at least one step");
    let mut ranked: Vec<(u32, f64)> = log_softmax(last)
        .into_iter()
        .enumerate()
        .map(|(t, l)| (t as u32, l))
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
    let want = oracle_top5(&handle, recorded.prompt, CAPACITY);
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
    "deepseek4" => deepseek4_first_step_matches_the_oracle_wgpu, deepseek4_first_step_matches_the_oracle_rocm;
    "qwen3next" => qwen3next_first_step_matches_the_oracle_wgpu, qwen3next_first_step_matches_the_oracle_rocm;
    "qwen4exp" => qwen4exp_first_step_matches_the_oracle_wgpu, qwen4exp_first_step_matches_the_oracle_rocm;
    "qwen4exp/packed" => qwen4exp_packed_first_step_matches_the_oracle_wgpu, qwen4exp_packed_first_step_matches_the_oracle_rocm;
    "qwen4exp/finetune" => qwen4exp_finetune_first_step_matches_the_oracle_wgpu, qwen4exp_finetune_first_step_matches_the_oracle_rocm;
    "qwen35" => qwen35_first_step_matches_the_oracle_wgpu, qwen35_first_step_matches_the_oracle_rocm;
    "glm53_flash" => glm53_flash_first_step_matches_the_oracle_wgpu, glm53_flash_first_step_matches_the_oracle_rocm;
}

// ---------------------------------------------------------------------------------------------
// POOT-739: `Driver::encode` and prompts with embedding rows, on a device against the CPU oracle.
// ---------------------------------------------------------------------------------------------

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_graph_ir::{Slot, SlotKey};
use poot_models::components::embed::EMBEDS;
use poot_models::io::{InputValue, ProgramId, RequestInputs};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::registry::Fixture;
use poot_tensor::HostTensor;

use crate::driver::{EncodeRequest, EncodeRow};
use crate::encoders::segment_inputs;

/// The model of `family`'s fixture (or its named variant) as a handle, with the fixture's text
/// services (ids are fed directly).
fn encoder_family_handle(family: &str, variant: Option<&str>) -> (Arc<ModelHandle>, Fixture) {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap();
    let fixture = match variant {
        None => (entry.fixture)(),
        Some(name) => entry
            .variants
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, f)| f())
            .unwrap(),
    };
    let handle =
        ModelHandle::from_checkpoint(&fixture.raw(), fixture.store.clone(), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap();
    (Arc::new(handle), fixture)
}

/// The oracle's output of `program` over `slots`, from the handle's own weights.
fn oracle(
    handle: &ModelHandle,
    program: ProgramId,
    shape: StepShape,
    slots: &[(SlotKey, HostTensor)],
) -> Vec<f32> {
    let model = handle.model();
    let graph = model
        .trace(
            &StepSpec::new(program, shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let state: Vec<HostTensor> = graph
        .state()
        .map(|(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
        .collect();
    eval_mapped(&graph, handle.store(), model.weights(), slots, &state)
        .0
        .as_f32()
        .unwrap()
        .to_vec()
}

fn encode_shape(rows: usize, tokens: usize) -> StepShape {
    StepShape::new(nz(rows), nz(tokens), nz(tokens)).with_logits(LogitRows::All)
}

fn named(tag: &str) -> SlotKey {
    SlotKey::new(Slot::Activation, Some(tag))
}

/// The oracle's BERT output for one unpadded row: `[tokens, width]` (or the score).
fn oracle_bert(handle: &ModelHandle, ids: &[i32], types: &[i32]) -> Vec<f32> {
    let n = ids.len();
    let ints = |values: &[i32]| HostTensor::i32(vec![1, n], values.to_vec());
    oracle(
        handle,
        ProgramId::ENCODE,
        encode_shape(1, n),
        &[
            (SlotKey::new(Slot::Token, None), ints(ids)),
            (
                SlotKey::new(Slot::Pos, None),
                ints(&(0..n as i32).collect::<Vec<_>>()),
            ),
            (named("token_type"), ints(types)),
            (named("attention_mask"), ints(&vec![1; n])),
        ],
    )
}

/// Tier 2 (ADR-0101): every element within `1e-4 * (1 + |want|)`; NaN fails.
fn close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{label}: element {i}: device {g}, oracle {w}"
        );
    }
}

const ENCODE_SHORT: [u32; 3] = [5, 17, 3];
const ENCODE_LONG: [u32; 7] = [9, 2, 31, 4, 12, 6, 1];

fn encode<S: State>(driver: &mut Driver<S>, rows: &[EncodeRow<'_>]) -> crate::driver::Encoded {
    driver.encode(&EncodeRequest { rows }).unwrap()
}

fn as_i32(ids: &[u32]) -> Vec<i32> {
    ids.iter().map(|&t| t as i32).collect()
}

/// The request inputs of one 3x8x8 idefics3 frame: its normalized pixels, channel first.
fn frame_inputs(pixels: &[f32]) -> RequestInputs {
    RequestInputs::none().with(
        poot_models::io::InputId::new("pixels"),
        InputValue::Row(HostTensor::f32(vec![3, 8, 8], pixels.to_vec())),
    )
}

/// The request input `embeds` of a prompt: `rows` (`positions.len() * width` values, row-major) at the
/// prompt's `positions`.
fn embed_inputs(width: usize, positions: Vec<usize>, rows: Vec<f32>) -> RequestInputs {
    poot_models::components::embed::embeds_inputs(width, positions, rows).unwrap()
}

/// Greedy generation of `max_new` tokens from `prompt` with the request `inputs`.
fn generate_inputs(
    driver: &mut Driver<Serving>,
    prompt: Vec<u32>,
    max_new: usize,
    inputs: RequestInputs,
) -> Vec<u32> {
    driver
        .generate_with_inputs(
            GenerateRequest {
                prompt,
                max_new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            inputs,
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap()
        .tokens
}

/// SC-003 (device half): sentence embeddings of the `bert` fixture through `Driver::encode` equal
/// the CPU oracle's (tier 2), and padding does not matter: a short row batched beside a long one
/// (both padded to eight positions) equals the same row encoded alone (padded to four) and the
/// oracle's unpadded forward. Mutation: skip the attention mask in `bert`; the padded rows differ
/// by how far they are padded and the row goes red.
fn bert_encode_matches_the_oracle_whatever_the_padding(choice: BackendChoice) {
    let (handle, _) = encoder_family_handle("bert", None);
    let Some(mut driver) = token_encoder(choice, &handle, 16, 16) else {
        return;
    };
    let width = 32;
    let none = RequestInputs::none();
    let token_row = |ids: &'static [u32]| EncodeRow {
        tokens: Some(ids),
        inputs: &none,
    };
    let batch = encode(
        &mut driver,
        &[token_row(&ENCODE_SHORT), token_row(&ENCODE_LONG)],
    );
    let alone = encode(&mut driver, &[token_row(&ENCODE_SHORT)]);
    assert_eq!(batch.width, width);
    assert_eq!(batch.rows[0].len(), ENCODE_SHORT.len() * width);
    assert_eq!(batch.rows[1].len(), ENCODE_LONG.len() * width);
    let short = oracle_bert(&handle, &as_i32(&ENCODE_SHORT), &[0; 3]);
    let long = oracle_bert(&handle, &as_i32(&ENCODE_LONG), &[0; 7]);
    close("short row in a batch", &batch.rows[0], &short);
    close("long row in a batch", &batch.rows[1], &long);
    close("short row alone", &alone.rows[0], &short);
    let caps = driver.caps().expect("prepare_encode publishes the caps");
    assert_eq!(
        caps.encode.as_ref().map(|encode| encode.rows.get()),
        Some(4),
        "the caps carry the prepared encode shapes"
    );
    assert_eq!(caps.kv_blocks, 0, "an encoder holds no KV pool");
    assert!(caps.inputs.encode && caps.widths.is_empty(), "{caps:?}");
}
per_backend!(
    BackendChoice,
    bert_encode_matches_the_oracle_whatever_the_padding,
    bert_encode_matches_the_oracle_whatever_the_padding_wgpu,
    bert_encode_matches_the_oracle_whatever_the_padding_rocm,
    bert_encode_matches_the_oracle_whatever_the_padding_ptx
);

/// SC-003 (rerank half): the cross-encoder fixture scores three (query, document) pairs in one
/// batched `Encode` step, each equal to the oracle's score of that pair alone (tier 2), and the
/// ranking they give is the oracle's.
fn cross_encoder_scores_pairs_in_the_oracles_order(choice: BackendChoice) {
    let (handle, _) = encoder_family_handle("bert", Some("cross_encoder"));
    let Some(mut driver) = token_encoder(choice, &handle, 16, 16) else {
        return;
    };
    let pairs: [(&[u32], &[u32]); 3] = [
        (&[5, 17, 3, 9], &[0, 0, 0, 1]),
        (&[9, 2, 31, 4, 12], &[0, 0, 1, 1, 1]),
        (&[1, 6], &[0, 1]),
    ];
    let segments: Vec<RequestInputs> = pairs
        .iter()
        .map(|&(_, types)| segment_inputs(types))
        .collect();
    let rows: Vec<EncodeRow<'_>> = pairs
        .iter()
        .zip(&segments)
        .map(|(&(ids, _), inputs)| EncodeRow {
            tokens: Some(ids),
            inputs,
        })
        .collect();
    let encoded = encode(&mut driver, &rows);
    let got: Vec<f32> = encoded.rows.iter().map(|row| row[0]).collect();
    let want: Vec<f32> = pairs
        .iter()
        .map(|&(ids, types)| oracle_bert(&handle, &as_i32(ids), &as_i32(types))[0])
        .collect();
    close("pair scores", &got, &want);
    let order = |scores: &[f32]| {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        idx
    };
    assert_eq!(order(&got), order(&want), "the ranking");
    assert!(
        want.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-3),
        "the pair scores differ: {want:?}"
    );
}
per_backend!(
    BackendChoice,
    cross_encoder_scores_pairs_in_the_oracles_order,
    cross_encoder_scores_pairs_in_the_oracles_order_wgpu,
    cross_encoder_scores_pairs_in_the_oracles_order_rocm,
    cross_encoder_scores_pairs_in_the_oracles_order_ptx
);

/// The first four dimensions of the mean over `PROMPT`'s hidden states of the qwen2 fixture: the
/// sentence vector before normalization (recorded from the model's own `Encode` graph on the
/// oracle; `qwen2::model::tests`).
const RECORDED_MEAN: [f32; 4] = [-0.6346052, 0.3129701, 0.27594122, -0.24472162];

/// SC-004 (device half): a decoder's `Encode` phase through `Driver::encode` returns the hidden
/// state the oracle computes (tier 2), and its host mean is the recorded sentence vector.
/// Mutation: take the head input one layer early; the hidden states and the mean differ and the
/// row goes red.
fn decoder_encode_matches_the_oracle_and_its_recorded_mean(choice: BackendChoice) {
    let (handle, _) = encoder_family_handle("qwen2", None);
    let Some(mut driver) = token_encoder(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let ids = [3u32, 17, 40, 8, 25];
    let none = RequestInputs::none();
    let encoded = encode(
        &mut driver,
        &[EncodeRow {
            tokens: Some(&ids),
            inputs: &none,
        }],
    );
    let tokens = HostTensor::i32(vec![1, 5], as_i32(&ids));
    let pos = HostTensor::i32(vec![1, 5], (0..5).collect());
    let want = oracle(
        &handle,
        ProgramId::ENCODE,
        encode_shape(1, 5),
        &[
            (SlotKey::new(Slot::Token, None), tokens),
            (SlotKey::new(Slot::Pos, None), pos),
        ],
    );
    assert_eq!(encoded.width, 64);
    close("hidden states", &encoded.rows[0], &want);
    let mean: Vec<f32> = (0..4)
        .map(|j| (0..5).map(|i| encoded.rows[0][i * 64 + j]).sum::<f32>() / 5.0)
        .collect();
    close("host mean pooling", &mean, &RECORDED_MEAN);
}
per_backend!(
    BackendChoice,
    decoder_encode_matches_the_oracle_and_its_recorded_mean,
    decoder_encode_matches_the_oracle_and_its_recorded_mean_wgpu,
    decoder_encode_matches_the_oracle_and_its_recorded_mean_rocm,
    decoder_encode_matches_the_oracle_and_its_recorded_mean_ptx
);

/// SC-001 (device half, plumbing): a prompt whose embedding rows are the table rows of the tokens
/// they replace generates the plain prompt's tokens; a row that is another token's embedding
/// generates that token's continuation; and the rows are step inputs, so a second prompt of the
/// same shape with different rows runs on the entries the first prepared. Mutation: feed zeroed
/// rows; the tokens change and the row goes red.
fn prompt_embeds_replace_the_tokens_they_stand_for(choice: BackendChoice) {
    let (handle, fixture) = encoder_family_handle("qwen2", None);
    let Some(mut driver) = driver(choice, &handle, 4, CAPACITY) else {
        return;
    };
    let w = 64;
    let table: Vec<f32> = {
        let entry = fixture.store.get("model.embed_tokens.weight").unwrap();
        let mut one = WeightStore::builder();
        one.insert("w", entry.clone()).unwrap();
        poot_eval::materialize_dense(&one.build(), "w")
            .unwrap()
            .to_f32()
            .unwrap()
            .into_owned()
    };
    let rows_of = |tokens: &[u32]| -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&t| table[t as usize * w..(t as usize + 1) * w].to_vec())
            .collect()
    };
    let with_embeds =
        |driver: &mut Driver<Serving>, prompt: &[u32], positions: Vec<usize>, rows| {
            generate_inputs(driver, prompt.to_vec(), 8, embed_inputs(w, positions, rows))
        };
    // Placeholders (token 0) at positions 1 and 3, replaced by the rows of tokens 17 and 8.
    let plain = [3, 17, 40, 8, 25, 11, 2];
    let placeholders = [3, 0, 40, 0, 25, 11, 2];
    let want = generate(&mut driver, &plain, 8, Sampler::greedy());
    let got = with_embeds(&mut driver, &placeholders, vec![1, 3], rows_of(&[17, 8]));
    assert_eq!(got, want, "the table rows of the replaced tokens");
    let compiles = driver.stats().compiles;

    // Different rows, same shape: another continuation, and no new program.
    let swapped_prompt = [3, 9, 40, 30, 25, 11, 2];
    let swapped_want = generate(&mut driver, &swapped_prompt, 8, Sampler::greedy());
    let compiles_after_plain = driver.stats().compiles;
    let swapped = with_embeds(&mut driver, &placeholders, vec![1, 3], rows_of(&[9, 30]));
    assert_eq!(swapped, swapped_want, "another token's rows");
    assert_ne!(swapped, got, "the rows reach the logits");
    assert_eq!(
        driver.stats().compiles,
        compiles_after_plain,
        "different rows of the same shape compile nothing"
    );
    assert!(compiles >= 1);
}
per_backend!(
    BackendChoice,
    prompt_embeds_replace_the_tokens_they_stand_for,
    prompt_embeds_replace_the_tokens_they_stand_for_wgpu,
    prompt_embeds_replace_the_tokens_they_stand_for_rocm,
    prompt_embeds_replace_the_tokens_they_stand_for_ptx
);

/// SC-002 (device half): two frames of equal numel and different content through one vision
/// driver give different embedding rows, each the oracle's (tier 2): the frame is a bound input,
/// so the second encode never reuses the first's pixels, and nothing is evicted between them.
/// Both frames in one two-row step equal the two single-frame steps. Mutation: bind `pixels` as a
/// const again; the second frame reads the first's and the row goes red.
fn two_frames_of_one_size_embed_differently(choice: BackendChoice) {
    let (handle, _) = smolvlm_fixture_handle();
    let tower = Arc::clone(&handle.towers()[0].1);
    let Some(mut driver) = bare_driver(choice, &tower, 16, 16) else {
        return;
    };
    let frame = |seed: u64| -> Vec<f32> {
        (0..3 * 8 * 8)
            .map(|i| (((i as u64 * 7 + seed * 13) % 17) as f32 - 8.0) / 8.0)
            .collect()
    };
    let (first, second) = (frame(1), frame(2));
    let (first_inputs, second_inputs) = (frame_inputs(&first), frame_inputs(&second));
    fn row(inputs: &RequestInputs) -> EncodeRow<'_> {
        EncodeRow {
            tokens: None,
            inputs,
        }
    }
    let one =
        |driver: &mut Driver, inputs: &RequestInputs| encode(driver, &[row(inputs)]).rows.remove(0);
    let (a, b) = (
        one(&mut driver, &first_inputs),
        one(&mut driver, &second_inputs),
    );
    let model = |pixels: &[f32]| {
        oracle(
            &tower,
            ProgramId::ENCODE,
            encode_shape(1, 16),
            &[(
                named("pixels"),
                HostTensor::f32(vec![1, 3, 8, 8], pixels.to_vec()),
            )],
        )
    };
    assert_eq!(a.len(), 4 * 24, "four visual rows of the text width");
    close("first frame", &a, &model(&first));
    close("second frame", &b, &model(&second));
    assert_ne!(a, b, "two different frames embed alike");
    let both = encode(&mut driver, &[row(&first_inputs), row(&second_inputs)]);
    close("first frame in a two-row step", &both.rows[0], &a);
    close("second frame in a two-row step", &both.rows[1], &b);
}
per_backend!(
    BackendChoice,
    two_frames_of_one_size_embed_differently,
    two_frames_of_one_size_embed_differently_wgpu,
    two_frames_of_one_size_embed_differently_rocm,
    two_frames_of_one_size_embed_differently_ptx
);

/// The ids of the SmolVLM fixture tokenizer: letters and digits first (`b c d` is the question),
/// then the idefics3 markers the front end's prompt names, then a filler up to the vocabulary.
mod smolvlm_ids {
    pub(super) const QUESTION: [u32; 3] = [1, 2, 3];
    pub(super) const USER: u32 = 36;
    pub(super) const ASSISTANT: u32 = 37;
    pub(super) const IM_START: u32 = 38;
    pub(super) const END_OF_UTTERANCE: u32 = 39;
    pub(super) const FAKE: u32 = 40;
    pub(super) const GLOBAL: u32 = 41;
    pub(super) const IMAGE: u32 = 42;
    /// `<row_1_col_1>`, `<row_1_col_2>`, `<row_2_col_1>`, `<row_2_col_2>`: a 2x2 tile grid.
    pub(super) const fn tile(row: usize, col: usize) -> u32 {
        43 + (row * 2 + col) as u32
    }
    pub(super) const VOCAB: usize = 48;
}

/// The SmolVLM fixture tokenizer: [`smolvlm_ids`]' words, with every marker a special token so the
/// prompt's `<image>` runs split out of the surrounding text.
fn smolvlm_tokenizer() -> Tokenizer {
    use smolvlm_ids as ids;
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    let markers = [
        "User:",
        "Assistant:",
        "<|im_start|>",
        "<end_of_utterance>",
        "<fake_token_around_image>",
        "<global-img>",
        "<image>",
        "<row_1_col_1>",
        "<row_1_col_2>",
        "<row_2_col_1>",
        "<row_2_col_2>",
    ];
    words.extend(markers.map(String::from));
    words.push("filler".to_string());
    assert_eq!(words.len(), ids::VOCAB);
    assert_eq!(words[ids::IMAGE as usize], "<image>");
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
    let specials: Vec<tokenizers::AddedToken> = markers[2..]
        .iter()
        .map(|marker| tokenizers::AddedToken::from(*marker, true))
        .collect();
    tokenizer.add_special_tokens(&specials);
    tokenizer
}

/// The registry's SmolVLM fixture: a SmolVLM of 8-pixel frames (16 patches, 4 visual tokens of the
/// 24-wide llama text tower), as the registry builds it in memory.
fn smolvlm_fixture_handle() -> (Arc<ModelHandle>, Fixture) {
    let registry = Registry::builtin().unwrap();
    let fixture = poot_models::test_support::vlm::smolvlm_fixture();
    let handle =
        ModelHandle::from_checkpoint(&fixture.raw(), fixture.store.clone(), &registry, |_| {
            Ok(fixture_text())
        })
        .unwrap();
    (Arc::new(handle), fixture)
}

/// The SmolVLM fixture as an HF directory with the checkpoint's `config.json`,
/// `preprocessor_config.json`, tokenizer and chat template, loaded the way a server loads it. The text
/// tower's weights are scaled up so the image rows reach the greedy token.
fn smolvlm_checkpoint() -> (poot_test_util::UniqueTempPath, Arc<ModelHandle>) {
    let (_, fixture) = smolvlm_fixture_handle();
    let text_keys = |name: &str| name.starts_with("model.text_model.") || name == "lm_head.weight";
    let mut text = WeightStore::builder();
    for (key, entry) in fixture.store.iter().filter(|(k, _)| text_keys(k.as_str())) {
        text.insert(key.clone(), entry.clone()).unwrap();
    }
    let text = scaled(&text.build(), 8.0);
    let tensors: Vec<ArchiveTensor> = fixture
        .store
        .iter()
        .map(|(key, entry)| {
            let entry = text.get(key.as_str()).unwrap_or(entry);
            let WeightEntry::Dense(dense) = entry else {
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
    let mut config = fixture.config.clone();
    config["text_config"]["eos_token_id"] = NO_EOS.into();
    let dir = write_hf_checkpoint("smolvlm", &config, &tensors);
    smolvlm_tokenizer()
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": poot_test_util::smolvlm::CHAT_TEMPLATE }).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        fixture.preprocessor.unwrap().to_string(),
    )
    .unwrap();
    let handle = ModelHandle::load(
        &dir,
        &Registry::builtin().unwrap(),
        &crate::driver::LoadOptions::default(),
    )
    .unwrap();
    (dir, Arc::new(handle))
}

/// An HF directory carries its front end's `preprocessor_config.json` to the registry entry: the
/// loaded handle holds the vision tower and the front end. Mutation: build `RawConfig::HfJson` in
/// `ModelHandle::load_hf` with `preprocessor: None`; the load is refused (`preprocessor_config` missing).
#[test]
fn a_vlm_directory_loads_with_its_tower_and_front_end() {
    let (_dir, handle) = smolvlm_checkpoint();
    assert_eq!(handle.towers().len(), 1);
    assert!(handle.front_end().is_some());
}

/// The towers of `handle` on `executors` (one per tower), keeping the inputs of every row they encode:
/// the frames the front end made, which the by-hand requests below feed to the CPU oracle.
struct RecordingTowers {
    drivers: crate::TowerDrivers,
    rows: Vec<RequestInputs>,
}

impl RecordingTowers {
    fn open(
        handle: &ModelHandle,
        executor: Box<dyn poot_executor::Executor>,
        positions: usize,
    ) -> Self {
        let mut executor = Some(executor);
        let drivers = crate::TowerDrivers::open(handle, |_, _| {
            Ok::<_, DriverError>((
                executor.take().expect("one tower"),
                options(positions, positions),
            ))
        })
        .unwrap();
        Self {
            drivers,
            rows: Vec::new(),
        }
    }
}

impl poot_models::multimodal::front_end::Towers for RecordingTowers {
    fn encode(
        &mut self,
        tower: poot_models::multimodal::front_end::TowerId,
        inputs: &RequestInputs,
    ) -> Result<Vec<f32>, poot_models::multimodal::front_end::TowerError> {
        self.rows.push(inputs.clone());
        self.drivers.encode(tower, inputs)
    }
}

/// The 16x16 image of `value` everywhere.
fn solid_image(value: u8, side: usize) -> poot_models::multimodal::front_end::Image {
    poot_models::multimodal::front_end::Image::from_rgb8(side, side, vec![value; side * side * 3])
        .unwrap()
}

/// A 16x16 image black in the top-left quadrant and white elsewhere.
fn quadrant_image(black: usize) -> poot_models::multimodal::front_end::Image {
    let rgb = (0..16 * 16)
        .flat_map(|p| {
            let (y, x) = (p / 16, p % 16);
            let v = if (y / 8) * 2 + x / 8 == black { 0 } else { 255 };
            [v, v, v]
        })
        .collect();
    poot_models::multimodal::front_end::Image::from_rgb8(16, 16, rgb).unwrap()
}

/// What the front end lowers `image` and `question` to, with the frames it encoded.
fn lower(
    handle: &ModelHandle,
    towers: &mut RecordingTowers,
    image: &poot_models::multimodal::front_end::Image,
    question: &str,
) -> poot_models::multimodal::front_end::Prepared {
    use poot_models::multimodal::front_end::{ChatTurn, Media};
    towers.rows.clear();
    handle
        .front_end()
        .expect("the SmolVLM fixture has a front end")
        .prepare(
            &[ChatTurn::user(1, question)],
            &mut [Media::Image(image)],
            towers,
            handle.text(),
        )
        .expect("the front end lowers the request")
}

/// The prompt a SmolVLM front end builds for a 16x16 image (a 2x2 tile grid and a global frame, four
/// visual tokens each) asking `question`, and the vision oracle's rows for the `frames` it encoded,
/// by hand from token ids.
fn smolvlm_request(
    tower: &ModelHandle,
    frames: &[RequestInputs],
    question: &[u32],
) -> (Vec<u32>, RequestInputs) {
    use smolvlm_ids as ids;

    assert_eq!(frames.len(), 5, "four tiles and the global frame");
    let mut prompt = vec![ids::IM_START, ids::USER];
    let mut positions = Vec::new();
    let mut placeholders = |prompt: &mut Vec<u32>| {
        for _ in 0..4 {
            positions.push(prompt.len());
            prompt.push(ids::IMAGE);
        }
    };
    for row in 0..2 {
        for col in 0..2 {
            prompt.extend([ids::FAKE, ids::tile(row, col)]);
            placeholders(&mut prompt);
        }
    }
    prompt.extend([ids::FAKE, ids::GLOBAL]);
    placeholders(&mut prompt);
    prompt.push(ids::FAKE);
    prompt.extend(question);
    prompt.extend([ids::END_OF_UTTERANCE, ids::ASSISTANT]);
    let rows: Vec<f32> = frames
        .iter()
        .flat_map(|inputs| {
            let Some(InputValue::Row(pixels)) = inputs.get(poot_models::io::InputId::new("pixels"))
            else {
                panic!("a frame carries its pixels")
            };
            oracle(
                tower,
                ProgramId::ENCODE,
                encode_shape(1, 16),
                &[(
                    named("pixels"),
                    HostTensor::f32(vec![1, 3, 8, 8], pixels.as_f32().unwrap().to_vec()),
                )],
            )
        })
        .collect();
    (prompt, embed_inputs(24, positions, rows))
}

/// POOT-1046 (the SmolVLM front end on the required lanes): the front end of a fixture checkpoint
/// loaded from disk lowers an image and a question to a request whose generation equals the same text
/// tower's on a prompt built by hand from token ids with the vision oracle's rows for the frames (tier
/// 2): the front end's prompt, placeholder positions, per-frame encode and scatter all have to agree.
/// A different image captions differently (the rows reach the logits). Mutation: bind zeroed
/// embedding rows in `driver/bind.rs`; both captions collapse to the zero-row continuation and the row
/// goes red.
fn smolvlm_caption_matches_the_oracle_rows(choice: BackendChoice) {
    let (_dir, handle) = smolvlm_checkpoint();
    let tower = Arc::clone(&handle.towers()[0].1);
    let (Some(vision_executor), Some(mut caption_driver), Some(mut text_driver)) = (
        open(choice),
        driver(choice, &handle, 64, 128),
        driver(choice, &handle, 64, 128),
    ) else {
        return;
    };
    let mut towers = RecordingTowers::open(&handle, vision_executor, 16);
    let max_new = 8;
    let question = smolvlm_ids::QUESTION;
    let mut compare = |image: &poot_models::multimodal::front_end::Image| -> Vec<u32> {
        let prepared = lower(&handle, &mut towers, image, "b c d");
        let got = generate_inputs(
            &mut caption_driver,
            prepared.prompt,
            max_new,
            prepared.inputs,
        );
        let (prompt, embeds) = smolvlm_request(&tower, &towers.rows, &question);
        assert_eq!(
            got,
            generate_inputs(&mut text_driver, prompt, max_new, embeds),
            "the caption equals the oracle-row run"
        );
        got
    };
    // A 16x16 image splits into a 2x2 tile grid and a global frame. The two images are the extremes
    // of the pixel range, so their rows differ as much as the fixture's weights let them.
    let (got_first, got_second) = (compare(&solid_image(0, 16)), compare(&solid_image(255, 16)));
    assert_ne!(got_first, got_second, "the image rows reach the logits");
    // Solid images give every tile the same rows; an image with one black quadrant gives that tile
    // its own, so the order the frames go into the prompt is part of the caption.
    for quadrant in 0..4 {
        compare(&quadrant_image(quadrant));
    }
}
per_backend!(
    BackendChoice,
    smolvlm_caption_matches_the_oracle_rows,
    smolvlm_caption_matches_the_oracle_rows_wgpu,
    smolvlm_caption_matches_the_oracle_rows_rocm,
    smolvlm_caption_matches_the_oracle_rows_ptx
);

/// POOT-1076: the SmolVLM text tower on the paged pool serves several prompts with embedding rows
/// in one batched run, each token for token the single-row contiguous `generate_with_inputs`: three
/// images' full prompts (equal length, different rows), a short prompt with its own rows at other
/// positions, and a plain token prompt of another length, all stepped together so their prefill
/// pieces share embeds-carrying steps. Mutation: bind row 0's embeds values for every row (in
/// `bind::token_values`, read the first lane's values for all lanes); the other lanes read the wrong
/// image's rows and the row goes red.
fn smolvlm_batched_paged_embeds_match_single_row_runs(choice: BackendChoice) {
    use smolvlm_ids as ids;

    let (_dir, handle) = smolvlm_checkpoint();
    let tower = Arc::clone(&handle.towers()[0].1);
    let (Some(vision_executor), Some(mut reference), Some(driver)) = (
        open(choice),
        driver(choice, &handle, 16, 128),
        bare_driver(choice, &handle, 16, 128),
    ) else {
        return;
    };
    let mut towers = RecordingTowers::open(&handle, vision_executor, 16);
    let max_new = 6;
    let images = [solid_image(0, 16), solid_image(255, 16), quadrant_image(0)];
    let mut requests: Vec<(Vec<u32>, RequestInputs)> = images
        .iter()
        .map(|image| {
            lower(&handle, &mut towers, image, "b c d");
            smolvlm_request(&tower, &towers.rows, &ids::QUESTION)
        })
        .collect();
    // A short prompt: four placeholders at positions 2..6 take the first image's first four rows.
    let (_, first) = &requests[0];
    let Some(InputValue::Tokens(first_rows)) = first.get(EMBEDS) else {
        panic!("the image prompt carries embedding rows")
    };
    let rows = first_rows.data().as_f32().unwrap()[..4 * 24].to_vec();
    requests.push((
        vec![
            ids::IM_START,
            ids::USER,
            ids::IMAGE,
            ids::IMAGE,
            ids::IMAGE,
            ids::IMAGE,
            ids::QUESTION[0],
        ],
        embed_inputs(24, vec![2, 3, 4, 5], rows),
    ));
    // A plain token prompt of yet another length shares the batch.
    requests.push((
        (0..21).map(|i| (i * 5 + 1) % 36).collect::<Vec<u32>>(),
        RequestInputs::none(),
    ));

    let want: Vec<Vec<u32>> = requests
        .iter()
        .map(|(prompt, inputs)| {
            generate_inputs(&mut reference, prompt.clone(), max_new, inputs.clone())
        })
        .collect();
    assert_ne!(want[0], want[1], "the image rows reach the logits");

    let pool = pool_shape(20, 5);
    let mut driver = prepare_serving(driver, pool, &[5], &[Head::GREEDY], &[]);
    driver
        .prepare_carried(
            &ServingShapes {
                layout: Layout::Paged(pool),
                rows: &[nz(5)],
                heads: &[Head::GREEDY],
                warm: Warm::Admitted,
                windows: &[],
                adapters: &[],
            },
            &[EMBEDS],
        )
        .unwrap();
    let mut lanes: Vec<Lane> = requests
        .iter()
        .map(|(prompt, inputs)| Lane::open_embeds(&mut driver, prompt, inputs.clone(), max_new).0)
        .collect();
    run_to_end(&mut driver, &mut lanes);
    for (i, lane) in lanes.iter().enumerate() {
        assert_eq!(lane.out, want[i], "request {i} batched equals its own run");
    }
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
}
per_backend!(
    BackendChoice,
    smolvlm_batched_paged_embeds_match_single_row_runs,
    smolvlm_batched_paged_embeds_match_single_row_runs_wgpu,
    smolvlm_batched_paged_embeds_match_single_row_runs_rocm,
    smolvlm_batched_paged_embeds_match_single_row_runs_ptx
);

/// POOT-1075: requests lowered by the SmolVLM front end (preprocess, per-frame vision encode, prompt,
/// placeholder positions) and served together on a paged text driver equal each request's own
/// single-row run token for token. The requests differ in image (solid black, solid white, one black
/// quadrant, a single-frame 8x8 image) and in question length, so their prompts have four lengths and
/// their rows differ. Mutation: in `Idefics3FrontEnd::encode_frames` return the first frame's rows for
/// every frame (`encoded[0]` cloned per frame); every request carries the same rows, the images stop
/// reaching the logits and the row goes red.
fn smolvlm_front_end_requests_served_batched_equal_single_row_captions(choice: BackendChoice) {
    let (_dir, handle) = smolvlm_checkpoint();
    let tower = Arc::clone(&handle.towers()[0].1);
    // Every executor opens before any runs (the ROCm runtime does not survive a second open after a drop).
    let (Some(vision_executor), Some(batched), Some(single), Some(reference)) =
        (open(choice), open(choice), open(choice), open(choice))
    else {
        return;
    };
    let mut towers = RecordingTowers::open(&handle, vision_executor, 16);
    let driver = Driver::new(Arc::clone(&handle), batched, options(16, 128)).unwrap();
    let mut single = Driver::new(Arc::clone(&handle), single, options(16, 128))
        .unwrap()
        .serve(Layout::Contiguous);
    let mut reference = Driver::new(Arc::clone(&handle), reference, options(16, 128))
        .unwrap()
        .serve(Layout::Contiguous);

    let max_new = 6;
    let (black, white, quadrant, small) = (
        solid_image(0, 16),
        solid_image(255, 16),
        quadrant_image(0),
        solid_image(255, 8),
    );
    let cases = [
        (&black, "b c d"),
        (&white, "b c d e f"),
        (&quadrant, "b"),
        (&small, "b c d e f g h"),
    ];
    let prepared: Vec<_> = cases
        .iter()
        .map(|(image, question)| lower(&handle, &mut towers, image, question))
        .collect();
    let want: Vec<Vec<u32>> = prepared
        .iter()
        .map(|p| generate_inputs(&mut single, p.prompt.clone(), max_new, p.inputs.clone()))
        .collect();
    assert_ne!(want[0], want[1], "the image rows reach the logits");
    // The front end's lowering is also checked against the vision oracle's rows on a prompt built by
    // hand (the 16x16 images; a shared path would otherwise be equal to itself on both sides).
    for (i, (image, question)) in cases.iter().take(3).enumerate() {
        lower(&handle, &mut towers, image, question);
        let question_ids: Vec<u32> = question
            .split(' ')
            .map(|word| u32::from(word.as_bytes()[0] - b'a'))
            .collect();
        let (prompt, embeds) = smolvlm_request(&tower, &towers.rows, &question_ids);
        assert_eq!(
            generate_inputs(&mut reference, prompt, max_new, embeds),
            want[i],
            "request {i}: caption equals the oracle-row run"
        );
    }
    let lengths: std::collections::BTreeSet<usize> =
        prepared.iter().map(|p| p.prompt.len()).collect();
    assert_eq!(lengths.len(), 4, "four prompt lengths share the batch");

    let pool = pool_shape(24, 4);
    let mut driver = prepare_serving(driver, pool, &[4], &[Head::GREEDY], &[]);
    driver
        .prepare_carried(
            &ServingShapes {
                layout: Layout::Paged(pool),
                rows: &[nz(4)],
                heads: &[Head::GREEDY],
                warm: Warm::Admitted,
                windows: &[],
                adapters: &[],
            },
            &[EMBEDS],
        )
        .unwrap();
    let mut lanes: Vec<Lane> = prepared
        .iter()
        .map(|p| Lane::open_embeds(&mut driver, &p.prompt, p.inputs.clone(), max_new).0)
        .collect();
    run_to_end(&mut driver, &mut lanes);
    for (i, lane) in lanes.iter().enumerate() {
        assert_eq!(lane.out, want[i], "request {i} batched equals its caption");
    }
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
}
per_backend!(
    BackendChoice,
    smolvlm_front_end_requests_served_batched_equal_single_row_captions,
    smolvlm_front_end_requests_served_batched_equal_single_row_captions_wgpu,
    smolvlm_front_end_requests_served_batched_equal_single_row_captions_rocm,
    smolvlm_front_end_requests_served_batched_equal_single_row_captions_ptx
);

/// POOT-1076: the paged prefix cache keys on tokens alone, so a prompt with embedding rows reuses no
/// block and registers none. Two images give prompts of identical tokens and different rows: the
/// second finds nothing to reuse and equals its own single-row run; the first's release reports
/// `RequestInputs`; and a plain prompt of the same tokens afterwards reuses nothing either. Mutation:
/// admit the embeds prompt through `admit_prefix_for` and register it on release; the second image
/// reuses the first's two blocks (and its tokens), and the row goes red.
fn paged_embeds_prompts_neither_reuse_nor_register_prefix_blocks(choice: BackendChoice) {
    let (_dir, handle) = smolvlm_checkpoint();
    let tower = Arc::clone(&handle.towers()[0].1);
    let (Some(vision_executor), Some(mut reference), Some(driver)) = (
        open(choice),
        driver(choice, &handle, 16, 128),
        bare_driver(choice, &handle, 16, 128),
    ) else {
        return;
    };
    let mut towers = RecordingTowers::open(&handle, vision_executor, 16);
    let max_new = 6;
    lower(&handle, &mut towers, &solid_image(0, 16), "b c d");
    let (prompt, first) = smolvlm_request(&tower, &towers.rows, &smolvlm_ids::QUESTION);
    lower(&handle, &mut towers, &solid_image(255, 16), "b c d");
    let (same_prompt, second) = smolvlm_request(&tower, &towers.rows, &smolvlm_ids::QUESTION);
    assert_eq!(prompt, same_prompt, "equal tokens, different rows");
    let single = |driver: &mut Driver<Serving>, inputs: &RequestInputs| {
        generate_inputs(driver, prompt.clone(), max_new, inputs.clone())
    };
    let (want_first, want_second) = (
        single(&mut reference, &first),
        single(&mut reference, &second),
    );
    assert_ne!(
        want_first, want_second,
        "the two images continue differently"
    );
    let want_plain = generate(&mut reference, &prompt, max_new, Sampler::greedy());

    let pool = pool_shape(12, 2);
    let mut driver = prepare_serving(driver, pool, &[1], &[Head::GREEDY], &[]);
    driver
        .prepare_carried(
            &ServingShapes {
                layout: Layout::Paged(pool),
                rows: &[nz(1)],
                heads: &[Head::GREEDY],
                warm: Warm::Admitted,
                windows: &[],
                adapters: &[],
            },
            &[EMBEDS],
        )
        .unwrap();
    let run = |driver: &mut Driver<Serving>, inputs: &RequestInputs| {
        let (lane, reused) = Lane::open_embeds(driver, &prompt, inputs.clone(), max_new);
        let mut lanes = [lane];
        run_to_end(driver, &mut lanes);
        let outcome = driver.release(lanes[0].seq, Release::Finished).unwrap();
        (lanes[0].out.clone(), reused, outcome)
    };
    let (tokens, reused, outcome) = run(&mut driver, &first);
    assert_eq!((tokens, reused), (want_first, 0));
    assert_eq!(outcome, PrefixOutcome::Skipped(PrefixSkip::RequestInputs));
    let (tokens, reused, _) = run(&mut driver, &second);
    assert_eq!(reused, 0, "another image's rows reuse no block");
    assert_eq!(tokens, want_second);
    let (tokens, reused, _) = run(&mut driver, &RequestInputs::none());
    assert_eq!(reused, 0, "the embeds prompts registered nothing");
    assert_eq!(tokens, want_plain);
}
per_backend!(
    BackendChoice,
    paged_embeds_prompts_neither_reuse_nor_register_prefix_blocks,
    paged_embeds_prompts_neither_reuse_nor_register_prefix_blocks_wgpu,
    paged_embeds_prompts_neither_reuse_nor_register_prefix_blocks_rocm,
    paged_embeds_prompts_neither_reuse_nor_register_prefix_blocks_ptx
);

/// An encode request is refused while a sequence is open (its cache would alias the sequence's),
/// and a request of mixed kinds or an empty row is a typed refusal.
fn encode_refuses_what_it_cannot_batch(choice: BackendChoice) {
    let (handle, _) = encoder_family_handle("bert", None);
    let Some(mut driver) = token_encoder(choice, &handle, 16, 16) else {
        return;
    };
    let refusal = |error: DriverError| match error {
        DriverError::InvalidRequest(request) => request,
        other => panic!("{other:?}"),
    };
    let none = RequestInputs::none();
    let empty = driver.encode(&EncodeRequest {
        rows: &[EncodeRow {
            tokens: Some(&[]),
            inputs: &none,
        }],
    });
    assert_eq!(
        refusal(empty.unwrap_err()),
        InvalidRequest::Encode(crate::driver::error::EncodeError::EmptyRow { row: 0 })
    );
    let too_long = driver.encode(&EncodeRequest {
        rows: &[EncodeRow {
            tokens: Some(&[1; 17]),
            inputs: &none,
        }],
    });
    assert_eq!(
        refusal(too_long.unwrap_err()),
        InvalidRequest::Encode(crate::driver::error::EncodeError::RowTooLong {
            row: 0,
            tokens: 17,
            max: 16
        }),
        "a row past the prepared length is a typed refusal"
    );

    // A step has one kind of row: a vision tower's token row beside a frame row (no tokens) is mixed.
    let (smolvlm, _) = smolvlm_fixture_handle();
    let vision = Arc::clone(&smolvlm.towers()[0].1);
    let Some(mut vision_driver) = bare_driver(choice, &vision, 16, 16) else {
        return;
    };
    let pixels = frame_inputs(&[0.0; 3 * 8 * 8]);
    let mixed = vision_driver.encode(&EncodeRequest {
        rows: &[
            EncodeRow {
                tokens: Some(&ENCODE_SHORT),
                inputs: &pixels,
            },
            EncodeRow {
                tokens: None,
                inputs: &pixels,
            },
        ],
    });
    assert_eq!(
        refusal(mixed.unwrap_err()),
        InvalidRequest::Encode(crate::driver::error::EncodeError::Mixed)
    );
}
per_backend!(
    BackendChoice,
    encode_refuses_what_it_cannot_batch,
    encode_refuses_what_it_cannot_batch_wgpu,
    encode_refuses_what_it_cannot_batch_rocm,
    encode_refuses_what_it_cannot_batch_ptx
);

// ---------------------------------------------------------------------------------------------
// POOT-571: Qwen4Exp through the driver. The recorded-token and first-step rows above cover the
// fixtures (dense experts, packed E4M3 experts, the finetune-sized config) and SC-008's granule row
// (the 7-token prompt is whole-granule prefill plus decode steps); the rows below are the PLE
// histories (SC-006), the weights' residency (SC-007) and the PLE token guard (SC-009).
// ---------------------------------------------------------------------------------------------

fn qwen4exp_handle() -> Arc<ModelHandle> {
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "qwen4exp")
        .unwrap();
    recorded_family_handle(recorded)
}

/// Three six-token prompts with different histories for the PLE (a plain one, one with the
/// end-of-sequence id inside its window, one opening with two), and the independent reference's
/// first-step top five for each (`qwen38::tests::the_ple_history_literals_are_the_references_first_step`
/// pins these same numbers to the plain-loop f64 reference, which shares no code with the tracer).
const PLE_HISTORIES: [[u32; 6]; 3] = [
    [3, 17, 40, 8, 25, 11],
    [8, 40, 5, 22, 47, 9],
    [47, 47, 3, 9, 22, 31],
];
const PLE_HISTORIES_TOP5: [[(u32, f64); 5]; 3] = [
    [
        (25, -1.903_203_6),
        (33, -2.256_853_4),
        (15, -3.129_722),
        (21, -3.237_417),
        (1, -3.352_075_7),
    ],
    [
        (35, -2.689_749_4),
        (25, -2.868_450_9),
        (30, -2.914_160_6),
        (26, -2.922_117_3),
        (2, -3.010_46),
    ],
    [
        (32, -2.643_691_5),
        (21, -2.647_688_7),
        (38, -2.836_676_7),
        (23, -2.846_178_8),
        (29, -2.876_545),
    ],
];

/// SC-006: the three histories go through the one binder (the family's state, the PLE's carried
/// history and conv window among it) on the device, each giving the reference's first token and
/// top-five log-probabilities (within `1e-3`), and the three differ. Mutation: index the
/// per-layer-embedding table with the previous step's id (`components::ple` hashes the carried
/// history as this step's ids); the first prompt's top five leaves the reference's and the row goes
/// red.
fn qwen4exp_ple_histories_match_the_reference(choice: BackendChoice) {
    ple_histories_match_the_reference(choice, &qwen4exp_handle());
}

/// `handle`'s first-step top five on the three [`PLE_HISTORIES`] prompts against the reference's.
fn ple_histories_match_the_reference(choice: BackendChoice, handle: &Arc<ModelHandle>) {
    let Some(mut driver) = driver(choice, handle, 8, CAPACITY) else {
        return;
    };
    for (prompt, want) in PLE_HISTORIES.iter().zip(PLE_HISTORIES_TOP5) {
        let generated = driver
            .generate(
                GenerateRequest {
                    prompt: prompt.to_vec(),
                    max_new: 1,
                    sampler: Sampler::greedy().with_logprobs(5),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .unwrap();
        assert_eq!(generated.tokens, [want[0].0], "{prompt:?} on {choice:?}");
        let got = &generated.logprobs[0].top;
        assert_eq!(
            got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            "{prompt:?} on {choice:?}: the five most likely tokens"
        );
        for ((token, got), (_, want)) in got.iter().zip(want) {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-3,
                "{prompt:?} on {choice:?}: token {token} has log-probability {got}, the reference gives {want}"
            );
        }
    }
}
per_backend!(
    BackendChoice,
    qwen4exp_ple_histories_match_the_reference,
    qwen4exp_ple_histories_match_the_reference_wgpu,
    qwen4exp_ple_histories_match_the_reference_rocm,
    qwen4exp_ple_histories_match_the_reference_ptx
);

/// The qwen4exp checkpoint with its n-gram table stored the way the official release does: three
/// row-range shards of different sizes under a scale of one.
fn qwen4exp_sharded_handle() -> Arc<ModelHandle> {
    const TABLE: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    let recorded = RECORDED
        .iter()
        .find(|recorded| recorded.family == "qwen4exp")
        .unwrap();
    recorded_family_handle_with(recorded, &|key, weight| {
        if key != format!("{TABLE}.weight") {
            return vec![(key.to_string(), weight.clone())];
        }
        let WeightEntry::Dense(dense) = weight else {
            panic!("the fixture's n-gram table is dense");
        };
        let (rows, width) = (dense.shape()[0], dense.shape()[1]);
        let bytes = dense.bytes().as_slice();
        let row_bytes = bytes.len() / rows;
        let counts = [rows / 4, rows / 2, rows - rows / 4 - rows / 2];
        let mut at = 0;
        let mut out = Vec::new();
        for (ordinal, count) in counts.into_iter().enumerate() {
            let cut = bytes[at * row_bytes..(at + count) * row_bytes].to_vec();
            let shard = DenseWeight::try_new(dense.dtype(), vec![count, width], cut.into());
            out.push((
                format!("{TABLE}.shard_{ordinal}.weight"),
                WeightEntry::Dense(shard.unwrap()),
            ));
            at += count;
        }
        let one = DenseWeight::try_new(
            poot_tensor::DType::F32,
            vec![],
            1.0f32.to_le_bytes().to_vec().into(),
        );
        out.push((
            format!("{TABLE}.weight_scale"),
            WeightEntry::Dense(one.unwrap()),
        ));
        out
    })
}

/// A PLE table stored as shards (three row ranges of different sizes under a scale of one) gives the
/// reference's first-step top five on the device, the literals the single table is held to: the
/// per-shard masked gather keeps the ids I32 and selects the owning shard's row. (The single table's
/// run stays in `qwen4exp_ple_histories_match_the_reference`: ROCm cannot reopen an executor in one
/// process, so both layouts are not run in one row; their bit-for-bit equality is the component row
/// `a_sharded_f32_table_equals_the_single_table_bit_for_bit`.) Mutation: offset a shard's local row by
/// one in `components::ple`'s `Table::lookup`; the rows on the shard edges read their neighbours and
/// the top five leaves the reference's.
fn qwen4exp_sharded_ple_table_matches_the_reference(choice: BackendChoice) {
    ple_histories_match_the_reference(choice, &qwen4exp_sharded_handle());
}
per_backend!(
    BackendChoice,
    qwen4exp_sharded_ple_table_matches_the_reference,
    qwen4exp_sharded_ple_table_matches_the_reference_wgpu,
    qwen4exp_sharded_ple_table_matches_the_reference_rocm,
    qwen4exp_sharded_ple_table_matches_the_reference_ptx
);

/// SC-006, with no device: the CPU oracle over the family's traced graph gives the same literals, so
/// a tracer that reads the PLE's history wrongly leaves them on every machine.
#[test]
fn qwen4exp_ple_histories_match_the_reference_on_the_cpu_oracle() {
    let handle = qwen4exp_handle();
    for (prompt, want) in PLE_HISTORIES.iter().zip(PLE_HISTORIES_TOP5) {
        let got = oracle_top5(&handle, prompt, CAPACITY);
        for ((token, got), (want_token, want)) in got.iter().zip(want) {
            assert_eq!(
                *token, want_token,
                "{prompt:?}: the five most likely tokens"
            );
            assert!(
                (got - want).abs() <= 1e-3,
                "{prompt:?}: token {token} has log-probability {got}, the reference gives {want}"
            );
        }
    }
}

/// The weight allocations and live weight bytes the executor holds.
fn resident_weights<S: State>(driver: &Driver<S>) -> (u64, u64) {
    driver
        .executor_stats()
        .memory
        .iter()
        .find(|(role, _)| *role == poot_runtime_common::BufferRole::Weight)
        .map(|(_, snapshot)| (snapshot.allocations, snapshot.live_bytes))
        .unwrap()
}

/// SC-007 (R475-002): the BF16 owners stay resident as stored and are not re-widened per step.
/// After the admitted entries are prepared (the load), eight decode steps through the driver leave
/// the executor's weight allocations and live bytes as they were, and the live bytes are those of the
/// stored owners (never more: an F32 copy of a BF16 owner is twice its bytes), and more than the
/// stored bytes minus the embedding table (which a backend may keep on the host). Mutation: rebind
/// every BF16 owner as F32 in the body (`Weight::declare` casts a dense BF16 const to F32 when it
/// declares it); the resident bytes pass the stored bytes and the row goes red.
fn qwen4exp_bf16_owners_are_not_widened_per_step(choice: BackendChoice) {
    use poot_quant::weights::HandleFormat;
    let handle = qwen4exp_handle();
    let stored: u64 = handle
        .model()
        .weights()
        .iter()
        .map(|(_, _, h)| match h.format {
            HandleFormat::Dense(dtype) => {
                (h.shape.iter().product::<usize>() * dtype.byte_size()) as u64
            }
            HandleFormat::Packed(_) => panic!("the fixture is dense"),
        })
        .sum();
    let embedding = (VOCAB * 32 * 2) as u64;
    let Some(driver) = bare_driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let mut driver = prepare_admitted(driver).unwrap();
    let loaded = resident_weights(&driver);
    assert!(
        loaded.1 <= stored,
        "{choice:?}: {} resident weight bytes exceed the {stored} bytes the owners are stored in",
        loaded.1
    );
    assert!(
        loaded.1 >= stored - embedding,
        "{choice:?}: {} resident weight bytes, the owners are {stored} stored",
        loaded.1
    );
    let tokens = generate(&mut driver, &PROMPT[..6], 8, Sampler::greedy());
    assert_eq!(tokens.len(), 8);
    assert_eq!(
        resident_weights(&driver),
        loaded,
        "{choice:?}: eight decode steps upload nothing"
    );
}
per_backend!(
    BackendChoice,
    qwen4exp_bf16_owners_are_not_widened_per_step,
    qwen4exp_bf16_owners_are_not_widened_per_step_wgpu,
    qwen4exp_bf16_owners_are_not_widened_per_step_rocm,
    qwen4exp_bf16_owners_are_not_widened_per_step_ptx
);

/// SC-009 (dfamily.md F4): a token outside the vocabulary is refused at the driver boundary with the
/// typed `TokenOutOfVocab`, before a device step runs. The PLE's device-side range witness (declared by
/// the graph, pinned by `poot-models` `an_out_of_vocabulary_token_is_a_typed_validation_fault`) is
/// defence in depth for a caller that feeds an executor directly: the driver is the only producer of
/// a model's token inputs, validates every prompt id, and generates only ids below the vocabulary,
/// so the witness cannot fire through it. Mutation: drop the vocabulary check from `check_admission`;
/// the PLE guard fires instead (a validation fault) and the row goes red.
fn qwen4exp_out_of_vocab_token_is_refused_at_the_driver(choice: BackendChoice) {
    let handle = qwen4exp_handle();
    let Some(mut driver) = driver(choice, &handle, 8, CAPACITY) else {
        return;
    };
    let error = driver
        .generate(
            GenerateRequest {
                // One id past the 48-token vocabulary.
                prompt: vec![3, 48],
                max_new: 2,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .expect_err("a token outside the vocabulary must not publish tokens");
    assert!(
        matches!(
            error,
            DriverError::InvalidRequest(InvalidRequest::TokenOutOfVocab {
                token: 48,
                at: 1,
                vocab: 48
            })
        ),
        "{error:?}"
    );
}
per_backend!(
    BackendChoice,
    qwen4exp_out_of_vocab_token_is_refused_at_the_driver,
    qwen4exp_out_of_vocab_token_is_refused_at_the_driver_wgpu,
    qwen4exp_out_of_vocab_token_is_refused_at_the_driver_rocm,
    qwen4exp_out_of_vocab_token_is_refused_at_the_driver_ptx
);

/// SC-008 (the driver's half): a prefill chunk the family's granule does not divide is a typed
/// option error before anything is traced, compiled or dispatched. Mutation: drop the granule check
/// from the options; the driver is built and the first generate asks the model for an off-granule
/// prefill, which the model refuses, so the assertion on the construction error goes red.
#[test]
fn qwen4exp_chunk_off_the_granule_is_a_typed_option_error_wgpu() {
    use crate::driver::error::InvalidOptions;
    let handle = qwen4exp_handle();
    let Some(executor) = open(BackendChoice::Wgpu) else {
        return;
    };
    let error = Driver::new(handle, executor, options(3, CAPACITY))
        .map(|_| ())
        .expect_err("a chunk of three is not a whole number of two-position blocks");
    assert!(
        matches!(
            error,
            DriverError::InvalidOptions(InvalidOptions::ChunkNotGranuleAligned {
                chunk: 3,
                granule: 2
            })
        ),
        "{error:?}"
    );
}

/// The greedy tokens of four distinct prompts on qwen2, 24 new each, from single-row `Driver::generate`
/// runs (the `PROMPT` row is `FIXTURE_GREEDY`, recorded when POOT-734 landed; the other three were
/// recorded from the same single-row path when POOT-579 added the four-row row).
const FOUR_ROW_GREEDY: [[u32; 24]; 4] = [
    [
        23, 41, 2, 32, 3, 2, 41, 41, 25, 41, 36, 23, 35, 19, 5, 36, 3, 23, 35, 18, 44, 44, 2, 32,
    ],
    [
        24, 5, 1, 14, 42, 32, 45, 35, 41, 25, 35, 33, 2, 26, 35, 12, 35, 25, 35, 20, 45, 37, 10, 10,
    ],
    [
        20, 38, 20, 19, 10, 19, 30, 22, 0, 5, 22, 0, 5, 6, 6, 28, 19, 0, 24, 6, 28, 19, 30, 6,
    ],
    FIXTURE_GREEDY,
];

/// POOT-579 SC-002: qwen2's four rows decode paged through the driver in one step per round (the
/// 13-token prompt prefills in pieces beside rows already decoding) and produce the recorded literals
/// of their single-row runs. Distinct prompts, so a row reading another row's block table reads K/V that
/// is not its own. Mutation: swap two rows' block tables in the bind (`read` map); the tokens differ.
fn paged_four_rows_equal_the_recorded_literals(choice: BackendChoice) {
    let handle = family_handle("qwen2", NO_EOS);
    let Some(driver) = serving_driver(choice, &handle) else {
        return;
    };
    let mut driver = prepare_serving(driver, pool_shape(16, 4), &[4], &[Head::GREEDY], &[]);
    let prompts: [&[u32]; 4] = [&LONG, &MEDIUM, &SHORT, &PROMPT];
    let mut lanes: Vec<Lane> = prompts
        .iter()
        .map(|p| Lane::open(&mut driver, p, 24, Sampler::greedy()).0)
        .collect();
    run_to_end(&mut driver, &mut lanes);
    for (i, lane) in lanes.iter().enumerate() {
        assert_eq!(lane.out, FOUR_ROW_GREEDY[i], "row {i}");
    }
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
}
per_backend!(
    BackendChoice,
    paged_four_rows_equal_the_recorded_literals,
    paged_four_rows_equal_the_recorded_literals_wgpu,
    paged_four_rows_equal_the_recorded_literals_rocm,
    paged_four_rows_equal_the_recorded_literals_ptx
);

// ---------------------------------------------------------------------------------------------
// LoRA generations (POOT-579 SC-005): a same-named adapter replaced by new weights while the old
// generation's entries and sequence stay live, each generation's logits equal to its own merged-weight
// reference, and the old generation's unique allocations returned once it completes.
// ---------------------------------------------------------------------------------------------

use poot_load::lora::{LoraAdapter, LoraAdapterConfig, LoraWeight};
use poot_quant::weights::{LoraTarget, WeightKey, WeightView};

const LORA_RANK: usize = 2;

fn lora_module(layer: usize, target: LoraTarget) -> String {
    let (block, name) = match target {
        LoraTarget::AttnQ => ("self_attn", "q_proj"),
        LoraTarget::AttnK => ("self_attn", "k_proj"),
        LoraTarget::AttnV => ("self_attn", "v_proj"),
        LoraTarget::AttnO => ("self_attn", "o_proj"),
        LoraTarget::FfnGate => ("mlp", "gate_proj"),
        LoraTarget::FfnUp => ("mlp", "up_proj"),
        LoraTarget::FfnDown => ("mlp", "down_proj"),
    };
    format!("model.layers.{layer}.{block}.{name}")
}

fn lora_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// An adapter over every targetable projection of every layer of `handle`'s model, `alpha / rank`
/// scaled, deterministic from `seed`.
fn fixture_adapter(handle: &ModelHandle, seed: u64, alpha: f32) -> LoraAdapter {
    let mut weights = HashMap::new();
    for (id, _, shape) in handle.model().weights().iter() {
        let (Some(layer), Some(target)) = (id.layer, LoraTarget::of(id.role)) else {
            continue;
        };
        let &[out_dim, in_dim] = shape.shape.as_slice() else {
            continue;
        };
        let s = seed + (layer * 16) as u64 + target as u64;
        weights.insert(
            lora_module(layer, target),
            LoraWeight {
                a: HostTensor::f32(vec![in_dim, LORA_RANK], lora_fill(in_dim * LORA_RANK, s)),
                b: HostTensor::f32(
                    vec![LORA_RANK, out_dim],
                    lora_fill(LORA_RANK * out_dim, s + 1000),
                ),
            },
        );
    }
    LoraAdapter {
        config: LoraAdapterConfig {
            r: LORA_RANK,
            lora_alpha: alpha,
            target_modules: Vec::new(),
            use_rslora: false,
        },
        weights,
    }
}

/// `handle`'s checkpoint with `adapter` merged into every adapted projection: `W + scale * (A B)^T`, as an
/// F32 entry (the stored `[out, in]` layout; BF16 values widen exactly).
fn merged_store(handle: &ModelHandle, adapter: &LoraAdapter) -> WeightStore {
    let base = handle.shared_store();
    let scale = adapter.config.scaling();
    let mut merged: HashMap<WeightKey, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for (id, view, _) in handle.model().weights().iter() {
        let (Some(layer), Some(target)) = (id.layer, LoraTarget::of(id.role)) else {
            continue;
        };
        let WeightView::Stored(key) = view else {
            panic!("{id} is not one stored tensor");
        };
        let WeightEntry::Dense(dense) = base.get(key.as_str()).expect("the store holds the key")
        else {
            panic!("the fixture is dense");
        };
        let (shape, bytes) = (dense.shape().to_vec(), dense.bytes().as_slice());
        let mut w: Vec<f32> = match dense.dtype() {
            poot_tensor::DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&word| f32::from_le_bytes(word))
                .collect(),
            poot_tensor::DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16))
                .collect(),
            other => panic!("the fixture is F32 or BF16, not {other:?}"),
        };
        let weight = &adapter.weights[&lora_module(layer, target)];
        let (a, b) = (weight.a.as_f32().unwrap(), weight.b.as_f32().unwrap());
        let (out_dim, in_dim) = (shape[0], shape[1]);
        for o in 0..out_dim {
            for i in 0..in_dim {
                let delta: f32 = (0..LORA_RANK)
                    .map(|k| a[i * LORA_RANK + k] * b[k * out_dim + o])
                    .sum();
                w[o * in_dim + i] += scale * delta;
            }
        }
        merged.insert(key.clone(), (shape, w));
    }
    let mut out = WeightStore::builder();
    for (key, entry) in base.iter() {
        let entry = match merged.remove(key) {
            Some((shape, values)) => {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                WeightEntry::Dense(
                    DenseWeight::try_new(poot_tensor::DType::F32, shape, bytes.into()).unwrap(),
                )
            }
            None => entry.clone(),
        };
        out.insert(key.clone(), entry).unwrap();
    }
    out.build()
}

fn family_handle_over(family: &str, eos: u32, store: WeightStore) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("no builtin family {family}"));
    let mut config = (entry.fixture)().config.clone();
    config["eos_token_id"] = serde_json::json!(eos);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(ModelHandle::from_checkpoint(&raw, store, &registry, |_| Ok(fixture_text())).unwrap())
}

/// The bytes the live weight allocations hold, and how many allocations were ever made.
fn weight_counters<S: State>(driver: &Driver<S>) -> (u64, u64) {
    let stats = driver.executor_stats();
    let (_, weights) = stats
        .memory
        .iter()
        .find(|(role, _)| *role == poot_runtime_common::BufferRole::Weight)
        .expect("the executor counts weights");
    (weights.live_bytes, weights.allocations)
}

/// What a sequence produced with `RowWork` steps: its tokens and the log-probabilities (chosen and the
/// top five) the host-path head gave each.
struct Traced {
    tokens: Vec<u32>,
    logprobs: Vec<crate::core::sampler::TokenLogprob>,
}

/// Drive `seq` for up to `steps` steps (prefill pieces included), committing every pick.
fn drive(
    driver: &mut Driver<Serving>,
    seq: SeqId,
    prompt_len: usize,
    prefilled: &mut usize,
    steps: usize,
    out: &mut Traced,
    max_new: usize,
) {
    let mut sampler = Sampler::greedy().with_logprobs(5);
    for _ in 0..steps {
        if out.tokens.len() == max_new {
            return;
        }
        let work = if *prefilled < prompt_len {
            RowWork::Prefill { upto: prompt_len }
        } else {
            RowWork::Decode { ahead: &[] }
        };
        let result = {
            let mut rows = [StepRow {
                seq,
                work,
                sampler: &mut sampler,
            }];
            driver.step(&mut rows).unwrap().remove(0)
        };
        *prefilled += result.consumed;
        let kept = result.tokens.len();
        out.tokens.extend(&result.tokens);
        out.logprobs.extend(result.logprobs);
        driver.commit(seq, kept, &mut sampler).unwrap();
    }
}

fn assert_traced_matches(label: &str, got: &Traced, want: &Traced) {
    assert_eq!(got.tokens, want.tokens, "{label}: tokens");
    assert_eq!(
        got.logprobs.len(),
        want.logprobs.len(),
        "{label}: logprob records"
    );
    for (step, (got, want)) in got.logprobs.iter().zip(&want.logprobs).enumerate() {
        assert_eq!(got.token, want.token, "{label} step {step}: token");
        assert!(
            (got.logprob - want.logprob).abs() <= 2e-3,
            "{label} step {step}: logprob {} vs reference {}",
            got.logprob,
            want.logprob
        );
        for ((gt, gl), (wt, wl)) in got.top.iter().zip(&want.top) {
            assert_eq!(gt, wt, "{label} step {step}: top-5 order");
            assert!(
                (gl - wl).abs() <= 2e-3,
                "{label} step {step}: top-5 logprob {gl} vs reference {wl}"
            );
        }
    }
}

/// POOT-579 SC-005: capture adapter set A, prepare it, then capture B - the same adapter name and
/// shapes, other source and weights - and prepare it beside A. Both generations run (A mid-generation
/// while B is admitted), and every step's chosen and top-five log-probabilities equal the run of a model
/// with that generation's adapter merged into its weights. Unloading A while its sequence is live is a
/// typed refusal and leaves its run correct; once A's sequence is released, unloading A returns exactly
/// A's own stacked tensors and moves no base upload (live weight bytes and the count of weight
/// allocations stay the base's plus the live generations'). Mutation: serve B from A's executable (the
/// adapter bytes resident under the logical name alone); B's run equals A's reference, not its own.
/// Mutation: unload A's executable before the in-use check; A's remaining steps fail.
fn adapter_generation_replacement_is_isolated_and_freed(choice: BackendChoice) {
    let handle = family_handle("qwen2", NO_EOS);
    let (adapter_a, adapter_b) = (
        fixture_adapter(&handle, 11, 4.0),
        fixture_adapter(&handle, 77, 6.0),
    );
    let prompt: Vec<u32> = PROMPT.to_vec();
    let max_new = 6;

    // Every executor opens before any runs (the ROCm runtime does not survive a second open after a drop).
    let Some(served) = serving_driver(choice, &handle) else {
        return;
    };
    let reference = |adapter: &LoraAdapter| {
        let store = merged_store(&handle, adapter);
        let merged = family_handle_over("qwen2", NO_EOS, store);
        solo_driver(choice, &merged).map(|driver| (driver, merged))
    };
    let Some((mut ref_a, _keep_a)) = reference(&adapter_a) else {
        return;
    };
    let Some((mut ref_b, _keep_b)) = reference(&adapter_b) else {
        return;
    };
    let record = |driver: &mut Driver<Serving>| {
        let generated = driver
            .generate(
                GenerateRequest {
                    prompt: prompt.clone(),
                    max_new,
                    sampler: Sampler::greedy().with_logprobs(5),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .unwrap();
        Traced {
            tokens: generated.tokens,
            logprobs: generated.logprobs,
        }
    };
    let (want_a, want_b) = (record(&mut ref_a), record(&mut ref_b));
    assert_ne!(
        want_a.tokens, want_b.tokens,
        "the two adapters must change the run, or the rows below prove nothing"
    );

    let shapes = |adapters: &[crate::driver::GenerationId]| {
        let pool = pool_shape(8, 4);
        let (rows, heads) = ([nz(1)], [Head::GREEDY, Head::Logits]);
        (pool, rows, heads, adapters.to_vec())
    };
    let prepare = |driver: Driver, adapters: &[crate::driver::GenerationId]| {
        let (pool, rows, heads, adapters) = shapes(adapters);
        driver
            .prepare(&ServingShapes {
                layout: Layout::Paged(pool),
                rows: &rows,
                heads: &heads,
                warm: Warm::Prompts(&[prompt.len()]),
                windows: &[],
                adapters: &adapters,
            })
            .unwrap()
    };
    // The same shapes admitted again under the layout already served, with more adapters.
    let extend = |driver: &mut Driver<Serving>, adapters: &[crate::driver::GenerationId]| {
        let (pool, rows, heads, adapters) = shapes(adapters);
        driver
            .extend(&ServingShapes {
                layout: Layout::Paged(pool),
                rows: &rows,
                heads: &heads,
                warm: Warm::Prompts(&[prompt.len()]),
                windows: &[],
                adapters: &adapters,
            })
            .unwrap();
    };

    // The base alone.
    let mut served = prepare(served, &[]);
    let (base_bytes, base_allocations) = weight_counters(&served);
    assert!(base_bytes > 0, "the base entries uploaded their weights");

    // A tensor's worth of stack: per adapted projection `(adapters + 1)` entries of `A [in, r]` and
    // `B [r, out]` in F32, plus the model-level scale.
    let stack_bytes = |adapters: usize| -> u64 {
        let mut bytes = 4 * (adapters as u64 + 1);
        for (id, _, shape) in handle.model().weights().iter() {
            if id.layer.is_some() && LoraTarget::of(id.role).is_some() {
                let &[out_dim, in_dim] = shape.shape.as_slice() else {
                    continue;
                };
                bytes +=
                    4 * (adapters as u64 + 1) * (in_dim * LORA_RANK + LORA_RANK * out_dim) as u64;
            }
        }
        bytes
    };
    let tensor_count = {
        let adapted = handle
            .model()
            .weights()
            .iter()
            .filter(|(id, _, _)| id.layer.is_some() && LoraTarget::of(id.role).is_some())
            .count() as u64;
        2 * adapted + 1
    };

    let source = |n: u64| crate::driver::AdapterSource(std::num::NonZeroU64::new(n).unwrap());
    let set = |n: u64| crate::driver::AdapterSet {
        adapters: vec![crate::driver::AdapterSpec {
            name: "style".to_string(),
            shape: vec![LORA_RANK],
        }],
        source: source(n),
    };
    let generation_a = served
        .capture_adapter_set(set(1), std::slice::from_ref(&adapter_a))
        .unwrap();
    extend(&mut served, &[generation_a]);
    let (bytes_a, allocations_a) = weight_counters(&served);
    assert_eq!(
        bytes_a - base_bytes,
        stack_bytes(1),
        "generation A uploaded its own stacked tensors and no base weight"
    );
    assert_eq!(allocations_a - base_allocations, tensor_count);

    // A's sequence starts, then B replaces the adapter while A is mid-generation.
    let on = |generation| AdapterRef::Adapter {
        generation,
        index: crate::driver::LoraIdx::new(1).unwrap(),
    };
    let open = |driver: &mut Driver<Serving>, adapter| {
        let Admission::Opened { seq, .. } = driver
            .open(SeqRequest::new(&prompt, max_new, adapter))
            .unwrap()
        else {
            panic!("no room");
        };
        seq
    };
    let seq_a = open(&mut served, on(generation_a));
    let (mut done_a, mut traced_a) = (
        0usize,
        Traced {
            tokens: Vec::new(),
            logprobs: Vec::new(),
        },
    );
    drive(
        &mut served,
        seq_a,
        prompt.len(),
        &mut done_a,
        3,
        &mut traced_a,
        max_new,
    );

    let generation_b = served
        .capture_adapter_set(set(2), std::slice::from_ref(&adapter_b))
        .unwrap();
    assert_ne!(
        generation_a, generation_b,
        "a new source is a new generation"
    );
    extend(&mut served, &[generation_a, generation_b]);
    let (bytes_b, allocations_b) = weight_counters(&served);
    assert_eq!(
        bytes_b - bytes_a,
        stack_bytes(1),
        "generation B uploaded only its own stack beside A's"
    );
    assert_eq!(allocations_b - allocations_a, tensor_count);

    // A cannot be unloaded while its sequence runs, and B (its own sequence, run to the end) matches
    // its own reference.
    let refused = served.unload_adapter_set(generation_a).unwrap_err();
    assert!(
        matches!(
            refused,
            DriverError::InvalidRequest(InvalidRequest::Adapter(
                crate::driver::lora::LoraError::InUse { .. }
            ))
        ),
        "{refused:?}"
    );
    let seq_b = open(&mut served, on(generation_b));
    let (mut done_b, mut traced_b) = (
        0usize,
        Traced {
            tokens: Vec::new(),
            logprobs: Vec::new(),
        },
    );
    drive(
        &mut served,
        seq_b,
        prompt.len(),
        &mut done_b,
        32,
        &mut traced_b,
        max_new,
    );
    assert_traced_matches("generation B", &traced_b, &want_b);

    // A finishes after B ran, and still matches its own reference.
    drive(
        &mut served,
        seq_a,
        prompt.len(),
        &mut done_a,
        32,
        &mut traced_a,
        max_new,
    );
    assert_traced_matches("generation A", &traced_a, &want_a);

    served.release(seq_a, Release::Finished).unwrap();
    served.release(seq_b, Release::Finished).unwrap();
    served.unload_adapter_set(generation_a).unwrap();
    let (bytes_after, allocations_after) = weight_counters(&served);
    assert_eq!(
        bytes_after,
        bytes_b - stack_bytes(1),
        "unloading A returned exactly A's unique tensors"
    );
    assert_eq!(
        allocations_after, allocations_b,
        "allocations are monotonic: nothing was re-uploaded"
    );
    assert_eq!(
        bytes_after - base_bytes,
        stack_bytes(1),
        "base weights and B remain"
    );
    served.unload_adapter_set(generation_b).unwrap();
    assert_eq!(
        weight_counters(&served).0,
        base_bytes,
        "only the base weights remain"
    );
}
per_backend!(
    BackendChoice,
    adapter_generation_replacement_is_isolated_and_freed,
    adapter_generation_replacement_is_isolated_and_freed_wgpu,
    adapter_generation_replacement_is_isolated_and_freed_rocm,
    adapter_generation_replacement_is_isolated_and_freed_ptx
);

/// The Qwen2.5-VL fixture's 48-word vocabulary: the vision markers and the tokens around them.
mod qwen_vl_ids {
    pub(super) const VOCAB: usize = 48;
    pub(super) const VISION_START: (&str, u32) = ("<|vision_start|>", 40);
    pub(super) const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 41);
    pub(super) const VIDEO_PAD: (&str, u32) = ("<|video_pad|>", 42);
    pub(super) const VISION_END: (&str, u32) = ("<|vision_end|>", 43);
    pub(super) const IM_END: (&str, u32) = ("<|im_end|>", 47);
}

/// A word-level tokenizer: word `t<i>` is token `i` but for `user`, `hello` and the vision markers.
fn qwen_vl_tokenizer() -> Tokenizer {
    use qwen_vl_ids as ids;
    let mut vocab: HashMap<String, u32> = (0..ids::VOCAB as u32)
        .map(|id| (format!("t{id}"), id))
        .collect();
    let markers = [
        ids::VISION_START,
        ids::IMAGE_PAD,
        ids::VIDEO_PAD,
        ids::VISION_END,
        ids::IM_END,
    ];
    for (word, id) in [("user", 2), ("hello", 4)].into_iter().chain(markers) {
        vocab.retain(|_, v| *v != id);
        vocab.insert(word.to_string(), id);
    }
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("t0".to_string())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    let specials: Vec<tokenizers::AddedToken> = markers
        .iter()
        .map(|&(word, _)| tokenizers::AddedToken::from(word, true))
        .collect();
    tokenizer.add_special_tokens(&specials);
    tokenizer
}

/// The registry's Qwen2.5-VL fixture (the text model and the vision tower over one set of weights, all
/// multiplied by `factor`: the fixtures' random weights are small enough that a tiny tower's logits
/// barely depend on its inputs) as an HF directory with the checkpoint's `config.json`, the processor
/// config `preprocessor`, the 48-word tokenizer and a chat template; loaded the way a server loads it.
fn qwen_vl_checkpoint(
    factor: f32,
    preprocessor: Option<serde_json::Value>,
) -> (poot_test_util::UniqueTempPath, Arc<ModelHandle>) {
    let fixture = poot_models::test_support::vlm::qwen_vl_fixture();
    let keep = |name: &str| name.contains("norm") || name.contains("ln_q");
    let scaled_store = {
        let mut out = WeightStore::builder();
        for (key, entry) in fixture.store.iter() {
            let WeightEntry::Dense(dense) = entry else {
                panic!("the fixture is dense")
            };
            let bytes: Vec<u8> = if keep(key.as_str()) {
                dense.bytes().as_slice().to_vec()
            } else {
                dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&pair| {
                        let value = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                        (((value * factor).to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect()
            };
            let dense =
                DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into()).unwrap();
            out.insert(key.clone(), WeightEntry::Dense(dense)).unwrap();
        }
        out.build()
    };
    let tensors: Vec<ArchiveTensor> = scaled_store
        .iter()
        .map(|(key, entry)| {
            let WeightEntry::Dense(dense) = entry else {
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
    let dir = write_hf_checkpoint("qwen_vl", &fixture.config, &tensors);
    qwen_vl_tokenizer()
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": poot_test_util::qwen_vl::CHAT_TEMPLATE }).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        preprocessor
            .unwrap_or_else(|| fixture.preprocessor.clone().unwrap())
            .to_string(),
    )
    .unwrap();
    let handle = ModelHandle::load(
        &dir,
        &Registry::builtin().unwrap(),
        &crate::driver::LoadOptions::default(),
    )
    .unwrap();
    (dir, Arc::new(handle))
}

/// A `height x width` RGB image whose pixels differ with `seed`.
fn qwen_vl_image(
    height: usize,
    width: usize,
    seed: u8,
) -> poot_models::multimodal::front_end::Image {
    let rgb = (0..height * width * 3)
        .map(|i| {
            (i as u8)
                .wrapping_mul(37)
                .wrapping_add(seed.wrapping_mul(91))
        })
        .collect();
    poot_models::multimodal::front_end::Image::from_rgb8(height, width, rgb).unwrap()
}

/// The vision tower `handle` runs on its driver, each row also evaluated on the CPU oracle over the same
/// inputs and compared (tier 2).
struct OracleCheckedTowers {
    drivers: crate::TowerDrivers,
    tower: Arc<ModelHandle>,
    checked: usize,
}

impl OracleCheckedTowers {
    fn open(handle: &ModelHandle, executor: Box<dyn poot_executor::Executor>) -> Self {
        let mut executor = Some(executor);
        let drivers = crate::TowerDrivers::open(handle, |_, positions| {
            Ok::<_, DriverError>((executor.take().expect("one tower"), options(64, 64))).inspect(
                |_| {
                    assert!(
                        positions.get() >= 64,
                        "the tower takes the fixture's images"
                    )
                },
            )
        })
        .unwrap();
        Self {
            drivers,
            tower: Arc::clone(&handle.towers()[0].1),
            checked: 0,
        }
    }
}

impl poot_models::multimodal::front_end::Towers for OracleCheckedTowers {
    fn encode(
        &mut self,
        tower: poot_models::multimodal::front_end::TowerId,
        inputs: &RequestInputs,
    ) -> Result<Vec<f32>, poot_models::multimodal::front_end::TowerError> {
        let device = self.drivers.encode(tower, inputs)?;
        let model = self.tower.model();
        let schema = model.schema();
        let one_row = |value: &InputValue| {
            let InputValue::Row(tensor) = value else {
                panic!("a tower row's inputs are per-row tensors")
            };
            let mut shape = vec![1];
            shape.extend_from_slice(tensor.shape());
            match tensor.dtype() {
                poot_tensor::DType::I32 => {
                    HostTensor::i32(shape, tensor.as_i32().unwrap().to_vec())
                }
                _ => HostTensor::f32(shape, tensor.as_f32().unwrap().to_vec()),
            }
        };
        let slots: Vec<(SlotKey, HostTensor)> = inputs
            .iter()
            .map(|(id, value)| (schema.input(id).unwrap().slot().clone(), one_row(value)))
            .collect();
        let patches = slots
            .iter()
            .map(|(_, tensor)| tensor.shape()[1])
            .next()
            .unwrap();
        let graph = model
            .trace(
                &StepSpec::encode(
                    StepShape::new(nz(1), nz(patches), nz(patches)).with_logits(LogitRows::All),
                ),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let (want, _) = eval_mapped(&graph, self.tower.store(), model.weights(), &slots, &[]);
        close("the tower's rows", &device, want.as_f32().unwrap());
        self.checked += 1;
        Ok(device)
    }
}

/// The log-probabilities (natural log softmax) of every vocabulary token after `prepared`'s prompt and
/// the `generated` tokens that follow it, on the CPU oracle: the text tower's `step` graph (carrying the
/// embedding rows) over the whole sequence, with the rows scattered over the placeholders, the prompt at
/// its sectioned positions and each generated token one further than the last on every axis.
fn qwen_vl_oracle_logprobs(
    handle: &ModelHandle,
    prepared: &poot_models::multimodal::front_end::Prepared,
    generated: &[u32],
) -> Vec<f32> {
    use poot_models::io::{Fill, Support};
    let model = handle.model();
    let prompt: Vec<u32> = prepared.prompt.iter().chain(generated).copied().collect();
    let n = prompt.len();
    let Some(InputValue::Tokens(rows)) = prepared.inputs.get(EMBEDS) else {
        panic!("the prompt carries embedding rows")
    };
    let Some(InputValue::Tokens(sections)) =
        prepared.inputs.get(poot_models::components::rope::MROPE)
    else {
        panic!("the prompt carries its sectioned positions")
    };
    let width = rows.width();
    let Support::Sparse(placeholders) = rows.support() else {
        panic!("the rows sit at the placeholders")
    };
    let mut embeds = vec![0.0f32; n * width];
    let mut at = vec![0i32; n];
    for (row, &position) in placeholders.iter().enumerate() {
        embeds[position * width..(position + 1) * width]
            .copy_from_slice(&rows.data().as_f32().unwrap()[row * width..(row + 1) * width]);
        at[position] = 1;
    }
    let table = sections.data().as_i32().unwrap();
    // Past the prompt every axis continues from the collapsed position after its last token.
    let Fill::Advance { from, step } = sections.beyond() else {
        panic!("the positions advance past the prompt")
    };
    let positions: Vec<i32> = (0..3)
        .flat_map(|axis| {
            (0..prepared.prompt.len())
                .map(move |i| table[i * 3 + axis])
                .chain((0..generated.len() as i32).map(move |k| from[axis] + k * step))
        })
        .collect();
    let graph = model
        .trace(
            &StepSpec::step(StepShape::new(
                nz(1),
                nz(n),
                nz(model.config().max_positions),
            ))
            .carrying(EMBEDS),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let state: Vec<HostTensor> = graph
        .state()
        .map(|(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
        .collect();
    let activation = |tag| SlotKey::new(Slot::Activation, Some(tag));
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![1, n], prompt.iter().map(|&t| t as i32).collect()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![1, n], (0..n as i32).collect()),
        ),
        (
            SlotKey::new(Slot::MropePosition, None),
            HostTensor::i32(vec![3, n], positions),
        ),
        (
            activation("embeds"),
            HostTensor::f32(vec![1, n, width], embeds),
        ),
        (activation("embeds_at"), HostTensor::i32(vec![1, n], at)),
    ];
    let (logits, _) = eval_mapped(&graph, handle.store(), model.weights(), &slots, &state);
    let logits = logits.as_f32().unwrap();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    logits.iter().map(|&l| l - max - sum.ln()).collect()
}

/// What the front end of the Qwen2.5-VL `handle` lowers `image` and the question `hello t1` to.
fn qwen_vl_lower(
    handle: &ModelHandle,
    towers: &mut dyn poot_models::multimodal::front_end::Towers,
    image: &poot_models::multimodal::front_end::Image,
) -> Result<
    poot_models::multimodal::front_end::Prepared,
    poot_models::multimodal::front_end::FrontEndError,
> {
    use poot_models::multimodal::front_end::{ChatTurn, Media};
    handle.front_end().expect("a Qwen2.5-VL front end").prepare(
        &[ChatTurn::user(1, "hello t1")],
        &mut [Media::Image(image)],
        towers,
        handle.text(),
    )
}

/// POOT-650 SC-001: a Qwen2.5-VL image-plus-prompt request generates through `Driver` with the front
/// end producing the step inputs (the vision tower's rows through `Driver::encode`, the prompt's
/// sectioned positions bound to `Slot::MropePosition`), and the log-probabilities of every vocabulary
/// token at the first step, and at the two decode steps after it, match the CPU oracle's over the same
/// inputs (tier 2: every element within `1e-4 * (1 + |want|)`), for two different images through one
/// front end; the tower's rows are the oracle's too. The oracle's two images' distributions differ by
/// far more than the tolerance, so a row that ignored the image could not pass; the decode steps run at
/// the positions that continue after the image. Mutation: feed zeroed image embeddings (zero the rows
/// in the front end's `prompt_inputs`); the log-probabilities leave the oracle's.
fn qwen25_vl_image_prompt_first_step_matches_the_cpu_oracle(choice: BackendChoice) {
    const NEW: usize = 3;
    let (_dir, handle) = qwen_vl_checkpoint(2.0, None);
    let (first, second) = (qwen_vl_image(12, 8, 1), qwen_vl_image(12, 8, 2));
    let (Some(vision), Some(text)) = (open(choice), open(choice)) else {
        return;
    };
    let mut towers = OracleCheckedTowers::open(&handle, vision);
    let mut text = Driver::new(Arc::clone(&handle), text, options(16, 32))
        .unwrap()
        .serve(Layout::Contiguous);
    let prepared =
        [&first, &second].map(|image| qwen_vl_lower(&handle, &mut towers, image).unwrap());
    assert_eq!(towers.checked, 2, "each image's tower row was checked");
    let apart = poot_test_util::max_abs_error(
        &qwen_vl_oracle_logprobs(&handle, &prepared[0], &[]),
        &qwen_vl_oracle_logprobs(&handle, &prepared[1], &[]),
    );
    assert!(
        apart > 1e-2,
        "the two images' oracle distributions differ by {apart}"
    );
    for (n, prepared) in prepared.iter().enumerate() {
        let generation = text
            .generate_with_inputs(
                GenerateRequest {
                    prompt: prepared.prompt.clone(),
                    max_new: NEW,
                    sampler: Sampler::greedy().with_logprobs(qwen_vl_ids::VOCAB),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                prepared.inputs.clone(),
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .unwrap();
        assert_eq!(generation.tokens.len(), NEW, "image {n}");
        for step in 0..NEW {
            // Step `k` is the distribution after the prompt and the `k` tokens generated before it.
            let want = qwen_vl_oracle_logprobs(&handle, prepared, &generation.tokens[..step]);
            let record = &generation.logprobs[step];
            assert_eq!(
                record.top.len(),
                qwen_vl_ids::VOCAB,
                "image {n} step {step}"
            );
            let worst = record
                .top
                .iter()
                .map(|&(token, got)| {
                    let want = want[token as usize];
                    ((got - want).abs() / (1.0 + want.abs()), token, got, want)
                })
                .fold((0.0f32, 0, 0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
            assert!(
                worst.0 <= 1e-4,
                "image {n} step {step}, token {}: device log-probability {}, oracle {}",
                worst.1,
                worst.2,
                worst.3
            );
        }
    }
}
per_backend!(
    BackendChoice,
    qwen25_vl_image_prompt_first_step_matches_the_cpu_oracle,
    qwen25_vl_image_prompt_first_step_matches_the_cpu_oracle_wgpu,
    qwen25_vl_image_prompt_first_step_matches_the_cpu_oracle_rocm,
    qwen25_vl_image_prompt_first_step_matches_the_cpu_oracle_ptx
);

/// POOT-650: an image of more patches than the vision tower takes in one step (4100 against 4096) is
/// refused by the front end as the client's (the tower's own bound, `MAX_PATCHES`), before any encode
/// step: no encode step and no generation step runs and nothing panics. Mutation: raise `MAX_PATCHES`
/// in the tower to 8192; the front end admits the image, the tower's trace refuses or runs it, and the
/// row goes red.
fn qwen25_vl_an_image_past_the_patch_cap_is_a_typed_refusal(choice: BackendChoice) {
    use poot_models::multimodal::front_end::FrontEndError;

    let (_dir, handle) = qwen_vl_checkpoint(
        2.0,
        Some(serde_json::json!({
            "do_resize": false, "min_pixels": 1, "max_pixels": 1048576, "patch_size": 2,
            "temporal_patch_size": 2, "merge_size": 2,
        })),
    );
    // 164 x 100 pixels of 2-pixel patches: an 82 x 50 grid, 4100 patches.
    let image = qwen_vl_image(164, 100, 1);
    let (Some(vision), Some(text)) = (open(choice), open(choice)) else {
        return;
    };
    let mut vision = Some(vision);
    let mut towers = crate::TowerDrivers::open(&handle, |_, positions| {
        assert_eq!(positions.get(), 4096, "the tower's own bound");
        Ok::<_, DriverError>((
            vision.take().expect("one tower"),
            // The driver's own trace bound is out of the way: the refusal is the tower's.
            DriverOptions {
                trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(8192)),
                ..options(4096, 4096)
            },
        ))
    })
    .unwrap();
    let text = Driver::new(Arc::clone(&handle), text, options(16, 32)).unwrap();
    let error = qwen_vl_lower(&handle, &mut towers, &image).unwrap_err();
    assert!(
        matches!(&error, FrontEndError::Refused(reason) if reason.contains("4100")),
        "{error:?}"
    );
    let tower_id = handle.towers()[0].0;
    assert_eq!(
        towers.driver(tower_id).unwrap().stats().steps,
        0,
        "no encode step ran"
    );
    assert_eq!(text.stats().steps, 0, "no generation step ran");
}
per_backend!(
    BackendChoice,
    qwen25_vl_an_image_past_the_patch_cap_is_a_typed_refusal,
    qwen25_vl_an_image_past_the_patch_cap_is_a_typed_refusal_wgpu,
    qwen25_vl_an_image_past_the_patch_cap_is_a_typed_refusal_rocm,
    qwen25_vl_an_image_past_the_patch_cap_is_a_typed_refusal_ptx
);

/// `generate_text` on a fresh `choice` executor: the generation and every piece the sink saw, joined.
fn text_run(
    handle: &Arc<ModelHandle>,
    choice: BackendChoice,
    request: &crate::driver::TextRequest,
) -> Option<(crate::driver::Generation, String)> {
    let executor = open(choice)?;
    let mut text = String::new();
    let generation =
        crate::driver::generate_text(handle, executor, request, &mut |piece| text.push_str(piece))
            .unwrap();
    Some((generation, text))
}

fn text_request(ctx: Option<usize>) -> crate::driver::TextRequest {
    crate::driver::TextRequest {
        // Seven one-character words of the fixture's vocabulary.
        prompt: "b c d e f g h".into(),
        system: None,
        raw: false,
        stops: Vec::new(),
        max_new: None,
        ctx,
        sampling: crate::driver::Sampling::GREEDY,
        ignore_eos: false,
    }
}

/// A command-line run with no `-n` ends at the window: a 7-token prompt in a 16-position window
/// generates 9 tokens and stops on the length, the text begins with the prompt it continues, and the
/// engine's own timing covers the 7 prompt tokens and the 8 tokens after the first (the first comes out
/// of the prefill step). Mutation: count the first token as a decode token (decode reads 9).
#[test]
fn a_text_run_ends_at_the_window_and_reports_engine_timing_wgpu() {
    let handle = handle(Weights::ZeroHead, NO_EOS);
    let Some((generation, text)) = text_run(&handle, BackendChoice::Wgpu, &text_request(Some(16)))
    else {
        return;
    };
    assert_eq!(generation.tokens.len(), 9);
    assert_eq!(generation.finish, crate::driver::FinishReason::MaxTokens);
    assert!(text.starts_with("b c d e f g h"), "{text}");
    let timing = generation
        .timing
        .expect("generate_text measures its phases");
    assert_eq!(timing.prefill.tokens, 7);
    assert_eq!(timing.decode.tokens, 8);
    assert!(!timing.prefill.wall.is_zero() && !timing.decode.wall.is_zero());
    assert!(timing.prefill.tokens_per_second().is_some());
    assert!(timing.decode.tokens_per_second().is_some());
}

/// `--ignore-eos` and `-r`: with every logit zero the greedy token is id 0, which this model declares
/// its EOS. A run ends on it at once; with `ignore_eos` it runs to the window; and a stop string ends a
/// run that has no EOS after the token that completes it. Mutations: pass `ignore_eos: false` always;
/// drop the framed stops from the generate request.
#[test]
fn ignore_eos_and_reverse_prompts_decide_where_a_text_run_ends_wgpu() {
    let ends_at_zero = handle(Weights::ZeroHead, 0);
    let Some((ended, _)) = text_run(&ends_at_zero, BackendChoice::Wgpu, &text_request(Some(16)))
    else {
        return;
    };
    assert_eq!(ended.finish, crate::driver::FinishReason::Eos);
    assert!(ended.tokens.is_empty());

    let ignoring = crate::driver::TextRequest {
        ignore_eos: true,
        ..text_request(Some(16))
    };
    let (through, _) = text_run(&ends_at_zero, BackendChoice::Wgpu, &ignoring).unwrap();
    assert_eq!(through.finish, crate::driver::FinishReason::MaxTokens);
    assert_eq!(through.tokens.len(), 9);

    let never_ends = handle(Weights::ZeroHead, NO_EOS);
    let stopping = crate::driver::TextRequest {
        stops: vec!["a".into()],
        ..text_request(Some(16))
    };
    let (stopped, _) = text_run(&never_ends, BackendChoice::Wgpu, &stopping).unwrap();
    assert_eq!(stopped.finish, crate::driver::FinishReason::Stop);
    assert_eq!(stopped.tokens.len(), 1);
}

/// The two phases of a generation partition the engine's wall time: the prefill wall plus the decode
/// wall is, to the nanosecond, the wall of every step the executor recorded during the run. A phase
/// that reads another phase's window, or a window that drops steps, breaks the sum. Mutation: take the
/// prefill wall from the decode window (the sum becomes twice the decode wall).
#[test]
fn a_generations_phase_timings_partition_the_engine_wall_wgpu() {
    let handle = handle(Weights::ZeroHead, NO_EOS);
    let Some(driver) = bare_driver(BackendChoice::Wgpu, &handle, 4, CAPACITY) else {
        return;
    };
    let mut driver = driver.with_phase_timing().serve(Layout::Contiguous);
    let mark = driver.executor_stats().timing.mark();
    let generation = driver
        .generate(
            GenerateRequest {
                prompt: PROMPT.to_vec(),
                max_new: 9,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap();
    let timing = driver.executor_stats().timing;
    let entries = timing.entries_with_activity_since(&mark);
    let engine_wall = poot_profile::Report::window(&timing, &mark, &entries)
        .host
        .wall;
    assert!(!engine_wall.is_zero());
    let timing = generation.timing.expect("the driver measures its phases");
    assert_eq!(timing.prefill.wall + timing.decode.wall, engine_wall);
}

/// A chat reply stops at the template's turn end and the marker is not in the text: the template
/// ends its turn with the token string `a`, which this zero-head model generates forever, so the run
/// generates the marker, finishes on the stop, and the sink sees none of it. A `-r` string of the
/// caller's own, by contrast, stays in the text. Mutation: forward the piece that completes the stop to
/// the sink (the reply ends with `a`).
#[test]
fn a_chat_turn_end_is_cut_from_the_text_and_a_reverse_prompt_is_kept_wgpu() {
    let template = ChatTemplate {
        template: Some(
            "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}{{ ' b' }}{% endif %}"
                .into(),
        ),
        bos_token: None,
        eos_token: Some("a".into()),
    };
    let handle = handle_with(Weights::ZeroHead, NO_EOS, template);
    let Some((chat, text)) = text_run(&handle, BackendChoice::Wgpu, &text_request(Some(16))) else {
        return;
    };
    assert_eq!(chat.finish, crate::driver::FinishReason::Stop);
    assert_eq!(chat.tokens.len(), 1, "the turn end is the first token");
    assert!(
        !text.contains('a'),
        "the turn end reached the sink: {text:?}"
    );

    // The same model asked for the raw text: the caller's `a` stop is in the output.
    let raw = crate::driver::TextRequest {
        raw: true,
        stops: vec!["a".into()],
        ..text_request(Some(16))
    };
    let (kept, text) = text_run(&handle, BackendChoice::Wgpu, &raw).unwrap();
    assert_eq!(kept.finish, crate::driver::FinishReason::Stop);
    assert!(text.ends_with('a'), "{text}");
}

/// A chat template over the fixture's vocabulary whose roles are single letters (`u`, `a`) and whose
/// generation prompt is the assistant's own role marker, so a conversation's earlier turns render to the
/// exact tokens the sequence already holds.
const ROLE_TEMPLATE: &str = "{% for m in messages %}{{ m['role'][0] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}a {% endif %}";

/// The same without role markers: the generation prompt `a` of a finished turn is not in the rendered
/// history, so a later turn shares only the run before it with the sequence.
const BARE_TEMPLATE: &str = "{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}a {% endif %}";

fn chat_handle(template: &str) -> Arc<ModelHandle> {
    handle_with(
        Weights::Scaled,
        NO_EOS,
        ChatTemplate {
            template: Some(template.into()),
            bos_token: None,
            eos_token: None,
        },
    )
}

fn chat_options() -> crate::driver::ChatOptions {
    crate::driver::ChatOptions {
        system: None,
        stops: Vec::new(),
        max_new: Some(5),
        ctx: None,
        sampling: crate::driver::Sampling::GREEDY,
        ignore_eos: false,
    }
}

/// What a two-turn conversation and a single run of its concatenated rendered prompt say.
struct TwoTurns {
    /// The second reply's generation and the engine-measured tokens its prefill fed.
    second: crate::driver::Generation,
    /// The one-shot run over `[user, reply, user]` rendered as the template renders them.
    reference: crate::driver::Generation,
    /// Tokens of the second turn's context.
    context: usize,
    /// Tokens of the first turn's context and reply.
    held: usize,
}

fn two_turns(template: &str, choice: BackendChoice) -> Option<TwoTurns> {
    use poot_text::Speaker;
    let handle = chat_handle(template);
    let mut session =
        crate::driver::ChatSession::new(&handle, open(choice)?, chat_options()).unwrap();
    let mut first_reply = String::new();
    let first = session
        .say("b c d", &mut |piece| first_reply.push_str(piece))
        .unwrap();
    let mut second_reply = String::new();
    let second = session
        .say("e f g", &mut |piece| second_reply.push_str(piece))
        .unwrap();
    session.finish().unwrap();

    let text = handle.text();
    let first_context = text
        .render_turns(None, &[(Speaker::User, "b c d")])
        .unwrap()
        .unwrap()
        .prompt;
    let rendered = text
        .render_turns(
            None,
            &[
                (Speaker::User, "b c d"),
                (Speaker::Assistant, &first_reply),
                (Speaker::User, "e f g"),
            ],
        )
        .unwrap()
        .unwrap()
        .prompt;
    let reference = crate::driver::TextRequest {
        prompt: rendered.clone(),
        system: None,
        raw: true,
        stops: Vec::new(),
        max_new: Some(5),
        ctx: None,
        sampling: crate::driver::Sampling::GREEDY,
        ignore_eos: false,
    };
    let (reference, _) = text_run(&handle, choice, &reference)?;
    assert_eq!(first.tokens.len(), 5);
    assert_eq!(second.tokens.len(), 5);
    Some(TwoTurns {
        second,
        reference,
        context: text.encode(&rendered).unwrap().len(),
        held: text.encode(&first_context).unwrap().len() + first.tokens.len(),
    })
}

/// POOT-1094 SC-001: the second reply of a two-turn conversation equals one run of the concatenated
/// rendered prompt, and the second turn prefills only what the sequence did not hold: the context less the
/// positions of the first turn that have K/V (all but the last token it generated). Mutations: reset the
/// sequence between turns (`Driver::resume` keeps nothing; the prefill count is the whole context);
/// forget the earlier turns (`ChatSession::say` renders only the new user turn; the reply is another).
#[test]
fn a_second_turn_equals_one_run_of_the_concatenated_prompt_and_prefills_only_the_new_tokens_wgpu() {
    let Some(run) = two_turns(ROLE_TEMPLATE, BackendChoice::Wgpu) else {
        return;
    };
    assert_eq!(run.second.tokens, run.reference.tokens);
    let timing = run.second.timing.expect("a session measures its phases");
    assert_eq!(timing.prefill.tokens, run.context - (run.held - 1));
}

/// POOT-1094 SC-002: when the template does not render the history back to the tokens the sequence
/// holds, the turn keeps the run they share and prefills from the first difference, and the reply is still
/// the single run's. Mutation: keep the whole resident run without comparing it to the context
/// (`shared = resident`); the K/V past the difference is another conversation's and the reply changes.
#[test]
fn a_second_turn_over_a_rewritten_history_still_equals_one_run_wgpu() {
    let Some(run) = two_turns(BARE_TEMPLATE, BackendChoice::Wgpu) else {
        return;
    };
    assert_eq!(run.second.tokens, run.reference.tokens);
    // The first context is `b c d a`; the second renders `b c d <reply> e f g a`: three tokens are shared.
    let timing = run.second.timing.expect("a session measures its phases");
    assert_eq!(timing.prefill.tokens, run.context - 3);
}
