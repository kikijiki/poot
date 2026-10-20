//! POOT-583 SC-001 (Gemma4 half): the Gemma4 MoE fixture's routed experts paged through a pool smaller
//! than its table (`P = 4 < E = 8`, two experts per token) generate the tokens the unpooled driver
//! generates, which are the family's recorded literals. Pooling is structural (ADR-0072, ruling Z29): no
//! Gemma4 code knows about the pool, so this file only drives the production entry point
//! (`Driver::with_expert_pool`) and reads what the residency manager did.
//!
//! Every row skips when its device does not open (`POOT_REQUIRE_WGPU=1` / `POOT_REQUIRE_ROCM=1` /
//! `POOT_REQUIRE_PTX=1` turn that into a failure). Device rows run serially under `/tmp/poot-gpu.lock`.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, Serving, open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::{Fixture, Registry};
use poot_runtime_common::DeviceBackend;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const VOCAB: usize = 48;
/// An end-of-sequence id outside the vocabulary: no recorded token ends the run early.
const NO_EOS: u32 = 999;
const CAPACITY: usize = 32;
const NEW_TOKENS: usize = 24;
/// The fixture has eight experts and routes two per token: four pool rows are half the table and serve
/// two tokens per step.
const POOL_ROWS: usize = 4;

const PROMPT: [u32; 8] = [1, 3, 17, 40, 8, 25, 11, 2];
/// The greedy tokens of the Gemma4 MoE fixture (`gemma4/moe` of the driver's recorded rows): the CPU
/// oracle's decode over the fixture, which the unpooled driver reproduces on every backend.
const RECORDED: [u32; NEW_TOKENS] = [
    34, 12, 5, 47, 33, 43, 13, 45, 36, 7, 22, 32, 2, 1, 10, 43, 8, 21, 41, 24, 11, 6, 20, 33,
];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..VOCAB - words.len()).map(|i| format!("p{i}")));
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
    tokenizer
}

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

/// The Gemma4 MoE fixture (`gemma4`'s `moe` variant: eight experts, two per token) as a loaded handle.
fn moe_handle() -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family == poot_models::gemma4::FAMILY)
        .expect("gemma4 is a shipped family");
    let fixture = poot_models::test_support::family_fixtures::variant(entry.family, "moe");
    let dir = poot_test_util::unique_temp_path("pooled_gemma4");
    write_checkpoint(&dir, &fixture);
    Arc::new(ModelHandle::load(&dir, &registry, &LoadOptions::default()).unwrap())
}

fn options() -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

fn open(choice: BackendChoice) -> Option<Box<dyn poot_executor::Executor>> {
    let backend = match choice {
        BackendChoice::Wgpu => DeviceBackend::Wgpu,
        BackendChoice::Rocm => DeviceBackend::Rocm,
        BackendChoice::Ptx => DeviceBackend::Ptx,
        BackendChoice::Vulkan => DeviceBackend::Vulkan,
        BackendChoice::Auto => unreachable!("rows name their backend"),
    };
    poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))
}

fn generate(driver: &mut Driver<Serving>, prompt: &[u32], max_new: usize) -> Vec<u32> {
    driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap()
        .tokens
}

/// SC-001 (Gemma4): the MoE fixture pooled at `P = 4 < E = 8` decodes its recorded 24 tokens, and so does
/// the unpooled driver (so the literals are the unpooled answer on this backend); the pool evicted, every
/// stage recorded once, and the executor served and read back at its sync points.
/// Mutation: in `ExpertResidency::sync`, skip the refill of a miss (answer no rows); the pool serves
/// another expert's bytes and the pooled tokens diverge from the recorded ones.
fn pooled_gemma4_decodes_the_unpooled_tokens_with_evictions(choice: BackendChoice) {
    let handle = moe_handle();
    // Both executors open before either driver runs (the ROCm runtime does not reopen one).
    let Some(plain_executor) = open(choice) else {
        return;
    };
    let Some(pooled_executor) = open(choice) else {
        return;
    };
    let mut plain = Driver::new(Arc::clone(&handle), plain_executor, options())
        .unwrap()
        .serve(Layout::Contiguous);
    let mut pooled = Driver::new(handle, pooled_executor, options())
        .unwrap()
        .with_expert_pool(nz(POOL_ROWS))
        .unwrap()
        .serve(Layout::Contiguous);
    assert_eq!(
        generate(&mut plain, &PROMPT, NEW_TOKENS),
        RECORDED,
        "unpooled gemma4/moe on {choice:?}"
    );
    assert_eq!(
        generate(&mut pooled, &PROMPT, NEW_TOKENS),
        RECORDED,
        "pooled gemma4/moe on {choice:?}"
    );
    let counters = pooled.stats().residency;
    eprintln!("residency counters: {counters:?}");
    assert!(
        counters.refills > 0 && counters.evictions > 0,
        "{counters:?}"
    );
    let stats = pooled.executor_stats();
    assert!(stats.syncs > 0 && stats.sync_read_bytes > 0, "{stats:?}");
    assert!(
        stats.stages.iter().all(|stage| stage.recordings == 1),
        "every stage records once: {:?}",
        stats.stages
    );
    assert_eq!(
        plain.stats().residency,
        poot_llm::driver::ResidencyCounters::default(),
        "the unpooled driver runs no residency manager"
    );
}

#[test]
fn pooled_gemma4_decodes_the_unpooled_tokens_with_evictions_wgpu() {
    pooled_gemma4_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Wgpu);
}

#[test]
fn pooled_gemma4_decodes_the_unpooled_tokens_with_evictions_rocm() {
    pooled_gemma4_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Rocm);
}

#[test]
fn pooled_gemma4_decodes_the_unpooled_tokens_with_evictions_ptx() {
    pooled_gemma4_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Ptx);
}
