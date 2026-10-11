//! POOT-583 SC-001 (Qwen3-Next half): the Qwen3-Next fixture's routed experts paged through a pool smaller
//! than its table (`P = 3 < E = 4`, two experts per token) generate the tokens the unpooled driver
//! generates, and the Gated DeltaNet layers' per-row recurrent state (ruling V1: one state row per
//! sequence, carried between steps) survives the pooled stage walk: two sequences stepped alternately
//! through the same pooled sites each equal their solo, unpooled run.
//!
//! Pooling is structural (ADR-0072, ruling Z29): no Qwen3-Next code knows about the pool, so this file only
//! drives the production entry points (`Driver::with_expert_pool`, `Driver::step`) and reads what the
//! residency manager did. Every row skips when its device does not open (`POOT_REQUIRE_WGPU=1` /
//! `POOT_REQUIRE_ROCM=1` / `POOT_REQUIRE_PTX=1` turn that into a failure). Device rows run serially under
//! `/tmp/poot-gpu.lock`.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::driver::step::PickPolicy;
use poot_llm::driver::{
    AdapterRef, Admission, BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest,
    Head, Layout, LoadOptions, ModelHandle, PoolShape, PreparedSetLimits, Release,
    ResidencyCounters, RowWork, SeqId, SeqRequest, ServingShapes, StepRow, Warm, open_executor,
    program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::{Fixture, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::DType;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const VOCAB: usize = 48;
/// An end-of-sequence id outside the vocabulary: no recorded token ends the run early.
const NO_EOS: u32 = 999;
const CAPACITY: usize = 64;
/// The fixture has four experts and routes two per token: a pool of three rows is smaller than the table
/// and serves one token per step.
const POOL_ROWS: usize = 3;

const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
/// The CPU oracle's greedy decode over the fixture (`qwen3next::tests::the_fixture_generates_the_recorded_tokens`),
/// which the unpooled driver reproduces on every backend.
const RECORDED: [u32; 24] = [
    20, 2, 20, 3, 6, 27, 25, 17, 18, 24, 20, 18, 45, 34, 45, 5, 45, 12, 35, 40, 5, 46, 35, 29,
];

const MEDIUM: [u32; 5] = [7, 22, 31, 12, 44];
const SHORT: [u32; 3] = [19, 6, 28];
const ROW_NEW: usize = 12;
/// What each row's tokens are on the fixture with its weights scaled by [`SCALE`], recorded from the solo
/// unpooled driver before the pool existed: at the fixture's own scale a wrong state row still picks the same
/// tokens, at this one the recurrent state decides them.
const ROW_RECORDED: [[u32; ROW_NEW]; 2] = [
    [20, 0, 17, 20, 0, 34, 2, 5, 3, 3, 2, 11],
    [37, 14, 34, 17, 16, 29, 6, 3, 37, 22, 46, 15],
];
const SCALE: f32 = 4.0;

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

/// `store` with every tensor but the norm scales multiplied by `factor`.
fn scaled(store: &WeightStore, factor: f32) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (key, entry) in store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture is dense")
        };
        assert_eq!(dense.dtype(), DType::BF16, "{key}");
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

fn write_checkpoint(dir: &Path, fixture: &Fixture, store: &WeightStore) {
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_test_util::safetensors::write(&dir.join("model.safetensors"), store);
}

/// The Qwen3-Next fixture as a loaded handle, its weights scaled by `scale` when given.
fn handle(scale: Option<f32>) -> Arc<ModelHandle> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family == poot_models::qwen3next::FAMILY)
        .expect("qwen3next is a shipped family");
    let fixture = (entry.fixture)();
    let store = match scale {
        Some(factor) => scaled(&fixture.store, factor),
        None => fixture.store.clone(),
    };
    let dir = poot_test_util::unique_temp_path("pooled_qwen3next");
    write_checkpoint(&dir, &fixture, &store);
    Arc::new(ModelHandle::load(&dir, &registry, &LoadOptions::default()).unwrap())
}

fn options(chunk: usize) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(chunk),
        max_trace_tokens: nz(64),
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
    poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
}

fn generate(driver: &mut Driver, prompt: &[u32], max_new: usize) -> Vec<u32> {
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

/// SC-001 (Qwen3-Next tokens): the fixture pooled at `P = 3 < E = 4` decodes its recorded 24 tokens, and so
/// does the unpooled driver; the pool evicted, every stage recorded once, and the executor served and read
/// back at its sync points.
/// Mutation: in `ExpertResidency::sync`, skip the refill of a miss (answer no rows); the pool serves
/// another expert's bytes and the pooled tokens diverge from the recorded ones.
fn pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions(choice: BackendChoice) {
    let handle = handle(None);
    // Both executors open before either driver runs (the ROCm runtime does not reopen one).
    let Some(plain_executor) = open(choice) else {
        return;
    };
    let Some(pooled_executor) = open(choice) else {
        return;
    };
    let mut plain = Driver::new(Arc::clone(&handle), plain_executor, options(8)).unwrap();
    let mut pooled = Driver::new(handle, pooled_executor, options(8))
        .unwrap()
        .with_expert_pool(nz(POOL_ROWS))
        .unwrap();
    assert_eq!(
        generate(&mut plain, &PROMPT, RECORDED.len()),
        RECORDED,
        "unpooled qwen3next on {choice:?}"
    );
    assert_eq!(
        generate(&mut pooled, &PROMPT, RECORDED.len()),
        RECORDED,
        "pooled qwen3next on {choice:?}"
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
        ResidencyCounters::default(),
        "the unpooled driver runs no residency manager"
    );
}

/// One sequence stepped through the serving surface, one row per step.
struct Lane {
    seq: SeqId,
    prompt: &'static [u32],
    prefilled: usize,
    out: Vec<u32>,
    sampler: Sampler,
}

impl Lane {
    fn open(driver: &mut Driver, prompt: &'static [u32]) -> Self {
        let request = SeqRequest {
            prompt,
            max_new: ROW_NEW,
            adapter: AdapterRef::Base,
        };
        let Admission::Opened { seq, reused } = driver.open(request).unwrap() else {
            panic!("no room for a sequence");
        };
        Lane {
            seq,
            prompt,
            prefilled: reused,
            out: Vec::new(),
            sampler: Sampler::greedy(),
        }
    }

    fn done(&self) -> bool {
        self.out.len() == ROW_NEW
    }

    /// One step of this sequence alone: the next prompt chunk, or one decode token.
    fn step(&mut self, driver: &mut Driver) {
        let work = if self.prefilled < self.prompt.len() {
            RowWork::Prefill {
                upto: self.prompt.len(),
            }
        } else {
            RowWork::Decode {
                ahead: &[],
                pick: PickPolicy::Greedy,
            }
        };
        let mut rows = [StepRow {
            seq: self.seq,
            work,
            sampler: &mut self.sampler,
        }];
        let results = driver.step(&mut rows).unwrap();
        let [result] = results.as_slice() else {
            panic!("one row, one result");
        };
        self.prefilled += result.consumed;
        self.out.extend(&result.tokens);
        driver
            .commit(self.seq, result.tokens.len(), &mut self.sampler)
            .unwrap();
    }
}

/// The two sequences' tokens when stepped alternately, one row per step, through `driver`'s paged pool of
/// two state rows: sequence 0 holds row 0 and sequence 1 holds row 1, so each step reads its row's state
/// the other sequence's step has just left beside it.
fn alternating_rows(driver: &mut Driver) -> [Vec<u32>; 2] {
    driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(PoolShape {
                blocks: nz(8),
                max_seqs: nz(2),
            }),
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .unwrap();
    let mut lanes = [Lane::open(driver, &MEDIUM), Lane::open(driver, &SHORT)];
    while lanes.iter().any(|lane| !lane.done()) {
        for lane in lanes.iter_mut().filter(|lane| !lane.done()) {
            lane.step(driver);
        }
    }
    for lane in &lanes {
        driver.release(lane.seq, Release::Finished).unwrap();
    }
    lanes.map(|lane| lane.out)
}

/// SC-001 (Qwen3-Next state rows): two sequences share one pooled executable and one two-row recurrent state
/// pool, and their steps alternate, so between two steps of one sequence the other's step has run every
/// pooled stage and refilled the expert rows. Each sequence's tokens equal the recorded tokens of its solo
/// unpooled run (the scaled fixture's, where the Gated DeltaNet state decides the pick), and the pool
/// evicted. Mutation: in the step data of `Driver::run_plans`, never write state row 1 back
/// (`state_write[1]` stays masked); sequence 1 restarts its recurrence at every step (a zeroed state for
/// row 1) and its tokens diverge from the recorded ones while sequence 0's stay equal.
fn pooled_qwen3next_keeps_each_state_row_across_alternating_steps(choice: BackendChoice) {
    let handle = handle(Some(SCALE));
    let Some(solo_executor) = open(choice) else {
        return;
    };
    let Some(pooled_executor) = open(choice) else {
        return;
    };
    let mut solo = Driver::new(Arc::clone(&handle), solo_executor, options(4)).unwrap();
    let mut pooled = Driver::new(handle, pooled_executor, options(4))
        .unwrap()
        .with_expert_pool(nz(POOL_ROWS))
        .unwrap();
    for (row, prompt) in [&MEDIUM[..], &SHORT[..]].into_iter().enumerate() {
        assert_eq!(
            generate(&mut solo, prompt, ROW_NEW),
            ROW_RECORDED[row],
            "the solo unpooled run of sequence {row} on {choice:?}"
        );
    }
    let got = alternating_rows(&mut pooled);
    for (row, tokens) in got.iter().enumerate() {
        assert_eq!(
            tokens, &ROW_RECORDED[row],
            "pooled qwen3next sequence {row} on {choice:?}"
        );
    }
    let counters = pooled.stats().residency;
    eprintln!("residency counters: {counters:?}");
    assert!(
        counters.refills > 0 && counters.evictions > 0,
        "{counters:?}"
    );
}

#[test]
fn pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions_wgpu() {
    pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Wgpu);
}

#[test]
fn pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions_rocm() {
    pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Rocm);
}

#[test]
fn pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions_ptx() {
    pooled_qwen3next_decodes_the_unpooled_tokens_with_evictions(BackendChoice::Ptx);
}

#[test]
fn pooled_qwen3next_keeps_each_state_row_across_alternating_steps_wgpu() {
    pooled_qwen3next_keeps_each_state_row_across_alternating_steps(BackendChoice::Wgpu);
}

#[test]
fn pooled_qwen3next_keeps_each_state_row_across_alternating_steps_rocm() {
    pooled_qwen3next_keeps_each_state_row_across_alternating_steps(BackendChoice::Rocm);
}

#[test]
fn pooled_qwen3next_keeps_each_state_row_across_alternating_steps_ptx() {
    pooled_qwen3next_keeps_each_state_row_across_alternating_steps(BackendChoice::Ptx);
}
