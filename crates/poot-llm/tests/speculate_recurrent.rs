//! Speculation for recurrent-state families (POOT-631's accepted design, POOT-647): a model whose
//! rejected draft tokens were folded into `StateRole::Recurrent` state verifies windows of drafts,
//! because the driver snapshots the rows it verifies, restores the partial rows and replays their kept
//! prefix (`driver/snapshot.rs`, `driver/speculate.rs`).
//!
//! - Host half (model-free): a recurrent family whose prefill granule is above one token is refused
//!   typed at `prepare`; the same family at granule one prepares and publishes its window.
//! - Device half: both shipped hybrid families (Qwen3-Next's gated delta rule, Nemotron-H's Mamba2)
//!   through their synthetic fixtures, on wgpu and ROCm: a forced partial accept leaves every recurrent
//!   state element where a fresh run of the replay chunk puts it; served greedy speculation returns the
//!   tokens `DraftPolicy::Off` returns; a full-accept row's state does not depend on a partial row
//!   beside it; a sampled row speculates and its rounds restore and replay too.

#[path = "add_a_model/family.rs"]
mod family;

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::{
    Device, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor, ExecutorStats, HostSync,
    NoSync, StateScope, StepInputs, StepOutputs,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::StagedProgram,
    staged::TargetSet, staged::compile_staged,
};
use poot_llm::Sampler;
use poot_llm::driver::block_table::BLOCK_SIZE;
use poot_llm::driver::error::{DriverError, InvalidRequest, SpeculationRefusal, Unsupported};
use poot_llm::driver::speculate::{SpeculationStats, propose_lookup, spec_sample_round};
use poot_llm::driver::{
    AdapterRef, Admission, BackendChoice, DraftPolicy, Driver, DriverOptions, Feature, Head,
    Incompatible, Layout, LoadOptions, ModelHandle, PoolShape, PreparedSetLimits, Release,
    RowResult, RowSpeculation, RowWork, SeqId, SeqRequest, Serving, ServingShapes, State, StepRow,
    Unprepared, Warm, open_executor, program_retention,
};
use poot_models::components::standard::{Step, check_step};
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec, StateSpec};
use poot_models::model::{
    FamilyKey, HiddenCarry, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput,
    StepShape, StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use poot_text::ChatFormat;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

/// The positions of every driver here (both hybrid fixtures' `max_position_embeddings`).
const CAPACITY: usize = 64;
/// The vocabulary of every fixture here.
const VOCAB: usize = 48;
/// Drafts a round proposes.
const K: usize = 3;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Checkpoints.
// ---------------------------------------------------------------------------------------------

/// The hybrid families under test: Qwen3-Next (gated delta rule) and Nemotron-H (Mamba2).
const FAMILIES: [(&str, FamilyKey); 2] = [
    ("gdn (qwen3next)", poot_models::qwen3next::FAMILY),
    ("mamba2 (nemotron_h)", poot_models::nemotron_h::FAMILY),
];

fn entry(family: FamilyKey) -> FamilyEntry {
    Registry::builtin()
        .unwrap()
        .entries()
        .iter()
        .find(|entry| entry.family == family)
        .copied()
        .unwrap_or_else(|| panic!("{family} is a shipped family"))
}

fn tokenizer() -> Tokenizer {
    let words = (0..VOCAB).map(|i| (format!("w{i}"), i as u32)).collect();
    let model = WordLevel::builder()
        .vocab(words)
        .unk_token("w0".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

/// A family's shipped fixture checkpoint, loaded.
fn fixture_handle(family: FamilyKey) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path("speculate_recurrent");
    write_checkpoint(&dir, &(entry(family).fixture)());
    let handle =
        ModelHandle::load(&dir, &Registry::builtin().unwrap(), &LoadOptions::default()).unwrap();
    (Arc::new(handle), dir)
}

fn options(draft: DraftPolicy) -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(4),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(CAPACITY)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
        },
        charge: program_retention,
        draft,
        hidden_window: None,
    }
}

fn lookup(ngram: usize) -> DraftPolicy {
    lookup_k(ngram, K)
}

fn lookup_k(ngram: usize, k: usize) -> DraftPolicy {
    DraftPolicy::PromptLookup {
        ngram: nz(ngram),
        k: nz(k),
    }
}

// ---------------------------------------------------------------------------------------------
// Host half: a recurrent fake family and a fake device.
// ---------------------------------------------------------------------------------------------

const RECURRENT: FamilyKey = FamilyKey::new("poot_fixture_recurrentish");

const RECURRENT_ENTRY: FamilyEntry = FamilyEntry {
    family: RECURRENT,
    keys: &[(ConfigSource::HfModelType, "poot_fixture_recurrentish")],
    build: build_recurrent,
    fixture: recurrent_fixture,
    variants: &[],
};

/// One `StateRole::Recurrent` state folded with every token. Its prefill granule is the config's
/// `prefill_granule`, so the same family is a granule-one and a granule-two model.
#[derive(Debug)]
struct Recurrentish {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
}

fn build_recurrent(raw: &RawConfig<'_>, _: &WeightStore) -> Result<ModelParts, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        panic!("the fixture is an HF checkpoint")
    };
    let granule = config["prefill_granule"].as_u64().unwrap_or(1) as usize;
    Ok(ModelParts::new(Box::new(Recurrentish {
        config: ModelConfig {
            family: RECURRENT,
            vocab: VOCAB,
            max_positions: CAPACITY,
            eos: BTreeSet::from([VOCAB as u32 - 1]),
            bos: None,
            prompt_bos: None,
            prefill_granule: nz(granule),
            chat: ChatFormat::ChatML,
        },
        schema: ModelSchema::builder()
            .step(
                ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: VOCAB })
                    .carrying(StateSpec::recurrent("toy.state")),
            )
            .build_for(RECURRENT)?,
        weights: WeightMap::default(),
    })))
}

fn recurrent_fixture() -> Fixture {
    recurrent_fixture_with(1)
}

fn recurrent_fixture_with(granule: usize) -> Fixture {
    let mut store = WeightStore::builder();
    let dummy = DenseWeight::try_new(DType::BF16, vec![1], vec![0u8, 0].into()).unwrap();
    store.insert("dummy", WeightEntry::Dense(dummy)).unwrap();
    Fixture {
        config: serde_json::json!({
            "model_type": "poot_fixture_recurrentish",
            "vocab_size": VOCAB,
            "max_position_embeddings": CAPACITY,
            "eos_token_id": VOCAB - 1,
            "prefill_granule": granule,
        }),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

impl Model for Recurrentish {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        check_step(RECURRENT, shape, self.config.max_positions)?;
        let step = Step::new(&b, shape)?;
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let first = if shape.logits == LogitRows::All {
            0
        } else {
            n - 1
        };
        let out = n - first;
        let token = b.cast(
            b.slice(
                b.guard_index_bounds(step.tokens(), VOCAB)
                    .expect("the step's ids are a rank-2 I32 tensor")
                    .guarded,
                1,
                first,
                n,
            )?,
            DType::F32,
        )?;
        let pos = b.cast(b.slice(step.pos(), 1, first, n)?, DType::F32)?;
        let seed = b.binary(BinOp::Add, token, pos)?;
        let dims = vec![rows, 4];
        let state = b.state_input(
            "toy.state",
            TensorType::f32(dims.clone()),
            StateRole::Recurrent,
        )?;
        // The state folds the step's last token, whatever width the step has.
        let last = b.slice(seed, 1, out - 1, out)?;
        let folded = b.binary(BinOp::Add, state, b.broadcast(last, dims)?)?;
        step.carry(state, folded);
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1])?, vec![rows, out, VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

/// The recurrent fake at `granule`, loaded through a registry that holds it beside the built-ins.
fn recurrent_handle(granule: usize) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path("speculate_recurrent_fake");
    write_checkpoint(&dir, &recurrent_fixture_with(granule));
    let handle = ModelHandle::load(&dir, &fake_registry(), &LoadOptions::default()).unwrap();
    (Arc::new(handle), dir)
}

/// The built-in families, the recurrent fake and the dense add-a-model fixture family.
fn fake_registry() -> Registry {
    let mut registry = Registry::builtin().unwrap();
    registry.register(RECURRENT_ENTRY).unwrap();
    registry.register(family::ENTRY).unwrap();
    registry
}

/// The dense add-a-model fixture, loaded with the same tokenizer and vocabulary as the fakes.
fn dense_handle() -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    assert_eq!(
        family::VOCAB,
        VOCAB,
        "the dense fixture shares the fakes' vocabulary"
    );
    let dir = poot_test_util::unique_temp_path("speculate_recurrent_dense");
    write_checkpoint(&dir, &family::fixture());
    let handle = ModelHandle::load(&dir, &fake_registry(), &LoadOptions::default()).unwrap();
    (Arc::new(handle), dir)
}

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

/// A device that does nothing: the driver, engine and planner are real, so `prepare` compiles.
struct FakeDevice;

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: DeviceCaps::wgpu_rdna3_igpu(),
        }
    }
    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        Vec::new()
    }
    fn allocate(
        &mut self,
        _role: BufferRole,
        _storage: poot_target::BufferStorage,
        _elems: usize,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    /// This fake keeps no storage, so it states no bound: the engine's length check passes every
    /// dispatch it records.
    fn buffer_elems(&self, _buffer: &()) -> usize {
        usize::MAX
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn begin(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FakeError> {
        Ok(())
    }
    /// Every output is read as the sampling suffix's pairs `(token 0, no non-finite logit)`: a greedy
    /// pick of token 0 on every position.
    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        for pair in out.chunks_mut(8) {
            let mut bytes = [0u8; 8];
            bytes[4..].copy_from_slice(&(-1i32).to_le_bytes());
            let n = pair.len();
            pair.copy_from_slice(&bytes[..n]);
        }
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
}

fn contiguous() -> ServingShapes<'static> {
    ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

fn fake_driver(handle: Arc<ModelHandle>, draft: DraftPolicy) -> Driver {
    Driver::new(handle, Box::new(Engine::new(FakeDevice)), options(draft)).unwrap()
}

/// SC-004: a recurrent family whose prefill granule is 2 is refused at `prepare` with
/// `RecurrentGranule { granule: 2 }`, read from the config of what compiled, and publishes nothing;
/// the same family at granule 1 prepares and publishes its window beside decode. Mutation: drop the
/// granule check in `check_speculation_state`; the granule-2 `prepare` succeeds and the first
/// assertion is red.
#[test]
fn a_granule_two_recurrent_family_refuses_speculation_and_granule_one_prepares() {
    let (coarse, _dir) = recurrent_handle(2);
    let driver = fake_driver(coarse, lookup(2));
    let refused = driver
        .prepare(&contiguous())
        .expect_err("a replay chunk is not a multiple of a two-token granule");
    let (error, driver) = (refused.error, refused.driver);
    let DriverError::Unsupported(Unsupported::Speculation(refusal)) = error else {
        panic!("not a speculation refusal: {error:?}");
    };
    assert_eq!(refusal, SpeculationRefusal::RecurrentGranule { granule: 2 });
    assert!(
        driver.caps().is_none(),
        "a refused prepare publishes nothing"
    );

    let (fine, _dir) = recurrent_handle(1);
    let driver = fake_driver(fine, lookup(2))
        .prepare(&contiguous())
        .unwrap_or_else(|e| panic!("{e}"));
    let caps = driver.caps().unwrap();
    let speculation = caps.speculation.expect("the policy is published");
    assert_eq!((speculation.k.get(), speculation.window.get()), (K, K + 1));
    assert!(
        caps.windows
            .iter()
            .any(|w| w.rows.get() == 1 && w.width.get() == K + 1),
        "one verify entry beside decode: {:?}",
        caps.windows
    );
}

/// A draft model with recurrent state is refused typed at `prepare`: only the target's state is
/// rolled back, so a rejected draft token would stay folded into the draft's. The target is the
/// dense fixture; the draft is the recurrent fake. Mutation: remove the `key.role == Draft` check in
/// `Driver::prepared`; the draft compiles, `prepare` succeeds and the first assertion is red.
#[test]
fn a_recurrent_draft_model_is_refused() {
    let (target, _target_dir) = dense_handle();
    let (draft, _draft_dir) = recurrent_handle(1);
    let driver = fake_driver(
        target,
        DraftPolicy::DraftModel {
            handle: draft,
            k: nz(K),
        },
    );
    let refused = driver
        .prepare(&contiguous())
        .expect_err("a recurrent draft cannot be rolled back");
    let (error, driver) = (refused.error, refused.driver);
    assert!(
        matches!(
            error,
            DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                first: Feature::DraftModel,
                second: Feature::RecurrentDraftState,
                ..
            }))
        ),
        "not a draft-model/recurrent-draft-state refusal: {error:?}"
    );
    assert!(
        driver.caps().is_none(),
        "a refused prepare publishes nothing"
    );
}

/// One verified round of the fake device, which reads every pick as token 0, so prompt lookup over a prompt of
/// zeros proposes drafts that all verify.
fn verified_round(driver: &mut Driver<Serving>) -> (SeqId, RowResult) {
    let (seq, _) = open_prefilled(driver, &[0, 0, 0, 0], 8);
    let result = decode_step(driver, seq, &mut Sampler::greedy());
    assert!(
        matches!(
            result.speculation,
            RowSpeculation::Verified { proposed, accepted } if proposed > 0 && accepted == proposed
        ),
        "the round verified its drafts: {:?}",
        result.speculation
    );
    (seq, result)
}

/// A recurrent sequence whose round folded a verified result into its state cannot commit fewer
/// tokens and keep stepping: the state holds the dropped ones. The shorter commit is accepted (the
/// sequence may end there), and the next step is `RecurrentStateAhead`; committing the whole result
/// steps on. Mutation: drop the `state_ahead` check in `plan_rows`; the step after the short commit
/// succeeds and the `expect_err` is red.
#[test]
fn a_short_commit_of_a_verified_recurrent_result_refuses_the_next_step() {
    let (handle, _dir) = recurrent_handle(1);
    let mut driver = fake_driver(Arc::clone(&handle), lookup(1))
        .prepare(&contiguous())
        .unwrap_or_else(|e| panic!("{e}"));
    let (seq, result) = verified_round(&mut driver);
    assert!(
        result.tokens.len() >= 2,
        "a drafted result holds more than one token"
    );
    driver
        .commit(seq, result.tokens.len() - 1, &mut Sampler::greedy())
        .unwrap();
    let mut sampler = Sampler::greedy();
    let mut rows = [StepRow {
        seq,
        work: RowWork::Decode { ahead: &[] },
        sampler: &mut sampler,
    }];
    let error = driver
        .step(&mut rows)
        .expect_err("the state holds the dropped token");
    assert!(
        matches!(
            error,
            DriverError::InvalidRequest(InvalidRequest::RecurrentStateAhead(s)) if s == seq
        ),
        "{error:?}"
    );

    // The whole result keeps the state and the sequence consistent: the next step runs.
    let mut whole = fake_driver(handle, lookup(1))
        .prepare(&contiguous())
        .unwrap_or_else(|e| panic!("{e}"));
    let (seq, result) = verified_round(&mut whole);
    whole
        .commit(seq, result.tokens.len(), &mut Sampler::greedy())
        .unwrap();
    decode_step(&mut whole, seq, &mut Sampler::greedy());
}

// ---------------------------------------------------------------------------------------------
// Device half.
// ---------------------------------------------------------------------------------------------

/// One decode step of `seq` under the driver's policy.
fn decode_step(
    driver: &mut Driver<Serving>,
    seq: poot_llm::driver::SeqId,
    sampler: &mut Sampler,
) -> RowResult {
    let mut rows = [StepRow {
        seq,
        work: RowWork::Decode { ahead: &[] },
        sampler,
    }];
    driver
        .step(&mut rows)
        .unwrap_or_else(|e| panic!("{e}"))
        .remove(0)
}

/// What a served request returned: the tokens, and what the policy did each round.
struct Served {
    tokens: Vec<u32>,
    rounds: Vec<RowSpeculation>,
}

/// `new` tokens after `prompt` through `open`/`step`/`commit`, committing each round whole (a
/// recurrent row's state holds its folded kept prefix, so a round is kept entirely or not at all).
fn serve(
    driver: &mut Driver<Serving>,
    prompt: &[u32],
    new: usize,
    sampler: &mut Sampler,
    heads: &[Head],
) -> Served {
    driver
        .extend(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads,
            warm: Warm::Prompts(&[prompt.len()]),
            windows: &[],
            adapters: &[],
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let compiled = driver.stats().compiles;
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest::new(prompt, new, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits one sequence")
    };
    let mut tokens = Vec::new();
    let mut rounds = Vec::new();
    let mut result = loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut *sampler,
        }];
        let result = driver
            .step(&mut rows)
            .unwrap_or_else(|e| panic!("{e}"))
            .remove(0);
        if !result.tokens.is_empty() {
            break result;
        }
    };
    loop {
        let kept = result.tokens.len().min(new - tokens.len());
        driver.commit(seq, kept, sampler).unwrap();
        tokens.extend_from_slice(&result.tokens[..kept]);
        rounds.push(result.speculation);
        if tokens.len() == new {
            break;
        }
        result = decode_step(driver, seq, sampler);
    }
    driver.release(seq, Release::Finished).unwrap();
    assert_eq!(
        driver.stats().compiles,
        compiled,
        "serving the prepared set compiles nothing, whatever the rounds propose"
    );
    Served { tokens, rounds }
}

/// A driver for `family`'s fixture on `backend`, or `None` when the device is unavailable.
fn device_driver(
    handle: &Arc<ModelHandle>,
    choice: BackendChoice,
    backend: DeviceBackend,
    draft: DraftPolicy,
) -> Option<Driver> {
    let executor = poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))?;
    Some(Driver::new(Arc::clone(handle), executor, options(draft)).unwrap())
}

/// Whether a round verified drafts and kept some but not all of them.
fn is_partial(round: &RowSpeculation) -> bool {
    matches!(round, RowSpeculation::Verified { proposed, accepted } if 0 < *accepted && accepted < proposed)
}

/// SC-002: served greedy speculation (prompt lookup over a repetitive prompt) on a hybrid fixture
/// returns the tokens `DraftPolicy::Off` returns for the same prompt and chunking, and at least one
/// round kept some but not all of its drafts, so the restore-and-replay path ran. The driver's own
/// counters agree that rounds restored and replayed. Mutation: replay only the accepted drafts, not
/// the query token (`roll_back` feeds `pending[..kept - 1]` alone); the state after a partial round
/// misses the query token and the tokens diverge from `Off`.
fn assert_greedy_lookup_equals_off(
    family: FamilyKey,
    choice: BackendChoice,
    backend: DeviceBackend,
    prompt: &[u32],
) {
    let (handle, _dir) = fixture_handle(family);
    let Some(off) = device_driver(&handle, choice, backend, DraftPolicy::Off) else {
        return;
    };
    let mut off = off.serve(Layout::Contiguous);
    let reference = serve(
        &mut off,
        prompt,
        NEW_TOKENS,
        &mut Sampler::greedy(),
        &[Head::GREEDY],
    );
    let mut driver = device_driver(&handle, choice, backend, lookup(1))
        .expect("the device opened a moment ago")
        .serve(Layout::Contiguous);
    let served = serve(
        &mut driver,
        prompt,
        NEW_TOKENS,
        &mut Sampler::greedy(),
        &[Head::GREEDY],
    );
    assert_eq!(
        served.tokens, reference.tokens,
        "speculation differs from Off; rounds {:?}",
        served.rounds
    );
    assert!(
        served.rounds.iter().any(is_partial),
        "no round kept some but not all of its drafts: {:?}",
        served.rounds
    );
    let SpeculationStats {
        restored, replayed, ..
    } = driver.stats().speculation;
    assert!(
        restored > 0 && replayed >= restored,
        "the partial rounds restored and replayed: {restored} rows, {replayed} tokens"
    );
}

/// Tokens every device row generates.
const NEW_TOKENS: usize = 32;

/// A prompt that repeats itself, so prompt lookup proposes from the first round.
const REPETITIVE: [u32; 9] = [5, 27, 3, 5, 27, 3, 5, 27, 3];

const GDN: FamilyKey = FAMILIES[0].1;
const MAMBA2: FamilyKey = FAMILIES[1].1;

#[test]
fn greedy_prompt_lookup_equals_off_on_gdn_wgpu() {
    assert_greedy_lookup_equals_off(GDN, BackendChoice::Wgpu, DeviceBackend::Wgpu, &REPETITIVE);
}

#[test]
fn greedy_prompt_lookup_equals_off_on_gdn_rocm() {
    assert_greedy_lookup_equals_off(GDN, BackendChoice::Rocm, DeviceBackend::Rocm, &REPETITIVE);
}

#[test]
fn greedy_prompt_lookup_equals_off_on_gdn_ptx() {
    assert_greedy_lookup_equals_off(GDN, BackendChoice::Ptx, DeviceBackend::Ptx, &REPETITIVE);
}

#[test]
fn greedy_prompt_lookup_equals_off_on_mamba2_wgpu() {
    assert_greedy_lookup_equals_off(
        MAMBA2,
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        &REPETITIVE,
    );
}

#[test]
fn greedy_prompt_lookup_equals_off_on_mamba2_rocm() {
    assert_greedy_lookup_equals_off(
        MAMBA2,
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        &REPETITIVE,
    );
}

#[test]
fn greedy_prompt_lookup_equals_off_on_mamba2_ptx() {
    assert_greedy_lookup_equals_off(MAMBA2, BackendChoice::Ptx, DeviceBackend::Ptx, &REPETITIVE);
}

// ---------------------------------------------------------------------------------------------
// Reading carried state through a production step output.
// ---------------------------------------------------------------------------------------------

/// What a [`Probed`] executor shares with its test.
#[derive(Default)]
struct ProbeShared {
    /// The first executable the driver loaded: the target's.
    exe: Option<ExecutableId>,
    /// The recurrent values to read before the next step, by state name and aval.
    armed: Vec<(String, TensorType)>,
    /// What the last armed read returned, in the order it was armed.
    read: Vec<HostTensor>,
    entries: HashMap<String, EntryId>,
}

/// An executor that reads carried state on request. Executor state has no read accessor; the test adds
/// an entry per state value whose program is the identity over it (a state input aliased to itself,
/// returned as the output), shares its buffer by name and aval with the driver's entries, and steps
/// it. The read happens before the next step the driver issues, so it sees the state the step before
/// left.
struct Probed {
    inner: Box<dyn Executor>,
    shared: Rc<RefCell<ProbeShared>>,
}

impl Probed {
    fn probe_entry(
        &mut self,
        exe: ExecutableId,
        name: &str,
        aval: &TensorType,
    ) -> Result<EntryId, ExecError> {
        if let Some(&entry) = self.shared.borrow().entries.get(name) {
            return Ok(entry);
        }
        let b = Builder::for_tests();
        let state = b
            .state_input(name, aval.clone(), StateRole::Recurrent)
            .unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
            .unwrap();
        let partition = Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        };
        let options = CompileOptions::new(CompileLimits::STANDARD);
        let program = compile_staged(&graph, &self.inner.target_set(), &partition, &options)
            .unwrap_or_else(|e| panic!("the probe of {name} compiles: {e}"));
        let entry = self.inner.add_entry(exe, &program)?;
        self.shared
            .borrow_mut()
            .entries
            .insert(name.to_string(), entry);
        Ok(entry)
    }
}

impl Executor for Probed {
    fn target_set(&self) -> TargetSet {
        self.inner.target_set()
    }
    fn free_memory_bytes(&self) -> Option<u64> {
        self.inner.free_memory_bytes()
    }
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError> {
        let exe = self.inner.load_weights(store, weights)?;
        self.shared.borrow_mut().exe.get_or_insert(exe);
        Ok(exe)
    }
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
    ) -> Result<EntryId, ExecError> {
        self.inner.add_entry(exe, program)
    }
    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        let armed = std::mem::take(&mut self.shared.borrow_mut().armed);
        if !armed.is_empty() && self.shared.borrow().exe == Some(exe) {
            let mut read = Vec::with_capacity(armed.len());
            for (name, aval) in &armed {
                let probe = self.probe_entry(exe, name, aval)?;
                let mut outputs = self
                    .inner
                    .step(exe, probe, &StepInputs::new(), &mut NoSync)?;
                read.push(outputs.to_host()?);
            }
            self.shared.borrow_mut().read = read;
        }
        self.inner.step(exe, entry, inputs, sync)
    }
    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError> {
        self.inner.reset_state(exe, scope)
    }
    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        self.inner.remove_entry(exe, entry)
    }
    fn stats(&self) -> ExecutorStats {
        self.inner.stats()
    }
    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        self.inner.unload(exe)
    }
}

/// A driver whose recurrent state `shared` can read.
struct Probing<S: State = Unprepared> {
    driver: Driver<S>,
    shared: Rc<RefCell<ProbeShared>>,
    schema: Vec<(String, TensorType)>,
}

impl Probing<Unprepared> {
    /// The same probe over the driver once it has prepared `shapes`.
    fn prepare(self, shapes: &ServingShapes<'_>) -> Probing<Serving> {
        Probing {
            driver: self
                .driver
                .prepare(shapes)
                .unwrap_or_else(|e| panic!("{e}")),
            shared: self.shared,
            schema: self.schema,
        }
    }
}

impl<S: State> Probing<S> {
    /// Read the recurrent state before the next step the driver issues.
    fn arm(&self) {
        self.shared.borrow_mut().armed.clone_from(&self.schema);
    }

    /// The state the last armed read returned, per recurrent value.
    fn read(&self) -> Vec<Vec<f32>> {
        let shared = self.shared.borrow();
        assert_eq!(shared.read.len(), self.schema.len(), "an armed read ran");
        shared
            .read
            .iter()
            .map(|t| t.as_f32().expect("recurrent state is f32").to_vec())
            .collect()
    }
}

/// The values of `family`'s decode program over `kv` a probe reads: its recurrent state, or under a
/// hidden-state carry of `hidden` positions the carry itself. Name and aval, as the driver's entries
/// declare them.
fn probe_schema(
    family: FamilyKey,
    kv: KvLayout,
    hidden: Option<NonZeroUsize>,
) -> Vec<(String, TensorType)> {
    let fixture = (entry(family).fixture)();
    let model = (entry(family).build)(&fixture.raw(), &fixture.store).unwrap();
    let graph = model
        .model
        .trace(
            &StepSpec::step(
                StepShape::new(NonZeroUsize::MIN, NonZeroUsize::MIN, nz(CAPACITY))
                    .with_kv(kv)
                    .with_hidden(hidden.map(|positions| HiddenCarry { positions })),
            ),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let role = if hidden.is_some() {
        StateRole::Handoff
    } else {
        StateRole::Recurrent
    };
    graph
        .state_pairs()
        .filter(|pair| pair.role == role)
        .map(|pair| {
            let meta = graph.meta(pair.input);
            (
                meta.name.clone().expect("a state input carries its name"),
                meta.aval.clone(),
            )
        })
        .collect()
}

fn probing_driver(
    family: FamilyKey,
    handle: &Arc<ModelHandle>,
    choice: BackendChoice,
    backend: DeviceBackend,
    draft: DraftPolicy,
    kv: KvLayout,
    hidden: Option<NonZeroUsize>,
) -> Option<Probing> {
    let executor = poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))?;
    let shared = Rc::new(RefCell::new(ProbeShared::default()));
    let probed = Probed {
        inner: executor,
        shared: Rc::clone(&shared),
    };
    Some(Probing {
        driver: Driver::new(
            Arc::clone(handle),
            Box::new(probed),
            DriverOptions {
                hidden_window: hidden,
                ..options(draft)
            },
        )
        .unwrap(),
        shared,
        schema: probe_schema(family, kv, hidden),
    })
}

// ---------------------------------------------------------------------------------------------
// SC-001: a forced partial accept leaves the state a fresh replay chunk puts it in.
// ---------------------------------------------------------------------------------------------

/// Open `prompt` on a prepared driver and prefill it, returning the sequence and its first token
/// (committed).
fn open_prefilled(driver: &mut Driver<Serving>, prompt: &[u32], max_new: usize) -> (SeqId, u32) {
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest::new(prompt, max_new, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits the sequence")
    };
    let mut sampler = Sampler::greedy();
    let first = loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut sampler,
        }];
        let result = driver
            .step(&mut rows)
            .unwrap_or_else(|e| panic!("{e}"))
            .remove(0);
        if let Some(&token) = result.tokens.first() {
            break token;
        }
    };
    driver.commit(seq, 1, &mut sampler).unwrap();
    (seq, first)
}

fn prepare_for(probing: Probing, prompt_len: usize) -> Probing<Serving> {
    probing.prepare(&ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Prompts(&[prompt_len]),
        windows: &[],
        adapters: &[],
    })
}

/// SC-001: on `family`'s fixture with `k = 3`, a round whose first draft verifies and whose second
/// does not (`a = 1`) leaves every recurrent state element exactly where a fresh run of the replay
/// chunk (the query token and the one accepted draft, one width-2 prefill piece) over the same prompt
/// state puts it, on the same device: the reference driver (`DraftPolicy::Off`) prefills the prompt
/// in the same pieces and then the two kept tokens as one piece. The window had folded all four
/// positions, so only the restore undoes the two rejected ones. Mutation: skip the restore entry
/// (`roll_back` returns before `copy_state_rows(Restore, ..)`); the live state keeps the rejected
/// drafts, the replay folds the query token and the accepted draft on top of them and the state
/// diverges from the reference.
fn assert_partial_accept_replays_the_kept_prefix(
    family: FamilyKey,
    choice: BackendChoice,
    backend: DeviceBackend,
    prompt: &[u32],
) {
    let (handle, _dir) = fixture_handle(family);
    let Some(speculating) = probing_driver(
        family,
        &handle,
        choice,
        backend,
        lookup(1),
        KvLayout::Contiguous,
        None,
    ) else {
        return;
    };
    let mut speculating = prepare_for(speculating, prompt.len());
    let (seq, query) = open_prefilled(&mut speculating.driver, prompt, 8);
    let mut sampler = Sampler::greedy();
    let round = decode_step(&mut speculating.driver, seq, &mut sampler);
    assert_eq!(
        round.speculation,
        RowSpeculation::Verified {
            proposed: K,
            accepted: 1
        },
        "the prompt forces a partial accept in its first round"
    );
    let kept = round.tokens.clone();
    speculating
        .driver
        .commit(seq, kept.len(), &mut sampler)
        .unwrap();
    // The next step reads the state the round left, before it runs.
    speculating.arm();
    decode_step(&mut speculating.driver, seq, &mut sampler);
    let state = speculating.read();

    // The reference: the prompt in the same pieces, then [query, accepted draft] as one piece.
    let mut replayed = prompt.to_vec();
    replayed.extend([query, kept[0]]);
    let Some(reference) = probing_driver(
        family,
        &handle,
        choice,
        backend,
        DraftPolicy::Off,
        KvLayout::Contiguous,
        None,
    ) else {
        panic!("the device opened a moment ago")
    };
    let mut reference = prepare_for(reference, prompt.len());
    let Admission::Opened { seq, .. } = reference
        .driver
        .open(SeqRequest::new(&replayed, 8, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits the sequence")
    };
    let mut sampler = Sampler::greedy();
    let mut fed = 0;
    for upto in [prompt.len(), replayed.len()] {
        while fed < upto {
            let mut rows = [StepRow {
                seq,
                work: RowWork::Prefill { upto },
                sampler: &mut sampler,
            }];
            let result = reference
                .driver
                .step(&mut rows)
                .unwrap_or_else(|e| panic!("{e}"))
                .remove(0);
            fed += result.consumed;
        }
    }
    assert_eq!(fed, replayed.len());
    reference.driver.commit(seq, 1, &mut sampler).unwrap();
    reference.arm();
    decode_step(&mut reference.driver, seq, &mut sampler);
    let expected = reference.read();

    assert_eq!(
        speculating.schema.len(),
        expected.len(),
        "every recurrent value is compared"
    );
    for ((name, _), (got, want)) in speculating.schema.iter().zip(state.iter().zip(&expected)) {
        assert_eq!(got, want, "recurrent state {name} after restore and replay");
    }
}

/// Prompts whose first round proposes three drafts of which exactly the first verifies, found by
/// searching random prompts of the fixture (greedy continuation against what prompt lookup proposes).
const GDN_PARTIAL_FIRST_ROUND: [u32; 18] = [
    13, 2, 7, 37, 37, 37, 30, 8, 7, 15, 1, 23, 25, 20, 10, 15, 27, 26,
];
const MAMBA2_PARTIAL_FIRST_ROUND: [u32; 18] = [
    33, 36, 40, 16, 39, 29, 19, 21, 41, 17, 10, 13, 31, 26, 31, 27, 36, 27,
];

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_gdn_wgpu() {
    assert_partial_accept_replays_the_kept_prefix(
        GDN,
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        &GDN_PARTIAL_FIRST_ROUND,
    );
}

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_gdn_rocm() {
    assert_partial_accept_replays_the_kept_prefix(
        GDN,
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        &GDN_PARTIAL_FIRST_ROUND,
    );
}

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_gdn_ptx() {
    assert_partial_accept_replays_the_kept_prefix(
        GDN,
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        &GDN_PARTIAL_FIRST_ROUND,
    );
}

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_mamba2_wgpu() {
    assert_partial_accept_replays_the_kept_prefix(
        MAMBA2,
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        &MAMBA2_PARTIAL_FIRST_ROUND,
    );
}

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_mamba2_rocm() {
    assert_partial_accept_replays_the_kept_prefix(
        MAMBA2,
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        &MAMBA2_PARTIAL_FIRST_ROUND,
    );
}

#[test]
fn a_partial_accept_replays_the_kept_prefix_on_mamba2_ptx() {
    assert_partial_accept_replays_the_kept_prefix(
        MAMBA2,
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        &MAMBA2_PARTIAL_FIRST_ROUND,
    );
}

// ---------------------------------------------------------------------------------------------
// SC-003: a full-accept row's state does not depend on the partial row beside it.
// ---------------------------------------------------------------------------------------------

/// Rounds each row of the batch row steps through.
const BATCH_ROUNDS: usize = 9;

/// One decode step over `seqs` (greedy rows), each result kept whole.
fn decode_rows(driver: &mut Driver<Serving>, seqs: &[SeqId]) -> Vec<RowResult> {
    let mut samplers: Vec<Sampler> = seqs.iter().map(|_| Sampler::greedy()).collect();
    let mut rows: Vec<StepRow<'_>> = seqs
        .iter()
        .zip(&mut samplers)
        .map(|(&seq, sampler)| StepRow {
            seq,
            work: RowWork::Decode { ahead: &[] },
            sampler,
        })
        .collect();
    let results = driver.step(&mut rows).unwrap_or_else(|e| panic!("{e}"));
    drop(rows);
    for (result, sampler) in results.iter().zip(&mut samplers) {
        driver
            .commit(result.seq, result.tokens.len(), sampler)
            .unwrap();
    }
    results
}

/// A paged driver over a two-sequence pool for `family` with `policy`, prepared for `prompts` at the
/// two-row entry (a step with one live row runs the same entry, its second row masked).
fn paged_probing_driver(
    family: FamilyKey,
    handle: &Arc<ModelHandle>,
    choice: BackendChoice,
    backend: DeviceBackend,
    policy: DraftPolicy,
    prompts: &[&[u32]],
) -> Option<Probing<Serving>> {
    let pool = PoolShape {
        blocks: nz(16),
        max_seqs: nz(2),
    };
    let kv = KvLayout::Paged {
        pool_slots: nz(pool.blocks.get() * BLOCK_SIZE),
        seqs: pool.max_seqs,
    };
    let probing = probing_driver(family, handle, choice, backend, policy, kv, None)?;
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    Some(probing.prepare(&ServingShapes {
        layout: Layout::Paged(pool),
        rows: &[nz(2)],
        heads: &[Head::GREEDY],
        warm: Warm::Prompts(&lens),
        windows: &[],
        adapters: &[],
    }))
}

/// The first row slot (sequence slot 0) of every recurrent value of a two-sequence pool.
fn first_slot(state: &[Vec<f32>]) -> Vec<&[f32]> {
    state.iter().map(|v| &v[..v.len() / 2]).collect()
}

/// SC-003: two rows share a step. Row 0 (`prompts[0]`) fully accepts its drafts in some round in which
/// row 1 (`prompts[1]`) has a draft rejected and restores. After every round, row 0's recurrent state
/// equals the state of a driver that served row 0 alone, every element: the restore and the replay
/// touch the partial row's slot and no other. The row asserts that such a round occurred, so the
/// comparison is not vacuous. Mutation: restore every verified row instead of the partial rows only
/// (`roll_back` collects a slot for every settled row); row 0 loses its accepted drafts' state and the
/// states diverge in that round.
fn assert_a_partial_row_leaves_a_full_accept_row_alone(
    family: FamilyKey,
    choice: BackendChoice,
    backend: DeviceBackend,
    k: usize,
    prompts: [&[u32]; 2],
) {
    let (handle, _dir) = fixture_handle(family);
    let policy = || lookup_k(1, k);
    let Some(mut batch) =
        paged_probing_driver(family, &handle, choice, backend, policy(), &prompts)
    else {
        return;
    };
    let mut solo = paged_probing_driver(family, &handle, choice, backend, policy(), &prompts)
        .expect("the device opened a moment ago");
    let max_new = 40;
    let (batch_row0, _) = open_prefilled(&mut batch.driver, prompts[0], max_new);
    let (batch_row1, _) = open_prefilled(&mut batch.driver, prompts[1], max_new);
    let (solo_row0, _) = open_prefilled(&mut solo.driver, prompts[0], max_new);

    let mut isolated_rounds = 0;
    for round in 0..BATCH_ROUNDS {
        batch.arm();
        solo.arm();
        let both = decode_rows(&mut batch.driver, &[batch_row0, batch_row1]);
        let alone = decode_rows(&mut solo.driver, &[solo_row0]);
        // What each step read is the state the round before left.
        let (together, apart) = (batch.read(), solo.read());
        for ((name, _), (got, want)) in batch
            .schema
            .iter()
            .zip(first_slot(&together).into_iter().zip(first_slot(&apart)))
        {
            assert_eq!(
                got, want,
                "round {round}: row 0's recurrent state {name} depends on row 1"
            );
        }
        assert_eq!(
            both[0].tokens, alone[0].tokens,
            "round {round}: row 0's tokens depend on row 1"
        );
        let full = both[0].speculation
            == RowSpeculation::Verified {
                proposed: k,
                accepted: k,
            };
        let rejected = matches!(
            both[1].speculation,
            RowSpeculation::Verified { proposed, accepted } if accepted < proposed
        );
        isolated_rounds += usize::from(full && rejected);
    }
    // The read after the last round: one more step on each.
    batch.arm();
    solo.arm();
    decode_rows(&mut batch.driver, &[batch_row0, batch_row1]);
    decode_rows(&mut solo.driver, &[solo_row0]);
    let (together, apart) = (batch.read(), solo.read());
    for ((name, _), (got, want)) in batch
        .schema
        .iter()
        .zip(first_slot(&together).into_iter().zip(first_slot(&apart)))
    {
        assert_eq!(got, want, "final: row 0's recurrent state {name}");
    }
    assert!(
        isolated_rounds > 0,
        "no round had row 0 fully accept beside a rejecting row 1"
    );
}

/// Prompt pairs found by simulating prompt lookup against `DraftPolicy::Off`'s greedy tokens: row 0
/// fully accepts and row 1 is rejected in the same step (the eighth).
const GDN_FULL_BESIDE_PARTIAL: ([u32; 9], [u32; 13]) = (
    [1, 1, 3, 2, 4, 0, 4, 2, 4],
    [0, 7, 2, 3, 7, 0, 6, 4, 3, 4, 8, 5, 5],
);
const MAMBA2_FULL_BESIDE_PARTIAL: ([u32; 10], [u32; 13]) = (
    [1, 4, 0, 3, 2, 4, 2, 3, 6, 4],
    [0, 7, 2, 3, 7, 0, 6, 4, 3, 4, 8, 5, 5],
);

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_gdn_wgpu() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        GDN,
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        2,
        [&GDN_FULL_BESIDE_PARTIAL.0, &GDN_FULL_BESIDE_PARTIAL.1],
    );
}

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_gdn_rocm() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        GDN,
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        2,
        [&GDN_FULL_BESIDE_PARTIAL.0, &GDN_FULL_BESIDE_PARTIAL.1],
    );
}

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_gdn_ptx() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        GDN,
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        2,
        [&GDN_FULL_BESIDE_PARTIAL.0, &GDN_FULL_BESIDE_PARTIAL.1],
    );
}

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_mamba2_wgpu() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        MAMBA2,
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        3,
        [&MAMBA2_FULL_BESIDE_PARTIAL.0, &MAMBA2_FULL_BESIDE_PARTIAL.1],
    );
}

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_mamba2_rocm() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        MAMBA2,
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        3,
        [&MAMBA2_FULL_BESIDE_PARTIAL.0, &MAMBA2_FULL_BESIDE_PARTIAL.1],
    );
}

#[test]
fn a_partial_row_leaves_a_full_accept_row_alone_on_mamba2_ptx() {
    assert_a_partial_row_leaves_a_full_accept_row_alone(
        MAMBA2,
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        3,
        [&MAMBA2_FULL_BESIDE_PARTIAL.0, &MAMBA2_FULL_BESIDE_PARTIAL.1],
    );
}

// ---------------------------------------------------------------------------------------------
// The sampled path.
// ---------------------------------------------------------------------------------------------

/// The logits of every position of `tokens` under `family`'s fixture, from one prefill over the
/// prefix at position 0 on the CPU oracle: state-free, so a recurrent family's position `i` reads the
/// state its first `i` tokens built.
fn oracle_rows(family: FamilyKey, tokens: &[u32]) -> Vec<Vec<f32>> {
    let n = tokens.len();
    let fixture = (entry(family).fixture)();
    let model = (entry(family).build)(&fixture.raw(), &fixture.store).unwrap();
    let shape = StepShape::new(NonZeroUsize::MIN, nz(n), nz(CAPACITY)).with_logits(LogitRows::All);
    let graph = model
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let state: Vec<HostTensor> = graph
        .state()
        .map(|(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
        .collect();
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![1, n], tokens.iter().map(|&t| t as i32).collect()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![1, n], (0..n as i32).collect()),
        ),
    ];
    let (logits, _) = eval_mapped(
        &graph,
        &fixture.store,
        model.model.weights(),
        &slots,
        &state,
    );
    logits
        .as_f32()
        .expect("f32 logits")
        .chunks(VOCAB)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The tokens a host replay of the sampled rounds produces: the oracle's `p` rows, prompt lookup's
/// drafts with their one-hot `q`, and `spec_sample_round` over them with the row's own sampler. A
/// round with no drafts is one ordinary pick.
fn sampled_reference(
    family: FamilyKey,
    prompt: &[u32],
    new: usize,
    seed: u64,
) -> (Vec<u32>, usize) {
    let mut sampler = Sampler::new(1.0, 0, 1.0, seed);
    let mut tokens = prompt.to_vec();
    let first = sampler
        .pick(oracle_rows(family, &tokens).last().unwrap())
        .unwrap() as u32;
    tokens.push(first);
    let reserved = prompt.len() + new;
    let mut restoring_rounds = 0;
    while tokens.len() < reserved {
        let committed = tokens.len();
        let room = (reserved - committed - 1).min(K);
        let drafts = propose_lookup(&tokens, 1, room);
        let mut window = tokens.clone();
        window.extend_from_slice(&drafts);
        let rows = oracle_rows(family, &window);
        if drafts.is_empty() {
            let token = sampler.pick(&rows[committed - 1]).unwrap() as u32;
            tokens.push(token);
            continue;
        }
        let q: Vec<Vec<f32>> = drafts
            .iter()
            .map(|&d| {
                let mut row = vec![0.0; VOCAB];
                row[d as usize] = 1.0;
                row
            })
            .collect();
        let p: Vec<Vec<f32>> = rows[committed - 1..]
            .iter()
            .map(|row| sampler.probs(row).unwrap())
            .collect();
        let (accepted, next) = spec_sample_round(&drafts, &q, &p, &mut sampler).unwrap();
        // Rounds the driver restores: those that reject a draft.
        restoring_rounds += usize::from(accepted < drafts.len());
        tokens.extend_from_slice(&drafts[..accepted]);
        tokens.push(next);
    }
    (tokens.split_off(prompt.len()), restoring_rounds)
}

/// The seed of the sampled row, and the temperature (T = 1: the distribution is the softmax).
const SEED: u64 = 0x647;

/// A short prompt with a small alphabet, so the sampled tokens recur and lookup finds drafts.
const SAMPLED_PROMPT: [u32; 8] = [5, 9, 3, 5, 9, 3, 5, 9];

/// The sampled path: a temperature-1 row speculating by prompt lookup on a hybrid fixture returns
/// the tokens a host replay of the same rounds returns (the oracle's `p`, the one-hot `q`, accept and
/// residual draws from the row's own stream), and the rounds it restored and replayed are the ones
/// the replay says rejected a draft, so the sampled logits head verifies
/// through the same rollback as the greedy suffix. Mutation: take the argmax for a sampled row's
/// verify (greedy acceptance in `verify_row`), or skip the restore; the tokens diverge.
fn assert_sampled_lookup_matches_the_host_replay(
    family: FamilyKey,
    choice: BackendChoice,
    backend: DeviceBackend,
) {
    let (handle, _dir) = fixture_handle(family);
    let Some(driver) = device_driver(&handle, choice, backend, lookup(1)) else {
        return;
    };
    let mut driver = driver.serve(Layout::Contiguous);
    let (reference, restoring) = sampled_reference(family, &SAMPLED_PROMPT, NEW_TOKENS, SEED);
    assert!(
        restoring > 0,
        "the replay has no round that restores: the prompt proves nothing"
    );
    let mut sampler = Sampler::new(1.0, 0, 1.0, SEED);
    let heads = [Head::Sample(poot_graph_ir::op::SampleRule::Gumbel)];
    let served = serve(
        &mut driver,
        &SAMPLED_PROMPT,
        NEW_TOKENS,
        &mut sampler,
        &heads,
    );
    assert_eq!(
        served.tokens, reference,
        "the device's sampled rounds {:?} differ from the host replay",
        served.rounds
    );
    let SpeculationStats { restored, .. } = driver.stats().speculation;
    assert_eq!(
        restored as usize, restoring,
        "rows restored on the device against rounds the replay restores"
    );
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_gdn_wgpu() {
    assert_sampled_lookup_matches_the_host_replay(GDN, BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_gdn_rocm() {
    assert_sampled_lookup_matches_the_host_replay(GDN, BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_gdn_ptx() {
    assert_sampled_lookup_matches_the_host_replay(GDN, BackendChoice::Ptx, DeviceBackend::Ptx);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_mamba2_wgpu() {
    assert_sampled_lookup_matches_the_host_replay(MAMBA2, BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_mamba2_rocm() {
    assert_sampled_lookup_matches_the_host_replay(MAMBA2, BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_on_mamba2_ptx() {
    assert_sampled_lookup_matches_the_host_replay(MAMBA2, BackendChoice::Ptx, DeviceBackend::Ptx);
}

// ---------------------------------------------------------------------------------------------
// Cost of the rollback on a device (recorded, never gated).
// ---------------------------------------------------------------------------------------------

/// Prompts of the cost rows: a cycle (lookup always has a match), the first-round-partial prompts
/// and the full-beside-partial prompts of the rows above, so the measured rounds mix full accepts,
/// partial accepts, rejections and rounds with no proposal.
fn cost_prompts() -> Vec<Vec<u32>> {
    vec![
        REPETITIVE.to_vec(),
        GDN_PARTIAL_FIRST_ROUND.to_vec(),
        MAMBA2_PARTIAL_FIRST_ROUND.to_vec(),
        GDN_FULL_BESIDE_PARTIAL.0.to_vec(),
        GDN_FULL_BESIDE_PARTIAL.1.to_vec(),
        MAMBA2_FULL_BESIDE_PARTIAL.0.to_vec(),
    ]
}

/// Decode tokens per second over `ITERS` runs: the median of the engine's own decode-phase rate.
const COST_ITERS: usize = 5;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// Run `driver` over `prompt` `COST_ITERS` times: the generated tokens, the median engine decode rate
/// and the counters the runs added.
fn timed_generate(
    driver: &mut Driver<Serving>,
    prompt: &[u32],
) -> (
    Vec<u32>,
    f64,
    SpeculationStats,
    poot_llm::driver::RollbackTiming,
) {
    use poot_llm::GenerationControl;
    use poot_llm::driver::GenerateRequest;
    let before = driver.stats();
    let mut rates = Vec::new();
    let mut tokens = Vec::new();
    for _ in 0..COST_ITERS {
        let generation = driver
            .generate(
                GenerateRequest {
                    prompt: prompt.to_vec(),
                    max_new: NEW_TOKENS,
                    sampler: Sampler::greedy(),
                    stops: Vec::new(),
                    ignore_eos: true,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let decode = generation.timing.expect("a timed driver").decode;
        rates.push(
            decode
                .tokens_per_second()
                .expect("the executor measured time"),
        );
        tokens = generation.tokens;
    }
    let after = driver.stats();
    let (a, b) = (after.speculation, before.speculation);
    let speculation = SpeculationStats {
        rounds: a.rounds - b.rounds,
        proposed: a.proposed - b.proposed,
        accepted: a.accepted - b.accepted,
        skipped_sampler: a.skipped_sampler - b.skipped_sampler,
        skipped_no_room: a.skipped_no_room - b.skipped_no_room,
        skipped_no_proposal: a.skipped_no_proposal - b.skipped_no_proposal,
        skipped_inputs: a.skipped_inputs - b.skipped_inputs,
        restored: a.restored - b.restored,
        replayed: a.replayed - b.replayed,
    };
    let step =
        |a: poot_llm::driver::StepTime, b: poot_llm::driver::StepTime| poot_llm::driver::StepTime {
            steps: a.steps - b.steps,
            wall: a.wall - b.wall,
        };
    let rollback = poot_llm::driver::RollbackTiming {
        snapshots: step(after.rollback.snapshots, before.rollback.snapshots),
        restores: step(after.rollback.restores, before.rollback.restores),
        replays: step(after.rollback.replays, before.rollback.replays),
    };
    (tokens, median(rates), speculation, rollback)
}

/// The cost rows' output, one line per prompt and family, for the design note: acceptance, the share
/// of rounds that restored (`p`), engine decode tokens per second with speculation on and off, and
/// what a round's snapshot, restore and replay cost as microseconds and as a fraction of one decode
/// step of the Off run. Recorded, never gated; the tokens must still equal Off's.
fn measure_cost(family: FamilyKey, label: &str, choice: BackendChoice, backend: DeviceBackend) {
    let (handle, _dir) = fixture_handle(family);
    let Some(off) = device_driver(&handle, choice, backend, DraftPolicy::Off) else {
        return;
    };
    let mut off = off.with_phase_timing().serve(Layout::Contiguous);
    let mut on = device_driver(&handle, choice, backend, lookup(1))
        .expect("the device opened a moment ago")
        .with_phase_timing()
        .serve(Layout::Contiguous);
    for prompt in cost_prompts() {
        let (reference, off_rate, _, _) = timed_generate(&mut off, &prompt);
        let (tokens, on_rate, spec, rollback) = timed_generate(&mut on, &prompt);
        assert_eq!(tokens, reference, "speculation changed the tokens");
        let runs = COST_ITERS as f64;
        let per_round = |step: poot_llm::driver::StepTime| {
            if step.steps == 0 {
                0.0
            } else {
                step.wall.as_secs_f64() * 1e6 / step.steps as f64
            }
        };
        let decode_step_us = 1e6 / off_rate;
        eprintln!(
            "COST {label} {backend:?} prompt_len={} accepted/proposed={}/{} rounds/run={:.1} \
             restored/run={:.1} replayed_tok/run={:.1} no_proposal/run={:.1} \
             off_tok_s={off_rate:.0} on_tok_s={on_rate:.0} decode_step_us={decode_step_us:.0} \
             snapshot_us={:.0} restore_us={:.0} replay_chunk_us={:.0} replay_over_decode={:.2}",
            prompt.len(),
            spec.accepted,
            spec.proposed,
            spec.rounds as f64 / runs,
            spec.restored as f64 / runs,
            spec.replayed as f64 / runs,
            spec.skipped_no_proposal as f64 / runs,
            per_round(rollback.snapshots),
            per_round(rollback.restores),
            per_round(rollback.replays),
            per_round(rollback.replays) / decode_step_us,
        );
    }
}

#[test]
fn recurrent_rollback_cost_on_gdn_wgpu() {
    measure_cost(GDN, "gdn", BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn recurrent_rollback_cost_on_gdn_rocm() {
    measure_cost(GDN, "gdn", BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn recurrent_rollback_cost_on_gdn_ptx() {
    measure_cost(GDN, "gdn", BackendChoice::Ptx, DeviceBackend::Ptx);
}

#[test]
fn recurrent_rollback_cost_on_mamba2_wgpu() {
    measure_cost(MAMBA2, "mamba2", BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn recurrent_rollback_cost_on_mamba2_rocm() {
    measure_cost(MAMBA2, "mamba2", BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn recurrent_rollback_cost_on_mamba2_ptx() {
    measure_cost(MAMBA2, "mamba2", BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// The shadow state set's share of the state a paged driver holds at `n_slots` sequence rows, from
/// the avals the executor allocates (the snapshot entry's shadow is one copy of every recurrent
/// value; `recurrent_snapshot_cost.rs` checks the allocation against these bytes on a counting
/// device). Recorded, never gated: the figures go to the design note.
#[test]
fn shadow_state_share_of_the_state_budget() {
    for (label, family) in FAMILIES {
        let fixture = (entry(family).fixture)();
        let model = (entry(family).build)(&fixture.raw(), &fixture.store).unwrap();
        let bytes =
            |t: &TensorType| (t.shape.iter().product::<usize>() * t.dtype.byte_size()) as u64;
        for n_slots in [1usize, 8, 32] {
            let pool_slots = 4 * BLOCK_SIZE * n_slots;
            let graph = model
                .model
                .trace(
                    &StepSpec::step(
                        StepShape::new(nz(1), NonZeroUsize::MIN, nz(CAPACITY)).with_kv(
                            KvLayout::Paged {
                                pool_slots: nz(pool_slots),
                                seqs: nz(n_slots),
                            },
                        ),
                    ),
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .unwrap();
            let (mut recurrent, mut positional) = (0u64, 0u64);
            for pair in graph.state_pairs() {
                let b = bytes(&graph.meta(pair.input).aval);
                if pair.role == StateRole::Recurrent {
                    recurrent += b;
                } else {
                    positional += b;
                }
            }
            eprintln!(
                "SHADOW {label} n_slots={n_slots} recurrent={recurrent} B positional(4 blocks/seq)={positional} B \
                 shadow=+{recurrent} B share_of_state_with_shadow={:.1}%",
                100.0 * recurrent as f64 / (2 * recurrent + positional) as f64
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The hidden-state carry is not rolled back (POOT-1109).
// ---------------------------------------------------------------------------------------------

/// Prompts of the search below: 18 tokens over a small alphabet, so prompt lookup finds drafts.
fn candidate_prompts() -> impl Iterator<Item = Vec<u32>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    std::iter::repeat_with(move || {
        (0..18)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                1 + ((state >> 33) % 7) as u32
            })
            .collect()
    })
}

/// The carry (`[1, K + 1, width]`, flat) a driver holds after `prompt` and then `extra` as one prefill
/// piece, read before its next step.
fn carry_after(
    handle: &Arc<ModelHandle>,
    choice: BackendChoice,
    backend: DeviceBackend,
    prompt: &[u32],
    extra: &[u32],
) -> Vec<f32> {
    let reference = probing_driver(
        poot_models::glm53_flash::FAMILY,
        handle,
        choice,
        backend,
        DraftPolicy::Off,
        KvLayout::Contiguous,
        Some(nz(K + 1)),
    )
    .expect("the device opened a moment ago");
    let mut reference = prepare_admitted(reference);
    let tokens = [prompt, extra].concat();
    let Admission::Opened { seq, .. } = reference
        .driver
        .open(SeqRequest::new(&tokens, 8, AdapterRef::Base))
        .unwrap()
    else {
        panic!("a fresh driver admits the sequence")
    };
    let mut sampler = Sampler::greedy();
    let mut fed = 0;
    for upto in [prompt.len(), tokens.len()] {
        while fed < upto {
            let mut rows = [StepRow {
                seq,
                work: RowWork::Prefill { upto },
                sampler: &mut sampler,
            }];
            let result = reference
                .driver
                .step(&mut rows)
                .unwrap_or_else(|e| panic!("{e}"))
                .remove(0);
            fed += result.consumed;
        }
    }
    reference.driver.commit(seq, 1, &mut sampler).unwrap();
    reference.arm();
    decode_step(&mut reference.driver, seq, &mut sampler);
    reference.read().remove(0)
}

fn prepare_admitted(probing: Probing) -> Probing<Serving> {
    probing.prepare(&contiguous())
}

/// POOT-1109 SC-002: a speculative round on GLM (KDA recurrent state, so it snapshots, runs a window
/// of `K + 1`, restores and replays the kept prefix) leaves the carry buffer holding what the window
/// wrote: the rejected positions' rows are not rolled back. The reference is a driver that runs the
/// query token and all `K` drafts as one prefill piece; the carry the round left equals its carry at
/// every row, the restored rows `a + 1 ..` included. The prompt is searched for a round that rejects
/// at least one draft and keeps at most one. Mutation: let `snapshot_entries` copy `Handoff` pairs
/// too (filter `!= Positional`); the restore puts the pre-window carry back and the rows past the
/// accepted ones differ from the window's.
fn assert_a_restore_leaves_the_hidden_carry_alone(choice: BackendChoice, backend: DeviceBackend) {
    let family = poot_models::glm53_flash::FAMILY;
    let (handle, _dir) = fixture_handle(family);
    let Some(speculating) = probing_driver(
        family,
        &handle,
        choice,
        backend,
        lookup(1),
        KvLayout::Contiguous,
        Some(nz(K + 1)),
    ) else {
        return;
    };
    let mut speculating = prepare_admitted(speculating);
    for prompt in candidate_prompts().take(48) {
        let (seq, query) = open_prefilled(&mut speculating.driver, &prompt, 8);
        let mut history = prompt.clone();
        history.push(query);
        let drafts = propose_lookup(&history, 1, K);
        let mut sampler = Sampler::greedy();
        let round = decode_step(&mut speculating.driver, seq, &mut sampler);
        let RowSpeculation::Verified { proposed, accepted } = round.speculation else {
            speculating.driver.release(seq, Release::Finished).unwrap();
            continue;
        };
        if proposed != K || accepted > 1 || accepted == proposed {
            speculating.driver.release(seq, Release::Finished).unwrap();
            continue;
        }
        speculating
            .driver
            .commit(seq, round.tokens.len(), &mut sampler)
            .unwrap();
        speculating.arm();
        decode_step(&mut speculating.driver, seq, &mut sampler);
        let got = speculating.read().remove(0);

        let window: Vec<u32> = std::iter::once(query).chain(drafts).collect();
        assert_eq!(window.len(), K + 1);
        let want = carry_after(&handle, choice, backend, &prompt, &window);
        assert_eq!(got.len(), want.len(), "the carry is [1, K + 1, width]");
        let worst = poot_test_util::max_unit_floor_error(&got, &want);
        assert!(
            worst < 1e-4,
            "after a round that accepted {accepted} of {proposed} the carry differs from the \
             window's at some element by {worst}"
        );
        assert!(
            got.iter().any(|v| v.abs() > 1e-3),
            "the carry is written, not the zeros it started as"
        );
        return;
    }
    panic!("no candidate prompt gave a round that rejected a draft and kept at most one");
}

#[test]
fn a_restore_leaves_the_hidden_carry_alone_on_glm_wgpu() {
    assert_a_restore_leaves_the_hidden_carry_alone(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn a_restore_leaves_the_hidden_carry_alone_on_glm_rocm() {
    assert_a_restore_leaves_the_hidden_carry_alone(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn a_restore_leaves_the_hidden_carry_alone_on_glm_ptx() {
    assert_a_restore_leaves_the_hidden_carry_alone(BackendChoice::Ptx, DeviceBackend::Ptx);
}
