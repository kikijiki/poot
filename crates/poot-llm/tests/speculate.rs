//! Speculative decoding as a driver policy (Card 751, dserve.md ruling V4/S7), on the add-a-model fixture
//! family: a family defined in this test target, outside every production crate, speculates through
//! `Driver::step` exactly as a built-in one does, because no list of families decides it.
//!
//! - Host half (model-free): a recurrent family is refused at `prepare` and the dense one prepares; a
//!   draft the target cannot verify is refused before anything compiles or uploads; the policy's window
//!   entry is published.
//! - Device half: greedy speculation (prompt lookup, and a perturbed copy of the fixture as the draft
//!   model) returns the tokens `DraftPolicy::Off` returns; sampled speculation returns the tokens a host
//!   replay of the same rounds returns.

#[path = "add_a_model/family.rs"]
mod family;

use std::cell::Cell;
use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    BufferRole, Device, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, StateScope, StepInputs, StepOutputs, WeightSource,
};
use poot_graph_ir::{
    BinOp, Builder, Graph, Slot, SlotKey, StateRole, TensorType, ValidationOutputs,
};
use poot_graph_plan::{
    CompileLimits, CompileOptions, StagedProgram, Submission, Target, TargetSet,
};
use poot_llm::driver::AdapterRef;
use poot_llm::driver::error::{DraftMismatch, DriverError, SpeculationRefusal, Unsupported};
use poot_llm::driver::speculate::{DRAFT_STREAM_SALT, propose_lookup, spec_sample_round};
use poot_llm::driver::step::PickPolicy;
use poot_llm::driver::{
    Admission, BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Head, Layout,
    LoadOptions, ModelHandle, PreparedSetLimits, Release, RowResult, RowWork, SeqRequest,
    ServingShapes, StepRow, Warm, open_executor, program_retention,
};
use poot_llm::driver::{NotSpeculated, PoolShape, RowSpeculation, SpeculationStats};
use poot_llm::{GenerationControl, Sampler};
use poot_models::chat::ChatFormat;
use poot_models::components::standard::{Step, check_step};
use poot_models::model::{
    FamilyKey, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use poot_test_util::weight_map_oracle::eval_mapped;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

use family::{ENTRY, VOCAB};

/// The KV positions of every driver in this file (the fixture family's `max_position_embeddings`).
const CAPACITY: usize = 64;
/// Tokens every device row generates.
const NEW_TOKENS: usize = 32;
/// Drafts a round proposes.
const K: usize = 4;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Checkpoints: the fixture, a perturbed copy of it (the draft), and variants a draft may not be.
// ---------------------------------------------------------------------------------------------

/// A word-level tokenizer over `words` (`a`..`z`, `0`..`9`, then `p0`..): the fixture's, or one
/// that names a different vocabulary.
fn tokenizer(vocab: usize, swap_first_two: bool) -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..vocab - words.len()).map(|i| format!("p{i}")));
    if swap_first_two {
        words.swap(0, 1);
    }
    let vocab = words
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

fn write_checkpoint(dir: &Path, fixture: &Fixture, tokenizer: &Tokenizer) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    poot_test_util::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

/// The built-in families plus the fixture family and the recurrent fake, registered from outside the
/// production crates.
fn registry() -> Registry {
    let mut registry = Registry::builtin().unwrap();
    registry.register(ENTRY).unwrap();
    registry.register(RECURRENT_ENTRY).unwrap();
    registry
}

fn load(dir: &Path) -> Arc<ModelHandle> {
    Arc::new(ModelHandle::load(dir, &registry(), &LoadOptions::default()).unwrap())
}

/// The fixture checkpoint, loaded.
fn target_handle() -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path("speculate_target");
    write_checkpoint(&dir, &family::fixture(), &tokenizer(VOCAB, false));
    (load(&dir), dir)
}

/// Every BF16 weight of `fixture` scaled by `1 + scale * u`, `u` a deterministic value in
/// `[-0.5, 0.5)`: a model that mostly agrees with the fixture and sometimes does not.
fn perturbed(fixture: &Fixture, scale: f32) -> Fixture {
    let mut store = WeightStore::builder();
    let mut seed = 0x51ed_270bu32;
    for (key, entry) in fixture.store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture is dense")
        };
        let mut bytes = Vec::with_capacity(dense.bytes().len());
        for pair in dense.bytes().as_slice().as_chunks::<2>().0 {
            let value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let u = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let value = value * (1.0 + scale * u);
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let weight = DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
            .expect("the bytes keep their shape");
        store
            .insert(key.as_str(), WeightEntry::Dense(weight))
            .expect("the names are unique");
    }
    Fixture {
        config: fixture.config.clone(),
        generation: None,
        store: store.build(),
    }
}

/// The draft of the device rows: the fixture with its weights perturbed by `scale`.
fn draft_handle(scale: f32) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path("speculate_draft");
    write_checkpoint(
        &dir,
        &perturbed(&family::fixture(), scale),
        &tokenizer(VOCAB, false),
    );
    (load(&dir), dir)
}

// ---------------------------------------------------------------------------------------------
// A recurrent fake family: one `StateRole::Recurrent` state folded with every token.
// ---------------------------------------------------------------------------------------------

const RECURRENT: FamilyKey = FamilyKey::new("poot_fixture_recurrentish");

const RECURRENT_ENTRY: FamilyEntry = FamilyEntry {
    family: RECURRENT,
    keys: &[(ConfigSource::HfModelType, "poot_fixture_recurrentish")],
    build: build_recurrent,
    fixture: recurrent_fixture,
    variants: &[],
};

#[derive(Debug)]
struct Recurrentish {
    config: ModelConfig,
    weights: WeightMap,
}

fn build_recurrent(_: &RawConfig<'_>, _: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    Ok(Box::new(Recurrentish {
        config: ModelConfig {
            family: RECURRENT,
            vocab: VOCAB,
            max_positions: CAPACITY,
            eos: BTreeSet::from([family::EOS]),
            bos: None,
            prompt_bos: None,
            output: ModelOutput::Logits { vocab: VOCAB },
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
        },
        weights: WeightMap::default(),
    }))
}

fn recurrent_fixture() -> Fixture {
    let mut store = WeightStore::builder();
    let dummy = DenseWeight::try_new(DType::BF16, vec![1], vec![0u8, 0].into()).unwrap();
    store.insert("dummy", WeightEntry::Dense(dummy)).unwrap();
    Fixture {
        config: serde_json::json!({
            "model_type": "poot_fixture_recurrentish",
            "vocab_size": VOCAB,
            "max_position_embeddings": CAPACITY,
            "eos_token_id": family::EOS,
        }),
        generation: None,
        store: store.build(),
    }
}

impl Model for Recurrentish {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(
        &self,
        phase: Phase,
        shape: StepShape,
    ) -> Result<Graph<ValidationOutputs>, TraceError> {
        check_step(RECURRENT, phase, shape, self.config.max_positions)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let first = if shape.logits == LogitRows::All {
            0
        } else {
            n - 1
        };
        let out = n - first;
        let token = b.cast(b.slice(step.tokens(), 1, first, n), DType::F32);
        let pos = b.cast(b.slice(step.pos(), 1, first, n), DType::F32);
        let seed = b.binary(BinOp::Add, token, pos);
        let dims = vec![rows, 4];
        let state = b.state_input(
            "toy.state",
            TensorType::f32(dims.clone()),
            StateRole::Recurrent,
        );
        // The state folds the step's last token, whatever width the step has.
        let last = b.slice(seed, 1, out - 1, out);
        let folded = b.binary(BinOp::Add, state, b.broadcast(last, dims));
        step.carry(state, folded);
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1]), vec![rows, out, VOCAB]);
        Ok(step.finish(b, logits))
    }
}

fn recurrent_handle() -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let dir = poot_test_util::unique_temp_path("speculate_recurrent");
    write_checkpoint(&dir, &recurrent_fixture(), &tokenizer(VOCAB, false));
    (load(&dir), dir)
}

// ---------------------------------------------------------------------------------------------
// A driver over a fake device: the real driver, engine and planner; only the device is fake.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

struct FakeDevice;

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

    fn target(&self) -> Target {
        Target {
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
    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn begin(&mut self, _submission: Submission) -> Result<(), FakeError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Option<()>, FakeError> {
        Ok(Some(()))
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn write(&mut self, _dst: &(), _bytes: &[u8]) -> Result<(), FakeError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        out.fill(0);
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

/// The fake's engine, counting the executables a driver loads: a refused policy must load none.
struct Counting {
    inner: Engine<FakeDevice>,
    loads: Rc<Cell<usize>>,
}

impl Executor for Counting {
    fn target_set(&self) -> TargetSet {
        self.inner.target_set()
    }
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: WeightSource,
    ) -> Result<ExecutableId, ExecError> {
        self.loads.set(self.loads.get() + 1);
        self.inner.load_weights(store, weights)
    }
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<ValidationOutputs>,
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

fn options(draft: DraftPolicy) -> DriverOptions {
    let compile = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    DriverOptions {
        prefill: compile,
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        max_trace_tokens: nz(64),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft,
    }
}

fn lookup() -> DraftPolicy {
    DraftPolicy::PromptLookup {
        ngram: nz(2),
        k: nz(K),
    }
}

fn draft_model(handle: Arc<ModelHandle>) -> DraftPolicy {
    DraftPolicy::DraftModel { handle, k: nz(K) }
}

/// A driver over `target` on the fake device, and the executables it has loaded.
fn fake_driver(target: Arc<ModelHandle>, draft: DraftPolicy) -> (Driver, Rc<Cell<usize>>) {
    let loads = Rc::new(Cell::new(0));
    let executor = Counting {
        inner: Engine::new(FakeDevice),
        loads: loads.clone(),
    };
    let driver = Driver::new(target, Box::new(executor), options(draft)).unwrap();
    (driver, loads)
}

fn prepare(driver: &mut Driver, heads: &[Head]) -> Result<(), DriverError> {
    driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads,
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .map(|_| ())
}

fn refusal(error: DriverError) -> SpeculationRefusal {
    match error {
        DriverError::Unsupported(Unsupported::Speculation(refusal)) => refusal,
        other => panic!("not a speculation refusal: {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Host half.
// ---------------------------------------------------------------------------------------------

/// SC-004: a recurrent family (one `StateRole::Recurrent` state, registered from outside the
/// production crates) with prompt lookup is refused at `prepare` with the typed recurrent refusal,
/// read from the compiled program's state roles; the dense fixture with the same policy prepares and
/// publishes its window. Mutation: skip the role check in `check_speculation_state`; the recurrent
/// `prepare` succeeds and the first assertion is red.
#[test]
fn a_recurrent_family_refuses_speculation_and_the_dense_fixture_prepares() {
    let (recurrent, _dir) = recurrent_handle();
    let (mut driver, _) = fake_driver(recurrent, lookup());
    let error =
        prepare(&mut driver, &[Head::GREEDY]).expect_err("recurrent state cannot roll back");
    assert_eq!(refusal(error), SpeculationRefusal::RecurrentState);
    assert!(
        driver.caps().is_none(),
        "a refused prepare publishes nothing"
    );

    let (dense, _dir) = target_handle();
    let (mut driver, _) = fake_driver(dense, lookup());
    prepare(&mut driver, &[Head::GREEDY]).unwrap_or_else(|e| panic!("{e}"));
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

/// The window entry is compiled once beside decode, whatever the policy proposes: `Off` compiles
/// none, and a sampled head adds the logits-head window.
#[test]
fn the_policy_compiles_one_window_entry_per_head_and_off_compiles_none() {
    let (target, _dir) = target_handle();
    let (mut off, _) = fake_driver(target.clone(), DraftPolicy::Off);
    prepare(&mut off, &[Head::GREEDY]).unwrap();
    assert!(off.caps().unwrap().windows.is_empty());
    assert!(off.caps().unwrap().speculation.is_none());
    let without = off.prepared_entries();

    let (mut greedy, _) = fake_driver(target.clone(), lookup());
    prepare(&mut greedy, &[Head::GREEDY]).unwrap();
    assert_eq!(greedy.prepared_entries(), without + 1);

    let (mut sampled, _) = fake_driver(target, lookup());
    prepare(
        &mut sampled,
        &[
            Head::GREEDY,
            Head::Sample(poot_graph_ir::op::SampleRule::Gumbel),
        ],
    )
    .unwrap();
    let sampled_off = {
        let (target, _dir) = target_handle();
        let (mut off, _) = fake_driver(target, DraftPolicy::Off);
        prepare(
            &mut off,
            &[
                Head::GREEDY,
                Head::Sample(poot_graph_ir::op::SampleRule::Gumbel),
            ],
        )
        .unwrap();
        off.prepared_entries()
    };
    assert_eq!(
        sampled.prepared_entries(),
        sampled_off + 2,
        "a greedy window and a logits window"
    );
}

/// A draft the target cannot verify fails `prepare` typed before anything compiles or loads: the
/// executor has loaded only the target's weights and no entry exists.
fn assert_refused_before_any_work(draft: Arc<ModelHandle>) -> DraftMismatch {
    let (target, _dir) = target_handle();
    let (mut driver, loads) = fake_driver(target, draft_model(draft));
    assert_eq!(loads.get(), 1, "the target's weights");
    let error = prepare(&mut driver, &[Head::GREEDY]).expect_err("the draft does not fit");
    let SpeculationRefusal::DraftMismatch(mismatch) = refusal(error) else {
        panic!("not a draft mismatch")
    };
    assert_eq!(loads.get(), 1, "the draft's weights were never uploaded");
    assert_eq!(driver.prepared_entries(), 0, "nothing compiled");
    assert_eq!(driver.stats().compiles, 0);
    mismatch
}

/// A draft checkpoint of the fixture family with the config and tokenizer edited.
fn edited_draft(
    edit: impl FnOnce(&mut Fixture),
    tokenizer: &Tokenizer,
) -> (Arc<ModelHandle>, poot_test_util::UniqueTempPath) {
    let mut fixture = family::fixture();
    edit(&mut fixture);
    let dir = poot_test_util::unique_temp_path("speculate_edited_draft");
    write_checkpoint(&dir, &fixture, tokenizer);
    (load(&dir), dir)
}

/// SC-005: a draft whose vocabulary exceeds the target's is `DraftMismatch::Vocab` at `prepare`,
/// before any upload (the executor loaded only the target's weights, upload counter unchanged) and
/// before any compile. The draft's embedding and head are widened with zero rows and its tokenizer
/// names the target's tokens with the same ids first, so only the vocabulary check refuses it.
/// Mutation: drop the vocab check in `check_draft`; `prepare` goes on to load and compile the draft
/// and the row is red.
#[test]
fn a_draft_with_a_larger_vocabulary_is_refused_before_any_upload() {
    const WIDER: usize = VOCAB + 8;
    let widen = |fixture: &mut Fixture| {
        fixture.config["vocab_size"] = WIDER.into();
        let mut store = WeightStore::builder();
        for (key, entry) in fixture.store.iter() {
            let WeightEntry::Dense(dense) = entry else {
                panic!("the fixture is dense")
            };
            let mut shape = dense.shape().to_vec();
            let mut bytes = dense.bytes().as_slice().to_vec();
            if key.as_str() == "gpt_neox.embed_in.weight" || key.as_str() == "embed_out.weight" {
                shape[0] = WIDER;
                bytes.resize(WIDER * shape[1] * 2, 0);
            }
            let weight = DenseWeight::try_new(dense.dtype(), shape, bytes.into()).unwrap();
            store
                .insert(key.as_str(), WeightEntry::Dense(weight))
                .unwrap();
        }
        fixture.store = store.build();
    };
    let (draft, _dir) = edited_draft(widen, &tokenizer(WIDER, false));
    assert_eq!(
        assert_refused_before_any_work(draft),
        DraftMismatch::Vocab {
            draft: WIDER,
            target: VOCAB
        }
    );
}

/// A draft whose tokenizer gives a token another id than the target does proposes tokens the target
/// would read as different ones: refused before any upload. Mutation: drop the token check; the
/// draft loads and the row is red.
#[test]
fn a_draft_that_names_tokens_differently_is_refused_before_any_upload() {
    let (draft, _dir) = edited_draft(|_| {}, &tokenizer(VOCAB, true));
    assert_eq!(assert_refused_before_any_work(draft), DraftMismatch::Tokens);
}

/// A draft that holds fewer positions than the driver's capacity cannot mirror the target's
/// sequences. Mutation: drop the positions check; the draft loads and the row is red.
#[test]
fn a_draft_with_fewer_positions_than_the_capacity_is_refused_before_any_upload() {
    let (draft, _dir) = edited_draft(
        |fixture| fixture.config["max_position_embeddings"] = (CAPACITY / 2).into(),
        &tokenizer(VOCAB, false),
    );
    assert_eq!(
        assert_refused_before_any_work(draft),
        DraftMismatch::Positions {
            draft: CAPACITY / 2,
            capacity: CAPACITY
        }
    );
}

/// One decode step of `seq` under the driver's policy, committing nothing.
fn decode_step(
    driver: &mut Driver,
    seq: poot_llm::driver::SeqId,
    sampler: &mut Sampler,
) -> RowResult {
    let mut rows = [StepRow {
        seq,
        work: RowWork::Decode {
            ahead: &[],
            pick: PickPolicy::Greedy,
        },
        sampler,
    }];
    driver
        .step(&mut rows)
        .unwrap_or_else(|e| panic!("{e}"))
        .remove(0)
}

/// Open `prompt` on a prepared driver, prefill it and commit its first token.
fn open_prefilled(
    driver: &mut Driver,
    prompt: &[u32],
    max_new: usize,
    sampler: &mut Sampler,
) -> poot_llm::driver::SeqId {
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest {
            prompt,
            max_new,
            adapter: AdapterRef::Base,
        })
        .unwrap()
    else {
        panic!("a fresh driver admits one sequence")
    };
    loop {
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut *sampler,
        }];
        let result = driver
            .step(&mut rows)
            .unwrap_or_else(|e| panic!("{e}"))
            .remove(0);
        assert_eq!(
            result.speculation,
            RowSpeculation::NotApplicable,
            "a prefill piece"
        );
        if !result.tokens.is_empty() {
            break;
        }
    }
    driver.commit(seq, 1, sampler).unwrap();
    seq
}

// ---------------------------------------------------------------------------------------------
// The CPU oracle: the family's own traced graphs, evaluated state-free over a whole token prefix.
// ---------------------------------------------------------------------------------------------

/// The logits of every position of `tokens` under the fixture family with `store`'s weights, from one
/// prefill over the prefix at position 0.
fn oracle_rows(store: &WeightStore, tokens: &[u32]) -> Vec<Vec<f32>> {
    let n = tokens.len();
    let fixture = family::fixture();
    let model = registry().build(&fixture.raw(), store).unwrap();
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: nz(n),
        capacity: nz(CAPACITY),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
    };
    let graph = model.trace(Phase::Prefill, shape).unwrap();
    let state: Vec<HostTensor> = graph
        .state
        .iter()
        .map(|&(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
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
    let (logits, _) = eval_mapped(&graph, store, model.weights(), &slots, &state);
    logits
        .as_f32()
        .expect("f32 logits")
        .chunks(VOCAB)
        .map(<[f32]>::to_vec)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Device half.
// ---------------------------------------------------------------------------------------------

/// A prompt that repeats itself, so prompt lookup finds a continuation from the first round. Its 32
/// greedy tokens all lead their runner-up by more than 0.05.
const REPETITIVE: [u32; 9] = [5, 27, 3, 5, 27, 3, 5, 27, 3];
/// A prompt whose greedy continuation does not repeat itself, for the draft model (same margin).
const VARIED: [u32; 12] = [8, 22, 36, 4, 8, 22, 36, 4, 8, 22, 36, 4];
/// A short prompt for the sampled row.
const PROMPT: [u32; 6] = [3, 17, 8, 29, 5, 21];

fn nothing(_: u32, _: &str) -> GenerationControl {
    GenerationControl::Continue(())
}

/// `new` tokens after `prompt` by `Driver::generate`, ignoring the end-of-sequence token.
fn generate(driver: &mut Driver, prompt: &[u32], new: usize, sampler: Sampler) -> Vec<u32> {
    driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new: new,
                sampler,
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut nothing,
        )
        .unwrap_or_else(|e| panic!("{e}"))
        .tokens
}

/// `new` tokens after `prompt` through the serving surface (`open`, `step`, `commit`): what each
/// step returned is kept, so a caller can see how many tokens a round committed.
fn serve(
    driver: &mut Driver,
    prompt: &[u32],
    new: usize,
    sampler: &mut Sampler,
    heads: &[Head],
) -> (Vec<u32>, Vec<usize>) {
    driver
        .prepare(&ServingShapes {
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
        .open(SeqRequest {
            prompt,
            max_new: new,
            adapter: AdapterRef::Base,
        })
        .unwrap()
    else {
        panic!("a fresh driver admits one sequence")
    };
    let step = |driver: &mut Driver, work, sampler: &mut Sampler| -> RowResult {
        let mut rows = [StepRow { seq, work, sampler }];
        driver
            .step(&mut rows)
            .unwrap_or_else(|e| panic!("{e}"))
            .remove(0)
    };
    let mut out = Vec::new();
    let mut rounds = Vec::new();
    let mut result = step(driver, RowWork::Prefill { upto: prompt.len() }, sampler);
    while result.tokens.is_empty() {
        result = step(driver, RowWork::Prefill { upto: prompt.len() }, sampler);
    }
    loop {
        let kept = result.tokens.len().min(new - out.len());
        driver.commit(seq, kept, sampler).unwrap();
        out.extend_from_slice(&result.tokens[..kept]);
        rounds.push(kept);
        if out.len() == new {
            break;
        }
        result = step(
            driver,
            RowWork::Decode {
                ahead: &[],
                pick: PickPolicy::Greedy,
            },
            sampler,
        );
    }
    driver.release(seq, Release::Finished).unwrap();
    assert_eq!(
        driver.stats().compiles,
        compiled,
        "serving the prepared set compiles nothing, whatever the rounds propose"
    );
    (out, rounds)
}

/// A device driver for `policy`.
fn device_driver(
    handle: Arc<ModelHandle>,
    executor: Box<dyn Executor>,
    policy: DraftPolicy,
) -> Driver {
    Driver::new(handle, executor, options(policy)).unwrap()
}

/// The first index of the largest logit, and how far it leads the runner-up.
fn greedy(logits: &[f32]) -> (u32, f32) {
    let best = logits
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > logits[best] { i } else { best });
    let runner_up = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != best)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    (best as u32, logits[best] - runner_up)
}

/// The greedy tokens after `prompt` on the CPU oracle: what `DraftPolicy::Off` generates. Every
/// token leads its runner-up by a margin no device rounding closes, so the comparison is exact.
fn oracle_greedy(store: &WeightStore, prompt: &[u32], new: usize) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    for i in 0..new {
        let rows = oracle_rows(store, &tokens);
        let (token, lead) = greedy(rows.last().unwrap());
        assert!(lead > 0.05, "token {i} ({token}) leads by only {lead}");
        tokens.push(token);
    }
    tokens.split_off(prompt.len())
}

/// `DraftPolicy::Off` on a device generates the oracle's greedy tokens, for both prompts of the
/// speculation rows below: the reference they are compared with is what `Off` produces.
fn assert_off_equals_the_oracle(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _dir) = target_handle();
    let mut off = device_driver(target.clone(), executor, DraftPolicy::Off);
    for prompt in [&REPETITIVE[..], &VARIED[..]] {
        let generated = generate(&mut off, prompt, NEW_TOKENS, Sampler::greedy());
        assert_eq!(
            generated,
            oracle_greedy(target.store(), prompt, NEW_TOKENS),
            "Off differs from the oracle on {prompt:?}"
        );
    }
}

#[test]
fn off_generate_equals_the_oracle_wgpu() {
    assert_off_equals_the_oracle(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn off_generate_equals_the_oracle_rocm() {
    assert_off_equals_the_oracle(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn off_generate_equals_the_oracle_ptx() {
    assert_off_equals_the_oracle(BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// SC-001: with `policy`, the fixture family's greedy tokens equal `DraftPolicy::Off`'s (the oracle's,
/// per [`assert_off_equals_the_oracle`]), whether the request runs through `generate` or through
/// `open`/`step`/`commit`, and at least one round committed more than one token. Mutations: feed the
/// window one position short (leave the query token out of a speculative row's window) or accept
/// every draft; the tokens diverge from `Off`.
fn assert_greedy_speculation_equals_off(
    choice: BackendChoice,
    backend: DeviceBackend,
    policy: impl FnOnce() -> (DraftPolicy, Option<poot_test_util::UniqueTempPath>),
    prompt: &[u32],
) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _target_dir) = target_handle();
    let reference = oracle_greedy(target.store(), prompt, NEW_TOKENS);
    let (policy, _dir) = policy();
    let mut driver = device_driver(target, executor, policy);
    let spoken = generate(&mut driver, prompt, NEW_TOKENS, Sampler::greedy());
    assert_eq!(
        spoken, reference,
        "generate under the policy differs from Off"
    );
    let (served, rounds) = serve(
        &mut driver,
        prompt,
        NEW_TOKENS,
        &mut Sampler::greedy(),
        &[Head::GREEDY],
    );
    assert_eq!(served, reference, "step under the policy differs from Off");
    assert!(
        rounds.iter().any(|&kept| kept > 1),
        "no round committed more than one token: {rounds:?}"
    );
}

fn lookup_policy() -> (DraftPolicy, Option<poot_test_util::UniqueTempPath>) {
    (lookup(), None)
}

fn model_policy() -> (DraftPolicy, Option<poot_test_util::UniqueTempPath>) {
    let (draft, dir) = draft_handle(0.02);
    (draft_model(draft), Some(dir))
}

#[test]
fn greedy_prompt_lookup_equals_off_wgpu() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        lookup_policy,
        &REPETITIVE,
    );
}

#[test]
fn greedy_prompt_lookup_equals_off_rocm() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        lookup_policy,
        &REPETITIVE,
    );
}

#[test]
fn greedy_prompt_lookup_equals_off_ptx() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        lookup_policy,
        &REPETITIVE,
    );
}

#[test]
fn greedy_draft_model_equals_off_wgpu() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Wgpu,
        DeviceBackend::Wgpu,
        model_policy,
        &VARIED,
    );
}

#[test]
fn greedy_draft_model_equals_off_rocm() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Rocm,
        DeviceBackend::Rocm,
        model_policy,
        &VARIED,
    );
}

#[test]
fn greedy_draft_model_equals_off_ptx() {
    assert_greedy_speculation_equals_off(
        BackendChoice::Ptx,
        DeviceBackend::Ptx,
        model_policy,
        &VARIED,
    );
}

/// The seed of the sampled row, and the temperature (T = 1: the distribution is the softmax).
const SEED: u64 = 0x751;

/// Who proposes the drafts a sampled replay verifies.
#[derive(Clone, Copy)]
enum Proposer<'a> {
    /// A draft model: its oracle `q` rows, its proposals drawn from the stream forked off the row's
    /// sampler once.
    Model(&'a ModelHandle),
    /// Prompt lookup over `ngram` tokens: its proposals are the earlier continuation and its `q` is
    /// the one-hot row of each.
    Lookup { ngram: usize },
}

/// The tokens a host replay of the sampled rounds produces: the oracle's `p` rows for the target, the
/// `proposer`'s drafts and `q` rows, and `spec_sample_round` over them with the row's own sampler. A
/// round with no drafts is one ordinary pick.
fn sampled_reference(
    target: &ModelHandle,
    proposer: Proposer<'_>,
    prompt: &[u32],
    new: usize,
    temperature: f32,
    seed: u64,
) -> Vec<u32> {
    let target_store = target.store();
    let mut sampler = Sampler::new(temperature, 0, 1.0, seed);
    let mut tokens = prompt.to_vec();
    let first = {
        let rows = oracle_rows(target_store, &tokens);
        sampler.pick(rows.last().unwrap()).unwrap() as u32
    };
    tokens.push(first);
    let reserved = prompt.len() + new;
    let mut draws: Option<Sampler> = None;
    while tokens.len() < reserved {
        let committed = tokens.len();
        let room = (reserved - committed - 1).min(K);
        let mut drafts = Vec::new();
        let mut q = Vec::new();
        match proposer {
            Proposer::Model(draft) if room > 0 => {
                let draws = draws.get_or_insert_with(|| sampler.fork(DRAFT_STREAM_SALT));
                let mut prefix = tokens.clone();
                for _ in 0..room {
                    let rows = oracle_rows(draft.store(), &prefix);
                    let mut row = draws.probs(rows.last().unwrap()).unwrap();
                    row.resize(target.config().vocab, 0.0);
                    let token = draws.sample_from_probs(&row).unwrap();
                    q.push(row);
                    drafts.push(token);
                    prefix.push(token);
                }
            }
            Proposer::Model(_) => {}
            Proposer::Lookup { ngram } => {
                drafts = propose_lookup(&tokens, ngram, room);
                q = drafts
                    .iter()
                    .map(|&d| {
                        let mut row = vec![0.0; target.config().vocab];
                        row[d as usize] = 1.0;
                        row
                    })
                    .collect();
            }
        }
        let mut window = tokens.clone();
        window.extend_from_slice(&drafts);
        let rows = oracle_rows(target_store, &window);
        if drafts.is_empty() {
            let token = sampler.pick(&rows[committed - 1]).unwrap() as u32;
            tokens.push(token);
            continue;
        }
        let p: Vec<Vec<f32>> = rows[committed - 1..]
            .iter()
            .map(|row| sampler.probs(row).unwrap())
            .collect();
        let (accepted, next) = spec_sample_round(&drafts, &q, &p, &mut sampler).unwrap();
        tokens.extend_from_slice(&drafts[..accepted]);
        tokens.push(next);
    }
    tokens.split_off(prompt.len())
}

/// SC-003: the sampled device run (fixture, T = 1, a fixed seed, a perturbed draft model) returns the
/// tokens the host replay returns: the target's `p` and the draft's `q` come from the oracle, the
/// accept and residual draws from the same sampler stream. The draft is chosen far enough from the
/// target that rounds reject, so the residual path is part of the tokens. Mutation: take the argmax
/// for a sampled row's verify (greedy acceptance in `verify_row`); the tokens diverge.
fn assert_sampled_speculation_matches_the_host_replay(
    choice: BackendChoice,
    backend: DeviceBackend,
) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _target_dir) = target_handle();
    let (draft, _draft_dir) = draft_handle(0.3);
    let reference = sampled_reference(
        &target,
        Proposer::Model(&draft),
        &PROMPT,
        NEW_TOKENS,
        1.0,
        SEED,
    );

    let mut driver = device_driver(target, executor, draft_model(draft));
    let mut sampler = Sampler::new(1.0, 0, 1.0, SEED);
    let heads = [Head::Sample(poot_graph_ir::op::SampleRule::Gumbel)];
    let (served, rounds) = serve(&mut driver, &PROMPT, NEW_TOKENS, &mut sampler, &heads);
    assert_eq!(
        served, reference,
        "the device's sampled rounds {rounds:?} differ from the host replay"
    );
    assert!(
        rounds.iter().any(|&kept| kept > 1) && rounds.iter().any(|&kept| kept < K + 1),
        "the draft neither agreed nor disagreed: {rounds:?}"
    );
}

#[test]
fn sampled_draft_model_matches_the_host_replay_wgpu() {
    assert_sampled_speculation_matches_the_host_replay(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn sampled_draft_model_matches_the_host_replay_rocm() {
    assert_sampled_speculation_matches_the_host_replay(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn sampled_draft_model_matches_the_host_replay_ptx() {
    assert_sampled_speculation_matches_the_host_replay(BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// A decode row under a draft policy either verifies drafts or says why it decoded one token, and the
/// engine counts both: a row with a penalty (its draws are not the plain ones the window reproduces),
/// a sequence with no room for a window and a prompt lookup that finds nothing each report their own
/// reason; a row that verifies reports its drafts and what was accepted; and the driver's counters are
/// exactly the sum of what the rows reported. Mutation: make `speculation_sampler_ok` accept every
/// sampler; the penalty row verifies instead and the first assertion is red.
fn assert_rows_report_why_they_did_not_speculate(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _dir) = target_handle();
    let mut driver = device_driver(target, executor, lookup());
    driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY, Head::Logits],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let mut expected = SpeculationStats::default();
    let mut round = |driver: &mut Driver,
                     prompt: &[u32],
                     max_new: usize,
                     sampler: &mut Sampler,
                     steps: usize| {
        let seq = open_prefilled(driver, prompt, max_new, sampler);
        let mut generated = 1;
        let mut results = Vec::new();
        for _ in 0..steps {
            let result = decode_step(driver, seq, sampler);
            match result.speculation {
                RowSpeculation::NotApplicable => {}
                RowSpeculation::Verified { proposed, accepted } => {
                    expected.rounds += 1;
                    expected.proposed += proposed as u64;
                    expected.accepted += accepted as u64;
                }
                RowSpeculation::NotSpeculated(NotSpeculated::Sampler) => {
                    expected.skipped_sampler += 1;
                }
                RowSpeculation::NotSpeculated(NotSpeculated::NoRoom) => {
                    expected.skipped_no_room += 1;
                }
                RowSpeculation::NotSpeculated(NotSpeculated::NoProposal) => {
                    expected.skipped_no_proposal += 1;
                }
            }
            let kept = result.tokens.len().min(max_new - generated);
            driver.commit(seq, kept, sampler).unwrap();
            generated += kept;
            results.push(result);
        }
        driver.release(seq, Release::Finished).unwrap();
        results
    };

    let mut penalised = Sampler::greedy().with_penalties(1.1, 0.0, 0.0);
    let results = round(&mut driver, &REPETITIVE, 16, &mut penalised, 6);
    assert!(
        results.iter().all(|r| r.speculation
            == RowSpeculation::NotSpeculated(NotSpeculated::Sampler)
            && r.tokens.len() == 1),
        "a penalty row decodes one token a step and says why: {results:?}"
    );

    let results = round(&mut driver, &REPETITIVE, 16, &mut Sampler::greedy(), 8);
    let verified: Vec<_> = results
        .iter()
        .filter_map(|r| match r.speculation {
            RowSpeculation::Verified { proposed, accepted } => Some((r, proposed, accepted)),
            _ => None,
        })
        .collect();
    assert!(!verified.is_empty(), "a repetitive prompt verifies drafts");
    for (result, proposed, accepted) in verified {
        assert!(accepted <= proposed);
        assert_eq!(result.tokens.len(), accepted + 1);
    }

    let results = round(
        &mut driver,
        &[10, 11, 12, 13],
        16,
        &mut Sampler::greedy(),
        1,
    );
    assert_eq!(
        results[0].speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::NoProposal)
    );

    // One token reserved: once it is committed the sequence has no position left for a window.
    let results = round(&mut driver, &REPETITIVE, 1, &mut Sampler::greedy(), 1);
    assert_eq!(
        results[0].speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::NoRoom)
    );

    let stats = driver.stats().speculation;
    assert_eq!(stats, expected, "the counters are what the rows reported");
    assert_eq!(stats.skipped_sampler, 6);
    assert!(stats.skipped_no_proposal >= 1 && stats.skipped_no_room >= 1);
    assert!(stats.rounds > 0 && stats.accepted <= stats.proposed);
}

#[test]
fn a_row_that_does_not_speculate_says_why_wgpu() {
    assert_rows_report_why_they_did_not_speculate(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn a_row_that_does_not_speculate_says_why_rocm() {
    assert_rows_report_why_they_did_not_speculate(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn a_row_that_does_not_speculate_says_why_ptx() {
    assert_rows_report_why_they_did_not_speculate(BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// The temperature of the lookup row: low enough that the target's distribution is peaked and a
/// proposed token is sometimes the likely one.
const COLD: f32 = 0.25;

/// A sampled row under prompt lookup (the one-hot `q` path: no draft distribution, so acceptance is
/// `u < p(d)`) returns the tokens a host replay returns, and the engine's counters show that drafts
/// were verified and some of them rejected. Mutation: draw the proposal's `q` from `p` instead of the
/// one-hot row (`propose`, prompt lookup's sampled `q`); the tokens diverge from the replay.
fn assert_sampled_prompt_lookup_matches_the_host_replay(
    choice: BackendChoice,
    backend: DeviceBackend,
) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _target_dir) = target_handle();
    // Sampled tokens rarely repeat a bigram, so the lookup keys on one token: any token the prompt
    // already holds proposes what followed it.
    let prompt: Vec<u32> = PROMPT.iter().cycle().take(18).copied().collect();
    let reference = sampled_reference(
        &target,
        Proposer::Lookup { ngram: 1 },
        &prompt,
        NEW_TOKENS,
        COLD,
        SEED,
    );
    let policy = DraftPolicy::PromptLookup {
        ngram: nz(1),
        k: nz(K),
    };
    let mut driver = device_driver(target, executor, policy);
    let mut sampler = Sampler::new(COLD, 0, 1.0, SEED);
    let heads = [Head::Sample(poot_graph_ir::op::SampleRule::Gumbel)];
    let (served, rounds) = serve(&mut driver, &prompt, NEW_TOKENS, &mut sampler, &heads);
    assert_eq!(
        served, reference,
        "the device's sampled rounds {rounds:?} differ from the host replay"
    );
    let stats = driver.stats().speculation;
    assert!(
        stats.rounds > 0 && stats.accepted > 0 && stats.accepted < stats.proposed,
        "lookup drafts were neither accepted nor rejected: {stats:?}"
    );
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_wgpu() {
    assert_sampled_prompt_lookup_matches_the_host_replay(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_rocm() {
    assert_sampled_prompt_lookup_matches_the_host_replay(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn sampled_prompt_lookup_matches_the_host_replay_ptx() {
    assert_sampled_prompt_lookup_matches_the_host_replay(BackendChoice::Ptx, DeviceBackend::Ptx);
}

// ---------------------------------------------------------------------------------------------
// Several rows on the paged pool: one verify entry for the row count, mixed with prefill pieces.
// ---------------------------------------------------------------------------------------------

/// A row of the paged rows below: its prompt and the sampler its picks go through.
struct PagedRow {
    prompt: Vec<u32>,
    sampler: Sampler,
}

/// What a paged row produced: its tokens, and how many tokens each of its decode rounds committed.
struct PagedOutput {
    tokens: Vec<u32>,
    rounds: Vec<usize>,
}

/// Serve `rows` together on a paged pool of one two-row entry set: every step feeds each unfinished
/// row's next prefill piece or decode round, so a short prompt decodes (and speculates) while a long
/// one is still prefilling. Each row commits what its step returned, up to `new` tokens.
fn serve_paged(
    driver: &mut Driver,
    mut rows: Vec<PagedRow>,
    new: usize,
    heads: &[Head],
) -> Vec<PagedOutput> {
    driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(PoolShape {
                blocks: nz(16),
                max_seqs: nz(rows.len()),
            }),
            rows: &[nz(rows.len())],
            heads,
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let compiled = driver.stats().compiles;
    let seqs: Vec<_> = rows
        .iter()
        .map(|row| {
            match driver
                .open(SeqRequest {
                    prompt: &row.prompt,
                    max_new: new,
                    adapter: AdapterRef::Base,
                })
                .unwrap()
            {
                Admission::Opened { seq, .. } => seq,
                Admission::NoRoom { .. } => panic!("the pool holds every row"),
            }
        })
        .collect();
    let mut out: Vec<PagedOutput> = rows
        .iter()
        .map(|_| PagedOutput {
            tokens: Vec::new(),
            rounds: Vec::new(),
        })
        .collect();
    let mut prefilled = vec![false; rows.len()];
    while out.iter().any(|o| o.tokens.len() < new) {
        let live: Vec<usize> = (0..rows.len())
            .filter(|&i| out[i].tokens.len() < new)
            .collect();
        let mut step_rows: Vec<StepRow<'_>> = rows
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| live.contains(i))
            .map(|(i, row)| StepRow {
                seq: seqs[i],
                work: if prefilled[i] {
                    RowWork::Decode {
                        ahead: &[],
                        pick: PickPolicy::Greedy,
                    }
                } else {
                    RowWork::Prefill {
                        upto: row.prompt.len(),
                    }
                },
                sampler: &mut row.sampler,
            })
            .collect();
        let results = driver
            .step(&mut step_rows)
            .unwrap_or_else(|e| panic!("{e}"));
        drop(step_rows);
        for (&i, result) in live.iter().zip(results) {
            if result.tokens.is_empty() {
                continue;
            }
            let decoded = std::mem::replace(&mut prefilled[i], true);
            let kept = result.tokens.len().min(new - out[i].tokens.len());
            driver.commit(seqs[i], kept, &mut rows[i].sampler).unwrap();
            out[i].tokens.extend_from_slice(&result.tokens[..kept]);
            if decoded {
                out[i].rounds.push(kept);
            }
        }
    }
    for seq in seqs {
        driver.release(seq, Release::Finished).unwrap();
    }
    assert_eq!(
        driver.stats().compiles,
        compiled,
        "serving the prepared set compiles nothing, whatever the rows propose"
    );
    out
}

/// A second prompt that repeats itself, for the second row: its 32 greedy tokens lead by more than 0.05.
const REPETITIVE_2: [u32; 12] = [1, 6, 12, 3, 1, 6, 12, 3, 1, 6, 12, 3];

/// Two greedy rows on the paged pool speculate together: each row's 32 tokens equal what `Off`
/// generates for its prompt (the oracle's), and each row had a round that committed more than one
/// token. The rows' prompts differ in length, so one decodes (and speculates) in a step where the
/// other still prefills. Mutation: read every row's picks from the first row's offset in the greedy
/// suffix readback (`run_plans`: drop `i * positions`); the second row decodes the first row's tokens.
fn assert_two_paged_greedy_rows_equal_off(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _dir) = target_handle();
    let prompts = [REPETITIVE.to_vec(), REPETITIVE_2.to_vec()];
    let references: Vec<_> = prompts
        .iter()
        .map(|prompt| oracle_greedy(target.store(), prompt, NEW_TOKENS))
        .collect();
    let mut driver = device_driver(target, executor, lookup());
    let rows = prompts
        .iter()
        .map(|prompt| PagedRow {
            prompt: prompt.clone(),
            sampler: Sampler::greedy(),
        })
        .collect();
    let served = serve_paged(&mut driver, rows, NEW_TOKENS, &[Head::GREEDY]);
    for (i, (row, reference)) in served.iter().zip(&references).enumerate() {
        assert_eq!(&row.tokens, reference, "row {i} differs from Off");
        assert!(
            row.rounds.iter().any(|&kept| kept > 1),
            "row {i} never committed more than one token in a round: {:?}",
            row.rounds
        );
    }
}

#[test]
fn two_paged_greedy_rows_equal_off_wgpu() {
    assert_two_paged_greedy_rows_equal_off(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn two_paged_greedy_rows_equal_off_rocm() {
    assert_two_paged_greedy_rows_equal_off(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn two_paged_greedy_rows_equal_off_ptx() {
    assert_two_paged_greedy_rows_equal_off(BackendChoice::Ptx, DeviceBackend::Ptx);
}

/// Three rows of different prompt lengths on one draft model on the paged pool: a greedy row and two
/// sampled rows of their own seeds. Each sampled row returns the tokens a host replay of its own
/// rounds returns and the greedy row the oracle's greedy tokens, so the host path reads each row's
/// window logits at its own offset, and each row's draft cache and settle are its own. The long
/// prompt prefills over several steps while the others already decode. Mutations: read every
/// sampled row's window logits from the first row's offset (`verify_row`: drop `index * positions`);
/// settle every row at the longest sequence's length (`step`: `committed` from the longest open
/// sequence); the tokens diverge from the replay.
fn assert_paged_mixed_draft_rows_match(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let (target, _target_dir) = target_handle();
    let (draft, _draft_dir) = draft_handle(0.3);
    let greedy_prompt = VARIED.to_vec();
    let short_prompt = PROMPT.to_vec();
    let long_prompt: Vec<u32> = PROMPT.iter().cycle().take(20).copied().collect();
    let greedy_reference = oracle_greedy(target.store(), &greedy_prompt, NEW_TOKENS);
    let replay = |prompt: &[u32], seed| {
        sampled_reference(
            &target,
            Proposer::Model(&draft),
            prompt,
            NEW_TOKENS,
            1.0,
            seed,
        )
    };
    let references = [replay(&short_prompt, SEED), replay(&long_prompt, SEED + 1)];

    let mut driver = device_driver(target.clone(), executor, draft_model(draft.clone()));
    let rows = vec![
        PagedRow {
            prompt: greedy_prompt,
            sampler: Sampler::greedy(),
        },
        PagedRow {
            prompt: short_prompt,
            sampler: Sampler::new(1.0, 0, 1.0, SEED),
        },
        PagedRow {
            prompt: long_prompt,
            sampler: Sampler::new(1.0, 0, 1.0, SEED + 1),
        },
    ];
    // Rows of different rules in one step pick on the host, from the logits entry.
    let heads = [
        Head::GREEDY,
        Head::Sample(poot_graph_ir::op::SampleRule::Gumbel),
        Head::Logits,
    ];
    let served = serve_paged(&mut driver, rows, NEW_TOKENS, &heads);
    assert_eq!(
        served[0].tokens, greedy_reference,
        "the greedy row differs from Off"
    );
    for (i, reference) in references.iter().enumerate() {
        let row = &served[i + 1];
        assert_eq!(
            &row.tokens,
            reference,
            "sampled row {} (rounds {:?}) differs from the host replay",
            i + 1,
            row.rounds
        );
        assert!(
            row.rounds.iter().any(|&kept| kept > 1) && row.rounds.iter().any(|&kept| kept < K + 1),
            "the draft neither agreed nor disagreed on row {}: {:?}",
            i + 1,
            row.rounds
        );
    }
}

#[test]
fn paged_mixed_draft_rows_match_wgpu() {
    assert_paged_mixed_draft_rows_match(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

#[test]
fn paged_mixed_draft_rows_match_rocm() {
    assert_paged_mixed_draft_rows_match(BackendChoice::Rocm, DeviceBackend::Rocm);
}

#[test]
fn paged_mixed_draft_rows_match_ptx() {
    assert_paged_mixed_draft_rows_match(BackendChoice::Ptx, DeviceBackend::Ptx);
}
