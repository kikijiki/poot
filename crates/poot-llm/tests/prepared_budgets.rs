//! Card 666: the driver's prepared set has explicit size and lifetime budgets. Every row runs the real
//! `Driver` over the real `Engine` and planner with only the device faked, through a family registered
//! from outside the crate (nothing here is a built-in family), so a bound that held only for built-in
//! models would fail. Rows read counters and refusals, never a token's value.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use poot_executor::{
    BufferRole, Device, DeviceError, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, LoadError, StateScope, StepInputs, StepOutputs, WeightSource,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, ValidationOutputs};
use poot_graph_plan::{
    CompileLimits, CompileOptions, FusionPolicy, StagedProgram, Submission, Target, TargetSet,
};
use poot_llm::driver::error::{DriverError, PreparedSetRefusal};
use poot_llm::driver::{
    Driver, DriverOptions, GenerateRequest, Head, Layout, ModelHandle, PoolShape,
    PreparedSetLimits, Retention, ServingShapes, SharedOwner, Warm,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::chat::ChatFormat;
use poot_models::components::standard::Step;
use poot_models::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput,
    Phase, ShapeReason, StepShape, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const VOCAB: usize = 12;
const EOS: u32 = 11;
const TINY: FamilyKey = FamilyKey::new("budget-tiny");

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// The family: logits that depend on the step's tokens and positions, plus an optional recurrent
// state folded with every token. Registered through the public `Registry`, like any third-party family.
// ---------------------------------------------------------------------------------------------

type TraceLog = Arc<Mutex<Vec<(Phase, StepShape)>>>;

#[derive(Debug)]
struct Tiny {
    config: ModelConfig,
    weights: WeightMap,
    log: TraceLog,
    recurrent: bool,
}

impl Model for Tiny {
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
        let refuse = |reason| {
            Err(TraceError::ShapeUnsupported {
                family: TINY,
                shape,
                reason,
            })
        };
        let granule = self.config.prefill_granule.get();
        if phase == Phase::Decode && shape.tokens.get() != 1 {
            return refuse(ShapeReason::DecodeTokens);
        }
        if phase == Phase::Prefill && !shape.tokens.get().is_multiple_of(granule) {
            return refuse(ShapeReason::Granule { granule });
        }
        self.log.lock().unwrap().push((phase, shape));
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
        if self.recurrent {
            let dims = vec![rows, 4];
            let state = b.state_input(
                "toy.state",
                TensorType::f32(dims.clone()),
                StateRole::Recurrent,
            );
            let folded = b.binary(BinOp::Add, state, b.broadcast(seed, dims));
            step.carry(state, folded);
        }
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1]), vec![rows, out, VOCAB]);
        Ok(step.finish(b, logits))
    }
}

thread_local! {
    static TRACE_LOG: RefCell<Option<TraceLog>> = const { RefCell::new(None) };
}

fn build_tiny(raw: &RawConfig<'_>, log: TraceLog) -> Result<Box<dyn Model>, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        unreachable!("the family names only HF configs")
    };
    let field = |name: &'static str| {
        config
            .get(name)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or(ModelError::Config {
                family: TINY,
                field: name,
                reason: ConfigReason::Missing,
            })
    };
    let granule = NonZeroUsize::new(field("granule")?).ok_or(ModelError::Config {
        family: TINY,
        field: "granule",
        reason: ConfigReason::Zero,
    })?;
    Ok(Box::new(Tiny {
        recurrent: config.get("state").and_then(|v| v.as_str()) == Some("recurrent"),
        config: ModelConfig {
            family: TINY,
            vocab: VOCAB,
            max_positions: field("max_positions")?,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            output: ModelOutput::Logits { vocab: VOCAB },
            prefill_granule: granule,
            chat: ChatFormat::ChatML,
        },
        weights: WeightMap::default(),
        log,
    }))
}

/// The family's tokenizer, a word-level vocabulary over `a`..`k`.
fn tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0u32);
    for (i, w) in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        .iter()
        .enumerate()
    {
        vocab.insert((*w).to_string(), (i + 1) as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

/// A checkpoint directory the public loader reads: the family's `config.json`, a tokenizer and a
/// safetensors file with no tensors (the family has no weights).
fn checkpoint_dir(config: &serde_json::Value, name: &str) -> poot_test_util::UniqueTempPath {
    let dir = poot_test_util::unique_temp_path(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    let header = b"{}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    dir
}

// ---------------------------------------------------------------------------------------------
// The fake device, and the executor wrapper that records what the driver asked of it.
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
    /// Every row picks token 1: the sampling suffix answers `[token, non_finite_index]`, the logits head
    /// a row favoring token 1.
    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        let bytes: Vec<u8> = if out.len().is_multiple_of(VOCAB * 4) {
            (0..out.len() / (VOCAB * 4))
                .flat_map(|_| (0..VOCAB).map(|t| if t == 1 { 5.0f32 } else { 0.0 }))
                .flat_map(f32::to_le_bytes)
                .collect()
        } else {
            (0..out.len() / 8)
                .flat_map(|_| {
                    let mut row = 1i32.to_le_bytes().to_vec();
                    row.extend((-1i32).to_le_bytes());
                    row
                })
                .collect()
        };
        out.copy_from_slice(&bytes);
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

/// What the driver asked of the executor.
#[derive(Default)]
struct Asked {
    /// `(phase, shape)` of each entry, in `add_entry` order, from the family's trace log.
    entries: Vec<(EntryId, (Phase, StepShape))>,
    steps: Vec<EntryId>,
    resets: usize,
    /// Fail the next `add_entry`, after the driver has measured and reserved the entry.
    fail_add: bool,
}

struct Wrapped {
    inner: Engine<FakeDevice>,
    asked: Rc<RefCell<Asked>>,
    log: TraceLog,
}

fn device_failure() -> ExecError {
    ExecError::Device(Box::new(DeviceError {
        backend: "fake",
        source: Box::new(FakeError),
    }))
}

impl Executor for Wrapped {
    fn target_set(&self) -> TargetSet {
        self.inner.target_set()
    }
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: WeightSource,
    ) -> Result<ExecutableId, ExecError> {
        self.inner.load_weights(store, weights)
    }
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<ValidationOutputs>,
    ) -> Result<EntryId, ExecError> {
        if std::mem::take(&mut self.asked.borrow_mut().fail_add) {
            return Err(device_failure());
        }
        let id = self.inner.add_entry(exe, program)?;
        let traced = *self.log.lock().unwrap().last().expect("the model traced");
        self.asked.borrow_mut().entries.push((id, traced));
        Ok(id)
    }
    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        self.asked.borrow_mut().steps.push(entry);
        self.inner.step(exe, entry, inputs, sync)
    }
    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError> {
        self.asked.borrow_mut().resets += 1;
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

// ---------------------------------------------------------------------------------------------
// The rig.
// ---------------------------------------------------------------------------------------------

struct Rig {
    driver: Driver,
    asked: Rc<RefCell<Asked>>,
    log: TraceLog,
    /// The checkpoint directory, removed when the rig drops.
    _checkpoint: poot_test_util::UniqueTempPath,
}

struct RigOptions {
    recurrent: bool,
    granule: usize,
    chunk: usize,
    max_trace_tokens: usize,
    max_entries: usize,
    max_retained_bytes: u64,
    charge: poot_llm::driver::RetentionCharge,
    limits: CompileLimits,
}

impl Default for RigOptions {
    fn default() -> Self {
        Self {
            recurrent: false,
            granule: 1,
            chunk: 4,
            max_trace_tokens: 64,
            max_entries: 64,
            max_retained_bytes: 1 << 40,
            charge: poot_llm::driver::program_retention,
            limits: CompileLimits::STANDARD,
        }
    }
}

fn rig(options: RigOptions) -> Rig {
    let log: TraceLog = Arc::default();
    TRACE_LOG.with(|slot| *slot.borrow_mut() = Some(log.clone()));
    let config = serde_json::json!({
        "model_type": "budget-tiny",
        "granule": options.granule,
        "max_positions": 256,
        "state": if options.recurrent { "recurrent" } else { "plain" },
    });
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: TINY,
            keys: &[(ConfigSource::HfModelType, "budget-tiny")],
            build: |raw, _store| {
                let log = TRACE_LOG.with(|slot| slot.borrow().clone().expect("the rig sets it"));
                build_tiny(raw, log)
            },
            fixture: || Fixture {
                config: serde_json::json!({"model_type": "budget-tiny", "granule": 1, "max_positions": 16}),
                generation: None,
                store: WeightStore::default(),
            },
        })
        .unwrap();
    let dir = checkpoint_dir(&config, "prepared_budgets_checkpoint");
    let handle = ModelHandle::load(&dir, &registry).unwrap();
    let asked: Rc<RefCell<Asked>> = Rc::default();
    let executor = Wrapped {
        inner: Engine::new(FakeDevice),
        asked: asked.clone(),
        log: log.clone(),
    };
    let compile = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: options.limits,
    };
    let driver = Driver::new(
        Arc::new(handle),
        Box::new(executor),
        DriverOptions {
            prefill: compile,
            decode: compile,
            capacity: nz(32),
            prefill_chunk: nz(options.chunk),
            max_trace_tokens: nz(options.max_trace_tokens),
            prepared: PreparedSetLimits {
                max_entries: nz(options.max_entries),
                max_retained_bytes: NonZeroU64::new(options.max_retained_bytes).unwrap(),
            },
            charge: options.charge,
        },
    )
    .unwrap();
    Rig {
        driver,
        asked,
        log,
        _checkpoint: dir,
    }
}

fn contiguous(heads: &[Head]) -> ServingShapes<'_> {
    ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads,
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

fn request(prompt: &[u32], max_new: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_vec(),
        max_new,
        sampler: Sampler::greedy(),
        stops: Vec::new(),
        ignore_eos: false,
    }
}

fn generate(rig: &mut Rig, request: GenerateRequest) {
    rig.driver
        .generate(request, &mut |_: u32, _: &str| {
            GenerationControl::Continue(())
        })
        .unwrap();
}

/// The tokens one program's step carries: the extent of its `Slot::Token` input.
fn token_extent(program: &StagedProgram<ValidationOutputs>) -> usize {
    let (_, _, stage) = program.stages().next().expect("one stage");
    let graph = stage.graph();
    graph
        .slots
        .iter()
        .find(|(_, slot)| *slot == Slot::Token)
        .map(|(value, _)| graph.aval(*value).shape.iter().product())
        .expect("every entry declares its token slot")
}

/// One fixture-owned 32-byte kernel per distinct token extent, no private bytes: entries of one extent
/// (the one-token step) hold the same owner across heads.
fn one_kernel_per_extent(program: &StagedProgram<ValidationOutputs>) -> Retention {
    Retention {
        shared: vec![SharedOwner {
            key: format!("kernel@{}", token_extent(program)),
            bytes: 32,
        }],
        private: 0,
    }
}

/// A 32-byte kernel every entry shares, plus 8 private bytes each.
fn one_shared_kernel(_: &StagedProgram<ValidationOutputs>) -> Retention {
    Retention {
        shared: vec![SharedOwner {
            key: "the-one-kernel".to_string(),
            bytes: 32,
        }],
        private: 8,
    }
}

// ---------------------------------------------------------------------------------------------
// Rows.
// ---------------------------------------------------------------------------------------------

/// SC-003 (count): under `max_entries = 2` and `max_retained_bytes = 96`, two distinct 32-byte
/// fixture-owned kernels are admitted and the third entry is refused by count, with the first two still
/// serving (a request that needs only them replays with no compile). Mutation: skip the entry check;
/// the third enters and the typed refusal assertion fails.
#[test]
fn the_third_distinct_owner_is_refused_by_count_and_the_first_two_keep_serving() {
    let mut rig = rig(RigOptions {
        max_entries: 2,
        max_retained_bytes: 96,
        charge: one_kernel_per_extent,
        ..RigOptions::default()
    });
    let err = rig
        .driver
        .prepare(&contiguous(&[Head::GREEDY]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::Entries { held: 2, limit: 2 })
        ),
        "{err:?}"
    );
    assert_eq!(rig.driver.prepared_entries(), 2);
    assert_eq!(rig.driver.retained_bytes(), 64);
    // The first admitted piece is the full chunk: a 4-token prompt replays it.
    let before = rig.driver.stats().compiles;
    generate(&mut rig, request(&[1; 4], 1));
    assert_eq!(rig.driver.stats().compiles, before);
}

/// SC-003 (bytes): with room for four entries but 63 bytes, the second 32-byte owner is refused on bytes
/// and no counter moves. Mutation: skip the byte check; the second owner enters.
#[test]
fn the_second_owner_is_refused_by_bytes_without_moving_the_counters() {
    let mut rig = rig(RigOptions {
        max_entries: 4,
        max_retained_bytes: 63,
        charge: one_kernel_per_extent,
        ..RigOptions::default()
    });
    let err = rig
        .driver
        .prepare(&contiguous(&[Head::GREEDY]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::Bytes {
                charge: 32,
                held: 32,
                would_retain: 64,
                limit: 63
            })
        ),
        "{err:?}"
    );
    assert_eq!(rig.driver.prepared_entries(), 1);
    assert_eq!(rig.driver.retained_bytes(), 32);
}

/// SC-003 (sharing): the three admitted greedy entries (prefill 4 and 2, and the one-token step) name
/// three kernels. The logits head's three entries have the same token extents, so they hold the same
/// three owners: six entries with eight private bytes each plus three 32-byte owners is 144 held. One
/// byte under and the last entry, which adds only its private bytes, is refused, with the shared owner
/// not charged again. Mutation: charge a shared owner to every holder; the total is 240 and both
/// assertions fail. Mutation: drop the private bytes; the total is 96.
#[test]
fn a_shared_owner_is_charged_once_and_private_bytes_separately() {
    let mut rig = rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let heads = [Head::GREEDY, Head::Logits];
    rig.driver.prepare(&contiguous(&heads)).unwrap();
    assert_eq!(rig.driver.prepared_entries(), 6);
    assert_eq!(rig.driver.retained_bytes(), 3 * 32 + 6 * 8);

    let mut tight = self::rig(RigOptions {
        max_retained_bytes: 143,
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let err = tight.driver.prepare(&contiguous(&heads)).unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::Bytes {
                charge: 8,
                held: 136,
                would_retain: 144,
                limit: 143
            })
        ),
        "the last logits entry holds the shared owner free and adds only its private bytes: {err:?}"
    );
}

/// SC-003 (one owner for every entry): a single shared kernel is charged once however many entries
/// hold it. Mutation: key an owner by entry rather than by its key; every entry is charged 40.
#[test]
fn one_kernel_shared_by_every_entry_is_charged_once() {
    let mut rig = rig(RigOptions {
        charge: one_shared_kernel,
        ..RigOptions::default()
    });
    rig.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    assert_eq!(rig.driver.prepared_entries(), 3);
    assert_eq!(rig.driver.retained_bytes(), 32 + 3 * 8);
}

/// SC-003 (failure after reservation): a load that fails after the driver measured and reserved the
/// entry leaves every counter exactly where it was, and the same entry then prepares cleanly: nothing
/// was held, so nothing leaked. Mutation: record the hold before `add_entry`; the failed attempt leaves
/// 40 bytes held and the counters row goes red.
#[test]
fn a_failed_load_after_reservation_leaves_the_counters_where_they_were() {
    let mut rig = rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    rig.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    let held = (
        rig.driver.prepared_entries(),
        rig.driver.retained_bytes(),
        rig.driver.stats().entries_added,
    );

    // A new schema (the logits head): the load of its first entry fails.
    rig.asked.borrow_mut().fail_add = true;
    let heads = [Head::GREEDY, Head::Logits];
    let err = rig.driver.prepare(&contiguous(&heads)).unwrap_err();
    assert!(matches!(err, DriverError::Device(_)), "{err:?}");
    assert_eq!(
        (
            rig.driver.prepared_entries(),
            rig.driver.retained_bytes(),
            rig.driver.stats().entries_added,
        ),
        held,
        "a failed attempt holds nothing"
    );

    rig.driver.prepare(&contiguous(&heads)).unwrap();
    let mut clean = self::rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    clean.driver.prepare(&contiguous(&heads)).unwrap();
    assert_eq!(
        rig.driver.retained_bytes(),
        clean.driver.retained_bytes(),
        "the retry holds what a clean run holds, with no residue of the failure"
    );
}

/// SC-002 and SC-003 on the real load path: a kernel artifact over `max_artifact_bytes` is refused when
/// the engine loads the entry, the typed error names the limit, and the driver holds nothing for the
/// failed entry. The limit is the program's own (`CompileLimits` on the driver's compile options), so
/// this is the propagation from compile to executor. A cap the artifacts fit loads the same entry.
/// Mutation: load kernels under a fixed cap instead of `Program::limits()` in `Engine::add_entry`; the
/// one-byte cap admits the entry and the refusal assertion fails.
#[test]
fn an_artifact_over_the_program_limit_fails_the_load_and_holds_nothing() {
    let mut rig = rig(RigOptions {
        limits: CompileLimits {
            max_artifact_bytes: NonZeroU64::new(1).unwrap(),
            ..CompileLimits::STANDARD
        },
        ..RigOptions::default()
    });
    let err = rig
        .driver
        .prepare(&contiguous(&[Head::GREEDY]))
        .unwrap_err();
    let DriverError::Device(ExecError::Load(load)) = &err else {
        panic!("expected a load refusal, got {err:?}");
    };
    let LoadError::Codegen { source, .. } = &**load else {
        panic!("expected a codegen load error, got {load:?}");
    };
    assert!(
        matches!(
            source,
            poot_codegen::CompileError::ArtifactTooLarge(over) if over.limit == 1 && over.bytes > 1
        ),
        "{source:?}"
    );
    assert_eq!(
        (rig.driver.prepared_entries(), rig.driver.retained_bytes()),
        (0, 0),
        "the failed entry holds nothing"
    );

    let mut roomy = self::rig(RigOptions::default());
    roomy.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    assert_eq!(roomy.driver.prepared_entries(), 3);
}

/// SC-004 (schema identity): a new head is a new admitted schema. It compiles once, adds its own entries
/// and no others, and preparing it again compiles nothing, so no schema reuses another's entry. Then
/// requests drawn from the warmed set compile nothing while prompts, tokens and positions change.
/// Mutation: drop the head from the entry key; the second `prepare` finds every entry present and adds
/// none, so the compile delta row fails.
#[test]
fn a_new_schema_compiles_once_and_a_warmed_set_never_compiles_again() {
    let mut rig = rig(RigOptions::default());
    rig.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    let greedy = rig.driver.stats();
    assert_eq!((greedy.compiles, rig.driver.prepared_entries()), (3, 3));

    let heads = [Head::GREEDY, Head::Logits];
    rig.driver.prepare(&contiguous(&heads)).unwrap();
    let both = rig.driver.stats();
    assert_eq!(
        (
            both.compiles - greedy.compiles,
            rig.driver.prepared_entries()
        ),
        (3, 6),
        "the logits head compiles its own three entries, once"
    );
    rig.driver.prepare(&contiguous(&heads)).unwrap();
    assert_eq!(
        rig.driver.stats().compiles,
        both.compiles,
        "a repeat adds nothing"
    );

    for (prompt, max_new) in [(vec![1u32; 4], 3usize), (vec![2; 7], 4), (vec![3; 11], 5)] {
        generate(&mut rig, request(&prompt, max_new));
    }
    let after = rig.driver.stats();
    assert_eq!(
        (after.compiles, after.entries_added),
        (both.compiles, both.entries_added),
        "positions, tokens and prompts never specialize an entry"
    );
    let asked = rig.asked.borrow();
    let prefill: Vec<usize> = asked
        .steps
        .iter()
        .filter_map(|step| asked.entries.iter().find(|(id, _)| id == step))
        .filter(|(_, (phase, _))| *phase == Phase::Prefill)
        .map(|(_, (_, shape))| shape.tokens.get())
        .take(1 + 2 + 3)
        .collect();
    assert_eq!(
        prefill,
        [4, 4, 2, 4, 4, 2],
        "the prompts of 4, 7 and 11 plan to exactly these pieces; a one-token tail is the decode-phase entry"
    );
}

/// SC-005 (trace extent): a step above `max_trace_tokens` is refused with a typed error before the
/// family traces it, and one at the limit is admitted. The family is registered outside the crate, so
/// the bound is not a built-in's (a recurrent family serves no verify windows, so the window route
/// that reaches an over-limit step is a plain one). Mutation: drop the extent
/// check in `Driver::prepared`; the family is traced for 5 tokens and the trace-log assertion fails.
#[test]
fn a_step_above_the_trace_limit_is_refused_before_the_family_traces_it() {
    let mut rig = rig(RigOptions {
        granule: 2,
        chunk: 4,
        max_trace_tokens: 4,
        ..RigOptions::default()
    });
    let pool = PoolShape {
        blocks: nz(6),
        max_seqs: nz(1),
    };
    let window = |width: usize| [nz(width)];
    let shapes = |windows: &[NonZeroUsize]| {
        // The slice must outlive the call: copy into a local the caller borrows.
        windows.to_vec()
    };
    let at_limit = shapes(&window(4));
    rig.driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool),
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[]),
            windows: &at_limit,
            adapters: &[],
        })
        .expect("a window at the trace limit is admitted");
    let traced = rig.log.lock().unwrap().len();

    let over = shapes(&window(6));
    let err = rig
        .driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool),
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[]),
            windows: &over,
            adapters: &[],
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::TraceTokens {
                tokens: 6,
                limit: 4
            })
        ),
        "{err:?}"
    );
    assert_eq!(
        rig.log.lock().unwrap().len(),
        traced,
        "the family was never asked to trace the 6-token step"
    );
    assert!(
        rig.log
            .lock()
            .unwrap()
            .iter()
            .all(|(_, shape)| shape.tokens.get() <= 4),
        "no traced step exceeds the limit"
    );
    assert!(matches!(
        rig.log.lock().unwrap()[0].1.kv,
        KvLayout::Paged { .. }
    ));
}

/// SC-005 (legal chunks): a 6-token prompt over `max_trace_tokens = 4` at granule 2 runs as the legal
/// chunks [4, 2] on one continuing state: the driver resets the executor's state once, before the first
/// chunk, and never between chunks. Mutation: reset before every piece (discard the continuation
/// state); the reset count is 3. Mutation: plan the prompt as one piece of 6; the trace limit refuses it.
#[test]
fn a_prompt_over_the_trace_limit_runs_as_legal_chunks_on_one_continuing_state() {
    let mut rig = rig(RigOptions {
        recurrent: true,
        granule: 2,
        chunk: 4,
        max_trace_tokens: 4,
        ..RigOptions::default()
    });
    generate(&mut rig, request(&[1; 6], 1));
    let asked = rig.asked.borrow();
    let shapes: Vec<(Phase, usize)> = asked
        .steps
        .iter()
        .map(|step| {
            let (_, (phase, shape)) = asked
                .entries
                .iter()
                .find(|(id, _)| id == step)
                .expect("a step runs a prepared entry");
            (*phase, shape.tokens.get())
        })
        .collect();
    assert_eq!(shapes[..2], [(Phase::Prefill, 4), (Phase::Prefill, 2)]);
    assert_eq!(asked.resets, 1, "the state continues across chunks");
}
