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
    Device, DeviceError, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, LoadError, StateScope, StepInputs, StepOutputs,
};
use poot_graph_ir::{BinOp, Graph, Slot, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, staged::StagedProgram, staged::TargetSet,
};
use poot_llm::driver::error::{DriverError, PreparedSetRefusal, Unsupported};
use poot_llm::driver::{
    Driver, DriverOptions, GenerateRequest, Head, Layout, ModelHandle, PoolShape,
    PreparedSetLimits, Retention, Serving, ServingShapes, SharedOwner, State, Unprepared, Warm,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::components::standard::Step;
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec, StateSpec};
use poot_models::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput,
    ShapeReason, StepShape, StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig, Registry};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;
use poot_text::{ChatFormat, HermesToolCodec};
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

type TraceLog = Arc<Mutex<Vec<StepShape>>>;

#[derive(Debug)]
struct Tiny {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
    log: TraceLog,
    recurrent: bool,
}

impl Model for Tiny {
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
        let refuse = |reason| {
            Err(TraceError::ShapeUnsupported {
                family: TINY,
                shape,
                reason,
            })
        };
        let granule = self.config.prefill_granule.get();
        if !shape.is_decode() && !shape.tokens.get().is_multiple_of(granule) {
            return refuse(ShapeReason::Granule { granule });
        }
        self.log.lock().unwrap().push(shape);
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
        if self.recurrent {
            let dims = vec![rows, 4];
            let state = b.state_input(
                "toy.state",
                TensorType::f32(dims.clone()),
                StateRole::Recurrent,
            )?;
            let folded = b.binary(BinOp::Add, state, b.broadcast(seed, dims)?)?;
            step.carry(state, folded);
        }
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1])?, vec![rows, out, VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

thread_local! {
    static TRACE_LOG: RefCell<Option<TraceLog>> = const { RefCell::new(None) };
}

fn build_tiny(raw: &RawConfig<'_>, log: TraceLog) -> Result<ModelParts, ModelError> {
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
    let recurrent = config.get("state").and_then(|v| v.as_str()) == Some("recurrent");
    let program = ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: VOCAB });
    let program = if recurrent {
        program.carrying(StateSpec::recurrent("toy.state"))
    } else {
        program
    };
    Ok(ModelParts::new(Box::new(Tiny {
        recurrent,
        schema: ModelSchema::builder().step(program).build_for(TINY)?,
        config: ModelConfig {
            family: TINY,
            vocab: VOCAB,
            max_positions: field("max_positions")?,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            prefill_granule: granule,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        weights: WeightMap::default(),
        log,
    })))
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
    /// `shape` of each entry, in `add_entry` order, from the family's trace log.
    entries: Vec<(EntryId, StepShape)>,
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
    fn free_memory_bytes(&self) -> Option<u64> {
        self.inner.free_memory_bytes()
    }
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError> {
        self.inner.load_weights(store, weights)
    }
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
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

struct Rig<S: State = Unprepared> {
    driver: Driver<S>,
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
                preprocessor: None,
                store: WeightStore::default(),
            },
            variants: &[],
        })
        .unwrap();
    let dir = checkpoint_dir(&config, "prepared_budgets_checkpoint");
    let handle =
        ModelHandle::load(&dir, &registry, &poot_llm::driver::LoadOptions::default()).unwrap();
    let asked: Rc<RefCell<Asked>> = Rc::default();
    let executor = Wrapped {
        inner: Engine::new(FakeDevice),
        asked: asked.clone(),
        log: log.clone(),
    };
    let compile = CompileOptions::new(options.limits);
    let driver = Driver::new(
        Arc::new(handle),
        Box::new(executor),
        DriverOptions {
            prefill: compile.clone(),
            decode: compile,
            capacity: nz(32),
            prefill_chunk: nz(options.chunk),
            trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(options.max_trace_tokens)),
            prepared: PreparedSetLimits {
                max_entries: nz(options.max_entries),
                max_retained_bytes: NonZeroU64::new(options.max_retained_bytes).unwrap(),
            },
            charge: options.charge,
            draft: poot_llm::driver::DraftPolicy::Off,
            hidden_window: None,
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

impl Rig<Unprepared> {
    /// Prepares `shapes`, which the rig's driver must admit.
    fn prepare(self, shapes: &ServingShapes<'_>) -> Rig<Serving> {
        let Rig {
            driver,
            asked,
            log,
            _checkpoint,
        } = self;
        Rig {
            driver: driver.prepare(shapes).unwrap(),
            asked,
            log,
            _checkpoint,
        }
    }

    /// Prepares `shapes`, which the driver must refuse: the rig as the refusal left it, and the cause.
    fn prepare_refused(self, shapes: &ServingShapes<'_>) -> (Rig, DriverError) {
        let Rig {
            driver,
            asked,
            log,
            _checkpoint,
        } = self;
        let refused = driver.prepare(shapes).unwrap_err();
        let rig = Rig {
            driver: refused.driver,
            asked,
            log,
            _checkpoint,
        };
        (rig, refused.error)
    }

    /// Serves `layout` with nothing prepared: what `generate` prepares for its request.
    fn serve(self, layout: Layout) -> Rig<Serving> {
        let Rig {
            driver,
            asked,
            log,
            _checkpoint,
        } = self;
        Rig {
            driver: driver.serve(layout),
            asked,
            log,
            _checkpoint,
        }
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

fn generate(rig: &mut Rig<Serving>, request: GenerateRequest) {
    rig.driver
        .generate(request, &mut |_: u32, _: &str| {
            GenerationControl::Continue(())
        })
        .unwrap();
}

/// The tokens one program's step carries: the extent of its `Slot::Token` input.
fn token_extent(program: &StagedProgram) -> usize {
    let (_, _, stage) = program.stages().next().expect("one stage");
    let graph = stage.graph();
    graph
        .slots()
        .iter()
        .find(|(_, slot)| *slot == Slot::Token)
        .map(|(value, _)| graph.aval(*value).shape.iter().product())
        .expect("every entry declares its token slot")
}

/// One fixture-owned 32-byte kernel per distinct token extent, no private bytes: entries of one extent
/// (the one-token step) hold the same owner across heads.
fn one_kernel_per_extent(program: &StagedProgram) -> Retention {
    Retention {
        shared: vec![SharedOwner {
            key: format!("kernel@{}", token_extent(program)),
            bytes: 32,
        }],
        private: 0,
    }
}

/// A 32-byte kernel every entry shares, plus 8 private bytes each.
fn one_shared_kernel(_: &StagedProgram) -> Retention {
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
    let rig = rig(RigOptions {
        max_entries: 2,
        max_retained_bytes: 96,
        charge: one_kernel_per_extent,
        ..RigOptions::default()
    });
    let (rig, err) = rig.prepare_refused(&contiguous(&[Head::GREEDY]));
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
    let mut rig = rig.serve(Layout::Contiguous);
    generate(&mut rig, request(&[1; 4], 1));
    assert_eq!(rig.driver.stats().compiles, before);
}

/// SC-003 (bytes): with room for four entries but 63 bytes, the second 32-byte owner is refused on bytes
/// and no counter moves. Mutation: skip the byte check; the second owner enters.
#[test]
fn the_second_owner_is_refused_by_bytes_without_moving_the_counters() {
    let rig = rig(RigOptions {
        max_entries: 4,
        max_retained_bytes: 63,
        charge: one_kernel_per_extent,
        ..RigOptions::default()
    });
    let (rig, err) = rig.prepare_refused(&contiguous(&[Head::GREEDY]));
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
    let rig = rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let heads = [Head::GREEDY, Head::Logits];
    let rig = rig.prepare(&contiguous(&heads));
    assert_eq!(rig.driver.prepared_entries(), 6);
    assert_eq!(rig.driver.retained_bytes(), 3 * 32 + 6 * 8);

    let tight = self::rig(RigOptions {
        max_retained_bytes: 143,
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let (_, err) = tight.prepare_refused(&contiguous(&heads));
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
    let rig = rig(RigOptions {
        charge: one_shared_kernel,
        ..RigOptions::default()
    });
    let rig = rig.prepare(&contiguous(&[Head::GREEDY]));
    assert_eq!(rig.driver.prepared_entries(), 3);
    assert_eq!(rig.driver.retained_bytes(), 32 + 3 * 8);
}

/// SC-003 (failure after reservation): a load that fails after the driver measured and reserved the
/// entry leaves every counter exactly where it was, and the same entry then prepares cleanly: nothing
/// was held, so nothing leaked. Mutation: record the hold before `add_entry`; the failed attempt leaves
/// 40 bytes held and the counters row goes red.
#[test]
fn a_failed_load_after_reservation_leaves_the_counters_where_they_were() {
    let rig = rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let mut rig = rig.prepare(&contiguous(&[Head::GREEDY]));
    let held = (
        rig.driver.prepared_entries(),
        rig.driver.retained_bytes(),
        rig.driver.stats().entries_added,
    );

    // A new schema (the logits head): the load of its first entry fails.
    rig.asked.borrow_mut().fail_add = true;
    let heads = [Head::GREEDY, Head::Logits];
    let err = rig.driver.extend(&contiguous(&heads)).unwrap_err();
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

    rig.driver.extend(&contiguous(&heads)).unwrap();
    let clean = self::rig(RigOptions {
        charge: |program| {
            let mut retention = one_kernel_per_extent(program);
            retention.private = 8;
            retention
        },
        ..RigOptions::default()
    });
    let clean = clean.prepare(&contiguous(&heads));
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
    let rig = rig(RigOptions {
        limits: CompileLimits {
            max_artifact_bytes: NonZeroU64::new(1).unwrap(),
            ..CompileLimits::STANDARD
        },
        ..RigOptions::default()
    });
    let (rig, err) = rig.prepare_refused(&contiguous(&[Head::GREEDY]));
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

    let roomy = self::rig(RigOptions::default());
    let roomy = roomy.prepare(&contiguous(&[Head::GREEDY]));
    assert_eq!(roomy.driver.prepared_entries(), 3);
}

/// SC-004 (schema identity): a new head is a new admitted schema. It compiles once, adds its own entries
/// and no others, and preparing it again compiles nothing, so no schema reuses another's entry. Then
/// requests drawn from the warmed set compile nothing while prompts, tokens and positions change.
/// Mutation: drop the head from the entry key; the second `prepare` finds every entry present and adds
/// none, so the compile delta row fails.
#[test]
fn a_new_schema_compiles_once_and_a_warmed_set_never_compiles_again() {
    let rig = rig(RigOptions::default());
    let mut rig = rig.prepare(&contiguous(&[Head::GREEDY]));
    let greedy = rig.driver.stats();
    assert_eq!((greedy.compiles, rig.driver.prepared_entries()), (3, 3));

    let heads = [Head::GREEDY, Head::Logits];
    rig.driver.extend(&contiguous(&heads)).unwrap();
    let both = rig.driver.stats();
    assert_eq!(
        (
            both.compiles - greedy.compiles,
            rig.driver.prepared_entries()
        ),
        (3, 6),
        "the logits head compiles its own three entries, once"
    );
    rig.driver.extend(&contiguous(&heads)).unwrap();
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
        .filter(|(_, shape)| !shape.is_decode())
        .map(|(_, shape)| shape.tokens.get())
        .take(1 + 2 + 3)
        .collect();
    assert_eq!(
        prefill,
        [4, 4, 2, 4, 4, 2],
        "the prompts of 4, 7 and 11 plan to exactly these pieces; a one-token tail is the decode-phase entry"
    );
}

/// SC-005 (trace extent): a step above `max_tokens` is refused with the typed budget cause before the
/// family traces it, and one at the limit is admitted. The family is registered outside the crate, so
/// the bound is not a built-in's (a recurrent family serves no verify windows, so the window route
/// that reaches an over-limit step is a plain one). Mutation: drop the token admission in
/// `Model::trace`; the family is traced for 6 tokens and the trace-log assertion fails.
#[test]
fn a_step_above_the_trace_limit_is_refused_before_the_family_traces_it() {
    let rig = rig(RigOptions {
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
    let mut rig = rig.prepare(&ServingShapes {
        layout: Layout::Paged(pool),
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Prompts(&[]),
        windows: &at_limit,
        adapters: &[],
    });
    let traced = rig.log.lock().unwrap().len();

    let over = shapes(&window(6));
    let err = rig
        .driver
        .extend(&ServingShapes {
            layout: Layout::Paged(pool),
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[]),
            windows: &over,
            adapters: &[],
        })
        .unwrap_err();
    let DriverError::Unsupported(Unsupported::Trace(TraceError::Budget(refused))) = &err else {
        panic!("a step above the trace limit is the typed budget refusal: {err:?}");
    };
    assert_eq!(
        (refused.resource, refused.limit, refused.requested),
        (poot_graph_ir::BudgetResource::Tokens, 4, 6)
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
            .all(|shape| shape.tokens.get() <= 4),
        "no traced step exceeds the limit"
    );
    assert!(matches!(
        rig.log.lock().unwrap()[0].kv,
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
    })
    .serve(Layout::Contiguous);
    generate(&mut rig, request(&[1; 6], 1));
    let asked = rig.asked.borrow();
    let shapes: Vec<usize> = asked
        .steps
        .iter()
        .map(|step| {
            let (_, shape) = asked
                .entries
                .iter()
                .find(|(id, _)| id == step)
                .expect("a step runs a prepared entry");
            shape.tokens.get()
        })
        .collect();
    assert_eq!(shapes[..2], [4, 2]);
    assert_eq!(asked.resets, 1, "the state continues across chunks");
}

// ---------------------------------------------------------------------------------------------
// POOT-733 SC-006: the shipped qwen2 family through the real `Driver`, under explicit caller limits.
// ---------------------------------------------------------------------------------------------

/// The shipped qwen2 family's synthetic checkpoint, loaded through the public loader and served by a
/// `Driver` over the real engine with only the device faked. The caller's `trace` limits are the only
/// policy; nothing about the family is registered from outside.
struct Qwen2Rig {
    driver: Driver,
    /// The same family built straight from the registry entry: what the driver traces.
    model: Box<dyn Model>,
    _checkpoint: poot_test_util::UniqueTempPath,
}

fn qwen2_rig(trace: poot_graph_ir::TraceLimits) -> Qwen2Rig {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family == poot_models::qwen2::model::FAMILY)
        .copied()
        .expect("qwen2 is a shipped family");
    let fixture = (entry.fixture)();
    let dir = poot_test_util::unique_temp_path("prepared_budgets_qwen2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    let words = (0..48usize).map(|i| (format!("w{i}"), i as u32)).collect();
    let model = WordLevel::builder()
        .vocab(words)
        .unk_token("w0".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
    let model = (entry.build)(&fixture.raw(), &fixture.store).unwrap();
    let handle = Arc::new(
        ModelHandle::load(&dir, &registry, &poot_llm::driver::LoadOptions::default()).unwrap(),
    );
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let driver = Driver::new(
        handle,
        Box::new(Engine::new(FakeDevice)),
        DriverOptions {
            prefill: compile.clone(),
            decode: compile,
            capacity: nz(16),
            prefill_chunk: nz(4),
            trace,
            prepared: PreparedSetLimits {
                max_entries: nz(64),
                max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
            },
            charge: poot_llm::driver::program_retention,
            draft: poot_llm::driver::DraftPolicy::Off,
            hidden_window: None,
        },
    )
    .unwrap();
    Qwen2Rig {
        driver,
        model: model.model,
        _checkpoint: dir,
    }
}

fn decode_only() -> ServingShapes<'static> {
    ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        // Raw logits: no sampling suffix extends the traced graph, so the trace limits bind the
        // family's graph alone.
        heads: &[Head::Logits],
        warm: Warm::Prompts(&[]),
        windows: &[],
        adapters: &[],
    }
}

fn with_window(tokens: usize) -> ServingShapes<'static> {
    let windows: &'static [NonZeroUsize] = Box::leak(vec![nz(tokens)].into_boxed_slice());
    ServingShapes {
        windows,
        ..decode_only()
    }
}

fn trace_budget(error: &DriverError) -> poot_graph_ir::TraceBudgetExceeded {
    match error {
        DriverError::Unsupported(Unsupported::Trace(TraceError::Budget(budget))) => *budget,
        other => panic!("expected the typed trace-budget cause, got {other:?}"),
    }
}

/// SC-006: the driver passes its caller's limits to the family's tracer. Adequate limits prepare the
/// entries a roomy policy does (the same count and the same retained bytes: the traced graphs are the
/// same); `max_tokens = 4` refuses a 5-token window with the exact typed cause and leaves the prepared
/// set as it was; `max_eqns = 2` refuses at the third attempted equation, publishing no entry; an owned-byte ceiling
/// below what the tracer's records need refuses at the named allocation. Mutations: the driver traces
/// under `TraceLimits::for_tests()` instead of its options (every refusal row goes red); a nested
/// unbudgeted builder in the family's `Step`; the budget stringified into a plain build error (the
/// typed-cause match fails); the equation count checked only against the finished graph (the refusal's
/// `attempted` is the whole graph's, not the first equation past the limit).
#[test]
fn the_shipped_qwen2_path_consumes_the_callers_limits_and_refuses_with_the_exact_cause() {
    use poot_graph_ir::{AllocationSite, Attempted, BudgetResource, TraceLimits};

    let roomy = TraceLimits::for_tests_with_tokens(nz(8));
    let reference = qwen2_rig(roomy);
    let reference_driver = reference.driver.prepare(&with_window(3)).unwrap();
    let (entries, retained) = (
        reference_driver.prepared_entries(),
        reference_driver.retained_bytes(),
    );
    assert!(entries >= 2, "a decode entry and a 3-token window");

    // Adequate finite limits: exactly the graphs' sizes, the same prepared set.
    let shape = |tokens: usize| StepShape::new(NonZeroUsize::MIN, nz(tokens), nz(16));
    let model = &reference.model;
    let biggest = model.trace(&StepSpec::step(shape(3)), roomy).unwrap();
    let decode = model.trace(&StepSpec::step(shape(1)), roomy).unwrap();
    let adequate = TraceLimits {
        max_tokens: nz(4),
        max_values: nz(biggest.values().len().max(decode.values().len()) + 16),
        max_eqns: nz(biggest.eqns().len().max(decode.eqns().len()) + 16),
        max_literal_bytes: NonZeroU64::new(1 << 20).unwrap(),
        max_owned_bytes: NonZeroU64::new(1 << 30).unwrap(),
    };
    let adequate_rig = qwen2_rig(adequate);
    let adequate_driver = adequate_rig.driver.prepare(&with_window(3)).unwrap();
    assert_eq!(adequate_driver.prepared_entries(), entries);
    assert_eq!(adequate_driver.retained_bytes(), retained);

    // `max_tokens = 4`: a 5-token window is refused before the family traces it, and what was
    // prepared before stays.
    let rig = qwen2_rig(adequate);
    let mut driver = rig.driver.prepare(&decode_only()).unwrap();
    let held = (driver.prepared_entries(), driver.retained_bytes());
    let compiles = driver.stats().compiles;
    let refused = trace_budget(&driver.extend(&with_window(5)).unwrap_err());
    assert_eq!(
        (refused.resource, refused.limit, refused.requested),
        (BudgetResource::Tokens, 4, 5)
    );
    assert_eq!(refused.attempted, Attempted::Total(5));
    assert_eq!(refused.site, AllocationSite::TokenRequest);
    assert_eq!((driver.prepared_entries(), driver.retained_bytes()), held);
    assert_eq!(driver.stats().compiles, compiles, "nothing compiled");

    // `max_eqns = 2`: the third attempted equation is refused where it is attempted, the driver
    // returns the exact typed cause, nothing is published, and asking again changes nothing.
    let rig = qwen2_rig(TraceLimits {
        max_eqns: nz(2),
        ..roomy
    });
    let mut driver = rig.driver;
    let compiles = driver.stats().compiles;
    for _ in 0..2 {
        let refusal = driver.prepare(&decode_only()).unwrap_err();
        let refused = trace_budget(&refusal.error);
        driver = refusal.driver;
        assert_eq!(refused.resource, BudgetResource::Equations);
        assert_eq!((refused.limit, refused.used, refused.requested), (2, 2, 1));
        assert_eq!(refused.attempted, Attempted::Total(3));
        assert_eq!(refused.site, AllocationSite::EquationRecord);
        assert_eq!(
            (driver.prepared_entries(), driver.retained_bytes()),
            (0, 0),
            "the refusal published an entry"
        );
        assert_eq!(driver.stats().compiles, compiles);
    }

    // An owned-byte ceiling below the family's records: the smallest ceiling that admits the decode
    // step, then one byte less refuses at the allocation site the error names.
    let admits = |ceiling: u64| {
        let limits = TraceLimits {
            max_owned_bytes: NonZeroU64::new(ceiling).unwrap(),
            ..roomy
        };
        model.trace(&StepSpec::step(shape(1)), limits).map(drop)
    };
    let (mut low, mut high) = (1u64, 1u64 << 24);
    assert!(admits(high).is_ok());
    while low < high {
        let mid = low + (high - low) / 2;
        if admits(mid).is_ok() {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let ceiling = TraceLimits {
        max_owned_bytes: NonZeroU64::new(low - 1).unwrap(),
        ..roomy
    };
    let rig = qwen2_rig(ceiling);
    let refusal = rig.driver.prepare(&decode_only()).unwrap_err();
    let refused = trace_budget(&refusal.error);
    let driver = refusal.driver;
    assert_eq!(refused.resource, BudgetResource::OwnedBytes);
    assert_eq!(refused.limit, low - 1);
    assert!(refused.used + refused.requested > low - 1);
    // The Driver names the same allocation the model's own trace refuses at under that ceiling.
    let direct = model
        .trace(&StepSpec::step(shape(1)), ceiling)
        .map(drop)
        .unwrap_err();
    let TraceError::Budget(direct) = direct else {
        panic!("the model's own trace refuses with the budget cause: {direct:?}");
    };
    assert_eq!(refused.site, direct.site);
    assert_eq!(refused.site, AllocationSite::FinalState);
    assert_eq!((driver.prepared_entries(), driver.retained_bytes()), (0, 0));
}
