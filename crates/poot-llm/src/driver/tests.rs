//! Driver rows on a counting fake executor (Card 734). The fake is a [`Device`] under the
//! real `poot_executor::Engine`: a step still traces through a [`Model`], compiles for a fixture
//! [`Target`] through the real planner, and loads (with real SPIR-V codegen from `nix develop`'s `llc`)
//! through the real contract, so planner refusals, slot binding and entry sharing are production
//! behavior; only the device is a fake. The fake records what each step bound (`Slot::Token`,
//! `Slot::Pos`, the sampler slots) and answers each output readback from a test script, so a row sees
//! exactly what reached the executor and chooses exactly what came back.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use poot_executor::{
    BufferRole, Device, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, StateScope, StepInputs, StepOutputs, WeightSource,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, ValidationOutputs};
use poot_graph_plan::{CompileOptions, FusionPolicy, StagedProgram, Submission, Target, TargetSet};
use poot_models::chat::ChatFormat;
use poot_models::components::standard::Step;
use poot_models::model::{
    ConfigReason, FamilyKey, LogitRows, Model, ModelConfig, ModelError, ModelOutput, Phase,
    ShapeReason, StepShape, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_target::{Backend, DeviceCaps, ElementKind};
use poot_tensor::DType;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;

use crate::GenerationControl;
use crate::core::sampler::{Sampler, SamplerFault};
use crate::driver::error::{
    DriverError, InvalidOptions, InvalidRequest, PreparedSetRefusal, Unsupported,
};
use crate::driver::{
    Driver, DriverOptions, FinishReason, GenerateRequest, Generation, Head, Layout, ModelHandle,
    PreparedSetLimits, Retention, ServingShapes, Warm,
};
use crate::text::tokenize::{ChatTemplate, TextCodec};

/// The fake model's vocabulary: `<unk>` and `a` .. `k`, ids 0 ..= 11.
const VOCAB: usize = 12;
const EOS: u32 = 11;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// The fake model: a family registered from outside the crate, tracing a tiny graph over the step's
// own slots (its logits depend on the last token and position, so both slots stay live).
// ---------------------------------------------------------------------------------------------

type TraceLog = Arc<Mutex<Vec<(Phase, StepShape)>>>;

const TINY: FamilyKey = FamilyKey::new("tiny");

/// What state or extra input the fake family carries beside its logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Variant {
    /// Positions and tokens only: nothing carried.
    Plain,
    /// One `StateRole::Recurrent` conv-like state, `[rows, 4]`, folded with every token.
    Recurrent,
    /// A recurrent state whose row axis is axis 1 (`[4, rows]`): malformed.
    BadRowAxis,
    /// A `Slot::LoraIdx` input added into the logits.
    Lora,
}

impl Variant {
    fn parse(name: Option<&str>) -> Option<Self> {
        Some(match name {
            None | Some("plain") => Self::Plain,
            Some("recurrent") => Self::Recurrent,
            Some("bad_row_axis") => Self::BadRowAxis,
            Some("lora") => Self::Lora,
            Some(_) => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Recurrent => "recurrent",
            Self::BadRowAxis => "bad_row_axis",
            Self::Lora => "lora",
        }
    }
}

#[derive(Debug)]
struct Tiny {
    config: ModelConfig,
    weights: WeightMap,
    log: TraceLog,
    variant: Variant,
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
        if shape.capacity.get() > self.config.max_positions {
            return refuse(ShapeReason::CapacityAboveMax {
                max: self.config.max_positions,
            });
        }
        if shape.tokens > shape.capacity {
            return refuse(ShapeReason::TokensAboveCapacity);
        }
        self.log.lock().unwrap().push((phase, shape));
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        // The logits rows the shape asks for: every position under `All`, else the last.
        let first = if shape.logits == LogitRows::All {
            0
        } else {
            n - 1
        };
        let out = n - first;
        let token = b.cast(b.slice(step.tokens(), 1, first, n), DType::F32);
        let pos = b.cast(b.slice(step.pos(), 1, first, n), DType::F32);
        let mut seed = b.binary(BinOp::Add, token, pos);
        match self.variant {
            Variant::Plain => {}
            Variant::Recurrent | Variant::BadRowAxis => {
                let dims = if self.variant == Variant::Recurrent {
                    vec![rows, 4]
                } else {
                    vec![4, rows]
                };
                let state = b.state_input(
                    "toy.state",
                    TensorType::f32(dims.clone()),
                    StateRole::Recurrent,
                );
                let spread = if self.variant == Variant::Recurrent {
                    b.broadcast(seed, dims)
                } else {
                    b.broadcast(b.reshape(seed, vec![1, rows]), dims)
                };
                let folded = b.binary(BinOp::Add, state, spread);
                step.carry(state, folded);
            }
            Variant::Lora => {
                let idx = b.slot(Slot::LoraIdx, TensorType::f32(vec![rows]));
                seed = b.binary(BinOp::Add, seed, b.reshape(idx, vec![rows, 1]));
            }
        }
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1]), vec![rows, out, VOCAB]);
        Ok(step.finish(b, logits))
    }
}

/// `config.json` of the fake family: `{"model_type": "tiny", "granule": G, "max_positions": M,
/// "eos_token_id": [..]}`.
fn build_tiny(
    raw: &RawConfig<'_>,
    _store: &WeightStore,
    log: TraceLog,
) -> Result<Box<dyn Model>, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        unreachable!("the fake family names only HF configs")
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
    let eos: BTreeSet<u32> = config
        .get("eos_token_id")
        .and_then(|v| v.as_array())
        .map(|ids| {
            ids.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default();
    let granule = NonZeroUsize::new(field("granule")?).ok_or(ModelError::Config {
        family: TINY,
        field: "granule",
        reason: ConfigReason::Zero,
    })?;
    let variant =
        Variant::parse(config.get("state").and_then(|v| v.as_str())).ok_or(ModelError::Config {
            family: TINY,
            field: "state",
            reason: ConfigReason::Unsupported,
        })?;
    Ok(Box::new(Tiny {
        variant,
        config: ModelConfig {
            family: TINY,
            vocab: VOCAB,
            max_positions: field("max_positions")?,
            eos,
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

fn tiny_config(granule: usize, max_positions: usize, variant: Variant) -> serde_json::Value {
    serde_json::json!({
        "model_type": "tiny",
        "granule": granule,
        "max_positions": max_positions,
        "eos_token_id": [EOS],
        "state": variant.name(),
    })
}

/// The fake family's text services: a word-level tokenizer over `a`..`k` (decode joins with spaces).
fn tiny_text() -> TextCodec {
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
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    TextCodec::new(
        tokenizer,
        None,
        None,
        EOS,
        ChatFormat::ChatML,
        ChatTemplate::default(),
    )
}

// ---------------------------------------------------------------------------------------------
// The fake device and the world a row observes it through.
// ---------------------------------------------------------------------------------------------

/// What one replayed step had bound.
#[derive(Clone, Debug, Default)]
struct StepRecord {
    phase: Option<Phase>,
    shape: Option<StepShape>,
    tokens: Vec<i32>,
    positions: Vec<i32>,
    seed: Vec<i32>,
    params: Vec<f32>,
    top_k: Vec<i32>,
    /// Every input buffer the entry owns, in allocation order, as the step left it.
    inputs: Vec<Vec<u8>>,
}

/// Answers a committed step's output readback: `(record of the step, requested byte length)`.
type Script = Box<dyn FnMut(&StepRecord, usize) -> Vec<u8>>;

struct World {
    script: Script,
    /// `(phase, shape)` of each entry in `add_entry` order, from the model's trace log.
    entries: Vec<(Phase, StepShape)>,
    /// Input-role buffers allocated by each entry, in graph input order.
    entry_inputs: Vec<Vec<usize>>,
    allocated: Vec<(BufferRole, usize, ElementKind)>,
    writes: HashMap<usize, Vec<u8>>,
    last: StepRecord,
    steps: Vec<StepRecord>,
    weight_allocations: usize,
    add_entry_calls: usize,
    /// Make `remove_entry` fail, for the unload-atomicity row.
    fail_remove: bool,
}

type Shared = Rc<RefCell<World>>;

#[derive(Clone)]
struct FakeBuffer {
    id: usize,
    role: BufferRole,
}

struct FakeDevice {
    world: Shared,
    memory: poot_runtime_common::MemoryCounters,
}

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

/// The values of a slot buffer, in the lane the plan stored it in: an index slot (`Slot::Pos`) the
/// planner reads as f32 arrives as f32 words, an I32 one as I32 words.
fn ints(kind: ElementKind, bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| match kind {
            ElementKind::F32 => f32::from_le_bytes(c.try_into().unwrap()) as i32,
            _ => i32::from_le_bytes(c.try_into().unwrap()),
        })
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl Device for FakeDevice {
    type Buffer = FakeBuffer;
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
        self.memory.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<FakeBuffer, FakeError> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        let mut world = self.world.borrow_mut();
        let id = world.allocated.len();
        world.allocated.push((role, elems, storage.element()));
        if role == BufferRole::Weight {
            world.weight_allocations += 1;
        }
        Ok(FakeBuffer { id, role })
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

    fn copy(&mut self, _src: &FakeBuffer, _dst: &FakeBuffer) -> Result<(), FakeError> {
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<()>, FakeError> {
        Ok(Some(()))
    }

    /// A replay is one step: the writes since the last one, grouped by the entry that owns them,
    /// become its record.
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        let mut world = self.world.borrow_mut();
        let written: Vec<usize> = world.writes.keys().copied().collect();
        let Some(entry) = world
            .entry_inputs
            .iter()
            .position(|inputs| written.iter().any(|w| inputs.contains(w)))
        else {
            return Ok(());
        };
        let (phase, shape) = world.entries[entry];
        let bound = |world: &World, ordinal: usize| -> (ElementKind, Vec<u8>) {
            let id = world.entry_inputs[entry].get(ordinal).copied();
            (
                id.map_or(ElementKind::I32, |id| world.allocated[id].2),
                id.and_then(|id| world.writes.get(&id))
                    .cloned()
                    .unwrap_or_default(),
            )
        };
        let ints_at = |world: &World, ordinal| {
            let (kind, bytes) = bound(world, ordinal);
            ints(kind, &bytes)
        };
        // Input buffers are allocated in graph input order: token, pos, then the sampler suffix's
        // seed, params and top_k when the head declares them.
        let record = StepRecord {
            phase: Some(phase),
            shape: Some(shape),
            tokens: ints_at(&world, 0),
            positions: ints_at(&world, 1),
            seed: ints_at(&world, 2),
            params: floats(&bound(&world, 3).1),
            top_k: ints_at(&world, 4),
            inputs: world.entry_inputs[entry]
                .iter()
                .map(|id| world.writes.get(id).cloned().unwrap_or_default())
                .collect(),
        };
        world.writes.clear();
        world.steps.push(record.clone());
        world.last = record;
        Ok(())
    }

    fn write(&mut self, dst: &FakeBuffer, bytes: &[u8]) -> Result<(), FakeError> {
        if dst.role == BufferRole::Input {
            self.world
                .borrow_mut()
                .writes
                .insert(dst.id, bytes.to_vec());
        }
        Ok(())
    }

    /// The only readback a step makes is its primary output (a zero-lane validation packet reads
    /// nothing), so every read is answered by the script.
    fn read(&mut self, _src: &FakeBuffer, out: &mut [u8]) -> Result<(), FakeError> {
        let mut world = self.world.borrow_mut();
        let record = world.last.clone();
        let bytes = (world.script)(&record, out.len());
        assert_eq!(
            bytes.len(),
            out.len(),
            "the script answers the requested length"
        );
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

/// `Engine<FakeDevice>` plus the bookkeeping that tells the fake which buffers an entry owns.
struct Counting {
    inner: Engine<FakeDevice>,
    world: Shared,
    log: TraceLog,
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
        self.inner.load_weights(store, weights)
    }

    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<ValidationOutputs>,
    ) -> Result<EntryId, ExecError> {
        let before = self.world.borrow().allocated.len();
        let id = self.inner.add_entry(exe, program)?;
        let mut world = self.world.borrow_mut();
        world.add_entry_calls += 1;
        let inputs: Vec<usize> = (before..world.allocated.len())
            .filter(|&i| world.allocated[i].0 == BufferRole::Input)
            .collect();
        world.entry_inputs.push(inputs);
        let traced = *self
            .log
            .lock()
            .unwrap()
            .last()
            .expect("the model traced this entry");
        world.entries.push(traced);
        Ok(id)
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
        if self.world.borrow().fail_remove {
            return Err(ExecError::Device(Box::new(poot_executor::DeviceError {
                backend: "fake",
                source: Box::new(FakeError),
            })));
        }
        self.inner.remove_entry(exe, entry)
    }

    fn stats(&self) -> ExecutorStats {
        self.inner.stats()
    }

    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        self.inner.unload(exe)
    }
}

/// The fixed logits the fake device answers with, per committed step: `rows[i]` answers the i-th
/// readback, the last row repeats. A request for the suffix's `[token, non_finite_index]` is answered
/// with the greedy argmax of the row (or its first non-finite index), what the device would have
/// computed; a request for the logits row is answered with the row itself.
fn logits_script(rows: Vec<Vec<f32>>) -> Script {
    let mut reads = 0usize;
    Box::new(move |_, len| {
        let row = &rows[reads.min(rows.len() - 1)];
        reads += 1;
        if len == 8 {
            let non_finite = row.iter().position(|v| !v.is_finite());
            let token = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 { (i, v) } else { best }
                })
                .0;
            let mut out = (token as i32).to_le_bytes().to_vec();
            out.extend((non_finite.map_or(-1, |i| i as i32)).to_le_bytes());
            out
        } else {
            assert_eq!(len, VOCAB * 4, "the logits row is [1, 1, vocab] f32");
            row.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    })
}

/// A row that favors `token`.
fn favoring(token: usize) -> Vec<f32> {
    let mut row = vec![0.0; VOCAB];
    row[token] = 5.0;
    row
}

struct Rig {
    driver: Driver,
    world: Shared,
}

pub(super) struct RigOptions {
    variant: Variant,
    granule: usize,
    chunk: usize,
    capacity: usize,
    max_entries: usize,
    max_retained_bytes: u64,
    charge: crate::driver::RetentionCharge,
    max_positions: usize,
}

impl Default for RigOptions {
    fn default() -> Self {
        Self {
            variant: Variant::Plain,
            granule: 1,
            chunk: 4,
            capacity: 64,
            max_entries: 64,
            max_retained_bytes: 1 << 40,
            charge: crate::driver::program_retention,
            max_positions: 256,
        }
    }
}

fn build_rig(options: RigOptions, script: Script) -> Result<Rig, DriverError> {
    let log: TraceLog = Arc::default();
    let family_log = log.clone();
    // The registry holds fn pointers; the fake family reads its trace log through a thread-local so the
    // build function stays a plain `fn`.
    TRACE_LOG.with(|slot| *slot.borrow_mut() = Some(family_log));
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: TINY,
            keys: &[(ConfigSource::HfModelType, "tiny")],
            build: |raw, store| {
                let log = TRACE_LOG.with(|slot| slot.borrow().clone().expect("rig sets the log"));
                build_tiny(raw, store, log)
            },
            fixture: || Fixture {
                config: tiny_config(1, 16, Variant::Plain),
                generation: None,
                store: WeightStore::default(),
            },
        })
        .unwrap();
    let config = tiny_config(options.granule, options.max_positions, options.variant);
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let handle =
        ModelHandle::from_checkpoint(&raw, WeightStore::default(), &registry, |_| Ok(tiny_text()))?;
    let world: Shared = Rc::new(RefCell::new(World {
        script,
        entries: Vec::new(),
        entry_inputs: Vec::new(),
        allocated: Vec::new(),
        writes: HashMap::new(),
        last: StepRecord::default(),
        steps: Vec::new(),
        weight_allocations: 0,
        add_entry_calls: 0,
        fail_remove: false,
    }));
    let device = FakeDevice {
        world: world.clone(),
        memory: poot_runtime_common::MemoryCounters::default(),
    };
    let executor = Counting {
        inner: Engine::new(device),
        world: world.clone(),
        log,
    };
    let compile = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let driver = Driver::new(
        Arc::new(handle),
        Box::new(executor),
        DriverOptions {
            prefill: compile,
            decode: compile,
            capacity: nz(options.capacity),
            prefill_chunk: nz(options.chunk),
            max_trace_tokens: nz(64),
            prepared: PreparedSetLimits {
                max_entries: nz(options.max_entries),
                max_retained_bytes: NonZeroU64::new(options.max_retained_bytes).unwrap(),
            },
            charge: options.charge,
        },
    )?;
    Ok(Rig { driver, world })
}

thread_local! {
    static TRACE_LOG: RefCell<Option<TraceLog>> = const { RefCell::new(None) };
}

fn rig_default(script: Script) -> Rig {
    build_rig(RigOptions::default(), script).unwrap()
}

/// The shapes of `generate`'s single contiguous row: every admitted prefill piece and decode at one
/// heads.
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

fn request(prompt: &[u32], max_new: usize, sampler: Sampler) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_vec(),
        max_new,
        sampler,
        stops: Vec::new(),
        ignore_eos: false,
    }
}

fn run(rig: &mut Rig, request: GenerateRequest) -> Result<Generation, DriverError> {
    rig.driver.generate(request, &mut |_: u32, _: &str| {
        GenerationControl::Continue(())
    })
}

/// The `(phase, tokens)` of every replayed step, in order.
fn step_shapes(rig: &Rig) -> Vec<(Phase, usize)> {
    rig.world
        .borrow()
        .steps
        .iter()
        .map(|s| (s.phase.unwrap(), s.tokens.len()))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Rows.
// ---------------------------------------------------------------------------------------------

/// SC-002 (EOS): the model's end-of-sequence token ends the run and is not part of the output; no
/// step runs after it. Mutation: treat EOS as an ordinary token (skip the `is_eos` break); the run
/// then continues to `max_new` and the tokens hold the EOS id.
#[test]
fn eos_ends_the_run_and_is_not_emitted() {
    let mut rig = rig_default(logits_script(vec![
        favoring(1),
        favoring(2),
        favoring(EOS as usize),
        favoring(3),
    ]));
    let out = run(&mut rig, request(&[5, 6], 10, Sampler::greedy())).unwrap();
    assert_eq!(out.tokens, [1, 2]);
    assert_eq!(out.finish, FinishReason::Eos);
    // 2 prefill tokens (one chunk), then one decode step per token that was fed back: 1 and 2.
    assert_eq!(rig.driver.stats().steps, 3);
}

/// `ignore_eos` keeps the run going through the end-of-sequence token, which is then an ordinary
/// token of the output (fixed-length measurements). Mutation: ignore the flag when building the stop
/// rule; the run ends at the EOS and the tokens are `[1]`.
#[test]
fn ignore_eos_runs_through_the_end_of_sequence_token() {
    let mut rig = rig_default(logits_script(vec![
        favoring(1),
        favoring(EOS as usize),
        favoring(2),
    ]));
    let mut req = request(&[5], 3, Sampler::greedy());
    req.ignore_eos = true;
    let out = run(&mut rig, req).unwrap();
    assert_eq!(out.tokens, [1, EOS, 2]);
    assert_eq!(out.finish, FinishReason::MaxTokens);
}

/// SC-002 (`max_new = 0`): zero tokens, and nothing is traced, compiled, added or dispatched.
/// Mutation: drop the early return; the loop then prefills and emits a token.
#[test]
fn max_new_zero_generates_nothing_and_touches_no_device() {
    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    let out = run(&mut rig, request(&[5, 6], 0, Sampler::greedy())).unwrap();
    assert!(out.tokens.is_empty());
    let stats = rig.driver.stats();
    assert_eq!(
        (stats.steps, stats.compiles, stats.entries_added),
        (0, 0, 0)
    );
    assert_eq!(rig.world.borrow().add_entry_calls, 0);
}

/// SC-002 (stop strings): generation ends at the token that completes a stop string, with no decode
/// step after it. Mutation: check the stop after the next decode step instead of before it; one more
/// step is dispatched and the step count assertion fails.
#[test]
fn a_stop_string_ends_generation_without_decoding_past_it() {
    let mut rig = rig_default(logits_script(vec![
        favoring(1),
        favoring(2),
        favoring(3),
        favoring(4),
    ]));
    let mut req = request(&[5], 10, Sampler::greedy());
    req.stops = vec!["b c".to_string()];
    let out = run(&mut rig, req).unwrap();
    assert_eq!(out.tokens, [1, 2, 3]);
    assert_eq!(out.finish, FinishReason::Stop);
    // one prefill token, then the decode steps that fed back 1 and 2 (3 completed the stop).
    assert_eq!(rig.driver.stats().steps, 3);
}

/// SC-002 (penalty history): the prompt and each generated token feed the penalty. A presence
/// penalty of 1.0 turns [5.0, 4.5, 4.2] over tokens 3, 4, 5 into 4 (3 is in the prompt) and then 5
/// (4 was just generated). Mutations: skip `seed_context` (the first token is 3) and skip `observe`
/// (the second token is 4).
#[test]
fn the_penalty_history_sees_the_prompt_and_each_generated_token() {
    let mut row = vec![0.0; VOCAB];
    row[3] = 5.0;
    row[4] = 4.5;
    row[5] = 4.2;
    let mut rig = rig_default(logits_script(vec![row]));
    let sampler = Sampler::greedy().with_penalties(1.0, 1.0, 0.0);
    let out = run(&mut rig, request(&[3], 2, sampler)).unwrap();
    assert_eq!(out.tokens, [4, 5]);
    assert_eq!(
        rig.driver.stats().host_picks,
        2,
        "a penalty is not a suffix request: both picks took the host path"
    );
}

/// SC-002 (guided exhaustion forces EOS): once a choice constraint has produced its whole sequence
/// it allows only EOS, so the run ends with it even though the logits favor a token the mask
/// forbids. Mutation: do not advance the constraint with each generated token (skip `observe`); the
/// constraint never completes and the run reaches `max_new` repeating the first token.
#[test]
fn an_exhausted_constraint_forces_eos() {
    let text = tiny_text();
    let constraint = text.build_choice_constraint(&["a b".to_string()]).unwrap();
    let mut rig = rig_default(logits_script(vec![favoring(9)]));
    let sampler = Sampler::greedy().with_constraint(constraint);
    let out = run(&mut rig, request(&[5], 10, sampler)).unwrap();
    assert_eq!(out.tokens, [1, 2]);
    assert_eq!(out.finish, FinishReason::Eos);
}

/// SC-003 (suffix path): a non-finite logit the device reports is a typed sampler error from the
/// driver, not a token. Mutation: ignore the readback's non-finite index in `read_tokens`.
#[test]
fn a_non_finite_logit_on_the_device_suffix_is_a_typed_error() {
    let mut row = favoring(1);
    row[5] = f32::NAN;
    let mut rig = rig_default(logits_script(vec![row]));
    let err = run(&mut rig, request(&[5], 3, Sampler::greedy())).unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::Sampler(SamplerFault::NonFiniteLogit {
                index: 5,
                value: None
            })
        ),
        "{err:?}"
    );
}

/// SC-003 (host path): a NaN in the logits row a host-sampled request reads back is a typed error with
/// the offending value. Mutation: clamp non-finite logits to zero when reading them back.
#[test]
fn a_non_finite_logit_on_the_host_path_is_a_typed_error() {
    let mut row = favoring(1);
    row[2] = f32::NAN;
    let mut rig = rig_default(logits_script(vec![row]));
    let sampler = Sampler::greedy().with_penalties(1.0, 1.0, 0.0);
    let err = run(&mut rig, request(&[5], 3, sampler)).unwrap_err();
    match err {
        DriverError::Sampler(SamplerFault::NonFiniteLogit { index: 2, value }) => {
            assert!(value.is_some_and(f32::is_nan), "{value:?}");
        }
        other => panic!("{other:?}"),
    }
}

/// SC-005: with the capacity at 8x the live length, every step still binds `Slot::Pos` to the live
/// absolute positions, and capacity sizes only the step's shape. Mutation: fill `Pos` from the
/// capacity (`capacity - 1`); the positions assertion fails.
#[test]
fn pos_is_the_live_position_whatever_the_capacity() {
    let options = RigOptions {
        capacity: 64,
        ..RigOptions::default()
    };
    let mut rig = build_rig(options, logits_script(vec![favoring(1)])).unwrap();
    run(&mut rig, request(&[5, 6, 7], 3, Sampler::greedy())).unwrap();
    let world = rig.world.borrow();
    let positions: Vec<Vec<i32>> = world.steps.iter().map(|s| s.positions.clone()).collect();
    // 3 prompt tokens plan as [2, 1] at chunk 4, then two decode steps.
    assert_eq!(positions, [vec![0, 1], vec![2], vec![3], vec![4]]);
    for step in &world.steps {
        assert_eq!(
            step.shape.unwrap().capacity.get(),
            64,
            "capacity sizes buffers"
        );
    }
}

/// SC-006: a guided request takes the host sampling path and gets a token its mask allows, while an
/// ordinary request never takes it. Mutation: route every request through the host path (the head is
/// always `Logits`); the ordinary request's host-pick count is no longer zero.
#[test]
fn the_host_path_serves_only_what_the_suffix_cannot_express() {
    let text = tiny_text();
    let constraint = text.build_choice_constraint(&["c".to_string()]).unwrap();
    let mut rig = rig_default(logits_script(vec![favoring(9)]));
    let guided = Sampler::greedy().with_constraint(constraint);
    let out = run(&mut rig, request(&[5], 1, guided)).unwrap();
    assert_eq!(
        out.tokens,
        [3],
        "the mask allows only `c` (id 3), the logits favor 9"
    );
    assert!(rig.driver.stats().host_picks > 0);

    let mut rig = rig_default(logits_script(vec![favoring(9)]));
    let out = run(&mut rig, request(&[5], 1, Sampler::greedy())).unwrap();
    assert_eq!(out.tokens, [9]);
    assert_eq!(rig.driver.stats().host_picks, 0);
}

/// SC-001 (host side): a sampled request binds the sampler slots the suffix declares from its own
/// `Sampler`: one seed per committed step, in the stream order a host draw would use, and a
/// placeholder (which does not advance the stream) on each prefill step whose output is discarded.
/// Mutation: build every row with `SuffixRows::push`; the discarded prefill steps consume seeds and
/// every committed seed shifts.
#[test]
fn a_sampled_request_binds_its_sampler_slots_and_discarded_steps_leave_the_stream_alone() {
    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    run(
        &mut rig,
        request(&[1; 11], 3, Sampler::new(1.0, 0, 1.0, 42)),
    )
    .unwrap();
    let mut reference = Sampler::new(1.0, 0, 1.0, 42);
    let drawn: Vec<i32> = (0..3)
        .map(|_| reference.next_device_seed() as i32)
        .collect();
    let world = rig.world.borrow();
    let seeds: Vec<Vec<i32>> = world.steps.iter().map(|s| s.seed.clone()).collect();
    assert_eq!(
        seeds,
        [
            vec![0],
            vec![0],
            vec![0],
            vec![drawn[0]],
            vec![drawn[1]],
            vec![drawn[2]]
        ]
    );
    for step in &world.steps {
        assert_eq!(
            step.params,
            [1.0, f32::NEG_INFINITY, 1.0],
            "[inv_temp, floor_offset, noise]"
        );
        assert!(
            step.top_k.is_empty(),
            "a plain temperature draw declares no top_k slot"
        );
    }
    drop(world);

    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    run(&mut rig, request(&[1; 2], 1, Sampler::new(1.0, 5, 1.0, 7))).unwrap();
    let world = rig.world.borrow();
    assert_eq!(world.steps[0].top_k, [5]);
}

/// The token count of every replayed step, in order.
fn piece_sizes(rig: &Rig) -> Vec<usize> {
    step_shapes(rig).into_iter().map(|(_, n)| n).collect()
}

/// SC-007: chunk 4 / granule 1 plans an 11-token prompt as [4, 4, 2, 1] through the real loop, and
/// every tail 0..3 stays in the admitted set {1, 2, 4}. Mutation: plan the tail as one piece of 3
/// ([4, 4, 3]); the exact-sequence assertion fails.
#[test]
fn the_chunk_plan_runs_as_exactly_its_pieces() {
    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    run(&mut rig, request(&[1; 11], 1, Sampler::greedy())).unwrap();
    assert_eq!(piece_sizes(&rig), [4, 4, 2, 1]);

    let mut seen = BTreeSet::new();
    for tail in 0..4 {
        let mut rig = rig_default(logits_script(vec![favoring(1)]));
        run(&mut rig, request(&vec![1; 8 + tail], 1, Sampler::greedy())).unwrap();
        seen.extend(piece_sizes(&rig));
    }
    assert_eq!(seen, BTreeSet::from([1, 2, 4]));
}

/// SC-007 (granule 2): a 5-token prompt is [4, decode(1)] and a 7-token prompt [4, 2, decode(1)];
/// the tail below one granule runs as decode steps and the model never sees a misaligned prefill.
#[test]
fn a_tail_below_the_granule_runs_as_decode_steps() {
    let options = || RigOptions {
        granule: 2,
        ..RigOptions::default()
    };
    let mut rig = build_rig(options(), logits_script(vec![favoring(1)])).unwrap();
    run(&mut rig, request(&[1; 5], 1, Sampler::greedy())).unwrap();
    assert_eq!(step_shapes(&rig), [(Phase::Prefill, 4), (Phase::Decode, 1)]);
    let mut rig = build_rig(options(), logits_script(vec![favoring(1)])).unwrap();
    run(&mut rig, request(&[1; 7], 1, Sampler::greedy())).unwrap();
    assert_eq!(
        step_shapes(&rig),
        [(Phase::Prefill, 4), (Phase::Prefill, 2), (Phase::Decode, 1)]
    );
}

/// SC-009: after `prepare` warms the admitted set at one capacity and head, requests drawn from it
/// compile and add nothing, however their prompts, tokens and seeds vary. Mutations: key the entry
/// cache on the live position (every new position compiles), and skip the decode entry in `prepare`
/// (the first request compiles it).
#[test]
fn a_prepared_set_replays_without_compiling() {
    let options = RigOptions {
        capacity: 32,
        ..RigOptions::default()
    };
    let mut rig = build_rig(options, logits_script(vec![favoring(1), favoring(2)])).unwrap();
    rig.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    let warmed = rig.driver.stats();
    let entries = rig.driver.prepared_entries();
    for (prompt, max_new) in [(vec![1u32; 4], 3usize), (vec![2; 7], 4), (vec![3; 11], 5)] {
        run(&mut rig, request(&prompt, max_new, Sampler::greedy())).unwrap();
    }
    let after = rig.driver.stats();
    assert_eq!(
        (after.compiles, after.entries_added),
        (warmed.compiles, warmed.entries_added)
    );
    assert_eq!(
        rig.world.borrow().add_entry_calls as u64,
        warmed.entries_added
    );
    assert_eq!(rig.driver.prepared_entries(), entries);
    assert_eq!(
        entries, 3,
        "prefill 4 and 2, and the one-token step prefill 1 and decode share"
    );
}

/// SC-010 (entry limit): with `max_entries = 2` and a fixed charge of 32 bytes per entry against 96,
/// the first two candidates enter, the third is a typed entry refusal, and both earlier entries stay
/// usable (a request that needs only them replays with no compile). Mutations: omit the entry check
/// (the third enters); drop the prepared entries when refusing (the replay recompiles).
#[test]
fn the_entry_limit_refuses_the_third_and_keeps_the_first_two() {
    let options = RigOptions {
        max_entries: 2,
        max_retained_bytes: 96,
        charge: |_| Retention::private(32),
        capacity: 8,
        ..RigOptions::default()
    };
    let mut rig = build_rig(options, logits_script(vec![favoring(1)])).unwrap();
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
    // the first admitted candidate is prefill(4): a 4-token prompt replays it.
    let before = rig.driver.stats();
    let steps_before = rig.world.borrow().steps.len();
    let out = rig.driver.generate(
        request(&[1; 4], 1, Sampler::greedy()),
        &mut |_: u32, _: &str| GenerationControl::Continue(()),
    );
    assert_eq!(out.unwrap().tokens, [1]);
    let after = rig.driver.stats();
    assert_eq!(
        after.compiles, before.compiles,
        "a prepared entry replays without compiling"
    );
    assert_eq!(rig.world.borrow().steps.len(), steps_before + 1);
}

/// SC-010 (byte limit): with room for four entries but 63 bytes at 32 per entry, the second candidate
/// is refused on bytes and the counters do not move. Mutation: omit the byte check; the second enters.
#[test]
fn the_byte_limit_refuses_the_second_without_moving_the_counters() {
    let options = RigOptions {
        max_entries: 4,
        max_retained_bytes: 63,
        charge: |_| Retention::private(32),
        ..RigOptions::default()
    };
    let mut rig = build_rig(options, logits_script(vec![favoring(1)])).unwrap();
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
    assert_eq!(rig.world.borrow().add_entry_calls, 1);
}

/// A chunk the model's granule does not divide, and one above the trace limit, are typed option
/// errors before any entry exists.
#[test]
fn options_the_model_cannot_honor_are_typed_errors() {
    let misaligned = build_rig(
        RigOptions {
            granule: 2,
            chunk: 5,
            ..RigOptions::default()
        },
        logits_script(vec![favoring(1)]),
    );
    assert!(matches!(
        misaligned,
        Err(DriverError::InvalidOptions(
            InvalidOptions::ChunkNotGranuleAligned {
                chunk: 5,
                granule: 2
            }
        ))
    ));
    let too_big = build_rig(
        RigOptions {
            capacity: 512,
            ..RigOptions::default()
        },
        logits_script(vec![favoring(1)]),
    );
    assert!(matches!(
        too_big,
        Err(DriverError::InvalidOptions(
            InvalidOptions::CapacityAboveMax {
                capacity: 512,
                max_positions: 256
            }
        ))
    ));
    let over = build_rig(
        RigOptions {
            chunk: 128,
            ..RigOptions::default()
        },
        logits_script(vec![favoring(1)]),
    );
    assert!(matches!(
        over,
        Err(DriverError::InvalidOptions(
            InvalidOptions::ChunkAboveTraceLimit {
                chunk: 128,
                max_tokens: 64
            }
        ))
    ));
}

#[test]
fn an_invalid_request_is_a_typed_error_that_dispatches_nothing() {
    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    let err = run(&mut rig, request(&[], 3, Sampler::greedy())).unwrap_err();
    assert!(matches!(
        err,
        DriverError::InvalidRequest(InvalidRequest::EmptyPrompt)
    ));
    let err = run(&mut rig, request(&[1; 60], 10, Sampler::greedy())).unwrap_err();
    assert!(matches!(
        err,
        DriverError::InvalidRequest(InvalidRequest::Capacity {
            prompt: 60,
            max_new: 10,
            max: 64
        })
    ));
    assert_eq!(rig.driver.stats().steps, 0);
}

/// A family the registry does not hold is a typed `Unsupported::Registry`, not a panic or a string.
#[test]
fn an_unregistered_family_is_a_typed_refusal() {
    let config = serde_json::json!({"model_type": "no-such-family"});
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let err =
        ModelHandle::from_checkpoint(&raw, WeightStore::default(), &Registry::empty(), |_| {
            Ok(tiny_text())
        })
        .unwrap_err();
    assert!(
        matches!(err, DriverError::Unsupported(Unsupported::Registry(_))),
        "{err:?}"
    );
}

/// A sink that stops ends the run after the token it consumed, with no further step.
#[test]
fn a_sink_that_stops_ends_the_run() {
    let mut rig = rig_default(logits_script(vec![favoring(1), favoring(2), favoring(3)]));
    let mut seen = Vec::new();
    let out = rig
        .driver
        .generate(
            request(&[5], 10, Sampler::greedy()),
            &mut |id: u32, _: &str| {
                seen.push(id);
                if seen.len() == 2 {
                    GenerationControl::Break(())
                } else {
                    GenerationControl::Continue(())
                }
            },
        )
        .unwrap();
    assert_eq!(out.tokens, [1, 2]);
    assert_eq!(out.finish, FinishReason::Cancelled);
    assert_eq!(rig.driver.stats().steps, 2);
}

// ---------------------------------------------------------------------------------------------
// The production loader: `ModelHandle::load` over an HF directory and a GGUF file, model-free.
// ---------------------------------------------------------------------------------------------

/// A safetensors file holding `store`'s dense entries (the HF layout `load_weight_store` reads).
fn write_safetensors(path: &std::path::Path, store: &WeightStore) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, entry) in store.iter() {
        let poot_quant::weights::WeightEntry::Dense(dense) = entry else {
            panic!("the fixture store is dense")
        };
        let start = data.len();
        data.extend_from_slice(dense.bytes().as_slice());
        header.insert(
            key.to_string(),
            serde_json::json!({
                "dtype": match dense.dtype() {
                    DType::BF16 => "BF16",
                    DType::F32 => "F32",
                    other => panic!("{other:?}"),
                },
                "shape": dense.shape(),
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

/// `ModelHandle::load` of an HF directory builds the model from `config.json`, takes the end-of-sequence
/// set from the union of `config.json` and `generation_config.json`, reads the weights as stored and
/// the tokenizer from `tokenizer.json`. The checkpoint is the registry's own qwen2 fixture written out.
#[test]
fn a_hf_directory_loads_through_the_registry() {
    let registry = Registry::builtin().unwrap();
    let fixture = (registry.entries()[0].fixture)();
    let dir = poot_test_util::unique_temp_path("driver_hf_checkpoint");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::json!({"eos_token_id": [46, 47]}).to_string(),
    )
    .unwrap();
    write_safetensors(&dir.join("model.safetensors"), &fixture.store);
    tiny_text()
        .tokenizer
        .save(dir.join("tokenizer.json"), false)
        .unwrap();

    let handle = ModelHandle::load(&dir, &registry).unwrap();
    let config = handle.config();
    assert_eq!(config.family.as_str(), "qwen2");
    assert_eq!(config.vocab, 48);
    assert_eq!(
        config.eos,
        BTreeSet::from([46, 47]),
        "the union of both configs"
    );
    assert_eq!(handle.store(), &fixture.store, "weights exactly as stored");
    assert_eq!(handle.text().encode("a b").unwrap(), [1, 2]);

    let missing = ModelHandle::load(&dir.join("absent"), &registry).unwrap_err();
    assert!(matches!(missing, DriverError::Load(_)), "{missing:?}");
}

/// `ModelHandle::load` of a GGUF file: the family comes from `general.architecture`, the weights from
/// `read_gguf`, the end-of-sequence id and tokenizer from the file's own metadata.
#[test]
fn a_gguf_file_loads_through_the_registry() {
    use poot_load::gguf::{GgufValue, write_gguf};
    let (h, inter, vocab) = (8usize, 16usize, 8usize);
    let mut tensors = Vec::new();
    for (name, shape) in [
        ("token_embd.weight", vec![h, vocab]),
        ("output.weight", vec![h, vocab]),
        ("output_norm.weight", vec![h]),
        ("blk.0.attn_q.weight", vec![h, h]),
        ("blk.0.attn_k.weight", vec![h, h]),
        ("blk.0.attn_v.weight", vec![h, h]),
        ("blk.0.attn_output.weight", vec![h, h]),
        ("blk.0.attn_q.bias", vec![h]),
        ("blk.0.attn_k.bias", vec![h]),
        ("blk.0.attn_v.bias", vec![h]),
        ("blk.0.attn_norm.weight", vec![h]),
        ("blk.0.ffn_norm.weight", vec![h]),
        ("blk.0.ffn_gate.weight", vec![h, inter]),
        ("blk.0.ffn_up.weight", vec![h, inter]),
        ("blk.0.ffn_down.weight", vec![inter, h]),
    ] {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| (((i % 7) as f32) * 0.1 - 0.3).to_le_bytes())
            .collect();
        tensors.push((
            name,
            shape.into_iter().map(|d| d as u64).collect(),
            0,
            bytes,
        ));
    }
    let kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(h as u32)),
        ("qwen2.block_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(1)),
        ("qwen2.feed_forward_length", GgufValue::U32(inter as u32)),
        ("qwen2.context_length", GgufValue::U32(64)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(7)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "c", "d", "e", "f", "g", "ab"]
                    .into_iter()
                    .map(|s| GgufValue::Str(s.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let path = poot_test_util::unique_temp_path("driver_gguf_checkpoint.gguf");
    std::fs::write(&path, write_gguf(&kvs, &tensors)).unwrap();

    let handle = ModelHandle::load(&path, &Registry::builtin().unwrap()).unwrap();
    assert_eq!(handle.config().family.as_str(), "qwen2");
    assert_eq!(handle.config().eos, BTreeSet::from([7]));
    assert_eq!(handle.config().max_positions, 64);
    assert_eq!(
        handle.text().encode("ab").unwrap(),
        [7],
        "the file's own merges"
    );
}

#[path = "serving_tests.rs"]
mod serving_tests;
