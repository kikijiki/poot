//! The add-a-model standing test (ADR-0104 rule 3, "adding a model touches only the model").
//!
//! The fixture family ([`family`]) is defined in this test target, outside every production crate, and
//! reaches the product only through its public surface: `Registry::register`, `ModelHandle::load`, the
//! shared components and `Driver::generate`. If a production edit were needed to serve it, this file
//! could not be written; the rows below fail the other way, when production starts to treat a family
//! it was not compiled with differently from a built-in one.
//!
//! - Host half (model-free): the family registers, loads from a written checkpoint, compiles Prefill and
//!   Decode for the wgpu, ROCm and PTX reference targets, and greedy-decodes the pinned tokens on the
//!   CPU oracle. A driver over a fake device prepares and serves it under the same typed limits as a
//!   built-in family.
//! - Device half: `Driver::generate` on a real device equals the oracle's tokens.

mod family;

use std::cell::Cell;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    BufferRole, Device, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, StateScope, StepInputs, StepOutputs, WeightSource,
};
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::{Slot, SlotKey, ValidationOutputs};
use poot_graph_plan::{
    CompileLimits, CompileOptions, StagedProgram, Submission, Target, TargetSet, WeightFormats,
    bind_packed_weights, compile,
};
use poot_llm::driver::error::{DriverError, InvalidOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Head, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, ServingShapes, Warm, open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use poot_models::registry::{Fixture, Registry, RegistryError};
use poot_quant::weights::WeightStore;
use poot_runtime_common::DeviceBackend;
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_tensor::HostTensor;
use poot_test_util::device_caps::default_caps_for;
use poot_test_util::weight_map_oracle::eval_mapped;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

use family::{ENTRY, FAMILY, VOCAB};

/// The prompt, as token ids: the family's tokenizer is only needed because the loader reads one.
const PROMPT: [u32; 6] = [3, 17, 8, 29, 5, 21];
/// How many tokens every row generates.
const NEW_TOKENS: usize = 8;
/// The KV positions of every step and driver in this file.
const CAPACITY: usize = 32;
/// The greedy tokens of [`PROMPT`] on the CPU oracle, pinned: the fixture's weights are fixed bytes,
/// so a different token here is a different model.
const PINNED: [u32; NEW_TOKENS] = [7, 39, 21, 18, 16, 29, 29, 26];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// The checkpoint: config, weights and a tokenizer, written the way a downloaded one is laid out.
// ---------------------------------------------------------------------------------------------

/// A word-level tokenizer over the fixture's 48-word vocabulary (`a`..`z`, `0`..`9`, `p0`..`p11`).
fn tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..VOCAB - words.len()).map(|i| format!("p{i}")));
    assert_eq!(words.len(), VOCAB);
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

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_test_util::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

/// The built-in families plus the fixture family, registered from outside the production crates.
fn registry() -> Registry {
    let mut registry = Registry::builtin().unwrap();
    registry.register(ENTRY).unwrap();
    registry
}

fn load(dir: &Path) -> ModelHandle {
    ModelHandle::load(dir, &registry(), &LoadOptions::default()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// The CPU oracle: the family's own traced graphs evaluated by `poot_eval`, one state across steps.
// ---------------------------------------------------------------------------------------------

fn shape(tokens: usize) -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: nz(tokens),
        capacity: nz(CAPACITY),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    }
}

/// One oracle step: `tokens` at positions `start..`, over the carried `state` (zeros on the first
/// step). Returns the last token's logits and leaves the new state in `state`.
fn oracle_step(
    model: &dyn Model,
    store: &WeightStore,
    phase: Phase,
    tokens: &[u32],
    start: usize,
    state: &mut Vec<HostTensor>,
) -> Vec<f32> {
    let n = tokens.len();
    let graph = model.trace(phase, shape(n)).unwrap();
    if state.is_empty() {
        *state = graph
            .state
            .iter()
            .map(|&(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
            .collect();
    }
    let slots = [
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![1, n], tokens.iter().map(|&t| t as i32).collect()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![1, n], (start..start + n).map(|p| p as i32).collect()),
        ),
    ];
    let (logits, next) = eval_mapped(&graph, store, model.weights(), &slots, state);
    *state = next;
    logits.as_f32().expect("f32 logits").to_vec()
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

/// `new` greedy tokens after `prompt`: prefill once, then one decode step per token. Each token's
/// lead over the runner-up is returned beside it, so a caller can tell a decided token from a tie.
fn oracle_tokens(
    model: &dyn Model,
    store: &WeightStore,
    prompt: &[u32],
    new: usize,
    decode_position: impl Fn(usize) -> usize,
) -> Vec<(u32, f32)> {
    let mut state = Vec::new();
    let mut out = vec![greedy(&oracle_step(
        model,
        store,
        Phase::Prefill,
        prompt,
        0,
        &mut state,
    ))];
    for i in 1..new {
        let last = out[i - 1].0;
        out.push(greedy(&oracle_step(
            model,
            store,
            Phase::Decode,
            &[last],
            decode_position(prompt.len() + i - 1),
            &mut state,
        )));
    }
    out
}

fn fixture_model(fixture: &Fixture) -> Box<dyn Model> {
    registry().build(&fixture.raw(), &fixture.store).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Host half.
// ---------------------------------------------------------------------------------------------

/// SC-001: the family registers through the public `Registry::register`, loads from a written
/// checkpoint through `ModelHandle::load`, and its Prefill and Decode graphs compile for the wgpu,
/// ROCm and PTX reference targets. Before registration the same checkpoint is refused with the
/// typed unregistered-family error. Mutation: make `Registry::register` crate-private; this file
/// fails to compile (E0624).
#[test]
fn the_fixture_family_registers_loads_and_compiles_for_every_reference_target() {
    let fixture = family::fixture();
    let dir = poot_test_util::unique_temp_path("neoxish_checkpoint");
    write_checkpoint(&dir, &fixture);

    let raw = fixture.raw();
    assert!(
        matches!(
            Registry::builtin().unwrap().resolve(&raw),
            Err(RegistryError::Unregistered { key, .. }) if key == "poot_fixture_neoxish"
        ),
        "the built-in registry does not know the fixture family"
    );
    let handle = load(&dir);
    assert_eq!(handle.config().family, FAMILY);
    assert_eq!(handle.config().vocab, VOCAB);
    assert_eq!(handle.store().len(), fixture.store.len());

    let model = registry().build(&raw, handle.store()).unwrap();
    let targets = [
        (
            "wgpu",
            Target {
                backend: Backend::SpirvVulkan,
                caps: default_caps_for(Backend::SpirvVulkan),
            },
        ),
        (
            "ROCm",
            Target {
                backend: Backend::AmdGcn(AmdArch::gfx1151()),
                caps: default_caps_for(Backend::AmdGcn(AmdArch::gfx1151())),
            },
        ),
        (
            "PTX",
            Target {
                backend: Backend::Nvptx,
                caps: default_caps_for(Backend::Nvptx),
            },
        ),
    ];
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    let formats = WeightFormats::from_weight_map(model.weights());
    for (name, target) in &targets {
        for (phase, tokens) in [(Phase::Prefill, 4), (Phase::Decode, 1)] {
            let graph = model.trace(phase, shape(tokens)).unwrap();
            let graph = bind_packed_weights(&graph, &formats).unwrap();
            compile(&graph, target, &options)
                .unwrap_or_else(|e| panic!("{phase:?} does not compile for {name}: {e}"));
        }
    }
}

/// SC-001: 8 greedy tokens of the oracle equal the pinned list. The oracle is the family's own
/// graphs evaluated on the CPU, prefill then eight decode-continued steps, and every token leads
/// its runner-up by a margin no device rounding closes. A second, state-free derivation (prefill
/// over the whole prefix) gives every token again, so the pin is not a number the carried state
/// made up. Mutation: change the family's RoPE width (`rotary_pct`); the tokens differ.
#[test]
fn the_oracle_decodes_the_pinned_tokens() {
    let fixture = family::fixture();
    let model = fixture_model(&fixture);
    let tokens = oracle_tokens(&*model, &fixture.store, &PROMPT, NEW_TOKENS, |p| p);
    for (i, &(token, lead)) in tokens.iter().enumerate() {
        assert!(lead > 0.05, "token {i} ({token}) leads by only {lead}");
    }
    let tokens: Vec<u32> = tokens.into_iter().map(|(token, _)| token).collect();
    assert_eq!(tokens, PINNED, "the oracle's greedy tokens");

    let mut prefix = PROMPT.to_vec();
    for (i, &token) in tokens.iter().enumerate() {
        let mut fresh = Vec::new();
        let logits = oracle_step(
            &*model,
            &fixture.store,
            Phase::Prefill,
            &prefix,
            0,
            &mut fresh,
        );
        assert_eq!(
            greedy(&logits).0,
            token,
            "token {i}: a stateless prefill over the prefix disagrees with the carried state"
        );
        prefix.push(token);
    }
}

/// The pin can see a wrong position: an oracle whose decode steps read position 0 every time (the
/// SC-002 mutation, applied to the oracle) decodes different tokens. Without this the device row's
/// mutation could be inert at this fixture.
#[test]
fn the_pinned_tokens_depend_on_the_decode_position() {
    let fixture = family::fixture();
    let model = fixture_model(&fixture);
    let at_zero: Vec<u32> = oracle_tokens(&*model, &fixture.store, &PROMPT, NEW_TOKENS, |_| 0)
        .into_iter()
        .map(|(token, _)| token)
        .collect();
    assert_ne!(
        at_zero, PINNED,
        "position 0 at every decode step decodes the pinned tokens"
    );
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
    /// Every row picks token 1: the sampling suffix answers `[token, non_finite_index]`, the logits
    /// head a row favouring token 1.
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

/// The fake's engine, counting the executables a driver loads: a refused driver must load none.
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

fn driver_options(chunk: usize, max_trace_tokens: usize) -> DriverOptions {
    let compile = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    DriverOptions {
        prefill: compile,
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(chunk),
        max_trace_tokens: nz(max_trace_tokens),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
    }
}

/// A driver over the fixture family on the fake device, and the number of executables it loaded.
fn fake_driver(
    options: DriverOptions,
) -> (
    Result<Driver, DriverError>,
    Rc<Cell<usize>>,
    poot_test_util::UniqueTempPath,
) {
    let dir = poot_test_util::unique_temp_path("neoxish_driver_checkpoint");
    write_checkpoint(&dir, &family::fixture());
    let loads = Rc::new(Cell::new(0));
    let executor = Counting {
        inner: Engine::new(FakeDevice),
        loads: loads.clone(),
    };
    let driver = Driver::new(Arc::new(load(&dir)), Box::new(executor), options);
    (driver, loads, dir)
}

const GREEDY: Head = Head::GREEDY;
const SAMPLED: Head = Head::Sample(SampleRule::Gumbel);

fn warm_admitted(driver: &mut Driver, heads: &[Head]) -> Result<(), DriverError> {
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

fn request(prompt_len: usize, sampler: Sampler) -> GenerateRequest {
    GenerateRequest {
        prompt: (0..prompt_len as u32).map(|t| t % 40 + 2).collect(),
        max_new: 3,
        sampler,
        stops: Vec::new(),
        ignore_eos: false,
    }
}

/// SC-003: the driver prepares a family that is not a `Family` variant exactly as it prepares a
/// built-in one: every admitted entry (the step extents 4, 2 and 1, per head) compiles through the common planner and is added to the executor. Mutation: add a family match to
/// `Driver::prepared` (`match family { "qwen2" => .., _ => refuse }`); the fixture family is refused
/// and this row is red.
#[test]
fn the_driver_prepares_the_fixture_family_with_no_family_branch() {
    let (driver, _, _dir) = fake_driver(driver_options(4, 64));
    let mut driver = driver.unwrap();
    warm_admitted(&mut driver, &[GREEDY, SAMPLED]).unwrap_or_else(|e| panic!("{e}"));
    // chunk 4 over granule 1: the step extents 4, 2 and 1 (the one-token step serves decode and a
    // one-token prefill tail), for two heads.
    assert_eq!(driver.prepared_entries(), 2 * 3);
    let stats = driver.stats();
    assert_eq!(stats.compiles, 6);
    assert_eq!(stats.entries_added, 6);
}

// SC-004 max_eqns row (a family whose third equation the common Builder refuses): deferred to POOT-733, which
// adds TraceLimits to the Builder; there is no equation budget to drive yet and a test-only one is not allowed.

/// SC-004 (trace limit): a 6-token chunk is aligned to the family's granule, and a driver whose
/// trace limit is 4 tokens refuses it with the typed error before it loads any executable; the same
/// chunk under an adequate limit constructs and loads one. Mutation: apply `max_trace_tokens` only
/// to the built-in family names (in `Driver::new`); the refusal row is red.
#[test]
fn a_chunk_above_the_trace_limit_is_refused_before_construction_for_an_external_family() {
    let granule = {
        let fixture = family::fixture();
        fixture_model(&fixture).config().prefill_granule.get()
    };
    assert_eq!(
        6 % granule,
        0,
        "the chunk is granule-aligned: only the limit refuses it"
    );

    let (refused, loads, _dir) = fake_driver(driver_options(6, 4));
    assert!(
        matches!(
            refused,
            Err(DriverError::InvalidOptions(
                InvalidOptions::ChunkAboveTraceLimit {
                    chunk: 6,
                    max_tokens: 4
                }
            ))
        ),
        "{:?}",
        refused.err()
    );
    assert_eq!(loads.get(), 0, "a refused driver loads no executable");

    let (admitted, loads, _dir) = fake_driver(driver_options(6, 6));
    admitted.unwrap();
    assert_eq!(loads.get(), 1);
}

/// SC-004 (warm reuse): once the admitted set is prepared, generating with prompts of other lengths
/// (other positions) and other sampler seeds compiles nothing and adds no entry: positions and seeds
/// are bound inputs, never part of an entry's key.
#[test]
fn replaying_the_prepared_set_compiles_nothing_as_positions_and_seeds_change() {
    let (driver, _, _dir) = fake_driver(driver_options(4, 64));
    let mut driver = driver.unwrap();
    warm_admitted(&mut driver, &[GREEDY, SAMPLED]).unwrap();
    let warmed = driver.stats();
    let entries = driver.prepared_entries();
    assert_eq!(warmed.compiles as usize, entries);

    let mut sink = |_: u32, _: &str| GenerationControl::Continue(());
    for (prompt_len, seed) in [(1, 11), (2, 12), (3, 13), (7, 14), (9, 15), (13, 16)] {
        for sampler in [Sampler::greedy(), Sampler::new(0.8, 0, 1.0, seed)] {
            driver
                .generate(request(prompt_len, sampler), &mut sink)
                .unwrap_or_else(|e| panic!("a prompt of {prompt_len} tokens: {e}"));
        }
    }
    let after = driver.stats();
    assert_eq!(
        after.compiles, warmed.compiles,
        "replaying the prepared set compiled"
    );
    assert_eq!(after.entries_added, warmed.entries_added);
    assert_eq!(driver.prepared_entries(), entries);
    assert!(
        after.steps > warmed.steps,
        "the rows ran steps ({} before, {} after)",
        warmed.steps,
        after.steps
    );
    assert!(
        driver.executor_stats().replays > 0,
        "the steps replayed recordings"
    );
}

// ---------------------------------------------------------------------------------------------
// Device half.
// ---------------------------------------------------------------------------------------------

/// `Driver::generate` on `choice`'s device over the loaded fixture checkpoint: the tokens equal the
/// oracle's, which are the pinned list. The device skips cleanly when absent (and fails when the lane
/// requires it).
fn assert_generate_matches_the_oracle(choice: BackendChoice, backend: DeviceBackend) {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(backend, open_executor(choice))
    else {
        return;
    };
    let fixture = family::fixture();
    let dir = poot_test_util::unique_temp_path("neoxish_device_checkpoint");
    write_checkpoint(&dir, &fixture);
    let handle = Arc::new(load(&dir));
    let oracle: Vec<u32> = oracle_tokens(
        &*fixture_model(&fixture),
        &fixture.store,
        &PROMPT,
        NEW_TOKENS,
        |p| p,
    )
    .into_iter()
    .map(|(token, _)| token)
    .collect();
    assert_eq!(oracle, PINNED);

    let mut driver = Driver::new(handle, executor, driver_options(8, 64)).unwrap();
    let generation = driver
        .generate(
            GenerateRequest {
                prompt: PROMPT.to_vec(),
                max_new: NEW_TOKENS,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        generation.tokens, oracle,
        "the device's greedy tokens differ from the oracle's"
    );
}

/// SC-002 on wgpu. Mutation: bind position 0 at every step of the driver's decode input; the
/// device's tokens diverge from the oracle's.
#[test]
fn generate_equals_the_oracle_tokens_wgpu() {
    assert_generate_matches_the_oracle(BackendChoice::Wgpu, DeviceBackend::Wgpu);
}

/// SC-002 on ROCm (the `rocm` feature; without it the device does not open and the row skips).
#[test]
fn generate_equals_the_oracle_tokens_rocm() {
    assert_generate_matches_the_oracle(BackendChoice::Rocm, DeviceBackend::Rocm);
}

/// SC-002 on PTX, in the M5 family pod batch.
#[test]
fn generate_equals_the_oracle_tokens_ptx() {
    assert_generate_matches_the_oracle(BackendChoice::Ptx, DeviceBackend::Ptx);
}
