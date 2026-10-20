//! POOT-631: measurement and design evidence for rolling recurrent state back, so a GDN or Mamba2
//! model can verify speculative drafts (the rollback itself is POOT-647's, covered end to end by
//! `speculate_recurrent.rs`); this binary measures what a per-row state snapshot and restore cost, over
//! the same row-gather machinery the beam reorder uses, and proves on the CPU oracle that
//! snapshot-then-restore undoes folded draft tokens.
//!
//! Families: Qwen3-Next (Gated DeltaNet layers plus gated attention) and Nemotron-H (Mamba2 layers
//! plus attention and MLP), both through their shipped synthetic fixtures. Both are hybrids: only the
//! `StateRole::Recurrent` pairs need a snapshot, the positional KV pairs do not (a rejected token's
//! write is overwritten, never read past its position).
//!
//! The model-free rows run everywhere; the timing row needs a device and prints, never gates, speed.

mod common;

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use common::{step, token_pos, zero_state};
use poot_executor::{DetailedTiming, Device, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::{Builder, Graph, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::compile_staged,
};
use poot_llm::driver::capacity::Planner;
use poot_llm::driver::error::DriverError;
use poot_llm::driver::placement::Placement;
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Head, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, ServingShapes, Warm, open_timed_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::{FamilyKey, LogitRows, StepSpec};
use poot_models::registry::{FamilyEntry, Fixture, Registry};
use poot_models::test_support::dense::step as step_shape;
use poot_models::test_support::weight_map::MappedModel;
use poot_profile::Report;
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::{BufferRole, DeviceBackend, MemoryCounters};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostTensor};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const CAPACITY: usize = 32;
/// The tag of the row-index slot of the snapshot and restore graphs: identity unless a row is moved.
const ROWS_TAG: &str = "snapshot_rows";

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------------------------

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

/// The family's shipped fixture, built into a model over its own weights for the CPU oracle.
fn mapped(family: FamilyKey) -> MappedModel {
    let entry = entry(family);
    let fixture = poot_models::test_support::family_fixtures::base(entry.family);
    let model = (entry.build)(&fixture.raw(), &fixture.store).unwrap();
    let map = Arc::new(model.model.weights().clone());
    MappedModel {
        model: model.model,
        store: Arc::new(fixture.store),
        map,
    }
}

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    let vocab = fixture.config["vocab_size"]
        .as_u64()
        .expect("the fixture config names its vocab_size") as usize;
    let words = (0..vocab).map(|i| (format!("w{i}"), i as u32)).collect();
    let model = WordLevel::builder()
        .vocab(words)
        .unk_token("w0".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

fn options() -> DriverOptions {
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
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

// ---------------------------------------------------------------------------------------------
// The state a decode step carries, and the snapshot / restore / reorder graphs over it.
// ---------------------------------------------------------------------------------------------

/// One carried state value of a decode program.
#[derive(Clone, Debug)]
struct Carried {
    name: String,
    aval: TensorType,
    role: StateRole,
}

impl Carried {
    fn bytes(&self) -> u64 {
        (self.aval.shape.iter().product::<usize>() * self.aval.dtype.byte_size()) as u64
    }
}

fn decode_graph(m: &MappedModel, rows: usize) -> Graph {
    m.model
        .trace(
            &StepSpec::step(step_shape(rows, 1, CAPACITY, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// Every carried state value of `decode`, in its state order.
fn carried(decode: &Graph) -> Vec<Carried> {
    decode
        .state_pairs()
        .map(|pair| {
            let meta = decode.meta(pair.input);
            Carried {
                name: meta.name.clone().expect("a state input carries its name"),
                aval: meta.aval.clone(),
                role: pair.role,
            }
        })
        .collect()
}

fn recurrent(state: &[Carried]) -> Vec<Carried> {
    state
        .iter()
        .filter(|s| s.role == StateRole::Recurrent)
        .cloned()
        .collect()
}

fn total_bytes(state: &[Carried]) -> u64 {
    state.iter().map(Carried::bytes).sum()
}

fn shadow_name(name: &str) -> String {
    format!("{name}@snapshot")
}

fn rows_slot() -> SlotKey {
    SlotKey::new(Slot::Activation, Some(ROWS_TAG))
}

fn identity_rows(rows: usize) -> Vec<(SlotKey, HostTensor)> {
    vec![(
        rows_slot(),
        HostTensor::i32(vec![rows], (0..rows as i32).collect()),
    )]
}

/// `live_i` and `shadow_i` for each recurrent value, in that interleaved order: snapshot copies live
/// into the shadow, restore copies the shadow back into live. The untouched side is its own state
/// output (an alias, no copy).
fn copy_graph(state: &[Carried], rows: usize, restore: bool) -> Graph {
    let b = Builder::for_tests();
    let index = b
        .slot_named(
            Slot::Activation,
            ROWS_TAG,
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap();
    let mut pairs = Vec::new();
    let mut out = None;
    for s in state {
        let live = b
            .state_input(&s.name, s.aval.clone(), StateRole::Recurrent)
            .unwrap();
        let shadow = b
            .state_input(&shadow_name(&s.name), s.aval.clone(), StateRole::Recurrent)
            .unwrap();
        let copied = if restore {
            let copied = b.gather(shadow, 0, index).unwrap();
            pairs.push((live, copied));
            pairs.push((shadow, shadow));
            copied
        } else {
            let copied = b.gather(live, 0, index).unwrap();
            pairs.push((live, live));
            pairs.push((shadow, copied));
            copied
        };
        out = Some(copied);
    }
    b.finish(
        poot_graph_ir::Outputs::new(out.expect("a recurrent model carries state"))
            .with_state(&pairs),
    )
    .unwrap()
}

/// The graphs POOT-647's driver compiles (`driver/snapshot.rs`): the destination's rows kept or taken
/// from the source by index, `Gather(Concat0(destination, source), idx)` with `idx[r] = r` to keep and
/// `rows + r` to copy. Each row of a round it copies is a bandwidth cost the plain gather above does
/// not show, so the cost rows time both.
fn concat_copy_graph(state: &[Carried], rows: usize, restore: bool) -> Graph {
    let b = Builder::for_tests();
    let index = b
        .slot_named(
            Slot::Activation,
            ROWS_TAG,
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap();
    let mut pairs = Vec::new();
    let mut out = None;
    for s in state {
        let live = b
            .state_input(&s.name, s.aval.clone(), StateRole::Recurrent)
            .unwrap();
        let shadow = b
            .state_input(&shadow_name(&s.name), s.aval.clone(), StateRole::Recurrent)
            .unwrap();
        let copied = if restore {
            let copied = b
                .gather(b.concat(0, &[live, shadow]).unwrap(), 0, index)
                .unwrap();
            pairs.push((live, copied));
            pairs.push((shadow, shadow));
            copied
        } else {
            let copied = b
                .gather(b.concat(0, &[shadow, live]).unwrap(), 0, index)
                .unwrap();
            pairs.push((live, live));
            pairs.push((shadow, copied));
            copied
        };
        out = Some(copied);
    }
    b.finish(
        poot_graph_ir::Outputs::new(out.expect("a recurrent model carries state"))
            .with_state(&pairs),
    )
    .unwrap()
}

/// The beam reorder's shape: every state value of the decode program gathered by row.
fn reorder_graph(state: &[Carried], rows: usize) -> Graph {
    let b = Builder::for_tests();
    let index = b
        .slot_named(
            Slot::Activation,
            ROWS_TAG,
            TensorType::new(vec![rows], DType::I32),
        )
        .unwrap();
    let mut pairs = Vec::new();
    let mut out = None;
    for s in state {
        let value = b.state_input(&s.name, s.aval.clone(), s.role).unwrap();
        let moved = b.gather(value, 0, index).unwrap();
        pairs.push((value, moved));
        out = Some(moved);
    }
    b.finish(poot_graph_ir::Outputs::new(out.expect("a model carries state")).with_state(&pairs))
        .unwrap()
}

// ---------------------------------------------------------------------------------------------
// SC-001: the measured byte count agrees with the planner's.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("counting device error")]
struct CountingError;

/// A device that allocates nothing and counts every byte it was asked for, by role.
struct CountingDevice {
    memory: MemoryCounters,
}

impl Device for CountingDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = CountingError;

    fn target(&self) -> Target {
        Target::Gpu {
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
    ) -> Result<(), CountingError> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
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
    ) -> Result<(), CountingError> {
        Ok(())
    }
    fn begin(&mut self) -> Result<(), CountingError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), CountingError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), CountingError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), CountingError> {
        Ok(())
    }
    fn replay(&mut self, _recording: &()) -> Result<(), CountingError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), CountingError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), _out: &mut [u8]) -> Result<(), CountingError> {
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), CountingError> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), CountingError> {
        Ok(())
    }
}

fn partition() -> Partition {
    Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Single(DeviceId(0)),
    }
}

fn compile_options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}

/// The state bytes the production planner publishes for one decode row of the fixture checkpoint.
fn planned_state_bytes(family: FamilyKey) -> u64 {
    let dir = poot_test_util::unique_temp_path("recurrent_snapshot_cost");
    write_checkpoint(
        &dir,
        &poot_models::test_support::family_fixtures::base(entry(family).family),
    );
    let handle = Arc::new(
        ModelHandle::load(&dir, &Registry::builtin().unwrap(), &LoadOptions::default()).unwrap(),
    );
    let engine = Engine::new(CountingDevice {
        memory: MemoryCounters::new(),
    });
    let planner = Planner::new(handle, &engine.target_set(), Placement::Single).unwrap();
    let planned = planner
        .plan(
            options(),
            &ServingShapes {
                layout: Layout::Contiguous,
                rows: &[NonZeroUsize::MIN],
                heads: &[Head::GREEDY],
                warm: Warm::Admitted,
                windows: &[],
                adapters: &[],
                chain: None,
            },
            &[],
        )
        .unwrap();
    planned
        .footprint
        .role_bytes(DeviceId(0))
        .unwrap()
        .into_iter()
        .find(|(role, _)| *role == BufferRole::State)
        .map_or(0, |(_, bytes)| bytes)
}

/// The bytes the snapshot of `state` allocates for its shadow on a counting device: the live state
/// bytes after adding the snapshot entry hold the live recurrent values and one shadow of each.
fn live_shadow_bytes(state: &[Carried]) -> u64 {
    let graph = copy_graph(state, 1, false);
    let mut engine = Engine::new(CountingDevice {
        memory: MemoryCounters::new(),
    });
    let program = compile_staged(
        &graph,
        &engine.target_set(),
        &partition(),
        &compile_options(),
    )
    .unwrap();
    let exe = engine
        .load_weights(
            Arc::new(WeightStore::builder().build()),
            Arc::new(WeightMap::default()),
        )
        .unwrap();
    engine.add_entry(exe, &program).unwrap();
    let live = engine
        .stats()
        .memory
        .into_iter()
        .find(|(role, _)| *role == BufferRole::State)
        .map_or(0, |(_, snapshot)| snapshot.live_bytes);
    assert_eq!(live % 2, 0, "live values and shadows are the same size");
    live / 2
}

/// SC-001 (reworded by the orchestrator, option B): the harness's per-row state byte count equals the
/// planner's planned `State` role bytes for the same fixture and capacity, and the bytes a snapshot
/// allocates equal the recurrent subset of it. The planned figure is the production planner's
/// (`Planner::plan`, the number admission uses); the measured figure comes from the state avals of the
/// traced decode program and from a snapshot entry's live allocations on a counting device.
/// Mutation: make the snapshot graph skip its first recurrent value (`live_shadow_bytes` passes
/// `&state[1..]` to `copy_graph`); the live shadow bytes fall short of the recurrent bytes the planner accounts for and the
/// row is red.
#[test]
fn the_measured_state_bytes_equal_the_planned_state_bytes() {
    for (label, family) in FAMILIES {
        let m = mapped(family);
        let state = carried(&decode_graph(&m, 1));
        let recurrent_state = recurrent(&state);
        assert!(
            !recurrent_state.is_empty(),
            "{label}: a recurrent fixture carries a recurrent value"
        );
        let positional_bytes = total_bytes(&state) - total_bytes(&recurrent_state);

        let planned = planned_state_bytes(family);
        let shadow = live_shadow_bytes(&recurrent_state);
        eprintln!(
            "{label}: planned state {planned} B/row = recurrent {shadow} B + positional {positional_bytes} B"
        );
        assert_eq!(
            shadow + positional_bytes,
            planned,
            "{label}: the snapshot's live bytes plus the positional bytes vs the planner's State role bytes"
        );

        // The per-row figure is per row: four rows hold four times the one-row bytes.
        let four = carried(&decode_graph(&m, 4));
        assert_eq!(
            total_bytes(&recurrent(&four)),
            4 * total_bytes(&recurrent_state),
            "{label}: recurrent bytes are linear in rows"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The rollback contract on the CPU oracle.
// ---------------------------------------------------------------------------------------------

const PROMPT: [i32; 4] = [3, 5, 1, 7];
const DRAFTS: [i32; 2] = [2, 6];
const NEXT: i32 = 4;

/// Run decode steps of `tokens` from position `at`, carrying `state`.
fn run(
    m: &MappedModel,
    graph: &Graph,
    mut state: Vec<HostTensor>,
    tokens: &[i32],
    at: usize,
) -> (HostTensor, Vec<HostTensor>) {
    let mut logits = None;
    for (i, &token) in tokens.iter().enumerate() {
        let slots = token_pos(1, 1, &[token], &[(at + i) as i32]);
        let (out, next) = step(m, graph, &slots, &state);
        logits = Some(out);
        state = next;
    }
    (logits.expect("at least one token"), state)
}

/// The recurrent entries of `full`, by position in the decode state order.
fn recurrent_at(state: &[Carried]) -> Vec<usize> {
    state
        .iter()
        .enumerate()
        .filter(|(_, s)| s.role == StateRole::Recurrent)
        .map(|(i, _)| i)
        .collect()
}

/// Evaluate a snapshot or restore graph over `live` and `shadow` values (interleaved per value, the
/// graph's state order): the pair after the copy.
fn copy(
    m: &MappedModel,
    graph: &Graph,
    live: &[HostTensor],
    shadow: &[HostTensor],
) -> (Vec<HostTensor>, Vec<HostTensor>) {
    let mut input = Vec::new();
    for (l, s) in live.iter().zip(shadow) {
        input.push(l.clone());
        input.push(s.clone());
    }
    let (_, out) = step(m, graph, &identity_rows(1), &input);
    let live = out.iter().step_by(2).cloned().collect();
    let shadow = out.iter().skip(1).step_by(2).cloned().collect();
    (live, shadow)
}

/// The rollback contract on both recurrent families: after a prompt, snapshot the recurrent values,
/// fold a rejected draft block into the state, restore, and decode the next token; its logits equal
/// those of the run that never saw the drafts, every element. The control skips the restore: the
/// state the drafts folded changes the logits, which is why `commit(kept)` alone cannot serve a
/// recurrent model and why the refusal stands. Mutation: make `copy_graph(.., restore = true)` copy
/// from `live` (a restore that restores nothing); the restored logits are the folded ones and the
/// first assertion is red.
#[test]
fn restoring_a_snapshot_undoes_folded_drafts_on_both_recurrent_families() {
    for (label, family) in FAMILIES {
        let m = mapped(family);
        let graph = decode_graph(&m, 1);
        let state = carried(&graph);
        let at = recurrent_at(&state);
        let recurrent_state = recurrent(&state);
        let snapshot = copy_graph(&recurrent_state, 1, false);
        let restore = copy_graph(&recurrent_state, 1, true);

        let (_, prompted) = run(&m, &graph, zero_state(&graph), &PROMPT, 0);
        let (clean, _) = run(&m, &graph, prompted.clone(), &[NEXT], PROMPT.len());

        // Snapshot at the start of the verify round, fold the drafts, then roll back.
        let live: Vec<_> = at.iter().map(|&i| prompted[i].clone()).collect();
        let zeros: Vec<_> = live
            .iter()
            .map(|t| HostTensor::zeros(t.shape().to_vec()))
            .collect();
        let (_, shadow) = copy(&m, &snapshot, &live, &zeros);
        let (_, folded) = run(&m, &graph, prompted, &DRAFTS, PROMPT.len());
        let live: Vec<_> = at.iter().map(|&i| folded[i].clone()).collect();
        let (restored_live, _) = copy(&m, &restore, &live, &shadow);
        let mut rolled_back = folded.clone();
        for (&i, value) in at.iter().zip(restored_live) {
            rolled_back[i] = value;
        }

        let (after_restore, _) = run(&m, &graph, rolled_back, &[NEXT], PROMPT.len());
        assert_eq!(
            after_restore.as_f32().unwrap(),
            clean.as_f32().unwrap(),
            "{label}: restored logits vs the run that never folded the drafts"
        );
        let (unrestored, _) = run(&m, &graph, folded, &[NEXT], PROMPT.len() + DRAFTS.len());
        assert_ne!(
            unrestored.as_f32().unwrap(),
            clean.as_f32().unwrap(),
            "{label}: folded drafts must change the next logits, or the fixture proves nothing"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Cost on a device (recorded, never gated).
// ---------------------------------------------------------------------------------------------

const ROWS: usize = 4;
const WARMUP: usize = 3;
const REPS: usize = 30;

struct Arm {
    label: &'static str,
    graph: Graph,
    /// The row index the arm binds: identity for a gather, `rows + r` for a concat copy of every row.
    index: Vec<i32>,
}

/// Device time per step of `arm` on `executor`, and the host wall time per step, over [`REPS`] steps
/// after [`WARMUP`] unmeasured ones.
fn time_arm(
    executor: &mut dyn Executor,
    exe: poot_executor::ExecutableId,
    arm: &Arm,
) -> (Option<std::time::Duration>, std::time::Duration, u64) {
    let program = compile_staged(
        &arm.graph,
        &executor.target_set(),
        &partition(),
        &compile_options(),
    )
    .unwrap();
    let entry = executor.add_entry(exe, &program).unwrap();
    let index = HostTensor::i32(vec![ROWS], arm.index.clone());
    let key = rows_slot();
    let mut inputs = StepInputs::new();
    inputs.push(key, index.shape(), index.view());
    let step_once = |executor: &mut dyn Executor| {
        executor
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("{}: {e}", arm.label));
    };
    for _ in 0..WARMUP {
        step_once(executor);
    }
    let mark = executor.stats().timing.mark();
    let started = Instant::now();
    for _ in 0..REPS {
        step_once(executor);
    }
    let wall = started.elapsed() / REPS as u32;
    let timing = executor.stats().timing;
    let entries = timing.entries_with_activity_since(&mark);
    let window = Report::window(&timing, &mark, &entries);
    (
        window
            .device
            .sum_of_dispatch_durations
            .map(|sum| sum / REPS as u32),
        wall,
        window.steps,
    )
}

/// GDN state at checkpoint scale, a synthetic stand-in the fixtures are too small to be: 36 delta-rule
/// layers (three quarters of a 48-layer stack) each carrying a `[32 value heads, 128, 128]` f32
/// matrix per row. The cost of a copy is bytes moved, so this arm shows where a real model lands.
fn checkpoint_scale_gdn(rows: usize) -> Vec<Carried> {
    (0..36)
        .map(|layer| Carried {
            name: format!("layers.{layer}.gdn.state"),
            aval: TensorType::new(vec![rows, 32, 128, 128], DType::F32),
            role: StateRole::Recurrent,
        })
        .collect()
}

fn measure(backend: DeviceBackend, choice: BackendChoice) {
    let Some(mut executor) = poot_runtime_common::device_skip::open_or_skip(
        backend,
        open_timed_executor(choice, DetailedTiming::default_diagnostic()),
    ) else {
        return;
    };
    let mut cases: Vec<(&str, Vec<Carried>)> = FAMILIES
        .iter()
        .map(|&(label, family)| (label, carried(&decode_graph(&mapped(family), ROWS))))
        .collect();
    cases.push(("synthetic checkpoint-scale gdn", checkpoint_scale_gdn(ROWS)));
    for (label, state) in cases {
        let recurrent_state = recurrent(&state);
        let exe = executor
            .load_weights(
                Arc::new(WeightStore::builder().build()),
                Arc::new(WeightMap::default()),
            )
            .unwrap();
        let arms = [
            Arm {
                label: "reorder (gather all state)",
                graph: reorder_graph(&state, ROWS),
                index: (0..ROWS as i32).collect(),
            },
            Arm {
                label: "snapshot (recurrent -> shadow)",
                graph: copy_graph(&recurrent_state, ROWS, false),
                index: (0..ROWS as i32).collect(),
            },
            Arm {
                label: "restore (shadow -> recurrent)",
                graph: copy_graph(&recurrent_state, ROWS, true),
                index: (0..ROWS as i32).collect(),
            },
            Arm {
                label: "snapshot, concat+gather (driver)",
                graph: concat_copy_graph(&recurrent_state, ROWS, false),
                index: (ROWS as i32..2 * ROWS as i32).collect(),
            },
            Arm {
                label: "restore, concat+gather (driver)",
                graph: concat_copy_graph(&recurrent_state, ROWS, true),
                index: (ROWS as i32..2 * ROWS as i32).collect(),
            },
        ];
        eprintln!(
            "{label}: {ROWS} rows, recurrent {} B, all state {} B",
            total_bytes(&recurrent_state),
            total_bytes(&state)
        );
        for arm in &arms {
            let (device, wall, steps) = time_arm(&mut *executor, exe, arm);
            assert_eq!(
                steps, REPS as u64,
                "{label} {}: every step is timed",
                arm.label
            );
            let device = device.unwrap_or_else(|| {
                panic!(
                    "{label} {}: a timed executor measures device time",
                    arm.label
                )
            });
            eprintln!(
                "  {:<32} device {:>9.1?}/step  host wall {:>9.1?}/step",
                arm.label, device, wall
            );
        }
        executor.unload(exe).unwrap();
    }
}

/// The snapshot, restore and reorder steps execute on wgpu and report device time; the figures print
/// for the design note. Speed is recorded, never gated.
#[test]
fn snapshot_restore_and_reorder_cost_on_wgpu() {
    measure(DeviceBackend::Wgpu, BackendChoice::Wgpu);
}

/// The same measurement on ROCm.
#[test]
fn snapshot_restore_and_reorder_cost_on_rocm() {
    measure(DeviceBackend::Rocm, BackendChoice::Rocm);
}

// ---------------------------------------------------------------------------------------------
// Guard: speculation on a recurrent family does not diverge from greedy decode.
// ---------------------------------------------------------------------------------------------

fn generate(
    handle: &Arc<ModelHandle>,
    executor: Box<dyn Executor>,
    draft: DraftPolicy,
) -> Result<Vec<u32>, DriverError> {
    let mut driver = Driver::new(
        Arc::clone(handle),
        executor,
        DriverOptions { draft, ..options() },
    )?
    .serve(Layout::Contiguous);
    // A repeating prompt and a one-token n-gram, so prompt lookup proposes drafts the random fixture
    // then rejects: a rejected draft left folded in the GDN state would change what greedy decode
    // generates.
    let prompt = [3u32, 5, 1, 3, 5, 1, 3, 5].to_vec();
    driver
        .generate(
            GenerateRequest {
                prompt,
                max_new: 12,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .map(|generated| generated.tokens)
}

/// A GDN fixture generating with prompt lookup (ngram 1, k 3) over a prompt whose drafts the model
/// rejects returns the tokens plain greedy decode returns, on the production `generate` path (a real
/// device). Before POOT-647 this row asserted the typed refusal instead, because a rejected draft
/// stayed folded in the state and the tokens diverged; it is the guard that speculation does not do
/// that. Mutation: make `roll_back` skip the restore (return before `copy_state_rows(Restore, ..)`);
/// the rejected drafts stay folded, the tokens diverge from greedy and the assertion prints both lists.
#[test]
fn a_gdn_fixture_speculates_with_prompt_lookup_without_diverging_from_greedy_wgpu() {
    let Some(greedy_executor) = poot_runtime_common::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        poot_llm::driver::open_executor(BackendChoice::Wgpu),
    ) else {
        return;
    };
    let dir = poot_test_util::unique_temp_path("recurrent_snapshot_cost_guard");
    write_checkpoint(
        &dir,
        &poot_models::test_support::family_fixtures::base(
            entry(poot_models::qwen3next::FAMILY).family,
        ),
    );
    let handle = Arc::new(
        ModelHandle::load(&dir, &Registry::builtin().unwrap(), &LoadOptions::default()).unwrap(),
    );
    let greedy = generate(&handle, greedy_executor, DraftPolicy::Off).unwrap();

    let executor = poot_llm::driver::open_executor(BackendChoice::Wgpu).unwrap();
    let lookup = DraftPolicy::PromptLookup {
        ngram: nz(1),
        k: nz(3),
    };
    let speculated = generate(&handle, executor, lookup).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        speculated, greedy,
        "speculation over recurrent state differs from greedy decode"
    );
}
