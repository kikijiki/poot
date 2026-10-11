//! POOT-745: the staged walk, on a fake `Device` that logs every call.
//!
//! A pipeline (the layered fixture cut after layers 0 and 1, all on one device) must behave as one
//! step to the host: each stage records once and replays every step, a link is a shared device
//! buffer and never a host write, the stages enqueue back to back and wait once, a stage on another
//! device is refused with a typed placement error before anything is allocated, and a recording
//! failure in a later stage leaves no partial entry behind. Model-free: only real SPIR-V codegen to
//! compile the fixture's kernels, which the fake never runs. What the stages compute is the device
//! lanes' business (`poot-executor-parity`'s pipeline rows).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use std::time::Duration;

use poot_executor::{
    BufferRole, DetailedTiming, Device, DeviceTime, Dispatch, DispatchDeviceTime, DrainPolicy,
    Engine, ExecError, ExecutableId, Executor, LoadError, MeasuredDeviceTime, NoSync, StepInputs,
    TimingOptions,
};
use poot_graph_ir::{
    BinOp, BoundaryId, Builder, Export, Slot, SlotKey, StateRole, TensorType, ValidationOutputs,
};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, Link, Partition,
    StageId, StagedParts, StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostView};

const LAYERS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Allocate,
    Begin,
    Finish,
    BeginStep,
    Replay(u32),
    Write(usize),
    Synchronize,
    Abort,
}

#[derive(Debug, thiserror::Error)]
#[error("fake device: {0} failed (injected)")]
struct Injected(&'static str);

/// The test's handle on the device once the engine owns it.
#[derive(Clone, Default)]
struct Control {
    log: Rc<RefCell<Vec<Event>>>,
    /// Fail the `n`th `begin` (1-based) of the device's life.
    fail_begin_at: Rc<RefCell<Option<usize>>>,
    fail_abort: Rc<RefCell<bool>>,
    /// Dispatches replayed since the counter was last taken: what the whole step ran, whatever the
    /// device went on to report.
    replayed: Rc<std::cell::Cell<usize>>,
}

impl Control {
    fn events(&self) -> Vec<Event> {
        self.log.borrow().clone()
    }

    fn clear(&self) {
        self.log.borrow_mut().clear();
    }

    /// Bytes the device was handed by `write` since the last `clear`: the independent instrument
    /// beside the engine's own `host_write_bytes` counter.
    fn written_bytes(&self) -> u64 {
        self.log
            .borrow()
            .iter()
            .map(|e| match e {
                Event::Write(bytes) => *bytes as u64,
                _ => 0,
            })
            .sum()
    }

    fn count(&self, wanted: Event) -> usize {
        self.log.borrow().iter().filter(|e| **e == wanted).count()
    }
}

struct FakeDevice {
    control: Control,
    begins: usize,
    recordings: u32,
    /// Dispatches the recording being made holds, and each finished recording's total (recording
    /// `n` is `recorded[n - 1]`).
    dispatched: u32,
    recorded: Vec<u32>,
    /// The step's device timing, accumulated across its replays since `begin_step`, as a real timed
    /// device does: each replay appends its recording's dispatches after the earlier stages'.
    timed: Vec<DispatchDeviceTime>,
    memory: poot_runtime_common::MemoryCounters,
}

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = u32;
    type Error = Injected;

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
    ) -> Result<(), Injected> {
        self.control.log.borrow_mut().push(Event::Allocate);
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        Ok(())
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), Injected> {
        Ok(())
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), Injected> {
        self.begins += 1;
        self.dispatched = 0;
        self.control.log.borrow_mut().push(Event::Begin);
        if *self.control.fail_begin_at.borrow() == Some(self.begins) {
            return Err(Injected("begin"));
        }
        Ok(())
    }

    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), Injected> {
        self.dispatched += 1;
        Ok(())
    }

    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), Injected> {
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<u32>, Injected> {
        self.control.log.borrow_mut().push(Event::Finish);
        self.recordings += 1;
        self.recorded.push(self.dispatched);
        Ok(Some(self.recordings))
    }

    fn begin_step(&mut self) -> Result<(), Injected> {
        self.control.log.borrow_mut().push(Event::BeginStep);
        self.timed.clear();
        Ok(())
    }

    fn replay(&mut self, recording: &u32) -> Result<(), Injected> {
        self.control
            .replayed
            .set(self.control.replayed.get() + self.recorded[*recording as usize - 1] as usize);
        for _ in 0..self.recorded[*recording as usize - 1] {
            self.timed.push(DispatchDeviceTime {
                index: self.timed.len(),
                duration: Duration::from_micros(1),
            });
        }
        self.control
            .log
            .borrow_mut()
            .push(Event::Replay(*recording));
        Ok(())
    }

    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Injected> {
        self.control.log.borrow_mut().push(Event::Write(len));
        Ok(())
    }

    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), Injected> {
        out.fill(0);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), Injected> {
        self.control.log.borrow_mut().push(Event::Synchronize);
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Measured(MeasuredDeviceTime {
            sum_of_dispatch_durations: Some(Duration::from_micros(self.timed.len() as u64)),
            device_span: Some(Duration::from_micros(self.timed.len() as u64)),
            dispatches: self.timed.clone(),
        })
    }

    fn abort(&mut self) -> Result<(), Injected> {
        self.control.log.borrow_mut().push(Event::Abort);
        if *self.control.fail_abort.borrow() {
            return Err(Injected("abort"));
        }
        Ok(())
    }
}

fn new_engine() -> (Engine<FakeDevice>, Control) {
    new_engine_timed(TimingOptions::CountersOnly)
}

fn new_engine_timed(timing: TimingOptions) -> (Engine<FakeDevice>, Control) {
    let control = Control::default();
    let device = FakeDevice {
        control: control.clone(),
        begins: 0,
        recordings: 0,
        dispatched: 0,
        recorded: Vec::new(),
        timed: Vec::new(),
        memory: poot_runtime_common::MemoryCounters::new(),
    };
    (Engine::with_timing(device, timing), control)
}

/// `LAYERS` blocks in the shape a pipeline partition cuts: one activation slot, one shared
/// constant, and one carried state pair per block. Each block is tagged with its layer, so a cut
/// after layer `k` lands between blocks.
fn layered() -> (
    poot_graph_ir::Graph<ValidationOutputs>,
    Arc<WeightStore>,
    Arc<WeightMap>,
) {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let w = b.constant("w", TensorType::f32(vec![4]));
    let mut hidden = x;
    let mut state = Vec::new();
    for layer in 0..LAYERS {
        let _scope = b.layer_scope(layer);
        let scaled = b.binary(BinOp::Mul, hidden, w);
        let k_in = b.state_input(
            &format!("k{layer}"),
            TensorType::f32(vec![4]),
            StateRole::Recurrent,
        );
        let k_out = b.binary(BinOp::Add, k_in, scaled);
        hidden = b.binary(BinOp::Add, scaled, k_out);
        state.push((k_in, k_out));
    }
    let g = b
        .finish_with_state(hidden, &state)
        .with_validations(Vec::new());
    let mut builder = WeightStore::builder();
    let bytes: Vec<u8> = [0.5f32, 2.0, -1.25, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    builder
        .insert(
            "w",
            WeightEntry::Dense(
                DenseWeight::try_new(DType::F32, vec![4], StoredBytes::from(bytes)).unwrap(),
            ),
        )
        .unwrap();
    let store = Arc::new(builder.build());
    let mut names = poot_test_util::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());
    (g, store, map)
}

fn target() -> Target {
    Target {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

fn compile(
    g: &poot_graph_ir::Graph<ValidationOutputs>,
    device: DeviceId,
    devices: DevicePlacement,
) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(device, target()),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices,
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .expect("the layered fixture compiles for the fake target")
}

fn single(g: &poot_graph_ir::Graph<ValidationOutputs>) -> StagedProgram<ValidationOutputs> {
    compile(g, DeviceId(0), DevicePlacement::Single(DeviceId(0)))
}

/// Cut after layers 0 and 1: three stages, all on device 0.
fn pipeline(g: &poot_graph_ir::Graph<ValidationOutputs>) -> StagedProgram<ValidationOutputs> {
    compile(
        g,
        DeviceId(0),
        DevicePlacement::Pipeline {
            after_layers: vec![0, 1],
            devices: vec![DeviceId(0); 3],
        },
    )
}

/// Two executables over one engine, one holding the uncut program and one the pipeline.
struct Rig {
    engine: Engine<FakeDevice>,
    control: Control,
    single: (ExecutableId, poot_executor::EntryId),
    pipeline: (ExecutableId, poot_executor::EntryId),
}

fn rig() -> Rig {
    rig_timed(TimingOptions::CountersOnly)
}

fn rig_timed(timing: TimingOptions) -> Rig {
    let (mut engine, control) = new_engine_timed(timing);
    let (g, store, map) = layered();
    let load = |engine: &mut Engine<FakeDevice>, program: &StagedProgram<ValidationOutputs>| {
        let exe = engine.load_weights(store.clone(), map.clone()).unwrap();
        let entry = engine.add_entry(exe, program).unwrap();
        (exe, entry)
    };
    let single = load(&mut engine, &single(&g));
    let pipeline = load(&mut engine, &pipeline(&g));
    Rig {
        engine,
        control,
        single,
        pipeline,
    }
}

fn step(engine: &mut Engine<FakeDevice>, (exe, entry): (ExecutableId, poot_executor::EntryId)) {
    let values = [1.5f32, -0.25, 3.0, 0.75];
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let shape = [4usize];
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        &shape,
        HostView::new(DType::F32, 4, bytes).unwrap(),
    );
    engine
        .step(exe, entry, &inputs, &mut NoSync)
        .expect("the step runs")
        .read()
        .expect("the output reads");
}

fn stage_counts(
    rig: &Rig,
    (exe, entry): (ExecutableId, poot_executor::EntryId),
) -> Vec<(StageId, u64, u64)> {
    rig.engine
        .stats()
        .stages
        .iter()
        .filter(|stats| stats.exe == exe && stats.entry == entry)
        .map(|stats| (stats.stage, stats.recordings, stats.replays))
        .collect()
}

/// SC-002: the host writes the same bytes for a three-stage step as for the one-stage step. A link
/// is a buffer both stages bind, never a read back and a write; the cut adds none.
///
/// Mutation: implement a link as `read` then `write` through the host (engine `bind` of a boundary
/// slot); the pipeline's counter grows past the single stage's and this goes red. A second mutation,
/// an extra `device.write` in `bind` that bypasses the counting helper, makes the counter disagree
/// with the device's own write log.
#[test]
fn a_three_stage_step_writes_the_host_bytes_a_one_stage_step_does() {
    let mut rig = rig();
    // Counted twice: by the engine's counter and by the device's own log of the bytes it was handed.
    // A write the engine does not count shows up as a disagreement between the two.
    let measure = |rig: &mut Rig, program| {
        let before = rig.engine.stats().host_write_bytes;
        rig.control.clear();
        step(&mut rig.engine, program);
        let counted = rig.engine.stats().host_write_bytes - before;
        assert_eq!(
            counted,
            rig.control.written_bytes(),
            "the engine's host-write counter counts exactly the bytes the device was handed"
        );
        counted
    };
    let single = rig.single;
    let one_stage = measure(&mut rig, single);
    let pipeline = rig.pipeline;
    let three_stage = measure(&mut rig, pipeline);
    assert!(
        one_stage > 0,
        "the step writes the activation slot, or the counter proves nothing"
    );
    assert_eq!(
        three_stage, one_stage,
        "a link must not add a host write: {three_stage} bytes against the one-stage {one_stage}"
    );
}

/// SC-003: after 16 steps every stage has recorded once and replayed 16 times.
///
/// Mutation: re-record every stage on every step (clear `recording` in `enqueue`); the recordings
/// column reads 16 and this goes red.
#[test]
fn every_stage_records_once_across_sixteen_steps() {
    let mut rig = rig();
    for _ in 0..16 {
        step(&mut rig.engine, rig.pipeline);
    }
    let counts = stage_counts(&rig, rig.pipeline);
    assert_eq!(
        counts,
        vec![
            (StageId(0), 1, 16),
            (StageId(1), 1, 16),
            (StageId(2), 1, 16)
        ],
        "each stage records once and replays every step"
    );
}

/// SC-004: a program compiled for `DeviceId(1)` is refused by an engine at `DeviceId(0)` with the
/// typed placement error, before the engine allocates anything for it.
///
/// Mutation: drop the placement check in `add_entry`; the program loads and this goes red.
#[test]
fn a_program_for_another_device_is_refused_before_allocation() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = layered();
    let exe = engine.load_weights(store, map).unwrap();
    let foreign = compile(&g, DeviceId(1), DevicePlacement::Single(DeviceId(1)));
    control.clear();
    let error = engine
        .add_entry(exe, &foreign)
        .expect_err("a stage placed on device 1 is not this engine's");
    assert!(
        matches!(
            error,
            ExecError::Placement {
                stage: StageId(0),
                program: DeviceId(1),
                engine: DeviceId(0),
            }
        ),
        "expected the typed placement error, got {error:?}"
    );
    assert_eq!(
        control.count(Event::Allocate),
        0,
        "the refusal comes before any allocation"
    );
}

/// SC-006: a three-stage step enqueues every stage's replay back to back and waits once, at the end;
/// the first step records every stage before it replays any.
///
/// Mutation: complete (synchronize) after each stage, as a public per-stage step would; the wait
/// count reads 3 and this goes red. Replaying the stages in reverse order reads 3, 2, 1 and goes
/// red on the order.
#[test]
fn a_pipeline_step_waits_once_at_the_end_and_never_between_stages() {
    let mut rig = rig();
    rig.control.clear();
    step(&mut rig.engine, rig.pipeline);
    let events: Vec<Event> = rig
        .control
        .events()
        .into_iter()
        .filter(|e| !matches!(e, Event::Allocate | Event::Write(_) | Event::BeginStep))
        .collect();
    let replays: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, Event::Replay(_)).then_some(i))
        .collect();
    assert_eq!(replays.len(), 3, "one replay per stage: {events:?}");
    let order: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            Event::Replay(recording) => Some(*recording),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        vec![1, 2, 3],
        "the stages replay in stage order (recordings are made in stage order): {events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| **e == Event::Synchronize).count(),
        1,
        "one completion wait for the whole step: {events:?}"
    );
    assert_eq!(
        events.last(),
        Some(&Event::Synchronize),
        "the wait is after the last replay: {events:?}"
    );
    assert!(
        replays.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "no wait or other call between two stage replays: {events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| **e == Event::Begin).count(),
        3,
        "the first step records each stage: {events:?}"
    );

    // Later steps only replay and wait once.
    rig.control.clear();
    step(&mut rig.engine, rig.pipeline);
    assert_eq!(rig.control.count(Event::Synchronize), 1);
    assert_eq!(rig.control.count(Event::Begin), 0);
}

/// SC-009: a recording failure in the second stage aborts the open recording once, publishes no
/// recording for any stage (not even the first, which recorded fine), replays nothing, and leaves
/// the entry usable: the retry records all three and waits once.
///
/// Mutation: publish each stage's recording as it finishes (move the `recording` write into the
/// recording loop of `enqueue`); stage 0 keeps a recording across the failure and this goes red.
#[test]
fn a_recording_failure_in_stage_two_publishes_no_partial_entry() {
    let mut rig = rig();
    // Begins so far: 0 (nothing recorded). The pipeline's second stage is the entry's second begin.
    *rig.control.fail_begin_at.borrow_mut() = Some(2);
    rig.control.clear();
    let (exe, entry) = rig.pipeline;
    let values = [1.5f32, -0.25, 3.0, 0.75];
    let shape = [4usize];
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        &shape,
        HostView::new(DType::F32, 4, bytemuck::cast_slice(&values)).unwrap(),
    );
    let error = match rig.engine.step(exe, entry, &inputs, &mut NoSync) {
        Ok(_) => panic!("the injected stage-two begin failure must surface"),
        Err(error) => error,
    };
    assert!(matches!(error, ExecError::Device(_)), "{error:?}");
    assert_eq!(
        rig.control.count(Event::Abort),
        1,
        "the open recording is aborted exactly once"
    );
    assert!(
        !rig.control
            .events()
            .iter()
            .any(|e| matches!(e, Event::Replay(_) | Event::Synchronize)),
        "nothing replays or waits after a failed recording: {:?}",
        rig.control.events()
    );
    assert_eq!(
        stage_counts(&rig, rig.pipeline),
        vec![(StageId(0), 0, 0), (StageId(1), 0, 0), (StageId(2), 0, 0)],
        "no stage published a recording"
    );

    // Recovery: the failure was in capture, which executes nothing, so the entry records afresh.
    *rig.control.fail_begin_at.borrow_mut() = None;
    rig.control.clear();
    step(&mut rig.engine, rig.pipeline);
    assert_eq!(
        stage_counts(&rig, rig.pipeline),
        vec![(StageId(0), 1, 1), (StageId(1), 1, 1), (StageId(2), 1, 1)]
    );
    assert_eq!(rig.control.count(Event::Synchronize), 1);
}

/// SC-009 (poison): when the abort itself fails the device cannot be reused, and the engine says so
/// for every later call.
#[test]
fn a_failed_abort_poisons_the_engine() {
    let mut rig = rig();
    *rig.control.fail_begin_at.borrow_mut() = Some(2);
    *rig.control.fail_abort.borrow_mut() = true;
    let (exe, entry) = rig.pipeline;
    let shape = [4usize];
    let values = [1.0f32; 4];
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        &shape,
        HostView::new(DType::F32, 4, bytemuck::cast_slice(&values)).unwrap(),
    );
    assert!(rig.engine.step(exe, entry, &inputs, &mut NoSync).is_err());
    *rig.control.fail_begin_at.borrow_mut() = None;
    let error = match rig.engine.step(exe, entry, &inputs, &mut NoSync) {
        Ok(_) => panic!("a poisoned engine takes no further work"),
        Err(error) => error,
    };
    assert!(matches!(error, ExecError::Poisoned), "{error:?}");
}

/// SC-006, timing half: the step's device timing is reset once, ahead of the first stage's replay, so
/// a three-stage step under detailed timing reports every stage's dispatches (the fake accumulates
/// across replays as a real timed device does), each under its op kind: the report's dispatch count
/// is exactly what the device replayed across the three stages.
///
/// Mutation: reset the fake's timing in `replay` instead of `begin_step` (the old wgpu behaviour);
/// only the last stage's dispatches are reported and the count falls short of what was replayed.
#[test]
fn a_detailed_timing_pipeline_step_reports_every_stage() {
    let mut rig = rig_timed(TimingOptions::Detailed(DetailedTiming {
        sample_every: 1,
        max_retained_steps: 8,
        max_retained_dispatch_records: 1024,
        max_in_flight_queries: 4,
        drain_policy: DrainPolicy::WaitForCapacity,
    }));
    let (exe, entry) = rig.pipeline;
    let mark = rig.engine.stats().timing.mark();
    rig.control.replayed.set(0);
    step(&mut rig.engine, (exe, entry));
    let report = poot_profile::Report::window(&rig.engine.stats().timing, &mark, &[entry.raw()]);
    let reported: u64 = report.buckets.values().map(|b| b.dispatch_count).sum();
    assert!(rig.control.replayed.get() >= 3, "every stage dispatches");
    assert_eq!(
        reported,
        rig.control.replayed.get() as u64,
        "the step reports the dispatches of all three stages: {:?}",
        report.buckets
    );
}

/// The load-time invariants the engine keeps for programs `compile_staged` never admits. Each row
/// assembles such a program from parts and expects the typed refusal, never a panic or a silent load.
fn load_refusal(
    parts: StagedParts<ValidationOutputs>,
    store: Arc<WeightStore>,
    map: Arc<WeightMap>,
) -> LoadError {
    let (mut engine, _control) = new_engine();
    let exe = engine.load_weights(store, map).unwrap();
    let program = StagedProgram::from_parts(parts);
    match engine.add_entry(exe, &program) {
        Err(ExecError::Load(error)) => *error,
        other => panic!("expected a load refusal, got {:?}", other.map(|_| ())),
    }
}

/// A link whose consumer loads before its producer has exported anything is `BoundaryUnresolved`.
///
/// Mutation: bind the link's slot as an ordinary input when the exported map has no entry; the
/// program loads and this goes red.
#[test]
fn a_link_read_before_its_producer_exported_is_unresolved() {
    let (g, store, map) = layered();
    let StagedParts {
        stages,
        links,
        sync_points,
    } = pipeline(&g).into_parts();
    let last = stages.len() as u32 - 1;
    let flip = |stage: StageId| StageId(last - stage.0);
    let stages = stages
        .into_iter()
        .rev()
        .map(|(stage, device, program)| (flip(stage), device, program))
        .collect();
    let links = links
        .into_iter()
        .map(|link| Link {
            from: flip(link.from),
            to: flip(link.to),
            ..link
        })
        .collect();
    let error = load_refusal(
        StagedParts {
            stages,
            links,
            sync_points,
        },
        store,
        map,
    );
    assert!(
        matches!(
            error,
            LoadError::BoundaryUnresolved {
                stage: StageId(0),
                ..
            }
        ),
        "got {error:?}"
    );
}

/// Two stages that declare one slot key with different avals would share one host-written buffer:
/// `SlotConflict`.
///
/// Mutation: drop the aval/storage comparison where a stage reuses an earlier stage's slot buffer;
/// the second stage binds a 4-element buffer for 8 elements and this goes red.
#[test]
fn two_stages_declaring_one_slot_differently_conflict() {
    let (g, store, map) = layered();
    let wide = {
        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::f32(vec![8]));
        let out = b.binary(BinOp::Add, x, x);
        b.finish(out).with_validations(Vec::new())
    };
    let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let mut parts = single(&g).into_parts();
    parts.stages.push((
        StageId(1),
        DeviceId(0),
        poot_graph_plan::compile(&wide, &target(), &options).unwrap(),
    ));
    let error = load_refusal(parts, store, map);
    assert!(
        matches!(&error, LoadError::SlotConflict { key } if *key == SlotKey::new(Slot::Activation, None)),
        "got {error:?}"
    );
}

/// An export that views carried state is `BoundaryAliasesState` at load, however the program was
/// assembled: `compile_staged` refuses such a cut, and the engine keeps the check as the invariant.
///
/// Mutation: skip the `export_aliasing_state` check in `load_entry`; the program loads and this goes
/// red.
#[test]
fn an_export_viewing_carried_state_is_refused_at_load() {
    let (_, store, map) = layered();
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let k_in = b.state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent);
    let view = b.reshape(k_in, vec![2, 2]);
    let k_out = b.binary(BinOp::Add, k_in, x);
    let out = b.binary(BinOp::Add, view, view);
    let mut graph = b
        .finish_with_state(out, &[(k_in, k_out)])
        .with_validations(Vec::new());
    graph.exports.push(Export {
        id: BoundaryId(view.id()),
        value: view.id(),
    });
    let options = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let program = poot_graph_plan::compile(&graph, &target(), &options).unwrap();
    let error = load_refusal(
        StagedParts {
            stages: vec![(StageId(0), DeviceId(0), program)],
            links: Vec::new(),
            sync_points: Vec::new(),
        },
        store,
        map,
    );
    assert!(
        matches!(error, LoadError::BoundaryAliasesState { .. }),
        "got {error:?}"
    );
}

/// Load the layered fixture with its first dispatching equation rebound by `rebind` into a plan declaring
/// `scratch` scratch buffers, and return the load refusal. The planner emits no such plan; the engine's
/// binding walk is the invariant for a program assembled some other way.
fn rebound_refusal(
    scratch: usize,
    rebind: impl Fn(&poot_graph_plan::Binding) -> poot_graph_plan::Binding,
) -> LoadError {
    let (g, store, map) = layered();
    let mut parts = single(&g).into_parts();
    let program = &mut parts.stages[0].2;
    let index = program
        .planned()
        .position(|(_, plan)| !plan.dispatches().is_empty())
        .expect("a dispatching equation");
    program
        .rebind_dispatches(index, scratch, rebind)
        .expect("the rebound plan is a valid DispatchPlan");
    load_refusal(parts, store, map)
}

/// A dispatch bound to a scratch buffer is refused at load: the engine allocates none until POOT-1113.
/// (A scratch read needs an earlier dispatch's scratch write, which this refusal stops first.)
///
/// Mutation: treat `Sink::Scratch` as `Sink::Out` in `load_stage`'s binding walk; the program loads and
/// this goes red.
#[test]
fn a_dispatch_writing_scratch_is_refused_at_load() {
    use poot_graph_plan::{Binding, ScratchId, Sink};
    let error = rebound_refusal(1, |binding| {
        Binding::new(binding.reads().to_vec(), Sink::Scratch(ScratchId::new(0)))
    });
    assert!(
        matches!(error, LoadError::Unimplemented(what) if what.contains("scratch")),
        "got {error:?}"
    );
}

/// A binding naming an operand the equation lacks is refused at load, whoever assembled the program.
///
/// Mutation: replace the `ok_or` on the operand lookup with `unwrap_or(&operands[0])`; the program loads
/// and this goes red.
#[test]
fn a_dispatch_binding_an_operand_its_equation_lacks_is_refused_at_load() {
    use poot_graph_plan::{Binding, Source};
    let error = rebound_refusal(0, |binding| {
        let mut reads = binding.reads().to_vec();
        reads[0] = Source::Operand(reads.len() + 4);
        Binding::new(reads, binding.write())
    });
    assert!(
        matches!(error, LoadError::Unimplemented(what) if what.contains("operand")),
        "got {error:?}"
    );
}
