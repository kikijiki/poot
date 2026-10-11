//! POOT-748: the captured coordinator, on fake devices that execute.
//!
//! A fake device keeps real bytes in real buffers and gives every dispatch a deterministic function
//! of its kernel and its input bytes, so an output equals the ordinary engine's only when every
//! boundary value reached the stage that reads it, in the right step, from the right producer. The
//! fake also keeps the one log all of a coordinator's devices write to, so these rows see the order
//! of recordings, copies, event records and waits, the completion waits, and when each buffer and
//! event is finally released. What the kernels compute is the device lanes' business
//! (`poot-ptx-gpu`'s two-device rows).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    BufferRole, Collectives, Coordinator, CoordinatorError, Device, DeviceTime, Dispatch, Engine,
    ExecError, ExecutableId, Executor, NativeCounters, NoSync, StepInputs, SupportBound,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{
    DenseWeight, FfnRole, WeightEntry, WeightId, WeightMap, WeightRole, WeightStore, WeightView,
};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostView};

const LAYERS: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Seam {
    Dispatch,
    Copy,
    PeerCopy { bytes: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    Begin(usize),
    Finish(usize),
    Abort(usize),
    BeginStep(usize),
    Replay(usize, Vec<Seam>),
    Synchronize(usize),
    FreeBuffer(usize),
    FreeEvent(usize),
    /// `Record(device, event)`: the event was recorded on `device` right after its replay.
    Record(usize, usize),
    /// `Wait(device, event)`: `device` waits for the event before its next replay.
    Wait(usize, usize),
    /// A device waited on an event no device had recorded in this step.
    StaleWait(usize),
}

#[derive(Debug, thiserror::Error)]
#[error("fake device {0}: {1} failed (injected)")]
struct Injected(usize, &'static str);

/// Everything the devices of one test share.
#[derive(Default)]
struct World {
    log: RefCell<Vec<Op>>,
    /// Incremented when device 0 starts a step's replays: what an event must have been recorded in.
    epoch: Cell<u64>,
    next_id: Cell<usize>,
    allocations: Cell<u64>,
    events_created: Cell<u64>,
    /// Fail the `n`th `finish` (1-based, over all devices).
    fail_finish_at: Cell<Option<usize>>,
    finishes: Cell<usize>,
    fail_abort: Cell<bool>,
    /// Directed routes the fake refuses to open.
    refused_routes: RefCell<Vec<(usize, usize)>>,
    /// Free memory every device reports; `None` is a device that cannot say.
    free_memory: Cell<Option<u64>>,
    /// Each device's queue of enqueued-but-unexecuted zeroing of a fresh allocation, and how many it
    /// has executed, as a real stream would: work another device's stream does is not ordered after
    /// it unless an event says so.
    queues: RefCell<Vec<std::collections::VecDeque<Buf>>>,
    executed: RefCell<Vec<usize>>,
    enqueued: RefCell<Vec<usize>>,
}

impl World {
    /// A world whose devices report ample free memory.
    fn new() -> Self {
        let world = Self::default();
        world.free_memory.set(Some(1 << 40));
        world
    }

    /// Execute `device`'s queued zeroing until `upto` allocations of its life have been executed.
    fn drain(&self, device: usize, upto: usize) {
        let mut queues = self.queues.borrow_mut();
        let mut executed = self.executed.borrow_mut();
        let Some(queue) = queues.get_mut(device) else {
            return;
        };
        while executed[device] < upto {
            let Some(buf) = queue.pop_front() else { break };
            buf.data.borrow_mut().fill(0);
            executed[device] += 1;
        }
    }

    fn drain_all(&self, device: usize) {
        let upto = self.enqueued.borrow().get(device).copied().unwrap_or(0);
        self.drain(device, upto);
    }

    fn push(&self, op: Op) {
        self.log.borrow_mut().push(op);
    }

    fn id(&self) -> usize {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    fn ops(&self) -> Vec<Op> {
        self.log.borrow().clone()
    }

    fn clear(&self) {
        self.log.borrow_mut().clear();
    }

    fn count(&self, wanted: impl Fn(&Op) -> bool) -> usize {
        self.log.borrow().iter().filter(|op| wanted(op)).count()
    }
}

/// One device allocation: freed (and logged) when the last handle drops, whether that is the engine's
/// or a recording's.
struct Alloc {
    id: usize,
    data: RefCell<Vec<u8>>,
    world: Rc<World>,
}

impl Drop for Alloc {
    fn drop(&mut self) {
        self.world.push(Op::FreeBuffer(self.id));
    }
}

type Buf = Rc<Alloc>;

struct EventState {
    id: usize,
    recorded_in: Cell<Option<u64>>,
    /// The recording device and how many of its allocations' zeroing precede the record.
    mark: Cell<Option<(usize, usize)>>,
    world: Rc<World>,
}

impl Drop for EventState {
    fn drop(&mut self) {
        self.world.push(Op::FreeEvent(self.id));
    }
}

type Ev = Rc<EventState>;

/// One recorded operation, holding the handles it names.
enum Recorded {
    Dispatch {
        kernel: u64,
        inputs: Vec<Buf>,
        output: Buf,
    },
    Copy(Buf, Buf),
    PeerCopy {
        src: Buf,
        dst: Buf,
        bytes: usize,
    },
}

struct Recording {
    ops: Vec<Recorded>,
}

struct FakeDevice {
    index: usize,
    world: Rc<World>,
    open: Option<Vec<Recorded>>,
    graphs: u64,
    launches: u64,
    events: u64,
    memory: poot_runtime_common::MemoryCounters,
}

fn target() -> Target {
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: DeviceCaps {
            memory_budget_bytes: Some(1 << 30),
            ..DeviceCaps::ptx_default()
        },
    }
}

fn mix(mut h: u64, v: u64) -> u64 {
    h ^= v;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^ (h >> 29)
}

impl Device for FakeDevice {
    type Buffer = Buf;
    type Kernel = u64;
    type Recording = Rc<Recording>;
    type Error = Injected;

    fn target(&self) -> Target {
        target()
    }

    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        self.world.free_memory.get()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Buf, Injected> {
        let bytes = elems * storage.element().byte_width();
        self.world.allocations.set(self.world.allocations.get() + 1);
        std::mem::forget(self.memory.record_alloc(role, bytes as u64));
        let buf = Rc::new(Alloc {
            id: self.world.id(),
            // Fresh memory is garbage until the enqueued zeroing runs.
            data: RefCell::new(vec![0xAA; bytes]),
            world: Rc::clone(&self.world),
        });
        let mut queues = self.world.queues.borrow_mut();
        let mut enqueued = self.world.enqueued.borrow_mut();
        let mut executed = self.world.executed.borrow_mut();
        while queues.len() <= self.index {
            queues.push(Default::default());
            enqueued.push(0);
            executed.push(0);
        }
        queues[self.index].push_back(Rc::clone(&buf));
        enqueued[self.index] += 1;
        Ok(buf)
    }

    fn load_kernel(
        &mut self,
        key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<u64, Injected> {
        Ok(key
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325, |h, b| mix(h, u64::from(b))))
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), Injected> {
        self.world.push(Op::Begin(self.index));
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Injected> {
        self.open
            .as_mut()
            .expect("a dispatch inside a recording")
            .push(Recorded::Dispatch {
                kernel: *d.kernel,
                inputs: d.inputs.iter().map(|arg| Rc::clone(arg.buffer)).collect(),
                output: Rc::clone(d.output.buffer),
            });
        Ok(())
    }

    fn copy(&mut self, src: &Buf, dst: &Buf) -> Result<(), Injected> {
        self.open
            .as_mut()
            .expect("a copy inside a recording")
            .push(Recorded::Copy(Rc::clone(src), Rc::clone(dst)));
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<Rc<Recording>>, Injected> {
        let finishes = self.world.finishes.get() + 1;
        self.world.finishes.set(finishes);
        if self.world.fail_finish_at.get() == Some(finishes) {
            return Err(Injected(self.index, "finish"));
        }
        self.world.push(Op::Finish(self.index));
        self.graphs += 1;
        let ops = self.open.take().expect("finish closes a recording");
        Ok(Some(Rc::new(Recording { ops })))
    }

    fn begin_step(&mut self) -> Result<(), Injected> {
        self.world.push(Op::BeginStep(self.index));
        if self.index == 0 {
            self.world.epoch.set(self.world.epoch.get() + 1);
        }
        Ok(())
    }

    fn replay(&mut self, recording: &Rc<Recording>) -> Result<(), Injected> {
        self.world.drain_all(self.index);
        self.launches += 1;
        let mut seams = Vec::new();
        for op in &recording.ops {
            match op {
                Recorded::Dispatch {
                    kernel,
                    inputs,
                    output,
                } => {
                    let mut out = output.data.borrow_mut();
                    let sources: Vec<Vec<u8>> =
                        inputs.iter().map(|buf| buf.data.borrow().clone()).collect();
                    for (i, byte) in out.iter_mut().enumerate() {
                        let mut h = mix(*kernel, i as u64);
                        for source in &sources {
                            if !source.is_empty() {
                                h = mix(h, u64::from(source[i % source.len()]));
                            }
                        }
                        *byte = (h >> 24) as u8;
                    }
                    seams.push(Seam::Dispatch);
                }
                Recorded::Copy(src, dst) => {
                    let bytes = src.data.borrow().clone();
                    dst.data.borrow_mut()[..bytes.len()].copy_from_slice(&bytes);
                    seams.push(Seam::Copy);
                }
                Recorded::PeerCopy { src, dst, bytes } => {
                    let count = *bytes;
                    let moved = src.data.borrow()[..count].to_vec();
                    dst.data.borrow_mut()[..count].copy_from_slice(&moved);
                    seams.push(Seam::PeerCopy { bytes: *bytes });
                }
            }
        }
        self.world.push(Op::Replay(self.index, seams));
        Ok(())
    }

    fn write_with(
        &mut self,
        dst: &Buf,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Injected> {
        self.world.drain_all(self.index);
        let mut data = dst.data.borrow_mut();
        fill(offset..offset + len, &mut data[offset..offset + len]);
        Ok(())
    }

    fn read(&mut self, src: &Buf, out: &mut [u8]) -> Result<(), Injected> {
        self.world.drain_all(self.index);
        out.copy_from_slice(&src.data.borrow()[..out.len()]);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), Injected> {
        self.world.push(Op::Synchronize(self.index));
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), Injected> {
        self.world.push(Op::Abort(self.index));
        self.open = None;
        if self.world.fail_abort.get() {
            return Err(Injected(self.index, "abort"));
        }
        Ok(())
    }
}

impl Collectives for FakeDevice {
    type Event = Ev;

    fn native_counters(&self) -> NativeCounters {
        NativeCounters {
            graph_instantiations: self.graphs,
            graph_launches: self.launches,
            events: self.events,
            modules: 0,
            peer_copies: 0,
        }
    }

    fn support_bound(&self) -> SupportBound {
        SupportBound {
            graph_bytes: 4096,
            event_bytes: 64,
        }
    }

    fn open_route(&mut self, peer: &Self) -> Result<(), Injected> {
        if self
            .world
            .refused_routes
            .borrow()
            .contains(&(self.index, peer.index))
        {
            return Err(Injected(self.index, "route (no peer access)"));
        }
        Ok(())
    }

    fn create_event(&mut self) -> Result<Ev, Injected> {
        self.events += 1;
        self.world
            .events_created
            .set(self.world.events_created.get() + 1);
        Ok(Rc::new(EventState {
            id: self.world.id(),
            recorded_in: Cell::new(None),
            mark: Cell::new(None),
            world: Rc::clone(&self.world),
        }))
    }

    fn record_event(&mut self, event: &Ev) -> Result<(), Injected> {
        assert!(
            self.open.is_none(),
            "an event is recorded outside a recording"
        );
        event.recorded_in.set(Some(self.world.epoch.get()));
        let enqueued = self.world.enqueued.borrow().get(self.index).copied();
        event.mark.set(Some((self.index, enqueued.unwrap_or(0))));
        self.world.push(Op::Record(self.index, event.id));
        Ok(())
    }

    fn wait_event(&mut self, event: &Ev) -> Result<(), Injected> {
        assert!(
            self.open.is_none(),
            "an event is waited outside a recording"
        );
        if event.recorded_in.get() != Some(self.world.epoch.get()) {
            self.world.push(Op::StaleWait(event.id));
        }
        // Everything the recording device had enqueued before the record has executed once this
        // device's work after the wait runs.
        if let Some((device, upto)) = event.mark.get() {
            self.world.drain(device, upto);
        }
        self.world.push(Op::Wait(self.index, event.id));
        Ok(())
    }

    fn peer_copy(&mut self, src: &Buf, dst: &Buf, bytes: usize) -> Result<(), Injected> {
        self.open
            .as_mut()
            .expect("a peer copy inside a recording")
            .push(Recorded::PeerCopy {
                src: Rc::clone(src),
                dst: Rc::clone(dst),
                bytes,
            });
        Ok(())
    }
}

fn device(world: &Rc<World>, index: usize) -> FakeDevice {
    FakeDevice {
        index,
        world: Rc::clone(world),
        open: None,
        graphs: 0,
        launches: 0,
        events: 0,
        memory: poot_runtime_common::MemoryCounters::new(),
    }
}

/// `LAYERS` blocks in the shape a pipeline partition cuts: one activation slot, one shared
/// constant and one carried state pair per block, each block tagged with its layer.
fn layered() -> (Graph, Arc<WeightStore>, Arc<WeightMap>) {
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
        .finish(poot_graph_ir::Outputs::new(hidden).with_state(&state))
        .unwrap();
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

fn targets(devices: usize) -> TargetSet {
    TargetSet::new(
        (0..devices)
            .map(|index| (DeviceId(index as u32), target()))
            .collect(),
    )
    .unwrap()
}

/// The layered graph cut after layers 0 and 1, stage `i` on `devices[i]`.
fn pipeline(g: &Graph, devices: &[u32]) -> StagedProgram {
    compile_staged(
        g,
        &targets(2),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Pipeline {
                after_layers: vec![0, 1],
                devices: devices.iter().map(|&d| DeviceId(d)).collect(),
            },
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .expect("the layered fixture compiles as a pipeline on two PTX devices")
}

/// A graph whose sole output is one named carried value: reads state through a production entry.
fn state_probe(name: &str) -> Graph {
    let b = Builder::new();
    let state = b.state_input(name, TensorType::f32(vec![4]), StateRole::Recurrent);
    b.finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
        .unwrap()
}

fn probe_program(name: &str, device: u32) -> StagedProgram {
    compile_staged(
        &state_probe(name),
        &targets(2),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(device)),
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .unwrap()
}

fn inputs_for(step: usize) -> [f32; 4] {
    [
        1.5 + step as f32,
        -0.25 * step as f32,
        3.0 - step as f32,
        0.75 + 0.5 * step as f32,
    ]
}

fn step_inputs<'a>(values: &'a [f32; 4], shape: &'a [usize; 1], inputs: &mut StepInputs<'a>) {
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        shape,
        HostView::new(DType::F32, 4, bytemuck::cast_slice(values)).unwrap(),
    );
}

/// One executor, the entry of the pipeline and a probe entry for each carried state value.
struct Rig<E: Executor> {
    exec: E,
    exe: ExecutableId,
    entry: poot_executor::EntryId,
    probes: Vec<poot_executor::EntryId>,
}

/// The ordinary engine running all three stages of the layered pipeline on one device: the oracle
/// the coordinator's cross-device run must equal.
fn reference(world: &Rc<World>) -> Rig<Engine<FakeDevice>> {
    let (g, store, map) = layered();
    let mut exec = Engine::new(device(world, 0));
    let exe = exec.load_weights(store, map).unwrap();
    let entry = exec.add_entry(exe, &pipeline(&g, &[0, 0, 0])).unwrap();
    let probes = (0..LAYERS)
        .map(|layer| {
            exec.add_entry(exe, &probe_program(&format!("k{layer}"), 0))
                .unwrap()
        })
        .collect();
    Rig {
        exec,
        exe,
        entry,
        probes,
    }
}

fn coordinated(world: &Rc<World>, devices: &[u32]) -> Rig<Coordinator<FakeDevice>> {
    let (g, store, map) = layered();
    let mut exec = Coordinator::new(vec![
        (DeviceId(0), device(world, 0)),
        (DeviceId(1), device(world, 1)),
    ])
    .unwrap();
    let exe = exec.load_weights(store, map).unwrap();
    let entry = exec.add_entry(exe, &pipeline(&g, devices)).unwrap();
    // State lives with the stage that owns it: layer `k` on the device of stage `k`.
    let probes = (0..LAYERS)
        .map(|layer| {
            exec.add_entry(exe, &probe_program(&format!("k{layer}"), devices[layer]))
                .unwrap()
        })
        .collect();
    Rig {
        exec,
        exe,
        entry,
        probes,
    }
}

fn run<E: Executor>(rig: &mut Rig<E>, step: usize) -> Vec<u8> {
    let values = inputs_for(step);
    let shape = [4usize];
    let mut inputs = StepInputs::new();
    step_inputs(&values, &shape, &mut inputs);
    rig.exec
        .step(rig.exe, rig.entry, &inputs, &mut NoSync)
        .expect("the step runs")
        .read()
        .expect("the output reads")
}

fn carried<E: Executor>(rig: &mut Rig<E>) -> Vec<Vec<u8>> {
    let probes = rig.probes.clone();
    probes
        .into_iter()
        .map(|probe| {
            rig.exec
                .step(rig.exe, probe, &StepInputs::new(), &mut NoSync)
                .expect("a state probe steps")
                .read()
                .expect("the probe reads")
        })
        .collect()
}

/// SC-007: producer, transfer, consumer in order, one completion wait per device and none between
/// stages, and every output and carried value equal to the ordinary engine's over many steps with
/// changing inputs.
///
/// Mutation: omit the `peer_copy` in the stage seams; the consumer reads an unwritten import buffer
/// and the first output differs. Complete (synchronize) each device after each stage, as a public
/// per-stage `step` would; the wait count per step reads 3 and the "no wait before the last replay"
/// assertion fails.
#[test]
fn a_two_device_step_logs_producer_transfer_consumer_and_one_completion_wait() {
    let world = Rc::new(World::new());
    let mut reference_rig = reference(&world);
    let expected: Vec<Vec<u8>> = (0..4).map(|i| run(&mut reference_rig, i)).collect();
    let expected_state = carried(&mut reference_rig);

    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    world.clear();
    let first = run(&mut rig, 0);
    assert_eq!(first, expected[0], "step 0 output equals the engine's");

    let ops = world.ops();
    let replay_positions: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| matches!(op, Op::Replay(..)).then_some(i))
        .collect();
    let first_wait = ops
        .iter()
        .position(|op| matches!(op, Op::Synchronize(_)))
        .expect("the step completes");
    assert!(
        replay_positions.iter().all(|&at| at < first_wait),
        "every replay is submitted before the first completion wait: {ops:?}"
    );
    assert_eq!(
        replay_positions.len(),
        3,
        "one recording per stage: {ops:?}"
    );
    // Rank 0 holds stages 0 and 2, rank 1 stage 1. First each consumer device marks its enqueued
    // work (`ready`); then, in stage order: the sender waits for the readiness of the device it
    // copies into, replays, and records its event; the consumer waits for that event, replays.
    let log: Vec<&Op> = ops
        .iter()
        .filter(|op| {
            matches!(
                op,
                Op::Replay(..) | Op::Record(..) | Op::Wait(..) | Op::StaleWait(_)
            )
        })
        .collect();
    let [
        Op::Record(1, ready_0),
        Op::Record(0, ready_1),
        Op::Wait(0, waited_ready_0),
        Op::Replay(0, first),
        Op::Record(0, sent_0),
        Op::Wait(1, waited_0),
        Op::Wait(1, waited_ready_1),
        Op::Replay(1, second),
        Op::Record(1, sent_1),
        Op::Wait(0, waited_1),
        Op::Replay(0, third),
    ] = log.as_slice()
    else {
        panic!("readiness, then producer, event, wait, consumer in stage order: {log:?}");
    };
    assert_eq!(
        ready_0, waited_ready_0,
        "stage 0 waits for rank 1's readiness"
    );
    assert_eq!(
        ready_1, waited_ready_1,
        "stage 1 waits for rank 0's readiness"
    );
    assert_eq!(
        sent_0, waited_0,
        "stage 1 waits on the event stage 0 recorded"
    );
    assert_eq!(
        sent_1, waited_1,
        "stage 2 waits on the event stage 1 recorded"
    );
    let mut distinct = vec![ready_0, ready_1, sent_0, sent_1];
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 4, "every ordering point is its own event");
    let ends_in_peer_copy =
        |seams: &[Seam]| matches!(seams.last(), Some(Seam::PeerCopy { bytes: 16 }));
    assert!(
        ends_in_peer_copy(first),
        "stage 0 ends in its transfer: {first:?}"
    );
    assert!(
        ends_in_peer_copy(second),
        "stage 1 ends in its transfer: {second:?}"
    );
    assert!(
        !third
            .iter()
            .any(|seam| matches!(seam, Seam::PeerCopy { .. })),
        "the last stage sends nothing: {third:?}"
    );
    assert_eq!(
        world.count(|op| matches!(op, Op::Synchronize(_))),
        2,
        "one completion wait per device, none between stages: {ops:?}"
    );
    assert_eq!(world.count(|op| matches!(op, Op::StaleWait(_))), 0);

    for (i, want) in expected.iter().enumerate().skip(1) {
        assert_eq!(&run(&mut rig, i), want, "step {i} output");
    }
    assert_eq!(carried(&mut rig), expected_state, "carried state");
}

/// SC-005: rank 0 -> rank 1 -> rank 0 over many generations with changing input bytes: every output
/// and carried value equals the oracle's, no wait is ever submitted ahead of its producer's event for
/// the same step, and nothing native is created after the entry is prepared.
///
/// Mutation: replay the stages in reverse order; the consumer waits on an event the current step has
/// not recorded (`StaleWait`) and reads the previous step's bytes.
#[test]
fn rank_zero_to_rank_one_to_rank_zero_replays_never_consume_a_prior_generation() {
    let world = Rc::new(World::new());
    let mut oracle = reference(&world);
    let expected: Vec<Vec<u8>> = (0..24).map(|i| run(&mut oracle, i)).collect();
    let expected_state = carried(&mut oracle);

    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    assert_eq!(run(&mut rig, 0), expected[0]);
    let after_prepare = rig.exec.counters();
    let allocations = world.allocations.get();
    let events = world.events_created.get();
    for (i, want) in expected.iter().enumerate().skip(1) {
        assert_eq!(&run(&mut rig, i), want, "generation {i} output");
    }
    let now = rig.exec.counters();
    assert_eq!(carried(&mut rig), expected_state, "carried state");
    assert_eq!(
        world.count(|op| matches!(op, Op::StaleWait(_))),
        0,
        "no replay waited on a prior generation's event"
    );
    assert_eq!(
        world.allocations.get(),
        allocations,
        "warm replays and state probes allocate nothing"
    );
    assert_eq!(world.events_created.get(), events);
    for (before, after) in after_prepare.devices.iter().zip(&now.devices) {
        assert_eq!(
            before.native.graph_instantiations, after.native.graph_instantiations,
            "device {:?}: no native graph instantiated after preparation",
            before.device
        );
        assert_eq!(before.native.events, after.native.events);
        assert!(
            after.launches > before.launches,
            "phase launches grow with every replay"
        );
    }
    let launches: u64 = now.devices.iter().map(|d| d.launches).sum();
    assert!(
        launches > 2 * 24 - 2,
        "three phase launches per step on two devices exceed the device count"
    );
}

fn no_open_recording(world: &World) {
    let begins = world.count(|op| matches!(op, Op::Begin(_)));
    let ends = world.count(|op| matches!(op, Op::Finish(_) | Op::Abort(_)));
    assert_eq!(
        begins, ends,
        "every begun recording is finished or aborted ({begins} begins, {ends} ends)"
    );
}

/// SC-006: a recording failure on a later rank aborts every open capture, publishes no partial
/// recording set, and leaves the next valid step able to record the whole entry; a failed abort is an
/// explicit poison refusal for the whole coordinator.
///
/// Mutation: publish each rank's recordings as they are made (a partial set survives the failure);
/// the retried step then holds two recordings of stage 0 and the recordings assertion fails.
#[test]
fn a_recording_failure_on_a_later_rank_publishes_nothing_and_a_retry_records_once() {
    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    // The pipeline's first recording on rank 1 is the second `finish` of the step.
    let base = world.finishes.get();
    world.fail_finish_at.set(Some(base + 2));
    let values = inputs_for(0);
    let shape = [4usize];
    let mut inputs = StepInputs::new();
    step_inputs(&values, &shape, &mut inputs);
    let error = match rig.exec.step(rig.exe, rig.entry, &inputs, &mut NoSync) {
        Err(error) => error,
        Ok(_) => panic!("the injected finish failure must fail the step"),
    };
    assert!(
        matches!(error, ExecError::Device(_)),
        "a device failure, got {error:?}"
    );
    no_open_recording(&world);
    assert_eq!(
        world.count(|op| matches!(op, Op::Abort(1))),
        1,
        "the failing device aborted exactly once"
    );
    assert_eq!(
        rig.exec.stats().recordings,
        0,
        "no partial recording set was published"
    );
    assert_eq!(
        world.count(|op| matches!(op, Op::Replay(..))),
        0,
        "nothing was submitted"
    );

    world.fail_finish_at.set(None);
    let output = run(&mut rig, 0);
    assert_eq!(output.len(), 16);
    let stats = rig.exec.stats();
    let entry_stages: Vec<_> = stats
        .stages
        .iter()
        .filter(|stage| stage.entry == rig.entry)
        .map(|stage| (stage.recordings, stage.replays))
        .collect();
    assert_eq!(
        entry_stages,
        vec![(1, 1); 3],
        "recorded once, replayed once"
    );

    // A failed abort poisons the coordinator: an explicit refusal, never a retry on a dirty device.
    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    let base = world.finishes.get();
    world.fail_finish_at.set(Some(base + 2));
    world.fail_abort.set(true);
    let mut inputs = StepInputs::new();
    step_inputs(&values, &shape, &mut inputs);
    assert!(
        rig.exec
            .step(rig.exe, rig.entry, &inputs, &mut NoSync)
            .is_err()
    );
    world.fail_finish_at.set(None);
    world.fail_abort.set(false);
    let mut inputs = StepInputs::new();
    step_inputs(&values, &shape, &mut inputs);
    let refused = match rig.exec.step(rig.exe, rig.entry, &inputs, &mut NoSync) {
        Err(error) => error,
        Ok(_) => panic!("a poisoned coordinator must refuse"),
    };
    assert!(
        matches!(refused, ExecError::Poisoned),
        "expected the poison refusal, got {refused:?}"
    );
}

/// SC-008, first half: a forced non-peer route is refused before anything is recorded or submitted,
/// and a route only one direction of which is refused still refuses the program.
///
/// Mutation: let `open_route` succeed for a non-peer route (an eager host-staged copy); the program
/// loads and this fails.
#[test]
fn a_non_peer_route_refuses_before_any_recording() {
    let world = Rc::new(World::new());
    world.refused_routes.borrow_mut().push((1, 0));
    let (g, store, map) = layered();
    let mut exec = Coordinator::new(vec![
        (DeviceId(0), device(&world, 0)),
        (DeviceId(1), device(&world, 1)),
    ])
    .unwrap();
    let exe = exec.load_weights(store, map).unwrap();
    let before = world.allocations.get();
    let error = exec
        .add_entry(exe, &pipeline(&g, &[0, 1, 0]))
        .expect_err("rank 1 cannot send to rank 0");
    let ExecError::Coordinator(refused) = &error else {
        panic!("expected a coordinator refusal, got {error:?}");
    };
    assert!(
        matches!(
            **refused,
            CoordinatorError::Route {
                from: DeviceId(1),
                to: DeviceId(0),
                ..
            }
        ),
        "the refused route is named: {refused:?}"
    );
    assert_eq!(world.count(|op| matches!(op, Op::Begin(_))), 0);
    assert_eq!(world.count(|op| matches!(op, Op::Replay(..))), 0);
    assert_eq!(
        world.allocations.get(),
        before,
        "the route is refused before the entry allocates anything"
    );
    assert_eq!(world.events_created.get(), 0);
}

/// SC-008, second half: every device's planned support bound covers what preparing the entry
/// actually created, and a budget below the bound refuses the entry before it allocates.
///
/// Mutation: report no support bytes for a rank (drop its import bytes from the plan); the plan no
/// longer covers the staging bytes the device's own counters gained and this fails. A small free
/// budget that the admission ignores loads the entry and the refusal row fails.
#[test]
fn the_planned_support_covers_the_actual_and_a_small_budget_refuses() {
    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    let _ = run(&mut rig, 0);
    let support = rig.exec.support(rig.exe, rig.entry).unwrap();
    assert_eq!(support.len(), 2, "both devices hold stages");
    let staged: u64 = support.iter().map(|s| s.actual.import_bytes).sum();
    assert!(staged > 0, "transfers deliver into staging: {support:?}");
    for device in &support {
        assert!(
            device.planned.import_bytes >= device.actual.import_bytes,
            "{device:?}: the planned import bytes cover the staging the device gained"
        );
        assert!(
            device.planned.events >= device.actual.events,
            "{device:?}: planned events cover the created events"
        );
        assert!(
            device.planned.graphs >= device.actual.graph_instantiations,
            "{device:?}: planned graphs cover the instantiated graphs"
        );
        assert!(
            device.planned.bound_bytes
                >= device.planned.import_bytes
                    + device.planned.events * 64
                    + device.planned.graphs * 4096,
            "{device:?}: the bound includes the native support"
        );
    }

    let world = Rc::new(World::new());
    world.free_memory.set(Some(8));
    let (g, store, map) = layered();
    let mut exec = Coordinator::new(vec![
        (DeviceId(0), device(&world, 0)),
        (DeviceId(1), device(&world, 1)),
    ])
    .unwrap();
    let exe = exec.load_weights(store, map).unwrap();
    let error = exec
        .add_entry(exe, &pipeline(&g, &[0, 1, 0]))
        .expect_err("8 free bytes cannot hold the support allocations");
    let ExecError::Coordinator(refused) = &error else {
        panic!("expected a coordinator refusal, got {error:?}");
    };
    assert!(
        matches!(**refused, CoordinatorError::SupportBound { free: 8, .. }),
        "{refused:?}"
    );
    assert_eq!(world.allocations.get(), 0, "refused before any allocation");

    // A device that cannot say how much is free is refused too: unknown is not unlimited.
    let world = Rc::new(World::new());
    world.free_memory.set(None);
    let (g, store, map) = layered();
    let mut exec = Coordinator::new(vec![
        (DeviceId(0), device(&world, 0)),
        (DeviceId(1), device(&world, 1)),
    ])
    .unwrap();
    let exe = exec.load_weights(store, map).unwrap();
    let error = exec
        .add_entry(exe, &pipeline(&g, &[0, 1, 0]))
        .expect_err("an unknown free-memory figure cannot bound the support allocations");
    let ExecError::Coordinator(refused) = &error else {
        panic!("expected a coordinator refusal, got {error:?}");
    };
    assert!(
        matches!(**refused, CoordinatorError::UnknownFreeMemory { .. }),
        "{refused:?}"
    );
    assert_eq!(world.allocations.get(), 0);
}

/// SC-004 on the fake: every device has one recording session over 16 unchanged steps and no new
/// native graph after preparation; launches and dispatches are counted independently of the device
/// count.
///
/// Mutation: clear the entry's `recorded` flag every step (re-record per step); the sessions column
/// reads 16 and the instantiation count grows.
#[test]
fn sixteen_unchanged_steps_are_one_recording_session_per_device() {
    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    for i in 0..16 {
        let _ = run(&mut rig, i);
    }
    let counters = rig.exec.counters();
    let by_device = |device: u32| {
        counters
            .devices
            .iter()
            .find(|d| d.device == DeviceId(device))
            .copied()
            .unwrap()
    };
    // The probe entries of `coordinated` are single-stage entries on their own devices; their
    // sessions are not stepped here.
    assert_eq!(by_device(0).sessions, 1, "one session on device 0");
    assert_eq!(by_device(1).sessions, 1, "one session on device 1");
    assert_eq!(by_device(0).phases, 2, "stages 0 and 2 record on device 0");
    assert_eq!(by_device(1).phases, 1);
    assert_eq!(
        by_device(0).native.graph_instantiations,
        2,
        "two native graphs on device 0, instantiated once"
    );
    assert_eq!(by_device(1).native.graph_instantiations, 1);
    assert_eq!(by_device(0).launches, 32, "two phase launches per step");
    assert_eq!(by_device(1).launches, 16);
    assert!(by_device(0).dispatches > 0 && by_device(1).dispatches > 0);
    assert_eq!(by_device(0).transfers + by_device(1).transfers, 32);
    assert_eq!(counters.steps, 16);
}

/// Unloading after a failed step proves every device finished before any is released: the buffers and
/// events a device's graph still addresses on a peer outlive the first device to complete.
///
/// Mutation: unload each device's executable as it is synchronized (release owners on the first
/// rank's completion); a `FreeBuffer` lands before the last `Synchronize` and this fails.
#[test]
fn unload_drains_every_device_before_releasing_any() {
    let world = Rc::new(World::new());
    let mut rig = coordinated(&world, &[0, 1, 0]);
    let _ = run(&mut rig, 0);
    world.clear();
    rig.exec.unload(rig.exe).unwrap();
    let ops = world.ops();
    let last_sync = ops
        .iter()
        .rposition(|op| matches!(op, Op::Synchronize(_)))
        .expect("unload synchronizes");
    let first_free = ops
        .iter()
        .position(|op| matches!(op, Op::FreeBuffer(_) | Op::FreeEvent(_)))
        .expect("unload releases the executable's buffers");
    assert!(
        last_sync < first_free,
        "all devices synchronized ({last_sync}) before the first release ({first_free}): {ops:?}"
    );
    assert_eq!(
        world.count(|op| matches!(op, Op::Synchronize(_))),
        2,
        "each device synchronized once"
    );
}

// ---- POOT-749: a tensor-parallel program on the coordinator ------------------------------------

const HIDDEN: usize = 8;
const INTER: usize = 16;

/// `y = (x @ gate^T) @ down^T` with the weights mapped under the roles the Megatron rule cuts: the gate
/// is column-parallel and the down projection row-parallel, so one all-reduce joins the two ranks.
fn tensor_mlp() -> (Graph, Arc<WeightStore>, Arc<WeightMap>) {
    let gate = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));
    let down = WeightId::layer(0, WeightRole::Ffn(FfnRole::Down));
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![1, HIDDEN]));
    let wg = b.constant(&gate.const_name(), TensorType::f32(vec![INTER, HIDDEN]));
    let wd = b.constant(&down.const_name(), TensorType::f32(vec![HIDDEN, INTER]));
    let hidden = b.matmul(x, b.transpose(wg, vec![1, 0]));
    let y = b.matmul(hidden, b.transpose(wd, vec![1, 0]));
    let g = b.finish(y).unwrap();

    let entry = |rows: usize, cols: usize, seed: f32| {
        let bytes: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (seed + (i % 7) as f32 * 0.25).to_le_bytes())
            .collect();
        WeightEntry::Dense(
            DenseWeight::try_new(DType::F32, vec![rows, cols], StoredBytes::from(bytes)).unwrap(),
        )
    };
    let mut store = WeightStore::builder();
    store.insert("gate", entry(INTER, HIDDEN, 0.5)).unwrap();
    store.insert("down", entry(HIDDEN, INTER, -0.25)).unwrap();
    let store = Arc::new(store.build());
    let mut map = WeightMap::builder(&store);
    map.map(gate, WeightView::Stored("gate".into())).unwrap();
    map.map(down, WeightView::Stored("down".into())).unwrap();
    let map = Arc::new(map.build());
    (g, store, map)
}

fn tensor_program(g: &Graph, map: &Arc<WeightMap>) -> StagedProgram {
    compile_staged(
        g,
        &targets(2),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Tensor {
                rule: poot_graph_plan::ShardRule::Megatron,
                devices: vec![DeviceId(0), DeviceId(1)],
                weights: Arc::clone(map),
            },
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .expect("the MLP shards over two PTX devices")
}

fn tensor_step<E: Executor>(
    exec: &mut E,
    exe: ExecutableId,
    entry: poot_executor::EntryId,
    step: usize,
) {
    let values: Vec<f32> = (0..HIDDEN).map(|i| 0.1 * (i + step) as f32).collect();
    let shape = [1usize, HIDDEN];
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        &shape,
        HostView::new(DType::F32, HIDDEN, bytemuck::cast_slice(&values)).unwrap(),
    );
    exec.step(exe, entry, &inputs, &mut NoSync)
        .expect("the step runs");
}

fn coordinator_over_two(world: &Rc<World>) -> Coordinator<FakeDevice> {
    Coordinator::new(vec![
        (DeviceId(0), device(world, 0)),
        (DeviceId(1), device(world, 1)),
    ])
    .unwrap()
}

/// SC-004: the shares of the weights upload once per load, not per token, and each rank reads only its
/// own share (the two ranks together read the model's bytes once, not twice). The stats read the
/// engines' own counters of bytes read from the store and of live weight allocations.
///
/// Mutation (the driver's): rebuild the executable per token; the counters grow every step. The last
/// block runs that driver, so the row proves its own counter can move.
#[test]
fn tensor_parallel_weights_upload_once_per_load_and_each_rank_reads_only_its_share() {
    let (g, store, map) = tensor_mlp();
    let program = tensor_program(&g, &map);
    let tensor = program.tensor().expect("a tensor-parallel program").clone();
    assert_eq!(tensor.shares.len(), 2, "gate and down are cut");

    let world = Rc::new(World::new());
    let mut exec = coordinator_over_two(&world);
    let exe = exec
        .load_tensor_weights(Arc::clone(&store), Arc::clone(&map), &tensor)
        .unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    tensor_step(&mut exec, exe, entry, 0);

    let model_bytes = 2 * INTER * HIDDEN * 4;
    let stats = exec.stats();
    assert_eq!(
        stats.weight_bytes_read, model_bytes as u64,
        "two ranks read each weight's share once: the model's bytes, not twice"
    );
    let weights = |stats: &poot_executor::ExecutorStats| {
        stats
            .memory
            .iter()
            .find(|(role, _)| *role == BufferRole::Weight)
            .map(|(_, snapshot)| (snapshot.live_bytes, snapshot.allocations))
            .expect("weights are allocated")
    };
    let after_first = weights(&stats);
    assert_eq!(
        after_first.0, model_bytes as u64,
        "live weight bytes are the shares"
    );
    let transfers_after_first: u64 = exec.counters().devices.iter().map(|d| d.transfers).sum();
    assert!(transfers_after_first > 0, "the ranks exchange partial sums");

    for step in 1..9 {
        tensor_step(&mut exec, exe, entry, step);
    }
    let stats = exec.stats();
    assert_eq!(stats.weight_bytes_read, model_bytes as u64, "no re-upload");
    assert_eq!(weights(&stats), after_first, "no weight is re-allocated");
    let transfers: u64 = exec.counters().devices.iter().map(|d| d.transfers).sum();
    assert_eq!(
        transfers,
        9 * transfers_after_first,
        "every step exchanges the same partials"
    );

    // The counter can move: a driver that rebuilds the executable for every token re-reads the store.
    let mut rebuilt = coordinator_over_two(&Rc::new(World::new()));
    let mut read = 0;
    for step in 0..3 {
        let exe = rebuilt
            .load_tensor_weights(Arc::clone(&store), Arc::clone(&map), &tensor)
            .unwrap();
        let entry = rebuilt.add_entry(exe, &program).unwrap();
        tensor_step(&mut rebuilt, exe, entry, step);
        rebuilt.remove_entry(exe, entry).unwrap();
        rebuilt.unload(exe).unwrap();
        read = rebuilt.stats().weight_bytes_read;
    }
    assert!(
        read > model_bytes as u64,
        "a per-token rebuild reads the store again: {read}"
    );
}

/// A rank handed the whole weight map instead of its share is refused by name at load: the program's
/// shapes are the shares'.
///
/// Mutation: bind the whole map on every rank (`load_weights`); this is the refusal checked here, and
/// binding the share on rank 0 only leaves rank 1's refusal.
#[test]
fn a_rank_that_binds_the_whole_weight_is_refused_by_name() {
    let (g, store, map) = tensor_mlp();
    let program = tensor_program(&g, &map);
    let world = Rc::new(World::new());
    let mut exec = coordinator_over_two(&world);
    let exe = exec.load_weights(store, map).unwrap();
    let error = exec
        .add_entry(exe, &program)
        .expect_err("the whole [16, 8] gate is not the rank's [8, 8] share");
    let ExecError::Load(load) = &error else {
        panic!("expected a load refusal, got {error:?}");
    };
    assert!(
        matches!(**load, poot_executor::LoadError::WeightShape { .. }),
        "{error:?}"
    );
}

/// SC-005 on a tensor-parallel program: a forced non-peer route is refused by name before anything is
/// recorded, allocated or submitted, and no eager copy stands in for it.
///
/// Mutation: let `open_route` succeed for the non-peer route (the coordinator ignores the refusal and
/// copies eagerly); the entry loads and this row goes red.
#[test]
fn a_tensor_parallel_program_on_a_non_peer_route_refuses_before_device_work() {
    let (g, store, map) = tensor_mlp();
    let program = tensor_program(&g, &map);
    let tensor = program.tensor().unwrap().clone();
    let world = Rc::new(World::new());
    world.refused_routes.borrow_mut().push((1, 0));
    let mut exec = coordinator_over_two(&world);
    let exe = exec.load_tensor_weights(store, map, &tensor).unwrap();
    let before = world.allocations.get();
    let error = exec
        .add_entry(exe, &program)
        .expect_err("rank 1 cannot send its partial to rank 0");
    let ExecError::Coordinator(refused) = &error else {
        panic!("expected a coordinator refusal, got {error:?}");
    };
    assert!(
        matches!(
            **refused,
            CoordinatorError::Route {
                from: DeviceId(1),
                to: DeviceId(0),
                ..
            }
        ),
        "the refused route is named: {refused:?}"
    );
    assert_eq!(world.count(|op| matches!(op, Op::Begin(_))), 0);
    assert_eq!(world.count(|op| matches!(op, Op::Replay(..))), 0);
    assert_eq!(world.allocations.get(), before, "nothing was allocated");
    assert_eq!(world.events_created.get(), 0);
}
