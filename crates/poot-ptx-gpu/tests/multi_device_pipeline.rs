//! POOT-748 on two NVIDIA devices: the captured coordinator runs a layer pipeline across GPUs 0 and 1
//! and agrees with the single-device program.
//!
//! Each row loads the same model twice, once as one stage on one device through an ordinary
//! [`Engine`] and once as a [`Coordinator`] over two [`PtxDevice`]s with the layer cut between them,
//! steps both over the same tokens and compares every logit. A boundary value crosses devices as a
//! peer copy recorded in the producing stage's CUDA graph, ordered by a stream event the consuming
//! device waits for before its graph launches, so a skipped copy or a stage run before its producer
//! changes the logits. The coordinator's own counters assert what the device runtime created once and
//! launched every step.
//!
//! Skips cleanly with fewer than two NVIDIA GPUs (and fails loudly under `POOT_REQUIRE_PTX=1`).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    BufferRole, Collectives, Coordinator, CoordinatorCounters, Device, DeviceTime, Dispatch,
    Engine, ExecError, Executor, NativeCounters, NoSync, SupportBound,
};
use poot_executor_parity::dense::{Dense, Family};
use poot_executor_parity::weight_map::{
    MappedModel, Step, TIER2, assert_close, decode_steps, trace,
};
use poot_graph_ir::{IndexRange, Slot};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_models::model::Phase;
use poot_models::registry::Registry;
use poot_ptx_gpu::{PtxDevice, PtxGpuError};
use poot_runtime_common::DeviceBackend;

/// The qwen2/minimax/glm tokens every row decodes: distinct so a stale or missing boundary value
/// changes the logits.
const TOKENS: [i32; 6] = [3, 11, 7, 29, 5, 17];

/// The cache capacity every row traces with: at least the longest run any row drives (24 steps), so
/// every position stays inside the `Pos` slot's declared range `[0, CAPACITY)`.
const CAPACITY: usize = 32;

/// Token-by-token decode of `ids` from position 0, refusing a run that would leave the traced
/// capacity (a position past it is rejected at bind, and the row would test the rejection).
fn decode(ids: &[i32]) -> Vec<Step> {
    assert!(
        ids.len() <= CAPACITY,
        "{} steps exceed the traced capacity {CAPACITY}",
        ids.len()
    );
    decode_steps(ids)
}

/// Both devices, or a skip. A host with one GPU skips every row here (the single-GPU PTX lane runs this
/// binary too); `POOT_REQUIRE_PTX_2GPU=1`, which `just test-device-ptx-2gpu` sets, makes a missing
/// second GPU a failure instead.
fn open_two() -> Option<(PtxDevice, PtxDevice)> {
    let first = poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::open(0))?;
    match PtxDevice::open(1) {
        Ok(second) => Some((first, second)),
        Err(error) => {
            assert!(
                std::env::var_os("POOT_REQUIRE_PTX_2GPU").is_none(),
                "a second NVIDIA device is required (POOT_REQUIRE_PTX_2GPU) and unavailable: {error}"
            );
            eprintln!("SKIP: a second NVIDIA device is unavailable: {error}");
            None
        }
    }
}

fn compile(
    g: &poot_graph_ir::Graph,
    targets: &TargetSet,
    devices: DevicePlacement,
) -> StagedProgram {
    compile_staged(
        g,
        targets,
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices,
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .unwrap_or_else(|error| panic!("compile_staged: {error}"))
}

fn targets(first: Target, second: Target) -> TargetSet {
    TargetSet::new(vec![(DeviceId(0), first), (DeviceId(1), second)]).unwrap()
}

/// `model`'s decode logits for `steps` on one device, all stages of `placement` on device 0.
fn single_device(
    device: PtxDevice,
    mapped: &MappedModel,
    steps: &[Step],
    after_layers: &[usize],
) -> Vec<Vec<f32>> {
    let target = device.target();
    let mut engine = Engine::new(device);
    let exe = engine
        .load_weights(Arc::clone(&mapped.store), Arc::clone(&mapped.map))
        .unwrap();
    let set = TargetSet::single(DeviceId(0), target);
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let placement = if after_layers.is_empty() {
        DevicePlacement::Single(DeviceId(0))
    } else {
        DevicePlacement::Pipeline {
            after_layers: after_layers.to_vec(),
            devices: vec![DeviceId(0); after_layers.len() + 1],
        }
    };
    let entry = engine
        .add_entry(exe, &compile(&g, &set, placement))
        .unwrap();
    steps
        .iter()
        .map(|step| {
            engine
                .step(exe, entry, &step.inputs(), &mut NoSync)
                .expect("a single-device step")
                .to_host()
                .expect("logits read")
                .as_f32()
                .expect("f32 logits")
                .to_vec()
        })
        .collect()
}

/// A coordinator that has run some steps of one entry.
struct Pipelined<D: Collectives> {
    logits: Vec<Vec<f32>>,
    counters: CoordinatorCounters,
    coordinator: Coordinator<D>,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
}

/// The same steps through a coordinator over `devices`, stage `i` on `placement[i]`.
fn pipelined<D: Collectives>(
    devices: Vec<(DeviceId, D)>,
    mapped: &MappedModel,
    steps: &[Step],
    after_layers: &[usize],
    placement: &[u32],
) -> Pipelined<D> {
    let set = TargetSet::new(
        devices
            .iter()
            .map(|(id, device)| (*id, device.target()))
            .collect(),
    )
    .unwrap();
    let mut coordinator = Coordinator::new(devices).unwrap();
    let exe = coordinator
        .load_weights(Arc::clone(&mapped.store), Arc::clone(&mapped.map))
        .unwrap();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let program = compile(
        &g,
        &set,
        DevicePlacement::Pipeline {
            after_layers: after_layers.to_vec(),
            devices: placement.iter().map(|&d| DeviceId(d)).collect(),
        },
    );
    let entry = coordinator.add_entry(exe, &program).unwrap();
    let logits = steps
        .iter()
        .map(|step| {
            coordinator
                .step(exe, entry, &step.inputs(), &mut NoSync)
                .expect("a two-device step")
                .to_host()
                .expect("logits read")
                .as_f32()
                .expect("f32 logits")
                .to_vec()
        })
        .collect();
    let counters = coordinator.counters();
    Pipelined {
        logits,
        counters,
        coordinator,
        exe,
        entry,
    }
}

fn mapped_fixture(model_type: &str) -> MappedModel {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.keys.iter().any(|&(_, key)| key == model_type))
        .unwrap_or_else(|| panic!("no family registered for {model_type}"));
    let fixture = (entry.fixture)();
    let model = (entry.build)(&fixture.raw(), &fixture.store)
        .unwrap_or_else(|error| panic!("{model_type} fixture: {error}"));
    let map = Arc::new(model.weights().clone());
    MappedModel {
        model,
        store: Arc::new(fixture.store),
        map,
    }
}

fn assert_non_constant(logits: &[Vec<f32>], label: &str) {
    assert!(
        logits.iter().flatten().all(|value| value.is_finite()),
        "{label}: a non-finite logit makes the comparison meaningless"
    );
    assert!(
        logits.windows(2).any(|pair| pair[0] != pair[1]),
        "{label}: constant logits across steps prove nothing about the boundary values"
    );
}

/// One fixture, two devices: every logit of every step within the tier-2 tolerance of the
/// single-device program's.
fn assert_two_device_matches_single(mapped: &MappedModel, after_layers: &[usize], label: &str) {
    let Some((first, second)) = open_two() else {
        return;
    };
    let steps = decode(&TOKENS);
    let want = single_device(PtxDevice::open(0).unwrap(), mapped, &steps, &[]);
    assert_non_constant(&want, label);
    let mut placement = vec![0u32; after_layers.len() + 1];
    for (stage, slot) in placement.iter_mut().enumerate() {
        *slot = (stage % 2) as u32;
    }
    let run = pipelined(
        vec![(DeviceId(0), first), (DeviceId(1), second)],
        mapped,
        &steps,
        after_layers,
        &placement,
    );
    for (step, (got, want)) in run.logits.iter().zip(&want).enumerate() {
        assert_close(got, want, TIER2, &format!("{label}: step {step}"));
    }
    for device in &run.counters.devices {
        assert!(
            device.sessions >= 1 && device.launches >= 1,
            "{label}: device {:?} took part: {device:?}",
            device.device
        );
    }
}

/// SC-001: MiniMax-M2's fixture (POOT-570) as a two-device pipeline matches single-device decode.
///
/// Mutation: skip one stage-boundary transfer (the coordinator's seams record the event but no peer
/// copy); the consumer reads a never-written import buffer and the logits diverge.
#[test]
fn minimax_m2_fixture_two_device_pipeline_matches_single_device_decode() {
    assert_two_device_matches_single(&mapped_fixture("minimax_m2"), &[0], "minimax_m2");
}

/// SC-002: qwen2's two-stage split matches single-device decode.
///
/// Mutation: swap the stage order (replay the consuming stage before its producer); the consumer
/// waits on the previous step's event and reads the previous step's boundary.
#[test]
fn qwen2_two_stage_split_matches_single_device_decode() {
    let mapped = Dense::new(Family::Qwen2).model();
    assert_two_device_matches_single(&mapped, &[0], "qwen2");
}

/// SC-003: GLM-5.3's fixture (POOT-569) matches single-device decode.
///
/// Mutation: drop one transfer row from the program; its boundary is never delivered and the
/// logits diverge.
#[test]
fn glm_5_3_fixture_two_device_pipeline_matches_single_device_decode() {
    assert_two_device_matches_single(&mapped_fixture("glm5_next"), &[1], "glm5_next");
}

/// SC-004: over 16 unchanged steps each device has one recording session and no native graph is
/// instantiated after the first step; launches and dispatches are reported independently of the
/// device count, and the planned support bound covers what preparing created.
///
/// Mutation: re-record every step; the session count reads 16 and the instantiation counter grows.
#[test]
fn sixteen_unchanged_steps_are_one_recording_session_per_device_on_ptx() {
    let Some((first, second)) = open_two() else {
        return;
    };
    let mapped = Dense::new(Family::Qwen2).dims(64, 96, 3).model();
    let steps = decode(&(0..16).map(|i| (i * 5 + 1) % 48).collect::<Vec<_>>());
    let Pipelined {
        counters: after_first,
        mut coordinator,
        exe,
        entry,
        ..
    } = pipelined(
        vec![(DeviceId(0), first), (DeviceId(1), second)],
        &mapped,
        &steps[..1],
        &[0, 1],
        &[0, 1, 0],
    );
    for step in &steps[1..] {
        coordinator
            .step(exe, entry, &step.inputs(), &mut NoSync)
            .expect("a warm step");
    }
    let after_sixteen = coordinator.counters();
    for (before, after) in after_first.devices.iter().zip(&after_sixteen.devices) {
        assert_eq!(
            after.sessions, 1,
            "device {:?}: one recording session",
            after.device
        );
        assert_eq!(
            after.native.graph_instantiations, before.native.graph_instantiations,
            "device {:?}: no native graph instantiated after preparation",
            after.device
        );
        assert_eq!(after.native.events, before.native.events);
        assert_eq!(after.native.modules, before.native.modules);
        assert!(after.launches > before.launches);
    }
    let launches: u64 = after_sixteen.devices.iter().map(|d| d.launches).sum();
    assert!(
        launches > 2 * 16,
        "{launches} phase launches over two devices in 16 steps: native launches exceed the device count"
    );
    for support in coordinator.support(exe, entry).unwrap() {
        assert!(
            support.planned.import_bytes >= support.actual.import_bytes
                && support.planned.events >= support.actual.events
                && support.planned.graphs >= support.actual.graph_instantiations,
            "{support:?}: the planned bound covers the actual"
        );
    }
}

/// SC-005: rank 0 -> rank 1 -> rank 0 with changing input bytes for many generations: every logit
/// equals the same three stages run on one device (identical stage kernels), and after preparation
/// nothing native is created.
///
/// Mutation: enqueue a dependent wait before its producer's event generation (replay the stages
/// consumer first); the consumer consumes the previous generation and the logits differ.
#[test]
fn rank_zero_to_rank_one_to_rank_zero_never_consumes_a_prior_generation_on_ptx() {
    let Some((first, second)) = open_two() else {
        return;
    };
    let mapped = Dense::new(Family::Qwen2).dims(64, 96, 3).model();
    let tokens: Vec<i32> = (0..24).map(|i| (i * 7 + 2) % 48).collect();
    let steps = decode(&tokens);
    let want = single_device(PtxDevice::open(0).unwrap(), &mapped, &steps, &[0, 1]);
    assert_non_constant(&want, "three-stage reference");
    let Pipelined {
        logits,
        counters: prepared,
        mut coordinator,
        exe,
        entry,
    } = pipelined(
        vec![(DeviceId(0), first), (DeviceId(1), second)],
        &mapped,
        &steps[..1],
        &[0, 1],
        &[0, 1, 0],
    );
    let memory_after_prepare = coordinator.stats().memory;
    let mut got = logits;
    for step in &steps[1..] {
        got.push(
            coordinator
                .step(exe, entry, &step.inputs(), &mut NoSync)
                .expect("a warm generation")
                .to_host()
                .expect("logits read")
                .as_f32()
                .expect("f32 logits")
                .to_vec(),
        );
    }
    for (generation, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            got, want,
            "generation {generation}: the same stage kernels agree bit for bit"
        );
    }
    let counters = coordinator.counters();
    assert_eq!(counters.steps, 24);
    for (before, after) in prepared.devices.iter().zip(&counters.devices) {
        assert_eq!(
            (
                after.native.graph_instantiations,
                after.native.events,
                after.native.modules
            ),
            (
                before.native.graph_instantiations,
                before.native.events,
                before.native.modules
            ),
            "device {:?}: nothing native is created after preparation",
            after.device
        );
        assert!(after.launches > before.launches);
    }
    let allocations = |memory: &[(
        poot_executor::BufferRole,
        poot_executor::MemoryCounterSnapshot,
    )]| {
        memory
            .iter()
            .map(|(role, snapshot)| (*role, snapshot.allocations))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        allocations(&coordinator.stats().memory),
        allocations(&memory_after_prepare),
        "no buffer is allocated after preparation"
    );
    let device_0 = &counters.devices[0];
    assert_eq!(device_0.phases, 2, "stages 0 and 2 record on device 0");
    assert_eq!(
        device_0.native.graph_instantiations, 2,
        "two native graphs, instantiated once each"
    );
    assert_eq!(device_0.launches, 48, "two phase launches per step");
}

/// A [`PtxDevice`] that fails its `n`th `finish` after opening the capture, so the engine's abort has
/// a real native capture to close.
struct FailingFinish {
    inner: PtxDevice,
    finishes: Rc<Cell<usize>>,
    fail_at: Rc<Cell<Option<usize>>>,
}

impl Device for FailingFinish {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = PtxGpuError;

    fn target(&self) -> Target {
        self.inner.target()
    }

    fn memory(&self) -> Vec<(BufferRole, poot_executor::MemoryCounterSnapshot)> {
        self.inner.memory()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        self.inner.free_memory_bytes()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, PtxGpuError> {
        self.inner.allocate(role, storage, elems)
    }

    fn load_kernel(
        &mut self,
        key: &str,
        kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<Self::Kernel, PtxGpuError> {
        self.inner.load_kernel(key, kernel)
    }

    fn begin(&mut self, submission: Submission) -> Result<(), PtxGpuError> {
        self.inner.begin(submission)
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), PtxGpuError> {
        let inputs: Vec<poot_executor::Arg<'_, PtxDevice>> = d
            .inputs
            .iter()
            .map(|arg| poot_executor::Arg {
                buffer: arg.buffer,
                elems: arg.elems,
            })
            .collect();
        self.inner.dispatch(Dispatch {
            kernel: d.kernel,
            inputs: &inputs,
            output: poot_executor::Arg {
                buffer: d.output.buffer,
                elems: d.output.elems,
            },
            launch: d.launch,
            work: d.work,
        })
    }

    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), PtxGpuError> {
        self.inner.copy(src, dst)
    }

    fn finish(&mut self) -> Result<Option<Self::Recording>, PtxGpuError> {
        let finishes = self.finishes.get() + 1;
        self.finishes.set(finishes);
        if self.fail_at.get() == Some(finishes) {
            // The capture is still open: the engine's abort must close it natively.
            return Err(PtxGpuError::Graph("injected finish failure".into()));
        }
        self.inner.finish()
    }

    fn begin_step(&mut self) -> Result<(), PtxGpuError> {
        self.inner.begin_step()
    }

    fn replay(&mut self, recording: &Self::Recording) -> Result<(), PtxGpuError> {
        self.inner.replay(recording)
    }

    fn write_with(
        &mut self,
        dst: &Self::Buffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), PtxGpuError> {
        self.inner.write_with(dst, offset, len, fill)
    }

    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), PtxGpuError> {
        self.inner.read(src, out)
    }

    fn synchronize(&mut self) -> Result<(), PtxGpuError> {
        self.inner.synchronize()
    }

    fn device_time(&self) -> DeviceTime {
        self.inner.device_time()
    }

    fn abort(&mut self) -> Result<(), PtxGpuError> {
        self.inner.abort()
    }
}

impl Collectives for FailingFinish {
    type Event = <PtxDevice as Collectives>::Event;

    fn native_counters(&self) -> NativeCounters {
        self.inner.native_counters()
    }

    fn support_bound(&self) -> SupportBound {
        self.inner.support_bound()
    }

    fn open_route(&mut self, peer: &Self) -> Result<(), PtxGpuError> {
        self.inner.open_route(&peer.inner)
    }

    fn create_event(&mut self) -> Result<Self::Event, PtxGpuError> {
        self.inner.create_event()
    }

    fn record_event(&mut self, event: &Self::Event) -> Result<(), PtxGpuError> {
        self.inner.record_event(event)
    }

    fn wait_event(&mut self, event: &Self::Event) -> Result<(), PtxGpuError> {
        self.inner.wait_event(event)
    }

    fn peer_copy(
        &mut self,
        src: &Self::Buffer,
        dst: &Self::Buffer,
        bytes: usize,
    ) -> Result<(), PtxGpuError> {
        self.inner.peer_copy(src, dst, bytes)
    }
}

/// SC-006: a capture that fails on the later rank aborts every open capture, publishes no partial
/// recording set, and the next valid step on the same coordinator records and runs, with logits equal
/// to a coordinator that never failed.
///
/// Mutation: leave the capture open when the engine aborts (skip `PtxDevice::abort`'s
/// `abort_capture`), or publish rank 0's recordings before rank 1's succeed; the retried step is
/// refused with a capture still open or records a stage twice.
#[test]
fn a_capture_failure_on_the_later_rank_aborts_cleanly_and_the_retry_runs_on_ptx() {
    let Some((first, second)) = open_two() else {
        return;
    };
    let mapped = Dense::new(Family::Qwen2).model();
    let steps = decode(&TOKENS[..3]);
    let want = single_device(PtxDevice::open(0).unwrap(), &mapped, &steps, &[0]);

    let fail_at = Rc::new(Cell::new(Some(1)));
    let wrap = |inner: PtxDevice| FailingFinish {
        inner,
        finishes: Rc::new(Cell::new(0)),
        fail_at: Rc::new(Cell::new(None)),
    };
    let first = wrap(first);
    let mut second = wrap(second);
    second.fail_at = Rc::clone(&fail_at);

    let set = targets(first.target(), second.target());
    let mut coordinator =
        Coordinator::new(vec![(DeviceId(0), first), (DeviceId(1), second)]).unwrap();
    let exe = coordinator
        .load_weights(Arc::clone(&mapped.store), Arc::clone(&mapped.map))
        .unwrap();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let program = compile(
        &g,
        &set,
        DevicePlacement::Pipeline {
            after_layers: vec![0],
            devices: vec![DeviceId(0), DeviceId(1)],
        },
    );
    let entry = coordinator.add_entry(exe, &program).unwrap();
    let error = match coordinator.step(exe, entry, &steps[0].inputs(), &mut NoSync) {
        Err(error) => error,
        Ok(_) => panic!("the injected failure on rank 1's first recording must fail the step"),
    };
    assert!(matches!(error, ExecError::Device(_)), "{error:?}");
    assert_eq!(
        coordinator.stats().recordings,
        0,
        "no partial recording set was published"
    );
    assert_eq!(coordinator.counters().devices[0].sessions, 0);

    fail_at.set(None);
    for (index, step) in steps.iter().enumerate() {
        let got = coordinator
            .step(exe, entry, &step.inputs(), &mut NoSync)
            .unwrap_or_else(|error| panic!("retry step {index}: {error}"))
            .to_host()
            .unwrap()
            .as_f32()
            .unwrap()
            .to_vec();
        assert_close(&got, &want[index], TIER2, &format!("retry step {index}"));
    }
    let recorded: Vec<(u64, u64)> = coordinator
        .stats()
        .stages
        .iter()
        .map(|stage| (stage.recordings, stage.replays))
        .collect();
    assert_eq!(
        recorded,
        vec![(1, 3); 2],
        "each stage recorded once after the retry"
    );
}

/// The primitive under every transfer row: a peer copy recorded in a capture on device 0 delivers
/// exactly the bytes of a device-0 buffer into a device-1 buffer when the graph launches, and a stream
/// event orders device 1's next work after it.
///
/// Mutation: record the copy against the wrong buffer pair (swap `src` and `dst`); device 1's
/// buffer stays zero.
#[test]
fn a_peer_copy_in_a_graph_delivers_its_bytes_and_an_event_orders_the_peer_on_ptx() {
    let Some((first, second)) = open_two() else {
        return;
    };
    let (zero, one) = (first.context(), second.context());
    let storage = poot_ptx_runtime::BufferStorage::f32();
    let source = zero
        .alloc_storage(poot_ptx_runtime::BufferRole::Activation, storage, 64)
        .unwrap();
    let destination = one
        .alloc_storage(poot_ptx_runtime::BufferRole::Staging, storage, 64)
        .unwrap();
    let pattern: Vec<u8> = (0..256u32).map(|i| (i * 7 + 3) as u8).collect();
    zero.write_with(&source, 0, 256, &mut |range, out| {
        out.copy_from_slice(&pattern[range]);
    })
    .unwrap();
    zero.synchronize().unwrap();
    one.synchronize().unwrap();

    zero.enable_peer_access(one).unwrap();
    let event = zero.create_sync_event().unwrap();
    zero.begin_capture().unwrap();
    zero.copy_to_peer(&source, &destination, 256).unwrap();
    let graph = zero.end_capture().unwrap();
    zero.launch_timed(&graph).unwrap();
    zero.record_sync_event(&event).unwrap();
    one.wait_sync_event(&event).unwrap();
    one.synchronize().unwrap();

    let mut got = vec![0u8; 256];
    one.read_bytes(&destination, &mut got).unwrap();
    assert_eq!(got, pattern, "the peer copy delivered the source bytes");
}

/// POOT-1124 (POOT-1113): the split-KV decode attention in each stage of a two-device layer pipeline.
/// Layer 0 attends on GPU 0 and layer 1 on GPU 1, each over a 1024-key cache of one key/value head with the
/// live length `Slot::Pos` bounds; layer 0's output crosses the cut as a peer copy and layer 1 adds it to its
/// own. Each stage must plan the split schedule (two launches, one scratch) on its own device's caps, and
/// every step of the live lengths 1024, 5, 400, 1, 1024 must equal the CPU oracle's sum of the two layers.
///
/// Mutation: in `poot-kernelgen`'s `attention_body`, skip the numerator term of split 0 in the combine; the
/// first step goes red. Skip the `(m, l)` publication of a split with no key; the length-5 step goes red.
/// Deliver zero bytes at the stage boundary (`peer_copy` of `bytes.min(0)`); the first step goes red.
#[test]
fn split_kv_decode_in_each_stage_of_a_two_device_pipeline_matches_the_oracle_on_ptx() {
    let Some((first, second)) = open_two() else {
        return;
    };
    let ids = [DeviceId(0), DeviceId(1)];
    let set = targets(first.target(), second.target());
    let mut coordinator = Coordinator::new(vec![(ids[0], first), (ids[1], second)]).unwrap();
    poot_executor_parity::attention::assert_split_kv_pipeline_matches_the_oracle(
        &mut coordinator,
        &set,
        ids,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

/// Model-free twin of the long-run rows on a device-less box: the fixture traced over
/// [`CAPACITY`] declares its `Pos` slot `[0, CAPACITY)`, and every position the 24-step
/// run drives is inside it, so the device rows exercise the programs, not the bind rejection.
///
/// Mutation: `CAPACITY = 8` (the old shared capacity); `decode` refuses the run with "N steps exceed
/// the traced capacity 8" (with that guard removed, the range check names position 8).
#[test]
fn the_long_run_stays_inside_the_traced_position_range() {
    let mapped = Dense::new(Family::Qwen2).dims(64, 96, 3).model();
    let g = trace(&*mapped.model, &mapped.map, Phase::Decode, 1, CAPACITY);
    let pos = g
        .slots()
        .iter()
        .find(|(_, slot)| *slot == Slot::Pos)
        .map(|(id, _)| g.meta(*id).range())
        .expect("a decode trace has a Pos slot")
        .expect("the production tracer declares the Pos range");
    assert_eq!(pos, IndexRange::below(CAPACITY).unwrap());
    for step in &decode(&(0..24).map(|i| (i * 7 + 2) % 48).collect::<Vec<_>>()) {
        let last = step.start + step.ids.len() - 1;
        assert!(
            pos.contains(last as i32),
            "position {last} is outside the declared Pos range {pos:?}"
        );
    }
}
