//! Card 546a core review CR30: the failure-lifecycle tests (SC-018 through SC-022) a real backend
//! cannot drive deterministically - `WgpuDevice::abort` never fails (recording never touches the
//! device, so there is nothing to fail cleanup on), and nothing in this tree yet drives a real
//! `begin`/`dispatch`/`copy`/`finish`/`replay` failure on wgpu hardware. `FaultyDevice` is a minimal
//! second `Device` implementation whose every method independently, deterministically succeeds or
//! fails on command: not a hidden accessor on the contract (the contract is `Executor`/
//! `Engine`, which `FaultyDevice` never modifies), the ordinary way to unit-test generic code against
//! a controllable collaborator. Model-free: no GPU adapter, only real SpirV codegen (`nix develop`'s
//! `llc`) to compile the fixture graph's kernels, which `FaultyDevice` then never actually runs.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    Device, Dispatch, Engine, ExecError, Executor, NoSync, StateScope, StepInputs,
};
use poot_graph_ir::{BinOp, Builder, StateRole, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;

#[derive(Debug, Default, Clone, Copy)]
struct Failures {
    begin: bool,
    dispatch: bool,
    copy: bool,
    finish: bool,
    replay: bool,
    abort: bool,
    /// an injected `synchronize` failure. `classify_as_fault` picks which
    /// of `Engine::step`'s two post-synchronize-error paths it takes: `Some` (a kernel-assert fault,
    /// Z10) when set, `None` (a real device-level failure, e.g. a timeout) when clear.
    synchronize: bool,
    classify_as_fault: bool,
    /// Make `read` return nonzero words: a validation packet read back this way is rejected.
    reads_are_nonzero: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("faulty device: {0} failed (injected)")]
struct FaultyError(&'static str);

/// A handle the test keeps to flip `FaultyDevice`'s injected failures and read its call counters
/// after the device has moved into an `Engine`.
#[derive(Clone, Default)]
struct Control {
    failures: Rc<RefCell<Failures>>,
    aborts: Rc<RefCell<u32>>,
    dispatches: Rc<RefCell<u32>>,
}

struct FaultyDevice {
    control: Control,
    recording: bool,
    /// `Buffer = ()` has no handle whose `Drop` could release an allocation, so this never
    /// decrements: every call just leaks its guard (Card 547a - `Engine::stats().memory` now reads
    /// this instead of its own tally, so the fake device must track what it was asked to allocate).
    memory: poot_runtime_common::MemoryCounters,
}

impl Device for FaultyDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FaultyError;

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
    ) -> Result<(), FaultyError> {
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
    ) -> Result<(), FaultyError> {
        Ok(())
    }

    fn begin(&mut self) -> Result<(), FaultyError> {
        if self.control.failures.borrow().begin {
            return Err(FaultyError("begin"));
        }
        self.recording = true;
        Ok(())
    }

    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FaultyError> {
        *self.control.dispatches.borrow_mut() += 1;
        if self.control.failures.borrow().dispatch {
            return Err(FaultyError("dispatch"));
        }
        Ok(())
    }

    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FaultyError> {
        if self.control.failures.borrow().copy {
            return Err(FaultyError("copy"));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), FaultyError> {
        if self.control.failures.borrow().finish {
            return Err(FaultyError("finish"));
        }
        self.recording = false;
        Ok(())
    }

    fn replay(&mut self, _recording: &()) -> Result<(), FaultyError> {
        if self.control.failures.borrow().replay {
            return Err(FaultyError("replay"));
        }
        Ok(())
    }

    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FaultyError> {
        Ok(())
    }

    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FaultyError> {
        out.fill(if self.control.failures.borrow().reads_are_nonzero {
            1
        } else {
            0
        });
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), FaultyError> {
        if self.control.failures.borrow().synchronize {
            return Err(FaultyError("synchronize"));
        }
        Ok(())
    }

    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), FaultyError> {
        *self.control.aborts.borrow_mut() += 1;
        if self.control.failures.borrow().abort {
            return Err(FaultyError("abort"));
        }
        Ok(())
    }

    fn classify_fault(&self, error: &FaultyError) -> Option<(String, u32)> {
        (self.control.failures.borrow().classify_as_fault && error.0 == "synchronize")
            .then(|| ("faulty_kernel".to_string(), 7))
    }
}

/// A two-state swap-plus-increment fixture (mirrors `poot-wgpu-device`'s own SC-007 fixture): `output = a +
/// s`, `new_a = s + bias`, `new_s = a + bias` - three real dispatches, two non-donated commit copies
/// (neither new state can be written in place before the other reads it), no slot inputs at all, so
/// `step` needs nothing but `StepInputs::new()`.
fn fixture() -> (
    poot_graph_ir::Graph,
    Arc<WeightStore>,
    Arc<poot_quant::weights::WeightMap>,
) {
    let b = Builder::for_tests();
    let a = b
        .state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent)
        .unwrap();
    let s = b
        .state_input("s", TensorType::f32(vec![2]), StateRole::Recurrent)
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![2])).unwrap();
    let output = b.binary(BinOp::Add, a, s).unwrap();
    let new_a = b.binary(BinOp::Add, s, bias).unwrap();
    let new_s = b.binary(BinOp::Add, a, bias).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(output).with_state(&[(a, new_a), (s, new_s)]))
        .unwrap();

    let mut builder = WeightStore::builder();
    builder
        .insert(
            "bias",
            WeightEntry::Dense(
                DenseWeight::try_new(
                    DType::F32,
                    vec![2],
                    StoredBytes::from(vec![0u8, 0, 0x80, 0x3f, 0, 0, 0, 0x40]), // [1.0, 2.0]
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let store = Arc::new(builder.build());
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());
    (g, store, map)
}

fn staged(g: &poot_graph_ir::Graph, target: Target) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("fixture graph compiles for the fake target")
}

fn new_engine() -> (Engine<FaultyDevice>, Control) {
    let control = Control::default();
    let device = FaultyDevice {
        control: control.clone(),
        recording: false,
        memory: poot_runtime_common::MemoryCounters::new(),
    };
    (Engine::new(device), control)
}

/// `engine.step(...)`, asserting it is an `Err` (panicking with `msg` otherwise) without requiring
/// `StepOutputs: Debug` the way `Result::expect_err` would.
fn step_err(
    engine: &mut Engine<FaultyDevice>,
    exe: poot_executor::ExecutableId,
    entry: poot_executor::EntryId,
    msg: &str,
) -> ExecError {
    match engine.step(exe, entry, &StepInputs::new(), &mut NoSync) {
        Ok(_) => panic!("{msg}"),
        Err(error) => error,
    }
}

/// SC-019: inject a failure at `begin`. The engine calls `abort` exactly once, publishes no partial
/// recording, and leaves the attempted recording in no open state (a later step with the failure
/// cleared succeeds cleanly, proving nothing was left half-open).
#[test]
fn begin_failure_aborts_exactly_once_and_leaves_no_open_recording() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().begin = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected begin failure must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1, "abort must run exactly once");

    control.failures.borrow_mut().begin = false;
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("a later step with the failure cleared must succeed cleanly")
        .read()
        .expect("read the now-successful step's output");
}

/// SC-020 (dispatch point): under the contract's only submission mode, a
/// dispatch failure can only happen during the first step's recording pass (every later step only
/// replays) - and recording never touches the device (ADR-0003: the walk records metadata, nothing
/// runs until `replay`), so the failure cannot have mutated state. The engine aborts exactly once,
/// does NOT mark NeedsReset, and leaves `e.recording` unset so a retry with the failure cleared
/// re-records and steps cleanly - the dispatch-point twin of SC-019's begin-point row.
#[test]
fn dispatch_failure_during_recording_aborts_once_and_does_not_mark_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().dispatch = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected dispatch failure during recording must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1, "abort must run exactly once");

    control.failures.borrow_mut().dispatch = false;
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("recording never touched the device, so a retry must record and step cleanly")
        .read()
        .expect("read the retried step's output");
}

/// SC-020 (copy point): same shape as the dispatch-point case, at the two-phase commit's copy
/// instead of a dispatch.
#[test]
fn copy_failure_during_recording_aborts_once_and_does_not_mark_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().copy = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected copy failure during recording must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1);

    control.failures.borrow_mut().copy = false;
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("recording never touched the device, so a retry must record and step cleanly")
        .read()
        .expect("read the retried step's output");
}

/// SC-020 (finish point).
#[test]
fn finish_failure_during_recording_aborts_once_and_does_not_mark_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().finish = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected finish failure during recording must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1);

    control.failures.borrow_mut().finish = false;
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("recording never touched the device, so a retry must record and step cleanly")
        .read()
        .expect("read the retried step's output");
}

/// SC-020 (replay point): the first step records successfully (recording never
/// touches the device, so it cannot itself fail here); the SECOND step replays, and a replay
/// failure really has submitted (partial) work, so it too marks `NeedsReset`.
#[test]
fn replay_failure_aborts_once_and_marks_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("the first replay-mode step records and replays cleanly")
        .read()
        .unwrap();

    control.failures.borrow_mut().replay = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected replay failure must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1);

    control.failures.borrow_mut().replay = false;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "a replay failure really submitted work, so it must mark NeedsReset too",
    );
    assert!(matches!(error, ExecError::NeedsReset(id) if id == exe));
}

/// SC-015: a classifiable fault surfaces as `ExecError::Fault` from
/// `Engine::step` itself, not just from `Device::synchronize`/`classify_fault` driven directly
/// (`poot-llm/tests/wgpu_executor_contract.rs`'s `wgpu_device_classifies_a_kernel_assert_fault_from_
/// synchronize` exercises the real `WgpuDevice::classify_fault` on a real kernel-assert trap, but
/// never calls `step`, so a mutation that made `step` wrap every synchronize error as a plain
/// `ExecError::Device` instead of asking `classify_fault` first would not be caught there). No
/// output is read: `step` returns `Err` before ever constructing a `StepOutputs`. A fault is not a
/// device-level failure (Z10): no abort and no poison, the device stays usable. The replay already
/// committed state before the trap, so the executable is `NeedsReset` until a complete reset.
#[test]
fn synchronize_fault_surfaces_as_exec_error_fault_from_step_with_no_output_read() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().synchronize = true;
    control.failures.borrow_mut().classify_as_fault = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected, classifiable synchronize failure must surface as ExecError::Fault",
    );
    assert!(
        matches!(&error, ExecError::Fault { kernel, code } if kernel == "faulty_kernel" && *code == 7),
        "expected ExecError::Fault{{kernel: \"faulty_kernel\", code: 7}}, got {error:?}"
    );
    assert_eq!(
        *control.aborts.borrow(),
        0,
        "a fault is not a device failure: no abort"
    );

    control.failures.borrow_mut().synchronize = false;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "a faulted step committed state, so the executable must refuse the next step",
    );
    assert!(matches!(error, ExecError::NeedsReset(id) if id == exe));
    engine
        .reset_state(exe, StateScope::All)
        .expect("a complete reset clears NeedsReset: a fault never poisons the device");
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("the reset executable steps again")
        .read()
        .expect("read the now-successful step's output");
}

/// CR30: a non-fault `synchronize` failure (anything `classify_fault` does not
/// name, e.g. a device timeout) is classified exactly like a replay failure - it already ran after
/// a successful replay, so state may have been mutated: abort, and mark NeedsReset since rollback is
/// not proven.
#[test]
fn non_fault_synchronize_failure_aborts_and_marks_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().synchronize = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "an injected, non-fault synchronize failure (e.g. a timeout) must surface",
    );
    assert!(matches!(error, ExecError::Device(_)));
    assert_eq!(*control.aborts.borrow(), 1, "abort must run exactly once");

    control.failures.borrow_mut().synchronize = false;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "a non-fault synchronize failure ran after a successful replay, so it must mark \
         NeedsReset too",
    );
    assert!(matches!(error, ExecError::NeedsReset(id) if id == exe));
}

/// SC-022: a failed abort poisons the device; no `reset_state` can clear that poison, and every
/// later entry point refuses with `Poisoned`, not just the one that failed.
#[test]
fn failed_abort_poisons_the_device_and_reset_cannot_clear_it() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().dispatch = true;
    control.failures.borrow_mut().abort = true;
    step_err(
        &mut engine,
        exe,
        entry,
        "the injected dispatch failure must surface",
    );

    control.failures.borrow_mut().dispatch = false;
    control.failures.borrow_mut().abort = false;
    let error = engine
        .reset_state(exe, StateScope::All)
        .expect_err("a poisoned device must refuse reset_state, not clear the poison");
    assert!(
        matches!(error, ExecError::Poisoned),
        "expected Poisoned, got {error:?}"
    );
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "a poisoned device must refuse every later step too",
    );
    assert!(matches!(error, ExecError::Poisoned));
}

/// SC-018: the dense fixture's two carried states live in one engine buffer each, shared by name
/// across a second entry on the same executable rather than ping-ponged into a second
/// pair - `stats().memory` reports the same 16 bytes of `State`-role buffers whether one or two
/// entries reference `a`/`s`.
#[test]
fn carried_state_has_no_ping_pong_pair() {
    let (mut engine, _control) = new_engine();
    let (g, store, map) = fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    // A second entry over the identical graph/state pair must share the same two state buffers
    // (Z5: name/aval/storage match), not allocate its own second pair.
    let entry2 = engine.add_entry(exe, &staged(&g, target)).unwrap();
    engine
        .step(exe, entry2, &StepInputs::new(), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();

    let stats = engine.stats();
    // `FaultyDevice::allocate` records `elems * storage.element().byte_width()` under the role the
    // engine passed it, so this total is meaningful even on the fake device: two f32[2] states is
    // exactly 16 bytes (2 states * 2 elems * 4 bytes) with one buffer each; a ping-ponged pair would
    // double it to 32.
    let state_bytes = stats
        .memory
        .iter()
        .find(|&&(role, _)| role == BufferRole::State)
        .map(|&(_, snapshot)| snapshot.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        state_bytes, 16,
        "two f32[2] states with no ping-pong pair must total 16 bytes, stats={:?}",
        stats.memory
    );
}

/// A graph with one validation witness (`1` where `a` is unordered with itself, else `0`) and no
/// constants: `output = a + s`, witness over `a`.
fn validated_fixture() -> (
    poot_graph_ir::Graph,
    Arc<WeightStore>,
    Arc<poot_quant::weights::WeightMap>,
) {
    let b = Builder::for_tests();
    let a = b
        .state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent)
        .unwrap();
    let s = b
        .state_input("s", TensorType::f32(vec![2]), StateRole::Recurrent)
        .unwrap();
    let output = b.binary(BinOp::Add, a, s).unwrap();
    let ordered = b.binary(BinOp::Ge, a, a).unwrap();
    let flipped = b
        .binary_scalar(BinOp::Mul, ordered, poot_graph_ir::Scalar::F32(-1.0))
        .unwrap();
    let unordered = b
        .binary_scalar(BinOp::Add, flipped, poot_graph_ir::Scalar::F32(1.0))
        .unwrap();
    let g = poot_graph_ir::test_support::graph_fixtures::finish_with_state_and_validations(
        b,
        output,
        &[(a, output), (s, output)],
        &[(poot_graph_ir::ValidationId(1), "unordered", unordered)],
    )
    .unwrap();
    let store = Arc::new(WeightStore::builder().build());
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());
    (g, store, map)
}

/// A step whose validation packet is rejected committed its state before the check, so the executable
/// is `NeedsReset`: the next step is refused, a complete reset clears it, and the device is not
/// poisoned. Mutation: delete `mark_needs_reset` from the rejected-validation branch of
/// `Engine::complete`; the second step runs instead of being refused (red).
#[test]
fn a_rejected_validation_packet_marks_needs_reset() {
    let (mut engine, control) = new_engine();
    let (g, store, map) = validated_fixture();
    let exe = engine.load_weights(store, map).unwrap();
    let target = engine.device().target();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();

    control.failures.borrow_mut().reads_are_nonzero = true;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "a nonzero witness lane must be rejected",
    );
    assert!(matches!(error, ExecError::Validation(_)), "got {error:?}");
    assert_eq!(
        *control.aborts.borrow(),
        0,
        "a rejected packet is not a device failure"
    );

    control.failures.borrow_mut().reads_are_nonzero = false;
    let error = step_err(
        &mut engine,
        exe,
        entry,
        "the rejected step committed state, so the executable must refuse the next step",
    );
    assert!(
        matches!(error, ExecError::NeedsReset(id) if id == exe),
        "got {error:?}"
    );
    engine
        .reset_state(exe, StateScope::All)
        .expect("a complete reset clears NeedsReset");
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .expect("the reset executable steps again");
}
