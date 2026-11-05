//! Card 547b acceptance: the decoupled-length contract (SC-002) and the arena byte-accounting
//! contract (SC-003, SC-004), driven through the real `Engine::load_entry`/`step` entry point
//! against a controllable fake `Device` (mirroring `tests/cr30_lifecycle.rs`'s `FaultyDevice`).
//! Model-free: no GPU adapter, only real SpirV codegen to compile the fixture's kernels, which the
//! fake device then never actually runs - this file checks the *engine's own* bookkeeping (what it
//! asks `Device::allocate`/`Device::dispatch` for), not a backend's numeric result (that half is
//! `poot-wgpu-device/tests/buffer_plan.rs`, on real wgpu hardware).
//!
//! Fixture (compiled through the full pipeline; its reductions drop their axis, which no row region takes in, so the
//! program is four dispatches):
//!
//! ```text
//! a, b: const [K]            K = 64
//! v1  = a + b                 // eqn 0: [K]
//! s1  = reduce_sum(v1, axis0) // eqn 1: one element; v1 dies
//! c:    const [1]
//! s2  = reduce_sum(c, axis0)  // eqn 2: one element; reuses v1's freed [K] arena slot
//! out = s1 + s2                // eqn 3
//! ```

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    Arg, Device, Dispatch, DispatchError, Engine, ExecError, Executor, NoSync, StepInputs,
};
use poot_graph_ir::op::RedOp;
use poot_graph_ir::{BinOp, Builder, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::Backend;
use poot_tensor::DType;

type Graph = poot_graph_ir::Graph;

const K: usize = 64;

const REPLAY: CompileOptions = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);

fn fixture() -> (Graph, Arc<WeightStore>, Arc<poot_weights::WeightMap>) {
    let b = Builder::for_tests();
    let ta = b.constant("a", TensorType::f32(vec![K])).unwrap();
    let tb = b.constant("b", TensorType::f32(vec![K])).unwrap();
    let v1 = b.binary(BinOp::Add, ta, tb).unwrap();
    // The reductions drop their axis: a row region takes in only a keepdim reduction (and, since POOT-1456, the
    // pointwise feeding it), so each equation here stays its own dispatch.
    let s1 = b.reduce(RedOp::Sum, v1, 0, false).unwrap();
    let tc = b.constant("c", TensorType::f32(vec![1])).unwrap();
    let s2 = b.reduce(RedOp::Sum, tc, 0, false).unwrap();
    let out = b.binary(BinOp::Add, s1, s2).unwrap();
    let g = b.finish(out).unwrap();

    let mut wb = WeightStore::builder();
    for (name, n) in [("a", K), ("b", K), ("c", 1)] {
        let bytes: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        wb.insert(
            name,
            WeightEntry::Dense(
                DenseWeight::try_new(DType::F32, vec![n], StoredBytes::from(bytes)).unwrap(),
            ),
        )
        .unwrap();
    }
    let store = Arc::new(wb.build());
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());
    (g, store, map)
}

fn staged(g: &Graph, target: Target) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &REPLAY,
    )
    .expect("the fixture compiles")
}

fn fake_target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// One `Device::dispatch` call's recorded argument lengths, inputs then output, in binding order.
#[derive(Debug, Clone)]
struct DispatchRecord {
    input_elems: Vec<usize>,
    output_elems: usize,
    /// The storage of every bound handle, inputs then the output.
    storages: Vec<poot_target::BufferStorage>,
}

#[derive(Clone, Default)]
struct Recorder {
    dispatches: Rc<RefCell<Vec<DispatchRecord>>>,
    memory: Rc<poot_runtime_common::MemoryCounters>,
    /// Review F2: each `BufferRole::Activation` `Device::allocate` call's own `elems`, in call
    /// order - one entry per arena slot (`load_entry`'s `buffer_plan.slots()` loop), so a test can
    /// assert the exact per-slot sizes an independently hand-traced plan predicts, never a number
    /// re-derived from `BufferPlan` itself.
    activation_allocs: Rc<RefCell<Vec<usize>>>,
    /// How many bytes short of the asked count each untyped arena buffer (POOT-1450) is really
    /// allocated: a device whose allocation disagrees with the plan's lengths.
    activation_shortfall: usize,
}

/// What the device really allocated, or a typed view of it: what [`Device::buffer_elems`] reports, in
/// units of `storage`.
#[derive(Clone, Copy, Debug)]
struct Buf {
    elems: usize,
    storage: poot_target::BufferStorage,
}

struct RecordingDevice {
    recorder: Recorder,
}

impl Device for RecordingDevice {
    type Buffer = Buf;
    type Kernel = ();
    type Recording = ();
    type Error = std::convert::Infallible;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        poot_executor::CopyMechanism::Native
    }
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn target(&self) -> Target {
        fake_target()
    }

    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.recorder.memory.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Buf, std::convert::Infallible> {
        let mut capacity = elems;
        if role == BufferRole::Activation {
            self.recorder.activation_allocs.borrow_mut().push(elems);
        }
        if storage == poot_target::BufferStorage::untyped() {
            capacity -= self.recorder.activation_shortfall;
        }
        let bytes = (elems * storage.element().byte_width()) as u64;
        // `Buffer = ()` has no `Drop` to tie the guard's live-bytes decrement to (mirrors
        // `cr30_lifecycle.rs`'s `FaultyDevice`): leaked on purpose, read back via `snapshot_all`.
        std::mem::forget(self.recorder.memory.record_alloc(role, bytes));
        Ok(Buf {
            elems: capacity,
            storage,
        })
    }

    fn view(
        &self,
        buffer: &Buf,
        storage: poot_target::BufferStorage,
    ) -> Result<Buf, std::convert::Infallible> {
        let elems = buffer
            .storage
            .view_elems(buffer.elems, storage)
            .expect("the engine views an arena buffer as a storage it admits");
        Ok(Buf { elems, storage })
    }

    fn buffer_elems(&self, buffer: &Buf) -> usize {
        buffer.elems
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_codegen::CompiledKernel,
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn begin(&mut self) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
        self.recorder.dispatches.borrow_mut().push(DispatchRecord {
            input_elems: d.inputs.iter().map(|a: &Arg<'_, Self>| a.elems).collect(),
            output_elems: d.output.elems,
            storages: d
                .inputs
                .iter()
                .chain(std::iter::once(&d.output))
                .map(|a: &Arg<'_, Self>| a.buffer.storage)
                .collect(),
        });
        Ok(())
    }

    fn copy(&mut self, _src: &Buf, _dst: &Buf) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn finish(&mut self) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn write_with(
        &mut self,
        _dst: &Buf,
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn read(&mut self, _src: &Buf, out: &mut [u8]) -> Result<(), std::convert::Infallible> {
        out.fill(0);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), std::convert::Infallible> {
        Ok(())
    }
}

/// SC-002 (the engine's own half of the decoupled-length contract): every dispatch's `Arg::elems`
/// is the operand's own logical element count - `v1` [64], `s1`/`c`/`s2`/`out` [1] - never a slot's
/// shared capacity, in particular at eqn 2 (`s2 = reduce_sum(c)`), whose output shares `v1`'s freed
/// [64] arena slot but must still publish length 1.
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): in
/// `crates/poot-executor/src/engine.rs`'s `walk`, build every `Arg` with `elems: 0` again (the
/// pre-547b placeholder). Every assertion below failed (`0` instead of the value's real length,
/// starting with eqn 0's inputs: expected `[64, 64]`, got `[0, 0]`); restoring `elems` from the
/// stored `(Loc, u32)` pairs made every one green again.
#[test]
fn dispatch_args_carry_the_values_own_length_never_the_slots_capacity() {
    let (g, store, map) = fixture();
    let target = fake_target();
    let recorder = Recorder::default();
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    let entry = engine.add_entry(exe, &staged(&g, target)).unwrap();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();

    let dispatches = recorder.dispatches.borrow();
    assert_eq!(dispatches.len(), 4, "four equations, four dispatches");
    let expected: [(&[usize], usize); 4] = [
        (&[64, 64], 64), // eqn 0: v1 = a + b
        (&[64], 1),      // eqn 1: s1 = reduce_sum(v1)
        (&[1], 1),       // eqn 2: s2 = reduce_sum(c) (output shares v1's freed [64] slot)
        (&[1, 1], 1),    // eqn 3: out = s1 + s2
    ];
    for (i, (record, (want_inputs, want_output))) in dispatches.iter().zip(expected).enumerate() {
        assert_eq!(
            record.input_elems, want_inputs,
            "eqn {i}: input lengths must be each operand's own, never a reused slot's capacity"
        );
        assert_eq!(
            record.output_elems, want_output,
            "eqn {i}: output length must be the value's own, never a reused slot's capacity"
        );
    }
}

/// SC-003: `Program::buffer_plan().arena_bytes()` equals the bytes the engine actually asks
/// `Device::allocate` for under `BufferRole::Activation` - the one role the arena uses (never the
/// weight/state/slot-input buffers, which are not part of the buffer plan at all).
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): in
/// `crates/poot-graph-plan/src/buffer_plan.rs`'s `BufferPlan::arena_bytes`, sum `self.slots.iter()`
/// twice (`.chain(self.slots.iter())`) instead of once. Observed: `arena_bytes` reported double the
/// engine's real `Activation` allocation (the assertion's two sides disagreed by exactly 2x);
/// reverting to the single sum restored equality.
#[test]
fn program_arena_bytes_equals_what_the_engine_allocates() {
    let (g, store, map) = fixture();
    let target = fake_target();
    let program = poot_graph_plan::compile(&g, &target, &REPLAY).unwrap();

    let recorder = Recorder::default();
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    engine.add_entry(exe, &staged(&g, target)).unwrap();

    let activation_bytes = engine
        .stats()
        .memory
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        activation_bytes,
        program.buffer_plan().arena_bytes() as u64,
        "Program::buffer_plan().arena_bytes() must equal the engine's own Activation allocation"
    );
    // The reused layout from `dispatch_args_carry_...` above: one [64] slot (v1/s2), two [1]
    // slots (s1, out) - 66 f32 elements, never the naive 67 (v1, s1, s2, out unshared).
    assert_eq!(activation_bytes, 66 * 4);
}

/// POOT-1433: the stages of an entry share the buffers of values that are dead between them. Layer 0 sums a
/// `[64]` sum into the `[1]` boundary `s`; layer 1 recomputes two `[64]` temporaries from `s`. Each stage
/// alone needs a `[64]` slot; the entry allocates one, plus the two exact-size values that outlive their
/// stage (`s`, and the result), where the stages' own arenas would add up to two.
///
/// Mutation: allocate every stage's slots separately again (`EntryArena::plan` called once per stage in
/// `Engine::load_entry`); the engine's Activation bytes exceed the shared figure by one `[64]` slot (256 B).
#[test]
fn staged_entries_allocate_one_arena_for_the_values_their_stages_share() {
    let (_, store, _map) = fixture();
    let b = Builder::for_tests();
    let ta = b.constant("a", TensorType::f32(vec![K])).unwrap();
    let tb = b.constant("b", TensorType::f32(vec![K])).unwrap();
    let s;
    {
        let _layer = b.layer_scope(0);
        let v1 = b.binary(BinOp::Add, ta, tb).unwrap();
        s = b.reduce(RedOp::Sum, v1, 0, false).unwrap();
    }
    let out;
    {
        let _layer = b.layer_scope(1);
        let wide = b.broadcast(s, vec![K]).unwrap();
        let m1 = b.binary(BinOp::Add, wide, ta).unwrap();
        let m2 = b.binary(BinOp::Mul, m1, tb).unwrap();
        out = b.reduce(RedOp::Sum, m2, 0, false).unwrap();
    }
    let g = b.finish(out).unwrap();
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());

    let target = fake_target();
    let program = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Pipeline {
                after_layers: vec![0],
                devices: vec![DeviceId(0); 2],
            },
        },
        &REPLAY,
    )
    .expect("the two-layer fixture compiles");
    assert_eq!(program.stages().count(), 2, "one cut, two stages");
    let separate: usize = program
        .stages()
        .map(|(_, _, p)| p.buffer_plan().arena_bytes())
        .sum();

    let mut engine = Engine::new(RecordingDevice {
        recorder: Recorder::default(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    engine.add_entry(exe, &program).unwrap();
    let activation_bytes = engine
        .stats()
        .memory
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    // Hand-traced: the [64] temporaries (one shared slot), `s` and the result, both [1].
    assert_eq!(activation_bytes, (64 + 1 + 1) * 4);
    assert!(
        activation_bytes < separate as u64,
        "the stages' own arenas ({separate} B) would each hold a [64] slot"
    );
}

/// SC-004 (accounting half): two simultaneously retained entries with distinct scratch allocations
/// (the fixture compiled twice, once per `add_entry`, into the same executable) have a planned
/// aggregate equal to the engine's actual retained `Activation` bytes. The expected aggregate is
/// `2 * 66 * 4` - the same independently hand-traced per-entry byte count
/// [`program_arena_bytes_equals_what_the_engine_allocates`] (SC-003) already names (one `[64]` slot,
/// two `[1]` slots), never `program.buffer_plan().arena_bytes()` itself (review F2: `Engine::
/// load_entry` allocates from that exact same `BufferPlan` object, so comparing the engine's bytes
/// against the plan's own accessor is true by construction for any plan, correct or not - a
/// `BufferPlan::arena_bytes` double-count bug, or any other wrong plan, would pass this assertion as
/// long as the engine keeps reading the same (wrong) plan it is being checked against). Entries never
/// share arena buffers with each other (each `Engine::load_entry` call allocates its own
/// `arena: Vec<D::Buffer>`), so the real aggregate is `2x` one entry's bytes, not a coincidence of
/// the fixture repeating.
#[test]
fn two_retained_entries_have_a_planned_aggregate_equal_to_actual_retained_bytes() {
    let (g, store, map) = fixture();
    let target = fake_target();
    let recorder = Recorder::default();
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    engine.add_entry(exe, &staged(&g, target)).unwrap();
    engine.add_entry(exe, &staged(&g, target)).unwrap();

    let activation_bytes = engine
        .stats()
        .memory
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    const EXPECTED_PER_ENTRY_BYTES: u64 = 66 * 4;
    assert_eq!(
        activation_bytes,
        2 * EXPECTED_PER_ENTRY_BYTES,
        "two retained entries' independently hand-traced arena aggregate must equal the engine's \
         actual retained bytes"
    );
}

/// SC-004 (export half): a state-commit source that nothing else reads must keep its own arena slot
/// through the epilogue, even though the ordinary birth/death rule alone would free it mid-program.
/// Fixture (all `f32 [K=2]`, one storage class, so every value below shares the same coloring group):
///
/// ```text
/// a, s: state inputs                  // not locally computed, no slot
/// bias: const
/// v1    = a + bias      // traced twice (`v1` and `new_s`); `cse` merges them into one value
/// new_a = s + bias      // state-commit source for "a", read by nothing else
/// junk  = v1 + v1
/// result = junk + new_s // graph output; `fuse` folds junk and result into one region over v1
/// ```
///
/// After `cse` and `fuse` the program is three equations: `e0: v1`, `e1: new_a`, `e2: result` (the fused
/// region, reading `v1`). `v1` is also the state-commit source for "s", and it cannot write into "s"'s own
/// buffer in place because `e1` still reads `s`, so it takes an arena slot like the others. Hand-traced
/// coloring with `v1`, `new_a` and `result` correctly forced live through the epilogue (`death = n = 3`,
/// never handed back to the free list): three distinct one-occupant slots, each 2 elements:
/// `3 * 2 * 4 = 24` bytes.
///
/// Without the export guard (mutation below), `new_a` (birth 1, nothing else reads it) dies at its own
/// birth 1, freeing before `result`'s birth 2 - `result` reuses `new_a`'s slot, exactly the corruption the
/// guard exists to prevent: `result`'s dispatch would overwrite the buffer the state-commit epilogue later
/// copies out as the new value of state "a". The group collapses to two slots - `2 * 2 * 4 = 16` bytes - a
/// different, independently wrong number, never a value this test derives from `BufferPlan` itself.
///
/// Mutation (applied by hand, reverted, never left in the tree): in
/// `crates/poot-graph-plan/src/buffer_plan.rs`'s `plan_buffers`, make the `export` closure a no-op.
/// Observed: `activation_allocs` becomes `[2, 2]` (16 bytes) instead of `[2, 2, 2]` (24 bytes);
/// restoring the closure's body made both green again.
#[test]
fn state_commit_export_keeps_its_own_slot_through_the_epilogue() {
    const K: usize = 2;
    let b = poot_graph_ir::Builder::for_tests();
    let a = b
        .state_input(
            "a",
            TensorType::f32(vec![K]),
            poot_graph_ir::StateRole::Recurrent,
        )
        .unwrap();
    let s = b
        .state_input(
            "s",
            TensorType::f32(vec![K]),
            poot_graph_ir::StateRole::Recurrent,
        )
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![K])).unwrap();
    let v1 = b.binary(BinOp::Add, a, bias).unwrap();
    let new_a = b.binary(BinOp::Add, s, bias).unwrap();
    let junk = b.binary(BinOp::Add, v1, v1).unwrap();
    let new_s = b.binary(BinOp::Add, a, bias).unwrap();
    let result = b.binary(BinOp::Add, junk, new_s).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(result).with_state(&[(a, new_a), (s, new_s)]))
        .unwrap();

    let mut wb = WeightStore::builder();
    let bias_bytes: Vec<u8> = (0..K).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    wb.insert(
        "bias",
        WeightEntry::Dense(
            DenseWeight::try_new(DType::F32, vec![K], StoredBytes::from(bias_bytes)).unwrap(),
        ),
    )
    .unwrap();
    let store = Arc::new(wb.build());
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());

    let target = fake_target();
    let recorder = Recorder::default();
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    engine.add_entry(exe, &staged(&g, target)).unwrap();

    let activation_allocs = recorder.activation_allocs.borrow().clone();
    assert_eq!(
        activation_allocs,
        vec![K, K, K],
        "v1/new_a/result must each get their own arena slot (hand-traced above); a shorter list \
         means an export's slot was handed to a later value"
    );
    let activation_bytes: u64 = engine
        .stats()
        .memory
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        activation_bytes,
        (3 * K * 4) as u64,
        "independently hand-traced arena bytes for this fixture (three never-reused [K=2] f32 slots)"
    );
}

/// POOT-1205 SC-001: a dispatch whose asserted length exceeds its bound buffer by one element is refused
/// with the shared typed error by the engine, before `Device::dispatch` is called. The device allocates
/// every untyped arena buffer one f32 (four bytes) short of what the plan asked, so eqn 0's output (`v1`,
/// 64 elements) asserts 64 against its view of 63; the same fixture and device allocate exactly and step
/// cleanly when no buffer is short.
///
/// Mutation: remove the `check_arg_lengths` call in `Engine::walk` (`crates/poot-executor/src/engine.rs`);
/// the device then records the dispatches and the step succeeds, so the `Err` assertion goes red.
#[test]
fn a_dispatch_longer_than_its_buffer_is_refused_before_the_device_sees_it() {
    let (g, store, map) = fixture();
    let recorder = Recorder {
        activation_shortfall: 4,
        ..Recorder::default()
    };
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    let entry = engine.add_entry(exe, &staged(&g, fake_target())).unwrap();

    let error = engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .err()
        .expect("a length beyond its buffer must be refused");

    let ExecError::Dispatch(refusal) = error else {
        panic!("expected the shared dispatch error, got {error:?}");
    };
    assert_eq!(
        *refusal,
        DispatchError::ArgExceedsBuffer {
            arg: 2,
            elems: 64,
            capacity: 63
        },
        "eqn 0 asserts its 64-element output (argument 2) against a 63-element buffer"
    );
    assert!(
        recorder.dispatches.borrow().is_empty(),
        "the device must never see the refused dispatch: {:?}",
        recorder.dispatches.borrow()
    );
}

/// POOT-1450: the engine binds every arena value through a view of its buffer as the value's own storage,
/// never the untyped slot itself: the device sees an f32 handle for each f32 operand, including `s2`, whose
/// output shares `v1`'s untyped slot. Mutation: make `StageArena::bind` view each buffer as the raw untyped
/// slot (`BufferStorage::untyped()` for the value's storage); the device refuses that view
/// (`Untypable { view: .. Untyped }`) and the row goes red.
#[test]
fn every_arena_binding_is_a_view_of_the_values_own_storage() {
    let (g, store, map) = fixture();
    let recorder = Recorder::default();
    let mut engine = Engine::new(RecordingDevice {
        recorder: recorder.clone(),
    });
    let exe = engine.load_weights(store, map).unwrap();
    let entry = engine.add_entry(exe, &staged(&g, fake_target())).unwrap();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let dispatches = recorder.dispatches.borrow();
    assert_eq!(dispatches.len(), 4);
    for (i, record) in dispatches.iter().enumerate() {
        assert!(
            record
                .storages
                .iter()
                .all(|&storage| storage == poot_target::BufferStorage::f32()),
            "eqn {i}: every operand of the f32 fixture is bound as f32: {:?}",
            record.storages
        );
    }
}
