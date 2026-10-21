//! The staged-pipeline device rows, shared by the wgpu, ROCm and PTX twins (POOT-745).
//!
//! A program cut into stages on one device must be the one-stage program to everything outside the
//! executor: the same outputs and carried state after every step, the same host-write bytes per step
//! (a link is a shared device buffer), and one recording per stage however many steps run. Each row
//! loads the uncut program and the pipeline into two executables of one executor (so each owns its
//! own carried state), steps both in lockstep and compares them, reading carried state back through
//! production state-probe entries rather than an accessor.
//!
//! The oracle half of each row (the compiled stages chained on the CPU evaluator equal the uncut
//! traced graph bit for bit) lives with `compile_staged` in `poot-graph-plan`; this crate never
//! depends on `poot-eval`.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::model::{LogitRows, StepSpec};
use poot_executor::{
    Arg, Device, DeviceTime, Dispatch, EntryId, ExecutableId, Executor, NoSync, StepInputs,
    WeightUpload,
};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{
    BinOp, Builder, Graph, RedOp, Scalar, Slot, SlotKey, StateRole, Storage, TensorType, UnOp,
    ValidationId,
};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::StagedProgram,
    staged::TargetSet, staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::{BufferRole, CompiledKernel, MemoryCounterSnapshot};
use poot_target::BufferStorage;
use poot_tensor::{DType, HostView};
use poot_weights::WeightMap;

use super::dense::{Dense, Family, step};

/// Decode positions of the qwen2 row: the cache capacity, and the number of steps it runs.
const CAP: usize = 16;

/// How two executables' float results are compared.
#[derive(Clone, Copy, Debug)]
pub enum Agreement {
    /// Every byte equal: a cut must not change what a device computes.
    BitIdentical,
    /// Within `rel_tol` of the single-stage result per element (relative to `1 + |expected|`), for a
    /// backend whose fusion differs across the cut. The difference is recorded by the caller.
    Tolerance(f32),
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

fn const_data(name: &str, shape: &[usize]) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(n, seed)
}

/// A store holding every const `g` names, as F32 stored tensors.
fn store_for(g: &Graph) -> WeightStore {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for &id in g.inputs() {
        let meta = g.meta(id);
        if meta.storage == Storage::Const {
            entries.insert(meta.name().unwrap().into_owned(), meta.aval.shape.clone());
        }
    }
    let mut builder = WeightStore::builder();
    for (name, shape) in entries {
        let bytes: Vec<u8> = const_data(&name, &shape)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let dense = DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn compile(g: &Graph, target: Target, devices: DevicePlacement) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices,
        },
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .unwrap_or_else(|error| panic!("compile_staged: {error}"))
}

fn pipeline(after_layers: &[usize]) -> DevicePlacement {
    DevicePlacement::Pipeline {
        after_layers: after_layers.to_vec(),
        devices: vec![DeviceId(0); after_layers.len() + 1],
    }
}

/// A graph whose sole output is one named state value: a production entry that reads a carried
/// buffer through the same bind, commit and readback path as any other.
fn state_probe(name: &str, aval: TensorType) -> Graph {
    let b = Builder::for_tests();
    let state = b.state_input(name, aval, StateRole::Recurrent).unwrap();
    b.finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
        .unwrap()
}

/// One slot's host data for a step.
struct SlotValue {
    key: SlotKey,
    shape: Vec<usize>,
    f32s: Vec<f32>,
    i32s: Vec<i32>,
    dtype: DType,
}

impl SlotValue {
    fn view(&self) -> HostView<'_> {
        match self.dtype {
            DType::F32 => HostView::new(
                DType::F32,
                self.f32s.len(),
                bytemuck::cast_slice(&self.f32s),
            )
            .unwrap(),
            DType::I32 => HostView::new(
                DType::I32,
                self.i32s.len(),
                bytemuck::cast_slice(&self.i32s),
            )
            .unwrap(),
            other => panic!("unexpected slot dtype {other:?}"),
        }
    }
}

fn step_inputs(values: &[SlotValue]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for v in values {
        inputs.push(v.key.clone(), &v.shape, v.view());
    }
    inputs
}

/// One executable holding the same graph twice over: the uncut program and the pipeline.
struct Pair {
    single: (ExecutableId, EntryId),
    pipeline: (ExecutableId, EntryId),
    /// A state probe per carried buffer, on each executable.
    probes: Vec<((ExecutableId, EntryId), (ExecutableId, EntryId))>,
    stages: usize,
}

fn load_pair(
    exec: &mut dyn Executor,
    target: Target,
    g: &Graph,
    plain_graph: &Graph,
    after_layers: &[usize],
) -> Pair {
    let store = Arc::new(store_for(plain_graph));
    let mut load = |placement: DevicePlacement| -> (ExecutableId, EntryId, Arc<WeightMap>) {
        let mut weights = ConstWeights::new(&store);
        let named = weights.graph(g);
        let map = Arc::new(weights.map());
        let exe = exec.load_weights(store.clone(), map.clone()).unwrap();
        let program = compile(&named, target, placement);
        let entry = exec.add_entry(exe, &program).unwrap();
        (exe, entry, map)
    };
    let (single_exe, single_entry, _) = load(DevicePlacement::Single(DeviceId(0)));
    let (pipe_exe, pipe_entry, _) = load(pipeline(after_layers));
    let probes = g
        .state_pairs()
        .map(|pair| {
            let meta = g.meta(pair.input);
            let probe = state_probe(&meta.name().unwrap(), meta.aval.clone());
            let mut on = |exe| {
                let program = compile(&probe, target, DevicePlacement::Single(DeviceId(0)));
                (exe, exec.add_entry(exe, &program).unwrap())
            };
            (on(single_exe), on(pipe_exe))
        })
        .collect();
    Pair {
        single: (single_exe, single_entry),
        pipeline: (pipe_exe, pipe_entry),
        probes,
        stages: after_layers.len() + 1,
    }
}

fn assert_agree(label: &str, agreement: Agreement, single: &[u8], piped: &[u8]) {
    assert_eq!(single.len(), piped.len(), "{label}: output length");
    let floats = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let (want, got) = (floats(single), floats(piped));
    assert!(
        want.iter().all(|v| v.is_finite()),
        "{label}: the single-stage reference holds a non-finite value, so agreement proves nothing"
    );
    match agreement {
        Agreement::BitIdentical => {
            if single != piped {
                let at = single
                    .iter()
                    .zip(piped)
                    .position(|(a, b)| a != b)
                    .expect("unequal slices differ somewhere")
                    / 4;
                panic!(
                    "{label}: the pipeline differs from the single stage at element {at}: \
                     pipeline {} against single stage {}",
                    got[at], want[at]
                );
            }
        }
        Agreement::Tolerance(rel_tol) => {
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.is_finite() && (g - w).abs() <= rel_tol * (1.0 + w.abs()),
                    "{label}: element {i}: pipeline {g} against single stage {w} (rel_tol {rel_tol})"
                );
            }
        }
    }
}

fn stage_recordings(exec: &dyn Executor, (exe, entry): (ExecutableId, EntryId)) -> Vec<(u64, u64)> {
    exec.stats()
        .stages
        .iter()
        .filter(|stats| stats.exe == exe && stats.entry == entry)
        .map(|stats| (stats.recordings, stats.replays))
        .collect()
}

/// Steps `pair` in lockstep, `steps` times: after every step the two outputs agree and every carried
/// buffer (read back through its probe) agrees. The first step's host-write bytes are equal for the
/// two programs.
fn run_lockstep(
    exec: &mut dyn Executor,
    pair: &Pair,
    agreement: Agreement,
    steps: usize,
    inputs_for: &dyn Fn(usize) -> Vec<SlotValue>,
) {
    for index in 0..steps {
        let values = inputs_for(index);
        let before = exec.stats().host_write_bytes;
        let single = exec
            .step(
                pair.single.0,
                pair.single.1,
                &step_inputs(&values),
                &mut NoSync,
            )
            .unwrap_or_else(|e| panic!("step {index}: single stage: {e}"))
            .read()
            .unwrap();
        let middle = exec.stats().host_write_bytes;
        let piped = exec
            .step(
                pair.pipeline.0,
                pair.pipeline.1,
                &step_inputs(&values),
                &mut NoSync,
            )
            .unwrap_or_else(|e| panic!("step {index}: pipeline: {e}"))
            .read()
            .unwrap();
        let after = exec.stats().host_write_bytes;
        assert_eq!(
            after - middle,
            middle - before,
            "step {index}: the {}-stage step wrote {} host bytes, the single stage {}: a link must \
             not touch the host",
            pair.stages,
            after - middle,
            middle - before
        );
        assert_agree(&format!("step {index} output"), agreement, &single, &piped);
        for (i, (single_probe, pipe_probe)) in pair.probes.iter().enumerate() {
            let read = |exec: &mut dyn Executor, (exe, entry): (ExecutableId, EntryId)| {
                exec.step(exe, entry, &StepInputs::new(), &mut NoSync)
                    .unwrap()
                    .read()
                    .unwrap()
            };
            let single_state = read(exec, *single_probe);
            let piped_state = read(exec, *pipe_probe);
            assert_agree(
                &format!("step {index} carried state {i}"),
                agreement,
                &single_state,
                &piped_state,
            );
        }
    }
}

/// SC-001/SC-002/SC-003: qwen2's decode step cut after layers 0 and 1 (three stages, one device)
/// agrees with the one-stage program on every output and every carried KV buffer for a full cache of
/// steps, writes the same host bytes per step, and records each stage once.
pub fn assert_qwen2_pipeline_matches_single_stage(
    exec: &mut dyn Executor,
    target: Target,
    agreement: Agreement,
) {
    let model = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 3)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model();
    let graph = model
        .model
        .trace(
            &StepSpec::step(step(1, 1, CAP, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let validated = graph.clone();
    let pair = load_pair(exec, target, &validated, &graph, &[0, 1]);
    assert_eq!(pair.stages, 3);

    let slots: Vec<(SlotKey, Slot, TensorType)> = graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let meta = graph.meta(id);
            let Storage::Slot(slot) = meta.storage else {
                return None;
            };
            Some((meta.slot_key().unwrap().clone(), slot, meta.aval.clone()))
        })
        .collect();
    let inputs_for = |index: usize| -> Vec<SlotValue> {
        slots
            .iter()
            .map(|(key, slot, aval)| {
                let value = match slot {
                    Slot::Token => ((index * 7 + 3) % 32) as i32,
                    Slot::Pos => index as i32,
                    other => panic!("the qwen2 decode graph has no {other:?} slot"),
                };
                SlotValue {
                    key: key.clone(),
                    shape: aval.shape.clone(),
                    f32s: vec![value as f32; aval.numel()],
                    i32s: vec![value; aval.numel()],
                    dtype: aval.dtype,
                }
            })
            .collect()
    };
    run_lockstep(exec, &pair, agreement, CAP, &inputs_for);

    assert_eq!(
        stage_recordings(exec, pair.pipeline),
        vec![(1, CAP as u64); 3],
        "after {CAP} steps every stage has recorded once and replayed every step"
    );
    assert_eq!(
        stage_recordings(exec, pair.single),
        vec![(1, CAP as u64)],
        "the single stage records once too"
    );
}

/// The export fixture of `poot-graph-plan`'s `staged::tests`, on a device: four independent values
/// leave layer 0 (one a structural duplicate CSE merges), `c` fans out to both later stages, layer
/// 0's carried state is fed by a boundary value and a witness hangs off another boundary. The
/// pipeline must agree with the single stage on the output and both carried buffers, and the
/// witness (checked per stage after the single wait) must hold.
pub fn assert_export_fixture_matches_single_stage(
    exec: &mut dyn Executor,
    target: Target,
    agreement: Agreement,
) {
    let (validated, plain_graph) = export_fixture();
    let pair = load_pair(exec, target, &validated, &plain_graph, &[0, 1]);
    let x_key = SlotKey::new(Slot::Activation, None);
    let inputs_for = |index: usize| -> Vec<SlotValue> {
        let data: Vec<f32> = fill(4, 900 + index as u64);
        vec![SlotValue {
            key: x_key.clone(),
            shape: vec![4],
            i32s: Vec::new(),
            f32s: data,
            dtype: DType::F32,
        }]
    };
    run_lockstep(exec, &pair, agreement, 4, &inputs_for);
    assert_eq!(
        stage_recordings(exec, pair.pipeline),
        vec![(1, 4); 3],
        "three stages, one recording each"
    );
}

fn export_fixture() -> (Graph, Graph) {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let v = b.constant("v", TensorType::f32(vec![4])).unwrap();
    let (a, a2, c, d, witness, k0_in, k0_out);
    {
        let _layer = b.layer_scope(0);
        a = b.binary(BinOp::Mul, x, w).unwrap();
        a2 = b.binary(BinOp::Mul, x, w).unwrap();
        c = b.binary(BinOp::Add, a, v).unwrap();
        d = b.binary(BinOp::Sub, a, v).unwrap();
        k0_in = b
            .state_input("k0", TensorType::f32(vec![4]), StateRole::Recurrent)
            .unwrap();
        k0_out = b.binary(BinOp::Add, k0_in, c).unwrap();
        let total = b.reduce(RedOp::Sum, d, 0, true).unwrap();
        let negated = b.unary(UnOp::Neg, total).unwrap();
        witness = b
            .binary_scalar(BinOp::Ge, negated, Scalar::F32(1.0e9))
            .unwrap();
    }
    let (g, k1_in, k1_out);
    {
        let _layer = b.layer_scope(1);
        let e = b.binary(BinOp::Mul, a, a2).unwrap();
        let f = b.binary(BinOp::Add, e, c).unwrap();
        g = b.binary(BinOp::Mul, f, d).unwrap();
        k1_in = b
            .state_input("k1", TensorType::f32(vec![4]), StateRole::Recurrent)
            .unwrap();
        k1_out = b.binary(BinOp::Add, k1_in, g).unwrap();
    }
    let out;
    {
        let _layer = b.layer_scope(2);
        out = b.binary(BinOp::Add, g, c).unwrap();
    }
    let state = [(k0_in, k0_out), (k1_in, k1_out)];
    let validated = b
        .finish(
            poot_graph_ir::Outputs::new(out)
                .with_state(&state)
                .with_witness(ValidationId(1), "d_not_positive", witness),
        )
        .expect("the export fixture is well formed");
    // The same graph without its witness: a second finish over the same tables.
    let mut plain = poot_graph_ir::GraphEdit::new();
    *plain.values_mut() = validated.values().clone();
    *plain.inputs_mut() = validated.inputs().to_vec();
    *plain.consts_mut() = validated.consts().to_vec();
    *plain.slots_mut() = validated.slots().to_vec();
    *plain.eqns_mut() = validated.eqns().to_vec();
    plain.set_result(validated.result());
    for (input, next) in validated.state() {
        plain.push_state(input, next);
    }
    let plain = plain.commit().expect("the export fixture is well formed");
    (validated, plain)
}

/// What a [`Counting`] device was asked for.
#[derive(Debug, Default)]
pub struct Counts {
    /// Completion waits ([`Device::synchronize`] calls).
    pub waits: Cell<u64>,
    /// Dispatches recorded ([`Device::dispatch`] calls): a step records them once, then replays.
    pub dispatches: Cell<u64>,
}

/// A [`Device`] that counts the completion waits and recorded dispatches its inner device is asked
/// for, and otherwise forwards every call unchanged: the instrument behind
/// [`assert_pipeline_waits_once`] and [`assert_detailed_timing_reports_every_stage`].
pub struct Counting<D: Device> {
    inner: D,
    counts: Rc<Counts>,
}

impl<D: Device> Counting<D> {
    /// `inner`, and the counters the test keeps once the engine owns the device.
    pub fn new(inner: D) -> (Self, Rc<Counts>) {
        let counts = Rc::new(Counts::default());
        (
            Self {
                inner,
                counts: counts.clone(),
            },
            counts,
        )
    }
}

impl<D: Device> Device for Counting<D> {
    type Buffer = D::Buffer;
    type Kernel = D::Kernel;
    type Recording = D::Recording;
    type Error = D::Error;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        self.inner.copy_mechanism()
    }
    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    fn target(&self) -> Target {
        self.inner.target()
    }
    fn weight_upload(&self) -> WeightUpload {
        self.inner.weight_upload()
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.inner.memory()
    }
    fn free_memory_bytes(&self) -> Option<u64> {
        self.inner.free_memory_bytes()
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.inner.allocate(role, storage, elems)
    }
    fn buffer_elems(&self, buffer: &Self::Buffer) -> usize {
        self.inner.buffer_elems(buffer)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.inner.load_kernel(key, kernel)
    }
    fn begin(&mut self) -> Result<(), Self::Error> {
        self.inner.begin()
    }
    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        self.counts.dispatches.set(self.counts.dispatches.get() + 1);
        let inputs: Vec<Arg<'_, D>> = d
            .inputs
            .iter()
            .map(|arg| Arg {
                buffer: arg.buffer,
                elems: arg.elems,
            })
            .collect();
        self.inner.dispatch(Dispatch {
            kernel: d.kernel,
            inputs: &inputs,
            output: Arg {
                buffer: d.output.buffer,
                elems: d.output.elems,
            },
            launch: d.launch,
            work: d.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.inner.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Self::Recording, Self::Error> {
        self.inner.finish()
    }
    fn begin_step(&mut self) -> Result<(), Self::Error> {
        self.inner.begin_step()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.inner.replay(recording)
    }
    fn write_with(
        &mut self,
        dst: &Self::Buffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Self::Error> {
        self.inner.write_with(dst, offset, len, fill)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.counts.waits.set(self.counts.waits.get() + 1);
        self.inner.synchronize()
    }
    fn device_time(&self) -> DeviceTime {
        self.inner.device_time()
    }
    fn abort(&mut self) -> Result<(), Self::Error> {
        self.inner.abort()
    }
    fn classify_fault(&self, error: &Self::Error) -> Option<(String, u32)> {
        self.inner.classify_fault(error)
    }
}

/// The qwen2 decode graph loaded as the uncut program and as the three-stage pipeline of one
/// executor, with one decode step's inputs.
fn decode_pair(exec: &mut dyn Executor, target: Target) -> (Pair, Vec<SlotValue>) {
    let model = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 3)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model();
    let graph = model
        .model
        .trace(
            &StepSpec::step(step(1, 1, CAP, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let validated = graph.clone();
    let pair = load_pair(exec, target, &validated, &graph, &[0, 1]);
    let values: Vec<SlotValue> = graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let meta = graph.meta(id);
            let Storage::Slot(slot) = meta.storage else {
                return None;
            };
            let value = match slot {
                Slot::Token => 5,
                Slot::Pos => 0,
                other => panic!("the qwen2 decode graph has no {other:?} slot"),
            };
            Some(SlotValue {
                key: meta.slot_key().unwrap().clone(),
                shape: meta.aval.shape.clone(),
                f32s: vec![value as f32; meta.aval.numel()],
                i32s: vec![value; meta.aval.numel()],
                dtype: meta.aval.dtype,
            })
        })
        .collect();
    (pair, values)
}

/// SC-006 on a device: a three-stage step of the qwen2 pipeline waits for the device once, where the
/// single stage also waits once, and the outputs agree. `counts.waits` counts the wrapped device's
/// [`Device::synchronize`] calls; `device_waits` reads the backend's own count of the times it stood
/// waiting on the device (a backend that blocked inside `replay` would pass the first and fail this
/// one; a backend with no finer count passes `&|| counts.waits.get()`).
pub fn assert_pipeline_waits_once(
    exec: &mut dyn Executor,
    counts: &Counts,
    device_waits: &dyn Fn() -> u64,
    target: Target,
    agreement: Agreement,
) {
    let (pair, values) = decode_pair(exec, target);
    let stepped = |exec: &mut dyn Executor, (exe, entry): (ExecutableId, EntryId)| {
        let before = (counts.waits.get(), device_waits());
        let out = exec
            .step(exe, entry, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        (
            out,
            counts.waits.get() - before.0,
            device_waits() - before.1,
        )
    };
    let (single, single_waits, single_device) = stepped(exec, pair.single);
    let (piped, pipeline_waits, pipeline_device) = stepped(exec, pair.pipeline);
    assert_eq!(single_waits, 1, "the single stage waits once for its step");
    assert_eq!(
        single_device, 1,
        "the single stage waits once on the device for its step"
    );
    assert_eq!(
        pipeline_waits, 1,
        "the {}-stage step enqueues every stage and waits once, at the end",
        pair.stages
    );
    assert_eq!(
        pipeline_device, 1,
        "the {}-stage step waits on the device once, not once per stage",
        pair.stages
    );
    assert_agree("the waiting pipeline's output", agreement, &single, &piped);
}

/// SC-006 on a device, timing half: a three-stage step under detailed timing reports the dispatches of
/// every stage. The first step records the pipeline, so `counts.dispatches` over it is what the three
/// stages hold between them; the step's report holds exactly that many, none dropped with an earlier
/// stage's timing and none attributed by another stage's plan (the report's per-kind buckets are
/// built from the dispatch index). `exec` must have been built with detailed timing over a device
/// that measures per dispatch, wrapped in [`Counting`].
pub fn assert_detailed_timing_reports_every_stage(
    exec: &mut dyn Executor,
    counts: &Counts,
    target: Target,
) {
    let (pair, values) = decode_pair(exec, target);
    let (exe, entry) = pair.pipeline;
    let mark = exec.stats().timing.mark();
    let recorded_before = counts.dispatches.get();
    exec.step(exe, entry, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    let recorded = counts.dispatches.get() - recorded_before;
    let report = poot_profile::Report::window(&exec.stats().timing, &mark, &[entry.raw()]);
    assert!(
        matches!(report.detail, poot_runtime_common::Coverage::Complete),
        "the step's per-dispatch detail is retained: {:?}",
        report.detail
    );
    let reported: u64 = report
        .buckets
        .values()
        .map(|bucket| bucket.dispatch_count)
        .sum();
    assert!(
        recorded >= pair.stages as u64,
        "every stage holds a dispatch ({recorded} recorded)"
    );
    assert_eq!(
        reported, recorded,
        "the {}-stage step reports the dispatches of every stage: {:?}",
        pair.stages, report.buckets
    );
}

/// The timing half of SC-006 on a device that times replays as a whole (PTX measures one span around
/// the step's graph launches): the three-stage step and the single-stage step each report a
/// measured, nonzero device span.
pub fn assert_pipeline_step_reports_a_device_span(exec: &mut dyn Executor, target: Target) {
    let (pair, values) = decode_pair(exec, target);
    let span = |exec: &mut dyn Executor, (exe, entry): (ExecutableId, EntryId)| {
        let outputs = exec
            .step(exe, entry, &step_inputs(&values), &mut NoSync)
            .unwrap();
        match outputs.device_time() {
            DeviceTime::Measured(measured) => measured.device_span,
            DeviceTime::Unknown => None,
        }
    };
    let single = span(exec, pair.single).expect("the single stage reports a device span");
    let piped = span(exec, pair.pipeline).expect("the pipeline reports a device span");
    assert!(!single.is_zero(), "the single stage's span is nonzero");
    assert!(
        !piped.is_zero(),
        "the {}-stage step's span is nonzero",
        pair.stages
    );
}
