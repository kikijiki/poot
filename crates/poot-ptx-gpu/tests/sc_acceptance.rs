//! Card 549 acceptance rows SC-001, SC-002, SC-006, SC-007, SC-009, SC-010 and SC-011: real-hardware
//! verification of properties the shared `poot_executor::Engine<D: Device>` contract promises,
//! specifically against `PtxDevice`/`PtxContext` on real NVIDIA hardware (not a mock `Device`, which
//! `poot-executor/tests/cr30_lifecycle.rs` already covers generically). Every `MUTATION` doc comment
//! below names the exact external edit used to verify red->green; none is left in the tree.
//!
//! Skips cleanly with no NVIDIA GPU; `POOT_REQUIRE_PTX=1` turns that skip into a failure.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{
    BufferRole, Device, Dispatch, Engine, EntryId, ExecutableId, Executor, ExecutorStats,
    MemoryCounterSnapshot, NoSync, StepInputs,
};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp, SampleRule};
use poot_graph_ir::ops::sampling::sample_head;
use poot_graph_ir::{Graph, Slot, SlotKey, Storage, TensorType, ValueId};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    Submission, Target, TargetSet, compile_staged,
};
use poot_models::model::{LogitRows, Phase};
use poot_ptx_gpu::PtxDevice;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_tensor::{DType, HostView};
use poot_test_util::{assert_close_rel, device_skip};

const CAP: usize = 8;

const VOCAB: usize = 32;
const HIDDEN: usize = 16;

/// The fixture's qwen2 (16 hidden, 4 heads over 2 key-value heads, F32 consts) traced by the
/// registry's family for `phase` over `tokens` new tokens and `CAP` cached positions.
fn qwen2_graph(phase: Phase, tokens: usize) -> Graph {
    let m = Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(HIDDEN, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model();
    plain(
        m.model
            .trace(phase, step(1, tokens, CAP, LogitRows::Last))
            .unwrap(),
    )
}

fn prefill_graph(n: usize) -> Graph {
    qwen2_graph(Phase::Prefill, n)
}

fn decode_graph() -> Graph {
    qwen2_graph(Phase::Decode, 1)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
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
    if name == "causal.mask" {
        let l = shape[shape.len() - 1];
        return (0..n)
            .map(|i| if i % l <= i / l { 0.0 } else { -1.0e9 })
            .collect();
    }
    let seed = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(n, seed)
}

fn store_for(graphs: &[&Graph]) -> WeightStore {
    let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
    for g in graphs {
        for &id in &g.inputs {
            let meta = g.meta(id);
            if meta.storage == Storage::Const {
                entries.insert(meta.name.clone().unwrap(), meta.aval.shape.clone());
            }
        }
    }
    let mut builder = WeightStore::builder();
    for (name, shape) in entries {
        let bytes: Vec<u8> = const_data(&name, &shape)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn staged(
    g: &Graph,
    target: Target,
    submission: Submission,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: submission,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap()
}

struct SlotValue {
    key: SlotKey,
    shape: Vec<usize>,
    f32s: Vec<f32>,
    i32s: Vec<i32>,
    dtype: DType,
}

fn slot_values(g: &Graph, tokens: &[u32], start: usize) -> Vec<SlotValue> {
    let pos = start + tokens.len() - 1;
    g.inputs
        .iter()
        .filter_map(|&id| {
            let meta = g.meta(id);
            let Storage::Slot(slot) = meta.storage else {
                return None;
            };
            let TensorType { shape, dtype } = meta.aval.clone();
            let values: Vec<f64> = match slot {
                Slot::Token => tokens.iter().map(|&t| t as f64).collect(),
                // Card 550: `Slot::Pos` is `[rows, tokens]`: `[1,1] = [pos]` for a decode step, or
                // `[1,n] = [start, .., start+n-1]` for this one-shot prefill (always `start=0` at
                // every call site here, matching the mask's in-graph computation).
                Slot::Pos => {
                    let tokens_axis = *shape.last().unwrap_or(&1);
                    (0..tokens_axis).map(|i| (start + i) as f64).collect()
                }
                Slot::SeqLen => vec![(pos + 1) as f64],
                other => panic!("qwen2 test graph has no {other:?} slot"),
            };
            Some(SlotValue {
                key: meta.slot_key().unwrap().clone(),
                shape,
                f32s: values.iter().map(|&v| v as f32).collect(),
                i32s: values.iter().map(|&v| v as i32).collect(),
                dtype,
            })
        })
        .collect()
}

fn view(v: &SlotValue) -> HostView<'_> {
    match v.dtype {
        DType::F32 => {
            HostView::new(DType::F32, v.f32s.len(), bytemuck::cast_slice(&v.f32s)).unwrap()
        }
        DType::I32 => {
            HostView::new(DType::I32, v.i32s.len(), bytemuck::cast_slice(&v.i32s)).unwrap()
        }
        other => panic!("unexpected slot dtype {other:?}"),
    }
}

fn step_inputs(values: &[SlotValue]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for v in values {
        inputs.push(v.key.clone(), &v.shape, view(v));
    }
    inputs
}

fn oracle_step(
    g: &Graph,
    store: &WeightStore,
    values: &[SlotValue],
    state: &mut HashMap<String, HostTensor>,
) -> Vec<f32> {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let value = match meta.storage {
            Storage::Const => {
                let name = meta.name.as_deref().unwrap();
                let WeightEntry::Dense(dense) = store.get(name).unwrap() else {
                    unreachable!()
                };
                let data: Vec<f32> = bytemuck_f32(dense.bytes().as_slice());
                Value::from(HostTensor::f32(meta.aval.shape.clone(), data))
            }
            Storage::State => Value::from(
                state
                    .entry(meta.name.clone().unwrap())
                    .or_insert_with(|| HostTensor::zeros(meta.aval.shape.clone()))
                    .clone(),
            ),
            Storage::Computed(c) => Value::from(HostTensor::f32(c.shape(), c.values_f32())),
            Storage::Slot(_) => {
                let key = meta.slot_key().unwrap();
                let v = values.iter().find(|v| &v.key == key).unwrap();
                Value::from(match v.dtype {
                    DType::I32 => HostTensor::i32(v.shape.clone(), v.i32s.clone()),
                    _ => HostTensor::f32(v.shape.clone(), v.f32s.clone()),
                })
            }
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, value);
    }
    let evaluation = eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    for (&(si, _), t) in g.state.iter().zip(evaluation.state) {
        state.insert(
            g.meta(si).name.clone().unwrap(),
            t.into_host().expect("dense state tensor"),
        );
    }
    evaluation
        .output
        .into_host()
        .expect("dense output")
        .to_f32()
        .expect("an F32 or exact-I32 output widens")
        .into_owned()
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn try_ptx() -> Option<PtxDevice> {
    device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
}

fn assert_close(got: &[f32], want: &[f32], label: &str, tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let t = tol * b.abs().max(1e-3);
        assert!(
            (a - b).abs() <= t,
            "{label} elem {i}: ptx {a} vs cpu {b} (tol={t})"
        );
    }
}

fn make_inputs(ids: &[(ValueId, Vec<usize>, Vec<f32>)]) -> HashMap<ValueId, Value> {
    ids.iter()
        .map(|(id, shape, data)| (*id, HostTensor::f32(shape.clone(), data.clone()).into()))
        .collect()
}

mod common;

// ── SC-001 ───────────────────────────────────────────────────────────────

/// SC-001: a qwen2-shaped prefill entry and a decode entry sharing KV state by name (Z5) on one
/// executable match the CPU oracle over a full cache on real PTX hardware, and each entry records
/// exactly once (`ExecutorStats::recordings == 2`) - capture is the default, and an entry never
/// replays a recording it did not itself capture.
///
/// MUTATION (recorded here, never left in the tree): in `poot-executor/src/engine.rs`'s `step`, change
/// the replay lookup `loaded.entries.get(entry.index())` (around line 1222, the one feeding
/// `device.replay(recording)`) to `loaded.entries.get(0)` unconditionally. Decode (entry index 1) then
/// replays prefill's (index 0) recording, captured against prefill's own shapes/bindings. Result: RED -
/// decode's output diverges from the CPU oracle (or the dispatch ABI mismatches outright). Reverted:
/// GREEN.
#[test]
fn sc001_dense_decode_golden_records_each_entry_once_ptx() {
    let Some(ptx) = try_ptx() else { return };
    let target = Device::target(&ptx);
    let n = 3;
    let prefill_g = prefill_graph(n);
    let decode_g = decode_graph();
    let store = Arc::new(store_for(&[&prefill_g, &decode_g]));
    let mut engine = Engine::new(ptx);
    let exe = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let prefill = engine
        .add_entry(exe, &staged(&prefill_g, target, Submission::Replay))
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&decode_g, target, Submission::Replay))
        .unwrap();

    let mut oracle_state = HashMap::new();
    let prompt: Vec<u32> = (0..n as u32).map(|t| (t * 11 + 2) % 32).collect();
    let values = slot_values(&prefill_g, &prompt, 0);
    let expected = oracle_step(&prefill_g, &store, &values, &mut oracle_state);
    let got = bytemuck_f32(
        &engine
            .step(exe, prefill, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    assert_close_rel(&got, &expected, 5e-3);

    for pos in n..CAP {
        let values = slot_values(&decode_g, &[((pos * 3 + 1) % 32) as u32], pos);
        let expected = oracle_step(&decode_g, &store, &values, &mut oracle_state);
        let got = bytemuck_f32(
            &engine
                .step(exe, decode, &step_inputs(&values), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        assert_close_rel(&got, &expected, 5e-3);
    }

    let ExecutorStats { recordings, .. } = engine.stats();
    assert_eq!(
        recordings, 2,
        "prefill + decode, each recorded exactly once on real PTX"
    );
}

// ── SC-002 ───────────────────────────────────────────────────────────────

/// SC-002: a small parity suite (elementwise add, matmul, reduce-sum - three different `OpKind`/plan
/// kinds in one test) each matches the CPU oracle on real PTX hardware, proving the suite has real
/// per-op bite rather than passing by construction.
///
/// MUTATION (recorded here, never left in the tree): in `poot-graph-plan/src/planner/elementwise.rs`'s
/// `OpKind::Reduce { op, axis, .. }` lowering arm (around line 314), force `axis` to `0` regardless of
/// the equation's real axis. Result: RED - only `sc002_reduce` diverges from the CPU oracle (a
/// reduction over the wrong axis), while `sc002_add`/`sc002_matmul` stay GREEN, demonstrating the
/// mutation's effect is isolated to the one plan kind it targets. Reverted: GREEN.
#[test]
fn sc002_parity_suite_has_real_per_op_bite_ptx() {
    let Some(mut ptx) = try_ptx() else { return };

    {
        let n = 32usize;
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![n]));
        let c = b.constant("c", TensorType::f32(vec![n]));
        let out = b.binary(BinOp::Add, a, c);
        let (ai, ci) = (a.id, c.id);
        let g = b.finish(out);
        let inputs = make_inputs(&[(ai, vec![n], fill(n, 701)), (ci, vec![n], fill(n, 702))]);
        let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = common::run_resident(&mut ptx, &g, &inputs);
        assert_close(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            "sc002_add",
            1e-5,
        );
    }

    {
        let (m, k, n) = (4usize, 8usize, 4usize);
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let out = b.matmul(a, w);
        let (ai, wi) = (a.id, w.id);
        let g = b.finish(out);
        let inputs = make_inputs(&[
            (ai, vec![m, k], fill(m * k, 703)),
            (wi, vec![k, n], fill(k * n, 704)),
        ]);
        let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = common::run_resident(&mut ptx, &g, &inputs);
        assert_close(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            "sc002_matmul",
            1e-4,
        );
    }

    {
        let (rows, cols) = (2usize, 16usize);
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![rows, cols]));
        let out = b.reduce(RedOp::Sum, x, 1, false);
        let xi = x.id;
        let g = b.finish(out);
        let inputs = make_inputs(&[(xi, vec![rows, cols], fill(rows * cols, 705))]);
        let cpu = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let got = common::run_resident(&mut ptx, &g, &inputs);
        let tol = 1e-4 * (cols as f32).sqrt();
        assert_close(
            got.as_f32().unwrap(),
            cpu.as_f32().unwrap(),
            "sc002_reduce",
            tol,
        );
    }
}

// ── SC-006 ───────────────────────────────────────────────────────────────

/// Forwards every `Device` call to a real `PtxDevice`, except `replay`, which sleeps 200ms on the host
/// after the real replay returns. The real device-timed span is bracketed by CUDA events recorded
/// inside the real `replay` call itself (`PtxContext::launch_timed`), so this host-side pause after it
/// returns must NOT inflate the reported span - only a host-wall fabrication would be fooled by it.
struct SlowReplayDevice(PtxDevice);

impl Device for SlowReplayDevice {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = <PtxDevice as Device>::Error;

    fn target(&self) -> Target {
        self.0.target()
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.0.memory()
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.0.allocate(role, storage, elems)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.0.load_kernel(key, kernel)
    }
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error> {
        self.0.begin(submission)
    }
    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        self.0.dispatch(Dispatch {
            kernel: d.kernel,
            inputs: unsafe {
                std::mem::transmute::<
                    &[poot_executor::Arg<'_, Self>],
                    &[poot_executor::Arg<'_, PtxDevice>],
                >(d.inputs)
            },
            output: poot_executor::Arg {
                buffer: d.output.buffer,
                elems: d.output.elems,
            },
            threads: d.threads,
            workgroup: d.workgroup,
            work: d.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.0.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
        self.0.finish()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        let result = self.0.replay(recording);
        std::thread::sleep(Duration::from_millis(200));
        result
    }
    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error> {
        self.0.write(dst, bytes)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.0.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.0.synchronize()
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        self.0.device_time()
    }
    fn abort(&mut self) -> Result<(), Self::Error> {
        self.0.abort()
    }
}

/// SC-006: `Device::device_time` reports `Measured` with a real, honest device span when the device
/// was built via `PtxDevice::new_with_device_timing`, and `Unknown` for a plain `PtxDevice` - never a
/// fabricated host-wall duration. A 200ms host-side sleep injected right after the real `replay()`
/// call (via `SlowReplayDevice`, forwarding everything else to the real device) must not move the
/// reported span: it is bracketed by CUDA events recorded inside the real replay, not by anything the
/// caller does afterward.
///
/// MUTATION (recorded here, never left in the tree): in `poot-ptx-gpu/src/device.rs`'s
/// `PtxDevice::device_time`, replace the body with `DeviceTime::Measured(MeasuredDeviceTime {
/// sum_of_dispatch_durations: Some(self.last_step_wall), device_span: Some(self.last_step_wall),
/// dispatches: Vec::new() })` fed by an `Instant` captured at `replay()` entry/exit (a host-wall
/// duration, not a device one). Result: RED - the reported span is ~200ms (the injected sleep), not the
/// real sub-millisecond device time. Reverted: GREEN.
#[test]
fn sc006_device_time_is_measured_or_unknown_never_host_wall_ptx() {
    let Some(timed) =
        device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new_with_device_timing())
    else {
        return;
    };
    let Some(plain) = try_ptx() else { return };
    let target = Device::target(&timed);
    let g = decode_graph();
    let store = Arc::new(store_for(&[&g]));

    let mut plain_engine = Engine::new(plain);
    let exe_p = plain_engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_p = plain_engine
        .add_entry(exe_p, &staged(&g, target, Submission::Replay))
        .unwrap();
    let values = slot_values(&g, &[5], 0);
    let time_p = plain_engine
        .step(exe_p, decode_p, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .device_time();
    assert_eq!(
        time_p,
        poot_executor::DeviceTime::Unknown,
        "a counters-only PTX device must never fabricate a measurement"
    );

    let mut timed_engine = Engine::new(SlowReplayDevice(timed));
    let exe_t = timed_engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_t = timed_engine
        .add_entry(exe_t, &staged(&g, target, Submission::Replay))
        .unwrap();
    // First step records (capture + first replay, overhead-heavy); the span assertion below reads a
    // later pure-replay step.
    let _ = timed_engine
        .step(exe_t, decode_t, &step_inputs(&values), &mut NoSync)
        .unwrap();
    let time_t = timed_engine
        .step(exe_t, decode_t, &step_inputs(&values), &mut NoSync)
        .unwrap()
        .device_time();
    let poot_executor::DeviceTime::Measured(measured) = time_t else {
        panic!("a device-timed PTX device must report Measured, got {time_t:?}");
    };
    let span = measured
        .device_span
        .expect("PTX device timing reports one span per replay");
    assert!(span.as_nanos() > 0, "positive device span: {span:?}");
    assert!(
        span < Duration::from_millis(50),
        "the reported device span ({span:?}) must reflect the real CUDA-event-bracketed replay, not \
         the 200ms host-side sleep injected after it returns"
    );
}

// ── SC-007 ───────────────────────────────────────────────────────────────

/// SC-007: `Engine::stats().memory` (ultimately `PtxDevice::memory`, Card 547a's shared
/// `MemoryCounters`) reports a real, non-zero allocation under every role a qwen2 decode step actually
/// uses on real PTX hardware (Weight for consts, State for the KV cache, Activation for intermediates),
/// each role independently charged. `Engine::stats().memory` is `Device::memory` verbatim (the same
/// counter shapes every backend reports), never an engine-side tally.
///
/// MUTATION (recorded here, never left in the tree): in `poot-ptx-runtime/src/context.rs`'s
/// `PtxContext::alloc_storage`, ignore the `role` parameter and always call `self.alloc_zeroed(...,
/// BufferRole::Activation)`. Result: RED - `BufferRole::Weight` and `BufferRole::State` report zero
/// allocations (everything lands under `Activation` instead). Reverted: GREEN.
#[test]
fn sc007_memory_counters_track_each_role_ptx() {
    let Some(ptx) = try_ptx() else { return };
    let target = Device::target(&ptx);
    let g = decode_graph();
    let store = Arc::new(store_for(&[&g]));
    let mut engine = Engine::new(ptx);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode = engine
        .add_entry(exe, &staged(&g, target, Submission::Replay))
        .unwrap();
    let values = slot_values(&g, &[5], 0);
    let _ = engine
        .step(exe, decode, &step_inputs(&values), &mut NoSync)
        .unwrap();

    let stats = engine.stats();
    let by_role: HashMap<BufferRole, MemoryCounterSnapshot> = stats.memory.into_iter().collect();
    for role in [
        BufferRole::Weight,
        BufferRole::State,
        BufferRole::Activation,
    ] {
        let snap = by_role.get(&role).copied().unwrap_or_default();
        assert!(
            snap.allocations > 0,
            "{role:?}: expected at least one real allocation, got {snap:?}"
        );
        assert!(
            snap.live_bytes > 0,
            "{role:?}: expected live bytes charged, got {snap:?}"
        );
    }
}

// ── SC-009 ───────────────────────────────────────────────────────────────

/// Forwards every `Device` call to a real `PtxDevice`, except the first `dispatch` call, which returns
/// a synthetic error instead of reaching the driver - exercising the Engine's real CR30 abort path
/// (`Device::abort` -> the real `PtxContext::abort_capture`, a real open capture genuinely torn down)
/// without feeding the driver a malformed launch.
struct FaultyOnceDevice {
    inner: PtxDevice,
    armed: bool,
}

impl Device for FaultyOnceDevice {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = <PtxDevice as Device>::Error;

    fn target(&self) -> Target {
        self.inner.target()
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.inner.memory()
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.inner.allocate(role, storage, elems)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.inner.load_kernel(key, kernel)
    }
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error> {
        self.inner.begin(submission)
    }
    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        if self.armed {
            self.armed = false;
            return Err(poot_ptx_gpu::PtxGpuError::Graph(
                "sc009 injected dispatch failure".to_string(),
            ));
        }
        self.inner.dispatch(Dispatch {
            kernel: d.kernel,
            inputs: unsafe {
                std::mem::transmute::<
                    &[poot_executor::Arg<'_, Self>],
                    &[poot_executor::Arg<'_, PtxDevice>],
                >(d.inputs)
            },
            output: poot_executor::Arg {
                buffer: d.output.buffer,
                elems: d.output.elems,
            },
            threads: d.threads,
            workgroup: d.workgroup,
            work: d.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.inner.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
        self.inner.finish()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.inner.replay(recording)
    }
    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error> {
        self.inner.write(dst, bytes)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.inner.synchronize()
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        self.inner.device_time()
    }
    fn abort(&mut self) -> Result<(), Self::Error> {
        self.inner.abort()
    }
}

/// SC-009: a dispatch failure during an entry's first (recording) step on real PTX hardware aborts
/// exactly once through the real `PtxContext::abort_capture`, does not poison the engine, does not mark
/// the executable `NeedsReset` (CR30: only a replay-time failure does, since nothing committed), and
/// leaves every other executable on the same engine/device completely unaffected - proving the real
/// driver's capture-abort genuinely cleans up rather than leaking or corrupting shared context state.
/// `FaultyOnceDevice` is the fault injection; the assertions below exercise the real `abort()` path.
///
/// MUTATION (recorded here, never left in the tree): in `poot-ptx-gpu/src/device.rs`'s
/// `PtxDevice::abort`, delete the `self.ctx.abort_capture();` call (keep `self.pending_span = None;
/// Ok(())`). Result: RED - the real CUDA stream stays stuck in capture mode, so the retry step below
/// fails with `CaptureOpen` instead of succeeding. Reverted: GREEN.
#[test]
fn sc009_failure_injection_aborts_once_and_retains_other_executables_ptx() {
    let Some(inner) = try_ptx() else { return };
    let target = Device::target(&inner);
    let g = decode_graph();
    let store = Arc::new(store_for(&[&g]));

    let mut engine = Engine::new(FaultyOnceDevice { inner, armed: true });
    let exe_bad = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_bad = engine
        .add_entry(exe_bad, &staged(&g, target, Submission::Replay))
        .unwrap();
    let exe_ok = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_ok = engine
        .add_entry(exe_ok, &staged(&g, target, Submission::Replay))
        .unwrap();

    let values = slot_values(&g, &[5], 0);
    let failed = engine.step(exe_bad, decode_bad, &step_inputs(&values), &mut NoSync);
    assert!(
        failed.is_err(),
        "the injected dispatch failure must surface as a step error"
    );

    // CR30: a recording-time dispatch failure aborts once and does not mark NeedsReset - the same
    // executable must still be steppable (it simply re-records, since no recording survived the abort).
    let retried = engine.step(exe_bad, decode_bad, &step_inputs(&values), &mut NoSync);
    if let Err(e) = retried {
        panic!("a fresh step after an aborted recording must succeed on real PTX hardware: {e:?}");
    }

    // A completely different executable on the same engine/device must be unaffected by the real
    // abort_capture() call above: no poisoning, no leaked capture state.
    let mut oracle_state = HashMap::new();
    let expected = oracle_step(&g, &store, &values, &mut oracle_state);
    let got = bytemuck_f32(
        &engine
            .step(exe_ok, decode_ok, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    assert_close_rel(&got, &expected, 5e-3);
}

// ── SC-010 / SC-011 ──────────────────────────────────────────────────────

/// Forwards every `Device` call to a real `PtxDevice`, except `synchronize`, which fails exactly
/// once while `armed` is set (shared across the wrapper's clones via `Rc<Cell<_>>`, so a test can
/// arm it after the device has already been moved into an `Engine`). The real CR30 "non-fault
/// synchronize failure" path (card 549 F4, SC-010/SC-011 moved from 546a): a simulated device
/// timeout right after a real `dispatch`/`replay` already ran, so `Engine::step` marks the
/// executable `NeedsReset` over a genuinely pending resource, not a contrived one.
struct FaultyOnceSyncDevice {
    inner: PtxDevice,
    armed: Rc<Cell<bool>>,
}

impl Device for FaultyOnceSyncDevice {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = <PtxDevice as Device>::Error;

    fn target(&self) -> Target {
        self.inner.target()
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.inner.memory()
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.inner.allocate(role, storage, elems)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.inner.load_kernel(key, kernel)
    }
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error> {
        self.inner.begin(submission)
    }
    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        self.inner.dispatch(Dispatch {
            kernel: d.kernel,
            inputs: unsafe {
                std::mem::transmute::<
                    &[poot_executor::Arg<'_, Self>],
                    &[poot_executor::Arg<'_, PtxDevice>],
                >(d.inputs)
            },
            output: poot_executor::Arg {
                buffer: d.output.buffer,
                elems: d.output.elems,
            },
            threads: d.threads,
            workgroup: d.workgroup,
            work: d.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.inner.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
        self.inner.finish()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.inner.replay(recording)
    }
    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error> {
        self.inner.write(dst, bytes)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
        if self.armed.get() {
            self.armed.set(false);
            // The real replay already ran for real above; only the *proof* of completion is faked.
            let _ = self.inner.synchronize();
            return Err(poot_ptx_gpu::PtxGpuError::Graph(
                "sc010/sc011 injected synchronize failure (simulated device timeout)".to_string(),
            ));
        }
        self.inner.synchronize()
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        self.inner.device_time()
    }
    fn abort(&mut self) -> Result<(), Self::Error> {
        self.inner.abort()
    }
}

/// Total live bytes charged across every role (Card 547a): the one scalar both rows below check, so
/// neither hardcodes which specific role this fixture's decode step happens to use.
fn total_live_bytes(engine: &Engine<FaultyOnceSyncDevice>) -> u64 {
    engine
        .stats()
        .memory
        .into_iter()
        .map(|(_, snap)| snap.live_bytes)
        .sum()
}

/// Builds one engine over two executables of the same decode-kv fixture on real PTX hardware -
/// `exe_bad` and `exe_ok`, sharing the same device/native queue - steps `exe_bad` twice (a real
/// capture+replay, then a real replay that fails to synchronize), and `unload`s it while still
/// `NeedsReset`. Returns the engine (with `exe_ok` still loaded and un-stepped) and the live-byte
/// totals immediately before and after that `unload` call.
fn sc010_sc011_fixture() -> Option<(
    Engine<FaultyOnceSyncDevice>,
    ExecutableId,
    EntryId,
    u64,
    u64,
)> {
    let inner = device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())?;
    let target = Device::target(&inner);
    let g = decode_graph();
    let store = Arc::new(store_for(&[&g]));
    let armed = Rc::new(Cell::new(false));

    let mut engine = Engine::new(FaultyOnceSyncDevice {
        inner,
        armed: Rc::clone(&armed),
    });
    let exe_bad = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_bad = engine
        .add_entry(exe_bad, &staged(&g, target, Submission::Replay))
        .unwrap();
    let exe_ok = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let decode_ok = engine
        .add_entry(exe_ok, &staged(&g, target, Submission::Replay))
        .unwrap();

    let values = slot_values(&g, &[5], 0);
    // First step on exe_bad: records + replays + syncs for real, allocating its real buffers.
    engine
        .step(exe_bad, decode_bad, &step_inputs(&values), &mut NoSync)
        .expect("first exe_bad step must succeed to allocate real, live buffers");

    // Second step: the injected synchronize failure fires after a real dispatch/replay already ran,
    // so exe_bad's buffers are genuinely live when NeedsReset is set.
    armed.set(true);
    let failed = engine.step(exe_bad, decode_bad, &step_inputs(&values), &mut NoSync);
    assert!(
        failed.is_err(),
        "the injected synchronize failure must surface as a step error"
    );

    let before_unload = total_live_bytes(&engine);
    assert!(
        before_unload > 0,
        "exe_bad's real buffers must be live before unload: {before_unload}"
    );

    engine
        .unload(exe_bad)
        .expect("unload of a NeedsReset executable must still succeed");
    let after_unload = total_live_bytes(&engine);

    Some((engine, exe_ok, decode_ok, before_unload, after_unload))
}

/// SC-010: an executable `unload`ed while `NeedsReset` (completion of its last transaction not
/// proven - here, a real injected synchronize failure right after a real replay) retains its
/// buffers, modules and events rather than dropping them with the handle (moved from Card 546a,,
/// F4) - driven through `Engine::unload` and the 547a memory counters, not
/// `PtxContext` directly.
///
/// MUTATION (recorded here, never left in the tree): in `poot-executor/src/engine.rs`'s `unload`,
/// delete the `if loaded.needs_reset { self.pending_release.push(loaded); }` branch (back to an
/// unconditional `slot.take()` with nothing retaining it). Result: RED - `after_unload` drops to 0
/// immediately instead of staying at `before_unload`'s level. Reverted: GREEN.
#[test]
fn sc010_engine_unload_retains_a_needs_reset_executables_buffers_ptx() {
    let Some((_engine, _exe_ok, _decode_ok, before_unload, after_unload)) = sc010_sc011_fixture()
    else {
        return;
    };
    assert_eq!(
        after_unload, before_unload,
        "unload must retain a NeedsReset executable's buffers, not release them: \
         before_unload={before_unload} after_unload={after_unload}"
    );
}

/// SC-011: once completion is proven for the removed executable - here, by a real successful
/// `synchronize()` on the *same* device/native queue, from stepping the sibling `exe_ok` - its
/// unique role-counter allocations finally release (moved from Card 546a,
/// F4). Shares SC-010's fixture and mutation.
///
/// MUTATION: shares SC-010's `Engine::unload` edit above. Result: RED - `after_unload` (already 0
/// under the mutation) is not strictly less than `after_drain` (also 0 - nothing was ever retained
/// to release), so `after_drain < after_unload` fails. Reverted: GREEN.
#[test]
fn sc011_engine_drains_pending_release_once_synchronize_proves_completion_ptx() {
    let Some((mut engine, exe_ok, decode_ok, _before_unload, after_unload)) = sc010_sc011_fixture()
    else {
        return;
    };

    // A real step on exe_ok - a different executable, same device/native queue - synchronizes for
    // real, which (per the production code under test) proves every op this device had queued
    // before it, including exe_bad's retained-but-pending one, has completed.
    let values = slot_values(&decode_graph(), &[5], 0);
    engine
        .step(exe_ok, decode_ok, &step_inputs(&values), &mut NoSync)
        .expect("exe_ok's own step must succeed independently of exe_bad's earlier failure");

    let after_drain = total_live_bytes(&engine);
    assert!(
        after_drain < after_unload,
        "once a real synchronize proves completion, the retained executable's role-counter \
         allocations must finally release: after_unload={after_unload} after_drain={after_drain}"
    );
}

// ── Card 551b SC-001 / SC-003 (PTX leg) ─────────────────────────────────────

/// Card 551b's sampling suffix (`poot_graph_ir::ops::sampling::sample_head`, appended via
/// `Builder::resume` after a finished dense decode graph's raw logits - the same mechanism
/// `poot-llm`'s `driver::suffix::with_head` uses, exercised here directly at the IR/executor
/// level since `poot-ptx-gpu` cannot depend on `poot-llm`'s `Runner`) on real PTX hardware,
/// verified the same way the wgpu/ROCm legs are (`poot-llm`'s
/// `tests/card551b_sampling_suffix.rs`): a device suffix row and the CPU oracle (`poot_eval::eval`
/// on the identical suffix graph) must agree step by step given the same per-step seed/params and
/// the same read-back logits.
/// Returns `(base, suffix, seed_id, params_id)`: `base` (no `Sampler` slot - safe for
/// [`slot_values`], which panics on one) and `suffix` (its `Builder::resume` extension) share the
/// same token/pos/seq_len `ValueId`s, since `resume` never renumbers a graph's existing values.
fn card551b_suffix_graph(rule: SampleRule) -> (Graph, Graph, ValueId, ValueId) {
    let base = decode_graph();
    let base_for_slots = base.clone();
    let resumed = Builder::resume(base);
    let sh = sample_head(&resumed.builder, resumed.out, rule);
    let seed_id = sh.seed.expect("a non-Greedy rule declares a seed slot").id;
    let params_id = sh
        .params
        .expect("a non-Greedy rule declares a params slot")
        .id;
    let g = resumed.builder.finish_with_state(sh.out, &resumed.state);
    (base_for_slots, g, seed_id, params_id)
}

/// `values` plus the suffix's `seed`/`params` rows (card 551b's three fixed tags - `top_k` is
/// unused here, `SampleRule::Gumbel` declares none), so both [`step_inputs`] (the device) and
/// [`oracle_step`] (the CPU oracle, generic over any `Storage::Slot` by its `SlotKey`) bind them
/// identically.
fn card551b_push_sampler_rows(
    values: &mut Vec<SlotValue>,
    g: &Graph,
    seed_id: ValueId,
    params_id: ValueId,
    seed: i32,
    params: &[f32],
) {
    let seed_shape = g.meta(seed_id).aval.shape.clone();
    let params_shape = g.meta(params_id).aval.shape.clone();
    values.push(SlotValue {
        key: SlotKey::new(Slot::Sampler, Some("seed")),
        shape: seed_shape,
        f32s: vec![],
        i32s: vec![seed],
        dtype: DType::I32,
    });
    values.push(SlotValue {
        key: SlotKey::new(Slot::Sampler, Some("params")),
        shape: params_shape,
        f32s: params.to_vec(),
        i32s: vec![],
        dtype: DType::F32,
    });
}

/// MUTATION (recorded here, never left in the tree; card 551b): in this function, change
/// `ints[1]` (the non-finite flag) to always compare `< 0` regardless of its actual value - i.e.
/// return `(ints[0], -1)` unconditionally, the same defect `driver::suffix::read_tokens` getting
/// mutated to "ignore the non-finite column in the readback" produces on wgpu/ROCm. Result: RED -
/// `card551b_sc003_nan_logits_are_a_typed_fault_ptx` observes a finite token instead of the typed
/// fault (`assert!(non_finite >= 0, ...)` fails: `non_finite` is forced to `-1`).
/// `card551b_sc001_device_suffix_matches_host_oracle_ptx` stays GREEN (it never hits a non-finite
/// row). Reverted: GREEN.
fn card551b_read_suffix_output(bytes: &[u8]) -> (i32, i32) {
    let ints: &[i32] = bytemuck::cast_slice(bytes);
    (ints[0], ints[1])
}

/// Card 551b SC-001's PTX leg: a seeded dense decode's device suffix (`Sample(Gumbel)`, appended
/// via `Builder::resume`) matches the CPU oracle evaluated on the identical suffix graph, step by
/// step, over a real KV-carrying decode sequence on real PTX hardware.
#[test]
fn card551b_sc001_device_suffix_matches_host_oracle_ptx() {
    let Some(ptx) = try_ptx() else { return };
    let target = Device::target(&ptx);
    let temperature = 0.8f32;
    let (base_g, suffix_g, seed_id, params_id) = card551b_suffix_graph(SampleRule::Gumbel);
    let store = Arc::new(store_for(&[&suffix_g]));
    let mut engine = Engine::new(ptx);
    let exe = engine
        .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine
        .add_entry(exe, &staged(&suffix_g, target, Submission::Replay))
        .unwrap();

    let mut oracle_state = HashMap::new();
    let mut device_tokens = Vec::new();
    let mut oracle_tokens = Vec::new();
    for pos in 0..CAP {
        let token = ((pos * 7 + 3) % VOCAB) as u32;
        let seed = pos as i32 + 1;
        let params = vec![1.0 / temperature, f32::NEG_INFINITY, 1.0f32];

        let mut values = slot_values(&base_g, &[token], pos);
        card551b_push_sampler_rows(&mut values, &suffix_g, seed_id, params_id, seed, &params);

        let bytes = engine
            .step(exe, entry, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        let (device_token, device_non_finite) = card551b_read_suffix_output(&bytes);
        assert!(
            device_non_finite < 0,
            "pos={pos}: unexpected non-finite logit flagged by the device suffix"
        );
        device_tokens.push(device_token);

        let oracle_out = oracle_step(&suffix_g, &store, &values, &mut oracle_state);
        let oracle_ints: Vec<i32> = oracle_out.iter().map(|&v| v as i32).collect();
        assert!(
            oracle_ints[1] < 0,
            "pos={pos}: unexpected non-finite logit flagged by the CPU oracle"
        );
        oracle_tokens.push(oracle_ints[0]);
    }

    assert_eq!(
        device_tokens, oracle_tokens,
        "card 551b SC-001 (ptx): the device suffix must match the CPU oracle step by step on the \
         same read-back logits"
    );
    eprintln!("card551b SC-001 ptx: device={device_tokens:?} oracle={oracle_tokens:?}");
}

/// Card 551b SC-003's PTX leg: a dense decode whose first-step embedding (and therefore every
/// downstream logit) is NaN returns the typed sampler fault shape (`non_finite_index >= 0`, the
/// token forced to 0) from the device suffix on real PTX hardware, for both a greedy and a sampled
/// rule.
#[test]
fn card551b_sc003_nan_logits_are_a_typed_fault_ptx() {
    let Some(probe) = try_ptx() else { return };
    let target = Device::target(&probe);
    drop(probe);
    let poisoned_token = 5u32;

    for rule in [SampleRule::Greedy, SampleRule::Gumbel] {
        let base = decode_graph();
        let base_for_slots = base.clone();
        let resumed = Builder::resume(base);
        let sh = sample_head(&resumed.builder, resumed.out, rule);
        let (seed_id, params_id) = (sh.seed.map(|t| t.id), sh.params.map(|t| t.id));
        let suffix_g = resumed.builder.finish_with_state(sh.out, &resumed.state);

        let mut entries: HashMap<String, Vec<usize>> = HashMap::new();
        for &id in &suffix_g.inputs {
            let meta = suffix_g.meta(id);
            if meta.storage == Storage::Const {
                entries.insert(meta.name.clone().unwrap(), meta.aval.shape.clone());
            }
        }
        let mut builder = WeightStore::builder();
        for (name, shape) in entries {
            let mut data = const_data(&name, &shape);
            if name == "w.embed" {
                let hidden = HIDDEN;
                let row = poisoned_token as usize * hidden;
                data[row..row + hidden].fill(f32::NAN);
            }
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
            builder.insert(name, WeightEntry::Dense(dense)).unwrap();
        }
        let store = Arc::new(builder.build());

        let mut engine = Engine::new(PtxDevice::new().expect("just verified PTX is available"));
        let exe = engine
            .load_weights(Arc::clone(&store), poot_executor::WeightSource::ConstNames)
            .unwrap();
        let entry = engine
            .add_entry(exe, &staged(&suffix_g, target, Submission::Replay))
            .unwrap();

        let mut values = slot_values(&base_for_slots, &[poisoned_token], 0);
        if let (Some(seed_id), Some(params_id)) = (seed_id, params_id) {
            card551b_push_sampler_rows(
                &mut values,
                &suffix_g,
                seed_id,
                params_id,
                1,
                &[1.0 / 0.8, f32::NEG_INFINITY, 1.0],
            );
        }

        let bytes = engine
            .step(exe, entry, &step_inputs(&values), &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        let (token, non_finite) = card551b_read_suffix_output(&bytes);
        assert!(
            non_finite >= 0,
            "rule={rule:?}: a NaN-embedding decode step must flag a non-finite logit, got \
             (token={token}, non_finite={non_finite})"
        );
        assert_eq!(
            token, 0,
            "rule={rule:?}: a non-finite row's token is always forced to 0 (R-551a-2)"
        );
        eprintln!("card551b SC-003 ptx rule={rule:?}: (token={token}, non_finite={non_finite})");
    }
}
