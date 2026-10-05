//! Shared test support for the PTX device-contract tests (Card 549): one-shot and decode-entry
//! helpers over `Engine<PtxDevice>`, replacing `PtxGraphExecutor::run_resident[_kv]`/`capture_decode`/
//! `decode_step`. Every graph here is bound by name through a [`poot_quant::weights::WeightStore`]
//! built from the caller's `HashMap<ValueId, Value>` (Card 546a's Z8 name-equality binding), since
//! these fixtures have no `Runner` to supply one.
//!
//! `tests/common/mod.rs` compiles once per test binary that declares `mod common;`; each binary uses a
//! different subset (some only `run_resident`, some only `DecodeEntry`), so dead-code warnings here are
//! expected and suppressed rather than meaningful per binary.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_executor::{Device, EntryId, ExecutableId, Executor, NoSync, StepInputs};
use poot_graph_ir::{Graph, PackedSourceName, Storage, ValueId};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_ptx_gpu::PtxDevice;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::HostTensor;

/// `g` staged for the executor contract against `target` (Card 546a: `Submission::Replay` only).
pub fn staged_program(
    g: &Graph,
    target: Target,
    fusion: FusionPolicy,
) -> StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("stage ptx graph")
}

/// A [`WeightStore`] holding every `Storage::Const` value `g` declares, read from `inputs` by name
/// (Z8). A dense `Value` becomes one `WeightEntry::Dense`; a `Value::Packed` becomes one
/// `WeightEntry::Packed` keyed by its [`PackedSourceName`]'s `linear_id` (card 642's const naming)
/// - every role-component of one packed linear shares its owner's identical
/// `Arc<PackedPayload>`, so the first role seen for a `linear_id` inserts it and later roles are a
/// no-op, exactly mirroring `poot-llm`'s `Runner::load_on`/`weight_store`.
fn const_store(g: &Graph, inputs: &HashMap<ValueId, Value>) -> WeightStore {
    let mut builder = WeightStore::builder();
    let mut packed_linear_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage != Storage::Const {
            continue;
        }
        let name = meta.name.clone().expect("unnamed const");
        let value = inputs
            .get(&id)
            .unwrap_or_else(|| panic!("no value bound for const {name}"));
        match value {
            Value::Host(dense) => {
                let bytes: Arc<[u8]> =
                    Arc::from(bytemuck::cast_slice::<f32, u8>(dense.as_f32().unwrap()));
                let entry =
                    DenseWeight::try_new(poot_tensor::DType::F32, dense.shape().to_vec(), bytes)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                builder
                    .insert(name.clone(), WeightEntry::Dense(entry))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            Value::Packed(component) => {
                let parsed = PackedSourceName::parse(&name).unwrap_or_else(|| {
                    panic!("{name}: packed value with a non-packed-source name")
                });
                if packed_linear_ids.insert(parsed.linear_id().to_string()) {
                    builder
                        .insert(
                            parsed.linear_id().to_string(),
                            WeightEntry::Packed(Arc::clone(component.owner())),
                        )
                        .unwrap_or_else(|e| panic!("{}: {e}", parsed.linear_id()));
                }
            }
            other => {
                panic!("{name}: const value kind {other:?} is not supported by this test helper")
            }
        }
    }
    builder.build()
}

/// `inputs`'s `Storage::Slot` entries only, as the `SlotKey`-keyed [`StepInputs`] `step` takes.
fn slot_inputs<'a>(g: &Graph, inputs: &'a HashMap<ValueId, Value>) -> StepInputs<'a> {
    let mut step_inputs = StepInputs::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let Storage::Slot(_) = meta.storage else {
            continue;
        };
        let key = meta.slot_key().expect("slot without a SlotKey").clone();
        let value = inputs.get(&id).expect("no value bound for slot");
        let tensor = value
            .as_host()
            .expect("a slot value must be a dense tensor");
        let view = tensor.view();
        step_inputs.push(key, tensor.shape(), view);
    }
    step_inputs
}

/// Device-resident, stateless run of `g` on the executor contract: a fresh one-shot executable loaded
/// from `inputs`'s own consts, stepped once, unloaded. `g` must declare no carried state. The PTX
/// twin of `PtxGraphExecutor::run_resident`/`run_resident_kv` before Card 549 (state must already be
/// zero, since the contract always zero-seeds a freshly declared state buffer - every fixture below
/// that carries state starts from zero, matching the old tests' own "zero seeded" setup).
pub fn run_resident(
    device: &mut PtxDevice,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> HostTensor {
    run_resident_with_fusion(device, g, inputs, FusionPolicy::Full)
}

/// Like [`run_resident`], but with an explicit fusion policy (for a test that compares two policies'
/// output, e.g. a fused-vs-unfused regression).
pub fn run_resident_with_fusion(
    device: &mut PtxDevice,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
    fusion: FusionPolicy,
) -> HostTensor {
    let mut engine = poot_executor::Engine::new(PtxDeviceProxy(device));
    let exe = engine
        .load_weights(
            Arc::new(const_store(g, inputs)),
            poot_executor::WeightSource::ConstNames,
        )
        .expect("load_weights");
    let staged = staged_program(g, Device::target(engine.device()), fusion);
    let entry = engine.add_entry(exe, &staged).expect("add_entry");
    let bytes = engine
        .step(exe, entry, &slot_inputs(g, inputs), &mut NoSync)
        .expect("step")
        .read()
        .expect("read");
    engine.remove_entry(exe, entry).expect("remove_entry");
    engine.unload(exe).expect("unload");
    HostTensor::f32(
        g.aval(g.output).shape.clone(),
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
    )
}

/// Like [`run_resident`], for a graph whose output is I32 (card 551a: `SampleToken`'s `(token,
/// non_finite_index)` pair) rather than F32.
pub fn run_resident_i32(
    device: &mut PtxDevice,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Vec<i32> {
    let mut engine = poot_executor::Engine::new(PtxDeviceProxy(device));
    let exe = engine
        .load_weights(
            Arc::new(const_store(g, inputs)),
            poot_executor::WeightSource::ConstNames,
        )
        .expect("load_weights");
    let staged = staged_program(g, Device::target(engine.device()), FusionPolicy::Full);
    let entry = engine.add_entry(exe, &staged).expect("add_entry");
    let bytes = engine
        .step(exe, entry, &slot_inputs(g, inputs), &mut NoSync)
        .expect("step")
        .read()
        .expect("read");
    engine.remove_entry(exe, entry).expect("remove_entry");
    engine.unload(exe).expect("unload");
    bytemuck::cast_slice::<u8, i32>(&bytes).to_vec()
}

/// A long-lived decode entry over `g`, loaded with `inputs`'s own consts: `step` replays it once per
/// call, with the carried state (if any) evolving in place across calls, exactly like the pre-549
/// `CapturedDecode`/`decode_step` pair but through the executor contract.
pub struct DecodeEntry<'d> {
    engine: poot_executor::Engine<PtxDeviceProxy<'d>>,
    exe: ExecutableId,
    entry: EntryId,
    output_shape: Vec<usize>,
}

impl<'d> DecodeEntry<'d> {
    pub fn capture(
        device: &'d mut PtxDevice,
        g: &Graph,
        inputs: &HashMap<ValueId, Value>,
        fusion: FusionPolicy,
    ) -> Self {
        let mut engine = poot_executor::Engine::new(PtxDeviceProxy(device));
        let exe = engine
            .load_weights(
                Arc::new(const_store(g, inputs)),
                poot_executor::WeightSource::ConstNames,
            )
            .expect("load_weights");
        let staged = staged_program(g, Device::target(engine.device()), fusion);
        let entry = engine.add_entry(exe, &staged).expect("add_entry");
        Self {
            engine,
            exe,
            entry,
            output_shape: g.aval(g.output).shape.clone(),
        }
    }

    /// Step once with `inputs`'s `Storage::Slot` values (e.g. this token/position/mask) and read the
    /// primary output back.
    pub fn step(&mut self, g: &Graph, inputs: &HashMap<ValueId, Value>) -> HostTensor {
        let step_inputs = slot_inputs(g, inputs);
        let bytes = self
            .engine
            .step(self.exe, self.entry, &step_inputs, &mut NoSync)
            .expect("step")
            .read()
            .expect("read");
        HostTensor::f32(
            self.output_shape.clone(),
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
        )
    }

    pub fn memory(
        &self,
    ) -> Vec<(
        poot_executor::BufferRole,
        poot_executor::MemoryCounterSnapshot,
    )> {
        self.engine.stats().memory
    }

    /// Read back a carried state buffer's current contents by name, through a one-shot "probe" entry
    /// whose own graph output IS the state value (no computation, no state-pair write) - an ordinary
    /// production-shaped entry sharing the real entry's buffer by (name, aval, storage), Z5, never a
    /// test-only accessor into the engine's internals.
    pub fn read_state(&mut self, name: &str, shape: Vec<usize>) -> HostTensor {
        let b = poot_graph_ir::Builder::new();
        let state = b.state_input(
            name,
            poot_graph_ir::TensorType::f32(shape.clone()),
            poot_graph_ir::StateRole::Recurrent,
        );
        let probe = b.finish(state);
        let staged = staged_program(
            &probe,
            Device::target(self.engine.device()),
            FusionPolicy::Full,
        );
        let probe_entry = self
            .engine
            .add_entry(self.exe, &staged)
            .expect("add_entry (probe)");
        let bytes = self
            .engine
            .step(self.exe, probe_entry, &StepInputs::new(), &mut NoSync)
            .expect("probe step")
            .read()
            .expect("probe read");
        self.engine
            .remove_entry(self.exe, probe_entry)
            .expect("remove_entry (probe)");
        HostTensor::f32(shape, bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
    }
}

impl Drop for DecodeEntry<'_> {
    fn drop(&mut self) {
        let _ = self.engine.remove_entry(self.exe, self.entry);
        let _ = self.engine.unload(self.exe);
    }
}

/// `Engine<D>` needs to own its device, but these tests want to keep reusing one `PtxDevice` (and its
/// resident kernel/module caches) across several one-shot `run_resident`/`DecodeEntry` calls in the
/// same `#[test]` fn, as the pre-549 `PtxGraphExecutor` did implicitly via its own cache fields. This
/// thin `Device` forwarder lets `Engine::new` borrow a `&mut PtxDevice` instead of owning one, so nothing
/// here re-opens a CUDA context per call.
pub struct PtxDeviceProxy<'d>(&'d mut PtxDevice);

impl Device for PtxDeviceProxy<'_> {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = <PtxDevice as Device>::Error;

    fn target(&self) -> Target {
        self.0.target()
    }
    fn memory(
        &self,
    ) -> Vec<(
        poot_executor::BufferRole,
        poot_executor::MemoryCounterSnapshot,
    )> {
        self.0.memory()
    }
    fn allocate(
        &mut self,
        role: poot_executor::BufferRole,
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
    fn dispatch(&mut self, d: poot_executor::Dispatch<'_, Self>) -> Result<(), Self::Error> {
        // SAFETY-free reinterpretation: `Self::Buffer`/`Self::Kernel` are literally `PtxDevice`'s own
        // associated types (see above), so a `Dispatch<Self>` is layout-identical to a
        // `Dispatch<PtxDevice>`; only the generic parameter differs.
        self.0.dispatch(poot_executor::Dispatch {
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
        self.0.replay(recording)
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
