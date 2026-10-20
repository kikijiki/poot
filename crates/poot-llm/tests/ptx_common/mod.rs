//! Shared test support for the PTX device-contract tests (Card 549): the one-shot
//! helper over `Engine<PtxDevice>`, replacing `PtxGraphExecutor::run_resident[_kv]`/`capture_decode`/
//! `decode_step`. Every graph here is bound by name through a [`poot_quant::weights::WeightStore`]
//! built from the caller's `HashMap<ValueId, Value>` through a `ConstWeights` map, since
//! these fixtures have no driver to supply one.
//!
//! `tests/ptx_common/mod.rs` compiles once per test binary that declares `mod ptx_common;`; each binary uses a
//! different subset of these helpers, so the blanket allow below is
//! the only way to share them (a per-binary `dead_code` is never meaningful here).
#![allow(
    dead_code,
    reason = "compiled once per test binary, each using a different subset"
)]

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_executor::{Device, ExecutableId, Executor, NoSync, StepInputs};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Graph, PackedSourceName, Storage, ValueId};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_ptx_device::PtxDevice;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::HostTensor;

/// `g` staged for the executor contract against `target`.
pub(crate) fn staged_program(g: &Graph, target: Target) -> StagedProgram {
    let g = g.clone();
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("stage ptx graph")
}

/// A [`WeightStore`] holding every `Storage::Const` value `g` declares, read from `inputs` by name
/// (Z8). A dense `Value` becomes one `WeightEntry::Dense`; a `Value::Packed` becomes one
/// `WeightEntry::Packed` keyed by its [`PackedSourceName`]'s `linear_id` (card 642's const naming),
/// where every role-component of one packed linear shares its owner's identical
/// `Arc<PackedPayload>`, so the first role seen for a `linear_id` inserts it and later roles are a
/// no-op, exactly mirroring `poot-llm`'s driver weight store.
fn const_store(g: &Graph, inputs: &HashMap<ValueId, Value>) -> WeightStore {
    let mut builder = WeightStore::builder();
    let mut packed_linear_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for &id in g.inputs() {
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
                let (dtype, bytes): (_, StoredBytes) = match dense.dtype() {
                    poot_tensor::DType::F32 => (
                        poot_tensor::DType::F32,
                        StoredBytes::from(
                            bytemuck::cast_slice::<f32, u8>(dense.as_f32().unwrap()).to_vec(),
                        ),
                    ),
                    poot_tensor::DType::BF16 => (
                        poot_tensor::DType::BF16,
                        StoredBytes::from(
                            bytemuck::cast_slice::<u16, u8>(dense.as_half().unwrap()).to_vec(),
                        ),
                    ),
                    poot_tensor::DType::I32 => (
                        poot_tensor::DType::I32,
                        StoredBytes::from(
                            bytemuck::cast_slice::<i32, u8>(dense.as_i32().unwrap()).to_vec(),
                        ),
                    ),
                    other => panic!(
                        "{name}: dense const dtype {other:?} is not supported by this test helper"
                    ),
                };
                let entry = DenseWeight::try_new(dtype, dense.shape().to_vec(), bytes)
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
            // An expert stack: its experts are the store entries `{stack id}.{expert}`, which
            // `ConstWeights` views as one `ExpertStack`.
            Value::PackedStack(component) => {
                let parsed = PackedSourceName::parse(&name).unwrap_or_else(|| {
                    panic!("{name}: packed value with a non-packed-source name")
                });
                if packed_linear_ids.insert(parsed.linear_id().to_string()) {
                    let stack = component.stack();
                    for expert in 0..stack.experts() {
                        let key = format!("{}.{expert}", parsed.linear_id());
                        builder
                            .insert(
                                key.clone(),
                                WeightEntry::Packed(Arc::clone(stack.owner(expert))),
                            )
                            .unwrap_or_else(|e| panic!("{key}: {e}"));
                    }
                }
            }
            other => {
                panic!("{name}: const value kind {other:?} is not supported by this test helper")
            }
        }
    }
    builder.build()
}

/// Load `inputs`'s consts as one executable bound through a weight map, and return `g` with its
/// consts named by the ids the map holds: stage that graph, not `g`.
fn load_consts(
    engine: &mut poot_executor::Engine<PtxDeviceProxy<'_>>,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (ExecutableId, Graph) {
    let store = const_store(g, inputs);
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(g);
    let exe = engine
        .load_weights(Arc::new(store.clone()), Arc::new(weights.map()))
        .expect("load_weights");
    (exe, named)
}

/// `inputs`'s `Storage::Slot` entries only, as the `SlotKey`-keyed [`StepInputs`] `step` takes.
fn slot_inputs<'a>(g: &Graph, inputs: &'a HashMap<ValueId, Value>) -> StepInputs<'a> {
    let mut step_inputs = StepInputs::new();
    for &id in g.inputs() {
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
/// twin of the deleted `PtxGraphExecutor::run_resident`/`run_resident_kv` before Card 549 (state must already be
/// zero, since the contract always zero-seeds a freshly declared state buffer - every fixture below
/// that carries state starts from zero, matching the old tests' own "zero seeded" setup).
pub(crate) fn run_resident(
    device: &mut PtxDevice,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> HostTensor {
    run_resident_prepared(device, g, inputs, Graph::clone)
}

/// [`run_resident`] over `prepare(named)`: the graph with its consts named by the weight map, rewritten by
/// `prepare` (a legalization for a smaller buffer limit than the device's) before it is compiled for the device.
pub(crate) fn run_resident_prepared(
    device: &mut PtxDevice,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
    prepare: impl FnOnce(&Graph) -> Graph,
) -> HostTensor {
    let mut engine = poot_executor::Engine::new(PtxDeviceProxy(device));
    let (exe, named) = load_consts(&mut engine, g, inputs);
    let staged = staged_program(&prepare(&named), Device::target(engine.device()));
    let entry = engine.add_entry(exe, &staged).expect("add_entry");
    let bytes = engine
        .step(exe, entry, &slot_inputs(g, inputs), &mut NoSync)
        .expect("step")
        .read()
        .expect("read");
    engine.remove_entry(exe, entry).expect("remove_entry");
    engine.unload(exe).expect("unload");
    HostTensor::f32(
        g.aval(g.result()).shape.clone(),
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
    )
}

/// `Engine<D>` needs to own its device, but these tests want to keep reusing one `PtxDevice` (and its
/// resident kernel/module caches) across several one-shot `run_resident` calls in the
/// same `#[test]` fn, as the pre-549 `PtxGraphExecutor` did implicitly via its own cache fields. This
/// thin `Device` forwarder lets `Engine::new` borrow a `&mut PtxDevice` instead of owning one, so nothing
/// here re-opens a CUDA context per call.
pub(crate) struct PtxDeviceProxy<'d>(&'d mut PtxDevice);

impl Device for PtxDeviceProxy<'_> {
    type Buffer = <PtxDevice as Device>::Buffer;
    type Kernel = <PtxDevice as Device>::Kernel;
    type Recording = <PtxDevice as Device>::Recording;
    type Error = <PtxDevice as Device>::Error;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        self.0.copy_mechanism()
    }
    fn backend_name(&self) -> &'static str {
        self.0.backend_name()
    }

    fn target(&self) -> Target {
        self.0.target()
    }
    fn memory(
        &self,
    ) -> Vec<(
        poot_runtime_common::BufferRole,
        poot_runtime_common::MemoryCounterSnapshot,
    )> {
        self.0.memory()
    }
    fn allocate(
        &mut self,
        role: poot_runtime_common::BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.0.allocate(role, storage, elems)
    }
    fn buffer_elems(&self, buffer: &Self::Buffer) -> usize {
        self.0.buffer_elems(buffer)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.0.load_kernel(key, kernel)
    }
    fn begin(&mut self) -> Result<(), Self::Error> {
        self.0.begin()
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
            launch: d.launch,
            work: d.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.0.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Self::Recording, Self::Error> {
        self.0.finish()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.0.replay(recording)
    }
    fn write_with(
        &mut self,
        dst: &Self::Buffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Self::Error> {
        self.0.write_with(dst, offset, len, fill)
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
