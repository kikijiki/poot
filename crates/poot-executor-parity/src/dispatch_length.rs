//! POOT-1205: the engine refuses a dispatch whose asserted length exceeds the buffer bound to it, before
//! any device sees the dispatch, on whichever real `Device` the caller opens.
//!
//! The fixture is the arena program (`a + b` reduced, then added to a reduced `c + c`; four dispatches,
//! the first writing a 64-element activation). [`ShortActivations`] wraps the caller's device and
//! allocates every multi-element activation buffer one element short of what the plan asks, as a device whose
//! allocation disagrees with the plan would; the length a dispatch then asserts for it is one element
//! more than its buffer holds. The device is real, so its `buffer_elems` reports the real (short)
//! capacity, and the wrapper counts the dispatches that reach the device.

use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    Arg, Device, DeviceTime, Dispatch, DispatchError, Engine, ExecError, Executor, NoSync,
    StepInputs, WeightUpload,
};
use poot_graph_ir::op::{BinOp, RedOp};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::{BufferRole, CompiledKernel, MemoryCounterSnapshot};
use poot_target::BufferStorage;
use poot_tensor::DType;

const K: usize = 64;

/// A device that allocates each activation buffer one element short, and counts the dispatches it is
/// handed.
struct ShortActivations<D> {
    inner: D,
    dispatches: Rc<Cell<usize>>,
}

impl<D: Device> Device for ShortActivations<D> {
    type Buffer = D::Buffer;
    type Kernel = D::Kernel;
    type Recording = D::Recording;
    type Error = D::Error;

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
        // A one-element buffer stays whole: a zero-byte allocation is a driver error, not the short buffer
        // under test.
        let short = if role == BufferRole::Activation && elems > 1 {
            elems - 1
        } else {
            elems
        };
        self.inner.allocate(role, storage, short)
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
        self.dispatches.set(self.dispatches.get() + 1);
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
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
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
        fill: &mut dyn FnMut(Range<usize>, &mut [u8]),
    ) -> Result<(), Self::Error> {
        self.inner.write_with(dst, offset, len, fill)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
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

/// The fixture as a named graph with its weight store.
fn fixture() -> (
    poot_graph_ir::Graph,
    Arc<WeightStore>,
    Arc<poot_quant::weights::WeightMap>,
) {
    let b = Builder::for_tests();
    let ta = b.constant("a", TensorType::f32(vec![K])).unwrap();
    let tb = b.constant("b", TensorType::f32(vec![K])).unwrap();
    let v1 = b.binary(BinOp::Add, ta, tb).unwrap();
    let s1 = b.reduce(RedOp::Sum, v1, 0, true).unwrap();
    let tc = b.constant("c", TensorType::f32(vec![1])).unwrap();
    let v2 = b.binary(BinOp::Add, tc, tc).unwrap();
    let s2 = b.reduce(RedOp::Sum, v2, 0, true).unwrap();
    let out = b.binary(BinOp::Add, s1, s2).unwrap();
    let graph = b.finish(out).unwrap();

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
    let mut names = ConstWeights::new(&store);
    let named = names.graph(&graph);
    let map = Arc::new(names.map());
    (named, store, map)
}

/// Drive the fixture through `Engine::step` on `device` with every activation buffer one element short:
/// the step is refused with [`DispatchError::ArgExceedsBuffer`] naming a length exactly one beyond its
/// buffer, and the device was handed no dispatch.
pub fn over_length_dispatch_is_refused<D: Device + 'static>(device: D) {
    let (graph, store, map) = fixture();
    let dispatches = Rc::new(Cell::new(0));
    let device = ShortActivations {
        inner: device,
        dispatches: Rc::clone(&dispatches),
    };
    let target = device.target();
    let program = compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("the fixture compiles");
    let mut engine = Engine::new(device);
    let exe = engine.load_weights(store, map).unwrap();
    let entry = engine.add_entry(exe, &program).unwrap();

    let error = engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .err()
        .expect("a length beyond its buffer must be refused");

    let ExecError::Dispatch(refusal) = error else {
        panic!("expected the shared dispatch error, got {error:?}");
    };
    let DispatchError::ArgExceedsBuffer {
        elems, capacity, ..
    } = *refusal;
    assert_eq!(
        elems as usize,
        capacity + 1,
        "the asserted length is one element beyond the buffer bound to it"
    );
    assert_eq!(
        dispatches.get(),
        0,
        "the device must never be handed a refused dispatch"
    );
}
