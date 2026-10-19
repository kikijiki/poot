//! POOT-1206: an engine releases an executable's recordings before the weights, state and resident
//! buffers they address.
//!
//! A `Device::Recording` owns what it addresses, but the engine does not lean on that: `Loaded` drops its
//! entries (and so their recordings) first. A fake device logs every buffer and recording drop in
//! order. Model-free: only real SPIR-V codegen to compile the fixture's kernels, which the fake never
//! runs.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{Device, DeviceTime, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::{BinOp, Builder, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    production_plan::Submission, staged::DevicePlacement, staged::ExpertPlacement,
    staged::Partition, staged::StagedProgram, staged::TargetSet, staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightMap, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::{DType, HostView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Released {
    Recording,
    Buffer(BufferRole),
}

type ReleaseLog = Rc<RefCell<Vec<Released>>>;

struct LoggedBuffer {
    role: BufferRole,
    log: ReleaseLog,
}

impl Drop for LoggedBuffer {
    fn drop(&mut self) {
        self.log.borrow_mut().push(Released::Buffer(self.role));
    }
}

struct LoggedRecording(ReleaseLog);

impl Drop for LoggedRecording {
    fn drop(&mut self) {
        self.0.borrow_mut().push(Released::Recording);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("fake device failed")]
struct Failed;

struct LoggingDevice {
    log: ReleaseLog,
    memory: poot_runtime_common::MemoryCounters,
}

impl Device for LoggingDevice {
    type Buffer = LoggedBuffer;
    type Kernel = ();
    type Recording = LoggedRecording;
    type Error = Failed;

    fn target(&self) -> Target {
        target()
    }

    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        _storage: poot_target::BufferStorage,
        _elems: usize,
    ) -> Result<LoggedBuffer, Failed> {
        Ok(LoggedBuffer {
            role,
            log: Rc::clone(&self.log),
        })
    }

    fn buffer_elems(&self, _buffer: &LoggedBuffer) -> usize {
        usize::MAX
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), Failed> {
        Ok(())
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), Failed> {
        Ok(())
    }

    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), Failed> {
        Ok(())
    }

    fn copy(&mut self, _src: &LoggedBuffer, _dst: &LoggedBuffer) -> Result<(), Failed> {
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<LoggedRecording>, Failed> {
        Ok(Some(LoggedRecording(Rc::clone(&self.log))))
    }

    fn begin_step(&mut self) -> Result<(), Failed> {
        Ok(())
    }

    fn replay(&mut self, _recording: &LoggedRecording) -> Result<(), Failed> {
        Ok(())
    }

    fn write_with(
        &mut self,
        _dst: &LoggedBuffer,
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Failed> {
        Ok(())
    }

    fn read(&mut self, _src: &LoggedBuffer, out: &mut [u8]) -> Result<(), Failed> {
        out.fill(0);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), Failed> {
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), Failed> {
        Ok(())
    }
}

fn target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

/// One block with a weight and a carried state pair, so the executable owns weight and state buffers.
fn program() -> (StagedProgram, Arc<WeightStore>, Arc<WeightMap>) {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let scaled = b.binary(BinOp::Mul, x, w).unwrap();
    let k_in = b
        .state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent)
        .unwrap();
    let k_out = b.binary(BinOp::Add, k_in, scaled).unwrap();
    let out = b.binary(BinOp::Add, scaled, k_out).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(out).with_state(&[(k_in, k_out)]))
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
    let mut names = poot_graph_ir::test_support::const_weights::ConstWeights::new(&store);
    let g = names.graph(&g);
    let map = Arc::new(names.map());
    let program = compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target()),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
    .expect("the fixture compiles for the fake target");
    (program, store, map)
}

/// SC-003: dropping the engine releases every recording before any weight or state buffer.
///
/// Mutation: move `entries` back below `state` and `resident` in `Loaded`; the weight and state
/// buffers then drop before the recording and this row goes red.
#[test]
fn an_executable_releases_its_recordings_before_its_weights_and_state() {
    let log = ReleaseLog::default();
    let device = LoggingDevice {
        log: Rc::clone(&log),
        memory: poot_runtime_common::MemoryCounters::new(),
    };
    let mut engine = Engine::new(device);
    let (program, store, map) = program();
    let exe = engine.load_weights(store, map).unwrap();
    let entry = engine.add_entry(exe, &program).unwrap();
    // The entry records its stages on its first step.
    let values = [1.5f32, -0.25, 3.0, 0.75];
    let shape = [4usize];
    let mut inputs = StepInputs::new();
    inputs.push(
        SlotKey::new(Slot::Activation, None),
        &shape,
        HostView::new(DType::F32, 4, bytemuck::cast_slice(&values)).unwrap(),
    );
    engine
        .step(exe, entry, &inputs, &mut NoSync)
        .expect("the step runs")
        .read()
        .expect("the output reads");
    assert!(
        log.borrow().is_empty(),
        "nothing is released while the engine lives"
    );

    drop(engine);

    let released = log.borrow().clone();
    let last_recording = released
        .iter()
        .rposition(|release| *release == Released::Recording)
        .expect("the entry's recording was released");
    let first_weight_or_state = released
        .iter()
        .position(|release| {
            matches!(
                release,
                Released::Buffer(BufferRole::Weight | BufferRole::State)
            )
        })
        .expect("the weight and state buffers were released");
    assert!(
        last_recording < first_weight_or_state,
        "recordings must drop before weights and state: {released:?}"
    );
}
