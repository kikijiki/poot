//! POOT-1221 (R2-C5-003): one refusal row per index and range guard on a pooled-expert sync. The
//! host's `SyncWrites` are untrusted data to the executor: it validates every range against the site's
//! tables before it writes a byte. Each row drives the production `Engine::step` over a compiled
//! pooled program with a host that answers the sync with an input exactly one past the line, and
//! requires the named `BindError`; the matching control one step inside the line must be accepted, so
//! a guard that refuses everything does not pass. A refused write publishes nothing and the executable
//! then needs a complete reset. Model-free: a recording `Device`, real SPIR-V codegen.
//!
//! The loader's two guards of the same card (a tensor range one byte past the file, two shards
//! declaring one name) are rows of `poot-load`'s `hostile_inputs`.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1221 report):
//! GR-1 `apply_sync_writes` relaxes `expert >= *experts` to `expert > *experts`; GR-2 relaxes
//! `values.len() != *experts` to `values.len() > *experts` (a short map passes); GR-3 relaxes
//! `end > *pool_rows` to `end > *pool_rows + 1`.

use std::sync::Arc;

use poot_executor::{
    BindError, Device, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor, HostSync,
    RowFill, StepInputs, SyncReads, SyncWrites,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, TensorType};
use poot_graph_plan::{
    CompileOptions, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::SyncPoint, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{
    DenseWeight, ExpertProj, MoeRole, WeightEntry, WeightId, WeightKey, WeightMap, WeightRole,
    WeightStore, WeightView,
};
use poot_runtime_common::{BufferRole, MemoryCounters};
use poot_target::{Backend, BufferStorage, DeviceCaps};
use poot_tensor::DType;

const EXPERTS: usize = 6;
const POOL: usize = 3;
const WIDTH: usize = 4;
const INTER: usize = 2;

type Never = std::convert::Infallible;

/// A device that records allocations and writes and runs nothing.
#[derive(Default)]
struct Recorder {
    allocations: Vec<(BufferRole, BufferStorage, usize)>,
    writes: usize,
    memory: MemoryCounters,
}

impl Device for Recorder {
    type Buffer = usize;
    type Kernel = ();
    type Recording = ();
    type Error = Never;

    fn target(&self) -> poot_graph_plan::Target {
        poot_graph_plan::Target::Gpu {
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
        storage: BufferStorage,
        elems: usize,
    ) -> Result<usize, Never> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        self.allocations.push((role, storage, elems));
        Ok(self.allocations.len() - 1)
    }
    fn buffer_elems(&self, buffer: &usize) -> usize {
        self.allocations[*buffer].2
    }
    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), Never> {
        Ok(())
    }
    fn begin(&mut self) -> Result<(), Never> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), Never> {
        Ok(())
    }
    fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), Never> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Option<()>, Never> {
        Ok(Some(()))
    }
    fn replay(&mut self, _recording: &()) -> Result<(), Never> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &usize,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), Never> {
        fill(offset..offset + len, &mut vec![0u8; len]);
        self.writes += 1;
        Ok(())
    }
    fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), Never> {
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), Never> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), Never> {
        Ok(())
    }
}

fn moe_roles() -> [WeightId; 3] {
    [
        WeightId::layer(0, WeightRole::Moe(MoeRole::Router)),
        WeightId::layer(0, WeightRole::Moe(MoeRole::Stack(ExpertProj::GateUp))),
        WeightId::layer(0, WeightRole::Moe(MoeRole::Stack(ExpertProj::Down))),
    ]
}

/// One routed decode layer over a one-token activation `x`.
fn routed_decode() -> Graph {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 1, WIDTH]))
        .unwrap();
    let _scope = b.layer_scope(0);
    let [router, gate_up, down] = moe_roles();
    let router = b
        .constant(&router.const_name(), TensorType::f32(vec![WIDTH, EXPERTS]))
        .unwrap();
    let gate_up = b
        .constant(
            &gate_up.const_name(),
            TensorType::f32(vec![EXPERTS, WIDTH, 2 * INTER]),
        )
        .unwrap();
    let down = b
        .constant(
            &down.const_name(),
            TensorType::f32(vec![EXPERTS, INTER, WIDTH]),
        )
        .unwrap();
    let y = poot_graph_ir::ops::moe(&b, x, router, gate_up, down, EXPERTS, 2, INTER).unwrap();
    let out = b.binary(BinOp::Add, x, y).unwrap();
    b.finish(out).unwrap()
}

fn routed_weights() -> (Arc<WeightStore>, Arc<WeightMap>) {
    let shapes = [
        vec![WIDTH, EXPERTS],
        vec![EXPERTS, WIDTH, 2 * INTER],
        vec![EXPERTS, INTER, WIDTH],
    ];
    let mut store = WeightStore::builder();
    let mut views = Vec::new();
    for (n, (id, shape)) in moe_roles().into_iter().zip(shapes).enumerate() {
        let key = id.const_name();
        let elems: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..elems)
            .flat_map(|i| ((n + 1) as f32 * 1000.0 + i as f32).to_le_bytes())
            .collect();
        store
            .insert(
                key.as_str(),
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap(),
                ),
            )
            .unwrap();
        views.push((id, WeightView::Stored(WeightKey::from(key.as_str()))));
    }
    let store = Arc::new(store.build());
    let mut map = WeightMap::builder(&store);
    for (id, view) in views {
        map.map(id, view).unwrap();
    }
    let map = Arc::new(map.build());
    (store, map)
}

/// A host that answers every sync with the same writes.
struct FixedHost(SyncWrites);

impl HostSync for FixedHost {
    fn sync(&mut self, _point: &SyncPoint, _reads: &SyncReads) -> Result<SyncWrites, ExecError> {
        Ok(self.0.clone())
    }
}

fn pooled_engine() -> (Engine<Recorder>, ExecutableId, EntryId) {
    let mut engine = Engine::new(Recorder::default());
    let program = compile_staged(
        &routed_decode(),
        &TargetSet::single(DeviceId(0), engine.device().target()),
        &Partition {
            experts: ExpertPlacement::Pooled {
                slots_per_layer: POOL,
            },
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("the routed fixture pools");
    let (weights, mapped) = routed_weights();
    let exe = engine.load_weights(weights, mapped).unwrap();
    let entry = engine.add_entry(exe, &program).unwrap();
    (engine, exe, entry)
}

/// One step of a fresh pooled engine whose host answers the sync with `writes`: what the step
/// returned and how many device writes landed beyond the step's own input write.
fn step_with(writes: SyncWrites) -> (Result<(), ExecError>, usize) {
    let (mut engine, exe, entry) = pooled_engine();
    let bytes: Vec<u8> = (0..WIDTH)
        .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
        .collect();
    let mut inputs = StepInputs::new();
    inputs.push(
        poot_graph_ir::SlotKey::new(Slot::Activation, Some("x")),
        &[1, 1, WIDTH],
        poot_tensor::HostView::new(DType::F32, WIDTH, &bytes).unwrap(),
    );
    let before = engine.device().writes;
    let outcome = engine
        .step(exe, entry, &inputs, &mut FixedHost(writes))
        .map(|_| ());
    let landed = engine.device().writes - before;
    (outcome, landed)
}

fn fill(table: usize, first_row: usize, experts: Vec<usize>) -> SyncWrites {
    SyncWrites {
        rows: vec![RowFill {
            table,
            first_row,
            experts,
        }],
        map: None,
    }
}

fn bind_error(outcome: Result<(), ExecError>) -> BindError {
    match outcome {
        Err(ExecError::Bind(error)) => *error,
        other => panic!("expected a bind refusal, got {other:?}"),
    }
}

/// An expert id one past the table is `ExpertRange` naming it; the last expert is accepted, and the
/// refused fill wrote nothing but the step's input.
#[test]
fn an_expert_id_one_past_the_table_is_refused() {
    let (accepted, _) = step_with(fill(0, 0, vec![EXPERTS - 1]));
    accepted.expect("the last expert of the table is a valid fill");

    let (outcome, landed) = step_with(fill(0, 0, vec![EXPERTS]));
    let BindError::ExpertRange {
        expert, experts, ..
    } = bind_error(outcome)
    else {
        panic!("expected ExpertRange");
    };
    assert_eq!((expert, experts), (EXPERTS, EXPERTS));
    assert_eq!(landed, 1, "only the step's own input write may land");
}

/// A map one element short (or one long) is `MapLength` with both counts; the full map is accepted.
#[test]
fn a_map_one_element_off_is_refused() {
    let map = |len: usize| SyncWrites {
        rows: Vec::new(),
        map: Some(vec![-1; len]),
    };
    let (accepted, _) = step_with(map(EXPERTS));
    accepted.expect("a map of one entry per expert is valid");

    for got in [EXPERTS - 1, EXPERTS + 1] {
        let (outcome, landed) = step_with(map(got));
        let BindError::MapLength {
            got: reported,
            experts,
            ..
        } = bind_error(outcome)
        else {
            panic!("expected MapLength for {got} entries");
        };
        assert_eq!((reported, experts), (got, EXPERTS));
        assert_eq!(landed, 1, "a refused map writes nothing");
    }
}

/// A fill whose rows end one past the pool is `RowRange`; ending exactly at the pool's last row is
/// accepted.
#[test]
fn a_row_fill_ending_one_past_the_pool_is_refused() {
    let (accepted, _) = step_with(fill(0, POOL - 1, vec![0]));
    accepted.expect("the pool's last row is a valid fill");

    let (outcome, landed) = step_with(fill(0, POOL, vec![0]));
    let BindError::RowRange {
        first_row,
        end,
        pool_rows,
        ..
    } = bind_error(outcome)
    else {
        panic!("expected RowRange");
    };
    assert_eq!((first_row, end, pool_rows), (POOL, POOL + 1, POOL));
    assert_eq!(landed, 1, "a refused fill writes nothing");
}

/// A table index the site does not have is `ExpertRange` naming the table count, not an out-of-bounds
/// panic.
#[test]
fn a_table_the_site_does_not_have_is_refused() {
    let (outcome, landed) = step_with(fill(usize::MAX, 0, vec![0]));
    let BindError::ExpertRange {
        expert, experts, ..
    } = bind_error(outcome)
    else {
        panic!("expected ExpertRange");
    };
    assert_eq!(expert, usize::MAX);
    assert!(experts >= 1, "the site's table count is reported");
    assert_eq!(landed, 1);
}

/// A refused sync leaves the executable needing a complete reset: the next step is refused, not run
/// over tables the refused fill may have half described.
#[test]
fn a_refused_sync_needs_a_complete_reset() {
    let (mut engine, exe, entry) = pooled_engine();
    let bytes: Vec<u8> = (0..WIDTH)
        .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
        .collect();
    let step = |engine: &mut Engine<Recorder>, writes: SyncWrites| {
        let mut inputs = StepInputs::new();
        inputs.push(
            poot_graph_ir::SlotKey::new(Slot::Activation, Some("x")),
            &[1, 1, WIDTH],
            poot_tensor::HostView::new(DType::F32, WIDTH, &bytes).unwrap(),
        );
        engine
            .step(exe, entry, &inputs, &mut FixedHost(writes))
            .map(|_| ())
    };
    assert!(matches!(
        step(&mut engine, fill(0, 0, vec![EXPERTS])),
        Err(ExecError::Bind(_))
    ));
    assert!(matches!(
        step(&mut engine, SyncWrites::default()),
        Err(ExecError::NeedsReset(_))
    ));
}
