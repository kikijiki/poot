//! POOT-587 SC-004/SC-005: the planned footprint of a candidate equals the live role bytes the same
//! preparation leaves on a counting device. The family is registered from outside the crate and
//! carries a state whose size depends on the candidate in a different way per variant: a dense cache
//! that grows with capacity (contiguous) or with the pool (paged), a window-clamped cache that stops
//! growing past its window, and a recurrent state that does not depend on capacity at all. A planner
//! that extrapolated from two samples would be wrong on the clamped and the flat variant.

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_executor::{Device, Dispatch, Engine};
use poot_graph_ir::{BinOp, Graph, StateRole, TensorType};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, multi_device::topology::DeviceId,
    production_plan::Submission,
};
use poot_llm::driver::capacity::Planner;
use poot_llm::driver::placement::Placement;
use poot_llm::driver::{
    Driver, DriverOptions, Head, Layout, ModelHandle, PoolShape, PreparedSetLimits, ServingShapes,
    Warm,
};
use poot_models::components::standard::Step;
use poot_models::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput,
    Phase, ShapeReason, StepShape, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::{BufferRole, MemoryCounters};
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;
use poot_text::ChatFormat;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const VOCAB: usize = 12;
const EOS: u32 = 11;
const FAMILY: FamilyKey = FamilyKey::new("capacity-tiny");
/// The window of the clamped variant, in positions.
const WINDOW: usize = 8;
/// Bytes of one cached position: four f32 lanes.
const POSITION_BYTES: u64 = 16;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Dense,
    Windowed,
    Recurrent,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Windowed => "windowed",
            Self::Recurrent => "recurrent",
        }
    }
}

#[derive(Debug)]
struct Tiny {
    config: ModelConfig,
    weights: WeightMap,
    variant: Variant,
}

impl Model for Tiny {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(
        &self,
        phase: Phase,
        shape: StepShape,
        b: poot_graph_ir::Builder,
    ) -> Result<Graph, TraceError> {
        let refuse = |reason| {
            Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason,
            })
        };
        if phase == Phase::Decode && shape.tokens.get() != 1 {
            return refuse(ShapeReason::DecodeTokens);
        }
        let step = Step::new(&b, shape)?;
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let first = if shape.logits == LogitRows::All {
            0
        } else {
            n - 1
        };
        let out = n - first;
        let token = b.cast(
            b.slice(
                b.guard_index_bounds(step.tokens(), VOCAB)
                    .expect("the step's ids are a rank-2 I32 tensor")
                    .guarded,
                1,
                first,
                n,
            )?,
            DType::F32,
        )?;
        let pos = b.cast(b.slice(step.pos(), 1, first, n)?, DType::F32)?;
        let seed = b.binary(BinOp::Add, token, pos)?;
        let (dims, role) = match (self.variant, shape.kv) {
            (Variant::Dense, KvLayout::Paged { pool_slots, .. }) => {
                (vec![pool_slots.get(), 4], StateRole::Positional { axis: 0 })
            }
            (Variant::Dense, _) => (
                vec![rows, shape.capacity.get(), 4],
                StateRole::Positional { axis: 1 },
            ),
            (Variant::Windowed, _) => (
                vec![rows, shape.capacity.get().min(WINDOW), 4],
                StateRole::Positional { axis: 1 },
            ),
            (Variant::Recurrent, _) => (vec![rows, 4], StateRole::Recurrent),
        };
        let state = b.state_input("toy.state", TensorType::f32(dims.clone()), role)?;
        // The state folds the last position's value; one row, so the folded scalar reshapes to ones.
        let last = b.binary(
            BinOp::Add,
            b.cast(
                b.slice(
                    b.guard_index_bounds(step.tokens(), VOCAB)
                        .expect("the step's ids are a rank-2 I32 tensor")
                        .guarded,
                    1,
                    n - 1,
                    n,
                )?,
                DType::F32,
            )?,
            b.cast(b.slice(step.pos(), 1, n - 1, n)?, DType::F32)?,
        )?;
        let spread = b.broadcast(b.reshape(last, vec![1; dims.len()])?, dims)?;
        step.carry(state, b.binary(BinOp::Add, state, spread)?);
        let logits = b.broadcast(b.reshape(seed, vec![rows, out, 1])?, vec![rows, out, VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

fn build_tiny(raw: &RawConfig<'_>) -> Result<Box<dyn Model>, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        unreachable!("the family names only HF configs")
    };
    let variant = match config.get("state").and_then(|v| v.as_str()) {
        Some("dense") => Variant::Dense,
        Some("windowed") => Variant::Windowed,
        Some("recurrent") => Variant::Recurrent,
        _ => {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "state",
                reason: ConfigReason::Unsupported,
            });
        }
    };
    Ok(Box::new(Tiny {
        variant,
        config: ModelConfig {
            family: FAMILY,
            vocab: VOCAB,
            max_positions: 256,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            output: ModelOutput::Logits { vocab: VOCAB },
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
        },
        weights: WeightMap::default(),
    }))
}

/// The family's tokenizer, a word-level vocabulary over `a`..`k`.
fn tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0u32);
    for (i, w) in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        .iter()
        .enumerate()
    {
        vocab.insert((*w).to_string(), (i + 1) as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

/// The family's checkpoint directory: `config.json`, a tokenizer and an empty safetensors file.
fn checkpoint(variant: Variant) -> (poot_test_util::UniqueTempPath, ModelHandle) {
    let dir = poot_test_util::unique_temp_path("capacity_checkpoint");
    std::fs::create_dir_all(&dir).unwrap();
    let config = serde_json::json!({
        "model_type": "capacity-tiny",
        "state": variant.name(),
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    let header = b"{}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: FAMILY,
            keys: &[(ConfigSource::HfModelType, "capacity-tiny")],
            build: |raw, _store| build_tiny(raw),
            fixture: || Fixture {
                config: serde_json::json!({"model_type": "capacity-tiny", "state": "dense"}),
                generation: None,
                store: WeightStore::default(),
            },
            variants: &[],
        })
        .unwrap();
    let handle =
        ModelHandle::load(&dir, &registry, &poot_llm::driver::LoadOptions::default()).unwrap();
    (dir, handle)
}

// ---------------------------------------------------------------------------------------------
// A counting device: an independent implementation of the allocation counters, whose buffers stay
// live for the device's life like a real one's, so the driver's live role bytes are what it holds.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

struct CountingDevice {
    memory: MemoryCounters,
}

impl Device for CountingDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

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
    ) -> Result<(), FakeError> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        Ok(())
    }
    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn begin(&mut self, _submission: Submission) -> Result<(), FakeError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Option<()>, FakeError> {
        Ok(Some(()))
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), _out: &mut [u8]) -> Result<(), FakeError> {
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
}

fn options(capacity: usize) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(4),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
        },
        charge: poot_llm::driver::program_retention,
        draft: poot_llm::driver::DraftPolicy::Off,
        hidden_window: None,
    }
}

fn shapes(layout: Layout) -> ServingShapes<'static> {
    ServingShapes {
        layout,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY, Head::Logits],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

/// Bytes per role, in the order the counters report them.
type RoleBytes = Vec<(BufferRole, u64)>;

/// A variant's state-size law: bytes of state at a capacity.
type SizeLaw = fn(usize) -> u64;

/// What the planner says a candidate allocates and what a counting device holds after the same
/// preparation, per role.
fn planned_and_live(
    handle: &Arc<ModelHandle>,
    capacity: usize,
    layout: Layout,
) -> (RoleBytes, RoleBytes) {
    let device = CountingDevice {
        memory: MemoryCounters::new(),
    };
    let executor = Engine::new(device);
    let planner = {
        use poot_executor::Executor;
        Planner::new(
            Arc::clone(handle),
            &executor.target_set(),
            Placement::Single,
        )
        .unwrap()
    };
    let planned = planner
        .plan(options(capacity), &shapes(layout), false)
        .unwrap();
    assert!(
        planned.residue.is_empty(),
        "a planning device releases everything it recorded"
    );
    let mut driver =
        Driver::new(Arc::clone(handle), Box::new(executor), options(capacity)).unwrap();
    driver.prepare(&shapes(layout)).unwrap();
    (
        planned.footprint.role_bytes(DeviceId(0)).unwrap(),
        driver.live_role_bytes(),
    )
}

fn state_bytes(roles: &[(BufferRole, u64)]) -> u64 {
    roles
        .iter()
        .find(|(role, _)| *role == BufferRole::State)
        .map_or(0, |&(_, bytes)| bytes)
}

/// SC-004: for each state shape and capacity, every role's planned bytes equal the live bytes the same
/// preparation leaves, and the state bytes equal an independent oracle of the variant's own size law:
/// linear in capacity, clamped at the window, flat. Capacities 4, 8 and 64 straddle the window, so a
/// plan extrapolated from the first two points is wrong on the third.
#[test]
fn planned_role_bytes_equal_live_role_bytes_for_each_state_shape() {
    let cases: [(Variant, SizeLaw); 3] = [
        (Variant::Dense, |capacity| capacity as u64 * POSITION_BYTES),
        (Variant::Windowed, |capacity| {
            capacity.min(WINDOW) as u64 * POSITION_BYTES
        }),
        (Variant::Recurrent, |_| POSITION_BYTES),
    ];
    for (variant, oracle) in cases {
        let (_dir, handle) = checkpoint(variant);
        let handle = Arc::new(handle);
        for capacity in [4, 8, 64] {
            let (planned, live) = planned_and_live(&handle, capacity, Layout::Contiguous);
            let live: Vec<_> = live
                .into_iter()
                .filter(|&(role, _)| role != BufferRole::Staging)
                .collect();
            assert_eq!(
                planned, live,
                "{variant:?} at capacity {capacity}: planned vs live role bytes"
            );
            assert_eq!(
                state_bytes(&planned),
                oracle(capacity),
                "{variant:?} at capacity {capacity}: state bytes vs the variant's size law"
            );
        }
    }
}

/// SC-004: a paged pool is sized by its blocks, not by capacity, and the plan agrees with the live bytes.
#[test]
fn a_paged_pool_plans_by_its_blocks() {
    let (_dir, handle) = checkpoint(Variant::Dense);
    let handle = Arc::new(handle);
    for blocks in [2usize, 5] {
        let layout = Layout::Paged(PoolShape {
            blocks: nz(blocks),
            max_seqs: NonZeroUsize::MIN,
        });
        let (planned, live) = planned_and_live(&handle, 16, layout);
        let live: Vec<_> = live
            .into_iter()
            .filter(|&(role, _)| role != BufferRole::Staging)
            .collect();
        assert_eq!(planned, live, "{blocks} blocks: planned vs live role bytes");
        assert_eq!(
            state_bytes(&planned),
            (blocks * poot_llm::driver::block_table::BLOCK_SIZE) as u64 * POSITION_BYTES,
            "{blocks} blocks: the pool's slots times a position"
        );
    }
}
