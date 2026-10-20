//! Placement rows (POOT-1103): one driver over several devices, on fake devices that execute.
//!
//! The fake device keeps real bytes in real buffers and runs every dispatch through the kernel-IR
//! reference interpreter, so a token comes out of arithmetic: the pipeline's and the tensor placement's
//! tokens equal the single-device driver's only when every boundary value and every collective carried
//! the right numbers. The model is a small stack of gated MLP blocks whose weights carry the roles the
//! Megatron rule cuts, with values on a grid that makes every sum exact, so the split sums equal the
//! whole ones bit for bit.
//!
//! The interpreter needs each dispatch's kernel body, which the executor does not hand a device. The
//! driver's own seam supplies it: [`DriverOptions::charge`] sees every program the driver compiles
//! before it is added, and a test's charge function records its bodies by kernel key.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::rc::Rc;
use std::sync::Arc;

use poot_executor::{
    Collectives, Coordinator, Device, DeviceTime, Dispatch, Engine, NativeCounters, SupportBound,
};
use poot_graph_ir::{BinOp, Graph};
use poot_graph_plan::{
    CompileOptions, ShardRule, Target, multi_device::topology::DeviceId, staged::StagedProgram,
    staged::TargetSet,
};
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::{Body, Ty};
use poot_models::components::attention::{AttentionParams, AttentionWeights, attention};
use poot_models::components::linear::{Weight, linear};
use poot_models::components::standard::{Step, decoder_program};
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec, StateSpec};
use poot_models::model::{
    ConfigReason, FamilyKey, LogitRows, Model, ModelConfig, ModelError, ModelOutput, ShapeReason,
    StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig, Registry};
use poot_quant::StoredBytes;
use poot_quant::weights::{
    AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightKey, WeightMap, WeightRole,
    WeightStore, WeightView,
};
use poot_runtime_common::{
    AllocGuard, BufferRole, CompiledKernel, MemoryCounterSnapshot, MemoryCounters,
};
use poot_target::{Backend, BufferStorage, DeviceCaps};
use poot_tensor::DType;
use poot_text::{ChatFormat, HermesToolCodec};

use super::capacity::{AllocationOwner, Footprint, Planner};
use super::error::{DriverError, InvalidOptions, Unsupported};
use super::placement::Placement;
use super::tests::tiny_text;
use super::{
    AdapterRef, Admission, DraftPolicy, Driver, DriverOptions, EncodeShapes, Feature,
    GenerateRequest, Generation, GenerationId, Head, Incompatible, Layout, ModelHandle, PoolShape,
    PreparedSetLimits, Retention, RowWork, SeqRequest, Serving, ServingShapes, State, StepRow,
    Warm,
};
use crate::GenerationControl;
use crate::core::sampler::Sampler;

const MLP: FamilyKey = FamilyKey::new("mlp2");
const VOCAB: usize = 12;
const HIDDEN: usize = 4;
const INTER: usize = 4;
const LAYERS: usize = 2;
const EOS: u32 = 11;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn gate(layer: usize) -> WeightId {
    WeightId::layer(layer, WeightRole::Ffn(FfnRole::Gate))
}

fn up(layer: usize) -> WeightId {
    WeightId::layer(layer, WeightRole::Ffn(FfnRole::Up))
}

fn down(layer: usize) -> WeightId {
    WeightId::layer(layer, WeightRole::Ffn(FfnRole::Down))
}

fn head() -> WeightId {
    WeightId::model(WeightRole::Head)
}

/// `LAYERS` gated MLP blocks over a seed row built from the token and its position, then a head.
#[derive(Debug)]
struct Mlp {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
}

impl Model for Mlp {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        if shape.capacity.get() > self.config.max_positions {
            return Err(TraceError::ShapeUnsupported {
                family: MLP,
                shape,
                reason: ShapeReason::CapacityAboveMax {
                    max: self.config.max_positions,
                },
            });
        }
        let weight = |id, shape: &[usize]| {
            Weight::new(&self.weights, id, shape).expect("the fixture maps every weight it reads")
        };
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
        let flat = rows * out;
        let mut x = b.broadcast(b.reshape(seed, vec![flat, 1])?, vec![flat, HIDDEN])?;
        for layer in 0..LAYERS {
            let _scope = b.layer_scope(layer);
            let g = linear(&b, x, &weight(gate(layer), &[INTER, HIDDEN]), None)?;
            let u = linear(&b, x, &weight(up(layer), &[INTER, HIDDEN]), None)?;
            let h = b.binary(BinOp::Mul, g, u)?;
            let d = linear(&b, h, &weight(down(layer), &[HIDDEN, INTER]), None)?;
            x = b.binary(BinOp::Add, x, d)?;
        }
        let logits = linear(&b, x, &weight(head(), &[VOCAB, HIDDEN]), None)?;
        let logits = b.reshape(logits, vec![rows, out, VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

fn build_mlp(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        unreachable!("the fixture family names only HF configs")
    };
    let max_positions =
        config
            .get("max_positions")
            .and_then(|v| v.as_u64())
            .ok_or(ModelError::Config {
                family: MLP,
                field: "max_positions",
                reason: ConfigReason::Missing,
            })? as usize;
    let mut map = WeightMap::builder(store);
    for layer in 0..LAYERS {
        for (id, key) in [
            (gate(layer), format!("gate{layer}")),
            (up(layer), format!("up{layer}")),
            (down(layer), format!("down{layer}")),
        ] {
            map.map(id, WeightView::Stored(WeightKey::from(key.as_str())))
                .unwrap();
        }
    }
    map.map(head(), WeightView::Stored(WeightKey::from("head")))
        .unwrap();
    let schema = ModelSchema::builder()
        .step(
            ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: VOCAB })
                .carrying(StateSpec::hand_off()),
        )
        .build_for(MLP)?;
    Ok(ModelParts::new(Box::new(Mlp {
        schema,
        config: ModelConfig {
            family: MLP,
            vocab: VOCAB,
            max_positions,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        weights: map.build(),
    })))
}

/// Values on a quarter grid, so every product and sum of the fixture is exact in f32. `mul` and `salt`
/// pick the pattern; the ones below make the greedy decode wander over several tokens.
fn grid(rows: usize, cols: usize, mul: usize, salt: usize) -> WeightEntry {
    let values: Vec<u8> = (0..rows * cols)
        .flat_map(|i| {
            let quarter = ((i * mul + salt) % 7) as f32 - 3.0;
            (quarter * 0.25).to_le_bytes()
        })
        .collect();
    WeightEntry::Dense(
        DenseWeight::try_new(DType::F32, vec![rows, cols], StoredBytes::from(values)).unwrap(),
    )
}

fn store() -> WeightStore {
    // Per layer: gate, up and down patterns; then the head's.
    const GATE: [(usize, usize); LAYERS] = [(6, 6), (1, 4)];
    const UP: [(usize, usize); LAYERS] = [(4, 4), (4, 3)];
    const DOWN: [(usize, usize); LAYERS] = [(5, 0), (4, 2)];
    let mut store = WeightStore::builder();
    for layer in 0..LAYERS {
        let (mul, salt) = GATE[layer];
        store
            .insert(format!("gate{layer}"), grid(INTER, HIDDEN, mul, salt))
            .unwrap();
        let (mul, salt) = UP[layer];
        store
            .insert(format!("up{layer}"), grid(INTER, HIDDEN, mul, salt))
            .unwrap();
        let (mul, salt) = DOWN[layer];
        store
            .insert(format!("down{layer}"), grid(HIDDEN, INTER, mul, salt))
            .unwrap();
    }
    store.insert("head", grid(VOCAB, HIDDEN, 2, 3)).unwrap();
    store.build()
}

fn handle() -> Arc<ModelHandle> {
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: MLP,
            keys: &[(ConfigSource::HfModelType, "mlp2")],
            build: build_mlp,
            fixture: || Fixture {
                config: serde_json::json!({"model_type": "mlp2", "max_positions": 32}),
                generation: None,
                preprocessor: None,
                store: WeightStore::default(),
            },
            variants: &[],
        })
        .unwrap();
    let config = serde_json::json!({"model_type": "mlp2", "max_positions": 32});
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, store(), &registry, |_| Ok(tiny_text()))
            .expect("the fixture model loads"),
    )
}

// ---------------------------------------------------------------------------------------------
// An attention model, for the paged KV rows (POOT-1104).
// ---------------------------------------------------------------------------------------------

const ATTN: FamilyKey = FamilyKey::new("attn1");
const HEADS: usize = 2;
const HEAD_DIM: usize = 2;
/// Pool blocks of the paged rows: `BLOCK_SIZE` slots each.
const POOL_BLOCKS: usize = 2;

fn attn_weight(role: AttnRole) -> WeightId {
    WeightId::layer(0, WeightRole::Attn(role))
}

/// One attention block (no rotary embedding: the step's mask carries position) over the same seed
/// row as [`Mlp`], then a head. `HEADS` KV heads of `HEAD_DIM`, so two ranks hold one head each.
#[derive(Debug)]
struct Attn {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
}

impl Model for Attn {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        if shape.capacity.get() > self.config.max_positions {
            return Err(TraceError::ShapeUnsupported {
                family: ATTN,
                shape,
                reason: ShapeReason::CapacityAboveMax {
                    max: self.config.max_positions,
                },
            });
        }
        let step = Step::new(&b, shape)?;
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let first = if shape.logits == LogitRows::All {
            0
        } else {
            n - 1
        };
        let out = n - first;
        let params = AttentionParams::nope(HEADS, HEADS, HEAD_DIM, false)
            .expect("the fixture's attention parameters are valid");
        let weights = AttentionWeights::new(&self.weights, 0, HIDDEN, &params)
            .expect("the fixture maps every attention weight");
        let token = b.cast(
            b.guard_index_bounds(step.tokens(), VOCAB)
                .expect("the step's ids are a rank-2 I32 tensor")
                .guarded,
            DType::F32,
        )?;
        let pos = b.cast(step.pos(), DType::F32)?;
        let seed = b.binary(BinOp::Add, token, pos)?;
        let seed = b.broadcast(b.reshape(seed, vec![rows, n, 1])?, vec![rows, n, HIDDEN])?;
        // A ramp over the hidden columns: with equal columns every projection is one direction scaled
        // by the seed, and the decode would be one token.
        let ramp = b.broadcast(b.iota(HIDDEN)?, vec![rows, n, HIDDEN])?;
        let x = b.binary(BinOp::Add, seed, ramp)?;
        let x = {
            let _scope = b.layer_scope(0);
            let attended = attention(&b, &step, 0, x, &weights, &params)?;
            b.binary(BinOp::Add, x, attended)?
        };
        let x = b.slice(x, 1, first, n)?;
        let head = Weight::new(&self.weights, head(), &[VOCAB, HIDDEN])
            .expect("the fixture maps the head");
        let logits = b.reshape(linear(&b, x, &head, None)?, vec![rows, out, VOCAB])?;
        Ok(step.finish(b, logits)?)
    }
}

fn build_attn(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        unreachable!("the fixture family names only HF configs")
    };
    let max_positions =
        config
            .get("max_positions")
            .and_then(|v| v.as_u64())
            .ok_or(ModelError::Config {
                family: ATTN,
                field: "max_positions",
                reason: ConfigReason::Missing,
            })? as usize;
    let mut map = WeightMap::builder(store);
    for (id, key) in [
        (attn_weight(AttnRole::Q), "q"),
        (attn_weight(AttnRole::K), "k"),
        (attn_weight(AttnRole::V), "v"),
        (attn_weight(AttnRole::O), "o"),
        (head(), "head"),
    ] {
        map.map(id, WeightView::Stored(WeightKey::from(key)))
            .unwrap();
    }
    let schema = ModelSchema::builder()
        .step(decoder_program(VOCAB))
        .build_for(ATTN)?;
    Ok(ModelParts::new(Box::new(Attn {
        schema,
        config: ModelConfig {
            family: ATTN,
            vocab: VOCAB,
            max_positions,
            eos: BTreeSet::from([EOS]),
            bos: None,
            prompt_bos: None,
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        weights: map.build(),
    })))
}

/// The patterns were picked so the greedy decode wanders over several tokens.
fn attn_store() -> WeightStore {
    let mut store = WeightStore::builder();
    for (key, mul, salt) in [("q", 5, 3), ("k", 5, 4), ("v", 4, 4), ("o", 4, 6)] {
        store
            .insert(key.to_string(), grid(HIDDEN, HIDDEN, mul, salt))
            .unwrap();
    }
    store.insert("head", grid(VOCAB, HIDDEN, 6, 3)).unwrap();
    store.build()
}

fn attn_handle() -> Arc<ModelHandle> {
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: ATTN,
            keys: &[(ConfigSource::HfModelType, "attn1")],
            build: build_attn,
            fixture: || Fixture {
                config: serde_json::json!({"model_type": "attn1", "max_positions": 32}),
                generation: None,
                preprocessor: None,
                store: WeightStore::default(),
            },
            variants: &[],
        })
        .unwrap();
    let config = serde_json::json!({"model_type": "attn1", "max_positions": 32});
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    Arc::new(
        ModelHandle::from_checkpoint(&raw, attn_store(), &registry, |_| Ok(tiny_text()))
            .expect("the attention fixture loads"),
    )
}

// ---------------------------------------------------------------------------------------------
// The kernel bodies the fake device interprets.
// ---------------------------------------------------------------------------------------------

thread_local! {
    static BODIES: RefCell<HashMap<String, Body>> = RefCell::new(HashMap::new());
}

/// [`DriverOptions::charge`] for these rows: remember every dispatch's body by its kernel key, then
/// charge as the production measure does.
fn record_bodies(program: &StagedProgram) -> Retention {
    for (_, _, stage) in program.stages() {
        for (_, plan) in stage.planned() {
            if let poot_graph_plan::Plan::Dispatches(dispatches) = plan {
                for dispatch in dispatches.iter() {
                    BODIES.with(|bodies| {
                        bodies
                            .borrow_mut()
                            .entry(dispatch.key().to_string())
                            .or_insert_with(|| dispatch.body().clone());
                    });
                }
            }
        }
    }
    super::program_retention(program)
}

// ---------------------------------------------------------------------------------------------
// The device.
// ---------------------------------------------------------------------------------------------

/// What every device of one test shares, and the per-device tallies a row reads back.
#[derive(Default)]
struct World {
    /// Recordings replayed, by device index.
    replays: RefCell<Vec<u64>>,
    /// Dispatches executed, by device index.
    dispatches: RefCell<Vec<u64>>,
    /// Peer copies executed, by the sending device's index.
    peer_copies: RefCell<Vec<u64>>,
}

impl World {
    fn bump(counter: &RefCell<Vec<u64>>, index: usize) {
        let mut counter = counter.borrow_mut();
        if counter.len() <= index {
            counter.resize(index + 1, 0);
        }
        counter[index] += 1;
    }

    fn tally(counter: &RefCell<Vec<u64>>, devices: usize) -> Vec<u64> {
        let mut counter = counter.borrow().clone();
        counter.resize(devices, 0);
        counter
    }
}

struct Alloc {
    /// The native element capacity the buffer was allocated with.
    elems: usize,
    data: RefCell<Vec<u8>>,
    _guard: AllocGuard,
}

type Buf = Rc<Alloc>;

enum Recorded {
    Dispatch {
        body: Body,
        /// Each bound buffer and its logical element count, in parameter order, output last.
        args: Vec<(Buf, usize)>,
        workgroups: [u32; 3],
    },
    Copy(Buf, Buf),
    PeerCopy {
        src: Buf,
        dst: Buf,
        bytes: usize,
    },
}

struct Interp {
    index: usize,
    world: Rc<World>,
    memory: MemoryCounters,
    open: Option<Vec<Recorded>>,
    free_bytes: u64,
}

fn interp(world: &Rc<World>, index: usize) -> Interp {
    Interp {
        index,
        world: Rc::clone(world),
        memory: MemoryCounters::new(),
        open: None,
        free_bytes: 1 << 30,
    }
}

fn target() -> Target {
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: DeviceCaps {
            memory_budget_bytes: Some(1 << 30),
            ..DeviceCaps::ptx_default()
        },
    }
}

#[derive(Debug, thiserror::Error)]
#[error("interpreter device: {0}")]
struct InterpFault(String);

fn fault(reason: impl std::fmt::Display) -> InterpFault {
    InterpFault(reason.to_string())
}

/// The element type of kernel parameter `index` (zero-based).
fn element_of(body: &Body, index: usize) -> Result<Ty, InterpFault> {
    let param = body
        .params()
        .nth(index)
        .ok_or_else(|| fault(format!("kernel {} has no parameter {index}", body.name)))?;
    match body.local_ty(param) {
        Ty::Ref { pointee, .. } => match &**pointee {
            Ty::Slice(elem) => Ok((**elem).clone()),
            other => Err(fault(format!("parameter {index} points at {other:?}"))),
        },
        other => Err(fault(format!("parameter {index} is {other:?}"))),
    }
}

fn execute(body: &Body, args: &[(Buf, usize)], workgroups: [u32; 3]) -> Result<(), InterpFault> {
    let mut buffers = Vec::with_capacity(args.len());
    for (index, (buf, elems)) in args.iter().enumerate() {
        let data = buf.data.borrow();
        let (words, _) = data[..elems * 4].as_chunks::<4>();
        let buffer = match element_of(body, index)? {
            Ty::F32 => Buffer::from_f32s(
                &words
                    .iter()
                    .map(|w| f32::from_le_bytes(*w))
                    .collect::<Vec<_>>(),
            ),
            Ty::I32 => Buffer::from_i32s(
                &words
                    .iter()
                    .map(|w| i32::from_le_bytes(*w))
                    .collect::<Vec<_>>(),
            ),
            Ty::U32 => Buffer::from_u32s(
                &words
                    .iter()
                    .map(|w| u32::from_le_bytes(*w))
                    .collect::<Vec<_>>(),
            ),
            other => return Err(fault(format!("kernel {} binds {other:?}", body.name))),
        };
        buffers.push(buffer);
    }
    run(body, workgroups, &mut buffers).map_err(|e| fault(format!("kernel {}: {e}", body.name)))?;
    for (index, ((buf, elems), buffer)) in args.iter().zip(&buffers).enumerate() {
        let mut data = buf.data.borrow_mut();
        let bytes: Vec<u8> = match element_of(body, index)? {
            Ty::F32 => buffer
                .to_f32s()
                .map_err(fault)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Ty::I32 => buffer
                .to_i32s()
                .map_err(fault)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => buffer
                .to_u32s()
                .map_err(fault)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        data[..elems * 4].copy_from_slice(&bytes);
    }
    Ok(())
}

impl Device for Interp {
    type Buffer = Buf;
    type Kernel = Body;
    type Recording = Rc<Vec<Recorded>>;
    type Error = InterpFault;

    fn target(&self) -> Target {
        target()
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        Some(self.free_bytes)
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Buf, InterpFault> {
        let bytes = elems * storage.element().byte_width();
        Ok(Rc::new(Alloc {
            elems,
            data: RefCell::new(vec![0; bytes]),
            _guard: self.memory.record_alloc(role, bytes as u64),
        }))
    }

    fn buffer_elems(&self, buffer: &Buf) -> usize {
        buffer.elems
    }

    fn load_kernel(&mut self, key: &str, _kernel: CompiledKernel) -> Result<Body, InterpFault> {
        BODIES
            .with(|bodies| bodies.borrow().get(key).cloned())
            .ok_or_else(|| fault(format!("no recorded body for kernel {key}")))
    }

    fn begin(&mut self) -> Result<(), InterpFault> {
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), InterpFault> {
        let args = d
            .inputs
            .iter()
            .map(|arg| (Rc::clone(arg.buffer), arg.elems as usize))
            .chain([(Rc::clone(d.output.buffer), d.output.elems as usize)])
            .collect();
        let poot_executor::LaunchKind::Grid { threads, workgroup } = d.launch else {
            panic!("the interpreter device runs thread grids only");
        };
        let workgroups = [0, 1, 2].map(|axis| threads[axis].div_ceil(workgroup[axis].max(1)));
        self.open
            .as_mut()
            .expect("a dispatch inside a recording")
            .push(Recorded::Dispatch {
                body: d.kernel.clone(),
                args,
                workgroups,
            });
        Ok(())
    }

    fn copy(&mut self, src: &Buf, dst: &Buf) -> Result<(), InterpFault> {
        self.open
            .as_mut()
            .expect("a copy inside a recording")
            .push(Recorded::Copy(Rc::clone(src), Rc::clone(dst)));
        Ok(())
    }

    fn finish(&mut self) -> Result<Rc<Vec<Recorded>>, InterpFault> {
        Ok(Rc::new(
            self.open.take().expect("finish closes a recording"),
        ))
    }

    fn replay(&mut self, recording: &Rc<Vec<Recorded>>) -> Result<(), InterpFault> {
        World::bump(&self.world.replays, self.index);
        for op in recording.iter() {
            match op {
                Recorded::Dispatch {
                    body,
                    args,
                    workgroups,
                } => {
                    execute(body, args, *workgroups)?;
                    World::bump(&self.world.dispatches, self.index);
                }
                Recorded::Copy(src, dst) => {
                    let bytes = src.data.borrow().clone();
                    dst.data.borrow_mut()[..bytes.len()].copy_from_slice(&bytes);
                }
                Recorded::PeerCopy { src, dst, bytes } => {
                    let moved = src.data.borrow()[..*bytes].to_vec();
                    dst.data.borrow_mut()[..*bytes].copy_from_slice(&moved);
                    World::bump(&self.world.peer_copies, self.index);
                }
            }
        }
        Ok(())
    }

    fn write_with(
        &mut self,
        dst: &Buf,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), InterpFault> {
        let mut data = dst.data.borrow_mut();
        fill(offset..offset + len, &mut data[offset..offset + len]);
        Ok(())
    }

    fn read(&mut self, src: &Buf, out: &mut [u8]) -> Result<(), InterpFault> {
        out.copy_from_slice(&src.data.borrow()[..out.len()]);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), InterpFault> {
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), InterpFault> {
        self.open = None;
        Ok(())
    }
}

impl Collectives for Interp {
    type Event = ();

    fn native_counters(&self) -> NativeCounters {
        NativeCounters::default()
    }

    fn support_bound(&self) -> SupportBound {
        SupportBound::default()
    }

    fn open_route(&mut self, _peer: &Self) -> Result<(), InterpFault> {
        Ok(())
    }

    fn create_event(&mut self) -> Result<(), InterpFault> {
        Ok(())
    }

    fn record_event(&mut self, _event: &()) -> Result<(), InterpFault> {
        Ok(())
    }

    fn wait_event(&mut self, _event: &()) -> Result<(), InterpFault> {
        Ok(())
    }

    fn peer_copy(&mut self, src: &Buf, dst: &Buf, bytes: usize) -> Result<(), InterpFault> {
        self.open
            .as_mut()
            .expect("a peer copy inside a recording")
            .push(Recorded::PeerCopy {
                src: Rc::clone(src),
                dst: Rc::clone(dst),
                bytes,
            });
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Rigs.
// ---------------------------------------------------------------------------------------------

fn options() -> DriverOptions {
    let compile = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(16),
        prefill_chunk: nz(4),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(8)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
        },
        charge: record_bodies,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

fn coordinator(world: &Rc<World>) -> Coordinator<Interp> {
    Coordinator::new(vec![
        (DeviceId(0), interp(world, 0)),
        (DeviceId(1), interp(world, 1)),
    ])
    .unwrap()
}

fn single(handle: &Arc<ModelHandle>) -> (Driver, Rc<World>) {
    let world = Rc::new(World::default());
    let device = Engine::new(interp(&world, 0));
    let driver = Driver::new(Arc::clone(handle), Box::new(device), options())
        .expect("the single-device driver opens");
    (driver, world)
}

fn placed(
    handle: &Arc<ModelHandle>,
    placement: Placement,
) -> Result<(Driver, Rc<World>), DriverError> {
    let world = Rc::new(World::default());
    let driver = Driver::placed(
        Arc::clone(handle),
        coordinator(&world),
        options(),
        placement,
    )?;
    Ok((driver, world))
}

fn pipeline() -> Placement {
    Placement::Pipeline {
        after_layers: vec![0],
    }
}

fn tensor() -> Placement {
    Placement::Tensor {
        rule: ShardRule::Megatron,
    }
}

fn generate(driver: &mut Driver<Serving>) -> Generation {
    driver
        .generate(
            GenerateRequest {
                prompt: vec![1, 2, 3],
                max_new: 8,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .expect("the request generates")
}

fn weight_bytes<S: State>(driver: &Driver<S>) -> u64 {
    driver
        .live_role_bytes()
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Weight)
        .map_or(0, |(_, bytes)| bytes)
}

// ---------------------------------------------------------------------------------------------
// Rows.
// ---------------------------------------------------------------------------------------------

/// The fixture decodes something worth comparing: several distinct tokens, not one repeated.
#[test]
fn the_single_device_reference_decodes_a_varied_sequence() {
    let (driver, _) = single(&handle());
    let mut driver = driver.serve(Layout::Contiguous);
    let tokens = generate(&mut driver).tokens;
    // The same model evaluated by hand in f64: a constant sequence would let a dropped boundary or a
    // missing reduction go unseen.
    assert_eq!(tokens, [1, 1, 1, 1, 3, 5, 0, 6]);
}

/// A two-stage pipeline over two devices decodes the single-device tokens, with both devices running
/// their stage and the boundary crossing by peer copies.
///
/// Mutation: build the partition as `Single(devices[0])` whatever the placement (`Placed::partition`):
/// device 1 never replays and the row goes red on the replay tally.
#[test]
fn a_two_device_pipeline_decodes_the_single_device_tokens() {
    let handle = handle();
    let (reference, _) = single(&handle);
    let mut reference = reference.serve(Layout::Contiguous);
    let want = generate(&mut reference).tokens;

    let (driver, world) = placed(&handle, pipeline()).expect("the pipeline opens");
    let mut driver = driver.serve(Layout::Contiguous);
    let got = generate(&mut driver).tokens;
    assert_eq!(got, want, "pipeline tokens against the single device's");
    assert_eq!(driver.placement(), &pipeline());

    let replays = World::tally(&world.replays, 2);
    assert!(
        replays.iter().all(|&n| n > 0),
        "every device replayed its stage: {replays:?}"
    );
    let copies = World::tally(&world.peer_copies, 2);
    assert!(
        copies[0] > 0 && copies[1] == 0,
        "the boundary flows from stage 0 to stage 1: {copies:?}"
    );
}

/// A tensor placement over two devices decodes the single-device tokens: each rank runs every layer
/// over its share of the weights, and the all-reduce after each row-parallel projection joins them.
///
/// Mutation: load the whole weight map on every rank (`load_weights` instead of `load_tensor_weights`):
/// the sharded programs bind weights of the wrong shape and the driver refuses to open its entries, so
/// the row goes red.
#[test]
fn a_tensor_placement_decodes_the_single_device_tokens() {
    let handle = handle();
    let (reference, _) = single(&handle);
    let mut reference = reference.serve(Layout::Contiguous);
    let want = generate(&mut reference).tokens;
    let whole = weight_bytes(&reference);

    let (driver, world) = placed(&handle, tensor()).expect("the tensor placement opens");
    let mut driver = driver.serve(Layout::Contiguous);
    let got = generate(&mut driver).tokens;
    assert_eq!(
        got, want,
        "tensor-parallel tokens against the single device's"
    );

    let replays = World::tally(&world.replays, 2);
    assert!(
        replays.iter().all(|&n| n > 0),
        "every rank replayed: {replays:?}"
    );
    let copies = World::tally(&world.peer_copies, 2);
    assert!(
        copies.iter().all(|&n| n > 0),
        "both ranks exchange partials: {copies:?}"
    );
    // Each rank reads its own share: the cut weights are held once between the ranks.
    assert_eq!(
        weight_bytes(&driver),
        whole,
        "the two ranks together hold the model once"
    );
}

/// A placement the executor's devices cannot honor is refused before any weight loads.
#[test]
fn a_placement_that_disagrees_with_the_devices_is_a_typed_refusal() {
    let handle = handle();
    let refused = |placement| match placed(&handle, placement) {
        Ok(_) => panic!("the placement opened"),
        Err(error) => error,
    };
    assert!(matches!(
        refused(Placement::Single),
        DriverError::InvalidOptions(InvalidOptions::Devices { devices: 2 })
    ));
    assert!(matches!(
        refused(Placement::Pipeline {
            after_layers: vec![0, 1]
        }),
        DriverError::InvalidOptions(InvalidOptions::PipelineStages {
            stages: 3,
            devices: 2
        })
    ));
}

/// A driver opened over `placement` with `draft`: the refusal `Driver::placed` returns for a draft
/// policy the placement has no form of.
pub(in crate::driver) fn witness_draft_placement(
    draft: DraftPolicy,
    placement: Placement,
) -> DriverError {
    let world = Rc::new(World::default());
    let mut opts = options();
    opts.draft = draft;
    let Err(error) = Driver::placed(Arc::clone(&handle()), coordinator(&world), opts, placement)
    else {
        panic!("a draft policy opened over a placement");
    };
    error
}

/// A draft model's layers are not the target's, so a placement that cuts or divides layers refuses it.
pub(in crate::driver) fn witness_draft_model_placement(placed_as: Feature) -> DriverError {
    let handle = handle();
    witness_draft_placement(
        DraftPolicy::DraftModel {
            handle: Arc::clone(&handle),
            k: nz(2),
        },
        placement_of(placed_as),
    )
}

/// The MTP layer's carry and cache sit on one device past the target's stages.
pub(in crate::driver) fn witness_mtp_placement(placed_as: Feature) -> DriverError {
    witness_draft_placement(DraftPolicy::Mtp { k: nz(2) }, placement_of(placed_as))
}

/// The placement a placement feature names.
fn placement_of(feature: Feature) -> Placement {
    match feature {
        Feature::PipelinePlacement => pipeline(),
        Feature::TensorPlacement => tensor(),
        other => panic!("{other} is not a placement"),
    }
}

/// A tensor placement loads one executable whose weights are cut per rank, so `prepare` of a request
/// under an adapter generation is refused before the registry is asked whether the generation exists.
pub(in crate::driver) fn witness_tensor_adapters() -> DriverError {
    let (driver, _) = placed(&handle(), tensor()).expect("the tensor placement opens");
    driver
        .prepare(&ServingShapes {
            adapters: &[GenerationId::unissued(1)],
            ..serving()
        })
        .expect_err("an adapter generation under a tensor placement is refused")
        .error
}

/// A tensor placement loads one executable, so an encode step beside it is refused.
pub(in crate::driver) fn witness_tensor_encode() -> DriverError {
    let (mut driver, _) = placed(&handle(), tensor()).expect("the tensor placement opens");
    driver
        .prepare_encode(&EncodeShapes {
            rows: nz(1),
            max_len: nz(2),
        })
        .expect_err("an encode step under a tensor placement is refused")
}

/// A draft model's layers are not the target's, so a placement that cuts or divides layers refuses it.
#[test]
fn a_draft_model_has_no_placed_form() {
    for placed_as in [Feature::PipelinePlacement, Feature::TensorPlacement] {
        let error = witness_draft_model_placement(placed_as);
        assert!(
            matches!(
                error,
                DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                    first: Feature::DraftModel,
                    second,
                    ..
                })) if second == placed_as
            ),
            "got {error}"
        );
    }
}

/// The MTP layer has no placed form either, and a tensor placement serves no adapter and no encode
/// step beside its one executable. Mutation: drop the placement arms of `config_features`; each opens.
#[test]
fn an_mtp_layer_an_adapter_and_an_encode_step_have_no_tensor_or_pipeline_form() {
    for placed_as in [Feature::PipelinePlacement, Feature::TensorPlacement] {
        let error = witness_mtp_placement(placed_as);
        assert!(
            matches!(
                error,
                DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                    first: Feature::Mtp,
                    second,
                    ..
                })) if second == placed_as
            ),
            "got {error}"
        );
    }
    let error = witness_tensor_adapters();
    assert!(
        matches!(
            error,
            DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                first: Feature::Adapters,
                second: Feature::TensorPlacement,
                ..
            }))
        ),
        "got {error}"
    );
    let error = witness_tensor_encode();
    assert!(
        matches!(
            error,
            DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                first: Feature::Encode,
                second: Feature::TensorPlacement,
                ..
            }))
        ),
        "got {error}"
    );
}

fn serving() -> ServingShapes<'static> {
    static ROWS: [NonZeroUsize; 1] = [NonZeroUsize::MIN];
    ServingShapes {
        layout: Layout::Contiguous,
        rows: &ROWS,
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

fn per_device(footprint: &Footprint, role: BufferRole) -> [u64; 2] {
    [DeviceId(0), DeviceId(1)].map(|device| {
        footprint
            .role_bytes(device)
            .unwrap()
            .into_iter()
            .find(|&(r, _)| r == role)
            .map_or(0, |(_, bytes)| bytes)
    })
}

/// The planner keeps one footprint per device: a pipeline's weights split by stage, a tensor
/// placement's by share, each device owning only the allocations its own stages make.
///
/// Mutation: record every allocation on the first device (`PlanningDevice::id`): device 1 owns no
/// weight bytes and the row goes red.
#[test]
fn the_planner_keeps_a_footprint_per_device() {
    let handle = handle();
    let targets = TargetSet::new(vec![(DeviceId(0), target()), (DeviceId(1), target())]).unwrap();
    let shapes = serving();
    let reference = {
        let one = TargetSet::single(DeviceId(0), target());
        Planner::new(Arc::clone(&handle), &one, Placement::Single)
            .unwrap()
            .plan(options(), &shapes, &[])
            .unwrap()
    };
    let whole = per_device(&reference.footprint, BufferRole::Weight)[0];
    let head_bytes = (VOCAB * HIDDEN * 4) as u64;
    assert!(whole > head_bytes);

    for (placement, expected) in [
        // Layer 0 on device 0, layer 1 and the head on device 1.
        (pipeline(), None),
        // Half of every cut weight on each rank; the head, which the final stage reads, on one.
        (tensor(), Some(head_bytes)),
    ] {
        let planned = Planner::new(Arc::clone(&handle), &targets, placement.clone())
            .unwrap()
            .plan(options(), &shapes, &[])
            .unwrap();
        let weights = per_device(&planned.footprint, BufferRole::Weight);
        assert!(
            weights.iter().all(|&bytes| bytes > 0),
            "{placement:?}: every device owns weights: {weights:?}"
        );
        assert_eq!(
            weights.iter().sum::<u64>(),
            whole,
            "{placement:?}: the devices hold the model once: {weights:?}"
        );
        if let Some(head) = expected {
            assert_eq!(
                weights[0].abs_diff(weights[1]),
                head,
                "{placement:?}: the ranks hold equal shares, and the head sits on one: {weights:?}"
            );
        }
    }
}

/// Each device's planned role bytes equal that device's own live counters once the same placed driver
/// prepares the same shapes: the plan is the production preparation, rank by rank.
///
/// Mutation: report the executor's total under every device (`Coordinator::device_memory` without the
/// per-engine counters): both devices read the sum and this row goes red.
#[test]
fn each_devices_planned_role_bytes_equal_its_own_prepared_counters() {
    let handle = handle();
    let targets = TargetSet::new(vec![(DeviceId(0), target()), (DeviceId(1), target())]).unwrap();
    let shapes = serving();
    for placement in [pipeline(), tensor()] {
        let planned = Planner::new(Arc::clone(&handle), &targets, placement.clone())
            .unwrap()
            .plan(options(), &shapes, &[])
            .unwrap();
        let (driver, _) = placed(&handle, placement.clone()).unwrap();
        let driver = driver.prepare(&shapes).expect("the shapes prepare");
        let live = driver.live_device_bytes();
        assert_eq!(live.len(), 2, "{placement:?}: one entry per device");
        for role in [BufferRole::Weight, BufferRole::State] {
            let planned = per_device(&planned.footprint, role);
            for (index, (device, roles)) in live.iter().enumerate() {
                let live = roles
                    .iter()
                    .find(|&&(r, _)| r == role)
                    .map_or(0, |&(_, bytes)| bytes);
                assert_eq!(planned[index], live, "{placement:?} {role:?} on {device:?}");
            }
        }
    }
}

/// A footprint the planner composes names its device on every owner: no allocation is anonymous.
#[test]
fn every_planned_owner_names_a_device_of_the_placement() {
    let handle = handle();
    let targets = TargetSet::new(vec![(DeviceId(0), target()), (DeviceId(1), target())]).unwrap();
    let planned = Planner::new(Arc::clone(&handle), &targets, tensor())
        .unwrap()
        .plan(options(), &serving(), &[])
        .unwrap();
    let owners: Vec<&AllocationOwner> = planned.footprint.owners().collect();
    assert!(!owners.is_empty());
    assert!(
        owners
            .iter()
            .all(|owner| owner.device == DeviceId(0) || owner.device == DeviceId(1))
    );
    assert_eq!(
        planned.footprint.devices(),
        vec![DeviceId(0), DeviceId(1)],
        "both ranks own allocations"
    );
}

/// The memory fit over a placement admits each device against its own budget: with rank 0 roomy and rank 1
/// a byte short of its share, every candidate is refused naming rank 1 for a tensor placement as for a
/// pipeline; with both ranks sized to their own plans the first candidate is chosen, so the refusal was
/// the one rank's.
///
/// Mutation: admit a footprint against the summed budgets (`Budget::admit`): the short rank is hidden by
/// the roomy one and the refusal row goes red.
#[test]
fn the_fit_refuses_a_short_rank_of_either_placement_naming_it() {
    use super::capacity::{Budget, CapacityRefusal, DeviceBudget, FitError, Rejection, fit};

    let handle = handle();
    let targets = TargetSet::new(vec![(DeviceId(0), target()), (DeviceId(1), target())]).unwrap();
    let shapes = serving();
    for placement in [pipeline(), tensor()] {
        let planner = Planner::new(Arc::clone(&handle), &targets, placement.clone()).unwrap();
        let plan = |_: &u8| {
            planner
                .plan(options(), &shapes, &[])
                .map_err(super::capacity::PlanFailure::Fatal)
        };
        let reference = plan(&0).unwrap();
        let need = |device| reference.footprint.device_bytes(device).unwrap();
        let known = |bytes| DeviceBudget::Known { bytes, reserve: 0 };
        let budget = |first: u64, second: u64| {
            Budget::new()
                .with(DeviceId(0), known(first))
                .with(DeviceId(1), known(second))
        };

        let short = budget(u64::MAX / 2, need(DeviceId(1)) - 1);
        let refused = fit(&[0u8], plan, || short.clone());
        let Err(FitError::NoCandidateFits { rejected }) = refused else {
            panic!("{placement:?}: the short rank was admitted");
        };
        match &rejected[0].why {
            Rejection::Capacity(CapacityRefusal::OverBudget { device, .. }) => {
                assert_eq!(*device, DeviceId(1), "{placement:?}");
            }
            other => panic!("{placement:?}: got {other:?}"),
        }

        let exact = budget(need(DeviceId(0)), need(DeviceId(1)));
        let chosen = fit(&[0u8], plan, || exact.clone()).expect("each rank fits its own plan");
        assert_eq!(chosen.chosen, 0, "{placement:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Paged KV rows (POOT-1104).
// ---------------------------------------------------------------------------------------------

fn paged_serving() -> ServingShapes<'static> {
    static ROWS: [NonZeroUsize; 1] = [NonZeroUsize::MIN];
    ServingShapes {
        layout: Layout::Paged(PoolShape {
            blocks: NonZeroUsize::new(POOL_BLOCKS).unwrap(),
            max_seqs: NonZeroUsize::MIN,
        }),
        rows: &ROWS,
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

/// Greedy decode of `max_new` tokens from a three-token prompt through the paged serving surface: one
/// sequence on the shared pool, a block table, the slot maps bound each step.
fn paged_tokens(driver: Driver, max_new: usize) -> Vec<u32> {
    let mut driver = driver
        .prepare(&paged_serving())
        .expect("the paged shapes prepare");
    let prompt = [1u32, 2, 3];
    let Admission::Opened { seq, .. } = driver
        .open(SeqRequest::new(&prompt, max_new, AdapterRef::Base))
        .expect("the sequence opens")
    else {
        panic!("the pool has room for one sequence")
    };
    let mut sampler = Sampler::greedy();
    let mut work = RowWork::Prefill { upto: prompt.len() };
    let mut tokens = Vec::new();
    while tokens.len() < max_new {
        let mut rows = [StepRow {
            seq,
            work,
            sampler: &mut sampler,
        }];
        let result = driver.step(&mut rows).expect("the step runs").remove(0);
        let Some(&token) = result.tokens.first() else {
            continue;
        };
        driver
            .commit(seq, 1, &mut sampler)
            .expect("the pick commits");
        tokens.push(token);
        work = RowWork::Decode { ahead: &[] };
    }
    tokens
}

/// Bytes of the paged KV pools one rank holds: K and V of one layer, `pool_slots` rows of `kv_heads`
/// heads of `HEAD_DIM` f32 values, the heads divided over `world` ranks.
fn pool_bytes(world: usize) -> u64 {
    let slots = POOL_BLOCKS * super::block_table::BLOCK_SIZE;
    (2 * slots * (HEADS / world) * HEAD_DIM * 4) as u64
}

/// A paged tensor-parallel decode over two executing devices equals the single-device driver's tokens:
/// each rank scatters its own KV heads into its own pool and gathers them back by the same slot maps,
/// and the attention output joins in the row-parallel O projection.
///
/// Mutation: rank 1 scatters rank 0's K rows (`expand_ranks` wiring rank 1's `ScatterUpdate` source
/// to rank 0's copy): the program is not a valid rank program and the driver refuses to prepare it
/// (`EquationOperandNotDefined`), so the row goes red.
#[test]
fn a_paged_tensor_placement_decodes_the_single_device_tokens() {
    let handle = attn_handle();
    let (reference, _) = single(&handle);
    let want = paged_tokens(reference, 8);
    // Several distinct tokens, not one repeated: a wrong head in a pool or a lost scatter would
    // otherwise decode the same constant.
    assert_eq!(want, [4, 6, 1, 4, 1, 6, 1, 1]);

    let (driver, world) = placed(&handle, tensor()).expect("the tensor placement opens");
    let got = paged_tokens(driver, 8);
    assert_eq!(
        got, want,
        "paged tensor-parallel tokens against the single device's"
    );
    let copies = World::tally(&world.peer_copies, 2);
    assert!(
        copies[1] > 0,
        "rank 1 sends its partial of the one reduction to the rank that finishes: {copies:?}"
    );
    let replays = World::tally(&world.replays, 2);
    assert!(
        replays.iter().all(|&n| n > 0),
        "every rank replayed: {replays:?}"
    );
}

/// The KV pool is divided by heads: the plan's per-rank State bytes are the rank's heads of the pool,
/// the two ranks together hold the single device's pool once, and each rank's prepared counters equal
/// its planned bytes.
///
/// Mutation: charge State allocations twice in `PlanningDevice::allocate`: the single device plans
/// 2048 bytes where the pool is 1024 and the row goes red. (Leaving the carried pool undivided in
/// `Sharder::bind_input` is refused earlier, as a `StateTypeMismatch` of the rank program.)
#[test]
fn each_ranks_paged_kv_footprint_is_its_heads_of_the_pool_and_equals_the_prepared_bytes() {
    let handle = attn_handle();
    let targets = TargetSet::new(vec![(DeviceId(0), target()), (DeviceId(1), target())]).unwrap();
    let shapes = paged_serving();

    let one = TargetSet::single(DeviceId(0), target());
    let reference = Planner::new(Arc::clone(&handle), &one, Placement::Single)
        .unwrap()
        .plan(options(), &shapes, &[])
        .unwrap();
    let whole = per_device(&reference.footprint, BufferRole::State)[0];
    assert_eq!(
        whole,
        pool_bytes(1),
        "the single device holds the whole pool"
    );

    let planned = Planner::new(Arc::clone(&handle), &targets, tensor())
        .unwrap()
        .plan(options(), &shapes, &[])
        .unwrap();
    let state = per_device(&planned.footprint, BufferRole::State);
    assert_eq!(
        state,
        [pool_bytes(2), pool_bytes(2)],
        "each rank plans its own heads of the pool"
    );
    assert_eq!(
        state.iter().sum::<u64>(),
        whole,
        "the ranks hold the pool once"
    );

    let (driver, _) = placed(&handle, tensor()).unwrap();
    let driver = driver.prepare(&shapes).expect("the paged shapes prepare");
    for (index, (device, roles)) in driver.live_device_bytes().iter().enumerate() {
        let live = roles
            .iter()
            .find(|&&(role, _)| role == BufferRole::State)
            .map_or(0, |&(_, bytes)| bytes);
        assert_eq!(state[index], live, "State on {device:?}");
    }
}
