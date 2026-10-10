//! POOT-749 on the CPU oracle: a tensor-parallel `StagedProgram`, walked rank by rank with each
//! rank's own weight shares, equals the whole graph (ADR-0101 tier 2: the ranks sum partial products
//! in a different order), and the sharding pass refuses what it cannot split by name.
//!
//! The walk binds every stage input the way an executor does: a const by the rank's view of its
//! mapped weight ([`WeightMap::rank_view`], the production cut), a slot from the step, a carried value
//! from its rank's slice of the whole cache, and a boundary from the stage that exported it. Nothing
//! here reads the sharding pass's own bookkeeping to decide what a rank holds.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor_parity::dense::{Dense, Family, paged_step, plain, step};
use poot_executor_parity::weight_map::{MappedModel, TIER2, assert_close};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::{
    ComputedConst, Graph, Slot, SlotKey, Storage, TensorType, ValidationChannel, ValueId,
};
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, LoraPlan, Partition,
    ShardError, ShardRule, StagedCompileError, StagedProgram, Submission, Target, TargetSet,
    compile_staged, shard,
};
use poot_models::model::{LogitRows, Phase};
use poot_quant::format::WeightFormat;
use poot_quant::weights::{
    AttnRole, DenseWeight, LoraRole, LoraTarget, MlaRole, WeightEntry, WeightId, WeightMap,
    WeightRole, WeightStore, WeightView,
};
use poot_quant::{PackedPayload, PackedWeight, SourceRole};
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_caps::default_caps_for;

const WORLD: u32 = 2;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn target(backend: Backend) -> Target {
    let mut caps = default_caps_for(backend);
    caps.memory_budget_bytes = Some(1 << 30);
    Target { backend, caps }
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

fn devices() -> Vec<DeviceId> {
    ranks(WORLD)
}

/// Devices `0..world`.
fn ranks(world: u32) -> Vec<DeviceId> {
    (0..world).map(DeviceId).collect()
}

fn tensor_partition(weights: Arc<WeightMap>) -> Partition {
    tensor_partition_over(weights, WORLD)
}

fn tensor_partition_over(weights: Arc<WeightMap>, world: u32) -> Partition {
    Partition {
        experts: ExpertPlacement::AllResident,
        devices: DevicePlacement::Tensor {
            rule: ShardRule::Megatron,
            devices: ranks(world),
            weights,
        },
    }
}

fn two_gpus(backend: Backend) -> TargetSet {
    gpus(backend, WORLD)
}

/// `world` identical devices of `backend`: the fake topology the lowering is measured on.
fn gpus(backend: Backend, world: u32) -> TargetSet {
    TargetSet::new(
        ranks(world)
            .into_iter()
            .map(|d| (d, target(backend)))
            .collect(),
    )
    .unwrap()
}

fn tiny_qwen2() -> MappedModel {
    Dense::new(Family::Qwen2).f32_model()
}

fn decode_graph(mapped: &MappedModel, rows: usize) -> Graph {
    plain(
        mapped
            .model
            .trace(Phase::Decode, step(rows, 1, 6, LogitRows::Last))
            .expect("the fixture traces a decode step"),
    )
}

/// One paged decode step of `rows` rows over `capacity` positions and a pool of `pool_slots` slots.
fn paged_decode_graph(mapped: &MappedModel, rows: usize, pool_slots: usize) -> Graph {
    plain(
        mapped
            .model
            .trace(
                Phase::Decode,
                paged_step(rows, 1, CAPACITY, pool_slots, LogitRows::Last),
            )
            .expect("the fixture traces a paged decode step"),
    )
}

/// The capacity of the paged steps below: positions each row reads.
const CAPACITY: usize = 6;

/// Where position `t` of row `r` lives in a pool of `pool` slots: scattered, so a read that ignored
/// the map would read the wrong slot.
fn pool_slot(row: usize, t: usize, pool: usize) -> usize {
    (5 * t + 3 * row + 2) % pool
}

/// What names an input binder: stage graphs renumber their values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Identity {
    Named(String),
    Slot(SlotKey),
    Computed(ComputedConst),
}

fn identity<V: ValidationChannel>(graph: &Graph<V>, value: ValueId) -> Identity {
    let meta = graph.meta(value);
    match meta.storage {
        Storage::Const | Storage::State => Identity::Named(meta.name.clone().expect("named")),
        Storage::Slot(_) => Identity::Slot(meta.slot_key().cloned().expect("a keyed slot")),
        Storage::Computed(payload) => Identity::Computed(payload),
        Storage::Device => unreachable!("an input binder is never a device value"),
    }
}

fn host_f32(entry: &WeightEntry) -> HostTensor {
    let WeightEntry::Dense(dense) = entry else {
        panic!("the fixture's weights are dense");
    };
    assert_eq!(dense.dtype(), DType::F32);
    let values = dense
        .bytes()
        .as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    HostTensor::f32(dense.shape().to_vec(), values)
}

/// Every const a graph reads, as `map` binds it.
fn weight_tensors(map: &WeightMap, store: &WeightStore) -> HashMap<String, HostTensor> {
    map.iter()
        .map(|(id, _, _)| {
            (
                id.const_name(),
                host_f32(&map.materialize(id, store).expect("the view materializes")),
            )
        })
        .collect()
}

/// Rank `rank`'s equal slice of `tensor` along the one axis where `local` is narrower.
fn slice_to(tensor: &HostTensor, local: &[usize], rank: usize) -> HostTensor {
    let shape = tensor.shape().to_vec();
    if shape == local {
        return tensor.clone();
    }
    let axes: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] != local[a]).collect();
    assert_eq!(axes.len(), 1, "one axis is split: {shape:?} -> {local:?}");
    let axis = axes[0];
    assert_eq!(shape[axis] % local[axis], 0, "{shape:?} -> {local:?}");
    assert!(rank < shape[axis] / local[axis], "rank {rank} of {shape:?}");
    let data = tensor.as_f32().expect("f32 state");
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = Vec::with_capacity(local.iter().product());
    for o in 0..outer {
        let base = (o * shape[axis] + rank * local[axis]) * inner;
        out.extend_from_slice(&data[base..base + local[axis] * inner]);
    }
    HostTensor::f32(local.to_vec(), out)
}

/// The step's slots by key and the carried cache by name, for a graph with `rows` rows.
struct Step {
    slots: HashMap<SlotKey, HostTensor>,
    state: HashMap<String, HostTensor>,
}

fn decode_step<V: ValidationChannel>(graph: &Graph<V>) -> Step {
    let mut slots = HashMap::new();
    let mut state = HashMap::new();
    // A paged step's pool, and its rows, from the inputs that declare them.
    let slot_shape = |slot: Slot, rank: usize| {
        graph.inputs.iter().find_map(|&input| {
            let meta = graph.meta(input);
            (meta.storage == Storage::Slot(slot) && meta.aval.shape.len() == rank)
                .then(|| meta.aval.shape.clone())
        })
    };
    let pool_slots = slot_shape(Slot::SlotMap, 1).map_or(1, |shape| shape[0]);
    let rows = slot_shape(Slot::Token, 2).map_or(1, |shape| shape[0]);
    for &input in &graph.inputs {
        let meta = graph.meta(input);
        let shape = meta.aval.shape.clone();
        let numel = shape.iter().product::<usize>().max(1);
        match meta.storage {
            Storage::Slot(Slot::Token) => {
                let ids = (0..numel).map(|row| 5 + 3 * row as i32).collect();
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::i32(shape, ids),
                );
            }
            Storage::Slot(Slot::Activation) => {
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::f32(shape, fill(numel, 31)),
                );
            }
            Storage::Slot(Slot::Pos) => {
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::i32(shape, vec![3; numel]),
                );
            }
            // The paged slot maps: a row's logical position `t` reads pool slot `pool_slot(row, t)`;
            // the step's one token per row (position 3) writes the slot its position maps to.
            Storage::Slot(Slot::SlotMap) if shape.len() == 2 => {
                let cap = shape[1];
                let pool = pool_slots;
                let map = (0..rows)
                    .flat_map(|r| (0..cap).map(move |t| pool_slot(r, t, pool) as i32))
                    .collect();
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::i32(shape, map),
                );
            }
            Storage::Slot(Slot::SlotMap) => {
                let pool = shape[0];
                let mut map = vec![-1i32; pool];
                for row in 0..rows {
                    map[pool_slot(row, 3, pool)] = row as i32;
                }
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::i32(shape, map),
                );
            }
            Storage::Slot(Slot::LoraIdx) => {
                // Alternating rows: adapter 1, no adapter, adapter 2, ...
                let idx = (0..numel).map(|row| ((row + 1) % 3) as f32).collect();
                slots.insert(
                    meta.slot_key().cloned().unwrap(),
                    HostTensor::f32(shape, idx),
                );
            }
            Storage::State => {
                let seed = 7000 + state.len() as u64;
                state.insert(
                    meta.name.clone().unwrap(),
                    HostTensor::f32(shape, fill(numel, seed)),
                );
            }
            _ => {}
        }
    }
    Step { slots, state }
}

/// A stage graph's input binders, bound.
struct Walked {
    output: HostTensor,
    /// The carried values each rank wrote, by the whole graph's state name.
    state: HashMap<(String, usize), HostTensor>,
}

fn base_name(name: &str) -> (&str, usize) {
    match name.rsplit_once(".tp") {
        Some((base, rank)) if rank.chars().all(|c| c.is_ascii_digit()) => {
            (base, rank.parse().unwrap())
        }
        _ => (name, 0),
    }
}

/// The program's stages in order, each bound as its rank binds it.
fn walk<V: ValidationChannel>(
    program: &StagedProgram<V>,
    weights: (&WeightStore, &WeightMap),
    step: &Step,
    extra: &HashMap<String, HostTensor>,
) -> Walked {
    let (store, map) = weights;
    let tensor = program.tensor().expect("a tensor-parallel program");
    let world = NonZeroU32::new(tensor.devices.len() as u32).unwrap();
    let ranks: Vec<HashMap<String, HostTensor>> = (0..world.get() as usize)
        .map(|rank| {
            let view = map
                .rank_view(store, rank, world, &tensor.shares)
                .expect("every share cuts");
            let mut tensors = weight_tensors(&view, store);
            for (name, whole) in extra {
                tensors.insert(name.clone(), whole.clone());
            }
            tensors
        })
        .collect();
    let mut boundaries: HashMap<SlotKey, Value> = HashMap::new();
    let mut state: HashMap<(String, usize), HostTensor> = HashMap::new();
    let mut output = None;
    let count = program.stages().count();
    for (index, (_, device, stage)) in program.stages().enumerate() {
        let rank = tensor
            .devices
            .iter()
            .position(|&d| d == device)
            .expect("a stage runs on a rank's device");
        let g = stage.graph();
        let inputs: HashMap<ValueId, Value> = g
            .inputs
            .iter()
            .map(|&id| {
                let tensor = match identity(g, id) {
                    Identity::Computed(payload) => Value::from(payload.tensor()),
                    Identity::Slot(key) => match boundaries.get(&key) {
                        Some(value) => value.clone(),
                        None => Value::from(
                            step.slots
                                .get(&key)
                                .unwrap_or_else(|| panic!("stage {index}: {key:?} is unbound"))
                                .clone(),
                        ),
                    },
                    Identity::Named(name) => {
                        if let Some(value) = ranks[rank].get(&name) {
                            Value::from(value.clone())
                        } else {
                            let (base, from) = base_name(&name);
                            assert_eq!(from, rank, "{name} belongs to rank {from}");
                            let whole = step
                                .state
                                .get(base)
                                .unwrap_or_else(|| panic!("stage {index}: {name} is unbound"));
                            Value::from(slice_to(whole, &g.aval(id).shape, rank))
                        }
                    }
                };
                (id, tensor)
            })
            .collect();
        let result = eval(
            g,
            &inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap_or_else(|error| panic!("stage {index} did not evaluate: {error}"));
        let environment = result.environment.expect("the environment was kept");
        for export in &g.exports {
            let value = environment[export.value]
                .clone()
                .unwrap_or_else(|| panic!("stage {index} lacks boundary {:?}", export.id));
            boundaries.insert(export.id.slot_key(), value);
        }
        for (&(state_in, _), value) in g.state.iter().zip(result.state) {
            let name = g.meta(state_in).name.clone().unwrap();
            let (base, from) = base_name(&name);
            state.insert(
                (base.to_string(), from),
                value.as_host().expect("dense state").clone(),
            );
        }
        if index + 1 == count {
            output = Some(result.output);
        }
    }
    Walked {
        output: output
            .expect("the final stage has the output")
            .as_host()
            .expect("dense")
            .clone(),
        state,
    }
}

/// The whole graph on the oracle, bound from the same weights and step.
fn whole<V: ValidationChannel>(
    graph: &Graph<V>,
    weights: (&WeightStore, &WeightMap),
    step: &Step,
    extra: &HashMap<String, HostTensor>,
) -> (HostTensor, HashMap<String, HostTensor>) {
    let mut tensors = weight_tensors(weights.1, weights.0);
    for (name, value) in extra {
        tensors.insert(name.clone(), value.clone());
    }
    let inputs: HashMap<ValueId, Value> = graph
        .inputs
        .iter()
        .map(|&id| {
            let value = match identity(graph, id) {
                Identity::Computed(payload) => payload.tensor(),
                Identity::Slot(key) => step.slots[&key].clone(),
                Identity::Named(name) => tensors
                    .get(&name)
                    .or_else(|| step.state.get(&name))
                    .unwrap_or_else(|| panic!("{name} is unbound"))
                    .clone(),
            };
            (id, Value::from(value))
        })
        .collect();
    let result = eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).expect("evaluates");
    let state = graph
        .state
        .iter()
        .zip(result.state)
        .map(|(&(state_in, _), value)| {
            (
                graph.meta(state_in).name.clone().unwrap(),
                value.as_host().unwrap().clone(),
            )
        })
        .collect();
    (result.output.as_host().unwrap().clone(), state)
}

fn compile<V: ValidationChannel>(
    graph: &Graph<V>,
    weights: &Arc<WeightMap>,
    options: &CompileOptions,
) -> Result<StagedProgram<V>, StagedCompileError> {
    compile_over(graph, weights, options, WORLD)
}

fn compile_over<V: ValidationChannel>(
    graph: &Graph<V>,
    weights: &Arc<WeightMap>,
    options: &CompileOptions,
    world: u32,
) -> Result<StagedProgram<V>, StagedCompileError> {
    compile_staged(
        graph,
        &gpus(Backend::Nvptx, world),
        &tensor_partition_over(Arc::clone(weights), world),
        options,
    )
}

fn assert_walk_matches<V: ValidationChannel>(
    program: &StagedProgram<V>,
    graph: &Graph<V>,
    weights: (&WeightStore, &WeightMap),
    step: &Step,
    extra: &HashMap<String, HostTensor>,
    label: &str,
) {
    let walked = walk(program, weights, step, extra);
    let (output, state) = whole(graph, weights, step, extra);
    assert_close(
        walked.output.as_f32().unwrap(),
        output.as_f32().unwrap(),
        TIER2,
        &format!("{label}: output"),
    );
    for (name, want) in &state {
        for rank in 0..program.tensor().unwrap().devices.len() {
            let got = walked
                .state
                .get(&(name.clone(), rank))
                .unwrap_or_else(|| panic!("{label}: rank {rank} wrote no {name}"));
            let want = slice_to(want, got.shape(), rank);
            assert_close(
                got.as_f32().unwrap(),
                want.as_f32().unwrap(),
                TIER2,
                &format!("{label}: rank {rank} {name}"),
            );
        }
    }
}

/// SC-001: the sharded qwen2 decode step, walked by rank (host collectives are the exchanged
/// partials the stage split carries across), equals the whole graph on the logits and on every
/// rank's slice of every carried KV pair.
///
/// Mutation: skip the `AllReduce` after layer 0's `o_proj` (`Sharder::all_reduce` returns the
/// partial); every later value of layer 0 is a rank's addend instead of the sum, and the walk goes red.
#[test]
fn a_sharded_qwen2_decode_step_walked_by_rank_equals_the_whole_graph() {
    let mapped = tiny_qwen2();
    let graph = decode_graph(&mapped, 1);
    let program = compile(&graph, &mapped.map, &options()).expect("qwen2 shards over two ranks");

    let tensor = program.tensor().unwrap();
    assert_eq!(tensor.devices, devices());
    for role in [AttnRole::Q, AttnRole::K, AttnRole::V, AttnRole::O] {
        let id = WeightId::layer(0, WeightRole::Attn(role));
        assert!(
            tensor.shares.iter().any(|&(share, _)| share == id),
            "{id} is cut"
        );
    }
    assert!(
        !program.transfers().is_empty(),
        "ranks exchange partial sums, or this row proves nothing"
    );
    let on_device = |device| program.stages().filter(|s| s.1 == device).count();
    assert!(on_device(DeviceId(0)) > 1 && on_device(DeviceId(1)) > 1);

    let step = decode_step(&graph);
    assert_walk_matches(
        &program,
        &graph,
        (&*mapped.store, &*mapped.map),
        &step,
        &HashMap::new(),
        "qwen2",
    );
}

// ---- refusals ------------------------------------------------------------------------------

fn dense_entry(shape: &[usize]) -> WeightEntry {
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = fill(n, 11).iter().flat_map(|v| v.to_le_bytes()).collect();
    WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape.to_vec(), bytes.into()).unwrap())
}

fn packed_entry(format: WeightFormat, shape: [usize; 2]) -> WeightEntry {
    let weight = PackedWeight::try_new(format, shape).unwrap();
    let bytes = vec![0u8; weight.source_bytes(SourceRole::Blocks)];
    WeightEntry::Packed(Arc::new(
        PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes.into())]).unwrap(),
    ))
}

/// `y = x @ w^T` with `w` the mapped weight `id`, spelled the way a tracer spells a dense linear.
fn dense_linear(id: WeightId, out: usize, input: usize) -> (Graph, WeightMap, WeightStore) {
    let mut store = WeightStore::builder();
    store.insert("w", dense_entry(&[out, input])).unwrap();
    let store = store.build();
    let mut map = WeightMap::builder(&store);
    map.map(id, WeightView::Stored("w".into())).unwrap();
    let map = map.build();
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![1, 1, input]));
    let w = b.constant(&id.const_name(), TensorType::f32(vec![out, input]));
    let wt = b.transpose(w, vec![1, 0]);
    let y = b.matmul(x, wt);
    (b.finish(y), map, store)
}

/// SC-002 (MLA): a weight of multi-head latent attention has no Megatron rule; `shard` names it.
///
/// Mutation: give `WeightRole::Mla(_)` the split of a dense projection in `weight_axis`; the graph
/// shards and this row goes red.
#[test]
fn shard_refuses_multi_head_latent_attention_by_name() {
    let id = WeightId::layer(0, WeightRole::Mla(MlaRole::QB));
    let (graph, map, _) = dense_linear(id, 8, 8);
    let refusal = shard(
        &graph,
        &map,
        ShardRule::Megatron,
        NonZeroU32::new(WORLD).unwrap(),
    )
    .map(|_| ())
    .unwrap_err();
    assert_eq!(
        refusal,
        ShardError::NoRule {
            weight: id,
            role: id.role,
            rule: ShardRule::Megatron,
        }
    );
}

/// SC-002 (encoding): a row-parallel Q4_K weight whose per-rank `K` is not a whole number of
/// 256-value super-blocks is `IllegalEncodingSplit`; one that is, shards.
///
/// Mutation: drop the block-alignment check in `Sharder::cut_weight`; the 768-wide weight shards
/// into 384-value halves and this row goes red.
#[test]
fn shard_refuses_a_q4_k_column_split_mid_block() {
    let id = WeightId::layer(0, WeightRole::Attn(AttnRole::O));
    // A column-parallel Q projection feeds the row-parallel O projection, so the activation O reads
    // is split exactly as its weight's columns are.
    let q = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    let build = |k: usize| {
        let mut store = WeightStore::builder();
        store.insert("q", dense_entry(&[k, 8])).unwrap();
        store
            .insert("o", packed_entry(WeightFormat::Q4_K, [4, k]))
            .unwrap();
        let store = store.build();
        let mut map = WeightMap::builder(&store);
        map.map(q, WeightView::Stored("q".into())).unwrap();
        map.map(id, WeightView::Stored("o".into())).unwrap();
        let map = map.build();
        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::f32(vec![1, 8]));
        let wq = b.constant(&q.const_name(), TensorType::f32(vec![k, 8]));
        let wq = b.transpose(wq, vec![1, 0]);
        let h = b.matmul(x, wq);
        let weight = PackedWeight::try_new(WeightFormat::Q4_K, [4, k]).unwrap();
        let y = poot_graph_ir::ops::packed_linear(&b, h, &id.const_name(), weight, None, None)
            .expect("a packed linear");
        (b.finish(y), map)
    };
    let world = NonZeroU32::new(WORLD).unwrap();

    let (graph, map) = build(768);
    let refusal = shard(&graph, &map, ShardRule::Megatron, world)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        refusal,
        ShardError::IllegalEncodingSplit {
            weight: id,
            format: WeightFormat::Q4_K,
            shape: [4, 768],
            axis: 1,
        }
    );

    let (graph, map) = build(1024);
    let sharded = shard(&graph, &map, ShardRule::Megatron, world).expect("512 = two super-blocks");
    assert_eq!(sharded.weights, vec![(q, 0), (id, 1)]);
    let local = sharded
        .graph
        .consts
        .iter()
        .map(|&c| sharded.graph.aval(c).shape.clone())
        .collect::<Vec<_>>();
    let weight = PackedWeight::try_new(WeightFormat::Q4_K, [4, 512]).unwrap();
    assert_eq!(
        local,
        vec![
            vec![512, 8],
            weight.source_shape(SourceRole::Blocks).to_vec()
        ]
    );
    assert_eq!(
        sharded.collectives.len(),
        1,
        "the row-parallel product is reduced"
    );
}

// ---- adapters ------------------------------------------------------------------------------

/// `mapped` with an adapter stack mapped beside its base weights: `A [stack, in, r]` and
/// `B [stack, r, out]` per layer and projection (nonzero, so every row's correction is visible) and
/// one scale vector, each named by its `WeightId` exactly as `apply_lora` names its consts.
fn with_adapters(mut mapped: MappedModel, layers: usize, stack: usize, rank: usize) -> MappedModel {
    let mut store = WeightStore::builder();
    store.extend_from(&mapped.store).unwrap();
    let mut adapters: Vec<(WeightId, String)> = Vec::new();
    let mut seed = 100u64;
    let mut add = |id: WeightId, shape: Vec<usize>, scale: f32, store: &mut _| {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = fill(n, seed)
            .iter()
            .flat_map(|v| (v * scale).to_le_bytes())
            .collect();
        seed += 1;
        let key = format!("adapter.{}", id.const_name());
        poot_quant::weights::WeightStoreBuilder::insert(
            store,
            key.clone(),
            WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape, bytes.into()).unwrap()),
        )
        .unwrap();
        adapters.push((id, key));
    };
    for layer in 0..layers {
        for target in LoraTarget::ALL {
            let base = WeightId::layer(layer, target.weight_role());
            let [out, input] = <[usize; 2]>::try_from(
                mapped
                    .map
                    .handle(base)
                    .expect("the fixture maps it")
                    .shape
                    .clone(),
            )
            .unwrap();
            add(
                WeightId::layer(layer, WeightRole::Lora(LoraRole::A(target))),
                vec![stack, input, rank],
                0.3,
                &mut store,
            );
            add(
                WeightId::layer(layer, WeightRole::Lora(LoraRole::B(target))),
                vec![stack, rank, out],
                0.3,
                &mut store,
            );
        }
    }
    add(
        WeightId::model(WeightRole::Lora(LoraRole::Scale)),
        vec![stack],
        1.0,
        &mut store,
    );
    let store = store.build();
    let mut map = mapped.map.extending(&store);
    for (id, key) in adapters {
        map.map(id, WeightView::Stored(key.as_str().into()))
            .unwrap();
    }
    mapped.map = Arc::new(map.build());
    mapped.store = Arc::new(store);
    mapped
}

/// SC-006: a sharded dense fixture with nonzero adapters on every projection (column-parallel Q/K/V,
/// gate/up and row-parallel O/down), three rows with alternating adapter ids (1, 2, none), matches
/// the full adapted graph on the logits and on every rank's slice of every carried value.
///
/// Mutations: (a) in `Sharder::indexed_matmul` read the row-parallel adapter's product as whole (no
/// reduction of the adapter's partial); (b) in `Sharder::binary` reduce only the base projection and
/// still call the sum partial (the base is applied once per rank). Each goes red.
#[test]
fn a_sharded_adapted_qwen2_with_alternating_rows_equals_the_full_adapted_graph() {
    let rows = 3;
    let mapped = with_adapters(tiny_qwen2(), 2, 3, 2);
    let graph = decode_graph(&mapped, rows);
    let lora = LoraPlan {
        rows: NonZeroUsize::new(rows).unwrap(),
        adapters: NonZeroUsize::new(2).unwrap(),
        rank: NonZeroUsize::new(2).unwrap(),
        projections: LoraTarget::ALL.to_vec(),
    };
    let with_lora = options().with_lora(lora);

    let program = compile(&graph, &mapped.map, &with_lora).expect("the adapted graph shards");
    let single = compile_staged(
        &graph,
        &two_gpus(Backend::Nvptx),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &with_lora,
    )
    .expect("the adapted graph compiles whole");
    let adapted = single.stages().next().unwrap().2.graph();

    let shares = &program.tensor().unwrap().shares;
    for target in LoraTarget::ALL {
        let a = WeightId::layer(0, WeightRole::Lora(LoraRole::A(target)));
        let b = WeightId::layer(0, WeightRole::Lora(LoraRole::B(target)));
        let row_parallel = matches!(target, LoraTarget::AttnO | LoraTarget::FfnDown);
        assert_eq!(
            shares.iter().any(|&(id, axis)| id == a && axis == 1),
            row_parallel,
            "{target:?}: A is cut on its input axis exactly for a row-parallel projection"
        );
        assert_eq!(
            shares.iter().any(|&(id, axis)| id == b && axis == 2),
            !row_parallel,
            "{target:?}: B is cut on its output axis exactly for a column-parallel projection"
        );
    }

    let step = decode_step(adapted);
    assert_walk_matches(
        &program,
        adapted,
        (&*mapped.store, &*mapped.map),
        &step,
        &HashMap::new(),
        "adapted qwen2",
    );

    // The adapters change the logits, or the row proves nothing.
    let plain_program = compile(&graph, &mapped.map, &options()).unwrap();
    let base_step = decode_step(&graph);
    let weights = (&*mapped.store, &*mapped.map);
    let with = walk(&program, weights, &step, &HashMap::new());
    let without = walk(&plain_program, weights, &base_step, &HashMap::new());
    assert_ne!(
        with.output.as_f32().unwrap(),
        without.output.as_f32().unwrap(),
        "the adapters change the logits"
    );
}

// ---- the collectives the rule can emit, and what lowers or refuses them -----------------------

/// A column-parallel product that is the graph's output is gathered: its `AllGather` lowers to a
/// concatenation of the two ranks' slices in rank order, and the walk equals the whole graph.
///
/// Mutation: concatenate in reverse rank order in `expand_ranks`; the halves swap and this row goes
/// red.
#[test]
fn a_split_output_is_gathered_and_the_ranks_slices_concatenate_in_rank_order() {
    let id = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    let (graph, map, store) = dense_linear(id, 8, 6);
    let world = NonZeroU32::new(WORLD).unwrap();
    let sharded =
        shard(&graph, &map, ShardRule::Megatron, world).expect("a column-parallel linear");
    assert_eq!(sharded.collectives.len(), 1);
    assert_eq!(
        sharded.graph.eqns.last().unwrap().op,
        poot_graph_ir::OpKind::AllGather { axis: 2, world },
        "the split output is gathered on its split axis"
    );

    let program = compile(&graph, &Arc::new(map.clone()), &options()).expect("it compiles");
    let step = decode_step(&graph);
    assert_walk_matches(
        &program,
        &graph,
        (&store, &map),
        &step,
        &HashMap::new(),
        "gathered linear",
    );
}

/// An op with no rule that reads a split value is a typed refusal that names the equation, the op
/// and the split axis.
///
/// Mutation: let `Slice` pass a split axis through unchanged; the slice cuts rank-local columns and
/// this row goes red (the refusal is missing).
#[test]
fn an_op_with_no_rule_on_a_split_axis_is_refused_by_name() {
    let id = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    let (_, map, _) = dense_linear(id, 8, 6);
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![1, 1, 6]));
    let w = b.constant(&id.const_name(), TensorType::f32(vec![8, 6]));
    let wt = b.transpose(w, vec![1, 0]);
    let y = b.matmul(x, wt);
    let cut = b.slice(y, 2, 0, 4);
    let graph = b.finish(cut);
    let refusal = shard(
        &graph,
        &map,
        ShardRule::Megatron,
        NonZeroU32::new(WORLD).unwrap(),
    )
    .map(|_| ())
    .unwrap_err();
    assert_eq!(
        refusal,
        ShardError::Unpropagatable {
            eqn: 2,
            op: "slice ax=2 0..4".to_string(),
            axis: 2
        }
    );
}

/// SC-VK: a tensor-parallel program on devices whose runtime has no cross-device collective is
/// refused by compile before any stage is compiled, naming the collective equation and the missing
/// capability.
///
/// Mutation: skip `devices::admit_collectives` in `compile_staged`; the program is planned anyway
/// and the refusal that remains names a point-to-point row of a different equation, so this row goes
/// red.
#[test]
fn a_collective_on_a_target_without_collectives_is_refused_by_compile() {
    let mapped = tiny_qwen2();
    let graph = decode_graph(&mapped, 1);
    let backend = Backend::SpirvVulkan;
    let error = compile_staged(
        &graph,
        &two_gpus(backend),
        &tensor_partition(Arc::clone(&mapped.map)),
        &options(),
    )
    .map(|_| ())
    .unwrap_err();
    let StagedCompileError::Refused(refusal) = error else {
        panic!("{backend:?}: expected a capability refusal, got {error}");
    };
    assert_eq!(
        refusal.missing,
        poot_graph_plan::Capability::Collective(poot_graph_plan::CollectiveKind::AllReduce),
        "{backend:?}"
    );
    assert!(
        matches!(refusal.op, poot_graph_ir::OpKind::AllReduce { .. }),
        "{backend:?}: the refusal names the collective equation, not {:?}",
        refusal.op
    );
    assert_eq!(refusal.target, backend);
}

/// POOT-1102 SC-001: a group of four ranks shards, and the walk equals the whole graph on the logits
/// and on every rank's slice of every carried KV pair. Each reduction sums four partials, so a hop
/// the lowering drops changes the logits.
///
/// Mutation: in `expand_ranks` fold only the first two ranks' partials of each `AllReduce` (leave
/// `for from in 1..world` as `1..world.min(2)`); every sum misses two addends and the walk goes red.
#[test]
fn a_group_of_four_ranks_walks_to_the_whole_graph() {
    let mapped = Dense::new(Family::Qwen2).heads(4, 4).f32_model();
    let graph = decode_graph(&mapped, 1);
    let program =
        compile_over(&graph, &mapped.map, &options(), 4).expect("qwen2 shards over four ranks");
    assert_eq!(program.tensor().unwrap().devices, ranks(4));
    let step = decode_step(&graph);
    assert_walk_matches(
        &program,
        &graph,
        (&*mapped.store, &*mapped.map),
        &step,
        &HashMap::new(),
        "qwen2 over four ranks",
    );
}

/// POOT-1102 SC-002: the cost of the N-rank lowering on two and four fake devices.
///
/// Every `AllReduce` is one segment, so the program has `world * reductions` stages: one boundary
/// per collective whatever the world, never a log-depth ladder. Every rank folds every peer's
/// partial, so a reduction moves `world * (world - 1)` vectors; only the last reduction is read by
/// one rank (the output's), which moves `world - 1`. Measured on this fixture (qwen2, 2 layers, 4
/// reductions, `[1, 64]` f32 partials): two ranks 8 stages / 7 copies / 1792 bytes, four ranks 16
/// stages / 39 copies / 9984 bytes. A recursive-doubling exchange was prototyped on the same
/// fixture and measured 30 stages / 27 copies / 6912 bytes at four ranks: it moves 31% fewer bytes
/// and costs 14 more stage boundaries, which is the wrong trade for a decode-sized vector. A fused
/// reduce row would not remove the boundary the collective is, only the `world * (world - 1)`
/// copy rows, so it is not planned.
///
/// Mutation: lower `AllReduce` as recursive doubling in `expand_ranks` (a segment per level, partner
/// `rank ^ (1 << level)`); the walk stays green and this row goes red with `left: Cost { stages: 30,
/// transfers: 27, bytes: 6912 }`.
#[test]
fn the_lowering_cost_on_two_and_four_ranks_is_one_boundary_per_collective() {
    let mapped = Dense::new(Family::Qwen2).heads(4, 4).f32_model();
    let graph = decode_graph(&mapped, 1);
    let mut measured = Vec::new();
    for world in [2u32, 4] {
        let reductions = shard(
            &graph,
            &mapped.map,
            ShardRule::Megatron,
            NonZeroU32::new(world).unwrap(),
        )
        .expect("the fixture shards")
        .collectives
        .len();
        let program = compile_over(&graph, &mapped.map, &options(), world).unwrap();
        let transfers = program.transfers();
        let vector = &transfers[0].aval;
        assert!(
            transfers.iter().all(|transfer| transfer.aval == *vector),
            "every exchanged value is one partial vector"
        );
        let bytes = vector.shape.iter().product::<usize>() * 4;
        let (w, r) = (world as usize, reductions);
        let got = Cost {
            stages: program.stages().count(),
            transfers: transfers.len(),
            bytes: transfers.len() * bytes,
        };
        let want = Cost {
            stages: w * r,
            transfers: (r - 1) * w * (w - 1) + (w - 1),
            bytes: ((r - 1) * w * (w - 1) + (w - 1)) * bytes,
        };
        assert_eq!(got, want, "world {world} over {r} reductions");
        measured.push(got);
    }
    assert_eq!(
        measured,
        [
            Cost {
                stages: 8,
                transfers: 7,
                bytes: 1792
            },
            Cost {
                stages: 16,
                transfers: 39,
                bytes: 9984
            }
        ]
    );
}

#[derive(Debug, PartialEq, Eq)]
struct Cost {
    stages: usize,
    transfers: usize,
    bytes: usize,
}

/// The same walk over the other dense families' fixtures: Llama, Qwen3, Gemma 2 and Gemma 3 shard over
/// two ranks and match their whole graphs. OLMo 2 norms the query and key over the whole projection
/// width, an operand whose scale is a replicated vector over a split axis: it is refused by name.
#[test]
fn the_other_dense_families_shard_and_walk_to_the_whole_graph() {
    for (label, family) in [
        ("llama", Family::Llama),
        ("qwen3", Family::Qwen3),
        ("gemma3", Family::Gemma3),
        ("gemma2", Family::Gemma2),
    ] {
        let mapped = Dense::new(family).f32_model();
        let graph = decode_graph(&mapped, 1);
        let program = compile(&graph, &mapped.map, &options())
            .unwrap_or_else(|error| panic!("{label} shards: {error}"));
        let step = decode_step(&graph);
        assert_walk_matches(
            &program,
            &graph,
            (&*mapped.store, &*mapped.map),
            &step,
            &HashMap::new(),
            label,
        );
    }

    let mapped = Dense::new(Family::Olmo2).f32_model();
    let graph = decode_graph(&mapped, 1);
    let refusal = compile(&graph, &mapped.map, &options())
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(
            refusal,
            StagedCompileError::Shard(ShardError::Unpropagatable { axis: 2, .. })
        ),
        "{refusal}"
    );
}

// ---- paged KV (POOT-1104) -----------------------------------------------------------------

const POOL_SLOTS: usize = 16;

/// SC-001 (POOT-1104): a paged decode step of two rows, sharded over two ranks and walked by rank,
/// equals the whole graph on the logits and on every rank's slice of every carried KV pool, and each
/// rank's pool holds its own KV heads only: `[pool_slots, kv_heads / world, head_dim]`.
///
/// Mutation: let `Sharder::scatter_update` decide a `Whole` pool and result whatever `src` is: the
/// split K rows meet a whole pool and the pass refuses the attention read that follows
/// (`Unpropagatable` on a `matmul`), so this row goes red at `compile`.
#[test]
fn a_sharded_paged_qwen2_decode_step_equals_the_whole_graph_with_each_rank_holding_its_heads() {
    let mapped = tiny_qwen2();
    let graph = paged_decode_graph(&mapped, 2, POOL_SLOTS);
    let program = compile(&graph, &mapped.map, &options()).expect("a paged qwen2 shards");
    let step = decode_step(&graph);
    let weights = (&*mapped.store, &*mapped.map);
    assert_walk_matches(
        &program,
        &graph,
        weights,
        &step,
        &HashMap::new(),
        "paged qwen2",
    );

    let walked = walk(&program, weights, &step, &HashMap::new());
    let pools: Vec<&String> = step
        .state
        .keys()
        .filter(|name| name.starts_with("kv.pool."))
        .collect();
    assert_eq!(pools.len(), 4, "two layers, K and V: {pools:?}");
    for name in pools {
        let whole = step.state[name].shape().to_vec();
        assert_eq!(whole[0], POOL_SLOTS);
        for rank in 0..WORLD as usize {
            let held = walked.state[&(name.clone(), rank)].shape().to_vec();
            assert_eq!(
                held,
                [whole[0], whole[1] / WORLD as usize, whole[2]],
                "{name} on rank {rank}"
            );
        }
    }
}

/// A head count that does not divide over the ranks is refused by name, on a paged step as on a
/// contiguous one, before any stage compiles: three heads of 16 over two ranks is 1.5 heads a rank
/// (`NotDivisible` on the reshape that exposes the head axis), and a single KV head cannot be cut
/// between ranks by heads at all (the pass will not split its width: the rotary half-slice refuses).
///
/// Mutation: fold the divisibility test back into the head-axis search of `Sharder::reshape`
/// (`running == prefix && extent > 1 && extent.is_multiple_of(world)`): the axis is skipped and the
/// refusal degrades to `Unpropagatable { op: "reshape [1, 1, 3, 16]" }`, so this row goes red.
#[test]
fn heads_that_do_not_divide_over_the_ranks_are_a_typed_refusal() {
    let threes = Dense::new(Family::Qwen2)
        .heads(3, 3)
        .head_dim(16)
        .f32_model();
    for graph in [
        paged_decode_graph(&threes, 1, POOL_SLOTS),
        decode_graph(&threes, 1),
    ] {
        let refusal = compile(&graph, &threes.map, &options())
            .map(|_| ())
            .unwrap_err();
        let StagedCompileError::Shard(ShardError::NotDivisible {
            extent: 3,
            world: 2,
            what,
        }) = &refusal
        else {
            panic!("three heads over two ranks: {refusal:?}");
        };
        assert!(what.contains("reshape"), "{what}");
    }

    let single = Dense::new(Family::Qwen2).heads(4, 1).f32_model();
    for graph in [
        paged_decode_graph(&single, 1, POOL_SLOTS),
        decode_graph(&single, 1),
    ] {
        let refusal = compile(&graph, &single.map, &options())
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                refusal,
                StagedCompileError::Shard(ShardError::Unpropagatable { axis: 3, .. })
            ),
            "one KV head over two ranks: {refusal:?}"
        );
    }
}
