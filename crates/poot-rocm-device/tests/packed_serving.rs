//! Card 642: packed decode through the ROCm executor contract
//! (`Engine<RocmDevice>`).
//!
//! A `packed_linear` graph (compiled, so `compile_staged` claims it as one `PackedContraction`) runs
//! through the one contract path (capture on the entry's first step, replay every step after -
//! Card 548: the eager/recorded/capture-replay distinction the pre-contract `RocmGraphExecutor` had
//! three copies of collapses into this one path, since the contract admits `Submission::Replay`
//! only), with the weight bound through the executable's `WeightStore` as a packed component -
//! never a dense f32 copy.
//!
//! The activation is one-hot, so every output is exactly one decoded weight value (`y[o] =
//! decode(o, k)`); every other product is a signed zero. Compared bit for bit with the 541 oracle's
//! decoder (`PackedPayload::decode_row`) after canonicalizing the sign of zero (`+ 0.0`), which a sum
//! of zero-valued partials does not preserve. The carried `[steps, out]` cache (written by
//! `DynamicUpdateSlice` at the `Pos` slot) is compared the same way after the last step.
//!
//! Mutation (role swap, W11a): bind a packed carrier under the wrong role's component name; the
//! planner's/binder's role check must refuse it before any upload, and the AWQ/GPTQ rows go red if
//! that check is skipped.

use std::collections::HashMap;
use std::sync::Arc;

use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Builder, Graph, PackedSourceName, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::test_support::gpu_caps;
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, production_plan::Submission,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_quant::{PackedPayload, StoredBytes};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::DType;

const K: usize = 256;
const OUT: usize = 8;
const STEPS: usize = 4;

fn formats() -> Vec<WeightFormat> {
    let nz = |n| std::num::NonZeroUsize::new(n).unwrap();
    vec![
        WeightFormat::Q4_K,
        WeightFormat::Q8_0,
        WeightFormat::Awq { group_size: nz(64) },
        WeightFormat::Gptq {
            groups: GroupMap::Indexed { groups: nz(4) },
        },
    ]
}

/// The layouts of a routed expert table: [`formats`] and the codebook-scaled 4-bit block format.
fn expert_formats() -> Vec<WeightFormat> {
    let mut formats = formats();
    formats.push(WeightFormat::Mxfp4);
    formats
}

/// The decode column one step selects.
fn column(step: usize) -> usize {
    (step * 67 + 5) % K
}

/// The oracle value a one-hot contraction produces, zero sign canonicalized.
fn canonical(value: f32) -> u32 {
    (value + 0.0).to_bits()
}

fn staged(target: Target, graph: &Graph) -> poot_graph_plan::staged::StagedProgram {
    let graph = graph.clone();
    compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .unwrap()
}

/// A `WeightStore` with one packed entry per named linear carrier the graph's inputs name, all
/// pointing at the same payload (single-expert fixtures) unless `owners` maps a distinct one per
/// `linear_id`.
fn store_for(graph: &Graph, owners: &HashMap<String, Arc<PackedPayload>>) -> Arc<WeightStore> {
    let mut builder = WeightStore::builder();
    let mut seen = std::collections::HashSet::new();
    for &id in graph.inputs() {
        let Some(name) = graph
            .meta(id)
            .name
            .as_deref()
            .and_then(PackedSourceName::parse)
        else {
            continue;
        };
        if !seen.insert(name.linear_id().to_string()) {
            continue;
        }
        if let Some(owner) = owners.get(name.linear_id()) {
            builder
                .insert(name.linear_id(), WeightEntry::Packed(Arc::clone(owner)))
                .unwrap();
            continue;
        }
        // An expert stack: its experts are the entries `{stack id}.{expert}`, which `ConstWeights` views as
        // one `ExpertStack`.
        let experts: Vec<_> = (0..)
            .map_while(|expert| owners.get(&format!("{}.{expert}", name.linear_id())))
            .collect();
        assert!(
            !experts.is_empty(),
            "no payload registered for linear_id {}",
            name.linear_id()
        );
        for (expert, owner) in experts.into_iter().enumerate() {
            builder
                .insert(
                    format!("{}.{expert}", name.linear_id()),
                    WeightEntry::Packed(Arc::clone(owner)),
                )
                .unwrap();
        }
    }
    Arc::new(builder.build())
}

fn slot_inputs<'a>(
    graph: &Graph,
    values: &'a [(SlotKey, Vec<usize>, DType, Vec<u8>)],
) -> StepInputs<'a> {
    let mut inputs = StepInputs::new();
    for (key, shape, dtype, bytes) in values {
        let elems = shape.iter().product();
        inputs.push(
            key.clone(),
            shape,
            poot_tensor::HostView::new(*dtype, elems, bytes).unwrap(),
        );
    }
    let _ = graph;
    inputs
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(data: &[i32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn one_hot_bytes(rows: usize, cols: usize, hot: impl Fn(usize) -> usize) -> Vec<u8> {
    let mut data = vec![0.0f32; rows * cols];
    for r in 0..rows {
        data[r * cols + hot(r)] = 1.0;
    }
    f32_bytes(&data)
}

fn assert_decode_row(what: &str, owner: &PackedPayload, got: &[f32], k: usize) {
    assert_eq!(got.len(), OUT, "{what}");
    let mut row = vec![0.0f32; K];
    for (o, value) in got.iter().enumerate() {
        owner.decode_row(o, &mut row).unwrap();
        let expected = row[k];
        assert_eq!(
            canonical(*value),
            canonical(expected),
            "{what}: [{o}, k={k}] device {value} vs oracle {expected}"
        );
    }
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A graph whose sole output is the named state value: a production entry stepped like any other,
/// which happens to let the test read the carried cache back through its real step output (no
/// `read_state` escape hatch).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph {
    let b = Builder::for_tests();
    let state = b.state_input(name, aval, StateRole::Recurrent).unwrap();
    b.finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
        .unwrap()
}

/// Decode over an explicit one-hot activation slot: `y = x @ W^T`, the `[STEPS, OUT]` cache written
/// at the `Pos` slot.
fn slot_decode_graph(descriptor: poot_quant::PackedWeight) -> (Graph, SlotKey, SlotKey) {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "onehot", TensorType::f32(vec![1, K]))
        .unwrap();
    let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
    let pos = b
        .slot(Slot::Pos, TensorType::new(vec![], DType::I32))
        .unwrap();
    let cache = b
        .state_input(
            "decoded",
            TensorType::f32(vec![STEPS, OUT]),
            StateRole::Recurrent,
        )
        .unwrap();
    let cache_out = b.dynamic_update_slice_dyn(cache, y, pos, 0).unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
        .unwrap();
    let xk = g.meta(x.id()).slot_key().unwrap().clone();
    let pk = g.meta(pos.id()).slot_key().unwrap().clone();
    (g, xk, pk)
}

/// W9: a packed decode entry on `Engine<RocmDevice>` is bit-exact against the oracle decoder across
/// `STEPS` steps, and the carried cache matches once read back through a state-probe entry sharing
/// the same (name, aval, storage).
#[test]
fn packed_decode_through_the_contract_is_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());
        let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
        let store = store_for(&graph, &owners);
        let mut weights = ConstWeights::new(&store);
        let named = weights.graph(&graph);
        let exe = engine
            .load_weights(Arc::clone(&store), Arc::new(weights.map()))
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
        for step in 0..STEPS {
            let k = column(step);
            let values = [
                (
                    x_key.clone(),
                    vec![1, K],
                    DType::F32,
                    one_hot_bytes(1, K, |_| k),
                ),
                (
                    pos_key.clone(),
                    vec![],
                    DType::I32,
                    i32_bytes(&[step as i32]),
                ),
            ];
            let inputs = slot_inputs(&graph, &values);
            let bytes = engine
                .step(exe, entry, &inputs, &mut NoSync)
                .unwrap_or_else(|e| panic!("{format:?} step {step}: {e}"))
                .read()
                .unwrap();
            assert_decode_row(
                &format!("{format:?} step {step}"),
                &owner,
                &bytemuck_f32(&bytes),
                k,
            );
        }
        engine.remove_entry(exe, entry).unwrap();

        let probe_graph = state_probe_graph("decoded", TensorType::f32(vec![STEPS, OUT]));
        let probe_entry = engine
            .add_entry(exe, &staged(target, &probe_graph))
            .unwrap();
        let cache = bytemuck_f32(
            &engine
                .step(exe, probe_entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        for step in 0..STEPS {
            assert_decode_row(
                &format!("{format:?} carried row {step}"),
                &owner,
                &cache[step * OUT..(step + 1) * OUT],
                column(step),
            );
        }
        engine.unload(exe).unwrap();
    }
}

/// SC-004 (card 548): a packed E4M3-per-channel linear decode entry on `Engine<RocmDevice>` is
/// bit-exact against the 541 oracle decoder, same contract path and same assertion shape as
/// `packed_decode_through_the_contract_is_bit_exact` above - its own dedicated row because E4M3 is
/// the planar-storage format `admits_planar`'s `Contraction` admission (card 542c) covers, replacing
/// the FP8 resident region Card 546c deleted (R472-005), not one more entry in `formats()` (which
/// this file's role-swap mutation already exercises for Q4_K/Q8_0/AWQ/GPTQ). Mutation: in
/// `poot-kernelgen/src/packed/planar.rs::PlanarPlan::emit_value`, drop the `Some(scale) => Mul(...)`
/// arm (always fall through to the unscaled `value`); every decoded element then reads the raw E4M3
/// code with no per-channel scale applied, and this row goes red.
#[test]
fn packed_e4m3_per_channel_decode_through_the_contract_is_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let format = WeightFormat::E4m3PerChannel {
        scale: poot_quant::format::ScaleEncoding::F32,
    };
    let owner = Arc::new(poot_quant::test_support::packed::random_payload(
        format,
        [OUT, K],
        97,
    ));
    let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());
    let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
    let store = store_for(&graph, &owners);
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(&graph);
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
    for step in 0..STEPS {
        let k = column(step);
        let values = [
            (
                x_key.clone(),
                vec![1, K],
                DType::F32,
                one_hot_bytes(1, K, |_| k),
            ),
            (
                pos_key.clone(),
                vec![],
                DType::I32,
                i32_bytes(&[step as i32]),
            ),
        ];
        let inputs = slot_inputs(&graph, &values);
        let bytes = engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("{format:?} step {step}: {e}"))
            .read()
            .unwrap();
        assert_decode_row(
            &format!("{format:?} step {step}"),
            &owner,
            &bytemuck_f32(&bytes),
            k,
        );
    }
    engine.unload(exe).unwrap();
}

/// W9 prefill: an identity activation decodes the whole weight in one forward, bit-exact per
/// element, output and carried state.
#[test]
fn packed_prefill_through_the_contract_is_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 29,
        ));
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "onehot", TensorType::f32(vec![K, K]))
            .unwrap();
        let y =
            poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None).unwrap();
        let cache = b
            .state_input(
                "decoded",
                TensorType::f32(vec![K, OUT]),
                StateRole::Recurrent,
            )
            .unwrap();
        let cache_out = b.dynamic_update_slice(cache, y, 0, 0).unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
            .unwrap();
        let x_key = graph.meta(x.id()).slot_key().unwrap().clone();

        let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
        let store = store_for(&graph, &owners);
        let mut weights = ConstWeights::new(&store);
        let named = weights.graph(&graph);
        let exe = engine
            .load_weights(Arc::clone(&store), Arc::new(weights.map()))
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
        let values = [(x_key, vec![K, K], DType::F32, one_hot_bytes(K, K, |r| r))];
        let inputs = slot_inputs(&graph, &values);
        let y_bytes = engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("{format:?} prefill: {e}"))
            .read()
            .unwrap();
        let y_data = bytemuck_f32(&y_bytes);
        engine.remove_entry(exe, entry).unwrap();

        let probe_graph = state_probe_graph("decoded", TensorType::f32(vec![K, OUT]));
        let probe_entry = engine
            .add_entry(exe, &staged(target, &probe_graph))
            .unwrap();
        let cache = bytemuck_f32(
            &engine
                .step(exe, probe_entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        for r in 0..K {
            let row = &y_data[r * OUT..(r + 1) * OUT];
            assert_decode_row(&format!("{format:?} row {r}"), &owner, row, r);
            let carried = &cache[r * OUT..(r + 1) * OUT];
            assert_decode_row(&format!("{format:?} carried {r}"), &owner, carried, r);
        }
        engine.unload(exe).unwrap();
    }
}

/// W11a: a packed carrier bound under the wrong role's component is refused before any device write.
/// Mutation: in the `WeightStore` builder, register the owner's two source roles with their names
/// swapped; the AWQ binding runs with scales where zeros belong (or vice versa) instead of being
/// refused, and this row goes red.
#[test]
fn a_swapped_packed_role_is_refused_before_upload() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let format = formats()[2];
    let owner = Arc::new(poot_quant::test_support::packed::random_payload(
        format,
        [OUT, K],
        3,
    ));
    let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());

    // W11a probe: bind "layer" to a payload whose component roles report K and OUT swapped (an
    // AWQ scale table shaped as if it were the quantized weight table, and vice versa) - the same
    // observable fault a role-swapped upload produces, reached through the production
    // `weight_buffer`/`PackedComponentRef` path rather than a forged test-only field.
    let swapped_owner = Arc::new(poot_quant::test_support::packed::random_payload(
        format,
        [K, OUT],
        3,
    ));
    let owners = HashMap::from([("layer".to_string(), swapped_owner)]);
    let store = store_for(&graph, &owners);
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(&graph);
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let staged_program = staged(target, &named);
    let Err(error) = engine.add_entry(exe, &staged_program) else {
        panic!("a role/shape-swapped packed carrier must be refused, not silently bound");
    };
    eprintln!("a_swapped_packed_role_is_refused_before_upload: observed {error}");
    let _ = (x_key, pos_key);
}

const EXPERTS: usize = 3;
const ROWS: usize = 5;
/// Rows past `INDEXED_GEMV_MAX_ROWS` (32), so an indexed contraction of them runs the segmented Tiled schedule.
const PREFILL_ROWS: usize = 70;
const PREFILL_OUT: usize = 11;
/// Row `m` routed to expert `(2m + 1) % EXPERTS`: every window of rows holds all the experts.
fn interleaved(m: usize) -> usize {
    (m * 2 + 1) % EXPERTS
}

/// Prefill routing in which a tile can skip an expert: the first half of every 64 rows takes only experts 2 and 0
/// in an unsorted order (the segmented walk must step from expert 0 to expert 2, not to the next integer), the
/// second half is [`interleaved`].
fn gapped(m: usize) -> usize {
    if m % 64 < 32 {
        [2, 0, 2, 0, 0, 2, 2, 0][m % 8]
    } else {
        interleaved(m)
    }
}

/// 17.0M outputs: past the 65,535 x 256 the packed Gemv body can address on wgpu.
const GRID_LIMIT_ROWS: usize = 4096;
const GRID_LIMIT_OUT: usize = 4160;

/// The decode column row `m` selects of a `k`-wide table.
fn column_of(m: usize, k: usize) -> usize {
    (m * 67 + 5) % k
}

/// Every row of `owner` decoded by the oracle, `[out, k]` flattened: decoded once per expert, so a table of
/// millions of outputs is not re-decoded for each of the rows that read it.
fn decoded_rows(owner: &PackedPayload) -> Vec<f32> {
    let [out, k] = owner.weight().shape();
    let mut decoded = vec![0.0f32; out * k];
    for (o, row) in decoded.chunks_mut(k).enumerate() {
        owner.decode_row(o, row).unwrap();
    }
    decoded
}

fn assert_indexed_row(what: &str, decoded: &[f32], k: usize, got: &[f32], column: usize) {
    assert_eq!(got.len() * k, decoded.len(), "{what}");
    for (o, value) in got.iter().enumerate() {
        let want = decoded[o * k + column];
        assert_eq!(
            canonical(*value),
            canonical(want),
            "{what}: [{o}, k={column}] device {value} vs oracle {want}"
        );
    }
}

/// How the stacked table of an indexed packed launch binds (card 1067).
#[derive(Clone, Copy, PartialEq, Eq)]
enum StackBuffers {
    /// Every source is one buffer.
    One,
    /// The target's single-buffer limit holds two of the three experts of the largest source, so `legalize` splits
    /// the table into the expert ranges 0..2 and 2..3, each its own buffer.
    Split,
}

impl StackBuffers {
    /// `g` as `legalize` leaves it for a device of `target` whose single-buffer limit is this layout's: the stacked
    /// table split into the expert ranges the limit holds. The device keeps its real caps, so the result is
    /// compiled and run for the real target; `g` itself is the unsplit graph.
    fn graph(self, g: &Graph, target: Target, descriptor: poot_quant::PackedWeight) -> Graph {
        match self {
            Self::One => g.clone(),
            Self::Split => {
                let slab = descriptor
                    .sources()
                    .into_iter()
                    .map(|role| descriptor.slab_bytes(role))
                    .max()
                    .expect("a packed weight has a source") as u64;
                assert!(
                    2 * slab < 4096,
                    "a limit this small is used as is, not rounded"
                );
                let caps = poot_target::DeviceCaps {
                    max_buffer_bytes: 2 * slab,
                    ..gpu_caps(&target)
                };
                let split =
                    poot_graph_plan::legalize(g, &caps, &poot_graph_plan::CompileLimits::STANDARD)
                        .expect("the stacked table splits");
                let carriers: Vec<usize> = split
                    .consts()
                    .iter()
                    .map(|&id| split.aval(id).shape[0])
                    .collect();
                assert_eq!(
                    carriers,
                    self.carrier_experts(descriptor.sources().len()),
                    "the experts each stacked carrier holds after the split"
                );
                split
            }
        }
    }

    /// The leading extents of the compiled graph's stacked carriers: one buffer of every expert, or the 2 + 1
    /// split with every source's piece 0 before its piece 1.
    fn carrier_experts(self, sources: usize) -> Vec<usize> {
        match self {
            Self::One => vec![EXPERTS; sources],
            Self::Split => [[2].repeat(sources), [1].repeat(sources)].concat(),
        }
    }
}

/// One `packed_indexed_linear` launch over `EXPERTS` stacked `[out, k]` experts of `format`, row `m` routed to
/// the expert `expert_of(m)` routes it to with a one-hot activation: one `PackedContraction { rows: Indexed }`, each row
/// bit-exact against the oracle decoder (zero sign canonicalized).
fn run_indexed_case(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    format: WeightFormat,
    (rows, out, k): (usize, usize, usize),
    expert_of: fn(usize) -> usize,
    buffers: StackBuffers,
) {
    let owners_vec: Vec<Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| {
            Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
                format,
                [out, k],
                e as u64 * 7 + 101,
            ))
        })
        .collect();
    let table = poot_graph_ir::ops::PackedExpertStack {
        stack_id: "experts".to_string(),
        descriptor: owners_vec[0].weight(),
        experts: EXPERTS,
    };
    let what = format!("{format:?}");
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![rows, k]))
        .unwrap();
    let ids = b
        .slot_named_ranged(
            Slot::Activation,
            "moe-ids",
            TensorType::new(vec![rows], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let y = poot_graph_ir::ops::packed_indexed_linear(&b, x, ids, &table).unwrap();
    let graph = b.finish(y).unwrap();
    let x_key = graph.meta(x.id()).slot_key().unwrap().clone();
    let ids_key = graph.meta(ids.id()).slot_key().unwrap().clone();

    let owners: HashMap<String, Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| (format!("experts.{e}"), Arc::clone(&owners_vec[e])))
        .collect();
    let store = store_for(&graph, &owners);
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(&graph);
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let named = buffers.graph(&named, target, table.descriptor);
    let program = staged(target, &named);
    let compiled = program.stages().next().expect("single stage").2.graph();
    let carriers: Vec<usize> = compiled
        .consts()
        .iter()
        .map(|&id| compiled.aval(id).shape[0])
        .collect();
    assert_eq!(
        carriers,
        buffers.carrier_experts(table.descriptor.sources().len()),
        "{what}: the experts each stacked carrier holds"
    );
    if buffers == StackBuffers::One {
        assert!(
            matches!(
                compiled.eqns(),
                [poot_graph_ir::Eqn {
                    op: poot_graph_ir::OpKind::PackedContraction {
                        rows: poot_graph_ir::ContractionRows::Indexed { experts: EXPERTS },
                        ..
                    },
                    ..
                }]
            ),
            "{what}: {:?}",
            compiled.eqns()
        );
    }
    let entry = engine.add_entry(exe, &program).unwrap();

    let selected: Vec<i32> = (0..rows).map(|m| expert_of(m) as i32).collect();
    let values = [
        (
            x_key,
            vec![rows, k],
            DType::F32,
            one_hot_bytes(rows, k, |m| column_of(m, k)),
        ),
        (ids_key, vec![rows], DType::I32, i32_bytes(&selected)),
    ];
    let inputs = slot_inputs(&graph, &values);
    let got = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("{what}: {e}"))
            .read()
            .unwrap(),
    );
    let decoded: Vec<Vec<f32>> = owners_vec.iter().map(|owner| decoded_rows(owner)).collect();
    for m in 0..rows {
        assert_indexed_row(
            &format!("{what} row {m} expert {}", expert_of(m)),
            &decoded[expert_of(m)],
            k,
            &got[m * out..(m + 1) * out],
            column_of(m, k),
        );
    }
    engine.unload(exe).unwrap();
}

/// SC-001 (ROCm): `packed_indexed_linear` over a stacked table of every format runs through the contract
/// as one `PackedContraction { rows: Indexed }`, one-hot rows bit-exact against the oracle decoder.
/// Mutation: slab stride 0 in the planner's metadata (every row reads expert 0); the rows routed to
/// experts 1 and 2 go red.
#[test]
fn packed_indexed_contraction_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for format in expert_formats() {
        run_indexed_case(
            &mut engine,
            target,
            format,
            (ROWS, OUT, K),
            interleaved,
            StackBuffers::One,
        );
    }
}

/// Card 1067 (ROCm): a stacked table over the target's single-buffer limit runs split into expert ranges
/// (0..2 and 2..3, each its own buffer per source) and is bit-exact against the oracle decoder on every format,
/// with rows routed to every expert, the last one alone in its buffer. The reduced limit is `legalize`'s one input
/// (`StackBuffers::Split`); the graph, weights and executor are those of the unsplit case.
/// Mutation: in `legalize`'s `split_oversized_expert_stacks` shift the ids of the later pieces by one less
/// than `range.start`; the row routed to expert 2 reads the wrong slab and goes red.
#[test]
fn a_split_expert_stack_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for format in expert_formats() {
        run_indexed_case(
            &mut engine,
            target,
            format,
            (ROWS, 2, K),
            interleaved,
            StackBuffers::Split,
        );
    }
}

/// SC-002 (ROCm): a Q8_0 table with `out = 3, K = 32` is 102 bytes per expert, not a whole number of words,
/// so the binder pads each expert to 104 bytes and the kernel strides by that; one-hot rows routed to
/// experts 1 and 2 are bit-exact. Mutation: drop the per-expert padding in `WeightWriter::read_stream`; experts 1 and
/// 2 read misaligned blocks and go red.
#[test]
fn an_unaligned_expert_slab_table_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    run_indexed_case(
        &mut engine,
        target,
        WeightFormat::Q8_0,
        (ROWS, 3, 32),
        interleaved,
        StackBuffers::One,
    );
}

/// A prefill-sized routed table (ROCm): `PREFILL_ROWS` rows is past the decode row count, so the indexed packed
/// contraction runs the segmented Tiled schedule (rows routed by `gapped`, so a tile holds several
/// experts and may skip one) over every format, with `N = 11` leaving edge columns in the last column tile. One-hot rows bit-exact
/// against the oracle decoder. Mutation: take the Tiled step's slab from the wrong segment (`weight.step(.., lo,
/// ..)` in `indexed_tiled_body`); rows routed to an expert after a gap read another expert and go red.
#[test]
fn packed_indexed_prefill_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for format in expert_formats() {
        run_indexed_case(
            &mut engine,
            target,
            format,
            (PREFILL_ROWS, PREFILL_OUT, K),
            gapped,
            StackBuffers::One,
        );
    }
}

/// A routed prefill above the wgpu grid limit (ROCm): `4096 x 4160` outputs of a Q8_0 table is 17.0M, past
/// 65,535 x 256, which the Gemv-body route refused; the Tiled schedule plans it (folded onto `GroupY` where the
/// device's X extent is smaller than the tile count) and every one of the 17M outputs is bit-exact against the
/// oracle decoder. Mutation: read the tile index from `GroupX` alone in `indexed_tiled_body`; every tile past
/// the first fold row is never computed and goes red.
#[test]
fn packed_indexed_prefill_above_the_gemv_grid_limit_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    run_indexed_case(
        &mut engine,
        target,
        WeightFormat::Q8_0,
        (GRID_LIMIT_ROWS, GRID_LIMIT_OUT, 64),
        gapped,
        StackBuffers::One,
    );
}

/// Card 545a: a quantized token-embedding lookup (`packed_embedding`, claimed as one
/// `PackedRowGather`) through the contract, for a decode token and a prefill id vector, bit-exact
/// against the oracle decoder's rows (a pure decode, so no zero-sign canonicalization). Mutation:
/// read row `ids[r] + 1` in the RowGather body; every row goes red.
#[test]
fn packed_embedding_row_gather_is_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    const VOCAB: usize = 12;
    for (seed, format) in [WeightFormat::Q4_K, WeightFormat::Q8_0, WeightFormat::Q6_K]
        .into_iter()
        .enumerate()
    {
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [VOCAB, K],
            seed as u64 + 61,
        ));
        for ids in [vec![7usize], vec![3, 0, 11, 3, 5]] {
            let what = format!("{format:?} ids {ids:?}");
            let shape = if ids.len() == 1 {
                vec![]
            } else {
                vec![ids.len()]
            };
            let b = Builder::for_tests();
            let token = b
                .slot_named(
                    Slot::Token,
                    "token",
                    TensorType::new(shape.clone(), DType::I32),
                )
                .unwrap();
            let rows = poot_graph_ir::ops::packed_embedding(
                &b,
                token,
                "model.embed_tokens",
                owner.weight(),
            )
            .unwrap();
            let graph = b.finish(rows).unwrap();
            let token_key = graph.meta(token.id()).slot_key().unwrap().clone();

            let owners = HashMap::from([("model.embed_tokens".to_string(), Arc::clone(&owner))]);
            let store = store_for(&graph, &owners);
            let mut weights = ConstWeights::new(&store);
            let named = weights.graph(&graph);
            let exe = engine
                .load_weights(Arc::clone(&store), Arc::new(weights.map()))
                .unwrap();
            let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
            let token_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let values = [(token_key, shape.clone(), DType::I32, i32_bytes(&token_ids))];
            let inputs = slot_inputs(&graph, &values);
            let out = bytemuck_f32(
                &engine
                    .step(exe, entry, &inputs, &mut NoSync)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .read()
                    .unwrap(),
            );
            assert_eq!(out.len(), ids.len() * K, "{what}");
            let mut expected = vec![0.0f32; K];
            for (r, &id) in ids.iter().enumerate() {
                owner.decode_row(id, &mut expected).unwrap();
                for (c, (got, want)) in out[r * K..(r + 1) * K].iter().zip(&expected).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{what}: row {r} (id {id}) col {c}: device {got} vs oracle {want}"
                    );
                }
            }
            engine.unload(exe).unwrap();
        }
    }
}

/// POOT-1060: the sharded-PLE lookup's table shape - a dense E4M3FN `[rows, width]` checkpoint const
/// gathered by an exact I32 index and widened in one `DenseRowGather` - through the executor contract,
/// bit-exact against `poot_quant::scalar::e4m3fn_to_f32`. The table holds all 256 codes (NaN
/// included) and is read at its first and last row, a repeated row and an interior row, at widths 1, 3,
/// 4, 6 and 8: a width that is not a multiple of four pads each row to a whole `u32` word
/// (`words_per_row = ceil(width / 4)`), so a flat `element / 4` lane read goes red on the odd widths.
/// Mutations (each red, then restored): pad rows to `width / 4` words in `lanes::row_padded_words`;
/// flip the sign/exponent decode in `packed_e4m3_row_gather.rs`; read `index[s] + 1`.
#[test]
fn dense_e4m3_row_gather_is_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for width in [1usize, 3, 4, 6, 8] {
        let rows = 256 / width + 3;
        let table: Vec<u8> = (0..rows * width).map(|i| (i % 256) as u8).collect();
        let ids: Vec<i32> = vec![rows as i32 - 1, 0, rows as i32 / 2, rows as i32 - 1, 1];
        let b = Builder::for_tests();
        let table_value = b
            .constant(
                "ple.table",
                TensorType::new(vec![rows, width], DType::E4M3FN),
            )
            .unwrap();
        let idx = b
            .slot_named(
                Slot::Activation,
                "idx",
                TensorType::new(vec![ids.len()], DType::I32),
            )
            .unwrap();
        let gathered = b
            .cast(b.gather(table_value, 0, idx).unwrap(), DType::F32)
            .unwrap();
        let graph = b.finish(gathered).unwrap();
        let idx_key = graph.meta(idx.id()).slot_key().unwrap().clone();

        let store = {
            let mut builder = WeightStore::builder();
            let dense = DenseWeight::try_new(
                DType::E4M3FN,
                vec![rows, width],
                StoredBytes::from(table.clone()),
            )
            .unwrap();
            builder
                .insert("ple.table", WeightEntry::Dense(dense))
                .unwrap();
            Arc::new(builder.build())
        };
        let mut weights = ConstWeights::new(&store);
        let named = weights.graph(&graph);
        let exe = engine
            .load_weights(Arc::clone(&store), Arc::new(weights.map()))
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let shape = [ids.len()];
        let mut inputs = StepInputs::new();
        inputs.push(
            idx_key,
            &shape,
            poot_tensor::HostView::new(DType::I32, ids.len(), &id_bytes).unwrap(),
        );
        let bytes = engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        let out = bytemuck_f32(&bytes);
        assert_eq!(out.len(), ids.len() * width, "width {width}");
        for (r, &id) in ids.iter().enumerate() {
            for c in 0..width {
                let byte = table[id as usize * width + c];
                let want = poot_quant::scalar::e4m3fn_to_f32(byte);
                let got = out[r * width + c];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "width {width} row {r} (id {id}) col {c} byte {byte:#04x}: device {got} vs oracle {want}"
                );
            }
        }
        engine.unload(exe).unwrap();
    }
}

// --- SC-001 and SC-003 of POOT-728: dense activations through the packed loaders, and a packed head's epilogue ---

/// The packed formats of the loader rows: the K-quant and block-32 GGUF layouts and the FP8 planar cells (a
/// per-row and a 128x128-block scale), so the block-storage and planar front ends both serve every row.
fn loader_formats() -> Vec<WeightFormat> {
    use poot_quant::format::ScaleEncoding;
    vec![
        WeightFormat::Q4_K,
        WeightFormat::Q8_0,
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
    ]
}

/// A deterministic fill in `[-0.5, 0.5)` with no zero element, so a dropped term changes every output.
fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.9 - 0.45 + 0.05
        })
        .collect()
}

/// `graph` stepped once on `engine`, the packed weight bound as `owner`, the activation slot
/// `x_key` bound to `activation` (`[rows, K]`) and every named dense `[OUT]`-or-`[1, OUT]` const in `consts`
/// bound by name; the output read back.
#[allow(clippy::too_many_arguments)]
fn run_dense_activation(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    graph: &Graph,
    owner: &Arc<PackedPayload>,
    x_key: &SlotKey,
    activation: (&[usize], &[f32]),
    consts: &[(&str, &[usize], &[f32])],
) -> Vec<f32> {
    let mut builder = WeightStore::builder();
    let mut seen = std::collections::HashSet::new();
    for &id in graph.inputs() {
        let Some(name) = graph
            .meta(id)
            .name
            .as_deref()
            .and_then(PackedSourceName::parse)
        else {
            continue;
        };
        if seen.insert(name.linear_id().to_string()) {
            builder
                .insert(name.linear_id(), WeightEntry::Packed(Arc::clone(owner)))
                .unwrap();
        }
    }
    for (name, shape, values) in consts {
        let dense = poot_quant::weights::DenseWeight::try_new(
            DType::F32,
            shape.to_vec(),
            StoredBytes::from(f32_bytes(values)),
        )
        .unwrap();
        builder
            .insert(name.to_string(), WeightEntry::Dense(dense))
            .unwrap();
    }
    let store = Arc::new(builder.build());
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(graph);
    let exe = engine
        .load_weights(Arc::clone(&store), Arc::new(weights.map()))
        .unwrap();
    let entry = engine.add_entry(exe, &staged(target, &named)).unwrap();
    let activation_bytes = f32_bytes(activation.1);
    let values = [(
        x_key.clone(),
        activation.0.to_vec(),
        DType::F32,
        activation_bytes,
    )];
    let inputs = slot_inputs(graph, &values);
    let bytes = engine
        .step(exe, entry, &inputs, &mut NoSync)
        .unwrap()
        .read()
        .unwrap();
    engine.remove_entry(exe, entry).unwrap();
    engine.unload(exe).unwrap();
    bytemuck_f32(&bytes)
}

/// SC-001: Q4_K, Q8_0 and FP8 contractions through the packed loaders meet their tier-2 oracle contract. A
/// dense activation makes every output a sum over every `K` block, so a dropped block or a wrong scale moves
/// every element; `m == 1` takes the Gemv schedule and `m == 17` the Tiled schedule (ragged against its tile),
/// and every output element is compared. Mutation: omit one block's contribution in `DecodePlan`'s address or
/// scale a factor by a nonzero constant; every row goes red.
#[test]
fn packed_loaders_match_the_oracle_on_every_output_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for (seed, format) in loader_formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
            format,
            [OUT, K],
            seed as u64 + 31,
        ));
        for m in [1usize, 17] {
            let what = format!("{format:?} m={m}");
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, K]))
                .unwrap();
            let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None)
                .unwrap();
            let graph = b.finish(y).unwrap();
            let x_key = graph.meta(x.id()).slot_key().unwrap().clone();
            let activation = fill(7 + m as u64, m * K);
            let got = run_dense_activation(
                &mut engine,
                target,
                &graph,
                &owner,
                &x_key,
                (&[m, K], &activation),
                &[],
            );
            let mut oracle_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> =
                HashMap::new();
            for &id in graph.inputs() {
                if let Some(name) = graph
                    .meta(id)
                    .name
                    .as_deref()
                    .and_then(PackedSourceName::parse)
                {
                    oracle_inputs.insert(
                        id,
                        poot_quant::PackedComponentRef::new(Arc::clone(&owner), name.role()).into(),
                    );
                }
            }
            oracle_inputs.insert(
                x.id(),
                poot_tensor::HostTensor::f32(vec![m, K], activation).into(),
            );
            let want = poot_eval::eval(
                &graph,
                &oracle_inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .unwrap()
            .output
            .into_host()
            .unwrap();
            poot_executor_parity::assert_close(
                &got,
                want.as_f32().unwrap(),
                poot_executor_parity::TIER2,
                &what,
            );
        }
    }
}

/// SC-003, the head half: with an epilogue that cannot be contracted (`head + bias`, no multiply to fuse
/// with the add), the fused packed Gemv and the standalone one produce bit-identical results, so the head
/// sum is the same computation in both bodies. Every loader format, a generic and a cancelling bias, at
/// tolerance 0.0. A format whose decode carries `BinaryOpNoContract` (Q4_K's min term) loses that
/// protection if the no-contraction marker lands on a different op in the longer fused body, and its fused
/// head then drifts by an ulp or two. Mutation: place the SPIR-V `NoContraction` decoration by instruction
/// ordinal, as card 628 first did; the Q4_K row goes red on wgpu while the other formats stay green.
#[test]
fn a_packed_gemv_head_is_shared_with_the_standalone_gemv_on_rocm() {
    use poot_executor_parity::assert_close;
    use poot_graph_ir::op::{BinOp, ContractionHead};
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for (seed, format) in loader_formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
            format,
            [OUT, K],
            seed as u64 + 51,
        ));
        let what = format!("{format:?}");
        let activation: Vec<f32> = fill(3, K).into_iter().map(|v| v * 0.02).collect();
        let build = |with_bias: bool| {
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]))
                .unwrap();
            let head =
                poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None)
                    .unwrap();
            let out = if with_bias {
                let bias = b.constant("bias", TensorType::f32(vec![OUT])).unwrap();
                b.binary(BinOp::Add, head, bias).unwrap()
            } else {
                head
            };
            let g = b.finish(out).unwrap();
            let key = g.meta(x.id()).slot_key().unwrap().clone();
            (g, key)
        };

        let (head_graph, head_key) = build(false);
        let head = run_dense_activation(
            &mut engine,
            target,
            &head_graph,
            &owner,
            &head_key,
            (&[1, K], &activation),
            &[],
        );

        let (graph, x_key) = build(true);
        let fused_program = staged(target, &graph);
        let fused_graph = fused_program.stages().next().unwrap().2.graph();
        let heads: Vec<ContractionHead> = fused_graph
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                poot_graph_ir::OpKind::ContractionEpilogue { head, .. } => Some(*head),
                _ => None,
            })
            .collect();
        assert!(
            matches!(heads.as_slice(), [ContractionHead::Packed { .. }]),
            "{what}: one fused packed contraction: {:?}",
            fused_graph.eqns()
        );
        for (label, bias_values) in [
            ("generic bias", fill(4, OUT)),
            (
                "cancelling bias",
                head.iter().map(|v| -v).collect::<Vec<f32>>(),
            ),
        ] {
            let fused = run_dense_activation(
                &mut engine,
                target,
                &graph,
                &owner,
                &x_key,
                (&[1, K], &activation),
                &[("bias", &[OUT], &bias_values)],
            );
            let want: Vec<f32> = head
                .iter()
                .zip(&bias_values)
                .map(|(head, bias)| head + bias)
                .collect();
            assert_close(
                &fused,
                &want,
                0.0,
                &format!("{what} {label}: fused head vs the standalone head plus the bias"),
            );
        }
    }
}

/// SC-003: a packed head with a pointwise epilogue. `tanh(head + bias) * 0.5 + residual` over the `m == 1` Gemv
/// forms one `ContractionEpilogue { head: Packed }`, and the fused device program agrees with the chain over
/// the standalone head (the head summed on the device). The head itself is the standalone Gemv's (the
/// `head_is_shared` row); the epilogue's `bias` add and `0.5` multiply may contract to an FMA
/// (`contraction/epilogue.rs` deliberately does not forbid it), so this comparison is 1e-5, and is sharpened by a
/// cancellation fixture: `bias` is the negated device head, so every pre-activation is within a few ulps of zero
/// when the epilogue sees each column's completed sum, and the fused result is `residual` to a few ulps of the
/// head. An epilogue applied to a lane's partial accumulator (before the LDS reduction) adds the bias once per
/// lane and saturates the activation, moving the result by about half the lane count. Mutation: apply the
/// epilogue to the per-lane partial before the LDS reduction in `gemv_body`; the cancellation row and the
/// chain row go red.
#[test]
fn a_packed_gemv_epilogue_agrees_with_the_standalone_head_chain_on_rocm() {
    use poot_executor_parity::assert_close;
    use poot_graph_ir::op::{BinOp, ContractionHead, UnOp};
    use poot_graph_ir::types::Scalar;
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for (seed, format) in loader_formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
            format,
            [OUT, K],
            seed as u64 + 51,
        ));
        let what = format!("{format:?}");
        // Small activations keep the head sums O(1), so the activation does not saturate on the generic bias.
        let activation: Vec<f32> = fill(3, K).into_iter().map(|v| v * 0.02).collect();
        let build = |with_chain: bool| {
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]))
                .unwrap();
            let head =
                poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None)
                    .unwrap();
            if !with_chain {
                let g = b.finish(head).unwrap();
                let key = g.meta(x.id()).slot_key().unwrap().clone();
                return (g, x.id(), key);
            }
            let bias = b.constant("bias", TensorType::f32(vec![OUT])).unwrap();
            let residual = b
                .constant("residual", TensorType::f32(vec![1, OUT]))
                .unwrap();
            let biased = b.binary(BinOp::Add, head, bias).unwrap();
            let activated = b.unary(UnOp::Tanh, biased).unwrap();
            let scaled = b
                .binary_scalar(BinOp::Mul, activated, Scalar::F32(0.5))
                .unwrap();
            let out = b.binary(BinOp::Add, scaled, residual).unwrap();
            let g = b.finish(out).unwrap();
            let key = g.meta(x.id()).slot_key().unwrap().clone();
            (g, x.id(), key)
        };

        // The device head alone, to cancel against.
        let (head_graph, _, head_key) = build(false);
        let head = run_dense_activation(
            &mut engine,
            target,
            &head_graph,
            &owner,
            &head_key,
            (&[1, K], &activation),
            &[],
        );
        let head_scale = head.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        let (graph, _, x_key) = build(true);
        let residual_values = fill(5, OUT);
        let fused_program = staged(target, &graph);
        let fused_graph = fused_program.stages().next().unwrap().2.graph();
        let heads: Vec<ContractionHead> = fused_graph
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                poot_graph_ir::OpKind::ContractionEpilogue { head, .. } => Some(*head),
                _ => None,
            })
            .collect();
        assert!(
            matches!(heads.as_slice(), [ContractionHead::Packed { .. }]),
            "{what}: one fused packed contraction: {:?}",
            fused_graph.eqns()
        );
        for (label, bias_values) in [
            ("generic bias", fill(4, OUT)),
            (
                "cancelling bias",
                head.iter().map(|v| -v).collect::<Vec<f32>>(),
            ),
        ] {
            let fused = run_dense_activation(
                &mut engine,
                target,
                &graph,
                &owner,
                &x_key,
                (&[1, K], &activation),
                &[
                    ("bias", &[OUT], &bias_values),
                    ("residual", &[1, OUT], &residual_values),
                ],
            );
            // The chain over the standalone head: the epilogue's `bias` add and `0.5` multiply may contract
            // to an FMA (`contraction/epilogue.rs` deliberately does not forbid it), hence 1e-5.
            let want: Vec<f32> = head
                .iter()
                .zip(&bias_values)
                .zip(&residual_values)
                .map(|((head, bias), residual)| (head + bias).tanh() * 0.5 + residual)
                .collect();
            assert_close(
                &fused,
                &want,
                1e-5,
                &format!("{what} {label}: fused vs the chain over the standalone head"),
            );
            if label == "cancelling bias" {
                // With the head cancelled, the result is the residual to a few ulps of the head's size.
                let tolerance = 64.0 * f32::EPSILON * head_scale.max(1.0);
                for (index, (&got, &want)) in fused.iter().zip(&residual_values).enumerate() {
                    assert!(
                        (got - want).abs() <= tolerance,
                        "{what}: column {index}: {got} against the residual {want} (tolerance {tolerance}); \
                         the epilogue must see each column's completed sum"
                    );
                }
            }
        }
    }
}
