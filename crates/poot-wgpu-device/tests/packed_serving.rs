//! Card 642: packed decode through the wgpu SERVING executor paths.
//!
//! A `packed_linear` graph (compiled, so `compile` claims it as one `PackedContraction`) runs through
//! the executor contract (a decode entry with carried state and a zero-seeded prefill entry), with the weight bound as `Value::Packed` carriers - never a dense f32
//! copy.
//!
//! The activation is one-hot, so every output is exactly one decoded weight value: `y[o] =
//! decode(o, k)` for a decode row selecting column `k`, and `Y[r, o] = decode(o, r)` for the identity
//! prefill. Every other product is a signed zero, so the contraction's sum is the decoded value itself,
//! compared bit for bit with the 541 oracle's decoder (`PackedPayload::decode_row`) after canonicalizing
//! the sign of zero (`+ 0.0`), which a sum of zero-valued partials does not preserve. The carried
//! state (a `[steps, out]` cache written by `DynamicUpdateSlice` at the `Pos` slot) is compared the
//! same way after the last step.
//!
//! Mutation (role swap): in `bind_resident`'s packed-source arm, upload the component's owner with the
//! first two source roles exchanged; the AWQ and GPTQ rows go red.

use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_executor::{Device, Executor, NoSync, StepInputs};
use poot_graph_ir::{
    Builder, Graph, PackedSourceName, Slot, SlotKey, StateRole, Storage, TensorType, ValueId,
};
use poot_graph_plan::test_support::gpu_caps;
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, production_plan::Submission,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::StagedProgram,
    staged::TargetSet, staged::compile_staged,
};
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_quant::{
    PackedComponentRef, PackedPayload, PackedStack, PackedStackComponentRef, StoredBytes,
};
use poot_runtime_common::DeviceBackend;
use poot_tensor::{HostTensor, HostView};
use poot_wgpu_device::device::WgpuDevice;

/// Compile `g` for `target` with `Submission::Replay` (Card 546b: the contract admits no other
/// submission).
fn staged(g: &Graph, target: Target) -> StagedProgram {
    let g = g.clone();
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the packed graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// One slot's bytes, ready for [`StepInputs::push`].
struct SlotBind {
    key: SlotKey,
    shape: Vec<usize>,
    dtype: DType,
    bytes: Vec<u8>,
}

fn step_inputs(binds: &[SlotBind]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for b in binds {
        let elems = b.shape.iter().product();
        inputs.push(
            b.key.clone(),
            &b.shape,
            HostView::new(b.dtype, elems, &b.bytes).unwrap(),
        );
    }
    inputs
}

/// Split a `HashMap<ValueId, Value>` bind into the contract's two bind mechanisms: every
/// `Storage::Const` becomes a named [`WeightStore`] entry, and every `Storage::Slot` becomes a
/// [`StepInputs`] row. `Value::Packed` carriers are deduplicated by linear id into one
/// `WeightEntry::Packed` each; the contract binds one payload per linear id and reads each operand's
/// role from the graph's own declared name (`Engine::weight_buffer`), never a per-value
/// caller-supplied role the way the deleted `GpuExecutor::bind_resident` did. `Value::Host` tensors become a
/// `WeightEntry::Dense`. `Storage::State`/`Storage::Computed` are left out for the contract to seed
/// itself.
fn split_consts_and_slots(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut packed_owners: HashMap<String, Arc<PackedPayload>> = HashMap::new();
    let mut dense_consts: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut slots = Vec::new();
    for &id in g.inputs() {
        let m = g.meta(id);
        match m.storage {
            Storage::Const => {
                let name = m.name.clone().expect("named const");
                match inputs.get(&id) {
                    Some(Value::Packed(component)) => {
                        let parsed = PackedSourceName::parse(&name)
                            .unwrap_or_else(|| panic!("{name}: not a packed carrier name"));
                        packed_owners
                            .entry(parsed.linear_id().to_string())
                            .or_insert_with(|| Arc::clone(component.owner()));
                    }
                    // An expert stack: its experts are the store entries `{stack id}.{expert}`, which
                    // `ConstWeights` views as one `ExpertStack`.
                    Some(Value::PackedStack(component)) => {
                        let parsed = PackedSourceName::parse(&name)
                            .unwrap_or_else(|| panic!("{name}: not a packed carrier name"));
                        let stack = component.stack();
                        for expert in 0..stack.experts() {
                            packed_owners
                                .entry(format!("{}.{expert}", parsed.linear_id()))
                                .or_insert_with(|| Arc::clone(stack.owner(expert)));
                        }
                    }
                    Some(Value::Host(t)) => {
                        let bytes: Vec<u8> = t
                            .as_f32()
                            .unwrap()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
                        dense_consts.push((name, t.shape().to_vec(), bytes));
                    }
                    other => panic!("{name}: unexpected const value {other:?}"),
                }
            }
            Storage::Slot(_) => {
                let t = inputs
                    .get(&id)
                    .and_then(Value::as_host)
                    .unwrap_or_else(|| panic!("missing slot input {id:?}"));
                let dtype = m.aval.dtype;
                let bytes: Vec<u8> = if dtype == DType::I32 {
                    t.as_i32()
                        .expect("I32-declared slot needs an I32 HostTensor")
                        .iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect()
                } else {
                    t.as_f32()
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                };
                slots.push(SlotBind {
                    key: m.slot_key().unwrap().clone(),
                    shape: t.shape().to_vec(),
                    dtype,
                    bytes,
                });
            }
            Storage::State | Storage::Computed(_) => {}
            Storage::Device => unreachable!(),
        }
    }
    let mut builder = WeightStore::builder();
    for (key, owner) in packed_owners {
        builder.insert(key, WeightEntry::Packed(owner)).unwrap();
    }
    for (name, shape, bytes) in dense_consts {
        let dense = DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    (builder.build(), slots)
}

/// A graph whose sole output IS one named state value, for reading a carried state buffer back
/// through a real step output (same pattern as `poot-llm/tests/wgpu_executor_contract.rs`'s `state_probe_graph` - no
/// `read_state` accessor exists).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph {
    let b = Builder::for_tests();
    let state = b.state_input(name, aval, StateRole::Recurrent).unwrap();
    b.finish(poot_graph_ir::Outputs::new(state).with_state(&[(state, state)]))
        .unwrap()
}

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
        // The lattice quants: a constant grid table and a sign field per block.
        WeightFormat::Iq2_Xs,
        WeightFormat::Iq3_Xxs,
        WeightFormat::Iq3_S,
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

struct PackedGraph {
    graph: Graph,
    x: ValueId,
    pos: Option<ValueId>,
}

/// `y = x @ W^T` over a packed `[OUT, K]` weight, `x` an `[m, K]` activation slot, carrying a cache
/// written by `DynamicUpdateSlice`: a `[STEPS, OUT]` cache at the `Pos` slot for decode (`m == 1`),
/// the whole `[K, OUT]` result for prefill (`m == K`).
fn packed_graph(descriptor: poot_quant::PackedWeight, m: usize) -> PackedGraph {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "onehot", TensorType::f32(vec![m, K]))
        .unwrap();
    let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
    if m == 1 {
        let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
        let cache = b
            .state_input(
                "decoded",
                TensorType::f32(vec![STEPS, OUT]),
                StateRole::Recurrent,
            )
            .unwrap();
        let cache_out = b.dynamic_update_slice_dyn(cache, y, pos, 0).unwrap();
        PackedGraph {
            graph: b
                .finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
                .unwrap(),
            x: x.id(),
            pos: Some(pos.id()),
        }
    } else {
        let cache = b
            .state_input(
                "decoded",
                TensorType::f32(vec![m, OUT]),
                StateRole::Recurrent,
            )
            .unwrap();
        let cache_out = b.dynamic_update_slice(cache, y, 0, 0).unwrap();
        PackedGraph {
            graph: b
                .finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
                .unwrap(),
            x: x.id(),
            pos: None,
        }
    }
}

fn assert_claims_packed_contraction(g: &Graph) {
    assert!(
        g.eqns()
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
        "compile claims the packed linear"
    );
}

/// The packed carriers of `graph`, bound as components of `owner` by the role their names carry.
fn carriers(graph: &Graph, owner: &Arc<PackedPayload>) -> HashMap<ValueId, Value> {
    graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let name = PackedSourceName::parse(graph.meta(id).name.as_deref()?)?;
            assert_eq!(graph.aval(id).dtype, DType::I8);
            Some((
                id,
                PackedComponentRef::new(Arc::clone(owner), name.role()).into(),
            ))
        })
        .collect()
}

fn one_hot(rows: usize, hot: impl Fn(usize) -> usize) -> HostTensor {
    let mut data = vec![0.0f32; rows * K];
    for r in 0..rows {
        data[r * K + hot(r)] = 1.0;
    }
    HostTensor::f32(vec![rows, K], data)
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

/// W9 decode: the contract's `add_entry`/`step` (one entry, held across all STEPS steps so its
/// zero-seeded "decoded" cache carries forward exactly as the old `state_in`/returned-buffers
/// hand-off did) over STEPS steps with carried state, bit-exact against the oracle decoder.
#[test]
fn packed_decode_through_resident_serving_path_is_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let packed = packed_graph(owner.weight(), 1);
        let weights = carriers(&packed.graph, &owner);
        let path = "step";
        let mut init_inputs = weights.clone();
        init_inputs.insert(packed.x, one_hot(1, |_| column(0)).into());
        init_inputs.insert(packed.pos.unwrap(), HostTensor::i32(vec![], vec![0]).into());
        let (store, _) = split_consts_and_slots(&packed.graph, &init_inputs);
        let (bound, map) = ConstWeights::bind(&packed.graph, &store);
        let program = staged(&bound, target);
        assert_claims_packed_contraction(program.stages().next().unwrap().2.graph());
        let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
        let entry = exec.add_entry(exe, &program).unwrap();
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = weights.clone();
            inputs.insert(packed.x, one_hot(1, |_| k).into());
            inputs.insert(
                packed.pos.unwrap(),
                HostTensor::i32(vec![], vec![step as i32]).into(),
            );
            let (_, slot_binds) = split_consts_and_slots(&packed.graph, &inputs);
            let step_in = step_inputs(&slot_binds);
            let y = f32_bytes(
                &exec
                    .step(exe, entry, &step_in, &mut NoSync)
                    .unwrap_or_else(|error| panic!("{format:?} {path} step {step}: {error}"))
                    .read()
                    .unwrap_or_else(|error| panic!("{format:?} {path} step {step}: {error}")),
            );
            assert_decode_row(&format!("{format:?} {path} step {step}"), &owner, &y, k);
        }

        let probe = state_probe_graph("decoded", TensorType::f32(vec![STEPS, OUT]));
        let probe_program = staged(&probe, target);
        let probe_entry = exec.add_entry(exe, &probe_program).unwrap();
        let cache = f32_bytes(
            &exec
                .step(exe, probe_entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        for step in 0..STEPS {
            assert_decode_row(
                &format!("{format:?} {path} carried row {step}"),
                &owner,
                &cache[step * OUT..(step + 1) * OUT],
                column(step),
            );
        }
        exec.remove_entry(exe, probe_entry).unwrap();
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
    }
}

/// W9 prefill: an identity activation decodes the whole weight through the contract's `add_entry`/
/// `step`, bit-exact per element, output and carried state.
///
/// Card 546b: the old test ran this twice, through the deleted `GpuExecutor::run_prefill` and
/// through the deleted `run_resident_kv` with an explicit zero seed - two call-sites into the SAME
/// deleted `run_resident_kv_prepared` body (the deleted `run_prefill` was only ever a zero-seed convenience wrapper around
/// it), so the "two paths" never actually exercised independent code. The contract has exactly one
/// path (state is zero-seeded automatically on first use, Card 546a); this runs it once per format and
/// keeps the real cross-check this test always relied on - bit-exact agreement with the independent
/// oracle decoder (`assert_decode_row`), for both the output and the carried state.
#[test]
fn packed_prefill_through_serving_paths_is_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 29,
        ));
        let packed = packed_graph(owner.weight(), K);
        let mut inputs = carriers(&packed.graph, &owner);
        inputs.insert(packed.x, one_hot(K, |r| r).into());
        let (store, slot_binds) = split_consts_and_slots(&packed.graph, &inputs);
        let (bound, map) = ConstWeights::bind(&packed.graph, &store);
        let program = staged(&bound, target);
        assert_claims_packed_contraction(program.stages().next().unwrap().2.graph());
        let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
        let entry = exec.add_entry(exe, &program).unwrap();
        let step_in = step_inputs(&slot_binds);
        let y = f32_bytes(
            &exec
                .step(exe, entry, &step_in, &mut NoSync)
                .unwrap_or_else(|error| panic!("{format:?} prefill: {error}"))
                .read()
                .unwrap_or_else(|error| panic!("{format:?} prefill: {error}")),
        );

        let probe = state_probe_graph("decoded", TensorType::f32(vec![K, OUT]));
        let probe_program = staged(&probe, target);
        let probe_entry = exec.add_entry(exe, &probe_program).unwrap();
        let cache = f32_bytes(
            &exec
                .step(exe, probe_entry, &StepInputs::new(), &mut NoSync)
                .unwrap()
                .read()
                .unwrap(),
        );
        for r in 0..K {
            let row = &y[r * OUT..(r + 1) * OUT];
            assert_decode_row(&format!("{format:?} prefill row {r}"), &owner, row, r);
            let carried = &cache[r * OUT..(r + 1) * OUT];
            assert_decode_row(
                &format!("{format:?} prefill carried {r}"),
                &owner,
                carried,
                r,
            );
        }
        exec.remove_entry(exe, probe_entry).unwrap();
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();
    }
}

// `a_swapped_packed_role_is_refused_before_upload` deleted (Card 546b, W11a): it tested
// the deleted `GpuExecutor::bind_resident`'s `check_packed_bindings` call, which refused a caller-supplied
// `Value::Packed` whose OWN `role` field disagreed with the role the graph's name declared
// for that `ValueId` - a per-value, caller-supplied role the old `HashMap<ValueId, Value>` bind API
// could misuse by construction (this test did so by swapping which component value two neighboring
// ids held). The contract's `WeightStore` binds one `WeightEntry::Packed` payload per LINEAR ID (never
// a per-role caller value) and `Engine::weight_buffer` derives each operand's role purely from the
// GRAPH's own declared name at that `ValueId` (`PackedSourceName::parse(name).role()`), so there is no
// longer a caller-supplied role to get wrong - W11a's invariant moved from a runtime refusal to a
// structural one (verified by reading `Engine::weight_buffer`, `crates/poot-executor/src/engine.rs`).

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

fn indexed_one_hot(rows: usize, k: usize) -> HostTensor {
    let mut data = vec![0.0f32; rows * k];
    for r in 0..rows {
        data[r * k + column_of(r, k)] = 1.0;
    }
    HostTensor::f32(vec![rows, k], data)
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
    exec: &mut dyn Executor,
    target: Target,
    format: WeightFormat,
    (rows, out, k): (usize, usize, usize),
    expert_of: fn(usize) -> usize,
    buffers: StackBuffers,
) {
    let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
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
        descriptor: owners[0].weight(),
        experts: EXPERTS,
    };
    let stack = Arc::new(PackedStack::try_new(owners.clone()).unwrap());
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
    let g = b.finish(y).unwrap();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in g.inputs() {
        let Some(name) = g.meta(id).name.as_deref().and_then(PackedSourceName::parse) else {
            continue;
        };
        inputs.insert(
            id,
            PackedStackComponentRef::new(Arc::clone(&stack), name.role()).into(),
        );
    }
    inputs.insert(x.id(), indexed_one_hot(rows, k).into());
    let selected: Vec<i32> = (0..rows).map(|m| expert_of(m) as i32).collect();
    inputs.insert(ids.id(), HostTensor::i32(vec![rows], selected).into());

    let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
    let (bound, map) = ConstWeights::bind(&g, &store);
    let bound = buffers.graph(&bound, target, table.descriptor);
    let program = staged(&bound, target);
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
        let [contraction] = compiled.eqns() else {
            panic!("{what}: one equation, got {:?}", compiled.eqns());
        };
        assert!(
            matches!(
                contraction.op,
                poot_graph_ir::OpKind::PackedContraction {
                    rows: poot_graph_ir::ContractionRows::Indexed { experts: EXPERTS },
                    ..
                }
            ),
            "{what}: {:?}",
            contraction.op
        );
    }
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let got = f32_bytes(
        &exec
            .step(exe, entry, &step_in, &mut NoSync)
            .unwrap_or_else(|error| panic!("{what}: {error}"))
            .read()
            .unwrap_or_else(|error| panic!("{what}: {error}")),
    );
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    let decoded: Vec<Vec<f32>> = owners.iter().map(|owner| decoded_rows(owner)).collect();
    for m in 0..rows {
        assert_indexed_row(
            &format!("{what} row {m} expert {}", expert_of(m)),
            &decoded[expert_of(m)],
            k,
            &got[m * out..(m + 1) * out],
            column_of(m, k),
        );
    }
}

/// SC-001 (wgpu): `packed_indexed_linear` over a stacked table of every format (Q4_K, Q8_0, AWQ, GPTQ
/// act-order, MXFP4) compiles to one `PackedContraction { rows: Indexed }` and runs through the executor
/// contract with one-hot rows bit-exact against the oracle decoder.
/// Mutation: slab stride 0 in the planner's metadata (every row reads expert 0); the rows routed to
/// experts 1 and 2 go red.
#[test]
fn packed_indexed_contraction_runs_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for format in expert_formats() {
        run_indexed_case(
            &mut *exec,
            target,
            format,
            (ROWS, OUT, K),
            interleaved,
            StackBuffers::One,
        );
    }
}

/// Card 1067 (wgpu): a stacked table over the target's single-buffer limit runs split into expert ranges
/// (0..2 and 2..3, each its own buffer per source) through the executor contract and is bit-exact against the
/// oracle decoder on every format, with rows routed to every expert, the last one alone in its buffer. The buffer
/// limit is set on the target's `DeviceCaps` (`StackBuffers::Split`), the one input of `legalize`'s split; the
/// graph, weights and executor are those of the unsplit case.
/// Mutation: in `legalize`'s `split_oversized_expert_stacks` shift the ids of the later pieces by one less
/// than `range.start`; the row routed to expert 2 reads expert 1's slab of the piece and goes red.
#[test]
fn a_split_expert_stack_runs_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for format in expert_formats() {
        run_indexed_case(
            &mut *exec,
            target,
            format,
            (ROWS, 2, K),
            interleaved,
            StackBuffers::Split,
        );
    }
}

/// SC-002 (wgpu): a Q8_0 table with `out = 3, K = 32` is 102 bytes per expert, not a whole number of words,
/// so the binder pads each expert to 104 bytes and the kernel strides by that: one-hot rows routed to
/// experts 1 and 2 (whose slabs start at byte 104 and 208, not 102 and 204) are bit-exact. Mutation: drop the
/// per-expert padding in `WeightWriter::read_stream`; experts 1 and 2 read misaligned blocks and go red.
#[test]
fn an_unaligned_expert_slab_table_runs_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    run_indexed_case(
        &mut *exec,
        target,
        WeightFormat::Q8_0,
        (ROWS, 3, 32),
        interleaved,
        StackBuffers::One,
    );
}

/// A prefill-sized routed table (wgpu): `PREFILL_ROWS` rows is past the decode row count, so the indexed packed
/// contraction runs the segmented Tiled schedule (rows routed by `gapped`, so a tile holds several
/// experts and may skip one) over every format, with `N = 11` leaving edge columns in the last column tile. One-hot rows bit-exact
/// against the oracle decoder. Mutation: take the Tiled step's slab from the wrong segment (`weight.step(.., lo,
/// ..)` in `indexed_tiled_body`); rows routed to an expert after a gap read another expert and go red.
#[test]
fn packed_indexed_prefill_runs_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for format in expert_formats() {
        run_indexed_case(
            &mut *exec,
            target,
            format,
            (PREFILL_ROWS, PREFILL_OUT, K),
            gapped,
            StackBuffers::One,
        );
    }
}

/// A routed prefill above the wgpu Gemv-body route's `DispatchGridLimit` (more than 65,535 x 256 outputs) (wgpu): `4096 x 4160` outputs of a Q8_0 table is 17.0M, past
/// 65,535 x 256, which the Gemv-body route refused; the Tiled schedule plans it (folded onto `GroupY` where the
/// device's X extent is smaller than the tile count) and every one of the 17M outputs is bit-exact against the
/// oracle decoder. Mutation: read the tile index from `GroupX` alone in `indexed_tiled_body`; every tile past
/// the first fold row is never computed and goes red.
#[test]
fn packed_indexed_prefill_above_the_gemv_grid_limit_runs_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    run_indexed_case(
        &mut *exec,
        target,
        WeightFormat::Q8_0,
        (GRID_LIMIT_ROWS, GRID_LIMIT_OUT, 64),
        gapped,
        StackBuffers::One,
    );
}

/// Card 545a: a quantized token-embedding lookup (`packed_embedding`, claimed by
/// `compile` as one `PackedRowGather`) through `run_resident`, for the decode token and a prefill id
/// vector, bit-exact against the oracle decoder's rows (a pure decode, so no zero-sign
/// canonicalization). Mutation: read row `ids[r] + 1` in the RowGather body; every row goes red.
#[test]
fn packed_embedding_row_gather_is_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
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
                .slot(Slot::Token, TensorType::new(shape.clone(), DType::I32))
                .unwrap();
            let rows = poot_graph_ir::ops::packed_embedding(
                &b,
                token,
                "model.embed_tokens",
                owner.weight(),
            )
            .unwrap();
            let g = b.finish(rows).unwrap();
            let mut inputs = carriers(&g, &owner);
            let token_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            inputs.insert(token.id(), HostTensor::i32(shape, token_ids).into());
            let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
            let (bound, map) = ConstWeights::bind(&g, &store);
            let program = staged(&bound, target);
            assert!(
                program
                    .stages()
                    .next()
                    .unwrap()
                    .2
                    .graph()
                    .eqns()
                    .iter()
                    .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
                "{what}: compile claims the embedding"
            );
            let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
            let entry = exec.add_entry(exe, &program).unwrap();
            let step_in = step_inputs(&slot_binds);
            let out = f32_bytes(
                &exec
                    .step(exe, entry, &step_in, &mut NoSync)
                    .unwrap_or_else(|error| panic!("{what}: {error}"))
                    .read()
                    .unwrap_or_else(|error| panic!("{what}: {error}")),
            );
            exec.remove_entry(exe, entry).unwrap();
            exec.unload(exe).unwrap();
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
        }
    }
}

/// Card 545a SC-004: a MoE expert stack traced DENSE - one `[E, out, K]` constant read through
/// `Transpose([0, 2, 1])` by `IndexedMatMul`, the storage-agnostic form every MoE tracer emits - is bound to
/// four Q6_K expert owners by `bind_packed_weights` (rewritten to one indexed packed contraction), planned by
/// `compile`, and run on wgpu; it matches the CPU oracle over the same traced graph with the decoded dense
/// stack at tier 2. Each row reads a different expert, so an owner bound to the wrong expert changes that
/// row. Mutation: swap two experts' keys in the store (`experts.weight.1` and `.2`) turns this red.
#[test]
fn a_dense_traced_q6_k_expert_stack_binds_and_matches_the_oracle() {
    use poot_graph_plan::{PackedConst, PackedLayout, WeightFormats, bind_packed_weights};

    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    const EXPERTS: usize = 4;
    const ROWS: usize = 6;
    let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| {
            Arc::new(poot_quant::test_support::packed::random_payload(
                WeightFormat::Q6_K,
                [OUT, K],
                e as u64 * 13 + 5,
            ))
        })
        .collect();
    let expert_of = |m: usize| (m * 3 + 1) % EXPERTS;

    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, K]))
        .unwrap();
    let ids = b
        .slot_named_ranged(
            Slot::Activation,
            "moe-ids",
            TensorType::new(vec![ROWS], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let w = b
        .constant("experts.weight", TensorType::f32(vec![EXPERTS, OUT, K]))
        .unwrap();
    let y = b
        .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), ids)
        .unwrap();
    let traced = b.finish(y).unwrap();

    let mut formats = WeightFormats::default();
    formats
        .insert(
            "experts.weight".to_string(),
            PackedConst {
                linear_id: "experts.weight".to_string(),
                weight: owners[0].weight(),
                layout: PackedLayout::ExpertRows { experts: EXPERTS },
            },
        )
        .unwrap();
    let bound = bind_packed_weights(&traced, &formats).expect("bind the expert stack");
    let graph = &bound;

    let activation = HostTensor::f32(
        vec![ROWS, K],
        (0..ROWS * K)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<f32>>(),
    );
    let selected = HostTensor::i32(
        vec![ROWS],
        (0..ROWS).map(|m| expert_of(m) as i32).collect::<Vec<i32>>(),
    );
    let stack = Arc::new(PackedStack::try_new(owners.clone()).unwrap());
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        match meta.name.as_deref() {
            Some("activation.moe-x") => {
                inputs.insert(id, activation.clone().into());
            }
            Some("activation.moe-ids") => {
                inputs.insert(id, selected.clone().into());
            }
            Some(name) => {
                let name = PackedSourceName::parse(name)
                    .unwrap_or_else(|| panic!("input v{id} {name:?} is not a packed carrier"));
                assert_eq!(name.linear_id(), "experts.weight");
                inputs.insert(
                    id,
                    PackedStackComponentRef::new(Arc::clone(&stack), name.role()).into(),
                );
            }
            None => panic!("unnamed input v{id}"),
        }
    }
    let (store, slot_binds) = split_consts_and_slots(graph, &inputs);
    let (named, map) = ConstWeights::bind(graph, &store);
    let program = staged(&named, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let got = f32_bytes(
        &exec
            .step(exe, entry, &step_in, &mut NoSync)
            .expect("wgpu step")
            .read()
            .expect("wgpu read"),
    );
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();

    // The oracle: the traced (dense) graph over the decoded stack `W[e, o, k] = decode(e)[o, k]`.
    let mut decoded = vec![0.0f32; EXPERTS * OUT * K];
    for (e, owner) in owners.iter().enumerate() {
        for o in 0..OUT {
            let at = (e * OUT + o) * K;
            owner.decode_row(o, &mut decoded[at..at + K]).unwrap();
        }
    }
    let mut dense: HashMap<ValueId, Value> = HashMap::new();
    for &id in traced.inputs() {
        let value = match traced.meta(id).name.as_deref() {
            Some("activation.moe-x") => activation.clone(),
            Some("activation.moe-ids") => selected.clone(),
            Some("experts.weight") => HostTensor::f32(vec![EXPERTS, OUT, K], decoded.clone()),
            other => panic!("unexpected traced input {other:?}"),
        };
        dense.insert(id, value.into());
    }
    let oracle = poot_eval::eval(
        &traced,
        &dense,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .expect("CPU oracle")
    .output
    .into_host()
    .expect("dense oracle");
    assert_eq!(got.len(), oracle.as_f32().unwrap().len());
    for (i, (got, want)) in got.iter().zip(oracle.as_f32().unwrap().iter()).enumerate() {
        let (m, o) = (i / OUT, i % OUT);
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "row {m} (expert {}) col {o}: device {got} vs oracle {want}",
            expert_of(m)
        );
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
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = poot_executor::Engine::new(device);
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
        let entry = engine.add_entry(exe, &staged(&named, target)).unwrap();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let shape = [ids.len()];
        let mut inputs = StepInputs::new();
        inputs.push(
            idx_key,
            &shape,
            HostView::new(DType::I32, ids.len(), &id_bytes).unwrap(),
        );
        let bytes = engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .read()
            .unwrap();
        let out = f32_bytes(&bytes);
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

/// `y = x @ W^T` over a packed `[OUT, K]` weight with a dense `[m, K]` activation (every product contributes),
/// run on `exec`, as one step reading the output.
fn run_dense_activation(
    exec: &mut dyn Executor,
    target: Target,
    graph: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Vec<f32> {
    let (store, slot_binds) = split_consts_and_slots(graph, inputs);
    let (bound, map) = ConstWeights::bind(graph, &store);
    let program = staged(&bound, target);
    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();
    let step_in = step_inputs(&slot_binds);
    let out = f32_bytes(
        &exec
            .step(exe, entry, &step_in, &mut NoSync)
            .unwrap()
            .read()
            .unwrap(),
    );
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();
    out
}

/// SC-001: Q4_K, Q8_0 and FP8 contractions through the packed loaders meet their tier-2 oracle contract. A
/// dense activation makes every output a sum over every `K` block, so a dropped block or a wrong scale moves
/// every element; `m == 1` takes the Gemv schedule and `m == 17` the Tiled schedule (ragged against its tile),
/// and every output element is compared. Mutation: omit one block's contribution in `DecodePlan`'s address or
/// scale a factor by a nonzero constant; every row goes red.
#[test]
fn packed_loaders_match_the_oracle_on_every_output_on_wgpu() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
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
            let mut inputs = carriers(&graph, &owner);
            inputs.insert(
                x.id(),
                HostTensor::f32(vec![m, K], fill(7 + m as u64, m * K)).into(),
            );
            let got = run_dense_activation(&mut *exec, target, &graph, &inputs);
            let oracle_inputs = inputs.clone();
            let want = poot_eval::eval(
                &graph,
                &oracle_inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .unwrap()
            .output
            .into_host()
            .unwrap();
            assert_eq!(got.len(), want.shape().iter().product::<usize>(), "{what}");
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
/// with the add), the fused packed Gemv equals the standalone one's head plus the bias on the host, bit for
/// bit, so the head sum is the same computation in both bodies. Every loader format, a generic and a cancelling bias, at
/// tolerance 0.0. A format whose decode carries `BinaryOpNoContract` (Q4_K's min term) loses that
/// protection if the no-contraction marker lands on a different op in the longer fused body, and its fused
/// head then drifts by an ulp or two. Mutation: place the SPIR-V `NoContraction` decoration by instruction
/// ordinal, as card 628 first did; the Q4_K row goes red on wgpu while the other formats stay green.
#[test]
fn a_packed_gemv_head_is_shared_with_the_standalone_gemv_on_wgpu() {
    use poot_executor_parity::assert_close;
    use poot_graph_ir::op::{BinOp, ContractionHead};
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in loader_formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
            format,
            [OUT, K],
            seed as u64 + 51,
        ));
        let what = format!("{format:?}");
        let activation = HostTensor::f32(
            vec![1, K],
            fill(3, K).into_iter().map(|v| v * 0.02).collect(),
        );
        let build = |with_bias: bool| {
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]))
                .unwrap();
            let head =
                poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None)
                    .unwrap();
            if !with_bias {
                return (b.finish(head).unwrap(), x.id(), None);
            }
            let bias = b.constant("bias", TensorType::f32(vec![OUT])).unwrap();
            let out = b.binary(BinOp::Add, head, bias).unwrap();
            (b.finish(out).unwrap(), x.id(), Some(bias.id()))
        };

        let (head_graph, head_x, _) = build(false);
        let mut head_inputs = carriers(&head_graph, &owner);
        head_inputs.insert(head_x, activation.clone().into());
        let head = run_dense_activation(&mut *exec, target, &head_graph, &head_inputs);

        let (graph, x, bias) = build(true);
        let bias = bias.unwrap();
        let fused_program = staged(&graph, target);
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
            let mut inputs = carriers(&graph, &owner);
            inputs.insert(x, activation.clone().into());
            inputs.insert(bias, HostTensor::f32(vec![OUT], bias_values.clone()).into());
            let fused = run_dense_activation(&mut *exec, target, &graph, &inputs);
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
/// forms one `ContractionEpilogue { head: Packed }`, and the fused device program agrees with the oracle (the
/// head summed on the device). The head itself is the standalone Gemv's (the `head_is_shared` row); the
/// epilogue's `bias` add and `0.5` multiply may contract to an FMA (`contraction/epilogue.rs` deliberately does
/// not forbid it), so the comparison is the tier-2 tolerance, and is sharpened by a cancellation fixture: `bias` is the negated device head, so every pre-activation is within a few ulps of zero
/// when the epilogue sees each column's completed sum, and the fused result is `residual` to a few ulps of the
/// head. An epilogue applied to a lane's partial accumulator (before the LDS reduction) adds the bias once per
/// lane and saturates the activation, moving the result by about half the lane count. Mutation: apply the
/// epilogue to the per-lane partial before the LDS reduction in `gemv_body`; the cancellation row and the
/// oracle row go red.
#[test]
fn a_packed_gemv_epilogue_agrees_with_the_oracle_on_wgpu() {
    use poot_executor_parity::assert_close;
    use poot_graph_ir::op::{BinOp, ContractionHead, UnOp};
    use poot_graph_ir::types::Scalar;
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in loader_formats().into_iter().enumerate() {
        let owner = Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
            format,
            [OUT, K],
            seed as u64 + 51,
        ));
        let what = format!("{format:?}");
        // Small activations keep the head sums O(1), so the activation does not saturate on the generic bias.
        let activation = HostTensor::f32(
            vec![1, K],
            fill(3, K).into_iter().map(|v| v * 0.02).collect(),
        );
        let build = |with_chain: bool| {
            let b = Builder::for_tests();
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]))
                .unwrap();
            let head =
                poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None)
                    .unwrap();
            if !with_chain {
                return (b.finish(head).unwrap(), x.id(), None);
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
            (
                b.finish(out).unwrap(),
                x.id(),
                Some((bias.id(), residual.id())),
            )
        };

        // The device head alone, to cancel against.
        let (head_graph, head_x, _) = build(false);
        let mut head_inputs = carriers(&head_graph, &owner);
        head_inputs.insert(head_x, activation.clone().into());
        let head = run_dense_activation(&mut *exec, target, &head_graph, &head_inputs);
        let head_scale = head.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        let (graph, x, leaves) = build(true);
        let (bias, residual) = leaves.unwrap();
        let residual_values = fill(5, OUT);
        let fused_program = staged(&graph, target);
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
            let mut inputs = carriers(&graph, &owner);
            inputs.insert(x, activation.clone().into());
            inputs.insert(bias, HostTensor::f32(vec![OUT], bias_values).into());
            inputs.insert(
                residual,
                HostTensor::f32(vec![1, OUT], residual_values.clone()).into(),
            );
            let fused = run_dense_activation(&mut *exec, target, &graph, &inputs);
            let oracle = poot_eval::eval(
                &graph,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .unwrap()
            .output
            .into_host()
            .unwrap();
            if label == "generic bias" {
                assert_close(
                    &fused,
                    oracle.as_f32().unwrap(),
                    poot_executor_parity::TIER2,
                    &format!("{what}: fused vs the oracle"),
                );
            } else {
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
