//! Card 642: packed decode through the wgpu SERVING executor paths.
//!
//! A `packed_linear` graph (compiled, so `compile` claims it as one `PackedContraction`) runs through
//! `run_resident_kv` (eager resident decode with carried state) and `run_prefill`/`run_resident_prefill`
//! (zero-seeded prefill), with the weight bound as `Value::Packed` carriers - never a dense f32
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

use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_executor::{Device, Executor, HostView, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::{
    Builder, Graph, PackedSourceName, Slot, SlotKey, StateRole, Storage, TensorType, ValueId,
};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_quant::{PackedComponentRef, PackedPayload};
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;

/// Compile `g` for `target` with `Submission::Replay` (Card 546b: the contract admits no other
/// submission).
fn staged(g: &Graph, target: Target) -> StagedProgram<poot_graph_ir::ValidationOutputs> {
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
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile the packed graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
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
/// caller-supplied role the way `GpuExecutor::bind_resident` did. `Value::Host` tensors become a
/// `WeightEntry::Dense`. `Storage::State`/`Storage::Computed` are left out for the contract to seed
/// itself.
fn split_consts_and_slots(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (WeightStore, Vec<SlotBind>) {
    let mut packed_owners: HashMap<String, Arc<PackedPayload>> = HashMap::new();
    let mut dense_consts: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut slots = Vec::new();
    for &id in &g.inputs {
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
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    (builder.build(), slots)
}

/// A graph whose sole output IS one named state value, for reading a carried state buffer back
/// through a real step output (same pattern as `executor_contract.rs`'s `state_probe_graph` - no
/// `read_state` accessor exists).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph {
    let b = Builder::new();
    let state = b.state_input(name, aval, StateRole::Recurrent);
    b.finish_with_state(state, &[(state, state)])
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
    ]
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
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "onehot", TensorType::f32(vec![m, K]));
    let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
    if m == 1 {
        let pos = b.slot(Slot::Pos, TensorType::f32(vec![]));
        let cache = b.state_input(
            "decoded",
            TensorType::f32(vec![STEPS, OUT]),
            StateRole::Recurrent,
        );
        let cache_out = b.dynamic_update_slice_dyn(cache, y, pos, 0);
        PackedGraph {
            graph: b.finish_with_state(y, &[(cache, cache_out)]),
            x: x.id,
            pos: Some(pos.id),
        }
    } else {
        let cache = b.state_input(
            "decoded",
            TensorType::f32(vec![m, OUT]),
            StateRole::Recurrent,
        );
        let cache_out = b.dynamic_update_slice(cache, y, 0, 0);
        PackedGraph {
            graph: b.finish_with_state(y, &[(cache, cache_out)]),
            x: x.id,
            pos: None,
        }
    }
}

fn assert_claims_packed_contraction<V: poot_graph_ir::ValidationChannel>(g: &Graph<V>) {
    assert!(
        g.eqns
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
        "compile claims the packed linear"
    );
}

/// The packed carriers of `graph`, bound as components of `owner` by the role their names carry.
fn carriers(graph: &Graph, owner: &Arc<PackedPayload>) -> HashMap<ValueId, Value> {
    graph
        .inputs
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
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let packed = packed_graph(owner.weight(), 1);
        let program = staged(&packed.graph, target);
        assert_claims_packed_contraction(program.stages().next().unwrap().2.graph());
        let weights = carriers(&packed.graph, &owner);
        let path = "step";
        let mut init_inputs = weights.clone();
        init_inputs.insert(packed.x, one_hot(1, |_| column(0)).into());
        init_inputs.insert(packed.pos.unwrap(), HostTensor::scalar(0.0).into());
        let (store, _) = split_consts_and_slots(&packed.graph, &init_inputs);
        let exe = exec
            .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
            .unwrap();
        let entry = exec.add_entry(exe, &program).unwrap();
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = weights.clone();
            inputs.insert(packed.x, one_hot(1, |_| k).into());
            inputs.insert(packed.pos.unwrap(), HostTensor::scalar(step as f32).into());
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
/// Card 546b: the old test ran this twice, through `GpuExecutor::run_prefill` and
/// through `run_resident_kv` with an explicit zero seed - two call-sites into the SAME
/// `run_resident_kv_prepared` body (`run_prefill` was only ever a zero-seed convenience wrapper around
/// it), so the "two paths" never actually exercised independent code. The contract has exactly one
/// path (state is zero-seeded automatically on first use, Card 546a); this runs it once per format and
/// keeps the real cross-check this test always relied on - bit-exact agreement with the independent
/// oracle decoder (`assert_decode_row`), for both the output and the carried state.
#[test]
fn packed_prefill_through_serving_paths_is_bit_exact() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 29,
        ));
        let packed = packed_graph(owner.weight(), K);
        let program = staged(&packed.graph, target);
        assert_claims_packed_contraction(program.stages().next().unwrap().2.graph());
        let mut inputs = carriers(&packed.graph, &owner);
        inputs.insert(packed.x, one_hot(K, |r| r).into());
        let (store, slot_binds) = split_consts_and_slots(&packed.graph, &inputs);
        let exe = exec
            .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
            .unwrap();
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
// `GpuExecutor::bind_resident`'s `check_packed_bindings` call, which refused a caller-supplied
// `Value::Packed` whose OWN `role` field disagreed with the role the graph's name declared
// for that `ValueId` - a per-value, caller-supplied role the old `HashMap<ValueId, Value>` bind API
// could misuse by construction (this test did so by swapping which component value two neighboring
// ids held). The contract's `WeightStore` binds one `WeightEntry::Packed` payload per LINEAR ID (never
// a per-role caller value) and `Engine::weight_buffer` derives each operand's role purely from the
// GRAPH's own declared name at that `ValueId` (`PackedSourceName::parse(name).role()`), so there is no
// longer a caller-supplied role to get wrong - W11a's invariant moved from a runtime refusal to a
// structural one (verified by reading `Engine::weight_buffer`, `crates/poot-executor/src/engine.rs`).

/// Card 642: the canonical packed MoE chain (`packed_indexed_linear` and
/// `packed_grouped_linear`, every expert a `PackedDequant` planned as `Materialize`, the table the
/// dense `IndexedMatMul`) compiles and runs through `run_resident`, row `m` reading expert `ids[m]`:
/// with a one-hot row the output is exactly that expert's decoded column, bit-exact against the
/// oracle decoder (zero sign canonicalized).
#[test]
fn canonical_packed_moe_chain_runs_bit_exact() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    const EXPERTS: usize = 3;
    const ROWS: usize = 5;
    for format in formats() {
        let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
            .map(|e| {
                Arc::new(poot_test_util::packed::random_payload(
                    format,
                    [OUT, K],
                    e as u64 * 7 + 101,
                ))
            })
            .collect();
        let rows: Vec<poot_graph_ir::ops::PackedLinearGraphRow> = (0..EXPERTS)
            .map(|ordinal| poot_graph_ir::ops::PackedLinearGraphRow {
                ordinal,
                linear_id: format!("expert.{ordinal}"),
                descriptor: owners[ordinal].weight(),
            })
            .collect();
        let expert_of = |m: usize| (m * 2 + 1) % EXPERTS;
        for grouped in [false, true] {
            let what = format!("{format:?} grouped={grouped}");
            let b = Builder::new();
            let x = b.slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, K]));
            let ids = b.slot_named(Slot::Activation, "moe-ids", TensorType::f32(vec![ROWS]));
            let y = if grouped {
                poot_graph_ir::ops::packed_grouped_linear(&b, x, ids, &rows)
            } else {
                poot_graph_ir::ops::packed_indexed_linear(&b, x, ids, &rows)
            }
            .unwrap();
            let g = b.finish(y);
            let mut inputs: HashMap<ValueId, Value> = HashMap::new();
            for &id in &g.inputs {
                let Some(name) = g.meta(id).name.as_deref().and_then(PackedSourceName::parse)
                else {
                    continue;
                };
                let expert: usize = name
                    .linear_id()
                    .strip_prefix("expert.")
                    .unwrap()
                    .parse()
                    .unwrap();
                inputs.insert(
                    id,
                    PackedComponentRef::new(Arc::clone(&owners[expert]), name.role()).into(),
                );
            }
            inputs.insert(x.id, one_hot(ROWS, column).into());
            let selected: Vec<f32> = (0..ROWS).map(|m| expert_of(m) as f32).collect();
            inputs.insert(ids.id, HostTensor::f32(vec![ROWS], selected).into());

            let program = staged(&g, target);
            let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
            let exe = exec
                .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
                .unwrap();
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
            for m in 0..ROWS {
                assert_decode_row(
                    &format!("{what} row {m} expert {}", expert_of(m)),
                    &owners[expert_of(m)],
                    &out[m * OUT..(m + 1) * OUT],
                    column(m),
                );
            }
        }
    }
}

/// Card 545a: a quantized token-embedding lookup (`packed_embedding`, claimed by
/// `compile` as one `PackedRowGather`) through `run_resident`, for the decode token and a prefill id
/// vector, bit-exact against the oracle decoder's rows (a pure decode, so no zero-sign
/// canonicalization). Mutation: read row `ids[r] + 1` in the RowGather body; every row goes red.
#[test]
fn packed_embedding_row_gather_is_bit_exact() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
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
        let owner = Arc::new(poot_test_util::packed::random_payload(
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
            let b = Builder::new();
            let token = b.slot(Slot::Token, TensorType::f32(shape.clone()));
            let rows = poot_graph_ir::ops::packed_embedding(
                &b,
                token,
                "model.embed_tokens",
                owner.weight(),
            )
            .unwrap();
            let g = b.finish(rows);
            let program = staged(&g, target);
            assert!(
                program
                    .stages()
                    .next()
                    .unwrap()
                    .2
                    .graph()
                    .eqns
                    .iter()
                    .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
                "{what}: compile claims the embedding"
            );
            let mut inputs = carriers(&g, &owner);
            let token_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
            inputs.insert(token.id, HostTensor::f32(shape, token_ids).into());
            let (store, slot_binds) = split_consts_and_slots(&g, &inputs);
            let exe = exec
                .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
                .unwrap();
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

/// Card 545a SC-004: a MoE expert stack traced DENSE - one `[E, K, out]`
/// constant read by `IndexedMatMul`, the storage-agnostic form every MoE tracer emits - is bound to
/// four Q6_K expert owners by `bind_packed_weights`, claimed and planned by `compile`, and run on
/// wgpu; it matches the CPU oracle over the same traced graph with the decoded dense stack at tier 2.
/// Each row reads a different expert, so an owner bound to the wrong expert changes that row.
/// Mutation (recorded, never left in the tree): swapping two experts' owners (`linear_ids` order
/// `[0, 2, 1, 3]`) in the `WeightFormats` placement turns this red: "row 0 (expert 1) col 0: device
/// -24.379936 vs oracle 7.571762".
#[test]
fn a_dense_traced_q6_k_expert_stack_binds_and_matches_the_oracle() {
    use poot_graph_plan::{PackedConst, PackedLayout, WeightFormats, bind_packed_weights};

    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    const EXPERTS: usize = 4;
    const ROWS: usize = 6;
    let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| {
            Arc::new(poot_test_util::packed::random_payload(
                WeightFormat::Q6_K,
                [OUT, K],
                e as u64 * 13 + 5,
            ))
        })
        .collect();
    let expert_of = |m: usize| (m * 3 + 1) % EXPERTS;

    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, K]));
    let ids = b.slot_named(Slot::Activation, "moe-ids", TensorType::f32(vec![ROWS]));
    let w = b.constant("experts.weight", TensorType::f32(vec![EXPERTS, K, OUT]));
    let y = b.indexed_matmul(x, w, ids);
    let traced = b.finish(y);

    let mut formats = WeightFormats::default();
    formats
        .insert(
            "experts.weight".to_string(),
            PackedConst {
                linear_ids: (0..EXPERTS).map(|e| format!("experts.{e}")).collect(),
                weight: owners[0].weight(),
                layout: PackedLayout::StackedColumns,
            },
        )
        .unwrap();
    let bound = bind_packed_weights(&traced, &formats).expect("bind the expert stack");
    let graph = &bound;
    let program = staged(&bound, target);

    let activation = HostTensor::f32(
        vec![ROWS, K],
        (0..ROWS * K)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<f32>>(),
    );
    let selected = HostTensor::f32(
        vec![ROWS],
        (0..ROWS).map(|m| expert_of(m) as f32).collect::<Vec<f32>>(),
    );
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &graph.inputs {
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
                let expert: usize = name
                    .linear_id()
                    .strip_prefix("experts.")
                    .unwrap()
                    .parse()
                    .unwrap();
                inputs.insert(
                    id,
                    PackedComponentRef::new(Arc::clone(&owners[expert]), name.role()).into(),
                );
            }
            None => panic!("unnamed input v{id}"),
        }
    }
    let (store, slot_binds) = split_consts_and_slots(graph, &inputs);
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
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

    // The oracle: the traced (dense) graph over the decoded stack `W[e, k, o] = decode(e)[o, k]`.
    let mut stack = vec![0.0f32; EXPERTS * K * OUT];
    let mut row = vec![0.0f32; K];
    for (e, owner) in owners.iter().enumerate() {
        for o in 0..OUT {
            owner.decode_row(o, &mut row).unwrap();
            for (k, value) in row.iter().enumerate() {
                stack[(e * K + k) * OUT + o] = *value;
            }
        }
    }
    let mut dense: HashMap<ValueId, Value> = HashMap::new();
    for &id in &traced.inputs {
        let value = match traced.meta(id).name.as_deref() {
            Some("activation.moe-x") => activation.clone(),
            Some("activation.moe-ids") => selected.clone(),
            Some("experts.weight") => HostTensor::f32(vec![EXPERTS, K, OUT], stack.clone()),
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
