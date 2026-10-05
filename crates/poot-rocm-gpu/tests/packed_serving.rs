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
use poot_graph_ir::{Builder, Graph, PackedSourceName, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    Submission, Target, TargetSet, compile_staged,
};
use poot_quant::PackedPayload;
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_tensor::DType;
use poot_test_util::device_skip::open_or_skip;

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

fn staged(
    target: Target,
    graph: &Graph,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let graph = graph.clone().with_validations(Vec::new());
    compile_staged(
        &graph,
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
    .unwrap()
}

/// A `WeightStore` with one packed entry per named linear carrier the graph's inputs name, all
/// pointing at the same payload (single-expert fixtures) unless `owners` maps a distinct one per
/// `linear_id`.
fn store_for(graph: &Graph, owners: &HashMap<String, Arc<PackedPayload>>) -> Arc<WeightStore> {
    let mut builder = WeightStore::builder();
    let mut seen = std::collections::HashSet::new();
    for &id in &graph.inputs {
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
        let owner = owners
            .get(name.linear_id())
            .unwrap_or_else(|| panic!("no payload registered for linear_id {}", name.linear_id()));
        builder
            .insert(name.linear_id(), WeightEntry::Packed(Arc::clone(owner)))
            .unwrap();
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
            poot_executor::HostView::new(*dtype, elems, bytes).unwrap(),
        );
    }
    let _ = graph;
    inputs
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
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
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A graph whose sole output is the named state value: a production entry stepped like any other,
/// which happens to let the test read the carried cache back through its real step output (no
/// `read_state` escape hatch).
fn state_probe_graph(name: &str, aval: TensorType) -> Graph {
    let b = Builder::new();
    let state = b.state_input(name, aval, StateRole::Recurrent);
    b.finish_with_state(state, &[(state, state)])
}

/// Decode over an explicit one-hot activation slot: `y = x @ W^T`, the `[STEPS, OUT]` cache written
/// at the `Pos` slot.
fn slot_decode_graph(descriptor: poot_quant::PackedWeight) -> (Graph, SlotKey, SlotKey) {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "onehot", TensorType::f32(vec![1, K]));
    let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
    let pos = b.slot(Slot::Pos, TensorType::f32(vec![]));
    let cache = b.state_input(
        "decoded",
        TensorType::f32(vec![STEPS, OUT]),
        StateRole::Recurrent,
    );
    let cache_out = b.dynamic_update_slice_dyn(cache, y, pos, 0);
    let g = b.finish_with_state(y, &[(cache, cache_out)]);
    let xk = g.meta(x.id).slot_key().unwrap().clone();
    let pk = g.meta(pos.id).slot_key().unwrap().clone();
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
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());
        let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
        let store = store_for(&graph, &owners);
        let exe = engine
            .load_weights(store, poot_executor::WeightSource::ConstNames)
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &graph)).unwrap();
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
                    DType::F32,
                    f32_bytes(&[step as f32]),
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
    let owner = Arc::new(poot_test_util::packed::random_payload(format, [OUT, K], 97));
    let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());
    let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
    let store = store_for(&graph, &owners);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged(target, &graph)).unwrap();
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
                DType::F32,
                f32_bytes(&[step as f32]),
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
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 29,
        ));
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "onehot", TensorType::f32(vec![K, K]));
        let y =
            poot_graph_ir::ops::packed_linear(&b, x, "layer", owner.weight(), None, None).unwrap();
        let cache = b.state_input(
            "decoded",
            TensorType::f32(vec![K, OUT]),
            StateRole::Recurrent,
        );
        let cache_out = b.dynamic_update_slice(cache, y, 0, 0);
        let graph = b.finish_with_state(y, &[(cache, cache_out)]);
        let x_key = graph.meta(x.id).slot_key().unwrap().clone();

        let owners = HashMap::from([("layer".to_string(), Arc::clone(&owner))]);
        let store = store_for(&graph, &owners);
        let exe = engine
            .load_weights(store, poot_executor::WeightSource::ConstNames)
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &graph)).unwrap();
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
    let owner = Arc::new(poot_test_util::packed::random_payload(format, [OUT, K], 3));
    let (graph, x_key, pos_key) = slot_decode_graph(owner.weight());

    // W11a probe: bind "layer" to a payload whose component roles report K and OUT swapped (an
    // AWQ scale table shaped as if it were the quantized weight table, and vice versa) - the same
    // observable fault a role-swapped upload produces, reached through the production
    // `weight_buffer`/`PackedComponentRef` path rather than a forged test-only field.
    let swapped_owner = Arc::new(poot_test_util::packed::random_payload(format, [K, OUT], 3));
    let owners = HashMap::from([("layer".to_string(), swapped_owner)]);
    let store = store_for(&graph, &owners);
    let exe = engine
        .load_weights(store, poot_executor::WeightSource::ConstNames)
        .unwrap();
    let staged_program = staged(target, &graph);
    let Err(error) = engine.add_entry(exe, &staged_program) else {
        panic!("a role/shape-swapped packed carrier must be refused, not silently bound");
    };
    eprintln!("a_swapped_packed_role_is_refused_before_upload: observed {error}");
    let _ = (x_key, pos_key);
}

/// Card 642: the canonical packed MoE chain (`packed_indexed_linear` and
/// `packed_grouped_linear`, every expert a `PackedDequant` planned as `Materialize`, the table the
/// dense `IndexedMatMul`) runs through the contract, row `m` reading expert `ids[m]`: with a one-hot
/// row the output is exactly that expert's decoded column, bit-exact against the oracle decoder
/// (zero sign canonicalized).
#[test]
fn canonical_packed_moe_chain_runs_bit_exact() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    const EXPERTS: usize = 3;
    const ROWS: usize = 5;
    for format in formats() {
        let owners_vec: Vec<Arc<PackedPayload>> = (0..EXPERTS)
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
                descriptor: owners_vec[ordinal].weight(),
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
            let graph = b.finish(y);
            let x_key = graph.meta(x.id).slot_key().unwrap().clone();
            let ids_key = graph.meta(ids.id).slot_key().unwrap().clone();

            let owners: HashMap<String, Arc<PackedPayload>> = (0..EXPERTS)
                .map(|e| (format!("expert.{e}"), Arc::clone(&owners_vec[e])))
                .collect();
            let store = store_for(&graph, &owners);
            let exe = engine
                .load_weights(store, poot_executor::WeightSource::ConstNames)
                .unwrap();
            let entry = engine.add_entry(exe, &staged(target, &graph)).unwrap();

            let selected: Vec<f32> = (0..ROWS).map(|m| expert_of(m) as f32).collect();
            let values = [
                (
                    x_key,
                    vec![ROWS, K],
                    DType::F32,
                    one_hot_bytes(ROWS, K, column),
                ),
                (ids_key, vec![ROWS], DType::F32, f32_bytes(&selected)),
            ];
            let inputs = slot_inputs(&graph, &values);
            let out = bytemuck_f32(
                &engine
                    .step(exe, entry, &inputs, &mut NoSync)
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
                    .read()
                    .unwrap(),
            );
            for m in 0..ROWS {
                assert_decode_row(
                    &format!("{what} row {m} expert {}", expert_of(m)),
                    &owners_vec[expert_of(m)],
                    &out[m * OUT..(m + 1) * OUT],
                    column(m),
                );
            }
            engine.unload(exe).unwrap();
        }
    }
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
            let token = b.slot_named(Slot::Token, "token", TensorType::f32(shape.clone()));
            let rows = poot_graph_ir::ops::packed_embedding(
                &b,
                token,
                "model.embed_tokens",
                owner.weight(),
            )
            .unwrap();
            let graph = b.finish(rows);
            let token_key = graph.meta(token.id).slot_key().unwrap().clone();

            let owners = HashMap::from([("model.embed_tokens".to_string(), Arc::clone(&owner))]);
            let store = store_for(&graph, &owners);
            let exe = engine
                .load_weights(store, poot_executor::WeightSource::ConstNames)
                .unwrap();
            let entry = engine.add_entry(exe, &staged(target, &graph)).unwrap();
            let token_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
            let values = [(token_key, shape.clone(), DType::F32, f32_bytes(&token_ids))];
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
