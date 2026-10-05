//! Card 642: packed decode through the PTX SERVING executor paths (run on
//! the M4 pod: this box has no NVIDIA GPU).
//!
//! A `packed_linear` graph (compiled, so `compile` claims it as one `PackedContraction`) runs through
//! the executor contract - a one-shot entry for `run_resident`/`run_prefill`-shaped cases, a
//! [`common::DecodeEntry`] stepped per token for the capture/replay-shaped cases - with the weight
//! bound as `Value::Packed` carriers - never a dense f32 copy.
//!
//! The activation is one-hot, so every output is exactly one decoded weight value (`y[o] =
//! decode(o, k)`); every other product is a signed zero. Compared bit for bit with the 541 oracle's
//! decoder (`PackedPayload::decode_row`) after canonicalizing the sign of zero (`+ 0.0`), which a sum of
//! zero-valued partials does not preserve. The carried `[steps, out]` cache (written by
//! `DynamicUpdateSlice` at the `Pos` slot) is compared the same way after the last step.
//!
//! Card 549: `a_swapped_packed_role_is_refused_before_upload_on_ptx` (W11a) is not ported. The pre-549
//! `PtxGraphExecutor::bind` matched each bound `Value::Packed`'s own role tag against what the
//! graph's const name expected, so a caller could swap two components between value-ids and trigger a
//! role mismatch. The executor contract's packed binding (`Engine::weight_buffer`, Z8/S46-4) is
//! name-driven instead: the graph's own const name carries the expected `(linear_id, role)`, and the
//! engine reads that role straight from the `WeightStore`'s one owner payload for that `linear_id` -
//! there is no caller-supplied "this value's role" to swap any more, so this bug class cannot occur
//! under the new binding.

use poot_tensor::DType;
use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_graph_ir::{
    BinOp, Builder, Graph, PackedSourceName, RedOp, Slot, StateRole, TensorType, ValueId,
};
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_ptx_gpu::PtxDevice;
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::{PackedComponentRef, PackedPayload};
use poot_runtime_common::DeviceBackend;
use poot_target::Backend;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;

mod common;

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

fn target(ptx: &PtxDevice) -> Target {
    Target {
        backend: Backend::Nvptx,
        caps: poot_executor::Device::target(ptx).caps,
    }
}

/// Only for the structural assertion (checking the compiled plan carries the expected op); the
/// executor contract's own `common::run_resident`/`DecodeEntry` stage the raw graph themselves.
fn compiled(ptx: &PtxDevice, graph: &Graph) -> Graph {
    let graph = compiled_graph(ptx, graph);
    assert!(
        graph
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
        "compile claims the packed linear"
    );
    graph
}

fn compiled_graph(ptx: &PtxDevice, graph: &Graph) -> Graph {
    compile(
        graph,
        &target(ptx),
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile the packed graph")
    .graph()
    .clone()
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

/// Decode over an explicit one-hot activation slot: `y = x @ W^T`, the `[STEPS, OUT]` cache written
/// at the `Pos` slot (the resident path binds any slot by value).
fn slot_decode_graph(descriptor: poot_quant::PackedWeight) -> (Graph, ValueId, ValueId) {
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
    (b.finish_with_state(y, &[(cache, cache_out)]), x.id, pos.id)
}

/// The model decode contract `capture_decode` binds (Token, Pos, Mask slots): the token selects a
/// row of an identity table as the one-hot activation, and the mask (whose max is 0 at every step:
/// position 0 is always visible) is added exactly so the slot is live.
fn captured_decode_graph(descriptor: poot_quant::PackedWeight) -> Graph {
    let b = Builder::new();
    let token = b.slot(Slot::Token, TensorType::f32(vec![]));
    let eye = b.constant("eye", TensorType::f32(vec![K, K]));
    let row = b.gather_scalar(eye, 0, token);
    let x = b.reshape(row, vec![1, K]);
    let y = poot_graph_ir::ops::packed_linear(&b, x, "layer", descriptor, None, None).unwrap();
    let mask = b.slot(Slot::Mask, TensorType::f32(vec![STEPS]));
    let visible = b.reduce(RedOp::Max, mask, 0, false);
    let y = b.binary(BinOp::Add, y, visible);
    let pos = b.slot(Slot::Pos, TensorType::f32(vec![]));
    let cache = b.state_input(
        "decoded",
        TensorType::f32(vec![STEPS, OUT]),
        StateRole::Recurrent,
    );
    let cache_out = b.dynamic_update_slice_dyn(cache, y, pos, 0);
    b.finish_with_state(y, &[(cache, cache_out)])
}

fn eye() -> HostTensor {
    let mut data = vec![0.0f32; K * K];
    for i in 0..K {
        data[i * K + i] = 1.0;
    }
    HostTensor::f32(vec![K, K], data)
}

/// W9 decode: one decode entry stepped STEPS times with carried state, bit-exact against the oracle
/// decoder.
#[test]
fn packed_decode_through_resident_path_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let (graph, x, pos) = slot_decode_graph(owner.weight());
        assert!(
            compiled(&ptx, &graph)
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
            "compile claims the packed linear"
        );
        let weights = carriers(&graph, &owner);
        let path = "decode entry";
        let first_inputs = {
            let mut inputs = weights.clone();
            inputs.insert(x, one_hot(1, |_| column(0)).into());
            inputs.insert(pos, HostTensor::scalar(0.0).into());
            inputs
        };
        let mut decode =
            common::DecodeEntry::capture(&mut ptx, &graph, &first_inputs, FusionPolicy::Full);
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = weights.clone();
            inputs.insert(x, one_hot(1, |_| k).into());
            inputs.insert(pos, HostTensor::scalar(step as f32).into());
            let y = decode.step(&graph, &inputs);
            assert_decode_row(
                &format!("{format:?} {path} step {step}"),
                &owner,
                y.as_f32().unwrap(),
                k,
            );
        }
        let cache = decode.read_state("decoded", vec![STEPS, OUT]);
        for step in 0..STEPS {
            assert_decode_row(
                &format!("{format:?} {path} carried row {step}"),
                &owner,
                &cache.as_f32().unwrap()[step * OUT..(step + 1) * OUT],
                column(step),
            );
        }
    }
}

/// W9 capture/replay: the decode graph is captured once ([`common::DecodeEntry::capture`]) and
/// replayed per token ([`common::DecodeEntry::step`]), the token selecting the one-hot column,
/// bit-exact against the oracle decoder.
#[test]
fn packed_decode_through_capture_and_replay_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    for (seed, format) in formats().into_iter().enumerate() {
        let owner = Arc::new(poot_test_util::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 47,
        ));
        let graph = captured_decode_graph(owner.weight());
        assert!(
            compiled(&ptx, &graph)
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
            "compile claims the packed linear"
        );
        let mut consts = carriers(&graph, &owner);
        let eye_id = graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).name.as_deref() == Some("eye"))
            .unwrap();
        consts.insert(eye_id, eye().into());

        let mask_id = graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).storage == poot_graph_ir::Storage::Slot(Slot::Mask))
            .unwrap();
        let token_id = graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).storage == poot_graph_ir::Storage::Slot(Slot::Token))
            .unwrap();
        let pos_id = graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).storage == poot_graph_ir::Storage::Slot(Slot::Pos))
            .unwrap();
        let mask = HostTensor::f32(vec![STEPS], vec![0.0f32; STEPS]);

        let mut first_inputs = consts.clone();
        first_inputs.insert(token_id, HostTensor::scalar(column(0) as f32).into());
        first_inputs.insert(pos_id, HostTensor::scalar(0.0).into());
        first_inputs.insert(mask_id, mask.clone().into());
        let mut decode =
            common::DecodeEntry::capture(&mut ptx, &graph, &first_inputs, FusionPolicy::Full);
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = consts.clone();
            inputs.insert(token_id, HostTensor::scalar(k as f32).into());
            inputs.insert(pos_id, HostTensor::scalar(step as f32).into());
            inputs.insert(mask_id, mask.clone().into());
            let y = decode.step(&graph, &inputs);
            assert_decode_row(
                &format!("{format:?} decode_step {step}"),
                &owner,
                y.as_f32().unwrap(),
                k,
            );
        }
    }
}

/// W9 prefill: an identity activation decodes the whole weight through `run_prefill`, bit-exact per
/// element, output and carried state.
#[test]
fn packed_prefill_through_serving_path_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
        assert!(
            compiled(&ptx, &graph)
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedContraction { .. })),
            "compile claims the packed linear"
        );
        let mut inputs = carriers(&graph, &owner);
        inputs.insert(x.id, one_hot(K, |r| r).into());
        let path = "prefill entry";
        // A one-shot entry (the contract's twin of the pre-549 `run_prefill`): state starts zero
        // (Z5), the one step both computes `y` and commits its whole-cache write, then a probe entry
        // reads the carried result back by name.
        let mut decode =
            common::DecodeEntry::capture(&mut ptx, &graph, &inputs, FusionPolicy::Full);
        let y = decode.step(&graph, &inputs);
        let cache = decode.read_state("decoded", vec![K, OUT]);
        for r in 0..K {
            let row = &y.as_f32().unwrap()[r * OUT..(r + 1) * OUT];
            assert_decode_row(&format!("{format:?} {path} row {r}"), &owner, row, r);
            let carried = &cache.as_f32().unwrap()[r * OUT..(r + 1) * OUT];
            assert_decode_row(
                &format!("{format:?} {path} carried {r}"),
                &owner,
                carried,
                r,
            );
        }
    }
}

// `a_swapped_packed_role_is_refused_before_upload_on_ptx` (W11a) is not ported - see the module doc's
// Card 549 note: the executor contract's packed binding is name-driven, so there is no caller-supplied
// per-value role left to swap.

/// Card 642: the canonical packed MoE chain (`packed_indexed_linear` and
/// `packed_grouped_linear`, every expert a `PackedDequant` planned as `Materialize`, the table the
/// dense `IndexedMatMul`) compiles and runs through `run_resident`, row `m` reading expert `ids[m]`:
/// with a one-hot row the output is exactly that expert's decoded column, bit-exact against the
/// oracle decoder (zero sign canonicalized).
#[test]
fn canonical_packed_moe_chain_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let graph = b.finish(y);
            let mut inputs: HashMap<ValueId, Value> = HashMap::new();
            for &id in &graph.inputs {
                let Some(name) = graph
                    .meta(id)
                    .name
                    .as_deref()
                    .and_then(PackedSourceName::parse)
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
            let out = common::run_resident(&mut ptx, &graph, &inputs);
            for m in 0..ROWS {
                assert_decode_row(
                    &format!("{what} row {m} expert {}", expert_of(m)),
                    &owners[expert_of(m)],
                    &out.as_f32().unwrap()[m * OUT..(m + 1) * OUT],
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
fn packed_embedding_row_gather_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let graph = b.finish(rows);
            assert!(
                compiled_graph(&ptx, &graph)
                    .eqns
                    .iter()
                    .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
                "{what}: compile claims the embedding"
            );
            let mut inputs = carriers(&graph, &owner);
            let token_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
            inputs.insert(token.id, HostTensor::f32(shape, token_ids).into());
            let out = common::run_resident(&mut ptx, &graph, &inputs);
            assert_eq!(out.as_f32().unwrap().len(), ids.len() * K, "{what}");
            let mut expected = vec![0.0f32; K];
            for (r, &id) in ids.iter().enumerate() {
                owner.decode_row(id, &mut expected).unwrap();
                for (c, (got, want)) in out.as_f32().unwrap()[r * K..(r + 1) * K]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                {
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
