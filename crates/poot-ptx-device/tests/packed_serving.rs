//! Card 642: packed decode through the PTX SERVING executor paths (run on
//! the M4 pod: this box has no NVIDIA GPU).
//!
//! A `packed_linear` graph (compiled, so `compile` claims it as a `PackedContraction`, or as the packed head of a `ContractionEpilogue`) runs through
//! the executor contract - a one-shot entry for the cases the deleted `run_resident`/`run_prefill` served, a
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
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::op::ContractionHead;
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{
    BinOp, Builder, Graph, OpKind, PackedSourceName, Slot, StateRole, TensorType, ValueId,
};
use poot_graph_plan::test_support::{gpu_backend, gpu_caps};
use poot_graph_plan::{CompileOptions, Target, compile};
use poot_ptx_device::PtxDevice;
use poot_quant::format::{GroupMap, WeightFormat};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_quant::{PackedComponentRef, PackedPayload, StoredBytes};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::Backend;
use poot_tensor::{HostTensor, HostView};

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

fn target(ptx: &PtxDevice) -> Target {
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: gpu_caps(&poot_executor::Device::target(ptx)),
    }
}

/// Compiles `graph` and asserts `compile` claims the packed linear: as a bare `PackedContraction`, or as the
/// head of a `ContractionEpilogue` when a single-use pointwise consumer fused into it. The executor contract's own `common::run_resident`/`DecodeEntry` stage the raw graph
/// themselves.
fn compiled(ptx: &PtxDevice, graph: &Graph) -> Graph {
    let graph = compiled_graph(ptx, graph);
    assert!(
        graph.eqns().iter().any(|eqn| matches!(
            eqn.op,
            OpKind::PackedContraction { .. }
                | OpKind::ContractionEpilogue {
                    head: ContractionHead::Packed { .. },
                    ..
                }
        )),
        "compile claims the packed linear"
    );
    graph
}

fn compiled_graph(ptx: &PtxDevice, graph: &Graph) -> Graph {
    compile(
        graph,
        &target(ptx),
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the packed graph")
    .graph()
    .clone()
}

/// The packed carriers of `graph`, bound as components of `owner` by the role their names carry.
fn carriers(graph: &Graph, owner: &Arc<PackedPayload>) -> HashMap<ValueId, Value> {
    graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let name = PackedSourceName::parse(graph.meta(id).name().as_deref()?)?;
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
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "onehot", TensorType::f32(vec![1, K]))
        .unwrap();
    let y = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
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
    (
        b.finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
            .unwrap(),
        x.id(),
        pos.id(),
    )
}

/// The model decode contract `capture_decode` binds (Token, Pos slots): the token selects a
/// row of an identity table as the one-hot activation.
fn captured_decode_graph(descriptor: poot_quant::PackedWeight) -> Graph {
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![], DType::I32))
        .unwrap();
    let eye = b.constant("eye", TensorType::f32(vec![K, K])).unwrap();
    let row = b.gather_scalar(eye, 0, token).unwrap();
    let x = b.reshape(row, vec![1, K]).unwrap();
    let y = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
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
    b.finish(poot_graph_ir::Outputs::new(y).with_state(&[(cache, cache_out)]))
        .unwrap()
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
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 11,
        ));
        let (graph, x, pos) = slot_decode_graph(owner.weight());
        compiled(&ptx, &graph);
        let weights = carriers(&graph, &owner);
        let path = "decode entry";
        let first_inputs = {
            let mut inputs = weights.clone();
            inputs.insert(x, one_hot(1, |_| column(0)).into());
            inputs.insert(pos, HostTensor::scalar(0.0).into());
            inputs
        };
        let mut decode = common::DecodeEntry::capture(&mut ptx, &graph, &first_inputs);
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = weights.clone();
            inputs.insert(x, one_hot(1, |_| k).into());
            inputs.insert(pos, HostTensor::i32(vec![], vec![step as i32]).into());
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
        let owner = Arc::new(poot_quant::test_support::packed::random_payload(
            format,
            [OUT, K],
            seed as u64 + 47,
        ));
        let graph = captured_decode_graph(owner.weight());
        compiled(&ptx, &graph);
        let mut consts = carriers(&graph, &owner);
        let eye_id = graph
            .inputs()
            .iter()
            .copied()
            .find(|&id| graph.meta(id).name().as_deref() == Some("eye"))
            .unwrap();
        consts.insert(eye_id, eye().into());

        let token_id = graph
            .inputs()
            .iter()
            .copied()
            .find(|&id| graph.meta(id).storage == poot_graph_ir::Storage::Slot(Slot::Token))
            .unwrap();
        let pos_id = graph
            .inputs()
            .iter()
            .copied()
            .find(|&id| graph.meta(id).storage == poot_graph_ir::Storage::Slot(Slot::Pos))
            .unwrap();

        let mut first_inputs = consts.clone();
        first_inputs.insert(
            token_id,
            HostTensor::i32(vec![], vec![column(0) as i32]).into(),
        );
        first_inputs.insert(pos_id, HostTensor::i32(vec![], vec![0]).into());
        let mut decode = common::DecodeEntry::capture(&mut ptx, &graph, &first_inputs);
        for step in 0..STEPS {
            let k = column(step);
            let mut inputs = consts.clone();
            inputs.insert(token_id, HostTensor::i32(vec![], vec![k as i32]).into());
            inputs.insert(pos_id, HostTensor::i32(vec![], vec![step as i32]).into());
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

/// W9 prefill: an identity activation decodes the whole weight through a prefill entry, bit-exact per
/// element, output and carried state.
#[test]
fn packed_prefill_through_serving_path_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
        let y = poot_graph_ir::test_support::packed_graphs::packed_linear(
            &b,
            x,
            "layer",
            owner.weight(),
            None,
            None,
        )
        .unwrap();
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
        compiled(&ptx, &graph);
        let mut inputs = carriers(&graph, &owner);
        inputs.insert(x.id(), one_hot(K, |r| r).into());
        let path = "prefill entry";
        // A one-shot entry (the contract's twin of the pre-549 `run_prefill`): state starts zero
        // (Z5), the one step both computes `y` and commits its whole-cache write, then a probe entry
        // reads the carried result back by name.
        let mut decode = common::DecodeEntry::capture(&mut ptx, &graph, &inputs);
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

const EXPERTS: usize = 3;
const ROWS: usize = 5;
/// Rows past `INDEXED_GEMV_MAX_ROWS` (32), so an indexed contraction of them runs the register-tiled expert walk.
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

/// POOT-1323: 17 Q8_0 `[4096, 8192]` experts are 606 MB of one source, past the 2^32 bits a 32-bit backend's kernels
/// address; the last expert's slab starts at bit 4.56e9. A 64-bit backend keeps the table whole (one carrier per
/// source), and the rows routed to the first, a middle and the last expert are bit-exact against the oracle decoder.
const WIDE_EXPERTS: usize = 17;

/// Rows 0, 1 and 2 routed to the first, a middle and the last expert of the [`WIDE_EXPERTS`] table.
fn across_the_wide_table(m: usize) -> usize {
    [0, WIDE_EXPERTS / 2, WIDE_EXPERTS - 1][m]
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
                let split = poot_graph_plan::legalize(
                    g,
                    gpu_backend(&target),
                    &caps,
                    &poot_graph_plan::CompileLimits::STANDARD,
                )
                .expect("the stacked table splits");
                let carriers: Vec<usize> = split
                    .consts()
                    .iter()
                    .map(|&id| split.aval(id).shape[0])
                    .collect();
                assert_eq!(
                    carriers,
                    self.carrier_experts(descriptor.sources().len(), EXPERTS),
                    "the experts each stacked carrier holds after the split"
                );
                split
            }
        }
    }

    /// The leading extents of the compiled graph's stacked carriers: one buffer of every expert, or the 2 + 1
    /// split with every source's piece 0 before its piece 1.
    fn carrier_experts(self, sources: usize, experts: usize) -> Vec<usize> {
        match self {
            Self::One => vec![experts; sources],
            Self::Split => [[2].repeat(sources), [1].repeat(sources)].concat(),
        }
    }
}

/// One `packed_indexed_linear` launch over `EXPERTS` stacked `[out, k]` experts of `format`, row `m` routed to
/// the expert `expert_of(m)` routes it to with a one-hot activation: one `PackedContraction { rows: Indexed }`, each row
/// bit-exact against the oracle decoder (zero sign canonicalized).
fn run_indexed_case(
    ptx: &mut PtxDevice,
    format: WeightFormat,
    experts: usize,
    (rows, out, k): (usize, usize, usize),
    expert_of: fn(usize) -> usize,
    buffers: StackBuffers,
) {
    let owners: Vec<Arc<PackedPayload>> = (0..experts)
        .map(|e| {
            Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
                format,
                [out, k],
                e as u64 * 7 + 101,
            ))
        })
        .collect();
    let table = poot_graph_ir::test_support::packed_graphs::PackedExpertStack {
        stack: poot_graph_ir::test_support::packed_graphs::TEST_STACK,
        descriptor: owners[0].weight(),
        experts,
    };
    let stack = Arc::new(poot_quant::PackedStack::try_new(owners.clone()).unwrap());
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
            poot_graph_ir::IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let y = poot_graph_ir::test_support::packed_graphs::packed_indexed_linear(&b, x, ids, &table)
        .unwrap();
    let graph = b.finish(y).unwrap();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in graph.inputs() {
        let Some(name) = graph
            .meta(id)
            .name()
            .as_deref()
            .and_then(PackedSourceName::parse)
        else {
            continue;
        };
        inputs.insert(
            id,
            poot_quant::PackedStackComponentRef::new(Arc::clone(&stack), name.role()).into(),
        );
    }
    inputs.insert(x.id(), indexed_one_hot(rows, k).into());
    let selected: Vec<i32> = (0..rows).map(|m| expert_of(m) as i32).collect();
    inputs.insert(ids.id(), HostTensor::i32(vec![rows], selected).into());
    let target = poot_executor::Device::target(&*ptx);
    let got = common::run_resident_prepared(ptx, &graph, &inputs, |named| {
        buffers.graph(named, target, table.descriptor)
    });
    // Only the routed experts are decoded: a stack past 512 MiB must not cost a host decode of every expert.
    let decoded: HashMap<usize, Vec<f32>> = (0..rows)
        .map(expert_of)
        .map(|e| (e, decoded_rows(&owners[e])))
        .collect();
    for m in 0..rows {
        assert_indexed_row(
            &format!("{what} row {m} expert {}", expert_of(m)),
            &decoded[&expert_of(m)],
            k,
            &got.as_f32().unwrap()[m * out..(m + 1) * out],
            column_of(m, k),
        );
    }
}

/// SC-001 (PTX): `packed_indexed_linear` over a stacked table of every format compiles and runs through
/// `run_resident` as one `PackedContraction { rows: Indexed }`, one-hot rows bit-exact against the oracle
/// decoder. Mutation: slab stride 0 in the planner's metadata (every row reads expert 0); the rows routed to
/// experts 1 and 2 go red.
#[test]
fn packed_indexed_contraction_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    for format in expert_formats() {
        run_indexed_case(
            &mut ptx,
            format,
            EXPERTS,
            (ROWS, OUT, K),
            interleaved,
            StackBuffers::One,
        );
    }
}

/// Card 1067 (PTX): a stacked table over the target's single-buffer limit runs split into expert ranges
/// (0..2 and 2..3, each its own buffer per source) and is bit-exact against the oracle decoder on every format,
/// with rows routed to every expert, the last one alone in its buffer. The reduced limit is `legalize`'s one input
/// (`StackBuffers::Split`); the graph, weights and executor are those of the unsplit case.
/// Mutation: in `legalize`'s `split_oversized_expert_stacks` shift the ids of the later pieces by one less
/// than `range.start`; the row routed to expert 2 reads the wrong slab and goes red.
#[test]
fn a_split_expert_stack_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    for format in expert_formats() {
        run_indexed_case(
            &mut ptx,
            format,
            EXPERTS,
            (ROWS, 2, K),
            interleaved,
            StackBuffers::Split,
        );
    }
}

/// POOT-1323 (PTX): a stacked table past 512 MiB stays one buffer per source and runs bit-exact, including the
/// expert whose slab offset passes `u32::MAX` bits: the 64-bit `usize` offset chain of the indexed kernel is honoured
/// end to end. Mutation: bound `legalize`'s `expert_ranges` by `u32::MAX` bits again; the table splits 15 + 2 and the
/// carrier-extent assertion goes red.
#[test]
fn a_stack_past_512_mib_runs_whole_and_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    run_indexed_case(
        &mut ptx,
        WeightFormat::Q8_0,
        WIDE_EXPERTS,
        (3, 4096, 8192),
        across_the_wide_table,
        StackBuffers::One,
    );
}

/// SC-002 (PTX): a Q8_0 table with `out = 3, K = 32` is 102 bytes per expert, not a whole number of words,
/// so the binder pads each expert to 104 bytes and the kernel strides by that; one-hot rows routed to
/// experts 1 and 2 are bit-exact. Mutation: drop the per-expert padding in `WeightWriter::read_stream`; experts 1 and
/// 2 read misaligned blocks and go red.
#[test]
fn an_unaligned_expert_slab_table_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    run_indexed_case(
        &mut ptx,
        WeightFormat::Q8_0,
        EXPERTS,
        (ROWS, 3, 32),
        interleaved,
        StackBuffers::One,
    );
}

/// A prefill-sized routed table (PTX): `PREFILL_ROWS` rows is past the decode row count, so the indexed packed
/// contraction runs the register-tiled expert walk (rows routed by `gapped`, so a tile holds several
/// experts and may skip one) over every format, with `N = 11` leaving edge columns in the last column tile. One-hot rows bit-exact
/// against the oracle decoder. Mutation: take the Tiled step's slab from the wrong segment (`weight.step(.., lo,
/// ..)` in `tiled_body`); rows routed to an expert after a gap read another expert and go red.
#[test]
fn packed_indexed_prefill_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    for format in expert_formats() {
        run_indexed_case(
            &mut ptx,
            format,
            EXPERTS,
            (PREFILL_ROWS, PREFILL_OUT, K),
            gapped,
            StackBuffers::One,
        );
    }
}

/// A routed prefill above the wgpu grid limit (PTX): `4096 x 4160` outputs of a Q8_0 table is 17.0M, past
/// 65,535 x 256, which the Gemv-body route refused; the register-tiled schedule plans it (folded onto `GroupY` where the
/// device's X extent is smaller than the tile count) and every one of the 17M outputs is bit-exact against the
/// oracle decoder. Mutation: read the tile index from `GroupX` alone in `tiled_body`; every tile past
/// the first fold row is never computed and goes red.
#[test]
fn packed_indexed_prefill_above_the_gemv_grid_limit_runs_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    run_indexed_case(
        &mut ptx,
        WeightFormat::Q8_0,
        EXPERTS,
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
fn packed_embedding_row_gather_is_bit_exact_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let rows = poot_graph_ir::test_support::packed_graphs::packed_embedding(
                &b,
                token,
                "model.embed_tokens",
                owner.weight(),
            )
            .unwrap();
            let graph = b.finish(rows).unwrap();
            assert!(
                compiled_graph(&ptx, &graph)
                    .eqns()
                    .iter()
                    .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
                "{what}: compile claims the embedding"
            );
            let mut inputs = carriers(&graph, &owner);
            let token_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            inputs.insert(token.id(), HostTensor::i32(shape, token_ids).into());
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

/// POOT-1060: the sharded-PLE lookup's table shape - a dense E4M3FN `[rows, width]` checkpoint const
/// gathered by an exact I32 index and widened in one `DenseRowGather` - through the executor contract,
/// bit-exact against `poot_quant::scalar::e4m3fn_to_f32`. The table holds all 256 codes (NaN
/// included) and is read at its first and last row, a repeated row and an interior row, at widths 1, 3,
/// 4, 6 and 8: a width that is not a multiple of four pads each row to a whole `u32` word
/// (`words_per_row = ceil(width / 4)`), so a flat `element / 4` lane read goes red on the odd widths.
/// Mutations (each red, then restored): pad rows to `width / 4` words in `lanes::row_padded_words`;
/// flip the sign/exponent decode in `packed_e4m3_row_gather.rs`; read `index[s] + 1`.
#[test]
fn dense_e4m3_row_gather_is_bit_exact_on_ptx() {
    let Some(ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let mut engine = Engine::new(ptx);
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
        let entry = engine
            .add_entry(
                exe,
                &common::staged_program(&named, Device::target(engine.device())),
            )
            .unwrap();
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
        let out = bytemuck::cast_slice::<u8, f32>(&bytes);
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

fn oracle(graph: &Graph, inputs: &HashMap<ValueId, Value>) -> HostTensor {
    poot_eval::eval(graph, inputs, poot_eval::EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap()
}

/// SC-001: Q4_K, Q8_0 and FP8 contractions through the packed loaders meet their tier-2 oracle contract. A
/// dense activation makes every output a sum over every `K` block, so a dropped block or a wrong scale moves
/// every element; `m == 1` takes the Gemv schedule and `m == 17` the register-tiled schedule (ragged against its tile),
/// and every output element is compared. Mutation: omit one block's contribution in `DecodePlan`'s address or
/// scale a factor by a nonzero constant; every row goes red. Runs on the PTX pod.
#[test]
fn packed_loaders_match_the_oracle_on_every_output_on_ptx() {
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let y = poot_graph_ir::test_support::packed_graphs::packed_linear(
                &b,
                x,
                "layer",
                owner.weight(),
                None,
                None,
            )
            .unwrap();
            let graph = b.finish(y).unwrap();
            let mut inputs = carriers(&graph, &owner);
            inputs.insert(
                x.id(),
                HostTensor::f32(vec![m, K], fill(7 + m as u64, m * K)).into(),
            );
            let (got, class) = common::run_resident_classed(&mut ptx, &graph, &inputs);
            let want = oracle(&graph, &inputs);
            poot_test_util::assert_close_scaled(
                got.as_f32().unwrap(),
                want.as_f32().unwrap(),
                class.tolerance(),
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
fn a_packed_gemv_head_is_shared_with_the_standalone_gemv_on_ptx() {
    use poot_graph_ir::op::BinOp;
    use poot_test_util::assert_close_scaled;
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let head = poot_graph_ir::test_support::packed_graphs::packed_linear(
                &b,
                x,
                "layer",
                owner.weight(),
                None,
                None,
            )
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
        let head = common::run_resident(&mut ptx, &head_graph, &head_inputs);
        let head = head.as_f32().unwrap().to_vec();

        let (graph, x, bias) = build(true);
        let bias = bias.unwrap();
        let heads: Vec<ContractionHead> = compiled_graph(&ptx, &graph)
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                poot_graph_ir::OpKind::ContractionEpilogue { head, .. } => Some(*head),
                _ => None,
            })
            .collect();
        assert!(
            matches!(heads.as_slice(), [ContractionHead::Packed { .. }]),
            "{what}: one fused packed contraction"
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
            let fused = common::run_resident(&mut ptx, &graph, &inputs);
            let want: Vec<f32> = head
                .iter()
                .zip(&bias_values)
                .map(|(head, bias)| head + bias)
                .collect();
            assert_close_scaled(
                fused.as_f32().unwrap(),
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
/// oracle row go red. Runs on the PTX pod.
#[test]
fn a_packed_gemv_epilogue_agrees_with_the_oracle_on_ptx() {
    use poot_graph_ir::op::UnOp;
    use poot_graph_ir::types::Scalar;
    use poot_test_util::assert_close_scaled;
    let Some(mut ptx) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
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
            let head = poot_graph_ir::test_support::packed_graphs::packed_linear(
                &b,
                x,
                "layer",
                owner.weight(),
                None,
                None,
            )
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
        let head = common::run_resident(&mut ptx, &head_graph, &head_inputs);
        let head = head.as_f32().unwrap().to_vec();
        let head_scale = head.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        let (graph, x, leaves) = build(true);
        let (bias, residual) = leaves.unwrap();
        let residual_values = fill(5, OUT);
        let heads: Vec<ContractionHead> = compiled_graph(&ptx, &graph)
            .eqns()
            .iter()
            .filter_map(|eqn| match &eqn.op {
                poot_graph_ir::OpKind::ContractionEpilogue { head, .. } => Some(*head),
                _ => None,
            })
            .collect();
        assert!(
            matches!(heads.as_slice(), [ContractionHead::Packed { .. }]),
            "{what}: one fused packed contraction"
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
            let (fused, class) = common::run_resident_classed(&mut ptx, &graph, &inputs);
            if label == "generic bias" {
                assert_close_scaled(
                    fused.as_f32().unwrap(),
                    oracle(&graph, &inputs).as_f32().unwrap(),
                    class.tolerance(),
                    &format!("{what}: fused vs the oracle"),
                );
            } else {
                // With the head cancelled, the result is the residual to a few ulps of the head's size.
                let tolerance = 64.0 * f32::EPSILON * head_scale.max(1.0);
                for (index, (&got, &want)) in fused
                    .as_f32()
                    .unwrap()
                    .iter()
                    .zip(&residual_values)
                    .enumerate()
                {
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
