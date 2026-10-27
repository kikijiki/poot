//! POOT-1337: the decode packed GEMV against its byte roofline, at the Flash-Next decode shapes.
//!
//! Each row builds the production graph of one decode projection (a packed linear at one activation row, or
//! the routed `packed_indexed_linear` at the ten active experts of a 512-expert stack), checks that the plan
//! claims it as one packed contraction, and steps it: first with a one-hot activation per row, where every
//! output is one decoded weight element, compared bit for bit with the CPU oracle's decode (a sample of rows
//! on the wide outputs), then with a dense deterministic activation, timed by the host wall of the step
//! (launch, replay, synchronize, readback: about 25-30 us of floor per step). The wall is printed against the
//! bytes the step has to read and the card's 1792 GB/s memory peak; run with `--nocapture` on a pod.
//!
//! This is a bench with a correctness half. The required PTX row built from it guards the bit-exact half only
//! (`ptx.txt`); the timing half asserts nothing and a host without an NVIDIA device skips. The CUDA-event span
//! of a replay is not used: this harness's `StepOutputs::device_time` span does not follow one step.
//! The independent reference (llama.cpp's values) is `packed_reference_blocks.rs`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use poot_eval::Value;
use poot_executor::Device;
use poot_graph_ir::test_support::packed_graphs::{PackedExpertStack, TEST_STACK};
use poot_graph_ir::{
    Builder, ContractionRows, Graph, IndexRange, OpKind, PackedSourceName, Slot, TensorType,
    ValueId,
};
use poot_graph_plan::test_support::gpu_caps;
use poot_graph_plan::{CompileOptions, Target, compile};
use poot_ptx_device::PtxDevice;
use poot_quant::format::{Storage, WeightFormat};
use poot_quant::{PackedComponentRef, PackedPayload, PackedStack, PackedStackComponentRef};
use poot_runtime_common::DeviceBackend;
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};

mod common;

/// The RTX PRO 6000 Blackwell's memory peak, GB/s (NVIDIA's product page, as the POOT-1332 profile uses).
const PEAK_GB_S: f64 = 1792.0;
const WARMUP: usize = 5;
const TIMED: usize = 40;

/// One decode projection: `[out, k]` weights of `format`, `rows` activation rows, and for a routed one the
/// stacked expert count.
struct Case {
    name: &'static str,
    format: WeightFormat,
    shape: [usize; 2],
    rows: usize,
    experts: Option<usize>,
}

/// The Flash-Next decode GEMV families (POOT-1332's profile): the IQ2_XS gate/up and IQ4_NL down expert
/// GEMVs at ten active experts, and one projection per packed family at its largest repeated shape.
const CASES: [Case; 8] = [
    Case {
        name: "expert gate/up",
        format: WeightFormat::Iq2_Xs,
        shape: [640, 2560],
        rows: 10,
        experts: Some(512),
    },
    Case {
        name: "expert down",
        format: WeightFormat::Iq4_Nl,
        shape: [2560, 640],
        rows: 10,
        experts: Some(512),
    },
    Case {
        name: "q5_k qkv",
        format: WeightFormat::Q5_K,
        shape: [10240, 2560],
        rows: 1,
        experts: None,
    },
    Case {
        name: "q5_k wide",
        format: WeightFormat::Q5_K,
        shape: [12288, 2560],
        rows: 1,
        experts: None,
    },
    Case {
        name: "q5_k down",
        format: WeightFormat::Q5_K,
        shape: [2560, 6144],
        rows: 1,
        experts: None,
    },
    Case {
        name: "q6_k down",
        format: WeightFormat::Q6_K,
        shape: [2560, 6144],
        rows: 1,
        experts: None,
    },
    Case {
        name: "q8_0 hyper",
        format: WeightFormat::Q8_0,
        shape: [10240, 320],
        rows: 1,
        experts: None,
    },
    Case {
        name: "lm_head q4_k",
        format: WeightFormat::Q4_K,
        shape: [248_320, 2560],
        rows: 1,
        experts: None,
    },
];

/// `graph` compiled for the device, with the plan's claim asserted: exactly one packed contraction, indexed
/// when the case is routed. The entry below compiles the raw graph, as the serving tests do.
fn assert_claims_one_packed_contraction(gpu: &PtxDevice, graph: &Graph, indexed: bool) {
    let plan = compile(
        graph,
        &Target::Gpu {
            backend: Backend::Nvptx,
            caps: gpu_caps(&Device::target(gpu)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the packed graph");
    let contractions: Vec<_> = plan
        .graph()
        .eqns()
        .iter()
        .filter_map(|eqn| match &eqn.op {
            OpKind::PackedContraction { rows, .. } => {
                Some(matches!(rows, ContractionRows::Indexed { .. }))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        contractions,
        [indexed],
        "the plan claims one packed contraction (indexed: {indexed})"
    );
}

/// Bytes of packed weight one row of the contraction reads: `out * k` elements at the format's block rate.
fn row_bytes(format: WeightFormat, [out, k]: [usize; 2]) -> usize {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        panic!("{format:?}: this bench covers the block formats");
    };
    out * k.div_ceil(layout.values) * layout.bytes
}

/// A deterministic dense activation in `[-1, 1)`.
fn dense_activation(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u64).wrapping_mul(2_654_435_761) >> 7) % 2048) as f32 / 1024.0 - 1.0)
        .collect()
}

struct Bound {
    graph: Graph,
    inputs: HashMap<ValueId, Value>,
    x: ValueId,
    ids: Option<ValueId>,
    /// The payload each activation row's expert (or the one weight) decodes from.
    owners: Vec<Arc<PackedPayload>>,
    selected: Vec<usize>,
}

fn bind(gpu: &PtxDevice, case: &Case) -> Bound {
    let k = case.shape[1];
    let distinct = case.experts.map_or(1, |_| case.rows);
    // Ten experts spread across the 512 (the router picks distinct experts per token); only those are
    // generated, the rest of the stack is the first one's payload so the table has its real stride.
    let selected: Vec<usize> = match case.experts {
        Some(experts) => (0..case.rows).map(|m| (m * 53 + 7) % experts).collect(),
        None => vec![0],
    };
    let payload = |seed: u64| {
        Arc::new(poot_quant::test_support::packed::random_payload(
            case.format,
            case.shape,
            seed,
        ))
    };
    let distinct_owners: Vec<Arc<PackedPayload>> =
        (0..distinct).map(|i| payload(i as u64 + 1337)).collect();
    let b = Builder::for_tests();
    match case.experts {
        Some(experts) => {
            let owners: Vec<Arc<PackedPayload>> = (0..experts)
                .map(|e| {
                    selected.iter().position(|&s| s == e).map_or_else(
                        || Arc::clone(&distinct_owners[0]),
                        |i| Arc::clone(&distinct_owners[i]),
                    )
                })
                .collect();
            let table = PackedExpertStack {
                stack: TEST_STACK,
                descriptor: owners[0].weight(),
                experts,
            };
            let stack = Arc::new(PackedStack::try_new(owners.clone()).unwrap());
            let x = b
                .slot_named(
                    Slot::Activation,
                    "moe-x",
                    TensorType::f32(vec![case.rows, k]),
                )
                .unwrap();
            let ids = b
                .slot_named_ranged(
                    Slot::Activation,
                    "moe-ids",
                    TensorType::new(vec![case.rows], DType::I32),
                    IndexRange::below(experts).unwrap(),
                )
                .unwrap();
            let y = poot_graph_ir::test_support::packed_graphs::packed_indexed_linear(
                &b, x, ids, &table,
            )
            .unwrap();
            let graph = b.finish(y).unwrap();
            assert_claims_one_packed_contraction(gpu, &graph, true);
            let inputs: HashMap<ValueId, Value> = graph
                .inputs()
                .iter()
                .filter_map(|&id| {
                    let name = PackedSourceName::parse(graph.meta(id).name().as_deref()?)?;
                    Some((
                        id,
                        PackedStackComponentRef::new(Arc::clone(&stack), name.role()).into(),
                    ))
                })
                .collect();
            Bound {
                graph,
                inputs,
                x: x.id(),
                ids: Some(ids.id()),
                owners,
                selected,
            }
        }
        None => {
            let owner = Arc::clone(&distinct_owners[0]);
            let x = b
                .slot_named(Slot::Activation, "x", TensorType::f32(vec![case.rows, k]))
                .unwrap();
            let y = poot_graph_ir::test_support::packed_graphs::packed_linear(
                &b,
                x,
                "bench.linear",
                owner.weight(),
                None,
                None,
            )
            .unwrap();
            let graph = b.finish(y).unwrap();
            assert_claims_one_packed_contraction(gpu, &graph, false);
            let inputs: HashMap<ValueId, Value> = graph
                .inputs()
                .iter()
                .filter_map(|&id| {
                    let name = PackedSourceName::parse(graph.meta(id).name().as_deref()?)?;
                    Some((
                        id,
                        PackedComponentRef::new(Arc::clone(&owner), name.role()).into(),
                    ))
                })
                .collect();
            Bound {
                graph,
                inputs,
                x: x.id(),
                ids: None,
                owners: vec![owner],
                selected,
            }
        }
    }
}

#[test]
fn decode_gemv_bandwidth_at_the_flash_next_shapes() {
    let Some(mut gpu) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    println!(
        "{:<16} {:<8} {:>14} {:>5} {:>10} {:>10} {:>8}",
        "case", "format", "shape", "rows", "bytes", "wall us", "GB/s"
    );
    for case in &CASES {
        let [out, k] = case.shape;
        let mut bound = bind(&gpu, case);
        let rows = case.rows;
        if let Some(ids) = bound.ids {
            bound.inputs.insert(
                ids,
                HostTensor::i32(
                    vec![rows],
                    bound.selected.iter().map(|&e| e as i32).collect(),
                )
                .into(),
            );
        }
        // Correctness: row m's one-hot picks column (m * 37 + 11) % k, so output (m, o) is the decoded
        // weight element (expert(m), o, column(m)), exact.
        let column = |m: usize| (m * 37 + 11) % k;
        let mut one_hot = vec![0.0f32; rows * k];
        for m in 0..rows {
            one_hot[m * k + column(m)] = 1.0;
        }
        bound
            .inputs
            .insert(bound.x, HostTensor::f32(vec![rows, k], one_hot).into());
        let mut entry = common::DecodeEntry::capture(&mut gpu, &bound.graph, &bound.inputs);
        let got = entry.step(&bound.graph, &bound.inputs);
        let got = got.as_f32().unwrap().to_vec();
        let mut row = vec![0.0f32; k];
        for m in 0..rows {
            let owner = &bound.owners[bound.selected[m]];
            // Every row of a small output; a spread of 512 rows (and the last) of a wide one, so the host
            // oracle stays quick at the 248k-row lm_head.
            let stride = (out / 512).max(1);
            for o in (0..out).step_by(stride).chain([out - 1]) {
                owner.decode_row(o, &mut row).unwrap();
                let want = row[column(m)];
                let device = got[m * out + o];
                assert_eq!(
                    (device + 0.0).to_bits(),
                    (want + 0.0).to_bits(),
                    "{} {:?} {:?}: output ({m}, {o}) device {device} vs oracle {want}",
                    case.name,
                    case.format,
                    case.shape
                );
            }
        }
        // Bandwidth: a dense activation, the median host wall of the replays.
        bound.inputs.insert(
            bound.x,
            HostTensor::f32(vec![rows, k], dense_activation(rows * k)).into(),
        );
        let mut walls: Vec<Duration> = Vec::with_capacity(TIMED);
        for step in 0..WARMUP + TIMED {
            let started = std::time::Instant::now();
            entry.step(&bound.graph, &bound.inputs);
            if step >= WARMUP {
                walls.push(started.elapsed());
            }
        }
        walls.sort();
        let median = walls[walls.len() / 2].as_secs_f64();
        let bytes = rows * row_bytes(case.format, case.shape);
        let gb_s = bytes as f64 / median / 1e9;
        println!(
            "{:<16} {:<8} {:>14} {:>5} {:>10} {:>10.1} {:>8.0}  ({:.1}% of {PEAK_GB_S})",
            case.name,
            format!("{:?}", case.format),
            format!("{out}x{k}"),
            rows,
            bytes,
            median * 1e6,
            gb_s,
            gb_s / PEAK_GB_S * 100.0
        );
    }
}
