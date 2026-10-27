//! POOT-761 acceptance on ROCm: `IndexedMatMul` through `compile` and the executor contract, on the indexed
//! contraction (`ContractionSpec::Indexed`: the Gemv at a decode row count, the register-tiled expert walk above it).
//!
//! Operands are integer-valued with `K * 3 * 3` far below `2^24`, so every partial sum is exact and every summation
//! order and fused multiply-add gives the same bits: the device output equals the oracle `to_bits` on every element,
//! and a wrong expert changes a value by an integer. The wgpu rows (`poot-wgpu-device`'s `graph/gemm.rs`) carry the
//! mutation record; this file runs the same fixtures on the ROCm device.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Engine};
use poot_graph_ir::op::BinOp;
use poot_graph_ir::{Builder, Graph, Storage, TensorType, ValueId};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::{DType, HostTensor};

const EXPERTS: usize = 4;
/// `K` is not a multiple of the 8-row tile and `N` not a multiple of its column edge, so the tail masks run.
const K: usize = 36;
const N: usize = 21;

struct Case {
    name: &'static str,
    ids: Vec<i32>,
}

/// Integer-valued fill in `[-3, 3]`.
fn int_fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) % 7) as f32 - 3.0
        })
        .collect()
}

/// Sorted expert ids, runs of unequal length so tile boundaries (every 8 rows) cut across expert boundaries.
fn sorted_ids(rows: usize) -> Vec<i32> {
    let bounds = [37, 78, 116];
    (0..rows)
        .map(|r| bounds.iter().filter(|&&b| r >= b).count() as i32)
        .collect()
}

/// The same multiset in a deterministic shuffled order (stride 7 is coprime to 160).
fn shuffled_ids(rows: usize) -> Vec<i32> {
    let sorted = sorted_ids(rows);
    (0..rows).map(|r| sorted[(r * 7) % rows]).collect()
}

/// One expert per 8-row tile.
fn tile_aligned_ids(rows: usize) -> Vec<i32> {
    (0..rows).map(|r| ((r / 8) % EXPERTS) as i32).collect()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "decode, unsorted ids with a repeat",
            ids: vec![2, 0, 3, 2, 1],
        },
        Case {
            name: "prefill, sorted ids",
            ids: sorted_ids(160),
        },
        Case {
            name: "prefill, shuffled ids",
            ids: shuffled_ids(160),
        },
        Case {
            name: "prefill, one expert per tile",
            ids: tile_aligned_ids(160),
        },
    ]
}

struct Program {
    graph: Graph,
    inputs: HashMap<ValueId, HostTensor>,
}

/// `IndexedMatMul(x, w, ids)` with every operand a named const; with `gate`, that product times `gate[m]`.
fn program(ids: &[i32], gate: Option<Vec<f32>>) -> Program {
    let m = ids.len();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, K, N]))
        .unwrap();
    let idx = b
        .constant_ranged(
            "idx",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(x.id(), HostTensor::f32(vec![m, K], int_fill(1, m * K)));
    inputs.insert(
        w.id(),
        HostTensor::f32(vec![EXPERTS, K, N], int_fill(2, EXPERTS * K * N)),
    );
    inputs.insert(idx.id(), HostTensor::i32(vec![m], ids.to_vec()));
    let mut out = b.indexed_matmul(x, w, idx).unwrap();
    if let Some(gate) = gate {
        let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
        let column = b.reshape(gate_t, vec![m, 1]).unwrap();
        out = b.binary(BinOp::Mul, out, column).unwrap();
    }
    Program {
        graph: b.finish(out).unwrap(),
        inputs,
    }
}

/// Experts of the checkpoint-order fixtures: more than the `EXPERTS` of the `[E, K, N]` rows, so an expert id that
/// is read in the wrong orientation lands on a different slab.
const CHECKPOINT_EXPERTS: usize = 8;

/// The stored-orientation spelling: the stacked weight is held `[E, N, K]` (the checkpoint order) and the graph
/// reads it as `indexed_matmul(x, transpose(w, [0, 2, 1]), ids)`. `compile` must fold that transpose into the
/// contraction, and the oracle evaluates the transpose literally.
fn checkpoint_program(ids: &[i32]) -> Program {
    checkpoint_program_gated(ids, None)
}

/// [`checkpoint_program`], with `gate` the routed gate weighting after the product (`ContractionEpilogue`).
fn checkpoint_program_gated(ids: &[i32], gate: Option<Vec<f32>>) -> Program {
    let m = ids.len();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![CHECKPOINT_EXPERTS, N, K]))
        .unwrap();
    let idx = b
        .constant_ranged(
            "idx",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(CHECKPOINT_EXPERTS).unwrap(),
        )
        .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(x.id(), HostTensor::f32(vec![m, K], int_fill(1, m * K)));
    inputs.insert(
        w.id(),
        HostTensor::f32(
            vec![CHECKPOINT_EXPERTS, N, K],
            int_fill(2, CHECKPOINT_EXPERTS * N * K),
        ),
    );
    inputs.insert(idx.id(), HostTensor::i32(vec![m], ids.to_vec()));
    let mut out = b
        .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), idx)
        .unwrap();
    if let Some(gate) = gate {
        let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
        out = b
            .binary(BinOp::Mul, out, b.reshape(gate_t, vec![m, 1]).unwrap())
            .unwrap();
    }
    Program {
        graph: b.finish(out).unwrap(),
        inputs,
    }
}

fn oracle(program: &Program) -> HostTensor {
    let inputs: HashMap<_, Value> = program
        .inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    eval(&program.graph, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap()
}

fn run(
    engine: &mut Engine<RocmDevice>,
    target: poot_graph_plan::Target,
    p: &Program,
) -> HostTensor {
    let consts: Vec<_> = p
        .inputs
        .iter()
        .map(|(&id, t)| {
            let meta = p.graph.meta(id);
            assert_eq!(meta.storage, Storage::Const);
            let name: &'static str = Box::leak(meta.name().unwrap().into_owned().into_boxed_str());
            poot_executor_parity::ConstFixture {
                name,
                tensor: t.clone(),
            }
        })
        .collect();
    poot_executor_parity::run_once(engine, target, &p.graph, &consts, &[])
        .unwrap_or_else(|e| panic!("{e}"))
}

fn bits(t: &HostTensor) -> Vec<u32> {
    t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// SC-001, SC-002: equal to the oracle `to_bits` on every element at a decode shape and at prefill shapes whose
/// tiles cross expert boundaries (sorted, shuffled), and on the tile-aligned fixture.
#[test]
fn indexed_matmul_matches_the_oracle_bit_for_bit() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let mut failures = Vec::new();
    for case in cases() {
        let p = program(&case.ids, None);
        let got = run(&mut engine, target, &p);
        let (got_bits, want_bits) = (bits(&got), bits(&oracle(&p)));
        let wrong: Vec<usize> = (0..got_bits.len())
            .filter(|&i| got_bits[i] != want_bits[i])
            .collect();
        if let Some(&first) = wrong.first() {
            failures.push(format!(
                "{}: {} of {} elements differ; row {} col {} is {} on the device, {} in the oracle",
                case.name,
                wrong.len(),
                got_bits.len(),
                first / N,
                first % N,
                f32::from_bits(got_bits[first]),
                f32::from_bits(want_bits[first]),
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// SC-005: a routed gate weighting runs as one fused `ContractionEpilogue` and equals the oracle (the unfused
/// program's semantics) `to_bits`.
#[test]
fn routed_gate_weighting_fused_equals_the_oracle_bit_for_bit() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for case in cases().into_iter().take(3) {
        // Dyadic gates, so the product with an exact integer sum is exact.
        let gate: Vec<f32> = (0..case.ids.len())
            .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
            .collect();
        let p = program(&case.ids, Some(gate));
        let got = run(&mut engine, target, &p);
        assert_eq!(bits(&got), bits(&oracle(&p)), "{}", case.name);
    }
}

/// POOT-646 SC-001: experts held in checkpoint `[E, N, K]` order and read through `transpose(w, [0, 2, 1])` equal
/// the oracle `to_bits` on every element: `M = 1` and `M = 5` (the Gemv, ids unsorted with repeats) and a prefill
/// shape (the register-tiled expert walk, shuffled ids).
#[test]
fn indexed_matmul_checkpoint_order_experts_match_the_oracle_bit_for_bit() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let prefill: Vec<i32> = (0..160)
        .map(|r| [7, 2, 2, 5, 0, 7, 3][(r * 5) % 7])
        .collect();
    let cases = [
        ("M=1", vec![5]),
        ("M=5, unsorted with repeats", vec![6, 0, 6, 3, 1]),
        ("M=160, shuffled", prefill),
    ];
    let mut failures = Vec::new();
    for (name, ids) in cases {
        let p = checkpoint_program(&ids);
        let got = run(&mut engine, target, &p);
        let (got_bits, want_bits) = (bits(&got), bits(&oracle(&p)));
        if let Some(first) = (0..got_bits.len()).find(|&i| got_bits[i] != want_bits[i]) {
            failures.push(format!(
                "{name}: row {} col {} is {} on the device, {} in the oracle",
                first / N,
                first % N,
                f32::from_bits(got_bits[first]),
                f32::from_bits(want_bits[first]),
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// POOT-646: the routed gate weighting fused over checkpoint-order experts (`ContractionEpilogue { head: Indexed }`
/// reading the `[E, N, K]` weight in place) equals the oracle `to_bits`.
#[test]
fn routed_gate_weighting_checkpoint_order_equals_the_oracle_bit_for_bit() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    for ids in [vec![6, 0, 6, 3, 1], shuffled_ids(160)] {
        // Dyadic gates, so the product with an exact integer sum is exact.
        let gate: Vec<f32> = (0..ids.len())
            .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
            .collect();
        let p = checkpoint_program_gated(&ids, Some(gate));
        let got = run(&mut engine, target, &p);
        assert_eq!(bits(&got), bits(&oracle(&p)), "M={}", ids.len());
    }
}

/// POOT-1028: the BF16 expert stack as a BF16 const, `[E, K, N]` or (`checkpoint_order`) the checkpoint's
/// `[E, N, K]` read through `transpose(w, [0, 2, 1])`, with an f32 activation and output; with `gate`, the routed
/// gate weighting after the product (`ContractionEpilogue`). The fill is integer-valued, so every element is exactly
/// a BF16 and the oracle's widened sum is the device's sum bit for bit.
fn bf16_program(ids: &[i32], checkpoint_order: bool, gate: Option<Vec<f32>>) -> Program {
    let m = ids.len();
    let experts = CHECKPOINT_EXPERTS;
    let words = |values: Vec<f32>| -> Vec<u16> {
        values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
    };
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, K])).unwrap();
    let idx = b
        .constant_ranged(
            "idx",
            TensorType::new(vec![m], DType::I32),
            poot_graph_ir::IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(x.id(), HostTensor::f32(vec![m, K], int_fill(1, m * K)));
    inputs.insert(idx.id(), HostTensor::i32(vec![m], ids.to_vec()));
    let stored = if checkpoint_order {
        vec![experts, N, K]
    } else {
        vec![experts, K, N]
    };
    let w = b
        .constant(
            "w",
            TensorType::new(stored.clone(), poot_tensor::DType::BF16),
        )
        .unwrap();
    inputs.insert(
        w.id(),
        HostTensor::bf16(stored.clone(), words(int_fill(2, experts * K * N))),
    );
    let weight = if checkpoint_order {
        b.transpose(w, vec![0, 2, 1]).unwrap()
    } else {
        w
    };
    let mut out = b.indexed_matmul(x, weight, idx).unwrap();
    if let Some(gate) = gate {
        let gate_t = b.constant("gate", TensorType::f32(vec![m])).unwrap();
        inputs.insert(gate_t.id(), HostTensor::f32(vec![m], gate));
        out = b
            .binary(BinOp::Mul, out, b.reshape(gate_t, vec![m, 1]).unwrap())
            .unwrap();
    }
    Program {
        graph: b.finish(out).unwrap(),
        inputs,
    }
}

/// POOT-1028: `IndexedMatMul` over a BF16-resident expert stack equals the oracle `to_bits` on every element, in
/// both orientations, with and without the gate epilogue, at a decode shape (the Gemv) and a prefill shape whose
/// tiles cross expert boundaries (the register-tiled expert walk): the packed lanes are widened in-register and no f32 copy of
/// the stack exists.
///
/// Mutation (`WeightLane::decode_low_half`, `poot-kernelgen/src/helpers.rs`): shift the BF16 bits by 15 instead of
/// 16 when widening; every case goes red.
#[test]
fn indexed_matmul_bf16_experts_match_the_oracle_bit_for_bit() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    let decode = vec![6, 0, 6, 3, 1];
    let prefill: Vec<i32> = (0..160)
        .map(|r| [7, 2, 2, 5, 0, 7, 3][(r * 5) % 7])
        .collect();
    let mut failures = Vec::new();
    for (name, ids) in [("M=5", decode), ("M=160, shuffled", prefill)] {
        for checkpoint_order in [false, true] {
            for gated in [false, true] {
                // Dyadic gates, so the product with an exact integer sum is exact.
                let gate = gated.then(|| {
                    (0..ids.len())
                        .map(|r| [0.5, -2.0, 1.25, 3.0, -0.75][r % 5])
                        .collect()
                });
                let p = bf16_program(&ids, checkpoint_order, gate);
                let got = run(&mut engine, target, &p);
                let (got_bits, want_bits) = (bits(&got), bits(&oracle(&p)));
                if let Some(first) = (0..got_bits.len()).find(|&i| got_bits[i] != want_bits[i]) {
                    failures.push(format!(
                        "{name} nk={checkpoint_order} gate={gated}: row {} col {} is {} on the device, {} in the oracle",
                        first / N,
                        first % N,
                        f32::from_bits(got_bits[first]),
                        f32::from_bits(want_bits[first]),
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
