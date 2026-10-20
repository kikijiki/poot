//! SC-001: for a one-equation graph over every `OpKind` family and every dtype tuple `infer` admits,
//! `eval` returns the output aval's dtype and shape, never panics, and the same bits come back under
//! every `EvalOptions` combination this walk offers (an observer attached or not, `keep_environment`
//! set or not).
//!
//! SC-005 (card 555): `Cast` used to admit every `(from, to)` pair at `infer` time and
//! refuse nine of them only at `eval` time (the walk's declared `REFUSED_CASTS` table). Those nine
//! pairs are now refused by `infer` itself as a typed `ShapeError::CastUnsupported`, so a graph
//! naming one can no longer be built; `cast_infer_refuses_the_nine_unevaluated_pairs` below is the one
//! exception table, and `every_infer_admitted_pair_evaluates` runs with no exception set at all - every
//! pair `infer` admits, this walk evaluates.
//!
//! Scope: every `OpKind` variant `Builder` exposes gets a row here, covering the dtype tuples that
//! make this walk dispatch differently (F32 vs I32 for the word-level ops; the full `Cast` dtype x
//! dtype sweep), plus the compiler-only `ContractionEpilogue` (staged through an append plan). The other
//! compiler-only variants (`DenseContraction`, `DenseRowGather`, `PackedDequant`,
//! `PackedContraction`, `PackedRowGather`, `Fused`, `FusedRow`) and the ops with one fixed, intricate
//! operand shape (`Rope`, `Attention`) already have dedicated
//! correctness suites elsewhere (`exact_bf16.rs`,
//! `packed_dequant.rs`, `fuzz_primitives.rs`, `cast_quant_norm_rope.rs`, `alibi_attention.rs`,
//! `flash_optimize.rs`) that exercise their one real shape repeatedly; this file does not duplicate
//! them. Reused by Cards 555 and 556 per the card's own note - extend the `CASES` table below rather
//! than writing a parallel corpus.

//!
//! SC-001/SC-003/SC-004 (card 555): the index rule, movement engine and index-consuming op rows live
//! here too, in the three functions at the end of this file.

use poot_tensor::DType;
use std::collections::HashMap;
use std::num::NonZeroU32;

use poot_graph_ir::op::{ContractionHead, FusedOp, FusedOperand, FusedRegion, FusedStep};
use poot_graph_ir::{BinOp, Builder, OpKind, Operand, RedOp, StateRole, TensorType, UnOp};

use crate::exact_value::ExactValue;
use crate::observer::EvalObserver;
use crate::ops;
use crate::{EvalError, EvalOptions, Value, eval};
use poot_graph_ir::test_support::graph_fixtures;
use poot_tensor::HostTensor;

/// A no-op observer that proves the walk tolerates one being attached without changing any bit it
/// publishes (SC-001's "observer on or off" clause). It only needs the default (no-op) trait methods.
struct NoOpObserver;
impl EvalObserver for NoOpObserver {}

/// Evaluate `graph`/`inputs` under every `EvalOptions` combination SC-001 names, asserting they all
/// agree bit for bit (`ADR-0101`) and none panics. Returns the shared output on success, or the
/// shared error (asserted identical in `Display` text across every combination) on failure.
fn eval_every_combination(
    label: &str,
    graph: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, Value>,
) -> Result<Value, EvalError> {
    let plain = eval(graph, inputs, EvalOptions::default()).map(|r| r.output);

    let mut observer = NoOpObserver;
    let observed = eval(
        graph,
        inputs,
        EvalOptions::default().observer(&mut observer),
    )
    .map(|r| r.output);

    let kept = eval(graph, inputs, EvalOptions::default().keep_environment()).map(|r| r.output);

    let combos: Vec<(&str, &Result<Value, EvalError>)> =
        vec![("observer", &observed), ("keep_environment", &kept)];

    match &plain {
        Ok(value) => {
            let bits = value_bits(value);
            assert_digest(label, value, &bits);
            for (combo_name, combo) in combos {
                let combo_value = combo.as_ref().unwrap_or_else(|error| {
                    panic!("{label}: default combo succeeded but {combo_name} failed: {error}")
                });
                assert_eq!(
                    value_bits(combo_value),
                    bits,
                    "{label}: {combo_name} published different bits than the default combo"
                );
            }
        }
        Err(error) => {
            let message = error.to_string();
            for (combo_name, combo) in combos {
                let combo_error = combo.as_ref().err().unwrap_or_else(|| {
                    panic!("{label}: default combo failed but {combo_name} succeeded")
                });
                assert_eq!(
                    combo_error.to_string(),
                    message,
                    "{label}: {combo_name} reported a different error than the default combo"
                );
            }
        }
    }
    plain
}

/// The published bits of every row this corpus evaluates, recorded at Card 721's dispatch base
/// (`e76f40c1b`, before the typed host carrier moved below the oracle): `label`, then an FNV-1a digest of
/// the output shape, a separator and the logical bits (`value_bits`). The carrier change must not move a
/// bit of any row (SC-002).
const BASE_DIGESTS: &str = include_str!("admission_corpus_digest.tsv");

fn bits_digest(shape: &[usize], bits: &[u32]) -> u64 {
    let words = shape
        .iter()
        .map(|&dim| dim as u32)
        .chain([u32::MAX])
        .chain(bits.iter().copied());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

fn assert_digest(label: &str, value: &Value, bits: &[u32]) {
    let recorded = BASE_DIGESTS
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .find(|(name, _)| *name == label)
        .unwrap_or_else(|| panic!("{label}: no digest recorded at the dispatch base"))
        .1;
    let observed = format!("{:016x}", bits_digest(&value.shape(), bits));
    assert_eq!(
        observed, recorded,
        "{label}: published bits differ from the digest recorded at the dispatch base"
    );
}

/// A representation-agnostic comparison of one value's published bits (ADR-0101: every element),
/// dtype-semantic so a row's recorded digest is independent of the carrier's storage class: I32 words
/// as `u32`, F32/BF16/F16 values widened to f32 (`to_f32` is exact) as their bits, E4M3FN as its raw
/// bytes. Every `Value::Exact` kind too: the Spec-376 BF16 lane and the
/// exact-I32 lane both publish `Value::Exact`, not just `Value::Host`.
fn value_bits(value: &Value) -> Vec<u32> {
    match value {
        Value::Host(tensor) => host_bits(tensor),
        Value::Exact(ExactValue::I32(view)) => view.i32_words().iter().map(|&i| i as u32).collect(),
        Value::Exact(exact) => host_bits(&exact.materialize_dense()),
        other => panic!("value_bits: unexpected published carrier {other:?}"),
    }
}

fn host_bits(tensor: &HostTensor) -> Vec<u32> {
    match tensor.dtype() {
        DType::I32 => tensor
            .as_i32()
            .expect("an I32 tensor holds words")
            .iter()
            .map(|&i| i as u32)
            .collect(),
        DType::F32 | DType::BF16 | DType::F16 => tensor
            .to_f32()
            .expect("a float tensor widens exactly")
            .iter()
            .map(|f| f.to_bits())
            .collect(),
        // I8 is two's-complement bytes: the sign-extended value, as the I32 word it was cast from.
        DType::I8 => tensor
            .view()
            .bytes()
            .iter()
            .map(|&b| i32::from(b as i8) as u32)
            .collect(),
        DType::E4M3FN => tensor
            .view()
            .bytes()
            .iter()
            .map(|&b| u32::from(b))
            .collect(),
        other => panic!("value_bits: no recorded bit semantics for a {other} tensor"),
    }
}

fn dense_inputs<const N: usize>(
    pairs: [(poot_graph_ir::ValueId, HostTensor); N],
) -> HashMap<poot_graph_ir::ValueId, Value> {
    pairs
        .into_iter()
        .map(|(id, t)| (id, Value::from(t)))
        .collect()
}

/// A BF16 tensor of `values` (each narrowed once, round-to-nearest-even): what a BF16 const binds,
/// since the walk does not convert an F32 tensor at bind.
fn bf16_host(shape: Vec<usize>, values: &[f32]) -> HostTensor {
    ops::cast::narrow_f32(shape, values, DType::BF16).unwrap()
}

/// As [`bf16_host`], for an F16 const.
fn f16_host(shape: Vec<usize>, values: &[f32]) -> HostTensor {
    ops::cast::narrow_f32(shape, values, DType::F16).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Unary / Binary / Select: F32 and I32 are genuinely different dispatch paths (word ops vs float
// ops); every other arithmetic dtype (BF16/F16) shares the F32 path through `to_f32`.
// ---------------------------------------------------------------------------------------------

#[test]
fn unary_admits_f32_and_i32_word_forms() {
    // F32 arithmetic unary (Neg): admitted for every arithmetic dtype, not just F32.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![3])).unwrap();
    let y = b.unary(UnOp::Neg, x).unwrap();
    let g = b.finish(y).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![3], vec![1.0, -2.0, 0.0]))]);
    let out = eval_every_combination("unary f32 neg", &g, &inputs).unwrap();
    let Value::Host(t) = out else {
        panic!("expected dense")
    };
    assert_eq!(t.shape(), vec![3]);
    assert_eq!(t.as_f32().unwrap(), &[-1.0, 2.0, -0.0]);

    // I32 word unary (Not): admitted only for I32, and only Not/Clz.
    let b = Builder::for_tests();
    let x = graph_fixtures::i32_constant(&b, "x", vec![2]).unwrap();
    let y = b.unary(UnOp::Not, x).unwrap();
    let g = b.finish(y).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::i32(vec![2], vec![0, -1]))]);
    let out = eval_every_combination("unary i32 not", &g, &inputs).unwrap();
    let Value::Host(t) = out else {
        panic!("expected dense")
    };
    assert_eq!(t.shape(), vec![2]);
    assert_eq!(t.as_i32(), Some(&[-1, 0][..]));
}

#[test]
fn binary_admits_f32_arithmetic_and_i32_word_forms() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![2])).unwrap();
    let z = b.binary(BinOp::Add, x, y).unwrap();
    let g = b.finish(z).unwrap();
    let inputs = dense_inputs([
        (x.id(), HostTensor::f32(vec![2], vec![1.0, 2.0])),
        (y.id(), HostTensor::f32(vec![2], vec![3.0, 4.0])),
    ]);
    let out = eval_every_combination("binary f32 add", &g, &inputs).unwrap();
    let Value::Host(t) = out else {
        panic!("expected dense")
    };
    assert_eq!(t.as_f32().unwrap(), &[4.0, 6.0]);

    // I32 word binary, including the unsigned-only forms `infer` admits exclusively for I32.
    for (op, a, bb, want) in [
        (BinOp::Add, [1, -1], [1, 1], [2, 0]),
        (BinOp::GeU, [-1, 1], [1, -1], [1, 0]),
        (BinOp::RemU, [7, -7], [3, 3], [1, 0]),
    ] {
        let b = Builder::for_tests();
        let x = graph_fixtures::i32_constant(&b, "x", vec![2]).unwrap();
        let y = graph_fixtures::i32_constant(&b, "y", vec![2]).unwrap();
        let z = b.binary(op, x, y).unwrap();
        let g = b.finish(z).unwrap();
        let inputs = dense_inputs([
            (x.id(), HostTensor::i32(vec![2], a.to_vec())),
            (y.id(), HostTensor::i32(vec![2], bb.to_vec())),
        ]);
        let out = eval_every_combination(&format!("binary i32 {op:?}"), &g, &inputs).unwrap();
        let Value::Host(t) = out else {
            panic!("expected dense")
        };
        assert_eq!(t.as_i32(), Some(&want[..]), "{op:?}");
    }
}

#[test]
fn select_evaluates_i32_branches_through_every_binding() {
    let b = Builder::for_tests();
    let cond = graph_fixtures::i32_constant(&b, "cond", vec![2]).unwrap();
    let t = graph_fixtures::i32_constant(&b, "t", vec![2]).unwrap();
    let f = graph_fixtures::i32_constant(&b, "f", vec![2]).unwrap();
    let out = b.select(cond, t, f).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (cond.id(), HostTensor::i32(vec![2], vec![1, 0])),
        (t.id(), HostTensor::i32(vec![2], vec![10, 10])),
        (f.id(), HostTensor::i32(vec![2], vec![20, 20])),
    ]);
    let out = eval_every_combination("select i32", &g, &inputs).unwrap();
    let Value::Host(tensor) = out else {
        panic!("expected dense")
    };
    assert_eq!(tensor.as_i32(), Some(&[10, 20][..]));
}

// ---------------------------------------------------------------------------------------------
// Reduce / Broadcast / Reshape / Transpose / Slice / Concat / Iota: shape-level ops, F32.
// ---------------------------------------------------------------------------------------------

#[test]
fn reduce_broadcast_reshape_transpose_slice_concat_iota_evaluate() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let out = b.reduce(RedOp::Sum, x, 1, false).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(
        x.id(),
        HostTensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )]);
    let out = eval_every_combination("reduce sum", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[6.0, 15.0]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1])).unwrap();
    let out = b.broadcast(x, vec![3]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![1], vec![7.0]))]);
    let out = eval_every_combination("broadcast", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[7.0, 7.0, 7.0]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let out = b.reshape(x, vec![3, 2]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(
        x.id(),
        HostTensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )]);
    let out = eval_every_combination("reshape", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().shape(), vec![3, 2]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
    let out = b.transpose(x, vec![1, 0]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(
        x.id(),
        HostTensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )]);
    let out = eval_every_combination("transpose", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().shape(), vec![3, 2]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let out = b.slice(x, 0, 1, 3).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]))]);
    let out = eval_every_combination("slice", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[2.0, 3.0]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![2])).unwrap();
    let out = b.concat(0, &[x, y]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (x.id(), HostTensor::f32(vec![2], vec![1.0, 2.0])),
        (y.id(), HostTensor::f32(vec![2], vec![3.0, 4.0])),
    ]);
    let out = eval_every_combination("concat", &g, &inputs).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let b = Builder::for_tests();
    let out = b.iota(4).unwrap();
    let g = b.finish(out).unwrap();
    let out = eval_every_combination("iota", &g, &HashMap::new()).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[0.0, 1.0, 2.0, 3.0]
    );
}

// ---------------------------------------------------------------------------------------------
// Gather / Scatter / ScatterUpdate / MatMul / ContractionEpilogue / DynamicUpdateSlice / PackI8 / UnpackI8 /
// ArgTopK / IndexedMatMul / AllReduce / AllGather.
// ---------------------------------------------------------------------------------------------

#[test]
fn gather_scatter_scatter_update_evaluate() {
    let b = Builder::for_tests();
    let table = b.constant("table", TensorType::f32(vec![3, 2])).unwrap();
    let index = graph_fixtures::i32_constant(&b, "index", vec![2]).unwrap();
    let out = b.gather(table, 0, index).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (
            table.id(),
            HostTensor::f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (index.id(), HostTensor::i32(vec![2], vec![2, 0])),
    ]);
    let out = eval_every_combination("gather", &g, &inputs).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[5.0, 6.0, 1.0, 2.0]
    );

    let b = Builder::for_tests();
    let src = b.constant("src", TensorType::f32(vec![2, 2])).unwrap();
    let index = graph_fixtures::i32_constant(&b, "index", vec![2]).unwrap();
    let out = b.scatter(src, index).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (
            src.id(),
            HostTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (index.id(), HostTensor::i32(vec![2], vec![1, 0])),
    ]);
    let out = eval_every_combination("scatter", &g, &inputs).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[3.0, 4.0, 1.0, 2.0]
    );

    let b = Builder::for_tests();
    let base = b.constant("base", TensorType::f32(vec![3, 1])).unwrap();
    let src = b.constant("src", TensorType::f32(vec![2, 1])).unwrap();
    let slots = graph_fixtures::i32_constant(&b, "slots", vec![2]).unwrap();
    let out = b.scatter_update(base, src, slots).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (base.id(), HostTensor::f32(vec![3, 1], vec![0.0, 0.0, 0.0])),
        (src.id(), HostTensor::f32(vec![2, 1], vec![9.0, 8.0])),
        (slots.id(), HostTensor::i32(vec![2], vec![2, 1])),
    ]);
    let out = eval_every_combination("scatter_update", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[0.0, 8.0, 9.0]);
}

#[test]
fn matmul_contraction_epilogue_indexed_matmul_evaluate() {
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 2])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![2, 2])).unwrap();
    let out = b.matmul(a, w).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (a.id(), HostTensor::f32(vec![1, 2], vec![1.0, 2.0])),
        (
            w.id(),
            HostTensor::f32(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        ),
    ]);
    let out = eval_every_combination("matmul", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[1.0, 2.0]);

    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 2])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![2, 2])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![2])).unwrap();
    // `ContractionEpilogue` is compiler-only (the fuse rule forms it), so this row stages it through an
    // append plan to evaluate the op itself: the matmul, then `head + bias`.
    let mut plan = b.append_plan(0).unwrap();
    let staged = plan
        .equation(
            OpKind::ContractionEpilogue {
                head: ContractionHead::MatMul,
                region: FusedRegion {
                    n_inputs: 2,
                    steps: vec![FusedStep {
                        op: FusedOp::Binary(BinOp::Add),
                        inputs: vec![FusedOperand::Local(0), FusedOperand::Local(1)],
                    }],
                    output: 2,
                    pack: Vec::new(),
                },
            },
            vec![
                Operand::Value(a.id()),
                Operand::Value(w.id()),
                Operand::Value(bias.id()),
            ],
        )
        .expect("stage ContractionEpilogue");
    plan.declare_result(staged).expect("declare result");
    let mut prepared = b.preflight_append(plan).expect("preflight append");
    let out = b.commit_append(&mut prepared).expect("commit append");
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (a.id(), HostTensor::f32(vec![1, 2], vec![1.0, 2.0])),
        (
            w.id(),
            HostTensor::f32(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        ),
        (bias.id(), HostTensor::f32(vec![2], vec![10.0, 20.0])),
    ]);
    let out = eval_every_combination("contraction_epilogue", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[11.0, 22.0]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 2])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![2, 2, 2])).unwrap();
    let idx = graph_fixtures::i32_constant(&b, "idx", vec![1]).unwrap();
    let out = b.indexed_matmul(x, w, idx).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (x.id(), HostTensor::f32(vec![1, 2], vec![1.0, 2.0])),
        (
            w.id(),
            HostTensor::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        ),
        (idx.id(), HostTensor::i32(vec![1], vec![1])),
    ]);
    let out = eval_every_combination("indexed_matmul", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[2.0, 1.0]);
}

#[test]
fn dynamic_update_slice_pack_unpack_i8_arg_top_k_evaluate() {
    let b = Builder::for_tests();
    let base = b.constant("base", TensorType::f32(vec![4])).unwrap();
    let update = b.constant("update", TensorType::f32(vec![2])).unwrap();
    let out = b.dynamic_update_slice(base, update, 1, 0).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (
            base.id(),
            HostTensor::f32(vec![4], vec![0.0, 0.0, 0.0, 0.0]),
        ),
        (update.id(), HostTensor::f32(vec![2], vec![9.0, 8.0])),
    ]);
    let out = eval_every_combination("dynamic_update_slice", &g, &inputs).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[0.0, 9.0, 8.0, 0.0]
    );

    let b = Builder::for_tests();
    let x = graph_fixtures::i32_constant(&b, "x", vec![4]).unwrap();
    let packed = b.pack_i8(x).unwrap();
    let g = b.finish(packed).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::i32(vec![4], vec![1, 2, 3, 4]))]);
    let out = eval_every_combination("pack_i8", &g, &inputs).unwrap();
    assert!(out.as_host().unwrap().as_i32().is_some());

    let b = Builder::for_tests();
    let x = graph_fixtures::i32_constant(&b, "x", vec![4]).unwrap();
    let packed = b.pack_i8(x).unwrap();
    let out = b.unpack_i8(packed, 4).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::i32(vec![4], vec![1, 2, 3, 4]))]);
    let out = eval_every_combination("unpack_i8", &g, &inputs).unwrap();
    assert_eq!(
        out.as_host().unwrap().as_f32().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    // ArgTopK's input is a precomputed per-element RANK (0 = best), not a raw score: `out[r] = i` for
    // every element `i` whose rank `r` is `< k`. Element 1 has rank 0 and element 3 has rank 1, so
    // `out = [1, 3]`.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let out = b.arg_top_k(x, 2).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![4], vec![2.0, 0.0, 3.0, 1.0]))]);
    let out = eval_every_combination("arg_top_k", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_i32().unwrap(), &[1, 3]);
}

#[test]
fn all_reduce_all_gather_are_the_single_rank_identity() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![3])).unwrap();
    let out = b.all_reduce(x, RedOp::Sum, 0).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![3], vec![1.0, 2.0, 3.0]))]);
    let out = eval_every_combination("all_reduce", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[1.0, 2.0, 3.0]);

    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![3])).unwrap();
    let out = b.all_gather(x, 0, NonZeroU32::MIN).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![3], vec![1.0, 2.0, 3.0]))]);
    let out = eval_every_combination("all_gather", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[1.0, 2.0, 3.0]);
}

/// POOT-747: the oracle holds one rank, so an `AllGather` over more than one rank is a typed
/// refusal on every option combination, never the input returned as if it were the gathered tensor.
#[test]
fn all_gather_over_more_than_one_rank_is_refused() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![3])).unwrap();
    let out = b.all_gather(x, 0, NonZeroU32::new(2).unwrap()).unwrap();
    let eqn = out.id();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![3], vec![1.0, 2.0, 3.0]))]);
    match eval_every_combination("all_gather_world_2", &g, &inputs) {
        Err(EvalError::Unsupported {
            eqn: refused,
            op: "all_gather",
            ..
        }) => assert_eq!(refused, eqn),
        other => panic!("expected the all_gather refusal, got {other:?}"),
    }
}

/// Card 551a: `RandomUniform` and every `SampleToken` rule evaluate without panicking,
/// return I32/F32 as `infer` declares, and agree with every `EvalOptions` combination (SC-001).
/// Bit-exact correctness (ties, the non-finite flag, the Gumbel formula) is covered by the dedicated
/// sampling-primitives suite (`poot-graph-plan`'s `sampler_bodies` tests and `pootc`'s
/// `import_run.rs` device rows); this corpus entry only proves every rule type-checks and evaluates.
#[test]
fn random_uniform_and_sample_token_evaluate() {
    let b = Builder::for_tests();
    let seed = graph_fixtures::i32_constant(&b, "seed", vec![2]).unwrap();
    let out = b.random_uniform(seed, 3).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(seed.id(), HostTensor::i32(vec![2], vec![1, 2]))]);
    let out = eval_every_combination("random_uniform", &g, &inputs).unwrap();
    let data = out.as_host().unwrap();
    assert_eq!(data.shape(), vec![2, 3]);
    for &u in data.as_f32().unwrap().iter() {
        assert!(u > 0.0 && u < 1.0, "u={u} must stay strictly inside (0,1)");
    }

    let b = Builder::for_tests();
    let logits = b.constant("logits", TensorType::f32(vec![2, 5])).unwrap();
    let out = b
        .sample_token(
            poot_graph_ir::op::SampleRule::Greedy,
            logits,
            None,
            None,
            None,
        )
        .unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(
        logits.id(),
        HostTensor::f32(
            vec![2, 5],
            vec![1.0, 2.0, 3.0, 0.0, -1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ),
    )]);
    let out = eval_every_combination("sample_token_greedy", &g, &inputs).unwrap();
    let data = out.as_host().unwrap();
    assert_eq!(data.shape(), vec![2, 2]);
    assert_eq!(data.as_i32(), Some([2i32, -1, 0, -1].as_slice()));

    for (rule, cols) in [
        (poot_graph_ir::op::SampleRule::Gumbel, 3),
        (poot_graph_ir::op::SampleRule::GumbelTopK, 3),
        (poot_graph_ir::op::SampleRule::GumbelTopKTopP, 4),
    ] {
        let b = Builder::for_tests();
        let logits = b.constant("logits", TensorType::f32(vec![1, 4])).unwrap();
        let noise = b.constant("noise", TensorType::f32(vec![1, 4])).unwrap();
        let params = b
            .constant("params", TensorType::f32(vec![1, cols]))
            .unwrap();
        let top_k = if matches!(
            rule,
            poot_graph_ir::op::SampleRule::GumbelTopK
                | poot_graph_ir::op::SampleRule::GumbelTopKTopP
        ) {
            Some(graph_fixtures::i32_constant(&b, "top_k", vec![1]).unwrap())
        } else {
            None
        };
        let out = b
            .sample_token(rule, logits, Some(noise), Some(params), top_k)
            .unwrap();
        let g = b.finish(out).unwrap();
        let mut params_data = vec![1.0f32, 0.0, 1.0];
        if cols == 4 {
            params_data.push(1.0); // top_p: disabled
        }
        let mut inputs: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
        inputs.insert(
            logits.id(),
            Value::from(HostTensor::f32(vec![1, 4], vec![1.0, 2.0, 3.0, 0.5])),
        );
        inputs.insert(
            noise.id(),
            Value::from(HostTensor::f32(vec![1, 4], vec![0.0, 0.0, 0.0, 0.0])),
        );
        inputs.insert(
            params.id(),
            Value::from(HostTensor::f32(vec![1, cols], params_data)),
        );
        if let Some(top_k) = top_k {
            inputs.insert(top_k.id(), Value::from(HostTensor::i32(vec![1], vec![0])));
        }
        let out = eval_every_combination(&format!("sample_token_{rule:?}"), &g, &inputs).unwrap();
        let data = out.as_host().unwrap();
        assert_eq!(data.shape(), vec![1, 2]);
        assert_eq!(
            data.as_i32(),
            Some([2i32, -1].as_slice()),
            "rule={rule:?}: zero noise reduces every Gumbel-family rule to plain argmax"
        );
    }
}

/// SC-003/SC-006: `Builder::resume` on a finished graph, then `sample_head(Greedy)` and
/// `finish_with_state`: the token equals the oracle's `SampleToken` on the same graph's logits, and
/// the state pair is unchanged (carried through, not dropped). Mutation (recorded, not left in the
/// tree): drop `state: &resumed.outputs.state` from the second `finish_with_state` call (pass `&[]` instead) -
/// `g2.state` goes empty and the row's state-output assertion fails (the row goes red, matching the
/// card's named mutation "resume drops the state pairs; the second step diverges").
#[test]
fn resume_appends_a_greedy_sample_head_and_keeps_state_unchanged() {
    let b = Builder::for_tests();
    let logits = b.constant("logits", TensorType::f32(vec![4])).unwrap();
    let cache_in = b
        .state_input("cache", TensorType::f32(vec![4]), StateRole::Recurrent)
        .unwrap();
    let g = b
        .finish(poot_graph_ir::Outputs::new(logits).with_state(&[(cache_in, cache_in)]))
        .unwrap();

    let resumed = Builder::resume(g, poot_graph_ir::TraceLimits::for_tests()).unwrap();
    let head = poot_graph_ir::ops::sampling::sample_head(
        &resumed.builder,
        resumed.outputs.result,
        poot_graph_ir::ops::sampling::SampleSpec::draw(poot_graph_ir::op::SampleRule::Greedy),
        None,
    )
    .unwrap();
    let g2 = resumed
        .builder
        .finish(poot_graph_ir::Outputs::new(head.out).with_state(&resumed.outputs.state))
        .unwrap();
    assert_eq!(
        g2.state().collect::<Vec<_>>(),
        vec![(cache_in.id(), cache_in.id())],
        "resume must carry the state pair through unchanged into the re-finished graph"
    );

    let cache_data = vec![9.0f32, 8.0, 7.0, 6.0];
    let inputs = dense_inputs([
        (
            logits.id(),
            HostTensor::f32(vec![4], vec![1.0, 5.0, 3.0, 2.0]),
        ),
        (cache_in.id(), HostTensor::f32(vec![4], cache_data.clone())),
    ]);
    let result = eval(&g2, &inputs, EvalOptions::default()).unwrap();
    let token_out = result.output.as_host().unwrap();
    assert_eq!(
        token_out.as_i32(),
        Some([1i32, -1].as_slice()),
        "sample_head(Greedy) on [1,5,3,2] must pick index 1 (the max), non_finite -1"
    );
    assert_eq!(
        result.state.len(),
        1,
        "the resumed graph's one state pair must still publish a state output"
    );
    assert_eq!(
        result.state[0].as_host().unwrap().as_f32().unwrap(),
        cache_data.as_slice(),
        "an untouched state pair (state_out == state_in) must publish the input unchanged"
    );
}

// ---------------------------------------------------------------------------------------------
// Cast: the full dtype x dtype sweep, matching `infer`'s X9 refusal table exactly (card 555
// deleted `resolve::REFUSED_CASTS`; the nine refused pairs now live in
// `poot_graph_ir::OpKind::Cast::infer` as `REFUSED_AT_INFER` mirrors below).
// ---------------------------------------------------------------------------------------------

/// A literal value of `dtype`'s own storage, for the one-element `x` constant each cast pair binds.
fn literal_input(dtype: DType) -> Value {
    match dtype {
        DType::F32 => Value::from(HostTensor::f32(vec![1], vec![2.0])),
        DType::BF16 => Value::from(bf16_host(vec![1], &[2.0])),
        DType::F16 => Value::from(f16_host(vec![1], &[2.0])),
        DType::I32 | DType::I8 => Value::from(HostTensor::i32(vec![1], vec![2])),
        DType::E4M3FN => Value::Host(crate::fp8::encode_e4m3fn_tensor(vec![1], &[2.0]).unwrap()),
        DType::Bool
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64
        | DType::E8M0 => unreachable!("{dtype:?} is not in the DTYPES sweep below"),
    }
}

/// Cast pairs `infer` admits but neither has a direct oracle definition (X3 is a value-level fault,
/// not an admission gap): every pair not in [`REFUSED_AT_INFER`].
const DTYPES: [DType; 6] = [
    DType::F32,
    DType::BF16,
    DType::F16,
    DType::I32,
    DType::I8,
    DType::E4M3FN,
];

/// The nine pairs `infer` itself refuses (card 555's `ShapeError::CastUnsupported`): a
/// graph naming one of these can no longer be built at all, so they are no longer part of the
/// Builder-level sweep below.
const REFUSED_AT_INFER: &[(DType, DType)] = &[
    (DType::BF16, DType::I32),
    (DType::F16, DType::I32),
    (DType::BF16, DType::I8),
    (DType::F16, DType::I8),
    (DType::I32, DType::E4M3FN),
    (DType::I8, DType::E4M3FN),
    (DType::E4M3FN, DType::I32),
    (DType::E4M3FN, DType::I8),
    (DType::F32, DType::I8),
];

/// SC-005: `infer` rejects exactly the nine X9 pairs, each as a typed `ShapeError::CastUnsupported`
/// naming the pair; `OpKind::Cast::infer` on any other pair in [`DTYPES`] (including every identity
/// pair) succeeds. Mutation: re-admit `BF16 -> I32` (drop it from `REFUSED_AT_INFER`): this test's row
/// goes red (`infer` returns `Ok`, not the expected `CastUnsupported`).
#[test]
fn cast_infer_refuses_the_nine_unevaluated_pairs() {
    for &from in &DTYPES {
        for &to in &DTYPES {
            let ins = [TensorType::new(vec![1], from)];
            let result = poot_graph_ir::OpKind::Cast { to }.infer(&ins);
            let refused = REFUSED_AT_INFER.contains(&(from, to));
            match (result, refused) {
                (Ok(_), false) => {}
                (
                    Err(poot_graph_ir::ShapeError::CastUnsupported {
                        from: got_from,
                        to: got_to,
                    }),
                    true,
                ) => {
                    assert_eq!((got_from, got_to), (from, to));
                }
                (Ok(aval), true) => {
                    panic!("cast {from:?} -> {to:?}: expected X9 refusal, got {aval:?}")
                }
                (Err(error), false) => {
                    panic!("cast {from:?} -> {to:?}: expected admission, got {error}")
                }
                (Err(error), true) => {
                    panic!("cast {from:?} -> {to:?}: expected CastUnsupported, got {error:?}")
                }
            }
        }
    }
}

/// The admitted-pair half of the old combined sweep: every pair `infer` admits (i.e. every pair not
/// in [`REFUSED_AT_INFER`]) builds and evaluates successfully through the one walk, under every
/// `EvalOptions` combination.
///
/// I8 is a packed byte-code carrier with no scalar `Builder` constant, so an I8 *source* can only be
/// bound as a `Value::Packed` - never the plain one-element graph this sweep binds - and is
/// skipped here; [`cast_i8_source_chain_rows_evaluate`] below covers the four I8-source pairs this
/// card moved out of `REFUSED_CASTS` the way a real graph actually produces an I8 value (a prior
/// `Cast(I32, I8)`), so the skip no longer leaves that whole source side untested (card 555).
#[test]
fn cast_sweeps_every_admitted_dtype_pair() {
    for &from in &DTYPES {
        for &to in &DTYPES {
            if from == DType::I8 {
                continue;
            }
            if REFUSED_AT_INFER.contains(&(from, to)) {
                continue;
            }
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::new(vec![1], from)).unwrap();
            let y = b.cast(x, to).unwrap();
            let g = b.finish(y).unwrap();
            let inputs: HashMap<_, Value> = HashMap::from([(x.id(), literal_input(from))]);
            let label = format!("cast {from:?} -> {to:?}");
            eval_every_combination(&label, &g, &inputs)
                .unwrap_or_else(|error| panic!("{label}: expected success, got {error}"));
        }
    }
}

/// Card 555: the four I8-source cast pairs (`I8 -> {F32, BF16, F16, I32}`), moved out
/// of `REFUSED_CASTS` by this card, with the I8 value produced inside the graph (`Cast(I32, I8)`),
/// the only way a real graph ever binds one (a bound I8 *input* is a `Value::Packed`, not
/// this dense carrier at all). `127`/`-3` are exact in every one of these dtypes, so each row's
/// expected value is the literal integer back.
///
/// Mutation (recorded): in `walk.rs`'s `(I32 | I8, BF16 | F16)` arm, widen through `words[i] as f32`
/// but skip the per-word X3 range fault loop (an always-succeeds `Ok` filter). Confirmed red: a fifth
/// row added to this test with word `20_000_000` (outside `F32_EXACT_I32_MAX`) then expects
/// `EvalError::Cast` and instead gets `Ok` (silently rounds); reverted, confirmed green.
#[test]
fn cast_i8_source_chain_rows_evaluate() {
    for (to, want) in [
        (DType::F32, [127.0f32, -3.0]),
        (DType::BF16, [127.0, -3.0]),
        (DType::F16, [127.0, -3.0]),
        (DType::I32, [127.0, -3.0]),
    ] {
        let b = Builder::for_tests();
        let x = graph_fixtures::i32_constant(&b, "x", vec![2]).unwrap();
        let i8 = b.cast(x, DType::I8).unwrap();
        let y = b.cast(i8, to).unwrap();
        let g = b.finish(y).unwrap();
        let inputs = dense_inputs([(x.id(), HostTensor::i32(vec![2], vec![127, -3]))]);
        let label = format!("I32 -> I8 -> {to:?}");
        let out = eval_every_combination(&label, &g, &inputs).unwrap();
        let bits = value_bits(&out);
        let got: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        if to == DType::I32 {
            // I32's own published bits are the raw word, not an f32 reinterpretation.
            assert_eq!(
                out.as_host().unwrap().as_i32(),
                Some(&[127, -3][..]),
                "{label}"
            );
        } else {
            assert_eq!(got, want, "{label}");
        }
    }
}

/// Card 555: `Cast(BF16, E4M3FN)` chained after a `Cast(F32, BF16)` - the BF16 value
/// is one the walk itself published (a BF16 host tensor of words), not an owner/exact or bound BF16
/// value. Every pair `infer` admits must evaluate on that value too, not only on the bound BF16
/// operands the rest of this file's rows use.
#[test]
fn cast_bf16_to_e4m3fn_evaluates_a_walk_published_bf16() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2])).unwrap();
    let bf16 = b.cast(x, DType::BF16).unwrap();
    let y = b.cast(bf16, DType::E4M3FN).unwrap();
    let g = b.finish(y).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![2], vec![1.0, 2.0]))]);
    let Value::Host(got) = eval(&g, &inputs, EvalOptions::default()).unwrap().output else {
        panic!("Cast(BF16, E4M3FN) must publish e4m3fn storage")
    };
    // 1.0 and 2.0 round-trip exactly through BF16 and encode to E4M3's exact 1.0/2.0 bytes.
    assert_eq!(got.view().bytes(), [0x38, 0x40]);
}

// ---------------------------------------------------------------------------------------------
// SC-001's own named row: the BF16/F16 dtype tuples `infer` admits for
// the families found reading a narrowed operand's empty `data` directly (`Gather` x2,
// `Scatter`, `IndexedMatMul`, `PackI8`, `Reshape`), plus the Spec-376 BF16 admission table's own
// `Gather` row (`resolve::classify`), whose mutation this file did not reach before (every row
// above binds a plain F32 table, never a BF16 one).
// ---------------------------------------------------------------------------------------------

/// SC-001's named row. Mutation: disable the `Gather` arm in `resolve::classify` (the card's own
/// named mutation - comment out its match arm, since a `-D dead-code`-clean literal deletion needs
/// the same no-op edit); the BF16-table/I32-index row below still evaluates (the I32-index Gather
/// fallback made robust to a narrowed table), but publishes
/// `Value::Host` instead of the Spec-376 lane's `Value::Exact(ExactValue::Bf16(_))` - a different
/// carrier for the same bits, so the `matches!` assertion goes red even though the bits agree.
#[test]
fn every_infer_admitted_pair_evaluates() {
    // Spec-376 lane: a BF16 table (narrowed at bind time from a plain f32 const)
    // gathered by I32 ids matches `resolve::classify`'s `Gather` row and publishes the exact
    // Bf16 carrier, never a materialized `Dense` fallback.
    let b = Builder::for_tests();
    let table = b
        .constant("table", TensorType::new(vec![3, 2], DType::BF16))
        .unwrap();
    let index = graph_fixtures::i32_constant(&b, "index", vec![2]).unwrap();
    let out = b.gather(table, 0, index).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (
            table.id(),
            bf16_host(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (index.id(), HostTensor::i32(vec![2], vec![2, 0])),
    ]);
    let out =
        eval_every_combination("gather bf16 table, i32 index (spec-376)", &g, &inputs).unwrap();
    assert!(
        matches!(out, Value::Exact(ExactValue::Bf16(_))),
        "expected the Spec-376 exact Bf16 carrier, got {out:?}"
    );
    assert_eq!(
        value_bits(&out),
        vec![
            5.0f32.to_bits(),
            6.0f32.to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits()
        ]
    );

    // Probe: an F16 table (never in the Spec-376 admission table, which only matches
    // BF16) gathered by I32 ids falls to the dense `Value::Host` arm of `walk::evaluate_gather`
    // (`ops::gather::gather` over its f32 lane), which must decode through `to_f32` too.
    let b = Builder::for_tests();
    let table = b
        .constant("table", TensorType::new(vec![3, 2], DType::F16))
        .unwrap();
    let index = graph_fixtures::i32_constant(&b, "index", vec![2]).unwrap();
    let out = b.gather(table, 0, index).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (
            table.id(),
            f16_host(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (index.id(), HostTensor::i32(vec![2], vec![2, 0])),
    ]);
    let out = eval_every_combination("gather f16 table, i32 index", &g, &inputs).unwrap();
    let Value::Host(t) = &out else {
        panic!("expected dense, got {out:?}")
    };
    assert!(t.numel() > 0, "published an empty payload");
    assert_eq!(t.to_f32().unwrap().as_ref(), &[5.0, 6.0, 1.0, 2.0]);

    // Probe: a BF16 `src` for axis-0 `Scatter` (no `Scatter` row exists in the Spec-376
    // table at all) must decode through `ops::movement::scatter_axis0`'s `f32_view` read.
    let b = Builder::for_tests();
    let src = b
        .constant("src", TensorType::new(vec![2, 2], DType::BF16))
        .unwrap();
    let index = graph_fixtures::i32_constant(&b, "index", vec![2]).unwrap();
    let out = b.scatter(src, index).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (src.id(), bf16_host(vec![2, 2], &[1.0, 2.0, 3.0, 4.0])),
        (index.id(), HostTensor::i32(vec![2], vec![1, 0])),
    ]);
    let out = eval_every_combination("scatter bf16 src", &g, &inputs).unwrap();
    let Value::Host(t) = &out else {
        panic!("expected dense, got {out:?}")
    };
    assert!(t.numel() > 0, "published an empty payload");
    assert_eq!(t.to_f32().unwrap().as_ref(), &[3.0, 4.0, 1.0, 2.0]);

    // Probe: `IndexedMatMul` has no Spec-376 row either, so an F32 activation times a
    // BF16 weight must decode the weight through `to_f32` in `ops::contraction::indexed_matmul`.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 2])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![2, 2, 2], DType::BF16))
        .unwrap();
    let idx = graph_fixtures::i32_constant(&b, "idx", vec![1]).unwrap();
    let out = b.indexed_matmul(x, w, idx).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([
        (x.id(), HostTensor::f32(vec![1, 2], vec![1.0, 2.0])),
        (
            w.id(),
            bf16_host(vec![2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        ),
        (idx.id(), HostTensor::i32(vec![1], vec![1])),
    ]);
    let out = eval_every_combination("indexed_matmul f32 x bf16 w", &g, &inputs).unwrap();
    assert_eq!(out.as_host().unwrap().as_f32().unwrap(), &[2.0, 1.0]);

    // Probe: `PackI8`'s input declared I32 (rather than its usual F32 quantization
    // result): `ops::index::pack_i8` must read the I32 words.
    let b = Builder::for_tests();
    let x = graph_fixtures::i32_constant(&b, "x", vec![4]).unwrap();
    let packed = b.pack_i8(x).unwrap();
    let g = b.finish(packed).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::i32(vec![4], vec![1, 2, 3, 4]))]);
    let out = eval_every_combination("pack_i8 exact i32 (words only)", &g, &inputs).unwrap();
    let t = out.as_host().unwrap();
    assert_eq!(
        t.as_i32(),
        Some(&[0x0403_0201_i32][..]),
        "PackI8 must publish the 4 codes packed into one word, read from the I32 words"
    );

    // Probe: an F32 tensor bound to an F16-declared input is a bind-time
    // `EvalError::Input` (`walk::preflight_bindings`): the walk never converts at bind, so `Reshape`
    // can never relabel a payload of the wrong dtype.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4], DType::F16))
        .unwrap();
    let out = b.reshape(x, vec![4]).unwrap();
    let g = b.finish(out).unwrap();
    let wrong_dtype = HostTensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = dense_inputs([(x.id(), wrong_dtype)]);
    let error = eval(&g, &inputs, EvalOptions::default()).unwrap_err();
    assert!(
        matches!(error, EvalError::Input { .. }),
        "expected EvalError::Input, got {error:?}"
    );
    assert!(error.to_string().contains("different dtype"), "{error}");

    // The same op on a real F16 tensor: `Reshape` must publish the values, never an empty payload.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4], DType::F16))
        .unwrap();
    let out = b.reshape(x, vec![2, 2]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), f16_host(vec![4], &[1.0, 2.0, 3.0, 4.0]))]);
    let out = eval_every_combination("reshape f16 (legitimately narrowed)", &g, &inputs).unwrap();
    let Value::Host(t) = &out else {
        panic!("expected dense, got {out:?}")
    };
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.to_f32().unwrap().as_ref(), &[1.0, 2.0, 3.0, 4.0]);
}

/// Round 2 : an F32-declared input bound as a BF16 tensor has the right
/// element count for the declared shape, so a preflight check comparing only `numel()` against the
/// aval would wrongly accept it and an f32-lane reader would then see no f32 payload.
/// `preflight_bindings` requires a `Value::Host` input to be exactly the declared dtype and shape.
///
/// Red mutation: drop the dtype comparison from the `Value::Host` arm of `preflight_bindings` (keep
/// the element-count check); this binding then passes preflight and `eval` returns `Ok` or fails
/// later in an operand read instead of `EvalError::Input`.
#[test]
fn f32_declared_input_bound_as_a_bf16_tensor_is_refused_at_bind_time() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let out = b.reshape(x, vec![4]).unwrap();
    let g = b.finish(out).unwrap();
    let inputs = dense_inputs([(x.id(), HostTensor::bf16(vec![4], vec![0u16; 4]))]);
    let error = eval(&g, &inputs, EvalOptions::default()).unwrap_err();
    assert!(
        matches!(error, EvalError::Input { .. }),
        "expected EvalError::Input, got {error:?}"
    );
    assert!(error.to_string().contains("different dtype"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// SC-001 (card 555), POOT-725: one index rule, every data dtype. `Gather` with an I32 index of `-1` and `len`
// over F32, BF16, I32 and E4M3FN data: each is `EvalError::Index` with the matching `IndexFaultKind`,
// `position` and `len`. An F32 index never reaches the oracle: shape inference refuses it by name.
//
// Mutation (recorded): in `ops::index_rule::index_at`, drop the `v < 0` check of the I32 arm
// (`v as usize` wraps and the `>= len` check still fires, reporting OutOfRange): the `-1` rows go red on
// the kind (`OutOfRange`, expected `Negative`).
#[test]
fn gather_index_faults_cover_every_data_dtype() {
    use crate::ops::index_rule::IndexFaultKind;

    #[derive(Clone, Copy)]
    enum Data {
        F32,
        Bf16,
        I32,
        E4m3Fn,
    }

    fn run(data: Data, bad: i32) -> Result<Value, EvalError> {
        let b = Builder::for_tests();
        let table_ty = match data {
            Data::F32 => TensorType::f32(vec![3]),
            Data::Bf16 => TensorType::new(vec![3], DType::BF16),
            Data::I32 => TensorType::new(vec![3], DType::I32),
            Data::E4m3Fn => TensorType::new(vec![3], DType::E4M3FN),
        };
        let table = b.constant("table", table_ty).unwrap();
        let index = graph_fixtures::i32_constant(&b, "index", vec![]).unwrap();
        let out = b.gather(table, 0, index).unwrap();
        let g = b.finish(out).unwrap();

        let table_value = match data {
            Data::F32 => Value::from(HostTensor::f32(vec![3], vec![10.0, 20.0, 30.0])),
            Data::Bf16 => Value::from(bf16_host(vec![3], &[10.0, 20.0, 30.0])),
            Data::I32 => Value::from(HostTensor::i32(vec![3], vec![10, 20, 30])),
            Data::E4m3Fn => {
                Value::Host(crate::fp8::encode_e4m3fn_tensor(vec![3], &[1.0, 2.0, 3.0]).unwrap())
            }
        };
        let index_value = Value::from(HostTensor::i32(vec![], vec![bad]));
        let inputs = HashMap::from([(table.id(), table_value), (index.id(), index_value)]);
        eval(&g, &inputs, EvalOptions::default()).map(|r| r.output)
    }

    let cases: [(i32, IndexFaultKind); 2] = [
        (-1, IndexFaultKind::Negative),
        (3, IndexFaultKind::OutOfRange),
    ];

    for &(data, label) in &[
        (Data::F32, "F32"),
        (Data::Bf16, "BF16"),
        (Data::I32, "I32"),
        (Data::E4m3Fn, "E4M3FN"),
    ] {
        for &(bad, kind) in &cases {
            let result = run(data, bad);
            let Err(EvalError::Index(fault)) = &result else {
                panic!("{label} data, I32 index {bad}: expected EvalError::Index, got {result:?}")
            };
            assert_eq!(fault.position, 0, "{label} data, I32 index {bad}");
            assert_eq!(fault.len, 3, "{label} data, I32 index {bad}");
            assert_eq!(fault.kind, kind, "{label} data, I32 index {bad}");
        }
        // A valid index still reads the right row (the fault rows above are not vacuous).
        run(data, 1).expect("a valid I32 index must still gather");
    }

    // An F32 index is a typed refusal of shape inference, whatever the data dtype.
    for data in [DType::F32, DType::BF16, DType::I32, DType::E4M3FN] {
        let err = poot_graph_ir::OpKind::Gather { axis: 0 }
            .infer(&[TensorType::new(vec![3], data), TensorType::f32(vec![])])
            .expect_err("an F32 gather index must be refused");
        assert!(
            matches!(
                err,
                poot_graph_ir::ShapeError::DtypeOp {
                    op: "Gather",
                    dtype: DType::F32
                }
            ),
            "{data:?}: {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// SC-003 (card 555): one generic function per movement op, over `T: Copy`. `Transpose`, `Slice`,
// `Concat` and `Broadcast` publish the literal index permutation of an index-tagged input,
// identically whether the carrier is dense f32, the authoritative I32 word lane, a BF16/F16
// operand's native `u16` words, or E4M3FN's raw bytes (including an odd, non-multiple-of-4 E4M3 row
// length, so a byte-packing assumption cannot sneak back in).
//
// Mutation (recorded): in `ops::movement::generic::transpose`, index the *input* row-major flat
// position with the *output* strides (`in_st[i]` instead of `in_st[perm[i]]`) only for `u8`/`u16`
// sources (simulating "those instantiations grew their own, wrong stride code"), by adding a
// `std::any::TypeId::of::<T>()` branch (`T: 'static` added locally for the probe). Confirmed red:
// the E4M3FN and the BF16/F16 transpose rows both published a shuffled table instead of the literal
// permutation (F32/I32 unaffected, proving the carriers are not sharing the mutated code by
// coincidence); reverted, confirmed green.
#[test]
fn movement_ops_match_the_literal_index_permutation_on_every_t_copy_lane() {
    // index-tagged source: element i (row-major) carries the literal value `i`, so a correct
    // transpose/slice/concat/broadcast output is just the permuted/sliced/concatenated/broadcast
    // index itself, trivially checkable by hand for small shapes.
    fn tagged_f32(shape: Vec<usize>) -> HostTensor {
        let n: usize = shape.iter().product();
        HostTensor::f32(shape, (0..n).map(|i| i as f32).collect())
    }
    fn tagged_i32(shape: Vec<usize>) -> HostTensor {
        let n: usize = shape.iter().product();
        HostTensor::i32(shape, (0..n).map(|i| i as i32).collect())
    }
    fn tagged_e4m3(shape: Vec<usize>) -> HostTensor {
        let n: usize = shape.iter().product();
        // Every byte 0..=0x7d decodes to a distinct finite value and round-trips through
        // `encode_e4m3fn` (card 555's own `encode_specials_boundaries_and_all_halfway_ties`), so a
        // literal byte tag survives the carrier's own codec unchanged.
        crate::fp8::e4m3fn_tensor(shape, (0..n).map(|i| (i % 0x7e) as u8).collect()).unwrap()
    }
    // A BF16 tensor whose `u16` words are the literal index itself - raw tag bits, not real
    // bf16-encoded floats (the generic engine moves `u16` words verbatim; it has no float semantics
    // to probe here).
    fn tagged_bf16(shape: Vec<usize>) -> HostTensor {
        let n: usize = shape.iter().product();
        HostTensor::bf16(shape, (0..n).map(|i| i as u16).collect())
    }
    fn tagged_f16(shape: Vec<usize>) -> HostTensor {
        let n: usize = shape.iter().product();
        HostTensor::f16(shape, (0..n).map(|i| i as u16).collect())
    }
    /// The BF16/F16 result's native words, as `i64` tags; also asserts the result kept the
    /// operand's own half-width dtype rather than widening to f32.
    fn narrowed_words(t: &HostTensor, bf16: bool) -> Vec<i64> {
        assert_eq!(
            t.dtype(),
            if bf16 { DType::BF16 } else { DType::F16 },
            "must keep its half-width dtype, not widen"
        );
        t.as_half()
            .expect("must publish native half words")
            .iter()
            .map(|&word| i64::from(word))
            .collect()
    }

    // Transpose, a [2,3] -> [3,2] permutation (an odd row length: 2*3=6 is not a multiple of 4,
    // deliberately, for the E4M3FN row-packing probe).
    let (shape, perm) = (vec![2, 3], vec![1, 0]);
    let expect_transpose = |src: &[i64]| -> Vec<i64> {
        let mut out = vec![0i64; src.len()];
        for r in 0..2 {
            for c in 0..3 {
                out[c * 2 + r] = src[r * 3 + c];
            }
        }
        out
    };
    let tagged: Vec<i64> = (0..6).collect();
    let expected = expect_transpose(&tagged);

    let f32_t = ops::movement::transpose(&tagged_f32(shape.clone()), &perm).unwrap();
    assert_eq!(
        f32_t
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected,
        "f32 transpose"
    );
    let i32_t = ops::movement::transpose(&tagged_i32(shape.clone()), &perm).unwrap();
    assert_eq!(
        i32_t
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected,
        "i32 transpose"
    );
    let bf16_t = ops::movement::transpose(&tagged_bf16(shape.clone()), &perm).unwrap();
    assert_eq!(
        narrowed_words(&bf16_t, true),
        expected,
        "bf16 (u16) transpose"
    );
    let f16_t = ops::movement::transpose(&tagged_f16(shape.clone()), &perm).unwrap();
    assert_eq!(
        narrowed_words(&f16_t, false),
        expected,
        "f16 (u16) transpose"
    );
    let e4m3_table = tagged_e4m3(shape.clone());
    let e4m3_t = ops::movement::generic::transpose(e4m3_table.view().bytes(), &shape, &perm);
    assert_eq!(
        e4m3_t.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        expected,
        "e4m3fn (u8) transpose"
    );

    // Slice axis 1, [1..3) of a [2,3] source.
    let expected_slice: Vec<i64> = vec![1, 2, 4, 5];
    let f32_s = ops::movement::slice(&tagged_f32(shape.clone()), 1, 1, 3).unwrap();
    assert_eq!(
        f32_s
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected_slice,
        "f32 slice"
    );
    let i32_s = ops::movement::slice(&tagged_i32(shape.clone()), 1, 1, 3).unwrap();
    assert_eq!(
        i32_s
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected_slice,
        "i32 slice"
    );
    let bf16_s = ops::movement::slice(&tagged_bf16(shape.clone()), 1, 1, 3).unwrap();
    assert_eq!(
        narrowed_words(&bf16_s, true),
        expected_slice,
        "bf16 (u16) slice"
    );
    let e4m3_s = ops::movement::generic::slice(e4m3_table.view().bytes(), &shape, 1, 1, 3);
    assert_eq!(
        e4m3_s.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        expected_slice,
        "e4m3fn (u8) slice"
    );

    // Concat two [1,3] parts along axis 0 -> [2,3]: the literal index itself (0..6), since the two
    // parts are tagged 0..3 and 3..6 (mirrors `expected_transpose`'s pre-transpose source).
    let f32_c = ops::movement::concat(
        &[
            &tagged_f32(vec![1, 3]),
            &HostTensor::f32(vec![1, 3], vec![3.0, 4.0, 5.0]),
        ],
        0,
    )
    .unwrap();
    assert_eq!(
        f32_c
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        tagged,
        "f32 concat"
    );
    let i32_c = ops::movement::concat(
        &[
            &tagged_i32(vec![1, 3]),
            &HostTensor::i32(vec![1, 3], vec![3, 4, 5]),
        ],
        0,
    )
    .unwrap();
    assert_eq!(
        i32_c
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        tagged,
        "i32 concat"
    );
    let bf16_part1 = HostTensor::bf16(vec![1, 3], vec![3, 4, 5]);
    let bf16_c = ops::movement::concat(&[&tagged_bf16(vec![1, 3]), &bf16_part1], 0).unwrap();
    assert_eq!(narrowed_words(&bf16_c, true), tagged, "bf16 (u16) concat");
    let part0 = tagged_e4m3(vec![1, 3]);
    let part1 = crate::fp8::e4m3fn_tensor(vec![1, 3], vec![3, 4, 5]).unwrap();
    let e4m3_c = ops::movement::generic::concat(
        &[
            (part0.view().bytes(), [1, 3].as_slice()),
            (part1.view().bytes(), [1, 3].as_slice()),
        ],
        0,
    );
    assert_eq!(
        e4m3_c.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        tagged,
        "e4m3fn (u8) concat"
    );

    // Broadcast [1,3] -> [2,3]: each row repeats the literal index.
    let expected_broadcast: Vec<i64> = vec![0, 1, 2, 0, 1, 2];
    let e4m3_b = ops::movement::generic::broadcast(part1.view().bytes(), &[1, 3], &[2, 3]);
    assert_eq!(
        e4m3_b.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        vec![3, 4, 5, 3, 4, 5],
        "e4m3fn (u8) broadcast keeps its own tag"
    );
    let f32_b = ops::movement::broadcast_to(&tagged_f32(vec![1, 3]), &[2, 3]).unwrap();
    assert_eq!(
        f32_b
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected_broadcast,
        "f32 broadcast"
    );
    let i32_b = ops::movement::broadcast_to(&tagged_i32(vec![1, 3]), &[2, 3]).unwrap();
    assert_eq!(
        i32_b
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        expected_broadcast,
        "i32 broadcast"
    );
    let bf16_b = ops::movement::broadcast_to(&tagged_bf16(vec![1, 3]), &[2, 3]).unwrap();
    assert_eq!(
        narrowed_words(&bf16_b, true),
        expected_broadcast,
        "bf16 (u16) broadcast"
    );
}

// ---------------------------------------------------------------------------------------------
// SC-004 (card 555): every other index reader goes through the one index rule.
//
// Mutation (recorded): in `ops::movement::generic::scatter_update`, change the `-1` padding guard
// from `matches!(value, IndexValue::I32(-1))` to `match value { IndexValue::I32(v) => v < 0, .. }`
// (treat every negative as padding). Confirmed red: the `-2` row below, which must fault, instead
// silently skipped the row (no error, wrong-but-plausible output); reverted, confirmed green.
#[test]
fn scatter_update_dynamic_update_slice_indexed_matmul_arg_top_k_route_through_index_at() {
    use crate::ops::index_rule::IndexFaultKind;

    // ScatterUpdate: slot `-1` is padding (the row writes nothing); `-2`, `pool` (== 3) fault, and a slot
    // named twice faults as a write race.
    {
        let b = Builder::for_tests();
        let base = b.constant("base", TensorType::f32(vec![3, 1])).unwrap();
        let src = b.constant("src", TensorType::f32(vec![2, 1])).unwrap();
        let slots = graph_fixtures::i32_constant(&b, "slots", vec![2]).unwrap();
        let out = b.scatter_update(base, src, slots).unwrap();
        let g = b.finish(out).unwrap();
        let base_value = HostTensor::f32(vec![3, 1], vec![7.0, 8.0, 9.0]);
        let src_value = HostTensor::f32(vec![2, 1], vec![1.0, 2.0]);
        let run = |words: [i32; 2]| {
            let inputs = dense_inputs([
                (base.id(), base_value.clone()),
                (src.id(), src_value.clone()),
                (slots.id(), HostTensor::i32(vec![2], words.to_vec())),
            ]);
            eval(&g, &inputs, EvalOptions::default())
        };

        let out = run([1, 2]).unwrap().output.into_host().unwrap();
        assert_eq!(
            out.as_f32().unwrap(),
            &[7.0, 1.0, 2.0],
            "rows land at slots"
        );
        let out = run([-1, 0]).unwrap().output.into_host().unwrap();
        assert_eq!(
            out.as_f32().unwrap(),
            &[2.0, 8.0, 9.0],
            "-1 writes nothing; the other row lands"
        );

        for (words, position, kind) in [
            ([-2, 1], 0, IndexFaultKind::Negative),
            ([0, 3], 1, IndexFaultKind::OutOfRange),
            ([2, 2], 1, IndexFaultKind::Duplicate),
        ] {
            let err = run(words).unwrap_err();
            let EvalError::Index(fault) = &err else {
                panic!("ScatterUpdate slots {words:?}: expected EvalError::Index, got {err:?}")
            };
            assert_eq!(
                (fault.kind, fault.position),
                (kind, position),
                "ScatterUpdate slots {words:?}"
            );
        }
    }

    // DynamicUpdateSlice: a runtime start past the window and a negative start both fault; a valid
    // runtime start still writes.
    {
        let b = Builder::for_tests();
        let operand = b.constant("operand", TensorType::f32(vec![4])).unwrap();
        let update = b.constant("update", TensorType::f32(vec![2])).unwrap();
        let index = graph_fixtures::i32_constant(&b, "index", vec![]).unwrap();
        let out = b
            .dynamic_update_slice_dyn(operand, update, index, 0)
            .unwrap();
        let g = b.finish(out).unwrap();
        let operand_value = HostTensor::f32(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
        let update_value = HostTensor::f32(vec![2], vec![9.0, 8.0]);

        let inputs = dense_inputs([
            (operand.id(), operand_value.clone()),
            (update.id(), update_value.clone()),
            (index.id(), HostTensor::i32(vec![], vec![1])),
        ]);
        let out = eval(&g, &inputs, EvalOptions::default())
            .unwrap()
            .output
            .into_host()
            .unwrap();
        assert_eq!(out.as_f32().unwrap(), &[0.0, 9.0, 8.0, 0.0]);

        for (bad, kind) in [
            (3, IndexFaultKind::OutOfRange),
            (-1, IndexFaultKind::Negative),
        ] {
            let inputs = dense_inputs([
                (operand.id(), operand_value.clone()),
                (update.id(), update_value.clone()),
                (index.id(), HostTensor::i32(vec![], vec![bad])),
            ]);
            let err = eval(&g, &inputs, EvalOptions::default()).unwrap_err();
            let EvalError::Index(fault) = &err else {
                panic!("DynamicUpdateSlice start {bad}: expected EvalError::Index, got {err:?}")
            };
            assert_eq!(fault.kind, kind, "DynamicUpdateSlice start {bad}");
        }
    }

    // IndexedMatMul: an expert id of `E` (the expert count) faults.
    {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 2])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![2, 2, 2])).unwrap();
        let idx = graph_fixtures::i32_constant(&b, "idx", vec![1]).unwrap();
        let out = b.indexed_matmul(x, w, idx).unwrap();
        let g = b.finish(out).unwrap();
        let inputs = dense_inputs([
            (x.id(), HostTensor::f32(vec![1, 2], vec![1.0, 2.0])),
            (
                w.id(),
                HostTensor::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            ),
            (idx.id(), HostTensor::i32(vec![1], vec![2])),
        ]);
        let err = eval(&g, &inputs, EvalOptions::default()).unwrap_err();
        let EvalError::Index(fault) = &err else {
            panic!("IndexedMatMul expert id 2: expected EvalError::Index, got {err:?}")
        };
        assert_eq!(fault.kind, IndexFaultKind::OutOfRange);
        assert_eq!(fault.len, 2, "expert count");
    }

    // ArgTopK: a non-integral rank faults.
    {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
        let out = b.arg_top_k(x, 2).unwrap();
        let g = b.finish(out).unwrap();
        let inputs = dense_inputs([(x.id(), HostTensor::f32(vec![4], vec![2.0, 0.5, 3.0, 1.0]))]);
        let err = eval(&g, &inputs, EvalOptions::default()).unwrap_err();
        let EvalError::Index(fault) = &err else {
            panic!("ArgTopK rank 0.5: expected EvalError::Index, got {err:?}")
        };
        assert_eq!(fault.kind, IndexFaultKind::NotIntegral);
    }
}
