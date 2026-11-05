//! POOT-1430: operand precision is part of an op's semantics, and the oracle evaluates exactly it.
//!
//! The rows state the definition of `OperandPrecision` against values worked out by hand (the rounding
//! ties and the range edge of each fragment dtype), against an independent f64 product (the error each
//! precision leaves), and against each composite's own chain (a `DenseContraction`, a
//! `GroupedDenseContraction`, a `PackedContraction`, a `ContractionEpilogue`, `Attention`): a composite
//! reads its operands as its attribute says.

use poot_graph_ir::op::{ContractionHead, FusedOp, FusedOperand, FusedRegion, FusedStep, LocalId};
use poot_graph_ir::{AttentionAttrs, BinOp, FragmentDtype, OpKind, OperandPrecision, ops};
use poot_tensor::{DType, HostTensor};

use super::composites::{data, evaluate, graph_over, one_equation};

const BF16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::Bf16);
const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);
const SPLIT_BF16: OperandPrecision = OperandPrecision::Split(FragmentDtype::Bf16);

/// `[1, k] x [k, 1]` under `precision`: the dot product of `a` and `b`.
fn dot(a: &HostTensor, b: &HostTensor, precision: OperandPrecision) -> f32 {
    let (a, b) = (a.clone(), b.clone());
    let inputs = [a, b];
    let graph = graph_over(&inputs, |g, t| g.matmul_as(t[0], t[1], precision).unwrap());
    evaluate(&graph).unwrap().to_f32().unwrap()[0]
}

fn row(values: &[f32]) -> HostTensor {
    HostTensor::f32(vec![1, values.len()], values.to_vec())
}

fn column(values: &[f32]) -> HostTensor {
    HostTensor::f32(vec![values.len(), 1], values.to_vec())
}

fn ones(count: usize) -> HostTensor {
    column(&vec![1.0; count])
}

/// `Narrow` rounds every operand to nearest even before the product, in each dtype.
///
/// The activation holds, in bf16, the tie `1 + 2^-8` (between 1 and `1 + 2^-7`: goes to the even 1.0) and
/// `1 + 3 * 2^-8` (a tie that goes up to `1 + 2^-6`); in f16 the ties `1 + 2^-11` (down to 1.0) and `1 + 3 * 2^-11`
/// (up to `1 + 2^-9`). Against a column of ones the dot is the sum of the rounded values. Mutation: in
/// `round_to_fragment`, truncate instead of rounding to nearest (clear the dropped bits); the tie-up values change
/// and this row goes red.
#[test]
fn narrow_rounds_each_operand_to_nearest_even() {
    let bf16_ties = [1.0 + 2f32.powi(-8), 1.0 + 3.0 * 2f32.powi(-8)];
    let want = 1.0 + (1.0 + 2f32.powi(-6));
    assert_eq!(
        dot(&row(&bf16_ties), &ones(2), BF16).to_bits(),
        want.to_bits()
    );

    let f16_ties = [1.0 + 2f32.powi(-11), 1.0 + 3.0 * 2f32.powi(-11)];
    let want = 1.0 + (1.0 + 2f32.powi(-9));
    assert_eq!(
        dot(&row(&f16_ties), &ones(2), F16).to_bits(),
        want.to_bits()
    );

    // Exact keeps the f32 values: the same dot differs from both narrow ones.
    let exact = dot(&row(&bf16_ties), &ones(2), OperandPrecision::Exact);
    assert_eq!(exact, bf16_ties[0] + bf16_ties[1]);
}

/// The narrowing conversion does not saturate: an f32 beyond the f16 range becomes infinity and reaches the
/// result, where a non-finite output is a fault (ADR-0101 decision 4); bf16 keeps the f32 range and is finite.
/// Mutation: in `round_to_fragment` clamp an f16 result to 65504; the product stays finite and this row goes red.
#[test]
fn an_operand_beyond_the_f16_range_becomes_infinity_not_a_clamp() {
    let big = row(&[70000.0]);
    assert_eq!(dot(&big, &ones(1), F16), f32::INFINITY);
    assert_eq!(
        dot(&big, &ones(1), BF16),
        70144.0,
        "bf16 rounds 70000 to its grid, finite"
    );
}

/// `Split` enters an f32 operand as `hi + lo`, which recovers the f32 value to second order: against an f64
/// product its error is orders of magnitude below `Narrow`'s, and below the 6.4e-6 componentwise bound the
/// design records (about 0.42 u^2 of the dtype). Mutation: drop the `lo` term in `operand_terms`; the split dot
/// equals the narrow one and this row goes red.
#[test]
fn split_recovers_the_f32_product_far_better_than_narrow() {
    let k = 257;
    let a = row(data(&[1, k], 1, 1.0).as_f32().unwrap());
    let b = column(data(&[k, 1], 2, 1.0).as_f32().unwrap());
    let (av, bv) = (a.as_f32().unwrap(), b.as_f32().unwrap());
    let reference: f64 = av
        .iter()
        .zip(bv)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let magnitude: f64 = av
        .iter()
        .zip(bv)
        .map(|(&x, &y)| f64::from(x * y).abs())
        .sum();
    let error = |precision| (f64::from(dot(&a, &b, precision)) - reference).abs() / magnitude;
    let (narrow, split) = (error(BF16), error(SPLIT_BF16));
    assert!(split < 6.4e-6, "split error {split:e}");
    assert!(
        narrow > 1e-4,
        "narrow error {narrow:e} must be visible at this fixture"
    );
    assert!(
        narrow > 20.0 * split,
        "narrow {narrow:e} vs split {split:e}"
    );
}

/// An operand already stored in the fragment dtype is exact in it and enters once: two bf16 tensors read
/// under `Narrow(bf16)` and `Split(bf16)` give the bits of the exact product.
#[test]
fn an_operand_stored_in_the_fragment_dtype_enters_once() {
    let words = |values: &[f32]| -> Vec<u16> {
        values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
    };
    let (av, bv) = (
        data(&[1, 9], 5, 1.0).as_f32().unwrap().to_vec(),
        data(&[9, 1], 6, 1.0).as_f32().unwrap().to_vec(),
    );
    let (a, b) = (
        HostTensor::bf16(vec![1, 9], words(&av)),
        HostTensor::bf16(vec![9, 1], words(&bv)),
    );
    let exact = dot(&a, &b, OperandPrecision::Exact).to_bits();
    for precision in [BF16, SPLIT_BF16] {
        assert_eq!(dot(&a, &b, precision).to_bits(), exact, "{precision}");
    }
}

/// `IndexedMatMul` reads its activation and expert stack as declared: each row's output equals the narrow
/// `MatMul` of that row against its own expert.
#[test]
fn indexed_matmul_reads_its_operands_as_declared() {
    let (m, k, n, experts) = (3usize, 5usize, 2usize, 2usize);
    let x = data(&[m, k], 7, 1.0);
    let w = data(&[experts, k, n], 8, 1.0);
    let ids = HostTensor::i32(vec![m], vec![1, 0, 1]);
    let eval_indexed = |precision: OperandPrecision| {
        one_equation(
            OpKind::indexed_matmul().with_operand_precision(precision),
            &[x.clone(), w.clone(), ids.clone()],
        )
    };
    let xs = x.as_f32().unwrap();
    let ws = w.as_f32().unwrap();
    for precision in [OperandPrecision::Exact, BF16, F16, SPLIT_BF16] {
        let got = evaluate(&eval_indexed(precision)).unwrap();
        let got = got.as_f32().unwrap();
        for (row, &id) in ids.as_i32().unwrap().iter().enumerate() {
            let xr = row_of(&xs[row * k..(row + 1) * k]);
            for col in 0..n {
                let wc = column(
                    &(0..k)
                        .map(|kk| ws[(id as usize * k + kk) * n + col])
                        .collect::<Vec<_>>(),
                );
                assert_eq!(
                    got[row * n + col].to_bits(),
                    dot(&xr, &wc, precision).to_bits(),
                    "{precision} row {row} col {col}"
                );
            }
        }
    }
}

fn row_of(values: &[f32]) -> HostTensor {
    row(values)
}

/// A `DenseContraction` over a `[N, K]` weight is, under every precision, the `MatMul` of the activation with
/// the transposed weight at that precision (the chain the dense fold replaces): the composite carries the attribute
/// into its definition.
#[test]
fn a_dense_contraction_reads_its_operands_as_declared() {
    let (m, k, n) = (3usize, 8usize, 4usize);
    let (x, w) = (data(&[m, k], 11, 1.0), data(&[n, k], 12, 1.0));
    for precision in [OperandPrecision::Exact, BF16, F16, SPLIT_BF16] {
        let composite = evaluate(&one_equation(
            OpKind::dense_contraction(DType::F32).with_operand_precision(precision),
            &[x.clone(), w.clone()],
        ))
        .unwrap();
        let chain = evaluate(&graph_over(&[x.clone(), w.clone()], |b, t| {
            b.matmul_as(t[0], b.transpose(t[1], vec![1, 0]).unwrap(), precision)
                .unwrap()
        }))
        .unwrap();
        assert_eq!(bits(&composite), bits(&chain), "{precision}");
    }
}

/// The grouped sibling: each group's block of rows contracted at the declared precision.
#[test]
fn a_grouped_dense_contraction_reads_its_operands_as_declared() {
    let (groups, m, n, k) = (2usize, 3usize, 2usize, 6usize);
    let (x, w) = (
        data(&[groups, m, k], 13, 1.0),
        data(&[groups * n, k], 14, 1.0),
    );
    let bf16_words = |t: &HostTensor| {
        HostTensor::bf16(
            t.shape().to_vec(),
            t.as_f32()
                .unwrap()
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect(),
        )
    };
    let w = bf16_words(&w);
    for precision in [BF16, F16, SPLIT_BF16] {
        let composite = evaluate(&one_equation(
            OpKind::grouped_dense_contraction(DType::BF16, groups)
                .with_operand_precision(precision),
            &[x.clone(), w.clone()],
        ))
        .unwrap();
        let chain = evaluate(&graph_over(&[x.clone(), w.clone()], |b, t| {
            let blocks = b.reshape(t[1], vec![groups, n, k]).unwrap();
            b.matmul_as(t[0], b.transpose(blocks, vec![0, 2, 1]).unwrap(), precision)
                .unwrap()
        }))
        .unwrap();
        assert_eq!(bits(&composite), bits(&chain), "{precision}");
    }
}

/// A `ContractionEpilogue` reads its head's operands as the head's precision says, then runs its region.
/// Mutation: in `ContractionHead::op`, build the `MatMul` head with `Exact`; the narrow row goes red.
#[test]
fn a_contraction_epilogue_reads_its_head_operands_as_declared() {
    let (m, k, n) = (3usize, 7usize, 4usize);
    let (x, w, bias) = (
        data(&[m, k], 15, 1.0),
        data(&[k, n], 16, 1.0),
        data(&[n], 17, 1.0),
    );
    // local 0 is the head's result, local 1 the bias; the region adds them.
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![FusedStep {
            op: FusedOp::Binary(BinOp::Add),
            inputs: vec![
                FusedOperand::Local(LocalId::new(0)),
                FusedOperand::Local(LocalId::new(1)),
            ],
        }],
        output: LocalId::new(2),
        pack: Vec::new(),
    };
    for precision in [OperandPrecision::Exact, BF16, SPLIT_BF16] {
        let epilogue = evaluate(&one_equation(
            OpKind::ContractionEpilogue {
                head: ContractionHead::MatMul { precision },
                region: region.clone(),
            },
            &[x.clone(), w.clone(), bias.clone()],
        ))
        .unwrap();
        let chain = evaluate(&graph_over(
            &[x.clone(), w.clone(), bias.clone()],
            |b, t| {
                let head = b.matmul_as(t[0], t[1], precision).unwrap();
                b.binary(BinOp::Add, head, t[2]).unwrap()
            },
        ))
        .unwrap();
        assert_eq!(bits(&epilogue), bits(&chain), "{precision}");
    }
}

/// `Attention` rounds Q, K, V and the probabilities as declared: under `Narrow(f16)` the result moves off the
/// exact one (the rounding is applied, so the row is not inert) by no more than the f16 bound on this fixture,
/// and equals the chain `ops::attention_masked` builds with the same precision on both its products. Mutation:
/// in `attention_over_keys` pass `Exact` to the `P V` product; the probabilities are not rounded and this row
/// goes red.
#[test]
fn attention_rounds_its_operands_as_declared() {
    let (batch, hkv, n_rep, cap, d) = (1usize, 2usize, 2usize, 9usize, 8usize);
    let hq = hkv * n_rep;
    let scale = 0.35f32;
    let inputs = [
        data(&[batch, hq, 1, d], 21, 1.0),
        data(&[batch, hkv, cap, d], 22, 1.0),
        data(&[batch, hkv, cap, d], 23, 1.0),
        HostTensor::f32(vec![batch, 1, 1, cap], vec![0.0; cap]),
    ];
    let attention = |precision: OperandPrecision| {
        evaluate(&one_equation(
            OpKind::Attention(AttentionAttrs {
                precision,
                ..AttentionAttrs::new(n_rep, scale, None)
            }),
            &inputs,
        ))
        .unwrap()
    };
    let exact = attention(OperandPrecision::Exact);
    let narrow = attention(F16);
    let worst = |a: &HostTensor, b: &HostTensor| {
        poot_test_util::max_abs_error(a.as_f32().unwrap(), b.as_f32().unwrap())
    };
    let drift = worst(&exact, &narrow);
    assert!(
        drift > 1e-5,
        "the rounding must be visible on this fixture, got {drift:e}"
    );
    assert!(
        drift < 2e-3,
        "f16 attention stays within the f16 bound, got {drift:e}"
    );
    let chain = evaluate(&graph_over(&inputs, |b, t| {
        // The same chain with both products at the precision: the traced `ops::attention_masked`, whose
        // definition is `attention_over_keys`, is exact; spell its dataflow with `matmul_as`.
        let k = ops::repeat_kv(b, t[1], n_rep).unwrap();
        let v = ops::repeat_kv(b, t[2], n_rep).unwrap();
        let kt = b.transpose(k, vec![0, 1, 3, 2]).unwrap();
        let scores = b.matmul_as(t[0], kt, F16).unwrap();
        let scores = b
            .binary_scalar(BinOp::Mul, scores, poot_graph_ir::Scalar::F32(scale))
            .unwrap();
        let scores = b.binary(BinOp::Add, scores, t[3]).unwrap();
        let last = 3;
        let m = b
            .reduce(poot_graph_ir::RedOp::Max, scores, last, true)
            .unwrap();
        let e = b
            .unary(
                poot_graph_ir::UnOp::Exp,
                b.binary(BinOp::Sub, scores, m).unwrap(),
            )
            .unwrap();
        let denom = b.reduce(poot_graph_ir::RedOp::Sum, e, last, true).unwrap();
        let p = b.binary(BinOp::Div, e, denom).unwrap();
        b.matmul_as(p, v, F16).unwrap()
    }))
    .unwrap();
    assert_eq!(bits(&narrow), bits(&chain));
}

fn bits(t: &HostTensor) -> Vec<u32> {
    t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}
