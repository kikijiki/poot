//! Card 556: composite ops are defined by their decomposition (ADR-0101 tier 1), so the oracle has one
//! definition of each.
//!
//! - SC-002: a one-equation graph holding each composite evaluates to exactly the bits of the same
//!   `ops::` chain traced directly with the same parameters (an independent trace, not
//!   `decompose()`'s own graph, so a wrong parameter mapping in the decomposition turns a row red).
//! - SC-003: `rope`, `rope_batched`, `rope_sectioned` (equal positions) and `rope_sectioned_batched`
//!   agree bit for bit: one half-split pairing.
//! - SC-005: a prefill attention composite's decomposition materializes `[1, Hq, L, L]` scores, which
//!   the evaluation budget refuses as a typed error naming the composite equation.

use std::collections::HashMap;

use poot_graph_ir::AttentionAttrs;
use poot_graph_ir::op::{
    ContractionHead, FusedOp, FusedOperand, FusedRegion, FusedStep, LocalId, RowRegion, RowStep,
};
use poot_graph_ir::{
    BinOp, Builder, Graph, OpKind, Operand, RedOp, Scalar, TensorType, Traced, UnOp, ValueId, ops,
};
use poot_tensor::{DType, HostTensor};

use super::helpers::fill;
use crate::{EvalError, EvalOptions, Value, eval};

/// `n` deterministic values in `[-scale, scale)`.
fn data(shape: &[usize], seed: u64, scale: f32) -> HostTensor {
    let n = shape.iter().product::<usize>();
    HostTensor::f32(
        shape.to_vec(),
        fill(n, seed).iter().map(|v| v * scale).collect(),
    )
}

/// An additive mask `[.., rows, keys]`: `0` where key `j <= i + offset`, `-1e9` elsewhere.
fn causal_mask(shape: &[usize], offset: usize) -> HostTensor {
    let (rows, keys) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let n = shape.iter().product::<usize>();
    HostTensor::f32(
        shape.to_vec(),
        (0..n)
            .map(|flat| {
                let (i, j) = ((flat / keys) % rows, flat % keys);
                if j <= i + offset { 0.0 } else { -1.0e9 }
            })
            .collect(),
    )
}

/// A graph whose constants are `inputs` (in order) and whose output is what `body` builds over them,
/// with the inputs bound.
fn graph_over(
    inputs: &[HostTensor],
    body: impl FnOnce(&Builder, &[Traced]) -> Traced,
) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let traced: Vec<Traced> = inputs
        .iter()
        .enumerate()
        .map(|(position, tensor)| {
            b.constant(
                &format!("input.{position}"),
                TensorType::new(tensor.shape().to_vec(), tensor.dtype()),
            )
            .unwrap()
        })
        .collect();
    let out = body(&b, &traced);
    let bound = traced
        .iter()
        .zip(inputs)
        .map(|(traced, tensor)| (traced.id(), Value::from(tensor.clone())))
        .collect();
    (b.finish(out).unwrap(), bound)
}

/// A one-equation graph holding `op` over `inputs`. A composite is compiler-only (`Builder` has no
/// method for one), so it is staged through an append plan.
fn one_equation(op: OpKind, inputs: &[HostTensor]) -> (Graph, HashMap<ValueId, Value>) {
    graph_over(inputs, |b, traced| {
        let mut plan = b.append_plan(0).unwrap();
        let operands = traced.iter().map(|t| Operand::Value(t.id())).collect();
        let out = plan.equation(op, operands).expect("the composite types");
        plan.declare_result(out).expect("declare the result");
        let mut prepared = b.preflight_append(plan).expect("preflight the append");
        b.commit_append(&mut prepared).expect("commit the append")
    })
}

fn evaluate((g, inputs): &(Graph, HashMap<ValueId, Value>)) -> Result<HostTensor, EvalError> {
    let output = eval(g, inputs, EvalOptions::default())?.output;
    Ok(output.into_host().expect("a dense output"))
}

/// Every element equal bit for bit (I32 words or F32 bits), and no non-finite F32 element.
fn assert_bits_equal(got: &HostTensor, want: &HostTensor, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
    if let (Some(got), Some(want)) = (got.as_i32(), want.as_i32()) {
        assert_eq!(got, want, "{what}: I32 words");
        return;
    }
    let (got, want) = (got.as_f32().unwrap(), want.as_f32().unwrap());
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{what}: element {index} is not finite: {g} vs {w}"
        );
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {index}: {g} vs {w}"
        );
    }
}

/// SC-002's row: the composite, evaluated, equals the independently traced chain bit for bit.
fn assert_composite_matches_chain(
    what: &str,
    op: OpKind,
    inputs: &[HostTensor],
    chain: impl FnOnce(&Builder, &[Traced]) -> Traced,
) {
    let composite = evaluate(&one_equation(op, inputs))
        .unwrap_or_else(|error| panic!("{what}: the composite evaluates: {error}"));
    let traced = evaluate(&graph_over(inputs, chain))
        .unwrap_or_else(|error| panic!("{what}: the traced chain evaluates: {error}"));
    assert_bits_equal(&composite, &traced, what);
}

/// SC-002 (decode): `Attention { n_rep, scale }` is `ops::attention_masked` with the same
/// GQA ratio and scale, over a batched cache and a per-row mask.
#[test]
fn flash_decode_composite_is_the_traced_masked_attention() {
    let (batch, hkv, n_rep, cap, d) = (2usize, 2usize, 2usize, 6usize, 8usize);
    let hq = hkv * n_rep;
    let scale = 0.35f32;
    let inputs = [
        data(&[batch, hq, 1, d], 11, 1.0),
        data(&[batch, hkv, cap, d], 12, 1.0),
        data(&[batch, hkv, cap, d], 13, 1.0),
        causal_mask(&[batch, 1, 1, cap], 3),
    ];
    assert_composite_matches_chain(
        "flash decode",
        OpKind::Attention(AttentionAttrs::new(n_rep, scale, None)),
        &inputs,
        |b, t| ops::attention_masked(b, t[0], t[1], t[2], n_rep, scale, t[3]).unwrap(),
    );
}

/// SC-002 (prefill, no softcap): `Attention { softcap: None }` is `ops::attention_prefill`,
/// here with a per-head (ALiBi-shaped) mask.
#[test]
fn flash_prefill_composite_is_the_traced_prefill_attention() {
    let (hkv, n_rep, l, d) = (2usize, 2usize, 5usize, 8usize);
    let hq = hkv * n_rep;
    let scale = 0.35f32;
    let mut mask = causal_mask(&[1, hq, l, l], 0);
    let bias = data(&[1, hq, l, l], 14, 0.5);
    let summed: Vec<f32> = mask
        .as_f32()
        .unwrap()
        .iter()
        .zip(bias.as_f32().unwrap())
        .map(|(m, b)| m + b)
        .collect();
    mask = HostTensor::f32(vec![1, hq, l, l], summed);
    let inputs = [
        data(&[1, hq, l, d], 15, 1.0),
        data(&[1, hkv, l, d], 16, 1.0),
        data(&[1, hkv, l, d], 17, 1.0),
        mask,
    ];
    assert_composite_matches_chain(
        "flash prefill",
        OpKind::Attention(AttentionAttrs::new(n_rep, scale, None)),
        &inputs,
        |b, t| ops::attention_prefill(b, t[0], t[1], t[2], n_rep, scale, t[3]).unwrap(),
    );
}

/// A per-head sink: `Attention { sink: PerHead }` is `ops::attention_masked_sinks` (bit for bit), and both
/// equal the definition written out in `f64` — the sink is one more logit in each row's softmax that
/// weighs into the maximum and the normalizer and has no value row. Sinks above and below the scores both
/// occur, so a sink dropped from the maximum or from the normalizer changes the result.
#[test]
fn sink_composite_is_the_traced_sink_attention_and_the_softmax_with_one_more_logit() {
    let (hkv, n_rep, l, d) = (2usize, 2usize, 5usize, 8usize);
    let hq = hkv * n_rep;
    let scale = 0.35f32;
    let sinks = HostTensor::f32(vec![hq], vec![-3.0, 0.5, 2.0, 6.0]);
    let inputs = [
        data(&[1, hq, l, d], 31, 1.0),
        data(&[1, hkv, l, d], 32, 1.0),
        data(&[1, hkv, l, d], 33, 1.0),
        causal_mask(&[1, 1, l, l], 0),
        sinks.clone(),
    ];
    assert_composite_matches_chain(
        "sink prefill",
        OpKind::Attention(AttentionAttrs {
            sink: poot_graph_ir::SinkOperand::PerHead,
            ..AttentionAttrs::new(n_rep, scale, None)
        }),
        &inputs,
        |b, t| ops::attention_masked_sinks(b, t[0], t[1], t[2], n_rep, scale, t[3], t[4]).unwrap(),
    );

    let composite = evaluate(&one_equation(
        OpKind::Attention(AttentionAttrs {
            sink: poot_graph_ir::SinkOperand::PerHead,
            ..AttentionAttrs::new(n_rep, scale, None)
        }),
        &inputs,
    ))
    .unwrap();
    let [q, k, v, mask, sink] = [0, 1, 2, 3, 4].map(|i| inputs[i].as_f32().unwrap());
    let got = composite.as_f32().unwrap();
    for h in 0..hq {
        for i in 0..l {
            let scores: Vec<f64> = (0..l)
                .map(|t| {
                    let dot: f64 = (0..d)
                        .map(|c| {
                            f64::from(q[(h * l + i) * d + c])
                                * f64::from(k[((h / n_rep) * l + t) * d + c])
                        })
                        .sum();
                    dot * f64::from(scale) + f64::from(mask[i * l + t])
                })
                .collect();
            let top = scores.iter().copied().fold(f64::from(sink[h]), f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
            let denominator = weights.iter().sum::<f64>() + (f64::from(sink[h]) - top).exp();
            for c in 0..d {
                let want: f64 = (0..l)
                    .map(|t| weights[t] * f64::from(v[((h / n_rep) * l + t) * d + c]))
                    .sum::<f64>()
                    / denominator;
                let at = (h * l + i) * d + c;
                assert!(
                    (f64::from(got[at]) - want).abs() < 1e-5,
                    "head {h} row {i} channel {c}: {} vs {want}",
                    got[at]
                );
            }
        }
    }
}

/// A causal mask stated by its positions: `Attention { mask: Causal { window } }` is the attention under
/// `ops::causal_mask_from_pos` (bit for bit), and both equal the visibility written out in `f64`: key `t` is
/// seen by the query at position `p` when `t <= p` and `p - t < window`. The positions are not `0..L`, so a
/// kernel that assumed the row index as the position reads other keys.
#[test]
fn causal_composite_is_the_traced_attention_under_the_mask_of_its_positions() {
    let (hkv, n_rep, l, cap, d) = (2usize, 2usize, 3usize, 7usize, 8usize);
    let hq = hkv * n_rep;
    let scale = 0.35f32;
    let positions = [1i32, 4, 6];
    for window in [None, Some(3)] {
        let inputs = [
            data(&[1, hq, l, d], 41, 1.0),
            data(&[1, hkv, cap, d], 42, 1.0),
            data(&[1, hkv, cap, d], 43, 1.0),
            HostTensor::i32(vec![1, l], positions.to_vec()),
        ];
        let op = OpKind::Attention(AttentionAttrs {
            mask: poot_graph_ir::MaskOperand::Causal { window },
            ..AttentionAttrs::new(n_rep, scale, None)
        });
        assert_composite_matches_chain("causal attention", op.clone(), &inputs, |b, t| {
            let mask = ops::causal_mask_from_pos(b, t[3], cap, window).unwrap();
            ops::attention_masked(b, t[0], t[1], t[2], n_rep, scale, mask).unwrap()
        });

        let composite = evaluate(&one_equation(op, &inputs)).unwrap();
        let [q, k, v] = [0, 1, 2].map(|i| inputs[i].as_f32().unwrap());
        let got = composite.as_f32().unwrap();
        for h in 0..hq {
            for (i, &position) in positions.iter().enumerate() {
                let visible: Vec<usize> = (0..cap)
                    .filter(|&t| {
                        t as i32 <= position
                            && window.is_none_or(|w| position - (t as i32) < w as i32)
                    })
                    .collect();
                let scores: Vec<f64> = visible
                    .iter()
                    .map(|&t| {
                        let dot: f64 = (0..d)
                            .map(|c| {
                                f64::from(q[(h * l + i) * d + c])
                                    * f64::from(k[((h / n_rep) * cap + t) * d + c])
                            })
                            .sum();
                        dot * f64::from(scale)
                    })
                    .collect();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for c in 0..d {
                    let want: f64 = visible
                        .iter()
                        .zip(&weights)
                        .map(|(&t, w)| w * f64::from(v[((h / n_rep) * cap + t) * d + c]))
                        .sum::<f64>()
                        / denominator;
                    let at = (h * l + i) * d + c;
                    assert!(
                        (f64::from(got[at]) - want).abs() < 1e-5,
                        "window {window:?} head {h} row {i} channel {c}: {} vs {want}",
                        got[at]
                    );
                }
            }
        }
    }
}

/// SC-002 (prefill, softcap): `Attention { softcap: Some(c) }` is
/// `ops::attention_prefill_softcap` with the same cap. The scores reach past the cap, so a dropped or
/// different softcap changes the bits.
#[test]
fn softcapped_flash_prefill_composite_is_the_traced_softcapped_prefill() {
    let (hkv, n_rep, l, d) = (2usize, 2usize, 5usize, 8usize);
    let hq = hkv * n_rep;
    let (scale, cap) = (0.35f32, 2.0f32);
    let inputs = [
        data(&[1, hq, l, d], 18, 4.0),
        data(&[1, hkv, l, d], 19, 4.0),
        data(&[1, hkv, l, d], 20, 1.0),
        causal_mask(&[1, 1, l, l], 0),
    ];
    assert_composite_matches_chain(
        "softcapped flash prefill",
        OpKind::Attention(AttentionAttrs::new(n_rep, scale, Some(cap))),
        &inputs,
        |b, t| {
            ops::attention_prefill_softcap(b, t[0], t[1], t[2], n_rep, scale, t[3], Some(cap))
                .unwrap()
        },
    );
}

/// SC-002 (Rope, full and partial): `Rope { rot }` is `ops::rope_partial` over `x`'s last axis with the
/// same rotary width; the partial row passes `[rot, D)` through.
#[test]
fn rope_composite_is_the_traced_half_split_rotation() {
    let (heads, rows, d) = (4usize, 3usize, 8usize);
    for rot in [d, d / 2] {
        let inputs = [
            data(&[1, heads, rows, d], 21, 1.0),
            data(&[rows, rot], 22, 1.0),
            data(&[rows, rot], 23, 1.0),
        ];
        assert_composite_matches_chain(
            &format!("rope rot={rot}"),
            OpKind::Rope { rot },
            &inputs,
            |b, t| ops::rope_partial(b, t[0], t[1], t[2], d, rot, 3).unwrap(),
        );
    }
}

/// SC-002 (`ContractionEpilogue`): the head contraction, then the region's steps over its result: a bias
/// add, a residual add and a literal scale, `ops::linear` then the chain the tracer states.
#[test]
fn contraction_epilogue_composite_is_the_traced_matmul_then_chain() {
    let local = |index: usize| FusedOperand::Local(LocalId::new(index));
    let step = |op, inputs| FusedStep { op, inputs };
    let region = FusedRegion {
        n_inputs: 3,
        steps: vec![
            step(FusedOp::Binary(BinOp::Add), vec![local(0), local(1)]),
            step(FusedOp::Binary(BinOp::Add), vec![local(3), local(2)]),
            step(
                FusedOp::Binary(BinOp::Mul),
                vec![local(4), FusedOperand::Lit(Scalar::F32(0.5))],
            ),
        ],
        output: LocalId::new(5),
        pack: Vec::new(),
    };
    let inputs = [
        data(&[2, 3, 5], 24, 1.0),
        data(&[5, 4], 25, 1.0),
        data(&[4], 26, 1.0),
        data(&[2, 3, 4], 27, 1.0),
    ];
    let op = OpKind::ContractionEpilogue {
        head: ContractionHead::MatMul,
        region,
    };
    assert_composite_matches_chain("contraction_epilogue", op, &inputs, |b, t| {
        let biased = ops::linear(b, t[0], t[1], Some(t[2])).unwrap();
        let summed = b.binary(BinOp::Add, biased, t[3]).unwrap();
        b.binary_scalar(BinOp::Mul, summed, Scalar::F32(0.5))
            .unwrap()
    });
}

/// SC-002 (`Fused`, F32): a SwiGLU region is `ops::silu(x) * y`, step for step (operand order and
/// literal kept in place).
#[test]
fn fused_composite_is_the_traced_pointwise_chain() {
    let local = |index: usize| FusedOperand::Local(LocalId::new(index));
    let step = |op, inputs| FusedStep { op, inputs };
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![
            step(FusedOp::Unary(UnOp::Neg), vec![local(0)]),
            step(FusedOp::Unary(UnOp::Exp), vec![local(2)]),
            step(
                FusedOp::Binary(BinOp::Add),
                vec![local(3), FusedOperand::Lit(Scalar::F32(1.0))],
            ),
            step(FusedOp::Binary(BinOp::Div), vec![local(0), local(4)]),
            step(FusedOp::Binary(BinOp::Mul), vec![local(5), local(1)]),
        ],
        output: LocalId::new(6),
        pack: Vec::new(),
    };
    let inputs = [data(&[3, 7], 27, 3.0), data(&[3, 7], 28, 1.0)];
    assert_composite_matches_chain("fused swiglu", OpKind::Fused(region), &inputs, |b, t| {
        let silu = ops::silu(b, t[0]).unwrap();
        b.binary(BinOp::Mul, silu, t[1]).unwrap()
    });
}

/// SC-002 (`Fused`, I32 with pack lanes): an exact-I32 region reads two lanes of a packed input and
/// packs two results, which is the un-fused unit `Slice`s, wrapping arithmetic and last-axis `Concat`.
#[test]
fn i32_fused_composite_with_pack_lanes_is_the_traced_word_chain() {
    let lane = |lane| FusedOperand::PackLane {
        input: LocalId::new(0),
        lane,
    };
    let step = |op, inputs| FusedStep { op, inputs };
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![
            step(FusedOp::Binary(BinOp::Add), vec![lane(0), lane(1)]),
            step(
                FusedOp::Binary(BinOp::Mul),
                vec![
                    FusedOperand::Local(LocalId::new(2)),
                    FusedOperand::Lit(Scalar::I32(3)),
                ],
            ),
            step(
                FusedOp::Binary(BinOp::Sub),
                vec![
                    FusedOperand::Local(LocalId::new(3)),
                    FusedOperand::Local(LocalId::new(1)),
                ],
            ),
        ],
        output: LocalId::new(4),
        pack: vec![LocalId::new(2), LocalId::new(4)],
    };
    // Words near the I32 range ends, so the region's wrapping arithmetic is exercised.
    let packed = HostTensor::i32(
        vec![3, 2],
        vec![
            i32::MAX - 1,
            5,
            -7,
            i32::MIN + 2,
            1_000_000_007,
            123_456_789,
        ],
    );
    let other = HostTensor::i32(vec![3, 1], vec![9, i32::MAX, -42]);
    assert_composite_matches_chain(
        "I32 fused with pack lanes",
        OpKind::Fused(region),
        &[packed, other],
        |b, t| {
            let first = b.slice(t[0], 1, 0, 1).unwrap();
            let second = b.slice(t[0], 1, 1, 2).unwrap();
            let sum = b.binary(BinOp::Add, first, second).unwrap();
            let scaled = b.binary_scalar(BinOp::Mul, sum, Scalar::I32(3)).unwrap();
            let difference = b.binary(BinOp::Sub, scaled, t[1]).unwrap();
            b.concat(1, &[sum, difference]).unwrap()
        },
    );
}

/// SC-002 (`FusedRow`): a softmax row region is `ops::softmax`, reductions keepdim over the row axis.
#[test]
fn fused_row_composite_is_the_traced_softmax() {
    let local = |index: usize| FusedOperand::Local(LocalId::new(index));
    let pointwise = |op, inputs| RowStep::Pointwise { op, inputs };
    let region = RowRegion {
        n_inputs: 1,
        axis: 1,
        steps: vec![
            RowStep::Reduce {
                op: RedOp::Max,
                input: local(0),
            },
            pointwise(FusedOp::Binary(BinOp::Sub), vec![local(0), local(1)]),
            pointwise(FusedOp::Unary(UnOp::Exp), vec![local(2)]),
            RowStep::Reduce {
                op: RedOp::Sum,
                input: local(3),
            },
            pointwise(FusedOp::Binary(BinOp::Div), vec![local(3), local(4)]),
        ],
        output: LocalId::new(5),
    };
    let inputs = [data(&[3, 7], 29, 4.0)];
    assert_composite_matches_chain(
        "fused-row softmax",
        OpKind::FusedRow(region),
        &inputs,
        |b, t| ops::softmax(b, t[0]).unwrap(),
    );
}

/// SC-003: the four rope entry points end in one half-split pairing, so on equivalent inputs (one
/// position for every section, a batch of one or the same rows batched) they agree bit for bit.
#[test]
fn every_rope_entry_point_agrees_bit_for_bit() {
    let (heads, max_pos) = (2usize, 12usize);
    // (head dim, rotary width, sections over rot/2, positions per batch row)
    let table: [(usize, usize, &[usize], &[i32]); 4] = [
        (8, 8, &[2, 1, 1], &[5]),
        (8, 4, &[1, 1], &[3]),
        (16, 16, &[3, 2, 3], &[0, 11]),
        (16, 8, &[2, 2], &[7, 2]),
    ];
    for (row, &(d, rot, sections, positions)) in table.iter().enumerate() {
        let batch = positions.len();
        let seed = 100 * row as u64;
        let x = data(&[batch, heads, 1, d], seed + 1, 1.0);
        let cos = data(&[max_pos, rot], seed + 2, 1.0);
        let sin = data(&[max_pos, rot], seed + 3, 1.0);
        let pos = HostTensor::i32(vec![batch], positions.to_vec());
        let what = |entry: &str| format!("row {row} (d={d}, rot={rot}): {entry}");

        let batched = evaluate(&graph_over(
            &[x.clone(), cos.clone(), sin.clone(), pos.clone()],
            |b, t| ops::rope_batched(b, t[0], t[1], t[2], t[3], batch).unwrap(),
        ))
        .unwrap();
        let sectioned_batched = evaluate(&graph_over(
            &[x.clone(), cos.clone(), sin.clone(), pos.clone()],
            |b, t| {
                let pos = vec![t[3]; sections.len()];
                ops::rope_sectioned_batched(b, t[0], t[1], t[2], &pos, sections, batch).unwrap()
            },
        ))
        .unwrap();
        assert_bits_equal(
            &sectioned_batched,
            &batched,
            &what("rope_sectioned_batched"),
        );

        // Each batch row alone, at its one position, through `rope` and `rope_sectioned`.
        let x_rows = x.as_f32().unwrap().chunks(heads * d);
        let batched_rows = batched.as_f32().unwrap().chunks(heads * d);
        for ((&position, x_row), batched_row) in positions.iter().zip(x_rows).zip(batched_rows) {
            let x_row = HostTensor::f32(vec![1, heads, 1, d], x_row.to_vec());
            let batched_row = HostTensor::f32(vec![1, heads, 1, d], batched_row.to_vec());
            let scalar = HostTensor::i32(Vec::new(), vec![position]);
            let rope = evaluate(&graph_over(
                &[x_row.clone(), cos.clone(), sin.clone(), scalar.clone()],
                |b, t| ops::rope(b, t[0], t[1], t[2], t[3]).unwrap(),
            ))
            .unwrap();
            let sectioned = evaluate(&graph_over(
                &[x_row, cos.clone(), sin.clone(), scalar],
                |b, t| {
                    let pos = vec![t[3]; sections.len()];
                    ops::rope_sectioned(b, t[0], t[1], t[2], &pos, sections).unwrap()
                },
            ))
            .unwrap();
            assert_bits_equal(&batched_row, &rope, &what("rope_batched vs rope"));
            assert_bits_equal(&sectioned, &rope, &what("rope_sectioned vs rope"));
        }
    }
}

/// SC-005: a prefill attention composite's decomposition materializes `[1, Hq, L, L]` scores behind a
/// `[1, Hq, L, D]` result and evaluates to the traced chain's bits.
#[test]
fn prefill_composite_evaluates_to_the_traced_chain() {
    let (hq, l, d) = (2usize, 16usize, 4usize);
    let scale = 0.5f32;
    let inputs = [
        data(&[1, hq, l, d], 30, 1.0),
        data(&[1, hq, l, d], 31, 1.0),
        data(&[1, hq, l, d], 32, 1.0),
        causal_mask(&[1, 1, l, l], 0),
    ];
    let op = OpKind::Attention(AttentionAttrs::new(1, scale, None));
    let graph = one_equation(op, &inputs);
    let composite = evaluate(&graph).expect("the composite evaluates");
    let traced = evaluate(&graph_over(&inputs, |b, t| {
        ops::attention_prefill(b, t[0], t[1], t[2], 1, scale, t[3]).unwrap()
    }))
    .unwrap();
    assert_bits_equal(&composite, &traced, "prefill composite");
    assert_eq!(composite.dtype(), DType::F32);
}

/// Card 556 review: an error a primitive raises inside a decomposition names the composite equation,
/// the id the caller's graph holds, never an id from the decomposition's own graph. An I32 region's
/// `RemU` by a zero word is refused at evaluation; the refusal carries the `Fused` equation's id and
/// keeps the primitive's label in its detail. Mutation: return the inner error unmapped
/// (`attributed_to` skipped): the refusal names `v0`, the decomposition's first placeholder, and the
/// row goes red.
#[test]
fn an_error_inside_a_decomposition_names_the_composite_equation() {
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![FusedStep {
            op: FusedOp::Binary(BinOp::RemU),
            inputs: vec![
                FusedOperand::Local(LocalId::new(0)),
                FusedOperand::Local(LocalId::new(1)),
            ],
        }],
        output: LocalId::new(2),
        pack: Vec::new(),
    };
    let inputs = [
        HostTensor::i32(vec![2], vec![7, 9]),
        HostTensor::i32(vec![2], vec![3, 0]),
    ];
    let graph = one_equation(OpKind::Fused(region), &inputs);
    let composite = graph.0.eqns()[0].out;
    assert_ne!(
        composite,
        ValueId::new(0),
        "the composite's id must differ from the inner placeholder's"
    );
    match evaluate(&graph) {
        Err(EvalError::Unsupported { eqn, op, detail }) => {
            assert_eq!(
                (eqn, op),
                (composite, "fused"),
                "the refusal names the composite equation: {detail}"
            );
            assert!(
                detail.starts_with("binary in its decomposition: "),
                "{detail}"
            );
        }
        other => panic!("a zero RemU divisor must be refused: {other:?}"),
    }
}

/// The output as f32 values (an exact widening of every dtype the oracle holds).
fn widened(tensor: &HostTensor) -> Vec<f32> {
    tensor.to_f32().expect("a float tensor").into_owned()
}

/// A `[1, 1]` epilogue region `head <op> leaf`, one step.
fn one_step_region(op: BinOp) -> FusedRegion {
    FusedRegion {
        n_inputs: 2,
        steps: vec![FusedStep {
            op: FusedOp::Binary(op),
            inputs: vec![
                FusedOperand::Local(LocalId::new(0)),
                FusedOperand::Local(LocalId::new(1)),
            ],
        }],
        output: LocalId::new(2),
        pack: Vec::new(),
    }
}

/// SC-006 (CPU oracle; BF16 and F16): a narrow contraction head rounds to its dtype before the epilogue reads
/// it. `A = [1, 1]` against `B = [1, tie]^T`, where `tie` is half a unit in the last place of 1.0 (`2^-8` for
/// BF16, `2^-11` for F16), accumulates to exactly halfway between 1.0 and its successor; round-to-nearest-even
/// rounds the head to 1.0, so the residual `-1` leaves zero. The composite, evaluated through its
/// decomposition, and the traced matmul-then-add are bit-identical and zero; an epilogue that read the
/// unrounded accumulator would give `tie`.
///
/// This is the reference every device row of the same name compares against: the oracle's zero is what
/// ROCm's fused and unfused programs must reproduce. Its mutation lives on the device rows (remove the head's
/// narrow and widen from the generated kernel: the device yields `tie` instead of zero).
#[test]
fn a_narrow_head_rounds_before_its_residual_in_the_composite_and_the_chain() {
    // (dtype, words of 1.0, words of the tie, words of -1.0)
    let cases = [
        (DType::BF16, 0x3f80u16, 0x3b80u16, 0xbf80u16),
        (DType::F16, 0x3c00, 0x1000, 0xbc00),
    ];
    for (dtype, one, tie, minus_one) in cases {
        let tensor = |shape: &[usize], words: Vec<u16>| match dtype {
            DType::BF16 => HostTensor::bf16(shape.to_vec(), words),
            _ => HostTensor::f16(shape.to_vec(), words),
        };
        let inputs = [
            tensor(&[1, 2], vec![one, one]),
            tensor(&[2, 1], vec![one, tie]),
            tensor(&[1, 1], vec![minus_one]),
        ];
        let op = OpKind::ContractionEpilogue {
            head: ContractionHead::MatMul,
            region: one_step_region(BinOp::Add),
        };
        let composite = evaluate(&one_equation(op, &inputs)).unwrap();
        let chain = evaluate(&graph_over(&inputs, |b, t| {
            let head = b.matmul(t[0], t[1]).unwrap();
            b.binary(BinOp::Add, head, t[2]).unwrap()
        }))
        .unwrap();
        assert_eq!(
            composite.dtype(),
            dtype,
            "{dtype:?}: the epilogue keeps the head's dtype"
        );
        assert_eq!(
            widened(&composite),
            [0.0],
            "{dtype:?}: the head rounds to 1.0 first"
        );
        assert_eq!(
            widened(&composite)[0].to_bits(),
            widened(&chain)[0].to_bits(),
            "{dtype:?}: the composite equals the traced chain"
        );
    }
}

/// SC-006 (CPU oracle; multistage narrow pointwise): every step of an epilogue over a narrow head rounds through
/// the head's dtype. `1.0 + tie` rounds to 1.0 (a tie to even) at each of two adds, so the chain ends at 1.0; an
/// epilogue that kept the intermediate in f32 would end at `1 + 2 tie`, which BF16 holds exactly.
#[test]
fn every_step_of_a_narrow_epilogue_rounds_through_the_head_dtype() {
    let (one, tie) = (0x3f80u16, 0x3b80u16);
    let tensor = |words: Vec<u16>| HostTensor::bf16(vec![1, 1], words);
    let inputs = [tensor(vec![one]), tensor(vec![one]), tensor(vec![tie])];
    let local = |index: usize| FusedOperand::Local(LocalId::new(index));
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![
            FusedStep {
                op: FusedOp::Binary(BinOp::Add),
                inputs: vec![local(0), local(1)],
            },
            FusedStep {
                op: FusedOp::Binary(BinOp::Add),
                inputs: vec![local(2), local(1)],
            },
        ],
        output: LocalId::new(3),
        pack: Vec::new(),
    };
    let op = OpKind::ContractionEpilogue {
        head: ContractionHead::MatMul,
        region,
    };
    // Inputs: A = [1], B = [1] (head 1.0), then the leaf `tie` read by both steps.
    let composite = evaluate(&one_equation(
        op,
        &[inputs[0].clone(), inputs[1].clone(), inputs[2].clone()],
    ))
    .unwrap();
    assert_eq!(
        widened(&composite),
        [1.0],
        "each step rounds to nearest even"
    );
}
