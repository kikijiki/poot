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

use poot_graph_ir::op::{FusedOp, FusedOperand, FusedRegion, FusedStep, RowRegion, RowStep};
use poot_graph_ir::{
    BinOp, Builder, Graph, OpKind, Operand, RedOp, Scalar, TensorType, Traced, UnOp, ValueId, ops,
};
use poot_tensor::{DType, HostTensor};

use super::helpers::fill;
use crate::{EvalBudget, EvalError, EvalOptions, Value, eval};

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
    let b = Builder::new();
    let traced: Vec<Traced> = inputs
        .iter()
        .enumerate()
        .map(|(position, tensor)| {
            b.constant(
                &format!("input.{position}"),
                TensorType::new(tensor.shape().to_vec(), tensor.dtype()),
            )
        })
        .collect();
    let out = body(&b, &traced);
    let bound = traced
        .iter()
        .zip(inputs)
        .map(|(traced, tensor)| (traced.id, Value::from(tensor.clone())))
        .collect();
    (b.finish(out), bound)
}

/// A one-equation graph holding `op` over `inputs`. A composite is compiler-only (`Builder` has no
/// method for one), so it is staged through an append plan.
fn one_equation(op: OpKind, inputs: &[HostTensor]) -> (Graph, HashMap<ValueId, Value>) {
    graph_over(inputs, |b, traced| {
        let mut plan = b.append_plan(0);
        let operands = traced.iter().map(|t| Operand::Value(t.id)).collect();
        let out = plan.equation(op, operands).expect("the composite types");
        plan.declare_result(out).expect("declare the result");
        let mut prepared = b.preflight_append(plan).expect("preflight the append");
        Traced {
            id: b.commit_append(&mut prepared).expect("commit the append"),
        }
    })
}

fn evaluate(
    (g, inputs): &(Graph, HashMap<ValueId, Value>),
    budget: EvalBudget,
) -> Result<HostTensor, EvalError> {
    let output = eval(g, inputs, EvalOptions::new(budget))?.output;
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
    let composite = evaluate(&one_equation(op, inputs), EvalBudget::UNBOUNDED)
        .unwrap_or_else(|error| panic!("{what}: the composite evaluates: {error}"));
    let traced = evaluate(&graph_over(inputs, chain), EvalBudget::UNBOUNDED)
        .unwrap_or_else(|error| panic!("{what}: the traced chain evaluates: {error}"));
    assert_bits_equal(&composite, &traced, what);
}

/// SC-002 (decode): `FlashAttentionDecode { n_rep, scale }` is `ops::attention_masked` with the same
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
        OpKind::FlashAttentionDecode { n_rep, scale },
        &inputs,
        |b, t| ops::attention_masked(b, t[0], t[1], t[2], n_rep, scale, t[3]),
    );
}

/// SC-002 (prefill, no softcap): `FlashAttentionPrefill { softcap: None }` is `ops::attention_prefill`,
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
        OpKind::FlashAttentionPrefill {
            n_rep,
            scale,
            softcap: None,
        },
        &inputs,
        |b, t| ops::attention_prefill(b, t[0], t[1], t[2], n_rep, scale, t[3]),
    );
}

/// SC-002 (prefill, softcap): `FlashAttentionPrefill { softcap: Some(c) }` is
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
        OpKind::FlashAttentionPrefill {
            n_rep,
            scale,
            softcap: Some(cap),
        },
        &inputs,
        |b, t| ops::attention_prefill_softcap(b, t[0], t[1], t[2], n_rep, scale, t[3], Some(cap)),
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
            |b, t| ops::rope_partial(b, t[0], t[1], t[2], d, rot, 3),
        );
    }
}

/// SC-002 (`MatMulBias`): the contraction then the `[N]` bias add, `ops::linear` with the bias.
#[test]
fn matmul_bias_composite_is_the_traced_biased_linear() {
    let inputs = [
        data(&[2, 3, 5], 24, 1.0),
        data(&[5, 4], 25, 1.0),
        data(&[4], 26, 1.0),
    ];
    assert_composite_matches_chain("matmul_bias", OpKind::MatMulBias, &inputs, |b, t| {
        ops::linear(b, t[0], t[1], Some(t[2]))
    });
}

/// SC-002 (`Fused`, F32): a SwiGLU region is `ops::silu(x) * y`, step for step (operand order and
/// literal kept in place).
#[test]
fn fused_composite_is_the_traced_pointwise_chain() {
    let local = FusedOperand::Local;
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
        output: 6,
        pack: Vec::new(),
    };
    let inputs = [data(&[3, 7], 27, 3.0), data(&[3, 7], 28, 1.0)];
    assert_composite_matches_chain("fused swiglu", OpKind::Fused(region), &inputs, |b, t| {
        let silu = ops::silu(b, t[0]);
        b.binary(BinOp::Mul, silu, t[1])
    });
}

/// SC-002 (`Fused`, I32 with pack lanes): an exact-I32 region reads two lanes of a packed input and
/// packs two results, which is the un-fused unit `Slice`s, wrapping arithmetic and last-axis `Concat`.
#[test]
fn i32_fused_composite_with_pack_lanes_is_the_traced_word_chain() {
    let lane = |lane| FusedOperand::PackLane { input: 0, lane };
    let step = |op, inputs| FusedStep { op, inputs };
    let region = FusedRegion {
        n_inputs: 2,
        steps: vec![
            step(FusedOp::Binary(BinOp::Add), vec![lane(0), lane(1)]),
            step(
                FusedOp::Binary(BinOp::Mul),
                vec![FusedOperand::Local(2), FusedOperand::Lit(Scalar::I32(3))],
            ),
            step(
                FusedOp::Binary(BinOp::Sub),
                vec![FusedOperand::Local(3), FusedOperand::Local(1)],
            ),
        ],
        output: 4,
        pack: vec![2, 4],
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
            let first = b.slice(t[0], 1, 0, 1);
            let second = b.slice(t[0], 1, 1, 2);
            let sum = b.binary(BinOp::Add, first, second);
            let scaled = b.binary_scalar(BinOp::Mul, sum, Scalar::I32(3));
            let difference = b.binary(BinOp::Sub, scaled, t[1]);
            b.concat(1, &[sum, difference])
        },
    );
}

/// SC-002 (`FusedRow`): a softmax row region is `ops::softmax`, reductions keepdim over the row axis.
#[test]
fn fused_row_composite_is_the_traced_softmax() {
    let local = FusedOperand::Local;
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
        output: 5,
    };
    let inputs = [data(&[3, 7], 29, 4.0)];
    assert_composite_matches_chain(
        "fused-row softmax",
        OpKind::FusedRow(region),
        &inputs,
        |b, t| ops::softmax(b, t[0]),
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

        let batched = evaluate(
            &graph_over(
                &[x.clone(), cos.clone(), sin.clone(), pos.clone()],
                |b, t| ops::rope_batched(b, t[0], t[1], t[2], t[3], batch),
            ),
            EvalBudget::UNBOUNDED,
        )
        .unwrap();
        let sectioned_batched = evaluate(
            &graph_over(
                &[x.clone(), cos.clone(), sin.clone(), pos.clone()],
                |b, t| {
                    let pos = vec![t[3]; sections.len()];
                    ops::rope_sectioned_batched(b, t[0], t[1], t[2], &pos, sections, batch)
                },
            ),
            EvalBudget::UNBOUNDED,
        )
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
            let rope = evaluate(
                &graph_over(
                    &[x_row.clone(), cos.clone(), sin.clone(), scalar.clone()],
                    |b, t| ops::rope(b, t[0], t[1], t[2], t[3]),
                ),
                EvalBudget::UNBOUNDED,
            )
            .unwrap();
            let sectioned = evaluate(
                &graph_over(&[x_row, cos.clone(), sin.clone(), scalar], |b, t| {
                    let pos = vec![t[3]; sections.len()];
                    ops::rope_sectioned(b, t[0], t[1], t[2], &pos, sections)
                }),
                EvalBudget::UNBOUNDED,
            )
            .unwrap();
            assert_bits_equal(&batched_row, &rope, &what("rope_batched vs rope"));
            assert_bits_equal(&sectioned, &rope, &what("rope_sectioned vs rope"));
        }
    }
}

/// SC-005: a prefill attention composite's decomposition materializes `[1, Hq, L, L]` scores behind a
/// `[1, Hq, L, D]` result. Under a budget below the scores and above every operand and the result, the
/// evaluation is a typed `Budget` error naming the composite equation (nothing publishes); unbounded,
/// it evaluates to the traced chain's bits.
#[test]
fn prefill_composite_scores_are_charged_to_the_budget() {
    let (hq, l, d) = (2usize, 16usize, 4usize);
    let scale = 0.5f32;
    let inputs = [
        data(&[1, hq, l, d], 30, 1.0),
        data(&[1, hq, l, d], 31, 1.0),
        data(&[1, hq, l, d], 32, 1.0),
        causal_mask(&[1, 1, l, l], 0),
    ];
    let op = OpKind::FlashAttentionPrefill {
        n_rep: 1,
        scale,
        softcap: None,
    };
    let graph = one_equation(op, &inputs);
    let composite = graph.0.eqns[0].out;
    let scores = hq * l * l;
    let limit = 300usize;
    assert!(hq * l * d < limit && l * l < limit && scores > limit);

    match evaluate(&graph, EvalBudget::bounded(limit as u64, u64::MAX)) {
        Err(EvalError::Budget {
            eqn,
            resource,
            needed,
            limit: refused_at,
        }) => {
            assert_eq!(
                (eqn, resource, needed, refused_at),
                (composite, "elements", scores, limit)
            );
        }
        other => panic!("a bounded prefill composite must be refused by the budget: {other:?}"),
    }

    let unbounded = evaluate(&graph, EvalBudget::UNBOUNDED).expect("unbounded evaluates");
    let traced = evaluate(
        &graph_over(&inputs, |b, t| {
            ops::attention_prefill(b, t[0], t[1], t[2], 1, scale, t[3])
        }),
        EvalBudget::UNBOUNDED,
    )
    .unwrap();
    assert_bits_equal(&unbounded, &traced, "unbounded prefill composite");
    assert_eq!(unbounded.dtype(), DType::F32);
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
            inputs: vec![FusedOperand::Local(0), FusedOperand::Local(1)],
        }],
        output: 2,
        pack: Vec::new(),
    };
    let inputs = [
        HostTensor::i32(vec![2], vec![7, 9]),
        HostTensor::i32(vec![2], vec![3, 0]),
    ];
    let graph = one_equation(OpKind::Fused(region), &inputs);
    let composite = graph.0.eqns[0].out;
    assert_ne!(
        composite, 0,
        "the composite's id must differ from the inner placeholder's"
    );
    match evaluate(&graph, EvalBudget::UNBOUNDED) {
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
