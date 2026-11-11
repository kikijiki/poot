//! The stable rank composite (POOT-1543) through the real Builder, inference, decomposition and evaluator
//! seams: its primitive definition against an independent plain-Rust sort of `(class, canonical score, index)`
//! tuples, the masked class `eligible >= 1`, and its typing refusals.

use std::collections::HashMap;

use poot_graph_ir::test_support::composites;
use poot_graph_ir::test_support::stable_rank::{eligibility_rows, reference_rank, score_rows};
use poot_graph_ir::{
    Builder, OpKind, Outputs, ShapeError, StableRankAttrs, TensorType, ValueId, ops,
};
use poot_tensor::HostTensor;

use crate::{EvalOptions, Value, eval};

/// The composite's evaluation over `rows` rows of `scores` and, when given, the eligibility operand values.
fn composite_ranks(scores: &[Vec<f32>], eligible: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
    let (rows, e) = (scores.len(), scores[0].len());
    let b = Builder::for_tests();
    let ty = TensorType::f32(vec![rows, e]);
    let s = b.constant("scores", ty.clone()).unwrap();
    let m = eligible.map(|_| b.constant("eligible", ty.clone()).unwrap());
    let rank = composites::stable_rank(&b, s, m).unwrap();
    let graph = b.finish(Outputs::new(rank)).unwrap();
    let mut values: HashMap<ValueId, Value> = HashMap::from([(
        s.id(),
        HostTensor::f32(vec![rows, e], scores.concat()).into(),
    )]);
    if let (Some(m), Some(eligible)) = (m, eligible) {
        values.insert(
            m.id(),
            HostTensor::f32(vec![rows, e], eligible.concat()).into(),
        );
    }
    let out = eval(&graph, &values, EvalOptions::default()).unwrap();
    let out = out.output.into_host().expect("dense result");
    out.as_f32()
        .expect("an F32 rank")
        .chunks(e)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The primitive chain the composite replaces, evaluated directly: the two must agree bit for bit.
fn chain_ranks(scores: &[Vec<f32>], eligible: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
    let (rows, e) = (scores.len(), scores[0].len());
    let b = Builder::for_tests();
    let ty = TensorType::f32(vec![rows, e]);
    let s = b.constant("scores", ty.clone()).unwrap();
    let m = eligible.map(|_| b.constant("eligible", ty.clone()).unwrap());
    let rank = match m {
        Some(m) => ops::stable_descending_rank_masked(&b, s, m).unwrap(),
        None => ops::stable_descending_rank(&b, s).unwrap(),
    };
    let graph = b.finish(Outputs::new(rank)).unwrap();
    let mut values: HashMap<ValueId, Value> = HashMap::from([(
        s.id(),
        HostTensor::f32(vec![rows, e], scores.concat()).into(),
    )]);
    if let (Some(m), Some(eligible)) = (m, eligible) {
        values.insert(
            m.id(),
            HostTensor::f32(vec![rows, e], eligible.concat()).into(),
        );
    }
    let out = eval(&graph, &values, EvalOptions::default()).unwrap();
    let out = out.output.into_host().expect("dense result");
    out.as_f32()
        .expect("an F32 rank")
        .chunks(e)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The composite equals the plain sort and the primitive chain on every row, bit for bit (design section 5:
/// the oracle evaluates the decomposition). Mutations in `ops/moe.rs`: the tie-break `j <= i` for `j < i`
/// (higher index first on equal scores), or no `-0` equality (drop the canonicalization).
#[test]
fn the_definition_ranks_what_the_plain_sort_ranks() {
    for e in [1, 2, 5, 8, 31] {
        let scores = score_rows(e);
        let want: Vec<Vec<f32>> = scores.iter().map(|row| reference_rank(row, None)).collect();
        let got = composite_ranks(&scores, None);
        assert_eq!(got, want, "E={e}: the composite against the plain sort");
        assert_eq!(chain_ranks(&scores, None), got, "E={e}: composite vs chain");
    }
}

/// The masked rank puts the eligible class first, then the same order inside each class.
#[test]
fn the_masked_definition_ranks_the_eligible_class_first() {
    let e = 9;
    let scores = score_rows(e);
    let masks = eligibility_rows(scores.len(), e);
    let want: Vec<Vec<f32>> = scores
        .iter()
        .zip(&masks)
        .map(|(row, mask)| reference_rank(row, Some(mask)))
        .collect();
    let got = composite_ranks(&scores, Some(&masks));
    assert_eq!(got, want);
    assert_eq!(chain_ranks(&scores, Some(&masks)), got);
}

/// Eligibility is a class: a value of 0.5 is not eligible, 2.0 is, and NaN is not. Mutation in `ops/moe.rs`:
/// drop the `Ge(eligible, 1.0)` so the operand's value is read as a number.
#[test]
fn eligibility_is_the_class_eligible_at_least_one() {
    let scores = vec![vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]];
    let masks = vec![vec![0.5, 2.0, f32::NAN, 1.0, 0.0, 1.0]];
    let got = composite_ranks(&scores, Some(&masks));
    assert_eq!(got, vec![reference_rank(&scores[0], Some(&masks[0]))]);
    // Eligible 6, 5, 4 first, then 3, 2, 1: the 0.5 and NaN entries are in the ineligible class.
    assert_eq!(got[0], [5.0, 1.0, 3.0, 2.0, 4.0, 0.0]);
}

/// The typing rule: F32 operands, a last axis of at least one element, and an eligibility of the scores' shape.
#[test]
fn the_typing_rule_refuses_what_it_cannot_rank() {
    let unmasked = OpKind::StableRank(StableRankAttrs { masked: false });
    let masked = OpKind::StableRank(StableRankAttrs { masked: true });
    let f32t = |shape: &[usize]| TensorType::f32(shape.to_vec());
    assert_eq!(unmasked.infer(&[f32t(&[3, 8])]).unwrap(), f32t(&[3, 8]));
    assert_eq!(
        masked.infer(&[f32t(&[3, 8]), f32t(&[3, 8])]).unwrap(),
        f32t(&[3, 8])
    );
    assert!(matches!(
        unmasked.infer(&[f32t(&[3, 0])]).unwrap_err(),
        ShapeError::StableRankNoExperts
    ));
    assert!(matches!(
        unmasked.infer(&[f32t(&[])]).unwrap_err(),
        ShapeError::RankTooLow { .. }
    ));
    assert!(matches!(
        masked.infer(&[f32t(&[3, 8]), f32t(&[3, 7])]).unwrap_err(),
        ShapeError::StableRankEligibility { .. }
    ));
    assert!(matches!(
        masked.infer(&[f32t(&[3, 8])]).unwrap_err(),
        ShapeError::Arity {
            expected: 2,
            got: 1
        }
    ));
    assert!(matches!(
        unmasked
            .infer(&[TensorType::new(vec![3, 8], poot_tensor::DType::I32)])
            .unwrap_err(),
        ShapeError::DtypeOp { .. }
    ));
}
