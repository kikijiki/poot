//! Card 372c: the helper emits exactly the shape the recognizer accepts, and every edit of that shape is
//! rejected with the reason that names it.

use super::*;
use crate::builder::Builder;
use crate::graph::ValidationId;
use crate::types::TensorType;

const UPPER: usize = 4;

fn i32_input(builder: &Builder, name: &str, shape: Vec<usize>) -> Traced {
    builder.constant(name, TensorType::new(shape, DType::I32))
}

/// A graph whose primary output is the guarded selector and whose only validation output is the witness.
fn guarded_graph() -> (crate::Graph<crate::ValidationOutputs>, ValueId, ValueId) {
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let graph = crate::test_support::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(7), "bounds", guard.witness)],
    )
    .unwrap();
    (graph, guard.guarded.id, guard.witness.id)
}

#[test]
fn guard_helper_emits_the_chain_the_recognizer_accepts() {
    let (graph, guarded, witness) = guarded_graph();
    let recognized = recognize_index_bounds_guard(&graph, guarded, witness).unwrap();

    assert_eq!(recognized.guarded, guarded);
    assert_eq!(recognized.witness, witness);
    assert_eq!(recognized.upper_exclusive, UPPER as i32);
    // The guard condition and the witness observation are one value, not two that agree.
    let cast = graph
        .eqns
        .iter()
        .find(|eqn| eqn.out == recognized.invalid_f32)
        .unwrap();
    assert!(
        matches!(cast.inputs.as_slice(), [Operand::Value(value)] if *value == recognized.invalid)
    );
    // The declared validation value is the witness, and it is a single F32 lane.
    assert_eq!(graph.validation_outputs().len(), 1);
    assert_eq!(graph.validation_outputs()[0].value, witness);
    assert_eq!(graph.aval(witness).dtype, DType::F32);
    assert_eq!(graph.aval(witness).numel(), 1);
    // The comparison is unsigned, which is what makes one compare cover both bounds.
    let condition = graph
        .eqns
        .iter()
        .find(|eqn| eqn.out == recognized.invalid)
        .unwrap();
    assert_eq!(condition.op, OpKind::Binary(BinOp::GeU));
    assert!(matches!(
        condition.inputs.as_slice(),
        [Operand::Value(value), Operand::Lit(Scalar::I32(bound))]
            if *value == recognized.raw && *bound == UPPER as i32
    ));
    // The fallback is a value, not a literal, because the Select kernel needs three value leaves.
    let select = graph.eqns.iter().find(|eqn| eqn.out == guarded).unwrap();
    assert!(
        select
            .inputs
            .iter()
            .all(|operand| matches!(operand, Operand::Value(_)))
    );
}

#[test]
fn guard_helper_rejects_a_selector_it_cannot_guard() {
    let builder = Builder::new();
    let float = builder.constant("x", TensorType::f32(vec![4]));
    assert_eq!(
        builder.guard_index_bounds(float, UPPER).unwrap_err(),
        IndexGuardError::RawNotI32 {
            raw: float.id,
            dtype: DType::F32
        }
    );

    let scalar = i32_input(&builder, "scalar", vec![]);
    assert_eq!(
        builder.guard_index_bounds(scalar, UPPER).unwrap_err(),
        IndexGuardError::RawNotRanked { raw: scalar.id }
    );

    let raw = i32_input(&builder, "raw", vec![4]);
    assert_eq!(
        builder.guard_index_bounds(raw, usize::MAX).unwrap_err(),
        IndexGuardError::BoundNotRepresentable {
            upper_exclusive: usize::MAX
        }
    );
}

/// Each case rebuilds the chain by hand with exactly one edge changed, and asserts the reason, so a
/// dropped recognizer check turns exactly the case that names it red.
#[test]
fn guard_recognizer_rejects_each_noncanonical_edit() {
    // A signed comparison: a negative lane would pass, so this must not be recognized.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let invalid = builder.binary_scalar(BinOp::Ge, raw, Scalar::I32(UPPER as i32));
    let zero = builder.binary(BinOp::Sub, raw, raw);
    let guarded = builder.select(invalid, zero, raw);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::ConditionNotUnsignedBound {
            invalid: invalid.id
        })
    );

    // The condition bounds a different value than the one the Select falls back to.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let other = i32_input(&builder, "other", vec![4]);
    let invalid = builder.binary_scalar(BinOp::GeU, other, Scalar::I32(UPPER as i32));
    let zero = builder.binary(BinOp::Sub, raw, raw);
    let guarded = builder.select(invalid, zero, raw);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::ConditionComparesAnotherValue {
            invalid: invalid.id,
            expected: raw.id,
            actual: other.id,
        })
    );

    // The fallback is not zero: an out-of-range lane would still reach the consumer. `Add(raw, raw)`
    // rather than a literal form, so the operand check alone cannot reject it and the `Binary(Sub)`
    // filter is exercised too.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let invalid = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
    let not_zero = builder.binary(BinOp::Add, raw, raw);
    let guarded = builder.select(invalid, not_zero, raw);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::FallbackNotZeroOfSource {
            zero: not_zero.id,
            expected: raw.id
        })
    );

    // The guarded value is not a Select at all.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let invalid = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
    let guarded = builder.binary(BinOp::Sub, raw, invalid);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::GuardedNotSelect {
            guarded: guarded.id
        })
    );
}

#[test]
fn guard_recognizer_rejects_a_witness_over_another_observation() {
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let other = i32_input(&builder, "other", vec![4]);
    let invalid = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
    let zero = builder.binary(BinOp::Sub, raw, raw);
    let guarded = builder.select(invalid, zero, raw);
    // A well-formed witness of the wrong flag: it passes for the very ids this guard clamps.
    let decoy = builder.binary_scalar(BinOp::GeU, other, Scalar::I32(UPPER as i32));
    let decoy_f32 = builder.cast(decoy, DType::F32);
    let witness = builder.reduce(RedOp::Sum, decoy_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::WitnessObservesAnotherValue {
            witness: witness.id,
            observed: decoy.id,
            invalid: invalid.id,
        })
    );
}

#[test]
fn guard_recognizer_rejects_a_witness_that_is_not_a_keepdim_last_axis_sum() {
    // A non-keepdim reduce and a Max each stop the packet from speaking for every lane of the flag.
    let variants: [fn(&Builder, Traced) -> Traced; 2] = [
        |builder, flag| builder.reduce(RedOp::Sum, flag, 0, false),
        |builder, flag| builder.reduce(RedOp::Max, flag, 0, true),
    ];
    for build in variants {
        let builder = Builder::new();
        let raw = i32_input(&builder, "raw", vec![4]);
        let invalid = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
        let zero = builder.binary(BinOp::Sub, raw, raw);
        let guarded = builder.select(invalid, zero, raw);
        let invalid_f32 = builder.cast(invalid, DType::F32);
        let witness = build(&builder, invalid_f32);
        let graph = crate::test_support::finish_with_validations(
            builder,
            guarded,
            &[(ValidationId(0), "bounds", witness)],
        )
        .unwrap();
        assert_eq!(
            recognize_index_bounds_guard(&graph, guarded.id, witness.id),
            Err(IndexGuardRejection::WitnessNotSumOfCast {
                witness: witness.id
            })
        );
    }

    // A rank-2 flag reduced over axis 0 leaves the last axis unreduced.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![2, 2]);
    let invalid = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
    let zero = builder.binary(BinOp::Sub, raw, raw);
    let guarded = builder.select(invalid, zero, raw);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(0), "bounds", witness)],
    )
    .unwrap();
    assert_eq!(
        recognize_index_bounds_guard(&graph, guarded.id, witness.id),
        Err(IndexGuardRejection::WitnessNotSumOfCast {
            witness: witness.id
        })
    );
}

/// Coverage rows for [`IndexGuardRejection`]. Exhaustive on purpose: a new rejection fails to compile
/// until it is classified here, the same way the authorization and exact-I32 error tables work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum IndexGuardRejectionRow {
    GuardedNotSelect,
    GuardedLiteralOperand,
    ConditionNotUnsignedBound,
    ConditionComparesAnotherValue,
    BoundNotI32Literal,
    FallbackNotZeroOfSource,
    WitnessNotSumOfCast,
    WitnessObservesAnotherValue,
}

const INDEX_GUARD_REJECTION_ROWS: [IndexGuardRejectionRow; 8] = [
    IndexGuardRejectionRow::GuardedNotSelect,
    IndexGuardRejectionRow::GuardedLiteralOperand,
    IndexGuardRejectionRow::ConditionNotUnsignedBound,
    IndexGuardRejectionRow::ConditionComparesAnotherValue,
    IndexGuardRejectionRow::BoundNotI32Literal,
    IndexGuardRejectionRow::FallbackNotZeroOfSource,
    IndexGuardRejectionRow::WitnessNotSumOfCast,
    IndexGuardRejectionRow::WitnessObservesAnotherValue,
];

fn index_guard_rejection_row(rejection: &IndexGuardRejection) -> IndexGuardRejectionRow {
    use IndexGuardRejection as R;
    match rejection {
        R::GuardedNotSelect { .. } => IndexGuardRejectionRow::GuardedNotSelect,
        R::GuardedLiteralOperand { .. } => IndexGuardRejectionRow::GuardedLiteralOperand,
        R::ConditionNotUnsignedBound { .. } => IndexGuardRejectionRow::ConditionNotUnsignedBound,
        R::ConditionComparesAnotherValue { .. } => {
            IndexGuardRejectionRow::ConditionComparesAnotherValue
        }
        R::BoundNotI32Literal { .. } => IndexGuardRejectionRow::BoundNotI32Literal,
        R::FallbackNotZeroOfSource { .. } => IndexGuardRejectionRow::FallbackNotZeroOfSource,
        R::WitnessNotSumOfCast { .. } => IndexGuardRejectionRow::WitnessNotSumOfCast,
        R::WitnessObservesAnotherValue { .. } => {
            IndexGuardRejectionRow::WitnessObservesAnotherValue
        }
    }
}

/// Build the canonical chain, hand the graph to `edit`, and recognize. `edit` returns the `(guarded,
/// witness)` pair to recognize, so a case can redirect either end.
fn recognize_edited(
    edit: impl FnOnce(
        &mut crate::Graph<crate::ValidationOutputs>,
        ValueId,
        ValueId,
    ) -> (ValueId, ValueId),
) -> Result<IndexBoundsGuard, IndexGuardRejection> {
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let mut graph = crate::test_support::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(1), "bounds", guard.witness)],
    )
    .unwrap();
    let (guarded, witness) = edit(&mut graph, guard.guarded.id, guard.witness.id);
    recognize_index_bounds_guard(&graph, guarded, witness)
}

#[test]
fn guard_recognizer_rejects_a_select_with_a_literal_operand() {
    // A literal fallback would leave the Select with two value leaves, which the planner's three-leaf
    // kernel cannot build; the recognizer refuses the shape before that becomes a lowering problem.
    let rejection = recognize_edited(|graph, guarded, witness| {
        let select = graph
            .eqns
            .iter_mut()
            .find(|eqn| eqn.out == guarded)
            .expect("the guard's Select");
        select.inputs[1] = Operand::Lit(Scalar::I32(0));
        (guarded, witness)
    })
    .unwrap_err();
    assert_eq!(
        index_guard_rejection_row(&rejection),
        IndexGuardRejectionRow::GuardedLiteralOperand
    );
}

#[test]
fn guard_recognizer_rejects_a_bound_that_is_not_an_i32_literal() {
    // A runtime bound is not a static bound: authorization could not prove it equals the expert count,
    // so the recognizer does not admit the chain at all.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let limit = i32_input(&builder, "limit", vec![4]);
    let invalid = builder.binary(BinOp::GeU, raw, limit);
    let zero = builder.binary(BinOp::Sub, raw, raw);
    let guarded = builder.select(invalid, zero, raw);
    let invalid_f32 = builder.cast(invalid, DType::F32);
    let witness = builder.reduce(RedOp::Sum, invalid_f32, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guarded,
        &[(ValidationId(1), "bounds", witness)],
    )
    .unwrap();
    let rejection = recognize_index_bounds_guard(&graph, guarded.id, witness.id).unwrap_err();
    assert_eq!(
        rejection,
        IndexGuardRejection::BoundNotI32Literal {
            invalid: invalid.id
        }
    );
}

#[test]
fn index_guard_rejection_table_covers_every_reason() {
    let mut covered = std::collections::BTreeSet::new();

    // Each case edits exactly one edge of the canonical chain.
    let cases: [fn() -> IndexGuardRejection; 10] = [
        // The guarded value is not a Select.
        || {
            recognize_edited(|graph, guarded, witness| {
                let select = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap();
                // Only the operation changes: the operands stay the guard's own three values, so the
                // `OpKind::Select` filter is the only thing that can reject this. Truncating the
                // operand list too would let the arity check mask a dropped filter.
                select.op = OpKind::Binary(BinOp::Sub);
                (guarded, witness)
            })
            .unwrap_err()
        },
        // A literal Select operand.
        || {
            recognize_edited(|graph, guarded, witness| {
                let select = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap();
                select.inputs[1] = Operand::Lit(Scalar::I32(0));
                (guarded, witness)
            })
            .unwrap_err()
        },
        // A signed comparison.
        || {
            recognize_edited(|graph, guarded, witness| {
                let select_condition = match graph
                    .eqns
                    .iter()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap()
                    .inputs[0]
                {
                    Operand::Value(value) => value,
                    Operand::Lit(_) => unreachable!("the helper emits value operands"),
                };
                let condition = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == select_condition)
                    .unwrap();
                condition.op = OpKind::Binary(BinOp::Ge);
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The witness is not a keepdim last-axis Sum of a Cast.
        || {
            recognize_edited(|graph, guarded, witness| {
                let reduce = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == witness)
                    .unwrap();
                reduce.op = OpKind::Reduce {
                    op: RedOp::Max,
                    axis: 0,
                    keepdim: true,
                };
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The condition bounds a value the Select does not fall back to.
        || {
            recognize_edited(|graph, guarded, witness| {
                let select = graph.eqns.iter().find(|eqn| eqn.out == guarded).unwrap();
                let (condition, zero) = match (&select.inputs[0], &select.inputs[1]) {
                    (Operand::Value(condition), Operand::Value(zero)) => (*condition, *zero),
                    _ => unreachable!("the helper emits value operands"),
                };
                let bound = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == condition)
                    .unwrap();
                bound.inputs[0] = Operand::Value(zero);
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The fallback is a Sub, but not of the guarded source twice, so the operand filter is the
        // only thing that rejects it and `Sub(a, b)` of two other values cannot pass as zero.
        || {
            recognize_edited(|graph, guarded, witness| {
                let (zero, condition) = match graph
                    .eqns
                    .iter()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap()
                    .inputs
                    .as_slice()
                {
                    [Operand::Value(condition), Operand::Value(zero), _] => (*zero, *condition),
                    _ => unreachable!("the helper emits three value operands"),
                };
                let fallback = graph.eqns.iter_mut().find(|eqn| eqn.out == zero).unwrap();
                fallback.inputs[1] = Operand::Value(condition);
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The bound is a literal, but not an I32 one, so it is not a static index bound.
        || {
            recognize_edited(|graph, guarded, witness| {
                let condition = match graph
                    .eqns
                    .iter()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap()
                    .inputs[0]
                {
                    Operand::Value(value) => value,
                    Operand::Lit(_) => unreachable!("the helper emits value operands"),
                };
                let bound = graph
                    .eqns
                    .iter_mut()
                    .find(|eqn| eqn.out == condition)
                    .unwrap();
                bound.inputs[1] = Operand::Lit(Scalar::F32(4.0));
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The fallback is not a Sub at all, so the operation filter is what rejects it.
        || {
            recognize_edited(|graph, guarded, witness| {
                let zero = match graph
                    .eqns
                    .iter()
                    .find(|eqn| eqn.out == guarded)
                    .unwrap()
                    .inputs[1]
                {
                    Operand::Value(value) => value,
                    Operand::Lit(_) => unreachable!("the helper emits value operands"),
                };
                let fallback = graph.eqns.iter_mut().find(|eqn| eqn.out == zero).unwrap();
                fallback.op = OpKind::Binary(BinOp::Add);
                (guarded, witness)
            })
            .unwrap_err()
        },
        // The witness reduces a single-operand value that is not a Cast to F32. Built rather than
        // edited, because the `Cast { to: F32 }` filter is only exercised when the operand shape
        // downstream of it still matches: a two-operand source would be caught by the arity check
        // instead, hiding a dropped filter.
        || {
            let builder = Builder::new();
            let raw = i32_input(&builder, "raw", vec![4]);
            let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
            let invalid_f32 = builder.cast(
                builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32)),
                DType::F32,
            );
            let negated = builder.unary(crate::op::UnOp::Neg, invalid_f32);
            let witness = builder.reduce(RedOp::Sum, negated, 0, true);
            let graph = crate::test_support::finish_with_validations(
                builder,
                guard.guarded,
                &[(ValidationId(1), "bounds", witness)],
            )
            .unwrap();
            recognize_index_bounds_guard(&graph, guard.guarded.id, witness.id).unwrap_err()
        },
        // The witness observes some other value.
        || {
            let builder = Builder::new();
            let raw = i32_input(&builder, "raw", vec![4]);
            let other = i32_input(&builder, "other", vec![4]);
            let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
            let decoy = builder.binary_scalar(BinOp::GeU, other, Scalar::I32(UPPER as i32));
            let decoy_f32 = builder.cast(decoy, DType::F32);
            let witness = builder.reduce(RedOp::Sum, decoy_f32, 0, true);
            let graph = crate::test_support::finish_with_validations(
                builder,
                guard.guarded,
                &[(ValidationId(1), "bounds", witness)],
            )
            .unwrap();
            recognize_index_bounds_guard(&graph, guard.guarded.id, witness.id).unwrap_err()
        },
    ];
    for case in cases {
        covered.insert(index_guard_rejection_row(&case()));
    }

    assert_eq!(
        covered,
        INDEX_GUARD_REJECTION_ROWS
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
    );
}

#[test]
fn guard_recognizer_rejects_a_witness_whose_source_is_not_a_cast() {
    // The reduce is a well-formed keepdim last-axis `Sum` and its source has the one operand the
    // recognizer expects, so only the `Cast { to: F32 }` filter stands between this and a guard that
    // would be "recognized" against a value nothing casts from the flag.
    //
    // This needs its own assertion rather than a row in `index_guard_rejection_table_covers_every_reason`:
    // dropping the filter makes the walk report `WitnessObservesAnotherValue` instead, a row another
    // case already supplies, so the table's set stays complete and the mutation survives it.
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let flag = builder.binary_scalar(BinOp::GeU, raw, Scalar::I32(UPPER as i32));
    let flag_f32 = builder.cast(flag, DType::F32);
    let negated = builder.unary(crate::op::UnOp::Neg, flag_f32);
    let witness = builder.reduce(RedOp::Sum, negated, 0, true);
    let graph = crate::test_support::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(1), "bounds", witness)],
    )
    .unwrap();

    assert_eq!(
        recognize_index_bounds_guard(&graph, guard.guarded.id, witness.id),
        Err(IndexGuardRejection::WitnessNotSumOfCast {
            witness: witness.id
        })
    );
}
