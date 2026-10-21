//! Card 372c: the helper emits the chain its module docs describe, and refuses a selector it cannot guard.
//! What the chain proves about the guarded selector is `range::tests`' (POOT-726): no recognizer reads it.

use super::*;
use crate::builder::{Builder, Traced};
use crate::graph::{Operand, ValidationId};
use crate::op::OpKind;
use crate::types::{DType, Scalar, TensorType};

const UPPER: usize = 4;

fn i32_input(builder: &Builder, name: &str, shape: Vec<usize>) -> Traced {
    builder
        .constant(name, TensorType::new(shape, DType::I32))
        .unwrap()
}

/// A graph whose primary output is the guarded selector and whose only validation output is the witness.
fn guarded_graph() -> (crate::Graph, ValueId, ValueId, ValueId) {
    let builder = Builder::for_tests();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let graph = crate::test_support::graph_fixtures::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(7), "bounds", guard.witness)],
    )
    .unwrap();
    (graph, raw.id(), guard.guarded.id(), guard.witness.id())
}

fn producer(graph: &crate::Graph, value: ValueId) -> &crate::Eqn {
    graph.eqns().iter().find(|eqn| eqn.out == value).unwrap()
}

#[test]
fn guard_helper_emits_the_documented_chain() {
    let (graph, raw, guarded, witness) = guarded_graph();

    // The guarded value is `Select(invalid, 0, raw)`: the fallback is an inline literal.
    let select = producer(&graph, guarded);
    assert_eq!(select.op, OpKind::Select);
    let [
        Operand::Value(invalid),
        Operand::Lit(Scalar::I32(0)),
        Operand::Value(fallback),
    ] = select.inputs.as_slice()
    else {
        panic!(
            "the Select reads the flag, a zero literal and the raw value: {:?}",
            select.inputs
        )
    };
    assert_eq!(*fallback, raw);

    // One unsigned comparison against the bound decides both limits.
    let condition = producer(&graph, *invalid);
    assert_eq!(condition.op, OpKind::Binary(BinOp::GeU));
    assert!(matches!(
        condition.inputs.as_slice(),
        [Operand::Value(value), Operand::Lit(Scalar::I32(bound))]
            if *value == raw && *bound == UPPER as i32
    ));

    // The witness reduces the very condition the guard selects on: a keepdim last-axis Sum of its F32 cast.
    let reduce = producer(&graph, witness);
    assert!(matches!(
        reduce.op,
        OpKind::Reduce {
            op: RedOp::Sum,
            axis: 0,
            keepdim: true
        }
    ));
    let [Operand::Value(flag)] = reduce.inputs.as_slice() else {
        panic!("the witness reduces one value")
    };
    let cast = producer(&graph, *flag);
    assert_eq!(cast.op, OpKind::Cast { to: DType::F32 });
    assert!(matches!(cast.inputs.as_slice(), [Operand::Value(value)] if value == invalid));

    // The declared validation value is the witness, and it is a single F32 lane.
    let witnesses: Vec<_> = graph.witnesses().collect();
    assert_eq!(witnesses.len(), 1);
    assert_eq!(witnesses[0].value, witness);
    assert_eq!(graph.aval(witness).dtype, DType::F32);
    assert_eq!(graph.aval(witness).numel(), 1);
}

#[test]
fn guard_helper_rejects_a_selector_it_cannot_guard() {
    let builder = Builder::for_tests();
    let float = builder.constant("x", TensorType::f32(vec![4])).unwrap();
    assert_eq!(
        builder.guard_index_bounds(float, UPPER).unwrap_err(),
        IndexGuardError::RawNotI32 {
            raw: float.id(),
            dtype: DType::F32
        }
    );

    let scalar = i32_input(&builder, "scalar", vec![]);
    assert_eq!(
        builder.guard_index_bounds(scalar, UPPER).unwrap_err(),
        IndexGuardError::RawNotRanked { raw: scalar.id() }
    );

    let raw = i32_input(&builder, "raw", vec![4]);
    assert_eq!(
        builder.guard_index_bounds(raw, usize::MAX).unwrap_err(),
        IndexGuardError::BoundNotRepresentable {
            upper_exclusive: usize::MAX
        }
    );
}

/// The declared witness of the emitted guard is proven to observe its invalid flag, and no other value: the
/// fact comes from the `Cast` and `Reduce` rules of the range analysis, not from matching the emitted shape.
///
/// Mutation (never left in the tree): make the `Reduce` rule of `observed_flags` return nothing; the first
/// assertion goes red.
#[test]
fn the_declared_witness_is_proven_to_observe_the_invalid_flag_and_only_that_one() {
    let (graph, raw, guarded, witness) = guarded_graph();
    let facts = crate::value_ranges(&graph);
    let Operand::Value(invalid) = producer(&graph, guarded).inputs[0] else {
        panic!("the Select reads its condition as a value")
    };
    assert_eq!(facts.observes(witness), &[invalid]);
    assert!(!facts.observes(witness).contains(&raw));
    assert!(facts.observes(guarded).is_empty());
    assert!(facts.observes(raw).is_empty());
}

/// A reduction that can cancel is not an observation: a witness over `Sub` of two flags may read zero while a
/// lane is set, so the analysis derives nothing for it.
#[test]
fn a_reduction_over_arithmetic_on_flags_observes_nothing() {
    let builder = Builder::for_tests();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let other = builder.guard_index_bounds(raw, UPPER + 1).unwrap();
    let difference = builder
        .binary(BinOp::Sub, guard.witness, other.witness)
        .unwrap();
    let graph = crate::test_support::graph_fixtures::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(7), "bounds", difference)],
    )
    .unwrap();
    let facts = crate::value_ranges(&graph);
    assert!(facts.observes(difference.id()).is_empty());
}

/// The graph-level emitter writes the four equations in order, ends on the witness, and surfaces a sink
/// that refuses an equation as `Emit`, never a panic.
#[test]
fn the_emitter_orders_its_equations_and_surfaces_a_refusing_sink() {
    #[derive(Debug, PartialEq, Eq, thiserror::Error)]
    #[error("sink full")]
    struct Full;

    let raw = ValueId::new(0);
    let ty = TensorType::new(vec![4], DType::I32);
    let mut ops = Vec::new();
    let values = emit_index_guard::<Full>(raw, &ty, UPPER, |op, _| {
        ops.push(op);
        Ok(ValueId::new(ops.len() as u32))
    })
    .unwrap();
    assert_eq!(values.witness, ValueId::new(4));
    assert_eq!(values.guarded, ValueId::new(2));
    assert!(matches!(
        ops.as_slice(),
        [
            OpKind::Binary(BinOp::GeU),
            OpKind::Select,
            OpKind::Cast { to: DType::F32 },
            OpKind::Reduce {
                op: RedOp::Sum,
                keepdim: true,
                ..
            }
        ]
    ));

    let mut accepted = 0u32;
    let refused = emit_index_guard(raw, &ty, UPPER, |_, _| {
        accepted += 1;
        if accepted == 3 {
            Err(Full)
        } else {
            Ok(ValueId::new(accepted))
        }
    });
    assert_eq!(refused, Err(IndexGuardEmitError::Emit(Full)));
}
