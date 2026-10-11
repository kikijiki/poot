//! Card 372c: the helper emits the chain its module docs describe, and refuses a selector it cannot guard.
//! What the chain proves about the guarded selector is `range::tests`' (POOT-726): no recognizer reads it.

use super::*;
use crate::builder::{Builder, Traced};
use crate::graph::{Operand, ValidationId};
use crate::op::OpKind;
use crate::types::{DType, Scalar, TensorType};

const UPPER: usize = 4;

fn i32_input(builder: &Builder, name: &str, shape: Vec<usize>) -> Traced {
    builder.constant(name, TensorType::new(shape, DType::I32))
}

/// A graph whose primary output is the guarded selector and whose only validation output is the witness.
fn guarded_graph() -> (
    crate::Graph<crate::ValidationOutputs>,
    ValueId,
    ValueId,
    ValueId,
) {
    let builder = Builder::new();
    let raw = i32_input(&builder, "raw", vec![4]);
    let guard = builder.guard_index_bounds(raw, UPPER).unwrap();
    let graph = crate::test_support::finish_with_validations(
        builder,
        guard.guarded,
        &[(ValidationId(7), "bounds", guard.witness)],
    )
    .unwrap();
    (graph, raw.id(), guard.guarded.id(), guard.witness.id())
}

fn producer(graph: &crate::Graph<crate::ValidationOutputs>, value: ValueId) -> &crate::Eqn {
    graph.eqns.iter().find(|eqn| eqn.out == value).unwrap()
}

#[test]
fn guard_helper_emits_the_documented_chain() {
    let (graph, raw, guarded, witness) = guarded_graph();

    // The guarded value is `Select(invalid, zero, raw)`, every operand a value.
    let select = producer(&graph, guarded);
    assert_eq!(select.op, OpKind::Select);
    let [
        Operand::Value(invalid),
        Operand::Value(zero),
        Operand::Value(fallback),
    ] = select.inputs.as_slice()
    else {
        panic!("the Select reads three values: {:?}", select.inputs)
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

    // The fallback is a value, `Sub(raw, raw)`, because the Select kernel needs three value leaves.
    let fallback = producer(&graph, *zero);
    assert_eq!(fallback.op, OpKind::Binary(BinOp::Sub));
    assert!(matches!(
        fallback.inputs.as_slice(),
        [Operand::Value(left), Operand::Value(right)] if *left == raw && *right == raw
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
    assert_eq!(graph.validation_outputs().len(), 1);
    assert_eq!(graph.validation_outputs()[0].value, witness);
    assert_eq!(graph.aval(witness).dtype, DType::F32);
    assert_eq!(graph.aval(witness).numel(), 1);
}

#[test]
fn guard_helper_rejects_a_selector_it_cannot_guard() {
    let builder = Builder::new();
    let float = builder.constant("x", TensorType::f32(vec![4]));
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
