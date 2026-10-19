//! The walk drops a value from its environment after the last equation that reads it, so evaluating a graph
//! that dequantizes weight after weight holds one at a time (the CPU oracle over a 27B checkpoint held every
//! dequantized weight to the end and needed more than 64 GiB of host memory).

use std::collections::{HashMap, HashSet};

use poot_graph_ir::test_support::graph_fixtures;
use poot_graph_ir::{BinOp, Builder, StateRole, TensorType, ValueId};
use poot_tensor::HostTensor;

use crate::walk::releases_after_each_equation;
use crate::{EvalOptions, Value, eval};

fn dense(values: Vec<f32>) -> Value {
    HostTensor::f32(vec![values.len()], values).into()
}

/// `a = x * x`, `b = a + x`, `c = b * a` (the result), and `next = x + c` as the carried state's next value.
fn chain() -> (
    poot_graph_ir::Graph,
    ValueId,
    [ValueId; 5],
    HashMap<ValueId, Value>,
) {
    let b = Builder::for_tests();
    let state = b
        .state_input("state", TensorType::f32([2]), StateRole::Recurrent)
        .unwrap();
    let x = b.constant("x", TensorType::f32([2])).unwrap();
    let a = b.binary(BinOp::Mul, x, x).unwrap();
    let sum = b.binary(BinOp::Add, a, x).unwrap();
    let c = b.binary(BinOp::Mul, sum, a).unwrap();
    let next = b.binary(BinOp::Add, state, c).unwrap();
    let graph =
        graph_fixtures::finish_with_state_and_validations(b, c, &[(state, next)], &[]).unwrap();
    let inputs = HashMap::from([
        (state.id(), dense(vec![1.0, 2.0])),
        (x.id(), dense(vec![2.0, 3.0])),
    ]);
    (
        graph,
        x.id(),
        [a.id(), sum.id(), c.id(), next.id(), state.id()],
        inputs,
    )
}

#[test]
fn a_value_is_released_after_its_last_reader_and_a_root_never_is() {
    let (graph, x, [a, sum, c, next, state], _) = chain();
    let releases: Vec<HashSet<ValueId>> = releases_after_each_equation(&graph)
        .into_iter()
        .map(|values| values.into_iter().collect())
        .collect();
    // Equations in order: a, sum, c, next.
    assert_eq!(releases.len(), 4);
    assert_eq!(releases[0], HashSet::new(), "x and a are read again");
    assert_eq!(
        releases[1],
        HashSet::from([x]),
        "sum is the last reader of x"
    );
    assert_eq!(
        releases[2],
        HashSet::from([a, sum]),
        "c is the last reader of a and sum"
    );
    assert_eq!(
        releases[3],
        HashSet::from([state]),
        "next reads the state input for the last time; the result c and the carried next value stay"
    );
    assert!(
        releases
            .iter()
            .all(|values| !values.contains(&c) && !values.contains(&next))
    );
}

#[test]
fn releasing_dead_values_changes_neither_the_result_nor_the_carried_state() {
    let (graph, _, _, inputs) = chain();
    let released = eval(&graph, &inputs, EvalOptions::default()).unwrap();
    let kept = eval(&graph, &inputs, EvalOptions::default().keep_environment()).unwrap();
    let host = |value: &Value| match value {
        Value::Host(tensor) => tensor.as_f32().unwrap().to_vec(),
        other => panic!("a dense value, got {other:?}"),
    };
    assert_eq!(host(&released.output), vec![24.0, 108.0]);
    assert_eq!(host(&released.output), host(&kept.output));
    assert_eq!(host(&released.state[0]), vec![25.0, 110.0]);
    assert_eq!(host(&released.state[0]), host(&kept.state[0]));
    assert!(released.environment.is_none() && kept.environment.is_some());
}

/// Keeps a `Weak` handle on every F32 value the walk produces and, at each later equation, records which
/// earlier ones are already gone from the environment.
#[derive(Default)]
struct LiveProbe {
    produced: Vec<(ValueId, std::sync::Weak<[f32]>)>,
    gone_at: Vec<(usize, Vec<ValueId>)>,
}

impl crate::observer::EvalObserver for LiveProbe {
    fn value(&mut self, eqn: usize, out: ValueId, value: &Value) {
        let gone = self
            .produced
            .iter()
            .filter(|(_, weak)| weak.strong_count() == 0)
            .map(|(id, _)| *id)
            .collect();
        self.gone_at.push((eqn, gone));
        if let Value::Host(tensor) = value
            && let poot_tensor::HostData::F32(data) = tensor.data()
        {
            self.produced.push((out, std::sync::Arc::downgrade(data)));
        }
    }
}

/// Mutation: replace the release loop in `eval` with a no-op; `a` and `sum` are still held when the last
/// equation runs and this goes red.
#[test]
fn the_walk_really_drops_dead_values_while_it_runs() {
    let (graph, _, [a, sum, _c, _next, _state], inputs) = chain();
    let mut probe = LiveProbe::default();
    eval(&graph, &inputs, EvalOptions::default().observer(&mut probe)).unwrap();
    let (_, gone) = probe.gone_at.last().expect("four equations ran");
    assert!(
        gone.contains(&a) && gone.contains(&sum),
        "a and sum are dead after the third equation; gone at the fourth: {gone:?}"
    );
}
