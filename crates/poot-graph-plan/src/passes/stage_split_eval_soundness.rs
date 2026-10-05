//! Card 626: moved here from poot-eval/src/tests/stage_split.rs with the pass itself - poot-eval
//! must never depend on poot-graph-plan (its own architecture test). An in-crate test module, not an
//! integration test: the split machinery is held non-`pub` for Card 581a (dead-pub W10 form), which
//! an integration test could not reach.
//! ADR-0099 S2: split-then-evaluate equivalence. The stage graphs, fed the boundary values the
//! earlier stages hand on, reproduce the unsplit graph's outputs bit for bit on the CPU evaluator.

use std::collections::HashMap;

use super::stage_split::{StageAssignment, split_stages};
use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::Graph;
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::graph::Slot;
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::op::BinOp;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;

/// A `layers`-block graph in the shape a pipeline partition cuts: one Activation input, one shared
/// constant, and one carried state pair per block. Three equations per block, so block `l` owns
/// equations `3l..3l + 3` and a cut after `3l` is a cut between blocks. The payloads are
/// non-constant and the blocks mix `Mul` and `Add`, so a boundary bound to the wrong tensor, to a
/// zero, or to another block's state changes the result instead of cancelling.
fn layered_graph(layers: usize) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let w = b.constant("w", TensorType::f32(vec![4]));
    let mut hidden = x;
    let mut state = Vec::new();
    for layer in 0..layers {
        let scaled = b.binary(BinOp::Mul, hidden, w);
        let k_in = b.state_input(
            &format!("k{layer}"),
            TensorType::f32(vec![4]),
            StateRole::Recurrent,
        );
        let k_out = b.binary(BinOp::Add, k_in, scaled);
        let next = b.binary(BinOp::Add, scaled, k_out);
        state.push((k_in, k_out));
        hidden = next;
    }
    let graph = b.finish_with_state(hidden, &state);

    let mut inputs = HashMap::new();
    inputs.insert(
        x.id,
        Value::from(HostTensor::f32(vec![4], vec![1.5, -0.25, 3.0, 0.75])),
    );
    inputs.insert(
        w.id,
        Value::from(HostTensor::f32(vec![4], vec![0.5, 2.0, -1.25, 1.0])),
    );
    for (layer, &(state_input, _)) in graph.state.iter().enumerate() {
        let offset = 0.125 * layer as f32;
        inputs.insert(
            state_input,
            Value::from(HostTensor::f32(
                vec![4],
                vec![offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0],
            )),
        );
    }
    (graph, inputs)
}

fn bits(tensor: &HostTensor) -> Vec<u32> {
    tensor
        .as_f32()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// Two splits of the same three-block graph: a two-stage cut after block 0 and a three-stage cut
/// after every block. For each, every boundary value the stage graph hands on equals the unsplit
/// graph's value for that id, and the final stage's output equals the unsplit output.
#[test]
fn adr0099_s2_split_then_chain_matches_the_unsplit_graph() {
    let (graph, inputs) = layered_graph(3);
    assert_eq!(graph.eqns.len(), 9, "three equations per block");
    let reference = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .expect("the unsplit graph evaluates")
    .environment
    .unwrap();
    let reference_output = reference[graph.output]
        .as_ref()
        .expect("the unsplit graph computes its output")
        .as_host()
        .expect("the unsplit graph's output is dense");

    for (cuts, expected_boundaries) in [(vec![3usize], 1usize), (vec![3usize, 6], 2usize)] {
        let assignment =
            StageAssignment::from_cuts(&cuts, graph.eqns.len()).expect("the cuts are legal");
        let split = split_stages(&graph, &assignment).expect("the graph splits");
        assert_eq!(split.stages.len(), cuts.len() + 1);
        assert_eq!(
            split.boundaries.len(),
            expected_boundaries,
            "the fixture must actually cross the cut, or this row proves nothing"
        );

        let mut bound = inputs.clone();
        let last = split.stages.len() - 1;
        for (index, stage) in split.stages.iter().enumerate() {
            let env = eval(
                &stage.graph,
                &bound,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )
            .unwrap_or_else(|error| panic!("stage {index} did not evaluate: {error}"))
            .environment
            .unwrap();
            for &value in &stage.outputs {
                let handed = env[value].as_ref().unwrap_or_else(|| {
                    panic!("stage {index} did not compute the boundary value v{value}")
                });
                let expected = reference[value].as_ref().unwrap_or_else(|| {
                    panic!("the unsplit graph did not compute the boundary value v{value}")
                });
                assert_eq!(
                    bits(handed.as_host().expect("boundary value is dense")),
                    bits(expected.as_host().expect("boundary value is dense")),
                    "stage {index} handed on a different v{value}"
                );
                bound.insert(value, handed.clone());
            }
            if index == last {
                let output = env[stage.graph.output]
                    .as_ref()
                    .expect("the final stage computes the graph output")
                    .as_host()
                    .expect("the graph output is dense");
                assert_eq!(
                    bits(output),
                    bits(reference_output),
                    "the chained stages disagree with the unsplit output"
                );
            }
        }
    }
}
