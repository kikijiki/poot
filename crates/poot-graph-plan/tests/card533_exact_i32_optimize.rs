//! Card 533 acceptance: the optimizer is generic over the validation channel and its I32 fused regions
//! evaluate on the one walk, bit for bit against the unoptimized graph.
//!
//! Card 626: moved here from `poot-eval/src/tests/exact_i32.rs`'s `card533_tests` module with the
//! passes themselves - poot-eval must never depend on poot-graph-plan (its own architecture test);
//! this crate already dev-depends on poot-eval, so an integration test here can drive both.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, ExactI32TensorView, ExactValue, Value, eval};
use poot_graph_ir::{BinOp, Builder, StateRole, TensorType, UnOp, ValidationId, ValueId};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use poot_test_util::graph_fixtures as helpers;

use poot_graph_plan::passes_without_target as optimize;

fn exact_i32(shape: Vec<usize>, words: Vec<i32>) -> Value {
    Value::Owner(ExactValue::I32(
        ExactI32TensorView::try_from_words(shape, Arc::from(words)).unwrap(),
    ))
}

fn words(value: &Value) -> Vec<i32> {
    match value {
        Value::Owner(ExactValue::I32(view)) => view.i32_words().to_vec(),
        Value::Host(tensor) => tensor
            .as_i32()
            .expect("a dense I32 publication carries I32 words")
            .to_vec(),
        other => panic!("expected an I32 value, got {other:?}"),
    }
}

fn i32_inputs(entries: &[(ValueId, Vec<usize>, Vec<i32>)]) -> HashMap<ValueId, Value> {
    entries
        .iter()
        .map(|(value, shape, words)| (*value, exact_i32(shape.clone(), words.clone())))
        .collect()
}

/// SC-001: a Qwen4Exp-shaped exact-I32 graph (an I32 hash/select recurrence over a carried state)
/// optimized equals the unoptimized graph on the one walk bit for bit over three stateful steps,
/// and the optimized graph carries an I32 fused region.
///
/// Mutation: make `fuse` form an I32 region the evaluator does not support (for example rewrite a
/// step to `FusedOp::Binary(BinOp::Le)`); the I32 fused arm refuses it with "unsupported" and this
/// test goes red.
#[test]
fn card533_optimized_exact_i32_region_matches_the_unoptimized_graph() {
    let b = Builder::new();
    let history = b.state_input(
        "history",
        TensorType::new(vec![4], DType::I32),
        StateRole::Recurrent,
    );
    let zero = helpers::i32_constant(&b, "zero", vec![4]).unwrap();
    let ones = helpers::i32_constant(&b, "ones", vec![4]).unwrap();
    let decremented = b.binary(BinOp::Sub, history, ones);
    let nonnegative = b.binary(BinOp::Ge, history, zero);
    let selected = b.select(nonnegative, decremented, history);
    let graph = b.finish_with_state(selected, &[(history, selected)]);

    let optimized = optimize(&graph);
    assert!(
        optimized.eqns.iter().any(|eqn| {
            matches!(eqn.op, poot_graph_ir::OpKind::Fused(_))
                && optimized.aval(eqn.out).dtype == DType::I32
        }),
        "the I32 hash/select chain must fuse into an I32 region, or this row proves nothing"
    );

    let mut graph_state = vec![4, 4, 4, 4];
    let mut optimized_state = vec![4, 4, 4, 4];
    for step in 0..3 {
        let result_a = eval(
            &graph,
            &i32_inputs(&[
                (history.id, vec![4], graph_state.clone()),
                (zero.id, vec![4], vec![0; 4]),
                (ones.id, vec![4], vec![1; 4]),
            ]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let result_b = eval(
            &optimized,
            &i32_inputs(&[
                (history.id, vec![4], optimized_state.clone()),
                (zero.id, vec![4], vec![0; 4]),
                (ones.id, vec![4], vec![1; 4]),
            ]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        assert_eq!(
            words(&result_a.output),
            words(&result_b.output),
            "step {step}: the optimized I32 output differs"
        );
        assert_eq!(
            words(&result_a.state[0]),
            words(&result_b.state[0]),
            "step {step}: the optimized carried state differs"
        );
        graph_state = words(&result_a.state[0]);
        optimized_state = words(&result_b.state[0]);
    }
    assert_eq!(
        graph_state,
        vec![1, 1, 1, 1],
        "the recurrence must move the state, or this row proves nothing"
    );
}

/// SC-001 (pack form): the fused arm replays a `fuse_i32_pack_concat` region, whose leaves arrive as
/// `PackLane`s of one packed input and whose output is a last-axis concatenation. The unoptimized
/// graph is evaluated on the dense walk (no standalone `Slice` admits an exact I32 carrier), the
/// optimized one through the same one walk.
///
/// Mutation: drop the `PackLane` arm in `fused_operand_words`; the optimized graph fails with
/// "exact-I32 fused operand storage" and this test goes red.
#[test]
fn card533_exact_i32_pack_region_replays_its_concat() {
    let b = Builder::new();
    let packed_source = b.constant("packed", TensorType::new(vec![2, 2], DType::I32));
    let lane0 = b.slice(packed_source, 1, 0, 1);
    let lane1 = b.slice(packed_source, 1, 1, 2);
    let not0 = b.unary(UnOp::Not, lane0);
    let not1 = b.unary(UnOp::Not, lane1);
    let concatenated = b.concat(1, &[not0, not1]);
    let graph = b.finish(concatenated);
    let optimized = optimize(&graph);

    assert_eq!(
        optimized
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Fused(_)))
            .count(),
        1,
        "the unit-lane concat must fuse into exactly one I32 pack region"
    );

    let dense_inputs: HashMap<ValueId, Value> = HashMap::from([(
        packed_source.id,
        HostTensor::i32(vec![2, 2], vec![1, 2, 3, 4]).into(),
    )]);
    let dense = eval(
        &graph,
        &dense_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let after = eval(
        &optimized,
        &i32_inputs(&[(packed_source.id, vec![2, 2], vec![1, 2, 3, 4])]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    assert_eq!(dense.as_i32(), Some(&[-2, -3, -4, -5][..]));
    assert_eq!(words(&after), vec![-2, -3, -4, -5]);
}

/// SC-002 (probe P1): a validation-bearing, stateful F32 graph is accepted by `optimize`, and the
/// optimized graph is bit-identical to the unoptimized one on the one walk over three chained
/// steps, witness included.
///
/// Mutation: drop the `ValidationChannel` generic on one pass (for example make `optimize` take
/// `&Graph`); this call stops compiling, which is the type-check row recorded in the card.
#[test]
fn card533_optimize_accepts_a_validation_bearing_stateful_graph_bit_for_bit() {
    let b = Builder::new();
    let cache = b.state_input("cache", TensorType::f32([4]), StateRole::Recurrent);
    let update = b.constant("update", TensorType::f32([1]));
    let written = b.dynamic_update_slice(cache, update, 1, 0);
    // A validation witness is an error flag: every lane must observe a clean zero, so the recipe
    // checks the written value against itself.
    let witness = b.binary(BinOp::Sub, written, written);
    let graph = helpers::finish_with_state_and_validations(
        b,
        written,
        &[(cache, written)],
        &[(ValidationId(533), "glm-shape-cut", witness)],
    )
    .unwrap();

    let optimized = optimize(&graph);
    assert_eq!(
        optimized.validation_outputs(),
        graph.validation_outputs(),
        "optimize must keep the witness declaration"
    );

    let mut graph_state = vec![0.0f32, 0.0, 0.0, 0.0];
    let mut optimized_state = vec![0.0f32, 0.0, 0.0, 0.0];
    for step in 0..3 {
        let update_value = [2.0, -0.5, 3.0][step];
        let inputs = |state: &[f32]| -> HashMap<ValueId, Value> {
            HashMap::from([
                (cache.id, HostTensor::f32(vec![4], state.to_vec()).into()),
                (
                    update.id,
                    HostTensor::f32(vec![1], vec![update_value]).into(),
                ),
            ])
        };
        let result_a = eval(
            &graph,
            &inputs(&graph_state),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let result_b = eval(
            &optimized,
            &inputs(&optimized_state),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let out_a = result_a.output.as_host().unwrap().clone();
        let out_b = result_b.output.as_host().unwrap().clone();
        assert_eq!(
            out_a.as_f32().unwrap(),
            out_b.as_f32().unwrap(),
            "step {step}: primary output differs"
        );
        let state_a = result_a.state[0].as_host().unwrap().clone();
        let state_b = result_b.state[0].as_host().unwrap().clone();
        assert_eq!(
            state_a.as_f32().unwrap(),
            state_b.as_f32().unwrap(),
            "step {step}: carried state differs"
        );

        let env_a = eval(
            &graph,
            &inputs(&graph_state),
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap()
        .environment
        .unwrap();
        let env_b = eval(
            &optimized,
            &inputs(&optimized_state),
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .unwrap()
        .environment
        .unwrap();
        assert_eq!(
            env_a[graph.output]
                .as_ref()
                .unwrap()
                .as_host()
                .unwrap()
                .as_f32()
                .unwrap(),
            env_b[optimized.output]
                .as_ref()
                .unwrap()
                .as_host()
                .unwrap()
                .as_f32()
                .unwrap(),
            "step {step}: primary output differs in the full environment"
        );
        assert_eq!(
            env_a[witness.id]
                .as_ref()
                .unwrap()
                .as_host()
                .unwrap()
                .as_f32()
                .unwrap(),
            env_b[witness.id]
                .as_ref()
                .unwrap()
                .as_host()
                .unwrap()
                .as_f32()
                .unwrap(),
            "step {step}: the witness output differs"
        );

        graph_state = state_a.as_f32().unwrap().to_vec();
        optimized_state = state_b.as_f32().unwrap().to_vec();
    }
    assert_ne!(
        graph_state,
        vec![0.0f32, 0.0, 0.0, 0.0],
        "the cache must move, or this row proves nothing"
    );
}

/// SC-003 value half: the pipeline's witness output is bit-identical before and after `optimize`,
/// while a neighbouring chain is actually fused.
///
/// Mutation: drop validation roots from `fuse`'s pinned set; the witness producer is absorbed and
/// the kept environment refuses with "UseBeforeDef" instead of comparing.
#[test]
fn card533_pipeline_witness_output_survives_every_pass() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32([4]));
    let y = b.constant("y", TensorType::f32([4]));
    // A validation witness is an error flag: every lane must observe a clean zero.
    let witness = b.binary(BinOp::Sub, x, x);
    let neighbouring = b.binary(BinOp::Mul, y, y);
    let primary = b.binary(BinOp::Add, witness, neighbouring);
    let graph =
        helpers::finish_with_validations(b, primary, &[(ValidationId(533), "neighbour", witness)])
            .unwrap();
    let optimized = optimize(&graph);
    assert!(
        optimized
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Fused(_))),
        "the neighbouring chain must fuse, or this row proves nothing"
    );

    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            x.id,
            HostTensor::f32(vec![4], vec![1.0, -2.0, 3.0, 0.5]).into(),
        ),
        (
            y.id,
            HostTensor::f32(vec![4], vec![0.5, 4.0, -1.0, 2.0]).into(),
        ),
    ]);
    let before = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap();
    let after = eval(
        &optimized,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap();
    assert_eq!(
        before[witness.id]
            .as_ref()
            .unwrap()
            .as_host()
            .unwrap()
            .as_f32()
            .unwrap(),
        after[witness.id]
            .as_ref()
            .unwrap()
            .as_host()
            .unwrap()
            .as_f32()
            .unwrap(),
        "the witness output changed across the pipeline"
    );
}
