//! SC-001 (card 558a; moved from poot-eval with the `fold_iota` pass, which poot-eval cannot depend on): a graph whose range is computed with `iota` produces the same oracle result
//! (ADR-0101, every element) as the pre-card convention where the same range was bound as an F32
//! constant, and the folded `Storage::Computed` constant materializes the identical range.
//!
//! Mutation that turns it red: start `iota` at 1 (`(1..=len)` instead of `(0..len)`) in the oracle
//! arm; `iota` then disagrees with the bound `[0, 1, .., len)` range.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Builder, OpKind, Storage, TensorType};
use poot_tensor::HostTensor;

use poot_graph_plan::fold_iota;

/// The bound-range graph the pre-card convention produced and the computed-iota graph must match.
fn bound_range_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let out = b.cast(b.constant("range", TensorType::f32(vec![6])), DType::F32);
    b.finish(out)
}

fn computed_iota_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let out = b.iota(6);
    b.finish(out)
}

#[test]
fn iota_matches_the_bound_range_oracle_and_folds_to_the_same_values() {
    let bound = bound_range_graph();
    let mut bound_inputs = HashMap::new();
    bound_inputs.insert(
        bound.inputs[0],
        Value::from(HostTensor::f32(vec![6], (0..6).map(|i| i as f32).collect())),
    );
    let expected = eval(
        &bound,
        &bound_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("oracle eval")
    .output
    .into_host()
    .expect("dense oracle output");

    let computed = computed_iota_graph();
    let got = eval(
        &computed,
        &HashMap::new(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("oracle eval")
    .output
    .into_host()
    .expect("dense oracle output");
    assert_eq!(
        got.as_f32().unwrap(),
        expected.as_f32().unwrap(),
        "the computed iota must equal the bound [0..6) range element for element"
    );
    assert_eq!(expected.as_f32().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let folded = fold_iota(&computed);
    assert!(
        folded
            .eqns
            .iter()
            .all(|eqn| !matches!(eqn.op, OpKind::Iota { .. })),
        "fold_iota must remove every Iota equation"
    );
    let folded_value = folded
        .inputs
        .iter()
        .copied()
        .find(|&id| matches!(folded.meta(id).storage, Storage::Computed(_)))
        .expect("the folded iota must be a computed graph constant");
    let materialized = eval(
        &folded,
        &HashMap::new(),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("oracle eval")
    .output
    .into_host()
    .expect("dense oracle output");
    assert_eq!(
        materialized.as_f32().unwrap(),
        expected.as_f32().unwrap(),
        "the folded computed constant must materialize the same [0..6) range (v{folded_value})"
    );
}
