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
    b.finish(out).unwrap()
}

fn computed_iota_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let out = b.iota(6);
    b.finish(out).unwrap()
}

#[test]
fn iota_matches_the_bound_range_oracle_and_folds_to_the_same_values() {
    let bound = bound_range_graph();
    let mut bound_inputs = HashMap::new();
    bound_inputs.insert(
        bound.inputs()[0],
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

    let folded = fold_iota(&computed).unwrap();
    assert!(
        folded
            .eqns()
            .iter()
            .all(|eqn| !matches!(eqn.op, OpKind::Iota { .. })),
        "fold_iota must remove every Iota equation"
    );
    let folded_value = folded
        .inputs()
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
        "the folded computed constant must materialize the same [0..6) range ({folded_value})"
    );
}

/// POOT-1040: the exact I32 iota evaluates to the integer range on the oracle, folds to a computed constant
/// that materializes the identical I32 words, and an F32 range past 2^24 (not exact) is a typed shape error
/// while the same I32 length is not. Mutation: build the I32 constant through `values_f32` (F32 tensor);
/// the dtype assertion fails.
#[test]
fn the_i32_iota_is_exact_through_the_oracle_and_the_folded_constant() {
    let b = Builder::new();
    let out = b.iota_i32(6);
    let graph = b.finish(out).unwrap();
    let run = |g: &poot_graph_ir::Graph| {
        eval(g, &HashMap::new(), EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("oracle eval")
            .output
            .into_host()
            .expect("dense oracle output")
    };
    let direct = run(&graph);
    assert_eq!(direct.dtype(), DType::I32);
    assert_eq!(direct.as_i32().unwrap(), &[0, 1, 2, 3, 4, 5]);
    let folded = run(&fold_iota(&graph).unwrap());
    assert_eq!(folded.dtype(), DType::I32);
    assert_eq!(folded.as_i32().unwrap(), direct.as_i32().unwrap());

    let beyond_f32 = OpKind::Iota {
        len: (1 << 24) + 1,
        dtype: DType::F32,
    };
    assert!(
        matches!(
            beyond_f32.infer(&[]),
            Err(poot_graph_ir::ShapeError::IotaRange { .. })
        ),
        "an F32 iota past 2^24 is not an exact range"
    );
    let exact_i32 = OpKind::Iota {
        len: (1 << 24) + 1,
        dtype: DType::I32,
    };
    assert_eq!(
        exact_i32.infer(&[]).unwrap(),
        TensorType::new(vec![(1 << 24) + 1], DType::I32)
    );
}
