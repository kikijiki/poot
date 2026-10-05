//! The one walk's validation gate (ADR-0090/ADR-0101): every bound graph - dense, storage-aware
//! (E4M3FN), exact-I32 carried state - rejects the same validation failure, and nothing is published
//! before the gate passes, regardless of which `EvalOptions` combination the caller chose. Card 554d:
//! this file used to drive seven separate entry points through the same fixtures to prove they agreed;
//! there is one entry point now, so the axis of variation is `EvalOptions`, not the entry point.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_graph_ir::{
    BinOp, Builder, ExecutionValidationFailure, Graph, GraphValidationError, Scalar, StateRole,
    TensorType, ValidationId, ValidationOutput, ValidationOutputs,
};

use super::helpers;
use crate::fp8::encode_e4m3fn_tensor;
use crate::walk::publication_probe;
use crate::{EvalBudget, EvalError, EvalOptions, ExactI32TensorView, Value, eval};
use poot_tensor::HostTensor;

type ValueFixture = fn(f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>);

fn stateless_fixture(value: f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>) {
    let b = Builder::new();
    let primary = b.constant("primary", TensorType::f32([1]));
    let witness = b.constant("witness", TensorType::f32([2]));
    let graph = helpers::finish_with_validations(
        b,
        primary,
        &[(ValidationId(17), "synthetic-check", witness)],
    )
    .unwrap();
    let inputs = HashMap::from([
        (primary.id, dense_value(vec![42.0])),
        (witness.id, dense_value(vec![0.0, value])),
    ]);
    (graph, inputs)
}

/// A dense graph with one computed primary output and one computed state update.
fn stateful_fixture(value: f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>) {
    let b = Builder::new();
    let state = b.state_input("state", TensorType::f32([2]), StateRole::Recurrent);
    let x = b.constant("x", TensorType::f32([2]));
    let witness = b.constant("witness", TensorType::f32([2]));
    let primary = b.binary(BinOp::Mul, x, state);
    let next = b.binary(BinOp::Add, x, state);
    let graph = helpers::finish_with_state_and_validations(
        b,
        primary,
        &[(state, next)],
        &[(ValidationId(17), "synthetic-check", witness)],
    )
    .unwrap();
    let inputs = HashMap::from([
        (state.id, dense_value(vec![11.0, 13.0])),
        (x.id, dense_value(vec![2.0, 3.0])),
        (witness.id, dense_value(vec![0.0, value])),
    ]);
    (graph, inputs)
}

/// A graph that touches E4M3FN: the one walk dispatches its equations through the storage-aware path
/// instead of the plain dense one.
fn storage_aware_fixture(value: f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>) {
    let b = Builder::new();
    let raw = b.constant("raw", TensorType::new([2], DType::E4M3FN));
    let witness = b.constant("witness", TensorType::f32([2]));
    let graph =
        helpers::finish_with_validations(b, raw, &[(ValidationId(17), "synthetic-check", witness)])
            .unwrap();
    let inputs = HashMap::from([
        (raw.id, e4m3_value()),
        (witness.id, dense_value(vec![0.0, value])),
    ]);
    (graph, inputs)
}

/// Storage-aware (E4M3FN) and stateful: the state update is an ordinary dense equation.
fn storage_aware_stateful_fixture(value: f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>) {
    let b = Builder::new();
    let raw = b.constant("raw", TensorType::new([2], DType::E4M3FN));
    let state = b.state_input("state", TensorType::f32([2]), StateRole::Recurrent);
    let witness = b.constant("witness", TensorType::f32([2]));
    let next = b.binary(BinOp::Add, state, state);
    let graph = helpers::finish_with_state_and_validations(
        b,
        raw,
        &[(state, next)],
        &[(ValidationId(17), "synthetic-check", witness)],
    )
    .unwrap();
    let inputs = HashMap::from([
        (raw.id, e4m3_value()),
        (state.id, dense_value(vec![11.0, 13.0])),
        (witness.id, dense_value(vec![0.0, value])),
    ]);
    (graph, inputs)
}

/// An exact-I32 graph with carried state (card 371's carrier). The counter's update is an exact I32
/// equation, so the walk keeps the exact carrier through to publication.
fn exact_i32_stateful_fixture(value: f32) -> (Graph<ValidationOutputs>, HashMap<usize, Value>) {
    let b = Builder::new();
    let counter = b.state_input(
        "counter",
        TensorType::new([2], DType::I32),
        StateRole::Recurrent,
    );
    let x = b.constant("x", TensorType::f32([2]));
    let witness = b.constant("witness", TensorType::f32([2]));
    let primary = b.binary(BinOp::Add, x, x);
    let next = b.binary_scalar(BinOp::Sub, counter, Scalar::I32(-1));
    let graph = helpers::finish_with_state_and_validations(
        b,
        primary,
        &[(counter, next)],
        &[(ValidationId(17), "synthetic-check", witness)],
    )
    .unwrap();
    let inputs = HashMap::from([
        (
            counter.id,
            Value::from(
                ExactI32TensorView::try_from_words(vec![2], std::sync::Arc::from(vec![5, 7]))
                    .unwrap(),
            ),
        ),
        (x.id, dense_value(vec![2.0, 3.0])),
        (witness.id, dense_value(vec![0.0, value])),
    ]);
    (graph, inputs)
}

fn e4m3_value() -> Value {
    Value::Host(encode_e4m3fn_tensor(vec![2], &[1.0, 2.0]).unwrap())
}

fn dense_value(data: Vec<f32>) -> Value {
    Value::Host(HostTensor::f32(vec![data.len()], data))
}

fn assert_same_failure(error: EvalError, expected_bits: u32) {
    match error {
        EvalError::Validation(ExecutionValidationFailure {
            id,
            name,
            lane,
            observed_bits,
        }) => {
            assert_eq!(id, ValidationId(17));
            assert_eq!(name, "synthetic-check");
            assert_eq!(lane, 1);
            assert_eq!(observed_bits, expected_bits);
        }
        other => panic!("expected validation failure, got {other:?}"),
    }
}

/// Every `EvalOptions` combination the walk offers (unbounded budget; plus `keep_environment`; plus a
/// scratch `cast_authority` that answers nothing, for a graph with no I32 `Cast`) rejects the same
/// validation failure, on a dense, a stateful, and a storage-aware (E4M3FN) fixture.
#[test]
fn eval_rejects_the_same_lane_regardless_of_options() {
    let dense_fixtures: [(&str, ValueFixture); 2] = [
        ("stateless", stateless_fixture),
        ("stateful", stateful_fixture),
    ];
    for (name, fixture) in dense_fixtures {
        let (graph, values) = fixture(f32::NAN);
        assert_same_failure(
            eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err(),
            f32::NAN.to_bits(),
        );
        assert_same_failure(
            eval(
                &graph,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )
            .unwrap_err(),
            f32::NAN.to_bits(),
        );
        let mut authority =
            |_: poot_graph_ir::ValueId| -> Option<crate::cast_authority::ExactI32CastRole> { None };
        assert_same_failure(
            eval(
                &graph,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut authority),
            )
            .unwrap_err(),
            f32::NAN.to_bits(),
        );
        let _ = name;
    }

    let storage_fixtures: [(&str, ValueFixture); 3] = [
        ("storage-aware stateless", storage_aware_fixture),
        ("storage-aware stateful", storage_aware_stateful_fixture),
        ("exact-I32 stateful", exact_i32_stateful_fixture),
    ];
    for (name, fixture) in storage_fixtures {
        let (graph, values) = fixture(f32::NAN);
        assert_same_failure(
            eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap_err(),
            f32::NAN.to_bits(),
        );
        let _ = name;
    }
}

#[test]
fn oversized_packet_is_rejected_before_evaluation() {
    let b = Builder::new();
    let unbound = b.constant("unbound", TensorType::f32([1]));
    let primary = b.binary(BinOp::Mul, unbound, unbound);
    let witness = b.constant("oversized", TensorType::f32([1025]));
    // `with_validations` does not validate, so the oversized declaration reaches the evaluator.
    let graph = b.finish(primary).with_validations(vec![ValidationOutput {
        id: ValidationId(17),
        name: "oversized".into(),
        value: witness.id,
    }]);
    // No input is bound: evaluating the equation would report `MissingInput` instead.
    let inputs = HashMap::new();

    for (name, opts) in [
        ("default", EvalOptions::new(EvalBudget::UNBOUNDED)),
        (
            "keep_environment",
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        ),
    ] {
        let result = eval(&graph, &inputs, opts).map(drop);
        assert!(
            matches!(
                result,
                Err(EvalError::InvalidGraph(
                    GraphValidationError::ValidationPacketTooLarge {
                        byte_len: 4100,
                        max_bytes: 4096,
                    }
                ))
            ),
            "{name}: {result:?}"
        );
    }
}

#[test]
fn empty_validation_graph_is_compatible() {
    fn build() -> (Graph<ValidationOutputs>, Graph, HashMap<usize, Value>) {
        let b = Builder::new();
        let state = b.state_input("state", TensorType::f32([1]), StateRole::Recurrent);
        let x = b.constant("x", TensorType::f32([1]));
        let primary = b.binary(BinOp::Mul, x, state);
        let next = b.binary(BinOp::Add, x, state);
        let inputs = HashMap::from([
            (state.id, dense_value(vec![3.0])),
            (x.id, dense_value(vec![5.0])),
        ]);
        let plain = b.finish_with_state(primary, &[(state, next)]);
        (plain.clone().with_validations(Vec::new()), plain, inputs)
    }

    let (empty, plain, inputs) = build();
    let plain_result = eval(&plain, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let empty_result = eval(&empty, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let plain_output = plain_result.output.into_host().unwrap();
    assert_eq!(plain_output.as_f32().unwrap(), &[15.0]);
    assert_eq!(
        plain_result.state[0].as_host().unwrap().as_f32().unwrap(),
        &[8.0]
    );
    assert_eq!(empty_result.output, Value::from(plain_output.clone()));
    assert_eq!(empty_result.state, plain_result.state);
}

/// Validation must be checked before any value is copied into a caller-visible result. The probe
/// counts the publication helpers that perform that copy: zero on failure, and on success equal to the
/// number of caller-visible values (primary output plus every state output, plus one more if
/// `keep_environment` was set), so the walk publishes through the probed helpers on every success and
/// nothing is counted while validation is still pending.
#[test]
fn cpu_publication_probe_stays_untouched_on_failure() {
    type OptsCombo = (&'static str, fn() -> EvalOptions<'static>, usize);
    let combos: [OptsCombo; 2] = [
        ("default", || EvalOptions::new(EvalBudget::UNBOUNDED), 0),
        (
            "keep_environment",
            || EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            1,
        ),
    ];

    let dense_cases: [(&str, ValueFixture, usize); 2] = [
        ("stateless", stateless_fixture, 1),
        ("stateful", stateful_fixture, 2),
    ];
    for (combo_name, make_opts, extra) in combos {
        for (shape, fixture, state_values) in dense_cases {
            for (witness, passes) in [(1.0, false), (0.0, true)] {
                let (graph, values) = fixture(witness);
                assert_probe(
                    &format!("{shape} {combo_name}"),
                    witness,
                    passes,
                    state_values + extra,
                    || eval(&graph, &values, make_opts()).map(drop),
                );
            }
        }
    }

    let storage_cases: [(&str, ValueFixture, usize); 3] = [
        ("storage-aware stateless", storage_aware_fixture, 1),
        ("storage-aware stateful", storage_aware_stateful_fixture, 2),
        ("exact-I32 stateful", exact_i32_stateful_fixture, 2),
    ];
    for (shape, fixture, expected) in storage_cases {
        for (witness, passes) in [(1.0, false), (0.0, true)] {
            let (graph, values) = fixture(witness);
            assert_probe(shape, witness, passes, expected, || {
                eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).map(drop)
            });
        }
    }
}

fn assert_probe(
    case: &str,
    witness: f32,
    passes: bool,
    expected_on_success: usize,
    run: impl FnOnce() -> Result<(), EvalError>,
) {
    publication_probe::take();
    let result = run();
    let publications = publication_probe::take();
    if passes {
        assert!(result.is_ok(), "{case}: {result:?}");
        assert_eq!(
            publications, expected_on_success,
            "{case} did not publish every caller-visible value exactly once through the probed helper"
        );
    } else {
        assert_same_failure(result.unwrap_err(), witness.to_bits());
        assert_eq!(publications, 0, "{case} published before validation failed");
    }
}

/// SC-009: `NanTap` names the first equation whose output is NaN (a `0 * inf` MatMul fixture) on a
/// validation graph whose packet then fails, so the tap fires though nothing is published.
///
/// Mutation: move `observer.value(...)` to run only after `validate_environment` succeeds in
/// `walk.rs::eval`; the tap then sees nothing on this failing graph and `tap.first()` goes back to
/// `None`.
#[test]
fn nan_tap_fires_on_a_failing_packet_even_though_nothing_publishes() {
    let b = Builder::new();
    let zero = b.constant("zero", TensorType::f32([1, 1]));
    let inf = b.constant("inf", TensorType::f32([1, 1]));
    let product = b.matmul(zero, inf);
    let witness = b.constant("witness", TensorType::f32([2]));
    let graph = helpers::finish_with_validations(
        b,
        product,
        &[(ValidationId(533), "nan-witness-fails", witness)],
    )
    .unwrap();
    let inputs = HashMap::from([
        (zero.id, Value::Host(HostTensor::f32(vec![1, 1], vec![0.0]))),
        (
            inf.id,
            Value::Host(HostTensor::f32(vec![1, 1], vec![f32::INFINITY])),
        ),
        (witness.id, dense_value(vec![0.0, 1.0])),
    ]);

    let mut tap = crate::observer::NanTap::default();
    let err = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).observer(&mut tap),
    )
    .expect_err("the non-zero witness lane must fail validation");
    assert!(
        matches!(err, EvalError::Validation(_)),
        "expected a validation failure, got {err:?}"
    );

    let hit = tap
        .first()
        .expect("the tap must see the NaN MatMul output even though nothing published");
    assert_eq!(hit.eqn, 0, "the MatMul is this graph's only equation");
    assert_eq!(hit.out, product.id);
    assert_eq!(hit.dtype, DType::F32);
}
