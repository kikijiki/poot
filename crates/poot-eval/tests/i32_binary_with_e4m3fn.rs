//! Card 423: coverage for the I32-Binary dispatch arm in `eval_value_with_state_observed`
//! (`poot-eval/src/lib.rs`). Once any `E4M3FN` value is present in a graph, a plain I32-output
//! `OpKind::Binary` equation (mask/compare/select arithmetic over an exact index, as the sharded-embedding
//! composition of card 363 needs; ids above `2^24` cannot round-trip through f32) used to hit the final
//! `"e4m3fn value execution is not wired for {:?}"` catch-all. Exercises the arm on a minimal graph.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::fp8::e4m3fn_tensor;
use poot_eval::{EvalOptions, Value};
use poot_graph_ir::{
    BinOp, Eqn, Graph, GraphEdit, OpKind, Operand, Scalar, Slot, Storage, TensorType, ValueId,
    ValueMeta,
};
use poot_tensor::HostTensor;

/// A minimal graph that (a) takes the storage-aware dispatch path via one `E4M3FN` constant that no equation
/// consumes (`has_storage_value` only checks that some declared value is `E4M3FN`-typed), and (b) chains two
/// I32 `Binary` equations over an exact index: `local_row = id - 3` (`Value`+`Lit` operand, the shape of card
/// 363's `local_row`/`over` equations), then `squared = local_row * local_row` (`Value`+`Value` operand, the
/// shape of `mask = lo_ok * hi_ok`). The chain shows the arm's output re-enters `env` as `Value::Host` for a
/// second I32 equation to consume.
fn build_graph() -> (Graph, ValueId, ValueId) {
    let mut g = GraphEdit::new();

    let e4m3_id = g.values_mut().push(ValueMeta::new(
        TensorType::new(vec![1], DType::E4M3FN),
        Storage::Const,
        Some("i32_binary_test.e4m3_dummy".to_string()),
    ));
    g.inputs_mut().push(e4m3_id);
    g.consts_mut().push(e4m3_id);

    let id_val = g.values_mut().push(ValueMeta::new(
        TensorType::scalar(DType::I32),
        Storage::Slot(Slot::Token),
        None,
    ));
    g.inputs_mut().push(id_val);
    g.slots_mut().push((id_val, Slot::Token));

    let local_row = g.values_mut().push(ValueMeta::new(
        TensorType::scalar(DType::I32),
        Storage::Device,
        None,
    ));
    g.eqns_mut().push(Eqn {
        op: OpKind::Binary(BinOp::Sub),
        inputs: vec![Operand::Value(id_val), Operand::Lit(Scalar::I32(3))],
        out: local_row,
        layer: None,
    });

    let squared = g.values_mut().push(ValueMeta::new(
        TensorType::scalar(DType::I32),
        Storage::Device,
        None,
    ));
    g.eqns_mut().push(Eqn {
        op: OpKind::Binary(BinOp::Mul),
        inputs: vec![Operand::Value(local_row), Operand::Value(local_row)],
        out: squared,
        layer: None,
    });

    g.set_result(squared);
    let g = g.commit().expect("the fixture is a well-formed graph");
    (g, e4m3_id, id_val)
}

/// Required red mutation: delete the `i32_binary_equation` arm in `eval_value_with_state_observed`
/// (`poot-eval/src/lib.rs`, right after `dense_equation`). `poot_eval::eval(&g, &inputs, ...)` then returns
/// `Err(EvalError::Unsupported(message))` with message exactly
/// `"e4m3fn value execution is not wired for Binary(Sub)"` (`local_row`'s equation is the first I32 Binary
/// equation, so the catch-all names it). Asserted on the message text since `EvalError::Unsupported(String)`
/// has no further structure.
#[test]
fn i32_binary_evaluates_despite_an_unrelated_e4m3fn_value() {
    let (g, e4m3_id, id_val) = build_graph();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        e4m3_id,
        Value::Host(e4m3fn_tensor(vec![1], vec![0x38]).expect("bounded literal E4M3 fixture")),
    );
    inputs.insert(id_val, Value::Host(HostTensor::i32(vec![], vec![10])));

    let output = poot_eval::eval(&g, &inputs, EvalOptions::default())
        .map(|r| r.output)
        .expect("I32 Binary arithmetic must evaluate despite the unrelated E4M3FN value present");
    let Value::Host(tensor) = output else {
        panic!("expected a dense I32 result");
    };
    // local_row = 10 - 3 = 7; squared = 7 * 7 = 49. Printed as well as asserted.
    eprintln!(
        "i32_binary_evaluates_despite_an_unrelated_e4m3fn_value: words={:?}",
        tensor.as_i32().unwrap()
    );
    assert_eq!(tensor.as_i32(), Some([49].as_slice()));
}

/// A second, independent id value, showing the arm is not hardcoded to one input.
#[test]
fn i32_binary_result_tracks_a_different_id() {
    let (g, e4m3_id, id_val) = build_graph();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    inputs.insert(
        e4m3_id,
        Value::Host(e4m3fn_tensor(vec![1], vec![0x38]).expect("bounded literal E4M3 fixture")),
    );
    // local_row = -2 - 3 = -5; squared = (-5) * (-5) = 25.
    inputs.insert(id_val, Value::Host(HostTensor::i32(vec![], vec![-2])));

    let output = poot_eval::eval(&g, &inputs, EvalOptions::default())
        .map(|r| r.output)
        .expect("I32 Binary arithmetic must evaluate despite the unrelated E4M3FN value present");
    let Value::Host(tensor) = output else {
        panic!("expected a dense I32 result");
    };
    eprintln!(
        "i32_binary_result_tracks_a_different_id: words={:?}",
        tensor.as_i32().unwrap()
    );
    assert_eq!(tensor.as_i32(), Some([25].as_slice()));
    assert_ne!(
        tensor.as_i32(),
        Some([49].as_slice()),
        "a different id must not decode to the first test's own value - guards against a hardcoded/constant \
         result passing both tests for the wrong reason"
    );
}
