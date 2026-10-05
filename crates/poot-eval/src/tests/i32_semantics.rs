//! Card 554d: one I32 pointwise semantics through the one walk (R480-014).
//!
//! SC-004 drives a literal-word table through the one `eval`, with I32 inputs bound once as a dense
//! `HostTensor::i32` (`Value::Host`) and once as an authoritative `ExactI32TensorView`
//! (`Value::Owner(ExactValue::I32)`). The expectations are hand-written literals, not one binding
//! compared with the other: a mutation of the shared `ops::elementwise` word function turns both
//! bindings red, where a carrier that still held its own arm would stay green. The second test is the
//! Reproduction: an I32 `Add` published as the graph output (card 383's deleted gather-address
//! pattern admission no longer applies to anything).

use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::{BinOp, Builder, Graph, Traced, ValueId};

use super::helpers;
use crate::{EvalBudget, EvalOptions, ExactI32TensorView, ExactValue, Value, eval};
use poot_tensor::HostTensor;

/// The graph `build(inputs)` over one `[n]` I32 constant per entry of `inputs`, evaluated through the
/// one `eval`, once with each operand bound `Value::Host` and once bound `Value::Owner(ExactValue::I32)`;
/// returns `(dense-bound words, exact-bound words)`.
fn words_through_both_bindings(
    inputs: &[&[i32]],
    build: impl Fn(&Builder, &[Traced]) -> Traced,
) -> (Vec<i32>, Vec<i32>) {
    let builder = Builder::new();
    let constants: Vec<Traced> = inputs
        .iter()
        .enumerate()
        .map(|(position, words)| {
            helpers::i32_constant(&builder, &format!("operand{position}"), vec![words.len()])
                .unwrap()
        })
        .collect();
    let out = build(&builder, &constants);
    let graph = builder.finish(out);
    let ids: Vec<ValueId> = constants.iter().map(|constant| constant.id).collect();
    (
        dense_bound_words(&graph, &ids, inputs),
        exact_bound_words(&graph, &ids, inputs),
    )
}

fn dense_bound_words(graph: &Graph, ids: &[ValueId], inputs: &[&[i32]]) -> Vec<i32> {
    let bound: HashMap<ValueId, Value> = ids
        .iter()
        .zip(inputs)
        .map(|(&id, words)| {
            (
                id,
                Value::from(HostTensor::i32(vec![words.len()], words.to_vec())),
            )
        })
        .collect();
    let out = eval(graph, &bound, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the one walk evaluates the I32 table through a dense binding")
        .output;
    out.into_host()
        .expect("an I32 result publishes as a dense value")
        .as_i32()
        .expect("an I32 result carries authoritative words")
        .to_vec()
}

fn exact_bound_words(graph: &Graph, ids: &[ValueId], inputs: &[&[i32]]) -> Vec<i32> {
    let bound: HashMap<ValueId, Value> = ids
        .iter()
        .zip(inputs)
        .map(|(&id, words)| {
            let view =
                ExactI32TensorView::try_from_words(vec![words.len()], Arc::from(*words)).unwrap();
            (id, Value::Owner(ExactValue::I32(view)))
        })
        .collect();
    let out = eval(graph, &bound, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the one walk evaluates the I32 table through an exact binding")
        .output;
    out.into_host()
        .expect("an I32 result publishes as a dense value")
        .as_i32()
        .expect("an I32 result carries authoritative words")
        .to_vec()
}

/// Assert both entries produced `expected`. The pair is compared at once, so a failure shows both
/// bindings' words: `(HostTensor::i32 binding, ExactValue::I32 binding)`.
fn assert_both(row: &str, words: (Vec<i32>, Vec<i32>), expected: &[i32]) {
    assert_eq!(
        words,
        (expected.to_vec(), expected.to_vec()),
        "{row}: (HostTensor::int binding, ExactValue::I32 binding) words"
    );
}

#[test]
fn i32_literal_word_table_is_one_semantics_through_both_entries() {
    let binary = |op: BinOp| move |b: &Builder, x: &[Traced]| b.binary(op, x[0], x[1]);

    // Modulo-2^32: i32::MAX + 1 wraps to i32::MIN; -1 + 1 is 0.
    assert_both(
        "Add wraps",
        words_through_both_bindings(&[&[i32::MAX, -1], &[1, 1]], binary(BinOp::Add)),
        &[i32::MIN, 0],
    );
    // i32::MAX * 2 = 2^32 - 2 wraps to -2; 2^16 * 2^16 = 2^32 wraps to 0 (saturating would give
    // i32::MAX for both).
    assert_both(
        "Mul wraps",
        words_through_both_bindings(&[&[i32::MAX, 65_536], &[2, 65_536]], binary(BinOp::Mul)),
        &[-2, 0],
    );
    // i32::MIN - 1 wraps to i32::MAX; 0 - i32::MIN wraps back to i32::MIN.
    assert_both(
        "Sub wraps",
        words_through_both_bindings(&[&[i32::MIN, 0], &[1, i32::MIN]], binary(BinOp::Sub)),
        &[i32::MAX, i32::MIN],
    );
    // Unsigned order: -1 is u32::MAX, so GeU(-1, 1) == 1 and GeU(1, -1) == 0.
    assert_both(
        "GeU is unsigned",
        words_through_both_bindings(&[&[-1, 1], &[1, -1]], binary(BinOp::GeU)),
        &[1, 0],
    );
    // Signed order: Ge(-1, 1) == 0 and Ge(1, -1) == 1.
    assert_both(
        "Ge is signed",
        words_through_both_bindings(&[&[-1, 1], &[1, -1]], binary(BinOp::Ge)),
        &[0, 1],
    );
    // Select by the condition word: 1 picks if_true, 0 picks if_false. The IR defines Select as
    // `if_false + cond * (if_true - if_false)` in wrapping arithmetic, so a condition word of 2 gives
    // 20 + 2 * (10 - 20) = 0, and with i32::MAX - i32::MIN the difference wraps to -1.
    assert_both(
        "Select by word",
        words_through_both_bindings(
            &[
                &[1, 0, 2, 1],
                &[10, 10, 10, i32::MAX],
                &[20, 20, 20, i32::MIN],
            ],
            |b: &Builder, x: &[Traced]| b.select(x[0], x[1], x[2]),
        ),
        &[10, 20, 0, i32::MAX],
    );
}

/// the exact lane no longer admits I32 `Add`/`Mul`/`Reshape` only as a gather address. An
/// `Add` whose result is the graph output (card 383 refused it as `NonIndexArithmetic::Observable`)
/// evaluates to the wrapped words.
#[test]
fn exact_i32_add_published_as_the_output_evaluates_to_wrapped_words() {
    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "left", vec![2]).unwrap();
    let right = helpers::i32_constant(&builder, "right", vec![2]).unwrap();
    let sum = builder.binary(BinOp::Add, left, right);
    let graph = builder.finish(sum);
    let view = |words: [i32; 2]| {
        Value::Owner(ExactValue::I32(
            ExactI32TensorView::try_from_words(vec![2], Arc::from(words)).unwrap(),
        ))
    };
    let inputs = HashMap::from([(left.id, view([i32::MAX, -5])), (right.id, view([1, 2]))]);
    let out = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("an observable I32 Add is ordinary I32 arithmetic through the one walk")
        .output;
    let host = out
        .into_host()
        .expect("an I32 result publishes as a dense value");
    let words = host
        .as_i32()
        .expect("an I32 result carries authoritative words");
    assert_eq!(words, [i32::MIN, -3]);
}
