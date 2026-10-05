//! SC-009: `SampleToken{Greedy}` over a vocab above the graph
//! decomposition's threshold (`poot_graph_plan::decompose_large_vocab_greedy`) evaluates
//! bit-identical to the single-stage oracle (`eval(decompose(g)) == eval(g)`, the same
//! pass-equivalence framing ADR-0101 tier 1 states), at vocab sizes not divisible by the chunk size,
//! with ties across a chunk boundary, and with non-finite logits spread across several chunks
//! including the last (partial) one.
//!
//! Card 626: moved here from `poot-eval/src/tests/admission_corpus.rs` with the pass itself -
//! poot-eval must never depend on poot-graph-plan (its own architecture test); this crate already
//! dev-depends on poot-eval, so an integration test here can drive both. `eval_every_combination`/
//! `value_bits`/`dense_inputs`/`NoOpObserver`/`SilentAuthority` are duplicated from that file's own
//! (much more widely used there) copies - all built from poot-eval's already-public API, so there is
//! no drift risk the way duplicating a pass itself would carry.

use poot_eval::cast_authority::CastAuthority;
use poot_eval::exact_value::ExactValue;
use poot_eval::observer::EvalObserver;
use poot_eval::{EvalBudget, EvalError, EvalOptions, Value, eval};
use poot_graph_ir::Builder;
use poot_graph_ir::op::SampleRule;
use poot_graph_plan::decompose_large_vocab_greedy;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

struct NoOpObserver;
impl EvalObserver for NoOpObserver {}

struct SilentAuthority;
impl CastAuthority for SilentAuthority {
    fn role(
        &mut self,
        _cast: poot_graph_ir::ValueId,
    ) -> Option<poot_eval::cast_authority::ExactI32CastRole> {
        None
    }
}

fn eval_every_combination(
    label: &str,
    graph: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, Value>,
) -> Result<Value, EvalError> {
    let has_i32_cast = graph.eqns.iter().any(|eqn| {
        matches!(eqn.op, poot_graph_ir::OpKind::Cast { .. })
            && matches!(eqn.inputs.first(), Some(poot_graph_ir::Operand::Value(id)) if graph.aval(*id).dtype == DType::I32)
    });

    let plain = eval(graph, inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output);

    let mut observer = NoOpObserver;
    let observed = eval(
        graph,
        inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).observer(&mut observer),
    )
    .map(|r| r.output);

    let kept = eval(
        graph,
        inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .map(|r| r.output);

    let mut authority = SilentAuthority;
    let authorized = (!has_i32_cast).then(|| {
        eval(
            graph,
            inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut authority),
        )
        .map(|r| r.output)
    });

    let mut combos: Vec<(&str, &Result<Value, EvalError>)> =
        vec![("observer", &observed), ("keep_environment", &kept)];
    if let Some(authorized) = &authorized {
        combos.push(("cast_authority", authorized));
    }

    match &plain {
        Ok(value) => {
            let bits = value_bits(value);
            for (combo_name, combo) in combos {
                let combo_value = combo.as_ref().unwrap_or_else(|error| {
                    panic!("{label}: default combo succeeded but {combo_name} failed: {error}")
                });
                assert_eq!(
                    value_bits(combo_value),
                    bits,
                    "{label}: {combo_name} published different bits than the default combo"
                );
            }
        }
        Err(error) => {
            let message = error.to_string();
            for (combo_name, combo) in combos {
                let combo_error = combo.as_ref().err().unwrap_or_else(|| {
                    panic!("{label}: default combo failed but {combo_name} succeeded")
                });
                assert_eq!(
                    combo_error.to_string(),
                    message,
                    "{label}: {combo_name} reported a different error than the default combo"
                );
            }
        }
    }
    plain
}

/// `SampleToken{Greedy}`'s output is always an integer token index, published as either a dense I32
/// tensor or (card 554d's exact-I32 lane) `Value::Owner(ExactValue::I32)` - this file's narrower
/// cousin of `admission_corpus.rs`'s general `value_bits` (which also handles the F32/E4M3FN
/// carriers) never needs either of those branches.
fn value_bits(value: &Value) -> Vec<u32> {
    match value {
        Value::Host(tensor) => tensor
            .as_i32()
            .expect("SampleToken publishes dense output as I32")
            .iter()
            .map(|&i| i as u32)
            .collect(),
        Value::Owner(ExactValue::I32(view)) => view.i32_words().iter().map(|&i| i as u32).collect(),
        other => panic!("value_bits: unexpected published carrier for SampleToken {other:?}"),
    }
}

fn dense_inputs<const N: usize>(
    pairs: [(poot_graph_ir::ValueId, HostTensor); N],
) -> HashMap<poot_graph_ir::ValueId, Value> {
    pairs
        .into_iter()
        .map(|(id, t)| (id, Value::from(t)))
        .collect()
}

fn check_large_vocab_decomposition(logits: Vec<f32>) {
    let v = logits.len();
    let b = Builder::new();
    let x = b.constant("logits", poot_graph_ir::TensorType::f32(vec![v]));
    let out = b.sample_token(SampleRule::Greedy, x, None, None, None);
    let g = b.finish(out);
    let decomposed =
        decompose_large_vocab_greedy(&g, &poot_graph_plan::CompileLimits::STANDARD).unwrap();
    assert!(
        decomposed
            .eqns
            .iter()
            .filter(|e| matches!(e.op, poot_graph_ir::OpKind::SampleToken { .. }))
            .count()
            >= 2,
        "fixture must actually exercise the decomposition (vocab={v})"
    );

    let inputs = dense_inputs([(x.id, HostTensor::f32(vec![v], logits))]);
    let want = eval_every_combination("sample_token_greedy_undecomposed", &g, &inputs).unwrap();
    let got =
        eval_every_combination("sample_token_greedy_decomposed", &decomposed, &inputs).unwrap();
    assert_eq!(
        got.as_host().unwrap().as_i32().unwrap(),
        want.as_host().unwrap().as_i32().unwrap(),
        "vocab={v}: the decomposed graph must match the single-stage oracle bit for bit"
    );
}

#[test]
fn large_vocab_greedy_decomposition_matches_the_oracle_at_a_vocab_not_divisible_by_chunk_len() {
    // CHUNK_LEN=1024 (poot-graph-plan's private constant, mirrored here); 3*1024+777 leaves a
    // 777-element tail chunk (neither empty nor full).
    let v = 3 * 1024 + 777;
    let mut logits: Vec<f32> = (0..v as u32)
        .map(|i| ((i.wrapping_mul(2654435761) >> 8) % 9973) as f32)
        .collect();
    logits[v - 1] = 1.0e9; // force the true max into the partial tail chunk
    check_large_vocab_decomposition(logits);
}

#[test]
fn large_vocab_greedy_decomposition_breaks_a_cross_chunk_tie_to_the_lower_global_index() {
    let v = 2 * 1024 + 500;
    let mut logits = vec![0.0f32; v];
    // Tie the row max across chunk 0 (near its end) and chunk 1 (near its start): the lower global
    // index (chunk 0's) must win on both the decomposed graph and the single-stage oracle.
    logits[1024 - 3] = 42.0;
    logits[1024 + 5] = 42.0;
    check_large_vocab_decomposition(logits);
}

#[test]
fn large_vocab_greedy_decomposition_reports_the_lowest_non_finite_index_in_the_tail_chunk() {
    let v = 2 * 1024 + 300;
    let mut logits: Vec<f32> = (0..v).map(|i| (i % 50) as f32).collect();
    logits[v - 1] = f32::NAN; // tail of the last (partial) chunk
    logits[1024 + 7] = f32::INFINITY;
    logits[3] = f32::NEG_INFINITY;
    check_large_vocab_decomposition(logits);
}

#[test]
fn large_vocab_greedy_decomposition_matches_the_oracle_at_card_scale_b1_and_b4() {
    // SC-009's own fixture: V=131071 (not a multiple of 1024: tail = 131071 - 127*1024 = 1023), B in
    // {1, 4}.
    let v = 131_071usize;
    let mut state = 0x9E3779B9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for b_rows in [1usize, 4] {
        let mut logits = vec![0f32; b_rows * v];
        for row in 0..b_rows {
            for i in 0..v {
                logits[row * v + i] = (next() % 65536) as f32 * 0.25;
            }
        }
        // a deliberate cross-chunk tie and a tail non-finite logit in row 0.
        logits[1024 - 1] = 99999.0;
        logits[1024] = 99999.0;
        logits[v - 1] = f32::NAN;

        let b = Builder::new();
        let x = b.constant("logits", poot_graph_ir::TensorType::f32(vec![b_rows, v]));
        let out = b.sample_token(poot_graph_ir::op::SampleRule::Greedy, x, None, None, None);
        let g = b.finish(out);
        let decomposed =
            decompose_large_vocab_greedy(&g, &poot_graph_plan::CompileLimits::STANDARD).unwrap();
        assert!(
            decomposed
                .eqns
                .iter()
                .filter(|e| matches!(e.op, poot_graph_ir::OpKind::SampleToken { .. }))
                .count()
                >= 2,
            "B={b_rows}: fixture must exercise the decomposition"
        );
        let inputs = dense_inputs([(x.id, HostTensor::f32(vec![b_rows, v], logits))]);
        let want = eval_every_combination("card_scale_undecomposed", &g, &inputs).unwrap();
        let got = eval_every_combination("card_scale_decomposed", &decomposed, &inputs).unwrap();
        assert_eq!(
            got.as_host().unwrap().as_i32().unwrap(),
            want.as_host().unwrap().as_i32().unwrap(),
            "B={b_rows} V={v}"
        );
    }
}
