//! The qwen2-family table fixtures of the executor contract: a decode and a prefill graph traced through the
//! shipped registry at small dimensions, with the deterministic weights and step inputs `poot-executor-parity`
//! builds. They sit here, above the tracers, so the parity crate depends on no model.

use poot_executor_parity::{Fixture, step_fixtures, store_for, store_with};
use poot_graph_ir::Graph;
use poot_quant::weights::WeightStore;

use super::dense::{Dense, Family, step};
use crate::model::{LogitRows, Phase};

/// The qwen2 dimensions of the decode and prefill fixtures.
fn fixture_dense(family: Family) -> Dense {
    Dense::new(family)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
}

/// A qwen2 decode fixture: `cap` decode steps at increasing positions, enough to see at least one
/// recorded step followed by a replayed one with different inputs, plus a repeat of step 0's inputs
/// later to test record-vs-replay bit-exactness directly.
pub fn qwen2_decode_fixture(cap: usize) -> Fixture {
    decode_fixture(
        "qwen2_decode",
        &fixture_dense(Family::Qwen2),
        cap,
        store_for,
    )
}

/// A decode fixture of `dense` over `cap` positions: `cap` one-token steps, the store from `store`.
pub(crate) fn decode_fixture(
    name: &'static str,
    dense: &Dense,
    cap: usize,
    store: impl Fn(&Graph) -> WeightStore,
) -> Fixture {
    let m = dense.f32_model();
    let g = m
        .model
        .trace(
            Phase::Decode,
            step(1, 1, cap, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let store = store(&g);
    let steps = (0..cap)
        .map(|pos| step_fixtures(&g, &[((pos * 7 + 3) % 32) as u32], pos))
        .collect();
    Fixture {
        name,
        graph: g,
        store,
        steps,
    }
}

/// A prefill fixture of `dense` over `cap` positions: an `n`-token forward that returns the logits of
/// every position, then a second `n`-token forward at positions `n..2n`. The second step attends over
/// the cache the first wrote, so a cache entry the first step stored wrongly (or a prefill that drops
/// its cache write) moves the second step's logits; comparing one step's last position alone sees
/// neither. One graph carries both steps: the second is the same prefill program one chunk later, the
/// shape chunked prefill and prefix reuse execute.
fn prefill_fixture(
    name: &'static str,
    dense: &Dense,
    n: usize,
    cap: usize,
    store: impl Fn(&Graph) -> WeightStore,
) -> Fixture {
    assert!(
        2 * n <= cap,
        "{name}: two {n}-token chunks need {} cache positions, the fixture has {cap}",
        2 * n
    );
    let m = dense.f32_model();
    let g = m
        .model
        .trace(
            Phase::Prefill,
            step(1, n, cap, LogitRows::All),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let store = store(&g);
    let chunk = |start: usize| -> Vec<u32> {
        (start..start + n)
            .map(|t| ((t * 11 + 2) % 32) as u32)
            .collect()
    };
    let steps = vec![
        step_fixtures(&g, &chunk(0), 0),
        step_fixtures(&g, &chunk(n), n),
    ];
    Fixture {
        name,
        graph: g,
        store,
        steps,
    }
}

/// A qwen2 prefill fixture: [`prefill_fixture`] over the qwen2 body.
pub fn qwen2_prefill_fixture(n: usize, cap: usize) -> Fixture {
    prefill_fixture(
        "qwen2_prefill",
        &fixture_dense(Family::Qwen2),
        n,
        cap,
        store_for,
    )
}

/// The Gemma2/Grok attention-logit softcap the softcap fixtures trace (Card 557 SC-004).
pub const ATTN_LOGIT_SOFTCAP: f32 = 50.0;

/// The dimensions of the softcap fixtures: a Gemma 2 whose window no position reaches and whose
/// final logits are not capped, so the attention-logit softcap is the only cap in the graph.
fn softcap_dense(softcap: Option<f32>) -> Dense {
    fixture_dense(Family::Gemma2)
        .with("attn_logit_softcapping", softcap)
        .with("final_logit_softcapping", serde_json::Value::Null)
        .with("sliding_window", 1 << 20)
}

/// The softcap fixtures' weights. Softmax is shift-invariant and `50 * tanh(s / 50)` is not: large Q
/// and K projections put the scores where tanh compresses them, so a dropped softcap moves the
/// softmax weights. Small scores leave the softcap the identity; either way a dropped softcap would
/// be invisible. The path from the attention output to the logits (V and O projections, the norms,
/// the LM head) is scaled so the logits are order one and the difference reaches them above the
/// parity tolerance.
fn softcap_weights(name: &str, data: &mut [f32]) {
    let gain = if name.ends_with(".attn.q") || name.ends_with(".attn.k") {
        30.0
    } else if name.contains("norm")
        || name.ends_with(".attn.v")
        || name.ends_with(".attn.o")
        || name == "w.head"
    {
        10.0
    } else {
        1.0
    };
    data.iter_mut().for_each(|value| *value *= gain);
}

/// Card 557 SC-004: [`qwen2_decode_fixture`] over a Gemma 2 body traced with `softcap` as its
/// attention-logit softcap. A softcapped decode stays the attention decomposition (the matcher does not
/// fuse a softcapped decode yet), with the softcap in the graph.
pub fn qwen2_softcap_decode_fixture(cap: usize, softcap: Option<f32>) -> Fixture {
    decode_fixture("qwen2_softcap_decode", &softcap_dense(softcap), cap, |g| {
        store_with(g, softcap_weights)
    })
}

/// Card 557 SC-004: [`qwen2_prefill_fixture`] over a Gemma 2 body traced with `softcap`. `compile`
/// fuses a softcapped prefill into `Attention { softcap }`, which the planner lowers to the generated
/// attention kernel with its softcap score transform.
pub fn qwen2_softcap_prefill_fixture(n: usize, cap: usize, softcap: Option<f32>) -> Fixture {
    prefill_fixture(
        "qwen2_softcap_prefill",
        &softcap_dense(softcap),
        n,
        cap,
        |g| store_with(g, softcap_weights),
    )
}

/// A qwen2 decode over `n_heads` query heads sharing `n_kv_heads` key-value heads: `4 x 2` (the
/// grouped-query layout) is [`qwen2_decode_fixture`]; these add multi-head (`2 x 2`, every
/// query head its own key-value head) and multi-query (`2 x 1`, one key-value head shared by all).
pub fn head_layout_decode_fixtures(cap: usize) -> Vec<Fixture> {
    [
        ("qwen2_decode_multi_head", 2, 2),
        ("qwen2_decode_multi_query", 2, 1),
    ]
    .into_iter()
    .map(|(name, n_heads, n_kv_heads)| {
        let dense = Dense::new(Family::Qwen2)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(n_heads, n_kv_heads)
            .head_dim(4)
            .max_positions(16);
        decode_fixture(name, &dense, cap, store_for)
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::Slot;
    use poot_tensor::HostTensor;

    use super::*;

    /// Prefill parity compares every position and a step that reads the prefilled cache: the logits come
    /// back for all `n` positions of a step, and the second step starts where the first ended.
    ///
    /// Mutation: trace the prefill with `LogitRows::Last`; the result has one row and this goes red on the
    /// output shape.
    #[test]
    fn the_prefill_fixture_returns_every_position_and_reads_its_own_cache() {
        let (n, cap) = (4, 8);
        let fixture = qwen2_prefill_fixture(n, cap);
        let logits = fixture.graph.aval(fixture.graph.result());
        assert_eq!(
            logits.shape,
            [1, n, 32],
            "one logit row per prefilled position"
        );
        assert_eq!(
            fixture.steps.len(),
            2,
            "a prefill and a step over its cache"
        );
        let positions = |step: usize| -> Vec<i32> {
            let pos = fixture.steps[step]
                .iter()
                .find(|row| row.key.role() == Slot::Pos)
                .expect("a prefill step binds positions");
            pos.tensor.as_i32().unwrap().to_vec()
        };
        assert_eq!(positions(0), [0, 1, 2, 3]);
        assert_eq!(
            positions(1),
            [4, 5, 6, 7],
            "the second step starts after the cache the first wrote"
        );
    }

    /// A prefill fixture's tokens with the token at position 3 replaced: the cache entry stored at
    /// position 3 then differs from the oracle's.
    fn corrupted_first_step(fixture: &Fixture) -> Vec<poot_eval::test_support::StepFixture> {
        let tokens: Vec<u32> = (0..4u32)
            .map(|t| {
                let token = (t * 11 + 2) % 32;
                if t == 3 { (token + 5) % 32 } else { token }
            })
            .collect();
        step_fixtures(&fixture.graph, &tokens, 0)
    }

    /// The CPU oracle for `fixture`, the same one every device lane uses.
    fn cpu_oracle(
        fixture: &Fixture,
    ) -> impl FnMut(&[poot_eval::test_support::StepFixture]) -> HostTensor {
        poot_eval::test_support::weight_map_oracle::oracle_for(&fixture.graph, &fixture.store)
    }

    /// Prefill parity compares every position: a device whose logits are right at the last position and
    /// wrong at position 1 fails step 0. Runs the real `run_parity` over the echo device.
    ///
    /// Mutation: trace the prefill with `LogitRows::Last` (compare the last position only); the logits
    /// have one row and this goes red on `the prefill returns a row per position`.
    #[test]
    fn a_wrong_logit_at_a_non_final_position_fails_the_prefill() {
        let (n, cap) = (4, 8);
        let fixture = qwen2_prefill_fixture(n, cap);
        let (mut exec, target, answer) = poot_executor_parity::echo::echo_executor();
        let mut oracle = cpu_oracle(&fixture);
        let mut calls = 0;
        let mut checked = |step: &[poot_eval::test_support::StepFixture]| {
            let want = oracle(step);
            let mut device = want.as_f32().unwrap().to_vec();
            if calls == 0 {
                assert_eq!(
                    want.shape(),
                    [1, n, 32],
                    "the prefill returns a row per position"
                );
                device[32..64].iter_mut().for_each(|v| *v += 1.0);
            }
            calls += 1;
            answer.set(&HostTensor::f32(want.shape().to_vec(), device));
            want
        };
        let err = poot_executor_parity::run_parity(exec.as_mut(), target, &fixture, &mut checked)
            .expect_err("a wrong position-1 logit must fail the prefill");
        assert!(err.contains("step 0"), "{err}");
    }

    /// A cache entry stored wrongly at position 3 by the prefill is caught by the step that reads it. The
    /// device's step-0 logits are the oracle's, so only the second step can show the corruption: its
    /// answers come from an oracle that prefilled a different token at position 3.
    ///
    /// Mutation: drop the second step from `prefill_fixture`; `run_parity` passes and this goes red on
    /// `a corrupted cache entry must fail the step that reads it`.
    #[test]
    fn a_corrupted_cache_entry_fails_the_step_that_reads_it() {
        let fixture = qwen2_prefill_fixture(4, 8);
        let (mut exec, target, answer) = poot_executor_parity::echo::echo_executor();
        let mut clean = cpu_oracle(&fixture);
        let mut corrupt = cpu_oracle(&fixture);
        let corrupted = corrupted_first_step(&fixture);
        let mut calls = 0;
        let mut oracle = |step: &[poot_eval::test_support::StepFixture]| {
            let want = clean(step);
            let device = if calls == 0 {
                // The corrupt device prefills the wrong token, but its logits are reported correctly.
                corrupt(&corrupted);
                want.clone()
            } else {
                corrupt(step)
            };
            calls += 1;
            answer.set(&device);
            want
        };
        let err = poot_executor_parity::run_parity(exec.as_mut(), target, &fixture, &mut oracle)
            .expect_err("a corrupted cache entry must fail the step that reads it");
        assert!(err.contains("step 1"), "{err}");
    }
}
