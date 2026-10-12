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

/// A prefill fixture of `dense`: one `n`-token forward over `cap` positions.
fn prefill_fixture(
    name: &'static str,
    dense: &Dense,
    n: usize,
    cap: usize,
    store: impl Fn(&Graph) -> WeightStore,
) -> Fixture {
    let m = dense.f32_model();
    let g = m
        .model
        .trace(
            Phase::Prefill,
            step(1, n, cap, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let store = store(&g);
    let prompt: Vec<u32> = (0..n as u32).map(|t| (t * 11 + 2) % 32).collect();
    let steps = vec![step_fixtures(&g, &prompt, 0)];
    Fixture {
        name,
        graph: g,
        store,
        steps,
    }
}

/// A qwen2 prefill fixture: one `n`-token forward.
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
