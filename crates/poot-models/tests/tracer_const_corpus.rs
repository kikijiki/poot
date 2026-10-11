//! Card 550a SC-002: over the traced corpus, the count of tracer-synthesized
//! `Storage::Const` inputs whose `aval` depends on the call reads 0.
//!
//! Every row traces one production family graph at two call sizes (prompt length and KV capacity)
//! and diffs the two graphs' `Storage::Const` inputs by name: a const present at only one size, or
//! with a different declared shape, is a call-dependent const the card exists to remove (the
//! baked-store growth of). Per-model tables (`rope.cos`/`rope.sin`, weights) are fixed
//! by the config and match across both calls; per-call tables must now be step inputs
//! (`Slot::Activation`), in-graph `iota` computations, or absent.
//!
//! The corpus can only trace what this crate exposes (no `#[cfg(test)]` tracers): DeepSeek-V4,
//! GLM-5.3 is test-only and counted by its in-crate tests' own assertions instead.

use std::collections::BTreeMap;

use poot_executor_parity::dense::{Dense, Family, paged_step, step};
use poot_graph_ir::{Graph, Storage};
use poot_models::model::{LogitRows, Phase};

const LAYERS: usize = 2;

/// Call A: prompt length and KV capacity.
const PROMPT_A: usize = 4;
const CAP_A: usize = 8;
/// Call B: a different prompt length and capacity (every row's multiples-of constraints hold).
const PROMPT_B: usize = 6;
const CAP_B: usize = 12;

struct Row {
    name: &'static str,
    build: fn(prompt: usize, cap: usize) -> Graph,
}

/// A dense family at the corpus dimensions, traced through `Model::trace` at `shape`.
fn dense_trace(family: Family, phase: Phase, shape: poot_models::model::StepShape) -> Graph {
    let mut dense = Dense::new(family)
        .vocab(24)
        .dims(16, 16, LAYERS)
        .heads(4, 2)
        .max_positions(32);
    // OLMo 2 normalizes the whole projection width, so its head width is `hidden / heads`.
    if family != Family::Olmo2 {
        dense = dense.head_dim(6);
    }
    dense
        .f32_model()
        .model
        .trace(phase, shape)
        .unwrap_or_else(|e| panic!("{family:?}: {e}"))
}

/// `family`'s registry fixture, traced through `Model::trace` at `shape`.
fn registered_trace(family: &str, phase: Phase, shape: poot_models::model::StepShape) -> Graph {
    let registry = poot_models::registry::Registry::builtin().expect("the builtin registry");
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("{family} is registered"));
    let fixture = (entry.fixture)();
    let model = (entry.build)(&fixture.raw(), &fixture.store).expect("the fixture builds");
    // A routed family declares router witnesses; the corpus counts consts, which they do not touch.
    let traced = model.trace(phase, shape).expect("the fixture traces");
    let mut plain = poot_graph_ir::GraphEdit::new();
    *plain.values_mut() = traced.values().clone();
    *plain.inputs_mut() = traced.inputs().to_vec();
    *plain.consts_mut() = traced.consts().to_vec();
    *plain.slots_mut() = traced.slots().to_vec();
    *plain.eqns_mut() = traced.eqns().to_vec();
    plain.set_result(traced.result());
    for (input, next) in traced.state() {
        plain.push_state(input, next);
    }
    plain.commit().expect("the traced tables stay well formed")
}

/// The named corpus: every production family tracer whose graph this crate can reach, at two call
/// sizes. Decode rows ignore `prompt`; prefill rows may use both.
fn corpus() -> Vec<Row> {
    vec![
        Row {
            name: "qwen2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.decode_batched",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    step(2, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.decode_paged",
            build: |_, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Decode,
                    paged_step(2, 1, cap, 2 * cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "qwen2.prefill_paged",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    paged_step(1, prompt, cap, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill_batched_paged",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::Prefill,
                    paged_step(2, prompt, cap, 2 * cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen2.prefill_embeds",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen2,
                    Phase::PrefillEmbeds,
                    step(1, prompt, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen3.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Qwen3,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "olmo2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Olmo2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "olmo2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Olmo2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "gemma3.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Gemma3,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "gemma3.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Gemma3,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "gemma2.prefill",
            build: |prompt, cap| {
                dense_trace(
                    Family::Gemma2,
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "gemma2.decode",
            build: |_, cap| {
                dense_trace(
                    Family::Gemma2,
                    Phase::Decode,
                    step(1, 1, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "bloom.prefill",
            build: |prompt, _| {
                registered_trace(
                    "bloom",
                    Phase::Prefill,
                    step(1, prompt, prompt, LogitRows::Last),
                )
            },
        },
        Row {
            name: "bloom.decode",
            build: |_, cap| {
                registered_trace("bloom", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "mpt.prefill",
            build: |prompt, _| {
                registered_trace(
                    "mpt",
                    Phase::Prefill,
                    step(1, prompt, prompt, LogitRows::Last),
                )
            },
        },
        Row {
            name: "mpt.decode",
            build: |_, cap| {
                registered_trace("mpt", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "smollm3.prefill",
            build: |prompt, _| {
                registered_trace(
                    "smollm3",
                    Phase::Prefill,
                    step(1, prompt, prompt, LogitRows::Last),
                )
            },
        },
        Row {
            name: "smollm3.decode",
            build: |_, cap| {
                registered_trace("smollm3", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "qwen3next.decode",
            build: |_, cap| {
                registered_trace("qwen3next", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "qwen3next.prefill",
            build: |prompt, cap| {
                registered_trace(
                    "qwen3next",
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        // The PLE sits on the fixture's second layer, so both rows count its consts.
        Row {
            name: "qwen4exp.decode",
            build: |_, cap| {
                registered_trace("qwen4exp", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "qwen4exp.prefill",
            build: |prompt, cap| {
                registered_trace(
                    "qwen4exp",
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
        Row {
            name: "qwen4exp.decode_batched",
            build: |_, cap| {
                registered_trace("qwen4exp", Phase::Decode, step(2, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "qwen4exp.prefill_paged",
            build: |prompt, cap| {
                registered_trace(
                    "qwen4exp",
                    Phase::Prefill,
                    paged_step(1, prompt, cap, cap, LogitRows::Last),
                )
            },
        },
        Row {
            name: "qwen35.decode",
            build: |_, cap| {
                registered_trace("qwen35", Phase::Decode, step(1, 1, cap, LogitRows::Last))
            },
        },
        Row {
            name: "qwen35.prefill",
            build: |prompt, cap| {
                registered_trace(
                    "qwen35",
                    Phase::Prefill,
                    step(1, prompt, cap, LogitRows::All),
                )
            },
        },
    ]
}

/// Every `Storage::Const` input of `g`, keyed by name with its declared shape.
fn const_avals(g: &Graph) -> BTreeMap<String, Vec<usize>> {
    g.inputs()
        .iter()
        .filter_map(|&id| {
            let m = g.meta(id);
            if m.storage != Storage::Const {
                return None;
            }
            let name = m.name.clone()?;
            Some((name, m.aval.shape.clone()))
        })
        .collect()
}

/// The call-dependent `Storage::Const` inputs of the (call A, call B) graph pair: a const missing at
/// one size or with a shape that changed between them.
fn call_dependent_consts(a: &Graph, b: &Graph) -> Vec<String> {
    let (ca, cb) = (const_avals(a), const_avals(b));
    let mut found = Vec::new();
    for (name, shape_a) in &ca {
        match cb.get(name) {
            None => found.push(format!(
                "{name}: present at call A only (shape {shape_a:?})"
            )),
            Some(shape_b) if shape_b != shape_a => found.push(format!(
                "{name}: shape {shape_a:?} at call A vs {shape_b:?} at call B"
            )),
            Some(_) => {}
        }
    }
    for name in cb.keys() {
        if !ca.contains_key(name) {
            found.push(format!("{name}: present at call B only"));
        }
    }
    found.sort();
    found
}

/// SC-002: the count over every family's traced graphs reads 0. Mutation: restore one
/// `b.constant("causal.mask", ..)` in any tracer above and this row reads its name once.
#[test]
fn traced_corpus_counts_zero_call_dependent_const_avals() {
    let rows = corpus();
    assert!(
        rows.len() >= 26,
        "the corpus must cover every family's tracers; got {} rows",
        rows.len()
    );
    let mut failures = Vec::new();
    for row in &rows {
        let a = (row.build)(PROMPT_A, CAP_A);
        let b = (row.build)(PROMPT_B, CAP_B);
        for found in call_dependent_consts(&a, &b) {
            failures.push(format!("{}: {found}", row.name));
        }
    }
    assert_eq!(
        failures.len(),
        0,
        "tracer-synthesized consts whose aval depends on the call (S46-3's count must read 0):\n{}",
        failures.join("\n")
    );
}

/// Card 550's dense families, by this corpus's row names: every row card 550 converted to the new
/// Pos/mask contract (`Slot::Pos` drives an in-graph `causal_mask_from_pos`/`alibi_mask_from_pos`
/// instead of a tracer-synthesized mask `Storage::Const`).
const CARD_550_DENSE_ROWS: &[&str] = &[
    "qwen2.decode",
    "qwen2.decode_batched",
    "qwen2.decode_paged",
    "qwen2.prefill",
    "qwen2.prefill_paged",
    "qwen2.prefill_batched_paged",
    "qwen2.prefill_embeds",
    "qwen3.prefill",
    "olmo2.prefill",
    "olmo2.decode",
    "gemma3.prefill",
    "gemma3.decode",
    "gemma2.prefill",
    "gemma2.decode",
    "bloom.prefill",
    "bloom.decode",
    "mpt.prefill",
    "mpt.decode",
    "smollm3.prefill",
    "smollm3.decode",
];

/// SC-005: over card 550's dense families' traced graphs, no mask is a graph input: the count of
/// tracer-synthesized mask `Storage::Const` inputs (named `mask.*`/`*.mask`/`causal.mask`, the pre-card
/// convention) reads 0. Mutation: declare a `causal.mask` const in one qwen2 prefill tracer; the count
/// reads 1 and the row goes red.
#[test]
fn card_550_dense_corpus_counts_zero_mask_step_inputs_and_consts() {
    let rows = corpus();
    let dense: Vec<&Row> = rows
        .iter()
        .filter(|r| CARD_550_DENSE_ROWS.contains(&r.name))
        .collect();
    assert_eq!(
        dense.len(),
        CARD_550_DENSE_ROWS.len(),
        "expected every name in CARD_550_DENSE_ROWS to match exactly one corpus row"
    );

    let mut mask_count = 0usize;
    let mut offenders = Vec::new();
    for row in &dense {
        let g = (row.build)(PROMPT_A, CAP_A);
        for &id in g.inputs() {
            let m = g.meta(id);
            let is_mask_const = m.storage == Storage::Const
                && m.name
                    .as_deref()
                    .is_some_and(|n| n.contains("mask") || n == "causal.mask");
            if is_mask_const {
                mask_count += 1;
                offenders.push(format!("{}: {:?} (name={:?})", row.name, m.storage, m.name));
            }
        }
    }
    assert_eq!(
        mask_count,
        0,
        "card 550's dense families must have zero mask consts:\n{}",
        offenders.join("\n")
    );
}
