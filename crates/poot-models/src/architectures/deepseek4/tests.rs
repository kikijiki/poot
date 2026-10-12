//! DeepSeek-V4 rows: the fixtures against their independent reference, the step shapes, a missing
//! tensor, the hash table's range check, the carried state and the packed projections.

mod packed;
mod reference;

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::Slot;
use poot_graph_plan::{
    CompileLimits, CompileOptions, Target, compile, production_plan::Submission,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_tensor::DType;

use super::*;
use crate::components::standard::oracle::{eval, run_step};
use crate::components::testing::{assert_missing_weight_is_named, build, close, close_all};
use crate::model::{ConfigReason, KvLayout, LogitRows, Phase};
use crate::registry::Registry;
use packed::{E4M3, FP4, literal, payload};
use poot_quant::weights::{AttnRole, HandleFormat, WeightId, WeightMap, WeightRole};
use reference::Reference;

/// The cache capacity of the rows here: 8 windows of the CSA rate 2, 4 of the HCA rate 4.
const CAPACITY: usize = 16;

/// Ten tokens: two prefill chunks of the fixture's granule (4) and two decode steps.
const PROMPT: [i32; 10] = [3, 17, 40, 8, 25, 9, 31, 2, 44, 12];

fn model_of(fx: &Fixture) -> Box<dyn Model> {
    build(&ENTRY, &fx.raw(), &fx.store)
}

fn finetune() -> Fixture {
    finetune_fixture()
}

fn shape(tokens: usize, capacity: usize, logits: LogitRows) -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits,
        hidden: None,
    }
}

/// The names of the state a one-token decode step carries, in carry order.
fn state_names(model: &dyn Model, capacity: usize) -> Vec<String> {
    let g = model
        .trace(
            Phase::Decode,
            shape(1, capacity, LogitRows::All),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    g.state()
        .map(|(input, _)| g.meta(input).name.clone().unwrap())
        .collect()
}

/// `tokens` through the model as the prefill `chunks` and then one decode step per remaining token
/// over a cache of `capacity` positions: every step's logits, and the carried state at the end.
fn run_chunked(
    model: &dyn Model,
    store: &WeightStore,
    capacity: usize,
    chunks: &[usize],
    tokens: &[i32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (mut logits, mut state, mut at) = (Vec::new(), Vec::new(), 0usize);
    for &chunk in chunks {
        let (out, next) = run_step(
            model,
            store,
            Phase::Prefill,
            capacity,
            &tokens[at..at + chunk],
            at as i32,
            &state,
        );
        logits.extend(out);
        (state, at) = (next, at + chunk);
    }
    while at < tokens.len() {
        let (out, next) = run_step(
            model,
            store,
            Phase::Decode,
            capacity,
            &tokens[at..=at],
            at as i32,
            &state,
        );
        logits.extend(out);
        (state, at) = (next, at + 1);
    }
    (logits, state)
}

/// The reference's run over `tokens` against the model's steps (prefill `chunks`, then decode): every
/// logit of every position and every element of every carried state, tier 2.
fn assert_matches_reference(fx: &Fixture, capacity: usize, chunks: &[usize], tokens: &[i32]) {
    let model = model_of(fx);
    let want = Reference::new(&fx.config, &fx.store).run(
        &tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
        capacity,
    );
    let (logits, state) = run_chunked(&*model, &fx.store, capacity, chunks, tokens);
    let vocab = model.config().vocab;
    for (position, (got, want)) in logits.chunks(vocab).zip(&want.logits).enumerate() {
        let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        eprintln!("position {position}");
        close(got, &want);
    }
    assert_eq!(
        state_names(&*model, capacity),
        want.state
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>(),
        "the carried state, in carry order"
    );
    let want_state: Vec<Vec<f32>> = want
        .state
        .iter()
        .map(|(_, v)| v.iter().map(|&x| x as f32).collect())
        .collect();
    close_all(&state, &want_state);
}

/// SC-001: the fixture (sliding, CSA, HCA and CSA layers, a hash-routed first layer and score-routed
/// others over experts of different weights) equals its independent reference at every position
/// and every carried state element: token by token, as two granule-sized prefill chunks and decode
/// steps, and as one prefill. Mutation: swap two experts' routing indices in the tracer
/// (`hash_routes`' ids reversed); the first logit of the first routed token moves and the row goes
/// red.
#[test]
fn the_fixture_matches_its_independent_reference_at_every_position_and_state_element() {
    let fx = fixture();
    for (chunks, tokens) in [
        (&[][..], &PROMPT[..]),
        (&[4, 4][..], &PROMPT[..]),
        (&[8][..], &PROMPT[..8]),
    ] {
        assert_matches_reference(&fx, CAPACITY, chunks, tokens);
    }
}

/// SC-002 (F5): a fine-tune of the architecture the old pinned counts refused (three layers in
/// another order, other compression rates, two hash-routed layers, fewer experts and heads, a
/// narrower window) and no manifest builds as the same family and matches its own reference, with
/// its own prefill granule. Mutation: reinstate a pinned layer count in `from_raw`; the build is
/// refused and the row goes red.
#[test]
fn a_finetune_like_fixture_builds_and_matches_its_reference() {
    let fx = finetune();
    let model = model_of(&fx);
    assert_eq!(model.config().family, FAMILY);
    assert_eq!(
        model.config().prefill_granule.get(),
        4,
        "lcm of rates 4 and 2"
    );
    assert_matches_reference(&fx, CAPACITY, &[4], &PROMPT[..8]);
}

/// SC-003 (F6): a store missing one required tensor, of any kind, fails `Registry::build` with the
/// typed missing-weight error naming it, before any trace. Mutation: skip the row check in `Rows`;
/// the missing tensor is found only at bind time and the row goes red.
#[test]
fn a_missing_required_tensor_is_named_before_any_trace() {
    let registry = Registry::builtin().unwrap();
    let fx = fixture();
    for removed in [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "layers.0.ffn.gate.tid2eid",
        "layers.1.ffn.gate.bias",
        "layers.1.attn_norm.weight",
        "layers.1.ffn_norm.weight",
        "layers.1.attn.wq_a.weight",
        "layers.1.attn.q_norm.weight",
        "layers.1.attn.wq_b.weight",
        "layers.1.attn.wkv.weight",
        "layers.1.attn.kv_norm.weight",
        "layers.1.attn.attn_sink",
        "layers.1.attn.wo_a.weight",
        "layers.1.attn.wo_b.weight",
        "layers.1.hc_attn_fn",
        "layers.1.hc_ffn_scale",
        "layers.1.ffn.gate.weight",
        "layers.1.ffn.shared_experts.w2.weight",
        "layers.1.ffn.experts.5.w3.weight",
        "layers.2.attn.compressor.ape",
        "layers.2.attn.compressor.norm.weight",
        "layers.1.attn.indexer.wq_b.weight",
        "layers.1.attn.indexer.weights_proj.weight",
        "layers.1.attn.indexer.compressor.wgate.weight",
        "layers.3.attn.indexer.compressor.ape",
    ] {
        assert_missing_weight_is_named(&registry, &fx, removed);
    }
}

/// `fx`'s store with `key` replaced by `entry`.
fn with_entry(fx: &Fixture, key: &str, entry: WeightEntry) -> Fixture {
    let mut store = WeightStore::builder();
    for (k, e) in fx.store.iter() {
        let e = if k.as_str() == key {
            entry.clone()
        } else {
            e.clone()
        };
        store.insert(k.clone(), e).unwrap();
    }
    Fixture {
        store: store.build(),
        ..fx.clone()
    }
}

/// An I64 `[vocab, top_k]` table entry of `words`.
fn table_entry(shape: &[usize], words: &[i64]) -> WeightEntry {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    WeightEntry::Dense(DenseWeight::try_new(DType::I64, shape.to_vec(), bytes.into()).unwrap())
}

/// The fixture's hash table words (layer 0), row-major `[vocab, top_k]`.
fn table_words(fx: &Fixture) -> Vec<i64> {
    let WeightEntry::Dense(dense) = fx.store.get("layers.0.ffn.gate.tid2eid").unwrap() else {
        panic!("the hash table is dense")
    };
    dense
        .bytes()
        .as_slice()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

/// SC-009 (F14): a table word above `i32::MAX` (or below `i32::MIN`) fails `build` with the typed
/// value error naming the table and the element; in-range tables route the fixture exactly as the
/// reference does, and the routing is live (a permuted table changes the logits). Mutation: narrow
/// with `as i32` instead of the range-checked view; the out-of-range word silently wraps, the build
/// succeeds and the row goes red.
#[test]
fn a_hash_table_word_outside_i32_is_a_typed_error_and_in_range_tables_route_as_the_reference() {
    let fx = fixture();
    let (vocab, k) = (48, 2);
    let words = table_words(&fx);
    for (index, value) in [(5, i64::from(i32::MAX) + 1), (9, i64::from(i32::MIN) - 1)] {
        let mut bad = words.clone();
        bad[index] = value;
        let fx = with_entry(
            &fx,
            "layers.0.ffn.gate.tid2eid",
            table_entry(&[vocab, k], &bad),
        );
        match (ENTRY.build)(&fx.raw(), &fx.store) {
            Err(ModelError::WeightValue {
                id,
                index: at,
                value: got,
                ..
            }) => {
                assert_eq!(id.const_name(), "w.l0.moe.hash_table");
                assert_eq!((at, got), (index, value));
            }
            other => panic!("expected a value error for word {index}, got {other:?}"),
        }
    }

    // The same words, row 0 and row 1 swapped for the prompt's first tokens: still in range, and
    // matches a reference that reads the swapped table.
    let mut permuted = words.clone();
    for slot in 0..k {
        permuted.swap(3 * k + slot, 17 * k + slot);
    }
    let swapped = with_entry(
        &fx,
        "layers.0.ffn.gate.tid2eid",
        table_entry(&[vocab, k], &permuted),
    );
    assert_matches_reference(&swapped, CAPACITY, &[], &PROMPT[..4]);
    let model = model_of(&fx);
    let (base, _) = run_chunked(&*model, &fx.store, CAPACITY, &[], &PROMPT[..4]);
    let model = model_of(&swapped);
    let (moved, _) = run_chunked(&*model, &swapped.store, CAPACITY, &[], &PROMPT[..4]);
    assert_ne!(base, moved, "the hash table decides the experts of layer 0");
}

/// POOT-1114: a hash layer's table is traced as a ranged const: every word is declared to be an
/// expert id, `[0, experts)` (the fixture routes 8), which the weight binder checks at bind. The
/// score-routed layers have no table. Mutation: trace the table with `Builder::constant`; its
/// range is every `i32` word and this row goes red.
#[test]
fn the_hash_table_is_declared_in_the_expert_range() {
    let fx = fixture();
    let model = model_of(&fx);
    let g = model
        .trace(
            Phase::Decode,
            shape(1, CAPACITY, LogitRows::All),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let tables: Vec<_> = g
        .inputs()
        .iter()
        .map(|&id| g.meta(id))
        .filter(|meta| {
            meta.name
                .as_deref()
                .is_some_and(|name| name.ends_with(".moe.hash_table"))
        })
        .collect();
    assert!(!tables.is_empty(), "the fixture has a hash-routed layer");
    for meta in tables {
        assert_eq!(
            meta.range(),
            poot_graph_ir::IndexRange::below(8),
            "{:?}",
            meta.name
        );
    }
}

/// POOT-1114: the expert count that bounds the hash table's words is config data, so a count that
/// cannot bound I32 words (zero, or more than `2^31` ids) is a typed refusal naming the config field, not
/// a panic. The positive control is the fixture's own count. Mutation: replace the `ok_or(..)?` in
/// `HashTable::new` with an `expect`; the zero row panics and this row goes red.
#[test]
fn an_expert_count_that_cannot_bound_the_hash_table_is_a_typed_config_refusal() {
    use super::moe::HashTable;
    use crate::components::linear::WeightError;

    let fx = fixture();
    let cfg = super::config::DeepseekV4Config::from_raw(&fx.raw()).unwrap();
    let map = super::names::weights_of(&cfg, &fx.store).unwrap();
    let (vocab, k) = (48, 2);
    HashTable::new(&map, 0, vocab, k, 8).expect("the fixture's 8 experts bound the table");
    for (bound, reason) in [
        (0, ConfigReason::Zero),
        (
            i32::MAX as usize + 2,
            ConfigReason::Exceeds {
                max: i32::MAX as usize + 1,
            },
        ),
    ] {
        let error = HashTable::new(&map, 0, vocab, k, bound).unwrap_err();
        assert!(
            matches!(error, WeightError::IndexBound { bound: got, .. } if got == bound),
            "{error:?}"
        );
        match error.for_family(FAMILY) {
            ModelError::Config {
                field, reason: got, ..
            } => assert_eq!((field, got), ("n_routed_experts", reason)),
            other => panic!("expected a config refusal, got {other:?}"),
        }
    }
}

/// The prefill granule: a prefill step must cover whole windows (a multiple of 4 here), a decode
/// step is one token, and a capacity above the model's maximum is refused, each with its typed
/// reason and before any graph is built. Mutation: drop the granule check in `trace`; the 6-token
/// prefill traces and the row goes red.
#[test]
fn step_shapes_the_family_cannot_trace_are_typed_refusals() {
    let model = model_of(&fixture());
    let reason = |phase, tokens, capacity| match model.trace(
        phase,
        shape(tokens, capacity, LogitRows::Last),
        poot_graph_ir::TraceLimits::for_tests(),
    ) {
        Err(TraceError::ShapeUnsupported { reason, .. }) => reason,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    assert_eq!(
        reason(Phase::Prefill, 6, CAPACITY),
        ShapeReason::Granule { granule: 4 }
    );
    assert_eq!(
        reason(Phase::Prefill, 1, CAPACITY),
        ShapeReason::Granule { granule: 4 }
    );
    assert_eq!(
        reason(Phase::Decode, 2, CAPACITY),
        ShapeReason::DecodeTokens
    );
    assert_eq!(
        reason(Phase::Decode, 1, 65),
        ShapeReason::CapacityAboveMax { max: 64 }
    );
    for tokens in [4, 8] {
        model
            .trace(
                Phase::Prefill,
                shape(tokens, CAPACITY, LogitRows::Last),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
    }
}

/// A cache shorter than a compression window holds no compressed entries: the layers attend their
/// window alone, as the reference does (it has no window complete yet either).
#[test]
fn a_cache_shorter_than_a_window_has_no_compressed_entries() {
    let fx = fixture();
    for capacity in [1, 3] {
        assert_matches_reference(&fx, capacity, &[], &PROMPT[..capacity]);
    }
}

/// The `rows` of a step do not mix: two rows with different prompts give, row by row, the logits
/// and carried state of the same prompts run alone (prefill of one granule, then a decode step).
#[test]
fn two_rows_do_not_share_state() {
    let fx = fixture();
    let model = model_of(&fx);
    let prompts: [[i32; 5]; 2] = [[3, 17, 40, 8, 25], [9, 31, 2, 44, 12]];
    let single: Vec<_> = prompts
        .iter()
        .map(|p| run_chunked(&*model, &fx.store, CAPACITY, &[4], p))
        .collect();

    let two = |tokens: usize| StepShape {
        rows: NonZeroUsize::new(2).unwrap(),
        ..shape(tokens, CAPACITY, LogitRows::All)
    };
    let run = |phase, tokens: usize, ids: Vec<i32>, pos: Vec<i32>, state: Vec<Vec<f32>>| {
        let g = model
            .trace(phase, two(tokens), poot_graph_ir::TraceLimits::for_tests())
            .unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state = if state.is_empty() { zeros } else { state };
        let views: Vec<&[f32]> = state.iter().map(Vec::as_slice).collect();
        eval(
            &g,
            &fx.store,
            model.weights(),
            &[],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &views,
        )
    };
    let prefix = |row: usize| prompts[row][..4].to_vec();
    let (logits, state) = run(
        Phase::Prefill,
        4,
        [prefix(0), prefix(1)].concat(),
        (0..4).chain(0..4).collect(),
        Vec::new(),
    );
    let (logits_after, state) = run(
        Phase::Decode,
        1,
        vec![prompts[0][4], prompts[1][4]],
        vec![4, 4],
        state,
    );
    let vocab = model.config().vocab;
    for row in 0..2 {
        let (want, want_state) = &single[row];
        let mut got: Vec<f32> = logits[row * 4 * vocab..(row + 1) * 4 * vocab].to_vec();
        got.extend_from_slice(&logits_after[row * vocab..(row + 1) * vocab]);
        close(&got, want);
        for (carried, want) in state.iter().zip(want_state) {
            let per_row = carried.len() / 2;
            close(&carried[row * per_row..(row + 1) * per_row], want);
        }
    }
}

fn config_error_of(edit: impl FnOnce(&mut serde_json::Value)) -> (&'static str, ConfigReason) {
    let mut fx = fixture();
    edit(&mut fx.config);
    match (ENTRY.build)(&fx.raw(), &fx.store) {
        Err(ModelError::Config { field, reason, .. }) => (field, reason),
        other => panic!("{}: expected a config error, got {other:?}", fx.config),
    }
}

/// Every config fact is checked once, with its field named: nothing in the schedule or the
/// dimensions is assumed. Mutation: accept any compression ratio as HCA; the unknown-ratio row
/// goes red.
#[test]
fn a_config_the_family_cannot_trace_is_a_typed_error_naming_the_field() {
    use ConfigReason::*;
    use serde_json::json;
    assert_eq!(
        config_error_of(|c| c["compress_ratios"] = json!([0, 3, 4, 2])),
        ("compress_ratios", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["compress_ratios"] = json!([0, 2, 4])),
        ("compress_ratios", Length { expected: 4 })
    );
    assert_eq!(
        config_error_of(|c| c
            .as_object_mut()
            .unwrap()
            .remove("compress_ratios")
            .map(drop)
            .unwrap()),
        ("compress_ratios", Missing)
    );
    assert_eq!(
        config_error_of(|c| c["compress_rates"] = json!({
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        })),
        ("compress_rates", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["o_groups"] = json!(3)),
        ("o_groups", NotDivisible { by: 3 })
    );
    assert_eq!(
        config_error_of(|c| c["qk_rope_head_dim"] = json!(32)),
        ("qk_rope_head_dim", Exceeds { max: 16 })
    );
    assert_eq!(
        config_error_of(|c| c["index_head_dim"] = json!(4)),
        ("index_head_dim", Exceeds { max: 8 })
    );
    assert_eq!(
        config_error_of(|c| c["num_hash_layers"] = json!(5)),
        ("num_hash_layers", Exceeds { max: 4 })
    );
    assert_eq!(
        config_error_of(|c| c["hc_mult"] = json!(0)),
        ("hc_mult", Zero)
    );
    assert_eq!(
        config_error_of(|c| c["scoring_func"] = json!("softmax")),
        ("scoring_func", Unsupported)
    );
    assert_eq!(
        config_error_of(
            |c| c["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 16})
        ),
        ("rope_scaling", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["num_experts_per_tok"] = json!(9)),
        ("top_k", Exceeds { max: 8 })
    );
}

/// The reference's greedy decode: `new` tokens after `prompt`, each the argmax of the reference's
/// logits over the sequence so far, and the smallest margin between the best and the second logit.
fn reference_greedy(fx: &Fixture, prompt: &[i32], new: usize, capacity: usize) -> (Vec<u32>, f64) {
    let reference = Reference::new(&fx.config, &fx.store);
    let mut sequence: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
    let (mut tokens, mut margin) = (Vec::new(), f64::MAX);
    for _ in 0..new {
        let run = reference.run(&sequence, capacity);
        let logits = run.logits.last().unwrap();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        margin = margin.min(logits[order[0]] - logits[order[1]]);
        sequence.push(order[0]);
        tokens.push(order[0] as u32);
    }
    (tokens, margin)
}

/// The 24 tokens the fixture generates after `[3, 17, 40, 8, 25, 11, 2]`: the independent reference's
/// greedy decode, which the driver rows on every backend compare their generation with
/// (`recorded_deepseek4_tokens_*`). Its smallest top-two logit margin is far above the device's
/// float noise (ADR-0101 tier 3).
const GREEDY: [u32; 24] = [
    28, 31, 6, 41, 20, 26, 15, 17, 9, 32, 33, 13, 40, 26, 9, 41, 42, 30, 11, 15, 32, 32, 31, 7,
];

/// SC-001 (tier 3): the reference's greedy decode over the fixture is the recorded literal; the
/// model on the oracle generates the same tokens, the prompt as the driver runs it (a prefill of the
/// largest multiple of the granule and decode steps for the rest). Mutation: swap two experts'
/// routing indices in the tracer; the oracle's tokens leave the literal and the row goes red.
#[test]
fn the_driver_literals_are_the_references_greedy_decode() {
    let fx = fixture();
    let prompt = [3, 17, 40, 8, 25, 11, 2];
    let (tokens, margin) = reference_greedy(&fx, &prompt, 24, 32);
    assert!(margin > 1e-3, "the smallest top-two margin is {margin}");
    assert_eq!(tokens, GREEDY);
    let model = model_of(&fx);
    assert_eq!(
        crate::components::testing::greedy_tokens(&*model, &fx.store, &prompt, 24, 32),
        GREEDY
    );
}

/// Which packed format a tensor of the fixture takes in a checkpoint stored as the release is: the
/// routed experts E2M1, every projection the attention and the shared experts read E4M3, and
/// nothing else packed. `None` for a tensor that stays dense.
fn packed_format_of(key: &str) -> Option<poot_quant::format::WeightFormat> {
    let linear = |suffix: &str| key.ends_with(suffix);
    if key.contains(".ffn.experts.") && linear(".weight") {
        Some(FP4)
    } else if [
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "shared_experts.w1.weight",
        "shared_experts.w2.weight",
        "shared_experts.w3.weight",
        "attn.indexer.wq_b.weight",
    ]
    .into_iter()
    .any(linear)
    {
        Some(E4M3)
    } else {
        None
    }
}

/// The fixture's checkpoint stored as the release stores it (experts E2M1, projections E4M3, each
/// under its bare linear id), and its twin with every packed tensor as the F32 values the literal
/// decoder reads from the same payload.
fn packed_and_decoded_fixture() -> (Fixture, Fixture) {
    let fx = fixture();
    let (mut packed, mut twin) = (WeightStore::builder(), WeightStore::builder());
    for (i, (key, entry)) in fx.store.iter().enumerate() {
        let Some(format) = packed_format_of(key.as_str()) else {
            packed.insert(key.clone(), entry.clone()).unwrap();
            twin.insert(key.clone(), entry.clone()).unwrap();
            continue;
        };
        let [out, k] = entry.shape()[..] else {
            panic!("{key} is not a projection")
        };
        // Scales around one, different in every block, so a wrong block shows.
        let payload = payload(format, [out, k], i as u64, |r, c| {
            122 + ((r * 3 + c * 5 + i) % 11) as u8
        });
        let decoded = literal(&payload);
        let bare = key.as_str().strip_suffix(".weight").unwrap();
        packed
            .insert(bare, WeightEntry::Packed(Arc::new(payload)))
            .unwrap();
        twin.insert(
            key.clone(),
            crate::components::testing::f32_entry(vec![out, k], &decoded),
        )
        .unwrap();
    }
    (
        Fixture {
            store: packed.build(),
            ..fx.clone()
        },
        Fixture {
            store: twin.build(),
            ..fx
        },
    )
}

/// A checkpoint that stores its routed experts E2M1 and its projections E4M3 loads through the one
/// body: the weight map holds one packed stack handle per expert projection and a packed handle per projection, and the
/// model equals (tier 2) the same model over the store decoded by an independent decoder, prefill
/// chunk and decode steps, every logit and carried state element. Mutation: map the packed
/// experts' stack to a dense-only view (`Stored`) in `names.rs`; the build fails on the
/// shape mismatch and the row goes red.
#[test]
fn a_checkpoint_with_packed_experts_and_projections_matches_its_decoded_twin() {
    let (packed, twin) = packed_and_decoded_fixture();
    let model = model_of(&packed);
    let count = model
        .weights()
        .iter()
        .filter(|(_, _, handle)| matches!(handle.format, HandleFormat::Packed(_)))
        .count();
    // Per layer: 3 expert-projection stacks (8 experts each), the shared expert's 3, the 5 attention
    // linears and, on the two CSA layers, the indexer's query projection.
    assert_eq!(count, 4 * (3 + 3 + 5) + 2);
    let want_model = model_of(&twin);
    let (want, want_state) = run_chunked(&*want_model, &twin.store, CAPACITY, &[4], &PROMPT[..6]);
    let (got, state) = run_chunked(&*model, &packed.store, CAPACITY, &[4], &PROMPT[..6]);
    close(&got, &want);
    close_all(&state, &want_state);
}

/// SC-005 (R470-004, tier 1) with the dimensions of the release's routed expert: a gate projection
/// `[2048, 4096]` stored E2M1 (the config's `moe_intermediate_size` by `hidden_size`) with a payload
/// whose scale differs in every row and block, contracted against a one-hot input, gives exactly
/// the hand-decoded column of the weight, through the oracle's row decode and a projection the
/// baseline's CPU route refused. Mutation: decode a row with the scale of the block before it
/// (`column / 32 - 1`); the column leaves the hand-decoded one and the row goes red.
#[test]
fn a_real_dims_fp4_expert_projection_equals_the_hand_decoded_column_real_dims() {
    use poot_graph_ir::{Builder, TensorType};
    use poot_quant::weights::{ExpertProj, MoeRole, WeightRole};

    use crate::components::linear::{Weight, linear};
    use crate::components::standard::oracle::Fx;

    let (out, k) = (2048, 4096);
    let id = WeightId::layer(3, WeightRole::Moe(MoeRole::Shared(ExpertProj::Gate)));
    let payload = payload(FP4, [out, k], 0x5eed, |row, block| {
        118 + ((row * 7 + block * 3) % 19) as u8
    });
    let decoded = literal(&payload);
    let mut fx = Fx::new(0);
    fx.packed(id, payload);
    let map: WeightMap = fx.map();
    let w = Weight::new(&map, id, &[out, k]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, k])).unwrap();
    let y = linear(&b, x, &w, None).unwrap();
    let g = b.finish(y).unwrap();
    let formats = poot_graph_plan::WeightFormats::from_weight_map(&map);
    let g = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
    // Columns on both sides of a scale-block edge and in the last (ragged-free) block.
    for column in [0, 31, 32, 2051, k - 1] {
        let mut one_hot = vec![0.0f32; k];
        one_hot[column] = 1.0;
        let got = fx.eval(&g, &[("x", &one_hot)], &[], &[]).0;
        let want: Vec<f32> = (0..out).map(|row| decoded[row * k + column]).collect();
        assert_eq!(got, want, "column {column}");
    }
}

/// The eight tokens the fixture generates after the ten-token `PROMPT`, token by token: the
/// reference's greedy decode (it reads no prefill), which the driver row of the same name compares a
/// prefill of eight plus two decode steps with.
const GRANULE_GREEDY: [u32; 8] = [36, 13, 21, 9, 32, 15, 32, 2];

/// SC-007 (F3, the model half): the ten-token prompt as a prefill of eight and two decode steps, as
/// the driver runs a prompt of the granule-four fixture, generates the tokens the reference's
/// token-by-token decode does, and so does the oracle decoding every token. Mutation: prefill at six
/// (the family's trace then refuses the step; the helper's alignment drops to six) or drop the
/// granule check; the row goes red.
#[test]
fn a_granule_four_prompt_decodes_as_the_reference() {
    let fx = fixture();
    let (tokens, margin) = reference_greedy(&fx, &PROMPT, 8, 32);
    assert!(margin > 1e-3, "the smallest top-two margin is {margin}");
    assert_eq!(tokens, GRANULE_GREEDY);
    let model = model_of(&fx);
    assert_eq!(model.config().prefill_granule.get(), 4);
    assert_eq!(
        crate::components::testing::greedy_tokens(&*model, &fx.store, &PROMPT, 8, 32),
        GRANULE_GREEDY,
        "prefill of eight and two decode steps"
    );
    // The same tokens decoding every prompt token one at a time.
    let (mut logits, mut state) = (Vec::new(), Vec::new());
    for (pos, &token) in PROMPT.iter().enumerate() {
        (logits, state) = run_step(
            &*model,
            &fx.store,
            Phase::Decode,
            32,
            &[token],
            pos as i32,
            &state,
        );
    }
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as i32;
    let mut token = argmax(&logits);
    let mut by_decode = vec![token as u32];
    for pos in PROMPT.len()..PROMPT.len() + 7 {
        (logits, state) = run_step(
            &*model,
            &fx.store,
            Phase::Decode,
            32,
            &[token],
            pos as i32,
            &state,
        );
        token = argmax(&logits);
        by_decode.push(token as u32);
    }
    assert_eq!(by_decode, GRANULE_GREEDY, "token by token");
}

fn target_of(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: match backend {
            Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

fn backends() -> [Backend; 3] {
    [
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
        Backend::SpirvVulkan,
    ]
}

const REPLAY: CompileOptions = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);

/// The traced and weight-bound graph of one step of `phase` at `tokens` tokens.
fn bound_step(model: &dyn Model, phase: Phase, tokens: usize) -> poot_graph_ir::Graph {
    let graph = model
        .trace(
            phase,
            shape(tokens, CAPACITY, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    poot_graph_plan::bind_packed_weights(
        &graph,
        &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
    )
    .unwrap()
}

/// The decode and the prefill graphs of the fixture, dense and with packed experts and projections,
/// compile for the wgpu, PTX and ROCm targets: a family runs on a backend when its graph plans
/// there.
#[test]
fn the_decode_and_prefill_graphs_compile_for_every_backend() {
    let (packed, _) = packed_and_decoded_fixture();
    for (label, fx) in [("dense", fixture()), ("packed", packed)] {
        let model = model_of(&fx);
        for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
            let graph = bound_step(&*model, phase, tokens);
            for backend in backends() {
                compile(&graph, &target_of(backend), &REPLAY)
                    .unwrap_or_else(|e| panic!("{label} {phase:?} on {backend:?}: {e}"));
            }
        }
    }
}

/// The grouped output projection of the fixture's BF16 checkpoint stays BF16 resident: every layer's `attn.o_a` const is
/// planned at BF16 (the checkpoint's bytes, no widened F32 copy) and read by one grouped dense contraction,
/// in the decode and the prefill graph on every backend. Mutation: declare `o_a` with `declare_f32` in
/// `sliding::attend`; the const is read through a `Cast` at F32, the storage dtype is F32 and the row goes red.
#[test]
fn a_bf16_checkpoint_keeps_the_grouped_o_projection_bf16_resident() {
    let fx = fixture();
    let model = model_of(&fx);
    let consts: Vec<String> = (0..4)
        .map(|layer| WeightId::layer(layer, WeightRole::Attn(AttnRole::OA)).const_name())
        .collect();
    for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
        let graph = bound_step(&*model, phase, tokens);
        for backend in backends() {
            let program = compile(&graph, &target_of(backend), &REPLAY)
                .unwrap_or_else(|e| panic!("{phase:?} on {backend:?}: {e}"));
            let g = program.graph();
            for name in &consts {
                let id = g
                    .values()
                    .iter_enumerated()
                    .find(|(_, meta)| meta.name.as_deref() == Some(name))
                    .map(|(id, _)| id)
                    .unwrap_or_else(|| panic!("{phase:?} on {backend:?}: no const {name}"));
                assert_eq!(
                    program.storage().storage(id).dtype(),
                    DType::BF16,
                    "{phase:?} on {backend:?}: {name} is resident at its checkpoint dtype"
                );
                assert!(
                    !g.eqns().iter().any(|eqn| matches!(
                        eqn.op,
                        poot_graph_ir::OpKind::Cast { .. }
                    ) && matches!(
                        eqn.inputs.first(),
                        Some(poot_graph_ir::Operand::Value(w)) if *w == id
                    )),
                    "{phase:?} on {backend:?}: {name} has no widened F32 copy"
                );
                assert!(
                    g.eqns().iter().any(|eqn| matches!(
                        eqn.op,
                        poot_graph_ir::OpKind::GroupedDenseContraction {
                            weight: DType::BF16,
                            ..
                        }
                    ) && matches!(
                        eqn.inputs.get(1),
                        Some(poot_graph_ir::Operand::Value(w)) if *w == id
                    )),
                    "{phase:?} on {backend:?}: {name} is read by a grouped dense contraction"
                );
            }
        }
    }
}

/// SC-006: `compile` keeps every witness lane: the compiled program's validation packet has the layout
/// of the traced graph's (three router lanes per layer, the hash-routed layer's included, and the
/// token guard), on every backend; the NaN row of the driver (`a_nan_deepseek4_router_weight_*`) shows
/// the lane reports. Mutation: make `dce` drop the validation outputs; the lane count differs and the
/// row goes red.
#[test]
fn compile_preserves_every_witness_lane() {
    let fx = fixture();
    let model = model_of(&fx);
    let layers = 4;
    for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
        let graph = bound_step(&*model, phase, tokens);
        let traced: Vec<(poot_graph_ir::ValidationId, String, usize)> = graph
            .witnesses()
            .map(|o| (o.id, o.name.to_owned(), graph.aval(o.value).numel()))
            .collect();
        assert_eq!(
            traced.len(),
            3 * layers + 1,
            "{phase:?}: three router witnesses per layer and the token guard"
        );
        for backend in backends() {
            let program = compile(&graph, &target_of(backend), &REPLAY).unwrap();
            let layout = &program.validation().layout;
            let compiled: Vec<_> = layout
                .entries
                .iter()
                .map(|e| (e.id, e.name.clone(), e.lane_count))
                .collect();
            assert_eq!(
                compiled, traced,
                "{phase:?} on {backend:?}: the packet layout"
            );
            assert_eq!(
                layout.lane_count,
                traced.iter().map(|(_, _, lanes)| lanes).sum::<usize>(),
                "{phase:?} on {backend:?}: the lane count"
            );
        }
    }
}

/// The config of `DeepSeek-V4-Flash-0731` at two layers (a CSA layer routed by hash, then an HCA layer
/// routed by score), 8 experts instead of 256 and a vocabulary of 1024: every dimension of a layer is
/// the release's (hidden 4096, 64 heads of 512, 1024-rank queries, 8 output groups, 2048-wide experts,
/// an indexer of 64 heads of 128 keeping 512, compression rates 4 and 128).
fn real_dims_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v4",
        "vocab_size": 1024,
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        "compress_ratios": [4, 128],
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "n_routed_experts": 8,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "num_hash_layers": 1,
        "moe_intermediate_size": 2048,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "swiglu_limit": 10.0,
        "max_position_embeddings": 4096,
        "eos_token_id": 1,
    })
}

/// A checkpoint of `config` as the release stores it, every value zero: the shapes and storage
/// formats are real, which is all a compile reads.
fn zero_checkpoint(config: serde_json::Value) -> Fixture {
    let mut store = WeightStore::builder();
    for (key, shape, stored) in tensors(&config) {
        let entry = match packed_format_of(&key) {
            Some(format) => {
                let bare = key.strip_suffix(".weight").unwrap().to_string();
                let weight =
                    poot_quant::PackedWeight::try_new(format, [shape[0], shape[1]]).unwrap();
                let source = |role| {
                    (
                        role,
                        StoredBytes::from(vec![0u8; weight.source_bytes(role)]),
                    )
                };
                let codes = poot_quant::SourceRole::Planar(poot_quant::OperandRole::Codes);
                let scales = poot_quant::SourceRole::Planar(poot_quant::OperandRole::Scale);
                store
                    .insert(
                        bare,
                        WeightEntry::Packed(Arc::new(
                            poot_quant::PackedPayload::try_new(
                                weight,
                                [source(codes), source(scales)],
                            )
                            .unwrap(),
                        )),
                    )
                    .unwrap();
                continue;
            }
            None => {
                let (dtype, width) = match stored {
                    Stored::Bf16 => (DType::BF16, 2),
                    Stored::F32 => (DType::F32, 4),
                    Stored::I64Table => (DType::I64, 8),
                };
                let n: usize = shape.iter().product();
                WeightEntry::Dense(
                    DenseWeight::try_new(dtype, shape, vec![0u8; n * width].into()).unwrap(),
                )
            }
        };
        store.insert(key, entry).unwrap();
    }
    Fixture {
        config,
        generation: None,
        store: store.build(),
    }
}

/// Recorded, not acceptance: the decode graph and a 128-token prefill graph (the family's granule at
/// the release's rates) of a real-dims two-layer V4 compile for the wgpu, PTX and ROCm targets. The
/// compile results are printed.
#[test]
fn a_real_dims_layer_fixture_compiles_for_every_backend_real_dims() {
    let fx = zero_checkpoint(real_dims_config());
    let model = model_of(&fx);
    assert_eq!(model.config().prefill_granule.get(), 128);
    let capacity = 256;
    for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 128)] {
        let graph = model
            .trace(
                phase,
                shape(tokens, capacity, LogitRows::Last),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let graph = poot_graph_plan::bind_packed_weights(
            &graph,
            &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
        )
        .unwrap();
        for backend in backends() {
            let program = compile(&graph, &target_of(backend), &REPLAY)
                .unwrap_or_else(|e| panic!("{phase:?} on {backend:?}: {e}"));
            eprintln!(
                "real-dims V4 {phase:?} on {backend:?}: compiled, {} equations",
                program.planned().count()
            );
        }
    }
}
