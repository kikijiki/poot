//! GLM-5.3-Flash rows: the fixtures against their independent reference, chunked prefill against
//! decode, the step shapes, a missing tensor, the multi-token-prediction tensors, the mHC streams and
//! the config.

mod reference;

use std::num::NonZeroUsize;

use poot_graph_ir::Slot;
use poot_graph_plan::{CompileLimits, CompileOptions, Target, compile};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_tensor::DType;

use super::*;
use crate::components::standard::oracle::{eval, run_shape, run_step, step_shape};
use crate::components::testing::{assert_missing_weight_is_named, build, close, close_all, values};
use crate::model::{ConfigReason, HiddenCarry, LogitRows, ShapeReason, StepShape};
use crate::registry::Registry;
use reference::Reference;
use std::collections::BTreeMap;

/// The cache capacity of the rows here: 8 pools of the indexer's pool size 2.
const CAPACITY: usize = 16;

/// Ten tokens.
const PROMPT: [i32; 10] = [3, 17, 40, 8, 25, 9, 31, 2, 44, 12];

fn model_of(fx: &Fixture) -> Box<dyn Model> {
    build(&ENTRY, &fx.raw(), &fx.store)
}

fn shape(tokens: usize, capacity: usize, logits: LogitRows) -> StepShape {
    StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(tokens).unwrap(),
        NonZeroUsize::new(capacity).unwrap(),
    )
    .with_logits(logits)
}

/// The names of the state a one-token decode step carries, in carry order.
fn state_names(model: &dyn Model, capacity: usize) -> Vec<String> {
    let g = model
        .trace(
            &StepSpec::step(shape(1, capacity, LogitRows::All)),
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
            capacity,
            &tokens[at..at + chunk],
            at as i32,
            &state,
        );
        logits.extend(out);
        (state, at) = (next, at + chunk);
    }
    while at < tokens.len() {
        let (out, next) = run_step(model, store, capacity, &tokens[at..=at], at as i32, &state);
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

/// SC-001: the fixture (a dense KDA layer, then DSA, KDA and DSA layers over routed experts of
/// different weights, with the pooled indexer leaving pools out) equals its independent reference at
/// every position and every carried state element: token by token, as a prefill, as two prefill
/// chunks and decode steps. Mutation: swap two experts' routing indices in the tracer (the router's
/// selected ids `0` and `1` exchanged before the experts read them); the first logit of the first
/// routed token moves and the row goes red.
#[test]
fn the_fixture_matches_its_independent_reference_at_every_position_and_state_element() {
    let fx = fixture();
    for (chunks, tokens) in [
        (&[][..], &PROMPT[..]),
        (&[10][..], &PROMPT[..]),
        (&[6, 3][..], &PROMPT[..]),
        (&[4][..], &PROMPT[..7]),
    ] {
        assert_matches_reference(&fx, CAPACITY, chunks, tokens);
    }
}

/// SC-002 (F5): a fine-tune of the architecture the old pinned counts refused (three layers in
/// another order over other expert counts and heads, a pool size of three, no multi-token-prediction
/// layer) and no manifest builds as the same family and matches its own reference. Mutation:
/// reinstate a pinned layer count in `from_raw`; the build is refused and the row goes red.
#[test]
fn a_finetune_like_fixture_builds_and_matches_its_reference() {
    let fx = finetune_fixture();
    let model = model_of(&fx);
    assert_eq!(model.config().family, FAMILY);
    assert_matches_reference(&fx, CAPACITY, &[5], &PROMPT[..8]);
}

/// SC-003 (F6): a store missing one required tensor, of any kind, fails `Registry::build` with the
/// typed missing-weight error naming it, before any trace. Mutation: skip the row check in `Rows`;
/// the missing tensor is found only at bind time and the row goes red.
#[test]
fn a_missing_required_tensor_is_named_before_any_trace() {
    let registry = Registry::builtin().unwrap();
    let fx = fixture();
    let p = "model.language_model.";
    for removed in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.1.post_attention_layernorm.weight",
        "model.language_model.layers.1.hc_attn_fn",
        "model.language_model.layers.2.hc_ffn_scale",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.v_conv1d.weight",
        "model.language_model.layers.0.self_attn.f_b_proj.weight",
        "model.language_model.layers.2.self_attn.A_log",
        "model.language_model.layers.2.self_attn.dt_bias",
        "model.language_model.layers.2.self_attn.o_norm.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.1.self_attn.q_a_proj.weight",
        "model.language_model.layers.1.self_attn.kv_b_proj.weight",
        "model.language_model.layers.1.self_attn.indexer.wq_b.weight",
        "model.language_model.layers.3.self_attn.indexer.k_norm.bias",
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate",
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape",
        "model.language_model.layers.1.mlp.gate.weight",
        "model.language_model.layers.1.mlp.gate.e_score_correction_bias",
        "model.language_model.layers.2.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.5.up_proj.weight",
    ] {
        assert!(removed.starts_with(p) || removed == "lm_head.weight");
        assert_missing_weight_is_named(&registry, &fx, removed);
    }
}

/// SC-005 (F2): a six-token prefill equals two three-token prefills, which equal one three-token
/// prefill followed by three decode steps, at every logit and every carried element of every layer
/// (the conv windows and matrices of the KDA layers, the latent and indexer caches of the DSA
/// ones), each against the reference. Mutation: ignore the carried recurrent state in the second
/// chunk (read zeros for the matrices of a KDA layer); the continued rows go red.
#[test]
fn a_prefill_equals_the_same_tokens_in_chunks_and_as_a_prefill_then_decodes() {
    let fx = fixture();
    for chunks in [&[6][..], &[3, 3][..], &[3][..]] {
        assert_matches_reference(&fx, CAPACITY, chunks, &PROMPT[..6]);
    }
}

/// SC-007 (F8): a four-token prefill and a decode step leave every KDA and DSA state element where
/// five decode steps do, and give the same logits. Mutation: skip the KDA state write in prefill
/// (carry the input matrices unchanged when the step has several tokens); the matrices of the
/// prefill path stay zero and the row goes red.
#[test]
fn a_prefill_then_a_decode_equals_decode_steps_at_every_state_element() {
    let fx = fixture();
    let model = model_of(&fx);
    let tokens = &PROMPT[..5];
    let (stepped, stepped_state) = run_chunked(&*model, &fx.store, CAPACITY, &[], tokens);
    let (chunked, chunked_state) = run_chunked(&*model, &fx.store, CAPACITY, &[4], tokens);
    close(&chunked, &stepped);
    close_all(&chunked_state, &stepped_state);
    let names = state_names(&*model, CAPACITY);
    assert!(names.iter().any(|n| n.starts_with("kda.")));
    assert!(names.iter().any(|n| n.starts_with("kv.")));
    // The matrices are live: a KDA layer's state is not the zero it started from.
    for (name, state) in names.iter().zip(&chunked_state) {
        if name.ends_with(".state") {
            assert!(state.iter().any(|&v| v.abs() > 1e-6), "{name} stayed zero");
        }
    }
}

/// POOT-1109 SC-001: a step that asks for the hidden-state carry writes the reference's closing-norm
/// output (the normalized mean of the `hc_mult` streams) at every position it carries, for a prefill
/// chunk read as a last-token step, a verify window over all positions and a replay of the kept
/// prefix after the rollback. The carry is not rolled back: the replay leaves the window's later
/// rows where they were. The head stays last-only (the chunk's logits are the reference's last
/// position). Mutation: write the pre-norm mean (`carry_hidden(b, mean)`) or one stream's value in
/// the tracer; the first assertion is red at every element.
#[test]
fn the_hidden_carry_is_the_references_post_norm_stream_mean_for_a_chunk_a_window_and_a_replay() {
    let fx = fixture();
    let model = model_of(&fx);
    let tokens = &PROMPT[..9];
    let want = Reference::new(&fx.config, &fx.store).run(
        &tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
        CAPACITY,
    );
    let width = want.hidden[0].len();
    let rows = |range: std::ops::Range<usize>| -> Vec<f32> {
        range
            .flat_map(|p| want.hidden[p].iter().map(|&v| v as f32))
            .collect()
    };
    let positions = 8;
    let carry = HiddenCarry {
        positions: NonZeroUsize::new(positions).unwrap(),
    };
    let step = |logits, count, hidden| step_shape(count, CAPACITY, logits, hidden);
    let carry_of = |state: &[Vec<f32>]| -> Vec<f32> {
        assert_eq!(state.last().unwrap().len(), positions * width);
        state.last().unwrap().clone()
    };

    // A prefill chunk of six, logits at the last position only.
    let (logits, after_chunk) = run_shape(
        &*model,
        &fx.store,
        &StepSpec::step(step(LogitRows::Last, 6, Some(carry))),
        &tokens[..6],
        0,
        &[],
    );
    let mut expect = rows(0..6);
    expect.resize(positions * width, 0.0);
    close(&carry_of(&after_chunk), &expect);
    let last: Vec<f32> = want.logits[5].iter().map(|&v| v as f32).collect();
    close(&logits, &last);

    // A window of three over all positions, continuing the chunk: its positions are the carry's
    // first three, the older rows beyond stay.
    let (_, after_window) = run_shape(
        &*model,
        &fx.store,
        &StepSpec::step(step(LogitRows::All, 3, Some(carry))),
        &tokens[6..9],
        6,
        &after_chunk,
    );
    let mut expect = rows(6..9);
    expect.extend(&rows(3..6));
    expect.resize(positions * width, 0.0);
    close(&carry_of(&after_window), &expect);

    // The rollback restores the state the chunk left, not the carry; the kept prefix of two replays
    // as a chunk from there. Only its own two positions change.
    let mut restored = after_chunk.clone();
    *restored.last_mut().unwrap() = after_window.last().unwrap().clone();
    let (_, after_replay) = run_shape(
        &*model,
        &fx.store,
        &StepSpec::step(step(LogitRows::Last, 2, Some(carry))),
        &tokens[6..8],
        6,
        &restored,
    );
    let mut expect = rows(6..8);
    expect.extend(&rows(8..9));
    expect.extend(&rows(3..6));
    expect.resize(positions * width, 0.0);
    close(&carry_of(&after_replay), &expect);
}

/// SC-011 (the driver's alignment half): a six-token prompt split in the pieces `[4, 2]` the
/// driver plans for a chunk of four matches six decode steps at every output and every recurrent
/// state. A KDA step is a per-token unrolled recurrence, so the pieces are bounded by the caller's
/// trace limit and nothing in the family aligns them. (The equation budget half is the common
/// Builder's, which does not exist yet.)
#[test]
fn driver_pieces_of_four_and_two_equal_six_decode_steps() {
    let fx = fixture();
    let model = model_of(&fx);
    let tokens = &PROMPT[..6];
    let (stepped, stepped_state) = run_chunked(&*model, &fx.store, CAPACITY, &[], tokens);
    let (pieces, pieces_state) = run_chunked(&*model, &fx.store, CAPACITY, &[4, 2], tokens);
    close(&pieces, &stepped);
    close_all(&pieces_state, &stepped_state);
}

/// The step shapes the family cannot trace are typed refusals before any graph is built: a
/// capacity above the model's maximum and more tokens than the cache holds. The family has no
/// granule: any step length traces.
#[test]
fn step_shapes_the_family_cannot_trace_are_typed_refusals() {
    let model = model_of(&fixture());
    let trace = |tokens, capacity| {
        model.trace(
            &StepSpec::step(shape(tokens, capacity, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
    };
    let reason = |tokens, capacity| match trace(tokens, capacity) {
        Err(TraceError::ShapeUnsupported { reason, .. }) => reason,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    assert_eq!(reason(1, 65), ShapeReason::CapacityAboveMax { max: 64 });
    assert_eq!(reason(17, CAPACITY), ShapeReason::TokensAboveCapacity);
    assert_eq!(model.config().prefill_granule.get(), 1);
    for tokens in [1, 3, 6] {
        trace(tokens, CAPACITY).unwrap();
    }
}

/// A cache shorter than a pool holds no complete pool: the DSA layers attend the positions after
/// the last complete pool alone (all of them), as the reference does.
#[test]
fn a_cache_shorter_than_a_pool_has_no_complete_pool() {
    let fx = fixture();
    assert_matches_reference(&fx, 1, &[], &PROMPT[..1]);
    assert_matches_reference(&fx, 3, &[3], &PROMPT[..3]);
}

/// The `rows` of a step do not mix: two rows with different prompts give, row by row, the logits
/// and carried state of the same prompts run alone (a prefill of three, then a decode step).
#[test]
fn two_rows_do_not_share_state() {
    let fx = fixture();
    let model = model_of(&fx);
    let prompts: [[i32; 4]; 2] = [[3, 17, 40, 8], [9, 31, 2, 44]];
    let single: Vec<_> = prompts
        .iter()
        .map(|p| run_chunked(&*model, &fx.store, CAPACITY, &[3], p))
        .collect();

    let two = |tokens: usize| {
        StepShape::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(tokens).unwrap(),
            NonZeroUsize::new(CAPACITY).unwrap(),
        )
        .with_logits(LogitRows::All)
    };
    let run = |tokens: usize, ids: Vec<i32>, pos: Vec<i32>, state: Vec<Vec<f32>>| {
        let g = model
            .trace(
                &StepSpec::step(two(tokens)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
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
    let prefix = |row: usize| prompts[row][..3].to_vec();
    let (logits, state) = run(
        3,
        [prefix(0), prefix(1)].concat(),
        (0..3).chain(0..3).collect(),
        Vec::new(),
    );
    let (logits_after, state) = run(1, vec![prompts[0][3], prompts[1][3]], vec![3, 3], state);
    let vocab = model.config().vocab;
    for row in 0..2 {
        let (want, want_state) = &single[row];
        let mut got: Vec<f32> = logits[row * 3 * vocab..(row + 1) * 3 * vocab].to_vec();
        got.extend_from_slice(&logits_after[row * vocab..(row + 1) * vocab]);
        close(&got, want);
        for (carried, want) in state.iter().zip(want_state) {
            let per_row = carried.len() / 2;
            close(&carried[row * per_row..(row + 1) * per_row], want);
        }
    }
}

/// `fx` with `key` shifted by `delta`, kept in its stored dtype.
fn shifted(fx: &Fixture, key: &str, delta: f32) -> Fixture {
    let mut store = WeightStore::builder();
    for (k, entry) in fx.store.iter() {
        let entry = if k.as_str() == key {
            let dense = match entry {
                WeightEntry::Dense(dense) => dense,
                WeightEntry::Packed(_) => panic!("shift a dense tensor"),
            };
            let values: Vec<f32> = values(entry).into_iter().map(|v| v + delta).collect();
            let bytes: Vec<u8> = match dense.dtype() {
                DType::BF16 => values
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
                DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                other => panic!("shift a float tensor, not {other:?}"),
            };
            WeightEntry::Dense(
                DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into()).unwrap(),
            )
        } else {
            entry.clone()
        };
        store.insert(k.clone(), entry).unwrap();
    }
    Fixture {
        config: fx.config.clone(),
        generation: None,
        store: store.build(),
    }
}

/// SC-009: the stack runs the mHC streams: shifting any of a layer's hyper-connection tensors (the
/// projection, base and scale of either site) changes the logits. Mutation: bypass the mHC block in
/// the stack (read the first stream and add the block's output to every stream); the shifted
/// weights are read by nothing, the logits do not move and the row goes red.
#[test]
fn the_hyper_connection_weights_decide_the_logits() {
    let fx = fixture();
    let tokens = &PROMPT[..4];
    let base = {
        let model = model_of(&fx);
        run_chunked(&*model, &fx.store, CAPACITY, &[4], tokens).0
    };
    for layer in 0..4 {
        for name in [
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ] {
            let key = format!("model.language_model.layers.{layer}.{name}");
            let moved = shifted(&fx, &key, 0.4);
            let model = model_of(&moved);
            let (logits, _) = run_chunked(&*model, &moved.store, CAPACITY, &[4], tokens);
            assert!(
                logits.iter().zip(&base).any(|(a, b)| (a - b).abs() > 1e-5),
                "{key} does not move the logits"
            );
        }
    }
}

/// SC-010 (R-640-2, POOT-1110): a store carrying a multi-token-prediction layer loads it: the
/// first layer's fusing projection, its three norms and its DSA block and feed-forward are read, not
/// merely known, and the copies of the embedding and the head the layer may store are known. The same
/// store under a config with no such layer reports every tensor of the layer unknown, so the claim
/// comes from the config's layer count. Mutation: skip the `Rows::mtp` call in `weights_of`; the
/// tower cannot find the layer's weights and the build panics naming the missing map entry.
#[test]
fn a_multi_token_prediction_layer_is_loaded_and_its_shared_copies_are_known() {
    let fx = fixture();
    let model = model_of(&fx);
    let report = model.weights().report(&fx.store);
    let prefix = "model.language_model.layers.4.";
    let layer: Vec<_> = fx
        .store
        .keys()
        .filter(|key| key.as_str().starts_with(prefix))
        .cloned()
        .collect();
    assert!(
        layer.len() > 30,
        "the fixture stores a whole layer: {}",
        layer.len()
    );
    for key in &layer {
        assert!(report.loaded.contains(key), "{key} is loaded");
        assert!(!report.unknown.contains(key), "{key} is not unknown");
    }
    for name in ["eh_proj.weight", "enorm.weight", "hnorm.weight"] {
        let key = poot_quant::weights::WeightKey::from(format!("{prefix}{name}"));
        assert!(report.loaded.contains(&key), "{key} is loaded");
    }
    assert_eq!(
        report.unknown.len(),
        0,
        "every tensor is read: {:?}",
        report.unknown
    );

    // No MTP layer declared: the same tensors are not recognised.
    let mut config = fx.config.clone();
    config["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
    let model = build(
        &ENTRY,
        &RawConfig::HfJson {
            config: &config,
            generation: None,
        },
        &fx.store,
    );
    let report = model.weights().report(&fx.store);
    assert_eq!(report.known.len(), 0);
    assert_eq!(report.unknown.len(), layer.len());
}

/// What the layer's checkpoint rows name as extra copies: a stored `embed_tokens.weight` or
/// `shared_head.head.weight` of the first MTP layer (the shared head reads the target's) and every
/// tensor of a second layer are known, not unknown and not loaded.
#[test]
fn the_shared_copies_and_a_second_layer_are_known_and_not_loaded() {
    let fx = fixture();
    let mut store = poot_quant::weights::WeightStore::builder();
    for (key, entry) in fx.store.iter() {
        store
            .insert(key.as_str().to_string(), entry.clone())
            .unwrap();
    }
    let copies = [
        "model.language_model.layers.4.embed_tokens.weight",
        "model.language_model.layers.4.shared_head.head.weight",
        "model.language_model.layers.5.eh_proj.weight",
    ];
    let dense = |shape: Vec<usize>| {
        WeightEntry::Dense(
            DenseWeight::try_new(
                DType::F32,
                shape.clone(),
                vec![0u8; 4 * shape.iter().product::<usize>()].into(),
            )
            .unwrap(),
        )
    };
    for key in copies {
        store.insert(key.to_string(), dense(vec![2, 2])).unwrap();
    }
    let store = store.build();
    let mut config = fx.config.clone();
    config["text_config"]["num_nextn_predict_layers"] = serde_json::json!(2);
    let model = build(
        &ENTRY,
        &RawConfig::HfJson {
            config: &config,
            generation: None,
        },
        &store,
    );
    let report = model.weights().report(&store);
    for key in copies {
        let key = poot_quant::weights::WeightKey::from(key);
        assert!(report.known.contains(&key), "{key} is known");
        assert!(!report.unknown.contains(&key), "{key} is not unknown");
        assert!(!report.loaded.contains(&key), "{key} is not loaded");
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
/// dimensions is assumed. Mutation: accept any layer type as DSA; the unknown-type row goes red.
#[test]
fn a_config_the_family_cannot_trace_is_a_typed_error_naming_the_field() {
    use ConfigReason::*;
    use serde_json::json;
    assert_eq!(
        config_error_of(|c| c["text_config"]["layer_types"][1] = json!("full_attention")),
        ("layer_types", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["layer_types"] = json!(["linear_attention"])),
        ("layer_types", Length { expected: 4 })
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["mlp_layer_types"][2] = json!("moe")),
        ("mlp_layer_types", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| {
            c.as_object_mut().unwrap().remove("text_config");
        }),
        ("text_config", Missing)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["qk_rope_head_dim"] = json!(8)),
        ("qk_rope_head_dim", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["indexer_types"][2] = json!("shared")),
        ("indexer_types", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["index_topk"] = json!(5)),
        ("index_topk", NotDivisible { by: 2 })
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["scoring_func"] = json!("softmax")),
        ("scoring_func", Unsupported)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["hc_mult"] = json!(0)),
        ("hc_mult", Zero)
    );
    assert_eq!(
        config_error_of(|c| c["text_config"]["num_experts_per_tok"] = json!(9)),
        ("top_k", Exceeds { max: 6 })
    );
    assert_eq!(
        config_error_of(
            |c| c["text_config"]["linear_attn_config"]["gate_lower_bound"] = json!(0.5)
        ),
        ("gate_lower_bound", Unsupported)
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
/// (`recorded_glm53_flash_tokens_*`). Its smallest top-two logit margin is far above the device's
/// float noise (ADR-0101 tier 3).
const GREEDY: [u32; 24] = [
    29, 46, 25, 32, 34, 21, 17, 7, 2, 31, 37, 12, 30, 0, 4, 8, 4, 24, 21, 43, 9, 31, 24, 21,
];

/// The reference's log-probabilities over the whole vocabulary at each of the 24 positions the driver
/// samples [`GREEDY`] from: one line per position, the 48 values in token order, which the device row
/// `glm53_flash_every_position_matches_the_reference` includes. Position `k` is the one after the
/// prompt and `k` generated tokens.
fn reference_logprob_rows(fx: &Fixture) -> Vec<Vec<f64>> {
    let prompt = [3usize, 17, 40, 8, 25, 11, 2];
    let sequence: Vec<usize> = prompt
        .iter()
        .copied()
        .chain(GREEDY[..GREEDY.len() - 1].iter().map(|&t| t as usize))
        .collect();
    let run = Reference::new(&fx.config, &fx.store).run(&sequence, 32);
    run.logits[prompt.len() - 1..]
        .iter()
        .map(|logits| {
            let max = logits.iter().copied().fold(f64::MIN, f64::max);
            let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
            logits.iter().map(|l| l - log_sum).collect()
        })
        .collect()
}

/// POOT-1056: the recorded per-position log-probabilities the device rows compare with are the
/// independent reference's over the fixture, within `1e-6` (they are printed with eight significant
/// digits). Regenerate by replacing the file with the text in the failure message. Mutation: change
/// one value in the file; the row goes red.
#[test]
fn the_device_logprob_literals_are_the_references() {
    let recorded = include_str!(
        "../../../../poot-llm/src/driver/device_tests/glm53_flash_reference_logprobs.txt"
    );
    let want = reference_logprob_rows(&fixture());
    let regenerated: String = want
        .iter()
        .map(|row| {
            let values: Vec<String> = row.iter().map(|v| format!("{v:.8e}")).collect();
            values.join(" ") + "\n"
        })
        .collect();
    let got: Vec<Vec<f64>> = recorded
        .lines()
        .map(|line| line.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(got.len(), want.len(), "regenerated:\n{regenerated}");
    for (position, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(got.len(), want.len(), "regenerated:\n{regenerated}");
        for (token, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6,
                "position {position} token {token}: recorded {got}, reference {want}; regenerated:\n{regenerated}"
            );
        }
    }
}

/// The prefix and the bonus token of the device rows' MTP chain, and its length.
const MTP_DEVICE_PREFIX: [usize; 6] = [3, 17, 40, 8, 25, 9];
const MTP_DEVICE_BONUS: usize = 31;
const MTP_DEVICE_K: usize = 3;

/// POOT-1110: the logits the device rows of the MTP program compare with (one line per draft step)
/// are the independent reference's chain over the fixture, within `1e-6`. Regenerate by replacing the
/// file with the text in the failure message. Mutation: change one value in the file; the row goes
/// red.
#[test]
fn the_device_mtp_logit_literals_are_the_references_chain() {
    let recorded = include_str!(
        "../../../../poot-llm/src/driver/device_tests/glm53_flash_reference_mtp_logits.txt"
    );
    let fx = fixture();
    let want = Reference::new(&fx.config, &fx.store)
        .mtp_chain(&MTP_DEVICE_PREFIX, MTP_DEVICE_BONUS, MTP_DEVICE_K)
        .logits;
    let regenerated: String = want
        .iter()
        .map(|row| {
            let values: Vec<String> = row.iter().map(|v| format!("{v:.8e}")).collect();
            values.join(" ") + "\n"
        })
        .collect();
    let got: Vec<Vec<f64>> = recorded
        .lines()
        .map(|line| line.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(got.len(), want.len(), "regenerated:\n{regenerated}");
    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(got.len(), want.len(), "regenerated:\n{regenerated}");
        for (token, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6,
                "step {step} token {token}: recorded {got}, reference {want}; regenerated:\n{regenerated}"
            );
        }
    }
}

/// SC-001 (tier 3): the reference's greedy decode over the fixture is the recorded literal, and the
/// model on the oracle generates the same tokens. Mutation: swap two experts' routing indices in
/// the tracer; the oracle's tokens leave the literal and the row goes red.
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

const REPLAY: CompileOptions = CompileOptions::new(CompileLimits::STANDARD);

/// The traced and weight-bound graph of one step of `tokens` tokens.
fn bound_step(model: &dyn Model, tokens: usize) -> poot_graph_ir::Graph {
    let graph = model
        .trace(
            &StepSpec::step(shape(tokens, CAPACITY, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    poot_graph_plan::bind_packed_weights(
        &graph,
        &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
    )
    .unwrap()
}

/// The decode and the prefill graphs of the fixture and of the fine-tune compile for the wgpu, PTX and
/// ROCm targets: a family runs on a backend when its graph plans there. Mutation: refuse one
/// operation the tracer emits on the ROCm planner (`plan_eqn` on `AmdGcn`); the ROCm rows go red
/// naming the equation.
#[test]
fn the_decode_and_prefill_graphs_compile_for_every_backend() {
    for (label, fx) in [("fixture", fixture()), ("finetune", finetune_fixture())] {
        let model = model_of(&fx);
        for tokens in [1, 4] {
            let graph = bound_step(&*model, tokens);
            for backend in backends() {
                compile(&graph, &target_of(backend), &REPLAY)
                    .unwrap_or_else(|e| panic!("{label} {tokens}-token step on {backend:?}: {e}"));
            }
        }
    }
}

/// `compile` keeps every witness lane: the compiled program's validation packet has the layout of the
/// traced graph's (three router lanes per routed layer), on every backend. Mutation: make `dce` drop
/// the validation outputs; the lane count differs and the row goes red.
#[test]
fn compile_preserves_every_witness_lane() {
    let fx = fixture();
    let model = model_of(&fx);
    let routed_layers = 3;
    for tokens in [1, 4] {
        let graph = bound_step(&*model, tokens);
        let traced: Vec<(poot_graph_ir::ValidationId, String, usize)> = graph
            .witnesses()
            .map(|o| (o.id, o.name.to_owned(), graph.aval(o.value).numel()))
            .collect();
        assert_eq!(
            traced.len(),
            3 * routed_layers,
            "{tokens}-token step: three router witnesses per routed layer"
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
                "{tokens}-token step on {backend:?}: the packet layout"
            );
            assert_eq!(
                layout.lane_count,
                traced.iter().map(|(_, _, lanes)| lanes).sum::<usize>(),
                "{tokens}-token step on {backend:?}: the lane count"
            );
        }
    }
}

/// The carried state of a program by the name the graph gives each value, so the two programs of a
/// driver step (the target's and the multi-token-prediction layer's) hand the carry over by name
/// and each keeps its own caches.
type NamedState = BTreeMap<String, Vec<f32>>;

/// One step of `model` over the named state: every value the graph carries is read from `state` by
/// name (zeros when absent) and written back after the step. Returns the step's logits.
fn run_named(
    model: &dyn Model,
    store: &WeightStore,
    step: &StepSpec,
    tokens: &[i32],
    start: i32,
    state: &mut NamedState,
) -> Vec<f32> {
    let graph = model
        .trace(step, poot_graph_ir::TraceLimits::for_tests())
        .unwrap();
    let names: Vec<String> = graph
        .state()
        .map(|(input, _)| graph.meta(input).name.clone().unwrap())
        .collect();
    let input: Vec<Vec<f32>> = graph
        .state()
        .zip(&names)
        .map(|((input, _), name)| {
            state
                .get(name)
                .cloned()
                .unwrap_or_else(|| vec![0.0; graph.aval(input).numel()])
        })
        .collect();
    let (logits, next) = run_shape(model, store, step, tokens, start, &input);
    state.extend(names.into_iter().zip(next));
    logits
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

const CARRY_NAME: &str = crate::components::standard::HIDDEN_CARRY_STATE;

/// Run the traced `draft` chain the driver of a round would: the target prefills `prefix` with
/// the carry, then the layer catches up over every position with the token that follows it (the last
/// with `bonus`), then `k - 1` one-token steps, each fed the greedy token of the step before and the
/// post-norm state that step left in the carry. Returns every draft step's logits and tokens.
fn traced_mtp_chain(
    fx: &Fixture,
    prefix: &[i32],
    bonus: i32,
    k: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let model = model_of(fx);
    let n = prefix.len();
    let carry = HiddenCarry {
        positions: NonZeroUsize::new((k + 1).max(n)).unwrap(),
    };
    let mut target = NamedState::new();
    run_named(
        &*model,
        &fx.store,
        &StepSpec::step(step_shape(n, CAPACITY, LogitRows::Last, Some(carry))),
        prefix,
        0,
        &mut target,
    );
    // The layer's program has its own caches and reads the carry the target wrote.
    let mut mtp = NamedState::new();
    mtp.insert(CARRY_NAME.to_string(), target[CARRY_NAME].clone());
    let mtp_step =
        |tokens: usize| StepSpec::draft(step_shape(tokens, CAPACITY, LogitRows::Last, Some(carry)));
    let following: Vec<i32> = prefix[1..].iter().copied().chain([bonus]).collect();
    let mut logits = vec![run_named(
        &*model,
        &fx.store,
        &mtp_step(n),
        &following,
        0,
        &mut mtp,
    )];
    let mut drafts = vec![argmax(&logits[0])];
    for i in 1..k {
        let token = *drafts.last().unwrap() as i32;
        logits.push(run_named(
            &*model,
            &fx.store,
            &mtp_step(1),
            &[token],
            (n - 1 + i) as i32,
            &mut mtp,
        ));
        drafts.push(argmax(&logits[i]));
    }
    (logits, drafts)
}

/// SC-001 (POOT-1110): the traced `draft` program over the hidden-state carry equals a
/// from-scratch f64 chain for `k = 3`, at every draft step's every logit and every drawn token. The
/// reference runs the target and the layer one position at a time with no carry and no cache but its
/// own history; the traced run catches up over the whole prefix in one step of width `n`, then
/// recycles the layer's own post-norm state through the carry for two one-token steps. Mutations
/// (each restored before the next; a failure message is recorded in the card): swap the concatenation
/// order in `MtpLayer::trace` (`norm(hnorm)` before `norm(enorm)`); carry the closing mean before
/// the final norm (`step.carry_hidden(b, mean)` in `Tower::trace`); write the block's output before
/// `shared_head.norm` into the carry (`carry_last_hidden(.., block)`); leave the position-0 embedding
/// unmasked; each moves the first or the later draft logits and the row goes red.
#[test]
fn the_traced_mtp_chain_over_the_carry_equals_the_from_scratch_chain_for_k_3() {
    let fx = fixture();
    let reference = Reference::new(&fx.config, &fx.store);
    for (prefix, bonus) in [(&PROMPT[..6], PROMPT[6]), (&PROMPT[..3], PROMPT[3])] {
        let k = 3;
        let want = reference.mtp_chain(
            &prefix.iter().map(|&t| t as usize).collect::<Vec<_>>(),
            bonus as usize,
            k,
        );
        let (logits, drafts) = traced_mtp_chain(&fx, prefix, bonus, k);
        assert_eq!(logits.len(), k);
        for (step, (got, want)) in logits.iter().zip(&want.logits).enumerate() {
            let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
            eprintln!("prefix {} draft step {step}", prefix.len());
            close(got, &want);
        }
        assert_eq!(drafts, want.drafts, "the greedy draft tokens");
        // The drafts are not one repeated token: every step's input differs.
        assert!(
            drafts.windows(2).any(|w| w[0] != w[1]) || drafts.len() < 2,
            "the chain is not degenerate: {drafts:?}"
        );
    }
}

/// The layer's step needs the carry it reads: a `draft` shape that names none is a typed
/// refusal before any graph is built, and a carry narrower than the step's tokens is the carry's own
/// refusal. Mutation: drop the `DraftWithoutCarry` return in `check_draft_step`; the shape traces the
/// layer over a carry that does not exist (or panics at the `expect`) and the row goes red.
#[test]
fn an_mtp_step_without_a_carry_or_with_a_narrow_one_is_a_typed_refusal() {
    let model = model_of(&fixture());
    let refusal = |shape| match model.trace(
        &StepSpec::draft(shape),
        poot_graph_ir::TraceLimits::for_tests(),
    ) {
        Err(TraceError::ShapeUnsupported { reason, .. }) => reason,
        other => panic!("expected a shape refusal, got {:?}", other.map(|_| ())),
    };
    assert_eq!(
        refusal(shape(1, CAPACITY, LogitRows::Last)),
        ShapeReason::DraftWithoutCarry
    );
    let narrow = HiddenCarry {
        positions: NonZeroUsize::new(2).unwrap(),
    };
    assert_eq!(
        refusal(step_shape(3, CAPACITY, LogitRows::Last, Some(narrow))),
        ShapeReason::HiddenCarryNarrow {
            positions: 2,
            tokens: 3
        }
    );
}

/// A checkpoint that stores no MTP layer, or more than one (the reference cycles them by draft step,
/// which a step shape does not name), refuses the `draft` program typed and still traces its `step` program.
/// Mutation: drop the `count == 1` filter in `Tower::trace_mtp`; the two-layer checkpoint traces the
/// first layer for every step and the row goes red.
#[test]
fn a_checkpoint_without_exactly_one_mtp_layer_refuses_the_draft_program_typed() {
    let carry = HiddenCarry {
        positions: NonZeroUsize::new(4).unwrap(),
    };
    let step = step_shape(1, CAPACITY, LogitRows::Last, Some(carry));
    let model = model_of(&finetune_fixture());
    assert!(matches!(
        model.trace(
            &StepSpec::draft(step),
            poot_graph_ir::TraceLimits::for_tests()
        ),
        Err(TraceError::ProgramUnsupported {
            program: ProgramId::DRAFT,
            ..
        })
    ));

    let fx = fixture();
    let mut config = fx.config.clone();
    config["text_config"]["num_nextn_predict_layers"] = serde_json::json!(2);
    let model = build(
        &ENTRY,
        &RawConfig::HfJson {
            config: &config,
            generation: None,
        },
        &fx.store,
    );
    assert!(matches!(
        model.trace(
            &StepSpec::draft(step),
            poot_graph_ir::TraceLimits::for_tests()
        ),
        Err(TraceError::ProgramUnsupported {
            program: ProgramId::DRAFT,
            ..
        })
    ));
    model
        .trace(
            &StepSpec::step(shape(1, CAPACITY, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
}

/// Every other registered family refuses the `draft` program typed, whatever the shape: the program
/// is the family's to opt into, and none but this one does. Mutation: declare a `draft` program in
/// a dense decoder's schema; it traces an ordinary step for it and the row goes red naming the
/// family.
#[test]
fn every_other_family_refuses_the_draft_program_typed() {
    let carry = HiddenCarry {
        positions: NonZeroUsize::new(4).unwrap(),
    };
    let step = step_shape(1, 8, LogitRows::Last, Some(carry));
    let registry = Registry::builtin().unwrap();
    let mut refused = 0;
    for entry in registry.entries() {
        if entry.family == FAMILY {
            continue;
        }
        let fx = (entry.fixture)();
        let model = build(entry, &fx.raw(), &fx.store);
        match model.trace(
            &StepSpec::draft(step),
            poot_graph_ir::TraceLimits::for_tests(),
        ) {
            Err(TraceError::ProgramUnsupported {
                program: ProgramId::DRAFT,
                ..
            }) => refused += 1,
            other => panic!(
                "{} must refuse the draft program typed, got {:?}",
                entry.family,
                other.map(|_| ())
            ),
        }
    }
    assert!(
        refused > 10,
        "the registry's families were walked: {refused}"
    );
}

/// The MTP program plans for the wgpu, PTX and ROCm targets: a family runs on a backend when its
/// graph plans there (the catch-up width and the one-token chain step). Mutation: refuse the
/// `Select` the position-0 mask lowers to on the ROCm planner; the ROCm rows go red naming it.
#[test]
fn the_mtp_graphs_compile_for_every_backend() {
    let model = model_of(&fixture());
    let carry = HiddenCarry {
        positions: NonZeroUsize::new(6).unwrap(),
    };
    for tokens in [1, 6] {
        let graph = model
            .trace(
                &StepSpec::draft(step_shape(tokens, CAPACITY, LogitRows::Last, Some(carry))),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let graph = poot_graph_plan::bind_packed_weights(
            &graph,
            &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
        )
        .unwrap();
        for backend in backends() {
            compile(&graph, &target_of(backend), &REPLAY)
                .unwrap_or_else(|e| panic!("Mtp at {tokens} tokens on {backend:?}: {e}"));
        }
    }
}

/// The fixture with the embedding row of `token` set to NaN (stored as bf16, as the checkpoint).
fn with_nan_embedding_row(fx: &Fixture, token: usize) -> Fixture {
    let key = "model.language_model.embed_tokens.weight";
    let mut store = WeightStore::builder();
    for (name, entry) in fx.store.iter() {
        let entry = if name.as_str() == key {
            let mut rows = values(entry);
            let width =
                rows.len() / fx.config["text_config"]["vocab_size"].as_u64().unwrap() as usize;
            rows[token * width..(token + 1) * width].fill(f32::NAN);
            let bytes: Vec<u8> = rows
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let shape = match entry {
                WeightEntry::Dense(dense) => dense.shape().to_vec(),
                other => panic!("the embedding is stored dense: {other:?}"),
            };
            WeightEntry::Dense(DenseWeight::try_new(DType::BF16, shape, bytes.into()).unwrap())
        } else {
            entry.clone()
        };
        store.insert(name.as_str().to_string(), entry).unwrap();
    }
    Fixture {
        config: fx.config.clone(),
        generation: None,
        store: store.build(),
    }
}

/// A non-finite embedding row at position 0 is zeroed, not multiplied by zero: the step holds the
/// row of the token that follows position 0 (here the bonus token of a one-token prefix), which is
/// NaN, and the layer's logits equal the reference's, whose position-0 embedding is a literal zero.
/// Mutation: build the masked value as `embedded * 0` (the unselected branch of the select); the NaN
/// reaches `enorm` and every logit is NaN, so the row goes red.
#[test]
fn a_non_finite_embedding_row_at_position_zero_is_zeroed_not_propagated() {
    let fx = fixture();
    let (prefix, bonus) = ([PROMPT[0]], PROMPT[1] + 20);
    let want = Reference::new(&fx.config, &fx.store)
        .mtp_chain(&[prefix[0] as usize], bonus as usize, 1)
        .logits
        .remove(0);
    let poisoned = with_nan_embedding_row(&fx, bonus as usize);
    let (logits, _) = traced_mtp_chain(&poisoned, &prefix, bonus, 1);
    let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    assert!(
        logits[0].iter().all(|v| v.is_finite()),
        "the logits are finite"
    );
    close(&logits[0], &want);
}

/// POOT-569 SC-011 limit rows (moved to POOT-733): the KDA recurrence unrolls one body per token, so
/// the step's token count is the only thing that bounded its construction before the Builder had an
/// equation budget. A direct 6-token prefill under `max_tokens = 4` is refused before any tracer growth
/// (the body never runs), and under `max_eqns = 6` a prefill whose recurrence would append a seventh
/// equation is refused at it: no graph, the typed budget cause naming the equation count. Mutation:
/// drop the budget from `Builder::emit` (the equation is appended unbudgeted); the seventh-equation
/// row builds the whole unrolled graph and this row is red.
#[test]
fn the_kda_token_loop_is_bounded_by_the_callers_token_and_equation_limits() {
    use poot_graph_ir::{Attempted, BudgetResource, TraceLimits};
    use std::num::NonZeroU64;

    let fx = fixture();
    let model = model_of(&fx);
    let roomy = TraceLimits::for_tests();
    let six = shape(6, CAPACITY, LogitRows::Last);
    let full = model.trace(&StepSpec::step(six), roomy).unwrap();
    assert!(
        full.eqns().len() > 6,
        "the unrolled prefill is larger than the limit under test"
    );

    let tokens = TraceLimits {
        max_tokens: NonZeroUsize::new(4).unwrap(),
        max_literal_bytes: NonZeroU64::new(1 << 20).unwrap(),
        ..roomy
    };
    let Err(TraceError::Budget(refused)) = model.trace(&StepSpec::step(six), tokens) else {
        panic!("a 6-token step under max_tokens = 4 must be the typed budget refusal");
    };
    assert_eq!(refused.resource, BudgetResource::Tokens);
    assert_eq!((refused.limit, refused.requested), (4, 6));
    assert_eq!(refused.attempted, Attempted::Total(6));
    // The exact boundary traces.
    model
        .trace(&StepSpec::step(shape(4, CAPACITY, LogitRows::Last)), tokens)
        .unwrap();

    let equations = TraceLimits {
        max_eqns: NonZeroUsize::new(6).unwrap(),
        ..roomy
    };
    let Err(TraceError::Budget(refused)) = model.trace(&StepSpec::step(six), equations) else {
        panic!("the seventh equation must be the typed budget refusal");
    };
    assert_eq!(refused.resource, BudgetResource::Equations);
    assert_eq!((refused.limit, refused.used, refused.requested), (6, 6, 1));
    assert_eq!(refused.attempted, Attempted::Total(7));
}
