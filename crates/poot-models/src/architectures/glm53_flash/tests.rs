//! GLM-5.3-Flash rows: the fixtures against their independent reference, chunked prefill against
//! decode, the step shapes, a missing tensor, the multi-token-prediction tensors, the mHC streams and
//! the config.

mod reference;

use std::num::NonZeroUsize;

use poot_graph_ir::Slot;
use poot_graph_plan::{CompileLimits, CompileOptions, Submission, Target, compile};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_target::{AmdArch, Backend, DeviceCaps};
use poot_tensor::DType;

use super::*;
use crate::components::standard::oracle::{eval, run_step};
use crate::components::testing::{assert_missing_weight_is_named, build, close, close_all, values};
use crate::model::{ConfigReason, KvLayout, LogitRows, Phase, ShapeReason};
use crate::registry::Registry;
use reference::Reference;

/// The cache capacity of the rows here: 8 pools of the indexer's pool size 2.
const CAPACITY: usize = 16;

/// Ten tokens.
const PROMPT: [i32; 10] = [3, 17, 40, 8, 25, 9, 31, 2, 44, 12];

fn model_of(fx: &Fixture) -> Box<dyn Model> {
    build(&ENTRY, &fx.raw(), &fx.store)
}

fn shape(tokens: usize, capacity: usize, logits: LogitRows) -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits,
    }
}

/// The names of the state a one-token decode step carries, in carry order.
fn state_names(model: &dyn Model, capacity: usize) -> Vec<String> {
    let g = model
        .trace(Phase::Decode, shape(1, capacity, LogitRows::All))
        .unwrap();
    g.state
        .iter()
        .map(|&(input, _)| g.meta(input).name.clone().unwrap())
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
/// multi-token decode, and a capacity above the model's maximum. The family has no granule: any
/// prefill length traces.
#[test]
fn step_shapes_the_family_cannot_trace_are_typed_refusals() {
    let model = model_of(&fixture());
    let reason = |phase, tokens, capacity| match model
        .trace(phase, shape(tokens, capacity, LogitRows::Last))
    {
        Err(TraceError::ShapeUnsupported { reason, .. }) => reason,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    assert_eq!(
        reason(Phase::Decode, 2, CAPACITY),
        ShapeReason::DecodeTokens
    );
    assert_eq!(
        reason(Phase::Decode, 1, 65),
        ShapeReason::CapacityAboveMax { max: 64 }
    );
    assert_eq!(
        reason(Phase::Prefill, 17, CAPACITY),
        ShapeReason::TokensAboveCapacity
    );
    assert_eq!(model.config().prefill_granule.get(), 1);
    for tokens in [1, 3, 6] {
        model
            .trace(Phase::Prefill, shape(tokens, CAPACITY, LogitRows::Last))
            .unwrap();
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

    let two = |tokens: usize| StepShape {
        rows: NonZeroUsize::new(2).unwrap(),
        ..shape(tokens, CAPACITY, LogitRows::All)
    };
    let run = |phase, tokens: usize, ids: Vec<i32>, pos: Vec<i32>, state: Vec<Vec<f32>>| {
        let g = model.trace(phase, two(tokens)).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state
            .iter()
            .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
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
        Phase::Prefill,
        3,
        [prefix(0), prefix(1)].concat(),
        (0..3).chain(0..3).collect(),
        Vec::new(),
    );
    let (logits_after, state) = run(
        Phase::Decode,
        1,
        vec![prompts[0][3], prompts[1][3]],
        vec![3, 3],
        state,
    );
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

/// SC-010 (R-640-2): a store carrying a multi-token-prediction layer loads, and the load report
/// lists its tensors as known: not unknown and not loaded. The same store under a config with no
/// such layer reports them unknown, so the claim comes from the config's layer count. Mutation: drop
/// the known claims in `Rows::known`; the tensors are reported unknown and the row goes red.
#[test]
fn a_multi_token_prediction_layer_is_known_and_not_loaded() {
    let fx = fixture();
    let model = model_of(&fx);
    let report = model.weights().report(&fx.store);
    let mtp: Vec<&str> = [
        "eh_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "shared_head.norm.weight",
    ]
    .into_iter()
    .collect();
    for name in mtp {
        let key =
            poot_quant::weights::WeightKey::from(format!("model.language_model.layers.4.{name}"));
        assert!(fx.store.contains(key.as_str()), "{key} is in the fixture");
        assert!(report.known.contains(&key), "{key} is known");
        assert!(!report.unknown.contains(&key), "{key} is not unknown");
        assert!(!report.loaded.contains(&key), "{key} is not loaded");
    }
    assert_eq!(
        report.unknown.len(),
        0,
        "every other tensor is read: {:?}",
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
    assert_eq!(report.unknown.len(), 4);
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
    Target {
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
fn bound_step(
    model: &dyn Model,
    phase: Phase,
    tokens: usize,
) -> poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs> {
    let graph = model
        .trace(phase, shape(tokens, CAPACITY, LogitRows::Last))
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
        for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
            let graph = bound_step(&*model, phase, tokens);
            for backend in backends() {
                compile(&graph, &target_of(backend), &REPLAY)
                    .unwrap_or_else(|e| panic!("{label} {phase:?} on {backend:?}: {e}"));
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
    for (phase, tokens) in [(Phase::Decode, 1), (Phase::Prefill, 4)] {
        let graph = bound_step(&*model, phase, tokens);
        let traced: Vec<(poot_graph_ir::ValidationId, String, usize)> = graph
            .validation_outputs()
            .iter()
            .map(|o| (o.id, o.name.clone(), graph.aval(o.value).numel()))
            .collect();
        assert_eq!(
            traced.len(),
            3 * routed_layers,
            "{phase:?}: three router witnesses per routed layer"
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
