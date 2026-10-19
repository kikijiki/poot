//! MiniMax-M2 rows: the family's fixture and a finetune-like variant against an independent
//! reference written in plain f64 loops, the load-time refusals, and the compile rows.
//!
//! [`MiniMaxReference`] shares no math and no name table with the tracer or the components: it
//! reads the checkpoint tensors by their literal HF names and computes the model one token at a
//! time.

use std::num::NonZeroUsize;

use poot_graph_plan::{CompileLimits, CompileOptions, Target, compile};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_target::{AmdArch, Backend, DeviceCaps};

use super::*;
use crate::components::standard::oracle::{self, run_step};
use crate::components::testing::{
    CAP, assert_chunked_prefill_equals_decode, assert_missing_weight_is_named, build, close_all,
    q8_0_store, step_shape, values,
};
use crate::model::{LogitRows, ShapeReason, StepShape};
use crate::registry::Registry;

const PROMPT: [i32; 4] = [3, 17, 40, 8];

/// The dimensions the reference needs, read from a MiniMax-M2 `config.json`.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
    experts: usize,
    top_k: usize,
    eps: f64,
    theta: f64,
}

impl Dims {
    fn of(config: &serde_json::Value) -> Self {
        let n = |f: &str| crate::registry::Fixture::count(config, f);
        let x = |f: &str| config[f].as_f64().unwrap();
        Self {
            hidden: n("hidden_size"),
            layers: n("num_hidden_layers"),
            heads: n("num_attention_heads"),
            kv_heads: n("num_key_value_heads"),
            head_dim: n("head_dim"),
            rotary: n("rotary_dim"),
            experts: n("num_local_experts"),
            top_k: n("num_experts_per_tok"),
            eps: x("rms_norm_eps"),
            theta: x("rope_theta"),
        }
    }
}

fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let mean_square = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inverse = 1.0 / (mean_square + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inverse * w).collect()
}

/// `weight[out, in] * x`.
fn matvec(weight: &[f64], x: &[f64]) -> Vec<f64> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// A token-by-token MiniMax-M2 text model in plain f64 loops over the checkpoint `store`.
struct MiniMaxReference<'a> {
    dims: Dims,
    store: &'a WeightStore,
    /// Per layer, per position: the post-RoPE key and the value, each `[kv_heads * head_dim]`.
    keys: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<Vec<f64>>>,
}

impl<'a> MiniMaxReference<'a> {
    fn new(config: &serde_json::Value, store: &'a WeightStore) -> Self {
        let dims = Dims::of(config);
        Self {
            dims,
            store,
            keys: vec![Vec::new(); dims.layers],
            values: vec![Vec::new(); dims.layers],
        }
    }

    fn tensor(&self, name: &str) -> Vec<f64> {
        let entry: &WeightEntry = self
            .store
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no {name}"));
        values(entry).into_iter().map(f64::from).collect()
    }

    /// The logits of `token` at `position`; the caches take its key and value.
    fn step(&mut self, token: i32, position: usize) -> Vec<f64> {
        let d = self.dims;
        let embedding = self.tensor("model.embed_tokens.weight");
        let token = usize::try_from(token).unwrap();
        let mut h = embedding[token * d.hidden..(token + 1) * d.hidden].to_vec();
        for layer in 0..d.layers {
            let root = format!("model.layers.{layer}");
            let normed = rms_norm(
                &h,
                &self.tensor(&format!("{root}.input_layernorm.weight")),
                d.eps,
            );
            let attention = self.attention(layer, &normed, position);
            h.iter_mut().zip(&attention).for_each(|(v, u)| *v += u);
            let normed = rms_norm(
                &h,
                &self.tensor(&format!("{root}.post_attention_layernorm.weight")),
                d.eps,
            );
            let routed = self.routed_moe(layer, &normed);
            h.iter_mut().zip(&routed).for_each(|(v, u)| *v += u);
        }
        let normed = rms_norm(&h, &self.tensor("model.norm.weight"), d.eps);
        matvec(&self.tensor("lm_head.weight"), &normed)
    }

    /// Half-split rotation over the leading `rotary` values of a head; the tail passes through.
    fn rope(&self, head: &mut [f64], position: usize) {
        let d = self.dims;
        let half = d.rotary / 2;
        let source = head[..d.rotary].to_vec();
        for i in 0..half {
            let angle = position as f64 * d.theta.powf(-(2.0 * i as f64) / d.rotary as f64);
            let (sin, cos) = angle.sin_cos();
            head[i] = source[i] * cos - source[i + half] * sin;
            head[i + half] = source[i + half] * cos + source[i] * sin;
        }
    }

    fn attention(&mut self, layer: usize, x: &[f64], position: usize) -> Vec<f64> {
        let d = self.dims;
        let root = format!("model.layers.{layer}.self_attn");
        let project = |name: &str| matvec(&self.tensor(&format!("{root}.{name}.weight")), x);
        let (q, k, v) = (project("q_proj"), project("k_proj"), project("v_proj"));
        // The norm runs over the whole flattened projection, before the head split.
        let mut q = rms_norm(&q, &self.tensor(&format!("{root}.q_norm.weight")), d.eps);
        let mut k = rms_norm(&k, &self.tensor(&format!("{root}.k_norm.weight")), d.eps);
        for head in q.chunks_mut(d.head_dim) {
            self.rope(head, position);
        }
        for head in k.chunks_mut(d.head_dim) {
            self.rope(head, position);
        }
        assert_eq!(
            self.keys[layer].len(),
            position,
            "positions arrive in order"
        );
        self.keys[layer].push(k);
        self.values[layer].push(v);

        let groups = d.heads / d.kv_heads;
        let scale = 1.0 / (d.head_dim as f64).sqrt();
        let mut context = vec![0.0; d.heads * d.head_dim];
        for head in 0..d.heads {
            let kv = head / groups;
            let at = |cache: &Vec<Vec<f64>>, slot: usize| {
                cache[slot][kv * d.head_dim..(kv + 1) * d.head_dim].to_vec()
            };
            let query = &q[head * d.head_dim..(head + 1) * d.head_dim];
            let scores: Vec<f64> = (0..=position)
                .map(|slot| {
                    let key = at(&self.keys[layer], slot);
                    query.iter().zip(&key).map(|(a, b)| a * b).sum::<f64>() * scale
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f64 = weights.iter().sum();
            for (slot, weight) in weights.iter().enumerate() {
                let value = at(&self.values[layer], slot);
                for (dim, v) in value.iter().enumerate() {
                    context[head * d.head_dim + dim] += weight / total * v;
                }
            }
        }
        matvec(&self.tensor(&format!("{root}.o_proj.weight")), &context)
    }

    /// Sigmoid scores, selection on score plus the correction bias, the unbiased scores of the
    /// selected experts divided by their sum, and `w2 (silu(w1 x) * w3 x)` over the selection.
    fn routed_moe(&self, layer: usize, x: &[f64]) -> Vec<f64> {
        let d = self.dims;
        let root = format!("model.layers.{layer}.block_sparse_moe");
        let raw: Vec<f64> = matvec(&self.tensor(&format!("{root}.gate.weight")), x)
            .into_iter()
            .map(sigmoid)
            .collect();
        let bias = self.tensor(&format!("{root}.e_score_correction_bias"));
        let mut order: Vec<usize> = (0..d.experts).collect();
        order.sort_by(|&a, &b| {
            (raw[b] + bias[b])
                .total_cmp(&(raw[a] + bias[a]))
                .then(a.cmp(&b))
        });
        let selected = &order[..d.top_k];
        let total: f64 = selected.iter().map(|&e| raw[e]).sum();
        let mut out = vec![0.0; d.hidden];
        for &e in selected {
            let w = |name: &str| self.tensor(&format!("{root}.experts.{e}.{name}.weight"));
            let gate = matvec(&w("w1"), x);
            let up = matvec(&w("w3"), x);
            let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = matvec(&w("w2"), &act);
            out.iter_mut()
                .zip(&down)
                .for_each(|(o, v)| *o += raw[e] / total * v);
        }
        out
    }

    /// The caches as the model carries them: per layer a key and a value stack, each
    /// `[1, kv_heads, capacity, head_dim]`, positions not yet written zero.
    fn state(&self, capacity: usize) -> Vec<Vec<f32>> {
        let d = self.dims;
        let stack = |cache: &Vec<Vec<f64>>| {
            let mut out = vec![0.0f32; d.kv_heads * capacity * d.head_dim];
            for (position, row) in cache.iter().enumerate() {
                for head in 0..d.kv_heads {
                    for dim in 0..d.head_dim {
                        out[(head * capacity + position) * d.head_dim + dim] =
                            row[head * d.head_dim + dim] as f32;
                    }
                }
            }
            out
        };
        (0..d.layers)
            .flat_map(|l| [stack(&self.keys[l]), stack(&self.values[l])])
            .collect()
    }
}

fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
    let raw = RawConfig::HfJson {
        config,
        generation: None,
        preprocessor: None,
    };
    build(&ENTRY, &raw, store)
}

/// The graph's prefill of `prompt` and its next decode step against the reference, at every position
/// and every carried state element (tier 2; NaN fails): logits `[T, vocab]`, the caches after the
/// prefill and the logits and caches after one decode.
fn assert_matches_the_reference(config: &serde_json::Value, store: &WeightStore, prompt: &[i32]) {
    let model = model_of(config, store);
    let mut reference = MiniMaxReference::new(config, store);
    let want: Vec<f64> = prompt
        .iter()
        .enumerate()
        .flat_map(|(position, &token)| reference.step(token, position))
        .collect();
    let (got, state) = run_step(&*model, store, CAP, prompt, 0, &[]);
    oracle::assert_matches_f64(&got, &want, 1e-4);
    close_all(&state, &reference.state(CAP));

    let next = 5;
    let want = reference.step(next, prompt.len());
    let (got, state) = run_step(&*model, store, CAP, &[next], prompt.len() as i32, &state);
    oracle::assert_matches_f64(&got, &want, 1e-4);
    close_all(&state, &reference.state(CAP));
}

/// SC-001: the fixture (non-identical expert weights, so a routing error is observable) matches the
/// reference at every position and every carried state element. Mutation: reverse the expert
/// numbering in `components::router` (`ids = experts - 1 - ids`); the first logit of the prefill
/// moves and the row goes red.
#[test]
fn the_fixture_matches_its_independent_reference_at_every_position() {
    let fx = fixture();
    assert_matches_the_reference(&fx.config, &fx.store, &PROMPT);
}

/// SC-005 (the K/V half): three positions against the reference, with the keys and values of each
/// layer compared separately and in order, so a swapped cache write changes the carried state.
/// Mutation: write `v` into the key cache and `k` into the value cache; the state of layer 0 no
/// longer matches the reference.
#[test]
fn three_positions_carry_keys_then_values_equal_to_the_reference() {
    let fx = fixture();
    let model = model_of(&fx.config, &fx.store);
    let mut reference = MiniMaxReference::new(&fx.config, &fx.store);
    let mut state = Vec::new();
    for (position, token) in [3, 17, 40].into_iter().enumerate() {
        let want = reference.step(token, position);
        let (got, next) = run_step(&*model, &fx.store, CAP, &[token], position as i32, &state);
        oracle::assert_matches_f64(&got, &want, 1e-4);
        state = next;
    }
    let want = reference.state(CAP);
    assert_eq!(state.len(), 2 * 2, "a key and a value stack per layer");
    close_all(&state, &want);
    // The keys and the values really differ, so the order is observable.
    assert_ne!(want[0], want[1]);
}

/// Chunked prefill continuing from `Pos` equals token-by-token decode (one body for both phases).
#[test]
fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
    let fx = fixture();
    assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
}

/// The fixture's store with every projection and expert weight Q8_0, the router and norms dense.
#[test]
fn bf16_and_q8_0_experts_trace_through_the_one_body_and_match_the_decoded_reference() {
    crate::architectures::olmoe::moe_support::assert_bf16_and_q8_0_experts_trace_and_match(
        &ENTRY,
        &fixture(),
        |key| {
            key.ends_with("_proj.weight") || (key.contains(".experts.") && key.ends_with(".weight"))
        },
    );
}

/// SC-002: a config the old pinned counts (62 layers, 256 experts, 48 heads...) rejected, with no
/// manifest, builds through the registry, traces and matches its reference. Mutation: reinstate a pinned layer count
/// in `from_raw`; the build is refused (`num_hidden_layers`) and the row goes red.
#[test]
fn a_finetune_like_config_builds_traces_and_matches_its_reference() {
    let fx = finetune_fixture();
    assert_matches_the_reference(&fx.config, &fx.store, &PROMPT);
    let registry = Registry::builtin().unwrap();
    let model = registry.build(&fx.raw(), &fx.store).unwrap();
    assert_eq!(model.model.config().family, FAMILY);
}

/// SC-003: a store missing one required tensor, a router, its bias or one expert's projection
/// included, fails `Registry::build` with the typed missing-weight error naming it, before any
/// trace. Mutation: skip the row check in `map_moe_layer`; red.
#[test]
fn a_missing_required_tensor_is_named_before_any_trace() {
    let registry = Registry::builtin().unwrap();
    let fx = fixture();
    for removed in [
        "model.layers.1.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.e_score_correction_bias",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.1.block_sparse_moe.experts.7.w2.weight",
        "model.layers.0.block_sparse_moe.experts.3.w3.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "lm_head.weight",
    ] {
        assert_missing_weight_is_named(&registry, &fx, removed);
    }
}

fn config_error_of(config: &serde_json::Value) -> (&'static str, ConfigReason) {
    let raw = RawConfig::HfJson {
        config,
        generation: None,
        preprocessor: None,
    };
    match build_error(&raw) {
        ModelError::Config { field, reason, .. } => (field, reason),
        other => panic!("{config}: expected a config error, got {other:?}"),
    }
}

fn build_error(raw: &RawConfig<'_>) -> ModelError {
    match (ENTRY.build)(raw, &fixture().store) {
        Err(e) => e,
        Ok(_) => panic!("{raw:?}: expected a refusal"),
    }
}

/// A config this family cannot trace is a typed error naming the field, never a mis-traced graph.
#[test]
fn a_config_minimax_cannot_trace_is_a_typed_error_naming_the_field() {
    use serde_json::json;
    let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
        (
            "scoring_func",
            json!("softmax"),
            "scoring_func",
            ConfigReason::Unsupported,
        ),
        (
            "use_routing_bias",
            json!(false),
            "use_routing_bias",
            ConfigReason::Unsupported,
        ),
        (
            "use_qk_norm",
            json!(false),
            "use_qk_norm",
            ConfigReason::Unsupported,
        ),
        (
            "shared_intermediate_size",
            json!(64),
            "shared_intermediate_size",
            ConfigReason::Unsupported,
        ),
        (
            "num_experts_per_tok",
            json!(9),
            "top_k",
            ConfigReason::Exceeds { max: 8 },
        ),
        (
            "num_experts_per_tok",
            json!(0),
            "num_experts_per_tok",
            ConfigReason::Zero,
        ),
        (
            "num_local_experts",
            json!(0),
            "num_local_experts",
            ConfigReason::Zero,
        ),
        (
            "rotary_dim",
            json!(5),
            "rotary_dim",
            ConfigReason::NotDivisible { by: 2 },
        ),
        (
            "rotary_dim",
            json!(32),
            "rotary_dim",
            ConfigReason::Exceeds { max: 16 },
        ),
    ];
    for (key, value, field, reason) in cases {
        let mut config = fixture_config();
        config[*key] = value.clone();
        assert_eq!(
            config_error_of(&config),
            (*field, *reason),
            "{key} = {value}"
        );
    }
    let mut missing = fixture_config();
    missing.as_object_mut().unwrap().remove("rotary_dim");
    assert_eq!(
        config_error_of(&missing),
        ("rotary_dim", ConfigReason::Missing)
    );
}

/// A shape the family cannot trace is a typed refusal.
#[test]
fn a_step_above_the_cache_is_a_typed_refusal() {
    let fx = fixture();
    let model = model_of(&fx.config, &fx.store);
    let shape = step_shape(CAP + 1, CAP, LogitRows::All);
    match model.trace(
        &StepSpec::step(shape),
        poot_graph_ir::TraceLimits::for_tests(),
    ) {
        Err(TraceError::ShapeUnsupported { reason, .. }) => {
            assert_eq!(reason, ShapeReason::TokensAboveCapacity)
        }
        other => panic!("expected a shape refusal, got {other:?}"),
    }
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

/// The traced and weight-bound graph of one step of `tokens` tokens.
fn bound_step(model: &dyn Model, tokens: usize) -> poot_graph_ir::Graph {
    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(tokens).unwrap(),
        NonZeroUsize::new(CAP).unwrap(),
    );
    let graph = model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    poot_graph_plan::bind_packed_weights(
        &graph,
        &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
    )
    .unwrap()
}

const REPLAY: CompileOptions = CompileOptions::new(CompileLimits::STANDARD);

/// SC-006 (compile half): the decode and the prefill graphs compile for the wgpu, PTX and ROCm
/// targets. Mutation: skip the BF16-retyping step `compile` runs for the target
/// (`prepare_backend_dtypes`); `compile` refuses the graph and the row goes red.
#[test]
fn the_decode_and_prefill_graphs_compile_for_every_backend() {
    let fx = fixture();
    let model = model_of(&fx.config, &fx.store);
    for tokens in [1, 4] {
        let graph = bound_step(&*model, tokens);
        for backend in backends() {
            compile(&graph, &target_of(backend), &REPLAY)
                .unwrap_or_else(|e| panic!("{tokens}-token step on {backend:?}: {e}"));
        }
    }
}

/// SC-007: `compile` keeps every witness lane: the compiled program's validation packet has the
/// layout of the traced graph's (three lanes per layer plus the token guard), on every backend.
/// Mutation: make `dce` drop the validation outputs; the lane count differs and the row goes red.
#[test]
fn compile_preserves_every_witness_lane() {
    let fx = fixture();
    let model = model_of(&fx.config, &fx.store);
    let layers = Dims::of(&fx.config).layers;
    for tokens in [1, 4] {
        let graph = bound_step(&*model, tokens);
        let traced: Vec<(poot_graph_ir::ValidationId, String, usize)> = graph
            .witnesses()
            .map(|o| (o.id, o.name.to_owned(), graph.aval(o.value).numel()))
            .collect();
        assert!(
            traced.len() >= 3 * layers,
            "{tokens}-token step: the trace declares {} witnesses",
            traced.len()
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

/// Q8_0 experts are a checkpoint format, not a family: the packed copy of the fixture builds.
#[test]
fn a_packed_store_maps_every_expert_as_its_own_weight() {
    let fx = fixture();
    let store = q8_0_store(&fx.store, |key| key.contains(".experts."));
    let model = model_of(&fx.config, &store);
    let packed = model
        .weights()
        .iter()
        .filter(|(id, _, handle)| {
            matches!(
                id.role,
                poot_quant::weights::WeightRole::Moe(poot_quant::weights::MoeRole::Stack(..))
            ) && matches!(handle.format, poot_quant::weights::HandleFormat::Packed(_))
                && handle.shape[0] == 8
        })
        .count();
    assert_eq!(
        packed,
        2 * 3,
        "layers x projections: one packed stack of 8 experts each"
    );
}

/// The reference's greedy decode of `prompt`: `new` tokens, each with its log-probability and the
/// margin to the runner-up (a tier-3 token comparison is only meaningful where this is not tiny).
fn reference_greedy(prompt: &[i32], new: usize) -> Vec<(i32, f64, f64)> {
    let fx = fixture();
    let mut reference = MiniMaxReference::new(&fx.config, &fx.store);
    let mut logits = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        logits = reference.step(token, position);
    }
    let mut out = Vec::new();
    for i in 0..new {
        let mut ranked: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max = ranked[0].1;
        let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
        let token = ranked[0].0 as i32;
        out.push((token, ranked[0].1 - log_sum, ranked[0].1 - ranked[1].1));
        logits = reference.step(token, prompt.len() + i);
    }
    out
}

/// The literals the driver rows (`poot-llm`'s device tests) check, derived here from the reference:
/// 24 greedy tokens after a seven-token prompt (the smallest top-two logit margin over them is
/// 1.2e-4, far above an f32 rounding difference) and three greedy steps after a three-token prompt
/// with the log-probability of each. The device rows must reproduce them; this row keeps them from
/// drifting away from the reference.
#[test]
fn the_driver_literals_are_the_references_greedy_decode() {
    let long = reference_greedy(&[3, 17, 40, 8, 25, 11, 2], 24);
    let tokens: Vec<i32> = long.iter().map(|t| t.0).collect();
    assert_eq!(tokens, LONG_TOKENS);
    assert!(long.iter().all(|t| t.2 > 1e-4), "{long:?}");
    let short = reference_greedy(&[3, 17, 40], 3);
    assert_eq!(short.iter().map(|t| t.0).collect::<Vec<_>>(), SHORT_TOKENS);
    for ((_, logprob, _), want) in short.iter().zip(SHORT_LOGPROBS) {
        assert!((logprob - want).abs() < 1e-6, "{logprob} vs {want}");
    }
}

const LONG_TOKENS: [i32; 24] = [
    40, 38, 40, 38, 10, 21, 20, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 24, 46, 21, 46, 21,
];
const SHORT_TOKENS: [i32; 3] = [40, 40, 38];
const SHORT_LOGPROBS: [f64; 3] = [
    -0.006896560093689885,
    -0.14563738785911085,
    -0.8262569441422158,
];
