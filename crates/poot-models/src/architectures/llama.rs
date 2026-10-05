//! llama behind [`Model`]: the plain GQA + RoPE + SwiGLU decoder. HF `llama` and `mistral` (which
//! adds an optional sliding `window`) and GGUF `llama` (which llama.cpp also uses for Mistral) are
//! keys of this one family; there is no separate Mistral tracer.
//!
//! The body is `standard_stack` over `standard_layer` with the shared attention and gated FFN. A
//! llama-architecture GGUF stores Q and K rows permuted for interleaved RoPE, so the GGUF
//! family traces the interleaved rotation over the stored rows instead of un-permuting the weights
//! (the dot products Q.K are the same).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{WeightEntry, WeightMap, WeightStore};
use poot_tensor::DType;

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::standard::{
    LayerNorms, LayerParams, StackParams, StackWeights, Step, check_step, standard_layer,
    standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::GGUF_BASE;
use crate::names::{HF_BASE, family_weights, hf_base_shapes};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("llama");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "llama"),
        (ConfigSource::HfModelType, "mistral"),
        (ConfigSource::GgufArchitecture, "llama"),
    ],
    build,
    fixture,
};

/// llama's config facts: `LlamaConfig`'s and `MistralConfig`'s defaults (they agree), and
/// llama.cpp's GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("llama"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: Some(2),
};

/// A GGUF tensor llama.cpp writes for a llama3-style rope scaling: the per-frequency factors the
/// config's RoPE is rescaled by.
const GGUF_ROPE_FREQS: &str = "rope_freqs.weight";

/// The per-frequency factors of a `rope_freqs.weight`: llama.cpp writes them as one F32 vector.
fn rope_freq_factors(entry: &WeightEntry) -> Result<Vec<f32>, ModelError> {
    let refuse = |reason| ModelError::Config {
        family: FAMILY,
        field: GGUF_ROPE_FREQS,
        reason,
    };
    match entry {
        WeightEntry::Dense(dense) if dense.dtype() == DType::F32 => Ok(dense
            .bytes()
            .as_slice()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        _ => Err(refuse(ConfigReason::WrongType)),
    }
}

/// llama's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaParams {
    vocab: usize,
    width: usize,
    inter: usize,
    layers: usize,
    max_positions: usize,
    attention: AttentionParams,
    layer: LayerParams,
    stack: StackParams,
    head_required: bool,
    eos: BTreeSet<u32>,
    bos: Option<u32>,
    chat: ChatFormat,
}

impl LlamaParams {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        // `attention_bias` biases the output projection too, which no weight role carries yet.
        for unsupported in ["attention_bias", "mlp_bias"] {
            if fields.flag(Field::hf(unsupported), false)? {
                return Err(fields.error(unsupported, ConfigReason::Unsupported));
            }
        }
        let mut attention = dims.attention(&fields, &SPEC, false)?;
        if let Some(window) = fields.opt_u64(Field::hf("sliding_window"))? {
            let window = usize::try_from(window)
                .map_err(|_| fields.error("sliding_window", ConfigReason::WrongType))?;
            attention = attention
                .with_window(window)
                .map_err(|e| e.for_family(FAMILY))?;
        }
        // The `mistral` key is a Mistral checkpoint: its own `[INST]` template, not Llama 3's.
        let chat = match raw {
            RawConfig::HfJson { config, .. }
                if config.get("model_type").and_then(|v| v.as_str()) == Some("mistral") =>
            {
                ChatFormat::Mistral
            }
            RawConfig::HfJson { .. } | RawConfig::Gguf(_) => ChatFormat::Llama3,
        };
        if matches!(raw, RawConfig::Gguf(_)) {
            attention = attention.with_interleaved_rope();
        }
        let norm = dims.norm(&fields, &SPEC)?;
        Ok(Self {
            attention,
            layer: LayerParams::new(norm),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
            chat,
        })
    }
}

#[derive(Debug)]
struct Layer {
    norms: LayerNorms,
    attn: AttentionWeights,
    ffn: GatedFfnWeights,
}

#[derive(Debug)]
pub struct Llama {
    params: LlamaParams,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let mut p = LlamaParams::from_raw(raw)?;
    let rows = match raw.source() {
        ConfigSource::HfModelType => HF_BASE,
        ConfigSource::GgufArchitecture => {
            if let Some(entry) = store.get(GGUF_ROPE_FREQS) {
                p.attention = p
                    .attention
                    .with_rope_freq_factors(&rope_freq_factors(entry)?)
                    .map_err(|e| e.for_family(FAMILY))?;
            }
            GGUF_BASE
        }
    };
    let weights = family_weights(FAMILY, store, &[rows], p.layers, p.head_required)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let stack = StackWeights::new(&weights, p.vocab, p.width).map_err(weight_error)?;
    let layers = (0..p.layers)
        .map(|l| {
            Ok(Layer {
                norms: LayerNorms::new(&weights, l, p.width)?,
                attn: AttentionWeights::new(&weights, l, p.width, &p.attention)?,
                ffn: GatedFfnWeights::new(&weights, l, p.width, p.inter)?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: p.vocab,
        max_positions: p.max_positions,
        eos: p.eos.clone(),
        bos: p.bos,
        prompt_bos: None,
        output: ModelOutput::Logits { vocab: p.vocab },
        prefill_granule: NonZeroUsize::MIN,
        chat: p.chat,
    };
    Ok(Box::new(Llama {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Llama {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(
        &self,
        phase: Phase,
        shape: StepShape,
    ) -> Result<Graph<ValidationOutputs>, TraceError> {
        let p = &self.params;
        check_step(FAMILY, phase, shape, p.max_positions)?;
        let b = Builder::new();
        let step = Step::new(&b, shape);
        let logits = standard_stack(&b, &step, &self.stack, &p.stack, p.layers, |b, l, x| {
            let layer = &self.layers[l];
            standard_layer(
                b,
                x,
                &layer.norms,
                &p.layer,
                |b, h| attention(b, &step, l, h, &layer.attn, &p.attention),
                |b, h| gated_ffn(b, h, &layer.ffn),
            )
        });
        Ok(step.finish(b, logits))
    }
}

/// The fixture's config: a tiny untied llama whose projection widths are multiples of 32 (so a
/// Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "llama",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

fn fixture() -> Fixture {
    let config = fixture_config();
    let shapes = hf_base_shapes(&config, false);
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufValue, write_gguf};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, assert_same_graph, build as built,
        close, close_all, f32_entry, gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store,
        read_back_owned, step_shape, values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn config_with(model_type: &str, window: Option<usize>) -> serde_json::Value {
        let mut config = fixture_config();
        config["model_type"] = json!(model_type);
        if let Some(window) = window {
            config["sliding_window"] = json!(window);
        }
        config
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// The new body against the logits recorded from the legacy tracers at the base commit
    /// `bbd5c3232`, over the fixture, for a window of `window` keys or full attention.
    fn assert_matches_recorded_at(model_type: &str, window: Option<usize>, rec: &Recorded) {
        let fx = fixture();
        let model = model_of(&config_with(model_type, window), &fx.store);
        assert_matches_recorded(&*model, &fx.store, rec);
    }

    /// SC-002: llama's body gives the top-five logits recorded from the legacy prefill and decode
    /// tracers on the fixture (ADR-0101 tiers 2 and 3). Mutation: pass `Post` placement or change
    /// the RoPE theta; red.
    #[test]
    fn llama_matches_the_recorded_legacy_logits() {
        assert_matches_recorded_at(
            "llama",
            None,
            &Recorded {
                prefill: [
                    (47, 2.5254447),
                    (46, 2.19044),
                    (24, 1.8708938),
                    (13, 1.7305588),
                    (20, 1.2541254),
                ],
                decode: [
                    [
                        (47, 2.554118),
                        (1, 1.6575682),
                        (14, 1.5247586),
                        (39, 1.5132793),
                        (29, 1.3596327),
                    ],
                    [
                        (24, 2.7636833),
                        (46, 2.3170958),
                        (45, 1.5511551),
                        (0, 1.2757467),
                        (41, 1.1615623),
                    ],
                    [
                        (35, 2.2088459),
                        (46, 2.1401362),
                        (45, 2.0528457),
                        (34, 1.5905961),
                        (10, 1.3851273),
                    ],
                ],
            },
        );
    }

    /// SC-002: the `mistral` key with a sliding window of 3 keys, which the prompt and the decode
    /// steps exceed, gives the top-five logits recorded from the legacy windowed tracers, and the
    /// window changes the output.
    #[test]
    fn mistral_with_a_window_matches_the_recorded_legacy_logits() {
        assert_matches_recorded_at(
            "mistral",
            Some(3),
            &Recorded {
                prefill: [
                    (39, 2.187981),
                    (47, 2.0849376),
                    (6, 1.6817052),
                    (38, 1.4759992),
                    (32, 1.2051036),
                ],
                decode: [
                    [
                        (47, 2.9925487),
                        (15, 2.0763311),
                        (41, 1.4485627),
                        (36, 1.4396846),
                        (29, 1.3139923),
                    ],
                    [
                        (15, 2.247693),
                        (1, 1.9912384),
                        (25, 1.6233634),
                        (7, 1.4112468),
                        (13, 1.3536422),
                    ],
                    [
                        (20, 2.7654626),
                        (19, 1.9462117),
                        (33, 1.6563671),
                        (25, 1.4697057),
                        (27, 1.467036),
                    ],
                ],
            },
        );
        let fx = fixture();
        let run = |window| {
            let model = model_of(&config_with("mistral", window), &fx.store);
            run_step(
                &*model,
                &fx.store,
                Phase::Prefill,
                CAP,
                &[3, 17, 40, 8, 25],
                0,
                &[],
            )
            .0
        };
        assert_ne!(run(Some(3)), run(None), "the window must change the logits");
    }

    /// SC-001 (spec 999 SC-002): `llama` and `mistral` with no window trace structurally identical
    /// graphs, compared equation by equation with equal names, through one registry key each.
    /// Mutation: add one op to the mistral path outside the standard functions (a scale on the
    /// embedding); red.
    #[test]
    fn llama_and_mistral_without_a_window_trace_identical_graphs() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        let mut graphs = Vec::new();
        for model_type in ["llama", "mistral"] {
            let config = config_with(model_type, None);
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            assert_eq!(
                registry.resolve(&raw).unwrap().family,
                FAMILY,
                "{model_type}"
            );
            let model = registry.build(&raw, &fx.store).unwrap();
            graphs.push(
                model
                    .trace(Phase::Prefill, step_shape(3, CAP, LogitRows::All))
                    .unwrap(),
            );
        }
        assert_same_graph(&graphs[0], &graphs[1]);
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_chunked_prefill_equals_decode(&*model, &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 stores trace through the one body.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
    }

    /// SC-003: a store missing one required tensor fails `Registry::build` with the typed
    /// missing-weight error naming it, before any trace. Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.norm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert_missing_weight_is_named(&registry, &fx, removed);
        }
    }

    fn config_error_of(config: &serde_json::Value) -> (&'static str, ConfigReason) {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        match build(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_llama_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "mlp_bias",
                json!(true),
                "mlp_bias",
                ConfigReason::Unsupported,
            ),
            ("sliding_window", json!(0), "window", ConfigReason::Zero),
            (
                "sliding_window",
                json!("4"),
                "sliding_window",
                ConfigReason::WrongType,
            ),
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
            ),
            (
                "rope_scaling",
                json!({"type": "longrope", "short_factor": [1.0]}),
                "rope_scaling.rope_type",
                ConfigReason::Unsupported,
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
    }

    /// The eos set a config omits is llama's reference default (`2`); the union with the
    /// generation config otherwise.
    #[test]
    fn a_missing_eos_is_the_reference_default() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([2]));
        assert_eq!(model.config().chat, ChatFormat::Llama3);
        let config = config_with("mistral", None);
        assert_eq!(
            model_of(&config, &fx.store).config().chat,
            ChatFormat::Mistral
        );
    }

    /// Un-permute the rows of llama.cpp's Q/K layout for interleaved RoPE: llama.cpp stores,
    /// within each head block of `d` rows, HF row `j` at `2j` and HF row `j + d / 2` at `2j + 1`.
    fn unpermute_rows(rows: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
        let mut out = rows.to_vec();
        for (head, block) in rows.chunks(d).enumerate() {
            for j in 0..d / 2 {
                out[head * d + j] = block[2 * j].clone();
                out[head * d + d / 2 + j] = block[2 * j + 1].clone();
            }
        }
        out
    }

    /// A Q8_0 llama GGUF (Q/K payloads in llama.cpp's permuted order, projections Q8_0, the rest
    /// F32) and the HF F32 store holding the same values un-permuted.
    fn gguf_case() -> (poot_load::gguf::GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
            let shape = entry.shape();
            let decoded = values(entry);
            let is_qk =
                key.as_str().ends_with("q_proj.weight") || key.as_str().ends_with("k_proj.weight");
            let decoded = if is_qk {
                let rows: Vec<Vec<f32>> = decoded.chunks(shape[1]).map(<[f32]>::to_vec).collect();
                unpermute_rows(&rows, 16).concat()
            } else {
                decoded
            };
            reference
                .insert(key.clone(), f32_entry(shape, &decoded))
                .unwrap();
        }
        let kvs = gguf_dense_kvs("llama", &fx.config);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004 (R-562-3): a Q8_0 llama GGUF resolves by its architecture key, builds packed
    /// projections traced as `PackedDequant`, and computes what the HF body computes over the same
    /// values un-permuted: the interleaved rotation over the stored rows is the half-split one over
    /// the HF rows. Mutation: drop the GGUF key (`Unregistered`), or trace the GGUF rows half-split
    /// (the logits diverge).
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body_over_unpermuted_rows() {
        let (index, store, reference_store) = gguf_case();
        let registry = Registry::builtin().unwrap();
        let model = assert_gguf_resolves_and_traces_packed(&registry, &ENTRY, &index, &store, 14);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        let reference = model_of(&fixture_config(), &reference_store);

        let prompt = [3, 17, 40, 8];
        let (got, state) = run_step(&*model, &store, Phase::Prefill, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(
            &*reference,
            &reference_store,
            Phase::Prefill,
            CAP,
            &prompt,
            0,
            &[],
        );
        close(&got, &want);
        // The K cache holds rotated, permuted rows: compare the V cache and the next logits.
        let (got, _) = run_step(&*model, &store, Phase::Decode, CAP, &[5], 4, &state);
        let (want, _) = run_step(
            &*reference,
            &reference_store,
            Phase::Decode,
            CAP,
            &[5],
            4,
            &want_state,
        );
        close(&got, &want);
        close_all(
            &[state[1].clone(), state[3].clone()],
            &[want_state[1].clone(), want_state[3].clone()],
        );
    }

    /// llama.cpp's converter rule for `rope_freqs.weight` (llama3 scaling), in f64 over the fixture's
    /// head dim: the per-frequency divisor of each inverse frequency.
    fn llama3_rope_freqs(factor: f64, low: f64, high: f64, original: f64) -> Vec<f32> {
        let (head_dim, theta) = (16, 10_000.0f64);
        (0..head_dim / 2)
            .map(|j| {
                let wavelen =
                    2.0 * std::f64::consts::PI * theta.powf((2 * j) as f64 / head_dim as f64);
                let divisor = if wavelen < original / high {
                    1.0
                } else if wavelen > original / low {
                    factor
                } else {
                    let smooth = (original / wavelen - low) / (high - low);
                    1.0 / ((1.0 - smooth) / factor + smooth)
                };
                divisor as f32
            })
            .collect()
    }

    fn with_rope_freqs(store: &WeightStore, freqs: &[f32]) -> WeightStore {
        let mut out = WeightStore::builder();
        out.extend_from(store).unwrap();
        out.insert(GGUF_ROPE_FREQS, f32_entry(vec![freqs.len()], freqs))
            .unwrap();
        out.build()
    }

    /// A GGUF carrying llama3's `rope_freqs.weight` rotates by the rescaled frequencies: it equals
    /// the HF body whose config states the same `rope_scaling`, over the same un-permuted values, and
    /// differs from plain RoPE (so the tensor is not ignored). Mutation: skip the tensor in `build`
    /// (the logits equal plain RoPE's, no longer the HF llama3 body's).
    #[test]
    fn a_gguf_with_llama3_rope_freqs_equals_the_hf_llama3_body() {
        let (index, store, reference_store) = gguf_case();
        let freqs = llama3_rope_freqs(8.0, 1.0, 4.0, 64.0);
        assert!(freqs.iter().any(|f| *f > 1.0 && *f < 8.0), "{freqs:?}");
        let store = with_rope_freqs(&store, &freqs);
        let model = build(&RawConfig::Gguf(&index), &store).unwrap();
        let mut config = fixture_config();
        config["rope_scaling"] = json!({
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
        });
        let reference = model_of(&config, &reference_store);
        let plain = model_of(&fixture_config(), &reference_store);

        let prompt = [3, 17, 40, 8, 22, 9, 31, 12];
        let run =
            |m: &dyn Model, s: &WeightStore| run_step(m, s, Phase::Prefill, CAP, &prompt, 0, &[]).0;
        let got = run(&*model, &store);
        close(&got, &run(&*reference, &reference_store));
        assert!(
            got.iter()
                .zip(run(&*plain, &reference_store))
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "llama3 rescaling changed nothing at these positions"
        );
    }

    /// A `rope_freqs.weight` that is not a llama3 rescale of the head's frequencies, or not a
    /// per-frequency F32 vector, is refused naming the tensor, never approximated.
    #[test]
    fn a_gguf_rope_freqs_that_is_not_a_llama3_rescale_is_refused_naming_the_tensor() {
        let (index, store, _) = gguf_case();
        let refused = |store: &WeightStore| match build(&RawConfig::Gguf(&index), store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{other:?}"),
        };
        // One blended frequency pins no slope; a descending ramp is no llama3 shape.
        let one_blend = [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 8.0, 8.0];
        let descending = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5];
        for freqs in [&one_blend[..], &descending[..]] {
            assert_eq!(
                refused(&with_rope_freqs(&store, freqs)),
                (GGUF_ROPE_FREQS, ConfigReason::Unsupported),
                "{freqs:?}"
            );
        }
        // Eight frequencies are expected for a 16-wide head.
        assert_eq!(
            refused(&with_rope_freqs(&store, &[1.0; 4])),
            (GGUF_ROPE_FREQS, ConfigReason::WrongType)
        );
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = poot_load::gguf::GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("llama".into())),
                ("llama.embedding_length", GgufValue::U32(64)),
                ("llama.feed_forward_length", GgufValue::U32(96)),
                ("llama.block_count", GgufValue::U32(2)),
                ("llama.attention.head_count", GgufValue::U32(0)),
                ("llama.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = LlamaParams::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "llama.attention.head_count",
                    reason: ConfigReason::Zero,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_shape_the_body_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        let n = |v| NonZeroUsize::new(v).unwrap();
        let shape = |rows, tokens, capacity| StepShape {
            rows: n(rows),
            tokens: n(tokens),
            capacity: n(capacity),
            ..step_shape(1, 1, LogitRows::Last)
        };
        assert!(matches!(
            model.trace(Phase::Decode, shape(1, 2, 8)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(Phase::Prefill, shape(1, 1, 65)),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }
}
