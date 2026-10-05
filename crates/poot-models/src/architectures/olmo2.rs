//! OLMo 2 behind [`Model`]: the post-norm block (`x += norm(mixer(x))`, no input norms) with an
//! RMSNorm over the whole Q and K projections before RoPE (`QkNorm::Projection`).
//!
//! The body is `standard_stack` over `standard_layer` with `NormPlacement::Post`: the layer's two
//! norm weights are the post-attention and post-feed-forward norms.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, Graph, ValidationOutputs};
use poot_quant::weights::{NormRole, WeightMap, WeightRole, WeightStore};

use crate::chat::ChatFormat;
use crate::components::attention::{AttentionParams, AttentionWeights, QkNorm, attention};
use crate::components::dims::{DenseDims, DenseSpec, Field, Fields, dense_keys};
use crate::components::ffn::{GatedFfnWeights, gated_ffn};
use crate::components::linear::WeightError;
use crate::components::standard::{
    LayerNorms, LayerParams, NormPlacement, StackParams, StackWeights, Step, check_step,
    standard_layer, standard_stack,
};
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, StepShape,
    TraceError,
};
use crate::names::gguf::{GGUF_BASE, GGUF_POST_NORMS, GGUF_QK_NORM};
use crate::names::{
    HF_BASE, HF_QK_NORM, NameRow, Presence, family_weights, hf_base_shapes, layer_tensor,
};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};

pub const FAMILY: FamilyKey = FamilyKey::new("olmo2");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[
        (ConfigSource::HfModelType, "olmo2"),
        (ConfigSource::GgufArchitecture, "olmo2"),
    ],
    build,
    fixture,
};

/// OLMo 2's config facts: `Olmo2Config`'s defaults (its `eos_token_id` is `50279`) and llama.cpp's
/// GGUF defaults.
const SPEC: DenseSpec = DenseSpec {
    family: FAMILY,
    gguf: dense_keys!("olmo2"),
    hf_theta: 10_000.0,
    gguf_theta: 10_000.0,
    eps: 1e-6,
    eos: Some(50279),
};

/// OLMo 2's HF norms: the layer's two norm roles read the norms of each block's output.
const HF_POST_NORMS: &[NameRow] = &[
    layer_tensor(
        WeightRole::Norm(NormRole::Attn),
        ".post_attention_layernorm.weight",
        Presence::Required,
    ),
    layer_tensor(
        WeightRole::Norm(NormRole::Ffn),
        ".post_feedforward_layernorm.weight",
        Presence::Required,
    ),
];

/// OLMo 2's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Olmo2Params {
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
}

impl Olmo2Params {
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let fields = Fields::new(raw, FAMILY);
        let dims = DenseDims::read(&fields, &SPEC)?;
        // `attention_bias` biases the output projection too, which no weight role carries yet.
        if fields.flag(Field::hf("attention_bias"), false)? {
            return Err(fields.error("attention_bias", ConfigReason::Unsupported));
        }
        let norm = dims.norm(&fields, &SPEC)?;
        Ok(Self {
            attention: dims
                .attention(&fields, &SPEC, false)?
                .with_qk_norm(QkNorm::Projection(norm)),
            layer: LayerParams::new(norm).with_placement(NormPlacement::Post),
            stack: StackParams::new(norm),
            vocab: dims.vocab,
            width: dims.width,
            inter: dims.inter,
            layers: dims.layers,
            max_positions: dims.max_positions,
            head_required: dims.head_required,
            eos: dims.eos,
            bos: dims.bos,
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
pub struct Olmo2 {
    params: Olmo2Params,
    weights: WeightMap,
    stack: StackWeights,
    layers: Vec<Layer>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Olmo2Params::from_raw(raw)?;
    let tables: [&[NameRow]; 3] = match raw.source() {
        ConfigSource::HfModelType => [HF_BASE, HF_QK_NORM, HF_POST_NORMS],
        ConfigSource::GgufArchitecture => [GGUF_BASE, GGUF_QK_NORM, GGUF_POST_NORMS],
    };
    let weights = family_weights(FAMILY, store, &tables, p.layers, p.head_required)?;
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
        chat: ChatFormat::Olmo2,
    };
    Ok(Box::new(Olmo2 {
        params: p,
        weights,
        stack,
        layers,
        config,
    }))
}

impl Model for Olmo2 {
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

/// The fixture's config: a tiny untied OLMo 2 whose projection widths are multiples of 32 (so a
/// Q8_0 copy of its store is a valid checkpoint too).
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "olmo2",
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
    let n = |f: &str| config[f].as_u64().unwrap() as usize;
    let d = n("hidden_size") / n("num_attention_heads");
    let (q, kv) = (n("num_attention_heads") * d, n("num_key_value_heads") * d);
    let mut shapes: Vec<_> = hf_base_shapes(&config, false)
        .into_iter()
        .filter(|(key, _)| !key.ends_with("input_layernorm.weight"))
        .collect();
    for l in 0..n("num_hidden_layers") {
        shapes.push((
            format!("model.layers.{l}.post_feedforward_layernorm.weight"),
            vec![n("hidden_size")],
        ));
        shapes.push((format!("model.layers.{l}.self_attn.q_norm.weight"), vec![q]));
        shapes.push((
            format!("model.layers.{l}.self_attn.k_norm.weight"),
            vec![kv],
        ));
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, build as built, close, f32_entry,
        gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store, read_back_owned, step_shape,
        values,
    };
    use crate::model::LogitRows;
    use crate::registry::Registry;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-002: OLMo 2's body gives the top-five logits recorded from the legacy olmo2 tracers at
    /// the base commit `bbd5c3232` (token ids exact, logits tier 2). Mutation: flip
    /// `NormPlacement::Post` to `Pre` in the params; red.
    #[test]
    fn olmo2_matches_the_recorded_legacy_logits() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (35, 1.9959298),
                    (40, 1.8527719),
                    (27, 1.6218243),
                    (30, 1.5931289),
                    (36, 1.4598099),
                ],
                decode: [
                    [
                        (35, 2.0894735),
                        (38, 1.8781209),
                        (9, 1.6058707),
                        (36, 1.4470187),
                        (40, 1.2422943),
                    ],
                    [
                        (27, 2.41425),
                        (26, 1.8934437),
                        (46, 1.567908),
                        (35, 1.0317863),
                        (34, 0.9928058),
                    ],
                    [
                        (40, 1.7016371),
                        (13, 1.3301094),
                        (7, 1.2650198),
                        (35, 1.1228518),
                        (9, 1.1095176),
                    ],
                ],
            },
        );
    }

    #[test]
    fn chunked_prefill_continuing_from_pos_equals_token_by_token_decode() {
        let fx = fixture();
        assert_chunked_prefill_equals_decode(&*model_of(&fx.config, &fx.store), &fx.store);
    }

    /// SC-004 (HF half): BF16 and Q8_0 stores trace through the one body.
    #[test]
    fn bf16_and_q8_0_stores_trace_through_the_one_body_and_the_packed_transform() {
        assert_bf16_and_q8_0_trace_through_the_packed_transform(&ENTRY, &fixture(), is_projection);
    }

    /// SC-003: a store missing one required tensor (a post norm, a Q/K norm) fails
    /// `Registry::build` with the typed missing-weight error naming it, before any trace.
    /// Mutation: skip the row check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
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
    fn a_config_olmo2_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
            ),
            (
                "rms_norm_eps",
                json!(-1.0),
                "rms_norm_eps",
                ConfigReason::NotFinitePositive,
            ),
            (
                "hidden_size",
                json!("64"),
                "hidden_size",
                ConfigReason::WrongType,
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

    /// The eos a config omits is OLMo 2's reference default (`50279`).
    #[test]
    fn a_missing_eos_is_the_reference_default() {
        let fx = fixture();
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        let model = model_of(&config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([50279]));
        assert_eq!(model.config().chat, ChatFormat::Olmo2);
    }

    /// llama.cpp's GGUF name of an OLMo 2 HF tensor.
    fn olmo2_gguf_name(hf: &str) -> String {
        if let Some(rest) = hf.strip_prefix("model.layers.") {
            let (layer, tensor) = rest.split_once('.').unwrap();
            let renamed = match tensor {
                "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                "post_feedforward_layernorm.weight" => Some("post_ffw_norm.weight"),
                _ => None,
            };
            if let Some(renamed) = renamed {
                return format!("blk.{layer}.{renamed}");
            }
        }
        gguf_name_with_head(hf)
    }

    /// A Q8_0 OLMo 2 GGUF (projections Q8_0, norms and the rest F32, untied head) and the HF F32
    /// store with the same values.
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&olmo2_gguf_name(key.as_str()), entry));
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let kvs = gguf_dense_kvs("olmo2", &fx.config);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004: a Q8_0 OLMo 2 GGUF resolves by its architecture key, builds packed projections traced
    /// as `PackedDequant`, and computes what the HF body computes over the same values. Mutation:
    /// drop the GGUF key from `ENTRY.keys`; `Unregistered`.
    #[test]
    fn a_q8_0_gguf_resolves_by_architecture_and_equals_the_hf_body() {
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
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("olmo2".into())),
                ("olmo2.embedding_length", GgufValue::U32(64)),
                ("olmo2.feed_forward_length", GgufValue::U32(96)),
                ("olmo2.block_count", GgufValue::U32(2)),
                ("olmo2.attention.head_count", GgufValue::U32(0)),
                ("olmo2.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        ))
        .unwrap();
        let err = Olmo2Params::from_raw(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "olmo2.attention.head_count",
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
