//! qwen3: the qwen2 body without QKV biases, with a per-head RMSNorm on Q and K before RoPE
//! (`QkNorm::PerHead`) and an explicit `head_dim` that may exceed `width / heads`.
//!
//! A dense family is its keys, its config defaults and one attention option ([`DenseFamily`]); the
//! model is the shared [`super::standard_family::StandardModel`].

use crate::components::dims::{DenseSpec, dense_keys};
use crate::model::FamilyKey;
use crate::names::gguf::{GGUF_BASE, GGUF_QK_NORM};
use crate::names::{HF_BASE, HF_QK_NORM, NameRow};
use crate::registry::{ConfigSource, FamilyEntry};
#[cfg(feature = "test-support")]
use crate::{names::hf_base_shapes, registry::Fixture};
use std::sync::Arc;

use poot_text::{ChatFormat, HermesToolCodec, ToolCallCodec};

use crate::components::standard_family::{DenseFamily, QkNormScope, entry};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("qwen3");

/// qwen3: `Qwen3Config`'s defaults (the class declares no `eos_token_id`: a checkpoint names it) and
/// llama.cpp's GGUF base default; the qwen2 body without QKV biases, with a per-head RMSNorm on Q
/// and K before RoPE and an explicit `head_dim` that may exceed `width / heads`.
pub(crate) struct Qwen3;

impl DenseFamily for Qwen3 {
    const FAMILY: FamilyKey = FAMILY;
    const KEYS: &'static [(ConfigSource, &'static str)] = &[
        (ConfigSource::HfModelType, "qwen3"),
        (ConfigSource::GgufArchitecture, "qwen3"),
    ];
    const SPEC: DenseSpec = DenseSpec {
        family: FAMILY,
        gguf: dense_keys!("qwen3"),
        hf_theta: 10_000.0,
        gguf_theta: Some(1_000_000.0),
        eps: 1e-6,
        eos: None,
    };
    const CHAT: ChatFormat = ChatFormat::ChatML;

    fn tool_codec() -> Option<Arc<dyn ToolCallCodec>> {
        Some(Arc::new(HermesToolCodec))
    }
    // `attention_bias` biases the output projection too, which no weight role carries yet.
    const UNSUPPORTED: &'static [&'static str] = &["attention_bias", "use_sliding_window"];
    const QK_NORM: Option<QkNormScope> = Some(QkNormScope::PerHead);
    const HF_TABLES: &'static [&'static [NameRow]] = &[HF_BASE, HF_QK_NORM];
    const GGUF_TABLES: &'static [&'static [NameRow]] = &[GGUF_BASE, GGUF_QK_NORM];
}

pub(crate) const ENTRY: FamilyEntry = entry::<Qwen3>();

/// The fixture's config: a tiny untied qwen3 with `head_dim` 32 (above `64 / 4`), projection widths
/// that are multiples of 32 (so a Q8_0 copy of its store is a valid checkpoint too).
#[cfg(feature = "test-support")]
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3",
        "vocab_size": 48,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": 47,
        "bos_token_id": 1,
    })
}

#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    let config = fixture_config();
    let mut shapes = hf_base_shapes(&config, false);
    let d = config["head_dim"].as_u64().unwrap() as usize;
    for l in 0..config["num_hidden_layers"].as_u64().unwrap() {
        shapes.push((format!("model.layers.{l}.self_attn.q_norm.weight"), vec![d]));
        shapes.push((format!("model.layers.{l}.self_attn.k_norm.weight"), vec![d]));
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use poot_quant::StoredBytes;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::run_step;
    use crate::components::standard_family::StandardFamily;
    use crate::components::testing::{
        CAP, Recorded, assert_bf16_and_q8_0_trace_through_the_packed_transform,
        assert_chunked_prefill_equals_decode, assert_gguf_resolves_and_traces_packed,
        assert_matches_recorded, assert_missing_weight_is_named, build as built, close, f32_entry,
        gguf_dense_kvs, gguf_name_with_head, gguf_tensor, q8_0_store, read_back_owned, values,
    };
    use crate::model::ConfigReason;
    use crate::model::Model;
    use crate::model::ModelError;
    use crate::model::StepShape;
    use crate::model::StepSpec;
    use crate::model::TraceError;
    use crate::registry::RawConfig;
    use crate::registry::Registry;
    use poot_quant::weights::WeightStore;
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    fn is_projection(key: &str) -> bool {
        key.ends_with("_proj.weight")
    }

    fn model_of(config: &serde_json::Value, store: &WeightStore) -> Box<dyn Model> {
        let raw = RawConfig::HfJson {
            config,
            generation: None,
            preprocessor: None,
        };
        built(&ENTRY, &raw, store)
    }

    /// SC-002: qwen3's body gives the top-five logits recorded from the legacy qk-norm tracers at
    /// the base commit `bbd5c3232` (token ids exact, logits tier 2), at the last prefill position
    /// and three decode steps. Mutation: drop the per-head Q/K norm (`with_qk_norm`) from the
    /// params; red.
    #[test]
    fn qwen3_matches_the_recorded_legacy_logits() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_matches_recorded(
            &*model,
            &fx.store,
            &Recorded {
                prefill: [
                    (12, 2.5361822),
                    (15, 2.3695672),
                    (21, 2.1812763),
                    (16, 1.7242117),
                    (9, 1.6488346),
                ],
                decode: [
                    [
                        (15, 2.5602214),
                        (10, 2.3774328),
                        (12, 1.776539),
                        (36, 1.6106731),
                        (16, 1.3364875),
                    ],
                    [
                        (15, 2.5237987),
                        (9, 1.8627532),
                        (10, 1.6837146),
                        (21, 1.6494192),
                        (36, 1.3549384),
                    ],
                    [
                        (40, 2.2254627),
                        (15, 2.1837013),
                        (16, 1.7121159),
                        (12, 1.6059074),
                        (42, 1.4986235),
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

    /// SC-003: a store missing one required tensor, a Q/K norm included, fails `Registry::build`
    /// with the typed missing-weight error naming it, before any trace. Mutation: skip the row
    /// check; red.
    #[test]
    fn a_missing_required_tensor_is_named_before_any_trace() {
        let registry = Registry::builtin().unwrap();
        let fx = fixture();
        for removed in [
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.1.self_attn.k_norm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
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
            preprocessor: None,
        };
        match (ENTRY.build)(&raw, &fixture().store) {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("{config}: expected a config error, got {other:?}"),
        }
    }

    #[test]
    fn a_config_qwen3_cannot_trace_is_a_typed_error_naming_the_field() {
        let cases: &[(&str, serde_json::Value, &str, ConfigReason)] = &[
            (
                "attention_bias",
                json!(true),
                "attention_bias",
                ConfigReason::Unsupported,
            ),
            (
                "use_sliding_window",
                json!(true),
                "use_sliding_window",
                ConfigReason::Unsupported,
            ),
            ("head_dim", json!(0), "head_dim", ConfigReason::Zero),
            (
                "head_dim",
                json!(31),
                "rotary_dim",
                ConfigReason::NotDivisible { by: 2 },
            ),
            (
                "num_attention_heads",
                json!(0),
                "num_attention_heads",
                ConfigReason::Zero,
            ),
            (
                "rms_norm_eps",
                json!(0.0),
                "rms_norm_eps",
                ConfigReason::NotFinitePositive,
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

    /// A qwen3 config declares its eos: absent, the set is empty (the class has no default).
    #[test]
    fn eos_comes_from_the_config_only() {
        let fx = fixture();
        let model = model_of(&fx.config, &fx.store);
        assert_eq!(model.config().eos, BTreeSet::from([47]));
        let mut config = fixture_config();
        config.as_object_mut().unwrap().remove("eos_token_id");
        assert!(model_of(&config, &fx.store).config().eos.is_empty());
    }

    /// A Q8_0 qwen3 GGUF (projections Q8_0, norms and the rest F32, untied head).
    fn gguf_case() -> (GgufIndex, WeightStore, WeightStore) {
        let fx = fixture();
        let q8 = q8_0_store(&fx.store, is_projection);
        let mut tensors = Vec::new();
        let mut reference = WeightStore::builder();
        for (key, entry) in q8.iter() {
            tensors.push(gguf_tensor(&gguf_name_with_head(key.as_str()), entry));
            reference
                .insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
                .unwrap();
        }
        let kvs = gguf_dense_kvs("qwen3", &fx.config);
        let (index, store) = read_back_owned(&kvs, &tensors);
        (index, store, reference.build())
    }

    /// SC-004: a Q8_0 qwen3 GGUF resolves by its architecture key, builds packed projections traced
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
        let (got, state) = run_step(&*model, &store, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(&*reference, &reference_store, CAP, &prompt, 0, &[]);
        close(&got, &want);
        let (got, _) = run_step(&*model, &store, CAP, &[5], 4, &state);
        let (want, _) = run_step(&*reference, &reference_store, CAP, &[5], 4, &want_state);
        close(&got, &want);
    }

    #[test]
    fn a_malformed_gguf_config_is_a_typed_error_naming_its_key() {
        let index = GgufIndex::parse(&StoredBytes::from(write_gguf(
            &[
                ("general.architecture", GgufValue::Str("qwen3".into())),
                ("qwen3.embedding_length", GgufValue::U32(64)),
                ("qwen3.feed_forward_length", GgufValue::U32(96)),
                ("qwen3.block_count", GgufValue::U32(2)),
                ("qwen3.attention.head_count", GgufValue::U32(0)),
                ("qwen3.context_length", GgufValue::U32(64)),
                (
                    "tokenizer.ggml.tokens",
                    GgufValue::Array(vec![GgufValue::Str("a".into())]),
                ),
            ],
            &[],
        )))
        .unwrap();
        let err = Qwen3::configure(&RawConfig::Gguf(&index)).unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "qwen3.attention.head_count",
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
        let shape = |rows, tokens, capacity| StepShape::new(n(rows), n(tokens), n(capacity));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(1, 9, 8)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
        assert!(matches!(
            model.trace(
                &StepSpec::step(shape(1, 1, 65)),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported { .. })
        ));
    }
}
