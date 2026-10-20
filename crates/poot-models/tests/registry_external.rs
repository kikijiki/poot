//! Card 562a SC-004: a family defined outside the production crates joins the registry through
//! `Registry::register` alone, and resolves by its config key exactly like a shipped family. An
//! integration test is its own crate, so nothing here can reach a `pub(crate)` item.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph, TensorType};
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec};
use poot_models::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, StepShape, StepSpec,
    TraceError,
};
use poot_models::registry::{
    ConfigSource, FamilyEntry, Fixture, LoadModelError, ModelParts, RawConfig, Registry,
    RegistryError,
};
use poot_quant::weights::{
    DenseWeight, WeightEntry, WeightId, WeightMap, WeightRole, WeightStore, WeightView,
};
use poot_tensor::DType;
use poot_text::{ChatFormat, HermesToolCodec};

const FAKE: FamilyKey = FamilyKey::new("fake-lm");

/// A one-weight "model": its trace reads the embedding table and returns it.
#[derive(Debug)]
struct FakeLm {
    config: ModelConfig,
    schema: ModelSchema,
    weights: WeightMap,
    width: usize,
}

impl Model for FakeLm {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, _spec: &StepSpec, b: Builder) -> Result<Graph, TraceError> {
        let embed = b.constant(
            &WeightId::model(WeightRole::Embed).const_name(),
            TensorType::f32(vec![self.config.vocab, self.width]),
        )?;
        Ok(b.finish(embed)?)
    }
}

fn usize_field(raw: &RawConfig<'_>, field: &'static str) -> Result<usize, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        return Err(ModelError::Config {
            family: FAKE,
            field,
            reason: ConfigReason::Unsupported,
        });
    };
    config
        .get(field)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or(ModelError::Config {
            family: FAKE,
            field,
            reason: ConfigReason::Missing,
        })
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let vocab = usize_field(raw, "vocab_size")?;
    let width = usize_field(raw, "hidden_size")?;
    let mut weights = WeightMap::builder(store);
    weights
        .map(
            WeightId::model(WeightRole::Embed),
            WeightView::Stored("embed.weight".into()),
        )
        .map_err(|source| ModelError::Weight {
            family: FAKE,
            source,
        })?;
    let eos = BTreeSet::from([usize_field(raw, "eos_token_id")? as u32]);
    Ok(ModelParts::new(Box::new(FakeLm {
        config: ModelConfig {
            family: FAKE,
            vocab,
            max_positions: 16,
            eos,
            bos: None,
            prompt_bos: None,
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        },
        schema: ModelSchema::builder()
            .step(ProgramSpec::new(
                ProgramId::STEP,
                ModelOutput::Logits { vocab },
            ))
            .build_for(FAKE)
            .expect("a stateless step program is a schema"),
        weights: weights.build(),
        width,
    })))
}

fn fixture() -> Fixture {
    let (vocab, width) = (3, 2);
    let bytes = poot_quant::StoredBytes::from(vec![0u8; vocab * width * 4]);
    let mut store = WeightStore::builder();
    store
        .insert(
            "embed.weight",
            WeightEntry::Dense(
                DenseWeight::try_new(DType::F32, vec![vocab, width], bytes).unwrap(),
            ),
        )
        .unwrap();
    Fixture {
        config: serde_json::json!({
            "model_type": "fake-lm",
            "vocab_size": vocab,
            "hidden_size": width,
            "eos_token_id": 2
        }),
        generation: None,
        preprocessor: None,
        store: store.build(),
    }
}

const ENTRY: FamilyEntry = FamilyEntry {
    family: FAKE,
    keys: &[
        (ConfigSource::HfModelType, "fake-lm"),
        (ConfigSource::GgufArchitecture, "fakelm"),
    ],
    build,
    fixture,
    variants: &[],
};

fn decode_shape() -> StepShape {
    StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        NonZeroUsize::new(8).unwrap(),
    )
}

#[test]
fn an_externally_registered_family_resolves_by_model_type_and_builds() {
    let mut registry = Registry::empty();
    registry.register(ENTRY).expect("a new key registers");

    let fx = (ENTRY.fixture)();
    let raw = fx.raw();
    assert_eq!(
        registry.resolve(&raw).expect("fake-lm resolves").family,
        FAKE
    );

    let model = registry
        .build(&raw, &fx.store)
        .expect("fake-lm builds")
        .model;
    assert_eq!(model.config().family, FAKE);
    assert_eq!(model.config().vocab, 3);
    assert_eq!(model.config().eos, BTreeSet::from([2]));
    let embed = WeightId::model(WeightRole::Embed);
    assert_eq!(model.weights().handle(embed).unwrap().shape, vec![3, 2]);

    let graph = model
        .trace(
            &StepSpec::step(decode_shape()),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let names: Vec<&str> = graph
        .consts()
        .iter()
        .filter_map(|&id| graph.values()[id].name.as_deref())
        .collect();
    assert_eq!(names, ["w.embed"]);
}

#[test]
fn an_unregistered_model_type_is_a_typed_refusal_naming_the_missing_registration() {
    let mut registry = Registry::empty();
    registry.register(ENTRY).unwrap();
    let config = serde_json::json!({ "model_type": "nobody-registered-this" });
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    };
    let err = registry.resolve(&raw).unwrap_err();
    assert_eq!(
        err,
        RegistryError::Unregistered {
            origin: ConfigSource::HfModelType,
            key: "nobody-registered-this".into(),
        }
    );
    let message = err.to_string();
    assert!(
        message.contains("\"nobody-registered-this\"") && message.contains("Registry::register"),
        "{message}"
    );
    let store = WeightStore::default();
    assert!(matches!(
        registry.build(&raw, &store),
        Err(LoadModelError::Registry(RegistryError::Unregistered { .. }))
    ));

    let keyless = serde_json::json!({ "vocab_size": 3 });
    let raw = RawConfig::HfJson {
        config: &keyless,
        generation: None,
        preprocessor: None,
    };
    assert_eq!(
        registry.resolve(&raw).unwrap_err(),
        RegistryError::NoKey(ConfigSource::HfModelType)
    );
}

#[test]
fn a_family_whose_weights_are_missing_is_a_typed_model_error() {
    let mut registry = Registry::empty();
    registry.register(ENTRY).unwrap();
    let fx = (ENTRY.fixture)();
    let err = registry
        .build(&fx.raw(), &WeightStore::default())
        .unwrap_err();
    assert!(
        matches!(
            err,
            LoadModelError::Model(ModelError::Weight { family: FAKE, .. })
        ),
        "{err}"
    );
    assert!(err.to_string().contains("embed.weight"), "{err}");
}
