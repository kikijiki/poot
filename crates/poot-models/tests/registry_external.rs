//! Card 562a SC-004: a family defined outside the production crates joins the registry through
//! `Registry::register` alone, and resolves by its config key exactly like a shipped family. An
//! integration test is its own crate, so nothing here can reach a `pub(crate)` item.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Builder, Graph, TensorType, ValidationOutputs};
use poot_models::chat::ChatFormat;
use poot_models::model::{
    ConfigReason, FamilyKey, KvLayout, LogitRows, Model, ModelConfig, ModelError, ModelOutput,
    Phase, StepShape, TraceError,
};
use poot_models::registry::{
    ConfigSource, FamilyEntry, Fixture, LoadModelError, RawConfig, Registry, RegistryError,
};
use poot_quant::weights::{
    DenseWeight, WeightEntry, WeightId, WeightMap, WeightRole, WeightStore, WeightView,
};
use poot_tensor::DType;

const FAKE: FamilyKey = FamilyKey::new("fake-lm");

/// A one-weight "model": its trace reads the embedding table and returns it.
#[derive(Debug)]
struct FakeLm {
    config: ModelConfig,
    weights: WeightMap,
    width: usize,
}

impl Model for FakeLm {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace(
        &self,
        _phase: Phase,
        _shape: StepShape,
    ) -> Result<Graph<ValidationOutputs>, TraceError> {
        let b = Builder::new();
        let embed = b.constant(
            &WeightId::model(WeightRole::Embed).const_name(),
            TensorType::f32(vec![self.config.vocab, self.width]),
        );
        Ok(b.finish(embed).with_validations(Vec::new()))
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

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
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
    Ok(Box::new(FakeLm {
        config: ModelConfig {
            family: FAKE,
            vocab,
            max_positions: 16,
            eos,
            bos: None,
            prompt_bos: None,
            output: ModelOutput::Logits { vocab },
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
        },
        weights: weights.build(),
        width,
    }))
}

fn fixture() -> Fixture {
    let (vocab, width) = (3, 2);
    let bytes: Arc<[u8]> = vec![0u8; vocab * width * 4].into();
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
};

fn decode_shape() -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::MIN,
        capacity: NonZeroUsize::new(8).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::Last,
    }
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

    let model = registry.build(&raw, &fx.store).expect("fake-lm builds");
    assert_eq!(model.config().family, FAKE);
    assert_eq!(model.config().vocab, 3);
    assert_eq!(model.config().eos, BTreeSet::from([2]));
    let embed = WeightId::model(WeightRole::Embed);
    assert_eq!(model.weights().handle(embed).unwrap().shape, vec![3, 2]);

    let graph = model.trace(Phase::Decode, decode_shape()).unwrap();
    let names: Vec<&str> = graph
        .consts
        .iter()
        .filter_map(|&id| graph.values[id].name.as_deref())
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
