//! The one family registry (ADR-0104): a checkpoint's config names its family, and the
//! family's [`FamilyEntry`] builds its [`ModelParts`]: the served [`Model`] and, for a vision-language
//! family, its towers and front end (ADR-0117 d6). Families are data: [`SHIPPED`] is the one list of
//! them, a const slice of entries (a family's keys and how to build it, nothing else), and
//! [`Registry::builtin`] indexes it. A family is added by writing its file under `architectures` and
//! adding its row to [`SHIPPED`]. There is no out-of-tree registration: a registry is the shipped
//! families, and only a test builds a different one. The driver and the loader take a `&Registry`;
//! there is no global registry.

use crate::architectures::{
    bloom, deepseek2, deepseek3, deepseek4, deepseek32, gemma2, gemma3, gpt_oss, granite, llama,
    minimax_m2, mpt, olmo2, olmoe, phi3, qwen3, qwen3moe, qwen38, qwen38_27b, smollm3,
};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

use poot_load::gguf::GgufIndex;
use poot_quant::weights::WeightStore;

use crate::model::{ConfigReason, FamilyKey, Model, ModelError};
use crate::multimodal::front_end::{FrontEnd, TowerId};

/// A checkpoint's raw config, as a family parses it.
#[derive(Clone, Copy)]
pub enum RawConfig<'a> {
    /// An HF checkpoint: `config.json`, and `generation_config.json` when the checkpoint ships one.
    HfJson {
        /// The parsed `config.json`.
        config: &'a serde_json::Value,
        /// The parsed `generation_config.json`, when present.
        generation: Option<&'a serde_json::Value>,
        /// The parsed `preprocessor_config.json`, when the checkpoint ships one: the image processor
        /// of a vision-language family.
        preprocessor: Option<&'a serde_json::Value>,
    },
    /// A GGUF checkpoint's metadata.
    Gguf(&'a GgufIndex),
}

/// Which config key names a checkpoint's family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConfigSource {
    /// `config.json`'s `model_type`.
    HfModelType,
    /// GGUF's `general.architecture`.
    GgufArchitecture,
}

impl fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ConfigSource::HfModelType => "HF model_type",
            ConfigSource::GgufArchitecture => "GGUF general.architecture",
        })
    }
}

impl fmt::Debug for RawConfig<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("RawConfig");
        s.field("source", &self.source());
        if let RawConfig::HfJson {
            config,
            generation,
            preprocessor,
        } = self
        {
            s.field("config", config)
                .field("generation", generation)
                .field("preprocessor", preprocessor);
        }
        s.finish_non_exhaustive()
    }
}

impl RawConfig<'_> {
    /// Which config format this is, and so which key names the family.
    pub fn source(&self) -> ConfigSource {
        match self {
            RawConfig::HfJson { .. } => ConfigSource::HfModelType,
            RawConfig::Gguf(_) => ConfigSource::GgufArchitecture,
        }
    }

    /// The family key the config names: `model_type`, or `general.architecture`.
    pub fn family_key(&self) -> Result<&str, RegistryError> {
        let key = match self {
            RawConfig::HfJson { config, .. } => config.get("model_type").and_then(|v| v.as_str()),
            RawConfig::Gguf(gguf) => gguf.get("general.architecture").and_then(|v| v.as_str()),
        };
        key.ok_or(RegistryError::NoKey(self.source()))
    }

    /// Every end-of-sequence token id the checkpoint declares. For an HF checkpoint this is the union
    /// of `config.json`'s and `generation_config.json`'s `eos_token_id` (each one id or a list): HF
    /// generation stops on the generation config's set, which often adds ids `config.json` does not
    /// name. For GGUF it is `tokenizer.ggml.eos_token_id`. A declared value that is not a token id is
    /// a typed config error for `family`.
    pub(crate) fn eos_token_ids(&self, family: FamilyKey) -> Result<BTreeSet<u32>, ModelError> {
        let wrong_type = |field| ModelError::Config {
            family,
            field,
            reason: ConfigReason::WrongType,
        };
        let mut eos = BTreeSet::new();
        match self {
            RawConfig::HfJson {
                config, generation, ..
            } => {
                let files = [
                    ("config.json eos_token_id", Some(*config)),
                    ("generation_config.json eos_token_id", *generation),
                ];
                for (field, json) in files {
                    let Some(value) = json.and_then(|j| j.get("eos_token_id")) else {
                        continue;
                    };
                    let ids: Vec<&serde_json::Value> = match value {
                        serde_json::Value::Array(ids) => ids.iter().collect(),
                        serde_json::Value::Null => Vec::new(),
                        id => vec![id],
                    };
                    for id in ids {
                        let id = id
                            .as_u64()
                            .and_then(|id| u32::try_from(id).ok())
                            .ok_or_else(|| wrong_type(field))?;
                        eos.insert(id);
                    }
                }
            }
            RawConfig::Gguf(gguf) => {
                if let Some(value) = gguf.get("tokenizer.ggml.eos_token_id") {
                    let id = value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| wrong_type("tokenizer.ggml.eos_token_id"))?;
                    eos.insert(id);
                }
            }
        }
        Ok(eos)
    }
}

/// A synthetic checkpoint of a family, for tests: it owns its raw config, because [`RawConfig`]
/// borrows. The table of them, keyed by family, is
/// [`crate::test_support::family_fixtures`]: a family's entry carries none.
#[cfg(feature = "test-support")]
#[derive(Clone, Debug)]
pub struct Fixture {
    /// The HF `config.json` the family parses.
    pub config: serde_json::Value,
    /// The `generation_config.json`, when the fixture ships one.
    pub generation: Option<serde_json::Value>,
    /// The `preprocessor_config.json`, when the fixture ships one.
    pub preprocessor: Option<serde_json::Value>,
    /// The synthetic weights, shaped from `config`.
    pub store: WeightStore,
}

#[cfg(feature = "test-support")]
impl Fixture {
    /// The count `field` of a fixture's literal config. A fixture derives its tensor shapes from the
    /// config it builds beside them, so a missing or non-integer field is an inconsistent literal in
    /// the fixture's own source: a defect of the test data, never an input.
    #[track_caller]
    pub(crate) fn count(config: &serde_json::Value, field: &str) -> usize {
        match config[field].as_u64().map(usize::try_from) {
            Some(Ok(count)) => count,
            _ => panic!("fixture config field {field:?} is not a count"),
        }
    }

    /// A tiny deterministic BF16 checkpoint of `config` holding one tensor per `(key, shape)`:
    /// norm scales (keys containing `norm`) near 1, everything else small and centred, so a greedy
    /// decode does not echo its input. The same seed every call.
    pub fn bf16(
        config: serde_json::Value,
        tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
    ) -> Self {
        let mut store = WeightStore::builder();
        let mut seed = 0x9e37_79b9u32;
        for (key, shape) in tensors {
            let n: usize = shape.iter().product();
            let is_norm = key.contains("norm");
            let mut bytes = Vec::with_capacity(2 * n);
            for _ in 0..n {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let u = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                let v = if is_norm { 1.0 + 0.2 * u } else { 0.5 * u };
                bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            let dense = poot_quant::weights::DenseWeight::try_new(
                poot_tensor::DType::BF16,
                shape,
                bytes.into(),
            )
            .expect("fixture bytes match their shape");
            store
                .insert(key, poot_quant::weights::WeightEntry::Dense(dense))
                .expect("fixture keys are unique");
        }
        Self {
            config,
            generation: None,
            preprocessor: None,
            store: store.build(),
        }
    }

    /// A checkpoint of `config` holding one all-zero BF16 tensor per `(key, shape)`, for a test that reads
    /// only the traced graph at a real checkpoint's dimensions: the zero pages are never touched, so a
    /// 27B-parameter store costs address space and no time.
    #[cfg(feature = "test-support")]
    pub fn zeroed_bf16(
        config: serde_json::Value,
        tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
    ) -> Self {
        let mut store = WeightStore::builder();
        for (key, shape) in tensors {
            let bytes = vec![0u8; 2 * shape.iter().product::<usize>()];
            let dense = poot_quant::weights::DenseWeight::try_new(
                poot_tensor::DType::BF16,
                shape,
                bytes.into(),
            )
            .expect("zero bytes match their shape");
            store
                .insert(key, poot_quant::weights::WeightEntry::Dense(dense))
                .expect("fixture keys are unique");
        }
        Self {
            config,
            generation: None,
            preprocessor: None,
            store: store.build(),
        }
    }

    /// The fixture as a [`RawConfig`] borrowing its config and generation config.
    pub fn raw(&self) -> RawConfig<'_> {
        RawConfig::HfJson {
            config: &self.config,
            generation: self.generation.as_ref(),
            preprocessor: self.preprocessor.as_ref(),
        }
    }
}

/// A vision tower of a model container: an `encode`-only model its family's front end runs.
#[derive(Debug)]
pub struct Tower {
    /// The name the front end runs the tower under.
    pub id: TowerId,
    /// The tower's model, built over the checkpoint's own weights.
    pub model: Box<dyn Model>,
}

/// What a family's entry builds from a checkpoint (the model container, ADR-0117 d6): the served
/// model and, for a vision-language family, the towers and the front end that prepares its media.
#[derive(Debug)]
pub struct ModelParts {
    /// The served model: the text decoder of a vision-language checkpoint.
    pub model: Box<dyn Model>,
    /// The weights `model` reads, when the family keys its served model under a section of the
    /// checkpoint's names (idefics3's `model.text_model.`): the section re-keyed, sharing the stored
    /// bytes. `None` when `model` reads the checkpoint's own names. The towers always read those.
    pub store: Option<WeightStore>,
    /// The encode-only models the front end runs.
    pub towers: Vec<Tower>,
    /// Preprocessing and prompt assembly for media; `None` for a family that serves token prompts and
    /// text chat only.
    pub front_end: Option<Box<dyn FrontEnd>>,
}

impl ModelParts {
    /// A container of `model` alone: no towers, no front end, the checkpoint's own weights.
    pub fn new(model: Box<dyn Model>) -> Self {
        Self {
            model,
            store: None,
            towers: Vec::new(),
            front_end: None,
        }
    }

    /// The served model reads `store`, a re-keyed section of the checkpoint.
    #[must_use]
    pub fn with_store(mut self, store: WeightStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Add the tower `id` the front end runs.
    #[must_use]
    pub fn with_tower(mut self, id: TowerId, model: Box<dyn Model>) -> Self {
        self.towers.push(Tower { id, model });
        self
    }

    /// The family's front end.
    #[must_use]
    pub fn with_front_end(mut self, front_end: Box<dyn FrontEnd>) -> Self {
        self.front_end = Some(front_end);
        self
    }
}

/// Builds a family's [`ModelParts`] from a raw config and the checkpoint's weights.
pub type BuildModel = fn(&RawConfig<'_>, &WeightStore) -> Result<ModelParts, ModelError>;

/// One family, as data: the config keys that name it and how to build it.
#[derive(Clone, Copy)]
pub struct FamilyEntry {
    /// The family's key.
    pub family: FamilyKey,
    /// The config keys that name this family, each with the config source it is read from.
    pub keys: &'static [(ConfigSource, &'static str)],
    /// Builds the family's [`ModelParts`].
    pub build: BuildModel,
}

impl fmt::Debug for FamilyEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FamilyEntry")
            .field("family", &self.family)
            .field("keys", &self.keys)
            .finish_non_exhaustive()
    }
}

/// Why a family could not be registered or resolved.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "no family is registered for {origin} {key:?}: a checkpoint loads only when its family is one of the shipped ones"
    )]
    /// No registered family is named by this key.
    Unregistered {
        /// The config source the key was read from.
        origin: ConfigSource,
        /// The key the config carries.
        key: String,
    },
    /// Two families name the same config key.
    #[error("{origin} {key:?} is registered by both {first} and {second}")]
    DuplicateKey {
        /// The config source the key is read from.
        origin: ConfigSource,
        /// The contested key.
        key: String,
        /// The family that already owns the key.
        first: FamilyKey,
        /// The family that tried to register it.
        second: FamilyKey,
    },
    /// The family is already registered.
    #[error("family {0} is already registered")]
    DuplicateFamily(FamilyKey),
    /// The config has no key naming a family.
    #[error("the config has no {0} key")]
    NoKey(ConfigSource),
}

/// A checkpoint could not be built into a model: its family is unknown, or the family refused it.
#[derive(Debug, thiserror::Error)]
pub enum LoadModelError {
    /// The registry could not resolve or register the family.
    #[error(transparent)]
    Registry(#[from] RegistryError),
    /// The family refused the checkpoint.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// The shipped families: the one list a registry is built from, one row per family. A family is added
/// by writing its file under `architectures` and adding its row here; nothing else names it. The
/// rows' family keys and config keys are disjoint, which `Registry::builtin` refuses to build
/// without and the registry tests check.
pub(crate) const SHIPPED: &[FamilyEntry] = &[
    crate::qwen2::model::ENTRY,
    llama::ENTRY,
    qwen3::ENTRY,
    olmo2::ENTRY,
    phi3::ENTRY,
    smollm3::ENTRY,
    granite::ENTRY,
    mpt::ENTRY,
    bloom::ENTRY,
    gemma3::ENTRY,
    gemma2::ENTRY,
    qwen3moe::ENTRY,
    gpt_oss::ENTRY,
    olmoe::ENTRY,
    crate::gemma4::ENTRY,
    minimax_m2::ENTRY,
    deepseek2::ENTRY,
    deepseek3::ENTRY,
    deepseek32::ENTRY,
    crate::nemotron_h::ENTRY,
    deepseek4::ENTRY,
    crate::bert::ENTRY,
    crate::multimodal::IDEFICS3,
    crate::qwen3next::ENTRY,
    qwen38_27b::ENTRY,
    qwen38::ENTRY,
    crate::glm53_flash::ENTRY,
    crate::multimodal::QWEN_VL,
];

/// The families a driver can load, keyed by the config keys that name them.
#[derive(Debug, Default)]
pub struct Registry {
    entries: Vec<FamilyEntry>,
    by_key: HashMap<(ConfigSource, String), usize>,
}

impl Registry {
    /// The registry of the [`SHIPPED`] families: the one a driver's caller hands to
    /// `ModelHandle::load`.
    pub fn builtin() -> Result<Self, RegistryError> {
        Self::index(SHIPPED.iter().copied())
    }

    /// The shipped families and `extra` ones, for a test that serves a family of its own.
    #[cfg(feature = "test-support")]
    pub fn builtin_plus(extra: &[FamilyEntry]) -> Result<Self, RegistryError> {
        Self::index(SHIPPED.iter().chain(extra).copied())
    }

    /// A registry of exactly `entries`, for a test of the registry itself.
    #[cfg(feature = "test-support")]
    pub fn of(entries: &[FamilyEntry]) -> Result<Self, RegistryError> {
        Self::index(entries.iter().copied())
    }

    /// The registry of `entries`. Refused when a family appears twice or one of its keys already
    /// names another family: every key resolves to exactly one family.
    fn index(entries: impl IntoIterator<Item = FamilyEntry>) -> Result<Self, RegistryError> {
        let mut registry = Self::default();
        for entry in entries {
            if registry.entries.iter().any(|e| e.family == entry.family) {
                return Err(RegistryError::DuplicateFamily(entry.family));
            }
            let mut keys = BTreeSet::new();
            for &(origin, key) in entry.keys {
                let taken = registry
                    .by_key
                    .get(&(origin, key.to_string()))
                    .map(|&i| registry.entries[i].family)
                    .or_else(|| (!keys.insert((origin, key))).then_some(entry.family));
                if let Some(first) = taken {
                    return Err(RegistryError::DuplicateKey {
                        origin,
                        key: key.to_string(),
                        first,
                        second: entry.family,
                    });
                }
            }
            let index = registry.entries.len();
            for &(origin, key) in entry.keys {
                registry.by_key.insert((origin, key.to_string()), index);
            }
            registry.entries.push(entry);
        }
        Ok(registry)
    }

    /// The family `raw` names.
    pub fn resolve(&self, raw: &RawConfig<'_>) -> Result<&FamilyEntry, RegistryError> {
        let origin = raw.source();
        let key = raw.family_key()?;
        self.by_key
            .get(&(origin, key.to_string()))
            .map(|&i| &self.entries[i])
            .ok_or_else(|| RegistryError::Unregistered {
                origin,
                key: key.to_string(),
            })
    }

    /// Resolve `raw`'s family and build its model container over `store`.
    pub fn build(
        &self,
        raw: &RawConfig<'_>,
        store: &WeightStore,
    ) -> Result<ModelParts, LoadModelError> {
        let entry = self.resolve(raw)?;
        Ok((entry.build)(raw, store)?)
    }

    /// Every registered family, in registration order.
    pub fn entries(&self) -> &[FamilyEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ModelSchema;
    use crate::model::{ModelConfig, StepSpec, TraceError};
    use crate::test_support::family_fixtures;
    use poot_graph_ir::Graph;
    use poot_quant::StoredBytes;
    use poot_weights::WeightMap;
    use std::sync::Arc;

    use poot_text::{ChatFormat, HermesToolCodec};
    use std::num::NonZeroUsize;

    /// The registry holds exactly the [`SHIPPED`] families, in order, and the fixture table has a
    /// row for each of them and for nothing else: a family is registered by its `SHIPPED` row alone,
    /// and a shipped family without a checkpoint to test it on, or a checkpoint of a family that is
    /// not shipped, fails here.
    #[test]
    fn builtin_registers_exactly_the_shipped_rows_and_each_has_fixtures() {
        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        let shipped: Vec<FamilyKey> = SHIPPED.iter().map(|entry| entry.family).collect();
        let registered: Vec<FamilyKey> = registry.entries().iter().map(|e| e.family).collect();
        assert_eq!(registered, shipped);
        let fixtures: Vec<FamilyKey> = family_fixtures::FIXTURES
            .iter()
            .map(|fixtures| fixtures.family)
            .collect();
        assert_eq!(
            fixtures, shipped,
            "the fixture table is one row per shipped family, in SHIPPED order"
        );
    }

    /// Card 732 SC-007: every shipped family's fixture greedy-decodes a non-constant sequence on
    /// the CPU oracle (the three-token prompt as the driver runs it, then eight decode steps), so its rows cover more
    /// than one argmax. Mutation: zero the fixture's head; the sequence is constant.
    #[test]
    fn every_fixture_greedy_decodes_a_non_constant_sequence() {
        use crate::components::standard::oracle::run_step;
        use crate::model::ModelOutput;

        fn argmax(logits: &[f32]) -> i32 {
            let mut best = 0;
            for (i, &v) in logits.iter().enumerate() {
                assert!(v.is_finite(), "logit {i} is {v}");
                if v > logits[best] {
                    best = i;
                }
            }
            best as i32
        }

        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        assert!(!registry.entries().is_empty());
        for entry in registry.entries() {
            let fixture = family_fixtures::base(entry.family);
            let parts = registry
                .build(&fixture.raw(), &fixture.store)
                .unwrap_or_else(|e| panic!("{}: {e}", entry.family));
            let (model, store) = (parts.model, parts.store.unwrap_or(fixture.store));
            // An encoder returns hidden states or scores, not logits: it has no decode to greedy
            // through (its own tests cover its `Encode` step).
            let Some(ModelOutput::Logits { .. }) = model.schema().step().map(|p| p.output()) else {
                continue;
            };
            let (capacity, prompt) = (16, [1, 2, 3]);
            let (logits, mut state) =
                crate::components::testing::run_prompt(&*model, &store, capacity, &prompt);
            let mut token = argmax(&logits);
            let mut sequence = vec![token];
            for pos in prompt.len()..prompt.len() + 8 {
                let (logits, next) =
                    run_step(&*model, &store, capacity, &[token], pos as i32, &state);
                token = argmax(&logits);
                sequence.push(token);
                state = next;
            }
            assert!(
                sequence.iter().any(|&t| t != sequence[0]),
                "{}: the fixture's greedy sequence {sequence:?} is constant",
                entry.family
            );
        }
    }

    /// Every `.rs` file under `dir`, as (path relative to `root`, text).
    fn sources(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(root, &path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let relative = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                out.push((relative, std::fs::read_to_string(&path).unwrap()));
            }
        }
    }

    /// SC-001: a family is registered by one row of [`SHIPPED`] and nowhere else. The scripted add: a new
    /// family is one file under `architectures`, its module line and its `SHIPPED` row (and its fixture
    /// row in test support, which holds checkpoints, not registrations). This counts the files that
    /// name a family's entry (`module::ENTRY`): `registry.rs` and the two places a container reuses an
    /// entry (the vision-language aliases, and idefics3's text tower built with llama's), never a
    /// second index. Mutation: keep an index of the families beside `SHIPPED` (a `const INDEX:
    /// &[FamilyEntry] = &[qwen3::ENTRY]` in any other file); the file is counted and the row is red.
    #[test]
    fn a_family_entry_is_named_in_the_shipped_slice_only() {
        const ALLOWED: [&str; 3] = [
            "registry.rs",
            "multimodal/mod.rs",
            "multimodal/idefics3/mod.rs",
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        sources(&root, &root, &mut files);
        let naming: Vec<&str> = files
            .iter()
            .filter(|(_, text)| {
                text.lines().any(|line| {
                    let code = line.split("//").next().unwrap_or("");
                    code.contains("::ENTRY") || code.contains("multimodal::IDEFICS3")
                })
            })
            .map(|(path, _)| path.as_str())
            .collect();
        let extra: Vec<&&str> = naming
            .iter()
            .filter(|path| !ALLOWED.contains(*path))
            .collect();
        assert!(
            extra.is_empty(),
            "these files name a family entry besides the SHIPPED slice: {extra:?}"
        );
        assert!(
            naming.contains(&"registry.rs"),
            "the SHIPPED slice is where the entries are named"
        );
    }

    /// Every key of every shipped family resolves, through the public `resolve` of a config naming
    /// it, to that family and no other: a registry has no key twice (it refuses to build), so the
    /// keys are a function from (config source, key) to family. An HF config names its family by
    /// `model_type`, a GGUF by `general.architecture`.
    #[test]
    fn every_builtin_key_resolves_to_exactly_its_family() {
        use poot_load::gguf::{GgufValue, write_gguf};
        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        let mut seen = BTreeSet::new();
        for entry in registry.entries() {
            assert!(!entry.keys.is_empty(), "{} names no key", entry.family);
            for &(origin, key) in entry.keys {
                assert!(
                    seen.insert((origin, key)),
                    "{origin} {key:?} is listed twice"
                );
                let resolved = match origin {
                    ConfigSource::HfModelType => {
                        let config = serde_json::json!({ "model_type": key });
                        let raw = RawConfig::HfJson {
                            config: &config,
                            generation: None,
                            preprocessor: None,
                        };
                        registry.resolve(&raw).map(|e| e.family)
                    }
                    ConfigSource::GgufArchitecture => {
                        let gguf = GgufIndex::parse(&StoredBytes::from(write_gguf(
                            &[("general.architecture", GgufValue::Str(key.into()))],
                            &[],
                        )))
                        .unwrap();
                        registry.resolve(&RawConfig::Gguf(&gguf)).map(|e| e.family)
                    }
                };
                assert_eq!(resolved, Ok(entry.family), "{origin} {key:?}");
            }
        }
    }

    #[derive(Debug)]
    struct Unbuildable;

    impl Model for Unbuildable {
        fn config(&self) -> &ModelConfig {
            unreachable!("never built")
        }
        fn schema(&self) -> &ModelSchema {
            unreachable!("never built")
        }
        fn weights(&self) -> &WeightMap {
            unreachable!("never built")
        }
        fn trace_into(&self, _: &StepSpec, _: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
            unreachable!("never built")
        }
    }

    fn entry(family: &'static str, keys: &'static [(ConfigSource, &'static str)]) -> FamilyEntry {
        FamilyEntry {
            family: FamilyKey::new(family),
            keys,
            build: |_, _| Ok(ModelParts::new(Box::new(Unbuildable))),
        }
    }

    /// SC-001 (spec 507): one key registered under two families is refused, and no registry is
    /// built; a key the same entry lists twice is refused the same way, and so is one family twice.
    #[test]
    fn a_key_registered_under_two_families_is_refused() {
        let err = Registry::of(&[
            entry("alpha", &[(ConfigSource::HfModelType, "shared")]),
            entry(
                "beta",
                &[
                    (ConfigSource::GgufArchitecture, "beta"),
                    (ConfigSource::HfModelType, "shared"),
                ],
            ),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            RegistryError::DuplicateKey {
                origin: ConfigSource::HfModelType,
                key: "shared".into(),
                first: FamilyKey::new("alpha"),
                second: FamilyKey::new("beta"),
            }
        );

        let err = Registry::of(&[entry(
            "gamma",
            &[
                (ConfigSource::HfModelType, "g"),
                (ConfigSource::HfModelType, "g"),
            ],
        )])
        .unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateKey { .. }), "{err}");

        assert_eq!(
            Registry::of(&[entry("alpha", &[]), entry("alpha", &[])]).unwrap_err(),
            RegistryError::DuplicateFamily(FamilyKey::new("alpha"))
        );

        let registry =
            Registry::of(&[entry("alpha", &[(ConfigSource::HfModelType, "a")])]).unwrap();
        let gguf =
            poot_load::gguf::GgufIndex::parse(&StoredBytes::from(poot_load::gguf::write_gguf(
                &[(
                    "general.architecture",
                    poot_load::gguf::GgufValue::Str("a".into()),
                )],
                &[],
            )))
            .unwrap();
        assert!(
            matches!(
                registry.resolve(&RawConfig::Gguf(&gguf)),
                Err(RegistryError::Unregistered { .. })
            ),
            "a key resolves only under the config source it was registered for"
        );
    }

    /// SC-003: qwen2.5-0.5b's end-of-sequence set is the union of its two config files. Literal
    /// snippets of `~/models/qwen2.5-0.5b/{config,generation_config}.json`.
    #[test]
    fn qwen2_5_eos_is_the_union_of_config_and_generation_config() {
        let config = serde_json::json!({
            "model_type": "qwen2",
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_position_embeddings": 32768,
            "vocab_size": 151936
        });
        let generation = serde_json::json!({
            "bos_token_id": 151643,
            "pad_token_id": 151643,
            "eos_token_id": [151645, 151643]
        });
        let raw = RawConfig::HfJson {
            config: &config,
            generation: Some(&generation),
            preprocessor: None,
        };
        let family = FamilyKey::new("qwen2");
        let eos = raw.eos_token_ids(family).unwrap();
        let config = ModelConfig {
            family,
            vocab: 151936,
            max_positions: 32768,
            eos,
            bos: Some(151643),
            prompt_bos: None,
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
            tool_codec: Some(Arc::new(HermesToolCodec)),
        };
        assert_eq!(config.eos, BTreeSet::from([151643, 151645]));
    }

    #[test]
    fn eos_ids_of_the_wrong_type_are_a_typed_config_error() {
        let config = serde_json::json!({ "eos_token_id": [1, "two"] });
        let err = RawConfig::HfJson {
            config: &config,
            generation: None,
            preprocessor: None,
        }
        .eos_token_ids(FamilyKey::new("x"))
        .unwrap_err();
        assert!(
            matches!(
                err,
                ModelError::Config {
                    field: "config.json eos_token_id",
                    reason: ConfigReason::WrongType,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn gguf_eos_is_the_tokenizer_eos_id() {
        use poot_load::gguf::{GgufValue, write_gguf};
        let gguf = GgufIndex::parse(&StoredBytes::from(write_gguf(
            &[
                ("general.architecture", GgufValue::Str("qwen2".into())),
                ("tokenizer.ggml.eos_token_id", GgufValue::U32(151645)),
            ],
            &[],
        )))
        .unwrap();
        let raw = RawConfig::Gguf(&gguf);
        assert_eq!(raw.family_key(), Ok("qwen2"));
        assert_eq!(raw.source(), ConfigSource::GgufArchitecture);
        assert_eq!(
            raw.eos_token_ids(FamilyKey::new("qwen2")).unwrap(),
            BTreeSet::from([151645])
        );
    }
}
