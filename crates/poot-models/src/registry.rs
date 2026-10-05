//! The one family registry (ADR-0104): a checkpoint's config names its family, and the
//! family's [`FamilyEntry`] builds its [`Model`]. Families are data: any crate adds one with
//! [`Registry::register`]. The shipped families are the closed [`Family`] enum, whose exhaustive
//! match is the completeness check ADR-0074 keeps (a drift test, not a support list). The driver and
//! the loader take a `&Registry`; there is no global registry.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use poot_load::gguf::GgufIndex;
use poot_quant::weights::WeightStore;

use crate::model::{ConfigReason, FamilyKey, Model, ModelError};

/// A checkpoint's raw config, as a family parses it.
#[derive(Clone, Copy)]
pub enum RawConfig<'a> {
    /// An HF checkpoint: `config.json`, and `generation_config.json` when the checkpoint ships one.
    HfJson {
        config: &'a serde_json::Value,
        generation: Option<&'a serde_json::Value>,
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
        if let RawConfig::HfJson { config, generation } = self {
            s.field("config", config).field("generation", generation);
        }
        s.finish_non_exhaustive()
    }
}

impl RawConfig<'_> {
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
            RawConfig::HfJson { config, generation } => {
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

/// A synthetic checkpoint a family ships for tests: it owns its raw config, because
/// [`RawConfig`] borrows.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub config: serde_json::Value,
    pub generation: Option<serde_json::Value>,
    pub store: WeightStore,
}

impl Fixture {
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
            store: store.build(),
        }
    }

    pub fn raw(&self) -> RawConfig<'_> {
        RawConfig::HfJson {
            config: &self.config,
            generation: self.generation.as_ref(),
        }
    }
}

/// Builds a family's [`Model`] from a raw config and the checkpoint's weights.
pub type BuildModel = fn(&RawConfig<'_>, &WeightStore) -> Result<Box<dyn Model>, ModelError>;

/// One family, as data: the config keys that name it, how to build it and its test fixture.
#[derive(Clone, Copy)]
pub struct FamilyEntry {
    pub family: FamilyKey,
    pub keys: &'static [(ConfigSource, &'static str)],
    pub build: BuildModel,
    pub fixture: fn() -> Fixture,
}

impl fmt::Debug for FamilyEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FamilyEntry")
            .field("family", &self.family)
            .field("keys", &self.keys)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "no family is registered for {origin} {key:?}: a family must be added with Registry::register before a checkpoint naming it can load"
    )]
    Unregistered { origin: ConfigSource, key: String },
    #[error("{origin} {key:?} is registered by both {first} and {second}")]
    DuplicateKey {
        origin: ConfigSource,
        key: String,
        first: FamilyKey,
        second: FamilyKey,
    },
    #[error("family {0} is already registered")]
    DuplicateFamily(FamilyKey),
    #[error("the config has no {0} key")]
    NoKey(ConfigSource),
}

/// A checkpoint could not be built into a model: its family is unknown, or the family refused it.
#[derive(Debug, thiserror::Error)]
pub enum LoadModelError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// The families a driver can load, keyed by the config keys that name them.
#[derive(Debug, Default)]
pub struct Registry {
    entries: Vec<FamilyEntry>,
    by_key: HashMap<(ConfigSource, String), usize>,
}

impl Registry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Every shipped family ([`Family::ALL`]): the registry a driver's caller hands to
    /// `ModelHandle::load`.
    pub fn builtin() -> Result<Self, RegistryError> {
        let mut registry = Self::empty();
        for family in Family::ALL {
            registry.register(family.entry())?;
        }
        Ok(registry)
    }

    /// Add a family. Refused, with the registry unchanged, when the family is already registered or
    /// one of its keys already names another family: every key resolves to exactly one family.
    pub fn register(&mut self, entry: FamilyEntry) -> Result<(), RegistryError> {
        if self.entries.iter().any(|e| e.family == entry.family) {
            return Err(RegistryError::DuplicateFamily(entry.family));
        }
        let mut keys = BTreeSet::new();
        for &(origin, key) in entry.keys {
            let taken = self
                .by_key
                .get(&(origin, key.to_string()))
                .map(|&i| self.entries[i].family)
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
        let index = self.entries.len();
        for &(origin, key) in entry.keys {
            self.by_key.insert((origin, key.to_string()), index);
        }
        self.entries.push(entry);
        Ok(())
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

    /// Resolve `raw`'s family and build its model over `store`.
    pub fn build(
        &self,
        raw: &RawConfig<'_>,
        store: &WeightStore,
    ) -> Result<Box<dyn Model>, LoadModelError> {
        let entry = self.resolve(raw)?;
        Ok((entry.build)(raw, store)?)
    }

    pub fn entries(&self) -> &[FamilyEntry] {
        &self.entries
    }
}

/// The shipped families: each variant is a family that implements [`Model`] (no
/// placeholders). A new variant does not build until [`Family::entry`] and the test-only index have
/// an arm for it, and `registry::tests` fail until [`Family::ALL`] lists it: the test counts the
/// variants with the compiler's `variant_count`, not a hand-kept number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    Qwen2,
    Llama,
    Qwen3,
    Olmo2,
    Phi3,
    Smollm3,
    Granite,
    Mpt,
    Bloom,
    Gemma3,
    Gemma2,
}

impl Family {
    /// Every variant, in [`Self::index`] order; [`Registry::builtin`] registers these.
    pub(crate) const ALL: &'static [Family] = &[
        Family::Qwen2,
        Family::Llama,
        Family::Qwen3,
        Family::Olmo2,
        Family::Phi3,
        Family::Smollm3,
        Family::Granite,
        Family::Mpt,
        Family::Bloom,
        Family::Gemma3,
        Family::Gemma2,
    ];

    pub(crate) const fn entry(self) -> FamilyEntry {
        match self {
            Family::Qwen2 => crate::qwen2::model::ENTRY,
            Family::Llama => crate::llama::ENTRY,
            Family::Qwen3 => crate::qwen3::ENTRY,
            Family::Olmo2 => crate::olmo2::ENTRY,
            Family::Phi3 => crate::phi3::ENTRY,
            Family::Smollm3 => crate::smollm3::ENTRY,
            Family::Granite => crate::granite::ENTRY,
            Family::Mpt => crate::mpt::ENTRY,
            Family::Bloom => crate::bloom::ENTRY,
            Family::Gemma3 => crate::gemma3::ENTRY,
            Family::Gemma2 => crate::gemma2::ENTRY,
        }
    }

    /// Dense 0-based position in [`Self::ALL`]. Exhaustive (no `_` arm), so a new variant needs one.
    /// `cfg(test)`: it exists for the drift test.
    #[cfg(test)]
    const fn index(self) -> usize {
        match self {
            Family::Qwen2 => 0,
            Family::Llama => 1,
            Family::Qwen3 => 2,
            Family::Olmo2 => 3,
            Family::Phi3 => 4,
            Family::Smollm3 => 5,
            Family::Granite => 6,
            Family::Mpt => 7,
            Family::Bloom => 8,
            Family::Gemma3 => 9,
            Family::Gemma2 => 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatFormat;
    use crate::model::{ModelConfig, ModelOutput, Phase, StepShape, TraceError};
    use poot_graph_ir::{Graph, ValidationOutputs};
    use poot_quant::weights::WeightMap;
    use std::num::NonZeroUsize;

    /// SC-001: `Family::ALL` lists every variant exactly once, in index order, and the built-in
    /// registry holds exactly those families, each under its own entry: the registry and the closed
    /// enum agree both ways. The variant count is the compiler's, so a variant added to the enum
    /// but not to `ALL` fails here, whether or not anything else constructs it.
    #[test]
    fn family_all_lists_every_variant_once_and_builtin_registers_exactly_them() {
        const VARIANTS: usize = std::mem::variant_count::<Family>();
        assert_eq!(
            Family::ALL.len(),
            VARIANTS,
            "Family::ALL must list each of the {VARIANTS} variants once"
        );
        let mut seen = [false; VARIANTS];
        for &family in Family::ALL {
            let i = family.index();
            assert!(
                i < VARIANTS,
                "{family:?} has index {i} but there are {VARIANTS} variants: indices must be 0..{VARIANTS}"
            );
            assert!(!seen[i], "{family:?} appears twice in Family::ALL");
            seen[i] = true;
            assert_eq!(Family::ALL[i], family, "Family::ALL is out of index order");
        }
        assert!(
            seen.iter().all(|&s| s),
            "an index has no variant in Family::ALL"
        );

        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        let registered: Vec<FamilyKey> = registry.entries().iter().map(|e| e.family).collect();
        let shipped: Vec<FamilyKey> = Family::ALL.iter().map(|f| f.entry().family).collect();
        assert_eq!(registered, shipped);
    }

    /// Card 732 SC-007: every shipped family's fixture greedy-decodes a non-constant sequence on
    /// the CPU oracle (a prefill of three tokens, then eight decode steps), so its rows cover more
    /// than one argmax. Mutation: zero the fixture's head; the sequence is constant.
    #[test]
    fn every_fixture_greedy_decodes_a_non_constant_sequence() {
        use crate::components::standard::oracle::run_step;
        use crate::model::Phase;

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
            let fixture = (entry.fixture)();
            let model = registry
                .build(&fixture.raw(), &fixture.store)
                .unwrap_or_else(|e| panic!("{}: {e}", entry.family));
            let vocab = model.config().vocab;
            let (capacity, prompt) = (16, [1, 2, 3]);
            let (logits, mut state) = run_step(
                &*model,
                &fixture.store,
                Phase::Prefill,
                capacity,
                &prompt,
                0,
                &[],
            );
            let mut token = argmax(&logits[(prompt.len() - 1) * vocab..]);
            let mut sequence = vec![token];
            for pos in prompt.len()..prompt.len() + 8 {
                let (logits, next) = run_step(
                    &*model,
                    &fixture.store,
                    Phase::Decode,
                    capacity,
                    &[token],
                    pos as i32,
                    &state,
                );
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

    /// SC-001: every key of every shipped family resolves to that family and no other.
    #[test]
    fn every_builtin_key_resolves_to_exactly_its_family() {
        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        for entry in registry.entries() {
            for &(origin, key) in entry.keys {
                let resolved = registry
                    .by_key
                    .get(&(origin, key.to_string()))
                    .map(|&i| registry.entries[i].family);
                assert_eq!(resolved, Some(entry.family), "{origin} {key:?}");
            }
        }
    }

    #[derive(Debug)]
    struct Unbuildable;

    impl Model for Unbuildable {
        fn config(&self) -> &ModelConfig {
            unreachable!("never built")
        }
        fn weights(&self) -> &WeightMap {
            unreachable!("never built")
        }
        fn trace(&self, _: Phase, _: StepShape) -> Result<Graph<ValidationOutputs>, TraceError> {
            unreachable!("never built")
        }
    }

    fn entry(family: &'static str, keys: &'static [(ConfigSource, &'static str)]) -> FamilyEntry {
        FamilyEntry {
            family: FamilyKey::new(family),
            keys,
            build: |_, _| Ok(Box::new(Unbuildable)),
            fixture: || Fixture {
                config: serde_json::json!({}),
                generation: None,
                store: WeightStore::default(),
            },
        }
    }

    /// SC-001 (spec 507): one key registered under two families is refused, and the registry keeps
    /// resolving it to the first; a key the same entry lists twice is refused the same way.
    #[test]
    fn a_key_registered_under_two_families_is_refused() {
        let mut registry = Registry::empty();
        registry
            .register(entry("alpha", &[(ConfigSource::HfModelType, "shared")]))
            .unwrap();
        let err = registry
            .register(entry(
                "beta",
                &[
                    (ConfigSource::GgufArchitecture, "beta"),
                    (ConfigSource::HfModelType, "shared"),
                ],
            ))
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
        assert_eq!(
            registry.entries().len(),
            1,
            "a refused entry leaves no trace"
        );
        let gguf = poot_load::gguf::GgufIndex::from_bytes(&poot_load::gguf::write_gguf(
            &[(
                "general.architecture",
                poot_load::gguf::GgufValue::Str("beta".into()),
            )],
            &[],
        ))
        .unwrap();
        assert!(matches!(
            registry.resolve(&RawConfig::Gguf(&gguf)),
            Err(RegistryError::Unregistered { .. })
        ));

        let err = Registry::empty()
            .register(entry(
                "gamma",
                &[
                    (ConfigSource::HfModelType, "g"),
                    (ConfigSource::HfModelType, "g"),
                ],
            ))
            .unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateKey { .. }), "{err}");

        let mut registry = Registry::empty();
        registry.register(entry("alpha", &[])).unwrap();
        assert_eq!(
            registry.register(entry("alpha", &[])),
            Err(RegistryError::DuplicateFamily(FamilyKey::new("alpha")))
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
            output: ModelOutput::Logits { vocab: 151936 },
            prefill_granule: NonZeroUsize::MIN,
            chat: ChatFormat::ChatML,
        };
        assert_eq!(config.eos, BTreeSet::from([151643, 151645]));
    }

    #[test]
    fn eos_ids_of_the_wrong_type_are_a_typed_config_error() {
        let config = serde_json::json!({ "eos_token_id": [1, "two"] });
        let err = RawConfig::HfJson {
            config: &config,
            generation: None,
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
        let gguf = GgufIndex::from_bytes(&write_gguf(
            &[
                ("general.architecture", GgufValue::Str("qwen2".into())),
                ("tokenizer.ggml.eos_token_id", GgufValue::U32(151645)),
            ],
            &[],
        ))
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
