//! The idefics3 / SmolVLM family: one registry entry keyed by the checkpoint's own `model_type`
//! (`idefics3`) that builds the whole container (ADR-0117 d6).
//!
//! The served model is the `llama` family built from the checkpoint's `text_config` over the
//! `model.text_model.` section of its weights; the vision tower ([`tower`]: SigLIP and the connector) is
//! a container part that reads the checkpoint's own `model.vision_model.` and `model.connector.` rows;
//! the [`front`] end splits and normalizes images ([`preprocess`]), runs the tower on each frame and
//! assembles the prompt around the `<image>` placeholders.

use poot_quant::weights::{WeightKey, WeightStore};

use self::front::Idefics3FrontEnd;
use self::preprocess::PreprocessConfig;
use crate::architectures::llama;
use crate::components::dims::{Field, Fields};
use crate::model::{ConfigReason, FamilyKey, ModelError};
use crate::multimodal::front_end::TowerId;
use crate::names::hf_base_shapes;
use crate::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig};

mod front;
pub(crate) mod preprocess;
pub(crate) mod tower;

/// The idefics3 family's key.
pub(crate) const FAMILY: FamilyKey = FamilyKey::new("idefics3");

/// The vision tower's name in the container.
pub(crate) const VISION: TowerId = TowerId::new("vision");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "idefics3")],
    build,
    fixture,
    variants: &[],
};

/// The checkpoint's names of the text decoder's weights: `model.text_model.X` is the `llama` family's `model.X`.
const TEXT_PREFIX: &str = "model.text_model.";

const SCALE: Field = Field::hf("scale_factor");
const IMAGE_TOKEN: Field = Field::hf("image_token_id");

/// Upper bound on the idefics3 frames of one request. Each image tiles into 17 to 34 frames (about
/// 3 MB of pixels each), all held before the tower runs, and the image count is client-controlled, so
/// the per-image pixel cap does not bound it.
const MAX_FRAMES: usize = 512;

fn config_error(field: &'static str, reason: ConfigReason) -> ModelError {
    ModelError::Config {
        family: FAMILY,
        field,
        reason,
    }
}

/// The object `field` of `config`.
fn object<'a>(
    config: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a serde_json::Value, ModelError> {
    match config.get(field) {
        Some(value) if value.is_object() => Ok(value),
        Some(serde_json::Value::Null) | None => Err(config_error(field, ConfigReason::Missing)),
        Some(_) => Err(config_error(field, ConfigReason::WrongType)),
    }
}

/// The entries of `store` under `prefix`, re-keyed under `to`, and the entries named in `keep`
/// unchanged: one section of a multimodal checkpoint as a store of its own. The entries share the
/// stored bytes; nothing is copied.
fn section(store: &WeightStore, prefix: &str, to: &str, keep: &[&str]) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        let name = key.as_str();
        let renamed = name
            .strip_prefix(prefix)
            .map(|rest| format!("{to}{rest}"))
            .or_else(|| keep.contains(&name).then(|| name.to_string()));
        if let Some(renamed) = renamed {
            out.insert(WeightKey::new(renamed), entry.clone())
                .expect("a section's keys are unique: its prefix is stripped uniformly");
        }
    }
    out.build()
}

/// `preprocessor_config.json`'s parameters of the idefics3 splitting pipeline.
fn preprocess_config(
    preprocessor: Option<&serde_json::Value>,
    image_size: usize,
) -> Result<PreprocessConfig, ModelError> {
    let config =
        preprocessor.ok_or_else(|| config_error("preprocessor_config", ConfigReason::Missing))?;
    let size = object(config, "size")?;
    let longest_edge = size
        .get("longest_edge")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| config_error("size.longest_edge", ConfigReason::Missing))?;
    let floats = |field: &'static str| -> Result<[f32; 3], ModelError> {
        let values = config
            .get(field)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| config_error(field, ConfigReason::Missing))?;
        let wrong = || config_error(field, ConfigReason::WrongType);
        if values.len() != 3 {
            return Err(config_error(field, ConfigReason::Length { expected: 3 }));
        }
        let mut out = [0.0; 3];
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = value.as_f64().ok_or_else(wrong)? as f32;
        }
        Ok(out)
    };
    let rescale_factor = config
        .get("rescale_factor")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| config_error("rescale_factor", ConfigReason::Missing))?
        as f32;
    Ok(PreprocessConfig {
        image_size,
        longest_edge,
        rescale_factor,
        mean: floats("image_mean")?,
        std: floats("image_std")?,
    })
}

/// The container of an idefics3 checkpoint: the text decoder, the vision tower and the front end.
fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let RawConfig::HfJson {
        config,
        generation,
        preprocessor,
    } = raw
    else {
        return Err(config_error(
            "general.architecture",
            ConfigReason::Unsupported,
        ));
    };
    let fields = Fields::new(raw, FAMILY);
    let scale = fields.count(SCALE, None)?;
    let image_token = fields
        .token(IMAGE_TOKEN)?
        .ok_or_else(|| config_error(IMAGE_TOKEN.hf, ConfigReason::Missing))?;
    let text_config = object(config, "text_config")?;
    let mut vision_config = object(config, "vision_config")?.clone();
    let text_width = Fields::new(
        &RawConfig::HfJson {
            config: text_config,
            generation: None,
            preprocessor: None,
        },
        FAMILY,
    )
    .count(Field::hf("hidden_size"), None)?;
    // The tower reads these two facts from its own config object; they live at the checkpoint's top level.
    vision_config["scale_factor"] = serde_json::json!(scale);
    vision_config["text_hidden_size"] = serde_json::json!(text_width);
    let vision_raw = RawConfig::HfJson {
        config: &vision_config,
        generation: None,
        preprocessor: None,
    };
    let params = tower::Idefics3Params::from_raw(&vision_raw)?;
    let front_end = Idefics3FrontEnd::new(
        preprocess_config(*preprocessor, params.image_size())?,
        image_token,
        params.visual_tokens(),
        params.frame(),
    );
    let vision = tower::build(&vision_raw, store)?;
    let text_store = section(store, TEXT_PREFIX, "model.", &["lm_head.weight"]);
    let text_raw = RawConfig::HfJson {
        config: text_config,
        generation: *generation,
        preprocessor: None,
    };
    let text = (llama::ENTRY.build)(&text_raw, &text_store)?;
    Ok(text
        .with_store(text_store)
        .with_tower(VISION, vision)
        .with_front_end(Box::new(front_end)))
}

/// The fixture's tokenizer-facing facts: a 48-word vocabulary whose id 42 is `<image>`.
const FIXTURE_VOCAB: usize = 48;
const FIXTURE_IMAGE_TOKEN: usize = 42;

/// A tiny SmolVLM: the tower fixture's 8-pixel frames (16 patches, 4 visual tokens after a 2x2 pixel shuffle)
/// beside a 24-wide llama text tower, in one checkpoint's names.
fn fixture() -> Fixture {
    let vision = tower::fixture();
    let mut vision_config = vision.config.clone();
    let scale = vision_config
        .as_object_mut()
        .and_then(|config| config.remove("scale_factor"))
        .expect("the tower fixture names its scale factor");
    vision_config
        .as_object_mut()
        .and_then(|config| config.remove("text_hidden_size"));
    let text_config = serde_json::json!({
        "model_type": "llama",
        "vocab_size": FIXTURE_VOCAB,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": FIXTURE_VOCAB - 1,
    });
    let text = Fixture::bf16(text_config.clone(), hf_base_shapes(&text_config, true));
    let mut store = WeightStore::builder();
    for (key, entry) in vision.store.iter() {
        store
            .insert(key.clone(), entry.clone())
            .expect("the tower's keys are unique");
    }
    for (key, entry) in text.store.iter() {
        let name = key.as_str();
        let name = name
            .strip_prefix("model.")
            .map_or_else(|| name.to_string(), |rest| format!("{TEXT_PREFIX}{rest}"));
        store
            .insert(WeightKey::new(name), entry.clone())
            .expect("the text tower's keys are unique");
    }
    Fixture {
        config: serde_json::json!({
            "model_type": "idefics3",
            "vision_config": vision_config,
            "text_config": text_config,
            "scale_factor": scale,
            "image_token_id": FIXTURE_IMAGE_TOKEN,
        }),
        generation: None,
        preprocessor: Some(serde_json::json!({
            "size": { "longest_edge": 16 },
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        })),
        store: store.build(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section re-keys the entries under its prefix and keeps the named ones: the text tower's
    /// store reads `model.layers.*` out of `model.text_model.layers.*`, and holds nothing of the
    /// vision tower.
    #[test]
    fn a_section_rekeys_its_prefix_and_drops_the_rest() {
        let fx = Fixture::bf16(
            serde_json::json!({}),
            [
                ("model.text_model.norm.weight".to_string(), vec![2]),
                (
                    "model.vision_model.post_layernorm.weight".to_string(),
                    vec![2],
                ),
                ("lm_head.weight".to_string(), vec![2, 2]),
            ],
        );
        let text = section(
            &fx.store,
            "model.text_model.",
            "model.",
            &["lm_head.weight"],
        );
        let mut keys: Vec<&str> = text.keys().map(WeightKey::as_str).collect();
        keys.sort();
        assert_eq!(keys, ["lm_head.weight", "model.norm.weight"]);
    }

    /// The entry builds the whole container from the checkpoint's top-level config: the text decoder over
    /// its section, the vision tower under its name, and a front end. A `text_config` or `vision_config`
    /// that is missing is a typed config error naming it, not a panic.
    #[test]
    fn the_entry_builds_the_text_model_the_tower_and_the_front_end() {
        let fixture = fixture();
        let parts = (ENTRY.build)(&fixture.raw(), &fixture.store).unwrap();
        assert_eq!(parts.model.config().family.as_str(), "llama");
        assert_eq!(parts.model.config().vocab, FIXTURE_VOCAB);
        assert_eq!(parts.towers.len(), 1);
        assert_eq!(parts.towers[0].id, VISION);
        assert_eq!(parts.towers[0].model.config().max_positions, 16);
        assert!(parts.front_end.is_some());
        assert!(
            parts
                .store
                .as_ref()
                .is_some_and(|store| store.contains("model.embed_tokens.weight")),
            "the text model reads the re-keyed section"
        );
        for field in ["text_config", "vision_config"] {
            let mut config = fixture.config.clone();
            config.as_object_mut().unwrap().remove(field);
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
                preprocessor: fixture.preprocessor.as_ref(),
            };
            let error = (ENTRY.build)(&raw, &fixture.store).unwrap_err();
            assert!(
                matches!(&error, ModelError::Config { field: named, reason: ConfigReason::Missing, .. } if *named == field),
                "{field}: {error}"
            );
        }
    }

    /// A checkpoint that ships no `preprocessor_config.json` cannot preprocess images: refused at build.
    #[test]
    fn a_checkpoint_without_a_preprocessor_config_is_refused() {
        let fixture = fixture();
        let raw = RawConfig::HfJson {
            config: &fixture.config,
            generation: None,
            preprocessor: None,
        };
        let error = (ENTRY.build)(&raw, &fixture.store).unwrap_err();
        assert!(
            matches!(
                &error,
                ModelError::Config {
                    field: "preprocessor_config",
                    ..
                }
            ),
            "{error}"
        );
    }
}
