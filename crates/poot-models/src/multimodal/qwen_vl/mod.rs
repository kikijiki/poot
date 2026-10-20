//! The Qwen2.5-VL family: one registry entry keyed by the checkpoint's own `model_type` (`qwen2_5_vl`)
//! that builds the whole container (ADR-0117 d6).
//!
//! The served model is the qwen2 body with sectioned RoPE (`crate::qwen2`); when the checkpoint
//! carries a `vision_config` the container also holds the vision tower ([`tower`]: the 3-D patch
//! embedding, windowed blocks and the patch merger) and the [`front`] end that preprocesses images and
//! videos ([`processor`]), runs the tower and assembles the prompt with its sectioned positions
//! (`super::mrope`). A checkpoint without a `vision_config` serves as the text model alone.

use poot_quant::weights::WeightStore;

use self::front::Qwen25VlFrontEnd;
use self::processor::Qwen25VlImageProcessorConfig;
use self::tower::Qwen25VlVisionParams;
#[cfg(feature = "test-support")]
use crate::architectures::qwen2::model::vl_fixture_config;
use crate::architectures::qwen2::model::{Qwen25VlText, VL_FAMILY};
use crate::components::standard_family::{StandardFamily, build_standard};
use crate::model::{ConfigReason, ModelError};
use crate::multimodal::front_end::TowerId;
use crate::multimodal::mrope::MropeConfigMetadata;
#[cfg(feature = "test-support")]
use crate::registry::Fixture;
use crate::registry::{FamilyEntry, ModelParts, RawConfig};

mod front;
mod media;
mod processor;
mod tower;

/// The vision tower's name in the container.
pub(crate) const VISION: TowerId = TowerId::new("vision");

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: VL_FAMILY,
    keys: <Qwen25VlText as StandardFamily>::KEYS,
    build,
};

fn front_end_error(error: impl std::error::Error + Send + Sync + 'static) -> ModelError {
    ModelError::FrontEnd {
        family: VL_FAMILY,
        source: Box::new(error),
    }
}

/// The container of a Qwen2.5-VL checkpoint: the text model alone for a text-only config, else with
/// the vision tower and the front end.
fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let text = build_standard::<Qwen25VlText>(raw, store)?;
    let RawConfig::HfJson {
        config,
        preprocessor,
        ..
    } = raw
    else {
        return Ok(text);
    };
    let Some(vision_config) = config.get("vision_config").filter(|v| v.is_object()) else {
        return Ok(text);
    };
    let vision_raw = RawConfig::HfJson {
        config: vision_config,
        generation: None,
        preprocessor: None,
    };
    let params = Qwen25VlVisionParams::from_raw(&vision_raw)?;
    let processor = (*preprocessor).ok_or(ModelError::Config {
        family: VL_FAMILY,
        field: "preprocessor_config",
        reason: ConfigReason::Missing,
    })?;
    let processor = Qwen25VlImageProcessorConfig::from_value(processor).map_err(front_end_error)?;
    let mrope = MropeConfigMetadata::from_raw(raw).map_err(front_end_error)?;
    let tower = tower::build(&vision_raw, store)?;
    Ok(text
        .with_tower(VISION, tower)
        .with_front_end(Box::new(Qwen25VlFrontEnd::new(processor, mrope, params))))
}

/// The marker ids of the fixture's 48-word vocabulary.
#[cfg(feature = "test-support")]
const FIXTURE_VISION_START: u32 = 40;
#[cfg(feature = "test-support")]
const FIXTURE_IMAGE: u32 = 41;
#[cfg(feature = "test-support")]
const FIXTURE_VIDEO: u32 = 42;

/// A tiny Qwen2.5-VL: the qwen2 text fixture beside the tower fixture (2-pixel patches that merge 2 x 2
/// into rows of the text tower's width), one set of weights.
#[cfg(feature = "test-support")]
pub(crate) fn fixture() -> Fixture {
    let text = crate::architectures::qwen2::model::vl_fixture();
    let vision = tower::fixture();
    let mut store = WeightStore::builder();
    for (key, entry) in text.store.iter().chain(vision.store.iter()) {
        store
            .insert(key.clone(), entry.clone())
            .expect("the text and tower keys are disjoint");
    }
    let mut config = vl_fixture_config();
    let mut vision_config = vision.config;
    vision_config["tokens_per_second"] = 2.into();
    config["vision_config"] = vision_config;
    config["vision_start_token_id"] = FIXTURE_VISION_START.into();
    config["image_token_id"] = FIXTURE_IMAGE.into();
    config["video_token_id"] = FIXTURE_VIDEO.into();
    Fixture {
        config,
        generation: None,
        preprocessor: Some(serde_json::json!({
            "do_resize": false,
            "min_pixels": 1,
            "max_pixels": 4096,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
        })),
        store: store.build(),
    }
}
