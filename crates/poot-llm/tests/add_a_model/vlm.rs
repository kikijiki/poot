//! A third out-of-tree family, "vlm", that takes images (POOT-1226 SC-002). Its model container is the
//! "tagged" text model ([`tagged`]) plus a front end defined in this test target: a preprocessor that
//! turns an image into four placeholders and one tag per prompt position. Nothing in `poot-llm` knows
//! the family: its registry entry builds the front end, the loader keeps it in the model handle, and a
//! [`FrontEndSession`](poot_llm::FrontEndSession) runs it. If serving an image needed a `poot-llm` edit,
//! this file could not make the family reachable.
//!
//! The preprocessor is the family's own code: the image's four quadrants (rows 0..h/2 and columns
//! 0..w/2, and so on) are averaged and bucketed into the tag table's rows, and the question is
//! tokenized with the model's own tokenizer.

use poot_models::io::{InputValue, RequestInputs};
use poot_models::model::FamilyKey;
use poot_models::model::ModelError;
use poot_models::multimodal::front_end::{
    ChatTurn, FrontEnd, FrontEndError, Image, Media, Prepared, Towers, check_turn_media,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, ModelParts, RawConfig};
use poot_quant::weights::WeightStore;
use poot_text::TextCodec;

use crate::tagged::{self, TAG, TAGS};

pub(crate) const FAMILY: FamilyKey = FamilyKey::new("poot_fixture_vlm");

/// The HF `model_type` that names the family.
const MODEL_TYPE: &str = "poot_fixture_vlm";

/// The registry entry: what `Registry::register` takes.
pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, MODEL_TYPE)],
    build,
    fixture,
    variants: &[],
};

/// The token the preprocessor puts where the image goes (any id in the vocabulary).
pub(crate) const PLACEHOLDER: u32 = 9;

/// The container: the tagged text model and this family's front end. The family has no tower, and its
/// build reads the one `RawConfig` like any other.
fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<ModelParts, ModelError> {
    let text = (tagged::ENTRY.build)(raw, store)?;
    Ok(text.with_front_end(Box::new(Preprocessor)))
}

/// The tagged checkpoint under this family's `model_type`.
pub(crate) fn fixture() -> Fixture {
    let mut fixture = tagged::fixture();
    fixture.config["model_type"] = MODEL_TYPE.into();
    fixture
}

/// The tag of each quadrant of `image`: its mean grey, bucketed into [`TAGS`] rows.
pub(crate) fn quadrant_tags(image: &Image) -> [i32; 4] {
    let (height, width) = (image.height(), image.width());
    let mut sums = [0u64; 4];
    let mut counts = [0u64; 4];
    for y in 0..height {
        for x in 0..width {
            let quadrant = usize::from(y >= height / 2) * 2 + usize::from(x >= width / 2);
            let at = (y * width + x) * 3;
            sums[quadrant] += u64::from(image.rgb()[at]);
            counts[quadrant] += 1;
        }
    }
    std::array::from_fn(|q| (sums[q] / counts[q].max(1) * TAGS as u64 / 256) as i32)
}

#[derive(Debug)]
struct Preprocessor;

impl FrontEnd for Preprocessor {
    fn prepare(
        &self,
        turns: &[ChatTurn<'_>],
        media: &mut [Media<'_>],
        _towers: &mut dyn Towers,
        text: &TextCodec,
    ) -> Result<Prepared, FrontEndError> {
        check_turn_media(turns, media.len())?;
        let [Media::Image(image)] = media else {
            return Err(FrontEndError::Refused(
                "this model takes exactly one image".into(),
            ));
        };
        let question = text
            .encode(turns[0].text)
            .map_err(|error| FrontEndError::Failed(Box::new(error)))?;
        let mut prompt = vec![PLACEHOLDER; 4];
        prompt.extend(&question);
        let mut tags = quadrant_tags(image).to_vec();
        tags.extend(question.iter().map(|_| 0));
        Ok(Prepared {
            prompt,
            inputs: RequestInputs::none().with(TAG, InputValue::Tokens(tagged::tag_values(&tags))),
        })
    }
}
