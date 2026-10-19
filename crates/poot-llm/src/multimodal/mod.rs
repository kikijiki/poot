//! Multimodal front ends over the driver: idefics3 / SmolVLM ([`front_end`]) and Qwen2.5-VL
//! ([`qwen_vl`]).
//!
//! Both lower a request with images to what any text driver takes: [`MultimodalPrompt`], the prompt's
//! token ids and the request inputs its model declares: the embedding rows at its placeholders and,
//! for a model that rotates at sectioned positions, the prompt's position table. A server holds one of them behind [`ImageFrontEnd`], opened
//! from the [`ImageCheckpoint`] that [`read_image_checkpoint`] finds in the model directory.

use std::path::Path;
use std::sync::Arc;

use poot_executor::Executor;
use poot_models::components::embed::embeds_inputs;
use poot_models::components::rope::MROPE;
use poot_models::io::{Fill, InputValue, RequestInputs, TokenValues, ValueError};
use poot_models::mrope::MropePositionIds;
use poot_models::registry::Registry;
use poot_tensor::HostTensor;

use crate::driver::{DriverOptions, ModelHandle};

pub mod encoders;
pub mod front_end;
pub mod qwen_vl;
pub(crate) mod qwen_vl_chat;
pub(crate) mod qwen_vl_image;
pub(crate) mod qwen_vl_mrope;

/// A request with images lowered to what the text driver takes.
#[derive(Clone, Debug)]
pub struct MultimodalPrompt {
    /// The prompt's token ids, the image placeholders included.
    pub prompt: Vec<u32>,
    /// The values for the inputs the text model declares: the embedding rows that replace the
    /// placeholders and, for a model with sectioned positions, where every prompt token rotates.
    pub inputs: RequestInputs,
}

/// The request inputs of a multimodal prompt: `rows` (`positions.len() * width` values, row-major) at
/// the prompt's `positions`, and for a model with sectioned positions the prompt's position table.
pub(crate) fn prompt_inputs(
    width: usize,
    positions: Vec<usize>,
    rows: Vec<f32>,
    mrope: Option<&MropePositionIds>,
) -> Result<RequestInputs, ValueError> {
    let inputs = embeds_inputs(width, positions, rows)?;
    Ok(match mrope {
        Some(ids) => inputs.with(MROPE, InputValue::Tokens(mrope_values(ids)?)),
        None => inputs,
    })
}

/// A prompt's sectioned positions as dense three-axis values over the prompt, continuing past it from
/// the collapsed position after the last token, one further per generated token.
fn mrope_values(ids: &MropePositionIds) -> Result<TokenValues, ValueError> {
    let [temporal, height, width] = ids.axes();
    let data: Vec<i32> = (0..ids.len())
        .flat_map(|i| [temporal[i], height[i], width[i]])
        .collect();
    let next = ids.next_text_position().temporal;
    Ok(
        TokenValues::dense(HostTensor::i32(vec![ids.len(), 3], data))?.with_beyond(Fill::Advance {
            from: vec![next; 3],
            step: 1,
        }),
    )
}

/// One turn of a conversation that holds images: the role, how many images the turn carries, and its
/// text. The turn's images come first in its content, then the text (the order a served request
/// flattens to). The images themselves are passed beside the turns, in the order of the turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatTurn<'a> {
    /// The turn's role as the chat template names it (`"user"`, `"assistant"`, `"system"`).
    pub role: &'a str,
    /// How many of the request's images this turn carries; they lead the turn's content.
    pub images: usize,
    /// The turn's text, after its images.
    pub text: &'a str,
}

impl<'a> ChatTurn<'a> {
    /// A user turn of `images` images and the question about them.
    pub fn user(images: usize, text: &'a str) -> Self {
        Self {
            role: "user",
            images,
            text,
        }
    }
}

/// Why a request's images did not become a prompt.
#[derive(Debug, thiserror::Error)]
pub enum ImageRequestError {
    /// The images or the question are the client's to fix.
    #[error("{0}")]
    Refused(String),
    /// The front end or its tower failed.
    #[error(transparent)]
    Failed(Box<dyn std::error::Error + Send + Sync>),
}

/// Why an image checkpoint cannot be read or its front end opened.
#[derive(Debug, thiserror::Error)]
pub enum ImageCheckpointError {
    /// The idefics3 / SmolVLM front end could not be read or opened.
    #[error(transparent)]
    Idefics3(#[from] front_end::FrontEndError),
    /// The Qwen2.5-VL front end could not be read or opened.
    #[error(transparent)]
    Qwen25Vl(#[from] qwen_vl::Qwen25VlError),
}

/// The image side of a vision-language model: image bytes and a question in, a prompt out. It never
/// generates.
pub trait ImageFrontEnd {
    /// Decode and preprocess `images` (PNG or JPEG bytes, in the order of the turns), encode them with
    /// the vision tower and render the conversation `turns` into the prompt.
    fn prepare(
        &mut self,
        images: &[&[u8]],
        turns: &[ChatTurn<'_>],
    ) -> Result<MultimodalPrompt, ImageRequestError>;
}

/// A vision-language checkpoint read from its directory. Nothing is on a device yet: the text tower
/// is a [`ModelHandle`] a driver is built over, and the image side opens on an executor of its own.
pub trait ImageCheckpoint: Send + Sync {
    /// The text tower: the model a server generates with, and the codec that tokenizes prompts.
    fn text(&self) -> &Arc<ModelHandle>;

    /// The positions the vision tower's driver needs: the patches of one image (SmolVLM: one frame).
    fn patches(&self) -> usize;

    /// Open the image side on `vision`'s executor.
    fn front_end(
        &self,
        vision: (Box<dyn Executor>, DriverOptions),
    ) -> Result<Box<dyn ImageFrontEnd>, ImageCheckpointError>;
}

/// The vision-language checkpoint in `dir`, or `None` for any other directory (a text model).
pub fn read_image_checkpoint(
    dir: &Path,
    registry: &Registry,
) -> Result<Option<Arc<dyn ImageCheckpoint>>, ImageCheckpointError> {
    if front_end::is_idefics3_dir(dir) {
        let checkpoint = front_end::Idefics3Checkpoint::read(dir, registry)?;
        return Ok(Some(Arc::new(checkpoint)));
    }
    if qwen_vl::is_qwen2_5_vl_dir(dir) {
        let checkpoint = qwen_vl::Qwen25VlCheckpoint::read(dir, registry)?;
        return Ok(Some(Arc::new(checkpoint)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use poot_models::components::embed::EMBEDS;
    use poot_models::mrope::{MropeGrid, MropeSegment, build_mrope_position_ids};

    use super::*;

    /// Two text tokens, one 1 x 4 x 4 image (a 2 x 2 block of merged tokens), two text tokens.
    fn ids() -> MropePositionIds {
        build_mrope_position_ids(
            &[
                MropeSegment::Text(2),
                MropeSegment::Image(MropeGrid::new(1, 4, 4)),
                MropeSegment::Text(2),
            ],
            2,
        )
        .unwrap()
    }

    /// A prompt's sectioned positions become dense three-axis values over the prompt: the image's four
    /// tokens sit on a 2 x 2 grid at one temporal position and the text after it continues from the
    /// largest position plus one, and past the prompt the values advance by one per position from that
    /// collapsed continuation. The binder (`driver::bind`) evaluates exactly this data, so the table here
    /// is the position every token of the prompt rotates at. Mutation: build the data in token-major
    /// instead of per-token `[temporal, height, width]` order, or start the continuation one early; the
    /// table below goes red.
    #[test]
    fn a_prompts_sectioned_positions_become_dense_values_with_a_continuation() {
        let values = mrope_values(&ids()).unwrap();
        assert_eq!(values.support().len(), 8);
        assert_eq!(values.width(), 3);
        let words = values.data().as_i32().unwrap();
        let at = |i: usize| [words[3 * i], words[3 * i + 1], words[3 * i + 2]];
        assert_eq!([at(0), at(1)], [[0, 0, 0], [1, 1, 1]]);
        assert_eq!(
            [at(2), at(3), at(4), at(5)],
            [[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]
        );
        assert_eq!([at(6), at(7)], [[4, 4, 4], [5, 5, 5]]);
        assert_eq!(
            values.beyond(),
            &Fill::Advance {
                from: vec![6; 3],
                step: 1
            }
        );
    }

    /// The embedding rows of a prompt are sparse values at its placeholders, and the prompt inputs hold
    /// both the rows and (for a model with sectioned positions) the position table.
    #[test]
    fn prompt_inputs_hold_the_rows_and_the_positions_a_model_declares() {
        let rows = vec![1.0, 2.0, 3.0, 4.0];
        let plain = prompt_inputs(2, vec![1, 3], rows.clone(), None).unwrap();
        assert!(plain.get(EMBEDS).is_some() && plain.get(MROPE).is_none());
        let sectioned = prompt_inputs(2, vec![1, 3], rows, Some(&ids())).unwrap();
        assert!(sectioned.get(EMBEDS).is_some() && sectioned.get(MROPE).is_some());
        let Some(InputValue::Tokens(embeds)) = sectioned.get(EMBEDS) else {
            panic!("the rows are per-token values");
        };
        assert_eq!(
            embeds.support(),
            &poot_models::io::Support::Sparse(vec![1, 3])
        );
        assert!(prompt_inputs(2, vec![3, 1], vec![0.0; 4], None).is_err());
    }
}
