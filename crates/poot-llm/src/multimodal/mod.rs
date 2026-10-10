//! Multimodal front ends over the driver: idefics3 / SmolVLM ([`front_end`]) and Qwen2.5-VL
//! ([`qwen_vl`]).
//!
//! Both lower a request with images to what any text driver takes: [`MultimodalPrompt`], the prompt's
//! token ids, the embedding rows at its placeholders and, for a model that rotates at sectioned
//! positions, the prompt's position table. A server holds one of them behind [`ImageFrontEnd`], opened
//! from the [`ImageCheckpoint`] that [`read_image_checkpoint`] finds in the model directory.

use std::path::Path;
use std::sync::Arc;

use poot_executor::Executor;
use poot_models::mrope::MropePositionIds;
use poot_models::registry::Registry;

use crate::driver::{DriverOptions, ModelHandle, PromptEmbeds};

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
    /// The embedding rows that replace the placeholders.
    pub embeds: PromptEmbeds,
    /// Where every prompt token rotates (`Slot::MropePosition`), for a model with sectioned
    /// positions; `None` when tokens rotate at their index.
    pub mrope: Option<MropePositionIds>,
}

/// One turn of a conversation that holds images: the role, how many images the turn carries, and its
/// text. The turn's images come first in its content, then the text (the order a served request
/// flattens to). The images themselves are passed beside the turns, in the order of the turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatTurn<'a> {
    pub role: &'a str,
    pub images: usize,
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
    #[error(transparent)]
    Idefics3(#[from] front_end::FrontEndError),
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
