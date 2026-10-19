//! A loaded model's front end, running on its towers' drivers.

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_executor::Executor;
use poot_models::multimodal::front_end::{
    ChatTurn, FrontEnd, FrontEndError, Media, Prepared, TowerId,
};

use super::decode::decode_image;
use super::towers::TowerDrivers;
use crate::driver::error::DriverError;
use crate::driver::{DriverOptions, ModelHandle};

/// The media side of a loaded vision-language model: its front end and the drivers of its towers.
/// It lowers a conversation with media to what the text model's driver takes ([`Prepared`]); it never
/// generates.
pub struct FrontEndSession {
    handle: Arc<ModelHandle>,
    front_end: Arc<dyn FrontEnd>,
    towers: TowerDrivers,
}

impl FrontEndSession {
    /// Open the session of `handle`, or `None` when its model takes no media. `open` supplies each
    /// tower's executor and driver options ([`TowerDrivers::open`]).
    pub fn open<E: From<DriverError>>(
        handle: Arc<ModelHandle>,
        open: impl FnMut(TowerId, NonZeroUsize) -> Result<(Box<dyn Executor>, DriverOptions), E>,
    ) -> Result<Option<Self>, E> {
        let Some(front_end) = handle.front_end().map(Arc::clone) else {
            return Ok(None);
        };
        let towers = TowerDrivers::open(&handle, open)?;
        Ok(Some(Self {
            handle,
            front_end,
            towers,
        }))
    }

    /// The towers' drivers, for their stats and caps.
    pub fn towers(&self) -> &TowerDrivers {
        &self.towers
    }

    /// Preprocess `media` (in the order of `turns`), run the model's towers on it and assemble the prompt
    /// and its request inputs.
    pub fn prepare(
        &mut self,
        turns: &[ChatTurn<'_>],
        media: &mut [Media<'_>],
    ) -> Result<Prepared, FrontEndError> {
        self.front_end
            .prepare(turns, media, &mut self.towers, self.handle.text())
    }

    /// [`Self::prepare`] for `images`, the encoded bytes (PNG or JPEG) of the images the `turns` carry.
    /// Bytes that are not an image, or are too large, are the client's to fix.
    pub fn prepare_images(
        &mut self,
        images: &[&[u8]],
        turns: &[ChatTurn<'_>],
    ) -> Result<Prepared, FrontEndError> {
        let decoded = images
            .iter()
            .map(|bytes| decode_image(bytes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| FrontEndError::Refused(error.to_string()))?;
        let mut media: Vec<Media<'_>> = decoded.iter().map(Media::Image).collect();
        self.prepare(turns, &mut media)
    }
}
