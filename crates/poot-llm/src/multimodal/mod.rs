//! Media handling over the driver, with no family in it.
//!
//! A vision-language model's front end (preprocessing, prompt assembly) is part of its registry
//! entry (`poot_models::multimodal::front_end`). This module is what the driver crate adds around it:
//! decoding an image's bytes ([`decode_image`]), decoding an animation's frames ([`Apng`]), running a
//! model's towers on drivers of their own ([`TowerDrivers`]) and the session that ties a loaded model's
//! front end to those drivers ([`FrontEndSession`]). The encoder clients ([`encoders`]) do the host
//! arithmetic over `encode` rows.

mod apng;
mod decode;
pub mod encoders;
mod session;
mod towers;

pub use apng::{Apng, ApngError, ApngFrames};
pub use decode::{ImageDecodeError, decode_image};
pub use session::FrontEndSession;
pub use towers::{TowerDrivers, tower_options};
