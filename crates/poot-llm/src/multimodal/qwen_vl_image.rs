//! CPU-only Qwen2.5-VL image decoding and image/video preprocessing.
//!
//! The output is one processor-compatible flattened-patch tensor plus ordered raw grids. Prompt
//! semantics, video codec/container decoding and vision tower execution are out of scope; the one
//! in-memory APNG adapter is bounded and composes with the caller-owned raw-video source contract.
//! PNG/JPEG RGB conversion and cubic resize use the `image` crate: deterministic for the pinned
//! dependency, but not bit-identical to Pillow for every color mode or resized pixel.

const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;

const MAX_VIDEO_FRAMES: usize = 16 * 1024;

mod apng;
mod assembly;
mod config;
mod preprocess;
mod raw_request;
mod video_sampling;

pub use apng::*;
pub use assembly::*;
pub use config::*;
pub use preprocess::*;
pub use raw_request::*;
pub use video_sampling::*;

#[cfg(test)]
mod tests;
