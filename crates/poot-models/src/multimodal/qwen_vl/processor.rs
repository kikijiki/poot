//! CPU-only Qwen2.5-VL image and video preprocessing.
//!
//! The output is one processor-compatible flattened-patch tensor plus ordered raw grids. Prompt
//! semantics, container decoding and vision tower execution are out of scope: a video's frames come from
//! the caller's [`FrameSource`](crate::multimodal::front_end::FrameSource).
//! Cubic resize uses the `image` crate: deterministic for the pinned dependency, but not bit-identical to
//! Pillow for every resized pixel.

const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;

const MAX_VIDEO_FRAMES: usize = 16 * 1024;

mod assembly;
mod config;
mod preprocess;
mod video_sampling;

pub(super) use assembly::{
    Qwen25VlProcessedImages, Qwen25VlProcessedMedia, Qwen25VlProcessedMediaAssemblyError,
    Qwen25VlProcessedVideo, assemble_qwen2_5_vl_processed_media,
};
pub(super) use config::Qwen25VlImageProcessorConfig;
pub(super) use preprocess::{
    Qwen25VlImagePreprocessError, Qwen25VlRawVideoPreprocessError,
    preprocess_qwen2_5_vl_images_with_config, preprocess_qwen2_5_vl_raw_video_with_config,
};

#[cfg(test)]
mod tests;
