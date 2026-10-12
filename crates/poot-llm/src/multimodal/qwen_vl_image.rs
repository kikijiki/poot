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

pub use apng::{
    Qwen25VlApng, Qwen25VlApngError, Qwen25VlApngFrameSource, Qwen25VlDecodedFrameSource,
};
pub use assembly::{
    Qwen25VlProcessedImages, Qwen25VlProcessedMedia, Qwen25VlProcessedMediaAssemblyError,
    Qwen25VlProcessedVideo, assemble_qwen2_5_vl_processed_media,
};
pub use config::{
    Qwen25VlImageProcessorConfig, Qwen25VlImageProcessorConfigError,
    Qwen25VlImageProcessorConfigValidationError, load_qwen2_5_vl_image_processor_config,
};
pub use preprocess::{
    Qwen25VlDecodedImage, Qwen25VlImageDecodeError, Qwen25VlImagePreprocessError,
    Qwen25VlRawVideoPreprocessError, Qwen25VlVideoPreprocessError, decode_qwen2_5_vl_image,
    preprocess_qwen2_5_vl_images_with_config, preprocess_qwen2_5_vl_raw_video_with_config,
    preprocess_qwen2_5_vl_video_with_config,
};
pub use raw_request::{
    QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS, Qwen25VlPadExpansionError, Qwen25VlRawMediaInput,
    Qwen25VlRawRequestAssemblyError, Qwen25VlRawRequestMropeModelInput,
    Qwen25VlRawRequestMropeModelInputRequest, assemble_qwen2_5_vl_raw_request_mrope_model_input,
};
pub use video_sampling::{
    Qwen25VlSampledVideo, Qwen25VlVideoMetadata, Qwen25VlVideoSamplePlan,
    Qwen25VlVideoSamplingError, plan_qwen2_5_vl_video_sampling,
};

#[cfg(test)]
mod tests;
