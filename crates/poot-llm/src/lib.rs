//! poot LLM driver: load a checkpoint and generate on the wgpu, ROCm and PTX device executors.
//!
//! A checkpoint loads as a [`driver::ModelHandle`] (its family from the registry, its weights as
//! stored, its text services) and generates through [`driver::Driver`], which also runs `Encode`
//! steps (encoder embeddings and reranking, a vision tower) and prompts with embedding rows in place
//! of tokens' own. A multimodal checkpoint's front end ([`front_end`]) drives the towers' drivers.
//!
//! # The graph passes are sealed behind `compile` (Card 626, SC-001)
//!
//! A consumer crate cannot name a pass or sequence its own pipeline: `poot_graph_plan::passes` is
//! private (E0603), and the pre-card-626 home `poot_graph_ir::transform` no longer exists (E0433).
//!
//! ```compile_fail,E0603
//! let g = poot_graph_ir::Graph::default();
//! poot_graph_plan::passes::cse(&g);
//! ```
//!
//! ```compile_fail,E0433
//! let g = poot_graph_ir::Graph::default();
//! poot_graph_ir::transform::cse(&g);
//! ```
//!
//! # There is no `Runner`
//!
//! The per-family `Runner` that owned weights and tracer selection is gone: every family is a
//! registered `Model` on the driver.
//!
//! ```compile_fail,E0432
//! use poot_llm::Runner; // deleted
//! ```

/// `TextError::Msg(format!(...))`. Same argument shape as `anyhow::anyhow!(...)`. Defined at the
/// crate root so it is textually in scope for the child modules declared below.
macro_rules! err {
    ($($arg:tt)*) => { $crate::TextError::Msg(format!($($arg)*)) };
}

/// `return Err(TextError::Msg(format!(...)))`. Same argument shape as `bail!(...)`.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::TextError::Msg(format!($($arg)*))) };
}

mod error;
pub use error::{ChatTemplateError, OptionExt, Result, TextError};

/// Shared control result for every single-sequence text-piece callback.
///
/// [`std::ops::ControlFlow::Continue`] continues generation. [`std::ops::ControlFlow::Break`], returned
/// after consuming the current piece, stops before the next model forward; generators then return the
/// tokens produced so far rather than an error.
pub type GenerationControl = std::ops::ControlFlow<()>;

mod core;
pub mod driver;
mod multimodal;
mod text;

pub use multimodal::{
    ChatTurn, ImageCheckpoint, ImageCheckpointError, ImageFrontEnd, ImageRequestError,
    MultimodalPrompt, encoders, front_end, qwen_vl, read_image_checkpoint,
};
pub use text::grammar;

pub use core::modality::Modality;
pub use core::sampler::{Sampler, SamplerFault, TokenLogprob};
pub use multimodal::qwen_vl_chat::{
    Qwen25VlChatContent, Qwen25VlChatImageFileError, Qwen25VlChatMessage,
    Qwen25VlChatMessageContent, Qwen25VlChatRequestError,
};
pub use multimodal::qwen_vl_image::{
    QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS, Qwen25VlApng, Qwen25VlApngError, Qwen25VlApngFrameSource,
    Qwen25VlDecodedFrameSource, Qwen25VlDecodedImage, Qwen25VlImageDecodeError,
    Qwen25VlImagePreprocessError, Qwen25VlImageProcessorConfig, Qwen25VlImageProcessorConfigError,
    Qwen25VlImageProcessorConfigValidationError, Qwen25VlPadExpansionError,
    Qwen25VlProcessedImages, Qwen25VlProcessedMedia, Qwen25VlProcessedMediaAssemblyError,
    Qwen25VlProcessedVideo, Qwen25VlRawMediaInput, Qwen25VlRawRequestAssemblyError,
    Qwen25VlRawRequestMropeModelInput, Qwen25VlRawRequestMropeModelInputRequest,
    Qwen25VlRawVideoPreprocessError, Qwen25VlSampledVideo, Qwen25VlVideoMetadata,
    Qwen25VlVideoPreprocessError, Qwen25VlVideoSamplePlan, Qwen25VlVideoSamplingError,
    assemble_qwen2_5_vl_processed_media, assemble_qwen2_5_vl_raw_request_mrope_model_input,
    decode_qwen2_5_vl_image, load_qwen2_5_vl_image_processor_config,
    plan_qwen2_5_vl_video_sampling, preprocess_qwen2_5_vl_images_with_config,
    preprocess_qwen2_5_vl_raw_video_with_config, preprocess_qwen2_5_vl_video_with_config,
};
pub use multimodal::qwen_vl_mrope::{
    Qwen25VlMropeModelInput, Qwen25VlMropeModelInputError, Qwen25VlMropeModelInputView,
    Qwen25VlProcessorMedia, Qwen25VlProcessorMediaError, Qwen25VlProcessorMediaKind,
    Qwen25VlProcessorMediaMropeModelInput, Qwen25VlProcessorMediaMropeModelInputError,
    Qwen25VlProcessorMediaMropeModelInputRequest, Qwen25VlRenderedPromptMropeModelInput,
    Qwen25VlRenderedPromptMropeModelInputError, Qwen25VlRenderedPromptMropeModelInputRequest,
    QwenVlMropeMetadata, QwenVlMropeMetadataLoadError, QwenVlMropeSourceMetadata,
    load_qwen_vl_mrope_source_metadata,
};
pub use poot_load::qwen_vl::QwenVlModelType;
pub use poot_models::mrope::{
    MropeConfigMetadata, MropeConfigMetadataError, MropeGrid, MropePosition,
    MropePositionAssemblyError, MropePositionError, MropePositionIds, MropeProcessorFpsInput,
    MropeProcessorSpecialToken, MropeProcessorTimingError, MropeSampledFramesPerSecond,
    MropeSecondsPerGridStep, MropeSegment, MropeSpanError, MropeSpecialTokenIds,
    MropeSpecialTokenKind, MropeSpecialTokenReconcileError, MropeTemporalPatchSize,
    MropeTemporalTokensPerSecond, MropeTimingUnit, MropeTokenizerSpecialToken, MropeVideoTiming,
    QWEN2_5_VL_PROCESSOR_DEFAULT_FPS, Qwen25VlMropePositionInput,
    assemble_qwen2_5_vl_mrope_positions, assemble_qwen2_5_vl_video_timings,
    assemble_qwen2_5_vl_video_timings_with_default, build_mrope_position_ids,
    discover_mrope_segments, reconcile_mrope_special_token_ids,
};
pub use text::chat::{
    ChatTemplateOverride, CheckedChatTemplate, RenderedChat, Speaker, derive_jinja_stops,
    render_jinja_value,
};
pub use text::guided::{Constraint, JsonAcceptor, json_schema_to_regex, tool_call_regex};
pub use text::tokenize::{ChatTemplate, TextCodec};
