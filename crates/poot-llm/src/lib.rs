//! poot LLM runner: load a checkpoint and generate on the CPU eager executor and the wgpu, ROCm and
//! PTX device executors.
//!
//! A registered family loads as a [`driver::ModelHandle`] and generates through [`driver::Driver`]. The
//! families not yet registered (MoE and hybrid, POOT-738) load as a `Runner`, which owns their weights
//! and tracer selection and runs their generate loops; the CPU eager path is the oracle that validates
//! the graph IR, decompositions and executors.
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

/// `RunnerError::Msg(format!(...))`. Same argument shape as `anyhow::anyhow!(...)`. Defined at the
/// crate root so it is textually in scope for the child modules (encoder, paged, vlm) declared below.
macro_rules! err {
    ($($arg:tt)*) => { $crate::RunnerError::Msg(format!($($arg)*)) };
}

/// `return Err(RunnerError::Msg(format!(...)))`. Same argument shape as `bail!(...)`.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::RunnerError::Msg(format!($($arg)*))) };
}

mod error;
pub use error::{OptionExt, Result, ResultExt, RunnerError};

/// Shared control result for every single-sequence text-piece callback.
///
/// [`std::ops::ControlFlow::Continue`] continues generation. [`std::ops::ControlFlow::Break`], returned
/// after consuming the current piece, stops before the next model forward; generators then return the
/// tokens produced so far rather than an error.
pub type GenerationControl = std::ops::ControlFlow<()>;

mod architectures;
mod backends;
mod checkpoint;
mod core;
pub mod driver;
mod multimodal;
mod text;

pub use core::cpu_oracle::arm_nan_tap;
pub use multimodal::{encoder, vlm};
pub use text::grammar;

pub use checkpoint::lora::LoraAdapterLease;
pub use core::modality::Modality;
pub use core::runner::Runner;
pub use core::sampler::{Sampler, SamplerFault, TokenLogprob};
pub use multimodal::mrope::{
    MropeBindError, MropeDecodeState, bind_mrope_decode_position, bind_mrope_prefill_positions,
};
pub use multimodal::qwen_vl_chat::{
    QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES, Qwen25VlChatContent, Qwen25VlChatFileUrlError,
    Qwen25VlChatImageDataUrlError, Qwen25VlChatImageFileError, Qwen25VlChatMessage,
    Qwen25VlChatMessageContent, Qwen25VlChatRemoteImageError, Qwen25VlChatRequestError,
};
pub use multimodal::qwen_vl_image::{
    QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS, Qwen25VlApngError, Qwen25VlDecodedFrameSource,
    Qwen25VlDecodedImage, Qwen25VlImageDecodeError, Qwen25VlImagePreprocessError,
    Qwen25VlImageProcessorConfig, Qwen25VlImageProcessorConfigError,
    Qwen25VlImageProcessorConfigValidationError, Qwen25VlPadExpansionError, Qwen25VlProcessedImage,
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
    Qwen25VlPrefillKvEmbedsBindError, Qwen25VlProcessorMedia, Qwen25VlProcessorMediaError,
    Qwen25VlProcessorMediaKind, Qwen25VlProcessorMediaMropeModelInput,
    Qwen25VlProcessorMediaMropeModelInputError, Qwen25VlProcessorMediaMropeModelInputRequest,
    Qwen25VlRenderedPromptMropeModelInput, Qwen25VlRenderedPromptMropeModelInputError,
    Qwen25VlRenderedPromptMropeModelInputRequest, Qwen25VlTextTowerConfigError,
    Qwen25VlVisualEmbedsPrefillBindError, Qwen25VlVisualSpliceMapError, QwenVlMropeMetadata,
    QwenVlMropeMetadataLoadError, QwenVlMropeSourceMetadata, load_qwen_vl_mrope_source_metadata,
};
/// Re-exported so `poot-serve` can resolve an adapter name to a pool index via
/// `LoraAdapterPool::index_of`.
pub use poot_load::lora::LoraAdapterPool;
pub use poot_load::qwen_vl::QwenVlModelType;
pub use poot_models::mrope::{
    MropeConfigMetadata, MropeConfigMetadataError, MropeGrid, MropePosition,
    MropePositionAssemblyError, MropePositionError, MropePositionIds,
    MropePositionSourceAssemblyError, MropeProcessorFpsInput, MropeProcessorSpecialToken,
    MropeProcessorTimingError, MropeSampledFramesPerSecond, MropeSecondsPerGridStep, MropeSegment,
    MropeSpanError, MropeSpecialTokenIds, MropeSpecialTokenKind, MropeSpecialTokenReconcileError,
    MropeTemporalPatchSize, MropeTemporalTokensPerSecond, MropeTimingUnit,
    MropeTokenizerSpecialToken, MropeVideoTiming, QWEN2_5_VL_PROCESSOR_DEFAULT_FPS,
    Qwen25VlMropePositionInput, Qwen25VlMropePositionSourceInput,
    assemble_qwen2_5_vl_mrope_positions, assemble_qwen2_5_vl_mrope_positions_from_sources,
    assemble_qwen2_5_vl_video_timings, assemble_qwen2_5_vl_video_timings_with_default,
    build_mrope_position_ids, discover_mrope_segments, reconcile_mrope_special_token_ids,
};
/// Re-exported so `poot-serve`'s `BatchDecodable` can name the type without depending on `poot-models`.
pub use poot_models::qwen2::LoraBatchedSpec;
pub use text::chat::{RenderedChat, derive_jinja_stops, render_jinja_value};
pub use text::guided::{Constraint, JsonAcceptor, json_schema_to_regex, tool_call_regex};
pub use text::tokenize::{ChatTemplate, TextCodec};

#[cfg(test)]
#[path = "tests/coherence.rs"]
mod coherence_private_tests;

#[cfg(test)]
#[path = "tests/gpu_batched_prefill.rs"]
mod gpu_batched_prefill_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/card234_rocm_batched_decode_gpu.rs"]
mod card234_rocm_batched_decode_gpu_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/moe_rocm_batched_decode_isolation.rs"]
mod moe_rocm_batched_decode_isolation_tests;

#[cfg(test)]
#[path = "tests/card271_dtype_diag.rs"]
mod card271_dtype_diag_tests;

#[cfg(test)]
#[path = "tests/moe_unpooled_greedy_tokens.rs"]
mod moe_unpooled_greedy_tokens_tests;

#[cfg(test)]
#[path = "tests/validation_mapping.rs"]
mod validation_mapping_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/kv_masked_special_arch_rocm.rs"]
mod kv_masked_special_arch_rocm_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/card135d_deepseek2_rocm_reprefill.rs"]
mod card135d_deepseek2_rocm_reprefill_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/card135d_gptoss_rocm_reprefill.rs"]
mod card135d_gptoss_rocm_reprefill_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/card135d_mixtral_rocm_reprefill.rs"]
mod card135d_mixtral_rocm_reprefill_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/card135d_olmoe_rocm_reprefill.rs"]
mod card135d_olmoe_rocm_reprefill_tests;

#[cfg(all(test, feature = "rocm"))]
#[path = "tests/deepseek3_synthetic_rocm_reprefill.rs"]
mod deepseek3_synthetic_rocm_reprefill_tests;

#[cfg(test)]
#[path = "tests/post_prefill_prepare.rs"]
mod post_prefill_prepare_tests;

#[cfg(test)]
#[path = "tests/card551b_sampling_suffix.rs"]
mod card551b_sampling_suffix_tests;
