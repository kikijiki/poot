use std::path::Path;

use poot_models::mrope::{
    MropeGrid, MropePositionError, MropePositionIds, MropeProcessorFpsInput,
    MropeProcessorTimingError, MropeSampledFramesPerSecond, MropeSegment, MropeSpanError,
    MropeTemporalPatchSize, MropeVisualKind, assemble_qwen2_5_vl_video_timings,
    discover_mrope_segments,
};

use crate::multimodal::qwen_vl_mrope::{
    Qwen25VlMropeModelInputView, Qwen25VlProcessorMedia,
    Qwen25VlProcessorMediaMropeModelInputError, Qwen25VlProcessorMediaMropeModelInputRequest,
    QwenVlMropeMetadata, QwenVlMropeMetadataLoadError,
    assemble_qwen2_5_vl_mrope_model_input_from_processor_media_with_metadata_snapshot,
    load_qwen_vl_mrope_metadata_snapshot,
};
use poot_load::qwen_vl::QwenVlModelType;

use super::apng::Qwen25VlDecodedFrameSource;
use super::assembly::{
    Qwen25VlProcessedMedia, Qwen25VlProcessedMediaAssemblyError,
    assemble_qwen2_5_vl_processed_media,
};
use super::config::{Qwen25VlImageProcessorConfigError, load_qwen2_5_vl_image_processor_config};
use super::preprocess::{
    Qwen25VlDecodedImage, Qwen25VlImagePreprocessError, Qwen25VlRawVideoPreprocessError,
    image_layout, preprocess_qwen2_5_vl_images_with_config,
    preprocess_qwen2_5_vl_raw_video_with_config,
};
use super::video_sampling::{Qwen25VlVideoMetadata, Qwen25VlVideoSamplePlan};

pub const QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS: usize = 64;

/// One item in an already-rendered Qwen2.5-VL request's explicit raw media order.
///
/// Images are decoded RGB8 values. Video metadata is already probed; the caller-owned decoded-frame
/// source is supplied separately in video order.
#[derive(Clone, Debug, PartialEq)]
pub enum Qwen25VlRawMediaInput<'a> {
    Image(Qwen25VlDecodedImage),
    Video(Qwen25VlVideoMetadata<'a>),
}

impl Qwen25VlRawMediaInput<'_> {
    pub const fn kind(&self) -> MropeVisualKind {
        match self {
            Self::Image(_) => MropeVisualKind::Image,
            Self::Video(_) => MropeVisualKind::Video,
        }
    }
}

/// Inputs for the owned raw rendered-request media assembler.
///
/// `video_sources` follows video-only order within `media`. All sources share one caller-selected
/// error type (an enum can keep distinct codec errors).
pub struct Qwen25VlRawRequestMropeModelInputRequest<'dir, 'prompt, 'data, 'sources, S> {
    pub model_dir: &'dir Path,
    pub rendered_prompt: &'prompt str,
    pub media: Vec<Qwen25VlRawMediaInput<'data>>,
    pub fps: Option<MropeProcessorFpsInput<'data>>,
    pub video_sources: &'sources mut [S],
}

/// Fully owned raw request assembly result.
///
/// The processor-media, FPS, and model-input views borrow this owner's fields on demand, avoiding a
/// self-referential handoff.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlRawRequestMropeModelInput {
    pub prompt_token_ids: Vec<u32>,
    pub media: Qwen25VlProcessedMedia,
    pub spatial_merge_size: usize,
    pub metadata: QwenVlMropeMetadata,
    pub mrope_positions: MropePositionIds,
    media_order: Vec<MropeVisualKind>,
    video_sample_plans: Vec<Qwen25VlVideoSamplePlan>,
}

impl Qwen25VlRawRequestMropeModelInput {
    pub fn processor_media(&self) -> Qwen25VlProcessorMedia<'_> {
        self.media.processor_media()
    }

    pub fn processor_fps_input(&self) -> MropeProcessorFpsInput<'_> {
        self.media.processor_fps_input()
    }

    pub fn media_order(&self) -> &[MropeVisualKind] {
        &self.media_order
    }

    pub fn video_sample_plans(&self) -> &[Qwen25VlVideoSamplePlan] {
        &self.video_sample_plans
    }
}

impl Qwen25VlMropeModelInputView for Qwen25VlRawRequestMropeModelInput {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn image_grids(&self) -> &[MropeGrid] {
        &self.media.image_grids
    }

    fn video_grids(&self) -> &[MropeGrid] {
        &self.media.video_grids
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }

    fn metadata(&self) -> &QwenVlMropeMetadata {
        &self.metadata
    }

    fn mrope_positions(&self) -> &MropePositionIds {
        &self.mrope_positions
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlPadExpansionError {
    #[error("Qwen2.5-VL {kind} pad marker is empty")]
    EmptyPadMarker { kind: MropeVisualKind },
    #[error(
        "Qwen2.5-VL rendered prompt has {actual} {kind} pads; processed media has {expected} {kind} grids"
    )]
    PadCountMismatch {
        kind: MropeVisualKind,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Qwen2.5-VL {kind} pad expansion at grid {index} overflows repeating {count} placeholders"
    )]
    ReplacementOverflow {
        kind: MropeVisualKind,
        index: usize,
        count: usize,
    },
    #[error(transparent)]
    Grid(#[from] MropePositionError),
}

const QWEN2_5_VL_PAD_EXPANSION_SENTINEL: &str = "<|placeholder|>";

fn expand_qwen2_5_vl_kind_pads(
    rendered: &str,
    kind: MropeVisualKind,
    pad: &str,
    grids: &[MropeGrid],
    merge_size: usize,
) -> Result<String, Qwen25VlPadExpansionError> {
    if grids.is_empty() {
        let actual = if pad.is_empty() {
            0
        } else {
            rendered.matches(pad).count()
        };
        return if actual == 0 {
            Ok(rendered.to_string())
        } else {
            Err(Qwen25VlPadExpansionError::PadCountMismatch {
                kind,
                expected: 0,
                actual,
            })
        };
    }
    if pad.is_empty() {
        return Err(Qwen25VlPadExpansionError::EmptyPadMarker { kind });
    }
    let actual = rendered.matches(pad).count();
    if actual != grids.len() {
        return Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind,
            expected: grids.len(),
            actual,
        });
    }
    let mut expanded = rendered.to_string();
    for (index, grid) in grids.iter().copied().enumerate() {
        let count = grid.merged_token_count(merge_size, kind)?;
        let sentinel_len = QWEN2_5_VL_PAD_EXPANSION_SENTINEL
            .len()
            .checked_mul(count)
            .ok_or(Qwen25VlPadExpansionError::ReplacementOverflow { kind, index, count })?;
        let mut replacement = String::new();
        replacement
            .try_reserve_exact(sentinel_len)
            .map_err(|_| Qwen25VlPadExpansionError::ReplacementOverflow { kind, index, count })?;
        for _ in 0..count {
            replacement.push_str(QWEN2_5_VL_PAD_EXPANSION_SENTINEL);
        }
        expanded = expanded.replacen(pad, &replacement, 1);
    }
    Ok(expanded.replace(QWEN2_5_VL_PAD_EXPANSION_SENTINEL, pad))
}

/// Expand chat-template image/video pads from merged-grid counts.
///
/// As in Transformers 4.49.0: the first pad occurrence is replaced by a sentinel repeated
/// `prod(grid) // merge_size**2` times, then the original pad marker is restored. Image pads expand
/// before video pads.
pub(crate) fn expand_qwen2_5_vl_rendered_pads(
    rendered: &str,
    image_pad: &str,
    video_pad: &str,
    image_grids: &[MropeGrid],
    video_grids: &[MropeGrid],
    merge_size: usize,
) -> Result<String, Qwen25VlPadExpansionError> {
    let with_images = expand_qwen2_5_vl_kind_pads(
        rendered,
        MropeVisualKind::Image,
        image_pad,
        image_grids,
        merge_size,
    )?;
    expand_qwen2_5_vl_kind_pads(
        &with_images,
        MropeVisualKind::Video,
        video_pad,
        video_grids,
        merge_size,
    )
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlRawRequestAssemblyError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    #[error("Qwen2.5-VL raw request does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error(transparent)]
    ProcessorConfig(#[from] Qwen25VlImageProcessorConfigError),
    #[error("Qwen2.5-VL raw request has {actual} media items, exceeding the {limit}-item cap")]
    TooManyMediaItems { actual: usize, limit: usize },
    #[error(
        "Qwen2.5-VL raw request has {video_count} videos but {source_count} decoded-frame sources"
    )]
    VideoSourceCountMismatch {
        video_count: usize,
        source_count: usize,
    },
    #[error("Qwen2.5-VL raw request with {video_count} videos requires explicit FPS")]
    MissingVideoFps { video_count: usize },
    #[error("Qwen2.5-VL raw request without videos must not supply FPS")]
    UnexpectedVideoFps,
    #[error(transparent)]
    VideoTiming(#[from] MropeProcessorTimingError),
    #[error("preprocess Qwen2.5-VL raw image source {source_index}: {source}")]
    ImagePreprocess {
        source_index: usize,
        #[source]
        source: Qwen25VlImagePreprocessError,
    },
    #[error("preprocess Qwen2.5-VL raw image batch: {0}")]
    ImageBatchPreprocess(#[source] Qwen25VlImagePreprocessError),
    #[error("process Qwen2.5-VL raw video source {source_index} (video {video_index}): {source}")]
    VideoPreprocess {
        source_index: usize,
        video_index: usize,
        #[source]
        source: Qwen25VlRawVideoPreprocessError<E>,
    },
    #[error(transparent)]
    ProcessedMedia(#[from] Qwen25VlProcessedMediaAssemblyError),
    #[error(transparent)]
    ModelInput(#[from] Qwen25VlProcessorMediaMropeModelInputError),
    #[error("rediscover Qwen2.5-VL raw request marker order: {0}")]
    MarkerSpan(#[source] MropeSpanError),
    #[error(
        "Qwen2.5-VL rendered marker {marker_index} is {actual}; raw media source order requires {expected}"
    )]
    MediaOrderMismatch {
        marker_index: usize,
        expected: MropeVisualKind,
        actual: MropeVisualKind,
    },
    #[error(
        "Qwen2.5-VL rendered prompt has {marker_count} visual markers; raw request has {raw_count} media sources"
    )]
    MediaCountMismatch {
        raw_count: usize,
        marker_count: usize,
    },
    #[error("Qwen2.5-VL raw request could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
    #[error(transparent)]
    PadExpansion(#[from] Qwen25VlPadExpansionError),
}

/// Assemble an already-rendered Qwen2.5-VL prompt and explicit ordered raw media into one owned host handoff.
///
/// CPU-only. Images are already decoded; video probing/decoding stays with the caller. Chat templates
/// emit one pad per visual item; this expands them from merged-grid counts after media processing and
/// before tokenization. Nothing is published until the processor-media, rendered-prompt, span, and
/// position paths succeed and the rendered marker order matches `media`. Calls to caller-owned
/// decoded-frame sources are observable and cannot be rolled back if a later phase fails.
pub fn assemble_qwen2_5_vl_raw_request_mrope_model_input<S>(
    input: Qwen25VlRawRequestMropeModelInputRequest<'_, '_, '_, '_, S>,
) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlRawRequestAssemblyError<S::Error>>
where
    S: Qwen25VlDecodedFrameSource,
{
    let Qwen25VlRawRequestMropeModelInputRequest {
        model_dir,
        rendered_prompt,
        media,
        fps,
        video_sources,
    } = input;

    // Reconcile marker strings/IDs and retain the tokenizer before processor work; one snapshot stops
    // caller-owned decoders from changing checkpoint files between preflight and prompt assembly.
    let metadata_snapshot = load_qwen_vl_mrope_metadata_snapshot(model_dir)?;
    if metadata_snapshot.metadata().model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlRawRequestAssemblyError::UnsupportedModelType {
            model_type: metadata_snapshot.metadata().model_type,
        });
    }
    let processor_config = load_qwen2_5_vl_image_processor_config(model_dir)?;

    if media.len() > QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS {
        return Err(Qwen25VlRawRequestAssemblyError::TooManyMediaItems {
            actual: media.len(),
            limit: QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS,
        });
    }
    let video_count = media
        .iter()
        .filter(|item| matches!(item, Qwen25VlRawMediaInput::Video(_)))
        .count();
    if video_sources.len() != video_count {
        return Err(Qwen25VlRawRequestAssemblyError::VideoSourceCountMismatch {
            video_count,
            source_count: video_sources.len(),
        });
    }

    let temporal_patch_size = MropeTemporalPatchSize::new(processor_config.temporal_patch_size())
        .expect("validated processor config must have a nonzero temporal patch size");
    let explicit_fps = match (video_count, fps) {
        (0, None) => None,
        (0, Some(_)) => return Err(Qwen25VlRawRequestAssemblyError::UnexpectedVideoFps),
        (count, None) => {
            return Err(Qwen25VlRawRequestAssemblyError::MissingVideoFps { video_count: count });
        }
        (_, Some(fps)) => {
            // This adapter owns scalar broadcast, per-video count/value validation and model timing
            // requirements; the final position handoff reconstructs the results.
            let _ = assemble_qwen2_5_vl_video_timings(
                video_count,
                temporal_patch_size,
                fps,
                &metadata_snapshot.metadata().config,
            )?;
            Some(fps)
        }
    };

    let image_count = media.len() - video_count;
    let mut images = Vec::new();
    images.try_reserve_exact(image_count).map_err(|_| {
        Qwen25VlRawRequestAssemblyError::Allocation {
            stage: "decoded images",
        }
    })?;
    let mut videos = Vec::new();
    videos.try_reserve_exact(video_count).map_err(|_| {
        Qwen25VlRawRequestAssemblyError::Allocation {
            stage: "video metadata",
        }
    })?;
    let mut media_order = Vec::new();
    media_order.try_reserve_exact(media.len()).map_err(|_| {
        Qwen25VlRawRequestAssemblyError::Allocation {
            stage: "raw media order",
        }
    })?;

    for (source_index, item) in media.into_iter().enumerate() {
        let kind = item.kind();
        media_order.push(kind);
        match item {
            Qwen25VlRawMediaInput::Image(image) => {
                // Preflight every source geometry so an image failure keeps the raw source index; the
                // batch call below can then fail only in aggregate count/allocation stages.
                image_layout(&image, &processor_config).map_err(|source| {
                    Qwen25VlRawRequestAssemblyError::ImagePreprocess {
                        source_index,
                        source,
                    }
                })?;
                images.push(image);
            }
            Qwen25VlRawMediaInput::Video(metadata) => videos.push((source_index, metadata)),
        }
    }

    let processed_images = if images.is_empty() {
        None
    } else {
        Some(
            preprocess_qwen2_5_vl_images_with_config(&images, &processor_config)
                .map_err(Qwen25VlRawRequestAssemblyError::ImageBatchPreprocess)?,
        )
    };

    let mut processed_videos = Vec::new();
    processed_videos
        .try_reserve_exact(video_count)
        .map_err(|_| Qwen25VlRawRequestAssemblyError::Allocation {
            stage: "processed videos",
        })?;
    let mut video_sample_plans = Vec::new();
    video_sample_plans
        .try_reserve_exact(video_count)
        .map_err(|_| Qwen25VlRawRequestAssemblyError::Allocation {
            stage: "video sample plans",
        })?;

    for (video_index, ((source_index, metadata), source)) in
        videos.into_iter().zip(video_sources.iter_mut()).enumerate()
    {
        let value = match explicit_fps.expect("positive video count requires explicit FPS") {
            MropeProcessorFpsInput::Scalar(value) => value,
            MropeProcessorFpsInput::PerVideo(values) => values[video_index],
        };
        let sampled_fps = MropeSampledFramesPerSecond::new(value)
            .expect("existing timing assembly validated every explicit FPS value");
        let sampled = preprocess_qwen2_5_vl_raw_video_with_config(
            metadata,
            sampled_fps,
            source,
            &processor_config,
        )
        .map_err(|source| Qwen25VlRawRequestAssemblyError::VideoPreprocess {
            source_index,
            video_index,
            source,
        })?;
        video_sample_plans.push(sampled.sample_plan);
        processed_videos.push(sampled.processed_video);
    }

    let processed_media =
        assemble_qwen2_5_vl_processed_media(processed_images, processed_videos, &processor_config)?;
    let expanded_prompt = expand_qwen2_5_vl_rendered_pads(
        rendered_prompt,
        metadata_snapshot.image_pad(),
        metadata_snapshot.video_pad(),
        &processed_media.image_grids,
        &processed_media.video_grids,
        processor_config.merge_size(),
    )?;
    let assembled =
        assemble_qwen2_5_vl_mrope_model_input_from_processor_media_with_metadata_snapshot(
            Qwen25VlProcessorMediaMropeModelInputRequest {
                model_dir,
                rendered_prompt: &expanded_prompt,
                media: processed_media.processor_media(),
                spatial_merge_size: processor_config.merge_size(),
                temporal_patch_size,
                fps: Some(processed_media.processor_fps_input()).filter(|_| video_count != 0),
            },
            metadata_snapshot,
        )?;

    let segments = discover_mrope_segments(
        &assembled.prompt_token_ids,
        assembled.metadata.special_tokens,
        assembled.media.image_grids,
        assembled.media.video_grids,
        assembled.spatial_merge_size,
    )
    .map_err(Qwen25VlRawRequestAssemblyError::MarkerSpan)?;
    validate_qwen2_5_vl_raw_media_order::<S::Error>(&media_order, &segments)?;

    let crate::multimodal::qwen_vl_mrope::Qwen25VlProcessorMediaMropeModelInput {
        prompt_token_ids,
        media: _,
        spatial_merge_size,
        metadata,
        mrope_positions,
    } = assembled;
    Ok(Qwen25VlRawRequestMropeModelInput {
        prompt_token_ids,
        media: processed_media,
        spatial_merge_size,
        metadata,
        mrope_positions,
        media_order,
        video_sample_plans,
    })
}

pub(super) fn validate_qwen2_5_vl_raw_media_order<E>(
    raw_order: &[MropeVisualKind],
    segments: &[MropeSegment],
) -> Result<(), Qwen25VlRawRequestAssemblyError<E>>
where
    E: std::error::Error + 'static,
{
    let rendered_order: Vec<MropeVisualKind> = segments
        .iter()
        .filter_map(|segment| match segment {
            MropeSegment::Image(_) => Some(MropeVisualKind::Image),
            MropeSegment::Video(_) | MropeSegment::TimestampedVideo { .. } => {
                Some(MropeVisualKind::Video)
            }
            MropeSegment::Text(_) => None,
        })
        .collect();
    if rendered_order.len() != raw_order.len() {
        return Err(Qwen25VlRawRequestAssemblyError::MediaCountMismatch {
            raw_count: raw_order.len(),
            marker_count: rendered_order.len(),
        });
    }
    for (marker_index, (expected, actual)) in
        raw_order.iter().copied().zip(rendered_order).enumerate()
    {
        if expected != actual {
            return Err(Qwen25VlRawRequestAssemblyError::MediaOrderMismatch {
                marker_index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}
