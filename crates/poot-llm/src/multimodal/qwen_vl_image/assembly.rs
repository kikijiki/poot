use poot_models::mrope::{
    MropeGrid, MropePositionError, MropeProcessorFpsInput, MropeProcessorTimingError,
    MropeSampledFramesPerSecond, MropeSecondsPerGridStep, MropeTemporalPatchSize, MropeVisualKind,
};
use poot_tensor::{DType, HostTensor};

use super::MAX_VIDEO_FRAMES;
use super::config::Qwen25VlImageProcessorConfig;
use super::preprocess::{Qwen25VlTemporalLayoutError, qwen25_vl_temporal_layout};
use crate::checkpoint::gguf::permute::gather_from;
use crate::multimodal::qwen_vl_mrope::Qwen25VlProcessorMedia;

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlProcessedImage {
    pub pixel_values: HostTensor,
    pub grid: MropeGrid,
}

/// Ordered Qwen2.5-VL image processor output, concatenated in processor item order.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlProcessedImages {
    pub pixel_values: HostTensor,
    pub grids: Vec<MropeGrid>,
}

impl Qwen25VlProcessedImages {
    /// Borrow this batch as the image-only portion of the existing processed-media handoff.
    pub fn processor_media(&self) -> Qwen25VlProcessorMedia<'_> {
        Qwen25VlProcessorMedia {
            pixel_values: Some(&self.pixel_values),
            image_grids: &self.grids,
            pixel_values_videos: None,
            video_grids: &[],
        }
    }
}

/// Processor-compatible output for one ordered set of caller-sampled decoded video frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlProcessedVideo {
    pub pixel_values_videos: HostTensor,
    pub grid: MropeGrid,
    pub sampled_frames_per_second: MropeSampledFramesPerSecond,
    pub seconds_per_grid_step: MropeSecondsPerGridStep,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub sampled_frame_count: usize,
    pub padded_frame_count: usize,
}

impl Qwen25VlProcessedVideo {
    /// Borrow this result as the video-only portion of the existing processed-media handoff.
    pub fn processor_media(&self) -> Qwen25VlProcessorMedia<'_> {
        Qwen25VlProcessorMedia {
            pixel_values: None,
            image_grids: &[],
            pixel_values_videos: Some(&self.pixel_values_videos),
            video_grids: std::slice::from_ref(&self.grid),
        }
    }

    /// Return the processor's owned scalar FPS shape for the existing Qwen2.5-VL timing handoff.
    ///
    /// The scalar owns its `f64`, so the returned value does not borrow this processed-video result.
    pub fn processor_fps_input(&self) -> MropeProcessorFpsInput<'static> {
        MropeProcessorFpsInput::Scalar(self.sampled_frames_per_second.get())
    }
}

/// Owned Qwen2.5-VL processor output assembled from already-processed media.
///
/// Images and videos stay separate tensors; their cross-kind order is carried only by the rendered
/// prompt and marker-span validation.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlProcessedMedia {
    pub pixel_values: Option<HostTensor>,
    pub image_grids: Vec<MropeGrid>,
    pub pixel_values_videos: Option<HostTensor>,
    pub video_grids: Vec<MropeGrid>,
    pub(crate) temporal_patch_size: MropeTemporalPatchSize,
    pub(crate) sampled_frames_per_second: Vec<f64>,
}

impl Qwen25VlProcessedMedia {
    /// Borrow the owned per-kind tensors and grids through the existing processor-media handoff.
    pub fn processor_media(&self) -> Qwen25VlProcessorMedia<'_> {
        Qwen25VlProcessorMedia {
            pixel_values: self.pixel_values.as_ref(),
            image_grids: &self.image_grids,
            pixel_values_videos: self.pixel_values_videos.as_ref(),
            video_grids: &self.video_grids,
        }
    }

    /// Borrow the retained per-video sampled FPS values through the existing timing-input shape.
    pub fn processor_fps_input(&self) -> MropeProcessorFpsInput<'_> {
        MropeProcessorFpsInput::PerVideo(&self.sampled_frames_per_second)
    }

    pub fn temporal_patch_size(&self) -> MropeTemporalPatchSize {
        self.temporal_patch_size
    }

    pub fn sampled_frames_per_second(&self) -> &[f64] {
        &self.sampled_frames_per_second
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlProcessedMediaAssemblyError {
    #[error("Qwen2.5-VL processed-media feature width overflows usize")]
    FeatureWidthOverflow,
    #[error("invalid Qwen2.5-VL processed image batch: {0}")]
    ImageMedia(#[source] crate::multimodal::qwen_vl_mrope::Qwen25VlProcessorMediaError),
    #[error(
        "Qwen2.5-VL processed image feature width is {actual}; processor config requires {expected}"
    )]
    ImageFeatureWidth { expected: usize, actual: usize },
    #[error("invalid Qwen2.5-VL processed image grid {index}: {source}")]
    ImageGrid {
        index: usize,
        #[source]
        source: MropePositionError,
    },
    #[error("Qwen2.5-VL processed image grid {index} has temporal dimension {actual}; expected 1")]
    ImageGridTemporal { index: usize, actual: usize },
    #[error("invalid Qwen2.5-VL processed video {index}: {source}")]
    VideoMedia {
        index: usize,
        #[source]
        source: crate::multimodal::qwen_vl_mrope::Qwen25VlProcessorMediaError,
    },
    #[error(
        "Qwen2.5-VL processed video {index} feature width is {actual}; processor config requires {expected}"
    )]
    VideoFeatureWidth {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Qwen2.5-VL processed video {index} temporal patch size is {actual}; processor config requires {expected}"
    )]
    VideoTemporalPatchSize {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("Qwen2.5-VL processed video {index} has no sampled frames")]
    VideoEmptySampledFrames { index: usize },
    #[error(
        "Qwen2.5-VL processed video {index} has {actual} sampled frames, exceeding the {limit}-frame cap"
    )]
    VideoTooManySampledFrames {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("Qwen2.5-VL processed video {index} padded frame count overflows usize")]
    VideoPaddedFrameCountOverflow { index: usize },
    #[error(
        "Qwen2.5-VL processed video {index} sampled count {sampled_frame_count} with temporal patch size {temporal_patch_size} pads to incompatible count {padded_frame_count}"
    )]
    VideoIncompatibleTemporalPadding {
        index: usize,
        sampled_frame_count: usize,
        temporal_patch_size: usize,
        padded_frame_count: usize,
    },
    #[error(
        "Qwen2.5-VL processed video {index} has {sampled_frame_count} sampled frames and derived padded count {padded_frame_count}, exceeding the {limit}-frame cap"
    )]
    VideoTooManyPaddedFrames {
        index: usize,
        sampled_frame_count: usize,
        padded_frame_count: usize,
        limit: usize,
    },
    #[error(
        "Qwen2.5-VL processed video {index} records padded frame count {actual}; sampled count and temporal patch size require {expected}"
    )]
    VideoPaddedFrameCount {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("invalid Qwen2.5-VL processed video grid {index}: {source}")]
    VideoGrid {
        index: usize,
        #[source]
        source: MropePositionError,
    },
    #[error(
        "Qwen2.5-VL processed video {index} grid temporal dimension is {actual}; padded frame count requires {expected}"
    )]
    VideoGridTemporal {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("validate Qwen2.5-VL processed video {index} timing: {source}")]
    VideoTiming {
        index: usize,
        #[source]
        source: MropeProcessorTimingError,
    },
    #[error(
        "Qwen2.5-VL processed video {index} seconds per grid step has bits {actual:#018x}; expected {expected:#018x} from temporal patch size and sampled FPS"
    )]
    VideoSecondsPerGridStep {
        index: usize,
        expected: u64,
        actual: u64,
    },
    #[error("Qwen2.5-VL combined processed-video row count overflows usize at video {index}")]
    VideoRowCountOverflow { index: usize },
    #[error("Qwen2.5-VL combined processed-video element count overflows usize")]
    VideoElementCountOverflow,
    #[error("Qwen2.5-VL combined processed-video byte count overflows usize")]
    VideoByteCountOverflow,
    #[error("Qwen2.5-VL processed video {index} has dtype {actual}; expected {expected}")]
    VideoDtypeMismatch {
        index: usize,
        expected: DType,
        actual: DType,
    },
    #[error("Qwen2.5-VL processed-media assembly could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Qwen25VlCombinedVideoLayout {
    pub(crate) rows: usize,
    pub(crate) feature_width: usize,
    pub(crate) elements: usize,
    pub(crate) dtype: DType,
}

pub(super) struct Qwen25VlAssembledVideos {
    pub(crate) pixel_values: Option<HostTensor>,
    pub(crate) grids: Vec<MropeGrid>,
    pub(crate) sampled_frames_per_second: Vec<f64>,
}

/// Assemble already-processed Qwen2.5-VL image and video outputs into owned per-kind processor tensors.
///
/// Consumes its sources. An image batch and a single video tensor are moved without copying; two or
/// more videos are concatenated here. All validation and count arithmetic complete before any output
/// reservation or copy.
pub fn assemble_qwen2_5_vl_processed_media(
    images: Option<Qwen25VlProcessedImages>,
    videos: Vec<Qwen25VlProcessedVideo>,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlProcessedMedia, Qwen25VlProcessedMediaAssemblyError> {
    let expected_feature_width = qwen25_vl_processor_feature_width(config)?;
    let temporal_patch_size = MropeTemporalPatchSize::new(config.temporal_patch_size)
        .expect("validated image processor temporal patch size must be nonzero");

    if let Some(images) = &images {
        images
            .processor_media()
            .validate()
            .map_err(Qwen25VlProcessedMediaAssemblyError::ImageMedia)?;
        let actual = images.pixel_values.shape()[1];
        if actual != expected_feature_width {
            return Err(Qwen25VlProcessedMediaAssemblyError::ImageFeatureWidth {
                expected: expected_feature_width,
                actual,
            });
        }
        for (index, grid) in images.grids.iter().copied().enumerate() {
            validate_qwen25_vl_processed_grid(MropeVisualKind::Image, grid, config.merge_size)
                .map_err(|source| Qwen25VlProcessedMediaAssemblyError::ImageGrid {
                    index,
                    source,
                })?;
            if grid.temporal != 1 {
                return Err(Qwen25VlProcessedMediaAssemblyError::ImageGridTemporal {
                    index,
                    actual: grid.temporal,
                });
            }
        }
    }

    for (index, video) in videos.iter().enumerate() {
        video
            .processor_media()
            .validate()
            .map_err(|source| Qwen25VlProcessedMediaAssemblyError::VideoMedia { index, source })?;
        let actual = video.pixel_values_videos.shape()[1];
        if actual != expected_feature_width {
            return Err(Qwen25VlProcessedMediaAssemblyError::VideoFeatureWidth {
                index,
                expected: expected_feature_width,
                actual,
            });
        }
        if video.temporal_patch_size != temporal_patch_size {
            return Err(
                Qwen25VlProcessedMediaAssemblyError::VideoTemporalPatchSize {
                    index,
                    expected: temporal_patch_size.get(),
                    actual: video.temporal_patch_size.get(),
                },
            );
        }
        if video.sampled_frame_count == 0 {
            return Err(Qwen25VlProcessedMediaAssemblyError::VideoEmptySampledFrames { index });
        }
        if video.sampled_frame_count > MAX_VIDEO_FRAMES {
            return Err(
                Qwen25VlProcessedMediaAssemblyError::VideoTooManySampledFrames {
                    index,
                    actual: video.sampled_frame_count,
                    limit: MAX_VIDEO_FRAMES,
                },
            );
        }
        let temporal_layout =
            qwen25_vl_temporal_layout(video.sampled_frame_count, video.temporal_patch_size.get())
                .map_err(|source| processed_media_temporal_layout_error(index, source))?;
        if video.padded_frame_count != temporal_layout.padded_frame_count {
            return Err(Qwen25VlProcessedMediaAssemblyError::VideoPaddedFrameCount {
                index,
                expected: temporal_layout.padded_frame_count,
                actual: video.padded_frame_count,
            });
        }
        validate_qwen25_vl_processed_grid(MropeVisualKind::Video, video.grid, config.merge_size)
            .map_err(|source| Qwen25VlProcessedMediaAssemblyError::VideoGrid { index, source })?;
        if video.grid.temporal != temporal_layout.grid_temporal {
            return Err(Qwen25VlProcessedMediaAssemblyError::VideoGridTemporal {
                index,
                expected: temporal_layout.grid_temporal,
                actual: video.grid.temporal,
            });
        }
        let expected_seconds = MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
            video.temporal_patch_size,
            video.sampled_frames_per_second,
        )
        .map_err(|source| Qwen25VlProcessedMediaAssemblyError::VideoTiming { index, source })?;
        if video.seconds_per_grid_step.get().to_bits() != expected_seconds.get().to_bits() {
            return Err(
                Qwen25VlProcessedMediaAssemblyError::VideoSecondsPerGridStep {
                    index,
                    expected: expected_seconds.get().to_bits(),
                    actual: video.seconds_per_grid_step.get().to_bits(),
                },
            );
        }
    }
    let combined_video = qwen25_vl_combined_video_layout(
        videos.iter().map(|video| {
            (
                video.pixel_values_videos.shape()[0],
                video.pixel_values_videos.dtype(),
            )
        }),
        expected_feature_width,
    )?;

    let (pixel_values, image_grids) = match images {
        Some(images) => (Some(images.pixel_values), images.grids),
        None => (None, Vec::new()),
    };
    let assembled_videos = assemble_qwen25_vl_videos(videos, combined_video)?;

    Ok(Qwen25VlProcessedMedia {
        pixel_values,
        image_grids,
        pixel_values_videos: assembled_videos.pixel_values,
        video_grids: assembled_videos.grids,
        temporal_patch_size,
        sampled_frames_per_second: assembled_videos.sampled_frames_per_second,
    })
}

fn qwen25_vl_processor_feature_width(
    config: &Qwen25VlImageProcessorConfig,
) -> Result<usize, Qwen25VlProcessedMediaAssemblyError> {
    config
        .patch_size
        .checked_mul(config.patch_size)
        .and_then(|patch_area| patch_area.checked_mul(config.temporal_patch_size))
        .and_then(|temporal_area| temporal_area.checked_mul(3))
        .ok_or(Qwen25VlProcessedMediaAssemblyError::FeatureWidthOverflow)
}

fn validate_qwen25_vl_processed_grid(
    kind: MropeVisualKind,
    grid: MropeGrid,
    merge: usize,
) -> Result<(), MropePositionError> {
    if merge == 0 {
        return Err(MropePositionError::ZeroSpatialMerge);
    }
    for (axis, dimension) in [
        ("temporal", grid.temporal),
        ("height", grid.height),
        ("width", grid.width),
    ] {
        if dimension == 0 {
            return Err(MropePositionError::ZeroGridDimension { kind, axis });
        }
    }
    for (axis, dimension) in [("height", grid.height), ("width", grid.width)] {
        if !dimension.is_multiple_of(merge) {
            return Err(MropePositionError::NonDivisibleGrid {
                kind,
                axis,
                dimension,
                merge,
            });
        }
    }
    Ok(())
}

fn processed_media_temporal_layout_error(
    index: usize,
    source: Qwen25VlTemporalLayoutError,
) -> Qwen25VlProcessedMediaAssemblyError {
    match source {
        Qwen25VlTemporalLayoutError::PaddedFrameCountOverflow => {
            Qwen25VlProcessedMediaAssemblyError::VideoPaddedFrameCountOverflow { index }
        }
        Qwen25VlTemporalLayoutError::IncompatibleTemporalPadding {
            sampled_frame_count,
            temporal_patch_size,
            padded_frame_count,
        } => Qwen25VlProcessedMediaAssemblyError::VideoIncompatibleTemporalPadding {
            index,
            sampled_frame_count,
            temporal_patch_size,
            padded_frame_count,
        },
        Qwen25VlTemporalLayoutError::TooManyPaddedFrames {
            sampled_frame_count,
            padded_frame_count,
        } => Qwen25VlProcessedMediaAssemblyError::VideoTooManyPaddedFrames {
            index,
            sampled_frame_count,
            padded_frame_count,
            limit: MAX_VIDEO_FRAMES,
        },
    }
}

pub(super) fn qwen25_vl_combined_video_layout<I>(
    video_parts: I,
    feature_width: usize,
) -> Result<Qwen25VlCombinedVideoLayout, Qwen25VlProcessedMediaAssemblyError>
where
    I: IntoIterator<Item = (usize, DType)> + Clone,
{
    let mut rows = 0usize;
    for (index, (part_rows, _)) in video_parts.clone().into_iter().enumerate() {
        rows = rows
            .checked_add(part_rows)
            .ok_or(Qwen25VlProcessedMediaAssemblyError::VideoRowCountOverflow { index })?;
    }
    let elements = rows
        .checked_mul(feature_width)
        .ok_or(Qwen25VlProcessedMediaAssemblyError::VideoElementCountOverflow)?;
    let dtype = video_parts
        .clone()
        .into_iter()
        .next()
        .map_or(DType::F32, |(_, dtype)| dtype);
    elements
        .checked_mul(dtype.byte_size())
        .ok_or(Qwen25VlProcessedMediaAssemblyError::VideoByteCountOverflow)?;
    for (index, (_, part_dtype)) in video_parts.into_iter().enumerate() {
        if part_dtype != dtype {
            return Err(Qwen25VlProcessedMediaAssemblyError::VideoDtypeMismatch {
                index,
                expected: dtype,
                actual: part_dtype,
            });
        }
    }
    Ok(Qwen25VlCombinedVideoLayout {
        rows,
        feature_width,
        elements,
        dtype,
    })
}

pub(super) fn assemble_qwen25_vl_videos(
    mut videos: Vec<Qwen25VlProcessedVideo>,
    layout: Qwen25VlCombinedVideoLayout,
) -> Result<Qwen25VlAssembledVideos, Qwen25VlProcessedMediaAssemblyError> {
    let mut grids = Vec::new();
    grids.try_reserve_exact(videos.len()).map_err(|_| {
        Qwen25VlProcessedMediaAssemblyError::Allocation {
            stage: "video grids",
        }
    })?;
    let mut fps = Vec::new();
    fps.try_reserve_exact(videos.len()).map_err(|_| {
        Qwen25VlProcessedMediaAssemblyError::Allocation {
            stage: "video FPS values",
        }
    })?;

    if videos.is_empty() {
        return Ok(Qwen25VlAssembledVideos {
            pixel_values: None,
            grids,
            sampled_frames_per_second: fps,
        });
    }
    if videos.len() == 1 {
        let video = videos.pop().expect("one video remains");
        grids.push(video.grid);
        fps.push(video.sampled_frames_per_second.get());
        return Ok(Qwen25VlAssembledVideos {
            pixel_values: Some(video.pixel_values_videos),
            grids,
            sampled_frames_per_second: fps,
        });
    }

    for video in &videos {
        grids.push(video.grid);
        fps.push(video.sampled_frames_per_second.get());
    }
    // Every part shares `layout.dtype` (checked by `qwen25_vl_combined_video_layout`), so the
    // concatenation runs in the parts' own storage class: bf16/f16 pixels stay 16-bit words.
    let parts: Vec<&HostTensor> = videos
        .iter()
        .map(|video| &video.pixel_values_videos)
        .collect();
    let offsets: Vec<usize> = parts.iter().map(|part| part.numel()).collect();
    let tensor = gather_from(
        &parts,
        vec![layout.rows, layout.feature_width],
        offsets
            .iter()
            .enumerate()
            .flat_map(|(part, &len)| (0..len).map(move |k| (part, k))),
    );
    debug_assert_eq!(tensor.numel(), layout.elements);
    Ok(Qwen25VlAssembledVideos {
        pixel_values: Some(tensor),
        grids,
        sampled_frames_per_second: fps,
    })
}
