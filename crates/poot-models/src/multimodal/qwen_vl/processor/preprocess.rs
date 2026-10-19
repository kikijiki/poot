//! Qwen2.5-VL image and video preprocessing.

use image::imageops::FilterType;
use poot_tensor::HostTensor;

use super::assembly::{Qwen25VlProcessedImages, Qwen25VlProcessedVideo};
use super::config::Qwen25VlImageProcessorConfig;
use super::video_sampling::{
    Qwen25VlSampledVideo, Qwen25VlVideoSamplingError, plan_qwen2_5_vl_video_sampling,
};
use super::{MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES};
use crate::multimodal::front_end::{FrameSource, Image, VideoMetadata};
use crate::multimodal::mrope::{
    MropeGrid, MropeProcessorTimingError, MropeSampledFramesPerSecond, MropeSecondsPerGridStep,
    MropeTemporalPatchSize,
};

/// Why an image batch cannot be resized and patched.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum Qwen25VlImagePreprocessError {
    /// An edge is smaller than `patch_size * merge_size`, the smallest side smart-resize can produce.
    #[error("Qwen2.5-VL image {height}x{width} has an edge below smart-resize factor {factor}")]
    EdgeBelowFactor {
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
        /// The resize factor, `patch_size * merge_size`.
        factor: usize,
    },
    /// The longer side is more than 200 times the shorter.
    #[error(
        "Qwen2.5-VL image {height}x{width} has an aspect ratio above the supported maximum 200"
    )]
    AspectRatioTooLarge {
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
    },
    /// An intermediate size computation overflows `usize`.
    #[error("Qwen2.5-VL image preprocessing arithmetic overflows usize while computing {stage}")]
    ArithmeticOverflow {
        /// The quantity being computed.
        stage: &'static str,
    },
    /// The resized dimensions do not fit the resampler's `u32` range.
    #[error("Qwen2.5-VL image preprocessing cannot represent resized dimensions {height}x{width}")]
    ResizedDimensionOutOfRange {
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
    },
    /// The resized output has a zero dimension.
    #[error("Qwen2.5-VL image processor resolved an empty output of {height}x{width}")]
    ZeroOutputDimension {
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
    },
    /// Resizing is disabled and a dimension is not a multiple of the resize factor.
    #[error(
        "Qwen2.5-VL image {height}x{width} must be divisible by factor {factor} when resize is disabled"
    )]
    NonDivisibleWithoutResize {
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
        /// The resize factor, `patch_size * merge_size`.
        factor: usize,
    },
    /// The source image has more pixels than the cap.
    #[error(
        "Qwen2.5-VL image {height}x{width} has {pixels} pixels, exceeding the {limit}-pixel cap"
    )]
    TooManySourcePixels {
        /// Source height in pixels.
        height: usize,
        /// Source width in pixels.
        width: usize,
        /// `height * width`.
        pixels: usize,
        /// The cap, `MAX_IMAGE_PIXELS`.
        limit: usize,
    },
    /// The resized image has more pixels than the cap.
    #[error(
        "Qwen2.5-VL resized image {height}x{width} has {pixels} pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyOutputPixels {
        /// Resized height in pixels.
        height: usize,
        /// Resized width in pixels.
        width: usize,
        /// `height * width`.
        pixels: usize,
        /// The cap, `MAX_IMAGE_PIXELS`.
        limit: usize,
    },
    /// The batch holds no image.
    #[error("Qwen2.5-VL image batch must contain at least one image")]
    EmptyBatch,
    /// Reserving memory failed.
    #[error("Qwen2.5-VL image preprocessing could not reserve memory for {stage}")]
    Allocation {
        /// The buffer that failed to allocate.
        stage: &'static str,
    },
}

/// Why decoded video frames cannot be padded, resized and patched.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum Qwen25VlVideoPreprocessError {
    /// The video has no sampled frame.
    #[error("Qwen2.5-VL video must contain at least one sampled frame")]
    EmptyFrames,
    /// The video has more sampled frames than the cap.
    #[error("Qwen2.5-VL video has {actual} sampled frames, exceeding the {limit}-frame cap")]
    TooManyFrames {
        /// Frames given.
        actual: usize,
        /// The cap, `MAX_VIDEO_FRAMES`.
        limit: usize,
    },
    /// A frame's size differs from the first frame's.
    #[error(
        "Qwen2.5-VL video frame {index} has dimensions {actual_height}x{actual_width}; expected {expected_height}x{expected_width}"
    )]
    FrameDimensionsMismatch {
        /// Index of the frame among the video's sampled frames.
        index: usize,
        /// The first frame's height.
        expected_height: usize,
        /// The first frame's width.
        expected_width: usize,
        /// This frame's height.
        actual_height: usize,
        /// This frame's width.
        actual_width: usize,
    },
    /// Padding the frame count to a temporal patch multiple gives a count the processor's reshape cannot group.
    #[error(
        "Qwen2.5-VL video frame count {frame_count} with temporal patch size {temporal_patch_size} pads to {padded_frame_count}, which the cached Transformers 4.49.0 reshape cannot group"
    )]
    IncompatibleTemporalPadding {
        /// Sampled frames.
        frame_count: usize,
        /// Frames merged into one temporal patch.
        temporal_patch_size: usize,
        /// The padded count derived.
        padded_frame_count: usize,
    },
    /// The padded frame count exceeds the cap.
    #[error(
        "Qwen2.5-VL video has {sampled_frame_count} sampled frames and {padded_frame_count} frames after temporal padding, exceeding the {limit}-frame cap"
    )]
    TooManyPaddedFrames {
        /// Sampled frames.
        sampled_frame_count: usize,
        /// The padded count derived.
        padded_frame_count: usize,
        /// The cap, `MAX_VIDEO_FRAMES`.
        limit: usize,
    },
    /// All frames together have more pixels than the image pixel cap.
    #[error(
        "Qwen2.5-VL processed video has {pixels} frame pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyOutputPixels {
        /// Pixels summed over the frames.
        pixels: usize,
        /// The cap, `MAX_IMAGE_PIXELS`.
        limit: usize,
    },
    /// An intermediate size computation overflows `usize`.
    #[error("Qwen2.5-VL video preprocessing arithmetic overflows usize while computing {stage}")]
    ArithmeticOverflow {
        /// The quantity being computed.
        stage: &'static str,
    },
    /// Reserving memory failed.
    #[error("Qwen2.5-VL video preprocessing could not reserve memory for {stage}")]
    Allocation {
        /// The buffer that failed to allocate.
        stage: &'static str,
    },
    /// A frame failed image preprocessing.
    #[error(transparent)]
    Image(#[from] Qwen25VlImagePreprocessError),
    /// The frame rate does not give valid grid timing.
    #[error(transparent)]
    Timing(#[from] MropeProcessorTimingError),
}

/// Why a raw video cannot be sampled, decoded and preprocessed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Qwen25VlRawVideoPreprocessError {
    /// The sampler refused the video's metadata or the requested frame rate.
    #[error(transparent)]
    Sampling(#[from] Qwen25VlVideoSamplingError),
    /// The caller's frame source failed on a selected frame.
    #[error("decode Qwen2.5-VL source frame {frame_index}: {source}")]
    Decode {
        /// Source frame index that failed.
        frame_index: usize,
        /// The frame source's error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Reserving memory for the decoded frames failed.
    #[error("Qwen2.5-VL raw video could not reserve memory for decoded frames")]
    Allocation,
    /// The decoded frames failed video preprocessing.
    #[error(transparent)]
    Preprocess(#[from] Qwen25VlVideoPreprocessError),
}

/// Preprocess and concatenate a nonempty ordered image list with one validated configuration.
pub(crate) fn preprocess_qwen2_5_vl_images_with_config(
    images: &[&Image],
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlProcessedImages, Qwen25VlImagePreprocessError> {
    if images.is_empty() {
        return Err(Qwen25VlImagePreprocessError::EmptyBatch);
    }
    let mut layouts = Vec::new();
    layouts.try_reserve_exact(images.len()).map_err(|_| {
        Qwen25VlImagePreprocessError::Allocation {
            stage: "image layouts",
        }
    })?;
    let mut total_rows = 0usize;
    for image in images {
        let layout = image_layout(image, config)?;
        total_rows = total_rows.checked_add(layout.rows).ok_or(
            Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "batched flattened patch rows",
            },
        )?;
        layouts.push(layout);
    }
    let feature_width = layouts[0].feature_width;
    let total_elements = total_rows.checked_mul(feature_width).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "batched flattened patch elements",
        },
    )?;
    let mut data = Vec::new();
    data.try_reserve_exact(total_elements).map_err(|_| {
        Qwen25VlImagePreprocessError::Allocation {
            stage: "batched pixel values",
        }
    })?;
    let mut grids = Vec::new();
    grids.try_reserve_exact(images.len()).map_err(|_| {
        Qwen25VlImagePreprocessError::Allocation {
            stage: "batched image grids",
        }
    })?;
    for (image, layout) in images.iter().zip(&layouts) {
        preprocess_image_into(image, layout, config, &mut data);
        grids.push(layout.grid);
    }
    debug_assert_eq!(data.len(), total_elements);
    Ok(Qwen25VlProcessedImages {
        pixel_values: HostTensor::f32(vec![total_rows, feature_width], data),
        grids,
    })
}

/// Sample, decode, and preprocess one raw video with a validated processor configuration.
pub(crate) fn preprocess_qwen2_5_vl_raw_video_with_config(
    metadata: VideoMetadata<'_>,
    requested_frames_per_second: MropeSampledFramesPerSecond,
    source: &mut dyn FrameSource,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlSampledVideo, Qwen25VlRawVideoPreprocessError> {
    let sample_plan = plan_qwen2_5_vl_video_sampling(metadata, requested_frames_per_second)?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(sample_plan.frame_indices.len())
        .map_err(|_| Qwen25VlRawVideoPreprocessError::Allocation)?;
    for &frame_index in &sample_plan.frame_indices {
        let frame = source.frame(frame_index).map_err(|source| {
            Qwen25VlRawVideoPreprocessError::Decode {
                frame_index,
                source,
            }
        })?;
        frames.push(frame);
    }
    let processed_video = preprocess_qwen2_5_vl_video_with_config(
        &frames,
        sample_plan.processor_frames_per_second,
        config,
    )?;
    Ok(Qwen25VlSampledVideo {
        sample_plan,
        processed_video,
    })
}

/// Preprocess one ordered nonempty sampled-frame sequence with a validated processor configuration.
pub(crate) fn preprocess_qwen2_5_vl_video_with_config(
    frames: &[Image],
    sampled_frames_per_second: MropeSampledFramesPerSecond,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlProcessedVideo, Qwen25VlVideoPreprocessError> {
    let layout = video_layout(frames, sampled_frames_per_second, config)?;
    let mut data = Vec::new();
    data.try_reserve_exact(layout.elements).map_err(|_| {
        Qwen25VlVideoPreprocessError::Allocation {
            stage: "video pixel values",
        }
    })?;
    let prepared = prepare_video_frames(frames, &layout.image)?;
    flatten_video_patches_into(&prepared, &layout, config, &mut data);
    debug_assert_eq!(data.len(), layout.elements);
    // `Tensor::new` materializes shared Arc storage from this Vec. Release resized RGB staging first
    // so peak memory does not also hold every resized frame.
    drop(prepared);
    let pixel_values_videos = HostTensor::f32(vec![layout.rows, layout.image.feature_width], data);
    Ok(Qwen25VlProcessedVideo {
        pixel_values_videos,
        grid: layout.grid,
        sampled_frames_per_second,
        seconds_per_grid_step: layout.seconds_per_grid_step,
        temporal_patch_size: layout.temporal_patch_size,
        sampled_frame_count: frames.len(),
        padded_frame_count: layout.padded_frame_count,
    })
}

pub(super) fn smart_resize(
    height: usize,
    width: usize,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<(usize, usize), Qwen25VlImagePreprocessError> {
    let factor = config.patch_size.checked_mul(config.merge_size).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "resize factor",
        },
    )?;
    if height < factor || width < factor {
        return Err(Qwen25VlImagePreprocessError::EdgeBelowFactor {
            height,
            width,
            factor,
        });
    }
    let short = height.min(width);
    let long = height.max(width);
    if long > short.saturating_mul(200) {
        return Err(Qwen25VlImagePreprocessError::AspectRatioTooLarge { height, width });
    }
    let pixels =
        height
            .checked_mul(width)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "source pixel count",
            })?;
    let mut resized_height = round_div_ties_even(height, factor)?
        .checked_mul(factor)
        .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "rounded height",
        })?;
    let mut resized_width = round_div_ties_even(width, factor)?
        .checked_mul(factor)
        .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "rounded width",
        })?;
    let rounded_pixels = resized_height.checked_mul(resized_width).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "rounded pixel count",
        },
    )?;
    if rounded_pixels > config.max_pixels {
        let beta = (pixels as f64 / config.max_pixels as f64).sqrt();
        resized_height = ((height as f64 / beta / factor as f64).floor() as usize)
            .checked_mul(factor)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "maximum-pixel height",
            })?;
        resized_width = ((width as f64 / beta / factor as f64).floor() as usize)
            .checked_mul(factor)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "maximum-pixel width",
            })?;
    } else if rounded_pixels < config.min_pixels {
        let beta = (config.min_pixels as f64 / pixels as f64).sqrt();
        resized_height = ((height as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "minimum-pixel height",
            })?;
        resized_width = ((width as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "minimum-pixel width",
            })?;
    }
    Ok((resized_height, resized_width))
}

fn round_div_ties_even(
    value: usize,
    divisor: usize,
) -> Result<usize, Qwen25VlImagePreprocessError> {
    let quotient = value / divisor;
    let remainder = value % divisor;
    let doubled =
        remainder
            .checked_mul(2)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "ties-to-even rounding",
            })?;
    if doubled < divisor || (doubled == divisor && quotient.is_multiple_of(2)) {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "ties-to-even quotient",
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ImageLayout {
    pub(super) height: usize,
    pub(super) width: usize,
    pub(crate) rows: usize,
    pub(crate) feature_width: usize,
    pub(crate) grid: MropeGrid,
}

#[derive(Clone, Copy, Debug)]
struct VideoLayout {
    pub(crate) image: ImageLayout,
    pub(crate) temporal_patch_size: MropeTemporalPatchSize,
    pub(crate) padded_frame_count: usize,
    pub(crate) rows: usize,
    pub(crate) elements: usize,
    pub(crate) grid: MropeGrid,
    pub(crate) seconds_per_grid_step: MropeSecondsPerGridStep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Qwen25VlTemporalLayout {
    pub(super) padded_frame_count: usize,
    pub(super) grid_temporal: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Qwen25VlTemporalLayoutError {
    PaddedFrameCountOverflow,
    IncompatibleTemporalPadding {
        sampled_frame_count: usize,
        temporal_patch_size: usize,
        padded_frame_count: usize,
    },
    TooManyPaddedFrames {
        sampled_frame_count: usize,
        padded_frame_count: usize,
    },
}

pub(super) fn qwen25_vl_temporal_layout(
    sampled_frame_count: usize,
    temporal_patch_size: usize,
) -> Result<Qwen25VlTemporalLayout, Qwen25VlTemporalLayoutError> {
    debug_assert_ne!(temporal_patch_size, 0);
    let padding = if sampled_frame_count.is_multiple_of(temporal_patch_size) {
        0
    } else {
        temporal_patch_size - 1
    };
    let padded_frame_count = sampled_frame_count
        .checked_add(padding)
        .ok_or(Qwen25VlTemporalLayoutError::PaddedFrameCountOverflow)?;
    if !padded_frame_count.is_multiple_of(temporal_patch_size) {
        return Err(Qwen25VlTemporalLayoutError::IncompatibleTemporalPadding {
            sampled_frame_count,
            temporal_patch_size,
            padded_frame_count,
        });
    }
    if padded_frame_count > MAX_VIDEO_FRAMES {
        return Err(Qwen25VlTemporalLayoutError::TooManyPaddedFrames {
            sampled_frame_count,
            padded_frame_count,
        });
    }
    Ok(Qwen25VlTemporalLayout {
        padded_frame_count,
        grid_temporal: padded_frame_count / temporal_patch_size,
    })
}

/// A source image of `height` x `width` pixels is within the pixel cap, judged by its dimensions alone.
pub(super) fn check_source_pixels(
    height: usize,
    width: usize,
) -> Result<(), Qwen25VlImagePreprocessError> {
    let pixels =
        height
            .checked_mul(width)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "source pixel count",
            })?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(Qwen25VlImagePreprocessError::TooManySourcePixels {
            height,
            width,
            pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    Ok(())
}

pub(super) fn image_layout(
    decoded: &Image,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<ImageLayout, Qwen25VlImagePreprocessError> {
    check_source_pixels(decoded.height(), decoded.width())?;
    let factor = config.patch_size.checked_mul(config.merge_size).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "resize factor",
        },
    )?;
    let (height, width) = if config.do_resize {
        smart_resize(decoded.height(), decoded.width(), config)?
    } else {
        if !decoded.height().is_multiple_of(factor) || !decoded.width().is_multiple_of(factor) {
            return Err(Qwen25VlImagePreprocessError::NonDivisibleWithoutResize {
                height: decoded.height(),
                width: decoded.width(),
                factor,
            });
        }
        (decoded.height(), decoded.width())
    };
    if height == 0 || width == 0 {
        return Err(Qwen25VlImagePreprocessError::ZeroOutputDimension { height, width });
    }
    if u32::try_from(height).is_err() || u32::try_from(width).is_err() {
        return Err(Qwen25VlImagePreprocessError::ResizedDimensionOutOfRange { height, width });
    }
    let pixels =
        height
            .checked_mul(width)
            .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
                stage: "resized pixel count",
            })?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(Qwen25VlImagePreprocessError::TooManyOutputPixels {
            height,
            width,
            pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    let grid_height = height / config.patch_size;
    let grid_width = width / config.patch_size;
    let rows = grid_height.checked_mul(grid_width).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "flattened patch rows",
        },
    )?;
    let patch_area = config.patch_size.checked_mul(config.patch_size).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "patch area",
        },
    )?;
    let feature_width = 3usize
        .checked_mul(config.temporal_patch_size)
        .and_then(|value| value.checked_mul(patch_area))
        .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "flattened patch feature width",
        })?;
    rows.checked_mul(feature_width)
        .ok_or(Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "flattened patch elements",
        })?;
    Ok(ImageLayout {
        height,
        width,
        rows,
        feature_width,
        grid: MropeGrid::new(1, grid_height, grid_width),
    })
}

fn video_layout(
    frames: &[Image],
    sampled_frames_per_second: MropeSampledFramesPerSecond,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<VideoLayout, Qwen25VlVideoPreprocessError> {
    let Some(first) = frames.first() else {
        return Err(Qwen25VlVideoPreprocessError::EmptyFrames);
    };
    if frames.len() > MAX_VIDEO_FRAMES {
        return Err(Qwen25VlVideoPreprocessError::TooManyFrames {
            actual: frames.len(),
            limit: MAX_VIDEO_FRAMES,
        });
    }
    for (index, frame) in frames.iter().enumerate().skip(1) {
        if (frame.height(), frame.width()) != (first.height(), first.width()) {
            return Err(Qwen25VlVideoPreprocessError::FrameDimensionsMismatch {
                index,
                expected_height: first.height(),
                expected_width: first.width(),
                actual_height: frame.height(),
                actual_width: frame.width(),
            });
        }
    }

    let temporal_patch_size = MropeTemporalPatchSize::new(config.temporal_patch_size)
        .expect("validated image processor temporal patch size must be nonzero");
    let seconds_per_grid_step = MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
        temporal_patch_size,
        sampled_frames_per_second,
    )?;
    let image = image_layout(first, config)?;
    let temporal_patch = temporal_patch_size.get();
    let temporal_layout =
        qwen25_vl_temporal_layout(frames.len(), temporal_patch).map_err(|source| match source {
            Qwen25VlTemporalLayoutError::PaddedFrameCountOverflow => {
                Qwen25VlVideoPreprocessError::ArithmeticOverflow {
                    stage: "padded frame count",
                }
            }
            Qwen25VlTemporalLayoutError::IncompatibleTemporalPadding {
                sampled_frame_count,
                temporal_patch_size,
                padded_frame_count,
            } => Qwen25VlVideoPreprocessError::IncompatibleTemporalPadding {
                frame_count: sampled_frame_count,
                temporal_patch_size,
                padded_frame_count,
            },
            Qwen25VlTemporalLayoutError::TooManyPaddedFrames {
                sampled_frame_count,
                padded_frame_count,
            } => Qwen25VlVideoPreprocessError::TooManyPaddedFrames {
                sampled_frame_count,
                padded_frame_count,
                limit: MAX_VIDEO_FRAMES,
            },
        })?;
    let padded_frame_count = temporal_layout.padded_frame_count;

    let pixels_per_frame = image.height.checked_mul(image.width).ok_or(
        Qwen25VlVideoPreprocessError::ArithmeticOverflow {
            stage: "resized frame pixels",
        },
    )?;
    let total_pixels = padded_frame_count.checked_mul(pixels_per_frame).ok_or(
        Qwen25VlVideoPreprocessError::ArithmeticOverflow {
            stage: "processed frame pixels",
        },
    )?;
    if total_pixels > MAX_IMAGE_PIXELS {
        return Err(Qwen25VlVideoPreprocessError::TooManyOutputPixels {
            pixels: total_pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }

    let grid_temporal = temporal_layout.grid_temporal;
    let rows = grid_temporal.checked_mul(image.rows).ok_or(
        Qwen25VlVideoPreprocessError::ArithmeticOverflow {
            stage: "flattened video patch rows",
        },
    )?;
    let elements = rows.checked_mul(image.feature_width).ok_or(
        Qwen25VlVideoPreprocessError::ArithmeticOverflow {
            stage: "flattened video patch elements",
        },
    )?;
    let _output_bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or(
        Qwen25VlVideoPreprocessError::ArithmeticOverflow {
            stage: "flattened video patch bytes",
        },
    )?;
    Ok(VideoLayout {
        image,
        temporal_patch_size,
        padded_frame_count,
        rows,
        elements,
        grid: MropeGrid::new(grid_temporal, image.grid.height, image.grid.width),
        seconds_per_grid_step,
    })
}

enum PreparedVideoFrame<'a> {
    Borrowed(&'a [u8]),
    Resized(Vec<u8>),
}

impl PreparedVideoFrame<'_> {
    fn rgb(&self) -> &[u8] {
        match self {
            Self::Borrowed(rgb) => rgb,
            Self::Resized(rgb) => rgb,
        }
    }
}

fn prepare_video_frames<'a>(
    frames: &'a [Image],
    layout: &ImageLayout,
) -> Result<Vec<PreparedVideoFrame<'a>>, Qwen25VlVideoPreprocessError> {
    let mut prepared = Vec::new();
    prepared.try_reserve_exact(frames.len()).map_err(|_| {
        Qwen25VlVideoPreprocessError::Allocation {
            stage: "video frame views",
        }
    })?;
    for frame in frames {
        if (layout.height, layout.width) == (frame.height(), frame.width()) {
            prepared.push(PreparedVideoFrame::Borrowed(frame.rgb()));
        } else {
            let source = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                frame.width() as u32,
                frame.height() as u32,
                frame.rgb(),
            )
            .expect("validated decoded RGB dimensions must match the payload");
            let resized = image::imageops::resize(
                &source,
                layout.width as u32,
                layout.height as u32,
                FilterType::CatmullRom,
            );
            prepared.push(PreparedVideoFrame::Resized(resized.into_raw()));
        }
    }
    Ok(prepared)
}

fn preprocess_image_into(
    decoded: &Image,
    layout: &ImageLayout,
    config: &Qwen25VlImageProcessorConfig,
    data: &mut Vec<f32>,
) {
    if (layout.height, layout.width) == (decoded.height(), decoded.width()) {
        flatten_patches_into(decoded.rgb(), layout.height, layout.width, config, data);
    } else {
        // Borrow the decoded buffer: the resize output is the only owned RGB copy needed.
        let source = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
            decoded.width() as u32,
            decoded.height() as u32,
            decoded.rgb(),
        )
        .expect("validated decoded RGB dimensions must match the payload");
        let resized = image::imageops::resize(
            &source,
            layout.width as u32,
            layout.height as u32,
            FilterType::CatmullRom,
        );
        flatten_patches_into(resized.as_raw(), layout.height, layout.width, config, data);
    }
}

fn flatten_patches_into(
    rgb: &[u8],
    height: usize,
    width: usize,
    config: &Qwen25VlImageProcessorConfig,
    data: &mut Vec<f32>,
) {
    let grid_height = height / config.patch_size;
    let grid_width = width / config.patch_size;
    for height_block in 0..grid_height / config.merge_size {
        for width_block in 0..grid_width / config.merge_size {
            for height_merge in 0..config.merge_size {
                for width_merge in 0..config.merge_size {
                    for channel in 0..3 {
                        for _temporal_patch in 0..config.temporal_patch_size {
                            for patch_y in 0..config.patch_size {
                                for patch_x in 0..config.patch_size {
                                    let y = (height_block * config.merge_size + height_merge)
                                        * config.patch_size
                                        + patch_y;
                                    let x = (width_block * config.merge_size + width_merge)
                                        * config.patch_size
                                        + patch_x;
                                    let value = rgb[(y * width + x) * 3 + channel];
                                    data.push(process_rgb8_value(value, channel, config));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn flatten_video_patches_into(
    frames: &[PreparedVideoFrame<'_>],
    layout: &VideoLayout,
    config: &Qwen25VlImageProcessorConfig,
    data: &mut Vec<f32>,
) {
    let grid_height = layout.image.grid.height;
    let grid_width = layout.image.grid.width;
    let temporal_patch = layout.temporal_patch_size.get();
    for temporal_grid in 0..layout.grid.temporal {
        for height_block in 0..grid_height / config.merge_size {
            for width_block in 0..grid_width / config.merge_size {
                for height_merge in 0..config.merge_size {
                    for width_merge in 0..config.merge_size {
                        for channel in 0..3 {
                            for temporal_offset in 0..temporal_patch {
                                let frame_index = (temporal_grid * temporal_patch
                                    + temporal_offset)
                                    .min(frames.len() - 1);
                                let rgb = frames[frame_index].rgb();
                                for patch_y in 0..config.patch_size {
                                    for patch_x in 0..config.patch_size {
                                        let y = (height_block * config.merge_size + height_merge)
                                            * config.patch_size
                                            + patch_y;
                                        let x = (width_block * config.merge_size + width_merge)
                                            * config.patch_size
                                            + patch_x;
                                        let value = rgb[(y * layout.image.width + x) * 3 + channel];
                                        data.push(process_rgb8_value(value, channel, config));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn process_rgb8_value(value: u8, channel: usize, config: &Qwen25VlImageProcessorConfig) -> f32 {
    // Transformers 4.49.0 rescales in f64, casts that result to f32, then performs normalization in f32.
    let mut processed = if config.do_rescale {
        (value as f64 * config.rescale_factor) as f32
    } else {
        value as f32
    };
    if config.do_normalize {
        processed = (processed - config.mean[channel]) / config.std[channel];
    }
    processed
}
