//! Qwen2.5-VL image and video preprocessing.
//!
//! Held for Card 566b (the VLM front end on the driver): the single-image entries below are the only
//! producers of the public `Qwen25VlProcessedImage` and have no production caller yet, so rustc reports
//! them dead where it reports them at all (`#[expect(dead_code, reason = "held for POOT-739 (formerly 566b)")]`). The
//! default-config wrappers only this crate's tests call are `#[cfg(test)]`; `raw_request.rs` calls the
//! live `*_with_config` batch and video entries.

use std::io::Cursor;

use image::{ImageFormat, ImageReader, imageops::FilterType};
use poot_models::mrope::{
    MropeGrid, MropeProcessorTimingError, MropeSampledFramesPerSecond, MropeSecondsPerGridStep,
    MropeTemporalPatchSize,
};
use poot_tensor::HostTensor;

use super::apng::Qwen25VlDecodedFrameSource;
use super::assembly::{Qwen25VlProcessedImage, Qwen25VlProcessedImages, Qwen25VlProcessedVideo};
use super::config::Qwen25VlImageProcessorConfig;
use super::video_sampling::{
    Qwen25VlSampledVideo, Qwen25VlVideoMetadata, Qwen25VlVideoSamplingError,
    plan_qwen2_5_vl_video_sampling,
};
use super::{MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25VlDecodedImage {
    pub(super) height: usize,
    pub(super) width: usize,
    pub(super) rgb: Vec<u8>,
}

impl Qwen25VlDecodedImage {
    /// Construct a decoded image from owned row-major, interleaved RGB8 pixels.
    pub fn from_rgb8(
        height: usize,
        width: usize,
        rgb: Vec<u8>,
    ) -> Result<Self, Qwen25VlImageDecodeError> {
        validate_decoded_image(height, width, rgb.len())?;
        Ok(Self { height, width, rgb })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlImageDecodeError {
    #[error("Qwen2.5-VL image dimensions must be nonzero, got {height}x{width}")]
    ZeroDimension { height: usize, width: usize },
    #[error("Qwen2.5-VL image dimensions {height}x{width} exceed the RGB decoder range")]
    DimensionOutOfRange { height: usize, width: usize },
    #[error("Qwen2.5-VL image {height}x{width} pixel count overflows usize")]
    PixelCountOverflow { height: usize, width: usize },
    #[error(
        "Qwen2.5-VL image {height}x{width} has {pixels} pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyPixels {
        height: usize,
        width: usize,
        pixels: usize,
        limit: usize,
    },
    #[error("Qwen2.5-VL RGB8 payload has {actual} bytes; dimensions require {expected}")]
    PixelDataLen { expected: usize, actual: usize },
    #[error("identify Qwen2.5-VL image bytes: {message}")]
    Identify { message: String },
    #[error("Qwen2.5-VL image format {format} is unsupported; expected PNG or JPEG")]
    UnsupportedFormat { format: String },
    #[error("decode Qwen2.5-VL image: {message}")]
    Decode { message: String },
    #[error(
        "decoded Qwen2.5-VL image dimensions changed from header {header_height}x{header_width} to {decoded_height}x{decoded_width}"
    )]
    DimensionMismatch {
        header_height: usize,
        header_width: usize,
        decoded_height: usize,
        decoded_width: usize,
    },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlImagePreprocessError {
    #[error("Qwen2.5-VL image {height}x{width} has an edge below smart-resize factor {factor}")]
    EdgeBelowFactor {
        height: usize,
        width: usize,
        factor: usize,
    },
    #[error(
        "Qwen2.5-VL image {height}x{width} has an aspect ratio above the supported maximum 200"
    )]
    AspectRatioTooLarge { height: usize, width: usize },
    #[error("Qwen2.5-VL image preprocessing arithmetic overflows usize while computing {stage}")]
    ArithmeticOverflow { stage: &'static str },
    #[error("Qwen2.5-VL image preprocessing cannot represent resized dimensions {height}x{width}")]
    ResizedDimensionOutOfRange { height: usize, width: usize },
    #[error("Qwen2.5-VL image processor resolved an empty output of {height}x{width}")]
    ZeroOutputDimension { height: usize, width: usize },
    #[error(
        "Qwen2.5-VL image {height}x{width} must be divisible by factor {factor} when resize is disabled"
    )]
    NonDivisibleWithoutResize {
        height: usize,
        width: usize,
        factor: usize,
    },
    #[error(
        "Qwen2.5-VL resized image {height}x{width} has {pixels} pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyOutputPixels {
        height: usize,
        width: usize,
        pixels: usize,
        limit: usize,
    },
    #[error("Qwen2.5-VL image batch must contain at least one image")]
    EmptyBatch,
    #[error("Qwen2.5-VL image preprocessing could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlVideoPreprocessError {
    #[error("Qwen2.5-VL video must contain at least one sampled frame")]
    EmptyFrames,
    #[error("Qwen2.5-VL video has {actual} sampled frames, exceeding the {limit}-frame cap")]
    TooManyFrames { actual: usize, limit: usize },
    #[error(
        "Qwen2.5-VL video frame {index} has dimensions {actual_height}x{actual_width}; expected {expected_height}x{expected_width}"
    )]
    FrameDimensionsMismatch {
        index: usize,
        expected_height: usize,
        expected_width: usize,
        actual_height: usize,
        actual_width: usize,
    },
    #[error(
        "Qwen2.5-VL video frame count {frame_count} with temporal patch size {temporal_patch_size} pads to {padded_frame_count}, which the cached Transformers 4.49.0 reshape cannot group"
    )]
    IncompatibleTemporalPadding {
        frame_count: usize,
        temporal_patch_size: usize,
        padded_frame_count: usize,
    },
    #[error(
        "Qwen2.5-VL video has {sampled_frame_count} sampled frames and {padded_frame_count} frames after temporal padding, exceeding the {limit}-frame cap"
    )]
    TooManyPaddedFrames {
        sampled_frame_count: usize,
        padded_frame_count: usize,
        limit: usize,
    },
    #[error(
        "Qwen2.5-VL processed video has {pixels} frame pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyOutputPixels { pixels: usize, limit: usize },
    #[error("Qwen2.5-VL video preprocessing arithmetic overflows usize while computing {stage}")]
    ArithmeticOverflow { stage: &'static str },
    #[error("Qwen2.5-VL video preprocessing could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
    #[error(transparent)]
    Image(#[from] Qwen25VlImagePreprocessError),
    #[error(transparent)]
    Timing(#[from] MropeProcessorTimingError),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlRawVideoPreprocessError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Sampling(#[from] Qwen25VlVideoSamplingError),
    #[error("decode Qwen2.5-VL source frame {frame_index}: {source}")]
    Decode {
        frame_index: usize,
        #[source]
        source: E,
    },
    #[error("Qwen2.5-VL raw video could not reserve memory for decoded frames")]
    Allocation,
    #[error(transparent)]
    Preprocess(#[from] Qwen25VlVideoPreprocessError),
}

/// Decode local PNG or JPEG bytes into an owned RGB8 image.
pub fn decode_qwen2_5_vl_image(
    bytes: &[u8],
) -> Result<Qwen25VlDecodedImage, Qwen25VlImageDecodeError> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|source| Qwen25VlImageDecodeError::Identify {
            message: source.to_string(),
        })?;
    let Some(format) = reader.format() else {
        return Err(Qwen25VlImageDecodeError::Identify {
            message: "unknown image format".to_string(),
        });
    };
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return Err(Qwen25VlImageDecodeError::UnsupportedFormat {
            format: format!("{format:?}"),
        });
    }
    let (header_width, header_height) =
        reader
            .into_dimensions()
            .map_err(|source| Qwen25VlImageDecodeError::Identify {
                message: source.to_string(),
            })?;
    validate_decoded_dimensions(header_height as usize, header_width as usize)?;

    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(|source| Qwen25VlImageDecodeError::Decode {
            message: source.to_string(),
        })?
        .into_rgb8();
    let decoded_height = decoded.height() as usize;
    let decoded_width = decoded.width() as usize;
    if (decoded_height, decoded_width) != (header_height as usize, header_width as usize) {
        return Err(Qwen25VlImageDecodeError::DimensionMismatch {
            header_height: header_height as usize,
            header_width: header_width as usize,
            decoded_height,
            decoded_width,
        });
    }
    Qwen25VlDecodedImage::from_rgb8(decoded_height, decoded_width, decoded.into_raw())
}

pub(super) fn validate_decoded_image(
    height: usize,
    width: usize,
    actual_bytes: usize,
) -> Result<(), Qwen25VlImageDecodeError> {
    let expected = validate_decoded_dimensions(height, width)?;
    if actual_bytes != expected {
        return Err(Qwen25VlImageDecodeError::PixelDataLen {
            expected,
            actual: actual_bytes,
        });
    }
    Ok(())
}

pub(super) fn validate_decoded_dimensions(
    height: usize,
    width: usize,
) -> Result<usize, Qwen25VlImageDecodeError> {
    if height == 0 || width == 0 {
        return Err(Qwen25VlImageDecodeError::ZeroDimension { height, width });
    }
    if u32::try_from(height).is_err() || u32::try_from(width).is_err() {
        return Err(Qwen25VlImageDecodeError::DimensionOutOfRange { height, width });
    }
    let pixels = height
        .checked_mul(width)
        .ok_or(Qwen25VlImageDecodeError::PixelCountOverflow { height, width })?;
    let expected = pixels
        .checked_mul(3)
        .ok_or(Qwen25VlImageDecodeError::PixelCountOverflow { height, width })?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(Qwen25VlImageDecodeError::TooManyPixels {
            height,
            width,
            pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    Ok(expected)
}

/// Preprocess one decoded RGB8 image into the existing processor-media handoff contract.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn preprocess_qwen2_5_vl_image(
    image: &Qwen25VlDecodedImage,
) -> Result<Qwen25VlProcessedImage, Qwen25VlImagePreprocessError> {
    preprocess_qwen2_5_vl_image_with_config(image, &Qwen25VlImageProcessorConfig::default())
}

/// Preprocess one decoded RGB8 image with a validated persisted processor configuration.
pub(crate) fn preprocess_qwen2_5_vl_image_with_config(
    image: &Qwen25VlDecodedImage,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlProcessedImage, Qwen25VlImagePreprocessError> {
    let layout = image_layout(image, config)?;
    let mut data = Vec::new();
    data.try_reserve_exact(layout.elements).map_err(|_| {
        Qwen25VlImagePreprocessError::Allocation {
            stage: "image pixel values",
        }
    })?;
    preprocess_image_into(image, &layout, config, &mut data);
    debug_assert_eq!(data.len(), layout.elements);
    Ok(Qwen25VlProcessedImage {
        pixel_values: HostTensor::f32(vec![layout.rows, layout.feature_width], data),
        grid: layout.grid,
    })
}

/// Preprocess and concatenate a nonempty ordered image list with the released defaults.
#[cfg(test)]
pub(crate) fn preprocess_qwen2_5_vl_images(
    images: &[Qwen25VlDecodedImage],
) -> Result<Qwen25VlProcessedImages, Qwen25VlImagePreprocessError> {
    preprocess_qwen2_5_vl_images_with_config(images, &Qwen25VlImageProcessorConfig::default())
}

/// Preprocess and concatenate a nonempty ordered image list with one validated configuration.
pub fn preprocess_qwen2_5_vl_images_with_config(
    images: &[Qwen25VlDecodedImage],
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

/// Sample, decode, and preprocess one raw video with the released processor defaults.
#[cfg(test)]
pub(crate) fn preprocess_qwen2_5_vl_raw_video<S: Qwen25VlDecodedFrameSource>(
    metadata: Qwen25VlVideoMetadata<'_>,
    requested_frames_per_second: MropeSampledFramesPerSecond,
    source: &mut S,
) -> Result<Qwen25VlSampledVideo, Qwen25VlRawVideoPreprocessError<S::Error>> {
    preprocess_qwen2_5_vl_raw_video_with_config(
        metadata,
        requested_frames_per_second,
        source,
        &Qwen25VlImageProcessorConfig::default(),
    )
}

/// Sample, decode, and preprocess one raw video with a validated processor configuration.
pub fn preprocess_qwen2_5_vl_raw_video_with_config<S: Qwen25VlDecodedFrameSource>(
    metadata: Qwen25VlVideoMetadata<'_>,
    requested_frames_per_second: MropeSampledFramesPerSecond,
    source: &mut S,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<Qwen25VlSampledVideo, Qwen25VlRawVideoPreprocessError<S::Error>> {
    let sample_plan = plan_qwen2_5_vl_video_sampling(metadata, requested_frames_per_second)?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(sample_plan.frame_indices.len())
        .map_err(|_| Qwen25VlRawVideoPreprocessError::Allocation)?;
    for &frame_index in &sample_plan.frame_indices {
        let frame = source.decode_frame(frame_index).map_err(|source| {
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

/// Preprocess one ordered nonempty sampled-frame sequence with the released defaults.
#[cfg(test)]
pub(crate) fn preprocess_qwen2_5_vl_video(
    frames: &[Qwen25VlDecodedImage],
    sampled_frames_per_second: MropeSampledFramesPerSecond,
) -> Result<Qwen25VlProcessedVideo, Qwen25VlVideoPreprocessError> {
    preprocess_qwen2_5_vl_video_with_config(
        frames,
        sampled_frames_per_second,
        &Qwen25VlImageProcessorConfig::default(),
    )
}

/// Preprocess one ordered nonempty sampled-frame sequence with a validated processor configuration.
pub fn preprocess_qwen2_5_vl_video_with_config(
    frames: &[Qwen25VlDecodedImage],
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
    pub(crate) elements: usize,
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

pub(super) fn image_layout(
    decoded: &Qwen25VlDecodedImage,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<ImageLayout, Qwen25VlImagePreprocessError> {
    let factor = config.patch_size.checked_mul(config.merge_size).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "resize factor",
        },
    )?;
    let (height, width) = if config.do_resize {
        smart_resize(decoded.height, decoded.width, config)?
    } else {
        if !decoded.height.is_multiple_of(factor) || !decoded.width.is_multiple_of(factor) {
            return Err(Qwen25VlImagePreprocessError::NonDivisibleWithoutResize {
                height: decoded.height,
                width: decoded.width,
                factor,
            });
        }
        (decoded.height, decoded.width)
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
    let elements = rows.checked_mul(feature_width).ok_or(
        Qwen25VlImagePreprocessError::ArithmeticOverflow {
            stage: "flattened patch elements",
        },
    )?;
    Ok(ImageLayout {
        height,
        width,
        rows,
        feature_width,
        elements,
        grid: MropeGrid::new(1, grid_height, grid_width),
    })
}

fn video_layout(
    frames: &[Qwen25VlDecodedImage],
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
        if (frame.height, frame.width) != (first.height, first.width) {
            return Err(Qwen25VlVideoPreprocessError::FrameDimensionsMismatch {
                index,
                expected_height: first.height,
                expected_width: first.width,
                actual_height: frame.height,
                actual_width: frame.width,
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
    frames: &'a [Qwen25VlDecodedImage],
    layout: &ImageLayout,
) -> Result<Vec<PreparedVideoFrame<'a>>, Qwen25VlVideoPreprocessError> {
    let mut prepared = Vec::new();
    prepared.try_reserve_exact(frames.len()).map_err(|_| {
        Qwen25VlVideoPreprocessError::Allocation {
            stage: "video frame views",
        }
    })?;
    for frame in frames {
        if (layout.height, layout.width) == (frame.height, frame.width) {
            prepared.push(PreparedVideoFrame::Borrowed(&frame.rgb));
        } else {
            let source = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                frame.width as u32,
                frame.height as u32,
                frame.rgb.as_slice(),
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
    decoded: &Qwen25VlDecodedImage,
    layout: &ImageLayout,
    config: &Qwen25VlImageProcessorConfig,
    data: &mut Vec<f32>,
) {
    if (layout.height, layout.width) == (decoded.height, decoded.width) {
        flatten_patches_into(&decoded.rgb, layout.height, layout.width, config, data);
    } else {
        // Borrow the decoded buffer: the resize output is the only owned RGB copy needed.
        let source = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
            decoded.width as u32,
            decoded.height as u32,
            decoded.rgb.as_slice(),
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
