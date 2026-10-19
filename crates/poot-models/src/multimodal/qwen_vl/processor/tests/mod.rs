use poot_tensor::HostTensor;

use super::assembly::*;
use super::config::*;
use super::preprocess::*;
use super::video_sampling::*;
use super::{MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES};
use crate::multimodal::front_end::{FrameSource, Image, VideoMetadata};
use crate::multimodal::mrope::MropeConfigMetadata;
use crate::multimodal::mrope::{
    MropeGrid, MropePositionError, MropeProcessorFpsInput, MropeProcessorTimingError,
    MropeSampledFramesPerSecond, MropeSecondsPerGridStep, MropeTemporalPatchSize, MropeVisualKind,
};
use crate::multimodal::qwen_vl::media::*;
use crate::registry::RawConfig;

/// One image through the batch preprocessing the request path runs, as the single image the
/// preprocessing tests ask about.
#[derive(Debug, PartialEq)]
struct ProcessedImage {
    pixel_values: HostTensor,
    grid: MropeGrid,
}

fn preprocess_qwen2_5_vl_image_with_config(
    image: &Image,
    config: &Qwen25VlImageProcessorConfig,
) -> Result<ProcessedImage, Qwen25VlImagePreprocessError> {
    let batch = preprocess_qwen2_5_vl_images_with_config(&[image], config)?;
    Ok(ProcessedImage {
        pixel_values: batch.pixel_values,
        grid: batch.grids[0],
    })
}

fn preprocess_qwen2_5_vl_image(
    image: &Image,
) -> Result<ProcessedImage, Qwen25VlImagePreprocessError> {
    preprocess_qwen2_5_vl_image_with_config(image, &Qwen25VlImageProcessorConfig::default())
}

/// A batch of images under the released defaults.
fn preprocess_qwen2_5_vl_images(
    images: &[&Image],
) -> Result<Qwen25VlProcessedImages, Qwen25VlImagePreprocessError> {
    preprocess_qwen2_5_vl_images_with_config(images, &Qwen25VlImageProcessorConfig::default())
}

/// A sampled-frame sequence under the released defaults.
fn preprocess_qwen2_5_vl_video(
    frames: &[Image],
    sampled_frames_per_second: MropeSampledFramesPerSecond,
) -> Result<Qwen25VlProcessedVideo, Qwen25VlVideoPreprocessError> {
    preprocess_qwen2_5_vl_video_with_config(
        frames,
        sampled_frames_per_second,
        &Qwen25VlImageProcessorConfig::default(),
    )
}

/// A raw video under the released defaults.
fn preprocess_qwen2_5_vl_raw_video(
    metadata: VideoMetadata<'_>,
    requested_frames_per_second: MropeSampledFramesPerSecond,
    source: &mut dyn FrameSource,
) -> Result<Qwen25VlSampledVideo, Qwen25VlRawVideoPreprocessError> {
    preprocess_qwen2_5_vl_raw_video_with_config(
        metadata,
        requested_frames_per_second,
        source,
        &Qwen25VlImageProcessorConfig::default(),
    )
}

mod assembly_tests;
mod config_tests;
mod image_preprocess_tests;
mod raw_video_tests;
mod video_preprocess_tests;
mod video_sampling_tests;

/// The mRoPE metadata of a parsed `config.json`.
fn mrope_metadata(json: &str) -> MropeConfigMetadata {
    let config: serde_json::Value = serde_json::from_str(json).unwrap();
    MropeConfigMetadata::from_raw(&RawConfig::HfJson {
        config: &config,
        generation: None,
        preprocessor: None,
    })
    .unwrap()
}

/// The persisted processor config `json` parsed.
fn processor_config(
    json: &str,
) -> Result<Qwen25VlImageProcessorConfig, Qwen25VlImageProcessorConfigError> {
    Qwen25VlImageProcessorConfig::from_value(&serde_json::from_str(json).unwrap())
}

fn config_validation_error(json: &str) -> Qwen25VlImageProcessorConfigValidationError {
    match processor_config(json).unwrap_err() {
        Qwen25VlImageProcessorConfigError::Validate(source) => source,
        error => panic!("expected validation error, got {error}"),
    }
}

fn rgb_fixture(height: usize, width: usize) -> Vec<u8> {
    (0..height * width)
        .flat_map(|pixel| {
            let value = (pixel % 180) as u8;
            [value, value + 20, value + 40]
        })
        .collect()
}

fn processed_media_config() -> Qwen25VlImageProcessorConfig {
    Qwen25VlImageProcessorConfig {
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 1,
        max_pixels: 64,
        patch_size: 1,
        temporal_patch_size: 2,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    }
}

fn processed_video_fixture(temporal: usize, fps: f64, first_value: f32) -> Qwen25VlProcessedVideo {
    let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();
    let sampled_frames_per_second = MropeSampledFramesPerSecond::new(fps).unwrap();
    let seconds_per_grid_step = MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
        temporal_patch_size,
        sampled_frames_per_second,
    )
    .unwrap();
    Qwen25VlProcessedVideo {
        pixel_values_videos: HostTensor::f32(
            vec![temporal, 6],
            (0..temporal * 6)
                .map(|index| first_value + index as f32)
                .collect(),
        ),
        grid: MropeGrid::new(temporal, 1, 1),
        sampled_frames_per_second,
        seconds_per_grid_step,
        temporal_patch_size,
        sampled_frame_count: temporal * 2,
        padded_frame_count: temporal * 2,
    }
}

fn processed_images_fixture() -> Qwen25VlProcessedImages {
    Qwen25VlProcessedImages {
        pixel_values: HostTensor::f32(
            vec![3, 6],
            (0..18).map(|index| 100.0 + index as f32).collect(),
        ),
        grids: vec![MropeGrid::new(1, 1, 1), MropeGrid::new(1, 1, 2)],
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("fixture decode failure at frame {0}")]
struct FixtureVideoDecodeError(usize);

struct RecordingFrameSource {
    frames: Vec<Image>,
    calls: Vec<usize>,
    fail_at: Option<usize>,
}

impl FrameSource for RecordingFrameSource {
    fn frame(
        &mut self,
        frame_index: usize,
    ) -> Result<Image, Box<dyn std::error::Error + Send + Sync>> {
        self.calls.push(frame_index);
        if self.fail_at == Some(frame_index) {
            return Err(Box::new(FixtureVideoDecodeError(frame_index)));
        }
        Ok(self.frames[frame_index].clone())
    }
}
