use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::ImageFormat;
use poot_load::qwen_vl::QwenVlModelType;
use poot_models::mrope::{
    MropeGrid, MropePositionError, MropeProcessorFpsInput, MropeProcessorTimingError,
    MropeSampledFramesPerSecond, MropeSecondsPerGridStep, MropeSegment, MropeTemporalPatchSize,
    MropeVisualKind,
};
use poot_tensor::HostTensor;

use super::*;
use crate::multimodal::qwen_vl_mrope::{
    Qwen25VlMropeModelInputView, Qwen25VlProcessorMedia, Qwen25VlProcessorMediaKind,
    Qwen25VlProcessorMediaMropeModelInputRequest, QwenVlMropeMetadataLoadError,
    assemble_qwen2_5_vl_mrope_model_input_from_processor_media,
};

mod apng_tests;
mod assembly_tests;
mod config_tests;
mod image_preprocess_tests;
mod raw_request_tests;
mod raw_video_tests;
mod video_preprocess_tests;
mod video_sampling_tests;

struct FixtureDir {
    path: PathBuf,
}

impl std::ops::Deref for FixtureDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn processor_config_fixture(name: &str, json: &str) -> FixtureDir {
    let path = std::env::temp_dir().join(format!(
        "poot_qwen25_vl_image_processor_{name}_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join(PREPROCESSOR_CONFIG_FILE), json).unwrap();
    FixtureDir { path }
}

fn config_validation_error(name: &str, json: &str) -> Qwen25VlImageProcessorConfigValidationError {
    let dir = processor_config_fixture(name, json);
    match load_qwen2_5_vl_image_processor_config(&dir).unwrap_err() {
        Qwen25VlImageProcessorConfigError::Validate { path, source } => {
            assert_eq!(path, dir.join(PREPROCESSOR_CONFIG_FILE));
            source
        }
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
    frames: Vec<Qwen25VlDecodedImage>,
    calls: Vec<usize>,
    fail_at: Option<usize>,
}

impl Qwen25VlDecodedFrameSource for RecordingFrameSource {
    type Error = FixtureVideoDecodeError;

    fn decode_frame(&mut self, frame_index: usize) -> Result<Qwen25VlDecodedImage, Self::Error> {
        self.calls.push(frame_index);
        if self.fail_at == Some(frame_index) {
            return Err(FixtureVideoDecodeError(frame_index));
        }
        Ok(self.frames[frame_index].clone())
    }
}
