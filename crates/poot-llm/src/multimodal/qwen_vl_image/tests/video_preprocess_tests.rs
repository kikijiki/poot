use super::*;

#[test]
fn qwen25_vl_video_matches_frozen_transformers_temporal_order_and_padding() {
    let frames = [1u8, 2, 3].map(|value| {
        let rgb = (0..4)
            .flat_map(|_| [value, value + 10, value + 20])
            .collect();
        Qwen25VlDecodedImage::from_rgb8(2, 2, rgb).unwrap()
    });
    let config = Qwen25VlImageProcessorConfig {
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 1,
        max_pixels: 16,
        patch_size: 1,
        temporal_patch_size: 2,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let sampled_fps = MropeSampledFramesPerSecond::new(4.0).unwrap();
    let actual = preprocess_qwen2_5_vl_video_with_config(&frames, sampled_fps, &config).unwrap();

    // Frozen from Transformers 4.49.0 with NumPy input [3,2,2,3]. The first two frames form
    // temporal group zero; group one repeats the final frame in its second slot.
    let expected_row_zero = [1.0, 2.0, 11.0, 12.0, 21.0, 22.0];
    let expected_row_one = [3.0, 3.0, 13.0, 13.0, 23.0, 23.0];
    assert_eq!(actual.grid, MropeGrid::new(2, 2, 2));
    assert_eq!(actual.pixel_values_videos.shape(), [8, 6]);
    assert_eq!(actual.sampled_frame_count, 3);
    assert_eq!(actual.padded_frame_count, 4);
    assert_eq!(actual.temporal_patch_size.get(), 2);
    assert_eq!(actual.sampled_frames_per_second, sampled_fps);
    assert_eq!(actual.seconds_per_grid_step.get(), 0.5);
    for row in 0..4 {
        assert_eq!(
            &actual.pixel_values_videos.as_f32().unwrap()[row * 6..(row + 1) * 6],
            &expected_row_zero
        );
    }
    for row in 4..8 {
        assert_eq!(
            &actual.pixel_values_videos.as_f32().unwrap()[row * 6..(row + 1) * 6],
            &expected_row_one
        );
    }

    let media = actual.processor_media();
    assert!(media.pixel_values.is_none());
    assert!(media.image_grids.is_empty());
    assert!(std::ptr::eq(
        media.pixel_values_videos.unwrap(),
        &actual.pixel_values_videos
    ));
    assert!(std::ptr::eq(media.video_grids.as_ptr(), &actual.grid));
    assert_eq!(media.video_grids, &[actual.grid]);
    assert_eq!(media.validate(), Ok(()));
    assert_eq!(
        actual.processor_fps_input(),
        MropeProcessorFpsInput::Scalar(4.0)
    );
}

#[test]
fn qwen25_vl_video_temporal_padding_matches_cached_rule_for_arbitrary_patch_sizes() {
    let frames = (0u8..8)
        .map(|index| {
            Qwen25VlDecodedImage::from_rgb8(1, 1, vec![index, index + 40, index + 80]).unwrap()
        })
        .collect::<Vec<_>>();
    let sampled_fps = MropeSampledFramesPerSecond::new(3.0).unwrap();

    for temporal_patch_size in 1..=6 {
        let config = Qwen25VlImageProcessorConfig {
            do_resize: false,
            do_rescale: false,
            do_normalize: false,
            min_pixels: 1,
            max_pixels: 64,
            patch_size: 1,
            temporal_patch_size,
            merge_size: 1,
            rescale_factor: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        for sampled_frame_count in 1..=frames.len() {
            let remainder = sampled_frame_count % temporal_patch_size;
            let padding = if remainder == 0 {
                0
            } else {
                temporal_patch_size - 1
            };
            let padded_frame_count = sampled_frame_count + padding;
            let actual = preprocess_qwen2_5_vl_video_with_config(
                &frames[..sampled_frame_count],
                sampled_fps,
                &config,
            );

            if !padded_frame_count.is_multiple_of(temporal_patch_size) {
                assert_eq!(
                    actual,
                    Err(Qwen25VlVideoPreprocessError::IncompatibleTemporalPadding {
                        frame_count: sampled_frame_count,
                        temporal_patch_size,
                        padded_frame_count,
                    }),
                    "temporal patch size {temporal_patch_size}, frame count {sampled_frame_count}",
                );
                continue;
            }

            let actual = actual.unwrap();
            let temporal_grid = padded_frame_count / temporal_patch_size;
            let mut expected = Vec::new();
            for group in 0..temporal_grid {
                for channel in 0..3 {
                    for temporal_offset in 0..temporal_patch_size {
                        let frame_index = (group * temporal_patch_size + temporal_offset)
                            .min(sampled_frame_count - 1);
                        expected.push(frames[frame_index].rgb[channel] as f32);
                    }
                }
            }
            assert_eq!(actual.sampled_frame_count, sampled_frame_count);
            assert_eq!(actual.padded_frame_count, padded_frame_count);
            assert_eq!(actual.grid, MropeGrid::new(temporal_grid, 1, 1));
            assert_eq!(
                actual.pixel_values_videos.shape(),
                [temporal_grid, 3 * temporal_patch_size]
            );
            assert_eq!(actual.pixel_values_videos.as_f32().unwrap(), expected);
            assert_eq!(
                actual.seconds_per_grid_step.get().to_bits(),
                (temporal_patch_size as f64 / 3.0).to_bits()
            );
        }
    }
}

#[test]
fn qwen25_vl_video_default_path_matches_one_image_transform_exactly() {
    let frame = Qwen25VlDecodedImage::from_rgb8(28, 28, rgb_fixture(28, 28)).unwrap();
    let image = preprocess_qwen2_5_vl_image(&frame).unwrap();
    let video = preprocess_qwen2_5_vl_video(
        std::slice::from_ref(&frame),
        MropeSampledFramesPerSecond::new(2.0).unwrap(),
    )
    .unwrap();

    assert_eq!(video.grid, image.grid);
    assert_eq!(video.pixel_values_videos, image.pixel_values);
    assert_eq!(video.sampled_frame_count, 1);
    assert_eq!(video.padded_frame_count, 2);

    let second = Qwen25VlDecodedImage::from_rgb8(
        28,
        28,
        rgb_fixture(28, 28)
            .into_iter()
            .map(|value| value.wrapping_add(17))
            .collect(),
    )
    .unwrap();
    let second_image = preprocess_qwen2_5_vl_image(&second).unwrap();
    let two_frame_video = preprocess_qwen2_5_vl_video(
        &[frame, second],
        MropeSampledFramesPerSecond::new(2.0).unwrap(),
    )
    .unwrap();
    let mut expected = Vec::with_capacity(image.pixel_values.as_f32().unwrap().len());
    let patch_area = 14 * 14;
    let feature_width = image.pixel_values.shape()[1];
    for row in 0..image.pixel_values.shape()[0] {
        for channel in 0..3 {
            let start = row * feature_width + channel * 2 * patch_area;
            expected.extend_from_slice(
                &image.pixel_values.as_f32().unwrap()[start..start + patch_area],
            );
            expected.extend_from_slice(
                &second_image.pixel_values.as_f32().unwrap()[start..start + patch_area],
            );
        }
    }
    assert_eq!(two_frame_video.grid, image.grid);
    assert_eq!(
        two_frame_video.pixel_values_videos.as_f32().unwrap(),
        expected
    );
}

#[test]
fn qwen25_vl_video_preserves_merge_and_frame_order_from_frozen_upstream_fixture() {
    let frames = [0u8, 50, 100].map(|base| {
        let rgb = (0..4)
            .flat_map(|pixel| {
                let value = base + pixel;
                [value, value + 10, value + 20]
            })
            .collect();
        Qwen25VlDecodedImage::from_rgb8(2, 2, rgb).unwrap()
    });
    let config = Qwen25VlImageProcessorConfig {
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 1,
        max_pixels: 16,
        patch_size: 1,
        temporal_patch_size: 2,
        merge_size: 2,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let actual = preprocess_qwen2_5_vl_video_with_config(
        &frames,
        MropeSampledFramesPerSecond::new(2.0).unwrap(),
        &config,
    )
    .unwrap();

    // Literal Transformers rows: fixes temporal group, spatial merge, channel, temporal feature and
    // duplicate-last order without using the image path as oracle.
    let expected = [
        [0.0, 50.0, 10.0, 60.0, 20.0, 70.0],
        [1.0, 51.0, 11.0, 61.0, 21.0, 71.0],
        [2.0, 52.0, 12.0, 62.0, 22.0, 72.0],
        [3.0, 53.0, 13.0, 63.0, 23.0, 73.0],
        [100.0, 100.0, 110.0, 110.0, 120.0, 120.0],
        [101.0, 101.0, 111.0, 111.0, 121.0, 121.0],
        [102.0, 102.0, 112.0, 112.0, 122.0, 122.0],
        [103.0, 103.0, 113.0, 113.0, 123.0, 123.0],
    ]
    .concat();
    assert_eq!(actual.grid, MropeGrid::new(2, 2, 2));
    assert_eq!(actual.pixel_values_videos.as_f32().unwrap(), expected);
}

#[test]
fn qwen25_vl_video_timing_facts_feed_existing_processor_timing_path() {
    use poot_models::mrope::{MropeConfigMetadata, assemble_qwen2_5_vl_video_timings};

    let frame = Qwen25VlDecodedImage::from_rgb8(28, 28, rgb_fixture(28, 28)).unwrap();
    let sampled_fps = MropeSampledFramesPerSecond::new(3.0).unwrap();
    let actual = preprocess_qwen2_5_vl_video(std::slice::from_ref(&frame), sampled_fps).unwrap();
    let config = MropeConfigMetadata::from_hf_config_slice(
        br#"{
            "model_type":"qwen2_5_vl","vocab_size":100,"hidden_size":8,
            "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":1,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
            "max_position_embeddings":128,"vision_start_token_id":10,"image_token_id":11,
            "video_token_id":12,"rope_scaling":{"mrope_section":[1,1,2]},
            "vision_config":{"tokens_per_second":4}
        }"#,
    )
    .unwrap();
    let timings = assemble_qwen2_5_vl_video_timings(
        1,
        actual.temporal_patch_size,
        actual.processor_fps_input(),
        &config,
    )
    .unwrap();
    assert_eq!(
        timings[0].seconds_per_grid_step,
        actual.seconds_per_grid_step
    );
    assert_eq!(
        actual.seconds_per_grid_step.get().to_bits(),
        (2.0f64 / 3.0).to_bits()
    );

    // A scalar FPS owns its value and must not inherit the media view's borrow.
    let detached_fps = {
        let processed = preprocess_qwen2_5_vl_video(
            std::slice::from_ref(&frame),
            MropeSampledFramesPerSecond::new(3.0).unwrap(),
        )
        .unwrap();
        processed.processor_fps_input()
    };
    assert_eq!(detached_fps, MropeProcessorFpsInput::Scalar(3.0));
}

#[test]
fn qwen25_vl_video_rejects_invalid_boundaries_before_processing() {
    let fps = MropeSampledFramesPerSecond::new(2.0).unwrap();
    assert_eq!(
        preprocess_qwen2_5_vl_video(&[], fps),
        Err(Qwen25VlVideoPreprocessError::EmptyFrames)
    );

    let first = Qwen25VlDecodedImage::from_rgb8(2, 2, rgb_fixture(2, 2)).unwrap();
    let second = Qwen25VlDecodedImage::from_rgb8(2, 3, rgb_fixture(2, 3)).unwrap();
    assert_eq!(
        preprocess_qwen2_5_vl_video_with_config(
            &[first.clone(), second],
            fps,
            &Qwen25VlImageProcessorConfig {
                do_resize: false,
                do_rescale: false,
                do_normalize: false,
                min_pixels: 1,
                max_pixels: 16,
                patch_size: 1,
                temporal_patch_size: 2,
                merge_size: 1,
                rescale_factor: 1.0,
                mean: [0.0; 3],
                std: [1.0; 3],
            }
        ),
        Err(Qwen25VlVideoPreprocessError::FrameDimensionsMismatch {
            index: 1,
            expected_height: 2,
            expected_width: 2,
            actual_height: 2,
            actual_width: 3,
        })
    );

    let temporal_three = Qwen25VlImageProcessorConfig {
        temporal_patch_size: 3,
        ..Qwen25VlImageProcessorConfig {
            do_resize: false,
            do_rescale: false,
            do_normalize: false,
            min_pixels: 1,
            max_pixels: 16,
            patch_size: 1,
            temporal_patch_size: 2,
            merge_size: 1,
            rescale_factor: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    };
    assert_eq!(
        preprocess_qwen2_5_vl_video_with_config(
            &[first.clone(), first.clone()],
            fps,
            &temporal_three,
        ),
        Err(Qwen25VlVideoPreprocessError::IncompatibleTemporalPadding {
            frame_count: 2,
            temporal_patch_size: 3,
            padded_frame_count: 4,
        })
    );

    let timing_overflow = Qwen25VlImageProcessorConfig {
        temporal_patch_size: usize::MAX,
        ..temporal_three.clone()
    };
    assert_eq!(
        preprocess_qwen2_5_vl_video_with_config(
            std::slice::from_ref(&first),
            MropeSampledFramesPerSecond::new(f64::MIN_POSITIVE).unwrap(),
            &timing_overflow,
        ),
        Err(Qwen25VlVideoPreprocessError::Timing(
            MropeProcessorTimingError::SecondsPerGridStep(
                poot_models::mrope::MropePositionError::NonFiniteTiming {
                    unit: poot_models::mrope::MropeTimingUnit::SecondsPerGridStep,
                }
            )
        ))
    );

    let tiny = Qwen25VlDecodedImage::from_rgb8(1, 1, vec![0; 3]).unwrap();
    let too_many = std::iter::repeat_n(tiny, MAX_VIDEO_FRAMES + 1).collect::<Vec<_>>();
    assert_eq!(
        preprocess_qwen2_5_vl_video(&too_many, fps),
        Err(Qwen25VlVideoPreprocessError::TooManyFrames {
            actual: MAX_VIDEO_FRAMES + 1,
            limit: MAX_VIDEO_FRAMES,
        })
    );

    let padded_cap = Qwen25VlImageProcessorConfig {
        temporal_patch_size: MAX_VIDEO_FRAMES + 1,
        ..temporal_three.clone()
    };
    assert_eq!(
        preprocess_qwen2_5_vl_video_with_config(std::slice::from_ref(&first), fps, &padded_cap,),
        Err(Qwen25VlVideoPreprocessError::TooManyPaddedFrames {
            sampled_frame_count: 1,
            padded_frame_count: MAX_VIDEO_FRAMES + 1,
            limit: MAX_VIDEO_FRAMES,
        })
    );

    let resize_to_ten_thousand = Qwen25VlImageProcessorConfig {
        do_resize: true,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 10_000,
        max_pixels: 10_000,
        patch_size: 1,
        temporal_patch_size: 2,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let frames = std::iter::repeat_n(first, 4096).collect::<Vec<_>>();
    assert_eq!(
        preprocess_qwen2_5_vl_video_with_config(&frames, fps, &resize_to_ten_thousand),
        Err(Qwen25VlVideoPreprocessError::TooManyOutputPixels {
            pixels: 40_960_000,
            limit: MAX_IMAGE_PIXELS,
        })
    );
}
