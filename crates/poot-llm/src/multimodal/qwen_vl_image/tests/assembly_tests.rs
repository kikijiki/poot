use super::*;
use poot_tensor::DType;

#[test]
fn qwen25_vl_processed_media_assembly_preserves_presence_order_and_owned_pointers() {
    let config = processed_media_config();

    let empty = assemble_qwen2_5_vl_processed_media(None, vec![], &config).unwrap();
    assert!(empty.pixel_values.is_none());
    assert!(empty.image_grids.is_empty());
    assert!(empty.pixel_values_videos.is_none());
    assert!(empty.video_grids.is_empty());
    assert_eq!(
        empty.processor_fps_input(),
        MropeProcessorFpsInput::PerVideo(&[])
    );
    assert_eq!(empty.temporal_patch_size().get(), 2);
    assert_eq!(empty.processor_media().validate(), Ok(()));

    let images = processed_images_fixture();
    let image_ptr = images.pixel_values.as_f32().unwrap().as_ptr();
    let image_grids = images.grids.clone();
    let image_grids_ptr = images.grids.as_ptr();
    let image_only = assemble_qwen2_5_vl_processed_media(Some(images), vec![], &config).unwrap();
    assert_eq!(
        image_only
            .pixel_values
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .as_ptr(),
        image_ptr,
        "an image batch is moved, not copied"
    );
    assert_eq!(image_only.image_grids, image_grids);
    assert_eq!(image_only.image_grids.as_ptr(), image_grids_ptr);
    assert!(image_only.pixel_values_videos.is_none());
    assert_eq!(image_only.processor_media().validate(), Ok(()));

    // A single video of 16-bit pixels is moved as is: same words, same allocation, same dtype.
    let mut video = processed_video_fixture(2, 3.0, 1.0);
    video.pixel_values_videos = HostTensor::bf16(vec![2, 6], vec![7; 12]);
    let video_ptr = video.pixel_values_videos.as_half().unwrap().as_ptr();
    let video_only = assemble_qwen2_5_vl_processed_media(None, vec![video], &config).unwrap();
    let video_pixels = video_only.pixel_values_videos.as_ref().unwrap();
    assert_eq!(video_pixels.dtype(), DType::BF16);
    assert_eq!(video_pixels.as_half().unwrap().as_ptr(), video_ptr);
    assert_eq!(video_only.video_grids, [MropeGrid::new(2, 1, 1)]);
    assert_eq!(video_only.sampled_frames_per_second(), [3.0]);
    assert_eq!(video_only.processor_media().validate(), Ok(()));

    let images = processed_images_fixture();
    let image_data = images.pixel_values.as_f32().unwrap().as_ptr();
    let image_grids_ptr = images.grids.as_ptr();
    let video = processed_video_fixture(1, 5.0, 20.0);
    let video_data = video.pixel_values_videos.as_f32().unwrap().as_ptr();
    let mixed = assemble_qwen2_5_vl_processed_media(Some(images), vec![video], &config).unwrap();
    assert_eq!(
        mixed
            .pixel_values
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .as_ptr(),
        image_data
    );
    assert_eq!(mixed.image_grids.as_ptr(), image_grids_ptr);
    assert_eq!(
        mixed
            .pixel_values_videos
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .as_ptr(),
        video_data
    );
    assert_eq!(mixed.image_grids.len(), 2);
    assert_eq!(mixed.video_grids, [MropeGrid::new(1, 1, 1)]);
    assert_eq!(
        mixed.processor_fps_input(),
        MropeProcessorFpsInput::PerVideo(&[5.0])
    );
    assert_eq!(mixed.processor_media().validate(), Ok(()));
}

#[test]
fn qwen25_vl_processed_media_assembly_concatenates_videos_and_timing_in_order() {
    use poot_models::mrope::{
        MropeConfigMetadata, Qwen25VlMropePositionInput, assemble_qwen2_5_vl_mrope_positions,
        assemble_qwen2_5_vl_video_timings,
    };

    let config = processed_media_config();
    let first = processed_video_fixture(1, 2.0, 10.0);
    let second = processed_video_fixture(2, 4.0, 30.0);
    let first_data = first.pixel_values_videos.as_f32().unwrap().as_ptr();
    let second_data = second.pixel_values_videos.as_f32().unwrap().as_ptr();
    let mut expected = first.pixel_values_videos.as_f32().unwrap().to_vec();
    expected.extend_from_slice(second.pixel_values_videos.as_f32().unwrap());
    let actual = assemble_qwen2_5_vl_processed_media(None, vec![first, second], &config).unwrap();

    assert_eq!(actual.pixel_values_videos.as_ref().unwrap().shape(), [3, 6]);
    assert_ne!(
        actual
            .pixel_values_videos
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .as_ptr(),
        first_data
    );
    assert_ne!(
        actual
            .pixel_values_videos
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .as_ptr(),
        second_data
    );
    assert_eq!(
        actual
            .pixel_values_videos
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap(),
        expected
    );
    assert_eq!(
        actual.video_grids,
        [MropeGrid::new(1, 1, 1), MropeGrid::new(2, 1, 1)]
    );
    assert_eq!(actual.sampled_frames_per_second(), [2.0, 4.0]);
    assert_eq!(actual.processor_media().validate(), Ok(()));

    let metadata = MropeConfigMetadata::from_hf_config_slice(
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
        actual.video_grids.len(),
        actual.temporal_patch_size(),
        actual.processor_fps_input(),
        &metadata,
    )
    .unwrap();
    assert_eq!(
        timings[0].seconds_per_grid_step.get().to_bits(),
        1.0f64.to_bits()
    );
    assert_eq!(
        timings[1].seconds_per_grid_step.get().to_bits(),
        0.5f64.to_bits()
    );

    // Token order alone fixes cross-kind order (image, video, image, video); the assembled media
    // stays two per-kind tensor/grid collections.
    let images = processed_images_fixture();
    let mixed = assemble_qwen2_5_vl_processed_media(
        Some(images),
        vec![
            processed_video_fixture(1, 2.0, 10.0),
            processed_video_fixture(2, 4.0, 30.0),
        ],
        &config,
    )
    .unwrap();
    let tokens = [10, 11, 10, 12, 10, 11, 11, 10, 12, 12];
    let positions = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
        tokens: &tokens,
        special_tokens: metadata.special_token_ids(),
        image_grids: &mixed.image_grids,
        video_grids: &mixed.video_grids,
        spatial_merge_size: 1,
        temporal_patch_size: mixed.temporal_patch_size(),
        fps: Some(mixed.processor_fps_input()),
        config: &metadata,
    })
    .unwrap();
    let [temporal, height, width] = positions.axes();
    assert_eq!(temporal, &[0, 1, 2, 3, 4, 5, 5, 7, 8, 10]);
    assert_eq!(height, &[0, 1, 2, 3, 4, 5, 5, 7, 8, 8]);
    assert_eq!(width, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 8]);
}

#[test]
fn qwen25_vl_processed_media_assembly_concatenates_16_bit_videos_in_their_own_dtype() {
    let config = processed_media_config();
    for dtype in [DType::BF16, DType::F16] {
        let half = |rows: usize, words: Vec<u16>| match dtype {
            DType::BF16 => HostTensor::bf16(vec![rows, 6], words),
            _ => HostTensor::f16(vec![rows, 6], words),
        };
        let mut first = processed_video_fixture(1, 2.0, 10.0);
        let mut second = processed_video_fixture(2, 4.0, 30.0);
        let first_words: Vec<u16> = (0..6).collect();
        let second_words: Vec<u16> = (6..18).rev().collect();
        first.pixel_values_videos = half(1, first_words.clone());
        second.pixel_values_videos = half(2, second_words.clone());

        let actual =
            assemble_qwen2_5_vl_processed_media(None, vec![first, second], &config).unwrap();
        let mut expected = first_words;
        expected.extend_from_slice(&second_words);
        let pixels = actual.pixel_values_videos.as_ref().unwrap();
        assert_eq!(pixels.dtype(), dtype, "concatenation must not widen");
        assert_eq!(pixels.shape(), [3, 6]);
        assert_eq!(pixels.as_half().unwrap(), expected.as_slice());
        assert_eq!(actual.processor_media().validate(), Ok(()));
    }
}

#[test]
fn qwen25_vl_processed_media_assembly_rejects_malformed_items_in_exact_order() {
    let config = processed_media_config();
    let bad_image = Qwen25VlProcessedImages {
        pixel_values: HostTensor::f32(vec![1, 1, 6], vec![0.0; 6]),
        grids: vec![MropeGrid::new(1, 1, 1)],
    };
    let mut bad_video = processed_video_fixture(1, 2.0, 0.0);
    bad_video.pixel_values_videos = HostTensor::f32(vec![1, 1, 6], vec![0.0; 6]);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(Some(bad_image), vec![bad_video], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::ImageMedia(
            crate::Qwen25VlProcessorMediaError::PixelRank { .. }
        ))
    ));

    let bad_width = Qwen25VlProcessedImages {
        pixel_values: HostTensor::f32(vec![1, 5], vec![0.0; 5]),
        grids: vec![MropeGrid::new(1, 1, 1)],
    };
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(Some(bad_width), vec![], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::ImageFeatureWidth {
            expected: 6,
            actual: 5,
        })
    );

    let mut video_width = processed_video_fixture(1, 2.0, 0.0);
    video_width.pixel_values_videos = HostTensor::f32(vec![1, 5], vec![0.0; 5]);
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![video_width], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoFeatureWidth {
            index: 0,
            expected: 6,
            actual: 5,
        })
    );

    let valid = processed_video_fixture(1, 2.0, 0.0);
    let mut wrong_rank = processed_video_fixture(1, 2.0, 0.0);
    wrong_rank.pixel_values_videos = HostTensor::f32(vec![1, 1, 6], vec![0.0; 6]);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![valid, wrong_rank], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoMedia {
            index: 1,
            source: crate::Qwen25VlProcessorMediaError::PixelRank { .. },
        })
    ));

    let mut integer = processed_video_fixture(1, 2.0, 0.0);
    integer.pixel_values_videos = HostTensor::i32(vec![1, 6], vec![0; 6]);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![integer], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoMedia {
            index: 0,
            source: crate::Qwen25VlProcessorMediaError::PixelDtype {
                actual: DType::I32,
                ..
            },
        })
    ));

    let mut wrong_rows = processed_video_fixture(1, 2.0, 0.0);
    wrong_rows.grid = MropeGrid::new(2, 1, 1);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![wrong_rows], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoMedia {
            source: crate::Qwen25VlProcessorMediaError::PixelRowCount { .. },
            ..
        })
    ));

    let mut grid_overflow = processed_video_fixture(1, 2.0, 0.0);
    grid_overflow.pixel_values_videos = HostTensor::f32(vec![0, 6], vec![]);
    grid_overflow.grid = MropeGrid::new(usize::MAX, 2, 1);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![grid_overflow], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoMedia {
            source: crate::Qwen25VlProcessorMediaError::GridRowCountOverflow { .. },
            ..
        })
    ));

    let overflow_config = Qwen25VlImageProcessorConfig {
        patch_size: usize::MAX,
        ..config
    };
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![], &overflow_config),
        Err(Qwen25VlProcessedMediaAssemblyError::FeatureWidthOverflow)
    );
}

#[test]
fn qwen25_vl_processed_media_assembly_rejects_impossible_public_grids() {
    let config = processed_media_config();
    let image = |grid, rows| Qwen25VlProcessedImages {
        pixel_values: HostTensor::f32(vec![rows, 6], vec![0.0; rows * 6]),
        grids: vec![grid],
    };

    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(
            Some(image(MropeGrid::new(1, 0, 1), 0)),
            vec![],
            &config,
        ),
        Err(Qwen25VlProcessedMediaAssemblyError::ImageGrid {
            index: 0,
            source: MropePositionError::ZeroGridDimension {
                kind: MropeVisualKind::Image,
                axis: "height",
            },
        })
    ));
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(
            Some(image(MropeGrid::new(2, 1, 1), 2)),
            vec![],
            &config,
        ),
        Err(Qwen25VlProcessedMediaAssemblyError::ImageGridTemporal {
            index: 0,
            actual: 2,
        })
    );

    let merge_two = Qwen25VlImageProcessorConfig {
        merge_size: 2,
        ..config.clone()
    };
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(
            Some(image(MropeGrid::new(1, 1, 2), 2)),
            vec![],
            &merge_two,
        ),
        Err(Qwen25VlProcessedMediaAssemblyError::ImageGrid {
            index: 0,
            source: MropePositionError::NonDivisibleGrid {
                kind: MropeVisualKind::Image,
                axis: "height",
                dimension: 1,
                merge: 2,
            },
        })
    ));

    let mut video = processed_video_fixture(1, 2.0, 0.0);
    video.grid = MropeGrid::new(1, 0, 1);
    video.pixel_values_videos = HostTensor::f32(vec![0, 6], vec![]);
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![video], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoGrid {
            index: 0,
            source: MropePositionError::ZeroGridDimension {
                kind: MropeVisualKind::Video,
                axis: "height",
            },
        })
    ));
}

#[test]
fn qwen25_vl_processed_media_assembly_rejects_impossible_public_frame_counts() {
    let config = processed_media_config();

    let mut empty = processed_video_fixture(1, 2.0, 0.0);
    empty.sampled_frame_count = 0;
    empty.padded_frame_count = 0;
    empty.seconds_per_grid_step = MropeSecondsPerGridStep::new(3.0).unwrap();
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![empty], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoEmptySampledFrames { index: 0 })
    );

    let mut oversized = processed_video_fixture(1, 2.0, 0.0);
    oversized.sampled_frame_count = MAX_VIDEO_FRAMES + 1;
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![oversized], &config),
        Err(
            Qwen25VlProcessedMediaAssemblyError::VideoTooManySampledFrames {
                index: 0,
                actual: MAX_VIDEO_FRAMES + 1,
                limit: MAX_VIDEO_FRAMES,
            }
        )
    );

    let mut wrong_padded = processed_video_fixture(1, 2.0, 0.0);
    wrong_padded.sampled_frame_count = 1;
    wrong_padded.padded_frame_count = 1;
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![wrong_padded], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoPaddedFrameCount {
            index: 0,
            expected: 2,
            actual: 1,
        })
    );

    let mut wrong_temporal = processed_video_fixture(2, 2.0, 0.0);
    wrong_temporal.sampled_frame_count = 2;
    wrong_temporal.padded_frame_count = 2;
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![wrong_temporal], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoGridTemporal {
            index: 0,
            expected: 1,
            actual: 2,
        })
    );

    let temporal_three = Qwen25VlImageProcessorConfig {
        temporal_patch_size: 3,
        ..config
    };
    let video_with_temporal_three = |sampled_frame_count, padded_frame_count| {
        let temporal_patch_size = MropeTemporalPatchSize::new(3).unwrap();
        let sampled_frames_per_second = MropeSampledFramesPerSecond::new(2.0).unwrap();
        Qwen25VlProcessedVideo {
            pixel_values_videos: HostTensor::f32(vec![1, 9], vec![0.0; 9]),
            grid: MropeGrid::new(1, 1, 1),
            sampled_frames_per_second,
            seconds_per_grid_step: MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
                temporal_patch_size,
                sampled_frames_per_second,
            )
            .unwrap(),
            temporal_patch_size,
            sampled_frame_count,
            padded_frame_count,
        }
    };
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(
            None,
            vec![video_with_temporal_three(2, 4)],
            &temporal_three,
        ),
        Err(
            Qwen25VlProcessedMediaAssemblyError::VideoIncompatibleTemporalPadding {
                index: 0,
                sampled_frame_count: 2,
                temporal_patch_size: 3,
                padded_frame_count: 4,
            }
        )
    );
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(
            None,
            vec![video_with_temporal_three(
                MAX_VIDEO_FRAMES,
                MAX_VIDEO_FRAMES + 2,
            )],
            &temporal_three,
        ),
        Err(
            Qwen25VlProcessedMediaAssemblyError::VideoTooManyPaddedFrames {
                index: 0,
                sampled_frame_count: MAX_VIDEO_FRAMES,
                padded_frame_count: MAX_VIDEO_FRAMES + 2,
                limit: MAX_VIDEO_FRAMES,
            }
        )
    );
    assert_eq!(
        qwen25_vl_temporal_layout(2, usize::MAX),
        Err(Qwen25VlTemporalLayoutError::PaddedFrameCountOverflow)
    );
}

#[test]
fn qwen25_vl_processed_media_assembly_rejects_temporal_timing_native_and_count_mismatch() {
    let config = processed_media_config();

    let mut temporal = processed_video_fixture(1, 2.0, 0.0);
    temporal.temporal_patch_size = MropeTemporalPatchSize::new(1).unwrap();
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![temporal], &config),
        Err(
            Qwen25VlProcessedMediaAssemblyError::VideoTemporalPatchSize {
                index: 0,
                expected: 2,
                actual: 1,
            }
        )
    );

    let mut timing = processed_video_fixture(1, 2.0, 0.0);
    timing.seconds_per_grid_step = MropeSecondsPerGridStep::new(3.0).unwrap();
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![timing], &config),
        Err(
            Qwen25VlProcessedMediaAssemblyError::VideoSecondsPerGridStep {
                index: 0,
                expected: 1.0f64.to_bits(),
                actual: 3.0f64.to_bits(),
            }
        )
    );

    let mut timing_overflow = processed_video_fixture(1, 2.0, 0.0);
    timing_overflow.sampled_frames_per_second =
        MropeSampledFramesPerSecond::new(f64::from_bits(1)).unwrap();
    assert!(matches!(
        assemble_qwen2_5_vl_processed_media(None, vec![timing_overflow], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoTiming {
            index: 0,
            source: MropeProcessorTimingError::SecondsPerGridStep(_),
        })
    ));

    let first = processed_video_fixture(1, 2.0, 0.0);
    let mut second = processed_video_fixture(1, 2.0, 0.0);
    second.pixel_values_videos = HostTensor::bf16(vec![1, 6], vec![0; 6]);
    assert_eq!(
        assemble_qwen2_5_vl_processed_media(None, vec![first, second], &config),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoDtypeMismatch {
            index: 1,
            expected: DType::F32,
            actual: DType::BF16,
        })
    );

    assert_eq!(
        qwen25_vl_combined_video_layout([(usize::MAX, DType::F32), (1, DType::F32)], 1),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoRowCountOverflow { index: 1 })
    );
    assert_eq!(
        qwen25_vl_combined_video_layout([(usize::MAX, DType::F32)], 2),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoElementCountOverflow)
    );
    assert_eq!(
        qwen25_vl_combined_video_layout([(usize::MAX / 2 + 1, DType::BF16)], 1),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoByteCountOverflow)
    );
    assert_eq!(
        qwen25_vl_combined_video_layout([(usize::MAX / 2 + 1, DType::BF16), (0, DType::F16)], 1,),
        Err(Qwen25VlProcessedMediaAssemblyError::VideoByteCountOverflow)
    );
}
