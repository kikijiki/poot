use super::*;

#[test]
fn qwen25_vl_raw_video_samples_in_order_and_composes_with_decoded_preprocessing() {
    let frames = (0..10u8)
        .map(|value| Qwen25VlDecodedImage::from_rgb8(1, 1, vec![value; 3]).unwrap())
        .collect::<Vec<_>>();
    let timestamps = (0..10).map(|index| index as f64 / 3.0).collect::<Vec<_>>();
    let metadata = Qwen25VlVideoMetadata {
        duration_seconds: 10.0 / 3.0,
        source_frames_per_second: 3.0,
        frame_count: 10,
        frame_timestamps_seconds: &timestamps,
    };
    let requested = MropeSampledFramesPerSecond::new(0.9).unwrap();
    let config = Qwen25VlImageProcessorConfig {
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 1,
        max_pixels: 16,
        patch_size: 1,
        temporal_patch_size: 1,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let expected_frames = [frames[0].clone(), frames[3].clone(), frames[6].clone()];
    let expected =
        preprocess_qwen2_5_vl_video_with_config(&expected_frames, requested, &config).unwrap();
    let mut source = RecordingFrameSource {
        frames: frames.clone(),
        calls: Vec::new(),
        fail_at: None,
    };
    let actual =
        preprocess_qwen2_5_vl_raw_video_with_config(metadata, requested, &mut source, &config)
            .unwrap();
    assert_eq!(source.calls, vec![0, 3, 6]);
    assert_eq!(actual.sample_plan.frame_indices(), &[0, 3, 6]);
    assert_eq!(actual.processed_video, expected);
    assert_eq!(
        actual
            .processed_video
            .sampled_frames_per_second
            .get()
            .to_bits(),
        requested.get().to_bits()
    );
    assert_ne!(
        requested.get().to_bits(),
        (actual.sample_plan.frame_indices().len() as f64 / metadata.duration_seconds).to_bits()
    );

    let default_frames = (0..2u8)
        .map(|value| Qwen25VlDecodedImage::from_rgb8(28, 28, vec![value; 28 * 28 * 3]).unwrap())
        .collect::<Vec<_>>();
    let default_timestamps = [0.0, 0.5];
    let default_metadata = Qwen25VlVideoMetadata {
        duration_seconds: 1.0,
        source_frames_per_second: 2.0,
        frame_count: 2,
        frame_timestamps_seconds: &default_timestamps,
    };
    let default_fps = MropeSampledFramesPerSecond::new(2.0).unwrap();
    let default_expected = preprocess_qwen2_5_vl_video(&default_frames, default_fps).unwrap();
    let mut default_source = RecordingFrameSource {
        frames: default_frames,
        calls: Vec::new(),
        fail_at: None,
    };
    let default_actual =
        preprocess_qwen2_5_vl_raw_video(default_metadata, default_fps, &mut default_source)
            .unwrap();
    assert_eq!(default_source.calls, vec![0, 1]);
    assert_eq!(default_actual.processed_video, default_expected);
}

#[test]
fn qwen25_vl_raw_video_source_and_preprocess_failures_are_typed_and_atomic() {
    let frames = (0..4u8)
        .map(|value| Qwen25VlDecodedImage::from_rgb8(1, 1, vec![value; 3]).unwrap())
        .collect::<Vec<_>>();
    let timestamps = [0.0, 0.25, 0.5, 0.75];
    let metadata = Qwen25VlVideoMetadata {
        duration_seconds: 1.0,
        source_frames_per_second: 4.0,
        frame_count: 4,
        frame_timestamps_seconds: &timestamps,
    };
    let requested = MropeSampledFramesPerSecond::new(3.0).unwrap();
    let config = Qwen25VlImageProcessorConfig {
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
        min_pixels: 1,
        max_pixels: 16,
        patch_size: 1,
        temporal_patch_size: 1,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let mut untouched_source = RecordingFrameSource {
        frames: frames.clone(),
        calls: Vec::new(),
        fail_at: None,
    };
    assert!(matches!(
        preprocess_qwen2_5_vl_raw_video_with_config(
            Qwen25VlVideoMetadata {
                duration_seconds: 0.0,
                ..metadata
            },
            requested,
            &mut untouched_source,
            &config,
        ),
        Err(Qwen25VlRawVideoPreprocessError::Sampling(
            Qwen25VlVideoSamplingError::NonPositiveDuration { actual: 0.0 }
        ))
    ));
    assert!(untouched_source.calls.is_empty());

    let mut failing_source = RecordingFrameSource {
        frames: frames.clone(),
        calls: Vec::new(),
        fail_at: Some(1),
    };
    assert!(matches!(
        preprocess_qwen2_5_vl_raw_video_with_config(
            metadata,
            requested,
            &mut failing_source,
            &config,
        ),
        Err(Qwen25VlRawVideoPreprocessError::Decode {
            frame_index: 1,
            source: FixtureVideoDecodeError(1),
        })
    ));
    assert_eq!(failing_source.calls, vec![0, 1]);

    let mut malformed_frames = frames;
    malformed_frames[1] = Qwen25VlDecodedImage::from_rgb8(1, 2, vec![0; 6]).unwrap();
    let mut malformed_source = RecordingFrameSource {
        frames: malformed_frames,
        calls: Vec::new(),
        fail_at: None,
    };
    assert!(matches!(
        preprocess_qwen2_5_vl_raw_video_with_config(
            metadata,
            requested,
            &mut malformed_source,
            &config,
        ),
        Err(Qwen25VlRawVideoPreprocessError::Preprocess(
            Qwen25VlVideoPreprocessError::FrameDimensionsMismatch { index: 1, .. }
        ))
    ));
    assert_eq!(malformed_source.calls, vec![0, 1, 2]);
}
