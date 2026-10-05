use super::*;

fn sampling_plan(
    frame_count: usize,
    source_frames_per_second: f64,
    requested_frames_per_second: f64,
) -> Result<Qwen25VlVideoSamplePlan, Qwen25VlVideoSamplingError> {
    let timestamps = (0..frame_count)
        .map(|index| index as f64 / source_frames_per_second)
        .collect::<Vec<_>>();
    plan_qwen2_5_vl_video_sampling(
        Qwen25VlVideoMetadata {
            duration_seconds: frame_count as f64 / source_frames_per_second,
            source_frames_per_second,
            frame_count,
            frame_timestamps_seconds: &timestamps,
        },
        MropeSampledFramesPerSecond::new(requested_frames_per_second).unwrap(),
    )
}

#[test]
fn qwen25_vl_video_sampling_matches_frozen_transformers_count_and_integer_step() {
    // Frozen outputs from Transformers 4.49.0 `default_sample_indices_fn` with NumPy 2.5.2. The
    // 2 FPS case pins NumPy's integer-dtype step conversion (not floor(k * float_step)).
    for (source_fps, requested_fps, expected) in [
        (3.0, 0.9, vec![0, 3, 6]),
        (3.0, 1.2, vec![0, 2, 4, 6]),
        (3.0, 1.5, vec![0, 2, 4, 6, 8]),
        (3.0, 2.0, vec![0, 1, 2, 3, 4, 5]),
        (
            3.0,
            2.999_999_999_999_999_6,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        ),
        (3.0, 3.0, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (
            3.0,
            3.000_000_000_000_000_4,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
    ] {
        let plan = sampling_plan(10, source_fps, requested_fps).unwrap();
        assert_eq!(plan.frame_indices(), expected);
        assert_eq!(
            plan.processor_frames_per_second().get().to_bits(),
            requested_fps.to_bits()
        );
    }

    let long = sampling_plan(100, 29.97, 2.0).unwrap();
    assert_eq!(long.frame_indices(), &[0, 16, 32, 48, 64, 80]);
    assert_eq!(
        long.frame_timestamps_seconds(),
        &[
            0.0,
            16.0 / 29.97,
            32.0 / 29.97,
            48.0 / 29.97,
            64.0 / 29.97,
            80.0 / 29.97,
        ]
    );

    // Floating stop/step division makes NumPy emit one more row than the nominal count.
    let rounded_length = sampling_plan(15, 15.0, 13.0).unwrap();
    assert_eq!(
        rounded_length.frame_indices(),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    );

    assert_eq!(
        sampling_plan(1, 30.0, 2.0),
        Err(Qwen25VlVideoSamplingError::EmptySelection)
    );
    assert_eq!(
        sampling_plan(10, 3.0, 3.3),
        Err(Qwen25VlVideoSamplingError::SampleCountExceedsSource {
            sampled_frame_count: 11,
            source_frame_count: 10,
        })
    );
}

#[test]
fn qwen25_vl_video_sampling_validates_metadata_caps_and_error_precedence() {
    let requested = MropeSampledFramesPerSecond::new(2.0).unwrap();
    let valid_timestamps = [0.0, 0.5];
    let plan = |duration_seconds, source_frames_per_second, frame_count, timestamps: &[f64]| {
        plan_qwen2_5_vl_video_sampling(
            Qwen25VlVideoMetadata {
                duration_seconds,
                source_frames_per_second,
                frame_count,
                frame_timestamps_seconds: timestamps,
            },
            requested,
        )
    };

    assert!(matches!(
        plan(f64::NAN, f64::NAN, 0, &[]),
        Err(Qwen25VlVideoSamplingError::NonFiniteDuration { .. })
    ));
    assert_eq!(
        plan(0.0, f64::NAN, 0, &[]),
        Err(Qwen25VlVideoSamplingError::NonPositiveDuration { actual: 0.0 })
    );
    assert!(matches!(
        plan(1.0, f64::INFINITY, 0, &[]),
        Err(Qwen25VlVideoSamplingError::NonFiniteSourceFramesPerSecond { .. })
    ));
    assert_eq!(
        plan(1.0, 0.0, 0, &[]),
        Err(Qwen25VlVideoSamplingError::NonPositiveSourceFramesPerSecond { actual: 0.0 })
    );
    assert_eq!(
        plan(1.0, 2.0, 0, &[]),
        Err(Qwen25VlVideoSamplingError::EmptySource)
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        plan(1.0, 2.0, (1usize << 53) + 1, &[]),
        Err(
            Qwen25VlVideoSamplingError::FrameCountNotExactlyRepresentable {
                actual: (1usize << 53) + 1,
            }
        )
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        plan(1.0, 2.0, (1usize << 53) + 2, &[]),
        Err(Qwen25VlVideoSamplingError::TimestampCountMismatch {
            frame_count: (1usize << 53) + 2,
            timestamp_count: 0,
        })
    );
    assert_eq!(
        plan(1.0, 2.0, 2, &[0.0]),
        Err(Qwen25VlVideoSamplingError::TimestampCountMismatch {
            frame_count: 2,
            timestamp_count: 1,
        })
    );
    assert!(matches!(
        plan(1.0, 2.0, 2, &[f64::NAN, 0.5]),
        Err(Qwen25VlVideoSamplingError::NonFiniteTimestamp { index: 0, .. })
    ));
    assert_eq!(
        plan(1.0, 2.0, 2, &[-0.1, 0.5]),
        Err(Qwen25VlVideoSamplingError::NegativeTimestamp {
            index: 0,
            actual: -0.1,
        })
    );
    assert_eq!(
        plan(1.0, 2.0, 2, &[0.5, 0.5]),
        Err(Qwen25VlVideoSamplingError::NonMonotonicTimestamp {
            index: 1,
            previous: 0.5,
            actual: 0.5,
        })
    );
    assert_eq!(
        plan(0.4, 2.0, 2, &valid_timestamps),
        Err(Qwen25VlVideoSamplingError::TimestampAfterDuration {
            index: 1,
            actual: 0.5,
            duration: 0.4,
        })
    );
    assert_eq!(
        plan_qwen2_5_vl_video_sampling(
            Qwen25VlVideoMetadata {
                duration_seconds: 1.0,
                source_frames_per_second: f64::MIN_POSITIVE,
                frame_count: 1,
                frame_timestamps_seconds: &[0.0],
            },
            MropeSampledFramesPerSecond::new(f64::MAX).unwrap(),
        ),
        Err(Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow)
    );

    let cap_frame_count = MAX_VIDEO_FRAMES + 1;
    let cap_timestamps = (0..cap_frame_count)
        .map(|index| index as f64 / cap_frame_count as f64)
        .collect::<Vec<_>>();
    let cap_metadata = Qwen25VlVideoMetadata {
        duration_seconds: 1.0,
        source_frames_per_second: cap_frame_count as f64,
        frame_count: cap_frame_count,
        frame_timestamps_seconds: &cap_timestamps,
    };
    let at_cap = plan_qwen2_5_vl_video_sampling(
        cap_metadata,
        MropeSampledFramesPerSecond::new(MAX_VIDEO_FRAMES as f64).unwrap(),
    )
    .unwrap();
    assert_eq!(at_cap.frame_indices().len(), MAX_VIDEO_FRAMES);
    assert_eq!(
        at_cap.frame_indices()[MAX_VIDEO_FRAMES - 1],
        MAX_VIDEO_FRAMES - 1
    );
    assert_eq!(
        plan_qwen2_5_vl_video_sampling(
            cap_metadata,
            MropeSampledFramesPerSecond::new((MAX_VIDEO_FRAMES + 1) as f64).unwrap(),
        ),
        Err(Qwen25VlVideoSamplingError::TooManySampledFrames {
            actual: MAX_VIDEO_FRAMES + 1,
            limit: MAX_VIDEO_FRAMES,
        })
    );
}

#[test]
fn qwen25_vl_video_sampling_covers_numpy_length_and_count_cliffs() {
    // Frozen NumPy 2.5.2 output: rounding the stop/step quotient adds a row even though the
    // integer step is greater than one.
    assert_eq!(
        sampling_plan(17, 17.0, 7.0).unwrap().frame_indices(),
        &[0, 2, 4, 6, 8, 10, 12, 14]
    );

    // Frozen count cliffs around 2 * PI / 3, given as bits so decimal formatting cannot cross the
    // f64 boundary.
    let source_fps = std::f64::consts::PI;
    for requested_bits in [0x4000_c152_382d_7364, 0x4000_c152_382d_7365] {
        assert_eq!(
            sampling_plan(3, source_fps, f64::from_bits(requested_bits))
                .unwrap()
                .frame_indices(),
            &[0]
        );
    }
    assert_eq!(
        sampling_plan(3, source_fps, f64::from_bits(0x4000_c152_382d_7366))
            .unwrap()
            .frame_indices(),
        &[0, 1]
    );

    // The analogous cliff above the source count keeps upstream's oversampling rejection ahead of
    // the selected-frame cap.
    for requested_bits in [0x4010_c152_382d_7364, 0x4010_c152_382d_7365] {
        assert_eq!(
            sampling_plan(3, source_fps, f64::from_bits(requested_bits))
                .unwrap()
                .frame_indices(),
            &[0, 1, 2]
        );
    }
    assert_eq!(
        sampling_plan(3, source_fps, f64::from_bits(0x4010_c152_382d_7366)),
        Err(Qwen25VlVideoSamplingError::SampleCountExceedsSource {
            sampled_frame_count: 4,
            source_frame_count: 3,
        })
    );

    // Every valid small source/count pair is strictly increasing, duplicate-free and in bounds,
    // including the nominal-plus-one NumPy lengths.
    for frame_count in 1..=256 {
        for nominal_count in 1..=frame_count {
            let plan =
                sampling_plan(frame_count, frame_count as f64, nominal_count as f64).unwrap();
            assert!(
                plan.frame_indices()
                    .iter()
                    .all(|&index| index < frame_count)
            );
            assert!(
                plan.frame_indices()
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            assert_eq!(
                plan.frame_indices().len(),
                plan.frame_timestamps_seconds().len()
            );
        }
    }
}

#[test]
fn qwen25_vl_video_sampling_timestamp_contract_is_strict_and_duration_inclusive() {
    let requested = MropeSampledFramesPerSecond::new(2.0).unwrap();
    let boundary_timestamps = [0.0, 1.0];
    let metadata = Qwen25VlVideoMetadata {
        duration_seconds: 1.0,
        source_frames_per_second: 2.0,
        frame_count: 2,
        frame_timestamps_seconds: &boundary_timestamps,
    };
    let plan = plan_qwen2_5_vl_video_sampling(metadata, requested).unwrap();
    assert_eq!(plan.frame_timestamps_seconds(), &boundary_timestamps);
    let container_duration = plan_qwen2_5_vl_video_sampling(
        Qwen25VlVideoMetadata {
            duration_seconds: 2.0,
            ..metadata
        },
        requested,
    )
    .unwrap();
    assert_eq!(container_duration.frame_indices(), plan.frame_indices());

    let after_duration = [0.0, f64::from_bits(1.0f64.to_bits() + 1)];
    assert_eq!(
        plan_qwen2_5_vl_video_sampling(
            Qwen25VlVideoMetadata {
                frame_timestamps_seconds: &after_duration,
                ..metadata
            },
            requested,
        ),
        Err(Qwen25VlVideoSamplingError::TimestampAfterDuration {
            index: 1,
            actual: after_duration[1],
            duration: 1.0,
        })
    );

    // Timestamps are validated before any sampling arithmetic or oversampling error.
    assert!(matches!(
        plan_qwen2_5_vl_video_sampling(
            Qwen25VlVideoMetadata {
                frame_timestamps_seconds: &[0.0, f64::NAN],
                ..metadata
            },
            MropeSampledFramesPerSecond::new(f64::MAX).unwrap(),
        ),
        Err(Qwen25VlVideoSamplingError::NonFiniteTimestamp { index: 1, .. })
    ));
}
