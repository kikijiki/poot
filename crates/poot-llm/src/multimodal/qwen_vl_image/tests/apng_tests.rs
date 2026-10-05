use super::*;

const VARIABLE_DELAY_APNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAACGFjVEwAAAADAAAAAM7tusAAAAAaZmNUTAAAAAAAAAACAAAAAQAAAAAAAAAAAAEAAwAA9viN9gAAAA5JREFUeJxj+M/AAEL/AQ/5A/2FEZl2AAAAGmZjVEwAAAABAAAAAQAAAAEAAAAAAAAAAAABAAQBAaWO9tkAAAARZmRBVAAAAAJ4nGNg+M/QAAADggGA4cSRgwAAABpmY1RMAAAAAwAAAAEAAAABAAAAAQAAAAAACgAAAACXU2LqAAAAEWZkQVQAAAAEeJxj+P+f4T8AB/0C/k1aIHsAAAAASUVORK5CYII=";

const ZERO_DELAY_APNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAACGFjVEwAAAADAAAAAM7tusAAAAAaZmNUTAAAAAAAAAACAAAAAQAAAAAAAAAAAAEAAgAA9zrnwQAAAA5JREFUeJxj+M/AAEL/AQ/5A/2FEZl2AAAAGmZjVEwAAAABAAAAAQAAAAEAAAAAAAAAAAAAAAQBAZju32kAAAARZmRBVAAAAAJ4nGNg+M/QAAADggGA4cSRgwAAABpmY1RMAAAAAwAAAAEAAAABAAAAAQAAAAAACgAAAACXU2LqAAAAEWZkQVQAAAAEeJxj+P+f4T8AB/0C/k1aIHsAAAAASUVORK5CYII=";

const STATIC_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGNkZGJmYGZm/g8AAVQBEGAwCA8AAAAASUVORK5CYII=";

fn decode_base64_fixture(encoded: &str) -> Vec<u8> {
    use base64::Engine as _;

    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap()
}

#[test]
fn qwen25_vl_apng_preserves_variable_timing_and_zero_denominator_policy() {
    let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
    let apng = Qwen25VlApng::from_bytes(&bytes).unwrap();
    let metadata = apng.metadata();
    assert_eq!((apng.height(), apng.width()), (1, 2));
    assert_eq!(metadata.frame_count, 3);
    let first_delay = (1_000.0f64 / 3.0) / 1_000.0;
    assert_eq!(
        metadata.frame_timestamps_seconds,
        &[0.0, first_delay, first_delay + 0.25]
    );
    let expected_duration = first_delay + 0.25 + 0.1;
    assert_eq!(
        metadata.duration_seconds.to_bits(),
        expected_duration.to_bits()
    );
    assert_eq!(
        metadata.source_frames_per_second.to_bits(),
        (3.0 / expected_duration).to_bits()
    );

    // The final frame encodes delay_num=10, delay_den=0; `image` applies PNG's denominator 100,
    // giving the 0.1 s above. The public reduced ratio does not record a zero denominator.
    let requested = MropeSampledFramesPerSecond::new(3.0).unwrap();
    let plan = plan_qwen2_5_vl_video_sampling(metadata, requested).unwrap();
    assert_eq!(plan.frame_indices(), &[0, 1]);
    assert_eq!(plan.frame_timestamps_seconds(), &[0.0, first_delay]);
}

#[test]
fn qwen25_vl_apng_source_scans_sparse_frames_and_returns_composited_rgb() {
    let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
    let apng = Qwen25VlApng::from_bytes(&bytes).unwrap();
    let mut source = apng.frame_source().unwrap();
    let first = source.decode_frame(0).unwrap();
    assert_eq!(first.rgb(), &[255, 0, 0, 0, 0, 255]);
    let third = source.decode_frame(2).unwrap();
    assert_eq!(third.rgb(), &[0, 0, 0, 255, 255, 0]);

    let mut repeated = apng.frame_source().unwrap();
    let second = repeated.decode_frame(1).unwrap();
    assert_eq!(second.rgb(), &[127, 128, 0, 0, 0, 255]);
    assert_eq!(
        repeated.decode_frame(1).unwrap_err(),
        Qwen25VlApngError::FrameIndexNotIncreasing {
            previous: 1,
            requested: 1,
        }
    );
    assert_eq!(
        repeated.decode_frame(2).unwrap_err(),
        Qwen25VlApngError::PoisonedFrameSource
    );

    let mut decreasing = apng.frame_source().unwrap();
    decreasing.decode_frame(2).unwrap();
    assert_eq!(
        decreasing.decode_frame(0).unwrap_err(),
        Qwen25VlApngError::FrameIndexNotIncreasing {
            previous: 2,
            requested: 0,
        }
    );

    let mut out_of_range = apng.frame_source().unwrap();
    assert_eq!(
        out_of_range.decode_frame(3).unwrap_err(),
        Qwen25VlApngError::FrameIndexOutOfRange {
            requested: 3,
            frame_count: 3,
        }
    );
    assert_eq!(
        out_of_range.decode_frame(0).unwrap_err(),
        Qwen25VlApngError::PoisonedFrameSource
    );
}

#[test]
fn qwen25_vl_apng_rejects_static_malformed_zero_delay_and_caps() {
    let static_png = decode_base64_fixture(STATIC_PNG_BASE64);
    assert_eq!(
        Qwen25VlApng::from_bytes(&static_png).unwrap_err(),
        Qwen25VlApngError::StaticPng
    );

    let valid = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
    let truncated = &valid[..valid.len() - 20];
    assert!(matches!(
        Qwen25VlApng::from_bytes(truncated),
        Err(Qwen25VlApngError::FrameDecode { .. }) | Err(Qwen25VlApngError::Decoder { .. })
    ));

    let zero_delay = decode_base64_fixture(ZERO_DELAY_APNG_BASE64);
    assert_eq!(
        Qwen25VlApng::from_bytes(&zero_delay).unwrap_err(),
        Qwen25VlApngError::ZeroFrameDelay { frame_index: 1 }
    );

    assert_eq!(
        validate_qwen25_vl_apng_encoded_len(MAX_ENCODED_APNG_BYTES + 1),
        Err(Qwen25VlApngError::TooManyEncodedBytes {
            actual: MAX_ENCODED_APNG_BYTES + 1,
            limit: MAX_ENCODED_APNG_BYTES,
        })
    );
    assert_eq!(
        validate_qwen25_vl_apng_decoded_work(MAX_VIDEO_FRAMES + 1, 1),
        Err(Qwen25VlApngError::TooManyFrames {
            actual: MAX_VIDEO_FRAMES + 1,
            limit: MAX_VIDEO_FRAMES,
        })
    );
    let pixels_per_frame = MAX_IMAGE_PIXELS / 2 + 1;
    assert_eq!(
        validate_qwen25_vl_apng_decoded_work(2, pixels_per_frame),
        Err(Qwen25VlApngError::TooManyDecodedPixels {
            pixels: pixels_per_frame * 2,
            limit: MAX_IMAGE_PIXELS,
        })
    );
    assert_eq!(
        validate_qwen25_vl_apng_decoded_work(2, usize::MAX),
        Err(Qwen25VlApngError::DecodedPixelCountOverflow)
    );
}

#[test]
fn qwen25_vl_apng_composes_with_existing_raw_video_path() {
    let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
    let apng = Qwen25VlApng::from_bytes(&bytes).unwrap();
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
    let mut expected_source = apng.frame_source().unwrap();
    let expected_frames = [
        expected_source.decode_frame(0).unwrap(),
        expected_source.decode_frame(1).unwrap(),
    ];
    let expected =
        preprocess_qwen2_5_vl_video_with_config(&expected_frames, requested, &config).unwrap();

    let mut source = apng.frame_source().unwrap();
    let actual = preprocess_qwen2_5_vl_raw_video_with_config(
        apng.metadata(),
        requested,
        &mut source,
        &config,
    )
    .unwrap();
    assert_eq!(actual.sample_plan.frame_indices(), &[0, 1]);
    assert_eq!(actual.processed_video, expected);
}
