use super::*;

#[test]
fn qwen25_vl_configured_image_preprocessing_honors_independent_toggles() {
    let decoded = Qwen25VlDecodedImage::from_rgb8(2, 2, rgb_fixture(2, 2)).unwrap();
    let config = |name, fields: &str| {
        let json = format!(
            r#"{{"do_resize":false,"min_pixels":1,"max_pixels":16,"patch_size":1,"temporal_patch_size":1,"merge_size":1,{fields}}}"#
        );
        let dir = processor_config_fixture(name, &json);
        load_qwen2_5_vl_image_processor_config(&dir).unwrap()
    };

    let raw = preprocess_qwen2_5_vl_image_with_config(
        &decoded,
        &config("raw_toggles", r#""do_rescale":false,"do_normalize":false"#),
    )
    .unwrap();
    assert_eq!(raw.grid, MropeGrid::new(1, 2, 2));
    assert_eq!(raw.pixel_values.shape(), vec![4, 3]);
    assert_eq!(
        raw.pixel_values.as_f32().unwrap(),
        &[
            0.0, 20.0, 40.0, 1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0
        ]
    );

    let rescaled = preprocess_qwen2_5_vl_image_with_config(
        &decoded,
        &config(
            "rescale_toggle",
            r#""do_rescale":true,"rescale_factor":0.5,"do_normalize":false"#,
        ),
    )
    .unwrap();
    assert_eq!(
        &rescaled.pixel_values.as_f32().unwrap()[..3],
        &[0.0, 10.0, 20.0]
    );

    let normalized = preprocess_qwen2_5_vl_image_with_config(
        &decoded,
        &config(
            "normalize_toggle",
            r#""do_rescale":false,"do_normalize":true,"image_mean":[1,2,3],"image_std":[1,2,4]"#,
        ),
    )
    .unwrap();
    assert_eq!(
        &normalized.pixel_values.as_f32().unwrap()[..3],
        &[-1.0, 9.0, 9.25]
    );
}

#[test]
fn qwen25_vl_configured_preprocessing_matches_frozen_transformers_f32_bits() {
    let decoded = Qwen25VlDecodedImage::from_rgb8(2, 2, rgb_fixture(2, 2)).unwrap();
    let config = processor_config_fixture(
        "f64_factor",
        r#"{
            "do_resize": false,
            "do_rescale": true,
            "rescale_factor": 0.12345678901234567,
            "do_normalize": true,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.7, 0.8, 0.9],
            "min_pixels": 1,
            "max_pixels": 16,
            "patch_size": 1,
            "temporal_patch_size": 3,
            "merge_size": 1
        }"#,
    );
    let config = load_qwen2_5_vl_image_processor_config(&config).unwrap();
    let actual = preprocess_qwen2_5_vl_image_with_config(&decoded, &config).unwrap();

    // Frozen from Transformers 4.49.0 Qwen2VLImageProcessor with NumPy 2.5.2: covers f64
    // multiplication, the f32 cast, f32 normalization, and temporal-column ordering.
    let expected = [
        0xbe12_4925,
        0xbe12_4925,
        0xbe12_4925,
        0x4035_87e6,
        0x4035_87e6,
        0x4035_87e6,
        0x40a4_ea94,
        0x40a4_ea94,
        0x40a4_ea94,
        0x3d09_4178,
        0x3d09_4178,
        0x3d09_4178,
        0x403f_684b,
        0x403f_684b,
        0x403f_684b,
        0x40a9_4e4f,
        0x40a9_4e4f,
        0x40a9_4e4f,
        0x3e56_e9e1,
        0x3e56_e9e1,
        0x3e56_e9e1,
        0x4049_48b1,
        0x4049_48b1,
        0x4049_48b1,
        0x40ad_b20a,
        0x40ad_b20a,
        0x40ad_b20a,
        0x3ec5_c1b1,
        0x3ec5_c1b1,
        0x3ec5_c1b1,
        0x4053_2916,
        0x4053_2916,
        0x4053_2916,
        0x40b2_15c5,
        0x40b2_15c5,
        0x40b2_15c5,
    ];
    assert_eq!(
        actual
            .pixel_values
            .as_f32()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn qwen25_vl_configured_preprocessing_rejects_geometry_before_allocation() {
    let nondivisible_dir = processor_config_fixture(
        "nondivisible",
        r#"{"do_resize":false,"min_pixels":1,"max_pixels":64,"patch_size":2,"merge_size":2}"#,
    );
    let nondivisible_config = load_qwen2_5_vl_image_processor_config(&nondivisible_dir).unwrap();
    let decoded = Qwen25VlDecodedImage::from_rgb8(4, 6, rgb_fixture(4, 6)).unwrap();
    assert_eq!(
        preprocess_qwen2_5_vl_image_with_config(&decoded, &nondivisible_config),
        Err(Qwen25VlImagePreprocessError::NonDivisibleWithoutResize {
            height: 4,
            width: 6,
            factor: 4,
        })
    );

    let empty_output_dir =
        processor_config_fixture("empty_output", r#"{"min_pixels":1,"max_pixels":784}"#);
    let empty_output_config = load_qwen2_5_vl_image_processor_config(&empty_output_dir).unwrap();
    let panoramic = Qwen25VlDecodedImage::from_rgb8(28, 5600, rgb_fixture(28, 5600)).unwrap();
    assert_eq!(
        preprocess_qwen2_5_vl_image_with_config(&panoramic, &empty_output_config),
        Err(Qwen25VlImagePreprocessError::ZeroOutputDimension {
            height: 0,
            width: 392,
        })
    );

    let cap_dir = processor_config_fixture(
        "output_cap",
        &format!(r#"{{"min_pixels":{0},"max_pixels":{0}}}"#, MAX_IMAGE_PIXELS),
    );
    let cap_config = load_qwen2_5_vl_image_processor_config(&cap_dir).unwrap();
    let small = Qwen25VlDecodedImage::from_rgb8(28, 28, rgb_fixture(28, 28)).unwrap();
    assert!(matches!(
        preprocess_qwen2_5_vl_image_with_config(&small, &cap_config),
        Err(Qwen25VlImagePreprocessError::TooManyOutputPixels { .. })
    ));
    assert_eq!(
        preprocess_qwen2_5_vl_images(&[]),
        Err(Qwen25VlImagePreprocessError::EmptyBatch)
    );
}

#[test]
fn qwen25_vl_image_batch_concatenates_in_order_into_processed_media() {
    let images = [
        Qwen25VlDecodedImage::from_rgb8(28, 28, rgb_fixture(28, 28)).unwrap(),
        Qwen25VlDecodedImage::from_rgb8(28, 56, rgb_fixture(28, 56)).unwrap(),
    ];
    let first = preprocess_qwen2_5_vl_image(&images[0]).unwrap();
    let second = preprocess_qwen2_5_vl_image(&images[1]).unwrap();
    let batch = preprocess_qwen2_5_vl_images(&images).unwrap();
    let mut expected = first.pixel_values.as_f32().unwrap().to_vec();
    expected.extend_from_slice(second.pixel_values.as_f32().unwrap());
    assert_eq!(batch.grids, vec![first.grid, second.grid]);
    assert_eq!(
        batch.pixel_values.shape(),
        vec![
            first.pixel_values.shape()[0] + second.pixel_values.shape()[0],
            first.pixel_values.shape()[1],
        ]
    );
    assert_eq!(batch.pixel_values.as_f32().unwrap(), expected.as_slice());

    let media = batch.processor_media();
    assert!(std::ptr::eq(
        media.pixel_values.unwrap(),
        &batch.pixel_values
    ));
    assert_eq!(media.image_grids.as_ptr(), batch.grids.as_ptr());
    assert_eq!(media.image_grids.len(), batch.grids.len());
    assert!(media.pixel_values_videos.is_none());
    assert!(media.video_grids.is_empty());
    assert_eq!(media.validate(), Ok(()));

    let configured =
        preprocess_qwen2_5_vl_images_with_config(&images, &Qwen25VlImageProcessorConfig::default())
            .unwrap();
    assert_eq!(configured.pixel_values, batch.pixel_values);
    assert_eq!(configured.grids, batch.grids);
}

#[test]
fn qwen25_vl_image_batch_matches_independent_item_and_merge_order_fixture() {
    let image = |height: usize, width: usize, base: u8| {
        let rgb = (0..height * width)
            .flat_map(|pixel| {
                let value = base + pixel as u8;
                [value, value + 40, value + 80]
            })
            .collect();
        Qwen25VlDecodedImage::from_rgb8(height, width, rgb).unwrap()
    };
    let images = [image(2, 2, 0), image(2, 4, 100)];
    let config = processor_config_fixture(
        "independent_batch_order",
        r#"{
            "do_resize": false,
            "do_rescale": false,
            "do_normalize": false,
            "min_pixels": 1,
            "max_pixels": 64,
            "patch_size": 1,
            "temporal_patch_size": 1,
            "merge_size": 2
        }"#,
    );
    let config = load_qwen2_5_vl_image_processor_config(&config).unwrap();
    let actual = preprocess_qwen2_5_vl_images_with_config(&images, &config).unwrap();

    // Literal upstream item-major order, then each item's merge-block order; not built by the
    // single-image implementation under test.
    let expected_pixels = [0u8, 1, 2, 3, 100, 101, 104, 105, 102, 103, 106, 107];
    let expected = expected_pixels
        .into_iter()
        .flat_map(|value| [value as f32, (value + 40) as f32, (value + 80) as f32])
        .collect::<Vec<_>>();
    assert_eq!(
        actual.grids,
        [MropeGrid::new(1, 2, 2), MropeGrid::new(1, 2, 4)]
    );
    assert_eq!(actual.pixel_values.shape(), [12, 3]);
    assert_eq!(actual.pixel_values.as_f32().unwrap(), expected);
}

#[test]
fn qwen25_vl_image_smart_resize_matches_upstream_integer_contract() {
    // Frozen outputs from Transformers 4.49.0 `smart_resize` (CPython): no-op, ties-to-even,
    // minimum-pixel expansion, maximum-pixel shrink, non-square scaling, inclusive aspect-ratio
    // boundary.
    for (input, expected) in [
        ((56, 84), (56, 84)),
        ((70, 126), (56, 112)),
        ((98, 154), (112, 168)),
        ((42, 70), (56, 56)),
        ((28, 28), (56, 56)),
        ((28, 56), (56, 84)),
        ((1400, 1400), (980, 980)),
        ((1000, 2000), (700, 1400)),
        ((300, 4000), (252, 3640)),
        ((28, 5600), (28, 5600)),
        ((4096, 8192), (700, 1400)),
    ] {
        assert_eq!(
            smart_resize(input.0, input.1, &Qwen25VlImageProcessorConfig::default()),
            Ok(expected)
        );
    }
    assert_eq!(
        smart_resize(27, 56, &Qwen25VlImageProcessorConfig::default()),
        Err(Qwen25VlImagePreprocessError::EdgeBelowFactor {
            height: 27,
            width: 56,
            factor: 28,
        })
    );
    assert_eq!(
        smart_resize(28, 5628, &Qwen25VlImageProcessorConfig::default()),
        Err(Qwen25VlImagePreprocessError::AspectRatioTooLarge {
            height: 28,
            width: 5628,
        })
    );
}

#[test]
fn qwen25_vl_image_decoded_input_rejects_invalid_shapes_and_payloads() {
    assert_eq!(
        Qwen25VlDecodedImage::from_rgb8(0, 1, vec![]),
        Err(Qwen25VlImageDecodeError::ZeroDimension {
            height: 0,
            width: 1,
        })
    );
    assert_eq!(
        Qwen25VlDecodedImage::from_rgb8(2, 3, vec![0; 17]),
        Err(Qwen25VlImageDecodeError::PixelDataLen {
            expected: 18,
            actual: 17,
        })
    );
    assert_eq!(
        Qwen25VlDecodedImage::from_rgb8(u32::MAX as usize, u32::MAX as usize, vec![]),
        Err(Qwen25VlImageDecodeError::PixelCountOverflow {
            height: u32::MAX as usize,
            width: u32::MAX as usize,
        })
    );
    assert!(matches!(
        Qwen25VlDecodedImage::from_rgb8(8192, 4097, vec![]),
        Err(Qwen25VlImageDecodeError::TooManyPixels { .. })
    ));
    // Dimension/cap validation runs before payload validation.
    assert!(matches!(
        Qwen25VlDecodedImage::from_rgb8(8192, 4097, vec![0]),
        Err(Qwen25VlImageDecodeError::TooManyPixels { .. })
    ));
}

#[test]
fn qwen25_vl_image_patchify_matches_upstream_axis_order() {
    let config = Qwen25VlImageProcessorConfig {
        do_resize: true,
        do_rescale: true,
        do_normalize: true,
        min_pixels: 4,
        max_pixels: 16,
        patch_size: 1,
        temporal_patch_size: 2,
        merge_size: 2,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let decoded = Qwen25VlDecodedImage::from_rgb8(4, 4, rgb_fixture(4, 4)).unwrap();
    let actual = preprocess_qwen2_5_vl_image_with_config(&decoded, &config).unwrap();
    assert_eq!(actual.grid, MropeGrid::new(1, 4, 4));
    assert_eq!(actual.pixel_values.shape(), vec![16, 6]);
    let row_pixels = [0usize, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    let expected = row_pixels
        .into_iter()
        .flat_map(|pixel| {
            let value = pixel as f32;
            [
                value,
                value,
                value + 20.0,
                value + 20.0,
                value + 40.0,
                value + 40.0,
            ]
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.pixel_values.as_f32().unwrap(), expected.as_slice());

    let patch_config = Qwen25VlImageProcessorConfig {
        do_resize: true,
        do_rescale: true,
        do_normalize: true,
        min_pixels: 4,
        max_pixels: 4,
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 1,
        rescale_factor: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };
    let patched = preprocess_qwen2_5_vl_image_with_config(
        &Qwen25VlDecodedImage::from_rgb8(2, 2, rgb_fixture(2, 2)).unwrap(),
        &patch_config,
    )
    .unwrap();
    assert_eq!(patched.pixel_values.shape(), vec![1, 24]);
    assert_eq!(
        patched.pixel_values.as_f32().unwrap(),
        &[
            0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0,
            40.0, 41.0, 42.0, 43.0, 40.0, 41.0, 42.0, 43.0,
        ]
    );
}

#[test]
#[allow(clippy::excessive_precision)] // Mirror the upstream constants used by the implementation.
fn qwen25_vl_image_default_output_satisfies_processed_media_contract() {
    let decoded = Qwen25VlDecodedImage::from_rgb8(56, 56, rgb_fixture(56, 56)).unwrap();
    let actual = preprocess_qwen2_5_vl_image(&decoded).unwrap();
    assert_eq!(actual.grid, MropeGrid::new(1, 4, 4));
    assert_eq!(actual.pixel_values.shape(), vec![16, 1176]);
    assert!(
        actual
            .pixel_values
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    // Frozen NumPy f32 bits from the Transformers 4.49.0 rescale/normalize path: f64 `1/255`
    // multiply, f32 cast, f32 subtraction/division, channel order, temporal duplication. Expected
    // values are not recomputed in Rust.
    for (index, expected_bits) in [
        (0, 0xbfe5_68dc),
        (196, 0xbfe5_68dc),
        (392, 0xbfb9_d93a),
        (784, 0xbf69_52a1),
    ] {
        assert_eq!(
            actual.pixel_values.as_f32().unwrap()[index].to_bits(),
            expected_bits
        );
    }
    let grids = [actual.grid];
    let media = Qwen25VlProcessorMedia {
        pixel_values: Some(&actual.pixel_values),
        image_grids: &grids,
        pixel_values_videos: None,
        video_grids: &[],
    };
    assert_eq!(media.validate(), Ok(()));
    assert_eq!(Qwen25VlProcessorMediaKind::Image.to_string(), "image");
}

#[test]
fn qwen25_vl_image_decoder_accepts_in_memory_png_and_jpeg() {
    let source = image::RgbImage::from_raw(28, 28, rgb_fixture(28, 28)).unwrap();
    for format in [ImageFormat::Png, ImageFormat::Jpeg] {
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(source.clone())
            .write_to(&mut bytes, format)
            .unwrap();
        let decoded = decode_qwen2_5_vl_image(bytes.get_ref()).unwrap();
        assert_eq!((decoded.height(), decoded.width()), (28, 28));
        if format == ImageFormat::Png {
            assert_eq!(decoded.rgb(), source.as_raw());
        }
        let processed = preprocess_qwen2_5_vl_image(&decoded).unwrap();
        assert_eq!(processed.grid, MropeGrid::new(1, 4, 4));
    }
    assert!(matches!(
        decode_qwen2_5_vl_image(b"not an image"),
        Err(Qwen25VlImageDecodeError::Identify { .. })
    ));
}

#[test]
fn qwen25_vl_image_decoder_uses_header_gate_and_rgb_conversion() {
    use base64::Engine as _;

    // Emitted by Pillow, not the image crate under test. PIL `convert("RGB")` discards alpha and
    // replicates 8-bit luminance, matching the decoded RGB8 contract.
    for (encoded, expected) in [
        (
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGNkZGJmYGZm/g8AAVQBEGAwCA8AAAAASUVORK5CYII=",
            &[1, 2, 3, 4, 5, 6][..],
        ),
        (
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAAAAADRSSBWAAAAC0lEQVR4nGNgPwEAANkA0NemIjwAAAAASUVORK5CYII=",
            &[7, 7, 7, 200, 200, 200][..],
        ),
    ] {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let decoded = decode_qwen2_5_vl_image(&bytes).unwrap();
        assert_eq!((decoded.height(), decoded.width()), (1, 2));
        assert_eq!(decoded.rgb(), expected);
    }

    // Pillow-emitted JPEG, so this crate's encoder is not its own oracle. JPEG pixels are lossy.
    let jpeg = base64::engine::general_purpose::STANDARD
        .decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAIDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDtv2O/+TR/hZ/2TjQ//SCCvw/PP+R1if8Ar5P/ANKZ+x5P/wAijD/9e4f+ko//2Q==")
        .unwrap();
    let decoded = decode_qwen2_5_vl_image(&jpeg).unwrap();
    assert_eq!((decoded.height(), decoded.width()), (1, 2));

    // A valid PNG IHDR advertising 8192x4097 with no pixel stream: the header-only gate must
    // return the cap error before any pixel allocation or missing-data error.
    let oversized_header = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAIAAAABABCAIAAACe9yv5AAAACElEQVR4nAMAAAAAAUgGidIAAAAASUVORK5CYII=")
        .unwrap();
    assert!(matches!(
        decode_qwen2_5_vl_image(&oversized_header),
        Err(Qwen25VlImageDecodeError::TooManyPixels {
            height: 4097,
            width: 8192,
            ..
        })
    ));
    assert_eq!(
        decode_qwen2_5_vl_image(b"GIF89a"),
        Err(Qwen25VlImageDecodeError::UnsupportedFormat {
            format: "Gif".to_string(),
        })
    );
}

#[test]
fn qwen25_vl_catmull_rom_tracks_frozen_pillow_bicubic_fixture() {
    let rgb = (0..28usize)
        .flat_map(|y| {
            (0..28usize).flat_map(move |x| {
                [
                    (x * 17 + y * 29 + 3) as u8,
                    (x * x + 7 * y + 11) as u8,
                    (13 * x + y * y + 19) as u8,
                ]
            })
        })
        .collect::<Vec<_>>();
    let decoded = Qwen25VlDecodedImage::from_rgb8(28, 28, rgb).unwrap();
    let actual = preprocess_qwen2_5_vl_image(&decoded).unwrap();
    assert_eq!(actual.grid, MropeGrid::new(1, 4, 4));

    // Frozen flattened f32 outputs from Pillow BICUBIC plus the Transformers 4.49.0 NumPy pipeline.
    // `image` Catmull-Rom is the same cubic family but not Pillow's exact coefficient arithmetic; for
    // this fixture RGB8 samples differ by at most one level (normalized delta below 0.0151). No
    // pixel identity or general bound is claimed.
    for (index, expected_bits) in [
        (0, 0xbfe5_68dc),
        (393, 0xbfcd_0ef9),
        (798, 0xbf9c_b482),
        (1361, 0xbf4c_b096),
        (4115, 0xbfd6_a9d8),
        (14896, 0x3ec3_e553),
        (17835, 0x3fc0_e4c8),
    ] {
        let expected = f32::from_bits(expected_bits);
        assert!((actual.pixel_values.as_f32().unwrap()[index] - expected).abs() < 0.0151);
    }
}
