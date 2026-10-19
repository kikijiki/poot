use super::*;

#[test]
fn qwen25_vl_image_processor_config_overlays_released_defaults() {
    let defaults_dir = r#"{"image_processor_type":"Qwen2VLImageProcessor","unknown_field":17}"#;
    assert_eq!(
        processor_config(defaults_dir).unwrap(),
        Qwen25VlImageProcessorConfig::default()
    );

    let overrides_dir = r#"{
            "do_resize": false,
            "resample": 3,
            "do_rescale": false,
            "rescale_factor": 0.5,
            "do_normalize": false,
            "image_mean": 0.25,
            "image_std": [1.0, 2.0, 4.0],
            "do_convert_rgb": true,
            "min_pixels": 4,
            "max_pixels": 64,
            "patch_size": 2,
            "temporal_patch_size": 3,
            "merge_size": 1
        }"#;
    let config = processor_config(overrides_dir).unwrap();
    assert!(!config.do_resize);
    assert!(!config.do_rescale);
    assert!(!config.do_normalize);
    assert_eq!(config.min_pixels, 4);
    assert_eq!(config.max_pixels, 64);
    assert_eq!(config.patch_size, 2);
    assert_eq!(config.temporal_patch_size(), 3);
    assert_eq!(config.merge_size(), 1);
    assert_eq!(config.rescale_factor, 0.5);
    assert_eq!(config.mean, [0.25; 3]);
    assert_eq!(config.std, [1.0, 2.0, 4.0]);
}

#[test]
fn qwen25_vl_image_processor_config_accepts_released_shape_and_rejects_null_fields() {
    let released = r#"{
            "do_convert_rgb": true,
            "do_normalize": true,
            "do_rescale": true,
            "do_resize": true,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_processor_type": "Qwen2VLImageProcessor",
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "max_pixels": 1003520,
            "merge_size": 2,
            "min_pixels": 3136,
            "patch_size": 14,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {"shortest_edge": 3136, "longest_edge": 1003520},
            "temporal_patch_size": 2
        }"#;
    assert_eq!(
        processor_config(released).unwrap(),
        Qwen25VlImageProcessorConfig::default()
    );

    // Omission selects the default; a present null must not.
    for field in [
        "do_resize",
        "resample",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "do_convert_rgb",
        "min_pixels",
        "max_pixels",
        "patch_size",
        "temporal_patch_size",
        "merge_size",
    ] {
        let dir = format!(r#"{{"{field}":null}}"#);
        assert!(matches!(
            processor_config(&dir),
            Err(Qwen25VlImageProcessorConfigError::Parse(_))
        ));
    }

    for (name, json) in [
        ("bool_number", r#"{"do_resize":0}"#),
        ("resample_string", r#"{"resample":"3"}"#),
        ("factor_string", r#"{"rescale_factor":"0.5"}"#),
        ("mean_object", r#"{"image_mean":{"r":0.1}}"#),
        ("pixels_float", r#"{"min_pixels":3136.0}"#),
    ] {
        assert!(
            matches!(
                processor_config(json),
                Err(Qwen25VlImageProcessorConfigError::Parse(_))
            ),
            "{name}"
        );
    }
}

#[test]
fn qwen25_vl_image_processor_config_reports_typed_validation_errors() {
    for (name, json, expected) in [
        (
            "zero_min",
            r#"{"min_pixels":0}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroValue {
                field: "min_pixels",
            },
        ),
        (
            "zero_max",
            r#"{"max_pixels":0}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroValue {
                field: "max_pixels",
            },
        ),
        (
            "zero_patch",
            r#"{"patch_size":0}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroValue {
                field: "patch_size",
            },
        ),
        (
            "zero_temporal_patch",
            r#"{"temporal_patch_size":0}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroValue {
                field: "temporal_patch_size",
            },
        ),
        (
            "zero_merge",
            r#"{"merge_size":0}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroValue {
                field: "merge_size",
            },
        ),
        (
            "range",
            r#"{"min_pixels":785,"max_pixels":784}"#,
            Qwen25VlImageProcessorConfigValidationError::InvalidPixelRange {
                min_pixels: 785,
                max_pixels: 784,
            },
        ),
        (
            "factor_overflow",
            &format!(r#"{{"patch_size":{},"merge_size":2}}"#, usize::MAX),
            Qwen25VlImageProcessorConfigValidationError::ArithmeticOverflow {
                stage: "resize factor",
            },
        ),
        (
            "factor_area_overflow",
            &format!(
                r#"{{"patch_size":{},"merge_size":1,"max_pixels":{}}}"#,
                usize::MAX / 2,
                usize::MAX
            ),
            Qwen25VlImageProcessorConfigValidationError::ArithmeticOverflow {
                stage: "resize factor area",
            },
        ),
        (
            "max_below_factor",
            r#"{"patch_size":14,"merge_size":2,"min_pixels":1,"max_pixels":783}"#,
            Qwen25VlImageProcessorConfigValidationError::MaxPixelsBelowFactorArea {
                max_pixels: 783,
                factor: 28,
            },
        ),
        (
            "mean_nonfinite_f32",
            r#"{"image_mean":[1e100,0,0]}"#,
            Qwen25VlImageProcessorConfigValidationError::NonFiniteChannelValue {
                field: "image_mean",
                index: 0,
            },
        ),
        (
            "nonfinite_output_range",
            r#"{"rescale_factor":1e100}"#,
            Qwen25VlImageProcessorConfigValidationError::NonFiniteOutputRange { channel: 0 },
        ),
        (
            "std_zero",
            r#"{"image_std":[1,0,1]}"#,
            Qwen25VlImageProcessorConfigValidationError::ZeroStd { index: 1 },
        ),
        (
            "mean_channels",
            r#"{"image_mean":[0,1]}"#,
            Qwen25VlImageProcessorConfigValidationError::ChannelCount {
                field: "image_mean",
                actual: 2,
            },
        ),
        (
            "resample",
            r#"{"resample":2}"#,
            Qwen25VlImageProcessorConfigValidationError::UnsupportedResample { actual: 2 },
        ),
        (
            "rgb",
            r#"{"do_convert_rgb":false}"#,
            Qwen25VlImageProcessorConfigValidationError::RgbConversionDisabled,
        ),
    ] {
        assert_eq!(config_validation_error(json), expected, "{name}");
    }
}

#[test]
fn qwen25_vl_image_processor_config_validation_precedence_is_stable() {
    assert_eq!(
        config_validation_error(
            r#"{"image_mean":[0,1],"min_pixels":0,"resample":2,"do_convert_rgb":false}"#,
        ),
        Qwen25VlImageProcessorConfigValidationError::ChannelCount {
            field: "image_mean",
            actual: 2,
        }
    );
    assert_eq!(
        config_validation_error(
            r#"{"min_pixels":0,"rescale_factor":1e100,"resample":2,"do_convert_rgb":false}"#,
        ),
        Qwen25VlImageProcessorConfigValidationError::ZeroValue {
            field: "min_pixels",
        }
    );
    assert_eq!(
        config_validation_error(r#"{"rescale_factor":1e100,"resample":2,"do_convert_rgb":false}"#,),
        Qwen25VlImageProcessorConfigValidationError::NonFiniteOutputRange { channel: 0 }
    );
    assert_eq!(
        config_validation_error(r#"{"resample":2,"do_convert_rgb":false}"#,),
        Qwen25VlImageProcessorConfigValidationError::UnsupportedResample { actual: 2 }
    );
}
