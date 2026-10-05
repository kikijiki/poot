use super::*;

const RAW_VISION_START: &str = "<|vision_start|>";

const RAW_IMAGE_PAD: &str = "<|image_pad|>";

const RAW_VIDEO_PAD: &str = "<|video_pad|>";

const RAW_VISION_START_ID: u32 = 10;

const RAW_IMAGE_ID: u32 = 11;

const RAW_VIDEO_ID: u32 = 12;

fn raw_request_fixture(name: &str, config_image_id: u32) -> FixtureDir {
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    let dir = processor_config_fixture(
        name,
        r#"{
            "do_resize":false,"do_rescale":false,"do_normalize":false,
            "min_pixels":1,"max_pixels":1024,"patch_size":1,
            "temporal_patch_size":2,"merge_size":1
        }"#,
    );
    let config = format!(
        r#"{{
            "model_type":"qwen2_5_vl","vocab_size":100,"hidden_size":8,
            "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":1,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
            "max_position_embeddings":128,"vision_start_token_id":{RAW_VISION_START_ID},
            "image_token_id":{config_image_id},"video_token_id":{RAW_VIDEO_ID},
            "rope_scaling":{{"mrope_section":[1,1,2]}},
            "vision_config":{{"tokens_per_second":4}}
        }}"#
    );
    fs::write(dir.join("config.json"), config).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        format!(
            r#"{{"vision_start_token":"{RAW_VISION_START}","image_token":"{RAW_IMAGE_PAD}","video_token":"{RAW_VIDEO_PAD}"}}"#
        ),
    )
    .unwrap();
    let mut vocab = HashMap::from([
        ("t0".to_string(), 0),
        ("t1".to_string(), 1),
        ("t2".to_string(), 2),
        (RAW_VISION_START.to_string(), RAW_VISION_START_ID),
        (RAW_IMAGE_PAD.to_string(), RAW_IMAGE_ID),
        (RAW_VIDEO_PAD.to_string(), RAW_VIDEO_ID),
    ]);
    for id in 0..=RAW_VIDEO_ID.max(config_image_id) {
        if !vocab.values().any(|&existing| existing == id) {
            vocab.insert(format!("fill{id}"), id);
        }
    }
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    use tokenizers::AddedToken;
    tokenizer.add_special_tokens(&[
        AddedToken::from(RAW_VISION_START, true),
        AddedToken::from(RAW_IMAGE_PAD, true),
        AddedToken::from(RAW_VIDEO_PAD, true),
    ]);
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    dir
}

fn raw_image(value: u8) -> Qwen25VlDecodedImage {
    Qwen25VlDecodedImage::from_rgb8(1, 1, vec![value; 3]).unwrap()
}

fn raw_video_source(values: &[u8]) -> RecordingFrameSource {
    RecordingFrameSource {
        frames: values.iter().copied().map(raw_image).collect(),
        calls: Vec::new(),
        fail_at: None,
    }
}

fn raw_video_metadata<'a>(timestamps: &'a [f64]) -> Qwen25VlVideoMetadata<'a> {
    Qwen25VlVideoMetadata {
        duration_seconds: timestamps.len() as f64 / 2.0,
        source_frames_per_second: 2.0,
        frame_count: timestamps.len(),
        frame_timestamps_seconds: timestamps,
    }
}

fn raw_image_hw(height: usize, width: usize, value: u8) -> Qwen25VlDecodedImage {
    Qwen25VlDecodedImage::from_rgb8(
        height,
        width,
        vec![value; height.checked_mul(width).unwrap().checked_mul(3).unwrap()],
    )
    .unwrap()
}

#[test]
fn qwen25_vl_rendered_pads_expand_from_merged_grid_counts() {
    let image = MropeGrid::new(1, 2, 2);
    let video = MropeGrid::new(1, 2, 2);
    let rendered = format!(
        "t0 {RAW_VISION_START}{RAW_IMAGE_PAD} t1 {RAW_VISION_START}{RAW_VIDEO_PAD} t2 {RAW_VISION_START}{RAW_IMAGE_PAD}"
    );
    let expanded = expand_qwen2_5_vl_rendered_pads(
        &rendered,
        RAW_IMAGE_PAD,
        RAW_VIDEO_PAD,
        &[image, image],
        &[video],
        1,
    )
    .unwrap();
    let image_run = RAW_IMAGE_PAD.repeat(4);
    let video_run = RAW_VIDEO_PAD.repeat(4);
    assert_eq!(
        expanded,
        format!(
            "t0 {RAW_VISION_START}{image_run} t1 {RAW_VISION_START}{video_run} t2 {RAW_VISION_START}{image_run}"
        )
    );
    assert_eq!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[MropeGrid::new(1, 1, 1)],
            &[],
            1,
        )
        .unwrap(),
        format!("{RAW_VISION_START}{RAW_IMAGE_PAD}")
    );
    assert_eq!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[MropeGrid::new(1, 4, 4)],
            &[],
            2,
        )
        .unwrap(),
        format!("{RAW_VISION_START}{}", RAW_IMAGE_PAD.repeat(4))
    );
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[image],
            &[],
            1,
        ),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Image,
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[image, image],
            &[],
            1,
        ),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Image,
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            "",
            RAW_VIDEO_PAD,
            &[image],
            &[],
            1,
        ),
        Err(Qwen25VlPadExpansionError::EmptyPadMarker {
            kind: MropeVisualKind::Image
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads("t0 t1", RAW_IMAGE_PAD, RAW_VIDEO_PAD, &[image], &[], 1),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Image,
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[],
            &[],
            1,
        ),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Image,
            expected: 0,
            actual: 1
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!(
                "{RAW_VISION_START}{RAW_IMAGE_PAD}{RAW_VISION_START}{RAW_VIDEO_PAD}{RAW_VIDEO_PAD}"
            ),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[image],
            &[video],
            1,
        ),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Video,
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        expand_qwen2_5_vl_rendered_pads(
            &format!("{RAW_VISION_START}{RAW_IMAGE_PAD}"),
            RAW_IMAGE_PAD,
            RAW_VIDEO_PAD,
            &[image],
            &[video],
            1,
        ),
        Err(Qwen25VlPadExpansionError::PadCountMismatch {
            kind: MropeVisualKind::Video,
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn qwen25_vl_raw_request_assembles_image_only_and_video_only() {
    let dir = raw_request_fixture("raw_request_single_kind", RAW_IMAGE_ID);
    let mut no_video_sources: [RecordingFrameSource; 0] = [];
    let image = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Image(raw_image(7))],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap();
    assert_eq!(image.media_order(), [MropeVisualKind::Image]);
    assert_eq!(image.video_sample_plans(), []);
    assert_eq!(image.media.image_grids, [MropeGrid::new(1, 1, 1)]);
    assert!(image.media.pixel_values.is_some());
    assert!(image.media.pixel_values_videos.is_none());
    assert_eq!(image.processor_media().validate(), Ok(()));
    assert_eq!(image.prompt_token_ids.len(), image.mrope_positions.len());
    assert_eq!(
        Qwen25VlMropeModelInputView::image_grids(&image),
        image.media.image_grids.as_slice()
    );

    let processor_config = load_qwen2_5_vl_image_processor_config(&dir).unwrap();
    let direct_images =
        preprocess_qwen2_5_vl_images_with_config(&[raw_image(7)], &processor_config).unwrap();
    let direct_media =
        assemble_qwen2_5_vl_processed_media(Some(direct_images), vec![], &processor_config)
            .unwrap();
    let direct = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
        Qwen25VlProcessorMediaMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: direct_media.processor_media(),
            spatial_merge_size: processor_config.merge_size(),
            temporal_patch_size: MropeTemporalPatchSize::new(
                processor_config.temporal_patch_size(),
            )
            .unwrap(),
            fps: None,
        },
    )
    .unwrap();
    assert_eq!(image.media, direct_media);
    assert_eq!(image.prompt_token_ids, direct.prompt_token_ids);
    assert_eq!(image.metadata, direct.metadata);
    assert_eq!(image.mrope_positions, direct.mrope_positions);

    let timestamps = [0.0, 0.5];
    let mut sources = [raw_video_source(&[10, 20])];
    let video = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|video_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Video(raw_video_metadata(
                &timestamps,
            ))],
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            video_sources: &mut sources,
        },
    )
    .unwrap();
    assert_eq!(video.media_order(), [MropeVisualKind::Video]);
    assert_eq!(video.media.sampled_frames_per_second(), [2.0]);
    assert_eq!(video.video_sample_plans()[0].frame_indices(), [0, 1]);
    assert_eq!(sources[0].calls, [0, 1]);
    assert!(video.media.pixel_values.is_none());
    assert!(video.media.pixel_values_videos.is_some());
}

#[test]
fn qwen25_vl_raw_request_expands_merged_image_pads_before_tokenization() {
    let dir = raw_request_fixture("raw_request_pad_expansion", RAW_IMAGE_ID);
    let mut no_video_sources: [RecordingFrameSource; 0] = [];
    let image = raw_image_hw(2, 2, 7);
    let actual = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Image(image.clone())],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap();
    assert_eq!(actual.media.image_grids, [MropeGrid::new(1, 2, 2)]);
    assert_eq!(
        actual
            .prompt_token_ids
            .iter()
            .filter(|&&id| id == RAW_IMAGE_ID)
            .count(),
        4
    );

    let processor_config = load_qwen2_5_vl_image_processor_config(&dir).unwrap();
    let direct_images =
        preprocess_qwen2_5_vl_images_with_config(&[image], &processor_config).unwrap();
    let direct_media =
        assemble_qwen2_5_vl_processed_media(Some(direct_images), vec![], &processor_config)
            .unwrap();
    let unexpanded = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
        Qwen25VlProcessorMediaMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: direct_media.processor_media(),
            spatial_merge_size: processor_config.merge_size(),
            temporal_patch_size: MropeTemporalPatchSize::new(
                processor_config.temporal_patch_size(),
            )
            .unwrap(),
            fps: None,
        },
    );
    assert!(unexpanded.is_err());

    let extra = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> <|image_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Image(raw_image_hw(2, 2, 8))],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap_err();
    assert!(matches!(
        extra,
        Qwen25VlRawRequestAssemblyError::PadExpansion(
            Qwen25VlPadExpansionError::PadCountMismatch {
                kind: MropeVisualKind::Image,
                expected: 1,
                actual: 2
            }
        )
    ));

    let missing = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 t1",
            media: vec![Qwen25VlRawMediaInput::Image(raw_image_hw(2, 2, 9))],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap_err();
    assert!(matches!(
        missing,
        Qwen25VlRawRequestAssemblyError::PadExpansion(
            Qwen25VlPadExpansionError::PadCountMismatch {
                kind: MropeVisualKind::Image,
                expected: 1,
                actual: 0
            }
        )
    ));

    let leftover = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: vec![
                Qwen25VlRawMediaInput::Image(raw_image_hw(2, 2, 3)),
                Qwen25VlRawMediaInput::Image(raw_image_hw(2, 2, 4)),
            ],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap_err();
    assert!(matches!(
        leftover,
        Qwen25VlRawRequestAssemblyError::PadExpansion(
            Qwen25VlPadExpansionError::PadCountMismatch {
                kind: MropeVisualKind::Image,
                expected: 2,
                actual: 1
            }
        )
    ));

    let stray = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: vec![],
            fps: None,
            video_sources: &mut no_video_sources,
        },
    )
    .unwrap_err();
    assert!(matches!(
        stray,
        Qwen25VlRawRequestAssemblyError::PadExpansion(
            Qwen25VlPadExpansionError::PadCountMismatch {
                kind: MropeVisualKind::Image,
                expected: 0,
                actual: 1
            }
        )
    ));

    let timestamps = [0.0, 0.5];
    let mut extra_video_sources = [raw_video_source(&[10, 20])];
    let extra_video = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|video_pad|> <|video_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Video(raw_video_metadata(
                &timestamps,
            ))],
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            video_sources: &mut extra_video_sources,
        },
    )
    .unwrap_err();
    assert!(matches!(
        extra_video,
        Qwen25VlRawRequestAssemblyError::PadExpansion(
            Qwen25VlPadExpansionError::PadCountMismatch {
                kind: MropeVisualKind::Video,
                expected: 1,
                actual: 2
            }
        )
    ));
    assert_eq!(extra_video_sources[0].calls, [0, 1]);
}

#[test]
fn qwen25_vl_raw_request_preserves_interleaved_order_multiple_items_and_per_video_fps() {
    let dir = raw_request_fixture("raw_request_interleaved", RAW_IMAGE_ID);
    let first_timestamps = [0.0, 0.5];
    let second_timestamps = [0.0, 0.5];
    let fps = [2.0, 1.0];
    let mut sources = [raw_video_source(&[20, 21]), raw_video_source(&[30, 31])];
    let actual = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: concat!(
                "t0 <|vision_start|> <|image_pad|> t1 ",
                "<|vision_start|> <|video_pad|> t2 ",
                "<|vision_start|> <|image_pad|> t0 ",
                "<|vision_start|> <|video_pad|> t1"
            ),
            media: vec![
                Qwen25VlRawMediaInput::Image(raw_image(1)),
                Qwen25VlRawMediaInput::Video(raw_video_metadata(&first_timestamps)),
                Qwen25VlRawMediaInput::Image(raw_image(2)),
                Qwen25VlRawMediaInput::Video(raw_video_metadata(&second_timestamps)),
            ],
            fps: Some(MropeProcessorFpsInput::PerVideo(&fps)),
            video_sources: &mut sources,
        },
    )
    .unwrap();

    assert_eq!(
        actual.media_order(),
        [
            MropeVisualKind::Image,
            MropeVisualKind::Video,
            MropeVisualKind::Image,
            MropeVisualKind::Video,
        ]
    );
    assert_eq!(actual.media.image_grids.len(), 2);
    assert_eq!(actual.media.video_grids.len(), 2);
    assert_eq!(actual.media.sampled_frames_per_second(), fps);
    assert_eq!(actual.video_sample_plans()[0].frame_indices(), [0, 1]);
    assert_eq!(actual.video_sample_plans()[1].frame_indices(), [0]);
    assert_eq!(sources[0].calls, [0, 1]);
    assert_eq!(sources[1].calls, [0]);
    assert_eq!(actual.processor_media().validate(), Ok(()));
    assert_eq!(actual.prompt_token_ids.len(), actual.mrope_positions.len());
}

#[test]
fn qwen25_vl_raw_request_rejects_source_and_fps_counts_before_decode() {
    let dir = raw_request_fixture("raw_request_counts", RAW_IMAGE_ID);
    let timestamps = [0.0, 0.5];
    let video = || Qwen25VlRawMediaInput::Video(raw_video_metadata(&timestamps));
    let mut no_sources: [RecordingFrameSource; 0] = [];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![video()],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut no_sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoSourceCountMismatch {
            video_count: 1,
            source_count: 0
        })
    ));

    let mut one_source = [raw_video_source(&[1, 2])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![video()],
                fps: None,
                video_sources: &mut one_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::MissingVideoFps { video_count: 1 })
    ));
    assert!(one_source[0].calls.is_empty());

    let short_fps = [];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![video()],
                fps: Some(MropeProcessorFpsInput::PerVideo(&short_fps)),
                video_sources: &mut one_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoTiming(
            MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                expected: 1,
                actual: 0
            }
        ))
    ));
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![video()],
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
                video_sources: &mut one_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoTiming(
            MropeProcessorTimingError::NonFiniteSampledFramesPerSecond
        ))
    ));

    let mut extra_source = [raw_video_source(&[1, 2])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![],
                fps: None,
                video_sources: &mut extra_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoSourceCountMismatch {
            video_count: 0,
            source_count: 1
        })
    ));
    let mut no_sources: [RecordingFrameSource; 0] = [];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut no_sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::UnexpectedVideoFps)
    ));
}

#[test]
fn qwen25_vl_raw_request_reports_metadata_decode_and_preprocess_source_indices() {
    let dir = raw_request_fixture("raw_request_source_errors", RAW_IMAGE_ID);
    let timestamps = [0.0, 0.5];
    let mut sources = [raw_video_source(&[1, 2])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![
                    Qwen25VlRawMediaInput::Image(raw_image(1)),
                    Qwen25VlRawMediaInput::Video(Qwen25VlVideoMetadata {
                        duration_seconds: 0.0,
                        ..raw_video_metadata(&timestamps)
                    }),
                ],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoPreprocess {
            source_index: 1,
            video_index: 0,
            source: Qwen25VlRawVideoPreprocessError::Sampling(
                Qwen25VlVideoSamplingError::NonPositiveDuration { actual: 0.0 }
            )
        })
    ));
    assert!(sources[0].calls.is_empty());

    sources[0].fail_at = Some(1);
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![
                    Qwen25VlRawMediaInput::Image(raw_image(1)),
                    Qwen25VlRawMediaInput::Video(raw_video_metadata(&timestamps)),
                ],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoPreprocess {
            source_index: 1,
            video_index: 0,
            source: Qwen25VlRawVideoPreprocessError::Decode {
                frame_index: 1,
                source: FixtureVideoDecodeError(1)
            }
        })
    ));

    let mut malformed_source = [raw_video_source(&[1, 2])];
    malformed_source[0].frames[1] = Qwen25VlDecodedImage::from_rgb8(1, 2, vec![2; 6]).unwrap();
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![
                    Qwen25VlRawMediaInput::Image(raw_image(1)),
                    Qwen25VlRawMediaInput::Video(raw_video_metadata(&timestamps)),
                ],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut malformed_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoPreprocess {
            source_index: 1,
            source: Qwen25VlRawVideoPreprocessError::Preprocess(
                Qwen25VlVideoPreprocessError::FrameDimensionsMismatch { index: 1, .. }
            ),
            ..
        })
    ));

    let image_error_dir = raw_request_fixture("raw_request_image_error", RAW_IMAGE_ID);
    fs::write(
        image_error_dir.join(PREPROCESSOR_CONFIG_FILE),
        r#"{
            "do_resize":false,"do_rescale":false,"do_normalize":false,
            "min_pixels":1,"max_pixels":1024,"patch_size":2,
            "temporal_patch_size":2,"merge_size":1
        }"#,
    )
    .unwrap();
    let mut image_error_source = [raw_video_source(&[1, 2])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &image_error_dir,
                rendered_prompt: "t0",
                media: vec![
                    Qwen25VlRawMediaInput::Video(raw_video_metadata(&timestamps)),
                    Qwen25VlRawMediaInput::Image(raw_image(1)),
                ],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut image_error_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::ImagePreprocess {
            source_index: 1,
            source: Qwen25VlImagePreprocessError::NonDivisibleWithoutResize {
                height: 1,
                width: 1,
                factor: 2,
            }
        })
    ));
    assert!(image_error_source[0].calls.is_empty());
}

#[test]
fn qwen25_vl_raw_request_enforces_metadata_precedence_and_media_cap() {
    let mismatched = raw_request_fixture("raw_request_marker_config", 99);
    let mut no_sources: [RecordingFrameSource; 0] = [];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &mismatched,
                rendered_prompt: "t0",
                media: vec![Qwen25VlRawMediaInput::Image(raw_image(1))],
                fps: None,
                video_sources: &mut no_sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::Metadata(
            QwenVlMropeMetadataLoadError::SpecialTokens(
                poot_models::mrope::MropeSpecialTokenReconcileError::ConfigSpecialTokenMismatch { .. }
            )
        ))
    ));

    let dir = raw_request_fixture("raw_request_cap", RAW_IMAGE_ID);
    let media = (0..=QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS)
        .map(|_| Qwen25VlRawMediaInput::Image(raw_image(1)))
        .collect();
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media,
                fps: None,
                video_sources: &mut no_sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::TooManyMediaItems {
            actual,
            limit: QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS
        }) if actual == QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS + 1
    ));

    let overflow_timestamps = [0.0];
    let mut overflow_source = [raw_video_source(&[1])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0",
                media: vec![Qwen25VlRawMediaInput::Video(Qwen25VlVideoMetadata {
                    duration_seconds: 1.0,
                    source_frames_per_second: f64::MIN_POSITIVE,
                    frame_count: 1,
                    frame_timestamps_seconds: &overflow_timestamps,
                })],
                fps: Some(MropeProcessorFpsInput::Scalar(f64::MAX)),
                video_sources: &mut overflow_source,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::VideoPreprocess {
            source_index: 0,
            source: Qwen25VlRawVideoPreprocessError::Sampling(
                Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow
            ),
            ..
        })
    ));
    assert!(overflow_source[0].calls.is_empty());
}

#[test]
fn qwen25_vl_raw_request_rejects_cross_kind_order_without_publishing_partial_output() {
    let dir = raw_request_fixture("raw_request_order", RAW_IMAGE_ID);
    let mut no_sources: [RecordingFrameSource; 0] = [];
    let expanded = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|image_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Image(
                Qwen25VlDecodedImage::from_rgb8(1, 2, vec![1; 6]).unwrap(),
            )],
            fps: None,
            video_sources: &mut no_sources,
        },
    )
    .unwrap();
    assert_eq!(expanded.media.image_grids, [MropeGrid::new(1, 1, 2)]);
    assert_eq!(
        expanded
            .prompt_token_ids
            .iter()
            .filter(|&&id| id == RAW_IMAGE_ID)
            .count(),
        2
    );

    let timestamps = [0.0, 0.5];
    let mut sources = [raw_video_source(&[1, 2])];
    let result = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: concat!(
                "t0 <|vision_start|> <|video_pad|> t1 ",
                "<|vision_start|> <|image_pad|> t2"
            ),
            media: vec![
                Qwen25VlRawMediaInput::Image(raw_image(1)),
                Qwen25VlRawMediaInput::Video(raw_video_metadata(&timestamps)),
            ],
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            video_sources: &mut sources,
        },
    );
    assert!(matches!(
        &result,
        Err(Qwen25VlRawRequestAssemblyError::MediaOrderMismatch {
            marker_index: 0,
            expected: MropeVisualKind::Image,
            actual: MropeVisualKind::Video,
        })
    ));
    assert_eq!(sources[0].calls, [0, 1]);

    let mut published = None;
    if let Ok(value) = result {
        published = Some(value);
    }
    assert!(published.is_none());
}

#[test]
fn qwen25_vl_raw_request_reuses_preflight_snapshot_if_decoder_mutates_checkpoint_files() {
    let dir = raw_request_fixture("raw_request_snapshot_toctou", RAW_IMAGE_ID);
    let timestamps = [0.0, 0.5];
    struct CheckpointMutatingSource {
        inner: RecordingFrameSource,
        dir: PathBuf,
    }
    impl Qwen25VlDecodedFrameSource for CheckpointMutatingSource {
        type Error = FixtureVideoDecodeError;

        fn decode_frame(
            &mut self,
            frame_index: usize,
        ) -> Result<Qwen25VlDecodedImage, Self::Error> {
            fs::write(self.dir.join("tokenizer.json"), "{").unwrap();
            fs::write(self.dir.join("config.json"), r#"{"model_type":"qwen2_vl"}"#).unwrap();
            self.inner.decode_frame(frame_index)
        }
    }
    let mut sources = [CheckpointMutatingSource {
        inner: raw_video_source(&[10, 20]),
        dir: dir.path.clone(),
    }];
    let actual = assemble_qwen2_5_vl_raw_request_mrope_model_input(
        Qwen25VlRawRequestMropeModelInputRequest {
            model_dir: &dir,
            rendered_prompt: "t0 <|vision_start|> <|video_pad|> t1",
            media: vec![Qwen25VlRawMediaInput::Video(raw_video_metadata(
                &timestamps,
            ))],
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            video_sources: &mut sources,
        },
    )
    .unwrap();
    assert_eq!(actual.media_order(), [MropeVisualKind::Video]);
    assert_eq!(actual.prompt_token_ids.len(), actual.mrope_positions.len());
    assert_eq!(sources[0].inner.calls, [0, 1]);
    assert_eq!(fs::read_to_string(dir.join("tokenizer.json")).unwrap(), "{");
}

#[test]
fn qwen25_vl_raw_request_rejects_qwen2_vl_before_decode() {
    let dir = raw_request_fixture("raw_request_qwen2_vl", RAW_IMAGE_ID);
    let config = fs::read_to_string(dir.join("config.json"))
        .unwrap()
        .replace("qwen2_5_vl", "qwen2_vl");
    fs::write(dir.join("config.json"), config).unwrap();
    let timestamps = [0.0, 0.5];
    let mut sources = [raw_video_source(&[1, 2])];
    assert!(matches!(
        assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0 <|vision_start|> <|video_pad|> t1",
                media: vec![Qwen25VlRawMediaInput::Video(raw_video_metadata(
                    &timestamps,
                ))],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut sources,
            }
        ),
        Err(Qwen25VlRawRequestAssemblyError::UnsupportedModelType {
            model_type: QwenVlModelType::Qwen2Vl
        })
    ));
    assert!(sources[0].calls.is_empty());
}

#[test]
fn qwen25_vl_raw_request_marker_compare_rejects_unequal_lengths() {
    let err = validate_qwen2_5_vl_raw_media_order::<FixtureVideoDecodeError>(
        &[MropeVisualKind::Image, MropeVisualKind::Video],
        &[
            MropeSegment::Text(2),
            MropeSegment::Image(MropeGrid::new(1, 1, 1)),
        ],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Qwen25VlRawRequestAssemblyError::MediaCountMismatch {
            raw_count: 2,
            marker_count: 1
        }
    ));
}
