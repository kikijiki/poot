//! Filesystem metadata and tokenizer adapter for Qwen-VL mRoPE model-input assembly.
//!
//! Reads on-disk metadata, tokenizes the rendered prompt and feeds both to the pure mRoPE source
//! boundaries. It does not render chat templates, construct grids, or run vision.
//!
//! [`Qwen25Vl`](super::qwen_vl::Qwen25Vl) feeds the positions and the visual placeholders this assembles to the
//! driver's steps.

use poot_tensor::DType;
use std::path::{Path, PathBuf};

use poot_load::qwen_vl::{
    QwenVlHfMetadata, QwenVlModelType, QwenVlProcessorMetadataError,
    QwenVlProcessorSpecialTokenStrings, QwenVlSpecialTokenRole,
};
use poot_models::mrope::{
    MropeConfigMetadata, MropeConfigMetadataError, MropeGrid, MropePositionAssemblyError,
    MropePositionIds, MropeProcessorFpsInput, MropeProcessorSpecialToken, MropeSpecialTokenIds,
    MropeSpecialTokenKind, MropeSpecialTokenReconcileError, MropeTemporalPatchSize,
    MropeTokenizerSpecialToken, Qwen25VlMropePositionInput, assemble_qwen2_5_vl_mrope_positions,
    reconcile_mrope_special_token_ids,
};
use poot_tensor::HostTensor;
use tokenizers::Tokenizer;

/// What a checkpoint directory declares before its markers are reconciled: the model type, the mRoPE
/// config, the processor's three marker strings and the tokenizer's id for each, where the vocabulary
/// has one.
#[derive(Clone, Debug)]
pub struct QwenVlMropeSourceMetadata {
    /// The model type `config.json` declares.
    pub model_type: QwenVlModelType,
    /// The mRoPE configuration `config.json` declares: section widths, marker token ids and, when present,
    /// temporal tokens per second.
    pub config: MropeConfigMetadata,
    processor_tokens: QwenVlProcessorSpecialTokenStrings,
    tokenizer_ids: [Option<u32>; 3],
}

impl QwenVlMropeSourceMetadata {
    /// The processor's three marker strings (vision start, image pad, video pad) tagged with their kind, for pad expansion and marker discovery.
    pub fn processor_special_tokens(&self) -> [MropeProcessorSpecialToken<'_>; 3] {
        [
            MropeProcessorSpecialToken::new(
                MropeSpecialTokenKind::VisionStart,
                self.processor_tokens
                    .token(QwenVlSpecialTokenRole::VisionStart),
            ),
            MropeProcessorSpecialToken::new(
                MropeSpecialTokenKind::Image,
                self.processor_tokens
                    .token(QwenVlSpecialTokenRole::ImagePad),
            ),
            MropeProcessorSpecialToken::new(
                MropeSpecialTokenKind::Video,
                self.processor_tokens
                    .token(QwenVlSpecialTokenRole::VideoPad),
            ),
        ]
    }

    /// The markers the tokenizer actually has, as `(string, id)` pairs; a marker the vocabulary lacks is left out.
    pub fn tokenizer_special_tokens(&self) -> Vec<MropeTokenizerSpecialToken<'_>> {
        let roles = [
            QwenVlSpecialTokenRole::VisionStart,
            QwenVlSpecialTokenRole::ImagePad,
            QwenVlSpecialTokenRole::VideoPad,
        ];
        roles
            .into_iter()
            .zip(self.tokenizer_ids)
            .filter_map(|(role, id)| {
                id.map(|id| MropeTokenizerSpecialToken::new(self.processor_tokens.token(role), id))
            })
            .collect()
    }
}

/// The checkpoint facts mRoPE assembly needs, with the marker token ids already reconciled between
/// `config.json` and the tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenVlMropeMetadata {
    /// The model type `config.json` declares.
    pub model_type: QwenVlModelType,
    /// The checkpoint's mRoPE configuration.
    pub config: MropeConfigMetadata,
    /// The vision-start, image and video token ids, reconciled between the config and the tokenizer.
    pub special_tokens: MropeSpecialTokenIds,
}

/// One reconciled filesystem snapshot retained through rendered-prompt assembly.
///
/// The tokenizer is kept beside the metadata so a caller-owned media source cannot change checkpoint
/// files between preflight and prompt tokenization.
pub(crate) struct QwenVlMropeMetadataSnapshot {
    metadata: QwenVlMropeMetadata,
    tokenizer_path: PathBuf,
    tokenizer: Tokenizer,
    image_pad: String,
    video_pad: String,
}

impl QwenVlMropeMetadataSnapshot {
    pub(crate) fn metadata(&self) -> &QwenVlMropeMetadata {
        &self.metadata
    }

    pub(crate) fn image_pad(&self) -> &str {
        &self.image_pad
    }

    pub(crate) fn video_pad(&self) -> &str {
        &self.video_pad
    }
}

/// Why a checkpoint's mRoPE and processor metadata cannot be loaded.
#[derive(Debug, thiserror::Error)]
pub enum QwenVlMropeMetadataLoadError {
    /// `config.json` has no usable mRoPE configuration.
    #[error(transparent)]
    Config(#[from] MropeConfigMetadataError),
    /// The processor metadata (marker strings) could not be read.
    #[error(transparent)]
    Processor(#[from] QwenVlProcessorMetadataError),
    /// The tokenizer file could not be loaded.
    #[error("load tokenizer {path:?}: {message}")]
    Tokenizer {
        /// The `tokenizer.json` that failed.
        path: PathBuf,
        /// The tokenizer library's error, rendered.
        message: String,
    },
    /// The config's special token ids and the tokenizer's disagree.
    #[error(transparent)]
    SpecialTokens(#[from] MropeSpecialTokenReconcileError),
}

/// A request's token ids, grids and positions, with the grids borrowed from the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25VlMropeModelInput<'a> {
    /// The prompt's token ids, markers expanded.
    pub prompt_token_ids: &'a [u32],
    /// One patch grid per image, in prompt order.
    pub image_grids: &'a [MropeGrid],
    /// One patch grid per video, in prompt order.
    pub video_grids: &'a [MropeGrid],
    /// The vision tower's spatial merge size the grids are counted with.
    pub spatial_merge_size: usize,
    /// The checkpoint facts the positions were assembled with.
    pub metadata: QwenVlMropeMetadata,
    /// Where every prompt token rotates (`Slot::MropePosition`).
    pub mrope_positions: MropePositionIds,
}

/// Why a model input cannot be assembled from prompt ids and grids.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlMropeModelInputError {
    /// The checkpoint metadata could not be loaded.
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    /// The checkpoint is a Qwen vision-language model of another type than Qwen2.5-VL.
    #[error("Qwen2.5-VL mRoPE model input does not support model type {model_type:?}")]
    UnsupportedModelType {
        /// The model type the checkpoint declares.
        model_type: QwenVlModelType,
    },
    /// The positions could not be assembled from the prompt and grids.
    #[error(transparent)]
    PositionAssembly(#[from] MropePositionAssemblyError),
}

/// What an mRoPE model input is assembled from: a rendered prompt, the grids its media produced and the checkpoint directory.
#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlRenderedPromptMropeModelInputRequest<'dir, 'prompt, 'data> {
    /// The checkpoint directory holding `config.json`, `tokenizer.json` and the processor config.
    pub model_dir: &'dir Path,
    /// The chat-template output with its pads already expanded to the merged-grid counts.
    pub rendered_prompt: &'prompt str,
    /// One patch grid per image, in prompt order.
    pub image_grids: &'data [MropeGrid],
    /// One patch grid per video, in prompt order.
    pub video_grids: &'data [MropeGrid],
    /// The vision tower's spatial merge size the grids are counted with.
    pub spatial_merge_size: usize,
    /// Frames merged into one temporal patch.
    pub temporal_patch_size: MropeTemporalPatchSize,
    /// The per-video processor frame rates; required when `video_grids` is not empty.
    pub fps: Option<MropeProcessorFpsInput<'data>>,
}

/// The tokenized prompt, grids and positions of one request, with the grids borrowed from the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25VlRenderedPromptMropeModelInput<'a> {
    /// The prompt's token ids.
    pub prompt_token_ids: Vec<u32>,
    /// One patch grid per image, in prompt order.
    pub image_grids: &'a [MropeGrid],
    /// One patch grid per video, in prompt order.
    pub video_grids: &'a [MropeGrid],
    /// The vision tower's spatial merge size the grids are counted with.
    pub spatial_merge_size: usize,
    /// The checkpoint facts the positions were assembled with.
    pub metadata: QwenVlMropeMetadata,
    /// Where every prompt token rotates (`Slot::MropePosition`).
    pub mrope_positions: MropePositionIds,
}

/// Borrowed access to the one established Qwen2.5-VL mRoPE model-input shape.
///
/// Owned-token adapters implement this so splice and prefill helpers need no copied prompt or
/// position object.
pub trait Qwen25VlMropeModelInputView {
    /// The prompt's token ids, the visual placeholders included.
    fn prompt_token_ids(&self) -> &[u32];
    /// One patch grid per image, in prompt order.
    fn image_grids(&self) -> &[MropeGrid];
    /// One patch grid per video, in prompt order.
    fn video_grids(&self) -> &[MropeGrid];
    /// The vision tower's spatial merge size the grids are counted with.
    fn spatial_merge_size(&self) -> usize;
    /// The checkpoint facts the positions were assembled with.
    fn metadata(&self) -> &QwenVlMropeMetadata;
    /// Where every prompt token rotates (`Slot::MropePosition`).
    fn mrope_positions(&self) -> &MropePositionIds;
}

macro_rules! impl_qwen25_vl_mrope_model_input_view {
    ($type:ty) => {
        impl Qwen25VlMropeModelInputView for $type {
            fn prompt_token_ids(&self) -> &[u32] {
                &self.prompt_token_ids
            }

            fn image_grids(&self) -> &[MropeGrid] {
                self.image_grids
            }

            fn video_grids(&self) -> &[MropeGrid] {
                self.video_grids
            }

            fn spatial_merge_size(&self) -> usize {
                self.spatial_merge_size
            }

            fn metadata(&self) -> &QwenVlMropeMetadata {
                &self.metadata
            }

            fn mrope_positions(&self) -> &MropePositionIds {
                &self.mrope_positions
            }
        }
    };
}

impl_qwen25_vl_mrope_model_input_view!(Qwen25VlMropeModelInput<'_>);
impl_qwen25_vl_mrope_model_input_view!(Qwen25VlRenderedPromptMropeModelInput<'_>);

/// Why a rendered prompt cannot be tokenized and positioned.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlRenderedPromptMropeModelInputError {
    /// The checkpoint metadata could not be loaded.
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    /// The tokenizer rejected the rendered prompt.
    #[error("tokenize rendered Qwen2.5-VL prompt with {path:?}: {message}")]
    Tokenize {
        /// The `tokenizer.json` that was used.
        path: PathBuf,
        /// The tokenizer's error, rendered.
        message: String,
    },
    /// The token ids and grids do not assemble into a model input.
    #[error(transparent)]
    ModelInput(#[from] Qwen25VlMropeModelInputError),
}

/// Which kind of visual media a processor tensor or grid list belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen25VlProcessorMediaKind {
    /// Still images.
    Image,
    /// Video clips.
    Video,
}

impl std::fmt::Display for Qwen25VlProcessorMediaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image => f.write_str("image"),
            Self::Video => f.write_str("video"),
        }
    }
}

/// Borrowed Qwen2.5-VL processor media output.
///
/// Each tensor is the concatenated rank-2 flattened-patch payload for its media kind; the grids stay
/// separate, in rendered-prompt order.
#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlProcessorMedia<'a> {
    /// The images' patches, `[rows, feature_width]`, concatenated in prompt order; `None` when there is no image.
    pub pixel_values: Option<&'a HostTensor>,
    /// One patch grid per image, in the order of the rows of `pixel_values`.
    pub image_grids: &'a [MropeGrid],
    /// The videos' patches, `[rows, feature_width]`, concatenated in prompt order; `None` when there is no video.
    pub pixel_values_videos: Option<&'a HostTensor>,
    /// One patch grid per video, in the order of the rows of `pixel_values_videos`.
    pub video_grids: &'a [MropeGrid],
}

/// What a model input is assembled from when the media are already processed: a rendered prompt, the processed media and the checkpoint directory.
#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlProcessorMediaMropeModelInputRequest<'dir, 'prompt, 'data> {
    /// The checkpoint directory holding `config.json`, `tokenizer.json` and the processor config.
    pub model_dir: &'dir Path,
    /// The chat-template output with its pads already expanded to the merged-grid counts.
    pub rendered_prompt: &'prompt str,
    /// The processed images and videos.
    pub media: Qwen25VlProcessorMedia<'data>,
    /// The vision tower's spatial merge size the grids are counted with.
    pub spatial_merge_size: usize,
    /// Frames merged into one temporal patch.
    pub temporal_patch_size: MropeTemporalPatchSize,
    /// The per-video processor frame rates; required when `media` holds a video.
    pub fps: Option<MropeProcessorFpsInput<'data>>,
}

/// A tokenized prompt, the processed media it refers to and the positions of its tokens; the media are borrowed from the request.
#[derive(Clone, Debug)]
pub struct Qwen25VlProcessorMediaMropeModelInput<'a> {
    /// The prompt's token ids.
    pub prompt_token_ids: Vec<u32>,
    /// The processed images and videos the prompt's placeholders stand for.
    pub media: Qwen25VlProcessorMedia<'a>,
    /// The vision tower's spatial merge size the grids are counted with.
    pub spatial_merge_size: usize,
    /// The checkpoint facts the positions were assembled with.
    pub metadata: QwenVlMropeMetadata,
    /// Where every prompt token rotates (`Slot::MropePosition`).
    pub mrope_positions: MropePositionIds,
}

impl Qwen25VlMropeModelInputView for Qwen25VlProcessorMediaMropeModelInput<'_> {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    fn image_grids(&self) -> &[MropeGrid] {
        self.media.image_grids
    }

    fn video_grids(&self) -> &[MropeGrid] {
        self.media.video_grids
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }

    fn metadata(&self) -> &QwenVlMropeMetadata {
        &self.metadata
    }

    fn mrope_positions(&self) -> &MropePositionIds {
        &self.mrope_positions
    }
}

/// Why processed media are not a consistent set of pixel tensors and grids.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlProcessorMediaError {
    /// Grids exist for a kind but its pixel tensor is absent.
    #[error("Qwen2.5-VL processor {kind} grids require a pixel tensor")]
    MissingPixels {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
    },
    /// A pixel tensor exists for a kind but has no grid.
    #[error("Qwen2.5-VL processor {kind} pixel tensor has no grids")]
    UnexpectedPixels {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
    },
    /// A pixel tensor is not rank 2.
    #[error("Qwen2.5-VL processor {kind} pixel tensor has shape {actual:?}; expected rank 2")]
    PixelRank {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
        /// The tensor's shape.
        actual: Vec<usize>,
    },
    /// A pixel tensor's feature width is zero.
    #[error("Qwen2.5-VL processor {kind} pixel tensor feature width must be nonzero")]
    ZeroFeatureWidth {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
    },
    /// Summing the grids' patch counts overflows `usize`.
    #[error("Qwen2.5-VL processor {kind} raw grid row count overflows usize")]
    GridRowCountOverflow {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
    },
    /// The pixel tensor's row count differs from the patches its grids describe.
    #[error(
        "Qwen2.5-VL processor {kind} pixel tensor has {actual} rows; raw grids require {expected}"
    )]
    PixelRowCount {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
        /// Rows the grids require.
        expected: usize,
        /// Rows the tensor has.
        actual: usize,
    },
    /// The pixel tensor's dtype is not f32, bf16 or f16.
    #[error(
        "Qwen2.5-VL processor {kind} pixel tensor has dtype {actual}; expected f32, bf16 or f16"
    )]
    PixelDtype {
        /// Whether the offending tensor and grids are the image or the video ones.
        kind: Qwen25VlProcessorMediaKind,
        /// The tensor's dtype.
        actual: DType,
    },
}

/// Why processed media and a rendered prompt do not assemble into a model input.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlProcessorMediaMropeModelInputError {
    /// The prompt could not be tokenized or positioned.
    #[error(transparent)]
    ModelInput(#[from] Qwen25VlRenderedPromptMropeModelInputError),
    /// The processed media are inconsistent.
    #[error(transparent)]
    Media(#[from] Qwen25VlProcessorMediaError),
}

/// Read the checkpoint directory `dir`'s mRoPE configuration, processor marker strings and the
/// tokenizer's ids for them. Fails when `config.json`, the processor config or `tokenizer.json` cannot
/// be read or is invalid; it does not reconcile the config's ids with the tokenizer's.
pub fn load_qwen_vl_mrope_source_metadata(
    dir: impl AsRef<Path>,
) -> Result<QwenVlMropeSourceMetadata, QwenVlMropeMetadataLoadError> {
    let (sources, _, _) = load_qwen_vl_mrope_source_metadata_and_tokenizer(dir.as_ref())?;
    Ok(sources)
}

fn load_qwen_vl_hf_and_config_metadata(
    dir: &Path,
) -> Result<(QwenVlHfMetadata, MropeConfigMetadata), QwenVlMropeMetadataLoadError> {
    let hf_config =
        QwenVlHfMetadata::load(dir.join("config.json")).map_err(MropeConfigMetadataError::from)?;
    let config = MropeConfigMetadata::try_from(hf_config)?;
    Ok((hf_config, config))
}

fn load_qwen_vl_tokenizer(
    tokenizer_path: &Path,
) -> Result<Tokenizer, QwenVlMropeMetadataLoadError> {
    Tokenizer::from_file(tokenizer_path).map_err(|source| QwenVlMropeMetadataLoadError::Tokenizer {
        path: tokenizer_path.to_path_buf(),
        message: source.to_string(),
    })
}

fn qwen_vl_tokenizer_ids(
    tokenizer: &Tokenizer,
    processor_tokens: &QwenVlProcessorSpecialTokenStrings,
) -> [Option<u32>; 3] {
    [
        tokenizer.token_to_id(processor_tokens.token(QwenVlSpecialTokenRole::VisionStart)),
        tokenizer.token_to_id(processor_tokens.token(QwenVlSpecialTokenRole::ImagePad)),
        tokenizer.token_to_id(processor_tokens.token(QwenVlSpecialTokenRole::VideoPad)),
    ]
}

fn load_qwen_vl_mrope_source_metadata_and_tokenizer(
    dir: &Path,
) -> Result<(QwenVlMropeSourceMetadata, PathBuf, Tokenizer), QwenVlMropeMetadataLoadError> {
    let (hf_config, config) = load_qwen_vl_hf_and_config_metadata(dir)?;
    let processor_tokens = QwenVlProcessorSpecialTokenStrings::load_from_dir(dir, &hf_config)?;
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = load_qwen_vl_tokenizer(&tokenizer_path)?;
    let tokenizer_ids = qwen_vl_tokenizer_ids(&tokenizer, &processor_tokens);
    Ok((
        QwenVlMropeSourceMetadata {
            model_type: hf_config.model_type(),
            config,
            processor_tokens,
            tokenizer_ids,
        },
        tokenizer_path,
        tokenizer,
    ))
}

fn reconcile_qwen_vl_mrope_source_metadata(
    sources: QwenVlMropeSourceMetadata,
) -> Result<QwenVlMropeMetadata, QwenVlMropeMetadataLoadError> {
    let processor_tokens = sources.processor_special_tokens();
    let tokenizer_tokens = sources.tokenizer_special_tokens();
    let special_tokens =
        reconcile_mrope_special_token_ids(&processor_tokens, &tokenizer_tokens, &sources.config)?;
    Ok(QwenVlMropeMetadata {
        model_type: sources.model_type,
        config: sources.config,
        special_tokens,
    })
}

pub(crate) fn load_qwen_vl_mrope_metadata_snapshot(
    dir: &Path,
) -> Result<QwenVlMropeMetadataSnapshot, QwenVlMropeMetadataLoadError> {
    let (sources, tokenizer_path, tokenizer) =
        load_qwen_vl_mrope_source_metadata_and_tokenizer(dir)?;
    let processor_tokens = sources.processor_special_tokens();
    let image_pad = processor_tokens
        .iter()
        .find(|token| token.role == MropeSpecialTokenKind::Image)
        .expect("processor special tokens include the image pad")
        .token
        .to_string();
    let video_pad = processor_tokens
        .iter()
        .find(|token| token.role == MropeSpecialTokenKind::Video)
        .expect("processor special tokens include the video pad")
        .token
        .to_string();
    let metadata = reconcile_qwen_vl_mrope_source_metadata(sources)?;
    Ok(QwenVlMropeMetadataSnapshot {
        metadata,
        tokenizer_path,
        tokenizer,
        image_pad,
        video_pad,
    })
}

fn assemble_qwen2_5_vl_mrope_model_input_parts(
    metadata: QwenVlMropeMetadata,
    prompt_token_ids: &[u32],
    image_grids: &[MropeGrid],
    video_grids: &[MropeGrid],
    spatial_merge_size: usize,
    temporal_patch_size: MropeTemporalPatchSize,
    fps: Option<MropeProcessorFpsInput<'_>>,
) -> Result<(QwenVlMropeMetadata, MropePositionIds), Qwen25VlMropeModelInputError> {
    if metadata.model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlMropeModelInputError::UnsupportedModelType {
            model_type: metadata.model_type,
        });
    }
    let mrope_positions = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
        tokens: prompt_token_ids,
        special_tokens: metadata.special_tokens,
        image_grids,
        video_grids,
        spatial_merge_size,
        temporal_patch_size,
        fps,
        config: &metadata.config,
    })?;
    Ok((metadata, mrope_positions))
}

pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt_with_metadata_snapshot<
    'data,
>(
    input: Qwen25VlRenderedPromptMropeModelInputRequest<'_, '_, 'data>,
    snapshot: QwenVlMropeMetadataSnapshot,
) -> Result<Qwen25VlRenderedPromptMropeModelInput<'data>, Qwen25VlRenderedPromptMropeModelInputError>
{
    let QwenVlMropeMetadataSnapshot {
        metadata,
        tokenizer_path,
        tokenizer,
        image_pad: _,
        video_pad: _,
    } = snapshot;
    if metadata.model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
            Qwen25VlMropeModelInputError::UnsupportedModelType {
                model_type: metadata.model_type,
            },
        ));
    }
    let encoded = tokenizer
        .encode(input.rendered_prompt, false)
        .map_err(
            |source| Qwen25VlRenderedPromptMropeModelInputError::Tokenize {
                path: tokenizer_path,
                message: source.to_string(),
            },
        )?;
    let prompt_token_ids = encoded.get_ids().to_vec();
    let (metadata, mrope_positions) = assemble_qwen2_5_vl_mrope_model_input_parts(
        metadata,
        &prompt_token_ids,
        input.image_grids,
        input.video_grids,
        input.spatial_merge_size,
        input.temporal_patch_size,
        input.fps,
    )?;
    Ok(Qwen25VlRenderedPromptMropeModelInput {
        prompt_token_ids,
        image_grids: input.image_grids,
        video_grids: input.video_grids,
        spatial_merge_size: input.spatial_merge_size,
        metadata,
        mrope_positions,
    })
}

impl Qwen25VlProcessorMedia<'_> {
    /// Validate the cached Transformers processor's flattened-patch row contract.
    ///
    /// Prompt placeholder counts are not checked here; the rendered-prompt adapter validates them via
    /// the merged-grid path before the full handoff calls this.
    pub fn validate(&self) -> Result<(), Qwen25VlProcessorMediaError> {
        validate_qwen25_processor_pixels(
            Qwen25VlProcessorMediaKind::Image,
            self.pixel_values,
            self.image_grids,
        )?;
        validate_qwen25_processor_pixels(
            Qwen25VlProcessorMediaKind::Video,
            self.pixel_values_videos,
            self.video_grids,
        )?;
        Ok(())
    }
}

pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_processor_media_with_metadata_snapshot<
    'data,
>(
    input: Qwen25VlProcessorMediaMropeModelInputRequest<'_, '_, 'data>,
    snapshot: QwenVlMropeMetadataSnapshot,
) -> Result<Qwen25VlProcessorMediaMropeModelInput<'data>, Qwen25VlProcessorMediaMropeModelInputError>
{
    let rendered =
        assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt_with_metadata_snapshot(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: input.model_dir,
                rendered_prompt: input.rendered_prompt,
                image_grids: input.media.image_grids,
                video_grids: input.media.video_grids,
                spatial_merge_size: input.spatial_merge_size,
                temporal_patch_size: input.temporal_patch_size,
                fps: input.fps,
            },
            snapshot,
        )?;
    input.media.validate()?;
    Ok(Qwen25VlProcessorMediaMropeModelInput {
        prompt_token_ids: rendered.prompt_token_ids,
        media: input.media,
        spatial_merge_size: rendered.spatial_merge_size,
        metadata: rendered.metadata,
        mrope_positions: rendered.mrope_positions,
    })
}

fn validate_qwen25_processor_pixels(
    kind: Qwen25VlProcessorMediaKind,
    pixels: Option<&HostTensor>,
    grids: &[MropeGrid],
) -> Result<(), Qwen25VlProcessorMediaError> {
    let Some(pixels) = pixels else {
        return if grids.is_empty() {
            Ok(())
        } else {
            Err(Qwen25VlProcessorMediaError::MissingPixels { kind })
        };
    };
    if grids.is_empty() {
        return Err(Qwen25VlProcessorMediaError::UnexpectedPixels { kind });
    }
    if pixels.shape().len() != 2 {
        return Err(Qwen25VlProcessorMediaError::PixelRank {
            kind,
            actual: pixels.shape().to_vec(),
        });
    }
    if pixels.shape()[1] == 0 {
        return Err(Qwen25VlProcessorMediaError::ZeroFeatureWidth { kind });
    }
    let expected_rows = grids.iter().try_fold(0usize, |rows, grid| {
        grid.temporal
            .checked_mul(grid.height)
            .and_then(|count| count.checked_mul(grid.width))
            .and_then(|count| rows.checked_add(count))
    });
    let Some(expected_rows) = expected_rows else {
        return Err(Qwen25VlProcessorMediaError::GridRowCountOverflow { kind });
    };
    if pixels.shape()[0] != expected_rows {
        return Err(Qwen25VlProcessorMediaError::PixelRowCount {
            kind,
            expected: expected_rows,
            actual: pixels.shape()[0],
        });
    }
    if !matches!(pixels.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Qwen25VlProcessorMediaError::PixelDtype {
            kind,
            actual: pixels.dtype(),
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use poot_models::mrope::{MropePositionAssemblyError, MropeSpanError};
    use std::collections::HashMap;

    const VISION_START_ID: u32 = 3;
    const IMAGE_ID: u32 = 4;
    const VIDEO_ID: u32 = 5;
    const VISION_START: &str = "<|vision_start|>";
    const IMAGE_PAD: &str = "<|image_pad|>";
    const VIDEO_PAD: &str = "<|video_pad|>";

    const CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 128000,
        "sliding_window": 32768,
        "use_sliding_window": false,
        "vision_start_token_id": 3,
        "vision_end_token_id": 6,
        "image_token_id": 4,
        "video_token_id": 5,
        "rope_scaling": {
            "type": "mrope",
            "mrope_section": [16, 24, 24]
        },
        "vision_config": {
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "tokens_per_second": 4
        }
    }"#;

    fn tokenizer_config(vision: &str, image: &str, video: &str) -> String {
        format!(
            r#"{{
        "added_tokens_decoder": {{
            "{VISION_START_ID}": {{"content": "{vision}", "special": true}},
            "{IMAGE_ID}": {{"content": "{image}", "special": true}},
            "{VIDEO_ID}": {{"content": "{video}", "special": true}}
        }}
    }}"#
        )
    }

    struct FixtureDir {
        path: PathBuf,
    }

    impl AsRef<Path> for FixtureDir {
        fn as_ref(&self) -> &Path {
            &self.path
        }
    }

    impl std::ops::Deref for FixtureDir {
        type Target = Path;

        fn deref(&self) -> &Self::Target {
            &self.path
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn fixture_dir(name: &str) -> FixtureDir {
        let path =
            std::env::temp_dir().join(format!("poot_qwen_vl_mrope_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        FixtureDir { path }
    }

    fn write_tokenizer(dir: &Path, image_id: u32, include_image: bool) {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

        let mut vocab = HashMap::from([
            ("t0".to_string(), 0),
            ("t1".to_string(), 1),
            ("t2".to_string(), 2),
            (VISION_START.to_string(), VISION_START_ID),
            (VIDEO_PAD.to_string(), VIDEO_ID),
        ]);
        if include_image {
            vocab.insert(IMAGE_PAD.to_string(), image_id);
        }
        for id in 0..=image_id.max(VIDEO_ID) {
            if !vocab.values().any(|&existing| existing == id) {
                vocab.insert(format!("fill{id}"), id);
            }
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("t0".to_string())
            .build()
            .expect("build wordlevel tokenizer fixture");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer
            .save(dir.join("tokenizer.json"), false)
            .expect("save tokenizer fixture");
    }

    fn write_tokenizer_with_missing_unk(dir: &Path) {
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "t0": 0,
                    "t1": 1,
                    "t2": 2,
                    VISION_START: VISION_START_ID,
                    IMAGE_PAD: IMAGE_ID,
                    VIDEO_PAD: VIDEO_ID
                },
                "unk_token": "<missing>"
            }
        });
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_vec_pretty(&tokenizer).unwrap(),
        )
        .unwrap();
    }

    fn write_fixture_with_config(
        name: &str,
        config_json: &str,
        image_id: u32,
        include_image: bool,
    ) -> FixtureDir {
        let dir = fixture_dir(name);
        std::fs::write(dir.join("config.json"), config_json).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            tokenizer_config(VISION_START, IMAGE_PAD, VIDEO_PAD),
        )
        .unwrap();
        write_tokenizer(&dir, image_id, include_image);
        dir
    }

    fn write_fixture(name: &str, image_id: u32, include_image: bool) -> FixtureDir {
        write_fixture_with_config(name, CONFIG_JSON, image_id, include_image)
    }

    fn qwen2_vl_config_json() -> String {
        CONFIG_JSON
            .replace(
                "\"architectures\": [\"Qwen2_5_VLForConditionalGeneration\"]",
                "\"architectures\": [\"Qwen2VLForConditionalGeneration\"]",
            )
            .replace(
                "\"model_type\": \"qwen2_5_vl\"",
                "\"model_type\": \"qwen2_vl\"",
            )
            .replace(",\n            \"tokens_per_second\": 4", "")
    }

    /// What a prompt of `tokens` makes of the model directory's metadata and the grids: the pure
    /// position assembly over the snapshot a request would hold.
    struct ExpectedModelInput {
        metadata: QwenVlMropeMetadata,
        mrope_positions: MropePositionIds,
    }

    fn expected_model_input(
        dir: &Path,
        tokens: &[u32],
        image_grids: &[MropeGrid],
        video_grids: &[MropeGrid],
        temporal_patch_size: MropeTemporalPatchSize,
    ) -> ExpectedModelInput {
        let snapshot = load_qwen_vl_mrope_metadata_snapshot(dir).unwrap();
        let (metadata, mrope_positions) = assemble_qwen2_5_vl_mrope_model_input_parts(
            snapshot.metadata,
            tokens,
            image_grids,
            video_grids,
            2,
            temporal_patch_size,
            None,
        )
        .unwrap();
        ExpectedModelInput {
            metadata,
            mrope_positions,
        }
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_tokenizes_plain_text_and_owns_ids() {
        let dir = write_fixture("rendered_prompt_plain_text", IMAGE_ID, true);
        let expected_tokens = [0, 1, 2];
        let expected = expected_model_input(
            &dir,
            &expected_tokens,
            &[],
            &[],
            MropeTemporalPatchSize::new(2).unwrap(),
        );

        let actual = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "t0 t1 t2",
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap();

        assert_eq!(actual.prompt_token_ids, expected_tokens);
        assert_eq!(actual.image_grids, &[]);
        assert_eq!(actual.video_grids, &[]);
        assert_eq!(actual.spatial_merge_size, 2);
        assert_eq!(actual.metadata, expected.metadata);
        assert_eq!(actual.mrope_positions, expected.mrope_positions);
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_preserves_empty_and_unknown_tokenizer_behavior() {
        let dir = write_fixture("rendered_prompt_empty_unknown", IMAGE_ID, true);
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let empty = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "",
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: None,
            },
        )
        .unwrap();
        assert!(empty.prompt_token_ids.is_empty());
        assert!(empty.mrope_positions.is_empty());

        let expected_tokens = [0, 1];
        let expected = expected_model_input(&dir, &expected_tokens, &[], &[], temporal_patch_size);
        let unknown = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "not_vocab t1",
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: None,
            },
        )
        .unwrap();
        assert_eq!(unknown.prompt_token_ids, expected_tokens);
        assert_eq!(unknown.mrope_positions, expected.mrope_positions);
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_preserves_placeholder_order_and_delegates() {
        let dir = write_fixture("rendered_prompt_placeholders", IMAGE_ID, true);
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(1, 2, 2), MropeGrid::new(2, 2, 2)];
        let expected_tokens = [
            0,
            VISION_START_ID,
            VIDEO_ID,
            1,
            VISION_START_ID,
            IMAGE_ID,
            2,
            VISION_START_ID,
            VIDEO_ID,
            VIDEO_ID,
            0,
        ];
        let expected = expected_model_input(
            &dir,
            &expected_tokens,
            &image_grids,
            &video_grids,
            MropeTemporalPatchSize::new(2).unwrap(),
        );
        let rendered = format!(
            "t0 {VISION_START} {VIDEO_PAD} t1 {VISION_START} {IMAGE_PAD} t2 \
             {VISION_START} {VIDEO_PAD} {VIDEO_PAD} t0"
        );

        let actual = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &rendered,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap();

        assert_eq!(actual.prompt_token_ids, expected_tokens);
        assert!(std::ptr::eq(actual.image_grids, image_grids.as_slice()));
        assert!(std::ptr::eq(actual.video_grids, video_grids.as_slice()));
        assert_eq!(actual.metadata, expected.metadata);
        assert_eq!(actual.mrope_positions, expected.mrope_positions);
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_outlives_prompt_and_model_dir_borrows() {
        fn assemble_with_ephemeral_prompt_and_dir<'data>(
            image_grids: &'data [MropeGrid],
            video_grids: &'data [MropeGrid],
        ) -> Qwen25VlRenderedPromptMropeModelInput<'data> {
            let dir = write_fixture("rendered_prompt_short_prompt_lifetime", IMAGE_ID, true);
            let prompt = format!("t0 {VISION_START} {IMAGE_PAD} t1");
            assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
                Qwen25VlRenderedPromptMropeModelInputRequest {
                    model_dir: &dir,
                    rendered_prompt: &prompt,
                    image_grids,
                    video_grids,
                    spatial_merge_size: 2,
                    temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                    fps: None,
                },
            )
            .unwrap()
        }

        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [];
        let actual = assemble_with_ephemeral_prompt_and_dir(&image_grids, &video_grids);

        assert_eq!(actual.prompt_token_ids, [0, VISION_START_ID, IMAGE_ID, 1]);
        assert!(std::ptr::eq(actual.image_grids, image_grids.as_slice()));
        assert_eq!(actual.video_grids, &[]);
        assert_eq!(actual.mrope_positions.len(), actual.prompt_token_ids.len());
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_reports_typed_failures() {
        let missing_tokenizer = write_fixture("rendered_prompt_missing_tokenizer", IMAGE_ID, true);
        std::fs::remove_file(missing_tokenizer.join("tokenizer.json")).unwrap();
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &missing_tokenizer,
                rendered_prompt: "not_vocab",
                image_grids: &[MropeGrid::new(0, 1, 1)],
                video_grids: &[],
                spatial_merge_size: 0,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlRenderedPromptMropeModelInputError::Metadata(
                QwenVlMropeMetadataLoadError::Tokenizer { ref path, .. }
            ) if path == &missing_tokenizer.join("tokenizer.json")
        ));

        let malformed_tokenizer =
            write_fixture("rendered_prompt_malformed_tokenizer", IMAGE_ID, true);
        std::fs::write(malformed_tokenizer.join("tokenizer.json"), "{").unwrap();
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &malformed_tokenizer,
                rendered_prompt: "t0",
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlRenderedPromptMropeModelInputError::Metadata(
                QwenVlMropeMetadataLoadError::Tokenizer { ref path, .. }
            ) if path == &malformed_tokenizer.join("tokenizer.json")
        ));

        let encode_error = write_fixture("rendered_prompt_encode_error", IMAGE_ID, true);
        write_tokenizer_with_missing_unk(&encode_error);
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &encode_error,
                rendered_prompt: "not_vocab",
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlRenderedPromptMropeModelInputError::Tokenize { ref path, .. }
                if path == &encode_error.join("tokenizer.json")
        ));
    }

    #[test]
    fn qwen25_vl_owned_processed_media_uses_rendered_tokens_for_mixed_span_order() {
        use crate::multimodal::qwen_vl_image::{
            Qwen25VlImageProcessorConfig, Qwen25VlProcessedImages, Qwen25VlProcessedVideo,
            assemble_qwen2_5_vl_processed_media,
        };
        use poot_models::mrope::{
            MropeSampledFramesPerSecond, MropeSecondsPerGridStep, MropeTemporalPatchSize,
        };

        let dir = write_fixture("owned_processed_media_rendered_order", IMAGE_ID, true);
        let config = Qwen25VlImageProcessorConfig::default();
        let feature_width =
            3 * config.temporal_patch_size() * config.patch_size() * config.patch_size();
        let images = Qwen25VlProcessedImages {
            pixel_values: HostTensor::f32(vec![4, feature_width], fill(4 * feature_width, 21)),
            grids: vec![MropeGrid::new(1, 2, 2)],
        };
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let video = |temporal: usize, fps: f64, seed| {
            let sampled_frames_per_second = MropeSampledFramesPerSecond::new(fps).unwrap();
            Qwen25VlProcessedVideo {
                pixel_values_videos: HostTensor::f32(
                    vec![temporal * 4, feature_width],
                    fill(temporal * 4 * feature_width, seed),
                ),
                grid: MropeGrid::new(temporal, 2, 2),
                sampled_frames_per_second,
                seconds_per_grid_step: MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
                    temporal_patch_size,
                    sampled_frames_per_second,
                )
                .unwrap(),
                temporal_patch_size,
                sampled_frame_count: temporal * 2,
                padded_frame_count: temporal * 2,
            }
        };
        let owned = assemble_qwen2_5_vl_processed_media(
            Some(images),
            vec![video(1, 2.0, 22), video(2, 4.0, 23)],
            &config,
        )
        .unwrap();
        let rendered = format!(
            "t0 {VISION_START} {VIDEO_PAD} t1 {VISION_START} {IMAGE_PAD} t2 {VISION_START} {VIDEO_PAD} {VIDEO_PAD}"
        );
        let actual = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
            Qwen25VlProcessorMediaMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &rendered,
                media: owned.processor_media(),
                spatial_merge_size: config.merge_size(),
                temporal_patch_size: owned.temporal_patch_size(),
                fps: Some(owned.processor_fps_input()),
            },
        )
        .unwrap();

        assert_eq!(
            actual.prompt_token_ids,
            [
                0,
                VISION_START_ID,
                VIDEO_ID,
                1,
                VISION_START_ID,
                IMAGE_ID,
                2,
                VISION_START_ID,
                VIDEO_ID,
                VIDEO_ID
            ]
        );
        let [temporal, height, width] = actual.mrope_positions.axes();
        assert_eq!(temporal, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]);
        assert_eq!(height, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 8]);
        assert_eq!(width, height);
        assert!(std::ptr::eq(
            actual.media.pixel_values.unwrap(),
            owned.pixel_values.as_ref().unwrap(),
        ));
        assert!(std::ptr::eq(
            actual.media.pixel_values_videos.unwrap(),
            owned.pixel_values_videos.as_ref().unwrap(),
        ));
        assert_eq!(actual.media.validate(), Ok(()));
    }

    #[test]
    fn qwen25_vl_processor_media_validation_reports_typed_contract_errors() {
        let image_grid = [MropeGrid::new(1, 2, 2)];
        let valid = HostTensor::f32(vec![4, 3], vec![0.0; 12]);
        let no_video_grids = [];

        let error = |pixel_values, image_grids| {
            Qwen25VlProcessorMedia {
                pixel_values,
                image_grids,
                pixel_values_videos: None,
                video_grids: &no_video_grids,
            }
            .validate()
            .unwrap_err()
        };
        assert_eq!(
            error(None, &image_grid),
            Qwen25VlProcessorMediaError::MissingPixels {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );
        assert_eq!(
            error(Some(&valid), &[]),
            Qwen25VlProcessorMediaError::UnexpectedPixels {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );

        let wrong_rank = HostTensor::f32(vec![4, 1, 3], vec![0.0; 12]);
        assert!(matches!(
            error(Some(&wrong_rank), &image_grid),
            Qwen25VlProcessorMediaError::PixelRank { .. }
        ));
        let zero_width = HostTensor::f32(vec![4, 0], vec![]);
        assert_eq!(
            error(Some(&zero_width), &image_grid),
            Qwen25VlProcessorMediaError::ZeroFeatureWidth {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );
        let wrong_rows = HostTensor::f32(vec![3, 3], vec![0.0; 9]);
        assert_eq!(
            error(Some(&wrong_rows), &image_grid),
            Qwen25VlProcessorMediaError::PixelRowCount {
                kind: Qwen25VlProcessorMediaKind::Image,
                expected: 4,
                actual: 3,
            }
        );
        let integer = HostTensor::i32(vec![4, 3], vec![0; 12]);
        assert_eq!(
            error(Some(&integer), &image_grid),
            Qwen25VlProcessorMediaError::PixelDtype {
                kind: Qwen25VlProcessorMediaKind::Image,
                actual: DType::I32,
            }
        );
        let bytes = HostTensor::from_le_bytes(DType::I8, vec![4, 3], &[0; 12]).unwrap();
        assert_eq!(
            error(Some(&bytes), &image_grid),
            Qwen25VlProcessorMediaError::PixelDtype {
                kind: Qwen25VlProcessorMediaKind::Image,
                actual: DType::I8,
            }
        );
        let overflow_grid = [MropeGrid::new(usize::MAX, 2, 2)];
        assert_eq!(
            error(Some(&valid), &overflow_grid),
            Qwen25VlProcessorMediaError::GridRowCountOverflow {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );
        let sum_overflow_grids = [MropeGrid::new(usize::MAX, 1, 1), MropeGrid::new(1, 1, 1)];
        assert_eq!(
            error(Some(&valid), &sum_overflow_grids),
            Qwen25VlProcessorMediaError::GridRowCountOverflow {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );
    }

    #[test]
    fn qwen25_vl_processor_media_handoff_preserves_rendered_error_precedence() {
        let qwen2_config = qwen2_vl_config_json();
        let dir = write_fixture_with_config(
            "processor_media_error_precedence",
            &qwen2_config,
            IMAGE_ID,
            true,
        );
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let err = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
            Qwen25VlProcessorMediaMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: "not_vocab",
                media: Qwen25VlProcessorMedia {
                    pixel_values: None,
                    image_grids: &image_grids,
                    pixel_values_videos: None,
                    video_grids: &[],
                },
                spatial_merge_size: 0,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlProcessorMediaMropeModelInputError::ModelInput(
                Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                    Qwen25VlMropeModelInputError::UnsupportedModelType {
                        model_type: QwenVlModelType::Qwen2Vl
                    }
                )
            )
        ));

        let dir = write_fixture("processor_media_span_error_precedence", IMAGE_ID, true);
        let rendered = format!("t0 {VISION_START} {IMAGE_PAD} t1");
        let image_grids = [MropeGrid::new(1, 2, 4)];
        let err = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
            Qwen25VlProcessorMediaMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &rendered,
                media: Qwen25VlProcessorMedia {
                    pixel_values: None,
                    image_grids: &image_grids,
                    pixel_values_videos: None,
                    video_grids: &[],
                },
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlProcessorMediaMropeModelInputError::ModelInput(
                Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                    Qwen25VlMropeModelInputError::PositionAssembly(
                        MropePositionAssemblyError::Span(
                            MropeSpanError::PlaceholderCountMismatch {
                                start_index: 1,
                                expected: 2,
                                actual: 1,
                                ..
                            }
                        )
                    )
                )
            )
        ));
    }

    #[test]
    fn qwen25_vl_processor_media_validation_has_exact_precedence_and_raw_grid_sum() {
        let image_grids = [MropeGrid::new(1, 2, 2), MropeGrid::new(2, 2, 4)];
        let video_grids = [MropeGrid::new(1, 2, 2)];
        let image_pixels = HostTensor::f32(vec![20, 3], vec![0.0; 60]);
        let video_pixels = HostTensor::f32(vec![4, 3], vec![0.0; 12]);

        for pixels in [
            image_pixels.clone(),
            HostTensor::bf16(vec![20, 3], vec![0; 60]),
            HostTensor::f16(vec![20, 3], vec![0; 60]),
        ] {
            Qwen25VlProcessorMedia {
                pixel_values: Some(&pixels),
                image_grids: &image_grids,
                pixel_values_videos: Some(&video_pixels),
                video_grids: &video_grids,
            }
            .validate()
            .unwrap();
        }

        let missing_both = Qwen25VlProcessorMedia {
            pixel_values: None,
            image_grids: &image_grids,
            pixel_values_videos: None,
            video_grids: &video_grids,
        };
        assert_eq!(
            missing_both.validate().unwrap_err(),
            Qwen25VlProcessorMediaError::MissingPixels {
                kind: Qwen25VlProcessorMediaKind::Image,
            }
        );

        let missing_video = Qwen25VlProcessorMedia {
            pixel_values: Some(&image_pixels),
            image_grids: &image_grids,
            pixel_values_videos: None,
            video_grids: &video_grids,
        };
        assert_eq!(
            missing_video.validate().unwrap_err(),
            Qwen25VlProcessorMediaError::MissingPixels {
                kind: Qwen25VlProcessorMediaKind::Video,
            }
        );

        let wrong_rank_integer = HostTensor::i32(vec![20, 1, 3], vec![0; 60]);
        let wrong_rank = Qwen25VlProcessorMedia {
            pixel_values: Some(&wrong_rank_integer),
            image_grids: &image_grids,
            pixel_values_videos: Some(&video_pixels),
            video_grids: &video_grids,
        };
        assert!(matches!(
            wrong_rank.validate(),
            Err(Qwen25VlProcessorMediaError::PixelRank {
                kind: Qwen25VlProcessorMediaKind::Image,
                ..
            })
        ));

        let wrong_rows_integer = HostTensor::i32(vec![19, 3], vec![0; 57]);
        let wrong_rows = Qwen25VlProcessorMedia {
            pixel_values: Some(&wrong_rows_integer),
            image_grids: &image_grids,
            pixel_values_videos: Some(&video_pixels),
            video_grids: &video_grids,
        };
        assert_eq!(
            wrong_rows.validate().unwrap_err(),
            Qwen25VlProcessorMediaError::PixelRowCount {
                kind: Qwen25VlProcessorMediaKind::Image,
                expected: 20,
                actual: 19,
            }
        );

        let integer = HostTensor::i32(vec![20, 3], vec![0; 60]);
        let integer_pixels = Qwen25VlProcessorMedia {
            pixel_values: Some(&integer),
            image_grids: &image_grids,
            pixel_values_videos: Some(&video_pixels),
            video_grids: &video_grids,
        };
        assert_eq!(
            integer_pixels.validate().unwrap_err(),
            Qwen25VlProcessorMediaError::PixelDtype {
                kind: Qwen25VlProcessorMediaKind::Image,
                actual: DType::I32,
            }
        );
    }

    #[test]
    fn qwen25_vl_processor_media_handoff_outlives_prompt_and_model_dir_borrows() {
        fn assemble_with_ephemeral_prompt_and_dir<'data>(
            image_pixels: &'data HostTensor,
            image_grids: &'data [MropeGrid],
            video_grids: &'data [MropeGrid],
        ) -> Qwen25VlProcessorMediaMropeModelInput<'data> {
            let dir = write_fixture("processor_media_short_prompt_lifetime", IMAGE_ID, true);
            let prompt = format!("t0 {VISION_START} {IMAGE_PAD} t1");
            assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
                Qwen25VlProcessorMediaMropeModelInputRequest {
                    model_dir: &dir,
                    rendered_prompt: &prompt,
                    media: Qwen25VlProcessorMedia {
                        pixel_values: Some(image_pixels),
                        image_grids,
                        pixel_values_videos: None,
                        video_grids,
                    },
                    spatial_merge_size: 2,
                    temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                    fps: None,
                },
            )
            .unwrap()
        }

        let image_pixels = HostTensor::f32(vec![4, 3], vec![0.0; 12]);
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [];
        let actual =
            assemble_with_ephemeral_prompt_and_dir(&image_pixels, &image_grids, &video_grids);

        assert_eq!(actual.prompt_token_ids(), [0, VISION_START_ID, IMAGE_ID, 1]);
        assert!(std::ptr::eq(
            actual.media.pixel_values.unwrap(),
            &image_pixels
        ));
        assert!(std::ptr::eq(actual.image_grids(), image_grids.as_slice()));
        assert!(std::ptr::eq(actual.video_grids(), video_grids.as_slice()));
        assert_eq!(
            actual.mrope_positions().len(),
            actual.prompt_token_ids().len()
        );
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn qwen_vl_filesystem_adapter_returns_typed_processor_json_error() {
        let dir = write_fixture("malformed_tokenizer_config", IMAGE_ID, true);
        std::fs::write(dir.join("tokenizer_config.json"), "{").unwrap();
        let err = load_qwen_vl_mrope_source_metadata(&dir).unwrap_err();
        assert!(matches!(
            err,
            QwenVlMropeMetadataLoadError::Processor(
                QwenVlProcessorMetadataError::ParseTokenizerConfig { ref path, .. }
            ) if path == &dir.join("tokenizer_config.json")
        ));
    }

    #[test]
    fn qwen_vl_filesystem_adapter_reports_config_parse_path() {
        let dir = write_fixture("malformed_config", IMAGE_ID, true);
        std::fs::write(dir.join("config.json"), "{").unwrap();
        let err = load_qwen_vl_mrope_source_metadata(&dir).unwrap_err();
        assert!(matches!(
            err,
            QwenVlMropeMetadataLoadError::Config(MropeConfigMetadataError::Load(
                poot_load::qwen_vl::QwenVlHfMetadataError::ParseConfig { ref path, .. }
            )) if path == &dir.join("config.json")
        ));
    }

    #[test]
    fn qwen_vl_filesystem_adapter_reports_tokenizer_path() {
        let dir = write_fixture("missing_tokenizer", IMAGE_ID, true);
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        let err = load_qwen_vl_mrope_source_metadata(&dir).unwrap_err();
        assert!(matches!(
            err,
            QwenVlMropeMetadataLoadError::Tokenizer { ref path, .. }
                if path == &dir.join("tokenizer.json")
        ));
    }

    /// The rendered-prompt model input of `input` over a snapshot of its model directory loaded here:
    /// what a request does in two steps (the raw-request assembler holds the snapshot across media
    /// processing), as one call for tests.
    pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt<'data>(
        input: Qwen25VlRenderedPromptMropeModelInputRequest<'_, '_, 'data>,
    ) -> Result<
        Qwen25VlRenderedPromptMropeModelInput<'data>,
        Qwen25VlRenderedPromptMropeModelInputError,
    > {
        let snapshot = load_qwen_vl_mrope_metadata_snapshot(input.model_dir)?;
        assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt_with_metadata_snapshot(
            input, snapshot,
        )
    }

    /// [`assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt`] for processor media.
    pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_processor_media<'data>(
        input: Qwen25VlProcessorMediaMropeModelInputRequest<'_, '_, 'data>,
    ) -> Result<
        Qwen25VlProcessorMediaMropeModelInput<'data>,
        Qwen25VlProcessorMediaMropeModelInputError,
    > {
        let snapshot = load_qwen_vl_mrope_metadata_snapshot(input.model_dir)
            .map_err(Qwen25VlRenderedPromptMropeModelInputError::from)?;
        assemble_qwen2_5_vl_mrope_model_input_from_processor_media_with_metadata_snapshot(
            input, snapshot,
        )
    }
}
