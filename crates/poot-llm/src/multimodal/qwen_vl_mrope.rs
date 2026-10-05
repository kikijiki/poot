//! Filesystem metadata adapter for Qwen-VL mRoPE model-input assembly.
//!
//! Reads on-disk metadata and feeds it to the pure mRoPE source boundaries. It does not tokenize raw
//! text, render chat templates, construct grids, or run vision.
//!
//! Held for Card 566b (the VLM front end on the driver): the `pub(crate)` entry points below assemble the
//! mRoPE positions and the visual-embedding binds that card feeds to `Driver::step`, and none has a
//! production caller yet. Where rustc reports one as dead it carries
//! `#[expect(dead_code, reason = "held for POOT-739 (formerly 566b)")]`.

use poot_tensor::DType;
use std::path::{Path, PathBuf};

use super::mrope::{MropeBindError, bind_mrope_prefill_positions};
use crate::architectures::qwen2_hf_config::qwen2_config_from_hf;
use crate::core::graphs::prefill_causal_mask;
use poot_eval::Value;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::{Graph, Slot, Storage, TensorType, ValueId};
use poot_load::{
    Qwen2HfConfig,
    qwen_vl::{
        QwenVlHfConfig, QwenVlHfMetadata, QwenVlModelType, QwenVlProcessorMetadataError,
        QwenVlProcessorSpecialTokenStrings, QwenVlSpecialTokenRole,
    },
};
use poot_models::mrope::{
    MropeConfigMetadata, MropeConfigMetadataError, MropeGrid, MropePositionAssemblyError,
    MropePositionIds, MropePositionSourceAssemblyError, MropeProcessorFpsInput,
    MropeProcessorSpecialToken, MropeSpanError, MropeSpecialTokenIds, MropeSpecialTokenKind,
    MropeSpecialTokenReconcileError, MropeTemporalPatchSize, MropeTokenizerSpecialToken,
    Qwen25VlMropePositionInput, Qwen25VlMropePositionSourceInput,
    assemble_qwen2_5_vl_mrope_positions, assemble_qwen2_5_vl_mrope_positions_from_sources,
    discover_mrope_segments, reconcile_mrope_special_token_ids,
};
use poot_models::qwen2::Qwen2Config;
use poot_tensor::HostTensor;
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct QwenVlMropeSourceMetadata {
    pub model_type: QwenVlModelType,
    pub config: MropeConfigMetadata,
    processor_tokens: QwenVlProcessorSpecialTokenStrings,
    tokenizer_ids: [Option<u32>; 3],
}

impl QwenVlMropeSourceMetadata {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenVlMropeMetadata {
    pub model_type: QwenVlModelType,
    pub config: MropeConfigMetadata,
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

#[derive(Debug, thiserror::Error)]
pub enum QwenVlMropeMetadataLoadError {
    #[error(transparent)]
    Config(#[from] MropeConfigMetadataError),
    #[error(transparent)]
    Processor(#[from] QwenVlProcessorMetadataError),
    #[error("load tokenizer {path:?}: {message}")]
    Tokenizer { path: PathBuf, message: String },
    #[error(transparent)]
    SpecialTokens(#[from] MropeSpecialTokenReconcileError),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen25VlMropeFilesystemPositionInput<'a> {
    pub model_dir: &'a Path,
    pub tokens: &'a [u32],
    pub image_grids: &'a [MropeGrid],
    pub video_grids: &'a [MropeGrid],
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'a>>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum Qwen25VlMropeFilesystemPositionError {
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    #[error(transparent)]
    Assembly(#[from] MropePositionSourceAssemblyError),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen25VlMropeModelInputRequest<'dir, 'data> {
    pub model_dir: &'dir Path,
    pub prompt_token_ids: &'data [u32],
    pub image_grids: &'data [MropeGrid],
    pub video_grids: &'data [MropeGrid],
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'data>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25VlMropeModelInput<'a> {
    pub prompt_token_ids: &'a [u32],
    pub image_grids: &'a [MropeGrid],
    pub video_grids: &'a [MropeGrid],
    pub spatial_merge_size: usize,
    pub metadata: QwenVlMropeMetadata,
    pub mrope_positions: MropePositionIds,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlMropeModelInputError {
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    #[error("Qwen2.5-VL mRoPE model input does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error(transparent)]
    PositionAssembly(#[from] MropePositionAssemblyError),
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlRenderedPromptMropeModelInputRequest<'dir, 'prompt, 'data> {
    pub model_dir: &'dir Path,
    pub rendered_prompt: &'prompt str,
    pub image_grids: &'data [MropeGrid],
    pub video_grids: &'data [MropeGrid],
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'data>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25VlRenderedPromptMropeModelInput<'a> {
    pub prompt_token_ids: Vec<u32>,
    pub image_grids: &'a [MropeGrid],
    pub video_grids: &'a [MropeGrid],
    pub spatial_merge_size: usize,
    pub metadata: QwenVlMropeMetadata,
    pub mrope_positions: MropePositionIds,
}

/// Borrowed access to the one established Qwen2.5-VL mRoPE model-input shape.
///
/// Owned-token adapters implement this so splice and prefill helpers need no copied prompt or
/// position object.
pub trait Qwen25VlMropeModelInputView {
    fn prompt_token_ids(&self) -> &[u32];
    fn image_grids(&self) -> &[MropeGrid];
    fn video_grids(&self) -> &[MropeGrid];
    fn spatial_merge_size(&self) -> usize;
    fn metadata(&self) -> &QwenVlMropeMetadata;
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

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlRenderedPromptMropeModelInputError {
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    #[error("tokenize rendered Qwen2.5-VL prompt with {path:?}: {message}")]
    Tokenize { path: PathBuf, message: String },
    #[error(transparent)]
    ModelInput(#[from] Qwen25VlMropeModelInputError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen25VlProcessorMediaKind {
    Image,
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
    pub pixel_values: Option<&'a HostTensor>,
    pub image_grids: &'a [MropeGrid],
    pub pixel_values_videos: Option<&'a HostTensor>,
    pub video_grids: &'a [MropeGrid],
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlProcessorMediaMropeModelInputRequest<'dir, 'prompt, 'data> {
    pub model_dir: &'dir Path,
    pub rendered_prompt: &'prompt str,
    pub media: Qwen25VlProcessorMedia<'data>,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'data>>,
}

#[derive(Clone, Debug)]
pub struct Qwen25VlProcessorMediaMropeModelInput<'a> {
    pub prompt_token_ids: Vec<u32>,
    pub media: Qwen25VlProcessorMedia<'a>,
    pub spatial_merge_size: usize,
    pub metadata: QwenVlMropeMetadata,
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

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlProcessorMediaError {
    #[error("Qwen2.5-VL processor {kind} grids require a pixel tensor")]
    MissingPixels { kind: Qwen25VlProcessorMediaKind },
    #[error("Qwen2.5-VL processor {kind} pixel tensor has no grids")]
    UnexpectedPixels { kind: Qwen25VlProcessorMediaKind },
    #[error("Qwen2.5-VL processor {kind} pixel tensor has shape {actual:?}; expected rank 2")]
    PixelRank {
        kind: Qwen25VlProcessorMediaKind,
        actual: Vec<usize>,
    },
    #[error("Qwen2.5-VL processor {kind} pixel tensor feature width must be nonzero")]
    ZeroFeatureWidth { kind: Qwen25VlProcessorMediaKind },
    #[error("Qwen2.5-VL processor {kind} raw grid row count overflows usize")]
    GridRowCountOverflow { kind: Qwen25VlProcessorMediaKind },
    #[error(
        "Qwen2.5-VL processor {kind} pixel tensor has {actual} rows; raw grids require {expected}"
    )]
    PixelRowCount {
        kind: Qwen25VlProcessorMediaKind,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Qwen2.5-VL processor {kind} pixel tensor has dtype {actual}; expected f32, bf16 or f16"
    )]
    PixelDtype {
        kind: Qwen25VlProcessorMediaKind,
        actual: DType,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlProcessorMediaMropeModelInputError {
    #[error(transparent)]
    ModelInput(#[from] Qwen25VlRenderedPromptMropeModelInputError),
    #[error(transparent)]
    Media(#[from] Qwen25VlProcessorMediaError),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlTextTowerConfigError {
    #[error("load text-tower config {path:?}: {source}")]
    HfConfig {
        path: PathBuf,
        source: poot_load::LoadError,
    },
    #[error(transparent)]
    Metadata(#[from] MropeConfigMetadataError),
    #[error("Qwen2.5-VL text-tower config does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error(
        "Qwen2.5-VL text-tower mrope_section {mrope_section:?} covers rotary width {actual}, expected rotary_dim = {expected}"
    )]
    MropeSectionRotaryMismatch {
        mrope_section: [usize; 3],
        actual: usize,
        expected: usize,
        rotary_dim: usize,
    },
    #[error(
        "Qwen2.5-VL text-tower mrope_section {mrope_section:?} rotary width overflows usize, expected rotary_dim = {rotary_dim}"
    )]
    MropeSectionRotaryOverflow {
        mrope_section: [usize; 3],
        rotary_dim: usize,
    },
    #[error(
        "Qwen2.5-VL text tower requires qkv_bias=true and qk_norm=false, got qkv_bias={qkv_bias} qk_norm={qk_norm}"
    )]
    UnsupportedAttentionConvention { qkv_bias: bool, qk_norm: bool },
    #[error(
        "Qwen2.5-VL text tower requires head_dim = hidden_size / num_attention_heads, got hidden_size={hidden_size} num_attention_heads={num_attention_heads} head_dim={head_dim:?}"
    )]
    UnsupportedHeadDimConvention {
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: Option<usize>,
    },
    #[error(
        "Qwen2.5-VL text tower requires full-head rotary, got rotary_dim={rotary_dim} head_dim={head_dim}"
    )]
    UnsupportedRotaryConvention { rotary_dim: usize, head_dim: usize },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlVisualSpliceMapError {
    #[error("Qwen2.5-VL visual splice map does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error(
        "Qwen2.5-VL model-input positions have length {positions_len}, prompt has {prompt_len}"
    )]
    PositionLengthMismatch {
        prompt_len: usize,
        positions_len: usize,
    },
    #[error(transparent)]
    Span(#[from] MropeSpanError),
    #[error(
        "Qwen2.5-VL visual splice map expected {expected} visual embedding rows from placeholders, got {actual}"
    )]
    VisualRowCountMismatch { expected: usize, actual: usize },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlPrefillKvEmbedsBindError {
    #[error("Qwen2.5-VL prefill embeds binder does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error(
        "Qwen2.5-VL model-input positions have length {positions_len}, prompt has {prompt_len}"
    )]
    PositionLengthMismatch {
        prompt_len: usize,
        positions_len: usize,
    },
    #[error("Qwen2.5-VL prefill embeds graph has no vlm.input_embeds step input")]
    MissingInputEmbeds,
    #[error(
        "Qwen2.5-VL prefill embeds graph has {count} vlm.input_embeds step inputs; exactly one is required"
    )]
    MultipleInputEmbeds { count: usize },
    #[error("Qwen2.5-VL prefill embeds tensor has shape {actual:?}; expected {expected:?}")]
    InputEmbedsShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Qwen2.5-VL prefill embeds graph has dtype {actual}; expected f32")]
    InputEmbedsGraphDType { actual: DType },
    #[error("Qwen2.5-VL prefill embeds tensor has dtype {actual}; expected f32 embeddings")]
    InputEmbedsDtype { actual: DType },
    #[error("Qwen2.5-VL prefill embeds graph has unexpected slot {slot:?}")]
    UnexpectedSlot { slot: Slot },
    #[error("Qwen2.5-VL prefill embeds graph input v{id} is a const without a name")]
    ConstWithoutName { id: ValueId },
    #[error("Qwen2.5-VL prefill embeds graph has no weight bound for {name}")]
    MissingWeight { name: String },
    #[error(transparent)]
    Mrope(#[from] MropeBindError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlVisualEmbedsPrefillBindError {
    #[error(transparent)]
    Prefill(#[from] Qwen25VlPrefillKvEmbedsBindError),
    #[error(
        "Qwen2.5-VL visual-embeds prefill graph vlm.input_embeds has shape {actual:?}; expected rank 2"
    )]
    GraphInputEmbedsRank { actual: Vec<usize> },
    #[error(
        "Qwen2.5-VL visual-embeds prefill graph vlm.input_embeds sequence length is {actual}; expected prompt length {expected}"
    )]
    GraphPromptLength { expected: usize, actual: usize },
    #[error(
        "Qwen2.5-VL visual-embeds prefill text tensor has shape {actual:?}; expected {expected:?}"
    )]
    TextEmbedsShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error(
        "Qwen2.5-VL visual-embeds prefill text tensor has dtype {actual}; expected f32 embeddings"
    )]
    TextEmbedsDtype { actual: DType },
    #[error(
        "Qwen2.5-VL visual-embeds prefill visual tensor has shape {actual:?}; expected rank 2 with hidden width {expected_hidden}"
    )]
    VisualEmbedsShape {
        expected_hidden: usize,
        actual: Vec<usize>,
    },
    #[error(
        "Qwen2.5-VL visual-embeds prefill visual tensor has dtype {actual}; expected f32 embeddings"
    )]
    VisualEmbedsDtype { actual: DType },
    #[error(transparent)]
    SpliceMap(#[from] Qwen25VlVisualSpliceMapError),
    #[error("Qwen2.5-VL visual-embeds prefill splice graph eval failed: {message}")]
    SpliceEval { message: String },
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn load_qwen2_5_vl_text_tower_config_from_files(
    dir: impl AsRef<Path>,
) -> Result<Qwen2Config, Qwen25VlTextTowerConfigError> {
    let config_path = dir.as_ref().join("config.json");
    let hf_snapshot = load_qwen_vl_text_tower_config_snapshot(&config_path)?;
    let hf_metadata = hf_snapshot.metadata;
    if hf_metadata.model_type() != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlTextTowerConfigError::UnsupportedModelType {
            model_type: hf_metadata.model_type(),
        });
    }
    let metadata = MropeConfigMetadata::try_from(hf_metadata)?;
    let hf_config = hf_snapshot.hf;
    let mut config = qwen2_config_from_hf(&hf_config);
    if !config.qkv_bias || config.qk_norm {
        return Err(
            Qwen25VlTextTowerConfigError::UnsupportedAttentionConvention {
                qkv_bias: config.qkv_bias,
                qk_norm: config.qk_norm,
            },
        );
    }
    if hf_config.num_attention_heads == 0
        || !hf_config
            .hidden_size
            .is_multiple_of(hf_config.num_attention_heads)
    {
        return Err(Qwen25VlTextTowerConfigError::UnsupportedHeadDimConvention {
            hidden_size: hf_config.hidden_size,
            num_attention_heads: hf_config.num_attention_heads,
            head_dim: hf_config.head_dim,
        });
    }
    let expected_head_dim = hf_config.hidden_size / hf_config.num_attention_heads;
    if config.head_dim != expected_head_dim {
        return Err(Qwen25VlTextTowerConfigError::UnsupportedHeadDimConvention {
            hidden_size: hf_config.hidden_size,
            num_attention_heads: hf_config.num_attention_heads,
            head_dim: hf_config.head_dim,
        });
    }
    if config.rotary_dim != config.head_dim {
        return Err(Qwen25VlTextTowerConfigError::UnsupportedRotaryConvention {
            rotary_dim: config.rotary_dim,
            head_dim: config.head_dim,
        });
    }
    let mrope_section = metadata.mrope_section();
    let expected = config.rotary_dim;
    let Some(actual) = mrope_section_rotary_width(mrope_section) else {
        return Err(Qwen25VlTextTowerConfigError::MropeSectionRotaryOverflow {
            mrope_section,
            rotary_dim: config.rotary_dim,
        });
    };
    if actual != expected {
        return Err(Qwen25VlTextTowerConfigError::MropeSectionRotaryMismatch {
            mrope_section,
            actual,
            expected,
            rotary_dim: config.rotary_dim,
        });
    }
    config.mrope_section = Some(mrope_section);
    Ok(config)
}

fn mrope_section_rotary_width(mrope_section: [usize; 3]) -> Option<usize> {
    mrope_section
        .iter()
        .try_fold(0usize, |sum, width| sum.checked_add(*width))
        .and_then(|sum| sum.checked_mul(2))
}

fn load_qwen_vl_text_tower_config_snapshot(
    config_path: &Path,
) -> Result<QwenVlHfConfig, Qwen25VlTextTowerConfigError> {
    let bytes =
        std::fs::read(config_path).map_err(|source| Qwen25VlTextTowerConfigError::HfConfig {
            path: config_path.to_path_buf(),
            source: source.into(),
        })?;
    let hf = serde_json::from_slice::<Qwen2HfConfig>(&bytes).map_err(|source| {
        Qwen25VlTextTowerConfigError::HfConfig {
            path: config_path.to_path_buf(),
            source: source.into(),
        }
    })?;
    let metadata = QwenVlHfMetadata::from_parsed_qwen2_config_and_slice(&hf, &bytes).map_err(
        |err| match err {
            poot_load::qwen_vl::QwenVlHfMetadataError::Json(source) => {
                MropeConfigMetadataError::from(
                    poot_load::qwen_vl::QwenVlHfMetadataError::ParseConfig {
                        path: config_path.to_path_buf(),
                        source,
                    },
                )
            }
            err => MropeConfigMetadataError::from(err),
        },
    )?;
    Ok(QwenVlHfConfig { hf, metadata })
}

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

pub(crate) fn load_qwen_vl_mrope_metadata(
    dir: impl AsRef<Path>,
) -> Result<QwenVlMropeMetadata, QwenVlMropeMetadataLoadError> {
    let sources = load_qwen_vl_mrope_source_metadata(dir)?;
    reconcile_qwen_vl_mrope_source_metadata(sources)
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

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn assemble_qwen2_5_vl_mrope_positions_from_files(
    input: Qwen25VlMropeFilesystemPositionInput<'_>,
) -> Result<MropePositionIds, Qwen25VlMropeFilesystemPositionError> {
    let sources = load_qwen_vl_mrope_source_metadata(input.model_dir)?;
    let processor_tokens = sources.processor_special_tokens();
    let tokenizer_tokens = sources.tokenizer_special_tokens();
    Ok(assemble_qwen2_5_vl_mrope_positions_from_sources(
        Qwen25VlMropePositionSourceInput {
            tokens: input.tokens,
            processor_special_tokens: &processor_tokens,
            tokenizer_special_tokens: &tokenizer_tokens,
            image_grids: input.image_grids,
            video_grids: input.video_grids,
            spatial_merge_size: input.spatial_merge_size,
            temporal_patch_size: input.temporal_patch_size,
            fps: input.fps,
            config: &sources.config,
        },
    )?)
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_files<'data>(
    input: Qwen25VlMropeModelInputRequest<'_, 'data>,
) -> Result<Qwen25VlMropeModelInput<'data>, Qwen25VlMropeModelInputError> {
    let metadata = load_qwen_vl_mrope_metadata(input.model_dir)?;
    let (metadata, mrope_positions) = assemble_qwen2_5_vl_mrope_model_input_parts(
        metadata,
        input.prompt_token_ids,
        input.image_grids,
        input.video_grids,
        input.spatial_merge_size,
        input.temporal_patch_size,
        input.fps,
    )?;
    Ok(Qwen25VlMropeModelInput {
        prompt_token_ids: input.prompt_token_ids,
        image_grids: input.image_grids,
        video_grids: input.video_grids,
        spatial_merge_size: input.spatial_merge_size,
        metadata,
        mrope_positions,
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

pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt<'data>(
    input: Qwen25VlRenderedPromptMropeModelInputRequest<'_, '_, 'data>,
) -> Result<Qwen25VlRenderedPromptMropeModelInput<'data>, Qwen25VlRenderedPromptMropeModelInputError>
{
    let (hf_config, config) = load_qwen_vl_hf_and_config_metadata(input.model_dir)?;
    let model_type = hf_config.model_type();
    if model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
            Qwen25VlMropeModelInputError::UnsupportedModelType { model_type },
        ));
    }
    let processor_tokens =
        QwenVlProcessorSpecialTokenStrings::load_from_dir(input.model_dir, &hf_config)
            .map_err(QwenVlMropeMetadataLoadError::from)?;
    let tokenizer_path = input.model_dir.join("tokenizer.json");
    let tokenizer = load_qwen_vl_tokenizer(&tokenizer_path)?;
    let tokenizer_ids = qwen_vl_tokenizer_ids(&tokenizer, &processor_tokens);
    let encoded = tokenizer
        .encode(input.rendered_prompt, false)
        .map_err(
            |source| Qwen25VlRenderedPromptMropeModelInputError::Tokenize {
                path: tokenizer_path,
                message: source.to_string(),
            },
        )?;
    let prompt_token_ids = encoded.get_ids().to_vec();
    let sources = QwenVlMropeSourceMetadata {
        model_type,
        config,
        processor_tokens,
        tokenizer_ids,
    };
    let metadata = reconcile_qwen_vl_mrope_source_metadata(sources)?;
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

/// Assemble the existing rendered-prompt mRoPE model input and retain validated processor media beside it.
///
/// Host handoff only: pixel columns stay opaque and no vision graph runs.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn assemble_qwen2_5_vl_mrope_model_input_from_processor_media<'data>(
    input: Qwen25VlProcessorMediaMropeModelInputRequest<'_, '_, 'data>,
) -> Result<Qwen25VlProcessorMediaMropeModelInput<'data>, Qwen25VlProcessorMediaMropeModelInputError>
{
    let rendered = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
        Qwen25VlRenderedPromptMropeModelInputRequest {
            model_dir: input.model_dir,
            rendered_prompt: input.rendered_prompt,
            image_grids: input.media.image_grids,
            video_grids: input.media.video_grids,
            spatial_merge_size: input.spatial_merge_size,
            temporal_patch_size: input.temporal_patch_size,
            fps: input.fps,
        },
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

/// Build the inverse map consumed by `poot_models::vision::trace_image_splice` for an already-validated
/// Qwen2.5-VL model input.
///
/// Re-runs the marker-span validator, then assigns visual rows only to image/video placeholder
/// tokens. The vision-start delimiter and ordinary text keep `-1`.
pub(crate) fn qwen2_5_vl_visual_splice_inverse_map(
    input: &impl Qwen25VlMropeModelInputView,
    visual_embedding_rows: usize,
) -> Result<Vec<f32>, Qwen25VlVisualSpliceMapError> {
    if input.metadata().model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlVisualSpliceMapError::UnsupportedModelType {
            model_type: input.metadata().model_type,
        });
    }
    if input.mrope_positions().len() != input.prompt_token_ids().len() {
        return Err(Qwen25VlVisualSpliceMapError::PositionLengthMismatch {
            prompt_len: input.prompt_token_ids().len(),
            positions_len: input.mrope_positions().len(),
        });
    }
    discover_mrope_segments(
        input.prompt_token_ids(),
        input.metadata().special_tokens,
        input.image_grids(),
        input.video_grids(),
        input.spatial_merge_size(),
    )?;

    let ids = input.metadata().special_tokens;
    let mut inv = vec![-1.0f32; input.prompt_token_ids().len()];
    let mut next_visual = 0usize;
    for (index, &token) in input.prompt_token_ids().iter().enumerate() {
        if token == ids.image || token == ids.video {
            inv[index] = next_visual as f32;
            next_visual += 1;
        }
    }
    if next_visual != visual_embedding_rows {
        return Err(Qwen25VlVisualSpliceMapError::VisualRowCountMismatch {
            expected: next_visual,
            actual: visual_embedding_rows,
        });
    }
    Ok(inv)
}

/// Bind a Qwen2.5-VL `trace_qwen2_5_vl_prefill_kv_embeds` graph for CPU oracle execution.
///
/// The graph takes already-spliced embeddings, so no token-embedding gather or vision tower runs.
/// Binds named text-tower weights, the causal mask, the packed mRoPE positions from the validated
/// model input, and zero initial KV state.
pub(crate) fn bind_qwen2_5_vl_prefill_kv_embeds_inputs(
    graph: &Graph,
    weights: &HashMap<String, Value>,
    model_input: &impl Qwen25VlMropeModelInputView,
    input_embeds: &HostTensor,
) -> Result<HashMap<ValueId, Value>, Qwen25VlPrefillKvEmbedsBindError> {
    if model_input.metadata().model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlPrefillKvEmbedsBindError::UnsupportedModelType {
            model_type: model_input.metadata().model_type,
        });
    }
    if model_input.mrope_positions().len() != model_input.prompt_token_ids().len() {
        return Err(Qwen25VlPrefillKvEmbedsBindError::PositionLengthMismatch {
            prompt_len: model_input.prompt_token_ids().len(),
            positions_len: model_input.mrope_positions().len(),
        });
    }

    let (_, expected_input_shape) = qwen25_prefill_input_embeds_meta(graph)?;
    if input_embeds.shape() != expected_input_shape {
        return Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsShape {
            expected: expected_input_shape,
            actual: input_embeds.shape().to_vec(),
        });
    }
    if input_embeds.dtype() != DType::F32 {
        return Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsDtype {
            actual: input_embeds.dtype(),
        });
    }

    let mut inputs = HashMap::new();
    for &id in &graph.inputs {
        let meta = graph.meta(id);
        let tensor = match meta.storage {
            Storage::Computed(computed) => HostTensor::f32(computed.shape(), computed.values_f32()),
            Storage::Slot(Slot::Mask) => {
                let name = meta
                    .name
                    .as_deref()
                    .ok_or(Qwen25VlPrefillKvEmbedsBindError::ConstWithoutName { id })?;
                if name != "mask.prefill" {
                    return Err(Qwen25VlPrefillKvEmbedsBindError::UnexpectedSlot {
                        slot: Slot::Mask,
                    });
                }
                prefill_causal_mask(model_input.prompt_token_ids().len(), None)
            }
            Storage::Slot(Slot::Activation) => {
                let name = meta
                    .name
                    .as_deref()
                    .ok_or(Qwen25VlPrefillKvEmbedsBindError::ConstWithoutName { id })?;
                if name != "activation.vlm.input_embeds" {
                    return Err(Qwen25VlPrefillKvEmbedsBindError::UnexpectedSlot {
                        slot: Slot::Activation,
                    });
                }
                input_embeds.clone()
            }
            Storage::Const => {
                let name = meta
                    .name
                    .as_deref()
                    .ok_or(Qwen25VlPrefillKvEmbedsBindError::ConstWithoutName { id })?;
                let weight = weights.get(name).cloned().ok_or_else(|| {
                    Qwen25VlPrefillKvEmbedsBindError::MissingWeight {
                        name: name.to_string(),
                    }
                })?;
                inputs.insert(id, weight);
                continue;
            }
            Storage::Slot(Slot::MropePosition) => continue,
            // Card 550: the causal mask (above) is now a graph computation over `Slot::Pos` and
            // `iota`; this one-shot prefill always starts at position 0. Distinct from the mRoPE
            // temporal/height/width `Slot::MropePosition` above, which drives RoPE angle, not
            // causal order.
            Storage::Slot(Slot::Pos) => crate::core::graphs::prefill_pos_rows(
                &meta.aval,
                model_input.prompt_token_ids().len(),
            ),
            Storage::Slot(slot) => {
                return Err(Qwen25VlPrefillKvEmbedsBindError::UnexpectedSlot { slot });
            }
            Storage::State => HostTensor::zeros(meta.aval.shape.clone()),
            Storage::Device => unreachable!("device values are not graph inputs"),
        };
        inputs.insert(id, tensor.into());
    }
    bind_mrope_prefill_positions(graph, &mut inputs, model_input.mrope_positions())?;
    Ok(inputs)
}

fn qwen25_prefill_input_embeds_meta(
    graph: &Graph,
) -> Result<(ValueId, Vec<usize>), Qwen25VlPrefillKvEmbedsBindError> {
    let input_embed_ids = graph
        .inputs
        .iter()
        .copied()
        .filter(|&id| {
            graph.meta(id).storage == Storage::Slot(Slot::Activation)
                && graph.meta(id).name.as_deref() == Some("activation.vlm.input_embeds")
        })
        .collect::<Vec<_>>();
    let input_embed_id = match input_embed_ids.as_slice() {
        [] => return Err(Qwen25VlPrefillKvEmbedsBindError::MissingInputEmbeds),
        [id] => *id,
        _ => {
            return Err(Qwen25VlPrefillKvEmbedsBindError::MultipleInputEmbeds {
                count: input_embed_ids.len(),
            });
        }
    };
    let input_embed_aval = graph.aval(input_embed_id);
    if input_embed_aval.dtype != DType::F32 {
        return Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsGraphDType {
            actual: input_embed_aval.dtype,
        });
    }
    Ok((input_embed_id, input_embed_aval.shape.clone()))
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
    graph: &Graph,
    weights: &HashMap<String, Value>,
    model_input: &impl Qwen25VlMropeModelInputView,
    text_embeds: &HostTensor,
    visual_embeds: &HostTensor,
) -> Result<HashMap<ValueId, Value>, Qwen25VlVisualEmbedsPrefillBindError> {
    let (_, expected_input_shape) = qwen25_prefill_input_embeds_meta(graph)?;
    if expected_input_shape.len() != 2 {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::GraphInputEmbedsRank {
            actual: expected_input_shape,
        });
    }
    if expected_input_shape[0] != model_input.prompt_token_ids().len() {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::GraphPromptLength {
            expected: model_input.prompt_token_ids().len(),
            actual: expected_input_shape[0],
        });
    }
    if text_embeds.shape() != expected_input_shape {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::TextEmbedsShape {
            expected: expected_input_shape,
            actual: text_embeds.shape().to_vec(),
        });
    }
    if text_embeds.dtype() != DType::F32 {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::TextEmbedsDtype {
            actual: text_embeds.dtype(),
        });
    }

    let hidden = expected_input_shape[1];
    if visual_embeds.shape().len() != 2 || visual_embeds.shape()[1] != hidden {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::VisualEmbedsShape {
            expected_hidden: hidden,
            actual: visual_embeds.shape().to_vec(),
        });
    }
    if visual_embeds.dtype() != DType::F32 {
        return Err(Qwen25VlVisualEmbedsPrefillBindError::VisualEmbedsDtype {
            actual: visual_embeds.dtype(),
        });
    }

    let inverse = qwen2_5_vl_visual_splice_inverse_map(model_input, visual_embeds.shape()[0])?;
    let builder = Builder::new();
    let text = builder.constant(
        "qwen25_vl.text_embeds",
        TensorType::f32(text_embeds.shape().to_vec()),
    );
    let visual = builder.constant(
        "qwen25_vl.visual_embeds",
        TensorType::f32(visual_embeds.shape().to_vec()),
    );
    let inv = builder.constant(
        "qwen25_vl.visual_splice_inverse",
        TensorType::f32(vec![model_input.prompt_token_ids().len()]),
    );
    let spliced = poot_models::vision::trace_image_splice(&builder, text, visual, inv);
    let splice_graph = builder.finish(spliced);
    let splice_inputs: HashMap<_, Value> = HashMap::from([
        (text.id, Value::from(text_embeds.clone())),
        (visual.id, Value::from(visual_embeds.clone())),
        (
            inv.id,
            Value::from(HostTensor::f32(
                vec![model_input.prompt_token_ids().len()],
                inverse.clone(),
            )),
        ),
    ]);
    let input_embeds = poot_eval::eval(
        &splice_graph,
        &splice_inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .map_err(|source| Qwen25VlVisualEmbedsPrefillBindError::SpliceEval {
        message: source.to_string(),
    })?
    .output
    .into_host()
    .map_err(|source| Qwen25VlVisualEmbedsPrefillBindError::SpliceEval {
        message: source.to_string(),
    })?;
    Ok(bind_qwen2_5_vl_prefill_kv_embeds_inputs(
        graph,
        weights,
        model_input,
        &input_embeds,
    )?)
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn qwen_vl_text_tower_config_applies_released_shape_mrope_section() {
        let dir = fixture_dir("text_tower_config_success");
        std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();

        let config = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap();

        assert_eq!(config.vocab, 152064);
        assert_eq!(config.hidden, 3584);
        assert_eq!(config.inter, 18944);
        assert_eq!(config.layers, 28);
        assert_eq!(config.n_heads, 28);
        assert_eq!(config.n_kv_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.max_pos, 128000);
        assert_eq!(config.sliding_window, None);
        assert!(config.qkv_bias);
        assert!(!config.qk_norm);
        assert_eq!(config.mrope_section, Some([16, 24, 24]));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_wrong_model_kind() {
        let config_json = qwen2_vl_config_json();
        let dir = fixture_dir("text_tower_config_qwen2_vl");
        std::fs::write(dir.join("config.json"), config_json).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::UnsupportedModelType {
                model_type: QwenVlModelType::Qwen2Vl
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_malformed_section_metadata() {
        let malformed = CONFIG_JSON.replace(
            "\"mrope_section\": [16, 24, 24]",
            "\"mrope_section\": [16, 24]",
        );
        let dir = fixture_dir("text_tower_config_malformed_section");
        std::fs::write(dir.join("config.json"), malformed).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::Metadata(MropeConfigMetadataError::Load(
                poot_load::qwen_vl::QwenVlHfMetadataError::WrongMropeSectionLength { actual: 2 }
            ))
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_section_sum_mismatch() {
        let mismatched = CONFIG_JSON.replace(
            "\"mrope_section\": [16, 24, 24]",
            "\"mrope_section\": [8, 8, 8]",
        );
        let dir = fixture_dir("text_tower_config_section_sum");
        std::fs::write(dir.join("config.json"), mismatched).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::MropeSectionRotaryMismatch {
                mrope_section: [8, 8, 8],
                actual: 48,
                expected: 128,
                rotary_dim: 128
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_odd_full_rotary_width() {
        let odd_rotary = CONFIG_JSON
            .replace("\"hidden_size\": 3584", "\"hidden_size\": 7")
            .replace("\"num_attention_heads\": 28", "\"num_attention_heads\": 1")
            .replace("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 1")
            .replace(
                "\"mrope_section\": [16, 24, 24]",
                "\"mrope_section\": [1, 1, 1]",
            );
        let dir = fixture_dir("text_tower_config_odd_rotary");
        std::fs::write(dir.join("config.json"), odd_rotary).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::MropeSectionRotaryMismatch {
                mrope_section: [1, 1, 1],
                actual: 6,
                expected: 7,
                rotary_dim: 7
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_non_qwen2_attention_convention() {
        let no_bias = CONFIG_JSON.replace(
            "\"use_sliding_window\": false,",
            "\"use_sliding_window\": false,\n        \"attention_bias\": false,",
        );
        let dir = fixture_dir("text_tower_config_attention");
        std::fs::write(dir.join("config.json"), no_bias).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::UnsupportedAttentionConvention {
                qkv_bias: false,
                qk_norm: false
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_explicit_head_dim_mismatch() {
        let mismatched_head = CONFIG_JSON.replace(
            "\"num_key_value_heads\": 4,",
            "\"num_key_value_heads\": 4,\n        \"head_dim\": 256,",
        );
        let dir = fixture_dir("text_tower_config_head_dim");
        std::fs::write(dir.join("config.json"), mismatched_head).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::UnsupportedHeadDimConvention {
                hidden_size: 3584,
                num_attention_heads: 28,
                head_dim: Some(256)
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_rejects_partial_rotary() {
        let partial_rotary = CONFIG_JSON.replace(
            "\"use_sliding_window\": false,",
            "\"use_sliding_window\": false,\n        \"partial_rotary_factor\": 0.5,",
        );
        let dir = fixture_dir("text_tower_config_partial_rotary");
        std::fs::write(dir.join("config.json"), partial_rotary).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::UnsupportedRotaryConvention {
                rotary_dim: 64,
                head_dim: 128
            }
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_reports_base_config_parse_path() {
        let missing_hidden = CONFIG_JSON.replace("\n        \"hidden_size\": 3584,", "");
        let dir = fixture_dir("text_tower_config_base_parse_path");
        std::fs::write(dir.join("config.json"), missing_hidden).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::HfConfig { ref path, .. }
                if path == &dir.join("config.json")
        ));
    }

    #[test]
    fn qwen_vl_text_tower_config_reports_metadata_parse_path() {
        let duplicate_section = CONFIG_JSON.replace(
            "\"mrope_section\": [16, 24, 24]",
            "\"mrope_section\": [16, 24, 24], \"mrope_section\": [8, 8, 8]",
        );
        let dir = fixture_dir("text_tower_config_metadata_parse_path");
        std::fs::write(dir.join("config.json"), duplicate_section).unwrap();

        let err = load_qwen2_5_vl_text_tower_config_from_files(&dir).unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlTextTowerConfigError::Metadata(MropeConfigMetadataError::Load(
                poot_load::qwen_vl::QwenVlHfMetadataError::ParseConfig { ref path, .. }
            )) if path == &dir.join("config.json")
        ));
    }

    #[test]
    fn qwen_vl_filesystem_adapter_delegates_to_existing_source_assembly() {
        let dir = write_fixture("success", IMAGE_ID, true);
        let tokens = [
            0,
            VISION_START_ID,
            IMAGE_ID,
            1,
            VISION_START_ID,
            VIDEO_ID,
            VIDEO_ID,
            2,
        ];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(2, 2, 2)];
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let sources = load_qwen_vl_mrope_source_metadata(&dir).unwrap();
        let processor_tokens = sources.processor_special_tokens();
        let tokenizer_tokens = sources.tokenizer_special_tokens();
        let pure =
            assemble_qwen2_5_vl_mrope_positions_from_sources(Qwen25VlMropePositionSourceInput {
                tokens: &tokens,
                processor_special_tokens: &processor_tokens,
                tokenizer_special_tokens: &tokenizer_tokens,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: None,
                config: &sources.config,
            })
            .unwrap();
        let from_files =
            assemble_qwen2_5_vl_mrope_positions_from_files(Qwen25VlMropeFilesystemPositionInput {
                model_dir: &dir,
                tokens: &tokens,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: None,
            })
            .unwrap();
        assert_eq!(from_files, pure);

        let metadata = load_qwen_vl_mrope_metadata(&dir).unwrap();
        assert_eq!(metadata.model_type, QwenVlModelType::Qwen25Vl);
        assert_eq!(
            metadata.special_tokens,
            MropeSpecialTokenIds::new(VISION_START_ID, IMAGE_ID, VIDEO_ID)
        );
    }

    #[test]
    fn qwen_vl_production_model_input_borrows_inputs_and_matches_pure_positions() {
        let dir = write_fixture("model_input_success", IMAGE_ID, true);
        let tokens = [
            0,
            VISION_START_ID,
            IMAGE_ID,
            1,
            VISION_START_ID,
            VIDEO_ID,
            VIDEO_ID,
            2,
        ];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(2, 2, 2)];
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let fps = [4.0];
        let metadata = load_qwen_vl_mrope_metadata(&dir).unwrap();
        let expected = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &tokens,
            special_tokens: metadata.special_tokens,
            image_grids: &image_grids,
            video_grids: &video_grids,
            spatial_merge_size: 2,
            temporal_patch_size,
            fps: Some(MropeProcessorFpsInput::PerVideo(&fps)),
            config: &metadata.config,
        })
        .unwrap();

        let actual =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &tokens,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: Some(MropeProcessorFpsInput::PerVideo(&fps)),
            })
            .unwrap();

        assert!(std::ptr::eq(actual.prompt_token_ids, tokens.as_slice()));
        assert!(std::ptr::eq(actual.image_grids, image_grids.as_slice()));
        assert!(std::ptr::eq(actual.video_grids, video_grids.as_slice()));
        assert_eq!(actual.metadata, metadata);
        assert_eq!(actual.mrope_positions, expected);
    }

    #[test]
    fn qwen_vl_production_model_input_rejects_qwen2_vl_before_prompt_errors() {
        let config = qwen2_vl_config_json();
        let dir = write_fixture_with_config("model_input_qwen2_vl", &config, IMAGE_ID, true);
        let metadata = load_qwen_vl_mrope_metadata(&dir).unwrap();
        assert_eq!(metadata.model_type, QwenVlModelType::Qwen2Vl);
        assert_eq!(metadata.config.temporal_tokens_per_second(), None);

        let err =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &[VISION_START_ID],
                image_grids: &[MropeGrid::new(0, 1, 1)],
                video_grids: &[],
                spatial_merge_size: 0,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            })
            .unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlMropeModelInputError::UnsupportedModelType {
                model_type: QwenVlModelType::Qwen2Vl
            }
        ));
    }

    #[test]
    fn qwen_vl_production_model_input_outlives_model_dir_borrow() {
        fn assemble_with_ephemeral_dir<'data>(
            tokens: &'data [u32],
            image_grids: &'data [MropeGrid],
            video_grids: &'data [MropeGrid],
        ) -> Qwen25VlMropeModelInput<'data> {
            let dir = write_fixture("model_input_short_dir_lifetime", IMAGE_ID, true);
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: tokens,
                image_grids,
                video_grids,
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            })
            .unwrap()
        }

        let tokens = [0, VISION_START_ID, IMAGE_ID, 1];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [];
        let actual = assemble_with_ephemeral_dir(&tokens, &image_grids, &video_grids);

        assert!(std::ptr::eq(actual.prompt_token_ids, tokens.as_slice()));
        assert!(std::ptr::eq(actual.image_grids, image_grids.as_slice()));
        assert_eq!(actual.video_grids, &[]);
        assert_eq!(
            actual.metadata.special_tokens,
            MropeSpecialTokenIds::new(VISION_START_ID, IMAGE_ID, VIDEO_ID)
        );
        assert_eq!(actual.mrope_positions.len(), tokens.len());
    }

    #[test]
    fn qwen_vl_rendered_prompt_model_input_tokenizes_plain_text_and_owns_ids() {
        let dir = write_fixture("rendered_prompt_plain_text", IMAGE_ID, true);
        let expected_tokens = [0, 1, 2];
        let expected =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &expected_tokens,
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            })
            .unwrap();

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
        let expected =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &expected_tokens,
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: None,
            })
            .unwrap();
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
        let expected =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &expected_tokens,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            })
            .unwrap();
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

        let qwen2_config = qwen2_vl_config_json();
        let qwen2_dir =
            write_fixture_with_config("rendered_prompt_qwen2_vl", &qwen2_config, IMAGE_ID, true);
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &qwen2_dir,
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
            Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                Qwen25VlMropeModelInputError::UnsupportedModelType {
                    model_type: QwenVlModelType::Qwen2Vl
                }
            )
        ));

        let qwen2_missing_tokenizer = write_fixture_with_config(
            "rendered_prompt_qwen2_vl_missing_tokenizer",
            &qwen2_config,
            IMAGE_ID,
            true,
        );
        std::fs::remove_file(qwen2_missing_tokenizer.join("tokenizer.json")).unwrap();
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &qwen2_missing_tokenizer,
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
            Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                Qwen25VlMropeModelInputError::UnsupportedModelType {
                    model_type: QwenVlModelType::Qwen2Vl
                }
            )
        ));

        let qwen2_malformed_tokenizer = write_fixture_with_config(
            "rendered_prompt_qwen2_vl_malformed_tokenizer",
            &qwen2_config,
            IMAGE_ID,
            true,
        );
        std::fs::write(qwen2_malformed_tokenizer.join("tokenizer.json"), "{").unwrap();
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &qwen2_malformed_tokenizer,
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
            Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                Qwen25VlMropeModelInputError::UnsupportedModelType {
                    model_type: QwenVlModelType::Qwen2Vl
                }
            )
        ));

        let span_mismatch = write_fixture("rendered_prompt_span_mismatch", IMAGE_ID, true);
        let image_grids = [MropeGrid::new(1, 2, 4)];
        let rendered = format!("t0 {VISION_START} {IMAGE_PAD} t1");
        let err = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
            Qwen25VlRenderedPromptMropeModelInputRequest {
                model_dir: &span_mismatch,
                rendered_prompt: &rendered,
                image_grids: &image_grids,
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlRenderedPromptMropeModelInputError::ModelInput(
                Qwen25VlMropeModelInputError::PositionAssembly(MropePositionAssemblyError::Span(
                    MropeSpanError::PlaceholderCountMismatch {
                        start_index: 1,
                        expected: 2,
                        actual: 1,
                        ..
                    }
                ))
            )
        ));
    }

    #[test]
    fn qwen25_vl_processor_media_handoff_reuses_rendered_input_and_cpu_prefill() {
        let dir = write_fixture("processor_media_success", IMAGE_ID, true);
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(2, 2, 2)];
        let image_pixels = HostTensor::f32(vec![4, 6], fill(4 * 6, 11));
        let video_pixels = HostTensor::f32(vec![8, 6], fill(8 * 6, 12));
        let rendered =
            format!("t0 {VISION_START} {VIDEO_PAD} {VIDEO_PAD} t1 {VISION_START} {IMAGE_PAD} t2");
        let media = Qwen25VlProcessorMedia {
            pixel_values: Some(&image_pixels),
            image_grids: &image_grids,
            pixel_values_videos: Some(&video_pixels),
            video_grids: &video_grids,
        };
        let actual = assemble_qwen2_5_vl_mrope_model_input_from_processor_media(
            Qwen25VlProcessorMediaMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &rendered,
                media,
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            },
        )
        .unwrap();
        let expected = assemble_qwen2_5_vl_mrope_model_input_from_rendered_prompt(
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
        let borrowed =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &expected.prompt_token_ids,
                image_grids: &image_grids,
                video_grids: &video_grids,
                spatial_merge_size: 2,
                temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
                fps: None,
            })
            .unwrap();

        assert_eq!(actual.prompt_token_ids, expected.prompt_token_ids);
        assert_eq!(actual.metadata, expected.metadata);
        assert_eq!(actual.mrope_positions, expected.mrope_positions);
        assert_eq!(
            qwen2_5_vl_visual_splice_inverse_map(&actual, 3).unwrap(),
            qwen2_5_vl_visual_splice_inverse_map(&expected, 3).unwrap(),
        );
        assert_eq!(
            qwen2_5_vl_visual_splice_inverse_map(&actual, 3).unwrap(),
            qwen2_5_vl_visual_splice_inverse_map(&borrowed, 3).unwrap(),
        );
        assert!(std::ptr::eq(
            actual.prompt_token_ids(),
            actual.prompt_token_ids.as_slice()
        ));
        assert!(std::ptr::eq(
            actual.mrope_positions(),
            &actual.mrope_positions
        ));
        assert!(std::ptr::eq(
            actual.media.pixel_values.unwrap(),
            &image_pixels
        ));
        assert!(std::ptr::eq(
            actual.media.pixel_values_videos.unwrap(),
            &video_pixels
        ));
        assert!(std::ptr::eq(actual.media.image_grids, &image_grids));
        assert!(std::ptr::eq(actual.media.video_grids, &video_grids));

        let cfg = tiny_qwen25_text_cfg();
        let graph = poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(
            cfg,
            actual.prompt_token_ids().len(),
            actual.prompt_token_ids().len() + 1,
        );
        let weights = weights_for_graph(&graph);
        let text_embeds = HostTensor::f32(
            vec![actual.prompt_token_ids().len(), cfg.hidden],
            fill(actual.prompt_token_ids().len() * cfg.hidden, 13),
        );
        let visual_embeds = HostTensor::f32(vec![3, cfg.hidden], fill(3 * cfg.hidden, 14));
        let bound = bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
            &graph,
            &weights,
            &actual,
            &text_embeds,
            &visual_embeds,
        )
        .unwrap();
        let rendered_bound = bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
            &graph,
            &weights,
            &expected,
            &text_embeds,
            &visual_embeds,
        )
        .unwrap();
        let borrowed_bound = bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
            &graph,
            &weights,
            &borrowed,
            &text_embeds,
            &visual_embeds,
        )
        .unwrap();
        assert_eq!(bound, rendered_bound);
        assert_eq!(bound, borrowed_bound);
        let (logits, state) = crate::core::cpu_oracle::cpu_eval_with_state(&graph, &bound).unwrap();
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            state
                .iter()
                .flat_map(|tensor| tensor.as_f32().unwrap().iter())
                .all(|value| value.is_finite())
        );
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

    fn tiny_qwen25_text_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab: 32,
            hidden: 16,
            inter: 32,
            layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            rotary_dim: 8,
            eps: 1e-6,
            max_pos: 16,
            qkv_bias: true,
            qk_norm: false,
            mrope_section: Some([1, 1, 2]),
            ..Default::default()
        }
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

    fn test_weight(name: &str, shape: &[usize]) -> HostTensor {
        let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
            (h ^ c as u64).wrapping_mul(1099511628211)
        });
        HostTensor::f32(
            shape.to_vec(),
            fill(shape.iter().product::<usize>(), seed)
                .iter()
                .map(|v| v * 0.1)
                .collect(),
        )
    }

    fn weights_for_graph(graph: &Graph) -> HashMap<String, Value> {
        graph
            .inputs
            .iter()
            .filter_map(|&id| {
                let meta = graph.meta(id);
                if meta.storage != Storage::Const {
                    return None;
                }
                let name = meta.name.as_deref().unwrap();
                Some((name.to_string(), test_weight(name, &meta.aval.shape).into()))
            })
            .collect()
    }

    fn model_input_for_tokens<'data>(
        dir: &Path,
        tokens: &'data [u32],
        image_grids: &'data [MropeGrid],
        video_grids: &'data [MropeGrid],
    ) -> Qwen25VlMropeModelInput<'data> {
        assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
            model_dir: dir,
            prompt_token_ids: tokens,
            image_grids,
            video_grids,
            spatial_merge_size: 2,
            temporal_patch_size: MropeTemporalPatchSize::new(2).unwrap(),
            fps: None,
        })
        .unwrap()
    }

    #[test]
    fn qwen25_vl_visual_splice_map_replaces_only_validated_placeholders() {
        use poot_graph_ir::TensorType;
        use poot_graph_ir::builder::Builder;
        use poot_models::vision::trace_image_splice;

        let dir = write_fixture("visual_splice_map", IMAGE_ID, true);
        let tokens = [
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
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(1, 2, 2), MropeGrid::new(2, 2, 2)];
        let input = model_input_for_tokens(&dir, &tokens, &image_grids, &video_grids);

        let inv = qwen2_5_vl_visual_splice_inverse_map(&input, 4).unwrap();
        assert_eq!(
            inv,
            vec![-1.0, -1.0, 0.0, -1.0, -1.0, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0]
        );

        let b = Builder::new();
        let text = b.constant("text", TensorType::f32(vec![tokens.len(), 4]));
        let visual = b.constant("visual", TensorType::f32(vec![4, 4]));
        let inv_traced = b.constant("inv", TensorType::f32(vec![tokens.len()]));
        let out = trace_image_splice(&b, text, visual, inv_traced);
        let graph = b.finish(out);

        let text_rows: Vec<f32> = (0..tokens.len() * 4).map(|v| v as f32).collect();
        let visual_rows: Vec<f32> = (100..116).map(|v| v as f32).collect();
        let inputs: HashMap<_, poot_eval::Value> = HashMap::from([
            (
                text.id,
                HostTensor::f32(vec![tokens.len(), 4], text_rows.clone()).into(),
            ),
            (
                visual.id,
                HostTensor::f32(vec![4, 4], visual_rows.clone()).into(),
            ),
            (
                inv_traced.id,
                HostTensor::f32(vec![tokens.len()], inv).into(),
            ),
        ]);
        let spliced = poot_eval::eval(
            &graph,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap();

        for row in 0..tokens.len() {
            let got = &spliced.as_f32().unwrap()[row * 4..(row + 1) * 4];
            let want = match row {
                2 => &visual_rows[0..4],
                5 => &visual_rows[4..8],
                8 => &visual_rows[8..12],
                9 => &visual_rows[12..16],
                _ => &text_rows[row * 4..(row + 1) * 4],
            };
            assert_eq!(got, want);
        }
    }

    #[test]
    fn qwen25_vl_visual_splice_map_validates_rows_and_model_input() {
        let dir = write_fixture("visual_splice_map_errors", IMAGE_ID, true);
        let tokens = [0, VISION_START_ID, IMAGE_ID, 1];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let input = model_input_for_tokens(&dir, &tokens, &image_grids, &[]);

        assert_eq!(
            qwen2_5_vl_visual_splice_inverse_map(&input, 2),
            Err(Qwen25VlVisualSpliceMapError::VisualRowCountMismatch {
                expected: 1,
                actual: 2,
            })
        );

        let qwen2_config = qwen2_vl_config_json();
        let qwen2_dir = write_fixture_with_config(
            "visual_splice_map_wrong_kind",
            &qwen2_config,
            IMAGE_ID,
            true,
        );
        let qwen2_metadata = load_qwen_vl_mrope_metadata(&qwen2_dir).unwrap();
        let wrong_kind = Qwen25VlMropeModelInput {
            prompt_token_ids: &tokens,
            image_grids: &image_grids,
            video_grids: &[],
            spatial_merge_size: 2,
            metadata: qwen2_metadata,
            mrope_positions: input.mrope_positions.clone(),
        };
        assert!(matches!(
            qwen2_5_vl_visual_splice_inverse_map(&wrong_kind, 1),
            Err(Qwen25VlVisualSpliceMapError::UnsupportedModelType {
                model_type: QwenVlModelType::Qwen2Vl
            })
        ));

        let malformed = Qwen25VlMropeModelInput {
            prompt_token_ids: &[VISION_START_ID],
            image_grids: &[],
            video_grids: &[],
            spatial_merge_size: 2,
            metadata: input.metadata.clone(),
            mrope_positions: input.mrope_positions.clone(),
        };
        assert!(matches!(
            qwen2_5_vl_visual_splice_inverse_map(&malformed, 0),
            Err(Qwen25VlVisualSpliceMapError::PositionLengthMismatch {
                prompt_len: 1,
                positions_len: 4,
            })
        ));

        let invalid_tokens = [VISION_START_ID, 0, 1, 2];
        let invalid_positions = poot_models::mrope::build_mrope_position_ids(
            &[poot_models::mrope::MropeSegment::Text(4)],
            1,
        )
        .unwrap();
        let invalid_span = Qwen25VlMropeModelInput {
            prompt_token_ids: &invalid_tokens,
            image_grids: &[],
            video_grids: &[],
            spatial_merge_size: 2,
            metadata: input.metadata.clone(),
            mrope_positions: invalid_positions,
        };
        assert!(matches!(
            qwen2_5_vl_visual_splice_inverse_map(&invalid_span, 0),
            Err(Qwen25VlVisualSpliceMapError::Span(
                MropeSpanError::MissingVisualMarker {
                    start_index: 0,
                    index: 1,
                }
            ))
        ));
    }

    #[test]
    fn qwen25_vl_prefill_kv_embeds_binder_binds_complete_cpu_inputs() {
        let dir = write_fixture("prefill_kv_embeds_bind_success", IMAGE_ID, true);
        let tokens = [0, 1, 2, 0];
        let input = model_input_for_tokens(&dir, &tokens, &[], &[]);
        let cfg = tiny_qwen25_text_cfg();
        let graph = poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(cfg, tokens.len(), 6);
        graph.validate().unwrap();
        let weights = weights_for_graph(&graph);
        let input_embeds = HostTensor::f32(
            vec![tokens.len(), cfg.hidden],
            fill(tokens.len() * cfg.hidden, 77),
        );

        let bound =
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &input_embeds)
                .unwrap();
        assert_eq!(bound.len(), graph.inputs.len());
        let mrope_id = graph
            .slots
            .iter()
            .find_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
            .unwrap();
        assert_eq!(
            bound[&mrope_id].as_host().expect("dense weight").as_i32(),
            Some(&[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3][..])
        );
        for &(state_in, _) in &graph.state {
            assert_eq!(
                bound[&state_in],
                poot_eval::Value::from(HostTensor::zeros(graph.aval(state_in).shape.clone()))
            );
        }

        let (logits, state) = crate::core::cpu_oracle::cpu_eval_with_state(&graph, &bound).unwrap();
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        assert_eq!(state.len(), 2 * cfg.layers);
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn qwen25_vl_prefill_kv_embeds_binder_reports_shape_and_contract_errors() {
        let dir = write_fixture("prefill_kv_embeds_bind_errors", IMAGE_ID, true);
        let tokens = [0, 1, 2];
        let input = model_input_for_tokens(&dir, &tokens, &[], &[]);
        let cfg = tiny_qwen25_text_cfg();
        let graph = poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(cfg, tokens.len(), 5);
        let weights = weights_for_graph(&graph);

        let wrong_shape = HostTensor::f32(
            vec![tokens.len() + 1, cfg.hidden],
            vec![0.0; 4 * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &wrong_shape),
            Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsShape {
                expected: vec![tokens.len(), cfg.hidden],
                actual: vec![tokens.len() + 1, cfg.hidden],
            })
        );

        let integer_embeds = HostTensor::i32(
            vec![tokens.len(), cfg.hidden],
            vec![0; tokens.len() * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &integer_embeds),
            Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsDtype { actual: DType::I32 })
        );

        let expected_embed_len = tokens.len() * cfg.hidden;
        // `vlm.input_embeds` is an f32 slot: a 16-bit float tensor is refused by dtype, never widened.
        let bf16_embeds =
            HostTensor::bf16(vec![tokens.len(), cfg.hidden], vec![0; expected_embed_len]);
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &bf16_embeds),
            Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsDtype {
                actual: DType::BF16,
            })
        );
        let f16_embeds =
            HostTensor::f16(vec![tokens.len(), cfg.hidden], vec![0; expected_embed_len]);
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &f16_embeds),
            Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsDtype { actual: DType::F16 })
        );

        let input_embed_id = graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).name.as_deref() == Some("activation.vlm.input_embeds"))
            .unwrap();
        let mut wrong_dtype_graph = graph.clone();
        wrong_dtype_graph.values[input_embed_id].aval.dtype = DType::BF16;
        let embeds = HostTensor::f32(
            vec![tokens.len(), cfg.hidden],
            vec![0.0; expected_embed_len],
        );
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&wrong_dtype_graph, &weights, &input, &embeds),
            Err(Qwen25VlPrefillKvEmbedsBindError::InputEmbedsGraphDType {
                actual: DType::BF16,
            })
        );

        let ordinary_graph = poot_models::qwen2::trace_prefill_kv_embeds(cfg, tokens.len(), 5);
        let ordinary_weights = weights_for_graph(&ordinary_graph);
        let embeds = HostTensor::f32(
            vec![tokens.len(), cfg.hidden],
            vec![0.0; tokens.len() * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(
                &ordinary_graph,
                &ordinary_weights,
                &input,
                &embeds
            ),
            Err(Qwen25VlPrefillKvEmbedsBindError::Mrope(
                MropeBindError::MissingSlot
            ))
        );

        let token_graph = poot_models::qwen2::trace_prefill_kv(cfg, tokens.len(), 5);
        let token_weights = weights_for_graph(&token_graph);
        assert_eq!(
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&token_graph, &token_weights, &input, &embeds),
            Err(Qwen25VlPrefillKvEmbedsBindError::MissingInputEmbeds)
        );
    }

    #[test]
    fn qwen25_vl_visual_embeds_prefill_binder_splices_then_delegates() {
        let dir = write_fixture("visual_embeds_prefill_bind_success", IMAGE_ID, true);
        let tokens = [
            0,
            VISION_START_ID,
            VIDEO_ID,
            1,
            VISION_START_ID,
            IMAGE_ID,
            2,
        ];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(1, 2, 2)];
        let input = model_input_for_tokens(&dir, &tokens, &image_grids, &video_grids);
        let cfg = tiny_qwen25_text_cfg();
        let graph = poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(cfg, tokens.len(), 8);
        graph.validate().unwrap();
        let weights = weights_for_graph(&graph);
        let text_embeds = HostTensor::f32(
            vec![tokens.len(), cfg.hidden],
            fill(tokens.len() * cfg.hidden, 41),
        );
        let visual_embeds = HostTensor::f32(vec![2, cfg.hidden], fill(2 * cfg.hidden, 99));

        let bound = bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
            &graph,
            &weights,
            &input,
            &text_embeds,
            &visual_embeds,
        )
        .unwrap();

        let mut spliced = text_embeds.as_f32().unwrap().to_vec();
        spliced[2 * cfg.hidden..3 * cfg.hidden]
            .copy_from_slice(&visual_embeds.as_f32().unwrap()[0..cfg.hidden]);
        spliced[5 * cfg.hidden..6 * cfg.hidden]
            .copy_from_slice(&visual_embeds.as_f32().unwrap()[cfg.hidden..2 * cfg.hidden]);
        let spliced = HostTensor::f32(vec![tokens.len(), cfg.hidden], spliced);
        let direct =
            bind_qwen2_5_vl_prefill_kv_embeds_inputs(&graph, &weights, &input, &spliced).unwrap();

        assert_eq!(bound, direct);
        let (logits, state) = crate::core::cpu_oracle::cpu_eval_with_state(&graph, &bound).unwrap();
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        assert_eq!(state.len(), 2 * cfg.layers);
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        for tensor in state {
            assert!(
                tensor
                    .as_f32()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn qwen25_vl_visual_embeds_prefill_binder_reports_typed_errors() {
        let dir = write_fixture("visual_embeds_prefill_bind_errors", IMAGE_ID, true);
        let tokens = [0, VISION_START_ID, IMAGE_ID, 1];
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let input = model_input_for_tokens(&dir, &tokens, &image_grids, &[]);
        let cfg = tiny_qwen25_text_cfg();
        let graph = poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(cfg, tokens.len(), 5);
        let weights = weights_for_graph(&graph);
        let text_embeds = HostTensor::f32(
            vec![tokens.len(), cfg.hidden],
            fill(tokens.len() * cfg.hidden, 11),
        );
        let visual_embeds = HostTensor::f32(vec![1, cfg.hidden], fill(cfg.hidden, 12));

        let mut rank_one_graph = graph.clone();
        let input_embed_id = rank_one_graph
            .inputs
            .iter()
            .copied()
            .find(|&id| {
                rank_one_graph.meta(id).name.as_deref() == Some("activation.vlm.input_embeds")
            })
            .unwrap();
        rank_one_graph.values[input_embed_id].aval.shape = vec![tokens.len() * cfg.hidden];
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &rank_one_graph,
                &weights,
                &input,
                &text_embeds,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::GraphInputEmbedsRank {
                actual: vec![tokens.len() * cfg.hidden],
            })
        );

        let length_mismatch_graph =
            poot_models::qwen2::trace_qwen2_5_vl_prefill_kv_embeds(cfg, tokens.len() + 1, 5);
        let length_mismatch_weights = weights_for_graph(&length_mismatch_graph);
        let length_mismatch_text = HostTensor::f32(
            vec![tokens.len() + 1, cfg.hidden],
            fill((tokens.len() + 1) * cfg.hidden, 19),
        );
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &length_mismatch_graph,
                &length_mismatch_weights,
                &input,
                &length_mismatch_text,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::GraphPromptLength {
                expected: tokens.len(),
                actual: tokens.len() + 1,
            })
        );

        let wrong_text_shape = HostTensor::f32(
            vec![tokens.len() + 1, cfg.hidden],
            vec![0.0; (tokens.len() + 1) * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &wrong_text_shape,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::TextEmbedsShape {
                expected: vec![tokens.len(), cfg.hidden],
                actual: vec![tokens.len() + 1, cfg.hidden],
            })
        );

        let wrong_visual_width =
            HostTensor::f32(vec![1, cfg.hidden + 1], vec![0.0; cfg.hidden + 1]);
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &text_embeds,
                &wrong_visual_width,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::VisualEmbedsShape {
                expected_hidden: cfg.hidden,
                actual: vec![1, cfg.hidden + 1],
            })
        );

        let extra_visual_row = HostTensor::f32(vec![2, cfg.hidden], vec![0.0; 2 * cfg.hidden]);
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &text_embeds,
                &extra_visual_row,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::SpliceMap(
                Qwen25VlVisualSpliceMapError::VisualRowCountMismatch {
                    expected: 1,
                    actual: 2,
                }
            ))
        );

        let integer_text = HostTensor::i32(
            vec![tokens.len(), cfg.hidden],
            vec![0; tokens.len() * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &integer_text,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::TextEmbedsDtype { actual: DType::I32 })
        );

        let integer_visual = HostTensor::i32(vec![1, cfg.hidden], vec![0; cfg.hidden]);
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &text_embeds,
                &integer_visual,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::VisualEmbedsDtype { actual: DType::I32 })
        );

        // The splice graph is traced in f32 (`vlm.input_embeds` is an f32 slot), so a 16-bit float
        // tensor is refused by dtype rather than silently widened.
        let bf16_text = HostTensor::bf16(
            vec![tokens.len(), cfg.hidden],
            vec![0; tokens.len() * cfg.hidden],
        );
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &bf16_text,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::TextEmbedsDtype {
                actual: DType::BF16,
            })
        );

        let f16_visual = HostTensor::f16(vec![1, cfg.hidden], vec![0; cfg.hidden]);
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &graph,
                &weights,
                &input,
                &text_embeds,
                &f16_visual,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::VisualEmbedsDtype { actual: DType::F16 })
        );

        let ordinary_graph = poot_models::qwen2::trace_prefill_kv_embeds(cfg, tokens.len(), 5);
        let ordinary_weights = weights_for_graph(&ordinary_graph);
        assert_eq!(
            bind_qwen2_5_vl_visual_embeds_prefill_kv_inputs(
                &ordinary_graph,
                &ordinary_weights,
                &input,
                &text_embeds,
                &visual_embeds,
            ),
            Err(Qwen25VlVisualEmbedsPrefillBindError::Prefill(
                Qwen25VlPrefillKvEmbedsBindError::Mrope(MropeBindError::MissingSlot)
            ))
        );
    }

    #[test]
    fn qwen_vl_production_model_input_reports_metadata_path_before_prompt_errors() {
        let dir = write_fixture("model_input_metadata_first", IMAGE_ID, true);
        std::fs::write(dir.join("config.json"), "{").unwrap();
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let err =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &[VISION_START_ID],
                image_grids: &[MropeGrid::new(0, 1, 1)],
                video_grids: &[],
                spatial_merge_size: 0,
                temporal_patch_size,
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            })
            .unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlMropeModelInputError::Metadata(QwenVlMropeMetadataLoadError::Config(
                MropeConfigMetadataError::Load(
                    poot_load::qwen_vl::QwenVlHfMetadataError::ParseConfig { ref path, .. }
                )
            )) if path == &dir.join("config.json")
        ));
    }

    #[test]
    fn qwen_vl_production_model_input_preserves_pure_span_before_fps_ordering() {
        let dir = write_fixture("model_input_span_first", IMAGE_ID, true);
        let temporal_patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let err =
            assemble_qwen2_5_vl_mrope_model_input_from_files(Qwen25VlMropeModelInputRequest {
                model_dir: &dir,
                prompt_token_ids: &[VISION_START_ID],
                image_grids: &[],
                video_grids: &[],
                spatial_merge_size: 2,
                temporal_patch_size,
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            })
            .unwrap_err();

        assert!(matches!(
            err,
            Qwen25VlMropeModelInputError::PositionAssembly(MropePositionAssemblyError::Span(
                MropeSpanError::MissingVisualMarker {
                    start_index: 0,
                    index: 1
                }
            ))
        ));
    }

    #[test]
    fn qwen_vl_filesystem_adapter_returns_existing_unknown_tokenizer_error() {
        let dir = write_fixture("unknown_tokenizer_marker", IMAGE_ID, false);
        let err = load_qwen_vl_mrope_metadata(&dir).unwrap_err();
        assert!(matches!(
            err,
            QwenVlMropeMetadataLoadError::SpecialTokens(
                MropeSpecialTokenReconcileError::UnknownTokenizerSpecialToken {
                    role: MropeSpecialTokenKind::Image,
                    ..
                }
            )
        ));
    }

    #[test]
    fn qwen_vl_filesystem_adapter_returns_existing_config_mismatch_error() {
        let dir = write_fixture("config_mismatch", 7, true);
        let err = load_qwen_vl_mrope_metadata(&dir).unwrap_err();
        assert!(matches!(
            err,
            QwenVlMropeMetadataLoadError::SpecialTokens(
                MropeSpecialTokenReconcileError::ConfigSpecialTokenMismatch {
                    role: MropeSpecialTokenKind::Image,
                    tokenizer_id: 7,
                    config_id: IMAGE_ID,
                    ..
                }
            )
        ));
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
}
