//! Pure loading of the mRoPE metadata stored in Qwen2-VL and Qwen2.5-VL HF `config.json` files.
//!
//! This is a validated view over [`crate::Qwen2HfConfig`], the repository's existing HF config schema.
//! Released VL configs add their marker IDs and vision metadata at the top level. A closed second serde
//! projection reads only those fields after the shared parse succeeds; every ordinary Qwen2/Qwen3 conversion
//! and Runner path remains unchanged.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenVlModelType {
    Qwen2Vl,
    Qwen25Vl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenVlSpecialTokenRole {
    VisionStart,
    ImagePad,
    VideoPad,
}

impl std::fmt::Display for QwenVlSpecialTokenRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VisionStart => f.write_str("vision_start"),
            Self::ImagePad => f.write_str("image_pad"),
            Self::VideoPad => f.write_str("video_pad"),
        }
    }
}

/// Loader-owned, representation-checked values from one Qwen-VL HF `config.json`.
///
/// Model-domain validation remains in `poot-models`: this type has not yet constructed
/// `MropeSpecialTokenIds` or `MropeTemporalTokensPerSecond`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenVlHfMetadata {
    model_type: QwenVlModelType,
    mrope_section: [usize; 3],
    vision_start_token_id: u32,
    image_token_id: u32,
    video_token_id: u32,
    temporal_tokens_per_second: Option<f64>,
}

/// One authoritative `config.json` snapshot for callers that need both the ordinary text config and the
/// Qwen-VL metadata projection.
#[derive(Clone, Debug)]
pub struct QwenVlHfConfig {
    pub hf: crate::Qwen2HfConfig,
    pub metadata: QwenVlHfMetadata,
}

impl QwenVlHfConfig {
    pub fn from_slice(json: &[u8]) -> Result<Self, QwenVlHfMetadataError> {
        let hf: crate::Qwen2HfConfig = serde_json::from_slice(json)?;
        let metadata = QwenVlHfMetadata::from_parsed_qwen2_config_and_slice(&hf, json)?;
        Ok(Self { hf, metadata })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, QwenVlHfMetadataError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| QwenVlHfMetadataError::ReadConfig {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_slice(&bytes).map_err(|err| match err {
            QwenVlHfMetadataError::Json(source) => QwenVlHfMetadataError::ParseConfig {
                path: path.to_path_buf(),
                source,
            },
            err => err,
        })
    }
}

impl QwenVlHfMetadata {
    pub fn from_slice(json: &[u8]) -> Result<Self, QwenVlHfMetadataError> {
        Ok(QwenVlHfConfig::from_slice(json)?.metadata)
    }

    pub fn from_parsed_qwen2_config_and_slice(
        _hf: &crate::Qwen2HfConfig,
        json: &[u8],
    ) -> Result<Self, QwenVlHfMetadataError> {
        let metadata: QwenVlConfigJson = serde_json::from_slice(json)?;
        metadata.try_into()
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, QwenVlHfMetadataError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| QwenVlHfMetadataError::ReadConfig {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_slice(&bytes).map_err(|err| match err {
            QwenVlHfMetadataError::Json(source) => QwenVlHfMetadataError::ParseConfig {
                path: path.to_path_buf(),
                source,
            },
            err => err,
        })
    }

    pub const fn model_type(&self) -> QwenVlModelType {
        self.model_type
    }

    pub const fn mrope_section(&self) -> [usize; 3] {
        self.mrope_section
    }

    pub const fn vision_start_token_id(&self) -> u32 {
        self.vision_start_token_id
    }

    pub const fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    pub const fn video_token_id(&self) -> u32 {
        self.video_token_id
    }

    pub const fn temporal_tokens_per_second(&self) -> Option<f64> {
        self.temporal_tokens_per_second
    }
}

/// Processor/template-owned Qwen-VL marker strings read from on-disk tokenizer metadata.
///
/// These are still strings, not IDs. The live tokenizer remains authoritative for string-to-ID mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenVlProcessorSpecialTokenStrings {
    vision_start: String,
    image_pad: String,
    video_pad: String,
}

impl QwenVlProcessorSpecialTokenStrings {
    pub fn from_tokenizer_config_slice(
        json: &[u8],
        config: &QwenVlHfMetadata,
    ) -> Result<Self, QwenVlProcessorMetadataError> {
        let raw: QwenVlTokenizerConfigJson = serde_json::from_slice(json)?;
        Self::from_tokenizer_config(raw, config)
    }

    pub fn load_from_dir(
        dir: impl AsRef<Path>,
        config: &QwenVlHfMetadata,
    ) -> Result<Self, QwenVlProcessorMetadataError> {
        let path = dir.as_ref().join("tokenizer_config.json");
        let bytes = std::fs::read(&path).map_err(|source| {
            QwenVlProcessorMetadataError::ReadTokenizerConfig {
                path: path.clone(),
                source,
            }
        })?;
        Self::from_tokenizer_config_slice(&bytes, config).map_err(|err| match err {
            QwenVlProcessorMetadataError::Json(source) => {
                QwenVlProcessorMetadataError::ParseTokenizerConfig { path, source }
            }
            err => err,
        })
    }

    pub fn token(&self, role: QwenVlSpecialTokenRole) -> &str {
        match role {
            QwenVlSpecialTokenRole::VisionStart => &self.vision_start,
            QwenVlSpecialTokenRole::ImagePad => &self.image_pad,
            QwenVlSpecialTokenRole::VideoPad => &self.video_pad,
        }
    }

    fn from_tokenizer_config(
        raw: QwenVlTokenizerConfigJson,
        config: &QwenVlHfMetadata,
    ) -> Result<Self, QwenVlProcessorMetadataError> {
        Ok(Self {
            vision_start: marker_string(
                QwenVlSpecialTokenRole::VisionStart,
                raw.vision_start_token,
                &raw.added_tokens_decoder,
                config.vision_start_token_id(),
            )?,
            image_pad: marker_string(
                QwenVlSpecialTokenRole::ImagePad,
                raw.image_token,
                &raw.added_tokens_decoder,
                config.image_token_id(),
            )?,
            video_pad: marker_string(
                QwenVlSpecialTokenRole::VideoPad,
                raw.video_token,
                &raw.added_tokens_decoder,
                config.video_token_id(),
            )?,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QwenVlProcessorMetadataError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("read tokenizer_config.json {path:?}: {source}")]
    ReadTokenizerConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parse tokenizer_config.json {path:?}: {source}")]
    ParseTokenizerConfig {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error(
        "Qwen-VL tokenizer_config.json has no processor marker for {role} at token id {token_id}"
    )]
    MissingProcessorMarker {
        role: QwenVlSpecialTokenRole,
        token_id: u32,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum QwenVlHfMetadataError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("read config.json {path:?}: {source}")]
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parse config.json {path:?}: {source}")]
    ParseConfig {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("unsupported Qwen-VL model_type {model_type:?}")]
    UnsupportedModelType { model_type: String },
    #[error("Qwen-VL config is missing required field {field}")]
    MissingField { field: &'static str },
    #[error("Qwen-VL rope_scaling.mrope_section must have length 3, got {actual}")]
    WrongMropeSectionLength { actual: usize },
    #[error("Qwen-VL rope_scaling.mrope_section[{index}] must be positive")]
    NonPositiveMropeSection { index: usize },
    #[error(
        "Qwen-VL rope_scaling.mrope_section[{index}] value {value} is outside the host usize range"
    )]
    MropeSectionOutOfRange { index: usize, value: u64 },
    #[error("Qwen-VL rope_scaling.mrope_section sum exceeds the host usize range")]
    MropeSectionSumOverflow,
    #[error("Qwen-VL {field} value {value} is outside the u32 token-id range")]
    SpecialTokenIdOutOfRange { field: &'static str, value: u64 },
}

#[derive(serde::Deserialize)]
struct QwenVlConfigJson {
    model_type: String,
    #[serde(default)]
    vision_start_token_id: Option<u64>,
    #[serde(default)]
    image_token_id: Option<u64>,
    #[serde(default)]
    video_token_id: Option<u64>,
    #[serde(default)]
    rope_scaling: Option<QwenVlRopeScalingJson>,
    #[serde(default)]
    vision_config: Option<QwenVlVisionConfigJson>,
}

#[derive(serde::Deserialize)]
struct QwenVlRopeScalingJson {
    #[serde(default)]
    mrope_section: Option<Vec<u64>>,
}

#[derive(serde::Deserialize)]
struct QwenVlVisionConfigJson {
    #[serde(default)]
    tokens_per_second: Option<f64>,
}

#[derive(serde::Deserialize)]
struct QwenVlTokenizerConfigJson {
    #[serde(default)]
    vision_start_token: Option<QwenVlTokenStringJson>,
    #[serde(default)]
    image_token: Option<QwenVlTokenStringJson>,
    #[serde(default)]
    video_token: Option<QwenVlTokenStringJson>,
    #[serde(default)]
    added_tokens_decoder: Option<std::collections::BTreeMap<String, QwenVlAddedTokenDecoderJson>>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum QwenVlTokenStringJson {
    String(String),
    Object { content: String },
}

impl QwenVlTokenStringJson {
    fn into_string(self) -> String {
        match self {
            Self::String(token) => token,
            Self::Object { content } => content,
        }
    }
}

#[derive(serde::Deserialize)]
struct QwenVlAddedTokenDecoderJson {
    content: String,
}

fn marker_string(
    role: QwenVlSpecialTokenRole,
    explicit: Option<QwenVlTokenStringJson>,
    decoder: &Option<std::collections::BTreeMap<String, QwenVlAddedTokenDecoderJson>>,
    token_id: u32,
) -> Result<String, QwenVlProcessorMetadataError> {
    if let Some(token) = explicit {
        return Ok(token.into_string());
    }
    decoder
        .as_ref()
        .and_then(|records| records.get(&token_id.to_string()))
        .map(|record| record.content.clone())
        .ok_or(QwenVlProcessorMetadataError::MissingProcessorMarker { role, token_id })
}

impl TryFrom<QwenVlConfigJson> for QwenVlHfMetadata {
    type Error = QwenVlHfMetadataError;

    fn try_from(raw: QwenVlConfigJson) -> Result<Self, Self::Error> {
        let model_type = match raw.model_type.as_str() {
            "qwen2_vl" => QwenVlModelType::Qwen2Vl,
            "qwen2_5_vl" => QwenVlModelType::Qwen25Vl,
            _ => {
                return Err(QwenVlHfMetadataError::UnsupportedModelType {
                    model_type: raw.model_type,
                });
            }
        };

        let sections = raw.rope_scaling.and_then(|rope| rope.mrope_section).ok_or(
            QwenVlHfMetadataError::MissingField {
                field: "rope_scaling.mrope_section",
            },
        )?;
        if sections.len() != 3 {
            return Err(QwenVlHfMetadataError::WrongMropeSectionLength {
                actual: sections.len(),
            });
        }
        let mut mrope_section = [0usize; 3];
        let mut section_sum = 0usize;
        for (index, value) in sections.into_iter().enumerate() {
            if value == 0 {
                return Err(QwenVlHfMetadataError::NonPositiveMropeSection { index });
            }
            let width = usize::try_from(value)
                .map_err(|_| QwenVlHfMetadataError::MropeSectionOutOfRange { index, value })?;
            section_sum = section_sum
                .checked_add(width)
                .ok_or(QwenVlHfMetadataError::MropeSectionSumOverflow)?;
            mrope_section[index] = width;
        }

        let token_id = |field, value| {
            u32::try_from(value)
                .map_err(|_| QwenVlHfMetadataError::SpecialTokenIdOutOfRange { field, value })
        };
        let required_token_id =
            |field, value: Option<u64>| value.ok_or(QwenVlHfMetadataError::MissingField { field });
        let vision_start_token_id = token_id(
            "vision_start_token_id",
            required_token_id("vision_start_token_id", raw.vision_start_token_id)?,
        )?;
        let image_token_id = token_id(
            "image_token_id",
            required_token_id("image_token_id", raw.image_token_id)?,
        )?;
        let video_token_id = token_id(
            "video_token_id",
            required_token_id("video_token_id", raw.video_token_id)?,
        )?;

        let temporal_tokens_per_second = match model_type {
            QwenVlModelType::Qwen2Vl => None,
            QwenVlModelType::Qwen25Vl => raw
                .vision_config
                .and_then(|vision| vision.tokens_per_second),
        };

        Ok(Self {
            model_type,
            mrope_section,
            vision_start_token_id,
            image_token_id,
            video_token_id,
            temporal_tokens_per_second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2_VL: &str = r#"{
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "model_type": "qwen2_vl",
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "rope_scaling": {
            "type": "mrope",
            "mrope_section": [16, 24, 24]
        },
        "vision_config": {
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }"#;

    const QWEN2_5_VL: &str = r#"{
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
        "max_position_embeddings": 32768,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "image_token_id": 151655,
        "video_token_id": 151656,
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

    fn replace_once(source: &str, old: &str, new: &str) -> String {
        assert_eq!(
            source.matches(old).count(),
            1,
            "fixture replacement must be unique"
        );
        source.replacen(old, new, 1)
    }

    #[test]
    fn qwen_vl_config_metadata_parses_released_json_shapes() {
        let qwen2 = QwenVlHfMetadata::from_slice(QWEN2_VL.as_bytes()).unwrap();
        assert_eq!(qwen2.model_type(), QwenVlModelType::Qwen2Vl);
        assert_eq!(qwen2.mrope_section(), [16, 24, 24]);
        assert_eq!(qwen2.vision_start_token_id(), 151652);
        assert_eq!(qwen2.image_token_id(), 151655);
        assert_eq!(qwen2.video_token_id(), 151656);
        assert_eq!(qwen2.temporal_tokens_per_second(), None);

        let qwen25 = QwenVlHfMetadata::from_slice(QWEN2_5_VL.as_bytes()).unwrap();
        assert_eq!(qwen25.model_type(), QwenVlModelType::Qwen25Vl);
        assert_eq!(qwen25.mrope_section(), [16, 24, 24]);
        assert_eq!(qwen25.temporal_tokens_per_second(), Some(4.0));
    }

    #[test]
    fn qwen25_config_metadata_does_not_default_absent_temporal_rate() {
        let without_rate = replace_once(QWEN2_5_VL, ",\n            \"tokens_per_second\": 4", "");
        let metadata = QwenVlHfMetadata::from_slice(without_rate.as_bytes()).unwrap();
        assert_eq!(metadata.temporal_tokens_per_second(), None);
    }

    #[test]
    fn qwen_vl_config_metadata_rejects_missing_duplicate_and_malformed_fields() {
        let missing = replace_once(QWEN2_VL, "\n        \"image_token_id\": 151655,", "");
        assert!(matches!(
            QwenVlHfMetadata::from_slice(missing.as_bytes()),
            Err(QwenVlHfMetadataError::MissingField {
                field: "image_token_id"
            })
        ));

        let missing_section = replace_once(
            QWEN2_VL,
            "\"type\": \"mrope\",\n            \"mrope_section\": [16, 24, 24]",
            "\"type\": \"mrope\"",
        );
        assert!(matches!(
            QwenVlHfMetadata::from_slice(missing_section.as_bytes()),
            Err(QwenVlHfMetadataError::MissingField {
                field: "rope_scaling.mrope_section"
            })
        ));

        let duplicate = replace_once(
            QWEN2_VL,
            "\"video_token_id\": 151656,",
            "\"video_token_id\": 151656,\n        \"video_token_id\": 151657,",
        );
        assert!(matches!(
            QwenVlHfMetadata::from_slice(duplicate.as_bytes()),
            Err(QwenVlHfMetadataError::Json(_))
        ));

        let duplicate_nested = replace_once(
            QWEN2_VL,
            "\"mrope_section\": [16, 24, 24]",
            "\"mrope_section\": [16, 24, 24], \"mrope_section\": [8, 8, 8]",
        );
        assert!(matches!(
            QwenVlHfMetadata::from_slice(duplicate_nested.as_bytes()),
            Err(QwenVlHfMetadataError::Json(_))
        ));

        let malformed_type = replace_once(
            QWEN2_VL,
            "\"image_token_id\": 151655",
            "\"image_token_id\": []",
        );
        assert!(matches!(
            QwenVlHfMetadata::from_slice(malformed_type.as_bytes()),
            Err(QwenVlHfMetadataError::Json(_))
        ));
        assert!(matches!(
            QwenVlHfMetadata::from_slice(br#"{"model_type": "qwen2_vl"#),
            Err(QwenVlHfMetadataError::Json(_))
        ));
    }

    #[test]
    fn qwen_vl_config_metadata_rejects_model_section_and_token_ranges() {
        let unsupported = replace_once(QWEN2_VL, "\"qwen2_vl\"", "\"qwen2\"");
        assert!(matches!(
            QwenVlHfMetadata::from_slice(unsupported.as_bytes()),
            Err(QwenVlHfMetadataError::UnsupportedModelType { .. })
        ));

        for widths in ["[16, 24]", "[16, 24, 24, 8]"] {
            let json = replace_once(QWEN2_VL, "[16, 24, 24]", widths);
            assert!(matches!(
                QwenVlHfMetadata::from_slice(json.as_bytes()),
                Err(QwenVlHfMetadataError::WrongMropeSectionLength { .. })
            ));
        }
        let zero = replace_once(QWEN2_VL, "[16, 24, 24]", "[16, 0, 24]");
        assert!(matches!(
            QwenVlHfMetadata::from_slice(zero.as_bytes()),
            Err(QwenVlHfMetadataError::NonPositiveMropeSection { index: 1 })
        ));

        let sum_overflow = replace_once(QWEN2_VL, "[16, 24, 24]", "[18446744073709551615, 1, 1]");
        let overflow_error = QwenVlHfMetadata::from_slice(sum_overflow.as_bytes()).unwrap_err();
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            overflow_error,
            QwenVlHfMetadataError::MropeSectionSumOverflow
        ));
        #[cfg(not(target_pointer_width = "64"))]
        assert!(matches!(
            overflow_error,
            QwenVlHfMetadataError::MropeSectionOutOfRange { index: 0, .. }
        ));

        let token_overflow = replace_once(QWEN2_VL, "151655", "4294967296");
        assert!(matches!(
            QwenVlHfMetadata::from_slice(token_overflow.as_bytes()),
            Err(QwenVlHfMetadataError::SpecialTokenIdOutOfRange {
                field: "image_token_id",
                value: 4_294_967_296,
            })
        ));
    }

    const QWEN_VL_TOKENIZER_CONFIG: &str = r#"{
        "added_tokens_decoder": {
            "151652": {"content": "<|vision_start|>", "special": true},
            "151655": {"content": "<|image_pad|>", "special": true},
            "151656": {"content": "<|video_pad|>", "special": true}
        }
    }"#;

    #[test]
    fn qwen_vl_processor_special_tokens_read_tokenizer_config_decoder() {
        let config = QwenVlHfMetadata::from_slice(QWEN2_5_VL.as_bytes()).unwrap();
        let tokens = QwenVlProcessorSpecialTokenStrings::from_tokenizer_config_slice(
            QWEN_VL_TOKENIZER_CONFIG.as_bytes(),
            &config,
        )
        .unwrap();
        assert_eq!(
            tokens.token(QwenVlSpecialTokenRole::VisionStart),
            "<|vision_start|>"
        );
        assert_eq!(
            tokens.token(QwenVlSpecialTokenRole::ImagePad),
            "<|image_pad|>"
        );
        assert_eq!(
            tokens.token(QwenVlSpecialTokenRole::VideoPad),
            "<|video_pad|>"
        );
    }

    #[test]
    fn qwen_vl_processor_special_tokens_prefer_explicit_role_fields() {
        let config = QwenVlHfMetadata::from_slice(QWEN2_5_VL.as_bytes()).unwrap();
        let explicit = replace_once(
            QWEN_VL_TOKENIZER_CONFIG,
            "\"added_tokens_decoder\"",
            "\"image_token\": {\"content\": \"<custom_image>\"},\n        \"added_tokens_decoder\"",
        );
        let tokens = QwenVlProcessorSpecialTokenStrings::from_tokenizer_config_slice(
            explicit.as_bytes(),
            &config,
        )
        .unwrap();
        assert_eq!(
            tokens.token(QwenVlSpecialTokenRole::ImagePad),
            "<custom_image>"
        );
        assert_eq!(
            tokens.token(QwenVlSpecialTokenRole::VisionStart),
            "<|vision_start|>"
        );
    }

    #[test]
    fn qwen_vl_processor_special_tokens_reject_missing_role_metadata() {
        let config = QwenVlHfMetadata::from_slice(QWEN2_5_VL.as_bytes()).unwrap();
        let missing = replace_once(
            QWEN_VL_TOKENIZER_CONFIG,
            "\"151655\": {\"content\": \"<|image_pad|>\", \"special\": true},\n            ",
            "",
        );
        assert!(matches!(
            QwenVlProcessorSpecialTokenStrings::from_tokenizer_config_slice(
                missing.as_bytes(),
                &config
            ),
            Err(QwenVlProcessorMetadataError::MissingProcessorMarker {
                role: QwenVlSpecialTokenRole::ImagePad,
                token_id: 151655,
            })
        ));
    }
}
