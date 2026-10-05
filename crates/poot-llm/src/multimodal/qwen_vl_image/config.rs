use std::{fs, path::Path, path::PathBuf};

use serde::Deserialize;

pub(super) const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

const PIL_BICUBIC_RESAMPLE: u32 = 3;

/// Validated Qwen2.5-VL image processor configuration.
///
/// The defaults and persisted override names match Transformers 4.49.0's `Qwen2VLImageProcessor`.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlImageProcessorConfig {
    pub(super) do_resize: bool,
    pub(super) do_rescale: bool,
    pub(super) do_normalize: bool,
    pub(super) min_pixels: usize,
    pub(super) max_pixels: usize,
    pub(super) patch_size: usize,
    pub(super) temporal_patch_size: usize,
    pub(super) merge_size: usize,
    pub(super) rescale_factor: f64,
    pub(super) mean: [f32; 3],
    pub(super) std: [f32; 3],
}

impl Default for Qwen25VlImageProcessorConfig {
    #[allow(clippy::excessive_precision)] // Preserve the upstream decimal constants before f32 rounding.
    fn default() -> Self {
        Self {
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            rescale_factor: 1.0 / 255.0,
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
        }
    }
}

impl Qwen25VlImageProcessorConfig {
    pub fn do_resize(&self) -> bool {
        self.do_resize
    }

    pub fn do_rescale(&self) -> bool {
        self.do_rescale
    }

    pub fn do_normalize(&self) -> bool {
        self.do_normalize
    }

    pub fn min_pixels(&self) -> usize {
        self.min_pixels
    }

    pub fn max_pixels(&self) -> usize {
        self.max_pixels
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    pub fn merge_size(&self) -> usize {
        self.merge_size
    }

    pub fn rescale_factor(&self) -> f64 {
        self.rescale_factor
    }

    pub fn image_mean(&self) -> [f32; 3] {
        self.mean
    }

    pub fn image_std(&self) -> [f32; 3] {
        self.std
    }

    fn from_overrides(
        overrides: Qwen25VlImageProcessorOverrides,
    ) -> Result<Self, Qwen25VlImageProcessorConfigValidationError> {
        let defaults = Self::default();
        let config = Self {
            do_resize: overrides.do_resize.unwrap_or(defaults.do_resize),
            do_rescale: overrides.do_rescale.unwrap_or(defaults.do_rescale),
            do_normalize: overrides.do_normalize.unwrap_or(defaults.do_normalize),
            min_pixels: overrides.min_pixels.unwrap_or(defaults.min_pixels),
            max_pixels: overrides.max_pixels.unwrap_or(defaults.max_pixels),
            patch_size: overrides.patch_size.unwrap_or(defaults.patch_size),
            temporal_patch_size: overrides
                .temporal_patch_size
                .unwrap_or(defaults.temporal_patch_size),
            merge_size: overrides.merge_size.unwrap_or(defaults.merge_size),
            rescale_factor: overrides.rescale_factor.unwrap_or(defaults.rescale_factor),
            mean: resolve_channels(overrides.image_mean, defaults.mean, "image_mean")?,
            std: resolve_channels(overrides.image_std, defaults.std, "image_std")?,
        };
        config.validate(
            overrides.resample.unwrap_or(PIL_BICUBIC_RESAMPLE),
            overrides.do_convert_rgb.unwrap_or(true),
        )?;
        Ok(config)
    }

    fn validate(
        &self,
        resample: u32,
        do_convert_rgb: bool,
    ) -> Result<(), Qwen25VlImageProcessorConfigValidationError> {
        for (field, value) in [
            ("min_pixels", self.min_pixels),
            ("max_pixels", self.max_pixels),
            ("patch_size", self.patch_size),
            ("temporal_patch_size", self.temporal_patch_size),
            ("merge_size", self.merge_size),
        ] {
            if value == 0 {
                return Err(Qwen25VlImageProcessorConfigValidationError::ZeroValue { field });
            }
        }
        if self.min_pixels > self.max_pixels {
            return Err(
                Qwen25VlImageProcessorConfigValidationError::InvalidPixelRange {
                    min_pixels: self.min_pixels,
                    max_pixels: self.max_pixels,
                },
            );
        }
        let factor = self.patch_size.checked_mul(self.merge_size).ok_or(
            Qwen25VlImageProcessorConfigValidationError::ArithmeticOverflow {
                stage: "resize factor",
            },
        )?;
        let factor_area = factor.checked_mul(factor).ok_or(
            Qwen25VlImageProcessorConfigValidationError::ArithmeticOverflow {
                stage: "resize factor area",
            },
        )?;
        if self.max_pixels < factor_area {
            return Err(
                Qwen25VlImageProcessorConfigValidationError::MaxPixelsBelowFactorArea {
                    max_pixels: self.max_pixels,
                    factor,
                },
            );
        }
        if !self.rescale_factor.is_finite() {
            return Err(
                Qwen25VlImageProcessorConfigValidationError::NonFiniteValue {
                    field: "rescale_factor",
                },
            );
        }
        for (field, values) in [("image_mean", self.mean), ("image_std", self.std)] {
            for (index, value) in values.into_iter().enumerate() {
                if !value.is_finite() {
                    return Err(
                        Qwen25VlImageProcessorConfigValidationError::NonFiniteChannelValue {
                            field,
                            index,
                        },
                    );
                }
                if field == "image_std" && value == 0.0 {
                    return Err(Qwen25VlImageProcessorConfigValidationError::ZeroStd { index });
                }
            }
        }
        for channel in 0..3 {
            for raw in [0u8, u8::MAX] {
                let mut processed = if self.do_rescale {
                    (raw as f64 * self.rescale_factor) as f32
                } else {
                    raw as f32
                };
                if self.do_normalize {
                    processed = (processed - self.mean[channel]) / self.std[channel];
                }
                if !processed.is_finite() {
                    return Err(
                        Qwen25VlImageProcessorConfigValidationError::NonFiniteOutputRange {
                            channel,
                        },
                    );
                }
            }
        }
        if resample != PIL_BICUBIC_RESAMPLE {
            return Err(
                Qwen25VlImageProcessorConfigValidationError::UnsupportedResample {
                    actual: resample,
                },
            );
        }
        if !do_convert_rgb {
            return Err(Qwen25VlImageProcessorConfigValidationError::RgbConversionDisabled);
        }
        Ok(())
    }
}

#[derive(Debug, Default, Deserialize)]
struct Qwen25VlImageProcessorOverrides {
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) do_resize: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(crate) resample: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) do_rescale: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) rescale_factor: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) do_normalize: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(crate) image_mean: Option<ChannelValues>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(crate) image_std: Option<ChannelValues>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(crate) do_convert_rgb: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) min_pixels: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) max_pixels: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) patch_size: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) temporal_patch_size: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_present_value")]
    pub(super) merge_size: Option<usize>,
}

/// Distinguish an omitted override from a present JSON `null`. Only typed values or omission are
/// accepted; treating a malformed present field as the default would hide bad model metadata.
fn deserialize_present_value<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChannelValues {
    Scalar(f64),
    Channels(Vec<f64>),
}

fn resolve_channels(
    value: Option<ChannelValues>,
    default: [f32; 3],
    field: &'static str,
) -> Result<[f32; 3], Qwen25VlImageProcessorConfigValidationError> {
    let values = match value {
        None => return Ok(default),
        Some(ChannelValues::Scalar(value)) => [value; 3],
        Some(ChannelValues::Channels(values)) => {
            let actual = values.len();
            let values: [f64; 3] = values.try_into().map_err(|_| {
                Qwen25VlImageProcessorConfigValidationError::ChannelCount { field, actual }
            })?;
            values
        }
    };
    let mut resolved = [0.0; 3];
    for (index, value) in values.into_iter().enumerate() {
        let value = value as f32;
        if !value.is_finite() {
            return Err(
                Qwen25VlImageProcessorConfigValidationError::NonFiniteChannelValue { field, index },
            );
        }
        resolved[index] = value;
    }
    Ok(resolved)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlImageProcessorConfigValidationError {
    #[error("Qwen2.5-VL image processor {field} must be nonzero")]
    ZeroValue { field: &'static str },
    #[error("Qwen2.5-VL image processor min_pixels {min_pixels} exceeds max_pixels {max_pixels}")]
    InvalidPixelRange {
        min_pixels: usize,
        max_pixels: usize,
    },
    #[error("Qwen2.5-VL image processor arithmetic overflows usize while computing {stage}")]
    ArithmeticOverflow { stage: &'static str },
    #[error(
        "Qwen2.5-VL image processor max_pixels {max_pixels} is below resize factor {factor} squared"
    )]
    MaxPixelsBelowFactorArea { max_pixels: usize, factor: usize },
    #[error("Qwen2.5-VL image processor {field} must be finite")]
    NonFiniteValue { field: &'static str },
    #[error("Qwen2.5-VL image processor {field}[{index}] must be finite")]
    NonFiniteChannelValue { field: &'static str, index: usize },
    #[error("Qwen2.5-VL image processor image_std[{index}] must be nonzero")]
    ZeroStd { index: usize },
    #[error("Qwen2.5-VL image processor channel {channel} can produce non-finite RGB8 output")]
    NonFiniteOutputRange { channel: usize },
    #[error("Qwen2.5-VL image processor {field} has {actual} channels; expected one or three")]
    ChannelCount { field: &'static str, actual: usize },
    #[error(
        "Qwen2.5-VL image processor resample {actual} is unsupported; expected Pillow BICUBIC 3"
    )]
    UnsupportedResample { actual: u32 },
    #[error("Qwen2.5-VL image processor do_convert_rgb=false is unsupported for the RGB8 boundary")]
    RgbConversionDisabled,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlImageProcessorConfigError {
    #[error("read Qwen2.5-VL image processor config {path:?}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse Qwen2.5-VL image processor config {path:?}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("validate Qwen2.5-VL image processor config {path:?}: {source}")]
    Validate {
        path: PathBuf,
        #[source]
        source: Qwen25VlImageProcessorConfigValidationError,
    },
}

/// Load persisted Qwen2.5-VL image processor overrides from a local model directory.
pub fn load_qwen2_5_vl_image_processor_config(
    model_dir: &Path,
) -> Result<Qwen25VlImageProcessorConfig, Qwen25VlImageProcessorConfigError> {
    let path = model_dir.join(PREPROCESSOR_CONFIG_FILE);
    let bytes = fs::read(&path).map_err(|source| Qwen25VlImageProcessorConfigError::Read {
        path: path.clone(),
        source,
    })?;
    let overrides = serde_json::from_slice(&bytes).map_err(|source| {
        Qwen25VlImageProcessorConfigError::Parse {
            path: path.clone(),
            source,
        }
    })?;
    Qwen25VlImageProcessorConfig::from_overrides(overrides)
        .map_err(|source| Qwen25VlImageProcessorConfigError::Validate { path, source })
}
