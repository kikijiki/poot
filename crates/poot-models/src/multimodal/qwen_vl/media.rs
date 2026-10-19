//! The Qwen2.5-VL processor's media output as the mRoPE assembly and the tower binding read it.

use poot_tensor::{DType, HostTensor};

use crate::multimodal::mrope::MropeGrid;

/// Which kind of visual media a processor tensor or grid list belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen25VlProcessorMediaKind {
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
pub(crate) struct Qwen25VlProcessorMedia<'a> {
    /// The images' patches, `[rows, feature_width]`, concatenated in prompt order; `None` when there is no image.
    pub pixel_values: Option<&'a HostTensor>,
    /// One patch grid per image, in the order of the rows of `pixel_values`.
    pub image_grids: &'a [MropeGrid],
    /// The videos' patches, `[rows, feature_width]`, concatenated in prompt order; `None` when there is no video.
    pub pixel_values_videos: Option<&'a HostTensor>,
    /// One patch grid per video, in the order of the rows of `pixel_values_videos`.
    pub video_grids: &'a [MropeGrid],
}

/// Why processed media are not a consistent set of pixel tensors and grids.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum Qwen25VlProcessorMediaError {
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

impl Qwen25VlProcessorMedia<'_> {
    /// Validate the cached Transformers processor's flattened-patch row contract.
    ///
    /// Prompt placeholder counts are not checked here; the prompt's pad expansion and span discovery
    /// validate them against the grids.
    pub(crate) fn validate(&self) -> Result<(), Qwen25VlProcessorMediaError> {
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
