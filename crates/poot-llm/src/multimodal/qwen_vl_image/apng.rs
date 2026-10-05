//! APNG probe and selected-frame source for Qwen2.5-VL.
//!
//! Held for Card 566b (the VLM front end on the driver): `Qwen25VlApngFrameSource` is the only production
//! `Qwen25VlDecodedFrameSource` implementation, so the raw-video preprocessing path needs it, and no
//! production caller constructs it yet. Where rustc reports one as dead it carries
//! `#[expect(dead_code, reason = "held for POOT-739 (formerly 566b)")]`.

use std::io::Cursor;

use image::{AnimationDecoder, Frames, ImageDecoder, Limits, codecs::png::PngDecoder};

use super::preprocess::{
    Qwen25VlDecodedImage, Qwen25VlImageDecodeError, validate_decoded_dimensions,
};
use super::video_sampling::Qwen25VlVideoMetadata;
use super::{MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES};

pub(super) const MAX_ENCODED_APNG_BYTES: usize = 64 * 1024 * 1024;

const MAX_APNG_DECODER_ALLOC_BYTES: u64 = 512 * 1024 * 1024;

/// Validated in-memory APNG metadata and the encoded bytes used to decode it.
///
/// Metadata describes one animation cycle. Timestamps keep variable frame delays; the scalar source
/// FPS is the average `frame_count / duration`.
#[derive(Debug, PartialEq)]
pub(crate) struct Qwen25VlApng<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) duration_seconds: f64,
    pub(crate) source_frames_per_second: f64,
    pub(crate) frame_count: usize,
    pub(crate) frame_timestamps_seconds: Vec<f64>,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
impl Qwen25VlApng<'_> {
    pub(crate) fn metadata(&self) -> Qwen25VlVideoMetadata<'_> {
        Qwen25VlVideoMetadata {
            duration_seconds: self.duration_seconds,
            source_frames_per_second: self.source_frames_per_second,
            frame_count: self.frame_count,
            frame_timestamps_seconds: &self.frame_timestamps_seconds,
        }
    }

    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    pub(crate) const fn height(&self) -> usize {
        self.height
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
impl<'a> Qwen25VlApng<'a> {
    /// Probe and validate one complete APNG animation from borrowed in-memory bytes.
    pub(crate) fn from_bytes(bytes: &'a [u8]) -> Result<Self, Qwen25VlApngError> {
        validate_qwen25_vl_apng_encoded_len(bytes.len())?;
        let (width, height, frames) = qwen25_vl_apng_frames(bytes)?;
        let canvas_pixels = width
            .checked_mul(height)
            .ok_or(Qwen25VlApngError::DecodedPixelCountOverflow)?;

        let mut frame_timestamps_seconds = Vec::new();
        let mut duration_seconds = 0.0f64;
        let mut frame_count = 0usize;
        for frame in frames {
            let frame = frame.map_err(|source| Qwen25VlApngError::FrameDecode {
                frame_index: frame_count,
                message: source.to_string(),
            })?;
            frame_count = frame_count
                .checked_add(1)
                .ok_or(Qwen25VlApngError::FrameCountOverflow)?;
            validate_qwen25_vl_apng_decoded_work(frame_count, canvas_pixels)?;
            let frame_index = frame_count - 1;
            let decoded_width = frame.buffer().width() as usize;
            let decoded_height = frame.buffer().height() as usize;
            if (decoded_width, decoded_height) != (width, height) {
                return Err(Qwen25VlApngError::FrameDimensionMismatch {
                    frame_index,
                    expected_height: height,
                    expected_width: width,
                    actual_height: decoded_height,
                    actual_width: decoded_width,
                });
            }

            let (delay_numerator_ms, delay_denominator) = frame.delay().numer_denom_ms();
            if delay_numerator_ms == 0 {
                return Err(Qwen25VlApngError::ZeroFrameDelay { frame_index });
            }
            if delay_denominator == 0 {
                return Err(Qwen25VlApngError::TimingArithmetic { frame_index });
            }
            frame_timestamps_seconds
                .try_reserve(1)
                .map_err(|_| Qwen25VlApngError::Allocation {
                    stage: "frame timestamps",
                })?;
            frame_timestamps_seconds.push(duration_seconds);
            let delay_seconds =
                (f64::from(delay_numerator_ms) / f64::from(delay_denominator)) / 1_000.0;
            if !delay_seconds.is_finite() || delay_seconds <= 0.0 {
                return Err(Qwen25VlApngError::TimingArithmetic { frame_index });
            }
            let next_duration = duration_seconds + delay_seconds;
            if !next_duration.is_finite() || next_duration <= duration_seconds {
                return Err(Qwen25VlApngError::TimingArithmetic { frame_index });
            }
            duration_seconds = next_duration;
        }
        if frame_count == 0 {
            return Err(Qwen25VlApngError::EmptyAnimation);
        }

        let source_frames_per_second = frame_count as f64 / duration_seconds;
        if !source_frames_per_second.is_finite() || source_frames_per_second <= 0.0 {
            return Err(Qwen25VlApngError::SourceFramesPerSecondArithmetic);
        }
        Ok(Self {
            bytes,
            width,
            height,
            duration_seconds,
            source_frames_per_second,
            frame_count,
            frame_timestamps_seconds,
        })
    }

    /// Create a fresh one-pass source for strictly increasing selected frame indices.
    pub(crate) fn frame_source(&self) -> Result<Qwen25VlApngFrameSource<'a>, Qwen25VlApngError> {
        let (width, height, frames) = qwen25_vl_apng_frames(self.bytes)?;
        if (width, height) != (self.width, self.height) {
            return Err(Qwen25VlApngError::SourceDimensionMismatch {
                metadata_height: self.height,
                metadata_width: self.width,
                decoded_height: height,
                decoded_width: width,
            });
        }
        Ok(Qwen25VlApngFrameSource {
            frames,
            width,
            height,
            frame_count: self.frame_count,
            next_frame_index: 0,
            previous_requested_index: None,
            poisoned: false,
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlApngError {
    #[error("Qwen2.5-VL APNG has {actual} encoded bytes, exceeding the {limit}-byte cap")]
    TooManyEncodedBytes { actual: usize, limit: usize },
    #[error("Qwen2.5-VL PNG is static; an acTL animation control chunk is required")]
    StaticPng,
    #[error("Qwen2.5-VL APNG animation iterator returned no frames")]
    EmptyAnimation,
    #[error("Qwen2.5-VL APNG frame count overflows usize")]
    FrameCountOverflow,
    #[error("Qwen2.5-VL APNG decoded {actual} frames, exceeding the {limit}-frame cap")]
    TooManyFrames { actual: usize, limit: usize },
    #[error("Qwen2.5-VL APNG decoded-pixel count overflows usize")]
    DecodedPixelCountOverflow,
    #[error(
        "Qwen2.5-VL APNG decoded {pixels} canvas-frame pixels, exceeding the {limit}-pixel cap"
    )]
    TooManyDecodedPixels { pixels: usize, limit: usize },
    #[error(transparent)]
    Image(#[from] Qwen25VlImageDecodeError),
    #[error("initialize Qwen2.5-VL APNG {stage}: {message}")]
    Decoder {
        stage: &'static str,
        message: String,
    },
    #[error(
        "Qwen2.5-VL APNG source dimensions changed from metadata {metadata_height}x{metadata_width} to decoder {decoded_height}x{decoded_width}"
    )]
    SourceDimensionMismatch {
        metadata_height: usize,
        metadata_width: usize,
        decoded_height: usize,
        decoded_width: usize,
    },
    #[error("decode Qwen2.5-VL APNG frame {frame_index}: {message}")]
    FrameDecode { frame_index: usize, message: String },
    #[error(
        "Qwen2.5-VL APNG frame {frame_index} is {actual_height}x{actual_width}; canvas is {expected_height}x{expected_width}"
    )]
    FrameDimensionMismatch {
        frame_index: usize,
        expected_height: usize,
        expected_width: usize,
        actual_height: usize,
        actual_width: usize,
    },
    #[error(
        "Qwen2.5-VL APNG frame {frame_index} has {actual} RGBA bytes; dimensions require {expected}"
    )]
    FramePixelDataLen {
        frame_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("Qwen2.5-VL APNG frame {frame_index} has a zero delay numerator")]
    ZeroFrameDelay { frame_index: usize },
    #[error("Qwen2.5-VL APNG timing arithmetic failed at frame {frame_index}")]
    TimingArithmetic { frame_index: usize },
    #[error("Qwen2.5-VL APNG average source-FPS arithmetic failed")]
    SourceFramesPerSecondArithmetic,
    #[error("Qwen2.5-VL APNG could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
    #[error("Qwen2.5-VL APNG frame index {requested} is outside {frame_count} frames")]
    FrameIndexOutOfRange {
        requested: usize,
        frame_count: usize,
    },
    #[error("Qwen2.5-VL APNG frame index {requested} is not later than prior request {previous}")]
    FrameIndexNotIncreasing { previous: usize, requested: usize },
    #[error("Qwen2.5-VL APNG ended after {decoded} frames before requested frame {requested}")]
    UnexpectedFrameEnd { requested: usize, decoded: usize },
    #[error("Qwen2.5-VL APNG frame source is poisoned by an earlier failure")]
    PoisonedFrameSource,
}

/// A one-pass APNG implementation of the established selected-frame source contract.
pub(crate) struct Qwen25VlApngFrameSource<'a> {
    pub(crate) frames: Frames<'a>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) frame_count: usize,
    pub(crate) next_frame_index: usize,
    pub(crate) previous_requested_index: Option<usize>,
    pub(crate) poisoned: bool,
}

pub(super) fn validate_qwen25_vl_apng_encoded_len(actual: usize) -> Result<(), Qwen25VlApngError> {
    if actual > MAX_ENCODED_APNG_BYTES {
        return Err(Qwen25VlApngError::TooManyEncodedBytes {
            actual,
            limit: MAX_ENCODED_APNG_BYTES,
        });
    }
    Ok(())
}

pub(super) fn validate_qwen25_vl_apng_decoded_work(
    frame_count: usize,
    canvas_pixels: usize,
) -> Result<(), Qwen25VlApngError> {
    if frame_count > MAX_VIDEO_FRAMES {
        return Err(Qwen25VlApngError::TooManyFrames {
            actual: frame_count,
            limit: MAX_VIDEO_FRAMES,
        });
    }
    let decoded_pixels = frame_count
        .checked_mul(canvas_pixels)
        .ok_or(Qwen25VlApngError::DecodedPixelCountOverflow)?;
    if decoded_pixels > MAX_IMAGE_PIXELS {
        return Err(Qwen25VlApngError::TooManyDecodedPixels {
            pixels: decoded_pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    Ok(())
}

fn qwen25_vl_apng_frames<'a>(
    bytes: &'a [u8],
) -> Result<(usize, usize, Frames<'a>), Qwen25VlApngError> {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_PIXELS as u32);
    limits.max_image_height = Some(MAX_IMAGE_PIXELS as u32);
    limits.max_alloc = Some(MAX_APNG_DECODER_ALLOC_BYTES);
    let decoder = PngDecoder::with_limits(Cursor::new(bytes), limits).map_err(|source| {
        Qwen25VlApngError::Decoder {
            stage: "PNG decoder",
            message: source.to_string(),
        }
    })?;
    if !decoder
        .is_apng()
        .map_err(|source| Qwen25VlApngError::Decoder {
            stage: "animation check",
            message: source.to_string(),
        })?
    {
        return Err(Qwen25VlApngError::StaticPng);
    }
    let (width, height) = decoder.dimensions();
    validate_decoded_dimensions(height as usize, width as usize)?;
    let frames = decoder
        .apng()
        .map_err(|source| Qwen25VlApngError::Decoder {
            stage: "animation decoder",
            message: source.to_string(),
        })?
        .into_frames();
    Ok((width as usize, height as usize, frames))
}

impl Qwen25VlApngFrameSource<'_> {
    fn fail<T>(&mut self, error: Qwen25VlApngError) -> Result<T, Qwen25VlApngError> {
        self.poisoned = true;
        Err(error)
    }
}

impl Qwen25VlDecodedFrameSource for Qwen25VlApngFrameSource<'_> {
    type Error = Qwen25VlApngError;

    fn decode_frame(&mut self, frame_index: usize) -> Result<Qwen25VlDecodedImage, Self::Error> {
        if self.poisoned {
            return Err(Qwen25VlApngError::PoisonedFrameSource);
        }
        if frame_index >= self.frame_count {
            return self.fail(Qwen25VlApngError::FrameIndexOutOfRange {
                requested: frame_index,
                frame_count: self.frame_count,
            });
        }
        if let Some(previous) = self.previous_requested_index
            && frame_index <= previous
        {
            return self.fail(Qwen25VlApngError::FrameIndexNotIncreasing {
                previous,
                requested: frame_index,
            });
        }

        loop {
            let source_index = self.next_frame_index;
            let frame = match self.frames.next() {
                Some(Ok(frame)) => frame,
                Some(Err(source)) => {
                    return self.fail(Qwen25VlApngError::FrameDecode {
                        frame_index: source_index,
                        message: source.to_string(),
                    });
                }
                None => {
                    return self.fail(Qwen25VlApngError::UnexpectedFrameEnd {
                        requested: frame_index,
                        decoded: source_index,
                    });
                }
            };
            self.next_frame_index = match self.next_frame_index.checked_add(1) {
                Some(next) => next,
                None => return self.fail(Qwen25VlApngError::FrameCountOverflow),
            };
            let actual_width = frame.buffer().width() as usize;
            let actual_height = frame.buffer().height() as usize;
            if (actual_width, actual_height) != (self.width, self.height) {
                return self.fail(Qwen25VlApngError::FrameDimensionMismatch {
                    frame_index: source_index,
                    expected_height: self.height,
                    expected_width: self.width,
                    actual_height,
                    actual_width,
                });
            }
            if source_index == frame_index {
                let decoded = match composited_apng_frame_to_rgb8(frame, source_index) {
                    Ok(decoded) => decoded,
                    Err(error) => return self.fail(error),
                };
                self.previous_requested_index = Some(frame_index);
                return Ok(decoded);
            }
        }
    }
}

fn composited_apng_frame_to_rgb8(
    frame: image::Frame,
    frame_index: usize,
) -> Result<Qwen25VlDecodedImage, Qwen25VlApngError> {
    let rgba = frame.into_buffer();
    let height = rgba.height() as usize;
    let width = rgba.width() as usize;
    let expected_rgb_bytes = validate_decoded_dimensions(height, width)?;
    let expected_rgba_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Qwen25VlApngError::DecodedPixelCountOverflow)?;
    let rgba = rgba.into_raw();
    if rgba.len() != expected_rgba_bytes {
        return Err(Qwen25VlApngError::FramePixelDataLen {
            frame_index,
            expected: expected_rgba_bytes,
            actual: rgba.len(),
        });
    }
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(expected_rgb_bytes)
        .map_err(|_| Qwen25VlApngError::Allocation {
            stage: "selected RGB frame",
        })?;
    for pixel in rgba.chunks_exact(4) {
        // `image` has already applied APNG disposal and alpha blending to this RGBA canvas; the
        // decoded-image boundary is RGB8, so alpha is dropped.
        rgb.extend_from_slice(&pixel[..3]);
    }
    Ok(Qwen25VlDecodedImage { height, width, rgb })
}

/// Caller-owned codec/container boundary used to decode only selected source frames.
pub trait Qwen25VlDecodedFrameSource {
    type Error: std::error::Error + Send + Sync + 'static;

    fn decode_frame(&mut self, frame_index: usize) -> Result<Qwen25VlDecodedImage, Self::Error>;
}
