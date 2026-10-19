//! APNG probe and selected-frame source: the container half of a video request.
//!
//! [`Apng`] probes an animated PNG for the facts frame sampling needs ([`VideoMetadata`]) and
//! [`ApngFrames`] is the one [`FrameSource`] over it: a front end asks only for the frames it selects
//! (`poot-caption` opens an animated PNG with it). Which frames to select, and what to do with them, is
//! the model family's.

use std::io::Cursor;

use image::{AnimationDecoder, Frames, ImageDecoder, Limits, codecs::png::PngDecoder};
use poot_models::multimodal::front_end::{FrameSource, Image, ImageError, VideoMetadata};

/// The most pixels the canvas frames of one animation may decode to together.
const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;

/// The most frames one animation may hold.
const MAX_VIDEO_FRAMES: usize = 16 * 1024;

const MAX_ENCODED_APNG_BYTES: usize = 64 * 1024 * 1024;

const MAX_APNG_DECODER_ALLOC_BYTES: u64 = 512 * 1024 * 1024;

/// Validated in-memory APNG metadata and the encoded bytes used to decode it.
///
/// Metadata describes one animation cycle. Timestamps keep variable frame delays; the scalar source
/// FPS is the average `frame_count / duration`.
#[derive(Debug, PartialEq)]
pub struct Apng<'a> {
    bytes: &'a [u8],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) duration_seconds: f64,
    pub(crate) source_frames_per_second: f64,
    pub(crate) frame_count: usize,
    pub(crate) frame_timestamps_seconds: Vec<f64>,
}

impl Apng<'_> {
    /// The clip's metadata in the form the frame sampler takes: duration, average frame rate, frame count and per-frame timestamps.
    pub fn metadata(&self) -> VideoMetadata<'_> {
        VideoMetadata {
            duration_seconds: self.duration_seconds,
            source_frames_per_second: self.source_frames_per_second,
            frame_count: self.frame_count,
            frame_timestamps_seconds: &self.frame_timestamps_seconds,
        }
    }

    /// Canvas width in pixels.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Canvas height in pixels.
    pub const fn height(&self) -> usize {
        self.height
    }
}

impl<'a> Apng<'a> {
    /// Probe and validate one complete APNG animation from borrowed in-memory bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ApngError> {
        validate_apng_encoded_len(bytes.len())?;
        let (width, height, frames) = apng_frames(bytes)?;
        let canvas_pixels = width
            .checked_mul(height)
            .ok_or(ApngError::DecodedPixelCountOverflow)?;

        let mut frame_timestamps_seconds = Vec::new();
        let mut duration_seconds = 0.0f64;
        let mut frame_count = 0usize;
        for frame in frames {
            let frame = frame.map_err(|source| ApngError::FrameDecode {
                frame_index: frame_count,
                message: source.to_string(),
            })?;
            frame_count = frame_count
                .checked_add(1)
                .ok_or(ApngError::FrameCountOverflow)?;
            validate_apng_decoded_work(frame_count, canvas_pixels)?;
            let frame_index = frame_count - 1;
            let decoded_width = frame.buffer().width() as usize;
            let decoded_height = frame.buffer().height() as usize;
            if (decoded_width, decoded_height) != (width, height) {
                return Err(ApngError::FrameDimensionMismatch {
                    frame_index,
                    expected_height: height,
                    expected_width: width,
                    actual_height: decoded_height,
                    actual_width: decoded_width,
                });
            }

            let (delay_numerator_ms, delay_denominator) = frame.delay().numer_denom_ms();
            if delay_numerator_ms == 0 {
                return Err(ApngError::ZeroFrameDelay { frame_index });
            }
            if delay_denominator == 0 {
                return Err(ApngError::TimingArithmetic { frame_index });
            }
            frame_timestamps_seconds
                .try_reserve(1)
                .map_err(|_| ApngError::Allocation {
                    stage: "frame timestamps",
                })?;
            frame_timestamps_seconds.push(duration_seconds);
            let delay_seconds =
                (f64::from(delay_numerator_ms) / f64::from(delay_denominator)) / 1_000.0;
            if !delay_seconds.is_finite() || delay_seconds <= 0.0 {
                return Err(ApngError::TimingArithmetic { frame_index });
            }
            let next_duration = duration_seconds + delay_seconds;
            if !next_duration.is_finite() || next_duration <= duration_seconds {
                return Err(ApngError::TimingArithmetic { frame_index });
            }
            duration_seconds = next_duration;
        }
        if frame_count == 0 {
            return Err(ApngError::EmptyAnimation);
        }

        let source_frames_per_second = frame_count as f64 / duration_seconds;
        if !source_frames_per_second.is_finite() || source_frames_per_second <= 0.0 {
            return Err(ApngError::SourceFramesPerSecondArithmetic);
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
    pub fn frame_source(&self) -> Result<ApngFrames<'a>, ApngError> {
        let (width, height, frames) = apng_frames(self.bytes)?;
        if (width, height) != (self.width, self.height) {
            return Err(ApngError::SourceDimensionMismatch {
                metadata_height: self.height,
                metadata_width: self.width,
                decoded_height: height,
                decoded_width: width,
            });
        }
        Ok(ApngFrames {
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

/// Why an APNG could not be probed or one of its frames could not be produced.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ApngError {
    /// The encoded file is larger than the byte cap.
    #[error("APNG has {actual} encoded bytes, exceeding the {limit}-byte cap")]
    TooManyEncodedBytes {
        /// Encoded size in bytes.
        actual: usize,
        /// The cap in bytes.
        limit: usize,
    },
    /// The PNG has no `acTL` animation control chunk.
    #[error("PNG is static; an acTL animation control chunk is required")]
    StaticPng,
    /// The animation iterator yielded no frame.
    #[error("APNG animation iterator returned no frames")]
    EmptyAnimation,
    /// The running frame count overflowed `usize`.
    #[error("APNG frame count overflows usize")]
    FrameCountOverflow,
    /// The animation has more frames than the video frame cap.
    #[error("APNG decoded {actual} frames, exceeding the {limit}-frame cap")]
    TooManyFrames {
        /// Frames decoded when the cap was crossed.
        actual: usize,
        /// The cap, `MAX_VIDEO_FRAMES`.
        limit: usize,
    },
    /// `frames * canvas pixels` overflowed `usize`.
    #[error("APNG decoded-pixel count overflows usize")]
    DecodedPixelCountOverflow,
    /// All frames together decode to more pixels than the image pixel cap.
    #[error("APNG decoded {pixels} canvas-frame pixels, exceeding the {limit}-pixel cap")]
    TooManyDecodedPixels {
        /// Canvas pixels summed over the frames decoded so far.
        pixels: usize,
        /// The cap, `MAX_IMAGE_PIXELS`.
        limit: usize,
    },
    /// A frame's RGB8 conversion or dimension check failed.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// The PNG decoder could not be initialized.
    #[error("initialize APNG {stage}: {message}")]
    Decoder {
        /// The initialization step that failed.
        stage: &'static str,
        /// The decoder's error, rendered.
        message: String,
    },
    /// A second pass over the file decoded a different canvas size than the probe saw.
    #[error(
        "APNG source dimensions changed from metadata {metadata_height}x{metadata_width} to decoder {decoded_height}x{decoded_width}"
    )]
    SourceDimensionMismatch {
        /// Canvas height the probe recorded.
        metadata_height: usize,
        /// Canvas width the probe recorded.
        metadata_width: usize,
        /// Canvas height the new decoder reports.
        decoded_height: usize,
        /// Canvas width the new decoder reports.
        decoded_width: usize,
    },
    /// The decoder failed on a frame.
    #[error("decode APNG frame {frame_index}: {message}")]
    FrameDecode {
        /// Index of the frame in the animation.
        frame_index: usize,
        /// The decoder's error, rendered.
        message: String,
    },
    /// A frame's size differs from the first frame's canvas.
    #[error(
        "APNG frame {frame_index} is {actual_height}x{actual_width}; canvas is {expected_height}x{expected_width}"
    )]
    FrameDimensionMismatch {
        /// Index of the frame in the animation.
        frame_index: usize,
        /// Canvas height.
        expected_height: usize,
        /// Canvas width.
        expected_width: usize,
        /// The frame's height.
        actual_height: usize,
        /// The frame's width.
        actual_width: usize,
    },
    /// A decoded frame's RGBA buffer is not `height * width * 4` bytes.
    #[error("APNG frame {frame_index} has {actual} RGBA bytes; dimensions require {expected}")]
    FramePixelDataLen {
        /// Index of the frame in the animation.
        frame_index: usize,
        /// Bytes the frame's dimensions require.
        expected: usize,
        /// Bytes the buffer holds.
        actual: usize,
    },
    /// A frame's delay numerator is zero, so timestamps would not increase.
    #[error("APNG frame {frame_index} has a zero delay numerator")]
    ZeroFrameDelay {
        /// Index of the frame in the animation.
        frame_index: usize,
    },
    /// A frame's delay, a timestamp or the running duration is not a finite positive number.
    #[error("APNG timing arithmetic failed at frame {frame_index}")]
    TimingArithmetic {
        /// Index of the frame in the animation.
        frame_index: usize,
    },
    /// The average frame rate `frame_count / duration` is not finite and positive.
    #[error("APNG average source-FPS arithmetic failed")]
    SourceFramesPerSecondArithmetic,
    /// Reserving memory failed.
    #[error("APNG could not reserve memory for {stage}")]
    Allocation {
        /// The buffer that failed to allocate.
        stage: &'static str,
    },
    /// A requested frame index is at or past the frame count.
    #[error("APNG frame index {requested} is outside {frame_count} frames")]
    FrameIndexOutOfRange {
        /// The index asked for.
        requested: usize,
        /// The animation's frame count.
        frame_count: usize,
    },
    /// A requested index is not later than the previous request; the source is one-pass.
    #[error("APNG frame index {requested} is not later than prior request {previous}")]
    FrameIndexNotIncreasing {
        /// The previous request's index.
        previous: usize,
        /// The index asked for.
        requested: usize,
    },
    /// The animation ended before the requested frame.
    #[error("APNG ended after {decoded} frames before requested frame {requested}")]
    UnexpectedFrameEnd {
        /// The index asked for.
        requested: usize,
        /// Frames the decoder produced.
        decoded: usize,
    },
    /// An earlier failure left the one-pass decoder in an unknown position; the source refuses further requests.
    #[error("APNG frame source is poisoned by an earlier failure")]
    PoisonedFrameSource,
}

/// A one-pass APNG implementation of the established selected-frame source contract.
pub struct ApngFrames<'a> {
    pub(crate) frames: Frames<'a>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) frame_count: usize,
    pub(crate) next_frame_index: usize,
    pub(crate) previous_requested_index: Option<usize>,
    pub(crate) poisoned: bool,
}

/// The canvas of `height` x `width` pixels is nonempty, within the `u32` range and within the pixel cap;
/// the RGB8 byte count it needs.
fn validate_canvas(height: usize, width: usize) -> Result<usize, ApngError> {
    if height == 0 || width == 0 {
        return Err(ImageError::ZeroDimension { height, width }.into());
    }
    let pixels = height
        .checked_mul(width)
        .ok_or(ApngError::DecodedPixelCountOverflow)?;
    if u32::try_from(height).is_err() || u32::try_from(width).is_err() || pixels > MAX_IMAGE_PIXELS
    {
        return Err(ApngError::TooManyDecodedPixels {
            pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    pixels
        .checked_mul(3)
        .ok_or(ApngError::DecodedPixelCountOverflow)
}

fn validate_apng_encoded_len(actual: usize) -> Result<(), ApngError> {
    if actual > MAX_ENCODED_APNG_BYTES {
        return Err(ApngError::TooManyEncodedBytes {
            actual,
            limit: MAX_ENCODED_APNG_BYTES,
        });
    }
    Ok(())
}

fn validate_apng_decoded_work(frame_count: usize, canvas_pixels: usize) -> Result<(), ApngError> {
    if frame_count > MAX_VIDEO_FRAMES {
        return Err(ApngError::TooManyFrames {
            actual: frame_count,
            limit: MAX_VIDEO_FRAMES,
        });
    }
    let decoded_pixels = frame_count
        .checked_mul(canvas_pixels)
        .ok_or(ApngError::DecodedPixelCountOverflow)?;
    if decoded_pixels > MAX_IMAGE_PIXELS {
        return Err(ApngError::TooManyDecodedPixels {
            pixels: decoded_pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }
    Ok(())
}

fn apng_frames<'a>(bytes: &'a [u8]) -> Result<(usize, usize, Frames<'a>), ApngError> {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_PIXELS as u32);
    limits.max_image_height = Some(MAX_IMAGE_PIXELS as u32);
    limits.max_alloc = Some(MAX_APNG_DECODER_ALLOC_BYTES);
    let decoder = PngDecoder::with_limits(Cursor::new(bytes), limits).map_err(|source| {
        ApngError::Decoder {
            stage: "PNG decoder",
            message: source.to_string(),
        }
    })?;
    if !decoder.is_apng().map_err(|source| ApngError::Decoder {
        stage: "animation check",
        message: source.to_string(),
    })? {
        return Err(ApngError::StaticPng);
    }
    let (width, height) = decoder.dimensions();
    validate_canvas(height as usize, width as usize)?;
    let frames = decoder
        .apng()
        .map_err(|source| ApngError::Decoder {
            stage: "animation decoder",
            message: source.to_string(),
        })?
        .into_frames();
    Ok((width as usize, height as usize, frames))
}

impl ApngFrames<'_> {
    fn fail<T>(&mut self, error: ApngError) -> Result<T, ApngError> {
        self.poisoned = true;
        Err(error)
    }
}

impl FrameSource for ApngFrames<'_> {
    fn frame(
        &mut self,
        frame_index: usize,
    ) -> Result<Image, Box<dyn std::error::Error + Send + Sync>> {
        self.decode(frame_index).map_err(Into::into)
    }
}

impl ApngFrames<'_> {
    fn decode(&mut self, frame_index: usize) -> Result<Image, ApngError> {
        if self.poisoned {
            return Err(ApngError::PoisonedFrameSource);
        }
        if frame_index >= self.frame_count {
            return self.fail(ApngError::FrameIndexOutOfRange {
                requested: frame_index,
                frame_count: self.frame_count,
            });
        }
        if let Some(previous) = self.previous_requested_index
            && frame_index <= previous
        {
            return self.fail(ApngError::FrameIndexNotIncreasing {
                previous,
                requested: frame_index,
            });
        }

        loop {
            let source_index = self.next_frame_index;
            let frame = match self.frames.next() {
                Some(Ok(frame)) => frame,
                Some(Err(source)) => {
                    return self.fail(ApngError::FrameDecode {
                        frame_index: source_index,
                        message: source.to_string(),
                    });
                }
                None => {
                    return self.fail(ApngError::UnexpectedFrameEnd {
                        requested: frame_index,
                        decoded: source_index,
                    });
                }
            };
            self.next_frame_index = match self.next_frame_index.checked_add(1) {
                Some(next) => next,
                None => return self.fail(ApngError::FrameCountOverflow),
            };
            let actual_width = frame.buffer().width() as usize;
            let actual_height = frame.buffer().height() as usize;
            if (actual_width, actual_height) != (self.width, self.height) {
                return self.fail(ApngError::FrameDimensionMismatch {
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
) -> Result<Image, ApngError> {
    let rgba = frame.into_buffer();
    let height = rgba.height() as usize;
    let width = rgba.width() as usize;
    let expected_rgb_bytes = validate_canvas(height, width)?;
    let expected_rgba_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(ApngError::DecodedPixelCountOverflow)?;
    let rgba = rgba.into_raw();
    if rgba.len() != expected_rgba_bytes {
        return Err(ApngError::FramePixelDataLen {
            frame_index,
            expected: expected_rgba_bytes,
            actual: rgba.len(),
        });
    }
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(expected_rgb_bytes)
        .map_err(|_| ApngError::Allocation {
            stage: "selected RGB frame",
        })?;
    for pixel in rgba.as_chunks::<4>().0 {
        // `image` has already applied APNG disposal and alpha blending to this RGBA canvas; the
        // decoded-image boundary is RGB8, so alpha is dropped.
        rgb.extend_from_slice(&pixel[..3]);
    }
    Ok(Image::from_rgb8(height, width, rgb)?)
}

#[cfg(test)]
mod tests {
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
    fn an_apng_preserves_variable_timing_and_the_zero_denominator_policy() {
        let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
        let apng = Apng::from_bytes(&bytes).unwrap();
        let metadata = apng.metadata();
        assert_eq!((apng.height(), apng.width()), (1, 2));
        assert_eq!(metadata.frame_count, 3);
        let first_delay = (1_000.0f64 / 3.0) / 1_000.0;
        assert_eq!(
            metadata.frame_timestamps_seconds,
            &[0.0, first_delay, first_delay + 0.25]
        );
        // The final frame encodes delay_num=10, delay_den=0; `image` applies PNG's denominator 100,
        // giving 0.1 s. The public reduced ratio does not record a zero denominator.
        let expected_duration = first_delay + 0.25 + 0.1;
        assert_eq!(
            metadata.duration_seconds.to_bits(),
            expected_duration.to_bits()
        );
        assert_eq!(
            metadata.source_frames_per_second.to_bits(),
            (3.0 / expected_duration).to_bits()
        );
    }

    #[test]
    fn the_frame_source_scans_sparse_frames_and_returns_composited_rgb() {
        let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
        let apng = Apng::from_bytes(&bytes).unwrap();
        let mut source = apng.frame_source().unwrap();
        let first = source.decode(0).unwrap();
        assert_eq!(first.rgb(), &[255, 0, 0, 0, 0, 255]);
        let third = source.decode(2).unwrap();
        assert_eq!(third.rgb(), &[0, 0, 0, 255, 255, 0]);

        let mut repeated = apng.frame_source().unwrap();
        let second = repeated.decode(1).unwrap();
        assert_eq!(second.rgb(), &[127, 128, 0, 0, 0, 255]);
        assert_eq!(
            repeated.decode(1).unwrap_err(),
            ApngError::FrameIndexNotIncreasing {
                previous: 1,
                requested: 1,
            }
        );
        assert_eq!(
            repeated.decode(2).unwrap_err(),
            ApngError::PoisonedFrameSource
        );

        let mut decreasing = apng.frame_source().unwrap();
        decreasing.decode(2).unwrap();
        assert_eq!(
            decreasing.decode(0).unwrap_err(),
            ApngError::FrameIndexNotIncreasing {
                previous: 2,
                requested: 0,
            }
        );

        let mut out_of_range = apng.frame_source().unwrap();
        assert_eq!(
            out_of_range.decode(3).unwrap_err(),
            ApngError::FrameIndexOutOfRange {
                requested: 3,
                frame_count: 3,
            }
        );
        assert_eq!(
            out_of_range.decode(0).unwrap_err(),
            ApngError::PoisonedFrameSource
        );
    }

    /// The decoder is a [`FrameSource`]: a front end asks it for frames through the trait, and a failure
    /// comes back as the source's own typed error.
    #[test]
    fn the_frame_source_serves_a_front_end_through_the_trait() {
        let bytes = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
        let apng = Apng::from_bytes(&bytes).unwrap();
        let mut frames = apng.frame_source().unwrap();
        let source: &mut dyn FrameSource = &mut frames;
        assert_eq!(source.frame(0).unwrap().rgb(), &[255, 0, 0, 0, 0, 255]);
        let error = source.frame(0).unwrap_err();
        assert_eq!(
            error.downcast_ref::<ApngError>(),
            Some(&ApngError::FrameIndexNotIncreasing {
                previous: 0,
                requested: 0,
            })
        );
    }

    #[test]
    fn static_malformed_zero_delay_and_capped_files_are_typed_errors() {
        let static_png = decode_base64_fixture(STATIC_PNG_BASE64);
        assert_eq!(
            Apng::from_bytes(&static_png).unwrap_err(),
            ApngError::StaticPng
        );

        let valid = decode_base64_fixture(VARIABLE_DELAY_APNG_BASE64);
        let truncated = &valid[..valid.len() - 20];
        assert!(matches!(
            Apng::from_bytes(truncated),
            Err(ApngError::FrameDecode { .. }) | Err(ApngError::Decoder { .. })
        ));

        let zero_delay = decode_base64_fixture(ZERO_DELAY_APNG_BASE64);
        assert_eq!(
            Apng::from_bytes(&zero_delay).unwrap_err(),
            ApngError::ZeroFrameDelay { frame_index: 1 }
        );

        assert_eq!(
            validate_apng_encoded_len(MAX_ENCODED_APNG_BYTES + 1),
            Err(ApngError::TooManyEncodedBytes {
                actual: MAX_ENCODED_APNG_BYTES + 1,
                limit: MAX_ENCODED_APNG_BYTES,
            })
        );
        assert_eq!(
            validate_apng_decoded_work(MAX_VIDEO_FRAMES + 1, 1),
            Err(ApngError::TooManyFrames {
                actual: MAX_VIDEO_FRAMES + 1,
                limit: MAX_VIDEO_FRAMES,
            })
        );
        let pixels_per_frame = MAX_IMAGE_PIXELS / 2 + 1;
        assert_eq!(
            validate_apng_decoded_work(2, pixels_per_frame),
            Err(ApngError::TooManyDecodedPixels {
                pixels: pixels_per_frame * 2,
                limit: MAX_IMAGE_PIXELS,
            })
        );
        assert_eq!(
            validate_apng_decoded_work(2, usize::MAX),
            Err(ApngError::DecodedPixelCountOverflow)
        );
    }
}
