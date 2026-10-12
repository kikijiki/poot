use poot_models::mrope::MropeSampledFramesPerSecond;

use super::assembly::Qwen25VlProcessedVideo;

use super::MAX_VIDEO_FRAMES;

/// Already-probed source metadata for deterministic Qwen2.5-VL frame sampling.
///
/// Transformers 4.49.0 `VideoMetadata` has no per-frame timestamps. This boundary requires
/// caller-normalized presentation timestamps so every selected index has clear provenance. They must
/// increase strictly and may equal the inclusive duration endpoint; they need not match a
/// constant-frame-rate derivation. The upstream default sampler uses only `frame_count` and
/// `source_frames_per_second` for index selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen25VlVideoMetadata<'a> {
    /// The clip's length in seconds; finite and positive, and no timestamp exceeds it.
    pub duration_seconds: f64,
    /// The source's frame rate in frames per second; finite and positive.
    pub source_frames_per_second: f64,
    /// The number of source frames; at least one, and equal to `frame_timestamps_seconds.len()`.
    pub frame_count: usize,
    /// One presentation timestamp in seconds per source frame: finite, nonnegative and strictly increasing.
    pub frame_timestamps_seconds: &'a [f64],
}

/// Completed, owned frame-selection plan in source order.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlVideoSamplePlan {
    pub(super) frame_indices: Vec<usize>,
    pub(super) frame_timestamps_seconds: Vec<f64>,
    pub(super) processor_frames_per_second: MropeSampledFramesPerSecond,
}

impl Qwen25VlVideoSamplePlan {
    /// Source frame indices the plan selects, strictly increasing.
    pub fn frame_indices(&self) -> &[usize] {
        &self.frame_indices
    }

    /// The presentation timestamp of each selected frame, in the order of [`Self::frame_indices`].
    pub fn frame_timestamps_seconds(&self) -> &[f64] {
        &self.frame_timestamps_seconds
    }

    /// The supplied processor FPS, intentionally not `selected_count / duration`.
    pub const fn processor_frames_per_second(&self) -> MropeSampledFramesPerSecond {
        self.processor_frames_per_second
    }
}

/// Raw-video sampling provenance paired with existing processed-video output.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlSampledVideo {
    /// Which source frames were selected and at what timestamps.
    pub sample_plan: Qwen25VlVideoSamplePlan,
    /// The selected frames after the image processor's resize, normalization and patching.
    pub processed_video: Qwen25VlProcessedVideo,
}

/// Why uniform frame sampling of a probed video is refused.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Qwen25VlVideoSamplingError {
    /// The duration is NaN or infinite.
    #[error("Qwen2.5-VL source video duration must be finite, got {actual}")]
    NonFiniteDuration {
        /// The duration given.
        actual: f64,
    },
    /// The duration is zero or negative.
    #[error("Qwen2.5-VL source video duration must be positive, got {actual}")]
    NonPositiveDuration {
        /// The duration given.
        actual: f64,
    },
    /// The source frame rate is NaN or infinite.
    #[error("Qwen2.5-VL source FPS must be finite, got {actual}")]
    NonFiniteSourceFramesPerSecond {
        /// The frame rate given.
        actual: f64,
    },
    /// The source frame rate is zero or negative.
    #[error("Qwen2.5-VL source FPS must be positive, got {actual}")]
    NonPositiveSourceFramesPerSecond {
        /// The frame rate given.
        actual: f64,
    },
    /// The source holds no frame.
    #[error("Qwen2.5-VL source video must contain at least one frame")]
    EmptySource,
    /// The frame count has no exact `f64` value, so the sampling arithmetic would round it.
    #[error(
        "Qwen2.5-VL source frame count {actual} cannot be represented exactly for f64 sampling"
    )]
    FrameCountNotExactlyRepresentable {
        /// The frame count given.
        actual: usize,
    },
    /// The timestamp list and the frame count disagree.
    #[error("Qwen2.5-VL source has {frame_count} frames but {timestamp_count} frame timestamps")]
    TimestampCountMismatch {
        /// The declared frame count.
        frame_count: usize,
        /// The number of timestamps given.
        timestamp_count: usize,
    },
    /// A timestamp is NaN or infinite.
    #[error("Qwen2.5-VL source frame timestamp {index} must be finite, got {actual}")]
    NonFiniteTimestamp {
        /// Position of the timestamp.
        index: usize,
        /// The value given.
        actual: f64,
    },
    /// A timestamp is negative.
    #[error("Qwen2.5-VL source frame timestamp {index} must be nonnegative, got {actual}")]
    NegativeTimestamp {
        /// Position of the timestamp.
        index: usize,
        /// The value given.
        actual: f64,
    },
    /// A timestamp is not later than the one before it.
    #[error(
        "Qwen2.5-VL source frame timestamp {index} is {actual}, not later than previous timestamp {previous}"
    )]
    NonMonotonicTimestamp {
        /// Position of the timestamp.
        index: usize,
        /// The preceding timestamp.
        previous: f64,
        /// The value given.
        actual: f64,
    },
    /// A timestamp is later than the clip's duration.
    #[error("Qwen2.5-VL source frame timestamp {index} is {actual}, after duration {duration}")]
    TimestampAfterDuration {
        /// Position of the timestamp.
        index: usize,
        /// The value given.
        actual: f64,
        /// The declared duration.
        duration: f64,
    },
    /// The sampled-frame-count arithmetic is not finite or does not fit `usize`.
    #[error("Qwen2.5-VL sampled frame-count arithmetic is not finite")]
    SampleCountArithmeticOverflow,
    /// The requested frame rate selects no frame from this source.
    #[error("Qwen2.5-VL requested FPS selects zero frames from this source")]
    EmptySelection,
    /// The requested frame rate selects more frames than the source has.
    #[error(
        "Qwen2.5-VL requested FPS selects {sampled_frame_count} frames from a {source_frame_count}-frame source"
    )]
    SampleCountExceedsSource {
        /// The number of frames the request asks for.
        sampled_frame_count: usize,
        /// The number of frames the source has.
        source_frame_count: usize,
    },
    /// The selection exceeds the per-request video frame cap.
    #[error("Qwen2.5-VL requested FPS selects {actual} frames, exceeding the {limit}-frame cap")]
    TooManySampledFrames {
        /// The selected frame count.
        actual: usize,
        /// The cap, `MAX_VIDEO_FRAMES`.
        limit: usize,
    },
    /// Computing a source index overflows `usize`.
    #[error("Qwen2.5-VL video sampling index arithmetic overflows usize")]
    IndexOverflow,
    /// A computed source index falls outside the source.
    #[error("Qwen2.5-VL video sampling selected source index {index} outside {frame_count} frames")]
    IndexOutOfRange {
        /// The computed index.
        index: usize,
        /// The source's frame count.
        frame_count: usize,
    },
    /// Reserving memory for the plan failed.
    #[error("Qwen2.5-VL video sampling could not reserve memory for {stage}")]
    Allocation {
        /// The vector that failed to allocate.
        stage: &'static str,
    },
}

/// Plan cached-Transformers-compatible uniform frame sampling from already-probed metadata.
pub fn plan_qwen2_5_vl_video_sampling(
    metadata: Qwen25VlVideoMetadata<'_>,
    requested_frames_per_second: MropeSampledFramesPerSecond,
) -> Result<Qwen25VlVideoSamplePlan, Qwen25VlVideoSamplingError> {
    if !metadata.duration_seconds.is_finite() {
        return Err(Qwen25VlVideoSamplingError::NonFiniteDuration {
            actual: metadata.duration_seconds,
        });
    }
    if metadata.duration_seconds <= 0.0 {
        return Err(Qwen25VlVideoSamplingError::NonPositiveDuration {
            actual: metadata.duration_seconds,
        });
    }
    if !metadata.source_frames_per_second.is_finite() {
        return Err(Qwen25VlVideoSamplingError::NonFiniteSourceFramesPerSecond {
            actual: metadata.source_frames_per_second,
        });
    }
    if metadata.source_frames_per_second <= 0.0 {
        return Err(
            Qwen25VlVideoSamplingError::NonPositiveSourceFramesPerSecond {
                actual: metadata.source_frames_per_second,
            },
        );
    }
    if metadata.frame_count == 0 {
        return Err(Qwen25VlVideoSamplingError::EmptySource);
    }
    let frame_count_f64 = metadata.frame_count as f64;
    if frame_count_f64 as u128 != metadata.frame_count as u128 {
        return Err(
            Qwen25VlVideoSamplingError::FrameCountNotExactlyRepresentable {
                actual: metadata.frame_count,
            },
        );
    }
    if metadata.frame_timestamps_seconds.len() != metadata.frame_count {
        return Err(Qwen25VlVideoSamplingError::TimestampCountMismatch {
            frame_count: metadata.frame_count,
            timestamp_count: metadata.frame_timestamps_seconds.len(),
        });
    }
    let mut previous_timestamp = None;
    for (index, &timestamp) in metadata.frame_timestamps_seconds.iter().enumerate() {
        if !timestamp.is_finite() {
            return Err(Qwen25VlVideoSamplingError::NonFiniteTimestamp {
                index,
                actual: timestamp,
            });
        }
        if timestamp < 0.0 {
            return Err(Qwen25VlVideoSamplingError::NegativeTimestamp {
                index,
                actual: timestamp,
            });
        }
        if let Some(previous) = previous_timestamp
            && timestamp <= previous
        {
            return Err(Qwen25VlVideoSamplingError::NonMonotonicTimestamp {
                index,
                previous,
                actual: timestamp,
            });
        }
        if timestamp > metadata.duration_seconds {
            return Err(Qwen25VlVideoSamplingError::TimestampAfterDuration {
                index,
                actual: timestamp,
                duration: metadata.duration_seconds,
            });
        }
        previous_timestamp = Some(timestamp);
    }

    // Keep Transformers 4.49.0's left-to-right f64 expression before positive `int` truncation.
    let sampled_count_f64 =
        frame_count_f64 / metadata.source_frames_per_second * requested_frames_per_second.get();
    if !sampled_count_f64.is_finite() {
        return Err(Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow);
    }
    let sampled_count_truncated = sampled_count_f64.trunc();
    // Convert through u128: `usize::MAX as f64` rounds upward on some targets and is exact on others,
    // which would make the inclusive maximum platform-dependent.
    let sampled_count_u128 = sampled_count_truncated as u128;
    if sampled_count_u128 > usize::MAX as u128 {
        return Err(Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow);
    }
    let nominal_sampled_frame_count = sampled_count_u128 as usize;
    if nominal_sampled_frame_count == 0 {
        return Err(Qwen25VlVideoSamplingError::EmptySelection);
    }
    if nominal_sampled_frame_count > metadata.frame_count {
        return Err(Qwen25VlVideoSamplingError::SampleCountExceedsSource {
            sampled_frame_count: nominal_sampled_frame_count,
            source_frame_count: metadata.frame_count,
        });
    }

    let sampling_step = frame_count_f64 / nominal_sampled_frame_count as f64;
    let actual_sampled_count_f64 = (frame_count_f64 / sampling_step).ceil();
    if !actual_sampled_count_f64.is_finite() {
        return Err(Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow);
    }
    let actual_sampled_count_u128 = actual_sampled_count_f64 as u128;
    if actual_sampled_count_u128 > usize::MAX as u128 {
        return Err(Qwen25VlVideoSamplingError::SampleCountArithmeticOverflow);
    }
    let sampled_frame_count = actual_sampled_count_u128 as usize;
    if sampled_frame_count > MAX_VIDEO_FRAMES {
        return Err(Qwen25VlVideoSamplingError::TooManySampledFrames {
            actual: sampled_frame_count,
            limit: MAX_VIDEO_FRAMES,
        });
    }

    // NumPy's actual progression is `dtype(start + step) - dtype(start)`. Start is zero and the nominal
    // count is at most the source count, so this is the positive truncated integer step.
    let integer_step = sampling_step.trunc() as usize;
    debug_assert_ne!(integer_step, 0);
    let mut frame_indices = Vec::new();
    frame_indices
        .try_reserve_exact(sampled_frame_count)
        .map_err(|_| Qwen25VlVideoSamplingError::Allocation {
            stage: "frame indices",
        })?;
    let mut frame_timestamps_seconds = Vec::new();
    frame_timestamps_seconds
        .try_reserve_exact(sampled_frame_count)
        .map_err(|_| Qwen25VlVideoSamplingError::Allocation {
            stage: "frame timestamps",
        })?;
    for sampled_index in 0..sampled_frame_count {
        let source_index = sampled_index
            .checked_mul(integer_step)
            .ok_or(Qwen25VlVideoSamplingError::IndexOverflow)?;
        if source_index >= metadata.frame_count {
            return Err(Qwen25VlVideoSamplingError::IndexOutOfRange {
                index: source_index,
                frame_count: metadata.frame_count,
            });
        }
        frame_indices.push(source_index);
        frame_timestamps_seconds.push(metadata.frame_timestamps_seconds[source_index]);
    }
    Ok(Qwen25VlVideoSamplePlan {
        frame_indices,
        frame_timestamps_seconds,
        processor_frames_per_second: requested_frames_per_second,
    })
}
