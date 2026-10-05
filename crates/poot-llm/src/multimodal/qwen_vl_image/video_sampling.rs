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
    pub duration_seconds: f64,
    pub source_frames_per_second: f64,
    pub frame_count: usize,
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
    pub fn frame_indices(&self) -> &[usize] {
        &self.frame_indices
    }

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
    pub sample_plan: Qwen25VlVideoSamplePlan,
    pub processed_video: Qwen25VlProcessedVideo,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Qwen25VlVideoSamplingError {
    #[error("Qwen2.5-VL source video duration must be finite, got {actual}")]
    NonFiniteDuration { actual: f64 },
    #[error("Qwen2.5-VL source video duration must be positive, got {actual}")]
    NonPositiveDuration { actual: f64 },
    #[error("Qwen2.5-VL source FPS must be finite, got {actual}")]
    NonFiniteSourceFramesPerSecond { actual: f64 },
    #[error("Qwen2.5-VL source FPS must be positive, got {actual}")]
    NonPositiveSourceFramesPerSecond { actual: f64 },
    #[error("Qwen2.5-VL source video must contain at least one frame")]
    EmptySource,
    #[error(
        "Qwen2.5-VL source frame count {actual} cannot be represented exactly for f64 sampling"
    )]
    FrameCountNotExactlyRepresentable { actual: usize },
    #[error("Qwen2.5-VL source has {frame_count} frames but {timestamp_count} frame timestamps")]
    TimestampCountMismatch {
        frame_count: usize,
        timestamp_count: usize,
    },
    #[error("Qwen2.5-VL source frame timestamp {index} must be finite, got {actual}")]
    NonFiniteTimestamp { index: usize, actual: f64 },
    #[error("Qwen2.5-VL source frame timestamp {index} must be nonnegative, got {actual}")]
    NegativeTimestamp { index: usize, actual: f64 },
    #[error(
        "Qwen2.5-VL source frame timestamp {index} is {actual}, not later than previous timestamp {previous}"
    )]
    NonMonotonicTimestamp {
        index: usize,
        previous: f64,
        actual: f64,
    },
    #[error("Qwen2.5-VL source frame timestamp {index} is {actual}, after duration {duration}")]
    TimestampAfterDuration {
        index: usize,
        actual: f64,
        duration: f64,
    },
    #[error("Qwen2.5-VL sampled frame-count arithmetic is not finite")]
    SampleCountArithmeticOverflow,
    #[error("Qwen2.5-VL requested FPS selects zero frames from this source")]
    EmptySelection,
    #[error(
        "Qwen2.5-VL requested FPS selects {sampled_frame_count} frames from a {source_frame_count}-frame source"
    )]
    SampleCountExceedsSource {
        sampled_frame_count: usize,
        source_frame_count: usize,
    },
    #[error("Qwen2.5-VL requested FPS selects {actual} frames, exceeding the {limit}-frame cap")]
    TooManySampledFrames { actual: usize, limit: usize },
    #[error("Qwen2.5-VL video sampling index arithmetic overflows usize")]
    IndexOverflow,
    #[error("Qwen2.5-VL video sampling selected source index {index} outside {frame_count} frames")]
    IndexOutOfRange { index: usize, frame_count: usize },
    #[error("Qwen2.5-VL video sampling could not reserve memory for {stage}")]
    Allocation { stage: &'static str },
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
