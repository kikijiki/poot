//! Host-side multimodal position construction for sectioned RoPE.
//!
//! The caller owns tokenization and supplies the tokenizer's special-token ids explicitly. This module
//! validates the resulting image/video marker spans, turns them into a resolved layout, and implements
//! Qwen2-VL unit-temporal-stride plus Qwen2.5-VL caller-resolved timestamp scaling. A pure config boundary
//! resolves only metadata present in HF `config.json`; it does not resolve token strings, derive video grid
//! timing, or inspect vision embeddings.

/// The largest position exactly mirrored by `poot_eval::Tensor`'s f32 compatibility payload.
/// The authoritative I32 payload can represent more, but the CPU oracle still reads gather indices
/// through that mirror, so accepting a larger host position would make source verification lossy.
pub const MAX_EXACT_MROPE_POSITION: usize = 1 << 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MropeVisualKind {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MropeSpecialTokenKind {
    VisionStart,
    Image,
    Video,
}

impl std::fmt::Display for MropeSpecialTokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VisionStart => f.write_str("vision start"),
            Self::Image => f.write_str("image placeholder"),
            Self::Video => f.write_str("video placeholder"),
        }
    }
}

/// Tokenizer-specific ids needed to recognize one Qwen-VL visual marker span. The caller resolves these
/// ids from its tokenizer/config; span discovery deliberately does not assign ids to token strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeSpecialTokenIds {
    pub vision_start: u32,
    pub image: u32,
    pub video: u32,
}

impl MropeSpecialTokenIds {
    pub const fn new(vision_start: u32, image: u32, video: u32) -> Self {
        Self {
            vision_start,
            image,
            video,
        }
    }

    /// Construct a marker set only when every semantic role has a distinct token ID.
    pub fn try_new(vision_start: u32, image: u32, video: u32) -> Result<Self, MropeSpanError> {
        let ids = Self::new(vision_start, image, video);
        ids.validate()?;
        Ok(ids)
    }

    fn classify(self, token: u32) -> Option<MropeSpecialTokenKind> {
        if token == self.vision_start {
            Some(MropeSpecialTokenKind::VisionStart)
        } else if token == self.image {
            Some(MropeSpecialTokenKind::Image)
        } else if token == self.video {
            Some(MropeSpecialTokenKind::Video)
        } else {
            None
        }
    }

    fn validate(self) -> Result<(), MropeSpanError> {
        let ids = [
            (MropeSpecialTokenKind::VisionStart, self.vision_start),
            (MropeSpecialTokenKind::Image, self.image),
            (MropeSpecialTokenKind::Video, self.video),
        ];
        for (index, &(first, token_id)) in ids.iter().enumerate() {
            if let Some(&(second, _)) = ids[index + 1..]
                .iter()
                .find(|(_, candidate)| *candidate == token_id)
            {
                return Err(MropeSpanError::DuplicateSpecialTokenId {
                    token_id,
                    first,
                    second,
                });
            }
        }
        Ok(())
    }
}

const MROPE_SPECIAL_TOKEN_ROLE_ORDER: [MropeSpecialTokenKind; 3] = [
    MropeSpecialTokenKind::VisionStart,
    MropeSpecialTokenKind::Image,
    MropeSpecialTokenKind::Video,
];

/// Processor/template-owned marker string for one Qwen-VL mRoPE semantic role.
///
/// The tokenizer, not this record, maps the string to a token ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeProcessorSpecialToken<'a> {
    pub role: MropeSpecialTokenKind,
    pub token: &'a str,
}

impl<'a> MropeProcessorSpecialToken<'a> {
    pub const fn new(role: MropeSpecialTokenKind, token: &'a str) -> Self {
        Self { role, token }
    }
}

/// Tokenizer-owned string-to-ID record.
///
/// This in-memory shape lets tests and future adapters preserve duplicate source metadata when that metadata
/// is representable. A live `tokenizers::Tokenizer::token_to_id` adapter can instead synthesize unique records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeTokenizerSpecialToken<'a> {
    pub token: &'a str,
    pub id: u32,
}

impl<'a> MropeTokenizerSpecialToken<'a> {
    pub const fn new(token: &'a str, id: u32) -> Self {
        Self { token, id }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropeSpecialTokenReconcileError {
    #[error("mRoPE processor metadata is missing {role} special token")]
    MissingProcessorSpecialToken { role: MropeSpecialTokenKind },
    #[error("mRoPE processor metadata has duplicate {role} special-token records")]
    DuplicateProcessorSpecialToken { role: MropeSpecialTokenKind },
    #[error("mRoPE processor marker {token:?} is assigned to both {first} and {second}")]
    DuplicateProcessorMarker {
        token: String,
        first: MropeSpecialTokenKind,
        second: MropeSpecialTokenKind,
    },
    #[error("mRoPE tokenizer has no ID for {role} marker {token:?}")]
    UnknownTokenizerSpecialToken {
        role: MropeSpecialTokenKind,
        token: String,
    },
    #[error(
        "mRoPE tokenizer has duplicate records for marker {token:?}: IDs {first_id} and {second_id}"
    )]
    DuplicateTokenizerSpecialToken {
        token: String,
        first_id: u32,
        second_id: u32,
    },
    #[error("mRoPE tokenizer ID {token_id} is assigned to both {first} and {second}")]
    DuplicateTokenizerSpecialTokenId {
        token_id: u32,
        first: MropeSpecialTokenKind,
        second: MropeSpecialTokenKind,
    },
    #[error(
        "mRoPE {role} marker {token:?} resolves to tokenizer ID {tokenizer_id}, but config metadata declares {config_id}"
    )]
    ConfigSpecialTokenMismatch {
        role: MropeSpecialTokenKind,
        token: String,
        tokenizer_id: u32,
        config_id: u32,
    },
}

/// Reconcile processor marker strings, tokenizer string IDs, and config metadata into existing mRoPE IDs.
///
/// The tokenizer records are authoritative for string-to-ID mapping. Config metadata must agree with the
/// resolved semantic IDs before callers can feed span discovery or position assembly.
pub fn reconcile_mrope_special_token_ids(
    processor_tokens: &[MropeProcessorSpecialToken<'_>],
    tokenizer_tokens: &[MropeTokenizerSpecialToken<'_>],
    config: &MropeConfigMetadata,
) -> Result<MropeSpecialTokenIds, MropeSpecialTokenReconcileError> {
    let mut role_tokens = [None; 3];
    for role in MROPE_SPECIAL_TOKEN_ROLE_ORDER {
        let mut matches = processor_tokens
            .iter()
            .filter(|record| record.role == role)
            .map(|record| record.token);
        let Some(token) = matches.next() else {
            return Err(MropeSpecialTokenReconcileError::MissingProcessorSpecialToken { role });
        };
        if matches.next().is_some() {
            return Err(MropeSpecialTokenReconcileError::DuplicateProcessorSpecialToken { role });
        }
        role_tokens[mrope_special_token_role_index(role)] = Some(token);
    }

    for first_index in 0..MROPE_SPECIAL_TOKEN_ROLE_ORDER.len() {
        for second_index in first_index + 1..MROPE_SPECIAL_TOKEN_ROLE_ORDER.len() {
            let first = MROPE_SPECIAL_TOKEN_ROLE_ORDER[first_index];
            let second = MROPE_SPECIAL_TOKEN_ROLE_ORDER[second_index];
            let first_token = role_tokens[first_index].expect("role token checked above");
            let second_token = role_tokens[second_index].expect("role token checked above");
            if first_token == second_token {
                return Err(MropeSpecialTokenReconcileError::DuplicateProcessorMarker {
                    token: first_token.to_string(),
                    first,
                    second,
                });
            }
        }
    }

    let mut role_ids = [0u32; 3];
    for role in MROPE_SPECIAL_TOKEN_ROLE_ORDER {
        let index = mrope_special_token_role_index(role);
        let token = role_tokens[index].expect("role token checked above");
        let mut matches = tokenizer_tokens
            .iter()
            .filter(|record| record.token == token)
            .map(|record| record.id);
        let Some(id) = matches.next() else {
            return Err(
                MropeSpecialTokenReconcileError::UnknownTokenizerSpecialToken {
                    role,
                    token: token.to_string(),
                },
            );
        };
        if let Some(second_id) = matches.next() {
            return Err(
                MropeSpecialTokenReconcileError::DuplicateTokenizerSpecialToken {
                    token: token.to_string(),
                    first_id: id,
                    second_id,
                },
            );
        }
        role_ids[index] = id;
    }

    for first_index in 0..MROPE_SPECIAL_TOKEN_ROLE_ORDER.len() {
        for second_index in first_index + 1..MROPE_SPECIAL_TOKEN_ROLE_ORDER.len() {
            if role_ids[first_index] == role_ids[second_index] {
                return Err(
                    MropeSpecialTokenReconcileError::DuplicateTokenizerSpecialTokenId {
                        token_id: role_ids[first_index],
                        first: MROPE_SPECIAL_TOKEN_ROLE_ORDER[first_index],
                        second: MROPE_SPECIAL_TOKEN_ROLE_ORDER[second_index],
                    },
                );
            }
        }
    }

    let config_ids = config.special_token_ids();
    for role in MROPE_SPECIAL_TOKEN_ROLE_ORDER {
        let index = mrope_special_token_role_index(role);
        let tokenizer_id = role_ids[index];
        let config_id = mrope_special_token_id_for_role(config_ids, role);
        if tokenizer_id != config_id {
            return Err(
                MropeSpecialTokenReconcileError::ConfigSpecialTokenMismatch {
                    role,
                    token: role_tokens[index]
                        .expect("role token checked above")
                        .to_string(),
                    tokenizer_id,
                    config_id,
                },
            );
        }
    }

    Ok(MropeSpecialTokenIds::new(
        role_ids[0],
        role_ids[1],
        role_ids[2],
    ))
}

const fn mrope_special_token_role_index(role: MropeSpecialTokenKind) -> usize {
    match role {
        MropeSpecialTokenKind::VisionStart => 0,
        MropeSpecialTokenKind::Image => 1,
        MropeSpecialTokenKind::Video => 2,
    }
}

const fn mrope_special_token_id_for_role(
    ids: MropeSpecialTokenIds,
    role: MropeSpecialTokenKind,
) -> u32 {
    match role {
        MropeSpecialTokenKind::VisionStart => ids.vision_start,
        MropeSpecialTokenKind::Image => ids.image,
        MropeSpecialTokenKind::Video => ids.video,
    }
}

/// Validated mRoPE values read directly from a Qwen2-VL or Qwen2.5-VL HF `config.json`.
///
/// The model config contains the section widths, marker IDs, and (for Qwen2.5-VL when explicitly
/// present) temporal position tokens per second. It does not contain processor-derived seconds per grid
/// step, so this type intentionally has no `MropeSecondsPerGridStep` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeConfigMetadata {
    mrope_section: [usize; 3],
    special_token_ids: MropeSpecialTokenIds,
    temporal_tokens_per_second: Option<MropeTemporalTokensPerSecond>,
}

impl MropeConfigMetadata {
    pub fn from_hf_config_slice(json: &[u8]) -> Result<Self, MropeConfigMetadataError> {
        poot_load::qwen_vl::QwenVlHfMetadata::from_slice(json)?.try_into()
    }

    pub const fn mrope_section(&self) -> [usize; 3] {
        self.mrope_section
    }

    pub const fn special_token_ids(&self) -> MropeSpecialTokenIds {
        self.special_token_ids
    }

    pub const fn temporal_tokens_per_second(&self) -> Option<MropeTemporalTokensPerSecond> {
        self.temporal_tokens_per_second
    }
}

impl TryFrom<poot_load::qwen_vl::QwenVlHfMetadata> for MropeConfigMetadata {
    type Error = MropeConfigMetadataError;

    fn try_from(raw: poot_load::qwen_vl::QwenVlHfMetadata) -> Result<Self, Self::Error> {
        let special_token_ids = MropeSpecialTokenIds::try_new(
            raw.vision_start_token_id(),
            raw.image_token_id(),
            raw.video_token_id(),
        )?;
        let temporal_tokens_per_second = raw
            .temporal_tokens_per_second()
            .map(MropeTemporalTokensPerSecond::new)
            .transpose()?;
        Ok(Self {
            mrope_section: raw.mrope_section(),
            special_token_ids,
            temporal_tokens_per_second,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MropeConfigMetadataError {
    #[error(transparent)]
    Load(#[from] poot_load::qwen_vl::QwenVlHfMetadataError),
    #[error(transparent)]
    SpecialTokenIds(#[from] MropeSpanError),
    #[error(transparent)]
    Timing(#[from] MropePositionError),
}

impl std::fmt::Display for MropeVisualKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image => f.write_str("image"),
            Self::Video => f.write_str("video"),
        }
    }
}

/// Raw vision-grid dimensions before the spatial merge performed by the vision connector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl MropeGrid {
    pub const fn new(temporal: usize, height: usize, width: usize) -> Self {
        Self {
            temporal,
            height,
            width,
        }
    }

    /// Merged visual-token count `T * (H / merge) * (W / merge)`.
    ///
    /// Cached Transformers 4.49.0 uses `prod(grid_thw) // merge_size**2` for prompt pad expansion. That
    /// quotient equals this count when height and width are divisible by merge size.
    pub fn merged_token_count(
        self,
        spatial_merge_size: usize,
        kind: MropeVisualKind,
    ) -> Result<usize, MropePositionError> {
        merged_grid_token_count(self, spatial_merge_size, kind)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MropeTimingUnit {
    SecondsPerGridStep,
    TemporalTokensPerSecond,
}

impl std::fmt::Display for MropeTimingUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SecondsPerGridStep => f.write_str("seconds per grid step"),
            Self::TemporalTokensPerSecond => f.write_str("temporal tokens per second"),
        }
    }
}

/// Caller-resolved seconds represented by one step of a video's temporal grid.
///
/// The processor owns frame sampling and timestamp/FPS interpretation. This type only makes the unit
/// explicit and rejects values that cannot define a forward temporal scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MropeSecondsPerGridStep(f64);

impl Eq for MropeSecondsPerGridStep {}

impl MropeSecondsPerGridStep {
    pub fn new(value: f64) -> Result<Self, MropePositionError> {
        validate_timing(value, MropeTimingUnit::SecondsPerGridStep)?;
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }

    /// Resolve the processor-owned portion of Qwen2.5-VL video timing: Transformers 4.49.0 computes
    /// `second_per_grid_t = temporal_patch_size / fps`, and this keeps that division in one place.
    pub fn from_qwen2_5_vl_processor(
        temporal_patch_size: MropeTemporalPatchSize,
        sampled_frames_per_second: MropeSampledFramesPerSecond,
    ) -> Result<Self, MropeProcessorTimingError> {
        Ok(Self::new(
            temporal_patch_size.get() as f64 / sampled_frames_per_second.get(),
        )?)
    }
}

/// Qwen2.5-VL processor temporal patch size, in sampled frames per temporal vision-grid step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeTemporalPatchSize(usize);

impl MropeTemporalPatchSize {
    pub fn new(value: usize) -> Result<Self, MropeProcessorTimingError> {
        if value == 0 {
            return Err(MropeProcessorTimingError::ZeroTemporalPatchSize);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// Processor-selected sampled video rate for one video, in frames per second.
///
/// Upstream may broadcast one rate or accept one value per video. This type represents the already-selected
/// per-video value, keeping broadcast and video-count reconciliation in processor/model-input assembly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MropeSampledFramesPerSecond(f64);

impl Eq for MropeSampledFramesPerSecond {}

impl MropeSampledFramesPerSecond {
    pub fn new(value: f64) -> Result<Self, MropeProcessorTimingError> {
        if !value.is_finite() {
            return Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond);
        }
        if value <= 0.0 {
            return Err(MropeProcessorTimingError::NonPositiveSampledFramesPerSecond);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Explicit Qwen2.5-VL processor FPS input before scalar/list reconciliation.
///
/// This type assigns no processor default. `PerVideo` values remain in processor order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MropeProcessorFpsInput<'a> {
    Scalar(f64),
    PerVideo(&'a [f64]),
}

/// The fixed FPS used by the Qwen2.5-VL processor when its caller omits `videos_kwargs.fps`.
pub const QWEN2_5_VL_PROCESSOR_DEFAULT_FPS: f64 = 2.0;

/// Caller-resolved number of mRoPE temporal position tokens per second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MropeTemporalTokensPerSecond(f64);

impl Eq for MropeTemporalTokensPerSecond {}

impl MropeTemporalTokensPerSecond {
    pub fn new(value: f64) -> Result<Self, MropePositionError> {
        validate_timing(value, MropeTimingUnit::TemporalTokensPerSecond)?;
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Complete already-resolved Qwen2.5-VL temporal scaling metadata for one video grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeVideoTiming {
    pub seconds_per_grid_step: MropeSecondsPerGridStep,
    pub temporal_tokens_per_second: MropeTemporalTokensPerSecond,
}

impl MropeVideoTiming {
    pub const fn new(
        seconds_per_grid_step: MropeSecondsPerGridStep,
        temporal_tokens_per_second: MropeTemporalTokensPerSecond,
    ) -> Self {
        Self {
            seconds_per_grid_step,
            temporal_tokens_per_second,
        }
    }

    /// Resolve the Qwen2.5-VL processor formula for one video and compose it with model metadata.
    ///
    /// Transformers 4.49.0 computes `second_per_grid_t = temporal_patch_size / fps`. The two typed
    /// processor values must already reflect the caller's sampling decision; this function assigns no default
    /// and does not inspect video data or frame counts.
    pub fn from_qwen2_5_vl_processor(
        temporal_patch_size: MropeTemporalPatchSize,
        sampled_frames_per_second: MropeSampledFramesPerSecond,
        config: &MropeConfigMetadata,
    ) -> Result<Self, MropeProcessorTimingError> {
        let temporal_tokens_per_second = config
            .temporal_tokens_per_second()
            .ok_or(MropeProcessorTimingError::MissingTemporalTokensPerSecond)?;
        let seconds_per_grid_step = MropeSecondsPerGridStep::from_qwen2_5_vl_processor(
            temporal_patch_size,
            sampled_frames_per_second,
        )?;
        Ok(Self::new(seconds_per_grid_step, temporal_tokens_per_second))
    }
}

/// Reconcile an explicit Qwen2.5-VL processor FPS input into one timing per video.
///
/// Transformers 4.49.0 broadcasts a numeric FPS to the number of video grids and otherwise requires one
/// FPS value per grid. This pure boundary takes only that count, validates every explicit rate before
/// config-dependent timing resolution, and reuses the one-video timing constructor for each output.
pub fn assemble_qwen2_5_vl_video_timings(
    video_count: usize,
    temporal_patch_size: MropeTemporalPatchSize,
    fps: MropeProcessorFpsInput<'_>,
    config: &MropeConfigMetadata,
) -> Result<Vec<MropeVideoTiming>, MropeProcessorTimingError> {
    let sampled_rates = match fps {
        MropeProcessorFpsInput::Scalar(value) => {
            let sampled_rate = MropeSampledFramesPerSecond::new(value)?;
            vec![sampled_rate; video_count]
        }
        MropeProcessorFpsInput::PerVideo(values) => {
            if values.len() != video_count {
                return Err(
                    MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                        expected: video_count,
                        actual: values.len(),
                    },
                );
            }
            values
                .iter()
                .copied()
                .map(MropeSampledFramesPerSecond::new)
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    sampled_rates
        .into_iter()
        .map(|sampled_rate| {
            MropeVideoTiming::from_qwen2_5_vl_processor(temporal_patch_size, sampled_rate, config)
        })
        .collect()
}

/// Resolve the Qwen2.5-VL processor's absent-FPS default before assembling video timings.
///
/// Transformers 4.49.0 supplies `2.0` for an omitted `videos_kwargs.fps`, then applies the same scalar
/// broadcast or per-video list validation as [`assemble_qwen2_5_vl_video_timings`]. `Some` preserves the
/// caller's explicit scalar/list input; `None` is the only case that selects the default. This boundary has
/// no access to video data, grids, timestamps, or frame counts.
pub fn assemble_qwen2_5_vl_video_timings_with_default(
    video_count: usize,
    temporal_patch_size: MropeTemporalPatchSize,
    fps: Option<MropeProcessorFpsInput<'_>>,
    config: &MropeConfigMetadata,
) -> Result<Vec<MropeVideoTiming>, MropeProcessorTimingError> {
    assemble_qwen2_5_vl_video_timings(
        video_count,
        temporal_patch_size,
        fps.unwrap_or(MropeProcessorFpsInput::Scalar(
            QWEN2_5_VL_PROCESSOR_DEFAULT_FPS,
        )),
        config,
    )
}

/// Inputs owned by a pure Qwen2.5-VL model-input position assembly boundary.
///
/// Tokenization, grid construction, and tokenizer/config ID reconciliation remain caller-owned. The config
/// metadata is used for the model's temporal-token rate when video timings are needed.
#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlMropePositionInput<'a> {
    pub tokens: &'a [u32],
    pub special_tokens: MropeSpecialTokenIds,
    pub image_grids: &'a [MropeGrid],
    pub video_grids: &'a [MropeGrid],
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'a>>,
    pub config: &'a MropeConfigMetadata,
}

/// Inputs for the pure Qwen2.5-VL position assembly boundary before special-token ID reconciliation.
///
/// The processor/template records provide semantic marker strings. The tokenizer records map those strings to
/// IDs. Reconciliation produces the existing `MropeSpecialTokenIds`, then delegates to
/// [`assemble_qwen2_5_vl_mrope_positions`].
#[derive(Clone, Copy, Debug)]
pub struct Qwen25VlMropePositionSourceInput<'a> {
    pub tokens: &'a [u32],
    pub processor_special_tokens: &'a [MropeProcessorSpecialToken<'a>],
    pub tokenizer_special_tokens: &'a [MropeTokenizerSpecialToken<'a>],
    pub image_grids: &'a [MropeGrid],
    pub video_grids: &'a [MropeGrid],
    pub spatial_merge_size: usize,
    pub temporal_patch_size: MropeTemporalPatchSize,
    pub fps: Option<MropeProcessorFpsInput<'a>>,
    pub config: &'a MropeConfigMetadata,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropePositionAssemblyError {
    #[error(transparent)]
    Span(#[from] MropeSpanError),
    #[error(transparent)]
    ProcessorTiming(#[from] MropeProcessorTimingError),
    #[error(transparent)]
    Position(#[from] MropePositionError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropePositionSourceAssemblyError {
    #[error(transparent)]
    SpecialTokens(#[from] MropeSpecialTokenReconcileError),
    #[error(transparent)]
    Assembly(#[from] MropePositionAssemblyError),
}

/// Assemble Qwen2.5-VL mRoPE positions from an already-tokenized prompt. Span discovery runs before timing
/// resolution; discovered video segments become timestamped segments in video-list order, and the existing
/// position constructor builds the result.
pub fn assemble_qwen2_5_vl_mrope_positions(
    input: Qwen25VlMropePositionInput<'_>,
) -> Result<MropePositionIds, MropePositionAssemblyError> {
    let segments = discover_mrope_segments(
        input.tokens,
        input.special_tokens,
        input.image_grids,
        input.video_grids,
        input.spatial_merge_size,
    )?;
    let timings = assemble_qwen2_5_vl_video_timings_with_default(
        input.video_grids.len(),
        input.temporal_patch_size,
        input.fps,
        input.config,
    )?;

    let mut next_video_timing = timings.into_iter();
    let timestamped_segments = segments
        .into_iter()
        .map(|segment| match segment {
            MropeSegment::Video(grid) => MropeSegment::TimestampedVideo {
                grid,
                timing: next_video_timing
                    .next()
                    .expect("span discovery and video timing assembly must agree on video count"),
            },
            segment => segment,
        })
        .collect::<Vec<_>>();
    debug_assert!(next_video_timing.next().is_none());

    Ok(build_mrope_position_ids(
        &timestamped_segments,
        input.spatial_merge_size,
    )?)
}

/// Reconcile Qwen-VL special-token source metadata, then run the existing Qwen2.5-VL position assembly.
pub fn assemble_qwen2_5_vl_mrope_positions_from_sources(
    input: Qwen25VlMropePositionSourceInput<'_>,
) -> Result<MropePositionIds, MropePositionSourceAssemblyError> {
    let special_tokens = reconcile_mrope_special_token_ids(
        input.processor_special_tokens,
        input.tokenizer_special_tokens,
        input.config,
    )?;
    Ok(assemble_qwen2_5_vl_mrope_positions(
        Qwen25VlMropePositionInput {
            tokens: input.tokens,
            special_tokens,
            image_grids: input.image_grids,
            video_grids: input.video_grids,
            spatial_merge_size: input.spatial_merge_size,
            temporal_patch_size: input.temporal_patch_size,
            fps: input.fps,
            config: input.config,
        },
    )?)
}

/// An already-resolved prompt segment. Text lengths include every non-placeholder token in that run,
/// including the vision-start delimiter preceding a visual placeholder span and any following
/// processor-specific delimiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MropeSegment {
    Text(usize),
    Image(MropeGrid),
    /// Qwen2-VL compatibility path. Temporal positions have unit stride.
    Video(MropeGrid),
    /// Qwen2.5-VL path. The caller has already resolved timing and model-rate metadata.
    TimestampedVideo {
        grid: MropeGrid,
        timing: MropeVideoTiming,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropePosition {
    pub temporal: i32,
    pub height: i32,
    pub width: i32,
}

impl MropePosition {
    pub const fn collapsed(position: i32) -> Self {
        Self {
            temporal: position,
            height: position,
            width: position,
        }
    }

    pub const fn packed(self) -> [i32; 3] {
        [self.temporal, self.height, self.width]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MropePositionIds {
    axes: [Vec<i32>; 3],
    next_text_position: i32,
}

impl MropePositionIds {
    pub fn len(&self) -> usize {
        self.axes[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.axes[0].is_empty()
    }

    pub fn axes(&self) -> [&[i32]; 3] {
        [&self.axes[0], &self.axes[1], &self.axes[2]]
    }

    /// Axis-major `[3,L]` data, matching `Slot::MropePosition`'s prefill shape.
    pub fn packed(&self) -> Vec<i32> {
        let mut packed = Vec::with_capacity(self.len() * 3);
        packed.extend_from_slice(&self.axes[0]);
        packed.extend_from_slice(&self.axes[1]);
        packed.extend_from_slice(&self.axes[2]);
        packed
    }

    pub fn packed_for_len(&self, expected: usize) -> Result<Vec<i32>, MropePositionError> {
        if self.len() != expected {
            return Err(MropePositionError::LengthMismatch {
                expected,
                actual: self.len(),
            });
        }
        Ok(self.packed())
    }

    pub fn position(&self, index: usize) -> Result<MropePosition, MropePositionError> {
        if index >= self.len() {
            return Err(MropePositionError::IndexOutOfBounds {
                index,
                len: self.len(),
            });
        }
        Ok(MropePosition {
            temporal: self.axes[0][index],
            height: self.axes[1][index],
            width: self.axes[2][index],
        })
    }

    /// The first generated text token collapses all axes to this position. Subsequent generated text
    /// increments it normally, so decode does not need to retain the full prompt grid.
    pub const fn next_text_position(&self) -> MropePosition {
        MropePosition::collapsed(self.next_text_position)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropePositionError {
    #[error("mRoPE spatial merge size must be nonzero")]
    ZeroSpatialMerge,
    #[error("mRoPE {kind} grid dimension {axis} must be nonzero")]
    ZeroGridDimension {
        kind: MropeVisualKind,
        axis: &'static str,
    },
    #[error(
        "mRoPE {kind} grid {axis} dimension {dimension} is not divisible by spatial merge size {merge}"
    )]
    NonDivisibleGrid {
        kind: MropeVisualKind,
        axis: &'static str,
        dimension: usize,
        merge: usize,
    },
    #[error("mRoPE token-count calculation overflowed")]
    TokenCountOverflow,
    #[error("mRoPE {unit} must be finite")]
    NonFiniteTiming { unit: MropeTimingUnit },
    #[error("mRoPE {unit} must be positive")]
    NonPositiveTiming { unit: MropeTimingUnit },
    #[error(
        "mRoPE timestamp-scaled temporal position at grid index {temporal_index} exceeds the CPU-oracle exact integer limit {MAX_EXACT_MROPE_POSITION}"
    )]
    TemporalPositionOverflow { temporal_index: usize },
    #[error(
        "mRoPE position {position} exceeds the CPU-oracle exact integer limit {MAX_EXACT_MROPE_POSITION}"
    )]
    PositionOverflow { position: usize },
    #[error("mRoPE position length mismatch: graph expects {expected} tokens, layout has {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("mRoPE decode position index {index} is out of bounds for {len} prompt positions")]
    IndexOutOfBounds { index: usize, len: usize },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropeProcessorTimingError {
    #[error("mRoPE processor temporal patch size must be nonzero")]
    ZeroTemporalPatchSize,
    #[error("mRoPE processor sampled frames per second must be finite")]
    NonFiniteSampledFramesPerSecond,
    #[error("mRoPE processor sampled frames per second must be positive")]
    NonPositiveSampledFramesPerSecond,
    #[error(
        "mRoPE processor FPS count mismatch: expected {expected} values for the videos, got {actual}"
    )]
    SampledFramesPerSecondCountMismatch { expected: usize, actual: usize },
    #[error("mRoPE config metadata has no temporal tokens per second")]
    MissingTemporalTokensPerSecond,
    #[error(transparent)]
    SecondsPerGridStep(#[from] MropePositionError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropeSpanError {
    #[error("mRoPE special-token id {token_id} is assigned to both {first} and {second}")]
    DuplicateSpecialTokenId {
        token_id: u32,
        first: MropeSpecialTokenKind,
        second: MropeSpecialTokenKind,
    },
    #[error("mRoPE {marker} token at index {index} is outside a vision span")]
    MarkerOutsideSpan {
        marker: MropeSpecialTokenKind,
        index: usize,
    },
    #[error(
        "mRoPE vision span starting at index {start_index} contains a nested vision start at index {index}"
    )]
    NestedVisionStart { start_index: usize, index: usize },
    #[error(
        "mRoPE vision start at index {start_index} has no image or video marker at index {index}"
    )]
    MissingVisualMarker { start_index: usize, index: usize },
    #[error(
        "mRoPE vision span starting at index {start_index} mixes {expected} and {actual} placeholders at index {index}"
    )]
    MixedVisualMarkers {
        start_index: usize,
        expected: MropeVisualKind,
        actual: MropeVisualKind,
        index: usize,
    },
    #[error(
        "mRoPE {kind} span starting at index {start_index} has no grid at {kind} grid index {grid_index}"
    )]
    MissingGrid {
        kind: MropeVisualKind,
        start_index: usize,
        grid_index: usize,
    },
    #[error("mRoPE has extra {kind} grids: first unused index {first_unused}, total {total}")]
    ExtraGrids {
        kind: MropeVisualKind,
        first_unused: usize,
        total: usize,
    },
    #[error(
        "mRoPE {kind} span starting at index {start_index} has {actual} placeholders; grid requires {expected}"
    )]
    PlaceholderCountMismatch {
        kind: MropeVisualKind,
        start_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Grid(#[from] MropePositionError),
}

/// Discover ordered mRoPE segments from an already-tokenized prompt.
///
/// A visual span is `vision_start` followed by exactly the merged grid's number of homogeneous image or
/// video placeholders. The start token and any following processor-specific delimiter remain part of text
/// runs; only the placeholder run becomes a visual segment. Image and video grids are consumed independently
/// in marker order. Discovery validates the complete prompt before returning, so an error never exposes a
/// partial segment list.
pub fn discover_mrope_segments(
    tokens: &[u32],
    special_tokens: MropeSpecialTokenIds,
    image_grids: &[MropeGrid],
    video_grids: &[MropeGrid],
    spatial_merge_size: usize,
) -> Result<Vec<MropeSegment>, MropeSpanError> {
    special_tokens.validate()?;
    if spatial_merge_size == 0 {
        return Err(MropePositionError::ZeroSpatialMerge.into());
    }

    let mut segments = Vec::new();
    let mut text_start = 0usize;
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut index = 0usize;

    while index < tokens.len() {
        match special_tokens.classify(tokens[index]) {
            None => index += 1,
            Some(marker @ (MropeSpecialTokenKind::Image | MropeSpecialTokenKind::Video)) => {
                return Err(MropeSpanError::MarkerOutsideSpan { marker, index });
            }
            Some(MropeSpecialTokenKind::VisionStart) => {
                let start_index = index;
                let marker_index = index + 1;
                let Some(&first_token) = tokens.get(marker_index) else {
                    return Err(MropeSpanError::MissingVisualMarker {
                        start_index,
                        index: marker_index,
                    });
                };
                let kind = match special_tokens.classify(first_token) {
                    Some(MropeSpecialTokenKind::Image) => MropeVisualKind::Image,
                    Some(MropeSpecialTokenKind::Video) => MropeVisualKind::Video,
                    Some(MropeSpecialTokenKind::VisionStart) => {
                        return Err(MropeSpanError::NestedVisionStart {
                            start_index,
                            index: marker_index,
                        });
                    }
                    None => {
                        return Err(MropeSpanError::MissingVisualMarker {
                            start_index,
                            index: marker_index,
                        });
                    }
                };

                let grid = match kind {
                    MropeVisualKind::Image => {
                        let grid_index = image_index;
                        let Some(&grid) = image_grids.get(grid_index) else {
                            return Err(MropeSpanError::MissingGrid {
                                kind,
                                start_index,
                                grid_index,
                            });
                        };
                        image_index += 1;
                        grid
                    }
                    MropeVisualKind::Video => {
                        let grid_index = video_index;
                        let Some(&grid) = video_grids.get(grid_index) else {
                            return Err(MropeSpanError::MissingGrid {
                                kind,
                                start_index,
                                grid_index,
                            });
                        };
                        video_index += 1;
                        grid
                    }
                };
                let expected = merged_grid_token_count(grid, spatial_merge_size, kind)?;
                for offset in 0..expected {
                    let marker_position = marker_index + offset;
                    match tokens
                        .get(marker_position)
                        .and_then(|&token| special_tokens.classify(token))
                    {
                        Some(MropeSpecialTokenKind::Image) if kind == MropeVisualKind::Image => {}
                        Some(MropeSpecialTokenKind::Video) if kind == MropeVisualKind::Video => {}
                        Some(MropeSpecialTokenKind::Image) => {
                            return Err(MropeSpanError::MixedVisualMarkers {
                                start_index,
                                expected: kind,
                                actual: MropeVisualKind::Image,
                                index: marker_position,
                            });
                        }
                        Some(MropeSpecialTokenKind::Video) => {
                            return Err(MropeSpanError::MixedVisualMarkers {
                                start_index,
                                expected: kind,
                                actual: MropeVisualKind::Video,
                                index: marker_position,
                            });
                        }
                        Some(MropeSpecialTokenKind::VisionStart) => {
                            return Err(MropeSpanError::NestedVisionStart {
                                start_index,
                                index: marker_position,
                            });
                        }
                        None => {
                            return Err(MropeSpanError::PlaceholderCountMismatch {
                                kind,
                                start_index,
                                expected,
                                actual: offset,
                            });
                        }
                    }
                }

                append_text_segment(&mut segments, index + 1 - text_start);
                segments.push(match kind {
                    MropeVisualKind::Image => MropeSegment::Image(grid),
                    MropeVisualKind::Video => MropeSegment::Video(grid),
                });
                text_start = marker_index + expected;
                index = text_start;
            }
        }
    }

    append_text_segment(&mut segments, tokens.len() - text_start);
    if image_index != image_grids.len() {
        return Err(MropeSpanError::ExtraGrids {
            kind: MropeVisualKind::Image,
            first_unused: image_index,
            total: image_grids.len(),
        });
    }
    if video_index != video_grids.len() {
        return Err(MropeSpanError::ExtraGrids {
            kind: MropeVisualKind::Video,
            first_unused: video_index,
            total: video_grids.len(),
        });
    }
    Ok(segments)
}

fn append_text_segment(segments: &mut Vec<MropeSegment>, len: usize) {
    if len == 0 {
        return;
    }
    if let Some(MropeSegment::Text(previous)) = segments.last_mut() {
        *previous += len;
    } else {
        segments.push(MropeSegment::Text(len));
    }
}

/// Construct Qwen-VL mRoPE ids from an already-resolved prompt layout.
///
/// Images and `MropeSegment::Video` use Qwen2-VL unit temporal stride. A `TimestampedVideo` consumes
/// explicit timing units already resolved by the caller and applies Qwen2.5-VL temporal scaling. Both
/// paths produce the same `MropePositionIds` contract used by binders and schedulers.
pub fn build_mrope_position_ids(
    segments: &[MropeSegment],
    spatial_merge_size: usize,
) -> Result<MropePositionIds, MropePositionError> {
    if spatial_merge_size == 0 {
        return Err(MropePositionError::ZeroSpatialMerge);
    }

    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut base = 0usize;

    for &segment in segments {
        match segment {
            MropeSegment::Text(len) => {
                let end = base
                    .checked_add(len)
                    .ok_or(MropePositionError::TokenCountOverflow)?;
                check_position(end)?;
                for position in base..end {
                    let position = position as i32;
                    for axis in &mut axes {
                        axis.push(position);
                    }
                }
                base = end;
            }
            MropeSegment::Image(grid) => append_grid(
                &mut axes,
                &mut base,
                grid,
                spatial_merge_size,
                MropeVisualKind::Image,
            )?,
            MropeSegment::Video(grid) => append_grid(
                &mut axes,
                &mut base,
                grid,
                spatial_merge_size,
                MropeVisualKind::Video,
            )?,
            MropeSegment::TimestampedVideo { grid, timing } => append_timestamped_video_grid(
                &mut axes,
                &mut base,
                grid,
                spatial_merge_size,
                timing,
            )?,
        }
    }

    check_position(base)?;
    Ok(MropePositionIds {
        axes,
        next_text_position: base as i32,
    })
}

fn append_timestamped_video_grid(
    axes: &mut [Vec<i32>; 3],
    base: &mut usize,
    grid: MropeGrid,
    merge: usize,
    timing: MropeVideoTiming,
) -> Result<(), MropePositionError> {
    let (height, width, count) = merged_grid_shape(grid, merge, MropeVisualKind::Video)?;
    axes[0]
        .len()
        .checked_add(count)
        .ok_or(MropePositionError::TokenCountOverflow)?;

    // Validate the largest temporal coordinate and the complete base transition before appending any
    // element. Positive timing makes the temporal mapping monotonic, so every earlier plane is then safe.
    let last_temporal = timestamped_temporal_position(grid.temporal - 1, timing)?;
    let temporal_span =
        last_temporal
            .checked_add(1)
            .ok_or(MropePositionError::TemporalPositionOverflow {
                temporal_index: grid.temporal - 1,
            })?;
    let next_base = (*base)
        .checked_add(temporal_span.max(height).max(width))
        .ok_or(MropePositionError::TokenCountOverflow)?;
    check_position(next_base)?;

    for temporal in 0..grid.temporal {
        let temporal = timestamped_temporal_position(temporal, timing)?;
        for row in 0..height {
            for col in 0..width {
                axes[0].push((*base + temporal) as i32);
                axes[1].push((*base + row) as i32);
                axes[2].push((*base + col) as i32);
            }
        }
    }
    *base = next_base;
    Ok(())
}

fn timestamped_temporal_position(
    temporal_index: usize,
    timing: MropeVideoTiming,
) -> Result<usize, MropePositionError> {
    let scaled = (temporal_index as f64 * timing.seconds_per_grid_step.get())
        * timing.temporal_tokens_per_second.get();
    if !scaled.is_finite() || scaled > MAX_EXACT_MROPE_POSITION as f64 {
        return Err(MropePositionError::TemporalPositionOverflow { temporal_index });
    }
    Ok(scaled.floor() as usize)
}

fn validate_timing(value: f64, unit: MropeTimingUnit) -> Result<(), MropePositionError> {
    if !value.is_finite() {
        return Err(MropePositionError::NonFiniteTiming { unit });
    }
    if value <= 0.0 {
        return Err(MropePositionError::NonPositiveTiming { unit });
    }
    Ok(())
}

fn append_grid(
    axes: &mut [Vec<i32>; 3],
    base: &mut usize,
    grid: MropeGrid,
    merge: usize,
    kind: MropeVisualKind,
) -> Result<(), MropePositionError> {
    let (height, width, count) = merged_grid_shape(grid, merge, kind)?;
    axes[0]
        .len()
        .checked_add(count)
        .ok_or(MropePositionError::TokenCountOverflow)?;

    let next_base = (*base)
        .checked_add(grid.temporal.max(height).max(width))
        .ok_or(MropePositionError::TokenCountOverflow)?;
    check_position(next_base)?;

    for temporal in 0..grid.temporal {
        for row in 0..height {
            for col in 0..width {
                axes[0].push((*base + temporal) as i32);
                axes[1].push((*base + row) as i32);
                axes[2].push((*base + col) as i32);
            }
        }
    }
    *base = next_base;
    Ok(())
}

fn merged_grid_token_count(
    grid: MropeGrid,
    merge: usize,
    kind: MropeVisualKind,
) -> Result<usize, MropePositionError> {
    Ok(merged_grid_shape(grid, merge, kind)?.2)
}

fn merged_grid_shape(
    grid: MropeGrid,
    merge: usize,
    kind: MropeVisualKind,
) -> Result<(usize, usize, usize), MropePositionError> {
    if merge == 0 {
        return Err(MropePositionError::ZeroSpatialMerge);
    }
    for (axis, dimension) in [
        ("temporal", grid.temporal),
        ("height", grid.height),
        ("width", grid.width),
    ] {
        if dimension == 0 {
            return Err(MropePositionError::ZeroGridDimension { kind, axis });
        }
    }
    for (axis, dimension) in [("height", grid.height), ("width", grid.width)] {
        if dimension % merge != 0 {
            return Err(MropePositionError::NonDivisibleGrid {
                kind,
                axis,
                dimension,
                merge,
            });
        }
    }

    let height = grid.height / merge;
    let width = grid.width / merge;
    let count = grid
        .temporal
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or(MropePositionError::TokenCountOverflow)?;
    Ok((height, width, count))
}

fn check_position(position: usize) -> Result<(), MropePositionError> {
    if position > MAX_EXACT_MROPE_POSITION {
        return Err(MropePositionError::PositionOverflow { position });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VISION_START: u32 = 10;
    const IMAGE: u32 = 12;
    const VIDEO: u32 = 13;
    const VISION_START_TOKEN: &str = "<|vision_start|>";
    const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
    const VIDEO_PAD_TOKEN: &str = "<|video_pad|>";

    fn special_tokens() -> MropeSpecialTokenIds {
        MropeSpecialTokenIds::new(VISION_START, IMAGE, VIDEO)
    }

    fn processor_special_tokens() -> [MropeProcessorSpecialToken<'static>; 3] {
        [
            MropeProcessorSpecialToken::new(MropeSpecialTokenKind::VisionStart, VISION_START_TOKEN),
            MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, IMAGE_PAD_TOKEN),
            MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Video, VIDEO_PAD_TOKEN),
        ]
    }

    fn tokenizer_special_tokens() -> [MropeTokenizerSpecialToken<'static>; 3] {
        [
            MropeTokenizerSpecialToken::new(VISION_START_TOKEN, VISION_START),
            MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, IMAGE),
            MropeTokenizerSpecialToken::new(VIDEO_PAD_TOKEN, VIDEO),
        ]
    }

    fn video_timing(
        seconds_per_grid_step: f64,
        temporal_tokens_per_second: f64,
    ) -> MropeVideoTiming {
        MropeVideoTiming::new(
            MropeSecondsPerGridStep::new(seconds_per_grid_step).unwrap(),
            MropeTemporalTokensPerSecond::new(temporal_tokens_per_second).unwrap(),
        )
    }

    fn qwen_vl_config_json(model_type: &str, token_ids: [u64; 3], rate: Option<&str>) -> String {
        let vision_rate = rate
            .map(|value| format!(", \"tokens_per_second\": {value}"))
            .unwrap_or_default();
        format!(
            r#"{{
                "model_type": "{model_type}",
                "vocab_size": 152064,
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 32768,
                "vision_start_token_id": {},
                "image_token_id": {},
                "video_token_id": {},
                "rope_scaling": {{
                    "type": "mrope",
                    "mrope_section": [16, 24, 24]
                }},
                "vision_config": {{"spatial_merge_size": 2{vision_rate}}}
            }}"#,
            token_ids[0], token_ids[1], token_ids[2]
        )
    }

    #[test]
    fn mrope_config_metadata_constructs_existing_domain_types_exactly() {
        let qwen2 = qwen_vl_config_json("qwen2_vl", [151652, 151655, 151656], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(qwen2.as_bytes()).unwrap();
        assert_eq!(metadata.mrope_section(), [16, 24, 24]);
        assert_eq!(
            metadata.special_token_ids(),
            MropeSpecialTokenIds::new(151652, 151655, 151656)
        );
        assert_eq!(metadata.temporal_tokens_per_second(), None);

        let qwen25 = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(qwen25.as_bytes()).unwrap();
        assert_eq!(
            metadata.temporal_tokens_per_second().map(|rate| rate.get()),
            Some(4.0)
        );
    }

    #[test]
    fn mrope_config_metadata_rejects_duplicate_special_ids_before_construction() {
        for token_ids in [
            [151652, 151652, 151656],
            [151652, 151655, 151652],
            [151652, 151655, 151655],
        ] {
            let json = qwen_vl_config_json("qwen2_5_vl", token_ids, Some("4"));
            assert!(matches!(
                MropeConfigMetadata::from_hf_config_slice(json.as_bytes()),
                Err(MropeConfigMetadataError::SpecialTokenIds(
                    MropeSpanError::DuplicateSpecialTokenId { .. }
                ))
            ));
        }
    }

    #[test]
    fn mrope_config_metadata_rejects_invalid_temporal_rates_before_construction() {
        for value in ["0", "-0.0", "-1", "-1e-400"] {
            let json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some(value));
            assert!(matches!(
                MropeConfigMetadata::from_hf_config_slice(json.as_bytes()),
                Err(MropeConfigMetadataError::Timing(
                    MropePositionError::NonPositiveTiming {
                        unit: MropeTimingUnit::TemporalTokensPerSecond
                    }
                ))
            ));
        }

        for value in ["NaN", "Infinity", "-Infinity", "1e400"] {
            let json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some(value));
            assert!(matches!(
                MropeConfigMetadata::from_hf_config_slice(json.as_bytes()),
                Err(MropeConfigMetadataError::Load(
                    poot_load::qwen_vl::QwenVlHfMetadataError::Json(_)
                ))
            ));
        }
    }

    #[test]
    fn mrope_qwen25_processor_timing_matches_upstream_formula_and_timestamp_path() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let sampled_fps = MropeSampledFramesPerSecond::new(3.0).unwrap();
        let timing =
            MropeVideoTiming::from_qwen2_5_vl_processor(patch_size, sampled_fps, &metadata)
                .unwrap();

        // Transformers 4.49.0 processor: second_per_grid_t = temporal_patch_size / fps.
        assert_eq!(
            timing.seconds_per_grid_step.get().to_bits(),
            (2.0f64 / 3.0).to_bits()
        );
        assert_eq!(timing.temporal_tokens_per_second.get(), 4.0);

        let ids = build_mrope_position_ids(
            &[
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(4, 1, 1),
                    timing,
                },
                MropeSegment::Text(1),
            ],
            1,
        )
        .unwrap();
        assert_eq!(
            ids.axes(),
            [&[0, 2, 5, 8, 9], &[0, 0, 0, 0, 9], &[0, 0, 0, 0, 9]]
        );
    }

    #[test]
    fn mrope_qwen25_processor_timing_resolves_multiple_videos_per_video() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let sampled_fps = [4.0, 1.6];
        let timings: Vec<_> = sampled_fps
            .into_iter()
            .map(|fps| {
                MropeVideoTiming::from_qwen2_5_vl_processor(
                    patch_size,
                    MropeSampledFramesPerSecond::new(fps).unwrap(),
                    &metadata,
                )
                .unwrap()
            })
            .collect();

        for (timing, fps) in timings.iter().zip(sampled_fps) {
            assert_eq!(
                timing.seconds_per_grid_step.get().to_bits(),
                (2.0 / fps).to_bits()
            );
        }

        let ids = build_mrope_position_ids(
            &[
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(3, 1, 1),
                    timing: timings[0],
                },
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(3, 1, 1),
                    timing: timings[1],
                },
                MropeSegment::Text(1),
            ],
            1,
        )
        .unwrap();
        assert_eq!(
            ids.axes(),
            [
                &[0, 2, 4, 5, 10, 15, 16],
                &[0, 0, 0, 5, 5, 5, 16],
                &[0, 0, 0, 5, 5, 5, 16]
            ]
        );
    }

    #[test]
    fn mrope_qwen25_processor_input_broadcasts_scalar_and_preserves_list_order() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let scalar = assemble_qwen2_5_vl_video_timings(
            3,
            patch_size,
            MropeProcessorFpsInput::Scalar(2.5),
            &metadata,
        )
        .unwrap();
        assert_eq!(scalar.len(), 3);
        for timing in scalar {
            assert_eq!(
                timing.seconds_per_grid_step.get().to_bits(),
                (2.0f64 / 2.5).to_bits()
            );
            assert_eq!(timing.temporal_tokens_per_second.get(), 4.0);
        }

        let fps = [4.0, 1.6, 8.0];
        let per_video = assemble_qwen2_5_vl_video_timings(
            fps.len(),
            patch_size,
            MropeProcessorFpsInput::PerVideo(&fps),
            &metadata,
        )
        .unwrap();
        let quotients: Vec<_> = per_video
            .iter()
            .map(|timing| timing.seconds_per_grid_step.get().to_bits())
            .collect();
        assert_eq!(
            quotients,
            fps.map(|value| (2.0f64 / value).to_bits()).to_vec()
        );
    }

    #[test]
    fn mrope_qwen25_processor_input_checks_list_length_before_values_or_config() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        for (fps, expected, actual) in [(&[f64::NAN][..], 2, 1), (&[f64::NAN, 0.0, -1.0][..], 2, 3)]
        {
            assert_eq!(
                assemble_qwen2_5_vl_video_timings(
                    expected,
                    patch_size,
                    MropeProcessorFpsInput::PerVideo(fps),
                    &metadata,
                ),
                Err(
                    MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                        expected,
                        actual,
                    }
                )
            );
        }
    }

    #[test]
    fn mrope_qwen25_processor_input_rejects_invalid_rates_in_input_order() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                assemble_qwen2_5_vl_video_timings(
                    1,
                    patch_size,
                    MropeProcessorFpsInput::Scalar(value),
                    &metadata,
                ),
                Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond)
            );
        }
        for value in [0.0, -0.0, -1.0] {
            assert_eq!(
                assemble_qwen2_5_vl_video_timings(
                    1,
                    patch_size,
                    MropeProcessorFpsInput::Scalar(value),
                    &metadata,
                ),
                Err(MropeProcessorTimingError::NonPositiveSampledFramesPerSecond)
            );
        }

        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                2,
                patch_size,
                MropeProcessorFpsInput::PerVideo(&[0.0, f64::NAN]),
                &metadata,
            ),
            Err(MropeProcessorTimingError::NonPositiveSampledFramesPerSecond)
        );

        let missing_rate = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], None);
        let missing_rate =
            MropeConfigMetadata::from_hf_config_slice(missing_rate.as_bytes()).unwrap();
        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                2,
                patch_size,
                MropeProcessorFpsInput::PerVideo(&[2.0, f64::NAN]),
                &missing_rate,
            ),
            Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond)
        );
    }

    #[test]
    fn mrope_qwen25_processor_input_zero_videos_matches_upstream_count_behavior() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                0,
                patch_size,
                MropeProcessorFpsInput::Scalar(2.0),
                &metadata,
            ),
            Ok(Vec::new())
        );
        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                0,
                patch_size,
                MropeProcessorFpsInput::PerVideo(&[]),
                &metadata,
            ),
            Ok(Vec::new())
        );
        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                0,
                patch_size,
                MropeProcessorFpsInput::PerVideo(&[2.0]),
                &metadata,
            ),
            Err(
                MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                    expected: 0,
                    actual: 1,
                }
            )
        );
        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                0,
                patch_size,
                MropeProcessorFpsInput::Scalar(f64::NAN),
                &metadata,
            ),
            Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond)
        );
    }

    #[test]
    fn mrope_qwen25_processor_default_fps_matches_explicit_scalar() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        assert_eq!(QWEN2_5_VL_PROCESSOR_DEFAULT_FPS.to_bits(), 2.0f64.to_bits());
        for video_count in [0, 1, 3] {
            let default = assemble_qwen2_5_vl_video_timings_with_default(
                video_count,
                patch_size,
                None,
                &metadata,
            )
            .unwrap();
            let explicit = assemble_qwen2_5_vl_video_timings(
                video_count,
                patch_size,
                MropeProcessorFpsInput::Scalar(2.0),
                &metadata,
            )
            .unwrap();
            assert_eq!(default, explicit);
        }

        let explicit_scalar = assemble_qwen2_5_vl_video_timings_with_default(
            3,
            patch_size,
            Some(MropeProcessorFpsInput::Scalar(2.5)),
            &metadata,
        )
        .unwrap();
        let direct_scalar = assemble_qwen2_5_vl_video_timings(
            3,
            patch_size,
            MropeProcessorFpsInput::Scalar(2.5),
            &metadata,
        )
        .unwrap();
        assert_eq!(explicit_scalar, direct_scalar);

        let fps = [4.0, 1.6, 8.0];
        let explicit_list = assemble_qwen2_5_vl_video_timings_with_default(
            fps.len(),
            patch_size,
            Some(MropeProcessorFpsInput::PerVideo(&fps)),
            &metadata,
        )
        .unwrap();
        let direct_list = assemble_qwen2_5_vl_video_timings(
            fps.len(),
            patch_size,
            MropeProcessorFpsInput::PerVideo(&fps),
            &metadata,
        )
        .unwrap();
        assert_eq!(explicit_list, direct_list);
    }

    #[test]
    fn mrope_qwen25_processor_default_boundary_preserves_explicit_errors_atomically() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        assert_eq!(
            assemble_qwen2_5_vl_video_timings_with_default(
                2,
                patch_size,
                Some(MropeProcessorFpsInput::PerVideo(&[2.0])),
                &metadata,
            ),
            Err(
                MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                    expected: 2,
                    actual: 1,
                }
            )
        );
        assert_eq!(
            assemble_qwen2_5_vl_video_timings_with_default(
                2,
                patch_size,
                Some(MropeProcessorFpsInput::PerVideo(&[2.0, f64::NAN])),
                &metadata,
            ),
            Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond)
        );
        assert_eq!(
            assemble_qwen2_5_vl_video_timings_with_default(
                1,
                patch_size,
                Some(MropeProcessorFpsInput::Scalar(2.0)),
                &metadata,
            ),
            Err(MropeProcessorTimingError::MissingTemporalTokensPerSecond)
        );
    }

    #[test]
    fn mrope_qwen25_processor_input_requires_config_rate_only_for_output_videos() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();

        assert_eq!(
            assemble_qwen2_5_vl_video_timings(
                2,
                MropeTemporalPatchSize::new(2).unwrap(),
                MropeProcessorFpsInput::Scalar(2.0),
                &metadata,
            ),
            Err(MropeProcessorTimingError::MissingTemporalTokensPerSecond)
        );
    }

    #[test]
    fn mrope_qwen25_processor_timing_rejects_invalid_inputs_overflow_and_missing_rate() {
        assert_eq!(
            MropeTemporalPatchSize::new(0),
            Err(MropeProcessorTimingError::ZeroTemporalPatchSize)
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                MropeSampledFramesPerSecond::new(value),
                Err(MropeProcessorTimingError::NonFiniteSampledFramesPerSecond)
            );
        }
        for value in [0.0, -0.0, -1.0] {
            assert_eq!(
                MropeSampledFramesPerSecond::new(value),
                Err(MropeProcessorTimingError::NonPositiveSampledFramesPerSecond)
            );
        }

        let config_json = qwen_vl_config_json("qwen2_5_vl", [151652, 151655, 151656], Some("4"));
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        assert_eq!(
            MropeVideoTiming::from_qwen2_5_vl_processor(
                MropeTemporalPatchSize::new(usize::MAX).unwrap(),
                MropeSampledFramesPerSecond::new(f64::MIN_POSITIVE).unwrap(),
                &metadata,
            ),
            Err(MropeProcessorTimingError::SecondsPerGridStep(
                MropePositionError::NonFiniteTiming {
                    unit: MropeTimingUnit::SecondsPerGridStep,
                }
            ))
        );

        for model_type in ["qwen2_vl", "qwen2_5_vl"] {
            let config_json = qwen_vl_config_json(model_type, [151652, 151655, 151656], None);
            let metadata =
                MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
            assert_eq!(
                MropeVideoTiming::from_qwen2_5_vl_processor(
                    MropeTemporalPatchSize::new(2).unwrap(),
                    MropeSampledFramesPerSecond::new(2.0).unwrap(),
                    &metadata,
                ),
                Err(MropeProcessorTimingError::MissingTemporalTokensPerSecond)
            );
        }
    }

    /// Independent Qwen2-VL unit-stride reference; shares no production helpers.
    fn legacy_unit_stride_reference(
        segments: &[MropeSegment],
        merge: usize,
    ) -> ([Vec<i32>; 3], i32) {
        let mut axes = [Vec::new(), Vec::new(), Vec::new()];
        let mut base = 0usize;
        for &segment in segments {
            match segment {
                MropeSegment::Text(len) => {
                    for position in base..base + len {
                        axes[0].push(position as i32);
                        axes[1].push(position as i32);
                        axes[2].push(position as i32);
                    }
                    base += len;
                }
                MropeSegment::Image(grid) | MropeSegment::Video(grid) => {
                    let height = grid.height / merge;
                    let width = grid.width / merge;
                    for temporal in 0..grid.temporal {
                        for row in 0..height {
                            for col in 0..width {
                                axes[0].push((base + temporal) as i32);
                                axes[1].push((base + row) as i32);
                                axes[2].push((base + col) as i32);
                            }
                        }
                    }
                    base += grid.temporal.max(height).max(width);
                }
                MropeSegment::TimestampedVideo { .. } => {
                    panic!("legacy reference accepts only pre-amendment segment variants")
                }
            }
        }
        (axes, base as i32)
    }

    #[test]
    fn mrope_span_discovery_emits_exact_all_text_and_mixed_layouts() {
        assert_eq!(
            discover_mrope_segments(&[1, 2, 3], special_tokens(), &[], &[], 2),
            Ok(vec![MropeSegment::Text(3)])
        );
        assert_eq!(
            discover_mrope_segments(&[], special_tokens(), &[], &[], 2),
            Ok(vec![])
        );

        let image = MropeGrid::new(1, 2, 4);
        let video = MropeGrid::new(2, 2, 2);
        let tokens = [
            1,
            VISION_START,
            IMAGE,
            IMAGE,
            2,
            VISION_START,
            VIDEO,
            VIDEO,
            3,
        ];
        let segments =
            discover_mrope_segments(&tokens, special_tokens(), &[image], &[video], 2).unwrap();
        assert_eq!(
            segments,
            vec![
                MropeSegment::Text(2),
                MropeSegment::Image(image),
                MropeSegment::Text(2),
                MropeSegment::Video(video),
                MropeSegment::Text(1),
            ]
        );
        assert_eq!(
            build_mrope_position_ids(&segments, 2).unwrap().len(),
            tokens.len()
        );
    }

    #[test]
    fn mrope_span_discovery_rejects_ambiguous_and_orphan_markers() {
        assert_eq!(
            discover_mrope_segments(
                &[],
                MropeSpecialTokenIds::new(VISION_START, IMAGE, IMAGE),
                &[],
                &[],
                2,
            ),
            Err(MropeSpanError::DuplicateSpecialTokenId {
                token_id: IMAGE,
                first: MropeSpecialTokenKind::Image,
                second: MropeSpecialTokenKind::Video,
            })
        );

        for (token, marker) in [
            (IMAGE, MropeSpecialTokenKind::Image),
            (VIDEO, MropeSpecialTokenKind::Video),
        ] {
            assert_eq!(
                discover_mrope_segments(&[1, token], special_tokens(), &[], &[], 2),
                Err(MropeSpanError::MarkerOutsideSpan { marker, index: 1 })
            );
        }
    }

    #[test]
    fn mrope_span_discovery_rejects_malformed_or_nested_spans() {
        assert_eq!(
            discover_mrope_segments(&[VISION_START, VISION_START], special_tokens(), &[], &[], 2),
            Err(MropeSpanError::NestedVisionStart {
                start_index: 0,
                index: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(&[VISION_START], special_tokens(), &[], &[], 2),
            Err(MropeSpanError::MissingVisualMarker {
                start_index: 0,
                index: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(&[VISION_START, 99], special_tokens(), &[], &[], 2,),
            Err(MropeSpanError::MissingVisualMarker {
                start_index: 0,
                index: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(
                &[VISION_START, IMAGE, VISION_START],
                special_tokens(),
                &[MropeGrid::new(1, 2, 4)],
                &[],
                2,
            ),
            Err(MropeSpanError::NestedVisionStart {
                start_index: 0,
                index: 2,
            })
        );
        assert_eq!(
            discover_mrope_segments(
                &[VISION_START, IMAGE, VIDEO],
                special_tokens(),
                &[MropeGrid::new(1, 2, 4)],
                &[],
                2,
            ),
            Err(MropeSpanError::MixedVisualMarkers {
                start_index: 0,
                expected: MropeVisualKind::Image,
                actual: MropeVisualKind::Video,
                index: 2,
            })
        );
        assert_eq!(
            discover_mrope_segments(
                &[VISION_START, IMAGE, 99],
                special_tokens(),
                &[MropeGrid::new(1, 2, 4)],
                &[],
                2,
            ),
            Err(MropeSpanError::PlaceholderCountMismatch {
                kind: MropeVisualKind::Image,
                start_index: 0,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn mrope_span_discovery_rejects_missing_extra_and_mismatched_grids() {
        let image_span = [VISION_START, IMAGE];
        let video_span = [VISION_START, VIDEO];
        assert_eq!(
            discover_mrope_segments(&image_span, special_tokens(), &[], &[], 2),
            Err(MropeSpanError::MissingGrid {
                kind: MropeVisualKind::Image,
                start_index: 0,
                grid_index: 0,
            })
        );
        assert_eq!(
            discover_mrope_segments(&video_span, special_tokens(), &[], &[], 2),
            Err(MropeSpanError::MissingGrid {
                kind: MropeVisualKind::Video,
                start_index: 0,
                grid_index: 0,
            })
        );
        assert_eq!(
            discover_mrope_segments(&[], special_tokens(), &[MropeGrid::new(1, 2, 2)], &[], 2,),
            Err(MropeSpanError::ExtraGrids {
                kind: MropeVisualKind::Image,
                first_unused: 0,
                total: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(&[], special_tokens(), &[], &[MropeGrid::new(1, 2, 2)], 2,),
            Err(MropeSpanError::ExtraGrids {
                kind: MropeVisualKind::Video,
                first_unused: 0,
                total: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(
                &image_span,
                special_tokens(),
                &[MropeGrid::new(1, 2, 4)],
                &[],
                2,
            ),
            Err(MropeSpanError::PlaceholderCountMismatch {
                kind: MropeVisualKind::Image,
                start_index: 0,
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            discover_mrope_segments(
                &[VISION_START, IMAGE, IMAGE, IMAGE],
                special_tokens(),
                &[MropeGrid::new(1, 2, 4)],
                &[],
                2,
            ),
            Err(MropeSpanError::MarkerOutsideSpan {
                marker: MropeSpecialTokenKind::Image,
                index: 3,
            })
        );
        assert_eq!(
            discover_mrope_segments(&[], special_tokens(), &[], &[], 0),
            Err(MropeSpanError::Grid(MropePositionError::ZeroSpatialMerge))
        );
    }

    #[test]
    fn mrope_all_text_is_three_identical_running_axes() {
        let ids = build_mrope_position_ids(&[MropeSegment::Text(5)], 2).unwrap();
        let running = [0, 1, 2, 3, 4];
        assert_eq!(ids.axes(), [&running, &running, &running]);
        assert_eq!(ids.packed(), running.repeat(3));
        assert_eq!(ids.next_text_position(), MropePosition::collapsed(5));
    }

    #[test]
    fn mrope_mixed_text_image_video_grid_positions_are_exact() {
        let ids = build_mrope_position_ids(
            &[
                MropeSegment::Text(2),
                MropeSegment::Image(MropeGrid::new(1, 4, 6)),
                MropeSegment::Text(1),
                MropeSegment::Video(MropeGrid::new(2, 2, 4)),
                MropeSegment::Text(2),
            ],
            2,
        )
        .unwrap();

        let temporal = [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 6, 7, 7, 8, 9];
        let height = [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 6, 6, 6, 8, 9];
        let width = [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 6, 7, 8, 9];
        assert_eq!(ids.axes(), [&temporal, &height, &width]);
        assert_eq!(ids.len(), 15);
        assert_eq!(ids.next_text_position(), MropePosition::collapsed(10));
        assert_eq!(
            ids.position(11).unwrap(),
            MropePosition {
                temporal: 7,
                height: 6,
                width: 6,
            }
        );
    }

    #[test]
    fn mrope_separate_image_and_video_layouts_cover_each_grid_kind() {
        let image =
            build_mrope_position_ids(&[MropeSegment::Image(MropeGrid::new(1, 2, 4))], 2).unwrap();
        assert_eq!(image.axes(), [&[0, 0], &[0, 0], &[0, 1]]);

        let video =
            build_mrope_position_ids(&[MropeSegment::Video(MropeGrid::new(2, 2, 2))], 2).unwrap();
        assert_eq!(video.axes(), [&[0, 1], &[0, 0], &[0, 0]]);
    }

    #[test]
    fn mrope_text_resumes_after_image_and_video_axis_maxima() {
        let image = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::Image(MropeGrid::new(1, 2, 4)),
                MropeSegment::Text(1),
            ],
            2,
        )
        .unwrap();
        assert_eq!(image.axes(), [&[0, 1, 1, 3], &[0, 1, 1, 3], &[0, 1, 2, 3]]);

        let video = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::Video(MropeGrid::new(2, 2, 2)),
                MropeSegment::Text(1),
            ],
            2,
        )
        .unwrap();
        assert_eq!(video.axes(), [&[0, 1, 2, 3], &[0, 1, 1, 3], &[0, 1, 1, 3]]);
    }

    #[test]
    fn mrope_timestamped_video_truncates_fractional_temporal_positions() {
        // The reference expression is `(t * seconds_per_grid_step * tokens_per_second) as i64`.
        // Timing is positive by construction, so Rust truncation and Qwen2.5-VL's `.long()` both floor.
        let below = build_mrope_position_ids(
            &[
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(3, 1, 1),
                    timing: video_timing(0.499_999, 2.0),
                },
                MropeSegment::Text(1),
            ],
            1,
        )
        .unwrap();
        assert_eq!(below.axes(), [&[0, 0, 1, 2], &[0, 0, 0, 2], &[0, 0, 0, 2]]);

        let above = build_mrope_position_ids(
            &[MropeSegment::TimestampedVideo {
                grid: MropeGrid::new(3, 1, 1),
                timing: video_timing(0.500_001, 2.0),
            }],
            1,
        )
        .unwrap();
        assert_eq!(above.axes(), [&[0, 1, 2], &[0, 0, 0], &[0, 0, 0]]);
    }

    #[test]
    fn mrope_multiple_timestamped_videos_resume_text_and_pack_axis_major_exactly() {
        let ids = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(3, 2, 2),
                    timing: video_timing(0.25, 2.0),
                },
                MropeSegment::Text(1),
                MropeSegment::TimestampedVideo {
                    grid: MropeGrid::new(3, 2, 2),
                    timing: video_timing(1.25, 2.0),
                },
                MropeSegment::Text(2),
            ],
            2,
        )
        .unwrap();

        // Independent arithmetic: the two local temporal sequences are trunc([0,.5,1]) = [0,0,1]
        // and trunc([0,2.5,5]) = [0,2,5]. Their bases are 1 and 4 respectively.
        let temporal = [0, 1, 1, 2, 3, 4, 6, 9, 10, 11];
        let height = [0, 1, 1, 1, 3, 4, 4, 4, 10, 11];
        let width = [0, 1, 1, 1, 3, 4, 4, 4, 10, 11];
        assert_eq!(ids.axes(), [&temporal, &height, &width]);
        assert_eq!(ids.next_text_position(), MropePosition::collapsed(12));

        let expected_packed = [temporal.as_slice(), height.as_slice(), width.as_slice()].concat();
        assert_eq!(ids.packed(), expected_packed);
    }

    #[test]
    fn mrope_timestamp_timing_validation_and_overflow_are_typed() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                MropeSecondsPerGridStep::new(value),
                Err(MropePositionError::NonFiniteTiming {
                    unit: MropeTimingUnit::SecondsPerGridStep,
                })
            );
            assert_eq!(
                MropeTemporalTokensPerSecond::new(value),
                Err(MropePositionError::NonFiniteTiming {
                    unit: MropeTimingUnit::TemporalTokensPerSecond,
                })
            );
        }
        for value in [0.0, -0.0, -1.0] {
            assert_eq!(
                MropeSecondsPerGridStep::new(value),
                Err(MropePositionError::NonPositiveTiming {
                    unit: MropeTimingUnit::SecondsPerGridStep,
                })
            );
            assert_eq!(
                MropeTemporalTokensPerSecond::new(value),
                Err(MropePositionError::NonPositiveTiming {
                    unit: MropeTimingUnit::TemporalTokensPerSecond,
                })
            );
        }

        let scaled_overflow = MropeSegment::TimestampedVideo {
            grid: MropeGrid::new(2, 1, 1),
            timing: video_timing((MAX_EXACT_MROPE_POSITION + 1) as f64, 1.0),
        };
        assert_eq!(
            build_mrope_position_ids(&[scaled_overflow], 1),
            Err(MropePositionError::TemporalPositionOverflow { temporal_index: 1 })
        );

        let base_overflow = MropeSegment::TimestampedVideo {
            grid: MropeGrid::new(2, 1, 1),
            timing: video_timing(MAX_EXACT_MROPE_POSITION as f64, 1.0),
        };
        assert_eq!(
            build_mrope_position_ids(&[MropeSegment::Text(1), base_overflow], 1),
            Err(MropePositionError::PositionOverflow {
                position: MAX_EXACT_MROPE_POSITION + 2,
            })
        );
    }

    #[test]
    fn mrope_qwen25_position_assembly_converts_videos_in_processor_order() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            Some("4"),
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let image_grids = [MropeGrid::new(1, 2, 2)];
        let video_grids = [MropeGrid::new(2, 2, 2), MropeGrid::new(1, 1, 1)];
        let tokens = [
            99,
            VISION_START,
            IMAGE,
            IMAGE,
            IMAGE,
            IMAGE,
            77,
            VISION_START,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VISION_START,
            VIDEO,
            88,
        ];
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let fps = [4.0, 1.0];
        let input = Qwen25VlMropePositionInput {
            tokens: &tokens,
            special_tokens: special_tokens(),
            image_grids: &image_grids,
            video_grids: &video_grids,
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::PerVideo(&fps)),
            config: &metadata,
        };

        let actual = assemble_qwen2_5_vl_mrope_positions(input).unwrap();
        let timings = assemble_qwen2_5_vl_video_timings(
            video_grids.len(),
            patch_size,
            MropeProcessorFpsInput::PerVideo(&fps),
            &metadata,
        )
        .unwrap();
        let expected = build_mrope_position_ids(
            &[
                MropeSegment::Text(2),
                MropeSegment::Image(image_grids[0]),
                MropeSegment::Text(2),
                MropeSegment::TimestampedVideo {
                    grid: video_grids[0],
                    timing: timings[0],
                },
                MropeSegment::Text(1),
                MropeSegment::TimestampedVideo {
                    grid: video_grids[1],
                    timing: timings[1],
                },
                MropeSegment::Text(1),
            ],
            1,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(
            actual.axes()[0],
            &[0, 1, 2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 8, 8, 8, 8, 9, 10, 11]
        );
        assert_eq!(
            actual.axes()[1],
            &[0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 6, 6, 7, 7, 9, 10, 11]
        );
        assert_eq!(
            actual.axes()[2],
            &[0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 9, 10, 11]
        );
        assert_eq!(actual.next_text_position(), MropePosition::collapsed(12));
    }

    #[test]
    fn mrope_special_token_reconciliation_matches_records_and_feeds_position_assembly() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            Some("4"),
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let processor_tokens = processor_special_tokens();
        let tokenizer_tokens = tokenizer_special_tokens();

        let reconciled =
            reconcile_mrope_special_token_ids(&processor_tokens, &tokenizer_tokens, &metadata)
                .unwrap();
        assert_eq!(reconciled, special_tokens());

        let video_grids = [MropeGrid::new(2, 1, 1)];
        let tokens = [VISION_START, VIDEO, VIDEO, 42];
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();
        let from_sources =
            assemble_qwen2_5_vl_mrope_positions_from_sources(Qwen25VlMropePositionSourceInput {
                tokens: &tokens,
                processor_special_tokens: &processor_tokens,
                tokenizer_special_tokens: &tokenizer_tokens,
                image_grids: &[],
                video_grids: &video_grids,
                spatial_merge_size: 1,
                temporal_patch_size: patch_size,
                fps: Some(MropeProcessorFpsInput::Scalar(4.0)),
                config: &metadata,
            })
            .unwrap();
        let direct = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &tokens,
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &video_grids,
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(4.0)),
            config: &metadata,
        })
        .unwrap();
        assert_eq!(from_sources, direct);
    }

    #[test]
    fn mrope_special_token_reconciliation_rejects_source_metadata_failures() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            Some("4"),
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let tokenizer_tokens = tokenizer_special_tokens();

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &[
                    MropeProcessorSpecialToken::new(
                        MropeSpecialTokenKind::VisionStart,
                        VISION_START_TOKEN,
                    ),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Video, VIDEO_PAD_TOKEN),
                ],
                &tokenizer_tokens,
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::MissingProcessorSpecialToken {
                    role: MropeSpecialTokenKind::Image,
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &[
                    MropeProcessorSpecialToken::new(
                        MropeSpecialTokenKind::VisionStart,
                        VISION_START_TOKEN,
                    ),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, IMAGE_PAD_TOKEN),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, "<image2>"),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Video, VIDEO_PAD_TOKEN),
                ],
                &tokenizer_tokens,
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::DuplicateProcessorSpecialToken {
                    role: MropeSpecialTokenKind::Image,
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &[
                    MropeProcessorSpecialToken::new(
                        MropeSpecialTokenKind::VisionStart,
                        VISION_START_TOKEN,
                    ),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, IMAGE_PAD_TOKEN),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Video, IMAGE_PAD_TOKEN),
                ],
                &tokenizer_tokens,
                &metadata,
            ),
            Err(MropeSpecialTokenReconcileError::DuplicateProcessorMarker {
                token: IMAGE_PAD_TOKEN.to_string(),
                first: MropeSpecialTokenKind::Image,
                second: MropeSpecialTokenKind::Video,
            })
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &[
                    MropeTokenizerSpecialToken::new(VISION_START_TOKEN, VISION_START),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, IMAGE),
                ],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::UnknownTokenizerSpecialToken {
                    role: MropeSpecialTokenKind::Video,
                    token: VIDEO_PAD_TOKEN.to_string(),
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &[
                    MropeTokenizerSpecialToken::new(VISION_START_TOKEN, VISION_START),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, IMAGE),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, 99),
                    MropeTokenizerSpecialToken::new(VIDEO_PAD_TOKEN, VIDEO),
                ],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::DuplicateTokenizerSpecialToken {
                    token: IMAGE_PAD_TOKEN.to_string(),
                    first_id: IMAGE,
                    second_id: 99,
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &[
                    MropeTokenizerSpecialToken::new(VISION_START_TOKEN, VISION_START),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, IMAGE),
                    MropeTokenizerSpecialToken::new(VIDEO_PAD_TOKEN, IMAGE),
                ],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::DuplicateTokenizerSpecialTokenId {
                    token_id: IMAGE,
                    first: MropeSpecialTokenKind::Image,
                    second: MropeSpecialTokenKind::Video,
                }
            )
        );
    }

    #[test]
    fn mrope_special_token_reconciliation_rejects_config_role_mismatch() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, 99, VIDEO as u64],
            Some("4"),
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &tokenizer_special_tokens(),
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::ConfigSpecialTokenMismatch {
                    role: MropeSpecialTokenKind::Image,
                    token: IMAGE_PAD_TOKEN.to_string(),
                    tokenizer_id: IMAGE,
                    config_id: 99,
                }
            )
        );
    }

    #[test]
    fn mrope_special_token_reconciliation_has_deterministic_error_ordering() {
        let config_json = qwen_vl_config_json("qwen2_5_vl", [100, 101, 102], None);
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &[
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, IMAGE_PAD_TOKEN),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Image, "<image2>"),
                    MropeProcessorSpecialToken::new(MropeSpecialTokenKind::Video, VIDEO_PAD_TOKEN),
                ],
                &[],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::MissingProcessorSpecialToken {
                    role: MropeSpecialTokenKind::VisionStart,
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &[MropeTokenizerSpecialToken::new(VISION_START_TOKEN, 100)],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::UnknownTokenizerSpecialToken {
                    role: MropeSpecialTokenKind::Image,
                    token: IMAGE_PAD_TOKEN.to_string(),
                }
            )
        );

        assert_eq!(
            reconcile_mrope_special_token_ids(
                &processor_special_tokens(),
                &[
                    MropeTokenizerSpecialToken::new(VISION_START_TOKEN, 100),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, 999),
                    MropeTokenizerSpecialToken::new(VIDEO_PAD_TOKEN, 998),
                ],
                &metadata,
            ),
            Err(
                MropeSpecialTokenReconcileError::ConfigSpecialTokenMismatch {
                    role: MropeSpecialTokenKind::Image,
                    token: IMAGE_PAD_TOKEN.to_string(),
                    tokenizer_id: 999,
                    config_id: 101,
                }
            )
        );
    }

    #[test]
    fn mrope_position_source_assembly_reconciles_before_later_errors() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            None,
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let processor_tokens = processor_special_tokens();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let err =
            assemble_qwen2_5_vl_mrope_positions_from_sources(Qwen25VlMropePositionSourceInput {
                tokens: &[VISION_START, 99],
                processor_special_tokens: &processor_tokens,
                tokenizer_special_tokens: &[
                    MropeTokenizerSpecialToken::new(VISION_START_TOKEN, VISION_START),
                    MropeTokenizerSpecialToken::new(IMAGE_PAD_TOKEN, IMAGE),
                ],
                image_grids: &[],
                video_grids: &[MropeGrid::new(1, 1, 1)],
                spatial_merge_size: 1,
                temporal_patch_size: patch_size,
                fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
                config: &metadata,
            });
        assert_eq!(
            err,
            Err(MropePositionSourceAssemblyError::SpecialTokens(
                MropeSpecialTokenReconcileError::UnknownTokenizerSpecialToken {
                    role: MropeSpecialTokenKind::Video,
                    token: VIDEO_PAD_TOKEN.to_string(),
                },
            ))
        );
    }

    #[test]
    fn mrope_qwen25_position_assembly_uses_default_and_preserves_zero_video_behavior() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            Some("4"),
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let video_grids = [MropeGrid::new(2, 1, 1)];
        let tokens = [VISION_START, VIDEO, VIDEO];
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let default = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &tokens,
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &video_grids,
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: None,
            config: &metadata,
        })
        .unwrap();
        let explicit = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &tokens,
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &video_grids,
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            config: &metadata,
        })
        .unwrap();
        assert_eq!(default, explicit);

        let missing_rate = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            None,
        );
        let missing_rate =
            MropeConfigMetadata::from_hf_config_slice(missing_rate.as_bytes()).unwrap();
        let zero_video = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[42],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[],
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: None,
            config: &missing_rate,
        })
        .unwrap();
        assert_eq!(zero_video.axes(), [&[0][..], &[0][..], &[0][..]]);
    }

    #[test]
    fn mrope_qwen25_position_assembly_preserves_discovery_before_timing_errors() {
        let config_json = qwen_vl_config_json(
            "qwen2_5_vl",
            [VISION_START as u64, IMAGE as u64, VIDEO as u64],
            None,
        );
        let metadata = MropeConfigMetadata::from_hf_config_slice(config_json.as_bytes()).unwrap();
        let patch_size = MropeTemporalPatchSize::new(2).unwrap();

        let malformed = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[VISION_START, 99],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[],
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            config: &metadata,
        });
        assert_eq!(
            malformed,
            Err(MropePositionAssemblyError::Span(
                MropeSpanError::MissingVisualMarker {
                    start_index: 0,
                    index: 1,
                }
            ))
        );

        let invalid_grid = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[VISION_START, VIDEO],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[MropeGrid::new(1, 3, 1)],
            spatial_merge_size: 2,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            config: &metadata,
        });
        assert_eq!(
            invalid_grid,
            Err(MropePositionAssemblyError::Span(MropeSpanError::Grid(
                MropePositionError::NonDivisibleGrid {
                    kind: MropeVisualKind::Video,
                    axis: "height",
                    dimension: 3,
                    merge: 2,
                },
            )))
        );

        let invalid_fps = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[VISION_START, VIDEO],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[MropeGrid::new(1, 1, 1)],
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(f64::NAN)),
            config: &metadata,
        });
        assert_eq!(
            invalid_fps,
            Err(MropePositionAssemblyError::ProcessorTiming(
                MropeProcessorTimingError::NonFiniteSampledFramesPerSecond,
            ))
        );

        let list_mismatch = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[VISION_START, VIDEO],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[MropeGrid::new(1, 1, 1)],
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::PerVideo(&[])),
            config: &metadata,
        });
        assert_eq!(
            list_mismatch,
            Err(MropePositionAssemblyError::ProcessorTiming(
                MropeProcessorTimingError::SampledFramesPerSecondCountMismatch {
                    expected: 1,
                    actual: 0,
                },
            ))
        );

        let missing_rate = assemble_qwen2_5_vl_mrope_positions(Qwen25VlMropePositionInput {
            tokens: &[VISION_START, VIDEO],
            special_tokens: special_tokens(),
            image_grids: &[],
            video_grids: &[MropeGrid::new(1, 1, 1)],
            spatial_merge_size: 1,
            temporal_patch_size: patch_size,
            fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
            config: &metadata,
        });
        assert_eq!(
            missing_rate,
            Err(MropePositionAssemblyError::ProcessorTiming(
                MropeProcessorTimingError::MissingTemporalTokensPerSecond,
            ))
        );
    }

    #[test]
    fn mrope_qwen2_unit_stride_layout_is_byte_identical_to_legacy_reference() {
        let segments = [
            MropeSegment::Text(2),
            MropeSegment::Image(MropeGrid::new(1, 4, 6)),
            MropeSegment::Text(1),
            MropeSegment::Video(MropeGrid::new(3, 2, 4)),
            MropeSegment::Text(2),
        ];
        let (reference_axes, reference_next) = legacy_unit_stride_reference(&segments, 2);
        let ids = build_mrope_position_ids(&segments, 2).unwrap();

        assert_eq!(
            ids.axes(),
            [&reference_axes[0], &reference_axes[1], &reference_axes[2]]
        );
        assert_eq!(
            ids.next_text_position(),
            MropePosition::collapsed(reference_next)
        );
        let reference_packed = reference_axes.concat();
        assert_eq!(ids.packed(), reference_packed);
        let got_bytes: Vec<u8> = ids
            .packed()
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let reference_bytes: Vec<u8> = reference_packed
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(got_bytes, reference_bytes);
    }

    #[test]
    fn mrope_layout_validation_is_typed() {
        assert_eq!(
            build_mrope_position_ids(&[MropeSegment::Text(1)], 0),
            Err(MropePositionError::ZeroSpatialMerge)
        );
        assert_eq!(
            build_mrope_position_ids(&[MropeSegment::Image(MropeGrid::new(0, 2, 2))], 2),
            Err(MropePositionError::ZeroGridDimension {
                kind: MropeVisualKind::Image,
                axis: "temporal",
            })
        );
        assert_eq!(
            build_mrope_position_ids(&[MropeSegment::Video(MropeGrid::new(1, 3, 2))], 2),
            Err(MropePositionError::NonDivisibleGrid {
                kind: MropeVisualKind::Video,
                axis: "height",
                dimension: 3,
                merge: 2,
            })
        );
        assert_eq!(
            build_mrope_position_ids(&[MropeSegment::Text(1), MropeSegment::Text(usize::MAX)], 1),
            Err(MropePositionError::TokenCountOverflow)
        );
    }

    #[test]
    fn mrope_binding_shape_and_index_validation_is_typed() {
        let ids = build_mrope_position_ids(&[MropeSegment::Text(2)], 1).unwrap();
        assert_eq!(
            ids.packed_for_len(3),
            Err(MropePositionError::LengthMismatch {
                expected: 3,
                actual: 2,
            })
        );
        assert_eq!(
            ids.position(2),
            Err(MropePositionError::IndexOutOfBounds { index: 2, len: 2 })
        );
    }
}
