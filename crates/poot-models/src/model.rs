//! The object-safe [`Model`] trait (target architecture section 2, ADR-0105): a family's config, its
//! weight map and its tracer behind one interface the driver holds as `Box<dyn Model>`. A model stays
//! plain Rust tracer code over shared component functions; the trait only names what the driver needs.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use poot_graph_ir::Graph;
use poot_quant::weights::{HandleFormat, WeightId, WeightMap, WeightMapError};

use crate::chat::ChatFormat;

/// A family's display name (`"qwen2"`): for messages and receipts only; nothing dispatches on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FamilyKey(&'static str);

impl FamilyKey {
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for FamilyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// What a traced step returns per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelOutput {
    Logits {
        vocab: usize,
    },
    /// The final hidden state of every position, `width` wide ([`Phase::Encode`]).
    Hidden {
        width: usize,
    },
    /// One `labels`-wide score vector per row, read at the first position: a cross-encoder's
    /// classifier ([`Phase::Encode`]).
    Scores {
        labels: usize,
    },
}

/// The family-neutral facts the driver needs about a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub family: FamilyKey,
    pub vocab: usize,
    pub max_positions: usize,
    /// Every token id that ends generation: the union of `config.json`'s and
    /// `generation_config.json`'s `eos_token_id` for an HF checkpoint, read by
    /// [`RawConfig::eos_token_ids`](crate::registry::RawConfig::eos_token_ids).
    pub eos: BTreeSet<u32>,
    /// The checkpoint's beginning-of-sequence id, when its config names one.
    pub bos: Option<u32>,
    /// The id the text encoder prepends to every prompt: `Some` only for a family trained with a
    /// mandatory BOS (Gemma 2 and 3). A family without one leaves its prompts as the tokenizer encodes
    /// them.
    pub prompt_bos: Option<u32>,
    pub output: ModelOutput,
    /// Semantic alignment of a prefill step: a prefill's token count is a multiple of it (1 except
    /// for compressed attention). It never limits how large a step may be.
    pub prefill_granule: NonZeroUsize,
    /// The fallback chat format, used only when the checkpoint ships no jinja `chat_template` (the
    /// template itself is tokenizer data, not model config).
    pub chat: ChatFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Prefill,
    Decode,
    /// A prefill whose step inputs are the prompt's tokens with caller-computed embedding rows
    /// scattered over some positions (`StepInput::TokensWithEmbeds`): a multimodal prompt's image
    /// rows. A family that takes them traces it as [`Phase::Prefill`] over the embedded inputs.
    PrefillEmbeds,
    /// The whole input in one step, returning the final hidden states of every position
    /// (`HeadKind::Hidden`; [`ModelOutput::Hidden`]) for the host to pool or score: an encoder's
    /// forward, or a decoder's embedding pass. A step carries no cache a later step reads.
    Encode,
    /// The multi-token-prediction layer over the hidden-state carry (POOT-1110): a step of `tokens`
    /// positions whose `Slot::Pos` input holds the absolute position `p` of each hidden row the
    /// carry holds and whose `Slot::Token` input holds the host token that follows it, `t[p + 1]`.
    /// The program reads the first `tokens` rows of the [`HiddenCarry`] the step's shape carries,
    /// continues the layer's own cache, and returns the logits of the next-next token. It writes its
    /// own post-norm hidden state of the step's last position into row 0 of the carry, the row a
    /// following one-token `Mtp` step of a draft chain reads. A family that does not trace it refuses
    /// it typed in [`check_step`](crate::components::standard::check_step).
    Mtp,
}

/// How the traced step's KV cache is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvLayout {
    /// One private `[rows, kv_heads, capacity, head_dim]` cache per layer; row `r` writes at its own
    /// first position. Every row of a step carries the same token count.
    Contiguous,
    /// One `[pool_slots, kv_heads, head_dim]` pool per layer shared by every row. Each row reads and
    /// writes through per-step slot maps bound as `Slot::SlotMap` (`"read"`: logical position to pool
    /// slot, `[rows, capacity]`; `"write"`: pool slot to the step's flat token index or -1,
    /// `[pool_slots]`), so a row's blocks may sit anywhere in the pool and a token whose write map
    /// entry is -1 (a padding token) leaves the pool untouched. Recurrent state is pooled the same way:
    /// one row per sequence the pool serves (`seqs`), which a step's rows address through the
    /// `state_read` and `state_write` index inputs (`Step::recurrent`).
    Paged {
        pool_slots: NonZeroUsize,
        seqs: NonZeroUsize,
    },
}

/// Which positions of a step produce logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogitRows {
    Last,
    All,
}

/// The semantic shape of one traced step: `rows` sequences of `tokens` new tokens each, against a
/// cache of `capacity` positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StepShape {
    pub rows: NonZeroUsize,
    pub tokens: NonZeroUsize,
    pub capacity: NonZeroUsize,
    pub kv: KvLayout,
    pub logits: LogitRows,
    /// The hidden-state carry the step also writes, when its caller asks for one.
    pub hidden: Option<HiddenCarry>,
}

/// A request that a step write its final hidden state, at every position of the step and before any
/// last-token slice, into a per-sequence carry a later program of the same driver step reads (the
/// MTP draft source, POOT-1109). The carry is one f32 `[seqs, positions, width]` value; `positions`
/// is the driver's one extent for it (the verify window or the widest prefill piece, whichever is
/// larger), so every program of an executable declares the same aval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HiddenCarry {
    pub positions: NonZeroUsize,
}

/// One model behind the driver: its config, its weights and its tracer.
pub trait Model: Send + Sync + fmt::Debug {
    fn config(&self) -> &ModelConfig;

    fn weights(&self) -> &WeightMap;

    /// The graph of one step of `phase` at `shape`. A shape the family cannot trace is a typed
    /// refusal, never a panic.
    fn trace(&self, phase: Phase, shape: StepShape) -> Result<Graph, TraceError>;
}

/// Why a config field cannot configure a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigReason {
    Missing,
    WrongType,
    Zero,
    NotDivisible {
        by: usize,
    },
    /// Above the largest value the field admits.
    Exceeds {
        max: usize,
    },
    /// A float that must be finite and positive (an epsilon, a RoPE base or factor).
    NotFinitePositive,
    Unsupported,
    /// A per-layer list or string whose length is not the layer count.
    Length {
        expected: usize,
    },
    /// A value that must equal a width the rest of the config fixes (a head width, a rotary width).
    Mismatch {
        expected: usize,
        found: usize,
    },
    /// A sum or product that does not fit `usize`.
    Overflow,
}

/// Why a family refuses a step shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeReason {
    DecodeTokens,
    CapacityAboveMax {
        max: usize,
    },
    /// A step of more new tokens than its cache holds.
    TokensAboveCapacity,
    Granule {
        granule: usize,
    },
    /// A carried state value whose leading axis is not its row axis: its leading axis has `leading`
    /// entries where it should have `rows` (the step's rows, or under a paged layout a recurrent
    /// state's one row per sequence the pool serves). A paged KV pool is the one exception, and only as
    /// the pooled `Positional` cache on axis 0.
    StateRowAxis {
        leading: usize,
        rows: usize,
    },
    /// A routed step whose `rows * tokens * top_k` expert rows exceed what the expert chain
    /// indexes exactly.
    RoutedRows {
        max: usize,
    },
    /// A hidden-state carry with fewer positions than the step has tokens: the step would write
    /// positions the carry does not have.
    HiddenCarryNarrow {
        positions: usize,
        tokens: usize,
    },
    /// A step that asked for a hidden-state carry and whose program carries none: the family does
    /// not trace one.
    HiddenCarryMissing,
    /// A carried [`StateRole::Handoff`](poot_graph_ir::StateRole::Handoff) value that is not an F32
    /// `[rows, positions, width]` value.
    HiddenCarryLayout,
    /// A hidden-state carry of `found` positions where the step asked for `expected`: the readers of
    /// the carry would bind another buffer than the writers.
    HiddenCarryPositions {
        found: usize,
        expected: usize,
    },
    /// A hidden-state carry written by a step that asked for none.
    HiddenCarryUnrequested,
    /// A [`Phase::Mtp`] step whose shape names no hidden-state carry: there is nothing for the
    /// layer to read.
    MtpWithoutCarry,
}

/// A model could not be built from its config and weights.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("{family}: config field {field}: {reason:?}")]
    Config {
        family: FamilyKey,
        field: &'static str,
        reason: ConfigReason,
    },
    #[error("{family}: {source}")]
    Weight {
        family: FamilyKey,
        source: WeightMapError,
    },
    #[error("{family}: weight {id} has shape {found:?}, expected {expected:?}")]
    WeightShape {
        family: FamilyKey,
        id: WeightId,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("{family}: weight {id} is stored as {format:?}, which this family cannot trace")]
    WeightFormat {
        family: FamilyKey,
        id: WeightId,
        format: HandleFormat,
    },
    /// A stored value the family cannot read: element `index` (row-major) of the index table `id`
    /// is `value`, outside the I32 range its tracer indexes with.
    #[error("{family}: weight {id} element {index} is {value}, outside the I32 range")]
    WeightValue {
        family: FamilyKey,
        id: WeightId,
        index: usize,
        value: i64,
    },
}

/// A model refused to trace a step.
#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("{family} does not trace phase {phase:?}")]
    PhaseUnsupported { family: FamilyKey, phase: Phase },
    #[error("{family} cannot trace shape {shape:?}: {reason:?}")]
    ShapeUnsupported {
        family: FamilyKey,
        shape: StepShape,
        reason: ShapeReason,
    },
    /// The traced graph is malformed: a defect of the tracer, never of the checkpoint.
    #[error("the traced graph is malformed: {0}")]
    Malformed(Box<poot_graph_ir::EditError>),
}

impl From<poot_graph_ir::EditError> for TraceError {
    fn from(error: poot_graph_ir::EditError) -> Self {
        TraceError::Malformed(Box::new(error))
    }
}
