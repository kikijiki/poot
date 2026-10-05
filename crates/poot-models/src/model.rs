//! The object-safe [`Model`] trait (target architecture section 2, ADR-0105): a family's config, its
//! weight map and its tracer behind one interface the driver holds as `Box<dyn Model>`. A model stays
//! plain Rust tracer code over shared component functions; the trait only names what the driver needs.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use poot_graph_ir::{Graph, ValidationOutputs};
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
    Logits { vocab: usize },
    Hidden { width: usize },
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
    /// entry is -1 (a padding token) leaves the pool untouched.
    Paged { pool_slots: NonZeroUsize },
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
}

/// One model behind the driver: its config, its weights and its tracer.
pub trait Model: Send + Sync + fmt::Debug {
    fn config(&self) -> &ModelConfig;

    fn weights(&self) -> &WeightMap;

    /// The graph of one step of `phase` at `shape`. A shape the family cannot trace is a typed
    /// refusal, never a panic.
    fn trace(&self, phase: Phase, shape: StepShape)
    -> Result<Graph<ValidationOutputs>, TraceError>;
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
    /// A carried state value whose leading axis is not the row axis: its
    /// leading axis has `leading` entries where the step has `rows` rows. A paged KV pool is the one
    /// exception, and only as the pooled `Positional` cache on axis 0.
    StateRowAxis {
        leading: usize,
        rows: usize,
    },
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
}
