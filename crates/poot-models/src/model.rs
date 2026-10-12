//! The object-safe [`Model`] trait (target architecture section 2, ADR-0105): a family's config, its
//! weight map and its tracer behind one interface the driver holds as `Box<dyn Model>`. A model stays
//! plain Rust tracer code over shared component functions; the trait only names what the driver needs.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, BuilderError, Graph, TraceBudgetExceeded, TraceLimits};
use poot_quant::weights::{HandleFormat, WeightId, WeightMap, WeightMapError};

use poot_text::ChatFormat;

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

    /// The body of one traced step: trace `phase` at `shape` into `b`, a builder already charging
    /// the caller's [`TraceLimits`]. A shape the family cannot trace is a typed refusal, never a
    /// panic. Callers use `<dyn Model>::trace`, which checks the step's token count first; the
    /// trait has no `trace` method a family could override, so the admission cannot be skipped.
    fn trace_into(&self, phase: Phase, shape: StepShape, b: Builder) -> Result<Graph, TraceError>;
}

impl dyn Model + '_ {
    /// The graph of one step of `phase` at `shape` under the caller's `limits`.
    ///
    /// A step of more tokens than `limits.max_tokens` is refused before the tracer body runs; every
    /// value, equation, literal and owned allocation the body makes is charged against the same
    /// limits, so a refusal names the exact resource and nothing past it was built. This is an
    /// inherent method of the trait object, so no family can replace it:
    ///
    /// ```compile_fail
    /// use poot_graph_ir::{Builder, Graph, TraceLimits};
    /// use poot_models::model::{Model, ModelConfig, Phase, StepShape, TraceError};
    /// use poot_quant::weights::WeightMap;
    ///
    /// #[derive(Debug)]
    /// struct Skips;
    /// impl Model for Skips {
    ///     fn config(&self) -> &ModelConfig { unimplemented!() }
    ///     fn weights(&self) -> &WeightMap { unimplemented!() }
    ///     fn trace_into(&self, _: Phase, _: StepShape, _: Builder) -> Result<Graph, TraceError> {
    ///         unimplemented!()
    ///     }
    ///     // a family cannot provide its own admission-free entry point
    ///     fn trace(&self, _: Phase, _: StepShape, _: TraceLimits) -> Result<Graph, TraceError> {
    ///         unimplemented!()
    ///     }
    /// }
    /// ```
    pub fn trace(
        &self,
        phase: Phase,
        shape: StepShape,
        limits: TraceLimits,
    ) -> Result<Graph, TraceError> {
        limits.admit_tokens(shape.tokens)?;
        self.trace_into(phase, shape, Builder::new(limits))
    }
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
    /// A guarantee the family's wiring relies on does not hold.
    #[error("{family}: {source}")]
    Wiring {
        family: FamilyKey,
        source: WiringError,
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

/// A guarantee the config reader or a step check establishes before tracing does not hold where a
/// shared component needs it: the family's wiring is defective (a parameter block a layer kind needs
/// is missing, a step input a phase carries is absent). Never a property of a checkpoint; reported as a
/// value so the tracer path has no panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WiringError {
    /// A layer kind needs a parameter block the config reader read only for configs that have it.
    #[error("a layer needs the {0} parameters, which the config did not read")]
    MissingParams(ParamBlock),
    /// A program that reads the hidden-state carry was traced for a step whose shape names none:
    /// the phase check ([`check_mtp_step`](crate::components::standard::check_mtp_step)) refuses it.
    #[error("the step's shape names no hidden-state carry")]
    MissingHiddenCarry,
    /// An MLA attention was given a key-position selector but a plain (not low-rank) query: the
    /// family guarantees the low-rank query beside every selector.
    #[error("a key-position selector needs the low-rank query")]
    SelectorWithoutLowRankQuery,
    /// A step with slot maps was traced against a contiguous KV layout.
    #[error("a step with slot maps must be paged")]
    SlotMapsOnContiguousStep,
    /// A sectioned rope was traced over a step that carries no mRoPE position input.
    #[error("a sectioned rope needs the step's mRoPE positions")]
    SectionedRopeWithoutMrope,
    /// `apply_rope_sectioned` was given a plain (unsectioned) rope.
    #[error("sectioned rotation needs a sectioned RopeParams")]
    RopeNotSectioned,
    /// A layer index whose validation id does not fit its id space.
    #[error("layer index {layer} has no validation id")]
    LayerIndexOutOfRange { layer: usize },
    /// A group-limited router over the sqrt-softplus arm, which the limit refuses at construction.
    #[error("a group limit is not defined for the sqrt-softplus router")]
    GroupLimitOnSqrtSoftplus,
    /// Per-expert scores asked of a softmax router.
    #[error("only a biased top-k router scores each expert on its own")]
    ScoresOfSoftmaxRouter,
    /// A hidden-state carry written with rows of another extent than the carry holds.
    #[error("carried hidden rows are {found} wide or long where the carry holds {expected}")]
    HiddenCarryRows { expected: usize, found: usize },
    /// A sandwich layer built without its two post norms.
    #[error("a sandwich layer needs its post norms")]
    SandwichWithoutPostNorms,
    /// A vision step whose patch count does not fit the rotary table's `u32` positions.
    #[error("{patches} patches do not fit a rotary table of u32 positions")]
    PatchesOutOfRange { patches: usize },
    /// A sharded embedding table with no shard to read.
    #[error("a sharded table has no shards")]
    TableWithoutShards,
    /// An order-3 n-gram head was traced for a spec that hashes only two tokens.
    #[error("an order-3 n-gram head has no third hashed token")]
    NgramOrderThree,
}

impl WiringError {
    /// `block` of a family config, or the wiring defect that it is absent where a layer needs it.
    pub fn require<T>(block: &Option<T>, name: ParamBlock) -> Result<&T, WiringError> {
        block.as_ref().ok_or(WiringError::MissingParams(name))
    }
}

/// A family-config parameter block that only some layer kinds need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamBlock {
    Dense,
    Moe,
    Kda,
    Dsa,
    Indexer,
    /// The low-rank query projection of an MLA attention.
    LowRankQuery,
    /// A hyper-connection layer's per-stream injection weights.
    Injection,
}

impl fmt::Display for ParamBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ParamBlock::Dense => "dense feed-forward",
            ParamBlock::Moe => "mixture-of-experts",
            ParamBlock::Kda => "KDA",
            ParamBlock::Dsa => "DSA",
            ParamBlock::Indexer => "indexer",
            ParamBlock::LowRankQuery => "low-rank query",
            ParamBlock::Injection => "stream injection",
        })
    }
}

/// The extent of the last axis of `shape`, for `operation` to read: a rank-0 value has none.
pub(crate) fn last_axis(operation: &'static str, shape: &[usize]) -> Result<usize, TraceError> {
    shape.last().copied().ok_or_else(|| {
        BuilderError::Requirement {
            operation,
            requirement: poot_graph_ir::Requirement::Scalar,
        }
        .into()
    })
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
    /// A rope table the step's capacity or the family's spec cannot build.
    #[error("rope table: {0}")]
    RopeTable(#[from] poot_graph_ir::rope_table::RopeSpecError),
    /// An index-bounds guard the tracer emitted refused its selector or bound.
    #[error("index guard: {0}")]
    IndexGuard(Box<poot_graph_ir::IndexGuardError>),
    /// A u64-hash composition refused its modulus or operand.
    #[error("u64 hash: {0}")]
    Hash(Box<poot_graph_ir::ops::U64HashError>),
    /// The family's wiring is defective: see [`WiringError`].
    #[error("the family's tracer wiring is defective: {0}")]
    Wiring(#[from] WiringError),
    /// The step does not fit the caller's [`TraceLimits`]: the resource, the limit and the attempted
    /// total are in the cause. Nothing past the refusal was built.
    #[error("the step exceeds the trace limits: {0}")]
    Budget(#[from] TraceBudgetExceeded),
    /// The tracer built something the graph IR refuses: a shape or dtype the operation does not
    /// admit, a handle of another builder, or a graph that is not well formed. A defect of the
    /// tracer, never of the checkpoint.
    #[error("the tracer could not build the graph: {0}")]
    Build(Box<BuilderError>),
}

impl From<poot_graph_ir::IndexGuardError> for TraceError {
    fn from(error: poot_graph_ir::IndexGuardError) -> Self {
        match error {
            poot_graph_ir::IndexGuardError::Build(build) => build.into(),
            other => TraceError::IndexGuard(Box::new(other)),
        }
    }
}

impl From<poot_graph_ir::ops::U64HashError> for TraceError {
    fn from(error: poot_graph_ir::ops::U64HashError) -> Self {
        match error {
            poot_graph_ir::ops::U64HashError::Build(build) => build.into(),
            other => TraceError::Hash(Box::new(other)),
        }
    }
}

impl From<BuilderError> for TraceError {
    fn from(error: BuilderError) -> Self {
        match error {
            BuilderError::Budget(budget) => TraceError::Budget(budget),
            other => TraceError::Build(Box::new(other)),
        }
    }
}
