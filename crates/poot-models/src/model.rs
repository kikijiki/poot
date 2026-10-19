//! The object-safe [`Model`] trait (target architecture section 2, ADR-0105): a family's config, its
//! weight map and its tracer behind one interface the driver holds as `Box<dyn Model>`. A model stays
//! plain Rust tracer code over shared component functions; the trait only names what the driver needs.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use poot_graph_ir::{Builder, BuilderError, Graph, TraceBudgetExceeded, TraceLimits};
use poot_quant::weights::{HandleFormat, WeightId, WeightMap, WeightMapError};

use poot_text::ChatFormat;

use crate::io::{InputId, ModelSchema, ProgramId, SchemaError, StateMismatch};

/// A family's display name (`"qwen2"`): for messages and receipts only; nothing dispatches on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FamilyKey(&'static str);

impl FamilyKey {
    /// A key displaying as `name`.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The display name.
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
    /// A vocabulary logit vector per position (a `step` or `draft` program).
    Logits {
        /// The vocabulary size: the logit vector's width.
        vocab: usize,
    },
    /// The final hidden state of every position, `width` wide (an `encode` program).
    Hidden {
        /// The model's hidden width.
        width: usize,
    },
    /// One `labels`-wide score vector per row, read at the first position: a cross-encoder's
    /// classifier (an `encode` program).
    Scores {
        /// The classifier's label count.
        labels: usize,
    },
}

/// The family-neutral facts the driver needs about a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// The family's key.
    pub family: FamilyKey,
    /// Vocabulary size: the embedding table's rows and, for a logits model, the output width.
    pub vocab: usize,
    /// The longest context the checkpoint declares, in positions; also the largest cache capacity a
    /// step may ask for.
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
    /// Semantic alignment of a prefill step: a prefill's token count is a multiple of it (1 except
    /// for compressed attention). It never limits how large a step may be.
    pub prefill_granule: NonZeroUsize,
    /// The fallback chat format, used only when the checkpoint ships no jinja `chat_template` (the
    /// template itself is tokenizer data, not model config).
    pub chat: ChatFormat,
}

/// How the traced step's KV cache is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvLayout {
    /// One private `[rows, kv_heads, capacity, head_dim]` cache per layer; row `r` writes at its own
    /// first position. Every row of a step carries the same token count.
    Contiguous,
    /// One `[pool_slots, kv_heads, head_dim]` pool per layer shared by every row. Each row reads and
    /// writes through per-step slot maps (`Slot::SlotMap` `"read"`: logical position to pool slot,
    /// `[rows, capacity]`; `Slot::WriteMap` `"write"`: the step's flat token index to its pool slot or
    /// -1, `[rows * tokens]`), so a row's blocks may sit anywhere in the pool and a token whose write
    /// map entry is -1 (a padding token) leaves the pool untouched. Recurrent state is pooled the same way:
    /// one row per sequence the pool serves (`seqs`), which a step's rows address through the
    /// `state_read` and `state_write` index inputs (`Step::recurrent`).
    Paged {
        /// Slots in each layer's KV pool.
        pool_slots: NonZeroUsize,
        /// Sequences the recurrent state pool serves: one state row each.
        seqs: NonZeroUsize,
    },
}

/// Which positions of a step produce logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogitRows {
    /// Only each row's last position, the one a sampler reads.
    Last,
    /// Every position of the step (a speculative verify, a perplexity pass).
    All,
}

/// The semantic shape of one traced step: `rows` sequences of `tokens` new tokens each, against a
/// cache of `capacity` positions.
///
/// Built with [`StepShape::new`] and the `with_*` setters, so a new field never breaks a call site: the
/// struct is `#[non_exhaustive]`, and a literal outside this crate does not compile.
///
/// ```compile_fail
/// use std::num::NonZeroUsize;
/// use poot_models::model::{KvLayout, LogitRows, StepShape};
///
/// let one = NonZeroUsize::MIN;
/// let _ = StepShape {
///     rows: one,
///     tokens: one,
///     capacity: one,
///     kv: KvLayout::Contiguous,
///     logits: LogitRows::Last,
///     hidden: None,
///     row_extent: None,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct StepShape {
    /// Sequences in the step.
    pub rows: NonZeroUsize,
    /// New tokens each sequence contributes.
    pub tokens: NonZeroUsize,
    /// Cache positions each sequence can hold; an upper bound on its position after the step.
    pub capacity: NonZeroUsize,
    /// How the KV cache is laid out.
    pub kv: KvLayout,
    /// Which positions produce logits.
    pub logits: LogitRows,
    /// The hidden-state carry the step also writes, when its caller asks for one.
    pub hidden: Option<HiddenCarry>,
    /// The padded leading extent of the step's per-row request inputs
    /// ([`Per::Row`](crate::io::Per::Row)): a vision tower's patch count, an encoder-decoder's source
    /// length. `None` for a step none of whose inputs is a padded row tensor.
    pub row_extent: Option<NonZeroUsize>,
}

impl StepShape {
    /// A step of `rows` sequences of `tokens` new tokens against `capacity` cache positions, over a
    /// contiguous cache, returning the last position's logits, carrying no hidden state and with no
    /// row extent.
    pub const fn new(rows: NonZeroUsize, tokens: NonZeroUsize, capacity: NonZeroUsize) -> Self {
        Self {
            rows,
            tokens,
            capacity,
            kv: KvLayout::Contiguous,
            logits: LogitRows::Last,
            hidden: None,
            row_extent: None,
        }
    }

    /// Whether the step is a decode step: one token per row, the last position's logits. A property of
    /// the shape, not a program of its own: the body that traces it is the one every step runs.
    pub fn is_decode(&self) -> bool {
        self.tokens.get() == 1 && self.logits == LogitRows::Last
    }

    /// This shape over `kv`.
    #[must_use]
    pub const fn with_kv(mut self, kv: KvLayout) -> Self {
        self.kv = kv;
        self
    }

    /// This shape producing logits at `logits`.
    #[must_use]
    pub const fn with_logits(mut self, logits: LogitRows) -> Self {
        self.logits = logits;
        self
    }

    /// This shape also writing the hidden-state carry `hidden`.
    #[must_use]
    pub const fn with_hidden(mut self, hidden: Option<HiddenCarry>) -> Self {
        self.hidden = hidden;
        self
    }

    /// This shape with padded row inputs of `row_extent` leading positions.
    #[must_use]
    pub const fn with_row_extent(mut self, row_extent: Option<NonZeroUsize>) -> Self {
        self.row_extent = row_extent;
        self
    }
}

/// A request that a step write its final hidden state, at every position of the step and before any
/// last-token slice, into a per-sequence carry a later program of the same driver step reads (the
/// MTP draft source, POOT-1109). The carry is one f32 `[seqs, positions, width]` value; `positions`
/// is the driver's one extent for it (the verify window or the widest prefill piece, whichever is
/// larger), so every program of an executable declares the same aval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HiddenCarry {
    /// The carry's position extent.
    pub positions: NonZeroUsize,
}

/// One traced step: which program, at what shape, and which carriable request inputs (cadence
/// [`Covers`](crate::io::Cadence::Covers) or [`AtStart`](crate::io::Cadence::AtStart)) the step
/// binds. A carriable input is declared by the program variant a step runs when it carries it, so
/// input variation is data over the schema and never a program of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepSpec {
    /// The program to trace: one of the schema's.
    pub program: ProgramId,
    /// The step's shape.
    pub shape: StepShape,
    /// The carriable inputs the step binds, ascending.
    pub carried: Vec<InputId>,
}

impl StepSpec {
    /// A step of `program` at `shape` carrying no carriable input.
    pub fn new(program: ProgramId, shape: StepShape) -> Self {
        Self {
            program,
            shape,
            carried: Vec::new(),
        }
    }

    /// A step of the `step` program (serve a sequence) at `shape`.
    pub fn step(shape: StepShape) -> Self {
        Self::new(ProgramId::STEP, shape)
    }

    /// A step of the `encode` program (one stateless pass over whole rows) at `shape`.
    pub fn encode(shape: StepShape) -> Self {
        Self::new(ProgramId::ENCODE, shape)
    }

    /// A step of the `draft` program (the draft head over the hidden-state carry) at `shape`.
    pub fn draft(shape: StepShape) -> Self {
        Self::new(ProgramId::DRAFT, shape)
    }

    /// This step also carrying `input`.
    #[must_use]
    pub fn carrying(mut self, input: InputId) -> Self {
        if let Err(at) = self.carried.binary_search(&input) {
            self.carried.insert(at, input);
        }
        self
    }

    /// Whether the step carries `input`.
    pub fn carries(&self, input: InputId) -> bool {
        self.carried.binary_search(&input).is_ok()
    }
}

/// One model behind the driver: its config, its input schema, its weights and its tracer.
pub trait Model: Send + Sync + fmt::Debug {
    /// The family-neutral facts the driver needs.
    fn config(&self) -> &ModelConfig;

    /// What the model's programs read, carry and require: the contract the driver binds every input
    /// of a traced program by.
    fn schema(&self) -> &ModelSchema;

    /// The model's weight map: every typed weight id the tracer declares, with its handle (shape and
    /// format).
    fn weights(&self) -> &WeightMap;

    /// The body of one traced step: trace `spec.program` at `spec.shape` into `b`, a builder already
    /// charging the caller's [`TraceLimits`]. A shape the family cannot trace is a typed refusal,
    /// never a panic. Callers use `<dyn Model>::trace`, which checks the step against the schema and
    /// its token count first; the trait has no `trace` method a family could override, so the
    /// admission cannot be skipped.
    fn trace_into(&self, spec: &StepSpec, b: Builder) -> Result<Graph, TraceError>;
}

impl dyn Model + '_ {
    /// The graph of one step `step` under the caller's `limits`.
    ///
    /// A step of a program the schema does not declare, or one carrying an input that program does not
    /// read as a carriable one, is refused before the tracer body runs. So is a step of more tokens
    /// than `limits.max_tokens`; every value, equation, literal and owned allocation the body makes is
    /// charged against the same limits, so a refusal names the exact resource and nothing past it was
    /// built. This is an inherent method of the trait object, so no family can replace it:
    ///
    /// ```compile_fail
    /// use poot_graph_ir::{Builder, Graph, TraceLimits};
    /// use poot_models::io::ModelSchema;
    /// use poot_models::model::{Model, ModelConfig, StepSpec, TraceError};
    /// use poot_quant::weights::WeightMap;
    ///
    /// #[derive(Debug)]
    /// struct Skips;
    /// impl Model for Skips {
    ///     fn config(&self) -> &ModelConfig { unimplemented!() }
    ///     fn schema(&self) -> &ModelSchema { unimplemented!() }
    ///     fn weights(&self) -> &WeightMap { unimplemented!() }
    ///     fn trace_into(&self, _: &StepSpec, _: Builder) -> Result<Graph, TraceError> {
    ///         unimplemented!()
    ///     }
    ///     // a family cannot provide its own admission-free entry point
    ///     fn trace(&self, _: &StepSpec, _: TraceLimits) -> Result<Graph, TraceError> {
    ///         unimplemented!()
    ///     }
    /// }
    /// ```
    pub fn trace(&self, step: &StepSpec, limits: TraceLimits) -> Result<Graph, TraceError> {
        let schema = self.schema();
        let family = self.config().family;
        let program = schema
            .program(step.program)
            .ok_or(TraceError::ProgramUnsupported {
                family,
                program: step.program,
            })?;
        let carriable = schema.carriable(program);
        for &input in &step.carried {
            let carriable_here = schema
                .index_of(input)
                .is_some_and(|index| carriable.contains(index));
            if !carriable_here {
                return Err(TraceError::NotCarriable {
                    family,
                    program: step.program,
                    input,
                });
            }
        }
        limits.admit_tokens(step.shape.tokens)?;
        let graph = self.trace_into(step, Builder::new(limits))?;
        program
            .check_states(graph.state_pairs().map(|pair| pair.role))
            .map_err(|source| TraceError::StateMismatch {
                family,
                program: step.program,
                source,
            })?;
        Ok(graph)
    }
}

/// Why a config field cannot configure a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigReason {
    /// The field is absent and has no default.
    Missing,
    /// The field is present with a value of another JSON/GGUF type.
    WrongType,
    /// A count that must be positive is zero.
    Zero,
    /// The value is not a multiple of `by`.
    NotDivisible {
        /// The divisor the value must be a multiple of.
        by: usize,
    },
    /// Above the largest value the field admits.
    Exceeds {
        /// The largest value the field admits.
        max: usize,
    },
    /// A float that must be finite and positive (an epsilon, a RoPE base or factor).
    NotFinitePositive,
    /// A value this family does not serve (a variant the checkpoint selects, an option it sets).
    Unsupported,
    /// A per-layer list or string whose length is not the layer count.
    Length {
        /// The layer count the list or string had to match.
        expected: usize,
    },
    /// A value that must equal a width the rest of the config fixes (a head width, a rotary width).
    Mismatch {
        /// The width the rest of the config fixes.
        expected: usize,
        /// The width the field carries.
        found: usize,
    },
    /// A sum or product that does not fit `usize`.
    Overflow,
}

/// Why a family refuses a step shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeReason {
    /// The step's cache capacity exceeds the checkpoint's `max_positions`.
    CapacityAboveMax {
        /// The checkpoint's `max_positions`.
        max: usize,
    },
    /// A step of more new tokens than its cache holds.
    TokensAboveCapacity,
    /// A prefill whose token count is not a multiple of the family's prefill granule
    /// ([`ModelConfig::prefill_granule`]).
    Granule {
        /// The granule the token count had to be a multiple of.
        granule: usize,
    },
    /// A carried state value whose leading axis is not its row axis: its leading axis has `leading`
    /// entries where it should have `rows` (the step's rows, or under a paged layout a recurrent
    /// state's one row per sequence the pool serves). A paged KV pool is the one exception, and only as
    /// the pooled `Positional` cache on axis 0.
    StateRowAxis {
        /// The leading axis the value has.
        leading: usize,
        /// The row count it should have.
        rows: usize,
    },
    /// A routed step whose `rows * tokens * top_k` expert rows exceed what the expert chain
    /// indexes exactly.
    RoutedRows {
        /// The most expert rows the chain indexes exactly.
        max: usize,
    },
    /// A hidden-state carry with fewer positions than the step has tokens: the step would write
    /// positions the carry does not have.
    HiddenCarryNarrow {
        /// The carry's position extent.
        positions: usize,
        /// The step's token count.
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
        /// The extent the carry has.
        found: usize,
        /// The extent the step asked for.
        expected: usize,
    },
    /// A hidden-state carry written by a step that asked for none.
    HiddenCarryUnrequested,
    /// A draft step whose shape names no hidden-state carry: there is nothing for the layer to read.
    DraftWithoutCarry,
    /// A paged-KV step of a family whose indexer keeps a per-row block-key cache. The cache's paging
    /// granule (a page must hold whole blocks) belongs to the paged-KV owner, POOT-1148; until it
    /// decides, the indexer refuses a paged layout rather than invent one.
    PagedIndexer,
}

/// A model could not be built from its config and weights.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A config field the family cannot serve.
    #[error("{family}: config field {field}: {reason:?}")]
    Config {
        /// The family that refused.
        family: FamilyKey,
        /// The key the checkpoint used.
        field: &'static str,
        /// Why the value was refused.
        reason: ConfigReason,
    },
    /// A weight could not be mapped from the checkpoint store.
    #[error("{family}: {source}")]
    Weight {
        /// The family that refused.
        family: FamilyKey,
        /// The map builder's refusal.
        source: WeightMapError,
    },
    /// A mapped weight has another shape than the family needs.
    #[error("{family}: weight {id} has shape {found:?}, expected {expected:?}")]
    WeightShape {
        /// The family that refused.
        family: FamilyKey,
        /// The weight asked for.
        id: WeightId,
        /// The shape the family needs.
        expected: Vec<usize>,
        /// The shape the checkpoint stores.
        found: Vec<usize>,
    },
    /// A mapped weight is stored in a format the family cannot read.
    #[error("{family}: weight {id} is stored as {format:?}, which this family cannot trace")]
    WeightFormat {
        /// The family that refused.
        family: FamilyKey,
        /// The weight asked for.
        id: WeightId,
        /// The format the checkpoint stores it in.
        format: HandleFormat,
    },
    /// The schema the family built is malformed.
    #[error("{family}: malformed input schema: {source}")]
    Schema {
        /// The family whose schema is malformed.
        family: FamilyKey,
        /// What is wrong with it.
        source: SchemaError,
    },
    /// A guarantee the family's wiring relies on does not hold.
    #[error("{family}: {source}")]
    Wiring {
        /// The family whose wiring is defective.
        family: FamilyKey,
        /// The guarantee that does not hold.
        source: WiringError,
    },
    /// A stored value the family cannot read: element `index` (row-major) of the index table `id`
    /// is `value`, outside the I32 range its tracer indexes with.
    #[error("{family}: weight {id} element {index} is {value}, outside the I32 range")]
    WeightValue {
        /// The family that refused.
        family: FamilyKey,
        /// The index table.
        id: WeightId,
        /// The row-major element that is out of range.
        index: usize,
        /// The stored value.
        value: i64,
    },
}

/// A guarantee the config reader or a step check establishes before tracing does not hold where a
/// shared component needs it: the family's wiring is defective (a parameter block a layer kind needs
/// is missing, a step input a program carries is absent). Never a property of a checkpoint; reported as a
/// value so the tracer path has no panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WiringError {
    /// A layer kind needs a parameter block the config reader read only for configs that have it.
    #[error("a layer needs the {0} parameters, which the config did not read")]
    MissingParams(ParamBlock),
    /// A program that reads the hidden-state carry was traced for a step whose shape names none:
    /// the draft check ([`check_draft_step`](crate::components::standard::check_draft_step)) refuses it.
    #[error("the step's shape names no hidden-state carry")]
    MissingHiddenCarry,
    /// An MLA attention was given a key-position selector but a plain (not low-rank) query: the
    /// family guarantees the low-rank query beside every selector.
    #[error("a key-position selector needs the low-rank query")]
    SelectorWithoutLowRankQuery,
    /// A block selection was asked of an attention that sinks, a window or ALiBi bias the scores: the
    /// selected-block kernel reads dense keys under the plain causal scores only.
    #[error("a block selection needs plain causal scores (no sinks, window or ALiBi)")]
    SelectionWithBiasedScores,
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
    LayerIndexOutOfRange {
        /// The layer index.
        layer: usize,
    },
    /// A group-limited router over the sqrt-softplus arm, which the limit refuses at construction.
    #[error("a group limit is not defined for the sqrt-softplus router")]
    GroupLimitOnSqrtSoftplus,
    /// Per-expert scores asked of a softmax router.
    #[error("only a biased top-k router scores each expert on its own")]
    ScoresOfSoftmaxRouter,
    /// A hidden-state carry written with rows of another extent than the carry holds.
    #[error("carried hidden rows are {found} wide or long where the carry holds {expected}")]
    HiddenCarryRows {
        /// The extent the carry holds.
        expected: usize,
        /// The extent the rows have.
        found: usize,
    },
    /// A sandwich layer built without its two post norms.
    #[error("a sandwich layer needs its post norms")]
    SandwichWithoutPostNorms,
    /// A vision step whose patch count does not fit the rotary table's `u32` positions.
    #[error("{patches} patches do not fit a rotary table of u32 positions")]
    PatchesOutOfRange {
        /// The step's patch count.
        patches: usize,
    },
    /// A sharded embedding table with no shard to read.
    #[error("a sharded table has no shards")]
    TableWithoutShards,
    /// An order-3 n-gram head was traced for a spec that hashes only two tokens.
    #[error("an order-3 n-gram head has no third hashed token")]
    NgramOrderThree,
    /// A grouped norm was given a LayerNorm: its scale is an RMS scale, as stored or an offset from one.
    #[error("a grouped norm has no LayerNorm form")]
    GroupedLayerNorm,
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
    /// The dense feed-forward block's parameters.
    Dense,
    /// The routed mixture-of-experts block's parameters.
    Moe,
    /// Kimi Delta Attention's parameters.
    Kda,
    /// DeepSeek sparse attention's parameters.
    Dsa,
    /// The sparse-attention indexer's parameters.
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
    /// The model's schema declares no program `program`.
    #[error("{family} has no program {program}")]
    ProgramUnsupported {
        /// The family that refused.
        family: FamilyKey,
        /// The program asked for.
        program: ProgramId,
    },
    /// The step carries an input the program does not read as a carriable one.
    #[error("{family}: program {program} cannot carry input {input}")]
    NotCarriable {
        /// The family that refused.
        family: FamilyKey,
        /// The program.
        program: ProgramId,
        /// The input the step carried.
        input: InputId,
    },
    /// The traced graph's carried states are not the ones the program declares.
    #[error("{family}: program {program}: {source}")]
    StateMismatch {
        /// The family whose schema and tracer disagree.
        family: FamilyKey,
        /// The program.
        program: ProgramId,
        /// How they disagree.
        source: StateMismatch,
    },
    /// The family cannot trace this step shape.
    #[error("{family} cannot trace shape {shape:?}: {reason:?}")]
    ShapeUnsupported {
        /// The family that refused.
        family: FamilyKey,
        /// The refused shape.
        shape: StepShape,
        /// Why the shape is refused.
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
