//! [`DriverError`]: the one failure taxonomy of the driver (Card 734). The failure
//! class is the variant; the backend and the family are data inside it. A planner [`Refusal`] stays
//! the typed `{eqn, op, dtypes, target, missing}` and is never flattened to a string.

use std::path::PathBuf;

use poot_executor::ExecError;
use poot_graph_ir::SlotMapError;
use poot_graph_plan::{Refusal, staged::StagedCompileError};
use poot_models::io::{InputError, InputId};
use poot_models::model::{ModelError, StepShape, TraceError};
use poot_models::registry::RegistryError;

use crate::core::sampler::SamplerFault;
use crate::driver::Program;
use crate::driver::features::Incompatible;
use crate::driver::lora::{AdapterError, GenerationId, LoraError};
use crate::driver::step::SeqId;
use crate::driver::suffix::Head;
use poot_text::{ChatTemplateError, TextError};

/// Why the driver cannot serve something: the model, the registry or the planner has no way to.
#[derive(Debug, thiserror::Error)]
pub enum Unsupported {
    /// The family refused to trace the step (a program or shape it does not support).
    #[error(transparent)]
    Trace(TraceError),
    /// The planner cannot lower one equation of the traced graph for the executor's target.
    #[error("{0:?}")]
    Plan(Box<Refusal>),
    /// Compilation failed for a reason other than a lowering refusal (a malformed graph, a numerics
    /// or legalization failure).
    #[error(transparent)]
    Compile(StagedCompileError),
    /// The traced graph cannot take the checkpoint's packed storage.
    #[error(transparent)]
    PackedBind(poot_graph_plan::PackedBindError),
    /// Extending the traced graph (the sampling suffix) failed: the tracer refused the suffix or it
    /// produced a malformed graph.
    #[error(transparent)]
    Graph(Box<poot_graph_ir::BuilderError>),
    /// A rollback or reorder graph over the program's carried state could not be built.
    #[error(transparent)]
    RowGraph(Box<crate::driver::reorder::RowGraphError>),
    /// No registered family names the checkpoint's config.
    #[error(transparent)]
    Registry(RegistryError),
    /// A slot the traced program declares is neither a step kind, a request input of the model's
    /// schema nor an input of a driver-side graph addition: a defect of the family's wiring.
    #[error("program {program} declares slot {key}, which no schema entry or driver pass binds")]
    UnboundSlot {
        /// The program whose graph declares the slot.
        program: poot_models::io::ProgramId,
        /// The slot key the traced program declares, which nothing binds.
        key: String,
    },
    /// The backend's devices cannot exchange values with one another, so no multi-device placement
    /// runs on it.
    #[error(
        "backend {backend} has no peer copies between its devices: a multi-device placement needs ptx"
    )]
    MultiDevice {
        /// The backend's name, as `--backend` spells it.
        backend: &'static str,
    },
    /// The draft policy cannot be served on this model or with this draft.
    #[error(transparent)]
    Speculation(SpeculationRefusal),
    /// The adapters do not stack against this model's weights.
    #[error(transparent)]
    Adapter(AdapterError),
    /// Two features of the driver or of the request cannot be combined, by the one compatibility table in
    /// `driver::features`.
    #[error(transparent)]
    Incompatible(Incompatible),
}

/// Why a [`DraftPolicy`](crate::driver::DraftPolicy) is refused at `prepare`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpeculationRefusal {
    /// The model's recurrent state advances in granules of more than one token: a rollback replays
    /// the kept prefix, whose length (the accepted drafts and the query token) is not a multiple of
    /// the granule.
    #[error(
        "speculation over recurrent state needs a prefill granule of 1; this model's is {granule}"
    )]
    RecurrentGranule {
        /// The model's prefill granule, in tokens.
        granule: usize,
    },
    /// The draft model cannot propose tokens the target verifies.
    #[error("the draft model does not fit the target: {0}")]
    DraftMismatch(DraftMismatch),
}

/// How a draft model disagrees with the target it proposes for.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DraftMismatch {
    /// A draft token id the target has no row for.
    #[error("the draft's vocabulary of {draft} exceeds the target's {target}")]
    Vocab {
        /// The draft model's vocabulary size.
        draft: usize,
        /// The target model's vocabulary size.
        target: usize,
    },
    /// A draft token the target's tokenizer names differently.
    #[error("a draft token has a different id in the target's tokenizer")]
    Tokens,
    /// The draft's positions cannot hold the target's cache capacity.
    #[error("the draft holds {draft} positions, fewer than the driver's capacity {capacity}")]
    Positions {
        /// The draft model's position count.
        draft: usize,
        /// The driver's KV capacity, which the draft must at least match.
        capacity: usize,
    },
}

/// A request that cannot run, whatever the model or device.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidRequest {
    /// A prompt of no tokens: there is nothing to prefill and no last position to pick from.
    #[error("the prompt is empty")]
    EmptyPrompt,
    /// The prompt and its reserved new tokens do not fit the model's positions.
    #[error("prompt of {prompt} tokens plus max_new {max_new} exceeds the model's {max} positions")]
    Capacity {
        /// Prompt length in tokens.
        prompt: usize,
        /// The request's `max_new`.
        max_new: usize,
        /// The positions the model has.
        max: usize,
    },
    /// A token id at or past the model's vocabulary: it names no embedding row.
    #[error("token id {token} at prompt position {at} is outside the vocabulary of {vocab} tokens")]
    TokenOutOfVocab {
        /// The offending id.
        token: u32,
        /// Its position in the prompt.
        at: usize,
        /// The model's vocabulary size.
        vocab: usize,
    },
    /// The id names no open sequence: never opened, or already released.
    #[error("sequence {0:?} is not open")]
    UnknownSeq(SeqId),
    /// One sequence named by two rows of the same step.
    #[error("sequence {0:?} appears twice in one step")]
    DuplicateRow(SeqId),
    /// A step with no rows.
    #[error("a step needs at least one row")]
    EmptyStep,
    /// The step has more rows than any prepared entry holds.
    #[error("a step of {rows} rows exceeds the {max} the prepared shapes admit")]
    TooManyRows {
        /// Rows the step carries.
        rows: usize,
        /// The most rows any prepared entry holds.
        max: usize,
    },
    /// The previous step's tokens for this sequence were never committed.
    #[error("sequence {0:?} has an uncommitted result")]
    UncommittedResult(SeqId),
    /// A speculative round folded its kept prefix into this sequence's recurrent state, and the
    /// caller then committed fewer tokens than that result held. The state cannot step back, so the
    /// sequence can only be released (or resumed from a reset).
    #[error(
        "sequence {0:?} committed fewer tokens than a verified result, and its recurrent state holds the rest"
    )]
    RecurrentStateAhead(SeqId),
    /// A prefill row asked for a position outside what remains of its prompt.
    #[error("prefill of sequence {seq:?} to {upto} is outside prompt positions {done}..={prompt}")]
    PrefillRange {
        /// The sequence the row names.
        seq: SeqId,
        /// Prompt positions already fed.
        done: usize,
        /// The position the row asked to feed up to, exclusive.
        upto: usize,
        /// The prompt's length in tokens.
        prompt: usize,
    },
    /// A decode row for a sequence that has not finished prefilling and committed its first token.
    #[error("sequence {0:?} decodes before its prompt is prefilled and a first token committed")]
    DecodeBeforePrefill(SeqId),
    /// The step would write a position past the blocks reserved at `open` (more than `max_new` tokens).
    #[error("sequence {0:?} runs past the positions reserved at open")]
    BeyondReservation(SeqId),
    /// An explicit verify window is a greedy pick per position; a sampled window is a draft policy's.
    #[error("sequence {0:?} asks for a verify window with a sampler that is not greedy")]
    WindowNeedsGreedy(SeqId),
    /// One step is one entry on one executable, and a generation's executable holds only the
    /// sequences opened under it.
    #[error(
        "one step cannot mix rows of different adapter generations, or base rows with adapter rows"
    )]
    MixedAdapters,
    /// A commit keeps more tokens than the last step produced for the row.
    #[error("commit of {kept} tokens exceeds the {produced} the last step produced")]
    CommitBeyondResult {
        /// Tokens the caller asked to commit.
        kept: usize,
        /// Tokens the last step produced for the row.
        produced: usize,
    },
    /// A verify window beside a row whose pick the greedy suffix cannot reproduce: the whole step runs
    /// one greedy suffix entry, so the other row would silently get an unmasked argmax.
    #[error(
        "sequence {0:?} picks with more than the greedy suffix, in a step that carries a verify window"
    )]
    WindowStepNeedsGreedy(SeqId),
    /// A chained step mixes rows that do not chain together: a row that is not a chain, or chains of
    /// different lengths.
    #[error("the rows of a chained step all chain the same number of steps")]
    MixedChain,
    /// A chained step needs a sampling head whose pick the device can carry into the next step: a plain
    /// or adjusted draw with no logprobs and no guided mask.
    #[error("a chained step cannot run on {head:?}")]
    ChainHead {
        /// The head the rows' samplers need.
        head: Option<Head>,
    },
    /// The rows of a step ask more of the sampler than any prepared sampling head serves: a draw rule,
    /// an adjustment or logprobs that no head in the prepared set covers. Every sampling rule runs on
    /// the device, so there is no host pick to fall back to; the caller prepares the head it serves.
    #[error("no prepared sampling head serves {need:?}")]
    HeadNotPrepared {
        /// What the step's picking rows ask of the head, joined.
        need: poot_graph_ir::ops::sampling::SampleSpec,
    },
    /// `generate` on a paged driver whose pool cannot hold the request.
    #[error("the paged pool cannot hold the request: {needed_blocks} more blocks needed")]
    NoRoom {
        /// Blocks the request needs beyond those free.
        needed_blocks: usize,
    },
    /// `generate` while every sequence slot is taken: the one slot of the contiguous layout, or each
    /// slot of the paged pool. No block count would help.
    #[error("every sequence slot is taken: release a sequence first")]
    SlotsBusy,
    /// The adapter reference the request names is not loaded, or its index is out of range.
    #[error(transparent)]
    Adapter(LoraError),
    /// The request's input values do not satisfy the model's schema for the program that serves it.
    #[error(transparent)]
    Input(InputError),
    /// The request carries an input no prepared entry of its weights binds: the deployment was not
    /// prepared for it ([`Driver::prepare_carried`](crate::driver::Driver::prepare_carried)).
    #[error("no prepared entry binds input {input} under the request's weights")]
    InputNotPrepared {
        /// The input the request carries.
        input: InputId,
    },
    /// An encode request the driver cannot batch into one step.
    #[error(transparent)]
    Encode(EncodeError),
}

/// Why an encode request is refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EncodeError {
    /// An encode request with no rows.
    #[error("an encode request needs at least one row")]
    Empty,
    /// One step has one kind of row: every row's extent and carried inputs are the same.
    #[error("an encode request cannot mix rows of different extents or inputs")]
    Mixed,
    /// A row whose padded per-row inputs disagree on their leading extent.
    #[error("encode row {row}: its per-row inputs have different leading extents")]
    RaggedRow {
        /// Index of the row in the request.
        row: usize,
    },
    /// A row with no tokens and no extent of its own.
    #[error("encode row {row} has no tokens")]
    EmptyRow {
        /// Index of the empty row in the request.
        row: usize,
    },
    /// The step's cache aliases an open sequence's.
    #[error("encode shares the cache of an open sequence: release it first")]
    SequenceOpen,
    /// A request of more token rows than the prepared shapes hold.
    #[error("an encode request of {rows} rows exceeds the {max} rows the engine is fitted for")]
    TooManyRows {
        /// Rows the request carries.
        rows: usize,
        /// Rows the prepared shapes hold.
        max: usize,
    },
    /// A token row longer than the longest prepared length.
    #[error("encode row {row} has {tokens} tokens, over the {max} the engine is fitted for")]
    RowTooLong {
        /// Index of the row in the request.
        row: usize,
        /// Tokens the row has.
        tokens: usize,
        /// The longest prepared length.
        max: usize,
    },
    /// Token rows on a driver that admitted no encode shapes: they are fitted at startup, never prepared on first use.
    #[error("the driver admitted no encode shapes: token rows are prepared before serving")]
    NotPrepared,
}

/// A [`DriverOptions`](crate::driver::DriverOptions) value the model cannot honor.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidOptions {
    /// A chain of 256 or more steps: a step of a chain is numbered in a byte.
    #[error("a chained dispatch is at most 255 steps")]
    ChainTooLong,
    /// An input to prepare entries for that the model's `step` program does not read as a carriable
    /// one (a per-token input read at every step needs no entry of its own).
    #[error("input {input} is not a carriable input of the model's step program")]
    NotCarriable {
        /// The input named.
        input: InputId,
    },
    /// More carriable inputs than one deployment prepares combinations of.
    #[error("{count} carriable inputs; a deployment prepares at most {max}")]
    TooManyCarried {
        /// The inputs named.
        count: usize,
        /// The most it prepares.
        max: usize,
    },
    /// Every prefill chunk starts at and is sized in multiples of the model's granule.
    #[error("prefill_chunk {chunk} is not a multiple of the model's prefill granule {granule}")]
    ChunkNotGranuleAligned {
        /// The requested `prefill_chunk`.
        chunk: usize,
        /// The model's prefill granule.
        granule: usize,
    },
    /// The KV capacity cannot exceed the positions the model has.
    #[error("capacity {capacity} exceeds the model's {max_positions} positions")]
    CapacityAboveMax {
        /// The requested KV capacity.
        capacity: usize,
        /// The positions the model has.
        max_positions: usize,
    },
    /// A step may not carry more new tokens than the trace limit admits.
    #[error("prefill_chunk {chunk} exceeds the trace limit of {max_tokens} tokens per step")]
    ChunkAboveTraceLimit {
        /// The requested `prefill_chunk`.
        chunk: usize,
        /// The most new tokens one traced step may carry.
        max_tokens: usize,
    },
    /// A single placement runs on exactly one device.
    #[error("a single placement runs on one device; the executor names {devices}")]
    Devices {
        /// Devices the executor names.
        devices: usize,
    },
    /// A pipeline needs a device for each of its stages, and at least two of them.
    #[error(
        "a pipeline cut into {stages} stages cannot run over {devices} devices (at least two, one per stage)"
    )]
    PipelineStages {
        /// Stages the pipeline cut makes.
        stages: usize,
        /// Devices the executor names.
        devices: usize,
    },
    /// A tensor placement divides the weights over the executor's devices: there is nothing to divide
    /// over zero.
    #[error("a tensor placement needs at least one device")]
    TensorWithoutDevices,
    /// A driver admits one KV layout, and one pool size: state avals pin one cache per driver.
    #[error("the driver already admitted a different KV layout or pool size")]
    LayoutChanged,
    /// A driver admits one set of encode shapes: its entries pin one row count and one longest row.
    #[error("the driver already admitted different encode shapes")]
    EncodeShapesChanged,
    /// A verify window is a query token plus at least one draft.
    #[error("a verify window of width {width} is narrower than two")]
    WindowBelowTwo {
        /// The requested window width.
        width: usize,
    },
    /// The contiguous layout is one private cache: one row.
    #[error("the contiguous layout serves one row")]
    ContiguousServesOneRow,
    /// A pooled driver needs a routed model: the graph has no expert selection to page.
    #[error("the model has no routed experts to pool")]
    PoolWithoutRoutedExperts,
    /// One step of the smallest chunk selects more experts than the pool has rows.
    #[error(
        "a pool of {slots} rows cannot serve one granule of {granule} tokens at {per_token} experts per token"
    )]
    PoolBelowGranule {
        /// Rows the expert pool has.
        slots: usize,
        /// Experts each token selects.
        per_token: usize,
        /// The model's prefill granule in tokens.
        granule: usize,
    },
    /// An MTP policy reads the hidden-state carry of its verify window: the options name none, or
    /// one narrower than the window (`k + 1` positions).
    #[error("an MTP policy of k = {k} needs a hidden-state carry window of at least {} positions, got {window:?}", k + 1)]
    MtpCarryWindow {
        /// Drafts per round the policy proposes.
        k: usize,
        /// The carry window the options name, if any.
        window: Option<usize>,
    },
    /// A timing bound of zero would disable what it bounds: a detailed run needs every bound positive.
    #[error("the timing bound `{bound}` must be positive, got 0")]
    TimingBoundZero {
        /// Name of the zero bound.
        bound: &'static str,
    },
}

/// A new prepared entry the retention budget refuses. The entries already prepared stay usable and
/// the counters are unchanged.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PreparedSetRefusal {
    /// More prepared entries than the entry limit allows.
    #[error("the prepared set already holds {held} entries (limit {limit})")]
    Entries {
        /// Entries already prepared.
        held: usize,
        /// The entry-count limit.
        limit: usize,
    },
    /// More retained bytes than the byte limit allows.
    #[error(
        "preparing {charge} more bytes would retain {would_retain}, over the limit of {limit} (held {held})"
    )]
    Bytes {
        /// Bytes the new entry would add.
        charge: u64,
        /// Bytes the prepared set already retains.
        held: u64,
        /// Bytes the set would retain with the new entry.
        would_retain: u64,
        /// The retained-byte limit.
        limit: u64,
    },
    /// A step needs an entry `prepare` never admitted: new static schemas are admitted explicitly,
    /// never compiled by a step.
    #[error("no prepared entry for {program:?} {shape:?} {head:?} (adapters {adapters:?})")]
    NotAdmitted {
        /// The driver service the step runs.
        program: Program,
        /// The step's shape.
        shape: StepShape,
        /// The sampling head the step needs.
        head: Head,
        /// The adapter generation the entry belongs to; `None` for the base weights.
        adapters: Option<GenerationId>,
    },
}

/// A checkpoint could not be read.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Reading a file of the checkpoint failed.
    #[error("{path}: {source}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A JSON file of the checkpoint did not parse.
    #[error("{path}: {source}")]
    Json {
        /// The file that failed.
        path: PathBuf,
        /// The underlying parse error.
        #[source]
        source: serde_json::Error,
    },
    /// The weight loader refused the checkpoint's tensors.
    #[error(transparent)]
    Weights(#[from] poot_load::LoadError),
    /// A checkpoint file does not match the pin its integrity manifest gives it.
    #[error(transparent)]
    Integrity(#[from] poot_load::packed_safetensors::PackedSafetensorsError),
    /// An integrity manifest was given but does not pin a weight file the load would read.
    #[error("the integrity manifest does not pin {file}")]
    Unpinned {
        /// The weight file the manifest lacks.
        file: String,
    },
    /// The tokenizer or chat template could not be built from the checkpoint.
    #[error("tokenizer: {0}")]
    Text(#[source] Box<TextError>),
    /// The chat template given in place of the checkpoint's own is refused.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),
}

/// Every failure the driver returns, by class; [`DriverError::class`] says how a caller should
/// react.
#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// A capability the model, registry or planner does not have.
    #[error(transparent)]
    Unsupported(Unsupported),
    /// The request itself cannot run.
    #[error(transparent)]
    InvalidRequest(InvalidRequest),
    /// The options the driver was built with cannot be honored.
    #[error(transparent)]
    InvalidOptions(InvalidOptions),
    /// The prepared-entry budget refused a new specialization.
    #[error(transparent)]
    PreparedSet(PreparedSetRefusal),
    /// The checkpoint does not satisfy the family.
    #[error(transparent)]
    Model(ModelError),
    /// The device or the executor contract failed: one variant for every backend.
    #[error(transparent)]
    Device(ExecError),
    /// An `Encode` step failed on the device and its executable was reset: the graph carries no state
    /// a later step reads, so the failure cost only that step's request.
    #[error("encode step failed, executable reset: {0}")]
    EncodeStep(ExecError),
    /// Non-finite or unpickable logits reached the sampler (Card 601).
    #[error(transparent)]
    Sampler(SamplerFault),
    /// The checkpoint could not be read.
    #[error(transparent)]
    Load(LoadError),
    /// The text codec could not decode the generated tokens.
    #[error("detokenization: {0}")]
    Text(#[source] Box<TextError>),
    /// A step's write map names a pool slot twice or one outside the pool: the block table handed two
    /// tokens one slot, so the step was not run.
    #[error("the step's write map is not one owner per slot: {0}")]
    WriteMap(#[from] SlotMapError),
}

/// How a [`DriverError`] relates to the request that met it and to the driver's state, which is what
/// a server needs to answer with the right status and to know whether it may keep serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureClass {
    /// The request itself cannot run (an empty or oversized prompt, an unknown adapter, rows that do not
    /// fit). Nothing was applied: the driver is as it was.
    Request,
    /// A step failed on the device, but the driver proved its state intact (a stateless step's
    /// executable was reset): that step's request is lost and the driver serves on.
    Step,
    /// The driver, as prepared, cannot serve this: a capability, a shape or a retention limit it does not
    /// have. Nothing was applied: the driver is as it was.
    Unsupported,
    /// The device, the checkpoint or a half-applied call: the driver's state cannot be trusted.
    Fault,
}

impl DriverError {
    /// The class of this failure.
    ///
    /// An adapter that does not stack against the model is the request's: the one it named does not
    /// fit. A sampler fault (no finite logit among a row's logits) is a request failure: the driver advances
    /// its sequences only after every row picked, so the step left no trace.
    pub fn class(&self) -> FailureClass {
        match self {
            DriverError::InvalidRequest(_)
            | DriverError::Sampler(_)
            | DriverError::Unsupported(Unsupported::Adapter(_)) => FailureClass::Request,
            DriverError::EncodeStep(_) => FailureClass::Step,
            DriverError::Unsupported(_)
            | DriverError::InvalidOptions(_)
            | DriverError::PreparedSet(_) => FailureClass::Unsupported,
            DriverError::Model(_)
            | DriverError::Device(_)
            | DriverError::Load(_)
            | DriverError::WriteMap(_)
            | DriverError::Text(_) => FailureClass::Fault,
        }
    }
}

impl From<Unsupported> for DriverError {
    fn from(error: Unsupported) -> Self {
        DriverError::Unsupported(error)
    }
}

impl From<AdapterError> for DriverError {
    fn from(error: AdapterError) -> Self {
        DriverError::Unsupported(Unsupported::Adapter(error))
    }
}

impl From<InvalidRequest> for DriverError {
    fn from(error: InvalidRequest) -> Self {
        DriverError::InvalidRequest(error)
    }
}

impl From<InvalidOptions> for DriverError {
    fn from(error: InvalidOptions) -> Self {
        DriverError::InvalidOptions(error)
    }
}

impl From<LoraError> for DriverError {
    fn from(error: LoraError) -> Self {
        DriverError::InvalidRequest(InvalidRequest::Adapter(error))
    }
}

impl From<PreparedSetRefusal> for DriverError {
    fn from(error: PreparedSetRefusal) -> Self {
        DriverError::PreparedSet(error)
    }
}

impl From<ModelError> for DriverError {
    fn from(error: ModelError) -> Self {
        DriverError::Model(error)
    }
}

impl From<ExecError> for DriverError {
    fn from(error: ExecError) -> Self {
        DriverError::Device(error)
    }
}

impl From<SamplerFault> for DriverError {
    fn from(error: SamplerFault) -> Self {
        DriverError::Sampler(error)
    }
}

impl From<LoadError> for DriverError {
    fn from(error: LoadError) -> Self {
        DriverError::Load(error)
    }
}
