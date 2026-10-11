//! [`DriverError`]: the one failure taxonomy of the driver (Card 734). The failure
//! class is the variant; the backend and the family are data inside it. A planner [`Refusal`] stays
//! the typed `{eqn, op, dtypes, target, missing}` and is never flattened to a string.

use std::path::PathBuf;

use poot_executor::ExecError;
use poot_graph_plan::{Refusal, StagedCompileError};
use poot_models::model::{ModelError, Phase, StepShape, TraceError};
use poot_models::registry::RegistryError;

use crate::core::sampler::SamplerFault;
use crate::driver::lora::{AdapterError, GenerationId, LoraError};
use crate::driver::step::SeqId;
use crate::driver::suffix::Head;
use crate::error::{ChatTemplateError, TextError};

/// Why the driver cannot serve something: the model, the registry or the planner has no way to.
#[derive(Debug, thiserror::Error)]
pub enum Unsupported {
    /// The family refused to trace the step (a phase or shape it does not support).
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
    /// Extending the traced graph (the sampling suffix) produced a malformed graph.
    #[error(transparent)]
    Graph(Box<poot_graph_ir::EditError>),
    /// No registered family names the checkpoint's config.
    #[error(transparent)]
    Registry(RegistryError),
    /// A slot the traced program declares has no binding in the driver (no role names it).
    #[error("the program declares slot {key}, which the driver has no binding for")]
    UnboundSlot { key: String },
    /// This build has no executor for the requested backend; the backend's cargo feature of the same
    /// name (`--features {backend}`) builds it in.
    #[error(
        "backend {backend} is not built into this binary: rebuild with `--features {backend}` to enable it"
    )]
    Backend { backend: &'static str },
    /// The backend's devices cannot exchange values with one another, so no multi-device placement
    /// runs on it.
    #[error(
        "backend {backend} has no peer copies between its devices: a multi-device placement needs ptx"
    )]
    MultiDevice { backend: &'static str },
    /// The draft policy cannot be served on this model or with this draft.
    #[error(transparent)]
    Speculation(SpeculationRefusal),
    /// The adapters do not stack against this model's weights.
    #[error(transparent)]
    Adapter(AdapterError),
    /// The placement has no form of something the driver was asked for: a tensor-parallel driver loads
    /// one executable whose weights are cut per rank, so a second executable over whole weights is
    /// refused rather than served over weights the program does not bind; a draft model's layers are
    /// not the target's, so the target's pipeline cut names nothing in them.
    #[error("a {placement} placement has no form of {what}")]
    Placement {
        placement: &'static str,
        what: &'static str,
    },
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
    RecurrentGranule { granule: usize },
    /// The draft model carries recurrent state: the target's rollback does not cover the draft's, so
    /// a rejected draft token would stay folded into it.
    #[error("a draft model with recurrent state cannot be rolled back, only the target's is")]
    RecurrentDraft,
    /// The draft model cannot propose tokens the target verifies.
    #[error("the draft model does not fit the target: {0}")]
    DraftMismatch(DraftMismatch),
    /// The family's MTP program carries recurrent state. Only the target's decode program defines
    /// the rollback's snapshot schema, so MTP-only recurrent state would never be restored after a
    /// rejected draft.
    #[error(
        "the MTP program carries recurrent state, which a rejected draft cannot be rolled back out of"
    )]
    RecurrentMtp,
    /// The MTP layer has the base weights only: an adapter changes the target's effective weights
    /// and the layer would draft for a different model.
    #[error("MTP speculation serves the base weights only, not adapter set generation {0:?}")]
    MtpAdapters(GenerationId),
    /// Beam search runs the target's programs without the hidden-state carry, so the MTP layer's
    /// cache would not follow its rows.
    #[error("beam search is not served under MTP speculation")]
    MtpBeam,
}

/// How a draft model disagrees with the target it proposes for.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DraftMismatch {
    /// A draft token id the target has no row for.
    #[error("the draft's vocabulary of {draft} exceeds the target's {target}")]
    Vocab { draft: usize, target: usize },
    /// A draft token the target's tokenizer names differently.
    #[error("a draft token has a different id in the target's tokenizer")]
    Tokens,
    /// The draft's positions cannot hold the target's cache capacity.
    #[error("the draft holds {draft} positions, fewer than the driver's capacity {capacity}")]
    Positions { draft: usize, capacity: usize },
}

/// A request that cannot run, whatever the model or device.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidRequest {
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("prompt of {prompt} tokens plus max_new {max_new} exceeds the model's {max} positions")]
    Capacity {
        prompt: usize,
        max_new: usize,
        max: usize,
    },
    /// `open` and `step` serve the shapes `Driver::prepare` admitted; nothing is admitted yet.
    #[error("the driver has no serving shapes: call prepare first")]
    NotPrepared,
    #[error("sequence {0:?} is not open")]
    UnknownSeq(SeqId),
    #[error("sequence {0:?} appears twice in one step")]
    DuplicateRow(SeqId),
    #[error("a step needs at least one row")]
    EmptyStep,
    #[error("a step of {rows} rows exceeds the {max} the prepared shapes admit")]
    TooManyRows { rows: usize, max: usize },
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
    #[error("prefill of sequence {seq:?} to {upto} is outside prompt positions {done}..={prompt}")]
    PrefillRange {
        seq: SeqId,
        done: usize,
        upto: usize,
        prompt: usize,
    },
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
    #[error("commit of {kept} tokens exceeds the {produced} the last step produced")]
    CommitBeyondResult { kept: usize, produced: usize },
    /// A verify window beside a row whose pick the greedy suffix cannot reproduce: the whole step runs
    /// one greedy suffix entry, so the other row would silently get an unmasked argmax.
    #[error(
        "sequence {0:?} picks with more than the greedy suffix, in a step that carries a verify window"
    )]
    WindowStepNeedsGreedy(SeqId),
    /// `generate` on a paged driver whose pool cannot hold the request.
    #[error("the paged pool cannot hold the request: {needed_blocks} more blocks needed")]
    NoRoom { needed_blocks: usize },
    /// `generate` while every sequence slot is taken: the one slot of the contiguous layout, or each
    /// slot of the paged pool. No block count would help.
    #[error("every sequence slot is taken: release a sequence first")]
    SlotsBusy,
    /// A paged pool shares earlier turns through its prefix cache (`open` finds them); only the contiguous
    /// layout's one private cache is resumed in place.
    #[error(
        "a paged sequence cannot be resumed: open the next turn and the prefix cache reuses it"
    )]
    ResumePaged,
    /// The sequence's prompt supplied embedding rows or sectioned positions that a new prompt does not.
    #[error("sequence {0:?} was opened with prompt embeddings or positions and cannot be resumed")]
    ResumeEmbedded(SeqId),
    #[error(transparent)]
    Adapter(LoraError),
    /// The embedding rows a prompt supplies do not fit it.
    #[error(transparent)]
    Embeds(EmbedsError),
    /// The sectioned positions a prompt supplies do not fit it or its model.
    #[error(transparent)]
    Mrope(MropeError),
    /// An encode request the driver cannot batch into one step.
    #[error(transparent)]
    Encode(EncodeError),
}

/// Why a prompt's sectioned (mRoPE) positions are refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropeError {
    /// The positions cover `positions` tokens, the prompt has `prompt`.
    #[error("sectioned positions cover {positions} tokens, the prompt has {prompt}")]
    Length { positions: usize, prompt: usize },
    /// The served model rotates at plain positions: it has no `Slot::MropePosition` to bind them to.
    #[error("the model has no sectioned rotary positions: a prompt cannot supply them")]
    NotSectioned,
}

/// Why a prompt's embedding rows are refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EmbedsError {
    /// `rows` holds `rows` values, not `positions * width`.
    #[error("{positions} embedding rows of width {width} need {} values, got {rows}", positions * width)]
    Length {
        positions: usize,
        width: usize,
        rows: usize,
    },
    /// The positions must be strictly ascending and inside the prompt.
    #[error("embedding position {position} is out of order or outside the {prompt}-token prompt")]
    Position { position: usize, prompt: usize },
    /// The rows are not as wide as the model's embedding.
    #[error("embedding rows are {got} wide, the model's embedding is {want}")]
    Width { got: usize, want: usize },
    /// Only the base weights have `PrefillEmbeds` entries: an adapter sequence cannot take embedding
    /// rows.
    #[error("a prompt with embedding rows cannot run under an adapter")]
    Adapter,
    /// A draft model prefills over the prompt's tokens and has no way to take the rows, so its
    /// proposals for such a prompt would come from placeholder tokens.
    #[error("a prompt with embedding rows cannot run under a draft-model speculation policy")]
    DraftModel,
    /// The MTP layer embeds the tokens the sequence holds, which are placeholders where the prompt
    /// supplies rows.
    #[error("a prompt with embedding rows cannot run under an MTP speculation policy")]
    Mtp,
}

/// Why an encode request is refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EncodeError {
    #[error("an encode request needs at least one row")]
    Empty,
    /// One step has one kind of input: token rows or image frames.
    #[error("an encode request cannot mix token rows and image frames")]
    Mixed,
    #[error("encode row {row} has no tokens")]
    EmptyRow { row: usize },
    /// The step's cache aliases an open sequence's.
    #[error("encode shares the cache of an open sequence: release it first")]
    SequenceOpen,
    #[error("encode row {row}: {types} token types for {tokens} tokens")]
    TokenTypes {
        row: usize,
        tokens: usize,
        types: usize,
    },
    #[error("encode row {row}: a frame of {got} values, the step takes {want}")]
    FrameLength { row: usize, got: usize, want: usize },
    /// The tower does not declare the grid inputs of a patch row (it is not a windowed vision tower).
    #[error("the model takes no patch rows with a grid geometry")]
    NoGeometry,
}

/// A [`DriverOptions`](crate::driver::DriverOptions) value the model cannot honor.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidOptions {
    /// Every prefill chunk starts at and is sized in multiples of the model's granule.
    #[error("prefill_chunk {chunk} is not a multiple of the model's prefill granule {granule}")]
    ChunkNotGranuleAligned { chunk: usize, granule: usize },
    /// The KV capacity cannot exceed the positions the model has.
    #[error("capacity {capacity} exceeds the model's {max_positions} positions")]
    CapacityAboveMax {
        capacity: usize,
        max_positions: usize,
    },
    /// A step may not carry more new tokens than the trace limit admits.
    #[error("prefill_chunk {chunk} exceeds the trace limit of {max_tokens} tokens per step")]
    ChunkAboveTraceLimit { chunk: usize, max_tokens: usize },
    /// A single placement runs on exactly one device.
    #[error("a single placement runs on one device; the executor names {devices}")]
    Devices { devices: usize },
    /// A pipeline needs a device for each of its stages, and at least two of them.
    #[error(
        "a pipeline cut into {stages} stages cannot run over {devices} devices (at least two, one per stage)"
    )]
    PipelineStages { stages: usize, devices: usize },
    /// A tensor placement divides the weights over the executor's devices: there is nothing to divide
    /// over zero.
    #[error("a tensor placement needs at least one device")]
    TensorWithoutDevices,
    /// A driver admits one KV layout, and one pool size: state avals pin one cache per driver.
    #[error("the driver already admitted a different KV layout or pool size")]
    LayoutChanged,
    /// A verify window is a query token plus at least one draft.
    #[error("a verify window of width {width} is narrower than two")]
    WindowBelowTwo { width: usize },
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
        slots: usize,
        per_token: usize,
        granule: usize,
    },
    /// A caller-named verify window runs drafts the sequence may reject, and only the driver's own
    /// speculative rounds snapshot recurrent state: the rejected drafts would already have advanced it.
    #[error("a model with recurrent state cannot verify a window of drafts")]
    RecurrentHasNoVerifyWindows,
    /// An MTP policy reads the hidden-state carry of its verify window: the options name none, or
    /// one narrower than the window (`k + 1` positions).
    #[error("an MTP policy of k = {k} needs a hidden-state carry window of at least {} positions, got {window:?}", k + 1)]
    MtpCarryWindow { k: usize, window: Option<usize> },
    /// A timing bound of zero would disable what it bounds: a detailed run needs every bound positive.
    #[error("the timing bound `{bound}` must be positive, got 0")]
    TimingBoundZero { bound: &'static str },
}

/// A new prepared entry the retention budget refuses. The entries already prepared stay usable and
/// the counters are unchanged.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PreparedSetRefusal {
    #[error("the prepared set already holds {held} entries (limit {limit})")]
    Entries { held: usize, limit: usize },
    #[error(
        "preparing {charge} more bytes would retain {would_retain}, over the limit of {limit} (held {held})"
    )]
    Bytes {
        charge: u64,
        held: u64,
        would_retain: u64,
        limit: u64,
    },
    /// A step of more tokens than the driver's `max_trace_tokens` is never traced: the set admits no
    /// entry of that extent, and a longer prompt runs as legal chunks instead.
    #[error("a step of {tokens} tokens is above the trace limit of {limit} tokens per step")]
    TraceTokens { tokens: usize, limit: usize },
    /// A step needs an entry `prepare` never admitted: new static schemas are admitted explicitly,
    /// never compiled by a step.
    #[error("no prepared entry for {phase:?} {shape:?} {head:?} (adapters {adapters:?})")]
    NotAdmitted {
        phase: Phase,
        shape: StepShape,
        head: Head,
        adapters: Option<GenerationId>,
    },
}

/// A checkpoint could not be read.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Weights(#[from] poot_load::LoadError),
    /// A checkpoint file does not match the pin its integrity manifest gives it.
    #[error(transparent)]
    Integrity(#[from] poot_load::packed_safetensors::PackedSafetensorsError),
    /// An integrity manifest was given but does not pin a weight file the load would read.
    #[error("the integrity manifest does not pin {file}")]
    Unpinned { file: String },
    /// The tokenizer or chat template could not be built from the checkpoint.
    #[error("tokenizer: {0}")]
    Text(#[source] Box<TextError>),
    /// The chat template given in place of the checkpoint's own is refused.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// A capability the model, registry or planner does not have.
    #[error(transparent)]
    Unsupported(Unsupported),
    #[error(transparent)]
    InvalidRequest(InvalidRequest),
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
    #[error(transparent)]
    Load(LoadError),
    /// The text codec could not decode the generated tokens.
    #[error("detokenization: {0}")]
    Text(#[source] Box<TextError>),
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
