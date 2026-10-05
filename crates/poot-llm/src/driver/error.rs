//! [`DriverError`]: the one failure taxonomy of the driver (Card 734). The failure
//! class is the variant; the backend and the family are data inside it. A planner [`Refusal`] stays
//! the typed `{eqn, op, dtypes, target, missing}` and is never flattened to a string.

use std::path::PathBuf;

use poot_executor::ExecError;
use poot_graph_plan::{Refusal, StagedCompileError};
use poot_models::model::{ModelError, Phase, StepShape, TraceError};
use poot_models::registry::RegistryError;

use crate::core::sampler::SamplerFault;
use crate::driver::lora::{GenerationId, LoraError};
use crate::driver::step::SeqId;
use crate::driver::suffix::Head;
use crate::error::RunnerError;

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
    /// No registered family names the checkpoint's config.
    #[error(transparent)]
    Registry(RegistryError),
    /// A slot the traced program declares has no binding in the driver (no role names it).
    #[error("the program declares slot {key}, which the driver has no binding for")]
    UnboundSlot { key: String },
    /// This build has no executor for the requested backend.
    #[error("backend {backend:?} is not built into this binary")]
    Backend { backend: &'static str },
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
    /// A verify window is a greedy pick per position; a sampled window is a speculation policy's.
    #[error("sequence {0:?} asks for a verify window with a sampler that is not greedy")]
    WindowNeedsGreedy(SeqId),
    /// One step is one entry, compiled for one adapter-set generation.
    #[error("one step cannot mix rows of different adapter generations")]
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
    #[error(transparent)]
    Adapter(LoraError),
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
    /// The executor must drive exactly one device (the M3 contract).
    #[error("the executor names {devices} devices; the driver drives exactly one")]
    Devices { devices: usize },
    /// A driver admits one KV layout, and one pool size: state avals pin one cache per driver.
    #[error("the driver already admitted a different KV layout or pool size")]
    LayoutChanged,
    /// A verify window is a query token plus at least one draft.
    #[error("a verify window of width {width} is narrower than two")]
    WindowBelowTwo { width: usize },
    /// The contiguous layout is one private cache: one row, no verify windows.
    #[error("the contiguous layout serves one row and no verify windows")]
    ContiguousServesOneRow,
    /// A model with recurrent state has no per-row state reset yet, so it serves one sequence at a
    /// time in one row, without verify windows.
    #[error("a model with recurrent state serves one sequence in one row, without verify windows")]
    RecurrentServesOneSequence,
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
    /// The tokenizer or chat template could not be built from the checkpoint.
    #[error("tokenizer: {0}")]
    Text(#[source] Box<RunnerError>),
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
    /// Non-finite or unpickable logits reached the sampler (Card 601).
    #[error(transparent)]
    Sampler(SamplerFault),
    #[error(transparent)]
    Load(LoadError),
    /// The text codec could not decode the generated tokens.
    #[error("detokenization: {0}")]
    Text(#[source] Box<RunnerError>),
}

impl From<Unsupported> for DriverError {
    fn from(error: Unsupported) -> Self {
        DriverError::Unsupported(error)
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
