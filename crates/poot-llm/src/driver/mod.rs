//! The driver: one decode loop over the executor contract (Card 734, dmodel.md section 3.5, ADR-0105).
//!
//! The loop is the serving surface ([`step`]): `open`, `step`, `commit` and `release`. `Driver::generate`
//! is that surface driven for one row, with a sink for the tokens; it holds no feed, pick or layout logic
//! of its own. It knows no family and no backend. A step's graph comes from [`Model::trace`], gets
//! the sampling suffix its request needs appended ([`suffix`]), compiles through
//! `poot_graph_plan::compile_staged` for the executor's own target, and runs through the object-safe
//! [`Executor`]. A request the suffix cannot express (a guided mask, a penalty or bias, logprobs)
//! falls back to the one host sampling path ([`host_sample`]). [`Driver::beam`] is beam search over
//! the rows of a batched step, with its own contiguous entries ([`beam`]).
//!
//! # Prepared entries
//!
//! An executor entry is the retained product of one specialization: `(phase, step shape, head)` inside
//! the driver's immutable scope (one model, one target, one set of compile options). Runtime token
//! values, absolute positions, seeds and sampler parameters are bound inputs and never specialize an
//! entry; capacity sizes buffers and is part of the shape. The set of entries is finite and bounded:
//! [`PreparedSetLimits`] caps the entry count and the retained bytes, a new entry over either limit is
//! a typed [`PreparedSetRefusal`] before it is published, and an entry once prepared is never replaced
//! or evicted (the one exception is the entries of an adapter-set generation no sequence holds, dropped
//! when it unloads), so work in flight on it stays valid. [`Driver::prepare`] enumerates and warms the
//! admitted set up front; a request drawn only from that set compiles nothing, and a serving step
//! never compiles at all.
//!
//! # Scope and key identity
//!
//! The scope is immutable for a driver's life: its model (config, weight binding schema), the target
//! the executor drives, and the two [`CompileOptions`]. A different value of any of them is a different
//! driver. Inside it the key is `(Phase, StepShape, Head)`: [`StepShape`] carries rows, tokens,
//! capacity, KV layout and logit rows; [`Head`] names the suffix. Adapter sets and paging arrive with
//! Card 735 as further key fields.

pub mod beam;
pub mod block_table;
pub mod capacity;
mod caps;
pub(crate) mod chunks;
pub mod embeds;
pub mod encode;
pub mod error;
pub mod expert_residency;
mod gguf_vocab;
pub mod handle;
mod host_sample;
pub mod lora;
pub mod mrope;
pub mod oneshot;
pub mod open;
pub mod placement;
pub mod prefix_cache;
mod reorder;
pub use reorder::RowGraphError;
mod snapshot;
pub mod speculate;
pub mod step;
pub mod stops;
pub(crate) mod suffix;

#[cfg(test)]
mod device_tests;
#[cfg(test)]
mod hidden_carry_tests;
#[cfg(test)]
mod llamacpp_bisect_tests;
#[cfg(test)]
mod llamacpp_kl_tests;
#[cfg(test)]
mod placed_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod text_checkpoint_tests;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_executor::{
    Collectives, Coordinator, Device, EntryId, ExecError, ExecutableId, Executor, HostSync, NoSync,
    StepInputs,
};
use poot_graph_ir::{ForwardSlotMap, Graph, Slot, SlotKey, StateRole};
use poot_graph_plan::{
    CompileOptions, ShardRule, WeightFormats, bind_packed_weights, shard, staged::ExpertPlacement,
    staged::ReadTraffic, staged::StagedCompileError, staged::StagedProgram, staged::TensorParallel,
    staged::compile_staged,
};
use poot_models::bert::{ATTENTION_MASK_SLOT, TOKEN_TYPE_SLOT};
use poot_models::components::embed::{EMBEDS_AT_SLOT, EMBEDS_SLOT};
use poot_models::components::standard::{STATE_READ_SLOT, STATE_WRITE_SLOT};
use poot_models::idefics3::PIXELS_SLOT;
use poot_models::model::{HiddenCarry, KvLayout, Phase, ShapeReason, StepShape, TraceError};
use poot_models::mrope::MropePositionIds;
use poot_models::qwen_vl_vision::{FRAME_SLOT, POS_H_SLOT, POS_W_SLOT, WINDOW_SLOT};
use poot_profile::{Mark, Report, TimingSnapshot};
use poot_quant::weights::WeightMap;
use poot_tensor::{DType, HostTensor};

use crate::GenerationControl;
use crate::core::sampler::{Sampler, TokenLogprob};
pub use beam::{Beam, BeamError, BeamRequest};
pub use caps::{
    DriverCaps, HeadSet, InputSet, Layout, LoraCaps, PoolShape, ServingShapes, StateResidency,
    WindowCap,
};
use chunks::ChunkPolicy;
pub use embeds::PromptEmbeds;
pub use encode::{EncodeRequest, EncodeRow, Encoded};
use error::{
    DriverError, InvalidOptions, InvalidRequest, PreparedSetRefusal, SpeculationRefusal,
    Unsupported,
};
pub use expert_residency::{ExpertResidency, ResidencyCounters};
pub use handle::{LoadOptions, ModelHandle};
pub use lora::{AdapterRef, AdapterSet, AdapterSource, AdapterSpec, GenerationId, LoraIdx};
pub use oneshot::{ChatOptions, ChatSession, OneShotError, Sampling, TextRequest, generate_text};
pub use open::{BackendChoice, open_executor, open_group, open_timed_executor};
use placement::{Placed, Placement};
pub use prefix_cache::{PrefixOutcome, PrefixSkip};
pub use speculate::{
    DraftPolicy, NotSpeculated, PassTiming, PositionAcceptance, RollbackTiming, RowSpeculation,
    SpeculationCaps, SpeculationStats, StepTime,
};
use step::PickPolicy;
pub use step::{Admission, Release, RowResult, RowWork, SeqId, SeqRequest, StepRow};
pub use stops::FinishReason;
pub use suffix::Head;

/// The bound on what a driver retains: prepared entries by count, and by the bytes their charge
/// ([`DriverOptions::charge`]) reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedSetLimits {
    /// Most prepared entries the driver holds.
    pub max_entries: NonZeroUsize,
    /// Most host bytes those entries retain together, as [`DriverOptions::charge`] counts them.
    pub max_retained_bytes: NonZeroU64,
}

/// One owner of retained bytes that several prepared entries can hold at once, such as a kernel every
/// entry of one shape family shares. Identity is the key: two entries naming the same key hold the same
/// owner, whatever copies of its data their programs carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedOwner {
    /// Identity of the owner: entries naming the same key hold the one owner.
    pub key: String,
    /// Host bytes the owner holds, charged once however many entries hold it.
    pub bytes: u64,
}

/// What a program retains on the host while its entry stays prepared. The first entry to hold a shared
/// owner is charged its bytes, later holders are charged nothing for it, and the last release returns
/// them; `private` bytes belong to the one entry. Weights are shared by every entry of an executable and
/// are never charged here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    /// The shared owners this entry holds, each charged once across the entries that hold it.
    pub shared: Vec<SharedOwner>,
    /// Bytes held by this entry alone.
    pub private: u64,
}

impl Retention {
    /// A retention of `bytes` held by one entry alone.
    pub fn private(bytes: u64) -> Self {
        Self {
            shared: Vec::new(),
            private: bytes,
        }
    }
}

/// How a driver measures what a compiled program retains.
pub type RetentionCharge = fn(&StagedProgram) -> Retention;

/// The host bytes a program's graph, plans and kernel bodies retain. Its value table, equations and
/// per-equation plans are private to the entry. Each distinct kernel body, keyed by its plan key, is a
/// shared owner: entries of one executable load a kernel once however many plans name it. The bytes are
/// the body's own size; the compiled artifact is the executor's and is bounded by
/// `CompileLimits::max_artifact_bytes` where it is loaded, and device memory is not counted here.
pub fn program_retention(program: &StagedProgram) -> Retention {
    let mut retention = Retention::default();
    let mut seen = HashSet::new();
    let mut own = |key: &str, body: &poot_kernel_ir::Body, retention: &mut Retention| {
        if seen.insert(key.to_string()) {
            retention.shared.push(SharedOwner {
                key: key.to_string(),
                bytes: body_bytes(body),
            });
        }
    };
    for (_, _, p) in program.stages() {
        let graph = p.graph();
        let values = graph.values().len() * size_of::<poot_graph_ir::ValueMeta>();
        let eqns = graph.eqns().len()
            * (size_of::<poot_graph_ir::Eqn>() + size_of::<poot_graph_plan::Plan>());
        retention.private += (values + eqns) as u64;
        for (_, plan) in p.planned() {
            match plan {
                poot_graph_plan::Plan::Dispatches(dispatches) => {
                    for dispatch in dispatches.iter() {
                        own(dispatch.key(), dispatch.body(), &mut retention);
                    }
                }
                poot_graph_plan::Plan::Alias(_)
                | poot_graph_plan::Plan::View { .. }
                | poot_graph_plan::Plan::Graph => {}
            }
        }
    }
    retention
}

/// The host bytes of a kernel body: its locals, blocks and statements.
fn body_bytes(body: &poot_kernel_ir::Body) -> u64 {
    let blocks: usize = body
        .blocks
        .iter()
        .map(|block| {
            size_of::<poot_kernel_ir::BasicBlock>()
                + block.statements.len() * size_of::<poot_kernel_ir::Statement>()
        })
        .sum();
    (body.locals.len() * size_of::<poot_kernel_ir::LocalDecl>() + blocks) as u64
}

/// The finite trace limits of the workloads this crate serves, for a step of at most `max_tokens`
/// tokens. Measured on the 659 workload (qwen2.5-0.5b, 24 layers, ISL=570 prefill; POOT-733): the
/// graph holds 2,778 values and 2,389 equations, 776 bytes of scalar literals and about 0.95 MB of
/// owned storage, and a 64-layer tower of the same family 7,378 values and 6,349 equations; decode is
/// one value fewer. The values, equations and literal bytes below are 100x and more above the 24-layer
/// measurement so a deeper or recurrence-unrolled family still fits, and the owned-byte cap is 1 GiB.
/// Each is a hard cap on what one trace may build, not a target.
///
/// A caller with another workload states its own [`poot_graph_ir::TraceLimits`]; nothing here is
/// consulted unless a caller asks for it.
pub const fn serving_trace_limits(max_tokens: NonZeroUsize) -> poot_graph_ir::TraceLimits {
    poot_graph_ir::TraceLimits {
        max_tokens,
        max_values: NonZeroUsize::new(1 << 20).unwrap(),
        max_eqns: NonZeroUsize::new(1 << 20).unwrap(),
        max_literal_bytes: NonZeroU64::new(1 << 24).unwrap(),
        max_owned_bytes: NonZeroU64::new(1 << 30).unwrap(),
    }
}

/// Every choice the caller makes about a driver; there is no `Default`.
#[derive(Clone, Debug)]
pub struct DriverOptions {
    /// Compile options of the prefill programs.
    pub prefill: CompileOptions,
    /// Compile options of the decode programs, the one-token step included.
    pub decode: CompileOptions,
    /// The KV positions the driver's state holds. State is keyed by (name, aval) inside an executable,
    /// so one driver has one capacity; a request of more positions is refused. Capacity only sizes
    /// buffers: each step carries its live positions in `Slot::Pos` (the kernel-side live-length bound
    /// is Card 730's).
    pub capacity: NonZeroUsize,
    /// New tokens per full prefill step: a multiple of the model's prefill granule.
    pub prefill_chunk: NonZeroUsize,
    /// The finite limits every traced step runs under: `trace.max_tokens` is the most new tokens a
    /// traced step may carry (the limit the prefill chunk must fit), and the values, equations, literal
    /// bytes and owned storage a step's graph may hold bound its construction. A step that exceeds one
    /// is refused with the exact resource named, before the excess is built.
    pub trace: poot_graph_ir::TraceLimits,
    /// The bound on the prepared set: entry count and retained bytes.
    pub prepared: PreparedSetLimits,
    /// How a compiled program's retained host bytes are measured; [`program_retention`] is the
    /// standard measure.
    pub charge: RetentionCharge,
    /// How a decode row proposes tokens to verify ([`speculate`]).
    pub draft: DraftPolicy,
    /// The widest verify window the programs of a later step read the hidden-state carry over; `Some`
    /// makes every target program write the carry ([`StepShape::hidden`]), sized for the window or the
    /// widest prefill piece, whichever is larger. The draft policy that consumes the carry names its
    /// window ([`DraftPolicy::carry_window`]): a [`DraftPolicy::Mtp`] driver refuses options that
    /// name none or a narrower one. `None` writes none.
    pub hidden_window: Option<NonZeroUsize>,
}

/// One sequence to generate. `sampler` carries its own knobs, penalty history, constraint and seed.
pub struct GenerateRequest {
    /// The prompt's token ids; must not be empty.
    pub prompt: Vec<u32>,
    /// The most new tokens to generate; the prompt plus this must fit the driver's capacity.
    pub max_new: usize,
    /// Picks each token; its knobs, penalty history, constraint and seed live here.
    pub sampler: Sampler,
    /// Stop strings: generation ends when the output contains one, and the returned text ends
    /// before it.
    pub stops: Vec<String>,
    /// Keep generating through the model's end-of-sequence token (it is then an ordinary token of the
    /// output): for fixed-length measurements. A request that ends on EOS leaves this off.
    pub ignore_eos: bool,
}

/// The sequence that carries a conversation between its turns ([`Driver::generate_turn`]): none until
/// the first turn opens it.
#[derive(Debug, Default)]
pub struct Conversation {
    seq: Option<SeqId>,
}

impl GenerateRequest {
    /// The `max_new` of a run over a `prompt_tokens`-token prompt in a window of `max_positions`
    /// positions. `None` asks for the rest of the window, so the run ends at EOS, a stop string or the
    /// window's end; an explicit count must fit it. A prompt that leaves no position for a generated
    /// token, or a count that overruns the window, is [`InvalidRequest::Capacity`].
    pub fn max_new_within(
        prompt_tokens: usize,
        max_positions: usize,
        requested: Option<usize>,
    ) -> Result<usize, InvalidRequest> {
        if prompt_tokens == 0 {
            return Err(InvalidRequest::EmptyPrompt);
        }
        let room = max_positions.saturating_sub(prompt_tokens);
        match requested {
            Some(max_new) if max_new <= room => Ok(max_new),
            None if room > 0 => Ok(room),
            _ => Err(InvalidRequest::Capacity {
                prompt: prompt_tokens,
                max_new: requested.unwrap_or(1),
                max: max_positions,
            }),
        }
    }
}

/// One finished generation: its tokens, why it ended, and the optional logprobs and phase timing.
#[derive(Debug)]
pub struct Generation {
    /// The generated tokens, not including the end-of-sequence token that ended the run.
    pub tokens: Vec<u32>,
    /// Why generation ended.
    pub finish: FinishReason,
    /// One record per token, when the request's sampler recorded logprobs.
    pub logprobs: Vec<TokenLogprob>,
    /// What the executor measured for the prompt and for the tokens after the first; `Some` only on a
    /// driver built [`Driver::with_phase_timing`], because taking it snapshots the executor's timing.
    pub timing: Option<GenerationTiming>,
}

/// The work of one phase of a generation and the host wall time the executor measured for its steps
/// (`poot_profile::HostTiming::wall` summed over them: the engine's own clock, which excludes sampling,
/// detokenization and the sink).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseTiming {
    /// Tokens the phase covered.
    pub tokens: usize,
    /// Engine wall time of the phase's steps.
    pub wall: std::time::Duration,
    /// Weight and carried-state bytes the plan reads for this phase's steps ([`ReadTraffic`]): the
    /// engine's own count from the planned storage of each value a dispatch reads.
    pub bytes: u64,
}

impl PhaseTiming {
    /// Tokens per second of engine wall time; `None` when the executor measured no time.
    pub fn tokens_per_second(&self) -> Option<f64> {
        (!self.wall.is_zero()).then(|| self.tokens as f64 / self.wall.as_secs_f64())
    }

    /// Milliseconds of engine wall time per token; `None` for no tokens or no measured time.
    pub fn ms_per_token(&self) -> Option<f64> {
        (self.tokens > 0 && !self.wall.is_zero())
            .then(|| self.wall.as_secs_f64() * 1e3 / self.tokens as f64)
    }

    /// Bytes the plan reads per token of this phase; `None` for no tokens.
    pub fn bytes_per_token(&self) -> Option<f64> {
        (self.tokens > 0).then(|| self.bytes as f64 / self.tokens as f64)
    }

    /// The phase against `ceiling`: `None` unless both a time and a byte count were measured.
    pub fn roofline(&self, ceiling: poot_profile::Ceiling) -> Option<poot_profile::Roofline> {
        poot_profile::Roofline::of(self.bytes_per_token()?, self.ms_per_token()?, ceiling)
    }
}

impl std::ops::Add for PhaseTiming {
    type Output = Self;

    /// The work and engine time of two phases run one after the other.
    fn add(self, other: Self) -> Self {
        Self {
            tokens: self.tokens + other.tokens,
            wall: self.wall + other.wall,
            bytes: self.bytes.saturating_add(other.bytes),
        }
    }
}

impl std::ops::Sub for PhaseTiming {
    type Output = Self;

    /// The work between two readings of a running total (`later - earlier`).
    fn sub(self, earlier: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_sub(earlier.tokens),
            wall: self.wall.saturating_sub(earlier.wall),
            bytes: self.bytes.saturating_sub(earlier.bytes),
        }
    }
}

/// The prefill and decode phases of one generation. Prefill ends with the step that yields the first
/// token, so the first token belongs to prefill and decode counts the tokens that follow it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationTiming {
    /// The prompt phase, ending with the step that yields the first token.
    pub prefill: PhaseTiming,
    /// The tokens after the first.
    pub decode: PhaseTiming,
    /// The decode phase against the device's pinned ceiling; `None` for a device without one, or a
    /// decode that took no measured time.
    pub decode_roofline: Option<poot_profile::Roofline>,
}

/// Receives each generated token with the text it adds. Any `FnMut(u32, &str) -> GenerationControl`
/// is one.
pub trait TokenSink {
    /// Receive one generated token `id` and the `text` it adds; `Break` ends generation, `Continue`
    /// goes on.
    fn token(&mut self, id: u32, text: &str) -> GenerationControl;
}

impl<F: FnMut(u32, &str) -> GenerationControl> TokenSink for F {
    fn token(&mut self, id: u32, text: &str) -> GenerationControl {
        self(id, text)
    }
}

/// What the driver has done, for the prepared-set and compile-once assertions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DriverStats {
    /// Programs compiled (cold preparations; a refused one counts, a warm replay never does).
    pub compiles: u64,
    /// Entries added to the executor.
    pub entries_added: u64,
    /// Steps run, of every kind.
    pub steps: u64,
    /// Bytes the plan reads, summed over the steps run, split by whether the step's entry was a
    /// decode entry ([`ReadTraffic`]).
    pub decode_read_bytes: u64,
    /// Bytes the plan reads, summed over the prefill steps run.
    pub prefill_read_bytes: u64,
    /// Picks that took the host sampling path.
    pub host_picks: u64,
    /// What the draft policy did across the rows stepped.
    pub speculation: SpeculationStats,
    /// The same rounds by draft position: where in a round the drafts stop being kept.
    pub positions: PositionAcceptance,
    /// What a recurrent target's rollback cost the executor, on a driver built `with_phase_timing`.
    pub rollback: RollbackTiming,
    /// What the draft passes and the verify windows cost the executor, on a driver built
    /// `with_phase_timing`.
    pub passes: PassTiming,
    /// What the expert residency managers did, summed over the executables (zero without a pool).
    pub residency: ResidencyCounters,
}

/// Which entries [`Driver::prepare`] warms.
#[derive(Clone, Copy, Debug)]
pub enum Warm<'a> {
    /// Every token piece any prompt can plan to: the full chunk and the power-of-two multiples of the
    /// granule below it (the finite set the chunk plan admits).
    Admitted,
    /// Exactly the pieces these prompt lengths plan to.
    Prompts(&'a [usize]),
}

/// Whose program an entry is: the served model's, the draft model's under a
/// [`DraftPolicy::DraftModel`], the served model's beam-search entries ([`beam`]): contiguous
/// layout at the beam width's row count, which serving never selects and the caps never count, or
/// the copies of a speculative round's rollback ([`snapshot`]), which move state and run no model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ModelRole {
    Target,
    Draft,
    /// The served model's multi-token-prediction layer under [`DraftPolicy::Mtp`]: its programs run
    /// on the target's executable, over the hidden-state carry the target's programs write, so the
    /// scope is the target's and never a draft model's.
    Mtp,
    Beam,
    /// Copy recurrent rows from live state into the shadow set.
    Snapshot,
    /// Copy recurrent rows from the shadow set back into live state.
    Restore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EntryKey {
    role: ModelRole,
    phase: Phase,
    shape: StepShape,
    head: Head,
    /// The adapter-set generation this entry was prepared for; `None` is the base weights.
    adapters: Option<GenerationId>,
}

/// What a step needs of a prepared entry beyond its id: every slot its program declares, with the
/// shape a bound value must have.
#[derive(Clone, Debug)]
struct Prepared {
    id: EntryId,
    /// The executable the entry was added to.
    exe: ExecutableId,
    /// What the entry holds against the retention limit.
    held: Held,
    /// What one step of the entry reads from weights and carried state.
    traffic: ReadTraffic,
    /// Whether the entry is a [`Phase::Decode`] entry, which the read-byte counters split on.
    decode: bool,
    slots: Vec<SlotSpec>,
    /// The shape of the program's output, as traced: `[rows, positions, width]` for an encode step.
    output: Vec<usize>,
}

/// What one prepared entry holds: its own bytes, and the keys of the shared owners it references.
#[derive(Clone, Debug)]
struct Held {
    private: u64,
    owners: Vec<String>,
}

/// A retention the driver has measured but not yet recorded.
struct Reservation {
    held: Held,
    /// The bytes of each of `held.owners`, in order.
    bytes: Vec<u64>,
    /// The bytes recording it adds: its own, plus each owner no entry holds yet.
    charge: u64,
}

/// A shared owner and how many entries reference it.
#[derive(Clone, Copy, Debug)]
struct Owner {
    bytes: u64,
    holders: usize,
}

#[derive(Clone, Debug)]
struct SlotSpec {
    key: SlotKey,
    shape: Vec<usize>,
    role: SlotRole,
}

/// Which bound input a slot is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotRole {
    Token,
    Pos,
    /// `Slot::MropePosition`: the step's sectioned positions ([`StepData::mrope`]).
    MropePosition,
    /// `Slot::SlotMap` `"read"`: each row's logical position to pool slot.
    MapRead,
    /// `Slot::WriteMap` `"write"`: each flat token's pool slot, or -1 for padding.
    MapWrite,
    LoraIdx,
    Seed,
    Params,
    TopK,
    /// `Slot::Activation` `"embeds"`: the caller's embedding rows of a `PrefillEmbeds` step.
    Embeds,
    /// `Slot::Activation` `"embeds_at"`: which positions those rows replace.
    EmbedsAt,
    /// `Slot::Activation` `"token_type"`: every token's segment id (an encoder's).
    TokenType,
    /// `Slot::Activation` `"attention_mask"`: 1 for a real token, 0 for padding (an encoder's).
    AttentionMask,
    /// `Slot::Activation` `"pixels"`: the frames of a vision tower's step.
    Pixels,
    /// `Slot::Activation` `"state_read"`: the recurrent state pool row each step row continues.
    StateRead,
    /// `Slot::WriteMap` `"state_write"`: the recurrent pool row each step row's result is written to, or -1
    /// for a masked row.
    StateWrite,
    /// `Slot::Activation` `"pos_h"`, `"pos_w"`, `"window"` and `"frame"`: the grid facts of a
    /// windowed vision tower's patches ([`PatchGeometry`](poot_models::qwen_vl_vision::PatchGeometry)).
    PosH,
    PosW,
    Window,
    Frame,
    /// `Slot::Activation` `"beam_rows"`: the row each state row takes in a beam reorder.
    BeamRows,
}

impl Prepared {
    /// The slots of `graph`, each classified. A slot no role names is a graph this driver cannot bind.
    fn slots_of(graph: &Graph) -> Result<Vec<SlotSpec>, DriverError> {
        let named = |slot, tag| SlotKey::new(slot, Some(tag));
        let roles = [
            (token_slot(), SlotRole::Token),
            (pos_slot(), SlotRole::Pos),
            (
                SlotKey::new(Slot::MropePosition, None),
                SlotRole::MropePosition,
            ),
            (named(Slot::SlotMap, "read"), SlotRole::MapRead),
            (named(Slot::WriteMap, "write"), SlotRole::MapWrite),
            (SlotKey::new(Slot::LoraIdx, None), SlotRole::LoraIdx),
            (sampler_slot("seed"), SlotRole::Seed),
            (sampler_slot("params"), SlotRole::Params),
            (sampler_slot("top_k"), SlotRole::TopK),
            (activation_slot(EMBEDS_SLOT), SlotRole::Embeds),
            (activation_slot(EMBEDS_AT_SLOT), SlotRole::EmbedsAt),
            (activation_slot(TOKEN_TYPE_SLOT), SlotRole::TokenType),
            (
                activation_slot(ATTENTION_MASK_SLOT),
                SlotRole::AttentionMask,
            ),
            (activation_slot(PIXELS_SLOT), SlotRole::Pixels),
            (activation_slot(STATE_READ_SLOT), SlotRole::StateRead),
            (
                named(Slot::WriteMap, STATE_WRITE_SLOT),
                SlotRole::StateWrite,
            ),
            (activation_slot(POS_H_SLOT), SlotRole::PosH),
            (activation_slot(POS_W_SLOT), SlotRole::PosW),
            (activation_slot(WINDOW_SLOT), SlotRole::Window),
            (activation_slot(FRAME_SLOT), SlotRole::Frame),
            (reorder::beam_rows_slot(), SlotRole::BeamRows),
        ];
        let mut slots = Vec::new();
        for &(value, _) in graph.slots() {
            let meta = graph.meta(value);
            let key = meta
                .slot_key()
                .expect("a slot declared through the builder carries its key")
                .clone();
            let role = roles
                .iter()
                .find(|(known, _)| *known == key)
                .map(|&(_, role)| role)
                .ok_or_else(|| Unsupported::UnboundSlot {
                    key: key.to_string(),
                })?;
            slots.push(SlotSpec {
                key,
                shape: meta.aval.shape.clone(),
                role,
            });
        }
        Ok(slots)
    }
}

/// The values one step binds. A role the entry does not declare is simply unread.
#[derive(Debug, Default)]
struct StepData {
    tokens: Vec<i32>,
    pos: Vec<i32>,
    read: Vec<i32>,
    /// `Slot::WriteMap` `"write"`: only a checked map reaches the bind.
    write: ForwardSlotMap,
    lora: Vec<i32>,
    seed: Vec<i32>,
    params: Vec<f32>,
    top_k: Vec<i32>,
    embeds: Vec<f32>,
    embeds_at: Vec<i32>,
    token_type: Vec<i32>,
    attention_mask: Vec<i32>,
    pixels: Vec<f32>,
    state_read: Vec<i32>,
    /// `Slot::WriteMap` `"state_write"`.
    state_write: ForwardSlotMap,
    pos_h: Vec<i32>,
    pos_w: Vec<i32>,
    window: Vec<f32>,
    frame: Vec<f32>,
    beam_rows: Vec<i32>,
    /// `Slot::MropePosition`, axis-major `[3, rows * tokens]`, when some row of the step carries
    /// sectioned positions ([`mrope::MropePositions`]); `None` is every token at its text position
    /// on all three axes, which is plain RoPE at that position.
    mrope: Option<Vec<i32>>,
}

impl StepData {
    fn mrope(&self) -> Vec<i32> {
        self.mrope.clone().unwrap_or_else(|| self.pos.repeat(3))
    }
}

/// One decode loop over an executor: it owns the prepared entries, the open sequences, the paged
/// pool and the adapter registry of one model on one target. Built from a [`ModelHandle`], an
/// executor and [`DriverOptions`]; every method that serves a request goes through the [`step`]
/// surface.
pub struct Driver {
    handle: Arc<ModelHandle>,
    executor: Box<dyn Executor>,
    exe: ExecutableId,
    /// Where the model runs: the executor's devices and how the placement uses them.
    placed: Placed,
    /// The served model's weights, which a tensor placement divides by role.
    weights: Arc<WeightMap>,
    options: DriverOptions,
    chunks: ChunkPolicy,
    /// Which of the model's weights the checkpoint stores packed: put on every traced graph.
    formats: WeightFormats,
    entries: HashMap<EntryKey, Prepared>,
    /// The row-gather entries beam search reorders its state with, by row count ([`beam`]).
    reorders: HashMap<NonZeroUsize, Prepared>,
    /// The shared owners the prepared entries reference, by key.
    owners: HashMap<String, Owner>,
    /// The bytes held: every shared owner once, plus each entry's own.
    retained_bytes: u64,
    stats: DriverStats,
    /// Whether any prepared entry carries recurrent state; read from the traced graphs' state roles.
    residency: StateResidency,
    lora: lora::LoraRegistry,
    /// The executable serving each captured adapter generation that has an entry: the base weights
    /// shared with the base executable, plus the generation's stacked adapter tensors.
    adapter_exes: HashMap<GenerationId, ExecutableId>,
    /// The executable each `Encode` shape runs on: a decoder's encode graph carries a cache of its own
    /// `[rows, positions]` extent, and state is keyed by (name, aval) inside an executable, so two encode
    /// shapes cannot share one. The weights are the base weights, shared with the serving executable.
    encode_exes: HashMap<(NonZeroUsize, NonZeroUsize), ExecutableId>,
    speculation: speculate::Speculation,
    /// The paged pool, the open sequences and the published caps; absent until `prepare`.
    serving: Option<step::Serving>,
    /// The one layout `prepare` has admitted, and the caps it published.
    layout: Option<Layout>,
    caps: Option<DriverCaps>,
    /// Whether `generate` reports [`Generation::timing`] ([`Driver::with_phase_timing`]).
    time_phases: bool,
    /// The expert pool, when routed experts are paged ([`Driver::with_expert_pool`]).
    pool: Option<ExpertPool>,
    /// The hidden-state carry every target program writes: one value per driver, built once at the
    /// first `prepare` from [`DriverOptions::hidden_window`] and the widest admitted prefill piece. State is keyed by
    /// (name, aval) inside an executable, so a program that declared another extent would bind a
    /// separate zero-filled buffer and its readers would see stale rows.
    hidden: Option<HiddenCarry>,
}

/// Paged routed experts: the pool's rows per selection and one residency manager per executable (each
/// executable has resident buffers of its own, so each has its own assignments).
struct ExpertPool {
    slots: NonZeroUsize,
    managers: HashMap<ExecutableId, ExpertResidency>,
}

fn token_slot() -> SlotKey {
    SlotKey::new(Slot::Token, None)
}

fn pos_slot() -> SlotKey {
    SlotKey::new(Slot::Pos, None)
}

fn sampler_slot(tag: &str) -> SlotKey {
    SlotKey::new(Slot::Sampler, Some(tag))
}

fn activation_slot(tag: &str) -> SlotKey {
    SlotKey::new(Slot::Activation, Some(tag))
}

impl Driver {
    /// A driver over `executor`'s one device: one executable holding the model's weights, shared by
    /// every entry. Nothing uploads until an entry is added. An executor over several devices is
    /// [`Driver::placed`]'s.
    pub fn new(
        handle: Arc<ModelHandle>,
        mut executor: Box<dyn Executor>,
        options: DriverOptions,
    ) -> Result<Self, DriverError> {
        let placed = Placed::resolve(executor.target_set(), Placement::Single)?;
        let chunks = Self::check_options(&handle, &options)?;
        let weights = Arc::new(handle.model().weights().clone());
        let exe = executor.load_weights(handle.shared_store().clone(), Arc::clone(&weights))?;
        Ok(Self::assemble(
            handle, executor, exe, options, chunks, placed, weights,
        ))
    }

    /// A driver over a multi-device `coordinator`, with the model placed as `placement` says: a pipeline
    /// loads the whole weight map on every device and each stage binds what it reads; a tensor
    /// placement loads each rank's own share of every weight the rule divides
    /// ([`Coordinator::load_tensor_weights`]). Either way the executable is loaded once, and nothing
    /// uploads until an entry is added.
    ///
    /// The coordinator is taken by value and concrete because a tensor placement loads through it
    /// before it becomes the driver's `dyn Executor`.
    pub fn placed<D: Device + Collectives + 'static>(
        handle: Arc<ModelHandle>,
        mut coordinator: Coordinator<D>,
        options: DriverOptions,
        placement: Placement,
    ) -> Result<Self, DriverError> {
        let placed = Placed::resolve(coordinator.target_set(), placement)?;
        if !matches!(placed.placement(), Placement::Single) {
            let what = match options.draft {
                DraftPolicy::DraftModel { .. } => {
                    Some("a draft model, whose layers are not the target's")
                }
                DraftPolicy::Mtp { .. } => Some(
                    "an MTP layer, whose hidden-state carry and cache sit on one device past the \
                     target's stages",
                ),
                DraftPolicy::Off | DraftPolicy::PromptLookup { .. } => None,
            };
            if let Some(what) = what {
                return Err(Unsupported::Placement {
                    placement: placed.name(),
                    what,
                }
                .into());
            }
        }
        let chunks = Self::check_options(&handle, &options)?;
        let weights = Arc::new(handle.model().weights().clone());
        let store = handle.shared_store().clone();
        let exe = match placed.placement() {
            Placement::Tensor { rule } => {
                let tensor = Self::tensor_shares(&handle, &options, &placed, &weights, *rule)?;
                coordinator.load_tensor_weights(store, Arc::clone(&weights), &tensor)?
            }
            Placement::Single | Placement::Pipeline { .. } => {
                coordinator.load_weights(store, Arc::clone(&weights))?
            }
        };
        Ok(Self::assemble(
            handle,
            Box::new(coordinator),
            exe,
            options,
            chunks,
            placed,
            weights,
        ))
    }

    /// The options a driver can honor, as the chunk policy they make.
    fn check_options(
        handle: &ModelHandle,
        options: &DriverOptions,
    ) -> Result<ChunkPolicy, DriverError> {
        if options.capacity.get() > handle.config().max_positions {
            return Err(InvalidOptions::CapacityAboveMax {
                capacity: options.capacity.get(),
                max_positions: handle.config().max_positions,
            }
            .into());
        }
        if let DraftPolicy::Mtp { k } = &options.draft {
            // The layer reads the carry the verify window wrote: the options must size it for the
            // window, whichever way they were built.
            let window = options.hidden_window.map(NonZeroUsize::get);
            if window.is_none_or(|window| window < k.get() + 1) {
                return Err(InvalidOptions::MtpCarryWindow { k: k.get(), window }.into());
            }
        }
        Ok(ChunkPolicy::new(
            options.prefill_chunk,
            handle.config().prefill_granule,
            options.trace.max_tokens,
        )?)
    }

    /// Which weights each rank of a tensor placement reads cut, and on which axis: the shards the rule
    /// makes of one decode step's graph, which names every weight of the model. A program compiled
    /// later that cut a weight differently would bind the wrong shape, which the executor refuses when
    /// the entry is added.
    fn tensor_shares(
        handle: &ModelHandle,
        options: &DriverOptions,
        placed: &Placed,
        weights: &WeightMap,
        rule: ShardRule,
    ) -> Result<TensorParallel, DriverError> {
        let devices = placed.devices();
        let world = u32::try_from(devices.len())
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or(InvalidOptions::TensorWithoutDevices)?;
        let shape = StepShape {
            rows: NonZeroUsize::MIN,
            tokens: NonZeroUsize::MIN,
            capacity: options.capacity,
            kv: KvLayout::Contiguous,
            logits: poot_models::model::LogitRows::Last,
            hidden: None,
        };
        let graph = handle
            .model()
            .trace(Phase::Decode, shape, options.trace)
            .map_err(Unsupported::Trace)?;
        let formats = WeightFormats::from_weight_map(weights);
        let graph = bind_packed_weights(&graph, &formats).map_err(Unsupported::PackedBind)?;
        let sharded = shard(&graph, weights, rule, world)
            .map_err(|e| Unsupported::Compile(StagedCompileError::Shard(e)))?;
        Ok(TensorParallel {
            devices,
            shares: sharded.weights,
        })
    }

    fn assemble(
        handle: Arc<ModelHandle>,
        executor: Box<dyn Executor>,
        exe: ExecutableId,
        options: DriverOptions,
        chunks: ChunkPolicy,
        placed: Placed,
        weights: Arc<WeightMap>,
    ) -> Self {
        let formats = WeightFormats::from_weight_map(handle.model().weights());
        let speculation = speculate::Speculation::new(options.draft.clone());
        Self {
            handle,
            executor,
            exe,
            placed,
            weights,
            options,
            chunks,
            formats,
            entries: HashMap::new(),
            reorders: HashMap::new(),
            owners: HashMap::new(),
            retained_bytes: 0,
            stats: DriverStats::default(),
            residency: StateResidency::KvOnly,
            lora: lora::LoraRegistry::default(),
            adapter_exes: HashMap::new(),
            encode_exes: HashMap::new(),
            speculation,
            serving: None,
            layout: None,
            caps: None,
            time_phases: false,
            pool: None,
            hidden: None,
        }
    }

    /// Fix the driver's hidden-state carry, once: `carry_window` positions for a verify window, at
    /// least the widest prefill piece the chunk plan admits (a prompt chunk writes its positions in
    /// one step), so the carry holds every position a step writes and every program declares the same
    /// extent. A later call keeps the first value.
    fn plan_hidden_carry(&mut self) {
        if self.hidden.is_some() {
            return;
        }
        let Some(window) = self.options.hidden_window else {
            return;
        };
        let widest = self
            .chunks
            .admitted_pieces()
            .into_iter()
            .filter_map(|piece| match piece {
                chunks::Piece::Prefill(tokens) => Some(tokens),
                chunks::Piece::Decode => None,
            })
            .fold(window, Ord::max);
        self.hidden = Some(HiddenCarry { positions: widest });
    }

    /// The hidden-state carry `role`'s programs write or read: the target's, identically for every
    /// key (the MTP layer's programs read it), and none for a draft model, a beam or an encode step
    /// (its own executable). The rollback copies trace the target's decode program as their state
    /// schema, so they see the carry too (and leave it out: a `Handoff` value is never rolled back).
    fn hidden_of(&self, role: ModelRole) -> Option<HiddenCarry> {
        if matches!(
            role,
            ModelRole::Target | ModelRole::Mtp | ModelRole::Snapshot | ModelRole::Restore
        ) {
            self.hidden
        } else {
            None
        }
    }

    /// Page every routed expert table through a pool of `slots_per_layer` rows per selection
    /// (ADR-0072): the experts a step selects are made resident at a sync point before they are read,
    /// least recently used rows making room. Call it before the first step. The prefill chunk shrinks
    /// to what the pool serves (`tokens * k <= slots_per_layer`, granule-aligned), so a longer prompt
    /// runs as more chunks; a pool that cannot serve one granule, a pool not smaller than the experts
    /// and a model with no routed experts are refused.
    pub fn with_expert_pool(mut self, slots_per_layer: NonZeroUsize) -> Result<Self, DriverError> {
        let shape = StepShape {
            rows: NonZeroUsize::MIN,
            tokens: NonZeroUsize::MIN,
            capacity: self.options.capacity,
            kv: KvLayout::Contiguous,
            logits: poot_models::model::LogitRows::Last,
            hidden: None,
        };
        let graph = self
            .handle
            .model()
            .trace(Phase::Decode, shape, self.options.trace)
            .map_err(Unsupported::Trace)?;
        let graph = bind_packed_weights(&graph, &self.formats).map_err(Unsupported::PackedBind)?;
        let per_token = poot_graph_plan::routed_step_demand(&graph)
            .map_err(|e| Unsupported::Compile(e.into()))?;
        if per_token == 0 {
            return Err(InvalidOptions::PoolWithoutRoutedExperts.into());
        }
        let granule = self.handle.config().prefill_granule.get();
        let fits = slots_per_layer.get() / per_token / granule * granule;
        let chunk = NonZeroUsize::new(fits.min(self.options.prefill_chunk.get())).ok_or(
            InvalidOptions::PoolBelowGranule {
                slots: slots_per_layer.get(),
                per_token,
                granule,
            },
        )?;
        self.options.prefill_chunk = chunk;
        self.chunks = ChunkPolicy::new(
            chunk,
            self.handle.config().prefill_granule,
            self.options.trace.max_tokens,
        )?;
        self.pool = Some(ExpertPool {
            slots: slots_per_layer,
            managers: HashMap::new(),
        });
        Ok(self)
    }

    /// What the expert residency managers have done, summed over the executables: all zero without a
    /// pool.
    fn residency_counters(&self) -> ResidencyCounters {
        let mut total = ResidencyCounters::default();
        for manager in self.pool.iter().flat_map(|pool| pool.managers.values()) {
            let c = manager.counters();
            total.syncs += c.syncs;
            total.id_reads += c.id_reads;
            total.refills += c.refills;
            total.evictions += c.evictions;
            total.refill_bytes += c.refill_bytes;
        }
        total
    }

    /// Reset every executable's carried state and, with it, every residency manager: the recovery
    /// after a failed step left an executable needing a reset. A manager that outlived the device's
    /// cleared pool rows would trust rows that are gone, so the two are cleared in one call.
    pub fn reset_executables(&mut self) -> Result<(), DriverError> {
        let mut exes = vec![self.exe];
        exes.extend(self.speculation.draft.iter().map(|draft| draft.exe));
        exes.extend(self.adapter_exes.values().copied());
        exes.extend(self.encode_exes.values().copied());
        for exe in exes {
            self.reset_exe(exe)?;
        }
        Ok(())
    }

    /// The answer to a failed run of the one-row surface: `error` itself, after the executables were
    /// reset when it left one needing it (a device failure, a fault, a refused host sync or an
    /// executable already refusing steps) and nothing else depends on their state. Open sequences
    /// would lose their cache to a reset, and a paged layout's registered prefixes would point at
    /// zeroed blocks, so only a driver with no open sequence on the contiguous layout recovers;
    /// anything else is returned as it came and its caller owns the recovery.
    fn recover_after(&mut self, error: DriverError) -> DriverError {
        let needs_reset = matches!(
            error,
            DriverError::Device(
                ExecError::Device(_)
                    | ExecError::Fault { .. }
                    | ExecError::NeedsReset(_)
                    | ExecError::Host(_)
            )
        );
        let alone = self.serving.as_ref().is_none_or(|s| s.seqs.is_empty())
            && matches!(self.layout, None | Some(Layout::Contiguous));
        if !(needs_reset && alone) {
            return error;
        }
        match self.reset_executables() {
            Ok(()) => error,
            Err(reset) => reset,
        }
    }

    /// `reset_state(All)` on `exe` and a reset of its residency manager, together.
    fn reset_exe(&mut self, exe: ExecutableId) -> Result<(), DriverError> {
        self.executor
            .reset_state(exe, poot_executor::StateScope::All)?;
        if let Some(manager) = self
            .pool
            .as_mut()
            .and_then(|pool| pool.managers.get_mut(&exe))
        {
            manager.reset();
        }
        Ok(())
    }

    /// Have [`Driver::generate`] report [`Generation::timing`]. Off by default: it snapshots the
    /// executor's timing at each phase boundary, which a server that never reads it should not pay for.
    pub fn with_phase_timing(mut self) -> Self {
        self.time_phases = true;
        self
    }

    /// What the driver has done so far, with the residency manager's counters read current.
    pub fn stats(&self) -> DriverStats {
        DriverStats {
            residency: self.residency_counters(),
            ..self.stats.clone()
        }
    }

    /// The pinned bandwidth ceiling of the one device this driver runs on ([`poot_profile::Ceiling`]);
    /// `None` for a device nobody calibrated, an NPU, or a placement over several devices.
    pub fn ceiling(&self) -> Option<poot_profile::Ceiling> {
        let targets = self.executor.target_set();
        let [(_, target)] = targets.devices() else {
            return None;
        };
        poot_profile::Ceiling::pinned(&target.id().calibration_label()?)
    }

    /// The executor's own counters: recordings, replays and live memory by role.
    pub fn executor_stats(&self) -> poot_executor::ExecutorStats {
        self.executor.stats()
    }

    /// Prepared entries held.
    pub fn prepared_entries(&self) -> usize {
        self.entries.len() + self.reorders.len()
    }

    /// Bytes the prepared entries are charged.
    pub fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    fn options_for(&self, phase: Phase) -> CompileOptions {
        match phase {
            Phase::Prefill | Phase::PrefillEmbeds | Phase::Encode | Phase::Mtp => {
                self.options.prefill.clone()
            }
            Phase::Decode => self.options.decode.clone(),
        }
    }

    /// The entry for `key`, preparing it on a miss: refuse a step above the trace limit before tracing,
    /// trace, append the head, compile for the executor's target, check the retention limits, then add
    /// it. A failure at any step publishes nothing and leaves every counter where it was.
    fn prepared(&mut self, key: EntryKey) -> Result<Prepared, DriverError> {
        if let Some(entry) = self.entries.get(&key) {
            return Ok(entry.clone());
        }
        let limits = self.options.prepared;
        if self.prepared_entries() >= limits.max_entries.get() {
            return Err(PreparedSetRefusal::Entries {
                held: self.prepared_entries(),
                limit: limits.max_entries.get(),
            }
            .into());
        }
        if matches!(self.speculation.policy, DraftPolicy::Mtp { .. }) {
            // The layer drafts for the base weights, and its cache follows the serving rows only.
            if let Some(generation) = key.adapters {
                return Err(
                    Unsupported::Speculation(SpeculationRefusal::MtpAdapters(generation)).into(),
                );
            }
            if key.role == ModelRole::Beam {
                return Err(Unsupported::Speculation(SpeculationRefusal::MtpBeam).into());
            }
        }
        if let Some(generation) = key.adapters
            && !self.lora.contains(generation)
        {
            return Err(lora::LoraError::UnknownGeneration(generation).into());
        }
        let (handle, exe) = self.scope(key.role)?;
        let graph = handle
            .model()
            .trace(key.phase, key.shape, self.options.trace)
            .map_err(Unsupported::Trace)?;
        let residency = Self::check_state_rows(&handle, &graph, key.shape)?;
        if key.role == ModelRole::Draft && residency == StateResidency::Recurrent {
            return Err(Unsupported::Speculation(SpeculationRefusal::RecurrentDraft).into());
        }
        if key.role == ModelRole::Mtp && residency == StateResidency::Recurrent {
            return Err(Unsupported::Speculation(SpeculationRefusal::RecurrentMtp).into());
        }
        let graph = match key.role {
            ModelRole::Snapshot | ModelRole::Restore => {
                // A rollback copy moves state and binds no weight: its graph is the decode
                // program's state schema, rows copied by index.
                let entries = snapshot::snapshot_entries(&graph, self.options.trace)
                    .map_err(|error| Unsupported::RowGraph(Box::new(error)))?
                    .ok_or(Unsupported::RowGraph(Box::new(
                        reorder::RowGraphError::NoState,
                    )))?;
                match key.role {
                    ModelRole::Snapshot => entries.snapshot,
                    _ => entries.restore,
                }
            }
            _ => {
                let graph = bind_packed_weights(&graph, self.formats_of(key.role))
                    .map_err(Unsupported::PackedBind)?;
                suffix::with_head(graph, key.head, self.options.trace)
                    .map_err(|error| Unsupported::Graph(Box::new(error)))?
            }
        };
        let mut slots = Prepared::slots_of(&graph)?;
        let output = graph.aval(graph.result()).shape.clone();
        let mut compile = self.options_for(key.phase);
        let stack = key
            .adapters
            .map(|generation| {
                self.lora
                    .stack(generation)
                    .ok_or(lora::LoraError::UnknownGeneration(generation))
            })
            .transpose()?;
        if let Some(stack) = &stack {
            // The LoRA pass adds the index slot to the program; the entry binds it every step.
            compile = compile.with_lora(stack.plan(key.shape.rows));
            slots.push(SlotSpec {
                key: SlotKey::new(Slot::LoraIdx, None),
                shape: vec![key.shape.rows.get()],
                role: SlotRole::LoraIdx,
            });
        }
        let program = self.compile(&graph, &compile)?;
        let reservation = self.admit(&program)?;
        let exe = match (key.adapters, key.phase) {
            (Some(generation), _) => self.adapter_exe(generation)?,
            (None, Phase::Encode) if key.role == ModelRole::Target => self.encode_exe(key.shape)?,
            (None, _) => exe,
        };
        let entry = self.add_entry(exe, &program, reservation, slots, output, key.phase)?;
        if key.role != ModelRole::Draft && residency == StateResidency::Recurrent {
            self.residency = StateResidency::Recurrent;
        }
        self.entries.insert(key, entry.clone());
        Ok(entry)
    }

    /// Compile `graph` for the executor's target.
    fn compile(
        &self,
        graph: &Graph,
        options: &CompileOptions,
    ) -> Result<StagedProgram, Unsupported> {
        // The pool pages the graphs that have routed experts; a dense draft model's or an encode graph's
        // stays resident, decided here from the graph rather than by catching the pool's refusal.
        let experts = match &self.pool {
            Some(pool)
                if poot_graph_plan::routed_step_demand(graph)
                    .map_err(|e| Unsupported::Compile(e.into()))?
                    > 0 =>
            {
                ExpertPlacement::Pooled {
                    slots_per_layer: pool.slots.get(),
                }
            }
            _ => ExpertPlacement::AllResident,
        };
        compile_staged(
            graph,
            self.placed.targets(),
            &self.placed.partition(experts, &self.weights),
            options,
        )
        .map_err(|e| match e {
            StagedCompileError::Compile(poot_graph_plan::CompileError::Plan(plan)) => match *plan {
                poot_graph_plan::PlanError::Refused(refusal) => Unsupported::Plan(refusal),
                other => Unsupported::Compile(StagedCompileError::Compile(other.into())),
            },
            other => Unsupported::Compile(other),
        })
    }

    /// Count the compile of `program` and check what retaining it would hold against the limits.
    /// Nothing is recorded until [`Self::add_entry`].
    fn admit(&mut self, program: &StagedProgram) -> Result<Reservation, DriverError> {
        self.stats.compiles += 1;
        let reservation = self.reserve((self.options.charge)(program));
        let would_retain = self.retained_bytes.saturating_add(reservation.charge);
        let limit = self.options.prepared.max_retained_bytes.get();
        if would_retain > limit {
            return Err(PreparedSetRefusal::Bytes {
                charge: reservation.charge,
                held: self.retained_bytes,
                would_retain,
                limit,
            }
            .into());
        }
        Ok(reservation)
    }

    /// Add `program` to `exe` as a new entry and record its reservation.
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
        reservation: Reservation,
        slots: Vec<SlotSpec>,
        output: Vec<usize>,
        phase: Phase,
    ) -> Result<Prepared, DriverError> {
        let id = self.executor.add_entry(exe, program)?;
        self.stats.entries_added += 1;
        self.hold(&reservation);
        Ok(Prepared {
            id,
            exe,
            held: reservation.held,
            traffic: program.read_traffic(),
            decode: phase == Phase::Decode,
            slots,
            output,
        })
    }

    /// What admitting `retention` would hold and add: every shared owner not yet held, once, plus the
    /// entry's own bytes. Nothing is recorded until [`Self::hold`], so a refusal or a failed load
    /// between the two leaves no trace.
    fn reserve(&self, retention: Retention) -> Reservation {
        let mut seen = HashSet::new();
        let mut charge = retention.private;
        let mut owners = Vec::with_capacity(retention.shared.len());
        let mut bytes = Vec::with_capacity(retention.shared.len());
        for owner in retention.shared {
            if !seen.insert(owner.key.clone()) {
                continue;
            }
            if !self.owners.contains_key(&owner.key) {
                charge = charge.saturating_add(owner.bytes);
            }
            owners.push(owner.key);
            bytes.push(owner.bytes);
        }
        Reservation {
            held: Held {
                private: retention.private,
                owners,
            },
            bytes,
            charge,
        }
    }

    /// Record a reservation once its entry exists: take a hold on each shared owner and add the charge.
    fn hold(&mut self, reservation: &Reservation) {
        for (key, &bytes) in reservation.held.owners.iter().zip(&reservation.bytes) {
            self.owners
                .entry(key.clone())
                .and_modify(|owner| owner.holders += 1)
                .or_insert(Owner { bytes, holders: 1 });
        }
        self.retained_bytes = self.retained_bytes.saturating_add(reservation.charge);
    }

    /// Drop an entry's holds: its own bytes, and each shared owner it was the last holder of.
    fn drop_holds(&mut self, held: &Held) {
        let mut freed = held.private;
        for key in &held.owners {
            let Some(owner) = self.owners.get_mut(key) else {
                continue;
            };
            owner.holders -= 1;
            if owner.holders == 0 {
                freed = freed.saturating_add(owner.bytes);
                self.owners.remove(key);
            }
        }
        self.retained_bytes = self.retained_bytes.saturating_sub(freed);
    }

    /// Every carried state value's leading axis is a row axis, so per-row reset addresses a row by its
    /// leading index: the step's rows for a private cache, one row per sequence for a paged layout's
    /// recurrent pool (a step's rows reach it through the state index inputs). A paged KV pool is the
    /// one value without a row axis: the `Positional` cache whose leading axis is the pool's slots.
    /// A [`StateRole::Handoff`] value is the hidden-state carry: it must be the aval the step's
    /// [`StepShape::hidden`] names, and a step that asked for one must carry one. Returns the graph's
    /// state residency, read from its state roles, never from a family.
    fn check_state_rows(
        handle: &ModelHandle,
        graph: &Graph,
        shape: StepShape,
    ) -> Result<StateResidency, Unsupported> {
        let mut residency = StateResidency::KvOnly;
        let mut carried = false;
        for pair in graph.state_pairs() {
            let leading = graph.aval(pair.input).shape.first().copied().unwrap_or(0);
            let pool = matches!(
                (shape.kv, pair.role),
                (KvLayout::Paged { pool_slots, .. }, StateRole::Positional { axis: 0 })
                    if pool_slots.get() == leading
            );
            let rows = match (shape.kv, pair.role) {
                (KvLayout::Paged { seqs, .. }, StateRole::Recurrent | StateRole::Handoff) => {
                    seqs.get()
                }
                _ => shape.rows.get(),
            };
            if leading != rows && !pool {
                return Err(Unsupported::Trace(TraceError::ShapeUnsupported {
                    family: handle.config().family,
                    shape,
                    reason: ShapeReason::StateRowAxis { leading, rows },
                }));
            }
            match pair.role {
                StateRole::Recurrent => residency = StateResidency::Recurrent,
                StateRole::Handoff => {
                    // The hidden-state carry: an f32 `[rows, positions, width]` value of exactly the
                    // positions the step asked for. It is never rolled back, so it leaves a
                    // KV-only graph KV-only.
                    let aval = graph.aval(pair.input);
                    let reason = match (aval.shape[..].get(1), shape.hidden) {
                        _ if aval.dtype != DType::F32 || aval.shape.len() != 3 => {
                            Some(ShapeReason::HiddenCarryLayout)
                        }
                        (_, None) => Some(ShapeReason::HiddenCarryUnrequested),
                        (Some(&found), Some(asked)) if found != asked.positions.get() => {
                            Some(ShapeReason::HiddenCarryPositions {
                                found,
                                expected: asked.positions.get(),
                            })
                        }
                        _ => None,
                    };
                    if let Some(reason) = reason {
                        return Err(Unsupported::Trace(TraceError::ShapeUnsupported {
                            family: handle.config().family,
                            shape,
                            reason,
                        }));
                    }
                    carried = true;
                }
                StateRole::Positional { .. } | StateRole::Blocked { .. } => {}
            }
        }
        if shape.hidden.is_some() && !carried {
            return Err(Unsupported::Trace(TraceError::ShapeUnsupported {
                family: handle.config().family,
                shape,
                reason: ShapeReason::HiddenCarryMissing,
            }));
        }
        Ok(residency)
    }

    /// Run `entry` over `data`: bind each slot the entry declares, in its declared shape.
    fn execute(
        &mut self,
        entry: &Prepared,
        data: &StepData,
    ) -> Result<poot_executor::StepOutputs<'_>, DriverError> {
        // What one slot binds: a tensor view, or a write map, which binds only as its checked type.
        enum Bound<'a> {
            Tensor(HostTensor),
            WriteMap(&'a ForwardSlotMap),
        }
        let bound: Vec<(&SlotKey, Bound<'_>)> = entry
            .slots
            .iter()
            .map(|spec| {
                let shape = spec.shape.clone();
                let tensor = match spec.role {
                    SlotRole::MapWrite => return (&spec.key, Bound::WriteMap(&data.write)),
                    SlotRole::StateWrite => {
                        return (&spec.key, Bound::WriteMap(&data.state_write));
                    }
                    SlotRole::Token => HostTensor::i32(shape, data.tokens.clone()),
                    SlotRole::Pos => HostTensor::i32(shape, data.pos.clone()),
                    SlotRole::MropePosition => HostTensor::i32(shape, data.mrope()),
                    SlotRole::MapRead => HostTensor::i32(shape, data.read.clone()),
                    SlotRole::LoraIdx => HostTensor::i32(shape, data.lora.clone()),
                    SlotRole::Seed => HostTensor::i32(shape, data.seed.clone()),
                    SlotRole::Params => HostTensor::f32(shape, data.params.clone()),
                    SlotRole::TopK => HostTensor::i32(shape, data.top_k.clone()),
                    SlotRole::Embeds => HostTensor::f32(shape, data.embeds.clone()),
                    SlotRole::EmbedsAt => HostTensor::i32(shape, data.embeds_at.clone()),
                    SlotRole::TokenType => HostTensor::i32(shape, data.token_type.clone()),
                    SlotRole::AttentionMask => HostTensor::i32(shape, data.attention_mask.clone()),
                    SlotRole::Pixels => HostTensor::f32(shape, data.pixels.clone()),
                    SlotRole::StateRead => HostTensor::i32(shape, data.state_read.clone()),
                    SlotRole::PosH => HostTensor::i32(shape, data.pos_h.clone()),
                    SlotRole::PosW => HostTensor::i32(shape, data.pos_w.clone()),
                    SlotRole::Window => HostTensor::f32(shape, data.window.clone()),
                    SlotRole::Frame => HostTensor::f32(shape, data.frame.clone()),
                    SlotRole::BeamRows => HostTensor::i32(shape, data.beam_rows.clone()),
                };
                (&spec.key, Bound::Tensor(tensor))
            })
            .collect();
        let mut inputs = StepInputs::new();
        for (key, bound) in &bound {
            match bound {
                Bound::Tensor(tensor) => inputs.push((*key).clone(), tensor.shape(), tensor.view()),
                Bound::WriteMap(map) => inputs.push_write_map((*key).clone(), map),
            }
        }
        self.stats.steps += 1;
        // A token at position `p` attends `p + 1` cached positions: the plan's KV bytes per position
        // times that count (a block-selected cache: times at most its selected positions), on top of the
        // weights and recurrent state every step streams.
        let bytes = entry.traffic.step_bytes(
            data.pos
                .iter()
                .map(|&p| u64::from(p.max(0).cast_unsigned()) + 1),
        );
        let counter = if entry.decode {
            &mut self.stats.decode_read_bytes
        } else {
            &mut self.stats.prefill_read_bytes
        };
        *counter = counter.saturating_add(bytes);
        let mut none = NoSync;
        let host: &mut dyn HostSync = match &mut self.pool {
            Some(pool) => pool.managers.entry(entry.exe).or_default(),
            None => &mut none,
        };
        // A failed step may have left rows half refilled; the manager forgets what it assigned so the
        // next step refills (the executable's own reset clears the device side).
        match self.executor.step(entry.exe, entry.id, &inputs, host) {
            Ok(outputs) => Ok(outputs),
            Err(error) => {
                if let Some(manager) = self
                    .pool
                    .as_mut()
                    .and_then(|pool| pool.managers.get_mut(&entry.exe))
                {
                    manager.reset();
                }
                Err(error.into())
            }
        }
    }

    /// The executable `Encode` steps of `shape` run on, loaded on first use over the base weights. A failed
    /// load leaves nothing registered.
    fn encode_exe(&mut self, shape: StepShape) -> Result<ExecutableId, DriverError> {
        let extent = (shape.rows, shape.tokens);
        if let Some(&exe) = self.encode_exes.get(&extent) {
            return Ok(exe);
        }
        if self.placed.is_tensor() {
            return Err(Unsupported::Placement {
                placement: self.placed.name(),
                what: "an encode step beside the served executable",
            }
            .into());
        }
        let weights = Arc::new(self.handle.model().weights().clone());
        let exe = self
            .executor
            .load_weights(self.handle.shared_store().clone(), weights)?;
        self.encode_exes.insert(extent, exe);
        Ok(exe)
    }

    /// The executable serving `generation`, loaded on first use: the base store and map, extended with
    /// the generation's stacked adapter tensors. A failed load leaves nothing registered.
    fn adapter_exe(&mut self, generation: GenerationId) -> Result<ExecutableId, DriverError> {
        if let Some(&exe) = self.adapter_exes.get(&generation) {
            return Ok(exe);
        }
        if self.placed.is_tensor() {
            return Err(Unsupported::Placement {
                placement: self.placed.name(),
                what: "an adapter generation's executable",
            }
            .into());
        }
        let stack = self
            .lora
            .stack(generation)
            .ok_or(lora::LoraError::UnknownGeneration(generation))?;
        let (store, map) =
            stack.weights(self.handle.shared_store(), self.handle.model().weights())?;
        let exe = self.executor.load_weights(Arc::new(store), Arc::new(map))?;
        self.adapter_exes.insert(generation, exe);
        Ok(exe)
    }

    /// The model and executable `role`'s entries run: the served model's, or the loaded draft's.
    fn scope(&self, role: ModelRole) -> Result<(Arc<ModelHandle>, ExecutableId), DriverError> {
        match role {
            ModelRole::Target
            | ModelRole::Mtp
            | ModelRole::Beam
            | ModelRole::Snapshot
            | ModelRole::Restore => Ok((Arc::clone(&self.handle), self.exe)),
            ModelRole::Draft => self
                .speculation
                .draft
                .as_ref()
                .map(|draft| (Arc::clone(&draft.handle), draft.exe))
                .ok_or_else(|| InvalidRequest::NotPrepared.into()),
        }
    }

    /// Which of `role`'s weights its checkpoint stores packed.
    fn formats_of(&self, role: ModelRole) -> &WeightFormats {
        match (role, &self.speculation.draft) {
            (ModelRole::Draft, Some(draft)) => &draft.formats,
            _ => &self.formats,
        }
    }

    /// Generate one sequence on the serving surface: prepare what the request needs (nothing when it is
    /// already admitted), `open` it, prefill the prompt in the planned chunks, then decode until the
    /// model ends it, `max_new` tokens exist, a stop string appears or `sink` stops it. Each generated
    /// token goes to `sink` with the text it adds. A driver that has admitted no layout admits the
    /// contiguous single row; one that has admitted a layout runs on it. The `prepare` inside the call can
    /// compile entries on a paged driver too (one row, a new head or prompt length), bounded by the
    /// prepared-set limits; a serving `step` never compiles.
    pub fn generate(
        &mut self,
        request: GenerateRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        self.generate_with(request, None, sink)
    }

    /// [`Driver::generate`] for a prompt that supplies embedding rows in place of some tokens'
    /// embeddings (a multimodal prompt's image rows): its prefill steps bind them
    /// (`StepInput::TokensWithEmbeds`), so nothing of them is compiled into a program
    /// ([`Driver::open_with_embeds`] says what the paged layout does with them).
    pub fn generate_with_embeds(
        &mut self,
        request: GenerateRequest,
        embeds: PromptEmbeds,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        self.generate_with(request, Some((embeds, None)), sink)
    }

    /// [`Driver::generate_with_embeds`] for a model that rotates at sectioned positions: `positions`
    /// are the prompt's ([`Driver::open_with_mrope`]).
    pub fn generate_with_mrope(
        &mut self,
        request: GenerateRequest,
        embeds: PromptEmbeds,
        positions: MropePositionIds,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        self.generate_with(request, Some((embeds, Some(positions))), sink)
    }

    /// Generate the next turn of `conversation`: `request.prompt` is the whole context so far (every
    /// earlier turn, then the new one). The first turn opens the conversation's sequence; each later
    /// turn [`Driver::resume`]s it, so the K/V of the leading run of context that has not changed stays
    /// resident and only the rest is prefilled. The sequence stays open for the next turn until
    /// [`Driver::end_conversation`] or an error, which closes it.
    ///
    /// Turns can be any length, so the prepared set is the admitted one ([`Warm::Admitted`]), not the
    /// pieces one prompt plans to.
    pub fn generate_turn(
        &mut self,
        conversation: &mut Conversation,
        request: GenerateRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        let head = match suffix::rule_of(&request.sampler) {
            Some(rule) => Head::Sample(rule),
            None => Head::Logits,
        };
        let shapes = ServingShapes {
            layout: self.layout.unwrap_or(Layout::Contiguous),
            rows: &[NonZeroUsize::MIN],
            heads: &[head],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        };
        self.prepare(&shapes)?;
        let seq = match conversation.seq {
            Some(seq) => {
                self.resume(seq, &request.prompt, request.max_new)?;
                seq
            }
            None => {
                let opened = self.open(SeqRequest {
                    prompt: &request.prompt,
                    max_new: request.max_new,
                    adapter: AdapterRef::Base,
                })?;
                match opened {
                    Admission::Opened { seq, .. } => seq,
                    Admission::NoRoom { needed_blocks: 0 } => {
                        return Err(InvalidRequest::SlotsBusy.into());
                    }
                    Admission::NoRoom { needed_blocks } => {
                        return Err(InvalidRequest::NoRoom { needed_blocks }.into());
                    }
                }
            }
        };
        conversation.seq = Some(seq);
        let generation = self.decode_sequence(seq, request, sink);
        match generation {
            Err(error) => {
                conversation.seq = None;
                let _ = self.release(seq, Release::Preempted);
                Err(self.recover_after(error))
            }
            ok => ok,
        }
    }

    /// Close the sequence `conversation` holds, if it has one.
    pub fn end_conversation(&mut self, conversation: Conversation) -> Result<(), DriverError> {
        if let Some(seq) = conversation.seq {
            self.release(seq, Release::Finished)?;
        }
        Ok(())
    }

    /// Check a prompt and its `max_new` against the model before a sequence holds anything: the prompt
    /// is not empty, every id names a vocabulary row, and the prompt plus `max_new` fits the
    /// positions. Returns the positions to reserve.
    fn check_admission(&self, prompt: &[u32], max_new: usize) -> Result<usize, InvalidRequest> {
        if prompt.is_empty() {
            return Err(InvalidRequest::EmptyPrompt);
        }
        let capacity = self.options.capacity.get();
        let reserved = prompt
            .len()
            .checked_add(max_new)
            .filter(|&reserved| reserved <= capacity)
            .ok_or(InvalidRequest::Capacity {
                prompt: prompt.len(),
                max_new,
                max: capacity,
            })?;
        let vocab = self.handle.config().vocab;
        if let Some((at, &token)) = prompt
            .iter()
            .enumerate()
            .find(|&(_, &token)| token as usize >= vocab)
        {
            return Err(InvalidRequest::TokenOutOfVocab { token, at, vocab });
        }
        Ok(reserved)
    }

    fn generate_with(
        &mut self,
        request: GenerateRequest,
        embedded: Option<(PromptEmbeds, Option<MropePositionIds>)>,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        let GenerateRequest {
            prompt,
            max_new,
            sampler,
            ..
        } = &request;
        let max_new = *max_new;
        if max_new == 0 {
            return Ok(Generation {
                tokens: Vec::new(),
                finish: FinishReason::MaxTokens,
                logprobs: Vec::new(),
                timing: None,
            });
        }
        self.check_admission(prompt, max_new)?;
        let head = match suffix::rule_of(sampler) {
            Some(rule) => Head::Sample(rule),
            None => Head::Logits,
        };
        let shapes = ServingShapes {
            layout: self.layout.unwrap_or(Layout::Contiguous),
            rows: &[NonZeroUsize::MIN],
            heads: &[head],
            warm: Warm::Prompts(&[prompt.len()]),
            windows: &[],
            adapters: &[],
        };
        self.prepare(&shapes)?;
        let request_seq = SeqRequest {
            prompt,
            max_new,
            adapter: AdapterRef::Base,
        };
        let admission = match embedded {
            Some((embeds, None)) => {
                self.prepare_embeds(&shapes)?;
                self.open_with_embeds(request_seq, embeds)?
            }
            Some((embeds, Some(positions))) => {
                self.prepare_embeds(&shapes)?;
                self.open_with_mrope(request_seq, embeds, positions)?
            }
            None => self.open(request_seq)?,
        };
        let seq = match admission {
            Admission::Opened { seq, .. } => seq,
            Admission::NoRoom { needed_blocks: 0 } => {
                return Err(InvalidRequest::SlotsBusy.into());
            }
            Admission::NoRoom { needed_blocks } => {
                return Err(InvalidRequest::NoRoom { needed_blocks }.into());
            }
        };
        match self.decode_sequence(seq, request, sink) {
            Ok(generation) => {
                self.release(seq, Release::Finished)?;
                Ok(generation)
            }
            Err(error) => {
                // The run's own error is the one worth reporting; a sequence that never ran to its
                // end registers no prefix.
                let _ = self.release(seq, Release::Preempted);
                Err(self.recover_after(error))
            }
        }
    }

    /// One step of the single row `seq`.
    fn step_one(
        &mut self,
        seq: SeqId,
        work: RowWork<'_>,
        sampler: &mut Sampler,
    ) -> Result<RowResult, DriverError> {
        let mut rows = [StepRow { seq, work, sampler }];
        let mut results = self.step(&mut rows)?;
        Ok(results.pop().expect("one row in, one result out"))
    }

    /// Run `seq` from its first prefill step to the end of its generation.
    fn decode_sequence(
        &mut self,
        seq: SeqId,
        request: GenerateRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, DriverError> {
        let GenerateRequest {
            prompt,
            max_new,
            mut sampler,
            stops,
            ignore_eos,
        } = request;
        let sampler = &mut sampler;
        let handle = Arc::clone(&self.handle);
        let config = handle.config();
        let upto = prompt.len();
        let mut prefilled_tokens = 0;
        let mut logprobs = Vec::new();
        let started = self
            .time_phases
            .then(|| self.executor.stats().timing.mark());
        let read_before = (self.stats.prefill_read_bytes, self.stats.decode_read_bytes);
        let mut pending = loop {
            let result = self.step_one(seq, RowWork::Prefill { upto }, sampler)?;
            logprobs.extend(result.logprobs);
            prefilled_tokens += result.consumed;
            if !result.tokens.is_empty() {
                break result.tokens;
            }
        };

        let text = handle.text();
        let no_eos = BTreeSet::new();
        let mut stop_rule = stops::Stops::new(
            if ignore_eos { &no_eos } else { &config.eos },
            max_new,
            &stops,
        );
        let mut context = prompt;
        let gen_start = context.len();
        let prefilled = self.time_phases.then(|| self.executor.stats().timing);
        let decoding = prefilled.as_ref().map(TimingSnapshot::mark);
        let mut decoded = 0;
        let finish = loop {
            // A step returns one token, or under a draft policy the accepted drafts and one more:
            // keep tokens until one ends the sequence, and commit what was kept.
            let mut ended = None;
            let mut kept = 0;
            for &next in &pending {
                if stop_rule.is_eos(next) {
                    ended = Some(FinishReason::Eos);
                    break;
                }
                kept += 1;
                context.push(next);
                let piece = text
                    .stream_piece(&context, gen_start)
                    .map_err(|e| DriverError::Text(Box::new(e)))?;
                if sink.token(next, &piece).is_break() {
                    ended = Some(FinishReason::Cancelled);
                    break;
                }
                if let Some(reason) = stop_rule.after_token(&piece, context.len() - gen_start) {
                    ended = Some(reason);
                    break;
                }
            }
            self.commit(seq, kept, sampler)?;
            if let Some(reason) = ended {
                break reason;
            }
            let result = self.step_one(
                seq,
                RowWork::Decode {
                    ahead: &[],
                    pick: PickPolicy::Greedy,
                },
                sampler,
            )?;
            logprobs.extend(result.logprobs);
            decoded += result.tokens.len();
            pending = result.tokens;
        };
        let timing = match (&started, &prefilled, &decoding) {
            (Some(started), Some(prefilled), Some(decoding)) => {
                let finished = self.executor.stats().timing;
                let decode = PhaseTiming {
                    tokens: decoded,
                    wall: window_wall(&finished, decoding),
                    bytes: self.stats.decode_read_bytes - read_before.1,
                };
                Some(GenerationTiming {
                    decode_roofline: self.ceiling().and_then(|ceiling| decode.roofline(ceiling)),
                    decode,
                    prefill: PhaseTiming {
                        tokens: prefilled_tokens,
                        wall: window_wall(prefilled, started),
                        bytes: self.stats.prefill_read_bytes - read_before.0,
                    },
                })
            }
            _ => None,
        };
        Ok(Generation {
            tokens: context.split_off(gen_start),
            finish,
            logprobs,
            timing,
        })
    }
}

/// The host wall time of every step `snapshot` recorded after `since`, on whichever entries ran.
fn window_wall(snapshot: &TimingSnapshot, since: &Mark) -> std::time::Duration {
    let entries = snapshot.entries_with_activity_since(since);
    Report::window(snapshot, since, &entries).host.wall
}
