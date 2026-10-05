//! The driver: one decode loop over the executor contract (Card 734, dmodel.md section 3.5, ADR-0105).
//!
//! The loop is the serving surface ([`step`]): `open`, `step`, `commit` and `release`. `Driver::generate`
//! is that surface driven for one row, with a sink for the tokens; it holds no feed, pick or layout logic
//! of its own. It knows no family and no backend. A step's graph comes from [`Model::trace`], gets
//! the sampling suffix its request needs appended ([`suffix`]), compiles through
//! `poot_graph_plan::compile_staged` for the executor's own target, and runs through the object-safe
//! [`Executor`]. A request the suffix cannot express (a guided mask, a penalty or bias, logprobs)
//! falls back to the one host sampling path ([`host_sample`]).
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

pub mod block_table;
mod caps;
pub(crate) mod chunks;
pub mod error;
pub mod handle;
mod host_sample;
pub mod lora;
pub mod open;
pub mod prefix_cache;
pub mod step;
pub mod stops;
pub(crate) mod suffix;

#[cfg(test)]
mod device_tests;
#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_executor::{EntryId, ExecutableId, Executor, NoSync, StepInputs, WeightSource};
use poot_graph_ir::{Graph, Slot, SlotKey, StateRole, ValidationOutputs};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, Partition, StagedCompileError,
    StagedProgram, Target, TargetSet, WeightFormats, bind_packed_weights, compile_staged,
};
use poot_models::model::{KvLayout, Phase, ShapeReason, StepShape, TraceError};
use poot_tensor::HostTensor;

use crate::GenerationControl;
use crate::core::sampler::{Sampler, TokenLogprob};
pub use caps::{
    DriverCaps, HeadSet, InputSet, Layout, LoraCaps, PoolShape, ServingShapes, StateResidency,
};
use chunks::ChunkPolicy;
use error::{DriverError, InvalidOptions, InvalidRequest, PreparedSetRefusal, Unsupported};
pub use handle::ModelHandle;
pub use lora::{AdapterRef, AdapterSet, AdapterSource, AdapterSpec, GenerationId, LoraIdx};
pub use open::{BackendChoice, open_executor};
pub use prefix_cache::{PrefixOutcome, PrefixSkip};
use step::PickPolicy;
pub use step::{Admission, Release, RowResult, RowWork, SeqId, SeqRequest, StepRow};
pub use stops::FinishReason;
pub use suffix::Head;

/// The bound on what a driver retains: prepared entries by count, and by the bytes their charge
/// ([`DriverOptions::charge`]) reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedSetLimits {
    pub max_entries: NonZeroUsize,
    pub max_retained_bytes: NonZeroU64,
}

/// One owner of retained bytes that several prepared entries can hold at once, such as a kernel every
/// entry of one shape family shares. Identity is the key: two entries naming the same key hold the same
/// owner, whatever copies of its data their programs carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedOwner {
    pub key: String,
    pub bytes: u64,
}

/// What a program retains on the host while its entry stays prepared. The first entry to hold a shared
/// owner is charged its bytes, later holders are charged nothing for it, and the last release returns
/// them; `private` bytes belong to the one entry. Weights are shared by every entry of an executable and
/// are never charged here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    pub shared: Vec<SharedOwner>,
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
pub type RetentionCharge = fn(&StagedProgram<ValidationOutputs>) -> Retention;

/// The host bytes a program's graph, plans and kernel bodies retain. Its value table, equations and
/// per-equation plans are private to the entry. Each distinct kernel body, keyed by its plan key, is a
/// shared owner: entries of one executable load a kernel once however many plans name it. The bytes are
/// the body's own size; the compiled artifact is the executor's and is bounded by
/// `CompileLimits::max_artifact_bytes` where it is loaded, and device memory is not counted here.
pub fn program_retention(program: &StagedProgram<ValidationOutputs>) -> Retention {
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
        let values = graph.values.len() * size_of::<poot_graph_ir::ValueMeta>();
        let eqns = graph.eqns.len()
            * (size_of::<poot_graph_ir::Eqn>() + size_of::<poot_graph_plan::Plan>());
        retention.private += (values + eqns) as u64;
        for (_, plan) in p.planned() {
            match plan {
                poot_graph_plan::Plan::Compute { body, key, .. }
                | poot_graph_plan::Plan::ComputeMeta { body, key, .. } => {
                    own(key, body, &mut retention)
                }
                poot_graph_plan::Plan::ComputeChunks(chunks) => {
                    for chunk in chunks {
                        own(&chunk.key, &chunk.body, &mut retention);
                    }
                }
                poot_graph_plan::Plan::Alias(_)
                | poot_graph_plan::Plan::View { .. }
                | poot_graph_plan::Plan::Collective { .. } => {}
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

/// Every choice the caller makes about a driver; there is no `Default`.
#[derive(Clone, Copy, Debug)]
pub struct DriverOptions {
    pub prefill: CompileOptions,
    pub decode: CompileOptions,
    /// The KV positions the driver's state holds. State is keyed by (name, aval) inside an executable,
    /// so one driver has one capacity; a request of more positions is refused. Capacity only sizes
    /// buffers: each step carries its live positions in `Slot::Pos` (the kernel-side live-length bound
    /// is Card 730's).
    pub capacity: NonZeroUsize,
    /// New tokens per full prefill step: a multiple of the model's prefill granule.
    pub prefill_chunk: NonZeroUsize,
    /// The most new tokens a traced step may carry (the trace limit the chunk must fit).
    pub max_trace_tokens: NonZeroUsize,
    pub prepared: PreparedSetLimits,
    pub charge: RetentionCharge,
}

/// One sequence to generate. `sampler` carries its own knobs, penalty history, constraint and seed.
pub struct GenerateRequest {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub sampler: Sampler,
    pub stops: Vec<String>,
    /// Keep generating through the model's end-of-sequence token (it is then an ordinary token of the
    /// output): for fixed-length measurements. A request that ends on EOS leaves this off.
    pub ignore_eos: bool,
}

#[derive(Debug)]
pub struct Generation {
    /// The generated tokens, not including the end-of-sequence token that ended the run.
    pub tokens: Vec<u32>,
    pub finish: FinishReason,
    /// One record per token, when the request's sampler recorded logprobs.
    pub logprobs: Vec<TokenLogprob>,
}

/// Receives each generated token with the text it adds. Any `FnMut(u32, &str) -> GenerationControl`
/// is one.
pub trait TokenSink {
    fn token(&mut self, id: u32, text: &str) -> GenerationControl;
}

impl<F: FnMut(u32, &str) -> GenerationControl> TokenSink for F {
    fn token(&mut self, id: u32, text: &str) -> GenerationControl {
        self(id, text)
    }
}

/// What the driver has done, for the prepared-set and compile-once assertions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverStats {
    /// Programs compiled (cold preparations; a refused one counts, a warm replay never does).
    pub compiles: u64,
    /// Entries added to the executor.
    pub entries_added: u64,
    pub steps: u64,
    /// Picks that took the host sampling path.
    pub host_picks: u64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EntryKey {
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
    /// What the entry holds against the retention limit.
    held: Held,
    slots: Vec<SlotSpec>,
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
    /// `Slot::SlotMap` `"read"`: each row's logical position to pool slot.
    MapRead,
    /// `Slot::SlotMap` `"write"`: pool slot to the flat token index writing it, or -1.
    MapWrite,
    LoraIdx,
    Seed,
    Params,
    TopK,
}

impl Prepared {
    /// The slots of `graph`, each classified. A slot no role names is a graph this driver cannot bind.
    fn slots_of(graph: &Graph<ValidationOutputs>) -> Result<Vec<SlotSpec>, DriverError> {
        let named = |slot, tag| SlotKey::new(slot, Some(tag));
        let roles = [
            (token_slot(), SlotRole::Token),
            (pos_slot(), SlotRole::Pos),
            (named(Slot::SlotMap, "read"), SlotRole::MapRead),
            (named(Slot::SlotMap, "write"), SlotRole::MapWrite),
            (SlotKey::new(Slot::LoraIdx, None), SlotRole::LoraIdx),
            (sampler_slot("seed"), SlotRole::Seed),
            (sampler_slot("params"), SlotRole::Params),
            (sampler_slot("top_k"), SlotRole::TopK),
        ];
        let mut slots = Vec::new();
        for &(value, _) in &graph.slots {
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
    write: Vec<i32>,
    lora: Vec<f32>,
    seed: Vec<i32>,
    params: Vec<f32>,
    top_k: Vec<i32>,
}

pub struct Driver {
    handle: Arc<ModelHandle>,
    executor: Box<dyn Executor>,
    exe: ExecutableId,
    target: Target,
    options: DriverOptions,
    chunks: ChunkPolicy,
    /// Which of the model's weights the checkpoint stores packed: put on every traced graph.
    formats: WeightFormats,
    entries: HashMap<EntryKey, Prepared>,
    /// The shared owners the prepared entries reference, by key.
    owners: HashMap<String, Owner>,
    /// The bytes held: every shared owner once, plus each entry's own.
    retained_bytes: u64,
    stats: DriverStats,
    /// Whether any prepared entry carries recurrent state; read from the traced graphs' state roles.
    residency: StateResidency,
    lora: lora::LoraRegistry,
    /// The paged pool, the open sequences and the published caps; absent until `prepare`.
    serving: Option<step::Serving>,
    /// The one layout `prepare` has admitted, and the caps it published.
    layout: Option<Layout>,
    caps: Option<DriverCaps>,
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

impl Driver {
    /// A driver over `executor`: one executable holding the model's weights, shared by every entry.
    /// Nothing uploads until an entry is added.
    pub fn new(
        handle: Arc<ModelHandle>,
        mut executor: Box<dyn Executor>,
        options: DriverOptions,
    ) -> Result<Self, DriverError> {
        let targets = executor.target_set();
        let devices = targets.devices();
        let &[(_, target)] = devices else {
            return Err(InvalidOptions::Devices {
                devices: devices.len(),
            }
            .into());
        };
        if options.capacity.get() > handle.config().max_positions {
            return Err(InvalidOptions::CapacityAboveMax {
                capacity: options.capacity.get(),
                max_positions: handle.config().max_positions,
            }
            .into());
        }
        let chunks = ChunkPolicy::new(
            options.prefill_chunk,
            handle.config().prefill_granule,
            options.max_trace_tokens,
        )?;
        let formats = WeightFormats::from_weight_map(handle.model().weights());
        let weights = Arc::new(handle.model().weights().clone());
        let exe =
            executor.load_weights(handle.shared_store().clone(), WeightSource::Map(weights))?;
        Ok(Self {
            handle,
            executor,
            exe,
            target,
            options,
            chunks,
            formats,
            entries: HashMap::new(),
            owners: HashMap::new(),
            retained_bytes: 0,
            stats: DriverStats::default(),
            residency: StateResidency::KvOnly,
            lora: lora::LoraRegistry::default(),
            serving: None,
            layout: None,
            caps: None,
        })
    }

    pub fn stats(&self) -> DriverStats {
        self.stats
    }

    /// The executor's own counters: recordings, replays and live memory by role.
    pub fn executor_stats(&self) -> poot_executor::ExecutorStats {
        self.executor.stats()
    }

    /// Prepared entries held.
    pub fn prepared_entries(&self) -> usize {
        self.entries.len()
    }

    /// Bytes the prepared entries are charged.
    pub fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    fn options_for(&self, phase: Phase) -> CompileOptions {
        match phase {
            Phase::Prefill => self.options.prefill,
            Phase::Decode => self.options.decode,
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
        if self.entries.len() >= limits.max_entries.get() {
            return Err(PreparedSetRefusal::Entries {
                held: self.entries.len(),
                limit: limits.max_entries.get(),
            }
            .into());
        }
        let trace_limit = self.options.max_trace_tokens.get();
        if key.shape.tokens.get() > trace_limit {
            return Err(PreparedSetRefusal::TraceTokens {
                tokens: key.shape.tokens.get(),
                limit: trace_limit,
            }
            .into());
        }
        if let Some(generation) = key.adapters
            && !self.lora.contains(generation)
        {
            return Err(lora::LoraError::UnknownGeneration(generation).into());
        }
        let graph = self
            .handle
            .model()
            .trace(key.phase, key.shape)
            .map_err(Unsupported::Trace)?;
        let residency = self.check_state_rows(&graph, key.shape)?;
        let graph = bind_packed_weights(&graph, &self.formats).map_err(Unsupported::PackedBind)?;
        let (graph, _) = suffix::append_head(graph, key.head);
        let slots = Prepared::slots_of(&graph)?;
        let program = compile_staged(
            &graph,
            &TargetSet::single(DeviceId(0), self.target),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &self.options_for(key.phase),
        )
        .map_err(|e| match e {
            StagedCompileError::Compile(poot_graph_plan::CompileError::Plan(plan)) => match *plan {
                poot_graph_plan::PlanError::Refused(refusal) => Unsupported::Plan(refusal),
                other => Unsupported::Compile(StagedCompileError::Compile(other.into())),
            },
            other => Unsupported::Compile(other),
        })?;
        self.stats.compiles += 1;
        let reservation = self.reserve((self.options.charge)(&program));
        let would_retain = self.retained_bytes.saturating_add(reservation.charge);
        if would_retain > limits.max_retained_bytes.get() {
            return Err(PreparedSetRefusal::Bytes {
                charge: reservation.charge,
                held: self.retained_bytes,
                would_retain,
                limit: limits.max_retained_bytes.get(),
            }
            .into());
        }
        let id = self.executor.add_entry(self.exe, &program)?;
        self.stats.entries_added += 1;
        self.hold(&reservation);
        let entry = Prepared {
            id,
            held: reservation.held,
            slots,
        };
        if residency == StateResidency::Recurrent {
            self.residency = StateResidency::Recurrent;
        }
        self.entries.insert(key, entry.clone());
        Ok(entry)
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

    /// Dserve.md every carried state value's leading axis is the row axis, so per-row
    /// reset and row reorder address a row by its leading index. A paged KV pool is the one value
    /// without a row axis: the `Positional` cache whose leading axis is the pool's slots. Returns the
    /// graph's state residency, read from its state roles, never from a family.
    fn check_state_rows(
        &self,
        graph: &Graph<ValidationOutputs>,
        shape: StepShape,
    ) -> Result<StateResidency, Unsupported> {
        let rows = shape.rows.get();
        let mut residency = StateResidency::KvOnly;
        for pair in graph.state_pairs() {
            let leading = graph.aval(pair.input).shape.first().copied().unwrap_or(0);
            let pool = matches!(
                (shape.kv, pair.role),
                (KvLayout::Paged { pool_slots }, StateRole::Positional { axis: 0 })
                    if pool_slots.get() == leading
            );
            if leading != rows && !pool {
                return Err(Unsupported::Trace(TraceError::ShapeUnsupported {
                    family: self.handle.config().family,
                    shape,
                    reason: ShapeReason::StateRowAxis { leading, rows },
                }));
            }
            if pair.role == StateRole::Recurrent {
                residency = StateResidency::Recurrent;
            }
        }
        Ok(residency)
    }

    /// Run `entry` over `data`: bind each slot the entry declares, in its declared shape.
    fn execute(
        &mut self,
        entry: &Prepared,
        data: &StepData,
    ) -> Result<poot_executor::StepOutputs<'_>, DriverError> {
        let tensors: Vec<(&SlotKey, HostTensor)> = entry
            .slots
            .iter()
            .map(|spec| {
                let shape = spec.shape.clone();
                let tensor = match spec.role {
                    SlotRole::Token => HostTensor::i32(shape, data.tokens.clone()),
                    SlotRole::Pos => HostTensor::i32(shape, data.pos.clone()),
                    SlotRole::MapRead => HostTensor::i32(shape, data.read.clone()),
                    SlotRole::MapWrite => HostTensor::i32(shape, data.write.clone()),
                    SlotRole::LoraIdx => HostTensor::f32(shape, data.lora.clone()),
                    SlotRole::Seed => HostTensor::i32(shape, data.seed.clone()),
                    SlotRole::Params => HostTensor::f32(shape, data.params.clone()),
                    SlotRole::TopK => HostTensor::i32(shape, data.top_k.clone()),
                };
                (&spec.key, tensor)
            })
            .collect();
        let mut inputs = StepInputs::new();
        for (key, tensor) in &tensors {
            inputs.push((*key).clone(), tensor.shape(), tensor.view());
        }
        self.stats.steps += 1;
        Ok(self
            .executor
            .step(self.exe, entry.id, &inputs, &mut NoSync)?)
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
            });
        }
        if prompt.is_empty() {
            return Err(InvalidRequest::EmptyPrompt.into());
        }
        let capacity = self.options.capacity.get();
        if prompt.len() + max_new > capacity {
            return Err(InvalidRequest::Capacity {
                prompt: prompt.len(),
                max_new,
                max: capacity,
            }
            .into());
        }
        let head = match suffix::rule_of(sampler) {
            Some(rule) => Head::Sample(rule),
            None => Head::Logits,
        };
        self.prepare(&ServingShapes {
            layout: self.layout.unwrap_or(Layout::Contiguous),
            rows: &[NonZeroUsize::MIN],
            heads: &[head],
            warm: Warm::Prompts(&[prompt.len()]),
            windows: &[],
            adapters: &[],
        })?;
        let seq = match self.open(SeqRequest {
            prompt,
            max_new,
            adapter: AdapterRef::Base,
        })? {
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
                Err(error)
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
        let mut logprobs = Vec::new();
        let mut next = loop {
            let result = self.step_one(seq, RowWork::Prefill { upto }, sampler)?;
            logprobs.extend(result.logprobs);
            if let Some(&token) = result.tokens.first() {
                break token;
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
        let finish = loop {
            if stop_rule.is_eos(next) {
                break FinishReason::Eos;
            }
            self.commit(seq, 1, sampler)?;
            context.push(next);
            let piece = text
                .stream_piece(&context, gen_start)
                .map_err(|e| DriverError::Text(Box::new(e)))?;
            if sink.token(next, &piece).is_break() {
                break FinishReason::Cancelled;
            }
            if let Some(reason) = stop_rule.after_token(&piece, context.len() - gen_start) {
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
            next = result.tokens[0];
        };
        Ok(Generation {
            tokens: context.split_off(gen_start),
            finish,
            logprobs,
        })
    }
}
