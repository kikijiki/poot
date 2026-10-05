//! The one generic engine (`Engine<D>` implements the object-safe [`crate::Executor`]).
//!
//! An executable uploads nothing at `load_weights` (the planned lane of a weight is only known once
//! an entry's program is loaded); each `add_entry` binds its consts through the engine's residency
//! map keyed by (store generation, byte range, planned storage), so executables over stores that
//! share a payload share its upload, allocates or shares state by (name, aval, storage),
//! allocates one buffer per computed value, loads every plan, and resolves the state commit.
//! A step writes the slots, checks the validation packet, and replays the entry's recording,
//! recording it on the entry's first step.
//!
//! Every `begin`/`dispatch`/`copy`/`finish` sequence runs as one scoped transaction (core review
//! CR30): a failure anywhere in it calls `Device::abort` exactly once. A failure that may have
//! mutated state marks the executable [`crate::ExecError::NeedsReset`]; a failure to clean up after
//! abort poisons the whole engine (`Device::abort` itself can never run again once poisoned - one
//! engine drives one device, so there is nothing left to recover).

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::{Arc, Weak};
use std::time::Instant;

use poot_graph_ir::{Eqn, Slot, SlotKey, Storage, TensorType, ValidationOutputs, ValueId};
use poot_graph_plan::{
    DeviceId, Plan, Program, StagedProgram, Submission, TargetSet, dispatch_work, value_ids,
};
use poot_kernel_ir::Body;
use poot_quant::SourceRole;
use poot_quant::weights::{StoreGeneration, WeightStore};
use poot_target::{Backend, BufferStorage};
use poot_tensor::DType;

use crate::binder::{self, BoundSlot, EmbedRows, StoredAs, WeightBinder};
use crate::device::{Arg, BufferRole, Device, DeviceTime, Dispatch};
use crate::error::{BindError, DeviceError, ExecError, LoadError};
use crate::timing::TimingOptions;
use crate::{
    EntryId, ExecutableId, Executor, ExecutorStats, HostSync, OutputSource, StateScope, StepInputs,
    StepOutputs, WeightSource,
};

/// The engine's only device id: one engine drives one device (Card 581a's multi-stage composition
/// is out of this card's scope; `add_entry` refuses a `StagedProgram` naming more than one stage).
const DEVICE: DeviceId = DeviceId(0);

pub struct Engine<D: Device> {
    // Field order is drop order: executables (and their recordings and buffers) drop before the
    // device, and the device drops before nothing (it is last).
    executables: Vec<Option<Loaded<D>>>,
    /// Executables `unload` removed while [`Loaded::needs_reset`] was set (core review CR30, cards
    /// 549 SC-010/SC-011, moved from 546a): a failed transaction's completion is not proven, so its
    /// buffers/modules/events must outlive the handle rather than drop with it. Drained (and so
    /// finally freed) the next time any `step` on this engine proves the device has quiesced - one
    /// `Device` owns one native queue/stream, so any later successful `synchronize()` here covers
    /// every op submitted before it, including this one's.
    pending_release: Vec<Loaded<D>>,
    kernels: Vec<D::Kernel>,
    /// Keyed by the compiled `Body`'s own content fingerprint (`KernelCache::key`, Card 656), never
    /// the planner's display `key` string: two equations can share a display key while compiling to
    /// different bodies (SC-024), and a string-keyed fast path here would alias the second one onto
    /// the first's kernel before ever reaching `kernel_cache`'s own content-keyed lookup below.
    kernel_index: HashMap<u64, usize>,
    kernel_cache: poot_codegen::KernelCache,
    /// Every live weight upload, by what it uploads. Weak: an upload lives exactly as long as some
    /// executable (or a pending release of one) holds it, so a replaced adapter's weights are freed
    /// once nothing references them while a base shared with another executable stays.
    residency: HashMap<Residency, Weak<ResidentWeight<D>>>,
    device: D,
    /// Set once `Device::abort` itself fails: no further work can run on this device (core review
    /// CR30). Checked at the top of every `Executor` method that would touch the device.
    poisoned: bool,
    /// Card 552: bounded typed timing, built from this engine's own host-side `Instant` spans and
    /// the device's own [`DeviceTime`]. `timing_options` governs only how much detail this
    /// snapshot retains; the host-wall partition below is always recorded (it costs only
    /// `Instant::now()` calls, never an extra device synchronization).
    timing: poot_profile::TimingSnapshot,
    timing_options: TimingOptions,
    /// Card 552: of every `TimingOptions::Detailed`'s `sample_every` steps, this counts
    /// up so `step()` records full device-time detail for 1 of them and `DeviceCoverage::Unknown`
    /// for the rest (never a device-side cost reduction yet - see `record_step_timing`'s doc).
    detail_sample_counter: u64,
}

/// Where a value's buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    /// An ad hoc buffer this entry owns outright: a metadata buffer, or the `preserve_output`
    /// snapshot (Card 547b: never arena-shared - each is sized and used exactly once per entry).
    Local(usize),
    /// An arena slot (Card 547b): the buffer at this index of `Entry::arena`, shared
    /// across every value [`Program::buffer_plan`] assigned to this slot. The slot's capacity may
    /// exceed this particular value's own logical size; a dispatch using it carries its own element
    /// count separately (`LoadedStep`'s `(Loc, u32)` pairs), never reads the buffer's capacity.
    Arena(usize),
    Weight(usize),
    State(usize),
}

/// What one weight upload is: its stored byte runs, each named by the generation of the store
/// payload it reads (never the entry's key or the const name that reached it), and its planned
/// storage. Two ids viewing one payload share one upload (S62-9); two payloads that merely share a
/// name and shape (a replaced adapter) do not.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Residency {
    runs: Vec<(StoreGeneration, Option<SourceRole>, Range<usize>)>,
    storage: BufferStorage,
}

/// One uploaded weight, shared by every executable that binds it. A weight is never arena-shared,
/// so `elems` is always the exact native device-element count a dispatch reading it should publish
/// (Card 547b) - for a packed source the uploaded `u32`-word count, never the logical element count
/// of the declared `TensorType`.
struct ResidentWeight<D: Device> {
    buffer: D::Buffer,
    elems: u32,
}

struct Loaded<D: Device> {
    binder: WeightBinder,
    /// This executable's reference to each upload it binds, indexed by [`Loc::Weight`]: the
    /// uploads outlive every recording and pending release of this executable.
    weights: Vec<Arc<ResidentWeight<D>>>,
    state: Vec<StateBuffer<D>>,
    entries: Vec<Option<Entry<D>>>,
    /// Set after a state-mutating failure; the next `step` refuses until `reset_state(All)` clears
    /// it (core review CR30).
    needs_reset: bool,
}

struct StateBuffer<D: Device> {
    name: String,
    aval: TensorType,
    storage: BufferStorage,
    bytes: usize,
    /// This state buffer's native device-element count (Card 547b): never arena-shared, so always
    /// the exact length a dispatch reading it should publish.
    elems: u32,
    buffer: D::Buffer,
}

struct Entry<D: Device> {
    // Field order is drop order: the recording drops before the buffers it references.
    recording: Option<D::Recording>,
    recordings: u64,
    replays: u64,
    locals: Vec<D::Buffer>,
    /// One buffer per [`poot_graph_plan::BufferPlan`] arena slot (Card 547b), allocated once up
    /// front from `Program::buffer_plan` and index-aligned with [`Loc::Arena`].
    arena: Vec<D::Buffer>,
    steps: Vec<LoadedStep>,
    /// Explicit two-phase commit copies (src, dst), run after every step's dispatches.
    commits: Vec<(Loc, Loc)>,
    /// The primary output snapshot taken before `commits`, when `preserve_output` is set (the output
    /// aliases a destination the commit copies overwrite).
    preserve_output: Option<Loc>,
    slots: Vec<BoundSlot>,
    /// Hosted embed gathers (`legalize`'s `Slot::TokenEmbed`), filled each step from the step's
    /// `Slot::Token` input and the executable's embed rows, never supplied by the caller.
    token_embeds: Vec<TokenEmbedSlot>,
    output: Loc,
    output_bytes: usize,
    /// The primary output as a host tensor: its dtype lane (`None` when the planned lane is not a
    /// dense natural-width one, e.g. a packed or mirrored output) and declared shape.
    output_host: Option<(DType, Vec<usize>)>,
    validation: ValidationLoad,
}

/// One hosted embed gather: where its rows come from, its planned lane, and its buffer (an index
/// into the entry's `locals`).
struct TokenEmbedSlot {
    rows: EmbedRows,
    numel: usize,
    storage: BufferStorage,
    buffer: usize,
}

/// The entry's validation packet sources, resolved to device locations (empty for a zero-lane
/// packet, which is checked the same way as any other: reading zero bytes).
struct ValidationLoad {
    layout: poot_graph_ir::ValidationPacketLayout,
    sources: Vec<(Loc, usize)>, // (buffer location, lane_count)
}

struct LoadedStep {
    kernel: usize,
    /// Each input's location and its own logical device-element count (Card 547b): the
    /// plan's own length, never the bound buffer's capacity (which an arena slot's largest occupant
    /// may have sized larger than this particular operand).
    inputs: Vec<(Loc, u32)>,
    output: (Loc, u32),
    threads: [u32; 3],
    workgroup: [u32; 3],
    work: u64,
    /// The originating equation's semantic report key (Card 552, R-552-2): the aggregation key for
    /// typed device-time buckets, never the planner's kernel `key` (two equations with the same
    /// `OpKind` planned to different kernel keys still land in one bucket; two different `OpKind`s
    /// that happen to plan to the same kernel key land in two).
    kind_name: &'static str,
    /// The plan's display label, reported beside `kind_name` in the per-dispatch timing so
    /// device time can be read per kernel shape; never an aggregation key.
    label: String,
}

/// A buffer's native device-element count as the `u32` every [`Arg`]/kernarg length field needs
/// (Card 547b). No buffer this engine allocates is expected to hold more than `u32::MAX` elements;
/// a typed refusal here, rather than a silent truncation, is what "sound architecture" means for the
/// case where one someday does.
fn checked_u32(elems: usize) -> Result<u32, ExecError> {
    u32::try_from(elems)
        .map_err(|_| LoadError::Unimplemented("value exceeds u32 element count").into())
}

fn device_error<E: std::error::Error + Send + Sync + 'static>(
    backend: Backend,
) -> impl Fn(E) -> ExecError {
    move |source| {
        ExecError::from(DeviceError {
            backend: match backend {
                Backend::SpirvVulkan => "wgpu",
                Backend::AmdGcn(_) => "rocm",
                Backend::Nvptx => "ptx",
            },
            source: Box::new(source),
        })
    }
}

fn codegen_target(backend: Backend) -> poot_codegen::Target {
    match backend {
        Backend::SpirvVulkan => poot_codegen::Target::SpirvVulkan,
        Backend::Nvptx => poot_codegen::Target::Nvptx,
        Backend::AmdGcn(arch) => poot_codegen::Target::AmdGcn(arch),
    }
}

fn resolve<'a, D: Device>(
    loc: Loc,
    locals: &'a [D::Buffer],
    arena: &'a [D::Buffer],
    weights: &'a [Arc<ResidentWeight<D>>],
    state: &'a [StateBuffer<D>],
) -> &'a D::Buffer {
    match loc {
        Loc::Local(i) => &locals[i],
        Loc::Arena(i) => &arena[i],
        Loc::Weight(i) => &weights[i].buffer,
        Loc::State(i) => &state[i].buffer,
    }
}

impl<D: Device> Engine<D> {
    /// An engine with the default [`TimingOptions::CountersOnly`] (no behavior change from before
    /// Card 552: cumulative counters are always tracked, no per-dispatch detail is retained).
    pub fn new(device: D) -> Self {
        Self::with_timing(device, TimingOptions::default())
    }

    /// An engine whose bounded timing detail follows `timing`. This governs only how much this
    /// engine itself retains/reports (Card 552); whether `device` can actually produce per-dispatch
    /// device durations is a property of how that concrete `Device` was constructed.
    pub fn with_timing(device: D, timing: TimingOptions) -> Self {
        let backend = device.target().backend;
        let snapshot = match &timing {
            TimingOptions::CountersOnly => poot_profile::TimingSnapshot::default(),
            TimingOptions::Detailed(d) => poot_profile::TimingSnapshot::new(
                d.max_retained_steps,
                d.max_retained_dispatch_records,
            ),
        };
        Self {
            executables: Vec::new(),
            pending_release: Vec::new(),
            kernels: Vec::new(),
            kernel_index: HashMap::new(),
            kernel_cache: poot_codegen::KernelCache::open(codegen_target(backend)),
            residency: HashMap::new(),
            device,
            poisoned: false,
            timing: snapshot,
            timing_options: timing,
            detail_sample_counter: 0,
        }
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    /// This engine's configured timing detail (Card 552).
    pub fn timing_options(&self) -> &TimingOptions {
        &self.timing_options
    }

    fn backend(&self) -> Backend {
        self.device.target().backend
    }

    fn check_not_poisoned(&self) -> Result<(), ExecError> {
        if self.poisoned {
            return Err(ExecError::Poisoned);
        }
        Ok(())
    }

    fn loaded(&mut self, exe: ExecutableId) -> Result<&mut Loaded<D>, ExecError> {
        self.executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or_else(|| LoadError::UnknownHandle.into())
    }

    /// Kernel index for `body`, compiling it for this device on first use (Card 656: the full
    /// verified `Body` is the cache's identity, not a display key - `kernel_index` itself keys on
    /// `KernelCache::key(body)`'s content fingerprint, so two equations sharing a planner `key`
    /// string but compiling to different bodies never collide here, before `kernel_cache`'s own
    /// content-keyed lookup ever runs). `key` is used only as the device's debug label.
    fn kernel(
        &mut self,
        key: &str,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<usize, ExecError> {
        let fingerprint = self.kernel_cache.key(body).fingerprint();
        if let Some(&index) = self.kernel_index.get(&fingerprint) {
            return Ok(index);
        }
        let backend = self.backend();
        let artifact = self
            .kernel_cache
            .load_or_compile(body, max_artifact_bytes)
            .map_err(|source| LoadError::Codegen {
                key: key.to_string(),
                source,
            })?;
        let compiled = poot_codegen::kernel_handle(body, codegen_target(backend), artifact.bytes);
        let kernel = self
            .device
            .load_kernel(key, compiled)
            .map_err(device_error(backend))?;
        self.kernels.push(kernel);
        let index = self.kernels.len() - 1;
        self.kernel_index.insert(fingerprint, index);
        Ok(index)
    }

    /// `numel` is a logical tensor element count (never already device-encoded - see
    /// `allocate_arena_slot` for a caller that already has one). Returns `(buffer, bytes, elems)`:
    /// `elems` is this buffer's native device-element count (Card 547b) - the same quantity
    /// `Device::allocate` itself was called with, handed back so a caller that also needs to publish
    /// it (as a dispatch argument's own length, never a reused arena slot's capacity) does not
    /// re-derive it a second time.
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        numel: usize,
    ) -> Result<(D::Buffer, usize, usize), ExecError> {
        let elems = storage
            .device_elems(numel)
            .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
        let buffer = self.allocate_arena_slot(role, storage, elems)?;
        Ok((buffer, elems * storage.element().byte_width(), elems))
    }

    /// Allocates one buffer from `elems`, an already native device-element count - `BufferPlan::slots()`
    /// hands these out pre-lane-encoded (`SlotInfo` doc, `poot_graph_plan::buffer_plan`), the same unit
    /// `Device::allocate` itself takes. Review F3: routing an arena slot's `slot_elems` back through
    /// [`Self::allocate`] would apply `BufferStorage::device_elems` a second time, under-allocating a
    /// packed slot by half; this is the one call both `Self::allocate` (from a logical `numel`) and
    /// `Self::load_entry`'s arena loop (from a slot's native count) route their actual
    /// `Device::allocate` call through.
    fn allocate_arena_slot(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<D::Buffer, ExecError> {
        let backend = self.backend();
        self.device
            .allocate(role, storage, elems.max(1))
            .map_err(device_error(backend))
    }

    fn load_entry(
        &mut self,
        exe: ExecutableId,
        program: &Program<ValidationOutputs>,
    ) -> Result<Entry<D>, ExecError> {
        let device_target = self.device.target();
        if *program.target() != device_target {
            return Err(LoadError::TargetMismatch {
                program: Box::new(*program.target()),
                device: Box::new(device_target),
            }
            .into());
        }
        let g = program.graph();
        let backend = self.backend();
        let storage_of =
            |id: ValueId| -> BufferStorage { program.storage().storage(id).buffer_storage() };
        // Card 547b: every locally-computed value's dispatch length is derived from
        // its own declared shape - never the bound buffer's capacity (an arena slot's buffer is
        // sized to its largest occupant, which may exceed this value's own need). A graph input
        // (state, const, slot) instead carries the exact elems its own allocation used (below,
        // alongside each input's `Loc`): a packed-source const's uploaded word count is not a
        // function of its declared `TensorType` alone, so this formula must never be applied to one.
        let elems_of = |id: ValueId| -> Result<u32, ExecError> {
            let elems = storage_of(id)
                .device_elems(g.aval(id).numel())
                .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
            checked_u32(elems)
        };
        let mut values: Vec<Option<(Loc, u32)>> = vec![None; g.values.len()];
        let mut locals: Vec<D::Buffer> = Vec::new();
        let mut slots: Vec<BoundSlot> = Vec::new();
        let mut token_embeds: Vec<TokenEmbedSlot> = Vec::new();

        // Card 547b: one buffer per arena slot, allocated once up front from the program's buffer
        // plan (`Program::buffer_plan`) - never one buffer per computed value.
        let buffer_plan = program.buffer_plan();
        let mut arena: Vec<D::Buffer> = Vec::with_capacity(buffer_plan.slot_count());
        for (_, slot_storage, slot_elems) in buffer_plan.slots() {
            // Review F3: `slot_elems` is already the slot's native device-element count - never
            // route it through `Self::allocate`, which would apply `BufferStorage::device_elems` to
            // it a second time and under-allocate a packed slot by half.
            let buffer =
                self.allocate_arena_slot(BufferRole::Activation, slot_storage, slot_elems)?;
            arena.push(buffer);
        }

        macro_rules! push_local {
            ($role:expr, $storage:expr, $numel:expr) => {{
                let (buffer, _bytes, elems) = self.allocate($role, $storage, $numel)?;
                locals.push(buffer);
                (Loc::Local(locals.len() - 1), checked_u32(elems)?)
            }};
        }

        let chunks = binder::chunks(g.inputs.iter().filter_map(|&id| {
            let meta = g.meta(id);
            (meta.storage == Storage::Const)
                .then(|| Some((id, meta.name.as_deref()?, meta.aval.shape.as_slice())))
                .flatten()
        }))?;
        for &id in &g.inputs {
            let meta = g.meta(id);
            let storage = storage_of(id);
            let (loc, elems) = match meta.storage {
                Storage::State => {
                    let name = meta
                        .name
                        .clone()
                        .ok_or(LoadError::UnnamedConst { value: id })?;
                    let (idx, elems) = self.state_buffer(exe, name, meta.aval.clone(), storage)?;
                    (Loc::State(idx), elems)
                }
                Storage::Const => {
                    let name = meta
                        .name
                        .clone()
                        .ok_or(LoadError::UnnamedConst { value: id })?;
                    let (idx, elems) =
                        self.weight_buffer(exe, id, &name, &meta.aval, chunks.get(&id), storage)?;
                    (Loc::Weight(idx), elems)
                }
                Storage::Computed(computed) => {
                    let bytes: Vec<u8> = computed
                        .values_f32()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let encoded =
                        crate::lanes::encode_host(poot_tensor::DType::F32, &bytes, storage)
                            .ok_or_else(|| LoadError::WeightFormat {
                                name: format!("computed v{id}"),
                                stored: "F32".into(),
                                planned: storage,
                            })?;
                    let (loc, elems) = push_local!(BufferRole::Input, storage, meta.aval.numel());
                    self.device
                        .write(resolve::<D>(loc, &locals, &arena, &[], &[]), &encoded)
                        .map_err(device_error(backend))?;
                    (loc, elems)
                }
                Storage::Slot(Slot::TokenEmbed) => {
                    let name = meta
                        .name
                        .as_deref()
                        .ok_or(LoadError::UnnamedConst { value: id })?;
                    let rows = self
                        .loaded(exe)?
                        .binder
                        .embed_rows(id, name, &meta.aval, storage)?;
                    let (loc, elems) = push_local!(BufferRole::Input, storage, meta.aval.numel());
                    let Loc::Local(buffer) = loc else {
                        unreachable!("push_local! always returns Loc::Local")
                    };
                    token_embeds.push(TokenEmbedSlot {
                        rows,
                        numel: meta.aval.numel(),
                        storage,
                        buffer,
                    });
                    (loc, elems)
                }
                Storage::Slot(_) => {
                    let key = meta
                        .slot_key()
                        .cloned()
                        .ok_or(LoadError::UnkeyedSlot { value: id })?;
                    let (loc, elems) = push_local!(BufferRole::Input, storage, meta.aval.numel());
                    let Loc::Local(buffer) = loc else {
                        unreachable!("push_local! always returns Loc::Local")
                    };
                    slots.push(BoundSlot {
                        key,
                        aval: meta.aval.clone(),
                        storage,
                        buffer,
                    });
                    (loc, elems)
                }
                Storage::Device => {
                    return Err(LoadError::Unimplemented("a Device-storage graph input").into());
                }
            };
            values[id] = Some((loc, elems));
        }

        // Donation: a state update written straight into its state buffer. Every backend donates
        // unconditionally when the state-commit plan says so: there is no
        // `Device::in_place_donation` capability to ask.
        let commit_plan = program.state_commit();

        let max_artifact_bytes = program.limits().max_artifact_bytes;
        let mut steps = Vec::new();
        for (eqn_index, (eqn, plan)) in program.planned().enumerate() {
            let operand =
                |values: &[Option<(Loc, u32)>], v: ValueId| -> Result<(Loc, u32), ExecError> {
                    values[v].ok_or(LoadError::Unimplemented("use before def").into())
                };
            // Card 547b: each operand's own `(Loc, elems)` pair, exactly as it was recorded when
            // that value was bound or computed above - never re-derived here, so a packed-source
            // const's own uploaded word count is carried through unchanged.
            let operands_with_elems =
                |values: &[Option<(Loc, u32)>], eqn: &Eqn| -> Result<Vec<(Loc, u32)>, ExecError> {
                    value_ids(eqn)
                        .into_iter()
                        .map(|v| operand(values, v))
                        .collect()
                };
            let work = dispatch_work(g, eqn);
            match plan {
                Plan::Alias(source) => values[eqn.out] = Some(operand(&values, *source)?),
                Plan::View { src, .. } => values[eqn.out] = Some(operand(&values, *src)?),
                Plan::Compute {
                    body,
                    key,
                    label,
                    grid,
                } => {
                    let kernel = self.kernel(key, body, max_artifact_bytes)?;
                    let inputs = operands_with_elems(&values, eqn)?;
                    let output_loc = out_loc(
                        OutLocArgs {
                            out: eqn.out,
                            commit_plan,
                            buffer_plan,
                        },
                        &values,
                    )?;
                    let output = (output_loc, elems_of(eqn.out)?);
                    steps.push(LoadedStep {
                        kernel,
                        inputs,
                        output,
                        threads: *grid,
                        workgroup: body.workgroup_size,
                        work,
                        kind_name: eqn.op.kind_name(),
                        label: label.clone(),
                    });
                    values[eqn.out] = Some(output);
                }
                Plan::ComputeMeta {
                    body,
                    key,
                    label,
                    meta,
                    grid,
                } => {
                    let kernel = self.kernel(key, body, max_artifact_bytes)?;
                    let mut inputs = operands_with_elems(&values, eqn)?;
                    let meta_loc = self.meta_buffer(meta, &mut locals)?;
                    inputs.push((meta_loc, meta.len() as u32));
                    let output_loc = out_loc(
                        OutLocArgs {
                            out: eqn.out,
                            commit_plan,
                            buffer_plan,
                        },
                        &values,
                    )?;
                    let output = (output_loc, elems_of(eqn.out)?);
                    steps.push(LoadedStep {
                        kernel,
                        inputs,
                        output,
                        threads: *grid,
                        workgroup: body.workgroup_size,
                        work,
                        kind_name: eqn.op.kind_name(),
                        label: label.clone(),
                    });
                    values[eqn.out] = Some(output);
                }
                Plan::ComputeChunks(chunks) => {
                    let output_loc = out_loc(
                        OutLocArgs {
                            out: eqn.out,
                            commit_plan,
                            buffer_plan,
                        },
                        &values,
                    )?;
                    let output = (output_loc, elems_of(eqn.out)?);
                    let per_chunk_work = work / (chunks.len().max(1) as u64);
                    for chunk in chunks {
                        let kernel = self.kernel(&chunk.key, &chunk.body, max_artifact_bytes)?;
                        let mut inputs = operands_with_elems(&values, eqn)?;
                        if !chunk.meta.is_empty() {
                            let meta_loc = self.meta_buffer(&chunk.meta, &mut locals)?;
                            inputs.push((meta_loc, chunk.meta.len() as u32));
                        }
                        let wg = chunk.body.workgroup_size;
                        steps.push(LoadedStep {
                            kernel,
                            inputs,
                            output,
                            threads: [chunk.groups as u32 * wg[0], wg[1], wg[2]],
                            workgroup: wg,
                            work: per_chunk_work,
                            kind_name: eqn.op.kind_name(),
                            label: chunk.label.clone(),
                        });
                    }
                    values[eqn.out] = Some(output);
                }
                Plan::Collective { .. } => {
                    return Err(LoadError::UnloadablePlan {
                        eqn: eqn_index,
                        kind: plan.kind(),
                    }
                    .into());
                }
            }
        }

        // The state commit: every non-identity edge not donated at its producer is an explicit copy,
        // after the step's dispatches. `preserve_output` snapshots the primary output
        // first when it aliases a destination the copies would otherwise overwrite before it is read.
        let mut commits = Vec::new();
        for &(state_input, state_output) in &g.state {
            let dst = values[state_input].expect("state bound").0;
            let src = values[state_output]
                .ok_or(LoadError::Unimplemented("unmaterialized state output"))?
                .0;
            if src == dst {
                continue;
            }
            commits.push((src, dst));
        }
        let output = values[g.output]
            .ok_or(LoadError::Unimplemented("unmaterialized output"))?
            .0;
        let preserve_output = if commit_plan.preserve_output {
            let (buf, _bytes, _elems) = self.allocate(
                BufferRole::Output,
                storage_of(g.output),
                g.aval(g.output).numel(),
            )?;
            locals.push(buf);
            Some(Loc::Local(locals.len() - 1))
        } else {
            None
        };
        let output_bytes = storage_of(g.output)
            .device_elems(g.aval(g.output).numel())
            .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?
            * storage_of(g.output).element().byte_width();
        let output_host =
            host_dtype(storage_of(g.output)).map(|dtype| (dtype, g.aval(g.output).shape.clone()));

        let validation_layout = program.validation().layout.clone();
        let mut validation_sources = Vec::new();
        for source in &program.validation().sources {
            let loc = values[source.value]
                .ok_or(LoadError::Unimplemented("validation source unmaterialized"))?
                .0;
            validation_sources.push((loc, source.lane_count));
        }

        Ok(Entry {
            recording: None,
            recordings: 0,
            replays: 0,
            locals,
            arena,
            steps,
            commits,
            preserve_output,
            slots,
            token_embeds,
            output,
            output_bytes,
            output_host,
            validation: ValidationLoad {
                layout: validation_layout,
                sources: validation_sources,
            },
        })
    }

    fn meta_buffer(&mut self, meta: &[u32], locals: &mut Vec<D::Buffer>) -> Result<Loc, ExecError> {
        let backend = self.backend();
        let storage = BufferStorage::dense(
            poot_target::ElementKind::I32,
            poot_target::LogicalDType::RawBytes,
        );
        // Not `self.allocate`: a meta buffer's "elements" are already native raw-byte words (the
        // plan's own dims/length metadata), never a logical value `device_elems` lane-encodes - the
        // E4M3/raw-byte refusal that function applies to graph *values* does not apply here.
        let elems = meta.len().max(1);
        let buffer = self
            .device
            .allocate(BufferRole::Meta, storage, elems)
            .map_err(device_error(backend))?;
        self.device
            .write(&buffer, bytemuck::cast_slice(meta))
            .map_err(device_error(backend))?;
        locals.push(buffer);
        Ok(Loc::Local(locals.len() - 1))
    }

    /// The executable's state buffer for (name, aval, storage), allocated zeroed on first
    /// declaration. The same name with another aval or storage is a typed refusal
    /// (SC-004). Returns `(index, elems)`: a state buffer is never arena-shared, so
    /// `elems` is always this buffer's exact native device-element count (Card 547b).
    fn state_buffer(
        &mut self,
        exe: ExecutableId,
        name: String,
        aval: TensorType,
        storage: BufferStorage,
    ) -> Result<(usize, u32), ExecError> {
        if let Some((index, existing)) = self
            .loaded(exe)?
            .state
            .iter()
            .enumerate()
            .find(|(_, s)| s.name == name)
        {
            if existing.aval != aval || existing.storage != storage {
                return Err(BindError::StateSchema {
                    name,
                    expected: existing.aval.clone(),
                    got: aval,
                }
                .into());
            }
            return Ok((index, existing.elems));
        }
        let (buffer, bytes, elems) = self.allocate(BufferRole::State, storage, aval.numel())?;
        let elems = checked_u32(elems)?;
        let state = &mut self.loaded(exe)?.state;
        state.push(StateBuffer {
            name,
            aval,
            storage,
            bytes,
            elems,
            buffer,
        });
        Ok((state.len() - 1, elems))
    }

    /// The executable's resident buffer for const `name` (input `value`, declared `aval`) in
    /// `storage`, uploaded on first use by any executable. The executable's
    /// [`WeightBinder`] resolves the const to stored byte runs; the store generations of those runs
    /// and the storage are the residency key, so two consts that read the same bytes (a tied head
    /// over the embedding) and two executables over one shared payload share one upload. A packed
    /// source never dequantizes or re-encodes: its bytes upload as the device's own `u32` words
    /// (SC-016). Returns `(index, elems)`: a weight buffer is never arena-shared, and
    /// a packed source's own uploaded `u32`-word count is not a function of its declared
    /// `TensorType` (Card 547b) - so `elems` is always this buffer's own real native device-element
    /// count, computed once here and recorded in the [`ResidentWeight`] every later dedup hit reads
    /// back unchanged.
    fn weight_buffer(
        &mut self,
        exe: ExecutableId,
        value: ValueId,
        name: &str,
        aval: &TensorType,
        chunk: Option<&binder::Chunk>,
        storage: BufferStorage,
    ) -> Result<(usize, u32), ExecError> {
        let loaded = self.loaded(exe)?;
        let bytes = loaded.binder.const_bytes(value, name, aval, chunk)?;
        let key = Residency {
            runs: bytes
                .spans
                .iter()
                .map(|span| (span.generation, span.role, span.bytes.clone()))
                .collect(),
            storage,
        };
        if let Some(resident) = self.residency.get(&key).and_then(Weak::upgrade) {
            let elems = resident.elems;
            let loaded = self.loaded(exe)?;
            loaded.weights.push(resident);
            return Ok((loaded.weights.len() - 1, elems));
        }
        let stored = self.loaded(exe)?.binder.read(&bytes.spans);
        let encoded = match bytes.stored {
            // A generated kernel's packed-source argument always reads `u32` words (Card 642,
            // dquant.md D9), regardless of this value's declared I8/RawBytes logical dtype:
            // `poot_quant::packed_source_words` is the one definition every executor's packed-source
            // const upload calls (card 546a review: no second copy of this byte layout).
            StoredAs::PackedSource => poot_quant::packed_source_words(&stored)
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            StoredAs::Dense(dtype) => {
                // The plan never retypes a const, so any stored/declared difference is the tracer's and
                // is refused.
                if dtype != aval.dtype {
                    return Err(LoadError::WeightDtype {
                        name: name.to_string(),
                        stored: dtype,
                        declared: aval.dtype,
                    }
                    .into());
                }
                crate::lanes::encode_stored(dtype, &stored, storage).map_err(
                    |error| match error {
                        crate::lanes::EncodeError::NoEncoding => LoadError::WeightFormat {
                            name: name.to_string(),
                            stored: format!("{dtype:?}"),
                            planned: storage,
                        },
                        crate::lanes::EncodeError::NonFiniteF16 { index, bits } => {
                            LoadError::NonFiniteF16Weight {
                                name: name.to_string(),
                                index,
                                bits,
                            }
                        }
                    },
                )?
            }
        };
        let backend = self.backend();
        let (buffer, elems) = if bytes.stored == StoredAs::PackedSource {
            // A packed source's device buffer is always `u32` words (Card 642), never this value's
            // declared I8/RawBytes logical storage (`BufferStorage::device_elems` explicitly
            // refuses that lane): bypass `self.allocate`'s value-lane gate exactly as `meta_buffer`
            // does, sized and typed by `encoded`'s own words rather than `storage`/`aval.numel()`.
            debug_assert_eq!(
                encoded.len() % 4,
                0,
                "packed_source_words pads to whole u32 words"
            );
            let elems = encoded.len() / 4;
            let buffer = self
                .device
                .allocate(BufferRole::Weight, BufferStorage::i32(), elems.max(1))
                .map_err(device_error(backend))?;
            (buffer, checked_u32(elems)?)
        } else {
            let (buffer, _bytes, elems) =
                self.allocate(BufferRole::Weight, storage, aval.numel())?;
            (buffer, checked_u32(elems)?)
        };
        self.device
            .write(&buffer, &encoded)
            .map_err(device_error(backend))?;
        let resident = Arc::new(ResidentWeight { buffer, elems });
        self.residency.insert(key, Arc::downgrade(&resident));
        let loaded = self.loaded(exe)?;
        loaded.weights.push(resident);
        Ok((loaded.weights.len() - 1, elems))
    }

    /// Match each input to exactly one slot (by `SlotKey`), check shape/count/dtype, write it. A
    /// hosted embed gather is filled from the `Slot::Token` input, which the entry accepts for it
    /// even when legalize dropped the token slot itself.
    fn bind(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
    ) -> Result<(), ExecError> {
        let backend = self.backend();
        let Self {
            executables,
            device,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let e = loaded
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        let mut seen = vec![false; e.slots.len()];
        // CR30: validate every StepInput before writing any of them, so a later
        // input's refusal cannot leave an earlier input's write already on the device.
        let mut writes: Vec<(usize, Vec<u8>)> = Vec::with_capacity(e.slots.len());
        let token_key = SlotKey::new(Slot::Token, None);
        let mut tokens = None;
        for (key, value) in inputs.iter() {
            let feeds_embed = !e.token_embeds.is_empty() && *key == token_key;
            if feeds_embed && tokens.replace(value).is_some() {
                return Err(BindError::Duplicate { key: key.clone() }.into());
            }
            let Some(index) = e.slots.iter().position(|slot| &slot.key == key) else {
                if feeds_embed {
                    continue;
                }
                return Err(BindError::Unknown { key: key.clone() }.into());
            };
            if std::mem::replace(&mut seen[index], true) {
                return Err(BindError::Duplicate { key: key.clone() }.into());
            }
            let slot = &e.slots[index];
            let bytes = binder::check_and_encode(slot, value.shape, value.view)?;
            writes.push((slot.buffer, bytes));
        }
        if let Some(missing) = seen.iter().position(|bound| !bound) {
            return Err(BindError::Missing {
                key: e.slots[missing].key.clone(),
            }
            .into());
        }
        for embed in &e.token_embeds {
            let value = tokens.ok_or_else(|| BindError::Missing {
                key: token_key.clone(),
            })?;
            if value.view.dtype() != DType::I32 {
                return Err(BindError::Lane {
                    key: token_key.clone(),
                    expected: DType::I32,
                    got: value.view.dtype(),
                }
                .into());
            }
            let ids: Vec<i32> = value
                .view
                .bytes()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let expected = embed.numel / embed.rows.width();
            if ids.len() != expected {
                return Err(BindError::ElementCount {
                    key: token_key.clone(),
                    expected,
                    got: ids.len(),
                }
                .into());
            }
            let bytes = loaded.binder.embed(&embed.rows, &ids, embed.storage)?;
            writes.push((embed.buffer, bytes));
        }
        for (buffer, bytes) in writes {
            device
                .write(&e.locals[buffer], &bytes)
                .map_err(device_error(backend))?;
        }
        Ok(())
    }

    /// Walk one entry into the device: every dispatch, then the state commit.
    fn walk(
        device: &mut D,
        kernels: &[D::Kernel],
        entry: &Entry<D>,
        weights: &[Arc<ResidentWeight<D>>],
        state: &[StateBuffer<D>],
    ) -> Result<(), D::Error> {
        let at = |loc| resolve::<D>(loc, &entry.locals, &entry.arena, weights, state);
        for step in &entry.steps {
            let inputs: Vec<Arg<'_, D>> = step
                .inputs
                .iter()
                .map(|&(loc, elems)| Arg {
                    buffer: at(loc),
                    elems,
                })
                .collect();
            let (output_loc, output_elems) = step.output;
            device.dispatch(Dispatch {
                kernel: &kernels[step.kernel],
                inputs: &inputs,
                output: Arg {
                    buffer: at(output_loc),
                    elems: output_elems,
                },
                threads: step.threads,
                workgroup: step.workgroup,
                work: step.work,
            })?;
        }
        // The two-phase commit: the primary output is snapshotted now, after every
        // dispatch has computed its final value but before any commit copy overwrites the buffer it
        // aliases; the commits themselves run as a simultaneous assignment immediately after.
        if let Some(preserve) = entry.preserve_output {
            device.copy(at(entry.output), at(preserve))?;
        }
        for &(src, dst) in &entry.commits {
            device.copy(at(src), at(dst))?;
        }
        Ok(())
    }

    /// Run `entry`'s transaction (begin, walk, finish) as one scoped unit: on any failure, abort
    /// exactly once and classify the outcome (core review CR30). `mutate` is true once the walk may
    /// have written donated state or a commit copy (anything past `begin`): a failure there marks
    /// the executable `NeedsReset`, never silently retried.
    fn run_transaction(
        &mut self,
        exe: ExecutableId,
        entry_idx: usize,
        submission: Submission,
        walk_recording: bool,
    ) -> Result<Option<D::Recording>, ExecError> {
        self.check_not_poisoned()?;
        let backend = self.backend();
        let error = device_error(backend);
        if let Err(e) = self.device.begin(submission) {
            self.abort_after_failure(exe, false);
            return Err(error(e));
        }
        let Self {
            executables,
            device,
            kernels,
            ..
        } = self;
        // CR30: an unknown handle after a successful `begin` still leaves an
        // open recording, so it must abort the same way a walk/finish failure does, not bypass
        // `abort_after_failure` through an early `?`.
        let loaded = match executables.get_mut(exe.index()).and_then(Option::as_mut) {
            Some(loaded) => loaded,
            None => {
                self.abort_after_failure(exe, false);
                return Err(LoadError::UnknownHandle.into());
            }
        };
        let Loaded {
            entries,
            weights,
            state,
            ..
        } = loaded;
        let e = match entries.get_mut(entry_idx).and_then(Option::as_mut) {
            Some(e) => e,
            None => {
                self.abort_after_failure(exe, false);
                return Err(LoadError::UnknownHandle.into());
            }
        };
        if let Err(walk_err) = Self::walk(device, kernels, e, weights, state) {
            self.abort_after_failure(exe, walk_recording);
            return Err(error(walk_err));
        }
        match self.device.finish() {
            Ok(recording) => Ok(recording),
            Err(finish_err) => {
                self.abort_after_failure(exe, walk_recording);
                Err(error(finish_err))
            }
        }
    }

    /// Call `Device::abort` exactly once; poison the engine if it fails, and mark the executable
    /// `NeedsReset` when the failed transaction may have mutated state (core review CR30).
    fn abort_after_failure(&mut self, exe: ExecutableId, may_have_mutated_state: bool) {
        if self.device.abort().is_err() {
            self.poisoned = true;
        }
        if may_have_mutated_state && let Some(Some(loaded)) = self.executables.get_mut(exe.index())
        {
            loaded.needs_reset = true;
        }
    }

    /// Check the entry's validation packet (ADR-0101 decision 2's validated session): a
    /// zero-lane packet reads zero bytes and always passes. Called after replay/eager execution and
    /// `synchronize`, strictly before any primary-output readback.
    fn check_validation(&mut self, exe: ExecutableId, entry_idx: usize) -> Result<(), ExecError> {
        let backend = self.backend();
        let Self {
            executables,
            device,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let e = loaded
            .entries
            .get(entry_idx)
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        if e.validation.layout.lane_count == 0 {
            return Ok(());
        }
        let mut bytes = vec![0u8; e.validation.layout.lane_count * 4];
        let mut offset = 0usize;
        for &(loc, lane_count) in &e.validation.sources {
            let buf = resolve::<D>(loc, &e.locals, &e.arena, &loaded.weights, &loaded.state);
            let len = lane_count * 4;
            device
                .read(buf, &mut bytes[offset..offset + len])
                .map_err(device_error(backend))?;
            offset += len;
        }
        let bits: &[u32] = bytemuck::cast_slice(&bytes);
        e.validation
            .layout
            .validate_f32_bits(bits)
            .map_err(ExecError::from)
    }

    /// Card 552: of every `TimingOptions::Detailed`'s `sample_every` steps, records 1
    /// (`true`) and skips the rest (`false`, counted but never recorded with detail - `record_step_
    /// timing` then builds a `DeviceCoverage::Unknown` record for it, same shape a backend that
    /// cannot measure at all would produce, and `Report::window` reports the ratio as `Coverage::
    /// Unsampled`). `CountersOnly` never samples. This does not yet reduce the *device's* own
    /// per-step collection cost (a real backend like `WgpuDevice` still requests timestamps on every
    /// replay when its own `device_timing` is on, regardless of this decision) - only what the
    /// engine retains/reports. Reducing device-side cost needs the device to be told per step,
    /// which no `Device` method exists for yet; this is recorded as a known limitation, not hidden
    /// behind a policy that silently does nothing.
    fn consume_sample_decision(&mut self) -> bool {
        match &self.timing_options {
            TimingOptions::CountersOnly => false,
            TimingOptions::Detailed(d) => {
                let n = self.detail_sample_counter;
                self.detail_sample_counter = n.wrapping_add(1);
                n.is_multiple_of(u64::from(d.sample_every.max(1)))
            }
        }
    }

    /// Card 552: convert this step's measured host partition and the device's own [`DeviceTime`]
    /// into a [`poot_profile::StepTiming`] and fold it into this engine's bounded
    /// [`poot_profile::TimingSnapshot`]. Pure bookkeeping over numbers the step already produced;
    /// never touches the device. Review F9: builds the per-dispatch name and label lookup and formats
    /// `device_label` only for a sampled step - counters-only mode (or a step `consume_sample_
    /// decision` skips) costs only the cumulative-counter update below, as the type's own doc
    /// already claimed.
    #[allow(clippy::too_many_arguments)]
    fn record_step_timing(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        replay: u64,
        encode: std::time::Duration,
        wait: std::time::Duration,
        wall: std::time::Duration,
        time: &DeviceTime,
    ) -> Result<(), ExecError> {
        let counters_only = matches!(self.timing_options, TimingOptions::CountersOnly);
        let sampled = self.consume_sample_decision();
        // Cheap sum/span-only conversion (no per-dispatch allocation): cumulative counts never
        // depend on timestamp coverage (Card 552 scope), so this always reflects `time` honestly,
        // even in counters-only mode.
        let device_summary = match time {
            DeviceTime::Unknown => poot_profile::DeviceCoverage::Unknown,
            DeviceTime::Measured(m) => {
                poot_profile::DeviceCoverage::Measured(poot_profile::MeasuredDevice {
                    sum_of_dispatch_durations: m.sum_of_dispatch_durations,
                    device_span: m.device_span,
                })
            }
        };
        let (device, dispatches, device_label) = if counters_only {
            // Review F9: never build the per-dispatch name lookup or format a device
            // label - counters-only retention is 0-sized, so any per-dispatch `Vec` built here
            // would only be immediately discarded.
            (device_summary, Vec::new(), String::new())
        } else if !sampled {
            // Review F3: this step's declared `sample_every` policy skips it - report the same
            // shape a backend that cannot measure at all would, regardless of what `time` held,
            // so `Coverage::Unsampled` has a real, visible effect.
            (
                poot_profile::DeviceCoverage::Unknown,
                Vec::new(),
                String::new(),
            )
        } else {
            let names: Vec<(&'static str, String)> = {
                let loaded = self.loaded(exe)?;
                let e = loaded
                    .entries
                    .get(entry.index())
                    .and_then(Option::as_ref)
                    .ok_or(LoadError::UnknownHandle)?;
                e.steps
                    .iter()
                    .map(|s| (s.kind_name, s.label.clone()))
                    .collect()
            };
            let dispatches = match time {
                DeviceTime::Unknown => Vec::new(),
                DeviceTime::Measured(m) => m
                    .dispatches
                    .iter()
                    .map(|d| {
                        let step = names.get(d.index);
                        poot_profile::DispatchTiming {
                            dispatch_index: d.index,
                            kind_name: step.map_or("unknown", |(kind, _)| kind),
                            plan_label: step.map(|(_, label)| label.clone()),
                            device: Some(d.duration),
                        }
                    })
                    .collect(),
            };
            (device_summary, dispatches, format!("{:?}", self.backend()))
        };
        self.timing.record(
            entry.raw(),
            replay,
            device_label,
            poot_profile::HostTiming { encode, wait, wall },
            device,
            dispatches,
        );
        Ok(())
    }
}

/// The output location for one computed equation: the paired state input's buffer when
/// `commit_plan` donates this output in place, or its arena slot otherwise (Card 547b).
struct OutLocArgs<'a> {
    out: ValueId,
    commit_plan: &'a poot_graph_plan::StateCommitPlan,
    buffer_plan: &'a poot_graph_plan::BufferPlan,
}

fn out_loc(args: OutLocArgs<'_>, values: &[Option<(Loc, u32)>]) -> Result<Loc, ExecError> {
    if let Some(&state_input) = args.commit_plan.inplace.get(&args.out) {
        return Ok(values[state_input].expect("state bound").0);
    }
    let slot = args
        .buffer_plan
        .slot(args.out)
        .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
    Ok(Loc::Arena(slot.index()))
}

impl<D: Device> Engine<D> {
    fn loaded_entry(&self, exe: ExecutableId, entry: EntryId) -> Result<&Entry<D>, ExecError> {
        let loaded = self
            .executables
            .get(exe.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        Ok(loaded
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?)
    }
}

/// The host dtype of a dense, natural-width output lane (`F32`, `I32`, `BF16`, `F16`); `None` for a
/// packed or mirrored lane, whose bytes are not a tensor of the declared dtype.
fn host_dtype(storage: BufferStorage) -> Option<DType> {
    [
        (BufferStorage::f32(), DType::F32),
        (BufferStorage::i32(), DType::I32),
        (BufferStorage::bf16(), DType::BF16),
        (BufferStorage::f16(), DType::F16),
    ]
    .into_iter()
    .find_map(|(lane, dtype)| (lane == storage).then_some(dtype))
}

impl<D: Device> OutputSource for Engine<D> {
    fn read_output(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        out: &mut [u8],
    ) -> Result<(), ExecError> {
        let backend = self.backend();
        let Self {
            executables,
            device,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let e = loaded
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        // When the output aliases a commit destination (`preserve_output`), the commit copy has
        // already overwritten it for the next step; the snapshot taken before that copy is the
        // real answer.
        let loc = e.preserve_output.unwrap_or(e.output);
        let buffer = resolve::<D>(loc, &e.locals, &e.arena, &loaded.weights, &loaded.state);
        device
            .read(buffer, &mut out[..e.output_bytes])
            .map_err(device_error(backend))
    }

    fn output_bytes(&self, exe: ExecutableId, entry: EntryId) -> Result<usize, ExecError> {
        Ok(self.loaded_entry(exe, entry)?.output_bytes)
    }

    fn output_host(
        &self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<(DType, Vec<usize>), ExecError> {
        self.loaded_entry(exe, entry)?
            .output_host
            .clone()
            .ok_or_else(|| {
                LoadError::Unimplemented("a host tensor of a packed or mirrored output lane").into()
            })
    }
}

impl<D: Device> Executor for Engine<D> {
    fn target_set(&self) -> TargetSet {
        TargetSet::single(DEVICE, self.device.target())
    }

    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: WeightSource,
    ) -> Result<ExecutableId, ExecError> {
        self.check_not_poisoned()?;
        self.executables.push(Some(Loaded {
            binder: WeightBinder::new(store, weights),
            weights: Vec::new(),
            state: Vec::new(),
            entries: Vec::new(),
            needs_reset: false,
        }));
        Ok(ExecutableId::new(self.executables.len() - 1))
    }

    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<ValidationOutputs>,
    ) -> Result<EntryId, ExecError> {
        self.check_not_poisoned()?;
        let mut stages = program.stages();
        let (Some((_, device, program)), None) = (stages.next(), stages.next()) else {
            return Err(LoadError::NotSingleStage(program.stages().count()).into());
        };
        if device != DEVICE {
            return Err(LoadError::Unimplemented("a stage on another device").into());
        }
        // The contract admits Submission::Replay only. Refused here,
        // before load_entry's first allocation - Submission::Eager survives as an enum and on the
        // pre-contract executors (the bench prefill arm, PTX/ROCm/VLM/speculative), never here.
        if program.submission() != Submission::Replay {
            return Err(LoadError::Unimplemented(
                "Submission::Eager (the contract admits Submission::Replay only)",
            )
            .into());
        }
        // CR30-lite: weight_buffer/state_buffer mutate the executable's resident
        // weight/state caches as load_entry walks the program, before the entry itself is ever
        // pushed. Snapshot their lengths so a mid-load refusal rolls back whatever this call newly
        // allocated, instead of leaving an orphaned buffer charged to the executable.
        let (weights_len, state_len) = {
            let loaded = self.loaded(exe)?;
            (loaded.weights.len(), loaded.state.len())
        };
        let entry = match self.load_entry(exe, program) {
            Ok(entry) => entry,
            Err(e) => {
                {
                    let loaded = self.loaded(exe)?;
                    loaded.weights.truncate(weights_len);
                    loaded.state.truncate(state_len);
                }
                self.residency.retain(|_, upload| upload.strong_count() > 0);
                return Err(e);
            }
        };
        let entries = &mut self.loaded(exe)?.entries;
        entries.push(Some(entry));
        Ok(EntryId::new(entries.len() - 1))
    }

    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        _sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        // Card 552: the host-wall partition is always measured (never gated by `TimingOptions`,
        // since `Instant::now()` costs nothing a step doesn't already pay). `encode` covers
        // binding + record/replay (building and submitting the work); `wait` is the explicit
        // blocking `synchronize()` below; `other` (never stored, only ever printed) is whatever
        // this step's wall includes past those two - never a device-time subtraction.
        let step_t0 = Instant::now();
        self.check_not_poisoned()?;
        if self
            .executables
            .get(exe.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?
            .needs_reset
        {
            return Err(ExecError::NeedsReset(exe));
        }
        self.bind(exe, entry, inputs)?;
        let backend = self.backend();
        let error = device_error(backend);
        // The contract admits Submission::Replay only (add_entry already refused anything else):
        // record once, then always replay.
        let needs_recording = {
            let loaded = self.loaded(exe)?;
            let e = loaded
                .entries
                .get(entry.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            e.recording.is_none()
        };
        if needs_recording {
            let recording = self.run_transaction(exe, entry.index(), Submission::Replay, false)?;
            let loaded = self.loaded(exe)?;
            let e = loaded
                .entries
                .get_mut(entry.index())
                .and_then(Option::as_mut)
                .ok_or(LoadError::UnknownHandle)?;
            e.recording = recording;
            e.recordings += 1;
        }
        let replay_result = {
            let Self {
                executables,
                device,
                ..
            } = self;
            let loaded = executables
                .get_mut(exe.index())
                .and_then(Option::as_mut)
                .ok_or(LoadError::UnknownHandle)?;
            let e = loaded
                .entries
                .get(entry.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            let recording = e.recording.as_ref().expect("recorded above");
            device.replay(recording)
        };
        if let Err(err) = replay_result {
            self.abort_after_failure(exe, true);
            return Err(error(err));
        }
        let loaded = self.loaded(exe)?;
        let e = loaded
            .entries
            .get_mut(entry.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        e.replays += 1;
        let replay_count = e.replays;
        let encode_dur = step_t0.elapsed();
        let sync_t0 = Instant::now();
        if let Err(err) = self.device.synchronize() {
            // A kernel-assert fault surfaces here (Z10): not a device-level failure, so it does not
            // poison the engine or mark the executable NeedsReset - the device itself is in a
            // consistent, reusable state, the program result is simply faulted.
            if let Some((kernel, code)) = self.device.classify_fault(&err) {
                return Err(ExecError::Fault { kernel, code });
            }
            // CR30: a non-fault synchronize failure (e.g. a device timeout) can
            // follow a replay that already wrote state, so it is classified the same way a replay
            // failure is - poisoned if abort fails, and NeedsReset since rollback is not proven.
            self.abort_after_failure(exe, true);
            return Err(error(err));
        }
        let wait_dur = sync_t0.elapsed();
        // CR30: this synchronize just proved every op this device had queued before it - including
        // whatever `unload` deferred into `pending_release` - has completed. Safe to finally drop.
        self.pending_release.clear();
        self.residency.retain(|_, upload| upload.strong_count() > 0);
        self.check_validation(exe, entry.index())?;
        let time = self.device.device_time();
        let wall_dur = step_t0.elapsed();
        self.record_step_timing(
            exe,
            entry,
            replay_count,
            encode_dur,
            wait_dur,
            wall_dur,
            &time,
        )?;
        Ok(StepOutputs::new(self, exe, entry, time))
    }

    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        let StateScope::All = scope;
        let backend = self.backend();
        let Self {
            executables,
            device,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        for state in &loaded.state {
            device
                .write(&state.buffer, &vec![0u8; state.bytes])
                .map_err(device_error(backend))?;
        }
        loaded.needs_reset = false;
        Ok(())
    }

    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        let loaded = self.loaded(exe)?;
        let slot = loaded
            .entries
            .get_mut(entry.index())
            .ok_or(LoadError::UnknownHandle)?;
        slot.take().ok_or(LoadError::UnknownHandle)?;
        Ok(())
    }

    fn stats(&self) -> ExecutorStats {
        let mut recordings = 0u64;
        let mut replays = 0u64;
        for loaded in self.executables.iter().flatten() {
            for entry in loaded.entries.iter().flatten() {
                recordings += entry.recordings;
                replays += entry.replays;
            }
        }
        // Card 547a: read the device's own runtime counters, not an engine-side tally over its own
        // bookkeeping, so cross-backend equality is not equal by construction.
        ExecutorStats {
            recordings,
            replays,
            memory: self.device.memory(),
            timing: self.timing.clone(),
        }
    }

    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        let slot = self
            .executables
            .get_mut(exe.index())
            .ok_or(LoadError::UnknownHandle)?;
        let loaded = slot.take().ok_or(LoadError::UnknownHandle)?;
        if loaded.needs_reset {
            // CR30 (SC-010/SC-011, moved from 546a): a failed transaction's completion is not
            // proven, so this executable's buffers/modules/events must outlive the handle - retained
            // here until `step`'s next successful `synchronize()` drains `pending_release`, never
            // dropped with the handle itself.
            self.pending_release.push(loaded);
        } else {
            drop(loaded);
        }
        self.residency.retain(|_, upload| upload.strong_count() > 0);
        Ok(())
    }
}

#[cfg(test)]
mod kernel_identity_tests {
    use super::*;
    use crate::device::DeviceTime;

    struct NoopDevice;

    impl Device for NoopDevice {
        type Buffer = ();
        type Kernel = String;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target {
                backend: Backend::SpirvVulkan,
                caps: poot_target::DeviceCaps::wgpu_rdna3_igpu(),
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            _role: BufferRole,
            _storage: BufferStorage,
            _elems: usize,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn load_kernel(
            &mut self,
            key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<String, std::convert::Infallible> {
            Ok(key.to_string())
        }
        fn begin(&mut self, _submission: Submission) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<Option<()>, std::convert::Infallible> {
            Ok(None)
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write(&mut self, _dst: &(), _bytes: &[u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn read(&mut self, _src: &(), _out: &mut [u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
    }

    /// SC-024: two equations whose planner-given display key happens to collide but whose compiled
    /// bodies differ must never alias onto the same device kernel, in either insertion order.
    /// Mutation (run by hand): revert `kernel_index`'s key from `KernelCache::key(body).
    /// fingerprint()` back to the display `key: &str`; the second body's `kernel()` call returns the
    /// first body's index instead of compiling its own, and the row goes red.
    #[test]
    fn colliding_display_keys_with_different_bodies_never_alias_in_either_order() {
        let body_a = poot_test_util::kernel_fixtures::reduce_last(
            "shared_key",
            poot_kernel_ir::BinOp::Add,
            4,
            0.0,
        );
        let body_b = poot_test_util::kernel_fixtures::reduce_last(
            "shared_key",
            poot_kernel_ir::BinOp::Add,
            8,
            0.0,
        );
        assert_ne!(
            format!("{body_a:?}"),
            format!("{body_b:?}"),
            "fixture bodies must actually differ"
        );

        let mut engine = Engine::new(NoopDevice);
        let index_a = engine
            .kernel(
                "shared_key",
                &body_a,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        let index_b = engine
            .kernel(
                "shared_key",
                &body_b,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        assert_ne!(
            index_a, index_b,
            "two different bodies sharing a display key must not alias (a-then-b)"
        );

        let mut engine = Engine::new(NoopDevice);
        let index_b2 = engine
            .kernel(
                "shared_key",
                &body_b,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        let index_a2 = engine
            .kernel(
                "shared_key",
                &body_a,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        assert_ne!(
            index_a2, index_b2,
            "two different bodies sharing a display key must not alias (b-then-a)"
        );

        let index_a_again = engine
            .kernel(
                "shared_key",
                &body_a,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        assert_eq!(
            index_a_again, index_a2,
            "the same body content must still hit the cache"
        );
    }
}

#[cfg(test)]
mod arena_slot_alloc_tests {
    use super::*;
    use crate::device::DeviceTime;

    /// Records every `elems` a call actually reached `Device::allocate` with (never recomputes one).
    #[derive(Default)]
    struct RecordingDevice {
        seen_elems: Vec<usize>,
    }

    impl Device for RecordingDevice {
        type Buffer = ();
        type Kernel = String;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target {
                backend: Backend::SpirvVulkan,
                caps: poot_target::DeviceCaps::wgpu_rdna3_igpu(),
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            _role: BufferRole,
            _storage: BufferStorage,
            elems: usize,
        ) -> Result<(), std::convert::Infallible> {
            self.seen_elems.push(elems);
            Ok(())
        }
        fn load_kernel(
            &mut self,
            key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<String, std::convert::Infallible> {
            Ok(key.to_string())
        }
        fn begin(&mut self, _submission: Submission) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<Option<()>, std::convert::Infallible> {
            Ok(None)
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write(&mut self, _dst: &(), _bytes: &[u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn read(&mut self, _src: &(), _out: &mut [u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
    }

    /// Review F3: `BufferPlan::slots()` hands out a packed slot's already native device-element count
    /// (e.g. 4 `u32` words for 8 packed BF16 logical elements) - `load_entry`'s arena loop must pass
    /// that straight to `Device::allocate`, never back through `Self::allocate`'s own
    /// `BufferStorage::device_elems` conversion, which would halve it again. Mutation: call
    /// `self.allocate(role, storage, elems)` here instead of `self.allocate_arena_slot` (as
    /// `load_entry` did before this fix) - `RecordingDevice` then observes `2`, not the slot's own `4`,
    /// and the row goes red.
    #[test]
    fn arena_slot_native_elems_reach_device_allocate_unconverted() {
        let mut engine = Engine::new(RecordingDevice::default());
        let packed = BufferStorage::bf16_packed();
        let slot_native_elems = 4usize;

        engine
            .allocate_arena_slot(BufferRole::Activation, packed, slot_native_elems)
            .unwrap();

        assert_eq!(
            engine.device.seen_elems,
            vec![slot_native_elems],
            "a packed arena slot's own native element count must reach Device::allocate unconverted, \
             not BufferStorage::device_elems applied to it a second time"
        );
    }
}

/// Card 1007: a packed-F16 `DenseContraction` weight reaches the device as its checkpoint bytes in `u32` words,
/// through the production `load_weights`/`add_entry` path of a compiled program.
#[cfg(test)]
mod f16_packed_weight_upload_tests {
    use super::*;
    use crate::Executor;
    use poot_graph_ir::Builder;
    use poot_graph_plan::{
        CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
        TargetSet, compile_staged,
    };
    use poot_quant::weights::{DenseWeight, WeightEntry};

    /// Records every allocation (role, storage, native elements) and every write, by buffer index.
    #[derive(Default)]
    struct UploadRecorder {
        allocations: Vec<(BufferRole, BufferStorage, usize)>,
        writes: Vec<(usize, Vec<u8>)>,
    }

    impl Device for UploadRecorder {
        type Buffer = usize;
        type Kernel = ();
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target {
                backend: Backend::SpirvVulkan,
                caps: poot_target::DeviceCaps::wgpu_rdna3_igpu(),
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            storage: BufferStorage,
            elems: usize,
        ) -> Result<usize, std::convert::Infallible> {
            self.allocations.push((role, storage, elems));
            Ok(self.allocations.len() - 1)
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn begin(&mut self, _submission: Submission) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<Option<()>, std::convert::Infallible> {
            Ok(None)
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write(&mut self, dst: &usize, bytes: &[u8]) -> Result<(), std::convert::Infallible> {
            self.writes.push((*dst, bytes.to_vec()));
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
    }

    /// `N * K` is odd, so the last packed word carries one element and a zero pad.
    const N: usize = 5;
    const K: usize = 7;

    /// `matmul(x, transpose(w))` with `w` an F16 `[N, K]` const holding `w_bits`, compiled for the recorder's
    /// target and added as an entry over a store holding both consts.
    fn add_projection(
        engine: &mut Engine<UploadRecorder>,
        w_bits: &[u16],
    ) -> Result<EntryId, ExecError> {
        add_projection_declared(engine, w_bits, DType::F16, DType::F16)
    }

    /// [`add_projection`] where the graph declares `w` as `declared` while the store holds `w_bits` as `stored`.
    fn add_projection_declared(
        engine: &mut Engine<UploadRecorder>,
        w_bits: &[u16],
        stored: DType,
        declared: DType,
    ) -> Result<EntryId, ExecError> {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, K]));
        let w = b.constant("w", TensorType::new(vec![N, K], declared));
        let out = b.matmul(x, b.transpose(w, vec![1, 0]));
        let g = b.finish(out).with_validations(Vec::new());
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .expect("the projection compiles");
        let x_bytes: Vec<u8> = (0..4 * K).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut store = WeightStore::builder();
        for (name, dtype, shape, bytes) in [
            ("x", DType::F32, vec![4, K], x_bytes),
            ("w", stored, vec![N, K], w_bytes),
        ] {
            store
                .insert(
                    name,
                    WeightEntry::Dense(
                        DenseWeight::try_new(dtype, shape, Arc::from(bytes)).unwrap(),
                    ),
                )
                .unwrap();
        }
        let exe = engine.load_weights(Arc::new(store.build()), crate::WeightSource::ConstNames)?;
        engine.add_entry(exe, &program)
    }

    /// Distinct finite binary16 words, one per weight element.
    fn finite_bits() -> Vec<u16> {
        (0..N * K).map(|i| 0x3c00 + 37 * i as u16).collect()
    }

    /// Card 1007 acceptance row 2: the F16 const is allocated as packed-F16 storage of `ceil(N * K / 2)` words and
    /// written with exactly the checkpoint's bytes, zero-padded to the last word: no f32 host copy exists. No
    /// weight buffer of `N * K` f32 elements is allocated either.
    ///
    /// Mutation: in `lanes::encode_stored`'s `(F16, f16_packed())` arm, widen each element to f32 bytes on upload;
    /// the written bytes are twice as long and this row goes red.
    #[test]
    fn an_f16_contraction_weight_uploads_its_two_byte_words_packed() {
        let mut engine = Engine::new(UploadRecorder::default());
        let bits = finite_bits();
        add_projection(&mut engine, &bits).expect("the entry loads");
        let device = &engine.device;
        let packed: Vec<usize> = device
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, (_, storage, _))| *storage == BufferStorage::f16_packed())
            .map(|(index, _)| index)
            .collect();
        assert_eq!(packed.len(), 1, "allocations: {:?}", device.allocations);
        let buffer = packed[0];
        assert_eq!(
            device.allocations[buffer],
            (
                BufferRole::Weight,
                BufferStorage::f16_packed(),
                (N * K).div_ceil(2)
            )
        );
        let mut want: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        want.resize((N * K).div_ceil(2) * 4, 0);
        let written: Vec<&Vec<u8>> = device
            .writes
            .iter()
            .filter(|(dst, _)| *dst == buffer)
            .map(|(_, bytes)| bytes)
            .collect();
        assert_eq!(
            written,
            vec![&want],
            "the packed weight's upload is its stored bytes, padded"
        );
        assert!(
            !device
                .allocations
                .iter()
                .any(|&(role, storage, elems)| role == BufferRole::Weight
                    && storage == BufferStorage::f32()
                    && elems == N * K),
            "no f32 weight buffer of the F16 weight's size: {:?}",
            device.allocations
        );
    }

    /// Card 1007: a packed-F16 weight holding an infinity is refused at upload with a typed error naming the
    /// weight and the element, since the generated bodies decode binary16 over its finite values only.
    ///
    /// Mutation: drop the non-finite scan in `lanes::encode_stored`'s `(F16, f16_packed())` arm; the entry loads
    /// and this row goes red.
    #[test]
    fn a_non_finite_packed_f16_weight_is_a_typed_refusal() {
        let mut engine = Engine::new(UploadRecorder::default());
        let mut bits = finite_bits();
        bits[5] = 0xfc00; // -inf
        match add_projection(&mut engine, &bits) {
            Err(ExecError::Load(error)) => match *error {
                LoadError::NonFiniteF16Weight { name, index, bits } => {
                    assert_eq!((name.as_str(), index, bits), ("w", 5, 0xfc00));
                }
                other => panic!("expected NonFiniteF16Weight, got {other}"),
            },
            Err(other) => panic!("expected NonFiniteF16Weight, got {other}"),
            Ok(_) => panic!("a non-finite packed-F16 weight loaded"),
        }
    }

    /// Card 1008 (and Card 1007's review row): a graph that declares a stored-F16 contraction weight F32 is
    /// refused at load with a typed error naming the weight and both dtypes. Neither the plan's f32 lane nor a
    /// host-widened copy ever holds the weight.
    ///
    /// Mutation: delete the `dense.dtype() != aval.dtype` check in `Engine::weight_buffer`; the weight then
    /// reaches `encode_stored` and fails as the untyped-by-dtype `WeightFormat`, so this row goes red.
    #[test]
    fn a_contraction_weight_declared_f32_but_stored_f16_is_a_typed_refusal() {
        let mut engine = Engine::new(UploadRecorder::default());
        match add_projection_declared(&mut engine, &finite_bits(), DType::F16, DType::F32) {
            Err(ExecError::Load(error)) => match *error {
                LoadError::WeightDtype {
                    name,
                    stored,
                    declared,
                } => assert_eq!(
                    (name.as_str(), stored, declared),
                    ("w", DType::F16, DType::F32)
                ),
                other => panic!("expected WeightDtype, got {other}"),
            },
            Err(other) => panic!("expected WeightDtype, got {other}"),
            Ok(_) => panic!("an F16-stored weight declared F32 loaded"),
        }
        assert!(
            !engine
                .device
                .allocations
                .iter()
                .any(|&(role, storage, elems)| role == BufferRole::Weight
                    && storage == BufferStorage::f32()
                    && elems == N * K),
            "no f32 weight buffer was allocated for the refused weight"
        );
    }

    /// Card 1011: the plan never retypes a BF16 const, so a BF16-stored weight declared F32 is a refusal like
    /// the F16 one above, not a widening on upload: a typed `WeightDtype` naming the weight and both dtypes,
    /// and no f32 buffer of the weight's size is allocated or written.
    ///
    /// Mutation: re-admit the BF16/F32 pair in `Engine::weight_buffer`'s dtype check (and widen it in
    /// `lanes::encode_stored`); the entry loads and this row goes red.
    #[test]
    fn a_bf16_weight_declared_f32_is_a_typed_refusal_never_widened() {
        let mut engine = Engine::new(UploadRecorder::default());
        let bits: Vec<u16> = (0..N * K).map(|i| 0x3f80 + i as u16).collect();
        match add_projection_declared(&mut engine, &bits, DType::BF16, DType::F32) {
            Err(ExecError::Load(error)) => match *error {
                LoadError::WeightDtype {
                    name,
                    stored,
                    declared,
                } => assert_eq!(
                    (name.as_str(), stored, declared),
                    ("w", DType::BF16, DType::F32)
                ),
                other => panic!("expected WeightDtype, got {other}"),
            },
            Err(other) => panic!("expected WeightDtype, got {other}"),
            Ok(_) => panic!("a BF16-stored weight declared F32 loaded"),
        }
        assert!(
            !engine
                .device
                .allocations
                .iter()
                .any(|&(role, storage, elems)| role == BufferRole::Weight
                    && storage == BufferStorage::f32()
                    && elems == N * K),
            "no f32 weight buffer was allocated for the refused weight"
        );
    }
}

/// Card 564: weight binding through the executable's `WeightSource` - the map's views to stored byte
/// runs, the typed unbound refusal, residency by what uploads, and the hosted embed fill - driven
/// through the production `load_weights`/`add_entry`/`step` path on a recording device.
#[cfg(test)]
mod weight_source_tests {
    use super::*;
    use crate::{NoSync, StepInputs};
    use poot_graph_ir::{BinOp, Builder, Graph};
    use poot_graph_plan::{
        CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
        TargetSet, WeightFormats, bind_packed_weights, compile_staged,
    };
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{
        AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightKey, WeightMap, WeightRole,
        WeightView,
    };
    use poot_quant::{PackedPayload, SourceRole};
    use poot_runtime_common::MemoryCounters;
    use std::ops::Range;

    /// Records every allocation (role, storage, native elements) and every write by buffer index,
    /// and charges each allocation to the runtime memory counters `Engine::stats` reads. `cap`, when
    /// set, is the target's single-buffer limit, so `compile`'s legalize sees it.
    #[derive(Default)]
    struct Recorder {
        allocations: Vec<(BufferRole, BufferStorage, usize)>,
        writes: Vec<(usize, Vec<u8>)>,
        memory: MemoryCounters,
        cap: Option<u64>,
    }

    impl Device for Recorder {
        type Buffer = usize;
        type Kernel = ();
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> poot_graph_plan::Target {
            let mut caps = poot_target::DeviceCaps::wgpu_rdna3_igpu();
            if let Some(cap) = self.cap {
                caps.max_buffer_bytes = cap;
            }
            poot_graph_plan::Target {
                backend: Backend::SpirvVulkan,
                caps,
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            self.memory.snapshot_all()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            storage: BufferStorage,
            elems: usize,
        ) -> Result<usize, std::convert::Infallible> {
            let bytes = (elems * storage.element().byte_width()) as u64;
            std::mem::forget(self.memory.record_alloc(role, bytes));
            self.allocations.push((role, storage, elems));
            Ok(self.allocations.len() - 1)
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn begin(&mut self, _submission: Submission) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<Option<()>, std::convert::Infallible> {
            Ok(Some(()))
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write(&mut self, dst: &usize, bytes: &[u8]) -> Result<(), std::convert::Infallible> {
            self.writes.push((*dst, bytes.to_vec()));
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
    }

    const N: usize = 4;
    const K: usize = 32;

    fn id(role: WeightRole) -> WeightId {
        WeightId::layer(0, role)
    }

    const Q: WeightRole = WeightRole::Attn(AttnRole::Q);
    const KR: WeightRole = WeightRole::Attn(AttnRole::K);
    const V: WeightRole = WeightRole::Attn(AttnRole::V);
    const GATE: WeightRole = WeightRole::Ffn(FfnRole::Gate);

    fn f32_entry(shape: Vec<usize>, seed: u32) -> WeightEntry {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| (seed as f32 * 1000.0 + i as f32).to_le_bytes())
            .collect();
        WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap())
    }

    fn store(entries: Vec<(&str, WeightEntry)>) -> Arc<WeightStore> {
        let mut store = WeightStore::builder();
        for (key, entry) in entries {
            store.insert(key, entry).unwrap();
        }
        Arc::new(store.build())
    }

    fn map(store: &WeightStore, views: Vec<(WeightId, WeightView)>) -> Arc<WeightMap> {
        let mut map = WeightMap::builder(store);
        for (id, view) in views {
            map.map(id, view).unwrap();
        }
        Arc::new(map.build())
    }

    fn compile(engine: &Engine<Recorder>, g: &Graph) -> StagedProgram<ValidationOutputs> {
        compile_staged(
            &g.clone().with_validations(Vec::new()),
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .expect("the fixture compiles")
    }

    /// `out = a + b`, both F32 `[N, K]` consts named `a` and `b`; returns the graph and `b`'s id.
    fn sum_of(a: &str, b_name: &str) -> (Graph, ValueId) {
        let b = Builder::new();
        let x = b.constant(a, TensorType::f32(vec![N, K]));
        let y = b.constant(b_name, TensorType::f32(vec![N, K]));
        let out = b.binary(BinOp::Add, x, y);
        (b.finish(out), y.id)
    }

    fn weight_allocations(engine: &Engine<Recorder>) -> u64 {
        engine
            .stats()
            .memory
            .iter()
            .find(|(role, _)| *role == BufferRole::Weight)
            .map_or(0, |(_, snapshot)| snapshot.allocations)
    }

    fn written(engine: &Engine<Recorder>) -> Vec<&[u8]> {
        engine
            .device()
            .writes
            .iter()
            .map(|(_, bytes)| bytes.as_slice())
            .collect()
    }

    fn unbound(result: Result<EntryId, ExecError>) -> (ValueId, String) {
        match result {
            Err(ExecError::Load(error)) => match *error {
                LoadError::Unbound { value, name } => (value, name),
                other => panic!("expected Unbound, got {other}"),
            },
            Err(other) => panic!("expected Unbound, got {other}"),
            Ok(_) => panic!("the entry loaded"),
        }
    }

    /// SC-002: a graph const with no store entry and no step value is a typed refusal naming the
    /// value and the const, before any step runs.
    ///
    /// Mutation: in `WeightBinder::entry`, answer a missing key with a zero-filled dense entry of
    /// the declared shape; the entry loads and this row goes red.
    #[test]
    fn a_const_with_no_store_entry_and_no_step_value_is_unbound() {
        let mut engine = Engine::new(Recorder::default());
        let (g, missing) = sum_of("a", "b");
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(
                store(vec![("a", f32_entry(vec![N, K], 1))]),
                WeightSource::ConstNames,
            )
            .unwrap();
        assert_eq!(
            unbound(engine.add_entry(exe, &program)),
            (missing, "b".to_string())
        );
    }

    /// SC-003: under `Map`, a const whose name equals a store key but which the map does not name is
    /// `Unbound`; the same graph and store bind under `ConstNames`, so the store key is there.
    ///
    /// Mutation: in `WeightBinder::locate`'s `Rule::Map` arm, fall back to the name-equality lookup
    /// when the map has no id for the name; the const binds and this row goes red.
    #[test]
    fn a_map_const_that_is_only_a_store_key_is_unbound() {
        let q = id(Q).const_name();
        let (g, embed) = sum_of(&q, "w.embed");
        let weights = store(vec![
            (q.as_str(), f32_entry(vec![N, K], 1)),
            ("w.embed", f32_entry(vec![N, K], 2)),
        ]);
        let mapped = map(
            &weights,
            vec![(id(Q), WeightView::Stored(q.as_str().into()))],
        );

        let mut engine = Engine::new(Recorder::default());
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(Arc::clone(&weights), WeightSource::Map(mapped))
            .unwrap();
        assert_eq!(
            unbound(engine.add_entry(exe, &program)),
            (embed, "w.embed".to_string())
        );

        let names = engine
            .load_weights(weights, WeightSource::ConstNames)
            .unwrap();
        engine
            .add_entry(names, &program)
            .expect("by name the store key binds");
    }

    /// SC-005 (S62-9 at the map level): one stored entry viewed by two ids (a head tied to the
    /// embedding) uploads once; two entries upload twice.
    ///
    /// Mutation: key `Loaded::residency` on the const name (`(name.to_string(), storage)`) instead
    /// of the stored spans; the tied map uploads twice and this row goes red.
    #[test]
    fn a_store_entry_viewed_by_two_ids_uploads_once() {
        let embed = WeightId::model(WeightRole::Embed);
        let head = WeightId::model(WeightRole::Head);
        let (g, _) = sum_of(&embed.const_name(), &head.const_name());

        let tied_store = store(vec![("tok", f32_entry(vec![N, K], 1))]);
        let tied = map(
            &tied_store,
            vec![
                (embed, WeightView::Stored("tok".into())),
                (head, WeightView::Stored("tok".into())),
            ],
        );
        let mut engine = Engine::new(Recorder::default());
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(tied_store, WeightSource::Map(tied))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        assert_eq!(weight_allocations(&engine), 1, "a tied head uploads once");

        let untied_store = store(vec![
            ("tok", f32_entry(vec![N, K], 1)),
            ("lm", f32_entry(vec![N, K], 2)),
        ]);
        let untied = map(
            &untied_store,
            vec![
                (embed, WeightView::Stored("tok".into())),
                (head, WeightView::Stored("lm".into())),
            ],
        );
        let mut engine = Engine::new(Recorder::default());
        let exe = engine
            .load_weights(untied_store, WeightSource::Map(untied))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        assert_eq!(weight_allocations(&engine), 2, "two entries upload twice");
    }

    /// `x @ w^T` for each of `ids`' weights, `[N, K]` each, summed; `x` is an activation slot.
    fn projections(ids: &[WeightId], formats: &WeightFormats) -> Graph {
        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::f32(vec![2, K]));
        let mut out = None;
        for id in ids {
            let w = b.constant(&id.const_name(), TensorType::f32(vec![N, K]));
            let y = b.matmul(x, b.transpose(w, vec![1, 0]));
            out = Some(match out {
                Some(acc) => b.binary(BinOp::Add, acc, y),
                None => y,
            });
        }
        bind_packed_weights(&b.finish(out.unwrap()), formats).unwrap()
    }

    /// Row views upload exactly the stored rows they name (SC-004's bytes; the device rows are in
    /// the backend suites): a fused `[3N, K]` entry viewed as Q/K/V by `RowRange` and a two-part
    /// `RowStack`, for a dense F32 store, uploaded as the stored element rows.
    #[test]
    fn dense_row_views_upload_their_stored_rows() {
        let fused = f32_entry(vec![3 * N, K], 1);
        let WeightEntry::Dense(dense) = &fused else {
            unreachable!()
        };
        let row = |r: usize| dense.bytes().as_slice()[r * K * 4..(r + 1) * K * 4].to_vec();
        let rows = |range: Range<usize>| range.flat_map(row).collect::<Vec<u8>>();
        let weights = store(vec![
            ("qkv", fused.clone()),
            ("g0", f32_entry(vec![N / 2, K], 2)),
            ("g1", f32_entry(vec![N / 2, K], 3)),
        ]);
        let mut gate = Vec::new();
        for key in ["g0", "g1"] {
            let Some(WeightEntry::Dense(part)) = weights.get(key) else {
                unreachable!()
            };
            gate.extend_from_slice(part.bytes().as_slice());
        }
        let row_range = |r: Range<usize>| WeightView::RowRange {
            key: "qkv".into(),
            rows: r,
        };
        let mapped = map(
            &weights,
            vec![
                (id(Q), row_range(0..N)),
                (id(KR), row_range(N..2 * N)),
                (id(V), row_range(2 * N..3 * N)),
                (
                    id(GATE),
                    WeightView::RowStack(vec![WeightKey::from("g0"), WeightKey::from("g1")]),
                ),
            ],
        );
        let g = projections(&[id(Q), id(KR), id(V), id(GATE)], &WeightFormats::default());
        let mut engine = Engine::new(Recorder::default());
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(weights, WeightSource::Map(mapped))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        let writes = written(&engine);
        for (label, want) in [
            ("q", rows(0..N)),
            ("k", rows(N..2 * N)),
            ("v", rows(2 * N..3 * N)),
            ("gate", gate),
        ] {
            assert!(
                writes.contains(&want.as_slice()),
                "{label}: no upload holds its stored rows"
            );
        }
        assert_eq!(weight_allocations(&engine), 4);
    }

    /// The packed half of the row-view bytes: a fused Q8_0 `[3N, K]` payload viewed as Q/K/V by
    /// `RowRange` uploads whole block rows (`K / 32` blocks of 34 bytes per row), and a two-part
    /// `RowStack` the parts back to back, each as the packed source's `u32` words.
    ///
    /// Mutation: in `WeightBinder::span`, compute a packed row range's byte offset from element rows
    /// (`rows.start * K`) instead of block rows (`rows.start * row_bytes`); K and V upload the wrong
    /// bytes and this row goes red.
    #[test]
    fn packed_row_views_upload_whole_block_rows() {
        let fused = poot_test_util::packed::random_payload(WeightFormat::Q8_0, [3 * N, K], 7);
        let parts = [
            poot_test_util::packed::random_payload(WeightFormat::Q8_0, [N / 2, K], 8),
            poot_test_util::packed::random_payload(WeightFormat::Q8_0, [N / 2, K], 9),
        ];
        let blocks = fused.bytes(SourceRole::Blocks).to_vec();
        let row_bytes = blocks.len() / (3 * N);
        let words = |bytes: &[u8]| -> Vec<u8> {
            poot_quant::packed_source_words(bytes)
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        let rows = |r: Range<usize>| words(&blocks[r.start * row_bytes..r.end * row_bytes]);
        let gate = words(
            &[
                parts[0].bytes(SourceRole::Blocks),
                parts[1].bytes(SourceRole::Blocks),
            ]
            .concat(),
        );
        let packed = |payload: &PackedPayload| WeightEntry::Packed(Arc::new(payload.clone()));
        let weights = store(vec![
            ("qkv", packed(&fused)),
            ("g0", packed(&parts[0])),
            ("g1", packed(&parts[1])),
        ]);
        let row_range = |r: Range<usize>| WeightView::RowRange {
            key: "qkv".into(),
            rows: r,
        };
        let mapped = map(
            &weights,
            vec![
                (id(Q), row_range(0..N)),
                (id(KR), row_range(N..2 * N)),
                (id(V), row_range(2 * N..3 * N)),
                (
                    id(GATE),
                    WeightView::RowStack(vec![WeightKey::from("g0"), WeightKey::from("g1")]),
                ),
            ],
        );
        let g = projections(
            &[id(Q), id(KR), id(V), id(GATE)],
            &WeightFormats::from_weight_map(&mapped),
        );
        let mut engine = Engine::new(Recorder::default());
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(weights, WeightSource::Map(mapped))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        let writes = written(&engine);
        for (label, want) in [
            ("q", rows(0..N)),
            ("k", rows(N..2 * N)),
            ("v", rows(2 * N..3 * N)),
            ("gate", gate),
        ] {
            assert!(
                writes.contains(&want.as_slice()),
                "{label}: no upload holds its stored block rows"
            );
        }
    }

    /// An engine whose target's buffer cap is one byte below a `[rows, K]` F32 table, and an embed
    /// gather of `w.embed` by the step's tokens compiled for it: `compile`'s legalize hosts the
    /// gather as a `Slot::TokenEmbed` input.
    fn hosted_embed(
        rows: usize,
        tokens: usize,
    ) -> (Engine<Recorder>, StagedProgram<ValidationOutputs>) {
        let engine = Engine::new(Recorder {
            cap: Some((rows * K * 4 - 1) as u64),
            ..Recorder::default()
        });
        let b = Builder::new();
        let embed = WeightId::model(WeightRole::Embed).const_name();
        let table = b.constant(&embed, TensorType::f32(vec![rows, K]));
        let ids = b.slot(Slot::Token, TensorType::new(vec![1, tokens], DType::I32));
        let out = b.gather(table, 0, ids);
        let program = compile(&engine, &b.finish(out));
        let (_, _, stage) = program.stages().next().unwrap();
        assert!(
            stage
                .graph()
                .slots
                .iter()
                .any(|&(_, slot)| slot == Slot::TokenEmbed),
            "the cap hosts the gather"
        );
        (engine, program)
    }

    fn step_tokens(engine: &mut Engine<Recorder>, exe: ExecutableId, entry: EntryId, ids: &[i32]) {
        let shape = [1, ids.len()];
        let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let mut inputs = StepInputs::new();
        inputs.push(
            SlotKey::new(Slot::Token, None),
            &shape,
            poot_tensor::HostView::new(DType::I32, ids.len(), &bytes).unwrap(),
        );
        engine.step(exe, entry, &inputs, &mut NoSync).unwrap();
    }

    /// The binder fills a hosted embed gather from the step's tokens and the map's embed view - here a
    /// two-part `RowStack`, so a token past the first part reads the second - for a dense table, and
    /// for a Q8_0 table through the payload's row decode (SC-006's bytes; the device row is in the
    /// wgpu suite).
    ///
    /// Mutation: in `WeightBinder::embed`, read row 0 for every token; the filled bytes are the first
    /// row repeated and this row goes red.
    #[test]
    fn a_hosted_embed_is_filled_from_the_tokens_and_the_embed_view() {
        let embed = WeightId::model(WeightRole::Embed);
        let ids = [5, 0, 3, 5];
        let dense = [f32_entry(vec![3, K], 1), f32_entry(vec![3, K], 2)];
        let packed = [
            poot_test_util::packed::random_payload(WeightFormat::Q8_0, [3, K], 3),
            poot_test_util::packed::random_payload(WeightFormat::Q8_0, [3, K], 4),
        ];
        let packed_entries = packed
            .iter()
            .map(|payload| WeightEntry::Packed(Arc::new(payload.clone())))
            .collect::<Vec<_>>();
        for (label, parts) in [("dense", dense.to_vec()), ("q8_0", packed_entries)] {
            let row = |token: i32| -> Vec<u8> {
                let (part, row) = (token as usize / 3, token as usize % 3);
                match &parts[part] {
                    WeightEntry::Dense(d) => {
                        d.bytes().as_slice()[row * K * 4..(row + 1) * K * 4].to_vec()
                    }
                    WeightEntry::Packed(p) => {
                        let mut values = vec![0.0f32; K];
                        p.decode_row(row, &mut values).unwrap();
                        bytemuck::cast_slice(&values).to_vec()
                    }
                }
            };
            let want: Vec<u8> = ids.iter().flat_map(|&t| row(t)).collect();
            let weights = store(vec![("e0", parts[0].clone()), ("e1", parts[1].clone())]);
            let mapped = map(
                &weights,
                vec![(
                    embed,
                    WeightView::RowStack(vec![WeightKey::from("e0"), WeightKey::from("e1")]),
                )],
            );
            let (mut engine, program) = hosted_embed(6, ids.len());
            let exe = engine
                .load_weights(weights, WeightSource::Map(mapped))
                .unwrap();
            let entry = engine.add_entry(exe, &program).unwrap();
            step_tokens(&mut engine, exe, entry, &ids);
            assert!(
                written(&engine).contains(&want.as_slice()),
                "{label}: no step write holds the tokens' embed rows"
            );
        }
    }

    /// POOT-1017: a head over the target's buffer cap, read as the tracers read every dense weight
    /// (`x @ Transpose(w)`), is split by `compile`'s legalize into `<head>.chunkN` consts; the binder
    /// binds chunk `N` as the next rows of the head's own bytes. The parent is never uploaded whole:
    /// every weight buffer and every weight upload fits the cap, and the uploads, in chunk order, are
    /// the head's row blocks.
    ///
    /// Mutations (each red): shift a chunk's row range by one row in `chunks` (wrong offset); sort
    /// `chunks`' members descending (order reversed); in `locate_chunk` keep the parent's whole parts
    /// instead of cutting them to the chunk's rows (the 1280-byte parent uploads under a 512-byte cap).
    #[test]
    fn a_head_over_the_cap_binds_as_row_chunks_of_its_own_bytes() {
        const ROWS: usize = 10;
        const CAP: u64 = 512; // four 128-byte rows per chunk: row blocks 0..4, 4..8, 8..10.
        let head = WeightId::model(WeightRole::Head);
        let weights = store(vec![("head", f32_entry(vec![ROWS, K], 7))]);
        let mapped = map(
            &weights,
            vec![(head, WeightView::Stored(WeightKey::from("head")))],
        );
        let mut engine = Engine::new(Recorder {
            cap: Some(CAP),
            ..Recorder::default()
        });
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]));
        let w = b.constant(&head.const_name(), TensorType::f32(vec![ROWS, K]));
        let out = b.matmul(x, b.transpose(w, vec![1, 0]));
        let program = compile(&engine, &b.finish(out));
        let exe = engine
            .load_weights(Arc::clone(&weights), WeightSource::Map(mapped))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();

        let WeightEntry::Dense(parent) = weights.get("head").unwrap() else {
            unreachable!("the fixture head is dense")
        };
        let parent = parent.bytes().as_slice();
        let row_block = |rows: Range<usize>| parent[rows.start * K * 4..rows.end * K * 4].to_vec();
        let device = engine.device();
        let uploads: Vec<&Vec<u8>> = device
            .writes
            .iter()
            .filter(|(buffer, _)| device.allocations[*buffer].0 == BufferRole::Weight)
            .map(|(_, bytes)| bytes)
            .collect();
        for (role, storage, elems) in &device.allocations {
            if *role == BufferRole::Weight {
                let bytes = (elems * storage.element().byte_width()) as u64;
                assert!(
                    bytes <= CAP,
                    "a {bytes}-byte weight buffer exceeds the {CAP}-byte cap"
                );
            }
        }
        for upload in &uploads {
            assert!(
                upload.len() as u64 <= CAP,
                "a {}-byte weight upload exceeds the {CAP}-byte cap: the parent went up whole",
                upload.len()
            );
        }
        assert_eq!(
            uploads,
            [row_block(0..4), row_block(4..8), row_block(8..10)]
                .iter()
                .collect::<Vec<_>>(),
            "the weight uploads are the head's row blocks, in chunk order"
        );
    }

    /// Chunks that do not tile from index 0 are a graph no binder can place: refused by name, never
    /// bound to guessed rows.
    #[test]
    fn a_chunk_set_with_a_missing_index_is_refused() {
        let shape = [4usize, K];
        let refused = binder::chunks([
            (0, "w.head.chunk0", shape.as_slice()),
            (1, "w.head.chunk2", shape.as_slice()),
        ])
        .expect_err("chunk 1 is missing");
        assert!(
            matches!(&refused, LoadError::Unbound { name, .. } if name.contains("chunk 1 of w.head is missing")),
            "got {refused}"
        );
    }

    /// A token past the embed table's rows is a typed refusal naming the embed, never a read past it.
    #[test]
    fn a_token_outside_the_embed_table_is_refused() {
        let weights = store(vec![("w.embed", f32_entry(vec![6, K], 1))]);
        let (mut engine, program) = hosted_embed(6, 2);
        let exe = engine
            .load_weights(weights, WeightSource::ConstNames)
            .unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();
        let bytes: Vec<u8> = [1i32, 6].iter().flat_map(|t| t.to_le_bytes()).collect();
        let mut inputs = StepInputs::new();
        inputs.push(
            SlotKey::new(Slot::Token, None),
            &[1, 2],
            poot_tensor::HostView::new(DType::I32, 2, &bytes).unwrap(),
        );
        match engine.step(exe, entry, &inputs, &mut NoSync) {
            Err(ExecError::Bind(error)) => match *error {
                BindError::TokenOutOfRange { name, token, rows } => {
                    assert_eq!((name.as_str(), token, rows), ("w.embed", 6, 6));
                }
                other => panic!("expected TokenOutOfRange, got {other}"),
            },
            Err(other) => panic!("expected TokenOutOfRange, got {other}"),
            Ok(_) => panic!("a token past the table bound"),
        }
    }

    /// A device that keeps what each buffer was last written and logs the buffer indices it frees,
    /// so residency rows read an upload's bytes back through a production step output and observe
    /// when an upload is released.
    #[derive(Default)]
    struct Resident {
        /// When set, `replay` fails: the step's transaction marks its executable `NeedsReset`, so a
        /// later `unload` defers the executable into `pending_release`.
        fail_replay: Arc<std::sync::atomic::AtomicBool>,
        roles: Vec<BufferRole>,
        contents: Vec<Vec<u8>>,
        freed: Arc<std::sync::Mutex<Vec<usize>>>,
    }

    #[derive(Debug)]
    struct Fault;

    impl std::fmt::Display for Fault {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("injected replay fault")
        }
    }

    impl std::error::Error for Fault {}

    struct Held {
        index: usize,
        freed: Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl Drop for Held {
        fn drop(&mut self) {
            self.freed.lock().unwrap().push(self.index);
        }
    }

    impl Device for Resident {
        type Buffer = Held;
        type Kernel = ();
        type Recording = ();
        type Error = Fault;

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target {
                backend: Backend::SpirvVulkan,
                caps: poot_target::DeviceCaps::wgpu_rdna3_igpu(),
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            storage: BufferStorage,
            elems: usize,
        ) -> Result<Held, Fault> {
            self.roles.push(role);
            self.contents
                .push(vec![0; elems * storage.element().byte_width()]);
            Ok(Held {
                index: self.contents.len() - 1,
                freed: Arc::clone(&self.freed),
            })
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), Fault> {
            Ok(())
        }
        fn begin(&mut self, _submission: Submission) -> Result<(), Fault> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), Fault> {
            Ok(())
        }
        fn copy(&mut self, _src: &Held, _dst: &Held) -> Result<(), Fault> {
            Ok(())
        }
        fn finish(&mut self) -> Result<Option<()>, Fault> {
            Ok(Some(()))
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Fault> {
            if self.fail_replay.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Fault);
            }
            Ok(())
        }
        fn write(&mut self, dst: &Held, bytes: &[u8]) -> Result<(), Fault> {
            self.contents[dst.index] = bytes.to_vec();
            Ok(())
        }
        fn read(&mut self, src: &Held, out: &mut [u8]) -> Result<(), Fault> {
            out.copy_from_slice(&self.contents[src.index][..out.len()]);
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), Fault> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), Fault> {
            Ok(())
        }
    }

    /// A graph whose output is the F32 `[N, K]` const `name` itself, so a step's output reads that
    /// const's resident upload back.
    fn identity_of(name: &str) -> Graph {
        let b = Builder::new();
        let w = b.constant(name, TensorType::f32(vec![N, K]));
        b.finish(w)
    }

    fn compile_resident(engine: &Engine<Resident>, g: &Graph) -> StagedProgram<ValidationOutputs> {
        compile_staged(
            &g.clone().with_validations(Vec::new()),
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .expect("the fixture compiles")
    }

    fn bytes_of(entry: &WeightEntry) -> Vec<u8> {
        let WeightEntry::Dense(dense) = entry else {
            unreachable!()
        };
        dense.bytes().as_slice().to_vec()
    }

    /// The base store every adapter shares, and one adapter store: the base entry shared plus its own
    /// `lora` entry (the same name and shape in every adapter, different bytes per `seed`).
    fn base_store() -> Arc<WeightStore> {
        store(vec![("base", f32_entry(vec![N, K], 1))])
    }

    fn adapter_store(base: &WeightStore, seed: u32) -> (Arc<WeightStore>, Vec<u8>) {
        let lora = f32_entry(vec![N, K], seed);
        let bytes = bytes_of(&lora);
        let mut builder = WeightStore::builder();
        builder.extend_from(base).unwrap();
        builder.insert("lora", lora).unwrap();
        (Arc::new(builder.build()), bytes)
    }

    /// A step's output: the const the entry's graph reads back.
    fn output(engine: &mut Engine<Resident>, exe: ExecutableId, entry: EntryId) -> Vec<u8> {
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap()
            .read()
            .unwrap()
    }

    /// One adapter executable over `store`: an entry reading `lora` and an entry reading `base`.
    fn adapter(
        engine: &mut Engine<Resident>,
        store: Arc<WeightStore>,
    ) -> (ExecutableId, EntryId, EntryId) {
        let lora = compile_resident(engine, &identity_of("lora"));
        let base = compile_resident(engine, &identity_of("base"));
        let exe = engine
            .load_weights(store, WeightSource::ConstNames)
            .unwrap();
        let lora = engine.add_entry(exe, &lora).unwrap();
        let base = engine.add_entry(exe, &base).unwrap();
        (exe, lora, base)
    }

    /// SC-007: adapter A binds, then adapter B with the same names and shapes and other bytes, then a
    /// retained A entry replays. Each matches its own bytes, B never reads A's upload, and binding B
    /// uploads only B's own weight (the shared base is not uploaded again).
    ///
    /// Mutation: in `StoreGeneration::next`, return one generation for every payload (omit the
    /// generation from residency identity); B's `lora` binds A's upload and this row goes red.
    #[test]
    fn a_second_adapter_with_the_same_names_never_reads_the_first_adapters_bytes() {
        let mut engine = Engine::new(Resident::default());
        let base = base_store();
        let base_bytes = bytes_of(base.get("base").unwrap());
        let (store_a, lora_a) = adapter_store(&base, 2);
        let (store_b, lora_b) = adapter_store(&base, 3);
        let (a, a_lora, a_base) = adapter(&mut engine, store_a);
        let after_a = weight_uploads(&engine).len();
        assert_eq!(after_a, 2, "A uploads the base and its lora");
        let (b, b_lora, b_base) = adapter(&mut engine, store_b);
        assert_eq!(
            weight_uploads(&engine).len() - after_a,
            1,
            "binding B uploads only B's lora, not the shared base"
        );

        assert_eq!(
            output(&mut engine, b, b_lora),
            lora_b,
            "B reads its own bytes"
        );
        assert_eq!(output(&mut engine, b, b_base), base_bytes);
        assert_eq!(
            output(&mut engine, a, a_lora),
            lora_a,
            "the retained A entry replays its own bytes after B bound"
        );
        assert_eq!(output(&mut engine, a, a_base), base_bytes);
    }

    /// The allocation indices of every weight upload, in order.
    fn weight_uploads(engine: &Engine<Resident>) -> Vec<usize> {
        let roles = &engine.device().roles;
        (0..roles.len())
            .filter(|&i| roles[i] == BufferRole::Weight)
            .collect()
    }

    /// The weight uploads released so far (activation and state buffers are not counted).
    fn freed(engine: &Engine<Resident>) -> Vec<usize> {
        let roles = &engine.device().roles;
        let mut freed = engine.device().freed.lock().unwrap().clone();
        freed.retain(|&i| roles[i] == BufferRole::Weight);
        freed.sort_unstable();
        freed
    }

    /// SC-008: two adapters live at once, each with two entries, over a shared base. The same-named
    /// `lora` payloads are two generations and stay distinct; the base counts once; two row views of
    /// one fused entry are two uploads of their own rows. Removing an entry releases no upload, and an
    /// upload is released only when its last executable unloads (base after both adapters).
    ///
    /// Mutation: in `Engine::weight_buffer`, leak one extra `Arc` of each new upload
    /// (`std::mem::forget(Arc::clone(&resident))`, a residency that outlives its executables); no
    /// upload is ever released and the release assertions go red.
    #[test]
    fn shared_base_uploads_count_once_and_each_generation_stays_distinct_and_live() {
        let mut engine = Engine::new(Resident::default());
        let base = base_store();
        let (store_a, lora_a) = adapter_store(&base, 2);
        let (store_b, lora_b) = adapter_store(&base, 3);
        let (a, a_lora, _) = adapter(&mut engine, store_a);
        let (b, b_lora, b_base) = adapter(&mut engine, store_b);
        let uploads = weight_uploads(&engine);
        assert_eq!(uploads.len(), 3, "one base shared, two distinct loras");

        // Interleaved steps across both retained executables.
        for _ in 0..2 {
            assert_eq!(output(&mut engine, a, a_lora), lora_a);
            assert_eq!(output(&mut engine, b, b_lora), lora_b);
        }

        // Removing an entry releases no upload: the executable still holds its weights, and a
        // pending recording of the entry may still read them.
        engine.remove_entry(b, b_base).unwrap();
        assert_eq!(freed(&engine), Vec::<usize>::new());

        // Uploads are in bind order: A's lora, the shared base, B's lora. A's unload releases A's
        // lora only: the base is still B's.
        engine.unload(a).unwrap();
        let a_lora_upload = uploads[0];
        assert_eq!(freed(&engine), vec![a_lora_upload]);
        assert_eq!(output(&mut engine, b, b_lora), lora_b);

        // B's unload releases the rest.
        engine.unload(b).unwrap();
        assert_eq!(freed(&engine), uploads);
    }

    /// SC-008 (views): two ids viewing different row runs of one fused payload are two uploads, each
    /// holding exactly its own rows, and a second executable over the same store shares both.
    ///
    /// Mutation: in `Engine::weight_buffer`, build `Residency::runs` with an empty byte range for
    /// every run; both views collapse to the first upload and this row goes red.
    #[test]
    fn distinct_row_views_of_one_payload_are_distinct_uploads_shared_across_executables() {
        let fused = f32_entry(vec![2 * N, K], 4);
        let bytes = bytes_of(&fused);
        let rows = |r: Range<usize>| bytes[r.start * K * 4..r.end * K * 4].to_vec();
        let weights = store(vec![("qkv", fused)]);
        let mapped = map(
            &weights,
            vec![
                (
                    id(Q),
                    WeightView::RowRange {
                        key: "qkv".into(),
                        rows: 0..N,
                    },
                ),
                (
                    id(KR),
                    WeightView::RowRange {
                        key: "qkv".into(),
                        rows: N..2 * N,
                    },
                ),
            ],
        );
        let mut engine = Engine::new(Resident::default());
        let bind = |engine: &mut Engine<Resident>| {
            let exe = engine
                .load_weights(Arc::clone(&weights), WeightSource::Map(Arc::clone(&mapped)))
                .unwrap();
            let entries: Vec<EntryId> = [Q, KR]
                .into_iter()
                .map(|role| {
                    let program = compile_resident(engine, &identity_of(&id(role).const_name()));
                    engine.add_entry(exe, &program).unwrap()
                })
                .collect();
            (exe, entries)
        };
        let (first, entries) = bind(&mut engine);
        assert_eq!(weight_uploads(&engine).len(), 2, "one upload per row view");
        assert_eq!(output(&mut engine, first, entries[0]), rows(0..N));
        assert_eq!(output(&mut engine, first, entries[1]), rows(N..2 * N));
        let (second, entries) = bind(&mut engine);
        assert_eq!(
            weight_uploads(&engine).len(),
            2,
            "a second executable over the same payload shares both"
        );
        assert_eq!(output(&mut engine, second, entries[1]), rows(N..2 * N));
    }

    /// SC-009: two retained entries on distinct adapter generations over a shared base. Replacing
    /// one (unload, then a new adapter of the same names and shapes) changes neither the other's
    /// output nor its resources: nothing the retained adapter holds is freed, and the replacement
    /// reads its own bytes.
    ///
    /// Mutation: in `StoreGeneration::next`, return one generation for every payload (residency by
    /// logical name) and, on a residency hit in `Engine::weight_buffer`, overwrite the resident
    /// bytes with the new binding's; the replacement and the retained entry no longer read their
    /// own bytes and this row goes red.
    #[test]
    fn replacing_one_adapter_changes_neither_the_retained_entry_nor_its_resources() {
        let mut engine = Engine::new(Resident::default());
        let base = base_store();
        let base_bytes = bytes_of(base.get("base").unwrap());
        let (store_a, lora_a) = adapter_store(&base, 2);
        let (store_b, lora_b) = adapter_store(&base, 3);
        let (a, _, _) = adapter(&mut engine, store_a);
        let (b, b_lora, b_base) = adapter(&mut engine, store_b);
        let uploads = weight_uploads(&engine);
        assert_eq!(uploads.len(), 3);
        assert_ne!(lora_a, lora_b);

        engine.unload(a).unwrap();
        let (store_c, lora_c) = adapter_store(&base, 5);
        let (c, c_lora, c_base) = adapter(&mut engine, store_c);

        assert_eq!(
            output(&mut engine, c, c_lora),
            lora_c,
            "the replacement reads its own bytes"
        );
        assert_eq!(output(&mut engine, c, c_base), base_bytes);
        assert_eq!(
            output(&mut engine, b, b_lora),
            lora_b,
            "the retained entry is unchanged"
        );
        assert_eq!(output(&mut engine, b, b_base), base_bytes);
        let freed = freed(&engine);
        assert!(
            !freed.contains(&uploads[1]) && !freed.contains(&uploads[2]),
            "B's uploads (base {}, lora {}) stay live; freed {freed:?}",
            uploads[1],
            uploads[2]
        );
    }

    /// SC-008 (pending completion): an executable whose step failed holds work the device has not
    /// proven complete, so `unload` defers it. Its uploads are not freed at `unload`, only when a
    /// later successful `synchronize` drains the pending release; a base shared with a live
    /// executable is never freed, and the drained upload's residency entry is pruned.
    ///
    /// Mutation: in `Executor::unload`, drop the executable instead of
    /// `self.pending_release.push(loaded)`; A's upload is freed at `unload` and this row goes red.
    /// Second mutation: drop the `self.residency.retain` after `self.pending_release.clear()` in
    /// `step`; the drained upload's residency entry lingers and the last assertion goes red.
    #[test]
    fn an_unloaded_executable_with_pending_work_keeps_its_uploads_until_synchronize() {
        let mut engine = Engine::new(Resident::default());
        let base = base_store();
        let (store_a, lora_a) = adapter_store(&base, 2);
        let (store_b, lora_b) = adapter_store(&base, 3);
        let (a, a_lora, _) = adapter(&mut engine, store_a);
        let (b, b_lora, _) = adapter(&mut engine, store_b);
        let uploads = weight_uploads(&engine);
        assert_eq!(output(&mut engine, a, a_lora), lora_a);

        engine
            .device()
            .fail_replay
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(
            matches!(
                engine.step(a, a_lora, &StepInputs::new(), &mut NoSync),
                Err(ExecError::Device(_))
            ),
            "the injected replay fault fails the step and marks A for reset"
        );
        engine
            .device()
            .fail_replay
            .store(false, std::sync::atomic::Ordering::Relaxed);

        engine.unload(a).unwrap();
        assert_eq!(
            freed(&engine),
            Vec::<usize>::new(),
            "A's completion is not proven: nothing it holds is freed at unload"
        );
        assert_eq!(engine.residency.len(), 3);

        // B's successful step synchronizes the device: the pending release drains.
        assert_eq!(output(&mut engine, b, b_lora), lora_b);
        assert_eq!(
            freed(&engine),
            vec![uploads[0]],
            "A's lora is freed once synchronize proves completion; the base stays B's"
        );
        assert_eq!(
            engine.residency.len(),
            2,
            "the drained upload's residency entry is pruned"
        );
    }
}
