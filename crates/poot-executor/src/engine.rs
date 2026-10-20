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

use poot_runtime_common::{BufferRole, DeviceIdentity, VpuBinding, VpuGraphArtifact, VpuLayer};
use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::Range;
use std::sync::{Arc, Weak};
use std::time::Instant;

use poot_graph_ir::{
    BoundaryId, Eqn, Slot, SlotKey, StateRole, Storage, TensorType, ValueId, ValueMeta, ValueTable,
};
use poot_graph_plan::multi_device::PlanNodeId;
use poot_graph_plan::{
    Plan, PoolSite, Program, Region, Sink, Source, Target, dispatch_work,
    multi_device::topology::DeviceId, staged::Link, staged::StageId, staged::StagedProgram,
    staged::SyncPoint, staged::TargetSet, staged::Transfer, validate_linked_artifact, value_ids,
};
use poot_kernel_ir::Body;
use poot_quant::SourceRole;
use poot_quant::weights::{StoreGeneration, WeightMap, WeightStore};
use poot_target::{AieCaps, AieTargetId, Backend, BufferStorage, TargetId, VpuTargetId};
use poot_tensor::DType;

use crate::binder::{self, BoundSlot, EmbedRows, StoredAs, WeightBinder};
use crate::device::{
    AieInvocation, Arg, Device, DeviceTime, Dispatch, LaunchKind, VpuInvocation, WeightUpload,
};
use crate::error::{BindError, ChainError, DeviceError, DispatchError, ExecError, LoadError};
use crate::shared_kernels::{SharedKernels, SharedKernelsMismatch};
use crate::timing::TimingOptions;
use crate::{
    ChainFeed, ChainOutputs, ChainStep, EntryId, ExecutableId, Executor, ExecutorStats, HostSync,
    LateInputs, NoLate, OutputSource, RowFill, SiteIdentity, StageStats, StateScope, StepInputs,
    StepOutputs, SyncReads, SyncWrites, TableIdentity,
};

/// The one executor over every backend: it owns one [`Device`], the executables loaded onto it, and
/// the kernel, residency and timing state they share. It implements the object-safe [`Executor`]
/// contract; everything backend-specific stays behind `D`.
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
    kernel_index: HashMap<poot_codegen::KernelId, usize>,
    /// The AIE regions this engine loaded: by the region's link recipe and the fingerprint of the body on
    /// every tile (one body can sit in more than one region, and one region shape can run other bodies),
    /// with the loaded index and the invocation its dispatches carry. Empty for a GPU device.
    aie_kernels: HashMap<String, (usize, AieInvocation)>,
    /// The Intel NPU graph this engine loaded, by the digest of its artifact.
    vpu_kernels: HashMap<u64, (usize, VpuInvocation)>,
    kernel_cache: Option<poot_codegen::KernelCache>,
    /// The external tools an AIE kernel is linked with. A GPU device's kernel cache resolves its own.
    toolchain: poot_codegen::Toolchain,
    /// Artifacts shared with the other engines of one planning pass; `None` for an executing engine,
    /// which keeps no artifact bytes beyond the kernel it loads.
    shared_kernels: Option<SharedKernels>,
    /// Kernels this engine resolved through `kernel_cache` ([`ExecutorStats::kernels_resolved`]).
    kernels_resolved: u64,
    /// Every live weight upload, by what it uploads. Weak: an upload lives exactly as long as some
    /// executable (or a pending release of one) holds it, so a replaced adapter's weights are freed
    /// once nothing references them while a base shared with another executable stays.
    residency: HashMap<Residency, Weak<ResidentWeight<D>>>,
    /// Bytes read from the weight store for uploads ([`ExecutorStats::weight_bytes_read`]).
    weight_bytes_read: u64,
    /// Bytes the host wrote to device buffers while binding steps ([`ExecutorStats::host_write_bytes`]).
    host_write_bytes: u64,
    /// The host-sync counters ([`ExecutorStats::syncs`] and the three beside it).
    sync_counters: SyncCounters,
    /// Bytes of step output read back ([`ExecutorStats::output_read_bytes`]).
    output_read_bytes: u64,
    /// Times the host blocked on the device ([`ExecutorStats::waits`]).
    waits: u64,
    device: D,
    /// The device this engine drives: `add_entry` refuses a stage placed on another
    /// ([`ExecError::Placement`]); the stages of one program all run on it, in order. A standalone
    /// engine is device 0; a coordinator opens one engine per device of its target set.
    id: DeviceId,
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
    /// An executable-resident pool table or map ([`Loaded::resident`]), written by a host sync.
    Resident(usize),
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

/// The device copy that carries the token one chained step picked into the next step's token slot,
/// recorded once per pair of entries.
struct ChainLink<D: Device> {
    from: EntryId,
    to: EntryId,
    recording: D::Recording,
}

struct Loaded<D: Device> {
    // Field order is drop order: the chain links and the entries, whose recordings address the weights,
    // state and resident buffers below, drop first. A recording owns what it addresses
    // ([`Device::Recording`]), so this order is what a device that records raw addresses relies on, not a
    // second owner.
    links: Vec<ChainLink<D>>,
    entries: Vec<Option<Entry<D>>>,
    binder: WeightBinder,
    /// This executable's reference to each upload it binds, indexed by [`Loc::Weight`]: the
    /// uploads outlive every recording and pending release of this executable.
    weights: Vec<Arc<ResidentWeight<D>>>,
    state: Vec<StateBuffer<D>>,
    /// Pool tables and maps, one per canonical site table ([`ResidentKey`]), shared by every entry
    /// that pages the same stored bytes and filled by host syncs, never by a step's inputs.
    resident: Vec<ResidentBuf<D>>,
    /// Set after a state-mutating failure; the next `step` refuses until `reset_state(All)` clears
    /// it (core review CR30).
    needs_reset: bool,
}

/// The sync counters behind [`ExecutorStats::syncs`] and its neighbours.
#[derive(Clone, Copy, Debug, Default)]
struct SyncCounters {
    syncs: u64,
    read_bytes: u64,
    rows: u64,
    refill_bytes: u64,
}

/// What one resident buffer is, for finding it again from another entry: a table by the stored bytes
/// it pages and its lane and height, a map by the identity of the site it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ResidentKey {
    Table(TableIdentity),
    Map(SiteIdentity),
}

enum ResidentKind {
    /// `pool_rows` rows of `row_bytes` each, refilled from `writer`'s stored experts.
    Table {
        name: String,
        experts: usize,
        pool_rows: usize,
        row_bytes: usize,
        /// `None` on a device that only sizes weights.
        writer: Option<binder::WeightWriter>,
    },
    /// The site's `[E]` map in its planned lane.
    Map {
        key: SlotKey,
        experts: usize,
        storage: BufferStorage,
    },
}

struct ResidentBuf<D: Device> {
    key: ResidentKey,
    kind: ResidentKind,
    buffer: D::Buffer,
    /// The buffer's native device-element count a dispatch reading it publishes.
    elems: u32,
    /// The allocation's bytes: what a complete reset zeroes.
    bytes: usize,
}

struct StateBuffer<D: Device> {
    name: String,
    aval: TensorType,
    /// The role the first graph to declare this state traced it with: whether the leading axis is
    /// a row axis [`StateScope::Row`] addresses.
    role: StateRole,
    storage: BufferStorage,
    bytes: usize,
    /// This state buffer's native device-element count (Card 547b): never arena-shared, so always
    /// the exact length a dispatch reading it should publish.
    elems: u32,
    buffer: D::Buffer,
}

/// An entry while its stages load: the buffers they share and what each has exported so far.
struct EntryBuild<D: Device> {
    locals: Vec<D::Buffer>,
    arena: Vec<D::Buffer>,
    slots: Vec<BoundSlot>,
    token_embeds: Vec<TokenEmbedSlot>,
    /// Each exported boundary's location and element count, by the stage that produced it.
    exported: HashMap<BoundaryId, (Loc, u32)>,
    /// The lane each exported boundary was planned in.
    export_lanes: HashMap<BoundaryId, BufferStorage>,
    stages: Vec<StageRun<D>>,
    validations: Vec<ValidationLoad>,
    /// The latest loaded stage's output: read location, bytes and host form.
    output: Option<OutputLoad>,
    /// Each pooled table's and map's index in [`Loaded::resident`], by the slot key a stage declares.
    pooled: HashMap<SlotKey, usize>,
    /// The buffer each transfer into one of the part's stages delivers to.
    imports: Vec<ImportBuffer>,
    /// Each transfer out of one of the part's stages: where the value it carries lives.
    outgoing: Vec<OutgoingLoad>,
}

/// The device buffer a transfer's row writes: a local buffer of the entry, sized to the boundary's
/// planned storage, that the consuming stage reads in place of the producer's buffer.
struct ImportBuffer {
    row: PlanNodeId,
    buffer: usize,
}

/// A value a transfer carries away: the producing stage (an index into the entry's stages), where the
/// value lives, and the bytes the row moves.
struct OutgoingLoad {
    row: PlanNodeId,
    stage: usize,
    loc: Loc,
    bytes: usize,
}

/// A stage's primary output as the host reads it.
struct OutputLoad {
    loc: Loc,
    bytes: usize,
    host: Option<(DType, Vec<usize>)>,
}

struct Entry<D: Device> {
    // Field order is drop order: the recordings drop before the buffers they reference.
    stages: Vec<StageRun<D>>,
    /// Steps replayed (one per `step`, whatever the stage count).
    replays: u64,
    /// Buffers shared by every stage of the entry: a stage binds a boundary to the producing stage's
    /// buffer by index, so no link copies. Index-aligned with [`Loc::Local`].
    locals: Vec<D::Buffer>,
    /// One buffer per [`poot_graph_plan::BufferPlan`] arena slot of each stage (Card 547b), allocated
    /// once up front and index-aligned with [`Loc::Arena`] (a stage's slots follow the previous
    /// stage's).
    arena: Vec<D::Buffer>,
    slots: Vec<BoundSlot>,
    /// Hosted embed gathers (`legalize`'s `Slot::TokenEmbed`), filled each step from the step's
    /// `Slot::Token` input and the executable's embed rows, never supplied by the caller.
    token_embeds: Vec<TokenEmbedSlot>,
    /// Where the final stage's primary output is read from: its pre-commit snapshot when the output
    /// aliases a commit destination, else the output itself.
    output: Loc,
    output_bytes: usize,
    /// The primary output as a host tensor: its dtype lane (`None` when the planned lane is not a
    /// dense natural-width one, e.g. a packed or mirrored output) and declared shape.
    output_host: Option<(DType, Vec<usize>)>,
    /// One per stage, checked after the step's single completion wait.
    validations: Vec<ValidationLoad>,
    /// The host syncs of a step, in stage order (empty unless experts are pooled).
    syncs: Vec<EntrySync>,
    /// The index of the stage after which a step binds its late inputs, `None` to bind them first.
    late_after: Option<usize>,
    /// Every value the entry exports, with the lane it was planned in.
    exports: Vec<ExportBuf>,
    imports: Vec<ImportBuffer>,
    outgoing: Vec<OutgoingLoad>,
}

/// A value an entry exports: where it lives and the lane it was planned in.
struct ExportBuf {
    id: BoundaryId,
    loc: Loc,
    elems: u32,
    storage: BufferStorage,
}

/// One host sync of an entry, resolved to the executable's resident buffers.
struct EntrySync {
    /// The stage that must complete before the host is called.
    after: usize,
    point: SyncPoint,
    identity: SiteIdentity,
    /// Device bytes of one row of each of the site's tables.
    row_bytes: Vec<usize>,
    /// Each value the point reads: its id, where it lives and its element count.
    reads: Vec<(BoundaryId, Loc, u32)>,
    /// The site's tables and map in [`Loaded::resident`], in [`PoolSite::weights`] order.
    tables: Vec<usize>,
    map: usize,
}

/// The stages of one entry that one engine loads, and how their boundaries reach them: the whole
/// program for a standalone engine, one device's share of it for a coordinator's engine. A boundary read
/// from a stage of the part is bound through a link (a shared buffer) or a transfer (a buffer the
/// engine owns and a peer's row writes); the part names every link and transfer of the program, and the
/// engine uses those that touch its stages.
pub(crate) struct EntryPart<'a> {
    pub(crate) stages: Vec<(StageId, &'a Program)>,
    pub(crate) links: &'a [Link],
    pub(crate) transfers: &'a [Transfer],
    pub(crate) sync_points: &'a [SyncPoint],
    /// The stage after which a step binds its late inputs ([`StagedProgram::late_after`]).
    pub(crate) late: Option<StageId>,
}

impl<'a> EntryPart<'a> {
    /// Every stage of `program`.
    fn whole(program: &'a StagedProgram) -> Self {
        Self {
            stages: program
                .stages()
                .map(|(stage, _, program)| (stage, program))
                .collect(),
            links: program.links(),
            transfers: program.transfers(),
            sync_points: program.sync_points(),
            late: program.late_after(),
        }
    }
}

/// One stage of an entry: its own recording over the entry's shared buffers.
struct StageRun<D: Device> {
    id: StageId,
    recording: Option<D::Recording>,
    recordings: u64,
    replays: u64,
    steps: Vec<LoadedStep>,
    /// Explicit two-phase commit copies (src, dst), run after every step's dispatches.
    commits: Vec<(Loc, Loc)>,
    /// This stage's primary output and, when it aliases a destination the commit copies overwrite,
    /// the snapshot taken before `commits`.
    output: Loc,
    preserve_output: Option<Loc>,
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
    launch: LaunchKind,
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

/// The cache of compiled kernel bodies, for a target whose kernels are bodies.
fn kernel_cache_of(
    cache: &Option<poot_codegen::KernelCache>,
    target: TargetId,
) -> Result<&poot_codegen::KernelCache, LoadError> {
    cache.as_ref().ok_or(LoadError::NoKernelBodies { target })
}

fn target_of<D: Device>(device: &D) -> TargetId {
    device.target().id()
}

pub(crate) fn device_error<E: std::error::Error + Send + Sync + 'static>(
    target: TargetId,
) -> impl Fn(E) -> ExecError {
    move |source| {
        ExecError::from(DeviceError {
            backend: match target {
                TargetId::Gpu(Backend::SpirvVulkan) => "wgpu",
                TargetId::Gpu(Backend::AmdGcn(_)) => "rocm",
                TargetId::Gpu(Backend::Nvptx) => "ptx",
                TargetId::Aie(_) => "xdna",
                TargetId::Vpu(_) => "intel-npu",
            },
            source: Box::new(source),
        })
    }
}

/// The one length check every backend shares: no argument of a dispatch may assert more native elements than
/// the buffer bound to it holds ([`Device::buffer_elems`]), because a kernel trusts the length it is
/// published. Arguments are numbered as the device sees them, the inputs then the output.
fn check_arg_lengths<D: Device>(
    device: &D,
    inputs: &[Arg<'_, D>],
    output: &Arg<'_, D>,
) -> Result<(), DispatchError> {
    for (arg, bound) in inputs.iter().chain(std::iter::once(output)).enumerate() {
        let capacity = device.buffer_elems(bound.buffer);
        if bound.elems as usize > capacity {
            return Err(DispatchError::ArgExceedsBuffer {
                arg,
                elems: bound.elems,
                capacity,
            });
        }
    }
    Ok(())
}

/// The one way a step puts host bytes on the device: every write of a step's bind goes through here,
/// so [`ExecutorStats::host_write_bytes`] counts exactly what the device was handed.
fn host_write<D: Device>(
    device: &mut D,
    counter: &mut u64,
    dst: &D::Buffer,
    bytes: &[u8],
) -> Result<(), D::Error> {
    crate::device::write_slice(device, dst, 0, bytes)?;
    *counter += bytes.len() as u64;
    Ok(())
}

/// A weight upload's [`EncodeError`] as the load error that names the weight.
fn encode_error(
    error: crate::lanes::EncodeError,
    bytes: &binder::ConstBytes,
    name: &str,
    storage: BufferStorage,
) -> ExecError {
    match error {
        crate::lanes::EncodeError::NoEncoding => LoadError::WeightFormat {
            name: name.to_string(),
            stored: match bytes.stored {
                StoredAs::Dense(dtype) => format!("{dtype:?}"),
                StoredAs::PackedSource | StoredAs::PackedStackSource { .. } => {
                    unreachable!("a packed source always has an encoding")
                }
            },
            planned: storage,
        },
        crate::lanes::EncodeError::NonFiniteF16 { index, bits } => LoadError::NonFiniteF16Weight {
            name: name.to_string(),
            index,
            bits,
        },
    }
    .into()
}

/// The buffers a [`Loc`] indexes besides the entry's own: the executable's weights, carried state
/// and resident pool buffers.
struct Held<'a, D: Device> {
    weights: &'a [Arc<ResidentWeight<D>>],
    state: &'a [StateBuffer<D>],
    resident: &'a [ResidentBuf<D>],
}

impl<'a, D: Device> Held<'a, D> {
    fn of(loaded: &'a Loaded<D>) -> Self {
        Self {
            weights: &loaded.weights,
            state: &loaded.state,
            resident: &loaded.resident,
        }
    }

    /// Only an entry's own buffers (a stage being loaded has no executable buffers in hand).
    fn none() -> Self {
        Self {
            weights: &[],
            state: &[],
            resident: &[],
        }
    }
}

/// A value a stage hands to a transfer: the buffer it lives in (never the whole of an arena slot's
/// capacity: `bytes` is the row's own length) and the row that carries it.
pub(crate) struct Outgoing<'a, D: Device> {
    pub(crate) row: PlanNodeId,
    pub(crate) buffer: &'a D::Buffer,
    pub(crate) bytes: usize,
}

/// What a coordinator records after a stage's own walk, inside the stage's one recording transaction:
/// the copies of the stage's exported values into the consuming devices' import buffers. There is one
/// recording model: the engine's own [`StageSeams`] is [`NoSeams`], the same transaction with nothing
/// after the walk.
pub(crate) trait StageSeams<D: Device> {
    fn after_walk(&mut self, device: &mut D, outgoing: &[Outgoing<'_, D>]) -> Result<(), D::Error>;
}

/// A stage that sends nothing.
pub(crate) struct NoSeams;

impl<D: Device> StageSeams<D> for NoSeams {
    fn after_walk(
        &mut self,
        _device: &mut D,
        _outgoing: &[Outgoing<'_, D>],
    ) -> Result<(), D::Error> {
        Ok(())
    }
}

fn resolve<'a, D: Device>(
    loc: Loc,
    locals: &'a [D::Buffer],
    arena: &'a [D::Buffer],
    held: &Held<'a, D>,
) -> &'a D::Buffer {
    match loc {
        Loc::Local(i) => &locals[i],
        Loc::Arena(i) => &arena[i],
        Loc::Weight(i) => &held.weights[i].buffer,
        Loc::State(i) => &held.state[i].buffer,
        Loc::Resident(i) => &held.resident[i].buffer,
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
        let target = device.target().id();
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
            aie_kernels: HashMap::new(),
            vpu_kernels: HashMap::new(),
            toolchain: poot_codegen::Toolchain::from_env(),
            // A target whose artifacts are not compiled kernel bodies (an Intel NPU's vendor graphs) has no
            // kernel cache; a per-equation kernel asked of such an engine is a typed `LoadError`.
            kernel_cache: poot_codegen::Target::try_from(target)
                .ok()
                .map(poot_codegen::KernelCache::open),
            shared_kernels: None,
            kernels_resolved: 0,
            residency: HashMap::new(),
            weight_bytes_read: 0,
            host_write_bytes: 0,
            sync_counters: SyncCounters::default(),
            output_read_bytes: 0,
            waits: 0,
            device,
            id: DeviceId(0),
            poisoned: false,
            timing: snapshot,
            timing_options: timing,
            detail_sample_counter: 0,
        }
    }

    /// This engine resolves kernels through `kernels`, shared with the other engines built over it: a
    /// kernel any of them already resolved is reused, not resolved again. For a planning pass, whose
    /// candidates each own an engine over mostly the same kernels.
    ///
    /// # Errors
    ///
    /// [`SharedKernelsMismatch`] when `kernels` is bound to another backend than this device's.
    pub fn with_shared_kernels(
        mut self,
        kernels: SharedKernels,
    ) -> Result<Self, SharedKernelsMismatch> {
        let engine = self.target_id();
        // The table holds byte artifacts; an AIE kernel is a linked artifact that never enters it.
        if let TargetId::Aie(_) = engine {
            return Err(SharedKernelsMismatch::LinkedArtifacts { engine });
        }
        if kernels.target() != engine {
            return Err(SharedKernelsMismatch::Target {
                table: kernels.target(),
                engine,
            });
        }
        self.shared_kernels = Some(kernels);
        Ok(self)
    }

    /// The device this engine drives, for a caller that needs the backend's own handle or counters.
    pub fn device(&self) -> &D {
        &self.device
    }

    /// This engine as the driver of device `id`, for a coordinator that opens one engine per device.
    pub(crate) fn on_device(mut self, id: DeviceId) -> Self {
        self.id = id;
        self
    }

    pub(crate) fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    /// This engine's configured timing detail (Card 552).
    pub fn timing_options(&self) -> &TimingOptions {
        &self.timing_options
    }

    fn target_id(&self) -> TargetId {
        self.device.target().id()
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

    /// Kernel index for `body`, compiling it for GPU `backend` on first use (Card 656: the full
    /// verified `Body` is the cache's identity, not a display key - `kernel_index` itself keys on
    /// `KernelCache::key(body)`'s content fingerprint, so two equations sharing a planner `key`
    /// string but compiling to different bodies never collide here, before `kernel_cache`'s own
    /// content-keyed lookup ever runs). `key` is used only as the device's debug label.
    fn kernel(
        &mut self,
        backend: Backend,
        key: &str,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<usize, ExecError> {
        let cache = kernel_cache_of(&self.kernel_cache, target_of(&self.device))?;
        let id = cache.key(body).id();
        if let Some(&index) = self.kernel_index.get(&id) {
            return Ok(index);
        }
        let resolve = || cache.load_or_compile(body, max_artifact_bytes);
        let codegen_error = |source| LoadError::Codegen {
            key: key.to_string(),
            source,
        };
        let artifact = match &self.shared_kernels {
            Some(shared) => {
                let (artifact, resolved) = shared
                    .get_or_resolve(id, max_artifact_bytes, resolve)
                    .map_err(codegen_error)?;
                self.kernels_resolved += u64::from(resolved);
                artifact
            }
            None => {
                self.kernels_resolved += 1;
                resolve().map_err(codegen_error)?
            }
        };
        let compiled = poot_codegen::kernel_handle(body, &artifact).map_err(|source| {
            LoadError::KernelArtifact {
                key: key.to_string(),
                source,
            }
        })?;
        let kernel = self
            .device
            .load_kernel(key, compiled)
            .map_err(device_error(backend.into()))?;
        self.kernels.push(kernel);
        let index = self.kernels.len() - 1;
        self.kernel_index.insert(id, index);
        Ok(index)
    }

    /// Kernel index and invocation for the AIE `region` of `program`, linked for `target` on first use: the
    /// link recipe is read from the program's own plans (`compile` already admitted it against the target's
    /// limits when it packaged the region), [`poot_codegen::compile_aie`] links the tile bodies against it
    /// into one artifact, the linked artifact is checked against `caps` before it is wrapped (what only a
    /// linked artifact can answer), and only then does the device load it. Every refusal is a typed
    /// [`LoadError`] naming the region; nothing is loaded on any of them.
    fn aie_region_kernel(
        &mut self,
        target: AieTargetId,
        caps: &AieCaps,
        program: &Program,
        region: &Region,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<(usize, AieInvocation), ExecError> {
        let members = region.members();
        let key = format!("region[{}..{}]", members.start, members.end);
        let (recipe, bodies) = program
            .region_link(region)
            .map_err(|source| LoadError::AieAbi {
                key: key.clone(),
                source,
            })?;
        // The recipe holds the wiring and shapes, the bodies hold the arithmetic: both name the region.
        let identity = {
            let mut text = recipe.canonical();
            let cache = kernel_cache_of(&self.kernel_cache, TargetId::Aie(target))?;
            for body in &bodies {
                text.push_str(&format!("{}\n", cache.key(body).id()));
            }
            text
        };
        if let Some(&(index, invocation)) = self.aie_kernels.get(&identity) {
            return Ok((index, invocation));
        }
        let compiled = poot_codegen::compile_aie(
            &bodies,
            &recipe,
            &self.toolchain,
            kernel_cache_of(&self.kernel_cache, TargetId::Aie(target))?,
            max_artifact_bytes,
        )
        .map_err(|source| LoadError::AieCompile {
            key: key.clone(),
            source,
        })?;
        self.kernels_resolved += 1;
        validate_linked_artifact(&compiled.artifact, caps).map_err(|source| {
            LoadError::AieLimit {
                key: key.clone(),
                source,
            }
        })?;
        let stream_bytes = compiled.artifact.instructions().len();
        let words = u32::try_from(stream_bytes / 4)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| LoadError::AieInstructionStream {
                key: key.clone(),
                bytes: stream_bytes,
            })?;
        let handle = poot_codegen::aie_kernel_handle(compiled.artifact);
        let kernel = self
            .device
            .load_kernel(&key, handle)
            .map_err(device_error(target.into()))?;
        self.kernels.push(kernel);
        let index = self.kernels.len() - 1;
        let invocation = AieInvocation::new(target, words);
        self.aie_kernels.insert(identity, (index, invocation));
        Ok((index, invocation))
    }

    /// Kernel index and invocation for the Intel NPU graph of `program`, loaded on first use: the planner's
    /// region is serialized to OpenVINO IR ([`poot_codegen::emit_vpu`]) and handed to the device with the
    /// region's argument ABI (the program's own region reads, then writes, with the storage the buffer plan
    /// gave each value). The device compiles it with the vendor compiler and checks the compiled graph's
    /// arguments against that ABI. Every refusal is a typed [`LoadError`] naming the region.
    fn vpu_region_kernel(
        &mut self,
        target: VpuTargetId,
        program: &Program,
        region: &Region,
    ) -> Result<(usize, VpuInvocation), ExecError> {
        let members = region.members();
        let key = format!("region[{}..{}]", members.start, members.end);
        let graph = program.vpu_region().ok_or(LoadError::Unimplemented(
            "an Intel NPU program without its graph",
        ))?;
        let ir = poot_codegen::emit_vpu(graph).map_err(|source| LoadError::VpuEmit {
            key: key.clone(),
            source,
        })?;
        let binding = |name: String, value: ValueId, shape: Vec<usize>| {
            VpuBinding::new(
                name,
                program.storage().storage(value).buffer_storage(),
                shape,
            )
        };
        let inputs = region
            .reads()
            .iter()
            .zip(graph.input_names())
            .zip(graph.input_shapes())
            .map(|((&value, name), shape)| binding(name, value, shape))
            .collect();
        let outputs = region
            .writes()
            .iter()
            .zip(graph.output_names())
            .zip(graph.output_shapes())
            .map(|((&value, name), shape)| binding(name, value, shape))
            .collect();
        let layers = graph
            .layers()
            .map(|(name, origin)| VpuLayer::new(name, origin))
            .collect();
        let artifact = VpuGraphArtifact::new(
            target,
            ir.xml.into_bytes(),
            ir.weights,
            inputs,
            outputs,
            layers,
        )
        .map_err(|source| LoadError::VpuArtifact {
            key: key.clone(),
            source,
        })?;
        let digest = artifact.digest();
        if let Some(&(index, invocation)) = self.vpu_kernels.get(&digest) {
            return Ok((index, invocation));
        }
        let handle = poot_codegen::vpu_kernel_handle(artifact);
        let kernel = self
            .device
            .load_kernel(&key, handle)
            .map_err(device_error(target.into()))?;
        self.kernels_resolved += 1;
        self.kernels.push(kernel);
        let index = self.kernels.len() - 1;
        let invocation = VpuInvocation::new(target, digest);
        self.vpu_kernels.insert(digest, (index, invocation));
        Ok((index, invocation))
    }

    /// `shape` is a logical tensor shape (never already device-encoded - see
    /// `allocate_arena_slot` for a caller that already has one). Returns `(buffer, bytes, elems)`:
    /// `elems` is this buffer's native device-element count (Card 547b) - the same quantity
    /// `Device::allocate` itself was called with, handed back so a caller that also needs to publish
    /// it (as a dispatch argument's own length, never a reused arena slot's capacity) does not
    /// re-derive it a second time.
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        shape: &[usize],
    ) -> Result<(D::Buffer, usize, usize), ExecError> {
        let elems = storage
            .device_elems(shape)
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
        let target = self.target_id();
        self.device
            .allocate(role, storage, elems.max(1))
            .map_err(device_error(target))
    }

    /// Loads every stage of `program` in order into one entry. A stage's boundary inputs bind the
    /// buffers earlier stages exported, so the whole entry shares one `locals`/`arena` pair.
    fn load_entry(
        &mut self,
        exe: ExecutableId,
        part: &EntryPart<'_>,
    ) -> Result<Entry<D>, ExecError> {
        let mut build = EntryBuild {
            locals: Vec::new(),
            arena: Vec::new(),
            slots: Vec::new(),
            token_embeds: Vec::new(),
            exported: HashMap::new(),
            export_lanes: HashMap::new(),
            stages: Vec::new(),
            validations: Vec::new(),
            output: None,
            pooled: HashMap::new(),
            imports: Vec::new(),
            outgoing: Vec::new(),
        };
        self.load_pools(exe, part, &mut build)?;
        for &(id, stage) in &part.stages {
            self.load_stage(exe, id, stage, part, &mut build)?;
        }
        let syncs = self.entry_syncs(exe, part, &build)?;
        let EntryBuild {
            locals,
            arena,
            slots,
            token_embeds,
            exported,
            export_lanes,
            stages,
            validations,
            output,
            imports,
            outgoing,
            ..
        } = build;
        let exports = exported
            .into_iter()
            .map(|(id, (loc, elems))| ExportBuf {
                id,
                loc,
                elems,
                storage: export_lanes[&id],
            })
            .collect();
        let OutputLoad {
            loc: output,
            bytes: output_bytes,
            host: output_host,
        } = output.expect("a staged program has at least one stage");
        Ok(Entry {
            stages,
            replays: 0,
            locals,
            arena,
            slots,
            token_embeds,
            output,
            output_bytes,
            output_host,
            validations,
            syncs,
            late_after: part
                .late
                .and_then(|late| part.stages.iter().position(|&(id, _)| id == late)),
            exports,
            imports,
            outgoing,
        })
    }

    /// The executable's resident buffer for each table and map the program's sync points name, found
    /// by canonical identity or allocated (zeroed) on first use, recorded in `build.pooled` by the
    /// slot key a stage declares them under. A table or map no stage reads is refused: its lane is a
    /// stage's planning decision.
    fn load_pools(
        &mut self,
        exe: ExecutableId,
        part: &EntryPart<'_>,
        build: &mut EntryBuild<D>,
    ) -> Result<(), ExecError> {
        let declared = |key: &SlotKey| {
            part.stages.iter().find_map(|&(_, stage)| {
                let g = stage.graph();
                g.inputs()
                    .iter()
                    .copied()
                    .find(|&id| g.meta(id).slot_key() == Some(key))
                    .map(|id| (id, stage.storage().storage(id).buffer_storage()))
            })
        };
        for point in part.sync_points {
            let site = &point.site;
            let (_, map_storage) =
                declared(&site.map).ok_or_else(|| LoadError::PoolUndeclared {
                    key: site.map.clone(),
                })?;
            let mut tables = Vec::with_capacity(site.weights.len());
            for weight in &site.weights {
                let (value, storage) =
                    declared(&weight.slot).ok_or_else(|| LoadError::PoolUndeclared {
                        key: weight.slot.clone(),
                    })?;
                let index = self.pool_table(exe, value, weight, site.experts, storage)?;
                build.pooled.insert(weight.slot.clone(), index);
                tables.push(index);
            }
            let identity = self.site_identity(exe, &tables)?;
            let index = self.pool_map(exe, identity, site, map_storage)?;
            build.pooled.insert(site.map.clone(), index);
        }
        Ok(())
    }

    /// The canonical identity of the site whose tables sit at `tables` in [`Loaded::resident`].
    fn site_identity(
        &mut self,
        exe: ExecutableId,
        tables: &[usize],
    ) -> Result<SiteIdentity, ExecError> {
        let loaded = self.loaded(exe)?;
        let identities = tables
            .iter()
            .map(|&index| match &loaded.resident[index].key {
                ResidentKey::Table(identity) => identity.clone(),
                ResidentKey::Map(_) => unreachable!("a site's tables are tables"),
            })
            .collect();
        Ok(SiteIdentity::new(identities))
    }

    /// The resident `[P, ..]` buffer of pooled table `weight`, whose full `[E, ..]` stored bytes the
    /// binder resolves by the const's name. Zeroed; a host sync fills rows.
    fn pool_table(
        &mut self,
        exe: ExecutableId,
        value: ValueId,
        weight: &poot_graph_plan::PooledWeight,
        experts: usize,
        storage: BufferStorage,
    ) -> Result<usize, ExecError> {
        let pool_rows = weight.pool.shape[0];
        let full = TensorType::new(
            std::iter::once(experts)
                .chain(weight.pool.shape[1..].iter().copied())
                .collect::<Vec<usize>>(),
            weight.pool.dtype,
        );
        let bytes = self
            .loaded(exe)?
            .binder
            .const_bytes(value, &weight.name, &full, None)?;
        let identity = TableIdentity {
            runs: bytes
                .spans
                .iter()
                .map(|span| (span.generation, span.role, span.bytes.clone()))
                .collect(),
            storage,
            pool_rows,
        };
        let key = ResidentKey::Table(identity);
        if let Some(index) = self
            .loaded(exe)?
            .resident
            .iter()
            .position(|resident| resident.key == key)
        {
            return Ok(index);
        }
        let name = weight.name.clone();
        let packed = matches!(bytes.stored, StoredAs::PackedStackSource { .. });
        if let StoredAs::Dense(dtype) = bytes.stored {
            if dtype != weight.pool.dtype {
                return Err(LoadError::WeightDtype {
                    name,
                    stored: dtype,
                    declared: weight.pool.dtype,
                }
                .into());
            }
            match crate::lanes::encoding_of(dtype, storage) {
                None => {
                    return Err(LoadError::WeightFormat {
                        name,
                        stored: format!("{dtype:?}"),
                        planned: storage,
                    }
                    .into());
                }
                Some(crate::lanes::Encoding::PackedWordsFiniteF16) => {
                    return Err(LoadError::Unimplemented(
                        "a pooled F16 table in a packed lane (its finite check reads every expert)",
                    )
                    .into());
                }
                Some(_) => {}
            }
        }
        let target = self.target_id();
        let (buffer, row_bytes, elems, allocated) = if packed {
            let StoredAs::PackedStackSource { slab_bytes } = bytes.stored else {
                unreachable!("packed is a packed stack source")
            };
            // The same word layout a whole stack uploads in: `u32` words plus the guard word.
            let words = (pool_rows * slab_bytes).div_ceil(4) + 1;
            let buffer = self
                .device
                .allocate(BufferRole::Weight, BufferStorage::i32(), words)
                .map_err(device_error(target))?;
            (buffer, slab_bytes, checked_u32(words)?, words * 4)
        } else {
            let total = storage
                .device_elems(&full.shape)
                .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
            if total % experts != 0 {
                return Err(LoadError::PoolRowAlignment {
                    name,
                    experts,
                    planned: storage,
                }
                .into());
            }
            let width = storage.element().byte_width();
            let (buffer, allocated, elems) =
                self.allocate(BufferRole::Weight, storage, &weight.pool.shape)?;
            (
                buffer,
                total / experts * width,
                checked_u32(elems)?,
                allocated,
            )
        };
        let writer = match self.device.weight_upload() {
            WeightUpload::Bytes => {
                let writer = self
                    .loaded(exe)?
                    .binder
                    .writer(&bytes, &full.shape, storage)
                    .map_err(|error| encode_error(error, &bytes, &weight.name, storage))?;
                Some(writer)
            }
            WeightUpload::SizeOnly => None,
        };
        let loaded = self.loaded(exe)?;
        loaded.resident.push(ResidentBuf {
            key,
            kind: ResidentKind::Table {
                name: weight.name.clone(),
                experts,
                pool_rows,
                row_bytes,
                writer,
            },
            buffer,
            elems,
            bytes: allocated,
        });
        Ok(loaded.resident.len() - 1)
    }

    /// The resident `[E]` map of `site`, found by the identity of its tables or allocated zeroed.
    fn pool_map(
        &mut self,
        exe: ExecutableId,
        identity: SiteIdentity,
        site: &PoolSite,
        storage: BufferStorage,
    ) -> Result<usize, ExecError> {
        let key = ResidentKey::Map(identity);
        if let Some(index) = self
            .loaded(exe)?
            .resident
            .iter()
            .position(|resident| resident.key == key)
        {
            return Ok(index);
        }
        let (buffer, bytes, elems) = self.allocate(BufferRole::Weight, storage, &[site.experts])?;
        let elems = checked_u32(elems)?;
        let loaded = self.loaded(exe)?;
        loaded.resident.push(ResidentBuf {
            key,
            kind: ResidentKind::Map {
                key: site.map.clone(),
                experts: site.experts,
                storage,
            },
            buffer,
            elems,
            bytes,
        });
        Ok(loaded.resident.len() - 1)
    }

    /// The entry's host syncs: each point's reads (the producing stage's exported buffers) and the
    /// resident buffers of its site. A read must be a plain F32 value a host can take as `f32`s.
    fn entry_syncs(
        &mut self,
        exe: ExecutableId,
        part: &EntryPart<'_>,
        build: &EntryBuild<D>,
    ) -> Result<Vec<EntrySync>, ExecError> {
        let mut syncs = Vec::with_capacity(part.sync_points.len());
        for point in part.sync_points {
            let after = point.after.0 as usize;
            let mut reads = Vec::with_capacity(point.reads.len());
            for &id in &point.reads {
                let &(loc, elems) =
                    build
                        .exported
                        .get(&id)
                        .ok_or(LoadError::BoundaryUnresolved {
                            id,
                            stage: point.after,
                        })?;
                let stage = part
                    .stages
                    .iter()
                    .find(|(stage, _)| *stage == point.after)
                    .map(|&(_, program)| program)
                    .ok_or(LoadError::UnknownHandle)?;
                let exported = stage.exports().find(|export| export.id == id).ok_or(
                    LoadError::BoundaryUnresolved {
                        id,
                        stage: point.after,
                    },
                )?;
                let lane = stage.storage().storage(exported.value).buffer_storage();
                if lane != BufferStorage::i32()
                    || stage.graph().aval(exported.value).dtype != DType::I32
                {
                    return Err(LoadError::SyncReadLane {
                        point: point.id,
                        id,
                    }
                    .into());
                }
                reads.push((id, loc, elems));
            }
            let site = &point.site;
            let tables: Vec<usize> = site
                .weights
                .iter()
                .map(|weight| build.pooled[&weight.slot])
                .collect();
            let map = build.pooled[&site.map];
            let identity = self.site_identity(exe, &tables)?;
            let loaded = self.loaded(exe)?;
            let row_bytes = tables
                .iter()
                .map(|&index| match &loaded.resident[index].kind {
                    ResidentKind::Table { row_bytes, .. } => *row_bytes,
                    ResidentKind::Map { .. } => unreachable!("a site's tables are tables"),
                })
                .collect();
            syncs.push(EntrySync {
                after,
                point: point.clone(),
                identity,
                row_bytes,
                reads,
                tables,
                map,
            });
        }
        Ok(syncs)
    }

    fn load_stage(
        &mut self,
        exe: ExecutableId,
        stage: StageId,
        program: &Program,
        part: &EntryPart<'_>,
        build: &mut EntryBuild<D>,
    ) -> Result<(), ExecError> {
        let EntryBuild {
            locals,
            arena,
            slots,
            token_embeds,
            exported,
            export_lanes,
            stages,
            validations,
            output: entry_output,
            pooled,
            imports,
            outgoing,
        } = &mut *build;
        let links = part.links;
        let device_target = self.device.target();
        if *program.target() != device_target {
            return Err(LoadError::TargetMismatch {
                program: Box::new(*program.target()),
                device: Box::new(device_target),
            }
            .into());
        }
        let g = program.graph();
        let target = self.target_id();
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
                .device_elems(&g.aval(id).shape)
                .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
            checked_u32(elems)
        };
        let mut values: ValueTable<Option<(Loc, u32)>> = ValueTable::filled(g.values().len(), None);

        // Card 547b: one buffer per arena slot, allocated once up front from the program's buffer
        // plan (`Program::buffer_plan`) - never one buffer per computed value. This stage's slots
        // follow the earlier stages' in the entry's shared arena.
        let buffer_plan = program.buffer_plan();
        let arena_base = arena.len();
        arena.reserve(buffer_plan.slot_count());
        for (_, slot_storage, slot_elems) in buffer_plan.slots() {
            // Review F3: `slot_elems` is already the slot's native device-element count - never
            // route it through `Self::allocate`, which would apply `BufferStorage::device_elems` to
            // it a second time and under-allocate a packed slot by half.
            let buffer =
                self.allocate_arena_slot(BufferRole::Activation, slot_storage, slot_elems)?;
            arena.push(buffer);
        }

        macro_rules! push_local {
            ($role:expr, $storage:expr, $shape:expr) => {{
                let (buffer, _bytes, elems) = self.allocate($role, $storage, $shape)?;
                locals.push(buffer);
                (Loc::Local(locals.len() - 1), checked_u32(elems)?)
            }};
        }

        let chunks = binder::chunks(g.inputs().iter().filter_map(|&id| {
            let meta = g.meta(id);
            (meta.storage == Storage::Const)
                .then(|| Some((id, meta.name.as_deref()?, meta.aval.shape.as_slice())))
                .flatten()
        }))?;
        for &id in g.inputs() {
            let meta = g.meta(id);
            let storage = storage_of(id);
            let (loc, elems) = match meta.storage {
                Storage::State => {
                    let name = meta
                        .name
                        .clone()
                        .ok_or(LoadError::UnnamedConst { value: id })?;
                    let role = g
                        .state_role(id)
                        .expect("a validated graph gives every state input a role");
                    let (idx, elems) =
                        self.state_buffer(exe, name, meta.aval.clone(), role, storage)?;
                    (Loc::State(idx), elems)
                }
                Storage::Const => {
                    let name = meta
                        .name
                        .clone()
                        .ok_or(LoadError::UnnamedConst { value: id })?;
                    let (idx, elems) =
                        self.weight_buffer(exe, id, &name, meta, chunks.get(&id), storage)?;
                    (Loc::Weight(idx), elems)
                }
                Storage::Computed(computed) => {
                    let tensor = computed.tensor();
                    let dtype = tensor.dtype();
                    let bytes: Vec<u8> = match (tensor.as_f32(), tensor.as_i32()) {
                        (Some(words), _) => words.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        (_, Some(words)) => words.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        (None, None) => {
                            unreachable!("a computed constant materializes as F32 or I32")
                        }
                    };
                    let encoded =
                        crate::lanes::encode_host(dtype, &bytes, storage).ok_or_else(|| {
                            LoadError::WeightFormat {
                                name: format!("computed {id}"),
                                stored: format!("{dtype:?}"),
                                planned: storage,
                            }
                        })?;
                    let (loc, elems) = push_local!(BufferRole::Input, storage, &meta.aval.shape);
                    crate::device::write_slice(
                        &mut self.device,
                        resolve::<D>(loc, locals, arena, &Held::none()),
                        0,
                        &encoded,
                    )
                    .map_err(device_error(target))?;
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
                    let (loc, elems) = push_local!(BufferRole::Input, storage, &meta.aval.shape);
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
                Storage::Slot(Slot::ExpertPool | Slot::ExpertPoolMap) => {
                    let key = meta
                        .slot_key()
                        .cloned()
                        .ok_or(LoadError::UnkeyedSlot { value: id })?;
                    let &index = pooled.get(&key).ok_or(LoadError::PoolUndeclared { key })?;
                    (
                        Loc::Resident(index),
                        self.loaded(exe)?.resident[index].elems,
                    )
                }
                Storage::Slot(_) => {
                    let key = meta
                        .slot_key()
                        .cloned()
                        .ok_or(LoadError::UnkeyedSlot { value: id })?;
                    if let Some(link) = links
                        .iter()
                        .find(|link| link.to == stage && link.id.slot_key() == key)
                    {
                        // A boundary: the producing stage's buffer, bound with no copy.
                        *exported
                            .get(&link.id)
                            .ok_or(LoadError::BoundaryUnresolved { id: link.id, stage })?
                    } else if let Some(transfer) = part
                        .transfers
                        .iter()
                        .find(|transfer| transfer.to == stage && transfer.id.slot_key() == key)
                    {
                        // A boundary produced on another device: a buffer of this device the
                        // transfer's row writes, in the storage both sides agreed on.
                        if transfer.storage != storage || transfer.aval != meta.aval {
                            return Err(LoadError::SlotConflict { key }.into());
                        }
                        let (loc, elems) =
                            push_local!(BufferRole::Staging, storage, &meta.aval.shape);
                        let Loc::Local(buffer) = loc else {
                            unreachable!("push_local! always returns Loc::Local")
                        };
                        imports.push(ImportBuffer {
                            row: transfer.row,
                            buffer,
                        });
                        (loc, elems)
                    } else if let Some(shared) = slots.iter().find(|slot| slot.key == key) {
                        // The same slot declared by an earlier stage: one host-written buffer.
                        if shared.aval != meta.aval
                            || shared.storage != storage
                            || shared.range != meta.range()
                        {
                            return Err(LoadError::SlotConflict { key }.into());
                        }
                        (Loc::Local(shared.buffer), elems_of(id)?)
                    } else {
                        let (loc, elems) =
                            push_local!(BufferRole::Input, storage, &meta.aval.shape);
                        let Loc::Local(buffer) = loc else {
                            unreachable!("push_local! always returns Loc::Local")
                        };
                        slots.push(BoundSlot {
                            key,
                            aval: meta.aval.clone(),
                            storage,
                            buffer,
                            range: meta.range(),
                        });
                        (loc, elems)
                    }
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
        // The region each equation belongs to, if any.
        let mut region_of: Vec<Option<usize>> = vec![None; g.eqns().len()];
        for (index, region) in program.regions().iter().enumerate() {
            for member in region.members() {
                region_of[member] = Some(index);
            }
        }
        let mut steps = Vec::new();
        for (eqn_index, (eqn, plan)) in program.planned().enumerate() {
            let operand = |values: &ValueTable<Option<(Loc, u32)>>,
                           v: ValueId|
             -> Result<(Loc, u32), ExecError> {
                values[v].ok_or(LoadError::Unimplemented("use before def").into())
            };
            // Card 547b: each operand's own `(Loc, elems)` pair, exactly as it was recorded when
            // that value was bound or computed above - never re-derived here, so a packed-source
            // const's own uploaded word count is carried through unchanged.
            let operands_with_elems = |values: &ValueTable<Option<(Loc, u32)>>,
                                       eqn: &Eqn|
             -> Result<Vec<(Loc, u32)>, ExecError> {
                value_ids(eqn)
                    .into_iter()
                    .map(|v| operand(values, v))
                    .collect()
            };
            // A region runs as one invocation, at the position of its last equation: its other equations
            // materialize nothing here, and every value it returns is placed when its step is built.
            if let Some(region) = region_of[eqn_index].map(|region| &program.regions()[region]) {
                if eqn_index + 1 != region.members().end {
                    continue;
                }
                let (kernel, launch) = match self.device.target() {
                    Target::Aie { target, caps } => {
                        let (kernel, invocation) = self.aie_region_kernel(
                            target,
                            &caps,
                            program,
                            region,
                            max_artifact_bytes,
                        )?;
                        (kernel, LaunchKind::Aie(invocation))
                    }
                    Target::Vpu { target, .. } => {
                        let (kernel, invocation) =
                            self.vpu_region_kernel(target, program, region)?;
                        (kernel, LaunchKind::VpuGraph(invocation))
                    }
                    Target::Gpu { .. } => {
                        return Err(
                            LoadError::Unimplemented("a region program on a GPU target").into()
                        );
                    }
                };
                // The region's bindings, in the order its shape ABI declares them: what it reads, then what
                // it returns. The dispatch's `output` is the last one; the others returned ride in `inputs`
                // after the reads, and the artifact's ABI says which of the arguments the region writes.
                let mut args = Vec::with_capacity(region.reads().len() + region.writes().len());
                for &read in region.reads() {
                    args.push(operand(&values, read)?);
                }
                for &write in region.writes() {
                    let loc = out_loc(
                        OutLocArgs {
                            out: write,
                            commit_plan,
                            buffer_plan,
                            arena_base,
                        },
                        &values,
                    )?;
                    let placed = (loc, elems_of(write)?);
                    values[write] = Some(placed);
                    args.push(placed);
                }
                let output = args
                    .pop()
                    .ok_or(LoadError::Unimplemented("a region that returns no value"))?;
                let work = region
                    .members()
                    .map(|member| dispatch_work(g, &g.eqns()[member]))
                    .sum();
                let members = region.members();
                steps.push(LoadedStep {
                    kernel,
                    inputs: args,
                    output,
                    launch,
                    work,
                    kind_name: "region",
                    label: format!("region[{}..{}]", members.start, members.end),
                });
                continue;
            }
            let work = dispatch_work(g, eqn);
            match plan {
                Plan::Alias(source) => {
                    values[eqn.out] = Some(operand(&values, *source)?);
                }
                Plan::View { src, .. } => {
                    values[eqn.out] = Some(operand(&values, *src)?);
                }
                // `package` puts every node of a graph artifact in the program's one region, and the
                // walk above handled it.
                Plan::Graph => {
                    return Err(LoadError::Unimplemented("a graph node outside any region").into());
                }
                Plan::Dispatches(dispatches) => {
                    let output_loc = out_loc(
                        OutLocArgs {
                            out: eqn.out,
                            commit_plan,
                            buffer_plan,
                            arena_base,
                        },
                        &values,
                    )?;
                    let output = (output_loc, elems_of(eqn.out)?);
                    let operands = operands_with_elems(&values, eqn)?;
                    // A scratch buffer is an arena slot of this equation's own index; its element count
                    // is the declared one, never the slot's capacity (the slot is shared with other
                    // layers' scratch).
                    let scratch_of =
                        |id: poot_graph_plan::ScratchId| -> Result<(Loc, u32), ExecError> {
                            let decl = dispatches.scratch().get(id.index() as usize).ok_or(
                                LoadError::Unimplemented("a dispatch binds an undeclared scratch"),
                            )?;
                            let slot = buffer_plan.scratch_slot(eqn_index, id).ok_or(
                                LoadError::Unimplemented(
                                    "a scratch buffer the buffer plan never colored",
                                ),
                            )?;
                            Ok((
                                Loc::Arena(arena_base + slot.index()),
                                checked_u32(decl.elems())?,
                            ))
                        };
                    let per_dispatch_work = work / (dispatches.len().max(1) as u64);
                    for dispatch in dispatches.iter() {
                        let (body, launch) = (dispatch.body(), dispatch.launch());
                        let (kernel, launch_kind) = match self.device.target() {
                            Target::Gpu { backend, .. } => (
                                self.kernel(backend, dispatch.key(), body, max_artifact_bytes)?,
                                LaunchKind::Grid {
                                    threads: launch.grid,
                                    workgroup: body.workgroup_size,
                                },
                            ),
                            // `package` puts every dispatching equation of an AIE program in a region,
                            // and the walk above handled those.
                            Target::Aie { .. } => {
                                return Err(LoadError::Unimplemented(
                                    "an AIE dispatch outside any region",
                                )
                                .into());
                            }
                            Target::Vpu { .. } => {
                                return Err(LoadError::Unimplemented(
                                    "an Intel NPU runs regions, not per-equation kernels",
                                )
                                .into());
                            }
                        };
                        let binding = dispatch.binding();
                        let output = match binding.write() {
                            Sink::Out => output,
                            Sink::Scratch(id) => scratch_of(id)?,
                        };
                        let mut inputs = Vec::with_capacity(binding.reads().len() + 1);
                        for source in binding.reads() {
                            inputs.push(match *source {
                                Source::Operand(operand) => {
                                    *operands.get(operand).ok_or(LoadError::Unimplemented(
                                        "a dispatch binds an operand its equation lacks",
                                    ))?
                                }
                                Source::Scratch(id) => scratch_of(id)?,
                            });
                        }
                        if !launch.meta.is_empty() {
                            let meta_loc = self.meta_buffer(&launch.meta, locals)?;
                            inputs.push((meta_loc, launch.meta.len() as u32));
                        }
                        steps.push(LoadedStep {
                            kernel,
                            inputs,
                            output,
                            launch: launch_kind,
                            work: per_dispatch_work,
                            kind_name: eqn.op.kind_name(),
                            label: dispatch.label().to_string(),
                        });
                    }
                    values[eqn.out] = Some(output);
                }
            }
        }

        // The state commit: every non-identity edge not donated at its producer is an explicit copy,
        // after the step's dispatches. `preserve_output` snapshots the primary output
        // first when it aliases a destination the copies would otherwise overwrite before it is read.
        let mut commits = Vec::new();
        for (state_input, state_output) in g.state() {
            let dst = values[state_input].expect("state bound").0;
            let src = values[state_output]
                .ok_or(LoadError::Unimplemented("unmaterialized state output"))?
                .0;
            if src == dst {
                continue;
            }
            commits.push((src, dst));
        }

        // Exports: the buffer each boundary value lives in, handed to the stages that read it. The
        // buffer plan gave every export root a slot of its own, so no later equation reuses it.
        // `compile_staged` refuses a boundary that views carried state; the check here is the
        // invariant for a program assembled some other way.
        if let Some((id, state)) = program.export_aliasing_state() {
            return Err(LoadError::BoundaryAliasesState { id, state }.into());
        }
        for export in g.exports() {
            let placed =
                values[export.value].ok_or(LoadError::ExportUnmaterialized { id: export.id })?;
            exported.insert(export.id, placed);
            export_lanes.insert(export.id, storage_of(export.value));
            for transfer in part
                .transfers
                .iter()
                .filter(|transfer| transfer.from == stage && transfer.id == export.id)
            {
                outgoing.push(OutgoingLoad {
                    row: transfer.row,
                    stage: stages.len(),
                    loc: placed.0,
                    bytes: usize::try_from(transfer.bytes)
                        .map_err(|_| LoadError::Unimplemented("a transfer larger than usize"))?,
                });
            }
        }

        let output = values[g.result()]
            .ok_or(LoadError::Unimplemented("unmaterialized output"))?
            .0;
        let preserve_output = if commit_plan.preserve_output {
            let (buf, _bytes, _elems) = self.allocate(
                BufferRole::Output,
                storage_of(g.result()),
                &g.aval(g.result()).shape,
            )?;
            locals.push(buf);
            Some(Loc::Local(locals.len() - 1))
        } else {
            None
        };
        let output_bytes = storage_of(g.result())
            .device_elems(&g.aval(g.result()).shape)
            .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?
            * storage_of(g.result()).element().byte_width();
        let output_host = host_dtype(storage_of(g.result()))
            .map(|dtype| (dtype, g.aval(g.result()).shape.clone()));

        let validation_layout = program.validation().layout.clone();
        let mut validation_sources = Vec::new();
        for source in &program.validation().sources {
            let loc = values[source.value]
                .ok_or(LoadError::Unimplemented("validation source unmaterialized"))?
                .0;
            validation_sources.push((loc, source.lane_count));
        }

        stages.push(StageRun {
            id: stage,
            recording: None,
            recordings: 0,
            replays: 0,
            steps,
            commits,
            output,
            preserve_output,
        });
        validations.push(ValidationLoad {
            layout: validation_layout,
            sources: validation_sources,
        });
        // The last stage loaded is the final one: its output is the entry's.
        *entry_output = Some(OutputLoad {
            loc: preserve_output.unwrap_or(output),
            bytes: output_bytes,
            host: output_host,
        });
        Ok(())
    }

    fn meta_buffer(&mut self, meta: &[u32], locals: &mut Vec<D::Buffer>) -> Result<Loc, ExecError> {
        let target = self.target_id();
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
            .map_err(device_error(target))?;
        crate::device::write_slice(&mut self.device, &buffer, 0, bytemuck::cast_slice(meta))
            .map_err(device_error(target))?;
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
        role: StateRole,
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
        let (buffer, bytes, elems) = self.allocate(BufferRole::State, storage, &aval.shape)?;
        let elems = checked_u32(elems)?;
        let state = &mut self.loaded(exe)?.state;
        state.push(StateBuffer {
            name,
            aval,
            role,
            storage,
            bytes,
            elems,
            buffer,
        });
        Ok((state.len() - 1, elems))
    }

    /// The writer for one weight upload, built once (the allocation and the write both use it) after checking
    /// the upload can land in its planned lane: nothing is read from the store and nothing is allocated yet, so a weight
    /// with no encoding, or a packed-F16 weight holding a non-finite element, is refused before it costs a
    /// buffer. A packed source is its stored bytes as `u32` words (SC-016).
    fn weight_writer(
        &mut self,
        exe: ExecutableId,
        bytes: &binder::ConstBytes,
        name: &str,
        aval: &TensorType,
        storage: BufferStorage,
    ) -> Result<binder::WeightWriter, ExecError> {
        let writer = self
            .loaded(exe)?
            .binder
            .writer(bytes, &aval.shape, storage)
            .map_err(|error| encode_error(error, bytes, name, storage))?;
        writer
            .check_encodable()
            .map_err(|error| encode_error(error, bytes, name, storage))?;
        self.weight_bytes_read += bytes.stored_len() as u64;
        Ok(writer)
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
        meta: &ValueMeta,
        chunk: Option<&binder::Chunk>,
        storage: BufferStorage,
    ) -> Result<(usize, u32), ExecError> {
        let aval = &meta.aval;
        let loaded = self.loaded(exe)?;
        let bytes = loaded.binder.const_bytes(value, name, aval, chunk)?;
        // Before the residency lookup: the bytes may be resident under another executable that
        // declared no range, and this graph's declaration is what its consumers rely on.
        if let Some(range) = meta.range() {
            loaded.binder.check_range(name, &bytes, range)?;
        }
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
        let packed = matches!(
            bytes.stored,
            StoredAs::PackedSource | StoredAs::PackedStackSource { .. }
        );
        if let StoredAs::Dense(dtype) = bytes.stored {
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
            if crate::lanes::encoding_of(dtype, storage).is_none() {
                return Err(LoadError::WeightFormat {
                    name: name.to_string(),
                    stored: format!("{dtype:?}"),
                    planned: storage,
                }
                .into());
            }
        }
        // A generated kernel's packed-source argument always reads `u32` words (Card 642, dquant.md D9),
        // regardless of this value's declared I8/RawBytes logical dtype, padded by one zero word
        // (`poot_quant::packed_source_words`, the one definition of this layout).
        let packed_words = bytes.stored_len().div_ceil(4) + 1;
        let writer = match self.device.weight_upload() {
            WeightUpload::Bytes => Some(self.weight_writer(exe, &bytes, name, aval, storage)?),
            WeightUpload::SizeOnly => None,
        };
        let target = self.target_id();
        let (buffer, elems) = if packed {
            // A packed source's device buffer is always `u32` words (Card 642), never this value's
            // declared I8/RawBytes logical storage (`BufferStorage::device_elems` explicitly
            // refuses that lane): bypass `self.allocate`'s value-lane gate exactly as `meta_buffer`
            // does, sized and typed by the packed word count rather than `storage`/`aval.numel()`.
            let buffer = self
                .device
                .allocate(BufferRole::Weight, BufferStorage::i32(), packed_words)
                .map_err(device_error(target))?;
            (buffer, checked_u32(packed_words)?)
        } else {
            let (buffer, _bytes, elems) =
                self.allocate(BufferRole::Weight, storage, &aval.shape)?;
            (buffer, checked_u32(elems)?)
        };
        if let Some(writer) = writer {
            self.device
                .write_with(&buffer, 0, writer.len(), &mut |range, out| {
                    writer.fill(range, out)
                })
                .map_err(device_error(target))?;
        }
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
        late: &[SlotKey],
    ) -> Result<(), ExecError> {
        let target = self.target_id();
        let Self {
            executables,
            device,
            host_write_bytes,
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
        // A late slot is bound by the caller's `LateInputs` after the stages before the late point; a
        // step that also binds it here has bound it twice.
        let mut seen: Vec<bool> = e
            .slots
            .iter()
            .map(|slot| late.contains(&slot.key))
            .collect();
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
            if key.role() == Slot::WriteMap {
                return Err(BindError::UncheckedWriteMap { key: key.clone() }.into());
            }
            let bytes = binder::check_and_encode(slot, value.shape, value.view)?;
            writes.push((slot.buffer, bytes));
        }
        for (key, map) in inputs.write_maps() {
            let Some(index) = e.slots.iter().position(|slot| &slot.key == key) else {
                return Err(BindError::Unknown { key: key.clone() }.into());
            };
            if std::mem::replace(&mut seen[index], true) {
                return Err(BindError::Duplicate { key: key.clone() }.into());
            }
            let slot = &e.slots[index];
            if key.role() != Slot::WriteMap {
                return Err(BindError::NotAWriteMap { key: key.clone() }.into());
            }
            // The map's own constructor refused a repeated or out-of-pool slot. The slot's declared range
            // is still checked: a map checked against a larger pool than the graph's must not bind.
            let words = map.as_slice();
            let shape = [words.len()];
            let view =
                poot_tensor::HostView::new(DType::I32, words.len(), bytemuck::cast_slice(words))
                    .expect("an i32 slice is a whole number of words");
            let bytes = binder::check_and_encode(slot, &shape, view)?;
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
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
            host_write(device, host_write_bytes, &e.locals[buffer], &bytes)
                .map_err(device_error(target))?;
        }
        Ok(())
    }

    /// Walk one stage into the device: every dispatch, then the state commit. Each dispatch's asserted
    /// lengths are checked against the buffers bound to them first ([`check_arg_lengths`]), so no device
    /// is handed a length its buffer cannot hold.
    fn walk(
        device: &mut D,
        kernels: &[D::Kernel],
        stage: &StageRun<D>,
        buffers: (&[D::Buffer], &[D::Buffer]),
        held: &Held<'_, D>,
    ) -> Result<(), ExecError> {
        let (locals, arena) = buffers;
        let at = |loc| resolve::<D>(loc, locals, arena, held);
        let on_device = device_error(target_of(device));
        for step in &stage.steps {
            let inputs: Vec<Arg<'_, D>> = step
                .inputs
                .iter()
                .map(|&(loc, elems)| Arg {
                    buffer: at(loc),
                    elems,
                })
                .collect();
            let (output_loc, output_elems) = step.output;
            let output = Arg {
                buffer: at(output_loc),
                elems: output_elems,
            };
            check_arg_lengths(device, &inputs, &output)?;
            device
                .dispatch(Dispatch {
                    kernel: &kernels[step.kernel],
                    inputs: &inputs,
                    output,
                    launch: step.launch,
                    work: step.work,
                })
                .map_err(&on_device)?;
        }
        // The two-phase commit: the primary output is snapshotted now, after every
        // dispatch has computed its final value but before any commit copy overwrites the buffer it
        // aliases; the commits themselves run as a simultaneous assignment immediately after.
        if let Some(preserve) = stage.preserve_output {
            device
                .copy(at(stage.output), at(preserve))
                .map_err(&on_device)?;
        }
        for &(src, dst) in &stage.commits {
            device.copy(at(src), at(dst)).map_err(&on_device)?;
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
        stage_idx: usize,
        walk_recording: bool,
        seams: &mut dyn StageSeams<D>,
    ) -> Result<D::Recording, ExecError> {
        self.check_not_poisoned()?;
        let target = self.target_id();
        let error = device_error(target);
        if let Err(e) = self.device.begin() {
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
        let held = Held::of(loaded);
        let Some(e) = loaded.entries.get(entry_idx).and_then(Option::as_ref) else {
            self.abort_after_failure(exe, false);
            return Err(LoadError::UnknownHandle.into());
        };
        let stage = &e.stages[stage_idx];
        let buffers = (e.locals.as_slice(), e.arena.as_slice());
        if let Err(walk_err) = Self::walk(device, kernels, stage, buffers, &held) {
            self.abort_after_failure(exe, walk_recording);
            return Err(walk_err);
        }
        let outgoing: Vec<Outgoing<'_, D>> = e
            .outgoing
            .iter()
            .filter(|out| out.stage == stage_idx)
            .map(|out| Outgoing {
                row: out.row,
                buffer: resolve::<D>(out.loc, buffers.0, buffers.1, &held),
                bytes: out.bytes,
            })
            .collect();
        // The seams run inside the one transaction: a failure aborts it exactly as a failed dispatch
        // does, and the recording the walk makes is the only one the stage has.
        if let Err(seam_err) = seams.after_walk(device, &outgoing) {
            self.abort_after_failure(exe, walk_recording);
            return Err(error(seam_err));
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

    /// Record every stage of `entry` that is not recorded yet, then replay every stage in order.
    /// Returns the entry's replay count.
    ///
    /// Recording is all-or-nothing: the stages record into a scratch list and are published
    /// together, so a failure in any stage (after which `Device::abort` ran exactly once) leaves
    /// the entry with no recording at all. Capture executes nothing, so a recording failure never
    /// marks the executable dirty. A replay failure may have run earlier stages' commits and does.
    fn enqueue(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        sync: &mut dyn HostSync,
        late: &mut dyn LateInputs,
    ) -> Result<u64, ExecError> {
        self.record_if_needed(exe, entry)?;
        self.replay_stages(exe, entry, sync, late)?;
        self.count_replay(exe, entry)?;
        Ok(self.loaded_entry(exe, entry)?.replays)
    }

    /// Record every stage of `entry` that is not recorded yet; see [`Self::enqueue`] for the contract.
    fn record_if_needed(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        let stage_count = {
            let e = self.loaded_entry(exe, entry)?;
            if e.stages.iter().any(|stage| stage.recording.is_some()) {
                0
            } else {
                e.stages.len()
            }
        };
        if stage_count > 0 {
            let mut recorded = Vec::with_capacity(stage_count);
            for stage_idx in 0..stage_count {
                recorded.push(self.run_transaction(
                    exe,
                    entry.index(),
                    stage_idx,
                    false,
                    &mut NoSeams,
                )?);
            }
            let loaded = self.loaded(exe)?;
            let e = loaded
                .entries
                .get_mut(entry.index())
                .and_then(Option::as_mut)
                .ok_or(LoadError::UnknownHandle)?;
            for (stage, recording) in e.stages.iter_mut().zip(recorded) {
                stage.recording = Some(recording);
                stage.recordings += 1;
            }
        }
        Ok(())
    }

    /// Replay every stage of `entry` in order. After a stage that has a sync point the device is
    /// drained, the host reads what the point names and answers with writes, and those land before
    /// the next stage is replayed. A device failure aborts and marks the executable `NeedsReset`; so
    /// does a host or write failure, since the stages already replayed committed state the rest of
    /// the step would have.
    fn replay_stages(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        sync: &mut dyn HostSync,
        late: &mut dyn LateInputs,
    ) -> Result<(), ExecError> {
        self.begin_replays(exe)?;
        self.replay_entry(exe, entry, sync, late)
    }

    /// Enqueue every stage of `entry` in order, with the late inputs and host syncs its program has.
    fn replay_entry(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        sync: &mut dyn HostSync,
        late: &mut dyn LateInputs,
    ) -> Result<(), ExecError> {
        let (stage_count, late_after) = {
            let e = self.loaded_entry(exe, entry)?;
            (e.stages.len(), e.late_after)
        };
        // A program with no late point binds its late inputs before anything runs.
        if late_after.is_none() {
            self.bind_late(exe, entry, late)?;
        }
        for stage_idx in 0..stage_count {
            self.replay_stage(exe, entry, stage_idx)?;
            if late_after == Some(stage_idx) {
                // The stages before the late point are on the device: the host builds what the rest
                // reads. A failure leaves them enqueued, like a failed sync.
                if let Err(err) = self.bind_late(exe, entry, late) {
                    self.mark_needs_reset(exe);
                    return Err(err);
                }
            }
            let due: Vec<usize> = self
                .loaded_entry(exe, entry)?
                .syncs
                .iter()
                .enumerate()
                .filter(|(_, sync_point)| sync_point.after == stage_idx)
                .map(|(index, _)| index)
                .collect();
            for index in due {
                if let Err(err) = self.run_sync(exe, entry, index, sync) {
                    // Whatever failed, an earlier stage already ran: the step is not complete.
                    self.mark_needs_reset(exe);
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    /// Ask `late` for its values and write each into its slot, validated like any bound input: every key
    /// must be a slot of the entry and the values must be exactly the keys, one each.
    fn bind_late(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        late: &mut dyn LateInputs,
    ) -> Result<(), ExecError> {
        if late.keys().is_empty() {
            return Ok(());
        }
        let values = late.build()?;
        let keys = late.keys();
        if values.len() != keys.len() {
            return Err(BindError::LateValues {
                keys: keys.len(),
                values: values.len(),
            }
            .into());
        }
        let target = self.target_id();
        let Self {
            executables,
            device,
            host_write_bytes,
            ..
        } = self;
        let loaded = executables
            .get(exe.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        let e = loaded
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        let mut writes = Vec::with_capacity(values.len());
        for (key, value) in keys.iter().zip(&values) {
            let slot = e
                .slots
                .iter()
                .find(|slot| &slot.key == key)
                .ok_or_else(|| BindError::Unknown { key: key.clone() })?;
            let bytes = binder::check_and_encode(slot, value.shape(), value.view())?;
            writes.push((slot.buffer, bytes));
        }
        for (buffer, bytes) in writes {
            host_write(device, host_write_bytes, &e.locals[buffer], &bytes)
                .map_err(device_error(target))?;
        }
        Ok(())
    }

    /// The copy that carries the token `from` exports into the token slot of `to`, recorded on first use.
    /// Both entries are checked for the feed before this runs.
    fn ensure_link(
        &mut self,
        exe: ExecutableId,
        from: EntryId,
        to: EntryId,
        feed: &ChainFeed,
    ) -> Result<(), ExecError> {
        if self
            .loaded(exe)?
            .links
            .iter()
            .any(|link| link.from == from && link.to == to)
        {
            return Ok(());
        }
        let error = device_error(self.target_id());
        if let Err(err) = self.device.begin() {
            self.abort_after_failure(exe, false);
            return Err(error(err));
        }
        let recorded = {
            let Self {
                executables,
                device,
                ..
            } = self;
            let loaded = executables
                .get(exe.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle);
            let loaded = match loaded {
                Ok(loaded) => loaded,
                Err(err) => {
                    self.abort_after_failure(exe, false);
                    return Err(err.into());
                }
            };
            let entry_of = |id: EntryId| loaded.entries.get(id.index()).and_then(Option::as_ref);
            let (Some(source), Some(target)) = (entry_of(from), entry_of(to)) else {
                self.abort_after_failure(exe, false);
                return Err(LoadError::UnknownHandle.into());
            };
            let picked = source
                .exports
                .iter()
                .find(|export| export.id == feed.picked)
                .expect("checked before the chain bound");
            let slot = target
                .slots
                .iter()
                .find(|slot| slot.key == feed.slot)
                .expect("checked before the chain bound");
            let held = Held::of(loaded);
            let src = resolve::<D>(picked.loc, &source.locals, &source.arena, &held);
            device
                .copy(src, &target.locals[slot.buffer])
                .and_then(|()| device.finish())
        };
        match recorded {
            Ok(recording) => {
                self.loaded(exe)?.links.push(ChainLink {
                    from,
                    to,
                    recording,
                });
                Ok(())
            }
            Err(err) => {
                self.abort_after_failure(exe, false);
                Err(error(err))
            }
        }
    }

    /// Enqueue the recorded copy from `from` into `to`.
    fn replay_link(
        &mut self,
        exe: ExecutableId,
        from: EntryId,
        to: EntryId,
    ) -> Result<(), ExecError> {
        let error = device_error(self.target_id());
        let replayed = {
            let Self {
                executables,
                device,
                ..
            } = self;
            let loaded = executables
                .get(exe.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            let link = loaded
                .links
                .iter()
                .find(|link| link.from == from && link.to == to)
                .expect("recorded before replay");
            device.replay(&link.recording)
        };
        if let Err(err) = replayed {
            self.abort_after_failure(exe, true);
            return Err(error(err));
        }
        Ok(())
    }

    fn mark_needs_reset(&mut self, exe: ExecutableId) {
        if let Some(Some(loaded)) = self.executables.get_mut(exe.index()) {
            loaded.needs_reset = true;
        }
    }

    /// One host sync: drain the device, read the point's values, call the host once and apply its
    /// writes (validated whole before the first byte moves).
    fn run_sync(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        index: usize,
        host: &mut dyn HostSync,
    ) -> Result<(), ExecError> {
        let target = self.target_id();
        self.waits += 1;
        if let Err(err) = self.device.synchronize() {
            if let Some((kernel, code)) = self.device.classify_fault(&err) {
                return Err(ExecError::Fault { kernel, code });
            }
            return Err(device_error(target)(err));
        }
        let (point, reads) = {
            let Self {
                executables,
                device,
                sync_counters,
                ..
            } = self;
            let loaded = executables
                .get(exe.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            let e = loaded
                .entries
                .get(entry.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            let sync = &e.syncs[index];
            let held = Held::of(loaded);
            let mut values = Vec::with_capacity(sync.reads.len());
            for &(id, loc, elems) in &sync.reads {
                let mut bytes = vec![0u8; elems as usize * 4];
                device
                    .read(resolve::<D>(loc, &e.locals, &e.arena, &held), &mut bytes)
                    .map_err(device_error(target))?;
                sync_counters.read_bytes += bytes.len() as u64;
                let words = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect();
                values.push((id, words));
            }
            sync_counters.syncs += 1;
            (
                sync.point.clone(),
                SyncReads {
                    site: sync.identity.clone(),
                    row_bytes: sync.row_bytes.clone(),
                    values,
                },
            )
        };
        let writes = host.sync(&point, &reads)?;
        self.apply_sync_writes(exe, entry, index, &writes)
    }

    /// Land a sync's writes: every range is checked against the site's tables first, then the
    /// requested experts' stored bytes are written to their rows in the table's planned lane, then
    /// the map.
    fn apply_sync_writes(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        index: usize,
        writes: &SyncWrites,
    ) -> Result<(), ExecError> {
        let target = self.target_id();
        let Self {
            executables,
            device,
            sync_counters,
            ..
        } = self;
        let loaded = executables
            .get(exe.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?;
        let sync = &loaded
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or(LoadError::UnknownHandle)?
            .syncs[index];
        let table_of = |fill: &RowFill| -> Result<&ResidentBuf<D>, ExecError> {
            let resident = sync
                .tables
                .get(fill.table)
                .map(|&table| &loaded.resident[table])
                .ok_or_else(|| BindError::ExpertRange {
                    table: format!("#{}", fill.table),
                    expert: fill.table,
                    experts: sync.tables.len(),
                })?;
            Ok(resident)
        };
        for fill in &writes.rows {
            let resident = table_of(fill)?;
            let ResidentKind::Table {
                name,
                experts,
                pool_rows,
                ..
            } = &resident.kind
            else {
                unreachable!("a site's tables are tables")
            };
            let end = fill.first_row + fill.experts.len();
            if end > *pool_rows {
                return Err(BindError::RowRange {
                    table: name.clone(),
                    first_row: fill.first_row,
                    end,
                    pool_rows: *pool_rows,
                }
                .into());
            }
            if let Some(&expert) = fill.experts.iter().find(|&&expert| expert >= *experts) {
                return Err(BindError::ExpertRange {
                    table: name.clone(),
                    expert,
                    experts: *experts,
                }
                .into());
            }
        }
        let map = &loaded.resident[sync.map];
        let ResidentKind::Map {
            key,
            experts,
            storage,
        } = &map.kind
        else {
            unreachable!("a site's map is a map")
        };
        if let Some(values) = &writes.map
            && values.len() != *experts
        {
            return Err(BindError::MapLength {
                key: key.clone(),
                got: values.len(),
                experts: *experts,
            }
            .into());
        }
        for fill in &writes.rows {
            let resident = table_of(fill)?;
            let ResidentKind::Table {
                row_bytes, writer, ..
            } = &resident.kind
            else {
                unreachable!("a site's tables are tables")
            };
            for (offset, &expert) in fill.experts.iter().enumerate() {
                let row = fill.first_row + offset;
                if let Some(writer) = writer {
                    let at = row * row_bytes;
                    let from = expert * row_bytes;
                    device
                        .write_with(&resident.buffer, at, *row_bytes, &mut |range, out| {
                            writer.fill(range.start - at + from..range.end - at + from, out)
                        })
                        .map_err(device_error(target))?;
                }
                sync_counters.rows += 1;
                sync_counters.refill_bytes += *row_bytes as u64;
            }
        }
        if let Some(values) = &writes.map {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let encoded = crate::lanes::encode_host(DType::I32, &bytes, *storage)
                .ok_or(LoadError::Unimplemented("a pool map lane that is not I32"))?;
            crate::device::write_slice(device, &map.buffer, 0, &encoded)
                .map_err(device_error(target))?;
        }
        Ok(())
    }

    /// Wait once for everything the step enqueued, drain deferred releases, and check every stage's
    /// validation packet before any output readback. Returns the host time spent waiting.
    fn complete(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<std::time::Duration, ExecError> {
        let target = self.target_id();
        let error = device_error(target);
        let sync_t0 = Instant::now();
        self.waits += 1;
        if let Err(err) = self.device.synchronize() {
            // A kernel-assert fault surfaces here (Z10): the device itself is in a consistent,
            // reusable state, so the engine is not poisoned. The replay already committed state
            // before the trap, so the executable needs a reset like any other failed step.
            if let Some((kernel, code)) = self.device.classify_fault(&err) {
                self.mark_needs_reset(exe);
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
        if let Err(rejected) = self.check_validation(exe, entry.index()) {
            // The rejected step's carried state was already committed; it is not steppable.
            self.mark_needs_reset(exe);
            return Err(rejected);
        }
        Ok(wait_dur)
    }

    /// Check the entry's validation packet (ADR-0101 decision 2's validated session): a
    /// zero-lane packet reads zero bytes and always passes. Called after replay/eager execution and
    /// `synchronize`, strictly before any primary-output readback.
    fn check_validation(&mut self, exe: ExecutableId, entry_idx: usize) -> Result<(), ExecError> {
        let target = self.target_id();
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
        for validation in &e.validations {
            if validation.layout.lane_count == 0 {
                continue;
            }
            let mut bytes = vec![0u8; validation.layout.lane_count * 4];
            let mut offset = 0usize;
            for &(loc, lane_count) in &validation.sources {
                let buf = resolve::<D>(loc, &e.locals, &e.arena, &Held::of(loaded));
                let len = lane_count * 4;
                device
                    .read(buf, &mut bytes[offset..offset + len])
                    .map_err(device_error(target))?;
                offset += len;
            }
            let bits: &[u32] = bytemuck::cast_slice(&bytes);
            validation.layout.validate_f32_bits(bits)?;
        }
        Ok(())
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
    #[expect(clippy::too_many_arguments)]
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
                e.stages
                    .iter()
                    .flat_map(|stage| &stage.steps)
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
            (
                device_summary,
                dispatches,
                format!("{:?}", self.target_id()),
            )
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
    commit_plan: &'a poot_graph_plan::state_commit::StateCommitPlan,
    buffer_plan: &'a poot_graph_plan::BufferPlan,
    /// Where this stage's arena slots start in the entry's shared arena.
    arena_base: usize,
}

fn out_loc(
    args: OutLocArgs<'_>,
    values: &ValueTable<Option<(Loc, u32)>>,
) -> Result<Loc, ExecError> {
    if let Some(&state_input) = args.commit_plan.inplace.get(&args.out) {
        return Ok(values[state_input].expect("state bound").0);
    }
    let slot = args
        .buffer_plan
        .slot(args.out)
        .ok_or(LoadError::Unimplemented("E4M3/raw-byte device lanes"))?;
    Ok(Loc::Arena(args.arena_base + slot.index()))
}

impl<D: Device> Engine<D> {
    /// Load `part` as a new entry of `exe`: [`Executor::add_entry`] for the whole program, or one
    /// device's share of a multi-device program for a coordinator's engine.
    pub(crate) fn add_part(
        &mut self,
        exe: ExecutableId,
        part: &EntryPart<'_>,
    ) -> Result<EntryId, ExecError> {
        self.check_not_poisoned()?;
        // ADR-0103 amendment, boundary 2: no page of a changed (possibly truncated) checkpoint file is
        // read, and nothing is allocated, once the store's mapped files no longer match what was loaded.
        self.loaded(exe)?.binder.check_sources()?;
        // CR30-lite: weight_buffer/state_buffer mutate the executable's resident
        // weight/state caches as load_entry walks the program, before the entry itself is ever
        // pushed. Snapshot their lengths so a mid-load refusal rolls back whatever this call newly
        // allocated, instead of leaving an orphaned buffer charged to the executable.
        let (weights_len, state_len, resident_len) = {
            let loaded = self.loaded(exe)?;
            (
                loaded.weights.len(),
                loaded.state.len(),
                loaded.resident.len(),
            )
        };
        let entry = match self.load_entry(exe, part) {
            Ok(entry) => entry,
            Err(e) => {
                {
                    let loaded = self.loaded(exe)?;
                    loaded.weights.truncate(weights_len);
                    loaded.state.truncate(state_len);
                    loaded.resident.truncate(resident_len);
                }
                self.residency.retain(|_, upload| upload.strong_count() > 0);
                return Err(e);
            }
        };
        let entries = &mut self.loaded(exe)?.entries;
        entries.push(Some(entry));
        Ok(EntryId::new(entries.len() - 1))
    }

    /// Whether a step of `entry` takes an input for `key`: one of its slots, or the token ids of a
    /// hosted embed.
    pub(crate) fn takes_input(
        &self,
        exe: ExecutableId,
        entry: EntryId,
        key: &SlotKey,
    ) -> Result<bool, ExecError> {
        let e = self.loaded_entry(exe, entry)?;
        let token = SlotKey::new(Slot::Token, None);
        Ok(e.slots.iter().any(|slot| &slot.key == key)
            || (!e.token_embeds.is_empty() && *key == token))
    }

    /// Dispatches stage `stage_idx` of `entry` replays per step.
    pub(crate) fn stage_dispatches(
        &self,
        exe: ExecutableId,
        entry: EntryId,
        stage_idx: usize,
    ) -> Result<usize, ExecError> {
        Ok(self.loaded_entry(exe, entry)?.stages[stage_idx].steps.len())
    }

    /// The buffer transfer `row` delivers into on this device.
    pub(crate) fn import_buffer(
        &self,
        exe: ExecutableId,
        entry: EntryId,
        row: PlanNodeId,
    ) -> Result<&D::Buffer, ExecError> {
        let e = self.loaded_entry(exe, entry)?;
        let import = e
            .imports
            .iter()
            .find(|import| import.row == row)
            .ok_or(LoadError::UnknownHandle)?;
        Ok(&e.locals[import.buffer])
    }

    /// Bind a step's inputs: [`Self::bind`] for a coordinator that splits one step's inputs per device.
    pub(crate) fn bind_inputs(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
    ) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        if self.loaded(exe)?.needs_reset {
            return Err(ExecError::NeedsReset(exe));
        }
        self.bind(exe, entry, inputs, &[])
    }

    /// Record stage `stage_idx` of `entry` as one transaction with `seams` around its walk, and hand
    /// the recording back unpublished: the coordinator publishes every device's recordings together or
    /// none ([`Self::publish_recordings`]). A failure has aborted the device exactly once.
    pub(crate) fn record_stage(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        stage_idx: usize,
        seams: &mut dyn StageSeams<D>,
    ) -> Result<D::Recording, ExecError> {
        self.run_transaction(exe, entry.index(), stage_idx, false, seams)
    }

    /// Publish one recording per stage of `entry`, in stage order.
    pub(crate) fn publish_recordings(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        recordings: Vec<D::Recording>,
    ) -> Result<(), ExecError> {
        let loaded = self.loaded(exe)?;
        let e = loaded
            .entries
            .get_mut(entry.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        for (stage, recording) in e.stages.iter_mut().zip(recordings) {
            stage.recording = Some(recording);
            stage.recordings += 1;
        }
        Ok(())
    }

    /// Start a step's replays on this device ([`Device::begin_step`]).
    pub(crate) fn begin_replays(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        let error = device_error(self.target_id());
        if let Err(err) = self.device.begin_step() {
            self.abort_after_failure(exe, true);
            return Err(error(err));
        }
        Ok(())
    }

    /// Submit the recording of stage `stage_idx` of `entry`.
    pub(crate) fn replay_stage(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        stage_idx: usize,
    ) -> Result<(), ExecError> {
        let error = device_error(self.target_id());
        let replayed = {
            let Self {
                executables,
                device,
                ..
            } = self;
            let e = executables
                .get(exe.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?
                .entries
                .get(entry.index())
                .and_then(Option::as_ref)
                .ok_or(LoadError::UnknownHandle)?;
            device.replay(
                e.stages[stage_idx]
                    .recording
                    .as_ref()
                    .expect("recorded before replay"),
            )
        };
        if let Err(err) = replayed {
            self.abort_after_failure(exe, true);
            return Err(error(err));
        }
        Ok(())
    }

    /// Run `op` on the device between replays (an event record or wait): a failure aborts the device
    /// and marks `exe` as needing a reset, exactly as a failed replay does.
    pub(crate) fn between_replays<R>(
        &mut self,
        exe: ExecutableId,
        op: impl FnOnce(&mut D) -> Result<R, D::Error>,
    ) -> Result<R, ExecError> {
        let error = device_error(self.target_id());
        match op(&mut self.device) {
            Ok(value) => Ok(value),
            Err(err) => {
                self.abort_after_failure(exe, true);
                Err(error(err))
            }
        }
    }

    /// Count one completed enqueue of `entry`: its replays and each stage's.
    pub(crate) fn count_replay(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<(), ExecError> {
        let loaded = self.loaded(exe)?;
        let e = loaded
            .entries
            .get_mut(entry.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        e.replays += 1;
        for stage in &mut e.stages {
            stage.replays += 1;
        }
        Ok(())
    }

    /// Wait for everything this device was handed and check `entry`'s validation packets: the step's
    /// one completion wait on this device.
    pub(crate) fn complete_entry(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<(), ExecError> {
        self.complete(exe, entry).map(drop)
    }

    /// Wait for everything this device was ever handed, without a step's validation: a coordinator
    /// calls it on every device before it releases anything another device's graph may still address.
    pub(crate) fn quiesce(&mut self) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        let error = device_error(self.target_id());
        match self.device.synchronize() {
            Ok(()) => {
                self.pending_release.clear();
                self.residency.retain(|_, upload| upload.strong_count() > 0);
                Ok(())
            }
            Err(err) => {
                if let Some((kernel, code)) = self.device.classify_fault(&err) {
                    return Err(ExecError::Fault { kernel, code });
                }
                Err(error(err))
            }
        }
    }

    /// Mark `exe` as needing a complete reset: a step failed after some of its stages ran.
    pub(crate) fn require_reset(&mut self, exe: ExecutableId) {
        self.mark_needs_reset(exe);
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

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
        let target = self.target_id();
        let Self {
            executables,
            device,
            output_read_bytes,
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
        *output_read_bytes += e.output_bytes as u64;
        // When the output aliases a commit destination (`preserve_output`), the commit copy has
        // already overwritten it for the next step; the snapshot taken before that copy is the
        // real answer.
        let buffer = resolve::<D>(e.output, &e.locals, &e.arena, &Held::of(loaded));
        device
            .read(buffer, &mut out[..e.output_bytes])
            .map_err(device_error(target))
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
        TargetSet::single(self.id, self.device.target())
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        self.device.free_memory_bytes()
    }

    fn device_identity(&self) -> Option<DeviceIdentity> {
        self.device.identity()
    }

    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError> {
        self.check_not_poisoned()?;
        weights.check_store(&store).map_err(LoadError::from)?;
        self.executables.push(Some(Loaded {
            binder: WeightBinder::new(store, weights),
            weights: Vec::new(),
            state: Vec::new(),
            resident: Vec::new(),
            links: Vec::new(),
            entries: Vec::new(),
            needs_reset: false,
        }));
        Ok(ExecutableId::new(self.executables.len() - 1))
    }

    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
    ) -> Result<EntryId, ExecError> {
        self.check_not_poisoned()?;
        for (stage, device, _) in program.stages() {
            if device != self.id {
                return Err(ExecError::Placement {
                    stage,
                    program: device,
                    engine: self.id,
                });
            }
        }
        self.add_part(exe, &EntryPart::whole(program))
    }

    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        self.step_late(exe, entry, inputs, &mut NoLate, sync)
    }

    fn step_chain(
        &mut self,
        exe: ExecutableId,
        steps: &[ChainStep<'_, '_>],
        feed: &ChainFeed,
        sync: &mut dyn HostSync,
    ) -> Result<ChainOutputs<'_>, ExecError> {
        let step_t0 = Instant::now();
        self.check_not_poisoned()?;
        if self.loaded(exe)?.needs_reset {
            return Err(ExecError::NeedsReset(exe));
        }
        if steps.is_empty() {
            return Err(ChainError::Empty.into());
        }
        // Everything the chain needs is checked before any input is written or anything enqueued.
        for (j, step) in steps.iter().enumerate() {
            if steps[..j].iter().any(|other| other.entry == step.entry) {
                return Err(ChainError::RepeatedEntry(step.entry).into());
            }
            let e = self.loaded_entry(exe, step.entry)?;
            if !e.syncs.is_empty() || e.late_after.is_some() || !e.token_embeds.is_empty() {
                return Err(ChainError::NeedsHost(step.entry).into());
            }
            let picked = e
                .exports
                .iter()
                .find(|export| export.id == feed.picked)
                .ok_or(ChainError::NoFeed {
                    entry: step.entry,
                    what: "export of the picked token",
                })?;
            let slot =
                e.slots
                    .iter()
                    .find(|slot| slot.key == feed.slot)
                    .ok_or(ChainError::NoFeed {
                        entry: step.entry,
                        what: "slot for the fed token",
                    })?;
            let slot_elems = slot.storage.device_elems(&slot.aval.shape);
            if slot.storage != picked.storage || slot_elems != Some(picked.elems as usize) {
                return Err(ChainError::FeedLanes {
                    entry: step.entry,
                    picked: (picked.storage, picked.elems),
                    slot: (slot.storage, slot_elems.map_or(0, |elems| elems as u32)),
                }
                .into());
            }
        }
        for (j, step) in steps.iter().enumerate() {
            let fed: &[SlotKey] = if j == 0 {
                &[]
            } else {
                std::slice::from_ref(&feed.slot)
            };
            self.bind(exe, step.entry, step.inputs, fed)?;
        }
        for step in steps {
            self.record_if_needed(exe, step.entry)?;
        }
        for pair in steps.windows(2) {
            self.ensure_link(exe, pair[0].entry, pair[1].entry, feed)?;
        }
        self.begin_replays(exe)?;
        for (j, step) in steps.iter().enumerate() {
            self.replay_entry(exe, step.entry, sync, &mut NoLate)?;
            self.count_replay(exe, step.entry)?;
            if let Some(next) = steps.get(j + 1) {
                self.replay_link(exe, step.entry, next.entry)?;
            }
        }
        let encode_dur = step_t0.elapsed();
        let last = steps.last().expect("a chain has a step").entry;
        let wait_dur = self.complete(exe, last)?;
        for step in &steps[..steps.len() - 1] {
            if let Err(rejected) = self.check_validation(exe, step.entry.index()) {
                self.mark_needs_reset(exe);
                return Err(rejected);
            }
        }
        let time = self.device.device_time();
        let wall_dur = step_t0.elapsed();
        for step in steps {
            let replays = self.loaded_entry(exe, step.entry)?.replays;
            let waited = if step.entry == last {
                wait_dur
            } else {
                std::time::Duration::ZERO
            };
            self.record_step_timing(
                exe, step.entry, replays, encode_dur, waited, wall_dur, &time,
            )?;
        }
        Ok(ChainOutputs::new(
            self,
            exe,
            steps.iter().map(|step| step.entry).collect(),
            time,
        ))
    }

    fn step_late(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        late: &mut dyn LateInputs,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        // Card 552: the host-wall partition is always measured (never gated by `TimingOptions`,
        // since `Instant::now()` costs nothing a step doesn't already pay). `encode` covers
        // binding + record/replay (building and submitting the work); `wait` is the explicit
        // blocking `synchronize()`; `other` (never stored, only ever printed) is whatever this
        // step's wall includes past those two - never a device-time subtraction.
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
        // The step is three private seams (core review CR30): bind the inputs, record-or-enqueue
        // every stage in order, then complete once. The engine itself waits only in `complete`: the
        // stages share one in-order queue, and every backend's `replay` only enqueues.
        self.bind(exe, entry, inputs, late.keys())?;
        let replay_count = self.enqueue(exe, entry, sync, late)?;
        let encode_dur = step_t0.elapsed();
        let wait_dur = self.complete(exe, entry)?;
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
        let target = self.target_id();
        let Self {
            executables,
            device,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        match scope {
            StateScope::All => {
                for state in &loaded.state {
                    device
                        .write_with(&state.buffer, 0, state.bytes, &mut |_, out| out.fill(0))
                        .map_err(device_error(target))?;
                }
                // A resident pool is carried state too: a host that resets its own residency
                // bookkeeping with this call must not find rows it believes are gone.
                for resident in &loaded.resident {
                    device
                        .write_with(&resident.buffer, 0, resident.bytes, &mut |_, out| {
                            out.fill(0)
                        })
                        .map_err(device_error(target))?;
                }
                loaded.needs_reset = false;
            }
            StateScope::Row(row) => {
                // A paged KV pool is the one carried value without a row axis: no row of it is
                // this sequence's.
                let rowed =
                    |state: &&StateBuffer<D>| state.role != StateRole::Positional { axis: 0 };
                // Check every buffer before writing any: a refused reset leaves the state as it was.
                let mut rows = Vec::new();
                for state in loaded.state.iter().filter(rowed) {
                    let leading = state.aval.shape.first().copied().unwrap_or(0);
                    let bytes = state.bytes.checked_div(leading).unwrap_or(0);
                    if row >= leading || bytes * leading != state.bytes {
                        return Err(BindError::StateRow {
                            name: state.name.clone(),
                            row,
                            rows: leading,
                        }
                        .into());
                    }
                    rows.push((&state.buffer, row * bytes, bytes));
                }
                for (buffer, offset, bytes) in rows {
                    device
                        .write_with(buffer, offset, bytes, &mut |_, out| out.fill(0))
                        .map_err(device_error(target))?;
                }
                // The other rows may be mid-failure: only `All` clears `needs_reset`.
            }
        }
        Ok(())
    }

    fn write_state_row(
        &mut self,
        exe: ExecutableId,
        name: &str,
        row: usize,
        values: poot_tensor::HostView<'_>,
    ) -> Result<(), ExecError> {
        self.check_not_poisoned()?;
        let target = self.target_id();
        let Self {
            executables,
            device,
            host_write_bytes,
            ..
        } = self;
        let loaded = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let state = loaded
            .state
            .iter()
            .find(|state| state.name == name)
            .ok_or_else(|| BindError::UnknownState {
                name: name.to_string(),
            })?;
        let rows = state.aval.shape.first().copied().unwrap_or(0);
        if row >= rows {
            return Err(BindError::StateRow {
                name: name.to_string(),
                row,
                rows,
            }
            .into());
        }
        let row_elems = state.aval.numel() / rows;
        if values.dtype() != state.aval.dtype || values.elems() != row_elems {
            return Err(BindError::StateRowValues {
                name: name.to_string(),
                expected: state.aval.dtype,
                expected_len: row_elems,
                got: values.dtype(),
                got_len: values.elems(),
            }
            .into());
        }
        let encoded = crate::lanes::encode_host(state.aval.dtype, values.bytes(), state.storage)
            .ok_or(LoadError::Unimplemented(
                "a state row in a lane the host cannot encode",
            ))?;
        let row_bytes = state.bytes / rows;
        if encoded.len() != row_bytes {
            return Err(BindError::StateRow {
                name: name.to_string(),
                row,
                rows,
            }
            .into());
        }
        crate::device::write_slice(device, &state.buffer, row * row_bytes, &encoded)
            .map_err(device_error(target))?;
        *host_write_bytes += encoded.len() as u64;
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
        let mut stages = Vec::new();
        for (exe, loaded) in self.executables.iter().enumerate() {
            let Some(loaded) = loaded else { continue };
            for (entry, loaded_entry) in loaded.entries.iter().enumerate() {
                let Some(loaded_entry) = loaded_entry else {
                    continue;
                };
                replays += loaded_entry.replays;
                for stage in &loaded_entry.stages {
                    recordings += stage.recordings;
                    stages.push(StageStats {
                        exe: ExecutableId::new(exe),
                        entry: EntryId::new(entry),
                        stage: stage.id,
                        recordings: stage.recordings,
                        replays: stage.replays,
                    });
                }
            }
        }
        // Card 547a: read the device's own runtime counters, not an engine-side tally over its own
        // bookkeeping, so cross-backend equality is not equal by construction.
        ExecutorStats {
            recordings,
            replays,
            stages,
            memory: self.device.memory(),
            weight_bytes_read: self.weight_bytes_read,
            syncs: self.sync_counters.syncs,
            sync_read_bytes: self.sync_counters.read_bytes,
            pool_rows_written: self.sync_counters.rows,
            pool_refill_bytes: self.sync_counters.refill_bytes,
            host_write_bytes: self.host_write_bytes,
            output_read_bytes: self.output_read_bytes,
            waits: self.waits,
            kernels_resolved: self.kernels_resolved,
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
    use poot_codegen::ArtifactTooLarge;
    use poot_runtime_common::BufferRole;

    struct NoopDevice;

    impl Device for NoopDevice {
        type Buffer = ();
        type Kernel = String;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target::Gpu {
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
        fn buffer_elems(&self, _buffer: &()) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<String, std::convert::Infallible> {
            Ok(key.to_string())
        }
        fn begin(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &(),
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), std::convert::Infallible> {
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
        let body_a = poot_kernelgen::reduce_last_dt(
            "shared_key",
            poot_kernel_ir::Ty::F32,
            poot_kernel_ir::BinOp::Add,
            4,
            0.0,
        );
        let body_b = poot_kernelgen::reduce_last_dt(
            "shared_key",
            poot_kernel_ir::Ty::F32,
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
                Backend::SpirvVulkan,
                "shared_key",
                &body_a,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        let index_b = engine
            .kernel(
                Backend::SpirvVulkan,
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
                Backend::SpirvVulkan,
                "shared_key",
                &body_b,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        let index_a2 = engine
            .kernel(
                Backend::SpirvVulkan,
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
                Backend::SpirvVulkan,
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

    /// POOT-1068: a table bound to another backend than the engine's is refused, not shared.
    /// Mutation: drop the backend comparison in `with_shared_kernels`; the engine accepts it and this row
    /// goes red.
    #[test]
    fn a_shared_table_bound_to_another_backend_is_refused() {
        let engine = Engine::new(NoopDevice);
        let table = SharedKernels::new(TargetId::Gpu(Backend::Nvptx));
        let Err(mismatch) = engine.with_shared_kernels(table) else {
            panic!("a table for another backend must be refused");
        };
        assert_eq!(
            mismatch,
            SharedKernelsMismatch::Target {
                table: TargetId::Gpu(Backend::Nvptx),
                engine: TargetId::Gpu(Backend::SpirvVulkan),
            }
        );
    }

    /// POOT-1068: the artifact limit holds on a hit of the shared table too: an artifact one engine
    /// resolved under a generous limit is refused to an engine whose program allows less, exactly as a
    /// kernel-cache hit would be. Mutation: skip the length check on a hit in `get_or_resolve`; the second
    /// engine loads the oversize artifact and this row goes red.
    #[test]
    fn a_shared_hit_is_checked_against_the_callers_artifact_limit() {
        let body = poot_kernelgen::reduce_last_dt(
            "limit",
            poot_kernel_ir::Ty::F32,
            poot_kernel_ir::BinOp::Add,
            4,
            0.0,
        );
        let table = SharedKernels::new(TargetId::Gpu(Backend::SpirvVulkan));
        let mut first = Engine::new(NoopDevice)
            .with_shared_kernels(table.clone())
            .unwrap();
        first
            .kernel(
                Backend::SpirvVulkan,
                "limit",
                &body,
                poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
            )
            .unwrap();
        assert_eq!(first.kernels_resolved, 1);
        let mut second = Engine::new(NoopDevice).with_shared_kernels(table).unwrap();
        let tiny = NonZeroU64::new(1).unwrap();
        let ExecError::Load(load) = second
            .kernel(Backend::SpirvVulkan, "limit", &body, tiny)
            .unwrap_err()
        else {
            panic!("expected a load error");
        };
        assert!(
            matches!(
                *load,
                LoadError::Codegen {
                    source: poot_codegen::CompileError::ArtifactTooLarge(ArtifactTooLarge {
                        limit: 1,
                        ..
                    }),
                    ..
                }
            ),
            "{load:?}"
        );
        assert_eq!(second.kernels_resolved, 0, "the hit resolved nothing");
    }
}

#[cfg(test)]
mod arena_slot_alloc_tests {
    use super::*;
    use crate::device::DeviceTime;
    use poot_runtime_common::BufferRole;

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
            poot_graph_plan::Target::Gpu {
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
        fn buffer_elems(&self, _buffer: &()) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<String, std::convert::Infallible> {
            Ok(key.to_string())
        }
        fn begin(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &(),
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), std::convert::Infallible> {
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
        CompileOptions, multi_device::topology::DeviceId, staged::DevicePlacement,
        staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
    };
    use poot_quant::StoredBytes;
    use poot_quant::weights::{
        AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightMap, WeightRole, WeightView,
    };
    use poot_runtime_common::BufferRole;

    const PROJECTION_INPUT: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const PROJECTION_WEIGHT: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));

    /// Records every allocation (role, storage, native elements) and every write, by buffer index.
    #[derive(Default)]
    struct UploadRecorder {
        allocations: Vec<(BufferRole, BufferStorage, usize)>,
        writes: Vec<(usize, Vec<u8>)>,
        upload: WeightUpload,
    }

    impl Device for UploadRecorder {
        type Buffer = usize;
        type Kernel = ();
        type Recording = ();
        type Error = std::convert::Infallible;

        fn weight_upload(&self) -> WeightUpload {
            self.upload
        }

        fn target(&self) -> poot_graph_plan::Target {
            poot_graph_plan::Target::Gpu {
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
        fn buffer_elems(&self, buffer: &usize) -> usize {
            self.allocations[*buffer].2
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn begin(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn write_with(
            &mut self,
            dst: &usize,
            offset: usize,
            len: usize,
            fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), std::convert::Infallible> {
            let mut bytes = vec![0u8; len];
            fill(offset..offset + len, &mut bytes);
            self.writes.push((*dst, bytes));
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
        let store = projection_store(w_bits, stored, StoredBytes::from);
        load_projection(engine, store, declared)
    }

    /// The projection's store: an f32 input and a weight of `stored` dtype holding `w_bits`, each tensor's
    /// bytes owned through `own` (heap bytes, or a view of a mapped file).
    fn projection_store(
        w_bits: &[u16],
        stored: DType,
        mut own: impl FnMut(Vec<u8>) -> StoredBytes,
    ) -> Arc<WeightStore> {
        let x_bytes: Vec<u8> = (0..4 * K).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut store = WeightStore::builder();
        for (id, dtype, shape, bytes) in [
            (PROJECTION_INPUT, DType::F32, vec![4, K], x_bytes),
            (PROJECTION_WEIGHT, stored, vec![N, K], w_bytes),
        ] {
            store
                .insert(
                    id.const_name(),
                    WeightEntry::Dense(DenseWeight::try_new(dtype, shape, own(bytes)).unwrap()),
                )
                .unwrap();
        }
        Arc::new(store.build())
    }

    /// Load the projection graph over `store` into `engine`, declaring `w` as `declared`.
    fn load_projection(
        engine: &mut Engine<UploadRecorder>,
        store: Arc<WeightStore>,
        declared: DType,
    ) -> Result<EntryId, ExecError> {
        let b = Builder::for_tests();
        let x = b
            .constant(&PROJECTION_INPUT.const_name(), TensorType::f32(vec![4, K]))
            .unwrap();
        let w = b
            .constant(
                &PROJECTION_WEIGHT.const_name(),
                TensorType::new(vec![N, K], declared),
            )
            .unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(out).unwrap();
        let mut weights = WeightMap::builder(&store);
        for id in [PROJECTION_INPUT, PROJECTION_WEIGHT] {
            weights
                .map(id, WeightView::Stored(id.const_name().into()))
                .unwrap();
        }
        let weights = weights.build();
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("the projection compiles");
        let exe = engine.load_weights(store, Arc::new(weights))?;
        engine.add_entry(exe, &program)
    }

    /// ADR-0103 amendment, boundary 2 (POOT-1099): a store whose mapped checkpoint file was truncated after
    /// it was loaded is refused by `add_entry` as the typed `LoadError::SourceChanged`, before anything is
    /// allocated or written, and without touching a page of the truncated file.
    ///
    /// Mutation: ignore the `check_sources` result in `Engine::add_entry`; the entry reads the truncated
    /// pages and the test process dies with `SIGBUS`.
    #[test]
    fn add_entry_refuses_a_store_whose_mapped_file_changed() {
        let path = poot_test_util::unique_temp_path("bind_changed.bin");
        let bits = finite_bits();
        std::fs::write(&path, vec![0u8; 4 * K * 4 + bits.len() * 2]).unwrap();
        let mapped = StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap();
        // The input is the file's head, the weight its tail.
        let mut next = 0;
        let store = projection_store(&bits, DType::F16, |bytes| {
            let view = mapped.slice(next..next + bytes.len()).unwrap();
            next += bytes.len();
            view
        });
        let mut engine = Engine::new(UploadRecorder::default());
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        let error = load_projection(&mut engine, store, DType::F16).unwrap_err();
        assert!(
            matches!(&error, ExecError::Load(load) if matches!(**load, LoadError::SourceChanged(_))),
            "{error:?}"
        );
        assert!(
            engine.device.allocations.is_empty() && engine.device.writes.is_empty(),
            "nothing was allocated or written"
        );
    }

    /// Distinct finite binary16 words, one per weight element.
    fn finite_bits() -> Vec<u16> {
        (0..N * K).map(|i| 0x3c00 + 37 * i as u16).collect()
    }

    /// Card 1007 acceptance row 2: the F16 const is allocated as packed-F16 storage of `ceil(N * K / 2)` words and
    /// written with exactly the checkpoint's bytes, zero-padded to the last word: no f32 host copy exists. No
    /// weight buffer of `N * K` f32 elements is allocated either.
    ///
    /// Mutation: in `lanes::encode_range`'s `PackedWordsFiniteF16` arm, widen each element to f32 bytes on upload;
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
    /// Mutation: drop the non-finite scan in `WeightWriter::check_encodable`; the entry loads
    /// and this row goes red.
    #[test]
    fn a_non_finite_packed_f16_weight_is_a_typed_refusal() {
        let mut engine = Engine::new(UploadRecorder::default());
        let mut bits = finite_bits();
        bits[5] = 0xfc00; // -inf
        match add_projection(&mut engine, &bits) {
            Err(ExecError::Load(error)) => match *error {
                LoadError::NonFiniteF16Weight { name, index, bits } => {
                    assert_eq!(
                        (name.as_str(), index, bits),
                        (PROJECTION_WEIGHT.const_name().as_str(), 5, 0xfc00)
                    );
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
    /// reaches `WeightBinder::writer` and fails as the untyped-by-dtype `WeightFormat`, so this row goes red.
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
                    (
                        PROJECTION_WEIGHT.const_name().as_str(),
                        DType::F16,
                        DType::F32
                    )
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

    /// POOT-1066: a size-only (planning) device refuses a stored/declared dtype mismatch exactly as the
    /// byte-taking one does, so a planning pass cannot admit a graph the real upload would refuse, and it
    /// reads none of the store. A packed-F16 weight is sized to the words the byte-taking upload allocates.
    ///
    /// Mutation: skip the dtype check when the device only sizes; the refused weight loads and this row goes
    /// red.
    #[test]
    fn a_size_only_device_refuses_a_dtype_mismatch_and_sizes_a_packed_weight_like_the_byte_taking_one()
     {
        let sized = || {
            Engine::new(UploadRecorder {
                upload: WeightUpload::SizeOnly,
                ..UploadRecorder::default()
            })
        };
        let mut engine = sized();
        match add_projection_declared(&mut engine, &finite_bits(), DType::F16, DType::F32) {
            Err(ExecError::Load(error)) => assert!(
                matches!(*error, LoadError::WeightDtype { .. }),
                "expected WeightDtype, got {error}"
            ),
            Err(other) => panic!("expected WeightDtype, got {other}"),
            Ok(_) => panic!("a size-only upload admitted an F16-stored weight declared F32"),
        }
        assert_eq!(engine.stats().weight_bytes_read, 0);

        let mut sized_engine = sized();
        add_projection(&mut sized_engine, &finite_bits()).expect("the size-only entry loads");
        let mut bytes_engine = Engine::new(UploadRecorder::default());
        add_projection(&mut bytes_engine, &finite_bits()).expect("the entry loads");
        assert_eq!(
            sized_engine.device.allocations,
            bytes_engine.device.allocations
        );
        assert_eq!(sized_engine.stats().weight_bytes_read, 0);
        assert!(bytes_engine.stats().weight_bytes_read > 0);
    }

    /// Card 1011: the plan never retypes a BF16 const, so a BF16-stored weight declared F32 is a refusal like
    /// the F16 one above, not a widening on upload: a typed `WeightDtype` naming the weight and both dtypes,
    /// and no f32 buffer of the weight's size is allocated or written.
    ///
    /// Mutation: re-admit the BF16/F32 pair in `Engine::weight_buffer`'s dtype check (and widen it in
    /// `WeightWriter::check_encodable`); the entry loads and this row goes red.
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
                    (
                        PROJECTION_WEIGHT.const_name().as_str(),
                        DType::BF16,
                        DType::F32
                    )
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

/// Card 564: weight binding through the executable's weight map - the map's views to stored byte
/// runs, the typed unbound refusal, residency by what uploads, and the hosted embed fill - driven
/// through the production `load_weights`/`add_entry`/`step` path on a recording device.
#[cfg(test)]
mod weight_source_tests {
    use super::*;
    use poot_runtime_common::BufferRole;

    use crate::{NoSync, StepInputs};
    use poot_graph_ir::test_support::const_weights::ConstWeights;
    use poot_graph_ir::{BinOp, Builder, Graph, IndexRange, WeightRef};

    use poot_graph_plan::{
        CompileOptions, WeightFormats, bind_packed_weights, multi_device::topology::DeviceId,
        staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::TargetSet,
        staged::compile_staged,
    };
    use poot_quant::format::WeightFormat;
    use poot_quant::weights::{
        AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightKey, WeightMap,
        WeightMapError, WeightRole, WeightView,
    };
    use poot_quant::{PackedPayload, SourceRole, StoredBytes};
    use poot_runtime_common::MemoryCounters;
    use std::ops::Range;

    /// Records every allocation (role, storage, native elements) and every write by buffer index,
    /// and charges each allocation to the runtime memory counters `Engine::stats` reads. `cap`, when
    /// set, is the target's single-buffer limit, so `compile`'s legalize sees it.
    #[derive(Default)]
    struct Recorder {
        allocations: Vec<(BufferRole, BufferStorage, usize)>,
        writes: Vec<(usize, Vec<u8>)>,
        /// The byte offset of each entry of `writes`.
        write_offsets: Vec<usize>,
        memory: MemoryCounters,
        cap: Option<u64>,
        upload: WeightUpload,
        /// When set, every replay and synchronize is appended, so a row can order them against a
        /// host's calls.
        log: Option<Arc<std::sync::Mutex<Vec<&'static str>>>>,
    }

    impl Device for Recorder {
        type Buffer = usize;
        type Kernel = ();
        type Recording = ();
        type Error = std::convert::Infallible;

        fn weight_upload(&self) -> WeightUpload {
            self.upload
        }

        fn target(&self) -> poot_graph_plan::Target {
            let mut caps = poot_target::DeviceCaps::wgpu_rdna3_igpu();
            if let Some(cap) = self.cap {
                caps.max_buffer_bytes = cap;
            }
            poot_graph_plan::Target::Gpu {
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
        fn buffer_elems(&self, buffer: &usize) -> usize {
            self.allocations[*buffer].2
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn begin(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), std::convert::Infallible> {
            if let Some(log) = &self.log {
                log.lock().unwrap().push("replay");
            }
            Ok(())
        }
        fn write_with(
            &mut self,
            dst: &usize,
            offset: usize,
            len: usize,
            fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), std::convert::Infallible> {
            let mut bytes = vec![0u8; len];
            fill(offset..offset + len, &mut bytes);
            if let Some(log) = &self.log {
                log.lock().unwrap().push("write");
            }
            self.writes.push((*dst, bytes));
            self.write_offsets.push(offset);
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), std::convert::Infallible> {
            if let Some(log) = &self.log {
                log.lock().unwrap().push("synchronize");
            }
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
        WeightEntry::Dense(
            DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap(),
        )
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

    fn compile(engine: &Engine<Recorder>, g: &Graph) -> StagedProgram {
        compile_staged(
            &g.clone(),
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("the fixture compiles")
    }

    /// `out = a + b`, both F32 `[N, K]` consts named `a` and `b`; returns the graph and `b`'s id.
    fn sum_of(a: &str, b_name: &str) -> (Graph, ValueId) {
        let b = Builder::for_tests();
        let x = b.constant(a, TensorType::f32(vec![N, K])).unwrap();
        let y = b.constant(b_name, TensorType::f32(vec![N, K])).unwrap();
        let out = b.binary(BinOp::Add, x, y).unwrap();
        (b.finish(out).unwrap(), y.id())
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

    /// Load `g` over `weights` twice, once taking the bytes and once only sizing them, and require the
    /// size-only engine to allocate exactly what the byte-taking one does while reading none of the store.
    /// Mutation: have the engine read the store in `WeightUpload::SizeOnly`; the read counter goes red.
    fn assert_size_only_matches_bytes(
        weights: &Arc<WeightStore>,
        mapped: &Arc<WeightMap>,
        g: &Graph,
    ) {
        let load = |upload| {
            let mut engine = Engine::new(Recorder {
                upload,
                ..Recorder::default()
            });
            let program = compile(&engine, g);
            let exe = engine
                .load_weights(Arc::clone(weights), Arc::clone(mapped))
                .unwrap();
            engine.add_entry(exe, &program).unwrap();
            engine
        };
        let bytes = load(WeightUpload::Bytes);
        let sized = load(WeightUpload::SizeOnly);
        assert_eq!(
            sized.device().allocations,
            bytes.device().allocations,
            "size-only allocates what the byte-taking upload allocates"
        );
        let held = |engine: &Engine<Recorder>| {
            engine
                .stats()
                .memory
                .iter()
                .map(|(role, s)| (*role, s.live_bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(held(&sized), held(&bytes));
        assert!(
            bytes.stats().weight_bytes_read > 0,
            "the byte-taking upload reads the store"
        );
        assert_eq!(
            sized.stats().weight_bytes_read,
            0,
            "size-only reads none of the store"
        );
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
    /// Mutation: in `WeightBinder::locate`, answer a name no id spells with the first mapped id; the
    /// entry loads and this row goes red.
    #[test]
    fn a_const_with_no_store_entry_and_no_step_value_is_unbound() {
        let mut engine = Engine::new(Recorder::default());
        let a = id(Q).const_name();
        let (g, missing) = sum_of(&a, "b");
        let program = compile(&engine, &g);
        let weights = store(vec![(a.as_str(), f32_entry(vec![N, K], 1))]);
        let mapped = map(
            &weights,
            vec![(id(Q), WeightView::Stored(a.as_str().into()))],
        );
        let exe = engine.load_weights(weights, mapped).unwrap();
        assert_eq!(
            unbound(engine.add_entry(exe, &program)),
            (missing, "b".to_string())
        );
    }

    fn weight_map_refusal(result: Result<ExecutableId, ExecError>) -> WeightMapError {
        match result {
            Err(ExecError::Load(error)) => match *error {
                LoadError::WeightMap(error) => error,
                other => panic!("expected WeightMap, got {other}"),
            },
            Err(other) => panic!("expected WeightMap, got {other}"),
            Ok(_) => panic!("the weights loaded"),
        }
    }

    /// POOT-1292: a map built against one store and loaded with another that lacks a key the map reads is
    /// a typed refusal at `load_weights` naming the weight and the key, not a panic in the embed gather.
    ///
    /// Mutation: delete the `check_store` call in `Engine::load_weights`; the weights load and this row
    /// goes red.
    #[test]
    fn load_weights_refuses_a_store_missing_a_key_the_map_reads() {
        let mut engine = Engine::new(Recorder::default());
        let embed = WeightId::model(WeightRole::Embed);
        let built = store(vec![("w.embed", f32_entry(vec![N, K], 2))]);
        let mapped = map(&built, vec![(embed, WeightView::Stored("w.embed".into()))]);
        let forged = store(vec![("w.other", f32_entry(vec![N, K], 2))]);
        assert_eq!(
            weight_map_refusal(engine.load_weights(forged, mapped)),
            WeightMapError::Missing {
                id: embed,
                key: "w.embed".into()
            }
        );
    }

    /// POOT-1292: a store whose entry under a mapped key is smaller than the map's shape is a typed
    /// refusal naming both shapes, not a slice panic when the embed rows are read.
    ///
    /// Mutation: in `WeightMap::check_store`, skip the comparison against the recorded handle; the
    /// weights load and this row goes red.
    #[test]
    fn load_weights_refuses_a_store_entry_smaller_than_the_mapped_shape() {
        let mut engine = Engine::new(Recorder::default());
        let embed = WeightId::model(WeightRole::Embed);
        let built = store(vec![("w.embed", f32_entry(vec![N, K], 2))]);
        let mapped = map(&built, vec![(embed, WeightView::Stored("w.embed".into()))]);
        let forged = store(vec![("w.embed", f32_entry(vec![N - 1, K], 2))]);
        let refusal = weight_map_refusal(engine.load_weights(forged, mapped));
        let WeightMapError::StoreMismatch { id, mapped, stored } = refusal else {
            panic!("expected StoreMismatch, got {refusal}");
        };
        assert_eq!(id, embed);
        assert_eq!(mapped.shape, vec![N, K]);
        assert_eq!(stored.shape, vec![N - 1, K]);
    }

    /// A matching pairing still loads: the check does not refuse what the loaders build.
    #[test]
    fn load_weights_accepts_the_store_the_map_was_built_against() {
        let mut engine = Engine::new(Recorder::default());
        let embed = WeightId::model(WeightRole::Embed);
        let built = store(vec![("w.embed", f32_entry(vec![N, K], 2))]);
        let mapped = map(&built, vec![(embed, WeightView::Stored("w.embed".into()))]);
        engine.load_weights(built, mapped).unwrap();
    }

    /// SC-003: under `Map`, a const whose name equals a store key but which the map does not name is
    /// `Unbound`; a map that does name it binds the same graph over the same store, so the store key
    /// is there.
    ///
    /// Mutation: in `WeightBinder::locate`, fall back to a store lookup by the const's name when the
    /// map has no id for it; the const binds and this row goes red.
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
        let exe = engine.load_weights(Arc::clone(&weights), mapped).unwrap();
        assert_eq!(
            unbound(engine.add_entry(exe, &program)),
            (embed, "w.embed".to_string())
        );

        let embed_id = WeightId::model(WeightRole::Embed);
        let both = map(
            &weights,
            vec![
                (id(Q), WeightView::Stored(q.as_str().into())),
                (embed_id, WeightView::Stored("w.embed".into())),
            ],
        );
        let named = engine.load_weights(weights, both).unwrap();
        engine
            .add_entry(named, &program)
            .expect("once the map names it the store key binds");
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
        let exe = engine.load_weights(tied_store, tied).unwrap();
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
        let exe = engine.load_weights(untied_store, untied).unwrap();
        engine.add_entry(exe, &program).unwrap();
        assert_eq!(weight_allocations(&engine), 2, "two entries upload twice");
    }

    /// `x @ w^T` for each of `ids`' weights, `[N, K]` each, summed; `x` is an activation slot.
    fn projections(ids: &[WeightId], formats: &WeightFormats) -> Graph {
        let b = Builder::for_tests();
        let x = b
            .slot(Slot::Activation, TensorType::f32(vec![2, K]))
            .unwrap();
        let mut out = None;
        for id in ids {
            let w = b
                .weight(WeightRef::whole(*id), TensorType::f32(vec![N, K]))
                .unwrap();
            let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
            out = Some(match out {
                Some(acc) => b.binary(BinOp::Add, acc, y).unwrap(),
                None => y,
            });
        }
        bind_packed_weights(&b.finish(out.unwrap()).unwrap(), formats).unwrap()
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
            .load_weights(Arc::clone(&weights), Arc::clone(&mapped))
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
        assert_size_only_matches_bytes(&weights, &mapped, &g);
    }

    /// An I64 index table viewed by `IndexNarrow` uploads its words as I32, narrowed from the
    /// stored I64 bytes (negative words included) and range-checked when the map was built.
    ///
    /// Mutation: in `WeightWriter::read_stream`, copy a narrowed span's stored bytes unchanged; the upload
    /// holds the eight-byte words, the declared I32 const is read at the wrong size and this row
    /// goes red.
    #[test]
    fn an_i64_index_table_view_uploads_its_words_as_i32() {
        let words = [3i64, -1, 7, i64::from(i32::MAX)];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let weights = store(vec![(
            "tid2eid",
            WeightEntry::Dense(
                DenseWeight::try_new(DType::I64, vec![2, 2], StoredBytes::from(bytes)).unwrap(),
            ),
        )]);
        let table = WeightId::layer(0, WeightRole::Moe(poot_quant::weights::MoeRole::HashTable));
        let mapped = map(
            &weights,
            vec![(table, WeightView::IndexNarrow("tid2eid".into()))],
        );
        let b = Builder::for_tests();
        let tokens = b
            .slot(Slot::Token, TensorType::new(vec![1], DType::I32))
            .unwrap();
        let rows = b
            .gather(
                b.constant(&table.const_name(), TensorType::new(vec![2, 2], DType::I32))
                    .unwrap(),
                0,
                tokens,
            )
            .unwrap();
        // The gathered words are the output as they are: the table's range is not declared, so no
        // `I32 -> F32` cast of it is provable (POOT-726).
        let g = b.finish(rows).unwrap();
        let mut engine = Engine::new(Recorder::default());
        let program = compile(&engine, &g);
        let exe = engine
            .load_weights(Arc::clone(&weights), Arc::clone(&mapped))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        let narrowed: Vec<u8> = words
            .iter()
            .flat_map(|&w| (w as i32).to_le_bytes())
            .collect();
        assert!(
            written(&engine).iter().any(|bytes| *bytes == narrowed),
            "no upload holds the table as its narrowed I32 words"
        );
        assert_size_only_matches_bytes(&weights, &mapped, &g);
    }

    /// A hash-routing table declared in `[0, experts)` ([`Builder::constant_ranged`]) is a checked
    /// fact: the weight binder reads every stored word (here narrowed from I64, as the checkpoint
    /// stores `tid2eid`) and refuses a table with one outside, naming the word and the range, before
    /// anything uploads. The same graph over an in-range table loads.
    ///
    /// Mutation: skip the `check_range` call in `weight_buffer`; the out-of-range table uploads and
    /// this row goes red (`load` returns Ok where a `WeightRange` refusal is required).
    #[test]
    fn a_ranged_index_table_with_a_word_outside_its_range_is_refused_at_weight_bind() {
        const EXPERTS: usize = 4;
        let table = WeightId::layer(0, WeightRole::Moe(poot_quant::weights::MoeRole::HashTable));
        let load = |words: [i64; 4]| {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let weights = store(vec![(
                "tid2eid",
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::I64, vec![2, 2], StoredBytes::from(bytes)).unwrap(),
                ),
            )]);
            let mapped = map(
                &weights,
                vec![(table, WeightView::IndexNarrow("tid2eid".into()))],
            );
            let b = Builder::for_tests();
            let tokens = b
                .slot(Slot::Token, TensorType::new(vec![1], DType::I32))
                .unwrap();
            let rows = b
                .gather(
                    b.constant_ranged(
                        &table.const_name(),
                        TensorType::new(vec![2, 2], DType::I32),
                        IndexRange::below(EXPERTS).unwrap(),
                    )
                    .unwrap(),
                    0,
                    tokens,
                )
                .unwrap();
            let g = b.finish(rows).unwrap();
            let mut engine = Engine::new(Recorder::default());
            let program = compile(&engine, &g);
            let exe = engine.load_weights(weights, mapped).unwrap();
            engine.add_entry(exe, &program)
        };

        load([0, 3, 2, 1]).expect("a table inside [0, 4) loads");
        let error = load([0, 3, 4, 1]).expect_err("word 4 is outside [0, 4)");
        match &error {
            ExecError::Load(load) => match &**load {
                LoadError::WeightRange {
                    name,
                    index,
                    value,
                    range,
                } => {
                    assert_eq!(name, &table.const_name());
                    assert_eq!((*index, *value), (2, 4));
                    assert_eq!(*range, IndexRange::below(EXPERTS).unwrap());
                }
                other => panic!("expected a weight range refusal, got {other:?}"),
            },
            other => panic!("expected a load refusal, got {other:?}"),
        }
        assert!(
            error.to_string().contains("[0, 4)"),
            "the refusal names the declared range: {error}"
        );
    }

    /// A cast of a gathered ranged table compiles (its words are proven inside the exactly
    /// representable integers of `f32` from the declaration, through the `Gather` rule), and the
    /// same graph over an undeclared table is refused as an unproven cast.
    ///
    /// Mutation: make `Builder::declare_constant` record `None` for a ranged const; the declared
    /// graph is refused and this row goes red naming the cast.
    #[test]
    fn a_cast_of_a_gathered_ranged_table_compiles_and_an_undeclared_one_is_refused() {
        let graph = |range: Option<IndexRange>| {
            let b = Builder::for_tests();
            let tokens = b
                .slot(Slot::Token, TensorType::new(vec![1], DType::I32))
                .unwrap();
            let aval = TensorType::new(vec![2, 2], DType::I32);
            let table = match range {
                Some(range) => b.constant_ranged("hash.table", aval, range).unwrap(),
                None => b.constant("hash.table", aval).unwrap(),
            };
            let out = b
                .cast(b.gather(table, 0, tokens).unwrap(), DType::F32)
                .unwrap();
            b.finish(out).unwrap()
        };
        let engine = Engine::new(Recorder::default());
        let compiled = |g: &Graph| {
            compile_staged(
                g,
                &TargetSet::single(DeviceId(0), engine.device().target()),
                &Partition {
                    experts: ExpertPlacement::AllResident,
                    devices: DevicePlacement::Single(DeviceId(0)),
                },
                &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
            )
        };
        compiled(&graph(Some(IndexRange::below(4).unwrap())))
            .expect("a gathered table declared in [0, 4) casts to F32");
        let Err(refused) = compiled(&graph(None)) else {
            panic!("an undeclared table's cast is refused");
        };
        assert!(
            matches!(
                refused,
                poot_graph_plan::staged::StagedCompileError::Compile(
                    poot_graph_plan::CompileError::CastRange(_)
                )
            ),
            "expected the cast-range refusal, got {refused:?}"
        );
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
        let fused =
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [3 * N, K], 7);
        let parts = [
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [N / 2, K], 8),
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [N / 2, K], 9),
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
            .load_weights(Arc::clone(&weights), Arc::clone(&mapped))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();
        // Golden, recorded from the per-candidate full-load path at 79bbd35479: Q, K, V and the stacked gate
        // are four Q8_0 sources of four 34-byte block rows, 136 bytes = 34 words + 1 pad word each.
        let weight_elems: Vec<usize> = engine
            .device()
            .allocations
            .iter()
            .filter(|a| a.0 == BufferRole::Weight)
            .map(|a| a.2)
            .collect();
        assert_eq!(weight_elems, [35, 35, 35, 35]);
        let writes = written(&engine);
        for (label, want) in [
            ("q", rows(0..N)),
            ("k", rows(N..2 * N)),
            ("v", rows(2 * N..3 * N)),
            ("gate", gate),
        ] {
            // The device buffer is zero-filled by allocation, so an upload writes the stored rows and
            // leaves the transport's zero guard word to the allocation: the bytes written followed by
            // zeros are the whole transport layout (`packed_source_words`).
            assert!(
                writes
                    .iter()
                    .any(|w| want.starts_with(w) && want[w.len()..].iter().all(|&b| b == 0)),
                "{label}: no upload holds its stored block rows"
            );
        }
        assert_size_only_matches_bytes(&weights, &mapped, &g);
    }

    /// An engine whose target's buffer cap is one byte below a `[rows, K]` F32 table, and an embed
    /// gather of `w.embed` by the step's tokens compiled for it: `compile`'s legalize hosts the
    /// gather as a `Slot::TokenEmbed` input.
    fn hosted_embed(rows: usize, tokens: usize) -> (Engine<Recorder>, StagedProgram) {
        let engine = Engine::new(Recorder {
            cap: Some((rows * K * 4 - 1) as u64),
            ..Recorder::default()
        });
        let b = Builder::for_tests();
        let embed = WeightId::model(WeightRole::Embed);
        let table = b
            .weight(WeightRef::whole(embed), TensorType::f32(vec![rows, K]))
            .unwrap();
        let ids = b
            .slot(Slot::Token, TensorType::new(vec![1, tokens], DType::I32))
            .unwrap();
        let out = b.gather(table, 0, ids).unwrap();
        let program = compile(&engine, &b.finish(out).unwrap());
        let (_, _, stage) = program.stages().next().unwrap();
        assert!(
            stage
                .graph()
                .slots()
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
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [3, K], 3),
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [3, K], 4),
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
            let exe = engine.load_weights(weights, mapped).unwrap();
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
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, K]))
            .unwrap();
        let w = b
            .weight(WeightRef::whole(head), TensorType::f32(vec![ROWS, K]))
            .unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let program = compile(&engine, &b.finish(out).unwrap());
        let exe = engine.load_weights(Arc::clone(&weights), mapped).unwrap();
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
            (ValueId::new(0), "w.head.chunk0", shape.as_slice()),
            (ValueId::new(1), "w.head.chunk2", shape.as_slice()),
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
        let embed = map(
            &weights,
            vec![(
                WeightId::model(WeightRole::Embed),
                WeightView::Stored("w.embed".into()),
            )],
        );
        let exe = engine.load_weights(weights, embed).unwrap();
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

    /// POOT-1289: `reset_state(Row(r))` for a row past the carried state's leading axis is a typed
    /// `StateRow` refusal naming the state and its row count, and zeroes nothing (the pool is written only
    /// after every buffer passed).
    ///
    /// Mutation: relax the guard in `Engine::reset_state` from `row >= leading` to `row > leading`; row 4
    /// of a 4-row pool passes the check and this row goes red (`expected StateRow, got Ok(())`).
    #[test]
    fn a_state_row_past_the_pool_is_a_state_row_refusal_that_zeroes_nothing() {
        use poot_graph_ir::StateRole;
        const POOL: usize = 4;
        let mut engine = Engine::new(Recorder::default());
        let b = Builder::for_tests();
        let pool = b
            .state_input("pool", TensorType::f32(vec![POOL, K]), StateRole::Recurrent)
            .unwrap();
        let src = b
            .slot_named(Slot::Activation, "src", TensorType::f32(vec![2, K]))
            .unwrap();
        let slots = b.write_map("write", 2, POOL).unwrap();
        let out = b.scatter_update(pool, src, slots).unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(out).with_state(&[(pool, out)]))
            .unwrap();
        let program = compile(&engine, &graph);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        engine.add_entry(exe, &program).unwrap();

        let writes_before = engine.device().writes.len();
        match engine.reset_state(exe, StateScope::Row(POOL)) {
            Err(ExecError::Bind(error)) => match *error {
                BindError::StateRow { name, row, rows } => {
                    assert_eq!((name.as_str(), row, rows), ("pool", POOL, POOL));
                }
                other => panic!("expected StateRow, got {other}"),
            },
            other => panic!("expected StateRow, got {other:?}"),
        }
        assert_eq!(
            engine.device().writes.len(),
            writes_before,
            "a refused row reset writes nothing"
        );
        engine
            .reset_state(exe, StateScope::Row(POOL - 1))
            .expect("the last row of the pool resets");
        assert!(engine.device().writes.len() > writes_before);
    }

    /// POOT-1289, POOT-1292: `load_weights` takes the store and the map separately, so a map built over one
    /// store can meet another whose packed entry under the same key is a different width. That pairing is
    /// refused at `load_weights` with both handles, so no embed row is decoded into a buffer of the map's
    /// width from a payload of another. The control binds the matching store.
    ///
    /// Mutation: in `WeightMap::check_store`, skip the comparison against the recorded handle; the
    /// mismatched pairing loads and the later bind panics on the decode (`a mapped row of a checked payload
    /// decodes`).
    #[test]
    fn a_packed_embed_of_another_width_than_the_map_is_refused_at_load_weights() {
        let embed = WeightId::model(WeightRole::Embed);
        let packed = |k: usize| {
            WeightEntry::Packed(Arc::new(poot_quant::test_support::packed::random_payload(
                WeightFormat::Q8_0,
                [6, k],
                5,
            )))
        };
        let mapped_over = store(vec![("e", packed(K))]);
        let map = map(
            &mapped_over,
            vec![(embed, WeightView::Stored(WeightKey::from("e")))],
        );
        let step_over = |weights: Arc<WeightStore>| {
            let (mut engine, program) = hosted_embed(6, 1);
            let exe = engine.load_weights(weights, Arc::clone(&map))?;
            let entry = engine.add_entry(exe, &program)?;
            let bytes = 3i32.to_le_bytes();
            let mut inputs = StepInputs::new();
            inputs.push(
                SlotKey::new(Slot::Token, None),
                &[1, 1],
                poot_tensor::HostView::new(DType::I32, 1, &bytes).unwrap(),
            );
            engine.step(exe, entry, &inputs, &mut NoSync).map(drop)
        };
        step_over(mapped_over.clone()).expect("control: the store the map was built over binds");
        match step_over(store(vec![("e", packed(2 * K))])) {
            Err(ExecError::Load(error)) => match *error {
                LoadError::WeightMap(WeightMapError::StoreMismatch { id, mapped, stored }) => {
                    assert_eq!(id, embed);
                    assert_eq!(mapped.shape, vec![6, K]);
                    assert_eq!(stored.shape, vec![6, 2 * K]);
                }
                other => panic!("expected StoreMismatch, got {other}"),
            },
            other => panic!("expected StoreMismatch, got {other:?}"),
        }
    }

    /// POOT-1148: a forward write map is the one input whose words the scatter kernel stores through, so it
    /// binds only as a checked `ForwardSlotMap`. A raw I32 view of the slot is refused by name; a typed map
    /// checked against a larger pool than the graph declares meets the slot's declared range at the bind; a
    /// typed map to a slot that is not a write map is refused; a map of distinct in-pool slots steps.
    ///
    /// Mutation: drop the `UncheckedWriteMap` arm in `bind`: the raw view binds and the first row goes red.
    #[test]
    fn a_write_map_binds_only_as_a_checked_forward_slot_map() {
        use poot_graph_ir::{ForwardSlotMap, StateRole};
        let mut engine = Engine::new(Recorder::default());
        let b = Builder::for_tests();
        let pool = b
            .state_input("pool", TensorType::f32(vec![4, K]), StateRole::Recurrent)
            .unwrap();
        let src = b
            .slot_named(Slot::Activation, "src", TensorType::f32(vec![2, K]))
            .unwrap();
        let slots = b.write_map("write", 2, 4).unwrap();
        let out = b.scatter_update(pool, src, slots).unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(out).with_state(&[(pool, out)]))
            .unwrap();
        let program = compile(&engine, &graph);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();

        let rows = [0u8; 2 * K * 4];
        let src_key = SlotKey::new(Slot::Activation, Some("src"));
        let write_key = SlotKey::new(Slot::WriteMap, Some("write"));
        let words: Vec<u8> = [3i32, -1].iter().flat_map(|w| w.to_le_bytes()).collect();
        let src_shape = [2, K];
        let src_view = poot_tensor::HostView::new(DType::F32, 2 * K, &rows).unwrap();
        let mut refusal =
            |inputs: &StepInputs<'_>| match engine.step(exe, entry, inputs, &mut NoSync) {
                Err(ExecError::Bind(error)) => Some(*error),
                Err(other) => panic!("expected a bind refusal, got {other}"),
                Ok(_) => None,
            };

        let mut raw = StepInputs::new();
        raw.push(src_key.clone(), &src_shape, src_view);
        raw.push(
            write_key.clone(),
            &[2],
            poot_tensor::HostView::new(DType::I32, 2, &words).unwrap(),
        );
        assert!(
            matches!(refusal(&raw), Some(BindError::UncheckedWriteMap { key }) if key == write_key),
            "a raw view of a write map must be refused"
        );

        let wide = ForwardSlotMap::new(vec![5, -1], 8).unwrap();
        let mut over_declared = StepInputs::new();
        over_declared.push(src_key.clone(), &src_shape, src_view);
        over_declared.push_write_map(write_key.clone(), &wide);
        assert!(
            matches!(
                refusal(&over_declared),
                Some(BindError::Range {
                    index: 0,
                    value: 5,
                    ..
                })
            ),
            "a map checked against a larger pool meets the slot's declared range"
        );

        let ok = ForwardSlotMap::new(vec![3, -1], 4).unwrap();
        let mut misplaced = StepInputs::new();
        misplaced.push(src_key.clone(), &src_shape, src_view);
        misplaced.push_write_map(write_key.clone(), &ok);
        misplaced.push_write_map(SlotKey::new(Slot::Activation, Some("src")), &ok);
        assert!(
            matches!(refusal(&misplaced), Some(BindError::Duplicate { .. })),
            "a second bind of the src slot is a duplicate"
        );
        let mut wrong_kind = StepInputs::new();
        wrong_kind.push_write_map(write_key.clone(), &ok);
        wrong_kind.push_write_map(src_key.clone(), &ok);
        assert!(
            matches!(refusal(&wrong_kind), Some(BindError::NotAWriteMap { key }) if key == src_key),
            "a checked map binds only a write-map slot"
        );

        let mut good = StepInputs::new();
        good.push(src_key, &src_shape, src_view);
        good.push_write_map(write_key, &ok);
        assert!(refusal(&good).is_none(), "distinct in-pool slots step");
    }

    /// POOT-1268: the host seeds one row of a carried state it then leaves to the device. The row lands at
    /// its own byte offset of the state's buffer; a name the executable does not hold, a row past the
    /// leading axis and a row of another dtype or length are typed refusals that write nothing; every step
    /// is one host wait and a step output read is counted by its bytes.
    ///
    /// Mutation: write the row at offset 0 in `write_state_row`; the offset assertion goes red.
    #[test]
    fn the_host_seeds_one_row_of_a_carried_state() {
        use poot_graph_ir::StateRole;
        const POOL: usize = 3;
        const WIDTH: usize = 4;
        let mut engine = Engine::new(Recorder::default());
        let b = Builder::for_tests();
        let history = b
            .state_input(
                "sampler.history",
                TensorType::f32(vec![POOL, 2, WIDTH]),
                StateRole::Sampler,
            )
            .unwrap();
        let out = b
            .binary_scalar(
                poot_graph_ir::BinOp::Add,
                history,
                poot_graph_ir::Scalar::F32(1.0),
            )
            .unwrap();
        let graph = b
            .finish(poot_graph_ir::Outputs::new(out).with_state(&[(history, out)]))
            .unwrap();
        let program = compile(&engine, &graph);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();

        let row: Vec<u8> = (0..2 * WIDTH)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let view = poot_tensor::HostView::new(DType::F32, 2 * WIDTH, &row).unwrap();
        let before = engine.device().writes.len();
        engine
            .write_state_row(exe, "sampler.history", 2, view)
            .unwrap();
        let device = engine.device();
        assert_eq!(device.writes.len(), before + 1, "one write");
        assert_eq!(device.writes[before].1, row, "the row's own bytes");
        assert_eq!(
            device.write_offsets[before],
            2 * 2 * WIDTH * 4,
            "pool row 2 starts after two rows of two channels of {WIDTH} floats"
        );

        let refused = |engine: &mut Engine<Recorder>, name: &str, at: usize, view| {
            let before = engine.device().writes.len();
            let error = engine
                .write_state_row(exe, name, at, view)
                .expect_err("a refused write");
            assert_eq!(
                engine.device().writes.len(),
                before,
                "a refusal writes nothing"
            );
            match error {
                ExecError::Bind(error) => *error,
                other => panic!("expected a bind refusal, got {other}"),
            }
        };
        assert!(matches!(
            refused(&mut engine, "sampler.nothing", 0, view),
            BindError::UnknownState { .. }
        ));
        assert!(matches!(
            refused(&mut engine, "sampler.history", POOL, view),
            BindError::StateRow { rows: POOL, .. }
        ));
        let short = poot_tensor::HostView::new(DType::F32, WIDTH, &row[..WIDTH * 4]).unwrap();
        assert!(matches!(
            refused(&mut engine, "sampler.history", 0, short),
            BindError::StateRowValues { .. }
        ));
        let words = poot_tensor::HostView::new(DType::I32, 2 * WIDTH, &row).unwrap();
        assert!(matches!(
            refused(&mut engine, "sampler.history", 0, words),
            BindError::StateRowValues { .. }
        ));

        let waits = engine.stats().waits;
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
        assert_eq!(engine.stats().waits, waits + 2, "a step waits once");
    }

    /// A late-input fixture: `y = x + 1` (exported), `out = y * 2 + late`, over F32 `[2]` slots `x` and
    /// `late`. Returns the graph and the exported boundary.
    fn late_graph() -> (Graph, poot_graph_ir::BoundaryId) {
        use poot_graph_ir::{BinOp, Outputs, Scalar};
        let b = Builder::for_tests();
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![2]))
            .unwrap();
        let late = b
            .slot_named(Slot::Activation, "late", TensorType::f32(vec![2]))
            .unwrap();
        let y = b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0)).unwrap();
        let out = b
            .binary(
                BinOp::Add,
                b.binary_scalar(BinOp::Mul, y, Scalar::F32(2.0)).unwrap(),
                late,
            )
            .unwrap();
        let boundary = poot_graph_ir::BoundaryId(y.id());
        let graph = b
            .finish(Outputs::new(out).with_export(boundary, y))
            .unwrap();
        (graph, boundary)
    }

    /// The late values of [`late_graph`], in the order the device events show them.
    struct LateMask {
        keys: [SlotKey; 1],
        values: usize,
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    impl crate::LateInputs for LateMask {
        fn keys(&self) -> &[SlotKey] {
            &self.keys
        }

        fn build(&mut self) -> Result<Vec<poot_tensor::HostTensor>, ExecError> {
            self.log.lock().unwrap().push("build");
            Ok(vec![
                poot_tensor::HostTensor::f32(vec![2], vec![0.5, 0.25]);
                self.values
            ])
        }
    }

    fn step_late_log(late_cut: bool, values: usize) -> (Result<(), ExecError>, Vec<&'static str>) {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine = Engine::new(Recorder {
            log: Some(Arc::clone(&log)),
            ..Recorder::default()
        });
        let (graph, boundary) = late_graph();
        let mut options = CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
        if late_cut {
            options = options.with_late(boundary);
        }
        let program = compile_staged(
            &graph,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &options,
        )
        .expect("the fixture compiles");
        assert_eq!(program.late_after().is_some(), late_cut);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();
        let x: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut inputs = StepInputs::new();
        inputs.push(
            SlotKey::new(Slot::Activation, Some("x")),
            &[2],
            poot_tensor::HostView::new(DType::F32, 2, &x).unwrap(),
        );
        let mut late = LateMask {
            keys: [SlotKey::new(Slot::Activation, Some("late"))],
            values,
            log: Arc::clone(&log),
        };
        log.lock().unwrap().clear();
        let result = engine
            .step_late(exe, entry, &inputs, &mut late, &mut NoSync)
            .map(|_| ());
        let events = log.lock().unwrap().clone();
        (result, events)
    }

    /// POOT-1268: a program cut after the stage that produces a value builds its late inputs once the
    /// first stage is on the device and writes them before the second is enqueued: the device sees the
    /// ordinary input written, the first stage replayed, the late input built and written, the second stage
    /// replayed, one wait. A program with no cut binds them before anything runs. The late values are the
    /// same either way; only the overlap differs.
    ///
    /// Mutation: bind the late inputs before the first stage in `replay_entry` whatever the program; the
    /// cut program's events lose the replay in front of the late write.
    #[test]
    fn late_inputs_are_built_between_the_stages_of_a_cut_program() {
        let (result, events) = step_late_log(true, 1);
        result.unwrap();
        assert_eq!(
            events,
            ["write", "replay", "build", "write", "replay", "synchronize"],
            "the late value is built and written between the two stages"
        );
        let (result, events) = step_late_log(false, 1);
        result.unwrap();
        assert_eq!(
            events,
            ["write", "build", "write", "replay", "synchronize"],
            "an uncut program binds its late value before the one stage runs"
        );
    }

    /// A late input that builds a different number of values than it names keys is a typed refusal, not a
    /// partial bind.
    #[test]
    fn late_inputs_that_build_the_wrong_number_of_values_are_refused() {
        let (result, _) = step_late_log(true, 0);
        match result {
            Err(ExecError::Bind(error)) => {
                assert!(
                    matches!(*error, BindError::LateValues { keys: 1, values: 0 }),
                    "{error}"
                );
            }
            other => panic!("expected LateValues, got {other:?}"),
        }
    }

    /// An engine, an executable, two entries, the feed between them and the device event log.
    type ChainFixture = (
        Engine<Recorder>,
        ExecutableId,
        [EntryId; 2],
        crate::ChainFeed,
        Arc<std::sync::Mutex<Vec<&'static str>>>,
    );

    /// A chain fixture: `next = tok + 1` over an I32 `[2, 1]` token slot, exported as the picked token and
    /// returned. Two entries of it, the second fed the first's pick. Returns the engine, the executable,
    /// the entries, the feed and the event log.
    fn chain_fixture() -> ChainFixture {
        use poot_graph_ir::{BinOp, Outputs, Scalar};
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine = Engine::new(Recorder {
            log: Some(Arc::clone(&log)),
            ..Recorder::default()
        });
        let b = Builder::for_tests();
        let tok = b
            .slot_named(Slot::Token, "tok", TensorType::new(vec![2, 1], DType::I32))
            .unwrap();
        let next = b.binary_scalar(BinOp::Add, tok, Scalar::I32(1)).unwrap();
        let picked = poot_graph_ir::BoundaryId(next.id());
        let graph = b
            .finish(Outputs::new(next).with_export(picked, next))
            .unwrap();
        let program = compile(&engine, &graph);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        let entries = [
            engine.add_entry(exe, &program).unwrap(),
            engine.add_entry(exe, &program).unwrap(),
        ];
        let feed = crate::ChainFeed {
            picked,
            slot: SlotKey::new(Slot::Token, Some("tok")),
        };
        (engine, exe, entries, feed, log)
    }

    /// POOT-1268: a chain enqueues step 0, the copy of its picked token into step 1's token slot, then
    /// step 1, behind one completion wait, with the first step's token the only input written; the chain
    /// is refused before any device work when it is empty, names an entry twice, or its entries export no
    /// picked token or need the host between their stages.
    ///
    /// Mutation: complete after each step in `step_chain`; the wait count is no longer one.
    #[test]
    fn a_chain_enqueues_every_step_and_waits_once() {
        let (mut engine, exe, entries, feed, log) = chain_fixture();
        let tokens: Vec<u8> = [4i32, 9].iter().flat_map(|t| t.to_le_bytes()).collect();
        let shape = [2usize, 1];
        let mut first = StepInputs::new();
        first.push(
            feed.slot.clone(),
            &shape,
            poot_tensor::HostView::new(DType::I32, 2, &tokens).unwrap(),
        );
        let second = StepInputs::new();
        let steps = [
            crate::ChainStep {
                entry: entries[0],
                inputs: &first,
            },
            crate::ChainStep {
                entry: entries[1],
                inputs: &second,
            },
        ];
        let waits = engine.stats().waits;
        log.lock().unwrap().clear();
        let outputs = engine.step_chain(exe, &steps, &feed, &mut NoSync).unwrap();
        assert_eq!(outputs.len(), 2);
        drop(outputs);
        assert_eq!(
            *log.lock().unwrap(),
            ["write", "replay", "replay", "replay", "synchronize"],
            "write the first token; step 0; the copy into step 1; step 1; one wait"
        );
        assert_eq!(engine.stats().waits - waits, 1);
        assert_eq!(engine.stats().replays, 2, "both entries counted a replay");

        let refused = |engine: &mut Engine<Recorder>, steps: &[crate::ChainStep<'_, '_>], feed| {
            let before = log.lock().unwrap().len();
            let error = match engine.step_chain(exe, steps, feed, &mut NoSync) {
                Err(ExecError::Chain(error)) => *error,
                Err(other) => panic!("expected a chain refusal, got {other}"),
                Ok(_) => panic!("the chain was admitted"),
            };
            assert_eq!(
                log.lock().unwrap().len(),
                before,
                "a refusal did no device work"
            );
            error
        };
        assert!(matches!(
            refused(&mut engine, &[], &feed),
            crate::ChainError::Empty
        ));
        let twice = [
            crate::ChainStep {
                entry: entries[0],
                inputs: &first,
            },
            crate::ChainStep {
                entry: entries[0],
                inputs: &second,
            },
        ];
        assert!(matches!(
            refused(&mut engine, &twice, &feed),
            crate::ChainError::RepeatedEntry(_)
        ));
        let unexported = crate::ChainFeed {
            picked: poot_graph_ir::BoundaryId(poot_graph_ir::ValueId::new(99)),
            ..feed.clone()
        };
        assert!(matches!(
            refused(&mut engine, &steps, &unexported),
            crate::ChainError::NoFeed { .. }
        ));
    }

    /// POOT-1149: the paged attention kernel reads its pools through the read map without a bound check, so a
    /// map word outside the pool is refused at the bind, naming the slot, the word and its index, before
    /// anything dispatches; the same step with every word in the pool runs.
    ///
    /// Mutation: declare the read map unranged in `Builder::read_map` (`slot_named` instead of
    /// `slot_named_ranged`): the word-4 step binds and the first assertion goes red with "expected a typed
    /// range refusal".
    #[test]
    fn a_paged_read_map_word_outside_the_pool_is_a_typed_bind_refusal() {
        use poot_graph_ir::{ForwardSlotMap, StateRole};
        const SLOTS: usize = 4;
        let mut engine = Engine::new(Recorder::default());
        let b = Builder::for_tests();
        let pool = |name: &str| {
            b.state_input(
                name,
                TensorType::f32(vec![SLOTS, 2, 4]),
                StateRole::Positional { axis: 0 },
            )
            .unwrap()
        };
        let (k_pool, v_pool) = (pool("k_pool"), pool("v_pool"));
        let activation = |name: &str, shape: Vec<usize>| {
            b.slot_named(Slot::Activation, name, TensorType::f32(shape))
                .unwrap()
        };
        let write = b.write_map("write", 1, SLOTS).unwrap();
        let read = b.read_map("read", 1, 3, SLOTS).unwrap();
        let k_written = b
            .scatter_update(k_pool, activation("new_k", vec![1, 2, 4]), write)
            .unwrap();
        let v_written = b
            .scatter_update(v_pool, activation("new_v", vec![1, 2, 4]), write)
            .unwrap();
        let k = poot_graph_ir::ops::paged_view(&b, k_written, read).unwrap();
        let v = poot_graph_ir::ops::paged_view(&b, v_written, read).unwrap();
        let q = activation("q", vec![1, 2, 1, 4]);
        let mask = activation("mask", vec![1, 1, 1, 3]);
        let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, 0.5, mask).unwrap();
        let graph = b
            .finish(
                poot_graph_ir::Outputs::new(out)
                    .with_state(&[(k_pool, k_written), (v_pool, v_written)]),
            )
            .unwrap();
        let program = compile(&engine, &graph);
        let weights = store(vec![]);
        let exe = engine
            .load_weights(Arc::clone(&weights), map(&weights, vec![]))
            .unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();

        let floats = |n: usize| vec![0u8; n * 4];
        let (new_rows, q_row, mask_row) = (floats(8), floats(8), floats(3));
        let words =
            |words: [i32; 3]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let write_map = ForwardSlotMap::new(vec![2], SLOTS).unwrap();
        let read_key = SlotKey::new(Slot::SlotMap, Some("read"));
        let mut step = |read_words: [i32; 3]| {
            let read_bytes = words(read_words);
            fn f32_view(bytes: &[u8]) -> poot_tensor::HostView<'_> {
                poot_tensor::HostView::new(DType::F32, bytes.len() / 4, bytes).unwrap()
            }
            let mut inputs = StepInputs::new();
            inputs.push(
                SlotKey::new(Slot::Activation, Some("new_k")),
                &[1, 2, 4],
                f32_view(&new_rows),
            );
            inputs.push(
                SlotKey::new(Slot::Activation, Some("new_v")),
                &[1, 2, 4],
                f32_view(&new_rows),
            );
            inputs.push(
                SlotKey::new(Slot::Activation, Some("q")),
                &[1, 2, 1, 4],
                f32_view(&q_row),
            );
            inputs.push(
                SlotKey::new(Slot::Activation, Some("mask")),
                &[1, 1, 1, 3],
                f32_view(&mask_row),
            );
            inputs.push_write_map(SlotKey::new(Slot::WriteMap, Some("write")), &write_map);
            inputs.push(
                read_key.clone(),
                &[1, 3],
                poot_tensor::HostView::new(DType::I32, 3, &read_bytes).unwrap(),
            );
            engine.step(exe, entry, &inputs, &mut NoSync).map(|_| ())
        };
        match step([0, SLOTS as i32, 1]) {
            Err(ExecError::Bind(error)) => assert!(
                matches!(
                    *error,
                    BindError::Range { ref key, index: 1, value, .. }
                        if *key == read_key && value == SLOTS as i32
                ),
                "{error:?}"
            ),
            other => panic!("expected a typed range refusal, got {other:?}"),
        }
        match step([0, -1, 1]) {
            Err(ExecError::Bind(error)) => assert!(
                matches!(
                    *error,
                    BindError::Range {
                        index: 1,
                        value: -1,
                        ..
                    }
                ),
                "{error:?}"
            ),
            other => panic!("expected a typed range refusal, got {other:?}"),
        }
        assert!(
            step([0, SLOTS as i32 - 1, 1]).is_ok(),
            "every word in the pool steps"
        );
    }

    const EXPERTS: usize = 6;
    const POOL: usize = 3;
    const WIDTH: usize = 4;
    const INTER: usize = 2;

    fn moe_roles(layer: usize) -> [WeightId; 3] {
        use poot_quant::weights::{ExpertProj, MoeRole};
        [
            WeightId::layer(layer, WeightRole::Moe(MoeRole::Router)),
            WeightId::layer(layer, WeightRole::Moe(MoeRole::Stack(ExpertProj::GateUp))),
            WeightId::layer(layer, WeightRole::Moe(MoeRole::Stack(ExpertProj::Down))),
        ]
    }

    /// `layers` routed decode layers over a one-token activation `x`.
    fn routed_decode(layers: usize) -> Graph {
        routed_decode_over(&(0..layers).map(|layer| vec![layer]).collect::<Vec<_>>())
    }

    /// One layer per entry of `layers`, each holding one routed block per weight layer it names: layer
    /// position `p` runs the tables of weight layers `layers[p]`, so the tables a position pages need
    /// not be the ones its number suggests, and one layer can hold two sites.
    fn routed_decode_over(layers: &[Vec<usize>]) -> Graph {
        let b = Builder::for_tests();
        let mut x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(vec![1, 1, WIDTH]))
            .unwrap();
        for (position, weights) in layers.iter().enumerate() {
            let _scope = b.layer_scope(position);
            for &layer in weights {
                let [router, gate_up, down] = moe_roles(layer);
                let router = b
                    .constant(&router.const_name(), TensorType::f32(vec![WIDTH, EXPERTS]))
                    .unwrap();
                let gate_up = b
                    .constant(
                        &gate_up.const_name(),
                        TensorType::f32(vec![EXPERTS, WIDTH, 2 * INTER]),
                    )
                    .unwrap();
                let down = b
                    .constant(
                        &down.const_name(),
                        TensorType::f32(vec![EXPERTS, INTER, WIDTH]),
                    )
                    .unwrap();
                let y = poot_graph_ir::ops::moe(&b, x, router, gate_up, down, EXPERTS, 2, INTER)
                    .unwrap();
                x = b.binary(BinOp::Add, x, y).unwrap();
            }
        }
        b.finish(x).unwrap()
    }

    fn routed_weights(layers: usize) -> (Arc<WeightStore>, Arc<WeightMap>) {
        let mut entries = Vec::new();
        let mut views = Vec::new();
        for layer in 0..layers {
            let [router, gate_up, down] = moe_roles(layer);
            let shapes = [
                vec![WIDTH, EXPERTS],
                vec![EXPERTS, WIDTH, 2 * INTER],
                vec![EXPERTS, INTER, WIDTH],
            ];
            for (n, (id, shape)) in [router, gate_up, down].into_iter().zip(shapes).enumerate() {
                let key = id.const_name();
                entries.push((key.clone(), f32_entry(shape, (layer * 3 + n + 1) as u32)));
                views.push((id, WeightView::Stored(WeightKey::from(key.as_str()))));
            }
        }
        let weights = store(
            entries
                .iter()
                .map(|(key, entry)| (key.as_str(), entry.clone()))
                .collect(),
        );
        let mapped = map(&weights, views);
        (weights, mapped)
    }

    fn compile_pooled(engine: &Engine<Recorder>, g: &Graph) -> StagedProgram {
        compile_staged(
            &g.clone(),
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::Pooled {
                    slots_per_layer: POOL,
                },
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("the routed fixture pools")
    }

    /// A host that logs each call into the device's own event log and answers every sync with `writes`.
    struct LoggingHost {
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
        calls: Vec<(poot_graph_plan::staged::SyncId, crate::SiteIdentity)>,
        writes: crate::SyncWrites,
    }

    impl HostSync for LoggingHost {
        fn sync(
            &mut self,
            point: &poot_graph_plan::staged::SyncPoint,
            reads: &crate::SyncReads,
        ) -> Result<crate::SyncWrites, ExecError> {
            self.log.lock().unwrap().push("host");
            self.calls.push((point.id, reads.site().clone()));
            Ok(self.writes.clone())
        }
    }

    fn pooled_engine(
        layers: usize,
    ) -> (
        Engine<Recorder>,
        ExecutableId,
        EntryId,
        Arc<std::sync::Mutex<Vec<&'static str>>>,
    ) {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine = Engine::new(Recorder {
            log: Some(Arc::clone(&log)),
            ..Recorder::default()
        });
        let program = compile_pooled(&engine, &routed_decode(layers));
        let (weights, mapped) = routed_weights(layers);
        let exe = engine.load_weights(weights, mapped).unwrap();
        let entry = engine.add_entry(exe, &program).unwrap();
        (engine, exe, entry, log)
    }

    fn step_x(
        engine: &mut Engine<Recorder>,
        exe: ExecutableId,
        entry: EntryId,
        host: &mut dyn HostSync,
    ) -> Result<(), ExecError> {
        let bytes: Vec<u8> = (0..WIDTH)
            .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
            .collect();
        let mut inputs = StepInputs::new();
        inputs.push(
            SlotKey::new(Slot::Activation, Some("x")),
            &[1, 1, WIDTH],
            poot_tensor::HostView::new(DType::F32, WIDTH, &bytes).unwrap(),
        );
        engine.step(exe, entry, &inputs, host).map(|_| ())
    }

    /// SC-004 (hook order and count): a step of a pooled program calls the host exactly once per sync
    /// point, each time after the stage before it was replayed and drained, and before the next stage
    /// replays. The pool tables and maps are resident, so the step binds no input for them and that is not an
    /// error.
    ///
    /// Mutation: in `Engine::run_sync` call `host.sync` before `self.device.synchronize()`; the log shows
    /// the host before the drain and this row goes red.
    #[test]
    fn a_pooled_step_calls_the_host_once_per_sync_point_after_its_stage_completes() {
        let (mut engine, exe, entry, log) = pooled_engine(2);
        let mut host = LoggingHost {
            log: Arc::clone(&log),
            calls: Vec::new(),
            writes: crate::SyncWrites::default(),
        };
        log.lock().unwrap().clear();
        step_x(&mut engine, exe, entry, &mut host).expect("the step runs with no resident input");
        assert_eq!(
            host.calls.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            [0, 1],
            "one call per sync point, in order"
        );
        assert_ne!(
            host.calls[0].1, host.calls[1].1,
            "two layers' tables are two canonical sites"
        );
        let events = log.lock().unwrap().clone();
        assert_eq!(
            events,
            [
                "write",
                "replay",
                "synchronize",
                "host",
                "replay",
                "synchronize",
                "host",
                "replay",
                "synchronize"
            ],
            "each host call follows the drain of its stage and precedes the next replay"
        );
        let stats = engine.stats();
        assert_eq!(stats.syncs, 2);
        assert!(stats.sync_read_bytes > 0);
    }

    /// SC-004 (range): a refill whose rows run past the pool is `BindError::RowRange`, refused before any
    /// byte is written, and the executable then needs a reset.
    ///
    /// Mutation: delete the `end > pool_rows` check in `Engine::apply_sync_writes`; the write is applied
    /// (the recorder accepts it) and this row goes red.
    #[test]
    fn a_row_write_at_the_end_of_the_pool_is_a_row_range_refusal() {
        let (mut engine, exe, entry, log) = pooled_engine(1);
        let mut host = LoggingHost {
            log,
            calls: Vec::new(),
            writes: crate::SyncWrites {
                rows: vec![crate::RowFill {
                    table: 0,
                    first_row: POOL,
                    experts: vec![0],
                }],
                map: None,
            },
        };
        let written_before = engine.device().writes.len();
        match step_x(&mut engine, exe, entry, &mut host) {
            Err(ExecError::Bind(error)) => match *error {
                BindError::RowRange {
                    first_row,
                    end,
                    pool_rows,
                    ..
                } => assert_eq!((first_row, end, pool_rows), (POOL, POOL + 1, POOL)),
                other => panic!("expected RowRange, got {other}"),
            },
            other => panic!("expected RowRange, got {:?}", other.map(|_| ())),
        }
        assert_eq!(
            engine.device().writes.len(),
            written_before + 1,
            "only the step's own input write landed, nothing of the refused refill"
        );
        let mut quiet = LoggingHost {
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            calls: Vec::new(),
            writes: crate::SyncWrites::default(),
        };
        assert!(matches!(
            step_x(&mut engine, exe, entry, &mut quiet),
            Err(ExecError::NeedsReset(_))
        ));
    }

    /// A sync's writes land before the next stage replays: the log reads input, replay, drain, host,
    /// row write, map write, replay, drain.
    ///
    /// Mutation: in `Engine::replay_stages` apply the writes after the next stage's replay (defer the
    /// call in `run_sync`); the rows land after the stage that reads them and this row goes red.
    #[test]
    fn a_syncs_writes_land_before_the_next_stage_replays() {
        let (mut engine, exe, entry, log) = pooled_engine(1);
        let mut host = LoggingHost {
            log: Arc::clone(&log),
            calls: Vec::new(),
            writes: crate::SyncWrites {
                rows: vec![crate::RowFill {
                    table: 0,
                    first_row: 0,
                    experts: vec![3],
                }],
                map: Some(vec![-1; EXPERTS]),
            },
        };
        log.lock().unwrap().clear();
        step_x(&mut engine, exe, entry, &mut host).unwrap();
        let events = log.lock().unwrap().clone();
        assert_eq!(
            events,
            [
                "write",
                "replay",
                "synchronize",
                "host",
                "write",
                "write",
                "replay",
                "synchronize"
            ]
        );
    }

    /// A refill lands the stored bytes of the named expert in the named row, in the table's lane, and
    /// the map lands whole.
    ///
    /// Mutation: in `apply_sync_writes` read the source from `row * row_bytes` instead of
    /// `expert * row_bytes`; the written row is expert 1's bytes and this row goes red.
    #[test]
    fn a_refill_writes_the_experts_stored_bytes_into_its_row() {
        let (mut engine, exe, entry, log) = pooled_engine(1);
        let mut map = vec![-1i32; EXPERTS];
        map[4] = 1;
        let mut host = LoggingHost {
            log,
            calls: Vec::new(),
            writes: crate::SyncWrites {
                rows: vec![crate::RowFill {
                    table: 0,
                    first_row: 1,
                    experts: vec![4],
                }],
                map: Some(map.clone()),
            },
        };
        step_x(&mut engine, exe, entry, &mut host).unwrap();
        let [_, gate_up, _] = moe_roles(0);
        let (weights, _) = routed_weights(1);
        let WeightEntry::Dense(table) = weights.get(&gate_up.const_name()).unwrap() else {
            unreachable!("dense fixture")
        };
        let per_expert = WIDTH * 2 * INTER * 4;
        let want = &table.bytes().as_slice()[4 * per_expert..5 * per_expert];
        let map_bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
        let writes = written(&engine);
        assert!(
            writes.contains(&want),
            "expert 4's stored bytes were written"
        );
        assert!(
            writes.contains(&map_bytes.as_slice()),
            "the map was written"
        );
        assert_eq!(engine.stats().pool_rows_written, 1);
        assert_eq!(engine.stats().pool_refill_bytes, per_expert as u64);
    }

    /// SC-007: a site's identity is its tables' stored bytes, not its number. Two entries that run the
    /// same tables in a different order report the same identities (permuted); two sites of one
    /// layer report different ones; a second executable over a same-named store of another generation
    /// reports none of them. Entries over one store share the pool buffers: the second entry
    /// allocates none.
    ///
    /// Mutation: in `Engine::pool_table` key the identity by `pool_rows` and lane only (drop `runs`); the
    /// two layers' identities coincide and this row goes red.
    #[test]
    fn a_site_is_identified_by_its_tables_bytes_across_entries_and_owners() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine = Engine::new(Recorder {
            log: Some(Arc::clone(&log)),
            ..Recorder::default()
        });
        let compile = |engine: &Engine<Recorder>, g: &Graph| compile_pooled(engine, g);
        let identities = |engine: &mut Engine<Recorder>, exe, entry| {
            let mut host = LoggingHost {
                log: Arc::clone(&log),
                calls: Vec::new(),
                writes: crate::SyncWrites::default(),
            };
            step_x(engine, exe, entry, &mut host).unwrap();
            host.calls
                .into_iter()
                .map(|(_, identity)| identity)
                .collect::<Vec<_>>()
        };
        let (weights, mapped) = routed_weights(2);
        let exe = engine.load_weights(weights, mapped).unwrap();
        let forward = compile(&engine, &routed_decode_over(&[vec![0], vec![1]]));
        let entry_a = engine.add_entry(exe, &forward).unwrap();
        let weight_allocations_after_a = weight_allocations(&engine);
        let swapped = compile(&engine, &routed_decode_over(&[vec![1], vec![0]]));
        let entry_b = engine.add_entry(exe, &swapped).unwrap();
        assert_eq!(
            weight_allocations(&engine),
            weight_allocations_after_a,
            "the second entry pages the same tables: it allocates no pool or map"
        );
        let a = identities(&mut engine, exe, entry_a);
        let b = identities(&mut engine, exe, entry_b);
        assert_ne!(a[0], a[1], "two layers' tables are two sites");
        assert_eq!(
            (b[0].clone(), b[1].clone()),
            (a[1].clone(), a[0].clone()),
            "the permuted entry reports the same sites in its own order"
        );

        // Two sites of one layer are two identities.
        let two = compile(&engine, &routed_decode_over(&[vec![0, 1]]));
        let entry_two = engine.add_entry(exe, &two).unwrap();
        let in_one_layer = identities(&mut engine, exe, entry_two);
        assert_eq!(in_one_layer.len(), 2);
        assert_ne!(in_one_layer[0], in_one_layer[1]);
        assert_eq!(in_one_layer, a, "the same tables, the same two sites");

        // A new owner of the same names: another store generation, nothing shared.
        let (other_weights, other_mapped) = routed_weights(2);
        let other = engine.load_weights(other_weights, other_mapped).unwrap();
        let entry_c = engine.add_entry(other, &forward).unwrap();
        let c = identities(&mut engine, other, entry_c);
        assert_eq!(c.len(), 2);
        for (new, old) in c.iter().zip(&a) {
            assert_ne!(
                new, old,
                "a same-named weight of another owner is another site"
            );
        }
    }

    /// A [`Recorder`] whose `n`th host write after arming fails: a refill that dies between rows.
    #[derive(Default)]
    struct FailingWrites {
        inner: Recorder,
        /// Writes still allowed before the failing one; `None` is disarmed.
        until_failure: Option<usize>,
    }

    #[derive(Debug)]
    struct WriteFault;

    impl std::fmt::Display for WriteFault {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("injected write fault")
        }
    }

    impl std::error::Error for WriteFault {}

    impl Device for FailingWrites {
        type Buffer = usize;
        type Kernel = ();
        type Recording = ();
        type Error = WriteFault;

        fn target(&self) -> poot_graph_plan::Target {
            self.inner.target()
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            self.inner.memory()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            storage: BufferStorage,
            elems: usize,
        ) -> Result<usize, WriteFault> {
            Ok(self.inner.allocate(role, storage, elems).unwrap())
        }
        fn buffer_elems(&self, buffer: &usize) -> usize {
            self.inner.buffer_elems(buffer)
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), WriteFault> {
            Ok(())
        }
        fn begin(&mut self) -> Result<(), WriteFault> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), WriteFault> {
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), WriteFault> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), WriteFault> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), WriteFault> {
            Ok(())
        }
        fn write_with(
            &mut self,
            dst: &usize,
            offset: usize,
            len: usize,
            fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), WriteFault> {
            match &mut self.until_failure {
                Some(0) => {
                    self.until_failure = None;
                    return Err(WriteFault);
                }
                Some(left) => *left -= 1,
                None => {}
            }
            self.inner.write_with(dst, offset, len, fill).unwrap();
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), WriteFault> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), WriteFault> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), WriteFault> {
            Ok(())
        }
    }

    struct FixedHost(crate::SyncWrites);

    impl HostSync for FixedHost {
        fn sync(
            &mut self,
            _point: &poot_graph_plan::staged::SyncPoint,
            _reads: &crate::SyncReads,
        ) -> Result<crate::SyncWrites, ExecError> {
            Ok(self.0.clone())
        }
    }

    /// SC-006: a refill that fails after some of its row writes publishes no map, the executable
    /// refuses the next step, and after a complete reset the same refill writes exactly what a fresh
    /// engine writes for it.
    ///
    /// Mutation: in `apply_sync_writes` write the map before the rows; the failed refill has already
    /// published the new mapping and this row goes red.
    #[test]
    fn a_refill_that_fails_midway_publishes_no_map_and_needs_a_complete_reset() {
        let fill = crate::SyncWrites {
            rows: (0..POOL)
                .map(|row| crate::RowFill {
                    table: 0,
                    first_row: row,
                    experts: vec![row + 1],
                })
                .collect(),
            map: Some(vec![-1, 0, 1, 2, -1, -1]),
        };
        let map_bytes: Vec<u8> = fill
            .map
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let load = || {
            let mut engine = Engine::new(FailingWrites::default());
            let compiled = {
                // Compile on a recorder-shaped target: the wrapper reports the recorder's.
                compile_staged(
                    &routed_decode(1),
                    &TargetSet::single(DeviceId(0), engine.device().target()),
                    &Partition {
                        experts: ExpertPlacement::Pooled {
                            slots_per_layer: POOL,
                        },
                        devices: DevicePlacement::Single(DeviceId(0)),
                    },
                    &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
                )
                .expect("the routed fixture pools")
            };
            let (weights, mapped) = routed_weights(1);
            let exe = engine.load_weights(weights, mapped).unwrap();
            let entry = engine.add_entry(exe, &compiled).unwrap();
            (engine, exe, entry)
        };
        let step = |engine: &mut Engine<FailingWrites>, exe, entry, host: &mut dyn HostSync| {
            let bytes: Vec<u8> = (0..WIDTH)
                .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
                .collect();
            let mut inputs = StepInputs::new();
            inputs.push(
                SlotKey::new(Slot::Activation, Some("x")),
                &[1, 1, WIDTH],
                poot_tensor::HostView::new(DType::F32, WIDTH, &bytes).unwrap(),
            );
            engine.step(exe, entry, &inputs, host).map(|_| ())
        };
        let writes_of = |engine: &Engine<FailingWrites>| -> Vec<Vec<u8>> {
            engine
                .device()
                .inner
                .writes
                .iter()
                .map(|(_, bytes)| bytes.clone())
                .collect()
        };

        // The reference: a fresh engine, an unfailing refill.
        let (mut fresh, exe, entry) = load();
        let before = writes_of(&fresh).len();
        step(&mut fresh, exe, entry, &mut FixedHost(fill.clone())).unwrap();
        let reference: Vec<Vec<u8>> = writes_of(&fresh)[before..].to_vec();

        // The failing run: the step's input write, the first row, then the second row fails.
        let (mut engine, exe, entry) = load();
        engine.device.until_failure = Some(2);
        let before = writes_of(&engine).len();
        let failed = step(&mut engine, exe, entry, &mut FixedHost(fill.clone()));
        assert!(
            matches!(failed, Err(ExecError::Device(_))),
            "got {failed:?}"
        );
        let landed: Vec<Vec<u8>> = writes_of(&engine)[before..].to_vec();
        assert!(
            !landed.contains(&map_bytes),
            "the map of a refill that did not complete was published"
        );
        assert!(matches!(
            step(
                &mut engine,
                exe,
                entry,
                &mut FixedHost(crate::SyncWrites::default())
            ),
            Err(ExecError::NeedsReset(_))
        ));

        // A complete reset, then the same refill, lands what a fresh engine lands.
        engine.reset_state(exe, StateScope::All).unwrap();
        let before = writes_of(&engine).len();
        step(&mut engine, exe, entry, &mut FixedHost(fill)).unwrap();
        assert_eq!(writes_of(&engine)[before..].to_vec(), reference);
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
            poot_graph_plan::Target::Gpu {
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
        fn buffer_elems(&self, _buffer: &Held) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            _kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<(), Fault> {
            Ok(())
        }
        fn begin(&mut self) -> Result<(), Fault> {
            Ok(())
        }
        fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), Fault> {
            Ok(())
        }
        fn copy(&mut self, _src: &Held, _dst: &Held) -> Result<(), Fault> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), Fault> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Fault> {
            if self.fail_replay.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Fault);
            }
            Ok(())
        }
        fn write_with(
            &mut self,
            dst: &Held,
            offset: usize,
            len: usize,
            fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), Fault> {
            assert_eq!(offset, 0, "this fake holds whole-buffer writes only");
            let mut bytes = vec![0u8; len];
            fill(0..len, &mut bytes);
            self.contents[dst.index] = bytes;
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
        let b = Builder::for_tests();
        let w = b.constant(name, TensorType::f32(vec![N, K])).unwrap();
        b.finish(w).unwrap()
    }

    fn compile_resident(engine: &Engine<Resident>, g: &Graph) -> StagedProgram {
        compile_staged(
            &g.clone(),
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
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
        let mut names = ConstWeights::new(&store);
        let lora = compile_resident(engine, &names.graph(&identity_of("lora")));
        let base = compile_resident(engine, &names.graph(&identity_of("base")));
        let exe = engine
            .load_weights(Arc::clone(&store), Arc::new(names.map()))
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
                .load_weights(Arc::clone(&weights), Arc::clone(&mapped))
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

/// POOT-1117: the engine's AIE arm. A device whose target is an AIE target loads its kernels through
/// `compile_aie` + `aie_kernel_handle`, against the shape ABI the planner's own choice derives, and every
/// dispatch carries the finite `LaunchKind::Aie` invocation of the stream it loaded. The device here is a
/// recorder with an AIE target; there is no XDNA device in this process, so no row here claims execution.
#[cfg(test)]
mod aie_engine_tests {
    use super::*;
    use crate::{Executor, NoSync};
    use poot_graph_ir::Builder;
    use poot_graph_plan::{
        CompileOptions, multi_device::topology::DeviceId, staged::DevicePlacement,
        staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
    };
    use poot_quant::weights::{
        AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightMap, WeightRole, WeightView,
    };
    use poot_runtime_common::BufferRole;
    use poot_target::Queried;

    impl<D: Device> Engine<D> {
        /// This engine links AIE kernels with `toolchain` instead of the process environment's.
        fn with_toolchain(mut self, toolchain: poot_codegen::Toolchain) -> Self {
            self.toolchain = toolchain;
            self
        }
    }

    const A: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const B: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));
    const C: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Up));

    /// What the device was handed: every loaded kernel's target, binding count and instruction words, and
    /// every dispatch's launch and argument element counts.
    #[derive(Default)]
    struct Seen {
        loaded: Vec<(TargetId, usize, usize)>,
        dispatches: Vec<(LaunchKind, Vec<u32>, u32)>,
    }

    struct AieRecorder {
        caps: AieCaps,
        seen: Seen,
        buffers: usize,
        /// The role of every buffer the engine allocated, in order.
        roles: Vec<BufferRole>,
    }

    impl Device for AieRecorder {
        type Buffer = usize;
        type Kernel = usize;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> Target {
            Target::Aie {
                target: AieTargetId::Xdna2,
                caps: self.caps,
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            _storage: BufferStorage,
            _elems: usize,
        ) -> Result<usize, Self::Error> {
            self.roles.push(role);
            self.buffers += 1;
            Ok(self.buffers - 1)
        }
        fn buffer_elems(&self, _buffer: &usize) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<usize, Self::Error> {
            let poot_runtime_common::KernelCode::Aie(artifact) = kernel.code() else {
                panic!("an AIE device was handed a byte artifact");
            };
            self.seen.loaded.push((
                kernel.target(),
                kernel.args().len(),
                artifact.instructions().len() / 4,
            ));
            Ok(self.seen.loaded.len() - 1)
        }
        fn begin(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
            let mut elems: Vec<u32> = d.inputs.iter().map(|arg| arg.elems).collect();
            elems.push(d.output.elems);
            self.seen.dispatches.push((d.launch, elems, d.work as u32));
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), Self::Error> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Self::Error> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &usize,
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), Self::Error> {
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    fn caps() -> AieCaps {
        // SYNTHETIC limits, roomy for a `[2, 3]` add: not a device measurement.
        AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(1 << 16),
            host_bindings: Queried::Known(8),
            instruction_bytes: Queried::Known(1 << 20),
            memory_budget_bytes: Queried::Known(1 << 30),
            region: poot_target::aie::RegionLimits {
                compute_tiles: Queried::Known(32),
                tile_input_channels: Queried::Known(2),
                tile_output_channels: Queried::Known(2),
                host_input_streams: Queried::Known(16),
                host_output_streams: Queried::Known(16),
            },
        }
    }

    /// `a + b` over two F32 `[2, 3]` weights, compiled for the recorder's AIE target and added as an entry.
    fn load_add(engine: &mut Engine<AieRecorder>) -> Result<(ExecutableId, EntryId), ExecError> {
        load_graph(engine, &[A, B], |b, w| {
            b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap()
        })
    }

    /// The graph `build` makes over F32 `[2, 3]` weights named by `ids`, compiled for the recorder's AIE
    /// target and added as an entry.
    fn load_graph(
        engine: &mut Engine<AieRecorder>,
        ids: &[WeightId],
        build: impl FnOnce(&Builder, &[poot_graph_ir::Traced]) -> poot_graph_ir::Traced,
    ) -> Result<(ExecutableId, EntryId), ExecError> {
        let mut store = WeightStore::builder();
        for &id in ids {
            let bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
            store
                .insert(
                    id.const_name(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F32, vec![2, 3], bytes.into()).unwrap(),
                    ),
                )
                .unwrap();
        }
        let store = Arc::new(store.build());
        let b = Builder::for_tests();
        let consts: Vec<_> = ids
            .iter()
            .map(|id| {
                b.constant(&id.const_name(), TensorType::f32(vec![2, 3]))
                    .unwrap()
            })
            .collect();
        let out = build(&b, &consts);
        let g = b.finish(out).unwrap();
        let mut weights = WeightMap::builder(&store);
        for &id in ids {
            weights
                .map(id, WeightView::Stored(id.const_name().into()))
                .unwrap();
        }
        let weights = weights.build();
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("the add compiles for an AIE target");
        let exe = engine.load_weights(store, Arc::new(weights))?;
        let entry = engine.add_entry(exe, &program)?;
        Ok((exe, entry))
    }

    fn engine(caps: AieCaps, with_iron: bool, dir: &std::path::Path) -> Engine<AieRecorder> {
        Engine::new(AieRecorder {
            caps,
            seen: Seen::default(),
            buffers: 0,
            roles: Vec::new(),
        })
        .with_toolchain(poot_codegen::test_support::scripted_toolchain(
            dir, with_iron,
        ))
    }

    fn load_error(error: ExecError) -> LoadError {
        match error {
            ExecError::Load(load) => *load,
            other => panic!("expected a load error, got {other:?}"),
        }
    }

    /// An AIE device's add is linked through `compile_aie` against the plan's own ABI, loaded as one linked
    /// artifact of three bindings and the stream's one instruction word, and dispatched as one finite
    /// `LaunchKind::Aie` invocation naming that stream; the three arguments carry the planned element counts.
    /// Mutation: have the engine's AIE arm hand the device `LaunchKind::Grid { threads: launch.grid, .. }`
    /// (the pre-POOT-1117 grid carrier); the launch is no `LaunchKind::Aie` and this row goes red.
    #[test]
    fn an_aie_device_loads_a_linked_artifact_and_dispatches_one_aie_invocation() {
        let dir = poot_test_util::unique_temp_path("aie-engine");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), true, &dir);
        let (exe, entry) = load_add(&mut engine).expect("the entry loads through compile_aie");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device.seen;
        assert_eq!(
            seen.loaded,
            vec![(TargetId::Aie(AieTargetId::Xdna2), 3, 1)],
            "one kernel: three bindings, one instruction word"
        );
        assert_eq!(seen.dispatches.len(), 1);
        let (launch, elems, _) = &seen.dispatches[0];
        assert_eq!(
            *launch,
            LaunchKind::Aie(AieInvocation::new(
                AieTargetId::Xdna2,
                NonZeroU32::new(1).unwrap()
            ))
        );
        assert_eq!(
            elems,
            &vec![6, 6, 6],
            "two reads and the write, six elements each"
        );
    }

    /// A chain of three equations is one region: the engine loads one linked artifact whose host side is the
    /// three weights read and the one value returned, and dispatches it once. The two values in between exist
    /// only inside the artifact: the engine allocates a buffer for the returned value and for no other
    /// computed value.
    /// Mutation: have `package` close a region after every equation (`fits` returns an error for any
    /// region of two); three kernels are loaded and three dispatched, and this row goes red.
    #[test]
    fn a_chain_is_one_linked_region_dispatched_once_with_no_buffer_for_its_intermediates() {
        let dir = poot_test_util::unique_temp_path("aie-engine-chain");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), true, &dir);
        let (exe, entry) = load_graph(&mut engine, &[A, B, C], |b, w| {
            let first = b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap();
            let second = b.binary(poot_graph_ir::BinOp::Mul, first, w[2]).unwrap();
            b.binary(poot_graph_ir::BinOp::Add, second, w[0]).unwrap()
        })
        .expect("the chain loads through compile_aie");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device.seen;
        assert_eq!(
            seen.loaded,
            vec![(TargetId::Aie(AieTargetId::Xdna2), 4, 1)],
            "one artifact for the three equations: three reads and the returned value"
        );
        assert_eq!(seen.dispatches.len(), 1, "one invocation for the region");
        assert_eq!(seen.dispatches[0].1, vec![6, 6, 6, 6]);
        let activations = engine
            .device
            .roles
            .iter()
            .filter(|role| **role == BufferRole::Activation)
            .count();
        assert_eq!(
            activations, 1,
            "only the returned value has an arena slot; the two values in between have none"
        );
    }

    /// What only the linked artifact can answer is checked after linking and before the artifact is wrapped:
    /// this target's instruction limit is below the stream the driver wrote. The engine refuses it as the
    /// typed `LoadError::AieLimit` and the device is never handed a kernel. Mutation: drop the
    /// `validate_linked_artifact` call in `aie_region_kernel`; the kernel loads and this row goes red.
    #[test]
    fn a_linked_artifact_past_the_targets_instruction_limit_is_refused_before_load() {
        let dir = poot_test_util::unique_temp_path("aie-engine-instr");
        std::fs::create_dir_all(&*dir).unwrap();
        let tight = AieCaps {
            instruction_bytes: poot_target::Queried::Known(3),
            ..caps()
        };
        let mut engine = engine(tight, true, &dir);
        let error = load_error(load_add(&mut engine).unwrap_err());
        assert!(
            matches!(
                &error,
                LoadError::AieLimit {
                    source: poot_graph_plan::AieLimitRefusal::Exceeded {
                        resource: "instruction bytes",
                        needed: 4,
                        available: 3,
                    },
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(engine.device.seen.loaded.is_empty(), "nothing was loaded");
    }

    /// A missing IRON driver is the compiler's typed `ToolchainMissing`, carried as the engine's
    /// `LoadError::AieCompile`; nothing is loaded. Mutation: swallow a failed `compile_aie` and claim a
    /// loaded kernel index anyway; `add_entry` succeeds and this row goes red.
    #[test]
    fn a_missing_link_driver_is_a_typed_load_error_and_loads_nothing() {
        let dir = poot_test_util::unique_temp_path("aie-engine-no-iron");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), false, &dir);
        let error = load_error(load_add(&mut engine).unwrap_err());
        assert!(
            matches!(
                &error,
                LoadError::AieCompile {
                    source: poot_codegen::AieCompileError::ToolchainMissing {
                        stage: poot_codegen::AieStage::Iron,
                        ..
                    },
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(engine.device.seen.loaded.is_empty(), "nothing was loaded");
    }
}

/// POOT-1164: the engine's Intel NPU arm. A device whose target is an Intel NPU loads one graph artifact per
/// program (the OpenVINO IR the planner's region emits, with its argument ABI), and every step dispatches it
/// once as a `LaunchKind::VpuGraph` invocation of the artifact it loaded. The device here is a recorder with an
/// Intel NPU target; there is no NPU in this process, so no row here claims execution (the device lane's rows
/// do).
#[cfg(test)]
mod vpu_engine_tests {
    use super::*;
    use crate::{Executor, NoSync};
    use poot_graph_ir::Builder;
    use poot_graph_plan::{
        CompileOptions, multi_device::topology::DeviceId, staged::DevicePlacement,
        staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
    };
    use poot_quant::weights::{
        AttnRole, DenseWeight, FfnRole, WeightEntry, WeightId, WeightMap, WeightRole, WeightView,
    };
    use poot_runtime_common::{BufferRole, KernelCode};
    use poot_target::{Queried, VendorCompilerId, VpuCaps};

    const A: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const B: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));
    const C: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Up));

    /// What the device was handed: every loaded artifact's digest, input and output counts, and every
    /// dispatch's launch and argument element counts.
    #[derive(Default)]
    struct Seen {
        loaded: Vec<(u64, usize, usize)>,
        dispatches: Vec<(LaunchKind, Vec<u32>)>,
    }

    struct VpuRecorder {
        seen: Seen,
        buffers: usize,
        roles: Vec<BufferRole>,
    }

    impl Device for VpuRecorder {
        type Buffer = usize;
        type Kernel = usize;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn target(&self) -> Target {
            Target::Vpu {
                target: VpuTargetId::Npu4,
                // SYNTHETIC: an NPU4 with a made-up compiler version, not a device measurement.
                caps: VpuCaps {
                    target: VpuTargetId::Npu4,
                    compiler: Queried::Known(VendorCompilerId { major: 8, minor: 3 }),
                    memory_budget_bytes: Queried::Unknown,
                    narrowing: None,
                },
            }
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            _storage: BufferStorage,
            _elems: usize,
        ) -> Result<usize, Self::Error> {
            self.roles.push(role);
            self.buffers += 1;
            Ok(self.buffers - 1)
        }
        fn buffer_elems(&self, _buffer: &usize) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            kernel: poot_runtime_common::CompiledKernel,
        ) -> Result<usize, Self::Error> {
            let KernelCode::Vpu(artifact) = kernel.code() else {
                panic!("an Intel NPU device was handed a non-graph artifact");
            };
            assert_eq!(kernel.target(), TargetId::Vpu(VpuTargetId::Npu4));
            self.seen.loaded.push((
                artifact.digest(),
                artifact.inputs().len(),
                artifact.outputs().len(),
            ));
            Ok(self.seen.loaded.len() - 1)
        }
        fn begin(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
            let mut elems: Vec<u32> = d.inputs.iter().map(|arg| arg.elems).collect();
            elems.push(d.output.elems);
            self.seen.dispatches.push((d.launch, elems));
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), Self::Error> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Self::Error> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &usize,
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), Self::Error> {
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    /// The graph `build` makes over F16 `[2, 3]` weights named by `ids`, compiled for the recorder's Intel NPU
    /// target and added as an entry.
    fn load_graph(
        engine: &mut Engine<VpuRecorder>,
        ids: &[WeightId],
        build: impl FnOnce(&Builder, &[poot_graph_ir::Traced]) -> poot_graph_ir::Traced,
    ) -> Result<(ExecutableId, EntryId), ExecError> {
        let mut store = WeightStore::builder();
        for &id in ids {
            let bytes: Vec<u8> = (0..6u16).flat_map(|i| (0x3c00 + i).to_le_bytes()).collect();
            store
                .insert(
                    id.const_name(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F16, vec![2, 3], bytes.into()).unwrap(),
                    ),
                )
                .unwrap();
        }
        let store = Arc::new(store.build());
        let b = Builder::for_tests();
        let consts: Vec<_> = ids
            .iter()
            .map(|id| {
                b.constant(&id.const_name(), TensorType::new(vec![2, 3], DType::F16))
                    .unwrap()
            })
            .collect();
        let out = build(&b, &consts);
        let g = b.finish(out).unwrap();
        let mut weights = WeightMap::builder(&store);
        for &id in ids {
            weights
                .map(id, WeightView::Stored(id.const_name().into()))
                .unwrap();
        }
        let weights = weights.build();
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
        )
        .expect("the graph compiles for an Intel NPU target");
        let exe = engine.load_weights(store, Arc::new(weights))?;
        let entry = engine.add_entry(exe, &program)?;
        Ok((exe, entry))
    }

    fn engine() -> Engine<VpuRecorder> {
        // `Engine::new` over an Intel NPU device must not need a kernel-body compiler.
        Engine::new(VpuRecorder {
            seen: Seen::default(),
            buffers: 0,
            roles: Vec::new(),
        })
    }

    /// A chain of three equations over three weights is one graph artifact whose host side is the three
    /// weights read and the one value returned, dispatched once as a graph invocation of that same artifact.
    /// The two values in between exist only inside the graph: the engine allocates a buffer for the returned
    /// value and for no other computed value.
    ///
    /// Mutation: have `package` return no region for `ArtifactGranularity::Graph`; the load is refused ("a
    /// graph node outside any region") and nothing is loaded.
    #[test]
    fn a_chain_is_one_graph_artifact_dispatched_once_with_no_buffer_for_its_intermediates() {
        let mut engine = engine();
        let (exe, entry) = load_graph(&mut engine, &[A, B, C], |b, w| {
            let first = b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap();
            let second = b.binary(poot_graph_ir::BinOp::Mul, first, w[2]).unwrap();
            b.binary(poot_graph_ir::BinOp::Add, second, w[0]).unwrap()
        })
        .expect("the chain loads");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device.seen;
        assert_eq!(seen.loaded.len(), 1, "one artifact for the three equations");
        let (digest, inputs, outputs) = seen.loaded[0];
        assert_eq!(
            (inputs, outputs),
            (3, 1),
            "three weights read, one value returned"
        );
        assert_eq!(seen.dispatches.len(), 1, "one invocation for the graph");
        assert_eq!(
            seen.dispatches[0].0,
            LaunchKind::VpuGraph(VpuInvocation::new(VpuTargetId::Npu4, digest)),
            "the invocation names the artifact the engine loaded"
        );
        assert_eq!(seen.dispatches[0].1, vec![6, 6, 6, 6]);
        let activations = engine
            .device
            .roles
            .iter()
            .filter(|role| **role == BufferRole::Activation)
            .count();
        assert_eq!(
            activations, 1,
            "only the returned value has an arena slot; the two values in between have none"
        );
    }

    /// Two entries over the same graph load the artifact once: the engine keys it by its digest.
    ///
    /// Mutation: drop the `vpu_kernels` lookup in `vpu_region_kernel`; the device loads the artifact twice.
    #[test]
    fn the_same_graph_added_twice_is_loaded_once() {
        let mut engine = engine();
        for _ in 0..2 {
            load_graph(&mut engine, &[A, B], |b, w| {
                b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap()
            })
            .expect("the add loads");
        }
        assert_eq!(engine.device.seen.loaded.len(), 1);
    }
}
