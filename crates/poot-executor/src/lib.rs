//! The one executor contract (ADR-0102 decision 2, Card 546a).
//!
//! One generic [`Engine`] over a thin per-backend [`Device`] trait implements the object-safe
//! [`Executor`]. An executable holds the weights and several entry programs
//! (prefill, decode, ..) that share state by (name, aval, storage); a step binds inputs
//! keyed by [`poot_graph_ir::SlotKey`] and replays the entry's recording (capture is the
//! default, ADR-0102). No test-only accessor and no `test-hooks` feature: a test
//! reads carried state through a production step output, never a `read_state` escape hatch.

mod binder;
mod coordinator;
mod device;
mod engine;
mod error;
mod lanes;
mod shared_kernels;
mod timing;

pub use coordinator::{
    Coordinator, CoordinatorCounters, CoordinatorError, DeviceCounters, DeviceSupport,
    ScheduleError, SupportActual, SupportPlan,
};
pub use device::{
    AieInvocation, Arg, Collectives, Device, DeviceTime, Dispatch, DispatchDeviceTime, LaunchKind,
    MeasuredDeviceTime, NativeCounters, SupportBound, VpuInvocation, WeightUpload, write_slice,
};
pub use engine::Engine;
pub use error::{BindError, DeviceError, ExecError, LoadError};
use poot_runtime_common::{BufferRole, MemoryCounterSnapshot};
use poot_tensor::{HostTensor, HostView};
pub use shared_kernels::{SharedKernels, SharedKernelsMismatch};
pub use timing::{DetailedTiming, DrainPolicy, TimingOptions};

#[cfg(test)]
mod tests;

use poot_graph_ir::{BoundaryId, ForwardSlotMap, SlotKey};
use poot_graph_plan::{
    multi_device::topology::DeviceId, staged::StageId, staged::StagedProgram, staged::SyncPoint,
    staged::TargetSet,
};
use poot_quant::weights::{WeightMap, WeightStore};
use std::sync::Arc;

/// One executable (weights plus entries) inside one executor. Not a heap address: engine-owned,
/// behind a typed handle (spike-546 F1 - an object-safe `Executor` cannot hand out a generic
/// `Executable<D>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutableId(usize);

/// One entry program of an executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntryId(usize);

impl ExecutableId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl EntryId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }
    pub(crate) fn index(self) -> usize {
        self.0
    }
    /// This entry's opaque identity for [`poot_profile::TimingSnapshot`]/[`poot_profile::Report`]
    /// (Card 552): a caller that knows which entry is semantically "decode" versus "prefill" (the
    /// bench runner, a future driver) uses this value as the `entries` filter in
    /// `Report::window`. `poot-profile` assigns it no further meaning.
    pub fn raw(self) -> u64 {
        self.0 as u64
    }
}

/// Which carried state [`Executor::reset_state`] zeroes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateScope {
    /// Every carried value of the executable; the only scope that clears a failed step's
    /// [`ExecError::NeedsReset`].
    All,
    /// Row `n` of the leading axis of every carried value that has a row axis: a recurrent state
    /// pool's sequence slot, a private KV cache's row. A paged KV pool has none and is left alone.
    /// A row beyond a value's leading axis is a [`BindError::StateRow`] and nothing is zeroed.
    Row(usize),
}

/// The canonical identity of one pooled site (POOT-746): which stored bytes its tables page, in
/// which planned lanes and at how many rows. Entry-local site numbering never reaches a host: two
/// entries of one executable that page the same tables (a prefill and a decode, a chunk) report equal
/// identities, two tables of one layer report different ones, and a same-named weight of another store
/// generation never compares equal. A host keys its per-site state by this and nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SiteIdentity(Vec<TableIdentity>);

/// One pooled table's identity inside a [`SiteIdentity`]: the stored byte runs it reads, each named by
/// the generation of the store payload that holds it, and its planned lane and pool height.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TableIdentity {
    pub(crate) runs: Vec<(
        poot_quant::weights::StoreGeneration,
        Option<poot_quant::SourceRole>,
        std::ops::Range<usize>,
    )>,
    pub(crate) storage: poot_target::BufferStorage,
    pub(crate) pool_rows: usize,
}

impl SiteIdentity {
    pub(crate) fn new(tables: Vec<TableIdentity>) -> Self {
        Self(tables)
    }
}

/// What the host reads at a sync point: the boundary values the point names (I32 expert ids, as the
/// stage that exported them left them) and the site's canonical identity and per-row costs.
#[derive(Clone, Debug)]
pub struct SyncReads {
    site: SiteIdentity,
    row_bytes: Vec<usize>,
    values: Vec<(BoundaryId, Vec<i32>)>,
}

impl SyncReads {
    /// The pooled site this sync serves, canonical across entries.
    pub fn site(&self) -> &SiteIdentity {
        &self.site
    }

    /// Device bytes one pool row of each table holds, in [`SyncPoint::site`] table order: what a
    /// refill of one expert writes.
    pub fn row_bytes(&self) -> &[usize] {
        &self.row_bytes
    }

    /// The values `id` named, as the point's stage left them.
    pub fn values(&self, id: BoundaryId) -> Option<&[i32]> {
        self.values
            .iter()
            .find(|(read, _)| *read == id)
            .map(|(_, values)| values.as_slice())
    }
}

/// Make experts `experts` resident in rows `first_row..` of table `table` (an index into the sync
/// point's [`SyncPoint::site`] weights): the executor writes each expert's stored bytes into its row
/// in the table's planned lane, never a re-encode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowFill {
    /// Index into the sync point's [`SyncPoint::site`] weights naming the table to fill.
    pub table: usize,
    /// First pool row written; the experts land in consecutive rows from here.
    pub first_row: usize,
    /// Global expert indices to make resident, in row order.
    pub experts: Vec<usize>,
}

/// What the host wants written before the next stage runs. The executor validates every range before
/// it writes any byte, then writes the rows and then the map.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SyncWrites {
    /// Pool rows to fill, applied before `map`.
    pub rows: Vec<RowFill>,
    /// The site's whole `[E]` map: the pool row of each global expert, `-1` when absent.
    pub map: Option<Vec<i32>>,
}

/// The host side of a step's sync points (the expert-residency manager is the one implementation):
/// called once per point, after the stage before it completed, with the values the point reads. Its
/// writes land before the next stage is replayed.
pub trait HostSync {
    /// Decide residency at `point` from `reads`, returning the row fills and map to write before the next stage replays.
    fn sync(&mut self, point: &SyncPoint, reads: &SyncReads) -> Result<SyncWrites, ExecError>;
}

/// The host sync of a program with no sync points: a program that reaches one is refused.
pub struct NoSync;

impl HostSync for NoSync {
    fn sync(&mut self, point: &SyncPoint, _reads: &SyncReads) -> Result<SyncWrites, ExecError> {
        Err(ExecError::NoHostSync { point: point.id })
    }
}

/// A borrowed host tensor bound to one slot of a step: its caller-declared shape (checked against
/// the slot's aval, SC-011) and its typed byte view (checked for element count and dtype lane,
/// SC-012/SC-006).
#[derive(Clone, Copy, Debug)]
pub struct StepValue<'a> {
    /// The caller-declared shape, checked against the slot's aval.
    pub shape: &'a [usize],
    /// The typed bytes, checked for element count and dtype lane.
    pub view: HostView<'a>,
}

/// The values one step binds, one per slot of the entry's schema.
///
/// A [`Slot::WriteMap`](poot_graph_ir::Slot::WriteMap) slot is bound only through
/// [`Self::push_write_map`], which takes a checked [`ForwardSlotMap`]: a repeated or out-of-pool slot is
/// unrepresentable there, and [`Self::push`] of a raw view to such a slot is a [`BindError`].
#[derive(Debug, Default)]
pub struct StepInputs<'a> {
    values: Vec<(SlotKey, StepValue<'a>)>,
    write_maps: Vec<(SlotKey, &'a ForwardSlotMap)>,
}

impl<'a> StepInputs<'a> {
    /// An empty input set; bind each slot of the entry's schema with [`StepInputs::push`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `view`, declared as `shape`, to the slot `key`.
    pub fn push(&mut self, key: SlotKey, shape: &'a [usize], view: HostView<'a>) {
        self.values.push((key, StepValue { shape, view }));
    }

    /// Bind the write map of `key`, a [`Slot::WriteMap`](poot_graph_ir::Slot::WriteMap) slot, from a map
    /// whose constructor already refused out-of-pool and repeated slots.
    pub fn push_write_map(&mut self, key: SlotKey, map: &'a ForwardSlotMap) {
        self.write_maps.push((key, map));
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &(SlotKey, StepValue<'a>)> {
        self.values.iter()
    }

    pub(crate) fn write_maps(&self) -> impl Iterator<Item = &(SlotKey, &'a ForwardSlotMap)> {
        self.write_maps.iter()
    }
}

/// Recordings, replays and live memory by role, aggregated over every executable this executor
/// holds (R-546-11): the bench profile report's one source, replacing the test-only `recordings()`/
/// `read_state()` accessors the contract no longer has. `memory` is [`Device::memory`] (Card 547a): the
/// device's own runtime counters, not an engine-side tally, so cross-backend equality is not equal by
/// construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecutorStats {
    /// Recordings made, summed over every executable.
    pub recordings: u64,
    /// Replays submitted, summed over every executable.
    pub replays: u64,
    /// Recordings and replays of each stage of each entry (a stage records once and replays every
    /// step).
    pub stages: Vec<StageStats>,
    /// Bytes the host wrote to device buffers while binding steps: the caller's inputs and hosted
    /// embeds. A link between stages is a shared device buffer, never a host write.
    pub host_write_bytes: u64,
    /// The device's own live bytes by role ([`Device::memory`]).
    pub memory: Vec<(BufferRole, MemoryCounterSnapshot)>,
    /// Bytes read from the weight store and encoded for upload, summed over every weight upload. A
    /// [`WeightUpload::SizeOnly`] device reads none.
    pub weight_bytes_read: u64,
    /// Host syncs run: one per sync point of each pooled step. The engine's own count, the source of
    /// truth for what a pooled step moved (an all-hit step adds to `syncs` and `sync_read_bytes` and
    /// nothing to `pool_rows_written` or `pool_refill_bytes`); a host's counters only cross-check it.
    pub syncs: u64,
    /// Bytes read back for those syncs (the ids a host decides residency from).
    pub sync_read_bytes: u64,
    /// Pool rows refilled and the stored bytes they wrote, summed over every sync.
    pub pool_rows_written: u64,
    /// Stored bytes those rows wrote.
    pub pool_refill_bytes: u64,
    /// Kernels this engine took through the kernel cache (a disk load or a codegen run), once per distinct
    /// kernel. An engine built over [`SharedKernels`] does not count a kernel another engine of that
    /// table already resolved.
    pub kernels_resolved: u64,
    /// Bounded typed timing (Card 552): exact cumulative step/host/device-sum counters per entry,
    /// plus whatever per-dispatch detail the engine's [`TimingOptions`] retained. Never an
    /// append-only history; see [`poot_profile::TimingSnapshot`].
    pub timing: poot_profile::TimingSnapshot,
}

/// One stage's recording and replay counts (see [`ExecutorStats::stages`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageStats {
    /// The executable the stage belongs to.
    pub exe: ExecutableId,
    /// The entry program the stage belongs to.
    pub entry: EntryId,
    /// The stage within the entry's staged program.
    pub stage: StageId,
    /// Times this stage was recorded.
    pub recordings: u64,
    /// Times this stage was replayed.
    pub replays: u64,
}

/// What one device of an executor holds and could still allocate, read now. Admission checks each device
/// against its own plan and budget, so a total over devices is never a substitute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMemory {
    /// The device the figures describe.
    pub device: DeviceId,
    /// What the device's runtime could still allocate ([`Device::free_memory_bytes`]); `None` where it
    /// cannot say.
    pub free_bytes: Option<u64>,
    /// The device's own live bytes by role, for the roles that have allocated.
    pub live: Vec<(BufferRole, MemoryCounterSnapshot)>,
}

/// The object-safe executor the driver holds as `Box<dyn Executor>`.
pub trait Executor {
    /// The devices and targets this executor drives.
    fn target_set(&self) -> TargetSet;
    /// Each device's own free memory and live bytes by role, in rank order. The default is the one
    /// device of a single-device executor, from [`Executor::free_memory_bytes`] and its counters; an
    /// executor over several devices reports each of them.
    fn device_memory(&self) -> Vec<DeviceMemory> {
        let targets = self.target_set();
        let &[(device, _)] = targets.devices() else {
            unreachable!("an executor over several devices reports each device's memory itself")
        };
        vec![DeviceMemory {
            device,
            free_bytes: self.free_memory_bytes(),
            live: self.stats().memory,
        }]
    }
    /// Device memory this executor's device could still allocate right now ([`Device::free_memory_bytes`]),
    /// read live; `None` where the runtime cannot say. Admission asks again at each decision instead of
    /// trusting the figure the device reported when it opened.
    fn free_memory_bytes(&self) -> Option<u64>;
    /// A new executable over `store`, its consts bound by `weights`. Records both and uploads
    /// nothing: a weight's device lane is a per-program planning decision, so upload happens at
    /// `add_entry`, through the engine's residency map keyed by (store generation, stored byte
    /// runs, planned storage), so executables over stores that share a payload share its upload
    /// and a same-named payload of another generation never aliases it.
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError>;
    /// Binds `program`'s consts to the store through the executable's weight map (each weight const
    /// is a [`poot_quant::weights::WeightId::const_name`], or a packed source name over one, bound
    /// through the map's view to byte runs of store entries; a const the map does not name is
    /// unbound, even when a store key equals its name), uploads each stored byte run once per
    /// planned storage and store generation, fills a hosted embed gather from the step's tokens,
    /// allocates or shares state by (name, aval, storage),
    /// and loads every plan. The entry records on its first `step`.
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
    ) -> Result<EntryId, ExecError>;
    /// Binds `inputs`, replays the entry's recording (recording it first if this is its first step),
    /// and checks the validation packet before any output readback (R472-006).
    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError>;
    /// Zero the carried state of `exe` selected by `scope`.
    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError>;
    /// Drops `entry`'s recording and arena; the executable's shared state is untouched (R-546-3).
    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError>;
    /// Counters aggregated over every executable this executor holds.
    fn stats(&self) -> ExecutorStats;
    /// Drop `exe`, its entries and its state; its [`ExecutableId`] is invalid afterwards.
    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError>;
}

/// Where a step's outputs are read from. Implemented by the engine; private to the contract.
pub(crate) trait OutputSource {
    fn read_output(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        out: &mut [u8],
    ) -> Result<(), ExecError>;
    fn output_bytes(&self, exe: ExecutableId, entry: EntryId) -> Result<usize, ExecError>;
    /// The primary output's host dtype and shape, for [`StepOutputs::to_host`].
    fn output_host(
        &self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<(poot_tensor::DType, Vec<usize>), ExecError>;
}

/// A step's outputs, borrowed from the executor: no step or unload can run while they are held
/// (restating dexec A4).
///
/// ```compile_fail,E0499
/// fn two_steps(exec: &mut dyn poot_executor::Executor, exe: poot_executor::ExecutableId,
///              entry: poot_executor::EntryId) {
///     let inputs = poot_executor::StepInputs::new();
///     let first = exec.step(exe, entry, &inputs, &mut poot_executor::NoSync).unwrap();
///     let _second = exec.step(exe, entry, &inputs, &mut poot_executor::NoSync);
///     drop(first);
/// }
/// ```
pub struct StepOutputs<'a> {
    source: &'a mut dyn OutputSource,
    exe: ExecutableId,
    entry: EntryId,
    time: DeviceTime,
}

impl<'a> StepOutputs<'a> {
    pub(crate) fn new<S: OutputSource>(
        source: &'a mut S,
        exe: ExecutableId,
        entry: EntryId,
        time: DeviceTime,
    ) -> Self {
        Self {
            source,
            exe,
            entry,
            time,
        }
    }

    /// The primary output's bytes, in its planned lane.
    pub fn read(&mut self) -> Result<Vec<u8>, ExecError> {
        let mut out = vec![0u8; self.source.output_bytes(self.exe, self.entry)?];
        self.source.read_output(self.exe, self.entry, &mut out)?;
        Ok(out)
    }

    /// The primary output as an owned [`HostTensor`] of its planned lane's dtype and the graph's
    /// declared shape, beside the borrowed byte [`Self::read`]. A packed or mirrored output lane has
    /// no host tensor form and is a typed error.
    pub fn to_host(&mut self) -> Result<HostTensor, ExecError> {
        let (dtype, shape) = self.source.output_host(self.exe, self.entry)?;
        let bytes = self.read()?;
        Ok(HostTensor::from_le_bytes(dtype, shape, &bytes)?)
    }

    /// The device time of the step that produced these outputs.
    pub fn device_time(&self) -> DeviceTime {
        self.time.clone()
    }
}
