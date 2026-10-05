//! The one executor contract (ADR-0102 decision 2, Card 546a).
//!
//! One generic [`Engine`] over a thin per-backend [`Device`] trait implements the object-safe
//! [`Executor`]. An executable holds the weights and several entry programs
//! (prefill, decode, ..) that share state by (name, aval, storage); a step binds inputs
//! keyed by [`poot_graph_ir::SlotKey`] and replays the entry's recording (capture is the
//! default, ADR-0102). No test-only accessor and no `test-hooks` feature: a test
//! reads carried state through a production step output, never a `read_state` escape hatch.

mod binder;
mod device;
mod engine;
mod error;
mod lanes;
mod timing;

pub use device::{
    Arg, BufferRole, Device, DeviceTime, Dispatch, DispatchDeviceTime, MeasuredDeviceTime,
    MemoryCounterSnapshot,
};
pub use engine::Engine;
pub use error::{BindError, DeviceError, ExecError, LoadError};
use poot_tensor::HostTensor;
pub use poot_tensor::HostView;
pub use timing::{DetailedTiming, DrainPolicy, TimingOptions};

#[cfg(test)]
mod tests;

use poot_graph_ir::SlotKey;
use poot_graph_plan::{StagedProgram, SyncId, TargetSet};
use poot_quant::weights::{WeightMap, WeightStore};
use std::sync::Arc;

/// How an executable's graph consts bind to its store.
#[derive(Clone, Debug)]
pub enum WeightSource {
    /// A `Model` graph: each weight const is a [`poot_quant::weights::WeightId::const_name`] (or a
    /// packed source name over one), bound through the map's view to byte runs of store entries.
    /// A const the map does not name is unbound, even when a store key equals its name.
    Map(Arc<WeightMap>),
    /// Name equality: a const binds the store entry its name equals (a packed source name, the
    /// entry at its linear id). Held for the Runner's graphs until Card 739 deletes it.
    ConstNames,
}

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

/// Which carried state [`Executor::reset_state`] zeroes (rows are the leading row axis,
/// a future card's `Rows` variant).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateScope {
    All,
}

/// The host side of a step's sync points. A single-stage program (this card's only
/// shape) has none.
pub trait HostSync {
    fn sync(&mut self, point: SyncId) -> Result<(), ExecError>;
}

/// The host sync of a single-stage program: it is never called.
pub struct NoSync;

impl HostSync for NoSync {
    fn sync(&mut self, point: SyncId) -> Result<(), ExecError> {
        unreachable!("single-stage program reached sync point {point:?}")
    }
}

/// A borrowed host tensor bound to one slot of a step: its caller-declared shape (checked against
/// the slot's aval, SC-011) and its typed byte view (checked for element count and dtype lane,
/// SC-012/SC-006).
#[derive(Clone, Copy, Debug)]
pub struct StepValue<'a> {
    pub shape: &'a [usize],
    pub view: HostView<'a>,
}

/// The values one step binds, one per slot of the entry's schema.
#[derive(Debug, Default)]
pub struct StepInputs<'a> {
    values: Vec<(SlotKey, StepValue<'a>)>,
}

impl<'a> StepInputs<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, key: SlotKey, shape: &'a [usize], view: HostView<'a>) {
        self.values.push((key, StepValue { shape, view }));
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &(SlotKey, StepValue<'a>)> {
        self.values.iter()
    }
}

/// Recordings, replays and live memory by role, aggregated over every executable this executor
/// holds (R-546-11): the bench profile report's one source, replacing the test-only `recordings()`/
/// `read_state()` accessors the contract no longer has. `memory` is [`Device::memory`] (Card 547a): the
/// device's own runtime counters, not an engine-side tally, so cross-backend equality is not equal by
/// construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecutorStats {
    pub recordings: u64,
    pub replays: u64,
    pub memory: Vec<(BufferRole, MemoryCounterSnapshot)>,
    /// Bounded typed timing (Card 552): exact cumulative step/host/device-sum counters per entry,
    /// plus whatever per-dispatch detail the engine's [`TimingOptions`] retained. Never an
    /// append-only history; see [`poot_profile::TimingSnapshot`].
    pub timing: poot_profile::TimingSnapshot,
}

/// The object-safe executor the driver holds as `Box<dyn Executor>`.
pub trait Executor {
    fn target_set(&self) -> TargetSet;
    /// A new executable over `store`, its consts bound by `weights`. Records both and uploads
    /// nothing: a weight's device lane is a per-program planning decision, so upload happens at
    /// `add_entry`, through the engine's residency map keyed by (store generation, stored byte
    /// runs, planned storage), so executables over stores that share a payload share its upload
    /// and a same-named payload of another generation never aliases it.
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: WeightSource,
    ) -> Result<ExecutableId, ExecError>;
    /// Binds `program`'s consts to the store through the executable's [`WeightSource`], uploads
    /// each stored byte run once per planned storage and store generation, fills a hosted embed
    /// gather from the step's tokens, allocates or shares state by (name, aval, storage),
    /// and loads every plan. The entry records on its first `step`.
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<poot_graph_ir::ValidationOutputs>,
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
    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError>;
    /// Drops `entry`'s recording and arena; the executable's shared state is untouched (R-546-3).
    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError>;
    fn stats(&self) -> ExecutorStats;
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

    pub fn device_time(&self) -> DeviceTime {
        self.time.clone()
    }
}
