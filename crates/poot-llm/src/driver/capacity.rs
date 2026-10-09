//! Exact device-memory accounting for a driver (POOT-587, core review CR30).
//!
//! A candidate shape is admitted by the bytes its compiled entries *will* allocate, never by an affine
//! extrapolation from two samples. Three pieces:
//!
//! - **Planning.** [`Planner::plan`] runs the production `Driver::prepare` over a [`PlanningDevice`]
//!   under the real `Engine`: the same trace, compile, residency map and buffer plan decide every
//!   allocation, and the device records each one instead of making it. The device only sizes weights
//!   ([`WeightUpload::SizeOnly`]), so planning a candidate reads and encodes none of the
//!   checkpoint, and one [`Planner`] shares its kernel artifacts across candidates
//!   ([`poot_executor::SharedKernels`]), so each distinct kernel is resolved once per pass. Shared weights and shared state are
//!   uploaded once by the engine, so they appear once; each entry's own arena, inputs and outputs are
//!   distinct allocations, so they sum. A [`Footprint`] is the set of those allocation owners.
//! - **Admission.** [`Budget::admit`] compares a footprint with one device's budget, checked, with an
//!   explicit reserve for runtime-private allocations the planner cannot see (recordings, staging) and
//!   with the bytes earlier attempts still hold ([`Ledger`]). An unknown budget is a typed refusal, never
//!   infinity.
//! - **Search.** [`fit`] walks a declared candidate domain, preferred first, plans each candidate
//!   against a *fresh* budget observation, and stops at the first that fits. It checks every candidate it
//!   reaches and assumes nothing about monotonicity; a refused candidate is dropped, not published, and
//!   whatever its planner could not prove released stays charged.
//!
//! After the chosen candidate is prepared on the real executor, [`verify_prepared`] checks the actual
//! role counters against the plan and the budget before anything serves.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use poot_executor::{
    BufferRole, Device, DeviceTime, Dispatch, Engine, Executor, MemoryCounterSnapshot,
    SharedKernels, SharedKernelsMismatch, WeightUpload,
};
use poot_graph_plan::{DeviceId, Submission, Target, TargetSet};
use poot_runtime_common::{AllocGuard, CompiledKernel, MemoryCounters};
use poot_target::BufferStorage;

use super::error::{DriverError, InvalidOptions};
use super::{Driver, DriverOptions, ModelHandle, ServingShapes};

/// One owner of device bytes: an allocation with an identity. Two records naming the same `(device, id)`
/// are the same allocation (a shared weight, a shared state), whatever number of entries reference it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationOwner {
    pub id: String,
    pub device: DeviceId,
    pub role: BufferRole,
    pub bytes: u64,
}

/// A footprint that cannot be composed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CapacityError {
    /// One identity declared with two different sizes or roles: the producers disagree about one
    /// allocation, so neither size is evidence.
    #[error(
        "allocation owner {id:?} on {device:?} is declared as {first_bytes} bytes ({first_role:?}) and again as {second_bytes} bytes ({second_role:?})"
    )]
    ConflictingOwner {
        id: String,
        device: DeviceId,
        first_role: BufferRole,
        first_bytes: u64,
        second_role: BufferRole,
        second_bytes: u64,
    },
    #[error("the bytes owned on {device:?} overflow u64")]
    Overflow { device: DeviceId },
}

/// The allocation owners a plan retains, composed: a shared owner once, distinct live owners summed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    owners: BTreeMap<(DeviceId, String), AllocationOwner>,
}

impl Footprint {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `owner`. The same identity with the same size and role is the same allocation, counted once;
    /// the same identity with another size or role is a [`CapacityError::ConflictingOwner`].
    pub fn insert(&mut self, owner: AllocationOwner) -> Result<(), CapacityError> {
        let key = (owner.device, owner.id.clone());
        match self.owners.get(&key) {
            None => {
                self.owners.insert(key, owner);
                Ok(())
            }
            Some(held) if held.bytes == owner.bytes && held.role == owner.role => Ok(()),
            Some(held) => Err(CapacityError::ConflictingOwner {
                id: owner.id,
                device: owner.device,
                first_role: held.role,
                first_bytes: held.bytes,
                second_role: owner.role,
                second_bytes: owner.bytes,
            }),
        }
    }

    pub fn owners(&self) -> impl Iterator<Item = &AllocationOwner> {
        self.owners.values()
    }

    /// The devices that own at least one allocation.
    pub fn devices(&self) -> Vec<DeviceId> {
        let mut devices: Vec<DeviceId> = self.owners.keys().map(|(device, _)| *device).collect();
        devices.dedup();
        devices
    }

    /// Every byte `device` owns, checked.
    pub fn device_bytes(&self, device: DeviceId) -> Result<u64, CapacityError> {
        self.owners
            .values()
            .filter(|owner| owner.device == device)
            .try_fold(0u64, |total, owner| total.checked_add(owner.bytes))
            .ok_or(CapacityError::Overflow { device })
    }

    /// The bytes `device` owns per role, in [`BufferRole::ALL`] order, for the roles that own any.
    pub fn role_bytes(&self, device: DeviceId) -> Result<Vec<(BufferRole, u64)>, CapacityError> {
        let mut roles = Vec::new();
        for role in BufferRole::ALL {
            let mut owned = self
                .owners
                .values()
                .filter(|owner| owner.device == device && owner.role == role)
                .peekable();
            if owned.peek().is_none() {
                continue;
            }
            let total = owned
                .try_fold(0u64, |total, owner| total.checked_add(owner.bytes))
                .ok_or(CapacityError::Overflow { device })?;
            roles.push((role, total));
        }
        Ok(roles)
    }
}

/// What one device may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceBudget {
    /// The runtime could not say. Admission refuses; it never reads this as unlimited.
    Unknown,
    /// `bytes` the process may allocate, of which `reserve` is kept for runtime-private allocations
    /// (recordings, staging, driver state) that no plan can see.
    Known { bytes: u64, reserve: u64 },
}

/// What an executor reports about its device memory at one moment, read when asked: the input of every
/// admission decision. The caps' memory figure is the runtime's query at open and can be stale; `free_bytes`
/// is the runtime's answer now.
#[derive(Clone, Debug)]
pub struct Observation {
    pub targets: TargetSet,
    /// [`poot_executor::Executor::free_memory_bytes`]: what the device could still allocate, or `None`
    /// where the runtime cannot say.
    pub free_bytes: Option<u64>,
    /// Bytes this process's own executor already holds on the device, which `free_bytes` no longer
    /// includes. Zero before anything is prepared.
    pub held_bytes: u64,
}

impl Observation {
    /// Observe `executor` before it holds anything.
    pub fn of(executor: &dyn Executor) -> Self {
        Self {
            targets: executor.target_set(),
            free_bytes: executor.free_memory_bytes(),
            held_bytes: 0,
        }
    }
}

/// The budgets of the devices a footprint may touch, per device: a global figure would admit a plan that
/// fits the sum while one device is over.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    devices: BTreeMap<DeviceId, DeviceBudget>,
}

/// Bytes still held by attempts that did not release them, per device: a poisoned or still-pending
/// allocation cannot be proven freed, so it stays charged against every later candidate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    charged: BTreeMap<DeviceId, u64>,
}

impl Ledger {
    pub fn charged(&self, device: DeviceId) -> u64 {
        self.charged.get(&device).copied().unwrap_or(0)
    }

    fn charge(&mut self, device: DeviceId, bytes: u64) -> Result<(), CapacityError> {
        if bytes == 0 {
            return Ok(());
        }
        let held = self.charged.entry(device).or_insert(0);
        *held = held
            .checked_add(bytes)
            .ok_or(CapacityError::Overflow { device })?;
        Ok(())
    }
}

/// Why a footprint is not admitted.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CapacityRefusal {
    #[error("the memory budget of {device:?} is unknown")]
    UnknownBudget { device: DeviceId },
    #[error("{device:?} has no budget entry")]
    UnknownDevice { device: DeviceId },
    #[error(
        "{device:?} needs {required} bytes plus {charged} still charged, over the {available} bytes available"
    )]
    OverBudget {
        device: DeviceId,
        required: u64,
        charged: u64,
        available: u64,
    },
    /// A role holds more live bytes than its plan allocated: the plan under-counted, so its admission
    /// proved nothing.
    #[error("{role:?} holds {actual} live bytes on {device:?}, over the {planned} planned")]
    Underestimated {
        device: DeviceId,
        role: BufferRole,
        planned: u64,
        actual: u64,
    },
    #[error(transparent)]
    Footprint(#[from] CapacityError),
}

impl Budget {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, device: DeviceId, budget: DeviceBudget) -> Self {
        self.devices.insert(device, budget);
        self
    }

    /// The budget of the device behind `observation`, with `reserve` kept back. What the process may
    /// allocate is what the runtime says is free now plus what the process already holds, capped by the
    /// ceiling the device reported at open when it had one: the open-time figure alone would miss memory
    /// other processes took since. A runtime that cannot say what is free gives an unknown budget, which
    /// admission refuses; the open-time ceiling is never promoted to a budget.
    pub fn observed(observation: &Observation, reserve: u64) -> Self {
        let mut budget = Self::new();
        for (device, target) in observation.targets.devices() {
            let live = observation
                .free_bytes
                .map(|free| free.saturating_add(observation.held_bytes));
            let device_budget = match (live, target.caps.memory_budget_bytes) {
                (None, _) => DeviceBudget::Unknown,
                (Some(live), Some(ceiling)) => DeviceBudget::Known {
                    bytes: live.min(ceiling),
                    reserve,
                },
                (Some(live), None) => DeviceBudget::Known {
                    bytes: live,
                    reserve,
                },
            };
            budget.devices.insert(*device, device_budget);
        }
        budget
    }

    /// The bytes `device` may allocate once its reserve is kept back; `None` when the budget is unknown
    /// or the device has no entry.
    pub fn available(&self, device: DeviceId) -> Option<u64> {
        match self.devices.get(&device)? {
            DeviceBudget::Unknown => None,
            DeviceBudget::Known { bytes, reserve } => Some(bytes.saturating_sub(*reserve)),
        }
    }

    /// Admit `footprint` when, on every device it touches, its bytes plus what `ledger` still charges
    /// fit the budget less its reserve.
    pub fn admit(&self, footprint: &Footprint, ledger: &Ledger) -> Result<(), CapacityRefusal> {
        for device in footprint.devices() {
            let required = footprint.device_bytes(device)?;
            let (bytes, reserve) = match self.devices.get(&device) {
                None => return Err(CapacityRefusal::UnknownDevice { device }),
                Some(DeviceBudget::Unknown) => {
                    return Err(CapacityRefusal::UnknownBudget { device });
                }
                Some(&DeviceBudget::Known { bytes, reserve }) => (bytes, reserve),
            };
            let charged = ledger.charged(device);
            let available = bytes.saturating_sub(reserve);
            let needed = required
                .checked_add(charged)
                .ok_or(CapacityError::Overflow { device })?;
            if needed > available {
                return Err(CapacityRefusal::OverBudget {
                    device,
                    required,
                    charged,
                    available,
                });
            }
        }
        Ok(())
    }
}

/// What planning one candidate produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    /// The allocations the candidate retains once prepared.
    pub footprint: Footprint,
    /// Bytes per device the planner still holds after it released the candidate: nonzero only for an
    /// allocation it cannot prove freed. Charged against every later candidate.
    pub residue: BTreeMap<DeviceId, u64>,
    /// Bytes of the weight store the planning pass read: zero, because the planning device only sizes
    /// weights. A receipt that planning does not load the checkpoint once per candidate.
    pub weight_bytes_read: u64,
    /// Kernels this candidate's planning engine took through the kernel cache: only those no earlier
    /// candidate of the same [`Planner`] had already resolved, so the sum over a planning pass is the
    /// number of distinct kernels (POOT-1068).
    pub kernels_resolved: u64,
}

/// A candidate the plan step could not produce a footprint for.
#[derive(Debug)]
pub enum PlanFailure<E> {
    /// This candidate's shape is not served (the model refuses it); the next candidate may be.
    Shape(E),
    /// The planner failed for a reason no other candidate escapes.
    Fatal(E),
}

/// Why a candidate was not chosen.
#[derive(Debug)]
pub enum Rejection<E> {
    Capacity(CapacityRefusal),
    Shape(E),
}

#[derive(Debug)]
pub struct Rejected<C, E> {
    pub candidate: C,
    pub why: Rejection<E>,
}

/// The candidate [`fit`] chose, with every candidate it passed over to reach it.
#[derive(Debug)]
pub struct Fit<C, E> {
    pub chosen: C,
    pub footprint: Footprint,
    pub rejected: Vec<Rejected<C, E>>,
    pub ledger: Ledger,
}

#[derive(Debug, thiserror::Error)]
pub enum FitError<C: std::fmt::Debug, E: std::fmt::Debug + std::error::Error + 'static> {
    /// Every candidate of the declared domain was refused.
    #[error("no candidate of the declared domain fits: {} refused", rejected.len())]
    NoCandidateFits { rejected: Vec<Rejected<C, E>> },
    #[error("planning failed: {0}")]
    Plan(#[source] E),
    #[error(transparent)]
    Footprint(#[from] CapacityError),
}

/// Choose the first candidate of `domain` (preferred first) whose planned footprint fits a fresh budget.
///
/// Every candidate reached is planned and checked on its own: the search never infers one candidate's
/// size from another's, so a domain that is not monotonic is searched correctly. `observe` is called once
/// per candidate, at admission, so a budget that moved since the device opened is the one used. A
/// candidate's [`Planned::residue`] is charged to the ledger whether or not the candidate fit.
pub fn fit<C, E>(
    domain: &[C],
    mut plan: impl FnMut(&C) -> Result<Planned, PlanFailure<E>>,
    mut observe: impl FnMut() -> Budget,
) -> Result<Fit<C, E>, FitError<C, E>>
where
    C: Copy + std::fmt::Debug,
    E: std::fmt::Debug + std::error::Error + 'static,
{
    let mut ledger = Ledger::default();
    let mut rejected = Vec::new();
    for &candidate in domain {
        let planned = match plan(&candidate) {
            Ok(planned) => planned,
            Err(PlanFailure::Shape(error)) => {
                rejected.push(Rejected {
                    candidate,
                    why: Rejection::Shape(error),
                });
                continue;
            }
            Err(PlanFailure::Fatal(error)) => return Err(FitError::Plan(error)),
        };
        for (&device, &bytes) in &planned.residue {
            ledger.charge(device, bytes)?;
        }
        match observe().admit(&planned.footprint, &ledger) {
            Ok(()) => {
                return Ok(Fit {
                    chosen: candidate,
                    footprint: planned.footprint,
                    rejected,
                    ledger,
                });
            }
            Err(refusal) => rejected.push(Rejected {
                candidate,
                why: Rejection::Capacity(refusal),
            }),
        }
    }
    Err(FitError::NoCandidateFits { rejected })
}

/// Check a prepared driver's live role bytes against its plan and a fresh budget, before it serves.
///
/// A role may not hold more than it planned (a plan that under-counted admitted nothing), except
/// [`BufferRole::Staging`], which runtimes allocate privately and the reserve bounds. The live bytes, not
/// the plan, are then admitted against the budget: the plan chose the candidate, the counters prove it.
pub fn verify_prepared(
    planned: &Footprint,
    device: DeviceId,
    live: &[(BufferRole, u64)],
    budget: &Budget,
    ledger: &Ledger,
) -> Result<(), CapacityRefusal> {
    let planned_roles = planned.role_bytes(device)?;
    let mut actual = Footprint::new();
    for &(role, bytes) in live {
        if role != BufferRole::Staging {
            let planned_bytes = planned_roles
                .iter()
                .find(|(planned_role, _)| *planned_role == role)
                .map_or(0, |&(_, bytes)| bytes);
            if bytes > planned_bytes {
                return Err(CapacityRefusal::Underestimated {
                    device,
                    role,
                    planned: planned_bytes,
                    actual: bytes,
                });
            }
        }
        actual.insert(AllocationOwner {
            id: format!("live {role:?}"),
            device,
            role,
            bytes,
        })?;
    }
    budget.admit(&actual, ledger)
}

impl Driver {
    /// The live bytes per role the executor's own runtime counters report, for the roles that have
    /// allocated.
    pub fn live_role_bytes(&self) -> Vec<(BufferRole, u64)> {
        self.executor
            .stats()
            .memory
            .into_iter()
            .map(|(role, snapshot): (BufferRole, MemoryCounterSnapshot)| {
                (role, snapshot.live_bytes)
            })
            .collect()
    }

    /// The targets the executor drives, as it reports them now.
    pub fn targets(&self) -> TargetSet {
        self.executor.target_set()
    }

    /// The device memory as the executor reports it now, with the bytes this driver already holds.
    pub fn observation(&self) -> Observation {
        Observation {
            held_bytes: self
                .live_role_bytes()
                .iter()
                .fold(0u64, |sum, &(_, bytes)| sum.saturating_add(bytes)),
            ..Observation::of(&*self.executor)
        }
    }
}

/// Why a candidate could not be planned.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The production `prepare` refused the candidate: the typed driver error, unflattened.
    #[error(transparent)]
    Driver(#[from] DriverError),
    #[error(transparent)]
    Footprint(#[from] CapacityError),
    #[error(transparent)]
    SharedKernels(#[from] SharedKernelsMismatch),
}

/// Plans candidate footprints for one model on one target without touching a device.
pub struct Planner {
    handle: Arc<ModelHandle>,
    device: DeviceId,
    target: Target,
    /// Shared by every candidate this planner plans, so codegen runs once per distinct kernel.
    kernels: SharedKernels,
}

impl Planner {
    /// A planner for the single device of `targets`.
    pub fn new(handle: Arc<ModelHandle>, targets: &TargetSet) -> Result<Self, DriverError> {
        let &[(device, target)] = targets.devices() else {
            return Err(InvalidOptions::Devices {
                devices: targets.devices().len(),
            }
            .into());
        };
        Ok(Self {
            handle,
            device,
            target,
            kernels: SharedKernels::new(target.backend),
        })
    }

    /// Prepare `shapes` under `options` on a [`PlanningDevice`] and return what that allocated; with
    /// `embeds`, the `PrefillEmbeds` entries ([`Driver::prepare_embeds`]) too. The preparation is the
    /// production one; only the device is a recorder, so nothing is uploaded and nothing is left
    /// behind when this returns.
    pub fn plan(
        &self,
        options: DriverOptions,
        shapes: &ServingShapes<'_>,
        embeds: bool,
    ) -> Result<Planned, PlanError> {
        let counters = MemoryCounters::new();
        let records: Records = Arc::default();
        let device = PlanningDevice {
            target: self.target,
            counters: counters.clone(),
            records: Arc::clone(&records),
        };
        let mut driver = Driver::new(
            Arc::clone(&self.handle),
            Box::new(Engine::new(device).with_shared_kernels(self.kernels.clone())?),
            options,
        )?;
        driver.prepare(shapes)?;
        if embeds {
            driver.prepare_embeds(shapes)?;
        }
        let mut footprint = Footprint::new();
        for record in records.lock().expect("planning records").iter() {
            if record.live {
                footprint.insert(AllocationOwner {
                    id: record.id.clone(),
                    device: self.device,
                    role: record.role,
                    bytes: record.bytes,
                })?;
            }
        }
        let stats = driver.executor_stats();
        drop(driver);
        let held = counters.total_live_bytes();
        Ok(Planned {
            weight_bytes_read: stats.weight_bytes_read,
            kernels_resolved: stats.kernels_resolved,
            footprint,
            residue: if held == 0 {
                BTreeMap::new()
            } else {
                BTreeMap::from([(self.device, held)])
            },
        })
    }
}

#[derive(Debug)]
struct Record {
    id: String,
    role: BufferRole,
    bytes: u64,
    live: bool,
}

type Records = Arc<Mutex<Vec<Record>>>;

/// A [`Device`] that records every allocation the engine asks for and executes nothing. Its target is
/// the real device's, so the planner and codegen decide exactly what they decide there.
struct PlanningDevice {
    target: Target,
    counters: MemoryCounters,
    records: Records,
}

struct PlanningBuffer {
    records: Records,
    index: usize,
    _guard: AllocGuard,
}

impl Drop for PlanningBuffer {
    fn drop(&mut self) {
        if let Ok(mut records) = self.records.lock() {
            records[self.index].live = false;
        }
    }
}

/// A planning device does not execute: only loading an entry is asked of it.
#[derive(Debug, thiserror::Error)]
#[error("a planning device executes nothing")]
struct PlanningOnly;

impl Device for PlanningDevice {
    type Buffer = PlanningBuffer;
    type Kernel = ();
    type Recording = ();
    type Error = PlanningOnly;

    fn target(&self) -> Target {
        self.target
    }

    /// Sizes weights from their shapes and stored lengths: a plan needs how many bytes a weight takes, never
    /// the bytes, so planning a candidate reads and encodes none of the checkpoint.
    fn weight_upload(&self) -> WeightUpload {
        WeightUpload::SizeOnly
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.counters.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<PlanningBuffer, PlanningOnly> {
        let bytes = (elems as u64).saturating_mul(storage.element().byte_width() as u64);
        let guard = self.counters.record_alloc(role, bytes);
        let mut records = self.records.lock().expect("planning records");
        let index = records.len();
        records.push(Record {
            id: format!("{role:?}#{index}"),
            role,
            bytes,
            live: true,
        });
        Ok(PlanningBuffer {
            records: Arc::clone(&self.records),
            index,
            _guard: guard,
        })
    }

    fn load_kernel(&mut self, _key: &str, _kernel: CompiledKernel) -> Result<(), PlanningOnly> {
        Ok(())
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), PlanningOnly> {
        Ok(())
    }

    fn dispatch(&mut self, _dispatch: Dispatch<'_, Self>) -> Result<(), PlanningOnly> {
        Ok(())
    }

    fn copy(&mut self, _src: &PlanningBuffer, _dst: &PlanningBuffer) -> Result<(), PlanningOnly> {
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<()>, PlanningOnly> {
        Ok(Some(()))
    }

    fn replay(&mut self, _recording: &()) -> Result<(), PlanningOnly> {
        Err(PlanningOnly)
    }

    /// Loading an entry uploads constants; the planning device drops the bytes.
    fn write(
        &mut self,
        _dst: &PlanningBuffer,
        _offset: usize,
        _bytes: &[u8],
    ) -> Result<(), PlanningOnly> {
        Ok(())
    }

    fn read(&mut self, _src: &PlanningBuffer, _out: &mut [u8]) -> Result<(), PlanningOnly> {
        Err(PlanningOnly)
    }

    fn synchronize(&mut self) -> Result<(), PlanningOnly> {
        Err(PlanningOnly)
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), PlanningOnly> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D0: DeviceId = DeviceId(0);
    const D1: DeviceId = DeviceId(1);

    fn owner(id: &str, device: DeviceId, role: BufferRole, bytes: u64) -> AllocationOwner {
        AllocationOwner {
            id: id.to_string(),
            device,
            role,
            bytes,
        }
    }

    fn known(bytes: u64) -> DeviceBudget {
        DeviceBudget::Known { bytes, reserve: 0 }
    }

    /// The fake owners of a serving candidate: weights shared by both entries (declared by each), a
    /// window-clamped KV (`min(max_seq, WINDOW)` positions, so size is not affine in `max_seq`), two
    /// distinct live arenas (prefill and decode), and a draft model's own weights on the second device.
    const WINDOW: u64 = 1024;
    const KV_PER_POSITION: u64 = 4;
    const PREFILL_ARENA: u64 = 300;
    const DECODE_ARENA: u64 = 100;
    const WEIGHTS: u64 = 1000;
    const DRAFT_WEIGHTS: u64 = 200;

    fn candidate_footprint(max_seq: u64) -> Footprint {
        let mut footprint = Footprint::new();
        for entry in ["prefill", "decode"] {
            // Every entry names the one shared weight allocation; it is counted once.
            footprint
                .insert(owner("weights", D0, BufferRole::Weight, WEIGHTS))
                .unwrap();
            let arena = if entry == "prefill" {
                PREFILL_ARENA
            } else {
                DECODE_ARENA
            };
            footprint
                .insert(owner(
                    &format!("arena {entry}"),
                    D0,
                    BufferRole::Activation,
                    arena,
                ))
                .unwrap();
        }
        footprint
            .insert(owner(
                "kv",
                D0,
                BufferRole::State,
                max_seq.min(WINDOW) * KV_PER_POSITION,
            ))
            .unwrap();
        footprint
            .insert(owner(
                "draft weights",
                D1,
                BufferRole::Weight,
                DRAFT_WEIGHTS,
            ))
            .unwrap();
        footprint
    }

    /// SC-004: shared owners once, distinct live arenas summed, a window-clamped size not extrapolated,
    /// and the draft owner on its own device.
    #[test]
    fn composition_counts_shared_once_and_sums_distinct_arenas() {
        let footprint = candidate_footprint(512);
        assert_eq!(
            footprint.device_bytes(D0).unwrap(),
            WEIGHTS + PREFILL_ARENA + DECODE_ARENA + 512 * KV_PER_POSITION,
            "weights once, both arenas summed (not their max), kv at its own size"
        );
        assert_eq!(footprint.device_bytes(D1).unwrap(), DRAFT_WEIGHTS);
        // Above the window the kv stops growing: two samples below it would extrapolate past it.
        assert_eq!(
            candidate_footprint(4096).device_bytes(D0).unwrap(),
            candidate_footprint(1024).device_bytes(D0).unwrap()
        );
    }

    /// SC-004: the largest candidate of the declared domain that fits is admitted and the next one is
    /// refused, with uneven per-device capacities: device 1 is small, so it alone refuses a draft that
    /// device 0's headroom could hold.
    #[test]
    fn fit_admits_the_largest_fitting_candidate_and_refuses_the_next() {
        let weights_and_arenas = WEIGHTS + PREFILL_ARENA + DECODE_ARENA;
        // 700 positions fit, 800 do not: the budget is exactly 700 positions of kv above the rest.
        let device0 = weights_and_arenas + 700 * KV_PER_POSITION;
        let budget = Budget::new()
            .with(D0, known(device0))
            .with(D1, known(DRAFT_WEIGHTS));
        let domain = [800u64, 700, 600];
        let fitted = fit(
            &domain,
            |&max_seq| {
                Ok::<_, PlanFailure<PlanError>>(Planned {
                    footprint: candidate_footprint(max_seq),
                    residue: BTreeMap::new(),
                    weight_bytes_read: 0,
                    kernels_resolved: 0,
                })
            },
            || budget.clone(),
        )
        .expect("a candidate fits");
        assert_eq!(fitted.chosen, 700);
        assert_eq!(fitted.rejected.len(), 1, "only the next-larger was refused");
        assert_eq!(fitted.rejected[0].candidate, 800);
        let Rejection::Capacity(CapacityRefusal::OverBudget {
            device,
            required,
            available,
            ..
        }) = &fitted.rejected[0].why
        else {
            panic!(
                "the refusal is typed over-budget: {:?}",
                fitted.rejected[0].why
            );
        };
        assert_eq!(
            (*device, *required, *available),
            (D0, device0 + 100 * KV_PER_POSITION, device0)
        );
    }

    /// SC-004: a budget is per device. Device 1 is a byte short for the draft; the devices' summed
    /// capacity would hold the plan, so a global figure admits what one device cannot hold.
    #[test]
    fn a_short_device_refuses_even_when_the_sum_fits() {
        let footprint = candidate_footprint(512);
        let total = footprint.device_bytes(D0).unwrap() + footprint.device_bytes(D1).unwrap();
        let budget = Budget::new()
            .with(D0, known(total))
            .with(D1, known(DRAFT_WEIGHTS - 1));
        assert_eq!(
            budget.admit(&footprint, &Ledger::default()),
            Err(CapacityRefusal::OverBudget {
                device: D1,
                required: DRAFT_WEIGHTS,
                charged: 0,
                available: DRAFT_WEIGHTS - 1,
            })
        );
    }

    /// SC-004: an unknown budget, a device with no budget entry, an overflow and conflicting owners are
    /// each their own typed outcome.
    #[test]
    fn unknown_budget_overflow_and_conflicts_are_typed() {
        let footprint = candidate_footprint(512);
        let unknown = Budget::new()
            .with(D0, DeviceBudget::Unknown)
            .with(D1, known(u64::MAX));
        assert_eq!(
            unknown.admit(&footprint, &Ledger::default()),
            Err(CapacityRefusal::UnknownBudget { device: D0 })
        );
        let missing = Budget::new().with(D0, known(u64::MAX));
        assert_eq!(
            missing.admit(&footprint, &Ledger::default()),
            Err(CapacityRefusal::UnknownDevice { device: D1 })
        );
        let mut huge = Footprint::new();
        huge.insert(owner("a", D0, BufferRole::State, u64::MAX))
            .unwrap();
        huge.insert(owner("b", D0, BufferRole::State, 1)).unwrap();
        assert_eq!(
            huge.device_bytes(D0),
            Err(CapacityError::Overflow { device: D0 })
        );
        assert_eq!(
            Budget::new()
                .with(D0, known(u64::MAX))
                .admit(&huge, &Ledger::default()),
            Err(CapacityRefusal::Footprint(CapacityError::Overflow {
                device: D0
            }))
        );
        let mut conflict = Footprint::new();
        conflict
            .insert(owner("kv", D0, BufferRole::State, 8))
            .unwrap();
        assert_eq!(
            conflict.insert(owner("kv", D0, BufferRole::State, 9)),
            Err(CapacityError::ConflictingOwner {
                id: "kv".to_string(),
                device: D0,
                first_role: BufferRole::State,
                first_bytes: 8,
                second_role: BufferRole::State,
                second_bytes: 9,
            })
        );
    }

    fn sized(bytes: u64, residue: u64) -> Planned {
        let mut footprint = Footprint::new();
        footprint
            .insert(owner("all", D0, BufferRole::State, bytes))
            .unwrap();
        Planned {
            footprint,
            residue: if residue == 0 {
                BTreeMap::new()
            } else {
                BTreeMap::from([(D0, residue)])
            },
            weight_bytes_read: 0,
            kernels_resolved: 0,
        }
    }

    /// SC-006: admission reads the budget at admission, not the one the device reported when it opened.
    /// The open-time budget (1000) holds the first candidate; the fresh one (600) does not, and the
    /// search stops at the first fit, planning three candidates of four.
    #[test]
    fn admission_uses_the_fresh_observation_not_the_open_time_budget() {
        let mut observations = 0;
        let mut planned = Vec::new();
        let fitted = fit(
            &[900u64, 700, 500, 300],
            |&bytes| {
                planned.push(bytes);
                Ok::<_, PlanFailure<PlanError>>(sized(bytes, 0))
            },
            || {
                observations += 1;
                Budget::new().with(D0, known(600)) // the open-time value was 1000
            },
        )
        .expect("500 fits the fresh budget");
        assert_eq!(fitted.chosen, 500);
        assert_eq!(
            planned,
            [900, 700, 500],
            "the search is bounded by the first fit"
        );
        assert_eq!(observations, 3, "one fresh observation per admission");
        let refusals: Vec<_> = fitted
            .rejected
            .iter()
            .map(|r| match &r.why {
                Rejection::Capacity(CapacityRefusal::OverBudget { available, .. }) => {
                    (r.candidate, *available)
                }
                other => panic!("typed over-budget expected: {other:?}"),
            })
            .collect();
        assert_eq!(refusals, [(900, 600), (700, 600)]);
    }

    /// SC-006: bytes a refused candidate's planner could not prove released stay charged and reduce
    /// what the later candidates may use. 550 fits 600 alone and does not once 100 stay charged.
    #[test]
    fn residue_of_a_refused_candidate_stays_charged() {
        let result = fit(
            &[900u64, 550],
            |&bytes| {
                Ok::<_, PlanFailure<PlanError>>(sized(bytes, if bytes == 900 { 100 } else { 0 }))
            },
            || Budget::new().with(D0, known(600)),
        );
        let Err(FitError::NoCandidateFits { rejected }) = result else {
            panic!("550 + 100 charged exceeds 600: {result:?}");
        };
        assert_eq!(
            rejected.iter().map(|r| r.candidate).collect::<Vec<_>>(),
            [900, 550]
        );
        let Rejection::Capacity(CapacityRefusal::OverBudget {
            required, charged, ..
        }) = &rejected[1].why
        else {
            panic!("over-budget expected: {:?}", rejected[1].why);
        };
        assert_eq!((*required, *charged), (550, 100));
    }

    /// Completed temporary owners go back to the initial count: a candidate that planned and was
    /// refused leaves the ledger empty.
    #[test]
    fn a_refused_candidate_with_no_residue_charges_nothing() {
        let fitted = fit(
            &[900u64, 500],
            |&bytes| Ok::<_, PlanFailure<PlanError>>(sized(bytes, 0)),
            || Budget::new().with(D0, known(600)),
        )
        .unwrap();
        assert_eq!(fitted.ledger.charged(D0), 0);
    }

    /// A shape refusal skips its candidate; a fatal plan failure ends the search.
    #[test]
    fn shape_refusals_skip_and_fatal_failures_end_the_search() {
        let fitted = fit(
            &[4u64, 1],
            |&rows| {
                if rows > 1 {
                    Err(PlanFailure::Shape(PlanError::Footprint(
                        CapacityError::Overflow { device: D0 },
                    )))
                } else {
                    Ok(sized(10, 0))
                }
            },
            || Budget::new().with(D0, known(100)),
        )
        .unwrap();
        assert_eq!(fitted.chosen, 1);
        assert!(matches!(fitted.rejected[0].why, Rejection::Shape(_)));
        let fatal = fit(
            &[4u64, 1],
            |_| {
                Err::<Planned, _>(PlanFailure::Fatal(PlanError::Footprint(
                    CapacityError::Overflow { device: D0 },
                )))
            },
            || Budget::new().with(D0, known(100)),
        );
        assert!(matches!(fatal, Err(FitError::Plan(_))));
    }

    /// The live budget is what the runtime says is free plus what this process holds, under the open-time
    /// ceiling; a runtime that cannot say is unknown whatever the ceiling. Mutations: drop `held_bytes`
    /// (the second assertion), ignore the ceiling (the third), or fall back to the ceiling on `None` (the
    /// fourth).
    #[test]
    fn observed_budget_is_free_plus_held_under_the_ceiling_and_unknown_without_a_live_figure() {
        let observe = |ceiling: Option<u64>, free: Option<u64>, held: u64| {
            let mut target = Target {
                backend: poot_target::Backend::SpirvVulkan,
                caps: poot_target::DeviceCaps::wgpu_rdna3_igpu(),
            };
            target.caps.memory_budget_bytes = ceiling;
            Budget::observed(
                &Observation {
                    targets: TargetSet::single(D0, target),
                    free_bytes: free,
                    held_bytes: held,
                },
                10,
            )
        };
        assert_eq!(observe(Some(1000), Some(300), 0).available(D0), Some(290));
        assert_eq!(observe(Some(1000), Some(300), 200).available(D0), Some(490));
        assert_eq!(observe(Some(400), Some(300), 200).available(D0), Some(390));
        assert_eq!(observe(None, Some(300), 0).available(D0), Some(290));
        assert_eq!(observe(Some(1000), None, 200).available(D0), None);
    }

    /// SC-005: live bytes over a role's plan are refused (the plan under-counted), and live bytes are
    /// then admitted against the fresh budget.
    #[test]
    fn verify_prepared_refuses_an_underestimate_and_an_over_budget_live_total() {
        let mut planned = Footprint::new();
        planned
            .insert(owner("kv", D0, BufferRole::State, 100))
            .unwrap();
        let roomy = Budget::new().with(D0, known(1000));
        let ledger = Ledger::default();
        assert_eq!(
            verify_prepared(&planned, D0, &[(BufferRole::State, 100)], &roomy, &ledger),
            Ok(())
        );
        assert_eq!(
            verify_prepared(&planned, D0, &[(BufferRole::State, 101)], &roomy, &ledger),
            Err(CapacityRefusal::Underestimated {
                device: D0,
                role: BufferRole::State,
                planned: 100,
                actual: 101,
            })
        );
        // Staging is runtime-private: bounded by the reserve, not by the plan.
        assert_eq!(
            verify_prepared(
                &planned,
                D0,
                &[(BufferRole::State, 100), (BufferRole::Staging, 50)],
                &roomy,
                &ledger
            ),
            Ok(())
        );
        let tight = Budget::new().with(D0, known(120));
        assert_eq!(
            verify_prepared(
                &planned,
                D0,
                &[(BufferRole::State, 100), (BufferRole::Staging, 50)],
                &tight,
                &ledger
            ),
            Err(CapacityRefusal::OverBudget {
                device: D0,
                required: 150,
                charged: 0,
                available: 120,
            })
        );
    }
}
