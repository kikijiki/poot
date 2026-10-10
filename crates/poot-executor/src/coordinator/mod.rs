//! The captured coordinator: one [`Executor`] over several devices (target architecture section 4).
//!
//! A [`Coordinator`] drives one [`Engine`] per device of its target set and runs the multi-device
//! [`StagedProgram`] `compile_staged` produces: each device loads its own stages, a boundary value that
//! crosses devices is a transfer row, and a step is the stages' recordings replayed in stage order.
//! Nothing here interprets a graph: binding, the exact export/import ABI, state commit and faults are
//! the engine's shared stage walk, and the coordinator adds only what sits *between* stages.
//!
//! A stage's recording is one scoped transaction, exactly the engine's: `begin`, the stage's own
//! dispatches and state commit, `finish`. The coordinator adds one thing inside that transaction (the
//! engine's `StageSeams`): for each transfer out of the stage, a peer copy of the exported value into
//! the consumer device's import buffer, after the last dispatch. Ordering between devices is by event
//! at replay: right after a stage's recording is replayed its device records the stage's events, and
//! right before a consumer's recording is replayed its device waits for them, so every wait names the
//! record enqueued for the same step, producer first. A device may therefore hold several recordings of
//! one entry (rank 0, rank 1, rank 0 again); the entry still has one recording session per device, and
//! a warm step creates nothing.
//!
//! The step's one host wait is the completion at the end: every device is synchronized once, in rank
//! order, after every stage has been enqueued. There is no host sync between stages (a pooled program
//! with sync points is refused; `compile_staged` already refuses to pool across devices).
//!
//! Admission is complete before any device work: the communication rows ([`schedule`]), the stage and
//! device placement, the directed routes (a route that would need host staging is refused, there is no
//! fallback), and a conservative bound of the support allocations against each device's free memory.

mod schedule;

use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::ValidationOutputs;
use poot_graph_plan::multi_device::{PlanNodeId, TopologyError};
use poot_graph_plan::{DeviceId, StageId, StagedProgram, TargetSet, TensorParallel};
use poot_quant::weights::{WeightMap, WeightStore};

use crate::device::{Collectives, Device, DeviceTime, NativeCounters};
use crate::engine::{Engine, EntryPart, Outgoing, StageSeams, device_error};
use crate::error::{BindError, ExecError, LoadError};
use crate::{
    BufferRole, DeviceMemory, EntryId, ExecutableId, Executor, ExecutorStats, HostSync,
    OutputSource, StageStats, StateScope, StepInputs, StepOutputs,
};

pub use schedule::ScheduleError;

/// Why a coordinator refused a program or a device. Every variant is raised before any device work
/// of the refused operation.
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("a coordinator needs at least one device")]
    NoDevices,
    #[error(transparent)]
    Topology(#[from] TopologyError),
    #[error("stage {stage:?} is placed on {device:?}, which this coordinator does not drive")]
    UnknownDevice { stage: StageId, device: DeviceId },
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
    #[error(
        "boundary transfer {row:?} from stage {from:?} to stage {to:?} disagrees with its \
         communication row: {why}"
    )]
    RowMismatch {
        row: PlanNodeId,
        from: StageId,
        to: StageId,
        why: &'static str,
    },
    #[error("boundary transfer {row:?} is read by stage {to:?}, which does not follow {from:?}")]
    StageOrder {
        row: PlanNodeId,
        from: StageId,
        to: StageId,
    },
    #[error("the program has {0} host sync point(s); a multi-device program has none")]
    HostSync(usize),
    #[error("route {from:?} -> {to:?} is not available: {source}")]
    Route {
        from: DeviceId,
        to: DeviceId,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error(
        "device {device:?}: recording support is bounded at {need} bytes and only {free} are free"
    )]
    SupportBound {
        device: DeviceId,
        need: u64,
        free: u64,
    },
    /// The device cannot report how much memory is free, so the entry's support allocations cannot be
    /// bounded: an unknown budget is refused, never read as unlimited.
    #[error(
        "device {device:?} cannot report its free memory, so recording support cannot be bounded"
    )]
    UnknownFreeMemory { device: DeviceId },
}

impl From<CoordinatorError> for ExecError {
    fn from(error: CoordinatorError) -> Self {
        ExecError::Coordinator(Box::new(error))
    }
}

/// What one device did for the coordinator's entries, by independent counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCounters {
    pub device: DeviceId,
    /// Recording sessions: once per entry that has stages on this device, when the entry first
    /// records. A step of an unchanged entry adds none.
    pub sessions: u64,
    /// Recordings published, one per stage (one native graph each on a graph backend).
    pub phases: u64,
    /// Recordings submitted to this device.
    pub launches: u64,
    /// Planned dispatches the replayed recordings hold.
    pub dispatches: u64,
    /// Peer copies this device issued as the sender.
    pub transfers: u64,
    /// The device runtime's own counters of what it created and launched.
    pub native: NativeCounters,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorCounters {
    pub devices: Vec<DeviceCounters>,
    /// Steps completed.
    pub steps: u64,
}

/// What preparing one entry on one device is planned to allocate beyond the engine's own buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupportPlan {
    /// Bytes of the import buffers the device's incoming transfers write: exact.
    pub import_bytes: u64,
    /// Events the device's outgoing transfers publish: exact.
    pub events: u64,
    /// Recordings (native graphs) the device holds for the entry: exact.
    pub graphs: u64,
    /// The conservative total: `import_bytes` plus the device's per-event and per-graph bounds.
    pub bound_bytes: u64,
}

/// What preparing one entry on one device actually allocated, by the device's own counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupportActual {
    /// Live bytes of transfer staging the device's memory counters gained while the entry loaded.
    pub import_bytes: u64,
    /// Events the device created while the entry loaded.
    pub events: u64,
    /// Native graphs the device instantiated while the entry recorded. Zero until the first step.
    pub graph_instantiations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSupport {
    pub device: DeviceId,
    pub planned: SupportPlan,
    pub actual: SupportActual,
}

/// Where a stage of the program runs: its device's rank, and its position among that rank's stages.
struct Placed {
    rank: usize,
    local: usize,
}

/// One boundary value crossing devices: the global stage indices it connects, its row, and the two
/// events that order it. `ready` is the consumer device's: recorded at the start of each step, after
/// everything already enqueued for that device (input writes, an allocation's zeroing, the previous
/// step), and waited for by the producer's device before it replays the stage whose copy writes the
/// import buffer. `event` is the producer device's: recorded right after the producing stage replays,
/// waited for by the consumer's device before it replays the consuming stage.
struct Hop<D: Collectives> {
    row: PlanNodeId,
    from_stage: usize,
    to_stage: usize,
    from_rank: usize,
    to_rank: usize,
    event: D::Event,
    ready: D::Event,
}

struct Entry<D: Collectives> {
    /// The entry on each rank that has stages of it.
    per_rank: Vec<Option<EntryId>>,
    stages: Vec<Placed>,
    hops: Vec<Hop<D>>,
    /// The rank of the program's final stage: the entry's output lives there.
    output_rank: usize,
    recorded: bool,
    replays: u64,
    support: Vec<DeviceSupport>,
}

struct Executable<D: Collectives> {
    /// The executable on each rank's engine.
    per_rank: Vec<ExecutableId>,
    entries: Vec<Option<Entry<D>>>,
}

/// One [`Executor`] over several devices, each driven by its own [`Engine`].
pub struct Coordinator<D: Device + Collectives> {
    // Field order is drop order: the entries (and the events they own) drop before the engines
    // that own the devices.
    executables: Vec<Option<Executable<D>>>,
    engines: Vec<Engine<D>>,
    devices: Vec<DeviceId>,
    counters: Vec<DeviceCounters>,
    steps: u64,
}

impl<D: Device + Collectives> Coordinator<D> {
    /// A coordinator over `devices`, one engine each, in rank order.
    pub fn new(devices: Vec<(DeviceId, D)>) -> Result<Self, CoordinatorError> {
        if devices.is_empty() {
            return Err(CoordinatorError::NoDevices);
        }
        TargetSet::new(
            devices
                .iter()
                .map(|(id, device)| (*id, device.target()))
                .collect(),
        )?;
        let ids: Vec<DeviceId> = devices.iter().map(|(id, _)| *id).collect();
        let counters = ids
            .iter()
            .map(|&device| DeviceCounters {
                device,
                sessions: 0,
                phases: 0,
                launches: 0,
                dispatches: 0,
                transfers: 0,
                native: NativeCounters::default(),
            })
            .collect();
        let engines = devices
            .into_iter()
            .map(|(id, device)| Engine::new(device).on_device(id))
            .collect();
        Ok(Self {
            executables: Vec::new(),
            engines,
            devices: ids,
            counters,
            steps: 0,
        })
    }

    /// The independent counters of what each device did: sessions, recordings, launches, dispatches,
    /// transfers, and the device runtime's native counts.
    pub fn counters(&self) -> CoordinatorCounters {
        CoordinatorCounters {
            devices: self
                .counters
                .iter()
                .zip(&self.engines)
                .map(|(counters, engine)| DeviceCounters {
                    native: engine.device().native_counters(),
                    ..*counters
                })
                .collect(),
            steps: self.steps,
        }
    }

    /// What `entry` is planned to allocate on each device that holds a stage of it, beside what it
    /// actually allocated.
    pub fn support(
        &self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<Vec<DeviceSupport>, ExecError> {
        Ok(self.entry(exe, entry)?.support.clone())
    }

    /// Load one executable whose weights each device reads through its own map: engine `i` binds
    /// `maps[i]`. Nothing is uploaded here; a refusal unloads what the devices before it loaded.
    fn load_ranked(
        &mut self,
        store: Arc<WeightStore>,
        maps: Vec<Arc<WeightMap>>,
    ) -> Result<ExecutableId, ExecError> {
        self.check_usable()?;
        let mut per_rank: Vec<ExecutableId> = Vec::with_capacity(self.engines.len());
        for (engine, map) in self.engines.iter_mut().zip(maps) {
            match engine.load_weights(Arc::clone(&store), map) {
                Ok(exe) => per_rank.push(exe),
                Err(error) => {
                    for (engine, exe) in self.engines.iter_mut().zip(per_rank) {
                        let _ = engine.unload(exe);
                    }
                    return Err(error);
                }
            }
        }
        self.executables.push(Some(Executable {
            per_rank,
            entries: Vec::new(),
        }));
        Ok(ExecutableId::new(self.executables.len() - 1))
    }

    /// Load the weights of a tensor-parallel program: each device that is a rank of `tensor` binds
    /// `weights` cut to its own share ([`WeightMap::rank_view`]), every other device the whole map.
    /// Each share is uploaded once, by the first entry that reads it, and stays resident.
    pub fn load_tensor_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
        tensor: &TensorParallel,
    ) -> Result<ExecutableId, ExecError> {
        let world = u32::try_from(tensor.devices.len())
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or(CoordinatorError::NoDevices)?;
        let mut maps = Vec::with_capacity(self.engines.len());
        for &device in &self.devices {
            maps.push(
                match tensor.devices.iter().position(|&rank| rank == device) {
                    Some(rank) => Arc::new(
                        weights
                            .rank_view(&store, rank, world, &tensor.shares)
                            .map_err(LoadError::from)?,
                    ),
                    None => Arc::clone(&weights),
                },
            );
        }
        self.load_ranked(store, maps)
    }

    fn rank_of(&self, device: DeviceId) -> Option<usize> {
        self.devices.iter().position(|&id| id == device)
    }

    fn check_usable(&self) -> Result<(), ExecError> {
        if self.engines.iter().any(Engine::is_poisoned) {
            return Err(ExecError::Poisoned);
        }
        Ok(())
    }

    fn executable(&self, exe: ExecutableId) -> Result<&Executable<D>, ExecError> {
        self.executables
            .get(exe.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| LoadError::UnknownHandle.into())
    }

    fn entry(&self, exe: ExecutableId, entry: EntryId) -> Result<&Entry<D>, ExecError> {
        self.executable(exe)?
            .entries
            .get(entry.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| LoadError::UnknownHandle.into())
    }

    /// Wait for every device to finish everything it was handed. Nothing another device's graph may
    /// still address is released before this has returned `Ok` for all of them.
    fn quiesce_all(&mut self) -> Result<(), ExecError> {
        let mut first = None;
        for engine in &mut self.engines {
            if let Err(error) = engine.quiesce() {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Open the directed routes `program` needs and bound the support allocations, both before the
    /// first allocation of the entry.
    fn admit_resources(&mut self, plan: &Plan) -> Result<(), ExecError> {
        for &(from, to) in &plan.routes {
            let (sender, receiver) = pair_mut(&mut self.engines, from, to);
            sender
                .device_mut()
                .open_route(receiver.device())
                .map_err(|source| CoordinatorError::Route {
                    from: self.devices[from],
                    to: self.devices[to],
                    source: Box::new(source),
                })?;
        }
        for (rank, support) in plan.support.iter().enumerate() {
            let Some(free) = self.engines[rank].device().free_memory_bytes() else {
                return Err(CoordinatorError::UnknownFreeMemory {
                    device: self.devices[rank],
                }
                .into());
            };
            if support.bound_bytes > free {
                return Err(CoordinatorError::SupportBound {
                    device: self.devices[rank],
                    need: support.bound_bytes,
                    free,
                }
                .into());
            }
        }
        Ok(())
    }
}

/// One boundary transfer of a checked program: its row and the stage indices it joins.
#[derive(Clone, Copy)]
struct PlannedTransfer {
    row: PlanNodeId,
    from_stage: usize,
    to_stage: usize,
}

/// The checked shape of one multi-device program on this coordinator's devices.
struct Plan {
    /// The rank of each stage, in stage order.
    ranks: Vec<usize>,
    /// Each transfer, in program order.
    transfers: Vec<PlannedTransfer>,
    /// The distinct directed routes the transfers use, as ranks.
    routes: Vec<(usize, usize)>,
    /// The planned support per rank.
    support: Vec<SupportPlan>,
}

impl<D: Device + Collectives> Coordinator<D> {
    fn plan(&self, program: &StagedProgram<ValidationOutputs>) -> Result<Plan, ExecError> {
        if !program.sync_points().is_empty() {
            return Err(CoordinatorError::HostSync(program.sync_points().len()).into());
        }
        let mut ranks = Vec::new();
        for (stage, device, _) in program.stages() {
            let rank = self
                .rank_of(device)
                .ok_or(CoordinatorError::UnknownDevice { stage, device })?;
            ranks.push(rank);
        }
        let schedule = schedule::admit(program.comm()).map_err(CoordinatorError::from)?;
        let mut transfers = Vec::with_capacity(program.transfers().len());
        let mut routes: Vec<(usize, usize)> = Vec::new();
        let mut support = vec![SupportPlan::default(); self.engines.len()];
        for transfer in program.transfers() {
            let mismatch = |why| CoordinatorError::RowMismatch {
                row: transfer.row,
                from: transfer.from,
                to: transfer.to,
                why,
            };
            let row = schedule
                .row(transfer.row)
                .ok_or_else(|| mismatch("the plan has no such row"))?;
            let (from, to) = (transfer.from.0 as usize, transfer.to.0 as usize);
            if from >= ranks.len() || to >= ranks.len() {
                return Err(mismatch("a stage the program does not have").into());
            }
            if from >= to {
                return Err(CoordinatorError::StageOrder {
                    row: transfer.row,
                    from: transfer.from,
                    to: transfer.to,
                }
                .into());
            }
            if row.from != self.devices[ranks[from]] || row.to != self.devices[ranks[to]] {
                return Err(mismatch("its participants are not its stages' devices").into());
            }
            if row.bytes as u64 != transfer.bytes {
                return Err(mismatch("its byte length is not the value's planned bytes").into());
            }
            if ranks[from] == ranks[to] {
                return Err(mismatch("both stages run on one device").into());
            }
            let route = (ranks[from], ranks[to]);
            if !routes.contains(&route) {
                routes.push(route);
            }
            support[ranks[to]].import_bytes += transfer.bytes;
            support[ranks[from]].events += 1;
            support[ranks[to]].events += 1;
            transfers.push(PlannedTransfer {
                row: transfer.row,
                from_stage: from,
                to_stage: to,
            });
        }
        for &rank in &ranks {
            support[rank].graphs += 1;
        }
        for (rank, plan) in support.iter_mut().enumerate() {
            let bound = self.engines[rank].device().support_bound();
            plan.bound_bytes = plan
                .import_bytes
                .saturating_add(plan.events.saturating_mul(bound.event_bytes))
                .saturating_add(plan.graphs.saturating_mul(bound.graph_bytes));
        }
        Ok(Plan {
            ranks,
            transfers,
            routes,
            support,
        })
    }
}

/// Two engines of the slice: the first mutably, the second shared.
fn pair_mut<D: Device>(
    engines: &mut [Engine<D>],
    first: usize,
    second: usize,
) -> (&mut Engine<D>, &Engine<D>) {
    debug_assert_ne!(first, second);
    if first < second {
        let (head, tail) = engines.split_at_mut(second);
        (&mut head[first], &tail[0])
    } else {
        let (head, tail) = engines.split_at_mut(first);
        (&mut tail[0], &head[second])
    }
}

/// The engines of the slice other than rank `rank`, shared, beside rank `rank` mutably.
struct Peers<'e, D: Device> {
    before: &'e [Engine<D>],
    after: &'e [Engine<D>],
}

impl<'e, D: Device> Peers<'e, D> {
    fn get(&self, rank: usize) -> &'e Engine<D> {
        if rank < self.before.len() {
            &self.before[rank]
        } else {
            &self.after[rank - self.before.len() - 1]
        }
    }
}

fn split_rank<D: Device>(engines: &mut [Engine<D>], rank: usize) -> (&mut Engine<D>, Peers<'_, D>) {
    let (before, rest) = engines.split_at_mut(rank);
    let (this, after) = rest
        .split_first_mut()
        .expect("a rank names one of the engines");
    (
        this,
        Peers {
            before: &*before,
            after: &*after,
        },
    )
}

/// The copies recorded after one stage's walk: each outgoing value into its consumer's import buffer.
struct PeerSeams<'a, D: Collectives> {
    sends: Vec<Handoff<'a, D>>,
}

struct Handoff<'a, D: Collectives> {
    row: PlanNodeId,
    destination: &'a D::Buffer,
}

impl<D: Collectives> StageSeams<D> for PeerSeams<'_, D> {
    fn after_walk(&mut self, device: &mut D, outgoing: &[Outgoing<'_, D>]) -> Result<(), D::Error> {
        for send in &self.sends {
            let source = outgoing
                .iter()
                .find(|out| out.row == send.row)
                .expect("the engine loaded an outgoing value for every transfer of the stage");
            device.peer_copy(source.buffer, send.destination, source.bytes)?;
        }
        Ok(())
    }
}

impl<D: Device + Collectives> Executor for Coordinator<D> {
    fn target_set(&self) -> TargetSet {
        TargetSet::new(
            self.devices
                .iter()
                .zip(&self.engines)
                .map(|(&id, engine)| (id, engine.device().target()))
                .collect(),
        )
        .expect("the devices were checked distinct at construction")
    }

    fn device_memory(&self) -> Vec<DeviceMemory> {
        self.devices
            .iter()
            .zip(&self.engines)
            .map(|(&device, engine)| DeviceMemory {
                device,
                free_bytes: engine.device().free_memory_bytes(),
                live: engine.device().memory(),
            })
            .collect()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        self.engines
            .iter()
            .map(|engine| engine.device().free_memory_bytes())
            .try_fold(u64::MAX, |least, free| Some(least.min(free?)))
    }

    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError> {
        let maps = vec![weights; self.engines.len()];
        self.load_ranked(store, maps)
    }

    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram<ValidationOutputs>,
    ) -> Result<EntryId, ExecError> {
        self.check_usable()?;
        let per_rank_exe = self.executable(exe)?.per_rank.clone();
        let plan = self.plan(program)?;
        self.admit_resources(&plan)?;

        // One event per transfer, owned by the sending device: created before any recording opens.
        let before: Vec<NativeCounters> = self
            .engines
            .iter()
            .map(|engine| engine.device().native_counters())
            .collect();
        let staging_before: Vec<u64> = self
            .engines
            .iter()
            .map(|engine| staging_bytes(engine.device()))
            .collect();
        let mut hops = Vec::with_capacity(plan.transfers.len());
        for &PlannedTransfer {
            row,
            from_stage,
            to_stage,
        } in &plan.transfers
        {
            let (from_rank, to_rank) = (plan.ranks[from_stage], plan.ranks[to_stage]);
            let mut create = |rank: usize| -> Result<D::Event, ExecError> {
                let engine = &mut self.engines[rank];
                let backend = engine.device().target().backend;
                engine
                    .device_mut()
                    .create_event()
                    .map_err(device_error(backend))
            };
            let event = create(from_rank)?;
            let ready = create(to_rank)?;
            hops.push(Hop::<D> {
                row,
                from_stage,
                to_stage,
                from_rank,
                to_rank,
                event,
                ready,
            });
        }

        // Each device loads its own stages; a refusal removes what the devices before it added.
        let mut per_rank: Vec<Option<EntryId>> = vec![None; self.engines.len()];
        let mut stages = Vec::with_capacity(plan.ranks.len());
        let mut local_counts = vec![0usize; self.engines.len()];
        for &rank in &plan.ranks {
            stages.push(Placed {
                rank,
                local: local_counts[rank],
            });
            local_counts[rank] += 1;
        }
        for rank in 0..self.engines.len() {
            if local_counts[rank] == 0 {
                continue;
            }
            let part = EntryPart {
                stages: program
                    .stages()
                    .zip(&plan.ranks)
                    .filter(|&(_, &stage_rank)| stage_rank == rank)
                    .map(|((stage, _, program), _)| (stage, program))
                    .collect(),
                links: program.links(),
                transfers: program.transfers(),
                sync_points: &[],
            };
            match self.engines[rank].add_part(per_rank_exe[rank], &part) {
                Ok(entry) => per_rank[rank] = Some(entry),
                Err(error) => {
                    for (engine, (exe, entry)) in self
                        .engines
                        .iter_mut()
                        .zip(per_rank_exe.iter().zip(&per_rank))
                    {
                        if let Some(entry) = entry {
                            let _ = engine.remove_entry(*exe, *entry);
                        }
                    }
                    return Err(error);
                }
            }
        }

        // Every transfer's import must exist on its consumer's device, or the program would copy
        // into nothing.
        for hop in &hops {
            let entry = per_rank[hop.to_rank].expect("a transfer's consumer rank holds the entry");
            if let Err(error) =
                self.engines[hop.to_rank].import_buffer(per_rank_exe[hop.to_rank], entry, hop.row)
            {
                for (engine, (exe, entry)) in self
                    .engines
                    .iter_mut()
                    .zip(per_rank_exe.iter().zip(&per_rank))
                {
                    if let Some(entry) = entry {
                        let _ = engine.remove_entry(*exe, *entry);
                    }
                }
                return Err(error);
            }
        }

        let support = (0..self.engines.len())
            .filter(|&rank| local_counts[rank] > 0)
            .map(|rank| DeviceSupport {
                device: self.devices[rank],
                planned: plan.support[rank],
                actual: SupportActual {
                    import_bytes: staging_bytes(self.engines[rank].device())
                        .saturating_sub(staging_before[rank]),
                    events: self.engines[rank]
                        .device()
                        .native_counters()
                        .events
                        .saturating_sub(before[rank].events),
                    graph_instantiations: 0,
                },
            })
            .collect();
        let output_rank = *plan.ranks.last().expect("a program has a stage");
        let executable = self.executables[exe.index()]
            .as_mut()
            .expect("checked above");
        executable.entries.push(Some(Entry {
            per_rank,
            stages,
            hops,
            output_rank,
            recorded: false,
            replays: 0,
            support,
        }));
        Ok(EntryId::new(executable.entries.len() - 1))
    }

    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        _sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        self.check_usable()?;
        let per_rank_exe = self.executable(exe)?.per_rank.clone();
        let (per_rank, recorded) = {
            let entry = self.entry(exe, entry)?;
            (entry.per_rank.clone(), entry.recorded)
        };
        let ranks: Vec<usize> = (0..self.engines.len())
            .filter(|&rank| per_rank[rank].is_some())
            .collect();

        // Bind: each device takes the inputs its own slots name; one no device names is unknown.
        let mut split: Vec<StepInputs<'_>> = ranks.iter().map(|_| StepInputs::new()).collect();
        for (key, value) in inputs.iter() {
            let mut taken = false;
            for (slot, &rank) in ranks.iter().enumerate() {
                let entry = per_rank[rank].expect("a participating rank");
                if self.engines[rank].takes_input(per_rank_exe[rank], entry, key)? {
                    split[slot].push(key.clone(), value.shape, value.view);
                    taken = true;
                }
            }
            if !taken {
                return Err(BindError::Unknown { key: key.clone() }.into());
            }
        }
        for (slot, &rank) in ranks.iter().enumerate() {
            let entry = per_rank[rank].expect("a participating rank");
            self.engines[rank].bind_inputs(per_rank_exe[rank], entry, &split[slot])?;
        }

        if !recorded {
            self.record(exe, entry)?;
        }
        self.replay(exe, entry, &ranks)?;
        self.steps += 1;
        Ok(StepOutputs::new(self, exe, entry, DeviceTime::Unknown))
    }

    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError> {
        self.check_usable()?;
        let per_rank = self.executable(exe)?.per_rank.clone();
        for (engine, exe) in self.engines.iter_mut().zip(per_rank) {
            engine.reset_state(exe, scope.clone())?;
        }
        Ok(())
    }

    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        self.check_usable()?;
        let per_rank_exe = self.executable(exe)?.per_rank.clone();
        let per_rank = self.entry(exe, entry)?.per_rank.clone();
        // A recording on one device may still be writing a buffer another device owns.
        self.quiesce_all()?;
        for (rank, engine) in self.engines.iter_mut().enumerate() {
            if let Some(entry) = per_rank[rank] {
                engine.remove_entry(per_rank_exe[rank], entry)?;
            }
        }
        self.executables[exe.index()]
            .as_mut()
            .expect("checked above")
            .entries[entry.index()] = None;
        Ok(())
    }

    fn stats(&self) -> ExecutorStats {
        let mut stats = ExecutorStats::default();
        let mut memory: HashMap<BufferRole, crate::MemoryCounterSnapshot> = HashMap::new();
        let engine_stats: Vec<ExecutorStats> =
            self.engines.iter().map(|engine| engine.stats()).collect();
        for device in &engine_stats {
            stats.host_write_bytes += device.host_write_bytes;
            stats.weight_bytes_read += device.weight_bytes_read;
            stats.kernels_resolved += device.kernels_resolved;
            stats.recordings += device.recordings;
            for &(role, snapshot) in &device.memory {
                let total = memory.entry(role).or_default();
                total.live_bytes += snapshot.live_bytes;
                total.peak_bytes += snapshot.peak_bytes;
                total.allocations += snapshot.allocations;
            }
        }
        for role in BufferRole::ALL {
            if let Some(snapshot) = memory.remove(&role) {
                stats.memory.push((role, snapshot));
            }
        }
        for (exe_index, executable) in self.executables.iter().enumerate() {
            let Some(executable) = executable else {
                continue;
            };
            for (entry_index, entry) in executable.entries.iter().enumerate() {
                let Some(entry) = entry else { continue };
                stats.replays += entry.replays;
                let mut per_stage: Vec<(StageId, u64, u64)> = Vec::new();
                for (rank, local) in entry.per_rank.iter().enumerate() {
                    let Some(local) = local else { continue };
                    for stage in &engine_stats[rank].stages {
                        if stage.exe == executable.per_rank[rank] && stage.entry == *local {
                            per_stage.push((stage.stage, stage.recordings, stage.replays));
                        }
                    }
                }
                per_stage.sort_by_key(|(stage, _, _)| stage.0);
                for (stage, recordings, replays) in per_stage {
                    stats.stages.push(StageStats {
                        exe: ExecutableId::new(exe_index),
                        entry: EntryId::new(entry_index),
                        stage,
                        recordings,
                        replays,
                    });
                }
            }
        }
        stats
    }

    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        self.check_usable()?;
        let per_rank = self.executable(exe)?.per_rank.clone();
        // Every device is proven finished before any is released: a graph on one device may still be
        // writing a buffer, or waiting on an event, that another device is about to free. A device
        // that cannot prove it keeps the whole executable.
        self.quiesce_all()?;
        for (engine, exe) in self.engines.iter_mut().zip(per_rank) {
            engine.unload(exe)?;
        }
        self.executables[exe.index()] = None;
        Ok(())
    }
}

impl<D: Device + Collectives> Coordinator<D> {
    /// Record every stage of the entry, in stage order, each on its own device with its waits and
    /// sends around the walk; publish all recordings or none.
    fn record(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        let Self {
            executables,
            engines,
            counters,
            devices,
            ..
        } = self;
        let executable = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let per_rank_exe = &executable.per_rank;
        let entry_data = executable
            .entries
            .get_mut(entry.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let graphs_before: Vec<u64> = engines
            .iter()
            .map(|engine| engine.device().native_counters().graph_instantiations)
            .collect();
        let mut scratch: Vec<Vec<D::Recording>> = engines.iter().map(|_| Vec::new()).collect();
        for (global, placed) in entry_data.stages.iter().enumerate() {
            let (this, peers) = split_rank(engines.as_mut_slice(), placed.rank);
            let mut sends = Vec::new();
            for hop in entry_data
                .hops
                .iter()
                .filter(|hop| hop.from_stage == global)
            {
                let consumer = entry_data.per_rank[hop.to_rank]
                    .expect("a transfer's consumer rank holds the entry");
                sends.push(Handoff {
                    row: hop.row,
                    destination: peers.get(hop.to_rank).import_buffer(
                        per_rank_exe[hop.to_rank],
                        consumer,
                        hop.row,
                    )?,
                });
            }
            let mut seams = PeerSeams::<D> { sends };
            let local_entry = entry_data.per_rank[placed.rank].expect("a participating rank");
            // A failure aborted this device exactly once and dropped nothing published: the
            // recordings already made are in `scratch` and go with it.
            let recording = this.record_stage(
                per_rank_exe[placed.rank],
                local_entry,
                placed.local,
                &mut seams,
            )?;
            scratch[placed.rank].push(recording);
        }
        for (rank, recordings) in scratch.into_iter().enumerate() {
            let Some(local_entry) = entry_data.per_rank[rank] else {
                continue;
            };
            let phases = recordings.len() as u64;
            engines[rank].publish_recordings(per_rank_exe[rank], local_entry, recordings)?;
            counters[rank].sessions += 1;
            counters[rank].phases += phases;
        }
        for support in &mut entry_data.support {
            let rank = devices
                .iter()
                .position(|&device| device == support.device)
                .expect("support is reported for the coordinator's own devices");
            support.actual.graph_instantiations = engines[rank]
                .device()
                .native_counters()
                .graph_instantiations
                .saturating_sub(graphs_before[rank]);
        }
        entry_data.recorded = true;
        Ok(())
    }

    /// Submit every stage's recording producer first, then wait once on every device.
    fn replay(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        ranks: &[usize],
    ) -> Result<(), ExecError> {
        let Self {
            executables,
            engines,
            counters,
            ..
        } = self;
        let executable = executables
            .get_mut(exe.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let per_rank_exe = executable.per_rank.clone();
        let entry_data = executable
            .entries
            .get_mut(entry.index())
            .and_then(Option::as_mut)
            .ok_or(LoadError::UnknownHandle)?;
        let per_rank = entry_data.per_rank.clone();
        let local = |rank: usize| per_rank[rank].expect("a participating rank");

        // Whatever fails after the first submission leaves some stages' state committed: every
        // device's executable needs a complete reset, not just the one that failed.
        let poison = |engines: &mut [Engine<D>]| {
            for &rank in ranks {
                engines[rank].require_reset(per_rank_exe[rank]);
            }
        };
        for &rank in ranks {
            if let Err(error) = engines[rank].begin_replays(per_rank_exe[rank]) {
                poison(engines.as_mut_slice());
                return Err(error);
            }
        }
        // Every consumer device marks the point its already-enqueued work (this step's input writes,
        // an import buffer's zeroing) ends at, before any producer copies into it.
        for hop in &entry_data.hops {
            let rank = hop.to_rank;
            if let Err(error) = engines[rank]
                .between_replays(per_rank_exe[rank], |device| device.record_event(&hop.ready))
            {
                poison(engines.as_mut_slice());
                return Err(error);
            }
        }
        for (global, placed) in entry_data.stages.iter().enumerate() {
            let rank = placed.rank;
            let exe = per_rank_exe[rank];
            // Producer first: this stage's device waits for the records of the stages it reads from
            // and for the readiness of the devices it copies into, replays, and then records the
            // events of the stages that read from it.
            let waits: Vec<&D::Event> = entry_data
                .hops
                .iter()
                .filter(|hop| hop.to_stage == global)
                .map(|hop| &hop.event)
                .chain(
                    entry_data
                        .hops
                        .iter()
                        .filter(|hop| hop.from_stage == global)
                        .map(|hop| &hop.ready),
                )
                .collect();
            let ordered = (|| -> Result<(), ExecError> {
                for event in waits {
                    engines[rank].between_replays(exe, |device| device.wait_event(event))?;
                }
                engines[rank].replay_stage(exe, local(rank), placed.local)?;
                for hop in entry_data
                    .hops
                    .iter()
                    .filter(|hop| hop.from_stage == global)
                {
                    engines[rank].between_replays(exe, |device| device.record_event(&hop.event))?;
                }
                Ok(())
            })();
            if let Err(error) = ordered {
                poison(engines.as_mut_slice());
                return Err(error);
            }
            counters[rank].launches += 1;
            counters[rank].dispatches +=
                engines[rank].stage_dispatches(per_rank_exe[rank], local(rank), placed.local)?
                    as u64;
        }
        for hop in &entry_data.hops {
            counters[hop.from_rank].transfers += 1;
        }
        for &rank in ranks {
            engines[rank].count_replay(per_rank_exe[rank], local(rank))?;
        }
        entry_data.replays += 1;

        // The one completion wait of the step, per device, in rank order. Every device is waited
        // for even when one fails, so nothing is left in flight behind the error.
        let mut first: Option<ExecError> = None;
        for &rank in ranks {
            if let Err(error) = engines[rank].complete_entry(per_rank_exe[rank], local(rank)) {
                first.get_or_insert(error);
            }
        }
        let Some(error) = first else { return Ok(()) };
        // A validation failure or a kernel fault leaves every device reusable; anything else may
        // have run some stages' commits.
        if !matches!(error, ExecError::Validation(_) | ExecError::Fault { .. }) {
            poison(engines.as_mut_slice());
        }
        Err(error)
    }
}

impl<D: Device + Collectives> OutputSource for Coordinator<D> {
    fn read_output(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        out: &mut [u8],
    ) -> Result<(), ExecError> {
        let (rank, local) = self.output_of(exe, entry)?;
        let exe = self.executable(exe)?.per_rank[rank];
        self.engines[rank].read_output(exe, local, out)
    }

    fn output_bytes(&self, exe: ExecutableId, entry: EntryId) -> Result<usize, ExecError> {
        let (rank, local) = self.output_of(exe, entry)?;
        self.engines[rank].output_bytes(self.executable(exe)?.per_rank[rank], local)
    }

    fn output_host(
        &self,
        exe: ExecutableId,
        entry: EntryId,
    ) -> Result<(poot_tensor::DType, Vec<usize>), ExecError> {
        let (rank, local) = self.output_of(exe, entry)?;
        self.engines[rank].output_host(self.executable(exe)?.per_rank[rank], local)
    }
}

impl<D: Device + Collectives> Coordinator<D> {
    /// The rank and engine-local entry that hold the program's final output.
    fn output_of(&self, exe: ExecutableId, entry: EntryId) -> Result<(usize, EntryId), ExecError> {
        let entry = self.entry(exe, entry)?;
        let rank = entry.output_rank;
        Ok((
            rank,
            entry.per_rank[rank].expect("the output rank holds the entry"),
        ))
    }
}

/// Live bytes of transfer staging (the role import buffers are charged to) on `device`.
fn staging_bytes<D: Device>(device: &D) -> u64 {
    device
        .memory()
        .into_iter()
        .filter(|(role, _)| *role == BufferRole::Staging)
        .map(|(_, snapshot)| snapshot.live_bytes)
        .sum()
}
