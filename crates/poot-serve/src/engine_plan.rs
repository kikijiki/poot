//! The typed engine plan, computed once at startup (POOT-587, target L12).
//!
//! [`EnginePlan::build`] is the one place the server decides what it will run: it opens the executor the
//! operator's [`BackendChoice`] and [`PlacementSpec`] name (once; a model the resolved backend refuses
//! fails startup, and nothing retries another backend), places the model over its devices, fits the
//! workload to each device's memory by exact planned footprints ([`poot_llm::driver::capacity`]),
//! prepares the fitted shapes, checks the live allocations against the plan and a fresh budget, and only
//! then publishes [`EngineState::Serving`]. The gauges and `/health` read the plan; no handler asks a
//! family or a backend.
//!
//! One device or several is the same pipeline: the placement only changes how the executor is opened,
//! which [`Placement`] the driver is built with, and over how many devices the footprints are kept.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, RwLock};

use poot_executor::{Collectives, Coordinator, Device, Executor, SharedKernelsMismatch};
use poot_graph_plan::multi_device::DeviceSpec;
use poot_graph_plan::{
    CompileLimits, CompileOptions, DeviceId, DevicePlacement, Refusal, ShardRule, Submission,
    TargetSet,
};
use poot_llm::driver::block_table::BLOCK_SIZE;
use poot_llm::driver::capacity::{
    Budget, CapacityRefusal, Fit, FitError, Observation, PlanError, PlanFailure, Planner,
    verify_prepared,
};
use poot_llm::driver::error::{DriverError, InvalidRequest, Unsupported};
use poot_llm::driver::placement::{BalanceError, LayerBytes, Placement, balance_layers};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverCaps, DriverOptions, Head, Layout, ModelHandle,
    PoolShape, PreparedSetLimits, ServingShapes, Warm, program_retention,
};
use poot_models::model::ModelOutput;
use poot_target::Backend as TargetBackend;

use crate::config::PlacementSpec;

/// Bytes kept back on every device for allocations the planner cannot see: recordings, staging and
/// driver state a runtime makes on its own. A bound, not a measurement; the live counters are checked
/// against the budget less this after prepare.
pub(crate) const SUPPORT_RESERVE_BYTES: u64 = 256 << 20;

/// Decode rows the server asks for before the memory fit reduces them.
const DEFAULT_ROWS: usize = 4;
/// Positions per sequence the server asks for before the memory fit reduces them.
const DEFAULT_MAX_SEQ: usize = 4096;
/// The fewest positions per sequence the fit will offer.
const MIN_MAX_SEQ: usize = 128;
/// The prefill chunk, in tokens, rounded down to the model's granule and capped by the capacity.
const PREFILL_CHUNK_TOKENS: usize = 128;
/// Entries the driver may retain, and the host bytes their programs may retain.
const MAX_ENTRIES: usize = 64;
const MAX_RETAINED_BYTES: u64 = 1 << 32;

/// What the operator asked for. Absent fields take the defaults above.
#[derive(Clone, Debug)]
pub(crate) struct EngineArgs {
    pub(crate) backend: BackendChoice,
    /// Which devices run the model, and how.
    pub(crate) placement: PlacementSpec,
    pub(crate) rows: NonZeroUsize,
    pub(crate) max_seq: NonZeroUsize,
    /// How decode rows propose tokens to verify; `Off` serves one token a step.
    pub(crate) draft: DraftPolicy,
    /// Prompts carry embedding rows (an image front end feeds the engine): the `PrefillEmbeds`
    /// entries are planned and prepared beside the token ones.
    pub(crate) embeds: bool,
    /// Pool rows per routed selection when experts are paged ([`Driver::with_expert_pool`]).
    pub(crate) expert_pool: Option<NonZeroUsize>,
}

impl EngineArgs {
    pub(crate) fn new(backend: BackendChoice, draft: DraftPolicy) -> Self {
        Self {
            backend,
            placement: PlacementSpec::Single,
            draft,
            embeds: false,
            expert_pool: None,
            rows: NonZeroUsize::new(DEFAULT_ROWS).expect("nonzero"),
            max_seq: NonZeroUsize::new(DEFAULT_MAX_SEQ).expect("nonzero"),
        }
    }

    pub(crate) fn with_expert_pool(self, expert_pool: Option<NonZeroUsize>) -> Self {
        Self {
            expert_pool,
            ..self
        }
    }

    pub(crate) fn with_placement(self, placement: PlacementSpec) -> Self {
        Self { placement, ..self }
    }

    pub(crate) fn with_embeds(self) -> Self {
        Self {
            embeds: true,
            ..self
        }
    }
}

/// What the server runs the model for. Only generation is planned: an encoder or cross-encoder prepares
/// its entries on first use, so their allocations cannot be accounted before `Serving`, and startup
/// refuses them ([`StartupError::NotAccounted`]) until their routes exist (POOT-589).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Workload {
    /// Generation over a paged pool: the rows and positions the memory fit settled on.
    Generate {
        rows: NonZeroUsize,
        max_seq: NonZeroUsize,
    },
}

impl Workload {
    /// The workload a model's output asks for, or the refusal when its allocations cannot be accounted.
    fn of(
        output: ModelOutput,
        rows: NonZeroUsize,
        max_seq: NonZeroUsize,
    ) -> Result<Self, StartupError> {
        match output {
            ModelOutput::Logits { .. } => Ok(Workload::Generate { rows, max_seq }),
            ModelOutput::Hidden { .. } => Err(StartupError::NotAccounted {
                output: "hidden states (embeddings)",
            }),
            ModelOutput::Scores { .. } => Err(StartupError::NotAccounted {
                output: "classifier scores (rerank)",
            }),
        }
    }
}

/// What the memory fit settled on for one device, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceBytes {
    pub(crate) device: DeviceId,
    /// Bytes the fitted plan allocates on the device.
    pub(crate) planned_bytes: u64,
    /// The device budget (less the reserve) the runtime reported when the device opened.
    pub(crate) discovered_bytes: Option<u64>,
    /// The same budget observed after prepare.
    pub(crate) available_bytes: Option<u64>,
}

/// What the memory fit settled on, in bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BudgetReport {
    /// Each device's own figures, in rank order: a budget is per device.
    pub(crate) devices: Vec<DeviceBytes>,
    /// Bytes of the weight store every planned candidate read together: zero, because planning sizes
    /// weights and the one real read is the prepared driver's own (POOT-1066).
    pub(crate) planning_weight_bytes_read: u64,
    /// Kernels the planned candidates took through the kernel cache together: each distinct kernel once,
    /// however many candidates use it (POOT-1068).
    pub(crate) planning_kernels_resolved: u64,
}

impl BudgetReport {
    /// The budget the devices have together once prepared: the sum of what each reports.
    pub(crate) fn available_bytes(&self) -> Option<u64> {
        self.devices
            .iter()
            .map(|device| device.available_bytes)
            .try_fold(0u64, |total, bytes| Some(total.saturating_add(bytes?)))
    }
}

/// The plan the server runs: computed once, read by the gauges and `/health`.
#[derive(Debug)]
pub(crate) struct EnginePlan {
    pub(crate) backend: TargetBackend,
    /// The devices the model runs on, in rank order.
    pub(crate) devices: Vec<DeviceId>,
    /// How the model is placed over them.
    pub(crate) placement: Placement,
    pub(crate) model: String,
    pub(crate) workload: Workload,
    /// What `prepare` compiled.
    pub(crate) caps: DriverCaps,
    pub(crate) budget: BudgetReport,
}

impl EnginePlan {
    /// The decode rows the engine serves; zero for a workload with no decode slots.
    pub(crate) fn rows(&self) -> usize {
        match self.workload {
            Workload::Generate { rows, .. } => rows.get(),
        }
    }

    /// The label the gauges and `/health` report for the resolved backend.
    pub(crate) fn backend_label(&self) -> String {
        backend_label(self.backend)
    }
}

/// The device budget the executor reports now, with the support reserve kept back on each device.
///
/// Read live at every admission: the figure the device queried when it opened (`DeviceCaps::
/// memory_budget_bytes`) is only a ceiling, because memory other processes took since is not in it.
pub(crate) fn device_budget(observation: &Observation) -> Budget {
    Budget::observed(observation, SUPPORT_RESERVE_BYTES)
}

fn backend_label(backend: TargetBackend) -> String {
    match backend {
        TargetBackend::Nvptx => "nvptx".to_string(),
        TargetBackend::SpirvVulkan => "spirv-vulkan".to_string(),
        TargetBackend::AmdGcn(arch) => format!("amdgcn-{}", arch.mcpu()),
        TargetBackend::Aie(target) => target.canonical(),
    }
}

/// Why the engine thread is gone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Death {
    /// The thread panicked; the payload's message.
    Panicked(String),
    /// The loop stopped itself: the device or the driver's state can no longer be trusted.
    Fault(String),
}

impl Death {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Death::Panicked(_) => "panicked",
            Death::Fault(_) => "fault",
        }
    }

    pub(crate) fn reason(&self) -> &str {
        match self {
            Death::Panicked(reason) | Death::Fault(reason) => reason,
        }
    }
}

/// The lifecycle `/health` reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EngineState {
    Starting,
    Serving {
        backend: String,
        rows: usize,
        max_seq: usize,
    },
    /// Shutdown began: in-flight requests finish, nothing new is admitted.
    Draining,
    /// The engine thread ended and will not serve again.
    Dead(Death),
}

/// The engine's state, shared between startup, which writes it, and the connection threads, which read
/// it.
#[derive(Clone, Debug)]
pub(crate) struct EngineHealth(Arc<RwLock<EngineState>>);

impl EngineHealth {
    pub(crate) fn starting() -> Self {
        Self(Arc::new(RwLock::new(EngineState::Starting)))
    }

    pub(crate) fn state(&self) -> EngineState {
        self.0.read().expect("engine health lock").clone()
    }

    /// `Serving` becomes `Draining`; a `Dead` or `Starting` engine stays as it is.
    pub(crate) fn begin_draining(&self) {
        let mut state = self.0.write().expect("engine health lock");
        if matches!(*state, EngineState::Serving { .. }) {
            *state = EngineState::Draining;
        }
    }

    /// The engine thread ended: whatever the state was, the engine is `Dead`.
    pub(crate) fn die(&self, death: Death) {
        *self.0.write().expect("engine health lock") = EngineState::Dead(death);
    }

    /// Whether an engine thread is accepting or finishing work.
    pub(crate) fn engine_loaded(&self) -> bool {
        matches!(
            *self.0.read().expect("engine health lock"),
            EngineState::Serving { .. } | EngineState::Draining
        )
    }

    /// A health already `Serving`, for the routes that need an engine in the state they run in.
    #[cfg(test)]
    pub(crate) fn serving_for_test(backend: &str, rows: usize, max_seq: usize) -> Self {
        Self(Arc::new(RwLock::new(EngineState::Serving {
            backend: backend.to_string(),
            rows,
            max_seq,
        })))
    }

    /// Publish `Serving` for `plan`. Called after `prepare` and the live accounting, never before.
    fn serving(&self, plan: &EnginePlan) {
        let max_seq = match plan.workload {
            Workload::Generate { max_seq, .. } => max_seq.get(),
        };
        *self.0.write().expect("engine health lock") = EngineState::Serving {
            backend: plan.backend_label(),
            rows: plan.rows(),
            max_seq,
        };
    }
}

/// Why the server cannot start. Every variant exits non-zero before the listener binds.
#[derive(Debug, thiserror::Error)]
pub(crate) enum StartupError {
    #[error("open the {choice:?} device: {source}")]
    Device {
        choice: BackendChoice,
        #[source]
        source: DriverError,
    },
    /// The resolved backend cannot lower an op of the model. The planner's refusal, unflattened: it
    /// names the target, the op and the capability the target lacks.
    #[error("the {} target refuses the model: {refusal:?}", backend_label(refusal.target))]
    Unsupported { refusal: Box<Refusal> },
    /// No candidate of the declared shape domain fits the device memory.
    #[error("the workload does not fit the device memory: {0}")]
    Capacity(#[source] FitError<Candidate, PlanError>),
    /// The prepared engine holds more than its plan, or more than the budget observed now.
    #[error("the prepared engine is not admitted: {0}")]
    Admission(#[source] CapacityRefusal),
    /// No layer pipeline over the devices holds the model's weights.
    #[error("the model has no balanced pipeline over the devices: {0}")]
    Pipeline(#[source] BalanceError),
    /// The model's workload prepares its entries on first use, so its allocations cannot be accounted
    /// before `Serving`. Refused rather than served unaccounted.
    #[error(
        "{output} models are prepared on first use and not accounted: refused until their routes exist"
    )]
    NotAccounted { output: &'static str },
    #[error(transparent)]
    Driver(DriverError),
    /// The planner's shared kernel table was bound to another backend than its planning engine's: a
    /// planner bug, never an operator error.
    #[error(transparent)]
    PlanningKernels(SharedKernelsMismatch),
}

impl From<DriverError> for StartupError {
    fn from(error: DriverError) -> Self {
        match error {
            DriverError::Unsupported(Unsupported::Plan(refusal)) => Self::Unsupported { refusal },
            other => Self::Driver(other),
        }
    }
}

impl From<PlanError> for StartupError {
    fn from(error: PlanError) -> Self {
        match error {
            PlanError::Driver(error) => error.into(),
            PlanError::Footprint(error) => Self::Admission(error.into()),
            PlanError::SharedKernels(error) => Self::PlanningKernels(error),
        }
    }
}

/// One point of the declared search domain: the rows and positions a paged pool is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub(crate) rows: NonZeroUsize,
    pub(crate) max_seq: NonZeroUsize,
}

impl Candidate {
    fn pool(self) -> PoolShape {
        let blocks_per_seq = self.max_seq.get().div_ceil(BLOCK_SIZE);
        PoolShape {
            blocks: NonZeroUsize::new(self.rows.get() * blocks_per_seq).expect("nonzero"),
            max_seqs: self.rows,
        }
    }
}

/// The declared search domain, preferred first: positions halve down to the floor at the requested
/// rows, then the rows halve and positions start over. Every candidate is checked on its own, so the
/// order is a preference and not a monotonicity claim.
fn domain(rows: NonZeroUsize, max_seq: NonZeroUsize) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut r = rows.get();
    loop {
        let mut s = max_seq.get();
        loop {
            candidates.push(Candidate {
                rows: NonZeroUsize::new(r).expect("nonzero"),
                max_seq: NonZeroUsize::new(s).expect("nonzero"),
            });
            if s <= MIN_MAX_SEQ {
                break;
            }
            s = (s / 2).max(MIN_MAX_SEQ);
        }
        if r == 1 {
            return candidates;
        }
        r = (r / 2).max(1);
    }
}

fn driver_options(
    handle: &ModelHandle,
    max_seq: NonZeroUsize,
    draft: DraftPolicy,
) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    // The largest granule multiple within the chunk budget and the capacity, and never below one granule.
    let granule = handle.config().prefill_granule.get();
    let chunk =
        NonZeroUsize::new((PREFILL_CHUNK_TOKENS.min(max_seq.get()) / granule).max(1) * granule)
            .expect("a granule multiple is nonzero");
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: max_seq,
        prefill_chunk: chunk,
        max_trace_tokens: chunk,
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(MAX_ENTRIES).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(MAX_RETAINED_BYTES).expect("nonzero"),
        },
        charge: program_retention,
        draft,
        hidden_window: None,
    }
}

/// The shapes a generation workload serves: every admitted prefill piece and the decode step, with the
/// greedy suffix and the logits head, over the candidate's paged pool. A temperature request, or any
/// mix of rules, picks on the host from the logits head. `rows` is the candidate's own
/// row count, held by the caller so the shapes can borrow it.
fn shapes_of<'a>(candidate: &Candidate, rows: &'a [NonZeroUsize; 1]) -> ServingShapes<'a> {
    ServingShapes {
        layout: Layout::Paged(candidate.pool()),
        rows,
        heads: &[Head::GREEDY, Head::Logits],
        warm: Warm::Admitted,
        windows: &[],
        adapters: &[],
    }
}

/// A started engine: the plan the gauges read and the driver the serving loop will run.
pub(crate) struct Started {
    pub(crate) plan: EnginePlan,
    pub(crate) driver: Driver,
}

/// What an open produced. The variant is the placement the engine runs: a group is opened as the
/// pipeline or the tensor group it was asked to be, and the driver is built from the variant, so an
/// executor and a placement that disagree cannot be written down.
pub(crate) enum Opened {
    /// One device behind a boxed executor.
    One(Box<dyn Executor>),
    /// Several devices under one coordinator, run as a layer pipeline.
    Pipeline(Box<dyn Group>),
    /// Several devices under one coordinator, run as a tensor-parallel group.
    Tensor(Box<dyn Group>),
}

/// A coordinator the driver takes concretely, because a tensor placement loads its weights through the
/// coordinator before it becomes the driver's `dyn Executor`.
pub(crate) trait Group {
    fn executor(&self) -> &dyn Executor;

    fn driver(
        self: Box<Self>,
        handle: Arc<ModelHandle>,
        options: DriverOptions,
        placement: Placement,
    ) -> Result<Driver, DriverError>;
}

/// A coordinator over devices of one kind.
pub(crate) struct Coordinated<D: Device + Collectives + 'static>(pub(crate) Coordinator<D>);

impl<D: Device + Collectives + 'static> Group for Coordinated<D> {
    fn executor(&self) -> &dyn Executor {
        &self.0
    }

    fn driver(
        self: Box<Self>,
        handle: Arc<ModelHandle>,
        options: DriverOptions,
        placement: Placement,
    ) -> Result<Driver, DriverError> {
        Driver::placed(handle, self.0, options, placement)
    }
}

/// An opened executor with the placement it runs, resolved: the one thing the driver is built from.
enum Launch {
    One(Box<dyn Executor>),
    Group(Box<dyn Group>),
}

impl Launch {
    fn executor(&self) -> &dyn Executor {
        match self {
            Launch::One(executor) => &**executor,
            Launch::Group(group) => group.executor(),
        }
    }

    fn driver(
        self,
        handle: Arc<ModelHandle>,
        options: DriverOptions,
        placement: Placement,
    ) -> Result<Driver, DriverError> {
        match self {
            Launch::One(executor) => Driver::new(handle, executor, options),
            Launch::Group(group) => group.driver(handle, options, placement),
        }
    }
}

/// The placement `opened` runs and the launch that builds its driver; a pipeline's cut is balanced over
/// the devices' budgets.
fn place(
    opened: Opened,
    handle: &ModelHandle,
    discovered: &Budget,
) -> Result<(Placement, Launch), StartupError> {
    match opened {
        Opened::One(executor) => Ok((Placement::Single, Launch::One(executor))),
        Opened::Tensor(group) => Ok((
            Placement::Tensor {
                rule: ShardRule::Megatron,
            },
            Launch::Group(group),
        )),
        Opened::Pipeline(group) => {
            let targets = group.executor().target_set();
            let mut specs = Vec::with_capacity(targets.devices().len());
            for &(id, target) in targets.devices() {
                let usable_bytes = discovered.available(id).ok_or(StartupError::Admission(
                    CapacityRefusal::UnknownBudget { device: id },
                ))?;
                specs.push(DeviceSpec {
                    id,
                    backend: target.backend,
                    usable_bytes,
                });
            }
            let bytes = LayerBytes::of(handle);
            match balance_layers(&bytes, &specs).map_err(StartupError::Pipeline)? {
                DevicePlacement::Pipeline { after_layers, .. } => {
                    Ok((Placement::Pipeline { after_layers }, Launch::Group(group)))
                }
                DevicePlacement::Single(_) | DevicePlacement::Tensor { .. } => {
                    unreachable!("a balance is a pipeline")
                }
            }
        }
    }
}

impl EnginePlan {
    /// Open the executor `args` name, place and fit the workload, prepare it, and publish `Serving`.
    ///
    /// `open` is called exactly once: the backend the operator named, or the one `Auto` resolves to in
    /// [`poot_llm::driver::open_executor`], over the devices `args.placement` names. A refusal on that
    /// backend is the startup error; no other backend is tried. `observe` reads the device budgets from
    /// the targets at each admission and again after prepare; [`device_budget`] reads the memory the
    /// targets report. `health` stays `Starting` until the prepared engine has been checked.
    pub(crate) fn build(
        args: &EngineArgs,
        handle: &Arc<ModelHandle>,
        open: impl FnOnce(&EngineArgs) -> Result<Opened, DriverError>,
        mut observe: impl FnMut(&Observation) -> Budget,
        health: &EngineHealth,
    ) -> Result<Started, StartupError> {
        let config = handle.config();
        let max_seq = NonZeroUsize::new(args.max_seq.get().min(config.max_positions))
            .expect("a model has positions");
        // Refused before any device opens: an unaccounted workload never touches the device.
        Workload::of(config.output, args.rows, max_seq)?;
        let opened = open(args).map_err(|source| StartupError::Device {
            choice: args.backend,
            source,
        })?;
        let discovered = observe(&Observation::of(match &opened {
            Opened::One(executor) => &**executor,
            Opened::Pipeline(group) | Opened::Tensor(group) => group.executor(),
        }));
        let (placement, launch) = place(opened, handle, &discovered)?;
        let targets = launch.executor().target_set();
        let devices: Vec<DeviceId> = targets.devices().iter().map(|&(id, _)| id).collect();
        let backend = targets.devices()[0].1.backend;
        let Fitted {
            candidate,
            driver,
            planned,
            planning_weight_bytes_read,
            planning_kernels_resolved,
        } = fit_and_prepare(
            args,
            handle,
            launch,
            &targets,
            &placement,
            max_seq,
            &mut observe,
        )?;
        // `prepare` published them: a driver without caps is a bug in the preparation, reported as the
        // driver's own typed refusal.
        let caps = driver
            .caps()
            .cloned()
            .ok_or_else(|| DriverError::from(InvalidRequest::NotPrepared))?;
        let observed = observe(&driver.observation());
        let workload = Workload::Generate {
            rows: candidate.rows,
            max_seq: candidate.max_seq,
        };
        let budget = BudgetReport {
            devices: planned
                .into_iter()
                .map(|(device, planned_bytes)| DeviceBytes {
                    device,
                    planned_bytes,
                    discovered_bytes: discovered.available(device),
                    available_bytes: observed.available(device),
                })
                .collect(),
            planning_weight_bytes_read,
            planning_kernels_resolved,
        };
        let plan = EnginePlan {
            backend,
            devices,
            placement,
            model: config.family.to_string(),
            workload,
            caps,
            budget,
        };
        health.serving(&plan);
        Ok(Started { plan, driver })
    }
}

/// The candidate the fit chose, prepared on the real executor.
struct Fitted {
    candidate: Candidate,
    driver: Driver,
    /// The bytes the chosen plan allocates on each device, in rank order.
    planned: Vec<(DeviceId, u64)>,
    planning_weight_bytes_read: u64,
    planning_kernels_resolved: u64,
}

/// Fit the generation workload to the devices, then prepare the chosen shapes on the real executor and
/// check what it allocated.
fn fit_and_prepare(
    args: &EngineArgs,
    handle: &Arc<ModelHandle>,
    launch: Launch,
    targets: &TargetSet,
    placement: &Placement,
    max_seq: NonZeroUsize,
    observe: &mut impl FnMut(&Observation) -> Budget,
) -> Result<Fitted, StartupError> {
    let mut planner = Planner::new(Arc::clone(handle), targets, placement.clone())?;
    if let Some(slots) = args.expert_pool {
        planner = planner.with_expert_pool(slots);
    }
    let mut planning_weight_bytes_read = 0u64;
    let mut planning_kernels_resolved = 0u64;
    let fit = poot_llm::driver::capacity::fit(
        &domain(args.rows, max_seq),
        |candidate| {
            let rows = [candidate.rows];
            planner
                .plan(
                    driver_options(handle, candidate.max_seq, args.draft.clone()),
                    &shapes_of(candidate, &rows),
                    args.embeds,
                )
                .inspect(|planned| {
                    planning_weight_bytes_read += planned.weight_bytes_read;
                    planning_kernels_resolved += planned.kernels_resolved;
                })
                .map_err(PlanFailure::Fatal)
        },
        // One observation per admission, never the budget read once before the search.
        || observe(&Observation::of(launch.executor())),
    )
    .map_err(|error| match error {
        FitError::Plan(error) => error.into(),
        other => StartupError::Capacity(other),
    })?;
    let Fit {
        chosen,
        footprint,
        ledger,
        ..
    } = fit;
    let devices: Vec<DeviceId> = targets.devices().iter().map(|&(id, _)| id).collect();
    let planned = devices
        .iter()
        .map(|&device| {
            footprint
                .device_bytes(device)
                .map(|bytes| (device, bytes))
                .map_err(|error| StartupError::Admission(error.into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut driver = launch.driver(
        Arc::clone(handle),
        driver_options(handle, chosen.max_seq, args.draft.clone()),
        placement.clone(),
    )?;
    if let Some(slots) = args.expert_pool {
        driver = driver.with_expert_pool(slots)?;
    }
    let rows = [chosen.rows];
    driver.prepare(&shapes_of(&chosen, &rows))?;
    if args.embeds {
        driver.prepare_embeds(&shapes_of(&chosen, &rows))?;
    }
    verify_prepared(
        &footprint,
        &driver.live_device_bytes(),
        &observe(&driver.observation()),
        &ledger,
    )
    .map_err(StartupError::Admission)?;
    Ok(Fitted {
        candidate: chosen,
        driver,
        planned,
        planning_weight_bytes_read,
        planning_kernels_resolved,
    })
}

#[cfg(test)]
mod tests;
