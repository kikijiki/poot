//! The engine plan over a real model and the real planner, with only the device faked. The model is
//! the tiny deepseek2 GGUF the other rows load; the executor is `Engine` over a counting device whose
//! target, budget and allocation behaviour a row scripts. Rows read the plan, the typed errors and the
//! counters, never a token.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use poot_executor::{
    Collectives, Coordinator, Device, DeviceTime, Dispatch, Engine, EntryId, ExecError,
    ExecutableId, Executor, ExecutorStats, HostSync, NativeCounters, StateScope, StepInputs,
    StepOutputs, SupportBound,
};
use poot_graph_plan::{Capability, Target, staged::StagedProgram, staged::TargetSet};
use poot_llm::driver::{BackendChoice, DraftPolicy, ModelHandle};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::{BufferRole, CompiledKernel, MemoryCounterSnapshot, MemoryCounters};
use poot_target::{AmdArch, Backend, DeviceCaps, TargetId};

use super::*;
use clap::Parser;

use crate::admin::LoraAdminPolicy;
use crate::config::ServeConfig;
use crate::http::handle;
use crate::tests::{load_handle, test_metrics, write_tiny_gguf};
use poot_graph_plan::test_support::gpu_caps_mut;
use poot_llm::driver::capacity::{CapacityRefusal, FitError, Observation, Rejection};

/// The production budget reader (`device_budget`) over a group whose devices hold exactly
/// `capacity[rank]` bytes: the synthetic per-device budget of the admission rows below.
fn build_group_with_capacity(
    args: &EngineArgs,
    handle: &Arc<ModelHandle>,
    loaded: &Arc<Mutex<HashSet<u64>>>,
    inflate: [u64; 2],
    capacity: [u64; 2],
    health: &EngineHealth,
) -> Result<Started, StartupError> {
    build_group(
        args,
        handle,
        group_of(loaded, inflate, capacity.map(Some)),
        device_budget,
        health,
    )
}

/// The two placements the admission rows run, each with the checkpoint it can divide and the bytes its
/// smallest candidate needs on each rank, from the planner on its own.
fn placed_cases() -> Vec<(&'static str, EngineArgs, Arc<ModelHandle>, [u64; 2])> {
    let pipeline_handle = fixture_model();
    let pipeline_args = group_args(pipeline_over_two());
    let pipeline_needs = smallest_candidate_bytes(
        &pipeline_handle,
        &pipeline_args,
        Placement::Pipeline {
            after_layers: vec![0],
        },
    );
    let tensor_handle = dense_model();
    let tensor_args = group_args(tensor_over_two());
    let tensor_needs = smallest_candidate_bytes(
        &tensor_handle,
        &tensor_args,
        Placement::Tensor {
            rule: ShardRule::Megatron,
        },
    );
    vec![
        ("pipeline", pipeline_args, pipeline_handle, pipeline_needs),
        ("tensor", tensor_args, tensor_handle, tensor_needs),
    ]
}

/// A group whose devices hold exactly the smallest candidate's bytes (and the production reserve) is
/// admitted, per placement, through the production budget reader: the smallest candidate is chosen after
/// every larger one is refused, each rank's published plan equals its own planned bytes, and the live
/// counters after prepare sit at the plan on each rank.
///
/// Mutation: read every device's budget as the sum of the devices' free memory (`device_budget`), or
/// observe only the first device (`fit_and_prepare`'s `Observation::of`): the exact figures no longer
/// hold per rank, the group is refused or the fit chooses a larger candidate, and this row goes red.
#[test]
fn an_exact_per_device_capacity_admits_each_placement_at_its_smallest_candidate() {
    for (label, args, handle, needs) in placed_cases() {
        let domain = candidates(&handle, &args);
        let smallest = *domain.last().expect("a domain");
        assert!(
            needs.iter().all(|&bytes| bytes > 0),
            "{label}: each rank owns bytes: {needs:?}"
        );
        let health = EngineHealth::starting();
        let loaded = Arc::new(Mutex::new(HashSet::new()));
        let capacity = needs.map(|bytes| bytes + SUPPORT_RESERVE_BYTES);
        let started = build_group_with_capacity(&args, &handle, &loaded, [0, 0], capacity, &health)
            .unwrap_or_else(|error| panic!("{label}: the exact capacity is refused: {error}"));
        let Workload::Generate { rows, max_seq } = started.plan.workload else {
            panic!("a generative model plans generation");
        };
        assert_eq!(
            (rows, max_seq),
            (smallest.rows, smallest.max_seq),
            "{label}: every larger candidate was refused and the smallest fits"
        );
        for (index, device) in started.plan.budget.devices.iter().enumerate() {
            assert_eq!(device.planned_bytes, needs[index], "{label}: {device:?}");
            assert_eq!(
                device.available_bytes,
                Some(needs[index]),
                "{label}: the budget after prepare is the exact capacity less the reserve"
            );
        }
        assert!(matches!(health.state(), EngineState::Serving { .. }));
        // The prepared counters sit at the plan on each rank: nothing was allocated beyond it.
        let live = started.driver.live_device_bytes();
        for (index, (device, roles)) in live.iter().enumerate() {
            let held: u64 = roles.iter().map(|&(_, bytes)| bytes).sum();
            assert!(
                held <= needs[index],
                "{label}: {device:?} holds {held} live bytes over its plan {}",
                needs[index]
            );
        }
    }
}

/// A group whose rank is one byte short of its own bytes is refused before `Serving`, per placement and
/// whichever rank it is, with the refusal naming that rank: the other rank has room for the whole
/// workload, so the summed capacities would admit it. Nothing is prepared on the devices.
///
/// Mutation: as above; the summed budget admits the plan and the row goes red.
#[test]
fn a_rank_one_byte_short_of_its_own_bytes_is_refused_before_serving_naming_it() {
    for (label, args, handle, needs) in placed_cases() {
        for short in [0usize, 1] {
            let health = EngineHealth::starting();
            let loaded = Arc::new(Mutex::new(HashSet::new()));
            let mut capacity = [u64::MAX / 4; 2];
            capacity[short] = needs[short] + SUPPORT_RESERVE_BYTES - 1;
            let result =
                build_group_with_capacity(&args, &handle, &loaded, [0, 0], capacity, &health);
            let Err(StartupError::Capacity(FitError::NoCandidateFits { rejected })) = result else {
                panic!(
                    "{label} rank {short}: expected every candidate refused, got {:?}",
                    result.err()
                );
            };
            assert_eq!(rejected.len(), candidates(&handle, &args).len());
            for rejection in &rejected {
                match &rejection.why {
                    Rejection::Capacity(CapacityRefusal::OverBudget {
                        device, available, ..
                    }) => {
                        assert_eq!(*device, DeviceId(short as u32), "{label}: {rejection:?}");
                        assert_eq!(*available, needs[short] - 1, "{label}: {rejection:?}");
                    }
                    other => panic!("{label}: expected an over-budget refusal, got {other:?}"),
                }
            }
            assert!(matches!(health.state(), EngineState::Starting));
            assert!(
                loaded.lock().unwrap().is_empty(),
                "{label} rank {short}: nothing was prepared on the devices"
            );
        }
    }
}

/// A tensor rank whose live state is over its own plan is refused before `Serving` and named, whichever
/// rank it is: the other rank sitting at its plan does not hide it.
///
/// Mutation: drop the `verify_prepared` call after prepare (`fit_and_prepare`): the group starts over a
/// rank holding more than it planned and this row goes red.
#[test]
fn a_tensor_rank_whose_live_state_exceeds_its_plan_is_refused_and_named() {
    let handle = dense_model();
    for (inflate, rank) in [([1 << 20, 0], DeviceId(0)), ([0, 1 << 20], DeviceId(1))] {
        let health = EngineHealth::starting();
        let loaded = Arc::new(Mutex::new(HashSet::new()));
        let result = build_group(
            &group_args(tensor_over_two()),
            &handle,
            group_inflating(&loaded, inflate),
            |_| known_pair(u64::MAX / 2, u64::MAX / 2),
            &health,
        );
        let Err(StartupError::Admission(CapacityRefusal::Underestimated { device, role, .. })) =
            result
        else {
            panic!("expected the underestimate refusal, got {:?}", result.err());
        };
        assert_eq!(
            (device, role),
            (rank, BufferRole::State),
            "inflate {inflate:?}"
        );
        assert!(matches!(health.state(), EngineState::Starting));
    }
}

#[derive(Debug, thiserror::Error)]
#[error("fixture device error")]
struct FixtureError;

/// A device that counts what the engine allocates on it, like a runtime's own counters, and keeps every
/// buffer live for its life.
struct FixtureDevice {
    target: Target,
    memory: MemoryCounters,
    /// Extra bytes charged to the `State` role per allocation, for a runtime that holds more than the
    /// plan counted.
    inflate_state: u64,
    /// When set, the identity (a hash of the whole handle: entry point, code, argument schema) of every
    /// kernel the engine loads - an observation of the kernels made independently of any kernel cache key.
    loaded_kernels: Option<Arc<Mutex<HashSet<u64>>>>,
    /// What the runtime reports as free now, independent of the target's open-time figure.
    free_bytes: Option<u64>,
    /// When set, the device holds this many bytes in all and reports the rest as free, so its free
    /// figure falls as the engine allocates, as a real runtime's does.
    capacity_bytes: Option<u64>,
}

impl Device for FixtureDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FixtureError;

    fn target(&self) -> Target {
        self.target
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }
    fn free_memory_bytes(&self) -> Option<u64> {
        match self.capacity_bytes {
            Some(capacity) => Some(capacity.saturating_sub(self.memory.total_live_bytes())),
            None => self.free_bytes,
        }
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<(), FixtureError> {
        let mut bytes = (elems * storage.element().byte_width()) as u64;
        if role == BufferRole::State {
            bytes += self.inflate_state;
        }
        std::mem::forget(self.memory.record_alloc(role, bytes));
        Ok(())
    }
    /// This fake keeps no storage, so it states no bound: the engine's length check passes every
    /// dispatch it records.
    fn buffer_elems(&self, _buffer: &()) -> usize {
        usize::MAX
    }

    fn load_kernel(&mut self, _key: &str, kernel: CompiledKernel) -> Result<(), FixtureError> {
        if let Some(loaded) = &self.loaded_kernels {
            let mut hasher = std::hash::DefaultHasher::new();
            std::hash::Hash::hash(&format!("{kernel:?}"), &mut hasher);
            loaded
                .lock()
                .expect("loaded kernels")
                .insert(std::hash::Hasher::finish(&hasher));
        }
        Ok(())
    }
    fn begin(&mut self) -> Result<(), FixtureError> {
        Ok(())
    }
    fn dispatch(&mut self, _dispatch: Dispatch<'_, Self>) -> Result<(), FixtureError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FixtureError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Option<()>, FixtureError> {
        Ok(Some(()))
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FixtureError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FixtureError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), _out: &mut [u8]) -> Result<(), FixtureError> {
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), FixtureError> {
        Ok(())
    }
    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), FixtureError> {
        Ok(())
    }
}

/// The executor the plan is handed: an `Engine<FixtureDevice>` that counts the entries added and can hold
/// `add_entry` at a gate.
struct Probe {
    inner: Engine<FixtureDevice>,
    /// Entries the real executor was asked to add: nonzero means something was published.
    add_entries: Arc<AtomicUsize>,
    /// `(entered, release)`: `add_entry` announces itself on `entered`, then waits for `release`.
    gate: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
}

impl Executor for Probe {
    fn target_set(&self) -> TargetSet {
        self.inner.target_set()
    }
    fn free_memory_bytes(&self) -> Option<u64> {
        self.inner.free_memory_bytes()
    }
    fn load_weights(
        &mut self,
        store: Arc<WeightStore>,
        weights: Arc<WeightMap>,
    ) -> Result<ExecutableId, ExecError> {
        self.inner.load_weights(store, weights)
    }
    fn add_entry(
        &mut self,
        exe: ExecutableId,
        program: &StagedProgram,
    ) -> Result<EntryId, ExecError> {
        self.add_entries.fetch_add(1, Ordering::SeqCst);
        if let Some((entered, release)) = &self.gate {
            entered.send(()).expect("announce the prepare");
            // A dropped sender releases this entry and every later one.
            let _ = release.recv();
        }
        self.inner.add_entry(exe, program)
    }
    fn step(
        &mut self,
        exe: ExecutableId,
        entry: EntryId,
        inputs: &StepInputs<'_>,
        sync: &mut dyn HostSync,
    ) -> Result<StepOutputs<'_>, ExecError> {
        self.inner.step(exe, entry, inputs, sync)
    }
    fn reset_state(&mut self, exe: ExecutableId, scope: StateScope) -> Result<(), ExecError> {
        self.inner.reset_state(exe, scope)
    }
    fn remove_entry(&mut self, exe: ExecutableId, entry: EntryId) -> Result<(), ExecError> {
        self.inner.remove_entry(exe, entry)
    }
    fn stats(&self) -> ExecutorStats {
        self.inner.stats()
    }
    fn unload(&mut self, exe: ExecutableId) -> Result<(), ExecError> {
        self.inner.unload(exe)
    }
}

fn spirv_target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

fn probe(target: Target, inflate_state: u64) -> (Probe, Arc<AtomicUsize>) {
    probe_reporting(target, inflate_state, Some(u64::MAX / 2))
}

/// A probe whose runtime reports `free_bytes` as free device memory.
fn probe_reporting(
    target: Target,
    inflate_state: u64,
    free_bytes: Option<u64>,
) -> (Probe, Arc<AtomicUsize>) {
    let add_entries = Arc::new(AtomicUsize::new(0));
    let executor = Probe {
        inner: Engine::new(FixtureDevice {
            target,
            memory: MemoryCounters::new(),
            inflate_state,
            loaded_kernels: None,
            free_bytes,
            capacity_bytes: None,
        }),
        add_entries: Arc::clone(&add_entries),
        gate: None,
    };
    (executor, add_entries)
}

/// The generation workload at `candidate`'s shapes.
fn generate(candidate: impl std::borrow::Borrow<Candidate>) -> Workload {
    let candidate = candidate.borrow();
    Workload::Generate {
        rows: candidate.rows,
        max_seq: candidate.max_seq,
    }
}

fn fixture_model() -> Arc<ModelHandle> {
    let path = write_tiny_gguf();
    let handle = load_handle(&path);
    let _ = std::fs::remove_file(&path);
    handle
}

/// Requests small enough for the one-model fixture, so a row's domain is a handful of candidates.
fn args(backend: BackendChoice) -> EngineArgs {
    EngineArgs {
        backend,
        placement: PlacementSpec::Single,
        rows: NonZeroUsize::new(4),
        max_seq: NonZeroUsize::new(64).unwrap(),
        draft: DraftPolicy::Off,
        embeds: false,
        expert_pool: None,
    }
}

/// A budget naming the one device `Known(bytes)` with the production reserve.
fn known(bytes: u64) -> Budget {
    Budget::new().with(
        DeviceId(0),
        poot_llm::driver::capacity::DeviceBudget::Known {
            bytes,
            reserve: SUPPORT_RESERVE_BYTES,
        },
    )
}

/// What an independent real preparation of `candidate` leaves live on a counting device, summed over
/// roles: the byte count a budget row sets its limit from, not read from the planner.
fn live_total(handle: &Arc<ModelHandle>, candidate: &Candidate) -> u64 {
    let (executor, _) = probe(spirv_target(), 0);
    let driver = Driver::new(
        Arc::clone(handle),
        Box::new(executor),
        driver_options(handle, generate(candidate), DraftPolicy::Off),
    )
    .expect("driver");
    let rows = [candidate.rows];
    let driver = driver
        .prepare(&shapes_of(candidate, &rows))
        .expect("the candidate prepares");
    driver
        .live_role_bytes()
        .iter()
        .map(|&(_, bytes)| bytes)
        .sum()
}

fn candidates(handle: &Arc<ModelHandle>, args: &EngineArgs) -> Vec<Candidate> {
    let max_seq = NonZeroUsize::new(args.max_seq.get().min(handle.config().max_positions)).unwrap();
    domain(args.rows.unwrap(), max_seq)
}

/// Build the plan over `executor`, recording which backend choices `open` was called with.
fn build_with(
    args: &EngineArgs,
    handle: &Arc<ModelHandle>,
    executor: Probe,
    observe: impl FnMut(&Observation) -> Budget,
    health: &EngineHealth,
) -> (Result<Started, StartupError>, Vec<BackendChoice>) {
    let mut opened = Vec::new();
    let mut executor = Some(executor);
    let started = EnginePlan::build(
        args,
        handle,
        |args| {
            opened.push(args.backend);
            Ok(Opened::One(Box::new(executor.take().expect("opened once"))))
        },
        observe,
        health,
    );
    (started, opened)
}

fn roomy(_: &Observation) -> Budget {
    known(u64::MAX / 2)
}

/// A request through the production `http::handle` for `GET /health`: status line and JSON body.
fn get_health(
    health: &EngineHealth,
    handle_model: &Arc<ModelHandle>,
) -> (String, serde_json::Value) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(20)))
        .expect("bound the read");
    let (server, _) = listener.accept().expect("accept");
    client
        .write_all(b"GET /health HTTP/1.1\r\n\r\n")
        .expect("write request");
    let (served, _queue) = crate::tests::served_over(Arc::clone(handle_model), 1);
    let metrics = test_metrics("health");
    let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
        .expect("loopback policy");
    handle(
        server,
        std::time::Instant::now(),
        &served,
        health,
        &metrics,
        &crate::tests::test_limits(),
        &policy,
    )
    .expect("handled");
    let mut response = String::new();
    client.read_to_string(&mut response).expect("read response");
    let status = response.lines().next().unwrap_or_default().to_string();
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    (status, serde_json::from_str(body).expect("json body"))
}

/// SC-001: `/health` is 503 `starting` while `prepare` runs - held at a gate in the executor - and 200
/// `serving` with the plan's backend and rows after. Mutation: answer the constant 200 `ok`; the 503
/// assertion goes red.
#[test]
fn health_is_503_starting_while_prepare_runs_and_200_serving_after() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let builder = {
        let (handle, health) = (Arc::clone(&handle), health.clone());
        std::thread::spawn(move || {
            let (mut executor, _) = probe(spirv_target(), 0);
            executor.gate = Some((entered_tx, release_rx));
            let (started, _) = build_with(
                &args(BackendChoice::Wgpu),
                &handle,
                executor,
                roomy,
                &health,
            );
            let started = started.expect("the plan builds");
            (started.plan.rows(), started.plan.backend_label())
        })
    };
    entered_rx
        .recv_timeout(Duration::from_secs(60))
        .expect("prepare reached the real executor");
    let (status, body) = get_health(&health, &handle);
    assert_eq!(status, "HTTP/1.1 503 Service Unavailable", "{body}");
    assert_eq!(body["status"], "starting", "{body}");
    // Release the gate for every entry the prepare adds.
    drop(release_tx);
    let (rows, label) = builder.join().expect("build thread");
    let (status, body) = get_health(&health, &handle);
    assert_eq!(status, "HTTP/1.1 200 OK", "{body}");
    assert_eq!(body["status"], "serving", "{body}");
    assert_eq!(body["backend"], label, "{body}");
    assert_eq!(body["rows"], rows, "{body}");
    assert!(rows > 0, "a generation plan serves at least one row");
}

/// A target for a ROCm device whose workgroup holds one invocation, below what the model's smallest
/// kernel needs: the planner refuses it on that target, naming the op and the capability.
fn amd_target_missing_workgroup_invocations() -> Target {
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps {
            max_workgroup_invocations: 1,
            ..DeviceCaps::rocm_default()
        },
    }
}

/// SC-002: `--backend rocm` against a device missing a capability the model needs is
/// `StartupError::Unsupported` carrying the planner's typed `Refusal`: the AmdGcn target, the op and the
/// missing capability. The executor was opened for ROCm, nothing was published, and `health` never left
/// `Starting`. Mutations: resolve `Auto` instead of the flag (the recorded choice differs); flatten the
/// refusal to a string (the variant differs).
#[test]
fn a_rocm_backend_missing_a_capability_fails_with_the_typed_refusal() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let (executor, add_entries) = probe(amd_target_missing_workgroup_invocations(), 0);
    let (result, opened) = build_with(
        &args(BackendChoice::Rocm),
        &handle,
        executor,
        roomy,
        &health,
    );
    assert_eq!(opened, [BackendChoice::Rocm], "the flag decides, not Auto");
    let Err(StartupError::Unsupported { refusal }) = result else {
        panic!("expected the typed refusal, got {:?}", result.err());
    };
    assert!(
        matches!(refusal.target, TargetId::Gpu(Backend::AmdGcn(_))),
        "the refusal names the ROCm target: {refusal:?}"
    );
    assert!(
        matches!(refusal.missing, Capability::KernelResources(_)),
        "and the capability the device lacks: {refusal:?}"
    );
    let _op = &refusal.op;
    assert_eq!(
        add_entries.load(Ordering::SeqCst),
        0,
        "nothing was published"
    );
    assert_eq!(health.state(), EngineState::Starting);
}

/// SC-003: `Auto` is resolved once by `open`, and a model the resolved backend refuses fails startup:
/// the next backend is never tried. Mutation: call `open` again after a refusal; the recorded choices
/// have a second entry.
#[test]
fn a_refusal_on_the_resolved_backend_is_not_retried_elsewhere() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let (executor, _) = probe(amd_target_missing_workgroup_invocations(), 0);
    let (result, opened) = build_with(
        &args(BackendChoice::Auto),
        &handle,
        executor,
        roomy,
        &health,
    );
    assert!(
        matches!(result, Err(StartupError::Unsupported { .. })),
        "{:?}",
        result.err()
    );
    assert_eq!(opened, [BackendChoice::Auto], "one open, no retry");
}

/// SC-004 through the production search: the budget of the k-th candidate's exact live total admits it,
/// one byte less admits the next candidate of the declared domain instead. The limits come from an
/// independent counting device. Mutation: admit by a global figure or an extrapolation instead of the
/// composed footprint and these two limits no longer separate the candidates.
#[test]
fn the_search_admits_the_largest_fitting_candidate_and_refuses_the_next() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let domain = candidates(&handle, &args);
    assert_eq!(
        domain.len(),
        3,
        "the fixture's 64 positions leave only the rows ladder: {domain:?}"
    );
    let totals: Vec<u64> = domain.iter().map(|c| live_total(&handle, c)).collect();
    let k = 1;
    assert!(totals[k] > totals[k + 1], "the domain shrinks: {totals:?}");
    for (limit, expected) in [(totals[k], k), (totals[k] - 1, k + 1)] {
        let (executor, _) = probe(spirv_target(), 0);
        let health = EngineHealth::starting();
        let budget = known(SUPPORT_RESERVE_BYTES + limit);
        let (started, _) = build_with(&args, &handle, executor, |_| budget.clone(), &health);
        let started = started.unwrap_or_else(|error| panic!("limit {limit}: {error}"));
        let Workload::Generate { rows, max_seq } = started.plan.workload else {
            panic!("a generative model plans generation");
        };
        assert_eq!(
            (rows, max_seq),
            (domain[expected].rows, domain[expected].max_seq),
            "limit {limit} admits candidate {expected}"
        );
    }
}

/// POOT-1066: sizing weights instead of loading them changes no number. For every candidate of the domain the
/// planned bytes per role equal what an independent real preparation leaves live (the figures the
/// per-candidate full-load pass produced), and the planner read none of the checkpoint where the real
/// preparation read it.
///
/// Mutation: size a dense weight as `numel` bytes instead of its lane's device elements in
/// `Engine::weight_buffer`; the weight role differs from the live one and this row goes red.
#[test]
fn every_candidates_planned_bytes_equal_a_real_preparation_without_reading_weights() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let planner = Planner::new(
        Arc::clone(&handle),
        &probe(spirv_target(), 0).0.target_set(),
        Placement::Single,
    )
    .unwrap();
    let non_staging = |roles: Vec<(BufferRole, u64)>| -> Vec<(BufferRole, u64)> {
        roles
            .into_iter()
            .filter(|&(role, _)| role != BufferRole::Staging)
            .collect()
    };
    for candidate in candidates(&handle, &args) {
        let rows = [candidate.rows];
        let planned = planner
            .plan(
                driver_options(&handle, generate(candidate), DraftPolicy::Off),
                &shapes_of(&candidate, &rows),
                &[],
            )
            .expect("plans");
        let (executor, _) = probe(spirv_target(), 0);
        let driver = Driver::new(
            Arc::clone(&handle),
            Box::new(executor),
            driver_options(&handle, generate(candidate), DraftPolicy::Off),
        )
        .unwrap();
        let driver = driver.prepare(&shapes_of(&candidate, &rows)).unwrap();
        assert_eq!(
            planned.weight_bytes_read, 0,
            "{candidate:?}: planning reads no weights"
        );
        assert!(
            driver.executor_stats().weight_bytes_read > 0,
            "{candidate:?}: a real preparation reads them"
        );
        let mut planned_roles = non_staging(planned.footprint.role_bytes(DeviceId(0)).unwrap());
        let mut live_roles = non_staging(driver.live_role_bytes());
        planned_roles.sort_by_key(|&(role, _)| format!("{role:?}"));
        live_roles.sort_by_key(|&(role, _)| format!("{role:?}"));
        assert_eq!(
            planned_roles, live_roles,
            "{candidate:?}: planned vs live role bytes"
        );
    }
}

/// POOT-1066: the planned footprint of the fixture's whole domain, frozen as literals recorded from the
/// per-candidate full-load planning pass at 79bbd35479 (before weights were sized instead of loaded), so the
/// equality holds against the old numbers and not only against the code that replaced them. Activation moved
/// with the plan, not with the weights: POOT-725 added the I32-to-F32 cast outputs (836616 -> 862664 at four
/// rows) and POOT-1108 folds those casts into the regions around them, so their output buffers (and the
/// intermediates of the chains that now end in the cast) are gone: 827336, below the pre-725 836616.
/// Input fell with POOT-1148: the paged write map is `[rows * tokens]` words instead of one word per pool
/// slot (55224 -> 44952 at four rows). `Meta` moved with POOT-1146: the greedy pick is two dispatches
/// (chunk and combine) with their own metadata words where it was one, 868 -> 980 and 1220 -> 1332 bytes.
///
/// Mutation: size a dense weight from `shape[..1]` in `Engine::weight_buffer`'s size-only path; `Weight`
/// is no longer 5344 and this row goes red.
#[test]
fn the_planned_footprints_match_the_numbers_the_full_load_pass_produced() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let planner = Planner::new(
        Arc::clone(&handle),
        &probe(spirv_target(), 0).0.target_set(),
        Placement::Single,
    )
    .unwrap();
    let golden = [
        (
            4,
            64,
            "Activation=827336 Input=44952 Meta=980 State=12288 Weight=5344",
        ),
        (
            2,
            64,
            "Activation=413752 Input=31688 Meta=980 State=6144 Weight=5344",
        ),
        (
            1,
            64,
            "Activation=206960 Input=25056 Meta=1332 State=3072 Weight=5344",
        ),
    ];
    let domain = candidates(&handle, &args);
    assert_eq!(domain.len(), golden.len());
    for (candidate, (rows, max_seq, want)) in domain.iter().zip(golden) {
        assert_eq!(
            (candidate.rows.get(), candidate.max_seq.get()),
            (rows, max_seq)
        );
        let r = [candidate.rows];
        let planned = planner
            .plan(
                driver_options(&handle, generate(candidate), DraftPolicy::Off),
                &shapes_of(candidate, &r),
                &[],
            )
            .unwrap();
        let mut roles: Vec<String> = planned
            .footprint
            .role_bytes(DeviceId(0))
            .unwrap()
            .iter()
            .map(|(role, bytes)| format!("{role:?}={bytes}"))
            .collect();
        roles.sort();
        assert_eq!(roles.join(" "), want, "rows {rows}, max_seq {max_seq}");
    }
}

/// POOT-1066: startup reads the checkpoint once however many candidates it plans. The budget admits only the
/// last candidate of the domain, so the search plans every earlier one and refuses it; planning reads no
/// weight bytes, and the published driver's reads equal one independent real preparation's.
///
/// Mutation: make `PlanningDevice::weight_upload` return `WeightUpload::Bytes` (reload per candidate); the
/// plan reports the planning reads and this row goes red.
#[test]
fn startup_reads_the_weights_once_however_many_candidates_it_plans() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let domain = candidates(&handle, &args);
    let last = domain.len() - 1;
    assert!(
        last >= 2,
        "the search must plan several candidates: {domain:?}"
    );
    let reference_reads = {
        let (executor, _) = probe(spirv_target(), 0);
        let chosen = domain[last];
        let driver = Driver::new(
            Arc::clone(&handle),
            Box::new(executor),
            driver_options(&handle, generate(chosen), DraftPolicy::Off),
        )
        .expect("driver");
        let rows = [chosen.rows];
        let driver = driver
            .prepare(&shapes_of(&chosen, &rows))
            .expect("the candidate prepares");
        driver.executor_stats().weight_bytes_read
    };
    assert!(
        reference_reads > 0,
        "a real preparation reads the checkpoint"
    );
    let (executor, _) = probe(spirv_target(), 0);
    let budget = known(SUPPORT_RESERVE_BYTES + live_total(&handle, &domain[last]));
    let health = EngineHealth::starting();
    let (started, _) = build_with(&args, &handle, executor, |_| budget.clone(), &health);
    let started = started.expect("the last candidate fits");
    let Workload::Generate { rows, max_seq } = started.plan.workload else {
        panic!("a generative model plans generation");
    };
    assert_eq!(
        (rows, max_seq),
        (domain[last].rows, domain[last].max_seq),
        "every earlier candidate was planned and refused"
    );
    assert_eq!(started.plan.budget.planning_weight_bytes_read, 0);
    assert_eq!(
        started.driver.executor_stats().weight_bytes_read,
        reference_reads,
        "the published driver read the checkpoint once"
    );
}

/// The kernels (by handle identity, independent of any cache key) an independent real preparation of
/// `candidate` loads on a recording device.
fn loaded_kernels(handle: &Arc<ModelHandle>, candidate: &Candidate) -> HashSet<u64> {
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let executor = Engine::new(FixtureDevice {
        target: spirv_target(),
        memory: MemoryCounters::new(),
        inflate_state: 0,
        free_bytes: None,
        capacity_bytes: None,
        loaded_kernels: Some(Arc::clone(&loaded)),
    });
    let driver = Driver::new(
        Arc::clone(handle),
        Box::new(executor),
        driver_options(handle, generate(candidate), DraftPolicy::Off),
    )
    .expect("driver");
    let rows = [candidate.rows];
    let driver = driver
        .prepare(&shapes_of(candidate, &rows))
        .expect("the candidate prepares");
    drop(driver);
    Arc::try_unwrap(loaded)
        .expect("the driver and its device are dropped")
        .into_inner()
        .expect("loaded kernels")
}

/// POOT-1068: one planner resolves each distinct kernel exactly once across the candidates it plans, and
/// sharing changes no plan. The kernel sets come from independent real preparations (what each candidate's
/// engine loads), not from the shared table. A fresh planner resolves exactly its candidate's kernels; the
/// shared planner resolves exactly the kernels no earlier candidate had (so a candidate that differs in a
/// body-affecting dimension resolves its own extra kernels, and a collapsed key resolves too few); the
/// totals are the sum and the union; planning the domain again resolves nothing.
///
/// Mutations: build the `SharedKernels` per `Planner::plan` call (shared total == fresh sum); key the table
/// by the constant 0 or by `fingerprint & 0xF` in `Engine::kernel` (shared total < the union).
#[test]
fn a_planner_resolves_each_distinct_kernel_once_across_candidates() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let targets = probe(spirv_target(), 0).0.target_set();
    let new_planner = || Planner::new(Arc::clone(&handle), &targets, Placement::Single).unwrap();
    let plan = |planner: &Planner, candidate: &Candidate| {
        let rows = [candidate.rows];
        planner
            .plan(
                driver_options(&handle, generate(candidate), DraftPolicy::Off),
                &shapes_of(candidate, &rows),
                &[],
            )
            .expect("plans")
    };
    let domain = candidates(&handle, &args);
    assert!(domain.len() >= 3, "the pass must plan several candidates");
    let shared = new_planner();
    let mut seen = HashSet::new();
    let (mut shared_total, mut fresh_total) = (0, 0);
    let (mut some_extra, mut some_reused) = (false, false);
    for candidate in &domain {
        let kernels = loaded_kernels(&handle, candidate);
        let new = kernels.difference(&seen).count() as u64;
        let with_shared = plan(&shared, candidate);
        let with_fresh = plan(&new_planner(), candidate);
        assert_eq!(with_shared.footprint, with_fresh.footprint, "{candidate:?}");
        assert_eq!(with_shared.residue, with_fresh.residue, "{candidate:?}");
        assert_eq!(
            with_fresh.kernels_resolved,
            kernels.len() as u64,
            "{candidate:?}: a fresh planner resolves its candidate's kernels"
        );
        assert_eq!(
            with_shared.kernels_resolved, new,
            "{candidate:?}: the shared planner resolves only the kernels no earlier candidate had"
        );
        some_extra |= !seen.is_empty() && new > 0;
        some_reused |= new < kernels.len() as u64;
        seen.extend(kernels);
        shared_total += with_shared.kernels_resolved;
        fresh_total += with_fresh.kernels_resolved;
    }
    assert!(
        some_extra,
        "a later candidate must need kernels the earlier ones did not"
    );
    assert!(some_reused, "a later candidate must reuse earlier kernels");
    assert_eq!(shared_total, seen.len() as u64, "each distinct kernel once");
    assert!(fresh_total > shared_total);
    for candidate in &domain {
        assert_eq!(
            plan(&shared, candidate).kernels_resolved,
            0,
            "{candidate:?}: every kernel was already resolved"
        );
    }
}

/// POOT-1068: the startup fit reports exactly the distinct kernels of the candidates it planned. The budget
/// admits only the last candidate, so every earlier one is planned too.
///
/// Mutations: as above; the reported count differs from the independently observed union.
#[test]
fn startup_resolves_each_distinct_kernel_once_however_many_candidates_it_plans() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let domain = candidates(&handle, &args);
    let distinct: HashSet<u64> = domain
        .iter()
        .flat_map(|candidate| loaded_kernels(&handle, candidate))
        .collect();
    let last = domain.len() - 1;
    let (executor, _) = probe(spirv_target(), 0);
    let budget = known(SUPPORT_RESERVE_BYTES + live_total(&handle, &domain[last]));
    let (started, _) = build_with(
        &args,
        &handle,
        executor,
        |_| budget.clone(),
        &EngineHealth::starting(),
    );
    let started = started.expect("the last candidate fits");
    assert_eq!(
        started.plan.budget.planning_kernels_resolved,
        distinct.len() as u64
    );
}

/// SC-004: an unknown budget is a typed refusal of every candidate, never infinity.
#[test]
fn an_unknown_budget_refuses_every_candidate() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let (executor, add_entries) = probe(spirv_target(), 0);
    let unknown = Budget::new().with(
        DeviceId(0),
        poot_llm::driver::capacity::DeviceBudget::Unknown,
    );
    let (result, _) = build_with(
        &args,
        &handle,
        executor,
        |_| unknown.clone(),
        &EngineHealth::starting(),
    );
    let Err(StartupError::Capacity(FitError::NoCandidateFits { rejected })) = result else {
        panic!("expected every candidate refused, got {:?}", result.err());
    };
    assert_eq!(rejected.len(), candidates(&handle, &args).len());
    assert!(rejected.iter().all(|r| matches!(
        r.why,
        poot_llm::driver::capacity::Rejection::Capacity(CapacityRefusal::UnknownBudget { .. })
    )));
    assert_eq!(
        add_entries.load(Ordering::SeqCst),
        0,
        "no rejected candidate was published"
    );
}

/// SC-006: the budget changes between discovery and admission. The open-time observation would admit
/// the first candidate; the fresh one admits only the k-th, and only the k-th is prepared on the real
/// executor (no rejected executable is published). Mutations: admit against the open-time observation
/// (the first candidate is chosen); prepare the real executor before admission (the entry count
/// exceeds the chosen candidate's).
#[test]
fn admission_uses_the_fresh_budget_and_publishes_only_the_chosen_candidate() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let domain = candidates(&handle, &args);
    let totals: Vec<u64> = domain.iter().map(|c| live_total(&handle, c)).collect();
    let k = 2;
    let (reference, reference_entries) = probe(spirv_target(), 0);
    {
        let driver = Driver::new(
            Arc::clone(&handle),
            Box::new(reference),
            driver_options(&handle, generate(domain[k]), DraftPolicy::Off),
        )
        .expect("driver");
        let rows = [domain[k].rows];
        let _prepared = driver
            .prepare(&shapes_of(&domain[k], &rows))
            .expect("prepares");
    }
    let chosen_entries = reference_entries.load(Ordering::SeqCst);
    let (executor, add_entries) = probe(spirv_target(), 0);
    let mut calls = 0;
    let (started, _) = build_with(
        &args,
        &handle,
        executor,
        |_| {
            calls += 1;
            if calls == 1 {
                known(u64::MAX / 2) // discovery: room for everything
            } else {
                known(SUPPORT_RESERVE_BYTES + totals[k]) // now: room for the k-th
            }
        },
        &EngineHealth::starting(),
    );
    let started = started.expect("the k-th candidate fits the fresh budget");
    let Workload::Generate { rows, max_seq } = started.plan.workload else {
        panic!("a generative model plans generation");
    };
    assert_eq!((rows, max_seq), (domain[k].rows, domain[k].max_seq));
    assert_eq!(
        add_entries.load(Ordering::SeqCst),
        chosen_entries,
        "only the chosen candidate's entries were added to the real executor"
    );
}

/// SC-005: live allocations over what the plan counted are refused before `Serving`: a runtime that
/// holds 64 bytes more `State` than the planned footprint. Mutation: skip `verify_prepared`; the plan
/// is admitted with the health `Serving`.
#[test]
fn live_allocations_over_the_plan_are_refused_before_serving() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let (executor, _) = probe(spirv_target(), 64);
    let (result, _) = build_with(
        &args(BackendChoice::Wgpu),
        &handle,
        executor,
        roomy,
        &health,
    );
    let Err(StartupError::Admission(CapacityRefusal::Underestimated {
        role,
        planned,
        actual,
        ..
    })) = result
    else {
        panic!(
            "expected the plan refused as under-counted, got {:?}",
            result.err()
        );
    };
    assert_eq!(role, BufferRole::State);
    assert!(actual > planned, "{actual} live over {planned} planned");
    assert_eq!(health.state(), EngineState::Starting);
}

/// SC-005: a budget that drops after the fit, before the live check, refuses the prepared engine: the
/// live bytes are admitted against the budget observed now, not the one the fit used. Mutation: skip the
/// final admission.
#[test]
fn live_allocations_are_admitted_against_the_budget_observed_after_prepare() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let first = candidates(&handle, &args)[0];
    let total = live_total(&handle, &first);
    let (executor, _) = probe(spirv_target(), 0);
    let mut calls = 0;
    let (result, _) = build_with(
        &args,
        &handle,
        executor,
        |_| {
            calls += 1;
            // Discovery, then the fit's one admission: roomy. The check after prepare: one byte short.
            if calls <= 2 {
                known(SUPPORT_RESERVE_BYTES + total)
            } else {
                known(SUPPORT_RESERVE_BYTES + total - 1)
            }
        },
        &EngineHealth::starting(),
    );
    let Err(StartupError::Admission(CapacityRefusal::OverBudget {
        required,
        available,
        ..
    })) = result
    else {
        panic!(
            "expected the prepared engine over the fresh budget, got {:?}",
            result.err()
        );
    };
    assert_eq!((required, available), (total, total - 1));
}

/// POOT-1204 SC-001: an embedding or rerank model is planned and prepared like a generative one: the
/// fitted encode shapes compile before `Serving`, so a request afterwards compiles nothing, adds no
/// executor entry and allocates no device byte. The request runs through the production
/// `Driver::encode` on the counting fixture device. Mutation: skip the entry loop in
/// `Driver::prepare_encode` and prepare on a miss in `Driver::encode` (the lazy path); the request
/// adds an entry, compiles, and the counters move.
#[test]
fn an_encoder_request_after_serving_compiles_and_allocates_nothing() {
    for (variant, label) in [
        (None, "hidden states"),
        (Some("cross_encoder"), "classifier scores"),
    ] {
        let handle = crate::tests::encoder_routes::encoder_handle(variant);
        let health = EngineHealth::starting();
        let (executor, add_entries) = probe(spirv_target(), 0);
        let (result, opened) = build_with(
            &args(BackendChoice::Wgpu),
            &handle,
            executor,
            roomy,
            &health,
        );
        let Started { plan, mut driver } =
            result.unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(opened.len(), 1, "{label}");
        assert!(matches!(health.state(), EngineState::Serving { .. }));
        let Workload::Encode { rows, max_seq } = plan.workload else {
            panic!(
                "{label}: an encoder is an encode workload, got {:?}",
                plan.workload
            );
        };
        assert_eq!(
            (rows.get(), max_seq.get()),
            (4, 16),
            "{label}: the requested rows, the model's positions"
        );
        let fitted = plan
            .caps
            .encode
            .as_ref()
            .expect("the caps carry the encode shapes");
        let lengths: Vec<usize> = fitted.lengths.iter().map(|len| len.get()).collect();
        assert_eq!(lengths, [1, 2, 4, 8, 16], "{label}");

        let entries = add_entries.load(Ordering::SeqCst);
        let compiles = driver.stats().compiles;
        let live = driver.live_role_bytes();
        let ids = [1u32, 2, 3];
        let none = poot_models::io::RequestInputs::none();
        let rows = [
            poot_llm::driver::EncodeRow {
                tokens: Some(&ids),
                inputs: &none,
            },
            poot_llm::driver::EncodeRow {
                tokens: Some(&ids[..2]),
                inputs: &none,
            },
        ];
        driver
            .encode(&poot_llm::driver::EncodeRequest { rows: &rows })
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(
            add_entries.load(Ordering::SeqCst),
            entries,
            "{label}: no entry added"
        );
        assert_eq!(
            driver.stats().compiles,
            compiles,
            "{label}: nothing compiled"
        );
        assert_eq!(driver.live_role_bytes(), live, "{label}: nothing allocated");
    }
}

/// POOT-1204 review: a generative model whose family traces an embedding pass plans its encode entries
/// beside its decode ones, from the same fit, so a decoder keeps `/v1/embeddings` and `/v1/rerank` and a
/// request after `Serving` compiles and allocates nothing. A family that refuses the `Encode` phase
/// serves generation only: no encode caps. Mutation: skip the `IfTraced` encode preparation in
/// `fit_and_prepare`: the decoder's caps carry no encode shapes and the request is refused.
#[test]
fn a_decoder_plans_its_embedding_pass_and_a_family_without_one_serves_generation_only() {
    let decoder = dense_model();
    let (executor, add_entries) = probe(spirv_target(), 0);
    let health = EngineHealth::starting();
    let (result, _) = build_with(
        &args(BackendChoice::Wgpu),
        &decoder,
        executor,
        roomy,
        &health,
    );
    let Started { plan, mut driver } = result.expect("the decoder starts");
    assert!(matches!(plan.workload, Workload::Generate { .. }));
    let fitted = plan
        .caps
        .encode
        .as_ref()
        .expect("the decoder plans its embedding pass");
    assert_eq!(
        fitted.rows.get(),
        plan.rows(),
        "the encode rows are the fitted rows"
    );
    let (entries, compiles, live) = (
        add_entries.load(Ordering::SeqCst),
        driver.stats().compiles,
        driver.live_role_bytes(),
    );
    let ids = [1u32, 2, 3];
    let none = poot_models::io::RequestInputs::none();
    driver
        .encode(&poot_llm::driver::EncodeRequest {
            rows: &[poot_llm::driver::EncodeRow {
                tokens: Some(&ids),
                inputs: &none,
            }],
        })
        .expect("the planned shape runs");
    assert_eq!(
        add_entries.load(Ordering::SeqCst),
        entries,
        "no entry added"
    );
    assert_eq!(driver.stats().compiles, compiles, "nothing compiled");
    assert_eq!(driver.live_role_bytes(), live, "nothing allocated");

    let recurrent = recurrent_handle();
    let (executor, _) = probe(spirv_target(), 0);
    let health = EngineHealth::starting();
    let (result, _) = build_with(
        &args(BackendChoice::Wgpu),
        &recurrent,
        executor,
        roomy,
        &health,
    );
    let started = result.expect("a family without an embedding pass still serves generation");
    assert!(
        started.plan.caps.encode.is_none(),
        "{:?}",
        started.plan.caps
    );
}

/// POOT-1204 SC-004: `--rows` and `--max-seq` are the shapes the plan is fitted at, through the real
/// parser and `engine_args`. A model with 1024 positions at `--rows 2 --max-seq 1024` is planned at
/// exactly those shapes. Mutation: build `EngineArgs` from the defaults (hard-code 4 rows) in
/// `engine_args`; the plan has four rows and the row goes red.
#[test]
fn rows_and_max_seq_flags_are_the_shapes_the_plan_is_fitted_at() {
    let config = ServeConfig::try_parse_from(["poot-serve", "--rows", "2", "--max-seq", "1024"])
        .expect("the flags parse");
    let args = crate::startup::engine_args(&config, DraftPolicy::Off).expect("engine args");
    let handle = dense_model_of(1024);
    let health = EngineHealth::starting();
    let (executor, _) = probe(spirv_target(), 0);
    let (result, _) = build_with(&args, &handle, executor, roomy, &health);
    let started = result.expect("a roomy device fits the requested shapes");
    assert_eq!(
        started.plan.workload,
        Workload::Generate {
            rows: NonZeroUsize::new(2).unwrap(),
            max_seq: NonZeroUsize::new(1024).unwrap(),
        }
    );
    assert_eq!(started.plan.caps.max_seq.get(), 1024);
    assert_eq!(started.plan.caps.rows.get(), 2);
}

/// The budget is the runtime's live answer, not the figure the device queried when it opened. The target
/// reports an enormous open-time budget; the runtime reports it has room for exactly the first
/// candidate's live bytes (plus the reserve) and no more. Through the production observer the first
/// candidate is admitted and the next is not, and a runtime reporting one byte less than the first
/// candidate's total refuses the whole domain with a typed over-budget refusal.
/// Mutation: read the open-time figure in `Budget::observed` (ignore `free_bytes`) and the lower live
/// budget admits the first candidate: both assertions go red.
#[test]
fn a_lower_live_budget_than_the_open_time_figure_refuses_the_plan() {
    let handle = fixture_model();
    let args = args(BackendChoice::Wgpu);
    let first = candidates(&handle, &args)[0];
    let mut target = spirv_target();
    gpu_caps_mut(&mut target).memory_budget_bytes = Some(u64::MAX / 2);
    // The independent oracle: what the first candidate allocates on a counting device.
    let (executor, _) = probe(target, 0);
    let oracle = Driver::new(
        Arc::clone(&handle),
        Box::new(executor),
        driver_options(&handle, generate(first), DraftPolicy::Off),
    )
    .unwrap();
    let oracle = oracle.prepare(&shapes_of(&first, &[first.rows])).unwrap();
    let first_bytes: u64 = oracle.live_role_bytes().iter().map(|&(_, b)| b).sum();
    drop(oracle);

    // Room for exactly the first candidate: admitted, with the open-time ceiling far above it.
    let (executor, _) = probe_reporting(target, 0, Some(first_bytes + SUPPORT_RESERVE_BYTES));
    let (started, _) = build_with(
        &args,
        &handle,
        executor,
        device_budget,
        &EngineHealth::starting(),
    );
    let started = started.expect("a runtime with room for the first candidate admits it");
    assert_eq!(
        started.plan.workload,
        Workload::Generate {
            rows: first.rows,
            max_seq: first.max_seq,
        }
    );

    // One byte less than the smallest candidate needs, though the open-time figure says plenty.
    let smallest = candidates(&handle, &args)
        .into_iter()
        .map(|c| {
            let rows = [c.rows];
            let targets = probe(target, 0).0.target_set();
            let &[(device, _)] = targets.devices() else {
                panic!("one device");
            };
            Planner::new(Arc::clone(&handle), &targets, Placement::Single)
                .unwrap()
                .plan(
                    driver_options(&handle, generate(c), DraftPolicy::Off),
                    &shapes_of(&c, &rows),
                    &[],
                )
                .unwrap()
                .footprint
                .device_bytes(device)
                .unwrap()
        })
        .min()
        .unwrap();
    let (executor, add_entries) =
        probe_reporting(target, 0, Some(smallest + SUPPORT_RESERVE_BYTES - 1));
    let health = EngineHealth::starting();
    let (result, _) = build_with(&args, &handle, executor, device_budget, &health);
    let Err(StartupError::Capacity(FitError::NoCandidateFits { rejected })) = result else {
        panic!("expected every candidate refused, got {:?}", result.err());
    };
    assert!(
        rejected.iter().all(|r| matches!(
            r.why,
            poot_llm::driver::capacity::Rejection::Capacity(CapacityRefusal::OverBudget { .. })
        )),
        "{rejected:?}"
    );
    assert_eq!(
        add_entries.load(Ordering::SeqCst),
        0,
        "nothing was published"
    );
    assert_eq!(health.state(), EngineState::Starting);
}

/// A runtime that cannot say what is free is an unknown budget, refused typed: the open-time figure is
/// never promoted to a budget. Mutation: fall back to the caps figure when `free_bytes` is `None`.
#[test]
fn a_runtime_that_cannot_report_free_memory_is_an_unknown_budget() {
    let handle = fixture_model();
    let mut target = spirv_target();
    gpu_caps_mut(&mut target).memory_budget_bytes = Some(u64::MAX / 2);
    let (executor, add_entries) = probe_reporting(target, 0, None);
    let health = EngineHealth::starting();
    let (result, _) = build_with(
        &args(BackendChoice::Wgpu),
        &handle,
        executor,
        device_budget,
        &health,
    );
    let Err(StartupError::Capacity(FitError::NoCandidateFits { rejected })) = result else {
        panic!("expected the typed refusal, got {:?}", result.err());
    };
    assert!(
        rejected.iter().all(|r| matches!(
            r.why,
            poot_llm::driver::capacity::Rejection::Capacity(CapacityRefusal::UnknownBudget { .. })
        )),
        "{rejected:?}"
    );
    assert_eq!(add_entries.load(Ordering::SeqCst), 0);
    assert_eq!(health.state(), EngineState::Starting);
}

/// Weights are the largest term. The tiny model carries real tensors: after prepare the plan's `Weight`
/// bytes equal the live ones from an independent counting device, and both equal the bytes of the stored
/// tensors. The oracle is computed from the GGUF's own tensor directory: every F32 tensor except the
/// router's selection bias, which the engine does not upload as a device weight (observed, not derived:
/// removing it from the sum is the one adjustment the fixture needs). A packed or quantized fixture does not exist
/// among the shipped fixtures, and a real-config re-derivation (weights + KV + state + activations) is
/// not possible from them.
#[test]
fn planned_weight_bytes_equal_the_stored_tensor_bytes() {
    let path = write_tiny_gguf();
    let handle = load_handle(&path);
    // The oracle reads the file's own tensor directory, not poot's store: every tensor is F32.
    let index = poot_load::gguf::GgufIndex::parse(
        &poot_quant::StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap(),
    )
    .expect("tensor directory");
    let _ = std::fs::remove_file(&path);
    let oracle: u64 = index
        .tensors
        .iter()
        .filter(|(name, _)| name.as_str() != "blk.1.exp_probs_b.bias")
        .map(|(_, info)| info.dims.iter().product::<u64>() * 4)
        .sum();
    let args = args(BackendChoice::Wgpu);
    let candidate = candidates(&handle, &args)[0];
    let planner = Planner::new(
        Arc::clone(&handle),
        &probe(spirv_target(), 0).0.target_set(),
        Placement::Single,
    )
    .unwrap();
    let rows = [candidate.rows];
    let planned = planner
        .plan(
            driver_options(&handle, generate(candidate), DraftPolicy::Off),
            &shapes_of(&candidate, &rows),
            &[],
        )
        .expect("plans");
    let weight = |roles: Vec<(BufferRole, u64)>| {
        roles
            .into_iter()
            .find(|(r, _)| *r == BufferRole::Weight)
            .map_or(0, |(_, b)| b)
    };
    let (executor, _) = probe(spirv_target(), 0);
    let driver = Driver::new(
        Arc::clone(&handle),
        Box::new(executor),
        driver_options(&handle, generate(candidate), DraftPolicy::Off),
    )
    .unwrap();
    let driver = driver.prepare(&shapes_of(&candidate, &rows)).unwrap();
    let live = weight(driver.live_role_bytes());
    assert!(live > 0, "the fixture carries weights");
    assert_eq!(
        weight(planned.footprint.role_bytes(DeviceId(0)).unwrap()),
        live,
        "planned vs live"
    );
    assert_eq!(live, oracle, "live vs the stored tensor sizes");
}

/// A Qwen3-Next fixture (gated delta rule layers beside attention) as an HF checkpoint directory the
/// serve crate loads like any other model: recurrent state, the family the engine plan once refused to
/// plan beyond one row.
fn recurrent_handle() -> Arc<ModelHandle> {
    let registry = poot_models::registry::Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen3next")
        .expect("the builtin qwen3next family");
    let fixture = (entry.fixture)();
    let dir = poot_test_util::unique_temp_path("poot_serve_qwen3next_fixture");
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(0);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
    let words: serde_json::Map<String, serde_json::Value> = (0..48)
        .map(|i| (format!("w{i}"), serde_json::json!(i)))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": words, "unk_token": "w0" },
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    Arc::new(
        ModelHandle::load(&dir, &registry, &poot_llm::driver::LoadOptions::default())
            .expect("load the qwen3next fixture"),
    )
}

/// POOT-1074: the engine plan fits a recurrent family at more than one row, and the planned `State` bytes
/// are the engine's own pool allocation: they grow by one row's bytes per sequence the pool serves, and do
/// not depend on the step's row count. Planning is the real planner over the counting device.
///
/// Mutation: trace the pool at `shape.rows` instead of `seqs` in `Step::recurrent_state`; the pool no longer
/// matches the sequences the driver admits (`check_state_rows` refuses it) and the plan fails.
#[test]
fn a_recurrent_family_plans_many_rows_and_its_state_bytes_grow_with_the_sequences() {
    let handle = recurrent_handle();
    let planner = Planner::new(
        Arc::clone(&handle),
        &probe(spirv_target(), 0).0.target_set(),
        Placement::Single,
    )
    .unwrap();
    let max_seq = NonZeroUsize::new(64).unwrap();
    let state_bytes = |seqs: usize, rows: usize| -> u64 {
        let pool = PoolShape {
            blocks: NonZeroUsize::new(8).unwrap(),
            max_seqs: NonZeroUsize::new(seqs).unwrap(),
        };
        let rows = [NonZeroUsize::new(rows).unwrap()];
        let planned = planner
            .plan(
                driver_options(
                    &handle,
                    Workload::Generate {
                        rows: rows[0],
                        max_seq,
                    },
                    DraftPolicy::Off,
                ),
                &ServingShapes {
                    layout: Layout::Paged(pool),
                    rows: &rows,
                    heads: &[Head::GREEDY],
                    warm: Warm::Admitted,
                    windows: &[],
                    adapters: &[],
                },
                &[],
            )
            .unwrap_or_else(|error| panic!("seqs {seqs} rows {rows:?}: {error}"));
        planned
            .footprint
            .role_bytes(DeviceId(0))
            .unwrap()
            .into_iter()
            .find(|&(role, _)| role == BufferRole::State)
            .map_or(0, |(_, bytes)| bytes)
    };
    let (one, two, four) = (state_bytes(1, 1), state_bytes(2, 2), state_bytes(4, 2));
    assert!(two > one, "a second sequence adds a row: {one} -> {two}");
    assert_eq!(
        four - one,
        3 * (two - one),
        "state grows by one row's bytes per sequence: {one}, {two}, {four}"
    );
    assert_eq!(
        state_bytes(4, 4),
        four,
        "the step's row count does not size the state"
    );

    let args = args(BackendChoice::Wgpu);
    let (executor, _) = probe(spirv_target(), 0);
    let health = EngineHealth::starting();
    let (started, _) = build_with(&args, &handle, executor, roomy, &health);
    let started = started.expect("the recurrent family starts");
    let Workload::Generate { rows, .. } = started.plan.workload else {
        panic!("a generative model plans generation");
    };
    assert!(rows.get() > 1, "the fitted plan serves {rows} rows");
}

// ---------------------------------------------------------------------------------------------
// A group of devices (POOT-1103).
// ---------------------------------------------------------------------------------------------

/// The fixture device takes part in a group: nothing it exchanges is real, but the coordinator admits its
/// routes, events and copies as it would a runtime's.
impl Collectives for FixtureDevice {
    type Event = ();

    fn native_counters(&self) -> NativeCounters {
        NativeCounters::default()
    }

    fn support_bound(&self) -> SupportBound {
        SupportBound::default()
    }

    fn open_route(&mut self, _peer: &Self) -> Result<(), FixtureError> {
        Ok(())
    }

    fn create_event(&mut self) -> Result<(), FixtureError> {
        Ok(())
    }

    fn record_event(&mut self, _event: &()) -> Result<(), FixtureError> {
        Ok(())
    }

    fn wait_event(&mut self, _event: &()) -> Result<(), FixtureError> {
        Ok(())
    }

    fn peer_copy(&mut self, _src: &(), _dst: &(), _bytes: usize) -> Result<(), FixtureError> {
        Ok(())
    }
}

fn ptx_target() -> Target {
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: DeviceCaps {
            memory_budget_bytes: Some(1 << 40),
            ..DeviceCaps::ptx_default()
        },
    }
}

/// Two fixture devices under one coordinator, each reporting `loaded` kernels it is handed.
fn group(loaded: &Arc<Mutex<HashSet<u64>>>) -> Coordinator<FixtureDevice> {
    group_inflating(loaded, [0, 0])
}

/// [`group`] whose devices charge `inflate[rank]` extra `State` bytes per allocation, as a runtime that
/// holds more on that rank than the plan counted.
fn group_inflating(
    loaded: &Arc<Mutex<HashSet<u64>>>,
    inflate: [u64; 2],
) -> Coordinator<FixtureDevice> {
    group_of(loaded, inflate, [None, None])
}

/// [`group`] whose device `rank` holds `capacity[rank]` bytes in all and reports the rest as free, with
/// `inflate[rank]` extra `State` bytes per allocation.
fn group_of(
    loaded: &Arc<Mutex<HashSet<u64>>>,
    inflate: [u64; 2],
    capacity: [Option<u64>; 2],
) -> Coordinator<FixtureDevice> {
    Coordinator::new(
        (0..2)
            .map(|index| {
                (
                    DeviceId(index),
                    FixtureDevice {
                        target: ptx_target(),
                        memory: MemoryCounters::new(),
                        inflate_state: inflate[index as usize],
                        loaded_kernels: Some(Arc::clone(loaded)),
                        free_bytes: Some(u64::MAX / 2),
                        capacity_bytes: capacity[index as usize],
                    },
                )
            })
            .collect(),
    )
    .expect("two distinct devices")
}

fn group_args(placement: PlacementSpec) -> EngineArgs {
    EngineArgs {
        placement,
        ..args(BackendChoice::Ptx)
    }
}

fn pipeline_over_two() -> PlacementSpec {
    PlacementSpec::Pipeline {
        ordinals: vec![0, 1],
    }
}

/// The two devices' budgets, each with the production reserve.
fn known_pair(first: u64, second: u64) -> Budget {
    let device = |bytes| poot_llm::driver::capacity::DeviceBudget::Known {
        bytes,
        reserve: SUPPORT_RESERVE_BYTES,
    };
    Budget::new()
        .with(DeviceId(0), device(first))
        .with(DeviceId(1), device(second))
}

fn build_group(
    args: &EngineArgs,
    handle: &Arc<ModelHandle>,
    group: Coordinator<FixtureDevice>,
    observe: impl FnMut(&Observation) -> Budget,
    health: &EngineHealth,
) -> Result<Started, StartupError> {
    let mut group = Some(group);
    EnginePlan::build(
        args,
        handle,
        |args| {
            let group = Box::new(Coordinated(group.take().expect("opened once")));
            Ok(match args.placement {
                PlacementSpec::Tensor { .. } => Opened::Tensor(group),
                PlacementSpec::Pipeline { .. } | PlacementSpec::Single => Opened::Pipeline(group),
            })
        },
        observe,
        health,
    )
}

/// The bytes the smallest candidate of the domain needs on each device of a two-stage pipeline, by the
/// planner on its own: the figure a budget row sets its limit from.
fn smallest_candidate_bytes(
    handle: &Arc<ModelHandle>,
    args: &EngineArgs,
    placement: Placement,
) -> [u64; 2] {
    let candidate = *candidates(handle, args).last().expect("a domain");
    let targets = TargetSet::new(vec![
        (DeviceId(0), ptx_target()),
        (DeviceId(1), ptx_target()),
    ])
    .unwrap();
    let rows = [candidate.rows];
    let planned = Planner::new(Arc::clone(handle), &targets, placement)
        .expect("a planner")
        .plan(
            driver_options(handle, generate(candidate), DraftPolicy::Off),
            &shapes_of(&candidate, &rows),
            &[],
        )
        .expect("the candidate plans");
    [DeviceId(0), DeviceId(1)].map(|device| planned.footprint.device_bytes(device).unwrap())
}

/// A two-stage pipeline over two devices opens as one executor, is balanced and fitted per device, and
/// publishes `Serving`: the plan names both devices, the placement it ran and each device's own bytes.
///
/// Mutation: report only the first device in the plan (`devices` from the first target): the row goes
/// red on the device list and the per-device figures.
#[test]
fn a_two_device_pipeline_is_fitted_per_device_and_published() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let started = build_group(
        &group_args(pipeline_over_two()),
        &handle,
        group(&loaded),
        |_| known_pair(u64::MAX / 2, u64::MAX / 2),
        &health,
    )
    .unwrap_or_else(|error| panic!("the pipeline starts: {error}"));
    let plan = &started.plan;
    assert_eq!(plan.devices, [DeviceId(0), DeviceId(1)]);
    assert_eq!(
        plan.placement,
        Placement::Pipeline {
            after_layers: vec![0]
        },
        "a two-layer model over two devices cuts after its first layer"
    );
    assert_eq!(plan.budget.devices.len(), 2);
    assert!(
        plan.budget.devices.iter().all(|d| d.planned_bytes > 0),
        "each device owns planned bytes: {:?}",
        plan.budget.devices
    );
    assert!(matches!(health.state(), EngineState::Serving { .. }));
    assert!(
        !loaded.lock().unwrap().is_empty(),
        "the prepared entries loaded kernels on the group"
    );
}

/// An under-budget device is refused before `Serving`, with the typed refusal naming that device: the
/// other device has room for the whole workload, so the devices' summed budgets would admit it.
///
/// Mutation: admit a footprint against the sum of the budgets (`Budget::admit` over the totals): the
/// plan starts over a device that cannot hold its stage and this row goes red.
#[test]
fn an_under_budget_device_is_refused_before_serving_naming_that_device() {
    let handle = fixture_model();
    let args = group_args(pipeline_over_two());
    let needs = smallest_candidate_bytes(
        &handle,
        &args,
        Placement::Pipeline {
            after_layers: vec![0],
        },
    );
    assert!(needs[1] > 1, "the second stage owns bytes: {needs:?}");
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    // The second device is a byte short of the smallest candidate's own bytes; the first has room for
    // everything, so the sum of the two figures is far above what the whole plan needs.
    let short = known_pair(u64::MAX / 2, SUPPORT_RESERVE_BYTES + needs[1] - 1);
    let result = build_group(&args, &handle, group(&loaded), |_| short.clone(), &health);
    let Err(StartupError::Capacity(FitError::NoCandidateFits { rejected })) = result else {
        panic!("expected every candidate refused, got {:?}", result.err());
    };
    assert_eq!(rejected.len(), candidates(&handle, &args).len());
    for rejection in &rejected {
        match &rejection.why {
            Rejection::Capacity(CapacityRefusal::OverBudget { device, .. }) => {
                assert_eq!(*device, DeviceId(1), "{rejection:?}");
            }
            other => panic!("expected an over-budget refusal, got {other:?}"),
        }
    }
    assert!(matches!(health.state(), EngineState::Starting));
    assert!(
        loaded.lock().unwrap().is_empty(),
        "nothing was prepared on the real devices"
    );
}

/// A device that cannot hold the weights of its own stage refuses the pipeline's balance, before any
/// candidate is planned.
#[test]
fn a_stage_that_cannot_hold_its_layers_is_refused_by_the_balance() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let tiny = known_pair(u64::MAX / 2, SUPPORT_RESERVE_BYTES + 1);
    let result = build_group(
        &group_args(pipeline_over_two()),
        &handle,
        group(&loaded),
        |_| tiny.clone(),
        &health,
    );
    let Err(StartupError::Pipeline(BalanceError::DeviceTooSmall { device, .. })) = result else {
        panic!("expected the balance refusal, got {:?}", result.err());
    };
    assert_eq!(device, DeviceId(1));
    assert!(matches!(health.state(), EngineState::Starting));
}

/// An unknown budget on one device of a pipeline is a typed refusal, never room.
#[test]
fn a_group_whose_budget_is_unknown_is_refused_not_assumed_unlimited() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let unknown = Budget::new()
        .with(
            DeviceId(0),
            poot_llm::driver::capacity::DeviceBudget::Known {
                bytes: u64::MAX / 2,
                reserve: SUPPORT_RESERVE_BYTES,
            },
        )
        .with(
            DeviceId(1),
            poot_llm::driver::capacity::DeviceBudget::Unknown,
        );
    let result = build_group(
        &group_args(pipeline_over_two()),
        &handle,
        group(&loaded),
        |_| unknown.clone(),
        &health,
    );
    assert!(
        matches!(
            result,
            Err(StartupError::Admission(CapacityRefusal::UnknownBudget {
                device: DeviceId(1)
            }))
        ),
        "{:?}",
        result.err()
    );
}

/// A tensor placement of a model the rule has no split for (this fixture's latent attention) is the
/// shard pass's typed refusal, surfaced as the startup error before anything is prepared.
#[test]
fn a_tensor_placement_of_an_unsplittable_model_is_a_typed_refusal() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let result = build_group(
        &group_args(PlacementSpec::Tensor {
            ordinals: vec![0, 1],
        }),
        &handle,
        group(&loaded),
        |_| known_pair(u64::MAX / 2, u64::MAX / 2),
        &health,
    );
    assert!(
        matches!(
            result,
            Err(StartupError::Driver(DriverError::Unsupported(
                Unsupported::Compile(poot_graph_plan::staged::StagedCompileError::Shard(_))
            )))
        ),
        "{:?}",
        result.err()
    );
    assert!(matches!(health.state(), EngineState::Starting));
}

/// A dense two-head qwen2 checkpoint the Megatron rule can divide over two ranks.
fn dense_model() -> Arc<ModelHandle> {
    dense_model_of(64)
}

/// [`dense_model`] with `positions` context positions.
fn dense_model_of(positions: u32) -> Arc<ModelHandle> {
    use poot_load::gguf::{GgufValue, write_gguf};
    let (h, inter, vocab) = (8usize, 16usize, 8usize);
    let mut tensors = Vec::new();
    for (name, shape) in [
        ("token_embd.weight", vec![h, vocab]),
        ("output.weight", vec![h, vocab]),
        ("output_norm.weight", vec![h]),
        ("blk.0.attn_q.weight", vec![h, h]),
        ("blk.0.attn_k.weight", vec![h, h]),
        ("blk.0.attn_v.weight", vec![h, h]),
        ("blk.0.attn_output.weight", vec![h, h]),
        ("blk.0.attn_q.bias", vec![h]),
        ("blk.0.attn_k.bias", vec![h]),
        ("blk.0.attn_v.bias", vec![h]),
        ("blk.0.attn_norm.weight", vec![h]),
        ("blk.0.ffn_norm.weight", vec![h]),
        ("blk.0.ffn_gate.weight", vec![h, inter]),
        ("blk.0.ffn_up.weight", vec![h, inter]),
        ("blk.0.ffn_down.weight", vec![inter, h]),
    ] {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| (((i % 7) as f32) * 0.1 - 0.3).to_le_bytes())
            .collect();
        tensors.push((
            name,
            shape.into_iter().map(|d| d as u64).collect::<Vec<u64>>(),
            0u32,
            bytes,
        ));
    }
    let kvs = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", GgufValue::U32(h as u32)),
        ("qwen2.block_count", GgufValue::U32(1)),
        ("qwen2.attention.head_count", GgufValue::U32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::U32(2)),
        ("qwen2.feed_forward_length", GgufValue::U32(inter as u32)),
        ("qwen2.context_length", GgufValue::U32(positions)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(7)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "c", "d", "e", "f", "g", "ab"]
                    .into_iter()
                    .map(|s| GgufValue::Str(s.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let path = std::env::temp_dir().join(format!(
        "poot_serve_qwen2_fixture_{}.gguf",
        crate::tests::unique_fixture_suffix()
    ));
    std::fs::write(&path, write_gguf(&kvs, &tensors)).expect("write the dense GGUF");
    let handle = load_handle(&path);
    let _ = std::fs::remove_file(&path);
    handle
}

fn tensor_over_two() -> PlacementSpec {
    PlacementSpec::Tensor {
        ordinals: vec![0, 1],
    }
}

/// The server's paged workload divides over tensor ranks (POOT-1104): the shard pass cuts the scatter
/// into the shared KV pool and the read back by KV heads, so a tensor group over the paged pool opens as
/// one executor, is fitted per rank and publishes `Serving`.
///
/// Mutation: remove the `ScatterUpdate` rule of the shard pass (`Sharder::rewrite`): the pass refuses
/// the paged write by name (`Unpropagatable`) before anything is prepared and this row goes red.
#[test]
fn a_tensor_group_over_the_paged_pool_is_fitted_per_rank_and_published() {
    let handle = dense_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let started = build_group(
        &group_args(tensor_over_two()),
        &handle,
        group(&loaded),
        |_| known_pair(u64::MAX / 2, u64::MAX / 2),
        &health,
    )
    .unwrap_or_else(|error| panic!("the tensor group starts over the paged pool: {error}"));
    let plan = &started.plan;
    assert_eq!(plan.devices, [DeviceId(0), DeviceId(1)]);
    assert!(
        plan.budget.devices.iter().all(|d| d.planned_bytes > 0),
        "each rank owns planned bytes: {:?}",
        plan.budget.devices
    );
    assert!(matches!(health.state(), EngineState::Serving { .. }));
    assert!(
        !loaded.lock().unwrap().is_empty(),
        "the prepared entries loaded kernels on the group"
    );
}

/// A rank whose live state is over its own plan is refused before `Serving` and named, whichever rank it
/// is: the other rank sitting at its plan does not hide it, and the totals over both ranks are not the
/// check.
///
/// Mutation: compare roles summed over the ranks with the plan's summed roles (the old group check):
/// a rank that overran by less than the plan's slack elsewhere is admitted, and the named rank is lost.
#[test]
fn a_rank_whose_live_state_exceeds_its_own_plan_is_refused_and_named() {
    let handle = fixture_model();
    for (inflate, rank) in [([1 << 20, 0], DeviceId(0)), ([0, 1 << 20], DeviceId(1))] {
        let health = EngineHealth::starting();
        let loaded = Arc::new(Mutex::new(HashSet::new()));
        let result = build_group(
            &group_args(pipeline_over_two()),
            &handle,
            group_inflating(&loaded, inflate),
            |_| known_pair(u64::MAX / 2, u64::MAX / 2),
            &health,
        );
        let Err(StartupError::Admission(CapacityRefusal::Underestimated { device, role, .. })) =
            result
        else {
            panic!("expected the underestimate refusal, got {:?}", result.err());
        };
        assert_eq!(
            (device, role),
            (rank, BufferRole::State),
            "inflate {inflate:?}"
        );
        assert!(matches!(health.state(), EngineState::Starting));
    }
}

/// A rank whose free memory fell after the fit is refused before `Serving` against its own fresh budget,
/// even though the other rank has room for everything.
///
/// Mutation: skip the fresh admission for a group (verify roles only): the group starts and this row
/// goes red.
#[test]
fn a_rank_whose_budget_dropped_after_the_fit_is_refused_before_serving() {
    let handle = fixture_model();
    let health = EngineHealth::starting();
    let loaded = Arc::new(Mutex::new(HashSet::new()));
    let result = build_group(
        &group_args(pipeline_over_two()),
        &handle,
        group(&loaded),
        |observation| {
            // Nothing is held until the engine prepares: the fit sees room, the check after it sees
            // rank 1 down to its reserve.
            if observation
                .devices
                .iter()
                .all(|device| device.held_bytes == 0)
            {
                known_pair(u64::MAX / 2, u64::MAX / 2)
            } else {
                known_pair(u64::MAX / 2, SUPPORT_RESERVE_BYTES + 1)
            }
        },
        &health,
    );
    let Err(StartupError::Admission(CapacityRefusal::OverBudget { device, .. })) = result else {
        panic!("expected the fresh-budget refusal, got {:?}", result.err());
    };
    assert_eq!(device, DeviceId(1));
    assert!(matches!(health.state(), EngineState::Starting));
}
