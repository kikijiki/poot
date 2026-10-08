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
    BufferRole, Device, DeviceTime, Dispatch, Engine, EntryId, ExecError, ExecutableId, Executor,
    ExecutorStats, HostSync, StateScope, StepInputs, StepOutputs,
};
use poot_graph_ir::ValidationOutputs;
use poot_graph_plan::{Capability, StagedProgram, Submission, Target, TargetSet};
use poot_llm::driver::{BackendChoice, ModelHandle};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::{CompiledKernel, MemoryCounterSnapshot, MemoryCounters};
use poot_target::{AmdArch, Backend, DeviceCaps, Queried};

use super::*;
use crate::admin::LoraAdminPolicy;
use crate::http::{RateLimiter, handle};
use crate::tests::{load_handle, test_metrics, write_tiny_gguf};
use crate::types::Backend as ServedBackend;
use poot_llm::driver::capacity::{CapacityRefusal, FitError, Observation};

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
        self.free_bytes
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
    fn begin(&mut self, _submission: Submission) -> Result<(), FixtureError> {
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
    fn write(&mut self, _dst: &(), _bytes: &[u8]) -> Result<(), FixtureError> {
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
        program: &StagedProgram<ValidationOutputs>,
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
    Target {
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
        }),
        add_entries: Arc::clone(&add_entries),
        gate: None,
    };
    (executor, add_entries)
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
        rows: NonZeroUsize::new(4).unwrap(),
        max_seq: NonZeroUsize::new(64).unwrap(),
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
    let mut driver = Driver::new(
        Arc::clone(handle),
        Box::new(executor),
        driver_options(handle, candidate.max_seq),
    )
    .expect("driver");
    let rows = [candidate.rows];
    driver
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
    domain(args.rows, max_seq)
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
        |choice| {
            opened.push(choice);
            Ok(Box::new(executor.take().expect("opened once")) as Box<dyn Executor>)
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
    let backend = ServedBackend::Decoder(Arc::clone(handle_model));
    let metrics = test_metrics("health");
    let limiter = RateLimiter {
        rpm: 0,
        windows: Mutex::new(Default::default()),
    };
    let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
        .expect("loopback policy");
    handle(server, None, health, &backend, &metrics, &limiter, &policy).expect("handled");
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

/// A target for a ROCm device whose per-dispatch work bound is below what the model's smallest
/// dispatch needs: the planner refuses it on that target, naming the op and the capability.
fn amd_target_missing_dispatch_work() -> Target {
    Target {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps {
            max_dispatch_work: Queried::Known(1),
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
    let (executor, add_entries) = probe(amd_target_missing_dispatch_work(), 0);
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
        matches!(refusal.target, Backend::AmdGcn(_)),
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
    let (executor, _) = probe(amd_target_missing_dispatch_work(), 0);
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
        let Workload::Generate { rows, max_seq } = started.plan.workload;
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
                driver_options(&handle, candidate.max_seq),
                &shapes_of(&candidate, &rows),
            )
            .expect("plans");
        let (executor, _) = probe(spirv_target(), 0);
        let mut driver = Driver::new(
            Arc::clone(&handle),
            Box::new(executor),
            driver_options(&handle, candidate.max_seq),
        )
        .unwrap();
        driver.prepare(&shapes_of(&candidate, &rows)).unwrap();
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
/// equality holds against the old numbers and not only against the code that replaced them.
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
    )
    .unwrap();
    let golden = [
        (
            4,
            64,
            "Activation=836616 Input=55224 Meta=868 State=12288 Weight=5344",
        ),
        (
            2,
            64,
            "Activation=418392 Input=36824 Meta=868 State=6144 Weight=5344",
        ),
        (
            1,
            64,
            "Activation=209312 Input=27624 Meta=1220 State=3072 Weight=5344",
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
                driver_options(&handle, candidate.max_seq),
                &shapes_of(candidate, &r),
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
        let mut driver = Driver::new(
            Arc::clone(&handle),
            Box::new(executor),
            driver_options(&handle, chosen.max_seq),
        )
        .expect("driver");
        let rows = [chosen.rows];
        driver
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
    let Workload::Generate { rows, max_seq } = started.plan.workload;
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
        loaded_kernels: Some(Arc::clone(&loaded)),
    });
    let mut driver = Driver::new(
        Arc::clone(handle),
        Box::new(executor),
        driver_options(handle, candidate.max_seq),
    )
    .expect("driver");
    let rows = [candidate.rows];
    driver
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
    let new_planner = || Planner::new(Arc::clone(&handle), &targets).unwrap();
    let plan = |planner: &Planner, candidate: &Candidate| {
        let rows = [candidate.rows];
        planner
            .plan(
                driver_options(&handle, candidate.max_seq),
                &shapes_of(candidate, &rows),
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
        let mut driver = Driver::new(
            Arc::clone(&handle),
            Box::new(reference),
            driver_options(&handle, domain[k].max_seq),
        )
        .expect("driver");
        let rows = [domain[k].rows];
        driver
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
    let Workload::Generate { rows, max_seq } = started.plan.workload;
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

/// SC-005: an encoder or cross-encoder prepares its entries on first use, so its allocations cannot be
/// accounted before `Serving`: startup refuses it with a typed variant before any device opens. Mutation:
/// let `Workload::of` plan them; the build opens a device and publishes `Serving`.
#[test]
fn encoder_models_are_refused_as_not_accounted_before_a_device_opens() {
    for (variant, label) in [
        (None, "hidden states"),
        (Some("cross_encoder"), "classifier scores"),
    ] {
        let handle = crate::tests::encoder_routes::encoder_handle(variant);
        let health = EngineHealth::starting();
        let (executor, _) = probe(spirv_target(), 0);
        let (result, opened) = build_with(
            &args(BackendChoice::Wgpu),
            &handle,
            executor,
            roomy,
            &health,
        );
        let Err(StartupError::NotAccounted { output }) = result else {
            panic!(
                "{label}: expected the typed refusal, got {:?}",
                result.err()
            );
        };
        assert!(output.contains(label), "{output}");
        assert!(opened.is_empty(), "no device was opened");
        assert_eq!(health.state(), EngineState::Starting);
    }
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
    target.caps.memory_budget_bytes = Some(u64::MAX / 2);
    // The independent oracle: what the first candidate allocates on a counting device.
    let (executor, _) = probe(target, 0);
    let mut oracle = Driver::new(
        Arc::clone(&handle),
        Box::new(executor),
        driver_options(&handle, first.max_seq),
    )
    .unwrap();
    oracle.prepare(&shapes_of(&first, &[first.rows])).unwrap();
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
            Planner::new(Arc::clone(&handle), &targets)
                .unwrap()
                .plan(driver_options(&handle, c.max_seq), &shapes_of(&c, &rows))
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
    target.caps.memory_budget_bytes = Some(u64::MAX / 2);
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
    let index = poot_load::gguf::GgufIndex::open(&path).expect("tensor directory");
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
    )
    .unwrap();
    let rows = [candidate.rows];
    let planned = planner
        .plan(
            driver_options(&handle, candidate.max_seq),
            &shapes_of(&candidate, &rows),
        )
        .expect("plans");
    let weight = |roles: Vec<(BufferRole, u64)>| {
        roles
            .into_iter()
            .find(|(r, _)| *r == BufferRole::Weight)
            .map_or(0, |(_, b)| b)
    };
    let (executor, _) = probe(spirv_target(), 0);
    let mut driver = Driver::new(
        Arc::clone(&handle),
        Box::new(executor),
        driver_options(&handle, candidate.max_seq),
    )
    .unwrap();
    driver.prepare(&shapes_of(&candidate, &rows)).unwrap();
    let live = weight(driver.live_role_bytes());
    assert!(live > 0, "the fixture carries weights");
    assert_eq!(
        weight(planned.footprint.role_bytes(DeviceId(0)).unwrap()),
        live,
        "planned vs live"
    );
    assert_eq!(live, oracle, "live vs the stored tensor sizes");
}
