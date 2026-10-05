//! Card 552 acceptance fixtures (SC-001, SC-002, SC-004, SC-005, SC-006): a fake [`Device`] whose
//! `device_time()` and `synchronize()` wall cost are fully controlled by the test, driven through
//! the real production `Engine::step()` (never a test-only accessor). SC-003 is
//! proved at `poot-graph-ir::OpKind::kind_name` and `poot-profile::Report::window`; SC-007 at
//! `poot-runtime-common::telemetry`.
//!
//! Model-free: no GPU adapter. Real SPIR-V codegen still runs (`nix develop`'s `llc`) to compile the
//! fixture graph's kernels, which the fake device never actually executes (same pattern as
//! `tests/cr30_lifecycle.rs`'s `FaultyDevice`).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::device::{DeviceTime, DispatchDeviceTime, MeasuredDeviceTime};
use crate::timing::{DetailedTiming, TimingOptions};
use crate::{BufferRole, Device, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::{BinOp, Builder, StateRole, TensorType, UnOp};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::weights::WeightStore;
use poot_target::{Backend, DeviceCaps};

#[derive(Default)]
struct ControlInner {
    /// Injected delay while `synchronize()` runs: counted by the engine as `wait`.
    sync_delay: Duration,
    /// What `device_time()` returns on the next call.
    device_time: DeviceTime,
    synchronize_calls: u32,
}

/// A handle the test keeps to program the fake device's timing after it has moved into an `Engine`.
#[derive(Clone)]
struct Control(Rc<RefCell<ControlInner>>);

impl Control {
    fn new() -> Self {
        Self(Rc::new(RefCell::new(ControlInner {
            device_time: DeviceTime::Unknown,
            ..Default::default()
        })))
    }
    fn set_sync_delay(&self, d: Duration) {
        self.0.borrow_mut().sync_delay = d;
    }
    fn set_device_time(&self, t: DeviceTime) {
        self.0.borrow_mut().device_time = t;
    }
    fn synchronize_calls(&self) -> u32 {
        self.0.borrow().synchronize_calls
    }
}

struct FakeDevice {
    control: Control,
    submission: Option<Submission>,
    memory: poot_runtime_common::MemoryCounters,
}

#[derive(Debug, thiserror::Error)]
#[error("fake device error (never injected in this fixture)")]
struct FakeError;

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

    fn target(&self) -> Target {
        Target {
            backend: Backend::SpirvVulkan,
            caps: DeviceCaps::wgpu_rdna3_igpu(),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: poot_target::BufferStorage,
        elems: usize,
    ) -> Result<(), FakeError> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        Ok(())
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }

    fn begin(&mut self, submission: Submission) -> Result<(), FakeError> {
        self.submission = Some(submission);
        Ok(())
    }

    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }

    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<()>, FakeError> {
        Ok(match self.submission.take() {
            Some(Submission::Replay) => Some(()),
            _ => None,
        })
    }

    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }

    fn write(&mut self, _dst: &(), _bytes: &[u8]) -> Result<(), FakeError> {
        Ok(())
    }

    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        out.fill(0);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), FakeError> {
        let mut c = self.control.0.borrow_mut();
        c.synchronize_calls += 1;
        let delay = c.sync_delay;
        drop(c);
        if delay > Duration::ZERO {
            std::thread::sleep(delay);
        }
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        self.control.0.borrow().device_time.clone()
    }

    fn abort(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
}

/// Two dispatches of two different `OpKind`s (`Binary`/`Unary`), feeding a state commit so the
/// recorded/replayed step has real work. `output = neg(a + bias)`; `new_a = a + bias` (commit).
fn fixture() -> (
    poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    Arc<WeightStore>,
) {
    use poot_quant::weights::{DenseWeight, WeightEntry};
    use poot_tensor::DType;

    let b = Builder::new();
    let a = b.state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent);
    let bias = b.constant("bias", TensorType::f32(vec![2]));
    let sum = b.binary(BinOp::Add, a, bias);
    let output = b.unary(UnOp::Neg, sum);
    let g = b
        .finish_with_state(output, &[(a, sum)])
        .with_validations(Vec::new());

    let mut builder = WeightStore::builder();
    builder
        .insert(
            "bias",
            WeightEntry::Dense(
                DenseWeight::try_new(
                    DType::F32,
                    vec![2],
                    Arc::from([0u8, 0, 0x80, 0x3f, 0, 0, 0, 0x40]), // [1.0, 2.0]
                )
                .unwrap(),
            ),
        )
        .unwrap();
    (g, Arc::new(builder.build()))
}

fn staged(
    g: &poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    target: Target,
) -> StagedProgram<poot_graph_ir::ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
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
    .expect("fixture graph compiles for the fake target")
}

fn new_engine(options: TimingOptions) -> (Engine<FakeDevice>, Control) {
    let control = Control::new();
    let device = FakeDevice {
        control: control.clone(),
        submission: None,
        memory: poot_runtime_common::MemoryCounters::new(),
    };
    (Engine::with_timing(device, options), control)
}

fn load(engine: &mut Engine<FakeDevice>) -> (crate::ExecutableId, crate::EntryId) {
    load_with(engine, fixture())
}

fn load_with(
    engine: &mut Engine<FakeDevice>,
    (g, store): (
        poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
        Arc<WeightStore>,
    ),
) -> (crate::ExecutableId, crate::EntryId) {
    let target = engine.device().target();
    let program = staged(&g, target);
    let exe = engine
        .load_weights(store, crate::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &program).unwrap();
    (exe, entry)
}

/// Two `Binary::Add` dispatches at different shapes (so the planner gives them different kernel
/// keys) and one `Unary::Neg` - the real-`Engine` SC-003 pair (review F6): one `OpKind` ("binary")
/// spans two kernel keys and must still land in one `kind_name` bucket; the different `OpKind`
/// ("unary") never shares that bucket however the planner keys its own kernel.
fn fixture_shared_kind_different_keys() -> (
    poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    Arc<WeightStore>,
) {
    use poot_quant::weights::{DenseWeight, WeightEntry};
    use poot_tensor::DType;

    let b = Builder::new();
    let a = b.state_input("a", TensorType::f32(vec![2]), StateRole::Recurrent);
    let bias2 = b.constant("bias2", TensorType::f32(vec![2]));
    let sum_a = b.binary(BinOp::Add, a, bias2); // Add #1, shape [2]
    let c = b.state_input("c", TensorType::f32(vec![3]), StateRole::Recurrent);
    let bias3 = b.constant("bias3", TensorType::f32(vec![3]));
    let sum_c = b.binary(BinOp::Add, c, bias3); // Add #2, shape [3] -> a different kernel key
    let output = b.unary(UnOp::Neg, sum_a);
    let g = b
        .finish_with_state(output, &[(a, sum_a), (c, sum_c)])
        .with_validations(Vec::new());

    let mut builder = WeightStore::builder();
    builder
        .insert(
            "bias2",
            WeightEntry::Dense(
                DenseWeight::try_new(
                    DType::F32,
                    vec![2],
                    Arc::from([0u8, 0, 0x80, 0x3f, 0, 0, 0, 0x40]), // [1.0, 2.0]
                )
                .unwrap(),
            ),
        )
        .unwrap();
    builder
        .insert(
            "bias3",
            WeightEntry::Dense(
                DenseWeight::try_new(
                    DType::F32,
                    vec![3],
                    Arc::from([
                        0u8, 0, 0x80, 0x3f, // 1.0
                        0, 0, 0, 0x40, // 2.0
                        0, 0, 0x40, 0x40, // 3.0
                    ]),
                )
                .unwrap(),
            ),
        )
        .unwrap();
    (g, Arc::new(builder.build()))
}

fn measured(sum: Duration, span: Duration, dispatches: Vec<DispatchDeviceTime>) -> DeviceTime {
    DeviceTime::Measured(MeasuredDeviceTime {
        sum_of_dispatch_durations: Some(sum),
        device_span: Some(span),
        dispatches,
    })
}

/// SC-001: host `encode + wait + other == wall` exactly, and a nonzero measured device time (bigger
/// than the host wait, as device-side concurrency would show) is reported alongside the host
/// partition, never subtracted into `other` or folded into `wait`.
///
/// MUTATION (recorded, not left in the tree): in `Engine::record_step_timing`, change
/// `wall` to `wall.saturating_sub(sum_of_dispatch_durations)` before building `HostTiming` - the
/// window's `other_nanos()` goes deeply negative/the wall shrinks below the measured `wait`, and
/// this test's `wall >= wait` assertion goes red. Restoring the unmodified `wall` makes it green
/// again.
#[test]
fn sc001_host_partition_sums_to_wall_without_device_subtraction() {
    let (mut engine, control) = new_engine(TimingOptions::CountersOnly);
    let (exe, entry) = load(&mut engine);
    control.set_sync_delay(Duration::from_millis(5));
    // A device time deliberately larger than the host wait: on real concurrent hardware the device
    // can be busy for longer than the host blocks, and that must never read as impossible/clamped.
    control.set_device_time(measured(
        Duration::from_millis(50),
        Duration::from_millis(9),
        Vec::new(),
    ));

    let mark = engine.stats().timing.mark();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);

    assert_eq!(report.steps, 1);
    assert!(
        report.host.wait >= Duration::from_millis(5),
        "measured wait must include the injected synchronize delay: {:?}",
        report.host.wait
    );
    assert!(
        report.host.wall >= report.host.wait,
        "wall must never read smaller than the wait it contains: wall={:?} wait={:?}",
        report.host.wall,
        report.host.wait
    );
    // encode + wait + other == wall, exactly (other is signed and computed, never clamped).
    let reconstructed = report.host.encode.as_nanos() as i64
        + report.host.wait.as_nanos() as i64
        + report.other_nanos();
    assert_eq!(reconstructed, report.host.wall.as_nanos() as i64);
    // The device sum is reported in full, independent of (and here, larger than) the host wait.
    assert_eq!(
        report.device.sum_of_dispatch_durations,
        Some(Duration::from_millis(50))
    );
}

/// SC-002: a mark taken after a warmup step, windowed to the one entry, counts exactly the steps
/// recorded after it.
///
/// MUTATION (recorded, not left in the tree): in `poot_profile::EntryCounters::since`, return
/// `self.clone()` unconditionally (ignoring `earlier`) instead of the per-field delta - the window
/// reports the raw cumulative total (3 steps: 1 warmup + 2 timed) instead of the delta since the
/// mark (2), and this test's `assert_eq!(report.steps, 2)` goes red with `left: 3, right: 2`.
/// Restoring the delta makes it green again.
#[test]
fn sc002_window_excludes_steps_before_the_mark() {
    let (mut engine, _control) = new_engine(TimingOptions::CountersOnly);
    let (exe, entry) = load(&mut engine);

    // Warmup: one untimed step, before the mark.
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let mark = engine.stats().timing.mark();
    // Two timed steps, after the mark.
    for _ in 0..2 {
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
    }
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);
    assert_eq!(report.steps, 2, "warmup step must be excluded by the mark");
}

/// SC-004: a device returning `DeviceTime::Unknown` renders as no device sum/bucket, never a
/// fabricated zero.
///
/// MUTATION (recorded, not left in the tree): in `poot_profile::EntryCounters::accumulate_step`,
/// fold every step (`Unknown` included) into `device_sum` as zero instead of only `Measured` steps.
/// Result: `report.device.sum_of_dispatch_durations` becomes `Some(0ns)` instead of `None`, and
/// this test's `assert_eq!(report.device.sum_of_dispatch_durations, None)` goes red with
/// `left: Some(0ns), right: None`. Restoring the `Measured`-only guard makes it green again.
#[test]
fn sc004_unknown_device_time_never_reports_a_fabricated_zero() {
    let (mut engine, control) = new_engine(TimingOptions::CountersOnly);
    let (exe, entry) = load(&mut engine);
    control.set_device_time(DeviceTime::Unknown);

    let mark = engine.stats().timing.mark();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);
    assert_eq!(report.steps, 1);
    assert_eq!(report.device.sum_of_dispatch_durations, None);
    assert!(report.buckets.is_empty());
}

/// SC-005: timing on (detailed per-dispatch collection) versus timing off (counters-only) never
/// changes how many times the fake device's `synchronize()` is called - the one place this fixture
/// can observe an "undisclosed extra synchronization" a perturbing collection path would add - and a
/// measured step with partial per-dispatch coverage (one dispatch sampled, one not) keeps its real
/// dispatch indices and clock-domain identity rather than collapsing or shifting them.
///
/// MUTATION (recorded): add a second `self.device.synchronize()` call in `Engine::step` guarded by
/// `matches!(self.timing_options, TimingOptions::Detailed(_))` (simulating a timing-only extra
/// sync) - `sync_calls_detailed` becomes `2 * sync_calls_counters_only` instead of equal, and this
/// test's equality assertion goes red.
#[test]
fn sc005_detailed_timing_adds_no_extra_synchronization_and_keeps_dispatch_identity() {
    let (mut engine_counters, control_counters) = new_engine(TimingOptions::CountersOnly);
    let (exe_c, entry_c) = load(&mut engine_counters);
    control_counters.set_device_time(DeviceTime::Unknown);
    engine_counters
        .step(exe_c, entry_c, &StepInputs::new(), &mut NoSync)
        .unwrap();

    let (mut engine_detailed, control_detailed) = new_engine(TimingOptions::Detailed(
        DetailedTiming::every_step(16, 64, 4),
    ));
    let (exe_d, entry_d) = load(&mut engine_detailed);
    // Partial per-dispatch coverage: dispatch 1 (the `Unary::Neg`) sampled, dispatch 0 is not -
    // its real index must survive, not shift down to 0.
    control_detailed.set_device_time(measured(
        Duration::from_micros(40),
        Duration::from_micros(55),
        vec![DispatchDeviceTime {
            index: 1,
            duration: Duration::from_micros(40),
        }],
    ));
    let mark = engine_detailed.stats().timing.mark();
    engine_detailed
        .step(exe_d, entry_d, &StepInputs::new(), &mut NoSync)
        .unwrap();

    assert_eq!(
        control_counters.synchronize_calls(),
        control_detailed.synchronize_calls(),
        "collecting detailed device timing must not add an extra synchronization"
    );

    let stats = engine_detailed.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry_d.raw()]);
    assert_eq!(report.detail, poot_runtime_common::Coverage::Complete);
    // The sampled dispatch keeps its real index (1) and the op's own kind_name (`unary`), never
    // renumbered to look like a complete pair.
    let unary = report
        .buckets
        .get("unary")
        .expect("the sampled dispatch's kind_name bucket must be present");
    assert_eq!(unary.dispatch_count, 1);
    assert!(
        !report.buckets.contains_key("binary"),
        "the unsampled dispatch must not appear as a fabricated binary record"
    );
}

/// SC-006: a long run with a small retention bound evicts the oldest detail first, exposes the
/// eviction as `Coverage::Truncated` (not a silent `Complete`), and never grows the retained ring
/// past its configured bound regardless of how many steps run.
///
/// MUTATION (recorded, not left in the tree): in `poot_profile::TimingSnapshot::record`, disable
/// the `while` eviction loop's condition (`while false && (..)`) - every step keeps `Coverage::
/// Complete` instead of truncating once the bound is exceeded, and this test's
/// `assert!(matches!(.., Coverage::Truncated { .. }))` goes red (reporting `Complete`). Restoring
/// the loop condition makes it green again.
#[test]
fn sc006_bounded_retention_evicts_oldest_and_reports_truncated_coverage() {
    let (mut engine, control) = new_engine(TimingOptions::Detailed(DetailedTiming::every_step(
        3, 64, 4,
    )));
    let (exe, entry) = load(&mut engine);
    control.set_device_time(measured(
        Duration::from_micros(10),
        Duration::from_micros(10),
        vec![
            DispatchDeviceTime {
                index: 0,
                duration: Duration::from_micros(4),
            },
            DispatchDeviceTime {
                index: 1,
                duration: Duration::from_micros(4),
            },
        ],
    ));

    let mark = engine.stats().timing.mark();
    for _ in 0..10 {
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
    }
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);

    // Exact cumulative counts survive eviction regardless of the small detail bound.
    assert_eq!(report.steps, 10);
    assert_eq!(
        report.device.sum_of_dispatch_durations,
        Some(Duration::from_micros(100))
    );
    // Detail itself is honestly reported as incomplete, never silently "Complete" over a partial
    // ring.
    assert!(
        matches!(
            report.detail,
            poot_runtime_common::Coverage::Truncated { .. }
        ),
        "{:?}",
        report.detail
    );
}

/// A slow/absent reader of a replay's device timestamps (represented here by the engine never
/// querying detail more often than once per step, regardless of how long a caller waits between
/// steps) never blocks execution: the fake device's own `synchronize()` always returns promptly,
/// independent of how many steps have already run or how large the retained ring's bound is.
#[test]
fn absent_consumer_never_blocks_execution() {
    let (mut engine, control) =
        new_engine(TimingOptions::Detailed(DetailedTiming::every_step(1, 4, 1)));
    let (exe, entry) = load(&mut engine);
    control.set_device_time(measured(
        Duration::from_micros(1),
        Duration::from_micros(1),
        vec![DispatchDeviceTime {
            index: 0,
            duration: Duration::from_micros(1),
        }],
    ));
    let t0 = Instant::now();
    for _ in 0..50 {
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
    }
    assert!(
        t0.elapsed() < Duration::from_secs(1),
        "a bounded ring with no reader must not stall execution"
    );
}

/// Card 552: `sample_every` is a real, implemented policy - of every 3 steps, exactly 1
/// keeps its measured per-dispatch detail; the other 2 report `DeviceCoverage::Unknown` for
/// themselves (even though the fake device returned `Measured` for every step), so the window's
/// `Coverage::Unsampled` ratio is exact, and the exact cumulative step count is unaffected by
/// sampling.
///
/// MUTATION (recorded here, not left in the tree; Card 552): in `Engine::
/// consume_sample_decision`, change `n % u64::from(d.sample_every.max(1)) == 0` to `true` (sample
/// every step, ignoring the declared policy - the pre-fix "dead config" bug). Result: RED - this
/// test's `assert_eq!(sampled_steps, 2)` fails with `left: 6, right: 2` (every step kept its
/// detail). Reverted: GREEN.
#[test]
fn sample_every_skips_detail_on_the_declared_ratio_of_steps() {
    let (mut engine, control) = new_engine(TimingOptions::Detailed(DetailedTiming {
        sample_every: 3,
        max_retained_steps: 16,
        max_retained_dispatch_records: 64,
        max_in_flight_queries: 4,
        drain_policy: crate::DrainPolicy::WaitForCapacity,
    }));
    let (exe, entry) = load(&mut engine);
    // The fake device reports Measured on *every* step; sampling must still suppress detail on
    // the steps the policy skips.
    control.set_device_time(measured(
        Duration::from_micros(10),
        Duration::from_micros(10),
        vec![
            DispatchDeviceTime {
                index: 0,
                duration: Duration::from_micros(4),
            },
            DispatchDeviceTime {
                index: 1,
                duration: Duration::from_micros(4),
            },
        ],
    ));

    let mark = engine.stats().timing.mark();
    for _ in 0..6 {
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .unwrap();
    }
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);

    assert_eq!(
        report.steps, 6,
        "sampling never changes the exact step count"
    );
    let sampled_steps = report
        .buckets
        .values()
        .map(|b| b.dispatch_count)
        .sum::<u64>()
        / 2; // 2 dispatches per sampled step
    assert_eq!(
        sampled_steps, 2,
        "steps 0 and 3 of 6 are sampled at sample_every=3"
    );
    assert_eq!(
        report.detail,
        poot_runtime_common::Coverage::Unsampled {
            sampled: 2,
            total: 6
        }
    );
}

/// Card 552 / SC-003, driven through the real `Engine`: two `Binary::Add` dispatches at
/// different shapes (the planner gives them two different kernel keys) still land in the one
/// "binary" `kind_name` bucket, and the `Unary::Neg` dispatch (a different `OpKind`) never shares
/// it however the planner happens to key its kernel. This is the same claim
/// `poot-profile::timing::tests::buckets_key_on_kind_name_not_plan_label` proves by forging
/// `DispatchTiming` directly; this test proves it from the production `Engine::load_entry`/
/// `record_step_timing` path instead.
///
/// MUTATION (recorded here, not left in the tree; Card 552): in
/// `OpKind::kind_name`, change `Self::Binary(_) => "binary"` to `Self::Binary(_) => "unary"`.
/// Result: RED - "two OpKinds (binary, unary) must give exactly two buckets:
/// {"unary": OpBucket { dispatch_count: 3, .. }}" (`left: 1, right: 2`). Reverted: GREEN.
#[test]
fn sc003_one_kind_name_spans_two_kernel_keys_through_a_real_engine_step() {
    let (mut engine, control) = new_engine(TimingOptions::Detailed(DetailedTiming::every_step(
        4, 64, 4,
    )));
    let (exe, entry) = load_with(&mut engine, fixture_shared_kind_different_keys());
    // 3 real dispatches in this fixture: Add[2], Add[3], Neg[2], in build order.
    control.set_device_time(measured(
        Duration::from_micros(30),
        Duration::from_micros(30),
        vec![
            DispatchDeviceTime {
                index: 0,
                duration: Duration::from_micros(10),
            },
            DispatchDeviceTime {
                index: 1,
                duration: Duration::from_micros(10),
            },
            DispatchDeviceTime {
                index: 2,
                duration: Duration::from_micros(10),
            },
        ],
    ));

    let mark = engine.stats().timing.mark();
    engine
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let stats = engine.stats();
    let report = poot_profile::Report::window(&stats.timing, &mark, &[entry.raw()]);

    assert_eq!(
        report.buckets.len(),
        2,
        "two OpKinds (binary, unary) must give exactly two buckets: {:?}",
        report.buckets
    );
    let binary = report
        .buckets
        .get("binary")
        .expect("both Add dispatches share the binary bucket");
    assert_eq!(
        binary.dispatch_count, 2,
        "the two Add dispatches merge into one bucket despite their different kernel keys"
    );
    assert_eq!(
        report.buckets.get("unary").map(|b| b.dispatch_count),
        Some(1)
    );
}
