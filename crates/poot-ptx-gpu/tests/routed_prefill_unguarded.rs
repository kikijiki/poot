//! POOT-1038 acceptance on PTX: the qwen38 routed-expert prefill runs on the live device UNGUARDED (the
//! `routed_expert_tiling_hang` fact cleared, so `flash_attention_capped`, `fuse` and `tile_matmuls` all run)
//! and completes, with the same logits as the guarded program and far fewer launches.
//!
//! The graph is the whole-model `trace_qwen38_prefill` (hyper-connections, gated delta net and QSA layers, a
//! routed-expert FFN with `arg_top_k` feeding `indexed_matmul`), at a small config; weights are deterministic
//! fills. Cards 186 and 192 recorded the hang this guard exists for; 186's root cause was ROCm HSA signal
//! churn (cured), and no PTX hang was ever observed. A hang is a red row, not a skip: a watchdog thread
//! aborts the process after [`DEADLINE`].
//!
//! Mutations (observed red): `let guarded = true` in `poot-graph-plan/src/compile.rs`'s `compile` fails the plan
//! row before any launch ("flash_attention_capped: the unguarded program runs it and the guarded one skips
//! it"); `DEADLINE` of 1 ms aborts the process with "HANG: guarded prefill did not complete within 1ms".

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use poot_executor::{
    Arg, BufferRole, Device, DeviceTime, Dispatch, Engine, Executor, MemoryCounterSnapshot, NoSync,
    StepInputs, WeightSource,
};
use poot_graph_ir::{Graph, Slot, Storage};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, KernelChoice, Partition,
    Submission, Target, TargetSet, compile, compile_staged,
};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_models::qwen38::{
    QsaConfig, Qwen4ExpConfig, Qwen4ExpGdnConfig, Qwen4ExpModelConfig, trace_qwen38_prefill,
};
use poot_ptx_gpu::PtxDevice;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::{CompiledKernel, DeviceBackend};
use poot_target::BufferStorage;
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_skip::open_or_skip;

/// A hang past this aborts the process: the row is red, never a skip.
const DEADLINE: Duration = Duration::from_secs(120);
const PROMPT: usize = 8;

/// The plan-summary qwen38 fixture's config: a mixed GDN then QSA schedule with a routed MoE FFN.
fn qwen38_config() -> Qwen4ExpModelConfig {
    Qwen4ExpModelConfig {
        cfg: Qwen4ExpConfig {
            hidden: 8,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            eps: 1e-5,
        },
        qcfg: QsaConfig {
            index_n_heads: 2,
            index_kv_heads: 1,
            index_head_dim: 3,
            index_budget: 4,
            index_compress_ratio: 2,
        },
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    }
}

/// Deterministic fill in `[-0.25, 0.25)`.
fn fill(seed: usize, count: usize) -> Vec<f32> {
    let mut s = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) % 1000) as f32 / 2000.0 - 0.25
        })
        .collect()
}

/// The live device, reporting the routed-expert hang only while `guarded` is set. The engine refuses a program
/// whose target differs from the device's, so an unguarded program can only run on a device that reports no hang:
/// this decorator is that device, with every other method the live device's own.
struct HangFact<D> {
    device: D,
    guarded: Rc<Cell<bool>>,
}

impl<D: Device> Device for HangFact<D> {
    type Buffer = D::Buffer;
    type Kernel = D::Kernel;
    type Recording = D::Recording;
    type Error = D::Error;

    fn target(&self) -> Target {
        let mut target = self.device.target();
        target.caps.known_miscompiles.routed_expert_tiling_hang = self.guarded.get();
        target
    }
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.device.memory()
    }
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.device.allocate(role, storage, elems)
    }
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.device.load_kernel(key, kernel)
    }
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error> {
        self.device.begin(submission)
    }
    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        let inputs: Vec<Arg<'_, D>> = dispatch
            .inputs
            .iter()
            .map(|arg| Arg {
                buffer: arg.buffer,
                elems: arg.elems,
            })
            .collect();
        self.device.dispatch(Dispatch {
            kernel: dispatch.kernel,
            inputs: &inputs,
            output: Arg {
                buffer: dispatch.output.buffer,
                elems: dispatch.output.elems,
            },
            threads: dispatch.threads,
            workgroup: dispatch.workgroup,
            work: dispatch.work,
        })
    }
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.device.copy(src, dst)
    }
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
        self.device.finish()
    }
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.device.replay(recording)
    }
    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error> {
        self.device.write(dst, bytes)
    }
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.device.read(src, out)
    }
    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.device.synchronize()
    }
    fn device_time(&self) -> DeviceTime {
        self.device.device_time()
    }
    fn abort(&mut self) -> Result<(), Self::Error> {
        self.device.abort()
    }
    fn classify_fault(&self, error: &Self::Error) -> Option<(String, u32)> {
        self.device.classify_fault(error)
    }
}

fn weights(graph: &Graph) -> Arc<WeightStore> {
    let mut builder = WeightStore::builder();
    for (index, meta) in graph.values.iter().enumerate() {
        if !matches!(meta.storage, Storage::Const) {
            continue;
        }
        let name = meta.name.as_deref().expect("a const is named");
        let shape = graph
            .aval(poot_graph_ir::ValueId::from(index))
            .shape
            .clone();
        let tensor = HostTensor::f32(shape.clone(), fill(index, shape.iter().product()));
        let dense =
            DenseWeight::try_new(DType::F32, shape, Arc::from(tensor.view().bytes())).unwrap();
        builder.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    Arc::new(builder.build())
}

/// Token ids, the causal mask (`0` visible, `-1e4` hidden), the QSA block-eligibility and tail masks.
fn step_inputs(graph: &Graph) -> Vec<poot_test_util::StepFixture> {
    graph
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, meta)| {
            let Storage::Slot(slot) = &meta.storage else {
                return None;
            };
            let shape = graph
                .aval(poot_graph_ir::ValueId::from(index))
                .shape
                .clone();
            let name = meta.name.clone().unwrap_or_default();
            let tensor = match slot {
                Slot::Token => HostTensor::i32(shape, (0..PROMPT as i32).map(|t| t % 6).collect()),
                Slot::Mask => {
                    HostTensor::f32(shape.clone(), causal(shape[2], shape[3], 0.0, -1.0e4))
                }
                Slot::Activation if name.contains("block_eligible") => HostTensor::f32(
                    shape.clone(),
                    // Block `c` (two tokens) is visible to the rows after it.
                    (0..shape[2])
                        .flat_map(|row| {
                            (0..shape[3]).map(move |block| f32::from(2 * block + 1 < row))
                        })
                        .collect(),
                ),
                Slot::Activation => {
                    HostTensor::f32(shape.clone(), causal(shape[2], shape[3], 0.0, -1.0e4))
                }
                other => panic!("unexpected slot {other:?}"),
            };
            Some(poot_test_util::StepFixture {
                key: meta.slot_key().expect("a slot has a key").clone(),
                tensor,
            })
        })
        .collect()
}

fn causal(rows: usize, cols: usize, visible: f32, hidden: f32) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| (0..cols).map(move |col| if col <= row { visible } else { hidden }))
        .collect()
}

/// What `compile` planned for the prefill on one target.
struct PlanFacts {
    launches: usize,
    passes: Vec<&'static str>,
    /// Contractions planned on a generated tiled schedule, and on the one-thread-per-output serial kernel.
    tiled: usize,
    serial: usize,
}

fn options() -> CompileOptions {
    CompileOptions {
        execution: Submission::Replay,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    }
}

fn plan(graph: &Graph, target: &Target) -> PlanFacts {
    let program = compile(graph, target, &options()).expect("the prefill compiles");
    let launches = program
        .planned()
        .map(|(_, plan)| plan.dispatches().len())
        .sum();
    let (mut tiled, mut serial) = (0, 0);
    for eqn in &program.graph().eqns {
        match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::Serial { .. } | ContractionSpec::DenseSerial { .. },
            )) => serial += 1,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::TiledRegion { .. }
                | ContractionSpec::DenseTiled { .. }
                | ContractionSpec::TensorCore { .. }
                | ContractionSpec::Coopmat { .. },
            )) => tiled += 1,
            _ => {}
        }
    }
    PlanFacts {
        launches,
        passes: program.passes().collect(),
        tiled,
        serial,
    }
}

struct Run {
    logits: HostTensor,
    recording: Duration,
    replay: Duration,
    engine_wall: Duration,
}

/// Loads, records and replays the prefill compiled for `target` on `engine` (the device reports its hang fact
/// through `guarded`).
fn run(
    engine: &mut Engine<HangFact<PtxDevice>>,
    guarded: &Cell<bool>,
    target: Target,
    graph: &Graph,
) -> Run {
    guarded.set(target.caps.known_miscompiles.routed_expert_tiling_hang);
    let options = options();
    let staged = compile_staged(
        &graph.clone().with_validations(Vec::new()),
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &options,
    )
    .expect("the prefill stages");
    let exe = engine
        .load_weights(weights(graph), WeightSource::ConstNames)
        .expect("weights load");
    let entry = engine.add_entry(exe, &staged).expect("the entry loads");
    let step = step_inputs(graph);
    let inputs = || {
        let mut inputs = StepInputs::new();
        for fixture in &step {
            inputs.push(
                fixture.key.clone(),
                fixture.tensor.shape(),
                fixture.tensor.view(),
            );
        }
        inputs
    };
    let mark = engine.stats().timing.mark();
    let started = Instant::now();
    let logits = engine
        .step(exe, entry, &inputs(), &mut NoSync)
        .expect("the recording step")
        .to_host()
        .expect("logits read back");
    let recording = started.elapsed();
    let started = Instant::now();
    let replayed = engine
        .step(exe, entry, &inputs(), &mut NoSync)
        .expect("the replay step")
        .to_host()
        .expect("replayed logits read back");
    let replay = started.elapsed();
    // The replay re-prefills over the state the recording left, so its logits differ; it must still be finite.
    assert!(
        replayed.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "replayed logits {:?}",
        replayed.as_f32().unwrap()
    );
    let timing = engine.stats().timing;
    let engine_wall = timing
        .entries_with_activity_since(&mark)
        .into_iter()
        .map(|id| timing.cumulative(id).host.wall)
        .sum();
    engine
        .remove_entry(exe, entry)
        .expect("the entry is removed");
    engine.unload(exe).expect("the executable unloads");
    Run {
        logits,
        recording,
        replay,
        engine_wall,
    }
}

/// Runs `work` with a watchdog: past [`DEADLINE`] the process aborts, so a hang is red. A panic in `work`
/// disarms the watchdog too (its message is the failure).
fn within_deadline<T>(what: &str, work: impl FnOnce() -> T) -> T {
    struct Disarm(mpsc::Sender<()>);
    impl Drop for Disarm {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (done, finished) = mpsc::channel::<()>();
    let what = what.to_string();
    let watchdog = std::thread::spawn(move || {
        if finished.recv_timeout(DEADLINE) == Err(mpsc::RecvTimeoutError::Timeout) {
            eprintln!("HANG: {what} did not complete within {DEADLINE:?}");
            std::process::abort();
        }
    });
    let disarm = Disarm(done);
    let out = work();
    drop(disarm);
    watchdog.join().unwrap();
    out
}

/// The unguarded program runs the passes the guard skips and plans tiled contractions, launches fewer
/// kernels, completes within the deadline and matches the guarded program's logits.
#[test]
fn routed_prefill_unguarded_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let guarded_target = device.target();
    assert!(
        guarded_target
            .caps
            .known_miscompiles
            .routed_expert_tiling_hang,
        "the live PTX device reports the routed-expert hang"
    );
    let mut unguarded_target = guarded_target;
    unguarded_target
        .caps
        .known_miscompiles
        .routed_expert_tiling_hang = false;
    let graph = trace_qwen38_prefill(&qwen38_config(), PROMPT);
    // The plan rows run before anything executes: a plan that does not differ is a red row without a launch.
    let (guarded_plan, unguarded_plan) = (
        plan(&graph, &guarded_target),
        plan(&graph, &unguarded_target),
    );
    for pass in ["flash_attention_capped", "fuse"] {
        assert!(
            unguarded_plan.passes.contains(&pass) && !guarded_plan.passes.contains(&pass),
            "{pass}: the unguarded program runs it and the guarded one skips it"
        );
    }
    assert_eq!(
        guarded_plan.tiled, 0,
        "the guard keeps every contraction off the generated tiled GEMM"
    );
    assert!(
        unguarded_plan.tiled > 0,
        "the unguarded program plans a generated tiled contraction"
    );
    assert!(
        unguarded_plan.launches < guarded_plan.launches,
        "unguarded {} launches, guarded {}",
        unguarded_plan.launches,
        guarded_plan.launches
    );

    let guarded_flag = Rc::new(Cell::new(true));
    let mut engine = Engine::new(HangFact {
        device,
        guarded: Rc::clone(&guarded_flag),
    });
    let guarded = within_deadline("guarded prefill", || {
        run(&mut engine, &guarded_flag, guarded_target, &graph)
    });
    let unguarded = within_deadline("unguarded prefill", || {
        run(&mut engine, &guarded_flag, unguarded_target, &graph)
    });
    for (name, plan, run) in [
        ("guarded", &guarded_plan, &guarded),
        ("unguarded", &unguarded_plan, &unguarded),
    ] {
        eprintln!(
            "POOT-1038 ptx {name}: launches={} tiled={} serial={} recording={:?} replay={:?} engine_wall={:?}",
            plan.launches, plan.tiled, plan.serial, run.recording, run.replay, run.engine_wall
        );
    }
    let (got, want) = (
        unguarded.logits.as_f32().unwrap(),
        guarded.logits.as_f32().unwrap(),
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "unguarded logits {got:?}"
    );
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
            "logit {index}: unguarded {g}, guarded {w}"
        );
    }
}
