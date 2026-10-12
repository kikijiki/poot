//! POOT-1038 acceptance on ROCm, kept as the standing proof that a routed-expert prefill needs no guard: the
//! qwen38 routed-expert prefill compiles through the full pipeline (`flash_attention_capped` and `fuse` run, the
//! router contraction plans a generated tiled schedule) and completes on the live device, with the CPU
//! oracle's logits.
//!
//! The graph is the registered Qwen4Exp fixture's prefill through `Model::trace` (hyper-connections, gated delta
//! net and QSA layers, a routed-expert FFN with `arg_top_k` feeding `indexed_matmul`), over the fixture's own
//! weights bound through its weight map. Cards 186 and 192 recorded the hang a guard was added for; 186's root
//! cause was ROCm HSA signal churn (cured), and no PTX hang was ever observed. A hang is a red row, not a skip:
//! a watchdog thread aborts the process after [`DEADLINE`].
//!
//! Mutations (observed red): skipping `flash_attention_capped` in `poot-graph-plan/src/compile.rs`'s `compile`
//! fails the plan row before any launch ("flash_attention_capped: the program runs it"); `DEADLINE` of 1 ms
//! aborts the process with "HANG: routed prefill did not complete within 1ms".

#![cfg(feature = "rocm")]

use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use poot_executor::{Device, Engine, Executor, NoSync};
use poot_graph_ir::Graph;
use poot_graph_plan::{KernelChoice, Target};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_models::model::{Model, Phase};
use poot_models::registry::Registry;
use poot_models::test_support::weight_map::{self, Step};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::HostTensor;

/// A hang past this aborts the process: the row is red, never a skip.
const DEADLINE: Duration = Duration::from_secs(120);
/// The prompt is a whole number of the fixture's blocks of two, and fills its cache.
const PROMPT: usize = weight_map::CAPACITY;

/// The registered Qwen4Exp fixture: its model, its store and its weight map.
struct Fixture {
    model: Box<dyn Model>,
    store: Arc<WeightStore>,
    map: Arc<WeightMap>,
}

fn fixture() -> Fixture {
    let registry = Registry::builtin().expect("the builtin registry");
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "qwen4exp")
        .expect("qwen4exp is registered");
    let fixture = (entry.fixture)();
    let model = (entry.build)(&fixture.raw(), &fixture.store).expect("the fixture builds");
    let map = Arc::new(model.weights().clone());
    Fixture {
        model,
        store: Arc::new(fixture.store),
        map,
    }
}

/// The prefill of the whole prompt: tokens inside the vocabulary and away from its end-of-sequence id.
fn prompt() -> Step {
    let ids: Vec<i32> = (0..PROMPT as i32).map(|t| (3 * t + 1) % 40).collect();
    Step::new(Phase::Prefill, &ids, 0)
}

fn is_tiled(choice: &KernelChoice) -> bool {
    matches!(
        choice,
        KernelChoice::Generated(KernelRequest::Contraction(
            ContractionSpec::TiledRegion { .. }
                | ContractionSpec::DenseTiled { .. }
                | ContractionSpec::TensorCore { .. }
                | ContractionSpec::Coopmat { .. },
        ))
    )
}

/// What `compile` planned for the prefill.
struct PlanFacts {
    passes: Vec<&'static str>,
    /// Contractions planned on a generated tiled schedule.
    tiled: usize,
    /// Router-logits contractions (`[PROMPT, experts]`) planned on a generated tiled schedule.
    tiled_routers: usize,
}

fn plan(graph: &Graph, target: &Target, experts: usize) -> PlanFacts {
    let program = poot_graph_plan::compile(
        graph,
        target,
        &poot_graph_plan::CompileOptions::new(
            poot_graph_plan::production_plan::Submission::Replay,
            poot_graph_plan::CompileLimits::STANDARD,
        ),
    )
    .expect("the prefill compiles");
    let router_shape = [PROMPT, experts];
    let (mut tiled, mut tiled_routers) = (0, 0);
    for eqn in program.graph().eqns() {
        if is_tiled(program.kernel_choice(eqn)) {
            tiled += 1;
            if program.graph().aval(eqn.out).shape == router_shape {
                tiled_routers += 1;
            }
        }
    }
    PlanFacts {
        passes: program.passes().collect(),
        tiled,
        tiled_routers,
    }
}

/// The CPU oracle's logits for `graph`: the fixture's weights through its map, every slot its step
/// input, every state zero (at its own dtype).
fn oracle_logits(fixture: &Fixture, graph: &Graph, step: &Step) -> HostTensor {
    let zeros: Vec<HostTensor> = graph
        .state()
        .map(|(state, _)| {
            let aval = graph.aval(state);
            match aval.dtype {
                poot_tensor::DType::I32 => {
                    HostTensor::i32(aval.shape.clone(), vec![0; aval.numel()])
                }
                _ => HostTensor::zeros(aval.shape.clone()),
            }
        })
        .collect();
    poot_eval::test_support::weight_map_oracle::eval_mapped(
        graph,
        &fixture.store,
        &fixture.map,
        &step.slots(),
        &zeros,
    )
    .0
}

struct Run {
    logits: HostTensor,
    recording: Duration,
    replay: Duration,
    engine_wall: Duration,
}

/// Loads, records and replays the prefill compiled for `target` on `engine`.
fn run(
    engine: &mut Engine<RocmDevice>,
    target: Target,
    fixture: &Fixture,
    graph: &Graph,
    step: &Step,
) -> Run {
    let staged = weight_map::compile(graph, target);
    let exe = engine
        .load_weights(Arc::clone(&fixture.store), Arc::clone(&fixture.map))
        .expect("weights load");
    let entry = engine.add_entry(exe, &staged).expect("the entry loads");
    let mark = engine.stats().timing.mark();
    let started = Instant::now();
    let logits = engine
        .step(exe, entry, &step.inputs(), &mut NoSync)
        .expect("the recording step")
        .to_host()
        .expect("logits read back");
    let recording = started.elapsed();
    let started = Instant::now();
    let replayed = engine
        .step(exe, entry, &step.inputs(), &mut NoSync)
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

/// The prefill runs every pass and plans tiled contractions (the router's among them), completes within the
/// deadline and matches the CPU oracle.
#[test]
fn routed_prefill_unguarded_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let target = device.target();
    let fixture = fixture();
    let step = prompt();
    let graph = weight_map::trace(
        &*fixture.model,
        &fixture.map,
        Phase::Prefill,
        PROMPT,
        weight_map::CAPACITY,
    );
    // The plan rows run before anything executes: a plan without the passes is a red row without a launch.
    let facts = plan(&graph, &target, 4);
    for pass in ["flash_attention_capped", "fuse"] {
        assert!(facts.passes.contains(&pass), "{pass}: the program runs it");
    }
    assert!(
        facts.tiled_routers > 0,
        "the router-logits contraction plans a generated tiled schedule"
    );

    let mut engine = Engine::new(device);
    let result = within_deadline("routed prefill", || {
        run(&mut engine, target, &fixture, &graph, &step)
    });
    eprintln!(
        "POOT-1038 rocm: tiled={} routers={} recording={:?} replay={:?} engine_wall={:?}",
        facts.tiled, facts.tiled_routers, result.recording, result.replay, result.engine_wall
    );
    let (got, want) = (
        result.logits.as_f32().unwrap(),
        oracle_logits(&fixture, &graph, &step)
            .as_f32()
            .unwrap()
            .to_vec(),
    );
    assert!(got.iter().all(|v| v.is_finite()), "logits {got:?}");
    assert_eq!(got.len(), want.len(), "logits length");
    for (index, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
            "logit {index}: device {g}, oracle {w}"
        );
    }
}
