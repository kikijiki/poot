//! POOT-1278: a real model's wide step runs on the Intel NPU, two consecutive chunks of one prompt.
//!
//! The model is Qwen2.5-0.5B-Instruct from its official F16 GGUF (`qwen2.5-0.5b-instruct-fp16.gguf`: every matrix
//! F16, every vector F32). F16 binds as stored on the NPU, so nothing is re-encoded. The shipped Qwen2 tracer's
//! step of 128 tokens over a 256-position cache compiles through the production route (planner under the
//! device's `NarrowingContract`, one whole-graph region, `emit_vpu`, the vendor graph compiler, Level Zero) and
//! runs twice: positions 0..127, then 128..255 over the KV the first chunk wrote.
//!
//! Each chunk is compared with the CPU oracle evaluating the same step in F32 over the same prompt: every element
//! of the 48 KV states and of the last position's logits, at tier 2 (ADR-0101: a stated tolerance, a NaN or
//! infinity where the oracle is finite fails). The F32 evaluation is the reference the narrowing contract bounds
//! (`NarrowingContract`: the narrowed region against "the same region evaluated in F32"). The CPU evaluation of
//! the *narrowed* graph is not the comparison: it rounds every intermediate to F16 where the device's fused
//! kernels do not (the receipt of the first run is in the card's outcome), so it is a harsher model than the part.
//!
//! The KV states are read back through a second entry of the same executable that shares them by (name, aval,
//! storage) and returns them concatenated: the engine reads only an entry's own output.
//!
//! ```text
//! POOT_MODELS_DIR=$HOME/models nix develop .#npu-intel -c cargo nextest run -p poot-npu-rows --test vpu_qwen_prompt --test-threads=1 --no-capture
//! ```

use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::{Builder, Graph};
use poot_graph_plan::{CompileLimits, WeightFormats, bind_packed_weights, staged::StagedProgram};
use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf};
use poot_models::model::{KvDtype, StepShape, StepSpec};
use poot_models::registry::{RawConfig, Registry};
use poot_models::test_support::weight_map::{Step, compile};
use poot_npu::{VpuConfig, VpuDevice};
use poot_quant::StoredBytes;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::{Queried, StepWidths, VendorCompilerId};
use poot_tensor::{DType, HostTensor};

/// Tokens per chunk: the one step width the device is configured for.
const WIDTH: usize = 128;
/// Cache positions: two chunks of the prompt.
const CAPACITY: usize = 256;

/// The tier 2 tolerance of a 24-layer step: `max |npu - oracle| / max |oracle|` per tensor, the quantity the
/// narrowing contract bounds. Measured on the laptop NPU (driver compiler 1.38.0): worst 1.9e-2 over the KV states
/// and 8.3e-3 over the logits of two chunks; the bound is twice the worst. This whole-step figure stands beside
/// the contract's recorded 5e-3 (`NPU4_F32_NARROWING`), which is the fragment-class figure (a block's MLP and
/// attention core) and does not extend to 24 chained layers. The reference is the F32 evaluation: the narrowed
/// graph's F16 types describe the region's interface, and the interior precision is the vendor compiler's, which
/// is why [`PINNED_STACK`] pins its version.
const RELATIVE_BOUND: f64 = 4.0e-2;

/// The stack pin (`poot-npu/npu-stack.pin`): the vendor compiler version the bound above was measured against.
/// `poot-npu`'s model-free `npu_stack_pin` row ties its `release` to the flake.
const PINNED_STACK: &str = include_str!("../../poot-npu/npu-stack.pin");

/// The pinned compiler `(major, minor)`.
fn pinned_compiler() -> VendorCompilerId {
    let value = PINNED_STACK
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| line.strip_prefix("compiler "))
        .expect("npu-stack.pin has a `compiler` line")
        .trim();
    let (major, minor) = value.split_once('.').expect("compiler is `major.minor`");
    VendorCompilerId {
        major: major.parse().expect("compiler major"),
        minor: minor.parse().expect("compiler minor"),
    }
}

/// `max |got - want| / max |want|`, panicking on a NaN or infinity in `got` where `want` is finite (ADR-0101).
fn normwise_error(label: &str, got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0.0f64, |m, w| m.max(f64::from(w.abs())));
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() || !w.is_finite(),
            "{label}: element {i} is {g} where the oracle is {w}"
        );
        worst = worst.max(f64::from((g - w).abs()));
    }
    if scale == 0.0 { worst } else { worst / scale }
}

fn zeros_like(graph: &Graph, input: poot_graph_ir::ValueId) -> HostTensor {
    let aval = graph.aval(input);
    match aval.dtype {
        DType::F16 => HostTensor::f16(
            aval.shape.clone(),
            vec![0; aval.shape.iter().product::<usize>()],
        ),
        DType::F32 => HostTensor::zeros(aval.shape.clone()),
        other => panic!("no zero state for {other:?}"),
    }
}

/// A program that returns every carried state of `graph` concatenated, flat, in `graph.state()` order: an entry
/// of the same executable shares those states and so reads what the step wrote.
fn state_reader(graph: &Graph) -> Graph {
    let b = Builder::for_tests();
    let parts: Vec<_> = graph
        .state()
        .map(|(input, _)| {
            let meta = graph.meta(input);
            let aval = graph.aval(input).clone();
            let numel = aval.shape.iter().product::<usize>();
            let role = graph.state_role(input).expect("a state has a role");
            let name = meta.name().expect("a state has a name");
            let state = b.state_input(&name, aval, role).unwrap();
            b.reshape(state, vec![numel]).unwrap()
        })
        .collect();
    let all = b.concat(0, &parts).unwrap();
    b.finish(all).unwrap()
}

#[test]
fn two_prompt_chunks_of_qwen2_5_0_5b_match_the_oracle_on_the_npu() {
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-fp16.gguf"
    )) else {
        return;
    };
    // The lock first: two runs never share the device.
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/poot-vpu.lock")
        .expect("open the NPU lock file");
    lock.lock().expect("take the NPU lock");
    let config = VpuConfig {
        run_timeout: Duration::from_secs(120),
        cache_dir: Some(std::env::temp_dir().join("poot-vpu-qwen-prompt-cache")),
        step_widths: StepWidths::new([WIDTH]).unwrap(),
    };
    let Some(device) = open_or_skip(DeviceBackend::IntelNpu, VpuDevice::open_with(config)) else {
        return;
    };
    // The bound is only evidence for the compiler it was measured against: the contract leaves interior precision
    // to the vendor compiler, so another version is a re-validation, not a pass.
    assert_eq!(
        device.caps().compiler,
        Queried::Known(pinned_compiler()),
        "the device's driver compiler differs from the pin in poot-npu/npu-stack.pin: re-validate the whole-step \
         bound against the new compiler, then update the pin and the flake together"
    );
    let mut engine = Engine::new(device);

    // The checkpoint, mapped once; the family builds the model over it.
    let bytes = StoredBytes::map(File::open(&path).unwrap()).unwrap();
    let index = GgufIndex::parse_shards(&[bytes]).unwrap();
    let raw = RawConfig::Gguf(&index);
    let store = read_gguf(&index, &IdentityNames).unwrap();
    let parts = Registry::builtin().unwrap().build(&raw, &store).unwrap();
    let model = parts.model;
    let store = Arc::new(parts.store.unwrap_or(store));
    let map = Arc::new(model.weights().clone());

    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::new(WIDTH).unwrap(),
        NonZeroUsize::new(CAPACITY).unwrap(),
    )
    .with_kv_dtype(KvDtype::F16);
    let traced = model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let traced = bind_packed_weights(&traced, &WeightFormats::from_weight_map(&map)).unwrap();

    let target = engine.device().target();
    let options = engine
        .compile_options(CompileLimits::STANDARD)
        .with_matmul_precision(poot_graph_ir::MatmulPrecision::Exact);
    let compile_started = Instant::now();
    let program: StagedProgram = compile(&traced, target, &options);
    println!("planned in {:?}", compile_started.elapsed());
    let stages: Vec<_> = program.stages().collect();
    assert_eq!(stages.len(), 1, "one program, no host stage");
    let narrowed = stages[0].2.graph().clone();
    let reader = compile(&state_reader(&narrowed), target, &options);

    let exe = engine
        .load_weights(Arc::clone(&store), Arc::clone(&map))
        .unwrap();
    let vendor_started = Instant::now();
    let entry = engine.add_entry(exe, &program).unwrap();
    let read_entry = engine.add_entry(exe, &reader).unwrap();
    println!(
        "vendor compile of both entries: {:?}",
        vendor_started.elapsed()
    );
    println!("compiles on the device: {}", engine.device().compiles());

    // One prompt of two chunks: ids spread over the common part of the vocabulary.
    let prompt: Vec<i32> = (0..2 * WIDTH)
        .map(|i| 1000 + (i as i32 * 37) % 4000)
        .collect();
    let steps = [
        Step::new(&prompt[..WIDTH], 0),
        Step::new(&prompt[WIDTH..], WIDTH),
    ];
    let state_names: Vec<String> = narrowed
        .state()
        .map(|(input, _)| narrowed.meta(input).name().unwrap().into_owned())
        .collect();
    // The F32 reference: the traced graph evaluated without narrowing, over the same prompt.
    let mut reference_state: Vec<HostTensor> = traced
        .state()
        .map(|(input, _)| zeros_like(&traced, input))
        .collect();
    let mut worst_kv = 0.0f64;
    let mut worst_logits = 0.0f64;
    for (i, step) in steps.iter().enumerate() {
        let started = Instant::now();
        let got = engine
            .step(exe, entry, &step.inputs(), &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        let step_time = started.elapsed();
        let (reference, next) = eval_mapped(&traced, &store, &map, &step.slots(), &reference_state);
        reference_state = next;
        let got_logits = got.to_f32().unwrap();
        let want_logits = reference.to_f32().unwrap();
        let logits = normwise_error(&format!("chunk {i} logits"), &got_logits, &want_logits);
        let flat = engine
            .step(exe, read_entry, &StepInputs::new(), &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        let flat = flat.to_f32().unwrap();
        let mut at = 0;
        let mut kv = 0.0f64;
        let mut worst_state = "";
        for (name, oracle) in state_names.iter().zip(&reference_state) {
            let want = oracle.to_f32().unwrap();
            let got = &flat[at..at + want.len()];
            at += want.len();
            let error = normwise_error(&format!("chunk {i} state {name}"), got, &want);
            if error > kv {
                (kv, worst_state) = (error, name);
            }
        }
        assert_eq!(at, flat.len(), "every state element is accounted for");
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        println!(
            "RECEIPT chunk {i}: step {step_time:?}; logits error {logits:.3e}; worst KV error {kv:.3e} \
             ({worst_state}) over {} states; argmax npu {} oracle {}",
            state_names.len(),
            argmax(&got_logits),
            argmax(&want_logits),
        );
        assert!(
            logits <= RELATIVE_BOUND,
            "chunk {i}: logits error {logits:.3e} exceeds {RELATIVE_BOUND}"
        );
        assert!(
            kv <= RELATIVE_BOUND,
            "chunk {i}: {worst_state} error {kv:.3e} exceeds {RELATIVE_BOUND}"
        );
        assert_eq!(
            argmax(&got_logits),
            argmax(&want_logits),
            "chunk {i}: the greedy token"
        );
        worst_kv = worst_kv.max(kv);
        worst_logits = worst_logits.max(logits);
    }
    println!(
        "RECEIPT worst: logits {worst_logits:.3e}, KV {worst_kv:.3e} (bound {RELATIVE_BOUND})"
    );
    engine.unload(exe).unwrap();
    drop(lock);
}
