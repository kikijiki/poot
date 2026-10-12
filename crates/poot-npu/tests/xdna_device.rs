//! POOT-641: the XDNA2 NPU `Device` under the shared `Engine`, on the real device.
//!
//! Every row here runs the ordinary route: the planner admits an equation for the AIE target, `poot-codegen`
//! links it with the real Peano and IRON driver (`$POOT_AIE_LLC`, `$POOT_AIE_IRON`, exported by
//! `nix develop .#npu`), the engine loads the linked artifact and the device runs it through XRT. A box
//! without the NPU or the toolchain fails these rows; it never skips them. Rows take `/tmp/poot-npu.lock`
//! so two runs never share the device.
//!
//! ```text
//! nix develop .#npu -c cargo nextest run -p poot-npu --features xdna --test-threads=1
//! ```

#![cfg(feature = "xdna")]

use std::fs::File;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine};
use poot_executor_parity::Fixture;
use poot_graph_ir::{BinOp, Builder, Graph, Outputs, Slot, SlotKey, StateRole, TensorType};
use poot_npu::{XdnaConfig, XdnaDevice};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::{DType, HostTensor};

/// The held lock and an engine over the NPU. The lock is released when this drops.
struct Npu {
    engine: Engine<XdnaDevice>,
    _lock: File,
}

/// Hold `/tmp/poot-npu.lock` until the returned file drops: two runs never share the device.
fn lock() -> File {
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/poot-npu.lock")
        .expect("open the NPU lock file");
    lock.lock().expect("take the NPU lock");
    lock
}

/// The lock, and the device opened under it with `config`; `None` (a printed SKIP) when the device or the
/// Peano/IRON toolchain is absent and `POOT_REQUIRE_NPU` is not `1`. With it set, either absence fails the
/// row: the lane never passes on skips.
fn device(config: XdnaConfig) -> Option<(File, XdnaDevice)> {
    let lock = lock();
    let get = |name: &str| std::env::var(name).ok();
    for variable in ["POOT_AIE_LLC", "POOT_AIE_IRON"] {
        if get(variable).is_none() {
            let missing = format!("${variable} is unset (run under `nix develop .#npu`)");
            let reason = DeviceBackend::Npu.fail_if_required(get, missing);
            eprintln!("SKIP: NPU toolchain unavailable: {reason}");
            return None;
        }
    }
    let device = open_or_skip(DeviceBackend::Npu, XdnaDevice::open_with(config))?;
    Some((lock, device))
}

/// An engine over the NPU (see [`device`]).
fn npu() -> Option<Npu> {
    let (lock, device) = device(XdnaConfig::default())?;
    Some(Npu {
        engine: Engine::new(device),
        _lock: lock,
    })
}

/// A named dense F32 constant of the fixture's weight store.
struct Const {
    name: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn constant(name: &'static str, shape: &[usize], f: impl Fn(usize) -> f32) -> Const {
    let numel = shape.iter().product();
    Const {
        name,
        shape: shape.to_vec(),
        values: (0..numel).map(f).collect(),
    }
}

fn store(consts: &[Const]) -> WeightStore {
    let mut builder = WeightStore::builder();
    for c in consts {
        let bytes: Vec<u8> = c.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense =
            DenseWeight::try_new(DType::F32, c.shape.clone(), StoredBytes::from(bytes)).unwrap();
        builder.insert(c.name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

fn slot_key(name: &str) -> SlotKey {
    SlotKey::new(Slot::Activation, Some(name))
}

fn x_step(name: &str, shape: &[usize], f: impl Fn(usize) -> f32) -> Vec<StepFixture> {
    let numel: usize = shape.iter().product();
    vec![StepFixture {
        key: slot_key(name),
        tensor: HostTensor::f32(shape.to_vec(), (0..numel).map(f).collect()),
    }]
}

/// One fixture of the bank: whether its values are small integers, which every IEEE or AIE rounding of
/// `+ - *` and `max` reproduces exactly, so the device must equal the oracle bit for bit.
struct Row {
    fixture: Fixture,
    exact: bool,
}

/// `a + b` over two `[2, 3]` constants.
fn add_consts() -> Row {
    let b = Builder::new();
    let x = b.constant("a", TensorType::f32(vec![2, 3]));
    let y = b.constant("b", TensorType::f32(vec![2, 3]));
    let out = b.binary(BinOp::Add, x, y);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("a", &[2, 3], |i| i as f32 + 1.0),
        constant("b", &[2, 3], |i| 10.0 * (i as f32 + 1.0)),
    ]);
    Row {
        fixture: Fixture {
            name: "add_consts",
            graph,
            store,
            steps: vec![Vec::new(), Vec::new()],
        },
        exact: true,
    }
}

/// `a - b`: not commutative, so a swapped argument order flips the sign of every element.
fn sub_consts() -> Row {
    let b = Builder::new();
    let x = b.constant("a", TensorType::f32(vec![2, 3]));
    let y = b.constant("b", TensorType::f32(vec![2, 3]));
    let out = b.binary(BinOp::Sub, x, y);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("a", &[2, 3], |i| 100.0 + i as f32),
        constant("b", &[2, 3], |i| 3.0 * i as f32),
    ]);
    Row {
        fixture: Fixture {
            name: "sub_consts",
            graph,
            store,
            steps: vec![Vec::new()],
        },
        exact: true,
    }
}

/// `x * w + b` over a `[4, 8]` activation slot and two constants: three steps with different `x`, so the
/// recording is replayed over inputs the device had not seen when it recorded.
fn scale_shift_slot() -> Row {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![4, 8]));
    let w = b.constant("w", TensorType::f32(vec![4, 8]));
    let bias = b.constant("bias", TensorType::f32(vec![4, 8]));
    let scaled = b.binary(BinOp::Mul, x, w);
    let out = b.binary(BinOp::Add, scaled, bias);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("w", &[4, 8], |i| (i % 7) as f32 - 3.0),
        constant("bias", &[4, 8], |i| (i % 5) as f32),
    ]);
    let steps = (0..3)
        .map(|s| x_step("x", &[4, 8], move |i| ((i * (s + 2)) % 11) as f32 - 5.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "scale_shift_slot",
            graph,
            store,
            steps,
        },
        exact: true,
    }
}

/// `x * w - d` over a `[3, 5]` slot with fractional values (the AIE core's float rounding is not asserted
/// bit for bit).
fn mul_sub_fractional() -> Row {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![3, 5]));
    let w = b.constant("w", TensorType::f32(vec![3, 5]));
    let d = b.constant("d", TensorType::f32(vec![3, 5]));
    let scaled = b.binary(BinOp::Mul, x, w);
    let out = b.binary(BinOp::Sub, scaled, d);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("w", &[3, 5], |i| 0.37 * i as f32 - 2.0),
        constant("d", &[3, 5], |i| 1.5 + 0.25 * i as f32),
    ]);
    let steps = (0..2)
        .map(|s| x_step("x", &[3, 5], move |i| 0.113 * (i + 3 * s) as f32 - 0.9))
        .collect();
    Row {
        fixture: Fixture {
            name: "mul_sub_fractional",
            graph,
            store,
            steps,
        },
        exact: false,
    }
}

/// A carried running sum: `state' = state + x`, output `state'`. Every step reads the state the previous
/// step committed, so a dropped commit or a stale read shows in the output.
fn running_sum_state() -> Row {
    let b = Builder::new();
    let state = b.state_input("acc", TensorType::f32(vec![2, 4]), StateRole::Recurrent);
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![2, 4]));
    let next = b.binary(BinOp::Add, state, x);
    let graph: Graph = b
        .finish(Outputs::new(next).with_state(&[(state, next)]))
        .unwrap();
    let steps = (0..4)
        .map(|s| x_step("x", &[2, 4], move |i| (i + 1) as f32 * (s + 1) as f32))
        .collect();
    Row {
        fixture: Fixture {
            name: "running_sum_state",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: true,
    }
}

fn bank() -> Vec<Row> {
    vec![
        add_consts(),
        sub_consts(),
        scale_shift_slot(),
        mul_sub_fractional(),
        running_sum_state(),
    ]
}

fn assert_matches(name: &str, step: usize, got: &HostTensor, want: &HostTensor, exact: bool) {
    assert_eq!(got.dtype(), want.dtype(), "{name} step {step}: dtype");
    assert_eq!(got.shape(), want.shape(), "{name} step {step}: shape");
    let (got, want) = (got.to_f32().unwrap(), want.to_f32().unwrap());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let ok = if exact {
            g.to_bits() == w.to_bits()
        } else {
            (g - w).abs() <= 1e-5 * w.abs().max(1.0)
        };
        assert!(
            ok && !g.is_nan(),
            "{name} step {step} element {i}: the NPU computed {g}, the oracle {w}"
        );
    }
}

/// SC-001: the declared fixture bank, through the ordinary route onto the NPU, every step's output (which
/// is the carried state for the running sum) against the shared CPU oracle, element by element.
///
/// Mutation (device): in `XdnaDevice::replay` run the dispatch's buffers in reversed order (ABI argument
/// order corrupted) or skip `Op::Dispatch` (a dropped dispatch). The `sub_consts` row sees `b - a`, every
/// row sees a zero output where the dispatch was dropped.
#[test]
fn the_fixture_bank_matches_the_oracle_on_the_npu() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    for row in bank() {
        let fixture = &row.fixture;
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, fixture)
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
        assert_eq!(outputs.len(), fixture.steps.len());
        for (i, (got, step)) in outputs.iter().zip(&fixture.steps).enumerate() {
            let want = oracle(step);
            assert_matches(fixture.name, i, got, &want, row.exact);
        }
    }
}

/// `a + b` over two `[n]` constants.
fn add_of(name: &'static str, n: usize) -> Row {
    let b = Builder::new();
    let x = b.constant("a", TensorType::f32(vec![n]));
    let y = b.constant("b", TensorType::f32(vec![n]));
    let out = b.binary(BinOp::Add, x, y);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("a", &[n], |i| (i % 97) as f32),
        constant("b", &[n], |i| (i % 13) as f32 * 2.0),
    ]);
    Row {
        fixture: Fixture {
            name,
            graph,
            store,
            steps: vec![Vec::new()],
        },
        exact: true,
    }
}

/// Compile `graph` for `target` the way every executor entry does, returning the error the planner raised.
fn refusal_of(graph: Graph, target: poot_graph_plan::Target) -> poot_graph_plan::Refusal {
    use poot_graph_plan::{
        CompileError, CompileLimits, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement,
        Partition, PlanError, StagedCompileError, Submission, TargetSet, compile_staged,
    };
    let Err(error) = compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    ) else {
        panic!("the equation is admitted for the NPU");
    };
    match error {
        StagedCompileError::Compile(CompileError::Plan(plan)) => match *plan {
            PlanError::Refused(refusal) => *refusal,
            other => panic!("expected a typed refusal, got {other:?}"),
        },
        other => panic!("expected a planner refusal, got {other:?}"),
    }
}

/// The elements of each of an add's three bindings that fill the tile: 2688 four-byte elements, double
/// buffered, three FIFOs, is 64512 bytes, 64 KiB less the core's 1 KiB stack. Measured on this device: 2688
/// links and runs, 2689 is refused by `aiecc` ("Failed to allocate buffer ... Not all requested buffers fit
/// in the available memory").
const FULL_TILE_ELEMENTS: usize = 2688;

/// The device's `tile_local_bytes` is the measured limit: an add that exactly fills the tile runs and
/// matches the oracle, and one element more per binding is refused as a typed tile-local limit before the
/// toolchain runs (not as an `aiecc` allocation failure).
///
/// Mutation (engine): drop the `validate_recipe_tile_local` call in `Engine::aie_kernel`. The 2689-element
/// add is linked and `aiecc` fails with "IRON (link) failed ... Failed to allocate buffer", and the last
/// assertion goes red. Mutation (caps): `TILE_LOCAL_BYTES` of 65536; the first assertion fails.
#[test]
fn the_declared_tile_memory_is_the_size_that_links_and_runs() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let reserved = (3 * FULL_TILE_ELEMENTS * 4 * 2) as u32;
    let poot_graph_plan::Target::Aie { caps, .. } = target else {
        panic!("the device's target is an AIE target");
    };
    assert_eq!(
        caps.tile_local_bytes,
        poot_target::Queried::Known(reserved),
        "the declared tile-local limit is what a full-tile add reserves"
    );

    let full = add_of("full_tile_add", FULL_TILE_ELEMENTS);
    let mut oracle = oracle_for(&full.fixture.graph, &full.fixture.store);
    let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, &full.fixture)
        .expect("an add that fills the tile runs");
    assert_matches(
        full.fixture.name,
        0,
        &outputs[0],
        &oracle(&full.fixture.steps[0]),
        true,
    );

    let over = add_of("over_tile_add", FULL_TILE_ELEMENTS + 1);
    let error = poot_executor_parity::run_outputs(&mut npu.engine, target, &over.fixture)
        .expect_err("one element past the tile does not link");
    let reserved_over = 3 * (FULL_TILE_ELEMENTS + 1) * 4 * 2;
    assert!(
        error.contains(&format!(
            "link recipe tile-local bytes: needed {reserved_over}, the target offers {reserved}"
        )),
        "{error}"
    );
}

/// SC-002: equations the NPU has no lowering for are refused by the planner with a typed capability naming
/// the equation and the target, before any buffer is allocated or kernel linked. There is no matcher that
/// admits them and no panic: each is a `Refusal` value.
///
/// Mutation: in `plan_aie_eqn` (poot-graph-plan) route the `matmul_ops!()` arm through the GPU planner. The
/// matmul is admitted for the NPU and this row's `GpuOnly` expectation fails.
#[test]
fn equations_the_npu_cannot_lower_are_refused_with_typed_capabilities() {
    use poot_graph_ir::RedOp;
    use poot_graph_plan::Capability;
    use poot_kernelgen::KernelGenError;

    let Some(npu) = npu() else { return };
    let target = npu.engine.device().target();
    let f32s = |shape: &[usize]| TensorType::f32(shape.to_vec());

    let unary = {
        let b = Builder::new();
        let x = b.constant("x", f32s(&[2, 3]));
        let out = b.unary(poot_graph_ir::UnOp::Neg, x);
        b.finish(out).unwrap()
    };
    let max = {
        let b = Builder::new();
        let x = b.constant("x", f32s(&[2, 3]));
        let y = b.constant("y", f32s(&[2, 3]));
        let out = b.binary(BinOp::Max, x, y);
        b.finish(out).unwrap()
    };
    let matmul = {
        let b = Builder::new();
        let x = b.constant("x", f32s(&[2, 3]));
        let y = b.constant("y", f32s(&[3, 2]));
        let out = b.matmul(x, y);
        b.finish(out).unwrap()
    };
    let reduce = {
        let b = Builder::new();
        let x = b.constant("x", f32s(&[2, 3]));
        let out = b.reduce(RedOp::Sum, x, 1, false);
        b.finish(out).unwrap()
    };
    let collective = {
        let b = Builder::new();
        let x = b.constant("x", f32s(&[2, 3]));
        let out = b.all_reduce(x, RedOp::Sum, 0);
        b.finish(out).unwrap()
    };

    let kernelgen_unsupported = |missing: &Capability| {
        matches!(
            missing,
            Capability::KernelGen(KernelGenError::Unsupported { .. })
        )
    };
    let unary = refusal_of(unary, target);
    assert!(kernelgen_unsupported(&unary.missing), "{unary:?}");
    let max = refusal_of(max, target);
    assert!(kernelgen_unsupported(&max.missing), "{max:?}");
    let matmul = refusal_of(matmul, target);
    assert_eq!(matmul.missing, Capability::GpuOnly, "{matmul:?}");
    let reduce = refusal_of(reduce, target);
    assert!(kernelgen_unsupported(&reduce.missing), "{reduce:?}");
    let collective = refusal_of(collective, target);
    assert_eq!(
        collective.missing,
        Capability::CollectiveInStage,
        "{collective:?}"
    );
    for refusal in [&unary, &max, &matmul, &reduce, &collective] {
        assert_eq!(
            refusal.target,
            poot_target::TargetId::Aie(poot_target::AieTargetId::Xdna2)
        );
    }
    assert!(
        npu.engine.device().memory().is_empty(),
        "no buffer was allocated for a refused plan"
    );
}

/// The device's byte accounting is its own and returns to zero: with an entry loaded and stepped the
/// device holds live bytes in the weight, input and activation roles, and after `unload` every buffer the
/// engine allocated, and the clones the recording held, is freed.
///
/// Mutation: keep the recording's buffers alive in `XdnaDevice::replay` (push each `Op::Dispatch` buffer
/// into `quarantine`). `live_bytes` stays nonzero after unload and this row fails.
#[test]
fn every_buffer_is_released_when_the_executable_is_unloaded() {
    use poot_executor::{Executor, NoSync, StepInputs};
    use poot_graph_plan::Submission;
    use std::sync::Arc;

    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let row = scale_shift_slot();
    let fixture = &row.fixture;
    let live = |engine: &Engine<XdnaDevice>| -> u64 {
        engine
            .device()
            .memory()
            .iter()
            .map(|(_, counters)| counters.live_bytes)
            .sum()
    };

    let exe = npu
        .engine
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor_parity::fixture_weights(fixture),
        )
        .expect("the weights load");
    let entry = npu
        .engine
        .add_entry(
            exe,
            &poot_executor_parity::staged(fixture, target, Submission::Replay),
        )
        .expect("the entry compiles and loads");
    let mut inputs = StepInputs::new();
    for step in &fixture.steps[0] {
        inputs.push(step.key.clone(), step.tensor.shape(), step.tensor.view());
    }
    npu.engine
        .step(exe, entry, &inputs, &mut NoSync)
        .expect("the step runs");
    assert!(
        live(&npu.engine) > 0,
        "a loaded and stepped entry holds device buffers"
    );

    npu.engine.unload(exe).expect("the executable unloads");
    let memory = npu.engine.device().memory();
    assert!(!memory.is_empty(), "the device counted its allocations");
    for (role, counters) in &memory {
        assert!(counters.allocations > 0, "{role:?}");
        assert_eq!(
            counters.live_bytes, 0,
            "{role:?} holds {} bytes after unload",
            counters.live_bytes
        );
    }
}

/// Capture-and-replay: the second pass replays the recording over reset state and equals the first pass bit
/// for bit, for a stateless kernel over changing inputs and for the carried running sum.
///
/// Mutation: in `XdnaDevice::write_with` skip the `sync_to_device` (or the `fill`) so a reset or an input
/// write never reaches the buffer; the second pass reads stale state and differs.
#[test]
fn replaying_a_recording_over_reset_state_is_bit_identical() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    for row in [scale_shift_slot(), running_sum_state()] {
        let fixture = &row.fixture;
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_capture_replay(&mut npu.engine, target, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// A real linked elementwise add of `n` elements: the compiler's output for the ordinary sequential request,
/// linked by the real Peano and IRON driver, wrapped by the ordinary handle.
struct LinkedAdd {
    kernel: poot_runtime_common::CompiledKernel,
    words: std::num::NonZeroU32,
}

fn linked_add(n: usize) -> LinkedAdd {
    use poot_codegen::{
        AieLinkRecipe, KernelCache, Target, Toolchain, aie_kernel_handle, compile_aie,
    };
    use poot_kernelgen::{
        BodyLimits, ElementwiseSpec, Fold, KernelRequest, Layout, ValueBinary, ViewOperand,
    };
    use poot_runtime_common::{AieBinding, AieShapeAbi, ArgAccess};
    use poot_target::{AieTargetId, BufferStorage};

    let shape = vec![n];
    let operand = || ViewOperand {
        shape: shape.clone(),
        layout: Layout::contiguous(&shape),
    };
    let request = KernelRequest::Elementwise(ElementwiseSpec::Binary {
        op: ValueBinary::Basic(poot_kernel_ir::BinOp::Add, poot_kernel_ir::Ty::F32),
        out_shape: shape.clone(),
        a: operand(),
        b: operand(),
        fold: Fold {
            numel: n,
            two_d: false,
            width: 256,
        },
    });
    let limits = BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(4096).unwrap(),
        max_locals: std::num::NonZeroUsize::new(256).unwrap(),
    };
    let generated = poot_kernelgen::sequential_schedule(&request, &limits).unwrap();
    let bindings = generated
        .bindings
        .iter()
        .enumerate()
        .map(|(index, b)| {
            let access = if b.writes {
                ArgAccess::Write
            } else {
                ArgAccess::Read
            };
            AieBinding::new(index, BufferStorage::f32(), access, &b.dims, b.accessed).unwrap()
        })
        .collect();
    let abi = AieShapeAbi::new(bindings).unwrap();
    let dir = poot_test_util::unique_temp_path("npu-linked-add");
    std::fs::create_dir_all(&*dir).unwrap();
    let compiled = compile_aie(
        &generated.body,
        &AieLinkRecipe::sequential(&generated.body.name, AieTargetId::Xdna2, &abi).unwrap(),
        &Toolchain::from_env(),
        &KernelCache::at(dir.join("cache"), Target::AieCore),
        std::num::NonZeroU64::new(1 << 24).unwrap(),
    )
    .unwrap_or_else(|e| panic!("the real Peano and IRON toolchain links the add: {e}"));
    let words = std::num::NonZeroU32::new((compiled.artifact.instructions().len() / 4) as u32)
        .expect("a linked artifact has an instruction stream");
    let kernel = aie_kernel_handle(&generated.body, compiled.artifact).unwrap();
    LinkedAdd { kernel, words }
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// SC-002, the device's own side: everything it can refuse before XRT runs anything, on a real linked
/// kernel. A grid launch, an invocation of another stream, an argument count or element count that is not
/// the shape ABI, and a written output that is not last are each a typed [`XdnaError`]; the refusal leaves
/// the device healthy, and the valid dispatch that follows runs and computes the sum.
///
/// Mutation: delete the `check_bindings` call in `XdnaDevice::dispatch`. The short-buffer dispatch is
/// recorded, runs, and the device fails or computes over the wrong length instead of returning `Bindings`.
#[test]
fn a_dispatch_that_is_not_the_kernels_abi_is_refused_before_it_runs() {
    use poot_executor::{AieInvocation, Arg, BufferRole, Dispatch, LaunchKind};
    use poot_graph_plan::Submission;
    use poot_npu::XdnaError;
    use poot_target::{AieTargetId, BufferStorage};

    const N: usize = 6;
    let Some((_lock, mut device)) = device(XdnaConfig::default()) else {
        return;
    };
    let linked = linked_add(N);
    let kernel = device
        .load_kernel("add", linked.kernel)
        .expect("the kernel loads");
    let alloc = |device: &mut XdnaDevice, role, n| {
        device
            .allocate(role, BufferStorage::f32(), n)
            .expect("allocate")
    };
    let a = alloc(&mut device, BufferRole::Weight, N);
    let b = alloc(&mut device, BufferRole::Weight, N);
    let out = alloc(&mut device, BufferRole::Activation, N);
    let short = alloc(&mut device, BufferRole::Activation, N - 1);
    poot_executor::write_slice(&mut device, &a, 0, &f32_bytes((0..N).map(|i| i as f32))).unwrap();
    poot_executor::write_slice(
        &mut device,
        &b,
        0,
        &f32_bytes((0..N).map(|i| 100.0 + i as f32)),
    )
    .unwrap();

    let invocation = AieInvocation::new(AieTargetId::Xdna2, linked.words);
    let aie = LaunchKind::Aie(invocation);
    let arg = |buffer, elems| Arg { buffer, elems };
    let try_dispatch = |device: &mut XdnaDevice,
                        launch: LaunchKind,
                        inputs: &[Arg<'_, XdnaDevice>],
                        output: Arg<'_, XdnaDevice>| {
        device.begin(Submission::Replay).unwrap();
        let result = device.dispatch(Dispatch {
            kernel: &kernel,
            inputs,
            output,
            launch,
            work: 0,
        });
        device.abort().unwrap();
        result
    };
    let inputs = [arg(&a, N as u32), arg(&b, N as u32)];

    let grid = LaunchKind::Grid {
        threads: [N as u32, 1, 1],
        workgroup: [1, 1, 1],
    };
    assert!(matches!(
        try_dispatch(&mut device, grid, &inputs, arg(&out, N as u32)),
        Err(XdnaError::GridLaunch { threads: [6, 1, 1] })
    ));
    let other_stream = LaunchKind::Aie(AieInvocation::new(
        AieTargetId::Xdna2,
        std::num::NonZeroU32::new(linked.words.get() + 1).unwrap(),
    ));
    assert!(matches!(
        try_dispatch(&mut device, other_stream, &inputs, arg(&out, N as u32)),
        Err(XdnaError::InvocationMismatch { .. })
    ));
    assert!(matches!(
        try_dispatch(&mut device, aie, &inputs[..1], arg(&out, N as u32)),
        Err(XdnaError::Bindings { .. })
    ));
    assert!(matches!(
        try_dispatch(&mut device, aie, &inputs, arg(&short, (N - 1) as u32)),
        Err(XdnaError::Bindings { .. })
    ));
    assert!(matches!(
        try_dispatch(&mut device, aie, &inputs, arg(&out, (N - 1) as u32)),
        Err(XdnaError::Bindings { .. })
    ));

    // The refusals left the device usable: the valid dispatch runs and computes a + b.
    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &inputs,
            output: arg(&out, N as u32),
            launch: aie,
            work: 0,
        })
        .expect("the valid dispatch is recorded");
    let recording = device.finish().unwrap().expect("a recording");
    device.begin_step().unwrap();
    device.replay(&recording).expect("the recording runs");
    device.synchronize().unwrap();
    let mut bytes = vec![0u8; N * 4];
    device.read(&out, &mut bytes).unwrap();
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let want: Vec<f32> = (0..N).map(|i| 100.0 + 2.0 * i as f32).collect();
    assert_eq!(got, want);
}

/// SC-002, load side: a kernel that is not a linked AIE artifact is refused by the device before XRT is
/// touched, a typed [`XdnaError::NotAnAieKernel`] naming the kernel and its target.
///
/// Mutation: in `XdnaDevice::load_kernel` replace the `let ... else` refusal with `unreachable!()`-free
/// handling that proceeds (returns `Ok` with a dummy kernel); the row's `Err` assertion goes red.
#[test]
fn a_byte_artifact_kernel_is_refused_at_load() {
    use poot_codegen::{GpuTarget, kernel_handle};
    use poot_npu::XdnaError;

    let Some((_lock, mut device)) = device(XdnaConfig::default()) else {
        return;
    };
    let body = poot_kernel_ir::fixtures::add_kernel();
    let handle = kernel_handle(&body, GpuTarget::Nvptx, b"// ptx".to_vec());
    let error = device
        .load_kernel("ptx_add", handle)
        .err()
        .expect("refused");
    assert!(
        matches!(&error, XdnaError::NotAnAieKernel { key, .. } if key == "ptx_add"),
        "{error:?}"
    );
    assert!(device.memory().is_empty(), "nothing was allocated");
}

/// Failure lifetime: a run that does not complete within the device's deadline is a typed `Xrt` error, the
/// device is poisoned (every later call is `Poisoned`, none touches XRT), and the buffers the run named stay
/// allocated and charged: nothing the hardware may still write is freed under it.
///
/// The deadline is one microsecond, which the NPU cannot meet; the run completes harmlessly after the
/// host stopped waiting.
///
/// Mutation: in `XdnaDevice::run` drop the `self.poisoned = true` assignment. The replay after the timeout
/// is attempted again instead of returning `Poisoned`.
#[test]
fn a_run_that_misses_its_deadline_poisons_the_device_and_keeps_its_buffers() {
    use poot_executor::{AieInvocation, Arg, BufferRole, Dispatch, LaunchKind};
    use poot_graph_plan::Submission;
    use poot_npu::XdnaError;
    use poot_target::{AieTargetId, BufferStorage};

    const N: usize = 6;
    let config = XdnaConfig {
        run_timeout: std::time::Duration::from_micros(1),
    };
    let Some((_lock, mut device)) = device(config) else {
        return;
    };
    let linked = linked_add(N);
    let kernel = device
        .load_kernel("add", linked.kernel)
        .expect("the kernel loads");
    let a = device
        .allocate(BufferRole::Weight, BufferStorage::f32(), N)
        .unwrap();
    let b = device
        .allocate(BufferRole::Weight, BufferStorage::f32(), N)
        .unwrap();
    let out = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), N)
        .unwrap();
    let inputs = [
        Arg {
            buffer: &a,
            elems: N as u32,
        },
        Arg {
            buffer: &b,
            elems: N as u32,
        },
    ];
    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &inputs,
            output: Arg {
                buffer: &out,
                elems: N as u32,
            },
            launch: LaunchKind::Aie(AieInvocation::new(AieTargetId::Xdna2, linked.words)),
            work: 0,
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("a recording");
    device.begin_step().unwrap();

    let error = device
        .replay(&recording)
        .expect_err("the deadline is missed");
    assert!(
        matches!(&error, XdnaError::Xrt { operation: "kernel run", source } if source.to_string().contains("timed out")),
        "{error:?}"
    );
    assert!(matches!(
        device.replay(&recording),
        Err(XdnaError::Poisoned)
    ));
    assert!(matches!(device.synchronize(), Err(XdnaError::Poisoned)));
    assert!(matches!(
        device.allocate(BufferRole::Activation, BufferStorage::f32(), 1),
        Err(XdnaError::Poisoned)
    ));

    drop((inputs, recording));
    drop((a, b, out));
    let live: u64 = device.memory().iter().map(|(_, c)| c.live_bytes).sum();
    assert_eq!(
        live,
        (3 * N * 4) as u64,
        "the failed run's three buffers stay charged after every handle to them dropped"
    );
}
