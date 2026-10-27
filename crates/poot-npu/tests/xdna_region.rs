//! POOT-1160: the XDNA2 region artifact on the real device.
//!
//! One loaded artifact is one hardware context and this NPU holds 16, so a program that needs more kernels
//! than that only runs if many planned equations share an artifact. Every row here runs the ordinary route:
//! `compile` packages the planned equations into regions, `poot-codegen` links each region with the real
//! Peano and IRON driver (one worker per equation, one object FIFO per value, host tensors at fixed offsets in
//! a few arenas), the engine loads one artifact per region and the device runs it once per step through XRT.
//! Outputs are compared with the shared CPU oracle. Rows take `/tmp/poot-npu.lock`; a box without the NPU or
//! the toolchain fails them, it never skips them.
//!
//! ```text
//! nix develop .#npu -c cargo nextest run -p poot-npu --features xdna --test-threads=1
//! ```

#![cfg(feature = "xdna")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_executor_parity::Fixture;
use poot_graph_ir::{BinOp, Builder, Outputs, Slot, StateRole, TensorType};
use poot_npu::{XdnaConfig, XdnaDevice};
use poot_quant::weights::WeightStore;
use poot_runtime_common::BufferRole;

use common::{assert_matches, constant, device, npu, store, x_step};

/// The slot value `x{index}` takes at `step`: small integers, so every rounding of `+ - *` agrees.
fn slot_value(index: usize, step: usize, element: usize) -> f32 {
    (((element + 3 * index) * (step + 2)) % 7) as f32 - 3.0
}

/// `n` independent updates `next_i = state_i <op> x_i` of distinct shapes `[i + 1, 3]`, alternating add and
/// sub, each state carried to the next step. The result is `next_{result}`. No equation reads another's
/// output, and no two have the same shape, so every one is a kernel of its own: `n` of them are `n` hardware
/// contexts on a device that packages nothing.
fn updates(n: usize, result: usize, steps: usize) -> Fixture {
    let b = Builder::for_tests();
    let mut carried = Vec::new();
    let mut published = None;
    let mut slots = Vec::new();
    for i in 0..n {
        let ty = || TensorType::f32(vec![i + 1, 3]);
        let state = b
            .state_input(&format!("s{i}"), ty(), StateRole::Recurrent)
            .unwrap();
        let x = b
            .slot_named(Slot::Activation, &format!("x{i}"), ty())
            .unwrap();
        slots.push((format!("x{i}"), vec![i + 1, 3]));
        let op = if i % 2 == 0 { BinOp::Add } else { BinOp::Sub };
        let next = b.binary(op, state, x).unwrap();
        if i == result {
            published = Some(next);
        }
        carried.push((state, next));
    }
    let graph = b
        .finish(Outputs::new(published.unwrap()).with_state(&carried))
        .unwrap();
    let steps = (0..steps)
        .map(|step| {
            slots
                .iter()
                .enumerate()
                .flat_map(|(i, (name, shape))| {
                    x_step(name, shape, move |element| slot_value(i, step, element))
                })
                .collect()
        })
        .collect();
    Fixture {
        name: "updates",
        graph,
        store: WeightStore::builder().build(),
        steps,
    }
}

/// `len` equations chained on `x`: add the constant `c`, multiply by the constant `d`, and so on, over
/// `[3, 4]`. Every equation reads the one before, and `c` and `d` are read by many.
fn chain(len: usize, steps: usize) -> Fixture {
    let shape = vec![3, 4];
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
        .unwrap();
    let c = b.constant("c", TensorType::f32(shape.clone())).unwrap();
    let d = b.constant("d", TensorType::f32(shape.clone())).unwrap();
    let mut acc = x;
    for i in 0..len {
        acc = if i % 2 == 0 {
            b.binary(BinOp::Add, acc, c).unwrap()
        } else {
            b.binary(BinOp::Mul, acc, d).unwrap()
        };
    }
    let graph = b.finish(acc).unwrap();
    let store = store(&[
        constant("c", &shape, |i| (i % 3) as f32 - 1.0),
        constant("d", &shape, |i| if i % 2 == 0 { 1.0 } else { -1.0 }),
    ]);
    let steps = (0..steps)
        .map(|s| x_step("x", &shape, move |i| ((i + s) % 5) as f32 - 2.0))
        .collect();
    Fixture {
        name: "chain",
        graph,
        store,
        steps,
    }
}

/// Every step of `fixture` through `engine`, each against the oracle, exactly.
fn run_against_oracle(engine: &mut Engine<XdnaDevice>, fixture: &Fixture) {
    let target = engine.device().target();
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let outputs = poot_executor_parity::run_outputs(engine, target, fixture)
        .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
    assert_eq!(outputs.len(), fixture.steps.len());
    for (i, (got, step)) in outputs.iter().zip(&fixture.steps).enumerate() {
        assert_matches(fixture.name, i, got, &oracle(step), true);
    }
}

/// `step`'s values as the engine's step inputs.
fn inputs_of(step: &[StepFixture]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for value in step {
        inputs.push(value.key.clone(), value.tensor.shape(), value.tensor.view());
    }
    inputs
}

/// The hardware contexts a device has open: one per kernel it loaded.
fn contexts(engine: &Engine<XdnaDevice>) -> u64 {
    engine.device().kernels_loaded()
}

/// SC-001, the headline: a graph of twenty equations with twenty distinct shapes needs twenty kernels, and
/// this NPU holds sixteen hardware contexts. Packaged into regions it needs three, and every one of its
/// twenty carried states is right: only the published result of a graph is observable, so the graph is run
/// once per equation with that equation's output as the result, each on a fresh engine over the device.
///
/// Without regions (every equation its own artifact) the seventeenth load fails with
/// `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)`.
///
/// Mutation: in `package` close a region after every equation (stop growing at `start + 1`). The first
/// engine fails at the seventeenth artifact with that error.
#[test]
fn twenty_equations_of_distinct_shapes_run_within_the_sixteen_hardware_contexts() {
    const EQUATIONS: usize = 20;
    let Some((_lock, first)) = device(XdnaConfig::default()) else {
        return;
    };
    let mut device = Some(first);
    for result in 0..EQUATIONS {
        let mut engine = Engine::new(
            device
                .take()
                .unwrap_or_else(|| XdnaDevice::open().expect("reopen the NPU")),
        );
        let fixture = updates(EQUATIONS, result, 3);
        run_against_oracle(&mut engine, &fixture);
        assert_eq!(
            contexts(&engine),
            3,
            "twenty equations, eight to a region (sixteen streams in), are three artifacts"
        );
    }
}

/// SC-002: values inside a region never reach host memory. A chain of six equations is one artifact whose
/// host side is the slot, the two constants and the one value it returns; the five values in between exist
/// only as on-chip FIFOs. Measured on the device: the only computed buffer the engine allocates is the
/// returned value's (`[3, 4]` F32, 48 bytes), where one buffer per equation would be 144 bytes at least, and
/// the artifact binds two host arenas, not one buffer per tensor.
///
/// Mutation: in `plan_buffers` drop the `inside_region` skip. Every intermediate gets an arena slot and the
/// activation bytes are no longer 48.
#[test]
fn a_chain_region_returns_one_value_and_allocates_nothing_for_the_rest() {
    let Some(mut npu) = npu() else { return };
    let fixture = chain(6, 2);
    let target = npu.engine.device().target();
    let exe = npu
        .engine
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor_parity::fixture_weights(&fixture),
        )
        .expect("the weights load");
    let program = poot_executor_parity::staged_on(&npu.engine, &fixture, target);
    let entry = npu
        .engine
        .add_entry(exe, &program)
        .expect("the chain loads");
    assert_eq!(contexts(&npu.engine), 1, "six equations, one artifact");
    let live = |engine: &Engine<XdnaDevice>, role: BufferRole| -> u64 {
        engine
            .device()
            .memory()
            .iter()
            .find(|(r, _)| *r == role)
            .map_or(0, |(_, counters)| counters.live_bytes)
    };
    assert_eq!(
        live(&npu.engine, BufferRole::Activation),
        3 * 4 * 4,
        "one [3, 4] F32 buffer for the returned value, none for the five in between"
    );
    // The host side of the artifact: the slot and both constants (read arena of 3 x 12 words) and the one
    // returned value (write arena of 12 words), owned by the recording.
    npu.engine
        .step(exe, entry, &inputs_of(&fixture.steps[0]), &mut NoSync)
        .expect("the step runs");
    assert_eq!(
        live(&npu.engine, BufferRole::Staging),
        (3 * 12 + 12) * 4,
        "two arenas hold the region's four host tensors"
    );
}

/// A value that is both returned and read on chip: `first = x + state` is the carried state of the next
/// step and also feeds `second = first * x`, the published result. The artifact streams `first` to the host
/// and to the second tile from one FIFO; both uses must see the same values over several steps.
///
/// Mutation: in `Uses::region` leave published values out of the region's writes. The carried state is never
/// returned and the second step's result differs from the oracle.
#[test]
fn a_value_that_is_returned_and_read_on_chip_is_right_on_both_paths() {
    let Some(mut npu) = npu() else { return };
    let shape = vec![3, 4];
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
        .unwrap();
    let state = b
        .state_input("s", TensorType::f32(shape.clone()), StateRole::Recurrent)
        .unwrap();
    let first = b.binary(BinOp::Add, x, state).unwrap();
    let second = b.binary(BinOp::Mul, first, x).unwrap();
    let graph = b
        .finish(Outputs::new(second).with_state(&[(state, first)]))
        .unwrap();
    let steps = (0..4)
        .map(|s| x_step("x", &shape, move |i| ((i + 2 * s) % 5) as f32 - 2.0))
        .collect();
    let fixture = Fixture {
        name: "returned_and_read_on_chip",
        graph,
        store: WeightStore::builder().build(),
        steps,
    };
    run_against_oracle(&mut npu.engine, &fixture);
    assert_eq!(contexts(&npu.engine), 1, "both equations are one region");
}

/// One equation `x + c` over `[3, 4]`: the unit a program without regions dispatches.
fn single_add() -> Fixture {
    let shape = vec![3, 4];
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
        .unwrap();
    let c = b.constant("c", TensorType::f32(shape.clone())).unwrap();
    let out = b.binary(BinOp::Add, x, c).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("c", &shape, |i| (i % 3) as f32 - 1.0)]);
    Fixture {
        name: "single_add",
        graph,
        store,
        steps: vec![x_step("x", &shape, |i| (i % 5) as f32 - 2.0)],
    }
}

/// SC-003: a region runs as one invocation however many equations it holds. Six chained equations are one
/// artifact and one `xrt::run` per step, counted by the device; the same six equations as six single-equation
/// entries take six runs per round. The wall time of the two is printed as evidence of what the shared context
/// saves (it is not a performance gate).
///
/// Mutation: in `package` stop growing a region at its first equation. Six artifacts are loaded and the
/// first assertion fails (the runs assertion that follows it needs the same region to hold).
#[test]
fn a_six_equation_region_is_one_run_per_step() {
    const ROUNDS: u64 = 20;
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let load = |npu: &mut common::Npu, fixture: &Fixture| {
        let exe = npu
            .engine
            .load_weights(
                Arc::new(fixture.store.clone()),
                poot_executor_parity::fixture_weights(fixture),
            )
            .expect("the weights load");
        let program = poot_executor_parity::staged_on(&npu.engine, fixture, target);
        let entry = npu
            .engine
            .add_entry(exe, &program)
            .expect("the entry loads");
        (exe, entry)
    };

    let region = chain(6, 1);
    let (exe, entry) = load(&mut npu, &region);
    assert_eq!(contexts(&npu.engine), 1, "six equations, one artifact");
    let before = npu.engine.device().kernel_runs();
    let start = Instant::now();
    for _ in 0..ROUNDS {
        npu.engine
            .step(exe, entry, &inputs_of(&region.steps[0]), &mut NoSync)
            .expect("the region steps");
    }
    let region_time = start.elapsed();
    let region_runs = npu.engine.device().kernel_runs() - before;
    assert_eq!(
        region_runs, ROUNDS,
        "six equations, one run per step: the region is one invocation"
    );

    let single = single_add();
    let entries: Vec<_> = (0..6).map(|_| load(&mut npu, &single)).collect();
    let before = npu.engine.device().kernel_runs();
    let start = Instant::now();
    for _ in 0..ROUNDS {
        for &(exe, entry) in &entries {
            npu.engine
                .step(exe, entry, &inputs_of(&single.steps[0]), &mut NoSync)
                .expect("the single equation steps");
        }
    }
    let single_time = start.elapsed();
    let single_runs = npu.engine.device().kernel_runs() - before;
    assert_eq!(
        single_runs,
        6 * ROUNDS,
        "six single-equation entries, six runs"
    );
    eprintln!(
        "SC-003 evidence over {ROUNDS} rounds: one 6-equation region {region_time:?} ({region_runs} runs); \
         six single-equation entries {single_time:?} ({single_runs} runs)"
    );
}

/// The sequential body of one F32 elementwise `op` over `dims`, as the ordinary request generates it.
fn body_of(op: poot_kernel_ir::BinOp, dims: &[u64]) -> poot_kernel_ir::Body {
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let operand = || poot_kernelgen::ViewOperand {
        shape: shape.clone(),
        layout: poot_kernelgen::Layout::contiguous(&shape),
    };
    let leaf = |index| poot_kernelgen::FusedInput::Leaf(poot_kernelgen::LeafId::new(index));
    let request = poot_kernelgen::KernelRequest::Pointwise(poot_kernelgen::PointwiseSpec {
        out_shape: shape.clone(),
        numel: shape.iter().product(),
        leaves: vec![operand(), operand()],
        kernel: poot_kernelgen::FusedKernel::one_step(
            poot_kernelgen::FusedScalarOp::Binary(op),
            vec![poot_kernelgen::LeafTy::F32; 2],
            vec![leaf(0), leaf(1)],
        ),
        form: poot_kernelgen::PointwiseForm::Float {
            fold: poot_kernelgen::Fold {
                numel: shape.iter().product(),
                two_d: false,
                width: 256,
            },
        },
    });
    let limits = poot_kernelgen::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(4096).unwrap(),
        max_locals: std::num::NonZeroUsize::new(256).unwrap(),
    };
    poot_kernelgen::serialized_schedule(&request, &limits)
        .unwrap()
        .body
}

/// A region wired by hand, below `package`: the bodies in tile order and the recipe linking them.
struct Region {
    bodies: Vec<poot_kernel_ir::Body>,
    recipe: poot_npu::AieLinkRecipe,
}

/// A host-side description of region wiring: each tensor is `[2, 3]` F32.
const ELEMENTS: u64 = 6;

fn fifo(producer: poot_npu::FifoEnd, consumers: Vec<poot_npu::FifoEnd>) -> poot_npu::AieFifo {
    poot_npu::AieFifo {
        producer,
        consumers,
        depth: poot_npu::AieLinkRecipe::FIFO_DEPTH,
        elements: ELEMENTS,
        element: poot_target::ElementKind::F32,
    }
}

fn abi(reads: usize, writes: usize) -> poot_npu_artifacts::AieShapeAbi {
    use poot_npu_artifacts::AieBinding;
    use poot_target::ArgAccess;
    let binding = |i, access| {
        AieBinding::new(
            i,
            poot_target::BufferStorage::f32(),
            access,
            &[2, 3],
            ELEMENTS,
        )
        .unwrap()
    };
    poot_npu_artifacts::AieShapeAbi::new(
        (0..reads)
            .map(|i| binding(i, ArgAccess::Read))
            .chain((0..writes).map(|i| binding(reads + i, ArgAccess::Write)))
            .collect(),
    )
    .unwrap()
}

/// `n` independent adds. With `shared`, they all read the same two host tensors (one broadcast stream each)
/// and each returns its own result: `n` host streams out, two in. Otherwise each reads its own two tensors:
/// `2n` streams in, `n` out.
fn independent_adds(n: usize, shared: bool) -> Region {
    use poot_npu::{AieLinkRecipe, AieTile, FifoEnd};
    let tile = |tile, port| FifoEnd::Tile { tile, port };
    let host = |binding| FifoEnd::Host { binding };
    let (mut fifos, mut tiles) = (Vec::new(), Vec::new());
    if shared {
        fifos.push(fifo(host(0), (0..n).map(|i| tile(i, 0)).collect()));
        fifos.push(fifo(host(1), (0..n).map(|i| tile(i, 1)).collect()));
        for i in 0..n {
            fifos.push(fifo(tile(i, 2), vec![host(2 + i)]));
            tiles.push(AieTile {
                ports: vec![0, 1, 2 + i],
            });
        }
    } else {
        for i in 0..n {
            fifos.push(fifo(host(2 * i), vec![tile(i, 0)]));
            fifos.push(fifo(host(2 * i + 1), vec![tile(i, 1)]));
            fifos.push(fifo(tile(i, 2), vec![host(2 * n + i)]));
            tiles.push(AieTile {
                ports: vec![3 * i, 3 * i + 1, 3 * i + 2],
            });
        }
    }
    let abi = if shared { abi(2, n) } else { abi(2 * n, n) };
    Region {
        bodies: vec![body_of(poot_kernel_ir::BinOp::Add, &[2, 3]); n],
        recipe: AieLinkRecipe::new(poot_target::AieTargetId::Xdna2, abi, tiles, fifos).unwrap(),
    }
}

/// `n` adds chained over on-chip links: tile `i` adds the host tensor `c`, which every tile reads (one
/// broadcast stream), to what tile `i - 1` produced; tile 0 reads the host tensor `x`.
fn broadcast_chain(n: usize) -> Region {
    use poot_npu::{AieLinkRecipe, AieTile, FifoEnd};
    let tile = |tile, port| FifoEnd::Tile { tile, port };
    let host = |binding| FifoEnd::Host { binding };
    // FIFO 0 is x, 1 is c, 2 + i is tile i's output.
    let mut fifos = vec![
        fifo(host(0), vec![tile(0, 0)]),
        fifo(host(1), (0..n).map(|i| tile(i, 1)).collect()),
    ];
    for i in 0..n {
        let consumer = if i + 1 < n { tile(i + 1, 0) } else { host(2) };
        fifos.push(fifo(tile(i, 2), vec![consumer]));
    }
    let tiles = (0..n)
        .map(|i| AieTile {
            ports: vec![if i == 0 { 0 } else { 2 + i - 1 }, 1, 2 + i],
        })
        .collect();
    Region {
        bodies: vec![body_of(poot_kernel_ir::BinOp::Add, &[2, 3]); n],
        recipe: AieLinkRecipe::new(poot_target::AieTargetId::Xdna2, abi(2, 1), tiles, fifos)
            .unwrap(),
    }
}

/// What `aiecc` does with `region`, with no packaging in front of it: the artifact it links, or the error.
fn link(
    region: &Region,
) -> Result<poot_npu_artifacts::AieLinkedArtifact, poot_npu::AieCompileError> {
    let dir = poot_test_util::unique_temp_path("npu-region-link");
    std::fs::create_dir_all(&*dir).unwrap();
    let bodies: Vec<&poot_kernel_ir::Body> = region.bodies.iter().collect();
    poot_npu::compile_aie(
        &bodies,
        &region.recipe,
        &poot_codegen::Toolchain::from_env(),
        &poot_codegen::KernelCache::at(dir.join("cache"), poot_codegen::Target::AieCore),
        std::num::NonZeroU64::new(1 << 26).unwrap(),
    )
    .map(|compiled| compiled.artifact)
}

fn iron_failure(error: poot_npu::AieCompileError) -> String {
    match error {
        poot_npu::AieCompileError::Tool {
            stage: poot_npu::AieStage::Iron,
            stderr,
            ..
        } => stderr,
        other => panic!("expected the IRON link to refuse, got {other}"),
    }
}

/// SC-004, the declared bounds are the real ones: the artifacts at exactly the limits link, and one past
/// each is refused by `aiecc` itself. `package` closes a region at these same numbers (the limits are
/// `XdnaDevice`'s declared `RegionLimits`), so a constant that drifted from the hardware fails here, not
/// at the user.
///
/// Mutation: set `compute_tiles` in the device's `REGION_LIMITS` to 33; the 33-tile artifact in the packaging
/// row below no longer splits and its link fails in `aiecc`. Set the host-input bound to 17 and the 9-add
/// region of the same row fails the same way.
#[test]
fn a_region_at_every_declared_limit_links_and_one_past_is_refused() {
    let Some((_lock, device)) = device(XdnaConfig::default()) else {
        return;
    };
    let poot_graph_plan::Target::Npu(poot_graph_plan::NpuTarget::Aie { caps, .. }) =
        device.target()
    else {
        panic!("the device's target is an AIE target");
    };
    let limits = caps.region;
    let known = |limit: poot_target::Queried<u32>| match limit {
        poot_target::Queried::Known(n) => n as usize,
        poot_target::Queried::Unknown => panic!("the device declares every region limit"),
    };
    let (tiles, reads, writes) = (
        known(limits.compute_tiles),
        known(limits.host_input_streams),
        known(limits.host_output_streams),
    );
    assert_eq!((tiles, reads, writes), (32, 16, 16));

    // Host streams in: two per add, so eight adds are the sixteen streams and nine are eighteen.
    link(&independent_adds(reads / 2, false)).expect("sixteen streams in link");
    let refused = iron_failure(link(&independent_adds(reads / 2 + 1, false)).unwrap_err());
    assert!(refused.contains("ShimNOCTile DMA budget"), "{refused}");

    // Host streams out: the adds share their inputs and each returns its own result.
    link(&independent_adds(writes, true)).expect("sixteen streams out link");
    let refused = iron_failure(link(&independent_adds(writes + 1, true)).unwrap_err());
    assert!(refused.contains("ShimNOCTile DMA budget"), "{refused}");

    // Compute tiles: a chain of adds sharing one broadcast stream.
    link(&broadcast_chain(tiles)).expect("thirty-two tiles link");
    iron_failure(link(&broadcast_chain(tiles + 1)).unwrap_err());
}

/// SC-004, end to end: `package` splits a program at the declared bounds and every split program runs and
/// equals the oracle. Nine independent updates read eighteen host tensors, past the sixteen streams in, so
/// they are a region of eight and a region of one. A chain of forty equations is past the thirty-two tiles,
/// so it is a region of thirty-two and one of eight, the second reading what the first returned.
///
/// Mutation: in `validate_region_recipe` drop the host-input-stream bound. The nine updates are one region
/// and the link fails in `aiecc` ("total ShimNOCTile DMA budget") instead of the program splitting; drop the
/// compute-tile bound and the chain of forty fails the same way.
#[test]
fn packaging_splits_at_the_declared_bounds_and_each_split_program_runs() {
    let Some((_lock, first)) = device(XdnaConfig::default()) else {
        return;
    };
    let mut device = Some(first);
    let engine = |device: &mut Option<XdnaDevice>| {
        Engine::new(
            device
                .take()
                .unwrap_or_else(|| XdnaDevice::open().expect("reopen the NPU")),
        )
    };
    for result in [0, 8] {
        let mut e = engine(&mut device);
        run_against_oracle(&mut e, &updates(9, result, 2));
        assert_eq!(
            contexts(&e),
            2,
            "nine updates are a region of eight and one"
        );
    }
    let mut e = engine(&mut device);
    run_against_oracle(&mut e, &chain(40, 2));
    assert_eq!(
        contexts(&e),
        2,
        "forty chained equations are a region of thirty-two and one of eight"
    );
}

/// SC-005 on the device: two chains of the same two bodies, one the other's reverse, are different artifacts
/// and compute different functions. Both are linked through one cache directory and run: `(a + b) - c` and
/// `(a - b) + c`. A cache key that ignored which body sits on which tile would hand the second chain the
/// first one's xclbin, and it would compute the wrong function.
///
/// Mutation: leave the `placed` lines out of the link text in `AieIdentity::new` and the tile and fifo lines
/// out of `AieLinkRecipe::canonical`. The second compile is served the first chain's xclbin and the second
/// assertion fails.
#[test]
fn a_reordered_chain_is_its_own_artifact_and_computes_its_own_function() {
    use poot_executor::{Arg, Dispatch, LaunchKind};
    use poot_kernel_ir::BinOp::{Add, Sub};
    use poot_npu::{AieFifo, AieLinkRecipe, AieTile, FifoEnd};
    use poot_npu_artifacts::AieInvocation;
    use poot_runtime_common::BufferRole;
    use poot_target::{AieTargetId, BufferStorage};

    let Some((_lock, mut device)) = device(XdnaConfig::default()) else {
        return;
    };
    let dir = poot_test_util::unique_temp_path("npu-reorder");
    std::fs::create_dir_all(&*dir).unwrap();
    let cache = poot_codegen::KernelCache::at(dir.join("cache"), poot_codegen::Target::AieCore);
    let toolchain = poot_codegen::Toolchain::from_env();

    // `out = (a op0 b) op1 c`: tile 0 reads host a and b, tile 1 reads tile 0's value and host c.
    let chain = |op0, op1| {
        let host = |binding| FifoEnd::Host { binding };
        let tile = |tile, port| FifoEnd::Tile { tile, port };
        let fifos: Vec<AieFifo> = vec![
            fifo(host(0), vec![tile(0, 0)]),
            fifo(host(1), vec![tile(0, 1)]),
            fifo(host(2), vec![tile(1, 1)]),
            fifo(tile(0, 2), vec![tile(1, 0)]),
            fifo(tile(1, 2), vec![host(3)]),
        ];
        let tiles = vec![
            AieTile {
                ports: vec![0, 1, 3],
            },
            AieTile {
                ports: vec![3, 2, 4],
            },
        ];
        Region {
            bodies: vec![body_of(op0, &[2, 3]), body_of(op1, &[2, 3])],
            recipe: AieLinkRecipe::new(AieTargetId::Xdna2, abi(3, 1), tiles, fifos).unwrap(),
        }
    };
    let run = |device: &mut XdnaDevice, region: &Region| -> Vec<f32> {
        let bodies: Vec<&poot_kernel_ir::Body> = region.bodies.iter().collect();
        let compiled = poot_npu::compile_aie(
            &bodies,
            &region.recipe,
            &toolchain,
            &cache,
            std::num::NonZeroU64::new(1 << 26).unwrap(),
        )
        .unwrap_or_else(|e| panic!("the real toolchain links the chain: {e}"));
        let words =
            std::num::NonZeroU32::new((compiled.artifact.instructions().len() / 4) as u32).unwrap();
        let kernel = device
            .load_kernel("chain", poot_npu::aie_kernel_handle(compiled.artifact))
            .expect("the chain loads");
        let alloc = |device: &mut XdnaDevice, role| {
            device
                .allocate(role, BufferStorage::f32(), ELEMENTS as usize)
                .expect("allocate")
        };
        let buffers: Vec<_> = [
            BufferRole::Input,
            BufferRole::Input,
            BufferRole::Input,
            BufferRole::Output,
        ]
        .into_iter()
        .map(|role| alloc(device, role))
        .collect();
        for (index, buffer) in buffers[..3].iter().enumerate() {
            let bytes: Vec<u8> = (0..ELEMENTS)
                .flat_map(|i| ((10u32.pow(index as u32) * (i as u32 + 1)) as f32).to_le_bytes())
                .collect();
            poot_executor::write_slice(device, buffer, 0, &bytes).unwrap();
        }
        let args: Vec<Arg<'_, XdnaDevice>> = buffers
            .iter()
            .map(|buffer| Arg {
                buffer,
                elems: ELEMENTS as usize,
            })
            .collect();
        device.begin().unwrap();
        device
            .dispatch(Dispatch {
                kernel: &kernel,
                inputs: &args[..3],
                output: Arg {
                    buffer: &buffers[3],
                    elems: ELEMENTS as usize,
                },
                launch: LaunchKind::Aie(AieInvocation::new(AieTargetId::Xdna2, words)),
                work: 0,
            })
            .expect("the dispatch is recorded");
        let recording = device.finish().unwrap();
        device.begin_step().unwrap();
        device.replay(&recording).expect("the recording runs");
        device.synchronize().unwrap();
        let mut bytes = vec![0u8; ELEMENTS as usize * 4];
        device.read(&buffers[3], &mut bytes).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    // a = 1.., b = 10.., c = 100.. (element i: (i + 1) * 10^k).
    let a_plus_b_minus_c: Vec<f32> = (0..ELEMENTS)
        .map(|i| (i as f32 + 1.0) * (1.0 + 10.0 - 100.0))
        .collect();
    let a_minus_b_plus_c: Vec<f32> = (0..ELEMENTS)
        .map(|i| (i as f32 + 1.0) * (1.0 - 10.0 + 100.0))
        .collect();
    assert_eq!(run(&mut device, &chain(Add, Sub)), a_plus_b_minus_c);
    assert_eq!(run(&mut device, &chain(Sub, Add)), a_minus_b_plus_c);
}
