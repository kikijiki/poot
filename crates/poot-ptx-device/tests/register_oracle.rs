//! POOT-1505 / POOT-1509: the PTX register oracle against the POOT-1509 sweep. Every kernel the sweep compiled for
//! sm_120 (784 tilings of eight shapes, 385 distinct bodies) is rebuilt the way the sweep built it, then asked of the
//! CUDA driver's JIT through the engine's compile feedback (the engine lowers the probe, [`PtxKernelRegisters`] compiles it); the verdict must be the one `ptxas` and the JIT gave then
//! (`tests/data/ptxas_sweep_sm120.tsv`). A host without an NVIDIA device skips.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use poot_executor::{Engine, Executor};
use poot_graph_ir::{
    Builder, FragmentDtype, IndexRange, MatmulPrecision, OperandPrecision, Slot, TensorType,
};
use poot_graph_plan::{
    CompileLimits, CompileOptions, FeedbackError, IndexedRows, RegisterFeedback, RegisterProbe,
    Target, compile,
};
use poot_kernelgen::{
    RegisterTiling, StagedTerms, StagedWeight, TiledRows, TiledShape, WeightLayout,
};
use poot_ptx_device::PtxDevice;
use poot_ptx_runtime::ProbeError;
use poot_quant::format::WeightFormat;
use poot_runtime_common::DeviceBackend;
use poot_target::calibration::Calibrated;
use poot_target::{Backend, DeviceCaps, KernelCompiler, MatrixUnits, Queried, RegisterFile};
use poot_tensor::DType;

/// The driver the sweep's verdicts were taken on (`cuDriverGetVersion` of 595.91.07): the table says nothing of
/// another compiler, so a row on a different driver fails by name and the sweep is repeated.
const SWEEP_DRIVER_RELEASE: u32 = 13_020;
const SWEEP_COMPUTE_CAPABILITY: (u32, u32) = (12, 0);
/// The distinct kernel bodies of the sm_120 sweep.
const SWEEP_DISTINCT_BODIES: usize = 385;

#[derive(Clone, Copy)]
enum Kind {
    Grouped(WeightFormat, usize),
    PackedDense(WeightFormat),
    Dense(DType),
}

struct Shape {
    id: &'static str,
    kind: Kind,
    k: usize,
    n: usize,
    rows: &'static [usize],
}

fn shapes() -> Vec<Shape> {
    use WeightFormat::*;
    vec![
        Shape {
            id: "G1",
            kind: Kind::Grouped(Iq2_Xs, 512),
            k: 2560,
            n: 640,
            rows: &[128, 1280, 5120, 20480, 81920],
        },
        Shape {
            id: "G2",
            kind: Kind::Grouped(Iq4_Nl, 512),
            k: 640,
            n: 2560,
            rows: &[1280, 20480],
        },
        Shape {
            id: "G3",
            kind: Kind::Grouped(Q4_K, 128),
            k: 2048,
            n: 768,
            rows: &[1024, 16384],
        },
        Shape {
            id: "D1",
            kind: Kind::PackedDense(Q4_K),
            k: 2560,
            n: 9728,
            rows: &[16, 512, 2048],
        },
        Shape {
            id: "D2",
            kind: Kind::PackedDense(Q6_K),
            k: 9728,
            n: 2560,
            rows: &[512],
        },
        Shape {
            id: "D3",
            kind: Kind::PackedDense(Q8_0),
            k: 2048,
            n: 2048,
            rows: &[512],
        },
        Shape {
            id: "D4",
            kind: Kind::Dense(DType::BF16),
            k: 1024,
            n: 3072,
            rows: &[2048],
        },
        Shape {
            id: "D5",
            kind: Kind::Dense(DType::F16),
            k: 2560,
            n: 9728,
            rows: &[512],
        },
    ]
}

/// sm_120 as the sweep planned it. `lifted` gives the lane a register file far larger than the device's, so the
/// planner's register model admits every tiling the sweep compiled (the sweep lifted that bound to see what the
/// compiler does with them); the kernels are the same, only the enumeration differs.
fn caps(lifted: bool) -> DeviceCaps {
    let file = if lifted {
        RegisterFile {
            per_lane: 4096,
            per_workgroup: 4096 * 1024,
        }
    } else {
        RegisterFile::nvidia(12).unwrap()
    };
    DeviceCaps {
        matrix_units: Queried::Known(MatrixUnits::nvidia((12, 0))),
        register_file: Queried::Known(file),
        compute_units: Queried::Known(188),
        perf_calibration: Calibrated::of_cuda_compute_capability(12, 0),
        ..DeviceCaps::ptx_default()
    }
}

fn graph(shape: &Shape, rows: usize) -> poot_graph_ir::Graph {
    use poot_graph_ir::test_support::packed_graphs::{
        PackedExpertStack, TEST_STACK, packed_indexed_linear, packed_linear,
    };
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![rows, shape.k]))
        .unwrap();
    let out = match shape.kind {
        Kind::Grouped(format, experts) => {
            let ids = b
                .slot_named_ranged(
                    Slot::Activation,
                    "ids",
                    TensorType::new(vec![rows], DType::I32),
                    IndexRange::below(experts).unwrap(),
                )
                .unwrap();
            let table = PackedExpertStack {
                stack: TEST_STACK,
                descriptor: poot_quant::PackedWeight::try_new(format, [shape.n, shape.k]).unwrap(),
                experts,
            };
            packed_indexed_linear(&b, x, ids, &table).unwrap()
        }
        Kind::PackedDense(format) => packed_linear(
            &b,
            x,
            "layer",
            poot_quant::PackedWeight::try_new(format, [shape.n, shape.k]).unwrap(),
            None,
            None,
        )
        .unwrap(),
        Kind::Dense(dtype) => {
            let w = b
                .constant("w", TensorType::new(vec![shape.n, shape.k], dtype))
                .unwrap();
            let wt = b.transpose(w, vec![1, 0]).unwrap();
            b.matmul(x, wt).unwrap()
        }
    };
    b.finish(out).unwrap()
}

/// The shape every candidate tiling is enumerated over: rows free, the weight as the planner stages it.
fn tiled_shape(shape: &Shape, rows: usize, narrow: bool) -> TiledShape {
    let precision = if narrow {
        OperandPrecision::Narrow(FragmentDtype::F16)
    } else {
        OperandPrecision::Exact
    };
    let (weight, stored) = match shape.kind {
        Kind::Grouped(format, _) | Kind::PackedDense(format) => {
            (StagedWeight::Packed(format), None)
        }
        Kind::Dense(DType::BF16) => (StagedWeight::Dense { bits: 16 }, Some(FragmentDtype::Bf16)),
        Kind::Dense(_) => (StagedWeight::Dense { bits: 16 }, Some(FragmentDtype::F16)),
    };
    TiledShape {
        blocks: 1,
        m_per_block: rows,
        out_per_block: shape.n,
        k: shape.k,
        layout: WeightLayout::Nk,
        rows: TiledRows::Dense,
        terms: StagedTerms::of(precision, None, stored),
        weight,
        activation_bits: 32,
    }
}

/// One sweep kernel: its table key, the verdict the sweep recorded, and the body the planner emits for it.
struct Kernel {
    key: String,
    ptxas_registers: u32,
    spills: bool,
    probe: RegisterProbe,
}

/// The sweep table's rows by key `shape rows precision block schedule`.
fn table() -> BTreeMap<String, (u32, bool)> {
    include_str!("data/ptxas_sweep_sm120.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            (f[..5].join(" "), (f[5].parse().unwrap(), f[6] == "1"))
        })
        .collect()
}

/// Plan `shape` at `rows` and `precision` with `tiling` pinned, and take the contraction's body.
fn contraction_body(
    shape: &Shape,
    rows: usize,
    narrow: bool,
    tiling: RegisterTiling,
    caps: DeviceCaps,
) -> Option<poot_kernel_ir::Body> {
    let target = Target::Gpu {
        backend: Backend::Nvptx,
        caps,
    };
    let mut options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(if narrow {
            MatmulPrecision::Narrow
        } else {
            MatmulPrecision::Exact
        });
    options.declare_precision = narrow;
    options.fragment_only = narrow.then_some(FragmentDtype::F16);
    options.tiling_pin = Some(tiling);
    if matches!(shape.kind, Kind::Grouped(..)) {
        options.indexed_rows_pin = Some(IndexedRows::Grouped);
    }
    let program = compile(&graph(shape, rows), &target, &options).ok()?;
    program
        .planned()
        .flat_map(|(_, plan)| plan.dispatches())
        .find(|d| d.label().starts_with("contraction"))
        .map(|d| d.body().clone())
}

/// Every kernel of the sm_120 sweep table, rebuilt: the candidate tilings of each shape under the device's register
/// model and under the lifted one (the sweep's union), those the table names, each planned with its tiling pinned.
fn sweep_kernels() -> Vec<Kernel> {
    let table = table();
    let mut seen = HashSet::new();
    let mut kernels = Vec::new();
    for shape in shapes() {
        for &rows in shape.rows {
            for narrow in [false, true] {
                for lifted in [false, true] {
                    let caps = caps(lifted);
                    for tiling in
                        RegisterTiling::candidates(tiled_shape(&shape, rows, narrow), &caps)
                    {
                        let key = format!(
                            "{} {} {} {} {:?}",
                            shape.id,
                            rows,
                            if narrow { "narrow" } else { "exact" },
                            tiling.output_tile().0,
                            tiling.schedule()
                        );
                        let Some(&(ptxas_registers, spills)) = table.get(&key) else {
                            continue;
                        };
                        if !seen.insert(key.clone()) {
                            continue;
                        }
                        let body = contraction_body(&shape, rows, narrow, tiling, caps)
                            .unwrap_or_else(|| panic!("{key}: no contraction body"));
                        kernels.push(Kernel {
                            key,
                            ptxas_registers,
                            spills,
                            probe: RegisterProbe { body },
                        });
                    }
                }
            }
        }
    }
    assert_eq!(
        kernels.len(),
        table.len(),
        "every row of the sweep table is a kernel the planner emits"
    );
    kernels
}

/// The sweep's kernels are rebuilt from the planner, not read from a file: all 784 tilings of the table, 385 distinct
/// bodies. Needs no device. Mutation: leave `lifted` out of the enumeration in `sweep_kernels`; the tilings only the
/// lifted register model admits are missing.
#[test]
fn the_sweep_table_is_the_planners_kernels() {
    let kernels = sweep_kernels();
    let distinct: HashSet<_> = kernels.iter().map(|k| k.probe.digest()).collect();
    assert_eq!(distinct.len(), SWEEP_DISTINCT_BODIES);
    assert_eq!(kernels.iter().filter(|k| k.spills).count(), 318);
}

fn open() -> Option<PtxDevice> {
    isolated_kernel_cache();
    poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
}

/// This process keeps its kernels and verdicts in a directory of its own: a verdict a run of the lane (or of a mutated
/// build) left in the shared cache must not stand in for a compile this row asks for. Set before any engine opens its
/// cache; every test of this binary asks for the same directory.
fn isolated_kernel_cache() {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("poot-register-oracle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // SAFETY: set once, before this process's tests open an engine, and read only by the kernel cache.
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &dir) };
        dir
    });
}

/// The engine over `device` and the compile feedback it plans with.
fn engine_feedback(device: PtxDevice) -> (Engine<PtxDevice>, RegisterFeedback) {
    let engine = Engine::new(device);
    let feedback = engine
        .compile_options(CompileLimits::STANDARD)
        .register_feedback
        .expect("the PTX engine plans with the driver's feedback");
    (engine, feedback)
}

/// The oracle reports the sweep's verdict on every kernel of the sm_120 sweep, from any thread, and names the
/// compiler it asked. Mutation: report `local_bytes: 0` in `PtxKernelRegisters::registers`; the 141 distinct bodies
/// (318 tilings) the compiler spills read "holds".
#[test]
fn the_ptx_oracle_matches_the_ptxas_sweep_verdicts() {
    let Some(device) = open() else { return };
    let kernels = sweep_kernels();
    let (_engine, feedback) = engine_feedback(device);
    assert_eq!(
        feedback.compiler(),
        KernelCompiler::CudaDriver {
            release: SWEEP_DRIVER_RELEASE,
            compute_capability: SWEEP_COMPUTE_CAPABILITY,
        },
        "the sweep's verdicts are the driver 595.91.07's; another compiler needs the sweep repeated"
    );
    let distinct: HashSet<_> = kernels.iter().map(|k| k.probe.digest()).collect();
    let next = AtomicUsize::new(0);
    let wrong = std::sync::Mutex::new(Vec::new());
    let register_gap = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                while let Some(kernel) = kernels.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let verdict = feedback.oracle().compile(&kernel.probe);
                    match verdict {
                        Ok(verdict) => {
                            if verdict.spills() != kernel.spills {
                                wrong.lock().unwrap().push(format!(
                                    "{}: the sweep says {}, the oracle {} bytes of local memory",
                                    kernel.key,
                                    if kernel.spills { "spills" } else { "clean" },
                                    verdict.local_bytes
                                ));
                            }
                            if verdict.registers.abs_diff(kernel.ptxas_registers) > 2 {
                                register_gap.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(error) => wrong
                            .lock()
                            .unwrap()
                            .push(format!("{}: {error}", kernel.key)),
                    }
                }
            });
        }
    });
    println!(
        "compiler {:?}; {} tilings, {} distinct bodies, {} off ptxas by more than 2 registers",
        feedback.compiler(),
        kernels.len(),
        distinct.len(),
        register_gap.load(Ordering::Relaxed)
    );
    let wrong = wrong.into_inner().unwrap();
    assert!(
        wrong.is_empty(),
        "{} of {} kernels disagree with the sweep:\n{}",
        wrong.len(),
        kernels.len(),
        wrong.join("\n")
    );
}

/// The smallest sweep kernel that spills, and one that does not: probes for the rows below.
fn probes() -> (RegisterProbe, RegisterProbe) {
    let kernels = sweep_kernels();
    let pick = |spills| {
        kernels
            .iter()
            .find(|k| k.spills == spills && k.key.starts_with("D3 512"))
            .expect("a D3 kernel")
            .probe
            .clone()
    };
    (pick(true), pick(false))
}

/// A verdict is compiled once: the second `compile` of a probe through the engine's feedback does not reach the
/// driver, and a fresh engine (a later process, here a second device handle) over the same cache directory reads the
/// verdict from disk and compiles nothing. Mutation: write no verdict file in `VerdictCache::compile`; the fresh
/// engine compiles.
#[test]
fn a_ptx_verdict_is_compiled_once() {
    let Some(device) = open() else { return };
    let (spilling, clean) = probes();
    let probes_of =
        |engine: &Engine<PtxDevice>| engine.device().context().native_counters().register_probes;
    let (first, feedback) = engine_feedback(device);
    let oracle = feedback.oracle();
    let verdict = oracle.compile(&spilling).unwrap();
    assert!(verdict.spills(), "the probe is a spilling kernel");
    assert_eq!(oracle.compile(&spilling).unwrap(), verdict);
    assert_eq!(oracle.compile(&clean).unwrap().local_bytes, 0);
    assert_eq!(probes_of(&first), 2, "each probe reaches the driver once");
    let second = PtxDevice::new().expect("a second handle on the device");
    let (second, feedback) = engine_feedback(second);
    let oracle = feedback.oracle();
    assert_eq!(
        oracle.compile(&spilling).unwrap(),
        verdict,
        "the verdict on disk"
    );
    assert_eq!(oracle.compile(&clean).unwrap().local_bytes, 0);
    assert_eq!(
        probes_of(&second),
        0,
        "a fresh engine over the same directory compiles nothing"
    );
}

/// A probe with no verdict is a `Compile` error that names it and does not hold, whether the body does not lower or
/// the driver's JIT refuses its PTX; a thread with no current context asks the same feedback about a probe it has not seen (a cached verdict would not reach the driver) and gets a verdict.
/// Mutation: do not bind the context in `PtxCompilerProbe::function_registers` (call the driver without
/// `with_current`); the second thread's call fails.
#[test]
fn a_probe_the_jit_refuses_is_a_compile_error() {
    let Some(device) = open() else { return };
    let (spilling, clean) = probes();
    let compiler = device
        .context()
        .compiler_probe()
        .expect("the device's compiler");
    let (_engine, feedback) = engine_feedback(device);
    // A body with no blocks does not lower.
    let unlowerable = RegisterProbe {
        body: poot_kernel_ir::Body::new("a_body_that_does_not_lower", 0, Vec::new(), Vec::new()),
    };
    match feedback.oracle().compile(&unlowerable) {
        Err(FeedbackError::Compile { name, reason }) => {
            assert_eq!(name, "a_body_that_does_not_lower");
            println!("does not lower: {reason}");
        }
        other => panic!("expected a compile error naming the probe, got {other:?}"),
    }
    assert!(!feedback.holds(&unlowerable));
    // Text the driver's JIT refuses is a refusal, not a device failure.
    match compiler.function_registers("this is not PTX", "a_kernel") {
        Err(ProbeError::Refused(reason)) => println!("refused: {reason}"),
        other => panic!("expected a JIT refusal, got {other:?}"),
    }
    assert!(
        feedback.holds(&clean),
        "the clean probe still holds after a refusal"
    );
    std::thread::scope(|scope| {
        scope
            .spawn(|| feedback.oracle().compile(&spilling))
            .join()
            .expect("the thread does not panic")
            .expect("a thread with no current context gets a verdict")
    });
}
