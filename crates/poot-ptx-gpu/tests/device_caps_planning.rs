//! Card 662, PTX half of SC-003: the capabilities a live NVIDIA device reports are the ones the planner's
//! finalizer reads, and the matrix-fragment kernel plans against them. The caps come from
//! `PtxDevice::target()` (the `cuDeviceGetAttribute` queries), not from a fixture; a device without a
//! CUDA driver skips (and fails when `POOT_REQUIRE_PTX` is set).

use poot_executor::Device;
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Submission, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_ptx_gpu::device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_target::{Backend, Queried, SubgroupSupport, TensorCoreSupport};
use poot_tensor::DType;
use poot_test_util::device_skip::open_or_skip;

/// A live device reports positive workgroup limits, a fixed warp that contains the 32 lanes every fragment
/// kernel is written for, and a matrix family consistent with its compute capability; a bf16 16-aligned
/// matmul then plans the tensor-core kernel against exactly those caps (no refusal from the finalizer).
///
/// Mutation: report `TensorCoreSupport::None` from `from_cuda_compute_capability`, and the compile is
/// refused with `FragmentUnsupported` for an sm_80+ device.
#[test]
fn a_live_device_reports_caps_the_fragment_kernel_plans_against() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    assert_eq!(target.backend, Backend::Nvptx);
    let caps = target.caps;
    eprintln!("live ptx caps: {caps:?}");
    assert!(caps.max_workgroup_invocations >= 1);
    assert!(caps.max_workgroup_size.iter().all(|&extent| extent >= 1));
    assert_eq!(
        caps.subgroup,
        Queried::Known(SubgroupSupport::Present {
            min_size: 32,
            max_size: 32
        })
    );
    // NVPTX codegen targets sm_80; the planner's tensor-core kernel needs a device that can run it.
    if caps.tensor_core != TensorCoreSupport::NvidiaWmma16x16x16Sm80 {
        eprintln!(
            "SKIP: pre-Ampere device ({:?}); the fragment kernel is refused by design",
            caps.tensor_core
        );
        return;
    }

    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 16, 32], DType::BF16));
    let w = b.constant("w", TensorType::new(vec![32, 16], DType::BF16));
    let out = b.matmul(a, w);
    let mut graph = b.finish(out);
    let out = graph.output;
    graph.values[out].aval.dtype = DType::F32;
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    let program = compile(&graph, &target, &options)
        .unwrap_or_else(|error| panic!("the live caps must admit the fragment kernel: {error}"));
    let (eqn, _) = program
        .planned()
        .find(|(_, plan)| plan.kind() == poot_graph_plan::PlanKind::Dispatches)
        .expect("a dispatching equation");
    assert!(
        matches!(
            program.kernel_choice(eqn),
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::TensorCore { .. }
            ))
        ),
        "expected the tensor-core kernel, got {:?}",
        program.kernel_choice(eqn)
    );
}

/// The routed-expert hang guard follows the live device's own caps: a graph with a routed-expert site
/// compiled for `PtxDevice::target()` skips `fuse` (the device reports the hang through its caps, not a
/// family test), and the same pointwise chain with no site still fuses. Runs on the PTX pod.
///
/// Mutation: build the live caps with `KnownMiscompiles::default()` in `poot-ptx-runtime`
/// (`caps_from_attributes`); the guarded half fuses and this row goes red.
#[test]
fn a_live_ptx_device_guards_a_routed_expert_graph() {
    use poot_graph_ir::{BinOp, Builder, TensorType};
    use poot_test_util::graph_fixtures::with_routed_expert_site;

    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let target = device.target();
    assert!(
        target.caps.known_miscompiles.routed_expert_tiling_hang,
        "a live PTX device reports the routed-expert hang"
    );
    let chain = || {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let y = b.constant("y", TensorType::f32(vec![4, 8]));
        let sum = b.binary(BinOp::Add, x, y);
        let out = b.binary(BinOp::Mul, sum, y);
        b.finish(out)
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        limits: CompileLimits::STANDARD,
    };
    let ran_fuse = |graph: &poot_graph_ir::Graph| {
        compile(graph, &target, &options)
            .expect("the chain compiles")
            .passes()
            .any(|pass| pass == "fuse")
    };
    assert!(
        !ran_fuse(&with_routed_expert_site(chain())),
        "a routed-expert graph must not fuse on the live PTX device"
    );
    assert!(ran_fuse(&chain()), "a graph with no site fuses");
}
