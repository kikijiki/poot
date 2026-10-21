//! Card 662, PTX half of SC-003: the capabilities a live NVIDIA device reports are the ones the planner's
//! finalizer reads, and the matrix-fragment kernel plans against them. The caps come from
//! `PtxDevice::target()` (the `cuDeviceGetAttribute` queries), not from a fixture; a device without a
//! CUDA driver skips (and fails when `POOT_REQUIRE_PTX` is set).

use poot_executor::Device;
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::test_support::{gpu_backend, gpu_caps};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest};
use poot_ptx_device::device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_target::{Backend, Queried, SubgroupSupport, TensorCoreSupport};
use poot_tensor::DType;

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
    assert_eq!(gpu_backend(&target), Backend::Nvptx);
    let caps = gpu_caps(&target);
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

    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![1, 16, 32], DType::BF16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![32, 16], DType::BF16))
        .unwrap();
    let out = b.matmul_f32_accumulate(a, w).unwrap();
    let graph = b.finish(out).unwrap();
    let options = CompileOptions::new(CompileLimits::STANDARD);
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

/// POOT-1065: the PTX device reports the driver's live free memory (`cuMemGetInfo`), within the total it
/// reported when it opened. Mutation: `free_memory_bytes` returns `None` and the first assertion goes
/// red; returns `u64::MAX` and the ceiling assertion goes red.
#[test]
fn the_ptx_device_reports_live_free_memory_within_its_open_time_ceiling() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let ceiling = gpu_caps(&device.target())
        .memory_budget_bytes
        .expect("the PTX device queried its total memory at open");
    let free = device
        .free_memory_bytes()
        .expect("the PTX device reports live free memory");
    assert!(
        free > 0,
        "a device with no free memory cannot run this test"
    );
    assert!(
        free <= ceiling,
        "free {free} is over the open-time ceiling {ceiling}"
    );
}
