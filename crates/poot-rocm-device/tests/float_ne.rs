//! POOT-1460 on ROCm: a float `Ne` compiled for gfx and dispatched is IEEE, `NaN != x` true for every `x`
//! (see [`poot_executor_parity::float_ne`]).
//!
//! Mutation (never left in the tree): lower float `Ne` to `fcmp one` in `poot-codegen/src/emit.rs`; the
//! `NaN` lanes read back 0.0 and this row goes red.

use poot_executor::Device;
use poot_graph_plan::test_support::gpu_backend;
use poot_rocm_device::device::RocmDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn float_ne_is_ieee_on_nan_on_rocm() {
    let Some(mut device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let arch = match gpu_backend(&device.target()) {
        poot_target::Backend::AmdGcn(arch) => arch,
        other => panic!("RocmDevice::target must report AmdGcn, got {other:?}"),
    };
    let body = poot_kernel_ir::test_support::float_ne_kernel();
    let artifact = poot_codegen::KernelCache::open(poot_codegen::Target::AmdGcn(arch))
        .load_or_compile(
            &body,
            poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
        )
        .expect("float_ne must compile to HSACO");
    let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let kernel = device
        .load_kernel("float_ne", compiled)
        .expect("load the float_ne kernel");
    poot_executor_parity::float_ne::check("rocm", &mut device, &kernel);
}
