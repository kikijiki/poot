//! POOT-1460 on PTX: a float `Ne` compiled for NVPTX and dispatched is IEEE, `NaN != x` true for every `x`
//! (see [`poot_executor_parity::float_ne`]).
//!
//! Mutation (never left in the tree): lower float `Ne` to `fcmp one` in `poot-codegen/src/emit.rs`; the
//! `NaN` lanes read back 0.0 and this row goes red.

use poot_executor::Device;
use poot_ptx_device::device::PtxDevice;
use poot_runtime_common::DeviceBackend;

#[test]
fn float_ne_is_ieee_on_nan_on_ptx() {
    let Some(mut device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    let body = poot_kernel_ir::test_support::float_ne_kernel();
    let artifact = poot_codegen::KernelCache::open(poot_codegen::Target::Nvptx)
        .load_or_compile(
            &body,
            poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
        )
        .expect("float_ne must compile to PTX");
    let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
    let kernel = device
        .load_kernel("float_ne", compiled)
        .expect("load the float_ne kernel");
    poot_executor_parity::float_ne::check("ptx", &mut device, &kernel);
}
