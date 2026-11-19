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

/// POOT-1566 W1 on PTX: an argument bound 256 bytes into its buffer reads from there (SC-e), and a binding off the
/// kernel floor is refused as `PtxError::MisalignedBinding` (SC-f). Mutation: drop `address()`'s offset in
/// `KernelArgs::pack` (the first row reads from the base and goes red); drop the alignment check (the second row
/// launches).
#[test]
fn an_offset_binding_is_read_from_its_offset_and_a_misaligned_one_is_refused_on_ptx() {
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
    poot_executor_parity::offset_binding::an_offset_binding_reads_from_its_offset(
        "ptx",
        &mut device,
        &kernel,
    );
    let refused =
        poot_executor_parity::offset_binding::a_misaligned_offset_is_refused(&mut device, &kernel);
    assert!(
        matches!(
            &refused,
            poot_ptx_device::PtxGpuError::Ptx(error)
                if matches!(**error, poot_ptx_runtime::PtxError::MisalignedBinding { .. })
        ),
        "got {refused:?}"
    );
}
