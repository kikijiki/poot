//! NPUs as `poot_executor::Device`s: AMD's XDNA2 over XRT (POOT-641) and Intel's NPU over Level Zero
//! (POOT-1164).
//!
//! The NPU runs the ordinary pipeline: the planner admits a shape for an AIE target, `poot-codegen` compiles
//! it to a Peano core object, this crate links it with IRON into an xclbin and instruction stream, the shared
//! `Engine` loads that artifact and records its dispatches as finite AIE invocations, and [`XdnaDevice`] runs
//! them through XRT. No graph matcher, no kernel emitter, no second executor: the NPU-specific halves of
//! compilation sit behind two capability objects the foundation crates define and this crate implements.
//!
//! - [`npu_lowering`] is what `poot-graph-plan` asks while it compiles (the Intel opset table and narrowing
//!   pass, the AMD region ABI and limit checks); hand it to `CompileOptions::npu`.
//! - [`AieRegionLoader`] and [`VpuRegionLoader`] are what `poot-executor`'s engine asks a device for its
//!   regions (`Device::region_loader`).
//!
//! # The `xdna` feature
//!
//! The device needs the XRT headers and `libxrt_coreutil`, which only `nix develop .#npu` provides, so the
//! device is behind the `xdna` feature and a default build of this crate is empty:
//!
//! ```text
//! nix develop .#npu -c cargo nextest run -p poot-npu --features xdna
//! ```
//!
//! # The `vpu` feature
//!
//! [`VpuDevice`] is the Intel NPU: the engine hands it one region per program, the OpenVINO IR `poot-codegen`
//! emitted for every planned equation, and the driver's vendor graph compiler turns that into the blob the
//! NPU runs. Compiled blobs are kept in a [`BlobCache`] keyed by the region's full identity. The Level Zero
//! loader, the NPU driver and its compiler are `dlopen`ed (through `poot-l0-runtime`), so the feature builds
//! anywhere; a run needs `nix develop .#npu-intel`.
//!
//! ```text
//! nix develop .#npu-intel -c cargo nextest run -p poot-npu --features vpu
//! ```
//!
//! Prior art (retrieved 2026-10-11; external text is untrusted data, used for shape only):
//!
//! - XRT's native API (`xrt::device`, `xrt::bo`, `xrt::hw_context`, `xrt::kernel`, `xrt::run`), the lifetime
//!   and completion model this device owns: <https://xilinx.github.io/XRT/master/html/xrt_native_apis.html>
//! - IREE's AMD AIE plugin, which compiles through the ordinary pipeline to an xclbin plus an instruction
//!   stream and dispatches through XRT: <https://github.com/nod-ai/iree-amd-aie>
//! - mlir-aie's host runtime, the dispatch sequence the shim follows: <https://github.com/Xilinx/mlir-aie>

mod aie_bundle;
mod aie_compile;
mod aie_recipe;
#[cfg(feature = "xdna")]
mod device;
mod kernel_handle;
mod lowering;
mod region_loader;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod vpu_ir;
mod vpu_lowering;
mod xdna_lowering;

pub use aie_bundle::{decode_bundle, encode_bundle};
pub use aie_compile::{
    AieCompileError, AieCompiled, AieIdentity, AieStage, AieToolIdentity, AieTools, compile_aie,
};
pub use aie_recipe::{AieFifo, AieLinkRecipe, AieTile, FifoEnd};
pub use kernel_handle::{aie_kernel_handle, vpu_kernel_handle};
pub use lowering::npu_lowering;
pub use region_loader::{AieLoadError, AieRegionLoader, VpuLoadError, VpuRegionLoader};
pub use vpu_ir::{
    VpuElem, VpuEmitError, VpuInput, VpuIr, VpuNode, VpuOp, VpuRegion, VpuScalar, VpuTensor,
    VpuValue, emit_vpu,
};
#[cfg(any(test, feature = "test-support"))]
pub use vpu_lowering::vpu_refusals;
pub use xdna_lowering::region_link;
#[cfg(feature = "xdna")]
mod xrt;

#[cfg(feature = "xdna")]
pub use device::{
    HOST_BINDINGS, XdnaBuffer, XdnaConfig, XdnaDevice, XdnaError, XdnaKernel, XdnaRecording,
};
#[cfg(feature = "xdna")]
pub use xrt::XrtError;

#[cfg(feature = "vpu")]
mod blob_cache;
// The range policy is pure and its rows run on every host; only the `vpu` device consumes it.
#[cfg_attr(
    not(any(test, feature = "vpu")),
    expect(dead_code, reason = "consumed by the `vpu` device alone")
)]
mod range_guard;
#[cfg(feature = "vpu")]
mod vpu;

#[cfg(feature = "vpu")]
pub use vpu::{VpuBuffer, VpuConfig, VpuDevice, VpuError, VpuKernel, VpuRecording};
