//! NPUs as `poot_executor::Device`s: AMD's XDNA2 over XRT (POOT-641) and Intel's NPU over Level Zero
//! (POOT-1164).
//!
//! The NPU runs the ordinary pipeline: the planner admits a shape for an AIE target, `poot-codegen` compiles
//! it to a Peano core object and links it with IRON into an xclbin and instruction stream, the shared
//! `Engine` loads that artifact and records its dispatches as finite AIE invocations, and [`XdnaDevice`] runs
//! them through XRT. This crate is XRT allocation, buffer copy, load, dispatch, completion and failure
//! lifetime, and nothing else: no graph matcher, no kernel emitter, no second executor.
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

#[cfg(feature = "xdna")]
mod device;
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
#[cfg(feature = "vpu")]
mod vpu;

#[cfg(feature = "vpu")]
pub use vpu::{VpuBuffer, VpuConfig, VpuDevice, VpuError, VpuKernel, VpuRecording};
