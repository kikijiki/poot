//! Level Zero for Intel devices, through the `dlopen`ed loader (`libze_loader.so.1`). The Intel analogue of
//! `poot-rocm-runtime`: every entry point is resolved with `dlsym`, so the crate builds without a Level Zero
//! installation, and a machine without one reports [`L0Error::LibraryNotFound`] or [`L0Error::NoDevice`] at
//! runtime. A context opens one kind of device ([`L0DeviceKind`]).
//!
//! # GPU: OpenCL SPIR-V kernels (`Target::SpirvOpenCl`)
//!
//! No production crate consumes the GPU half yet (the Intel GPU `Device` is a later card), so it is behind the
//! sanctioned `test-support` feature (POOT-ADR-0113); the first consumer removes the gate.
//!
//! - [`L0Context::open`]: `zeInit`, pick the first driver with a device of the kind, create a context and a
//!   queue on the device's compute queue group.
//! - [`L0Context::alloc_f32`] / [`L0Buffer::upload_f32`] / [`L0Buffer::download_f32`]: device memory, moved
//!   with a copy command list.
//! - [`L0Context::load_module`] / [`L0Module::kernel`] / [`L0Context::dispatch`]: `zeModuleCreate` from a
//!   SPIR-V module, one kernel object with its work-group size, and a launch that records a command list,
//!   executes it and waits for the queue. Every launch and copy waits for the queue.
//!
//! # NPU: vendor-compiled graphs (`ZE_extension_graph`)
//!
//! An Intel NPU has no kernel ISA; its only entry point is the driver's graph compiler.
//!
//! - [`L0Context::compile_graph`] turns OpenVINO IR ([`GraphIr`]) into an [`L0Graph`];
//!   [`L0Graph::native_binary`] returns the compiled blob and [`L0Context::load_native_graph`] loads one
//!   without compiling again. [`L0Graph::arguments`] is what the compiler produced, for a caller to check.
//! - [`L0Context::alloc_host`] allocates host memory the NPU addresses ([`L0HostBuffer`]).
//! - [`L0Context::begin_recording`] opens a command list of graph executions and copies; the closed
//!   [`L0Recording`] replays through [`L0Context::execute`] with a timeout.
//!
//! On NixOS the loader finds the drivers, IGC and the NPU compiler only through `LD_LIBRARY_PATH`;
//! `nix develop .#npu-intel` puts them there.

#![deny(clippy::undocumented_unsafe_blocks)]

mod buffer;
mod context;
mod error;
mod ffi;
#[cfg(feature = "test-support")]
mod gpu;
mod graph;
#[cfg(feature = "test-support")]
mod module;
mod platform;

pub use buffer::L0HostBuffer;
pub use context::{L0Context, L0DeviceKind};
pub use error::L0Error;
#[cfg(feature = "test-support")]
pub use gpu::{KernelArg, L0Buffer};
pub use graph::{
    GraphArgument, GraphDeviceProperties, GraphDirection, GraphIr, GraphPrecision, L0Graph,
    L0Recording, L0RecordingBuilder,
};
#[cfg(feature = "test-support")]
pub use module::{L0Kernel, L0Module};
