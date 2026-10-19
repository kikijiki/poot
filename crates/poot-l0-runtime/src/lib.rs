//! Level Zero for Intel devices, through the `dlopen`ed loader (`libze_loader.so.1`). The Intel analogue of
//! `poot-rocm-runtime`: every entry point is resolved with `dlsym`, so the crate builds without a Level Zero
//! installation, and a machine without one reports [`L0Error::LibraryNotFound`] or [`L0Error::NoDevice`] at
//! runtime. Only the NPU (VPU) is driven: there is no Level Zero GPU path.
//!
//! # NPU: vendor-compiled graphs (`ZE_extension_graph`)
//!
//! An Intel NPU has no kernel ISA; its only entry point is the driver's graph compiler.
//!
//! - [`L0Context::open`]: `zeInit` for the VPU drivers, pick the first driver with a device, create a context
//!   and a queue on the device's compute queue group.
//! - [`L0Context::compile_graph`] turns OpenVINO IR ([`GraphIr`]) into an [`L0Graph`];
//!   [`L0Graph::native_binary`] returns the compiled blob and [`L0Context::load_native_graph`] loads one
//!   without compiling again (`unsafe`: the driver does not validate a blob). [`L0Graph::arguments`] is what
//!   the compiler produced, for a caller to check.
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
mod graph;
mod platform;

pub use buffer::L0HostBuffer;
pub use context::L0Context;
pub use error::L0Error;
pub use graph::{
    GraphArgument, GraphDeviceProperties, GraphDirection, GraphIr, GraphPrecision, L0Graph,
    L0Recording, L0RecordingBuilder,
};
