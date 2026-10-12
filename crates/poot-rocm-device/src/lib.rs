//! AMD/ROCm implementation of the executor contract's [`poot_executor::Device`] (Card 548): the
//! AMD/HSA twin of [`poot_wgpu_device::device::WgpuDevice`], targeting the Strix Halo iGPU (Radeon 8060S,
//! gfx1151) via the HSA API in [`poot_rocm_runtime`]. [`device::RocmDevice`] is a thin backend
//! mechanism - no graph walk, no plan interpretation, no const/weight caching - over
//! [`poot_rocm_runtime::RocmContext`]'s AQL record/replay primitives; `poot_executor::Engine<RocmDevice>`
//! does everything else.

pub mod device;
mod error;

pub use device::{RocmDevice, RocmKernel, RocmRecording};
pub use error::RocmGpuError;
