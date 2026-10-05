//! AMD/ROCm implementation of the executor contract's [`poot_executor::Device`] (Card 548): the
//! AMD/HSA twin of [`poot_gpu::device::WgpuDevice`], targeting the Strix Halo iGPU (Radeon 8060S,
//! gfx1151) via the HSA API in [`poot_rocm_runtime`]. [`device::RocmDevice`] is a thin backend
//! mechanism - no graph walk, no plan interpretation, no const/weight caching - over
//! [`poot_rocm_runtime::RocmContext`]'s AQL record/replay primitives; `poot_executor::Engine<RocmDevice>`
//! does everything else.

pub mod device;
mod error;

pub use device::{RocmDevice, RocmKernel, RocmRecording};
pub use error::RocmGpuError;

// FNV-1a re-export (same primitive as `poot_rocm_runtime::fnv1a`) so callers can fingerprint kernels
// without depending on `poot-rocm-runtime` directly.
pub use poot_rocm_runtime::BufferRole;
pub use poot_rocm_runtime::RocmBuffer;
pub use poot_rocm_runtime::fnv1a as fingerprint;
