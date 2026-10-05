//! Raw-Vulkan implementation of the executor contract's [`poot_executor::Device`] (Card 553, ADR-0102
//! decision 3): the Vulkan twin of `poot-gpu`'s `WgpuDevice`, `poot-rocm-gpu`'s `RocmDevice` and
//! `poot-ptx-gpu`'s `PtxDevice`. [`device::VulkanDevice`] is a thin backend mechanism - no graph walk, no
//! plan interpretation, no const or weight caching - over [`poot_vulkan_runtime::Context`]'s native
//! command-buffer record and replay (ADR-0043); `poot_executor::Engine<VulkanDevice>` does everything
//! else.

pub mod device;
mod error;

pub use device::{VulkanDevice, VulkanKernel};
pub use error::VulkanGpuError;
