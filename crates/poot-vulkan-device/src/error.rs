//! Error type for [`crate::device::VulkanDevice`].

use poot_vulkan_runtime::RuntimeError;
use thiserror::Error;

/// Errors from the raw-Vulkan executor [`Device`](poot_executor::Device) implementation.
#[derive(Debug, Error)]
pub enum VulkanGpuError {
    /// The runtime refused or failed a call: a Vulkan failure, a lost submission, a bad binding.
    #[error("vulkan runtime: {0}")]
    Runtime(#[source] Box<RuntimeError>),
    /// A kernel's `Assert`/`Unreachable` fired during the last replay: a program result, raised at
    /// `synchronize` (the contract's fault point), not a device failure.
    #[error("kernel {kernel:?} asserted (code {code})")]
    Fault { kernel: String, code: u32 },
}

impl From<RuntimeError> for VulkanGpuError {
    fn from(error: RuntimeError) -> Self {
        VulkanGpuError::Runtime(Box::new(error))
    }
}
