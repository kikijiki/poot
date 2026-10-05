//! Backend-specific runner dispatch.

pub(crate) mod gpu_generate;
pub(crate) mod ptx;
pub(crate) mod rocm_vulkan;
