//! The one backend `match` above the executors (Card 734): [`open_executor`] turns a
//! [`BackendChoice`] into a `Box<dyn Executor>`. Nothing below it names a backend and nothing above it
//! branches on one: a caller parses its `--backend` flag into a [`BackendChoice`] and passes it here.

use poot_executor::{DeviceError, Engine, ExecError, Executor};

use crate::driver::error::DriverError;
#[cfg(not(feature = "rocm"))]
use crate::driver::error::Unsupported;

/// Which device executes the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendChoice {
    Wgpu,
    Rocm,
    Ptx,
    /// Raw Vulkan through `ash`, with no wgpu: the same SPIR-V kernels as [`BackendChoice::Wgpu`] on the
    /// native command-buffer replay of ADR-0043. Never chosen by [`BackendChoice::Auto`]: it is the same
    /// hardware as wgpu's Vulkan adapter, so it is asked for by name.
    Vulkan,
    /// The first device that opens, in one documented order: PTX if a CUDA device opens, else ROCm if
    /// an HSA GPU agent opens, else wgpu. Chosen by device presence only, before any program exists:
    /// a planner refusal on the chosen device is never retried on another.
    Auto,
}

/// A device that failed to open, as the contract's one device-fault variant.
fn open_failed(
    backend: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> DriverError {
    DriverError::Device(ExecError::Device(Box::new(DeviceError {
        backend,
        source: Box::new(source),
    })))
}

fn wgpu() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_gpu::device::WgpuDevice::new().map_err(|e| open_failed("wgpu", e))?;
    Ok(Box::new(Engine::new(device)))
}

fn ptx() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_ptx_gpu::device::PtxDevice::new().map_err(|e| open_failed("ptx", e))?;
    Ok(Box::new(Engine::new(device)))
}

fn vulkan() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_vulkan_device::VulkanDevice::new().map_err(|e| open_failed("vulkan", e))?;
    Ok(Box::new(Engine::new(device)))
}

#[cfg(feature = "rocm")]
fn rocm() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_rocm_gpu::device::RocmDevice::new().map_err(|e| open_failed("rocm", e))?;
    Ok(Box::new(Engine::new(device)))
}

#[cfg(not(feature = "rocm"))]
fn rocm() -> Result<Box<dyn Executor>, DriverError> {
    Err(Unsupported::Backend { backend: "rocm" }.into())
}

pub fn open_executor(choice: BackendChoice) -> Result<Box<dyn Executor>, DriverError> {
    match choice {
        BackendChoice::Wgpu => wgpu(),
        BackendChoice::Rocm => rocm(),
        BackendChoice::Ptx => ptx(),
        BackendChoice::Vulkan => vulkan(),
        BackendChoice::Auto => ptx().or_else(|_| rocm()).or_else(|_| wgpu()),
    }
}
