//! The one backend `match` above the executors (Card 734): [`open_executor`] turns a
//! [`BackendChoice`] into a `Box<dyn Executor>`. Nothing below it names a backend and nothing above it
//! branches on one: a caller parses its `--backend` flag into a [`BackendChoice`] and passes it here.

use std::num::NonZeroUsize;

use poot_executor::{DetailedTiming, DeviceError, Engine, ExecError, Executor, TimingOptions};

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

/// Whether the executor asks its device for device-side timestamps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Timing {
    /// Counters only: no device timestamps requested, no per-dispatch detail retained.
    Counters,
    /// Device timing on a backend that can produce it (wgpu timestamp queries, PTX CUDA events, ROCm
    /// HSA dispatch timestamps); a backend that cannot (Vulkan) opens as `Counters` and reports
    /// `DeviceTime::Unknown`.
    Device,
}

/// Bounds of the detail a profiled engine retains and of the timestamp queries a wgpu device keeps in flight.
const PROFILE_STEPS: usize = 4096;
const PROFILE_DISPATCH_RECORDS: usize = 1 << 16;
const PROFILE_QUERIES: NonZeroUsize = NonZeroUsize::new(64).unwrap();

fn engine<D: poot_executor::Device + 'static>(device: D, timing: Timing) -> Box<dyn Executor> {
    match timing {
        Timing::Counters => Box::new(Engine::new(device)),
        Timing::Device => Box::new(Engine::with_timing(
            device,
            TimingOptions::Detailed(DetailedTiming::every_step(
                PROFILE_STEPS,
                PROFILE_DISPATCH_RECORDS,
                PROFILE_QUERIES.get(),
            )),
        )),
    }
}

fn wgpu(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_gpu::device::WgpuDevice::new(),
        Timing::Device => poot_gpu::device::WgpuDevice::new_with_device_timing(PROFILE_QUERIES),
    }
    .map_err(|e| open_failed("wgpu", e))?;
    Ok(engine(device, timing))
}

fn ptx(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_ptx_gpu::device::PtxDevice::new(),
        Timing::Device => poot_ptx_gpu::device::PtxDevice::new_with_device_timing(),
    }
    .map_err(|e| open_failed("ptx", e))?;
    Ok(engine(device, timing))
}

fn vulkan() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_vulkan_device::VulkanDevice::new().map_err(|e| open_failed("vulkan", e))?;
    Ok(Box::new(Engine::new(device)))
}

#[cfg(feature = "rocm")]
fn rocm(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_rocm_gpu::device::RocmDevice::new(),
        Timing::Device => poot_rocm_gpu::device::RocmDevice::new_with_device_timing(),
    }
    .map_err(|e| open_failed("rocm", e))?;
    Ok(engine(device, timing))
}

#[cfg(not(feature = "rocm"))]
fn rocm(_timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    Err(Unsupported::Backend { backend: "rocm" }.into())
}

fn open(choice: BackendChoice, timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    match choice {
        BackendChoice::Wgpu => wgpu(timing),
        BackendChoice::Rocm => rocm(timing),
        BackendChoice::Ptx => ptx(timing),
        BackendChoice::Vulkan => vulkan(),
        BackendChoice::Auto => ptx(timing)
            .or_else(|_| rocm(timing))
            .or_else(|_| wgpu(timing)),
    }
}

pub fn open_executor(choice: BackendChoice) -> Result<Box<dyn Executor>, DriverError> {
    open(choice, Timing::Counters)
}

/// [`open_executor`] for a profiled run: the device requests timestamps where it can (wgpu, PTX, ROCm), and
/// the engine retains per-dispatch detail. Timing perturbs submission grouping, so a profiled run is
/// diagnostic, never throughput evidence. A backend with no device timing (Vulkan) opens as
/// [`open_executor`] does and its steps report `DeviceTime::Unknown`.
pub fn open_timed_executor(choice: BackendChoice) -> Result<Box<dyn Executor>, DriverError> {
    open(choice, Timing::Device)
}

#[cfg(all(test, not(feature = "rocm")))]
mod tests {
    use super::*;

    /// Without the `rocm` feature the ROCm choice is refused by name, and the message names the feature
    /// that builds it in. Mutation: drop the feature from the `Unsupported::Backend` message.
    #[test]
    fn a_rocm_choice_without_the_feature_names_the_feature_to_enable() {
        let Err(refused) = open_executor(BackendChoice::Rocm) else {
            panic!("a build without the rocm feature opened a ROCm executor");
        };
        assert!(
            refused.to_string().contains("--features rocm"),
            "got: {refused}"
        );
    }
}
