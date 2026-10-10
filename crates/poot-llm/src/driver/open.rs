//! The one backend `match` above the executors (Card 734): [`open_executor`] turns a
//! [`BackendChoice`] into a `Box<dyn Executor>`. Nothing below it names a backend and nothing above it
//! branches on one: a caller parses its `--backend` flag into a [`BackendChoice`] and passes it here.

use std::num::NonZeroUsize;

use poot_executor::{
    Coordinator, DetailedTiming, DeviceError, Engine, ExecError, Executor, TimingOptions,
};
use poot_graph_plan::DeviceId;
use poot_ptx_gpu::device::PtxDevice;

use crate::driver::error::{DriverError, InvalidOptions, Unsupported};

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

/// How the executor is built: counters only, or with device timestamps and the bounded detail
/// [`DetailedTiming`] names.
#[derive(Clone, Copy)]
enum Timing {
    /// Counters only: no device timestamps requested, no per-dispatch detail retained.
    Counters,
    /// Device timing on a backend that can produce it (wgpu timestamp queries, PTX CUDA events, ROCm
    /// HSA dispatch timestamps); a backend that cannot (Vulkan) opens as `Counters` and reports
    /// `DeviceTime::Unknown`.
    Detailed(DetailedTiming),
}

impl Timing {
    /// The in-flight timestamp-query bound a wgpu device is built with. [`open_timed_executor`] has
    /// refused a zero bound before any device opens, so a detailed run always has a positive one.
    fn queries(self) -> NonZeroUsize {
        match self {
            Timing::Counters => NonZeroUsize::MIN,
            Timing::Detailed(detail) => NonZeroUsize::new(detail.max_in_flight_queries)
                .expect("a detailed timing's bounds were validated when it was opened"),
        }
    }
}

/// Refuse a detailed timing whose bound is zero: it would disable what it bounds, and an invalid limit is
/// an error, never repaired.
fn check_bounds(detail: &DetailedTiming) -> Result<(), DriverError> {
    let bounds = [
        ("sample_every", detail.sample_every as usize),
        ("max_retained_steps", detail.max_retained_steps),
        (
            "max_retained_dispatch_records",
            detail.max_retained_dispatch_records,
        ),
        ("max_in_flight_queries", detail.max_in_flight_queries),
    ];
    match bounds.into_iter().find(|&(_, value)| value == 0) {
        Some((bound, _)) => Err(InvalidOptions::TimingBoundZero { bound }.into()),
        None => Ok(()),
    }
}

fn engine<D: poot_executor::Device + 'static>(device: D, timing: Timing) -> Box<dyn Executor> {
    match timing {
        Timing::Counters => Box::new(Engine::new(device)),
        Timing::Detailed(detail) => {
            Box::new(Engine::with_timing(device, TimingOptions::Detailed(detail)))
        }
    }
}

fn wgpu(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_gpu::device::WgpuDevice::new(),
        Timing::Detailed(_) => {
            poot_gpu::device::WgpuDevice::new_with_device_timing(timing.queries())
        }
    }
    .map_err(|e| open_failed("wgpu", e))?;
    Ok(engine(device, timing))
}

fn ptx(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_ptx_gpu::device::PtxDevice::new(),
        Timing::Detailed(_) => poot_ptx_gpu::device::PtxDevice::new_with_device_timing(),
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
        Timing::Detailed(_) => poot_rocm_gpu::device::RocmDevice::new_with_device_timing(),
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

/// Open the executor `choice` names with cumulative counters only: no device timestamps and no
/// per-dispatch detail retained.
pub fn open_executor(choice: BackendChoice) -> Result<Box<dyn Executor>, DriverError> {
    open(choice, Timing::Counters)
}

/// [`open_executor`] with device timing: the device requests timestamps where it can (wgpu, PTX, ROCm)
/// and the engine retains per-dispatch detail inside `detail`'s bounds. Timing perturbs submission
/// grouping, so a timed run is diagnostic, never throughput evidence. A zero bound in `detail` is refused
/// with [`InvalidOptions::TimingBoundZero`] before any device opens. A backend with no device timing
/// (Vulkan) opens as [`open_executor`] does and its steps report `DeviceTime::Unknown`.
pub fn open_timed_executor(
    choice: BackendChoice,
    detail: DetailedTiming,
) -> Result<Box<dyn Executor>, DriverError> {
    check_bounds(&detail)?;
    open(choice, Timing::Detailed(detail))
}

/// Open GPU `ordinals[i]` as rank `i` of one coordinator (`DeviceId(i)`), for a multi-device
/// [`Placement`](crate::driver::Placement). Only a backend whose devices copy to one another can: today
/// that is PTX, which `Auto` resolves to here without trying another, since a group is never split
/// across backends. A GPU that fails to open is the error; nothing is opened on a backend that cannot
/// group.
pub fn open_group(
    choice: BackendChoice,
    ordinals: &[usize],
) -> Result<Coordinator<PtxDevice>, DriverError> {
    match choice {
        BackendChoice::Ptx | BackendChoice::Auto => {}
        BackendChoice::Wgpu => return Err(Unsupported::MultiDevice { backend: "wgpu" }.into()),
        BackendChoice::Rocm => return Err(Unsupported::MultiDevice { backend: "rocm" }.into()),
        BackendChoice::Vulkan => {
            return Err(Unsupported::MultiDevice { backend: "vulkan" }.into());
        }
    }
    let mut devices = Vec::with_capacity(ordinals.len());
    for (rank, &ordinal) in ordinals.iter().enumerate() {
        let device = PtxDevice::open(ordinal).map_err(|e| open_failed("ptx", e))?;
        devices.push((DeviceId(rank as u32), device));
    }
    Coordinator::new(devices).map_err(|e| DriverError::Device(ExecError::from(e)))
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    /// A zero timing bound is refused with its name, before a device opens, and is not repaired to one.
    /// Mutation: clamp the bound (`.max(1)`) instead of refusing; the row stops being an error and goes red.
    #[test]
    fn a_zero_timing_bound_is_refused_by_name() {
        for (bound, detail) in [
            ("max_in_flight_queries", DetailedTiming::every_step(8, 8, 0)),
            ("max_retained_steps", DetailedTiming::every_step(0, 8, 8)),
            (
                "max_retained_dispatch_records",
                DetailedTiming::every_step(8, 0, 8),
            ),
            (
                "sample_every",
                DetailedTiming {
                    sample_every: 0,
                    ..DetailedTiming::every_step(8, 8, 8)
                },
            ),
        ] {
            let Err(refused) = open_timed_executor(BackendChoice::Wgpu, detail) else {
                panic!("a zero {bound} opened an executor");
            };
            assert!(
                matches!(
                    refused,
                    DriverError::InvalidOptions(InvalidOptions::TimingBoundZero { bound: named })
                        if named == bound
                ),
                "{bound}: got {refused}"
            );
        }
    }
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
