//! The one backend `match` above the executors (Card 734): [`open_executor`] turns a
//! [`BackendChoice`] into a `Box<dyn Executor>`. Nothing below it names a backend and nothing above it
//! branches on one: a caller parses its `--backend` flag into a [`BackendChoice`] and passes it here.

use std::fmt;
use std::num::NonZeroUsize;
use std::str::FromStr;

use poot_executor::{
    Coordinator, DetailedTiming, DeviceError, Engine, ExecError, Executor, TimingOptions,
};
use poot_graph_plan::multi_device::topology::DeviceId;
use poot_ptx_device::device::PtxDevice;

use crate::driver::error::{DriverError, InvalidOptions, Unsupported};

/// Which device executes the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendChoice {
    /// The wgpu backend, on whichever adapter wgpu selects.
    Wgpu,
    /// AMD GPUs through the HSA runtime.
    Rocm,
    /// NVIDIA GPUs: PTX kernels through the CUDA driver.
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

impl BackendChoice {
    /// Every choice, in the order its flag is documented.
    pub const ALL: [BackendChoice; 5] = [
        BackendChoice::Wgpu,
        BackendChoice::Rocm,
        BackendChoice::Ptx,
        BackendChoice::Vulkan,
        BackendChoice::Auto,
    ];

    /// The flag value that names this choice: what [`FromStr`] parses and [`fmt::Display`] prints.
    pub const fn name(self) -> &'static str {
        match self {
            BackendChoice::Wgpu => "wgpu",
            BackendChoice::Rocm => "rocm",
            BackendChoice::Ptx => "ptx",
            BackendChoice::Vulkan => "vulkan",
            BackendChoice::Auto => "auto",
        }
    }
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A `--backend` value that names no [`BackendChoice`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown backend {0:?}: expected wgpu, rocm, ptx, vulkan or auto")]
pub struct UnknownBackend(pub String);

/// The one parser of a backend name: every binary and the server take `--backend` through it.
impl FromStr for BackendChoice {
    type Err = UnknownBackend;

    fn from_str(name: &str) -> Result<Self, UnknownBackend> {
        match name {
            "wgpu" => Ok(BackendChoice::Wgpu),
            "rocm" => Ok(BackendChoice::Rocm),
            "ptx" => Ok(BackendChoice::Ptx),
            "vulkan" => Ok(BackendChoice::Vulkan),
            "auto" => Ok(BackendChoice::Auto),
            other => Err(UnknownBackend(other.to_string())),
        }
    }
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
        Timing::Counters => poot_wgpu_device::device::WgpuDevice::new(),
        Timing::Detailed(_) => {
            poot_wgpu_device::device::WgpuDevice::new_with_device_timing(timing.queries())
        }
    }
    .map_err(|e| open_failed("wgpu", e))?;
    Ok(engine(device, timing))
}

fn ptx(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_ptx_device::device::PtxDevice::new(),
        Timing::Detailed(_) => poot_ptx_device::device::PtxDevice::new_with_device_timing(),
    }
    .map_err(|e| open_failed("ptx", e))?;
    Ok(engine(device, timing))
}

fn vulkan() -> Result<Box<dyn Executor>, DriverError> {
    let device = poot_vulkan_device::VulkanDevice::new().map_err(|e| open_failed("vulkan", e))?;
    Ok(Box::new(Engine::new(device)))
}

fn rocm(timing: Timing) -> Result<Box<dyn Executor>, DriverError> {
    let device = match timing {
        Timing::Counters => poot_rocm_device::device::RocmDevice::new(),
        Timing::Detailed(_) => poot_rocm_device::device::RocmDevice::new_with_device_timing(),
    }
    .map_err(|e| open_failed("rocm", e))?;
    Ok(engine(device, timing))
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

#[cfg(test)]
mod name_tests {
    use super::*;

    /// Every choice round-trips through `Display` and `FromStr`, and the names are distinct.
    /// Mutation: drop a variant's arm from `from_str`; that choice stops parsing and the row goes red.
    #[test]
    fn every_backend_choice_round_trips_through_display_and_from_str() {
        for choice in BackendChoice::ALL {
            let name = choice.to_string();
            assert_eq!(name.parse::<BackendChoice>(), Ok(choice), "{name}");
        }
        let mut names: Vec<_> = BackendChoice::ALL.map(BackendChoice::name).to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), BackendChoice::ALL.len());
    }

    /// `ALL` lists every variant: the exhaustive match breaks the build when a variant is added, and the
    /// length pins the list to it.
    #[test]
    fn all_lists_every_variant() {
        let count = |choice: BackendChoice| match choice {
            BackendChoice::Wgpu
            | BackendChoice::Rocm
            | BackendChoice::Ptx
            | BackendChoice::Vulkan
            | BackendChoice::Auto => 1,
        };
        assert_eq!(BackendChoice::ALL.map(count).iter().sum::<usize>(), 5);
    }

    /// An unknown name is refused and reported verbatim, never read as a known backend.
    #[test]
    fn an_unknown_backend_name_is_refused_by_name() {
        let refused = "metal".parse::<BackendChoice>().unwrap_err();
        assert_eq!(refused, UnknownBackend("metal".to_string()));
        assert!(refused.to_string().contains("\"metal\""), "{refused}");
    }
}
