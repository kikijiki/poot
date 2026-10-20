//! Backend-neutral timing records. The executor contract's typed device-timing path
//! (`poot-executor`) folds per-dispatch and per-step device spans into a [`TimingSnapshot`], and
//! [`Report`] renders a window of it. The earlier label-keyed per-context `Profiler` registry is gone
//! (Card 626): nothing kept feeding it once the pre-contract executors died.
//!
//! Depends on `poot-target` for the device identity a snapshot is keyed by.

mod roofline;
mod timing;
pub use roofline::{Ceiling, Fit, GB, Roofline};
pub use timing::{
    BuildIdentity, DeviceCoverage, DeviceWindow, DispatchTiming, EntryCounters, HostTiming, Mark,
    MeasuredDevice, OpBucket, Report, StepTiming, TimingSnapshot, WindowReport,
};
