//! Backend-neutral timing records. The executor contract's typed device-timing path
//! (`poot-executor`) folds per-dispatch and per-step device spans into a [`TimingSnapshot`], and
//! [`Report`] renders a window of it. The earlier label-keyed per-context `Profiler` registry is gone
//! (Card 626): nothing kept feeding it once the pre-contract executors died.
//!
//! Depends on std only.

mod timing;
pub use timing::{
    Coverage, DeviceCoverage, DeviceWindow, DispatchTiming, EntryCounters, HostTiming, Mark,
    MeasuredDevice, OpBucket, Report, StepTiming, TimingSnapshot, WindowReport,
};
