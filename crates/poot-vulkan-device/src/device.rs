//! `VulkanDevice`: the raw-Vulkan implementation of the executor contract's [`poot_executor::Device`]
//! (Card 553, ADR-0102 decision 3, ADR-0043), over [`poot_vulkan_runtime`]'s native command-buffer
//! record and replay.
//!
//! A recording is a [`VulkanGraph`]: `begin` opens a [`RecordingGraph`]; `dispatch` and `copy` each append
//! one command (a pipeline bind, a descriptor bind and `vkCmdDispatch`, or `vkCmdCopyBuffer`, behind a
//! memory barrier) and never submit; `finish` closes it. `replay` is the one method that touches the
//! queue: it resubmits the recorded command buffers verbatim, fence-waiting after each, so by the time
//! `replay` returns `Ok` the work is done (`synchronize` only raises the replay's staged kernel fault).
//!
//! A long recording is cut into several command buffers by the planner's `Dispatch::work` weight, so a
//! replay is several submissions with a fence wait between them, never one past-watchdog submission
//! (the same bounds `WgpuDevice` flushes on).
//!
//! `Eager` has no separate meaning here: `begin` always opens a recording, as every `Device` does, and
//! `Engine::add_entry` refuses `Submission::Eager` before any method of this type runs.
//!
//! `copy` is `vkCmdCopyBuffer` in the same command stream, ordered by the barrier every command carries.

use poot_executor::{
    Arg, BufferRole, Device, DeviceTime, Dispatch, DispatchDeviceTime, MeasuredDeviceTime,
    MemoryCounterSnapshot,
};
use poot_graph_plan::{Submission, Target};
use poot_runtime_common::CompiledKernel;
use poot_target::{Backend, BufferStorage};
use poot_vulkan_runtime::{
    Binding, Context, DeviceBuffer, GraphTiming, KernelFault, Pipeline, RecordingGraph, VulkanGraph,
};
use std::sync::Arc;

use crate::error::VulkanGpuError;

/// Dispatches one command buffer may hold before a replay submits it on its own, and the summed
/// planner work (`Dispatch::work`) it may hold: the watchdog-safe bounds `WgpuDevice` flushes a submit
/// on, a production default and not environment-tunable (ADR-0104 decision 5's precedent).
const FLUSH_EVERY: usize = 1024;
const FLUSH_WORK: u64 = 4_000_000;

/// One loaded kernel: the plan key it is cached and labelled under, and its pipeline.
pub struct VulkanKernel {
    key: String,
    pipeline: Arc<Pipeline>,
}

/// The step being recorded between `begin` and `finish`, and how much it holds since its last cut.
struct OpenStep {
    graph: RecordingGraph,
    commands_since_cut: usize,
    work_since_cut: u64,
}

impl OpenStep {
    /// Whether adding a command of `work` would push the open command buffer past either bound.
    fn would_overflow(&self, work: u64) -> bool {
        self.commands_since_cut >= FLUSH_EVERY
            || (self.commands_since_cut > 0 && self.work_since_cut + work > FLUSH_WORK)
    }
}

pub struct VulkanDevice {
    ctx: Context,
    timing: GraphTiming,
    /// The open step's recording so far; `Some` between `begin` and `finish`.
    open: Option<OpenStep>,
    /// The last replay's kernel assert, raised by the next `synchronize`.
    fault: Option<KernelFault>,
    /// The last replay's device time; `Unknown` until a timed replay measured it.
    time: DeviceTime,
}

impl VulkanDevice {
    /// A device in counters-only mode: replays write no timestamps and [`Device::device_time`] is
    /// `Unknown` (Card 552: the counters-only path asks nothing extra of the device).
    pub fn new() -> Result<Self, VulkanGpuError> {
        Self::with_timing(GraphTiming::Off)
    }

    /// A device for the typed device-timing path: every dispatch is bracketed by two device timestamps,
    /// read back after the replay's fence wait (no extra synchronization), and [`Device::device_time`]
    /// reports them.
    pub fn new_with_timing() -> Result<Self, VulkanGpuError> {
        Self::with_timing(GraphTiming::PerDispatch)
    }

    fn with_timing(timing: GraphTiming) -> Result<Self, VulkanGpuError> {
        let ctx = Context::new()?;
        if timing == GraphTiming::PerDispatch && !ctx.timestamps_supported() {
            return Err(poot_vulkan_runtime::RuntimeError::TimestampsUnavailable.into());
        }
        Ok(Self {
            ctx,
            timing,
            open: None,
            fault: None,
            time: DeviceTime::Unknown,
        })
    }

    /// Borrow the underlying context (diagnostic probes: device name, recording counters).
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Cut the open command buffer first if `work` would overflow it, then account for one command.
    fn make_room(&mut self, work: u64) -> Result<(), VulkanGpuError> {
        let step = self.open.as_mut().expect("command outside begin/finish");
        if step.would_overflow(work) {
            self.ctx.cut_graph(&mut step.graph)?;
            step.commands_since_cut = 0;
            step.work_since_cut = 0;
        }
        Ok(())
    }
}

impl Device for VulkanDevice {
    type Buffer = DeviceBuffer;
    type Kernel = VulkanKernel;
    type Recording = VulkanGraph;
    type Error = VulkanGpuError;

    fn target(&self) -> Target {
        Target {
            backend: Backend::SpirvVulkan,
            caps: self.ctx.device_caps(),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.ctx.memory()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<DeviceBuffer, VulkanGpuError> {
        // A zero-element value still owns one element: a Vulkan buffer holds at least one byte.
        Ok(self.ctx.alloc_storage(role, storage, elems.max(1))?)
    }

    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<VulkanKernel, VulkanGpuError> {
        Ok(VulkanKernel {
            key: key.to_string(),
            pipeline: self.ctx.pipeline(key, &kernel)?,
        })
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), VulkanGpuError> {
        // Always opens a recording, regardless of `_submission` (see the module doc). Recording never
        // submits, so a failed begin/dispatch/copy/finish can never have mutated device state.
        assert!(self.open.is_none(), "begin inside an open step");
        let graph = self.ctx.begin_graph(self.timing)?;
        self.open = Some(OpenStep {
            graph,
            commands_since_cut: 0,
            work_since_cut: 0,
        });
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), VulkanGpuError> {
        self.make_room(d.work)?;
        // Each argument's own element count from the plan, never the bound buffer's capacity (an arena
        // slot's buffer is sized to its largest occupant; Card 547b). The output is last.
        let args: Vec<Binding<'_>> = d
            .inputs
            .iter()
            .chain(std::iter::once(&d.output))
            .map(|a: &Arg<'_, Self>| Binding {
                buffer: a.buffer,
                elems: a.elems,
            })
            .collect();
        let step = self.open.as_mut().expect("dispatch outside begin/finish");
        self.ctx.record_dispatch(
            &mut step.graph,
            &d.kernel.key,
            &d.kernel.pipeline,
            d.workgroup,
            d.threads,
            &args,
        )?;
        step.commands_since_cut += 1;
        step.work_since_cut += d.work;
        Ok(())
    }

    fn copy(&mut self, src: &DeviceBuffer, dst: &DeviceBuffer) -> Result<(), VulkanGpuError> {
        self.make_room(0)?;
        let step = self.open.as_mut().expect("copy outside begin/finish");
        self.ctx.record_copy(&mut step.graph, src, dst)?;
        step.commands_since_cut += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<VulkanGraph>, VulkanGpuError> {
        let step = self.open.take().expect("finish without begin");
        Ok(Some(self.ctx.end_graph(step.graph)?))
    }

    fn replay(&mut self, recording: &VulkanGraph) -> Result<(), VulkanGpuError> {
        // Whatever the previous step staged is stale the moment a new replay starts.
        self.fault = None;
        self.time = DeviceTime::Unknown;
        let replay = recording.replay()?;
        self.fault = replay.fault;
        if let Some(time) = replay.time {
            self.time = DeviceTime::Measured(MeasuredDeviceTime {
                sum_of_dispatch_durations: Some(time.sum_of_dispatch_durations),
                device_span: Some(time.device_span),
                dispatches: time
                    .dispatches
                    .into_iter()
                    .map(|d| DispatchDeviceTime {
                        index: d.index,
                        duration: d.duration,
                    })
                    .collect(),
            });
        }
        Ok(())
    }

    fn write(&mut self, dst: &DeviceBuffer, bytes: &[u8]) -> Result<(), VulkanGpuError> {
        Ok(dst.write_bytes(bytes)?)
    }

    fn read(&mut self, src: &DeviceBuffer, out: &mut [u8]) -> Result<(), VulkanGpuError> {
        Ok(src.read_bytes(out)?)
    }

    fn synchronize(&mut self) -> Result<(), VulkanGpuError> {
        // `replay` already fence-waited every submission (see the module doc). What is left to raise
        // is the replay's kernel assert, at the point the contract classifies faults.
        match self.fault.take() {
            Some(KernelFault { kernel, code }) => Err(VulkanGpuError::Fault { kernel, code }),
            None => Ok(()),
        }
    }

    fn device_time(&self) -> DeviceTime {
        self.time.clone()
    }

    fn abort(&mut self) -> Result<(), VulkanGpuError> {
        // Recording never submits (see the module doc): dropping the attempted recording is the whole
        // cleanup. `begin` panics (not errors) on a double-open, so `self.open` is the only Rust-level
        // state abort must clear, plus whatever the failed step staged.
        self.open = None;
        self.fault = None;
        self.time = DeviceTime::Unknown;
        // CR30: a lost submission already poisoned the device, and no later submission can run. Reporting
        // that here is what lets `Engine` set its own `poisoned` flag instead of re-attempting (and
        // re-failing) every later call.
        if self.ctx.is_poisoned() {
            return Err(poot_vulkan_runtime::RuntimeError::Poisoned { op: "abort" }.into());
        }
        Ok(())
    }

    fn classify_fault(&self, error: &VulkanGpuError) -> Option<(String, u32)> {
        match error {
            VulkanGpuError::Fault { kernel, code } => Some((kernel.clone(), *code)),
            VulkanGpuError::Runtime(_) => None,
        }
    }
}
