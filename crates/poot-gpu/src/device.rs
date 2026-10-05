//! `WgpuDevice`: the wgpu implementation of the executor contract's [`poot_executor::Device`] (Card
//! 546a, ADR-0003).
//!
//! A recording is the ADR-0003 re-encode list: every dispatch's pipeline, length buffer and bind
//! group are built once while recording, and `replay` re-encodes them into one submit per run of
//! dispatches. `begin` always opens a recording and never submits, regardless of `Submission`
//! (`Engine` refuses `Submission::Eager` at `add_entry`, before any
//! `Device` method ever runs, but `WgpuDevice` is a general `Device` impl a caller may still drive
//! directly, outside the contract - `poot-gpu/tests/executor_contract.rs`'s SC-015 fault-probe
//! row does exactly that); recording never touches the device (wgpu's `build_cached_dispatch` only
//! resolves metadata), so `begin`/`dispatch`/`copy`/`finish` failing can never have mutated device
//! state, and `abort` only needs to drop the attempted recording's buffered metadata.

use poot_executor::{
    Arg, BufferRole, Device, DeviceTime, Dispatch, DispatchDeviceTime, MeasuredDeviceTime,
};
use poot_graph_plan::{Submission, Target};
use poot_runtime::{
    BufferStorage, CachedDispatch, CompiledKernel, Context, DeviceBuffer, RuntimeError,
};
use poot_target::Backend;

/// One loaded kernel: its compiled code and the plan key the pipeline cache and labels use.
pub struct WgpuKernel {
    key: String,
    kernel: CompiledKernel,
}

enum Recorded {
    /// Each dispatch with its planner-owned watchdog work weight: `submit` splits a
    /// long run into several `submit_cached` calls once either bound is hit, the same constants
    /// `GpuExecutor::flush_bounds` used.
    Dispatches(Vec<(CachedDispatch, u64)>),
    Copy {
        src: DeviceBuffer,
        dst: DeviceBuffer,
    },
}

/// Card 546a (S46-11), moved from the pre-contract `GpuExecutor::flush_bounds`: the watchdog-safe
/// bounds one `submit_cached` call may cover. A production default, not environment-tunable (ADR-0104
/// decision 5's precedent).
const FLUSH_EVERY: usize = 1024;
const FLUSH_WORK: u64 = 4_000_000;

/// A recorded step.
pub struct WgpuRecording {
    items: Vec<Recorded>,
}

pub struct WgpuDevice {
    ctx: Context,
    /// The open step's recording so far; `Some` between `begin` and `finish`.
    open: Option<Vec<Recorded>>,
}

impl WgpuDevice {
    pub fn new() -> Result<Self, RuntimeError> {
        Ok(Self {
            ctx: Context::new()?,
            open: None,
        })
    }

    /// A device for the typed Card 552 timing path: a positive `max_in_flight_queries` requests
    /// per-dispatch device timestamps on every replay, bounded to that many concurrently un-drained
    /// query resources (review F3; retention of the *retained* records is the caller engine's job,
    /// via `TimingOptions`). `0` behaves exactly like [`Self::new`] (counters-only, no query
    /// resources ever created).
    pub fn new_with_timing(max_in_flight_queries: usize) -> Result<Self, RuntimeError> {
        Ok(Self {
            ctx: Context::new_with_device_timing(max_in_flight_queries)?,
            open: None,
        })
    }

    /// Diagnostic counters of the underlying context (bind groups, pipelines, submits).
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    fn submit(&self, item: &Recorded) -> Result<(), RuntimeError> {
        match item {
            Recorded::Dispatches(dispatches) => self.submit_chunked(dispatches),
            Recorded::Copy { src, dst } => self.ctx.copy_words_into(dst, src),
        }
    }

    /// Split `dispatches` into runs of at most `FLUSH_EVERY` dispatches and `FLUSH_WORK` summed work
    /// units, one `submit_cached` per run, polling between runs so in-flight work stays bounded
    /// (SC-008). A long prefill's one recorded run still issues as several real
    /// submits, never one past-watchdog submission.
    ///
    /// Card 552: always the one `submit_cached` method - whether it requests per-dispatch
    /// timestamps is `Context::device_timing`'s own decision, and the dispatch-index each chunk's
    /// records start at is `Context`'s own running counter (reset once per step by
    /// `Context::discard_device_timing`, called from `Self::replay` below), not a parameter this
    /// loop threads through.
    fn submit_chunked(&self, dispatches: &[(CachedDispatch, u64)]) -> Result<(), RuntimeError> {
        let mut start = 0;
        while start < dispatches.len() {
            let mut end = start;
            let mut work = 0u64;
            while end < dispatches.len()
                && end - start < FLUSH_EVERY
                && (end == start || work + dispatches[end].1 <= FLUSH_WORK)
            {
                work += dispatches[end].1;
                end += 1;
            }
            let refs: Vec<&CachedDispatch> =
                dispatches[start..end].iter().map(|(d, _)| d).collect();
            self.ctx.submit_cached(&refs)?;
            if end < dispatches.len() {
                self.ctx.poll_wait()?;
            }
            start = end;
        }
        Ok(())
    }

    fn push(&mut self, item: Recorded) -> Result<(), RuntimeError> {
        let items = self.open.as_mut().expect("dispatch outside begin/finish");
        match (items.last_mut(), item) {
            (Some(Recorded::Dispatches(run)), Recorded::Dispatches(mut more)) => {
                run.append(&mut more)
            }
            (_, item) => items.push(item),
        }
        Ok(())
    }
}

impl Device for WgpuDevice {
    type Buffer = DeviceBuffer;
    type Kernel = WgpuKernel;
    type Recording = WgpuRecording;
    type Error = RuntimeError;

    fn target(&self) -> Target {
        Target {
            backend: Backend::SpirvVulkan,
            caps: self.ctx.device_caps(),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, poot_executor::MemoryCounterSnapshot)> {
        self.ctx.memory()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<DeviceBuffer, RuntimeError> {
        Ok(self.ctx.alloc_storage(role, storage, elems))
    }

    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<WgpuKernel, RuntimeError> {
        Ok(WgpuKernel {
            key: key.to_string(),
            kernel,
        })
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), RuntimeError> {
        // Always opens a recording, regardless of `_submission` (see the module doc): the
        // contract's own restriction to Replay is enforced at `Engine::add_entry`, not here.
        assert!(self.open.is_none(), "begin inside an open step");
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), RuntimeError> {
        let inputs: Vec<&DeviceBuffer> =
            d.inputs.iter().map(|a: &Arg<'_, Self>| a.buffer).collect();
        // Card 547b: each argument's own logical element count from the plan, never
        // the bound buffer's capacity.
        let input_lens: Vec<u32> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.elems).collect();
        let cached = self.ctx.build_cached_dispatch(
            &d.kernel.key,
            &d.kernel.key,
            &d.kernel.kernel,
            d.workgroup,
            d.threads,
            &inputs,
            &input_lens,
            d.output.buffer,
            d.output.elems,
        )?;
        self.push(Recorded::Dispatches(vec![(cached, d.work)]))
    }

    fn copy(&mut self, src: &DeviceBuffer, dst: &DeviceBuffer) -> Result<(), RuntimeError> {
        self.push(Recorded::Copy {
            src: src.clone(),
            dst: dst.clone(),
        })
    }

    fn finish(&mut self) -> Result<Option<WgpuRecording>, RuntimeError> {
        let items = self.open.take().expect("finish without begin");
        Ok(Some(WgpuRecording { items }))
    }

    /// Card 552: discards any leftover pending/accumulated device timing from a previous
    /// step - one that failed after registering some but before `device_time()` ever drained it -
    /// before this step's own replay registers anything new. `replay` is the one `Device` method
    /// `Engine::step` calls unconditionally on every step (recording happens at most once per
    /// entry; replay happens every time), so this is "the start of every step" in practice.
    fn replay(&mut self, recording: &WgpuRecording) -> Result<(), RuntimeError> {
        self.ctx.discard_device_timing();
        recording
            .items
            .iter()
            .try_for_each(|item| self.submit(item))
    }

    fn write(&mut self, dst: &DeviceBuffer, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.ctx.write_bytes(dst, bytes)
    }

    fn read(&mut self, src: &DeviceBuffer, out: &mut [u8]) -> Result<(), RuntimeError> {
        self.ctx.read_bytes(src, out)
    }

    fn synchronize(&mut self) -> Result<(), RuntimeError> {
        self.ctx.flush_faults_and_wait()
    }

    /// Card 552: drains whatever `replay` registered this step (called strictly after
    /// `synchronize` in `Engine::step`, so every pending readback this drains is already mapped -
    /// no extra synchronization here). `Unknown` in counters-only mode (`device_timing` off) or
    /// when the step's replay dispatched nothing timed.
    fn device_time(&self) -> DeviceTime {
        match self.ctx.drain_device_timing() {
            None => DeviceTime::Unknown,
            Some(result) => DeviceTime::Measured(MeasuredDeviceTime {
                sum_of_dispatch_durations: Some(result.sum),
                device_span: Some(result.span),
                dispatches: result
                    .per_dispatch
                    .into_iter()
                    .map(|(index, duration)| DispatchDeviceTime { index, duration })
                    .collect(),
            }),
        }
    }

    fn abort(&mut self) -> Result<(), RuntimeError> {
        // Recording never touches the device (see the module doc); an eager walk's already-submitted
        // dispatches cannot be undone, but wgpu raises no device-level "unrecoverable" signal at this
        // granularity, so dropping the attempted recording's buffered metadata is always a safe,
        // reusable state. `begin` panics (not errors) on a double-open, so `self.open` is the only
        // state abort must clear.
        self.open = None;
        // Card 552: defense in depth alongside `replay`'s own discard - a failed replay
        // (e.g. a mid-chunk submit error) reaches here too, and this guarantees no pending timing
        // survives into whatever runs next even if a future caller ever drives `replay` without
        // going through `Engine::step`.
        self.ctx.discard_device_timing();
        Ok(())
    }

    fn classify_fault(&self, error: &RuntimeError) -> Option<(String, u32)> {
        match error {
            RuntimeError::KernelAssertFailed { kernel, code } => Some((kernel.clone(), *code)),
            _ => None,
        }
    }
}
