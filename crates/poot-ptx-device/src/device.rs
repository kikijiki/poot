//! `PtxDevice`: the PTX implementation of the executor contract's [`poot_executor::Device`] (Card
//! 549, mirroring `poot_wgpu_device::device::WgpuDevice`, ADR-0003).
//!
//! A recording is a CUDA graph (G3c): `begin` opens a stream capture
//! ([`PtxContext::begin_capture`]), `dispatch`/`copy` enqueue onto the capturing stream (recorded,
//! not run - `PtxContext::dispatch_dev` already records during an open capture), `finish` ends the
//! capture into a replayable [`PtxGraphExec`] (via [`PtxContext::end_capture`]), and `replay` is one
//! `cuGraphLaunch` ([`PtxContext::launch_timed`]). `begin` always opens a capture and never submits.
//!
//! `Device::copy` is a native device-to-device copy node recorded into the same capture
//! ([`PtxContext::copy_buffer`], `cuMemcpyDtoDAsync` on the capturing stream), ordered like any dispatch
//! and byte-exact for any length and element kind: the device generates no kernel.

use poot_executor::{Arg, Device, DeviceTime, Dispatch, LaunchKind, MeasuredDeviceTime};
use poot_graph_plan::Target;
use poot_ptx_runtime::{PtxBuffer, PtxContext, PtxError, PtxGraphExec, PtxSpan};
use poot_runtime_common::{BufferRole, CompiledKernel, DeviceIdentity};
use poot_target::{Backend, BufferStorage, TargetId};

use crate::PtxGpuError;

/// One loaded kernel: its compiled code and the plan key the native function cache and dispatch
/// labels use.
pub struct PtxKernel {
    key: String,
    compiled: CompiledKernel,
}

/// A recorded step: the captured CUDA graph `replay` launches.
pub struct PtxRecording {
    exec: PtxGraphExec,
}

/// The NVIDIA/PTX implementation of [`poot_executor::Device`]: owns one [`PtxContext`] (one CUDA
/// context on one GPU), records a step as a CUDA graph and replays it. A multi-device program holds
/// one per GPU.
pub struct PtxDevice {
    ctx: PtxContext,
    /// The open step's device-timed spans (Card 552 SC-006), one per [`Self::replay`] since
    /// [`Self::begin_step`], read by [`Self::device_time`]; empty when this device was not built with
    /// [`PtxContext::new_with_device_timing`], or no step has replayed yet.
    pending_spans: Vec<PtxSpan>,
}

impl PtxDevice {
    /// A device on the default GPU, with no device timing; see [`Self::open`] for a chosen ordinal.
    pub fn new() -> Result<Self, PtxGpuError> {
        Ok(Self::with_context(PtxContext::new()?))
    }

    /// Like [`Self::new`], but every graph replay is bracketed by a CUDA event pair, so
    /// [`Device::device_time`] is `Measured` (the device span; PTX has no per-dispatch timing inside a
    /// captured graph).
    pub fn new_with_device_timing() -> Result<Self, PtxGpuError> {
        Ok(Self::with_context(PtxContext::new_with_device_timing()?))
    }

    /// A device on GPU `ordinal`: one of the devices a [`poot_executor::Coordinator`] drives. Several
    /// may live on one host thread.
    pub fn open(ordinal: usize) -> Result<Self, PtxGpuError> {
        Ok(Self::with_context(PtxContext::open(ordinal)?))
    }

    fn with_context(ctx: PtxContext) -> Self {
        Self {
            ctx,
            pending_spans: Vec::new(),
        }
    }

    /// Diagnostic access to the underlying context (e.g. `profiler()`).
    pub fn context(&self) -> &PtxContext {
        &self.ctx
    }
}

impl Device for PtxDevice {
    type Buffer = PtxBuffer;
    type Kernel = PtxKernel;
    type Recording = PtxRecording;
    type Error = PtxGpuError;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        poot_executor::CopyMechanism::Native
    }
    fn backend_name(&self) -> &'static str {
        "ptx"
    }

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::Nvptx,
            caps: self.ctx.device_caps(),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.ctx.memory()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        self.ctx.vram_free_bytes()
    }

    fn identity(&self) -> Option<DeviceIdentity> {
        self.ctx.identity()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<PtxBuffer, PtxGpuError> {
        Ok(self.ctx.alloc_storage(role, storage, elems)?)
    }

    fn buffer_elems(&self, buffer: &Self::Buffer) -> usize {
        buffer.elem_count() as usize
    }

    fn load_kernel(&mut self, key: &str, kernel: CompiledKernel) -> Result<PtxKernel, PtxGpuError> {
        Ok(PtxKernel {
            key: key.to_string(),
            compiled: kernel,
        })
    }

    fn begin(&mut self) -> Result<(), PtxGpuError> {
        // Always opens a capture (see the module doc).
        Ok(self.ctx.begin_capture()?)
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), PtxGpuError> {
        // A thread-grid device: an AIE or NPU invocation has no grid to run, and its kernel was never loadable here.
        let (threads, workgroup) = match d.launch {
            LaunchKind::Grid { threads, workgroup } => (threads, workgroup),
            LaunchKind::Aie(invocation) => {
                return Err(PtxError::WrongKernelTarget {
                    actual: TargetId::Aie(invocation.target()),
                }
                .into());
            }
            LaunchKind::VpuGraph(invocation) => {
                return Err(PtxError::WrongKernelTarget {
                    actual: TargetId::Vpu(invocation.target()),
                }
                .into());
            }
        };
        // Card 547b: every argument's own logical element count from the plan, never
        // the bound buffer's capacity (an arena slot's buffer is sized to its largest occupant,
        // which can exceed this particular dispatch's operand).
        let ins: Vec<&PtxBuffer> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.buffer).collect();
        let in_lens: Vec<u32> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.elems).collect();
        Ok(self.ctx.dispatch_dev(
            &d.kernel.key,
            &d.kernel.compiled,
            workgroup,
            threads,
            &ins,
            &in_lens,
            d.output.buffer,
            d.output.elems,
        )?)
    }

    fn copy(&mut self, src: &PtxBuffer, dst: &PtxBuffer) -> Result<(), PtxGpuError> {
        Ok(self.ctx.copy_buffer(src, dst)?)
    }

    fn finish(&mut self) -> Result<PtxRecording, PtxGpuError> {
        let exec = self.ctx.end_capture()?;
        Ok(PtxRecording { exec })
    }

    fn begin_step(&mut self) -> Result<(), PtxGpuError> {
        self.pending_spans.clear();
        Ok(())
    }

    fn replay(&mut self, recording: &PtxRecording) -> Result<(), PtxGpuError> {
        self.pending_spans
            .extend(self.ctx.launch_timed(&recording.exec)?);
        Ok(())
    }

    fn write_with(
        &mut self,
        dst: &PtxBuffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), PtxGpuError> {
        Ok(self.ctx.write_with(dst, offset, len, fill)?)
    }

    fn read(&mut self, src: &PtxBuffer, out: &mut [u8]) -> Result<(), PtxGpuError> {
        Ok(self.ctx.read_bytes(src, out)?)
    }

    fn synchronize(&mut self) -> Result<(), PtxGpuError> {
        Ok(self.ctx.synchronize()?)
    }

    /// Card 552: the window [`Self::replay`] bracketed around the step's `cuGraphLaunch`es (first
    /// stage's start to last stage's end), read once the caller has synchronized (`Engine::step`
    /// always does, strictly before calling this). `Unknown` when this device was not built with
    /// [`PtxContext::new_with_device_timing`] (SC-006: never a fabricated host-wall measurement), or
    /// if reading the event pair fails.
    fn device_time(&self) -> DeviceTime {
        match self.ctx.spans_elapsed(&self.pending_spans) {
            Ok(Some(duration)) => DeviceTime::Measured(MeasuredDeviceTime {
                sum_of_dispatch_durations: None,
                device_span: Some(duration),
                dispatches: Vec::new(),
            }),
            Ok(None) | Err(_) => DeviceTime::Unknown,
        }
    }

    fn abort(&mut self) -> Result<(), PtxGpuError> {
        self.ctx.abort_capture();
        // Defense in depth, as `WgpuDevice::abort`'s identical comment explains: a failed replay
        // (e.g. a mid-launch error) reaches here too, and this guarantees no pending span survives
        // into whatever runs next.
        self.pending_spans.clear();
        // A lost device can never run work again. Reporting that here, rather than succeeding, lets
        // the engine poison itself and refuse every later call instead of re-failing each one.
        Ok(self.ctx.ensure_healthy()?)
    }
}
