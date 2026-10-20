//! `RocmDevice`: the ROCm/HSA implementation of the executor contract's [`poot_executor::Device`]
//! (Card 548, ADR-0003), over [`poot_rocm_runtime`]'s AQL record/replay primitives.
//!
//! A recording is a list of [`CapturedAql`] packets. `begin` opens it; `dispatch`/`copy` each build a
//! kernarg buffer and append one packet, compiling the kernel on first use and never touching the
//! device (no HSA dispatch, no wait); `finish` hands the list to the caller as a [`RocmRecording`].
//! `replay` is the one method that submits to hardware: it batches every packet into one
//! [`poot_rocm_runtime::RocmContext::enqueue_graph_batched`] call, which rings the doorbell once per
//! chunk and returns without waiting. A step's stages enqueue back to back (each batch orders after
//! the last behind the AQL barrier), and `synchronize` is the one host wait, on the newest batch's
//! completion signal.
//!
//! `Device::copy` has no native ROCm primitive to record into the same AQL stream (HSA's async-copy
//! engine is a separate queue with its own signal, not an AQL packet this recording's BARRIER-ordered
//! replay could sequence against), so it dispatches the engine's copy kernel instead
//! ([`CopyMechanism::Kernel`]): one thread per `u32` word, dtype-agnostic, so every copy is sized by the
//! buffers' byte capacity (`byte_capacity / 4` words) and never by their logical element count. The
//! runtime allocates whole words, so a buffer of an odd number of bytes owns the word its last byte is in
//! and a word copy covers it. The device compiles nothing: the engine owns the kernel and hands it over
//! through [`Device::copy_mechanism`].

use poot_executor::{
    Arg, CopyKernel, CopyKernelSink, CopyMechanism, Device, DeviceTime, Dispatch,
    DispatchDeviceTime, LaunchKind, MeasuredDeviceTime,
};
use poot_graph_plan::Target;
use poot_rocm_runtime::{
    BoundKernarg, DispatchSpan, KernelHandle, RocmBuffer, RocmContext, RocmContextOptions,
    RocmError,
};
use poot_runtime_common::{BufferRole, CompiledKernel, DeviceIdentity, MemoryCounterSnapshot};
use poot_target::{AmdArch, Backend, BufferStorage, ElementKind, TargetId};

use crate::error::RocmGpuError;

/// One loaded kernel: its handle shares the HSA executable, so the code object stays loaded for as
/// long as any handle (or recording holding one) does.
pub struct RocmKernel {
    kernel: KernelHandle,
}

/// One recorded AQL dispatch. A recording owns what it addresses: the kernel handle keeps the code
/// object loaded and the bound kernarg keeps the kernarg segment and every data buffer whose device
/// address it holds, so neither can be freed while the recording (or a batch it enqueued) lives.
struct CapturedAql {
    kernel: KernelHandle,
    kernarg: BoundKernarg,
    elements: Box<[ElementKind]>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
}

/// One flattened entry of [`poot_rocm_runtime::RocmContext::enqueue_graph_batched`]'s dispatch list.
type BatchedDispatch = (KernelHandle, BoundKernarg, (u32, u32, u32), (u32, u32, u32));

/// A recorded step: the AQL packets captured between `begin` and `finish`, which `replay` re-enqueues
/// on the context's queue.
pub struct RocmRecording {
    aqls: Vec<CapturedAql>,
}

/// The ROCm (HSA/AQL) implementation of [`poot_executor::Device`]: owns one [`RocmContext`], records a
/// step as a list of captured AQL dispatch packets and replays it. A multi-device program holds one
/// per GPU.
pub struct RocmDevice {
    ctx: RocmContext,
    arch: AmdArch,
    /// The open step's recording so far; `Some` between `begin` and `finish` (mirrors
    /// `poot_wgpu_device::device::WgpuDevice`).
    open: Option<Vec<CapturedAql>>,
    /// The engine's copy kernel and its compiled workgroup shape (see the module doc), once
    /// the engine has loaded it ([`Device::copy_mechanism`]): `Device::copy` takes no kernel or launch shape, so this
    /// is the only place either is known.
    copy: Option<CopyLaunch>,
}

/// The loaded copy kernel and the workgroup shape it was compiled for.
struct CopyLaunch {
    kernel: RocmKernel,
    workgroup: [u32; 3],
}

/// Receives the engine's copy kernel into the device's copy slot ([`Device::copy_mechanism`]).
struct CopyLoader<'a> {
    ctx: &'a RocmContext,
    slot: &'a mut Option<CopyLaunch>,
}

impl CopyKernelSink<RocmGpuError> for CopyLoader<'_> {
    fn load(&mut self, copy: CopyKernel) -> Result<(), RocmGpuError> {
        let module = self.ctx.load_hsaco(copy.kernel())?;
        let kernel = self
            .ctx
            .lookup_kernel(&module, copy.kernel().entry_point())?;
        *self.slot = Some(CopyLaunch {
            kernel: RocmKernel { kernel },
            workgroup: copy.workgroup(),
        });
        Ok(())
    }
}

impl RocmDevice {
    /// A device on the default GPU, with no device timing; see [`Self::new_with_device_timing`].
    pub fn new() -> Result<Self, RocmGpuError> {
        Self::from_context(RocmContext::new()?)
    }

    /// Like [`Self::new`], but the queue's profiler is on and every replayed dispatch carries its own
    /// completion signal, so [`Device::device_time`] is `Measured` from the dispatches' HSA
    /// timestamps. The per-packet signals cost one create, read and destroy per dispatch, so a timed
    /// device is diagnostic, not throughput evidence.
    pub fn new_with_device_timing() -> Result<Self, RocmGpuError> {
        Self::from_context(RocmContext::new_with_options(RocmContextOptions {
            device_timing: true,
            ..RocmContextOptions::default()
        })?)
    }

    /// A device over an already-opened `ctx`: derives the GPU architecture from the context's ISA name.
    ///
    /// # Errors
    ///
    /// [`RocmGpuError::Arch`] when the ISA name carries no `gfx` code: kernels are compiled for the
    /// architecture the GPU reports, never for an assumed one.
    pub fn from_context(ctx: RocmContext) -> Result<Self, RocmGpuError> {
        let arch = AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())?;
        Ok(Self {
            ctx,
            arch,
            open: None,
            copy: None,
        })
    }

    /// Borrow the underlying context (diagnostic probes: VRAM queries, device caps).
    pub fn context(&self) -> &RocmContext {
        &self.ctx
    }

    /// The open recording, or [`RocmGpuError::NotRecording`] for `op`.
    fn recording(&mut self, op: &'static str) -> Result<&mut Vec<CapturedAql>, RocmGpuError> {
        self.open.as_mut().ok_or(RocmGpuError::NotRecording { op })
    }

    /// Bind one kernarg of flat `(ptr: u64, len: u64)` pairs, `[in0, in1, ..., out]` (the
    /// output last; a launch metadata buffer is already the last entry of `inputs` by the
    /// time `Engine::walk` calls `dispatch`, so there is no separate meta slot to carry here - mirrors
    /// `poot-wgpu-device`'s single flat kernarg layout, ROCm's own flavor of "one dispatch, one buffer").
    /// `len`'s unit is the kernel body's own: a generated kernel's typed slice parameters (`&[f32]`,
    /// `&[i32]`, ...) take the buffer's logical element count (`RocmBuffer::elem_count`), but
    /// the copy kernel's `&[u32]` parameters take the buffer's u32 WORD count instead - never the same
    /// number for an element narrower than 4 bytes (bf16/f16/i8: `elem_count` over-reports by
    /// 2x/4x; `poot-target::ElementKind` has no element wider than 4 bytes today, so there is no
    /// under-report case to name). Callers pass each buffer's length explicitly so this one writer
    /// serves both without silently assuming every element is 4 bytes wide; `copy` computes the
    /// copy-specific one. The result owns a clone of every bound buffer.
    fn bind_kernarg(
        &self,
        kernel: &KernelHandle,
        in_bufs: &[(&RocmBuffer, u64)],
        out_buf: (&RocmBuffer, u64),
    ) -> Result<BoundKernarg, RocmGpuError> {
        let slots: Vec<(&RocmBuffer, u64)> = in_bufs.iter().copied().chain([out_buf]).collect();
        Ok(self.ctx.bind_kernarg(kernel, &slots)?)
    }
}

/// `grid[i] = ceil(threads[i] / workgroup[i]) * workgroup[i]`, `block[i] = workgroup[i]`: every AQL
/// dispatch here is padded up to a whole number of workgroups per dimension (mirrors the pre-contract
/// executor's `one_dimensional_launch`, generalized to all 3 dims). Every kernel body this crate
/// dispatches carries its own `i < len` tail guard, so padded threads never read or write past a
/// buffer's real length.
fn padded_launch(threads: [u32; 3], workgroup: [u32; 3]) -> ((u32, u32, u32), (u32, u32, u32)) {
    let dim = |i: usize| {
        let wg = workgroup[i].max(1);
        threads[i].max(1).div_ceil(wg).saturating_mul(wg)
    };
    (
        (dim(0), dim(1), dim(2)),
        (
            workgroup[0].max(1),
            workgroup[1].max(1),
            workgroup[2].max(1),
        ),
    )
}

impl Device for RocmDevice {
    type Buffer = RocmBuffer;
    type Kernel = RocmKernel;
    type Recording = RocmRecording;
    type Error = RocmGpuError;

    fn copy_mechanism(&mut self) -> CopyMechanism<'_, RocmGpuError> {
        CopyMechanism::Kernel(Box::new(CopyLoader {
            ctx: &self.ctx,
            slot: &mut self.copy,
        }))
    }

    fn backend_name(&self) -> &'static str {
        "rocm"
    }

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::AmdGcn(self.arch),
            caps: self.ctx.device_caps(),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
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
    ) -> Result<RocmBuffer, RocmGpuError> {
        Ok(self.ctx.allocate(role, storage, elems)?)
    }

    fn buffer_elems(&self, buffer: &Self::Buffer) -> usize {
        buffer.elem_count() as usize
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        kernel: CompiledKernel,
    ) -> Result<RocmKernel, RocmGpuError> {
        let module = self.ctx.load_hsaco(&kernel)?;
        let handle = self.ctx.lookup_kernel(&module, kernel.entry_point())?;
        Ok(RocmKernel { kernel: handle })
    }

    fn begin(&mut self) -> Result<(), RocmGpuError> {
        // Always opens a recording (see the module doc, mirroring `WgpuDevice::begin`). Recording never touches the device (building a kernarg
        // buffer is a host write into host-visible kernarg memory, not a dispatch), so a failed
        // begin/dispatch/copy/finish can never have mutated device state.
        if self.open.is_some() {
            return Err(RocmGpuError::AlreadyRecording);
        }
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), RocmGpuError> {
        self.recording("dispatch")?;
        // A thread-grid device: an AIE or NPU invocation has no grid to run, and its kernel was never loadable here.
        let (threads, workgroup) = match d.launch {
            LaunchKind::Grid { threads, workgroup } => (threads, workgroup),
            LaunchKind::Aie(invocation) => {
                return Err(RocmError::WrongKernelTarget {
                    actual: TargetId::Aie(invocation.target()),
                }
                .into());
            }
            LaunchKind::VpuGraph(invocation) => {
                return Err(RocmError::WrongKernelTarget {
                    actual: TargetId::Vpu(invocation.target()),
                }
                .into());
            }
        };
        let in_bufs: Vec<&RocmBuffer> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.buffer).collect();
        // Card 547b: each argument's own logical element count from the plan, never
        // the bound buffer's `elem_count()` (an arena slot's buffer is sized to its largest
        // occupant, which can exceed this particular dispatch's operand).
        let in_pairs: Vec<(&RocmBuffer, u64)> = d
            .inputs
            .iter()
            .map(|a: &Arg<'_, Self>| (a.buffer, u64::from(a.elems)))
            .collect();
        let out_buf = d.output.buffer;
        let kernarg = self.bind_kernarg(
            &d.kernel.kernel,
            &in_pairs,
            (out_buf, u64::from(d.output.elems)),
        )?;
        let mut elements: Vec<ElementKind> = in_bufs.iter().map(|b| b.element()).collect();
        elements.push(d.output.buffer.element());
        let (grid, block) = padded_launch(threads, workgroup);
        self.recording("dispatch")?.push(CapturedAql {
            kernel: d.kernel.kernel.clone(),
            kernarg,
            elements: elements.into_boxed_slice(),
            grid,
            block,
        });
        Ok(())
    }

    fn copy(&mut self, src: &RocmBuffer, dst: &RocmBuffer) -> Result<(), RocmGpuError> {
        self.recording("copy")?;
        let (src_bytes, dst_bytes) = (src.byte_capacity(), dst.byte_capacity());
        if src_bytes != dst_bytes {
            return Err(RocmGpuError::CopyLength {
                src: src_bytes,
                dst: dst_bytes,
            });
        }
        // The copy kernel moves whole words and the runtime allocates them, so the last word of an odd-width
        // buffer is owned; a buffer that is not word-sized would be written past its end.
        if !src_bytes.is_multiple_of(4) {
            return Err(RocmGpuError::CopyNotWordAligned { bytes: src_bytes });
        }
        let copy = self.copy.as_ref().ok_or(RocmGpuError::CopyKernelMissing)?;
        let words = src_bytes / 4;
        let threads = u32::try_from(words).map_err(|_| RocmGpuError::CopyTooLarge { words })?;
        let words = words as u64;
        let kernarg = self.bind_kernarg(&copy.kernel.kernel, &[(src, words)], (dst, words))?;
        let (grid, block) = padded_launch([threads, 1, 1], copy.workgroup);
        let aql = CapturedAql {
            kernel: copy.kernel.kernel.clone(),
            kernarg,
            elements: Box::new([ElementKind::I32, ElementKind::I32]),
            grid,
            block,
        };
        self.recording("copy")?.push(aql);
        Ok(())
    }

    fn finish(&mut self) -> Result<RocmRecording, RocmGpuError> {
        let aqls = self
            .open
            .take()
            .ok_or(RocmGpuError::NotRecording { op: "finish" })?;
        Ok(RocmRecording { aqls })
    }

    fn begin_step(&mut self) -> Result<(), RocmGpuError> {
        self.ctx.clear_dispatch_spans();
        Ok(())
    }

    fn replay(&mut self, recording: &RocmRecording) -> Result<(), RocmGpuError> {
        if recording.aqls.is_empty() {
            return Ok(());
        }
        let dispatches: Vec<BatchedDispatch> = recording
            .aqls
            .iter()
            .map(|aql| (aql.kernel.clone(), aql.kernarg.clone(), aql.grid, aql.block))
            .collect();
        let elements: Vec<&[ElementKind]> =
            recording.aqls.iter().map(|aql| &*aql.elements).collect();
        self.ctx.enqueue_graph_batched(&dispatches, &elements)?;
        Ok(())
    }

    fn write_with(
        &mut self,
        dst: &RocmBuffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), RocmGpuError> {
        Ok(self.ctx.write_with(dst, offset, len, fill)?)
    }

    fn read(&mut self, src: &RocmBuffer, out: &mut [u8]) -> Result<(), RocmGpuError> {
        Ok(self.ctx.read_bytes(src, out)?)
    }

    fn synchronize(&mut self) -> Result<(), RocmGpuError> {
        // The step's one host wait (the module doc): it completes every batch `replay` enqueued, and
        // a timeout surfaces here, classified the way a replay failure is (CR30: abort, mark
        // NeedsReset).
        Ok(self.ctx.synchronize()?)
    }

    /// The dispatches the step's `synchronize` completed, from the HSA profiler's start/end ticks
    /// (POOT-1079). `Unknown` for a device not built with [`Self::new_with_device_timing`], or a step
    /// that completed no dispatch - never host wall, never a fabricated zero (Card 552).
    fn device_time(&self) -> DeviceTime {
        measured(&self.ctx.dispatch_spans())
    }

    fn abort(&mut self) -> Result<(), RocmGpuError> {
        // Recording never touches the device (see the module doc). Batches an earlier stage of the
        // failed step enqueued are still in flight and read its buffers, so abort completes them
        // before the engine releases anything. `self.open` is the only other Rust-level state abort must
        // clear.
        self.open = None;
        self.ctx.clear_dispatch_spans();
        let drained = self.ctx.synchronize();
        // CR30: a failed replay/synchronize that timed out already poisoned the underlying
        // `RocmContext` (Card 602's `TimeoutLeaks`) - no later submission on it can ever run again.
        // Reporting that here, rather than returning `Ok` unconditionally, is what lets `Engine` set
        // its own `poisoned` flag and refuse every later call immediately instead of re-attempting
        // (and re-failing) each one against a context that is already permanently dead.
        if self.ctx.is_poisoned() {
            return Err(poot_rocm_runtime::RocmError::ContextPoisoned { op: "abort" }.into());
        }
        Ok(drained?)
    }
}

/// A step's device time from its dispatches' spans: `Unknown` for none, else the sum of the
/// durations, the first start to the last end, and each dispatch in submission order.
fn measured(spans: &[DispatchSpan]) -> DeviceTime {
    let (Some(start), Some(end)) = (
        spans.iter().map(|span| span.start).min(),
        spans.iter().map(|span| span.end).max(),
    ) else {
        return DeviceTime::Unknown;
    };
    DeviceTime::Measured(MeasuredDeviceTime {
        sum_of_dispatch_durations: Some(spans.iter().map(DispatchSpan::duration).sum()),
        device_span: Some(end.saturating_sub(start)),
        dispatches: spans
            .iter()
            .enumerate()
            .map(|(index, span)| DispatchDeviceTime {
                index,
                duration: span.duration(),
            })
            .collect(),
    })
}
