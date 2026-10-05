//! `PtxDevice`: the PTX implementation of the executor contract's [`poot_executor::Device`] (Card
//! 549, mirroring `poot_gpu::device::WgpuDevice`, ADR-0003).
//!
//! A recording is a CUDA graph (G3c): `begin` opens a stream capture
//! ([`PtxContext::begin_capture`]), `dispatch`/`copy` enqueue onto the capturing stream (recorded,
//! not run - `PtxContext::dispatch_dev` already records during an open capture), `finish` ends the
//! capture into a replayable [`PtxGraphExec`] (via [`PtxContext::end_capture`]), and `replay` is one
//! `cuGraphLaunch` ([`PtxContext::launch_timed`]). `begin` always opens a capture and never submits,
//! regardless of `Submission` (matching `WgpuDevice`'s doc): the contract's
//! own restriction to `Replay` is enforced at `Engine::add_entry`, not here.
//!
//! `Device::copy` (S46-6) is one generated copy kernel - [`poot_kernelgen::broadcast_dt`]
//! with matching in/out shapes, the same body `poot-rocm-gpu`'s native state commit already uses -
//! never a native `cuMemcpyDtoD`: PTX has no storage-neutral native copy primitive, and a generated
//! kernel captures into the graph exactly like any other dispatch. The element type is chosen by
//! native width (`Ty::U32` for 4-byte F32/I32 buffers, `Ty::BF16` for 2-byte Bf16/F16 buffers):
//! `poot_runtime_common::check_element_schema`'s width-based compatibility (same byte width, not
//! exact `ElementKind`) is exactly what lets one bit-mover kernel serve either storage in its class.

use poot_executor::{Arg, BufferRole, Device, DeviceTime, Dispatch, MeasuredDeviceTime};
use poot_graph_plan::{Submission, Target};
use poot_kernel_ir::Ty;
use poot_ptx_runtime::{
    BufferStorage, CompiledKernel, PtxBuffer, PtxContext, PtxGraphExec, PtxSpan,
};
use poot_target::{Backend, ElementKind};

use crate::PtxGpuError;

/// [`PtxDevice::copy`]'s generated kernel type and dispatch width for a buffer of `element`'s native
/// width class holding `elem_count` logical elements (S46-6). One thread per logical element, typed
/// to its own width (`Ty::U32` for 4-byte F32/I32/RawBytes, `Ty::BF16` for 2-byte Bf16/F16) - never a
/// fixed u32-word count recovered from byte capacity. Contrast ROCm's `copy_words` (card 548 finding): that kernel is always u32-word-granular regardless of the buffer's real element width,
/// so it must separately derive a word count from `byte_capacity`, and using the buffer's logical
/// `elem_count` there instead undercounted every narrower-than-4-byte buffer by exactly its width
/// ratio. PTX's kernel body is generated per `Ty`, so the buffer's own `elem_count` is already the
/// right dispatch width for whichever `Ty` this returns - there is no second, word-granular kernel to
/// mismatch it against.
fn copy_launch_params(elem_count: u32, element: ElementKind) -> (Ty, usize) {
    let ty = match element {
        ElementKind::F32 | ElementKind::I32 | ElementKind::RawBytes => Ty::U32,
        ElementKind::Bf16 | ElementKind::F16 => Ty::BF16,
    };
    (ty, elem_count as usize)
}

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

pub struct PtxDevice {
    ctx: PtxContext,
    /// This device's own kernel cache for [`Self::copy`]'s generated copy kernel, entirely separate
    /// from `Engine`'s plan-sourced one (`Engine::kernel`): `copy` is engine-internal state-commit
    /// logic `Engine::walk` calls directly, never through `Device::load_kernel`/`dispatch`.
    copy_kernel_cache: poot_codegen::KernelCache,
    /// The open step's device-timed span (Card 552 SC-006), set by [`Self::replay`] and read by
    /// [`Self::device_time`]; `None` when this device was not built with
    /// [`PtxContext::new_with_device_timing`], or no step has replayed yet.
    pending_span: Option<PtxSpan>,
}

impl PtxDevice {
    pub fn new() -> Result<Self, PtxGpuError> {
        Ok(Self::with_context(PtxContext::new()?))
    }

    /// A device for the Card 552 typed timing path: [`Device::device_time`] reports a measured
    /// device span instead of `Unknown` (SC-006). See [`PtxContext::new_with_device_timing`].
    pub fn new_with_device_timing() -> Result<Self, PtxGpuError> {
        Ok(Self::with_context(PtxContext::new_with_device_timing()?))
    }

    fn with_context(ctx: PtxContext) -> Self {
        Self {
            ctx,
            copy_kernel_cache: poot_codegen::KernelCache::open(poot_codegen::Target::Nvptx),
            pending_span: None,
        }
    }

    /// Diagnostic access to the underlying context (e.g. `profiler()`, `vram_used_bytes()`).
    pub fn context(&self) -> &PtxContext {
        &self.ctx
    }
}

impl Device for PtxDevice {
    type Buffer = PtxBuffer;
    type Kernel = PtxKernel;
    type Recording = PtxRecording;
    type Error = PtxGpuError;

    fn target(&self) -> Target {
        Target {
            backend: Backend::Nvptx,
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
    ) -> Result<PtxBuffer, PtxGpuError> {
        Ok(self.ctx.alloc_storage(role, storage, elems)?)
    }

    fn load_kernel(&mut self, key: &str, kernel: CompiledKernel) -> Result<PtxKernel, PtxGpuError> {
        Ok(PtxKernel {
            key: key.to_string(),
            compiled: kernel,
        })
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), PtxGpuError> {
        // Always opens a capture, regardless of `_submission` (see the module doc): the contract's
        // own restriction to Replay is enforced at `Engine::add_entry`, not here.
        Ok(self.ctx.begin_capture()?)
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), PtxGpuError> {
        // Card 547b: every argument's own logical element count from the plan, never
        // the bound buffer's capacity (an arena slot's buffer is sized to its largest occupant,
        // which can exceed this particular dispatch's operand).
        let ins: Vec<&PtxBuffer> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.buffer).collect();
        let in_lens: Vec<u32> = d.inputs.iter().map(|a: &Arg<'_, Self>| a.elems).collect();
        Ok(self.ctx.dispatch_dev(
            &d.kernel.key,
            &d.kernel.compiled,
            d.workgroup,
            d.threads,
            &ins,
            &in_lens,
            d.output.buffer,
            d.output.elems,
        )?)
    }

    fn copy(&mut self, src: &PtxBuffer, dst: &PtxBuffer) -> Result<(), PtxGpuError> {
        let (ty, n) = copy_launch_params(dst.elem_count(), dst.element());
        if n == 0 {
            return Ok(());
        }
        let body = poot_kernelgen::broadcast_dt("ptx_device_copy", ty, &[n], &[n]);
        let artifact = self.copy_kernel_cache.load_or_compile(
            &body,
            poot_graph_plan::CompileLimits::STANDARD.max_artifact_bytes,
        )?;
        let compiled =
            poot_codegen::kernel_handle(&body, poot_codegen::Target::Nvptx, artifact.bytes);
        let threads = u32::try_from(n).map_err(|_| {
            PtxGpuError::Graph(format!("copy of {n} elements exceeds the launch ABI"))
        })?;
        // `copy` has no plan of its own (a generated bit-mover, not a graph dispatch),
        // so each buffer's own `elem_count()` - never arena-shared for a whole-buffer copy's src/dst
        // (the buffer-plan module doc) - is exactly its planned length here.
        Ok(self.ctx.dispatch_dev(
            "ptx_device_copy",
            &compiled,
            [256, 1, 1],
            [threads, 1, 1],
            &[src],
            &[src.elem_count()],
            dst,
            dst.elem_count(),
        )?)
    }

    fn finish(&mut self) -> Result<Option<PtxRecording>, PtxGpuError> {
        let exec = self.ctx.end_capture()?;
        Ok(Some(PtxRecording { exec }))
    }

    fn replay(&mut self, recording: &PtxRecording) -> Result<(), PtxGpuError> {
        self.pending_span = self.ctx.launch_timed(&recording.exec)?;
        Ok(())
    }

    fn write(&mut self, dst: &PtxBuffer, bytes: &[u8]) -> Result<(), PtxGpuError> {
        Ok(self.ctx.write_bytes(dst, bytes)?)
    }

    fn read(&mut self, src: &PtxBuffer, out: &mut [u8]) -> Result<(), PtxGpuError> {
        Ok(self.ctx.read_bytes(src, out)?)
    }

    fn synchronize(&mut self) -> Result<(), PtxGpuError> {
        Ok(self.ctx.synchronize()?)
    }

    /// Card 552: the span [`Self::replay`] bracketed around the last `cuGraphLaunch`, read once the
    /// caller has synchronized (`Engine::step` always does, strictly before calling this). `Unknown`
    /// when this device was not built with [`PtxContext::new_with_device_timing`] (SC-006: never a
    /// fabricated host-wall measurement), or if reading the event pair fails.
    fn device_time(&self) -> DeviceTime {
        match &self.pending_span {
            None => DeviceTime::Unknown,
            Some(span) => match self.ctx.span_elapsed(span) {
                Ok(duration) => DeviceTime::Measured(MeasuredDeviceTime {
                    sum_of_dispatch_durations: None,
                    device_span: Some(duration),
                    dispatches: Vec::new(),
                }),
                Err(_) => DeviceTime::Unknown,
            },
        }
    }

    fn abort(&mut self) -> Result<(), PtxGpuError> {
        self.ctx.abort_capture();
        // Defense in depth, as `WgpuDevice::abort`'s identical comment explains: a failed replay
        // (e.g. a mid-launch error) reaches here too, and this guarantees no pending span survives
        // into whatever runs next.
        self.pending_span = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Card 548 review's copy-length finding, checked against PTX's own (different) copy kernel
    /// shape: unlike ROCm's single u32-word `copy_words` kernel, PTX generates one kernel per native
    /// width class, so the dispatch width is always the buffer's own logical `elem_count` - never a
    /// byte-capacity-derived word count. Demonstrates why a ROCm-style `byte_capacity / 4` word count
    /// would be the wrong length for a narrower-than-4-byte buffer here too, had `copy_launch_params`
    /// been written that way: a 2-byte-element (Bf16/F16) buffer's `byte_capacity / 4` under-covers
    /// its real element count by half, silently copying only the first half of the buffer.
    #[test]
    fn copy_launch_params_dispatches_the_real_elem_count_never_a_byte_capacity_word_count() {
        const ELEMS: u32 = 100;

        let (ty, n) = copy_launch_params(ELEMS, ElementKind::F32);
        assert_eq!((ty, n), (Ty::U32, ELEMS as usize));
        let (ty, n) = copy_launch_params(ELEMS, ElementKind::I32);
        assert_eq!((ty, n), (Ty::U32, ELEMS as usize));
        let (ty, n) = copy_launch_params(ELEMS, ElementKind::RawBytes);
        assert_eq!((ty, n), (Ty::U32, ELEMS as usize));

        let (ty, n) = copy_launch_params(ELEMS, ElementKind::Bf16);
        assert_eq!((ty, n), (Ty::BF16, ELEMS as usize));
        let (ty, n) = copy_launch_params(ELEMS, ElementKind::F16);
        assert_eq!((ty, n), (Ty::BF16, ELEMS as usize));

        // The ROCm-style word count a fixed-u32-word kernel would need for this 2-byte-element
        // buffer: byte_capacity (200) / 4 = 50, half of the real 100-element count. PTX's own
        // Ty::BF16-typed kernel must dispatch the real `elem_count`, not this.
        let rocm_style_word_count = (ELEMS as usize * 2) / 4;
        assert_ne!(
            n, rocm_style_word_count,
            "a byte-capacity/4 word count would under-cover a 2-byte-element buffer by half"
        );
    }
}
