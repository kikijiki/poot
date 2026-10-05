//! `RocmDevice`: the ROCm/HSA implementation of the executor contract's [`poot_executor::Device`]
//! (Card 548, ADR-0003), over [`poot_rocm_runtime`]'s AQL record/replay primitives.
//!
//! A recording is a list of [`CapturedAql`] packets. `begin` opens it; `dispatch`/`copy` each build a
//! kernarg buffer and append one packet, compiling the kernel on first use and never touching the
//! device (no HSA dispatch, no wait); `finish` hands the list to the caller as a [`RocmRecording`].
//! `replay` is the one method that actually touches hardware: it batches every packet into one
//! [`poot_rocm_runtime::RocmContext::replay_graph_batched`] call, which rings the doorbell once per
//! chunk and blocks on the last packet's completion signal, so by the time `replay` returns `Ok` the
//! work is already done (`synchronize` is a defensive second wait, not the real one).
//!
//! `Device::copy` has no native ROCm primitive to record into the same AQL stream (HSA's async-copy
//! engine is a separate queue with its own signal, not an AQL packet this recording's BARRIER-ordered
//! replay could sequence against), so it dispatches one generated kernel instead: the
//! committed `assets/copy_words.kir.json` fixture (`ImportedKernel::CopyWords`), a real `#[kernel]` fn
//! (`crates/pootc/kernels/probe/copy_words.rs`) compiled once at construction. One thread per `u32`
//! word, dtype-agnostic: every copy is sized by byte length (`byte_capacity / 4` words), never the
//! buffer's logical element count (`build_kernarg`'s `len` is per call site, exactly because a
//! generated typed kernel's `&[f32]`/`&[i32]`/... parameters and `copy_words`'s `&[u32]` parameters
//! disagree on what "one element" means for any dtype narrower or wider than 4 bytes).

use std::sync::Arc;

use poot_executor::{Arg, BufferRole, Device, DeviceTime, Dispatch, MemoryCounterSnapshot};
use poot_graph_plan::{ImportedKernel, Submission, Target};
use poot_kernel_ir::Body;
use poot_rocm_runtime::{HsacoModule, KernelHandle, RocmBuffer, RocmContext};
use poot_runtime_common::CompiledKernel;
use poot_target::{AmdArch, Backend, BufferStorage, ElementKind};

use crate::error::RocmGpuError;

/// One loaded kernel: its HSA executable (kept alive for the `kernel_object` handle) and launch
/// metadata.
pub struct RocmKernel {
    module: Arc<HsacoModule>,
    kernel: KernelHandle,
}

/// One recorded AQL dispatch: the kernarg buffer and the module that keeps its `kernel_object` valid,
/// held for the recording's lifetime (the kernarg's device pointer is baked into the packet at replay
/// time, so the buffer must outlive every replay).
struct CapturedAql {
    #[allow(
        dead_code,
        reason = "held so the kernel_object stays valid for the recording's lifetime"
    )]
    module: Arc<HsacoModule>,
    kernel: KernelHandle,
    kernarg: RocmBuffer,
    elements: Box<[ElementKind]>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
}

/// One flattened entry of [`poot_rocm_runtime::RocmContext::replay_graph_batched`]'s dispatch list.
type BatchedDispatch = (KernelHandle, RocmBuffer, (u32, u32, u32), (u32, u32, u32));

pub struct RocmRecording {
    aqls: Vec<CapturedAql>,
}

pub struct RocmDevice {
    ctx: RocmContext,
    arch: AmdArch,
    /// The open step's recording so far; `Some` between `begin` and `finish` (mirrors
    /// `poot_gpu::device::WgpuDevice`).
    open: Option<Vec<CapturedAql>>,
    /// The one generated copy kernel (see the module doc), compiled once at construction.
    copy_kernel: RocmKernel,
    /// `copy_kernel`'s compiled workgroup shape (`Body::workgroup_size`): `Device::copy` takes no
    /// threads/workgroup parameters, so this is the only place that shape is known.
    copy_workgroup: [u32; 3],
}

impl RocmDevice {
    pub fn new() -> Result<Self, RocmGpuError> {
        Self::from_context(RocmContext::new()?)
    }

    pub fn from_context(ctx: RocmContext) -> Result<Self, RocmGpuError> {
        let arch = AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
            .unwrap_or_else(|_| AmdArch::gfx1151());
        let (copy_kernel, copy_workgroup) = compile_copy_words_kernel(&ctx, arch)?;
        Ok(Self {
            ctx,
            arch,
            open: None,
            copy_kernel,
            copy_workgroup,
        })
    }

    /// Borrow the underlying context (diagnostic probes: VRAM queries, device caps).
    pub fn context(&self) -> &RocmContext {
        &self.ctx
    }

    fn push(&mut self, aql: CapturedAql) {
        self.open
            .as_mut()
            .expect("dispatch/copy outside begin/finish")
            .push(aql);
    }

    /// Build one kernarg buffer of flat `(ptr: u64, len: u64)` pairs, `[in0, in1, ..., out]` (the
    /// output last; a `Plan::ComputeMeta` metadata buffer is already the last entry of `inputs` by the
    /// time `Engine::walk` calls `dispatch`, so there is no separate meta slot to carry here - mirrors
    /// `poot-gpu`'s single flat kernarg layout, ROCm's own flavor of "one dispatch, one buffer").
    /// `len`'s unit is the kernel body's own: a generated kernel's typed slice parameters (`&[f32]`,
    /// `&[i32]`, ...) take the buffer's logical element count (`RocmBuffer::elem_count`), but
    /// `copy_words`'s `&[u32]` parameters take the buffer's u32 WORD count instead - never the same
    /// number for an element narrower than 4 bytes (bf16/f16/i8: `elem_count` over-reports by
    /// 2x/4x; `poot-target::ElementKind` has no element wider than 4 bytes today, so there is no
    /// under-report case to name). Callers pass each buffer's length explicitly so this one writer
    /// serves both without silently assuming every element is 4 bytes wide; [`copy_word_count`]
    /// computes the copy-specific one.
    fn build_kernarg(
        &self,
        kernel: &KernelHandle,
        in_bufs: &[(&RocmBuffer, u64)],
        out_buf: (&RocmBuffer, u64),
    ) -> Result<RocmBuffer, RocmGpuError> {
        let kernarg_size = kernel.kernarg_size() as usize;
        let slots = in_bufs
            .len()
            .checked_add(1)
            .ok_or_else(|| RocmGpuError::Graph("kernarg slot count overflow".to_string()))?;
        let required = slots
            .checked_mul(16)
            .ok_or_else(|| RocmGpuError::Graph("kernarg byte count overflow".to_string()))?;
        if required > kernarg_size {
            return Err(RocmGpuError::Graph(format!(
                "kernel kernarg segment is {kernarg_size} bytes, but {slots} flat pointer/length \
                 pairs require {required}"
            )));
        }
        let kernarg_buf = self.ctx.allocate_kernarg(kernarg_size)?;
        let mut karg = vec![0u8; kernarg_size];
        let write_pair = |karg: &mut [u8], slot: usize, ptr: u64, len: u64| {
            let off = slot * 16;
            karg[off..off + 8].copy_from_slice(&ptr.to_le_bytes());
            karg[off + 8..off + 16].copy_from_slice(&len.to_le_bytes());
        };
        let mut slot = 0;
        for &(buf, len) in in_bufs {
            write_pair(&mut karg, slot, buf.device_ptr(), len);
            slot += 1;
        }
        write_pair(&mut karg, slot, out_buf.0.device_ptr(), out_buf.1);
        self.ctx.write_raw_bytes(&kernarg_buf, 0, &karg)?;
        Ok(kernarg_buf)
    }
}

/// The u32 word count `Device::copy`'s kernarg length must carry for a buffer of `byte_capacity`
/// bytes: always `byte_capacity / 4`, regardless of the buffer's logical element count (the
/// over-wide `_elem_count` parameter below is accepted and ignored on purpose, not cleaned away: it
/// is exactly the wrong value an earlier version passed here instead, so a future edit that
/// reaches for it is a one-line regression this function's own name and the unit test below exist to
/// make a reader stop and ask "which one"). `copy_words`'s `&[u32]` parameters address by word;
/// `RocmBuffer::elem_count()` is the buffer's LOGICAL element count, equal to the word count only
/// when the element is exactly 4 bytes (`poot-target::ElementKind` has nothing wider today, so there
/// is no over-4-byte case to guard against, only narrower ones). A pure function, so it is
/// unit-testable with no device.
fn copy_word_count(byte_capacity: usize, _elem_count: u32) -> u64 {
    (byte_capacity / 4) as u64
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

fn compile_copy_words_kernel(
    ctx: &RocmContext,
    arch: AmdArch,
) -> Result<(RocmKernel, [u32; 3]), RocmGpuError> {
    let body: Body = ImportedKernel::CopyWords.body().clone();
    let workgroup = body.workgroup_size;
    let target = poot_codegen::Target::AmdGcn(arch);
    let tmp = poot_codegen::kernel_cache_root("rocm", target);
    let out_path = poot_codegen::artifact_path(&tmp, "copy_words", target);
    poot_codegen::compile(&body, target, &out_path)?;
    let bytes = std::fs::read(&out_path)?;
    let compiled = poot_codegen::kernel_handle(&body, target, bytes);
    let module = Arc::new(ctx.load_hsaco(&compiled)?);
    let kernel = ctx.lookup_kernel(&module, compiled.entry_point())?;
    Ok((RocmKernel { module, kernel }, workgroup))
}

impl Device for RocmDevice {
    type Buffer = RocmBuffer;
    type Kernel = RocmKernel;
    type Recording = RocmRecording;
    type Error = RocmGpuError;

    fn target(&self) -> Target {
        Target {
            backend: Backend::AmdGcn(self.arch),
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
    ) -> Result<RocmBuffer, RocmGpuError> {
        Ok(self.ctx.allocate(role, storage, elems)?)
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        kernel: CompiledKernel,
    ) -> Result<RocmKernel, RocmGpuError> {
        let module = Arc::new(self.ctx.load_hsaco(&kernel)?);
        let handle = self.ctx.lookup_kernel(&module, kernel.entry_point())?;
        Ok(RocmKernel {
            module,
            kernel: handle,
        })
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), RocmGpuError> {
        // Always opens a recording, regardless of `_submission` (see the module doc, mirroring
        // `WgpuDevice::begin`): the contract's own restriction to Replay is enforced at
        // `Engine::add_entry`, not here. Recording never touches the device (building a kernarg
        // buffer is a host write into host-visible kernarg memory, not a dispatch), so a failed
        // begin/dispatch/copy/finish can never have mutated device state.
        assert!(self.open.is_none(), "begin inside an open step");
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), RocmGpuError> {
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
        let kernarg = self.build_kernarg(
            &d.kernel.kernel,
            &in_pairs,
            (out_buf, u64::from(d.output.elems)),
        )?;
        let mut elements: Vec<ElementKind> = in_bufs.iter().map(|b| b.element()).collect();
        elements.push(d.output.buffer.element());
        let (grid, block) = padded_launch(d.threads, d.workgroup);
        self.push(CapturedAql {
            module: Arc::clone(&d.kernel.module),
            kernel: d.kernel.kernel.clone(),
            kernarg,
            elements: elements.into_boxed_slice(),
            grid,
            block,
        });
        Ok(())
    }

    fn copy(&mut self, src: &RocmBuffer, dst: &RocmBuffer) -> Result<(), RocmGpuError> {
        let src_bytes = src.byte_capacity();
        if src_bytes != dst.byte_capacity() {
            return Err(RocmGpuError::Graph(format!(
                "copy: src is {src_bytes} bytes, dst is {} bytes",
                dst.byte_capacity()
            )));
        }
        if !src_bytes.is_multiple_of(4) {
            return Err(RocmGpuError::Graph(format!(
                "copy: {src_bytes} bytes is not a whole number of u32 words"
            )));
        }
        let words = copy_word_count(src_bytes, src.elem_count());
        let kernarg =
            self.build_kernarg(&self.copy_kernel.kernel, &[(src, words)], (dst, words))?;
        let (grid, block) = padded_launch([words as u32, 1, 1], self.copy_workgroup);
        self.push(CapturedAql {
            module: Arc::clone(&self.copy_kernel.module),
            kernel: self.copy_kernel.kernel.clone(),
            kernarg,
            elements: Box::new([ElementKind::I32, ElementKind::I32]),
            grid,
            block,
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<RocmRecording>, RocmGpuError> {
        let aqls = self.open.take().expect("finish without begin");
        Ok(Some(RocmRecording { aqls }))
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
        self.ctx.replay_graph_batched(&dispatches, &elements)?;
        Ok(())
    }

    fn write(&mut self, dst: &RocmBuffer, bytes: &[u8]) -> Result<(), RocmGpuError> {
        Ok(self.ctx.write_bytes(dst, bytes)?)
    }

    fn read(&mut self, src: &RocmBuffer, out: &mut [u8]) -> Result<(), RocmGpuError> {
        Ok(self.ctx.read_bytes(src, out)?)
    }

    fn synchronize(&mut self) -> Result<(), RocmGpuError> {
        // `replay_graph_batched` already blocked on the batch's completion signal (the module doc);
        // this is a defensive second wait (the queue read-index spin), never the real one. Still the
        // right place for a timeout to surface if it ever does: it classifies the same way a replay
        // failure does (CR30: abort, mark NeedsReset).
        Ok(self.ctx.synchronize()?)
    }

    fn device_time(&self) -> DeviceTime {
        // Card 552's SC-001 (moved to Card 548): no HSA timestamp binding exists yet, so
        // ROCm reports `Unknown` from day one - never host wall. A real-timestamp follow-up is
        // recorded, not this card's scope.
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), RocmGpuError> {
        // Recording never touches the device (see the module doc): an eager replay's already-
        // submitted batch cannot be undone, but `replay` either fully completes (the batch blocks on
        // the last packet's signal) or returns a typed error the engine already treats as a failed
        // transaction - there is no partial, still-running state to clean up here. `begin` panics
        // (not errors) on a double-open, so `self.open` is the only Rust-level state abort must clear.
        self.open = None;
        // CR30: a failed replay/synchronize that timed out already poisoned the underlying
        // `RocmContext` (Card 602's `TimeoutLeaks`) - no later submission on it can ever run again.
        // Reporting that here, rather than returning `Ok` unconditionally, is what lets `Engine` set
        // its own `poisoned` flag and refuse every later call immediately instead of re-attempting
        // (and re-failing) each one against a context that is already permanently dead.
        if self.ctx.is_poisoned() {
            return Err(poot_rocm_runtime::RocmError::ContextPoisoned { op: "abort" }.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod copy_word_count_tests {
    use super::copy_word_count;

    /// Changing this function to return
    /// `_elem_count` instead of computing from `byte_capacity` must turn this test red for the
    /// 1- and 2-byte cases (where `elem_count` and the real word count disagree), proving the
    /// function (and so `Device::copy`) is not quietly reaching for `elem_count` again. Mutation
    /// applied and reverted for the done-report: `fn copy_word_count(byte_capacity: usize,
    /// elem_count: u32) -> u64 { elem_count as u64 }` failed the 1-byte and 2-byte cases below
    /// (`25 != 100`, `50 != 100`) and passed the 4-byte case only (`100 == 100`, since `elem_count
    /// == byte_capacity/4` exactly when the element is 4 bytes wide) - the real, documented shape of
    /// the original bug, not a contrived mismatch.
    #[test]
    fn copy_word_count_is_the_byte_capacity_never_the_element_count() {
        const ELEMS: u32 = 100;
        // 1-byte element (raw_bytes/i8): byte_capacity = 100, elem_count = 100, words = 25.
        assert_eq!(copy_word_count(ELEMS as usize, ELEMS), 25);
        // 2-byte element (bf16/f16): byte_capacity = 200, elem_count = 100, words = 50.
        assert_eq!(copy_word_count(ELEMS as usize * 2, ELEMS), 50);
        // 4-byte element (f32/i32): byte_capacity = 400, elem_count = 100, words = 100 (the one
        // case where the buggy `elem_count` value and the correct word count happen to coincide).
        assert_eq!(copy_word_count(ELEMS as usize * 4, ELEMS), 100);
    }
}
