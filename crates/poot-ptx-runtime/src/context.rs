use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{result, sys};
use poot_runtime_common::{BufferRole, CompiledKernel, DeviceBackend, KernelCode, fnv1a};
use poot_target::ElementKind;

use super::buffer::{PtxBuffer, allocate_owned, checked_copy};
use super::driver::{CudaDriver, DriverCleanup};
use super::error::{BufferStorage, PtxError};
use super::graph::{
    CaptureRetention, ModuleConstruction, PtxGraphExec, PtxSpan, create_event_pair,
    instantiate_graph,
};
use super::owner::{PtxOwner, construct_owner};

/// A CUDA context + stream, reusable across dispatches. Caches loaded modules/functions by kernel text.
pub struct PtxContext {
    pub(crate) owner: Arc<PtxOwner>,
    /// Cached `CUfunction` by "ptx-hash:entry".
    pub(crate) funcs: RefCell<HashMap<String, sys::CUfunction>>,
    /// The buffers the open capture (if any) has recorded; see [`PtxContext::begin_capture`].
    pub(crate) capture: CaptureRetention,
    /// The device-capability descriptor, measured once at construction (card 522 review M1):
    /// [`PtxContext::device_caps`] returns this stored value, never re-queries the driver.
    pub(crate) device_caps: poot_target::DeviceCaps,
    /// Card 552 (SC-006): true when this context was built via
    /// [`PtxContext::new_with_device_timing`], so [`PtxContext::launch_timed`] brackets each replay
    /// with one CUDA event pair instead of skipping timing (a plain [`PtxGraphExec::launch`] with no
    /// event overhead).
    pub(crate) device_timing: bool,
}

// SAFETY: `owner` is already `Send + Sync` (see its doc). `funcs` caches `CUfunction` handles -
// plain addresses, not Rust references - behind a private `RefCell` that only ever aliases
// within one call on whichever thread currently owns this `PtxContext`; `capture` and
// `device_caps` hold no thread-affine state either. `PtxContext` is never `Sync` (moving it, not
// sharing a live `&PtxContext`, is the only supported cross-thread use, R471-010 - see the crate
// doc), so no `RefCell` here is ever aliased from two threads at once.
unsafe impl Send for PtxContext {}

/// `POOT_REQUIRE_PTX=1` turns the "no PTX device -> tests skip cleanly" convention into a loud failure:
/// a failed context open panics instead of returning the Err the skip guards convert into a silent pass
/// (`just test-device-ptx` sets it). Mirrors `poot-runtime::require_gpu_check`.
fn require_gpu_check<E: std::fmt::Display>(e: E) -> E {
    require_gpu_check_with(|name| std::env::var(name).ok(), e)
}

pub(crate) fn require_gpu_check_with<E: std::fmt::Display>(
    get: impl Fn(&str) -> Option<String>,
    e: E,
) -> E {
    DeviceBackend::Ptx.fail_if_required(get, e)
}

/// The device-capability descriptor the planner reads (card 522): a single buffer is safe up to the
/// device's total VRAM (`cuMemGetInfo`'s total; no narrower ceiling is known for this backend), and the LDS,
/// grid, workgroup (block), warp and compute-capability values are `cuDeviceGetAttribute` queries. There
/// is no display watchdog or confirmed tiled-GEMM miscompile on this backend (both are wgpu/RADV
/// codegen-path defects). `tensor_core` is classified from the device's compute capability: poot's NVPTX
/// codegen targets `-mcpu=sm_80`, whose bf16 `wmma` fragments exist from compute capability 8.
///
/// Called once, at [`PtxContext::build_ordinal`], while the retained context is current on this thread;
/// [`PtxContext::device_caps`] returns the stored result forever after (card 522 review M1).
fn measure_device_caps(dev: sys::CUdevice) -> Result<poot_target::DeviceCaps, PtxError> {
    use sys::CUdevice_attribute as Attr;
    let attr = |a: Attr| -> Result<u32, PtxError> {
        // SAFETY: `dev` came from `result::device::get` at context construction.
        let value = unsafe { result::device::get_attribute(dev, a) }.map_err(PtxError::Driver)?;
        Ok(value as u32)
    };
    let (_, total) = result::mem_get_info()?;
    Ok(caps_from_attributes(
        total as u64,
        &CudaAttributes {
            shared_memory_per_block: attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            max_grid: [
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
            ],
            max_block: [
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
                attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
            ],
            max_threads_per_block: attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            warp_size: attr(Attr::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
            compute_capability_major: attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            multiprocessor_count: attr(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        },
    ))
}

/// The `cuDeviceGetAttribute` values [`caps_from_attributes`] reads: the seam a test feeds sentinels
/// through, since no real device reports them.
struct CudaAttributes {
    shared_memory_per_block: u32,
    max_grid: [u32; 3],
    max_block: [u32; 3],
    max_threads_per_block: u32,
    warp_size: u32,
    compute_capability_major: u32,
    multiprocessor_count: u32,
}

fn caps_from_attributes(total_memory: u64, attrs: &CudaAttributes) -> poot_target::DeviceCaps {
    poot_target::DeviceCaps {
        max_buffer_bytes: total_memory,
        lds_bytes: attrs.shared_memory_per_block,
        max_workgroup_size: attrs.max_block,
        max_workgroup_invocations: attrs.max_threads_per_block,
        subgroup: poot_target::Queried::Known(poot_target::SubgroupSupport::Present {
            min_size: attrs.warp_size,
            max_size: attrs.warp_size,
        }),
        max_dispatch_work: poot_target::Queried::Unknown,
        max_grid: attrs.max_grid,
        watchdog_budget: None,
        tensor_core: poot_target::TensorCoreSupport::from_cuda_compute_capability(
            attrs.compute_capability_major,
        ),
        known_miscompiles: poot_target::KnownMiscompiles::default(),
        compute_units: attrs.multiprocessor_count,
    }
}

impl PtxContext {
    /// Initialise CUDA, retain device 0's primary context, set it current, create a non-blocking stream.
    pub fn new() -> Result<Self, PtxError> {
        Self::build_ordinal(0).map_err(require_gpu_check)
    }

    /// Like [`PtxContext::new`], but [`PtxContext::launch_timed`] brackets every graph replay with one
    /// CUDA event pair (Card 552 SC-006): the executor contract's device-span timing.
    pub fn new_with_device_timing() -> Result<Self, PtxError> {
        let mut ctx = Self::build_ordinal(0).map_err(require_gpu_check)?;
        ctx.device_timing = true;
        Ok(ctx)
    }

    fn build_ordinal(ordinal: usize) -> Result<Self, PtxError> {
        // cudarc dynamic-loads libcuda lazily on its first call and panics (rather than returning Err) when the
        // NVIDIA driver library is absent (e.g. an AMD-only box). Catch that on the first cudarc call and return a
        // typed error so callers skip cleanly (AGENTS.md: tests skip when GPUs are absent). A real driver error
        // still returns via `?` below.
        std::panic::catch_unwind(result::init)
            .map_err(|_| {
                PtxError::CudaUnavailable("libcuda not loadable (no NVIDIA driver)".to_string())
            })?
            .map_err(PtxError::Driver)?;
        let device = result::device::get(ordinal as i32)?;
        let driver: Arc<dyn CudaDriver> = Arc::new(DriverCleanup);
        // NonBlocking is required for CUDA graph capture (a legacy-default-synchronizing stream cannot be captured).
        // The cross-stream hazard (a NULL-stream zeroing in `alloc_f32` racing a compute kernel on this stream,
        // spec 023: a varying block-aligned zero tail) is avoided by zeroing via a device memset on this stream.
        let owner = construct_owner(
            driver,
            device,
            // SAFETY: `device` came from `device::get`; `construct_owner` balances the retain with one release.
            || unsafe { result::primary_ctx::retain(device).map_err(PtxError::Driver) },
            || {
                result::stream::create(result::stream::StreamKind::NonBlocking)
                    .map_err(PtxError::Driver)
            },
        )?;
        // Measured once here (card 522 review M1), while the retained context is current on this
        // thread (`construct_owner`'s `guard.keep_current()`): every later `device_caps()` call
        // returns this stored `Copy` instead of re-querying the driver.
        let device_caps = measure_device_caps(device)?;
        Ok(PtxContext {
            owner,
            funcs: RefCell::new(HashMap::new()),
            capture: CaptureRetention::default(),
            device_caps,
            device_timing: false,
        })
    }

    /// Make this context current on the calling thread. Current-context state is per-thread: `PtxContext::new` sets
    /// it on the constructing thread, but a launch from any other thread needs its own `cuCtxSetCurrent`, or
    /// every driver call fails with `CUDA_ERROR_INVALID_CONTEXT`. Call once at the top of a thread that a
    /// `PtxContext` (or a `PtxGraphExec`/`Engine<PtxDevice>` built on one) was moved into.
    ///
    /// `#[cfg(test)]` (card 549): no production caller moves a `PtxContext` across
    /// threads today, so this stays test-only until one does; see the crate doc's Threading section.
    #[cfg(test)]
    pub(crate) fn make_current(&self) -> Result<(), PtxError> {
        // SAFETY: `owner.ctx` is the primary context this context retained; it stays alive while `owner` does.
        unsafe { result::ctx::set_current(self.owner.ctx) }?;
        Ok(())
    }

    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// this context.
    pub fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        self.owner.memory.snapshot_all()
    }

    /// Block until all enqueued work on the stream completes (once per token in resident decode, or before
    /// reading a result).
    pub fn synchronize(&self) -> Result<(), PtxError> {
        self.refuse_during_capture("synchronize")?;
        // SAFETY: `owner.stream` is this context's live stream, destroyed only when `owner` drops.
        unsafe { result::stream::synchronize(self.owner.stream) }?;
        Ok(())
    }

    /// Upload f32 data to a fresh device buffer (H2D, synchronous).
    pub fn upload_f32(&self, data: &[f32]) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("upload_f32")?;
        self.upload_elements(
            "upload_f32",
            data,
            data.len(),
            BufferStorage::f32(),
            BufferRole::Weight,
        )
    }

    /// Upload host i32 to the device (e.g. packed GPTQ qweight/qzeros/g_idx, spec 026). `elem_count` is the
    /// element count, matching a kernel's i32-slice GEPs.
    pub fn upload_i32(&self, data: &[i32]) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("upload_i32")?;
        self.upload_elements(
            "upload_i32",
            data,
            data.len(),
            BufferStorage::i32(),
            BufferRole::Weight,
        )
    }

    /// A zeroed device buffer of `elems` f32. The zeroing is a device memset enqueued on `self.stream`, so it is
    /// ordered before any later compute kernel on that stream that writes the buffer (a NULL-stream `memcpy_htod`
    /// of zeros would race the NonBlocking stream, spec 023).
    pub fn alloc_f32(&self, elems: usize) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("alloc_f32")?;
        self.alloc_zeroed(
            "alloc_f32",
            elems,
            BufferStorage::f32(),
            BufferRole::Activation,
        )
    }

    /// Upload raw bf16 bytes unchanged: the native-checkpoint counterpart of the host-f32 bf16 rounding
    /// path, so `Tensor::bf16`
    /// bytes are not widened to f32 and narrowed again. `bytes` are little-endian, two per element, and
    /// `elem_count` is the logical element count, not the byte length.
    pub fn upload_bf16_bytes(&self, bytes: &[u8]) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("upload_bf16_bytes")?;
        self.upload_raw(
            "upload_bf16_bytes",
            bytes,
            BufferStorage::bf16(),
            BufferRole::Weight,
        )
    }

    /// Upload raw f16 (IEEE-754 half) bytes unchanged: a straight H2D copy, not a host-f32-to-bf16 rounding
    /// re-encode like [`Self::upload_bf16_bytes`]'s bf16 twin used to be. Spec 135 Phase 2 residency
    /// (FR-008): the caller passes the checkpoint's little-endian bytes (`poot_eval::Tensor::f16`).
    /// `bytes.len()` must be a whole number of 2-byte elements; the returned buffer's `elem_count` is the
    /// element count, not the byte length.
    pub fn upload_f16(&self, bytes: &[u8]) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("upload_f16")?;
        self.upload_raw(
            "upload_f16",
            bytes,
            BufferStorage::f16(),
            BufferRole::Weight,
        )
    }

    /// Upload little-endian `bytes` as whole `element`s.
    fn upload_raw(
        &self,
        op: &'static str,
        bytes: &[u8],
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<PtxBuffer, PtxError> {
        let element_bytes = storage.element().byte_width();
        if !bytes.len().is_multiple_of(element_bytes) {
            return Err(PtxError::InvalidByteLength {
                op,
                bytes: bytes.len(),
                element_bytes,
            });
        }
        let elements = bytes.len() / element_bytes;
        self.upload_elements(op, bytes, elements, storage, role)
    }

    /// A fresh buffer of `elements` `element`s holding a synchronous copy of `data`, whose byte size must be
    /// exactly `elements * element.byte_width()`.
    fn upload_elements<T>(
        &self,
        op: &'static str,
        data: &[T],
        elements: usize,
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<PtxBuffer, PtxError> {
        allocate_owned(
            Arc::clone(&self.owner),
            op,
            elements,
            storage,
            role,
            device_alloc,
            |ptr, bytes| {
                assert_eq!(
                    std::mem::size_of_val(data),
                    bytes,
                    "{op}: host and device sizes differ"
                );
                // SAFETY: `ptr` is a live allocation of exactly `bytes` = `size_of_val(data)` bytes (checked
                // above), and the synchronous copy reads `data` only during the call.
                unsafe { result::memcpy_htod_sync(ptr, data) }?;
                Ok(())
            },
        )
    }

    /// A buffer of `elements` `element`s zeroed by a memset on this context's stream, ordered before any later
    /// kernel on the stream.
    fn alloc_zeroed(
        &self,
        op: &'static str,
        elements: usize,
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<PtxBuffer, PtxError> {
        allocate_owned(
            Arc::clone(&self.owner),
            op,
            elements,
            storage,
            role,
            device_alloc,
            |ptr, bytes| {
                // SAFETY: `ptr` is a live allocation of `bytes` bytes on this context, `stream` is its live
                // stream, and a zero byte pattern is a valid value of every `ElementKind`.
                unsafe { result::memset_d8_async(ptr, 0, bytes, self.owner.stream) }
                    .map_err(PtxError::Driver)
            },
        )
    }

    /// (free, total) device memory in bytes (`cuMemGetInfo`), used to receipt the bf16 VRAM win (spec 024).
    pub fn mem_info(&self) -> Result<(usize, usize), PtxError> {
        Ok(result::mem_get_info()?)
    }

    /// Device-wide VRAM in use (bytes): `total - free` from `cuMemGetInfo`, a driver-only device metric (no
    /// NVML/nvidia-smi), the CUDA analogue of `poot-runtime::Context::vram_used_bytes`'s Vulkan
    /// `VK_EXT_memory_budget` query (card 030). Device-wide, not per-process, so it reflects memory pressure
    /// rather than poot's own allocation gauges. `None` if the driver call fails (e.g. no context bound). One
    /// driver call, cheap enough per scrape.
    pub fn vram_used_bytes(&self) -> Option<u64> {
        let (free, total) = self.mem_info().ok()?;
        Some(total.saturating_sub(free) as u64)
    }

    /// Total device memory visible to this context (bytes), from `cuMemGetInfo`. Headroom is
    /// `vram_budget_bytes - vram_used_bytes` ([`Self::vram_used_bytes`]). `None` if the driver call fails.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        let (_, total) = self.mem_info().ok()?;
        Some(total as u64)
    }

    /// The device-capability descriptor the planner reads (card 522), measured once at context
    /// construction and stored (card 522 review M1: a `Copy` read, never a re-query of the driver on
    /// every call - the fields below are static device properties that cannot change across the
    /// context's lifetime, so a per-call driver round-trip only added latency and a spurious
    /// driver-error path to a plan call that has nothing to do with planning).
    pub fn device_caps(&self) -> poot_target::DeviceCaps {
        self.device_caps
    }

    /// Overwrite an existing device buffer's contents in place (H2D to its stable address), e.g. to refresh a
    /// per-token scalar slot before replaying a captured graph (G3c): the recorded nodes reference the fixed
    /// address, so new bytes there feed new values on replay.
    ///
    /// Enqueued on `self.stream` and then synchronized, not via the legacy stream's `memcpy_htod_sync`. The
    /// legacy stream does not implicitly synchronize with a `CU_STREAM_NON_BLOCKING` stream (required for graph
    /// capture), so a legacy write could land before an earlier `self.stream` op (this buffer's `alloc_f32`
    /// memset, or an in-flight graph replay reading the address), or after a later launch begins (the race
    /// class of `alloc_f32`, spec 023). Writing on the same stream restores FIFO ordering on both sides.
    pub fn update_f32(&self, buf: &PtxBuffer, data: &[f32]) -> Result<(), PtxError> {
        self.refuse_during_capture("update_f32")?;
        self.write_elements("update_f32", buf, BufferStorage::f32(), 0, data, true)
    }

    /// Copy `data` into `buf` at element `elem_offset` on this context's stream, then synchronize. `whole`
    /// requires `data` to cover the buffer exactly.
    fn write_elements<T>(
        &self,
        op: &'static str,
        buf: &PtxBuffer,
        storage: BufferStorage,
        elem_offset: usize,
        data: &[T],
        whole: bool,
    ) -> Result<(), PtxError> {
        checked_copy(buf, op, storage, elem_offset, data.len(), whole, |range| {
            assert_eq!(
                std::mem::size_of_val(data),
                range.byte_len,
                "{op}: element width differs"
            );
            let dst = buf.ptr() + range.byte_offset as sys::CUdeviceptr;
            // SAFETY: `checked_copy` proved `[byte_offset, byte_offset + byte_len)` lies inside `buf`'s live
            // allocation, and `byte_len` is the size of `data` (checked above). The copy runs on this context's
            // stream; the driver stages pageable `data` before returning, and the stream is synchronized below
            // before `data` can be released. No capture is open (every caller refuses first), so the copy is
            // not recorded into a graph.
            unsafe { result::memcpy_htod_async(dst, data, self.owner.stream) }?;
            self.synchronize()?;
            Ok(())
        })
    }

    /// Synchronize the stream, then copy all of `b` (stored as `element`) to the host.
    fn download_elements<T: Clone + Default>(
        &self,
        op: &'static str,
        b: &PtxBuffer,
        storage: BufferStorage,
    ) -> Result<Vec<T>, PtxError> {
        let elements = b.elem_count() as usize;
        checked_copy(b, op, storage, 0, elements, true, |range| {
            self.synchronize()?;
            let mut out = vec![T::default(); elements];
            assert_eq!(
                std::mem::size_of_val(&out[..]),
                range.byte_len,
                "{op}: element width differs"
            );
            // SAFETY: `b` is a live allocation holding `byte_len` bytes (checked by `checked_copy`), `out` has
            // exactly that many bytes (checked above), and the stream has finished every write to `b`.
            unsafe { result::memcpy_dtoh_sync(&mut out[..], b.ptr()) }?;
            Ok(out)
        })
    }

    /// Begin capturing the work enqueued on this context's stream into a CUDA graph (thread-local mode).
    /// Dispatches between `begin_capture` and [`PtxContext::end_capture`] are recorded, not run, and each
    /// dispatched buffer is retained for the graph. Every other operation on the context refuses with
    /// [`PtxError::CaptureOpen`] until the capture ends: allocation, freeing and stream synchronization are
    /// illegal during capture, and a recorded host copy would read caller memory at every replay. See G3c.
    pub fn begin_capture(&self) -> Result<(), PtxError> {
        self.refuse_during_capture("begin_capture")?;
        // SAFETY: `stream` is this context's live stream (owned by `owner`), and thread-local mode confines the
        // capture to this thread.
        unsafe {
            result::stream::begin_capture(
                self.owner.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }?;
        self.capture.begin();
        Ok(())
    }

    /// End capture and instantiate the recorded graph into a replayable [`PtxGraphExec`] that owns the
    /// buffers the capture dispatched. On an instantiate failure the recorded `graph` (a bare driver handle not
    /// yet wrapped in [`PtxGraphExec`]'s `Drop`) is destroyed before returning the error, or it would leak.
    pub fn end_capture(&self) -> Result<PtxGraphExec, PtxError> {
        let captured = self.capture.finish();
        // SAFETY: `stream` is this context's live stream; ending a capture takes no host memory.
        let graph = unsafe { result::stream::end_capture(self.owner.stream) }?;
        instantiate_graph(Arc::clone(&self.owner), graph, captured, |exec| {
            // SAFETY: `exec` is a valid out-pointer and `graph` is the live graph the capture just returned.
            unsafe { sys::cuGraphInstantiateWithFlags(exec, graph, 0) }
                .result()
                .map_err(PtxError::Driver)
        })
    }

    /// Replay `exec` (Card 552 SC-006): a plain [`PtxGraphExec::launch`] on this context's stream,
    /// bracketed by one CUDA event pair when this context was built via
    /// [`PtxContext::new_with_device_timing`] (`None` otherwise - never a fabricated host-wall
    /// measurement). Does not sync; the caller reads [`PtxContext::span_elapsed`] only after its own
    /// sync has proven the replay complete.
    pub fn launch_timed(&self, exec: &PtxGraphExec) -> Result<Option<PtxSpan>, PtxError> {
        if !self.device_timing {
            exec.launch()?;
            return Ok(None);
        }
        let (start, stop) = create_event_pair(Arc::clone(&self.owner), || {
            result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(PtxError::Driver)
        })?;
        // SAFETY: both events are live and recorded on this context's stream, which is the same
        // stream `exec.launch()` replays onto, so the pair brackets exactly this replay.
        unsafe { result::event::record(start.event, self.owner.stream) }?;
        exec.launch()?;
        // SAFETY: as above.
        unsafe { result::event::record(stop.event, self.owner.stream) }?;
        Ok(Some(PtxSpan { start, stop }))
    }

    /// The elapsed device time `span` bracketed (Card 552 SC-006). The caller must have synchronized
    /// since [`PtxContext::launch_timed`] recorded `span`, or the events may not have completed yet.
    pub fn span_elapsed(&self, span: &PtxSpan) -> Result<Duration, PtxError> {
        // SAFETY: both events are live and were recorded on this context's stream; the caller is
        // responsible for having synchronized first (documented above).
        let ms = unsafe { result::event::elapsed(span.start.event, span.stop.event) }?;
        Ok(Duration::from_secs_f64(ms as f64 / 1000.0))
    }

    /// Refuse `op` with [`PtxError::CaptureOpen`] while a capture is open on this context.
    fn refuse_during_capture(&self, op: &'static str) -> Result<(), PtxError> {
        if self.capture.is_open() {
            return Err(PtxError::CaptureOpen { op });
        }
        Ok(())
    }

    /// Best-effort exit from stream-capture mode after a mid-capture recording error. CUDA requires a matching
    /// `cuStreamEndCapture` for every `cuStreamBeginCapture`; skipping it leaves the stream stuck in capture
    /// mode, so later calls fail or are recorded into a graph nothing instantiates or destroys.
    /// `cuStreamEndCapture` commonly reports `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` here; that is expected and
    /// not surfaced. Any partial graph is destroyed, never instantiated.
    pub fn abort_capture(&self) {
        let captured = self.capture.finish();
        if let Ok(_guard) = self.owner.current_guard()
            // SAFETY: `stream` is this context's live stream; ending a capture takes no host memory.
            && let Ok(graph) = unsafe { result::stream::end_capture(self.owner.stream) }
        {
            self.owner.driver.destroy_graph(graph);
        }
        // The stream has left capture mode, so the retained buffers may be freed now.
        drop(captured);
    }

    /// A zero-filled device buffer holding `elems` native elements of `storage`, charged to `role`
    /// (Card 547a: the storage-typed, role-typed allocation the executor contract's memory service
    /// needs, instead of one allocator per dtype; mirrors `poot_runtime::Context::alloc_storage`).
    pub fn alloc_storage(
        &self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<PtxBuffer, PtxError> {
        self.refuse_during_capture("alloc_storage")?;
        self.alloc_zeroed("alloc_storage", elems, storage, role)
    }

    /// Overwrite the first `bytes.len()` bytes of `buf`, dtype-agnostic (Card 547a: mirrors
    /// `poot_runtime::Context::write_bytes`). Enqueued on `self.stream` and synchronized, like
    /// [`Self::update_f32`].
    pub fn write_bytes(&self, buf: &PtxBuffer, bytes: &[u8]) -> Result<(), PtxError> {
        self.refuse_during_capture("write_bytes")?;
        let _range = crate::buffer::checked_byte_range("write_bytes", buf, bytes.len())?;
        let dst = buf.ptr();
        // SAFETY: `_range` proved `bytes.len()` lies inside `buf`'s live allocation. The copy runs on
        // this context's stream; the driver stages pageable `bytes` before returning, and the stream
        // is synchronized below before `bytes` can be released. No capture is open (refused above).
        unsafe { result::memcpy_htod_async(dst, bytes, self.owner.stream) }?;
        self.synchronize()?;
        Ok(())
    }

    /// Read the first `out.len()` bytes of `buf`, dtype-agnostic (Card 547a: mirrors
    /// `poot_runtime::Context::read_bytes`). Syncs the stream first so pending writes have completed.
    pub fn read_bytes(&self, buf: &PtxBuffer, out: &mut [u8]) -> Result<(), PtxError> {
        self.refuse_during_capture("read_bytes")?;
        let _range = crate::buffer::checked_byte_range("read_bytes", buf, out.len())?;
        self.synchronize()?;
        // SAFETY: `_range` proved `out.len()` lies inside `buf`'s live allocation, and the stream has
        // finished every write to it.
        unsafe { result::memcpy_dtoh_sync(out, buf.ptr()) }?;
        Ok(())
    }

    /// Read a device buffer back to host f32 (D2H). Syncs the stream first so pending writes have completed.
    pub fn download_f32(&self, b: &PtxBuffer) -> Result<Vec<f32>, PtxError> {
        self.refuse_during_capture("download_f32")?;
        self.download_elements("download_f32", b, BufferStorage::f32())
    }

    /// Read a device buffer back to host i32. The bytes are unchanged, so callers may also use this for
    /// u32 kernel buffers when they need the corresponding bit pattern (for example, the global CAS
    /// counter probe). Syncs the compute stream before the copy, matching [`Self::download_f32`].
    pub fn download_i32(&self, b: &PtxBuffer) -> Result<Vec<i32>, PtxError> {
        self.refuse_during_capture("download_i32")?;
        self.download_elements("download_i32", b, BufferStorage::i32())
    }

    /// Dispatch a compiler-produced `kernel` (card 608) reading the persistent `ins` buffers and writing
    /// `out` (already allocated). `ins`/`out` are checked against `kernel`'s argument schema before
    /// anything is built (SC-002). `block` is the CUDA block shape; `threads` the per-axis logical thread
    /// counts (grid = ceil(threads / block)). Enqueues on the stream and returns without syncing (the
    /// resident fast path).
    ///
    /// `ins_lens`/`out_len` (Card 547b) are each buffer's own planned element count,
    /// index-aligned with `ins` plus `out` last: the kernel ABI's `(ptr, len)` pairs carry these,
    /// never `PtxBuffer::elem_count()` - a caller with no plan of its own (a test, a probe) passes
    /// each buffer's own `elem_count()` explicitly instead.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_dev(
        &self,
        // Card 626: unused now that the legacy label-keyed Profiler is gone (its only reader); this
        // crate's `function` cache keys on the PTX text hash + entry name instead. Kept for call-site
        // symmetry with `poot_runtime::Context::dispatch_dev`, which still uses its own `label` as a
        // pipeline-cache key.
        _label: &str,
        kernel: &CompiledKernel,
        block: [u32; 3],
        threads: [u32; 3],
        ins: &[&PtxBuffer],
        ins_lens: &[u32],
        out: &PtxBuffer,
        out_len: u32,
    ) -> Result<(), PtxError> {
        check_ptx_args(kernel, ins, out)?;
        debug_assert_eq!(
            ins.len(),
            ins_lens.len(),
            "ins_lens must be index-aligned with ins"
        );
        let (ptx, entry) = ptx_text_and_entry(kernel)?;
        let func = self.function(ptx, entry)?;

        let args = KernelArgs::pack(&self.capture, ins, ins_lens, out, out_len);
        let mut params = args.params();

        let grid = (
            threads[0].div_ceil(block[0].max(1)),
            threads[1].div_ceil(block[1].max(1)),
            threads[2].div_ceil(block[2].max(1)),
        );
        let block = (block[0], block[1], block[2]);

        // SAFETY: `func` belongs to a module the owner keeps loaded, and `params` follows the poot
        // NVPTX slice ABI: one `(ptr, i64 len)` pair per kernel slice parameter, each pointer a live
        // allocation of `len` elements. `args` outlives the call. The buffers stay allocated while the kernel
        // runs: an open capture retains them for its graph, and otherwise freeing a buffer (`cuMemFree`)
        // waits for the device work in flight. The kernel text is trusted to index its slices within `len`, as
        // poot-codegen's emitter does; this runtime does not verify it.
        unsafe {
            result::launch_kernel(func, grid, block, 0, self.owner.stream, &mut params)?;
        }
        Ok(())
    }

    /// Load (cached) the `CUfunction` for `entry` from the `ptx` module text.
    fn function(&self, ptx: &str, entry: &str) -> Result<sys::CUfunction, PtxError> {
        let key = format!("{:016x}:{entry}", fnv1a(ptx));
        if let Some(f) = self.funcs.borrow().get(&key) {
            return Ok(*f);
        }
        let entry_c = CString::new(entry).map_err(|e| PtxError::BadKernel(e.to_string()))?;
        let ptx_c = CString::new(ptx).map_err(|e| PtxError::BadKernel(e.to_string()))?;
        // SAFETY: `ptx_c` is a NUL-terminated PTX image that outlives the call.
        let module = unsafe { result::module::load_data(ptx_c.as_ptr().cast()) }?;
        let guard = ModuleConstruction::new(module, Arc::clone(&self.owner));
        // SAFETY: `module` was just loaded and is unloaded only by `guard` or the owner.
        let func = unsafe { result::module::get_function(module, entry_c) }?;
        guard.finish();
        self.funcs.borrow_mut().insert(key, func);
        Ok(func)
    }
}

/// The PTX text and entry-point name of a compiler-produced `kernel` (card 608), or a typed error if it
/// was compiled for another backend.
fn ptx_text_and_entry(kernel: &CompiledKernel) -> Result<(&str, &str), PtxError> {
    match kernel.code() {
        KernelCode::Ptx(text) => Ok((text.as_ref(), kernel.entry_point())),
        _ => Err(PtxError::WrongKernelTarget {
            actual: kernel.target(),
        }),
    }
}

/// Check `ins` then `out` (inputs first, the single output last) against `kernel`'s argument schema
/// (card 608, SC-002): a release check on argument count and each buffer's native element kind, before
/// any submission.
fn check_ptx_args(
    kernel: &CompiledKernel,
    ins: &[&PtxBuffer],
    out: &PtxBuffer,
) -> Result<(), PtxError> {
    let elements: Vec<ElementKind> = ins
        .iter()
        .map(|b| b.element())
        .chain(std::iter::once(out.element()))
        .collect();
    poot_runtime_common::check_kernel_args(kernel, &elements)?;
    Ok(())
}

/// `cuMemAlloc` of `bytes` for [`allocate_owned`], which wraps the pointer in its owning buffer before
/// initializing every byte.
fn device_alloc(bytes: usize) -> Result<sys::CUdeviceptr, PtxError> {
    // SAFETY: the returned memory is uninitialized; each caller passes it to `allocate_owned`, whose
    // initializer writes all `bytes` (upload copy or memset) before the buffer is returned.
    unsafe { result::malloc_sync(bytes) }.map_err(PtxError::Driver)
}

/// The kernel ABI's `(ptr, i64 len)` pair per buffer, inputs then the output. Packing is the only way a
/// dispatch obtains its launch parameters, and it retains every buffer in the open capture (if any), so a
/// captured graph can never outlive a buffer whose address it recorded.
pub(crate) struct KernelArgs {
    dptrs: Vec<sys::CUdeviceptr>,
    lens: Vec<i64>,
}

impl KernelArgs {
    /// `lens`' unit is the caller's own (Card 547b): each buffer's planned element
    /// count, index-aligned with `ins` plus `out` last - never `PtxBuffer::elem_count()` (an arena
    /// slot's buffer can be larger than this particular operand's own need).
    pub(crate) fn pack(
        capture: &CaptureRetention,
        ins: &[&PtxBuffer],
        ins_lens: &[u32],
        out: &PtxBuffer,
        out_len: u32,
    ) -> Self {
        let buffers = ins.iter().copied().chain(std::iter::once(out));
        let lens = ins_lens.iter().copied().chain(std::iter::once(out_len));
        let mut dptrs = Vec::with_capacity(ins.len() + 1);
        let mut packed_lens = Vec::with_capacity(ins.len() + 1);
        for (buffer, len) in buffers.zip(lens) {
            capture.retain(buffer);
            dptrs.push(buffer.ptr());
            packed_lens.push(i64::from(len));
        }
        Self {
            dptrs,
            lens: packed_lens,
        }
    }

    /// `cuLaunchKernel`'s parameter array: one pointer per scalar, interleaved `(ptr, len)` per buffer. The
    /// pointers borrow `self`, which must outlive the launch call.
    pub(crate) fn params(&self) -> Vec<*mut c_void> {
        self.dptrs
            .iter()
            .zip(&self.lens)
            .flat_map(|(dptr, len)| {
                [
                    std::ptr::from_ref(dptr).cast_mut().cast::<c_void>(),
                    std::ptr::from_ref(len).cast_mut().cast::<c_void>(),
                ]
            })
            .collect()
    }
}

/// Round an f32 to bf16 (the high 16 bits, round-to-nearest-even). bf16 shares f32's exponent, so only the
/// mantissa is rounded and the range is unchanged. Public PTX path; the body lives in
/// `poot-runtime-common` (quiet NaN stays quiet with its sign, sNaN is quieted).
pub use poot_runtime_common::f32_to_bf16;

#[cfg(test)]
mod caps_tests {
    use super::*;
    use poot_target::{Queried, SubgroupSupport, TensorCoreSupport};

    fn sentinel(compute_capability_major: u32) -> CudaAttributes {
        CudaAttributes {
            shared_memory_per_block: 11_111,
            max_grid: [22, 33, 44],
            max_block: [55, 66, 77],
            max_threads_per_block: 88,
            warp_size: 16,
            compute_capability_major,
            multiprocessor_count: 99,
        }
    }

    /// SC-003: every attribute lands in the field that names it (no two sentinels are equal, so a swap
    /// or a default substituted for a query cannot pass). Mutation: report `ptx_default()`'s 49152 LDS
    /// or its 32-lane warp, and the equality fails on that field.
    #[test]
    fn every_attribute_lands_in_its_own_caps_field() {
        let caps = caps_from_attributes(123_456, &sentinel(9));
        assert_eq!(caps.max_buffer_bytes, 123_456);
        assert_eq!(caps.lds_bytes, 11_111);
        assert_eq!(caps.max_grid, [22, 33, 44]);
        assert_eq!(caps.max_workgroup_size, [55, 66, 77]);
        assert_eq!(caps.max_workgroup_invocations, 88);
        assert_eq!(
            caps.subgroup,
            Queried::Known(SubgroupSupport::Present {
                min_size: 16,
                max_size: 16
            })
        );
        assert_eq!(caps.compute_units, 99);
        assert_eq!(caps.tensor_core, TensorCoreSupport::NvidiaWmma16x16x16Sm80);
    }

    /// A pre-Ampere device is a confirmed absence of the bf16 fragments poot's kernels use.
    #[test]
    fn a_pre_ampere_device_has_no_matrix_hardware_for_poots_kernels() {
        assert_eq!(
            caps_from_attributes(1, &sentinel(7)).tensor_core,
            TensorCoreSupport::None
        );
    }
}
