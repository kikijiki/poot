//! CUDA P2P transport + multi-device ring all-reduce (card 049a Phase 1), the transport under the captured
//! multi-device executor in [`crate::multi_device`].
//!
//! [`crate::ring`] implements the ring reduce-scatter + all-gather algorithm as pure functions over host
//! buffers, through the CPU [`crate::ring::ChunkTransport`] seam. This module supplies the cross-device
//! transport: [`PtxP2PTransport`] moves an eager chunk from rank i's device buffer into rank j's device
//! buffer with `cuMemcpyPeerAsync` (cudarc `result::memcpy_peer_async`), falling back to device-to-host /
//! host-to-device staging where `cuDeviceCanAccessPeer` is false (FR-005 / FR-006). Card 408's captured
//! path first requires successful `cuCtxEnablePeerAccess`, then records a capture-supported
//! `cuMemcpyDtoDAsync` against the peer-mapped pointer. [`p2p_ring_all_reduce_sum`] runs the same ring
//! schedule as
//! [`crate::ring::ring_all_reduce_sum`] (only the transport and the on-device `Binary(Add)` accumulate
//! differ) so every rank's device buffer ends holding the elementwise sum (SC-001).
//!
//! # Transport seam note (host slices vs device pointers)
//!
//! `ring::ChunkTransport::move_chunk` takes host `&[f32]` / `&mut [f32]`, which cannot express a
//! device-to-device peer copy (that needs two device pointers). So the GPU ring here reimplements the
//! schedule over device buffers and drives the staging form of [`PtxP2PTransport::move_chunk`], the
//! f32 [`RankDeviceBuffer`] analogue (source buffer chunk, destination buffer chunk, both checked
//! against the buffers' real extents). Card 408's general communication path will use
//! `PtxP2PTransport::move_bytes_capturable` so packed payloads keep byte-exact offsets and
//! lengths while rejecting a host-staged route before capture. The eager ring's private byte helper may
//! still use a per-transfer host bounce when P2P is unavailable; that fallback is explicitly not a warm
//! captured-replay path. The ring structure (chunk partition + step schedule) is shared via
//! [`crate::ring::chunk_range`], so the two paths cannot drift.
//!
//! # Execution model
//!
//! Eager path: correctness over throughput. One host thread owns all N per-rank `CUcontext`s (each a
//! primary context bound to its GPU ordinal) and drives the ring with a per-step device barrier.
//! Cross-device ordering within a reduce-scatter step uses events: the source stream records after each
//! peer copy (`cuEventRecord`), the destination stream waits before the accumulate (`cuStreamWaitEvent`).
//! Card 408's captured executor reuses the same single-thread owner with separately instantiated,
//! topologically ordered graph segments per rank. The eager thread-per-rank worker remains out of scope.
//!
//! Builds with no CUDA toolkit and no GPU (cudarc dynamic-loading dlopens `libcuda.so.1` at runtime); it
//! only runs on a machine with >= 2 NVIDIA GPUs, so the test below is `#[ignore]` and skips cleanly when
//! `cuDeviceGetCount() < 2`.

use std::cell::Cell;
use std::ffi::{CString, c_void};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{DriverError, result, sys};
use poot_kernel_ir::BinOp;

use crate::ring::chunk_range;

/// Errors from the multi-device P2P ring.
#[derive(Debug, thiserror::Error)]
pub enum P2pError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("kernel compile error: {0}")]
    Compile(#[source] Box<poot_codegen::CompileError>),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("rank {rank} is outside world size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },
    #[error("rank {rank} cannot be represented as a CUDA device ordinal")]
    DeviceOrdinalConversion { rank: usize },
    #[error("received {actual} rank buffers for world size {world_size}")]
    BufferCountMismatch { actual: usize, world_size: usize },
    #[error("declared peer table for world size {world_size} has shape {rows}x{cols}")]
    PeerTableShape {
        world_size: usize,
        rows: usize,
        cols: usize,
    },
    #[error("rank buffer at index {index} belongs to rank {rank}")]
    BufferRankMismatch { index: usize, rank: usize },
    #[error("{role} byte range offset {offset} + length {byte_len} overflowed usize")]
    ByteRangeOverflow {
        role: &'static str,
        offset: usize,
        byte_len: usize,
    },
    #[error("{role} element count {elements} overflowed its f32 byte length")]
    ElementByteLenOverflow { role: &'static str, elements: usize },
    #[error("{role} byte range [{offset}, {end}) exceeds buffer extent {extent}")]
    ByteRangeOutOfBounds {
        role: &'static str,
        offset: usize,
        end: usize,
        extent: usize,
    },
    #[error("{role} byte offset {offset} cannot be represented by a CUDA device pointer")]
    ByteOffsetConversion { role: &'static str, offset: usize },
    #[error("{role} device pointer {base:#x} + byte end {end} overflowed")]
    DevicePointerOverflow {
        role: &'static str,
        base: sys::CUdeviceptr,
        end: u64,
    },
    #[error("captured transfer {src_rank} -> {dst_rank} would require host staging")]
    NonCapturableHostStaging { src_rank: usize, dst_rank: usize },
    #[error("capturable transfer event pool for rank {rank} is exhausted at event {index}")]
    CaptureEventPoolExhausted { rank: usize, index: usize },
    #[error("a captured communication phase is already active")]
    CapturePhaseAlreadyActive,
    #[error("no captured communication phase is active")]
    CapturePhaseMissing,
    #[error("captured communication phase has no ranks")]
    EmptyCapturePhase,
    #[error("captured communication phase names rank {rank} more than once")]
    DuplicateCapturePhaseRank { rank: usize },
    #[error(
        "replay input write of {actual} bytes does not match the {expected}-byte resident buffer"
    )]
    RankByteLenMismatch { actual: usize, expected: usize },
    #[error("rank buffer was allocated by a different P2P transport")]
    ForeignRankBuffer,
    #[error("{0}")]
    Msg(String),
}

/// Source of [`PtxP2PTransport::id`]: distinct for every transport in the process.
static NEXT_TRANSPORT_ID: AtomicU64 = AtomicU64::new(0);

/// A declared directed route is usable by stream capture only after the source context has successfully
/// enabled direct access to the destination context. Keeping that state distinct from hardware
/// capability prevents a captured transfer from lazily performing peer setup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CapturablePeerAccess {
    Unavailable,
    Enabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PeerInitializationOperation {
    QueryCapability,
    /// Deliberately unused by correct initialization; retained so the forbidden constructor mutation is
    /// executable and observable in the host seam.
    #[allow(dead_code)]
    EnablePeerAccess,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CapturedDtoDCopy {
    current_context: sys::CUcontext,
    stream: sys::CUstream,
    destination: sys::CUdeviceptr,
    source: sys::CUdeviceptr,
    byte_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EagerPeerCopy {
    src_rank: usize,
    dst_rank: usize,
    current_context: sys::CUcontext,
    stream: sys::CUstream,
    destination_context: sys::CUcontext,
    destination: sys::CUdeviceptr,
    source_context: sys::CUcontext,
    source: sys::CUdeviceptr,
    byte_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PeerAccessEnable {
    src_rank: usize,
    dst_rank: usize,
    current_context: sys::CUcontext,
    peer_context: sys::CUcontext,
}

/// Captured and eager transfers deliberately expose different typed operations to the driver boundary.
trait CudaPeerCopyCalls {
    /// Activate the source context and return the driver's unclassified enable result. Context activation
    /// failures remain typed errors; callers own whether a returned CUDA status is strict or best-effort.
    fn enable_peer_access_raw(&mut self, args: PeerAccessEnable)
    -> Result<sys::CUresult, P2pError>;

    /// Ring and capture preparation require successful peer enablement. Standalone compatibility uses
    /// [`Self::enable_peer_access_raw`] directly and deliberately preserves best-effort driver statuses.
    fn enable_peer_access(&mut self, args: PeerAccessEnable) -> Result<(), P2pError> {
        let result = self.enable_peer_access_raw(args)?;
        enable_peer_access_with(true, args.src_rank, args.dst_rank, || result)
    }

    fn memcpy_dtod_async(&mut self, args: CapturedDtoDCopy) -> Result<(), P2pError>;
    fn memcpy_peer_async(
        &mut self,
        peer_access: &EagerPeerAccessPrepared,
        args: EagerPeerCopy,
    ) -> Result<(), P2pError>;
}

struct RawCudaPeerCopyCalls;

impl From<poot_codegen::CompileError> for P2pError {
    fn from(error: poot_codegen::CompileError) -> Self {
        Self::Compile(Box::new(error))
    }
}

/// One rank's CUDA state: a primary context bound to its GPU ordinal, a non-blocking stream, a sync-only
/// event, and the `Binary(Add)` accumulate function loaded into this context. `!Send` (holds raw CUDA
/// handles): created on and never leaving the driving thread.
struct RankContext {
    owner: Rc<RankOwner>,
    device: sys::CUdevice,
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    /// Recorded on `stream` after a peer copy so a peer's stream can wait on the copy (reduce-scatter).
    event: Rc<RankEvent>,
    /// The elementwise `Binary(Add)` kernel (`c[i] = a[i] + b[i]`) in THIS context's module.
    add_fn: sys::CUfunction,
    /// One event per captured transfer sourced by this rank. Reusing `event` for two graph edges would
    /// let a later record overwrite the generation an earlier wait was meant to observe.
    capture_events: Vec<Rc<RankEvent>>,
    next_capture_event: Cell<usize>,
}

struct CapturedRankGraph {
    rank: usize,
    owner: Rc<RankOwner>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    /// CUDA graph nodes bake these addresses and event handles by value.
    _buffers: Vec<RankByteBuffer>,
    _events: Vec<Rc<RankEvent>>,
}

/// One replay epoch segment. Graphs are stored in submission order. In a remote-transfer phase the
/// source graph is first and establishes the current event generation before the destination wait graph
/// is submitted; phases never rely on whole-rank launch order.
struct CapturedGraphPhase {
    graphs: Vec<CapturedRankGraph>,
}

struct UninstantiatedRankGraph {
    owner: Rc<RankOwner>,
    graph: sys::CUgraph,
}

/// Shared native parent for every buffer, event, and graph created on one rank. Children retain this
/// owner so failure unwinding and arbitrary Rust drop order cannot release the primary context first.
struct RankOwner {
    device: sys::CUdevice,
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    module: sys::CUmodule,
}

struct RankEvent {
    owner: Rc<RankOwner>,
    event: sys::CUevent,
}

struct RankByteAllocation {
    owner: Rc<RankOwner>,
    ptr: sys::CUdeviceptr,
    byte_len: usize,
}

/// A PTX module loaded into one rank's context for a compute segment (ADR-0099 item 1). The module
/// must outlive every captured graph that launched a function from it, so the transport keeps these
/// after `captured_phases` (Rust drops fields in declaration order).
struct RankModule {
    owner: Rc<RankOwner>,
    module: sys::CUmodule,
}

impl Drop for RankModule {
    fn drop(&mut self) {
        let Ok(_guard) = CurrentRankContext::activate(self.owner.ctx) else {
            return;
        };
        unsafe {
            let _ = result::module::unload(self.module);
        }
    }
}

/// Scope one raw-driver operation to its owning context and restore the exact prior current context.
/// The prior value may be null; preserving that state matters when the last primary-context retain drops.
struct CurrentRankContext {
    prior: Option<Option<sys::CUcontext>>,
}

impl CurrentRankContext {
    fn activate(ctx: sys::CUcontext) -> Result<Self, DriverError> {
        let prior = result::ctx::get_current()?;
        if let Err(error) = unsafe { result::ctx::set_current(ctx) } {
            let _ = unsafe { result::ctx::set_current(prior.unwrap_or(std::ptr::null_mut())) };
            return Err(error);
        }
        Ok(Self { prior: Some(prior) })
    }

    fn restore_before_owner_release(mut self, owner: sys::CUcontext) -> Result<(), DriverError> {
        let prior = self.prior.take().unwrap_or(None);
        let restore = if prior == Some(owner) { None } else { prior };
        unsafe { result::ctx::set_current(restore.unwrap_or(std::ptr::null_mut())) }
    }
}

impl Drop for CurrentRankContext {
    fn drop(&mut self) {
        if let Some(prior) = self.prior.take() {
            let _ = unsafe { result::ctx::set_current(prior.unwrap_or(std::ptr::null_mut())) };
        }
    }
}

impl CudaPeerCopyCalls for RawCudaPeerCopyCalls {
    fn enable_peer_access_raw(
        &mut self,
        args: PeerAccessEnable,
    ) -> Result<sys::CUresult, P2pError> {
        let _guard = CurrentRankContext::activate(args.current_context)?;
        Ok(unsafe { sys::cuCtxEnablePeerAccess(args.peer_context, 0) })
    }

    fn memcpy_dtod_async(&mut self, args: CapturedDtoDCopy) -> Result<(), P2pError> {
        unsafe {
            result::ctx::set_current(args.current_context)?;
            result::memcpy_dtod_async(args.destination, args.source, args.byte_len, args.stream)?;
        }
        Ok(())
    }

    fn memcpy_peer_async(
        &mut self,
        peer_access: &EagerPeerAccessPrepared,
        args: EagerPeerCopy,
    ) -> Result<(), P2pError> {
        peer_access.require_direct_route(args.src_rank, args.dst_rank)?;
        unsafe {
            result::ctx::set_current(args.current_context)?;
            result::memcpy_peer_async(
                args.destination_context,
                args.destination,
                args.source_context,
                args.source,
                args.byte_len,
                args.stream,
            )?;
        }
        Ok(())
    }
}

impl RankOwner {
    fn new(device: sys::CUdevice, ptx: &CString) -> Result<Self, P2pError> {
        let ctx = unsafe { result::primary_ctx::retain(device) }?;
        let mut owner = Self {
            device,
            ctx,
            stream: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
        };
        let _guard = CurrentRankContext::activate(ctx)?;
        owner.stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        owner.module = unsafe { result::module::load_data(ptx.as_ptr() as *const _) }?;
        Ok(owner)
    }
}

impl Drop for RankOwner {
    fn drop(&mut self) {
        let Ok(guard) = CurrentRankContext::activate(self.ctx) else {
            // Without a known current context, destroying any child or releasing the primary retain is
            // unsafe. A bounded leak is preferable to invalidating a foreign current context.
            return;
        };
        unsafe {
            if !self.module.is_null() {
                let _ = result::module::unload(self.module);
            }
            if !self.stream.is_null() {
                let _ = result::stream::destroy(self.stream);
            }
        }
        if guard.restore_before_owner_release(self.ctx).is_ok() {
            unsafe {
                let _ = result::primary_ctx::release(self.device);
            }
        }
    }
}

impl RankEvent {
    fn new(owner: Rc<RankOwner>) -> Result<Self, P2pError> {
        let _guard = CurrentRankContext::activate(owner.ctx)?;
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
        Ok(Self { owner, event })
    }
}

impl Drop for RankEvent {
    fn drop(&mut self) {
        let Ok(_guard) = CurrentRankContext::activate(self.owner.ctx) else {
            return;
        };
        unsafe {
            let _ = result::event::destroy(self.event);
        }
    }
}

impl Drop for RankByteAllocation {
    fn drop(&mut self) {
        let Ok(_guard) = CurrentRankContext::activate(self.owner.ctx) else {
            return;
        };
        unsafe {
            let _ = result::free_sync(self.ptr);
        }
    }
}

impl Drop for UninstantiatedRankGraph {
    fn drop(&mut self) {
        if self.graph.is_null() {
            return;
        }
        let Ok(_guard) = CurrentRankContext::activate(self.owner.ctx) else {
            return;
        };
        unsafe {
            let _ = result::graph::destroy(self.graph);
        }
    }
}

impl Drop for CapturedRankGraph {
    fn drop(&mut self) {
        let Ok(_guard) = CurrentRankContext::activate(self.owner.ctx) else {
            return;
        };
        unsafe {
            if !self.exec.is_null() {
                let _ = result::graph::exec_destroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = result::graph::destroy(self.graph);
            }
        }
    }
}

/// An exact-byte device allocation owned jointly by the executor and transport. The allocation retains
/// its rank owner, so it is freed under the originating context even during partial-construction unwind.
#[derive(Clone)]
pub(crate) struct RankByteBuffer {
    rank: usize,
    allocation: Rc<RankByteAllocation>,
}

impl RankByteBuffer {
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn ptr(&self) -> sys::CUdeviceptr {
        self.allocation.ptr
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.allocation.byte_len
    }
}

/// A pre-capture external event owned by one rank context. The transport owns the CUDA handle; this
/// token only identifies which event a plan dependency records or waits on.
#[derive(Clone)]
pub(crate) struct CapturableEvent {
    owner_rank: usize,
    event: Rc<RankEvent>,
}

/// Opaque proof that eager peer-access preparation completed for either every successor edge in one
/// ring or one standalone route. Only the child module that performs preparation can construct it.
use eager_peer_access::Prepared as EagerPeerAccessPrepared;

/// A per-rank device buffer plus its reduce-scatter staging chunk. Plain device pointers (`u64`), owned
/// by the [`PtxP2PTransport`] that allocated them and freed by [`PtxP2PTransport::free_rank_buffers`].
/// The pointers stay valid while that transport retains its rank's primary context, so every public
/// method refuses a buffer another transport allocated ([`P2pError::ForeignRankBuffer`]).
pub struct RankDeviceBuffer {
    /// [`PtxP2PTransport::id`] of the allocating transport.
    transport: u64,
    rank: usize,
    /// The rank's `len`-element f32 buffer.
    ptr: sys::CUdeviceptr,
    /// A `max_chunk`-element scratch receiving one incoming chunk per reduce-scatter step.
    staging: sys::CUdeviceptr,
    len: usize,
    staging_len: usize,
}

/// The CUDA P2P transport + ring driver. Holds every rank's context, device pointers, and the peerability
/// table as plain values. One host thread drives all ranks; contexts are set current as needed.
pub struct PtxP2PTransport {
    /// Process-unique identity, stamped into every [`RankDeviceBuffer`] this transport allocates.
    id: u64,
    ranks: Vec<RankContext>,
    /// `peerable[i][j]` = rank i's device can peer-access rank j's (`cuDeviceCanAccessPeer`); else the
    /// copy from i to j host-stages (FR-006).
    peerable: Vec<Vec<bool>>,
    /// Direct access successfully enabled before any Card 408 allocation or stream capture. Unlike the
    /// hardware capability table, this is the typed prerequisite for a capturable DtoD copy.
    capturable_peer_access: Vec<Vec<CapturablePeerAccess>>,
    /// Workgroup shape of the `Binary(Add)` accumulate kernel (grid = ceil(count / wg.x)).
    add_wg: [u32; 3],
    owned_byte_buffers: Vec<RankByteBuffer>,
    dependency_events: Vec<CapturableEvent>,
    captured_phases: Vec<CapturedGraphPhase>,
    /// PTX modules loaded for compute segments. Declared after `captured_phases` so every captured
    /// graph is destroyed before the module its kernels came from is unloaded.
    owned_modules: Vec<RankModule>,
    active_capture_ranks: Option<Vec<usize>>,
}

impl PtxP2PTransport {
    /// Initialise CUDA, retain and bind one primary context per ordinal `0..world_size`, create a
    /// non-blocking stream + event and load the `Binary(Add)` kernel per rank, then query the hardware
    /// peer table. Capture-only peer mappings are enabled later for admitted routes; eager construction
    /// never depends on that setup. Requires `world_size` physical GPUs; the caller checks
    /// `device_count` first.
    pub fn new(world_size: usize) -> Result<Self, P2pError> {
        result::init()?;

        // Compile the accumulate kernel once, then load its module into each rank's context.
        let add_body = poot_kernelgen::binary("p2p_add", BinOp::Add);
        let add_wg = add_body.workgroup_size;
        let add_ptx = compile_add_ptx(&add_body)?;
        let add_ptx_c = CString::new(add_ptx).map_err(|e| P2pError::Msg(e.to_string()))?;

        let mut ranks = Vec::with_capacity(world_size);
        for r in 0..world_size {
            let ordinal =
                i32::try_from(r).map_err(|_| P2pError::DeviceOrdinalConversion { rank: r })?;
            let device = result::device::get(ordinal)?;
            let owner = Rc::new(RankOwner::new(device, &add_ptx_c)?);
            let _guard = CurrentRankContext::activate(owner.ctx)?;
            let entry = CString::new("p2p_add").unwrap();
            let add_fn = unsafe { result::module::get_function(owner.module, entry) }?;
            let event = Rc::new(RankEvent::new(Rc::clone(&owner))?);
            ranks.push(RankContext {
                device,
                ctx: owner.ctx,
                stream: owner.stream,
                owner,
                event,
                add_fn,
                capture_events: Vec::new(),
                next_capture_event: Cell::new(0),
            });
        }

        // The eager table is only a hardware fact. Capture-only peer mappings remain unavailable until
        // the executor supplies its already-admitted directed routes.
        let peerable =
            discover_peer_capabilities_with(world_size, |operation, i, j| match operation {
                PeerInitializationOperation::QueryCapability => {
                    let mut can: i32 = 0;
                    unsafe {
                        sys::cuDeviceCanAccessPeer(&mut can, ranks[i].device, ranks[j].device)
                    }
                    .result()?;
                    Ok(can != 0)
                }
                PeerInitializationOperation::EnablePeerAccess => {
                    let _guard = CurrentRankContext::activate(ranks[i].ctx)?;
                    enable_peer_access_with(true, i, j, || unsafe {
                        sys::cuCtxEnablePeerAccess(ranks[j].ctx, 0)
                    })?;
                    Ok(true)
                }
            })?;
        let capturable_peer_access =
            vec![vec![CapturablePeerAccess::Unavailable; world_size]; world_size];

        Ok(Self {
            id: NEXT_TRANSPORT_ID.fetch_add(1, Ordering::Relaxed),
            ranks,
            peerable,
            capturable_peer_access,
            add_wg,
            owned_byte_buffers: Vec::new(),
            dependency_events: Vec::new(),
            captured_phases: Vec::new(),
            owned_modules: Vec::new(),
            active_capture_ranks: None,
        })
    }

    /// The number of ranks (GPUs) this transport drives.
    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }

    fn rank(&self, rank: usize) -> Result<&RankContext, P2pError> {
        self.ranks.get(rank).ok_or(P2pError::RankOutOfRange {
            rank,
            world_size: self.world_size(),
        })
    }

    /// Whether rank `i`'s device can peer-access rank `j`'s (else that copy host-stages).
    ///
    /// This preserves the original public API. Callers must pass ranks below [`Self::world_size`], as
    /// before; checked internal transfer paths use [`Self::checked_peerable`] instead.
    pub fn peerable(&self, i: usize, j: usize) -> bool {
        self.peerable[i][j]
    }

    fn checked_peerable(&self, i: usize, j: usize) -> Result<bool, P2pError> {
        self.rank(i)?;
        self.rank(j)?;
        Ok(self.peerable[i][j])
    }

    /// Enable only Card 360's admitted directed routes before buffers are allocated or a stream begins
    /// capture. The eager hardware-capability table remains unchanged.
    pub(crate) fn prepare_capturable_peer_access(
        &mut self,
        declared: &[Vec<bool>],
    ) -> Result<(), P2pError> {
        let world_size = self.world_size();
        if declared.len() != world_size || declared.iter().any(|row| row.len() != world_size) {
            return Err(P2pError::PeerTableShape {
                world_size,
                rows: declared.len(),
                cols: declared.first().map_or(0, Vec::len),
            });
        }
        let contexts: Vec<_> = self.ranks.iter().map(|rank| rank.ctx).collect();
        let mut calls = RawCudaPeerCopyCalls;
        prepare_capturable_peer_access_with(
            &self.peerable,
            &mut self.capturable_peer_access,
            declared,
            &contexts,
            |src_rank, dst_rank, src_context, dst_context| {
                calls.enable_peer_access(PeerAccessEnable {
                    src_rank,
                    dst_rank,
                    current_context: src_context,
                    peer_context: dst_context,
                })
            },
        )
    }

    /// Enable only the directed successor edges used by the eager ring. Capture preparation remains a
    /// separate Card 360 topology-scoped owner, and construction remains capability discovery only.
    fn prepare_eager_ring_peer_access(&self) -> Result<EagerPeerAccessPrepared, P2pError> {
        let contexts: Vec<_> = self.ranks.iter().map(|rank| rank.ctx).collect();
        let mut calls = RawCudaPeerCopyCalls;
        prepare_eager_ring_peer_access_with(&self.peerable, &contexts, &mut calls)
    }

    /// Prepare one standalone eager route without broadening constructor or capture-route ownership.
    fn prepare_eager_route_peer_access(
        &self,
        src_rank: usize,
        dst_rank: usize,
    ) -> Result<EagerPeerAccessPrepared, P2pError> {
        let src = self.rank(src_rank)?;
        let dst = self.rank(dst_rank)?;
        let mut calls = RawCudaPeerCopyCalls;
        prepare_eager_route_peer_access_with(
            self.peerable[src_rank][dst_rank],
            src_rank,
            dst_rank,
            src.ctx,
            dst.ctx,
            &mut calls,
        )
    }

    fn checked_capturable_peer_access(
        &self,
        src_rank: usize,
        dst_rank: usize,
    ) -> Result<CapturablePeerAccess, P2pError> {
        self.rank(src_rank)?;
        self.rank(dst_rank)?;
        Ok(self.capturable_peer_access[src_rank][dst_rank])
    }

    /// Allocate and initialize one exact-byte resident buffer on `rank`. Allocation and upload happen
    /// before capture; replay only reuses the stable address recorded in the graph.
    pub(crate) fn upload_rank_bytes(
        &mut self,
        rank: usize,
        bytes: &[u8],
    ) -> Result<RankByteBuffer, P2pError> {
        let rank_context = self.rank(rank)?;
        unsafe { result::ctx::set_current(rank_context.ctx) }?;
        let ptr = unsafe { result::malloc_sync(bytes.len().max(1)) }?;
        let buffer = RankByteBuffer {
            rank,
            allocation: Rc::new(RankByteAllocation {
                owner: Rc::clone(&rank_context.owner),
                ptr,
                byte_len: bytes.len(),
            }),
        };
        self.owned_byte_buffers.push(buffer.clone());
        if !bytes.is_empty() {
            unsafe { result::memcpy_htod_sync(ptr, bytes) }?;
        }
        Ok(buffer)
    }

    /// Read an exact-byte resident buffer after synchronizing its rank stream.
    pub(crate) fn download_rank_bytes(&self, buffer: &RankByteBuffer) -> Result<Vec<u8>, P2pError> {
        let rank = self.rank(buffer.rank)?;
        unsafe {
            result::ctx::set_current(rank.ctx)?;
            result::stream::synchronize(rank.stream)?;
        }
        let mut bytes = vec![0u8; buffer.byte_len()];
        if !bytes.is_empty() {
            unsafe { result::memcpy_dtoh_sync(&mut bytes, buffer.ptr()) }?;
        }
        Ok(bytes)
    }

    /// Rewrite one resident buffer's content before a replay (ADR-0099 item 3). The copy is enqueued
    /// on the rank's own stream, so it lands in stream order after every earlier launch and before
    /// every later graph launch, and the rank is synchronized before returning so the caller's host
    /// bytes need not outlive this call.
    #[cfg(test)]
    pub(crate) fn write_rank_bytes(
        &self,
        buffer: &RankByteBuffer,
        bytes: &[u8],
    ) -> Result<(), P2pError> {
        if bytes.len() != buffer.byte_len() {
            return Err(P2pError::RankByteLenMismatch {
                actual: bytes.len(),
                expected: buffer.byte_len(),
            });
        }
        let rank = self.rank(buffer.rank)?;
        unsafe {
            result::ctx::set_current(rank.ctx)?;
            if !bytes.is_empty() {
                result::memcpy_htod_async(buffer.ptr(), bytes, rank.stream)?;
            }
            result::stream::synchronize(rank.stream)?;
        }
        Ok(())
    }

    /// Load one compute segment's PTX program into `rank`'s context and return its entry point. Must
    /// run before any stream enters capture mode: module loading is host work the captured graph
    /// must not observe. The module stays loaded until the transport drops, after every captured
    /// graph that launched from it.
    pub(crate) fn load_rank_program(
        &mut self,
        rank: usize,
        ptx: &str,
        entry: &str,
    ) -> Result<sys::CUfunction, P2pError> {
        let (ctx, owner) = {
            let rank_context = self.rank(rank)?;
            (rank_context.ctx, Rc::clone(&rank_context.owner))
        };
        let image = CString::new(ptx).map_err(|error| P2pError::Msg(error.to_string()))?;
        let name = CString::new(entry).map_err(|error| P2pError::Msg(error.to_string()))?;
        let module = unsafe {
            result::ctx::set_current(ctx)?;
            result::module::load_data(image.as_ptr() as *const _)?
        };
        let function = match unsafe { result::module::get_function(module, name) } {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = result::module::unload(module);
                }
                return Err(error.into());
            }
        };
        self.owned_modules.push(RankModule { owner, module });
        Ok(function)
    }

    /// Record one compute segment kernel on `rank`'s stream. During capture this appends a kernel
    /// node to the phase's graph; the caller has already validated every buffer binding.
    pub(crate) fn launch_rank_program(
        &self,
        rank: usize,
        function: sys::CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> Result<(), P2pError> {
        let rank_context = self.rank(rank)?;
        unsafe {
            result::ctx::set_current(rank_context.ctx)?;
            result::launch_kernel(function, grid, block, 0, rank_context.stream, params)?;
        }
        Ok(())
    }

    /// Allocate the distinct source-owned events consumed by captured transfers. This must run before
    /// any stream enters capture mode.
    pub(crate) fn prepare_capturable_transfer_events(
        &mut self,
        per_rank: &[usize],
    ) -> Result<(), P2pError> {
        if per_rank.len() != self.world_size() {
            return Err(P2pError::BufferCountMismatch {
                actual: per_rank.len(),
                world_size: self.world_size(),
            });
        }
        for (rank_index, &count) in per_rank.iter().enumerate() {
            let rank = &mut self.ranks[rank_index];
            unsafe { result::ctx::set_current(rank.ctx) }?;
            rank.capture_events.clear();
            rank.capture_events.reserve(count);
            for _ in 0..count {
                rank.capture_events
                    .push(Rc::new(RankEvent::new(Rc::clone(&rank.owner))?));
            }
            rank.next_capture_event.set(0);
        }
        Ok(())
    }

    pub(crate) fn create_capturable_event(
        &mut self,
        owner_rank: usize,
    ) -> Result<CapturableEvent, P2pError> {
        let rank = self.rank(owner_rank)?;
        let event = CapturableEvent {
            owner_rank,
            event: Rc::new(RankEvent::new(Rc::clone(&rank.owner))?),
        };
        self.dependency_events.push(event.clone());
        Ok(event)
    }

    /// Begin one phase on exactly the listed ranks. The order is retained as the replay submission
    /// order. A remote transfer therefore passes `[source, destination]`: its source graph establishes
    /// the current replay's external-event generation before the destination wait graph is submitted.
    pub(crate) fn begin_capturable_phase(&mut self, ranks: &[usize]) -> Result<(), P2pError> {
        if self.active_capture_ranks.is_some() {
            return Err(P2pError::CapturePhaseAlreadyActive);
        }
        if ranks.is_empty() {
            return Err(P2pError::EmptyCapturePhase);
        }
        for (position, &rank_index) in ranks.iter().enumerate() {
            self.rank(rank_index)?;
            if ranks[..position].contains(&rank_index) {
                return Err(P2pError::DuplicateCapturePhaseRank { rank: rank_index });
            }
        }
        for (begun, &rank_index) in ranks.iter().enumerate() {
            let rank = &self.ranks[rank_index];
            let begin = unsafe {
                result::ctx::set_current(rank.ctx).and_then(|()| {
                    result::stream::begin_capture(
                        rank.stream,
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    )
                })
            };
            if let Err(error) = begin {
                self.abort_capturable_phase_prefix(&ranks[..begun]);
                return Err(error.into());
            }
        }
        self.active_capture_ranks = Some(ranks.to_vec());
        Ok(())
    }

    fn abort_capturable_phase_prefix(&self, ranks: &[usize]) {
        for &rank_index in ranks {
            let rank = &self.ranks[rank_index];
            unsafe {
                let _ = result::ctx::set_current(rank.ctx);
                if let Ok(graph) = result::stream::end_capture(rank.stream)
                    && !graph.is_null()
                {
                    let _ = result::graph::destroy(graph);
                }
            }
        }
    }

    pub(crate) fn abort_capturable_phase(&mut self) {
        if let Some(ranks) = self.active_capture_ranks.take() {
            self.abort_capturable_phase_prefix(&ranks);
        }
    }

    /// End and instantiate the active phase. No phase is published unless every participating rank's
    /// capture and instantiation succeeds.
    pub(crate) fn end_capturable_phase(&mut self) -> Result<usize, P2pError> {
        let ranks = self
            .active_capture_ranks
            .take()
            .ok_or(P2pError::CapturePhaseMissing)?;
        let mut graphs = Vec::with_capacity(ranks.len());
        for (position, &rank_index) in ranks.iter().enumerate() {
            let rank = &self.ranks[rank_index];
            let end = unsafe {
                result::ctx::set_current(rank.ctx)
                    .and_then(|()| result::stream::end_capture(rank.stream))
            };
            match end {
                Ok(graph) => graphs.push((
                    rank_index,
                    UninstantiatedRankGraph {
                        owner: Rc::clone(&rank.owner),
                        graph,
                    },
                )),
                Err(error) => {
                    self.abort_capturable_phase_prefix(&ranks[position + 1..]);
                    return Err(error.into());
                }
            }
        }

        let mut captured = Vec::with_capacity(graphs.len());
        let graph_buffers = self.owned_byte_buffers.clone();
        let graph_events: Vec<_> = self
            .ranks
            .iter()
            .flat_map(|rank| {
                std::iter::once(Rc::clone(&rank.event)).chain(rank.capture_events.iter().cloned())
            })
            .chain(
                self.dependency_events
                    .iter()
                    .map(|event| Rc::clone(&event.event)),
            )
            .collect();
        for (rank_index, mut graph) in graphs {
            let rank = &self.ranks[rank_index];
            let mut exec = std::ptr::null_mut();
            let instantiate = unsafe {
                result::ctx::set_current(rank.ctx).and_then(|()| {
                    sys::cuGraphInstantiateWithFlags(&mut exec, graph.graph, 0).result()
                })
            };
            match instantiate {
                Ok(()) => {
                    let graph_handle = std::mem::replace(&mut graph.graph, std::ptr::null_mut());
                    captured.push((
                        rank_index,
                        CapturedRankGraph {
                            rank: rank_index,
                            owner: Rc::clone(&rank.owner),
                            graph: graph_handle,
                            exec,
                            _buffers: graph_buffers.clone(),
                            _events: graph_events.clone(),
                        },
                    ));
                }
                Err(error) => {
                    if !exec.is_null() {
                        drop(CapturedRankGraph {
                            rank: rank_index,
                            owner: Rc::clone(&rank.owner),
                            graph: std::ptr::null_mut(),
                            exec,
                            _buffers: Vec::new(),
                            _events: Vec::new(),
                        });
                    }
                    return Err(error.into());
                }
            }
        }
        let graph_count = captured.len();
        self.captured_phases.push(CapturedGraphPhase {
            graphs: captured.into_iter().map(|(_, graph)| graph).collect(),
        });
        Ok(graph_count)
    }

    /// Explicitly upload each instantiated graph once. Later replays call only `cuGraphLaunch`.
    pub(crate) fn upload_captured_graphs(&self) -> Result<usize, P2pError> {
        let mut uploads = 0usize;
        for phase in &self.captured_phases {
            for graph in &phase.graphs {
                let rank = &self.ranks[graph.rank];
                unsafe {
                    result::ctx::set_current(rank.ctx)?;
                    result::graph::upload(graph.exec, rank.stream)?;
                }
                uploads += 1;
            }
        }
        self.barrier()?;
        Ok(uploads)
    }

    /// Launch the topologically ordered graph segments from the one owning host thread. Within a
    /// remote-transfer phase, source precedes destination; across phases, producer records precede every
    /// dependent wait. The executor barriers after the complete epoch before reusing its events.
    pub(crate) fn launch_captured_graphs(&self) -> Result<usize, P2pError> {
        let mut launches = 0usize;
        for phase in &self.captured_phases {
            for graph in &phase.graphs {
                let rank = &self.ranks[graph.rank];
                unsafe {
                    result::ctx::set_current(rank.ctx)?;
                    result::graph::launch(graph.exec, rank.stream)?;
                }
                launches += 1;
            }
        }
        Ok(launches)
    }

    pub(crate) fn captured_graph_count(&self) -> usize {
        self.captured_phases
            .iter()
            .map(|phase| phase.graphs.len())
            .sum()
    }

    /// Upload one host input per rank to a fresh device buffer on that rank's GPU (plus a staging chunk
    /// sized to the largest ring chunk). All inputs must have equal length.
    pub(crate) fn upload_rank_buffers(
        &self,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<RankDeviceBuffer>, P2pError> {
        let n = inputs.len();
        if n != self.world_size() {
            return Err(P2pError::BufferCountMismatch {
                actual: n,
                world_size: self.world_size(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let len = inputs[0].len();
        if !inputs.iter().all(|b| b.len() == len) {
            return Err(P2pError::Msg(
                "upload_rank_buffers: all rank inputs must have equal length".into(),
            ));
        }
        // Chunk 0 is the largest chunk (chunk_range gives the first `len % n` chunks one extra element).
        let max_chunk = chunk_range(len, n.max(1), 0).1;
        let buffer_bytes = f32_byte_len("rank buffer", len)?;
        let staging_bytes = f32_byte_len("rank staging", max_chunk)?;
        let mut bufs = Vec::with_capacity(n);
        for (r, input) in inputs.iter().enumerate() {
            let rank = self.rank(r)?;
            unsafe { result::ctx::set_current(rank.ctx) }?;
            let ptr = unsafe { result::malloc_sync(buffer_bytes.max(4)) }?;
            unsafe { result::memcpy_htod_sync(ptr, &input[..]) }?;
            let staging = unsafe { result::malloc_sync(staging_bytes.max(4)) }?;
            bufs.push(RankDeviceBuffer {
                transport: self.id,
                rank: r,
                ptr,
                staging,
                len,
                staging_len: max_chunk,
            });
        }
        Ok(bufs)
    }

    /// In-place ring all-reduce (`op = Sum`) over `bufs`, one per rank. After the call every rank's device
    /// buffer holds the elementwise sum of all ranks' inputs. Reduce-scatter (peer copy into staging +
    /// `Binary(Add)` accumulate, ordered by events) then all-gather (peer copy the reduced chunks directly),
    /// each phase R-1 steps with a per-step barrier. Mirrors [`crate::ring::ring_all_reduce_sum_with`].
    // `dst` indexes `bufs` AND feeds the ring modular arithmetic (`src = (dst-1) % n`), so a plain
    // iterator would obscure the schedule; keep the range loop.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn ring_all_reduce_sum(&self, bufs: &[RankDeviceBuffer]) -> Result<(), P2pError> {
        let n = bufs.len();
        if n != self.world_size() {
            return Err(P2pError::BufferCountMismatch {
                actual: n,
                world_size: self.world_size(),
            });
        }
        for (index, buffer) in bufs.iter().enumerate() {
            self.checked_owned(buffer)?;
            checked_rank(buffer.rank, self.world_size())?;
            if buffer.rank != index {
                return Err(P2pError::BufferRankMismatch {
                    index,
                    rank: buffer.rank,
                });
            }
        }
        if n <= 1 {
            // One replica (or none): AllReduce(x) = x, no transfers.
            return Ok(());
        }
        let len = bufs[0].len;
        if !bufs.iter().all(|b| b.len == len) {
            return Err(P2pError::Msg(
                "ring_all_reduce_sum: all rank buffers must have equal length".into(),
            ));
        }
        if len == 0 {
            return Ok(());
        }
        let peer_access = self.prepare_eager_ring_peer_access()?;

        // --- Reduce-scatter: R-1 steps. At step s, rank r sends chunk (r - s) to r+1 and accumulates the
        //     received chunk; after the phase rank r owns the full sum of one chunk. ---
        for step in 0..n - 1 {
            // Copy pass: each destination pulls its left neighbour's outgoing chunk into staging.
            for dst in 0..n {
                let src = (dst + n - 1) % n;
                let send_idx = (src + n - step) % n;
                let (a, b) = chunk_range(len, n, send_idx);
                if b > a {
                    self.move_buffer_chunk(&peer_access, &bufs[src], &bufs[dst], a, 0, b - a)?;
                }
            }
            // Accumulate pass: dst waits for src's copy (event) then adds staging into its own chunk.
            for dst in 0..n {
                let src = (dst + n - 1) % n;
                let recv_idx = (dst + n - step - 1) % n; // == send_idx above (src == dst-1)
                let (a, b) = chunk_range(len, n, recv_idx);
                if b > a {
                    self.wait_and_add(&bufs[dst], src, a, b - a)?;
                }
            }
            self.barrier()?;
        }

        // --- All-gather: R-1 steps circulating the reduced chunks (overwrite, no add) so every rank ends
        //     with the full sum. Copy straight into the dst buffer chunk (same byte offset in every rank);
        //     the per-step barrier orders the phases. ---
        for step in 0..n - 1 {
            for dst in 0..n {
                let src = (dst + n - 1) % n;
                let send_idx = (src + 1 + n - step) % n;
                let (a, b) = chunk_range(len, n, send_idx);
                if b > a {
                    self.copy_chunk_direct(&peer_access, &bufs[src], &bufs[dst], a, b - a)?;
                }
            }
            self.barrier()?;
        }
        Ok(())
    }

    /// The device-buffer analogue of [`crate::ring::ChunkTransport::move_chunk`]: copy `count` f32 from
    /// `src` at element `src_off` into `dst` at element `dst_off` (each on its own rank), then record the
    /// source rank's event so a waiter can order after the copy. Uses `cuMemcpyPeerAsync` when the pair
    /// peers, else host-staging.
    ///
    /// Both buffers must come from this transport's [`Self::upload_rank_buffers`]
    /// ([`P2pError::ForeignRankBuffer`] otherwise), and both ranges are checked against the buffers' real
    /// extents ([`P2pError::ByteRangeOutOfBounds`]), so no call can reach memory outside them. A raw
    /// device pointer is not accepted in place of either buffer:
    ///
    /// ```compile_fail,E0308
    /// fn copy_raw(
    ///     transport: &poot_ptx_gpu::p2p::PtxP2PTransport,
    ///     src: cudarc::driver::sys::CUdeviceptr,
    ///     dst: cudarc::driver::sys::CUdeviceptr,
    /// ) -> Result<(), poot_ptx_gpu::p2p::P2pError> {
    ///     transport.move_chunk(src, 0, dst, 0, 1)
    /// }
    /// ```
    pub fn move_chunk(
        &self,
        src: &RankDeviceBuffer,
        src_off: usize,
        dst: &RankDeviceBuffer,
        dst_off: usize,
        count: usize,
    ) -> Result<(), P2pError> {
        self.checked_owned(src)?;
        self.checked_owned(dst)?;
        let src_byte_off = f32_byte_len("move_chunk source offset", src_off)?;
        let dst_byte_off = f32_byte_len("move_chunk destination offset", dst_off)?;
        let byte_len = f32_byte_len("move_chunk length", count)?;
        let peer_access = self.prepare_eager_route_peer_access(src.rank, dst.rank)?;
        self.move_bytes_eager(
            &peer_access,
            src.rank,
            src.ptr,
            f32_byte_len("move_chunk source extent", src.len)?,
            src_byte_off,
            dst.rank,
            dst.ptr,
            f32_byte_len("move_chunk destination extent", dst.len)?,
            dst_byte_off,
            byte_len,
        )
    }

    /// Refuse a [`RankDeviceBuffer`] another transport allocated: its pointers are only guaranteed live
    /// while their allocating transport retains the rank's primary context.
    fn checked_owned(&self, buffer: &RankDeviceBuffer) -> Result<(), P2pError> {
        owned_by(self.id, buffer)
    }

    /// Ring-internal form of [`Self::move_chunk`] whose destination is `dst`'s staging chunk.
    fn move_buffer_chunk(
        &self,
        peer_access: &EagerPeerAccessPrepared,
        src: &RankDeviceBuffer,
        dst: &RankDeviceBuffer,
        src_off: usize,
        dst_off: usize,
        count: usize,
    ) -> Result<(), P2pError> {
        let src_byte_off = f32_byte_len("move_chunk source offset", src_off)?;
        let dst_byte_off = f32_byte_len("move_chunk destination offset", dst_off)?;
        let byte_len = f32_byte_len("move_chunk length", count)?;
        self.move_bytes_eager(
            peer_access,
            src.rank,
            src.ptr,
            f32_byte_len("move_chunk source extent", src.len)?,
            src_byte_off,
            dst.rank,
            dst.staging,
            f32_byte_len("move_chunk destination extent", dst.staging_len)?,
            dst_byte_off,
            byte_len,
        )
    }

    /// Copy an exact byte range between two rank-owned device buffers on a capture-safe P2P route. The
    /// source graph records an external event after the copy and the destination graph waits on that
    /// cross-device event. Packed and FP8 offsets and lengths need not be multiples of four. A pair that
    /// would require a per-transfer host allocation is rejected by name before copying; the caller must
    /// not present that route as captured warm replay. This method is valid only during stream capture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn move_bytes_capturable(
        &self,
        src_rank: usize,
        src_base: sys::CUdeviceptr,
        src_extent_bytes: usize,
        src_byte_off: usize,
        dst_rank: usize,
        dst_base: sys::CUdeviceptr,
        dst_extent_bytes: usize,
        dst_byte_off: usize,
        byte_len: usize,
    ) -> Result<(), P2pError> {
        let peer_access = self.checked_capturable_peer_access(src_rank, dst_rank)?;
        let event = self.next_capturable_transfer_event(src_rank)?;
        let src = self.rank(src_rank)?;
        let dst = self.rank(dst_rank)?;
        let mut calls = RawCudaPeerCopyCalls;
        execute_capturable_transfer(
            self.world_size(),
            peer_access,
            src_rank,
            src_base,
            src_extent_bytes,
            src_byte_off,
            dst_rank,
            dst_base,
            dst_extent_bytes,
            dst_byte_off,
            byte_len,
            src.ctx,
            src.stream,
            dst.ctx,
            dst.stream,
            &mut calls,
            |event_src, flag| self.record_capturable_event(event_src, event.event, flag),
            |wait_dst, event_src, flag| {
                self.wait_capturable_event(wait_dst, event_src, event.event, flag)
            },
        )
    }

    fn next_capturable_transfer_event(&self, src_rank: usize) -> Result<Rc<RankEvent>, P2pError> {
        let rank = self.rank(src_rank)?;
        let index = rank.next_capture_event.get();
        let event = if rank.capture_events.is_empty() {
            // Preserve the standalone public primitive's original one-transfer behavior. The captured
            // multi-device executor always prepares a distinct event for every transfer.
            Rc::clone(&rank.event)
        } else {
            rank.capture_events
                .get(index)
                .cloned()
                .ok_or(P2pError::CaptureEventPoolExhausted {
                    rank: src_rank,
                    index,
                })?
        };
        if !rank.capture_events.is_empty() {
            rank.next_capture_event.set(index + 1);
        }
        Ok(event)
    }

    /// Capture one same-rank exact-byte copy. All-to-all diagonal cells use this instead of pretending
    /// a device peers with itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_bytes_capturable_local(
        &self,
        rank_index: usize,
        src_base: sys::CUdeviceptr,
        src_extent_bytes: usize,
        src_byte_off: usize,
        dst_base: sys::CUdeviceptr,
        dst_extent_bytes: usize,
        dst_byte_off: usize,
        byte_len: usize,
    ) -> Result<(), P2pError> {
        let (src_ptr, dst_ptr) = checked_transfer(
            self.world_size(),
            rank_index,
            src_base,
            src_extent_bytes,
            src_byte_off,
            rank_index,
            dst_base,
            dst_extent_bytes,
            dst_byte_off,
            byte_len,
        )?;
        let rank = self.rank(rank_index)?;
        unsafe {
            result::ctx::set_current(rank.ctx)?;
            result::memcpy_dtod_async(dst_ptr, src_ptr, byte_len, rank.stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn move_bytes_eager(
        &self,
        peer_access: &EagerPeerAccessPrepared,
        src_rank: usize,
        src_base: sys::CUdeviceptr,
        src_extent_bytes: usize,
        src_byte_off: usize,
        dst_rank: usize,
        dst_base: sys::CUdeviceptr,
        dst_extent_bytes: usize,
        dst_byte_off: usize,
        byte_len: usize,
    ) -> Result<(), P2pError> {
        peer_access.require_route(src_rank, dst_rank)?;
        let (src_ptr, dst_ptr) = checked_transfer(
            self.world_size(),
            src_rank,
            src_base,
            src_extent_bytes,
            src_byte_off,
            dst_rank,
            dst_base,
            dst_extent_bytes,
            dst_byte_off,
            byte_len,
        )?;
        if self.checked_peerable(src_rank, dst_rank)? {
            self.copy_bytes_peer(peer_access, src_rank, dst_rank, src_ptr, dst_ptr, byte_len)?;
        } else {
            self.host_stage_bytes(peer_access, src_rank, dst_rank, src_ptr, dst_ptr, byte_len)?;
        }
        let src = self.rank(src_rank)?;
        unsafe { result::ctx::set_current(src.ctx) }?;
        unsafe { result::event::record(src.event.event, src.stream) }?;
        Ok(())
    }

    fn copy_bytes_peer(
        &self,
        peer_access: &EagerPeerAccessPrepared,
        src_rank: usize,
        dst_rank: usize,
        src_ptr: sys::CUdeviceptr,
        dst_ptr: sys::CUdeviceptr,
        byte_len: usize,
    ) -> Result<(), P2pError> {
        peer_access.require_direct_route(src_rank, dst_rank)?;
        let src = self.rank(src_rank)?;
        let dst = self.rank(dst_rank)?;
        let mut calls = RawCudaPeerCopyCalls;
        calls.memcpy_peer_async(
            peer_access,
            EagerPeerCopy {
                src_rank,
                dst_rank,
                current_context: src.ctx,
                stream: src.stream,
                destination_context: dst.ctx,
                destination: dst_ptr,
                source_context: src.ctx,
                source: src_ptr,
                byte_len,
            },
        )
    }

    /// Record the source-owned event into the source graph as an external event node. CUDA requires the
    /// event and stream passed to `cuEventRecordWithFlags` to belong to the current source context.
    fn record_capturable_event(
        &self,
        src_rank: usize,
        event: sys::CUevent,
        flag: sys::CUevent_record_flags,
    ) -> Result<(), P2pError> {
        let src = self.rank(src_rank)?;
        unsafe { result::ctx::set_current(src.ctx) }?;
        unsafe { sys::cuEventRecordWithFlags(event, src.stream, flag as u32).result() }?;
        Ok(())
    }

    /// Wait in the destination graph on the source-owned cross-device event. CUDA permits the event to
    /// come from another context or device; the destination stream's context must be current.
    fn wait_capturable_event(
        &self,
        dst_rank: usize,
        src_rank: usize,
        event: sys::CUevent,
        flag: sys::CUevent_wait_flags,
    ) -> Result<(), P2pError> {
        let dst = self.rank(dst_rank)?;
        self.rank(src_rank)?;
        unsafe { result::ctx::set_current(dst.ctx) }?;
        unsafe { result::stream::wait_event(dst.stream, event, flag) }?;
        Ok(())
    }

    pub(crate) fn record_plan_event(
        &self,
        rank_index: usize,
        event: CapturableEvent,
    ) -> Result<(), P2pError> {
        if event.owner_rank != rank_index {
            return Err(P2pError::Msg(format!(
                "rank {rank_index} cannot record capture event owned by rank {}",
                event.owner_rank
            )));
        }
        self.record_capturable_event(
            rank_index,
            event.event.event,
            ExternalCaptureEvent::RECORD_FLAG,
        )
    }

    pub(crate) fn wait_plan_event(
        &self,
        rank_index: usize,
        event: CapturableEvent,
    ) -> Result<(), P2pError> {
        self.wait_capturable_event(
            rank_index,
            event.owner_rank,
            event.event.event,
            ExternalCaptureEvent::WAIT_FLAG,
        )
    }

    /// Host-staging fallback for a non-peerable pair: device-to-host from the source then host-to-device
    /// into the destination (synchronous, through a byte-exact host bounce). Slower.
    fn host_stage_bytes(
        &self,
        peer_access: &EagerPeerAccessPrepared,
        src_rank: usize,
        dst_rank: usize,
        src_ptr: sys::CUdeviceptr,
        dst_ptr: sys::CUdeviceptr,
        byte_len: usize,
    ) -> Result<(), P2pError> {
        peer_access.require_route(src_rank, dst_rank)?;
        let src = self.rank(src_rank)?;
        let dst = self.rank(dst_rank)?;
        let mut host = vec![0u8; byte_len];
        unsafe {
            result::ctx::set_current(src.ctx)?;
            result::stream::synchronize(src.stream)?;
            result::memcpy_dtoh_sync(&mut host[..], src_ptr)?;
            result::ctx::set_current(dst.ctx)?;
            result::memcpy_htod_sync(dst_ptr, &host[..])?;
        }
        Ok(())
    }

    /// Order `dst_rank`'s stream after `src_rank`'s recorded copy, then launch the `Binary(Add)` kernel
    /// to accumulate `staging[0..count]` into `dst_base[off..off+count]` in place.
    fn wait_and_add(
        &self,
        dst: &RankDeviceBuffer,
        src_rank: usize,
        off: usize,
        count: usize,
    ) -> Result<(), P2pError> {
        let src = self.rank(src_rank)?;
        let dst_rank = self.rank(dst.rank)?;
        let byte_off = f32_byte_len("accumulate destination offset", off)?;
        let byte_len = f32_byte_len("accumulate byte length", count)?;
        let chunk = checked_device_span(
            "accumulate destination",
            dst.rank,
            self.world_size(),
            dst.ptr,
            f32_byte_len("accumulate destination extent", dst.len)?,
            byte_off,
            byte_len,
        )?;
        let staging = checked_device_span(
            "accumulate staging",
            dst.rank,
            self.world_size(),
            dst.staging,
            f32_byte_len("accumulate staging extent", dst.staging_len)?,
            0,
            byte_len,
        )?;
        unsafe {
            result::ctx::set_current(dst_rank.ctx)?;
            result::stream::wait_event(
                dst_rank.stream,
                src.event.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )?;
        }
        self.launch_add(dst.rank, chunk, staging, count)
    }

    /// Launch `Binary(Add)` on `rank`'s stream: `c[i] = a[i] + b[i]` with `a == c == chunk` (in-place) and
    /// `b == staging`, over `count` elements. Same interleaved `(ptr, i64 len)` ABI as
    /// `PtxContext::dispatch_dev`.
    fn launch_add(
        &self,
        rank: usize,
        chunk: sys::CUdeviceptr,
        staging: sys::CUdeviceptr,
        count: usize,
    ) -> Result<(), P2pError> {
        let rank_context = self.rank(rank)?;
        let a = chunk;
        let b = staging;
        let c = chunk;
        let la = count as i64;
        let lb = count as i64;
        let lc = count as i64;
        let mut params: Vec<*mut c_void> = vec![
            &a as *const _ as *mut c_void,
            &la as *const _ as *mut c_void,
            &b as *const _ as *mut c_void,
            &lb as *const _ as *mut c_void,
            &c as *const _ as *mut c_void,
            &lc as *const _ as *mut c_void,
        ];
        let block = (
            self.add_wg[0].max(1),
            self.add_wg[1].max(1),
            self.add_wg[2].max(1),
        );
        let grid = ((count as u32).div_ceil(block.0), 1, 1);
        unsafe {
            result::launch_kernel(
                rank_context.add_fn,
                grid,
                block,
                0,
                rank_context.stream,
                &mut params,
            )?;
        }
        Ok(())
    }

    /// All-gather copy: move `src_base[off..off+count]` into `dst_base[off..off+count]` (same chunk offset
    /// in both ranks, overwrite). Peer copy or host-stage; the caller's per-step barrier orders it.
    fn copy_chunk_direct(
        &self,
        peer_access: &EagerPeerAccessPrepared,
        src: &RankDeviceBuffer,
        dst: &RankDeviceBuffer,
        off: usize,
        count: usize,
    ) -> Result<(), P2pError> {
        peer_access.require_route(src.rank, dst.rank)?;
        let byte_off = f32_byte_len("all-gather byte offset", off)?;
        let byte_len = f32_byte_len("all-gather byte length", count)?;
        let (src_ptr, dst_ptr) = checked_transfer(
            self.world_size(),
            src.rank,
            src.ptr,
            f32_byte_len("all-gather source extent", src.len)?,
            byte_off,
            dst.rank,
            dst.ptr,
            f32_byte_len("all-gather destination extent", dst.len)?,
            byte_off,
            byte_len,
        )?;
        if self.checked_peerable(src.rank, dst.rank)? {
            self.copy_bytes_peer(peer_access, src.rank, dst.rank, src_ptr, dst_ptr, byte_len)
        } else {
            self.host_stage_bytes(peer_access, src.rank, dst.rank, src_ptr, dst_ptr, byte_len)
        }
    }

    /// Synchronize every rank's stream (the eager per-step ring barrier).
    pub fn barrier(&self) -> Result<(), P2pError> {
        for r in &self.ranks {
            unsafe {
                result::ctx::set_current(r.ctx)?;
                result::stream::synchronize(r.stream)?;
            }
        }
        Ok(())
    }

    /// Read a rank's device buffer back to host f32 (syncs the rank's stream first).
    pub fn download(&self, buf: &RankDeviceBuffer) -> Result<Vec<f32>, P2pError> {
        self.checked_owned(buf)?;
        let rank = self.rank(buf.rank)?;
        unsafe {
            result::ctx::set_current(rank.ctx)?;
            result::stream::synchronize(rank.stream)?;
        }
        let mut out = vec![0.0f32; buf.len];
        unsafe { result::memcpy_dtoh_sync(&mut out[..], buf.ptr) }?;
        Ok(out)
    }

    /// Free the device buffers + staging allocated by [`Self::upload_rank_buffers`].
    pub(crate) fn free_rank_buffers(&self, bufs: Vec<RankDeviceBuffer>) -> Result<(), P2pError> {
        for b in &bufs {
            self.checked_owned(b)?;
        }
        for b in bufs {
            let rank = self.rank(b.rank)?;
            unsafe {
                result::ctx::set_current(rank.ctx)?;
                result::free_sync(b.ptr)?;
                result::free_sync(b.staging)?;
            }
        }
        Ok(())
    }
}

fn owned_by(transport: u64, buffer: &RankDeviceBuffer) -> Result<(), P2pError> {
    if buffer.transport != transport {
        return Err(P2pError::ForeignRankBuffer);
    }
    Ok(())
}

fn checked_rank(rank: usize, world_size: usize) -> Result<(), P2pError> {
    if rank >= world_size {
        return Err(P2pError::RankOutOfRange { rank, world_size });
    }
    Ok(())
}

fn f32_byte_len(role: &'static str, elements: usize) -> Result<usize, P2pError> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(P2pError::ElementByteLenOverflow { role, elements })
}

fn checked_byte_end(role: &'static str, offset: usize, byte_len: usize) -> Result<usize, P2pError> {
    offset
        .checked_add(byte_len)
        .ok_or(P2pError::ByteRangeOverflow {
            role,
            offset,
            byte_len,
        })
}

fn require_capturable_peer(
    peer_access: CapturablePeerAccess,
    src_rank: usize,
    dst_rank: usize,
) -> Result<(), P2pError> {
    if peer_access != CapturablePeerAccess::Enabled {
        return Err(P2pError::NonCapturableHostStaging { src_rank, dst_rank });
    }
    Ok(())
}

fn discover_peer_capabilities_with<Operation>(
    world_size: usize,
    mut operation: Operation,
) -> Result<Vec<Vec<bool>>, P2pError>
where
    Operation: FnMut(PeerInitializationOperation, usize, usize) -> Result<bool, P2pError>,
{
    let mut peerable = vec![vec![false; world_size]; world_size];
    for (src_rank, row) in peerable.iter_mut().enumerate() {
        for (dst_rank, cell) in row.iter_mut().enumerate() {
            if src_rank != dst_rank {
                *cell = operation(
                    PeerInitializationOperation::QueryCapability,
                    src_rank,
                    dst_rank,
                )?;
            }
        }
    }
    Ok(peerable)
}

mod eager_peer_access {
    use super::*;

    enum Scope {
        Ring {
            world_size: usize,
            direct_routes: Vec<(usize, usize)>,
        },
        Route {
            src_rank: usize,
            dst_rank: usize,
            direct: bool,
        },
    }

    /// The fields and constructors stay private to this child module, so sibling/parent code can consume
    /// successful preparation but cannot manufacture its proof.
    pub(super) struct Prepared {
        scope: Scope,
    }

    impl Prepared {
        pub(super) fn require_route(
            &self,
            src_rank: usize,
            dst_rank: usize,
        ) -> Result<(), P2pError> {
            let prepared = match &self.scope {
                Scope::Ring { world_size, .. } => {
                    *world_size > 1
                        && dst_rank < *world_size
                        && src_rank == (dst_rank + *world_size - 1) % *world_size
                }
                Scope::Route {
                    src_rank: prepared_src,
                    dst_rank: prepared_dst,
                    ..
                } => src_rank == *prepared_src && dst_rank == *prepared_dst,
            };
            if prepared {
                Ok(())
            } else {
                Err(P2pError::Msg(format!(
                    "eager peer-access preparation does not cover route {src_rank} -> {dst_rank}"
                )))
            }
        }

        pub(super) fn require_direct_route(
            &self,
            src_rank: usize,
            dst_rank: usize,
        ) -> Result<(), P2pError> {
            self.require_route(src_rank, dst_rank)?;
            let direct = match &self.scope {
                Scope::Ring { direct_routes, .. } => direct_routes.contains(&(src_rank, dst_rank)),
                Scope::Route {
                    src_rank: prepared_src,
                    dst_rank: prepared_dst,
                    direct,
                } => src_rank == *prepared_src && dst_rank == *prepared_dst && *direct,
            };
            if direct {
                Ok(())
            } else {
                Err(P2pError::Msg(format!(
                    "eager route {src_rank} -> {dst_rank} was prepared for host staging, not peer copy"
                )))
            }
        }
    }

    pub(super) fn prepare_ring_with<Calls>(
        peerable: &[Vec<bool>],
        contexts: &[sys::CUcontext],
        calls: &mut Calls,
    ) -> Result<Prepared, P2pError>
    where
        Calls: CudaPeerCopyCalls,
    {
        let world_size = peerable.len();
        let mut direct_routes = Vec::with_capacity(world_size);
        if world_size > 1 {
            for dst_rank in 0..world_size {
                let src_rank = (dst_rank + world_size - 1) % world_size;
                if peerable[src_rank][dst_rank] {
                    calls.enable_peer_access(PeerAccessEnable {
                        src_rank,
                        dst_rank,
                        current_context: contexts[src_rank],
                        peer_context: contexts[dst_rank],
                    })?;
                    direct_routes.push((src_rank, dst_rank));
                }
            }
        }
        Ok(Prepared {
            scope: Scope::Ring {
                world_size,
                direct_routes,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_route_with<Calls>(
        peerable: bool,
        src_rank: usize,
        dst_rank: usize,
        src_context: sys::CUcontext,
        dst_context: sys::CUcontext,
        calls: &mut Calls,
    ) -> Result<Prepared, P2pError>
    where
        Calls: CudaPeerCopyCalls,
    {
        if peerable {
            // Legacy `move_chunk` treated peer enablement as best effort and let PeerAsync report whether
            // the route was actually usable. Preserve that behavior for this one standalone direction;
            // the raw seam still returns context-activation failures as typed errors.
            let _raw_result = calls.enable_peer_access_raw(PeerAccessEnable {
                src_rank,
                dst_rank,
                current_context: src_context,
                peer_context: dst_context,
            })?;
        }
        Ok(Prepared {
            scope: Scope::Route {
                src_rank,
                dst_rank,
                direct: peerable,
            },
        })
    }
}

fn prepare_eager_ring_peer_access_with<Calls>(
    peerable: &[Vec<bool>],
    contexts: &[sys::CUcontext],
    calls: &mut Calls,
) -> Result<EagerPeerAccessPrepared, P2pError>
where
    Calls: CudaPeerCopyCalls,
{
    eager_peer_access::prepare_ring_with(peerable, contexts, calls)
}

#[allow(clippy::too_many_arguments)]
fn prepare_eager_route_peer_access_with<Calls>(
    peerable: bool,
    src_rank: usize,
    dst_rank: usize,
    src_context: sys::CUcontext,
    dst_context: sys::CUcontext,
    calls: &mut Calls,
) -> Result<EagerPeerAccessPrepared, P2pError>
where
    Calls: CudaPeerCopyCalls,
{
    eager_peer_access::prepare_route_with(
        peerable,
        src_rank,
        dst_rank,
        src_context,
        dst_context,
        calls,
    )
}

fn enable_peer_access_with(
    hardware_peerable: bool,
    src_rank: usize,
    dst_rank: usize,
    enable: impl FnOnce() -> sys::CUresult,
) -> Result<(), P2pError> {
    if !hardware_peerable {
        return Err(P2pError::NonCapturableHostStaging { src_rank, dst_rank });
    }
    match enable() {
        sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
            Ok(())
        }
        error => Err(DriverError(error).into()),
    }
}

fn prepare_capturable_peer_access_with<Enable>(
    peerable: &[Vec<bool>],
    capturable_peer_access: &mut [Vec<CapturablePeerAccess>],
    declared: &[Vec<bool>],
    contexts: &[sys::CUcontext],
    mut enable: Enable,
) -> Result<(), P2pError>
where
    Enable: FnMut(usize, usize, sys::CUcontext, sys::CUcontext) -> Result<(), P2pError>,
{
    let world_size = peerable.len();

    // Validate the complete declaration before changing driver or published capture state.
    for (src_rank, declared_row) in declared.iter().enumerate() {
        for (dst_rank, &is_declared) in declared_row.iter().enumerate() {
            if is_declared && !peerable[src_rank][dst_rank] {
                return Err(P2pError::NonCapturableHostStaging { src_rank, dst_rank });
            }
        }
    }

    let mut prepared = vec![vec![CapturablePeerAccess::Unavailable; world_size]; world_size];
    for (src_rank, declared_row) in declared.iter().enumerate() {
        for (dst_rank, &is_declared) in declared_row.iter().enumerate() {
            if !is_declared {
                continue;
            }
            enable(src_rank, dst_rank, contexts[src_rank], contexts[dst_rank])?;
            prepared[src_rank][dst_rank] = CapturablePeerAccess::Enabled;
        }
    }
    for (published, prepared) in capturable_peer_access.iter_mut().zip(prepared) {
        *published = prepared;
    }
    Ok(())
}

/// The two CUDA capture flags that keep per-device graphs separate instead of joining their captures.
/// `cuEventRecordWithFlags` and cudarc's `stream::wait_event` consume these exact enum types.
struct ExternalCaptureEvent;

impl ExternalCaptureEvent {
    const RECORD_FLAG: sys::CUevent_record_flags =
        sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL;
    const WAIT_FLAG: sys::CUevent_wait_flags = sys::CUevent_wait_flags::CU_EVENT_WAIT_EXTERNAL;
}

/// Shared production seam for capture admission, driver-call selection, and operation ordering. Tests
/// inject the driver boundary itself, so a DtoD-to-PeerAsync regression is observable without a driver.
#[allow(clippy::too_many_arguments)]
fn execute_capturable_transfer<Calls, Record, Wait>(
    world_size: usize,
    peer_access: CapturablePeerAccess,
    src_rank: usize,
    src_base: sys::CUdeviceptr,
    src_extent_bytes: usize,
    src_byte_off: usize,
    dst_rank: usize,
    dst_base: sys::CUdeviceptr,
    dst_extent_bytes: usize,
    dst_byte_off: usize,
    byte_len: usize,
    src_context: sys::CUcontext,
    src_stream: sys::CUstream,
    _dst_context: sys::CUcontext,
    _dst_stream: sys::CUstream,
    calls: &mut Calls,
    mut record: Record,
    mut wait: Wait,
) -> Result<(), P2pError>
where
    Calls: CudaPeerCopyCalls,
    Record: FnMut(usize, sys::CUevent_record_flags) -> Result<(), P2pError>,
    Wait: FnMut(usize, usize, sys::CUevent_wait_flags) -> Result<(), P2pError>,
{
    let (src_ptr, dst_ptr) = checked_transfer(
        world_size,
        src_rank,
        src_base,
        src_extent_bytes,
        src_byte_off,
        dst_rank,
        dst_base,
        dst_extent_bytes,
        dst_byte_off,
        byte_len,
    )?;
    require_capturable_peer(peer_access, src_rank, dst_rank)?;
    calls.memcpy_dtod_async(CapturedDtoDCopy {
        current_context: src_context,
        stream: src_stream,
        destination: dst_ptr,
        source: src_ptr,
        byte_len,
    })?;
    record(src_rank, ExternalCaptureEvent::RECORD_FLAG)?;
    wait(dst_rank, src_rank, ExternalCaptureEvent::WAIT_FLAG)
}

#[allow(clippy::too_many_arguments)]
fn checked_transfer(
    world_size: usize,
    src_rank: usize,
    src_base: sys::CUdeviceptr,
    src_extent_bytes: usize,
    src_byte_off: usize,
    dst_rank: usize,
    dst_base: sys::CUdeviceptr,
    dst_extent_bytes: usize,
    dst_byte_off: usize,
    byte_len: usize,
) -> Result<(sys::CUdeviceptr, sys::CUdeviceptr), P2pError> {
    let src_ptr = checked_device_span(
        "source",
        src_rank,
        world_size,
        src_base,
        src_extent_bytes,
        src_byte_off,
        byte_len,
    )?;
    let dst_ptr = checked_device_span(
        "destination",
        dst_rank,
        world_size,
        dst_base,
        dst_extent_bytes,
        dst_byte_off,
        byte_len,
    )?;
    Ok((src_ptr, dst_ptr))
}

/// Validate the complete half-open byte range before deriving its CUDA pointer. Checking `end` against
/// both the logical extent and the pointer address prevents a valid start plus oversized length from
/// escaping admission.
fn checked_device_span(
    role: &'static str,
    rank: usize,
    world_size: usize,
    base: sys::CUdeviceptr,
    extent: usize,
    offset: usize,
    byte_len: usize,
) -> Result<sys::CUdeviceptr, P2pError> {
    checked_rank(rank, world_size)?;
    let end = checked_byte_end(role, offset, byte_len)?;
    if end > extent {
        return Err(P2pError::ByteRangeOutOfBounds {
            role,
            offset,
            end,
            extent,
        });
    }
    let offset_u64 =
        u64::try_from(offset).map_err(|_| P2pError::ByteOffsetConversion { role, offset })?;
    let end_u64 =
        u64::try_from(end).map_err(|_| P2pError::ByteOffsetConversion { role, offset: end })?;
    base.checked_add(end_u64)
        .ok_or(P2pError::DevicePointerOverflow {
            role,
            base,
            end: end_u64,
        })?;
    base.checked_add(offset_u64)
        .ok_or(P2pError::DevicePointerOverflow {
            role,
            base,
            end: offset_u64,
        })
}

/// Compile an elementwise `Binary(Add)` body to NVPTX and return the PTX text. This is the `p2p_add`
/// ring kernel's compiler and the one a compute segment uses to build its own add program, so it is
/// `pub(crate)` rather than private. Writes to a process-unique temp path (compile also emits a
/// sibling `.ll`), reads the `.ptx` back, and removes both. The counter keeps two in-process
/// compilations from sharing a path.
pub(crate) fn compile_add_ptx(body: &poot_kernel_ir::Body) -> Result<String, P2pError> {
    use poot_codegen::{Target, compile};
    static NEXT_COMPILE: AtomicU64 = AtomicU64::new(0);
    let unique = NEXT_COMPILE.fetch_add(1, Ordering::Relaxed);
    let mut ptx_path = std::env::temp_dir();
    ptx_path.push(format!(
        "poot_p2p_add_{}_{}.ptx",
        std::process::id(),
        unique
    ));
    compile(body, Target::Nvptx, &ptx_path)?;
    let ptx = std::fs::read_to_string(&ptx_path)?;
    let _ = std::fs::remove_file(&ptx_path);
    let _ = std::fs::remove_file(ptx_path.with_extension("ll"));
    Ok(ptx)
}

/// The number of visible NVIDIA GPUs (`cuDeviceGetCount`), or an error if the driver is absent.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-749 (formerly 585a)")
)]
pub(crate) fn device_count() -> Result<usize, P2pError> {
    result::init()?;
    Ok(result::device::get_count()? as usize)
}

/// Standalone multi-device ring `AllReduce(Sum)`. Uploads one host input per rank to its GPU, runs the
/// ring (reduce-scatter + all-gather over CUDA P2P), and returns each rank's result buffer read back to
/// host; every returned buffer equals the elementwise sum of all `inputs`. `world_size` is `inputs.len()`;
/// all inputs must have equal length.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-749 (formerly 585a)")
)]
pub(crate) fn p2p_ring_all_reduce_sum(inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, P2pError> {
    let n = inputs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let len = inputs[0].len();
    if !inputs.iter().all(|b| b.len() == len) {
        return Err(P2pError::Msg(
            "p2p_ring_all_reduce_sum: all rank inputs must have equal length".into(),
        ));
    }
    let transport = PtxP2PTransport::new(n)?;
    let bufs = transport.upload_rank_buffers(inputs)?;
    transport.ring_all_reduce_sum(&bufs)?;
    let out = (0..n)
        .map(|r| transport.download(&bufs[r]))
        .collect::<Result<Vec<_>, _>>()?;
    transport.free_rank_buffers(bufs)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum RecordedPeerCopy {
        CapturedDtoD(CapturedDtoDCopy),
        EagerPeerAsync(EagerPeerCopy),
    }

    #[derive(Default)]
    struct RecordingPeerCopyCalls<'a> {
        enables: Vec<PeerAccessEnable>,
        enable_results: Vec<sys::CUresult>,
        activation_error: Option<sys::CUresult>,
        calls: Vec<RecordedPeerCopy>,
        sequence: Option<&'a RefCell<Vec<&'static str>>>,
    }

    impl<'a> RecordingPeerCopyCalls<'a> {
        fn with_sequence(sequence: &'a RefCell<Vec<&'static str>>) -> Self {
            Self {
                enables: Vec::new(),
                enable_results: Vec::new(),
                activation_error: None,
                calls: Vec::new(),
                sequence: Some(sequence),
            }
        }

        fn with_enable_results(
            sequence: &'a RefCell<Vec<&'static str>>,
            enable_results: Vec<sys::CUresult>,
        ) -> Self {
            Self {
                enables: Vec::new(),
                enable_results,
                activation_error: None,
                calls: Vec::new(),
                sequence: Some(sequence),
            }
        }

        fn with_activation_error(
            sequence: &'a RefCell<Vec<&'static str>>,
            activation_error: sys::CUresult,
        ) -> Self {
            Self {
                enables: Vec::new(),
                enable_results: Vec::new(),
                activation_error: Some(activation_error),
                calls: Vec::new(),
                sequence: Some(sequence),
            }
        }
    }

    impl CudaPeerCopyCalls for RecordingPeerCopyCalls<'_> {
        fn enable_peer_access_raw(
            &mut self,
            args: PeerAccessEnable,
        ) -> Result<sys::CUresult, P2pError> {
            if let Some(error) = self.activation_error.take() {
                return Err(DriverError(error).into());
            }
            let result = self
                .enable_results
                .get(self.enables.len())
                .copied()
                .unwrap_or(sys::CUresult::CUDA_SUCCESS);
            self.enables.push(args);
            if let Some(sequence) = self.sequence {
                sequence.borrow_mut().push("enable");
            }
            Ok(result)
        }

        fn memcpy_dtod_async(&mut self, args: CapturedDtoDCopy) -> Result<(), P2pError> {
            self.calls.push(RecordedPeerCopy::CapturedDtoD(args));
            if let Some(sequence) = self.sequence {
                sequence.borrow_mut().push("copy");
            }
            Ok(())
        }

        fn memcpy_peer_async(
            &mut self,
            peer_access: &EagerPeerAccessPrepared,
            args: EagerPeerCopy,
        ) -> Result<(), P2pError> {
            peer_access.require_direct_route(args.src_rank, args.dst_rank)?;
            self.calls.push(RecordedPeerCopy::EagerPeerAsync(args));
            if let Some(sequence) = self.sequence {
                sequence.borrow_mut().push("copy");
            }
            Ok(())
        }
    }

    fn fake_context(value: usize) -> sys::CUcontext {
        value as sys::CUcontext
    }

    fn fake_stream(value: usize) -> sys::CUstream {
        value as sys::CUstream
    }

    fn standalone_copy_0_to_1() -> EagerPeerCopy {
        EagerPeerCopy {
            src_rank: 0,
            dst_rank: 1,
            current_context: fake_context(10),
            stream: fake_stream(40),
            destination_context: fake_context(20),
            destination: 200,
            source_context: fake_context(10),
            source: 100,
            byte_len: 4,
        }
    }

    fn assert_standalone_enable_result_continues_to_copy(enable_result: sys::CUresult) {
        let sequence = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_enable_results(&sequence, vec![enable_result]);
        let peer_access = prepare_eager_route_peer_access_with(
            true,
            0,
            1,
            fake_context(10),
            fake_context(20),
            &mut calls,
        )
        .unwrap();
        let copy = standalone_copy_0_to_1();

        calls.memcpy_peer_async(&peer_access, copy).unwrap();

        assert_eq!(
            calls.enables,
            vec![PeerAccessEnable {
                src_rank: 0,
                dst_rank: 1,
                current_context: fake_context(10),
                peer_context: fake_context(20),
            }]
        );
        assert_eq!(calls.calls, vec![RecordedPeerCopy::EagerPeerAsync(copy)]);
        assert_eq!(sequence.into_inner(), vec!["enable", "copy"]);
    }

    /// Deterministic xorshift64 -> f32 in [-1, 1), same generator style as `ring.rs`.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32; // 24 bits
            (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// A transport-free check of the ownership gate every public rank-buffer method runs first. The
    /// buffer is never dereferenced: the gate refuses it before any driver call.
    #[test]
    fn rank_buffers_of_another_transport_are_refused() {
        let foreign = RankDeviceBuffer {
            transport: u64::MAX,
            rank: 0,
            ptr: 0,
            staging: 0,
            len: 1,
            staging_len: 1,
        };
        let id = NEXT_TRANSPORT_ID.load(Ordering::Relaxed);
        assert_ne!(foreign.transport, id);
        assert!(matches!(
            owned_by(id, &foreign),
            Err(P2pError::ForeignRankBuffer)
        ));
        let own = RankDeviceBuffer {
            transport: id,
            ..foreign
        };
        assert!(owned_by(id, &own).is_ok());
    }

    #[test]
    fn eager_constructor_discovery_queries_capabilities_without_peer_enablement() {
        let operations = RefCell::new(Vec::new());
        let peerable = discover_peer_capabilities_with(3, |operation, src_rank, dst_rank| {
            operations
                .borrow_mut()
                .push((operation, src_rank, dst_rank));
            Ok(src_rank < dst_rank)
        })
        .unwrap();

        assert_eq!(
            peerable,
            vec![
                vec![false, true, true],
                vec![false, false, true],
                vec![false, false, false],
            ]
        );
        assert_eq!(
            operations.into_inner(),
            vec![
                (PeerInitializationOperation::QueryCapability, 0, 1),
                (PeerInitializationOperation::QueryCapability, 0, 2),
                (PeerInitializationOperation::QueryCapability, 1, 0),
                (PeerInitializationOperation::QueryCapability, 1, 2),
                (PeerInitializationOperation::QueryCapability, 2, 0),
                (PeerInitializationOperation::QueryCapability, 2, 1),
            ]
        );
    }

    #[test]
    fn eager_ring_successor_direction_is_exact_for_two_three_four_ranks() {
        let cases = [
            (2, vec![(1, 0), (0, 1)]),
            (3, vec![(2, 0), (0, 1), (1, 2)]),
            (4, vec![(3, 0), (0, 1), (1, 2), (2, 3)]),
        ];

        for (world_size, expected) in cases {
            let peerable = (0..world_size)
                .map(|src_rank| {
                    (0..world_size)
                        .map(|dst_rank| src_rank != dst_rank)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let contexts = (0..world_size)
                .map(|rank| fake_context(10 + rank))
                .collect::<Vec<_>>();
            let mut calls = RecordingPeerCopyCalls::default();

            let peer_access =
                prepare_eager_ring_peer_access_with(&peerable, &contexts, &mut calls).unwrap();
            let actual = calls
                .enables
                .iter()
                .map(|enable| (enable.src_rank, enable.dst_rank))
                .collect::<Vec<_>>();

            assert_eq!(actual, expected, "world_size={world_size}");
            for (src_rank, dst_rank) in expected {
                peer_access.require_route(src_rank, dst_rank).unwrap();
            }
            assert!(
                peer_access.require_route(0, 0).is_err(),
                "world_size={world_size}: non-successor route must not be covered"
            );
        }
    }

    #[test]
    fn eager_ring_enables_only_successor_edges_before_first_transfer() {
        let peerable = vec![
            vec![false, true, true],
            vec![true, false, true],
            vec![true, true, false],
        ];
        let contexts = vec![fake_context(10), fake_context(20), fake_context(30)];
        let sequence = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_enable_results(
            &sequence,
            vec![
                sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
                sys::CUresult::CUDA_SUCCESS,
                sys::CUresult::CUDA_SUCCESS,
            ],
        );

        let peer_access =
            prepare_eager_ring_peer_access_with(&peerable, &contexts, &mut calls).unwrap();
        calls
            .memcpy_peer_async(
                &peer_access,
                EagerPeerCopy {
                    src_rank: 2,
                    dst_rank: 0,
                    current_context: fake_context(30),
                    stream: fake_stream(40),
                    destination_context: fake_context(10),
                    destination: 200,
                    source_context: fake_context(30),
                    source: 100,
                    byte_len: 4,
                },
            )
            .unwrap();

        assert_eq!(
            calls.enables,
            vec![
                PeerAccessEnable {
                    src_rank: 2,
                    dst_rank: 0,
                    current_context: fake_context(30),
                    peer_context: fake_context(10),
                },
                PeerAccessEnable {
                    src_rank: 0,
                    dst_rank: 1,
                    current_context: fake_context(10),
                    peer_context: fake_context(20),
                },
                PeerAccessEnable {
                    src_rank: 1,
                    dst_rank: 2,
                    current_context: fake_context(20),
                    peer_context: fake_context(30),
                },
            ]
        );
        assert_eq!(
            sequence.into_inner(),
            vec!["enable", "enable", "enable", "copy"]
        );
    }

    #[test]
    fn eager_standalone_success_enable_continues_to_peer_copy() {
        assert_standalone_enable_result_continues_to_copy(sys::CUresult::CUDA_SUCCESS);
    }

    #[test]
    fn eager_standalone_already_enabled_continues_to_peer_copy() {
        assert_standalone_enable_result_continues_to_copy(
            sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
        );
    }

    #[test]
    fn eager_standalone_too_many_peers_enable_continues_to_peer_copy() {
        assert_standalone_enable_result_continues_to_copy(sys::CUresult::CUDA_ERROR_TOO_MANY_PEERS);
    }

    #[test]
    fn eager_standalone_context_activation_failure_stops_before_copy() {
        let sequence = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_activation_error(
            &sequence,
            sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
        );

        let result = prepare_eager_route_peer_access_with(
            true,
            0,
            1,
            fake_context(10),
            fake_context(20),
            &mut calls,
        )
        .and_then(|peer_access| calls.memcpy_peer_async(&peer_access, standalone_copy_0_to_1()));

        assert!(matches!(
            result,
            Err(P2pError::Driver(DriverError(
                sys::CUresult::CUDA_ERROR_INVALID_CONTEXT
            )))
        ));
        assert!(calls.enables.is_empty());
        assert!(calls.calls.is_empty());
        assert!(sequence.into_inner().is_empty());
    }

    #[test]
    fn eager_standalone_host_staging_skips_enable() {
        let mut calls = RecordingPeerCopyCalls::default();
        let staged = prepare_eager_route_peer_access_with(
            false,
            2,
            0,
            fake_context(30),
            fake_context(10),
            &mut calls,
        )
        .unwrap();

        staged.require_route(2, 0).unwrap();
        assert!(staged.require_direct_route(2, 0).is_err());
        assert!(calls.enables.is_empty());
        assert!(calls.calls.is_empty());
    }

    #[test]
    fn eager_standalone_reverse_token_is_rejected() {
        let sequence = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_sequence(&sequence);
        let peer_access = prepare_eager_route_peer_access_with(
            true,
            0,
            1,
            fake_context(10),
            fake_context(20),
            &mut calls,
        )
        .unwrap();

        let result = calls.memcpy_peer_async(
            &peer_access,
            EagerPeerCopy {
                src_rank: 1,
                dst_rank: 0,
                current_context: fake_context(20),
                stream: fake_stream(41),
                destination_context: fake_context(10),
                destination: 100,
                source_context: fake_context(20),
                source: 200,
                byte_len: 4,
            },
        );

        assert!(matches!(result, Err(P2pError::Msg(_))));
        assert_eq!(calls.enables.len(), 1);
        assert!(calls.calls.is_empty());
        assert_eq!(sequence.into_inner(), vec!["enable"]);
    }

    #[test]
    fn eager_ring_skips_peer_enable_for_host_staged_edges() {
        let peerable = vec![
            vec![false, true, false],
            vec![false, false, false],
            vec![false, false, false],
        ];
        let contexts = vec![fake_context(10), fake_context(20), fake_context(30)];
        let mut calls = RecordingPeerCopyCalls::default();

        let _peer_access =
            prepare_eager_ring_peer_access_with(&peerable, &contexts, &mut calls).unwrap();

        assert_eq!(
            calls.enables,
            vec![PeerAccessEnable {
                src_rank: 0,
                dst_rank: 1,
                current_context: fake_context(10),
                peer_context: fake_context(20),
            }]
        );
    }

    #[test]
    fn eager_ring_peer_enable_failure_stops_before_any_transfer() {
        let peerable = vec![
            vec![false, true, true],
            vec![true, false, true],
            vec![true, true, false],
        ];
        let contexts = vec![fake_context(10), fake_context(20), fake_context(30)];
        let sequence = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_enable_results(
            &sequence,
            vec![
                sys::CUresult::CUDA_SUCCESS,
                sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            ],
        );

        let result = prepare_eager_ring_peer_access_with(&peerable, &contexts, &mut calls)
            .and_then(|peer_access| {
                calls.memcpy_peer_async(
                    &peer_access,
                    EagerPeerCopy {
                        src_rank: 2,
                        dst_rank: 0,
                        current_context: fake_context(30),
                        stream: fake_stream(40),
                        destination_context: fake_context(10),
                        destination: 200,
                        source_context: fake_context(30),
                        source: 100,
                        byte_len: 4,
                    },
                )
            });

        assert!(matches!(
            result,
            Err(P2pError::Driver(DriverError(
                sys::CUresult::CUDA_ERROR_INVALID_CONTEXT
            )))
        ));
        assert_eq!(
            calls.enables,
            vec![
                PeerAccessEnable {
                    src_rank: 2,
                    dst_rank: 0,
                    current_context: fake_context(30),
                    peer_context: fake_context(10),
                },
                PeerAccessEnable {
                    src_rank: 0,
                    dst_rank: 1,
                    current_context: fake_context(10),
                    peer_context: fake_context(20),
                },
            ]
        );
        assert!(calls.calls.is_empty());
        assert_eq!(sequence.into_inner(), vec!["enable", "enable"]);
    }

    #[test]
    fn checked_device_span_accepts_exact_end_and_rejects_bad_rank() {
        assert_eq!(
            checked_device_span("test", 1, 2, 100, 8, 3, 5).unwrap(),
            103
        );
        assert!(matches!(
            checked_device_span("test", 2, 2, 100, 8, 0, 1),
            Err(P2pError::RankOutOfRange {
                rank: 2,
                world_size: 2
            })
        ));
    }

    #[test]
    fn checked_device_span_rejects_overflow_and_full_range_oob() {
        assert!(matches!(
            checked_device_span("test", 0, 1, 100, usize::MAX, usize::MAX, 1),
            Err(P2pError::ByteRangeOverflow {
                offset: usize::MAX,
                byte_len: 1,
                ..
            })
        ));
        assert!(matches!(
            checked_device_span("test", 0, 1, 100, 8, 7, 2),
            Err(P2pError::ByteRangeOutOfBounds {
                offset: 7,
                end: 9,
                extent: 8,
                ..
            })
        ));
    }

    #[test]
    fn checked_device_span_rejects_pointer_end_overflow() {
        assert!(matches!(
            checked_device_span("test", 0, 1, u64::MAX - 3, 8, 2, 4),
            Err(P2pError::DevicePointerOverflow { end: 6, .. })
        ));
    }

    #[test]
    fn checked_transfer_rejects_an_invalid_destination_extent() {
        assert!(matches!(
            checked_transfer(2, 0, 100, 8, 0, 1, 200, 8, 7, 2),
            Err(P2pError::ByteRangeOutOfBounds {
                role: "destination",
                offset: 7,
                end: 9,
                extent: 8,
            })
        ));
    }

    #[test]
    fn captured_route_rejects_host_staging() {
        assert!(require_capturable_peer(CapturablePeerAccess::Enabled, 0, 1).is_ok());
        assert!(matches!(
            require_capturable_peer(CapturablePeerAccess::Unavailable, 0, 1),
            Err(P2pError::NonCapturableHostStaging {
                src_rank: 0,
                dst_rank: 1
            })
        ));
    }

    #[test]
    fn capturable_transfer_integration_rejects_host_staging_before_cuda_actions() {
        let actions = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::default();
        let result = execute_capturable_transfer(
            2,
            CapturablePeerAccess::Unavailable,
            0,
            100,
            8,
            1,
            1,
            200,
            8,
            2,
            4,
            fake_context(10),
            fake_stream(20),
            fake_context(30),
            fake_stream(40),
            &mut calls,
            |_, _| {
                actions.borrow_mut().push("record");
                Ok(())
            },
            |_, _, _| {
                actions.borrow_mut().push("wait");
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(P2pError::NonCapturableHostStaging {
                src_rank: 0,
                dst_rank: 1,
            })
        ));
        assert!(calls.calls.is_empty());
        assert!(actions.into_inner().is_empty());
    }

    #[test]
    fn capturable_prepare_enables_only_declared_directed_routes() {
        let peerable = vec![
            vec![false, true, true],
            vec![true, false, true],
            vec![true, true, false],
        ];
        let declared = vec![
            vec![false, false, true],
            vec![false, false, false],
            vec![false, true, false],
        ];
        let contexts = vec![fake_context(10), fake_context(20), fake_context(30)];
        let mut capturable =
            vec![vec![CapturablePeerAccess::Unavailable; contexts.len()]; contexts.len()];
        let enabled = RefCell::new(Vec::new());

        prepare_capturable_peer_access_with(
            &peerable,
            &mut capturable,
            &declared,
            &contexts,
            |src_rank, dst_rank, src_context, dst_context| {
                enabled.borrow_mut().push((
                    src_rank,
                    dst_rank,
                    src_context as usize,
                    dst_context as usize,
                ));
                enable_peer_access_with(true, src_rank, dst_rank, || sys::CUresult::CUDA_SUCCESS)
            },
        )
        .unwrap();

        assert_eq!(
            capturable,
            vec![
                vec![
                    CapturablePeerAccess::Unavailable,
                    CapturablePeerAccess::Unavailable,
                    CapturablePeerAccess::Enabled,
                ],
                vec![CapturablePeerAccess::Unavailable; 3],
                vec![
                    CapturablePeerAccess::Unavailable,
                    CapturablePeerAccess::Enabled,
                    CapturablePeerAccess::Unavailable,
                ],
            ]
        );
        assert_eq!(enabled.into_inner(), vec![(0, 2, 10, 30), (2, 1, 30, 20)]);
    }

    #[test]
    fn capturable_transfer_uses_source_context_stream_and_dtod() {
        let actions = RefCell::new(Vec::new());
        let mut calls = RecordingPeerCopyCalls::with_sequence(&actions);
        execute_capturable_transfer(
            2,
            CapturablePeerAccess::Enabled,
            0,
            100,
            8,
            1,
            1,
            200,
            8,
            2,
            4,
            fake_context(10),
            fake_stream(20),
            fake_context(30),
            fake_stream(40),
            &mut calls,
            |event_src, flag| {
                assert_eq!(event_src, 0);
                assert_eq!(flag, sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL);
                actions.borrow_mut().push("record");
                Ok(())
            },
            |wait_dst, event_src, flag| {
                assert_eq!((wait_dst, event_src), (1, 0));
                assert_eq!(flag, sys::CUevent_wait_flags::CU_EVENT_WAIT_EXTERNAL);
                actions.borrow_mut().push("wait");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            calls.calls,
            vec![RecordedPeerCopy::CapturedDtoD(CapturedDtoDCopy {
                current_context: fake_context(10),
                stream: fake_stream(20),
                destination: 202,
                source: 101,
                byte_len: 4,
            })]
        );
        assert_eq!(actions.into_inner(), vec!["copy", "record", "wait"]);
    }

    #[test]
    fn capturable_prepare_propagates_enable_failure_without_publishing_routes() {
        let peerable = vec![
            vec![false, true, true],
            vec![true, false, true],
            vec![true, true, false],
        ];
        let declared = vec![
            vec![false, false, true],
            vec![false, false, false],
            vec![false, true, false],
        ];
        let contexts = vec![fake_context(10), fake_context(20), fake_context(30)];
        let unavailable = vec![vec![CapturablePeerAccess::Unavailable; 3]; 3];
        let mut capturable = unavailable.clone();
        let enabled = RefCell::new(Vec::new());
        let mut attempt = 0usize;

        let result = prepare_capturable_peer_access_with(
            &peerable,
            &mut capturable,
            &declared,
            &contexts,
            |src_rank, dst_rank, src_context, dst_context| {
                enabled.borrow_mut().push((
                    src_rank,
                    dst_rank,
                    src_context as usize,
                    dst_context as usize,
                ));
                attempt += 1;
                let result = if attempt == 2 {
                    sys::CUresult::CUDA_ERROR_INVALID_CONTEXT
                } else {
                    sys::CUresult::CUDA_SUCCESS
                };
                enable_peer_access_with(true, src_rank, dst_rank, || result)
            },
        );

        assert!(matches!(
            result,
            Err(P2pError::Driver(DriverError(
                sys::CUresult::CUDA_ERROR_INVALID_CONTEXT
            )))
        ));
        assert_eq!(capturable, unavailable);
        assert_eq!(enabled.into_inner(), vec![(0, 2, 10, 30), (2, 1, 30, 20)]);
    }

    #[test]
    fn captured_transfer_flags_are_external_not_default() {
        assert_eq!(
            ExternalCaptureEvent::RECORD_FLAG,
            sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL
        );
        assert_ne!(
            ExternalCaptureEvent::RECORD_FLAG,
            sys::CUevent_record_flags::CU_EVENT_RECORD_DEFAULT
        );
        assert_eq!(
            ExternalCaptureEvent::WAIT_FLAG,
            sys::CUevent_wait_flags::CU_EVENT_WAIT_EXTERNAL
        );
        assert_ne!(
            ExternalCaptureEvent::WAIT_FLAG,
            sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT
        );
    }

    /// Fresh-process regression for the legacy public standalone path. Run this exact ignored test by
    /// itself so no ring or capture preparation can enable the route first:
    ///   CUDA_VISIBLE_DEVICES=0,1 cargo test -p poot-ptx-gpu p2p::tests::probe_standalone_move_chunk_2gpu -- --exact --ignored --nocapture
    #[test]
    #[ignore = "requires exactly two visible peer-capable NVIDIA GPUs"]
    fn probe_standalone_move_chunk_2gpu() {
        let count = device_count()
            .unwrap_or_else(|error| panic!("Card 408 standalone stage=device_count: {error}"));
        assert_eq!(
            count, 2,
            "Card 408 standalone stage=device_count: expected exactly two NVIDIA GPUs"
        );

        let transport = PtxP2PTransport::new(2)
            .unwrap_or_else(|error| panic!("Card 408 standalone stage=transport_init: {error}"));
        assert!(
            transport.peerable(0, 1),
            "Card 408 standalone stage=peer_capability_0_to_1: rank 0 cannot access rank 1"
        );
        assert!(
            transport.peerable(1, 0),
            "Card 408 standalone stage=peer_capability_1_to_0: rank 1 cannot access rank 0"
        );

        let expected_0_to_1 = (0..4097)
            .map(|index| index as f32 * 0.25 - 256.0)
            .collect::<Vec<_>>();
        let expected_1_to_0 = (0..4103)
            .map(|index| ((index * 37 % 1009) as f32 - 500.0) * 0.125 + 0.03125)
            .collect::<Vec<_>>();
        let buffers_0_to_1 = transport
            .upload_rank_buffers(&[
                expected_0_to_1.clone(),
                vec![-777.25; expected_0_to_1.len()],
            ])
            .unwrap_or_else(|error| panic!("Card 408 standalone stage=upload_0_to_1: {error}"));
        let buffers_1_to_0 = match transport
            .upload_rank_buffers(&[vec![888.5; expected_1_to_0.len()], expected_1_to_0.clone()])
        {
            Ok(buffers) => buffers,
            Err(error) => {
                let cleanup = transport.free_rank_buffers(buffers_0_to_1);
                panic!(
                    "Card 408 standalone stage=upload_1_to_0: {error}; \
                     prior_pair_cleanup={cleanup:?}"
                );
            }
        };

        let execution = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            transport
                .move_chunk(
                    &buffers_0_to_1[0],
                    0,
                    &buffers_0_to_1[1],
                    0,
                    expected_0_to_1.len(),
                )
                .map_err(|error| format!("move_chunk_0_to_1: {error}"))?;
            transport
                .barrier()
                .map_err(|error| format!("synchronize_0_to_1: {error}"))?;
            let actual_0_to_1 = transport
                .download(&buffers_0_to_1[1])
                .map_err(|error| format!("download_0_to_1: {error}"))?;

            transport
                .move_chunk(
                    &buffers_1_to_0[1],
                    0,
                    &buffers_1_to_0[0],
                    0,
                    expected_1_to_0.len(),
                )
                .map_err(|error| format!("move_chunk_1_to_0: {error}"))?;
            transport
                .barrier()
                .map_err(|error| format!("synchronize_1_to_0: {error}"))?;
            let actual_1_to_0 = transport
                .download(&buffers_1_to_0[0])
                .map_err(|error| format!("download_1_to_0: {error}"))?;
            Ok((actual_0_to_1, actual_1_to_0))
        })();

        let free_0_to_1 = transport.free_rank_buffers(buffers_0_to_1);
        let free_1_to_0 = transport.free_rank_buffers(buffers_1_to_0);
        free_0_to_1
            .unwrap_or_else(|error| panic!("Card 408 standalone stage=free_0_to_1: {error}"));
        free_1_to_0
            .unwrap_or_else(|error| panic!("Card 408 standalone stage=free_1_to_0: {error}"));
        let (actual_0_to_1, actual_1_to_0) = execution.unwrap_or_else(|error| {
            panic!("Card 408 standalone stage=bidirectional_copy: {error}")
        });

        assert_eq!(
            actual_0_to_1, expected_0_to_1,
            "Card 408 standalone stage=result_0_to_1: destination values differ"
        );
        assert_eq!(
            actual_1_to_0, expected_1_to_0,
            "Card 408 standalone stage=result_1_to_0: destination values differ"
        );
        let to_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            to_bytes(&actual_0_to_1),
            to_bytes(&expected_0_to_1),
            "Card 408 standalone stage=result_0_to_1: destination bytes differ"
        );
        assert_eq!(
            to_bytes(&actual_1_to_0),
            to_bytes(&expected_1_to_0),
            "Card 408 standalone stage=result_1_to_0: destination bytes differ"
        );
    }

    /// Card 408's smallest hardware regression for the A100 capture failure. Peer admission, both
    /// allocations, their uploads, and the source-owned transfer event are deliberately complete before
    /// either stream begins capture. Run on exactly two peer-capable NVIDIA GPUs:
    ///   cargo test -p poot-ptx-gpu p2p::tests::probe_capturable_peer_copy_graph_2gpu -- --exact --ignored --nocapture
    #[test]
    #[ignore = "requires exactly two visible peer-capable NVIDIA GPUs"]
    fn probe_capturable_peer_copy_graph_2gpu() {
        let count = device_count()
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=device_count: {error}"));
        assert_eq!(
            count, 2,
            "Card 408 direct capture stage=device_count: expected exactly two NVIDIA GPUs"
        );

        let mut transport = PtxP2PTransport::new(2).unwrap_or_else(|error| {
            panic!("Card 408 direct capture stage=transport_init: {error}")
        });
        transport
            .prepare_capturable_peer_access(&[vec![false, true], vec![false, false]])
            .unwrap_or_else(|error| {
                panic!("Card 408 direct capture stage=peer_admission: {error}")
            });

        let expected: Vec<u8> = (0..4097).map(|index| (index % 251) as u8).collect();
        let source = transport
            .upload_rank_bytes(0, &expected)
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=source_upload: {error}"));
        let destination = transport
            .upload_rank_bytes(1, &vec![0; expected.len()])
            .unwrap_or_else(|error| {
                panic!("Card 408 direct capture stage=destination_upload: {error}")
            });
        transport
            .prepare_capturable_transfer_events(&[1, 0])
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=event_prepare: {error}"));

        transport
            .begin_capturable_phase(&[0, 1])
            .unwrap_or_else(|error| {
                panic!("Card 408 direct capture stage=begin_capturable_phase: {error}")
            });
        if let Err(error) = transport.move_bytes_capturable(
            0,
            source.ptr(),
            source.byte_len(),
            0,
            1,
            destination.ptr(),
            destination.byte_len(),
            0,
            expected.len(),
        ) {
            transport.abort_capturable_phase();
            panic!("Card 408 direct capture stage=move_bytes_capturable: {error}");
        }
        let graph_count = transport.end_capturable_phase().unwrap_or_else(|error| {
            panic!("Card 408 direct capture stage=end_capturable_phase: {error}")
        });
        assert_eq!(
            graph_count, 2,
            "Card 408 direct capture stage=end_capturable_phase: wrong graph count"
        );

        let uploads = transport
            .upload_captured_graphs()
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=graph_upload: {error}"));
        assert_eq!(
            uploads, 2,
            "Card 408 direct capture stage=graph_upload: wrong upload count"
        );
        let launches = transport
            .launch_captured_graphs()
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=graph_launch: {error}"));
        assert_eq!(
            launches, 2,
            "Card 408 direct capture stage=graph_launch: wrong launch count"
        );
        transport
            .barrier()
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=barrier: {error}"));
        let actual = transport
            .download_rank_bytes(&destination)
            .unwrap_or_else(|error| panic!("Card 408 direct capture stage=download: {error}"));
        assert_eq!(
            actual, expected,
            "Card 408 direct capture stage=result: destination bytes differ"
        );
    }

    /// Distinct random f32 on each of `world_size` GPUs, run the P2P ring `AllReduce(Sum)`, and assert
    /// every device's buffer equals the CPU elementwise sum. Pod-gated: skips cleanly when fewer than 2
    /// CUDA devices are visible. Run on a >= 2-GPU NVIDIA pod:
    ///   cargo test -p poot-ptx-gpu probe_p2p_ring_all_reduce_2gpu -- --ignored --nocapture
    #[test]
    #[ignore = "needs >= 2 NVIDIA GPUs (RunPod multi-GPU pod); skips on the dev box"]
    fn probe_p2p_ring_all_reduce_2gpu() {
        let count = match device_count() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP probe_p2p_ring_all_reduce_2gpu: no CUDA driver/device ({e})");
                return;
            }
        };
        if count < 2 {
            eprintln!("SKIP probe_p2p_ring_all_reduce_2gpu: need >= 2 GPUs, found {count}");
            return;
        }
        let world_size = count.min(4); // use up to 4 GPUs if the pod has them
        let len = 4096usize;

        let mut rng = Rng::new(0x0490A_u64);
        let inputs: Vec<Vec<f32>> = (0..world_size)
            .map(|_| (0..len).map(|_| rng.next_f32()).collect())
            .collect();

        // CPU reference elementwise sum.
        let mut expected = vec![0.0f32; len];
        for buf in &inputs {
            for (e, &x) in expected.iter_mut().zip(buf) {
                *e += x;
            }
        }

        let out =
            p2p_ring_all_reduce_sum(&inputs).expect("p2p ring all-reduce should run on the pod");
        assert_eq!(out.len(), world_size);
        for (r, got) in out.iter().enumerate() {
            assert_eq!(got.len(), len, "rank {r} wrong length");
            let mut max_abs = 0.0f32;
            for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                let d = (g - e).abs();
                if d > max_abs {
                    max_abs = d;
                }
                assert!(
                    d <= 1e-4 * (1.0 + e.abs()),
                    "rank {r} idx {i}: got {g} expected {e} (P2P ring != CPU sum)"
                );
            }
            eprintln!("rank {r}: matches CPU sum (max_abs={max_abs:.3e})");
        }
        eprintln!(
            "probe_p2p_ring_all_reduce_2gpu OK: world_size={world_size} len={len} - every GPU holds the elementwise sum"
        );
    }
}
