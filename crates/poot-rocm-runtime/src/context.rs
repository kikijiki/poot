use crate::*;

/// Card 125 (spec 120 FR-004/SC-004): whether a host<->device transfer into/out of
/// [`RocmContext::fine_pool`] (where every buffer + kernarg lives) can use a raw CPU pointer write or
/// needs a device DMA transfer.
///
/// - `HostWrite`: the pool is host-visible, so `ptr::copy_nonoverlapping` is valid. True on an
///   APU/unified-memory box (Strix Halo), where the fine-grained pool is a host-mapped alias of the
///   memory the GPU dispatches against.
/// - `Dma`: the pool is not host-visible (a discrete GPU's device-local memory); a raw host pointer
///   write is undefined behavior, so transfers use `hsa_amd_memory_async_copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadStrategy {
    HostWrite,
    Dma,
}

/// Decide the [`UploadStrategy`] for a pool from its `HSA_AMD_MEMORY_POOL_INFO_LOCATION` and
/// `HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS`. Pure function of the two u32s, so it is unit-checkable
/// without an HSA context (see `tests::upload_strategy_*`).
///
/// A pool is host-visible (safe for a raw CPU pointer write) iff it is CPU-located or carries
/// `EXTENDED_SCOPE_FINE_GRAINED`, AMD's signal that a GPU-local pool's fine-grained coherency extends
/// to host scope (the aliased view of Strix Halo's unified-memory carveout; see `RocmContext::new`'s
/// doc on the coarse/fine-alias pair). Plain `FINE_GRAINED` on a GPU-located pool only guarantees
/// device coherency and does not imply the pointer is valid in the host address space: CDNA's
/// coherent multi-GPU fabric can mark VRAM `FINE_GRAINED` without it being host-mappable.
pub fn upload_strategy_for_pool(location: u32, flags: u32) -> UploadStrategy {
    let host_visible = location == HSA_AMD_MEMORY_POOL_LOCATION_CPU
        || flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED != 0;
    if host_visible {
        UploadStrategy::HostWrite
    } else {
        UploadStrategy::Dma
    }
}

/// A HSA runtime context: the loaded library, the picked GPU agent, one user-mode queue, and the
/// fine-grained pool every buffer and kernarg lives in. Single-threaded; not `Send`.
///
/// The agent, pool, upload strategy and queue are private: every raw write, DMA and dispatch relies on
/// them staying the values construction chose, and `Drop` destroys the queue. Callers read the agent
/// and strategy through accessors and cannot replace any of them:
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.queue = std::ptr::null_mut();
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.fine_pool = poot_rocm_runtime::bindings::hsa_amd_memory_pool_t { handle: 0 };
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.upload_strategy = poot_rocm_runtime::UploadStrategy::HostWrite;
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.coarse_pool = poot_rocm_runtime::bindings::hsa_amd_memory_pool_t { handle: 0 };
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.gpu_agent = poot_rocm_runtime::bindings::hsa_agent_t { handle: 0 };
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.isa_name = "forged".to_string();
/// }
/// ```
///
/// ```compile_fail,E0616
/// fn forge(ctx: &mut poot_rocm_runtime::RocmContext) {
///     ctx.wavefront = 0;
/// }
/// ```
pub struct RocmContext {
    /// Keep the initialized runtime alive for the context and every resource created from it.
    pub(crate) hsa: Arc<HsaRuntimeOwner>,
    /// The first GPU agent returned by `hsa_iterate_agents`; the queue, code objects and DMA copies
    /// all target it.
    pub(crate) gpu_agent: hsa_agent_t,
    /// The chosen GPU agent's ISA name (e.g. `"gfx1151"` on Strix Halo after the env override). Not
    /// `pub` (card 608): read through [`RocmContext::isa_name`], so external code cannot rewrite it after
    /// codegen reads it. `pub(crate)` so `transfer.rs`'s construction can still build the literal.
    pub(crate) isa_name: String,
    /// The chosen GPU agent's wavefront size (32 or 64), from HSA. Threaded into codegen for
    /// wave-dependent lowering (lane masks, workgroup caps) instead of a hardcoded constant. Not `pub`
    /// for the same reason as `isa_name`.
    pub(crate) wavefront: u32,
    /// Fine-grained system pool (kernarg; on APU/HMM the pointer is host-visible, so `upload_f32`
    /// allocates here for zero-copy).
    pub(crate) fine_pool: hsa_amd_memory_pool_t,
    /// Coarse-grained GPU device pool (Card 547a): `Weight`/`Activation`/`State` allocate here.
    /// Discovered at construction alongside `fine_pool` but previously discarded after its one-time
    /// `SIZE` query (`vram_total_bytes`); now stored and used directly. Not assumed host-visible: every
    /// write/read bridges through a pinned host staging buffer and a device-to-device
    /// `hsa_amd_memory_async_copy` ([`RocmContext::upload_bytes_role_aware`]/
    /// [`RocmContext::download_bytes_role_aware`]), never a raw host pointer write, regardless of what
    /// this pool's own location/flags report (the gfx1151 host-mappability risk the card names).
    pub(crate) coarse_pool: hsa_amd_memory_pool_t,
    /// Card 125 (FR-004/SC-004): the [`UploadStrategy`] for `fine_pool`, decided once at construction
    /// from its location + flags. Gates every host<->device transfer (`allocate_f32`'s zero-init,
    /// `upload_f32`/`upload_i32`/`upload_bytes_batch`, `download_f32`) between a raw pointer write and
    /// `hsa_amd_memory_async_copy` DMA.
    pub(crate) upload_strategy: UploadStrategy,
    /// The single queue acquired for this context's immutable role. Destroyed (or leaked when
    /// poisoned) by `Drop`.
    pub(crate) queue: *mut hsa_queue_t,
    pub(crate) queue_role: QueueRole,
    pub(crate) queue_ring_provenance: QueueRingProvenance,
    /// Immutable body/header publication ordering established from the queue-ring allocation
    /// provenance before the queue becomes available to replay.
    pub(crate) queue_publication: publication::QueuePublication,
    /// Present only when constructed via
    /// [`RocmContext::new_for_pcie_device_ring_publication_receipt`].
    pub(crate) device_ring_receipt: Option<DeviceRingReceiptProvenance>,
    /// `HSA_AMD_MEMORY_POOL_INFO_SIZE` of the GPU's coarse-grained pool, cached at construction
    /// (capacity is fixed for the process lifetime). A single pool's size, not a sum over every
    /// global pool the agent owns (see [`RocmContext::new`] on the double-counting pitfall). The
    /// "total" half of the card 030 VRAM metric; see [`Self::vram_budget_bytes`] /
    /// [`Self::vram_used_bytes`]. 0 if the SIZE query failed (both methods then report unavailable).
    pub(crate) vram_total_bytes: u64,
    /// `HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT` of the GPU agent, cached at construction (card 653);
    /// `None` if the query failed ([`Self::device_caps`] then reports the ROCm default).
    pub(crate) compute_units: Option<u32>,
    /// Work a timed-out wait left in flight, and the poison flag that follows it.
    pub(crate) timeouts: TimeoutLeaks,
    /// The bounded-wait ceiling every `wait_completion_bounded`/`synchronize`/queue-space wait reads
    /// (Card 548): typed construction-time configuration, never an environment read in
    /// library code (`poot/AGENTS.md`). Set once, from [`RocmContextOptions`], at construction.
    pub(crate) wait_timeout: std::time::Duration,
}

/// Construction-time options for [`RocmContext::new_with_options`] (Card 548): the
/// `POOT_GPU_WAIT_TIMEOUT_SECS` environment read this replaces lived at `error.rs`'s deleted
/// `gpu_wait_timeout_secs()`. [`RocmContext::new`] uses [`Self::default`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RocmContextOptions {
    /// Wall-clock bound for every bounded HSA wait (`wait_completion_bounded`, the `synchronize`
    /// read-index spin, the publisher's queue-space wait): a hang detector, not a latency bound (see
    /// the former `GPU_WAIT_TIMEOUT_SECS` doc). [`Self::default`] is 600s, the prior constant default.
    pub wait_timeout: std::time::Duration,
}

impl Default for RocmContextOptions {
    fn default() -> Self {
        Self {
            wait_timeout: std::time::Duration::from_secs(600),
        }
    }
}

/// `POOT_REQUIRE_ROCM=1` turns the "no ROCm device -> tests skip cleanly" convention into a loud
/// failure: a failed context open panics instead of returning the Err that skip guards convert into a
/// silent PASS (`just test-device-rocm` sets it). Mirrors `poot-runtime::require_gpu_check`.
pub(crate) fn require_gpu_check<E: std::fmt::Display>(e: E) -> E {
    require_gpu_check_with(|name| std::env::var(name).ok(), e)
}

pub(crate) fn require_gpu_check_with<E: std::fmt::Display>(
    get: impl Fn(&str) -> Option<String>,
    e: E,
) -> E {
    poot_runtime_common::DeviceBackend::Rocm.fail_if_required(get, e)
}

/// RAII guard pairing a successful `hsa_init()` with `hsa_shut_down()` on every early return out of
/// `RocmContext::new_inner` between the two calls.
///
/// `Drop for HsaRuntimeOwner` normally runs `hsa_shut_down`, but it exists only once a full
/// `RocmContext` is built. An early `?` in `new_inner` (no GPU agent, a missing memory pool, ...)
/// would skip to dropping `Hsa`, whose `LibHandle` unconditionally `dlclose`s the library.
/// `hsa_init` can start HSA-internal worker threads, and dlclose-ing under a running one segfaults
/// (reproduced from `RocmContext::new()`'s `NoGpuAgent` path with the device hidden). The guard is
/// disarmed right before the final `Ok`, where the shared runtime owner takes over.
pub(crate) struct HsaShutdownGuard {
    pub(crate) shut_down: ffi::HsaShutDown,
    pub(crate) armed: bool,
}

impl Drop for HsaShutdownGuard {
    fn drop(&mut self) {
        if self.armed {
            // SAFETY: armed only between a successful `hsa_init` and the hand-over to the runtime owner,
            // while the library stays loaded (`Hsa` drops after this guard).
            unsafe {
                let _ = (self.shut_down)();
            }
        }
    }
}

/// A pinned host range: `hsa_amd_memory_lock` at construction, `hsa_amd_memory_unlock` on drop. Only
/// runtime-owned memory is pinned (a [`StagingBuffer`]), never a caller's slice, so an abandoned range
/// and the memory under it can both move to the [`TimeoutLeaks`] list, which never drops them.
pub(crate) struct PinnedHostRange {
    hsa: Arc<HsaRuntimeOwner>,
    host_ptr: *mut c_void,
    agent_ptr: *mut c_void,
}

impl PinnedHostRange {
    /// The agent-accessible address of the pinned range (the `hsa_amd_memory_lock` output).
    pub(crate) fn agent_ptr(&self) -> *mut c_void {
        self.agent_ptr
    }
}

impl Drop for PinnedHostRange {
    fn drop(&mut self) {
        // SAFETY: `host_ptr` was locked by `dma_staging` and is unlocked once, here, before its
        // `StagingBuffer` frees the memory; an abandoned range is never dropped.
        let status = unsafe { (self.hsa.funcs.hsa_amd_memory_unlock)(self.host_ptr) };
        if status != bindings::HSA_STATUS_SUCCESS {
            tracing::warn!(
                status = format_args!("0x{status:x}"),
                "hsa_amd_memory_unlock failed (leaking the lock)"
            );
        }
    }
}

/// Runtime-owned host bytes pinned for one DMA transfer. The engine copies into or out of it; the
/// caller's slice is touched only by the host, before submission or after proven completion. If the
/// copy's wait times out, the staging buffer moves to the [`TimeoutLeaks`] list, so the engine never
/// writes memory the caller has freed or reused.
pub(crate) struct StagingBuffer {
    /// Declared before `bytes`, so the range is unlocked before its memory is freed.
    pin: PinnedHostRange,
    bytes: Box<[u8]>,
}

impl StagingBuffer {
    /// The agent-accessible address of the staging bytes.
    pub(crate) fn agent_ptr(&self) -> *mut c_void {
        self.pin.agent_ptr()
    }

    /// The host address of the staging bytes, which the transfer reads or writes on the host.
    pub(crate) fn host_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub(crate) fn host_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl RocmContext {
    /// A zeroed, pinned, runtime-owned staging buffer of `size` bytes for one DMA transfer (Card 125,
    /// FR-004/SC-004): `hsa_amd_memory_lock` makes it a valid `hsa_amd_memory_async_copy` source or
    /// destination for `self.gpu_agent`, and the pin is released when the buffer drops. Card 602: only
    /// runtime-owned memory is pinned, never a caller's slice.
    pub(crate) fn dma_staging(&self, size: usize) -> Result<StagingBuffer, RocmError> {
        let mut bytes = vec![0u8; size].into_boxed_slice();
        let host_ptr: *mut c_void = bytes.as_mut_ptr().cast();
        let mut agent = self.gpu_agent;
        let mut locked: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host_ptr` addresses `size` bytes owned by `bytes`, which the returned buffer keeps
        // alive and unmoved (a boxed slice) until after the pin is released; `agent` and `locked` are
        // valid for the call.
        unsafe {
            check((self.hsa.funcs.hsa_amd_memory_lock)(
                host_ptr,
                size,
                &mut agent,
                1,
                &mut locked,
            ))?;
        }
        let pin = PinnedHostRange {
            hsa: Arc::clone(&self.hsa),
            host_ptr,
            agent_ptr: locked,
        };
        Ok(StagingBuffer { pin, bytes })
    }
}

/// Whether the device may still reference resources of one runtime. Shared by a [`RocmContext`] and
/// every allocation and module created from it, and set when a bounded wait times out. A kernarg holds
/// raw device addresses, so the runtime cannot tell which allocations or code objects a hung packet
/// uses; once set, no [`Drop`] of that runtime frees an allocation or destroys an executable.
#[derive(Clone, Default)]
pub(crate) struct DevicePoison(Arc<std::sync::atomic::AtomicBool>);

impl DevicePoison {
    pub(crate) fn poison(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// What a copy still references besides its completion signal: the runtime-owned staging buffer.
/// Allocations and code objects are not listed: they stay alive through [`DevicePoison`].
#[derive(Default)]
#[allow(
    dead_code,
    reason = "owned so a timed-out submission's resources are never released"
)]
pub(crate) struct Held {
    pub(crate) staging: Option<StagingBuffer>,
}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Held")
            .field(
                "staging_bytes",
                &self.staging.as_ref().map(StagingBuffer::len),
            )
            .finish()
    }
}

/// A completion signal and the resources of a submission whose bounded wait gave up. The device may
/// still decrement the signal and use the staging buffer, so neither is released.
#[allow(
    dead_code,
    reason = "owned so a timed-out submission's resources are never released"
)]
pub(crate) struct Abandoned {
    pub(crate) signal: bindings::hsa_signal_t,
    pub(crate) held: Held,
}

/// Ownership record of every timed-out wait. A timeout cannot prove the packet or copy finished, so
/// the signal and staging buffer move here instead of being freed, and the runtime is poisoned: no
/// later submission touches the queue, and no allocation, code object or queue of the runtime is
/// released. The record is never dropped. Retry and recovery are out of scope.
///
/// Per resource (Card 602): completion signal, staging buffer and pinned range are listed here; the
/// queue is listed here when the context drops; allocations and code objects are leaked in place by
/// their own `Drop`, which reads [`DevicePoison`]. No path aborts.
#[derive(Default)]
pub(crate) struct TimeoutLeaks {
    poison: DevicePoison,
    abandoned: Mutex<Vec<Abandoned>>,
    queues: Vec<*mut hsa_queue_t>,
}

impl TimeoutLeaks {
    pub(crate) fn poison(&self) {
        self.poison.poison();
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poison.is_poisoned()
    }

    /// The poison flag that allocations and modules created from this context share.
    pub(crate) fn device_poison(&self) -> DevicePoison {
        self.poison.clone()
    }

    /// Take ownership of an abandoned submission and poison the context.
    pub(crate) fn abandon(&self, abandoned: Abandoned) {
        self.abandoned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(abandoned);
        self.poison();
    }

    /// Release the context's queue: destroy it when healthy, or keep it when poisoned, since a packet
    /// may still be in flight on it.
    pub(crate) fn retire_queue<R>(
        &mut self,
        queue: &mut *mut hsa_queue_t,
        destroy: impl FnOnce(*mut hsa_queue_t) -> R,
    ) {
        let Some(queue) = take_queue_reference(queue) else {
            return;
        };
        if self.is_poisoned() {
            self.queues.push(queue);
        } else {
            destroy(queue);
        }
    }

    #[cfg(test)]
    pub(crate) fn abandoned(&self) -> std::sync::MutexGuard<'_, Vec<Abandoned>> {
        self.abandoned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    pub(crate) fn retired_queues(&self) -> &[*mut hsa_queue_t] {
        &self.queues
    }
}

impl Drop for TimeoutLeaks {
    fn drop(&mut self) {
        // The device may still use every listed resource, so none is released. This leaks for the
        // rest of the process, bounded by what a hung device held.
        let abandoned = std::mem::take(
            self.abandoned
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        std::mem::forget(abandoned);
        std::mem::forget(std::mem::take(&mut self.queues));
    }
}

impl RocmContext {
    /// The native HSA pool handle `pool` names (Card 547a scope).
    pub(crate) fn native_pool(&self, pool: Pool) -> hsa_amd_memory_pool_t {
        match pool {
            Pool::Coarse => self.coarse_pool,
            Pool::Fine => self.fine_pool,
        }
    }

    /// Record one allocation of `bytes` under `role` and return the guard that releases it (Card
    /// 547a): the one place every allocation primitive in this crate charges the shared
    /// [`poot_runtime_common::MemoryCounters`].
    pub(crate) fn tag_alloc(&self, role: BufferRole, bytes: usize) -> AllocGuard {
        self.hsa.memory.record_alloc(role, bytes as u64)
    }

    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// this context.
    pub fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.hsa.memory.snapshot_all()
    }

    /// The chosen GPU agent's ISA name (e.g. `"gfx1151"` on Strix Halo after the env override). Read-only
    /// (card 608): see the struct doc.
    pub fn isa_name(&self) -> &str {
        &self.isa_name
    }

    /// The chosen GPU agent's wavefront size (32 or 64), from HSA. Read-only (card 608): see the struct
    /// doc.
    pub fn wavefront(&self) -> u32 {
        self.wavefront
    }

    /// The device-capability descriptor the planner reads (card 522). The wavefront size, VRAM total,
    /// compute-unit count and tensor-core family are this agent's own queries. HSA exposes no group-segment
    /// (LDS) size or per-dimension grid-count query this crate's discovery layer has found, so those
    /// two fields, and the workgroup limits ([`poot_target::AMDGPU_MAX_WORKGROUP_INVOCATIONS`], the
    /// AMDGPU backend's documented ceiling), are the vendor-scoped default; the display watchdog and the
    /// card-095 tiled-GEMM miscompile are wgpu/RADV codegen-path defects this backend does not share
    /// (HSA queues dispatches on hardware with no host display-refresh TDR), so both are absent here.
    pub fn device_caps(&self) -> poot_target::DeviceCaps {
        caps_from_agent(&AgentProbe {
            vram_total_bytes: self.vram_total_bytes,
            isa_name: &self.isa_name,
            wavefront: self.wavefront,
            compute_units: self.compute_units,
        })
    }

    /// The context queue's fixed `hsa_queue_t` fields (ring base, doorbell, size, id).
    pub(crate) fn queue_layout(&self) -> &QueueLayout {
        // SAFETY: `queue` is non-null and live from construction until `Drop` (it is private and set
        // only there), `QueueLayout` mirrors `hsa_queue_t`, and the runtime never writes those fields
        // after creation, so a shared reference for `&self`'s lifetime is valid.
        unsafe { &*(self.queue as *const QueueLayout) }
    }

    /// Whether an earlier bounded wait timed out. A poisoned context refuses every later submission.
    pub fn is_poisoned(&self) -> bool {
        self.timeouts.is_poisoned()
    }

    /// Refuse `op` with [`RocmError::ContextPoisoned`] before it touches the queue.
    pub(crate) fn ensure_not_poisoned(&self, op: &'static str) -> Result<(), RocmError> {
        if self.is_poisoned() {
            return Err(RocmError::ContextPoisoned { op });
        }
        Ok(())
    }
}

/// What a live GPU agent reports, before it becomes the planner's [`poot_target::DeviceCaps`]: the seam
/// [`RocmContext::device_caps`] reads, so a test can feed values no real device reports.
pub(crate) struct AgentProbe<'a> {
    /// `HSA_AMD_MEMORY_POOL_INFO_SIZE` of the coarse pool; 0 when the query failed.
    pub(crate) vram_total_bytes: u64,
    pub(crate) isa_name: &'a str,
    /// `HSA_AGENT_INFO_WAVEFRONT_SIZE`; 0 is not a wavefront size.
    pub(crate) wavefront: u32,
    pub(crate) compute_units: Option<u32>,
}

pub(crate) fn caps_from_agent(probe: &AgentProbe<'_>) -> poot_target::DeviceCaps {
    let defaults = poot_target::DeviceCaps::rocm_default();
    poot_target::DeviceCaps {
        max_buffer_bytes: if probe.vram_total_bytes == 0 {
            defaults.max_buffer_bytes
        } else {
            probe.vram_total_bytes
        },
        subgroup: if probe.wavefront == 0 {
            poot_target::Queried::Unknown
        } else {
            poot_target::Queried::Known(poot_target::SubgroupSupport::Present {
                min_size: probe.wavefront,
                max_size: probe.wavefront,
            })
        },
        tensor_core: poot_target::TensorCoreSupport::from_device_name(probe.isa_name),
        compute_units: probe.compute_units.unwrap_or(defaults.compute_units),
        ..defaults
    }
}

impl Drop for RocmContext {
    fn drop(&mut self) {
        // Queue use is confined to the context. A poisoned queue may still run a packet, so it moves to
        // the leak list instead of being destroyed.
        let destroy = self.hsa.funcs.hsa_queue_destroy;
        self.timeouts
            // SAFETY: the context's own live queue, destroyed once and only when no packet can be in flight.
            .retire_queue(&mut self.queue, |queue| unsafe { destroy(queue) });
        // Allocations and modules carry their own shared runtime owner and may outlive this value; the
        // final owner shuts HSA down after they drop. A poisoned runtime is never shut down or
        // unloaded: the device may still use its resources.
        if self.timeouts.is_poisoned() {
            std::mem::forget(Arc::clone(&self.hsa));
        }
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use poot_target::{DeviceCaps, Queried, SubgroupSupport, TensorCoreSupport};

    /// SC-003: sentinel agent values (a wave64 CDNA part with 7 compute units and 12345 bytes of VRAM)
    /// come out in exactly the fields that name them. Mutation: report the RDNA default wavefront (32)
    /// or the default compute-unit count (40), and the equality fails on that field.
    #[test]
    fn agent_values_come_out_in_their_own_caps_fields() {
        let caps = caps_from_agent(&AgentProbe {
            vram_total_bytes: 12_345,
            isa_name: "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-",
            wavefront: 64,
            compute_units: Some(7),
        });
        assert_eq!(caps.max_buffer_bytes, 12_345);
        assert_eq!(
            caps.subgroup,
            Queried::Known(SubgroupSupport::Present {
                min_size: 64,
                max_size: 64
            })
        );
        assert_eq!(caps.tensor_core, TensorCoreSupport::CdnaMfma);
        assert_eq!(caps.compute_units, 7);
        assert_eq!(
            caps.max_workgroup_size,
            [poot_target::AMDGPU_MAX_WORKGROUP_INVOCATIONS; 3]
        );
    }

    /// Receipt (SC-003): the documented AMDGPU workgroup ceiling the caps report equals what the live HSA
    /// agent itself reports (`HSA_AGENT_INFO_WORKGROUP_MAX_SIZE` and `_MAX_DIM`), and the reported
    /// subgroup is the agent's own wavefront. A skip (no ROCm device) is reported as one.
    #[test]
    fn a_live_agent_reports_the_workgroup_ceiling_the_caps_document() {
        use std::ffi::c_void;
        let Ok(ctx) = RocmContext::new() else {
            eprintln!(
                "SKIP a_live_agent_reports_the_workgroup_ceiling_the_caps_document: no ROCm device"
            );
            return;
        };
        // hsa.h: HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7 (uint16_t[3]), _MAX_SIZE = 8 (uint32_t).
        let (mut dims, mut size) = ([0u16; 3], 0u32);
        // SAFETY: both attributes write exactly the documented type into the pointer given.
        unsafe {
            let get = ctx.hsa.hsa.funcs.hsa_agent_get_info;
            assert_eq!(get(ctx.gpu_agent, 7, dims.as_mut_ptr().cast::<c_void>()), 0);
            assert_eq!(get(ctx.gpu_agent, 8, (&raw mut size).cast::<c_void>()), 0);
        }
        let caps = ctx.device_caps();
        eprintln!("live rocm caps: {caps:?}; agent workgroup dims {dims:?} size {size}");
        assert_eq!(caps.max_workgroup_invocations, size);
        assert_eq!(caps.max_workgroup_size, dims.map(u32::from));
        assert_eq!(
            caps.subgroup,
            Queried::Known(SubgroupSupport::Present {
                min_size: ctx.wavefront(),
                max_size: ctx.wavefront(),
            })
        );
    }

    /// A failed query stays explicit: no wavefront is `Unknown`, never the RDNA default, and the
    /// documented defaults fill only what HSA cannot report.
    #[test]
    fn a_failed_query_stays_unknown_not_a_guessed_default() {
        let caps = caps_from_agent(&AgentProbe {
            vram_total_bytes: 0,
            isa_name: "gfx1151",
            wavefront: 0,
            compute_units: None,
        });
        assert_eq!(caps.subgroup, Queried::Unknown);
        assert_eq!(
            caps.max_buffer_bytes,
            DeviceCaps::rocm_default().max_buffer_bytes
        );
        assert_eq!(caps.compute_units, DeviceCaps::rocm_default().compute_units);
    }
}
