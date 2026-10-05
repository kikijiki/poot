use crate::*;

/// The SPIR-V words of a compiler-produced `kernel`, or a typed error if it was compiled for another
/// backend (card 608): the one check every wgpu entry point that reads a `CompiledKernel`'s code runs
/// before touching it.
pub(crate) fn spirv_words(kernel: &CompiledKernel) -> Result<&[u32], RuntimeError> {
    match kernel.code() {
        poot_runtime_common::KernelCode::SpirvWords(words) => Ok(words.as_ref()),
        _ => Err(RuntimeError::WrongKernelTarget {
            actual: kernel.target(),
        }),
    }
}

/// Process-global monotonic buffer id (card 156 phase 3): a cheap identity tag for a
/// [`DeviceBuffer`], distinct on every allocation, independent of wgpu's private resource ids. The
/// cached-decode ping-pong KV recovery (poot-gpu's `kv_parity`) uses it to tell whether a
/// caller-supplied state buffer is one of its two held physical buffers (B2). Clones share it, since
/// they share the same `wgpu::Buffer`.
pub(crate) static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_buffer_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

/// A cached compute pipeline + its bind-group layout (G4). wgpu pipeline/layout objects are
/// `Arc`-backed, so cloning is a refcount bump; the clone is handed out per dispatch.
pub(crate) struct CachedPipeline {
    /// The SPIR-V module, data-binding count and error-word binding this pipeline was built from:
    /// everything the module and layout depend on, so a cache hit is the same kernel, not only the
    /// same key (card 656).
    pub(crate) words: Box<[u32]>,
    pub(crate) bindings: usize,
    pub(crate) has_error_word: bool,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bgl: wgpu::BindGroupLayout,
}

impl CachedPipeline {
    /// Whether this pipeline was built from exactly `words` with `bindings` data buffers and (when
    /// `has_error_word`) the reserved error-word binding.
    pub(crate) fn built_from(&self, words: &[u32], bindings: usize, has_error_word: bool) -> bool {
        *self.words == *words && self.bindings == bindings && self.has_error_word == has_error_word
    }
}

/// An asserting dispatch's error-word storage buffer, staged but not yet read back (card 531c): `dispatch_dev`/`submit_cached` have no per-dispatch sync, so instead
/// of reading this immediately they push one of these onto `Context::pending_faults`. `label` names the
/// kernel for the eventual `KernelAssertFailed`; `storage` is the live device buffer the kernel
/// atomically OR's its trap code into. Cleared by `Context::stage_pending_faults`, which copies every
/// pending entry into a small staging buffer as part of the caller's own next submission (no new sync
/// added: the copy rides whichever submit the executor's step boundary already makes).
pub(crate) struct PendingFault {
    pub(crate) label: String,
    pub(crate) storage: wgpu::Buffer,
    /// Live-bytes guard for `storage` (Card 547a), held only when `storage` was freshly
    /// allocated right where this fault was recorded (`dispatch_dev`'s one-shot error word) rather than
    /// a cached buffer already charged elsewhere (`submit_cached`'s reused `CachedDispatch.error_storage`,
    /// whose guard lives on the `CachedDispatch` itself for its whole lifetime) - charging it twice here
    /// would double-count the same physical allocation. Dropped together with `storage` when this
    /// `PendingFault` drops, inside `Context::stage_pending_faults`'s per-entry closure.
    #[allow(dead_code, reason = "held only for its Drop side effect, Card 547a")]
    pub(crate) guard: Option<std::sync::Arc<poot_runtime_common::AllocGuard>>,
}

/// One [`Context::submit_encoded`] call's registered-but-not-yet-drained timestamp readback under
/// the typed Card 552 path (review F1: the one submit path, not a second encoder). `map_async` is
/// registered without a dedicated poll, so collecting this detail adds no synchronization beyond
/// whatever the caller's step already does, up to [`Context::max_in_flight_queries`]'s bound
/// (review F3): past it, [`Context::drain_device_timing`]'s capacity check polls early.
pub(crate) struct PendingDeviceTiming {
    pub(crate) read: wgpu::Buffer,
    /// The dispatch index (within the whole step's recording) of this chunk's first dispatch.
    pub(crate) base_index: usize,
    pub(crate) count: usize,
    #[allow(
        dead_code,
        reason = "held only for its Drop side effect (releases the Staging charge)"
    )]
    pub(crate) guard: std::sync::Arc<poot_runtime_common::AllocGuard>,
}

/// Running fold of every [`PendingDeviceTiming`] drained so far this step (review F3: a resource-
/// capacity drain mid-step and the step's own final drain both fold into this one accumulator, so
/// neither loses data and the final result is their sum).
#[derive(Default)]
pub(crate) struct DeviceTimingAccumulator {
    pub(crate) sum: std::time::Duration,
    pub(crate) min_start: Option<u64>,
    pub(crate) max_end: Option<u64>,
    pub(crate) per_dispatch: Vec<(usize, std::time::Duration)>,
}

impl DeviceTimingAccumulator {
    /// Read `pending`'s now-mapped buffer (the caller must already have polled it ready) and fold
    /// its per-dispatch ticks, converted by `period`, into this accumulator. The one tick-
    /// conversion helper (review F1), shared by the early-capacity drain and the step's own final
    /// drain.
    pub(crate) fn fold(&mut self, pending: &PendingDeviceTiming, period: f64) {
        let ticks: Vec<u64> = {
            let data = pending.read.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&data).to_vec()
        };
        for i in 0..pending.count {
            let start = ticks[i * 2];
            let end = ticks[i * 2 + 1];
            let delta = end.saturating_sub(start);
            let dur = std::time::Duration::from_nanos((delta as f64 * period) as u64);
            self.sum += dur;
            self.per_dispatch.push((pending.base_index + i, dur));
            self.min_start = Some(self.min_start.map_or(start, |m| m.min(start)));
            self.max_end = Some(self.max_end.map_or(end, |m| m.max(end)));
        }
    }
}

/// [`Context::drain_device_timing`]'s result (Card 552): the exact sum of every drained dispatch's
/// own duration, the diagnostic first-to-last span across them, and each one's own `(index,
/// duration)` pair in replay order.
pub struct DeviceTimingResult {
    pub sum: std::time::Duration,
    pub span: std::time::Duration,
    pub per_dispatch: Vec<(usize, std::time::Duration)>,
}

/// A borrowed view over one dispatch's built objects, for [`Context::submit_encoded`] (the shared
/// encode+submit tail). [`Context::submit_cached`] (borrows caller-held [`CachedDispatch`]es) hands
/// `submit_encoded` a lazy iterator of these directly (Card 600, SC-003) - never collected into a
/// `Vec`, so a replayed step makes no host allocation building this view.
pub(crate) struct EncodeItem<'a> {
    pub(crate) pipeline: &'a wgpu::ComputePipeline,
    pub(crate) bind_group: &'a wgpu::BindGroup,
    pub(crate) groups: [u32; 3],
    pub(crate) label: &'a str,
}

/// One dispatch's resolved, reusable objects (card 156 phase 3/4): the compute pipeline (cached by
/// key), a bind group over stable buffer handles, and the workgroup-count grid. Built once by
/// [`Context::build_cached_dispatch`] over buffers that keep their identity across decode steps (the
/// pooled output, the ping-pong KV pair member for this parity, the persistent Slot buffers) and
/// re-encoded every step via [`Context::submit_cached`] with no further buffer/bind-group work.
/// Opaque outside this crate (poot-gpu holds these per parity but never inspects the fields).
pub struct CachedDispatch {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) groups: [u32; 3],
    pub(crate) label: String,
    /// The reserved error-word storage buffer, if this dispatch's `Body` has a trap (card 531c). This same buffer is bound and reused every step, so `Context::submit_cached`
    /// re-zeroes it and re-registers it as a [`PendingFault`] on every call rather than once at build
    /// time.
    pub(crate) error_storage: Option<wgpu::Buffer>,
    /// Role-tagged live-bytes guard for the length buffer baked into `bind_group` (Card 547a SC-005:
    /// direct validation/timestamp-adjacent staging the one memory service must see). Held only for
    /// its `Drop`, which fires when this `CachedDispatch` (and the `bind_group` referencing the length
    /// buffer) drops.
    #[allow(dead_code, reason = "held only for its Drop side effect, Card 547a")]
    pub(crate) length_buffer_guard: std::sync::Arc<poot_runtime_common::AllocGuard>,
    /// Same accounting for [`Self::error_storage`], when present.
    #[allow(dead_code, reason = "held only for its Drop side effect, Card 547a")]
    pub(crate) error_storage_guard: Option<std::sync::Arc<poot_runtime_common::AllocGuard>>,
}
