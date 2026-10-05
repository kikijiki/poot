use crate::*;

impl Context {
    /// Number of pipeline builds (cache misses) so far. With the cache warm a repeated decode adds 0
    /// (SC-002). For tests / introspection.
    pub fn pipeline_builds(&self) -> usize {
        self.pipeline_builds.get()
    }

    /// Number of `alloc_f32` calls (real `create_buffer`s) so far. Card 156 phase 3 (SC-001): with the
    /// cached-decode buffer pool warm, a repeated step adds 0 (the `_argmax` path's single `alloc_f32(1)`
    /// is the documented exception, spec 137 CONSIDER-2).
    pub fn buffer_allocs(&self) -> usize {
        self.buffer_allocs.get()
    }

    /// Number of `create_bind_group` calls so far. Card 156 phase 4 (SC-001): 0 after token 0 on the
    /// cached-decode path once its two per-parity bind-group sets are built.
    pub fn bind_group_creates(&self) -> usize {
        self.bind_group_creates.get()
    }

    /// Number of length/meta `create_buffer_init` calls so far (the per-dispatch `[u32;N]` length buffer
    /// and any `ComputeMeta`/chunk metadata buffer). Card 156 phase 4 (SC-001): 0 after token 0 on the
    /// cached-decode path (step-invariant for a fixed-shape decode graph, built once at cache-build time).
    pub fn length_buffer_creates(&self) -> usize {
        self.length_buffer_creates.get()
    }

    /// Number of compute `queue.submit` calls so far (the launch-tax signal; readback/upload submits are
    /// not counted). The batched re-encode does one per [`Context::submit_cached`] call (one per token);
    /// the per-dispatch paths (`dispatch` / `dispatch_dev`) submit once per dispatch (SC-002).
    pub fn submits(&self) -> usize {
        self.submits.get()
    }

    /// Total device-local VRAM in use (bytes), read from the Vulkan runtime via
    /// `VK_EXT_memory_budget`: an engine-queried, vendor- and OS-agnostic metric (a runtime GPU API, not
    /// sysfs or a vendor CLI). Device-wide, so it includes other processes' allocations (e.g. a
    /// co-resident llama.cpp) and reflects memory pressure/headroom rather than poot's own allocation
    /// gauges. `None` if the backend is not Vulkan or the query is unavailable. Cheap (a
    /// physical-device property query), safe to call per scrape.
    pub fn vram_used_bytes(&self) -> Option<u64> {
        use ash::vk;
        // SAFETY: as_hal yields the live wgpu-hal Vulkan adapter for the lifetime of the guard; the
        // physical device + instance are valid for that borrow, and the property query only reads.
        unsafe {
            let hal = self._adapter.as_hal::<wgpu_hal::vulkan::Api>()?;
            let phys = hal.raw_physical_device();
            let instance = hal.shared_instance().raw_instance();
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            instance.get_physical_device_memory_properties2(phys, &mut props2);
            // Read out the heap table; this is `props2`'s last use, so its `&mut budget` borrow ends here and
            // `budget.heap_usage` below can be read.
            let mem = props2.memory_properties;
            let used: u64 = (0..mem.memory_heap_count as usize)
                .filter(|&i| {
                    mem.memory_heaps[i]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|i| budget.heap_usage[i])
                .sum();
            Some(used)
        }
    }

    /// Total device-local VRAM available to this process (bytes), from `VK_EXT_memory_budget`
    /// (`heap_budget`, the OS-arbitrated budget: can be below the physical heap size and shrinks when
    /// other processes hold memory). Same source as [`Self::vram_used_bytes`]; `free = budget - used` is
    /// the headroom for new allocations. `None` off Vulkan or if the query is unavailable.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        use ash::vk;
        // SAFETY: same contract as `vram_used_bytes`: a read-only physical-device property query over the
        // live wgpu-hal Vulkan adapter.
        unsafe {
            let hal = self._adapter.as_hal::<wgpu_hal::vulkan::Api>()?;
            let phys = hal.raw_physical_device();
            let instance = hal.shared_instance().raw_instance();
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            instance.get_physical_device_memory_properties2(phys, &mut props2);
            let mem = props2.memory_properties;
            let total: u64 = (0..mem.memory_heap_count as usize)
                .filter(|&i| {
                    mem.memory_heaps[i]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|i| budget.heap_budget[i])
                .sum();
            Some(total)
        }
    }

    /// Number of device-to-host buffer readbacks so far. A full pooled capture-proof replay has one
    /// readback (final logits) and none for router ids between layers.
    pub fn readbacks(&self) -> usize {
        self.readbacks.get()
    }

    /// Number of readbacks so far that allocated their own staging buffer (Card 375f). Card 547a: `PacketStaging` (the pre-bound staging buffer a read could once be served from
    /// without this count moving) is deleted; every `download_f32`/`download_segments_u32`/
    /// `read_bytes` call allocates its own staging buffer, so this always equals [`Self::readbacks`].
    pub fn readback_staging_allocs(&self) -> usize {
        self.readback_staging_allocs.get()
    }

    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// this context, read by `WgpuDevice::memory` (`poot-gpu`) for `Engine::stats().memory` and
    /// directly by tests. Every [`DeviceBuffer`] allocated through this context is tagged with a role
    /// at construction and decrements its role's live bytes when its last clone drops.
    pub fn memory(
        &self,
    ) -> Vec<(
        poot_runtime_common::BufferRole,
        poot_runtime_common::MemoryCounterSnapshot,
    )> {
        self.memory.snapshot_all()
    }

    /// Record one allocation of `bytes` under `role` and return the guard that releases it: the one
    /// place every allocation primitive in this crate charges the shared [`poot_runtime_common::MemoryCounters`]
    /// (Card 547a).
    pub(crate) fn tag_alloc(
        &self,
        role: poot_runtime_common::BufferRole,
        bytes: usize,
    ) -> std::sync::Arc<poot_runtime_common::AllocGuard> {
        std::sync::Arc::new(self.memory.record_alloc(role, bytes as u64))
    }

    /// Typed, purpose-tagged native call/transfer counters (Card 552): logical dispatch/replay
    /// counts plus physical submit/wait/transfer counts by [`poot_runtime_common::CallPurpose`].
    /// Independent of the diagnostic gauges above (`submits`, `pipeline_builds`, ...), and never
    /// reset by a profiler reset - this is the cumulative cross-backend contract.
    pub fn execution_counters(&self) -> poot_runtime_common::ExecutionCounters {
        let mut ec = self.exec_counters.borrow().clone();
        // `self.dispatches` already counts every `dispatch_workgroups` call (Card 156); read it as
        // the logical count rather than keeping a second incremented copy.
        ec.logical_dispatches = self.dispatches.get() as u64;
        ec
    }

    /// Record one native submission under `purpose` (Card 552's one physical-submission total: a
    /// call site records exactly one purpose per submission, never more than one).
    pub(crate) fn record_submit(&self, purpose: poot_runtime_common::CallPurpose) {
        self.exec_counters.borrow_mut().record_submit(purpose);
    }

    pub(crate) fn record_wait(&self, purpose: poot_runtime_common::CallPurpose) {
        self.exec_counters.borrow_mut().record_wait(purpose);
    }

    /// Shorthand for the common `record_transfer(Upload, HostToDevice, bytes)` call every
    /// `upload_*`/`write_*` site makes.
    pub(crate) fn record_upload_transfer(&self, bytes: usize) {
        self.record_transfer(
            poot_runtime_common::CallPurpose::Upload,
            poot_runtime_common::TransferDirection::HostToDevice,
            bytes as u64,
        );
    }

    pub(crate) fn record_transfer(
        &self,
        purpose: poot_runtime_common::CallPurpose,
        direction: poot_runtime_common::TransferDirection,
        bytes: u64,
    ) {
        self.exec_counters
            .borrow_mut()
            .record_transfer(purpose, direction, bytes);
    }
}

/// Cheap cumulative physical counts. Native submits include readback/copy submissions; waits count
/// blocking device polls. Dispatches count encoded compute workgroups calls, not graph equations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeCounterSnapshot {
    pub pipeline_builds: usize,
    pub submits: usize,
    pub buffer_allocs: usize,
    pub bind_group_creates: usize,
    pub length_buffer_creates: usize,
    pub readbacks: usize,
    pub native_submits: usize,
    pub waits: usize,
    pub dispatches: usize,
    /// Actual calls beginning a compute pass, independent of its dispatch count.
    pub compute_passes: usize,
}

/// Read-only handle usable inside a generation callback while the executor is mutably borrowed.
/// Retains only counters, never the device or its buffers. No event history or timing queries.
#[derive(Clone)]
pub struct RuntimeCounters {
    pipeline_builds: Counter,
    submits: Counter,
    buffer_allocs: Counter,
    bind_group_creates: Counter,
    length_buffer_creates: Counter,
    readbacks: Counter,
    native_submits: Counter,
    waits: Counter,
    dispatches: Counter,
    compute_passes: Counter,
}

impl RuntimeCounters {
    pub fn snapshot(&self) -> RuntimeCounterSnapshot {
        RuntimeCounterSnapshot {
            pipeline_builds: self.pipeline_builds.get(),
            submits: self.submits.get(),
            buffer_allocs: self.buffer_allocs.get(),
            bind_group_creates: self.bind_group_creates.get(),
            length_buffer_creates: self.length_buffer_creates.get(),
            readbacks: self.readbacks.get(),
            native_submits: self.native_submits.get(),
            waits: self.waits.get(),
            dispatches: self.dispatches.get(),
            compute_passes: self.compute_passes.get(),
        }
    }
}

impl Context {
    pub fn counters(&self) -> RuntimeCounters {
        RuntimeCounters {
            pipeline_builds: self.pipeline_builds.clone(),
            submits: self.submits.clone(),
            buffer_allocs: self.buffer_allocs.clone(),
            bind_group_creates: self.bind_group_creates.clone(),
            length_buffer_creates: self.length_buffer_creates.clone(),
            readbacks: self.readbacks.clone(),
            native_submits: self.native_submits.clone(),
            waits: self.waits.clone(),
            dispatches: self.dispatches.clone(),
            compute_passes: self.compute_passes.clone(),
        }
    }
}

// One runtime writer and observe-only clones. Relaxed counters do not synchronize GPU work and keep
// Context movable between threads; callbacks sample only after the writer reaches that boundary.
#[derive(Clone, Default)]
pub(crate) struct Counter(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl Counter {
    pub(crate) fn get(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(crate) fn set(&self, value: usize) {
        self.0.store(value, std::sync::atomic::Ordering::Relaxed);
    }
}
