use crate::*;

pub use poot_runtime_common::{AllocGuard, BufferRole, MemoryCounterSnapshot};
/// Element kind, logical dtype and layout (card 527): the storage contract every runtime's buffer
/// handle now carries. See [`poot_target::BufferStorage`].
pub use poot_target::{BufferStorage, ElementKind, LogicalDType, StorageLayout};

/// A persistent, host-visible+coherent Vulkan buffer (see the module doc's "Memory model"), mapped for
/// its whole lifetime. [`DeviceBuffer::read_f32`] copies from the mapped pointer into owned host data.
///
/// The Vulkan objects live behind an `Arc<DeviceBufferInner>` so the buffer is cheaply `Clone` (like
/// ROCm's `RocmBuffer`; spec 133 P2 slice 2b, card 139). Device-resident capture needs this for
/// `Plan::Alias` (Reshape): the aliased output shares its input's buffer by reference count. The count
/// is atomic and the shared owner is `Send + Sync`, so a buffer moves to another thread whole (R471-010,
/// as `PtxBuffer` does); host reads and writes through the mapping serialize on a per-allocation lock.
///
/// The element count a dispatch writes into the kernel's length buffer is derived from the allocation
/// ([`DeviceBuffer::elem_count`]), so no caller can widen a kernel's bound past the memory behind it:
///
/// ```compile_fail,E0615
/// fn widen(buffer: &mut poot_vulkan_runtime::DeviceBuffer) {
///     buffer.elem_count = u32::MAX;
/// }
/// ```
#[derive(Clone)]
pub struct DeviceBuffer {
    pub(crate) inner: Arc<DeviceBufferInner>,
}

pub(crate) struct DeviceBufferInner {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) ptr: *mut u8,
    pub(crate) byte_len: usize,
    /// Serializes host reads and writes through `ptr`, so two clones on two threads cannot race a copy.
    pub(crate) host_access: Mutex<()>,
    /// Element kind, logical dtype and layout, set at allocation (card 527, R471-009/R484-001): before
    /// this field, a Vulkan buffer carried no element kind at all (unlike ROCm/PTX's `BufferElement`).
    pub(crate) storage: BufferStorage,
    /// Role-tagged live-bytes guard (Card 547a: the one memory service), held only for its `Drop`, run
    /// manually in `Drop for DeviceBufferInner`, never twice (leaked, like the native allocation, when
    /// the device is poisoned).
    pub(crate) guard: std::mem::ManuallyDrop<AllocGuard>,
    /// Last so the native allocation is torn down before releasing its shared device owner.
    pub(crate) owner: Arc<DeviceOwner>,
}

// SAFETY: the only non-`Send` field is `ptr`, the start of a `vkMapMemory` mapping that is valid for
// `byte_len` bytes until `Drop` (which runs on whichever thread drops the last `Arc`; Vulkan allows
// `vkUnmapMemory`/`vkFreeMemory` from any thread given external synchronization, which sole ownership at
// drop provides). Every other field is `Send`.
unsafe impl Send for DeviceBufferInner {}
// SAFETY: a shared reference reaches `ptr` only through `read_bytes_at`/`write_bytes_at`, which hold
// `host_access` for the whole copy, so no two host copies overlap. The native handles are never mutated
// after construction. A host copy racing a GPU access of the same bytes is excluded by the owner's
// contract: every submission is fence-waited before the call that issued it returns (see
// `DeviceOwner::submit_and_wait`), and a single `&mut` device issues them.
unsafe impl Sync for DeviceBufferInner {}

impl DeviceBuffer {
    /// Wrap freshly allocated resources, charged to `role` (Card 547a). `byte_len` is the exact size
    /// the resources were created with; [`BufferResources`] only exist for a non-zero size whose
    /// 4-byte element count fits a `u32`.
    pub(crate) fn from_resources(
        resources: BufferResources,
        byte_len: usize,
        storage: BufferStorage,
        role: BufferRole,
        owner: &Arc<DeviceOwner>,
    ) -> Self {
        let guard = std::mem::ManuallyDrop::new(owner.memory.record_alloc(role, byte_len as u64));
        DeviceBuffer {
            inner: Arc::new(DeviceBufferInner {
                buffer: resources.buffer,
                memory: resources.memory,
                ptr: resources.ptr,
                byte_len,
                host_access: Mutex::new(()),
                storage,
                guard,
                owner: Arc::clone(owner),
            }),
        }
    }

    /// The number of 4-byte elements this buffer holds: the bound a dispatch passes the kernel for it.
    pub fn elem_count(&self) -> u32 {
        u32::try_from(self.inner.byte_len / 4)
            .expect("allocate_buffer_resources refuses a buffer whose element count exceeds u32")
    }

    /// The allocation's size in bytes.
    pub fn byte_len(&self) -> usize {
        self.inner.byte_len
    }

    /// The native device-word representation this buffer was allocated with (card 527).
    pub fn element(&self) -> ElementKind {
        self.inner.storage.element()
    }

    /// How [`Self::element`] packs the buffer's logical dtype (card 527).
    pub fn layout(&self) -> StorageLayout {
        self.inner.storage.layout()
    }

    /// The full storage contract (card 527, R471-009/R484-001): element kind, logical dtype and layout
    /// together, set at allocation.
    pub fn storage(&self) -> BufferStorage {
        self.inner.storage
    }

    /// Whether this buffer is a child of `owner`'s device (a dispatch may bind only its own buffers).
    pub(crate) fn belongs_to(&self, owner: &Arc<DeviceOwner>) -> bool {
        Arc::ptr_eq(&self.inner.owner, owner)
    }

    /// Validate a copy against this buffer's full storage contract (card 527, R484-001/R471-009): a
    /// release check on element kind, logical dtype and layout together, not element kind alone (which
    /// a packed or widened value can share with a buffer of a different logical meaning; see
    /// [`poot_target::BufferStorage`]). Card 547a: `read_bytes`/`write_bytes` (this crate's
    /// storage-typed primitives, like `poot_runtime::Context::read_bytes`/`write_bytes`) are
    /// deliberately dtype-agnostic and do not call this; it stays as this crate's own test evidence
    /// that the full storage contract (not element kind alone) is what a real binder must check.
    #[cfg(test)]
    pub(crate) fn check_storage(
        &self,
        op: &'static str,
        expected: BufferStorage,
    ) -> Result<(), RuntimeError> {
        let actual = self.storage();
        if actual != expected {
            return Err(RuntimeError::RepresentationMismatch {
                op,
                expected,
                actual,
            });
        }
        Ok(())
    }

    /// The underlying `VkBuffer` handle, for binding into a descriptor set. `pub(crate)`: outside callers only pass `&DeviceBuffer` to the dispatch methods.
    pub(crate) fn vk_buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    /// Read the first `out.len()` bytes of this buffer's `HOST_COHERENT` mapping into `out`, dtype-agnostic
    /// (Card 547a: the storage-typed read the executor contract's memory service needs, instead of one
    /// reader per dtype; mirrors `poot_runtime::Context::read_bytes`). No invalidate is needed once the
    /// writing submission has been fence-waited (see [`Context::dispatch`]). Refused with
    /// [`RuntimeError::Poisoned`] once a lost submission may still be writing the memory, and with
    /// [`RuntimeError::WriteCapacity`] if `out` is longer than the buffer.
    pub fn read_bytes(&self, out: &mut [u8]) -> Result<(), RuntimeError> {
        self.read_bytes_at("read_bytes", 0, out)
    }

    /// Overwrite the first `bytes.len()` bytes of this buffer in place through its persistent
    /// `HOST_COHERENT` mapping (spec 133 FR-009, P2 slice 2a; Card 547a: dtype-agnostic, mirrors
    /// `poot_runtime::Context::write_bytes`). A recorded graph's descriptor sets bake this buffer's
    /// `VkBuffer` handle, so replaying with new data means overwriting this buffer. Safe between
    /// replays, since [`VulkanGraph::replay`] fence-waits before returning. `bytes` longer than the
    /// buffer is refused ([`RuntimeError::WriteCapacity`]), and so is any write once the device is
    /// poisoned.
    pub fn write_bytes(&self, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.write_bytes_at("write_bytes", 0, bytes)
    }

    /// The byte range `[offset, offset + len)` of this buffer, or [`RuntimeError::WriteCapacity`] when
    /// it does not lie inside the allocation.
    fn checked_range(&self, offset: usize, len: usize) -> Result<(), RuntimeError> {
        match offset.checked_add(len) {
            Some(end) if end <= self.inner.byte_len => Ok(()),
            _ => Err(RuntimeError::WriteCapacity {
                requested: offset.saturating_add(len),
                capacity: self.inner.byte_len,
            }),
        }
    }

    /// Read `out.len()` bytes starting at byte `offset` (see [`Self::read_bytes`]).
    pub(crate) fn read_bytes_at(
        &self,
        op: &'static str,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), RuntimeError> {
        self.inner.owner.check_live(op)?;
        self.checked_range(offset, out.len())?;
        let _host = self
            .inner
            .host_access
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `ptr` is valid and mapped for `byte_len` bytes for this buffer's whole lifetime; it is
        // the start of a `vkMapMemory` mapping at offset 0, and `offset + out.len() <= byte_len` was
        // just checked. No GPU access is concurrent: every submission fence-waits before returning, and
        // a lost one poisons the device, refused above. No host copy overlaps this one: `_host` is held.
        unsafe {
            std::ptr::copy_nonoverlapping(self.inner.ptr.add(offset), out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    /// Overwrite `bytes.len()` bytes starting at byte `offset` (see [`Self::write_bytes`]).
    pub(crate) fn write_bytes_at(
        &self,
        op: &'static str,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        self.inner.owner.check_live(op)?;
        self.checked_range(offset, bytes.len())?;
        let _host = self
            .inner
            .host_access
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `self.inner.ptr` is mapped HOST_COHERENT for `self.inner.byte_len` bytes for this
        // buffer's whole lifetime (see `Context::alloc_storage`); `offset + bytes.len() <= byte_len` was
        // just checked; `bytes` is a disjoint host slice (a byte copy has no alignment need). No GPU
        // submission can still be reading/writing this buffer's memory: every submission fence-waits
        // before returning (`DeviceOwner::submit_and_wait`), and one whose wait failed poisoned the
        // device, which `check_live` refused above. No host copy overlaps this one: `_host` is held.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.inner.ptr.add(offset), bytes.len())
        };
        Ok(())
    }

    /// Zero the whole allocation (a fresh `vkAllocateMemory` block has undefined contents).
    pub(crate) fn zero(&self) -> Result<(), RuntimeError> {
        self.inner.owner.check_live("zero")?;
        let _host = self
            .inner
            .host_access
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `ptr` is mapped for `byte_len` bytes (see `write_bytes_at`); no GPU access is pending
        // for a buffer this call is given before its first submission or between fence-waited ones.
        unsafe { std::ptr::write_bytes(self.inner.ptr, 0, self.inner.byte_len) };
        Ok(())
    }
}

impl Drop for DeviceBufferInner {
    fn drop(&mut self) {
        if self.owner.is_poisoned() {
            return; // A lost submission may still access this memory: leak it, mapped, and keep its
            // role-tagged bytes charged (never drop `self.guard`).
        }
        // SAFETY: `owner` keeps the loader, instance, and device alive through these calls. This runs only
        // for the final `Arc<DeviceBufferInner>`, and the device is not poisoned, so every submission that
        // could reference the allocation has retired.
        unsafe {
            self.owner
                .dispatch
                .unmap_memory(&self.owner.device, self.memory);
            self.owner
                .dispatch
                .destroy_buffer(&self.owner.device, self.buffer);
            self.owner
                .dispatch
                .free_memory(&self.owner.device, self.memory);
        }
        // SAFETY: this is the only place `self.guard` is dropped, and only once, only on the
        // non-poisoned path above.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.guard);
        }
    }
}
