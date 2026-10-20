use crate::owner::QueueGuard;
use crate::resources::BufferResources;
use crate::{DeviceOwner, RuntimeError};
use ash::vk;
use poot_runtime_common::{AllocGuard, BufferRole};
use poot_target::{BufferStorage, ElementKind, StorageLayout};
use std::sync::Arc;

/// A persistent, host-visible+coherent Vulkan buffer (see the module doc's "Memory model"), mapped for
/// its whole lifetime. [`DeviceBuffer::read_f32`] copies from the mapped pointer into owned host data.
///
/// The Vulkan objects live behind an `Arc<DeviceBufferInner>` so the buffer is cheaply `Clone` (like
/// ROCm's `RocmBuffer`; spec 133 P2 slice 2b, card 139). Device-resident capture needs this for
/// `Plan::Alias` (Reshape): the aliased output shares its input's buffer by reference count. The count
/// is atomic and the shared owner is `Send + Sync`, so a buffer moves to another thread whole (R471-010,
/// as `PtxBuffer` does); host reads and writes through the mapping take the device's [`QueueGuard`], the
/// one lock a replay holds across its submits and fence waits, so no host copy overlaps device access.
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
// SAFETY: a shared reference reaches `ptr` only through `read_bytes_at`/`write_with_at`/`zero_at`, which
// take the device's `QueueGuard` as an argument and hold it for the whole copy, so no two host copies
// overlap. The guard is the lock `DeviceOwner::submit_and_wait` is given and holds across a submit and its
// fence wait, so a host copy racing a GPU access of the same bytes is excluded by type, whichever threads
// the buffer, the graph and the context are on. The native handles are never mutated after construction.
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
                storage,
                guard,
                owner: Arc::clone(owner),
            }),
        }
    }

    /// The number of native elements ([`Self::element`]) this buffer holds: the bound a dispatch may tell
    /// a kernel it has, in the unit of [`crate::Binding::elems`].
    pub fn native_elems(&self) -> usize {
        self.inner.byte_len / self.inner.storage.element().byte_width()
    }

    /// The number of 4-byte words this buffer holds, whatever its element kind; not the unit of a dispatch
    /// length (see [`Self::native_elems`]).
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

    /// The underlying `VkBuffer` handle, for binding into a descriptor set. `pub(crate)`: outside callers only pass `&DeviceBuffer` to the dispatch methods.
    pub(crate) fn vk_buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    /// Read the first `out.len()` bytes of this buffer's `HOST_COHERENT` mapping into `out`, dtype-agnostic
    /// (Card 547a: the storage-typed read the executor contract's memory service needs, instead of one
    /// reader per dtype; mirrors `poot_wgpu_runtime::Context::read_bytes`). No invalidate is needed once the
    /// writing submission has been fence-waited (see [`Context::dispatch`]). Refused with
    /// [`RuntimeError::Poisoned`] once a lost submission may still be writing the memory, and with
    /// [`RuntimeError::WriteCapacity`] if `out` is longer than the buffer.
    pub fn read_bytes(&self, out: &mut [u8]) -> Result<(), RuntimeError> {
        let queue = self.inner.owner.lock_queue();
        self.read_bytes_at(&queue, "read_bytes", 0, out)
    }

    /// Overwrite `len` bytes from byte `offset` of this buffer in place through its persistent `HOST_COHERENT`
    /// mapping (spec 133 FR-009; Card 547a: dtype-agnostic). A recorded graph's descriptor sets bake this
    /// buffer's `VkBuffer` handle, so replaying with new data means overwriting this buffer; safe between
    /// replays, since [`VulkanGraph::replay`] fence-waits before returning. `fill(range, out)` writes the whole
    /// range straight into the mapping (no staging; see [`Self::write_with_at`]). A range past the end of the
    /// buffer, or any write once the device is poisoned, is refused.
    pub fn write_with(
        &self,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), RuntimeError> {
        let queue = self.inner.owner.lock_queue();
        self.write_with_at(&queue, "write_with", offset, len, fill)
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
        _queue: &QueueGuard<'_>,
        op: &'static str,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), RuntimeError> {
        self.inner.owner.check_live(op)?;
        self.checked_range(offset, out.len())?;
        // SAFETY: `ptr` is valid and mapped for `byte_len` bytes for this buffer's whole lifetime; it is
        // the start of a `vkMapMemory` mapping at offset 0, and `offset + out.len() <= byte_len` was
        // just checked. No GPU access is concurrent: every submission fence-waits before returning, and
        // a lost one poisons the device, refused above. No host copy overlaps this one: the caller holds the device's `QueueGuard`.
        unsafe {
            std::ptr::copy_nonoverlapping(self.inner.ptr.add(offset), out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    /// Overwrite `len` bytes starting at byte `offset` in place, `fill` writing them straight into the
    /// buffer's mapping: `fill(range, out)` is called once with the whole destination range, `out` being
    /// the mapped bytes. There is no staging: the buffer is host memory the GPU reads.
    pub(crate) fn write_with_at(
        &self,
        _queue: &QueueGuard<'_>,
        op: &'static str,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), RuntimeError> {
        self.inner.owner.check_live(op)?;
        self.checked_range(offset, len)?;
        // SAFETY: `self.inner.ptr` is mapped HOST_COHERENT for `self.inner.byte_len` bytes for this
        // buffer's whole lifetime (see `Context::alloc_storage`); `offset + len <= byte_len` was just
        // checked. No GPU submission can still be reading/writing this buffer's memory: every submission
        // fence-waits before returning (`DeviceOwner::submit_and_wait`), and one whose wait failed
        // poisoned the device, which `check_live` refused above. No other host access overlaps this one:
        // the caller's `QueueGuard` is held, so this is the only reference to the range while `fill` runs.
        let out = unsafe { std::slice::from_raw_parts_mut(self.inner.ptr.add(offset), len) };
        fill(offset..offset + len, out);
        Ok(())
    }

    /// Zero the whole allocation (a fresh `vkAllocateMemory` block has undefined contents).
    pub(crate) fn zero(&self) -> Result<(), RuntimeError> {
        let queue = self.inner.owner.lock_queue();
        self.zero_at(&queue)
    }

    fn zero_at(&self, _queue: &QueueGuard<'_>) -> Result<(), RuntimeError> {
        self.inner.owner.check_live("zero")?;
        // SAFETY: `ptr` is mapped for `byte_len` bytes (see `write_with_at`); no GPU access is pending
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

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;
    use poot_target::BufferStorage;

    impl DeviceBuffer {
        /// Validate a copy against this buffer's full storage contract (card 527, R484-001/R471-009): a
        /// release check on element kind, logical dtype and layout together, not element kind alone (which
        /// a packed or widened value can share with a buffer of a different logical meaning; see
        /// [`poot_target::BufferStorage`]). Card 547a: `read_bytes`/`write_with` (this crate's
        /// storage-typed primitives, like `poot_wgpu_runtime::Context::read_bytes`/`write_with`) are
        /// deliberately dtype-agnostic and do not call this; it stays as this crate's own test evidence
        /// that the full storage contract (not element kind alone) is what a real binder must check.
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
    }
}
