use crate::RocmError;
use crate::context::DevicePoison;
use crate::owner::ResourceOwner;
use poot_runtime_common::{AllocGuard, BufferRole};
use poot_target::{BufferStorage, ElementKind, LogicalDType, StorageLayout};
use std::ffi::c_void;
use std::sync::Arc;

/// Which native pool a ROCm allocation lives in (Card 547a): `Weight`/`Activation`/`State` allocate
/// from the coarse-grained device pool, `Input`/`Output`/`Meta`/`Staging` from the fine-grained system
/// pool. Fixed for an allocation's whole life (unlike [`RocmBuffer::storage`], which a caller may
/// retag): it decides how [`RocmContext::upload_bytes_role_aware`]/[`RocmContext::download_bytes_role_aware`]
/// reach the buffer, not what it logically holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pool {
    Coarse,
    Fine,
}

impl Pool {
    /// Weight/Activation/State are coarse-pool roles; Input/Output/Meta/Staging are fine-pool roles
    /// (Card 547a scope).
    pub(crate) fn for_role(role: BufferRole) -> Self {
        match role {
            BufferRole::Weight | BufferRole::Activation | BufferRole::State => Self::Coarse,
            BufferRole::Input | BufferRole::Output | BufferRole::Meta | BufferRole::Staging => {
                Self::Fine
            }
        }
    }
}

/// A device-resident native buffer. Cloning shares the allocation, which is freed when the last clone
/// drops, using the runtime that created it even on another thread. After a timed-out wait on that
/// runtime the allocation is never freed, since a hung packet may still use it.
///
/// Allocation metadata is private and immutable; accessors expose it but cannot forge a logical length:
///
/// ```compile_fail,E0616
/// fn forge(mut buffer: poot_rocm_runtime::RocmBuffer) {
///     buffer.alloc = buffer.alloc.clone();
/// }
/// ```
#[derive(Clone)]
pub struct RocmBuffer {
    pub(crate) alloc: Arc<RocmAlloc>,
    /// Logical element count passed to the native slice ABI. Handle-level, like [`Self::storage`]
    /// (Card 547a: moved out of the shared [`RocmAlloc`] so a typed view ([`Self::view`], POOT-1450) is
    /// a clone with its own count, without needing sole ownership of the physical allocation).
    pub(crate) elem_count: usize,
    /// Byte capacity established before the native allocation call. Handle-level; see
    /// [`Self::elem_count`].
    pub(crate) byte_capacity: usize,
    /// The storage this handle records (card 527): the allocation's, or a typed view's ([`Self::view`]).
    /// Handle-level, so one view never changes another clone's.
    pub(crate) storage: BufferStorage,
}

impl RocmBuffer {
    /// Raw device pointer. Caller is responsible for not racing with a queued dispatch.
    pub fn device_ptr(&self) -> u64 {
        self.alloc.ptr as u64
    }

    /// Logical element count passed to the native slice ABI.
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Byte capacity established before the native allocation call.
    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }

    /// The native element representation (card 527: the physical word of [`Self::storage`]).
    pub fn element(&self) -> ElementKind {
        self.storage.element()
    }

    /// The logical dtype this buffer represents (card 527): may differ from [`Self::element`] for
    /// a packed or widened value.
    pub fn dtype(&self) -> LogicalDType {
        self.storage.dtype()
    }

    /// How [`Self::element`] packs [`Self::dtype`] (card 527).
    pub fn layout(&self) -> StorageLayout {
        self.storage.layout()
    }

    /// The full storage contract (card 527, R471-009/R484-001): element kind, logical dtype and
    /// layout together, set at allocation from the plan's storage record.
    pub fn storage(&self) -> BufferStorage {
        self.storage
    }

    /// This buffer's bytes typed as `storage` (POOT-1450): a handle sharing the allocation (nothing is
    /// allocated or charged) whose storage and element count are the view's, so every check that reads a
    /// handle's storage (a dispatch's element schema, a typed transfer) checks the view. A typed buffer
    /// admits only its own storage; an arena slot of bytes any encoded storage in whole elements.
    ///
    /// # Errors
    ///
    /// [`RocmError::View`] when [`BufferStorage::view_elems`] refuses the view.
    pub fn view(&self, storage: BufferStorage) -> Result<Self, RocmError> {
        let elem_count = self.storage.view_elems(self.elem_count, storage)?;
        Ok(Self {
            alloc: Arc::clone(&self.alloc),
            elem_count,
            byte_capacity: self.byte_capacity,
            storage,
        })
    }

    /// Which native pool this allocation lives in (Card 547a).
    pub(crate) fn pool(&self) -> Pool {
        self.alloc.pool
    }
}

pub(crate) struct RocmAlloc {
    /// Device pointer (system memory on Strix Halo via HMM/SVM, host-visible). Stored as an address so
    /// the owned cleanup record can cross threads; converted back to `*mut c_void` at the HSA call.
    pub(crate) ptr: usize,
    pub(crate) pool: Pool,
    pub(crate) owner: Arc<dyn ResourceOwner>,
    /// Role-tagged live-bytes guard (Card 547a: the one memory service), held only for its `Drop`;
    /// leaked (never dropped) while [`Self::poison`] is set, so a hung packet's allocation stays
    /// charged forever instead of appearing freed.
    pub(crate) guard: std::mem::ManuallyDrop<AllocGuard>,
    /// Set once a timed-out submission of the creating runtime may still reference this allocation.
    pub(crate) poison: DevicePoison,
}

impl RocmAlloc {
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}

impl poot_runtime_common::CopyErrors for RocmError {
    fn byte_size_overflow(op: &'static str, elements: usize, element_bytes: usize) -> Self {
        Self::ByteSizeOverflow {
            op,
            elements,
            element_bytes,
        }
    }

    fn representation_mismatch(
        op: &'static str,
        expected: BufferStorage,
        actual: BufferStorage,
    ) -> Self {
        Self::RepresentationMismatch {
            op,
            expected,
            actual,
        }
    }

    fn size_mismatch(op: &'static str, have: usize, got: usize) -> Self {
        Self::SizeMismatch { op, have, got }
    }

    fn range_overflow(op: &'static str, element_offset: usize, elements: usize) -> Self {
        Self::RangeOverflow {
            op,
            element_offset,
            elements,
        }
    }

    fn range_out_of_bounds(
        op: &'static str,
        byte_offset: usize,
        byte_end: usize,
        byte_capacity: usize,
    ) -> Self {
        Self::RangeOutOfBounds {
            op,
            byte_offset,
            byte_end,
            byte_capacity,
        }
    }
}

impl poot_runtime_common::CopyTarget for RocmBuffer {
    type Error = RocmError;

    fn storage(&self) -> BufferStorage {
        self.storage()
    }

    fn elem_count(&self) -> usize {
        self.elem_count()
    }

    fn byte_capacity(&self) -> usize {
        self.byte_capacity()
    }
}

impl Drop for RocmAlloc {
    fn drop(&mut self) {
        if self.poison.is_poisoned() {
            // A hung packet may still read or write this memory: leak it, and keep it counted as
            // live - never run the guard's Drop, so the role's live bytes never decrement.
            return;
        }
        if self.ptr != 0 {
            self.owner.free_memory(self.as_ptr());
        }
        // SAFETY: this is the only place `guard` is dropped, and only once, only on the
        // non-poisoned path above.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.guard);
        }
    }
}
