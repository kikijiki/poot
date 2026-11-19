use poot_runtime_common::{AllocGuard, BufferRole};
use poot_target::{BufferStorage, ElementKind, LogicalDType, StorageLayout};
use std::sync::Arc;

use cudarc::driver::sys;

use super::error::PtxError;
use super::owner::PtxOwner;

/// A device-resident native buffer. Cloning shares the underlying allocation (`Arc`-backed, like the wgpu
/// `DeviceBuffer`, R471-010); used for reshape (which aliases its input) and for carrying KV state across
/// steps. The allocation frees once the last clone drops (best-effort, on a thread where the owning
/// context can be made current). A `PtxBuffer` (and the [`PtxOwner`] it keeps alive) is `Send`: it may be
/// built on one thread and moved to another before use, as long as the new thread makes the context
/// current (see [`crate::PtxContext`]'s module doc) before touching it again.
///
/// Allocation metadata is private and immutable; consumers use accessors and cannot forge a logical length:
///
/// ```compile_fail,E0616
/// fn forge(mut buffer: poot_ptx_runtime::PtxBuffer) {
///     buffer.alloc = buffer.alloc.clone();
/// }
/// ```
#[derive(Clone)]
pub struct PtxBuffer {
    pub(crate) alloc: Arc<PtxAlloc>,
    /// Native element count passed to the slice ABI: the allocation's, or a typed view's
    /// ([`Self::view`], POOT-1450). Handle-level, so one view never changes another clone's.
    elem_count: usize,
    /// The storage this handle records: the allocation's, or a typed view's. Handle-level, as `elem_count`.
    storage: BufferStorage,
}

impl PtxBuffer {
    pub(crate) fn ptr(&self) -> sys::CUdeviceptr {
        self.alloc.ptr
    }

    /// This buffer's bytes typed as `storage` (POOT-1450): a handle sharing the allocation (nothing is
    /// allocated or charged) whose storage and element count are the view's, so every check that reads a
    /// handle's storage (a launch's element schema, a typed transfer) checks the view. A typed buffer
    /// admits only its own storage; an arena slot of bytes any encoded storage in whole elements.
    ///
    /// # Errors
    ///
    /// [`PtxError::View`] when [`BufferStorage::view_elems`] refuses the view.
    pub fn view(&self, storage: BufferStorage) -> Result<Self, PtxError> {
        Ok(Self {
            alloc: Arc::clone(&self.alloc),
            elem_count: self.storage.view_elems(self.elem_count, storage)?,
            storage,
        })
    }

    /// Immutable logical element count passed to the native slice ABI.
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Immutable byte capacity established before the native allocation call.
    pub fn byte_capacity(&self) -> usize {
        self.alloc.byte_capacity
    }

    /// Immutable native element representation (card 527: the physical word of [`Self::storage`]).
    pub fn element(&self) -> ElementKind {
        self.storage.element()
    }

    /// Immutable logical dtype this buffer represents (card 527): may differ from [`Self::element`] for
    /// a packed or widened value.
    pub fn dtype(&self) -> LogicalDType {
        self.storage.dtype()
    }

    /// Immutable layout (card 527): how [`Self::element`] packs [`Self::dtype`].
    pub fn layout(&self) -> StorageLayout {
        self.storage.layout()
    }

    /// Immutable full storage contract (card 527, R471-009/R484-001): element kind, logical dtype and
    /// layout together, set at allocation from the plan's storage record.
    pub fn storage(&self) -> BufferStorage {
        self.storage
    }
}

pub(crate) struct PtxAlloc {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) byte_capacity: usize,
    pub(crate) owner: Option<Arc<PtxOwner>>,
    /// Role-tagged live-bytes guard (Card 547a: the one memory service), `Some` iff `owner` is (a
    /// buffer with no owner frees nothing, so nothing is charged either); run manually alongside the
    /// native free in `Drop for PtxAlloc`, never twice.
    pub(crate) guard: Option<std::mem::ManuallyDrop<AllocGuard>>,
}

pub(crate) fn allocate_owned(
    owner: Arc<PtxOwner>,
    op: &'static str,
    elements: usize,
    storage: BufferStorage,
    role: BufferRole,
    allocate: impl FnOnce(usize) -> Result<sys::CUdeviceptr, PtxError>,
    initialize: impl FnOnce(sys::CUdeviceptr, usize) -> Result<(), PtxError>,
) -> Result<PtxBuffer, PtxError> {
    let bytes = poot_runtime_common::checked_layout::<PtxError>(op, elements, storage)?;
    let ptr = allocate(bytes)?;
    let guard = std::mem::ManuallyDrop::new(owner.memory.record_alloc(role, bytes as u64));
    let buffer = PtxBuffer {
        alloc: Arc::new(PtxAlloc {
            ptr,
            byte_capacity: bytes,
            owner: Some(owner),
            guard: Some(guard),
        }),
        elem_count: elements,
        storage,
    };
    initialize(ptr, bytes)?;
    Ok(buffer)
}

impl poot_runtime_common::CopyErrors for PtxError {
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

impl poot_runtime_common::CopyTarget for PtxBuffer {
    type Error = PtxError;

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

impl Drop for PtxAlloc {
    fn drop(&mut self) {
        if self.ptr != 0
            && let Some(owner) = &self.owner
            && let Ok(_guard) = owner.current_guard()
        {
            owner.driver.free(self.ptr);
            if let Some(guard) = &mut self.guard {
                // SAFETY: this is the only place `self.guard` is dropped, and only once, only on
                // the path that actually freed the native allocation above.
                unsafe {
                    std::mem::ManuallyDrop::drop(guard);
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    pub(crate) fn checked_buffer(
        ptr: sys::CUdeviceptr,
        op: &'static str,
        elements: usize,
        storage: BufferStorage,
        owner: Option<Arc<PtxOwner>>,
    ) -> Result<PtxBuffer, PtxError> {
        let byte_capacity = poot_runtime_common::checked_layout::<PtxError>(op, elements, storage)?;
        let guard = owner.as_ref().map(|owner| {
            std::mem::ManuallyDrop::new(
                owner
                    .memory
                    .record_alloc(BufferRole::Activation, byte_capacity as u64),
            )
        });
        Ok(PtxBuffer {
            alloc: Arc::new(PtxAlloc {
                ptr,
                byte_capacity,
                owner,
                guard,
            }),
            elem_count: elements,
            storage,
        })
    }
}
