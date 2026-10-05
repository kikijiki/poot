use std::sync::Arc;

use cudarc::driver::sys;

use super::error::{BufferElement, BufferStorage, LogicalDType, PtxError, StorageLayout};
use super::owner::PtxOwner;
use poot_runtime_common::CopyRange;
pub use poot_runtime_common::{AllocGuard, BufferRole, MemoryCounterSnapshot};

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
}

impl PtxBuffer {
    pub(crate) fn ptr(&self) -> sys::CUdeviceptr {
        self.alloc.ptr
    }

    /// Immutable logical element count passed to the native slice ABI.
    pub fn elem_count(&self) -> u32 {
        self.alloc.elem_count
    }

    /// Immutable byte capacity established before the native allocation call.
    pub fn byte_capacity(&self) -> usize {
        self.alloc.byte_capacity
    }

    /// Immutable native element representation (card 527: the physical word of [`Self::storage`]).
    pub fn element(&self) -> BufferElement {
        self.alloc.storage.element()
    }

    /// Immutable logical dtype this buffer represents (card 527): may differ from [`Self::element`] for
    /// a packed or widened value.
    pub fn dtype(&self) -> LogicalDType {
        self.alloc.storage.dtype()
    }

    /// Immutable layout (card 527): how [`Self::element`] packs [`Self::dtype`].
    pub fn layout(&self) -> StorageLayout {
        self.alloc.storage.layout()
    }

    /// Immutable full storage contract (card 527, R471-009/R484-001): element kind, logical dtype and
    /// layout together, set at allocation from the plan's storage record.
    pub fn storage(&self) -> BufferStorage {
        self.alloc.storage
    }

    /// Replace this buffer's recorded storage with `storage` (card 527). The low-level
    /// `PtxContext::upload_*` methods tag a buffer by which method was called, which is right for the
    /// common case but wrong when a caller needs to record a different logical decision than the upload
    /// method's default; the caller, which knows the plan's real decision for this value, retags the
    /// freshly uploaded buffer before binding it.
    ///
    /// Panics if this handle already has another clone (`with_storage` retags the one live allocation;
    /// a shared buffer must be retagged before its first clone, never after).
    pub fn with_storage(mut self, storage: BufferStorage) -> Self {
        Arc::get_mut(&mut self.alloc)
            .expect("with_storage: buffer already has another clone; retag before sharing it")
            .storage = storage;
        self
    }
}

pub(crate) struct PtxAlloc {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) elem_count: u32,
    pub(crate) byte_capacity: usize,
    pub(crate) storage: BufferStorage,
    pub(crate) owner: Option<Arc<PtxOwner>>,
    /// Role-tagged live-bytes guard (Card 547a: the one memory service), `Some` iff `owner` is (a
    /// buffer with no owner frees nothing, so nothing is charged either); run manually alongside the
    /// native free in `Drop for PtxAlloc`, never twice.
    pub(crate) guard: Option<std::mem::ManuallyDrop<AllocGuard>>,
}

pub(crate) fn checked_layout(
    op: &'static str,
    elements: usize,
    storage: BufferStorage,
) -> Result<(u32, usize), PtxError> {
    let element_bytes = storage.element().byte_width();
    poot_runtime_common::checked_layout(elements, element_bytes).map_err(|error| match error {
        poot_runtime_common::LayoutError::ElementCountTooLarge => {
            PtxError::ElementCountTooLarge { op, elements }
        }
        poot_runtime_common::LayoutError::ByteSizeOverflow => PtxError::ByteSizeOverflow {
            op,
            elements,
            element_bytes,
        },
    })
}

#[cfg(test)]
pub(crate) fn checked_buffer(
    ptr: sys::CUdeviceptr,
    op: &'static str,
    elements: usize,
    storage: BufferStorage,
    owner: Option<Arc<PtxOwner>>,
) -> Result<PtxBuffer, PtxError> {
    let (elem_count, byte_capacity) = checked_layout(op, elements, storage)?;
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
            elem_count,
            byte_capacity,
            storage,
            owner,
            guard,
        }),
    })
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
    let (elem_count, bytes) = checked_layout(op, elements, storage)?;
    let ptr = allocate(bytes)?;
    let guard = std::mem::ManuallyDrop::new(owner.memory.record_alloc(role, bytes as u64));
    let buffer = PtxBuffer {
        alloc: Arc::new(PtxAlloc {
            ptr,
            elem_count,
            byte_capacity: bytes,
            storage,
            owner: Some(owner),
            guard: Some(guard),
        }),
    };
    initialize(ptr, bytes)?;
    Ok(buffer)
}

fn map_copy_range_error(
    op: &'static str,
    element_offset: usize,
    elements: usize,
    error: poot_runtime_common::CopyRangeError,
) -> PtxError {
    match error {
        poot_runtime_common::CopyRangeError::SizeMismatch { have, got } => {
            PtxError::SizeMismatch { op, have, got }
        }
        poot_runtime_common::CopyRangeError::RangeOverflow => PtxError::RangeOverflow {
            op,
            element_offset,
            elements,
        },
        poot_runtime_common::CopyRangeError::RangeOutOfBounds {
            byte_offset,
            byte_end,
            byte_capacity,
        } => PtxError::RangeOutOfBounds {
            op,
            byte_offset,
            byte_end,
            byte_capacity,
        },
    }
}

/// Validate a copy against `buf`'s full storage contract (card 527, R484-001/R471-009): a release
/// check on element kind, logical dtype and layout together, not element kind alone (which a packed or
/// widened value can share with a buffer of a different logical meaning; see
/// [`poot_target::BufferStorage`]).
pub(crate) fn checked_copy<T>(
    buf: &PtxBuffer,
    op: &'static str,
    expected: BufferStorage,
    element_offset: usize,
    elements: usize,
    whole_buffer: bool,
    copy: impl FnOnce(CopyRange) -> Result<T, PtxError>,
) -> Result<T, PtxError> {
    if buf.storage() != expected {
        return Err(PtxError::RepresentationMismatch {
            op,
            expected,
            actual: buf.storage(),
        });
    }
    let range = poot_runtime_common::checked_copy_range(
        element_offset,
        elements,
        expected.element().byte_width(),
        whole_buffer,
        buf.elem_count(),
        buf.byte_capacity(),
    )
    .map_err(|error| map_copy_range_error(op, element_offset, elements, error))?;
    copy(range)
}

/// Validate a byte range against `buf`'s capacity alone, with no storage/dtype check (Card 547a: the
/// storage-typed primitive's own range check, deliberately dtype-agnostic - [`PtxContext::write_bytes`]/
/// [`PtxContext::read_bytes`] serve any role's buffer, mirroring `poot_runtime::Context::write_bytes`'s
/// lack of a storage check).
pub(crate) fn checked_byte_range(
    op: &'static str,
    buf: &PtxBuffer,
    len: usize,
) -> Result<CopyRange, PtxError> {
    poot_runtime_common::checked_copy_range(0, len, 1, false, buf.elem_count(), buf.byte_capacity())
        .map_err(|error| map_copy_range_error(op, 0, len, error))
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
