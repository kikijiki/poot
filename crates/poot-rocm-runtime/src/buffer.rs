use crate::RocmError;
use crate::context::DevicePoison;
use crate::owner::ResourceOwner;
use poot_runtime_common::{AllocGuard, BufferRole};
use poot_target::{BufferStorage, ElementKind, LogicalDType, StorageLayout};
use std::ffi::c_void;
use std::sync::Arc;

use poot_runtime_common::CopyRange;

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
    /// (Card 547a: moved out of the shared [`RocmAlloc`] so [`Self::with_storage`] can retag a clone
    /// without needing sole ownership of the physical allocation).
    pub(crate) elem_count: u32,
    /// Byte capacity established before the native allocation call. Handle-level; see
    /// [`Self::elem_count`].
    pub(crate) byte_capacity: usize,
    /// The storage this handle currently records (card 527). Handle-level: a caller that legitimately
    /// changes what a reused buffer holds retags one handle via [`Self::with_storage`] without
    /// affecting any other clone's view, exactly like `poot-wgpu-runtime`'s `DeviceBuffer`.
    pub(crate) storage: BufferStorage,
}

impl RocmBuffer {
    /// Raw device pointer. Caller is responsible for not racing with a queued dispatch.
    pub fn device_ptr(&self) -> u64 {
        self.alloc.ptr as u64
    }

    /// Logical element count passed to the native slice ABI.
    pub fn elem_count(&self) -> u32 {
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

pub(crate) fn checked_layout(
    op: &'static str,
    elements: usize,
    storage: BufferStorage,
) -> Result<(u32, usize), RocmError> {
    let element_bytes = storage.element().byte_width();
    poot_runtime_common::checked_layout(elements, element_bytes).map_err(|error| match error {
        poot_runtime_common::LayoutError::ElementCountTooLarge => {
            RocmError::ElementCountTooLarge { op, elements }
        }
        poot_runtime_common::LayoutError::ByteSizeOverflow => RocmError::ByteSizeOverflow {
            op,
            elements,
            element_bytes,
        },
    })
}

fn map_copy_range_error(
    op: &'static str,
    element_offset: usize,
    elements: usize,
    error: poot_runtime_common::CopyRangeError,
) -> RocmError {
    match error {
        poot_runtime_common::CopyRangeError::SizeMismatch { have, got } => {
            RocmError::SizeMismatch { op, have, got }
        }
        poot_runtime_common::CopyRangeError::RangeOverflow => RocmError::RangeOverflow {
            op,
            element_offset,
            elements,
        },
        poot_runtime_common::CopyRangeError::RangeOutOfBounds {
            byte_offset,
            byte_end,
            byte_capacity,
        } => RocmError::RangeOutOfBounds {
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
    buf: &RocmBuffer,
    op: &'static str,
    expected: BufferStorage,
    element_offset: usize,
    elements: usize,
    whole_buffer: bool,
    copy: impl FnOnce(CopyRange) -> Result<T, RocmError>,
) -> Result<T, RocmError> {
    if buf.storage() != expected {
        return Err(RocmError::RepresentationMismatch {
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
/// storage-typed primitive's own range check, deliberately dtype-agnostic - [`RocmContext::write_with`]/
/// [`RocmContext::read_bytes`] serve any role's buffer, mirroring `poot_wgpu_runtime::Context::write_with`'s
/// lack of a storage check).
pub(crate) fn checked_byte_range(
    op: &'static str,
    buf: &RocmBuffer,
    offset: usize,
    len: usize,
) -> Result<CopyRange, RocmError> {
    poot_runtime_common::checked_copy_range(
        offset,
        len,
        1,
        false,
        buf.elem_count(),
        buf.byte_capacity(),
    )
    .map_err(|error| map_copy_range_error(op, offset, len, error))
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
