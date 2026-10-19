use std::ffi::c_void;
use std::rc::Rc;

use crate::context::L0Context;
use crate::error::{L0Error, check};

/// A host-memory allocation the device can address (`zeMemAllocHost`), freed on drop. The CPU reads and writes
/// it directly, so a transfer is a `memcpy`, and the buffer keeps its context alive.
///
/// The handle is the allocation's one owner of CPU access: it is not `Clone`, and [`L0HostBuffer::write`]
/// takes `&mut self`, so no two live references into the bytes exist. A recording that binds the buffer holds
/// a [`HostLease`] of the allocation, which only keeps it alive for the device and never reads or writes it
/// from the CPU.
pub struct L0HostBuffer(HostLease);

/// A share of a host allocation that keeps it (and its context) alive and grants no CPU access to the bytes.
pub(crate) type HostLease = Rc<HostAllocation>;

pub(crate) struct HostAllocation {
    context: L0Context,
    ptr: *mut c_void,
    len: usize,
}

impl L0HostBuffer {
    pub(crate) fn new(context: L0Context, ptr: *mut c_void, len: usize) -> Self {
        L0HostBuffer(Rc::new(HostAllocation { context, ptr, len }))
    }

    /// The buffer's length in bytes.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// A lease on the allocation, for a recording that must outlive the borrow of this buffer.
    pub(crate) fn lease(&self) -> HostLease {
        Rc::clone(&self.0)
    }

    /// The address the device is handed for this buffer.
    pub(crate) fn address(&self) -> *const c_void {
        self.0.ptr
    }

    /// Whether this buffer was allocated on `context`.
    pub(crate) fn belongs_to(&self, context: &L0Context) -> bool {
        self.0.context.same(context)
    }

    /// Let `fill` write the bytes `range` of the buffer in place.
    ///
    /// Writing needs exclusive access to the buffer; through a shared reference it does not compile:
    ///
    /// ```compile_fail
    /// fn write_through_a_shared_reference(buffer: &poot_l0_runtime::L0HostBuffer) {
    ///     let _ = buffer.write(0..1, |bytes| bytes.fill(0));
    /// }
    /// ```
    pub fn write(
        &mut self,
        range: std::ops::Range<usize>,
        fill: impl FnOnce(&mut [u8]),
    ) -> Result<(), L0Error> {
        self.check_range(&range)?;
        // SAFETY: `ptr` is a live host allocation of `len` bytes (freed only when the last lease drops),
        // `range` lies inside it, and `&mut self` makes this the only CPU reference into the allocation while
        // `fill` runs (leases never dereference it, and the device is idle between executions, which wait for
        // the queue).
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.0.ptr.cast::<u8>().add(range.start), range.len())
        };
        fill(bytes);
        Ok(())
    }

    /// Copy `out.len()` bytes starting at byte `offset` into `out`.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), L0Error> {
        let range = offset..offset.saturating_add(out.len());
        self.check_range(&range)?;
        // SAFETY: as in `write`: a live allocation, `range` inside it, no concurrent writer.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.0.ptr.cast::<u8>().add(range.start), range.len())
        };
        out.copy_from_slice(bytes);
        Ok(())
    }

    fn check_range(&self, range: &std::ops::Range<usize>) -> Result<(), L0Error> {
        if range.start <= range.end && range.end <= self.0.len {
            Ok(())
        } else {
            Err(L0Error::HostRange {
                start: range.start,
                end: range.end,
                len: self.0.len,
            })
        }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `zeMemAllocHost` on this context and is freed exactly once, here; the owned
        // context handle keeps the context alive until now.
        let result = unsafe { (self.context.funcs().mem_free)(self.context.handle(), self.ptr) };
        if let Err(error) = check("zeMemFree", result) {
            tracing::warn!(%error, "leaking a Level Zero host allocation");
        }
    }
}
