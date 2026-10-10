//! The upload ring (dmap.md section 6.2): a device write is staged through a small ring of pinned host
//! buffers (`cuMemHostAlloc`) and copied with `cuMemcpyHtoDAsync` on the context's stream, one event per
//! slot, so the host fills slot `k + 1` while the copy engine drains slot `k`, and no host copy of the whole
//! write exists. This is llama.cpp's no-mmap loading scheme (4 pinned buffers of 64 MiB, async copies and
//! events).
//!
//! The copies run on the context's own stream, so everything submitted after a write (graph replays,
//! readbacks) is ordered after it by the stream and a write needs no synchronize of its own; a slot is
//! waited for (its event) only before the host refills it. The ring is freed after the events of the copies
//! still in flight have completed.

use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::{result, sys};

use crate::error::PtxError;
use crate::owner::PtxOwner;

/// Slots in the ring.
const RING_SLOTS: usize = 4;

/// Bytes of one slot: one chunk of a write.
pub const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

struct Slot {
    /// Pinned host memory of [`UPLOAD_CHUNK_BYTES`] bytes (`cuMemHostAlloc`).
    host: *mut c_void,
    /// Recorded on the stream right after the slot's copy; complete when the host may refill the slot.
    event: sys::CUevent,
    /// A copy was enqueued from `host` and its event not yet waited for.
    in_flight: bool,
}

/// The ring a context writes through; its slots are allocated by the first write that needs them.
pub(crate) struct UploadRing {
    owner: Arc<PtxOwner>,
    slots: Vec<Slot>,
    next: usize,
}

impl UploadRing {
    pub(crate) fn new(owner: Arc<PtxOwner>) -> Self {
        Self {
            owner,
            slots: Vec::with_capacity(RING_SLOTS),
            next: 0,
        }
    }

    /// The slot for the next chunk, ready for the host to fill: allocated if new, otherwise its previous
    /// copy proven complete. The cursor moves past the slot only once it is ready, so a failed allocation
    /// or wait leaves the ring where it was and the next write retries the same slot.
    fn next_slot(&mut self) -> Result<&mut Slot, PtxError> {
        self.next_slot_with(allocate_slot)
    }

    /// [`Self::next_slot`] with the slot allocator supplied (the driver in production).
    fn next_slot_with(
        &mut self,
        allocate: impl FnOnce() -> Result<Slot, PtxError>,
    ) -> Result<&mut Slot, PtxError> {
        let index = self.next;
        if index == self.slots.len() {
            self.slots.push(allocate()?);
        }
        let slot = &mut self.slots[index];
        if slot.in_flight {
            // SAFETY: `event` is a live event recorded on the context's stream after the slot's copy.
            unsafe { result::event::synchronize(slot.event) }?;
            slot.in_flight = false;
        }
        self.next = (index + 1) % RING_SLOTS;
        Ok(&mut self.slots[index])
    }

    /// Write `len` bytes at device address `dst`: `fill(range, out)` gets consecutive ranges `0..len` of at
    /// most [`UPLOAD_CHUNK_BYTES`] and writes every byte of `out`, the pinned slot the copy reads. The
    /// copies are enqueued on `stream` and may still be running on return.
    ///
    /// # Safety
    /// `dst..dst + len` must be a live device allocation of this context, and `stream` its live stream.
    pub(crate) unsafe fn write(
        &mut self,
        dst: sys::CUdeviceptr,
        len: usize,
        stream: sys::CUstream,
        fill: &mut dyn FnMut(Range<usize>, &mut [u8]),
    ) -> Result<(), PtxError> {
        let mut done = 0;
        while done < len {
            let chunk = UPLOAD_CHUNK_BYTES.min(len - done);
            let slot = self.next_slot()?;
            // SAFETY: `host` is `UPLOAD_CHUNK_BYTES` bytes of pinned memory this ring owns, `chunk` of
            // them are written here, and no copy reads the slot (its event completed above).
            let out = unsafe { std::slice::from_raw_parts_mut(slot.host.cast::<u8>(), chunk) };
            fill(done..done + chunk, out);
            // SAFETY: `dst + done` is inside the allocation (function contract); the pinned slot stays
            // alive and unmodified until its event is waited for, so the asynchronous copy reads valid bytes.
            unsafe {
                result::memcpy_htod_async(dst + done as sys::CUdeviceptr, out, stream)?;
                result::event::record(slot.event, stream)?;
            }
            slot.in_flight = true;
            done += chunk;
        }
        Ok(())
    }
}

/// A new slot: pinned host memory and its event.
fn allocate_slot() -> Result<Slot, PtxError> {
    // SAFETY: a plain pinned allocation of a fixed size; freed once, in the ring's `Drop`.
    let host = unsafe { result::malloc_host(UPLOAD_CHUNK_BYTES, 0) }?;
    match result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING) {
        Ok(event) => Ok(Slot {
            host,
            event,
            in_flight: false,
        }),
        Err(error) => {
            // SAFETY: `host` was just allocated and is referenced nowhere else.
            let _ = unsafe { result::free_host(host) };
            Err(PtxError::Driver(error))
        }
    }
}

impl Drop for UploadRing {
    fn drop(&mut self) {
        if self.slots.is_empty() {
            return;
        }
        let Ok(guard) = self.owner.current_guard() else {
            // Without a current context the pinned memory and events cannot be released safely: leak them.
            return;
        };
        for slot in &self.slots {
            if slot.in_flight {
                // SAFETY: `event` is live; the copy reading `host` must finish before the memory is freed.
                if unsafe { result::event::synchronize(slot.event) }.is_err() {
                    // A copy that cannot be proven complete may still read the slot: leak it.
                    continue;
                }
            }
            self.owner.driver.destroy_event(slot.event);
            // SAFETY: `host` came from `malloc_host`, no copy reads it any more, freed once.
            let _ = unsafe { result::free_host(slot.host) };
        }
        let _ = guard.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{MockCleanup, mock_owner};

    fn fake_slot() -> Slot {
        Slot {
            host: std::ptr::null_mut(),
            event: std::ptr::null_mut(),
            in_flight: false,
        }
    }

    /// POOT-1100 review: the cursor moves only past a slot that is ready. A failing allocator leaves the ring
    /// where it was (no slot, cursor unmoved), so the retry allocates the same index; after the failure
    /// clears, the cursor advances once per slot and wraps without allocating again.
    /// Mutation: advance `self.next` before the slot is ready (the old order); the cursor skips index 0 after a
    /// failed allocation and the first assertion goes red.
    #[test]
    fn the_cursor_moves_only_past_a_ready_slot() {
        let mock = std::sync::Arc::new(MockCleanup::default());
        let mut ring = UploadRing::new(mock_owner(&mock));
        let failed = ring.next_slot_with(|| {
            Err(PtxError::Driver(cudarc::driver::DriverError(
                sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            )))
        });
        assert!(failed.is_err());
        assert_eq!(
            (ring.next, ring.slots.len()),
            (0, 0),
            "a failed allocation moves nothing"
        );
        for expected in 1..=RING_SLOTS {
            ring.next_slot_with(|| Ok(fake_slot()))
                .expect("a ready slot");
            assert_eq!(ring.next, expected % RING_SLOTS);
        }
        assert_eq!(ring.slots.len(), RING_SLOTS);
        ring.next_slot_with(|| panic!("a full ring allocates nothing"))
            .expect("wraps to slot 0");
        assert_eq!(ring.next, 1);
        // The fake slots hold no driver memory; dropping the ring would free null pointers.
        ring.slots.clear();
    }
}
