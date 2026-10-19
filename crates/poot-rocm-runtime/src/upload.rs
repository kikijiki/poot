//! The upload ring (dmap.md section 6.2): a coarse-pool (device) write is staged through a small ring of
//! runtime-owned pinned host buffers and copied to the device with `hsa_amd_memory_async_copy`, so the host
//! fills slot `k + 1` while the DMA engine drains slot `k`, and no host copy of the whole write exists.
//! This is llama.cpp's no-mmap loading scheme (4 pinned buffers of 64 MiB, async copies and events), applied
//! to every coarse write instead of one buffer per write.
//!
//! A write returns with its last copies still in flight. They complete before anything else can observe or
//! overwrite the device bytes: [`RocmContext::drain_uploads`] runs ahead of every queue submission, every
//! synchronize, every other DMA and the context's drop, and a write whose destination overlaps an
//! in-flight copy drains first. The caller keeps the destination alive until then, exactly as it keeps the
//! buffers an enqueued dispatch binds. A copy whose bounded wait times out is abandoned with its staging
//! buffer and its signal (the device may still be reading it) and the context is poisoned, as for any
//! other DMA ([`RocmContext::wait_completion_bounded`]).

use crate::buffer::Pool;
use crate::context::{Held, StagingBuffer};
use crate::error::check;
use std::ops::Range;

use crate::{RocmContext, RocmError, bindings};
use std::ffi::c_void;

/// Slots in the ring.
const RING_SLOTS: usize = 4;

/// Bytes of one slot: one chunk of a write.
pub const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

/// The pinned size of a slot asked to hold `need` bytes: the next power of two, at least 64 KiB, at most one
/// chunk.
fn slot_bytes(need: usize) -> usize {
    need.next_power_of_two().clamp(64 << 10, UPLOAD_CHUNK_BYTES)
}

/// A copy issued and not yet proven complete.
struct InFlight {
    signal: bindings::hsa_signal_t,
    /// The device addresses it writes.
    dst: Range<usize>,
}

enum SlotState {
    /// Not allocated yet: a write that never needs the slot never pins its 64 MiB.
    Empty,
    Ready(StagingBuffer),
    Busy(StagingBuffer, InFlight),
    /// A timed-out copy owns the staging buffer now ([`TimeoutLeaks`]); the context is poisoned.
    Abandoned,
}

/// The ring and the slot the next chunk uses.
pub(crate) struct UploadRing {
    slots: [SlotState; RING_SLOTS],
    next: usize,
}

impl Default for UploadRing {
    fn default() -> Self {
        Self {
            slots: [
                SlotState::Empty,
                SlotState::Empty,
                SlotState::Empty,
                SlotState::Empty,
            ],
            next: 0,
        }
    }
}

impl UploadRing {
    /// A slot whose copy was never proven complete must not free its pinned memory: if a drain failed,
    /// the staging buffers of the copies still in flight are leaked rather than dropped.
    pub(crate) fn forget_in_flight(&mut self) {
        for slot in &mut self.slots {
            if let SlotState::Busy(staging, _) = std::mem::replace(slot, SlotState::Abandoned) {
                std::mem::forget(staging);
            }
        }
    }
}

impl RocmContext {
    pub(crate) fn lock_ring(&self) -> std::sync::MutexGuard<'_, UploadRing> {
        self.upload_ring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Prove slot `index`'s copy complete and make its staging buffer ready for a chunk of `need` bytes (0: just
    /// settle the slot); a copy still in flight is waited for here, and nowhere else but
    /// [`Self::drain_uploads`]. A slot is pinned at the size its first chunk asks for (a power of two, at most
    /// [`UPLOAD_CHUNK_BYTES`]) and grows only when a later chunk is larger, so a small write pins a small
    /// buffer, not 64 MiB.
    fn ready_slot(
        &self,
        ring: &mut UploadRing,
        index: usize,
        need: usize,
        tag: &'static str,
    ) -> Result<StagingBuffer, RocmError> {
        let staging = self.settled_slot(ring, index, need, tag)?;
        if staging.len() >= need {
            return Ok(staging);
        }
        // Its copy is complete, so the smaller buffer can be released; a failed pin leaves the slot empty.
        drop(staging);
        ring.slots[index] = SlotState::Empty;
        self.dma_staging(slot_bytes(need))
    }

    /// The slot's staging buffer with its last copy proven complete, allocated at [`slot_bytes`]`(need)` if new.
    fn settled_slot(
        &self,
        ring: &mut UploadRing,
        index: usize,
        need: usize,
        tag: &'static str,
    ) -> Result<StagingBuffer, RocmError> {
        match std::mem::replace(&mut ring.slots[index], SlotState::Abandoned) {
            SlotState::Empty => {
                let staging = self.dma_staging(slot_bytes(need));
                if staging.is_err() {
                    ring.slots[index] = SlotState::Empty;
                }
                staging
            }
            SlotState::Ready(staging) => Ok(staging),
            SlotState::Busy(staging, flight) => {
                // Held across the wait, which destroys the signal: a free must never poll a destroyed one.
                let mut listed = self
                    .hsa
                    .uploads_in_flight
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let held = self.wait_completion_bounded(
                    flight.signal,
                    tag,
                    Held {
                        staging: Some(staging),
                        ..Held::default()
                    },
                )?;
                listed.retain(|&signal| signal.handle != flight.signal.handle);
                Ok(held
                    .staging
                    .expect("a completed wait hands back the staging buffer it was given"))
            }
            SlotState::Abandoned => Err(RocmError::ContextPoisoned { op: tag }),
        }
    }

    /// Wait for every upload copy in flight whose destination overlaps `range` (all of them for `None`).
    fn drain_ring(
        &self,
        ring: &mut UploadRing,
        range: Option<Range<usize>>,
        tag: &'static str,
    ) -> Result<(), RocmError> {
        for index in 0..RING_SLOTS {
            let overlaps = match (&ring.slots[index], &range) {
                (SlotState::Busy(_, flight), Some(range)) => {
                    flight.dst.start < range.end && range.start < flight.dst.end
                }
                (SlotState::Busy(..), None) => true,
                _ => false,
            };
            if overlaps {
                let staging = self.ready_slot(ring, index, 0, tag)?;
                ring.slots[index] = SlotState::Ready(staging);
            }
        }
        Ok(())
    }

    /// Wait for every upload copy in flight: the ordering point ahead of a queue submission, a
    /// synchronize and any other DMA.
    pub(crate) fn drain_uploads(&self, tag: &'static str) -> Result<(), RocmError> {
        let mut ring = self.lock_ring();
        self.drain_ring(&mut ring, None, tag)
    }

    /// Write `len` bytes at device address `dst` (a coarse-pool allocation) through the ring:
    /// `fill(range, out)` gets consecutive ranges `0..len` of at most [`UPLOAD_CHUNK_BYTES`] and writes
    /// every byte of `out`, the pinned slot the copy reads. Returns with the last copies in flight.
    ///
    /// # Safety
    /// `dst` must point to a live coarse-pool allocation of at least `len` bytes that stays alive until
    /// the next [`Self::drain_uploads`] returns.
    pub(crate) unsafe fn ring_write(
        &self,
        dst: *mut c_void,
        len: usize,
        fill: &mut dyn FnMut(Range<usize>, &mut [u8]),
    ) -> Result<(), RocmError> {
        self.ensure_not_poisoned("ring_write")?;
        let base = dst as usize;
        let mut ring = self.lock_ring();
        // An earlier copy to bytes this write covers must land first, or the later write could be undone.
        self.drain_ring(&mut ring, Some(base..base + len), "ring_write")?;
        let mut done = 0;
        while done < len {
            let chunk = UPLOAD_CHUNK_BYTES.min(len - done);
            let index = ring.next;
            ring.next = (index + 1) % RING_SLOTS;
            let mut staging = self.ready_slot(&mut ring, index, chunk, "ring_write")?;
            fill(done..done + chunk, &mut staging.as_mut_slice()[..chunk]);
            // SAFETY: `dst + done` is inside the allocation (function contract) and `staging` holds `chunk`
            // freshly written bytes; the copy's staging buffer moves into the ring slot, which keeps it pinned
            // and alive until the copy is proven complete.
            let begun =
                unsafe { self.dma_copy_begin(dst.byte_add(done), staging.agent_ptr(), chunk, &[]) };
            match begun {
                Ok(signal) => {
                    self.hsa
                        .uploads_in_flight
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(signal);
                    ring.slots[index] = SlotState::Busy(
                        staging,
                        InFlight {
                            signal,
                            dst: base + done..base + done + chunk,
                        },
                    );
                }
                Err(error) => {
                    // Nothing was submitted for this chunk: the slot is free again.
                    ring.slots[index] = SlotState::Ready(staging);
                    // The chunks before it are in flight; the caller drops `dst` on this error, so wait.
                    self.drain_ring(&mut ring, None, "ring_write")?;
                    return Err(error);
                }
            }
            done += chunk;
        }
        Ok(())
    }

    /// Zero `len` bytes at device address `dst` on the device: `hsa_amd_memory_fill` over whole words, then
    /// the tail of fewer than four bytes through the ring. A fresh allocation is zeroed here, not by
    /// copying a host buffer of zeros.
    ///
    /// # Safety
    /// `dst` must point to a live allocation of at least `len` bytes, 4-byte aligned.
    pub(crate) unsafe fn zero_fill(
        &self,
        pool: Pool,
        dst: *mut c_void,
        len: usize,
    ) -> Result<(), RocmError> {
        // No poison check: filling touches no queue, and a poisoned context still hands out kernarg buffers.
        // A copy still in flight into these bytes (a write to an earlier allocation at this address, freed
        // since) must land before the fill, or it would overwrite the zeros. Freeing already waits for it;
        // this is the second line of defence for a range the ring still names. A poisoned context cannot
        // prove completion, and its allocations are leaked, so the result is ignored.
        let base = dst as usize;
        let _ = self.drain_ring(&mut self.lock_ring(), Some(base..base + len), "zero_fill");
        let words = len / 4;
        if words > 0 {
            // SAFETY: `dst` is 4-byte aligned and `words * 4 <= len` bytes of a live allocation (function
            // contract); the call fills synchronously.
            unsafe {
                check((self.hsa.funcs.hsa_amd_memory_fill)(dst, 0, words))?;
            }
        }
        let tail = len - words * 4;
        if tail > 0 {
            let zeros = [0u8; 3];
            // SAFETY: `dst + words * 4` is inside the allocation with `tail` bytes remaining; `zeros`
            // holds three readable bytes and `tail <= 3`.
            unsafe {
                self.upload_bytes_role_aware(pool, zeros.as_ptr(), dst.byte_add(words * 4), tail)?;
            }
        }
        Ok(())
    }
}

/// How long a free waits for a copy into the memory it releases before it leaks the memory instead.
const FREE_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

impl crate::owner::HsaRuntimeOwner {
    /// Wait until no upload copy is in flight; `false` if one cannot be proven complete in [`FREE_WAIT`].
    /// The signals are only observed: the ring still owns them and settles their slots on its next write.
    pub(crate) fn await_uploads(&self) -> bool {
        let signals = self
            .uploads_in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        signals.iter().all(|&signal| {
            // A free has no context to ask about a fault: it waits out its bound and leaks.
            crate::hsaco::await_below_one(
                FREE_WAIT,
                std::time::Duration::from_millis(2),
                // SAFETY: `signal` is live: the ring destroys a signal only after removing it from this list,
                // under the lock held here.
                || unsafe { (self.hsa.funcs.hsa_signal_load_relaxed)(signal) },
                // SAFETY: as above.
                || unsafe {
                    (self.hsa.funcs.hsa_signal_wait_scacquire)(
                        signal,
                        bindings::HSA_SIGNAL_CONDITION_LT,
                        1,
                        1_000_000,
                        bindings::HSA_WAIT_STATE_BLOCKED,
                    )
                },
                || false,
            ) == crate::hsaco::Awaited::Completed
        })
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl UploadRing {
        /// The pinned bytes the ring holds.
        pub(crate) fn pinned_bytes(&self) -> usize {
            self.slots
                .iter()
                .map(|slot| match slot {
                    SlotState::Ready(staging) | SlotState::Busy(staging, _) => staging.len(),
                    SlotState::Empty | SlotState::Abandoned => 0,
                })
                .sum()
        }
    }
}
