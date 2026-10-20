//! The upload-ring policy the device runtimes share (dmap.md section 6.2): a device write is cut into
//! chunks of at most [`UPLOAD_CHUNK_BYTES`], each staged through one slot of a ring of [`RING_SLOTS`]
//! pinned host buffers, so the host fills slot `k + 1` while the copy engine drains slot `k`. This is
//! llama.cpp's no-mmap loading scheme (4 pinned buffers of 64 MiB, async copies and events).
//!
//! What stays per runtime is the mechanism: the pinned allocation, the completion signal (an HSA signal, a
//! CUDA event) and the copy call. What is here is the policy that must not drift between them: how many
//! slots, how large, how a write is cut, which slot comes next, and which in-flight copies a write must wait
//! for.

use std::ops::Range;

/// Slots in the ring.
pub const RING_SLOTS: usize = 4;

/// Bytes of one slot: the largest chunk of a write.
pub const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

/// The smallest slot a ring pins.
const MIN_SLOT_BYTES: usize = 64 << 10;

/// The pinned size of a slot asked to hold `need` bytes: the next power of two, at least 64 KiB, at most one
/// chunk. A small write pins a small buffer, not [`UPLOAD_CHUNK_BYTES`].
pub fn slot_bytes(need: usize) -> usize {
    need.next_power_of_two()
        .clamp(MIN_SLOT_BYTES, UPLOAD_CHUNK_BYTES)
}

/// The chunks a write of `len` bytes is cut into: consecutive ranges of `0..len`, each at most
/// [`UPLOAD_CHUNK_BYTES`]. An empty write has none.
pub fn chunks(len: usize) -> impl Iterator<Item = Range<usize>> {
    (0..len)
        .step_by(UPLOAD_CHUNK_BYTES)
        .map(move |start| start..len.min(start + UPLOAD_CHUNK_BYTES))
}

/// Whether two byte ranges share a byte: the test for a write that must wait for an in-flight copy into
/// the same device memory.
pub fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Which slot the next chunk uses: round-robin over [`RING_SLOTS`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingCursor {
    next: usize,
}

impl RingCursor {
    /// The slot the next chunk uses.
    pub fn index(self) -> usize {
        self.next
    }

    /// Move past the current slot, wrapping to the first after the last. A runtime calls this once the
    /// slot is ready for its chunk, so a failed allocation or wait leaves the cursor where it was and the
    /// retry reuses the same slot.
    pub fn advance(&mut self) {
        self.next = (self.next + 1) % RING_SLOTS;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_size_is_a_power_of_two_between_the_floor_and_one_chunk() {
        assert_eq!(slot_bytes(0), MIN_SLOT_BYTES);
        assert_eq!(slot_bytes(1), MIN_SLOT_BYTES);
        assert_eq!(slot_bytes(MIN_SLOT_BYTES + 1), 2 * MIN_SLOT_BYTES);
        assert_eq!(slot_bytes((1 << 20) + 5), 2 << 20);
        assert_eq!(slot_bytes(UPLOAD_CHUNK_BYTES), UPLOAD_CHUNK_BYTES);
        assert_eq!(slot_bytes(UPLOAD_CHUNK_BYTES - 1), UPLOAD_CHUNK_BYTES);
    }

    #[test]
    fn a_write_is_cut_into_consecutive_chunks_that_cover_it_exactly() {
        assert_eq!(chunks(0).count(), 0);
        assert_eq!(chunks(7).next(), Some(0..7));
        assert_eq!(chunks(7).count(), 1);
        assert_eq!(
            chunks(UPLOAD_CHUNK_BYTES).next(),
            Some(0..UPLOAD_CHUNK_BYTES)
        );
        assert_eq!(chunks(UPLOAD_CHUNK_BYTES).count(), 1);
        let len = 2 * UPLOAD_CHUNK_BYTES + 3;
        assert_eq!(
            chunks(len).collect::<Vec<_>>(),
            [
                0..UPLOAD_CHUNK_BYTES,
                UPLOAD_CHUNK_BYTES..2 * UPLOAD_CHUNK_BYTES,
                2 * UPLOAD_CHUNK_BYTES..len
            ]
        );
    }

    #[test]
    fn touching_ranges_do_not_overlap_and_intersecting_ones_do() {
        assert!(!overlaps(&(0..4), &(4..8)));
        assert!(!overlaps(&(4..8), &(0..4)));
        assert!(overlaps(&(0..5), &(4..8)));
        assert!(overlaps(&(2..3), &(0..8)));
    }

    #[test]
    fn the_cursor_wraps_after_the_last_slot() {
        let mut cursor = RingCursor::default();
        let seen: Vec<usize> = (0..RING_SLOTS + 1)
            .map(|_| {
                let index = cursor.index();
                cursor.advance();
                index
            })
            .collect();
        assert_eq!(seen, [0, 1, 2, 3, 0]);
    }
}
