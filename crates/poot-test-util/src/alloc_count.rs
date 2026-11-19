//! A counting global allocator for "no source-sized copy" checks.
//!
//! A test binary installs it once (`#[global_allocator]` can only be declared once per binary):
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: poot_test_util::alloc_count::CountingAllocator =
//!     poot_test_util::alloc_count::CountingAllocator;
//! ```
//!
//! Counting is enabled per thread, so a test that measures a body sees only the bytes that body requested
//! on its own thread, even while other tests run in parallel. A body that spawns threads is not fully
//! measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // Const-initialized cells without destructors, so reading them never allocates.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

/// The system allocator, counting the bytes requested on a thread while [`allocated_bytes`] measures it.
pub struct CountingAllocator;

fn note_allocation(size: usize) {
    if RECORDING.try_with(Cell::get).unwrap_or(false) {
        let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().saturating_add(size)));
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged and only counts the
// requested size, so `System`'s allocator contract carries over.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation(layout.size());
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc` contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation(layout.size());
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc_zeroed` contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::dealloc` contract.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation(new_size);
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::realloc` contract.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

/// Run `body` and return its output with the bytes it requested from the allocator on this thread.
/// A reallocation counts its full new size. Reads zero unless the binary installed [`CountingAllocator`].
pub fn allocated_bytes<T>(body: impl FnOnce() -> T) -> (T, usize) {
    BYTES.with(|bytes| bytes.set(0));
    RECORDING.with(|recording| recording.set(true));
    let output = body();
    RECORDING.with(|recording| recording.set(false));
    (output, BYTES.with(Cell::get))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// The measurement sees what the body requested, and nothing outside the measured body: the callers'
    /// checks are upper bounds, so a counter stuck at zero would pass them all.
    #[test]
    fn allocated_bytes_counts_the_measured_body_only() {
        let _outside = vec![0u8; 1 << 20];
        let (kept, bytes) = allocated_bytes(|| vec![0u8; 4096]);
        assert!(bytes >= 4096, "measured {bytes} bytes for a 4096-byte vec");
        assert!(
            bytes < 1 << 16,
            "measured {bytes} bytes: the outside vec leaked in"
        );
        drop(kept);
        let (_, none) = allocated_bytes(|| 1 + 1);
        assert_eq!(none, 0);
    }
}
