//! Thread-local allocation byte counter for behavioral "no source-sized copy" checks.
//!
//! The unit-test binary installs a counting global allocator. Counting is enabled per thread, so a test that
//! measures a body sees only the bytes that body requested on its own thread, even while other tests run in
//! parallel. A body that spawns threads is not fully measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // Const-initialized cells without destructors, so reading them never allocates.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

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
/// A reallocation counts its full new size.
pub(super) fn allocated_bytes<T>(body: impl FnOnce() -> T) -> (T, usize) {
    BYTES.with(|bytes| bytes.set(0));
    RECORDING.with(|recording| recording.set(true));
    let output = body();
    RECORDING.with(|recording| recording.set(false));
    (output, BYTES.with(Cell::get))
}
