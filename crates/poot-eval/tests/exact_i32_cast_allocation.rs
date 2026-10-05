//! Behavioral allocation check for the exact-I32 selected cast.
//!
//! This binary installs a counting global allocator. It records every allocation at or above
//! `LARGE_BYTES` made on the test thread while one selected cast runs. The cast path must make exactly
//! one allocation for the published f32 result and no other allocation of comparable size, so a `Vec`
//! staging buffer, a `to_vec` copy, or a collected temporary shows up as an extra record whatever it is
//! spelled like in source.
//!
//! Limits: the allocator and its record table are process-global, so this binary holds exactly one
//! recording test; a second recording test would share `LARGE_COUNT` and `LARGE_RECORDS`. Recording is
//! enabled per thread, so allocations made on any other thread are not recorded. The check therefore
//! covers the evaluator's single-threaded cast path only.

use poot_tensor::DType;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use poot_eval::cast_authority::{ExactI32CastBounds, ExactI32CastRole};
use poot_eval::{EvalBudget, EvalOptions, ExactI32TensorView, Value, eval};
use poot_graph_ir::Builder;

/// Selected words. Both the Gather result and the f32 cast result are `SELECTED * 4` bytes.
const SELECTED: usize = 4096;
/// Anything this large is comparable to a selected buffer; smaller bookkeeping is ignored.
const LARGE_BYTES: usize = SELECTED;
const RECORD_SLOTS: usize = 16;

struct LargeRecord {
    pointer: AtomicUsize,
    size: AtomicUsize,
}

impl LargeRecord {
    const fn new() -> Self {
        Self {
            pointer: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
        }
    }
}

static LARGE_COUNT: AtomicUsize = AtomicUsize::new(0);
static LARGE_RECORDS: [LargeRecord; RECORD_SLOTS] = [const { LargeRecord::new() }; RECORD_SLOTS];

thread_local! {
    // A const-initialized `Cell<bool>` has no destructor, so reading it never allocates.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_large(pointer: *mut u8, size: usize) {
    if pointer.is_null() || size < LARGE_BYTES {
        return;
    }
    if !RECORDING.try_with(Cell::get).unwrap_or(false) {
        return;
    }
    let index = LARGE_COUNT.fetch_add(1, Ordering::SeqCst);
    if let Some(record) = LARGE_RECORDS.get(index) {
        record.pointer.store(pointer as usize, Ordering::SeqCst);
        record.size.store(size, Ordering::SeqCst);
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged and only observes
// the returned pointer, so `System`'s allocator contract carries over.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc` contract.
        let pointer = unsafe { System.alloc(layout) };
        record_large(pointer, layout.size());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc_zeroed` contract.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record_large(pointer, layout.size());
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::dealloc` contract.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::realloc` contract.
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        record_large(grown, new_size);
        grown
    }
}

/// Run `body` with large-allocation recording enabled on this thread and return what it recorded.
fn record_large_allocations<T>(body: impl FnOnce() -> T) -> (T, Vec<(usize, usize)>) {
    LARGE_COUNT.store(0, Ordering::SeqCst);
    RECORDING.with(|recording| recording.set(true));
    let output = body();
    RECORDING.with(|recording| recording.set(false));
    let count = LARGE_COUNT.load(Ordering::SeqCst);
    assert!(
        count <= RECORD_SLOTS,
        "{count} large allocations overflowed the record table"
    );
    let records = LARGE_RECORDS[..count]
        .iter()
        .map(|record| {
            (
                record.pointer.load(Ordering::SeqCst),
                record.size.load(Ordering::SeqCst),
            )
        })
        .collect();
    (output, records)
}

#[test]
fn selected_cast_allocates_only_its_published_result() {
    let builder = Builder::new();
    let table =
        poot_test_util::graph_fixtures::i32_constant(&builder, "route.table", vec![SELECTED])
            .unwrap();
    let index =
        poot_test_util::graph_fixtures::i32_constant(&builder, "route.indices", vec![SELECTED])
            .unwrap();
    let selected = builder.gather(table, 0, index);
    let cast = builder.cast(selected, DType::F32);
    let graph = builder.finish(cast);

    let table_words: Arc<[i32]> = (0..SELECTED).map(|row| (row % 3) as i32).collect();
    let index_words: Arc<[i32]> = (0..SELECTED)
        .map(|row| (SELECTED - 1 - row) as i32)
        .collect();
    let inputs = HashMap::from([
        (
            table.id,
            Value::from(ExactI32TensorView::try_from_words(vec![SELECTED], table_words).unwrap()),
        ),
        (
            index.id,
            Value::from(ExactI32TensorView::try_from_words(vec![SELECTED], index_words).unwrap()),
        ),
    ]);
    let mut authority = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: SELECTED,
            expert_count: 3,
        }))
    };

    let (result, records) = record_large_allocations(|| {
        eval(
            &graph,
            &inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut authority),
        )
    });

    let Value::Host(tensor) = result.unwrap().output else {
        panic!("selected cast must publish a dense f32 result")
    };
    assert_eq!(tensor.as_f32().unwrap().len(), SELECTED);
    assert_eq!(tensor.as_f32().unwrap()[0], ((SELECTED - 1) % 3) as f32);
    assert_eq!(tensor.as_f32().unwrap()[SELECTED - 1], 0.0);

    // Exactly two comparable allocations: the Gather result and the published cast result (554c: the word functions fill an Arc in place, so there is no extra Vec-to-Arc
    // copy to show up as a third allocation here).
    let result_bytes = SELECTED * std::mem::size_of::<f32>();
    assert_eq!(records.len(), 2, "large allocations: {records:?}");
    for &(_, size) in &records {
        assert!(
            (result_bytes..2 * result_bytes).contains(&size),
            "unexpected large allocation size {size}: {records:?}"
        );
    }
    let data = tensor.as_f32().unwrap().as_ptr() as usize;
    let holding_result = records
        .iter()
        .filter(|&&(pointer, size)| (pointer..pointer + size).contains(&data))
        .count();
    assert_eq!(
        holding_result, 1,
        "the published f32 data must live in exactly one recorded allocation: {records:?}"
    );
}
