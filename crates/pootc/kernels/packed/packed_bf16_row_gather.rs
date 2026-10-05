//! Card 381: the axis-0 embedding gather over a dense BF16 table, widened to F32 in the same kernel.
//!
//! One thread owns one output element `i`. The table arrives as packed `u32` lanes holding the checkpoint
//! bytes verbatim: two little-endian BF16 elements per lane, element `e` in lane `e / 2` at shift
//! `(e % 2) * 16` (the card 380 layout). This is the only portable BF16 representation on wgpu (the
//! SPIR-V emitter rejects `Ty::BF16` because RADV has no `VK_KHR_shader_bfloat16`). Widening happens
//! in-register, so no F32 copy of the table is materialized.
//!
//! Decode is `f32::from_bits(word << 16)` (card 370), exact.
//!
//! No baked dims and no metadata buffer, like `gather_axis0.rs`: `rest = out.len() / index.len()`, so this
//! is a 3-param `Plan::Compute`. The index is `i32`, not the f32 token convention of `gather_axis0.rs`:
//! the wgpu typed walk stores an I32 value as `Ty::I32`, and `OpKind::DenseRowGather` pins the operand
//! dtype to match.
//!
//! The element index is computed in elements and only then split into a lane and a shift. Indexing lanes
//! directly would be correct only for an even row width; with an odd `rest`, row `r` starts in the high
//! half of a lane whenever `r` is odd.
//!
//! The planner proves every index from the graph's shapes. The guards kept bound the two buffers a wrong
//! shape would over-read furthest: `out` and `table_words`. They are not total: `index[s]` is unguarded,
//! and `rest` would be zero with fewer output elements than index entries, but both need a binding the
//! planner's shape contract cannot produce.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_packed_bf16_row_gather(table_words: &[u32], index: &[i32], out: &mut [f32]) {
    let i = thread_index();
    // `index.len() > 0` rather than `!index.is_empty()`: the importer admits `len()` but rejects a call to
    // `is_empty`. An empty index with a non-empty output cannot be planned, so this only guards the
    // division below.
    if i < out.len() && index.len() > 0 {
        let rest = out.len() / index.len();
        let s = i / rest;
        let r = i % rest;
        let row = index[s] as usize;
        let element = row * rest + r;
        if element / 2 < table_words.len() {
            let shift = ((element % 2) * 16) as u32;
            let word = (table_words[element / 2] >> shift) & 0xffff;
            out[i] = f32::from_bits(word << 16);
        }
    }
}
