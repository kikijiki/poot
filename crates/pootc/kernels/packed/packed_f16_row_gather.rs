//! POOT-1477: the axis-0 embedding gather over a dense F16 table, widened to F32 in the same kernel.
//!
//! One thread owns one output element `i`. The table arrives as packed `u32` lanes: two little-endian binary16
//! elements per lane, element `e` in lane `e / 2` at shift `(e % 2) * 16`, the layout of every dense F16 weight
//! (`StorageKind::F16Packed`). It is the lane a BF16 embedding the driver stores as f16 lands in: the bind
//! rounds each bf16 element to binary16 once, so this gather and a tied head's contraction read one buffer.
//!
//! Decode matches `poot_quant::scalar::f16_to_f32` over binary16's finite domain, which is the domain the bind
//! admits (it refuses an infinity, a NaN and a value that rounds past 65504): a subnormal is `m * 2^-24`, a
//! normal `+-2^(e-15) * (1 + m/1024)`, both exact in f32. The sign is a multiplication rather than a bit
//! operation so the decode is one nested-if value chain feeding a single store, like the E4M3FN gather.
//!
//! No baked dims and no metadata buffer: `rest = out.len() / index.len()`, so this is a 3-param single-dispatch
//! plan. The index is `i32` (`OpKind::DenseRowGather` pins the exact-I32 index). The element index is computed
//! in elements and only then split into a lane and a shift, because a row of odd width starts in the high half
//! of a lane.
//!
//! The planner proves every index from the graph's shapes. The guards bound the two buffers a wrong shape would
//! over-read furthest: `out` and `table_words`.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index_x;

pub fn __poot_kernel_packed_f16_row_gather(table_words: &[u32], index: &[i32], out: &mut [f32]) {
    let i = thread_index_x();
    // `index.len() > 0` rather than `!index.is_empty()`: the importer admits `len()` but rejects a call to
    // `is_empty`. An empty index with a non-empty output cannot be planned, so this only guards the division.
    if i < out.len() && index.len() > 0 {
        let rest = out.len() / index.len();
        let s = i / rest;
        let r = i % rest;
        let row = index[s] as usize;
        let element = row * rest + r;
        if element / 2 < table_words.len() {
            let shift = ((element % 2) * 16) as u32;
            let half = (table_words[element / 2] >> shift) & 0xffff;
            let sign = if half & 0x8000 == 0 { 1.0 } else { -1.0 };
            let exponent = (half >> 10) & 0x1f;
            let mantissa = half & 0x3ff;
            let value = if exponent == 0 {
                sign * (mantissa as f32 * 0.000000059604645)
            } else {
                sign * f32::from_bits(((exponent + 112) << 23) | (mantissa << 13))
            };
            out[i] = value;
        }
    }
}
