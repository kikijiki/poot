//! Card 449 D1: the axis-0 embedding gather over a dense E4M3FN table, widened to F32 in the same kernel.
//!
//! One thread owns one output element `i`. The table arrives as packed `u32` lanes holding the checkpoint
//! bytes verbatim: four little-endian E4M3FN elements per lane, laid out **per row** the way
//! `poot_eval::fp8::PortableE4m3FnLayout` packs them and the way `poot_gpu`'s `BoundValueKind::E4m3Fn`
//! upload writes them - `words_per_row = row_width.div_ceil(4)`, so element `c` of row `r` lives in lane
//! `r * words_per_row + c / 4` at shift `(c % 4) * 8`. The per-row rounding is why the lane comes from
//! `words_per_row` and not from a flat `element / 4`: a row whose width is not a multiple of four starts
//! the next row in the *next* lane, not mid-lane. This is the only portable E4M3FN read on wgpu (the
//! SPIR-V emitter has no 8-bit float scalar type).
//!
//! Decode matches `poot_quant::scalar::e4m3fn_to_f32` exactly: subnormal `m * 2^-9`, finite
//! `+-2^(e-7) * (1 + m/8)`, and the canonical positive NaN for `e == 15 && m == 7`, so a device row is
//! bit-identical to the CPU oracle's row. The NaN test is spelled as two nested `if`s rather than one
//! `e == 15 && m == 7` condition: a two-operand `&&` in a condition at this depth makes `pootc`'s llc
//! lowering spin forever (observed on Card 449 D1 - `pootc` writes the `.kir.json`, then burns 100%
//! CPU and never emits `.spv`, with no child process. `row >= 0 && row < outer` one level up does not
//! trigger it, so the nesting is what matters).
//!
//! No baked dims and no metadata buffer, like `gather_axis0.rs`: `row_width = out.len() / index.len()`,
//! so this is a 3-param `Plan::Compute`. The index is `i32`, not the f32 token convention of
//! `gather_axis0.rs`: `OpKind::DenseRowGather` pins an authoritative exact-I32 index (Card 405's
//! `gather_exact_i32`), which is what keeps n-gram ids above `2^24` from rounding through f32.
//!
//! The guards bound the two buffers a wrong shape would over-read furthest: `out` and `table_words`
//! (`outer = table_words.len() / words_per_row` is the row count the layout implies, so an in-range
//! `row` plus an in-range `column` can never index past the buffer). `index[s]` is guarded by that row
//! count rather than proved, because the index is computed on the device like every other wgpu `Gather`.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_packed_e4m3_row_gather(table_words: &[u32], index: &[i32], out: &mut [f32]) {
    let i = thread_index();
    // `index.len() > 0` rather than `!index.is_empty()`: the importer admits `len()` but rejects a call to
    // `is_empty`. `row_width > 0` keeps the division below defined for a shape the planner should not
    // produce (an output shorter than its own index).
    if i < out.len() && index.len() > 0 {
        let row_width = out.len() / index.len();
        if row_width > 0 {
            let s = i / row_width;
            let column = i % row_width;
            let words_per_row = (row_width + 3) / 4;
            let outer = table_words.len() / words_per_row;
            let row = index[s];
            if row >= 0 && (row as usize) < outer {
                let word = (row as usize) * words_per_row + column / 4;
                let shift = ((column % 4) * 8) as u32;
                let byte = (table_words[word] >> shift) & 0xff;
                let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
                let exponent = (byte >> 3) & 0x0f;
                let mantissa = byte & 0x07;
                // One nested-if value chain feeding a single store. `e == 15 && m == 7` is spelled
                // as two nested conditions because a two-operand `&&` at this depth makes `pootc`'s
                // llc lowering spin forever (see the module doc). The finite expression appears in
                // both non-subnormal arms for that reason; it is deliberately identical.
                let value = if exponent == 15 {
                    if mantissa == 7 {
                        f32::from_bits(0x7fc00000)
                    } else {
                        sign * ((1.0 + mantissa as f32 * 0.125)
                            * f32::from_bits((exponent + 120) << 23))
                    }
                } else if exponent == 0 {
                    sign * (mantissa as f32 * 0.001953125)
                } else {
                    sign * ((1.0 + mantissa as f32 * 0.125)
                        * f32::from_bits((exponent + 120) << 23))
                };
                out[i] = value;
            }
        }
    }
}
