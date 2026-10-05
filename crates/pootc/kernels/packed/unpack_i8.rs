//! Unpack `len` signed-int8 codes from i32 words along the last axis (kernelgen `unpack_i8_dt`, the inverse
//! of `pack_i8`, spec 048). Input is i32 words `[.., W]` with `W = ceil(len/4)`; output is f32 codes
//! `[.., len]`. One thread per output code reads its word, extracts its byte, and sign-extends it via an
//! arithmetic right shift (`(word << (24 - 8*lane)) >> 24`, the unpack the Q8_0 dequant uses). `len` rides
//! in a `[len]` metadata buffer (`Plan::ComputeMeta`); `W = (len+3)/4` is derived in-kernel. Branchless.
//! card 044 / spec 055.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_unpack_i8(words: &[i32], dims: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let len = dims[0] as usize;
        let w = (len + 3) / 4; // words per row = ceil(len/4)
        let row = i / len;
        let c = i % len;
        let word = words[row * w + c / 4];
        let lane = c % 4;
        // extract byte `lane` and sign-extend: (word << (24 - 8*lane)) >> 24 (arithmetic shift).
        let shl = (24 - lane * 8) as i32;
        let signed = (word << shl) >> 24;
        out[i] = signed as f32;
    }
}
