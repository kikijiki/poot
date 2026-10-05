//! Card 1011: the `Cast(BF16 -> F32)` of a BF16 const, read from the packed `u32` lanes the const is stored in.
//!
//! One thread owns one output element `i`. The source arrives as packed `u32` lanes holding the checkpoint
//! bytes verbatim: two little-endian BF16 elements per lane, element `e` in lane `e / 2` at shift
//! `(e % 2) * 16` (the card 380 layout, the one `packed_bf16_row_gather.rs` reads). This is the only portable
//! BF16 representation on wgpu (the SPIR-V emitter rejects `Ty::BF16`), and every backend reads a BF16 const
//! whose readers are all packed readers through it, so a norm scale or a bias is never widened on upload.
//!
//! Decode is `f32::from_bits(word << 16)`, exact.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_packed_bf16_to_f32(words: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() && i / 2 < words.len() {
        let shift = ((i % 2) * 16) as u32;
        let word = (words[i / 2] >> shift) & 0xffff;
        out[i] = f32::from_bits(word << 16);
    }
}
