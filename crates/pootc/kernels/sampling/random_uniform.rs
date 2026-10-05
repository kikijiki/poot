//! Exact uniform noise, `OpKind::RandomUniform` (card 551a, R472-007, deval.md section 8.1): one thread
//! per output element, no workgroup reduction. `seed` is `[rows]` I32 (u32 bits); `dims[0]` carries
//! `cols`; output is `[rows, cols]` F32. For row `r` (`i / cols`) and column `c` (`i % cols`):
//! `x = fmix32(seed[r] XOR (r*cols+c)*0x9E3779B9)` (murmur3's finalizer, matching
//! `poot_graph_ir::ops::sampling::hash32` and the CPU oracle bit for bit), `u = (2*(x>>9)+1) * 2^-24`.
//! `x >> 9` keeps the top 23 bits (`< 2^23`), so `2*(x>>9)+1` is an odd integer in `[1, 2^24-1]`: both the
//! `u32 -> f32` conversion and the power-of-two multiply are exact, giving `u` in `[2^-24, 1-2^-24]` on
//! every backend (tier 1, ADR-0101) - always nonzero and never 1, so a later `log(-log(u))` (the Gumbel
//! transform, an ordinary graph composition) stays finite.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_random_uniform(seed: &[i32], dims: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let cols = dims[0] as usize;
        let r = i / cols;
        let c = i % cols;
        let s = seed[r] as u32;
        let idx = (r * cols + c) as u32;
        let mut x = s ^ (idx * 0x9E3779B9u32);
        x = x ^ (x >> 16);
        x = x * 0x85EBCA6Bu32;
        x = x ^ (x >> 13);
        x = x * 0xC2B2AE35u32;
        x = x ^ (x >> 16);
        let mantissa = (x >> 9) as i32;
        let u = (2.0f32 * (mantissa as f32) + 1.0f32) * 5.9604645e-8f32; // 2^-24
        out[i] = u;
    }
}
