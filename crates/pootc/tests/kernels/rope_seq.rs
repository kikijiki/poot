//! Sequential RoPE (half-split rotary positional embedding) kernel for the AIE-core (XDNA2 NPU).
//!
//! LLaMA-style half-split rotation with precomputed cos/sin.
//!
//! Inputs: x[D] (vector to rotate, a query or key head), cs[2D] = [cos[0..D], sin[0..D]] for the token
//! position. Output: y[D].
//!
//! With half = D / 2 (same as poot's rope_partial composite):
//!   y[i]      = x[i]*cs[i]        - x[i+half]*cs[D+i],      for i in 0..half
//!   y[i+half] = x[i+half]*cs[i+half] + x[i]*cs[D+i+half],   for i in 0..half
//! i.e. out = x*cos + rotate_half(x)*sin, with rotate_half = [-x[half..D], x[0..half]].
//!
//! 2 S2MM inputs (x, cs) + 1 MM2S output (y) fit the IRON single-tile budget. No transcendentals;
//! stack_size=2048.
//!
//! Build with:
//!   POOT_AIE_LLC=~/refs/peano/peano-llc \
//!   cargo test -p pootc --test import_run \
//!     imported_rope_sequential_kernel_lowers_to_aie2p --release

#![crate_type = "lib"]

pub fn __poot_kernel_rope_seq(x: &[f32], cs: &[f32], y: &mut [f32]) {
    let d = y.len();
    let half = d / 2;
    // cs layout: [cos[0..d], sin[0..d]] (2*d elements). For i in 0..half:
    //   y[i]      = x[i]*cos[i]      - x[i+half]*sin[i]
    //   y[i+half] = x[i+half]*cos[i+half] + x[i]*sin[i+half]
    let mut i = 0usize;
    while i < half {
        let c0 = cs[i];
        let s0 = cs[d + i];
        let c1 = cs[i + half];
        let s1 = cs[d + i + half];
        y[i] = x[i] * c0 - x[i + half] * s0;
        y[i + half] = x[i + half] * c1 + x[i] * s1;
        i += 1;
    }
}
