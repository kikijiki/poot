//! The axis-0 scatter (kernelgen `scatter_axis0_dt`, the write-side inverse of the axis-0 gather) in
//! ordinary Rust. One thread per source element `i`: `out[index[i/rest]*rest + i%rest] = src[i]`, where
//! `index` carries one f32 destination row per source row (a permutation, the MoE expert-routing
//! inversion). No baked dims and no metadata buffer: `rest = src.len() / index.len()`, so this is a
//! 3-param `Plan::Compute` drop-in for `scatter_axis0_dt(src, index, out)`. The index is read as f32 and
//! cast to usize, matching kernelgen. card 044.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_scatter_axis0(src: &[f32], index: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < src.len() {
        let rest = src.len() / index.len();
        let s = i / rest;
        let r = i % rest;
        let row = index[s] as usize;
        out[row * rest + r] = src[i];
    }
}
