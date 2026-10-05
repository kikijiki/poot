//! The axis-0 embedding gather (kernelgen `gather_axis0_dt`) in ordinary Rust. One thread per output
//! element `i`: the output is `[num_rows, rest]` gathered from `data` (the `[table_rows, rest]` embedding
//! table) by `index` (one f32 token id per row); `out[i] = data[row*rest + r]` with `row = index[i/rest]`,
//! `r = i%rest`. No baked dims and no metadata buffer: `rest = out.len() / index.len()`, so this is a
//! 3-param `Plan::Compute` drop-in for kernelgen's `gather_axis0_dt(data, index, out)`. The index is read
//! as f32 and cast to usize (the token id), matching kernelgen. card 044.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_gather_axis0(data: &[f32], index: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let rest = out.len() / index.len();
        let s = i / rest;
        let r = i % rest;
        let row = index[s] as usize;
        out[i] = data[row * rest + r];
    }
}
