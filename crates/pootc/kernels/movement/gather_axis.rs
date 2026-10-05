//! The general (non-axis-0) gather (kernelgen `gather_axis_dt`) in ordinary Rust. `data` is
//! `[outer.., axis_len, inner..]`, `index` holds `idx_numel` f32 positions, `out` is
//! `[outer.., idx_numel, inner..]`. One thread per output element `i`: split `i` into the contiguous
//! `inner` run (`inner_i = i % inner`), the gathered position (`gpos = (i/inner) % idx_numel`), and the
//! `outer` block (`outer_i = (i/inner) / idx_numel`); the axis row is `index[gpos]`, so
//! `out[i] = data[outer_i*axis_len*inner + axis_pos*inner + inner_i]`.
//!
//! `inner` and `axis_len` are not derivable from the buffer lengths (only `idx_numel = index.len()` is),
//! so they ride in a `[inner, axis_len]` metadata buffer (`Plan::ComputeMeta`). Branchless (one
//! `if i < len` guard, no loop / nested selection / barrier), so it imports on both backends. The index is
//! read as f32 and cast to usize, matching kernelgen. card 044.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_gather_axis(data: &[f32], index: &[f32], dims: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let inner = dims[0] as usize;
        let axis_len = dims[1] as usize;
        let idx_numel = index.len();
        let inner_i = i % inner;
        let mid = i / inner;
        let gpos = mid % idx_numel;
        let outer_i = mid / idx_numel;
        let axis_pos = index[gpos] as usize;
        let src = outer_i * (axis_len * inner) + axis_pos * inner + inner_i;
        out[i] = data[src];
    }
}
