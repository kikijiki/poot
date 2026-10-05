//! The shape-generic index-remap copy (kernelgen `index_remap_copy`, the primitive behind transpose /
//! slice / broadcast) in ordinary Rust. One thread per output element `i`: `out[i] = input[src]` with
//! `src = src_base + sum_d ((i / out_stride[d]) % out_dim[d]) * src_term[d]`. The per-dim strides ride in
//! a metadata buffer, so one imported Body serves transpose (src_term = in_stride[perm[d]]), slice
//! (src_term = in_stride[d], src_base = start*in_stride[axis]), and broadcast (src_term = 0). Layout:
//! `[rank, src_base, (out_stride, out_dim, src_term) x rank]`. Dispatched via `Plan::ComputeMeta`. card 044.
//!
//! 2-D grid fold as in `scatter_update.rs`/`dyn_update_slice.rs` (card 159): a flat grid overflows wgpu's
//! 65535 gridDim.x cap once an output exceeds `65535*256` elements (gemma4-dense batched decode's
//! `repeat_kv` broadcast and K^T transpose are each `1*32*512*1024` at cap=1024). `x_groups` is derived
//! from `out.len()`, so `dispatch_grid` only launches the matching `(x_groups, y_groups)` grid.

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, group_index_y, local_index};
const WORKGROUP_SIZE: usize = 256; // must match plan_eqn's non-custom-grid workgroup bump
const X_MAX: usize = 65535; // the wgpu workgroup X-dim cap

pub fn __poot_kernel_index_remap(input: &[f32], dims: &[u32], out: &mut [f32]) {
    let total_groups = (out.len() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let x_groups = if total_groups < X_MAX {
        total_groups
    } else {
        X_MAX
    };
    let group_id = group_index_y() * x_groups + group_index();
    let i = group_id * WORKGROUP_SIZE + local_index();
    if i < out.len() {
        let rank = dims[0] as usize;
        let mut src = dims[1] as usize;
        let mut d = 0usize;
        while d < rank {
            let out_stride = dims[2 + d * 3] as usize;
            let out_dim = dims[2 + d * 3 + 1] as usize;
            let src_term = dims[2 + d * 3 + 2] as usize;
            let coord = (i / out_stride) % out_dim;
            src = src + coord * src_term;
            d = d + 1;
        }
        out[i] = input[src];
    }
}
