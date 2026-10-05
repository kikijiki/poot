//! Two-input concat along `axis` (kernelgen `concat2_dt`), used by RoPE rotate_half, the KV-cache append,
//! and the MoE row-assembly reduce. One thread per output element `i`: decode `coord_axis`; below
//! `a_axis_len` the element comes from `a` at the same coords, else from `b` with the axis coordinate
//! shifted by `-a_axis_len`. The per-dim strides ride in a metadata buffer
//! `[rank, axis, a_axis_len, (out_stride, out_dim, a_stride, b_stride) x rank]`, dispatched via
//! `Plan::ComputeMeta`. card 044.
//!
//! Branchless (the importer used to emit invalid SPIR-V for a selection nested in the bounds guard): both
//! flat indices are computed every time and index-masked so neither read goes out of bounds, then an f32
//! mask selects a vs b.
//!
//! 2-D grid fold as in `scatter_update.rs`/`index_remap.rs`: a flat grid overflows wgpu's 65535 gridDim.x
//! cap once a batched shared-pool KV gather's output exceeds `65535*256` elements (card 159; hit by
//! gemma4-dense batched decode, `GridCap([65536,1,1])`). `x_groups` is derived from `out.len()`, so
//! `dispatch_grid` only launches the matching `(x_groups, y_groups)` grid.

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, group_index_y, local_index};
const WORKGROUP_SIZE: usize = 256; // must match plan_eqn's non-custom-grid workgroup bump
const X_MAX: usize = 65535; // the wgpu workgroup X-dim cap

pub fn __poot_kernel_concat2(a: &[f32], b: &[f32], dims: &[u32], out: &mut [f32]) {
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
        let axis = dims[1] as usize;
        let a_axis_len = dims[2] as usize;
        let axis_stride = dims[3 + axis * 4] as usize;
        let axis_dim = dims[3 + axis * 4 + 1] as usize;
        let coord_axis = (i / axis_stride) % axis_dim;
        let in_a = (coord_axis < a_axis_len) as usize; // 1 -> from a, 0 -> from b
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        let mut d = 0usize;
        while d < rank {
            let out_stride = dims[3 + d * 4] as usize;
            let out_dim = dims[3 + d * 4 + 1] as usize;
            let a_stride = dims[3 + d * 4 + 2] as usize;
            let b_stride = dims[3 + d * 4 + 3] as usize;
            let coord_d = (i / out_stride) % out_dim;
            a_flat = a_flat + coord_d * a_stride;
            // b's coordinate on the concat axis is coord_d - a_axis_len (wraps harmlessly when in_a, the b
            // read is masked off then); elsewhere it is coord_d.
            let is_axis = (d == axis) as usize;
            let b_coord = coord_d - is_axis * a_axis_len;
            b_flat = b_flat + b_coord * b_stride;
            d = d + 1;
        }
        let m = in_a as f32;
        out[i] = a[a_flat * in_a] * m + b[b_flat * (1 - in_a)] * (1.0 - m);
    }
}
