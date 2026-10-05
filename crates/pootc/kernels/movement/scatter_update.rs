//! The paged-KV scatter-update (kernelgen `scatter_update_dt`, card 059): write `src` rows into a copy of
//! the `base` pool at the positions given by the inverse map `inv`. One thread per output element `i` over
//! the `[POOL, rest]` pool: `row = i/rest`, `j = inv[row]` (the source row for this slot, or -1 to keep
//! `base`). If `j >= 0` the element is `src[j*rest + r]`, else `base[i]`. No baked dims and no metadata
//! buffer: `rest = out.len() / inv.len()`, so this is a 4-param `Plan::Compute` drop-in for
//! `scatter_update_dt(base, src, inv, out)`. card 044.
//!
//! Branchless (the importer emits invalid SPIR-V for a selection nested in the bounds guard): the source
//! index is sign-masked so `src` is never read out of bounds for an empty slot, and an f32 mask selects
//! src vs base.
//!
//! 2-D grid fold (card 159): a flat grid overflows wgpu's 65535 gridDim.x cap once a batched shared KV
//! pool exceeds ~16.7M elements (the `workgroup_size[0]=256` bump in `plan_eqn` covers up to `65535*256`;
//! a 4-slot/1024-cap gemma4-dense local-layer pool is `4097*16*256 = 16,781,312`). Folded onto a 2-D grid
//! like `gemv.rs`'s `col = group_y*x_groups + group_x`; `x_groups` is derived from `out.len()`, so
//! `dispatch_grid` only launches the matching `(x_groups, y_groups)` grid (`poot-graph-plan`'s
//! `elementwise_grid`).

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, group_index_y, local_index};
const WORKGROUP_SIZE: usize = 256; // must match plan_eqn's non-custom-grid workgroup bump
const X_MAX: usize = 65535; // the wgpu workgroup X-dim cap

pub fn __poot_kernel_scatter_update(base: &[f32], src: &[f32], inv: &[f32], out: &mut [f32]) {
    let total_groups = (out.len() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let x_groups = if total_groups < X_MAX {
        total_groups
    } else {
        X_MAX
    };
    let group_id = group_index_y() * x_groups + group_index();
    let i = group_id * WORKGROUP_SIZE + local_index();
    if i < out.len() {
        let rest = out.len() / inv.len();
        let row = i / rest;
        let r = i % rest;
        let j = inv[row] as i32; // source row, or -1 for an empty slot
        let valid = (j >= 0) as usize; // 1 -> from src, 0 -> keep base
        // mask the source row to 0 when the slot is empty, so `src` is never indexed by a negative j.
        let ju = (j * valid as i32) as usize;
        let sidx = ju * rest + r;
        let m = valid as f32;
        out[i] = src[sidx] * m + base[i] * (1.0 - m);
    }
}
