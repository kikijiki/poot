//! Dynamic-offset update-slice (kernelgen `dyn_update_slice_dynamic_dt`), poot's contiguous KV-cache write:
//! copy `operand` to `out`, but overwrite the `extent`-long slice along `axis` starting at the runtime
//! offset `index[0]` with `update`. One thread per output element `i`: decode
//! `coord_axis = (i/axis_stride) % axis_dim`; outside `[idx, idx+extent)` copy `operand[i]`, else read the
//! `update` element (axis coordinate shifted by `-idx`). The per-dim strides ride in a metadata buffer
//! `[rank, axis, extent, (out_stride, out_dim, upd_stride) x rank]`; `idx` is read at runtime so the
//! capture stays valid across positions. Dispatched via `Plan::ComputeMeta`. card 044.
//!
//! Uses only `<` comparisons and an else-if (no `&&`/`||`), mirroring kernelgen's nested guards.
//!
//! 2-D grid fold as in `scatter_update.rs` (card 159): a flat grid overflows wgpu's 65535 gridDim.x cap
//! once a batched shared KV pool exceeds ~16.7M elements. `x_groups` is derived from `out.len()`;
//! `dispatch_grid` launches the matching `(x_groups, y_groups)` grid.

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, group_index_y, local_index};
const WORKGROUP_SIZE: usize = 256; // must match plan_eqn's non-custom-grid workgroup bump
const X_MAX: usize = 65535; // the wgpu workgroup X-dim cap

pub fn __poot_kernel_dyn_update_slice(
    operand: &[f32],
    update: &[f32],
    index: &[f32],
    dims: &[u32],
    out: &mut [f32],
) {
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
        let extent = dims[2] as usize;
        let idx = index[0] as usize;
        let axis_stride = dims[3 + axis * 3] as usize;
        let axis_dim = dims[3 + axis * 3 + 1] as usize;
        let coord_axis = (i / axis_stride) % axis_dim;
        // Update offset for this element: the axis dimension subtracts the runtime offset `idx`, other dims
        // use the coordinate directly. `acc` is only read on the in-slice store branch below, so out-of-slice
        // elements (where the axis subtraction wraps) discard a garbage acc.
        let mut acc = 0usize;
        let mut d = 0usize;
        while d < rank {
            let out_stride = dims[3 + d * 3] as usize;
            let out_dim = dims[3 + d * 3 + 1] as usize;
            let upd_stride = dims[3 + d * 3 + 2] as usize;
            let coord_d = (i / out_stride) % out_dim;
            let c = if d == axis { coord_d - idx } else { coord_d };
            acc = acc + c * upd_stride;
            d = d + 1;
        }
        // `rel = coord_axis - idx` wraps to a huge usize when coord_axis < idx (GPU arithmetic wraps), so
        // `rel < extent` is the full `idx <= coord_axis < idx+extent` in-slice test.
        if coord_axis - idx < extent {
            out[i] = update[acc];
        } else {
            out[i] = operand[i];
        }
    }
}
