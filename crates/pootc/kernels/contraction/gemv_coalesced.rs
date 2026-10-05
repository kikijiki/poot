//! Coalesced decode GEMV: a workgroup owns a tile of output columns and reads
//! `W[k, col0:col0+TILE]` contiguously for each contraction index k (one lane per column of the tile).
//! The old `gemv.rs` gives one workgroup per column, so adjacent lanes stride the weight by `ncols`
//! (uncoalesced, ~20 GB/s on the olmo2 decode). Here adjacent lanes hit adjacent columns: at each k the
//! workgroup reads a contiguous `TILE`-float run. K is split across `STRIPS` lane-groups (one group per
//! wave when TILE matches the wave width); each strip accumulates a partial dot in registers, writes it
//! to LDS, a barrier, then strip 0 folds the `STRIPS` partials into `out[col]`. Same 2-D grid contract as
//! `gemv.rs` (`col0 = (group_y * x_groups + group_x) * TILE`, `x_groups` derived from `out.len()`), so
//! `dispatch_grid`'s tiled arm and this body stay bit-for-bit in sync. Shape-generic:
//! k = x.len(), N = out.len() (weight row-major [K, N]).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, group_index_y, local_index, wg_read, wg_write, workgroup_barrier,
};
const WORKGROUP_SIZE: usize = 128; // GEMV_WIDTH
const GEMV_TILE: usize = 32; // output columns owned by one workgroup
const X_MAX: usize = 65535; // GEMV_X_MAX - the wgpu workgroup X-dim cap
pub fn __poot_kernel_gemv_coalesced(x: &[f32], weight: &[f32], out: &mut [f32]) {
    let ncols = out.len();
    let nwg = (ncols + GEMV_TILE - 1) / GEMV_TILE;
    let x_groups = if nwg < X_MAX { nwg } else { X_MAX };
    let col0 = (group_index_y() * x_groups + group_index()) * GEMV_TILE;
    if col0 < ncols {
        let lane = local_index();
        let strip = lane / GEMV_TILE;
        let col_in = lane % GEMV_TILE;
        let col = col0 + col_in;
        let k = x.len();
        let valid = col < ncols;
        let mut partial = 0.0f32;
        let mut kk = strip;
        while kk < k {
            if valid {
                partial = partial + x[kk] * weight[kk * ncols + col];
            }
            kk = kk + WORKGROUP_SIZE / GEMV_TILE;
        }
        wg_write(0, lane, partial);
        workgroup_barrier();
        if strip == 0 && valid {
            let mut acc = 0.0f32;
            let mut s = 0usize;
            while s < WORKGROUP_SIZE / GEMV_TILE {
                acc = acc + wg_read(0, s * GEMV_TILE + col_in);
                s = s + 1;
            }
            out[col] = acc;
        }
    }
}
