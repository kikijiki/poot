//! Coalesced decode GEMV with the fused bias epilogue (`out[col] += bias[col]`, the q/k/v projections).
//! Same body as `gemv_coalesced.rs` except the reducing lanes store `out[col] = acc + bias[col]`.
//! A workgroup owns `GEMV_TILE` contiguous output columns and reads `W[k, col0:col0+TILE]` contiguously
//! per k; K is split across `STRIPS = WORKGROUP_SIZE/GEMV_TILE` lane-groups with an LDS partial-sum fold.
//! Shape-generic: k = x.len(), N = out.len() (weight row-major [K, N]).
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, group_index_y, local_index, wg_read, wg_write, workgroup_barrier,
};
const WORKGROUP_SIZE: usize = 128; // GEMV_WIDTH
const GEMV_TILE: usize = 32; // output columns owned by one workgroup
const X_MAX: usize = 65535; // GEMV_X_MAX - the wgpu workgroup X-dim cap
pub fn __poot_kernel_gemv_coalesced_bias(x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32]) {
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
            out[col] = acc + bias[col];
        }
    }
}
