//! Batched coalesced decode GEMV with a BF16-resident weight: the `gemv_coalesced_bf16.rs`
//! tile-of-columns body over the flattened `idx` in `0..B*N`, `dims = [B]` as the metadata buffer.
//! Weight lanes hold packed checkpoint BF16 (two elements per `u32`, card 380 layout), widened
//! in-register; K = x.len()/B and N = out.len()/B.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    group_index, group_index_y, local_index, wg_read, wg_write, workgroup_barrier,
};
const WORKGROUP_SIZE: usize = 128; // GEMV_WIDTH
const GEMV_TILE: usize = 32; // output columns owned by one workgroup
const X_MAX: usize = 65535; // GEMV_X_MAX
pub fn __poot_kernel_gemv_batched_coalesced_bf16(
    x: &[f32],
    weight_words: &[u32],
    dims: &[u32],
    out: &mut [f32],
) {
    let ncols = out.len();
    let nwg = (ncols + GEMV_TILE - 1) / GEMV_TILE;
    let x_groups = if nwg < X_MAX { nwg } else { X_MAX };
    let idx0 = (group_index_y() * x_groups + group_index()) * GEMV_TILE;
    if idx0 < ncols {
        let batch = dims[0] as usize;
        let n_per = ncols / batch; // N
        let k = x.len() / batch; // K
        let lane = local_index();
        let strip = lane / GEMV_TILE;
        let col_in = lane % GEMV_TILE;
        let idx = idx0 + col_in;
        let valid = idx < ncols;
        let mut b = 0usize;
        let mut col = 0usize;
        if valid {
            b = idx / n_per;
            col = idx % n_per;
        }
        let mut partial = 0.0f32;
        let mut kk = strip;
        while kk < k {
            if valid {
                let element = kk * n_per + col;
                let shift = ((element % 2) * 16) as u32;
                let word = (weight_words[element / 2] >> shift) & 0xffff;
                partial = partial + x[b * k + kk] * f32::from_bits(word << 16);
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
            out[idx] = acc;
        }
    }
}
