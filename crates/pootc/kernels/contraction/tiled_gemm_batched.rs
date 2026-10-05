//! Batched-weight thread-coarsened tiled GEMM (card 044), kernelgen's `tiled_gemm_dt` with `b_count>1`:
//! A[E,M,K] @ B[E,K,N] -> [E,M,N], each batch with its own weight (the per-expert MoE GEMM, and the rank-4
//! per-head prefill attention matmuls flattened to E = product of leading dims). Same coarsened TS x TS
//! tile and 128-slot A LDS as `tiled_gemm_coarsened.rs`, but each workgroup first decodes a batch index
//! from its global tile id and offsets the contiguous A/B/C buffers by per-batch bases. Ragged shapes via
//! the same branchless masking. Shapes ride in `dims = [E, M, K, N]`.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const TS: usize = 8; // tile edge; the workgroup is TS*TS = 64 lanes (the importer's default, one wave)
const LDS_SIZE: usize = 128; // 2*TS*TS - the A sub-tile holds two row-halves; the importer sizes LDS to this
pub fn __poot_kernel_tiled_gemm_batched(a: &[f32], b: &[f32], dims: &[u32], out: &mut [f32]) {
    let m = dims[1] as usize;
    let k = dims[2] as usize;
    let n = dims[3] as usize;
    let alen = a.len();
    let blen = b.len();
    let half = LDS_SIZE / 2;
    let lane = local_index();
    let tr = lane / TS;
    let tc = lane % TS;
    let tiles_n = (n + TS - 1) / TS;
    let tiles_m = (m + 2 * TS - 1) / (2 * TS);
    let tiles_per_batch = tiles_m * tiles_n;
    // decode this workgroup's batch + the tile within that batch from its global tile id.
    let group = group_index();
    let batch = group / tiles_per_batch;
    let within = group % tiles_per_batch;
    let trow = within / tiles_n;
    let tcol = within % tiles_n;
    let row0 = trow * (2 * TS) + tr;
    let row1 = row0 + TS;
    let col = tcol * TS + tc;
    // per-batch bases into the contiguous A[E,M,K] / B[E,K,N] / C[E,M,N] buffers.
    let a_base = batch * m * k;
    let b_base = batch * k * n;
    let c_base = batch * m * n;
    let kt_count = (k + TS - 1) / TS;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut kt = 0usize;
    while kt < kt_count {
        let a_k = kt * TS + tc;
        let kmask = (a_k < k) as u32 as f32;
        let r0mask = (row0 < m) as u32 as f32;
        let a0idx = (a_base + row0 * k + a_k).min(alen - 1);
        wg_write(0, tr * TS + tc, a[a0idx] * r0mask * kmask);
        let r1mask = (row1 < m) as u32 as f32;
        let a1idx = (a_base + row1 * k + a_k).min(alen - 1);
        wg_write(0, tr * TS + tc + half, a[a1idx] * r1mask * kmask);
        let b_k = kt * TS + tr;
        let bkmask = (b_k < k) as u32 as f32;
        let cmask = (col < n) as u32 as f32;
        let bidx = (b_base + b_k * n + col).min(blen - 1);
        wg_write(1, tr * TS + tc, b[bidx] * bkmask * cmask);
        workgroup_barrier();
        let mut i = 0usize;
        while i < TS {
            let bval = wg_read(1, i * TS + tc);
            acc0 = acc0 + wg_read(0, tr * TS + i) * bval;
            acc1 = acc1 + wg_read(0, tr * TS + i + half) * bval;
            i = i + 1;
        }
        workgroup_barrier();
        kt = kt + 1;
    }
    if col < n {
        if row0 < m {
            out[c_base + row0 * n + col] = acc0;
        }
        if row1 < m {
            out[c_base + row1 * n + col] = acc1;
        }
    }
}
