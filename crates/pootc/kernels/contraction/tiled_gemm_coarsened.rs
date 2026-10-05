//! Thread-coarsened workgroup-tiled GEMM (card 044), the structure of kernelgen's `tiled_gemm_dt`: each of
//! the TS*TS = 64 lanes computes two output rows (row0 and row1 = row0 + TS) of a 2*TS x TS tile, so the
//! workgroup stays one wave while each staged B (weight) element is reused for two rows. The A sub-tile is
//! 2*TS*TS = 128 slots (the two row-halves at LDS offset `LDS_SIZE/2`), larger than the lane count, so the
//! kernel declares `const LDS_SIZE: usize = 128;` and the importer sizes its LDS arrays to that. Ragged
//! shapes via the same branchless masking as `tiled_gemm_masked.rs`. Shapes ride in `dims = [M, K, N]`.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const TS: usize = 8; // tile edge; the workgroup is TS*TS = 64 lanes (the importer's default, one wave)
const LDS_SIZE: usize = 128; // 2*TS*TS - the A sub-tile holds two row-halves; the importer sizes LDS to this
pub fn __poot_kernel_tiled_gemm_coarsened(a: &[f32], b: &[f32], dims: &[u32], out: &mut [f32]) {
    let m = dims[0] as usize;
    let k = dims[1] as usize;
    let n = dims[2] as usize;
    // dims[3] = tile offset: card-096 chunking dispatches a large GEMM as several sub-2^15-tile chunks, each
    // covering a disjoint [offset, offset+groups) range of the tile grid. 0 for a single dispatch.
    let offset = dims[3] as usize;
    let alen = a.len();
    let blen = b.len();
    let half = LDS_SIZE / 2; // = TS*TS; the A tile's row1 half starts here
    let lane = local_index();
    let tr = lane / TS;
    let tc = lane % TS;
    let group = group_index() + offset;
    let tiles_n = (n + TS - 1) / TS;
    let trow = group / tiles_n;
    let tcol = group % tiles_n;
    let row0 = trow * (2 * TS) + tr; // this lane's two output rows
    let row1 = row0 + TS;
    let col = tcol * TS + tc;
    let kt_count = (k + TS - 1) / TS;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut kt = 0usize;
    while kt < kt_count {
        let a_k = kt * TS + tc;
        let kmask = (a_k < k) as u32 as f32;
        // stage both A row-halves (row0 at slot l, row1 at slot l + half), branchless-masked + clamped.
        let r0mask = (row0 < m) as u32 as f32;
        let a0idx = (row0 * k + a_k).min(alen - 1);
        wg_write(0, tr * TS + tc, a[a0idx] * r0mask * kmask);
        let r1mask = (row1 < m) as u32 as f32;
        let a1idx = (row1 * k + a_k).min(alen - 1);
        wg_write(0, tr * TS + tc + half, a[a1idx] * r1mask * kmask);
        // stage the shared B sub-tile.
        let b_k = kt * TS + tr;
        let bkmask = (b_k < k) as u32 as f32;
        let cmask = (col < n) as u32 as f32;
        let bidx = (b_k * n + col).min(blen - 1);
        wg_write(1, tr * TS + tc, b[bidx] * bkmask * cmask);
        workgroup_barrier();
        // accumulate both rows against the shared B column (branchless inner loop).
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
    // two guarded stores (past the last barrier, so divergence is harmless).
    if col < n {
        if row0 < m {
            out[row0 * n + col] = acc0;
        }
        if row1 < m {
            out[row1 * n + col] = acc1;
        }
    }
}
