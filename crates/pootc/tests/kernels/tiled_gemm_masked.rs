//! A workgroup-tiled GEMM for ragged shapes (card 044): `tiled_gemm.rs` plus bounds masking. Shapes need
//! not be multiples of TS: edge tiles stage out-of-bounds elements as 0 via branchless masks
//! (`* rmask * kmask`, the index clamped so the read stays in bounds), as kernelgen does. Branchless
//! because a `workgroup_barrier()` sits inside the K-tile loop and the mask must not introduce divergent
//! control flow before it. Only the final store is guarded by a real branch (past the last barrier).
//! Shapes ride in `dims = [M, K, N]`.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const TS: usize = 8; // tile edge; the workgroup is TS*TS = 64 lanes (the importer's default, one wave)
pub fn __poot_kernel_tiled_gemm_masked(a: &[f32], b: &[f32], dims: &[u32], out: &mut [f32]) {
    let m = dims[0] as usize;
    let k = dims[1] as usize;
    let n = dims[2] as usize;
    let alen = a.len();
    let blen = b.len();
    let lane = local_index();
    let tr = lane / TS;
    let tc = lane % TS;
    let group = group_index();
    let tiles_n = (n + TS - 1) / TS; // ceil(N/TS)
    let tile_row = (group / tiles_n) * TS;
    let tile_col = (group % tiles_n) * TS;
    let row = tile_row + tr;
    let col = tile_col + tc;
    let kt_count = (k + TS - 1) / TS; // ceil(K/TS), uniform across the workgroup
    let mut acc = 0.0f32;
    let mut kt = 0usize;
    while kt < kt_count {
        // stage A[row, kt*TS+tc] and B[kt*TS+tr, col]; out-of-bounds -> 0 via branchless mask (index
        // clamped so the read is always in bounds, value zeroed by the mask).
        let a_k = kt * TS + tc;
        let rmask = (row < m) as u32 as f32;
        let kmask = (a_k < k) as u32 as f32;
        let aidx = (row * k + a_k).min(alen - 1);
        wg_write(0, tr * TS + tc, a[aidx] * rmask * kmask);
        let b_k = kt * TS + tr;
        let bkmask = (b_k < k) as u32 as f32;
        let cmask = (col < n) as u32 as f32;
        let bidx = (b_k * n + col).min(blen - 1);
        wg_write(1, tr * TS + tc, b[bidx] * bkmask * cmask);
        workgroup_barrier();
        let mut i = 0usize;
        while i < TS {
            acc = acc + wg_read(0, tr * TS + i) * wg_read(1, i * TS + tc);
            i = i + 1;
        }
        workgroup_barrier();
        kt = kt + 1;
    }
    // store guard: a real branch, but past the last barrier, so divergence is harmless.
    if row < m {
        if col < n {
            out[row * n + col] = acc;
        }
    }
}
