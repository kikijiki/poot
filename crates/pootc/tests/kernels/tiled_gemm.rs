//! A workgroup-tiled GEMM (card 044), the essential shape of kernelgen's `tiled_gemm_dt`: each workgroup
//! computes one TS x TS output tile of C[M,N] = A[M,K] @ B[K,N], staging A and B sub-tiles through two
//! workgroup-local (LDS) arrays with a barrier inside the K-tile loop. Exercises an in-loop barrier plus
//! 2-D multi-array LDS tiling. Tile-aligned only (M, K, N multiples of TS), so the loop body is branchless.
//! Shapes ride in `dims = [M, K, N]`.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, wg_read, wg_write, workgroup_barrier};
const TS: usize = 8; // tile edge; the workgroup is TS*TS = 64 lanes (the importer's default, one wave)
pub fn __poot_kernel_tiled_gemm(a: &[f32], b: &[f32], dims: &[u32], out: &mut [f32]) {
    // dims = [M, K, N]. M is unused while tile-aligned (the grid covers exactly M/TS row tiles).
    let k = dims[1] as usize;
    let n = dims[2] as usize;
    // lane -> position within the workgroup's TS x TS tile.
    let lane = local_index();
    let tr = lane / TS;
    let tc = lane % TS;
    // workgroup -> which output tile (tile coords laid out row-major over the C grid).
    let group = group_index();
    let tiles_n = n / TS;
    let tile_row = (group / tiles_n) * TS;
    let tile_col = (group % tiles_n) * TS;
    let row = tile_row + tr;
    let col = tile_col + tc;
    let kt_count = k / TS;
    let mut acc = 0.0f32;
    let mut kt = 0usize;
    while kt < kt_count {
        // stage one TS x TS sub-tile of A and of B into LDS (array 0 = A, array 1 = B). Tile-aligned, so
        // every index is in bounds.
        wg_write(0, tr * TS + tc, a[row * k + (kt * TS + tc)]);
        wg_write(1, tr * TS + tc, b[(kt * TS + tr) * n + col]);
        workgroup_barrier();
        // accumulate this lane's dot over the staged tile (branchless inner loop).
        let mut i = 0usize;
        while i < TS {
            acc = acc + wg_read(0, tr * TS + i) * wg_read(1, i * TS + tc);
            i = i + 1;
        }
        workgroup_barrier();
        kt = kt + 1;
    }
    out[row * n + col] = acc;
}
