//! A workgroup-parallel LDS sum-reduction (card 044 / spec 054), the shape of kernelgen's `wg_sum`: each of
//! the 64 lanes writes its input element to the workgroup-local (LDS) array, a barrier, then lane 0 sums the
//! 64 slots into out[0]. Exercises local_index(), wg_write()/wg_read(), and workgroup_barrier().
#![crate_type = "lib"]
use poot_kernel_intrinsics::{local_index, wg_read, wg_write, workgroup_barrier};
const W: usize = 64;
pub fn __poot_kernel_wg_reduce(a: &[f32], out: &mut [f32]) {
    let lane = local_index();
    wg_write(0, lane, a[lane]);
    workgroup_barrier();
    if lane == 0 {
        let mut acc = 0.0f32;
        for j in 0..W {
            acc = acc + wg_read(0, j);
        }
        out[0] = acc;
    }
}
