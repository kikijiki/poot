//! Workgroup-parallel index intrinsics (card 044 / spec 054): `local_index_x()` is the lane id within the
//! workgroup, `group_index_x()` is the workgroup id, both distinct from the global `thread_index_x()`. Each
//! lane writes a value encoding both so the test can recover them: out[g] = local_index_x*1000 + group_index_x.
#![crate_type = "lib"]
use poot_kernel_attr::kernel;
use poot_kernel_intrinsics::{group_index_x, local_index_x, thread_index_x};
#[kernel(workgroup(64))]
pub fn wg_indices(out: &mut [f32]) {
    let g = thread_index_x();
    if g < out.len() {
        out[g] = (local_index_x() as f32) * 1000.0 + (group_index_x() as f32);
    }
}
