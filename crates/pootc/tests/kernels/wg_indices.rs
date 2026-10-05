//! Workgroup-parallel index intrinsics (card 044 / spec 054): `local_index()` is the lane id within the
//! workgroup, `group_index()` is the workgroup id, both distinct from the global `thread_index()`. Each
//! lane writes a value encoding both so the test can recover them: out[g] = local_index*1000 + group_index.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, local_index, thread_index};
pub fn __poot_kernel_wg_indices(out: &mut [f32]) {
    let g = thread_index();
    if g < out.len() {
        out[g] = (local_index() as f32) * 1000.0 + (group_index() as f32);
    }
}
