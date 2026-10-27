//! A kernel that reads its lane index without declaring a workgroup shape: `pootc` rejects it instead of
//! defaulting to 64 lanes.
#![crate_type = "lib"]
use poot_kernel_intrinsics::local_index_x;

pub fn __poot_kernel_workgroup_undeclared(out: &mut [f32]) {
    let lane = local_index_x();
    if lane < out.len() {
        out[lane] = 0.0;
    }
}
