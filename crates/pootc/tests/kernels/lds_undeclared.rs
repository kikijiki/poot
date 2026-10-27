//! A workgroup-local array that is written but never declared: `pootc` rejects it, naming the array, instead of
//! sizing one from the workgroup width.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{declare_workgroup, local_index_x, wg_write};

pub fn __poot_kernel_lds_undeclared(out: &mut [f32]) {
    declare_workgroup(64, 1, 1);
    let lane = local_index_x();
    wg_write::<f32>(0, lane, 1.0);
    if lane < out.len() {
        out[lane] = 0.0;
    }
}
