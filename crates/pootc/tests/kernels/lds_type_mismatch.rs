//! A workgroup-local array declared `u32` and accessed as `f32`: `pootc` rejects the access, naming the array
//! and both types, instead of reinterpreting the words.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{declare_lds, declare_workgroup, local_index_x, wg_write};

pub fn __poot_kernel_lds_type_mismatch(out: &mut [f32]) {
    declare_workgroup(64, 1, 1);
    declare_lds::<u32>(64);
    let lane = local_index_x();
    wg_write::<f32>(0, lane, 1.0);
    if lane < out.len() {
        out[lane] = 0.0;
    }
}
