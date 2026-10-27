//! A kernel that declares its launch: a 2-D workgroup and two workgroup-local arrays of different element
//! types and lengths, which the importer reads from the `declare_workgroup` / `declare_lds` calls
//! `#[kernel(workgroup(8, 4), lds(u32, 32), lds(f32, 16))]` expands to.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{
    declare_lds, declare_workgroup, local_index_x, local_index_y, wg_read, wg_write,
    workgroup_barrier,
};

pub fn __poot_kernel_declare_launch(out: &mut [f32]) {
    declare_workgroup(8, 4, 1);
    declare_lds::<u32>(32);
    declare_lds::<f32>(16);
    let lane = local_index_x() + 8 * local_index_y();
    wg_write::<u32>(0, lane, 1);
    wg_write::<f32>(1, lane % 16, 2.0);
    workgroup_barrier();
    if lane < out.len() {
        out[lane] = wg_read::<u32>(0, lane) as f32 + wg_read::<f32>(1, lane % 16);
    }
}
