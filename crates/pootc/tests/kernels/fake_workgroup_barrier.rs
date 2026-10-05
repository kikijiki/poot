//! A user-defined `workgroup_barrier` that is NOT the shared intrinsic (card 531c; SC-004, R468-009):
//! `pootc` must reject the call as an ordinary, unsupported function call, never silently lower it as the
//! workgroup barrier terminator. The importer recognizes the intrinsic by its resolved declaration
//! (`poot_kernel_intrinsics::workgroup_barrier`), not by the callee's bare name, so this locally declared
//! function of the same name is an ordinary (rejected) call.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

fn workgroup_barrier() {}

pub fn __poot_kernel_fake_workgroup_barrier(a: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        workgroup_barrier();
        out[i] = a[i];
    }
}
