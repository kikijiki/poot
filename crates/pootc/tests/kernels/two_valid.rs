//! Two valid kernels used to verify output-directory reuse removes a stale kernel set member.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] + b[i];
    }
}

pub fn __poot_kernel_scale(a: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] * 2.0;
    }
}
