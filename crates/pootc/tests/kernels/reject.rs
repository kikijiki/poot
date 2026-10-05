//! A kernel using a construct outside the supported subset (a call to an arbitrary function): `pootc` must
//! reject it with a named "unsupported" diagnostic rather than silently miscompile.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

#[inline(never)]
fn helper(x: f32) -> f32 {
    x * 2.0
}

pub fn __poot_kernel_bad(a: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        // calling an arbitrary function is not in the kernel subset.
        c[i] = helper(a[i]);
    }
}
