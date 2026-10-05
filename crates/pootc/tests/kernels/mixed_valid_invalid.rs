//! One valid and one unsupported kernel in the same module. Card 310 uses this to prove that a valid
//! sibling is not published when any import in the requested artifact set fails.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

#[inline(never)]
fn unsupported_helper(x: f32) -> f32 {
    x * 2.0
}

pub fn __poot_kernel_good(a: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        out[i] = a[i] + 1.0;
    }
}

pub fn __poot_kernel_bad_sibling(a: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        out[i] = unsupported_helper(a[i]);
    }
}
