//! A reversed range `for j in (0..N).rev()`, not in the supported subset (exclusive `a..b` and inclusive
//! `a..=b` are; `.rev()`, `.step_by()` and `for x in slice` are not). Must be rejected with a clear
//! diagnostic, never miscompiled into a wrong iteration order.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
const N: usize = 3;
pub fn __poot_kernel_rev(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let mut acc = 0.0f32;
        for j in (0..N).rev() {
            acc = acc + x[j];
        }
        out[r] = acc;
    }
}
