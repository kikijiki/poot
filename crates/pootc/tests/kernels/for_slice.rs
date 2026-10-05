//! `for v in x`: iterate a slice's elements directly. The yielded `&f32` is dereferenced in the body; the
//! importer reuses the loop var as a `0..x.len()` counter and rewrites `*v` to `x[counter]`. out[r] = sum
//! of every element of x (same for each row), so a wrong bound would undercount.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
pub fn __poot_kernel_for_slice(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let mut acc = 0.0f32;
        for v in x {
            acc = acc + *v;
        }
        out[r] = acc;
    }
}
