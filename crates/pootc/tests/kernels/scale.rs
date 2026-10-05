//! `c[i] = a[i] * 2.0`. Exercises the f32-literal constant path (`map_mir_const`) and the `Mul` binop
//! beyond the `add` kernel.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_scale(a: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] * 2.0;
    }
}
