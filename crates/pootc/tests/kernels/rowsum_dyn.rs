//! Shape-generic kernel (card 044): the row width is a runtime value read from a `dims: &[u32]` param, not
//! a const. out[r] = sum of x's row of `dims[0]` columns. One imported Body runs any shape; the dim rides
//! in a one-element slice, reusing the existing slice binding.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
pub fn __poot_kernel_rowsum_dyn(x: &[f32], dims: &[u32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let cols = dims[0] as usize;
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc = acc + x[r * cols + j];
        }
        out[r] = acc;
    }
}
