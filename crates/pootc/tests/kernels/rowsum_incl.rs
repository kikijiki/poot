//! An inclusive-range `for j in 0..=N` accumulation loop, the inclusive analog of rowsum_for:
//! out[r] = sum(x[r*WIDTH + j] for j in 0..=N), i.e. N+1 columns.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
const N: usize = 3; // inclusive: j = 0,1,2,3 -> 4 columns
const WIDTH: usize = 4;
pub fn __poot_kernel_rowsum_incl(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let mut acc = 0.0f32;
        for j in 0..=N {
            acc = acc + x[r * WIDTH + j];
        }
        out[r] = acc;
    }
}
