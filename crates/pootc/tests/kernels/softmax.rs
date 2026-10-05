//! Numerically-stable softmax imported from MIR (card 044). Per row: subtract the row max, exponentiate,
//! normalize by the sum. Exercises `x.max(y)`, `.exp()`, a `1..D` exclusive range, two reductions, and
//! division.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
const D: usize = 4;
pub fn __poot_kernel_softmax(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r * D < out.len() {
        let mut m = x[r * D];
        for j in 1..D {
            m = m.max(x[r * D + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..D {
            sum = sum + (x[r * D + j] - m).exp();
        }
        for j in 0..D {
            out[r * D + j] = (x[r * D + j] - m).exp() / sum;
        }
    }
}
