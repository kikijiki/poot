//! Unary float-math methods on a kernel value: `x.sqrt()` (softmax/rmsnorm/rope need sqrt/exp/sin/cos).
//! The importer maps the math method call to a `MathUnary` rvalue. sqrt of perfect squares is exact, so the
//! GPU output is asserted without tolerance.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
pub fn __poot_kernel_math_unary(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        out[r] = x[r].sqrt();
    }
}
