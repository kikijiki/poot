//! RMSNorm authored in ordinary Rust and imported from MIR (card 044). One thread per row:
//! out[r,j] = x[r,j] * rsqrt(mean_j(x[r,j]^2) + eps) * w[j]. Exercises a sum-of-squares reduction
//! (`for j in 0..D`), the `D as f32` cast, `.sqrt()`, division, and the broadcast weight.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
const D: usize = 4;
pub fn __poot_kernel_rmsnorm(x: &[f32], w: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r * D < out.len() {
        let mut ss = 0.0f32;
        for j in 0..D {
            let v = x[r * D + j];
            ss = ss + v * v;
        }
        let mean = ss / (D as f32);
        let scale = 1.0f32 / (mean + 0.00001f32).sqrt();
        for j in 0..D {
            out[r * D + j] = x[r * D + j] * scale * w[j];
        }
    }
}
