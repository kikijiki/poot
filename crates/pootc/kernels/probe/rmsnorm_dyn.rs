//! Shape-generic RMSNorm (card 044): the hidden dim is a runtime value from `dims: &[u32]`, not a const,
//! so one imported Body normalizes any hidden width. Same math as rmsnorm.rs.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
pub fn __poot_kernel_rmsnorm_dyn(x: &[f32], w: &[f32], dims: &[u32], out: &mut [f32]) {
    let r = thread_index();
    let d = dims[0] as usize;
    if r * d < out.len() {
        let mut ss = 0.0f32;
        for j in 0..d {
            let v = x[r * d + j];
            ss = ss + v * v;
        }
        let mean = ss / (d as f32);
        let scale = 1.0f32 / (mean + 0.00001f32).sqrt();
        for j in 0..d {
            out[r * d + j] = x[r * d + j] * scale * w[j];
        }
    }
}
