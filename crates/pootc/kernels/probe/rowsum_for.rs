#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;
const COLS: usize = 4;
pub fn __poot_kernel_rowsum_for(x: &[f32], out: &mut [f32]) {
    let r = thread_index();
    if r < out.len() {
        let mut acc = 0.0f32;
        for j in 0..COLS {
            acc = acc + x[r * COLS + j];
        }
        out[r] = acc;
    }
}
