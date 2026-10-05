//! A selection nested inside the `i < out.len()` bounds guard. This shape used to make the LLVM SPIR-V
//! backend emit invalid SPIR-V ("merge block not structurally dominated") and SIGSEGV RADV. With the
//! structurization pass (card 100 / spec 058) each selection gets a dedicated merge block, so it validates
//! and runs. Regression guard: an ordinary nested `if`, not branchless masking.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_nested_sel(a: &[f32], sel: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        if sel[i] != 0 {
            out[i] = a[i] * 2.0;
        } else {
            out[i] = a[i] + 1.0;
        }
    }
}
