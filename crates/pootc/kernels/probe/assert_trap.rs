//! A kernel with a reachable, ordinary bounds-check `Assert` (rustc's own, inserted for every slice
//! index - not an explicit `assert!()` macro, which lowers through a real `core::panicking::panic` call
//! outside the imported kernel subset), used to prove `Assert` lowers to a real device trap (ROCm, PTX)
//! or the SpirvVulkan error word, never a silent erasure (card 531c, R468-007; SC-001/SC-002/SC-003).
//! `idx[i]` is a caller-controlled index into `data`: with every index in range the kernel runs to
//! completion normally (no trap fires); with one index at or past `data.len()`, that lane's `data[j]`
//! bounds check fails.
#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_assert_trap(idx: &[u32], data: &[f32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let j = idx[i] as usize;
        out[i] = data[j];
    }
}
