//! The `add` kernel authored with `#[kernel]`: the macro renames `add` to `__poot_kernel_add`, which
//! `pootc` discovers. The test compiles it with the proc-macro wired in via `--extern poot_kernel_attr=...`.

#![crate_type = "lib"]
use poot_kernel_attr::kernel;
use poot_kernel_intrinsics::thread_index;

#[kernel]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] + b[i];
    }
}
