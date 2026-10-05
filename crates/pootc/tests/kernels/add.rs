//! An `add` kernel in ordinary Rust for the pootc MIR-import test, hand-mangled into the reserved
//! `__poot_kernel_` namespace so `pootc` discovers it via `poot_kernel_ir::naming::is_kernel`.
//!
//! `thread_index` is `poot_kernel_intrinsics::thread_index`: pootc recognizes the call by its resolved
//! declaration and replaces it with the GPU dispatch-id terminator (the crate's placeholder body is never
//! imported). Mirrors `fixtures::add_kernel`: one thread per element, a bounds guard, an indexed add.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = thread_index();
    if i < c.len() {
        c[i] = a[i] + b[i];
    }
}
