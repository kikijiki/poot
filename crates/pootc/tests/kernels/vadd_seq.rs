//! A sequential elementwise-add kernel for the AIE-core (XDNA2 NPU) import path. Unlike the SPMD `add.rs`
//! (one thread per element via `thread_index()`), one core walks the whole slice in a `while` loop, the
//! shape a single AIE core runs (the IRON harness tiles the data and dispatches this loop). pootc
//! MIR-imports it to a `Body` like the GPU kernels; the AIE-core emitter lowers it to AIE2p via Peano
//! (real Rust, real MIR, no hand-built IR). No `thread_index()` call, so the importer
//! produces a plain looped CFG.

#![crate_type = "lib"]

pub fn __poot_kernel_vadd_seq(a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut i = 0;
    while i < c.len() {
        c[i] = a[i] + b[i];
        i += 1;
    }
}
