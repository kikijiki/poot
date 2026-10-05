//! A sequential elementwise-relu kernel for the AIE-core (XDNA2 NPU) import path.
//!
//! ReLU: y[i] = max(x[i], 0.0). Single-input unary op: one `while` loop walking the whole slice. No
//! `thread_index()`; the IRON harness tiles the data and dispatches this loop. Peano needs no libm here (no
//! transcendentals, just a conditional move).
//!
//! pootc MIR-imports this to a `Body` like the binary kernels; the AIE-core emitter lowers it to AIE2p
//! via Peano.

#![crate_type = "lib"]

pub fn __poot_kernel_vrelu_seq(x: &[f32], y: &mut [f32]) {
    let mut i = 0;
    while i < y.len() {
        let v = x[i];
        y[i] = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}
