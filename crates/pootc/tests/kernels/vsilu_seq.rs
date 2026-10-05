//! A sequential elementwise-silu kernel for the AIE-core (XDNA2 NPU) import path.
//!
//! SiLU (swish): y[i] = x[i] / (1.0 + exp(-x[i])). Single-input unary op: one `while` loop walking the
//! whole slice. No `thread_index()`; the IRON harness tiles the data and dispatches this loop.
//!
//! `f32::exp()` is lowered by the AIE-core emitter to a software inline sequence (magic-rounding + Horner
//! poly + i32 exponent scaling). No external expf call is emitted; only soft-float builtins (__mulsf3,
//! __divsf3, __fixsfsi, __gtsf2, __ltsf2) from libclang_rt.builtins.a, which Peano always links.
//!
//! Build with:
//!   POOT_AIE_LLC=~/refs/peano/peano-llc \
//!   cargo test -p pootc --test import_run \
//!     imported_vsilu_sequential_kernel_lowers_to_aie2p --release

#![crate_type = "lib"]

pub fn __poot_kernel_vsilu_seq(x: &[f32], y: &mut [f32]) {
    let mut i = 0;
    while i < y.len() {
        let v = x[i];
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        y[i] = v / (1.0_f32 + (-v).exp());
        i += 1;
    }
}
