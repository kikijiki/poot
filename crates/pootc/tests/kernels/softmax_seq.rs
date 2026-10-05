//! Sequential numerically-stable softmax kernel for the AIE-core (XDNA2 NPU).
//!
//! y[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
//!
//! Three passes: find the max over all N elements; accumulate sum of exp(x[i] - max); write
//! y[i] = exp(x[i] - max) / sum.
//!
//! Single input, single output: 1 S2MM + 1 MM2S in IRON (same channel budget as vrelu_seq / vsilu_seq).
//! No cascade needed.
//!
//! No thread_index(): the IRON harness tiles data and dispatches this loop over the full input buffer on
//! one AIE2p tile.
//!
//! Build with:
//!   POOT_AIE_LLC=~/refs/peano/peano-llc \
//!   cargo test -p pootc --test import_run \
//!     imported_vsoftmax_sequential_kernel_lowers_to_aie2p --release

#![crate_type = "lib"]

pub fn __poot_kernel_vsoftmax_seq(x: &[f32], y: &mut [f32]) {
    let n = y.len();
    // Pass 1: find the running max.
    let mut m = x[0];
    let mut i = 1usize;
    while i < n {
        if x[i] > m {
            m = x[i];
        }
        i += 1;
    }
    // Pass 2: sum exp(x[i] - m).
    let mut s = 0.0f32;
    i = 0;
    while i < n {
        s += (x[i] - m).exp();
        i += 1;
    }
    // Pass 3: normalize.
    i = 0;
    while i < n {
        y[i] = (x[i] - m).exp() / s;
        i += 1;
    }
}
