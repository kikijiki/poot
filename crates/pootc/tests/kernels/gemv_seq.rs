//! A sequential scalar gemv kernel for the AIE-core (XDNA2 NPU).
//!
//! y[m] = sum_k A[m*K + k] * x[k] for m in 0..M, k in 0..K. A is row-major, a flat slice of M*K elements;
//! K = x.len(), M = y.len().
//!
//! The IRON harness streams A + x in via two ObjectFifos (one per tile S2MM channel) and drains y out. The
//! AIE2p tile runs this loop once per acquire cycle.
//!
//! Scalar (no vectorized MAC): correct for any M, K. Mirrors vadd_seq.rs: a plain while-loop body, no
//! thread_index(). The pootc MIR importer handles the nested loop via standard SwitchInt + Goto.

#![crate_type = "lib"]

pub fn __poot_kernel_gemv_seq(mat: &[f32], vec: &[f32], out: &mut [f32]) {
    let k = vec.len();
    let mut m = 0;
    while m < out.len() {
        let mut acc = 0.0f32;
        let mut j = 0;
        while j < k {
            acc = acc + mat[m * k + j] * vec[j];
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
}
