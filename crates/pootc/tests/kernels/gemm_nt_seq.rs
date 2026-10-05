//! Sequential scalar gemm-NT (B transposed) kernel for the AIE-core (XDNA2 NPU).
//!
//! C[i,j] = sum_d A[i*D+d] * B[j*D+d], with A[L,D] and B[L,D] row-major and C[L,L]. Equivalent to
//! A @ B^T without moving data.
//!
//! # Length encoding
//!
//! The IRON harness calls this as `kern(eq, L*D, ek, L*D, ec, L)`: `ec` is the L*L output buffer but its
//! length argument is L. The kernel recovers L = c.len(), D = a.len()/L. All index accesses stay within
//! [0, L*L).
//!
//! Used in the attention 3-tile cascade (compile_attention): tile 1 computes Q[L,D] @ K[L,D]^T -> scores[L,L].
//! The two S2MM channels hold Q and K; no cascade is needed for this tile.

#![crate_type = "lib"]

pub fn __poot_kernel_gemm_nt_seq(a: &[f32], b: &[f32], c: &mut [f32]) {
    // a.len() = L*D, b.len() = L*D (both row-major; b is treated as B^T in the loop), c.len() = L (the
    // IRON caller passes L, not L*L). Recover L and D:
    let l = c.len(); // L (sequence length; output is L*L square)
    let d = a.len() / l; // D (head dimension, inner contraction axis)
    let mut i = 0usize;
    while i < l {
        let mut j = 0usize;
        while j < l {
            let mut acc = 0.0f32;
            let mut dk = 0usize;
            while dk < d {
                // A[i,dk] * B[j,dk]: B accessed row-major, equivalent to B^T[dk,j].
                acc = acc + a[i * d + dk] * b[j * d + dk];
                dk += 1;
            }
            c[i * l + j] = acc;
            j += 1;
        }
        i += 1;
    }
}
