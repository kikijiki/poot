//! A sequential scalar gemm kernel for the AIE-core (XDNA2 NPU).
//!
//! C[m,n] = sum_k A[m*K+k] * B[k*N+n] for m in 0..M, n in 0..N, k in 0..K. A is [M,K] row-major, B is
//! [K,N] row-major, C is [M,N] row-major.
//!
//! # Length encoding
//!
//! The IRON harness calls this as `kern(ea, M*K, eb, K*N, ec, N)`: `ec` is the M*N output buffer but the
//! length argument passed is N. The kernel recovers N from `c.len()`, then K = b.len()/N and
//! M = a.len()/K, without a separate parameter channel. All index accesses stay within [0, M*N).
//!
//! The single AIE2p tile has 2 S2MM (host->tile) input DMA channels (A and B) and 1 MM2S (tile->host)
//! output channel, the same budget as gemv.
//!
//! Scalar (no vectorized MAC): correct for any M, K, N. Mirrors gemv_seq.rs: plain while-loop bodies, no
//! thread_index(). The pootc MIR importer handles the triple nested loop via standard SwitchInt + Goto.

#![crate_type = "lib"]

pub fn __poot_kernel_gemm_seq(a: &[f32], b: &[f32], c: &mut [f32]) {
    // a.len() = M*K, b.len() = K*N (both row-major), c.len() = N (the IRON caller passes N, not M*N).
    // Recover M, K, N:
    let n = c.len(); // N (received as c's "length" from the IRON caller)
    let k = b.len() / n; // K = (K*N) / N
    let m = a.len() / k; // M = (M*K) / K
    let mut mi = 0usize;
    while mi < m {
        let mut ni = 0usize;
        while ni < n {
            let mut acc = 0.0f32;
            let mut ki = 0usize;
            while ki < k {
                acc = acc + a[mi * k + ki] * b[ki * n + ni];
                ki += 1;
            }
            c[mi * n + ni] = acc;
            ni += 1;
        }
        mi += 1;
    }
}
