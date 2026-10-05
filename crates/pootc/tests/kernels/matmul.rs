//! A GEMM kernel in ordinary Rust: `c[M,N] = a[M,K] @ b[K,N]`, one thread per output element, with div/rem
//! index math and a `for k in 0..K` accumulation loop (kernelgen's `matmul_batched` shape).

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

const M: usize = 2;
const N: usize = 3;
const K: usize = 4;

pub fn __poot_kernel_matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
    let idx = thread_index();
    if idx < c.len() {
        let row = idx / N;
        let col = idx % N;
        let mut acc = 0.0f32;
        for k in 0..K {
            acc = acc + a[row * K + k] * b[k * N + col];
        }
        c[idx] = acc;
    }
    let _ = M; // M documents the row count; the kernel derives it from c.len()/N.
}
