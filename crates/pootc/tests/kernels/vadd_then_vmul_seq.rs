//! Fused 2-op sequential kernel for the AIE-core (XDNA2 NPU): d[i] = (a[i] + b[i]) * e[i].
//!
//! Three inputs, one output. Mirrors vadd_seq.rs and vmul_seq.rs but fuses the two ops into one tile pass,
//! so there is no intermediate buffer. The IRON harness wraps this with 4 ObjectFifos (3 in, 1 out) rather
//! than the 3-fifo (2 in, 1 out) layout of the single-op kernels.

#![crate_type = "lib"]

pub fn __poot_kernel_vadd_then_vmul_seq(a: &[f32], b: &[f32], e: &[f32], d: &mut [f32]) {
    let mut i = 0;
    while i < d.len() {
        d[i] = (a[i] + b[i]) * e[i];
        i += 1;
    }
}
