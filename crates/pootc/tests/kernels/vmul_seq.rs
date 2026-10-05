//! A sequential elementwise-multiply kernel for the AIE-core (XDNA2 NPU) import path. Mirrors vadd_seq.rs but
//! computes c[i] = a[i] * b[i]; the IRON dataflow (ObjectFifo wiring, single-tile design) is identical.

#![crate_type = "lib"]

pub fn __poot_kernel_vmul_seq(a: &[f32], b: &[f32], c: &mut [f32]) {
    let mut i = 0;
    while i < c.len() {
        c[i] = a[i] * b[i];
        i += 1;
    }
}
