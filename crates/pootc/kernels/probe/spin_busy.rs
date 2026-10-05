//! A controllable-duration busy-spin kernel: `n[0]` serial dependent-chain iterations per lane,
//! finite (it always eventually completes) but with no upper bound the host can predict from
//! outside. Used by `poot-rocm-gpu`'s Card 548 acceptance to force a real, deterministic HSA
//! completion-wait timeout (a tiny `RocmContextOptions::wait_timeout` against a real in-flight
//! dispatch), rather than reaching into private queue internals to fake one.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_spin_busy(n: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let mut acc: u32 = 0;
        let mut k: u32 = 0;
        while k < n[0] {
            acc = (acc + k) ^ (acc << 1) ^ (acc >> 3);
            k += 1;
        }
        out[i] = acc as f32;
    }
}
