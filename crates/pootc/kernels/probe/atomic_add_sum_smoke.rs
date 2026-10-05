//! The `atomic_add` imported-kernel intrinsic (card 148 phase 2b): an order-independent exact integer
//! reduction, mirroring kernelgen's `global_atomic_add_counter`. Every lane atomically adds 1 into the u32
//! counter at `out[0]`; with N lanes the counter must be exactly N, which only holds if the atomic
//! serializes the read-modify-writes. Lowers to `Rvalue::GlobalAtomic { op: AtomicOp::Add, .. }` on
//! `buffer[index]`. u32 so the sum is bit-reproducible (float atomic add is order-dependent).
#![crate_type = "lib"]
use poot_kernel_intrinsics::atomic_add;
pub fn __poot_kernel_atomic_add_sum_smoke(out: &mut [u32]) {
    let idx = 0usize;
    let _old = atomic_add(out, idx, 1u32);
}
