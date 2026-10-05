//! The kernel-authoring intrinsics: dispatch-axis indices, the workgroup barrier, workgroup-local (LDS)
//! array access, the integer atomic add (spec 054, card 148 phase 2b), and the no-contraction marked add
//! `no_contract_add` (card 675, ADR 0114 tier 1).
//!
//! A `#[kernel]`-authored kernel source depends on this crate (`--extern poot_kernel_intrinsics=...`) and
//! calls these functions directly; it never redeclares its own stubs. `pootc`'s importer
//! ([`import_body`](https://docs.rs/pootc) - see `crates/pootc/src/import.rs`) recognizes a call by its
//! *resolved declaration*: the callee's full path must be `poot_kernel_intrinsics::<name>`, not merely end
//! in a matching bare name. A kernel-authored helper that happens to share a name with one of these (for
//! example a local `fn workgroup_barrier()`) is therefore never mistaken for the intrinsic (R468-009).
//!
//! On a normal host build these bodies are placeholders: `pootc` never imports them (it recognizes the
//! *call*, replaces it with the device terminator or statement the call lowers to, and never looks at the
//! callee's MIR body), so the values below only matter for a plain `cargo check`/`cargo test` of this
//! crate or of a kernel source built outside `pootc`.

/// The global dispatch index along X (the thread's position in the whole grid). Alias of
/// [`thread_index_x`].
#[inline(never)]
pub fn thread_index() -> usize {
    0
}

/// The global dispatch index along X. Same axis as [`thread_index`].
#[cfg(test)]
#[inline(never)]
pub(crate) fn thread_index_x() -> usize {
    0
}

/// The lane's position within its workgroup along X (`0..workgroup_size[0]`), used to address per-lane
/// workgroup-local (LDS) slots (spec 054). Alias of [`local_index_x`].
#[inline(never)]
pub fn local_index() -> usize {
    0
}

/// The lane's position within its workgroup along X. Same axis as [`local_index`].
#[cfg(test)]
#[inline(never)]
pub(crate) fn local_index_x() -> usize {
    0
}

/// The workgroup's index along X (the row a per-row workgroup reduction owns). Alias of
/// [`group_index_x`].
#[inline(never)]
pub fn group_index() -> usize {
    0
}

/// The workgroup's index along X. Same axis as [`group_index`].
#[cfg(test)]
#[inline(never)]
pub(crate) fn group_index_x() -> usize {
    0
}

/// The workgroup's index along Y.
#[inline(never)]
pub fn group_index_y() -> usize {
    0
}

/// A workgroup-scope control and memory barrier: every lane in the workgroup waits here, and every
/// workgroup-local (LDS) write issued before the barrier is visible to every lane after it.
#[inline(never)]
pub fn workgroup_barrier() {}

/// Write `val` to workgroup-local (LDS) array `array` at `idx`.
#[inline(never)]
pub fn wg_write(_array: usize, _idx: usize, _val: f32) {}

/// Read workgroup-local (LDS) array `array` at `idx`.
#[inline(never)]
pub fn wg_read(_array: usize, _idx: usize) -> f32 {
    0.0
}

/// Atomically add `value` to `buffer[index]`, returning the old value: an order-independent exact integer
/// reduction (float atomic add is order-dependent, so this is `u32`-only).
#[inline(never)]
pub fn atomic_add(_buffer: &mut [u32], _index: usize, _value: u32) -> u32 {
    0
}

/// `a + b` as two separate roundings that codegen must never contract into a fused multiply-add: `pootc`
/// lowers this call to `poot_kernel_ir::Rvalue::BinaryOpNoContract(BinOp::Add, a, b)` (card 675), the same
/// per-op marker `poot-kernelgen`'s Body API already uses for the packed-dequant decode formula (card 628).
/// Write the preceding multiply as its own statement (`let prod = x * y;`) and pass it as `b`: a tier-1
/// (bit-exact) sampling primitive whose source has two roundings (e.g. the Gumbel select's
/// `v*inv_temp + noise_scale*noise`, ADR 0114) must keep both on every backend, never let one backend's
/// compiler fuse the trailing multiply into this add.
#[inline(never)]
pub fn no_contract_add(a: f32, b: f32) -> f32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aliases_match_their_documented_default() {
        assert_eq!(thread_index_x(), 0);
        assert_eq!(local_index_x(), 0);
        assert_eq!(group_index_x(), 0);
    }
}
