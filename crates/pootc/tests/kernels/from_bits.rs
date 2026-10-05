//! `f32::from_bits` import probe: read an f32 that rode through a u32 buffer (attention `scale` in the
//! dims buffer, since not every scale is `1/sqrt(D)`). `out[i] = inp[i] * f32::from_bits(bits[0])`. The
//! importer maps `from_bits` to a `Bitcast` (a reinterpret, not an `as` cast). card 044 / spec 055.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_from_bits(inp: &[f32], bits: &[u32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() {
        let scale = f32::from_bits(bits[0]);
        out[i] = inp[i] * scale;
    }
}
