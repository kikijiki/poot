//! Pack 4 signed-int8 codes per i32 word along the last axis (kernelgen `pack_i8_dt`, spec 048 INT8
//! KV-cache quant) in ordinary Rust. Input is f32 codes `[.., L]`; output is i32 words `[.., W]` with
//! `W = ceil(L/4)`. One thread per output word rounds and clamps each of its 4 codes to a signed int8 and
//! packs them into the word's bytes. `L` is not recoverable from `W` (the last word may hold fewer than 4
//! codes), so it rides in a `[L]` metadata buffer (`Plan::ComputeMeta`). Branchless (the c<L tail check is
//! a multiply mask), so it imports and `spirv-val`s on both backends. card 044 / spec 055.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_pack_i8(codes: &[f32], dims: &[u32], out: &mut [i32]) {
    let i = thread_index();
    if i < out.len() {
        let l = dims[0] as usize;
        let w = (l + 3) / 4; // words per row = ceil(L/4)
        let row = i / w;
        let wpos = i % w;
        let mut acc = 0i32;
        let mut j = 0usize;
        while j < 4 {
            let c = wpos * 4 + j;
            // branchless tail guard: when c >= L this lane is invalid; read code 0 and mask its bytes off.
            let valid = (c < l) as usize;
            let cc = c * valid;
            let code = codes[row * l + cc];
            let q = (code.round().max(-127.0).min(127.0)) as i32; // signed int8 value
            let mask = (valid as i32) * 0xFF; // 0xFF when valid, 0 otherwise
            // the shift amount is i32 (matching the shifted value): an i64 amount (`j * 8`, usize) makes
            // NVPTX llc reject the `shl` on a type mismatch; SPIR-V tolerates it.
            let sh = (j as i32) * 8;
            acc = acc | (((q & 0xFF) & mask) << sh);
            j = j + 1;
        }
        out[i] = acc;
    }
}
