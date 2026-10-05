//! Flash-attention decode (kernelgen `flash_attention_decode`), authored in ordinary Rust. One query token
//! per batch row attends to the fixed-capacity KV with GQA and a masked online softmax. One workgroup per
//! (batch row, query head) (grid = B*Hq threads, workgroup size 1, see `dispatch_grid`), so each one's
//! running output `o[D]` lives in its own workgroup's LDS array (array 0), disjoint, needing no barrier.
//! kernelgen keeps `o[D]` in a private array, which the importer cannot lower and which crashes LLVM's
//! SPIR-V backend (so it is NVPTX-only); holding it in LDS makes flash importable and runnable on wgpu.
//! card 044 / spec 055; batch dim added in card 038 (B=1 is the single-sequence path).
//!
//! Inputs (the `OpKind::FlashAttentionDecode` operands): `q[B,Hq,1,D]`, `k`/`v[B,Hkv,cap,D]`,
//! `mask[B,Hm,1,cap]`, plus a `[Hq, D, cap, n_rep, scale_bits, B, mask_head_stride, mask_batch_stride]`
//! metadata buffer (`Plan::ComputeMeta`). `scale` rides as raw bits (`f32::from_bits`) since it is not
//! always `1/sqrt(D)` (granite). The GQA kv head for query head `h` is `kv = h / n_rep`; `Hkv = Hq / n_rep`.
//! `out[B,Hq,1,D]`.
//!
//! card 259: the mask's head axis `Hm` is 1 (one row broadcast over every head) or Hq (one bias row per
//! head, needed by ALiBi). The two strides ride as element counts the planner computes from the mask
//! shape: `mask_head_stride` is 0 (broadcast) or `cap`, `mask_batch_stride` is `cap` or `Hq*cap`. The
//! planner's `flash_mask_per_head` states the rule; the CPU oracle's decomposition broadcasts the mask.

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, wg_read, wg_write};

const LDS_SIZE: usize = 256; // max supported head dim D (one LDS slot per output channel)
const WORKGROUP_SIZE: usize = 1; // one thread per workgroup; one workgroup per (batch row, query head)

pub fn __poot_kernel_flash_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    dims: &[u32],
    out: &mut [f32],
) {
    // `group_index() / WORKGROUP_SIZE` is just `group_index()`, but routing the const through a live
    // computation is how the importer reads it (an unused `let` is dead-code-eliminated); it sets
    // `Body.workgroup_size` to 1, so the grid becomes one 1-thread workgroup per (batch, head), each owning
    // a disjoint LDS region. The `d <= LDS_SIZE` guard likewise exposes LDS_SIZE and is a real bound.
    let gi = group_index() / WORKGROUP_SIZE; // global (batch, head) index in 0..B*Hq
    let hq = dims[0] as usize;
    let d = dims[1] as usize;
    let bsz = dims[5] as usize; // batch (1 for single-seq decode; B for the batched serving decode)
    if gi < bsz * hq && d <= LDS_SIZE {
        let cap = dims[2] as usize;
        let n_rep = dims[3] as usize;
        let scale = f32::from_bits(dims[4]);
        let bi = gi / hq; // batch row
        let h = gi % hq; // query head
        let hkv = hq / n_rep;
        let kv = h / n_rep;
        // q[B,Hq,1,D], k/v[B,Hkv,cap,D], mask[B,Hm,1,cap]; mask strides ride in dims[6]/dims[7].
        let mask_head_stride = dims[6] as usize;
        let mask_batch_stride = dims[7] as usize;
        let qbase = (bi * hq + h) * d;
        let kvbase = (bi * hkv + kv) * cap * d;
        let maskbase = bi * mask_batch_stride + h * mask_head_stride;
        // o[0..d] = 0 (LDS array 0, this (batch,head)'s private region)
        let mut d0 = 0;
        while d0 < d {
            wg_write(0, d0, 0.0);
            d0 = d0 + 1;
        }
        let mut m = -1.0e30f32; // running max
        let mut l = 0.0f32; // running denominator
        let mut j = 0;
        while j < cap {
            // s = scale * dot(q[bi,h], k[bi,kv,j]) + mask[bi,j]
            let kjbase = kvbase + j * d;
            let mut dot = 0.0f32;
            let mut e = 0;
            while e < d {
                dot = dot + q[qbase + e] * k[kjbase + e];
                e = e + 1;
            }
            // `maskbase` already carries this head's row on the per-head (ALiBi) mask, so the bias is
            // inside the running-max / rescale recurrence.
            let s = dot * scale + mask[maskbase + j];
            let m_new = m.max(s);
            let corr = (m - m_new).exp();
            let p = (s - m_new).exp();
            l = l * corr + p;
            // o[dd] = o[dd]*corr + p * v[kv,j,dd]
            let mut dd = 0;
            while dd < d {
                let ov = wg_read(0, dd) * corr + p * v[kjbase + dd];
                wg_write(0, dd, ov);
                dd = dd + 1;
            }
            m = m_new;
            j = j + 1;
        }
        // out[h,dd] = o[dd] / l
        let mut dd2 = 0;
        while dd2 < d {
            out[qbase + dd2] = wg_read(0, dd2) / l;
            dd2 = dd2 + 1;
        }
    }
}
