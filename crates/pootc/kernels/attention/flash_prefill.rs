//! Flash-attention prefill, authored in ordinary Rust: the M=L-queries analog of `flash_decode.rs`. Every
//! query row attends to the whole-prompt KV with GQA and a masked online softmax. One workgroup per
//! (query head, query row): grid = Hq*L threads, workgroup size 1 (see `dispatch_grid`), so each
//! (head,row)'s running output `o[D]` lives in its own workgroup's LDS array (array 0), disjoint, needing
//! no barrier. Same importable LDS design as decode (a private array is unimportable). card 038.
//!
//! Unlike the materialized `ops::attention_prefill` (full `[1,Hq,L,L]` scores, O(L^2) transient memory),
//! this keeps one row's running state, so peak attention memory is O(L*D).
//!
//! Inputs (the `OpKind::FlashAttentionPrefill` operands): `q[1,Hq,L,D]`, `k`/`v[1,Hkv,L,D]`, `mask[1,Hm,L,L]`,
//! plus a `[Hq, L, D, n_rep, scale_bits, x_groups, has_softcap, cap_bits, mask_head_stride]` metadata buffer
//! (`Plan::ComputeMeta`). `scale`/`cap` ride as raw bits (`f32::from_bits`). `x_groups` is the 2-D
//! workgroup-grid X width (so `Hq*L` can exceed the wgpu gridDim.x cap at long context). The GQA kv head
//! for query head `h` is `kv = h / n_rep`; the causal mask is `mask[row, j]`. `out[1,Hq,L,D]`.
//!
//! card 259: the mask's head axis `Hm` is 1 (one `[L,L]` plane broadcast over every head) or Hq (one
//! plane per head, needed by ALiBi). `mask_head_stride` (dims[8]) is the plane size in elements: 0
//! (broadcast) or `L*L`, computed by the planner (`flash_mask_per_head`); the CPU oracle's
//! decomposition broadcasts the mask.
//!
//! card 198: `has_softcap`/`cap_bits` implement Gemma2/Grok attention-logit softcapping
//! (`c * tanh(scores / c)`, `ops::softcap`), applied to `scale*dot` before the mask-add, matching
//! `ops::attention_prefill_softcap` (mul-by-reciprocal, the `exp(x-|x|)` tanh form `ops::tanh` builds,
//! then mul-by-cap). It is gated by an ordinary `if has_softcap_f > 0.5` rather than a branchless blend:
//! `has_softcap` is workgroup-uniform (read from `dims`), so this is not the divergent nested branch of
//! [[importer-nested-selection-invalid-spirv]]. `nested_sel.rs` and `barrier_branch_loop.rs` are the
//! regression-guard precedents that this shape emits valid SPIR-V. The `if` costs nothing on the
//! `has_softcap_f == 0.0` path every other model takes (a branchless form added 2 `exp` per KV step).

#![crate_type = "lib"]
use poot_kernel_intrinsics::{group_index, group_index_y, wg_read, wg_write};

const LDS_SIZE: usize = 256; // max supported head dim D (one LDS slot per output channel)
const WORKGROUP_SIZE: usize = 1; // one thread per workgroup; one workgroup per (head, query row)

pub fn __poot_kernel_flash_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    dims: &[u32],
    out: &mut [f32],
) {
    // `group_index() / WORKGROUP_SIZE` (= GroupX, since WORKGROUP_SIZE=1) routes the const through a live
    // computation so the importer reads it and sets `Body.workgroup_size` to 1. The flat (head, row) index
    // is laid over a 2-D grid `gi = GroupY*x_groups + GroupX` (x_groups in dims[5]) so Hq*L can exceed the
    // wgpu gridDim.x cap of 65535; a 1-D launch has GroupY=0. `d <= LDS_SIZE` exposes the LDS size.
    let x_groups = dims[5] as usize;
    let gi = group_index_y() * x_groups + group_index() / WORKGROUP_SIZE; // global (head, row) index 0..Hq*L
    let hq = dims[0] as usize;
    let l = dims[1] as usize;
    let d = dims[2] as usize;
    if gi < hq * l && d <= LDS_SIZE {
        let n_rep = dims[3] as usize;
        let scale = f32::from_bits(dims[4]);
        // has_softcap_f is 0.0 or 1.0, workgroup-uniform; see the module doc for why this is an ordinary `if`.
        let has_softcap_f = dims[6] as f32;
        let cap = f32::from_bits(dims[7]);
        let h = gi / l; // query head
        let row = gi % l; // query row (position)
        let kv = h / n_rep;
        let mask_head_stride = dims[8] as usize; // card 259: 0 (broadcast) or L*L (per-head, ALiBi)
        let qbase = (h * l + row) * d; // q[1,Hq,L,D] at [h, row, :]
        let kvbase = kv * l * d; // k/v[1,Hkv,L,D] at [kv, 0, :]
        let maskbase = h * mask_head_stride + row * l; // mask[1,Hm,L,L] at [h, row, :]
        // o[0..d] = 0 (LDS array 0, this (head,row)'s private region)
        let mut d0 = 0;
        while d0 < d {
            wg_write(0, d0, 0.0);
            d0 = d0 + 1;
        }
        let mut m = -1.0e30f32; // running max
        let mut ll = 0.0f32; // running denominator
        let mut j = 0;
        while j < l {
            // s = scale * dot(q[h,row], k[kv,j]) + mask[row,j]
            let kjbase = kvbase + j * d;
            let mut dot = 0.0f32;
            let mut e = 0;
            while e < d {
                dot = dot + q[qbase + e] * k[kjbase + e];
                e = e + 1;
            }
            let raw = dot * scale;
            let mask_val = mask[maskbase + j];
            let mut s = raw + mask_val;
            // softcap(raw, cap) = cap * tanh(raw / cap), via the `ops::tanh` form (exp(x-|x|)); skipped
            // entirely when has_softcap_f == 0.0.
            if has_softcap_f > 0.5 {
                let inv_cap = 1.0 / cap;
                let arg = raw * inv_cap;
                let neg_arg = -arg;
                let abs_arg = arg.max(neg_arg);
                let neg_abs = -abs_arg;
                let e_pos = (arg + neg_abs).exp();
                let e_neg = (neg_abs - arg).exp();
                let capped = ((e_pos - e_neg) / (e_pos + e_neg)) * cap;
                s = capped + mask_val;
            }
            let m_new = m.max(s);
            let corr = (m - m_new).exp();
            let p = (s - m_new).exp();
            ll = ll * corr + p;
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
        // out[h,row,dd] = o[dd] / ll
        let mut dd2 = 0;
        while dd2 < d {
            out[qbase + dd2] = wg_read(0, dd2) / ll;
            dd2 = dd2 + 1;
        }
    }
}
