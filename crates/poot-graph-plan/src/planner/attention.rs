//! Attention equation planning: one generated online-softmax kernel for every phase, its shape read from
//! the equation and its lane count from the device.

use super::*;
use poot_graph_ir::{AttentionAttrs, KeyLayout, KvBound, MaskOperand};
use poot_kernelgen::{
    AttentionSchedule, AttentionSpec, KeyBound, KeyLayout as KernelKeyLayout, MaskKind,
    ScoreTransform,
};
use poot_target::{Backend, DeviceCaps};

/// Plan one attention equation and record the kernel it takes.
///
/// `q[B,Hq,Lq,Dk]`, `k[B,Hkv,Lk,Dk]` (or its split parts), `v[B,Hkv,Lk,Dv]`, `mask[B,Hm,Lq,Lk]`: `Lq` and `Lk` are
/// independent (decode `Lq == 1`, prefill `Lq == Lk`, prefill into a cache `1 < Lq < Lk`), and all are the same
/// request. A live bound reads the `len [B]` operand at run time; the grid never depends on it. The streaming
/// schedule runs one workgroup of `lanes` threads per `(b, h, i)`; `lanes` is the widest power of two whose
/// partials fit the device's workgroup memory, and the equation is refused only when not even one lane's fit.
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let OpKind::Attention(AttentionAttrs {
        n_rep,
        scale,
        softcap,
        mask: MaskOperand::Additive,
        key,
        bound,
    }) = &eqn.op
    else {
        unreachable!("plan_eqn routes only attention ops here")
    };
    let site = Site::new(g, eqn, backend, limits);
    let ids = value_ids(eqn);
    let shape = |operand: usize| g.aval(ids[operand]).shape.clone();
    let (q, k, v, mask) = (shape(0), shape(1), shape(2), shape(3));
    let [batch, hq, lq, dk] = q[..] else {
        unreachable!("`infer` types attention over a rank-4 query")
    };
    let lk = k[2];
    let mask_kind = if flash_mask_per_head(&mask, hq, "attention")? {
        MaskKind::PerHead
    } else {
        MaskKind::Broadcast
    };
    let dv = v[3];
    let lanes = attention_lanes(dv, caps).ok_or_else(|| {
        site.refuse(Capability::AttentionLds {
            dv,
            lds_bytes: caps.lds_bytes,
        })
    })?;
    let score = match softcap {
        None => ScoreTransform::Scale {
            scale_bits: scale.to_bits(),
        },
        Some(cap) => ScoreTransform::ScaleSoftcap {
            scale_bits: scale.to_bits(),
            cap_bits: cap.to_bits(),
        },
    };
    Planned::generated(
        site,
        KernelRequest::Attention(AttentionSpec {
            batch,
            hq,
            n_rep: *n_rep,
            lq,
            lk,
            dk,
            dv,
            key: match key {
                KeyLayout::Single => KernelKeyLayout::Single,
                KeyLayout::Split { shared_dim } => KernelKeyLayout::Split {
                    shared_dim: *shared_dim,
                },
            },
            score,
            mask: mask_kind,
            bound: match bound {
                KvBound::Capacity => KeyBound::Static,
                KvBound::Live => KeyBound::Live,
            },
            schedule: AttentionSchedule::Streaming { lanes },
        }),
        backend,
        caps,
    )
}
