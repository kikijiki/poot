//! Attention equation planning: one generated online-softmax kernel for every phase, its shape read from
//! the equation and its lane count from the device.

use super::{
    Binding, BodyLimits, Capability, Eqn, KernelRequest, OpKind, PlanError, Planned, ScratchId,
    Sink, Source, value_ids,
};
use crate::graph_validation::flash_mask_per_head;
use crate::planner::GraphTables;
use crate::predicates::{attention_lanes, attention_splits};
use crate::refusal::Site;
use crate::types::ScratchDecl;
use poot_graph_ir::{AttentionAttrs, KeyLayout, KvBound, MaskOperand};
use poot_kernelgen::{
    AttentionCombineSpec, AttentionSchedule, AttentionSpec, KeyBound, KeyLayout as KernelKeyLayout,
    MaskKind, ScoreTransform,
};
use poot_target::{Backend, BufferStorage, DeviceCaps};

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
    let streaming = AttentionSpec {
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
    };
    // This is the GPU planner: an AIE target never reaches it (`Target::Aie` plans through the AIE schedule).
    let splits = attention_splits(&streaming, caps);
    match splits {
        None => Planned::generated(site, KernelRequest::Attention(streaming), backend, caps),
        Some(splits) => split_kv(
            site,
            streaming,
            SplitKv { lanes, splits },
            ids.len(),
            backend,
            caps,
        ),
    }
}

/// The lanes and key chunks of a split-KV attention.
struct SplitKv {
    lanes: usize,
    splits: usize,
}

/// The split-KV plan of a decode attention (POOT-1113): stage one writes the `splits` partial `(o, m, l)`
/// states of every row into one scratch buffer, stage two combines them into the output. The scratch is the
/// equation's own (`ScratchDecl`), colored by the buffer plan with the values, so it is budgeted and reused
/// across layers with no allocator of its own.
fn split_kv(
    site: Site<'_>,
    streaming: AttentionSpec,
    split: SplitKv,
    operands: usize,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let partials = AttentionSpec {
        schedule: AttentionSchedule::SplitKv {
            lanes: split.lanes,
            splits: split.splits,
        },
        ..streaming
    };
    let combine =
        AttentionCombineSpec::of(&partials).expect("a split schedule has a combine stage");
    let elems = partials
        .partial_elems()
        .expect("a split schedule has a partial state");
    let scratch = ScratchId::new(0);
    let stage_one =
        Planned::generated_dispatch(site, KernelRequest::Attention(partials), backend, caps)?;
    // Stage one reads the equation's operands in order, as the streaming kernel does, and writes the
    // scratch; its default binding names the output, so it is rebound with the scratch as the sink.
    let reads = (0..operands).map(Source::Operand).collect();
    let partials = stage_one.bound(Binding::new(reads, Sink::Scratch(scratch)));
    let combine = Planned::generated_dispatch(
        site,
        KernelRequest::AttentionCombine(combine),
        backend,
        caps,
    )?
    .bound(Binding::new(vec![Source::Scratch(scratch)], Sink::Out));
    Planned::pipeline(
        vec![partials, combine],
        vec![ScratchDecl::new(BufferStorage::f32(), elems)],
    )
}
