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
use poot_graph_ir::{AttentionAttrs, KeyLayout, KvBound, MaskOperand, SelectedAttentionAttrs};
use poot_kernelgen::{
    AttentionCombineSpec, AttentionSchedule, AttentionSpec, KeyBound, KeyLayout as KernelKeyLayout,
    MaskKind, ScoreTransform,
};
use poot_target::{Backend, BufferStorage, DeviceCaps};

/// Plan one attention equation and record the kernel it takes.
///
/// `q[B,Hq,Lq,Dk]`, `k[B,Hkv,Lk,Dk]` (or its split parts), `v[B,Hkv,Lk,Dv]` (or the `[S,Hkv,D]` pools a paged key
/// reads through its `[B,Lk]` slot map), `mask[B,Hm,Lq,Lk]`: `Lq` and `Lk` are
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
    if let OpKind::SelectedAttention(attrs) = &eqn.op {
        return plan_selected(g, eqn, attrs, backend, caps, limits);
    }
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
    // A dense key carries the extents itself; a paged one is `[S,Hkv,D]` pools and the `[B,Lk]` slot map.
    let (lk, dv) = match key {
        KeyLayout::Paged => (shape(4)[1], v[2]),
        KeyLayout::Single | KeyLayout::Split { .. } => (k[2], v[3]),
    };
    let mask_kind = if flash_mask_per_head(&mask, hq, "attention")? {
        MaskKind::PerHead
    } else {
        MaskKind::Broadcast
    };
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
            KeyLayout::Paged => KernelKeyLayout::Paged,
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

/// Plan a block-selected attention: the streaming kernel with its mask derived from the selection operands.
/// Each row visits `(K + 1) * ratio` candidate positions, so there is no split-KV schedule to choose: the
/// lanes are the widest the device's workgroup memory takes, as for a dense row.
fn plan_selected(
    g: GraphTables<'_>,
    eqn: &Eqn,
    attrs: &SelectedAttentionAttrs,
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let operand_types: Vec<_> = value_ids(eqn)
        .iter()
        .map(|&id| g.aval(id).clone())
        .collect();
    let layout = attrs
        .layout(&operand_types)
        .map_err(|e| PlanError::BadShape(format!("selected attention: {e}")))?;
    let lanes = attention_lanes(layout.dv, caps).ok_or_else(|| {
        site.refuse(Capability::AttentionLds {
            dv: layout.dv,
            lds_bytes: caps.lds_bytes,
        })
    })?;
    let score = match attrs.softcap {
        None => ScoreTransform::Scale {
            scale_bits: attrs.scale.to_bits(),
        },
        Some(cap) => ScoreTransform::ScaleSoftcap {
            scale_bits: attrs.scale.to_bits(),
            cap_bits: cap.to_bits(),
        },
    };
    let spec = AttentionSpec {
        batch: layout.batch,
        hq: layout.hq,
        n_rep: attrs.n_rep,
        lq: layout.lq,
        lk: layout.lk,
        dk: layout.dk,
        dv: layout.dv,
        key: KernelKeyLayout::Single,
        score,
        mask: MaskKind::Blocks {
            ratio: attrs.ratio,
            k: attrs.k,
        },
        bound: KeyBound::Static,
        schedule: AttentionSchedule::Streaming { lanes },
    };
    // A decode row considers `(K + 1) * ratio` positions however deep the cache is, but a few heads still leave a
    // many-CU device idle: the same split-KV rule divides the candidates.
    match attention_splits(&spec, caps) {
        None => Planned::generated(site, KernelRequest::Attention(spec), backend, caps),
        Some(splits) => split_kv(
            site,
            spec,
            SplitKv { lanes, splits },
            operand_types.len(),
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
